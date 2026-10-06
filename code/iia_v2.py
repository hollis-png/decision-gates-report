"""無關選項獨立性（IIA）跨架構實驗。

模型（--backend）：
  laya_base / laya_td / laya_multi   集合式編碼器（Laya 起點、typed-decisions 微調、多語版）
  von                                 Von 發布版（選項互相獨立的注意力遮罩）
  von_setwise                         Von 關掉獨立遮罩（偏離訓練設定，僅供診斷）
  qwen_set                            Qwen3-4B，所有選項列在提示裡，讀字母機率（集合式）
  qwen_point                          Qwen3-4B，每個選項分開問是否正確（逐一式）

操作（每道題都做）：
  permute    換順序                       filler     題目說明加中性文字（長度對照）
  irrel_0-4  加 5 種不同的無關選項之一     sibling    加與正確答案語意相近的新選項（gen_siblings.py）
  duplicate  複製正確答案選項

指標（相對 base）：
  排名層次：relevant 選項中第一名是否翻轉、對數比例偏移（用原始分數）
            sibling／duplicate 的 relevant 不含被模仿的選項（它被分走機率是合理的）
  機率層次：用部署時的溫度換算機率，第一名的信心是否跨過門檻 0.5／0.7／0.9
  duplicate 另記：兩個複製品的機率和 − 原本的機率（溫度 1）
逐題結果全部存檔，供配對拔靴法使用。
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset

FILLER = " Please read the whole text carefully before you decide."
GATES = (0.5, 0.7, 0.9)
N_IRREL = 5


# ---------------- 題目與操作 ----------------

def load_questions():
    qs = []
    for i, row in enumerate(load_dataset("LocalLLaMA/typed-decisions", "all", split="test")):
        state, questions, gold = (json.loads(row[k]) for k in ("state", "questions", "gold"))
        for qkey, q in questions.items():
            if q["type"] == "choice" and len(q["criteria"]) >= 3 and qkey in gold:
                anchor = str(gold[qkey]["label"])
                if anchor in q["criteria"]:
                    qs.append({"qid": f"{i}:{qkey}", "state": state, "ins": q["instructions"],
                               "crit": list(q["criteria"].items()), "anchor": anchor})
    return qs


def irrelevant_pool(n_per_family=4, seed=0):
    """無關選項池：從訓練集每種題型（以題目說明區分）各取幾個選項，並記下來源題型。
    抽樣時排除與該題同題型的選項，所以抽到的一定是別的領域的類別。"""
    by_family = {}
    for row in load_dataset("LocalLLaMA/typed-decisions", "all", split="train"):
        for q in json.loads(row["questions"]).values():
            if q["type"] == "choice":
                fam = by_family.setdefault(q["instructions"], {})
                fam.update({k: v for k, v in q["criteria"].items() if v})
    rng, pool = random.Random(seed), []
    for fam, opts in sorted(by_family.items()):
        items = sorted(opts.items())
        rng.shuffle(items)
        pool += [(k, v, fam) for k, v in items[:n_per_family]]
    return pool


def conditions(q, pool, siblings, rng):
    crit, keys = q["crit"], [k for k, _ in q["crit"]]
    descs = {v for _, v in crit}
    conds = [("base", q["ins"], crit, keys, None)]
    order = crit[:]
    rng.shuffle(order)
    conds.append(("permute", q["ins"], order, keys, None))
    conds.append(("filler", q["ins"] + FILLER, crit, keys, None))
    cands = [(k, v) for k, v, fam in pool if fam != q["ins"] and k not in keys and v not in descs]
    for j, opt in enumerate(rng.sample(cands, N_IRREL)):
        conds.append((f"irrel_{j}", q["ins"], crit + [opt], keys, None))
    others = [k for k in keys if k != q["anchor"]]
    sib = siblings.get(q["qid"])
    if sib and sib["key"] not in keys:
        conds.append(("sibling", q["ins"], crit + [(sib["key"], sib["description"])], others, None))
    anchor_desc = dict(crit)[q["anchor"]]
    conds.append(("duplicate", q["ins"], crit + [(q["anchor"] + "_copy", anchor_desc)], others,
                  q["anchor"]))
    return conds


# ---------------- 模型 ----------------

class LayaBackend:
    def __init__(self, sub, device):
        from iia_pilot import load
        from laya.common import clamp_temperature, temp_bucket, QTYPES
        self.model, self.tok, self.cfg = load("./laya_hf", sub, device)
        self.device = device
        t_by = self.cfg.get("temperature_by_options") or {}
        t_choice = self.cfg["temperature"][QTYPES["choice"]]
        self.temp = lambda k: clamp_temperature(t_by.get(temp_bucket(QTYPES["choice"], k), t_choice))

    def score(self, state, ins, crit):
        from iia_pilot import log_probs
        lp = log_probs(self.model, self.tok, self.cfg, state, ins, dict(crit), self.device)
        if lp is None:
            return None
        k = int(os.environ["LAYA_FIXED_K"]) if os.environ.get("LAYA_FIXED_K") else len(crit)
        return lp, self.temp(k)  # 對數機率與 logits 只差常數；LAYA_FIXED_K 用於固定溫度的反事實對照


class VonBackend:
    def __init__(self, independent, device):
        import os
        from huggingface_hub import snapshot_download
        os.environ["VON_CHAINS_DIR"] = "off"  # 關掉規則算式鏈，只測模型本身
        from von.backends.option_marker_backend import OptionMarkerBackend
        path = snapshot_download("wfzyx/von", allow_patterns=["*.json", "*.safetensors", "option_marker.pt"])
        self.b = OptionMarkerBackend(checkpoint_dir=path, device="cuda")
        self.m = self.b._get_model()
        self.b._independent_options = independent
        self.base_fit = None

    def score(self, state, ins, crit):
        state_text = json.dumps(state, ensure_ascii=False)
        descs = [v.strip() if v else k for k, v in crit]
        fitted = self.b._fit_state(self.m, state_text, ins, descs)
        if fitted != state_text:  # 狀態被裁切：與其他操作不可比，排除
            return None
        inputs = self.m.tokenizer(self.m.pack_sequence(fitted, ins, descs), return_tensors="pt").to(self.b.device)
        pos = (inputs["input_ids"][0] == self.m.mask_token_id).nonzero(as_tuple=True)[0].tolist()
        with torch.no_grad():
            logits = self.m(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                            mask_positions=[pos], independent_options=self.b._independent_options)[0].float()
            temp = self.b._effective_temperature(logits, fitted, len(crit), self.m.tokenizer, None)
        lp = torch.log_softmax(logits, -1).tolist()
        return dict(zip([k for k, _ in crit], lp)), float(temp)


class QwenBackend:
    MODEL = "Qwen/Qwen3-4B-Instruct-2507"

    def __init__(self, pointwise, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
        self.pointwise = pointwise
        self.cache = {}
        self.letters = [chr(65 + i) for i in range(16)]
        self.letter_ids = [self.tok.encode(c, add_special_tokens=False)[0] for c in self.letters]
        self.yes, self.no = (self.tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))

    @torch.no_grad()
    def _last_logits(self, content):
        text = self.tok.apply_chat_template([{"role": "user", "content": content}],
                                            add_generation_prompt=True, tokenize=False)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        return self.model(ids).logits[0, -1].float()

    def score(self, state, ins, crit):
        state_text = json.dumps(state, ensure_ascii=False)
        if self.pointwise:
            s = []
            for k, v in crit:
                key = (state_text, ins, k, v)
                if key not in self.cache:
                    z = self._last_logits(f"Context:\n{state_text}\n\nQuestion: {ins}\n\nCandidate answer: {k}: {v}\n\n"
                                          "Is this candidate the correct answer to the question? Answer Yes or No.")
                    self.cache[key] = (z[self.yes] - z[self.no]).item()
                s.append(self.cache[key])
            logits = torch.tensor(s)
        else:
            opts = "\n".join(f"{self.letters[i]}. {k}: {v}" for i, (k, v) in enumerate(crit))
            z = self._last_logits(f"Context:\n{state_text}\n\nQuestion: {ins}\n\nOptions:\n{opts}\n\n"
                                  "Answer with the letter of the correct option only.")
            logits = z[self.letter_ids[:len(crit)]]
        lp = torch.log_softmax(logits, -1).tolist()
        return dict(zip([k for k, _ in crit], lp)), 1.0


def make_backend(name, device):
    return {"laya_base": lambda: LayaBackend("", device),
            "laya_td": lambda: LayaBackend("typed-decisions", device),
            "laya_multi": lambda: LayaBackend("multilingual", device),
            "von": lambda: VonBackend(True, device),
            "von_setwise": lambda: VonBackend(False, device),
            "qwen_set": lambda: QwenBackend(False, device),
            "qwen_point": lambda: QwenBackend(True, device)}[name]()


# ---------------- 指標 ----------------

def centered(lp, keys):
    m = sum(lp[k] for k in keys) / len(keys)
    return {k: lp[k] - m for k in keys}


def deployed_probs(lp, temp):
    z = {k: v / max(temp, 1e-4) for k, v in lp.items()}
    mx = max(z.values())
    e = {k: math.exp(v - mx) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def compare(base, new, rel, dup_anchor):
    (blp, bt), (nlp, nt) = base, new
    cb, cn = centered(blp, rel), centered(nlp, rel)
    tb, tn = max(rel, key=cb.get), max(rel, key=cn.get)
    pb, pn = deployed_probs(blp, bt), deployed_probs(nlp, nt)
    r = {"flip": tb != tn, "shift": sum(abs(cb[k] - cn[k]) for k in rel) / len(rel),
         "temp_base": bt, "temp_new": nt,
         "gate_flip": {str(g): (pb[tb] >= g) != (pn[tb] >= g) for g in GATES},
         "conf_delta": pn[tb] - pb[tb]}
    if dup_anchor:
        p1b, p1n = math.exp(blp[dup_anchor]), math.exp(nlp[dup_anchor])
        r["dup_mass_inflation"] = p1n + math.exp(nlp[dup_anchor + "_copy"]) - p1b
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="iia_v2_results")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.6)
    device = torch.device("cuda")

    qs = load_questions()
    if args.limit:
        qs = qs[:args.limit]
    pool = irrelevant_pool()
    siblings = json.loads(Path("siblings.json").read_text()) if Path("siblings.json").exists() else {}
    backend = make_backend(args.backend, device)

    rows, acc, skipped = [], [], 0
    for qi, q in enumerate(qs):
        rng = random.Random(qi)  # 每道題的操作（換順序、抽哪些無關選項）在所有模型間完全相同
        conds = conditions(q, pool, siblings, rng)
        base = backend.score(q["state"], conds[0][1], conds[0][2])
        if base is None:
            skipped += 1
            continue
        keys = [k for k, _ in q["crit"]]
        acc.append(max(keys, key=base[0].get) == q["anchor"])
        for name, ins, crit, rel, dup in conds[1:]:
            new = backend.score(q["state"], ins, crit)
            if new is None:
                continue
            rows.append({"qid": q["qid"], "cond": name, **compare(base, new, rel, dup)})
        if qi % 100 == 0:
            print(f"  {args.backend}: {qi}/{len(qs)}", flush=True)

    tag = f"_fixedK{os.environ['LAYA_FIXED_K']}" if os.environ.get("LAYA_FIXED_K") else ""
    out = Path(args.out_dir) / f"{args.backend}{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": args.backend, "n_questions": len(acc), "skipped": skipped,
                               "accuracy": sum(acc) / max(1, len(acc)), "pool": pool, "rows": rows},
                              ensure_ascii=False))
    print(f"{args.backend}: n={len(acc)} skipped={skipped} acc={sum(acc)/max(1,len(acc)):.4f} -> {out}")


if __name__ == "__main__":
    main()
