"""無關選項獨立性實驗第三版：多資料集、多模型，存下所有原始分數供離線分析。

資料集（--task）：
  typed     typed-decisions 測試集，選項數 >= 3 的選擇題（同 v2，含語意相近選項）
  banking   Banking77 測試集抽 500 句；每句從 77 個意圖抽 5 個（含正確答案）當基準清單（隨機干擾）
  clinc     CLINC150 測試集抽 500 句（排除超出範圍類）；同上
  *_hard    同上，但干擾選項與加入的新意圖改挑「名稱和正確答案語意最近」的意圖（MiniLM 向量，與受測模型無關），
            模擬實務上相近類別並存的路由表

操作：
  permute        換順序
  filler_ins     題目說明加中性句（v2 的長度對照）
  filler_state   狀態文字加中性句
  gibberish      加一個亂碼選項（語意最少的無關選項）
  remove         刪一個非正確答案的選項
  irrel_*        typed：加 5 種別題型的選項之一
  add_1/3/5      banking／clinc：加 1、3、5 個同資料集的其他意圖（路由擴充的劑量反應）
  sibling        typed：加與正確答案語意相近的新選項
  duplicate      複製正確答案選項

輸出：每道題的基準分數與每個操作的分數、溫度、選項鍵，指標全部離線計算（analyze_iia_v3.py）。
"""
import argparse
import json
import math
import os
import random
import string
from pathlib import Path

import torch
from datasets import load_dataset

from iia_v2 import LayaBackend, VonBackend, irrelevant_pool, load_questions

FILLER_INS = " Please read the whole text carefully before you decide."
FILLER_STATE = " (No additional information is provided.)"
N_PER_INTENT_TASK = 500
BASE_K = 5
INTENT_TASKS = {
    "banking": ("legacy-datasets/banking77", None, "text", "label",
                "Which banking request does this customer message express?"),
    "clinc": ("clinc/clinc_oos", "plus", "text", "intent",
              "What does the user want the assistant to do?"),
}


# ---------------- 題目 ----------------

def humanize(label):
    return label.replace("_", " ").strip()


EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def label_embeddings(names):
    """意圖名稱的語意向量（MiniLM 平均池化後正規化）。與所有受測模型無關，只用來挑相近的干擾選項。"""
    from transformers import AutoModel, AutoTokenizer
    tok, enc = AutoTokenizer.from_pretrained(EMBEDDER), AutoModel.from_pretrained(EMBEDDER).eval()
    batch = tok([humanize(n) for n in names], padding=True, return_tensors="pt")
    with torch.no_grad():
        h = enc(**batch).last_hidden_state
    m = batch["attention_mask"].unsqueeze(-1).float()
    return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)


def intent_questions(task):
    hard = task.endswith("_hard")
    name, config, text_col, label_col, ins = INTENT_TASKS[task.removesuffix("_hard")]
    ds = load_dataset(name, config, split="test")
    names = ds.features[label_col].names
    valid = [i for i, n in enumerate(names) if n != "oos"]
    idx = [i for i in range(len(ds)) if ds[i][label_col] in valid]
    random.Random(0).shuffle(idx)
    emb = label_embeddings(names) if hard else None
    qs = []
    for i in idx[:N_PER_INTENT_TASK]:
        gold = ds[i][label_col]
        rng = random.Random(i)
        pool = [v for v in valid if v != gold]
        if hard:  # 和正確答案語意最近的 9 個意圖：前 4 個當干擾選項，後 5 個當「新增路由」
            sims = (emb[pool] @ emb[gold]).tolist()
            others = [v for _, v in sorted(zip(sims, pool), reverse=True)][:BASE_K - 1 + 5]
        else:
            others = rng.sample(pool, BASE_K - 1 + 5)
        base = [gold] + others[:BASE_K - 1]
        rng.shuffle(base)
        qs.append({"qid": f"{task}:{i}", "state": ds[i][text_col], "ins": ins,
                   "crit": [(names[v], humanize(names[v])) for v in base],
                   "extra": [(names[v], humanize(names[v])) for v in others[BASE_K - 1:]],
                   "anchor": names[gold]})
    return qs


def gibberish(rng):
    word = lambda: "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 7)))
    return "zz_" + word(), " ".join(word() for _ in range(3))


def with_state_filler(state):
    if isinstance(state, dict):
        return {**state, "note": FILLER_STATE.strip()}
    return state + FILLER_STATE


def conditions(task, q, pool, siblings, rng):
    crit, keys = q["crit"], [k for k, _ in q["crit"]]
    others = [k for k in keys if k != q["anchor"]]
    order = crit[:]
    rng.shuffle(order)
    conds = [("permute", q["state"], q["ins"], order, keys),
             ("filler_ins", q["state"], q["ins"] + FILLER_INS, crit, keys),
             ("filler_state", with_state_filler(q["state"]), q["ins"], crit, keys),
             ("gibberish", q["state"], q["ins"], crit + [gibberish(rng)], keys)]
    drop = rng.choice(others)
    kept = [c for c in crit if c[0] != drop]
    conds.append(("remove", q["state"], q["ins"], kept, [k for k, _ in kept]))
    if task == "typed":
        descs = {v for _, v in crit}
        cands = [(k, v) for k, v, fam in pool if fam != q["ins"] and k not in keys and v not in descs]
        for j, opt in enumerate(rng.sample(cands, 5)):
            conds.append((f"irrel_{j}", q["state"], q["ins"], crit + [opt], keys))
        sib = siblings.get(q["qid"])
        if sib and sib["key"] not in keys:
            conds.append(("sibling", q["state"], q["ins"], crit + [(sib["key"], sib["description"])], others))
    else:
        for m in (1, 3, 5):
            conds.append((f"add_{m}", q["state"], q["ins"], crit + q["extra"][:m], keys))
    anchor_desc = dict(crit)[q["anchor"]]
    conds.append(("duplicate", q["state"], q["ins"], crit + [(q["anchor"] + "_copy", anchor_desc)], others))
    return conds


# ---------------- 模型 ----------------

class VonStr(VonBackend):
    """Von 的狀態可以是字串（意圖資料集）或字典（typed）。"""

    def score(self, state, ins, crit):
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        return self._score_text(text, ins, crit)

    def _score_text(self, state_text, ins, crit):
        descs = [v.strip() if v else k for k, v in crit]
        fitted = self.b._fit_state(self.m, state_text, ins, descs)
        if fitted != state_text:
            return None
        inputs = self.m.tokenizer(self.m.pack_sequence(fitted, ins, descs), return_tensors="pt").to(self.b.device)
        pos = (inputs["input_ids"][0] == self.m.mask_token_id).nonzero(as_tuple=True)[0].tolist()
        with torch.no_grad():
            logits = self.m(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                            mask_positions=[pos], independent_options=self.b._independent_options)[0].float()
            temp = self.b._effective_temperature(logits, fitted, len(crit), self.m.tokenizer, None)
        return dict(zip([k for k, _ in crit], logits.tolist())), float(temp)


class LLMBackend:
    """任意指令模型；集合式（讀選項字母）或逐一式（每個選項問 Yes/No）。回傳原始分數，溫度 1。"""

    def __init__(self, model_id, pointwise):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
        self.pointwise = pointwise
        self.cache = {}
        self.letters = [chr(65 + i) for i in range(16)]
        self.letter_ids = [self._variants(c) for c in self.letters]
        self.yes, self.no = self._variants("Yes"), self._variants("No")

    def _variants(self, word):
        """同一個字可能被切成「無前導空白」或「有前導空白」兩種 token，兩者都收，之後取 logsumexp。"""
        return sorted({self.tok.encode(w, add_special_tokens=False)[-1] for w in (word, " " + word)})

    @staticmethod
    def _lse(z, ids):
        return torch.logsumexp(z[ids], 0).item()

    @torch.no_grad()
    def _last_logits(self, content):
        text = self.tok.apply_chat_template([{"role": "user", "content": content}],
                                            add_generation_prompt=True, tokenize=False)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        return self.model(ids).logits[0, -1].float()

    def score(self, state, ins, crit):
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        if self.pointwise:
            s = []
            for k, v in crit:
                key = (state_text, ins, k, v)
                if key not in self.cache:
                    z = self._last_logits(f"Context:\n{state_text}\n\nQuestion: {ins}\n\nCandidate answer: {k}: {v}\n\n"
                                          "Is this candidate the correct answer to the question? Answer Yes or No.")
                    self.cache[key] = self._lse(z, self.yes) - self._lse(z, self.no)
                s.append(self.cache[key])
        else:
            opts = "\n".join(f"{self.letters[i]}. {k}: {v}" for i, (k, v) in enumerate(crit))
            z = self._last_logits(f"Context:\n{state_text}\n\nQuestion: {ins}\n\nOptions:\n{opts}\n\n"
                                  "Answer with the letter of the correct option only.")
            s = [self._lse(z, self.letter_ids[i]) for i in range(len(crit))]
        return dict(zip([k for k, _ in crit], s)), 1.0


LLMS = {"qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507", "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
        "phi4mini": "microsoft/Phi-4-mini-instruct", "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3"}


def make_backend(name, device):
    if name.startswith("laya_"):
        return LayaBackend({"laya_base": "", "laya_td": "typed-decisions", "laya_multi": "multilingual"}[name], device)
    if name == "von":
        return VonStr(True, device)
    if name == "von_setwise":
        return VonStr(False, device)
    family, mode = name.rsplit("_", 1)
    return LLMBackend(LLMS[family], pointwise=(mode == "point"))


# ---------------- 主程式 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--task", required=True, choices=["typed", "banking", "clinc", "banking_hard", "clinc_hard"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="iia_v3_results")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.75)
    device = torch.device("cuda")

    qs = load_questions() if args.task == "typed" else intent_questions(args.task)
    if args.limit:
        qs = qs[:args.limit]
    pool = irrelevant_pool() if args.task == "typed" else []
    siblings = json.loads(Path("siblings.json").read_text()) if args.task == "typed" else {}
    backend = make_backend(args.backend, device)

    records, skipped = [], 0
    for qi, q in enumerate(qs):
        rng = random.Random(qi)  # 每道題的操作在所有模型間完全相同
        base = backend.score(q["state"], q["ins"], q["crit"])
        if base is None:
            skipped += 1
            continue
        rec = {"qid": q["qid"], "anchor": q["anchor"], "keys": [k for k, _ in q["crit"]],
               "base": {"scores": base[0], "temp": base[1]}, "conds": []}
        for name, state, ins, crit, rel in conditions(args.task, q, pool, siblings, rng):
            new = backend.score(state, ins, crit)
            if new is not None:
                rec["conds"].append({"cond": name, "rel": rel, "scores": new[0], "temp": new[1]})
        records.append(rec)
        if qi % 100 == 0:
            print(f"  {args.backend}/{args.task}: {qi}/{len(qs)}", flush=True)

    out = Path(args.out_dir) / f"{args.task}__{args.backend}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": args.backend, "task": args.task, "skipped": skipped,
                               "records": records}, ensure_ascii=False))
    acc = sum(max(r["keys"], key=r["base"]["scores"].get) == r["anchor"] for r in records) / max(1, len(records))
    print(f"{args.backend}/{args.task}: n={len(records)} skipped={skipped} acc={acc:.4f} -> {out}")


if __name__ == "__main__":
    main()
