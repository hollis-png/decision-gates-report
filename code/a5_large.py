"""A5：大選項數的劑量反應（審稿意見：K 只到 12）。同時修正 A2 的設計問題：A2 每個 K 的清單各自重新打亂，
5→12 的比較混入了「順序改變」。這裡改成巢狀且保留順序：

  每句話：正確答案先放在前 5 個位置中的一個隨機位置（固定），其餘依干擾鏈順序排；
  K 個選項的清單 = 這個固定序列的前 K 個；加入的新選項永遠接在最後（新路由加在路由表末端）。
  所以 K < K' 時，K 的清單是 K' 清單的前綴，原有選項的相對順序與位置都不變。

K 格點：Banking77 到 77（全部意圖），CLINC150 到 150。每個 K 存基準分數，以及加入第 K+1 個選項後的分數（K 為最大值時略過）。
LLM 用數字標籤（1..K），分數是受限解碼下標籤數字串的對數機率（qwen3_4b_num），見 NumLLM。
輸出格式同 a2_dose（records / conds）。
"""
import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset

from iia_v3 import INTENT_TASKS, LLMS, LLMBackend, humanize, label_embeddings, make_backend

KS = {"banking": [5, 6, 8, 12, 24, 48, 77], "clinc": [5, 6, 8, 12, 24, 48, 77, 100, 150]}


def utterances(task, n):
    hard = task.endswith("_hard")
    base = task.removesuffix("_hard")
    name, config, text_col, label_col, ins = INTENT_TASKS[base]
    ds = load_dataset(name, config, split="test")
    names = ds.features[label_col].names
    valid = [i for i, nm in enumerate(names) if nm != "oos"]
    idx = [i for i in range(len(ds)) if ds[i][label_col] in valid]
    random.Random(0).shuffle(idx)  # 與 A2 相同的句子順序
    emb = label_embeddings(names) if hard else None
    for i in idx[:n]:
        gold = ds[i][label_col]
        pool = [v for v in valid if v != gold]
        if hard:
            sims = (emb[pool] @ emb[gold]).tolist()
            order = [v for _, v in sorted(zip(sims, pool), reverse=True)]
        else:
            order = pool[:]
            random.Random(i).shuffle(order)  # 與 A2 相同的干擾鏈
        seq = [names[v] for v in order]
        seq.insert(random.Random(f"{task}:{i}:goldpos").randrange(5), names[gold])
        yield {"qid": f"{task}:{i}", "state": ds[i][text_col], "ins": ins, "gold": names[gold], "seq": seq}


class NumLLM(LLMBackend):
    """集合式，選項用數字 1..K 標示；分數 = 受限解碼下該數字串的對數機率。
    數字在 Qwen 的分詞器裡逐位切開。第 j 位之後若還有更長的標籤共用此前綴，才加上「到此結束」的機率，
    其餘情況標籤在該位已唯一確定（前綴碼），不需要結束機率。"""

    def __init__(self, model_id):
        super().__init__(model_id, pointwise=False)
        self.digit = [self.tok.encode(str(d), add_special_tokens=False) for d in range(10)]
        assert all(len(t) == 1 for t in self.digit), "digits must be single tokens"
        self.digit = [t[0] for t in self.digit]

    @torch.no_grad()
    def score(self, state, ins, crit):
        k = len(crit)
        labels = [str(i + 1) for i in range(k)]
        opts = "\n".join(f"{labels[i]}. {key}: {v}" for i, (key, v) in enumerate(crit))
        content = (f"Context:\n{state}\n\nQuestion: {ins}\n\nOptions:\n{opts}\n\n"
                   "Answer with the number of the correct option only.")
        text = self.tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        out = self.model(ids, use_cache=True)
        prefixes = sorted({lab[:j] for lab in labels for j in range(1, len(lab))})  # 需要「下一位」分布的前綴
        dist = {"": torch.log_softmax(out.logits[0, -1].float(), -1)}
        for plen in (1, 2):
            group = [p for p in prefixes if len(p) == plen]
            if not group:
                continue
            past = out.past_key_values
            seq = torch.tensor([[self.digit[int(c)] for c in p] for p in group], device="cuda")
            cache = _repeat_cache(past, len(group))
            att = torch.ones((len(group), ids.shape[1] + plen), dtype=torch.long, device="cuda")
            o = self.model(seq, past_key_values=cache, attention_mask=att, use_cache=False)
            for p, lg in zip(group, o.logits[:, -1].float()):
                dist[p] = torch.log_softmax(lg, -1)
        lab_set = set(labels)
        s = []
        for lab in labels:
            lp = 0.0
            for j, c in enumerate(lab):
                lp += dist[lab[:j]][self.digit[int(c)]].item()
            if any(o != lab and o.startswith(lab) for o in lab_set):  # 還有更長的標籤共用此前綴：加上結束機率
                p_digit = torch.logsumexp(dist[lab][self.digit], 0).exp().item()
                lp += float(torch.log(torch.tensor(max(1 - p_digit, 1e-12))))
            s.append(lp)
        return dict(zip([key for key, _ in crit], s)), 1.0


def _repeat_cache(past, n):
    import copy
    c = copy.deepcopy(past)
    c.batch_repeat_interleave(n)
    return c


def backend_for(name):
    if name == "qwen3_4b_num":
        return NumLLM(LLMS["qwen3_4b"])
    if name.endswith("_long"):  # Laya 出貨的選項 token 預算（head_max_len=256）放不下約 24–48 個以上的選項；放寬到 4096 作為延伸對照
        b = make_backend(name.removesuffix("_long"), torch.device("cuda"))
        b.cfg.update({"max_len": 4096, "head_max_len": 3072})  # 選項放在 head，預算由 head_max_len 決定
        return b
    return make_backend(name, torch.device("cuda"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--task", required=True, choices=["banking", "clinc", "banking_hard", "clinc_hard"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out-dir", default="a5_results")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.2 if args.backend.startswith(("laya", "von")) else 0.6)
    backend = backend_for(args.backend)
    ks = KS[args.task.removesuffix("_hard")]
    kmax = ks[-1]
    records, skipped = [], {}
    import time
    t0 = time.time()
    for n, u in enumerate(utterances(args.task, args.n)):
        for k in ks:
            crit = [(lab, humanize(lab)) for lab in u["seq"][:k]]
            base = backend.score(u["state"], u["ins"], crit)
            new = None
            if k < kmax:
                lab = u["seq"][k]
                new = backend.score(u["state"], u["ins"], crit + [(lab, humanize(lab))])
            if base is None or (k < kmax and new is None):
                skipped[k] = skipped.get(k, 0) + 1
                continue
            rec = {"qid": f"{u['qid']}:k{k}", "K": k, "anchor": u["gold"], "keys": [c[0] for c in crit],
                   "base": {"scores": base[0], "temp": base[1]}, "conds": []}
            if new is not None:
                rec["conds"].append({"cond": f"k{k}", "scores": new[0], "temp": new[1]})
            records.append(rec)
        if n % 50 == 0:
            print(f"  {args.backend}/{args.task}: {n} ({time.time() - t0:.0f}s)", flush=True)
    out = Path(args.out_dir) / f"{args.task}__{args.backend}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": args.backend, "task": args.task, "ks": ks, "skipped": skipped,
                               "records": records}, ensure_ascii=False))
    print(f"{args.backend}/{args.task}: records={len(records)} skipped={skipped} secs={time.time() - t0:.0f} -> {out}")


if __name__ == "__main__":
    main()
