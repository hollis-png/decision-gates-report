"""A2：選項數的劑量反應。基準選項數 K = 2..12，各加 1 個選項，存下原始分數供三因素拆解。

每句話先排好一串固定順序的干擾選項（隨機版：隨機順序；困難版：與正確答案語意相近度由高到低），
基準清單 = 正確答案 + 前 K−1 個干擾選項，新增的是第 K 個干擾選項。不同 K 之間是巢狀的。
輸出格式與 iia_v3 相同（records / conds），條件名稱為 k{K}，可直接用 analyze_a3.decompose 分析。
"""
import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset

from iia_v3 import INTENT_TASKS, humanize, label_embeddings, make_backend

K_RANGE = range(2, 13)
N_UTTER = 300


def utterances(task):
    hard = task.endswith("_hard")
    name, config, text_col, label_col, ins = INTENT_TASKS[task.removesuffix("_hard")]
    ds = load_dataset(name, config, split="test")
    names = ds.features[label_col].names
    valid = [i for i, n in enumerate(names) if n != "oos"]
    idx = [i for i in range(len(ds)) if ds[i][label_col] in valid]
    random.Random(0).shuffle(idx)
    emb = label_embeddings(names) if hard else None
    for i in idx[:N_UTTER]:
        gold = ds[i][label_col]
        pool = [v for v in valid if v != gold]
        if hard:
            sims = (emb[pool] @ emb[gold]).tolist()
            order = [v for _, v in sorted(zip(sims, pool), reverse=True)]
        else:
            order = pool[:]
            random.Random(i).shuffle(order)
        yield {"qid": f"{task}:{i}", "state": ds[i][text_col], "ins": ins, "gold": names[gold],
               "chain": [names[v] for v in order[:max(K_RANGE)]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--task", required=True, choices=["banking", "clinc", "banking_hard", "clinc_hard"])
    ap.add_argument("--out-dir", default="a2_results")
    args = ap.parse_args()
    # 編碼器 20% 就夠；大型語言模型（約 8 GB 以上）需要較多，否則載入時 OOM
    torch.cuda.set_per_process_memory_fraction(0.2 if args.backend.startswith(("laya", "von")) else 0.4)
    backend = make_backend(args.backend, torch.device("cuda"))

    records, skipped = [], 0
    for n, u in enumerate(utterances(args.task)):
        for k in K_RANGE:
            labels = [u["gold"]] + u["chain"][:k - 1]
            random.Random(f"{u['qid']}:{k}").shuffle(labels)  # 正確答案的位置隨機，避免位置偏差
            crit = [(lab, humanize(lab)) for lab in labels]
            added = (u["chain"][k - 1], humanize(u["chain"][k - 1]))
            base = backend.score(u["state"], u["ins"], crit)
            new = backend.score(u["state"], u["ins"], crit + [added])
            if base is None or new is None:
                skipped += 1
                continue
            records.append({"qid": f"{u['qid']}:k{k}", "K": k, "anchor": u["gold"], "keys": labels,
                            "base": {"scores": base[0], "temp": base[1]},
                            "conds": [{"cond": f"k{k}", "rel": labels, "scores": new[0], "temp": new[1]}]})
        if n % 50 == 0:
            print(f"  {args.backend}/{args.task}: {n}/{N_UTTER}", flush=True)

    out = Path(args.out_dir) / f"{args.task}__{args.backend}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": args.backend, "task": args.task, "skipped": skipped,
                               "records": records}, ensure_ascii=False))
    print(f"{args.backend}/{args.task}: records={len(records)} skipped={skipped} -> {out}")


if __name__ == "__main__":
    main()
