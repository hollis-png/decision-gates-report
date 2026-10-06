"""審稿意見兩項（Jev）：
  1. 分開送出：每個操作各自一個請求（原實驗把基準與同狀態的操作放在同一請求），檢查批次是否影響結果。
  2. 附加中性句子：狀態後附加 1/4/8 句（同 von_len.py），量門檻 0.9 翻轉與決策翻轉。
每份意圖資料取前 100 題（與 iia_v3 同順序、同種子），操作為 base、permute、add_1、add_5 與 len1/len4/len8。
金鑰只從環境變數 TYPESAFE_API_KEY 讀取；累計花費超過 --budget 美元即停止。
"""
import argparse
import asyncio
import json
import random
from pathlib import Path

import iia_v3
from jev_eval import Client, question, to_scores
from von_len import NEUTRAL

N = 100
KEEP = ("permute", "add_1", "add_5")


async def run(task, client):
    qs = iia_v3.intent_questions(task)[:N]

    async def one(qi, q):
        rng = random.Random(qi)  # 與 iia_v3／jev_eval 相同的種子 → 相同的操作
        conds = iia_v3.conditions(task, q, [], {}, rng)
        jobs = {"base": (q["state"], q["ins"], q["crit"])}
        for name, state, ins, crit, rel in conds:
            if name in KEEP:
                jobs[name] = (state, ins, crit)
        for n in (1, 4, 8):
            jobs[f"len{n}"] = (q["state"] + " " + " ".join(NEUTRAL[:n]), q["ins"], q["crit"])
        out = {}
        for name, (state, ins, crit) in jobs.items():  # 每個操作各自一個請求
            a = await client.ask(state, {"q": question(ins, crit)})
            if a:
                out[name] = a["q"]
        if "base" not in out:
            return None
        keys = [k for k, _ in q["crit"]]
        return {"qid": q["qid"], "anchor": q["anchor"], "keys": keys,
                "base": {"scores": to_scores(out["base"]), "temp": 1.0, "probs": out["base"]},
                "conds": [{"cond": k, "scores": to_scores(v), "temp": 1.0, "probs": v}
                          for k, v in out.items() if k != "base"]}

    recs = await asyncio.gather(*(one(i, q) for i, q in enumerate(qs)))
    return [r for r in recs if r]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out-dir", default="jev_separate_results")
    args = ap.parse_args()
    client = Client(args.budget, args.concurrency)
    Path(args.out_dir).mkdir(exist_ok=True)
    for task in ("banking", "banking_hard", "clinc", "clinc_hard"):
        recs = await run(task, client)
        out = Path(args.out_dir) / f"{task}__jev.json"
        out.write_text(json.dumps({"task": task, "records": recs}, ensure_ascii=False))
        print(f"{task}: records={len(recs)} calls={client.calls} errors={client.errors} "
              f"tokens={client.tokens} cost=${client.cost:.3f} versions={sorted(client.model_versions)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
