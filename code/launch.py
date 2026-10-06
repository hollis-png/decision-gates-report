"""審稿意見：新路由上線時，屬於新路由的請求開始能被正確處理（目前的實驗中加入的選項永遠不是正確答案）。

每句話（同 A2 的 300 句與干擾鏈）各算兩個清單：
  before  上線前：5 個干擾選項，沒有正確答案（理想行為是轉人工）
  after   上線後：同 5 個干擾選項＋正確答案（共 6 個），正確答案的位置隨機
離線分析時與 A2 的 K=5→6（正確答案一直都在、新增的是干擾）混合，模擬「新路由分到一部分流量」。
輸出格式同 a2_dose（records / conds），條件名稱為 launch。
"""
import argparse
import json
import random
from pathlib import Path

import torch

from a2_dose import utterances
from iia_v3 import humanize, make_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--task", required=True, choices=["banking", "clinc", "banking_hard", "clinc_hard"])
    ap.add_argument("--out-dir", default="launch_results")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.2 if args.backend.startswith(("laya", "von")) else 0.4)
    backend = make_backend(args.backend, torch.device("cuda"))
    records, skipped = [], 0
    for n, u in enumerate(utterances(args.task)):
        before = u["chain"][:5]
        random.Random(f"{u['qid']}:launch").shuffle(before)
        after = before[:]
        after.insert(random.Random(f"{u['qid']}:pos").randrange(6), u["gold"])
        b = backend.score(u["state"], u["ins"], [(lab, humanize(lab)) for lab in before])
        a = backend.score(u["state"], u["ins"], [(lab, humanize(lab)) for lab in after])
        if b is None or a is None:
            skipped += 1
            continue
        records.append({"qid": f"{u['qid']}:launch", "anchor": u["gold"], "keys": before,
                        "base": {"scores": b[0], "temp": b[1]},
                        "conds": [{"cond": "launch", "rel": before, "scores": a[0], "temp": a[1]}]})
        if n % 50 == 0:
            print(f"  {args.backend}/{args.task}: {n}", flush=True)
    out = Path(args.out_dir) / f"{args.task}__{args.backend}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": args.backend, "task": args.task, "skipped": skipped,
                               "records": records}, ensure_ascii=False))
    print(f"{args.backend}/{args.task}: records={len(records)} skipped={skipped} -> {out}")


if __name__ == "__main__":
    main()
