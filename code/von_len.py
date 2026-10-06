"""A1 補：狀態文字長度掃描。在狀態後附加 0/1/2/4/8 句不同的中性句子，量溫度與門檻 0.9 翻轉。
Von 的溫度公式含 log10(狀態 token 數)；Laya 的溫度與長度無關，作為對照。
基準清單：iia_v3 的 banking 與 clinc_hard（5 個選項）。存下原始分數與溫度。
"""
import argparse
import json
from pathlib import Path

import torch

from iia_v3 import intent_questions, make_backend

NEUTRAL = ["This message was received through the standard support channel.",
           "No attachments were included with this message.",
           "The customer did not specify a preferred contact time.",
           "This is the first message in the conversation.",
           "The message was written in English.",
           "No previous tickets are linked to this request.",
           "The customer's account status is not shown here.",
           "There is no additional context available for this request."]
LENGTHS = [0, 1, 2, 4, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", default="von_len_results")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.2)
    backend = make_backend(args.backend, torch.device("cuda"))
    out = {}
    for task in ("banking", "clinc_hard"):
        recs = []
        for q in intent_questions(task)[:300]:
            rows = {}
            for n in LENGTHS:
                state = q["state"] + ("" if n == 0 else " " + " ".join(NEUTRAL[:n]))
                r = backend.score(state, q["ins"], q["crit"])
                if r is None:
                    break
                rows[n] = {"scores": r[0], "temp": r[1]}
            if len(rows) == len(LENGTHS):
                recs.append({"qid": q["qid"], "anchor": q["anchor"], "keys": [k for k, _ in q["crit"]], "by_len": rows})
        out[task] = recs
        print(f"{args.backend}/{task}: {len(recs)} records", flush=True)
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / f"{args.backend}.json").write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
