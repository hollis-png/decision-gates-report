"""B2：依基準時 rel 選項第一二名的分數差距分層，比較各操作的翻轉率，排除「差距小本來就容易翻」的混淆。

每個模型內依差距分三等分（各模型分數尺度不同，所以在模型內分）；只看 rel 第一名是否翻轉。
重點比較：duplicate、sibling 與 gibberish（同為加入一個選項，但無語意）在同一層內的差異。
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

V3 = Path("iia_v3_results")
TASKS = ["typed", "banking", "banking_hard", "clinc", "clinc_hard"]
MODELS = ["laya_base", "laya_td", "laya_multi", "von", "von_setwise", "qwen3_4b_set", "qwen3_4b_point",
          "qwen25_7b_set", "qwen25_7b_point", "phi4mini_set", "phi4mini_point", "mistral7b_set", "mistral7b_point"]
CONDS = ["permute", "filler_state", "gibberish", "add_1", "irrel", "sibling", "duplicate"]


def group(c):
    return "irrel" if c.startswith("irrel_") else c


def main():
    print("# B2：依分數差距三等分的翻轉率（%；各格為 低／中／高 差距；所有資料集合併）\n")
    print("| 模型 | " + " | ".join(CONDS) + " |")
    print("|---|" + "---|" * len(CONDS))
    for m in MODELS:
        rows = []
        for t in TASKS:
            f = V3 / f"{t}__{m}.json"
            if not f.exists():
                continue
            for rec in json.loads(f.read_text())["records"]:
                b = rec["base"]["scores"]
                for c in rec["conds"]:
                    g = group(c["cond"])
                    if g not in CONDS:
                        continue
                    rel = c["rel"]
                    srt = sorted(rel, key=lambda k: -b[k])
                    margin = b[srt[0]] - b[srt[1]]
                    flip = srt[0] != max(rel, key=lambda k: c["scores"][k])
                    rows.append((g, margin, flip))
        if not rows:
            continue
        ms = sorted(r[1] for r in rows)
        cut1, cut2 = ms[len(ms) // 3], ms[2 * len(ms) // 3]
        agg = defaultdict(lambda: defaultdict(list))
        for g, margin, flip in rows:
            tier = 0 if margin < cut1 else 1 if margin < cut2 else 2
            agg[g][tier].append(flip)
        cells = []
        for g in CONDS:
            if g not in agg:
                cells.append("—")
                continue
            cells.append("／".join(f"{100*st.mean(agg[g][i]):.0f}" if agg[g][i] else "—" for i in range(3)))
        print(f"| {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
