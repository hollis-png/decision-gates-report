"""iia_v3 離線分析：從原始分數計算所有指標，輸出 Markdown 表格。

  排名層次  rel 選項中第一名翻轉率；「答對→答錯」率（只在 rel 含正確答案的操作上算）
  統計      以題目為單位的拔靴法 95% 信賴區間；答對／答錯變化的 McNemar 精確檢定
  對照      無關操作的翻轉率減去三種對照（題目加句、狀態加句、亂碼選項），同題配對
  門檻      部署溫度 vs 固定溫度（反事實：新分數套用基準溫度）的門檻翻轉率，門檻 0.5/0.7/0.9
  其他      重複選項的機率和膨脹（溫度 1）；有翻轉題目的原始分數差距（檢查近平手）
用法：python analyze_iia_v3.py [結果目錄] > 報告.md
"""
import glob
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "iia_v3_results")
TASKS = ["typed", "banking", "banking_hard", "clinc", "clinc_hard"]
BACKENDS = ["laya_base", "laya_td", "laya_multi", "von", "von_setwise",
            "qwen3_4b_set", "qwen3_4b_point", "qwen25_7b_set", "qwen25_7b_point",
            "phi4mini_set", "phi4mini_point", "mistral7b_set", "mistral7b_point"]
CONTROLS = ["permute", "filler_ins", "filler_state"]
GATES = (0.5, 0.7, 0.9)
B = 1000


def group(cond):
    return "irrel" if cond.startswith("irrel_") else cond


def softmax(scores, temp):
    z = {k: v / max(temp, 1e-4) for k, v in scores.items()}
    m = max(z.values())
    e = {k: math.exp(v - m) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def top(scores, keys):
    return max(keys, key=lambda k: scores[k])


def row_metrics(rec, c):
    base, new, rel = rec["base"]["scores"], c["scores"], c["rel"]
    tb, tn = top(base, rel), top(new, rel)
    pb = softmax(base, rec["base"]["temp"])
    out = {"flip": float(tb != tn)}
    if rec["anchor"] in rel:
        base_ok, new_ok = top(base, rec["keys"]) == rec["anchor"], tn == rec["anchor"]
        out["correct_to_wrong"] = (base_ok, new_ok)
    for fixed in (False, True):
        pn = softmax(new, rec["base"]["temp"] if fixed else c["temp"])
        for g in GATES:
            out[f"gate{g}{'_fixed' if fixed else ''}"] = float((pb[tb] >= g) != (pn[tb] >= g))
    if c["cond"] == "duplicate":
        p1b, p1n = softmax(base, 1.0), softmax(new, 1.0)
        a = rec["anchor"]
        out["dup_inflation"] = p1n[a] + p1n[a + "_copy"] - p1b[a]
    srt = sorted((base[k] for k in rel), reverse=True)
    out["base_margin"] = srt[0] - srt[1]
    return out


def per_question(records):
    """題目 → 群組 → 指標平均（irrel_0..4 先在題內平均）。"""
    pq = {}
    for rec in records:
        acc = defaultdict(lambda: defaultdict(list))
        for c in rec["conds"]:
            for k, v in row_metrics(rec, c).items():
                if k != "correct_to_wrong":
                    acc[group(c["cond"])][k].append(v)
        pq[rec["qid"]] = {g: {k: st.mean(v) for k, v in d.items()} for g, d in acc.items()}
    return pq


def boot(vals, seed=0):
    if not vals:
        return None
    rng, n = random.Random(seed), len(vals)
    ms = sorted(st.mean(vals[rng.randrange(n)] for _ in range(n)) for _ in range(B))
    return st.mean(vals), ms[int(0.025 * B)], ms[int(0.975 * B)]


def pct(t):
    return "—" if t is None else f"{100*t[0]:.1f} [{100*t[1]:.1f}, {100*t[2]:.1f}]"


def mcnemar(records, groups):
    """答對→答錯 (b) 與 答錯→答對 (c) 的次數與精確檢定 p 值（每題每操作一筆）。"""
    b = c = 0
    for rec in records:
        for cd in rec["conds"]:
            if group(cd["cond"]) in groups:
                m = row_metrics(rec, cd).get("correct_to_wrong")
                if m:
                    b += m[0] and not m[1]
                    c += (not m[0]) and m[1]
    p = binomtest(b, b + c, 0.5).pvalue if b + c else float("nan")
    return b, c, p


def main():
    files = {(t, b): ROOT / f"{t}__{b}.json" for t in TASKS for b in BACKENDS}
    data = {k: json.loads(f.read_text())["records"] for k, f in files.items() if f.exists()}
    print(f"# IIA v3 結果（已完成 {len(data)} / {len(files)} 組）\n")
    for t in TASKS:
        rows = [(b, data[(t, b)]) for b in BACKENDS if (t, b) in data]
        if not rows:
            continue
        test_groups = ["gibberish", "remove", "irrel", "sibling", "duplicate"] if t == "typed" \
            else ["gibberish", "remove", "add_1", "add_3", "add_5", "duplicate"]
        main_group = "irrel" if t == "typed" else "add_1"
        print(f"\n## {t}\n")
        print("### 準確率與排名翻轉率（%，95% 信賴區間）\n")
        print("| 模型 | 準確率 | " + " | ".join(CONTROLS + test_groups) + " |")
        print("|---|---|" + "---|" * (len(CONTROLS) + len(test_groups)))
        for b, recs in rows:
            pq = per_question(recs)
            acc = st.mean(float(top(r["base"]["scores"], r["keys"]) == r["anchor"]) for r in recs)
            cells = [pct(boot([v[g]["flip"] for v in pq.values() if g in v])) for g in CONTROLS + test_groups]
            print(f"| {b} | {100*acc:.1f} | " + " | ".join(cells) + " |")

        print(f"\n### {main_group}：超出各對照的翻轉率（同題配對，%）與答對→答錯的 McNemar 檢定\n")
        print("| 模型 | − 換順序 | − 題目加句 | − 狀態加句 | − 亂碼選項 | 答對→答錯 | 答錯→答對 | McNemar p |")
        print("|---|---|---|---|---|---|---|---|")
        for b, recs in rows:
            pq = per_question(recs)
            cells = []
            for ctl in CONTROLS + ["gibberish"]:
                diff = [v[main_group]["flip"] - v[ctl]["flip"] for v in pq.values() if main_group in v and ctl in v]
                cells.append(pct(boot(diff)))
            bb, cc, p = mcnemar(recs, {main_group})
            print(f"| {b} | " + " | ".join(cells) + f" | {bb} | {cc} | {p:.3g} |")

        print(f"\n### {main_group}：信心門檻翻轉率（%），部署溫度 / 固定溫度\n")
        print("| 模型 | " + " | ".join(f"門檻 {g}" for g in GATES) + " | 換順序（門檻 0.7） |")
        print("|---|" + "---|" * (len(GATES) + 1))
        for b, recs in rows:
            pq = per_question(recs)
            cells = []
            for g in GATES:
                dep = boot([v[main_group][f"gate{g}"] for v in pq.values() if main_group in v])
                fix = boot([v[main_group][f"gate{g}_fixed"] for v in pq.values() if main_group in v])
                cells.append(f"{100*dep[0]:.1f} / {100*fix[0]:.1f}")
            perm = boot([v["permute"]["gate0.7"] for v in pq.values() if "permute" in v])
            print(f"| {b} | " + " | ".join(cells) + f" | {100*perm[0]:.1f} |")

        print("\n### 重複選項的機率和膨脹（溫度 1，百分點）與有翻轉題目的原始分數差距\n")
        print(f"| 模型 | 機率和膨脹 | {main_group} 有翻轉題的第一二名差距中位數 | 全部題目的差距中位數 |")
        print("|---|---|---|---|")
        for b, recs in rows:
            dup, flip_m, all_m = [], [], []
            for rec in recs:
                for c in rec["conds"]:
                    m = row_metrics(rec, c)
                    if "dup_inflation" in m:
                        dup.append(m["dup_inflation"])
                    if group(c["cond"]) == main_group:
                        all_m.append(m["base_margin"])
                        if m["flip"]:
                            flip_m.append(m["base_margin"])
            fm = f"{st.median(flip_m):.4f}（{len(flip_m)} 筆）" if flip_m else "—"
            print(f"| {b} | {100*st.mean(dup):+.1f} | {fm} | {st.median(all_m):.4f} |")


if __name__ == "__main__":
    main()
