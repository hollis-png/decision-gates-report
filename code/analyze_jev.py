"""Jev 黑箱分析。Jev 只回傳四捨五入到 0.01 的機率，所以：
  - 排名只看「系統輸出的第一名」（原有選項中機率最高者），不看 0.00 之間的同分排名
  - 比例只用機率 >= 0.05 的前兩名原有選項：r = log(p1/p2)
  - 所有指標都和兩種雜訊底線比較：retest_in（同請求重複）、retest_out（跨請求重送）
溫度推論：若架構使各選項分數獨立、只有溫度隨選項數改變，則加選項後 r_new ≈ c × r_base，
c = T_base / T_new 對所有題目一致；比較 c 的分布與重測時的分布。
"""
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

V3 = Path("iia_v3_results")
A2 = Path("a2_results")
TASKS = ["banking", "banking_hard", "clinc", "clinc_hard", "typed"]
GATES = (0.7, 0.9)
MIN_P = 0.05


def group(c):
    return "irrel" if c.startswith("irrel_") else c


def top(p, keys):
    return max(keys, key=lambda k: p.get(k, 0.0))


def boot(vals, n=1000, seed=0):
    if not vals:
        return None
    rng, m = random.Random(seed), len(vals)
    ms = sorted(st.mean(vals[rng.randrange(m)] for _ in range(m)) for _ in range(n))
    return st.mean(vals), ms[int(0.025 * n)], ms[int(0.975 * n)]


def pct(t):
    return "—" if t is None else f"{100*t[0]:.1f} [{100*t[1]:.1f}, {100*t[2]:.1f}]"


def log_ratio(p, keys):
    """原有選項中前兩名（都 >= MIN_P）的對數機率比；不符合條件回傳 None。"""
    two = sorted(keys, key=lambda k: -p.get(k, 0.0))[:2]
    if len(two) < 2 or p.get(two[1], 0.0) < MIN_P:
        return None, two
    return math.log(p[two[0]] / p[two[1]]), two


def row(rec, c):
    # 移除選項時只在剩下的選項裡比較（被刪掉的若原本是第一名，換人是合理的）
    keys = c["rel"] if c["cond"] == "remove" else rec["keys"]
    pb, pn = rec["base"]["probs"], c["probs"]
    tb = top(pb, keys)
    out = {"flip": float(tb != top(pn, keys)), "dp_top": pn.get(tb, 0.0) - pb[tb]}
    for g in GATES:
        out[f"gate{g}"] = float((pb[tb] >= g) != (pn.get(tb, 0.0) >= g))
    rb, two = log_ratio(pb, keys)
    if rb is not None and pn.get(two[1], 0.0) >= MIN_P and pn.get(two[0], 0.0) >= MIN_P:
        rn = math.log(pn[two[0]] / pn[two[1]])
        out["dlogratio"] = rn - rb
        if abs(rb) > 0.2:
            out["scale_c"] = rn / rb
    if c["cond"] == "duplicate":
        a = rec["anchor"]
        out["dup_sum"] = pn.get(a, 0) + pn.get(a + "_copy", 0) - pb.get(a, 0)
        s = pn.get(a, 0) + pn.get(a + "_copy", 0)
        if s >= 0.1:
            out["dup_copy_share"] = pn.get(a + "_copy", 0) / s
    return out


def v3_report():
    print("# Jev 黑箱分析\n\n## 一、第三版操作（每題一筆；95% 信賴區間）\n")
    for task in TASKS:
        f = V3 / f"{task}__jev.json"
        if not f.exists():
            continue
        recs = json.loads(f.read_text())["records"]
        acc = st.mean(float(top(r["base"]["probs"], r["keys"]) == r["anchor"]) for r in recs)
        agg = defaultdict(lambda: defaultdict(list))
        for rec in recs:
            for c in rec["conds"]:
                for k, v in row(rec, c).items():
                    agg[group(c["cond"])][k].append(v)
        order = ["retest_in", "retest_out", "permute", "filler_ins", "filler_state", "gibberish", "remove"] + \
                (["irrel", "sibling"] if task == "typed" else ["add_1", "add_3", "add_5"]) + ["duplicate"]
        print(f"\n### {task}（準確率 {100*acc:.1f}%）\n")
        print("| 操作 | 第一名翻轉 | 門檻 0.7 翻轉 | 門檻 0.9 翻轉 | 第一名機率變化（點） | 前兩名對數比變化 |絕對值| | 比例縮放 c 中位數（n） |")
        print("|---|---|---|---|---|---|---|")
        for g in order:
            if g not in agg:
                continue
            a = agg[g]
            dl = a.get("dlogratio", [])
            sc = a.get("scale_c", [])
            print(f"| {g} | {pct(boot(a['flip']))} | {pct(boot(a['gate0.7']))} | {pct(boot(a['gate0.9']))} | "
                  f"{100*st.mean(a['dp_top']):+.1f} | {st.mean(abs(x) for x in dl):.3f}（{len(dl)}） | "
                  + (f"{st.median(sc):.2f}（{len(sc)}）" if sc else "—") + " |")
        if "dup_sum" in agg["duplicate"]:
            d = agg["duplicate"]
            share = d.get("dup_copy_share", [])
            print(f"\n重複選項：兩者機率和 − 原本 = {100*st.mean(d['dup_sum']):+.1f} 點；"
                  f"複製品所佔比例中位數 {st.median(share):.2f}（理想平分為 0.50，n={len(share)}）")


def a2_report():
    print("\n## 二、劑量反應（A2）：基準 K 加 1 個選項\n")
    print("| K | 門檻 0.9 翻轉 | 門檻 0.7 翻轉 | 第一名機率變化（點） | 比例縮放 c 中位數（n） |")
    print("|---|---|---|---|---|")
    by_k = defaultdict(lambda: defaultdict(list))
    for task in ["banking", "banking_hard", "clinc", "clinc_hard"]:
        f = A2 / f"{task}__jev.json"
        if not f.exists():
            continue
        for rec in json.loads(f.read_text())["records"]:
            for k, v in row(rec, rec["conds"][0]).items():
                by_k[rec["K"]][k].append(v)
    for k in sorted(by_k):
        a = by_k[k]
        sc = a.get("scale_c", [])
        print(f"| {k} | {pct(boot(a['gate0.9']))} | {pct(boot(a['gate0.7']))} | {100*st.mean(a['dp_top']):+.1f} | "
              + (f"{st.median(sc):.2f}（{len(sc)}）" if sc else "—") + " |")


if __name__ == "__main__":
    v3_report()
    a2_report()
