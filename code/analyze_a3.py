"""A3：把「決策信心的變化」拆成三個因素的 Shapley 貢獻。

決策 = 基準時原有選項中的第一名 t（系統實際輸出的答案）；信心 = t 的部署機率。
三個因素（每個都是「基準」或「操作後」兩種狀態）：
  S  原有選項的分數：用基準分數，或用操作後的分數（集合式架構造成的分數改變）
  D  新加入的選項：不放進正規化分母，或放進去（稀釋）
  T  溫度：基準溫度，或操作後的溫度（依選項數變動的校準）
8 種組合都算出 t 的信心，Shapley 值 = 各加入順序下邊際效應的平均；三項相加恰等於總變化。
門檻歸因：只改單一因素時是否已經跨過門檻。
只分析「有加入選項」的操作（換順序、加句只有 S 會變，作為參照一併列出）。
"""
import glob
import itertools
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "iia_v3_results")
TASKS = ["typed", "banking", "banking_hard", "clinc", "clinc_hard"]
BACKENDS = ["laya_base", "laya_td", "laya_multi", "von", "von_setwise",
            "qwen3_4b_set", "qwen3_4b_point", "qwen25_7b_set", "qwen25_7b_point",
            "phi4mini_set", "phi4mini_point", "mistral7b_set", "mistral7b_point"]
FACTORS = ("S", "D", "T")
GATES = (0.7, 0.9)


def group(cond):
    return "irrel" if cond.startswith("irrel_") else cond


def conf(scores, keys, temp, t):
    z = [scores[k] / max(temp, 1e-4) for k in keys]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return e[keys.index(t)] / sum(e)


def decompose(rec, c):
    orig = [k for k in rec["keys"] if k in c["scores"]]  # 刪除選項的操作不在此分析
    if len(orig) != len(rec["keys"]):
        return None
    added = [k for k in c["scores"] if k not in rec["keys"]]
    t = max(orig, key=lambda k: rec["base"]["scores"][k])

    def value(on):
        s_orig = c["scores"] if on["S"] else rec["base"]["scores"]
        scores = {k: s_orig[k] for k in orig}
        keys = orig[:]
        if on["D"]:
            scores.update({k: c["scores"][k] for k in added})
            keys += added
        temp = c["temp"] if on["T"] else rec["base"]["temp"]
        return conf(scores, keys, temp, t)

    vals = {combo: value(dict(zip(FACTORS, combo))) for combo in itertools.product((0, 1), repeat=3)}
    shap = {f: 0.0 for f in FACTORS}
    perms = list(itertools.permutations(range(3)))
    for order in perms:
        state = [0, 0, 0]
        for i in order:
            before = vals[tuple(state)]
            state[i] = 1
            shap[FACTORS[i]] += (vals[tuple(state)] - before) / len(perms)
    c0, c1 = vals[(0, 0, 0)], vals[(1, 1, 1)]
    out = {"delta": c1 - c0, **{f"shap_{f}": shap[f] for f in FACTORS}}
    for g in GATES:
        crossed = lambda v: (c0 >= g) != (v >= g)
        out[f"gate{g}"] = float(crossed(c1))
        for i, f in enumerate(FACTORS):  # 只改這一個因素就跨過門檻
            only = [0, 0, 0]
            only[i] = 1
            out[f"gate{g}_only_{f}"] = float(crossed(vals[tuple(only)]))
    return out


def main():
    print("# A3：決策信心變化的三因素拆解\n")
    print("S＝原有選項分數改變　D＝新選項稀釋　T＝溫度改變。數值為信心的平均變化（百分點），"
          "括號內為平均絕對貢獻；門檻欄為「總翻轉率｜只改 S／D／T 就跨過的比例」。\n")
    for task in TASKS:
        groups = ["permute", "filler_state", "gibberish", "irrel", "sibling", "duplicate"] if task == "typed" \
            else ["permute", "filler_state", "gibberish", "add_1", "add_3", "add_5", "duplicate"]
        printed = False
        for b in BACKENDS:
            f = ROOT / f"{task}__{b}.json"
            if not f.exists():
                continue
            if not printed:
                print(f"\n## {task}\n")
                print("| 模型 | 操作 | 總變化 | S | D | T | 門檻 0.7 | 門檻 0.9 |")
                print("|---|---|---|---|---|---|---|---|")
                printed = True
            recs = json.loads(f.read_text())["records"]
            agg = defaultdict(lambda: defaultdict(list))
            for rec in recs:
                for c in rec["conds"]:
                    g = group(c["cond"])
                    if g in groups:
                        m = decompose(rec, c)
                        if m:
                            for k, v in m.items():
                                agg[g][k].append(v)
            for g in groups:
                if g not in agg:
                    continue
                a = agg[g]
                mean = lambda k: st.mean(a[k])
                absm = lambda k: st.mean(abs(v) for v in a[k])
                parts = [f"{100*mean(f'shap_{x}'):+.1f}（{100*absm(f'shap_{x}'):.1f}）" for x in FACTORS]
                gates = [f"{100*mean(f'gate{g_}'):.1f}｜" + "／".join(f"{100*mean(f'gate{g_}_only_{x}'):.1f}" for x in FACTORS)
                         for g_ in GATES]
                print(f"| {b} | {g} | {100*mean('delta'):+.1f}（{100*absm('delta'):.1f}） | " + " | ".join(parts)
                      + " | " + " | ".join(gates) + " |")


if __name__ == "__main__":
    main()
