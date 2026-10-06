"""分組溫度的估計變異：每組只用 n 筆擬合時，溫度有多分散、會造成多少門檻翻轉。

假說：Laya 官方溫度表的極端值（如 11+ 組的 0.10）來自小樣本擬合的估計變異；
     依選項數分組的校準，把總樣本切成多組，每組樣本變少，估計就不穩，組間差異會變成跳躍。
做法：從擬合半每組抽 n 筆（n = 10/20/50/100/300），重複 50 次；
     對照：用同樣總數（4n）擬合單一溫度。在評估半量 A2「K→K+1」的門檻 0.9 翻轉與 ECE。
"""
import json
import random
import statistics as st

import numpy as np

from calib_methods import A2, TASKS, Calibrator, bucket, ece, fit_temperature, is_fit, softmax, arr

MODELS = ["laya_td", "von"]
NS = [10, 20, 50, 100, 300]
REPS = 50


def load(model):
    return [r for t in TASKS if (A2 / f"{t}__{model}.json").exists()
            for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]


def gate_and_ece(temps_of_k, ev):
    conf, corr, g9 = [], [], []
    for r in ev:
        keys, s = r["keys"], r["base"]["scores"]
        dec = max(keys, key=s.get)
        z0, z1 = arr(s, keys), arr(r["conds"][0]["scores"], list(r["conds"][0]["scores"]))
        c0 = float(softmax(z0 / temps_of_k(len(keys)))[keys.index(dec)])
        k1 = list(r["conds"][0]["scores"])
        c1 = float(softmax(z1 / temps_of_k(len(k1)))[k1.index(dec)])
        conf.append(c0)
        corr.append(dec == r["anchor"])
        g9.append((c0 >= 0.9) != (c1 >= 0.9))
    return float(np.mean(g9)), ece(conf, np.asarray(corr, dtype=float))


def main():
    print("# 分組溫度的估計變異（每組 n 筆擬合，重複 50 次；評估半的 A2 門檻 0.9 翻轉）\n")
    print("| 模型 | 每組 n | 分組溫度的組間全距（平均 ± 標準差） | 分組：門檻 0.9 翻轉 | 分組：ECE | 單一溫度（總數 4n）：門檻 0.9 翻轉 | 單一：ECE |")
    print("|---|---|---|---|---|---|---|")
    for model in MODELS:
        recs = load(model)
        fit = [r for r in recs if is_fit(r["qid"])]
        ev = [r for r in recs if not is_fit(r["qid"])]
        items = [(arr(r["base"]["scores"], r["keys"]), r["keys"].index(r["anchor"]), len(r["keys"])) for r in fit]
        by_b = {b: [(z, y) for z, y, k in items if bucket(k) == b] for b in range(4)}
        for n in NS:
            rng = random.Random(n)
            spreads, gb, eb, gg, eg = [], [], [], [], []
            for _ in range(REPS):
                samp = {b: rng.sample(v, min(n, len(v))) for b, v in by_b.items()}
                tb = {b: fit_temperature(s) for b, s in samp.items()}
                spreads.append(max(tb.values()) - min(tb.values()))
                g, e = gate_and_ece(lambda k: tb[bucket(k)], ev)
                gb.append(g)
                eb.append(e)
                tg = fit_temperature([x for s in samp.values() for x in s])
                g, e = gate_and_ece(lambda k: tg, ev)
                gg.append(g)
                eg.append(e)
            f = lambda v: f"{100*st.mean(v):.1f} ± {100*st.pstdev(v):.1f}"
            print(f"| {model} | {n} | {st.mean(spreads):.2f} ± {st.pstdev(spreads):.2f} | {f(gb)} | {f(eb)} | {f(gg)} | {f(eg)} |",
                  flush=True)


if __name__ == "__main__":
    main()
