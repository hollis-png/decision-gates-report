"""論文表 5（校準方式）與表 7（混合式）的 95% 信賴區間。完全離線，重用 calib_methods.py 與 hybrid.py。

校準參數固定在擬合半（與原表相同）；信賴區間只反映評估半的抽樣變異。
重抽單位：表 5 是句子（A2 的 K=2..12 一起抽；第三版每句一筆）；表 7 是請求（同一請求的所有操作一起抽）。
表 7 的選擇性準確率：每次重抽取信心最高的 50%（同 hybrid.py）；處理改變的門檻固定為原樣本 50% 覆蓋率的門檻。
"""
import json
from collections import defaultdict

import numpy as np

from calib_methods import A2, TASKS, V3, Calibrator, ece, is_fit, utt_key
import hybrid

N_BOOT = 1000
T5 = {"laya_td": ["deployed", "global_T", "bucket_T", "indep_top"], "von": ["deployed", "global_T", "indep_top"]}


def cluster_boot(groups, stat, rng):
    """groups: 每個重抽單位的資料；stat(被抽中的 groups 串列) → 純量。回傳 (點估計, 2.5%, 97.5%)。"""
    point = stat(groups)
    n = len(groups)
    reps = [stat([groups[i] for i in rng.integers(0, n, n)]) for _ in range(N_BOOT)]
    return point, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def clustered(items, keys):
    """依句子鍵分組，供 cluster_boot 以句子為單位重抽（隨機版與相近版的同一句話同進同出）。"""
    d = defaultdict(list)
    for it, k in zip(items, keys):
        d[k].append(it)
    return list(d.values())


def flat(gs):
    return [x for g in gs for x in g]


def selective_acc(conf, corr, cov):
    """信心最高的 cov 比例的準確率；門檻上的同分請求按比例計入（等於隨機打破同分的期望值）。"""
    conf, corr = np.asarray(conf), np.asarray(corr, float)
    n = int(cov * len(conf))
    t = np.sort(conf)[::-1][n - 1]
    above, tie = conf > t, conf == t
    need = n - above.sum()
    return 100 * float((corr[above].sum() + need * corr[tie].mean()) / n)


def table5(rng):
    print("## Table 5 with 95% CIs\n")
    print("| Model | Method | ECE | Brier | Gate flips +1 | Gate flips +5 |\n|---|---|---|---|---|---|")
    for model, methods in T5.items():
        a2 = [r for t in TASKS if (A2 / f"{t}__{model}.json").exists()
              for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]
        cal = Calibrator([r for r in a2 if is_fit(r["qid"])])
        v3 = [r for t in TASKS if (V3 / f"{t}__{model}.json").exists()
              for r in json.loads((V3 / f"{t}__{model}.json").read_text())["records"] if not is_fit(r["qid"])]
        for m in methods:
            utt = defaultdict(lambda: {"conf": [], "corr": [], "g1": [], "g5": []})
            for r in a2:
                if is_fit(r["qid"]):
                    continue
                keys, s = r["keys"], r["base"]["scores"]
                dec = max(keys, key=s.get)
                c0 = cal.conf(m, s, keys, r["base"]["temp"], dec)
                c = r["conds"][0]
                c1 = cal.conf(m, c["scores"], list(c["scores"]), c["temp"], dec)
                u = utt[utt_key(r["qid"])]
                u["conf"].append(c0)
                u["corr"].append(float(dec == r["anchor"]))
                u["g1"].append((c0 >= 0.9) != (c1 >= 0.9))
            for r in v3:
                keys, s = r["keys"], r["base"]["scores"]
                dec = max(keys, key=s.get)
                c0 = cal.conf(m, s, keys, r["base"]["temp"], dec)
                for c in r["conds"]:
                    if c["cond"] == "add_5":
                        c1 = cal.conf(m, c["scores"], list(c["scores"]), c["temp"], dec)
                        utt[utt_key(r["qid"])]["g5"].append((c0 >= 0.9) != (c1 >= 0.9))
            groups = list(utt.values())
            cat = lambda gs, k: [x for g in gs for x in g[k]]
            stats = {
                "ECE": lambda gs: 100 * ece(cat(gs, "conf"), cat(gs, "corr")),
                "Brier": lambda gs: float(np.mean((np.asarray(cat(gs, "conf")) - np.asarray(cat(gs, "corr"))) ** 2)),
                "+1": lambda gs: 100 * float(np.mean(cat(gs, "g1"))),
                "+5": lambda gs: 100 * float(np.mean(cat(gs, "g5"))),
            }
            cells = []
            for name, f in stats.items():
                p, lo, hi = cluster_boot(groups, f, rng)
                fmt = "{:.3f} [{:.3f}, {:.3f}]" if name == "Brier" else "{:.1f} [{:.1f}, {:.1f}]"
                cells.append(fmt.format(p, lo, hi))
            print(f"| {model} | {m} | " + " | ".join(cells) + " |")
    print()


def table7(rng):
    print("## Table 7 with 95% CIs (selective accuracy at 50% coverage / handling changes after +5)\n")
    print("| Family | Method | Selective accuracy | Handling changes +5 |\n|---|---|---|---|")
    for fam in hybrid.FAMILIES:
        pairs = hybrid.load_pairs(hybrid.V3, hybrid.V3_TASKS, fam)
        h = hybrid.Hybrid([pr for pr in pairs if is_fit(pr[0]["qid"])])
        ev = [pr for pr in pairs if not is_fit(pr[0]["qid"])]
        for m in hybrid.METHODS:
            base = [h.decide(m, s["base"]["scores"], p["base"]["scores"], s["keys"]) for s, p in ev]
            thr = hybrid.coverage_gates([c for _, c in base])[0.5]
            groups, gkeys = [], []
            for (s, p), (dec, c) in zip(ev, base):
                sc = {x["cond"]: x for x in s["conds"]}
                pc = {x["cond"]: x for x in p["conds"]}
                flip = None
                if "add_5" in sc and "add_5" in pc:
                    gf, _ = hybrid.flips(h, m, s["base"]["scores"], p["base"]["scores"], s["keys"],
                                         sc["add_5"]["scores"], pc["add_5"]["scores"], list(sc["add_5"]["scores"]),
                                         gates=[thr])
                    flip = float(gf[thr])
                groups.append((c, float(dec == s["anchor"]), flip))
                gkeys.append(utt_key(s["qid"]))
            groups = clustered(groups, gkeys)
            sel = lambda gs: selective_acc([g[0] for g in flat(gs)], [g[1] for g in flat(gs)], 0.5)
            fl = lambda gs: 100 * float(np.mean([f for _, _, f in flat(gs) if f is not None]))
            a, b = cluster_boot(groups, sel, rng), cluster_boot(groups, fl, rng)
            print(f"| {fam} | {m} | {a[0]:.1f} [{a[1]:.1f}, {a[2]:.1f}] | {b[0]:.1f} [{b[1]:.1f}, {b[2]:.1f}] |")
    print()


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)
    if "--t7" not in sys.argv:
        table5(rng)
    table7(rng)
