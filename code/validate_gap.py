"""獨立驗證「前兩名差距校準」（第 5 節是在看過第三版加 5 個選項的評估結果後才加入的）。

驗證資料都沒有參與選方法：
  sweep    劑量反應（A2，K=2..12 各加 1 個選項）：處理改變（決策換人或門檻翻轉）的 K 平均
  route    路由模擬（A2 巢狀清單 K=5→12）：K=6 與 K=12 時處理方式和 K=5 不同的比例
  split    第三版加 5 個選項，依隨機干擾／相近干擾／typed-decisions 分開（大型語言模型，集合式決策）
比較：單一溫度的正規化機率（global_T）vs Platt 校準集合式前兩名原始分數差距（gap）。
門檻：固定 0.9，以及各方法各自設門檻使 50% 自動處理。擬合／評估依句子雜湊切半（同 calib_methods.is_fit）。
"""
import json
import math
from collections import defaultdict

import numpy as np

import hybrid
from calib_methods import A2, TASKS, arr, fit_platt, fit_temperature, is_fit, softmax, utt_key
from ci_tables import N_BOOT, cluster_boot, clustered, flat, selective_acc
from review_extras import Learned

MODELS = ["laya_td", "laya_base", "laya_multi", "von", "qwen3_4b_set"]
KS = list(range(5, 13))


def gap(scores, keys, dec):
    return scores[dec] - max(scores[k] for k in keys if k != dec)


class Two:
    """single：正規化機率、單一溫度；gap：前兩名差距的 Platt。只用擬合半的基準請求擬合。"""

    def __init__(self, fit):
        items = [(arr(r["base"]["scores"], r["keys"]), r["keys"].index(r["anchor"])) for r in fit]
        self.t = fit_temperature(items)
        g = []
        for r in fit:
            s, k = r["base"]["scores"], r["keys"]
            d = max(k, key=s.get)
            g.append((np.array([gap(s, k, d)]), 0 if d == r["anchor"] else -1))
        self.pt, self.pb = fit_platt(g)

    def conf(self, method, scores, keys, dec):
        if method == "single":
            p = softmax(arr(scores, keys) / self.t)
            return float(p[keys.index(dec)])
        x = gap(scores, keys, dec) / self.pt + self.pb
        return float(1 / (1 + math.exp(-x))) if x > -30 else 0.0


def load(model):
    return [r for t in TASKS if (A2 / f"{t}__{model}.json").exists()
            for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]


def sweep_and_route(rng):
    print("## Sweep (+1 option, mean over K) and routing simulation (5 -> 12 routes), evaluation half, 95% CI\n")
    print("| Model | Method | Gate | Sel. acc @50% | Handling +1 (mean over K) | Route sim K=6 | Route sim K=12 |")
    print("|---|---|---|---|---|---|---|")
    for model in MODELS:
        recs = load(model)
        two = Two([r for r in recs if is_fit(r["qid"])])
        ev = [r for r in recs if not is_fit(r["qid"])]
        by_utt = defaultdict(dict)
        for r in ev:
            u, k = r["qid"].rsplit(":k", 1)
            by_utt[u][int(k)] = r
        for m in ("single", "gap"):
            base = []
            for r in ev:
                s, k = r["base"]["scores"], r["keys"]
                d = max(k, key=s.get)
                base.append((two.conf(m, s, k, d), float(d == r["anchor"])))
            c_all = np.sort([c for c, _ in base])[::-1]
            full = {u: rs for u, rs in by_utt.items() if all(k in rs for k in KS)}
            c5 = []
            for rs in full.values():
                r5 = rs[5]
                d5 = max(r5["keys"], key=r5["base"]["scores"].get)
                c5.append(two.conf(m, r5["base"]["scores"], r5["keys"], d5))
            c5 = np.sort(c5)[::-1]
            for gname, thr_all, thr5 in (("0.9", 0.9, 0.9),
                                         ("50% cov.", float(c_all[len(c_all) // 2 - 1]), float(c5[len(c5) // 2 - 1]))):
                groups = defaultdict(lambda: {"h1": [], "conf": [], "corr": []})
                for r, (c0, y) in zip(ev, base):
                    s, k = r["base"]["scores"], r["keys"]
                    d0 = max(k, key=s.get)
                    cn = r["conds"][0]
                    k1 = list(cn["scores"])
                    d1 = max(k1, key=cn["scores"].get)
                    c1 = two.conf(m, cn["scores"], k1, d1)
                    g = groups[utt_key(r["qid"])]
                    g["h1"].append(float(d1 != d0 or (c0 >= thr_all) != (c1 >= thr_all)))
                    g["conf"].append(c0)
                    g["corr"].append(y)
                gl = list(groups.values())
                cat = lambda gs, key: [x for g in gs for x in g[key]]
                sel = cluster_boot(gl, lambda gs: selective_acc(cat(gs, "conf"), cat(gs, "corr"), 0.5), rng)
                h1 = cluster_boot(gl, lambda gs: 100 * float(np.mean(cat(gs, "h1"))), rng)
                route, rkeys = [], []
                for u, rs in full.items():
                    rkeys.append(utt_key(u))
                    r5 = rs[5]
                    d5 = max(r5["keys"], key=r5["base"]["scores"].get)
                    a5 = two.conf(m, r5["base"]["scores"], r5["keys"], d5) >= thr5
                    row = []
                    for kk in (6, 12):
                        r = rs[kk]
                        d = max(r["keys"], key=r["base"]["scores"].get)
                        row.append(float(d != d5 or (two.conf(m, r["base"]["scores"], r["keys"], d) >= thr5) != a5))
                    route.append(row)
                route = np.asarray(route)
                r6 = cluster_boot(clustered(list(route[:, 0]), rkeys), lambda gs: 100 * float(np.mean(flat(gs))), rng)
                r12 = cluster_boot(clustered(list(route[:, 1]), rkeys), lambda gs: 100 * float(np.mean(flat(gs))), rng)
                f = lambda t: f"{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]"
                print(f"| {model} | {m} | {gname} | {f(sel)} | {f(h1)} | {f(r6)} | {f(r12)} |")
    print()


def split(rng):
    print("## +5 options split by distractor type (LLM joint decisions, 50% coverage), 95% CI\n")
    print("| Family | Subset | Method | Sel. acc @50% | Handling +5 |\n|---|---|---|---|---|")
    subsets = {"random": ["banking", "clinc"], "near": ["banking_hard", "clinc_hard"], "typed": ["typed"]}
    for fam in hybrid.FAMILIES:
        pairs = hybrid.load_pairs(hybrid.V3, hybrid.V3_TASKS, fam)
        fit = [pr for pr in pairs if is_fit(pr[0]["qid"])]
        h = hybrid.Hybrid(fit)
        gapm = Learned(h, fit, "gap")
        for sub, tasks in subsets.items():
            ev = [pr for pr in pairs if not is_fit(pr[0]["qid"]) and
                  (pr[0]["qid"].split(":")[0] in tasks if sub != "typed" else pr[0]["qid"].split(":")[0].isdigit())]
            for mname, dec_fn in (("single", lambda s, p, k: h.decide("set_T", s, p, k)),
                                  ("gap", lambda s, p, k: gapm.decide(s, p, k))):
                base = [dec_fn(s["base"]["scores"], p["base"]["scores"], s["keys"]) for s, p in ev]
                cs = np.sort([c for _, c in base])[::-1]
                thr = float(cs[len(cs) // 2 - 1])
                groups, gkeys = [], []
                for (s, p), (d0, c0) in zip(ev, base):
                    sc = {x["cond"]: x for x in s["conds"]}
                    pc = {x["cond"]: x for x in p["conds"]}
                    fl = None
                    if "add_5" in sc and "add_5" in pc:
                        k1 = list(sc["add_5"]["scores"])
                        d1, c1 = dec_fn(sc["add_5"]["scores"], pc["add_5"]["scores"], k1)
                        fl = float(d1 != d0 or (c0 >= thr) != (c1 >= thr))
                    groups.append((c0, float(d0 == s["anchor"]), fl))
                    gkeys.append(utt_key(s["qid"]))
                if not any(g[2] is not None for g in groups):
                    continue
                sel = cluster_boot(clustered(groups, gkeys), lambda gs: selective_acc([g[0] for g in flat(gs)], [g[1] for g in flat(gs)], 0.5), rng)
                fl = cluster_boot(clustered(groups, gkeys), lambda gs: 100 * float(np.mean([g[2] for g in flat(gs) if g[2] is not None])), rng)
                print(f"| {fam} | {sub} (n={len(ev)}) | {mname} | {sel[0]:.1f} [{sel[1]:.1f}, {sel[2]:.1f}] | "
                      f"{fl[0]:.1f} [{fl[1]:.1f}, {fl[2]:.1f}] |")
    print()


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)
    for job in sys.argv[1:] or ["sweep", "split"]:
        {"sweep": sweep_and_route, "split": split}[job](rng)
