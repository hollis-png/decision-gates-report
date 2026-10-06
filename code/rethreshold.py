"""決定性實驗（A 或 B）：每個 K 各自重設門檻、使自動處理比例固定時，各校準方式還剩多少處理改變。

A：單一溫度仍明顯比出貨校準穩 → 校準設計除了造成覆蓋率跳動，還會打亂請求間的信心順序。
B：重設門檻後各校準差不多 → 校準只決定覆蓋率怎麼跳，剩下的不穩定與校準無關。

兩種情境（A2 評估半，依句子鍵重抽 1,000 次）：
  route  路由模擬 K=5→6 與 5→12：每個 K 的門檻取該 K 的基準信心分位數，使自動處理比例 = cov
  sweep  劑量反應 K→K+1：基準與加選項後各自依自己的信心分位數設門檻（同一 cov），K 平均
處理改變＝決策換人，或「是否自動處理」改變。另報告 route-only（只看決策換人）作為底線。
"""
import json
from collections import defaultdict

import numpy as np

from calib_methods import A2, TASKS, Calibrator, is_fit, utt_key
from ci_tables import cluster_boot, clustered, flat
from validate_gap import Two

MODELS = ["laya_td", "laya_base", "laya_multi", "von", "qwen3_4b_set"]
METHODS = ["deployed", "global_T", "indep_top", "gap"]
COVS = (0.3, 0.5, 0.7)
KS = list(range(5, 13))


def thr_for(conf, cov):
    """使 conf >= t 的比例最接近 cov 的門檻（取第 n 大的值）。"""
    s = np.sort(np.asarray(conf))[::-1]
    return float(s[max(int(cov * len(s)) - 1, 0)])


def conf_fn(cal, two, m):
    if m == "gap":
        return lambda r, scores, keys, temp, d: two.conf("gap", scores, keys, d)
    return lambda r, scores, keys, temp, d: cal.conf(m, scores, keys, temp, d)


def run(rng):
    print("| Model | Method | Coverage | Route 5→6 | Route 5→12 | Sweep +1 (mean over K) |")
    print("|---|---|---|---|---|---|")
    for model in MODELS:
        recs = [r for t in TASKS if (A2 / f"{t}__{model}.json").exists()
                for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]
        fit = [r for r in recs if is_fit(r["qid"])]
        cal, two = Calibrator(fit), Two(fit)
        ev = [r for r in recs if not is_fit(r["qid"])]
        by = defaultdict(dict)
        for r in ev:
            u, k = r["qid"].rsplit(":k", 1)
            by[u][int(k)] = r
        full = {u: rs for u, rs in by.items() if all(k in rs for k in KS)}
        ukeys = [utt_key(u) for u in full]
        route_only = None
        for m in METHODS:
            f = conf_fn(cal, two, m)

            def base(r):
                d = max(r["keys"], key=r["base"]["scores"].get)
                return d, f(r, r["base"]["scores"], r["keys"], r["base"]["temp"], d)

            def edited(r):
                c = r["conds"][0]
                k1 = list(c["scores"])
                d = max(k1, key=c["scores"].get)
                return d, f(r, c["scores"], k1, c["temp"], d)

            # 路由模擬：每個 K 的基準信心
            dk = {k: [base(rs[k]) for rs in full.values()] for k in KS}
            # 劑量反應：每個基準 K 的基準與加選項後
            sw = defaultdict(list)
            for r in ev:
                sw[r["K"]].append((utt_key(r["qid"]), base(r), edited(r)))
            for cov in COVS:
                th = {k: thr_for([c for _, c in dk[k]], cov) for k in KS}
                auto = {k: np.array([c >= th[k] for _, c in dk[k]]) for k in KS}
                dec = {k: [d for d, _ in dk[k]] for k in KS}
                cells = []
                for K in (6, 12):
                    ch = [float(dec[K][i] != dec[5][i] or auto[K][i] != auto[5][i]) for i in range(len(full))]
                    p = cluster_boot(clustered(ch, ukeys), lambda gs: 100 * float(np.mean(flat(gs))), rng)
                    cells.append(f"{p[0]:.1f} [{p[1]:.1f}, {p[2]:.1f}]")
                    if m == METHODS[0] and cov == COVS[0]:
                        ro = [float(dec[K][i] != dec[5][i]) for i in range(len(full))]
                        route_only = (route_only or []) + [100 * np.mean(ro)]
                # 劑量反應：每個 K 各自重設
                items, keys = [], []
                for K, rows in sw.items():
                    t0 = thr_for([b[1] for _, b, _ in rows], cov)
                    t1 = thr_for([e[1] for _, _, e in rows], cov)
                    for k, b, e in rows:
                        items.append(float(b[0] != e[0] or (b[1] >= t0) != (e[1] >= t1)))
                        keys.append(k)
                p = cluster_boot(clustered(items, keys), lambda gs: 100 * float(np.mean(flat(gs))), rng)
                cells.append(f"{p[0]:.1f} [{p[1]:.1f}, {p[2]:.1f}]")
                print(f"| {model} | {m} | {int(cov*100)}% | " + " | ".join(cells) + " |")
        print(f"| {model} | route only | — | {route_only[0]:.1f} | {route_only[1]:.1f} | — |")


if __name__ == "__main__":
    run(np.random.default_rng(0))
