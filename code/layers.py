"""v3 主軸的兩層分析（完全離線，A2 評估半，依句子鍵重抽 1,000 次）。

路由模擬：同一句話 K=5..12 的巢狀清單。每個 K 報告
  coverage   門檻 0.9 時自動處理的比例（第一層：覆蓋率移動）
  err_auto   門檻 0.9 時自動處理請求中的錯誤率
  reorder    每個 K 各自重設門檻、使自動處理 50% 時，處理方式和 K=5 不同的比例（第二層：順序打亂）
  risk50     自動處理 50% 時的錯誤率（選擇性風險），看順序打亂是否「該變」
  wrong_share 順序打亂的改變中，涉及錯誤決策（改變前或後的決策錯誤）的比例
另：Jev 在劑量反應中門檻 0.9 的覆蓋率（四捨五入到 0.01，不做重設門檻）。
"""
import json
from collections import defaultdict

import numpy as np

from calib_methods import A2, TASKS, Calibrator, is_fit, utt_key
from ci_tables import N_BOOT

MODELS = ["laya_td", "laya_base", "laya_multi", "von", "qwen3_4b_set"]
METHODS = ["deployed", "global_T", "indep_top"]
KS = list(range(5, 13))
COV = 0.5


def thr_for(conf, cov):
    s = np.sort(np.asarray(conf))[::-1]
    return float(s[max(int(cov * len(s)) - 1, 0)])


def boot_rows(values, keys, rng, stat):
    """values：每列一筆（dict 或數值）；依 keys 叢集重抽，stat(被抽中的列) → 純量。"""
    by = defaultdict(list)
    for i, k in enumerate(keys):
        by[k].append(i)
    cl = list(by.values())
    point = stat(list(range(len(values))))
    reps = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(cl), len(cl))
        reps.append(stat([i for c in pick for i in cl[c]]))
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return point, lo, hi


def model_curves(model, rng):
    recs = [r for t in TASKS if (A2 / f"{t}__{model}.json").exists()
            for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]
    cal = Calibrator([r for r in recs if is_fit(r["qid"])])
    by = defaultdict(dict)
    for r in recs:
        if not is_fit(r["qid"]):
            u, k = r["qid"].rsplit(":k", 1)
            by[u][int(k)] = r
    full = {u: rs for u, rs in by.items() if all(k in rs for k in KS)}
    keys = [utt_key(u) for u in full]
    out = {}
    for m in METHODS:
        d = {}
        for k in KS:
            rows = []
            for rs in full.values():
                r = rs[k]
                dec = max(r["keys"], key=r["base"]["scores"].get)
                rows.append((dec, cal.conf(m, r["base"]["scores"], r["keys"], r["base"]["temp"], dec), dec == r["anchor"]))
            d[k] = rows
        th = {k: thr_for([c for _, c, _ in d[k]], COV) for k in KS}
        res = {}
        for k in KS:
            cov9 = lambda idx: 100 * np.mean([d[k][i][1] >= 0.9 for i in idx])
            err9 = lambda idx: (100 * np.mean([not d[k][i][2] for i in idx if d[k][i][1] >= 0.9])
                                if any(d[k][i][1] >= 0.9 for i in idx) else np.nan)
            risk = lambda idx: 100 * np.mean([not d[k][i][2] for i in idx if d[k][i][1] >= th[k]])

            def reorder(idx):
                return 100 * np.mean([(d[k][i][0] != d[5][i][0]) or ((d[k][i][1] >= th[k]) != (d[5][i][1] >= th[5]))
                                      for i in idx])

            def wrong_share(idx):
                ch = [i for i in idx if (d[k][i][0] != d[5][i][0]) or ((d[k][i][1] >= th[k]) != (d[5][i][1] >= th[5]))]
                return 100 * np.mean([(not d[k][i][2]) or (not d[5][i][2]) for i in ch]) if ch else np.nan

            res[k] = {name: boot_rows(full, keys, rng, f) for name, f in
                      [("coverage", cov9), ("err_auto", err9), ("reorder", reorder), ("risk50", risk),
                       ("wrong_share", wrong_share)]}
        out[m] = res
    return out, len(full)


def jev_coverage():
    print("## Jev: share of requests with confidence >= 0.9 in the dose-response sweep (rounded probabilities)\n")
    print("| Set | " + " | ".join(f"K={k}" for k in range(2, 13)) + " |\n|---|" + "---|" * 11)
    for t in TASKS:
        f = A2 / f"{t}__jev.json"
        if not f.exists():
            continue
        by = defaultdict(list)
        for r in json.loads(f.read_text())["records"]:
            p = r["base"]["probs"]
            dec = max(r["keys"], key=lambda k: p.get(k, 0.0))
            by[r["K"]].append(p[dec] >= 0.9)
        print(f"| {t} | " + " | ".join(f"{100*np.mean(by[k]):.1f}" for k in range(2, 13)) + " |")
    print()


def main():
    rng = np.random.default_rng(0)
    allres = {}
    for model in MODELS:
        res, n = model_curves(model, rng)
        allres[model] = res
        print(f"## {model} (n = {n})\n")
        print("| Method | Measure | " + " | ".join(f"K={k}" for k in KS) + " |\n|---|---|" + "---|" * len(KS))
        for m, r in res.items():
            for name in ("coverage", "err_auto", "reorder", "risk50", "wrong_share"):
                print(f"| {m} | {name} | " + " | ".join(
                    f"{r[k][name][0]:.1f} [{r[k][name][1]:.1f}, {r[k][name][2]:.1f}]" for k in KS) + " |")
        print()
    json.dump(allres, open("layers_results.json", "w"), default=float)
    jev_coverage()


if __name__ == "__main__":
    main()
