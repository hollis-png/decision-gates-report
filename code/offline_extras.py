"""第三輪審稿意見的離線分析（A2 評估半，路由模擬 K=5 vs 12，依句子鍵重抽 1,000 次）。

  paired    成對差：出貨校準 − 單一溫度 的順序打亂（自動處理 50%），以及自動處理 50% 時錯誤率的相對變化
  overlap   同一個 K 下，出貨校準與單一溫度的「前 50%」集合重疊比例；K=12 的同分比例
  risk      風險控制門檻：在擬合半上選使自動處理錯誤率 ≤ r 的最低門檻（每個 K 各自選），套到評估半
  hyst      遲滯門檻：信心名次在 50%±δ 之間的請求沿用 K=5 時的狀態
  cov90     自動處理 90% 時的順序打亂
"""
import json
import sys
from collections import defaultdict

import numpy as np

from calib_methods import A2, TASKS, Calibrator, is_fit, utt_key
from layers import boot_rows, thr_for

MODELS = ["laya_td", "laya_base", "laya_multi", "von", "qwen3_4b_set"]
KS = (5, 12)


def load(model):
    recs = [r for t in TASKS if (A2 / f"{t}__{model}.json").exists()
            for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]
    cal = Calibrator([r for r in recs if is_fit(r["qid"])])
    split = {True: defaultdict(dict), False: defaultdict(dict)}
    for r in recs:
        u, k = r["qid"].rsplit(":k", 1)
        split[is_fit(r["qid"])][u][int(k)] = r
    full = {f: {u: rs for u, rs in d.items() if all(k in rs for k in range(5, 13))} for f, d in split.items()}
    return cal, full


def rows(cal, m, rs_list, k):
    out = []
    for rs in rs_list:
        r = rs[k]
        d = max(r["keys"], key=r["base"]["scores"].get)
        out.append((d, cal.conf(m, r["base"]["scores"], r["keys"], r["base"]["temp"], d), d == r["anchor"]))
    return out


def ranks(conf):
    """信心名次的分位（0＝最高）。"""
    order = np.argsort(np.argsort(-np.asarray(conf), kind="stable"), kind="stable")
    return order / len(conf)


def paired(rng):
    print("## Paired differences (12 routes, 50% coverage)\n")
    print("| Model | Reordering shipped − single | Selective risk 5→12 routes, single (relative) |\n|---|---|---|")
    for model in MODELS:
        cal, full = load(model)
        ev = list(full[False].values())
        keys = [utt_key(u) for u in full[False]]
        res = {}
        for m in ("deployed", "global_T"):
            a, b = rows(cal, m, ev, 5), rows(cal, m, ev, 12)
            t5, t12 = thr_for([x[1] for x in a], .5), thr_for([x[1] for x in b], .5)
            res[m] = (a, b, t5, t12)

        def reorder(m, idx):
            a, b, t5, t12 = res[m]
            return 100 * np.mean([a[i][0] != b[i][0] or (a[i][1] >= t5) != (b[i][1] >= t12) for i in idx])
        d = boot_rows(ev, keys, rng, lambda idx: reorder("deployed", idx) - reorder("global_T", idx))
        a, b, t5, t12 = res["global_T"]
        risk = lambda x, t, idx: np.mean([not x[i][2] for i in idx if x[i][1] >= t])
        rel = boot_rows(ev, keys, rng, lambda idx: (risk(b, t12, idx) + 1e-9) / (risk(a, t5, idx) + 1e-9))
        print(f"| {model} | {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] | {100*risk(a,t5,range(len(a))):.1f}% → "
              f"{100*risk(b,t12,range(len(b))):.1f}% (×{rel[0]:.1f} [{rel[1]:.1f}, {rel[2]:.1f}]) |")
    print()


def overlap():
    print("## Overlap of the auto-handled half between shipped and single calibration, same K\n")
    print("| Model | K=5 | K=12 | ties among shipped confidences at K=12 |\n|---|---|---|---|")
    for model in MODELS:
        cal, full = load(model)
        ev = list(full[False].values())
        cells = []
        for k in KS:
            a, b = rows(cal, "deployed", ev, k), rows(cal, "global_T", ev, k)
            ta, tb = thr_for([x[1] for x in a], .5), thr_for([x[1] for x in b], .5)
            sa = {i for i, x in enumerate(a) if x[1] >= ta}
            sb = {i for i, x in enumerate(b) if x[1] >= tb}
            cells.append(f"{100*len(sa & sb)/max(len(sa), 1):.1f}%")
        c12 = np.array([x[1] for x in rows(cal, "deployed", ev, 12)])
        ties = 1 - len(np.unique(np.round(c12, 6))) / len(c12)
        print(f"| {model} | {cells[0]} | {cells[1]} | {100*ties:.1f}% |")
    print()


def risk_control(rng, targets=(0.02, 0.05)):
    print("## Plug-in risk thresholds, no finite-sample correction (chosen on the fitting half per K; evaluated on the evaluation half)\n")
    print("| Model | Calibration | Target risk | Auto-handled 5→12 | Realized risk 5→12 | Handling changes 5→12 |\n|---|---|---|---|---|---|")
    for model in MODELS:
        cal, full = load(model)
        fit, ev = list(full[True].values()), list(full[False].values())
        keys = [utt_key(u) for u in full[False]]
        for m in ("deployed", "global_T"):
            for r in targets:
                th = {}
                for k in KS:
                    f = sorted(rows(cal, m, fit, k), key=lambda x: -x[1])
                    best = f[0][1] + 1e-9  # 都達不到時不自動處理
                    errs = 0
                    for i, x in enumerate(f):
                        errs += (not x[2])
                        if errs / (i + 1) <= r:
                            best = x[1]
                    th[k] = best
                a, b = rows(cal, m, ev, 5), rows(cal, m, ev, 12)
                cov = [100 * np.mean([x[1] >= th[k] for x in z]) for k, z in ((5, a), (12, b))]
                rk = [100 * np.mean([not x[2] for x in z if x[1] >= th[k]]) if any(x[1] >= th[k] for x in z) else float("nan")
                      for k, z in ((5, a), (12, b))]
                h = boot_rows(ev, keys, rng, lambda idx: 100 * np.mean(
                    [a[i][0] != b[i][0] or (a[i][1] >= th[5]) != (b[i][1] >= th[12]) for i in idx]))
                print(f"| {model} | {m} | {int(100*r)}% | {cov[0]:.1f} → {cov[1]:.1f} | {rk[0]:.1f} → {rk[1]:.1f} | "
                      f"{h[0]:.1f} [{h[1]:.1f}, {h[2]:.1f}] |")
    print()


def hysteresis(rng, deltas=(0.0, 0.05, 0.1, 0.2)):
    print("## Hysteresis gate (50% target; requests ranked within 50%±δ keep their 5-route status)\n")
    print("| Model | Calibration | δ | Handling changes 5→12 | Auto-handled at 12 routes | Risk at 12 routes |\n|---|---|---|---|---|---|")
    for model in MODELS:
        cal, full = load(model)
        ev = list(full[False].values())
        keys = [utt_key(u) for u in full[False]]
        for m in ("deployed", "global_T"):
            a, b = rows(cal, m, ev, 5), rows(cal, m, ev, 12)
            ra, rb = ranks([x[1] for x in a]), ranks([x[1] for x in b])
            auto5 = ra < 0.5
            for dl in deltas:
                auto12 = np.where(rb < 0.5 - dl, True, np.where(rb >= 0.5 + dl, False, auto5))
                h = boot_rows(ev, keys, rng, lambda idx: 100 * np.mean(
                    [a[i][0] != b[i][0] or auto5[i] != auto12[i] for i in idx]))
                risk = 100 * np.mean([not b[i][2] for i in range(len(b)) if auto12[i]])
                print(f"| {model} | {m} | {dl:.2f} | {h[0]:.1f} [{h[1]:.1f}, {h[2]:.1f}] | {100*auto12.mean():.1f} | {risk:.1f} |")
    print()


def cov90(rng):
    print("## Reordering at 90% coverage (12 routes)\n")
    print("| Model | Calibration | Reordering |\n|---|---|---|")
    for model in MODELS:
        cal, full = load(model)
        ev = list(full[False].values())
        keys = [utt_key(u) for u in full[False]]
        for m in ("deployed", "global_T", "indep_top"):
            a, b = rows(cal, m, ev, 5), rows(cal, m, ev, 12)
            t5, t12 = thr_for([x[1] for x in a], .9), thr_for([x[1] for x in b], .9)
            h = boot_rows(ev, keys, rng, lambda idx: 100 * np.mean(
                [a[i][0] != b[i][0] or (a[i][1] >= t5) != (b[i][1] >= t12) for i in idx]))
            print(f"| {model} | {m} | {h[0]:.1f} [{h[1]:.1f}, {h[2]:.1f}] |")
    print()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    jobs = {"paired": lambda: paired(rng), "overlap": overlap, "risk": lambda: risk_control(rng),
            "hyst": lambda: hysteresis(rng), "cov90": lambda: cov90(rng)}
    for j in sys.argv[1:] or list(jobs):
        jobs[j]()
