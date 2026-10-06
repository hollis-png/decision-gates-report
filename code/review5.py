"""第五輪審稿意見的離線分析（只用已存分數）。

  ood       跨資料集校準：在 Banking 擬合、在 CLINC 評估（及反方向）；單一溫度、分組溫度、平滑溫度 T(K)=exp(a+b log K)
            報告門檻 0.9 的自動處理比例與錯誤率（K=5→12）、自動處理一半時的風險、重設門檻後的重排
  matched   錯誤上升的拆解：K=12 時門檻 0.9 的覆蓋率 c12 與錯誤 e12；K=5 時「同樣覆蓋率 c12」下的錯誤 e5m
            覆蓋率造成的部分 = e5m − e5，任務變難的部分 = e12 − e5m（依句子重抽的 95% 區間）
  null      重排的參考基準：iia_v3 的 5 選項基準清單，只打亂順序（permute）、加 1 個、加 5 個選項，各自重設門檻到 50%
            另報「限縮版」：只算至少在一邊被自動處理的請求
  launch    上線模擬加上信賴區間，新路由流量 10% 與 30%
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from calib_methods import A2, Calibrator, is_fit, utt_key
from layers import thr_for

MODELS = ["laya_td", "laya_base", "laya_multi", "von", "qwen3_4b_set"]
V3 = Path("iia_v3_results")
L = Path("launch_results")
B = 1000
RNG = np.random.default_rng(0)
BANK, CLINC = ["banking", "banking_hard"], ["clinc", "clinc_hard"]


def load(model, tasks, path=A2):
    return [r for t in tasks if (path / f"{t}__{model}.json").exists()
            for r in json.loads((path / f"{t}__{model}.json").read_text())["records"]]


def by_utt(recs):
    d = defaultdict(dict)
    for r in recs:
        u, k = r["qid"].rsplit(":k", 1)
        d[u][int(k)] = r
    return {u: rs for u, rs in d.items() if all(k in rs for k in range(5, 13))}


def rows(cal, m, rs_list, k):
    out = []
    for rs in rs_list:
        r = rs[k]
        d = max(r["keys"], key=r["base"]["scores"].get)
        out.append((d, cal.conf(m, r["base"]["scores"], r["keys"], r["base"]["temp"], d), d == r["anchor"]))
    return out


def clusters(keys):
    by = defaultdict(list)
    for i, k in enumerate(keys):
        by[k].append(i)
    return list(by.values())


def boot(keys, stat):
    cl = clusters(keys)
    pt = stat(np.arange(len(keys)))
    reps = []
    for _ in range(B):
        pick = RNG.integers(0, len(cl), len(cl))
        reps.append(stat(np.concatenate([cl[j] for j in pick])))
    reps = np.asarray(reps, dtype=float)
    reps = reps[~np.isnan(reps)]
    lo, hi = np.percentile(reps, [2.5, 97.5]) if len(reps) else (np.nan, np.nan)
    return pt, lo, hi


def fmt(t):
    return f"{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]"


def cov_err(x, idx, thr):
    c = np.array([x[i][1] for i in idx])
    ok = np.array([x[i][2] for i in idx])
    a = c >= thr
    return 100 * a.mean(), (100 * np.mean(~ok[a]) if a.any() else np.nan)


def reorder(a, b, idx):
    ca, cb = [a[i][1] for i in idx], [b[i][1] for i in idx]
    ta, tb = thr_for(ca, .5), thr_for(cb, .5)
    return 100 * np.mean([a[i][0] != b[i][0] or (a[i][1] >= ta) != (b[i][1] >= tb) for i in idx])


def risk50(x, idx):
    c = [x[i][1] for i in idx]
    t = thr_for(c, .5)
    return 100 * np.mean([not x[i][2] for i in idx if x[i][1] >= t])


# ---------------- ood ----------------

def ood():
    print("## Cross-dataset calibration (fit on one intent dataset, all K; evaluate on the other)\n")
    print("| Model | Direction | Calibration | Auto-handled τ=0.9, 5→12 | Error among auto, 5→12 | Sel. risk 50%, 5→12 | Reordering 12 routes |")
    print("|---|---|---|---|---|---|---|")
    for model in MODELS:
        for name, (fit_t, ev_t) in {"banking→clinc": (BANK, CLINC), "clinc→banking": (CLINC, BANK)}.items():
            cal = Calibrator(load(model, fit_t))
            full = by_utt(load(model, ev_t))
            ev, keys = list(full.values()), [utt_key(u) for u in full]
            for m in ("deployed", "global_T", "bucket_T", "smooth_T"):
                a, b = rows(cal, m, ev, 5), rows(cal, m, ev, 12)
                idx = np.arange(len(a))
                ca, ea = cov_err(a, idx, .9)
                cb, eb = cov_err(b, idx, .9)
                ra, rb = risk50(a, idx), risk50(b, idx)
                h = boot(keys, lambda i: reorder(a, b, i))
                print(f"| {model} | {name} | {m} | {ca:.1f} → {cb:.1f} | {ea:.1f} → {eb:.1f} | {ra:.1f} → {rb:.1f} | {fmt(h)} |")
        # 同資料的平滑溫度（擬合半 → 評估半），供對照
        recs = load(model, BANK + CLINC)
        cal = Calibrator([r for r in recs if is_fit(r["qid"])])
        full = {u: rs for u, rs in by_utt(recs).items() if not is_fit(u + ":k5")}
        ev, keys = list(full.values()), [utt_key(u) for u in full]
        for m in ("smooth_T",):
            a, b = rows(cal, m, ev, 5), rows(cal, m, ev, 12)
            idx = np.arange(len(a))
            ca, ea = cov_err(a, idx, .9)
            cb, eb = cov_err(b, idx, .9)
            h = boot(keys, lambda i: reorder(a, b, i))
            print(f"| {model} | in-domain | {m} (a, b = {cal.smooth[0]:.2f}, {cal.smooth[1]:.2f}) | {ca:.1f} → {cb:.1f} | "
                  f"{ea:.1f} → {eb:.1f} | {risk50(a, idx):.1f} → {risk50(b, idx):.1f} | {fmt(h)} |")
    print()


# ---------------- matched ----------------

def matched():
    print("## Error rise at τ=0.9, five → twelve routes, split into coverage and task difficulty (evaluation half)\n")
    print("e5m: error at five routes among the top-c12 requests by five-route confidence, where c12 is the twelve-route coverage at τ=0.9.\n")
    print("| Model | Calibration | c5 → c12 | e5 | e5m | e12 | Coverage part e5m−e5 | Task part e12−e5m |")
    print("|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        recs = load(model, BANK + CLINC)
        cal = Calibrator([r for r in recs if is_fit(r["qid"])])
        full = {u: rs for u, rs in by_utt(recs).items() if not is_fit(u + ":k5")}
        ev, keys = list(full.values()), [utt_key(u) for u in full]
        for m in ("deployed", "global_T", "indep_top"):
            a, b = rows(cal, m, ev, 5), rows(cal, m, ev, 12)

            def parts(idx):
                c5, e5 = cov_err(a, idx, .9)
                c12, e12 = cov_err(b, idx, .9)
                order = sorted(idx, key=lambda i: -a[i][1])
                n = max(int(round(c12 / 100 * len(idx))), 1)
                e5m = 100 * np.mean([not a[i][2] for i in order[:n]])
                return c5, c12, e5, e5m, e12

            c5, c12, e5, e5m, e12 = parts(np.arange(len(a)))
            cp = boot(keys, lambda i: (lambda p: p[3] - p[2])(parts(i)))
            tp = boot(keys, lambda i: (lambda p: p[4] - p[3])(parts(i)))
            e5s = "--" if np.isnan(e5) else f"{e5:.1f}"
            print(f"| {model} | {m} | {c5:.1f} → {c12:.1f} | {e5s} | {e5m:.1f} | {e12:.1f} | "
                  f"{fmt(cp) if not np.isnan(e5) else '--'} | {fmt(tp)} |")
    print()


# ---------------- null ----------------

def null():
    print("## Reference points for reordering (threshold reset to 50% for each option set; evaluation half)\n")
    print("v3: 500 utterances per intent set, five-option base set. Restricted: share among requests auto-handled in at least one of the two sets.\n")
    print("| Model | Calibration | Edit | Reordering | Route changes | Restricted reordering |")
    print("|---|---|---|---|---|---|")
    for model in MODELS:
        a2 = load(model, BANK + CLINC)
        cal = Calibrator([r for r in a2 if is_fit(r["qid"])])
        v3 = [r for r in load(model, BANK + CLINC, V3) if not is_fit(r["qid"])]
        keys = [utt_key(r["qid"]) for r in v3]
        for m in ("deployed", "global_T"):
            base = []
            for r in v3:
                d = max(r["keys"], key=r["base"]["scores"].get)
                base.append((d, cal.conf(m, r["base"]["scores"], r["keys"], r["base"]["temp"], d), d == r["anchor"]))
            for cond in ("permute", "add_1", "add_5"):
                ed, ok = [], True
                for r in v3:
                    c = next((c for c in r["conds"] if c["cond"] == cond), None)
                    if c is None:
                        ok = False
                        break
                    ks = list(c["scores"])
                    d = max(ks, key=c["scores"].get)
                    ed.append((d, cal.conf(m, c["scores"], ks, c["temp"], d), d == r["anchor"]))
                if not ok:
                    continue
                print(f"| {model} | {m} | {cond} | {fmt(boot(keys, lambda i: reorder(base, ed, i)))} | "
                      f"{100*np.mean([base[i][0] != ed[i][0] for i in range(len(base))]):.1f} | "
                      f"{fmt(boot(keys, lambda i: restricted(base, ed, i)))} |")
        # A2 5→12，限縮版
        full = {u: rs for u, rs in by_utt(a2).items() if not is_fit(u + ":k5")}
        ev, k2 = list(full.values()), [utt_key(u) for u in full]
        for m in ("deployed", "global_T"):
            a, b = rows(cal, m, ev, 5), rows(cal, m, ev, 12)
            print(f"| {model} | {m} | 5 → 12 routes (sweep) | {fmt(boot(k2, lambda i: reorder(a, b, i)))} | "
                  f"{100*np.mean([a[i][0] != b[i][0] for i in range(len(a))]):.1f} | {fmt(boot(k2, lambda i: restricted(a, b, i)))} |")
    print()


def restricted(a, b, idx):
    ta, tb = thr_for([a[i][1] for i in idx], .5), thr_for([b[i][1] for i in idx], .5)
    sel = [i for i in idx if a[i][1] >= ta or b[i][1] >= tb]
    return 100 * np.mean([a[i][0] != b[i][0] or (a[i][1] >= ta) != (b[i][1] >= tb) for i in sel])


# ---------------- launch ----------------

def launch():
    print("## Route launch with 95% intervals (evaluation half; thresholds for plug-in 5% chosen on the fitting half)\n")
    print("| Model | Calibration | New-route share | Policy | Auto-handled before → after | Error among auto before → after | New route correct |")
    print("|---|---|---|---|---|---|---|")
    for model in ["laya_td", "laya_base", "von", "qwen3_4b_set"]:
        a2_all = load(model, BANK + CLINC)
        la = load(model, BANK + CLINC, L)
        cal = Calibrator([r for r in a2_all if is_fit(r["qid"])])
        a2 = [r for r in a2_all if r["qid"].endswith(":k5")]
        fitq = lambda r: is_fit(r["qid"].replace(":launch", ""))

        def items(recs, after):
            out = []
            for r in recs:
                if after:
                    c = r["conds"][0]
                    s, k, t = c["scores"], list(c["scores"]), c["temp"]
                else:
                    s, k, t = r["base"]["scores"], r["keys"], r["base"]["temp"]
                d = max(k, key=s.get)
                out.append((cal.conf(m, s, k, t, d), d == r["anchor"]))
            return out

        for m in ("deployed", "global_T"):
            ex_f, nr_f = [r for r in a2 if fitq(r)], [r for r in la if fitq(r)]
            ex_e, nr_e = [r for r in a2 if not fitq(r)], [r for r in la if not fitq(r)]
            X = {"fb": items(ex_f, False), "fa": items(ex_f, True), "eb": items(ex_e, False), "ea": items(ex_e, True)}
            N = {"fb": items(nr_f, False), "fa": items(nr_f, True), "eb": items(nr_e, False), "ea": items(nr_e, True)}
            kx = [utt_key(r["qid"]) for r in ex_e]
            kn = [utt_key(r["qid"].replace(":launch", "")) for r in nr_e]
            for frac in (0.1, 0.3):
                rb = wrisk([(X["fb"], 1 - frac), (N["fb"], frac)], .05)
                ra = wrisk([(X["fa"], 1 - frac), (N["fa"], frac)], .05)

                def run(ix, inn, pol):
                    gb = [([X["eb"][i] for i in ix], 1 - frac), ([N["eb"][i] for i in inn], frac)]
                    ga = [([X["ea"][i] for i in ix], 1 - frac), ([N["ea"][i] for i in inn], frac)]
                    cb, erb = wstats(gb, .9 if pol != "plug" else rb)
                    if pol == "fixed":
                        ta = .9
                    elif pol == "reset":
                        ta = wquant(ga, cb / 100) if cb > 0 else 1.1
                    else:
                        ta = ra
                    ca, era = wstats(ga, ta)
                    newok = 100 * np.mean([(c >= ta) and ok for c, ok in ga[1][0]])
                    return cb, ca, erb, era, newok

                for pol in ("fixed", "reset", "plug"):
                    if m == "deployed" and pol == "plug":
                        continue
                    pt = run(range(len(kx)), range(len(kn)), pol)
                    reps = joint_boot(kx, kn, lambda ix, inn: run(ix, inn, pol))
                    ci = lambda j: (f"{pt[j]:.1f}" if not np.isnan(pt[j]) else "--") + (
                        f" [{np.nanpercentile(reps[:, j], 2.5):.1f}, {np.nanpercentile(reps[:, j], 97.5):.1f}]"
                        if not np.all(np.isnan(reps[:, j])) else "")
                    print(f"| {model} | {m} | {int(100*frac)}% | {pol} | {pt[0]:.1f} → {pt[1]:.1f} | {ci(2)} → {ci(3)} | {ci(4)} |")
    print()


def joint_boot(kx, kn, stat, n=300):
    """兩群流量依句子鍵一起重抽（同一句話的原有流量與新路由流量同進同出）。"""
    cx, cn = defaultdict(list), defaultdict(list)
    for i, k in enumerate(kx):
        cx[k].append(i)
    for i, k in enumerate(kn):
        cn[k].append(i)
    ks = sorted(set(cx) | set(cn))
    reps = []
    for _ in range(n):
        pick = [ks[j] for j in RNG.integers(0, len(ks), len(ks))]
        ix = [i for k in pick for i in cx.get(k, [])]
        inn = [i for k in pick for i in cn.get(k, [])]
        reps.append(stat(ix, inn))
    return np.asarray(reps, dtype=float)


def wstats(groups, thr):
    cov = sum(w * np.mean([c >= thr for c, _ in it]) for it, w in groups)
    err = sum(w * np.mean([(c >= thr) and not ok for c, ok in it]) for it, w in groups)
    return 100 * cov, (100 * err / cov if cov > 0 else np.nan)


def wquant(groups, share):
    pts = sorted([(c, w / len(it)) for it, w in groups for c, _ in it], key=lambda x: -x[0])
    acc = 0.0
    for c, w in pts:
        acc += w
        if acc >= share - 1e-12:
            return c
    return pts[-1][0]


def wrisk(groups, r):
    pts = sorted([(c, w / len(it), ok) for it, w in groups for c, ok in it], key=lambda x: -x[0])
    best, cw, ew = pts[0][0] + 1e-9, 0.0, 0.0
    for c, w, ok in pts:
        cw += w
        ew += w * (not ok)
        if ew / cw <= r:
            best = c
    return best


if __name__ == "__main__":
    jobs = {"ood": ood, "matched": matched, "null": null, "launch": launch}
    for j in sys.argv[1:] or list(jobs):
        jobs[j]()
