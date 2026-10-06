"""新路由上線模擬：一部分流量（比例 f）屬於新路由。
  原有流量：A2 的 K=5 → 6（正確答案一直在清單裡，新增的是干擾）
  新路由流量：launch.py 的 before（5 個干擾、無正確答案）→ after（加入正確答案）
校準在 A2 擬合半上擬合（與前面相同），評估在兩種流量的評估半。三種門檻政策：
  fixed   門檻固定 0.9
  reset   上線後重設門檻，使整體自動處理比例等於上線前（不需標籤）
  risk5   每個狀態在擬合半上選使錯誤率 ≤ 5% 的最低門檻（需要標籤；上線後的擬合資料也含新路由流量）
報告上線前／後：整體自動處理比例、自動處理中的錯誤率、新路由流量被正確自動處理的比例。
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calib_methods import A2, TASKS, Calibrator, is_fit

L = Path("launch_results")
MODELS = ["laya_td", "laya_base", "von", "qwen3_4b_set"]
FRACS = (0.1, 0.3)


def items(cal, m, recs, use_cond):
    out = []
    for r in recs:
        if use_cond:
            c = r["conds"][0]
            s, k, t = c["scores"], list(c["scores"]), c["temp"]
        else:
            s, k, t = r["base"]["scores"], r["keys"], r["base"]["temp"]
        d = max(k, key=s.get)
        out.append((cal.conf(m, s, k, t, d), d == r["anchor"]))
    return out


def wquant_thr(groups, share):
    """groups: [(items, weight)]；回傳使加權自動處理比例 ≈ share 的門檻。"""
    pts = sorted([(c, w / len(it)) for it, w in groups for c, _ in it], key=lambda x: -x[0])
    acc = 0.0
    for c, w in pts:
        acc += w
        if acc >= share:
            return c
    return pts[-1][0]


def wrisk_thr(groups, r):
    pts = sorted([(c, w / len(it), ok) for it, w in groups for c, ok in it], key=lambda x: -x[0])
    best, cw, ew = pts[0][0] + 1e-9, 0.0, 0.0
    for c, w, ok in pts:
        cw += w
        ew += w * (not ok)
        if ew / cw <= r:
            best = c
    return best


def stats(groups, thr):
    cov = sum(w * np.mean([c >= thr for c, _ in it]) for it, w in groups)
    err_w = sum(w * np.mean([(c >= thr) and not ok for c, ok in it]) for it, w in groups)
    return 100 * cov, (100 * err_w / cov if cov > 0 else float("nan"))


def main():
    print("| Model | Calibration | New-route share | Policy | Auto-handled before → after | Error among auto before → after | New-route traffic correctly auto-handled after |")
    print("|---|---|---|---|---|---|---|")
    for model in MODELS:
        a2 = [r for t in TASKS if (A2 / f"{t}__{model}.json").exists()
              for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"] if r["K"] == 5]
        la = [r for t in TASKS if (L / f"{t}__{model}.json").exists()
              for r in json.loads((L / f"{t}__{model}.json").read_text())["records"]]
        if not la:
            continue
        a2_all = [r for t in TASKS if (A2 / f"{t}__{model}.json").exists()
                  for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]
        cal = Calibrator([r for r in a2_all if is_fit(r["qid"])])
        split = lambda recs, fit: [r for r in recs if is_fit(r["qid"].replace(":launch", "")) == fit]
        for m in ("deployed", "global_T"):
            ex = {f: (items(cal, m, split(a2, f), False), items(cal, m, split(a2, f), True)) for f in (True, False)}
            nr = {f: (items(cal, m, split(la, f), False), items(cal, m, split(la, f), True)) for f in (True, False)}
            for frac in FRACS:
                before = lambda f: [(ex[f][0], 1 - frac), (nr[f][0], frac)]
                after = lambda f: [(ex[f][1], 1 - frac), (nr[f][1], frac)]
                t_fixed = 0.9
                cov_b, _ = stats(before(False), t_fixed)
                for pol in ("fixed", "reset", "risk5"):
                    if pol == "fixed":
                        tb, ta = t_fixed, t_fixed
                    elif pol == "reset":
                        tb = t_fixed
                        ta = wquant_thr(after(False), cov_b / 100)
                    else:
                        tb, ta = wrisk_thr(before(True), 0.05), wrisk_thr(after(True), 0.05)
                    cb, eb = stats(before(False), tb)
                    ca, ea = stats(after(False), ta)
                    newok = 100 * np.mean([(c >= ta) and ok for c, ok in nr[False][1]])
                    print(f"| {model} | {m} | {int(100*frac)}% | {pol} | {cb:.1f} → {ca:.1f} | {eb:.1f} → {ea:.1f} | {newok:.1f} |")


if __name__ == "__main__":
    main()
