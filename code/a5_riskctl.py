"""每個 K 的風險控制門檻（有限樣本保證）：在擬合半上，對 20 個候選門檻（擬合半信心的 5%..100% 分位）
做二項檢定 H0: 自動處理中的錯誤率 > α，p = P(Binom(n_acc, α) ≤ 錯誤數)，固定順序檢定（見 ltt）。
（以自動處理數為條件的檢定，同 Learn-then-Test 的二項 p 值寫法。）與其他政策比較：
  fixed_ship  出貨校準、門檻 0.9；fixed_single 單一溫度、門檻 0.9；
  reuse5      在 K=5 上做風險控制，之後不重新認證，沿用同一個門檻；
  ltt_K       每個 K 重新做風險控制；oracle 評估半上的可達上限。
報告評估半的自動處理比例與錯誤率。α = 5%，δ = 10%。"""
import numpy as np
from scipy.stats import binom
from a5_analysis import MODELS, FAMS, load, by_utt, rows
from calib_methods import Calibrator, is_fit
from a5_frontier import frontier

ALPHA, DELTA, GRID = 0.05, 0.10, 20

def ltt(fit, n_min=100):
    """固定順序檢定（Learn-then-Test）：候選門檻依信心由高到低排列，只保留至少 n_min 筆被接受者
    （只用信心、不用標籤決定，不影響保證）；逐一以 δ 檢定，第一次不拒絕即停，取最後通過者。"""
    c = np.array([t[1] for t in fit]); ok = np.array([t[2] for t in fit])
    best = np.inf
    for q in np.linspace(0.05, 1.0, GRID):
        lam = np.quantile(c, 1 - q)
        acc = c >= lam; n = acc.sum(); e = (~ok[acc]).sum()
        if n < n_min:
            continue
        if binom.cdf(e, n, ALPHA) <= DELTA:
            best = lam
        else:
            break
    return best

def apply(x, lam):
    a = [t for t in x if t[1] >= lam]
    return 100 * len(a) / len(x), (100 * np.mean([not t[2] for t in a]) if a else np.nan)

def f(v):
    return "--" if np.isnan(v) else f"{v:.0f}"

for fam, tasks in FAMS.items():
    print(f"## {fam}\n\n| Model | K | Oracle max auto | LTT per K: auto / err | Reuse K=5 LTT: auto / err | Fixed 0.9 single: auto / err | Fixed 0.9 shipped: auto / err |\n|---|---|---|---|---|---|---|")
    for model, title in MODELS:
        recs = load(model, tasks); cal = Calibrator([r for r in recs if is_fit(r["qid"])])
        ks = sorted({r["K"] for r in recs}); bu = by_utt(recs)
        fit = [rs for u, rs in bu.items() if is_fit(u + ":k5") and all(k in rs for k in ks)]
        ev = [rs for u, rs in bu.items() if not is_fit(u + ":k5") and all(k in rs for k in ks)]
        lam5 = ltt(rows(cal, "global_T", fit, 5))
        for k in ks:
            fk, ek = rows(cal, "global_T", fit, k), rows(cal, "global_T", ev, k)
            dk = rows(cal, "deployed", ev, k)
            cells = [apply(ek, ltt(fk)), apply(ek, lam5), apply(ek, .9), apply(dk, .9)]
            print(f"| {title} | {k} | {frontier(ek, ALPHA):.0f} | " + " | ".join(f"{f(a)} / {f(e)}" for a, e in cells) + " |")
    print()
