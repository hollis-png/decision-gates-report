"""技術報告圖 4：錯誤率 ≤5% 時的可達自動處理比例（上限）隨選項數的變化，單一溫度信心排序，評估半；
陰影為依句子重抽 300 次的 95% 區間。Laya 在 12 個選項之後改用虛線（超出出貨的選項 token 預算）。"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_figs import style_axes, INK, INK2, MUTED, SURFACE
from a5_analysis import MODELS, FAMS, load, by_utt, rows
from a5_frontier import frontier
from calib_methods import Calibrator, is_fit, utt_key

OUT = Path(sys.argv[1])
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, (fam, tasks) in zip(axes, FAMS.items()):
    style_axes(ax)
    for (model, title), col in zip(MODELS, COLORS):
        recs = load(model, tasks); cal = Calibrator([r for r in recs if is_fit(r["qid"])])
        ks = sorted({r["K"] for r in recs}); bu = by_utt(recs)
        us = [u for u, rs in bu.items() if not is_fit(u + ":k5") and all(k in rs for k in ks)]
        ev = [bu[u] for u in us]; keys = [utt_key(u) for u in us]
        cl = {}
        for i, k in enumerate(keys):
            cl.setdefault(k, []).append(i)
        cl = list(cl.values())
        pts, lo, hi = [], [], []
        for k in ks:
            x = rows(cal, "global_T", ev, k)
            pts.append(frontier(x, .05))
            reps = []
            for _ in range(300):
                idx = np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))])
                reps.append(frontier([x[i] for i in idx], .05))
            lo.append(np.percentile(reps, 2.5)); hi.append(np.percentile(reps, 97.5))
        ks_a = np.array(ks)
        ax.fill_between(ks_a, lo, hi, color=col, alpha=0.12, linewidth=0)
        if model.startswith("laya"):
            m = ks_a <= 12
            ax.plot(ks_a[m], np.array(pts)[m], color=col, lw=1.6, marker="o", ms=3.5, label=title)
            m2 = ks_a >= 12
            ax.plot(ks_a[m2], np.array(pts)[m2], color=col, lw=1.4, ls=(0, (3, 2)), marker="o", ms=3.5)
        else:
            ax.plot(ks_a, pts, color=col, lw=1.6, marker="o", ms=3.5, label=title)
    ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks], fontsize=7)
    ax.minorticks_off()
    ax.set_title(fam, fontsize=9, color=INK, loc="left")
    ax.set_xlabel("Options in the set", fontsize=9, color=INK2)
    ax.set_ylim(0, 100)
axes[0].set_ylabel("Max. auto-handled at 5% error (%)", fontsize=9, color=INK)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06), labelcolor=INK2)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig4_ceiling.pdf", bbox_inches="tight")
print("ok")
