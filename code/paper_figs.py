"""論文版圖 1～3：英文標籤、PDF 向量格式，配色沿用已驗證的分類色序。

圖 1（A4 部署模擬）加 95% 信賴區間：依句子重抽 1,000 次（每句話 K=5..12 一起抽）。
圖 2（A2 劑量反應）、圖 3（取捨圖）只改語言與格式，數值和中文版相同。
用法：python paper_figs.py <輸出資料夾>
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import a4_deploy
import analyze_a2
from calib_methods import Calibrator, is_fit, utt_key

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "paper_figs")
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
from matplotlib import font_manager
import os
_NOTO = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"  # 論文圖用 Noto Sans；沒有就用預設字型
if os.path.exists(_NOTO):
    font_manager.fontManager.addfont(_NOTO)
analyze_a2.ROOT = Path("a2_results")  # analyze_a2 預設把 argv[1] 當資料夾，這裡改回結果資料夾
plt.rcParams.update({"font.family": "Noto Sans" if os.path.exists(_NOTO) else "sans-serif", "axes.unicode_minus": True, "pdf.fonttype": 42,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
N_BOOT = 1000


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8)


# ---------------- 圖 1：部署模擬（含信賴區間） ----------------

FIG1_MODELS = [("laya_td", "Laya, fine-tuned"), ("von", "Von")]
FIG1_METHODS = [("deployed", "Shipped calibration", "#2a78d6", "o"),
                ("global_T", "Single temperature (refit)", "#eb6834", "s"),
                ("indep_top", "Top-score calibration", "#1baf7a", "^")]
KS = a4_deploy.KS


def simulate_per_utt(model, gate=None, coverage=None):
    """回傳每句話、每個 K 的「路由改變」與各方法「處理改變」布林矩陣（句子 × K）。
    gate：固定門檻（預設 a4_deploy.GATE）；coverage：改用各方法在 K=5 時自動處理該比例的門檻。"""
    A2 = a4_deploy.A2
    recs = [r for t in a4_deploy.TASKS if (A2 / f"{t}__{model}.json").exists()
            for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]
    cal = Calibrator([r for r in recs if is_fit(r["qid"])])
    by_utt = defaultdict(dict)
    for r in recs:
        if not is_fit(r["qid"]):
            utt, k = r["qid"].rsplit(":k", 1)
            by_utt[utt][int(k)] = r
    route, handling, keys = [], {m: [] for m, *_ in FIG1_METHODS}, []
    full = [rs for rs in by_utt.values() if all(k in rs for k in KS)]
    conf5 = {m: [cal.conf(m, rs[5]["base"]["scores"], rs[5]["keys"], rs[5]["base"]["temp"],
                          max(rs[5]["keys"], key=rs[5]["base"]["scores"].get)) for rs in full]
             for m, *_ in FIG1_METHODS}
    if coverage is not None:
        thr = {m: float(np.sort(v)[::-1][int(coverage * len(v)) - 1]) for m, v in conf5.items()}
    else:
        thr = {m: (gate if gate is not None else a4_deploy.GATE) for m in conf5}
    for utt, rs in by_utt.items():
        if not all(k in rs for k in KS):
            continue
        keys.append(utt_key(utt))
        r5 = rs[5]
        dec5 = max(r5["keys"], key=r5["base"]["scores"].get)
        auto5 = {m: cal.conf(m, r5["base"]["scores"], r5["keys"], r5["base"]["temp"], dec5) >= thr[m]
                 for m, *_ in FIG1_METHODS}
        decs = {k: max(rs[k]["keys"], key=rs[k]["base"]["scores"].get) for k in KS}
        route.append([decs[k] != dec5 for k in KS])
        for m, *_ in FIG1_METHODS:
            row = []
            for k in KS:
                r = rs[k]
                auto = cal.conf(m, r["base"]["scores"], r["keys"], r["base"]["temp"], decs[k]) >= thr[m]
                row.append(auto != auto5[m] or decs[k] != dec5)
            handling[m].append(row)
    return np.asarray(route, float), {m: np.asarray(v, float) for m, v in handling.items()}, keys


def boot_ci(mat, rng, keys):
    """句子層級重抽（隨機版與相近版的同一句話同進同出）；回傳 (平均, 2.5%, 97.5%)，各為長度 len(KS) 的陣列（百分比）。"""
    rows = defaultdict(list)
    for i, k in enumerate(keys):
        rows[k].append(i)
    clusters = list(rows.values())
    means = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(clusters), len(clusters))
        idx = [i for c in pick for i in clusters[c]]
        means.append(mat[idx].mean(axis=0))
    means = np.asarray(means)
    return 100 * mat.mean(0), 100 * np.percentile(means, 2.5, 0), 100 * np.percentile(means, 97.5, 0)


def fig1():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    print("# Figure 1 values (%, 95% CI)\n")
    for ax, (model, title) in zip(axes, FIG1_MODELS):
        route, handling, ukeys = simulate_per_utt(model)
        print(f"## {title} (n = {len(route)} utterances)\n")
        print("| | " + " | ".join(f"K={k}" for k in KS) + " |\n|---|" + "---|" * len(KS))
        style_axes(ax)
        series = [("route", "Route changes only", MUTED, None, route)] + \
                 [(m, lab, c, mk, handling[m]) for m, lab, c, mk in FIG1_METHODS]
        for key, label, color, marker, mat in series:
            mean, lo, hi = boot_ci(mat, rng, ukeys)
            print(f"| {label} | " + " | ".join(f"{m_:.1f} [{l:.1f}, {h:.1f}]" for m_, l, h in zip(mean, lo, hi)) + " |")
            ax.fill_between(KS, lo, hi, color=color, alpha=0.15, linewidth=0)
            if marker is None:
                ax.plot(KS, mean, color=color, linewidth=1.2, linestyle=(0, (3, 2)), label=label)
            else:
                ax.plot(KS, mean, color=color, linewidth=1.5, marker=marker, markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1, label=label)
        print()
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xticks(KS)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Routes in the table", fontsize=9, color=INK2)
    axes[0].set_ylabel("Requests handled differently\nthan with 5 routes (%)", fontsize=9, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.06), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_deploy.pdf", bbox_inches="tight")


def fig_a1():
    """附錄：圖 1 改用 τ = 0.7 與 50% 覆蓋率（各方法各自設門檻）。"""
    rng = np.random.default_rng(0)
    settings = [("$\\tau$ = 0.7", dict(gate=0.7)), ("50% coverage", dict(coverage=0.5))]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharey=True, sharex=True)
    print("# Figure A1 values (%, 95% CI)\n")
    for row, (sname, kw) in enumerate(settings):
        for col, (model, title) in enumerate(FIG1_MODELS):
            ax = axes[row, col]
            route, handling, ukeys = simulate_per_utt(model, **kw)
            style_axes(ax)
            print(f"## {title}, {sname}\n")
            series = [("route", "Route changes only", MUTED, None, route)] + \
                     [(m, lab, c, mk, handling[m]) for m, lab, c, mk in FIG1_METHODS]
            for key, label, color, marker, mat in series:
                mean, lo, hi = boot_ci(mat, rng, ukeys)
                print(f"- {label}: K=6 {mean[1]:.1f} [{lo[1]:.1f}, {hi[1]:.1f}]; K=12 {mean[-1]:.1f} [{lo[-1]:.1f}, {hi[-1]:.1f}]")
                ax.fill_between(KS, lo, hi, color=color, alpha=0.15, linewidth=0)
                if marker is None:
                    ax.plot(KS, mean, color=color, linewidth=1.2, linestyle=(0, (3, 2)), label=label)
                else:
                    ax.plot(KS, mean, color=color, linewidth=1.5, marker=marker, markersize=4.5,
                            markeredgecolor=SURFACE, markeredgewidth=1, label=label)
            print()
            ax.set_title(f"{title}, {sname}", fontsize=10, color=INK, loc="left")
            ax.set_xticks(KS)
            ax.set_ylim(0, 100)
            if row == 1:
                ax.set_xlabel("Routes in the table", fontsize=9, color=INK2)
        axes[row, 0].set_ylabel("Requests handled differently\nthan with 5 routes (%)", fontsize=9, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.04), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "figA1_deploy_thresholds.pdf", bbox_inches="tight")


def fig1v3():
    """v3 圖 1：兩層。欄：(a) 門檻 0.9 的自動處理比例、(b) 自動處理中的錯誤率、(c) 每個 K 重設門檻（自動處理 50%）後的處理改變。
    數值來自 layers.py 的 layers_results.json（依句子重抽的 95% 區間）。"""
    res = json.load(open("layers_results.json"))
    cols = [("coverage", "Auto-handled at $\\tau$ = 0.9 (%)", (0, 100)),
            ("err_auto", "Error among auto-handled (%)", (0, 20)),
            ("reorder", "Handled differently than with\n5 routes, threshold reset (%)", (0, 50))]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True)
    for row, (model, title) in enumerate(FIG1_MODELS):
        route, _, _ = simulate_per_utt(model)
        for col, (key, ylabel, ylim) in enumerate(cols):
            ax = axes[row, col]
            style_axes(ax)
            for m, label, color, marker in FIG1_METHODS:
                d = res[model][m]
                ys = np.array([d[str(k)][key] for k in KS], dtype=float)
                ok = ~np.isnan(ys[:, 0])
                ks = np.array(KS)[ok]
                ax.fill_between(ks, ys[ok, 1], ys[ok, 2], color=color, alpha=0.15, linewidth=0)
                ax.plot(ks, ys[ok, 0], color=color, linewidth=1.5, marker=marker, markersize=4,
                        markeredgecolor=SURFACE, markeredgewidth=1, label=label)
            if key == "reorder":
                ax.plot(KS, 100 * route.mean(0), color=MUTED, linewidth=1.2, linestyle=(0, (3, 2)),
                        label="Route changes only")
            ax.set_ylim(*ylim)
            ax.set_xticks(KS)
            if row == 0:
                ax.set_title(ylabel, fontsize=9, color=INK, loc="left")
            if col == 0:
                ax.set_ylabel(title, fontsize=10, color=INK)
            if row == 1:
                ax.set_xlabel("Routes in the table", fontsize=9, color=INK2)
    h1, l1 = axes[0, 2].get_legend_handles_labels()
    fig.legend(h1, l1, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.05), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_layers.pdf", bbox_inches="tight")


# ---------------- 圖 2：劑量反應 ----------------

TASK_NAMES = {"banking": "Banking77, random", "banking_hard": "Banking77, near",
              "clinc": "CLINC150, random", "clinc_hard": "CLINC150, near"}
MODEL_NAMES = {"laya_td": "Laya, fine-tuned", "von": "Von", "laya_multi": "Laya, multilingual",
               "laya_base": "Laya, base", "qwen3_4b_point": "Qwen3-4B, one at a time",
               "qwen3_4b_set": "Qwen3-4B, joint", "jev": "Jev"}


def fig2():
    data = analyze_a2.load()
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.8), sharex=True, sharey=True)
    for ax, task in zip(axes.flat, analyze_a2.TASKS):
        style_axes(ax)
        ax.set_ylim(0, 100)
        for x in analyze_a2.CROSSINGS:
            ax.axvspan(x - 0.35, x + 0.35, color=GRID, alpha=0.55, linewidth=0, zorder=0)
        for m, _, color, marker in analyze_a2.MODELS:
            if (task, m) not in data:
                continue
            ks = sorted(data[(task, m)])
            ys = [100 * st.mean(data[(task, m)][k]["gate0.9"]) for k in ks]
            ax.plot(ks, ys, color=color, linewidth=1.5, marker=marker, markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=MODEL_NAMES[m])
        ax.set_title(TASK_NAMES[task], fontsize=10, color=INK, loc="left")
        ax.set_xlim(1.5, 12.5)
        ax.set_xticks(range(2, 13))
    for ax in axes[1]:
        ax.set_xlabel("Options before the edit, K", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Gate flips at τ = 0.9 (%)", fontsize=9, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.07), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_dose.pdf", bbox_inches="tight")


# ---------------- 圖 3：鑑別力與穩定性的取捨 ----------------

FAMILIES = [("qwen3_4b", "Qwen3-4B"), ("qwen25_7b", "Qwen2.5-7B"), ("phi4mini", "Phi-4-mini"),
            ("mistral7b", "Mistral-7B")]
FIG3_METHODS = [("set_T", "Joint, single temperature", "#2a78d6", "o"),
                ("point_IT", "One-at-a-time, top score", "#eb6834", "s"),
                ("hybrid", "Joint decision, independent score", "#1baf7a", "^"),
                ("hybrid_margin", "Joint decision, independent margin", "#eda100", "D"),
                ("learned_indep", "Learned, independent features", "#e87ba4", "P")]


def load_t7(paths=("tradeoff_rt.md",)):
    """ci_tables.py --t7 的輸出（選擇性準確率已按比例處理同分）→ {家族: {方法: (準確率, 處理改變)}}。"""
    out = defaultdict(dict)
    for line in (ln for path in paths for ln in open(path)):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 4 and c[0] in dict(FAMILIES):
            out[c[0]][c[1]] = (float(c[2].split()[0]), float(c[3].split()[0]))
    return out


def fig3():
    res = load_t7()
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.0), sharey=True)
    for ax, (fam, title) in zip(axes, FAMILIES):
        v3 = res[fam]
        style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        for m, label, color, marker in FIG3_METHODS:
            ax.scatter([v3[m][1]], [v3[m][0]], s=55, color=color, marker=marker,
                       edgecolors=SURFACE, linewidths=1.2, label=label, zorder=3)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, 40)
        ax.set_ylim(65, 100)
        ax.set_xlabel("Reordering after +5 options (%)", fontsize=8.5, color=INK2)
    axes[0].set_ylabel("Selective accuracy\nat 50% coverage (%)", fontsize=8.5, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.08), labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_tradeoff.pdf", bbox_inches="tight")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = sys.argv[2:] or ["fig1", "fig2", "fig3", "fig_a1"]
    for j in jobs:
        globals()[j]()
    print("written:", *sorted(p.name for p in OUT.glob("*.pdf")))
