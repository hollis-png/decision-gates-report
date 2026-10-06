"""A2 分析：門檻翻轉率與三因素貢獻隨基準選項數 K 的變化；輸出 Markdown 表格與兩張圖。

圖一：門檻 0.9 翻轉率 vs K，四份資料集各一格，每個模型一條線（末端直接標名稱、不同標記形狀）。
圖二：Laya 微調與 Von 的信心變化三因素（S／D／T）vs K，四份資料集平均。
配色取自 dataviz 預設分類色序前 6 色（已用 validate_palette.js 驗證；對比度 WARN 以直接標籤＋表格補足）。
"""
import os
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_a3 import decompose

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "a2_results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
TASKS = ["banking", "banking_hard", "clinc", "clinc_hard"]
TASK_NAMES = {"banking": "Banking77（隨機干擾）", "banking_hard": "Banking77（相近干擾）",
              "clinc": "CLINC150（隨機干擾）", "clinc_hard": "CLINC150（相近干擾）"}
MODELS = [("laya_td", "Laya 微調", "#2a78d6", "o"), ("von", "Von", "#eb6834", "s"),
          ("laya_multi", "Laya 多語", "#1baf7a", "^"), ("laya_base", "Laya 起點", "#eda100", "D"),
          ("qwen3_4b_point", "Qwen 逐一式", "#e87ba4", "v"), ("qwen3_4b_set", "Qwen 集合式", "#008300", "P"),
          ("jev", "Jev（閉源）", "#4a3aa7", "X")]  # Jev 只有四捨五入後的機率：只用於門檻圖，不做三因素拆解
FACTOR_STYLE = {"S": ("S 原有選項分數改變", "#2a78d6", "o"), "D": ("D 新選項稀釋", "#eb6834", "s"),
                "T": ("T 溫度改變", "#1baf7a", "^")}
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CROSSINGS = (2, 5, 10)  # 基準 K 加 1 個後會跨越 Laya 溫度分組邊界（2 | 3–5 | 6–10 | 11+）的點


def mark_crossings(ax, label=False):
    for x in CROSSINGS:
        ax.axvspan(x - 0.35, x + 0.35, color=GRID, alpha=0.55, linewidth=0, zorder=0)
    if label:
        ax.text(5, ax.get_ylim()[1], "加 1 個後跨越 Laya 溫度分組邊界", fontsize=7.5, color=MUTED,
                ha="center", va="bottom")


def end_labels(ax, items, min_gap_frac=0.045):
    """線尾直接標籤；依 y 排序後強制最小間距，避免重疊。items = [(x, y, text)]。"""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    placed = []
    for x, y, text in sorted(items, key=lambda t: t[1]):
        y_lab = max(y, placed[-1] + gap) if placed else y
        placed.append(y_lab)
        ax.annotate(text, (x, y), xytext=(x + 0.25, y_lab), textcoords="data", va="center",
                    fontsize=7.5, color=INK2)

from matplotlib import font_manager
_CJK = os.environ.get("CJK_FONT", "")  # 中文圖表字型（.ttc）路徑；未設定則用 matplotlib 預設字型  # Noto Sans CJK 是 .ttc，matplotlib 不會自動找到，手動註冊
if _CJK and os.path.exists(_CJK):
    font_manager.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_CJK).get_name()
plt.rcParams.update({ "axes.unicode_minus": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def load():
    """(資料集, 模型) → K → 指標串列。"""
    out = {}
    for task in TASKS:
        for m, *_ in MODELS:
            f = ROOT / f"{task}__{m}.json"
            if not f.exists():
                continue
            by_k = defaultdict(lambda: defaultdict(list))
            for rec in json.loads(f.read_text())["records"]:
                d = decompose(rec, rec["conds"][0])
                if d:
                    for k, v in d.items():
                        by_k[rec["K"]][k].append(v)
                    by_k[rec["K"]]["temp_base"].append(rec["base"]["temp"])
                    by_k[rec["K"]]["temp_new"].append(rec["conds"][0]["temp"])
            out[(task, m)] = by_k
    return out


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig_gate(data):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, sharey=True)
    for ax, task in zip(axes.flat, TASKS):
        style_axes(ax)
        ax.set_ylim(0, 100)
        mark_crossings(ax, label=(task == TASKS[0]))
        ends = []
        for m, label, color, marker in MODELS:
            if (task, m) not in data:
                continue
            ks = sorted(data[(task, m)])
            ys = [100 * st.mean(data[(task, m)][k]["gate0.9"]) for k in ks]
            ax.plot(ks, ys, color=color, linewidth=1.5, marker=marker, markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=label)
            ends.append((ks[-1], ys[-1], label))
        if len(ends) <= 4:  # 超過 4 條線時只用圖例；錯開的線尾標籤會離線太遠、對不上
            end_labels(ax, ends)
        ax.set_title(TASK_NAMES[task], fontsize=10, color=INK, loc="left", pad=14)
        ax.set_xlim(1.5, 12.5 if len(ends) > 4 else 14.2)
        ax.set_xticks(range(2, 13))
    for ax in axes[1]:
        ax.set_xlabel("基準選項數 K（再加 1 個）", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("信心跨過 0.9 門檻的比例（%）", fontsize=9, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK2)
    fig.suptitle("加 1 個選項後，決策信心跨過 0.9 門檻的比例", fontsize=12, color=INK, y=1.04)
    fig.tight_layout()
    path = OUT / "a2_gate_vs_K.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def fig_factors(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (m, title) in zip(axes, [("laya_td", "Laya 微調（溫度依選項數分組）"),
                                     ("von", "Von（溫度為選項數的連續函數）")]):
        style_axes(ax)
        ax.axhline(0, color=BASE, linewidth=1)
        ends = []
        for f, (label, color, marker) in FACTOR_STYLE.items():
            ks = sorted({k for t in TASKS if (t, m) in data for k in data[(t, m)]})
            ys = [100 * st.mean(v for t in TASKS if (t, m) in data for v in data[(t, m)][k][f"shap_{f}"])
                  for k in ks]
            ax.plot(ks, ys, color=color, linewidth=1.5, marker=marker, markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=label)
            ends.append((ks[-1], ys[-1], f))
        ax.set_ylim(-12, 20)
        mark_crossings(ax, label=(m == "laya_td"))
        end_labels(ax, ends, min_gap_frac=0.06)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=14)
        ax.set_xticks(range(2, 13))
        ax.set_xlim(1.5, 13.4)
        ax.set_xlabel("基準選項數 K（再加 1 個）", fontsize=9, color=INK2)
    axes[0].set_ylabel("對決策信心的平均貢獻（百分點）", fontsize=9, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.02), labelcolor=INK2)
    fig.suptitle("加 1 個選項時，信心變化的三個來源（四份資料集平均）", fontsize=12, color=INK, y=1.09)
    fig.tight_layout()
    path = OUT / "a2_factors_vs_K.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def tables(data):
    print("### 門檻 0.9 翻轉率（%，四份資料集平均）：總計 / 只改溫度\n")
    ks = list(range(2, 13))
    print("| 模型 | " + " | ".join(f"K={k}" for k in ks) + " |")
    print("|---|" + "---|" * len(ks))
    for m, label, *_ in MODELS:
        cells = []
        for k in ks:
            tot = [v for t in TASKS if (t, m) in data for v in data[(t, m)][k]["gate0.9"]]
            onl = [v for t in TASKS if (t, m) in data for v in data[(t, m)][k]["gate0.9_only_T"]]
            cells.append(f"{100*st.mean(tot):.0f} / {100*st.mean(onl):.0f}" if tot else "—")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\n### 平均溫度：基準 → 加 1 個選項後\n")
    print("| 模型 | " + " | ".join(f"K={k}" for k in ks) + " |")
    print("|---|" + "---|" * len(ks))
    for m, label, *_ in MODELS[:2]:
        cells = []
        for k in ks:
            tb = [v for t in TASKS if (t, m) in data for v in data[(t, m)][k]["temp_base"]]
            tn = [v for t in TASKS if (t, m) in data for v in data[(t, m)][k]["temp_new"]]
            cells.append(f"{st.mean(tb):.2f}→{st.mean(tn):.2f}" if tb else "—")
        print(f"| {label} | " + " | ".join(cells) + " |")


def main():
    data = load()
    print(f"# A2 劑量反應（已完成 {len(data)} / {len(TASKS) * len(MODELS)} 組）\n")
    tables(data)
    print(f"\n圖：{fig_gate(data)}、{fig_factors(data)}")


if __name__ == "__main__":
    main()
