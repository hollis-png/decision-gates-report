"""A4 部署模擬：路由表從 5 個路由逐步擴充到 12 個，每一步有多少請求的處理方式和最初不同。

利用 A2 的巢狀清單：K 個選項的基準清單 = 正確答案 + 干擾鏈前 K−1 個，所以 K=5,6,...,12 就是同一張路由表逐步加路由。
每一步記錄兩件事（相對 K=5，評估半的句子）：
  路由改變   系統輸出的第一名換人（不論校準方式都一樣）
  處理改變   自動處理（信心 >= 0.9）／轉人工 的狀態和 K=5 時不同
校準方式：deployed、global_T、indep_top（擬合方式同 calib_methods.py）。
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
from matplotlib import font_manager

from calib_methods import A2, TASKS, Calibrator, is_fit

GATE = 0.9
MODELS = [("laya_td", "Laya 微調"), ("von", "Von")]
METHODS = [("deployed", "官方部署的校準", "#2a78d6", "o"), ("global_T", "單一溫度（重新擬合）", "#eb6834", "s"),
           ("indep_top", "獨立第一名（不正規化）", "#1baf7a", "^")]
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
_CJK = os.environ.get("CJK_FONT", "")  # 中文圖表字型（.ttc）路徑；未設定則用 matplotlib 預設字型
if _CJK and os.path.exists(_CJK):
    font_manager.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_CJK).get_name()
plt.rcParams.update({
                     "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
KS = list(range(5, 13))


def simulate(model):
    recs = [r for t in TASKS if (A2 / f"{t}__{model}.json").exists()
            for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]
    cal = Calibrator([r for r in recs if is_fit(r["qid"])])
    by_utt = defaultdict(dict)
    for r in recs:
        if not is_fit(r["qid"]):
            utt, k = r["qid"].rsplit(":k", 1)
            by_utt[utt][int(k)] = r
    route, handling = defaultdict(list), {m: defaultdict(list) for m, *_ in METHODS}
    for utt, rs in by_utt.items():
        if not all(k in rs for k in KS):
            continue
        r5 = rs[5]
        dec5 = max(r5["keys"], key=r5["base"]["scores"].get)
        auto5 = {m: cal.conf(m, r5["base"]["scores"], r5["keys"], r5["base"]["temp"], dec5) >= GATE
                 for m, *_ in METHODS}
        for k in KS:
            r = rs[k]
            dec = max(r["keys"], key=r["base"]["scores"].get)
            route[k].append(dec != dec5)
            for m, *_ in METHODS:
                auto = cal.conf(m, r["base"]["scores"], r["keys"], r["base"]["temp"], dec) >= GATE
                handling[m][k].append(auto != auto5[m] or dec != dec5)  # 路由或處理方式任一改變
    return ({k: 100 * st.mean(v) for k, v in route.items()},
            {m: {k: 100 * st.mean(v) for k, v in d.items()} for m, d in handling.items()}, len(by_utt))


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    print("# A4 部署模擬：路由表從 5 個擴充到 12 個（門檻 0.9）\n")
    print("數值為「和 5 個路由時相比，路由或處理方式改變的請求比例（%）」；第一列為只看路由改變。\n")
    for ax, (model, title) in zip(axes, MODELS):
        route, handling, n = simulate(model)
        print(f"\n## {title}（評估句子 {n} 句）\n")
        print("| 方式 | " + " | ".join(f"{k} 個路由" for k in KS) + " |")
        print("|---|" + "---|" * len(KS))
        print("| 只看路由改變 | " + " | ".join(f"{route[k]:.1f}" for k in KS) + " |")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(BASE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.plot(KS, [route[k] for k in KS], color=MUTED, linewidth=1.2, linestyle=(0, (3, 2)),
                label="只看路由改變（與校準無關）")
        for m, label, color, marker in METHODS:
            ys = [handling[m][k] for k in KS]
            print(f"| {label} | " + " | ".join(f"{y:.1f}" for y in ys) + " |")
            ax.plot(KS, ys, color=color, linewidth=1.5, marker=marker, markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=label)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xticks(KS)
        ax.set_xlabel("路由表中的路由數", fontsize=9, color=INK2)
    axes[0].set_ylabel("處理方式和 5 個路由時不同的請求（%）", fontsize=9, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.03), labelcolor=INK2)
    fig.suptitle("路由表擴充時，自動處理（信心 ≥ 0.9）／轉人工的決定有多少改變", fontsize=12, color=INK, y=1.1)
    fig.tight_layout()
    fig.savefig("a4_deploy.png", dpi=160, bbox_inches="tight")
    print("\n圖：a4_deploy.png")


if __name__ == "__main__":
    main()
