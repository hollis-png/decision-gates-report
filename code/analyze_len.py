"""長度掃描分析：附加 n 句中性句後，相對 n=0 的溫度、第一名翻轉、門檻 0.9 翻轉（部署溫度／固定為 n=0 的溫度）。"""
import json
import math
import statistics as st
from pathlib import Path


def conf(scores, keys, t, dec):
    z = [scores[k] / max(t, 1e-4) for k in keys]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return e[keys.index(dec)] / sum(e)


def main():
    print("# 狀態長度掃描（基準 5 個選項；相對附加 0 句時）\n")
    print("| 模型 | 資料集 | 附加句數 | 平均溫度 | 第一名翻轉 | 門檻 0.9 翻轉：部署溫度 | 固定溫度 |")
    print("|---|---|---|---|---|---|---|")
    for f in sorted(Path("von_len_results").glob("*.json")):
        data = json.loads(f.read_text())
        for task, recs in data.items():
            for n in ["0", "1", "2", "4", "8"]:
                temps, flips, g_dep, g_fix = [], [], [], []
                for r in recs:
                    b, x = r["by_len"]["0"], r["by_len"][n]
                    keys = r["keys"]
                    dec = max(keys, key=b["scores"].get)
                    c0 = conf(b["scores"], keys, b["temp"], dec)
                    temps.append(x["temp"])
                    flips.append(max(keys, key=x["scores"].get) != dec)
                    g_dep.append((c0 >= 0.9) != (conf(x["scores"], keys, x["temp"], dec) >= 0.9))
                    g_fix.append((c0 >= 0.9) != (conf(x["scores"], keys, b["temp"], dec) >= 0.9))
                print(f"| {f.stem} | {task} | {n} | {st.mean(temps):.2f} | {100*st.mean(flips):.1f}% | "
                      f"{100*st.mean(g_dep):.1f}% | {100*st.mean(g_fix):.1f}% |")


if __name__ == "__main__":
    main()
