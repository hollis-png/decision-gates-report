"""跨資料集校準：在一份資料擬合、部署到另一份，檢驗各校準方式的穩健性。

方向：Banking（隨機＋困難）→ CLINC（隨機＋困難），以及反方向。
假說：依選項數分組的溫度在跨資料時會產生跳躍（Laya 官方溫度表就是這個情況）；
     單一溫度與「獨立第一名」不會。指標同 calib_methods.py（評估資料的全部句子）。
"""
import json
from pathlib import Path

import numpy as np

from calib_methods import A2, METHODS, Calibrator, ece

MODELS = ["laya_td", "laya_base", "laya_multi", "von", "qwen3_4b_set", "qwen3_4b_point"]
SPLITS = {"banking→clinc": (["banking", "banking_hard"], ["clinc", "clinc_hard"]),
          "clinc→banking": (["clinc", "clinc_hard"], ["banking", "banking_hard"])}
SHOW = ["deployed", "global_T", "bucket_T", "indep_top"]


def load(model, tasks):
    return [r for t in tasks if (A2 / f"{t}__{model}.json").exists()
            for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]


def metrics(cal, recs, method):
    conf, corr, g9, g7 = [], [], [], []
    for r in recs:
        keys, s = r["keys"], r["base"]["scores"]
        dec = max(keys, key=s.get)
        c0 = cal.conf(method, s, keys, r["base"]["temp"], dec)
        c = r["conds"][0]
        c1 = cal.conf(method, c["scores"], list(c["scores"]), c["temp"], dec)
        conf.append(c0)
        corr.append(dec == r["anchor"])
        g9.append((c0 >= 0.9) != (c1 >= 0.9))
        g7.append((c0 >= 0.7) != (c1 >= 0.7))
    corr = np.asarray(corr, dtype=float)
    return {"ece": ece(conf, corr), "brier": float(np.mean((np.asarray(conf) - corr) ** 2)),
            "frac9": float(np.mean(np.asarray(conf) >= 0.9)), "g9": float(np.mean(g9)), "g7": float(np.mean(g7))}


def main():
    print("# 跨資料集校準：擬合資料 → 部署資料\n")
    print("門檻翻轉為 A2「K → K+1」的平均；溫度欄列出擬合出的分組溫度（2 | 3–5 | 6–10 | 11+）。\n")
    print("| 模型 | 方向 | 方法 | ECE | Brier | 信心≥0.9 | 門檻 0.9 翻轉 | 門檻 0.7 翻轉 | 擬合的分組溫度 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for name, (fit_tasks, eval_tasks) in SPLITS.items():
            fit, ev = load(model, fit_tasks), load(model, eval_tasks)
            if not fit or not ev:
                continue
            cal = Calibrator(fit)
            temps = " | ".join(f"{cal.t_bucket[b]:.2f}" for b in range(4))
            for m in SHOW:
                r = metrics(cal, ev, m)
                t = temps if m == "bucket_T" else ""
                print(f"| {model} | {name} | {m} | {100*r['ece']:.1f} | {r['brier']:.3f} | {100*r['frac9']:.0f}% | "
                      f"{100*r['g9']:.1f} | {100*r['g7']:.1f} | {t} |")


if __name__ == "__main__":
    main()
