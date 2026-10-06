"""審稿意見：只看「決策不變、只有自動處理狀態改變」的順序打亂，錯誤率是否高於整體；以及這些請求離門檻多近。
路由模擬 K=5 vs 12，每個 K 重設門檻使自動處理 50%。依句子鍵重抽。"""
import json
from collections import defaultdict
import numpy as np
from calib_methods import A2, TASKS, Calibrator, is_fit, utt_key
from layers import thr_for, boot_rows

for model in ["laya_td", "laya_base", "laya_multi", "von", "qwen3_4b_set"]:
    recs = [r for t in TASKS if (A2 / f"{t}__{model}.json").exists() for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]
    cal = Calibrator([r for r in recs if is_fit(r["qid"])])
    by = defaultdict(dict)
    for r in recs:
        if not is_fit(r["qid"]):
            u, k = r["qid"].rsplit(":k", 1); by[u][int(k)] = r
    full = {u: rs for u, rs in by.items() if all(k in rs for k in range(5, 13))}
    keys = [utt_key(u) for u in full]
    rng = np.random.default_rng(0)
    for m in ["deployed", "global_T", "indep_top"]:
        def row(r):
            d = max(r["keys"], key=r["base"]["scores"].get)
            return d, cal.conf(m, r["base"]["scores"], r["keys"], r["base"]["temp"], d), d == r["anchor"]
        a = [row(rs[5]) for rs in full.values()]; b = [row(rs[12]) for rs in full.values()]
        t5 = thr_for([x[1] for x in a], .5); t12 = thr_for([x[1] for x in b], .5)
        same = [i for i in range(len(a)) if a[i][0] == b[i][0]]
        gate = [i for i in same if (a[i][1] >= t5) != (b[i][1] >= t12)]
        # 名次分位：請求在各自集合中的信心百分位離 50% 門檻多遠
        ra = np.argsort(np.argsort([-x[1] for x in a])) / len(a); rb = np.argsort(np.argsort([-x[1] for x in b])) / len(b)
        dist = [max(abs(ra[i] - .5), abs(rb[i] - .5)) for i in gate]
        err_gate = lambda idx: 100 * np.mean([not a[i][2] for i in idx if i in gate_set]) if any(i in gate_set for i in idx) else np.nan
        gate_set = set(gate)
        err_all = 100 * np.mean([not a[i][2] for i in same])
        p = boot_rows(full, keys, rng, lambda idx: 100 * np.mean([not a[i][2] for i in idx if i in gate_set]) if any(i in gate_set for i in idx) else np.nan)
        print(f"{model:13s} {m:9s} gate-only reorder {100*len(gate)/len(a):.1f}% | error among them {p[0]:.1f} [{p[1]:.1f},{p[2]:.1f}] vs {err_all:.1f}% among unchanged-decision requests | median rank distance from threshold {np.median(dist):.2f} (90th pct {np.quantile(dist,.9):.2f})")
