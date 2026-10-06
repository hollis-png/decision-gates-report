"""A5 分析：大選項數、巢狀保留順序。每個模型 × 資料（Banking 隨機＋相近；CLINC 隨機＋相近），各校準方式：
  deployed（出貨）、global_T（單一溫度，在擬合半、所有 K 上擬合）、indep_top（只用 Von）
每個 K：準確率、門檻 0.9 的自動處理比例與其中錯誤率、自動處理一半時的風險、重設門檻後相對 K=5 的重排與路由改變、
K→K+1 的門檻翻轉。重排附依句子重抽的 95% 區間。
Laya 用放寬選項預算的 *_long 結果（K ≤ 12 時與出貨預算相同；出貨預算在 24 個以上的選項多半拒答）。
另外比較 A2（每個 K 重新打亂）與 A5（保留順序）在 K=12 的重排，估計重新打亂的貢獻。
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calib_methods import Calibrator, is_fit, utt_key
from layers import thr_for
from review5 import boot, fmt

R = Path("a5_results")
MODELS = [("laya_td_long", "Laya, fine-tuned"), ("laya_base_long", "Laya, base"), ("laya_multi_long", "Laya, multilingual"),
          ("von", "Von"), ("qwen3_4b_num", "Qwen3-4B, joint")]
FAMS = {"Banking77": ["banking", "banking_hard"], "CLINC150": ["clinc", "clinc_hard"]}


def load(model, tasks):
    return [r for t in tasks if (R / f"{t}__{model}.json").exists()
            for r in json.loads((R / f"{t}__{model}.json").read_text())["records"]]


def by_utt(recs):
    d = defaultdict(dict)
    for r in recs:
        u, k = r["qid"].rsplit(":k", 1)
        d[u][int(k)] = r
    return d


def rows(cal, m, rs_list, k):
    out = []
    for rs in rs_list:
        r = rs[k]
        d = max(r["keys"], key=r["base"]["scores"].get)
        out.append((d, cal.conf(m, r["base"]["scores"], r["keys"], r["base"]["temp"], d), d == r["anchor"]))
    return out


def gate_flip(cal, m, rs_list, k):
    f = []
    for rs in rs_list:
        r = rs[k]
        if not r["conds"]:
            return np.nan
        d = max(r["keys"], key=r["base"]["scores"].get)
        c0 = cal.conf(m, r["base"]["scores"], r["keys"], r["base"]["temp"], d)
        c = r["conds"][0]
        c1 = cal.conf(m, c["scores"], list(c["scores"]), c["temp"], d)
        f.append((c0 >= .9) != (c1 >= .9))
    return 100 * np.mean(f)


def main():
    out = {}
    for fam, tasks in FAMS.items():
        print(f"## {fam}\n")
        print("| Model | Calib. | K | Accuracy | Auto τ=0.9 | Err auto | Sel. risk 50% | Gate flips K→K+1 | Reordering vs K=5 | Route changes |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for model, title in MODELS:
            recs = load(model, tasks)
            if not recs:
                continue
            cal = Calibrator([r for r in recs if is_fit(r["qid"])])
            utts = by_utt(recs)
            ks = sorted({r["K"] for r in recs})
            ev = {u: rs for u, rs in utts.items() if not is_fit(u + ":k5") and all(k in rs for k in ks)}
            rs_list, keys = list(ev.values()), [utt_key(u) for u in ev]
            meths = ["deployed", "global_T"] + (["indep_top"] if model == "von" else [])
            for m in meths:
                a = rows(cal, m, rs_list, 5)
                ta = thr_for([x[1] for x in a], .5)
                for k in ks:
                    b = rows(cal, m, rs_list, k)
                    c = np.array([x[1] for x in b]); ok = np.array([x[2] for x in b])
                    auto = c >= .9
                    tb = thr_for(list(c), .5)
                    half = c >= tb
                    reo = lambda idx: 100 * np.mean([a[i][0] != b[i][0] or (a[i][1] >= ta) != (b[i][1] >= tb) for i in idx])
                    h = boot(keys, reo) if k > 5 else (0.0, 0.0, 0.0)
                    route = 100 * np.mean([a[i][0] != b[i][0] for i in range(len(a))])
                    row = dict(acc=100 * ok.mean(), cov=100 * auto.mean(),
                               err=100 * np.mean(~ok[auto]) if auto.any() else np.nan,
                               risk=100 * np.mean(~ok[half]), flip=gate_flip(cal, m, rs_list, k),
                               reo=h, route=route)
                    out.setdefault(fam, {}).setdefault(model, {}).setdefault(m, {})[k] = row
                    e = "--" if np.isnan(row["err"]) else f"{row['err']:.1f}"
                    fl = "--" if np.isnan(row["flip"]) else f"{row['flip']:.1f}"
                    print(f"| {title} | {m} | {k} | {row['acc']:.1f} | {row['cov']:.1f} | {e} | {row['risk']:.1f} | {fl} | "
                          f"{fmt(h)} | {route:.1f} |")
            print(f"|  | fitted single T = {cal.t_global:.2f}; smooth (a, b) = ({cal.smooth[0]:.2f}, {cal.smooth[1]:.2f}) | n = {len(rs_list)} |||||||")
        print()
    json.dump(out, open("a5_analysis.json", "w"), default=lambda x: list(x) if isinstance(x, tuple) else float(x))

    # 順序的貢獻：A2（各 K 重新打亂）vs A5（保留順序），K=12，單一溫度與出貨校準
    print("## Reordering at 12 routes: reshuffled per K (A2) vs order-preserving (A5), both datasets pooled\n")
    print("| Model | Calib. | A2 reshuffled | A5 order kept | A2 route changes | A5 route changes |\n|---|---|---|---|---|---|")
    from calib_methods import A2
    for (m5, title), m2 in zip(MODELS, ["laya_td", "laya_base", "laya_multi", "von", None]):
        for m in ("deployed", "global_T"):
            cells = []
            for root, model in ((A2, m2), (R, m5)):
                if model is None:
                    cells.append(("--", "--"))
                    continue
                recs = [r for t in FAMS["Banking77"] + FAMS["CLINC150"] if (root / f"{t}__{model}.json").exists()
                        for r in json.loads((root / f"{t}__{model}.json").read_text())["records"]]
                cal = Calibrator([r for r in recs if is_fit(r["qid"])])
                ev = {u: rs for u, rs in by_utt(recs).items() if not is_fit(u + ":k5") and 5 in rs and 12 in rs}
                rl, ky = list(ev.values()), [utt_key(u) for u in ev]
                a, b = rows(cal, m, rl, 5), rows(cal, m, rl, 12)
                ta, tb = thr_for([x[1] for x in a], .5), thr_for([x[1] for x in b], .5)
                h = boot(ky, lambda idx: 100 * np.mean([a[i][0] != b[i][0] or (a[i][1] >= ta) != (b[i][1] >= tb) for i in idx]))
                cells.append((fmt(h), f"{100*np.mean([a[i][0] != b[i][0] for i in range(len(a))]):.1f}"))
            print(f"| {title} | {m} | {cells[0][0]} | {cells[1][0]} | {cells[0][1]} | {cells[1][1]} |")
    print()


if __name__ == "__main__":
    main()
