"""想法 1：混合式——集合式負責決策、逐一式負責信心。完全離線，用第三版與 A2 已存的分數。

  set_T     集合式決策＋集合式分數的單一溫度（集合式基準線）
  point_IT  逐一式決策＋逐一式第一名分數的 Platt（逐一式基準線，「獨立第一名」）
  hybrid    集合式決策＋逐一式對「該決策選項」的分數做 Platt（擬合目標：集合式決策是否答對）

逐一式分數由構造保證不受其他選項影響，所以 hybrid 的信心只在集合式決策換人時才會變。
評估：準確率、ECE、Brier、信心≥0.9 比例；第三版加 1/3/5 個選項、亂碼選項、換順序、狀態加句的門檻翻轉；
     A2「K→K+1」門檻翻轉；A4 部署模擬（路由 5→12，只有 Qwen3-4B 有 A2 資料）。
擬合／評估依句子雜湊切半（同 calib_methods.is_fit）。
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from calib_methods import arr, ece, fit_platt, fit_temperature, is_fit, softmax

V3, A2 = Path("iia_v3_results"), Path("a2_results")
FAMILIES = ["qwen3_4b", "qwen25_7b", "phi4mini", "mistral7b"]
V3_TASKS = ["typed", "banking", "banking_hard", "clinc", "clinc_hard"]
A2_TASKS = ["banking", "banking_hard", "clinc", "clinc_hard"]
COND_GATES = ["add_1", "add_3", "add_5", "gibberish", "permute", "filler_state"]
METHODS = ["set_T", "point_IT", "hybrid", "point_margin", "hybrid_margin"]
# point_margin   逐一式決策＋逐一式邊際（決策分數 − 其他選項最高逐一式分數）的 Platt
# hybrid_margin  集合式決策＋同樣的逐一式邊際
# 逐一式邊際的性質：各選項分數獨立，加入低分的無關選項時最高對手不變、信心完全不變；
# 只有新選項超過原本的第二名時信心才下降——只對真正相關的選項有反應。
GATES = (0.7, 0.9)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def load_pairs(root, tasks, family):
    """回傳以 qid 配對的 (集合式紀錄, 逐一式紀錄)。"""
    pairs = []
    for t in tasks:
        fs, fp = root / f"{t}__{family}_set.json", root / f"{t}__{family}_point.json"
        if not fs.exists() or not fp.exists():
            continue
        point = {r["qid"]: r for r in json.loads(fp.read_text())["records"]}
        for r in json.loads(fs.read_text())["records"]:
            if r["qid"] in point:
                pairs.append((r, point[r["qid"]]))
    return pairs


class Hybrid:
    def __init__(self, fit_pairs):
        set_items = [(arr(s["base"]["scores"], s["keys"]), s["keys"].index(s["anchor"])) for s, _ in fit_pairs]
        self.t_set = fit_temperature(set_items)
        # 逐一式「獨立第一名」：逐一式第一名的分數 → 逐一式決策是否答對
        top_pt = []
        for _, p in fit_pairs:
            z = arr(p["base"]["scores"], p["keys"])
            top_pt.append((np.array([z.max()]), 0 if p["keys"][int(z.argmax())] == p["anchor"] else -1))
        self.platt_point = fit_platt(top_pt)
        # 混合式：逐一式對「集合式決策選項」的分數 → 集合式決策是否答對
        hy = []
        for s, p in fit_pairs:
            dec = max(s["keys"], key=s["base"]["scores"].get)
            hy.append((np.array([p["base"]["scores"][dec]]), 0 if dec == s["anchor"] else -1))
        self.platt_hybrid = fit_platt(hy)
        pm, hm = [], []
        for s, p in fit_pairs:
            pdec = max(p["keys"], key=p["base"]["scores"].get)
            pm.append((np.array([self.margin(p["base"]["scores"], p["keys"], pdec)]), 0 if pdec == p["anchor"] else -1))
            sdec = max(s["keys"], key=s["base"]["scores"].get)
            hm.append((np.array([self.margin(p["base"]["scores"], s["keys"], sdec)]), 0 if sdec == s["anchor"] else -1))
        self.platt_pm, self.platt_hm = fit_platt(pm), fit_platt(hm)

    @staticmethod
    def margin(point_scores, keys, dec):
        """決策選項的逐一式分數減去其他選項中最高的逐一式分數（可為負：集合式選了逐一式不認為最好的）。"""
        return point_scores[dec] - max(point_scores[k] for k in keys if k != dec)

    def decide(self, method, set_scores, point_scores, keys):
        """在清單 keys 下的（決策, 信心）。"""
        if method == "set_T":
            z = arr(set_scores, keys)
            p = softmax(z / self.t_set)
            i = int(p.argmax())
            return keys[i], float(p[i])
        if method == "point_IT":
            dec = max(keys, key=point_scores.get)
            t, b = self.platt_point
            return dec, sigmoid(point_scores[dec] / t + b)
        if method == "point_margin":
            dec = max(keys, key=point_scores.get)
            t, b = self.platt_pm
            return dec, sigmoid(self.margin(point_scores, keys, dec) / t + b)
        dec = max(keys, key=set_scores.get)
        if method == "hybrid_margin":
            t, b = self.platt_hm
            return dec, sigmoid(self.margin(point_scores, keys, dec) / t + b)
        t, b = self.platt_hybrid
        return dec, sigmoid(point_scores[dec] / t + b)


COVERAGES = (0.5, 0.7)  # 自動處理比例：各方法各自設門檻，使基準時有這個比例的請求被自動處理


def flips(h, method, s_base, p_base, keys0, s_new, p_new, keys1, gates=GATES):
    d0, c0 = h.decide(method, s_base, p_base, keys0)
    d1, c1 = h.decide(method, s_new, p_new, keys1)
    return {g: (d0 != d1) or ((c0 >= g) != (c1 >= g)) for g in gates}, d0 != d1


def selective_acc_tie(conf, corr, cov):
    """信心最高的 cov 比例的準確率；門檻上的同分按比例計入（同 ci_tables.selective_acc，2026-10-06 統一）。"""
    n = int(cov * len(conf))
    t = np.sort(conf)[::-1][n - 1]
    above, tie = conf > t, conf == t
    return float((corr[above].sum() + (n - above.sum()) * corr[tie].mean()) / n)


def coverage_gates(conf):
    """各自動處理比例對應的門檻（基準信心的分位數）。"""
    srt = np.sort(np.asarray(conf))[::-1]
    return {cov: float(srt[int(cov * len(srt)) - 1]) for cov in COVERAGES}


def evaluate_v3(family):
    pairs = load_pairs(V3, V3_TASKS, family)
    if not pairs:
        return None
    h = Hybrid([pr for pr in pairs if is_fit(pr[0]["qid"])])
    ev = [pr for pr in pairs if not is_fit(pr[0]["qid"])]
    out = {}
    for m in METHODS:
        conf, corr = [], []
        for s, p in ev:
            dec, c = h.decide(m, s["base"]["scores"], p["base"]["scores"], s["keys"])
            conf.append(c)
            corr.append(dec == s["anchor"])
        thr = coverage_gates(conf)
        g = defaultdict(list)
        for s, p in ev:
            sc = {c_["cond"]: c_ for c_ in s["conds"]}
            pc = {c_["cond"]: c_ for c_ in p["conds"]}
            for cond in COND_GATES:
                if cond in sc and cond in pc:
                    keys1 = list(sc[cond]["scores"])
                    gf, _ = flips(h, m, s["base"]["scores"], p["base"]["scores"], s["keys"],
                                  sc[cond]["scores"], pc[cond]["scores"], keys1, gates=list(thr.values()))
                    for cov, t in thr.items():
                        g[(cond, cov)].append(gf[t])
        conf_a, corr = np.asarray(conf), np.asarray(corr, dtype=float)
        sel = {cov: selective_acc_tie(conf_a, corr, cov) for cov in COVERAGES}
        out[m] = {"acc": float(corr.mean()), "ece": ece(conf, corr),
                  "brier": float(np.mean((conf_a - corr) ** 2)), "frac9": float(np.mean(conf_a >= 0.9)),
                  **{f"sel{cov}": v for cov, v in sel.items()},
                  **{f"{c}_c{cov}": float(np.mean(v)) for (c, cov), v in g.items()}}
    return out


def evaluate_a2(family):
    pairs = load_pairs(A2, A2_TASKS, family)
    if not pairs:
        return None
    h = Hybrid([pr for pr in pairs if is_fit(pr[0]["qid"])])
    ev = [pr for pr in pairs if not is_fit(pr[0]["qid"])]
    out, thr_m = {}, {}
    for m in METHODS:
        conf = [h.decide(m, s["base"]["scores"], p["base"]["scores"], s["keys"])[1] for s, p in ev]
        thr = coverage_gates(conf)
        thr_m[m] = thr
        g = defaultdict(list)
        for s, p in ev:
            gf, _ = flips(h, m, s["base"]["scores"], p["base"]["scores"], s["keys"],
                          s["conds"][0]["scores"], p["conds"][0]["scores"], list(s["conds"][0]["scores"]),
                          gates=list(thr.values()))
            for cov, t in thr.items():
                g[cov].append(gf[t])
        out[m] = {f"a2_c{cov}": float(np.mean(v)) for cov, v in g.items()}
    # A4 部署模擬：同一句話 K=5..12 的基準清單是巢狀的
    by_utt = defaultdict(dict)
    for s, p in ev:
        utt, k = s["qid"].rsplit(":k", 1)
        by_utt[utt][int(k)] = (s, p)
    ks = list(range(5, 13))
    sim = {m: {k: [] for k in ks} for m in METHODS}
    full = [rs for rs in by_utt.values() if all(k in rs for k in ks)]
    # 2026-10-06 修正：門檻取自 K=5 的基準信心（原本誤用 K=2..12 全部合併的分位數）
    thr5 = {m: coverage_gates([h.decide(m, rs[5][0]["base"]["scores"], rs[5][1]["base"]["scores"], rs[5][0]["keys"])[1]
                               for rs in full])[0.5] for m in METHODS}
    for rs in by_utt.values():
        if not all(k in rs for k in ks):
            continue
        for m in METHODS:
            s5, p5 = rs[5]
            d5, c5 = h.decide(m, s5["base"]["scores"], p5["base"]["scores"], s5["keys"])
            for k in ks:
                s, p = rs[k]
                d, c = h.decide(m, s["base"]["scores"], p["base"]["scores"], s["keys"])
                t = thr5[m]  # K=5 時自動處理 50% 的門檻
                sim[m][k].append((d != d5) or ((c >= t) != (c5 >= t)))
    out["_a4"] = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in sim.items()}
    return out


def main():
    names = {"set_T": "集合式＋單一溫度", "point_IT": "逐一式＋獨立第一名", "hybrid": "混合式（第一名分數）",
             "point_margin": "逐一式＋逐一式邊際", "hybrid_margin": "**混合式＋逐一式邊際**"}
    print("# 想法 1：混合式（集合式決策＋逐一式信心）\n")
    print("## 第三版（評估半）\n")
    print("各方法各自設門檻，使基準時 50% 的請求被自動處理；翻轉＝決策或「是否自動處理」任一改變。"
          "選擇性準確率＝被自動處理的那 50%／70% 中答對的比例。\n")
    print("| 模型 | 方法 | 準確率 | ECE | Brier | 選擇性準確率 50%／70% | 加1 | 加3 | 加5 | 亂碼 | 換順序 | 狀態加句 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    results = {}
    for fam in FAMILIES:
        r = evaluate_v3(fam)
        if not r:
            continue
        results[fam] = {"v3": r}
        for m in METHODS:
            x = r[m]
            cells = " | ".join(f"{100*x.get(f'{c}_c0.5', float('nan')):.1f}" for c in COND_GATES)
            print(f"| {fam} | {names[m]} | {100*x['acc']:.1f} | {100*x['ece']:.1f} | {x['brier']:.3f} | "
                  f"{100*x['sel0.5']:.1f}／{100*x['sel0.7']:.1f} | {cells} |")
    print("\n（加1～狀態加句 欄為自動處理比例 50% 時的翻轉率 %）\n")
    print("## A2 與 A4（Qwen3-4B）\n")
    r = evaluate_a2("qwen3_4b")
    if r:
        results.setdefault("qwen3_4b", {})["a2"] = r
        print("A2＝K→K+1 的翻轉率；A4＝路由從 5 個擴充時與 5 個相比的改變比例；皆為自動處理比例 50%／70% 的門檻。\n")
        print("| 方法 | A2（50%） | A2（70%） | " + " | ".join(f"A4 {k} 個路由" for k in range(6, 13)) + " |")
        print("|---|---|---|" + "---|" * 7)
        for m in METHODS:
            a4 = r["_a4"][m]
            print(f"| {names[m]} | {100*r[m]['a2_c0.5']:.1f} | {100*r[m]['a2_c0.7']:.1f} | "
                  + " | ".join(f"{100*a4[k]:.1f}" for k in range(6, 13)) + " |")
    Path("hybrid_results.json").write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
