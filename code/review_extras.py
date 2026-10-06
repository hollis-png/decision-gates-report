"""回應自查意見的補充計算（完全離線，用已存的原始分數）。

  table1   表 1（Banking77 加 1 個選項的三因素拆解）的 95% 信賴區間，依請求重抽
  dose     圖 2 劑量反應：各模型四份資料集平均的門檻 0.9 翻轉率與 95% 信賴區間（依句子重抽，四份資料各自抽）
  learned  學習式信心：用集合式與逐一式分數的特徵訓練邏輯迴歸，預測集合式決策是否答對；
           與 hybrid.py 的五種方法同樣評估（50% 覆蓋率的選擇性準確率、加 5 個選項後的處理改變）
用法：python review_extras.py table1|dose|learned
"""
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

import analyze_a2
import hybrid
from analyze_a3 import decompose
from calib_methods import arr, is_fit, softmax, utt_key
from ci_tables import N_BOOT, cluster_boot, clustered, flat, selective_acc

V3 = hybrid.V3
analyze_a2.ROOT = __import__("pathlib").Path("a2_results")  # analyze_a2 預設把 argv[1] 當資料夾
T1_MODELS = ["laya_td", "laya_multi", "von", "qwen3_4b_set", "qwen3_4b_point"]
T1_KEYS = ["delta", "shap_S", "shap_D", "shap_T", "gate0.9", "gate0.9_only_S", "gate0.9_only_D", "gate0.9_only_T"]


def table1(rng):
    print("## Table 1 with 95% CIs (Banking77, add_1)\n")
    print("| Model | " + " | ".join(T1_KEYS) + " |\n|---|" + "---|" * len(T1_KEYS))
    for m in T1_MODELS:
        recs = json.loads((V3 / f"banking__{m}.json").read_text())["records"]
        rows = []
        for r in recs:
            for c in r["conds"]:
                if c["cond"] == "add_1":
                    d = decompose(r, c)
                    if d:
                        rows.append(d)
        cells = []
        for k in T1_KEYS:
            p, lo, hi = cluster_boot(rows, lambda gs: 100 * float(np.mean([g[k] for g in gs])), rng)
            cells.append(f"{p:+.1f} [{lo:+.1f}, {hi:+.1f}]" if not k.startswith("gate") else f"{p:.1f} [{lo:.1f}, {hi:.1f}]")
        print(f"| {m} (n={len(rows)}) | " + " | ".join(cells) + " |")
    print()


def dose(rng):
    """每個 K：四份資料集各自的翻轉率先平均（同圖 2），重抽時四份資料各自依句子抽。"""
    print("## Figure 2: gate-0.9 flips averaged over four sets, 95% CI\n")
    root = analyze_a2.ROOT
    widths = []
    for m, *_ in analyze_a2.MODELS:
        by_task = {}
        for t in analyze_a2.TASKS:
            f = root / f"{t}__{m}.json"
            if not f.exists():
                continue
            per_k = defaultdict(list)
            for rec in json.loads(f.read_text())["records"]:
                d = decompose(rec, rec["conds"][0])
                if d:
                    per_k[rec["K"]].append(d["gate0.9"])
            by_task[t] = {k: np.asarray(v) for k, v in per_k.items()}
        if not by_task:
            continue
        ks = sorted(set.intersection(*[set(v) for v in by_task.values()]))
        cells = []
        for k in ks:
            arrs = [by_task[t][k] for t in by_task]
            point = 100 * np.mean([a.mean() for a in arrs])
            reps = [100 * np.mean([a[rng.integers(0, len(a), len(a))].mean() for a in arrs]) for _ in range(N_BOOT)]
            lo, hi = np.percentile(reps, [2.5, 97.5])
            widths.append((hi - lo) / 2)
            cells.append(f"K={k}: {point:.1f} [{lo:.1f}, {hi:.1f}]")
        print(f"- **{m}**: " + "; ".join(cells))
    print(f"\nmax half-width: {max(widths):.1f} points\n")


# ---------------- 學習式信心 ----------------

def features(h, set_scores, point_scores, keys, use_set):
    """集合式決策的特徵。use_set=False 時只用逐一式特徵（對新增的弱選項不變）。"""
    dec = max(keys, key=set_scores.get)
    if use_set == "gap":  # 只用集合式前兩名的原始分數差距（不經 softmax 正規化）
        srt = np.sort(arr(set_scores, keys))[::-1]
        return dec, [float((srt[0] - srt[1]) / h.t_set)]
    pt = [point_scores[dec], hybrid.Hybrid.margin(point_scores, keys, dec)]
    if not use_set:
        return dec, pt
    z = arr(set_scores, keys)
    p = softmax(z / h.t_set)
    srt = np.sort(z)[::-1]
    # 不放 log K：擬合資料的選項數只有 4–5 個，放進去會在 K=10 時嚴重外插（已實測，會人為推高信心）
    return dec, pt + [float(np.log(p.max() + 1e-12)), float((srt[0] - srt[1]) / h.t_set)]


class Learned:
    def __init__(self, h, fit_pairs, use_set):
        self.h, self.use_set = h, use_set
        X, y = [], []
        for s, p in fit_pairs:
            dec, f = features(h, s["base"]["scores"], p["base"]["scores"], s["keys"], use_set)
            X.append(f)
            y.append(int(dec == s["anchor"]))
        X = np.asarray(X)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.clf = LogisticRegression(C=1.0, max_iter=1000).fit((X - self.mu) / self.sd, y)

    def decide(self, set_scores, point_scores, keys):
        dec, f = features(self.h, set_scores, point_scores, keys, self.use_set)
        return dec, float(self.clf.predict_proba(((np.asarray(f) - self.mu) / self.sd)[None])[0, 1])


def learned(rng):
    print("## Learned confidence (logistic regression), 95% CI\n")
    print("| Family | Features | Overall acc | Selective acc @50% | Handling changes +5 |\n|---|---|---|---|---|")
    for fam in hybrid.FAMILIES:
        pairs = hybrid.load_pairs(V3, hybrid.V3_TASKS, fam)
        fit = [pr for pr in pairs if is_fit(pr[0]["qid"])]
        ev = [pr for pr in pairs if not is_fit(pr[0]["qid"])]
        h = hybrid.Hybrid(fit)
        for name, use_set in [("joint + independent", True), ("independent only", False), ("joint gap only", "gap")]:
            L = Learned(h, fit, use_set)
            base = [L.decide(s["base"]["scores"], p["base"]["scores"], s["keys"]) for s, p in ev]
            confs = np.sort([c for _, c in base])[::-1]
            thr = float(confs[int(0.5 * len(confs)) - 1])
            groups, gkeys = [], []
            for (s, p), (dec, c) in zip(ev, base):
                sc = {x["cond"]: x for x in s["conds"]}
                pc = {x["cond"]: x for x in p["conds"]}
                flip = None
                if "add_5" in sc and "add_5" in pc:
                    keys1 = list(sc["add_5"]["scores"])
                    d1, c1 = L.decide(sc["add_5"]["scores"], pc["add_5"]["scores"], keys1)
                    flip = float(d1 != dec or (c >= thr) != (c1 >= thr))
                groups.append((c, float(dec == s["anchor"]), flip))
                gkeys.append(utt_key(s["qid"]))
            acc = 100 * np.mean([g[1] for g in groups])
            sel = cluster_boot(clustered(groups, gkeys), lambda gs: selective_acc([g[0] for g in flat(gs)], [g[1] for g in flat(gs)], 0.5), rng)
            fl = cluster_boot(clustered(groups, gkeys), lambda gs: 100 * float(np.mean([g[2] for g in flat(gs) if g[2] is not None])), rng)
            print(f"| {fam} | {name} | {acc:.1f} | {sel[0]:.1f} [{sel[1]:.1f}, {sel[2]:.1f}] | "
                  f"{fl[0]:.1f} [{fl[1]:.1f}, {fl[2]:.1f}] |")
    print()


# ---------------- 前兩名差距校準（表 3 的框架） ----------------

def gapcal(rng):
    """Platt 校準集合式前兩名的原始分數差距（不經 softmax、不用部署溫度），與表 3 同樣評估。"""
    from calib_methods import A2, TASKS, Calibrator, ece, fit_platt
    print("## Gap calibration in the Table 3 framework (95% CI)\\n")
    print("| Model | Method | ECE | Brier | c>=0.9 | Gate flips +1 | Gate flips +5 |\\n|---|---|---|---|---|---|---|")

    def gap(scores, keys, dec):
        return scores[dec] - max(scores[k] for k in keys if k != dec)

    for model in ["laya_td", "laya_base", "laya_multi", "von", "qwen3_4b_set"]:
        a2 = [r for t in TASKS if (A2 / f"{t}__{model}.json").exists()
              for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]
        fit = [r for r in a2 if is_fit(r["qid"])]
        items = []
        for r in fit:
            s, k = r["base"]["scores"], r["keys"]
            dec = max(k, key=s.get)
            items.append((np.array([gap(s, k, dec)]), 0 if dec == r["anchor"] else -1))
        t, b = fit_platt(items)
        conf = lambda s, k, dec: float(1 / (1 + np.exp(-(gap(s, k, dec) / t + b))))
        v3 = [r for tk in TASKS if (V3 / f"{tk}__{model}.json").exists()
              for r in json.loads((V3 / f"{tk}__{model}.json").read_text())["records"] if not is_fit(r["qid"])]
        utt = defaultdict(lambda: {"conf": [], "corr": [], "g1": [], "g5": []})
        for r in a2:
            if is_fit(r["qid"]):
                continue
            s, k = r["base"]["scores"], r["keys"]
            dec = max(k, key=s.get)
            c0 = conf(s, k, dec)
            c = r["conds"][0]
            c1 = conf(c["scores"], list(c["scores"]), dec)
            u = utt[utt_key(r["qid"])]
            u["conf"].append(c0)
            u["corr"].append(float(dec == r["anchor"]))
            u["g1"].append((c0 >= 0.9) != (c1 >= 0.9))
        for r in v3:
            s, k = r["base"]["scores"], r["keys"]
            dec = max(k, key=s.get)
            c0 = conf(s, k, dec)
            for c in r["conds"]:
                if c["cond"] == "add_5":
                    c1 = conf(c["scores"], list(c["scores"]), dec)
                    utt[utt_key(r["qid"])]["g5"].append((c0 >= 0.9) != (c1 >= 0.9))
        groups = list(utt.values())
        cat = lambda gs, key: [x for g in gs for x in g[key]]
        allc = np.asarray(cat(groups, "conf"))
        cells = []
        for name, f in [("ECE", lambda gs: 100 * ece(cat(gs, "conf"), cat(gs, "corr"))),
                        ("Brier", lambda gs: float(np.mean((np.asarray(cat(gs, "conf")) - np.asarray(cat(gs, "corr"))) ** 2))),
                        ("+1", lambda gs: 100 * float(np.mean(cat(gs, "g1")))),
                        ("+5", lambda gs: 100 * float(np.mean(cat(gs, "g5"))))]:
            p_, lo, hi = cluster_boot(groups, f, rng)
            cells.append(f"{p_:.3f} [{lo:.3f}, {hi:.3f}]" if name == "Brier" else f"{p_:.1f} [{lo:.1f}, {hi:.1f}]")
        print(f"| {model} | gap | {cells[0]} | {cells[1]} | {100*np.mean(allc >= 0.9):.0f}% | {cells[2]} | {cells[3]} |")
    print()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for job in sys.argv[1:]:
        {"table1": table1, "dose": dose, "learned": learned, "gapcal": gapcal}[job](rng)
