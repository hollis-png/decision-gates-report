"""方法貢獻：比較五種校準方式的「校準品質」與「增減選項時的門檻穩定性」。完全離線，用已存的原始分數。

  M0 deployed     模型自帶的溫度（Laya 分組、Von 連續公式；其他模型為 1）
  M1 global_T     單一溫度
  M2 bucket_T     依 Laya 分組（2 | 3-5 | 6-10 | 11+）各擬合一個溫度
  M3 smooth_T     T(K) = exp(a + b·log K)
  M4 independent  每個選項各自 q = sigmoid(z/T + b)（Platt，對「此選項是否為正確答案」擬合），不做跨選項正規化
  M5 indep_top    只用第一名的分數 q = sigmoid(z_top/T + b)，對「第一名是否答對」擬合；同樣不做跨選項正規化

擬合：A2 的基準清單（K=2..12，四份意圖資料集），依句子編號的雜湊切成擬合／評估兩半。
評估：評估半的 ECE、Brier（決策信心對答對與否）；
     門檻穩定：A2「K → K+1」與第三版 add_1／add_5、gibberish 的門檻 0.7／0.9 翻轉率（只用評估半的句子）。
"""
import hashlib
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

A2 = Path("a2_results")
V3 = Path("iia_v3_results")
TASKS = ["banking", "banking_hard", "clinc", "clinc_hard"]
MODELS = ["laya_td", "laya_base", "laya_multi", "von", "qwen3_4b_set", "qwen3_4b_point"]
METHODS = ["deployed", "global_T", "bucket_T", "smooth_T", "independent", "indep_top"]
GATES = (0.7, 0.9)


def bucket(k):
    return 0 if k <= 2 else 1 if k <= 5 else 2 if k <= 10 else 3


def utt_key(qid):
    """同一句話（或同一個 typed 情境）的鍵：隨機版與相近版（banking／banking_hard）共用同一句話，
    typed 的 '0:action'、'0:outcome' 共用同一個情境，都必須落在同一半、在重抽時同進同出。"""
    head = qid.split(":k")[0]
    task, rest = head.split(":", 1)
    if task.isdigit():  # typed-decisions：'情境編號:問題'
        return f"typed:{task}"
    return f"{task.replace('_hard', '')}:{rest}"


def is_fit(qid):
    """擬合半／評估半。2026-10-06 修正：原本以含資料集名稱的編號雜湊，隨機版與相近版的同一句話會分到不同半（資料洩漏）。"""
    return int(hashlib.md5(utt_key(qid).encode()).hexdigest(), 16) % 2 == 0


def arr(scores, keys):
    return np.array([scores[k] for k in keys], dtype=float)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------- 擬合 ----------------

def fit_temperature(items):
    """items: [(z 向量, 正確答案索引)] → 使 NLL 最小的溫度。"""
    def nll(logt):
        t = math.exp(logt[0])
        return -sum(np.log(softmax(z / t)[y] + 1e-12) for z, y in items) / len(items)
    return math.exp(minimize(nll, [0.0], method="Nelder-Mead").x[0])


def fit_smooth(items_k):
    """items_k: [(z, y, K)] → (a, b) 使 T(K) = exp(a + b log K) 的 NLL 最小。"""
    def nll(p):
        a, b = p
        return -sum(np.log(softmax(z / math.exp(a + b * math.log(k)))[y] + 1e-12) for z, y, k in items_k) / len(items_k)
    return minimize(nll, [0.0, 0.0], method="Nelder-Mead").x


def fit_platt(items):
    """每個選項一筆（分數, 是否正確）→ sigmoid(z/T + b) 的 (T, b)。"""
    zs = np.concatenate([z for z, _ in items])
    ys = np.concatenate([np.eye(len(z))[y] if y >= 0 else np.zeros(len(z)) for z, y in items])

    def nll(p):
        t, b = math.exp(p[0]), p[1]
        q = 1 / (1 + np.exp(-(zs / t + b)))
        return -np.mean(ys * np.log(q + 1e-12) + (1 - ys) * np.log(1 - q + 1e-12))
    x = minimize(nll, [0.0, 0.0], method="Nelder-Mead").x
    return math.exp(x[0]), x[1]


class Calibrator:
    def __init__(self, fit_recs):
        items = [(arr(r["base"]["scores"], r["keys"]), r["keys"].index(r["anchor"]), len(r["keys"])) for r in fit_recs]
        self.t_global = fit_temperature([(z, y) for z, y, _ in items])
        self.t_bucket = {}
        for b in range(4):
            sub = [(z, y) for z, y, k in items if bucket(k) == b]
            self.t_bucket[b] = fit_temperature(sub) if len(sub) >= 30 else self.t_global
        self.smooth = fit_smooth(items)
        self.platt = fit_platt([(z, y) for z, y, _ in items])
        self.platt_top = fit_platt([(np.array([z.max()]), 0 if int(z.argmax()) == y else -1) for z, y, _ in items])

    def conf(self, method, scores, keys, deployed_temp, decision):
        """決策 decision 在清單 keys 下的信心。"""
        z, k = arr(scores, keys), len(keys)
        i = keys.index(decision)
        if method in ("independent", "indep_top"):
            t, b = self.platt if method == "independent" else self.platt_top
            return float(1 / (1 + math.exp(-(z[i] / t + b))))
        t = {"deployed": deployed_temp, "global_T": self.t_global, "bucket_T": self.t_bucket[bucket(k)],
             "smooth_T": math.exp(self.smooth[0] + self.smooth[1] * math.log(k))}[method]
        return float(softmax(z / max(t, 1e-4))[i])


# ---------------- 評估 ----------------

def ece(conf, corr, bins=15):
    conf, corr = np.asarray(conf), np.asarray(corr, dtype=float)
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        s = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - corr[s].mean())
    return float(e)


def evaluate(model):
    a2 = [r for t in TASKS if (A2 / f"{t}__{model}.json").exists()
          for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]
    if not a2:
        return None
    cal = Calibrator([r for r in a2 if is_fit(r["qid"])])
    ev = [r for r in a2 if not is_fit(r["qid"])]
    out = {m: {} for m in METHODS}
    for m in METHODS:
        conf, corr, gate = [], [], defaultdict(list)
        for r in ev:
            keys, s = r["keys"], r["base"]["scores"]
            dec = max(keys, key=s.get)
            c0 = cal.conf(m, s, keys, r["base"]["temp"], dec)
            conf.append(c0)
            corr.append(dec == r["anchor"])
            c = r["conds"][0]
            c1 = cal.conf(m, c["scores"], list(c["scores"]), c["temp"], dec)
            for g in GATES:
                gate[g].append((c0 >= g) != (c1 >= g))
        corr = np.asarray(corr, dtype=float)
        out[m]["ece"] = ece(conf, corr)
        out[m]["brier"] = float(np.mean((np.asarray(conf) - corr) ** 2))
        out[m]["mean_conf"] = float(np.mean(conf))
        out[m]["frac_ge_0.9"] = float(np.mean(np.asarray(conf) >= 0.9))  # 檢查門檻指標是否因信心從未達到門檻而失效
        out[m]["acc"] = float(corr.mean())
        for g in GATES:
            out[m][f"a2_gate{g}"] = float(np.mean(gate[g]))
        # 第三版的加選項操作（只用評估半的句子；基準清單 5 個選項）
        v3gate = defaultdict(list)
        for t in TASKS:
            f = V3 / f"{t}__{model}.json"
            if not f.exists():
                continue
            for r in json.loads(f.read_text())["records"]:
                if is_fit(r["qid"]):
                    continue
                keys, s = r["keys"], r["base"]["scores"]
                dec = max(keys, key=s.get)
                c0 = cal.conf(m, s, keys, r["base"]["temp"], dec)
                for c in r["conds"]:
                    if c["cond"] in ("add_1", "add_5", "gibberish"):
                        c1 = cal.conf(m, c["scores"], list(c["scores"]), c["temp"], dec)
                        v3gate[(c["cond"], 0.9)].append((c0 >= 0.9) != (c1 >= 0.9))
                        v3gate[(c["cond"], 0.7)].append((c0 >= 0.7) != (c1 >= 0.7))
        for (cond, g), v in v3gate.items():
            out[m][f"{cond}_gate{g}"] = float(np.mean(v))
    out["_fitted"] = {"t_global": cal.t_global, "t_bucket": cal.t_bucket, "smooth_ab": list(cal.smooth),
                      "platt_T_b": list(cal.platt), "platt_top_T_b": list(cal.platt_top)}
    return out


def main():
    results = {}
    print("# 方法貢獻：校準方式比較（評估半）\n")
    print("ECE／Brier 越低越好；門檻翻轉越低越穩定。A2 = K→K+1 平均；add_1／add_5／gibberish 為第三版（基準 5 個選項）。\n")
    print("| 模型 | 方法 | 準確率 | ECE | Brier | 平均信心 | 信心≥0.9 比例 | A2 門檻 0.9 | A2 門檻 0.7 | add_1 門檻 0.9 | add_5 門檻 0.9 | 亂碼 門檻 0.9 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        res = evaluate(model)
        if res is None:
            continue
        results[model] = res
        for m in METHODS:
            r = res[m]
            g = lambda k: f"{100*r[k]:.1f}" if k in r else "—"
            print(f"| {model} | {m} | {100*r['acc']:.1f} | {100*r['ece']:.1f} | {r['brier']:.3f} | {r['mean_conf']:.3f} | {100*r['frac_ge_0.9']:.0f}% | "
                  f"{g('a2_gate0.9')} | {g('a2_gate0.7')} | {g('add_1_gate0.9')} | {g('add_5_gate0.9')} | {g('gibberish_gate0.9')} |")
    Path("calib_methods_results.json").write_text(json.dumps(results, indent=1, default=float))
    print("\n擬合參數見 calib_methods_results.json 的 _fitted。")


if __name__ == "__main__":
    main()
