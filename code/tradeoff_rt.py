"""圖 3 的重設門檻版：加 5 個選項後，基準與加選項後各自依自己的信心設門檻（自動處理 50%），
只剩「順序打亂」的處理改變；y 軸仍是 50% 覆蓋率的選擇性準確率（同分按比例）。依句子鍵重抽。
輸出格式同 ci_t7.md：| 家族 | 方法 | 選擇性準確率 | 處理改變 |"""
import numpy as np

import hybrid
from calib_methods import is_fit, utt_key
from ci_tables import cluster_boot, clustered, flat, selective_acc
from review_extras import Learned

COV = 0.5


def thr_for(conf):
    s = np.sort(np.asarray(conf))[::-1]
    return float(s[int(COV * len(s)) - 1])


def main():
    rng = np.random.default_rng(0)
    print("| Family | Method | Selective accuracy | Handling changes +5 (re-thresholded) |\n|---|---|---|---|")
    for fam in hybrid.FAMILIES:
        pairs = hybrid.load_pairs(hybrid.V3, hybrid.V3_TASKS, fam)
        fit = [pr for pr in pairs if is_fit(pr[0]["qid"])]
        ev = [pr for pr in pairs if not is_fit(pr[0]["qid"])]
        h = hybrid.Hybrid(fit)
        learned = {"joint_gap": Learned(h, fit, "gap"), "learned_indep": Learned(h, fit, False),
                   "learned_all": Learned(h, fit, True)}
        methods = list(hybrid.METHODS) + list(learned)
        for m in methods:
            dec = (lambda s, p, k: learned[m].decide(s, p, k)) if m in learned else (lambda s, p, k: h.decide(m, s, p, k))
            base = [dec(s["base"]["scores"], p["base"]["scores"], s["keys"]) for s, p in ev]
            edited = []
            for s, p in ev:
                sc = {x["cond"]: x for x in s["conds"]}
                pc = {x["cond"]: x for x in p["conds"]}
                if "add_5" in sc and "add_5" in pc:
                    k1 = list(sc["add_5"]["scores"])
                    edited.append(dec(sc["add_5"]["scores"], pc["add_5"]["scores"], k1))
                else:
                    edited.append(None)
            idx = [i for i, e in enumerate(edited) if e is not None]
            t0 = thr_for([base[i][1] for i in idx])
            t1 = thr_for([edited[i][1] for i in idx])
            sel_items = [(c, float(d == s["anchor"])) for (s, _), (d, c) in zip(ev, base)]
            sel_keys = [utt_key(s["qid"]) for s, _ in ev]
            ch = [float(base[i][0] != edited[i][0] or (base[i][1] >= t0) != (edited[i][1] >= t1)) for i in idx]
            ch_keys = [utt_key(ev[i][0]["qid"]) for i in idx]
            sel = cluster_boot(clustered(sel_items, sel_keys),
                               lambda gs: selective_acc([a for a, _ in flat(gs)], [b for _, b in flat(gs)], COV), rng)
            fl = cluster_boot(clustered(ch, ch_keys), lambda gs: 100 * float(np.mean(flat(gs))), rng)
            print(f"| {fam} | {m} | {sel[0]:.1f} [{sel[1]:.1f}, {sel[2]:.1f}] | {fl[0]:.1f} [{fl[1]:.1f}, {fl[2]:.1f}] |")


if __name__ == "__main__":
    main()
