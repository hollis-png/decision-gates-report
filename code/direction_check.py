"""審稿意見：門檻翻轉的方向、新增自動處理中的錯誤、每個 K 各自重設門檻（同覆蓋率）後仍剩多少處理改變。
路由模擬 K=5→6 與 5→12，評估半，門檻 0.9；重設門檻：每個 K 各自取使自動處理比例等於 K=5 時的門檻。"""
import json
from collections import defaultdict
import numpy as np
from calib_methods import A2, TASKS, Calibrator, is_fit

for model in ["laya_td", "von"]:
    recs = [r for t in TASKS if (A2 / f"{t}__{model}.json").exists()
            for r in json.loads((A2 / f"{t}__{model}.json").read_text())["records"]]
    cal = Calibrator([r for r in recs if is_fit(r["qid"])])
    by = defaultdict(dict)
    for r in recs:
        if not is_fit(r["qid"]):
            u, k = r["qid"].rsplit(":k", 1); by[u][int(k)] = r
    full = [rs for rs in by.values() if all(k in rs for k in range(5, 13))]
    for m in ["deployed", "global_T"]:
        def cd(r):
            d = max(r["keys"], key=r["base"]["scores"].get)
            return d, cal.conf(m, r["base"]["scores"], r["keys"], r["base"]["temp"], d), d == r["anchor"]
        for K in (6, 12):
            a = [cd(rs[5]) for rs in full]; b = [cd(rs[K]) for rs in full]
            c5 = np.array([x[1] for x in a]); cK = np.array([x[1] for x in b])
            auto5, autoK = c5 >= 0.9, cK >= 0.9
            up = (~auto5 & autoK).mean(); down = (auto5 & ~autoK).mean()
            err5 = np.mean([not x[2] for x, s in zip(a, auto5) if s]); errK = np.mean([not x[2] for x, s in zip(b, autoK) if s])
            # 同覆蓋率重設門檻
            cov = auto5.mean(); tK = np.sort(cK)[::-1][max(int(cov * len(cK)) - 1, 0)]
            autoK2 = cK >= tK
            dec_change = np.array([x[0] != y[0] for x, y in zip(a, b)])
            h_raw = ((auto5 != autoK) | dec_change).mean(); h_re = ((auto5 != autoK2) | dec_change).mean()
            print(f"{model:8s} {m:9s} K=5->{K:2d}: coverage {100*auto5.mean():.1f}%->{100*autoK.mean():.1f}%  "
                  f"esc->auto {100*up:.1f}  auto->esc {100*down:.1f}  err among auto {100*err5:.1f}%->{100*errK:.1f}%  "
                  f"handling raw {100*h_raw:.1f}  after re-threshold {100*h_re:.1f}  (route {100*dec_change.mean():.1f})")
