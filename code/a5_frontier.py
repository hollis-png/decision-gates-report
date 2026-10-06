"""每個 K 的「可達自動化」：在評估半上，錯誤率 ≤ r 時最多能自動處理的比例（依信心排序，只取決於排序）；
以及固定門檻 0.9 實際落在哪裡（出貨／單一溫度）。"""
import numpy as np
from a5_analysis import MODELS, FAMS, load, by_utt, rows
from calib_methods import Calibrator, is_fit

def frontier(x, r):
    c = sorted(x, key=lambda t: -t[1]); err = 0; best = 0
    for i, t in enumerate(c):
        err += (not t[2])
        if err / (i + 1) <= r: best = i + 1
    return 100 * best / len(c)

if __name__ == "__main__":
  for fam, tasks in FAMS.items():
      print(f"## {fam}\n\n| Model | K | Accuracy | Max auto @5% risk (single T) | Max auto @5% (shipped) | τ=0.9 shipped: auto / err | τ=0.9 single: auto / err |\n|---|---|---|---|---|---|---|")
      for model, title in MODELS:
          recs = load(model, tasks); cal = Calibrator([r for r in recs if is_fit(r["qid"])])
          ks = sorted({r["K"] for r in recs})
          ev = [rs for u, rs in by_utt(recs).items() if not is_fit(u + ":k5") and all(k in rs for k in ks)]
          for k in ks:
              g, d = rows(cal, "global_T", ev, k), rows(cal, "deployed", ev, k)
              acc = 100 * np.mean([t[2] for t in g])
              def at(x):
                  a = [t for t in x if t[1] >= .9]
                  return f"{100*len(a)/len(x):.0f} / " + (f"{100*np.mean([not t[2] for t in a]):.0f}" if a else "--")
              print(f"| {title} | {k} | {acc:.0f} | {frontier(g, .05):.0f} | {frontier(d, .05):.0f} | {at(d)} | {at(g)} |")
      print()
