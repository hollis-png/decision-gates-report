"""先導實驗：Laya 是否違反無關選項獨立性（Luce 選擇公理）。

對 typed-decisions 測試集中選項數 >= 3 的選擇題，比較四種操作前後「原有選項之間的對數機率比」：
  permute    只換順序（雜訊底線：理論上應完全不變）
  remove     刪掉一個非第一名的選項
  add_irrel  加入一個明顯無關的選項
  duplicate  複製第一名選項的描述、換一個名稱（紅巴士／藍巴士測試）

全程 fp32、溫度 1（原始分數），排除精度與溫度的影響；選項被 token 預算截斷的題目一律排除。
"""
import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence

IRRELEVANT = ("weather_forecast", "questions about tomorrow's weather forecast and rainfall")
CHECKPOINTS = {"base": "", "typed-decisions": "typed-decisions"}


def load(root, subdir, device):
    ckpt = Path(root) / subdir if subdir else Path(root)
    _fix_tokenizer_config(str(ckpt))
    cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
    cfg.update({"max_len": 1024, "head_max_len": 256})
    tok = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
    model = build_model(cfg, encoder_dir=ckpt / "encoder")
    model.load_state_dict(load_file(str(ckpt / "model.safetensors")), strict=True)
    return model.float().to(device).eval(), tok, cfg


@torch.no_grad()
def log_probs(model, tok, cfg, state, ins, crit, device):
    """回傳 {選項鍵: log 機率}；若任何選項被 token 預算截斷則回傳 None。"""
    q = {"t": "choice", "ins": ins, "crit": crit}
    ids, markers, stats = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"], return_stats=True)
    if stats.get("tokens_per_option") is not None or stats.get("options_distinct") != len(crit):
        return None
    t = lambda x: torch.tensor([x], device=device)
    mask = torch.ones((1, len(markers)), dtype=torch.bool, device=device)
    logits, _ = model(t(ids), torch.ones_like(t(ids)), t(markers), mask, t(QTYPES["choice"]))
    lp = torch.log_softmax(logits[0].float(), -1).tolist()
    return dict(zip(crit.keys(), lp))


def ratio_shift(base, new, keys):
    """原有選項（keys）之間對數機率比的變化：以 keys 內的平均對數機率置中後比較，取平均絕對差。"""
    b = [base[k] for k in keys]
    n = [new[k] for k in keys]
    mb, mn = sum(b) / len(b), sum(n) / len(n)
    return sum(abs((x - mb) - (y - mn)) for x, y in zip(b, n)) / len(keys)


def top(d, keys):
    return max(keys, key=lambda k: d[k])


def run(model, tok, cfg, device, n_perm, seed):
    rng = random.Random(seed)
    rows = {m: [] for m in ("permute", "remove", "add_irrel", "duplicate")}
    skipped = 0
    for row in load_dataset("LocalLLaMA/typed-decisions", "all", split="test"):
        state, questions = json.loads(row["state"]), json.loads(row["questions"])
        for q in questions.values():
            if q["type"] != "choice" or len(q["criteria"]) < 3:
                continue
            crit, ins = dict(q["criteria"]), q["instructions"]
            base = log_probs(model, tok, cfg, state, ins, crit, device)
            if base is None:
                skipped += 1
                continue
            keys = list(crit)
            first = top(base, keys)

            for _ in range(n_perm):
                order = keys[:]
                rng.shuffle(order)
                new = log_probs(model, tok, cfg, state, ins, {k: crit[k] for k in order}, device)
                if new:
                    rows["permute"].append(record(base, new, keys))

            drop = rng.choice([k for k in keys if k != first])
            kept = [k for k in keys if k != drop]
            new = log_probs(model, tok, cfg, state, ins, {k: crit[k] for k in kept}, device)
            if new:
                rows["remove"].append(record(base, new, kept))

            if IRRELEVANT[0] not in crit:
                new = log_probs(model, tok, cfg, state, ins, {**crit, IRRELEVANT[0]: IRRELEVANT[1]}, device)
                if new:
                    r = record(base, new, keys)
                    r["new_option_p"] = math.exp(new[IRRELEVANT[0]])
                    rows["add_irrel"].append(r)

            dup_key = first + "_copy"
            new = log_probs(model, tok, cfg, state, ins, {**crit, dup_key: crit[first]}, device)
            if new:
                r = record(base, new, keys)
                # 紅巴士測試：兩個複製品加起來 vs 原本第一名；以及兩者是否平分
                r["dup_sum_minus_orig"] = math.exp(new[first]) + math.exp(new[dup_key]) - math.exp(base[first])
                r["dup_share_of_pair"] = math.exp(new[dup_key]) / (math.exp(new[first]) + math.exp(new[dup_key]))
                rows["duplicate"].append(r)
    return rows, skipped


def record(base, new, keys):
    up = max(math.exp(new[k]) - math.exp(base[k]) for k in keys)
    return {"shift": ratio_shift(base, new, keys),
            "flip": top(base, keys) != top(new, keys),
            "regularity_violation": up > 0.01,  # 任何原有選項機率上升超過 1 個百分點
            "max_increase": up}


def summarize(rows):
    out = {}
    for m, rs in rows.items():
        if not rs:
            continue
        s = {"n": len(rs),
             "shift_mean": st.mean(r["shift"] for r in rs),
             "shift_median": st.median(r["shift"] for r in rs),
             "flip_rate": st.mean(r["flip"] for r in rs),
             "regularity_violation_rate": st.mean(r["regularity_violation"] for r in rs)}
        for extra in ("new_option_p", "dup_sum_minus_orig", "dup_share_of_pair"):
            if extra in rs[0]:
                s[extra + "_mean"] = st.mean(r[extra] for r in rs)
        out[m] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", default="./laya_hf")
    ap.add_argument("--n-perm", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="iia_pilot_results.json")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.2)
    device = torch.device("cuda")
    results = {}
    for name, sub in CHECKPOINTS.items():
        model, tok, cfg = load(args.model_root, sub, device)
        rows, skipped = run(model, tok, cfg, device, args.n_perm, args.seed)
        results[name] = {"skipped_truncated": skipped, "summary": summarize(rows), "rows": rows}
        print(f"\n##### {name}（截斷排除 {skipped} 題）")
        for m, s in results[name]["summary"].items():
            print(f"  {m:10s} " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in s.items()))
        del model
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
