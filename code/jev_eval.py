"""用 Jev API 跑與開源模型完全相同的題目與操作（iia_v3 的 conditions、a2_dose 的劑量反應）。

Jev 只回傳四捨五入到小數兩位的校準後機率，所以 scores 存 log(max(p, 0.005))、溫度記為 1；
原始機率另存 probs。同一段狀態的基準與各操作放在同一個請求裡（Jev 各題獨立平行處理）。
另加兩種雜訊對照：retest_in（同請求重複同一題）、retest_out（跨請求重送，每資料集 100 題）。
金鑰只從環境變數 TYPESAFE_API_KEY 讀取；累計花費超過 --budget 美元即停止。
"""
import argparse
import asyncio
import json
import math
import os
import random
from pathlib import Path

import httpx

URL = "https://api.typesafe.ai/v1/systemone"
PRICE_PER_MTOK = 0.042
FLOOR = 0.005  # 四捨五入到 0.01，0 視為「小於 0.005」


def question(ins, crit):
    return {"type": "choice", "instructions": ins, "criteria": {k: v for k, v in crit}}


def to_scores(probs):
    return {k: math.log(max(float(p), FLOOR)) for k, p in probs.items()}


class Client:
    def __init__(self, budget, concurrency):
        self.http = httpx.AsyncClient(timeout=60, headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"})
        self.sem = asyncio.Semaphore(concurrency)
        self.budget, self.tokens, self.calls, self.errors = budget, 0, 0, 0
        self.model_versions = set()

    @property
    def cost(self):
        return self.tokens / 1e6 * PRICE_PER_MTOK

    async def ask(self, state, questions):
        if self.cost > self.budget:
            raise RuntimeError(f"budget exceeded: ${self.cost:.3f}")
        body = {"model": "jev-latest", "state": state, "questions": questions}
        async with self.sem:
            for attempt in range(6):
                r = await self.http.post(URL, json=body)
                if r.status_code in (429, 529, 500, 502, 503):
                    await asyncio.sleep(2 ** attempt + random.random())
                    continue
                if r.status_code != 200:
                    self.errors += 1
                    print(f"  HTTP {r.status_code}: {r.text[:200]}", flush=True)
                    return None
                d = r.json()
                self.calls += 1
                self.tokens += sum(d.get("usage", {}).values())
                self.model_versions.add(d.get("model"))
                return {qid: a["probabilities"] for qid, a in d["answers"].items()}
            self.errors += 1
            return None


async def run_v3(task, client, retest_n=100):
    """與 iia_v3 相同的題目與操作。"""
    import iia_v3
    from iia_v2 import irrelevant_pool, load_questions
    qs = load_questions() if task == "typed" else iia_v3.intent_questions(task)
    pool = irrelevant_pool() if task == "typed" else []
    siblings = json.loads(Path("siblings.json").read_text()) if task == "typed" else {}

    async def one(qi, q):
        rng = random.Random(qi)  # 與 iia_v3 相同的種子 → 相同的操作
        conds = iia_v3.conditions(task, q, pool, siblings, rng)
        same_state = {"base": question(q["ins"], q["crit"]), "retest_in": question(q["ins"], q["crit"])}
        other_state = []
        for j, (name, state, ins, crit, rel) in enumerate(conds):
            if state == q["state"]:
                same_state[f"c{j}"] = question(ins, crit)
            else:
                other_state.append((j, state, ins, crit))
        ans = await client.ask(q["state"], same_state)
        if ans is None:
            return None
        extra = {}
        for j, state, ins, crit in other_state:
            a = await client.ask(state, {f"c{j}": question(ins, crit)})
            if a:
                extra.update(a)
        if qi < retest_n:
            a = await client.ask(q["state"], {"base": question(q["ins"], q["crit"])})
            if a:
                extra["retest_out"] = a["base"]
        ans.update(extra)
        keys = [k for k, _ in q["crit"]]
        rec = {"qid": q["qid"], "anchor": q["anchor"], "keys": keys,
               "base": {"scores": to_scores(ans["base"]), "temp": 1.0, "probs": ans["base"]}, "conds": []}
        for name, src in (("retest_in", "retest_in"), ("retest_out", "retest_out")):
            if src in ans:
                rec["conds"].append({"cond": name, "rel": keys, "scores": to_scores(ans[src]), "temp": 1.0,
                                     "probs": ans[src]})
        for j, (name, state, ins, crit, rel) in enumerate(conds):
            if f"c{j}" in ans:
                rec["conds"].append({"cond": name, "rel": rel, "scores": to_scores(ans[f"c{j}"]), "temp": 1.0,
                                     "probs": ans[f"c{j}"]})
        return rec

    recs = await asyncio.gather(*(one(i, q) for i, q in enumerate(qs)))
    return [r for r in recs if r]


async def run_a2(task, client):
    """與 a2_dose 相同的劑量反應：一句話的 K=2..12 基準與加 1 個，共 22 題放同一個請求。"""
    import a2_dose
    from iia_v3 import humanize

    async def one(u):
        qs, meta = {}, {}
        for k in a2_dose.K_RANGE:
            labels = [u["gold"]] + u["chain"][:k - 1]
            random.Random(f"{u['qid']}:{k}").shuffle(labels)
            crit = [(lab, humanize(lab)) for lab in labels]
            added = (u["chain"][k - 1], humanize(u["chain"][k - 1]))
            qs[f"b{k}"], qs[f"n{k}"] = question(u["ins"], crit), question(u["ins"], crit + [added])
            meta[k] = labels
        ans = await client.ask(u["state"], qs)
        if ans is None:
            return []
        return [{"qid": f"{u['qid']}:k{k}", "K": k, "anchor": u["gold"], "keys": labels,
                 "base": {"scores": to_scores(ans[f"b{k}"]), "temp": 1.0, "probs": ans[f"b{k}"]},
                 "conds": [{"cond": f"k{k}", "rel": labels, "scores": to_scores(ans[f"n{k}"]), "temp": 1.0,
                            "probs": ans[f"n{k}"]}]}
                for k, labels in meta.items() if f"b{k}" in ans and f"n{k}" in ans]

    out = await asyncio.gather(*(one(u) for u in a2_dose.utterances(task)))
    return [r for rs in out for r in rs]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["v3", "a2"], required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    client = Client(args.budget, args.concurrency)
    recs = await (run_v3 if args.suite == "v3" else run_a2)(args.task, client)
    out_dir = Path("iia_v3_results" if args.suite == "v3" else "a2_results")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.task}__jev.json"
    out.write_text(json.dumps({"backend": "jev", "task": args.task, "model_versions": sorted(client.model_versions),
                               "records": recs}, ensure_ascii=False))
    print(f"jev/{args.suite}/{args.task}: records={len(recs)} calls={client.calls} errors={client.errors} "
          f"tokens={client.tokens} cost=${client.cost:.3f} versions={sorted(client.model_versions)} -> {out}")
    await client.http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
