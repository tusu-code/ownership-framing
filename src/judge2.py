"""Second blind judge (Claude Haiku 4.5) on a stratified subsample of neutralized open answers.

  python src/judge2.py   -> results/p3_judge_haiku.jsonl (keyed by the hash of the original answer)

Same JUDGE prompt and validity check as the main judge; reads only the neutralized text.
"""
import asyncio
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import JUDGE  # noqa: E402
from quick_check import answers, text_hash  # noqa: E402
from run import Client, load_config, parse_json  # noqa: E402
from run_p2 import FatalGuard, call_with_retry, valid_judge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, DATA = ROOT / "results", ROOT / "data"
OUT = RES / "p3_judge_haiku.jsonl"
ASKS = {"claude": ["open_author", "open_colleague", "open_venue", "open_reviewer"],
        "glm": ["open_author", "open_colleague", "open_venue", "open_reviewer"],
        "deepseek": ["open_author", "open_reviewer"]}
N_PAPERS = 170


def sample_papers():
    s = pd.read_csv(DATA / "sample_full.csv")
    pick = (s.groupby("stratum", group_keys=False)[["paper_id", "stratum"]]
              .apply(lambda g: g.sample(n=max(1, round(N_PAPERS * len(g) / len(s))), random_state=20261102)))
    return set(pick.paper_id)


async def main():
    cfg = load_config()
    A = answers()
    neu = {}
    for line in open(RES / "p3_neutral.jsonl"):
        r = json.loads(line)
        if r.get("ok"):
            neu[r["hash"]] = r["text"]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["hash"] for l in open(OUT) if json.loads(l).get("ok")}
    papers = sample_papers()
    jobs = []
    for (m, pid, cond), r in A.items():
        if m in ASKS and cond in ASKS[m] and pid in papers:
            h = text_hash(r["raw"])
            if h in neu and h not in done:
                jobs.append((m, pid, cond, h, neu[h]))
    random.Random(20261102).shuffle(jobs)
    client = Client("haiku", cfg["models"]["haiku"], cfg.get("defaults", {}))
    model_id = client.c["model"]
    print(f"[judge2:{model_id}] {len(jobs)} answers ({len(done)} done, {len(papers)} papers)", flush=True)

    def check(out):
        obj = parse_json(out["text"])
        return valid_judge(obj), obj

    guard = FatalGuard("judge2")
    n = {"k": 0}
    with open(OUT, "a") as f:
        async def job(m, pid, cond, h, text):
            async with client.sem:
                if guard.stopped:
                    return
                rec = await call_with_retry(client, [{"role": "user", "content": JUDGE.format(response=text)}], 3000, check)
            guard.see(rec)
            f.write(json.dumps({"model": m, "paper_id": pid, "condition": cond, "hash": h, "judge_model": model_id,
                                "ok": rec.get("ok"), "parsed": rec.get("parsed"), "error": rec.get("error"),
                                "key_slot": rec.get("key_slot")}, ensure_ascii=False) + "\n")
            f.flush()
            n["k"] += 1
            done_k = n["k"]
            if done_k % 100 == 0 or done_k == len(jobs):
                print(f"[judge2] {done_k}/{len(jobs)}", flush=True)
        await asyncio.gather(*[job(*j) for j in jobs])
    await client.http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
