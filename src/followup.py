"""Reviewer M1(a): the evaluated model scores its own open answer in a second turn.

  python src/followup.py --model gemini --sample sub80.csv --cond open_author --cond open_reviewer [--rep 0]

The conversation is the original open ask, the model's stored answer (verbatim) and SELFSCORE, so the
score comes from the model that wrote the answer, without any judge. The follow-up wording is identical
under every ask. Each call is still a fresh conversation that contains only this paper and this answer.
Output: results/p4_selfscore.jsonl keyed by (model, paper, condition, rep, hash of the answer).
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import P2_CONDITIONS, SELFSCORE  # noqa: E402
from quick_check import text_hash  # noqa: E402
from run import Client, load_config, parse_json  # noqa: E402
from run_p2 import FatalGuard, call_with_retry, load_docs, read, review_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
OUT = RES / "p4_selfscore.jsonl"
RATINGS = {0, 2, 4, 6, 8, 10}


def check(out):
    o = parse_json(out["text"])
    ok = isinstance(o, dict) and o.get("rating") in RATINGS and str(o.get("recommendation")).lower() in ("accept", "reject")
    return ok, o


def stored_answers(model, ids, conds, rep):
    """Latest valid answer per (paper, condition) at this rep on the current relay."""
    out = {}
    for r in read(review_path(model)):
        if (r.get("ok") and r["rep"] == rep and r["paper_id"] in ids and r["condition"] in conds
                and str(r.get("relay", "")).startswith("https://api.sailapi")):
            out[(r["paper_id"], r["condition"])] = r
    return out


async def main(args):
    cfg = load_config()
    docs = load_docs()
    ids = set(pd.read_csv(DATA / args.sample).paper_id) & set(docs)
    A = stored_answers(args.model, ids, set(args.cond), args.rep)
    done = {(r["paper_id"], r["condition"], r["rep"], r["hash"]) for r in read(OUT)
            if r.get("ok") and r["model"] == args.model}
    jobs = [(pid, cond, r) for (pid, cond), r in A.items()
            if (pid, cond, args.rep, text_hash(r["raw"])) not in done]
    random.Random(f"followup-{args.model}").shuffle(jobs)
    client = Client(args.model, cfg["models"][args.model], cfg.get("defaults", {}))
    print(f"[followup:{args.model}] {len(jobs)} calls ({len(done)} done, {len(A)} stored answers)", flush=True)
    guard = FatalGuard(f"followup-{args.model}")
    n = {"k": 0}
    with open(OUT, "a") as f:
        async def job(pid, cond, r):
            d = docs[pid]
            msgs = P2_CONDITIONS[cond][0](d["title"], d["text"]) + [
                {"role": "assistant", "content": r["raw"]}, {"role": "user", "content": SELFSCORE}]
            async with client.sem:
                if guard.stopped:
                    return
                rec = await call_with_retry(client, msgs, 8000, check)
            guard.see(rec)
            f.write(json.dumps({"model": args.model, "paper_id": pid, "condition": cond, "rep": args.rep,
                                "hash": text_hash(r["raw"]), "model_id": client.c["model"], **rec}, ensure_ascii=False) + "\n")
            f.flush()
            n["k"] += 1
            if n["k"] % 20 == 0 or n["k"] == len(jobs):
                print(f"[followup:{args.model}] {n['k']}/{len(jobs)}", flush=True)
        await asyncio.gather(*[job(*j) for j in jobs])
    await client.http.aclose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--cond", action="append", required=True)
    ap.add_argument("--rep", type=int, default=0)
    asyncio.run(main(ap.parse_args()))
