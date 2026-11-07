"""Confirmatory study: runner. Frozen at preregistration.

  python confirm/src/run_confirm.py run --role smoke --limit 1     # smoke-test papers, not analysed
  python confirm/src/run_confirm.py run --role sample              # the 160 preregistered papers
  python confirm/src/run_confirm.py recognize --role sample        # recognition check (secondary)
  python confirm/src/run_confirm.py status

Model: Claude Sonnet 4.6 (config entry "claude"). Each job is one paper x one relation x one wording, sent as a
fresh one-turn conversation (prompts_confirm.message); nothing is shared between jobs. All 12 jobs of every paper
are shuffled together with a fixed seed, so time on the API is not aligned with relation or wording.
Output, one line per attempt, keyed by (paper_id, condition): confirm/results/confirm_claude.jsonl. A cell counts
as done only if the answer is a valid review form and the API reports the preregistered model; a rerun redoes
the rest. Runs from the main study's directory (src/run.py Client, config.yaml, .env).
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from prompts_confirm import FAMILIES, WHO, condition, message  # noqa: E402
from run import Client, load_config, parse_json  # noqa: E402
from run_p2 import FatalGuard, call_with_retry, read, review_check  # noqa: E402

C = ROOT / "confirm"
MODEL = "claude"
SEED = 20261104


def load_docs(role):
    return [d for d in map(json.loads, open(C / "data" / "manuscripts.jsonl")) if d["role"] == role]


def out_path():
    return C / "results" / f"confirm_{MODEL}.jsonl"


def served_ok(rec, model_id):
    return str(rec.get("provider_model") or "").startswith(model_id)


async def run(args):
    cfg = load_config()
    client = Client(MODEL, cfg["models"][MODEL], cfg.get("defaults", {}))
    max_tokens = int(client.c.get("max_tokens_review", cfg["max_tokens_review"]))
    docs = load_docs(args.role)[: args.limit or None]
    done = {(r["paper_id"], r["condition"]) for r in read(out_path())
            if r.get("ok") and served_ok(r, client.c["model"])}
    jobs = [(d, who, fam) for d in docs for who in WHO for fam in FAMILIES
            if (d["paper_id"], condition(who, fam)) not in done]
    random.Random(f"{SEED}-{MODEL}-{args.role}").shuffle(jobs)
    print(f"[confirm:{args.role}] {len(jobs)} jobs to run, {len(done)} done", flush=True)
    guard = FatalGuard("confirm")
    n = {"k": 0}
    out_path().parent.mkdir(parents=True, exist_ok=True)
    with open(out_path(), "a") as f:
        async def job(d, who, fam):
            msgs = [{"role": "user", "content": message(who, fam, d["title"], d["text"])}]
            async with client.sem:
                if guard.stopped:
                    return
                rec = await call_with_retry(client, msgs, max_tokens, review_check("struct"))
            guard.see(rec)
            o = rec.get("parsed") or {}
            f.write(json.dumps({"model": MODEL, "model_id": client.c["model"], "paper_id": d["paper_id"], "role": d["role"],
                                "who": who, "family": fam, "condition": condition(who, fam),
                                "rating": o.get("rating") if rec.get("ok") else None,
                                "recommendation": str(o.get("recommendation")).lower() if rec.get("ok") else None,
                                **rec}, ensure_ascii=False) + "\n")
            f.flush()
            n["k"] += 1
            if n["k"] % 20 == 0 or n["k"] == len(jobs):
                print(f"[confirm:{args.role}] {n['k']}/{len(jobs)}", flush=True)
        await asyncio.gather(*[job(*j) for j in jobs])
    await client.http.aclose()


async def recognize(args):
    """Recognition check (secondary): the main study's RECOGNIZE probe on the first 2,500 tokens of the
    manuscript as shown. Output: confirm/results/recognize_claude.jsonl."""
    from prompts import RECOGNIZE
    from prepare import ENC
    cfg = load_config()
    client = Client(MODEL, cfg["models"][MODEL], cfg.get("defaults", {}))
    path = C / "results" / f"recognize_{MODEL}.jsonl"
    docs = load_docs(args.role)[: args.limit or None]
    done = {r["paper_id"] for r in read(path) if r.get("ok")}
    docs = [d for d in docs if d["paper_id"] not in done]

    def check(out):
        o = parse_json(out["text"])
        return isinstance(o, dict) and isinstance(o.get("recognized"), bool), o

    with open(path, "a") as f:
        async def one(d):
            opening = ENC.decode(ENC.encode(f"# {d['title']}\n\n{d['text']}")[:2500])
            async with client.sem:
                rec = await call_with_retry(client, [{"role": "user", "content": RECOGNIZE.format(opening=opening)}], 2000, check)
            f.write(json.dumps({"model": MODEL, "paper_id": d["paper_id"], "role": d["role"], **rec}, ensure_ascii=False) + "\n")
            f.flush()
        await asyncio.gather(*[one(d) for d in docs])
    await client.http.aclose()
    print(f"[recognize] {len(docs)} calls")


def status():
    cfg = load_config()
    mid = cfg["models"][MODEL]["model"]
    rows = read(out_path())
    ok = {(r["role"], r["paper_id"], r["condition"]) for r in rows if r.get("ok") and served_ok(r, mid)}
    by = {}
    for role, _, _ in ok:
        by[role] = by.get(role, 0) + 1
    print(out_path().name, "valid cells:", by, "(sample target 1920) | records:", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "recognize", "status"])
    ap.add_argument("--role", choices=["smoke", "sample"], default="smoke")
    ap.add_argument("--limit", type=int, default=0, help="only the first N papers of the role (smoke test)")
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a))
    elif a.cmd == "recognize":
        asyncio.run(recognize(a))
    else:
        status()
