"""Mitigation study: runner. Frozen at preregistration.

  python mitig/src/run_mitig.py run --model claude --role smoke --limit 1
  python mitig/src/run_mitig.py run --model gemini --role sample
  python mitig/src/run_mitig.py recognize --model claude --role sample
  python mitig/src/run_mitig.py status

Each job is one paper x relation x request x wording, a fresh conversation (prompts_mitig.first_turn); nothing is
shared between jobs. The 12 jobs of every paper are shuffled together with a fixed seed, so time on the API is not
aligned with relation, request or wording.
  claude: one turn, the review form; valid if the form is complete (main study's review_check("struct")).
  gemini: the free-text answer (valid if > 200 characters and not cut off, review_check("open")), then the
          SELFSCORE turn on that answer (valid if rating in {0,...,10} even and recommendation accept/reject).
          The answer is stored first; a rerun reuses the first valid stored answer and only asks for the score.
A cell counts only if the API reports the preregistered model. Reruns redo invalid cells only; the first valid
record of a cell is the one analysed. Runs from the main study's directory (src/run.py Client, config.yaml, .env).
"""
import argparse
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from prompts_mitig import ASKS, FAMILIES, SELFSCORE, WHO, condition, first_turn  # noqa: E402
from run import Client, load_config, parse_json  # noqa: E402
from run_p2 import FatalGuard, call_with_retry, read, review_check  # noqa: E402

M = ROOT / "mitig"
SEED = 20261105
EXPECTED = {"claude": "claude-sonnet-4-6", "gemini": "gemini-4-argon"}
RATINGS = {0, 2, 4, 6, 8, 10}


def load_docs(role):
    return [d for d in map(json.loads, open(M / "data" / "manuscripts.jsonl")) if d["role"] == role]


def path(model, kind):
    return M / "results" / f"{kind}_{model}.jsonl"


def served(rec, model):
    return str(rec.get("provider_model") or "").startswith(EXPECTED[model])


def h(text):
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


def score_check(out):
    o = parse_json(out["text"])
    ok = isinstance(o, dict) and o.get("rating") in RATINGS and str(o.get("recommendation")).lower() in ("accept", "reject")
    return ok, o


def first_valid(model, kind):
    out = {}
    for r in read(path(model, kind)):
        if r.get("ok") and served(r, model):
            out.setdefault((r["paper_id"], r["condition"]), r)
    return out


async def run(args):
    model = args.model
    cfg = load_config()
    client = Client(model, cfg["models"][model], cfg.get("defaults", {}))
    docs = load_docs(args.role)[: args.limit or None]
    kind = "form" if model == "claude" else "score"
    done = first_valid(model, kind)
    answers = first_valid(model, "open") if model == "gemini" else {}
    jobs = [(d, who, ask, fam) for d in docs for who in WHO for ask in ASKS for fam in FAMILIES
            if (d["paper_id"], condition(who, ask, fam)) not in done]
    random.Random(f"{SEED}-{model}-{args.role}").shuffle(jobs)
    print(f"[mitig:{model}:{args.role}] {len(jobs)} jobs to run, {len(done)} done", flush=True)
    guard = FatalGuard(f"mitig-{model}")
    n = {"k": 0}
    M.joinpath("results").mkdir(parents=True, exist_ok=True)
    base = lambda d, who, ask, fam: {"model": model, "model_id": client.c["model"], "paper_id": d["paper_id"], "role": d["role"],
                                     "who": who, "ask": ask, "family": fam, "condition": condition(who, ask, fam)}
    with open(path(model, kind), "a") as fout, open(path(model, "open"), "a") as fopen:
        async def job(d, who, ask, fam):
            cond = condition(who, ask, fam)
            user1 = {"role": "user", "content": first_turn(who, ask, fam, d["title"], d["text"], form=(model == "claude"))}
            if model == "claude":
                async with client.sem:
                    if guard.stopped:
                        return
                    rec = await call_with_retry(client, [user1], int(client.c.get("max_tokens_review", cfg["max_tokens_review"])),
                                                review_check("struct"))
                guard.see(rec)
                o = rec.get("parsed") or {}
                fout.write(json.dumps({**base(d, who, ask, fam), "rating": o.get("rating") if rec.get("ok") else None,
                                       "recommendation": str(o.get("recommendation")).lower() if rec.get("ok") else None,
                                       **rec}, ensure_ascii=False) + "\n")
                fout.flush()
            else:
                ans = answers.get((d["paper_id"], cond))
                if ans is None:
                    async with client.sem:
                        if guard.stopped:
                            return
                        rec = await call_with_retry(client, [user1], int(client.c.get("max_tokens_open", cfg["max_tokens_open"])),
                                                    review_check("open"))
                    guard.see(rec)
                    fopen.write(json.dumps({**base(d, who, ask, fam), "hash": h(rec.get("raw")), **rec}, ensure_ascii=False) + "\n")
                    fopen.flush()
                    if not (rec.get("ok") and served(rec, model)):
                        return
                    ans = rec
                msgs = [user1, {"role": "assistant", "content": ans["raw"]}, {"role": "user", "content": SELFSCORE}]
                async with client.sem:
                    if guard.stopped:
                        return
                    rec = await call_with_retry(client, msgs, 8000, score_check)
                guard.see(rec)
                o = rec.get("parsed") or {}
                fout.write(json.dumps({**base(d, who, ask, fam), "hash": h(ans["raw"]),
                                       "rating": o.get("rating") if rec.get("ok") else None,
                                       "recommendation": str(o.get("recommendation")).lower() if rec.get("ok") else None,
                                       **rec}, ensure_ascii=False) + "\n")
                fout.flush()
            n["k"] += 1
            if n["k"] % 20 == 0 or n["k"] == len(jobs):
                print(f"[mitig:{model}:{args.role}] {n['k']}/{len(jobs)}", flush=True)
        await asyncio.gather(*[job(*j) for j in jobs])
    await client.http.aclose()


async def recognize(args):
    """Recognition check (secondary): the main study's RECOGNIZE probe on the first 2,500 tokens of the
    manuscript as shown. Output: mitig/results/recognize_<model>.jsonl."""
    from prompts import RECOGNIZE
    from prepare import ENC
    model = args.model
    cfg = load_config()
    client = Client(model, cfg["models"][model], cfg.get("defaults", {}))
    p = path(model, "recognize")
    docs = load_docs(args.role)[: args.limit or None]
    done = {r["paper_id"] for r in read(p) if r.get("ok")}
    docs = [d for d in docs if d["paper_id"] not in done]

    def check(out):
        o = parse_json(out["text"])
        return isinstance(o, dict) and isinstance(o.get("recognized"), bool), o

    with open(p, "a") as f:
        async def one(d):
            opening = ENC.decode(ENC.encode(f"# {d['title']}\n\n{d['text']}")[:2500])
            async with client.sem:
                rec = await call_with_retry(client, [{"role": "user", "content": RECOGNIZE.format(opening=opening)}], 8000, check)
            f.write(json.dumps({"model": model, "paper_id": d["paper_id"], "role": d["role"], **rec}, ensure_ascii=False) + "\n")
            f.flush()
        await asyncio.gather(*[one(d) for d in docs])
    await client.http.aclose()
    print(f"[recognize:{model}] {len(docs)} calls")


def status():
    for model, kind in (("claude", "form"), ("gemini", "score")):
        ok = first_valid(model, kind)
        by = {}
        for r in ok.values():
            by[r["role"]] = by.get(r["role"], 0) + 1
        print(f"{model}: valid cells {by} (sample target 2880) | records {len(read(path(model, kind)))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "recognize", "status"])
    ap.add_argument("--model", choices=["claude", "gemini"], default="claude")
    ap.add_argument("--role", choices=["smoke", "sample"], default="smoke")
    ap.add_argument("--limit", type=int, default=0, help="only the first N papers of the role (smoke test)")
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a))
    elif a.cmd == "recognize":
        asyncio.run(recognize(a))
    else:
        status()
