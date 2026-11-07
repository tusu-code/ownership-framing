"""Extension studies A and B (exploratory): runner.

  python extend/src/run_extend.py run --study A --role smoke     # one paper per field, not analysed
  python extend/src/run_extend.py run --study A --role sample
  python extend/src/run_extend.py run --study B --role sample
  python extend/src/run_extend.py status

Model: Claude Sonnet 4.6 (config entry "claude"), review form, one fresh one-turn conversation per paper x ask.
All jobs of a study are shuffled together with a fixed seed, so time on the API is not aligned with the ask or
the field. Output, one line per attempt: extend/results/{A,B}_claude.jsonl. A cell counts if the form is valid
and the API reports claude-sonnet-4-6; reruns redo the rest. Runs from the main study's directory.
Papers:
  A  extend/data/manuscripts.jsonl (biology, physics, economics) and, as the same-batch ML control, the
     mitigation study's sample papers N_B .. N_B+ML_CONTROL-1 (no overlap with study B).
  B  the mitigation study's first N_B sample papers.
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
from prompts_extend import ASKS_A, ASKS_B, ML_CONTROL, N_B, message  # noqa: E402
from run import Client, load_config  # noqa: E402
from run_p2 import FatalGuard, call_with_retry, read, review_check  # noqa: E402

E = ROOT / "extend"
MODEL, EXPECTED = "claude", "claude-sonnet-4-6"
SEED = 20261106


def ml_papers(role="sample"):
    return [d for d in map(json.loads, open(ROOT / "mitig" / "data" / "manuscripts.jsonl")) if d["role"] == role]


def load_docs(study, role):
    if study == "B":
        return [{**d, "field": "ml"} for d in ml_papers()[:N_B]] if role == "sample" else [{**ml_papers("smoke")[0], "field": "ml"}]
    docs = [d for d in map(json.loads, open(E / "data" / "manuscripts.jsonl")) if d["role"] == role]
    if role == "sample":
        docs += [{**d, "field": "ml"} for d in ml_papers()[N_B:N_B + ML_CONTROL]]
    return docs


def out_path(study):
    return E / "results" / f"{study}_{MODEL}.jsonl"


def done_cells(study, role):
    return {(r["paper_id"], r["condition"]) for r in read(out_path(study))
            if r.get("role") == role and r.get("ok") and str(r.get("provider_model") or "").startswith(EXPECTED)}


async def run(args):
    cfg = load_config()
    client = Client(MODEL, cfg["models"][MODEL], cfg.get("defaults", {}))
    max_tokens = int(client.c.get("max_tokens_review", cfg["max_tokens_review"]))
    asks = ASKS_A if args.study == "A" else ASKS_B
    docs = load_docs(args.study, args.role)[: args.limit or None]
    done = done_cells(args.study, args.role)
    jobs = [(d, who) for d in docs for who in asks if (d["paper_id"], who) not in done]
    random.Random(f"{SEED}-{args.study}-{args.role}").shuffle(jobs)
    print(f"[extend:{args.study}:{args.role}] {len(jobs)} jobs to run, {len(done)} done", flush=True)
    guard = FatalGuard(f"extend-{args.study}")
    n = {"k": 0}
    out_path(args.study).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path(args.study), "a") as f:
        async def job(d, who):
            msgs = [{"role": "user", "content": message(args.study, who, d["title"], d["text"])}]
            async with client.sem:
                if guard.stopped:
                    return
                rec = await call_with_retry(client, msgs, max_tokens, review_check("struct"))
            guard.see(rec)
            o = rec.get("parsed") or {}
            f.write(json.dumps({"study": args.study, "model": MODEL, "model_id": client.c["model"], "paper_id": d["paper_id"],
                                "field": d["field"], "role": args.role, "who": who, "condition": who,
                                "rating": o.get("rating") if rec.get("ok") else None,
                                "recommendation": str(o.get("recommendation")).lower() if rec.get("ok") else None,
                                **rec}, ensure_ascii=False) + "\n")
            f.flush()
            n["k"] += 1
            if n["k"] % 20 == 0 or n["k"] == len(jobs):
                print(f"[extend:{args.study}:{args.role}] {n['k']}/{len(jobs)}", flush=True)
        await asyncio.gather(*[job(*j) for j in jobs])
    await client.http.aclose()


def status():
    for s in ("A", "B"):
        rows = read(out_path(s))
        ok = [r for r in rows if r.get("ok") and str(r.get("provider_model") or "").startswith(EXPECTED)]
        by = {}
        for r in ok:
            by[(r["role"], r["field"])] = by.get((r["role"], r["field"]), 0) + 1
        print(s, "valid cells:", dict(sorted(by.items())), "| records:", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--study", choices=["A", "B"], default="A")
    ap.add_argument("--role", choices=["smoke", "sample"], default="smoke")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(run(a)) if a.cmd == "run" else status()
