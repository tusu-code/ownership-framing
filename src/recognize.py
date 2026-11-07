"""Reviewer M6: can a model name the original paper from its opening (about 2,500 tokens)?

  python src/recognize.py --model claude      -> results/p4_recognize.jsonl
Every paper in the full sample is shown as the models saw it (disguised); a 61-paper subset (sub60.csv) is
also shown in its original form as a positive control. Scoring (title overlap, decision) is done in analysis.
"""
import argparse, asyncio, json, random, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import RECOGNIZE, render_submission  # noqa: E402
from run import Client, load_config, parse_json  # noqa: E402
from run_p2 import ENC, FatalGuard, call_with_retry, load_docs, read  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
OUT = RES / "p4_recognize.jsonl"
N_TOK = 2500


def opening(title, text):
    return ENC.decode(ENC.encode(render_submission(title, text))[:N_TOK])


def check(out):
    o = parse_json(out["text"])
    return isinstance(o, dict) and isinstance(o.get("recognized"), bool), o


async def main(model):
    cfg = load_config()
    docs = load_docs()
    orig = {r["paper_id"]: r for r in read(DATA / "manuscripts.jsonl")}
    full = [p for p in pd.read_csv(DATA / "sample_full.csv").paper_id if p in docs]
    jobs = [(p, "disguised", opening(docs[p]["title"], docs[p]["text"])) for p in full]
    jobs += [(p, "original", opening(orig[p]["title"], orig[p]["text"])) for p in pd.read_csv(DATA / "sub60.csv").paper_id if p in orig]
    done = {(r["paper_id"], r["version"]) for r in read(OUT) if r.get("ok") and r["model"] == model}
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    random.Random(f"recognize-{model}").shuffle(jobs)
    client = Client(model, cfg["models"][model], cfg.get("defaults", {}))
    print(f"[recognize:{model}] {len(jobs)} calls ({len(done)} done)", flush=True)
    guard = FatalGuard(f"recognize-{model}")
    with open(OUT, "a") as f:
        async def job(pid, version, text):
            async with client.sem:
                if guard.stopped:
                    return
                rec = await call_with_retry(client, [{"role": "user", "content": RECOGNIZE.format(opening=text)}], 4000, check)
            guard.see(rec)
            f.write(json.dumps({"model": model, "paper_id": pid, "version": version, "model_id": client.c["model"], **rec}, ensure_ascii=False) + "\n")
            f.flush()
        await asyncio.gather(*[job(*j) for j in jobs])
    await client.http.aclose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
    asyncio.run(main(ap.parse_args().model))
