"""Reviewer M2: an independent-family judge (MiniMax M3.1 Flash) on every main free-text answer.

  python src/judge3.py [neutral] [raw]   -> results/p4_judge_minimax.jsonl, keyed by (hash of the original answer, input)
input = "neutral": the neutralized text, as the main judge read it (rewriter: DeepSeek V4.1 Flash)
input = "raw":     the original answer, so that no DeepSeek-family model sits between answer and score
Covers rep 0 of every free-text condition, plus the same-batch re-run (rep 2) and the DeepSeek re-ask (rep 3);
the 100-paper retest (rep 1) is included since 15:20 (main judge switched to MiniMax). Rerun at any time; finished work is skipped.
"""
import asyncio, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import JUDGE  # noqa: E402
from quick_check import answers, text_hash  # noqa: E402
from run import Client, load_config, parse_json  # noqa: E402
from run_p2 import FatalGuard, call_with_retry, read, valid_judge  # noqa: E402
from run_p3_judge import OPENLIKE  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "p4_judge_minimax.jsonl"


def check(out):
    o = parse_json(out["text"])
    return valid_judge(o), o


async def main(inputs):
    cfg = load_config()
    neu = {r["hash"]: r["text"] for r in read(RES / "p3_neutral.jsonl") if r.get("ok")}
    done = {(r["hash"], r["input"]) for r in read(OUT) if r.get("ok")}
    jobs = []
    for (m, pid, cond), r in answers().items():
        base, _, rep = cond.partition("@r")
        if base.split("_")[0] not in OPENLIKE or rep not in ("", "1", "2", "3"):
            continue
        h = text_hash(r["raw"])
        for inp, text in (("neutral", neu.get(h)), ("raw", r["raw"])):
            if text and inp in inputs and (h, inp) not in done:
                jobs.append((m, pid, cond, h, inp, text))
    random.Random(20261103).shuffle(jobs)
    client = Client("minimax", cfg["models"]["minimax"], cfg.get("defaults", {}))
    print(f"[judge3:{client.c['model']}] {len(jobs)} calls ({len(done)} done)", flush=True)
    guard = FatalGuard("judge3")
    n = {"k": 0}
    with open(OUT, "a") as f:
        async def job(m, pid, cond, h, inp, text):
            async with client.sem:
                if guard.stopped:
                    return
                rec = await call_with_retry(client, [{"role": "user", "content": JUDGE.format(response=text)}], 3000, check)
            guard.see(rec)
            f.write(json.dumps({"model": m, "paper_id": pid, "condition": cond, "hash": h, "input": inp,
                                "judge_model": client.c["model"], "ok": rec.get("ok"), "parsed": rec.get("parsed"),
                                "error": rec.get("error"), "usage": rec.get("usage")}, ensure_ascii=False) + "\n")
            f.flush()
            n["k"] += 1
            if n["k"] % 500 == 0 or n["k"] == len(jobs):
                print(f"[judge3] {n['k']}/{len(jobs)}", flush=True)
        await asyncio.gather(*[job(*j) for j in jobs])
    await client.http.aclose()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["neutral", "raw"]))
