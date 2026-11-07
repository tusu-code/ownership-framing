"""Reviewer M1(c): does the judge read a warmer register as a higher rating when the content is fixed?

  python src/tone_swap.py rewrite   -> results/p4_tone_rewrite.jsonl (warm and cold version of each answer)
  python src/tone_swap.py judge     -> results/p4_tone_judge.jsonl  (main judge and MiniMax on both versions)
Answers: 100 neutralized open answers per model (Claude, Gemini, GLM, DeepSeek), half from the author ask and
half from the reviewer ask, drawn at random from the main sample. The rewriter is the main rewriting model.
"""
import asyncio, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import JUDGE, TONE_INSTR, TONE_SWAP  # noqa: E402
from quick_check import answers, text_hash  # noqa: E402
from run import Client, load_config, parse_json  # noqa: E402
from run_p2 import FatalGuard, call_with_retry, read, valid_judge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RW, JG = RES / "p4_tone_rewrite.jsonl", RES / "p4_tone_judge.jsonl"
MODELS, PER_ASK = ["claude", "gemini", "glm", "deepseek"], 50


def pick():
    neu = {r["hash"]: r["text"] for r in read(RES / "p3_neutral.jsonl") if r.get("ok")}
    A = answers()
    out = []
    for m in MODELS:
        for cond in ("open_author", "open_reviewer"):
            pool = sorted((pid, text_hash(r["raw"])) for (mm, pid, c), r in A.items()
                          if mm == m and c == cond and text_hash(r["raw"]) in neu)
            for pid, h in random.Random(f"tone-{m}-{cond}").sample(pool, PER_ASK):
                out.append(dict(model=m, paper_id=pid, condition=cond, hash=h, text=neu[h]))
    return out


def judge_check(out):
    o = parse_json(out["text"])
    return valid_judge(o), o


async def rewrite(cfg):
    items = pick()
    done = {(r["hash"], r["register"]) for r in read(RW) if r.get("ok")}
    jobs = [(it, reg) for it in items for reg in ("warm", "cold") if (it["hash"], reg) not in done]
    client = Client("dsflash", cfg["models"]["dsflash"], cfg.get("defaults", {}))
    print(f"[tone-rewrite] {len(jobs)} calls ({len(done)} done)", flush=True)
    guard = FatalGuard("tone-rewrite")
    with open(RW, "a") as f:
        async def job(it, reg):
            def check(out):
                t = (out["text"] or "").strip()
                return bool(t) and 0.7 < len(t) / max(1, len(it["text"])) < 1.4, None
            msg = TONE_SWAP.format(register=reg, instruction=TONE_INSTR[reg], text=it["text"])
            async with client.sem:
                if guard.stopped:
                    return
                rec = await call_with_retry(client, [{"role": "user", "content": msg}], 8000, check)
            guard.see(rec)
            f.write(json.dumps({**{k: it[k] for k in ("model", "paper_id", "condition", "hash")}, "register": reg,
                                "ok": rec.get("ok"), "text": rec.get("raw"), "error": rec.get("error")}, ensure_ascii=False) + "\n")
            f.flush()
        await asyncio.gather(*[job(*j) for j in jobs])
    await client.http.aclose()


async def judge(cfg):
    rws = [r for r in read(RW) if r.get("ok")]
    done = {(r["hash"], r["register"], r["judge"]) for r in read(JG) if r.get("ok")}
    with open(JG, "a") as f:
        for jname in ("dsflash", "minimax"):
            client = Client(jname, cfg["models"][jname], cfg.get("defaults", {}))
            jobs = [r for r in rws if (r["hash"], r["register"], jname) not in done]
            print(f"[tone-judge:{jname}] {len(jobs)} calls", flush=True)
            guard = FatalGuard(f"tone-judge-{jname}")

            async def job(r):
                async with client.sem:
                    if guard.stopped:
                        return
                    rec = await call_with_retry(client, [{"role": "user", "content": JUDGE.format(response=r["text"])}], 3000, judge_check)
                guard.see(rec)
                f.write(json.dumps({**{k: r[k] for k in ("model", "paper_id", "condition", "hash", "register")}, "judge": jname,
                                    "ok": rec.get("ok"), "parsed": rec.get("parsed"), "error": rec.get("error")}, ensure_ascii=False) + "\n")
                f.flush()
            await asyncio.gather(*[job(r) for r in jobs])
            await client.http.aclose()


if __name__ == "__main__":
    cfg = load_config()
    asyncio.run({"rewrite": rewrite, "judge": judge}[sys.argv[1]](cfg))
