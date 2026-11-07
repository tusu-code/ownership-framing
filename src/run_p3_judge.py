"""Full study: blind-judge every open answer, neutralize it, and blind-judge the neutral version.

  python src/run_p3_judge.py run [--struct]   # one pass over all current answers (skips finished work)
  python src/run_p3_judge.py report           # results/p3_report.md

Answers are the current-relay, rep-0 answers of papers in sample_full.csv (quick_check.answers).
Every result is keyed by a hash of the answer text, so a rerun on another relay or a re-asked
cell can never pick up a judgment of a different answer. Files:
  results/p3_quick_judge.jsonl     judge of the raw answer (shared with quick_check.py)
  results/p3_neutral.jsonl         neutral third-person rewrite of the answer
  results/p3_judge_neutral.jsonl   judge of the neutral rewrite
Open answers are always processed; --struct also processes struct answers (used only to check
the judge against the model's own scores).
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import JUDGE, NEUTRALIZE  # noqa: E402
from quick_check import MODELS, OUT as RAW_PATH, answers, flips, paired, struct_frame, table, text_hash  # noqa: E402
from run import Client, load_config, parse_json  # noqa: E402
from run_p2 import ASKER_CUES, FatalGuard, call_with_retry, valid_judge  # noqa: E402

OPENLIKE = ("open", "fixedopen", "ablorder", "ablrole", "ablmerits")
ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
NEU_PATH = RES / "p3_neutral.jsonl"
JN_PATH = RES / "p3_judge_neutral.jsonl"
REPORT = RES / "p3_report.md"


def read_ok(path, key="hash"):
    out = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            if r.get("ok"):
                out[r[key]] = r
    return out


def judge_check(out):
    obj = parse_json(out["text"])
    return valid_judge(obj), obj


def neutral_check(raw):
    def check(out):
        t = (out["text"] or "").strip()
        ratio = len(t) / max(1, len(raw))
        return bool(t) and 0.4 < ratio < 1.3 and len(ASKER_CUES.findall(t)) <= 1, None
    return check


async def run(cfg, include_struct):
    A = answers()
    # Fixed-instruction answers (fixed_*) are scored only by the model's own form; they are never judged.
    # 2026-11-03: free-text conditions of the revision (fixedopen_, ablorder_, ablrole_, ablmerits_) are judged like open.
    keys = [k for k in A if k[2].split("_")[0] in OPENLIKE or (include_struct and k[2].startswith("struct"))]
    raw_done, neu_done, jn_done = read_ok(RAW_PATH), read_ok(NEU_PATH), read_ok(JN_PATH)
    # Struct answers get only the raw judge: it is used to check the judge against the model's own scores.
    todo = [k for k in keys if text_hash(A[k]["raw"]) not in (raw_done if k[2].startswith("struct") else jn_done)]
    # Pair author and reviewer answers of the same paper and model, and interleave models,
    # so a partial pass already yields usable paired results for every model.
    groups = {}
    for k in todo:
        groups.setdefault((k[0], k[1]), []).append(k)
    order = list(groups)
    random.Random(20261030).shuffle(order)
    order.sort(key=lambda g: MODELS.index(g[0]))
    by_model = {m: [g for g in order if g[0] == m] for m in MODELS}
    jobs = []
    for i in range(max((len(v) for v in by_model.values()), default=0)):
        for m in MODELS:
            if i < len(by_model[m]):
                jobs += sorted(groups[by_model[m][i]], key=lambda k: k[2])
    name = cfg["judge_model"]
    client = Client(name, cfg["models"][name], cfg.get("defaults", {}))
    print(f"[p3-judge:{client.c['model']}] {len(jobs)} answers to finish "
          f"(raw judged {len(raw_done)}, neutralized {len(neu_done)}, neutral judged {len(jn_done)})", flush=True)
    guard = FatalGuard("p3-judge")
    n = {"done": 0}
    files = {p: open(p, "a") for p in (RAW_PATH, NEU_PATH, JN_PATH)}

    def write(path, rec):
        files[path].write(json.dumps(rec, ensure_ascii=False) + "\n")
        files[path].flush()

    async def job(k):
        m, pid, cond = k
        raw = A[k]["raw"]
        h = text_hash(raw)
        base = {"model": m, "paper_id": pid, "condition": cond, "hash": h, "judge_model": client.c["model"]}
        async with client.sem:
            if guard.stopped:
                return
            if h not in raw_done:
                rec = await call_with_retry(client, [{"role": "user", "content": JUDGE.format(response=raw)}], 3000, judge_check)
                guard.see(rec)
                write(RAW_PATH, {**base, "ok": rec.get("ok"), "parsed": rec.get("parsed"), "error": rec.get("error")})
                if rec.get("ok"):
                    raw_done[h] = rec
            if guard.stopped or cond.startswith("struct"):
                n["done"] += 1
                return
            neu = neu_done.get(h)
            if not neu:
                rec = await call_with_retry(client, [{"role": "user", "content": NEUTRALIZE.format(response=raw)}],
                                            8000, neutral_check(raw))
                guard.see(rec)
                neu = {**base, "ok": rec.get("ok"), "text": rec.get("raw"), "error": rec.get("error"),
                       "len_ratio": round(len(rec.get("raw") or "") / max(1, len(raw)), 3),
                       "cues_left": len(ASKER_CUES.findall(rec.get("raw") or ""))}
                write(NEU_PATH, neu)
                if not rec.get("ok"):
                    return
                neu_done[h] = neu
            if guard.stopped:
                return
            rec = await call_with_retry(client, [{"role": "user", "content": JUDGE.format(response=neu["text"])}],
                                        3000, judge_check)
            guard.see(rec)
            write(JN_PATH, {**base, "ok": rec.get("ok"), "parsed": rec.get("parsed"), "error": rec.get("error"),
                            "neutral_hash": text_hash(neu["text"])})
        n["done"] += 1
        if n["done"] % 50 == 0 or n["done"] == len(jobs):
            print(f"[p3-judge] {n['done']}/{len(jobs)}", flush=True)

    await asyncio.gather(*[job(k) for k in jobs])
    await client.http.aclose()
    for fh in files.values():
        fh.close()


def judged_frame(A, path):
    J = read_ok(path)
    rows = []
    for (m, pid, cond), r in A.items():
        j = J.get(text_hash(r.get("raw")))
        if not j:
            continue
        p = j["parsed"]
        rows.append(dict(model=m, paper_id=pid, fmt=cond.split("_")[0], who=cond.split("_")[1],
                         rating=float(p["implied_rating"]),
                         accept=float(str(p["implied_decision"]).lower() == "accept"),
                         tone=float(p["overall_tone"]), readiness=float(p["readiness"]),
                         n_major=float(p.get("n_major_concerns") or 0),
                         fixable=float(p.get("problems_framed_as") == "fixable")))
    return pd.DataFrame(rows)


def interaction(open_df, struct_df, m):
    """(author - reviewer | open, judge) - (author - reviewer | struct, self-score), per paper."""
    def per_paper(df):
        w = df[df.model == m].pivot_table(index="paper_id", columns="who", values="rating", aggfunc="first")
        if not {"author", "reviewer"} <= set(w.columns):
            return pd.Series(dtype=float)
        return (w["author"] - w["reviewer"]).dropna()
    x = (per_paper(open_df) - per_paper(struct_df)).dropna()
    if len(x) < 5:
        return None
    from quick_check import boot
    mu, lo, hi = boot(x)
    nz = x[x != 0]
    p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else float("nan")
    return f"{mu:+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.3f} (n={len(x)})"


def report():
    A = answers()
    s = struct_frame(A)
    raw = judged_frame(A, RAW_PATH)
    neu = judged_frame(A, JN_PATH)
    o_raw = raw[raw.fmt == "open"] if not raw.empty else raw
    o_neu = neu[neu.fmt == "open"] if not neu.empty else neu
    md = ["# Full study report", "",
          "Author − reviewer on the same paper; rep 0; current relay; ICLR 0–10 scale. "
          "Open asks are scored by a blind judge (deepseek-v4.1-flash) that reads only the answer.", ""]
    md += table(o_neu, "Open asks, judge read the neutralized third-person answer (main result)",
                ["rating", "tone", "fixable", "n_major"])
    md += [""]
    md += table(o_raw, "Open asks, judge read the raw answer (check)", ["rating", "tone", "fixable", "n_major"])
    md += [""]
    md += table(s, "Struct asks, model's own JSON scores", ["rating", "n_major", "soundness"])
    md += ["", "## Ask format × who asks (rating): (A−R | open, neutral judge) − (A−R | struct, self-score)", ""]
    for m in MODELS:
        md.append(f"- {m}: {interaction(o_neu, s, m) if not o_neu.empty and not s.empty else '—'}")
    md += ["", "## Coverage", ""]
    for m in MODELS:
        n_open = sum(1 for k in A if k[0] == m and k[2].startswith("open"))
        n_neu = len(o_neu[o_neu.model == m]) if not o_neu.empty else 0
        md.append(f"- {m}: open answers {n_open}, neutral-judged {n_neu}")
    text = "\n".join(md)
    REPORT.write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--struct", action="store_true")
    args = ap.parse_args()
    if args.cmd == "run":
        asyncio.run(run(load_config(), args.struct))
    report()


if __name__ == "__main__":
    main()
