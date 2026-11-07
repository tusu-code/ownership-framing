"""Interim look at the full study; does not touch the main pipeline's files.

  python src/quick_check.py struct              # self-reported scores of struct answers, no API calls
  python src/quick_check.py judge --pairs 100   # blind-judge a sample of open answer pairs (judge_model)
  python src/quick_check.py report              # struct + whatever open pairs are judged so far

Uses current-relay, rep-0 answers of papers in sample_full.csv. Judge results go to
results/p3_quick_judge.jsonl, keyed by a hash of the answer text, so they never mix with
Pilot 2 files and stay valid only for the exact answer they judged.
"""
import argparse
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import JUDGE  # noqa: E402
from run import Client, load_config, parse_json, valid_review  # noqa: E402
from run_p2 import call_with_retry, valid_judge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
MODELS = ["claude", "gpt", "glm", "deepseek", "gemini", "grok"]   # 2026-11-02: gemini, grok added
OUT = RES / "p3_quick_judge.jsonl"
REPORT = RES / "p3_quick_check.md"
RNG = np.random.default_rng(20261028)


def answers():
    """Latest valid rep-0 answer per (model, paper, condition) on the current relay."""
    full = set(pd.read_csv(DATA / "sample_full.csv", index_col="paper_id").index)
    out = {}
    for m in MODELS:
        for line in open(RES / f"p2_review_{m}.jsonl"):
            r = json.loads(line)
            if (r.get("ok") and r["paper_id"] in full
                    and str(r.get("relay", "")).startswith("https://api.sailapi")):
                # Retest answers (rep >= 1) get their own key so they are judged too but never replace rep 0.
                cond = r["condition"] if r["rep"] == 0 else f"{r['condition']}@r{r['rep']}"
                out[(m, r["paper_id"], cond)] = r
    return out


def text_hash(text):
    return hashlib.sha1((text or "").encode()).hexdigest()[:16]


def boot(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    bs = x[RNG.integers(0, len(x), (5000, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired(df, var):
    """Author minus reviewer on the same paper."""
    w = df.pivot_table(index="paper_id", columns="who", values=var, aggfunc="first")
    if not {"author", "reviewer"} <= set(w.columns):
        return None
    w = w[["author", "reviewer"]].dropna()
    d = w["author"] - w["reviewer"]
    m, lo, hi = boot(d)
    nz = d[d != 0]
    p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else np.nan
    return dict(n=len(w), author=float(w["author"].mean()), reviewer=float(w["reviewer"].mean()),
                diff=m, lo=lo, hi=hi, p=p)


def flips(df):
    w = df.pivot_table(index="paper_id", columns="who", values="accept", aggfunc="first")
    if not {"author", "reviewer"} <= set(w.columns):
        return None
    w = w[["author", "reviewer"]].dropna()
    a_only = int(((w.author == 1) & (w.reviewer == 0)).sum())
    r_only = int(((w.author == 0) & (w.reviewer == 1)).sum())
    p = float(stats.binomtest(a_only, a_only + r_only).pvalue) if a_only + r_only else 1.0
    return dict(n=len(w), flip=float((w.author != w.reviewer).mean()), a_only=a_only, r_only=r_only, p=p,
                acc_a=float(w.author.mean()), acc_r=float(w.reviewer.mean()))


def struct_frame(A):
    rows = []
    for (m, pid, cond), r in A.items():
        if not cond.startswith("struct"):
            continue
        o = r.get("parsed") or parse_json(r.get("raw"))
        if not valid_review(o):
            continue
        w = o.get("weaknesses") or []
        rows.append(dict(model=m, paper_id=pid, who=cond.split("_")[1], rating=float(o["rating"]),
                         accept=float(str(o["recommendation"]).lower() == "accept"),
                         soundness=float(o["soundness"]), contribution=float(o["contribution"]),
                         n_weak=float(len(w)),
                         n_major=float(sum(1 for x in w if isinstance(x, dict) and x.get("severity") == "major"))))
    return pd.DataFrame(rows)


def open_frame(A):
    if not OUT.exists():
        return pd.DataFrame()
    judged = {}
    for line in open(OUT):
        r = json.loads(line)
        if r.get("ok"):
            judged[(r["model"], r["paper_id"], r["condition"], r["hash"])] = r["parsed"]
    rows = []
    for (m, pid, cond), r in A.items():
        if not cond.startswith("open"):
            continue
        j = judged.get((m, pid, cond, text_hash(r.get("raw"))))
        if not j:
            continue
        rows.append(dict(model=m, paper_id=pid, who=cond.split("_")[1], rating=float(j["implied_rating"]),
                         accept=float(str(j["implied_decision"]).lower() == "accept"),
                         tone=float(j["overall_tone"]), readiness=float(j["readiness"]),
                         n_major=float(j.get("n_major_concerns") or 0),
                         fixable=float(j.get("problems_framed_as") == "fixable")))
    return pd.DataFrame(rows)


def fmt(e, k=2):
    if not e:
        return "—"
    p = "nan" if np.isnan(e["p"]) else f"{e['p']:.3f}"
    return f"{e['diff']:+.{k}f} [{e['lo']:+.{k}f}, {e['hi']:+.{k}f}] p={p} (n={e['n']})"


def table(df, title, variables):
    lines = [f"## {title}", "", "| model | " + " | ".join(variables) + " | accept flips (A-only : R-only, p) |",
             "|---|" + "---|" * len(variables) + "---|"]
    for m in MODELS:
        d = df[df.model == m] if not df.empty else df
        if d.empty:
            lines.append(f"| {m} | " + " | ".join("—" for _ in variables) + " | — |")
            continue
        cells = [fmt(paired(d, v)) for v in variables]
        f = flips(d)
        fl = (f"{f['flip']:.2f} ({f['a_only']}:{f['r_only']}, p={f['p']:.3f}); accept rate A {f['acc_a']:.2f} R {f['acc_r']:.2f}"
              if f else "—")
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {fl} |")
    return lines


async def judge(cfg, pairs):
    A = answers()
    done = set()
    if OUT.exists():
        done = {r["hash"] for r in map(json.loads, open(OUT)) if r.get("ok")}
    per = {}
    for m in MODELS:
        pa = {k[1] for k in A if k[0] == m and k[2] == "open_author"}
        pr = {k[1] for k in A if k[0] == m and k[2] == "open_reviewer"}
        pids = sorted(pa & pr)
        random.Random(f"quick-{m}").shuffle(pids)
        per[m] = pids[:pairs]
    # Interleave models pair by pair, so a partial run already gives usable pairs for every model.
    jobs = []
    for i in range(max(len(v) for v in per.values())):
        for m in MODELS:
            if i < len(per[m]):
                for cond in ("open_author", "open_reviewer"):
                    r = A[(m, per[m][i], cond)]
                    if text_hash(r["raw"]) not in done:
                        jobs.append((m, per[m][i], cond, r["raw"]))
    name = cfg["judge_model"]
    client = Client(name, cfg["models"][name], cfg.get("defaults", {}))
    print(f"[quick-judge:{client.c['model']}] pairs per model {{{', '.join(f'{m}: {len(v)}' for m, v in per.items())}}}; "
          f"{len(jobs)} answers to judge", flush=True)

    def check(out):
        obj = parse_json(out["text"])
        return valid_judge(obj), obj

    stats_ = {"n": 0}
    with open(OUT, "a") as f:
        async def job(m, pid, cond, raw):
            async with client.sem:
                rec = await call_with_retry(client, [{"role": "user", "content": JUDGE.format(response=raw)}], 3000, check)
            f.write(json.dumps({"model": m, "paper_id": pid, "condition": cond, "hash": text_hash(raw),
                                "judge_model": client.c["model"], "ok": rec.get("ok"), "parsed": rec.get("parsed"),
                                "error": rec.get("error")}, ensure_ascii=False) + "\n")
            f.flush()
            stats_["n"] += 1
            if stats_["n"] % 50 == 0 or stats_["n"] == len(jobs):
                print(f"[quick-judge] {stats_['n']}/{len(jobs)}", flush=True)
        await asyncio.gather(*[job(*j) for j in jobs])
    await client.http.aclose()


def report():
    A = answers()
    s, o = struct_frame(A), open_frame(A)
    md = ["# Full study: interim check", "",
          "Author − reviewer on the same paper, rep 0, current relay. Ratings use the ICLR 0–10 scale.", ""]
    md += table(s, "Struct asks (model's own JSON scores)", ["rating", "n_major", "soundness"])
    md += ["", "Validity (struct, mean of both asks vs human mean rating, Spearman):"]
    sample = pd.read_csv(DATA / "sample_full.csv", index_col="paper_id")
    for m in MODELS:
        d = s[s.model == m].groupby("paper_id")["rating"].mean() if not s.empty else pd.Series(dtype=float)
        if len(d) > 5:
            rho = stats.spearmanr(d, sample.human_mean.reindex(d.index)).statistic
            md.append(f"- {m}: {rho:.2f} (n={len(d)})")
    md += [""]
    md += table(o, "Open asks (blind judge reads only the answer; sample of pairs)",
                ["rating", "tone", "fixable", "n_major"])
    md += [""]
    text = "\n".join(md)
    REPORT.write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["struct", "judge", "report"])
    ap.add_argument("--pairs", type=int, default=100)
    args = ap.parse_args()
    if args.cmd == "judge":
        asyncio.run(judge(load_config(), args.pairs))
    report()


if __name__ == "__main__":
    main()
