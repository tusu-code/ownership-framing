"""Revision experiments of 3 November 2026 (reviewer M1a, M1c, M2, M4, M6), quick summary.

  python src/revision_check.py
Reads full_study/results/p4_*.jsonl (copied from the run host) and the main judge files.
  selfscore  the evaluated model scores its own open answer in a second turn (M1a)
  tone       the same neutral assessment rewritten warm or cold, scored by both judges (M1c)
  minimax    an independent-family judge on the neutralized and the raw answers (M2)
  recognize  can the model name the original paper from a 2,500-token opening (M6)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES, DATA = ROOT / "full_study" / "results", ROOT / "full_study" / "data"
RNG = np.random.default_rng(20261103)


def read(name):
    p = RES / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def est(z):
    z = np.asarray(z, float)
    z = z[~np.isnan(z)]
    if len(z) < 5:
        return f"n={len(z)}"
    bs = z[RNG.integers(0, len(z), (4000, len(z)))].mean(1)
    nz = z[z != 0]
    p = stats.wilcoxon(nz).pvalue if len(nz) >= 5 else 1.0
    return f"{z.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}] p={p:.2g} n={len(z)}"


def flash_neutral():
    return {r["hash"]: float(r["parsed"]["implied_rating"]) for r in read("p3_judge_neutral.jsonl") if r.get("ok")}


def selfscore():
    print("=== M1a: model's own score in a second turn vs the main judge on the same answer")
    S = pd.DataFrame([r for r in read("p4_selfscore.jsonl") if r.get("ok")])
    if S.empty:
        return
    S["self"] = S.parsed.apply(lambda o: float(o["rating"]))
    S["judge"] = S.hash.map(flash_neutral())
    S["who"] = S.condition.str.split("_", n=1).str[1]
    for (m, rep), g in S.groupby(["model", "rep"]):
        print(f"{m} rep {rep}: answers {len(g)}, self vs judge Spearman {stats.spearmanr(g.self, g.judge, nan_policy='omit').statistic:.2f}")
        for col in ("self", "judge"):
            w = g.pivot_table(index="paper_id", columns="who", values=col)
            pairs = [(a, b) for a, b in (("author", "reviewer"), ("author", "colleague"), ("colleague", "reviewer"),
                                         ("author", "venue"), ("venue", "reviewer")) if {a, b} <= set(w.columns)]
            print(f"   {col:5s} means " + ", ".join(f"{k} {v:.2f}" for k, v in w.mean().items()))
            for a, b in pairs:
                print(f"         {a} - {b}: {est(w[a] - w[b])}")


def tone():
    print("=== M1c: warm minus cold rewrite of the same assessment, rating by each judge")
    T = pd.DataFrame([r for r in read("p4_tone_judge.jsonl") if r.get("ok")])
    if T.empty:
        return
    T["rating"] = T.parsed.apply(lambda o: float(o["implied_rating"]))
    T["tone"] = T.parsed.apply(lambda o: float(o["overall_tone"]))
    T["fixable"] = T.parsed.apply(lambda o: float(o.get("problems_framed_as") == "fixable"))
    for j, g in T.groupby("judge"):
        w = g.pivot_table(index="hash", columns="register", values=["rating", "tone", "fixable"])
        if not {("rating", "warm"), ("rating", "cold")} <= set(w.columns):
            continue
        print(f"{j}: pairs {len(w.dropna())}")
        for v in ("tone", "fixable", "rating"):
            print(f"   {v:8s} warm - cold {est(w[(v, 'warm')] - w[(v, 'cold')])}")


def minimax():
    print("=== M2: author - reviewer and author - colleague, main judge (Flash) vs MiniMax")
    M = pd.DataFrame([r for r in read("p4_judge_minimax.jsonl") if r.get("ok")])
    if M.empty:
        return
    M["mm"] = M.parsed.apply(lambda o: float(o["implied_rating"]))
    M["flash"] = M.hash.map(flash_neutral())
    M = M[~M.condition.str.contains("@")]
    M["who"] = M.condition.str.split("_", n=1).str[1]
    M["fmt"] = M.condition.str.split("_").str[0]
    M = M[M.fmt == "open"]
    for (m, inp), g in M.groupby(["model", "input"]):
        rho = stats.spearmanr(g.mm, g.flash, nan_policy="omit").statistic
        line = f"{m:8s} {inp:7s} n={len(g)} rho={rho:.2f}"
        for col in ("flash", "mm"):
            w = g.pivot_table(index="paper_id", columns="who", values=col)
            if {"author", "reviewer"} <= set(w.columns):
                line += f" | {col} A-R {est(w.author - w.reviewer)}"
        print(line)


def words(t):
    return set(re.findall(r"[a-z0-9]+", (t or "").lower())) - {"a", "an", "the", "of", "for", "and", "in", "on", "with", "to", "via", "by"}


def recognize():
    print("=== M6: recognition of the original paper from a 2,500-token opening")
    R = pd.DataFrame([r for r in read("p4_recognize.jsonl") if r.get("ok")])
    if R.empty:
        return
    orig = {json.loads(l)["paper_id"]: json.loads(l)["title"] for l in open(ROOT / "data" / "manuscripts.jsonl")}
    R["claimed"] = R.parsed.apply(lambda o: bool(o.get("recognized")))
    R["title_j"] = [len(words(o.get("original_title")) & words(orig.get(p))) / max(1, len(words(o.get("original_title")) | words(orig.get(p))))
                    for o, p in zip(R.parsed, R.paper_id)]
    R["title_hit"] = R.title_j >= 0.5
    for (m, v), g in R.groupby(["model", "version"]):
        print(f"{m:8s} {v:9s} n={len(g)} claims recognition {g.claimed.mean():.1%} | names the true title (word overlap >= 0.5) {g.title_hit.mean():.1%}")


if __name__ == "__main__":
    selfscore()
    tone()
    minimax()
    recognize()
