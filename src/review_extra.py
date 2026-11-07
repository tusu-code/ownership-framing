"""Analyses added in response to the mock reviews of 6 November 2026 (exploratory; no new model calls).

  python src/review_extra.py   -> extend/out/review_extra.json, extend/out/clmm_input.csv (CLMM run by clmm.R)

practical   Claude, review form: share of papers whose rating moved up or down between the author and the reviewer
            ask (main study and confirmatory study), against the same counts for one ask re-run in a later batch.
cues        Blind-reader contrasts without the neutralized rewrites that kept an asker-directed cue word.
gemini_id   Mitigation study, Gemini: the author - reviewer gap computed separately on cells whose first turn came
            from gemini-4-argon and from gemini-4-argon-high.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "full_study"
OUT = ROOT / "extend" / "out"
RNG = np.random.default_rng(20261106)


def read(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def est(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    b = x[RNG.integers(0, len(x), (5000, len(x)))].mean(1)
    nz = x[x != 0]
    return dict(diff=float(x.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), n=int(len(x)),
                p=float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else 1.0)


def moves(a, b):
    """Shares of papers rated higher, equal, lower under a than under b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return dict(n=int(len(d)), up=float((d > 0).mean()), same=float((d == 0).mean()), down=float((d < 0).mean()),
                up_n=int((d > 0).sum()), down_n=int((d < 0).sum()), sign_p=float(stats.binomtest(int((d > 0).sum()), int((d != 0).sum())).pvalue))


def practical():
    out = {}
    c = pd.read_csv(FS / "out" / "cells.csv")
    w = c[(c.model == "claude") & (c.fmt == "struct")].pivot_table(index="paper_id", columns="who", values="rating").dropna(subset=["author", "reviewer"])
    out["main_author_vs_reviewer"] = moves(w.author, w.reviewer)
    # Re-run of the same ask in a later batch (100 retest papers, review form, rep 0 vs rep 1).
    ids = set(pd.read_csv(FS / "data" / "retest100.csv").paper_id)
    rows = []
    for r in read(FS / "results" / "p2_review_claude.jsonl"):
        if r.get("ok") and r["paper_id"] in ids and str(r.get("relay", "")).startswith("https://api.sailapi"):
            fmt, who = r["condition"].split("_", 1)
            if fmt == "struct" and who in ("author", "reviewer") and r["rep"] in (0, 1):
                rows.append(dict(pid=r["paper_id"], who=who, rep=r["rep"], y=float(r["parsed"]["rating"])))
    d = pd.DataFrame(rows).drop_duplicates(["pid", "who", "rep"], keep="last")
    rr = d.pivot_table(index=["pid", "who"], columns="rep", values="y").dropna()
    out["rerun_same_ask"] = moves(rr[1], rr[0])
    # Confirmatory study: per paper, mean over the three wordings.
    cf = [r for r in read(ROOT / "confirm" / "results" / "confirm_claude.jsonl")
          if r.get("ok") and r["role"] == "sample" and str(r.get("provider_model", "")).startswith("claude-sonnet-4-6")]
    cd = pd.DataFrame([dict(pid=r["paper_id"], who=r["who"], fam=r["family"], y=float(r["rating"])) for r in cf]).drop_duplicates(["pid", "who", "fam"])
    single = cd[cd.fam == "W1"].pivot_table(index="pid", columns="who", values="y").dropna(subset=["author", "reviewer"])
    out["confirm_W1_author_vs_reviewer"] = moves(single.author, single.reviewer)
    # Same-wording cells, all three wordings pooled: one paper x wording is one comparison.
    cw = cd.pivot_table(index=["pid", "fam"], columns="who", values="y").dropna(subset=["author", "reviewer"])
    out["confirm_cells_author_vs_reviewer"] = moves(cw.author, cw.reviewer)
    return out, c


def cues():
    """Blind-reader contrasts (selfscored answers) with and without rewrites that kept an asker-directed cue."""
    neu = {r["hash"]: r for r in read(FS / "results" / "p3_neutral.jsonl") if r.get("ok")}
    J = {"main": {r["hash"]: r["parsed"] for r in read(FS / "results" / "p4_judge_minimax.jsonl") if r.get("ok") and r.get("input") == "neutral"},
         "flash": {r["hash"]: r["parsed"] for r in read(FS / "results" / "p3_judge_neutral.jsonl") if r.get("ok")}}
    rows = []
    for r in read(FS / "results" / "p4_selfscore.jsonl"):
        if not r.get("ok"):
            continue
        fmt, who = r["condition"].split("_", 1)
        if fmt != "open" or who not in ("author", "reviewer"):
            continue
        n = neu.get(r["hash"])
        row = dict(model=r["model"], rep=r["rep"], pid=r["paper_id"], who=who, self=float(r["parsed"]["rating"]),
                   cue=bool(n and n.get("cues_left")))
        for k, D in J.items():
            j = D.get(r["hash"])
            row[k] = float(j["implied_rating"]) if j else np.nan
        rows.append(row)
    S = pd.DataFrame(rows).drop_duplicates(["model", "rep", "pid", "who"], keep="last")
    S = S[((S.model == "deepseek") & (S.rep == 3)) | ((S.model != "deepseek") & (S.rep == 0))]
    out = {"n_answers": int(len(S)), "n_cue": int(S.cue.sum())}
    for m, g in S.groupby("model"):
        bad = set(g[g.cue].pid)
        r_ = {"papers_with_cue": len(bad)}
        for col in ("main", "flash"):
            for lab, x in (("all", g), ("no_cue", g[~g.pid.isin(bad)])):
                w = x.pivot_table(index="pid", columns="who", values=col).dropna()
                x2 = x.dropna(subset=[col]).assign(gap=lambda t: t[col] - t.self)
                wg = x2.pivot_table(index="pid", columns="who", values="gap").dropna()
                r_[f"{col}/{lab}/author-reviewer"] = est((w.author - w.reviewer).values)
                r_[f"{col}/{lab}/reader_gap"] = est((wg.author - wg.reviewer).values)
        out[m] = r_
    return out


def gemini_id():
    """Mitigation study, Gemini: gap by the identifier that wrote the first turn of each cell."""
    op = read(ROOT / "mitig" / "results_156" / "open_gemini.jsonl")
    sc = read(ROOT / "mitig" / "results_156" / "score_gemini.jsonl")
    first, score = {}, {}
    for r in op:
        if r["role"] == "sample" and r.get("ok") and str(r.get("provider_model", "")).startswith("gemini-4-argon"):
            first.setdefault((r["paper_id"], r["condition"]), r)
    for r in sc:
        if r["role"] == "sample" and r.get("ok") and str(r.get("provider_model", "")).startswith("gemini-4-argon"):
            score.setdefault((r["paper_id"], r["condition"]), r)
    d = pd.DataFrame([dict(pid=k[0], who=v["who"], ask=v["ask"], fam=v["family"], y=float(v["rating"]),
                           ident=first[k]["provider_model"]) for k, v in score.items() if k in first])
    out = {"cells": d.groupby("ident").size().to_dict()}
    # Pairs (paper, request, wording) whose author and reviewer cells both came from one identifier.
    w = d.pivot_table(index=["pid", "ask", "fam"], columns="who", values="y")
    idn = d.pivot_table(index=["pid", "ask", "fam"], columns="who", values="ident", aggfunc="first")
    pair = w.join(idn, rsuffix="_id").dropna()
    for ask in ("natural", "review"):
        for ident in ("gemini-4-argon", "gemini-4-argon-high"):
            x = pair.xs(ask, level="ask")
            x = x[(x.author_id == ident) & (x.reviewer_id == ident)]
            out[f"{ask}/{ident}"] = est((x.author - x.reviewer).values)
        x = pair.xs(ask, level="ask")
        out[f"{ask}/all_pairs"] = est((x.author - x.reviewer).values)
    return out


def marginal_or(d):
    """Population-averaged (marginal) proportional-odds ratio, author vs reviewer, from an ordinal GEE with
    independence working correlation clustered by paper; wording as a fixed effect where there are several.
    The CLMM odds ratio (clmm.R) is conditional on the paper's random intercept and grows with the
    between-paper variance; the marginal one compares the two asks averaged over papers."""
    import statsmodels.api as sm
    out = {}
    for st, g in d.groupby("study"):
        g = g.sort_values("pid")
        X = (g.who == "author").astype(float).values[:, None]
        if g.family.nunique() > 1:
            X = np.column_stack([X, pd.get_dummies(g.family, drop_first=True).astype(float).values])
        f = sm.OrdinalGEE(g.rating.astype(int).values, X, g.pid.values, cov_struct=sm.cov_struct.Independence()).fit()
        k = len(np.unique(g.rating)) - 1                   # thresholds come first, then the author coefficient
        b, se = float(f.params[k]), float(f.bse[k])
        out[st] = dict(or_=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)), hi=float(np.exp(b + 1.96 * se)),
                       p=float(2 * stats.norm.sf(abs(b / se))))
    return out


def clmm_input(c):
    """Rows for the ordinal mixed models (clmm.R): Claude review form, author and reviewer asks."""
    rows = [dict(study="main", pid=p, family="W1", who=w, rating=int(y))
            for p, w, y in c[(c.model == "claude") & (c.fmt == "struct") & c.who.isin(["author", "reviewer"])][["paper_id", "who", "rating"]].dropna().values]
    for r in read(ROOT / "confirm" / "results" / "confirm_claude.jsonl"):
        if r.get("ok") and r["role"] == "sample" and r["who"] in ("author", "reviewer") and str(r.get("provider_model", "")).startswith("claude-sonnet-4-6"):
            rows.append(dict(study="confirm", pid=r["paper_id"], family=r["family"], who=r["who"], rating=int(r["rating"])))
    for r in read(ROOT / "mitig" / "results_156" / "form_claude.jsonl"):
        if r.get("ok") and r["role"] == "sample" and str(r.get("provider_model", "")).startswith("claude-sonnet-4-6"):
            rows.append(dict(study=f"mitig_{r['ask']}", pid=r["paper_id"], family=r["family"], who=r["who"], rating=int(r["rating"])))
    d = pd.DataFrame(rows).drop_duplicates(["study", "pid", "family", "who"])
    d.to_csv(OUT / "clmm_input.csv", index=False)
    return d.groupby("study").size().to_dict(), marginal_or(d)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pr, c = practical()
    rows, marg = clmm_input(c)
    res = {"practical": pr, "cues": cues(), "gemini_id": gemini_id(), "clmm_rows": rows, "marginal_or": marg}
    (OUT / "review_extra.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
