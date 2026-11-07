"""Analyse the pilot: paired framing effects, decision flips against a retest noise floor, go/no-go.

  python src/analyze.py            # all review_*.jsonl except mock
  python src/analyze.py --include-mock

Writes results/analysis.json and results/analysis.md.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import parse_json, valid_review  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, DATA = ROOT / "results", ROOT / "data"
CONTRASTS = [("author", "reviewer"), ("author", "neutral"), ("neutral", "reviewer"), ("author_stakes", "author")]
B = 5000
RNG = np.random.default_rng(20261025)


def load(include_mock):
    rows = []
    for p in sorted(RES.glob("review_*.jsonl")):
        model = p.stem.split("_", 1)[1]
        if model == "mock" and not include_mock:
            continue
        for line in open(p):
            r = json.loads(line)
            if not r.get("ok") and r.get("raw"):
                # Re-parse with the current parser (it learned to repair LaTeX escapes after the run).
                obj = parse_json(r["raw"])
                if valid_review(obj):
                    r["ok"], r["parsed"] = True, obj
            o = r.get("parsed") or {}
            row = dict(model=model, paper_id=r["paper_id"], condition=r["condition"], rep=r["rep"],
                       ok=bool(r.get("ok")), error="error" in r, provider_model=r.get("provider_model"))
            if r.get("ok"):
                w = o.get("weaknesses") or []
                row.update(rating=float(o["rating"]), accept=int(str(o["recommendation"]).lower() == "accept"),
                           soundness=float(o["soundness"]), presentation=float(o["presentation"]),
                           contribution=float(o["contribution"]), confidence=float(o.get("confidence") or np.nan),
                           n_weak=len(w), n_major=sum(1 for x in w if isinstance(x, dict) and x.get("severity") == "major"),
                           n_strength=len(o.get("strengths") or []))
            rows.append(row)
    df = pd.DataFrame(rows)
    # Keep the last ok record per key (reruns append).
    ok = df[df.ok].drop_duplicates(["model", "paper_id", "condition", "rep"], keep="last")
    return df, ok


def cluster_boot_mean(diff_by_paper):
    """diff_by_paper: per-paper mean differences; bootstrap over papers."""
    x = np.asarray(diff_by_paper, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(x), (B, len(x)))
    bs = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def wide(ok, model, var):
    sub = ok[ok.model == model]
    return sub.pivot_table(index=["paper_id", "rep"], columns="condition", values=var, aggfunc="first")


def paired(ok, model, a, b, var):
    w = wide(ok, model, var)
    if a not in w or b not in w:
        return None
    w = w[[a, b]].dropna()
    per_paper = (w[a] - w[b]).groupby(level=0).mean()
    m, lo, hi = cluster_boot_mean(per_paper)
    nz = per_paper[per_paper != 0]
    p = stats.wilcoxon(nz).pvalue if len(nz) >= 5 else np.nan
    sd = ok[ok.model == model][var].std()
    return dict(var=var, a=a, b=b, n_papers=int(per_paper.size), diff=m, ci=[lo, hi], p_wilcoxon=p,
                diff_sd_units=m / sd if sd else np.nan)


def flips(ok, model, a, b):
    """Recommendation disagreement between two conditions, same paper and rep."""
    w = wide(ok, model, "accept")
    if a not in w or b not in w:
        return None
    w = w[[a, b]].dropna()
    ab = int(((w[a] == 1) & (w[b] == 0)).sum())   # accept under a, reject under b
    ba = int(((w[a] == 0) & (w[b] == 1)).sum())
    n = len(w)
    p = stats.binomtest(ab, ab + ba).pvalue if ab + ba else 1.0
    per_paper = ((w[a] != w[b]).astype(float)).groupby(level=0).mean()
    m, lo, hi = cluster_boot_mean(per_paper)
    net = ((w[a] - w[b]).groupby(level=0).mean())
    nm, nlo, nhi = cluster_boot_mean(net)
    return dict(a=a, b=b, n=n, flip_rate=m, flip_ci=[lo, hi], a_acc_b_rej=ab, a_rej_b_acc=ba,
                net_accept_shift=nm, net_ci=[nlo, nhi], p_mcnemar_exact=p,
                accept_rate_a=float(w[a].mean()), accept_rate_b=float(w[b].mean()))


def retest(ok, model):
    """Noise floor: rep 0 vs rep 1 within the same condition."""
    sub = ok[ok.model == model]
    w = sub.pivot_table(index=["paper_id", "condition"], columns="rep", values=["accept", "rating"], aggfunc="first")
    if ("accept", 1) not in w:
        return None
    w = w.dropna()
    return dict(n=len(w), flip_rate=float((w[("accept", 0)] != w[("accept", 1)]).mean()),
                rating_mad=float((w[("rating", 0)] - w[("rating", 1)]).abs().mean()),
                rating_icc_like=float(np.corrcoef(w[("rating", 0)], w[("rating", 1)])[0, 1]) if len(w) > 2 else np.nan)


def human_validity(ok, model, sample):
    sub = ok[(ok.model == model) & (ok.condition == "neutral")].groupby("paper_id")[["rating", "accept"]].mean()
    j = sub.join(sample[["human_mean", "accept"]].rename(columns={"accept": "human_accept"}), how="inner")
    if len(j) < 5:
        return None
    rho = stats.spearmanr(j.rating, j.human_mean)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(j.human_accept.astype(int), j.rating) if j.human_accept.nunique() == 2 else np.nan
    return dict(n=len(j), spearman_vs_human_mean=float(rho.statistic), p=float(rho.pvalue),
                auc_rating_vs_decision=float(auc), llm_accept_rate=float(j.accept.mean()))


def by_stratum(ok, model, sample, a="author", b="reviewer"):
    w = wide(ok, model, "rating")
    if a not in w or b not in w:
        return None
    d = (w[a] - w[b]).groupby(level=0).mean().rename("diff").to_frame().join(sample["stratum"])
    acc = wide(ok, model, "accept")
    d["flip"] = (acc[a] != acc[b]).astype(float).groupby(level=0).mean()
    out = d.groupby("stratum").agg(n=("diff", "size"), rating_diff=("diff", "mean"), flip_rate=("flip", "mean"))
    border = d.stratum.str.startswith("border")
    if border.sum() >= 3 and (~border).sum() >= 3:
        out.attrs["p_border_vs_rest"] = float(stats.mannwhitneyu(d["diff"][border], d["diff"][~border]).pvalue)
    return out


def probe_summary():
    out = {}
    for p in sorted(RES.glob("probe_*.jsonl")):
        model = p.stem.split("_", 1)[1]
        recs = [json.loads(l) for l in open(p)]
        recs = [r for r in recs if r.get("ok")]
        if recs:
            out[model] = dict(n=len(recs), recognized=float(np.mean([bool((r["parsed"] or {}).get("recognized")) for r in recs])),
                              recognized_ids=[r["paper_id"] for r in recs if (r["parsed"] or {}).get("recognized")])
    return out


def go_no_go(res):
    """Pre-stated rule: go if, in >= 2 models, the author-vs-reviewer flip rate exceeds the retest
    floor by >= 10 points with a net shift toward accept under authorship (McNemar p < .05)."""
    hits = []
    for m, r in res.items():
        f, rt = r["flips"].get("author_vs_reviewer"), r.get("retest")
        if not f:
            continue
        floor = rt["flip_rate"] if rt else 0.0
        if (f["flip_rate"] - floor >= 0.10 and f["net_accept_shift"] > 0 and f["p_mcnemar_exact"] < 0.05):
            hits.append(m)
    return dict(models_meeting_rule=hits, decision="GO" if len(hits) >= 2 else "NO-GO / reassess")


def fmt(x, d=2):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-mock", action="store_true")
    args = ap.parse_args()
    df, ok = load(args.include_mock)
    if ok.empty:
        print("no parsed reviews yet")
        return
    sample = pd.read_csv(DATA / "sample_pilot.csv", index_col="paper_id")
    res, md = {}, ["# Ownership-framing pilot: analysis", ""]
    for model in sorted(ok.model.unique()):
        allr = df[df.model == model]
        r = dict(calls=int(len(allr)), parsed=int(allr.ok.sum()), errors=int(allr.error.sum()),
                 provider_models=sorted({str(x) for x in ok[ok.model == model].provider_model.unique()}),
                 means=ok[ok.model == model].groupby("condition")[["rating", "accept", "n_weak", "n_major", "soundness", "contribution"]].mean().round(3).to_dict("index"),
                 effects={}, flips={}, retest=retest(ok, model), human=human_validity(ok, model, sample))
        for a, b in CONTRASTS:
            for var in ["rating", "n_major", "n_weak", "soundness", "contribution"]:
                e = paired(ok, model, a, b, var)
                if e:
                    r["effects"][f"{var}:{a}_vs_{b}"] = e
            f = flips(ok, model, a, b)
            if f:
                r["flips"][f"{a}_vs_{b}"] = f
        strat = by_stratum(ok, model, sample)
        r["by_stratum"] = None if strat is None else {**strat.round(3).to_dict("index"), **strat.attrs}
        res[model] = r

        md += [f"## {model}", f"calls {r['calls']}, parsed {r['parsed']}, errors {r['errors']}; served as {', '.join(r['provider_models'])}", ""]
        md += ["| condition | rating | accept rate | weaknesses | major |", "|---|---:|---:|---:|---:|"]
        for c, v in r["means"].items():
            md.append(f"| {c} | {fmt(v['rating'])} | {fmt(v['accept'])} | {fmt(v['n_weak'], 1)} | {fmt(v['n_major'], 1)} |")
        md += ["", "| contrast | rating diff [95% CI] | p | flip rate [95% CI] | net accept shift | McNemar p |", "|---|---|---:|---|---:|---:|"]
        for a, b in CONTRASTS:
            e, f = r["effects"].get(f"rating:{a}_vs_{b}"), r["flips"].get(f"{a}_vs_{b}")
            if e and f:
                md.append(f"| {a} − {b} | {fmt(e['diff'])} [{fmt(e['ci'][0])}, {fmt(e['ci'][1])}] | {fmt(e['p_wilcoxon'], 3)} | "
                          f"{fmt(f['flip_rate'])} [{fmt(f['flip_ci'][0])}, {fmt(f['flip_ci'][1])}] | {fmt(f['net_accept_shift'])} | {fmt(f['p_mcnemar_exact'], 3)} |")
        if r["retest"]:
            md.append(f"\nRetest floor (same condition, rep 0 vs 1): flip {fmt(r['retest']['flip_rate'])}, "
                      f"mean |Δrating| {fmt(r['retest']['rating_mad'])}, r = {fmt(r['retest']['rating_icc_like'])}")
        if r["human"]:
            h = r["human"]
            md.append(f"Validity (neutral vs human): Spearman {fmt(h['spearman_vs_human_mean'])} (p {fmt(h['p'], 3)}), "
                      f"AUC vs decision {fmt(h['auc_rating_vs_decision'])}, LLM accept rate {fmt(h['llm_accept_rate'])}")
        if strat is not None:
            md += ["", "author − reviewer by stratum:", "", strat.round(2).to_markdown()]
        md.append("")

    # Cross-model agreement on which papers move.
    models = sorted(res)
    if len(models) >= 2:
        per = {m: (wide(ok, m, "rating").get("author") - wide(ok, m, "rating").get("reviewer")).groupby(level=0).mean()
               for m in models if {"author", "reviewer"} <= set(wide(ok, m, "rating").columns)}
        pairs = {f"{x}~{y}": float(pd.concat([per[x], per[y]], axis=1).dropna().corr(method="spearman").iloc[0, 1])
                 for x, y in combinations(per, 2)}
        res["_cross_model_effect_corr"] = pairs
        md += ["## Cross-model agreement of per-paper author − reviewer effect (Spearman)", "",
               *[f"- {k}: {fmt(v)}" for k, v in pairs.items()], ""]

    res["_probe"] = probe_summary()
    if res["_probe"]:
        md += ["## Memorization probe", "", *[f"- {m}: recognized {fmt(v['recognized'])} of {v['n']}" for m, v in res["_probe"].items()], ""]
    res["_go_no_go"] = go_no_go({m: v for m, v in res.items() if not m.startswith("_")})
    md += ["## Go / no-go", "", f"Rule: in ≥2 models, author-vs-reviewer flip rate ≥ retest floor + 10 points, net shift toward accept, McNemar p < .05.",
           f"Models meeting it: {res['_go_no_go']['models_meeting_rule'] or 'none'} → **{res['_go_no_go']['decision']}**", ""]

    (RES / "analysis.json").write_text(json.dumps(res, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o)))
    (RES / "analysis.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
