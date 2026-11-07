"""Numbers for the analyses and studies added after the mock reviews of 6 November 2026 (all exploratory).

  python src/extend_numbers.py  -> paper/numbers_extend.tex (keys \\V{ex/...}), paper/tables/extend.tex

Reads
  extend/out/review_extra.json, extend/out/clmm.json   (src/review_extra.py, src/clmm.R; no new model calls)
  extend/results_seeta/B_claude.jsonl                  study B, user-side requests (copied from the run server)
  extend/results_seeta/A_claude.jsonl                  study A, other fields (if present)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
X = ROOT / "extend"
OUT = ROOT / "paper" / "numbers_extend.tex"
TAB = ROOT / "paper" / "tables" / "extend.tex"
RNG = np.random.default_rng(20261106)
EXPECTED = "claude-sonnet-4-6"


def s(x):
    return f"${x:+.2f}$"


def u(x):
    return f"${x:.2f}$"


def pv(p):
    return "$p<0.001$" if p < 0.001 else (f"$p={p:.3f}$" if p < 0.01 else f"$p={p:.2f}$")


def pct(x):
    return f"{x * 100:.0f}\\%"


def est(x):
    x = np.asarray(x, float)
    b = x[RNG.integers(0, len(x), (10000, len(x)))].mean(1)
    nz = x[x != 0]
    return dict(diff=float(x.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), n=int(len(x)),
                p=float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else 1.0)


def emit_est(V, k, e):
    V[f"{k}/ci"] = f"{s(e['diff'])} [{u(e['lo'])}, {u(e['hi'])}]"
    V[f"{k}/d"] = s(e["diff"])
    V[f"{k}/u"] = u(abs(e["diff"]))
    V[f"{k}/p"] = pv(e["p"])
    V[f"{k}/n"] = str(e["n"])


def mcnemar(a, b):
    """Exact McNemar on paired accept indicators: (only a, only b, p)."""
    ao, bo = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = float(stats.binomtest(ao, ao + bo).pvalue) if ao + bo else 1.0
    return ao, bo, p


def first_valid(path):
    seen = {}
    for l in open(path):
        r = json.loads(l)
        if r.get("role") == "sample" and r.get("ok") and str(r.get("provider_model") or "").startswith(EXPECTED):
            seen.setdefault((r["paper_id"], r["who"]), r)
    return pd.DataFrame([dict(pid=k[0], who=k[1], field=v["field"], rating=float(v["rating"]),
                              acc=int(v["recommendation"] == "accept")) for k, v in seen.items()])


def study_b(V):
    d = first_valid(X / "results_seeta" / "B_claude.jsonl")
    w = d.pivot(index="pid", columns="who", values="rating").dropna()
    a = d.pivot(index="pid", columns="who", values="acc").loc[w.index]
    V["ex/B/n"] = str(len(w))
    rows = []
    for who, lab in (("author", "Author"), ("author_frank", "Author, asking for frankness"),
                     ("author_detach", "Author, asking to judge it as someone else's"), ("reviewer", "Reviewer")):
        V[f"ex/B/mean/{who.replace('_', '.')}"] = f"{w[who].mean():.2f}"
        V[f"ex/B/accept/{who.replace('_', '.')}"] = pct(a[who].mean())
        if who != "reviewer":
            e = est(w[who] - w.reviewer)
            emit_est(V, f"ex/B/{who.replace('_', '.')}-reviewer", e)
            ao, ro, p = mcnemar(a[who], a.reviewer)
            V[f"ex/B/{who.replace('_', '.')}/flips"] = f"{ao} to {ro}"
            V[f"ex/B/{who.replace('_', '.')}/flips/p"] = pv(p)
            rows.append((lab, f"{w[who].mean():.2f}", pct(a[who].mean()), f"{s(e['diff'])} [{u(e['lo'])}, {u(e['hi'])}]", pv(e["p"]), f"{ao} : {ro}"))
        else:
            rows.append((lab, f"{w[who].mean():.2f}", pct(a[who].mean()), "--", "--", "--"))
    for x in ("author_frank", "author_detach"):
        emit_est(V, f"ex/B/author-{x.replace('_', '.')}", est(w.author - w[x]))
    # Equivalence as in the second registered study: 90% t interval and two one-sided t-tests against +-0.15.
    dd = (w.author_detach - w.reviewer).to_numpy(float)
    n, m, se = len(dd), dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd))
    t = stats.t.ppf(0.95, n - 1)
    p_tost = max(stats.t.sf((m + 0.15) / se, n - 1), stats.t.cdf((m - 0.15) / se, n - 1))
    V["ex/B/author.detach-reviewer/ci90"] = f"{s(m)} [{s(m - t * se)}, {s(m + t * se)}]"
    V["ex/B/author.detach-reviewer/tost"] = f"$p_{{\\mathrm{{TOST}}}}={p_tost:.2f}$"
    return rows


def study_a(V):
    p = X / "results_seeta" / "A_claude.jsonl"
    if not p.exists():
        return []
    d = first_valid(p)
    rows = []
    names = {"biology": "Biology (q-bio)", "physics": "Condensed matter (cond-mat)", "economics": "Economics (econ)",
             "ml": "Machine learning (cs.LG), same batch", "nonml": "Three fields pooled"}
    for f in ("biology", "physics", "economics", "nonml", "ml"):
        x = d[d.field != "ml"] if f == "nonml" else d[d.field == f]
        w = x.pivot(index="pid", columns="who", values="rating").dropna()
        a = x.pivot(index="pid", columns="who", values="acc").loc[w.index]
        if not len(w):
            continue
        e = est(w.author - w.reviewer)
        emit_est(V, f"ex/A/{f}", e)
        ao, ro, pm = mcnemar(a.author, a.reviewer)
        V[f"ex/A/{f}/flips"] = f"{ao} to {ro}"
        V[f"ex/A/{f}/flips/p"] = pv(pm)
        V[f"ex/A/{f}/mean/author"] = f"{w.author.mean():.2f}"
        V[f"ex/A/{f}/mean/reviewer"] = f"{w.reviewer.mean():.2f}"
        rows.append((names[f], str(len(w)), f"{w.author.mean():.2f}", f"{w.reviewer.mean():.2f}",
                     f"{s(e['diff'])} [{u(e['lo'])}, {u(e['hi'])}]", pv(e["p"]), f"{ao} : {ro}"))
    # Field difference: non-ML gap minus same-batch ML gap (unpaired, bootstrap over papers).
    g = d.pivot_table(index=["pid", "field"], columns="who", values="rating").dropna().reset_index()
    g["gap"] = g.author - g.reviewer
    nm, ml = g[g.field != "ml"].gap.to_numpy(), g[g.field == "ml"].gap.to_numpy()
    if len(nm) and len(ml):
        bs = [RNG.choice(nm, len(nm)).mean() - RNG.choice(ml, len(ml)).mean() for _ in range(10000)]
        V["ex/A/nonml-ml/ci"] = f"{s(nm.mean() - ml.mean())} [{u(np.percentile(bs, 2.5))}, {u(np.percentile(bs, 97.5))}]"
        V["ex/A/nonml-ml/p"] = pv(float(stats.mannwhitneyu(nm, ml).pvalue))
    return rows


def extra(V):
    R = json.loads((X / "out" / "review_extra.json").read_text())
    P = R["practical"]
    for k, lab in (("main_author_vs_reviewer", "main"), ("rerun_same_ask", "rerun"), ("confirm_cells_author_vs_reviewer", "confirm")):
        m = P[k]
        V[f"ex/move/{lab}/up"], V[f"ex/move/{lab}/down"] = pct(m["up"]), pct(m["down"])
        V[f"ex/move/{lab}/upn"], V[f"ex/move/{lab}/downn"], V[f"ex/move/{lab}/n"] = str(m["up_n"]), str(m["down_n"]), str(m["n"])
        V[f"ex/move/{lab}/net"] = f"{(m['up'] - m['down']) * 100:.0f}"
        V[f"ex/move/{lab}/signp"] = pv(m["sign_p"])
    C = R["cues"]
    V["ex/cue/n"], V["ex/cue/of"] = str(C["n_cue"]), f"{C['n_answers']:,}"
    for m in ("claude", "gemini", "deepseek", "glm"):
        V[f"ex/cue/{m}/papers"] = str(C[m]["papers_with_cue"])
        for col in ("main", "flash"):
            for kind in ("author-reviewer", "reader_gap"):
                e = C[m][f"{col}/no_cue/{kind}"]
                emit_est(V, f"ex/cue/{m}/{col}/{kind.replace('_', '.')}", e)
    G = R["gemini_id"]
    for ask in ("natural", "review"):
        for ident, lab in (("gemini-4-argon", "argon"), ("gemini-4-argon-high", "high"), ("all_pairs", "all")):
            emit_est(V, f"ex/gid/{ask}/{lab}", G[f"{ask}/{ident}"])
    K = json.loads((X / "out" / "clmm.json").read_text())
    for st, v in K.items():
        k = st.replace("_", ".")
        V[f"ex/clmm/{k}/or"] = f"{v['or']:.2f}"
        V[f"ex/clmm/{k}/ci"] = f"{v['or']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]"
        V[f"ex/clmm/{k}/p"] = pv(v["p"])
    for st, v in R["marginal_or"].items():
        k = st.replace("_", ".")
        V[f"ex/mor/{k}/ci"] = f"{v['or_']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]"
    return K


def table(rows_a, rows_b):
    L = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"\multicolumn{6}{@{}l}{\textbf{a}\enspace User-side requests (100 machine-learning papers, ICLR wording)}\\",
         r"Ask & Mean rating & Accept & $-$ Reviewer & $p$ & Accepted only here : only for reviewer \\", r"\midrule"]
    L += [" & ".join(r) + r" \\" for r in rows_b]
    if rows_a:
        L += [r"\midrule", r"\multicolumn{6}{@{}l}{\textbf{b}\enspace Other fields (``a journal'' wording), author $-$ reviewer}\\",
              r"Field & Papers & Author / reviewer & Difference & $p$ & Accepted only for author : only for reviewer \\", r"\midrule"]
        L += [" & ".join([r[0], r[1], f"{r[2]} / {r[3]}", r[4], r[5], r[6]]) + r" \\" for r in rows_a]
    L += [r"\bottomrule", r"\end{tabular}"]
    TAB.write_text("\n".join(L) + "\n")


def main():
    V = {}
    rb = study_b(V)
    ra = study_a(V)
    extra(V)
    table(ra, rb)
    lines = ["% Generated by src/extend_numbers.py. Do not edit.", r"\makeatletter"]
    lines += [f"\\@namedef{{num@{k}}}{{{v}}}" for k, v in sorted(V.items())]
    lines += [r"\makeatother"]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{len(V)} numbers -> {OUT}; study A rows: {len(ra)}")


if __name__ == "__main__":
    main()
