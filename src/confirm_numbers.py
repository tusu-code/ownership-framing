"""Numbers and table for the preregistered confirmatory study.

  python src/confirm_numbers.py  -> paper/numbers_confirm.tex, paper/tables/confirm.tex

Reads only the output of the frozen analysis (confirm/results/confirm_stats_sample.json)
and, for the exploratory column, paper/numbers.tex. Keys are used as \\V{cf/...}.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "confirm" / "results" / "confirm_stats_sample.json"
OUT = ROOT / "paper" / "numbers_confirm.tex"
TAB = ROOT / "paper" / "tables" / "confirm.tex"

# Exploratory Claude review-form estimates that each hypothesis tests (preregistration, section 1).
EXPL = {"H1": "models/claude/lmm.rating/struct/d", "H2": "decomp/claude/ownership/lmm.rating/struct/d",
        "H3": "baseline/claude/reviewer/lmm.rating/struct/d", "H4": "baseline/claude/author/lmm.rating/struct/d"}
FLIP = {"H3": True, "H4": True}  # stored as relation - role-free; hypotheses are role-free - relation
LABEL = {"H1": r"H1 author $-$ reviewer", "H2": r"H2 author $-$ colleague",
         "H3": r"H3 role-free $-$ reviewer", "H4": r"H4 role-free $-$ author"}


def s(x):
    return f"${x:+.2f}$"


def u(x):
    return f"${x:.2f}$"


def pv(p):
    return "$p<0.001$" if p < 0.001 else (f"$p={p:.3f}$" if p < 0.01 else f"$p={p:.2f}$")


def ci(e):
    return f"{s(e['est'])} [{u(e['lo'])}, {u(e['hi'])}]"


def exploratory():
    txt = (ROOT / "paper" / "numbers.tex").read_text()
    out = {}
    for h, k in EXPL.items():
        m = re.search(r"\\@namedef\{num@" + re.escape(k) + r"\}\{\$([+-][0-9.]+)\$\}", txt)
        x = float(m.group(1))
        out[h] = -x if FLIP.get(h) else x
    return out


def post_hoc(r, emit):
    """Not preregistered (added 4 November 2026 after the confirmatory result): does the main study's paraphrase
    explain why W1 gave a smaller author - reviewer gap on original text? Claude review form, main study, by the
    model that paraphrased the paper, and against the share of original 8-grams left; W1 against the main study."""
    import numpy as np
    import pandas as pd
    from scipy import stats
    rng = np.random.default_rng(20261104)
    c = pd.read_csv(ROOT / "full_study" / "out" / "cells.csv")
    d = c[(c.model == "claude") & (c.fmt == "struct")]
    w = d.pivot_table(index="paper_id", columns="who", values="rating")
    w = w.join(d.groupby("paper_id")[["disguise", "overlap8"]].first())
    w["ar"] = w.author - w.reviewer
    w = w.dropna(subset=["ar"])

    def boot(x):
        x = np.asarray(x, float)
        bs = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
        return {"est": x.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "n": len(x)}

    allp = boot(w.ar)
    emit("ph/main/ci", ci(allp))
    for g, k in (("haiku-4.5", "haiku"), ("v4.1-flash", "flash")):
        e = boot(w[w.disguise == g].ar)
        emit(f"ph/{k}/ci", ci(e))
        emit(f"ph/{k}/n", str(e["n"]))
    hk, fl = w[w.disguise == "haiku-4.5"].ar, w[w.disguise == "v4.1-flash"].ar
    emit("ph/rewriter/diff", s(hk.mean() - fl.mean()))
    emit("ph/rewriter/p", pv(stats.ttest_ind(hk, fl, equal_var=False).pvalue))
    rho = stats.spearmanr(w.ar, w.overlap8)
    emit("ph/overlap/rho", f"$\\rho={rho.statistic:+.2f}$")
    emit("ph/overlap/p", pv(rho.pvalue))
    w1 = r["secondary"]["S1_by_wording"]["H1"]["W1"]
    se1, se0 = (w1["hi"] - w1["lo"]) / 3.92, (allp["hi"] - allp["lo"]) / 3.92
    z = (w1["est"] - allp["est"]) / np.hypot(se1, se0)
    emit("ph/w1vsmain/diff", s(w1["est"] - allp["est"]))
    emit("ph/w1vsmain/p", pv(2 * stats.norm.sf(abs(z))))


def main():
    r = json.load(open(STATS))
    sec = r["secondary"]
    ex = exploratory()
    V = {}

    def emit(k, v):
        V["cf/" + k] = v

    emit("n", str(r["n_papers"]))
    emit("cells", f"{r['n_valid_cells']:,}")
    for c in r["confirmatory"]:
        h = c["test"]
        emit(f"{h}/ci", ci(c))
        emit(f"{h}/d", s(c["est"]))
        emit(f"{h}/u", u(abs(c["est"])))
        emit(f"{h}/p", pv(c["p"]))
        emit(f"{h}/holm", pv(c["p_holm"]).replace("$p", r"$p_{\mathrm{Holm}}"))
        emit(f"{h}/expl", s(ex[h]))
        for w in ("W1", "W2", "W3"):
            e = sec["S1_by_wording"][h][w]
            emit(f"{h}/{w}/ci", ci(e))
            emit(f"{h}/{w}/d", s(e["est"]))
            emit(f"{h}/{w}/p", pv(e["p"]))
        it = sec["S1_interaction"][h]
        emit(f"{h}/int", f"$F={it['F']:.2f}$, {pv(it['p'])}")
        emit(f"{h}/int/p", pv(it["p"]))
        e = sec["S4_without_recognized"][h]
        emit(f"{h}/norec/ci", ci(e))
    for rel, x in sec["S2_accept_rate"].items():
        emit(f"accept/{rel}", f"{x * 100:.0f}\\%")
    for w, f in sec["S2_flips"].items():
        emit(f"flips/{w}", f"{f['author_only']} to {f['reviewer_only']}")
        emit(f"flips/{w}/p", pv(f["p"]))
    for k, h in (("soundness", "H1"), ("major", "H1"), ("major", "H2")):
        e = sec["S3_subscores"][k][h]
        emit(f"{k}/{h}/ci", ci(e))
        emit(f"{k}/{h}/u", u(abs(e["est"])))
    emit("nrec", str(sec["S4_without_recognized"]["n_recognized"]))
    m = sec["mean_rating"]
    for rel in ("author", "colleague", "reviewer", "rolefree"):
        emit(f"mean/{rel}", u(sum(m[f"{rel}_{w}"] for w in ("W1", "W2", "W3")) / 3))

    post_hoc(r, emit)

    lines = ["% Generated by src/confirm_numbers.py from confirm/results/confirm_stats_sample.json. Do not edit.",
             r"\makeatletter"]
    lines += [r"\@namedef{num@%s}{%s}" % (k, v) for k, v in sorted(V.items())]
    lines.append(r"\makeatother")
    OUT.write_text("\n".join(lines) + "\n")

    # Table 2
    def star(p):
        return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""

    def grey(e):
        return r"{\scriptsize\color{black!60}[%s, %s]}" % (u(e["lo"]), u(e["hi"]))

    def bold(x):
        return r"{\boldmath" + x + "}"

    t = [r"\begin{tabular}{@{}l r r rrr r@{}}", r"\toprule",
         r" & Main study & Registered & \multicolumn{3}{c}{By wording (unadjusted)} & Wording \\",
         r"\cmidrule(lr){3-3}\cmidrule(lr){4-6}",
         r"Hypothesis & 342 papers & mean, 160 papers & W1 & W2 & W3 & $\times$ relation $p$ \\", r"\midrule"]
    for c in r["confirmatory"]:
        h = c["test"]
        S1 = sec["S1_by_wording"][h]
        row1 = [LABEL[h], s(ex[h]), bold(s(c["est"])) + star(c["p_holm"])]
        row1 += [s(S1[w]["est"]) + star(S1[w]["p"]) for w in ("W1", "W2", "W3")]
        row1.append(f"${sec['S1_interaction'][h]['p']:.3f}$".replace("$0.000$", "$<0.001$"))
        row2 = ["", "", grey(c)] + [grey(S1[w]) for w in ("W1", "W2", "W3")] + [""]
        t += [" & ".join(row1) + r" \\", " & ".join(row2) + r" \\", r"\addlinespace[4pt]"]
    t[-1] = r"\bottomrule"
    t.append(r"\end{tabular}")
    TAB.write_text("\n".join(t) + "\n")
    print(f"{len(V)} numbers -> {OUT.relative_to(ROOT)}; table -> {TAB.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
