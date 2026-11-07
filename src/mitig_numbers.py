"""Numbers and table for the preregistered mitigation study (OSF sz6n4).

  python src/mitig_numbers.py  -> paper/numbers_mitig.tex, paper/tables/mitig.tex

Registered results come only from the frozen analysis output (mitig/results_156/mitig_stats_sample.json, copied
from the run server). The raw answer files in the same folder are read for counts (cells, serving identifiers,
missing cells and why) and for analyses that are NOT preregistered, marked "exploratory" below:
  S8 (registered but not implemented in the frozen script): W1 natural gap against the earlier studies, descriptive.
  refusal  Gemini first-turn answers that cite reviewer confidentiality, per cell; H1-H3 without papers that have any.
Keys are used as \\V{mt/...}.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "mitig" / "results_156"
OUT = ROOT / "paper" / "numbers_mitig.tex"
TAB = ROOT / "paper" / "tables" / "mitig.tex"
sys.path.insert(0, str(ROOT / "mitig" / "src"))
import analyze_mitig as am  # noqa: E402  (frozen analysis; only its pure functions are reused)

FILES = {"claude": ("form", "claude-sonnet-4-6"), "gemini": ("score", "gemini-4-argon")}
CONFIDENTIAL = re.compile(r"confidential", re.I)
FILTER = "blocked by Gemini's filters"


def s(x):
    return f"${x:+.2f}$"


def u(x):
    return f"${x:.2f}$"


def pv(p):
    return "$p<0.001$" if p < 0.001 else (f"$p={p:.3f}$" if p < 0.01 else f"$p={p:.2f}$")


def ci(e):
    return f"{s(e['est'])} [{u(e['lo'])}, {u(e['hi'])}]"


def ci90(e):
    lo, hi = e["ci90"]
    return f"{s(e['est'])} [{u(lo)}, {u(hi)}]"


def rows(path):
    return [json.loads(l) for l in open(path)]


def load(model):
    """Same rule as the frozen am.load: first valid sample record per cell from the registered identifier."""
    kind, expected = FILES[model]
    D = {}
    for r in rows(RES / f"{kind}_{model}.jsonl"):
        if r.get("ok") and r["role"] == "sample" and str(r.get("provider_model") or "").startswith(expected):
            D.setdefault(r["paper_id"], {}).setdefault((r["who"], r["ask"], r["family"]), r["parsed"])
    return D


def tests(D, rng):
    return {"H1": am.t_test(am.per_paper(D, "H1"), rng), "H2": am.t_test(am.per_paper(D, "H2"), rng),
            "H3": am.tost(am.per_paper(D, "H3"), rng)}


def exploratory_numbers():
    txt = (ROOT / "paper" / "numbers.tex").read_text() + (ROOT / "paper" / "numbers_confirm.tex").read_text()

    def get(k):
        return re.search(r"\\@namedef\{num@" + re.escape(k) + r"\}\{(.*)\}\s*$", txt, re.M).group(1)
    return {"claude_nat": get("models/claude/lmm.rating/struct/d"), "claude_fixed": get("fixed/claude/ci"),
            "gemini_nat": get("revision/selfscore/gemini.open/self/author-reviewer/d"),
            "gemini_fixed": get("revision/selfscore/gemini.fixedopen/self/author-reviewer/ci"),
            "confirm_w1": get("cf/H1/W1/d")}


def main():
    st = json.load(open(RES / "mitig_stats_sample.json"))
    V = {}

    def emit(k, v):
        V["mt/" + k] = v

    for m, r in st.items():
        P, S = r["primary"], r["secondary"]
        emit(f"{m}/papers", str(r["n_papers"]))
        emit(f"{m}/cells", f"{r['n_valid_cells']:,}")
        for h in ("H1", "H2", "H3"):
            e = P[h]
            emit(f"{m}/{h}/ci", ci(e))
            emit(f"{m}/{h}/d", s(e["est"]))
            emit(f"{m}/{h}/u", u(abs(e["est"])))
            emit(f"{m}/{h}/n", str(e["n"]))
            if h == "H3":
                emit(f"{m}/H3/ci90", ci90(e))
                emit(f"{m}/H3/hi90", u(e["ci90"][1]))
                emit(f"{m}/H3/tost", pv(e["p_tost"]).replace("$p", r"$p_{\mathrm{TOST}}"))
            else:
                emit(f"{m}/{h}/p", pv(e["p"]))
        emit(f"{m}/removes", "yes" if r["primary"]["removes_gap"] else "no")
        emit(f"{m}/share", f"{P['H2']['est'] / P['H1']['est'] * 100:.0f}\\%")
        for w in ("W1", "W2", "W3"):
            for h in ("H1", "H2", "H3"):
                emit(f"{m}/{w}/{h}/d", s(S["S1_by_wording"][w][h]["est"]))
                emit(f"{m}/{w}/{h}/ci", ci(S["S1_by_wording"][w][h]))
        it = S["S1_interaction"]
        emit(f"{m}/int", f"$F={it['F']:.2f}$, {pv(it['p'])}")
        for k, x in S["S2_accept_rate"].items():
            emit(f"{m}/accept/{k}", f"{x * 100:.0f}\\%")
        for k, f in S["S2_flips"].items():
            emit(f"{m}/flips/{k}", f"{f['author_only']} to {f['reviewer_only']}")
            emit(f"{m}/flips/{k}/p", pv(f["p"]))
        for k, x in S["S3_mean_rating"].items():
            emit(f"{m}/mean/{k}", u(x))
        emit(f"{m}/nrec", str(S["S4_without_recognized"]["n_recognized"]))
        emit(f"{m}/norec/H3/d", s(S["S4_without_recognized"]["H3"]["est"]))
        emit(f"{m}/avail/H3/d", s(S["S5_available_cases"]["H3"]["est"]))
        emit(f"{m}/wilcoxon/H3/ci90", ci90(S["S6_nonparametric"]["H3"]))
    sub = st["claude"]["secondary"]["S7_subscores"]
    emit("claude/major/H1/ci", ci(sub["major"]["H1"]))
    emit("claude/major/H3/ci", ci(sub["major"]["H3"]))

    # Earlier estimates for comparison (S8, descriptive; registered but not in the frozen script).
    ex = exploratory_numbers()
    for k, v in ex.items():
        emit(f"earlier/{k}", v)

    # Counts from the raw files.
    op = rows(RES / "open_gemini.jsonl")
    sc = rows(RES / "score_gemini.jsonl")
    first = {}
    for r in op:
        if r.get("ok") and r["role"] == "sample":
            first.setdefault((r["paper_id"], r["condition"]), r)
    served = Counter(str(r.get("provider_model")) for r in first.values())
    emit("gemini/first/argon", f"{served['gemini-4-argon']:,}")
    emit("gemini/first/high", f"{served['gemini-4-argon-high']:,}")
    scored = {}
    for r in sc:
        if r.get("ok") and r["role"] == "sample":
            scored.setdefault((r["paper_id"], r["condition"]), r)
    emit("gemini/score/high", f"{sum(1 for r in scored.values() if r.get('provider_model') == 'gemini-4-argon-high'):,}")
    missing = sorted(set(first) - set(scored))
    last = {}
    for r in sc:
        if r["role"] == "sample" and (r["paper_id"], r["condition"]) in missing:
            last[(r["paper_id"], r["condition"])] = r
    filt = [k for k in missing if FILTER in (last[k].get("raw") or "")]
    emit("gemini/missing", str(len(missing)))
    emit("gemini/missing/filter", str(len(filt)))
    emit("gemini/missing/filter/papers", str(len({p for p, _ in filt})))
    emit("gemini/missing/refuse", str(len(missing) - len(filt)))
    emit("gemini/missing/refuse/papers", str(len({k[0] for k in missing if k not in filt})))
    attempts = Counter((r["paper_id"], r["condition"]) for r in sc if r["role"] == "sample" and not r.get("ok"))
    emit("gemini/missing/attempts", str(min(attempts[k] for k in missing)))

    # Exploratory: first-turn answers that cite reviewer confidentiality.
    conf = {k for k, r in first.items() if CONFIDENTIAL.search(r.get("raw") or "")}
    for ask in ("natural", "review"):
        for who in ("author", "reviewer"):
            ks = [k for k in first if k[1].startswith(f"{ask}_{who}_")]
            emit(f"gemini/conf/{ask}/{who}", f"{sum(1 for k in ks if k in conf)}")
            emit(f"gemini/conf/{ask}/{who}/of", f"{len(ks):,}")
    bad = {p for p, _ in conf}
    emit("gemini/conf/papers", str(len(bad)))
    rng = np.random.default_rng(am.SEED)
    D = {p: c for p, c in load("gemini").items() if p not in bad}
    T = tests(D, rng)
    for h in ("H1", "H2", "H3"):
        emit(f"gemini/noconf/{h}/ci", ci(T[h]))
        emit(f"gemini/noconf/{h}/n", str(T[h]["n"]))
    emit("gemini/noconf/H3/ci90", ci90(T["H3"]))

    lines = ["% Generated by src/mitig_numbers.py from mitig/results_156/. Do not edit.", r"\makeatletter"]
    lines += [r"\@namedef{num@%s}{%s}" % (k, v) for k, v in sorted(V.items())]
    lines.append(r"\makeatother")
    OUT.write_text("\n".join(lines) + "\n")

    # Table
    def grey(lo, hi):
        return r"{\scriptsize\color{black!60}[%s, %s]}" % (u(lo), u(hi))

    def star(p):
        return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""

    def bold(x):
        return r"{\boldmath" + x + "}"

    name = {"claude": "Claude Sonnet~4.6, review form", "gemini": "Gemini~4 Argon, free text"}
    t = [r"\begin{tabular}{@{}l rrr rrr c@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Registered tests} & \multicolumn{3}{c}{H3 by wording} & Gap \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r" & H1 natural gap & H2 reduction & H3 review gap & W1 & W2 & W3 & removed \\", r"\midrule"]
    for m, r in st.items():
        P, S = r["primary"], r["secondary"]["S1_by_wording"]
        r1 = [name[m], bold(s(P["H1"]["est"])) + star(P["H1"]["p"]), bold(s(P["H2"]["est"])) + star(P["H2"]["p"]),
              bold(s(P["H3"]["est"]))] + [s(S[w]["H3"]["est"]) for w in ("W1", "W2", "W3")]
        r1.append("yes" if r["primary"]["removes_gap"] else "no")
        ns = sorted({P[h]["n"] for h in ("H1", "H2", "H3")})
        r2 = [r"\quad " + ("--".join(str(n) for n in (ns[0], ns[-1])) if len(ns) > 1 else str(ns[0])) + " papers",
              grey(P["H1"]["lo"], P["H1"]["hi"]), grey(P["H2"]["lo"], P["H2"]["hi"]), grey(*P["H3"]["ci90"])]
        r2 += [grey(S[w]["H3"]["lo"], S[w]["H3"]["hi"]) for w in ("W1", "W2", "W3")] + [""]
        t += [" & ".join(r1) + r" \\", " & ".join(r2) + r" \\", r"\addlinespace[4pt]"]
    t[-1] = r"\bottomrule"
    t.append(r"\end{tabular}")
    TAB.write_text("\n".join(t) + "\n")
    print(f"{len(V)} numbers -> {OUT.relative_to(ROOT)}; table -> {TAB.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
