"""LaTeX tables for the paper, generated from full_study/out/stats.json (no hand-copied numbers).

  python src/paper_tables.py   -> paper/tables/*.tex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "full_study" / "out" / "stats.json").read_text())
OUT = ROOT / "paper" / "tables"
MODELS = ["claude", "gemini", "gpt", "glm", "deepseek"]
LABEL = {"claude": "Claude Sonnet 4.6", "gpt": "GPT-6.1 Sol", "glm": "GLM-5.3", "deepseek": "DeepSeek V4 Pro",
         "gemini": "Gemini 4 Argon", "grok": "Grok 4.7"}
VAR = {"rating": "Rating (0--10)", "tone": "Tone (1--7)", "fixable": "Problems framed as fixable", "n_major": "Major concerns (count)",
       "accept": "Accept (share)", "soundness": "Soundness (1--4)", "contribution": "Contribution (1--4)"}


def p(x):
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def s(x, k=2):
    return f"${x:+.{k}f}$".replace("+", "+")


def ci(e, k=2):
    return f"{s(e['diff'] if 'diff' in e else e['est'], k)} [{e['lo']:+.{k}f}, {e['hi']:+.{k}f}]".replace("[", "[$").replace("]", "$]").replace(", ", "$, $")


def primary():
    pt = {(t["model"], t["contrast"]): t for t in R["primary_tests"]}
    L = [r"\begin{tabular}{llrrr}", r"\toprule", r"Model & Contrast & Estimate [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    names = {"open": "Open ask", "struct": "Review form", "interaction": "Open $-$ form"}
    tk = {"open": "open", "struct": "struct", "interaction": "open − struct"}
    for i, m in enumerate(MODELS):
        l = R["models"][m]["lmm_rating"]
        for j, k in enumerate(("open", "struct", "interaction")):
            e = l[k]
            L.append(f"{LABEL[m] if j == 0 else ''} & {names[k]} & {ci(e)} & {p(e['p'])} & {p(pt[(m, tk[k])]['p_adj'])} \\\\")
        if i < len(MODELS) - 1:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def secondary():
    st = {(t["model"], t["outcome"], t["contrast"]): t for t in R["secondary_tests"]}
    L = [r"\begin{tabular}{lllrrrrr}", r"\toprule",
         r"Model & Format & Outcome & Author & Reviewer & Difference [95\% CI] & $d_z$ & $p_{\mathrm{BH}}$ \\", r"\midrule"]
    for i, m in enumerate(MODELS):
        first = True
        for fmt in ("open", "struct"):
            for v, e in R["models"][m]["paired"][fmt].items():
                if not e:
                    continue
                t = st.get((m, v, fmt))
                pa = p(t["p_adj"]) if t else (r"\ref{tab:main}" if fmt == "struct" else r"\ref{tab:rating}")
                k = 2
                L.append(f"{LABEL[m] if first else ''} & {'Open' if fmt == 'open' else 'Form'} & {VAR[v]} & {e['author']:.2f} & {e['reviewer']:.2f} & {ci(e)} & {e['d_z']:+.2f} & {pa} \\\\")
                first = False
        if i < len(MODELS) - 1:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def accept():
    L = [r"\begin{tabular}{llrrrl}", r"\toprule",
         r"Model & Format & Accept, author / reviewer & Discordant pairs (A:R) & McNemar $p$ & GEE odds ratio [95\% CI] \\", r"\midrule"]
    for i, m in enumerate(MODELS):
        for j, fmt in enumerate(("open", "struct")):
            a = R["models"][m]["accept"][fmt]
            mc, g = a["mcnemar"], a["gee"]
            gs = f"{g['or_']:.2f} [{g['lo']:.2f}, {g['hi']:.2f}]" if g else "-- (too few accepts)"
            L.append(f"{LABEL[m] if j == 0 else ''} & {'Open' if fmt == 'open' else 'Form'} & {mc['acc_author']*100:.0f}\\% / {mc['acc_reviewer']*100:.0f}\\% & "
                     f"{mc['author_only']}:{mc['reviewer_only']} & {p(mc['p'])} & {gs} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def border():
    st = {(t["model"], t["outcome"], t["contrast"]): t for t in R["secondary_tests"]}
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Model & Format & Borderline [95\% CI] & Other [95\% CI] & Interaction $p$ ($p_{\mathrm{BH}}$) & Accept flips, border / other (A:R) (int.\ $p$; $p_{\mathrm{BH}}$) \\", r"\midrule"]
    for m in MODELS:
        for j, fmt in enumerate(("open", "struct")):
            sx = R["models"][m]["strat"][fmt]
            sa = R["models"][m]["strat_accept"][fmt]
            it = sx["interaction"]
            t = st.get((m, "rating author x borderline", fmt))
            fl = " / ".join(f"{sa[k]['author_only']}:{sa[k]['reviewer_only']}" for k in ("border", "other"))
            ia = sa.get("interaction")
            fl += f" ({p(ia['p'])}; {p(st[(m, 'accept author x borderline', fmt)]['p_adj'])})" if ia else ""
            L.append(f"{LABEL[m] if j == 0 else ''} & {'Open' if fmt == 'open' else 'Form'} & {ci(sx['border'])} & {ci(sx['other'])} & "
                     f"{p(it['p'])} ({p(t['p_adj'])}) & {fl} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def robust():
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Model & Raw-answer judge & Indistinguishable pairs ($n$) & Rewritten by Haiku ($n$) & Rewritten by Flash ($n$) \\", r"\midrule"]
    for m in MODELS:
        r = R["models"][m]
        raw = r["paired"]["open_raw_judge"]["rating"]
        ind = r["indistinguishable"]
        dz = r["robust"]["disguise"]
        L.append(f"{LABEL[m]} & {ci(raw)} & {ci(ind)} ({ind['n']}) & {ci(dz['haiku-4.5'])} ({dz['haiku-4.5']['n']}) & {ci(dz['v4.1-flash'])} ({dz['v4.1-flash']['n']}) \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def discrim():
    L = [r"\begin{tabular}{llrrr}", r"\toprule", r"Model & Format & AUC, author ask & AUC, reviewer ask & Difference, 95\% CI \\", r"\midrule"]
    for m in MODELS:
        for j, fmt in enumerate(("open", "struct")):
            v = R["discrimination"][f"{m}_{fmt}"]
            L.append(f"{LABEL[m] if j == 0 else ''} & {'Open' if fmt == 'open' else 'Form'} & {v['auc_author']:.3f} & {v['auc_reviewer']:.3f} & [{v['lo']:+.3f}, {v['hi']:+.3f}] \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def decomp():
    dt = {(t["model"], t["outcome"], t["contrast"]): t for t in R["decomp_tests"]}
    st = {(t["model"], t["outcome"], t["contrast"]): t for t in R["secondary_tests"]}
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Model & Format & Ownership: author $-$ colleague & $p_{\mathrm{Holm}}$ & Role and stage: colleague $-$ reviewer & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    for m, fmt in [(m, "open") for m in MODELS] + [("claude", "struct")]:
        o = R["decomp"][m]["ownership"]["lmm_rating"][fmt]
        r = R["decomp"][m]["role_stage"]["lmm_rating"][fmt]
        L.append(f"{LABEL[m]} & {'Open' if fmt == 'open' else 'Form'} & {ci(o)} & {p(dt[(m, 'rating author - colleague', fmt)]['p_adj'])} & "
                 f"{ci(r)} & {p(dt[(m, 'rating colleague - reviewer', fmt)]['p_adj'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def decomp_secondary():
    st = {(t["model"], t["outcome"], t["contrast"]): t for t in R["secondary_tests"]}
    L = [r"\begin{tabular}{llllrrrr}", r"\toprule",
         r"Model & Format & Contrast & Outcome & First & Second & Difference [95\% CI] & $p_{\mathrm{BH}}$ \\", r"\midrule"]
    for i, m in enumerate(LATE):
        first = True
        blocks = [("A $-$ C", "author - colleague", R["decomp"][m]["ownership"]["paired"])]
        blocks += [(f"{w[0].upper()} $-$ RF", f"{w} - venue", R["baseline"][m][w]["paired"]) for w in ("author", "colleague", "reviewer")]
        for tex, lab, vsets in blocks:
            for fmt, vs in vsets.items():
                for v, e in vs.items():
                    if not e or v == "rating":
                        continue
                    t = st.get((m, f"{v} {lab}", fmt))
                    L.append(f"{LABEL[m] if first else ''} & {'Open' if fmt == 'open' else 'Form'} & {tex} & {VAR[v]} & {e['author']:.2f} & {e['reviewer']:.2f} & {ci(e)} & {p(t['p_adj']) if t else ''} \\\\")
                    first = False
        if i < len(LATE) - 1:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def secondary_long():
    """Rows of secondary as a longtable body."""
    return "\n".join(secondary().split("\n")[4:-2])


def decomp_secondary_long():
    """Rows of decomp_secondary as a longtable body (the table runs over more than one page)."""
    L = decomp_secondary().split("\n")
    return "\n".join(L[4:-2])


def manip():
    order = ["author", "colleague", "reviewer", "venue", "neutral"]
    L = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
         r" & \multicolumn{5}{c}{Gives pre-submission advice} & \multicolumn{5}{c}{Mentions ICLR} \\",
         r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
         r"Model & A & C & R & RF & RF$_0$ & A & C & R & RF & RF$_0$ \\", r"\midrule"]
    for m in MODELS:
        v = R["manip_all"][m]
        cell = lambda w, k: f"{v[w][k]*100:.0f}\\%" if w in v else "--"
        L.append(f"{LABEL[m]} & " + " & ".join(cell(w, "addr_author") for w in order) + " & "
                 + " & ".join(cell(w, "mentions_iclr") for w in order) + " \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def rolefree():
    order = ["author", "colleague", "reviewer", "venue", "neutral"]
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Model & Format & Author & Colleague & Reviewer & Role-free & Role-free, no venue \\", r"\midrule"]
    for m in MODELS:
        for fmt, v in R["means_by_ask"][m].items():
            if not v:
                continue
            L.append(f"{LABEL[m]} & {'Open' if fmt == 'open' else 'Form'} & " + " & ".join(
                f"{v[w]['rating']:.2f} ({v[w]['accept']*100:.0f}\\%)" if w in v else "--" for w in order) + " \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


LATE = ["claude", "gemini", "gpt", "glm"]   # models whose later asks were run on a stable model version (see stats_full.py)


def star(pa):
    return "$^{***}$" if pa < 0.001 else "$^{**}$" if pa < 0.01 else "$^{*}$" if pa < 0.05 else ""


def cis(e, pa):
    return f"{e['est']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}]{star(pa)}".replace("[", "[$").replace("]", "$]").replace(", ", "$, $").replace("+", "+", 1)


def main_rating():
    """Table 1: rating differences between asks, with Holm-adjusted significance from each family."""
    pt = {(t["model"], t["contrast"]): t for t in R["primary_tests"]}
    dt = {(t["model"], t["outcome"], t["contrast"]): t for t in R["decomp_tests"]}
    bt = {(t["model"], t["outcome"], t["contrast"]): t for t in R["baseline_tests"]}
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r" & & & & \multicolumn{3}{c}{Relative to the role-free ask} \\", r"\cmidrule(lr){5-7}",
         r"Model & Format & Author $-$ reviewer & Author $-$ colleague & Author & Colleague & Reviewer \\", r"\midrule"]
    for i, m in enumerate(MODELS):
        for j, fmt in enumerate(("open", "struct")):
            row = [f"${R['models'][m]['lmm_rating'][fmt]['est']:+.2f}$ [${R['models'][m]['lmm_rating'][fmt]['lo']:+.2f}$, ${R['models'][m]['lmm_rating'][fmt]['hi']:+.2f}$]"
                   + star(pt[(m, fmt)]["p_adj"])]
            own = R["decomp"].get(m, {}).get("ownership", {}).get("lmm_rating", {}).get(fmt)
            row.append(f"${own['est']:+.2f}$ [${own['lo']:+.2f}$, ${own['hi']:+.2f}$]" + star(dt[(m, 'rating author - colleague', fmt)]["p_adj"]) if own else "--")
            for who in ("author", "colleague", "reviewer"):
                e = R["baseline"].get(m, {}).get(who, {}).get("lmm_rating", {}).get(fmt)
                row.append(f"${e['est']:+.2f}$ [${e['lo']:+.2f}$, ${e['hi']:+.2f}$]" + star(bt[(m, f'rating {who} - venue', fmt)]["p_adj"]) if e else "--")
            L.append(f"{LABEL[m] if j == 0 else ''} & {'Open' if fmt == 'open' else 'Form'} & " + " & ".join(row) + " \\\\")
        if i < len(MODELS) - 1:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def retest():
    L = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Model & Format & Shift, author & Shift, reviewer & Same rating & A $-$ R, original & A $-$ R, re-run & A $-$ role-free & Role-free $-$ R \\", r"\midrule"]
    f_ = lambda e: f"${e['diff']:+.2f}$ [${e['lo']:+.2f}$, ${e['hi']:+.2f}$]" if e else "--"
    for m in MODELS:
        for fmt in ("open", "struct"):
            r = R["retest"].get(f"{m}_{fmt}")
            if not r:
                continue
            L.append(f"{LABEL[m]} & {'Open' if fmt == 'open' else 'Form'} & {f_(r['drift']['author'])} & {f_(r['drift']['reviewer'])} & "
                     f"{r['same']*100:.0f}\\% & {f_(r['ar_old'])} & {f_(r['ar_new'])} & {f_(r.get('same_batch_author_venue'))} & {f_(r.get('same_batch_venue_reviewer'))} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def judge():
    L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Model & $n$ & Spearman $\rho$ & Exact rating & Same decision & Mean error & Error, A $-$ R ($p$) & A $-$ R, own form & A $-$ R, judge \\", r"\midrule"]
    for m in MODELS:
        v = R["judge_validity"][m]
        L.append(f"{LABEL[m]} & {v['n']} & {v['spearman']:.2f} & {v['exact']*100:.0f}\\% & {v['accept_agree']*100:.0f}\\% & "
                 f"{v['mean_bias']:+.2f} & {v['bias_author_minus_reviewer']:+.2f} ({p(v['p'])}) & {ci(v['ar_self'])} & {ci(v['ar_judge'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def fmt_e(e, k="diff"):
    return f"${e[k]:+.2f}$ [${e['lo']:+.2f}$, ${e['hi']:+.2f}$]"


def main_table():
    """Table 1: author effect on the models' own scores; estimate on the first line, 95% interval below."""
    pt = {(t["model"], t["contrast"]): t for t in R["primary_tests"]}
    dt = {(t["model"], t["outcome"], t["contrast"]): t for t in R["decomp_tests"]}
    st = {(t["contrast"], t["outcome"]): t for t in R["self_tests"]}
    ss = R["revision"]["selfscore"]
    short = {"claude": "Claude Sonnet 4.6", "gemini": "Gemini 4 Argon", "deepseek": "DeepSeek V4 Pro", "glm": "GLM-5.3", "gpt": "GPT-6.1 Sol"}

    def cell(e, k="diff", p=None):
        if not e:
            return ("--", "")
        return (f"${e[k]:+.2f}${star(p) if p is not None else ''}", f"[${e['lo']:.2f}$, ${e['hi']:.2f}$]")
    L = [r"\begin{tabular}{@{}l@{\hspace{6pt}}rrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Review form} & \multicolumn{2}{c}{Free text, second turn} & \multicolumn{2}{c}{Fixed instruction} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r" & A $-$ R & A $-$ C & Accept, A / R & A $-$ R & A $-$ C & Form & Free text \\", r"\midrule"]
    for n, m in enumerate(["claude", "gemini", "deepseek", "glm", "gpt"]):
        k = "deepseek_open_r3" if m == "deepseek" else f"{m}_open"
        own = R["decomp"].get(m, {}).get("ownership", {}).get("lmm_rating", {}).get("struct")
        cal = R["calibration"][m]["struct"]
        c = [cell(R["models"][m]["lmm_rating"]["struct"], "est", pt[(m, "struct")]["p_adj"]),
             cell(own, "est", dt[(m, "rating author - colleague", "struct")]["p_adj"]) if own else ("--", ""),
             (f"{cal['author']['accept']*100:.0f}\\% / {cal['reviewer']['accept']*100:.0f}\\%", "")]
        for cc in ("author - reviewer", "author - colleague"):
            x = ss.get(k, {}).get("self", {}).get(cc)
            t = st.get((k, cc))
            c.append(cell(x, "diff", t["p_adj"] if t else None) if x else ("--", ""))
        c.append(cell(R["fixed"][m]))
        x = ss.get(f"{m}_fixedopen", {}).get("self", {}).get("author - reviewer")
        c.append(cell(x) if x else ("--", ""))
        L.append(short[m] + " & " + " & ".join(a for a, _ in c) + r" \\")
        L.append(" & " + " & ".join(r"{\scriptsize\color{black!60}" + b + "}" if b else "" for _, b in c) + r" \\")
        if n < 4:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def judges():
    """Supplementary: author - reviewer on the same free-text answers, scored by the model itself and by two judges."""
    ss = R["revision"]["selfscore"]
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r" & \multicolumn{3}{c}{Same answers (second-turn subsample)} & \multicolumn{2}{c}{All papers, judges only} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
         r"Model & Own score & MiniMax M3.1 Flash & DeepSeek V4.1 Flash & MiniMax M3.1 Flash & DeepSeek V4.1 Flash \\", r"\midrule"]
    for m in ["claude", "gemini", "deepseek", "glm", "gpt"]:
        k = "deepseek_open_r3" if m == "deepseek" else f"{m}_open"
        x = ss.get(k)
        cells = [fmt_e(x[c]["author - reviewer"]) + f" ({x[c]['author - reviewer']['n']})" if x and "author - reviewer" in x.get(c, {}) else "--"
                 for c in ("self", "main", "flash")]
        cells.append(fmt_e(R["models"][m]["lmm_rating"]["open"], "est"))
        f = R["judge2"]["models"].get(m, {}).get("flash", {}).get("author - reviewer")
        cells.append(fmt_e(f) if f else "--")
        L.append(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def tone():
    t = R["revision"]["tone"]
    L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Judge & Tone (1--7) & Problems framed as fixable & Implied rating (0--10) \\", r"\midrule"]
    for j, lab in (("dsflash", "DeepSeek V4.1 Flash"), ("minimax", "MiniMax M3.1 Flash")):
        L.append(f"{lab} & " + " & ".join(fmt_e(t[j][v]) for v in ("tone", "fixable", "rating")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def ladder():
    lab = {"open": "Natural ask before the manuscript", "ablorder": "Same ask, after the manuscript",
           "ablrole": r"``Review it as a reviewer \dots\ would'' instead of ``tell me what you think''",
           "ablmerits": r"\dots\ plus ``Judge it on its merits''", "fixedopen": "Fixed instruction, free text"}
    L = [r"\begin{tabular}{lr}", r"\toprule", r"Step & Author $-$ reviewer, own score \\", r"\midrule"]
    for k, v in R["revision"]["ladder"].items():
        L.append(f"{lab[k]} & {fmt_e(v)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def recognition():
    rv = R["revision"]
    name = {"claude": "Claude Sonnet 4.6", "gemini": "Gemini 4 Argon", "gemini38": "Gemini 3.8 Flash", "glm": "GLM-5.3", "deepseek": "DeepSeek V4 Pro"}
    L = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Model & Opening shown & Papers & Claims to recognize & Names the true title & Decision correct / stated \\", r"\midrule"]
    for m in ["claude", "gemini", "gemini38", "deepseek", "glm"]:
        for v in ("disguised", "original"):
            x = rv["recognize"].get(f"{m}_{v}")
            if x:
                L.append(f"{name[m] if v == 'disguised' else ''} & {v} & {x['n']} & {x['claimed']*100:.1f}\\% & {x['hit']*100:.1f}\\% & "
                         f"{x['decision_correct']} / {x['decision_stated']} \\\\")
    L += [r"\midrule", r"\multicolumn{6}{l}{Without the " + str(len(rv["named_union"])) + r" papers any model named: author $-$ reviewer, own score} \\"]
    ex = rv["selfscore_excl"]
    for k, lab in (("claude_open", "Claude, second turn"), ("gemini_open", "Gemini, second turn"), ("deepseek_open_r3", "DeepSeek, second turn"), ("glm_open", "GLM, second turn")):
        if k in ex:
            L.append(f"\\multicolumn{{3}}{{l}}{{{lab}}} & \\multicolumn{{3}}{{r}}{{{fmt_e(ex[k]['author - reviewer'])} ({ex[k]['author - reviewer']['n']})}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def calib_own():
    c = R["revision"]["calibration_excl"]
    rows = [("claude_struct", "Claude Sonnet 4.6, review form"), ("claude_open", "Claude Sonnet 4.6, second turn"),
            ("gemini_struct", "Gemini 4 Argon, review form"), ("gemini_open", "Gemini 4 Argon, second turn"),
            ("glm_open", "GLM-5.3, second turn"), ("claude_fixed", "Claude Sonnet 4.6, fixed instruction"), ("gemini_fixed", "Gemini 4 Argon, fixed instruction")]
    L = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Model and score & Author & Colleague & Reviewer & Role-free \\", r"\midrule"]
    for k, lab in rows:
        if k in c:
            L.append(lab + " & " + " & ".join(fmt_e(c[k][w]) if w in c[k] else "--" for w in ("author", "colleague", "reviewer", "venue")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def same_batch():
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r" & & \multicolumn{3}{c}{Rating difference} & \multicolumn{3}{c}{Rating $-$ human reviewers} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Model & Format & Author $-$ reviewer & Author $-$ colleague & Own paper under review $-$ reviewer & Author & Colleague & Reviewer \\", r"\midrule"]
    for k, v in R["same_batch"].items():
        m, fmt = k.rsplit("_", 1)
        c = v["contrasts"]
        L.append(f"{LABEL[m]} & {'Open' if fmt == 'open' else 'Form'} & {fmt_e(c['author - reviewer'])} & {fmt_e(c['author - colleague'])} & "
                 f"{fmt_e(c['author_review - reviewer'])} & {fmt_e(v['bias']['author'])} & {fmt_e(v['bias']['colleague'])} & {fmt_e(v['bias']['reviewer'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def judge2():
    j = R["judge2"]
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r" & & \multicolumn{2}{c}{Author $-$ reviewer} & \multicolumn{2}{c}{Author $-$ colleague} \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Model & Spearman $\rho$ & Main judge & Second judge & Main judge & Second judge \\", r"\midrule"]
    for m in ("claude", "glm", "deepseek"):
        v = j["models"][m]
        ac = lambda col: fmt_e(v[col]["author - colleague"]) if "author - colleague" in v[col] else "--"
        L.append(f"{LABEL[m]} & {v['spearman']:.2f} & {fmt_e(v['flash']['author - reviewer'])} & {fmt_e(v['haiku']['author - reviewer'])} & {ac('flash')} & {ac('haiku')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def grok():
    """Supplementary: Grok 4.7, four open asks on the 100 retest papers (neutral judge)."""
    g = R["grok"]
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Contrast & Rating (0--10) & Tone (1--7) & Fixable (share) & Major concerns & Words \\", r"\midrule"]
    names = {"author - reviewer": "Author $-$ reviewer", "author - colleague": "Author $-$ colleague",
             "colleague - reviewer": "Colleague $-$ reviewer", "author - venue": "Author $-$ role-free",
             "venue - reviewer": "Role-free $-$ reviewer"}
    for k, lab in names.items():
        c = g["contrasts"][k]
        cells = [fmt_e(c[v]) if v != "words" else f"${c[v]['diff']:+.0f}$ [${c[v]['lo']:+.0f}$, ${c[v]['hi']:+.0f}$]"
                 for v in ("rating", "tone", "fixable", "n_major", "words")]
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\midrule", r"\multicolumn{6}{l}{Rating $-$ human reviewers: " + ", ".join(
        f"{w} ${g['bias'][w]['diff']:+.2f}$" for w in ("author", "colleague", "reviewer", "venue")).replace("venue", "role-free")
        + r"; accept " + ", ".join(f"{g['means'][w]['accept']*100:.0f}\\%" for w in ("author", "colleague", "reviewer", "venue")) + r"} \\"]
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("primary", primary), ("secondary", secondary), ("secondary_rows", secondary_long), ("accept", accept), ("border", border),
                     ("robust", robust), ("discrim", discrim), ("decomp_secondary", decomp_secondary), ("decomp_secondary_rows", decomp_secondary_long),
                     ("manip", manip), ("rolefree", rolefree), ("main_rating", main_rating), ("judge", judge), ("retest", retest),
                     ("main_table", main_table), ("same_batch", same_batch), ("judges", judges), ("tone", tone), ("ladder", ladder), ("recognition", recognition), ("calib_own", calib_own), ("grok", grok)]:
        (OUT / f"{name}.tex").write_text(fn() + "\n")
    print("ok")
