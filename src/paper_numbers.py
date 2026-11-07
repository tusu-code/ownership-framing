"""Every number quoted in the text of paper/main.tex, generated from full_study/out/stats.json.

  python src/paper_numbers.py   -> paper/numbers.tex

The text uses \\V{key}; a key missing here prints in red as ??key and is listed by paper/check_numbers.sh.
Two kinds of key:
  generic  every estimate in stats.json, addressed by its path (e.g. revision/selfscore/claude.open/self/author-reviewer)
           with suffix /ci, /d, /lo, /hi, /n or /p; every McNemar result with /aa, /ar, /ao, /ro, /p.
  alias    short names for the numbers the text uses most (ALIAS below).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "full_study" / "out" / "stats.json").read_text())
OUT = ROOT / "paper" / "numbers.tex"


def s(x, k=2):
    return f"${x:+.{k}f}$"


def u(x, k=2):
    return f"${x:.{k}f}$"


def pct(x):
    return f"{x * 100:.0f}\\%"


def pv(p):
    return "$p<0.001$" if p < 0.001 else (f"$p={p:.3f}$" if p < 0.01 else f"$p={p:.2f}$")


def key(path):
    return "/".join(path).replace(" - ", "-").replace(" ", "").replace("_", ".")


def est_fields(e):
    d = e.get("diff", e.get("est"))
    out = {"d": s(d), "u": u(abs(d)), "ci": f"{s(d)} [{u(e['lo'])}, {u(e['hi'])}]", "lo": u(e["lo"]), "hi": u(e["hi"]),
           "pp": f"${d * 100:+.0f}$", "ppci": f"${d * 100:+.0f}$ [${e['lo'] * 100:.0f}$, ${e['hi'] * 100:.0f}$]"}
    if "n" in e:
        out["n"] = f"{e['n']:,}"
    for pk in ("p", "p_wilcoxon"):
        if pk in e and e[pk] == e[pk]:
            out["p"] = pv(e[pk])
    return out


def walk(x, path, emit):
    if isinstance(x, dict):
        if ("diff" in x or "est" in x) and "lo" in x and "hi" in x and isinstance(x.get("lo"), float):
            for f, v in est_fields(x).items():
                emit(key(path + [f]), v)
        if "author_only" in x and "acc_author" in x:
            emit(key(path + ["aa"]), pct(x["acc_author"]))
            emit(key(path + ["ar"]), pct(x["acc_reviewer"]))
            emit(key(path + ["ao"]), str(x["author_only"]))
            emit(key(path + ["ro"]), str(x["reviewer_only"]))
            emit(key(path + ["p"]), pv(x["p"]))
            if "n" in x:
                emit(key(path + ["n"]), str(x["n"]))
        for k, v in x.items():
            walk(v, path + [str(k)], emit)


def aliases(emit):
    rv = R["revision"]
    ss = rv["selfscore"]
    # Holm-adjusted p for the six self-scored contrasts and the 15 primary tests.
    for t in R["self_tests"]:
        emit(key(["holm", "self", t["contrast"], t["outcome"]]), pv(t["p_adj"]))
    for t in R["primary_tests"]:
        emit(key(["holm", "primary", t["model"], t["contrast"]]), pv(t["p_adj"]))
    for t in R["decomp_tests"]:
        emit(key(["holm", "decomp", t["model"], t["outcome"], t["contrast"]]), pv(t["p_adj"]))
    for k, v in ss.items():
        emit(key(["self", k, "rho"]), u(v["spearman_self_main"]))
        emit(key(["self", k, "nans"]), f"{v['n_answers']:,}")
        for w, x in v["self_means"].items():
            emit(key(["self", k, "mean", w]), u(x))
    for k, v in rv["recognize"].items():
        emit(key(["rec", k, "hit"]), f"{v['hit'] * 100:.1f}\\%")
        emit(key(["rec", k, "nhit"]), str(v["n_hit"]))
        emit(key(["rec", k, "n"]), str(v["n"]))
        emit(key(["rec", k, "dstated"]), str(v["decision_stated"]))
        emit(key(["rec", k, "dcorrect"]), str(v["decision_correct"]))
    emit("rec/union", str(len(rv["named_union"])))
    rhos = [v["spearman_self_main"] for k, v in ss.items() if k.endswith("_open") or k.endswith("_open_r3")]
    emit("self/rho/min", u(min(rhos))); emit("self/rho/max", u(max(rhos)))
    rs = R.get("reasoning") or {}
    for w, v in rs.get("by_ask", {}).items():
        emit(f"reason/{w}/mention", pct(v["mention"])); emit(f"reason/{w}/adapt", pct(v["adapt"])); emit(f"reason/{w}/verdict", pct(v["verdict"]))
        emit(f"reason/{w}/n", str(v["n"]))
    for k, tag in (("self author-reviewer", "all"), ("self author-reviewer, author reasoning not planning a kinder verdict", "unplanned")):
        if k in rs:
            for f, v in est_fields(rs[k]).items():
                emit(f"reason/{tag}/{f}", v)
    for p_, v in R["judge2"]["pairs"].items():
        emit(key(["judges", p_, "rho"]), u(v["spearman"]))
        emit(key(["judges", p_, "n"]), f"{v['n']:,}")
    for m, v in R["manip_all"].items():
        for w, x in v.items():
            emit(key(["manip", m, w, "author"]), pct(x["addr_author"]))
            emit(key(["manip", m, w, "reviewer"]), pct(x["addr_reviewer"]))
    for m, v in R["calibration"].items():
        for fmt, ws in v.items():
            for w, x in ws.items():
                emit(key(["brier", m, fmt, w]), f"${x['brier']:.3f}$")
                emit(key(["accept", m, fmt, w]), pct(x["accept"]))
    aucs = [v[k] for v in R["discrimination"].values() for k in ("auc_author", "auc_reviewer")]
    emit("auc/min", u(min(aucs))); emit("auc/max", u(max(aucs)))
    dd = [v["auc_author"] - v["auc_reviewer"] for v in R["discrimination"].values()]
    emit("auc/dmin", s(min(dd))); emit("auc/dmax", s(max(dd)))
    emit("human/accept", pct(R["calibration_human_accept"]))
    import pandas as pd
    c = pd.read_csv(ROOT / "full_study" / "out" / "cells.csv")
    g = c[(c.model == "gpt") & (c.fmt == "struct")]
    emit("gpt/floor", pct(g.rating.isin([2, 4]).mean()))
    res = ROOT / "full_study" / "results"
    n_ans = 0
    for m in ("claude", "gemini", "gpt", "glm", "deepseek", "grok"):
        keys = set()
        for l in open(res / f"p2_review_{m}.jsonl"):
            r = json.loads(l)
            if r.get("ok") and str(r.get("relay", "")).startswith("https://api.sailapi"):
                keys.add((r["paper_id"], r["condition"], r["rep"]))
        n_ans += len(keys)
    emit("count/answers", f"{n_ans:,}")
    ss = {(r["model"], r["paper_id"], r["condition"], r["rep"]) for r in map(json.loads, open(res / "p4_selfscore.jsonl")) if r.get("ok")}
    emit("count/selfscore", f"{len(ss):,}")
    rc = {(r["model"], r["paper_id"], r["version"]) for r in map(json.loads, open(res / "p4_recognize.jsonl")) if r.get("ok")}
    emit("count/recognize", f"{len(rc):,}")
    mm = {(r["hash"], r["input"]) for r in map(json.loads, open(res / "p4_judge_minimax.jsonl")) if r.get("ok")}
    emit("count/minimax.neutral", f"{sum(1 for _, i in mm if i == 'neutral'):,}")
    nz = [r for r in map(json.loads, open(res / "p3_neutral.jsonl")) if r.get("ok")]
    emit("count/neutral", f"{len(nz):,}"); emit("count/neutral.clean", f"{sum(1 for r in nz if not r.get('cues_left')):,}")
    smp = pd.read_csv(ROOT / "full_study" / "data" / "sample_full.csv")
    emit("human/mean", u(smp.human_mean.mean())); emit("human/sd", u(smp.human_mean.std()))
    ia = R["models"]["claude"]["strat_accept"]["struct"]["interaction"]
    emit("claude/border/rd", f"{ia['rd_border'] * 100:.0f}"); emit("claude/other/rd", f"{ia['rd_other'] * 100:.0f}"); emit("claude/border/p", pv(ia["p"]))
    emit("claude/ownshare", pct(R["decomp"]["claude"]["ownership"]["lmm_rating"]["struct"]["est"] / R["models"]["claude"]["lmm_rating"]["struct"]["est"]))
    emit("titlesens/n", str(R["title_sensitivity"]["n_dropped"]))
    emit("residue/n", str(R["residue_sensitivity"]["n_dropped"]))
    for k, v in R["retest"].items():
        emit(key(["retest", k, "same"]), pct(v["same"]))


def main():
    vals = {}

    def emit(k, v):
        vals[k] = v
    walk(R, [], emit)
    aliases(emit)
    lines = ["% Generated by src/paper_numbers.py from full_study/out/stats.json. Do not edit.",
             r"\makeatletter",
             r"\newcommand{\V}[1]{\@ifundefined{num@#1}{\textcolor{red}{??#1}\typeout{NUMBER MISSING: #1}}{\csname num@#1\endcsname}}"]
    for k in sorted(vals):
        lines.append(f"\\@namedef{{num@{k}}}{{{vals[k]}}}")
    lines.append(r"\makeatother")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{len(vals)} numbers -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
