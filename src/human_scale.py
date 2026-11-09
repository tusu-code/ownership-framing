"""What the author-reviewer shift means for the people who act on the verdict (main study, Claude, review form).

  python src/human_scale.py  -> paper/numbers_human.tex (keys hs/...), extend/out/human_scale.json

1. Verdicts against ICLR's decision: on papers ICLR rejected, the share Claude recommends for acceptance under the
   author and the reviewer ask; on papers ICLR accepted, the share it recommends for rejection.
2. Human yardstick: the change in a paper's mean human rating when one of its reviewers raises their score by one
   step (2 points on the 0-10 ICLR 2026 scale), 2 / n_reviews, averaged over the sample.
3. ICLR's own acceptance curve: a logistic fit of decision on the mean human rating over all ICLR 2026 papers with
   a known decision (data/raw/papers_iclr2026.csv); the average change in predicted acceptance when every
   borderline paper's mean rating (4 to under 6) is raised by the author-reviewer shift (Claude review form,
   main study, read from paper/numbers.tex), also at the ends of its 95% interval.
Exploratory, not preregistered. Paired bootstrap over papers (sample) or over all papers (ICLR curve), 2,000 draws.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "paper" / "numbers_human.tex"
JS = ROOT / "extend" / "out" / "human_scale.json"
B = 2000
rng = np.random.default_rng(20261108)


def pct(x):
    return f"{x * 100:.0f}\\%"


def ci(d, lo, hi, nd=0, scale=100):
    f = f"{{:.{nd}f}}"
    s = "+" if d >= 0 else "-"
    return f"${s}{f.format(abs(d) * scale)}$ [${f.format(lo * scale)}$, ${f.format(hi * scale)}$]"


def sample():
    c = pd.read_csv(ROOT / "full_study" / "out" / "cells.csv")
    d = c[(c.model == "claude") & (c.fmt == "struct") & c.who.isin(["author", "reviewer"])]
    w = d.pivot_table(index="paper_id", columns="who", values="accept")
    w = w.join(d.groupby("paper_id")[["human_accept"]].first()).dropna()
    m = pd.read_csv(RAW / "meta_396.csv").set_index("paper_id")
    w = w.join(m[["n_reviews"]])
    return w


def verdicts(w, V, R):
    for key, h in (("rej", 0), ("acc", 1)):
        s = w[w.human_accept == h]
        a, r = s.author.to_numpy(), s.reviewer.to_numpy()
        if h == 1:                         # share recommended for rejection
            a, r = 1 - a, 1 - r
        diff = a.mean() - r.mean()
        bs = [(lambda i: a[i].mean() - r[i].mean())(rng.integers(0, len(a), len(a))) for _ in range(B)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        V[f"hs/{key}/n"] = str(len(s))
        V[f"hs/{key}/A"], V[f"hs/{key}/R"] = pct(a.mean()), pct(r.mean())
        V[f"hs/{key}/rdci"] = ci(diff, lo, hi)
        V[f"hs/{key}/rdu"] = f"{abs(diff) * 100:.0f}"
        V[f"hs/{key}/ao"] = str(int(((s.author == 1) & (s.reviewer == 0)).sum()))
        V[f"hs/{key}/ro"] = str(int(((s.author == 0) & (s.reviewer == 1)).sum()))
        R[key] = dict(n=len(s), author=a.mean(), reviewer=r.mean(), diff=diff, lo=lo, hi=hi)
    for k in ("author", "reviewer"):
        V[f"hs/agree/{k}"] = pct((w[k] == w.human_accept).mean())


def yardstick(w, shift, V, R):
    step = (2 / w.n_reviews).mean()
    V["hs/step"] = f"${step:.2f}$"
    V["hs/step/ratio"] = pct(shift / step)
    V["hs/nrev"] = f"{w.n_reviews.median():.0f}"
    one = (shift * w.n_reviews).mean()     # points one reviewer would have to add to move the mean by `shift`
    V["hs/onerev"] = f"${one:.1f}$"
    R["one_reviewer_points"] = one
    R["step"] = dict(step=step, ratio=shift / step, median_reviews=float(w.n_reviews.median()))


def iclr_curve(shift, V, R, shift_lo, shift_hi):
    p = pd.read_csv(RAW / "papers_iclr2026.csv")
    p = p[p.decision != "Unknown"].copy()
    p["acc"] = p.decision.str.contains("Accept").astype(int)
    x, y = p["mean"].to_numpy(), p.acc.to_numpy()
    border = (x >= 4) & (x < 6)

    def effect(idx):
        f = sm.Logit(y[idx], sm.add_constant(x[idx])).fit(disp=0).params
        xb = x[idx][border[idx]]
        pr = lambda v: 1 / (1 + np.exp(-(f[0] + f[1] * v)))
        return (pr(xb + shift) - pr(xb)).mean(), pr(xb).mean()

    est, base = effect(np.arange(len(x)))
    bs = [effect(rng.integers(0, len(x), len(x)))[0] for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    V["hs/iclr/n"] = f"{len(p):,}".replace(",", "{,}")
    V["hs/iclr/border/n"] = f"{int(border.sum()):,}".replace(",", "{,}")
    V["hs/iclr/border/base"] = pct(base)
    V["hs/iclr/border/ppci"] = ci(est, lo, hi)
    V["hs/iclr/border/pp"] = f"{est * 100:.0f}"
    f = sm.Logit(y, sm.add_constant(x)).fit(disp=0).params
    pr = lambda v: 1 / (1 + np.exp(-(f[0] + f[1] * v)))
    lo_s, hi_s = [(pr(x[border] + s) - pr(x[border])).mean() for s in (shift_lo, shift_hi)]
    V["hs/iclr/border/range"] = f"{lo_s * 100:.0f} to {hi_s * 100:.0f}"
    R["iclr_shift_interval"] = dict(shift_lo=shift_lo, shift_hi=shift_hi, pp_lo=lo_s, pp_hi=hi_s)
    R["iclr"] = dict(n=len(p), border=int(border.sum()), base=base, pp=est, lo=lo, hi=hi)


def main():
    tex = (ROOT / "paper" / "numbers.tex").read_text()
    key = "num@models/claude/lmm.rating/struct/u}{$"
    shift = float(tex.split(key)[1].split("$")[0])
    get = lambda k: float(tex.split(f"num@models/claude/lmm.rating/struct/{k}}}{{$")[1].split("$")[0])
    shift_lo, shift_hi = get("lo"), get("hi")
    V, R = {}, {"shift": shift}
    w = sample()
    verdicts(w, V, R)
    yardstick(w, shift, V, R)
    iclr_curve(shift, V, R, shift_lo, shift_hi)
    lines = ["% Generated by src/human_scale.py. Do not edit.", r"\makeatletter"]
    lines += [f"\\@namedef{{num@{k}}}{{{v}}}" for k, v in sorted(V.items())]
    lines += [r"\makeatother"]
    OUT.write_text("\n".join(lines) + "\n")
    JS.write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    main()
