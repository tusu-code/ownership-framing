"""Figures for the NHB version: what each side is told, and the two preregistered requests.

  python src/fig_people.py  -> paper/figs/fig_consequences.pdf/.png, paper/figs/fig_requests.pdf/.png

fig_consequences
  a, Claude's review form, main study: share of ICLR-rejected papers recommended for acceptance and share of
     ICLR-accepted papers recommended for rejection, under the author and the reviewer ask (Wilson 95% intervals).
  b, ICLR 2026 acceptance by mean reviewer rating (all submissions with a public decision; binned shares and a
     logistic fit), with the fit shifted by Claude's author-reviewer difference; borderline band shaded.
  c, WildChat-1M requests for a judgement of a scholarly manuscript, by the relation the user stated: majority of
     three classifiers, and the requests where all three agreed.
fig_requests
  Author - reviewer gap under the plain ask and under each tested request, from the frozen registered analyses
  (mitig/results_156/mitig_stats_sample.json, detach/results_seeta/detach_stats_sample.json): plain gap with its
  95% bootstrap interval, remaining gap with the 90% t interval used for the equivalence test, and the remaining
  gap under each wording; equivalence bounds +-0.15 shaded.
Uses the style of src/paper_figs.py. Numbers in the text come from src/human_scale.py, wildchat/src/summarize.py,
src/mitig_numbers.py and src/detach_numbers.py; this script only draws.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figs as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "data" / "raw"
plt = F.plt
A_COL, R_COL = F.G2[0], F.G2[1]          # author, reviewer: ends of the teal-to-khaki ramp
SHIFT_COL = "#C8553D"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def frame(ax, ylabel=None):
    for s in ("left", "top", "right", "bottom"):
        ax.spines[s].set_visible(True)
    ax.tick_params(axis="y", width=0.6, length=2.5)
    ax.grid(axis="y", color=F.GRID, lw=0.5, ls=(0, (3, 2)), zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def panel_verdicts(ax):
    c = pd.read_csv(ROOT / "full_study" / "out" / "cells.csv")
    d = c[(c.model == "claude") & (c.fmt == "struct") & c.who.isin(["author", "reviewer"])]
    w = d.pivot_table(index="paper_id", columns="who", values="accept")
    w = w.join(d.groupby("paper_id")[["human_accept"]].first()).dropna()
    groups = [("Rejected by\nICLR, advised\nto accept", w[w.human_accept == 0], 1),
              ("Accepted by\nICLR, advised\nto reject", w[w.human_accept == 1], 0)]
    frame(ax, "Verdict against ICLR's decision (%)")
    for i, (lab, s, target) in enumerate(groups):
        tops = []
        for j, (who, col) in enumerate((("author", A_COL), ("reviewer", R_COL))):
            k = int((s[who] == target).sum())
            p, lo, hi = wilson(k, len(s))
            x = i + (j - 0.5) * 0.36
            ax.bar(x, p * 100, 0.33, color=col, edgecolor=F.INK, lw=0.5, zorder=3,
                   label=("Author asks" if who == "author" else "Reviewer asks") if i == 0 else None)
            ax.errorbar(x, p * 100, yerr=[[(p - lo) * 100], [(hi - p) * 100]], fmt="none", ecolor=F.INK,
                        elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=4)
            ax.text(x, hi * 100 + 1.5, f"{p * 100:.0f}", ha="center", va="bottom", fontsize=F.FS_SMALL)
            tops.append(hi * 100)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{g[0]}\n(n = {len(g[1])})" for g in groups], fontsize=F.FS_TICK, linespacing=1.15)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 72)
    ax.legend(loc="upper right", handlelength=1.0, handletextpad=0.4, borderaxespad=0.2, fontsize=F.FS_TICK)


def panel_curve(ax, shift):
    p = pd.read_csv(RAW / "papers_iclr2026.csv")
    p = p[p.decision != "Unknown"].copy()
    p["acc"] = p.decision.str.contains("Accept").astype(int)
    f = sm.Logit(p.acc, sm.add_constant(p["mean"])).fit(disp=0).params.to_numpy()
    pr = lambda v: 1 / (1 + np.exp(-(f[0] + f[1] * v)))  # noqa: E731
    frame(ax, "ICLR acceptance (%)")
    ax.axvspan(4, 6, color=F.BAND, zorder=0, lw=0)
    ax.text(5, 1.5, "borderline", ha="center", va="bottom", fontsize=F.FS_SMALL, color=F.MUTED)
    b = p.assign(bin=(p["mean"] * 2).round() / 2).groupby("bin").acc.agg(["mean", "size"])
    b = b[(b.index >= 1.5) & (b.index <= 8) & (b["size"] >= 30)]
    ax.scatter(b.index, b["mean"] * 100, s=np.sqrt(b["size"]) * 1.1, color=F.MUTED, alpha=0.55, lw=0, zorder=3,
               label="ICLR 2026 submissions")
    xs = np.linspace(1.5, 8, 300)
    ax.plot(xs, pr(xs) * 100, color=F.INK, lw=1.1, zorder=4, label="Logistic fit")
    ax.plot(xs, pr(xs + shift) * 100, color=SHIFT_COL, lw=1.1, ls=(0, (3, 1.6)), zorder=4,
            label=f"Fit, rating +{shift:.2f}")
    x0 = 4.75
    ax.annotate("", xy=(x0, pr(x0 + shift) * 100), xytext=(x0, pr(x0) * 100),
                arrowprops=dict(arrowstyle="-|>", color=SHIFT_COL, lw=0.8, mutation_scale=6), zorder=5)
    xb = p.loc[(p["mean"] >= 4) & (p["mean"] < 6), "mean"].to_numpy()
    gain = (pr(xb + shift) - pr(xb)).mean() * 100
    ax.text(5.45, 30, f"+{gain:.0f} pp for\nborderline\npapers", ha="left",
            va="center", fontsize=F.FS_SMALL, color=SHIFT_COL, linespacing=1.15)
    ax.set_xlim(1.5, 8)
    ax.set_ylim(0, 108)
    ax.set_xlabel("Mean reviewer rating")
    ax.legend(loc="upper left", handlelength=1.4, handletextpad=0.4, borderaxespad=0.3, fontsize=F.FS_SMALL)


def panel_wildchat(ax):
    S = json.loads((ROOT / "wildchat" / "out" / "summary.json").read_text())
    cats = [("own", "Own paper", F.G5[0]), ("other", "Someone else's", F.G5[2]), ("reviewer", "Reviewing it", F.G5[1]),
            ("none", "None stated", "#D9DCE1"), ("unresolved", "No majority", "#F0F1F3")]
    rows = [("All three\nagree", {k: S["unanimous"].get(k, 0) for k, _, _ in cats}),
            ("Majority of\nthree", {k: S[k]["n"] for k, _, _ in cats})]
    ys = [0, 0.82]
    for i, (lab, cnt) in enumerate(rows):
        n = sum(cnt.values())
        left, y = 0, ys[i]
        for k, name, col in cats:
            v = cnt[k] / n * 100
            if v == 0:
                continue
            ax.barh(y, v, left=left, height=0.6, color=col, edgecolor=F.INK, lw=0.5, zorder=3,
                    label=name if i == 1 else None)
            if v >= 6:
                ax.text(left + v / 2, y, str(cnt[k]), ha="center", va="center", fontsize=F.FS_SMALL)
            left += v
        ax.text(101.5, y, f"n = {n}", ha="left", va="center", fontsize=F.FS_SMALL, color=F.MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=F.FS_TICK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.42, 1.98)
    ax.set_xlabel("Share of requests (%)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=0.9, handletextpad=0.35,
              columnspacing=0.8, borderaxespad=0, fontsize=F.FS_SMALL)


def fig_consequences():
    tex = (ROOT / "paper" / "numbers.tex").read_text()
    shift = float(tex.split("num@models/claude/lmm.rating/struct/u}{$")[1].split("$")[0])
    fig, axes = plt.subplots(1, 3, figsize=(F.FIG_W * 1.060, 2.55), gridspec_kw={"width_ratios": [1.05, 1.05, 1.0], "wspace": 0.42})
    panel_verdicts(axes[0])
    panel_curve(axes[1], shift)
    panel_wildchat(axes[2])
    F.header(axes[0], "a", "What each side is told", y=1.06)
    F.header(axes[1], "b", "Illustration on ICLR's own curve", y=1.06)
    F.header(axes[2], "c", "Relations users state", y=1.06)
    F.save(fig, "fig_consequences")


def fig_requests():
    mt = json.loads((ROOT / "mitig" / "results_156" / "mitig_stats_sample.json").read_text())
    dt = json.loads((ROOT / "detach" / "results_seeta" / "detach_stats_sample.json").read_text())
    PLAIN, REQ, WORD = F.G2[0], F.G2[1], "#5B5E3A"
    WMARK = {"W1": "o", "W2": "s", "W3": "^"}
    groups = [("Claude\nreview form", mt["claude"]), ("Gemini\nfree text", mt["gemini"]), ("Claude\nreview form", dt)]
    fig, ax = plt.subplots(figsize=(F.FIG_W * 0.74, 2.05))
    ax.axhspan(-0.15, 0.15, color=F.BAND, zorder=0, lw=0)
    for s in ("left", "top", "right", "bottom"):
        ax.spines[s].set_visible(True)
    ax.axhline(0, color=F.INK, lw=0.6, zorder=2)
    ax.grid(axis="y", color=F.GRID, lw=0.5, ls=(0, (3, 2)), zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", width=0.6, length=2.5)
    bw, xs = 0.3, [0, 1, 2.35]
    for i, (lab, st) in enumerate(groups):
        P, S = st["primary"], st["secondary"]
        for k, (est, lo, hi, col, name) in enumerate(((P["H1"]["est"], P["H1"]["lo"], P["H1"]["hi"], PLAIN, "Without the request (95% CI)"),
                                                      (P["H3"]["est"], *P["H3"]["ci90"], REQ, "With the request (90% CI)"))):
            x = xs[i] + (k - 0.5) * (bw + 0.03)
            ax.bar(x, est, bw, color=col, edgecolor=F.INK, lw=0.5, zorder=3, label=name if i == 0 else None)
            ax.errorbar(x, est, yerr=[[est - lo], [hi - est]], fmt="none", ecolor=F.INK, elinewidth=0.6, capsize=1.6,
                        capthick=0.6, zorder=4)
            ax.text(x, hi + 0.015, f"{est:+.2f}".replace("-", "\u2212"), ha="center", va="bottom", fontsize=F.FS_SMALL)
            if k == 1:
                for d, (wd, mk) in zip((0.04, 0.085, 0.13), WMARK.items()):
                    ax.plot([x + bw / 2 + d], [S["S1_by_wording"][wd]["H3"]["est"]], mk, ms=3.0, mfc="white", mec=WORD,
                            mew=0.7, zorder=6, ls="none", label=f"Wording {wd[1]} alone" if i == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], linespacing=1.2)
    ax.set_xlim(-0.55, 2.9)
    ax.set_ylim(-0.18, 1.1)
    ax.axvline(1.675, color=F.GRID, lw=0.8, zorder=1)
    for x0, txt in ((0.5, "Request to review: gap remains"), (2.35, "As someone else's: closed")):
        ax.text(x0, 1.075, txt, ha="center", va="top", fontsize=F.FS_SMALL, fontweight="bold")
    ax.set_ylabel("Author $-$ reviewer rating")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    hd, lb = ax.get_legend_handles_labels()
    order = sorted(range(len(lb)), key=lambda i: (lb[i].startswith("Wording"), not lb[i].startswith("Without"), lb[i]))
    hd, lb = [hd[i] for i in order], [lb[i] for i in order]
    hd.append(Patch(color=F.BAND, lw=0))
    lb.append("Equivalence bounds \u00b10.15")
    ax.legend(hd, lb, loc="center left", bbox_to_anchor=(1.02, 0.5), handletextpad=0.4, borderaxespad=0, frameon=False)
    F.save(fig, "fig_requests")


if __name__ == "__main__":
    fig_consequences()
    fig_requests()
    print("ok")
