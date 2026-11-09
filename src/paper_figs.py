"""Main-text result figures for the NMI version, drawn from full_study/out/stats.json.

  python src/make_icons.py   # once: icons -> paper/icons/png
  python src/paper_figs.py   -> paper/figs/fig_effects.pdf, fig_mechanism.pdf, fig_decisions.pdf

fig_effects:   a, author - reviewer rating per model and format; b, distance from human reviewers by ask.
fig_mechanism: author - reviewer differences in tone, fixable framing and major concerns (open ask).
fig_decisions: a, share recommending accept under each ask; b, author - reviewer rating on borderline
               vs other papers.
Every figure uses one colour per ask (ASK, shared with the design figure) and the model logos from
src/make_icons.py next to the model names.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "full_study" / "out" / "stats.json").read_text())
OUT = ROOT / "paper" / "figs"
ICON = ROOT / "paper" / "icons" / "png"
MODELS = ["claude", "gemini", "gpt", "glm", "deepseek"]
LATE = ["claude", "gemini", "gpt", "glm"]   # models whose later asks ran on a stable model version
LABEL = {"claude": "Claude Sonnet 4.6", "gpt": "GPT-6.1 Sol", "glm": "GLM-5.3", "deepseek": "DeepSeek V4 Pro",
         "gemini": "Gemini 4 Argon", "grok": "Grok 4.7"}
SHORT = {"claude": "Claude", "gpt": "GPT", "glm": "GLM", "deepseek": "DeepSeek", "gemini": "Gemini", "grok": "Grok"}

INK, MUTED, GRID, BAND = "#2E3440", "#6B7280", "#E5E7EB", "#F5F6F8"
ASK = {"author": "#C8553D", "colleague": "#E09F3E", "venue": "#7A7A7A", "reviewer": "#2E6F9E"}
ASK_LABEL = {"author": "I wrote it", "colleague": "A colleague wrote it", "venue": "No role stated",
             "reviewer": "I am reviewing it"}
FMT = {"open": ("#264653", "o", "Open ask"), "struct": ("#2A9D8F", "D", "Review form")}
BORDER, OTHER = "#6A4C93", "#A3A9B3"

# One type scale for every figure, in points at print size (figures are drawn at the text width, 6.45 in, and
# included unscaled): axis labels 7, tick labels and legends 6.5, values and notes 6, panel titles 7.5, letters 8.5.
FS_BASE, FS_TICK, FS_SMALL, FS_TITLE, FS_LETTER = 7.0, 6.5, 6.0, 7.5, 8.5
FIG_W = 6.45
plt.rcParams.update({
    "font.family": ["Arimo", "Liberation Sans", "DejaVu Sans"], "font.size": FS_BASE, "axes.labelsize": FS_BASE,
    "axes.titlesize": FS_TITLE, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TICK,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0, "xtick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True,
})
_LOGOS = {m: plt.imread(ICON / f"logo_{m}.png") for m in MODELS}


def logo_rows(ax, models, size=9.0):
    """Model logo between the y tick label and the axis, one per row."""
    ax.tick_params(axis="y", pad=size + 4)
    for i, m in enumerate(models):
        ab = AnnotationBbox(OffsetImage(_LOGOS[m], zoom=size / _LOGOS[m].shape[0]), (0, i),
                            xycoords=("axes fraction", "data"), box_alignment=(1.15, 0.5), frameon=False,
                            annotation_clip=False, pad=0)
        ax.add_artist(ab)


def logo_cols(ax, models, y=-0.11, size=9.5):
    """Model logo under each x tick label group."""
    for i, m in enumerate(models):
        ab = AnnotationBbox(OffsetImage(_LOGOS[m], zoom=size / _LOGOS[m].shape[0]), (i, y),
                            xycoords=("data", "axes fraction"), box_alignment=(0.5, 0.5), frameon=False,
                            annotation_clip=False, pad=0)
        ax.add_artist(ab)


def rows_frame(ax, n, labels):
    """Shared look for forest-style panels: banded rows, zero line, light vertical grid."""
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_ylim(n - 0.5, -0.5)


def point(ax, e, y, color, marker="o", key=("est", "lo", "hi"), scale=1.0, size=4.2, label=None):
    k, lo, hi = key
    x, a, b = e[k] * scale, e[lo] * scale, e[hi] * scale
    ax.plot([a, b], [y, y], color=color, lw=1.3, solid_capstyle="round", zorder=3)
    ax.plot([x], [y], marker=marker, ms=size, color=color, mec="white", mew=0.6, zorder=4, ls="none", label=label)


def header(ax, letter, title, y=1.13):
    ax.text(-0.02, y, letter, transform=ax.transAxes, fontsize=FS_LETTER, fontweight="bold", ha="right", va="bottom")
    ax.text(0.0, y, title, transform=ax.transAxes, fontsize=FS_TITLE, fontweight="bold", ha="left", va="bottom")


def save(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None:
            lb, ab = lg.get_window_extent(r), ax.get_window_extent(r)
            inside_y = lb.y0 >= ab.y1 - 1   # legends placed above the axes are allowed
            if lb.x0 < ab.x0 - 1 or lb.x1 > ab.x1 + 1 or (not inside_y and (lb.y0 < ab.y0 - 1 or lb.y1 > ab.y1 + 1)):
                print(f"WARNING {name}: legend outside its panel")
        for tx in ax.texts:
            tb = tx.get_window_extent(r)
            for other in fig.axes:
                if other is not ax and tx.get_text().strip() and tb.overlaps(other.get_window_extent(r)):
                    print(f"WARNING {name}: text '{tx.get_text()[:40]}' overlaps another panel")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


FIXED = "#8A5A9E"
HUMAN = "#1B998B"
JUDGE2 = "#9AA5B1"


SELF = "#264653"     # the model's own score (review form or second turn)
JUDGE_M, JUDGE_F = "#9AA5B1", "#C9CED6"
EFF_MODELS = ["claude", "gemini", "deepseek", "glm", "gpt"]


def sskey(m, fmt="open"):
    return "deepseek_open_r3" if (m == "deepseek" and fmt == "open") else f"{m}_{fmt}"


# Bar panels: one group per model (logo under the axis), one bar per series, 95% CI whiskers, value on the bar.
# Colour ramp taken from the reference figure: teal -> sage -> pale khaki. Bars take evenly spaced colours from it.
from matplotlib.colors import LinearSegmentedColormap, to_hex  # noqa: E402
GRAD = LinearSegmentedColormap.from_list("ramp", ["#5F9EAF", "#8DB3AF", "#B4C3A7", "#D6D2A3", "#EBD9A0"])


def grad(n):
    return [to_hex(GRAD(x)) for x in np.linspace(0, 1, n)]


G2, G3, G5 = grad(2), grad(3), grad(5)
MODEL_C = dict(zip(MODELS, G5))
WRAP = {"claude": "Claude\nSonnet 4.6", "gemini": "Gemini\n4 Argon", "deepseek": "DeepSeek\nV4 Pro", "glm": "GLM-5.3",
        "gpt": "GPT-6.1\nSol"}
FORM_C, FREE_C = G2       # review form, free text with the own score in a second turn
FIXED_FORM, FIXED_FREE = G2


def est3(e):
    return e.get("diff", e.get("est")), e["lo"], e["hi"]


def bars(ax, groups, series, width=0.78, fmt="{:+.2f}", fs=None):
    """groups: x labels; series: (label, colour, hatch, [estimate dict or None per group])."""
    fs = fs or FS_SMALL
    n = len(series)
    w = width / n
    for j, (lab, col, hatch, vals) in enumerate(series):
        for i, e in enumerate(vals):
            if not e:
                continue
            d, lo, hi = est3(e)
            x = i + (j - (n - 1) / 2) * w
            ax.bar(x, d, w * 0.9, color=col, hatch=hatch, edgecolor=INK, linewidth=0.5, zorder=3,
                   label=lab if i == min(k for k, v in enumerate(vals) if v) else None)
            ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.6,
                        capthick=0.6, zorder=4)
            top = hi if d >= 0 else lo
            ax.text(x, top + (0.012 if d >= 0 else -0.012) * np.ptp(ax.get_ylim()), fmt.format(d).replace("-", "\u2212"), ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=fs, color=INK, zorder=5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_xlim(-0.6, len(groups) - 0.4)


def bar_frame(ax, ylabel, ylim):
    ax.set_ylim(*ylim)
    for s in ("left", "top", "right", "bottom"):
        ax.spines[s].set_visible(True)
    ax.tick_params(axis="y", width=0.6, length=2.5)
    ax.tick_params(axis="x", length=0)
    ax.axhline(0, color=INK, lw=0.6, zorder=2)
    ax.grid(axis="y", color=GRID, lw=0.5, ls=(0, (3, 2)), zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)


def logo_under(ax, models, size=10.0, pad=26):
    """Logo just under the x axis; tick labels sit below it."""
    ax.tick_params(axis="x", pad=size + 4)
    for i, m in enumerate(models):
        ab = AnnotationBbox(OffsetImage(_LOGOS[m], zoom=size / _LOGOS[m].shape[0]), (i, 0),
                            xycoords=("data", "axes fraction"), box_alignment=(0.5, 1.15), frameon=False,
                            annotation_clip=False, pad=0)
        ax.add_artist(ab)


def fig_effects():
    """a, author - reviewer on the models' own scores; b, each ask minus the role-free ask, own scores."""
    ss = R["revision"]["selfscore"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 1.112, 2.9), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.3})
    ax = axes[0]
    bar_frame(ax, "Author $-$ reviewer rating (0–10)", (-0.4, 1.4))
    bars(ax, [WRAP[m] for m in EFF_MODELS], [
        ("Review form", FORM_C, None, [R["models"][m]["lmm_rating"]["struct"] for m in EFF_MODELS]),
        ("Free text, own score in a second turn", FREE_C, None,
         [ss.get(sskey(m), {}).get("self", {}).get("author - reviewer") for m in EFF_MODELS])])
    logo_under(ax, EFF_MODELS)
    ax.legend(loc="upper center", ncol=2, handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
    header(ax, "a", "Claiming the paper raises the model's own rating", y=1.03)

    ax = axes[1]
    ms = ["claude", "gemini", "glm"]
    bar_frame(ax, "Rating minus the role-free ask", (-2.6, 0.3))

    def rel(m, w):
        return R["baseline"]["claude"][w]["lmm_rating"]["struct"] if m == "claude" else ss[sskey(m)]["self"].get(f"{w} - venue")
    bars(ax, [WRAP["claude"] + "\n(review form)", WRAP["gemini"] + "\n(second turn)", WRAP["glm"] + "\n(second turn)"],
         [(ASK_LABEL[w], G3[k], None, [rel(m, w) for m in ms]) for k, w in enumerate(("author", "colleague", "reviewer"))])
    logo_under(ax, ms)
    ax.legend(loc="lower right", ncol=1, handlelength=1.2, handletextpad=0.4)
    header(ax, "b", "Any stated relation lowers it; ownership least", y=1.03)
    save(fig, "fig_effects")


def fig_judges():
    """a, author - reviewer on the same answers: own score vs two judges; b, warm minus cold rewrite of one assessment."""
    ss = R["revision"]["selfscore"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 1.124, 2.6), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.28})
    ax = axes[0]
    ms = ["claude", "gemini", "deepseek", "glm"]
    bar_frame(ax, "Author $-$ reviewer, same answers", (-0.4, 2.15))
    bars(ax, [WRAP[m] for m in ms], [
        (lab, c, None, [ss[sskey(m)][col]["author - reviewer"] for m in ms])
        for col, c, lab in (("self", G3[0], "Own score"), ("main", G3[1], "Read by MiniMax"),
                            ("flash", G3[2], "Read by DeepSeek Flash"))])
    logo_under(ax, ms)
    ax.legend(loc="upper left", ncol=3, handlelength=1.0, handletextpad=0.35, columnspacing=0.6, fontsize=FS_TICK,
              borderaxespad=0.3)
    header(ax, "a", "The text conveys a larger gap than the verdict", y=1.03)

    ax = axes[1]
    t = R["revision"]["tone"]
    bar_frame(ax, "Warm $-$ cold rewrite of one assessment", (-0.12, 0.55))
    bars(ax, ["DeepSeek V4.1\nFlash", "MiniMax M3.1\nFlash"], [
        ("Tone read by the judge (1–7)", G2[0], None, [t[j]["tone"] for j in ("dsflash", "minimax")]),
        ("Rating implied (0–10)", G2[1], None, [t[j]["rating"] for j in ("dsflash", "minimax")])], width=0.6)
    ax.tick_params(axis="x", pad=3)
    ax.legend(loc="upper left", ncol=1, handlelength=1.2, handletextpad=0.4)
    header(ax, "b", "A warmer register alone moves one judge", y=1.03)
    save(fig, "fig_judges")


def fig_instruction():
    """a, Gemini, own score, from the natural ask to the fixed instruction; b, fixed instruction in both formats."""
    ss = R["revision"]["selfscore"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.7), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.28})
    ax = axes[0]
    steps = [("open", "Natural\nask"), ("ablorder", "Ask after\nthe paper"), ("ablrole", "“Review it\nas a\nreviewer\nwould”"),
             ("ablmerits", "+ “Judge it\non its\nmerits”"), ("fixedopen", "Fixed\ninstruction")]
    bar_frame(ax, "Author $-$ reviewer, Gemini's own score", (-0.4, 1.35))
    lad = R["revision"]["ladder"]
    for j, (k, _) in enumerate(steps):      # one series per bar so the colour can change along the ladder
        vals = [lad[k] if i == j else None for i in range(len(steps))]
        bars(ax, [lab for _, lab in steps], [(None, G5[j], None, vals)], width=0.6)
    ax.tick_params(axis="x", pad=3, labelsize=FS_SMALL)
    header(ax, "a", "The request to review is the step that acts", y=1.03)

    ax = axes[1]
    ms = ["claude", "gemini", "deepseek", "glm", "gpt"]
    bar_frame(ax, "Author $-$ reviewer, own score", (-0.65, 0.65))
    bars(ax, [WRAP[m] for m in ms], [
        ("Fixed instruction, review form", FIXED_FORM, None, [R["fixed"][m] for m in ms]),
        ("Fixed instruction, free text", FIXED_FREE, None,
         [ss.get(f"{m}_fixedopen", {}).get("self", {}).get("author - reviewer") for m in ms])])
    logo_under(ax, ms)
    ax.legend(loc="upper right", ncol=1, handlelength=1.2, handletextpad=0.4)
    header(ax, "b", "Under the fixed instruction", y=1.03)
    save(fig, "fig_instruction")


def fig_ownership():
    """a, same-batch contrasts that isolate ownership; b, the main and a second judge on the same answers."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.75), gridspec_kw={"width_ratios": [1, 1], "wspace": 1.05})

    ax = axes[0]
    keys = [("claude_open", "claude", "Claude Sonnet 4.6"), ("claude_struct", "claude", "Claude Sonnet 4.6\n(review form)"),
            ("glm_open", "glm", "GLM-5.3")]
    rows_frame(ax, len(keys), [lab for _, _, lab in keys])
    style = (("author - colleague", ASK["colleague"], -0.2, "Own $-$ colleague's, before submission"),
             ("author_review - reviewer", ASK["reviewer"], 0.0, "Own $-$ reviewed, under review"),
             ("author - reviewer", INK, 0.2, "Author $-$ reviewer"))
    for i, (k, m, _) in enumerate(keys):
        for c, col, dy, lab in style:
            point(ax, R["same_batch"][k]["contrasts"][c], i + dy, col, key=("diff", "lo", "hi"), label=lab if i == 0 else None)
    logo_rows(ax, [m for _, m, _ in keys])
    ax.set_xlabel("Rating difference, same batch (200 papers)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, handletextpad=0.2, borderaxespad=0.2)
    header(ax, "a", "Ownership holds at both stages", y=1.32)

    ax = axes[1]
    rows = [("claude", "author - reviewer"), ("claude", "author - colleague"), ("deepseek", "author - reviewer"),
            ("glm", "author - reviewer"), ("glm", "author - colleague")]
    lab = {"author - reviewer": "author $-$ reviewer", "author - colleague": "author $-$ colleague"}
    rows_frame(ax, len(rows), [f"{SHORT[m]}: {lab[c]}" for m, c in rows])
    for i, (m, c) in enumerate(rows):
        v = R["judge2"]["models"][m]
        point(ax, v["flash"][c], i - 0.14, ASK["author"], key=("diff", "lo", "hi"), label="Main judge" if i == 0 else None)
        point(ax, v["haiku"][c], i + 0.14, JUDGE2, "D", key=("diff", "lo", "hi"), label="Second judge (other family)" if i == 0 else None)
    logo_rows(ax, [m for m, _ in rows])
    ax.set_xlabel("Rating difference, open ask (170 papers)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, handletextpad=0.2, borderaxespad=0.2)
    header(ax, "b", "A second judge from another family", y=1.32)
    save(fig, "fig_ownership")


def fig_mechanism():
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 1.127, 2.35), gridspec_kw={"wspace": 0.42})
    pp = lambda e: {k: e[k] * 100 for k in ("diff", "lo", "hi")}   # noqa: E731
    panels = [("Warmer tone", "Author $-$ reviewer (1–7 scale)", lambda m: R["models"][m]["paired"]["open"]["tone"], (0, 1.55), "{:+.2f}"),
              ("Problems framed as fixable", "Author $-$ reviewer (points)", lambda m: pp(R["models"][m]["paired"]["open"]["fixable"]), (0, 78), "{:+.0f}"),
              ("Major concerns per 1,000 words", "Author $-$ reviewer (count)", lambda m: R["concerns_per_kword"][m]["author_minus_reviewer"], (-0.95, 1.05), "{:+.2f}")]
    for k, (ax, (title, ylab, get, ylim, fmt)) in enumerate(zip(axes, panels)):
        bar_frame(ax, ylab, ylim)
        for j, m in enumerate(MODELS):      # one series per model: each bar in its model's colour
            bars(ax, [SHORT[x] if x != "deepseek" else "Deep-\nSeek" for x in MODELS], [(None, MODEL_C[m], None, [get(m) if i == j else None for i in range(len(MODELS))])],
                 width=0.62, fmt=fmt, fs=FS_SMALL)
        logo_under(ax, MODELS, size=9.0)
        ax.tick_params(axis="x", labelsize=FS_SMALL)     # horizontal labels; rotated ones crowded the logos
        header(ax, "abc"[k], title, y=1.03)
    save(fig, "fig_mechanism")


def fig_decisions():
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.15), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.55})

    ax = axes[0]
    groups = ["claude", "gemini", "glm", "deepseek"]
    bars = (("struct", "author", ASK["author"], None, "I wrote it"), ("struct", "colleague", ASK["colleague"], None, "A colleague wrote it"),
            ("struct", "reviewer", ASK["reviewer"], None, "I am reviewing it"),
            ("fixed", "author", ASK["author"], "////", "Fixed instruction, author"),
            ("fixed", "reviewer", ASK["reviewer"], "////", "Fixed instruction, reviewer"))
    w = 0.15
    for i, m in enumerate(groups):
        present = [b for b in bars if b[1] in R["calibration"][m].get(b[0], {})]
        for j, (fmt, who, col, hatch, lab) in enumerate(present):
            x = i + (j - (len(present) - 1) / 2) * w
            val = R["calibration"][m][fmt][who]["accept"]
            ax.bar(x, val, w * 0.92, color=col, hatch=hatch, edgecolor="white", linewidth=0.4, zorder=3,
                   alpha=0.55 if fmt == "fixed" else 1.0, label=lab if i == 0 else None)
            ax.text(x, val + 0.012, f"{val * 100:.0f}", ha="center", va="bottom", fontsize=FS_SMALL, color=INK)
    hum = R["calibration_human_accept"]
    ax.axhline(hum, color=HUMAN, lw=1.0, zorder=2)
    ax.text(len(groups) - 0.45, hum + 0.015, f"ICLR {hum * 100:.0f}%", ha="right", va="bottom", fontsize=FS_SMALL, color=HUMAN, style="italic")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([SHORT[m] for m in groups])
    ax.tick_params(axis="x", pad=15, length=0)
    logo_cols(ax, groups, y=-0.07)
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="y", width=0.6, length=2.5)
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0", "20", "40", "60", "80"])
    ax.set_ylabel("Recommended for acceptance (%)")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handletextpad=0.3, columnspacing=0.8,
              borderaxespad=0.2, handlelength=1.2, fontsize=FS_TICK)
    header(ax, "a", "Acceptance on the review form", y=1.2)

    ax = axes[1]
    rows = [(m, "open") for m in MODELS] + [("claude", "struct")]
    rows_frame(ax, len(rows), [LABEL[m] + ("\n(review form)" if f == "struct" else "") for m, f in rows])
    for i, (m, fmt) in enumerate(rows):
        st = R["models"][m]["strat"][fmt]
        point(ax, st["border"], i - 0.15, BORDER, key=("diff", "lo", "hi"), label="Borderline papers" if i == 0 else None)
        point(ax, st["other"], i + 0.15, OTHER, key=("diff", "lo", "hi"), label="Other papers" if i == 0 else None)
    logo_rows(ax, [m for m, _ in rows])
    ax.set_xlabel("Author $-$ reviewer rating")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
    header(ax, "b", "A similar offset across paper quality", y=1.2)
    save(fig, "fig_decisions")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_effects()
    fig_judges()
    fig_mechanism()
    fig_instruction()
    print("ok")
