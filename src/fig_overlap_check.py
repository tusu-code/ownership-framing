"""Report text that overlaps another panel's axes area or other text, for every figure in paper_figs.

  python src/fig_overlap_check.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import paper_figs as pf  # noqa: E402

found = []
_save = pf.save


def check(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axes = fig.axes
    texts = []
    for ax in axes:
        lo, hi = sorted(ax.get_xlim())
        xt = [t for t, v in zip(ax.get_xticklabels(), ax.get_xticks()) if lo <= v <= hi]
        items = list(ax.get_yticklabels()) + xt + list(ax.texts)
        items += [ax.xaxis.label, ax.yaxis.label]
        if ax.get_legend():
            items += list(ax.get_legend().get_texts())
        for t in items:
            if t.get_visible() and t.get_text().strip():
                texts.append((ax, t, t.get_window_extent(r)))
    for ax, t, bb in texts:
        for other in axes:
            if other is not ax and bb.overlaps(other.get_window_extent(r)):
                found.append(f"{name}: '{t.get_text()!r}' overlaps another panel")
    for i, (a1, t1, b1) in enumerate(texts):
        for a2, t2, b2 in texts[i + 1:]:
            if b1.overlaps(b2):
                found.append(f"{name}: '{t1.get_text()!r}' x '{t2.get_text()!r}'")
    _save(fig, name)


pf.save = check
pf.OUT.mkdir(parents=True, exist_ok=True)
for f in (pf.fig_effects, pf.fig_judges, pf.fig_mechanism, pf.fig_instruction):
    f()
print("\n".join(found) or "no overlaps")
