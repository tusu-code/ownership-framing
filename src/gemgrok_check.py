"""Preliminary look at the two models added on 2026-11-02 (Gemini 4 Argon, all asks; Grok 4.7, open asks on 100 papers).

  python src/gemgrok_check.py
Uses stats_full.build() with the model list widened; prints paired contrasts with bootstrap CIs and
Wilcoxon p, accept rates on the form, and distance from the human mean rating.
"""
import numpy as np
import pandas as pd
from scipy import stats

import stats_full as sf

sf.MODELS = ["gemini", "grok", "claude"]   # claude as a reference row
RNG = np.random.default_rng(11)


def ci(z):
    z = np.asarray(z, float)
    z = z[~np.isnan(z)]
    if len(z) < 5:
        return f"n={len(z)}"
    bs = z[RNG.integers(0, len(z), (4000, len(z)))].mean(1)
    nz = z[z != 0]
    p = stats.wilcoxon(nz).pvalue if len(nz) >= 5 else 1.0
    return f"{z.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}] p={p:.2g} n={len(z)}"


d, sample = sf.build()
grok_ids = set(pd.read_csv(sf.SRC / "data" / "retest100.csv").paper_id)
for m in ["gemini", "grok", "claude"]:
    for fmt in ["open", "struct", "fixed"]:
        x = d[(d.model == m) & (d.fmt == fmt)]
        if m == "grok" and fmt != "open":
            continue   # Grok ran only the open asks; its few form answers are from the 5-paper trial
        if m == "claude":
            x = x[x.paper_id.isin(grok_ids)] if fmt == "open" else x.iloc[0:0]
        if x.empty or "rating" not in x or x.rating.notna().sum() == 0:
            continue
        w = x.pivot_table(index="paper_id", columns="who", values="rating")
        hm = sample.human_mean.reindex(w.index)
        tag = " (Grok's 100 papers)" if m == "claude" else ""
        print(f"\n{m} {fmt}{tag}: papers {len(w)} | mean " + ", ".join(f"{k} {w[k].mean():.2f}" for k in w.columns)
              + f" | rating values {x.rating.value_counts().sort_index().to_dict()}")
        for a, b in [("author", "reviewer"), ("author", "colleague"), ("colleague", "reviewer"), ("author", "venue"), ("venue", "reviewer")]:
            if {a, b} <= set(w.columns):
                print(f"   {a:9s} - {b:9s} {ci(w[a] - w[b])}")
        print("   rating - human mean: " + ", ".join(f"{k} {(w[k] - hm).mean():+.2f}" for k in w.columns))
        if "reviewer" in w:
            print(f"   Spearman with human mean (reviewer ask): {stats.spearmanr(w['reviewer'], hm, nan_policy='omit').statistic:.2f}")
        if "accept" in x and x.accept.notna().any():
            wa = x.pivot_table(index="paper_id", columns="who", values="accept")
            print("   accept: " + ", ".join(f"{k} {wa[k].mean():.0%}" for k in wa.columns), end="")
            if {"author", "reviewer"} <= set(wa.columns):
                q = wa[["author", "reviewer"]].dropna()
                a_, b_ = int(((q.author == 1) & (q.reviewer == 0)).sum()), int(((q.author == 0) & (q.reviewer == 1)).sum())
                p = stats.binomtest(a_, a_ + b_).pvalue if a_ + b_ else 1.0
                print(f" | discordant A:R {a_}:{b_} McNemar p={p:.2g}")
            else:
                print()
        if fmt == "open":
            for v in ["tone", "fixable", "n_major", "words"]:
                if v in x:
                    wv = x.pivot_table(index="paper_id", columns="who", values=v)
                    print(f"   {v:8s} A-R {ci(wv.author - wv.reviewer)}" + (f" | A-C {ci(wv.author - wv.colleague)}" if "colleague" in wv else ""))
