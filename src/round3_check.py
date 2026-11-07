"""Quick look at the third round of runs (2026-11-02): fixed instruction, same-batch five-ask retest, second judge.

  python src/round3_check.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "full_study"
FULL = pd.read_csv(SRC / "data" / "sample_full.csv", index_col="paper_id")
RNG = np.random.default_rng(7)
ASKS = ["author", "author_review", "colleague", "venue", "reviewer"]


def h(t):
    return hashlib.sha1((t or "").encode()).hexdigest()[:16]


def ci(z):
    z = np.asarray(z, float)
    z = z[~np.isnan(z)]
    bs = z[RNG.integers(0, len(z), (4000, len(z)))].mean(1)
    nz = z[z != 0]
    p = stats.wilcoxon(nz).pvalue if len(nz) >= 5 else 1.0
    return f"{z.mean():+.2f} [{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}] p={p:.2g} n={len(z)}"


def load(m):
    out = {}
    for line in open(SRC / "results" / f"p2_review_{m}.jsonl"):
        r = json.loads(line)
        if r.get("ok") and str(r.get("relay", "")).startswith("https://api.sailapi") and r["paper_id"] in FULL.index:
            out[(r["paper_id"], r["condition"], r["rep"])] = r
    return out


def judged(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("ok"):
            out[r["hash"]] = r
    return out


JN = {k: v["parsed"] for k, v in judged(SRC / "results" / "p3_judge_neutral.jsonl").items()}


def fixed():
    print("=== 2. fixed reviewing instruction, full sample (model's own form)")
    for m in ["claude", "gpt", "glm", "deepseek"]:
        rows = []
        for (p, c, rep), r in load(m).items():
            if c.startswith("fixed") and rep == 0:
                o = r["parsed"]
                rows.append(dict(p=p, who=c.split("_")[1], y=float(o["rating"]),
                                 acc=float(str(o["recommendation"]).lower() == "accept")))
        d = pd.DataFrame(rows)
        if d.empty:
            continue
        w = d.pivot_table(index="p", columns="who", values="y").dropna()
        wa = d.pivot_table(index="p", columns="who", values="acc").dropna()
        a = int(((wa.author == 1) & (wa.reviewer == 0)).sum())
        b = int(((wa.author == 0) & (wa.reviewer == 1)).sum())
        print(f"  {m:9s} A-R rating {ci(w.author - w.reviewer)} | accept A {wa.author.mean():.0%} R {wa.reviewer.mean():.0%} "
              f"({a}:{b}) | mean A {w.author.mean():.2f} R {w.reviewer.mean():.2f}")


def same_batch():
    print("=== 1+3. same-batch retest (rep 2), five asks interleaved")
    for m, fmts in (("claude", ["open", "struct"]), ("glm", ["open"])):
        A = load(m)
        for fmt in fmts:
            rows = []
            for (p, c, rep), r in A.items():
                if rep != 2 or not c.startswith(fmt + "_"):
                    continue
                if fmt == "open":
                    j = JN.get(h(r["raw"]))
                    if not j:
                        continue
                    y = float(j["implied_rating"])
                else:
                    y = float(r["parsed"]["rating"])
                rows.append(dict(p=p, who=c[len(fmt) + 1:], y=y))
            d = pd.DataFrame(rows)
            if d.empty:
                continue
            w = d.pivot_table(index="p", columns="who", values="y")
            hm = FULL.human_mean.reindex(w.index)
            print(f"  {m} {fmt}: papers with all asks {w.dropna().shape[0]} | means "
                  + ", ".join(f"{k} {w[k].mean():.2f}" for k in ASKS if k in w))
            for a, b in [("author", "reviewer"), ("author", "colleague"), ("author", "author_review"),
                         ("author_review", "reviewer"), ("colleague", "reviewer"), ("author", "venue")]:
                if {a, b} <= set(w.columns):
                    x = w[[a, b]].dropna()
                    print(f"     {a:13s} - {b:13s} {ci(x[a] - x[b])}")
            print("     bias vs human mean: " + ", ".join(f"{k} {(w[k] - hm).mean():+.2f}" for k in ASKS if k in w))


def second_judge():
    print("=== 4. second judge (Haiku 4.5) vs main judge (Flash), same neutralized answers")
    rows = []
    for k, r in judged(SRC / "results" / "p3_judge_haiku.jsonl").items():
        f = JN.get(k)
        if f:
            rows.append(dict(model=r["model"], p=r["paper_id"], who=r["condition"].split("_", 1)[1],
                             haiku=float(r["parsed"]["implied_rating"]), flash=float(f["implied_rating"]),
                             h_tone=float(r["parsed"]["overall_tone"]), f_tone=float(f["overall_tone"])))
    d = pd.DataFrame(rows)
    print(f"  n {len(d)} | rating Spearman {stats.spearmanr(d.haiku, d.flash).statistic:.2f}, exact {(d.haiku == d.flash).mean():.0%}, "
          f"mean Haiku - Flash {(d.haiku - d.flash).mean():+.2f} | tone Spearman {stats.spearmanr(d.h_tone, d.f_tone).statistic:.2f}")
    for m, g in d.groupby("model"):
        print(f"  {m:9s} rating Spearman {stats.spearmanr(g.haiku, g.flash).statistic:.2f}")
        for col in ("haiku", "flash"):
            w = g.pivot_table(index="p", columns="who", values=col)
            line = f"     {col:5s} A-R {ci((w.author - w.reviewer).dropna())}"
            if "colleague" in w:
                line += f" | A-C {ci((w.author - w.colleague).dropna())}"
            print(line)


if __name__ == "__main__":
    fixed()
    same_batch()
    second_judge()
