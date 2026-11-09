"""Summarise the WildChat classification and check it.

  python wildchat/src/summarize.py  -> wildchat/out/summary.json, paper/numbers_wildchat.tex (keys wc/...),
                                       wildchat/out/audit_sample.csv (random sample for a hand check)

Three classifiers from three model families with the same prompt: DeepSeek V4.1 Flash and Claude Haiku 4.5 on all
candidates, GPT-6.1 Sol on every candidate either of the two calls an assessment. Final labels are the majority
of the three (a request counts as an assessment when at least two say so; its relation is the one at least two
give, else "unresolved"). Pairwise agreement is reported as raw agreement and Cohen's kappa. One user (hashed IP)
can send several requests, so intervals are bootstrapped over users.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "wildchat" / "out"
OUT = ROOT / "paper" / "numbers_wildchat.tex"
rng = np.random.default_rng(20261108)
REL = ["own", "reviewer", "other", "none", "unresolved"]


def kappa(a, b):
    a, b = pd.Series(a).astype(str), pd.Series(b).astype(str)
    po = (a == b).mean()
    cats = set(a) | set(b)
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def load(name):
    rows = [json.loads(l) for l in open(O / name)]
    d = pd.DataFrame([r for r in rows if r.get("ok")]).drop_duplicates("h", keep="last").set_index("h")
    return d


def pct(x):
    return f"{x * 100:.0f}\\%"


def main():
    cand = pd.DataFrame([json.loads(l) for l in open(O / "candidates.jsonl")]).set_index("h")
    ds = load("classified.jsonl")
    hk = load("classified_haiku.jsonl")
    gp = load("classified_gpt.jsonl")
    lab = pd.DataFrame(index=cand.index)
    for k, df in (("d", ds), ("h", hk), ("g", gp)):
        lab = lab.join(df[["assess", "relation", "role_play"]].add_suffix("_" + k), how="left")
    lab["assess_g"] = lab.assess_g.astype(object).where(lab.assess_g.notna(), False).astype(bool)          # GPT ran only where DeepSeek or Haiku said yes
    votes = lab[["assess_d", "assess_h", "assess_g"]].astype(bool).sum(axis=1)
    lab["assess"] = votes >= 2

    def maj(r, col):
        v = [r[f"{col}_{k}"] for k in "dhg" if r[f"assess_{k}"] is True or r[f"assess_{k}"] == 1]
        v = [x for x in v if isinstance(x, (str, bool))]
        for x in set(v):
            if v.count(x) >= 2:
                return x
        return "unresolved" if col == "relation" else False
    lab["relation"] = lab.apply(lambda r: maj(r, "relation"), axis=1)
    lab["role_play"] = lab.apply(lambda r: maj(r, "role_play"), axis=1)
    d = cand.join(lab[["assess", "relation", "role_play"]]).join(ds[["evidence"]])
    a = d[d.assess].copy()
    V, S = {}, {}
    V["wc/conv"] = "837{,}989"
    V["wc/cand"] = f"{len(cand):,}".replace(",", "{,}")
    V["wc/assess"] = f"{len(a):,}".replace(",", "{,}")
    V["wc/users"] = f"{a.ip.nunique():,}".replace(",", "{,}")
    S.update(candidates=len(cand), classified=len(d), assess=len(a), users=int(a.ip.nunique()),
             lang=a.lang.value_counts().to_dict())
    # shares with user-cluster bootstrap
    users = a.ip.unique()
    groups = {u: g for u, g in a.groupby("ip")}
    for rel in REL + ["stated"]:
        hit = a.relation.isin(["own", "reviewer", "other"]) if rel == "stated" else (a.relation == rel)
        est = hit.mean()
        bs = []
        for _ in range(2000):
            pick = rng.choice(users, len(users))
            g = pd.concat([groups[u] for u in pick])
            h = g.relation.isin(["own", "reviewer", "other"]) if rel == "stated" else (g.relation == rel)
            bs.append(h.mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        V[f"wc/{rel}/n"] = str(int(hit.sum()))
        V[f"wc/{rel}/pct"] = pct(est)
        V[f"wc/{rel}/ci"] = f"{est * 100:.0f}\\% [{lo * 100:.0f}, {hi * 100:.0f}]"
        S[rel] = dict(n=int(hit.sum()), share=est, lo=lo, hi=hi)
    # sensitivity: each user counted once (mean over users of their share of requests)
    per_user = a.assign(st=a.relation.isin(["own", "reviewer", "other"]), own=a.relation.eq("own")).groupby("ip")[["st", "own"]].mean()
    V["wc/user/stated"], V["wc/user/own"] = pct(per_user.st.mean()), pct(per_user.own.mean())
    S["per_user"] = dict(stated=per_user.st.mean(), own=per_user.own.mean(), users=len(per_user))
    V["wc/roleplay/n"] = str(int((a.role_play == True).sum()))  # noqa: E712
    V["wc/roleplay/pct"] = pct((a.role_play == True).mean())  # noqa: E712
    # own among role-play requests ("act as a reviewer of my paper")
    rp = a[a.role_play == True]  # noqa: E712
    nonrp = a[a.role_play != True]  # noqa: E712
    V["wc/nonrp/n"] = str(len(nonrp))
    V["wc/nonrp/stated/pct"] = pct((nonrp.relation.isin(["own", "reviewer", "other"])).mean())
    V["wc/nonrp/own/pct"] = pct((nonrp.relation == "own").mean())
    V["wc/roleplay/own/pct"] = pct((rp.relation == "own").mean()) if len(rp) else "--"
    S["roleplay"] = dict(n=len(rp), own=int((rp.relation == "own").sum()))
    # agreement, pairwise on the requests both members of a pair labelled
    S["agreement"] = {}
    for x, y, nx, ny in (("d", "h", "dsflash", "haiku"), ("d", "g", "dsflash", "gpt"), ("h", "g", "haiku", "gpt")):
        j = lab.dropna(subset=[f"assess_{x}", f"assess_{y}"])
        if y == "g":
            j = j[j.index.isin(gp.index)]
        po, k = kappa(j[f"assess_{x}"], j[f"assess_{y}"])
        both = j[(j[f"assess_{x}"] == True) & (j[f"assess_{y}"] == True)]  # noqa: E712
        po2, k2 = kappa(both[f"relation_{x}"], both[f"relation_{y}"])
        S["agreement"][f"{nx}-{ny}"] = dict(n=len(j), assess_po=po, assess_kappa=k, rel_n=len(both), rel_po=po2,
                                            rel_kappa=k2)
    ag = S["agreement"]["dsflash-haiku"]
    V["wc/agree/assess/k"] = f"${ag['assess_kappa']:.2f}$"
    V["wc/agree/rel/k/min"] = f"${min(v['rel_kappa'] for v in S['agreement'].values()):.2f}$"
    V["wc/agree/rel/k/max"] = f"${max(v['rel_kappa'] for v in S['agreement'].values()):.2f}$"
    V["wc/union"] = str(int(votes.ge(1).sum()))
    S["union"] = int(votes.ge(1).sum())
    # conservative count: the relation all three classifiers give
    un = lab.loc[a.index]
    un = un[(un.relation_d == un.relation_h) & (un.relation_h == un.relation_g)]
    V["wc/unan/n"] = str(len(un))
    for rel in ("own", "reviewer", "other", "none"):
        V[f"wc/unan/{rel}"] = str(int((un.relation_d == rel).sum()))
    V["wc/unan/stated"] = str(int(un.relation_d.isin(["own", "reviewer", "other"]).sum()))
    S["unanimous"] = un.relation_d.value_counts().to_dict()
    a[["lang", "relation", "role_play", "evidence", "head"]].assign(head=lambda x: x["head"].str.slice(0, 900)).to_csv(
        O / "assess_final.csv")
    # audit sample for the hand check
    aud = d.sample(min(120, len(d)), random_state=20261108)
    aud[["lang", "assess", "relation", "role_play", "evidence", "head"]].assign(
        head=lambda x: x["head"].str.slice(0, 900)).to_csv(O / "audit_sample.csv")
    (O / "summary.json").write_text(json.dumps(S, indent=1, default=float, ensure_ascii=False))
    lines = ["% Generated by wildchat/src/summarize.py. Do not edit.", r"\makeatletter"]
    lines += [f"\\@namedef{{num@{k}}}{{{v}}}" for k, v in sorted(V.items())]
    lines += [r"\makeatother"]
    OUT.write_text("\n".join(lines) + "\n")
    print(json.dumps(S, indent=1, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()
