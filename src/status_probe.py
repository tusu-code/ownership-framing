"""Does the disguise work? Ask each evaluated model whether the paper is already published.

  python src/status_probe.py run [--model gpt]
  python src/status_probe.py report

Each paper is shown twice, in separate fresh conversations: the cleaned original (original
title) and the disguised version, both rendered as an anonymous submission and cut to the
same opening length. Results go to results/status_<model>.jsonl.
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import STATUS_PROBE, render_submission  # noqa: E402
from run import Client, load_config, parse_json  # noqa: E402

import tiktoken  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
ENC = tiktoken.get_encoding("o200k_base")
OPEN_TOKENS = 1500
MODELS = ["gpt", "glm", "deepseek", "claude"]   # claude added 2026-10-31


def opening(title, text):
    ids = ENC.encode(render_submission(title, text))
    return ENC.decode(ids[:OPEN_TOKENS])


def jobs():
    sample = pd.read_csv(DATA / "sample_pilot.csv", index_col="paper_id")
    orig = {json.loads(l)["paper_id"]: json.loads(l) for l in open(DATA / "manuscripts.jsonl")}
    disg = {json.loads(l)["paper_id"]: json.loads(l) for l in open(DATA / "manuscripts_disguised.jsonl")}
    out = []
    for pid in sample.index:
        for version, d in (("original", orig[pid]), ("disguised", disg[pid])):
            out.append(dict(paper_id=pid, version=version,
                            prompt=STATUS_PROBE.format(opening=opening(d["title"], d["text"]))))
    return out


async def run_model(cfg, name):
    client = Client(name, cfg["models"][name], cfg.get("defaults", {}))
    path = RES / f"status_{name}.jsonl"
    done = set()
    if path.exists():
        done = {(r["paper_id"], r["version"]) for r in map(json.loads, open(path)) if r.get("ok")}
    todo = [j for j in jobs() if (j["paper_id"], j["version"]) not in done]
    random.Random(cfg["seed"]).shuffle(todo)
    print(f"[{name}] {len(todo)} calls ({len(done)} done)", flush=True)

    async def one(j, f):
        async with client.sem:
            for attempt in range(5):
                try:
                    out = await client.chat([{"role": "user", "content": j["prompt"]}], 2000)
                    obj = parse_json(out["text"])
                    ok = obj is not None and isinstance(obj.get("p_published"), (int, float))
                    if ok or attempt == 4:
                        f.write(json.dumps({"paper_id": j["paper_id"], "version": j["version"], "ok": ok,
                                            "parsed": obj, "raw": out["text"][:2000],
                                            "provider_model": out.get("model")}, ensure_ascii=False) + "\n")
                        f.flush()
                        return
                except Exception as e:  # noqa: BLE001
                    if attempt == 4:
                        print(f"  {j['paper_id']} {j['version']}: {str(e)[:100]}")
                await asyncio.sleep(5 * (attempt + 1))

    with open(path, "a") as f:
        await asyncio.gather(*[one(j, f) for j in todo])
    await client.http.aclose()


def report():
    sample = pd.read_csv(DATA / "sample_pilot.csv", index_col="paper_id")
    orig = {json.loads(l)["paper_id"]: json.loads(l)["title"] for l in open(DATA / "manuscripts.jsonl")}
    rows = []
    for name in MODELS:
        p = RES / f"status_{name}.jsonl"
        if not p.exists():
            continue
        for r in map(json.loads, open(p)):
            if not r.get("ok"):
                continue
            o = r["parsed"]
            guess = str(o.get("title_guess") or "").lower()
            true_title = orig[r["paper_id"]].lower()
            # A correct identification: the guessed title shares most of its words with the true title.
            tw, gw = set(true_title.split()), set(guess.split())
            hit = bool(gw) and len(tw & gw) / max(1, len(tw)) >= 0.6
            rows.append(dict(model=name, paper_id=r["paper_id"], version=r["version"],
                             p_pub=float(o["p_published"]), recognized=bool(o.get("recognized")), title_hit=hit,
                             accept=bool(sample.accept[r["paper_id"]])))
    d = pd.DataFrame(rows)
    if d.empty:
        print("no results")
        return
    print(d.groupby(["model", "version"])[["p_pub", "recognized", "title_hit"]].mean().round(3))
    print("\nPaired original - disguised, P(published):")
    for name, g in d.groupby("model"):
        w = g.pivot_table(index="paper_id", columns="version", values="p_pub")
        w = w.dropna()
        diff = w["original"] - w["disguised"]
        p = stats.wilcoxon(diff).pvalue if (diff != 0).sum() >= 5 else float("nan")
        print(f"  {name}: n={len(w)} mean diff {diff.mean():+.3f}, Wilcoxon p={p:.3g}")
    print("\nDoes P(published) still separate accepted from rejected papers? (AUC, 0.5 = cannot tell)")
    from sklearn.metrics import roc_auc_score
    for (name, ver), g in d.groupby(["model", "version"]):
        print(f"  {name:<9}{ver:<10} AUC {roc_auc_score(g.accept, g.p_pub):.3f}")
    d.to_csv(RES / "status_probe_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--model", action="append")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "run":
        for name in args.model or MODELS:
            asyncio.run(run_model(cfg, name))
    report()


if __name__ == "__main__":
    main()
