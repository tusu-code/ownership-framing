"""Turn published papers into convincing unpublished submissions (Pilot 2).

  python src/disguise.py run       # rename coined names, new title, paraphrase prose
  python src/disguise.py report    # fidelity and verbatim-overlap summary

Steps per paper, all done by a model family that is NOT evaluated (config: disguise_model):
  1. Coined names (method, model, dataset, metric) and the title get neutral replacements,
     applied to the whole text, tables included.
  2. Prose is paraphrased section by section. Tables and display equations are swapped for
     [[KEEP_n]] placeholders first and restored verbatim afterwards.
  3. Each paraphrased chunk must keep >= 95% of the original numbers and every placeholder;
     otherwise it is retried, and after the last retry the original chunk is kept and logged.
Every chunk call is cached in results/disguise_cache.jsonl, so a rerun resumes.
"""
import argparse
import asyncio
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import NAMEMAP, REWRITE  # noqa: E402
from run import Client, load_config, parse_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
CACHE = RES / "disguise_cache.jsonl"
OUT = DATA / "manuscripts_disguised.jsonl"
ENC = tiktoken.get_encoding("o200k_base")
NUM = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?")
CHUNK_TOKENS = 1400


def numbers(s):
    return collections.Counter(NUM.findall(s))


def ngram_overlap(a, b, n=8):
    """Share of word 8-grams of the original that survive verbatim in the rewrite."""
    wa, wb = a.split(), b.split()
    ga = {tuple(wa[i:i + n]) for i in range(len(wa) - n + 1)}
    gb = {tuple(wb[i:i + n]) for i in range(len(wb) - n + 1)}
    return len(ga & gb) / len(ga) if ga else 0.0


def protect(text):
    """Swap tables and display math for placeholders; return masked text and the blocks."""
    blocks = []

    def keep(m):
        blocks.append(m.group(0))
        return f"[[KEEP_{len(blocks) - 1}]]"

    t = re.sub(r"\$\$.*?\$\$", keep, text, flags=re.S)
    t = re.sub(r"(?m)(?:^\|.*\n?)+", lambda m: keep(m) + "\n", t)
    return t, blocks


def restore(text, blocks):
    return re.sub(r"\[\[KEEP_(\d+)\]\]", lambda m: blocks[int(m.group(1))], text)


def split_chunks(masked):
    """Split at headings, then pack sections into chunks of at most CHUNK_TOKENS."""
    sections = re.split(r"(?m)(?=^#{1,6} )", masked)
    chunks, cur = [], ""
    for sec in sections:
        if not sec.strip():
            continue
        if cur and len(ENC.encode(cur + sec)) > CHUNK_TOKENS:
            chunks.append(cur)
            cur = ""
        if len(ENC.encode(sec)) > CHUNK_TOKENS:
            for para in re.split(r"(?<=\n\n)", sec):
                if cur and len(ENC.encode(cur + para)) > CHUNK_TOKENS:
                    chunks.append(cur)
                    cur = ""
                cur += para
        else:
            cur += sec
    if cur.strip():
        chunks.append(cur)
    return chunks


def apply_names(text, names):
    names = [n for n in names if isinstance(n, dict) and n.get("original") and n.get("replacement")]
    # Coined names carry a capital letter, a digit or several words; this keeps "attention" or "model" safe.
    names = [n for n in names if len(n["original"]) >= 3 and n["original"] != n["replacement"]
             and (re.search(r"[A-Z0-9]", n["original"]) or " " in n["original"].strip())]
    for n in sorted(names, key=lambda n: -len(n["original"])):
        text = re.sub(r"(?<![\w-])" + re.escape(n["original"]) + r"(?![\w-])", n["replacement"], text)
    return text, names


def load_cache():
    c = {}
    if CACHE.exists():
        for line in open(CACHE):
            r = json.loads(line)
            c[r["key"]] = r
    return c


def key_of(*parts):
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:16]


async def call(client, prompt, max_tokens):
    out = await client.chat([{"role": "user", "content": prompt}], max_tokens)
    return out["text"]


async def namemap(client, cache, f, pid, title, text):
    k = key_of("names", client.c["model"], pid, title, text[:3000])
    if k in cache:
        return cache[k]["obj"]
    obj = None
    for _ in range(4):
        try:
            obj = parse_json(await call(client, NAMEMAP.format(title=title, opening=text[:3000]), 3000))
            if obj and obj.get("new_title"):
                break
        except Exception as e:  # noqa: BLE001
            print(f"  names {pid}: {str(e)[:120]}")
            await asyncio.sleep(10)
    if not (obj and obj.get("new_title")):
        print(f"  names {pid}: no usable answer, title and names left unchanged")
        return {"names": [], "new_title": title}
    rec = {"key": k, "kind": "names", "model": client.c["model"], "paper_id": pid, "obj": obj}
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()
    cache[k] = rec
    return obj


async def retitle(client, cache, f, pid, title, text):
    k = key_of("title", client.c["model"], pid, title)
    if k in cache:
        return cache[k]["title"]
    prompt = ("Give a new title for this machine learning paper with the same meaning but different wording. "
              "Output only the title.\n\nCurrent title: " + title + "\n\n" + text[:1500])
    for _ in range(4):
        try:
            new = (await call(client, prompt, 200)).strip().strip('"').splitlines()[0].strip()
            if new and new.lower() != title.lower():
                rec = {"key": k, "kind": "title", "model": client.c["model"], "paper_id": pid, "title": new}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                cache[k] = rec
                return new
        except Exception as e:  # noqa: BLE001
            print(f"  title {pid}: {str(e)[:120]}")
            await asyncio.sleep(10)
    return title


async def rewrite_chunk(client, cache, f, pid, i, chunk, retries=3):
    k = key_of("chunk", client.c["model"], pid, str(i), chunk)
    if k in cache:
        return cache[k]
    want, places = numbers(chunk), set(re.findall(r"\[\[KEEP_\d+\]\]", chunk))
    best = None
    for attempt in range(retries + 1):
        try:
            text = (await call(client, REWRITE.format(passage=chunk), 6000)).strip()
            text = re.sub(r"^<passage>\s*|\s*</passage>$", "", text)
        except Exception as e:  # noqa: BLE001
            print(f"  chunk {pid}#{i}: {str(e)[:120]}")
            await asyncio.sleep(10 * (attempt + 1))
            continue
        got = numbers(text)
        # A chunk without numbers (often the abstract) trivially keeps all of them.
        num_keep = sum((want & got).values()) / sum(want.values()) if want else 1.0
        ok = places <= set(re.findall(r"\[\[KEEP_\d+\]\]", text)) and num_keep >= 0.95 \
            and 0.6 < len(text) / max(1, len(chunk)) < 1.6
        cand = dict(text=text, num_keep=num_keep, ok=ok, attempt=attempt)
        if ok:
            best = cand
            break
        if best is None or num_keep > best["num_keep"]:
            best = cand
    if best is None or not best["ok"]:
        # Not cached: a rerun should try this chunk again instead of keeping the original.
        return dict(text=chunk, num_keep=1.0, ok=False, attempt=retries, fallback=True, overlap8=1.0)
    rec = {"key": k, "kind": "chunk", "model": client.c["model"], "paper_id": pid, "i": i, **best,
           "overlap8": ngram_overlap(chunk, best["text"])}
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()
    cache[k] = rec
    return rec


async def disguise_paper(client, cache, f, doc):
    pid = doc["paper_id"]
    nm = await namemap(client, cache, f, pid, doc["title"], doc["text"])
    renamed, used = apply_names(doc["text"], nm.get("names", []))
    masked, blocks = protect(renamed)
    chunks = split_chunks(masked)
    recs = await asyncio.gather(*[rewrite_chunk(client, cache, f, pid, i, c) for i, c in enumerate(chunks)])
    text = restore("".join(r["text"] if r["text"].endswith("\n") else r["text"] + "\n\n" for r in recs), blocks)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # Second pass: names inside restored tables/equations, or put back by the paraphrase.
    for n in sorted(used, key=lambda n: -len(n["original"])):
        text = re.sub(r"(?<![A-Za-z0-9])" + re.escape(n["original"]) + r"(?![A-Za-z0-9])", n["replacement"], text)
    title, _ = apply_names(nm.get("new_title") or doc["title"], used)
    if title.strip().lower() == doc["title"].strip().lower():
        title = await retitle(client, cache, f, pid, doc["title"], text)
    # Names the model missed but that still appear in the original title stay visible in the report.
    return dict(paper_id=pid, disguise_model=client.c["model"], title=title, text=text, n_tokens=len(ENC.encode(text)),
                original_title=doc["title"], names=used, n_chunks=len(chunks),
                chunks_fallback=sum(1 for r in recs if r.get("fallback")),
                num_keep=sum(r["num_keep"] for r in recs) / len(recs),
                overlap8=ngram_overlap(doc["text"], text))


async def run(cfg, sample_file, out_path, batch_size):
    """Disguise every paper in sample_file; each finished paper is appended to out_path at once,
    and papers already in out_path are skipped, so an interrupted run resumes."""
    sample = pd.read_csv(DATA / sample_file, index_col="paper_id")
    # A paper counts as done only if its latest record has no chunk left in the original wording.
    latest = {}
    if out_path.exists():
        for l in open(out_path):
            r = json.loads(l)
            latest[r["paper_id"]] = r
    # Also redo papers whose rename step got no answer (e.g. the balance ran out); a paper with no
    # coined names is fine as long as the rename step itself succeeded and is in the cache.
    # A clean paper stays as it is whichever model rewrote it (it may already have been reviewed).
    named = names_ok_ids()
    done = {pid for pid, r in latest.items() if r["chunks_fallback"] == 0 and pid in named}
    docs = [json.loads(l) for l in open(DATA / "manuscripts.jsonl")]
    docs = [d for d in docs if d["paper_id"] in sample.index and d["paper_id"] not in done]
    name = cfg["disguise_model"]
    client = Client(name, cfg["models"][name], cfg.get("defaults", {}))
    RES.mkdir(exist_ok=True)
    cache = load_cache()
    print(f"[disguise:{client.c['model']}] {len(docs)} papers to do ({len(done)} already in {out_path.name})", flush=True)
    with open(CACHE, "a") as f, open(out_path, "a") as fo:
        for i in range(0, len(docs), batch_size):
            batch = await asyncio.gather(*[disguise_paper(client, cache, f, d) for d in docs[i:i + batch_size]])
            for b in batch:
                fo.write(json.dumps(b, ensure_ascii=False) + "\n")
                fo.flush()
                print(f"[{len(done) + i + 1}] {b['paper_id']} chunks={b['n_chunks']} fallback={b['chunks_fallback']} "
                      f"num_keep={b['num_keep']:.3f} overlap8={b['overlap8']:.3f} names={len(b['names'])} | {b['title'][:60]}",
                      flush=True)
    await client.http.aclose()
    report(out_path)


def names_ok_ids():
    """Papers whose rename step returned a usable answer with any model (only successes are cached)."""
    if not CACHE.exists():
        return set()
    return {r["paper_id"] for r in map(json.loads, open(CACHE)) if r.get("kind") == "names"}


def load_latest(out_path):
    latest = {}
    for l in open(out_path):
        r = json.loads(l)
        latest[r["paper_id"]] = r
    return list(latest.values())


def report(out_path=OUT):
    d = pd.DataFrame(load_latest(out_path))
    s = pd.read_csv(DATA / "sample_full.csv", index_col="paper_id")
    d = d.join(s[["accept"]], on="paper_id")
    print(d.groupby("accept")[["num_keep", "overlap8", "chunks_fallback", "n_tokens"]].mean().round(3))
    print("papers:", len(d), "| with any fallback chunk:", int((d.chunks_fallback > 0).sum()))
    print("mean names renamed:", round(d.names.map(len).mean(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--sample", default="sample_new.csv")
    ap.add_argument("--out", default="manuscripts_disguised_new.jsonl")
    ap.add_argument("--batch", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config()
    out_path = DATA / args.out
    asyncio.run(run(cfg, args.sample, out_path, args.batch)) if args.cmd == "run" else report(out_path)


if __name__ == "__main__":
    main()
