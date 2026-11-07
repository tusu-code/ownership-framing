"""Extension study A (exploratory): manuscripts from biology, physics and economics.

  python extend/src/materials_extend.py download --per-field 70   -> extend/data/pdfs/
  python extend/src/materials_extend.py convert --workers 6       -> extend/data/converted.jsonl
  python extend/src/materials_extend.py select                    -> extend/data/manuscripts.jsonl, material_log.csv
  python extend/src/materials_extend.py audit                     -> extend/data/leak_audit.csv

Papers are taken per field strictly in the random order of extend/data/frame.csv. Download, conversion, cleaning,
pass criteria and truncation are the confirmatory study's (confirm/src/materials.py), called unchanged. Per field
the first passing paper is the smoke-test paper and the next N_SAMPLE form the sample.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "confirm" / "src"))
import materials as M  # noqa: E402  (confirmatory code, unchanged)

D = ROOT / "extend" / "data"
FIELDS = ("biology", "physics", "economics")
N_SMOKE, N_SAMPLE = 1, 34
DL_PER_FIELD = 55            # extend/src/dl2.sh downloaded the first 55 frame papers of each field
# Papers that pass the checks but fail the residual-cue audit; the next passing paper takes their place.
EXCLUDE = {"2610.19833": "excluded: author-contribution statement and references left after the cut",
           "2607.28230": "excluded: reference list on one line, not cut",
           "2606.15357": "excluded: publication notice left in text (Oxford University Press)"}
THESIS = re.compile(r"\b(dissertation|thesis) (submitted|presented)|in partial fulfil+ment of the requirements|"
                    r"\bdoctor of philosophy\b", re.I)
# Journal cues outside machine learning, added to the confirmatory audit for these fields.
JOURNAL_CUES = re.compile(r"\b(Phys\.\s?Rev\.|Physical Review [A-Z]|Nature (Physics|Communications)|PNAS|"
                          r"Econometrica|American Economic Review|Journal of [A-Z][a-z]+ Economics|Q\.\s?J\.\s?Econ|"
                          r"eLife|PLOS|Bioinformatics,|Nucleic Acids Res|"
                          r"(submitted|accepted|published) (to|in|for) (the )?(journal|[A-Z][\w.]+ (Journal|Review|Letters)))\b")


# Physics (APS/REVTeX) and some biology templates print the abstract as an unlabelled paragraph, which the
# confirmatory cleaner rejects ("no abstract"). Fallback, used only then: the abstract from the arXiv metadata,
# followed by the text from the first Introduction heading on; everything before that heading (title block,
# authors, affiliations, dates, the unlabelled abstract) is dropped before the confirmatory cleaning runs.
INTRO = re.compile(r"^\W{0,8}(?:[IVX]+|\d+)?\W{0,4}(introduction|background|motivation)\W*$", re.I | re.M)


def with_metadata_abstract(md, abstract):
    if not abstract:
        return None
    plain = re.sub(r"[*_`]", "", md)
    m = INTRO.search(plain, 0, max(len(plain) // 3, 1))
    if not m:
        return None
    return f"## Abstract\n\n{abstract}\n\n" + plain[m.start():]


# Numbered reference lists without a heading the confirmatory cut recognizes (APS and similar templates):
# cut at the first "[1]"-style entry that starts a run of at least three, after the first third of the text.
REF_ENTRY = re.compile(r"^\W{0,6}\[(\d{1,3})\]\s+\S", re.M)


def cut_numbered_refs(text):
    hits = [m for m in REF_ENTRY.finditer(text) if m.start() > len(text) // 3]
    for i, m in enumerate(hits[:-2]):
        if int(m.group(1)) <= 2 and int(hits[i + 2].group(1)) == int(m.group(1)) + 2:
            return text[:m.start()].rstrip(), True
    return text, False


def frame():
    rows = list(csv.DictReader(open(D / "frame.csv")))
    return {f: sorted([r for r in rows if r["field"] == f], key=lambda x: int(x["order"])) for f in FIELDS}


def download(per_field):
    (D / "pdfs").mkdir(parents=True, exist_ok=True)
    for f, rows in frame().items():
        for r in rows[:per_field]:
            p = D / "pdfs" / f"{r['arxiv_id']}.pdf"
            if p.exists() and p.stat().st_size > 10000:
                continue
            ok = M.download(r["arxiv_id"], p)
            print(f"{f} {r['order']} {r['arxiv_id']} {'ok' if ok else 'failed'}", flush=True)
            time.sleep(3)


def convert_one(aid):
    pdf = D / "pdfs" / f"{aid}.pdf"
    if not (pdf.exists() and pdf.stat().st_size > 10000):
        return {"arxiv_id": aid, "status": "download failed"}
    try:
        return {"arxiv_id": aid, "status": "converted", "md": M.to_text(pdf)}
    except Exception as e:  # noqa: BLE001
        return {"arxiv_id": aid, "status": f"conversion failed: {str(e)[:60]}"}


def convert(workers):
    from concurrent.futures import ProcessPoolExecutor
    cache = D / "converted.jsonl"
    done = {json.loads(l)["arxiv_id"] for l in open(cache)} if cache.exists() else set()
    todo = [r["arxiv_id"] for rows in frame().values() for r in rows
            if r["arxiv_id"] not in done and (D / "pdfs" / f"{r['arxiv_id']}.pdf").exists()]
    with open(cache, "a") as f, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(convert_one, todo):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(f"{r['arxiv_id']} {r['status']}", flush=True)


def select(final=False):
    """final: the download list is finished, so a frame paper with no PDF failed to download (or was over
    10 MB, which the download script skips) and is logged as such instead of stopping the walk."""
    conv = {}
    for l in open(D / "converted.jsonl"):
        r = json.loads(l); conv[r["arxiv_id"]] = r
    A = json.load(open(D / "arxiv_authors.json"))
    rows, docs = [], []
    for f, fr in frame().items():
        n_ok = 0
        for r in fr:
            if n_ok >= N_SMOKE + N_SAMPLE:
                break
            c = conv.get(r["arxiv_id"])
            if c is None and final and int(r["order"]) < DL_PER_FIELD and not (D / "pdfs" / f"{r['arxiv_id']}.pdf").exists():
                c = {"status": "download failed or over 10 MB"}
            if c is None:
                print(f"{f}: stopped at order {r['order']} ({r['arxiv_id']}): not converted yet"); break
            role, status, tokens, trunc = "", c["status"], 0, False
            if status == "converted":
                body, status = M.clean_confirm(c["md"], r["title"], A.get(r["arxiv_id"], []))
                if status == "no abstract":
                    md2 = with_metadata_abstract(c["md"], r.get("abstract", ""))
                    if md2:
                        body, status = M.clean_confirm(md2, r["title"], A.get(r["arxiv_id"], []))
                        if status == "ok":
                            status = "ok (abstract from arXiv metadata)"
                if status.startswith("ok"):
                    via = status
                    ok, status, tokens = M.check(r["title"], body)
                    if ok and via != "ok":
                        status = via
                    if ok and r["arxiv_id"] in EXCLUDE:
                        ok, status = False, EXCLUDE[r["arxiv_id"]]
                    if ok:
                        body, cut = cut_numbered_refs(body)
                        if cut:
                            ok, chk, tokens = M.check(r["title"], body)
                            status = (status + "; " if status != "ok" else "") + "numbered references cut"
                            if not ok:
                                status = chk
                    if ok and THESIS.search(c["md"][:6000]):
                        ok, status = False, "excluded: thesis or dissertation"
                    if ok:
                        body, tokens, trunc = M.truncate(body, M.MAX_TOKENS)
                        role = "smoke" if n_ok < N_SMOKE else "sample"; n_ok += 1
                        docs.append({"paper_id": r["arxiv_id"], "field": f, "primary": r["primary"], "title": r["title"],
                                     "text": body, "n_tokens": tokens, "truncated": trunc, "role": role, "order": int(r["order"])})
            rows.append(dict(field=f, order=r["order"], arxiv_id=r["arxiv_id"], status=status, tokens=tokens,
                             truncated=int(trunc), role=role))
        print(f"{f}: {n_ok} passing of {sum(x['field'] == f for x in rows)} examined")
    with open(D / "material_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(D / "manuscripts.jsonl", "w") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def audit():
    import leak_audit as LA
    LA.C = D
    LA.main()
    hits = []
    for d in map(json.loads, open(D / "manuscripts.jsonl")):
        m = JOURNAL_CUES.findall(d["text"])
        if m:
            hits.append((d["paper_id"], d["field"], d["role"], sorted({x[0] for x in m})[:4]))
    print(f"journal cues: {len(hits)} papers")
    for h in hits:
        print("  ", *h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "convert", "select", "audit"])
    ap.add_argument("--per-field", type=int, default=70)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--final", action="store_true", help="downloads finished: papers without a PDF count as failed")
    a = ap.parse_args()
    {"download": lambda: download(a.per_field), "convert": lambda: convert(a.workers), "select": lambda: select(a.final),
     "audit": audit}[a.cmd]()
