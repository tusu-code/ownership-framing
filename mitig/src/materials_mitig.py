"""Mitigation study: manuscripts. Uses the confirmatory study's frozen conversion and cleaning code unchanged.

  python mitig/src/materials_mitig.py convert --workers 6   -> mitig/data/converted.jsonl (PDFs in mitig/data/pdfs)
  python mitig/src/materials_mitig.py select                -> mitig/data/manuscripts.jsonl, material_log.csv, sample_ids.csv
  python mitig/src/materials_mitig.py audit                 -> mitig/data/leak_audit.csv

Papers come from the confirmatory study's random frame (confirm/data/frame.csv), strictly in its order, starting
right after the last paper the confirmatory study examined (order 178), so no paper is in both studies. Conversion
(confirm/src/materials.py: to_text) and cleaning (clean_confirm, check, truncate) are the confirmatory study's,
called as they are. Papers at orders 179-279 were already converted for the confirmatory study and are read from
its cache; later ones are converted here. The first 4 passing papers are the smoke-test set, the next 240 the sample.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "confirm" / "src"))
import materials as M  # noqa: E402  (frozen confirmatory code)

CF = ROOT / "confirm" / "data"
D = ROOT / "mitig" / "data"
START = 179                  # first frame position not examined by the confirmatory study
N_SMOKE, N_SAMPLE = 4, 240
# Papers that pass the frozen checks but fail the residual-cue audit (leak_audit.py, blocking cues). The confirmatory
# code is frozen, so they are excluded rather than fixed; the next passing paper in frame order takes their place.
EXCLUDE = {"2606.02237": "excluded: publication notice left in text (workshop at ICML 2026)",
           "2606.22874": "excluded: reference list not cut",
           "2609.13789": "excluded: reference list not cut"}


def frame():
    return sorted(csv.DictReader(open(CF / "frame.csv")), key=lambda x: int(x["order"]))


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
    ids = open(D / "ids_280_479.txt").read().split()
    todo = [a for a in ids if a not in done and (D / "pdfs" / f"{a}.pdf").exists()]
    with open(cache, "a") as f, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(convert_one, todo):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(f"{r['arxiv_id']} {r['status']}", flush=True)


def authors():
    a = json.load(open(CF / "arxiv_authors.json"))
    a.update(json.load(open(D / "arxiv_authors_280_479.json")))
    return a


def select():
    conv = {}
    for path in (CF / "converted.jsonl", D / "converted.jsonl"):
        if path.exists():
            for l in open(path):
                r = json.loads(l); conv[r["arxiv_id"]] = r
    A = authors()
    need = N_SMOKE + N_SAMPLE
    rows, docs, n_ok = [], [], 0
    for r in frame():
        if int(r["order"]) < START:
            continue
        if n_ok >= need:
            break
        c = conv.get(r["arxiv_id"])
        if c is None:
            print(f"stopped at order {r['order']} ({r['arxiv_id']}): not converted yet"); break
        role, status, tokens, trunc = "", c["status"], 0, False
        if status == "converted":
            body, status = M.clean_confirm(c["md"], r["title"], A.get(r["arxiv_id"], []))
            if status == "ok":
                ok, status, tokens = M.check(r["title"], body)
                if ok and r["arxiv_id"] in EXCLUDE:
                    ok, status = False, EXCLUDE[r["arxiv_id"]]
                if ok:
                    body, tokens, trunc = M.truncate(body, M.MAX_TOKENS)
                    role = "smoke" if n_ok < N_SMOKE else "sample"; n_ok += 1
                    docs.append({"paper_id": r["arxiv_id"], "title": r["title"], "text": body, "n_tokens": tokens,
                                 "truncated": trunc, "role": role, "order": int(r["order"])})
        rows.append(dict(order=r["order"], arxiv_id=r["arxiv_id"], status=status, tokens=tokens,
                         truncated=int(trunc), role=role))
    with open(D / "material_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(D / "manuscripts.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(D / "sample_ids.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame_order", "arxiv_id", "role", "n_tokens", "truncated"])
        for d in docs:
            w.writerow([d["order"], d["paper_id"], d["role"], d["n_tokens"], int(d["truncated"])])
    json.dump(A, open(D / "arxiv_authors.json", "w"), ensure_ascii=False)
    print(f"{n_ok} passing papers out of {len(rows)} examined (need {need})")


def audit():
    import leak_audit as LA
    LA.C = D
    LA.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "select", "audit"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"convert": lambda: convert(a.workers), "select": select, "audit": audit}[a.cmd]()
