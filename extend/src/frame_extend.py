"""Extension study A (exploratory): sampling frame of non-machine-learning papers.

  python extend/src/frame_extend.py   -> extend/data/frame.csv, extend/data/arxiv_authors.json

Frame: arXiv papers whose primary category is in one of three fields outside machine learning, first version
submitted between 1 April and 31 October 2026 (after Claude Sonnet 4.6's training cutoff):
  biology   q-bio.*
  physics   cond-mat.*
  economics econ.*
A paper is eligible if it is not cross-listed to any cs.* or stat.ML category, has no journal reference in the
arXiv metadata (so it does not read as published) and passes the confirmatory study's EXCLUDE rule on title,
abstract and comments (surveys, reviews, theses and the like). Eligible papers are put in random order per field
with SEED; extend/src/materials_extend.py takes them in that order.
"""
import csv
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "confirm" / "src"))
from frame import EXCLUDE, MONTHS, NS  # noqa: E402  (confirmatory frame rules, unchanged)

D = HERE.parents[0] / "data"
SEED = 20261106
FIELDS = {"biology": "q-bio", "physics": "cond-mat", "economics": "econ"}
ML_CROSS = re.compile(r"^(cs\.|stat\.ML$)")


def fetch(query, start, n=1000):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "start": start, "max_results": n, "sortBy": "submittedDate", "sortOrder": "ascending"})
    last = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ownership-framing-study/1.0 (mailto:anonymous@example.com)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return ET.fromstring(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(10 * (attempt + 1))
    raise last


def main():
    rows, authors, seen = [], {}, set()
    for field, prefix in FIELDS.items():
        for lo, hi in MONTHS:
            start = 0
            while True:
                root = fetch(f"cat:{prefix}* AND submittedDate:[{lo}0000 TO {hi}2359]", start)
                entries = root.findall("a:entry", NS)
                if not entries:
                    break
                for e in entries:
                    aid = e.find("a:id", NS).text.rsplit("/abs/", 1)[1].split("v")[0]
                    prim = e.find("arxiv:primary_category", NS).get("term")
                    if aid in seen or not prim.startswith(prefix):
                        continue
                    seen.add(aid)
                    cats = [c.get("term") for c in e.findall("a:category", NS)]
                    title = re.sub(r"\s+", " ", e.find("a:title", NS).text).strip()
                    abstract = re.sub(r"\s+", " ", e.find("a:summary", NS).text).strip()
                    c = e.find("arxiv:comment", NS)
                    comment = re.sub(r"\s+", " ", c.text).strip() if c is not None and c.text else ""
                    jref = e.find("arxiv:journal_ref", NS)
                    eligible = (not EXCLUDE.search(f"{title} {abstract[:400]} {comment}")
                                and not any(ML_CROSS.match(x) for x in cats) and jref is None)
                    authors[aid] = [re.sub(r"\s+", " ", a.find("a:name", NS).text).strip() for a in e.findall("a:author", NS)]
                    rows.append(dict(field=field, arxiv_id=aid, primary=prim, published=e.find("a:published", NS).text[:10],
                                     title=title, comment=comment, eligible=int(eligible), abstract=abstract))
                print(f"{field} {lo}: {start + len(entries)} fetched, {len(rows)} rows so far", flush=True)
                start += len(entries)
                time.sleep(3.5)                  # arXiv API terms: at most one request every three seconds
                if len(entries) < 1000:
                    break
    out = []
    for field in FIELDS:
        elig = [r for r in rows if r["field"] == field and r["eligible"]]
        random.Random(f"{SEED}-{field}").shuffle(elig)
        for i, r in enumerate(elig):
            r["order"] = i
        out += elig
    D.mkdir(parents=True, exist_ok=True)
    with open(D / "frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["field", "order", "arxiv_id", "primary", "published", "title", "comment", "eligible", "abstract"])
        w.writeheader()
        w.writerows(out)
    json.dump({r["arxiv_id"]: authors[r["arxiv_id"]] for r in out}, open(D / "arxiv_authors.json", "w"), ensure_ascii=False)
    for field in FIELDS:
        print(f"{field}: {sum(r['field'] == field for r in rows)} papers, {sum(r['field'] == field for r in out)} eligible")


if __name__ == "__main__":
    main()
