"""Confirmatory study: sampling frame and random order of candidate papers.

  python confirm/src/frame.py   -> confirm/data/frame.csv (every eligible paper, in random order)

Frame: arXiv papers whose primary category is cs.LG and whose first version was submitted between
1 April and 31 October 2026, after the training cutoff of both models (March 2026). Eligibility uses only the
arXiv metadata (title, abstract, comments); see EXCLUDE. The order is a permutation with SEED; papers are then
processed in this order (confirm/src/materials.py) and the first ones that pass the material checks are used.
"""
import csv
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "frame.csv"
SEED = 20261104
MONTHS = [("20260401", "20260430"), ("20260501", "20260531"), ("20260601", "20260630"), ("20260701", "20260731"),
          ("20260801", "20260831"), ("20260901", "20260930"), ("20261001", "20261031")]
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
EXCLUDE = re.compile(r"\b(survey|review of|a review|literature review|systematic review|overview|tutorial|position paper|"
                     r"perspective|thesis|dissertation|lecture notes|book chapter|erratum|corrigendum|withdrawn|"
                     r"workshop|extended abstract|short paper|tiny paper|challenge report|competition report)\b", re.I)


def fetch(start, lo, hi, n=1000):
    q = f"cat:cs.LG AND submittedDate:[{lo}0000 TO {hi}2359]"
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": q, "start": start, "max_results": n, "sortBy": "submittedDate", "sortOrder": "ascending"})
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ownership-framing-study/1.0 (mailto:anonymous@example.com)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return ET.fromstring(r.read())
        except Exception as e:  # noqa: BLE001
            time.sleep(10 * (attempt + 1))
            last = e
    raise last


def main():
    rows, seen = [], set()
    for lo, hi in MONTHS:
        start = 0
        while True:
            root = fetch(start, lo, hi)
            entries = root.findall("a:entry", NS)
            if not entries:
                break
            for e in entries:
                aid = e.find("a:id", NS).text.rsplit("/abs/", 1)[1].split("v")[0]
                prim = e.find("arxiv:primary_category", NS).get("term")
                if aid in seen or prim != "cs.LG":
                    continue
                seen.add(aid)
                title = re.sub(r"\s+", " ", e.find("a:title", NS).text).strip()
                abstract = re.sub(r"\s+", " ", e.find("a:summary", NS).text).strip()
                c = e.find("arxiv:comment", NS)
                comment = re.sub(r"\s+", " ", c.text).strip() if c is not None and c.text else ""
                eligible = not EXCLUDE.search(f"{title} {abstract[:400]} {comment}")
                rows.append(dict(arxiv_id=aid, published=e.find("a:published", NS).text[:10], title=title,
                                 comment=comment, eligible=int(eligible)))
            print(f"{lo}: {start + len(entries)} fetched, {len(rows)} primary cs.LG so far", flush=True)
            start += len(entries)
            time.sleep(3.5)                      # arXiv API terms: at most one request every three seconds
    elig = [r for r in rows if r["eligible"]]
    random.Random(SEED).shuffle(elig)
    for i, r in enumerate(elig):
        r["order"] = i
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order", "arxiv_id", "published", "title", "comment", "eligible"])
        w.writeheader()
        w.writerows(elig)
    print(f"frame: {len(rows)} primary cs.LG papers, {len(elig)} eligible -> {OUT}")


if __name__ == "__main__":
    main()
