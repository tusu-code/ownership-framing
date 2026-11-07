"""Confirmatory study: residual identity and publication cues in the cleaned manuscripts.

  python confirm/src/leak_audit.py   -> confirm/data/leak_audit.csv, summary on stdout

Counts per paper: author full names and initial forms (arXiv author list), e-mails, links, code hosts, venue
with year, ORCID/copyright, arXiv identifiers. Surnames alone are not counted (they are mostly citations).
A sample paper with any full-name, e-mail or venue-year hit is reported; such papers must be fixed or excluded
before the materials are frozen.
"""
import csv
import json
import re
import sys
from pathlib import Path

C = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from materials import name_variants  # noqa: E402

CUES = {"email": r"[\w.+-]+@[\w-]+\.[\w.]+", "link": r"https?://|www\.",
        "code_host": r"github\.com|gitlab\.com|huggingface\.co/|zenodo\.org",
        "venue_year": r"\b(NeurIPS|ICML|ICLR|AAAI|IJCAI|CVPR|ICCV|ECCV|KDD|AISTATS|UAI|COLT|TMLR|JMLR|ACL|EMNLP)\s?'?20\d\d\b",
        "orcid_copyright": r"orcid|copyright\s+(©\s*)?\d{4}|©|permission to make digital|personal use of this material",
        "pub_notice": r"accepted manuscript|version of record|has been accepted (to|for|at)|camera[- ]ready|\bto appear in (the )?proceedings",
        "ai_note": r"\b(AI|LLM)[- ](tools?|assist\w*)\b|use of (AI|LLMs?)\b",
        "arxiv_id": r"arXiv:\s?\d{4}\.\d{4,5}|\b26\d{2}\.\d{5}\b",
        # Found in the smoke test (2026-11-04): a repository path left after a split link, and template footers.
        "link_path": r"\[link removed\]\s*`?[\w.~-]*/[\w.~-]+",
        "footer": r"proceedings of the \d+|\bPMLR\s+\d+|^\W*preprint\b|copyright \d{4} by the author",
        "ref_list": r"^\W*\[\d+\]\s+[A-Z][\w.-]*\s?[A-Z]?\.?,? [A-Z]"}
BLOCKING = ("names", "email", "venue_year", "orcid_copyright", "pub_notice", "link_path", "footer")


def main():
    docs = [json.loads(l) for l in open(C / "manuscripts.jsonl")]
    authors = json.load(open(C / "arxiv_authors.json"))
    rows = []
    for d in docs:
        t = f"{d['title']}\n{d['text']}"
        t = re.sub(r"</?su[pb]>|[_*`]", "", t)                 # match on the text without Markdown marks
        names = sorted({v for a in authors.get(d["paper_id"], []) for v in name_variants(a) if v in t})
        r = {"paper_id": d["paper_id"], "role": d["role"], "names": len(names), "name_hits": "; ".join(names[:5])}
        r.update({k: len(re.findall(p, t, re.I | re.M)) for k, p in CUES.items()})
        rows.append(r)
    with open(C / "leak_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} papers; papers with any hit:")
    for k in ["names", *CUES]:
        print(f"  {k:16s} {sum(r[k] > 0 for r in rows)}")
    bad = [r for r in rows if any(r[k] for k in BLOCKING)]
    print(f"blocking hits: {len(bad)}")
    for r in bad:
        print("  ", r["paper_id"], r["role"], {k: r[k] for k in BLOCKING if r[k]}, r["name_hits"])


if __name__ == "__main__":
    main()
