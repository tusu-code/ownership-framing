"""Confirmatory study: build cleaned manuscripts in frame order until enough pass the material checks.

  python confirm/src/materials.py convert --first 280   -> confirm/data/converted.jsonl (PDFs already in confirm/data/pdfs)
  python confirm/src/materials.py select --need 166     -> confirm/data/manuscripts.jsonl, confirm/data/material_log.csv

Papers are taken strictly in the random order of confirm/data/frame.csv. For each: download the arXiv PDF of
the first version, convert to Markdown, cut references and appendices (convert, cached as raw Markdown), then
clean (select): drop everything before the abstract, apply the main study's rules (src/prepare.py: e-mails,
acknowledgements, ethics and similar statements, URLs, venue strings) and remove any line that names an author
in full, using the arXiv author list (clean_confirm). The title shown is the arXiv title. A paper passes if it
has an abstract, at least MIN_TOKENS tokens and is mostly English text; longer texts are truncated to MAX_TOKENS
as in the main study.
Every decision is logged. The first 6 passing papers are the smoke-test set; the next 160 are the sample.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

# One thread per process on a shared server: the layout model's onnxruntime sessions otherwise start a thread per
# core regardless of OMP_NUM_THREADS, so a few workers can occupy the whole machine.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import onnxruntime as _ort  # noqa: E402

_Session = _ort.InferenceSession


def _single_thread_session(path, sess_options=None, *a, **k):
    so = sess_options or _ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = 1, 1
    so.execution_mode = _ort.ExecutionMode.ORT_SEQUENTIAL
    return _Session(path, so, *a, **k)


_ort.InferenceSession = _single_thread_session

import pymupdf4llm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prepare import ENC, clean, truncate  # noqa: E402

C = ROOT / "confirm" / "data"
MIN_TOKENS, MAX_TOKENS = 4000, 16000
CUT = re.compile(r"^#{1,6}\s*(?:\**)?\s*(?:[A-Z]?\d*[.\s]*)?(references|bibliography|appendix|appendices|supplementary material)\b", re.I | re.M)
ARXIV = re.compile(r"arXiv:\s?\d{4}\.\d{4,5}(v\d+)?(\s*\[[^\]]*\])?(\s*\d{1,2}\s\w{3}\s\d{4})?", re.I)
VENUE_NAMES = r"(NeurIPS|ICML|ICLR|AAAI|IJCAI|CVPR|ICCV|ECCV|ACL|EMNLP|NAACL|COLING|KDD|WWW|AISTATS|UAI|COLT|TMLR|JMLR|TPAMI|conference|workshop|symposium|journal|transactions)"
# Only statements about where the paper appears; the venue word must follow within a few words.
VENUE2 = re.compile(r"\b(Accepted|Published|Presented|To appear|Under review)\s+(at|in|to|as a [\w ]{0,20}paper at)\s+(the\s+)?[^\n.]{0,60}?" + VENUE_NAMES + r"[^\n.]{0,80}|"
                    r"\bProceedings of the[^\n.]{0,120}|\b(NeurIPS|ICML|ICLR|AAAI|IJCAI|CVPR|ICCV|ECCV|ACL|EMNLP|NAACL|KDD|AISTATS|UAI|COLT|TMLR|JMLR)\s?20\d\d\b|"
                    r"^\s*Preprint\.?(\s*Under review\.?)?\s*$|\bCopyright\s+(©\s*)?\d{4}[^\n]*", re.I | re.M)


def download(aid, path):
    if path.exists() and path.stat().st_size > 10000:
        return True
    url = f"https://arxiv.org/pdf/{aid}v1"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ownership-framing-study/1.0 (mailto:anonymous@example.com)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if data[:4] == b"%PDF":
                path.write_bytes(data)
                return True
            return False
        except Exception:  # noqa: BLE001
            time.sleep(8 * (attempt + 1))
    return False


def to_text(pdf):
    md = pymupdf4llm.to_markdown(str(pdf), show_progress=False, use_ocr=False, header=False, footer=False)
    # Bold markers inside headings hide the title and the abstract heading from the main study's cleaner.
    md = re.sub(r"^(#{1,6})\s*\*\*(.*?)\*\*\s*$", r"\1 \2", md, flags=re.M)
    md = re.sub(r"^\s*\d{1,3}\s*$", "", md, flags=re.M)          # stray page numbers
    m = CUT.search(md, pos=min(len(md) // 4, 20000))        # first references/appendix heading after the first quarter
    md = md[:m.start()] if m else md
    md = ARXIV.sub("", md)
    md = VENUE2.sub("", md)
    return md


def check(title, text):
    n = len(ENC.encode(text))
    ascii_share = sum(ch.isascii() for ch in text) / max(1, len(text))
    if not title:
        return False, "no title", n
    if not text.startswith("## Abstract"):
        return False, "no abstract", n
    if n < MIN_TOKENS:
        return False, f"too short ({n} tokens)", n
    if ascii_share < 0.9:
        return False, "not mainly English text", n
    return True, "ok", n


def convert_one(aid):
    """Convert one PDF to raw Markdown (references and appendices already cut). Cleaning is a separate step,
    so the cleaning rules can change without converting again."""
    pdf = C / "pdfs" / f"{aid}.pdf"
    if not (pdf.exists() and pdf.stat().st_size > 10000):
        return {"arxiv_id": aid, "status": "download failed"}
    try:
        return {"arxiv_id": aid, "status": "converted", "md": to_text(pdf)}
    except Exception as e:  # noqa: BLE001
        return {"arxiv_id": aid, "status": f"conversion failed: {str(e)[:60]}"}


# Statements that name or identify the authors, beyond the main study's DROP_SECTIONS.
DROP2 = re.compile(r"credit|authorship|author contribution|contribution statement|competing interest|conflicts? of interest|"
                   r"declaration of|data availability|code availability|funding|acknowledg", re.I)
ABSTRACT_MARK = re.compile(r"^\W{0,6}abstract\b\W*", re.I)
# Publication and copyright notices: the whole line goes, whatever its length.
PUB_NOTICE = re.compile(r"accepted manuscript|version of record|author'?s'? (final|accepted) version|"
                        r"has been (submitted|accepted) (to|for|at)|permission to make digital|"
                        r"personal use of this material is permitted|copyrights? (for|held)|©\s*\d{4}|"
                        r"\bcopyright\s+(©\s*)?\d{4}|licensed under (a |the )?creative commons|\borcid\b", re.I)
# Running footers and preprint stamps (ICML/NeurIPS/IEEE templates), matched on the line without Markdown marks.
FOOTER_LINE = re.compile(r"proceedings of the \d+\w*\s|proceedings of the [\w ,]{0,60}(conference|workshop|symposium)|"
                         r"\bPMLR\s+\d+|^\W*preprint\b|copyright \d{4} by the author|conference on neural information "
                         r"processing systems \(neurips|^\W*(\d+(st|nd|rd|th)\s+)?(annual\s+)?(international\s+)?"
                         r"(conference|workshop|symposium) on\b", re.I)
# A heading-like line that starts the references or appendices, without '#' (bold or plain), for papers the
# first cut in to_text missed.
LOOSE_CUT = re.compile(r"^\W*(?:[A-Z]|\d+)?\W*(references|bibliography|appendix|appendices|supplementary materials?)\W*$", re.I)
AI_NOTE = re.compile(r"\b(AI|LLM|language model|generative)[- ](tools?|assist\w*|usage|use)\b|use of (AI|LLMs?)", re.I)
CODE_HOST = re.compile(r"\b(github\.com|gitlab\.com|huggingface\.co|anonymous\.4open\.science|zenodo\.org)/\S*", re.I)


def name_variants(name):
    """Full name and initial forms ("Erhan Bayraktar", "E. Bayraktar"); surnames alone are left, as citations."""
    parts = name.replace(",", " ").split()
    if len(parts) < 2:
        return []
    first, last = parts[0], parts[-1]
    return [name, f"{first} {last}", f"{first[0]}. {last}", f"{first[0]}.{last}"]


def clean_confirm(md, title, authors):
    """Return (body, status). Everything before the abstract is dropped (authors, affiliations, dates); the
    abstract may be a heading or an inline label ("**Abstract.** ...", IEEE "Abstract—"). The title shown to the
    model is the arXiv title. Then the main study's clean() runs, plus: author-identifying statements, any line
    naming an author in full, code-hosting links, ORCID and copyright lines."""
    lines = md.splitlines()
    start = None
    for i, l in enumerate(lines[: max(60, len(lines) // 6)]):
        s = re.sub(r"^#{1,6}\s*", "", l).replace("*", "").replace("_", "").strip()
        if ABSTRACT_MARK.match(s):
            rest = ABSTRACT_MARK.sub("", s, count=1).lstrip("—–-:. ").strip()
            lines = ["## Abstract", ""] + ([rest] if rest else []) + lines[i + 1:]
            start = i
            break
    if start is None:
        return "", "no abstract"
    _, body = clean(f"# {title}\n\n" + "\n".join(lines))
    kept, skip = [], None
    for l in body.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", l)
        if m:
            lvl = len(m.group(1))
            if skip is not None and lvl <= skip:
                skip = None
            if skip is None and (DROP2.search(m.group(2)) or AI_NOTE.search(m.group(2))):
                skip = lvl
                continue
        if skip is None:
            kept.append(l)
    plain = lambda s: re.sub(r"</?su[pb]>|[_*`]", "", s)
    for i in range(len(kept) // 4, len(kept)):                 # references or appendix the first cut missed
        if LOOSE_CUT.match(plain(kept[i])):
            kept = kept[:i]
            break
    variants = sorted({v for a in authors for v in name_variants(a)}, key=len, reverse=True)
    out = []
    for l in kept:
        if FOOTER_LINE.search(plain(l)) and len(l) < 300:
            continue                                            # template footer or preprint stamp
        if re.match(r"^\s*\**\s*(" + DROP2.pattern + r")[^*\n]{0,60}\**\s*[.:—-]", l, re.I):
            continue                                            # paragraph-style statement
        if PUB_NOTICE.search(l):
            continue
        if re.match(r"^\s*\**\s*[^*\n]{0,40}" + AI_NOTE.pattern + r"[^*\n]{0,40}\**\s*[.:—-]", l, re.I):
            continue                                            # paragraph-style AI-use statement
        if re.search(r"(?i)e-?mail addresse?s?\b|\bcorresponding author", plain(l)) and len(l) < 400:
            continue
        hit = [v for v in variants if v in l or v in re.sub(r"\s+", " ", plain(l))]
        if hit and len(l) < 400:
            continue                                            # bylines, footnotes, contribution lines
        for v in hit:
            l = l.replace(v, "the authors")
        l = CODE_HOST.sub("[link removed]", l)
        # Links broken by the PDF layout ("https: //host/path", "https:// [link removed]").
        l = re.sub(r"`?https?:\s*//\s*(\[link removed\]|[^\s`)\]]*)`?", "[link removed]", l)
        # The path left after a link that was split at a line break ("[link removed] user-name/repo").
        l = re.sub(r"\[link removed\]\s*`?[\w.~-]*(/[\w.~%#?=&+-]+)+/?`?", "[link removed]", l)
        out.append(l)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip(), "ok"


def convert(ids, workers):
    """Stage 1: convert every downloaded PDF (PDFs are fetched on another machine into confirm/data/pdfs)."""
    from concurrent.futures import ProcessPoolExecutor
    cache_path = C / "converted.jsonl"
    done = {json.loads(l)["arxiv_id"] for l in open(cache_path)} if cache_path.exists() else set()
    todo = [a for a in ids if a not in done and (C / "pdfs" / f"{a}.pdf").exists()]
    with open(cache_path, "a") as f, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(convert_one, todo):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(f"{r['arxiv_id']} {r['status']}", flush=True)


def select(need):
    """Stage 2: walk the frame in its random order; the first `need` passing papers are used
    (the first 6 as the smoke-test set, the rest as the sample). Stops at the first paper not yet converted."""
    frame = sorted(csv.DictReader(open(C / "frame.csv")), key=lambda x: int(x["order"]))
    conv = {}
    for l in open(C / "converted.jsonl"):
        r = json.loads(l); conv[r["arxiv_id"]] = r
    rows, docs, n_ok = [], [], 0
    authors = json.load(open(C / "arxiv_authors.json"))
    for r in frame:
        if n_ok >= need:
            break
        c = conv.get(r["arxiv_id"])
        if c is None:
            print(f"stopped at order {r['order']} ({r['arxiv_id']}): not converted yet"); break
        role, status, tokens, trunc = "", c["status"], 0, False
        if status == "converted":
            body, status = clean_confirm(c["md"], r["title"], authors.get(r["arxiv_id"], []))
            if status == "ok":
                ok, status, tokens = check(r["title"], body)
                if ok:
                    body, tokens, trunc = truncate(body, MAX_TOKENS)
                    role = "smoke" if n_ok < 6 else "sample"; n_ok += 1
                    docs.append({"paper_id": r["arxiv_id"], "title": r["title"], "text": body, "n_tokens": tokens,
                                 "truncated": trunc, "role": role, "order": int(r["order"])})
        rows.append(dict(order=r["order"], arxiv_id=r["arxiv_id"], status=status, tokens=tokens,
                         truncated=int(trunc), role=role))
    with open(C / "material_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(C / "manuscripts.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{n_ok} passing papers out of {len(rows)} examined")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "select"])
    ap.add_argument("--first", type=int, default=280, help="convert the first N papers of the frame")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--need", type=int, default=166)
    args = ap.parse_args()
    if args.cmd == "convert":
        frame = sorted(csv.DictReader(open(C / "frame.csv")), key=lambda x: int(x["order"]))
        convert([r["arxiv_id"] for r in frame[:args.first]], args.workers)
    else:
        select(args.need)


if __name__ == "__main__":
    main()
