"""Build de-identified manuscripts and the stratified pilot sample.

Input : data/raw/papers/*.txt (ICLR 2026 full text, references/appendix already cut),
        data/raw/meta_396.csv (decision, human ratings, authors, primary area).
Output: ownership_pilot/data/manuscripts.jsonl   all cleaned papers
        ownership_pilot/data/leak_audit.csv      residual identity/venue cues per paper
        ownership_pilot/data/sample_pilot.csv    the stratified pilot sample

Accepted papers in the source are camera-ready (named) and rejected ones are the
anonymous submission, so cleaning must make both look identical in format;
otherwise the format alone reveals the decision.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import tiktoken

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "ownership_pilot" / "data"
ENC = tiktoken.get_encoding("o200k_base")

DROP_SECTIONS = re.compile(
    r"acknowledg|ethic|reproducib|llm usage|use of (large )?language model|use of llm|"
    r"author contribution|funding|disclosure|impact statement|broader impact|^references$|^bibliography$",
    re.I,
)
MD_LINK = re.compile(r"\[[^\]]*\]\((https?:)?//[^)]*\)|\[https?://\]\([^)]*\)")
URL = re.compile(r"<?https?://[^\s>)\]]+>?")
FOOTNOTE = re.compile(r"(?i)correspondence|corresponding author|work done (during|while)|equal contribution|internship at|project lead")
EMAIL = re.compile(r"\[?[\w.+-]+@[\w-]+(\.[\w-]+)+\]?(\(mailto:[^)]*\))?")
PAGE_MARK = re.compile(r"^\{\d+\}-+\s*$")
IMAGE = re.compile(r"^!\[.*?\]\(.*?\)\s*$")
VENUE = re.compile(
    r"Published as a conference paper at ICLR 20\d\d|Under review as a conference paper at ICLR 20\d\d|"
    r"Paper under double-blind review|\*\*Anonymous authors\*\*|Anonymous authors|camera[- ]ready",
    re.I,
)


def heading_level(line):
    m = re.match(r"^(#{1,6})\s+(.*)$", line)
    return (len(m.group(1)), m.group(2)) if m else (None, None)


def clean(text):
    lines = [l for l in text.splitlines() if not PAGE_MARK.match(l) and not IMAGE.match(l)]
    lines = [l for l in lines if "Rest of paper (reference and Appendix) is removed" not in l]

    # Title = first level-1 heading; drop everything between it and the abstract heading
    # (author names, affiliations, emails, "Anonymous authors").
    title, body_start = None, 0
    for i, l in enumerate(lines):
        lvl, h = heading_level(l)
        if lvl == 1 and title is None:
            title = h.strip()
            continue
        if lvl and re.match(r"(?i)abstract", h.strip()):
            body_start = i
            break
    body = lines[body_start:] if body_start else lines

    # Drop non-scientific end sections (acknowledgements, ethics, reproducibility, LLM use ...).
    kept, skip_level = [], None
    for l in body:
        lvl, h = heading_level(l)
        if lvl:
            if skip_level is not None and lvl <= skip_level:
                skip_level = None
            if skip_level is None and DROP_SECTIONS.search(re.sub(r"^[\dA-Z.]*\s+", "", h.strip()).strip(" .:*")):
                skip_level = lvl
                continue
        if skip_level is None:
            kept.append(l)
    # Paragraph-style statements ("**Reproducibility Statement** We ...") are not headings.
    kept = [l for l in kept if not re.match(r"^\s*\*\*[^*]*(" + DROP_SECTIONS.pattern + r")[^*]*\*\*", l, re.I)]

    kept = [l for l in kept if not (FOOTNOTE.search(l) and len(l) < 400)]
    out = "\n".join(kept)
    out = MD_LINK.sub("[link removed]", out)
    out = URL.sub("[link removed]", out)
    out = EMAIL.sub("", out)
    out = VENUE.sub("", out)
    out = re.sub(r"ICLR\s?2026", "", out, flags=re.I)
    out = re.sub(r"<sup>[^<]{0,12}</sup>", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return (title or "").strip(), out


def leak_cues(text, authors):
    cues = {
        "email": bool(re.search(r"[\w.+-]+@[\w-]+\.\w+", text)),
        "venue_phrase": bool(VENUE.search(text)),
        "iclr26": bool(re.search(r"ICLR\s?2026", text, re.I)),
        "url": "http" in text,
        "ack_word": bool(re.search(r"(?i)\bwe (would like to )?thank\b|acknowledg", text)),
    }
    # Author full names in the text (surname-only hits are mostly citations and are expected).
    names = [a.strip() for a in str(authors).split(",") if len(a.strip().split()) >= 2]
    cues["author_fullname"] = sum(1 for n in names if n in text)
    return cues


def truncate(text, max_tokens):
    ids = ENC.encode(text)
    if len(ids) <= max_tokens:
        return text, len(ids), False
    return ENC.decode(ids[:max_tokens]) + "\n\n[Manuscript truncated for length.]", max_tokens, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--n-border", type=int, default=24, help="borderline papers, half accept half reject")
    ap.add_argument("--n-high", type=int, default=8, help="clearly accepted papers")
    ap.add_argument("--n-low", type=int, default=8, help="clearly rejected papers")
    ap.add_argument("--seed", type=int, default=20261025)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(RAW / "meta_396.csv", index_col="paper_id")
    recs, audit = [], []
    for pid, m in meta.iterrows():
        raw = (RAW / "papers" / f"{pid}.txt").read_text(errors="ignore")
        title, body = clean(raw)
        text, ntok, trunc = truncate(body, args.max_tokens)
        cues = leak_cues(text, m.authors)
        accept = str(m.decision).startswith("Accept")
        recs.append(dict(paper_id=pid, title=title, text=text, n_tokens=ntok, truncated=trunc))
        audit.append(dict(paper_id=pid, decision=m.decision, accept=accept, human_mean=m.human_mean,
                          n_tokens=ntok, truncated=trunc, raw_chars=len(raw), **cues))

    with open(OUT / "manuscripts.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    audit = pd.DataFrame(audit).set_index("paper_id")
    audit.to_csv(OUT / "leak_audit.csv")

    cue_cols = ["email", "venue_phrase", "iclr26", "url", "ack_word"]
    print("residual cues by decision (share of papers):")
    print(audit.groupby("accept")[cue_cols + ["truncated"]].mean().round(3))
    print(audit.groupby("accept")["author_fullname"].apply(lambda s: (s > 0).mean()).round(3))
    print(audit.groupby("accept")["n_tokens"].median())

    # Pilot sample: exclude withdrawn papers, papers with fewer than 3 human reviews and
    # papers that still carry an author full name or a venue phrase after cleaning.
    pool = audit.join(meta[["n_reviews", "primary_area", "title"]])
    pool = pool[~pool.decision.str.contains("Withdrawn") & (pool.n_reviews >= 3)]
    pool = pool[(pool.author_fullname == 0) & ~pool.venue_phrase & ~pool.email]
    # ICLR 2026 scale is 0/2/4/6/8/10; the accept/reject boundary sits between 4 and 6.
    border = pool[(pool.human_mean >= 4) & (pool.human_mean < 6)]
    picks = [
        border[border.accept].sample(args.n_border // 2, random_state=args.seed).assign(stratum="border_accept"),
        border[~border.accept].sample(args.n_border // 2, random_state=args.seed).assign(stratum="border_reject"),
        pool[pool.accept & (pool.human_mean >= 6)].sample(args.n_high, random_state=args.seed).assign(stratum="high_accept"),
        pool[~pool.accept & (pool.human_mean < 4)].sample(args.n_low, random_state=args.seed).assign(stratum="low_reject"),
    ]
    sample = pd.concat(picks)
    sample[["stratum", "decision", "accept", "human_mean", "n_reviews", "n_tokens", "truncated", "primary_area", "title"]].to_csv(OUT / "sample_pilot.csv")
    print(f"\neligible pool {len(pool)}; borderline {len(border)} "
          f"(accept {int(border.accept.sum())}, reject {int((~border.accept).sum())})")
    print(sample.groupby("stratum").agg(n=("accept", "size"), human_mean=("human_mean", "mean"), tokens=("n_tokens", "median")))


if __name__ == "__main__":
    main()
