"""Candidate conversations in WildChat-1M in which a user may ask for an assessment of a scholarly manuscript.

  python wildchat/src/candidates.py  -> wildchat/out/candidates.jsonl

Reads the first user turn of every conversation (data/wildchat/*.parquet, allenai/WildChat-1M, ODC-BY).
A turn is a candidate when its instruction part (first 1,200 and last 600 characters) names a scholarly
manuscript and an evaluative request, in English or Chinese. Exact repeats of the opening 300 characters are
dropped (the same prompt re-sent). The candidates are classified by wildchat/src/classify.py; this filter only
needs high recall.
"""
import glob
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "wildchat"   # allenai/WildChat-1M parquet files
OUT = ROOT / "wildchat" / "out" / "candidates.jsonl"
NOUN = re.compile(r"\b(research paper|paper|manuscript|thesis|dissertation|research article|journal article|"
                  r"conference paper|preprint|abstract|submission)\b|论文|稿件|文稿|手稿|毕业设计|学位论文", re.I)
ASK = re.compile(r"\b(review|evaluate|evaluation|assess|assessment|feedback|critique|criticism|comments?|opinion|"
                 r"what do you think|thoughts|rate|rating|score|grade|strengths?|weakness(es)?|accept(ed|ance)?|"
                 r"reject(ed|ion)?|publishable|publish|ready|chances?|good enough|quality|novelty|referee)\b|"
                 r"评价|评审|审稿|审阅|意见|打分|评分|评估|点评|优缺点|不足|能否发表|录用|质量|创新", re.I)


def main():
    seen, n_all, out = set(), 0, []
    for f in sorted(glob.glob(str(SRC / "train-*.parquet"))):
        t = pq.read_table(f, columns=["conversation_hash", "language", "conversation", "timestamp", "model",
                                      "hashed_ip", "country"])
        for r in t.to_pylist():
            n_all += 1
            u = [m for m in r["conversation"] if m["role"] == "user"]
            if not u:
                continue
            x = u[0]["content"] or ""
            part = x[:1200] + "\n" + x[-600:]
            if not (NOUN.search(part) and ASK.search(part)):
                continue
            key = re.sub(r"\s+", " ", x[:300]).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(h=r["conversation_hash"], lang=r["language"], model=r["model"],
                            ts=str(r["timestamp"])[:10], ip=r["hashed_ip"], country=r["country"],
                            n_chars=len(x), head=x[:2500], tail=x[-1200:] if len(x) > 3700 else ""))
    with open(OUT, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(dict(conversations=n_all, candidates=len(out))))


if __name__ == "__main__":
    main()
