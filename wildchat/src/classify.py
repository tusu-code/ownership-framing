"""Classify WildChat candidates: does the user ask for an assessment of a scholarly manuscript, and which
relation to the manuscript does the user state?

  python wildchat/src/classify.py [--limit N]  -> wildchat/out/classified.jsonl (resumable)

Classifier: DeepSeek V4.1 Flash through the study's gateway, temperature 0, no reasoning, JSON answer. It reads
the opening and closing part of the user's first turn only (never the assistant's reply). Validated against a
hand-labelled random sample (wildchat/src/validate.py).
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "wildchat" / "out" / "candidates.jsonl"
OUT = ROOT / "wildchat" / "out" / "classified.jsonl"
BASE = os.environ.get("WILDCHAT_BASE_URL", "https://api.sailapi.top")   # gateway used in the study
URL = BASE + "/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"
# third classifier, a third model family, run on requests either of the first two calls an assessment (--judge gpt)
GPT = dict(url=URL, model="gpt-6.1-sol", out=ROOT / "wildchat" / "out" / "classified_gpt.jsonl", key="SAIL_GPT_KEY")
# second classifier from another model family, for agreement (--judge haiku)
HAIKU = dict(url=BASE + "/v1/messages", model="claude-haiku-4-5-20251001",
             out=ROOT / "wildchat" / "out" / "classified_haiku.jsonl", key="SAIL_CLAUDE_KEY")

PROMPT = """You label requests that users sent to a chat assistant. Below is the first message a user sent \
(long pasted texts are shortened; [...] marks the cut). Answer with JSON only.

Fields:
"assess": true if the user asks the assistant to judge the quality or merit of a scholarly manuscript \
(research paper, journal or conference manuscript, thesis, dissertation, research proposal, or a section or \
abstract of one): for example to review it, evaluate it, critique it, give feedback or comments on it, list \
its strengths and weaknesses, rate or score it, or say whether it is good, ready, publishable or likely to be \
accepted. false if the user only asks to write, rewrite, polish, paraphrase, shorten, translate, summarize, \
explain, cite or format text, to answer questions, or if the text is not a scholarly manuscript (for example a \
school essay assignment, a story, a cover letter, a news article or a product description).
"relation": the user's stated relation to the manuscript, only from what the message says or clearly implies:
  "own" - the user or the user's team wrote it (e.g. "my paper", "our manuscript", "I wrote", "my thesis", \
"I am going to submit");
  "reviewer" - the user is reviewing or refereeing it for a journal, conference, editor or committee;
  "other" - someone else wrote it (a colleague, friend, student, or a published paper by other authors) and \
the user is not reviewing it;
  "none" - no relation is stated or implied.
"role_play": true if the user asks the assistant to act as a reviewer, referee, editor or examiner, or to \
write a peer review; otherwise false.
"evidence": up to 20 words quoted from the message that decide "relation" (empty string for "none").

Message:
<<<
{text}
>>>

JSON:"""


def text_of(r):
    t = r["head"]
    if r["tail"]:
        t += "\n[...]\n" + r["tail"]
    return t


def call(r, key, judge="dsflash"):
    msg = [{"role": "user", "content": PROMPT.replace("{text}", text_of(r))}]
    if judge == "haiku":
        url, body = HAIKU["url"], {"model": HAIKU["model"], "temperature": 0, "max_tokens": 200, "messages": msg}
        hdr = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    elif judge == "gpt":
        url, body = GPT["url"], {"model": GPT["model"], "max_completion_tokens": 4000, "store": False, "messages": msg}
        hdr = {"Authorization": f"Bearer {key}"}
    else:
        url, body = URL, {"model": MODEL, "temperature": 0, "max_tokens": 200, "reasoning_effort": "none", "messages": msg}
        hdr = {"Authorization": f"Bearer {key}"}
    for attempt in range(5):
        try:
            res = requests.post(url, headers=hdr, json=body, timeout=120)
            res.raise_for_status()
            js = res.json()
            c = js["content"][0]["text"] if judge == "haiku" else js["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", c, re.S)
            j = json.loads(m.group(0))
            assert isinstance(j.get("assess"), bool) and j.get("relation") in ("own", "reviewer", "other", "none")
            return dict(h=r["h"], ok=True, **{k: j.get(k) for k in ("assess", "relation", "role_play", "evidence")},
                        model_id=js.get("model"))
        except Exception as e:  # noqa: BLE001
            err = str(e)[:200]
            time.sleep(3 * (attempt + 1))
    return dict(h=r["h"], ok=False, error=err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--judge", choices=["dsflash", "haiku", "gpt"], default="dsflash")
    ap.add_argument("--only-union", action="store_true", help="only requests either first classifier calls an assessment")
    a = ap.parse_args()
    global OUT
    if a.judge in ("haiku", "gpt"):
        OUT = (HAIKU if a.judge == "haiku" else GPT)["out"]
    key = os.environ[{"haiku": HAIKU["key"], "gpt": GPT["key"]}.get(a.judge, "SAIL_DEEPSEEK_KEY")]
    rows = [json.loads(l) for l in open(IN)]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["h"] for l in open(OUT) if json.loads(l).get("ok")}
    todo = [r for r in rows if r["h"] not in done]
    if a.only_union:
        yes = set()
        for f in ("classified.jsonl", "classified_haiku.jsonl"):
            yes |= {j["h"] for j in map(json.loads, open(ROOT / "wildchat" / "out" / f)) if j.get("ok") and j.get("assess")}
        todo = [r for r in todo if r["h"] in yes]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(rows)} candidates, {len(done)} done, {len(todo)} to classify", flush=True)
    with open(OUT, "a") as fh, cf.ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(lambda r: call(r, key, a.judge), todo)):
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 200 == 0:
                print(i, time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
