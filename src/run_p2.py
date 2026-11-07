"""Pilot 2: natural asks on disguised manuscripts, then blind extraction.

  python src/run_p2.py review [--model gpt]   # 40 papers x 4 conditions x p2_reps per model
  python src/run_p2.py judge                  # blind extraction of every answer (judge_model)
  python src/run_p2.py status

Conditions (src/prompts.py P2_CONDITIONS): open_author, open_reviewer, struct_author,
struct_reviewer. Each call is a fresh one-message conversation with a random end-user id.
Results: results/p2_review_<model>.jsonl and results/p2_judge.jsonl; reruns skip finished keys.
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import JUDGE, NEUTRALIZE, P2_CONDITIONS  # noqa: E402
import re  # noqa: E402
from run import Client, load_config, parse_json, valid_review  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
EVAL_MODELS = ["gpt", "glm", "deepseek", "claude"]
JUDGE_PATH = RES / "p2_judge.jsonl"
NEUTRAL_PATH = RES / "p2_neutral.jsonl"
JUDGE_NEUTRAL_PATH = RES / "p2_judge_neutral.jsonl"
# Second-person or asker-directed wording that must not survive neutralization.
ASKER_CUES = re.compile(r"(?i)\b(you|your|yours|yourself|i'd|i would|i think|happy to|good luck|hope this helps|as a reviewer|as the author|before you submit)\b")


def review_path(model):
    return RES / f"p2_review_{model}.jsonl"


def read(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def valid_judge(obj):
    try:
        return (obj is not None and int(obj["implied_rating"]) in (0, 2, 4, 6, 8, 10)
                and 1 <= int(obj["overall_tone"]) <= 7 and 1 <= int(obj["readiness"]) <= 5
                and str(obj["implied_decision"]).lower() in ("accept", "reject", "unclear"))
    except (KeyError, TypeError, ValueError):
        return False


async def call_with_retry(client, messages, max_tokens, check, retries=5):
    """Return (record_fields, ok). Retries transport errors and malformed answers."""
    last = {}
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            out = await client.chat(messages, max_tokens)
            ok, parsed = check(out)
            last = {"ok": ok, "parsed": parsed, "raw": out["text"], "usage": out["usage"],
                    "provider_model": out.get("model"), "finish_reason": out.get("finish_reason"),
                    "latency_s": round(time.time() - t0, 1), "attempt": attempt,
                    "key_slot": out.get("key_slot"), "reasoning": out.get("reasoning")}
            if ok:
                return last
        except Exception as e:  # noqa: BLE001
            status = getattr(getattr(e, "response", None), "status_code", None)
            last = {"ok": False, "error": str(e)[:300], "attempt": attempt}
            if status in (400, 401, 403, 404) or type(e).__name__ == "OversizedInput":
                return last
        await asyncio.sleep(min(60, 3 * 2 ** attempt) + random.random() * 3)
    return last


FATAL_MARKERS = ("insufficient balance", "insufficient_balance", "model_not_found", "invalid token", "api_key_required")


class FatalGuard:
    """Stops a pass after a few calls fail with an error that retrying cannot fix (no balance, model gone),
    so an unattended loop does not spin through thousands of failed requests."""

    def __init__(self, name, limit=3):
        self.name, self.limit, self.hits, self.reason = name, limit, 0, None

    @property
    def stopped(self):
        return self.reason is not None

    def see(self, rec):
        err = str(rec.get("error") or "").lower()
        if any(m in err for m in FATAL_MARKERS):
            self.hits += 1
            if self.hits >= self.limit and not self.reason:
                self.reason = err[:160]
                print(f"[{self.name}] stopping this pass after {self.hits} unrecoverable errors: {self.reason}", flush=True)
        else:
            self.hits = 0


def searched(text):
    """Built-in web search leaves utm_source=openai on the links it cites."""
    return "utm_source=openai" in (text or "")


class OversizedInput(Exception):
    pass


MAX_EXTRA_PROMPT = 50000   # relay overhead is ~4-22k; far more means foreign context was prepended


def review_check(kind, sent_tokens=None):
    def check(out):
        if searched(out["text"]):
            return False, None
        if "<tool_call>" in (out["text"] or ""):
            # The relay hands Claude tool descriptions it cannot run; such answers are incomplete.
            return False, None
        pt = int((out.get("usage") or {}).get("prompt_tokens") or 0)
        if sent_tokens and pt - sent_tokens > MAX_EXTRA_PROMPT:
            # The relay repeats the same oversized prefix for the same request; retrying only adds cost.
            raise OversizedInput(f"relay prompt {pt} vs sent {sent_tokens}")
        if kind == "struct":
            obj = parse_json(out["text"])
            return valid_review(obj), obj
        # Open answers: any real text that did not run out of budget.
        text = (out["text"] or "").strip()
        return len(text) > 200 and out.get("finish_reason") != "length", None
    return check


def load_docs():
    """Disguised manuscripts: the frozen pilot 40 plus every new paper whose latest rewrite is clean."""
    docs = {r["paper_id"]: r for r in map(json.loads, open(DATA / "manuscripts_disguised.jsonl"))}
    new = DATA / "manuscripts_disguised_new.jsonl"
    if new.exists():
        latest = {}
        for r in map(json.loads, open(new)):
            latest[r["paper_id"]] = r
        named = {r["paper_id"] for r in read(RES / "disguise_cache.jsonl") if r.get("kind") == "names"}
        docs.update({pid: r for pid, r in latest.items()
                     if r["chunks_fallback"] == 0 and pid in named and pid not in docs})
    return docs


async def run_reviews(cfg, models, full=False, conds=None, retest=None, retest_only=False):
    sample = pd.read_csv(DATA / cfg["p3_sample_file" if full else "sample_file"], index_col="paper_id")
    docs = load_docs()
    reps = int(cfg["p3_reps" if full else "p2_reps"])
    missing = [pid for pid in sample.index if pid not in docs]
    sample = sample.drop(index=missing)
    if missing:
        print(f"{len(missing)} papers not yet disguised cleanly; skipped this run", flush=True)
    RES.mkdir(exist_ok=True)

    async def one_model(name):
        client = Client(name, cfg["models"][name], cfg.get("defaults", {}))
        excluded = {tuple(k[1:]) for k in load_excluded() if k[0] == name}
        give_up = {tuple(k[1:]) for k in load_give_up() if k[0] == name}
        relay = client.c["base_url"]
        def counts(r):
            # Records without a relay field come from rehdasu.cn. The full study is done entirely on the
            # current relay for every model, pilot papers included (decided 2026-10-26).
            return r.get("relay", OLD_RELAY) == relay

        done = {(r["paper_id"], r["condition"], r["rep"]) for r in read(review_path(name))
                if r.get("ok") and not r.get("excluded") and counts(r)} - excluded
        jobs = [] if retest_only else [(pid, cond, rep) for pid in sample.index for cond in (conds or cfg["p2_conditions"]) for rep in range(reps)
                if (pid, cond, rep) not in done and (pid, cond, rep) not in give_up]
        if retest:
            # 2026-11-01: same-batch retest (rep 1) of selected asks on a paper subset, shuffled in with the
            # other jobs so both sets run in the same time window.
            ids, rconds, rrep = retest
            jobs += [(pid, cond, rrep) for pid in ids if pid in sample.index for cond in rconds
                     if (pid, cond, rrep) not in done and (pid, cond, rrep) not in give_up]
        random.Random(f"{cfg['seed']}-{name}").shuffle(jobs)
        print(f"[{name}] {len(jobs)} calls ({len(done)} done)", flush=True)
        stats = {"ok": 0, "bad": 0}
        guard = FatalGuard(name)
        with open(review_path(name), "a") as f:
            async def job(pid, cond, rep):
                build, kind = P2_CONDITIONS[cond]
                d = docs[pid]
                # A model may raise its own ceiling; it only changes answers that would otherwise be cut off.
                key = "max_tokens_review" if kind == "struct" else "max_tokens_open"
                max_tokens = int(client.c.get(key, cfg[key]))
                msgs = build(d["title"], d["text"])
                sent = len(ENC.encode(msgs[0]["content"]))
                async with client.sem:
                    if guard.stopped:
                        return
                    rec = await call_with_retry(client, msgs, max_tokens, review_check(kind, sent))
                guard.see(rec)
                f.write(json.dumps({"paper_id": pid, "condition": cond, "rep": rep, "kind": kind, "model": name,
                                    "model_id": client.c["model"], "relay": client.c["base_url"],
                                    "ts": time.strftime("%H:%M:%S"), **rec},
                                   ensure_ascii=False) + "\n")
                f.flush()
                stats["ok" if rec.get("ok") else "bad"] += 1
                n = stats["ok"] + stats["bad"]
                if n % 20 == 0 or n == len(jobs):
                    print(f"[{name}] {n}/{len(jobs)} ok={stats['ok']} bad={stats['bad']}", flush=True)
            await asyncio.gather(*[job(*j) for j in jobs])
        await client.http.aclose()

    await asyncio.gather(*[one_model(m) for m in models])


async def run_judge(cfg):
    name = cfg["judge_model"]
    client = Client(name, cfg["models"][name], cfg.get("defaults", {}))
    done = {(r["model"], r["paper_id"], r["condition"], r["rep"]) for r in read(JUDGE_PATH) if r.get("ok")}
    answers = []
    for m in EVAL_MODELS:
        latest = {}
        for r in read(review_path(m)):
            if r.get("ok"):
                latest[(m, r["paper_id"], r["condition"], r["rep"])] = r
        answers += [(k, v["raw"]) for k, v in latest.items() if k not in done]
    random.Random(cfg["seed"]).shuffle(answers)
    print(f"[judge:{client.c['model']}] {len(answers)} answers to extract ({len(done)} done)", flush=True)
    stats = {"ok": 0, "bad": 0}

    def check(out):
        obj = parse_json(out["text"])
        return valid_judge(obj), obj

    guard = FatalGuard("judge_path")
    with open(JUDGE_PATH, "a") as f:
        async def job(key, raw):
            async with client.sem:
                if guard.stopped:
                    return
                rec = await call_with_retry(client, [{"role": "user", "content": JUDGE.format(response=raw)}], 3000, check)
            guard.see(rec)
            m, pid, cond, rep = key
            f.write(json.dumps({"model": m, "paper_id": pid, "condition": cond, "rep": rep,
                                "judge_model": client.c["model"], **{k: v for k, v in rec.items() if k != "raw"}},
                               ensure_ascii=False) + "\n")
            f.flush()
            stats["ok" if rec.get("ok") else "bad"] += 1
            n = stats["ok"] + stats["bad"]
            if n % 50 == 0 or n == len(answers):
                print(f"[judge] {n}/{len(answers)} ok={stats['ok']} bad={stats['bad']}", flush=True)
        await asyncio.gather(*[job(k, raw) for k, raw in answers])
    await client.http.aclose()


EXCLUDED_PATH = RES / "p2_excluded.json"
OLD_RELAY = "https://rehdasu.cn/v1"
ENC = __import__("tiktoken").get_encoding("o200k_base")


def load_excluded():
    """(model, paper_id, condition, rep) keys whose first answer is discarded and redone."""
    return [tuple(k) for k in json.loads(EXCLUDED_PATH.read_text())["keys"]] if EXCLUDED_PATH.exists() else []


def load_give_up():
    """Cells abandoned after every retry hit the oversized-relay-input guard; analysed as missing."""
    return [tuple(k) for k in json.loads(EXCLUDED_PATH.read_text()).get("give_up", [])] if EXCLUDED_PATH.exists() else []


def latest_answers():
    out = {}
    for m in EVAL_MODELS:
        for r in read(review_path(m)):
            if r.get("ok"):
                out[(m, r["paper_id"], r["condition"], r["rep"])] = r["raw"]
    return out


async def run_neutralize(cfg):
    """Rewrite every answer as a neutral third-person assessment, so the judge cannot tell who asked."""
    name = cfg["judge_model"]
    client = Client(name, cfg["models"][name], cfg.get("defaults", {}))
    done = {(r["model"], r["paper_id"], r["condition"], r["rep"]) for r in read(NEUTRAL_PATH) if r.get("ok")}
    todo = [(k, raw) for k, raw in latest_answers().items() if k not in done]
    random.Random(cfg["seed"]).shuffle(todo)
    print(f"[neutralize:{client.c['model']}] {len(todo)} answers ({len(done)} done)", flush=True)
    stats = {"ok": 0, "bad": 0}

    def check_for(raw):
        def check(out):
            t = (out["text"] or "").strip()
            ratio = len(t) / max(1, len(raw))
            return bool(t) and 0.4 < ratio < 1.3 and len(ASKER_CUES.findall(t)) <= 1, None
        return check

    guard = FatalGuard("neutral_path")
    with open(NEUTRAL_PATH, "a") as f:
        async def job(key, raw):
            async with client.sem:
                if guard.stopped:
                    return
                rec = await call_with_retry(client, [{"role": "user", "content": NEUTRALIZE.format(response=raw)}],
                                            8000, check_for(raw))
            guard.see(rec)
            m, pid, cond, rep = key
            f.write(json.dumps({"model": m, "paper_id": pid, "condition": cond, "rep": rep,
                                "neutral_model": client.c["model"], "ok": rec.get("ok"), "text": rec.get("raw"),
                                "len_ratio": round(len(rec.get("raw") or "") / max(1, len(raw)), 3),
                                "cues_left": len(ASKER_CUES.findall(rec.get("raw") or "")),
                                "error": rec.get("error")}, ensure_ascii=False) + "\n")
            f.flush()
            stats["ok" if rec.get("ok") else "bad"] += 1
            n = stats["ok"] + stats["bad"]
            if n % 50 == 0 or n == len(todo):
                print(f"[neutralize] {n}/{len(todo)} ok={stats['ok']} bad={stats['bad']}", flush=True)
        await asyncio.gather(*[job(k, raw) for k, raw in todo])
    await client.http.aclose()


async def run_judge_neutral(cfg):
    name = cfg["judge_model"]
    client = Client(name, cfg["models"][name], cfg.get("defaults", {}))
    texts = {}
    for r in read(NEUTRAL_PATH):
        if r.get("ok"):
            texts[(r["model"], r["paper_id"], r["condition"], r["rep"])] = r["text"]
    done = {(r["model"], r["paper_id"], r["condition"], r["rep"]) for r in read(JUDGE_NEUTRAL_PATH) if r.get("ok")}
    todo = [(k, t) for k, t in texts.items() if k not in done]
    random.Random(cfg["seed"] + 1).shuffle(todo)
    print(f"[judge-neutral:{client.c['model']}] {len(todo)} answers ({len(done)} done)", flush=True)

    def check(out):
        obj = parse_json(out["text"])
        return valid_judge(obj), obj

    guard = FatalGuard("judge_neutral_path")
    with open(JUDGE_NEUTRAL_PATH, "a") as f:
        async def job(key, text):
            async with client.sem:
                if guard.stopped:
                    return
                rec = await call_with_retry(client, [{"role": "user", "content": JUDGE.format(response=text)}], 3000, check)
            guard.see(rec)
            m, pid, cond, rep = key
            f.write(json.dumps({"model": m, "paper_id": pid, "condition": cond, "rep": rep,
                                "judge_model": client.c["model"], **{k: v for k, v in rec.items() if k != "raw"}},
                               ensure_ascii=False) + "\n")
            f.flush()
        await asyncio.gather(*[job(k, t) for k, t in todo])
    await client.http.aclose()


def status(cfg):
    full = pd.read_csv(DATA / cfg["p3_sample_file"], index_col="paper_id").index
    n_target = len(full) * len(cfg["p2_conditions"])
    print(f"clean disguised manuscripts: {len(set(load_docs()) & set(full))}/{len(full)}; "
          f"full-study target per model = {n_target} (rep 0 of every paper and condition)")
    for m in EVAL_MODELS:
        recs = read(review_path(m))
        ok = {(r["paper_id"], r["condition"], r["rep"]) for r in recs if r.get("ok")}
        relay = {**cfg.get("defaults", {}), **cfg["models"][m]}["base_url"]
        ok0 = {(r["paper_id"], r["condition"], r["rep"]) for r in recs
               if r.get("ok") and r["rep"] == 0 and r["paper_id"] in full and r.get("relay", OLD_RELAY) == relay}
        by = pd.Series([k[1] for k in ok0]).value_counts().to_dict() if ok0 else {}
        tin = sum(int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in recs)
        tout = sum(int((r.get("usage") or {}).get("completion_tokens") or 0) for r in recs)
        print(f"{m:<9} full {len(ok0)}/{n_target}  all-ok {len(ok)}  records {len(recs)}  tok_in {tin:,} tok_out {tout:,}  {by}")
    for label, path in (("judge", JUDGE_PATH), ("neutral", NEUTRAL_PATH), ("judge_n", JUDGE_NEUTRAL_PATH)):
        j = read(path)
        print(f"{label:<9} ok {len({(r['model'], r['paper_id'], r['condition'], r['rep']) for r in j if r.get('ok')})}  records {len(j)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["review", "judge", "neutralize", "judge-neutral", "status"])
    ap.add_argument("--model", action="append")
    ap.add_argument("--full", action="store_true", help="344-paper study, p3_reps per condition")
    ap.add_argument("--cond", action="append", help="conditions to run instead of config p2_conditions")
    ap.add_argument("--retest-sample", help="csv with paper_id: papers to re-ask as rep 1")
    ap.add_argument("--retest-cond", action="append", help="conditions to re-ask")
    ap.add_argument("--retest-rep", type=int, default=1, help="rep index for the re-asked cells")
    ap.add_argument("--retest-only", action="store_true", help="run only the retest cells (2026-11-03: DeepSeek re-ask)")
    ap.add_argument("--sample", help="csv in data/ to use instead of p3_sample_file (2026-11-02: Grok subset)")
    args = ap.parse_args()
    cfg = load_config()
    if args.sample:
        cfg["p3_sample_file"] = args.sample
    if args.cmd == "review":
        retest = None
        if args.retest_sample and args.retest_cond:
            retest = (list(pd.read_csv(DATA / args.retest_sample).paper_id), args.retest_cond, args.retest_rep)
        asyncio.run(run_reviews(cfg, args.model or EVAL_MODELS, full=args.full, conds=args.cond, retest=retest,
                                retest_only=args.retest_only))
    elif args.cmd == "judge":
        asyncio.run(run_judge(cfg))
    elif args.cmd == "neutralize":
        asyncio.run(run_neutralize(cfg))
    elif args.cmd == "judge-neutral":
        asyncio.run(run_judge_neutral(cfg))
    status(cfg)


if __name__ == "__main__":
    main()
