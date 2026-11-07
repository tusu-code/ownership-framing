"""Resumable runner for the ownership-framing pilot.

Usage
  python src/run.py models                 # list model ids each relay exposes
  python src/run.py smoke  --model gpt     # 1 paper x all conditions, prints parsed output
  python src/run.py review [--model gpt]   # full review matrix (skips finished calls)
  python src/run.py probe  [--model gpt]   # memorization probe on abstracts
  python src/run.py status                 # progress, token use, estimated cost

Every finished call is appended to results/<task>_<model>.jsonl at once, keyed by
(paper_id, condition, rep); a rerun skips keys that already parsed. Call order is
shuffled with a fixed seed so time drift on the relay is not aligned with condition.
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path

import httpx
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import REVIEW_CONDITIONS, probe_messages, review_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"


def load_config():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return cfg


def load_sample(cfg):
    sample = pd.read_csv(DATA / cfg["sample_file"], index_col="paper_id")
    docs = {}
    for line in open(DATA / "manuscripts.jsonl"):
        r = json.loads(line)
        if r["paper_id"] in sample.index:
            docs[r["paper_id"]] = r
    return sample, docs


def abstract_snippet(text, n_words=40):
    m = re.search(r"(?im)^#+\s*abstract\s*$", text)
    body = text[m.end():] if m else text
    words = re.sub(r"[*#_`]", "", body).split()
    return " ".join(words[:n_words])


def parse_json(text):
    if not text:
        return None
    t = re.sub(r"(?s)<think>.*?</think>", "", text).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip())
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        # LaTeX inside strings (\mathcal, \hat) is not a legal JSON escape; double the backslash.
        for c in (cand, re.sub(r'\\(?![\\"/bfnrtu])', r"\\\\", cand)):
            try:
                obj = json.loads(c)
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def valid_review(obj):
    if not obj:
        return False
    try:
        return (int(obj["rating"]) in (0, 2, 4, 6, 8, 10)
                and str(obj["recommendation"]).lower() in ("accept", "reject")
                and all(1 <= int(obj[k]) <= 4 for k in ("soundness", "presentation", "contribution")))
    except (KeyError, TypeError, ValueError):
        return False


class Client:
    def __init__(self, name, mcfg, defaults):
        self.name, self.c = name, {**defaults, **mcfg}
        # api_key_envs (optional list) spreads calls over several keys of the same relay; concurrency is per key.
        self.key_envs = list(self.c.get("api_key_envs") or [self.c["api_key_env"]])
        self.keys = [os.environ.get(e, "") for e in self.key_envs]
        self.key = self.keys[0]
        self.inflight = [0] * len(self.keys)
        self.sem = asyncio.Semaphore(int(self.c.get("concurrency", 4)) * len(self.keys))
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(float(self.c.get("timeout_s", 600)), connect=30))

    async def chat(self, messages, max_tokens):
        i = min(range(len(self.keys)), key=self.inflight.__getitem__)
        self.inflight[i] += 1
        try:
            out = await self._chat(messages, max_tokens, self.keys[i])
        finally:
            self.inflight[i] -= 1
        out["key_slot"] = self.key_envs[i]
        return out

    async def _chat(self, messages, max_tokens, key):
        if self.c.get("mock"):
            await asyncio.sleep(0.01)
            r = random.Random(json.dumps(messages)[-200:] + str(time.time()))
            rating = r.choice([2, 4, 6, 8])
            body = {"summary": "mock", "strengths": ["a"], "weaknesses": [{"text": "b", "severity": "major"}],
                    "soundness": r.randint(1, 4), "presentation": 3, "contribution": 2, "rating": rating,
                    "confidence": 3, "recommendation": "accept" if rating >= 6 else "reject"}
            return {"text": json.dumps(body), "usage": {"prompt_tokens": 10000, "completion_tokens": 400},
                    "model": "mock", "finish_reason": "stop"}
        api = self.c.get("api", "openai")
        base = self.c["base_url"].rstrip("/")
        if api == "anthropic":
            url = base + "/v1/messages"
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            payload = {"model": self.c["model"], "messages": messages, "max_tokens": max_tokens}
        else:
            url = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
            payload = {"model": self.c["model"], "messages": messages,
                       self.c.get("max_tokens_field", "max_tokens"): max_tokens}
        if self.c.get("temperature") is not None:
            payload["temperature"] = self.c["temperature"]
        if api != "anthropic":
            # Each call is a fresh one-message conversation; a random end-user id per call keeps the
            # relay from grouping calls into one session, and store=false asks OpenAI not to keep it.
            payload["user"] = uuid.uuid4().hex
            if self.c.get("store_false"):
                payload["store"] = False
        payload.update(self.c.get("extra_body") or {})
        if self.c.get("stream") and api != "anthropic":
            # Streaming keeps the connection busy, so the relay's ~100 s gateway timeout (HTTP 524)
            # no longer cuts off long answers.
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            parts, reasoning, usage, model, finish = [], [], {}, None, None
            async with self.http.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="ignore")
                    raise httpx.HTTPStatusError(f"{resp.status_code}: {body[:300]}", request=resp.request, response=resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    model = j.get("model") or model
                    if j.get("usage"):
                        usage = j["usage"]
                    for ch in j.get("choices") or []:
                        delta = ch.get("delta") or {}
                        if delta.get("content"):
                            parts.append(delta["content"])
                        if delta.get("reasoning_content"):
                            reasoning.append(delta["reasoning_content"])
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
            return {"text": "".join(parts), "usage": usage, "model": model, "finish_reason": finish,
                    "reasoning": "".join(reasoning) or None}
        resp = await self.http.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise httpx.HTTPStatusError(f"{resp.status_code}: {resp.text[:300]}", request=resp.request, response=resp)
        j = resp.json()
        if api == "anthropic":
            text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
            u = j.get("usage", {})
            usage = {"prompt_tokens": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                     + u.get("cache_creation_input_tokens", 0), "completion_tokens": u.get("output_tokens", 0)}
            return {"text": text, "usage": usage, "model": j.get("model"), "finish_reason": j.get("stop_reason")}
        ch = j["choices"][0]
        return {"text": ch["message"].get("content") or "", "usage": j.get("usage", {}),
                "model": j.get("model"), "finish_reason": ch.get("finish_reason"),
                "fingerprint": j.get("system_fingerprint")}


def out_path(task, model):
    return RES / f"{task}_{model}.jsonl"


def done_keys(task, model):
    p, keys = out_path(task, model), set()
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            if r.get("ok"):
                keys.add((r["paper_id"], r["condition"], r["rep"]))
    return keys


async def worker(client, job, task, max_tokens, retries, f, stats):
    messages = job.pop("messages")
    async with client.sem:
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                out = await client.chat(messages, max_tokens)
                obj = parse_json(out["text"])
                ok = valid_review(obj) if task == "review" else obj is not None
                rec = {**job, "ok": ok, "parsed": obj, "raw": out["text"], "usage": out["usage"],
                       "provider_model": out.get("model"), "fingerprint": out.get("fingerprint"),
                       "finish_reason": out.get("finish_reason"), "latency_s": round(time.time() - t0, 2),
                       "attempt": attempt, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
                # A well-formed answer, or a stable non-JSON answer on the last try (often a refusal), is kept.
                if ok or attempt == retries:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    stats["ok" if ok else "bad"] += 1
                    stats["in"] += int(out["usage"].get("prompt_tokens") or 0)
                    stats["out"] += int(out["usage"].get("completion_tokens") or 0)
                    return
            except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError) as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (400, 401, 403, 404) or attempt == retries:
                    f.write(json.dumps({**job, "ok": False, "error": str(e)[:500], "attempt": attempt,
                                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False) + "\n")
                    f.flush()
                    stats["err"] += 1
                    if status in (401, 403):
                        stats["fatal"] = str(e)[:200]
                    return
            await asyncio.sleep(min(60, 2 ** attempt * 3) + random.random() * 2)


async def run_task(cfg, task, models, smoke=False):
    sample, docs = load_sample(cfg)
    pids = list(sample.index)[:1] if smoke else list(sample.index)
    reps = 1 if (smoke or task == "probe") else int(cfg["reps"])
    conditions = cfg["conditions"] if task == "review" else ["probe"]
    max_tokens = int(cfg["max_tokens_review"] if task == "review" else cfg["max_tokens_probe"])
    RES.mkdir(exist_ok=True)
    for name in models:
        client = Client(name, cfg["models"][name], cfg.get("defaults", {}))
        if not client.c.get("mock") and not client.key:
            print(f"[{name}] missing key: set {client.c['api_key_env']} in .env; skipped")
            continue
        done = set() if smoke else done_keys(task, name)
        jobs = []
        for pid in pids:
            d = docs[pid]
            for cond in conditions:
                for rep in range(reps):
                    if (pid, cond, rep) in done:
                        continue
                    msgs = (review_messages(d["title"], d["text"], cond) if task == "review"
                            else probe_messages(abstract_snippet(d["text"])))
                    jobs.append({"paper_id": pid, "condition": cond, "rep": rep, "model": name,
                                 "model_id": client.c["model"], "messages": msgs})
        random.Random(cfg["seed"]).shuffle(jobs)
        cap = int(cfg.get("max_calls_per_run", 10 ** 9))
        jobs = jobs[:cap]
        print(f"[{name}] {task}: {len(jobs)} calls to make ({len(done)} already done)")
        if not jobs:
            continue
        stats = {"ok": 0, "bad": 0, "err": 0, "in": 0, "out": 0}
        path = RES / f"smoke_{name}.jsonl" if smoke else out_path(task, name)
        with open(path, "a") as f:
            tasks = [asyncio.create_task(worker(client, j, task, max_tokens, int(cfg["retries"]), f, stats))
                     for j in jobs]
            t0, n = time.time(), len(tasks)
            for i, fut in enumerate(asyncio.as_completed(tasks), 1):
                await fut
                if stats.get("fatal"):
                    print(f"[{name}] auth error, stopping: {stats['fatal']}")
                    for t in tasks:
                        t.cancel()
                    break
                if i % 20 == 0 or i == n:
                    print(f"[{name}] {i}/{n} ok={stats['ok']} bad={stats['bad']} err={stats['err']} "
                          f"tok_in={stats['in']:,} tok_out={stats['out']:,} {time.time() - t0:.0f}s", flush=True)
        await client.http.aclose()
        if smoke:
            for line in open(path):
                r = json.loads(line)
                p = r.get("parsed") or {}
                print(f"  {r['condition']:<14} ok={r['ok']} rating={p.get('rating')} rec={p.get('recommendation')} "
                      f"model={r.get('provider_model')} in={r.get('usage', {}).get('prompt_tokens')} "
                      f"out={r.get('usage', {}).get('completion_tokens')} err={r.get('error', '')[:120]}")


async def list_models(cfg):
    seen = set()
    for name, m in cfg["models"].items():
        m = {**cfg.get("defaults", {}), **m}
        if m.get("mock") or (m["base_url"], m["api_key_env"]) in seen:
            continue
        seen.add((m["base_url"], m["api_key_env"]))
        base = m["base_url"].rstrip("/")
        url = base + "/models" if base.endswith("/v1") else base + "/v1/models"
        try:
            async with httpx.AsyncClient(timeout=30) as h:
                r = await h.get(url, headers={"Authorization": f"Bearer {os.environ.get(m['api_key_env'], '')}"})
            ids = sorted(x["id"] for x in r.json().get("data", []))
            print(f"{m['base_url']}  ({len(ids)} models)\n  " + "\n  ".join(ids))
        except Exception as e:  # noqa: BLE001
            print(f"{m['base_url']}: {e}")


def status(cfg):
    rows = []
    for p in sorted(RES.glob("*.jsonl")):
        if p.name.startswith("smoke_"):
            continue
        task, model = p.stem.split("_", 1)
        recs = [json.loads(l) for l in open(p)]
        okk = {(r["paper_id"], r["condition"], r["rep"]) for r in recs if r.get("ok")}
        tin = sum(int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in recs)
        tout = sum(int((r.get("usage") or {}).get("completion_tokens") or 0) for r in recs)
        price = cfg["models"].get(model, {}).get("price_per_mtok", [0, 0])
        rows.append(dict(task=task, model=model, ok=len(okk), records=len(recs),
                         errors=sum(1 for r in recs if "error" in r), tok_in=tin, tok_out=tout,
                         est_cost=round(tin / 1e6 * price[0] + tout / 1e6 * price[1], 2),
                         provider_models=",".join(sorted({str(r.get("provider_model")) for r in recs if r.get("ok")}))))
    print(pd.DataFrame(rows).to_string(index=False) if rows else "no results yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["models", "smoke", "review", "probe", "status"])
    ap.add_argument("--model", action="append", help="model key in config.yaml; repeatable; default all enabled")
    args = ap.parse_args()
    cfg = load_config()
    models = args.model or [k for k, v in cfg["models"].items() if v.get("enabled", True)]
    if args.cmd == "models":
        asyncio.run(list_models(cfg))
    elif args.cmd == "status":
        status(cfg)
    else:
        task = "review" if args.cmd == "smoke" else args.cmd
        asyncio.run(run_task(cfg, task, models, smoke=args.cmd == "smoke"))


if __name__ == "__main__":
    main()
