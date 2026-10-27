"""OpenRouter chat client with sqlite cache, spend log, retries, concurrency.

Usage:
    from harness.client import chat, chat_many
    r = chat("3.8", [{"role":"user","content":"hi"}], sample=0)
    rs = chat_many([("3.8", msgs, i) for i in range(10)])
Each response dict: content, reasoning, finish_reason, truncated, usage, cost, provider, model_id, cached.
"""
import hashlib, json, os, sqlite3, threading, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from . import config

_lock = threading.Lock()
_db = None

def _load_env():
    p = config.ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
_load_env()
API_KEY = os.environ.get("OPENROUTER_API_KEY")

def _conn():
    global _db
    if _db is None:
        config.CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        _db = sqlite3.connect(str(config.CACHE_DB), check_same_thread=False)
        _db.execute("CREATE TABLE IF NOT EXISTS resp (key TEXT PRIMARY KEY, task TEXT, created REAL, request TEXT, response TEXT)")
        _db.commit()
    return _db

def _key(payload, sample):
    s = json.dumps({"p": payload, "sample": sample}, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

def total_spend():
    if not config.SPEND_LOG.exists():
        return 0.0
    return sum(json.loads(l).get("cost", 0) for l in config.SPEND_LOG.read_text().splitlines() if l.strip())

def _log_spend(task, model, cost, usage):
    with _lock:
        config.SPEND_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(config.SPEND_LOG, "a") as f:
            f.write(json.dumps({"t": time.time(), "task": task, "model": model, "cost": cost, "usage": usage}) + "\n")

def build_payload(version, messages, *, tools=None, temperature=None, top_p=None, max_tokens=None,
                  thinking=None, seed=None, stop=None, model_id=None, provider=None, extra=None):
    g = config.GEN_DEFAULTS
    mid = model_id or config.MODELS[version]
    payload = {
        "model": mid,
        "messages": messages,
        "temperature": g["temperature"] if temperature is None else temperature,
        "top_p": g["top_p"] if top_p is None else top_p,
        "max_tokens": g["max_tokens"] if max_tokens is None else max_tokens,
    }
    th = g["thinking"] if thinking is None else thinking
    payload["reasoning"] = {"enabled": bool(th)}
    if th:
        payload["reasoning"]["effort"] = g.get("reasoning_effort", "medium")
    if provider is not False:
        payload["provider"] = {"order": [provider or config.PROVIDER], "allow_fallbacks": False}
    if tools: payload["tools"] = tools
    if seed is not None: payload["seed"] = seed
    if stop: payload["stop"] = stop
    if extra: payload.update(extra)
    return payload

def chat(version, messages, sample=0, task="misc", retries=6, **kw):
    """One sampled response. `sample` only varies the cache key so repeated samples are stored separately."""
    payload = build_payload(version, messages, **kw)
    key = _key(payload, sample)
    db = _conn()
    with _lock:
        row = db.execute("SELECT response FROM resp WHERE key=?", (key,)).fetchone()
    if row:
        r = json.loads(row[0]); r["cached"] = True; return r
    if total_spend() > config.API_SPEND_CAP:
        raise RuntimeError(f"API spend cap ${config.API_SPEND_CAP} reached; ASK Olivia before continuing")
    last = None
    for attempt in range(retries):
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, timeout=600,
                                 headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json",
                                          "HTTP-Referer": "https://github.com/orvelte/psm-replications", "X-Title": "psm-replications"})
            if resp.status_code in (402, 429, 500, 502, 503, 504, 520, 524):
                last = f"{resp.status_code}: {resp.text[:200]}"; time.sleep(min(60, 2 ** attempt + random.random())); continue
            data = resp.json()
            if "error" in data:
                last = str(data["error"])[:300]
                if any(w in last.lower() for w in ("rate", "overload", "capacity", "budget", "credit", "402")):
                    time.sleep(min(60, 2 ** attempt + random.random())); continue
                raise RuntimeError(last)
            break
        except (requests.RequestException, ValueError) as e:
            last = repr(e); time.sleep(min(60, 2 ** attempt + random.random()))
    else:
        raise RuntimeError(f"chat failed after {retries} retries: {last}")
    ch = data["choices"][0]
    msg = ch.get("message", {})
    usage = data.get("usage", {})
    r = {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning") or "",
        "tool_calls": msg.get("tool_calls"),
        "finish_reason": ch.get("finish_reason"),
        "truncated": ch.get("finish_reason") == "length",
        "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")} | {"reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")},
        "cost": usage.get("cost", 0.0),
        "provider": data.get("provider"),
        "model_id": data.get("model"),
        "version": version,
        "sample": sample,
        "cached": False,
    }
    _log_spend(task, payload["model"], r["cost"], r["usage"])
    with _lock:
        db.execute("INSERT OR REPLACE INTO resp VALUES (?,?,?,?,?)", (key, task, time.time(), json.dumps(payload), json.dumps(r)))
        db.commit()
    return r

def chat_many(jobs, workers=12, task="misc", progress=True, **kw):
    """jobs: list of dicts with keys version, messages, sample and optional per-job kwargs. Returns results in order."""
    out = [None] * len(jobs)
    def run(i, j):
        j = dict(j); kws = {**kw, **{k: v for k, v in j.items() if k not in ("version", "messages", "sample")}}
        return i, chat(j["version"], j["messages"], j.get("sample", 0), task=task, **kws)
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run, i, j) for i, j in enumerate(jobs)]
        done = 0
        for f in as_completed(futs):
            i, r = f.result(); out[i] = r; done += 1
            if progress and done % 25 == 0:
                print(f"  {task}: {done}/{len(jobs)} done, spend so far ${total_spend():.2f}", flush=True)
    return out
