"""T7 raw arm against a self-hosted vLLM OpenAI-compatible server (no chat template).

Raw arm (default):   POST {base-url}/completions   prompt = raw text, max_tokens 200, temperature 0.6, top_p 0.95,
                     2 samples, skip_special_tokens=False so any <think>/<|im_start|> the model emits is visible.
Templated arm:       --templated -> POST {base-url}/chat/completions with the raw text as the sole user message,
                     chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "medium"} (matches the API arm),
                     max_tokens 2000 by default.
Run:
  python3 -m tasks.T7.run_raw --base-url https://<podid>-8000.proxy.runpod.net/v1 --version 3.8 --model Qwen/Qwen3.8-27B
  python3 -m tasks.T7.run_raw --base-url ... --version 3.8 --model Qwen/Qwen3.8-27B --templated
Output: results/T7_pilot/raw_<version>.jsonl  or  results/T7_pilot/templated_vllm_<version>.jsonl
Resumable: (id, sample) pairs already in the output file are skipped. Retries with backoff; per-request timeout.
No logprobs are requested.
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from harness import config
from tasks.T7.classify import classify, split_think

HERE = Path(__file__).resolve().parent
OUT_DIR = config.ROOT / "results" / "T7_pilot"
RAW_DEFAULTS = dict(max_tokens=200, temperature=0.6, top_p=0.95)
CHAT_TEMPLATE_KWARGS = {"enable_thinking": True, "reasoning_effort": "medium"}


def build_request(base_url, model, prompt_text, sample, templated=False, max_tokens=None, temperature=None,
                  top_p=None, seed_base=1000):
    """Return (url, payload). Pure function so it can be unit-tested without a server."""
    base = base_url.rstrip("/")
    t = RAW_DEFAULTS["temperature"] if temperature is None else temperature
    tp = RAW_DEFAULTS["top_p"] if top_p is None else top_p
    seed = seed_base + sample  # distinct seed per sample; vLLM honours it
    if templated:
        mt = 2000 if max_tokens is None else max_tokens
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": mt, "temperature": t, "top_p": tp, "n": 1, "seed": seed,
            "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
        }
        return base + "/chat/completions", payload
    mt = RAW_DEFAULTS["max_tokens"] if max_tokens is None else max_tokens
    payload = {
        "model": model, "prompt": prompt_text,
        "max_tokens": mt, "temperature": t, "top_p": tp, "n": 1, "seed": seed,
        "skip_special_tokens": False,  # expose <think>, <|im_start|> etc. if the model emits them
    }
    return base + "/completions", payload


def parse_response(data, templated):
    """Normalise a vLLM response into (text, reasoning, finish_reason, usage)."""
    ch = data["choices"][0]
    usage = data.get("usage") or {}
    if templated:
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if not reasoning and ("<think>" in content or "</think>" in content):  # server without a reasoning parser
            reasoning, content = split_think(content)
        elif not reasoning and ch.get("finish_reason") == "length" and "</think>" not in content:
            # Qwen templates pre-open <think>; if the cap hit before </think>, the whole output is reasoning
            reasoning, content = content, ""
        return content, reasoning, ch.get("finish_reason"), usage
    text = ch.get("text") or ""
    return text, "", ch.get("finish_reason"), usage


def post_with_retries(url, payload, timeout, retries, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout, headers=headers)
            if resp.status_code in (408, 429, 500, 502, 503, 504, 520, 524):
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            else:
                data = resp.json()
                if "error" in data:
                    last = str(data["error"])[:300]
                    if resp.status_code == 400:
                        raise RuntimeError(last)  # bad request; retrying won't help
                else:
                    return data
        except RuntimeError:
            raise
        except (requests.RequestException, ValueError) as e:
            last = repr(e)[:300]
        time.sleep(min(60, 2 ** attempt + random.random()))
    raise RuntimeError(f"request failed after {retries} retries: {last}")


def load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line); done.add((r["id"], r["sample"]))
                except Exception:
                    pass
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. https://<podid>-8000.proxy.runpod.net/v1")
    ap.add_argument("--version", required=True, choices=list(config.HF_IDS))
    ap.add_argument("--model", default=None, help="served model name; default harness.config.HF_IDS[version]")
    ap.add_argument("--templated", action="store_true", help="use /chat/completions (template applied by vLLM)")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=None, help="default 200 raw / 2000 templated")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    ap.add_argument("--dry-run", action="store_true", help="print the first request and exit")
    args = ap.parse_args(argv)

    model = args.model or config.HF_IDS[args.version]
    arm = "templated" if args.templated else "raw"
    out = Path(args.out) if args.out else OUT_DIR / (f"templated_vllm_{args.version}.jsonl" if args.templated else f"raw_{args.version}.jsonl")
    prompts = json.loads((HERE / "prompts.json").read_text())
    if args.limit:
        prompts = prompts[: args.limit]

    if args.dry_run:
        url, payload = build_request(args.base_url, model, prompts[0]["text"], 0, args.templated, args.max_tokens, args.temperature, args.top_p)
        print(url); print(json.dumps(payload, indent=1)); return 0

    done = load_done(out)
    todo = [(p, s) for p in prompts for s in range(args.samples) if (p["id"], s) not in done]
    print(f"{arm} arm, version {args.version}, model {model}: {len(done)} already saved, {len(todo)} to run -> {out}")
    if not todo:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)

    def one(p, s):
        url, payload = build_request(args.base_url, model, p["text"], s, args.templated, args.max_tokens, args.temperature, args.top_p)
        t0 = time.time()
        data = post_with_retries(url, payload, args.timeout, args.retries, args.api_key)
        text, reasoning, finish, usage = parse_response(data, args.templated)
        label, feats = classify(p["text"], text)
        if finish == "length" and not text.strip():
            feats["truncated_empty"] = True
        return {
            "id": p["id"], "family": p["family"], "arm": arm, "source": "vllm", "version": args.version,
            "model_id": data.get("model", model), "provider": "vllm", "sample": s,
            "settings": {"temperature": payload["temperature"], "top_p": payload["top_p"], "max_tokens": payload["max_tokens"],
                         "thinking": bool(args.templated), "reasoning_effort": CHAT_TEMPLATE_KWARGS["reasoning_effort"] if args.templated else None,
                         "chat_template": bool(args.templated), "system_prompt": None, "seed": payload["seed"]},
            "prompt": p["text"], "content": text, "reasoning": reasoning,
            "finish_reason": finish, "truncated": finish == "length",
            "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")},
            "latency_s": round(time.time() - t0, 2), "label": label, "features": feats,
        }

    n_ok = n_err = 0
    with ThreadPoolExecutor(args.workers) as ex, open(out, "a") as f:
        futs = {ex.submit(one, p, s): (p["id"], s) for p, s in todo}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                n_err += 1; print(f"  ERROR {key}: {e}", file=sys.stderr); continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"  {n_ok}/{len(todo)} saved", flush=True)
    print(f"saved {n_ok}, errors {n_err} (re-run to retry errors; saved rows are skipped)")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
