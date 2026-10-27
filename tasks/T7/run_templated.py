"""T7 WITH-template arm via OpenRouter: the raw text is the sole user message, no system prompt, thinking ON
(reasoning effort "medium", see harness/config.py), max_tokens 2000, 1 sample per prompt x version.

Run: python3 -m tasks.T7.run_templated [--versions 3.5,3.6,3.8] [--limit N] [--max-tokens 2000] [--samples 1]
Writes results/T7_pilot/templated_raw.jsonl (one row per call, classified) and prints a summary.
Budget guard: aborts if the projected worst case would push T7 spend past --budget (default $4).
"""
import argparse, json, time
from collections import Counter, defaultdict
from pathlib import Path
from harness import config
from harness.client import chat, total_spend
from concurrent.futures import ThreadPoolExecutor, as_completed
from harness.stats import fmt_rate
from tasks.T7.classify import classify

HERE = Path(__file__).resolve().parent
OUT_DIR = config.ROOT / "results" / "T7_pilot"
OUT = OUT_DIR / "templated_raw.jsonl"
PRICE_PER_1K = 0.003  # conservative $/1k tokens (prompt+completion); observed ~0.002 on the smoke test. Pre-run guard only


def tolerant_chat_many(jobs, workers, max_tokens, passes=3):
    """Like harness.client.chat_many but a failed call (e.g. upstream 429 after the client's own retries) yields None
    instead of aborting the run. Successful calls are cached by the harness, so re-running fills gaps for free."""
    out = [None] * len(jobs)
    for p in range(passes):
        todo = [i for i, r in enumerate(out) if r is None]
        if not todo:
            break
        if p:
            print(f"  pass {p + 1}: retrying {len(todo)} failed calls after 60s"); time.sleep(60)
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(chat, jobs[i]["version"], jobs[i]["messages"], jobs[i]["sample"], task="T7",
                              thinking=True, max_tokens=max_tokens): i for i in todo}
            done = 0
            for f in as_completed(futs):
                i = futs[f]
                try:
                    out[i] = f.result()
                except Exception as e:
                    print(f"  FAILED job {i} ({jobs[i]['version']}): {str(e)[:120]}", flush=True)
                done += 1
                if done % 25 == 0:
                    print(f"  T7: {done}/{len(todo)} done this pass; spend log total ${total_spend():.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", default=",".join(config.VERSIONS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--budget", type=float, default=4.0, help="max USD this script may add")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()

    prompts = json.loads((HERE / "prompts.json").read_text())
    if args.limit:
        prompts = prompts[: args.limit]
    versions = args.versions.split(",")
    jobs, meta = [], []
    for v in versions:
        for p in prompts:
            for s in range(args.samples):
                jobs.append({"version": v, "messages": [{"role": "user", "content": p["text"]}], "sample": s})
                meta.append((v, p, s))
    worst = len(jobs) * (args.max_tokens + 300) / 1000 * PRICE_PER_1K
    spend0 = total_spend()
    print(f"{len(jobs)} calls; worst-case ~${worst:.2f}; total spend so far ${spend0:.3f}")
    if worst > args.budget:
        raise SystemExit(f"worst-case ${worst:.2f} exceeds --budget ${args.budget}; lower --max-tokens or --limit")

    t0 = time.time()
    results = tolerant_chat_many(jobs, workers=args.workers, max_tokens=args.max_tokens, passes=args.passes)
    n_fail = sum(1 for r in results if r is None)
    if n_fail:
        print(f"WARNING: {n_fail} calls failed after {args.passes} passes (provider rate limit?); re-run to fill them from cache")
    t7_cost = sum((r or {}).get("cost") or 0 for r in results)
    n_cached = sum(1 for r in results if r and r.get("cached"))
    print(f"done in {time.time() - t0:.0f}s; this run's calls cost ${t7_cost:.3f} ({n_cached} cached); "
          f"spend log total ${total_spend():.3f} (shared with other tasks)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(OUT, "w") as f:
        for (v, p, s), r in zip(meta, results):
            if r is None:
                r = {"content": "", "reasoning": "", "finish_reason": "error", "truncated": False, "usage": None, "cost": 0.0}
            label, feats = classify(p["text"], r["content"])
            if r["finish_reason"] == "error":
                feats["call_failed"] = True
            if r["truncated"] and not (r["content"] or "").strip():
                feats["truncated_empty"] = True  # thinking ate the whole budget
            row = {
                "id": p["id"], "family": p["family"], "arm": "templated", "source": "openrouter", "version": v,
                "model_id": r.get("model_id"), "provider": r.get("provider"), "sample": s,
                "settings": {"temperature": config.GEN_DEFAULTS["temperature"], "top_p": config.GEN_DEFAULTS["top_p"],
                             "max_tokens": args.max_tokens, "thinking": True,
                             "reasoning_effort": config.GEN_DEFAULTS.get("reasoning_effort"), "system_prompt": None},
                "prompt": p["text"], "content": r["content"], "reasoning": r["reasoning"],
                "finish_reason": r["finish_reason"], "truncated": r["truncated"], "usage": r["usage"], "cost": r["cost"],
                "label": label, "features": feats,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", OUT)

    # ---- summary ----
    by = defaultdict(list)
    for row in rows:
        by[row["version"]].append(row)
    print("\nversion  ASSISTANT                 DOCUMENT                  OTHER                     truncated")
    for v in versions:
        rs = by[v]; n = len(rs)
        c = Counter(r["label"] for r in rs)
        tr = sum(r["truncated"] for r in rs)
        print(f"{v:7s}  {fmt_rate(c['ASSISTANT'], n):25s} {fmt_rate(c['DOCUMENT'], n):25s} {fmt_rate(c['OTHER'], n):25s} {fmt_rate(tr, n)}")
    print("\nASSISTANT rate by family x version")
    fams = sorted({r["family"] for r in rows}, key=lambda x: [p["family"] for p in prompts].index(x))
    for fam in fams:
        cells = []
        for v in versions:
            rs = [r for r in by[v] if r["family"] == fam]
            cells.append(fmt_rate(sum(r["label"] == "ASSISTANT" for r in rs), len(rs)))
        print(f"{fam:10s} " + " | ".join(f"{c:24s}" for c in cells))


if __name__ == "__main__":
    main()
