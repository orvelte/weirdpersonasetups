"""E-C Arm 2 pilot: 15 probes x 2 orders x 5 prefixes x 3 versions x 5 samples = 2,250 calls.

Harness defaults (thinking on, effort medium, temperature 0.6, top_p 0.95), max_tokens 4000, pinned provider.
Resumable: results/EC_pilot/raw.jsonl is appended per completed call and existing (prompt_id, version, sample) keys are
skipped on restart; failures are recorded in the log and left out of raw.jsonl so the next invocation retries them.
Every call is wrapped so one failure never aborts the run. Spend guard: stops submitting when the EC_pilot tag passes
$8 (checked between chunks). Parsing (judge-free, tasks/EC/parse_arm2.py) writes parsed.csv.
Run: python3 -m tasks.EC.run_arm2_pilot [--smoke] [--samples 5] [--workers 12] [--reparse]
"""
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from harness import config
from harness.client import chat
from tasks.EC.parse_arm2 import parse_rows, write_parsed

HERE = Path(__file__).resolve().parent
OUT_DIR = config.ROOT / "results" / "EC_pilot"
RAW = OUT_DIR / "raw.jsonl"
TASK = "EC_pilot"
TAG_CAP = 8.0
MAX_TOKENS = 4000


def tag_spend(tag=TASK):
    if not config.SPEND_LOG.exists():
        return 0.0
    tot = 0.0
    for l in config.SPEND_LOG.read_text().splitlines():
        if l.strip():
            d = json.loads(l)
            if d.get("task") == tag:
                tot += d.get("cost") or 0.0
    return tot


def key_of(r):
    return (r["prompt_id"], r["version"], int(r["sample"]))


def load_raw():
    if not RAW.exists():
        return []
    return [json.loads(l) for l in RAW.read_text().splitlines() if l.strip()]


def one(job):
    pr, v, s = job["prompt"], job["version"], job["sample"]
    rec = {"prompt_id": pr["id"], "probe_id": pr["probe_id"], "axis": pr["axis"], "order": pr["order"], "prefix": pr["prefix"],
           "a": pr["a"], "b": pr["b"], "first_named": pr["first_named"], "pole": pr["pole"], "probe_text": pr["probe_text"],
           "version": v, "sample": s, "model_id": config.MODELS[v],
           "settings": dict(config.GEN_DEFAULTS, max_tokens=MAX_TOKENS, provider=config.PROVIDER)}
    try:
        r = chat(v, pr["messages"], sample=s, task=TASK, max_tokens=MAX_TOKENS)
        rec.update({"content": r["content"], "reasoning": r["reasoning"], "tool_calls": r.get("tool_calls"), "finish_reason": r["finish_reason"],
                    "truncated": r["truncated"], "usage": r["usage"], "cost": r["cost"], "provider": r.get("provider"),
                    "model_id_returned": r.get("model_id"), "cached": r.get("cached"), "error": None})
    except Exception as e:  # noqa
        rec.update({"error": f"{type(e).__name__}: {str(e)[:300]}"})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--reparse", action="store_true")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = json.loads((HERE / "arm2_prompts.json").read_text())["prompts"]
    if a.reparse:
        rows = load_raw(); write_parsed(parse_rows(rows), OUT_DIR / "parsed.csv"); print("reparsed", len(rows)); return
    done = {key_of(r) for r in load_raw()}
    jobs = []
    if a.smoke:
        pick = [p for p in prompts if p["id"] in ("P13_delete_error_log|ab|coding_agent", "P06_extra_fix|ba|customer_support", "P01_eulogy_poem|ab|chat")]
        for p in pick:
            for v in config.VERSIONS:
                jobs.append({"prompt": p, "version": v, "sample": 0})
    else:
        for s in range(a.samples):  # sample-major so a spend stop leaves balanced cells
            for p in prompts:
                for v in config.VERSIONS:
                    jobs.append({"prompt": p, "version": v, "sample": s})
    jobs = [j for j in jobs if (j["prompt"]["id"], j["version"], j["sample"]) not in done]
    print(f"{len(jobs)} calls to run ({len(done)} already done); EC_pilot spend so far ${tag_spend():.2f}", flush=True)
    t0 = time.time(); n_ok = n_err = 0
    CHUNK = 120
    for ci in range(0, len(jobs), CHUNK):
        sp = tag_spend()
        if sp > TAG_CAP:
            print(f"STOP: EC_pilot spend ${sp:.2f} exceeds ${TAG_CAP}; {len(jobs) - ci} calls not submitted", flush=True); break
        chunk = jobs[ci:ci + CHUNK]
        with ThreadPoolExecutor(a.workers) as ex, open(RAW, "a") as fh:
            for f in as_completed([ex.submit(one, j) for j in chunk]):
                rec = f.result()
                if rec.get("error"):
                    n_err += 1; print("  ERROR", rec["prompt_id"], rec["version"], rec["sample"], rec["error"][:160], flush=True)
                else:
                    n_ok += 1; fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(f"  {min(ci + CHUNK, len(jobs))}/{len(jobs)} ({n_ok} ok, {n_err} failed), EC_pilot spend ${tag_spend():.2f}, {time.time() - t0:.0f}s", flush=True)
    rows = load_raw()
    write_parsed(parse_rows(rows), OUT_DIR / "parsed.csv")
    print(f"done: {n_ok} ok, {n_err} failed this run; raw.jsonl has {len(rows)} records; EC_pilot spend ${tag_spend():.2f}")
    if a.smoke:
        for r in rows[-len(jobs):] if jobs else []:
            print(f"\n--- {r['prompt_id']} {r['version']} finish={r['finish_reason']} usage={r['usage']}\nREASONING[:400]: {(r['reasoning'] or '')[:400]!r}\nCONTENT: {r['content'][:300]!r}")


if __name__ == "__main__":
    main()
