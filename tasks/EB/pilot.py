"""E-B Phase 0 pilot: no-preference variants only, {plain, mundane} x order {A, B} x 3 versions x N samples per order.

Default N=5 per order (10 per scenario x embedding x version cell): 30 x 2 x 2 x 3 x 5 = 1,800 calls.
Every call is wrapped: an API failure is recorded as an error row and never aborts the run. A spend guard stops issuing
new calls once the EB_pilot tag in data/spend.jsonl exceeds --cap dollars (cached calls are free and still run).
Outputs: results/EB_pilot/raw.jsonl (prompt, content, reasoning, usage, cost, truncation) and parsed.csv.
Run: python3 -m tasks.EB.pilot [--samples 5] [--embeddings plain,mundane] [--versions 3.5,3.6,3.8] [--only H1_novel_opening]
     python3 -m tasks.EB.pilot --reparse      # re-parse raw.jsonl with the current rules, no API calls
"""
import argparse, csv, json, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from harness import config
from harness.client import chat
from tasks.EB import parse as P

try:
    from tasks.T1.eval_awareness import find_hits
    EVAL_AWARE_SOURCE = "tasks/T1/eval_awareness.py"
except Exception:  # pragma: no cover
    from tasks.EB.eval_awareness import find_hits
    EVAL_AWARE_SOURCE = "tasks/EB/eval_awareness.py"

TASK = "EB_pilot"
OUT_DIR = config.ROOT / "results" / "EB_pilot"
SPEC = P.SPEC
CHAT_SYSTEM = SPEC["chat_system"]
_lock = threading.Lock()


def task_spend(task=TASK):
    if not config.SPEND_LOG.exists():
        return 0.0
    tot = 0.0
    for l in config.SPEND_LOG.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("task") == task:
            tot += r.get("cost", 0) or 0
    return tot


def messages_for(s, embedding, pref, order):
    return [{"role": "system", "content": CHAT_SYSTEM}] + s["variants"][embedding][pref][order]


def run_one(job, cap):
    rec = {k: job[k] for k in ("scenario_id", "axis", "embedding", "pref", "order", "version", "sample")}
    rec.update(model_id=config.MODELS[job["version"]], settings=dict(config.GEN_DEFAULTS, provider=config.PROVIDER),
               messages=job["messages"], content="", reasoning="", finish_reason=None, truncated=False, usage={}, cost=0.0,
               cached=False, error=None)
    if task_spend() > cap:
        rec["error"] = f"skipped: EB_pilot spend cap ${cap} reached"
        return rec
    try:
        r = chat(job["version"], job["messages"], sample=job["sample"], task=TASK)
        rec.update(content=r["content"], reasoning=r["reasoning"], finish_reason=r["finish_reason"], truncated=r["truncated"],
                   usage=r["usage"], cost=r["cost"], cached=r["cached"], provider=r.get("provider"), model_id=r.get("model_id") or rec["model_id"])
    except Exception as e:  # one failure never aborts the run
        rec["error"] = repr(e)[:500]
    return rec


def parse_rec(rec):
    s = P.SCEN[rec["scenario_id"]]
    pr = P.parse(rec["scenario_id"], rec["content"]) if not rec.get("error") else \
        {"category": "error", "choice": "error", "pole": "error", "sub": "", "matched": ""}
    hits = find_hits(rec.get("reasoning") or "") + find_hits(rec.get("content") or "")
    pr_hits = [h[0] for h in find_hits(rec.get("reasoning") or "")]
    ct_hits = [h[0] for h in find_hits(rec.get("content") or "")]
    first_pole = s["poles"][0]
    return {**{k: rec[k] for k in ("scenario_id", "axis", "embedding", "pref", "order", "version", "sample")},
            "category": pr["category"], "choice": pr["choice"], "pole": pr["pole"], "sub": pr["sub"], "matched": pr["matched"],
            "first_pole": int(pr["pole"] == first_pole) if pr["category"] == "option" else "",
            "aware": int(bool(hits)), "aware_patterns": "|".join(sorted(set(h[0] for h in hits))),
            "aware_where": ("reasoning" if pr_hits else "") + ("+" if pr_hits and ct_hits else "") + ("content" if ct_hits else ""),
            "truncated": int(bool(rec.get("truncated"))), "finish_reason": rec.get("finish_reason"),
            "completion_tokens": (rec.get("usage") or {}).get("completion_tokens"), "reasoning_tokens": (rec.get("usage") or {}).get("reasoning_tokens"),
            "cost": rec.get("cost", 0.0), "error": rec.get("error") or ""}


def write_outputs(recs, raw_path, parsed_path):
    with open(raw_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = [parse_rec(r) for r in recs]
    with open(parsed_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5, help="samples per order (x2 orders = per cell)")
    ap.add_argument("--embeddings", default="plain,mundane")
    ap.add_argument("--versions", default=",".join(config.VERSIONS))
    ap.add_argument("--only", default=None, help="comma-separated scenario ids")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cap", type=float, default=7.8, help="stop issuing new EB_pilot calls above this tag spend (USD)")
    ap.add_argument("--reparse", action="store_true")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path, parsed_path = OUT_DIR / "raw.jsonl", OUT_DIR / "parsed.csv"
    if a.reparse:
        recs = [json.loads(l) for l in open(raw_path) if l.strip()]
        rows = write_outputs(recs, raw_path, parsed_path)
        print(f"re-parsed {len(rows)} rows -> {parsed_path}"); return
    only = set(a.only.split(",")) if a.only else None
    jobs = []
    for s in SPEC["scenarios"]:
        if only and s["id"] not in only:
            continue
        for emb in a.embeddings.split(","):
            for order in ("A", "B"):
                for v in a.versions.split(","):
                    for k in range(a.samples):
                        # sample index distinct per order so the cache keeps A/B (identical text for axis 5) apart
                        jobs.append(dict(scenario_id=s["id"], axis=s["axis"], embedding=emb, pref="no_pref", order=order, version=v,
                                         sample=k + (100 if order == "B" else 0), messages=messages_for(s, emb, "no_pref", order)))
    print(f"{len(jobs)} calls; EB_pilot spend so far ${task_spend():.2f}; cap ${a.cap}", flush=True)
    t0 = time.time(); recs = [None] * len(jobs); done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_one, j, a.cap): i for i, j in enumerate(jobs)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                recs[i] = f.result()
            except Exception as e:  # belt and braces
                recs[i] = {**{k: jobs[i][k] for k in ("scenario_id", "axis", "embedding", "pref", "order", "version", "sample")},
                           "messages": jobs[i]["messages"], "content": "", "reasoning": "", "error": repr(e)[:500], "cost": 0.0, "usage": {}}
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} done, EB_pilot spend ${task_spend():.2f}, {time.time()-t0:.0f}s", flush=True)
    rows = write_outputs(recs, raw_path, parsed_path)
    n_err = sum(1 for r in rows if r["error"]); n_unclear = sum(1 for r in rows if r["category"] == "unclear")
    print(f"done: {len(rows)} rows, {n_err} errors, {n_unclear} unclear, truncated {sum(r['truncated'] for r in rows)}, "
          f"aware {sum(r['aware'] for r in rows)}; EB_pilot spend ${task_spend():.2f}; eval-awareness regex from {EVAL_AWARE_SOURCE}")


if __name__ == "__main__":
    main()
