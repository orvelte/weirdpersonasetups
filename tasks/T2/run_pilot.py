"""T2 Phase-0 pilot: sampled coin-flip continuations across Qwen 3.5/3.6/3.8.

Main arm: thinking OFF (the model is writing a non-Assistant turn; primary measure will be log-probs with no
thinking), temperature 1.0 / top_p 1.0 so sampling is unbiased, max_tokens 60.
  40 cells x 3 versions x 5 samples = 600 calls.
Contrast arm: thinking ON, framings a_user + e_agentic, variant 'coworker', both mappings, 5 samples,
max_tokens 4000.  2 x 2 x 3 x 5 = 60 calls.
Output: results/T2_pilot/raw.jsonl (one line per sample, cell metadata + response).
Run: python3 -m tasks.T2.run_pilot [--smoke]
"""
import json, sys, time
from pathlib import Path
from harness import config
from harness.client import chat_many, total_spend

HERE = Path(__file__).resolve().parent
OUT_DIR = config.ROOT / "results" / "T2_pilot"
OUT = OUT_DIR / "raw.jsonl"
N_SAMPLES = 5
SMOKE = "--smoke" in sys.argv

def main():
    prompts = json.loads((HERE / "prompts.json").read_text())
    cells = prompts["cells"]
    jobs, meta = [], []
    # Main arm
    for c in cells:
        for v in config.VERSIONS:
            for s in range(N_SAMPLES):
                jobs.append({"version": v, "messages": c["messages"], "sample": s,
                             "thinking": False, "max_tokens": 60, "temperature": 1.0, "top_p": 1.0})
                meta.append({"arm": "main", "thinking": False, **{k: c[k] for k in
                             ("id", "variant", "framing", "mapping", "preferred_outcome", "agentic_prefix")}})
    # Thinking-ON contrast
    for c in cells:
        if c["variant"] != "coworker" or c["framing"] not in ("a_user", "e_agentic"):
            continue
        for v in config.VERSIONS:
            for s in range(N_SAMPLES):
                jobs.append({"version": v, "messages": c["messages"], "sample": s,
                             "thinking": True, "max_tokens": 4000, "temperature": 1.0, "top_p": 1.0})
                meta.append({"arm": "think", "thinking": True, **{k: c[k] for k in
                             ("id", "variant", "framing", "mapping", "preferred_outcome", "agentic_prefix")}})
    if SMOKE:
        keep = [0, 1, len(jobs) - 1]
        jobs = [jobs[i] for i in keep]; meta = [meta[i] for i in keep]
    print("jobs:", len(jobs), "spend before: ${:.3f}".format(total_spend()), flush=True)
    t0 = time.time()
    res = chat_many(jobs, task="T2", workers=12)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / ("raw_smoke.jsonl" if SMOKE else "raw.jsonl")
    with open(out_path, "w") as f:
        for j, m, r in zip(jobs, meta, res):
            row = dict(m)
            row.update({"version": j["version"], "sample": j["sample"], "model_id": r.get("model_id"),
                        "provider": r.get("provider"), "content": r["content"], "reasoning": r.get("reasoning", ""),
                        "finish_reason": r["finish_reason"], "truncated": r["truncated"], "usage": r["usage"],
                        "cost": r["cost"], "cached": r["cached"]})
            f.write(json.dumps(row) + "\n")
    cost = sum(r["cost"] for r in res if not r["cached"])
    print("wrote", out_path, "| new spend this run: ${:.4f} | total spend: ${:.3f} | {:.0f}s".format(
        cost, total_spend(), time.time() - t0))
    if SMOKE:
        for m, r in zip(meta, res):
            print("---", m["id"], m["arm"], "\n", repr(r["content"][:300]), "\n reasoning:", repr(r.get("reasoning", "")[:200]))

if __name__ == "__main__":
    main()
