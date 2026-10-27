"""T11 confound fix: regenerate default answers under the library's generation settings.
Each version (3.5/3.6/3.8) answers the 200 questions in the chat frame only, thinking OFF, max_tokens 350 (matching the
persona libraries), temperature 0.6 / top_p 0.95, 1 sample, in two conditions:
  nosys  : no system prompt (as the original defaults, but thinking off and length-matched)
  minsys : minimal system prompt "You are a helpful assistant."
1,200 calls, spend tag T11_fix_def. Resumable. Run: python3 -m tasks.T11.gen_default_fix [--limit N]"""
import os, sys, time
os.environ.setdefault("T11_BUDGET", "26")  # common.TASK_BUDGET counts all T11_* tags; T11 had spent $15.08 before this script
from harness import config
from harness.client import total_spend
from tasks.T11.common import load_questions, frame_messages, safe_chat_many, read_jsonl, append_jsonl, OUT, t11_spend

MAX_TOKENS = 350
CONDITIONS = {"nosys": None, "minsys": "You are a helpful assistant."}
TAG = "T11_fix_def"
FIX_BUDGET = 10.0  # USD across all T11_fix* tags
BATCH = 150
PATH = OUT / "default_fix.jsonl"

def fix_spend():
    if not config.SPEND_LOG.exists(): return 0.0
    import json
    return sum(json.loads(l).get("cost", 0) for l in config.SPEND_LOG.read_text().splitlines() if l.strip() and '"T11_fix' in l)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    qs = load_questions()
    done = {(r["version"], r["condition"], r["qid"]) for r in read_jsonl(PATH) if not r.get("error")}
    todo = [(v, c, q) for c in CONDITIONS for v in config.VERSIONS for q in qs if (v, c, q["id"]) not in done]
    if limit: todo = todo[:limit]
    print(f"default_fix -> {PATH}: {len(done)} done, {len(todo)} to run; T11_fix spend ${fix_spend():.3f}, T11 ${t11_spend():.3f}, total ${total_spend():.3f}", flush=True)
    t0 = time.time()
    for b in range(0, len(todo), BATCH):
        if fix_spend() > FIX_BUDGET:
            print("T11_fix budget exceeded, stopping"); break
        chunk = todo[b:b + BATCH]
        jobs = [{"version": v, "messages": frame_messages("chat", q["messages"], system=CONDITIONS[c]), "sample": 0} for v, c, q in chunk]
        res = safe_chat_many(jobs, task=TAG, workers=16, thinking=False, max_tokens=MAX_TOKENS)
        recs = []
        for (v, c, q), r in zip(chunk, res):
            base = {"version": v, "condition": c, "frame": "chat", "qid": q["id"], "kind": q["kind"], "family": q["family"]}
            if r.get("error"):
                recs.append({**base, "error": r["error"]}); continue
            recs.append({**base, "system": CONDITIONS[c], "model_id": r.get("model_id"), "provider": r.get("provider"),
                         "content": r["content"] or "", "reasoning": r.get("reasoning") or "",
                         "truncated": int(bool(r["truncated"])), "finish_reason": r["finish_reason"], "usage": r["usage"],
                         "cost": r.get("cost", 0), "cached": r.get("cached")})
        append_jsonl(PATH, recs)
        nerr = sum(1 for r in recs if r.get("error"))
        print(f"batch {b // BATCH + 1}: {len(recs)} written ({nerr} errors); {time.time() - t0:.0f}s; T11_fix ${fix_spend():.3f}; total ${total_spend():.3f}", flush=True)
        if any(r.get("error") == "task budget exceeded" for r in recs):
            print("TASK BUDGET EXCEEDED, stopping"); break
    print(f"done; T11_fix spend ${fix_spend():.3f}; total ${total_spend():.3f}")

if __name__ == "__main__":
    main()
