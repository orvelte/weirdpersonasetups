"""T3 pilot: 50 prompts (13 a, 13 b, 12 c, 12 d) x 2 frames x 3 versions x 1 sample = 300 first-turn calls.
Agentic frame: if the model calls a tool, one canned result is returned and the model gets ONE more turn (max 2 assistant turns).
Writes results/T3_pilot/raw.jsonl (with reasoning) and results/T3_pilot/parsed.csv. Run: python3 -m tasks.T3.run_pilot"""
import csv, json, sys, time
from pathlib import Path
from harness import config
from harness.client import chat, total_spend
from concurrent.futures import ThreadPoolExecutor
from tasks.T3.frames import messages_for, tools_for, canned_result, TOOL_NAMES
from tasks.T3.regex import score, FAMILIES

OUT = config.ROOT / "results" / "T3_pilot"
PROMPTS = json.loads((Path(__file__).with_name("prompts.json")).read_text())
FRAMES = ["chat", "agentic"]
WORKERS = int(__import__("os").environ.get("T3_WORKERS", "4")); RETRIES = int(__import__("os").environ.get("T3_RETRIES", "10"))
PILOT_N = {"a_body": 13, "b_physical": 13, "c_logistics": 12, "d_execution": 12}

def chat_many(jobs, task="T3", workers=4, retries=10):
    """Like harness.client.chat_many, but also retries OpenRouter 402 in-flight-budget errors with a long backoff
    (those are raised, not retried, by the harness)."""
    def one(j):
        kws = {k: v for k, v in j.items() if k not in ("version", "messages", "sample")}
        for attempt in range(30):
            try:
                return chat(j["version"], j["messages"], j.get("sample", 0), task=task, retries=retries, **kws)
            except RuntimeError as e:
                if "402" in str(e) or "in_flight" in str(e) or "credits" in str(e):
                    time.sleep(20 + 10 * attempt); continue
                raise
        raise RuntimeError("gave up after repeated 402s")
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, jobs))
    print(f"  {task}: {len(out)} done, spend ${total_spend():.2f}", flush=True)
    return out

def pilot_prompts():
    seen = {f: 0 for f in PILOT_N}
    out = []
    for p in PROMPTS:
        if seen[p["family"]] < PILOT_N[p["family"]]:
            out.append(p); seen[p["family"]] += 1
    return out

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = pilot_prompts()
    print(f"{len(prompts)} prompts; spend before: ${total_spend():.3f}; settings: {config.GEN_DEFAULTS}; provider={config.PROVIDER}")
    cells = [(v, fr, p) for v in config.VERSIONS for fr in FRAMES for p in prompts]
    jobs = []
    for v, fr, p in cells:
        j = {"version": v, "messages": messages_for(fr, p["text"]), "sample": 0}
        if tools_for(fr): j["tools"] = tools_for(fr)
        jobs.append(j)
    t0 = time.time()
    r1 = chat_many(jobs, task="T3", workers=WORKERS, retries=RETRIES)
    print(f"turn 1 done in {time.time()-t0:.0f}s; spend ${total_spend():.3f}")
    # second turn for tool callers
    follow_idx, follow_jobs = [], []
    for i, (job, r) in enumerate(zip(jobs, r1)):
        tcs = r.get("tool_calls") or []
        if tcs and r["finish_reason"] != "length":
            msgs = list(job["messages"]) + [{"role": "assistant", "content": r["content"] or "", "tool_calls": tcs}]
            for tc in tcs:
                fn = tc.get("function", {}); name = fn.get("name")
                try: args = json.loads(fn.get("arguments") or "{}")
                except Exception: args = {}
                msgs.append({"role": "tool", "tool_call_id": tc.get("id"), "name": name, "content": canned_result(name, args)})
            j2 = {"version": job["version"], "messages": msgs, "sample": 0, "tools": job["tools"]}
            follow_idx.append(i); follow_jobs.append(j2)
    print(f"{len(follow_jobs)} tool-calling responses get a second turn")
    r2 = {}
    if follow_jobs:
        res = chat_many(follow_jobs, task="T3", workers=WORKERS, retries=RETRIES)
        r2 = dict(zip(follow_idx, res))
    print(f"all calls done; spend ${total_spend():.3f}")

    rows = []
    with open(OUT / "raw.jsonl", "w") as f:
        for i, ((v, fr, p), job, r) in enumerate(zip(cells, jobs, r1)):
            turns = [r] + ([r2[i]] if i in r2 else [])
            content_all = "\n\n".join(t["content"] for t in turns if t["content"]).strip()
            reasoning_all = "\n\n".join(t["reasoning"] for t in turns if t["reasoning"]).strip()
            tool_names = [tc.get("function", {}).get("name") for t in turns for tc in (t.get("tool_calls") or [])]
            rec = {"version": v, "model_id": r.get("model_id"), "provider": r.get("provider"), "frame": fr,
                   "prompt_id": p["id"], "family": p["family"], "prompt_text": p["text"],
                   "n_turns": len(turns), "tool_called": int(bool(tool_names)), "tool_names": tool_names,
                   "unfinished_after_2_turns": int(bool(turns[-1].get("tool_calls"))),
                   "truncated": int(any(t["truncated"] for t in turns)),
                   "finish_reasons": [t["finish_reason"] for t in turns],
                   "cost": sum(t.get("cost", 0) or 0 for t in turns),
                   "completion_tokens": sum((t["usage"].get("completion_tokens") or 0) for t in turns),
                   "reasoning_tokens": sum((t["usage"].get("reasoning_tokens") or 0) for t in turns),
                   "content": content_all, "reasoning": reasoning_all,
                   "turns": [{"content": t["content"], "reasoning": t["reasoning"], "tool_calls": t.get("tool_calls"),
                              "finish_reason": t["finish_reason"], "usage": t["usage"], "cached": t.get("cached")} for t in turns]}
            sc = score(content_all, p["text"]); sr = score(reasoning_all, p["text"])
            rec["matches"] = sc["matches"]; rec["reasoning_matches"] = sr["matches"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            row = {k: rec[k] for k in ("version", "frame", "prompt_id", "family", "n_turns", "tool_called", "unfinished_after_2_turns", "truncated", "completion_tokens", "reasoning_tokens", "cost")}
            row["words"] = sc["words"]; row["reasoning_words"] = sr["words"]
            for fam in list(FAMILIES) + ["ANY"]:
                for suf in ("_n", "_any", "_per1k"):
                    row[fam + suf] = round(sc[fam + suf], 3) if suf == "_per1k" else sc[fam + suf]
            row["excluded_n"] = sc["excluded_n"]
            for fam in list(FAMILIES) + ["ANY"]:
                row["r_" + fam + "_n"] = sr[fam + "_n"]; row["r_" + fam + "_per1k"] = round(sr[fam + "_per1k"], 3)
            row["content_head"] = content_all[:160].replace("\n", " ")
            rows.append(row)
    with open(OUT / "parsed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows; spend after: ${total_spend():.3f}")

if __name__ == "__main__":
    main()
