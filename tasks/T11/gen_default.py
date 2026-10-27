"""Default answers: each version (3.5/3.6/3.8) answers the 200 questions with NO persona system prompt, in two frames
(chat: bare messages; agentic: T3-style workspace-agent system prompt + 3 irrelevant tools). Thinking ON with harness
defaults (temperature 0.6, top_p 0.95, reasoning effort medium) but max_tokens 4000, 1 sample. Tool calls get one canned
result and the model gets up to 2 more turns; the FINAL content is what gets scored. 1,200 first-turn calls.
Then paraphrases of the 200 chat-frame answers of one version (PARA_VERSION) by config.JUDGE_MODEL (noise floor).
Resumable. Run: python3 -m tasks.T11.gen_default [--limit N] [--no-para]"""
import json, sys, time
from harness import config
from harness.client import chat, total_spend
from tasks.T11.common import (load_questions, frame_messages, safe_chat_many, read_jsonl, append_jsonl, OUT,
                              TOOLS, canned_result)

MAX_TOKENS = 4000
FRAMES = ["chat", "agentic"]
PARA_VERSION = "3.5"
MAX_EXTRA_TURNS = 2
BATCH = 96

PARA_PROMPT = ("Paraphrase the following text. Keep the same meaning, the same level of detail, the same length within "
               "about 10%, and the same overall structure (keep lists as lists, headings as headings), but change the "
               "wording and sentence construction throughout. Output only the paraphrase, nothing else.\n\n<text>\n{t}\n</text>")

def run_default(limit=None):
    qs = load_questions()
    path = OUT / "default.jsonl"
    done = {(r["version"], r["frame"], r["qid"]) for r in read_jsonl(path) if not r.get("error")}
    todo = [(v, fr, q) for v in config.VERSIONS for fr in FRAMES for q in qs if (v, fr, q["id"]) not in done]
    if limit: todo = todo[:limit]
    print(f"default: {len(done)} done, {len(todo)} to run; spend before ${total_spend():.3f}", flush=True)
    t0 = time.time()
    for b in range(0, len(todo), BATCH):
        chunk = todo[b:b + BATCH]
        jobs = []
        for v, fr, q in chunk:
            j = {"version": v, "messages": frame_messages(fr, q["messages"]), "sample": 0}
            if fr == "agentic": j["tools"] = TOOLS
            jobs.append(j)
        res = safe_chat_many(jobs, task="T11_def", workers=24, max_tokens=MAX_TOKENS)
        turns = [[r] for r in res]
        # follow-up turns for tool callers
        for extra in range(MAX_EXTRA_TURNS):
            idx, fjobs = [], []
            for i, (job, ts) in enumerate(zip(jobs, turns)):
                last = ts[-1]
                tcs = (last.get("tool_calls") or []) if not last.get("error") else []
                if tcs and last.get("finish_reason") != "length":
                    msgs = list(job["messages"])
                    for t in ts:
                        msgs.append({"role": "assistant", "content": t["content"] or "", "tool_calls": t.get("tool_calls")} if t.get("tool_calls")
                                    else {"role": "assistant", "content": t["content"] or ""})
                        for tc in (t.get("tool_calls") or []):
                            fn = tc.get("function", {}); name = fn.get("name")
                            try: args = json.loads(fn.get("arguments") or "{}")
                            except Exception: args = {}
                            msgs.append({"role": "tool", "tool_call_id": tc.get("id"), "name": name, "content": canned_result(name, args)})
                    idx.append(i); fjobs.append({"version": job["version"], "messages": msgs, "sample": 0, "tools": TOOLS})
            if not fjobs: break
            print(f"  turn {extra + 2}: {len(fjobs)} tool-calling responses", flush=True)
            fres = safe_chat_many(fjobs, task="T11_def", workers=24, max_tokens=MAX_TOKENS)
            for i, r in zip(idx, fres): turns[i].append(r)
        recs = []
        for (v, fr, q), ts in zip(chunk, turns):
            if ts[0].get("error"):
                recs.append({"version": v, "frame": fr, "qid": q["id"], "kind": q["kind"], "family": q["family"], "error": ts[0]["error"]}); continue
            ok = [t for t in ts if not t.get("error")]
            final = ok[-1]
            tool_names = [tc.get("function", {}).get("name") for t in ok for tc in (t.get("tool_calls") or [])]
            recs.append({"version": v, "frame": fr, "qid": q["id"], "kind": q["kind"], "family": q["family"],
                         "model_id": ts[0].get("model_id"), "provider": ts[0].get("provider"),
                         "content": final["content"] or "", "all_content": "\n\n".join(t["content"] for t in ok if t["content"]).strip(),
                         "reasoning": "\n\n".join(t["reasoning"] for t in ok if t.get("reasoning")).strip(),
                         "n_turns": len(ok), "tool_called": int(bool(tool_names)), "tool_names": tool_names,
                         "unfinished": int(bool(final.get("tool_calls"))), "truncated": int(any(t["truncated"] for t in ok)),
                         "finish_reasons": [t["finish_reason"] for t in ok],
                         "completion_tokens": sum((t["usage"].get("completion_tokens") or 0) for t in ok),
                         "reasoning_tokens": sum((t["usage"].get("reasoning_tokens") or 0) for t in ok),
                         "cost": sum(t.get("cost", 0) or 0 for t in ok)})
        append_jsonl(path, recs)
        nerr = sum(1 for r in recs if r.get("error"))
        print(f"default batch {b // BATCH + 1}: {len(recs)} written ({nerr} errors); {time.time() - t0:.0f}s; spend ${total_spend():.3f}", flush=True)
        if any(r.get("error") == "task budget exceeded" for r in recs):
            print("TASK BUDGET EXCEEDED, stopping"); return False
    return True

def run_paraphrase(limit=None):
    path = OUT / "paraphrase.jsonl"
    done = {r["qid"] for r in read_jsonl(path) if not r.get("error")}
    src = [r for r in read_jsonl(OUT / "default.jsonl") if not r.get("error") and r["version"] == PARA_VERSION
           and r["frame"] == "chat" and r["qid"] not in done and r["content"].strip()]
    if limit: src = src[:limit]
    print(f"paraphrase: {len(done)} done, {len(src)} to run with {config.JUDGE_MODEL}; spend ${total_spend():.3f}", flush=True)
    jobs = [{"version": PARA_VERSION, "messages": [{"role": "user", "content": PARA_PROMPT.format(t=r["content"])}], "sample": 0} for r in src]
    # The OpenRouter endpoint for the judge model refuses thinking=False ("Reasoning is mandatory"); as in T1, use
    # thinking=True with effort low, which keeps it cheap. provider=False: not pinned to deepinfra.
    res = safe_chat_many(jobs, task="T11_para", workers=6, model_id=config.JUDGE_MODEL, provider=False, thinking=True,
                         max_tokens=MAX_TOKENS, temperature=0.7, extra={"reasoning": {"enabled": True, "effort": "low"}})
    recs = []
    for r0, r in zip(src, res):
        if r.get("error"):
            recs.append({"qid": r0["qid"], "src_version": PARA_VERSION, "src_frame": "chat", "error": r["error"]}); continue
        recs.append({"qid": r0["qid"], "kind": r0["kind"], "family": r0["family"], "src_version": PARA_VERSION, "src_frame": "chat",
                     "model_id": r.get("model_id"), "content": r["content"], "truncated": int(bool(r["truncated"])),
                     "src_words": len(r0["content"].split()), "words": len(r["content"].split()), "cost": r.get("cost", 0)})
    append_jsonl(path, recs)
    print(f"paraphrase: {len(recs)} written ({sum(1 for r in recs if r.get('error'))} errors); spend ${total_spend():.3f}")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    ok = run_default(limit)
    if ok and "--no-para" not in sys.argv:
        run_paraphrase(limit)

if __name__ == "__main__":
    main()
