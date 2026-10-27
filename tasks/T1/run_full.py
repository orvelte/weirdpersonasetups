"""T1 full run (prereg/T1.md): 25 scenarios x 2 frames x 3 versions x 10 samples = 1,500 episodes.

Resumable: completed episodes are appended to results/T1_full/raw.jsonl as each chunk finishes and skipped on restart.
Every API call is wrapped, so one failure never aborts the run; failed episodes are retried from scratch in a second pass
(earlier turns are served from the cache) and, if they fail again, left out of raw.jsonl so the next invocation resumes them.
Agentic frame: tool loop up to env.MAX_TURNS (8) assistant turns with the stateful fake tools in tasks/T1/env.py.
Chat frame: one call with the minimal "Today is ..." system prompt. Parsed with tasks/T1/parse.py (rule-based, no judge).
Run: python3 -m tasks.T1.run_full [--samples 10] [--workers 10] [--only A1_five_sams] [--reparse]
"""
import argparse, csv, json, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from harness import config
from harness.client import chat, total_spend
from tasks.T1 import parse as P
from tasks.T1.env import SPEC, TOOL_DEFS, MAX_TURNS, EpisodeState, start_messages

HERE = Path(__file__).resolve().parent
OUT_DIR = config.ROOT / "results" / "T1_full"
PRED_PATH = HERE / "forward_predictions_full.json"
TASK = "T1_full"


def key_of(c):
    return (c["scenario_id"], c["version"], c["frame"], int(c["sample"]))


def cell_record(scen, version, frame, sample):
    return {"scenario_id": scen["id"], "family": scen["family"], "version": version, "frame": frame, "sample": sample,
            "model_id": config.MODELS[version], "settings": dict(config.GEN_DEFAULTS, provider=config.PROVIDER, max_turns=MAX_TURNS),
            "turns": [], "tool_results": [], "truncated_any": False, "n_turns": 0, "hit_turn_cap": False, "cost": 0.0,
            "messages": start_messages(scen, frame)}


def record_turn(rec, r):
    rec["turns"].append({"content": r["content"], "reasoning": r["reasoning"], "tool_calls": r.get("tool_calls"),
                         "finish_reason": r["finish_reason"], "truncated": r["truncated"], "usage": r["usage"], "cost": r["cost"],
                         "provider": r.get("provider"), "model_id_returned": r.get("model_id"), "cached": r.get("cached"),
                         **{k: r[k] for k in ("split_from_content", "think_unclosed") if k in r}})
    rec["n_turns"] += 1
    rec["truncated_any"] = rec["truncated_any"] or bool(r["truncated"])
    rec["cost"] += r["cost"] or 0.0


def safe_many(jobs, workers, task=TASK):
    """Like harness.client.chat_many but a failure is returned as {"error": ...} instead of raising."""
    out = [None] * len(jobs)
    def run(i, j):
        try:
            return i, chat(j["version"], j["messages"], j.get("sample", 0), task=task, **{k: v for k, v in j.items() if k not in ("version", "messages", "sample")})
        except Exception as e:  # noqa
            return i, {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run, i, j) for i, j in enumerate(jobs)]
        done = 0
        for f in as_completed(futs):
            i, r = f.result(); out[i] = r; done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(jobs)} calls done, spend ${total_spend():.2f}", flush=True)
    return out


def run_cells(cells, by_id, workers, task=TASK, extra_kw=None, msg_hook=None, resp_hook=None):
    """Runs the given cells to completion. Returns (completed, failed).
    task: spend-log tag. extra_kw: extra keyword args for harness.client.chat (e.g. {"extra": {...}}), identical for every call.
    msg_hook(messages) -> messages actually sent (e.g. to append a think prefill); the stored transcript is unchanged.
    resp_hook(response) -> response, applied before the turn is recorded (e.g. to split a prefilled think block out of content).
    The defaults reproduce the full run exactly."""
    failed = []
    extra_kw = extra_kw or {}
    msg_hook = msg_hook or (lambda m: m)
    resp_hook = resp_hook or (lambda r: r)
    chat_cells = [c for c in cells if c["frame"] == "chat"]
    if chat_cells:
        print(f"  chat: {len(chat_cells)} cells", flush=True)
        jobs = [dict({"version": c["version"], "messages": msg_hook(c["messages"]), "sample": c["sample"]}, **extra_kw) for c in chat_cells]
        for c, r in zip(chat_cells, safe_many(jobs, workers, task)):
            if "error" in r:
                c["error"] = r["error"]; failed.append(c); continue
            r = resp_hook(r)
            record_turn(c, r)
            c["messages"] = c["messages"] + [{"role": "assistant", "content": r["content"]}]
    active = [c for c in cells if c["frame"] == "agentic"]
    states = {id(c): EpisodeState(by_id[c["scenario_id"]]) for c in active}
    for turn in range(MAX_TURNS):
        if not active: break
        jobs = [dict({"version": c["version"], "messages": msg_hook(c["messages"]), "sample": c["sample"],
                      "tools": [TOOL_DEFS[t] for t in by_id[c["scenario_id"]]["agentic"]["tools"]]}, **extra_kw) for c in active]
        print(f"  agentic turn {turn + 1}: {len(active)} cells", flush=True)
        nxt = []
        for c, r in zip(active, safe_many(jobs, workers, task)):
            if "error" in r:
                c["error"] = r["error"]; failed.append(c); continue
            r = resp_hook(r)
            record_turn(c, r)
            tcs = r.get("tool_calls") or []
            asst = {"role": "assistant", "content": r["content"] or ""}
            if tcs: asst["tool_calls"] = tcs
            c["messages"] = c["messages"] + [asst]
            if not tcs:
                continue  # final answer
            st = states[id(c)]
            for t in tcs:
                fn = t.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                res = st.call(fn.get("name"), args if isinstance(args, dict) else {})
                c["tool_results"].append({"turn": turn, "tool": fn.get("name"), "args": args, "result": res})
                c["messages"].append({"role": "tool", "tool_call_id": t.get("id"), "content": res})
            if turn == MAX_TURNS - 1:
                c["hit_turn_cap"] = True
            else:
                nxt.append(c)
        active = nxt
    for c in cells:
        if c["frame"] == "agentic" and "error" not in c:
            c["env_final"] = states[id(c)].summary()
    completed = [c for c in cells if "error" not in c]
    return completed, failed


def append_raw(cells, raw_path):
    with open(raw_path, "a") as f:
        for c in cells:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_raw(raw_path):
    if not raw_path.exists(): return []
    return [json.loads(l) for l in open(raw_path) if l.strip()]


def write_csv(cells, csv_path):
    preds = {(p["id"], p["frame"]): p for p in json.load(open(PRED_PATH))["predictions"]}
    cols = ["scenario_id", "family", "version", "frame", "sample", "choice", "corner", "unclear", "rule_idx", "flags", "truncated_any",
            "n_turns", "hit_turn_cap", "tools_called", "n_duplicate_sends", "reasoning_tokens", "completion_tokens", "n_empty_reasoning_turns",
            "cost", "p_persona", "p_careful", "cached_turns"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for c in sorted(cells, key=key_of):
            choice, idx, flags = P.parse(c)
            corner_opt = P.SCEN[c["scenario_id"]]["corner_option"]
            p = preds.get((c["scenario_id"], c["frame"]), {"p_persona": "", "p_careful": ""})
            w.writerow({"scenario_id": c["scenario_id"], "family": c["family"], "version": c["version"], "frame": c["frame"], "sample": c["sample"],
                        "choice": choice, "corner": "" if choice == "unclear" else int(choice == corner_opt), "unclear": int(choice == "unclear"),
                        "rule_idx": idx, "flags": ";".join(flags), "truncated_any": int(c["truncated_any"]), "n_turns": c["n_turns"],
                        "hit_turn_cap": int(c["hit_turn_cap"]), "tools_called": ";".join(t["tool"] for t in c["tool_results"]),
                        "n_duplicate_sends": (c.get("env_final") or {}).get("n_duplicate_sends", 0),
                        "reasoning_tokens": sum((t["usage"].get("reasoning_tokens") or 0) for t in c["turns"]),
                        "completion_tokens": sum((t["usage"].get("completion_tokens") or 0) for t in c["turns"]),
                        "n_empty_reasoning_turns": sum(1 for t in c["turns"] if not (t["reasoning"] or "").strip()),
                        "cost": round(c["cost"], 5), "p_persona": p["p_persona"], "p_careful": p["p_careful"],
                        "cached_turns": sum(1 for t in c["turns"] if t.get("cached"))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--only", default=None, help="comma separated scenario ids")
    ap.add_argument("--versions", default=",".join(config.VERSIONS))
    ap.add_argument("--frames", default="chat,agentic")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=5, help="scenarios per chunk (raw.jsonl is appended after each chunk)")
    ap.add_argument("--reparse", action="store_true", help="re-parse existing raw.jsonl with the current rules; no API calls")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path, csv_path = OUT_DIR / "raw.jsonl", OUT_DIR / "parsed.csv"
    if a.reparse:
        cells = load_raw(raw_path); write_csv(cells, csv_path); print(f"re-parsed {len(cells)} cells -> {csv_path}"); return
    assert PRED_PATH.exists(), "run tasks.T1.forward_predict_full first (predictions must be frozen before any Qwen call)"
    frozen = json.load(open(PRED_PATH))
    t0 = time.time()
    assert frozen["frozen_at_unix"] < t0, "predictions must be frozen before the run starts"
    only = set(a.only.split(",")) if a.only else None
    predicted = {p["id"] for p in frozen["predictions"]}
    scen_list = [s for s in SPEC["scenarios"] if (only is None or s["id"] in only) and s["id"] in predicted]
    skipped = [s["id"] for s in SPEC["scenarios"] if s["id"] not in predicted]
    if skipped:
        print(f"skipping scenarios without frozen predictions (run them with tasks.T1.run_honesty_ext): {skipped}", flush=True)
    by_id = {s["id"]: s for s in scen_list}
    versions, frames = a.versions.split(","), a.frames.split(",")
    done = {key_of(c) for c in load_raw(raw_path)}
    todo = [cell_record(s, v, f, k) for s in scen_list for v in versions for f in frames for k in range(a.samples)]
    todo = [c for c in todo if key_of(c) not in done]
    spend0 = total_spend()
    print(f"{len(todo)} episodes to run ({len(done)} already in {raw_path}); predictions frozen at {frozen['frozen_at']}; spend before: ${spend0:.3f}", flush=True)
    all_failed = []
    for i in range(0, len(scen_list), a.chunk):
        ids = {s["id"] for s in scen_list[i:i + a.chunk]}
        cells = [c for c in todo if c["scenario_id"] in ids]
        if not cells: continue
        print(f"chunk {i // a.chunk + 1}: scenarios {sorted(ids)}, {len(cells)} episodes", flush=True)
        completed, failed = run_cells(cells, by_id, a.workers)
        append_raw(completed, raw_path)
        all_failed += failed
        print(f"  chunk done: {len(completed)} ok, {len(failed)} failed; spend ${total_spend():.3f} (+${total_spend() - spend0:.3f}); {time.time() - t0:.0f}s elapsed", flush=True)
    if all_failed:
        print(f"second pass: retrying {len(all_failed)} failed episodes from scratch (earlier turns come from the cache)", flush=True)
        for c in all_failed:
            print(f"    failed: {key_of(c)}: {c['error']}")
        retry = [cell_record(by_id[c["scenario_id"]], c["version"], c["frame"], c["sample"]) for c in all_failed]
        completed, failed = run_cells(retry, by_id, a.workers)
        append_raw(completed, raw_path)
        if failed:
            print(f"STILL FAILED after second pass ({len(failed)}); re-run this script to resume them:")
            for c in failed:
                print(f"    {key_of(c)}: {c['error']}")
    cells = load_raw(raw_path)
    write_csv(cells, csv_path)
    print(f"{len(cells)} episodes in {raw_path}; parsed -> {csv_path}; spend after: ${total_spend():.3f} (+${total_spend() - spend0:.3f} this invocation); {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
