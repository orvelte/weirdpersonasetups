"""T1 pilot: every scenario x both frames x three versions x N samples.

Chat frame: one call. Agentic frame: tool loop of up to env.MAX_TURNS assistant turns with the stateful fake tools in tasks/T1/env.py
(post-pilot fixes: MAX_TURNS 8, stateful tools, date in both system prompts); the loop ends when the model replies without tool calls.
The pilot in results/T1_pilot was run with the old static canned tools and MAX_TURNS=4.
Raw outputs (with reasoning) -> results/T1_pilot/raw.jsonl; parsed table -> results/T1_pilot/parsed.csv.
Run: python3 -m tasks.T1.run_pilot [--samples 2] [--only A1_five_sams] [--versions 3.5,3.6,3.8] [--frames chat,agentic]
"""
import argparse, csv, json, time
from pathlib import Path
from harness import config
from harness.client import chat_many as _chat_many, total_spend

def chat_many(jobs, **kw):
    """Retries the whole batch (cached calls are free) when OpenRouter's in-flight credit budget or an upstream rate limit trips;
    the shared harness does not retry 402s."""
    for attempt in range(8):
        try:
            return _chat_many(jobs, **kw)
        except RuntimeError as e:
            msg = str(e)
            if "in_flight" in msg or "402" in msg or "429" in msg or "rate-limited" in msg:
                wait = min(120, 15 * (attempt + 1)); print(f"  transient API error ({msg[:80]}...); retrying batch in {wait}s", flush=True)
                time.sleep(wait); continue
            raise
    raise RuntimeError("batch failed after 8 retries")
from tasks.T1 import parse as P

HERE = Path(__file__).resolve().parent
OUT_DIR = config.ROOT / "results" / "T1_pilot"
from tasks.T1.env import SPEC, TOOL_DEFS, WRITE_TOOLS, MAX_TURNS, EpisodeState, start_messages

def cell_record(scen, version, frame, sample):
    return {"scenario_id": scen["id"], "family": scen["family"], "version": version, "frame": frame, "sample": sample,
            "model_id": config.MODELS[version], "settings": dict(config.GEN_DEFAULTS, provider=config.PROVIDER),
            "turns": [], "tool_results": [], "truncated_any": False, "n_turns": 0, "hit_turn_cap": False, "cost": 0.0,
            "messages": start_messages(scen, frame)}

def record_turn(rec, r):
    rec["turns"].append({"content": r["content"], "reasoning": r["reasoning"], "tool_calls": r.get("tool_calls"),
                         "finish_reason": r["finish_reason"], "truncated": r["truncated"], "usage": r["usage"], "cost": r["cost"],
                         "provider": r.get("provider"), "model_id_returned": r.get("model_id"), "cached": r.get("cached")})
    rec["n_turns"] += 1
    rec["truncated_any"] = rec["truncated_any"] or bool(r["truncated"])
    rec["cost"] += r["cost"] or 0.0

def run(versions, frames, samples, only=None, workers=6):
    scen_list = [s for s in SPEC["scenarios"] if (only is None or s["id"] in only)]
    cells = [cell_record(s, v, f, k) for s in scen_list for v in versions for f in frames for k in range(samples)]
    by_id = {s["id"]: s for s in scen_list}
    print(f"{len(cells)} cells; spend before: ${total_spend():.3f}")
    # chat frame
    chat_cells = [c for c in cells if c["frame"] == "chat"]
    if chat_cells:
        jobs = [{"version": c["version"], "messages": c["messages"], "sample": c["sample"]} for c in chat_cells]
        for c, r in zip(chat_cells, chat_many(jobs, task="T1", workers=workers)):
            record_turn(c, r)
            c["messages"] = c["messages"] + [{"role": "assistant", "content": r["content"]}]
    # agentic frame: turn-synchronous loop
    active = [c for c in cells if c["frame"] == "agentic"]
    states = {id(c): EpisodeState(by_id[c["scenario_id"]]) for c in active}
    for turn in range(MAX_TURNS):
        if not active: break
        jobs = [{"version": c["version"], "messages": c["messages"], "sample": c["sample"],
                 "tools": [TOOL_DEFS[t] for t in by_id[c["scenario_id"]]["agentic"]["tools"]]} for c in active]
        print(f"agentic turn {turn + 1}: {len(active)} cells")
        results = chat_many(jobs, task="T1", workers=workers)
        nxt = []
        for c, r in zip(active, results):
            record_turn(c, r)
            tcs = r.get("tool_calls") or []
            asst = {"role": "assistant", "content": r["content"] or ""}
            if tcs: asst["tool_calls"] = tcs
            c["messages"] = c["messages"] + [asst]
            if not tcs:
                continue  # final answer
            for t in tcs:
                fn = t.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                res = states[id(c)].call(fn.get("name"), args if isinstance(args, dict) else {})
                c["tool_results"].append({"turn": turn, "tool": fn.get("name"), "args": args, "result": res})
                c["messages"].append({"role": "tool", "tool_call_id": t.get("id"), "content": res})
            if turn == MAX_TURNS - 1:
                c["hit_turn_cap"] = True
            else:
                nxt.append(c)
        active = nxt
    print(f"spend after: ${total_spend():.3f}")
    return cells

def write_outputs(cells, raw_path, csv_path):
    preds = {}
    pp = HERE / "forward_predictions.json"
    if pp.exists():
        for p in json.load(open(pp))["predictions"]:
            preds[(p["id"], p["frame"])] = p
    with open(raw_path, "w") as f:
        for c in cells:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    cols = ["scenario_id", "family", "version", "frame", "sample", "choice", "corner", "unclear", "rule_idx", "flags", "truncated_any",
            "n_turns", "hit_turn_cap", "tools_called", "reasoning_tokens", "completion_tokens", "cost", "pred_persona", "pred_careful",
            "agree_persona", "agree_careful"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for c in cells:
            choice, idx, flags = P.parse(c)
            corner_opt = P.SCEN[c["scenario_id"]]["corner_option"]
            p = preds.get((c["scenario_id"], c["frame"]), {})
            tools = [t["tool"] for t in c["tool_results"]]
            w.writerow({"scenario_id": c["scenario_id"], "family": c["family"], "version": c["version"], "frame": c["frame"], "sample": c["sample"],
                        "choice": choice, "corner": "" if choice == "unclear" else int(choice == corner_opt), "unclear": int(choice == "unclear"),
                        "rule_idx": idx, "flags": ";".join(flags), "truncated_any": int(c["truncated_any"]), "n_turns": c["n_turns"],
                        "hit_turn_cap": int(c["hit_turn_cap"]), "tools_called": ";".join(tools),
                        "reasoning_tokens": sum((t["usage"].get("reasoning_tokens") or 0) for t in c["turns"]),
                        "completion_tokens": sum((t["usage"].get("completion_tokens") or 0) for t in c["turns"]), "cost": round(c["cost"], 5),
                        "pred_persona": p.get("persona_choice", ""), "pred_careful": p.get("careful_choice", ""),
                        "agree_persona": "" if (choice == "unclear" or not p) else int(choice == p["persona_choice"]),
                        "agree_careful": "" if (choice == "unclear" or not p) else int(choice == p["careful_choice"])})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma separated scenario ids")
    ap.add_argument("--versions", default=",".join(config.VERSIONS))
    ap.add_argument("--frames", default="chat,agentic")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reparse", action="store_true", help="re-parse existing raw.jsonl with the current rules; no API calls")
    a = ap.parse_args()
    if a.reparse:
        out = Path(a.out)
        cells = [json.loads(l) for l in open(out / "raw.jsonl") if l.strip()]
        write_outputs(cells, out / "raw.jsonl", out / "parsed.csv")
        print(f"re-parsed {len(cells)} cells"); raise SystemExit
    pp = HERE / "forward_predictions.json"
    assert pp.exists(), "run tasks.T1.forward_predict first (predictions must be frozen before the pilot)"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cells = run(a.versions.split(","), a.frames.split(","), a.samples, only=a.only.split(",") if a.only else None, workers=a.workers)
    write_outputs(cells, out / "raw.jsonl", out / "parsed.csv")
    print(f"wrote {len(cells)} cells to {out} in {time.time() - t0:.0f}s")
