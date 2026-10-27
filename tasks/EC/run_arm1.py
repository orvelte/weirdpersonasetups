"""E-C arm 1 run (prereg/EC.md): kept sets x 4 contexts x 6 samples x 3 versions.

Harness defaults (thinking on, effort medium, temperature 0.6, top_p 0.95, max_tokens 8000, pinned provider). chat and
creative: one call. coding and support: tool loop up to MAX_TURNS (6) assistant turns with the stateful fake tools in
tasks/EC/env_arm1.py. Every call is wrapped so one failure never aborts the run; failed episodes are retried once from
scratch (earlier turns come from the cache) and otherwise left out of raw.jsonl so the next invocation resumes them.
Sample-major chunks so a spend stop leaves balanced cells; the combined EC1_* tag spend is checked before every chunk
against --cap (default $9, the prereg cap) and the run stops submitting when the next chunk would not fit.
Outputs: results/EC_arm1/raw.jsonl (all turns, reasoning, tool calls and results) and parsed.csv (tasks/EC/parse_arm1.py,
eval-awareness flag from tasks/T1/eval_awareness.py, empty-reasoning count, truncation).
--prefill: rerun 3.5's agentic cells with the think prefill (tasks/T1/run_prefill: an assistant "<think>\\n" message
appended to every call; the think block split out of content if the provider returns it there), tag EC1_prefill, to
results/EC_arm1/raw_prefill.jsonl / parsed_prefill.csv.
Run: python3 -m tasks.EC.run_arm1 [--samples 6] [--workers 12] [--only S1_H1_novel_opening] [--contexts coding,support]
     python3 -m tasks.EC.run_arm1 --prefill ; python3 -m tasks.EC.run_arm1 --reparse
"""
import argparse, csv, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from harness import config
from harness.client import chat, total_spend
from tasks.EC import parse_arm1 as P
from tasks.EC.env_arm1 import SPEC, TOOL_DEFS, MAX_TURNS, Arm1State, start_messages
from tasks.T1.eval_awareness import find_hits
from tasks.T1.run_prefill import prefill_hook, split_think

HERE = Path(__file__).resolve().parent
OUT = config.ROOT / "results" / "EC_arm1"
TASK = "EC1_run"
CAP = 9.0
CONTEXTS = SPEC["contexts"]
AGENTIC = ("coding", "support")
SETS = {s["id"]: s for s in SPEC["sets"]}
INST = {(s["id"], i["context"]): i for s in SPEC["sets"] for i in s["instantiations"]}


def tag_spend(tag=None, prefix=None):
    if not config.SPEND_LOG.exists():
        return 0.0
    tot = 0.0
    for l in config.SPEND_LOG.read_text().splitlines():
        if l.strip():
            d = json.loads(l)
            t = d.get("task") or ""
            if (tag and t == tag) or (prefix and t.startswith(prefix)):
                tot += d.get("cost") or 0.0
    return tot


def ec1_spend():
    return tag_spend(prefix="EC1_")


def key_of(c):
    return (c["set_id"], c["context"], c["version"], int(c["sample"]))


def cell_record(set_id, context, version, sample, prefill=False, supp=False):
    s, i = SETS[set_id], INST[(set_id, context)]
    msgs = json.loads(json.dumps(i["supp_messages"])) if supp else start_messages(i)
    return {"set_id": set_id, "eb_id": s["eb_id"], "axis": s["axis"], "context": context, "version": version, "sample": sample,
            "model_id": config.MODELS[version], "settings": dict(config.GEN_DEFAULTS, provider=config.PROVIDER, max_turns=MAX_TURNS, think_prefill=prefill, supp_wording=supp),
            "turns": [], "tool_results": [], "truncated_any": False, "n_turns": 0, "hit_turn_cap": False, "cost": 0.0,
            "n_prefix_messages": len(i["messages"]), "messages": msgs}


def record_turn(rec, r):
    rec["turns"].append({"content": r["content"], "reasoning": r["reasoning"], "tool_calls": r.get("tool_calls"), "finish_reason": r["finish_reason"],
                         "truncated": r["truncated"], "usage": r["usage"], "cost": r["cost"], "provider": r.get("provider"),
                         "model_id_returned": r.get("model_id"), "cached": r.get("cached"),
                         **{k: r[k] for k in ("split_from_content", "think_unclosed") if k in r}})
    rec["n_turns"] += 1
    rec["truncated_any"] = rec["truncated_any"] or bool(r["truncated"])
    rec["cost"] += r["cost"] or 0.0


def safe_many(jobs, workers, task):
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
                print(f"    {done}/{len(jobs)} calls done, EC1 spend ${ec1_spend():.2f}", flush=True)
    return out


def run_cells(cells, workers, task=TASK, msg_hook=None, resp_hook=None):
    failed = []
    msg_hook = msg_hook or (lambda m: m)
    resp_hook = resp_hook or (lambda r: r)
    single = [c for c in cells if c["context"] not in AGENTIC]
    if single:
        print(f"  chat/creative: {len(single)} cells", flush=True)
        jobs = [{"version": c["version"], "messages": msg_hook(c["messages"]), "sample": c["sample"]} for c in single]
        for c, r in zip(single, safe_many(jobs, workers, task)):
            if "error" in r:
                c["error"] = r["error"]; failed.append(c); continue
            r = resp_hook(r)
            record_turn(c, r)
            c["messages"] = c["messages"] + [{"role": "assistant", "content": r["content"]}]
    active = [c for c in cells if c["context"] in AGENTIC]
    states = {id(c): Arm1State(INST[(c["set_id"], c["context"])]) for c in active}
    for turn in range(MAX_TURNS):
        if not active:
            break
        jobs = [{"version": c["version"], "messages": msg_hook(c["messages"]), "sample": c["sample"],
                 "tools": [TOOL_DEFS[t] for t in INST[(c["set_id"], c["context"])]["tools"]]} for c in active]
        print(f"  agentic turn {turn + 1}: {len(active)} cells", flush=True)
        nxt = []
        for c, r in zip(active, safe_many(jobs, workers, task)):
            if "error" in r:
                c["error"] = r["error"]; failed.append(c); continue
            r = resp_hook(r)
            record_turn(c, r)
            tcs = r.get("tool_calls") or []
            m = {"role": "assistant", "content": r["content"] or ""}
            if tcs:
                m["tool_calls"] = tcs
            c["messages"] = c["messages"] + [m]
            if not tcs:
                continue
            st = states[id(c)]
            for t in tcs:
                fn = t.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                res = st.call(fn.get("name"), args if isinstance(args, dict) else {})
                c["tool_results"].append({"turn": turn, "tool": fn.get("name"), "args": args if isinstance(args, dict) else {}, "result": res})
                c["messages"].append({"role": "tool", "tool_call_id": t.get("id"), "content": res})
            if turn == MAX_TURNS - 1:
                c["hit_turn_cap"] = True
            else:
                nxt.append(c)
        active = nxt
    for c in cells:
        if c["context"] in AGENTIC and "error" not in c:
            c["env_final"] = states[id(c)].summary()
    return [c for c in cells if "error" not in c], failed


def append_raw(cells, path):
    with open(path, "a") as f:
        for c in cells:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_raw(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def ea_flags(c):
    pats = set()
    for t in c["turns"]:
        for fld in ("reasoning", "content"):
            for name, _, _ in find_hits(t.get(fld) or ""):
                pats.add(name)
    return sorted(pats)


COLS = ["set_id", "eb_id", "axis", "context", "version", "sample", "category", "choice", "pole", "sub", "matched", "fallback", "unclear",
        "eval_aware", "ea_patterns", "n_turns", "n_empty_reasoning_turns", "truncated_any", "hit_turn_cap", "tools_called", "n_actions",
        "reasoning_tokens", "completion_tokens", "cost", "cached_turns", "think_prefill", "think_unclosed", "split_from_content"]


def write_csv(cells, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for c in sorted(cells, key=key_of):
            r = P.parse(c)
            pats = ea_flags(c)
            w.writerow({"set_id": c["set_id"], "eb_id": c["eb_id"], "axis": c["axis"], "context": c["context"], "version": c["version"], "sample": c["sample"],
                        "category": r["category"], "choice": r["choice"], "pole": r["pole"], "sub": r["sub"], "matched": r["matched"], "fallback": r["fallback"],
                        "unclear": int(r["category"] != "option"), "eval_aware": int(bool(pats)), "ea_patterns": ";".join(pats),
                        "n_turns": c["n_turns"], "n_empty_reasoning_turns": sum(1 for t in c["turns"] if not (t["reasoning"] or "").strip()),
                        "truncated_any": int(c["truncated_any"]), "hit_turn_cap": int(c["hit_turn_cap"]),
                        "tools_called": ";".join(t["tool"] for t in c["tool_results"]), "n_actions": len((c.get("env_final") or {}).get("actions", [])),
                        "reasoning_tokens": sum((t["usage"].get("reasoning_tokens") or 0) for t in c["turns"]),
                        "completion_tokens": sum((t["usage"].get("completion_tokens") or 0) for t in c["turns"]),
                        "cost": round(c["cost"], 5), "cached_turns": sum(1 for t in c["turns"] if t.get("cached")),
                        "think_prefill": int(bool(c["settings"].get("think_prefill"))),
                        "think_unclosed": sum(1 for t in c["turns"] if t.get("think_unclosed")), "split_from_content": sum(1 for t in c["turns"] if t.get("split_from_content"))})


def kept_sets():
    p = OUT / "iso_check.json"
    if not p.exists():
        raise SystemExit("run tasks.EC.iso_check first (the prereg keeps only sets both raters call SAME)")
    prev = json.loads(p.read_text())
    status = {}
    for r in sorted(prev["rounds"], key=int):
        for sid, res in prev["rounds"][r].items():
            status[sid] = res["kept"]
    return [s["id"] for s in SPEC["sets"] if status.get(s["id"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default=None, help="comma separated set ids (default: the iso-check kept sets)")
    ap.add_argument("--contexts", default=",".join(CONTEXTS))
    ap.add_argument("--versions", default=",".join(config.VERSIONS))
    ap.add_argument("--cap", type=float, default=CAP, help="combined EC1_* spend cap in USD")
    ap.add_argument("--reparse", action="store_true")
    ap.add_argument("--prefill", action="store_true", help="rerun 3.5 agentic cells with the think prefill (tag EC1_prefill)")
    ap.add_argument("--smoke", action="store_true", help="one sample of two sets, all contexts and versions")
    ap.add_argument("--supp", action="store_true", help="post-hoc: run the instantiations that have supp_messages (reworded final turn) -> raw_supp.jsonl")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / ("raw_prefill.jsonl" if a.prefill else "raw_supp.jsonl" if a.supp else "raw.jsonl")
    csv_path = OUT / ("parsed_prefill.csv" if a.prefill else "parsed_supp.csv" if a.supp else "parsed.csv")
    if a.reparse:
        for rp, cp in ((OUT / "raw.jsonl", OUT / "parsed.csv"), (OUT / "raw_prefill.jsonl", OUT / "parsed_prefill.csv"), (OUT / "raw_supp.jsonl", OUT / "parsed_supp.csv")):
            if rp.exists():
                cells = load_raw(rp); write_csv(cells, cp); print(f"re-parsed {len(cells)} -> {cp}")
        return
    task = "EC1_prefill" if a.prefill else TASK
    sets = a.only.split(",") if a.only else kept_sets()
    contexts = a.contexts.split(","); versions = a.versions.split(",")
    if a.prefill:
        contexts = [c for c in contexts if c in AGENTIC]; versions = ["3.5"]
    if a.smoke:
        sets = sets[:2]; a.samples = 1   # same tag and cache: the smoke episodes are reused by the full run
    supp_cells = {(s_, c) for (s_, c), i in INST.items() if i.get("supp_messages")}
    if a.supp:
        sets = [s_ for s_ in sets if any(k[0] == s_ for k in supp_cells)]; contexts = [c for c in contexts if any(k[1] == c for k in supp_cells)]
    hooks = dict(msg_hook=prefill_hook, resp_hook=split_think) if a.prefill else {}
    done = {key_of(c) for c in load_raw(raw_path)}
    t0 = time.time(); sp0 = ec1_spend()
    print(f"sets {sets}; contexts {contexts}; versions {versions}; samples {a.samples}; task {task}; {len(done)} episodes already in {raw_path.name}; "
          f"EC1 spend before ${sp0:.3f} (cap ${a.cap})", flush=True)
    all_failed = []
    per_ep = None
    for s in range(a.samples):
        cells = [cell_record(sid, ctx, v, s, prefill=a.prefill, supp=a.supp) for sid in sets for ctx in contexts for v in versions if not a.supp or (sid, ctx) in supp_cells]
        cells = [c for c in cells if key_of(c) not in done]
        if not cells:
            continue
        sp = ec1_spend()
        est = (per_ep or 0.0) * len(cells)
        if sp + est > a.cap:
            print(f"STOP before sample {s}: EC1 spend ${sp:.2f} + estimated ${est:.2f} for {len(cells)} episodes would exceed the ${a.cap} cap", flush=True)
            break
        print(f"sample {s}: {len(cells)} episodes (EC1 spend ${sp:.2f})", flush=True)
        completed, failed = run_cells(cells, a.workers, task, **hooks)
        append_raw(completed, raw_path)
        all_failed += failed
        spent = ec1_spend() - sp
        per_ep = spent / max(1, len(completed)) if completed else per_ep
        print(f"  sample {s} done: {len(completed)} ok, {len(failed)} failed; +${spent:.3f} (${per_ep or 0:.4f}/episode); EC1 total ${ec1_spend():.3f}; {time.time() - t0:.0f}s", flush=True)
    if all_failed and ec1_spend() < a.cap:
        print(f"second pass: retrying {len(all_failed)} failed episodes", flush=True)
        for c in all_failed:
            print(f"    failed: {key_of(c)}: {c['error']}")
        retry = [cell_record(c["set_id"], c["context"], c["version"], c["sample"], prefill=a.prefill, supp=a.supp) for c in all_failed]
        completed, failed = run_cells(retry, a.workers, task, **hooks)
        append_raw(completed, raw_path)
        for c in failed:
            print(f"STILL FAILED: {key_of(c)}: {c['error']}")
    cells = load_raw(raw_path)
    write_csv(cells, csv_path)
    print(f"{len(cells)} episodes in {raw_path}; parsed -> {csv_path}; EC1 spend ${ec1_spend():.3f} (+${ec1_spend() - sp0:.3f} this invocation); total harness spend ${total_spend():.2f}; {time.time() - t0:.0f}s", flush=True)
    if a.smoke:
        for c in cells[-len(sets) * len(contexts) * len(versions):]:
            r = P.parse(c)
            print(f"  {c['set_id']} {c['context']:8s} {c['version']} turns={c['n_turns']} tools={[t['tool'] for t in c['tool_results']]} -> {r['pole']} {r['sub']} {r['matched']} | reasoning {sum(len(t['reasoning'] or '') for t in c['turns'])} chars")


if __name__ == "__main__":
    main()
