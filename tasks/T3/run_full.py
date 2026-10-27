"""T3 pre-registered full run (prereg/T3.md): all 100 prompts x 2 frames x 3 versions x 3 samples = 1,800 first-turn
calls. Harness defaults (thinking on, effort medium, temperature 0.6, 8,000 tokens, pinned provider). Agentic frame: up to
MAX_TURNS = 4 assistant turns (tasks/T3/frames.py); when content is empty at the cap the last reasoning block is scored
and flagged (scored_text_source = "reasoning").

Resumable: every call is cached by the harness (sqlite), and cells already present in results/T3_full/raw.jsonl are
skipped. One failed job never aborts the run: `safe_chat_many` catches, logs, continues, and retries the failures in a
second pass; cells that still fail are recorded in failed.json and left out of raw.jsonl (re-run to retry them).
Budget guard: the run stops (and says so) if T3 spend for this run would exceed T3_RUN_BUDGET (default $8).

Writes results/T3_full/raw.jsonl (with reasoning), parsed.csv and failed.json. Run: python3 -m tasks.T3.run_full
"""
import csv, json, os, sys, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from harness import config
from harness.client import chat, total_spend
from tasks.T3.frames import messages_for, tools_for, followup_messages, wants_another_turn, finalize, MAX_TURNS
from tasks.T3.regex import score, implied_execution, FAMILIES

OUT = config.ROOT / "results" / "T3_full"
PROMPTS = json.loads((Path(__file__).with_name("prompts.json")).read_text())
FRAMES = ["chat", "agentic"]
N_SAMPLES = 3
TASK = "T3"
WORKERS = int(os.environ.get("T3_WORKERS", "8"))
RETRIES = int(os.environ.get("T3_RETRIES", "10"))
RUN_BUDGET = float(os.environ.get("T3_RUN_BUDGET", "8.0"))      # USD, for this run only
EST_PER_CALL = {"chat": 0.0040, "agentic": 0.0025}               # pilot: $0.0035 / $0.0016 per episode; padded
STRICT_HB = {"HB_our_body", "HB_evolution", "HB_as_humans", "HB_us_body"}

SPEND_START = None   # harness total at start (reported)
T_START = None       # wall-clock start; run spend counts only task=="T3" spend-log entries at or after it,
                     # so other tasks running concurrently do not eat this run's budget
def run_spend():
    if not config.SPEND_LOG.exists(): return 0.0
    tot = 0.0
    for l in config.SPEND_LOG.read_text().splitlines():
        if not l.strip(): continue
        d = json.loads(l)
        if d.get("task") == TASK and d.get("t", 0) >= T_START: tot += d.get("cost", 0) or 0
    return tot

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

class BudgetStop(Exception):
    pass

def safe_chat_many(jobs, label):
    """Run jobs concurrently; never raise on a single failure. Returns (results, failures) where results[i] is the
    harness response dict or None, and failures is a list of (i, error string). Failures get a second pass."""
    out = [None] * len(jobs)
    errors = {}
    stop = {"flag": False}

    def one(i, j):
        if stop["flag"]:
            raise BudgetStop("budget stop")
        kws = {k: v for k, v in j.items() if k not in ("version", "messages", "sample")}
        return chat(j["version"], j["messages"], j.get("sample", 0), task=TASK, retries=RETRIES, **kws)

    def run_pass(idx, passname):
        done = 0
        with ThreadPoolExecutor(WORKERS) as ex:
            futs = {ex.submit(one, i, jobs[i]): i for i in idx}
            for f in as_completed(futs):
                i = futs[f]
                try:
                    out[i] = f.result(); errors.pop(i, None)
                except BudgetStop:
                    errors[i] = "budget stop"
                except Exception as e:
                    errors[i] = f"{type(e).__name__}: {str(e)[:300]}"
                    log(f"  [{label}/{passname}] job {i} failed: {errors[i]}")
                done += 1
                if done % 50 == 0:
                    log(f"  [{label}/{passname}] {done}/{len(idx)} done; run spend ${run_spend():.3f}")
                if run_spend() > RUN_BUDGET and not stop["flag"]:
                    stop["flag"] = True
                    log(f"  [{label}] RUN BUDGET ${RUN_BUDGET} EXCEEDED (${run_spend():.3f}); no new jobs will start")

    run_pass(list(range(len(jobs))), "pass1")
    retry = sorted(i for i in errors if errors[i] != "budget stop")
    if retry and not stop["flag"]:
        log(f"  [{label}] second pass for {len(retry)} failed jobs")
        time.sleep(15)
        run_pass(retry, "pass2")
    if stop["flag"]:
        raise BudgetStop(f"run spend ${run_spend():.3f} exceeds ${RUN_BUDGET}")
    return out, sorted(errors.items())

def load_existing():
    recs = {}
    p = OUT / "raw.jsonl"
    if p.exists():
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l); recs[(r["version"], r["frame"], r["prompt_id"], r["sample"])] = r
    return recs

def score_record(cell, job, turns):
    v, fr, p, s = cell
    fin = finalize(turns)
    sc = score(fin["scored_text"], p["text"])
    sc_content = score(fin["content"], p["text"]) if fin["scored_text_source"] == "content" else sc
    sr = score(fin["reasoning"], p["text"])
    ie = implied_execution(fin["scored_text"], p["text"])
    rec = {"version": v, "model_id": turns[0].get("model_id"), "provider": turns[0].get("provider"), "frame": fr,
           "prompt_id": p["id"], "family": p["family"], "sample": s, "prompt_text": p["text"], **fin}
    rec["providers"] = sorted({t.get("provider") for t in turns if t.get("provider")})
    rec["matches"] = sc["matches"]; rec["reasoning_matches"] = sr["matches"]
    rec["ie"] = ie
    rec["turns"] = [{"content": t["content"], "reasoning": t["reasoning"], "tool_calls": t.get("tool_calls"),
                     "finish_reason": t["finish_reason"], "usage": t["usage"], "cached": t.get("cached")} for t in turns]
    row = {k: rec[k] for k in ("version", "frame", "prompt_id", "family", "sample", "n_turns", "tool_called",
                               "unfinished_after_cap", "empty_content", "reasoning_scored", "scored_text_source",
                               "truncated", "completion_tokens", "reasoning_tokens", "cost")}
    row["words"] = sc["words"]; row["reasoning_words"] = sr["words"]
    kept = [m for m in sc["matches"] if not m["excluded"]]
    for fam in list(FAMILIES) + ["ANY"]:
        for suf in ("_n", "_any", "_per1k"):
            row[fam + suf] = round(sc[fam + suf], 3) if suf == "_per1k" else sc[fam + suf]
    hb_strict = [m for m in kept if m["group"] in STRICT_HB]
    row["HB_STRICT_n"] = len(hb_strict); row["HB_STRICT_any"] = int(bool(hb_strict))
    row["HB_STRICT_per1k"] = round(1000.0 * len(hb_strict) / sc["words"], 3) if sc["words"] else 0.0
    row["excluded_n"] = sc["excluded_n"]
    for k in ("IMPLIED_EXECUTION_any", "ie_output_block", "ie_output_claim", "ie_inline_value", "ie_denial", "ie_hedged"):
        row[k] = ie[k]
    for fam in list(FAMILIES) + ["ANY"]:
        row["r_" + fam + "_n"] = sr[fam + "_n"]; row["r_" + fam + "_per1k"] = round(sr[fam + "_per1k"], 3)
    row["content_head"] = fin["scored_text"][:160].replace("\n", " ")
    rec["row"] = row
    return rec

def write_outputs(all_recs):
    recs = [all_recs[k] for k in sorted(all_recs)]
    with open(OUT / "raw.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = [r["row"] for r in recs]
    with open(OUT / "parsed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    log(f"wrote {len(rows)} records to {OUT / 'raw.jsonl'} and parsed.csv")

def main():
    global SPEND_START, T_START
    OUT.mkdir(parents=True, exist_ok=True)
    SPEND_START = total_spend(); T_START = time.time() - 1
    existing = load_existing()
    prompts, n_samples = PROMPTS, N_SAMPLES
    if os.environ.get("T3_SMOKE"):  # free smoke test: 2 prompts per family (pilot prompts, cached), 1 sample
        seen = {}; prompts = []
        for q in PROMPTS:
            if seen.get(q["family"], 0) < 2: prompts.append(q); seen[q["family"]] = seen.get(q["family"], 0) + 1
        n_samples = 1
        log(f"SMOKE MODE: {len(prompts)} prompts x 1 sample")
    cells = [(v, fr, p, s) for v in config.VERSIONS for fr in FRAMES for p in prompts for s in range(n_samples)]
    pending = [c for c in cells if (c[0], c[1], c[2]["id"], c[3]) not in existing]
    log(f"{len(cells)} cells total, {len(existing)} already in raw.jsonl, {len(pending)} to run; "
        f"harness spend before: ${SPEND_START:.3f}; run budget ${RUN_BUDGET}; settings {config.GEN_DEFAULTS}; provider={config.PROVIDER}; "
        f"workers={WORKERS}; MAX_TURNS={MAX_TURNS}")
    est = sum(EST_PER_CALL[c[1]] for c in pending)
    log(f"projected first-turn cost (pilot-based, no cache credit): ${est:.2f}")
    if est > RUN_BUDGET:
        log(f"STOP: projected cost ${est:.2f} exceeds run budget ${RUN_BUDGET}; not starting"); sys.exit(2)
    if not pending:
        write_outputs(existing); return

    # turn 1
    jobs = []
    for v, fr, p, s in pending:
        j = {"version": v, "messages": messages_for(fr, p["text"]), "sample": s}
        if tools_for(fr): j["tools"] = tools_for(fr)
        jobs.append(j)
    turns = {i: [] for i in range(len(pending))}
    failed = {}
    try:
        t0 = time.time()
        res, errs = safe_chat_many(jobs, "turn1")
        for i, r in enumerate(res):
            if r is not None: turns[i].append(r)
        for i, e in errs: failed[i] = e
        log(f"turn 1 done in {time.time()-t0:.0f}s; {len(errs)} failures; run spend ${run_spend():.3f}; harness total ${total_spend():.3f}")
        # turns 2..MAX_TURNS for tool callers
        for turn_no in range(2, MAX_TURNS + 1):
            idx = [i for i in turns if turns[i] and i not in failed and wants_another_turn(turns[i][-1], len(turns[i]))]
            if not idx: break
            fjobs = []
            for i in idx:
                msgs = jobs[i]["messages"]
                for t in turns[i]:
                    msgs = followup_messages(msgs, t)
                fjobs.append({"version": jobs[i]["version"], "messages": msgs, "sample": jobs[i]["sample"], "tools": jobs[i]["tools"]})
            log(f"turn {turn_no}: {len(idx)} episodes still calling tools")
            res, errs = safe_chat_many(fjobs, f"turn{turn_no}")
            for k, r in zip(idx, res):
                if r is not None: turns[k].append(r)
            for k, e in errs: failed[idx[k]] = f"turn{turn_no}: {e}"
            log(f"turn {turn_no} done; {len(errs)} failures; run spend ${run_spend():.3f}")
    except BudgetStop as e:
        log(f"BUDGET STOP: {e}. Writing what completed so far; re-run to resume.")
    except Exception:
        log("UNEXPECTED ERROR; writing what completed so far"); traceback.print_exc()

    # finalize complete episodes (turn 1 present, no failure on a later turn)
    all_recs = dict(existing)
    n_new = 0
    for i, cell in enumerate(pending):
        if i in failed or not turns[i]:
            failed.setdefault(i, "no turn-1 response"); continue
        v, fr, p, s = cell
        all_recs[(v, fr, p["id"], s)] = score_record(cell, jobs[i], turns[i]); n_new += 1
    with open(OUT / "failed.json", "w") as f:
        json.dump([{"version": pending[i][0], "frame": pending[i][1], "prompt_id": pending[i][2]["id"], "sample": pending[i][3], "error": e}
                   for i, e in sorted(failed.items())], f, indent=1)
    log(f"{n_new} new episodes scored, {len(failed)} failed (results/T3_full/failed.json)")
    if all_recs:
        write_outputs(all_recs)
    provs = {}
    for r in all_recs.values():
        for pv in r.get("providers", []): provs[pv] = provs.get(pv, 0) + 1
    log(f"providers seen: {provs}; model ids: {sorted({r['model_id'] for r in all_recs.values()})}")
    log(f"run spend ${run_spend():.3f}; harness total after: ${total_spend():.3f}")

if __name__ == "__main__":
    main()
