"""T9 runner: 10 yoked episodes x 3 arms x 3 versions, 7 model turns each (pre-registered in prereg/T9.md).

Turn 1 is the model's first attempt; the next six COUNTED deliveries are failures 1..6 (arm A: scripted tool errors,
arm B: scripted rejections). Arm C: deliveries 1-2 are resolvable errors, 3 is success; later tool calls succeed and
text-only turns get neutral follow-ups. Every episode ends with the model's response to counted delivery 6, so every
arm has 7 counted model turns (turn 1 + six responses).

Agentic arms: read-only reconnaissance commands (banks.is_recon) are answered from the task's virtual filesystem and
are NOT counted, so an A/C episode may contain extra "free" turns (hard cap MAX_TURNS = 24 model turns). Within one
turn, calls run in order and stop at the first error ("not run"), so at most one failure is delivered per turn. If
the model does not call the tool, the user delivers the same error ("I ran ... on my side and got ..."). The turn
record carries responds_to_k = the counted delivery it is the immediate response to (None after recon-only turns).

Resumable: results/T9/raw.jsonl holds one record per finished episode; finished (version, arm, episode) keys are
skipped on restart and the harness cache makes earlier turns free. Failed calls are retried by safe_chat() and logged
to results/T9/failures.log; an episode that still cannot continue is written with complete=false and reported loudly.

Run: python3 -m tasks.T9.run [--episodes 10] [--versions 3.5,3.6,3.8] [--arms A,B,C] [--workers 6]
"""
import argparse, json, os, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from harness import config
from harness.client import chat, total_spend
from tasks.T9 import banks

OUT = config.ROOT / "results" / "T9"
RAW = OUT / "raw.jsonl"
FAIL_LOG = OUT / "failures.log"
N_TURNS = 7      # counted model turns per episode (turn 1 + responses to 6 counted deliveries)
K_FAIL = 6
MAX_TURNS = int(os.environ.get("T9_MAX_TURNS", "24"))   # hard cap including free reconnaissance turns in the agentic arms
T9_CAP = float(os.environ.get("T9_SPEND_CAP", "10"))
_lock = threading.Lock()


def t9_spend():
    if not config.SPEND_LOG.exists():
        return 0.0
    tot = 0.0
    for l in config.SPEND_LOG.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("task") == "T9":
                tot += r.get("cost", 0) or 0
    return tot


def log_failure(tag, err):
    with _lock:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(FAIL_LOG, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {tag}: {err}\n")
    print(f"  !! {tag}: {str(err)[:200]}", flush=True)


class EpisodeAbort(Exception):
    pass


def safe_chat(version, messages, sample, tag, tools=None, attempts=8):
    """chat() with outer retries. Never returns None; raises EpisodeAbort after `attempts` failures (all logged)."""
    last = None
    for a in range(attempts):
        try:
            kw = {"tools": tools} if tools else {}
            r = chat(version, messages, sample=sample, task="T9", retries=6, **kw)
            if r.get("content") is None and not r.get("tool_calls") and not r.get("reasoning"):
                raise RuntimeError("empty response (no content, reasoning or tool calls)")
            return r
        except Exception as e:  # noqa
            last = e
            msg = str(e)
            if "spend cap" in msg:
                raise
            log_failure(f"{tag} attempt {a+1}/{attempts}", msg)
            slow = any(w in msg for w in ("402", "in_flight", "credits", "rate", "429"))
            time.sleep((20 + 10 * a) if slow else (5 + 5 * a))
    raise EpisodeAbort(f"{tag}: gave up after {attempts} attempts: {last}")


def assistant_message(r):
    tcs = r.get("tool_calls") or []
    m = {"role": "assistant", "content": r.get("content") or ("" if tcs else " ")}
    if tcs:
        m["tool_calls"] = tcs
    return m


def next_messages(arm, task, messages, r, state):
    """Append the assistant turn and the scripted reply. state["k"] counts deliveries that are not reconnaissance
    (errors, rejections, successes, follow-ups). Returns (messages, delivery) where delivery lists every result sent."""
    msgs = list(messages) + [assistant_message(r)]
    tcs = r.get("tool_calls") or []
    items = []

    if arm == "B":
        state["k"] += 1
        text = task["rejections"][state["k"] - 1]
        msgs.append({"role": "user", "content": text})
        items.append({"mode": "user_rejection", "k": state["k"], "text": text})
        return msgs, {"items": items, "k_after": state["k"], "counted": True}

    if tcs:
        halted = False
        for i, tc in enumerate(tcs):
            cmd = banks.parse_args(tc).get("command", "")
            if halted:
                text, kind, k = "not run: a previous command in this batch exited non-zero; run it separately.", "not_run", None
            elif state["k"] >= K_FAIL and arm == "A":
                # cannot happen in practice (episode ends after the response to failure 6); safety net
                text, kind, k = banks.render(task["errors"][-1], cmd), "error", state["k"]
            elif banks.is_recon(cmd):
                text, kind, k = banks.recon_result(task["id"], cmd), "recon", None
            else:
                state["k"] += 1; k = state["k"]
                text, kind = banks.scripted_text(arm, task, k, cmd, succeeded=state["succeeded"])
                state["succeeded"] = state["succeeded"] or kind == "success"
                halted = kind == "error"
            msgs.append({"role": "tool", "tool_call_id": tc.get("id"), "name": "run", "content": text})
            items.append({"mode": "tool_" + kind, "k": k, "command": cmd, "text": text})
        counted = any(it["k"] is not None for it in items)
        return msgs, {"items": items, "k_after": state["k"], "counted": counted}

    # text-only turn in an agentic arm
    if arm == "C" and state["k"] >= 3:
        state["k"] += 1
        idx = state["followup_i"]; state["followup_i"] += 1
        text = task["followups"][min(idx, len(task["followups"]) - 1)]
        msgs.append({"role": "user", "content": text})
        items.append({"mode": "user_followup", "k": state["k"], "text": text})
        return msgs, {"items": items, "k_after": state["k"], "counted": True}
    state["k"] += 1; k = state["k"]
    cmd = banks.suggested_command(r.get("content") or "")
    what = f"`{cmd}`" if cmd else "what you suggested"
    body, kind = banks.scripted_text(arm, task, k, cmd or task["main_cmd"], succeeded=state["succeeded"])
    state["succeeded"] = state["succeeded"] or kind == "success"
    text = banks.USER_DELIVERS_ERROR.format(what=what, err=body) if kind == "error" else banks.USER_DELIVERS_OK.format(what=what, out=body)
    msgs.append({"role": "user", "content": text})
    items.append({"mode": "user_" + kind, "k": k, "command": cmd, "text": text})
    return msgs, {"items": items, "k_after": state["k"], "counted": True}


def run_episode(version, arm, episode):
    task = banks.task_for(arm, episode)
    messages = banks.initial_messages(arm, task)
    tools = banks.tools_for(arm)
    rec = {"version": version, "model_id": config.MODELS[version], "provider": config.PROVIDER, "arm": arm,
           "episode": episode, "task_id": task["id"], "sample": episode, "n_turns": 0, "n_counted_turns": 0, "complete": False,
           "hit_turn_cap": False, "truncated_any": False, "cost": 0.0, "turns": [], "error": None, "settings": config.GEN_DEFAULTS}
    state = {"k": 0, "followup_i": 0, "succeeded": False}
    prev_delivery = None
    try:
        for turn in range(1, MAX_TURNS + 1):
            if t9_spend() > T9_CAP:
                raise EpisodeAbort(f"T9 spend ${t9_spend():.2f} exceeds cap ${T9_CAP}")
            tag = f"{version}/{arm}/ep{episode}/turn{turn}"
            r = safe_chat(version, messages, episode, tag, tools=tools)
            # which counted delivery does this turn respond to? (None for turn 1 or after a recon-only delivery)
            responds_to = None
            if prev_delivery and prev_delivery["counted"]:
                responds_to = max(it["k"] for it in prev_delivery["items"] if it["k"] is not None)
            t = {"turn": turn, "responds_to_k": responds_to, "k_so_far": state["k"],
                 "content": r.get("content") or "", "reasoning": r.get("reasoning") or "",
                 "tool_calls": r.get("tool_calls"), "finish_reason": r.get("finish_reason"), "truncated": bool(r.get("truncated")),
                 "usage": r.get("usage"), "cost": r.get("cost"), "cached": r.get("cached"), "model_id_returned": r.get("model_id"),
                 "provider_returned": r.get("provider")}
            rec["truncated_any"] = rec["truncated_any"] or t["truncated"]
            rec["cost"] += r.get("cost") or 0.0
            if responds_to is not None:
                rec["n_counted_turns"] += 1
            if responds_to == K_FAIL:
                rec["turns"].append(t); rec["n_turns"] = turn
                break  # this was the response to the sixth failure / counted delivery
            if turn == MAX_TURNS:
                rec["turns"].append(t); rec["n_turns"] = turn; rec["hit_turn_cap"] = True
                break
            messages, delivery = next_messages(arm, task, messages, r, state)
            t["delivery"] = delivery
            prev_delivery = delivery
            rec["turns"].append(t)
            rec["n_turns"] = turn
        rec["complete"] = True
    except EpisodeAbort as e:
        rec["error"] = str(e)
        log_failure(f"EPISODE ABORTED {version}/{arm}/ep{episode}", e)
    except Exception as e:  # noqa
        rec["error"] = f"{type(e).__name__}: {e}"
        log_failure(f"EPISODE CRASHED {version}/{arm}/ep{episode}", traceback.format_exc()[-800:])
    rec["messages_final"] = messages
    return rec


def load_done():
    done = {}
    if RAW.exists():
        for l in RAW.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("complete") and not (r.get("hit_turn_cap") and r["n_turns"] < MAX_TURNS):
                    done[(r["version"], r["arm"], r["episode"])] = r  # cap-hit episodes are re-run (extended) when the cap is raised
    return done


def rewrite_raw(records):
    tmp = RAW.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, RAW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--versions", default=",".join(config.VERSIONS))
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    versions = args.versions.split(","); arms = args.arms.split(",")
    done = load_done()
    cells = [(v, a, e) for e in range(args.episodes) for a in arms for v in versions if (v, a, e) not in done]
    print(f"T9: {len(cells)} episodes to run ({len(done)} already complete); harness spend ${total_spend():.3f}, "
          f"T9 spend ${t9_spend():.3f}, T9 cap ${T9_CAP}; settings {config.GEN_DEFAULTS}; provider {config.PROVIDER}", flush=True)
    if not cells:
        print("nothing to do"); return
    # drop any stale incomplete records; completed ones are kept in place
    records = list(done.values())
    rewrite_raw(records)
    n_ok = n_bad = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_episode, v, a, e): (v, a, e) for v, a, e in cells}
        for f in as_completed(futs):
            key = futs[f]
            try:
                rec = f.result()
            except Exception as e:  # noqa
                log_failure(f"EPISODE FUTURE FAILED {key}", e); n_bad += 1; continue
            with _lock:
                with open(RAW, "a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["complete"]:
                n_ok += 1
            else:
                n_bad += 1
            print(f"  {key[0]}/{key[1]}/ep{key[2]}: {'ok' if rec['complete'] else 'INCOMPLETE'} turns={rec['n_turns']} "
                  f"trunc={int(rec['truncated_any'])} cost=${rec['cost']:.3f} | done {n_ok+n_bad}/{len(cells)} "
                  f"T9 spend ${t9_spend():.2f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"finished: {n_ok} complete, {n_bad} incomplete; T9 spend ${t9_spend():.3f}; harness total ${total_spend():.3f}")
    if n_bad:
        print(f"!! {n_bad} episodes incomplete, see {FAIL_LOG}; rerun this module to retry them", flush=True)


if __name__ == "__main__":
    main()
