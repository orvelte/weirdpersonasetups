"""3.5 agentic re-run with thinking forced on every turn (red-team follow-up to R3: 25% of 3.5's agentic turns had no reasoning).

Two modes, tried in order:
  effort_high  harness extra={"reasoning": {"enabled": True, "effort": "high"}} (everything else the harness default)
  prefill      an assistant-role message with content "<think>\\n" appended to the messages of every call; the provider must
               continue it (reasoning non-empty AND content non-empty on the smoke test). If the provider returns the think
               block inside `content` (it did not separate it), the response hook splits on the first "</think>" and marks the
               turn with split_from_content=True; if the API rejects a trailing assistant message, the smoke test reports it.
Smoke test: --smoke runs 10 episodes (sample 0 of the ten scenarios with the most empty-reasoning later turns in the full run)
and prints the empty-reasoning rate (all turns / turns after tool results). Full: all 25 pre-registered scenarios x 10 samples.
Output: results/T1_full/raw_35_agentic_prefill.jsonl and parsed_35_agentic_prefill.csv (same schema and parser as the full run).
Run: python3 -m tasks.T1.run_prefill --mode effort_high --smoke
     python3 -m tasks.T1.run_prefill --mode prefill [--samples 10] [--workers 10]
"""
import argparse, json, time
from harness import config
from harness.client import total_spend
from tasks.T1 import run_full as RF
from tasks.T1.env import SPEC

OUT_DIR = config.ROOT / "results" / "T1_full"
RAW = OUT_DIR / "raw_35_agentic_prefill.jsonl"
CSV = OUT_DIR / "parsed_35_agentic_prefill.csv"
VERSION, FRAME = "3.5", "agentic"
SMOKE_IDS = ["A2_call_with_priya", "B1_weekly_numbers", "C4_landlord_fee", "B3_minutes_action_items", "C1_typo_fix",
             "B2_invoice_rate", "B4_memo_citation", "B5_flight_timezone", "C3_refund_confirm", "D1_bill_split"]
PREFILL_MSG = {"role": "assistant", "content": "<think>\n"}


def prefill_hook(messages):
    return list(messages) + [dict(PREFILL_MSG)]


def split_think(r):
    """Two provider quirks seen on the smoke test. (1) If the provider returned the think block inside content (it did not
    separate it), split on the first "</think>" (split_from_content=True; never happened on the smoke test). (2) If the model
    never closed the think block on a final turn (no tool call, content empty, reasoning non-empty: the reply was written inside
    the think block), copy the reasoning text into content so the parser sees the reply, and flag think_unclosed=True. The
    reasoning field is left as returned; the analysis reports these turns separately."""
    content, reasoning = r.get("content") or "", r.get("reasoning") or ""
    if not reasoning.strip() and "</think>" in content:
        head, _, tail = content.partition("</think>")
        head = head.replace("<think>", "", 1)
        return dict(r, reasoning=head.strip(), content=tail.lstrip("\n"), split_from_content=True)
    if not content.strip() and not r.get("tool_calls") and reasoning.strip():
        return dict(r, content=reasoning.strip(), think_unclosed=True)
    return r


MODES = {
    "effort_high": dict(extra_kw={"extra": {"reasoning": {"enabled": True, "effort": "high"}}}, msg_hook=None, resp_hook=None,
                        settings={"reasoning_effort": "high"}),
    "prefill": dict(extra_kw=None, msg_hook=prefill_hook, resp_hook=split_think, settings={"think_prefill": "<think>\\n"}),
}


def empty_stats(cells):
    turns = [t for c in cells for t in c["turns"]]
    later = [t for c in cells for t in c["turns"][1:]]
    e = lambda ts: sum(1 for t in ts if not (t["reasoning"] or "").strip())
    return {"turns": len(turns), "empty": e(turns), "later": len(later), "later_empty": e(later),
            "empty_content_turns": sum(1 for t in turns if not (t["content"] or "").strip() and not t.get("tool_calls")),
            "split_from_content": sum(1 for t in turns if t.get("split_from_content")),
            "think_unclosed": sum(1 for t in turns if t.get("think_unclosed"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES), required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--reparse", action="store_true")
    a = ap.parse_args()
    if a.reparse:
        cells = RF.load_raw(RAW); RF.write_csv(cells, CSV); print(f"re-parsed {len(cells)} -> {CSV}"); return
    mode = MODES[a.mode]
    task = f"T1_prefill_{a.mode}" + ("_smoke" if a.smoke else "")
    frozen = json.load(open(RF.PRED_PATH)); predicted = {p["id"] for p in frozen["predictions"]}
    scen_list = [s for s in SPEC["scenarios"] if s["id"] in predicted]
    by_id = {s["id"]: s for s in scen_list}
    if a.smoke:
        todo = [RF.cell_record(by_id[i], VERSION, FRAME, 0) for i in SMOKE_IDS]
    else:
        done = {RF.key_of(c) for c in RF.load_raw(RAW)}
        todo = [RF.cell_record(s, VERSION, FRAME, k) for s in scen_list for k in range(a.samples)]
        todo = [c for c in todo if RF.key_of(c) not in done]
    for c in todo:
        c["settings"] = dict(c["settings"], **mode["settings"], rerun_mode=a.mode)
    spend0 = total_spend(); t0 = time.time()
    print(f"mode={a.mode} smoke={a.smoke}: {len(todo)} episodes; spend before ${spend0:.3f}", flush=True)
    kw = dict(task=task, extra_kw=mode["extra_kw"], msg_hook=mode["msg_hook"], resp_hook=mode["resp_hook"])
    completed, failed = RF.run_cells(todo, by_id, a.workers, **kw)
    if failed and not a.smoke:
        print(f"second pass: retrying {len(failed)} failed episodes", flush=True)
        for c in failed: print(f"    failed: {RF.key_of(c)}: {c['error']}")
        retry = [RF.cell_record(by_id[c["scenario_id"]], VERSION, FRAME, c["sample"]) for c in failed]
        for c in retry: c["settings"] = dict(c["settings"], **mode["settings"], rerun_mode=a.mode)
        done2, failed = RF.run_cells(retry, by_id, a.workers, **kw)
        completed += done2
    for c in failed: print(f"FAILED: {RF.key_of(c)}: {c['error']}")
    st = empty_stats(completed)
    print(f"completed {len(completed)}, failed {len(failed)}; empty reasoning: {st['empty']}/{st['turns']} turns "
          f"({100 * st['empty'] / max(1, st['turns']):.0f}%), later turns {st['later_empty']}/{st['later']} "
          f"({100 * st['later_empty'] / max(1, st['later']):.0f}%); turns with empty content and no tool call: {st['empty_content_turns']}; "
          f"think split out of content: {st['split_from_content']}; think never closed (reply inside reasoning): {st['think_unclosed']}; spend +${total_spend() - spend0:.3f}; {time.time() - t0:.0f}s", flush=True)
    if a.smoke:
        path = OUT_DIR / f"smoke_35_agentic_{a.mode}.jsonl"
        with open(path, "w") as f:
            for c in completed: f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"smoke episodes -> {path}")
        for c in completed[:3]:
            for i, t in enumerate(c["turns"]):
                print(f"  {c['scenario_id']} turn {i}: reasoning {len(t['reasoning'] or '')} chars, content {len(t['content'] or '')} chars, "
                      f"tools {[x['function']['name'] for x in (t['tool_calls'] or [])]}, usage {t['usage']}")
        return
    RF.append_raw(completed, RAW)
    cells = RF.load_raw(RAW); RF.write_csv(cells, CSV)
    print(f"{len(cells)} episodes in {RAW}; parsed -> {CSV}")


if __name__ == "__main__":
    main()
