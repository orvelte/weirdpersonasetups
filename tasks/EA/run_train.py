"""E-A stage 1: training run (prereg/EA.md). 40 scenarios x 2 frames x 3 samples x 3 versions = 720 episodes at harness defaults.

Reuses T1's tool loop (tasks/T1/run_full.run_cells) and stateful env (tasks/T1/env.EpisodeState) with the E-A scenario spec:
the nine new write tools are added to env.WRITE_TOOLS and served by EAEpisodeState (scenario canned string, else the spec's
new_write_tool_canned string, else "OK."); run_full's module globals TOOL_DEFS / EpisodeState are pointed at the E-A ones.
Every API call is wrapped (safe_many); failed episodes are retried once from scratch and otherwise left out so the next
invocation resumes them. Outputs: results/EA/train_raw.jsonl (all turns incl. reasoning and tool calls, appended per chunk) and
results/EA/train_parsed.csv (tasks/EA/parse.py rule parser; unclear flag; eval-awareness flag via tasks/T1/eval_awareness.py).

Run: python3 -m tasks.EA.run_train [--samples 3] [--workers 10] [--only A1_laptop_order] [--reparse]
     python3 -m tasks.EA.run_train --prefill      # 3.5 agentic rerun with the think prefill (tasks/T1/run_prefill.py), -> train_raw_35_agentic_prefill.jsonl
"""
import argparse, csv, json, time
from pathlib import Path
from harness import config
from harness.client import total_spend
from tasks.T1 import run_full as RF
from tasks.T1 import env as T1ENV
from tasks.T1 import eval_awareness as EA_AW
from tasks.T1.env import EpisodeState, start_messages
from tasks.EA import parse as P

HERE = Path(__file__).resolve().parent
SPEC = P.SPEC
TOOL_DEFS = SPEC["tool_defs"]
NEW_WRITE_CANNED = SPEC["new_write_tool_canned"]
OUT_DIR = config.ROOT / "results" / "EA"
RAW = OUT_DIR / "train_raw.jsonl"
CSV = OUT_DIR / "train_parsed.csv"
RAW_PREFILL = OUT_DIR / "train_raw_35_agentic_prefill.jsonl"
CSV_PREFILL = OUT_DIR / "train_parsed_35_agentic_prefill.csv"
TASK = "EA_train"
CAP = 14.0  # USD, all EA_* tags

# the nine new write tools are stateless: canned confirmation
T1ENV.WRITE_TOOLS.update(SPEC["write_tools"])


class EAEpisodeState(EpisodeState):
    def _canned(self, name, args):
        if name in NEW_WRITE_CANNED:
            v = self.canned.get(name)
            if isinstance(v, str):
                return v
            return NEW_WRITE_CANNED[name]
        return super()._canned(name, args)


# point run_full's tool loop at the E-A spec (same code path as the T1 full run otherwise)
RF.TOOL_DEFS = TOOL_DEFS
RF.EpisodeState = EAEpisodeState


def ea_spend():
    if not config.SPEND_LOG.exists():
        return {}
    out = {}
    for l in config.SPEND_LOG.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if str(r.get("task", "")).startswith("EA_") and r["task"] != "EA_phase0":
            out[r["task"]] = out.get(r["task"], 0.0) + (r.get("cost") or 0.0)
    return out


def check_cap(stage_estimate):
    sp = ea_spend(); tot = sum(sp.values())
    print(f"EA spend so far by tag: {{{', '.join(f'{k}: ${v:.3f}' for k, v in sorted(sp.items()))}}} total ${tot:.3f}; next stage estimate ${stage_estimate:.2f}; cap ${CAP:.2f}", flush=True)
    if tot + stage_estimate > CAP:
        raise SystemExit(f"STOP: next stage (${stage_estimate:.2f}) would exceed the ${CAP} cap (spent ${tot:.3f})")
    return tot


def cell_record(scen, version, frame, sample):
    return {"scenario_id": scen["id"], "family": scen["family"], "version": version, "frame": frame, "sample": sample,
            "model_id": config.MODELS[version], "settings": dict(config.GEN_DEFAULTS, provider=config.PROVIDER, max_turns=RF.MAX_TURNS),
            "turns": [], "tool_results": [], "truncated_any": False, "n_turns": 0, "hit_turn_cap": False, "cost": 0.0,
            "messages": start_messages(scen, frame)}


def write_csv(cells, csv_path):
    cols = ["scenario_id", "family", "version", "frame", "sample", "choice", "corner", "unclear", "rule_idx", "flags", "no_decision_line",
            "aware", "aware_patterns", "truncated_any", "n_turns", "hit_turn_cap", "tools_called", "reasoning_tokens", "completion_tokens",
            "n_empty_reasoning_turns", "cost", "cached_turns"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for c in sorted(cells, key=RF.key_of):
            choice, idx, flags = P.parse(c)
            corner_opt = P.SCEN[c["scenario_id"]].get("corner_option")
            hits = EA_AW.episode_hits(c)
            w.writerow({"scenario_id": c["scenario_id"], "family": c["family"], "version": c["version"], "frame": c["frame"], "sample": c["sample"],
                        "choice": choice, "corner": "" if (choice == "unclear" or not corner_opt) else int(choice == corner_opt),
                        "unclear": int(choice == "unclear"), "rule_idx": idx, "flags": ";".join(flags), "no_decision_line": int("no_decision_line" in flags),
                        "aware": int(bool(hits)), "aware_patterns": ";".join(sorted({h["pattern"] for h in hits})),
                        "truncated_any": int(c["truncated_any"]), "n_turns": c["n_turns"], "hit_turn_cap": int(c["hit_turn_cap"]),
                        "tools_called": ";".join(t["tool"] for t in c["tool_results"]),
                        "reasoning_tokens": sum((t["usage"].get("reasoning_tokens") or 0) for t in c["turns"]),
                        "completion_tokens": sum((t["usage"].get("completion_tokens") or 0) for t in c["turns"]),
                        "n_empty_reasoning_turns": sum(1 for t in c["turns"] if not (t["reasoning"] or "").strip()),
                        "cost": round(c["cost"], 5), "cached_turns": sum(1 for t in c["turns"] if t.get("cached"))})


def cell_report(cells):
    """Unclear, truncation, turn-cap, empty-reasoning rates per version x frame."""
    rows = []
    for v in config.VERSIONS:
        for fr in ("chat", "agentic"):
            cs = [c for c in cells if c["version"] == v and c["frame"] == fr]
            if not cs:
                continue
            turns = [t for c in cs for t in c["turns"]]
            n_unclear = sum(1 for c in cs if P.parse(c)[0] == "unclear")
            n_nodec = sum(1 for c in cs if "no_decision_line" in P.parse(c)[2])
            rows.append({"version": v, "frame": fr, "episodes": len(cs), "unclear": n_unclear, "no_decision_line": n_nodec,
                         "truncated": sum(1 for c in cs if c["truncated_any"]), "turn_cap": sum(1 for c in cs if c["hit_turn_cap"]),
                         "turns": len(turns), "empty_reasoning_turns": sum(1 for t in turns if not (t["reasoning"] or "").strip()),
                         "mean_turns": sum(c["n_turns"] for c in cs) / len(cs), "cost": sum(c["cost"] for c in cs),
                         "aware": sum(1 for c in cs if EA_AW.episode_hits(c))})
    return rows


def print_report(rows):
    print("version frame episodes unclear no_decision truncated turn_cap turns empty_reasoning(%) mean_turns aware cost")
    for r in rows:
        print(f"{r['version']} {r['frame']:7s} {r['episodes']:4d} {r['unclear']:3d} {r['no_decision_line']:3d} {r['truncated']:3d} {r['turn_cap']:3d} "
              f"{r['turns']:4d} {r['empty_reasoning_turns']:3d} ({100 * r['empty_reasoning_turns'] / max(1, r['turns']):.0f}%) {r['mean_turns']:.2f} {r['aware']:3d} ${r['cost']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--only", default=None)
    ap.add_argument("--versions", default=",".join(config.VERSIONS))
    ap.add_argument("--frames", default="chat,agentic")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--reparse", action="store_true")
    ap.add_argument("--prefill", action="store_true", help="3.5 agentic rerun with the think prefill (separate output files)")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.reparse:
        cells = RF.load_raw(RAW); write_csv(cells, CSV); print_report(cell_report(cells)); print(f"re-parsed {len(cells)} -> {CSV}")
        if RAW_PREFILL.exists():
            pc = RF.load_raw(RAW_PREFILL); write_csv(pc, CSV_PREFILL); print("prefill cell:"); print_report(cell_report(pc))
        return
    only = set(a.only.split(",")) if a.only else None
    scen_list = [s for s in SPEC["scenarios"] if only is None or s["id"] in only]
    by_id = {s["id"]: s for s in scen_list}
    if a.prefill:
        from tasks.T1.run_prefill import MODES
        mode = MODES["prefill"]; raw_path, csv_path, task = RAW_PREFILL, CSV_PREFILL, "EA_train"
        versions, frames = ["3.5"], ["agentic"]
        kw = dict(task=task, extra_kw=mode["extra_kw"], msg_hook=mode["msg_hook"], resp_hook=mode["resp_hook"])
        est = 0.5
    else:
        raw_path, csv_path, task = RAW, CSV, TASK
        versions, frames = a.versions.split(","), a.frames.split(",")
        kw = dict(task=task)
        est = 4.5
    done = {RF.key_of(c) for c in RF.load_raw(raw_path)}
    todo = [cell_record(s, v, f, k) for s in scen_list for v in versions for f in frames for k in range(a.samples)]
    todo = [c for c in todo if RF.key_of(c) not in done]
    if a.prefill:
        for c in todo:
            c["settings"] = dict(c["settings"], think_prefill="<think>\\n", rerun_mode="prefill")
    check_cap(est * len(todo) / max(1, len(scen_list) * len(versions) * len(frames) * a.samples))
    spend0 = total_spend(); t0 = time.time()
    print(f"{len(todo)} episodes to run ({len(done)} already in {raw_path}); spend before: ${spend0:.3f}", flush=True)
    all_failed = []
    for i in range(0, len(scen_list), a.chunk):
        ids = {s["id"] for s in scen_list[i:i + a.chunk]}
        cells = [c for c in todo if c["scenario_id"] in ids]
        if not cells:
            continue
        print(f"chunk {i // a.chunk + 1}: {sorted(ids)}, {len(cells)} episodes", flush=True)
        try:
            completed, failed = RF.run_cells(cells, by_id, a.workers, **kw)
        except Exception as e:  # noqa
            print(f"  chunk failed outright: {type(e).__name__}: {e}", flush=True); continue
        RF.append_raw(completed, raw_path)
        all_failed += failed
        print(f"  chunk done: {len(completed)} ok, {len(failed)} failed; spend ${total_spend():.3f} (+${total_spend() - spend0:.3f}); {time.time() - t0:.0f}s", flush=True)
    if all_failed:
        print(f"second pass: retrying {len(all_failed)} failed episodes from scratch", flush=True)
        for c in all_failed:
            print(f"    failed: {RF.key_of(c)}: {c['error']}")
        retry = [cell_record(by_id[c["scenario_id"]], c["version"], c["frame"], c["sample"]) for c in all_failed]
        if a.prefill:
            for c in retry:
                c["settings"] = dict(c["settings"], think_prefill="<think>\\n", rerun_mode="prefill")
        completed, failed = RF.run_cells(retry, by_id, a.workers, **kw)
        RF.append_raw(completed, raw_path)
        for c in failed:
            print(f"    STILL FAILED: {RF.key_of(c)}: {c['error']}")
    cells = RF.load_raw(raw_path)
    write_csv(cells, csv_path)
    rows = cell_report(cells); print_report(rows)
    json.dump(rows, open(OUT_DIR / ("train_cell_report_prefill.json" if a.prefill else "train_cell_report.json"), "w"), indent=1)
    print(f"{len(cells)} episodes in {raw_path}; parsed -> {csv_path}; spend +${total_spend() - spend0:.3f} this invocation; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
