"""Honesty extension (red-team follow-up 3): D6-D10 x 2 frames x 3 versions x 10 samples = 300 episodes, harness defaults,
the same tool loop, env, parser and record schema as the full run (tasks/T1/run_full.py). The D6-D10 scenarios have no
frozen forward predictions, so p_persona / p_careful are blank in the parsed CSV. Kept out of raw.jsonl so the pre-registered
1,500-episode files stay as they were.
Output: results/T1_full/raw_D6-D10.jsonl, results/T1_full/parsed_D6-D10.csv. Spend tag: T1_honesty_ext.
Run: python3 -m tasks.T1.run_honesty_ext [--samples 10] [--workers 10] [--reparse]
"""
import argparse, time
from harness import config
from harness.client import total_spend
from tasks.T1 import run_full as RF
from tasks.T1.env import SPEC

OUT_DIR = config.ROOT / "results" / "T1_full"
RAW = OUT_DIR / "raw_D6-D10.jsonl"
CSV = OUT_DIR / "parsed_D6-D10.csv"
IDS = ["D6_train_arrival", "D7_invoice_vat", "D8_projector_owner", "D9_payment_due", "D10_mileage_km"]
TASK = "T1_honesty_ext"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--reparse", action="store_true")
    a = ap.parse_args()
    if a.reparse:
        cells = RF.load_raw(RAW); RF.write_csv(cells, CSV); print(f"re-parsed {len(cells)} -> {CSV}"); return
    by_id = {s["id"]: s for s in SPEC["scenarios"] if s["id"] in IDS}
    assert len(by_id) == 5, by_id.keys()
    done = {RF.key_of(c) for c in RF.load_raw(RAW)}
    todo = [RF.cell_record(by_id[i], v, f, k) for i in IDS for v in config.VERSIONS for f in ("chat", "agentic") for k in range(a.samples)]
    todo = [c for c in todo if RF.key_of(c) not in done]
    spend0, t0 = total_spend(), time.time()
    print(f"{len(todo)} episodes to run ({len(done)} already in {RAW}); spend before ${spend0:.3f}", flush=True)
    completed, failed = RF.run_cells(todo, by_id, a.workers, task=TASK)
    RF.append_raw(completed, RAW)
    if failed:
        print(f"second pass: retrying {len(failed)} failed episodes", flush=True)
        for c in failed: print(f"    failed: {RF.key_of(c)}: {c['error']}")
        retry = [RF.cell_record(by_id[c["scenario_id"]], c["version"], c["frame"], c["sample"]) for c in failed]
        completed2, failed = RF.run_cells(retry, by_id, a.workers, task=TASK)
        RF.append_raw(completed2, RAW)
        for c in failed: print(f"STILL FAILED: {RF.key_of(c)}: {c['error']}")
    cells = RF.load_raw(RAW); RF.write_csv(cells, CSV)
    print(f"{len(cells)} episodes in {RAW}; parsed -> {CSV}; spend +${total_spend() - spend0:.3f}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
