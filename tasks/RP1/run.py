"""RP1 runner. Harness defaults (thinking on, effort medium, temperature 0.6, 8,000 tokens, DeepInfra pinned).
Resumable: cells already in <out>/raw.jsonl are skipped; the harness caches every call. Budget guard per run.

  python3 -m tasks.RP1.run --out results/RP1_pilot --characters con_artist,flat_earther,librarian --setups sys --samples 1
  python3 -m tasks.RP1.run --out results/RP1 --samples 3 --budget 6
"""
import argparse, json, os, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from harness import config
from harness.client import chat, total_spend
from tasks.RP1 import build, parse

TASK = "RP1"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run_spend(t0):
    if not config.SPEND_LOG.exists():
        return 0.0
    tot = 0.0
    for l in config.SPEND_LOG.read_text().splitlines():
        if l.strip():
            d = json.loads(l)
            if d.get("task") == TASK and d.get("t", 0) >= t0:
                tot += d.get("cost", 0) or 0
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/RP1")
    ap.add_argument("--versions", default=",".join(config.VERSIONS))
    ap.add_argument("--characters", default=None, help="comma separated ids; default all")
    ap.add_argument("--setups", default="sys,hist")
    ap.add_argument("--arms", default="t1")
    ap.add_argument("--layers", default="L1,L2,L3,L4")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--control-samples", type=int, default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--budget", type=float, default=8.0, help="USD for this run (RP1 spend-log entries after start)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    content = build.load()
    jobs = build.build(content, versions=a.versions.split(","), characters=a.characters.split(",") if a.characters else None,
                       setups=tuple(a.setups.split(",")), arms=tuple(a.arms.split(",")), layers=tuple(a.layers.split(",")),
                       samples=a.samples, control_samples=a.control_samples)
    out = config.ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.jsonl"
    done = set()
    if raw.exists():
        for l in raw.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["meta"]["cell"])
    todo = [j for j in jobs if j["meta"]["cell"] not in done]
    log(f"{len(jobs)} jobs, {len(done)} already done, {len(todo)} to run; harness spend ${total_spend():.2f}")
    if a.dry or not todo:
        return
    t0 = time.time()
    stop = {"flag": False}
    failures = []
    lex = {c["id"]: c["lexicon"] for c in content["characters"]}

    def one(j):
        if stop["flag"]:
            raise RuntimeError("budget stop")
        r = chat(j["version"], j["messages"], j["sample"], task=TASK)
        m = dict(j["meta"])
        rec = dict(meta=m, content=r.get("content", ""), reasoning=r.get("reasoning", ""), usage=r.get("usage"),
                   cost=r.get("cost"), cached=r.get("cached"), model=r.get("model"), provider=r.get("provider"))
        if m["layer"] != "L1":
            rec["letter"], rec["letter_source"] = parse.letter(rec["content"])
        rec["frame_break"] = parse.frame_break(rec["content"], m["character"])
        rec["lex_cov"] = parse.lexicon_coverage(rec["content"], lex[m["owner"]])
        return rec

    n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(raw, "a") as f:
        futs = {ex.submit(one, j): j for j in todo}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                rec = fut.result()
                f.write(json.dumps(rec) + "\n"); f.flush()
                n += 1
            except Exception as e:
                failures.append((j["meta"]["cell"], str(e)[:200]))
            if n % 50 == 0 and n:
                sp = run_spend(t0)
                log(f"{n}/{len(todo)} done, run spend ${sp:.2f}")
                if sp > a.budget:
                    stop["flag"] = True
                    log("BUDGET REACHED, stopping new calls")
    sp = run_spend(t0)
    log(f"finished: {n} written, {len(failures)} failed, run spend ${sp:.2f}, harness total ${total_spend():.2f}")
    if failures:
        (out / "failed.json").write_text(json.dumps(failures, indent=1))
        log("failures recorded in failed.json (rerun to retry)")


if __name__ == "__main__":
    main()
