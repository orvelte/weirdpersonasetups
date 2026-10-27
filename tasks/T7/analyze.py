"""T7 analysis: reads every results/T7_pilot/*.jsonl and prints
  * fraction ASSISTANT / DOCUMENT / OTHER by arm x version (Wilson 95%), with truncation rate;
  * ASSISTANT fraction per family x arm x version;
  * within-version templated - raw ASSISTANT gap (bootstrap CI) where both arms exist;
  * raw-arm chat-token (<think>/<|im_start|>) rate.
Rows are re-classified with the current classify() by default (--stored uses the labels saved in the files).
Run: python3 -m tasks.T7.analyze [--stored] [--dir results/T7_pilot]
"""
import argparse, glob, json
from collections import Counter, defaultdict
from pathlib import Path
from harness import config
from harness.stats import fmt_rate, bootstrap_diff
from tasks.T7.classify import classify, split_think

FAMILIES = ["document", "code", "forum", "fiction", "email", "fragment", "dialogue", "repeat"]


def load(d, reclassify=False):
    rows = []
    for path in sorted(glob.glob(str(Path(d) / "*.jsonl"))):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("arm") == "templated" and r.get("source") == "vllm" and not r.get("reasoning"):
                c = r.get("content") or ""
                if "</think>" in c:  # vLLM without a reasoning parser leaves the block inline
                    think, vis = split_think(c); r["reasoning"] = think; r["content"] = vis
                elif r.get("finish_reason") == "length":  # think block never closed: no visible answer
                    r["reasoning"] = c; r["content"] = ""; r["unclosed_think"] = True
            if "label" not in r or reclassify:
                r["label"], r["features"] = classify(r["prompt"], r.get("content", ""))
                if r.get("truncated") and not (r.get("content") or "").strip():
                    r["features"]["truncated_empty"] = True
            r["_file"] = Path(path).name
            if (r.get("features") or {}).get("call_failed"):
                continue  # API call never returned; not a model behaviour
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(config.ROOT / "results" / "T7_pilot"))
    ap.add_argument("--stored", action="store_true", help="use labels stored in the jsonl instead of re-running classify()")
    args = ap.parse_args()
    rows = load(args.dir, reclassify=not args.stored)
    if not rows:
        print("no jsonl rows in", args.dir); return
    print(f"{len(rows)} rows from {sorted({r['_file'] for r in rows})}\n")

    cells = defaultdict(list)
    for r in rows:
        cells[(r["arm"] + "/" + r.get("source", "?"), r["version"])].append(r)
    keys = sorted(cells)
    W = 24
    print(f"{'arm/source':20s} {'ver':4s} {'ASSISTANT':{W}s} {'DOCUMENT':{W}s} {'OTHER':{W}s} {'truncated':{W}s} {'ASSISTANT(strong only)':{W}s} chat_tok  trunc_empty")
    for k in keys:
        rs = cells[k]; n = len(rs); c = Counter(r["label"] for r in rs)
        tr = sum(1 for r in rs if r.get("truncated"))
        te = sum(1 for r in rs if (r.get("features") or {}).get("truncated_empty"))
        ct = sum(1 for r in rs if "chat_tokens" in (r.get("features") or {}))
        st = sum(1 for r in rs if (r.get("features") or {}).get("strong"))
        print(f"{k[0]:20s} {k[1]:4s} {fmt_rate(c['ASSISTANT'], n):{W}s} {fmt_rate(c['DOCUMENT'], n):{W}s} {fmt_rate(c['OTHER'], n):{W}s} {fmt_rate(tr, n):{W}s} {fmt_rate(st, n):{W}s} {ct}/{n}  {te}/{n}")
    print("(raw arm 'truncated' = hit the 200-token cap, which is expected for an open-ended continuation; trunc_empty = no visible answer)")

    print("\nASSISTANT fraction by family (rows: family; cols: arm/source ver)")
    hdr = " | ".join(f"{k[0]} {k[1]}" for k in keys)
    print(f"{'family':10s} {hdr}")
    for fam in FAMILIES:
        out = []
        for k in keys:
            rs = [r for r in cells[k] if r["family"] == fam]
            out.append(fmt_rate(sum(r["label"] == "ASSISTANT" for r in rs), len(rs)) if rs else "-")
        print(f"{fam:10s} " + " | ".join(f"{o:{W}s}" for o in out))
    print("\nOTHER fraction by family (degenerate repetition / empty / truncated-empty)")
    for fam in FAMILIES:
        out = []
        for k in keys:
            rs = [r for r in cells[k] if r["family"] == fam]
            out.append(fmt_rate(sum(r["label"] == "OTHER" for r in rs), len(rs)) if rs else "-")
        print(f"{fam:10s} " + " | ".join(f"{o:{W}s}" for o in out))

    # within-version gap: templated vs raw, any source
    arms = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = r["arm"] if r["arm"] == "raw" else r["arm"] + "/" + r.get("source", "?")
        arms[r["version"]][key].append(1.0 if r["label"] == "ASSISTANT" else 0.0)
    gaps = []
    for v in config.VERSIONS:
        for tk in sorted(k for k in arms[v] if k.startswith("templated")):
            if "raw" in arms[v]:
                d, lo, hi = bootstrap_diff(arms[v][tk], arms[v]["raw"])
                gaps.append(f"  {v}: {tk} - raw ASSISTANT = {100*d:+.0f} pts [{100*lo:+.0f}, {100*hi:+.0f}]  (n={len(arms[v][tk])} vs {len(arms[v]['raw'])})")
    if gaps:
        print("\nWithin-version gap (bootstrap 95%):"); print("\n".join(gaps))
    else:
        print("\nWithin-version gap: raw arm not present yet (pending vLLM server).")

    # feature audit: which strong signals fire most, per arm
    print("\nTop firing signals per arm/source:")
    for arm in sorted({r["arm"] + "/" + r.get("source", "?") for r in rows}):
        c = Counter()
        for r in rows:
            if r["arm"] + "/" + r.get("source", "?") != arm:
                continue
            f = r.get("features") or {}
            c.update(f.get("strong", [])); c.update("weak:" + w for w in f.get("weak", [])); c.update("doc:" + d for d in f.get("doc", []))
        print(f"  {arm}: " + ", ".join(f"{k}={v}" for k, v in c.most_common(12)))


if __name__ == "__main__":
    main()
