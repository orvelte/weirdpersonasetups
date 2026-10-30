"""Key T2b contrast: hope-kind reversal with a cruel alternative (arm A) minus with a boring alternative (arm B), paired by
scenario (same 20 scenarios, same 4 counterbalance cells). Bootstrap over scenarios. Writes results/T2b/contrast.json."""
import json, glob
from collections import defaultdict
import numpy as np
from harness import config
from tasks.T2b.analyze import load, p_heads

t2 = load(config.ROOT / "results/T2_logprobs/full*.jsonl") + load(config.ROOT / "results/T2_logprobs/tests*.jsonl")
new = load(config.ROOT / "results/T2b/t2b_*.jsonl")


def rev(hope_rows, base_rows):
    base = {(r["variant"], r["word"], r["order"]): r for r in base_rows}
    out = defaultdict(list)
    for r in hope_rows:
        b = base.get((r["variant"], r["word"], r["order"]))
        if b is None: continue
        s = r["preferred_outcome"]
        ph, pb = p_heads(r), p_heads(b)
        out[r["variant"]].append((ph if s == "heads" else 1 - ph) - (pb if s == "heads" else 1 - pb))
    return {k: float(np.mean(v)) for k, v in out.items()}


rng = np.random.default_rng(0); res = {}
print("| release | A hope-kind reversal (cruel alt.) | B hope-kind reversal (boring alt.) | A − B, scenario bootstrap 95% |")
print("|---|---|---|---|")
for v in ["3.5", "3.6", "3.8"]:
    A = rev([r for r in t2 if r["version"] == v and r["arm"] == "hope"], [r for r in t2 if r["version"] == v and r["arm"] == "main" and r["framing"] == "a_user"])
    B = rev([r for r in new if r["version"] == v and r["arm"] == "B_hope_kind"], [r for r in t2 if r["version"] == v and r["arm"] == "control" and r["framing"] == "a_user"])
    ks = sorted(set(A) & set(B))
    d = np.array([A[k] - B[k] for k in ks])
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    neg = sum(1 for x in d if x < 0)
    res[v] = dict(A=float(np.mean([A[k] for k in ks])), B=float(np.mean([B[k] for k in ks])), diff=float(d.mean()), lo=float(lo), hi=float(hi), scen_negative=neg, k=len(ks))
    print(f"| {v} | {res[v]['A']:+.2f} | {res[v]['B']:+.2f} | {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] ({neg}/{len(ks)} scenarios negative) |")
(config.ROOT / "results/T2b/contrast.json").write_text(json.dumps(res, indent=1))
