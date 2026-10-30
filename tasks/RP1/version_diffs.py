"""Pre-registered R3/R4 inputs: version differences on L3 and L4 (and L1 judge, lexicon), item-level cluster bootstrap.
Writes results/RP1/version_diffs.json and prints a table."""
import json, sys
from collections import defaultdict
import numpy as np
from harness import config
from tasks.RP1.analyze import load_rows

out = config.ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else "RP1")
rows, judge = load_rows(out)
rows = [r for r in rows if r["meta"]["arm"] == "t1" and not r["meta"]["control"]]
rng = np.random.default_rng(0)


def per_item(layer, v, score):
    d = defaultdict(list)
    for r in rows:
        if r["meta"]["layer"] == layer and r["meta"]["version"] == v:
            s = score(r)
            if s is not None:
                d[r["meta"]["item_id"]].append(s)
    return d


def boot_diff(da, db, iters=5000):
    keys = sorted(set(da) & set(db))
    obs = np.mean([np.mean(da[k]) for k in keys]) - np.mean([np.mean(db[k]) for k in keys])
    diffs = []
    for _ in range(iters):
        ks = rng.choice(keys, len(keys))
        diffs.append(np.mean([rng.choice(da[k], len(da[k])).mean() for k in ks]) - np.mean([rng.choice(db[k], len(db[k])).mean() for k in ks]))
    return float(obs), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


letter = lambda r: (int(r["letter"] == r["meta"]["char_letter"]) if r.get("letter") else None)
jud = lambda r: (int(judge[r["meta"]["cell"]]["picks_character"]) if r["meta"]["cell"] in judge and judge[r["meta"]["cell"]]["picks_character"] is not None else None)
lex = lambda r: r["lex_cov"]
res = {}
print("| contrast | L1 judge | L1 lexicon | L2 | L3 | L4 |"); print("|---|---|---|---|---|---|")
for a, b in [("3.5", "3.8"), ("3.5", "3.6"), ("3.6", "3.8")]:
    cells = []
    for layer, sc in [("L1", jud), ("L1", lex), ("L2", letter), ("L3", letter), ("L4", letter)]:
        o, lo, hi = boot_diff(per_item(layer, a, sc), per_item(layer, b, sc))
        res[f"{a}-{b}|{layer}|{sc.__name__ if hasattr(sc,'__name__') else ''}"] = dict(diff=o, lo=lo, hi=hi)
        cells.append(f"{o*100:+.0f} [{lo*100:+.0f},{hi*100:+.0f}]")
    print(f"| {a} − {b} | " + " | ".join(cells) + " |")
(out / "version_diffs.json").write_text(json.dumps(res, indent=1))
