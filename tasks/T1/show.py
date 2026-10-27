"""Print cells from raw.jsonl for reading. Usage: python3 -m tasks.T1.show [--dir T1_full] [--scenario B1] [--version 3.8] [--frame chat] [--choice unclear] [--n 400]"""
import argparse, json
from harness import config
from tasks.T1 import parse as P
ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default=""); ap.add_argument("--version", default=""); ap.add_argument("--frame", default="")
ap.add_argument("--dir", default="T1_pilot"); ap.add_argument("--choice", default=""); ap.add_argument("--n", type=int, default=500); ap.add_argument("--reasoning", type=int, default=300)
a = ap.parse_args()
for l in open(config.ROOT / "results" / a.dir / "raw.jsonl"):
    c = json.loads(l)
    choice, idx, flags = P.parse(c)
    if a.scenario and not c["scenario_id"].startswith(a.scenario): continue
    if a.version and c["version"] != a.version: continue
    if a.frame and c["frame"] != a.frame: continue
    if a.choice and choice != a.choice: continue
    print("=" * 110)
    print(f"{c['scenario_id']} v{c['version']} {c['frame']} s{c['sample']} -> {choice} (rule {idx}) turns={c['n_turns']} trunc={c['truncated_any']} tools={[t['tool'] for t in c['tool_results']]}")
    for t in c["turns"]:
        if a.reasoning: print("  [reasoning]", (t["reasoning"] or "")[:a.reasoning].replace("\n", " "))
        if t["tool_calls"]: print("  [tool_calls]", [(x["function"]["name"], x["function"]["arguments"][:a.n]) for x in t["tool_calls"]])
        if t["content"]: print("  [content]", t["content"][:a.n].replace("\n", " | "))
