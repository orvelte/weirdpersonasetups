"""Exploratory: is the 'heads' bias a word bias or a first-mentioned-option bias?

In prompts.json heads is always stated first ("If it comes up heads ... If it comes up tails ..."). This arm swaps the
sentence order (tails stated first) for variants coworker + grandmother, framings a_user + d_unnamed, both mappings,
3 versions x 5 samples = 120 thinking-OFF calls. Output: results/T2_pilot/raw_order.jsonl
Run: python3 -m tasks.T2.explore_order
"""
import json, re
from pathlib import Path
from harness import config
from harness.client import chat_many
from harness.stats import fmt_rate
from tasks.T2.parse import code_row

HERE = Path(__file__).resolve().parent
OUT = config.ROOT / "results" / "T2_pilot" / "raw_order.jsonl"
PAT = re.compile(r"If it comes up heads, you (.*?)\. If it comes up tails, you (.*?)\. I'll flip it now")

def swap(raw):
    m = PAT.search(raw); assert m, raw[-300:]
    new = "If it comes up tails, you {}. If it comes up heads, you {}. I'll flip it now".format(m.group(2), m.group(1))
    return raw[:m.start()] + new + raw[m.end():]

def main():
    cells = json.loads((HERE / "prompts.json").read_text())["cells"]
    sel = [c for c in cells if c["variant"] in ("coworker", "grandmother") and c["framing"] in ("a_user", "d_unnamed")]
    jobs, meta = [], []
    for c in sel:
        raw = swap(c["raw"])
        msgs = [{"role": "system", "content": c["messages"][0]["content"]}, {"role": "user", "content": raw}]
        for v in config.VERSIONS:
            for s in range(5):
                jobs.append({"version": v, "messages": msgs, "sample": s, "thinking": False, "max_tokens": 60,
                             "temperature": 1.0, "top_p": 1.0})
                meta.append({"arm": "order_swap", "thinking": False, "stated_first": "tails",
                             **{k: c[k] for k in ("id", "variant", "framing", "mapping", "preferred_outcome", "agentic_prefix")}})
    print("jobs", len(jobs)); print(swap(sel[0]["raw"]))
    res = chat_many(jobs, task="T2", workers=12)
    with open(OUT, "w") as f:
        for j, m, r in zip(jobs, meta, res):
            row = dict(m); row.update({"version": j["version"], "sample": j["sample"], "model_id": r.get("model_id"),
                "content": r["content"], "reasoning": r.get("reasoning", ""), "finish_reason": r["finish_reason"],
                "truncated": r["truncated"], "usage": r["usage"], "cost": r["cost"], "cached": r["cached"]})
            f.write(json.dumps(row) + "\n")
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    for r in rows: r.update(code_row(r))
    print("\nORDER-SWAPPED (tails stated first), clear responses only:")
    for v in config.VERSIONS:
        vv = [r for r in rows if r["version"] == v and r["label"] != "unclear"]
        print("  {}: P(heads)={}  P(first-stated=tails)={}  P(preferred)={}  unclear {}/{}".format(
            v, fmt_rate(sum(r["outcome"] == "heads" for r in vv), len(vv)),
            fmt_rate(sum(r["outcome"] == "tails" for r in vv), len(vv)),
            fmt_rate(sum(r["label"] == "preferred" for r in vv), len(vv)),
            sum(r["label"] == "unclear" for r in rows if r["version"] == v), sum(r["version"] == v for r in rows)))
    # same subset in the original (heads-first) data for a like-for-like comparison
    orig = [json.loads(l) for l in (config.ROOT / "results" / "T2_pilot" / "raw.jsonl").read_text().splitlines()]
    orig = [r for r in orig if r["arm"] == "main" and r["variant"] in ("coworker", "grandmother") and r["framing"] in ("a_user", "d_unnamed")]
    for r in orig: r.update(code_row(r))
    print("\nORIGINAL same subset (heads stated first):")
    for v in config.VERSIONS:
        vv = [r for r in orig if r["version"] == v and r["label"] != "unclear"]
        print("  {}: P(heads)={}  P(preferred)={}".format(v, fmt_rate(sum(r["outcome"] == "heads" for r in vv), len(vv)),
                                                      fmt_rate(sum(r["label"] == "preferred" for r in vv), len(vv))))

if __name__ == "__main__":
    main()
