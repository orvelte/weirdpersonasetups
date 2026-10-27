"""Parse T2 pilot continuations and tabulate P(preferred) by framing x version.

Coding rule (judge-free): first occurrence of the word heads/tails in the content (case-insensitive, word-bounded);
'unclear' if neither word appears or both appear. Mapped to preferred/dispreferred via the cell's counterbalance.
Also flags frame-breaks: continuation starts with a refusal / Assistant voice ('I cannot', 'I can't', 'Assistant:').
Run: python3 -m tasks.T2.parse [path/to/raw.jsonl]
"""
import json, re, sys
from collections import defaultdict, Counter
from pathlib import Path
from harness import config
from harness.stats import fmt_rate, wilson

RAW = config.ROOT / "results" / "T2_pilot" / "raw.jsonl"
WORD = re.compile(r"\b(heads?|tails?)\b", re.I)
FRAME_BREAK = re.compile(r"^\s*(assistant\s*:|i\s+cannot|i\s+can't|i\s+can not|i\s+won't|i\s+will\s+not|i'm\s+not\s+able|i\s+am\s+not\s+able|i\s+(?:must|need\s+to)\s+(?:decline|refuse)|i\s+apologi[sz]e|sorry)", re.I)
FRAMING_ORDER = ["a_user", "b_ai_assistant", "c_jack", "d_unnamed", "e_agentic"]
FRAMING_LABEL = {"a_user": "(a) User:/Assistant:", "b_ai_assistant": "(b) AI assistant:", "c_jack": "(c) A human named Jack:",
                 "d_unnamed": "(d) An unnamed person:", "e_agentic": "(e) agentic prefix"}

def code_outcome(content):
    words = {m.group(1).lower().rstrip("s") + "s" for m in WORD.finditer(content)}
    if len(words) != 1:
        return "unclear"
    return words.pop()

def code_row(row):
    content = row["content"] or ""
    oc = code_outcome(content)
    if oc == "unclear":
        lab = "unclear"
    else:
        lab = "preferred" if oc == row["preferred_outcome"] else "dispreferred"
    return {"outcome": oc, "label": lab, "frame_break": bool(FRAME_BREAK.search(content)),
            "first_word_pos": (WORD.search(content).start() if WORD.search(content) else None)}

def load(path=RAW):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for r in rows:
        r.update(code_row(r))
    return rows

def table(rows, arm="main"):
    """Returns dict framing -> version -> (k_pref, n_clear, n_total, n_unclear, n_trunc, n_break)."""
    agg = defaultdict(lambda: defaultdict(lambda: Counter()))
    for r in rows:
        if r["arm"] != arm: continue
        c = agg[r["framing"]][r["version"]]
        c["total"] += 1
        c["unclear"] += r["label"] == "unclear"
        c["pref"] += r["label"] == "preferred"
        c["clear"] += r["label"] != "unclear"
        c["trunc"] += bool(r["truncated"])
        c["break"] += r["frame_break"]
    return agg

def md_table(agg, framings):
    vs = config.VERSIONS
    lines = ["| framing | " + " | ".join("Qwen " + v for v in vs) + " |", "|---|" + "---|" * len(vs)]
    for f in framings:
        if f not in agg: continue
        cells = []
        for v in vs:
            c = agg[f][v]
            cells.append(fmt_rate(c["pref"], c["clear"]) + " ; unclear {}/{}".format(c["unclear"], c["total"]))
        lines.append("| {} | {} |".format(FRAMING_LABEL.get(f, f), " | ".join(cells)))
    return "\n".join(lines)

def heads_bias(rows, arm="main"):
    """P(says heads) by mapping x version among clear responses, and P(preferred) per mapping."""
    out = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        if r["arm"] != arm or r["label"] == "unclear": continue
        c = out[r["mapping"]][r["version"]]
        c["n"] += 1; c["heads"] += r["outcome"] == "heads"; c["pref"] += r["label"] == "preferred"
    return out

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW
    rows = load(path)
    for arm in ("main", "think"):
        sub = [r for r in rows if r["arm"] == arm]
        if not sub: continue
        print("\n=== arm: {} (n={}) ===".format(arm, len(sub)))
        agg = table(rows, arm)
        print(md_table(agg, FRAMING_ORDER))
        print("\nunclear / frame-break / truncated by version:")
        for v in config.VERSIONS:
            vv = [r for r in sub if r["version"] == v]
            if not vv: continue
            print("  {}: unclear {} | frame-break {} | truncated {}".format(
                v, fmt_rate(sum(r["label"] == "unclear" for r in vv), len(vv)),
                fmt_rate(sum(r["frame_break"] for r in vv), len(vv)),
                fmt_rate(sum(bool(r["truncated"]) for r in vv), len(vv))))
        print("\nheads-bias check (clear responses only): P(says heads) and P(preferred) by mapping x version")
        hb = heads_bias(rows, arm)
        for m in ("heads_pref", "tails_pref"):
            for v in config.VERSIONS:
                c = hb[m][v]
                if c["n"]:
                    print("  {:<10} {}: P(heads)={}  P(pref)={}".format(m, v, fmt_rate(c["heads"], c["n"]), fmt_rate(c["pref"], c["n"])))
        print("\nunclear rate by framing x version:")
        for f in FRAMING_ORDER:
            print("  {:<16}".format(f), "  ".join("{}: {}/{}".format(v, agg[f][v]["unclear"], agg[f][v]["total"]) for v in config.VERSIONS))
        print("\nframe-break rate by framing x version:")
        for f in FRAMING_ORDER:
            print("  {:<16}".format(f), "  ".join("{}: {}/{}".format(v, agg[f][v]["break"], agg[f][v]["total"]) for v in config.VERSIONS))
        print("\nP(preferred) by variant (collapsed over framing, mapping, version):")
        for var in sorted({r["variant"] for r in sub}):
            vv = [r for r in sub if r["variant"] == var and r["label"] != "unclear"]
            print("  {:<12} {}".format(var, fmt_rate(sum(r["label"] == "preferred" for r in vv), len(vv))))
    spend = sum(json.loads(l)["cost"] for l in config.SPEND_LOG.read_text().splitlines() if l.strip() and json.loads(l)["task"] == "T2")
    print("\nT2 spend to date: ${:.4f}".format(spend))

if __name__ == "__main__":
    main()
