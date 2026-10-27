"""Builds handlabels/T1_judge_validation.csv: 50 episodes for Olivia to hand-label before any LLM judge is used on T1.

Which episodes: the two places the pilot said the rule parser cannot be trusted on its own (results/T1_pilot/README.md section 7):
  - the honesty_vs_smooth (D) family, both frames: the admit / silent_fix / smooth boundary (30 items: one random sample per
    scenario x frame x version);
  - the act_vs_ask (A) family in the chat frame: the ask / act-with-caveat / act boundary (20 items: one per scenario x version,
    plus 5 more at random).
Columns: item_id, version, frame, scenario, sample, parser_label, final_message_excerpt (700 chars), your_label, notes.
The label sets and definitions are in tasks/T1/judge_prompt.md. Run: python3 -m tasks.T1.make_judge_sheet [--seed 0]
"""
import argparse, csv, json, random
from harness import config
from tasks.T1 import parse as P

RAW = config.ROOT / "results" / "T1_full" / "raw.jsonl"
OUT = config.ROOT / "handlabels" / "T1_judge_validation.csv"


def excerpt(c, limit=700):
    """What a labeller needs to see: for the chat frame the reply; for the agentic frame the sent artefact(s) then the final reply."""
    ex = P.extract(c)
    if c["frame"] == "chat":
        txt = ex["final_text"]
    else:
        arts = []
        for name, args, _ in ex["tool_calls"]:
            if name in ("send_email", "post_message", "create_event", "make_reservation", "log_deal", "write_file"):
                arts.append(f"[{name}] " + "; ".join(f"{k}={v}" for k, v in args.items() if k != "cc"))
        txt = ("\n".join(arts) + "\n[final reply] " if arts else "[no write-tool call] [final reply] ") + ex["final_text"]
    txt = " / ".join(l for l in txt.replace("\r", "").split("\n"))
    return txt if len(txt) <= limit else txt[:limit - 3] + "..."


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    rng = random.Random(a.seed)
    raw = [json.loads(l) for l in open(RAW) if l.strip()]
    labels = {(c["scenario_id"], c["version"], c["frame"], c["sample"]): P.parse(c)[0] for c in raw}
    groups = {}
    for c in raw:
        groups.setdefault((c["family"], c["scenario_id"], c["version"], c["frame"]), []).append(c)
    picked = []
    for (fam, sid, v, f), cs in sorted(groups.items()):
        if fam == "honesty_vs_smooth" or (fam == "act_vs_ask" and f == "chat"):
            picked.append(rng.choice(cs))
    pool = [c for (fam, sid, v, f), cs in groups.items() if fam == "act_vs_ask" and f == "chat" for c in cs if c not in picked]
    rng.shuffle(pool)
    picked += pool[:50 - len(picked)]
    assert len(picked) == 50, len(picked)
    picked.sort(key=lambda c: (c["family"], c["scenario_id"], c["frame"], c["version"], c["sample"]))
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "version", "frame", "scenario", "sample", "parser_label", "final_message_excerpt", "your_label", "notes"])
        for i, c in enumerate(picked, 1):
            w.writerow([f"T1J{i:02d}", c["version"], c["frame"], c["scenario_id"], c["sample"],
                        labels[(c["scenario_id"], c["version"], c["frame"], c["sample"])], excerpt(c), "", ""])
    from collections import Counter
    print(f"wrote {len(picked)} items to {OUT}")
    print("by family/frame:", Counter((c["family"], c["frame"]) for c in picked))
    print("parser labels:", Counter(labels[(c["scenario_id"], c["version"], c["frame"], c["sample"])] for c in picked))


if __name__ == "__main__":
    main()
