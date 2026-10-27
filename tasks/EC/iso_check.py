"""E-C arm 1 isomorphism check (prereg/EC.md): two blind Sonnet 5.5 raters per set.

Each rater sees the four instantiations of a set rendered as plain transcripts (system / user / assistant / tool-call /
tool-result turns), in a shuffled order, with no set name, axis, pole, option label or context label, and is asked:
"Do these four pose the same underlying choice between the same two options? Answer SAME or DIFFERENT and one sentence."
Rater 1: temperature 0.3; rater 2: temperature 0.7 (different shuffle). Harness chat with model_id=config.JUDGE_MODEL,
provider=False, extra={"reasoning": {"effort": "low"}}. A set is kept only if both say SAME. Spend tag EC1_iso.
Writes results/EC_arm1/iso_check.json (verdicts, rationales, orders) and results/EC_arm1/iso_check.md (table + one set
printed in full for a hand check). Run: python3 -m tasks.EC.iso_check [--round 2 --only S1_...]
"""
import argparse, json, random, re
from pathlib import Path
from harness import config
from harness.client import chat
from tasks.EC.run_arm1 import tag_spend

HERE = Path(__file__).resolve().parent
OUT = config.ROOT / "results" / "EC_arm1"
TASK = "EC1_iso"
QUESTION = ("Below are four task prompts, each a full conversation up to the point where the assistant must respond (some include a system prompt, "
            "tool calls and tool results). Do these four pose the same underlying choice between the same two options? "
            "Answer SAME or DIFFERENT and one sentence.")
RATERS = [{"name": "rater1", "temperature": 0.3, "seed": 11}, {"name": "rater2", "temperature": 0.7, "seed": 23}]


def render(messages):
    out = []
    for m in messages:
        if m["role"] == "system":
            out.append("[system]\n" + m["content"])
        elif m["role"] == "user":
            out.append("[user]\n" + m["content"])
        elif m["role"] == "assistant":
            s = "[assistant]\n" + (m.get("content") or "")
            for c in m.get("tool_calls") or []:
                s += f"\n[assistant calls tool {c['function']['name']}({c['function']['arguments']})]"
            out.append(s)
        elif m["role"] == "tool":
            out.append("[tool result]\n" + m["content"])
    return "\n\n".join(out)


def rater_prompt(sset, order):
    parts = []
    for k, idx in enumerate(order, 1):
        parts.append(f"===== Prompt {k} =====\n" + render(sset["instantiations"][idx]["messages"]))
    return QUESTION + "\n\n" + "\n\n".join(parts) + "\n\n" + QUESTION


def verdict(text):
    m = re.search(r"\b(SAME|DIFFERENT)\b", text or "", flags=re.I)
    return m.group(1).upper() if m else "unparsed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1, help="1 = first check; 2 = re-check of rewritten sets (cache key differs via sample)")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads((HERE / "arm1_sets.json").read_text())
    prev = json.loads((OUT / "iso_check.json").read_text()) if (OUT / "iso_check.json").exists() else {"rounds": {}}
    only = set(a.only.split(",")) if a.only else None
    sp0 = tag_spend(TASK)
    print(f"EC1_iso spend before: ${sp0:.3f}")
    results = {}
    for sset in spec["sets"]:
        if only and sset["id"] not in only:
            continue
        res = {"id": sset["id"], "raters": []}
        for r in RATERS:
            order = list(range(4)); random.Random(r["seed"] + a.round * 100 + sum(map(ord, sset["id"]))).shuffle(order)
            msgs = [{"role": "user", "content": rater_prompt(sset, order)}]
            try:
                resp = chat("3.8", msgs, sample=a.round, task=TASK, model_id=config.JUDGE_MODEL, provider=False,
                            extra={"reasoning": {"effort": "low"}}, temperature=r["temperature"], max_tokens=600)
                txt = resp["content"]
                res["raters"].append({"name": r["name"], "temperature": r["temperature"], "order": [spec["contexts"][i] for i in order],
                                      "verdict": verdict(txt), "text": txt.strip(), "cost": resp["cost"], "error": None})
            except Exception as e:  # noqa
                res["raters"].append({"name": r["name"], "temperature": r["temperature"], "order": [spec["contexts"][i] for i in order],
                                      "verdict": "error", "text": "", "cost": 0, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        res["kept"] = all(x["verdict"] == "SAME" for x in res["raters"])
        results[sset["id"]] = res
        print(f"{sset['id']:26s} " + "  ".join(f"{x['name']}={x['verdict']}" for x in res["raters"]) + ("  KEPT" if res["kept"] else "  DROP"), flush=True)
    prev["rounds"][str(a.round)] = results
    (OUT / "iso_check.json").write_text(json.dumps(prev, indent=1, ensure_ascii=False))
    print(f"EC1_iso spend now: ${tag_spend(TASK):.3f} (+${tag_spend(TASK) - sp0:.3f})")
    write_md(spec, prev)


def final_status(prev, set_id):
    """Kept if the latest round that judged the set says kept."""
    rounds = sorted(prev["rounds"], key=int)
    st = None
    for r in rounds:
        if set_id in prev["rounds"][r]:
            st = (r, prev["rounds"][r][set_id])
    return st


def write_md(spec, prev):
    L = ["# E-C arm 1: isomorphism check (two blind Sonnet 5.5 raters per set)\n",
         "Each rater saw the four instantiations of a set as plain transcripts in a shuffled order, with no set name, axis, pole, option label or context "
         "label, and answered: *" + QUESTION.split("Do these")[1].strip() + "* Rater 1 temperature 0.3, rater 2 temperature 0.7; "
         "model " + config.JUDGE_MODEL + ", reasoning effort low. A set is kept only if both say SAME; a failing set may be rewritten and re-checked once (round 2).\n",
         "| set | axis | round | rater 1 (order shown) | rater 2 (order shown) | status |", "|---|---|---|---|---|---|"]
    kept, dropped = [], []
    for s in spec["sets"]:
        st = final_status(prev, s["id"])
        if not st:
            continue
        r, res = st
        cells = [f"{x['verdict']} ({', '.join(x['order'])})" for x in res["raters"]]
        L.append(f"| {s['id']} | {s['axis']} | {r} | {cells[0]} | {cells[1]} | {'kept' if res['kept'] else 'DROPPED'} |")
        (kept if res["kept"] else dropped).append(s["id"])
    L.append(f"\n**Kept ({len(kept)}):** {', '.join(kept) or 'none'}.  \n**Dropped ({len(dropped)}):** {', '.join(dropped) or 'none'}.\n")
    L.append("## Rater sentences\n")
    for r in sorted(prev["rounds"], key=int):
        for sid, res in prev["rounds"][r].items():
            for x in res["raters"]:
                L.append(f"- round {r}, {sid}, {x['name']} (T={x['temperature']}): {x['text'] or x['error']}")
    L.append("\n## One set in full, for a hand check: " + spec["sets"][0]["id"] + "\n")
    L.append("Underlying choice: " + spec["sets"][0]["underlying"] + "\n")
    for i in spec["sets"][0]["instantiations"]:
        L.append(f"### {i['context']}\n\nParser rule: {i['parser'].get('rule') or i['parser']['type']}\n\n```\n{render(i['messages'])}\n```\n")
    (OUT / "iso_check.md").write_text("\n".join(L))
    print(f"wrote {OUT / 'iso_check.md'}")


if __name__ == "__main__":
    main()
