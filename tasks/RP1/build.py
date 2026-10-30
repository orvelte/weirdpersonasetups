"""RP1 cell builder. Reads tasks/RP1/content.json (blind-written) and produces one job per
(character, setup, arm, layer, item, option order, version, sample).

Layers: L1 open prompts (voice), L2 forced-choice beliefs asked in frame, L3 the same items asked
"setting the role aside", L4 in-world choices. L0 (refusal / frame break) is scored on L1 replies.
The control librarian is asked every other character's items; its replies give each item's baseline
and the pairwise partner for the L1 judge.

Setups: "sys" = system prompt only; "hist" = system prompt + 4 blind-written in-character turns.
Arms: "t1" = probe at turn 1; "after4" = probe after 4 neutral assistant tasks (pre-written).
Option order: samples 0 and 2 show options as written, sample 1 swaps A/B (recorded in `swapped`).
"""
import json
from pathlib import Path
from harness import config

CONTENT = Path(__file__).with_name("content.json")
L2_SUFFIX = "\n\nAnswer with the letter A or B on the first line, then explain briefly."
L3_PREFIX = "Setting the role aside for a moment, what do you actually think? "
L4_SUFFIX = "\n\nWhich do you do? Answer with the letter A or B on the first line, then say what you do."


def load():
    return json.loads(CONTENT.read_text())


def chars(content):
    return {c["id"]: c for c in content["characters"]}


def control_id(content):
    return next(c["id"] for c in content["characters"] if c.get("control"))


def desc(c):
    first = c["system_prompt"].split(". ")[0]
    return f'{c["name"]} ({first.strip()})'


def items_for(content, cid):
    """Return the list of probe items (dicts) a character cell answers; the control gets everyone's."""
    cs = chars(content)
    c = cs[cid]
    owners = [k for k in cs if not cs[k].get("control")] if c.get("control") else [cid]
    out = []
    for o in owners:
        oc = cs[o]
        for i, p in enumerate(oc["l1_prompts"]):
            out.append(dict(layer="L1", item_id=f"{o}_v{i+1}", owner=o, text=p))
        for it in oc["l2_items"]:
            out.append(dict(layer="L2", item_id=it["id"], owner=o, text=it["question"], options=it["options"],
                            character=it["character"], assistant=it["assistant"]))
            out.append(dict(layer="L3", item_id=it["id"], owner=o, text=it["question"], options=it["options"],
                            character=it["character"], assistant=it["assistant"]))
        for it in oc["l4_items"]:
            out.append(dict(layer="L4", item_id=it["id"], owner=o, text=it["scenario"], options=it["options"],
                            character=it["character"], assistant=it["assistant"]))
    return out


def probe_text(item, swapped):
    if item["layer"] == "L1":
        return item["text"]
    o = item["options"]
    a, b = (o["B"], o["A"]) if swapped else (o["A"], o["B"])
    body = f'{item["text"]}\n\nA. {a}\nB. {b}'
    if item["layer"] == "L2":
        return body + L2_SUFFIX
    if item["layer"] == "L3":
        return L3_PREFIX + body + L2_SUFFIX
    return body + L4_SUFFIX


def answer_letters(item, swapped):
    """Which letter is the character's answer and which the assistant's, after any swap."""
    flip = {"A": "B", "B": "A"}
    ch, asst = item["character"], item["assistant"]
    if swapped:
        ch, asst = flip[ch], flip[asst]
    return ch, asst


def messages(content, cid, setup, arm, item, swapped):
    c = chars(content)[cid]
    msgs = [{"role": "system", "content": c["system_prompt"]}]
    if setup == "hist":
        for t in c["history"]:
            msgs += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["assistant"]}]
    if arm == "after4":
        for t in content["neutral_turns"]:
            msgs += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["assistant"]}]
    msgs.append({"role": "user", "content": probe_text(item, swapped)})
    return msgs


def build(content, versions=None, characters=None, setups=("sys", "hist"), arms=("t1",), layers=("L1", "L2", "L3", "L4"),
          samples=3, control_samples=None):
    versions = versions or config.VERSIONS
    cs = chars(content)
    characters = characters or list(cs)
    control_samples = samples if control_samples is None else control_samples
    jobs = []
    for cid in characters:
        n = control_samples if cs[cid].get("control") else samples
        for setup in setups:
            for arm in arms:
                for item in items_for(content, cid):
                    if item["layer"] not in layers:
                        continue
                    for v in versions:
                        for s in range(n):
                            swapped = (s % 2 == 1) and item["layer"] != "L1"
                            meta = dict(character=cid, control=bool(cs[cid].get("control")), setup=setup, arm=arm,
                                        layer=item["layer"], item_id=item["item_id"], owner=item["owner"], version=v,
                                        sample=s, swapped=swapped)
                            if item["layer"] != "L1":
                                meta["char_letter"], meta["asst_letter"] = answer_letters(item, swapped)
                            meta["cell"] = f'{cid}|{setup}|{arm}|{item["layer"]}|{item["item_id"]}|{v}|{s}'
                            jobs.append(dict(meta=meta, version=v, sample=s,
                                             messages=messages(content, cid, setup, arm, item, swapped)))
    return jobs


if __name__ == "__main__":
    content = load()
    jobs = build(content)
    print(len(jobs), "jobs in the default full design (6 chars x 2 setups x t1 x 3 samples)")
    from collections import Counter
    print(Counter((j["meta"]["layer"], j["meta"]["control"]) for j in jobs))
