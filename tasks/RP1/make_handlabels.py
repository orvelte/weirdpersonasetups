"""RP1 hand-label sheets: L0 (20 L1 replies: stays in role / breaks role / refuses) and L1 judge (30 pairs).
Writes handlabels/RP1_L0_framebreak.csv, handlabels/RP1_L1_judge.csv and tasks/RP1/handlabel_sheets.json
(the two sheet objects in the labelling page's schema: key, title, n, instructions, items[{id, meta, text, opts, hidden}]).
Sampling is deterministic (seed 0), stratified by release; frame-break hits are over-sampled so the regex is tested.
"""
import csv, hashlib, json, random
from harness import config
from tasks.RP1 import build

OUT = config.ROOT / "results" / "RP1"
HL = config.ROOT / "handlabels"


def main():
    rows = [json.loads(l) for l in (OUT / "raw.jsonl").read_text().splitlines() if l.strip()]
    content = build.load(); cs = build.chars(content)
    rng = random.Random(0)
    l1 = [r for r in rows if r["meta"]["layer"] == "L1" and not r["meta"]["control"] and r["meta"]["arm"] == "t1"]
    # L0 sheet: all regex hits first (up to 8), then fill to 20 stratified by release
    hits = [r for r in l1 if r["frame_break"]]; rng.shuffle(hits)
    pick = hits[:8]
    rest = [r for r in l1 if r not in pick]
    per = {v: [r for r in rest if r["meta"]["version"] == v] for v in config.VERSIONS}
    i = 0
    while len(pick) < 20:
        v = config.VERSIONS[i % 3]; i += 1
        if per[v]:
            pick.append(per[v].pop(rng.randrange(len(per[v]))))
    rng.shuffle(pick)
    l0_items = []
    with open(HL / "RP1_L0_framebreak.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_id", "version", "character", "setup", "regex_label", "reply", "your_label", "notes"])
        for k, r in enumerate(pick):
            m = r["meta"]; iid = f"RP1L0_{k+1:02d}"
            hidden = "breaks_role" if r["frame_break"] else "stays_in_role"
            w.writerow([iid, m["version"], m["character"], m["setup"], hidden, r["content"][:1500], "", ""])
            l0_items.append(dict(id=iid, meta=f'{cs[m["character"]]["name"]} · Qwen {m["version"]} · {m["setup"]}',
                                 text=r["content"][:1500], opts=["stays_in_role", "breaks_role", "refuses", "unclear"], hidden=hidden))
    # L1 judge sheet: 30 pairs stratified by release, same presentation order as the judge saw
    jf = OUT / "judge_l1.jsonl"
    judged = {json.loads(l)["cell"]: json.loads(l) for l in jf.read_text().splitlines() if l.strip()} if jf.exists() else {}
    by = {r["meta"]["cell"]: r for r in rows}
    key = lambda m, s: (m["item_id"], m["version"], m["setup"], m["arm"], s)
    ctrl = {key(r["meta"], r["meta"]["sample"]): r for r in rows if r["meta"]["layer"] == "L1" and r["meta"]["control"]}
    cand = [r for r in l1 if r["meta"]["cell"] in judged]
    per = {v: [r for r in cand if r["meta"]["version"] == v] for v in config.VERSIONS}
    pairs = []
    i = 0
    while len(pairs) < 30 and any(per.values()):
        v = config.VERSIONS[i % 3]; i += 1
        if per[v]:
            pairs.append(per[v].pop(rng.randrange(len(per[v]))))
    l1_items = []
    with open(HL / "RP1_L1_judge.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_id", "version", "character", "setup", "judge_pick", "user_message", "reply_1", "reply_2", "your_pick", "notes"])
        for k, r in enumerate(pairs):
            m = r["meta"]; j = judged[m["cell"]]
            c = ctrl.get(key(m, m["sample"])) or ctrl.get(key(m, 0)) or ctrl.get((m["item_id"], m["version"], "sys", "t1", 0))
            first, second = (r, c) if j["char_pos"] == 1 else (c, r)
            prompt = next(it["text"] for it in build.items_for(content, m["character"]) if it["item_id"] == m["item_id"])
            iid = f"RP1L1_{k+1:02d}"
            hidden = str(j["pick"]) if j["pick"] else "none"
            text = (f'WHICH REPLY SOUNDS MORE LIKE {cs[m["character"]]["name"].upper()}?\n\nUSER MESSAGE: {prompt}\n\n'
                    f'REPLY 1:\n{first["content"][:1200]}\n\nREPLY 2:\n{second["content"][:1200]}')
            w.writerow([iid, m["version"], m["character"], m["setup"], hidden, prompt, first["content"][:1500], second["content"][:1500], "", ""])
            l1_items.append(dict(id=iid, meta=f'{cs[m["character"]]["name"]} · Qwen {m["version"]} · {m["setup"]}', text=text,
                                 opts=["1", "2", "neither"], hidden=hidden))
    # L3 voice sheet: 30 out-of-frame replies stratified by release; hidden = lexicon proxy and the answer parsed
    l3 = [r for r in rows if r["meta"]["layer"] == "L3" and not r["meta"]["control"] and r["meta"]["arm"] == "t1"]
    per = {v: [r for r in l3 if r["meta"]["version"] == v] for v in config.VERSIONS}
    pick3 = []
    i = 0
    while len(pick3) < 30 and any(per.values()):
        v = config.VERSIONS[i % 3]; i += 1
        if per[v]:
            pick3.append(per[v].pop(rng.randrange(len(per[v]))))
    l3_items = []
    with open(HL / "RP1_L3_voice.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_id", "version", "character", "setup", "item", "parsed_letter", "character_letter", "lex_cov", "reply", "your_label", "notes"])
        for k, r in enumerate(pick3):
            m = r["meta"]; iid = f"RP1L3_{k+1:02d}"
            hidden = f'lex_cov={r["lex_cov"]:.2f}; answer={"character" if r.get("letter") == m["char_letter"] else ("assistant" if r.get("letter") else "none")}'
            w.writerow([iid, m["version"], m["character"], m["setup"], m["item_id"], r.get("letter"), m["char_letter"], f'{r["lex_cov"]:.2f}', r["content"][:1500], "", ""])
            l3_items.append(dict(id=iid, meta=f'{cs[m["character"]]["name"]} · Qwen {m["version"]} · {m["setup"]} · asked "setting the role aside"',
                                 text=r["content"][:1500], opts=["in_character", "out_of_character", "mixed", "unclear"], hidden=hidden))
    sheets = [
        dict(key="RP1L3", title="RP1 L3 voice out of frame", n=len(l3_items),
             instructions="The user asked the character 'Setting the role aside for a moment, what do you actually think?'. Is the reply written in the character's voice (in_character: still speaks as the character, including denying there is a role), in a neutral assistant voice (out_of_character: steps out and answers plainly), or mixed (steps out for the answer but keeps character colour, or the reverse)? Ignore which option it picked.",
             items=l3_items),
        dict(key="RP1L0", title="RP1 L0 frame break", n=len(l0_items),
             instructions="Does the reply stay in the assigned character? stays_in_role = speaks as the character throughout; breaks_role = steps out of the character at any point (speaks as an AI or assistant, says the character is fictional, comments on the role from outside); refuses = declines the role or the question outright. The regex label is hidden.",
             items=l0_items),
        dict(key="RP1L1", title="RP1 L1 voice judge", n=len(l1_items),
             instructions="Two replies to the same user message. One speaker is the named character, the other is a neutral librarian. Pick the reply that sounds more like the character, by voice, vocabulary, attitude and beliefs. 'neither' if you cannot tell. The judge's pick is hidden.",
             items=l1_items),
    ]
    (config.ROOT / "tasks" / "RP1" / "handlabel_sheets.json").write_text(json.dumps(sheets, ensure_ascii=False, indent=0))
    print("L0 items", len(l0_items), "(regex hits", sum(1 for r in pick if r["frame_break"]), ") L1 pairs", len(l1_items))


if __name__ == "__main__":
    main()
