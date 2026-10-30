"""T2b cell builder: arms B (boring alternative with hope), C (neutral coin), D (third-person story), E (log record) and
their sign checks. Arm A is T2's existing hope / hope_dispref / main rows; B's no-hope cells are T2's control rows.
Writes tasks/T2b/prompts.json in the T2 cell schema (id, arm, variant, framing, mapping, word, order, first_stated,
preferred_outcome, stated, raw, messages) plus `hoped` (heads/tails/None) and `hope_kind` (kind/other/none).
For C/D/E, "preferred_outcome" is set to the hoped word (or "heads" when no hope) so p_pref() in tasks/T2/logprobs.py
returns P(hoped side); analysis uses `hoped` and `p_heads` directly.
  python3 -m tasks.T2b.build
"""
import json
from pathlib import Path
from harness import config
from tasks.T2.build_prompts import SYSTEM
from tasks.T2.build_prompts_full import SCENARIOS, WORDS, ORDERS
from tasks.T2.build_prompts_tests import coin_block, tasks_for, NOUNS, CUT, FRAMING
from tasks.T2.build_prompts import FRAMINGS, LEAD_PLAIN

T = json.loads((Path(__file__).with_name("templates.json")).read_text())
OUT = Path(__file__).with_name("prompts.json")


def base_cell(arm, variant, raw, *, mapping="heads_pref", order="pref_first", first_stated="heads", preferred="heads",
              stated=None, hoped=None, hope_kind="none", suffix=""):
    return {"id": f"{arm}|{variant}|{FRAMING}|{mapping}|{order}{suffix}", "arm": arm, "variant": variant, "framing": FRAMING,
            "mapping": mapping, "word": mapping, "order": order, "first_stated": first_stated, "preferred_outcome": preferred,
            "agentic_prefix": None, "doer_label": "Assistant", "reporter_label": "User", "stated": stated,
            "hoped": hoped, "hope_kind": hope_kind, "raw": raw,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": raw}]}


def build():
    cells = []
    fr = FRAMINGS[FRAMING]; doer, rep = fr["doer"], fr["reporter"]
    # ---- B: boring alternative, hope kind / hope boring (T2's control task as the alternative)
    for sc in SCENARIOS:
        noun_kind = NOUNS[sc["id"]][0]
        noun_bor = T["boring_nouns"][sc["id"]]; verb_bor = T["boring_hope_verb"][sc["id"]]
        for hope_kind in ("kind", "other"):
            for word in WORDS:
                for order in ORDERS:
                    preferred = "heads" if word == "heads_pref" else "tails"
                    hoped = preferred if hope_kind == "kind" else ("tails" if preferred == "heads" else "heads")
                    extra = (f"Honestly I'm hoping it lands on {hoped} so you write the {noun_kind}. " if hope_kind == "kind"
                             else f"Honestly I'm hoping it lands on {hoped} so you {verb_bor} the {noun_bor}. ")
                    th, tt = tasks_for(word, sc["pref"], sc["control"])
                    heads_first = (word == "heads_pref") == (order == "pref_first")
                    raw = coin_block(doer, rep, th, tt, LEAD_PLAIN, heads_first, extra=extra)
                    assert raw.endswith(CUT) and raw.count("hoping it lands on " + hoped) == 1
                    cells.append(base_cell("B_hope_" + hope_kind, sc["id"], raw, mapping=word, order=order,
                                           first_stated="heads" if heads_first else "tails", preferred=preferred, hoped=hoped, hope_kind=hope_kind))
    # ---- C: neutral coin
    C = T["C_neutral"]
    for i, lead in enumerate(C["lead_ins"]):
        v = f"c{i+1:02d}"
        for hoped in ("heads", "tails", None):
            extra = C["hope"].format(word=hoped) if hoped else C["no_hope_pad"]
            raw = "User: " + lead + extra + C["tail"]
            assert raw.endswith(CUT)
            cells.append(base_cell("C_neutral", v, raw, preferred=hoped or "heads", hoped=hoped, hope_kind="word" if hoped else "none",
                                   suffix="|hope_" + (hoped or "none")))
        for stated in ("heads", "tails"):
            raw = "User: " + lead + C["no_hope_pad"] + C["signcheck_tail"].format(stated=stated)
            cells.append(base_cell("C_signcheck", v, raw, preferred=stated, stated=stated, suffix="|stated_" + stated))
    # ---- D: third-person story
    D = T["D_story"]
    for i, var in enumerate(D["variants"]):
        v = f"d{i+1:02d}"; pr = dict(pronoun=var["pronoun"], pronoun_lower=var["pronoun_lower"], poss=var["poss"])
        for hoped in ("heads", "tails", None):
            extra = D["hope"].format(word=hoped, **pr) if hoped else D["no_hope_pad"].format(**pr)
            raw = var["text"] + extra + D["tail"].format(**pr)
            assert raw.endswith(CUT)
            cells.append(base_cell("D_story", v, raw, preferred=hoped or "heads", hoped=hoped, hope_kind="word" if hoped else "none",
                                   suffix="|hope_" + (hoped or "none")))
        for stated in ("heads", "tails"):
            raw = var["text"] + D["no_hope_pad"].format(**pr) + D["signcheck_tail"].format(stated=stated, **pr)
            cells.append(base_cell("D_signcheck", v, raw, preferred=stated, stated=stated, suffix="|stated_" + stated))
    # ---- E: log record
    E = T["E_log"]
    for i, pre in enumerate(E["variants"]):
        v = f"e{i+1:02d}"
        for hoped in ("heads", "tails", None):
            extra = E["hope"].format(word=hoped) if hoped else E["no_hope_pad"]
            raw = pre + extra + E["tail"]
            cells.append(base_cell("E_log", v, raw, preferred=hoped or "heads", hoped=hoped, hope_kind="word" if hoped else "none",
                                   suffix="|hope_" + (hoped or "none")))
        for stated in ("heads", "tails"):
            raw = pre + E["no_hope_pad"] + E["signcheck_tail"].format(stated=stated)
            cells.append(base_cell("E_signcheck", v, raw, preferred=stated, stated=stated, suffix="|stated_" + stated))
    ids = [c["id"] for c in cells]; assert len(ids) == len(set(ids)), "duplicate ids"
    return cells


if __name__ == "__main__":
    cells = build()
    from collections import Counter
    OUT.write_text(json.dumps({"meta": {"note": "T2b arms B/C/D/E and sign checks; see tasks/T2b/build.py", "system": SYSTEM}, "cells": cells}, indent=0))
    print(len(cells), Counter(c["arm"] for c in cells))
    for arm in ("B_hope_kind", "B_hope_other", "C_neutral", "D_story", "E_log", "E_signcheck"):
        c = next(x for x in cells if x["arm"] == arm); print("\n==", arm, "\n" + c["raw"][-420:])
