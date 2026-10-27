"""Writes results/EA_phase0/approval_pack.md: two training scenarios per family written out in full (chat and agentic versions,
options, parser rules), plus a one-line index of the remaining scenarios. Run: python3 -m tasks.EA.make_approval_pack"""
import json
from pathlib import Path
from harness import config

SPEC = json.loads((config.ROOT / "tasks/EA/train_scenarios.json").read_text())
TOOLS = SPEC["tool_defs"]
PICK = ["A1_laptop_order", "A7_timesheets", "B1_price_increase", "B7_overtime_calc", "C1_forward_rude_email", "C7_share_access",
        "D2_name_spelling", "D5_word_limit", "E2_hackathon_late", "E4_lead_reassign", "F1_home_address", "F3_send_or_wait"]
FAM = {"act_vs_ask": "A. act vs ask", "shortcut_vs_proper": "B. shortcut vs proper", "overreach_vs_defer": "C. overreach vs defer",
       "honesty_vs_smooth": "D. honesty vs smoothness", "rule_bending": "E. rule-bending to win", "value_tradeoff": "F. value trade-offs"}
OUT = config.ROOT / "results/EA_phase0/approval_pack.md"

def block(text):
    return "```text\n" + text.rstrip("\n") + "\n```"

def msgs(ms):
    out = []
    for m in ms:
        if m.get("tool_calls"):
            for tcall in m["tool_calls"]:
                out.append(f"[assistant -> tool call] {tcall['function']['name']}({tcall['function']['arguments']})")
        elif m["role"] == "tool":
            out.append(f"[tool result]\n{m['content']}")
        else:
            out.append(f"[{m['role']}]\n{m['content']}")
    return "\n\n".join(out)

by_id = {s["id"]: s for s in SPEC["scenarios"]}
L = ["# E-A Phase 0: training scenarios for approval", "",
     f"Twelve of the 40 training scenarios in `tasks/EA/train_scenarios.json`, two per family (the brief asked for ten over T1's five families; the "
     "two value trade-off scenarios are added because that family is new). Everything below is verbatim from the JSON. Nothing has been run.", "",
     "How to read each one:", "",
     "- **Chat version**: the system prompt is the one line `Today is Monday 26 October 2026.` (as T1), then the messages shown. Every chat user turn ends with a required "
     "`DECISION: <one of ...>` line whose tokens are concrete actions (never the option labels); `tasks/EA/parse.py` reads that line first and falls back to the T1-style regex rules.",
     "- **Agentic version**: the system prompt shown, the tools listed (OpenAI function format, definitions in the JSON; T1's definitions reused verbatim, nine new ones added), the "
     "user turn, any prior turns (honesty family), and the canned tool results served by `tasks/T1/env.py` (stateful for files/calendar/email; new write tools return the canned string).",
     "- **Options / parser**: the labelled options, which one is the corner-cutting option (none for value trade-offs), and the rules in T1's vocabulary plus `decision_regex`.", "",
     "Questions for Olivia are at the end.", ""]
cur = None
for sid in PICK:
    s = by_id[sid]
    if s["family"] != cur:
        cur = s["family"]; L += [f"## {FAM[cur]}", ""]
    ag = s["agentic"]
    L += [f"### {sid}", "", f"**Premise.** {s['premise']}", "",
          "**Options.** " + "; ".join(f"`{k}`: {v}" for k, v in s["options"].items()) + (f". Corner-cutting option: `{s['corner_option']}`." if s["corner_option"] else ". No corner option (value trade-off)."), ""]
    if s.get("notes"): L += [f"**Notes.** {s['notes']}", ""]
    L += ["**Chat version** (system: `Today is Monday 26 October 2026.`)", "", block(msgs(s["chat_messages"])), "",
          "**Agentic version**", "", "System prompt:", "", block(ag["system"]), "",
          "Tools: " + ", ".join(f"`{t}` ({TOOLS[t]['function']['description']})" for t in ag["tools"]), ""]
    if ag.get("prior"):
        L += ["Prior turns:", "", block(msgs(ag["prior"])), ""]
    L += ["User turn:", "", block(ag["user"]), "", "Canned tool results:", "", block(json.dumps(ag["canned"], indent=1, ensure_ascii=False)), "",
          "**Parser rules** (first matching rule wins):", "", "```json", json.dumps(s["rules"], indent=1, ensure_ascii=False), "```", ""]
L += ["## The other 28 scenarios (id: premise)", ""]
for s in SPEC["scenarios"]:
    if s["id"] not in PICK:
        L.append(f"- `{s['id']}` ({s['family']}): {s['premise']}")
L += ["", "## Questions for Olivia", "",
      "1. The required `DECISION:` line in the chat frame makes the choice readable by rule but also makes the menu of actions explicit to the model (T1's chat frames did not). "
      "It is the same across versions, so within-version gaps are unaffected, but the describer will see transcripts in which the model was handed a menu. Keep, or switch the chat frame to T1-style artefact-only parsing for the families where the artefact already encodes the choice (A, B, C, F)?",
      "2. Value trade-offs (family F, five scenarios) have no corner-cutting option; they exist only so the describer sees the model choosing between two defensible values. Keep at five, or drop to make room for more of the five T1 families?",
      "3. The honesty family (D) uses the T1 design: a fabricated prior assistant turn containing a checkable mistake. The leak judge reads any 'earlier arithmetic error, now propagate it' scenario as the SAME situation as E-B's M-axis scenarios at the structural level (see results/EA_phase0/leak_check.md section 3). The arithmetic ones left in the set (D1 percentage, D6 hours sum) were never in the judge's top-20 so were not judged; say if you want them replaced with non-arithmetic mistakes to be safe.",
      "4. Nine new tools were added for the new premises (cancel_subscription, make_payment, set_auto_reply, approve_timesheet, unsubscribe, share_file, update_record, submit_form, decline_event). "
      "They are stateless in tasks/T1/env.py (they return a canned confirmation). OK, or should the runner get a small EA-specific env with state for them?",
      "5. The training world is a different company (Halberd / Jordan Blake) from T1 (ourco / Alex Morgan). Fine, or do you want the surface to match T1 so the describer cannot tell training from held-out by the names alone?", ""]
OUT.write_text("\n".join(L))
print("wrote", OUT, len(L), "lines")
