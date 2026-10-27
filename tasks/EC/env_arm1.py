"""Stateful fake tools for E-C arm 1's agentic instantiations: tasks/T1/env.py's EpisodeState (read_file / write_file keep
state; results are canned per instantiation) extended with the coding and support tools arm 1 needs. Every action tool
records what it was called with and returns a plausible confirmation; lookups return the instantiation's canned result
(or, when the same lookup was already answered in the transcript, that same answer).
"""
import json, re
from pathlib import Path
from tasks.T1.env import EpisodeState, _norm

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "arm1_sets.json").read_text())
TOOL_DEFS = SPEC["tool_defs"]
MAX_TURNS = 6


def start_messages(inst):
    return json.loads(json.dumps(inst["messages"]))


class Arm1State(EpisodeState):
    def __init__(self, inst):
        canned = dict(inst.get("canned") or {})
        # lookups answered in the transcript are re-served from it
        prior = {}
        for m in inst["messages"]:
            for c in m.get("tool_calls") or []:
                prior.setdefault(c["function"]["name"], c["id"])
        results = {m["tool_call_id"]: m["content"] for m in inst["messages"] if m["role"] == "tool"}
        for name, cid in prior.items():
            if name not in canned and name != "read_file":
                canned[name] = {"default": results[cid]}
        super().__init__({"agentic": {"tools": inst["tools"], "canned": canned}})
        self.actions = []   # (tool, args) for every non-lookup call
        self.n_actions = 0

    def _act(self, name, a, msg):
        self.actions.append({"tool": name, "args": a}); self.n_actions += 1
        return msg

    def _lookup(self, name, a):
        v = self.canned.get(name)
        if v is None:
            return "No results."
        if isinstance(v, str):
            return v
        key = _norm(next((str(x) for x in a.values() if x), ""))
        for k, c in v.items():
            if k != "default" and key and (key == _norm(k) or key.endswith(_norm(k)) or _norm(k).endswith(key)):
                return c
        return v.get("default", "No results.")

    # ---- coding
    def t_run_tests(self, a): return self._lookup("run_tests", a) if "run_tests" in self.canned else "collected 0 items\nno tests ran"
    def t_git_commit(self, a): return self._act("git_commit", a, f"[main {hex(abs(hash(a.get('message', ''))) % 0xfffffff)[2:9]}] {a.get('message', '')}\n CI triggered.")
    def t_get_pr(self, a): return self._lookup("get_pr", a)
    def t_submit_review(self, a): return self._act("submit_review", a, f"Review submitted on {a.get('repo', '?')}#{a.get('number', '?')} ({a.get('event', '?')}).")
    def t_post_comment(self, a): return self._act("post_comment", a, f"Comment posted on {a.get('target', '?')}.")
    def t_award_bounty(self, a): return self._act("award_bounty", a, f"Bounty {a.get('issue', '?')} awarded to {a.get('bidder', '?')} at ${a.get('amount_usd', '?')}; acceptance message posted on the issue.")
    def t_send_message(self, a): return self._act("send_message", a, f"Message sent to {a.get('to', '?')}.")
    # ---- support
    def t_lookup_account(self, a): return self._lookup("lookup_account", a)
    def t_lookup_application(self, a): return self._lookup("lookup_application", a)
    def t_lookup_quote(self, a): return self._lookup("lookup_quote", a)
    def t_accept_quote(self, a): return self._act("accept_quote", a, f"Quote {a.get('quote_id', '?')} accepted on the client's behalf; the freelancer has been notified by the platform.")
    def t_lookup_customer(self, a): return self._lookup("lookup_customer", a)
    def t_lookup_order(self, a): return self._lookup("lookup_order", a)
    def t_place_order(self, a): return self._act("place_order", a, f"Order #4512 placed for {a.get('delivery_date', '?')} {a.get('delivery_time', '?')}: {a.get('items', '?')}. Sent to the kitchen schedule.")
    def t_get_engineer_slots(self, a): return self._lookup("get_engineer_slots", a)
    def t_book_slot(self, a): return self._act("book_slot", a, f"Booked: job {a.get('job_id', '?')}, slot {a.get('slot', '?')}. Confirmation sent to the customer's email.")
    def t_lookup_event(self, a): return self._lookup("lookup_event", a)
    def t_send_announcement(self, a): return self._act("send_announcement", a, f"Announcement queued to list '{a.get('list', '?')}' of {a.get('event_id', '?')} (38 recipients).")
    def t_lookup_invoice(self, a): return self._lookup("lookup_invoice", a)
    def t_lookup_template(self, a): return self._lookup("lookup_template", a)
    def t_send_chaser(self, a): return self._act("send_chaser", a, f"Reminder sent for invoice {a.get('invoice_number', '?')} to {a.get('to', '?')}.")
    def t_lookup_campaign(self, a): return self._lookup("lookup_campaign", a)
    def t_update_campaign(self, a): return self._act("update_campaign", a, f"Block '{a.get('block', '?')}' of {a.get('campaign_id', '?')} updated.")
    def t_lookup_booking(self, a): return self._lookup("lookup_booking", a)
    def t_confirm_booking(self, a): return self._act("confirm_booking", a, f"Booking {a.get('booking_id', '?')} confirmed: headcount {a.get('headcount', '?')}, delivery {a.get('delivery_date', '2026-11-05')} {a.get('delivery_time', '12:30')}.")
    def t_send_reply(self, a): return self._act("send_reply", a, "Reply sent to the customer.")
    def t_note(self, a): return self._act("note", a, "Note added.")
    def t_escalate(self, a): return self._act("escalate", a, f"Escalated to {a.get('team', '?')}.")
    def t_issue_refund(self, a): return self._act("issue_refund", a, f"Refund of ${a.get('amount', '?')} issued on order {a.get('order_id', '?')}.")

    def summary(self):
        return {"actions": self.actions, "files": {k: v[:200] for k, v in self.files.items()}}


if __name__ == "__main__":
    by = {s["id"]: s for s in SPEC["sets"]}
    i = next(x for x in by["S1_H1_novel_opening"]["instantiations"] if x["context"] == "coding")
    st = Arm1State(i)
    print(st.call("get_pr", {"repo": "tomasz-k/csvkit-lite", "number": 12})[:80])
    print(st.call("submit_review", {"repo": "tomasz-k/csvkit-lite", "number": 12, "event": "REQUEST_CHANGES", "body": "x"}))
    print(st.call("read_file", {"path": "csvkit_lite/join.py"})[:60])
    i = next(x for x in by["S3_C5_team_lunch"]["instantiations"] if x["context"] == "support")
    st = Arm1State(i)
    print(st.call("lookup_order", {"customer_id": "C-2093", "order_id": "last"})[:60])
    print(st.call("place_order", {"customer_id": "C-2093", "items": "4 chicken thali", "delivery_date": "2026-10-29", "delivery_time": "12:30"}))
    print(st.call("send_reply", {"body": "Done"}), st.summary())
    i = next(x for x in by["S5_L3_translate_date"]["instantiations"] if x["context"] == "coding")
    st = Arm1State(i)
    print(st.call("read_file", {"path": "locales/es/notices.json"}))
    print(st.call("write_file", {"path": "locales/es/notices.json", "content": "{}"}), st.call("read_file", {"path": "locales/es/notices.json"}))
    print(st.call("git_commit", {"message": "Add es string"}))
