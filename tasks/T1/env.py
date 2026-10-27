"""Shared T1 episode environment (fixes from the pilot, see results/T1_pilot/README.md section 7).

- MAX_TURNS = 8 assistant turns in the agentic frame (pilot: 4).
- Both frames get the date in a system prompt: the agentic frame's system prompt already contains
  "Today is Monday 26 October 2026."; the chat frame gets a minimal system prompt containing only that line.
- Fake tools are stateful per episode (EpisodeState): list_files reflects deletions, writes and moves; read_file of a
  deleted file fails; list_events reflects created / updated / deleted events; send_email records sent mail and a second
  identical send returns a "duplicate detected" result. Everything else returns the canned result from scenarios.json.
Scenario texts are unchanged.
"""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "scenarios.json").read_text())
TOOL_DEFS = SPEC["tool_defs"]
WRITE_TOOLS = set(SPEC["write_tools"])
MAX_TURNS = 8
TODAY_LINE = f"Today is {SPEC['today']}."
CHAT_SYSTEM = TODAY_LINE


def start_messages(scen, frame):
    if frame == "chat":
        return [{"role": "system", "content": CHAT_SYSTEM}] + list(scen["chat_messages"])
    ag = scen["agentic"]
    assert TODAY_LINE in ag["system"], scen["id"]
    msgs = [{"role": "system", "content": ag["system"]}]
    msgs += list(ag.get("prior", []))
    msgs.append({"role": "user", "content": ag["user"]})
    return msgs


def _norm(p):
    return str(p or "").strip().lstrip("./")


def _match(key, cands):
    """Returns the candidate path that matches `key` (exact after normalisation, or suffix match on either side)."""
    k = _norm(key)
    if not k:
        return None
    for c in cands:
        if _norm(c) == k:
            return c
    for c in cands:
        n = _norm(c)
        if n.endswith("/" + k) or k.endswith("/" + n) or n.endswith(k) or k.endswith(n):
            return c
    return None


_EVENT_LINE = re.compile(r"^(?P<time>\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?)\s+(?P<title>.*?)\s*(?P<paren>\([^()]*\bid \w+[^()]*\))?\s*$")


class EpisodeState:
    def __init__(self, scen):
        self.scen = scen
        self.tools = set(scen["agentic"]["tools"])
        self.canned = scen["agentic"]["canned"]
        rf = self.canned.get("read_file", {})
        self.files = dict(rf) if isinstance(rf, dict) else {}
        lf = self.canned.get("list_files")
        if isinstance(lf, str) and lf.strip() and not lf.lower().startswith("no "):
            self.listing = [l for l in lf.split("\n") if l.strip()]
        else:
            self.listing = list(self.files.keys())
        ev = self.canned.get("list_events")
        self.events, self.events_default = {}, "No events."
        if isinstance(ev, dict):
            for k, v in ev.items():
                if k == "default":
                    self.events_default = v
                else:
                    self.events[k] = [l for l in v.split("\n") if l.strip()]
        elif isinstance(ev, str):
            self.events_default = ev
        self.sent, self.sent_log, self.posted, self.n_new_events, self.deals, self.reservations = set(), [], [], 0, [], []
        self.log = []  # (tool, args, result) in order

    # ---- helpers
    @staticmethod
    def _path_of(line):
        return line.strip().split()[0] if line.strip() else ""

    def _entry(self, path):
        m = _match(path, [self._path_of(l) for l in self.listing])
        if m is None:
            return None
        return next(l for l in self.listing if self._path_of(l) == m)

    def _readable_key(self, path):
        return _match(path, list(self.files.keys()))

    # ---- tools
    def call(self, name, args):
        args = args if isinstance(args, dict) else {}
        if name not in self.tools:
            res = f"Error: unknown tool '{name}'."
        else:
            fn = getattr(self, "t_" + name, None)
            res = fn(args) if fn else self._canned(name, args)
        self.log.append((name, args, res))
        return res

    def _canned(self, name, args):
        if name in WRITE_TOOLS:
            return "OK."
        if name not in self.canned:
            return "No results."
        v = self.canned[name]
        if isinstance(v, str):
            return v
        key = _norm(args.get("path") or args.get("date") or args.get("dir") or args.get("query") or "")
        for k, c in v.items():
            if k == "default":
                continue
            if key and (key == _norm(k) or key.endswith(_norm(k)) or _norm(k).endswith(key)):
                return c
        if "default" in v:
            return v["default"]
        avail = ", ".join(k for k in v if k != "default")
        return f"Error: not found: '{key}'." + (f" Available: {avail}" if avail else "")

    def t_search_docs(self, a): return self._canned("search_docs", a)
    def t_list_contacts(self, a): return self._canned("list_contacts", a)

    def t_list_files(self, a):
        d = _norm(a.get("dir")).rstrip("/")
        if not self.listing:
            return "(empty)"
        if d and d not in (".", ""):
            sel = [l for l in self.listing if _norm(self._path_of(l)).startswith(d + "/") or _norm(self._path_of(l)) == d]
            if sel:
                return "\n".join(sel)
        return "\n".join(self.listing)

    def t_read_file(self, a):
        p = a.get("path", "")
        k = self._readable_key(p)
        if k is not None:
            return self.files[k]
        if self._entry(p) is not None:
            return f"Error: '{p}' is not a text file."
        avail = ", ".join(self.files)
        return f"Error: not found: '{p}'." + (f" Available: {avail}" if avail else "")

    def t_write_file(self, a):
        p, content = a.get("path", ""), a.get("content", "") or ""
        k = self._readable_key(p) or _norm(p)
        self.files[k] = content
        if self._entry(p) is None:
            self.listing.append(p)
        return f"Wrote {len(content)} characters to {p}."

    def t_delete_file(self, a):
        p = a.get("path", "")
        e = self._entry(p)
        if e is None:
            return f"Error: not found: '{p}'."
        self.listing.remove(e)
        k = self._readable_key(p)
        if k is not None:
            self.files.pop(k, None)
        return f"Deleted {self._path_of(e)}."

    def t_move_file(self, a):
        src, dst = a.get("src", ""), a.get("dst", "")
        e = self._entry(src)
        if e is None:
            return f"Error: not found: '{src}'."
        i = self.listing.index(e)
        self.listing[i] = e.replace(self._path_of(e), dst, 1)
        k = self._readable_key(src)
        if k is not None:
            self.files[dst] = self.files.pop(k)
        return f"Moved {self._path_of(e)} -> {dst}."

    def t_list_events(self, a):
        d = _norm(a.get("date"))[:10]
        if d in self.events:
            return "\n".join(self.events[d]) if self.events[d] else self.events_default
        return self.events_default

    @staticmethod
    def _hhmm(iso):
        m = re.search(r"T(\d{1,2}:\d{2})", iso or "")
        return m.group(1) if m else (iso or "")[-5:]

    def t_create_event(self, a):
        self.n_new_events += 1
        eid = f"ev_new_{self.n_new_events}"
        start, end, title = a.get("start", "") or "", a.get("end", "") or "", a.get("title", "") or "(untitled)"
        date = start[:10]
        extra = f"; attendees: {a['attendees']}" if a.get("attendees") else ""
        line = f"{self._hhmm(start)}-{self._hhmm(end)} {title} (id {eid}{extra})"
        self.events.setdefault(date, []).append(line)
        return f"Event created (id {eid}): '{title}' {start} to {end}."

    def _find_event(self, eid):
        for d, lines in self.events.items():
            for l in lines:
                if re.search(r"\bid " + re.escape(str(eid)) + r"\b", l):
                    return d, l
        return None, None

    def t_update_event(self, a):
        eid = a.get("event_id", "")
        d, l = self._find_event(eid)
        if l is None:
            return f"Error: no event with id '{eid}'."
        m = _EVENT_LINE.match(l)
        time_s, title, paren = (m.group("time"), m.group("title"), m.group("paren") or f"(id {eid})") if m else (l.split()[0], l, f"(id {eid})")
        if a.get("start"):
            time_s = self._hhmm(a["start"]) + ("-" + self._hhmm(a["end"]) if a.get("end") else "")
            nd = a["start"][:10]
            if nd and nd != d:
                self.events[d].remove(l)
                self.events.setdefault(nd, [])
                d = nd
                l = None
        if a.get("title"):
            title = a["title"]
        new = f"{time_s} {title} {paren}"
        if l is None:
            self.events[d].append(new)
        else:
            self.events[d][self.events[d].index(l)] = new
        return f"Event {eid} updated: {new}"

    def t_delete_event(self, a):
        eid = a.get("event_id", "")
        d, l = self._find_event(eid)
        if l is None:
            return f"Error: no event with id '{eid}'."
        self.events[d].remove(l)
        return f"Event {eid} deleted."

    def t_send_email(self, a):
        to, subject, body = a.get("to", "") or "", a.get("subject", "") or "", a.get("body", "") or ""
        key = (to.strip().lower(), subject.strip().lower(), body.strip())
        self.sent_log.append({"to": to, "subject": subject, "body": body, "cc": a.get("cc"), "duplicate": key in self.sent})
        if key in self.sent:
            return f"Duplicate detected: an identical email (to {to}, subject '{subject}') was already sent earlier in this session. Not sent again."
        self.sent.add(key)
        return f"Email sent to {to}."

    def t_post_message(self, a):
        self.posted.append({"channel": a.get("channel"), "text": a.get("text")})
        return f"Posted to {a.get('channel', '?')}."

    def t_make_reservation(self, a):
        self.reservations.append(a)
        return "Reservation confirmed (ref OL-5521)."

    def t_log_deal(self, a):
        self.deals.append(a)
        return "Deal logged (id deal_8810)."

    def summary(self):
        return {"files_remaining": [self._path_of(l) for l in self.listing], "events": self.events, "emails_sent": self.sent_log,
                "posted": self.posted, "deals": self.deals, "reservations": self.reservations,
                "n_duplicate_sends": sum(1 for e in self.sent_log if e["duplicate"])}


if __name__ == "__main__":
    # smoke test of the state machine (no API calls)
    by_id = {s["id"]: s for s in SPEC["scenarios"]}
    st = EpisodeState(by_id["A3_cleanup_backups"])
    print(st.call("list_files", {"dir": "/backups"}))
    print(st.call("delete_file", {"path": "/backups/db_2025-01-15.bak"}))
    print(st.call("delete_file", {"path": "db_2025-01-15.bak"}))
    print(st.call("list_files", {"dir": "/backups"}))
    print(st.call("read_file", {"path": "/backups/misc_old.tar"}))
    st = EpisodeState(by_id["C2_dentist_conflict"])
    print(st.call("list_events", {"date": "2026-10-29"}))
    print(st.call("create_event", {"title": "Dentist", "start": "2026-10-29T15:00", "end": "2026-10-29T16:00"}))
    print(st.call("update_event", {"event_id": "ev_102", "start": "2026-10-29T16:00", "end": "2026-10-29T17:00"}))
    print(st.call("delete_event", {"event_id": "ev_103"}))
    print(st.call("list_events", {"date": "2026-10-29"}))
    st = EpisodeState(by_id["D4_budget_summary"])
    print(st.call("send_email", {"to": "priya@ourco.example", "subject": "Q4", "body": "GBP 60,000"}))
    print(st.call("send_email", {"to": "priya@ourco.example", "subject": "Q4", "body": "GBP 60,000"}))
    print(st.call("read_file", {"path": "plans/q4_plan.txt"})[:40])
    st = EpisodeState(by_id["C5_rename_file"])
    print(st.call("move_file", {"src": "reports/draft_v3.txt", "dst": "reports/final.txt"}))
    print(st.call("list_files", {"dir": "reports"}))
    st = EpisodeState(by_id["B1_weekly_numbers"])
    print(st.call("write_file", {"path": "reports/weekly_status.txt", "content": "x"}))
    print(st.call("read_file", {"path": "reports/weekly_status.txt"}), st.call("list_files", {"dir": "reports"}))
    print(st.summary())
    print(start_messages(by_id["A1_five_sams"], "chat")[0])
