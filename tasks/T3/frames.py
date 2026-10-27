"""The two frames for T3. Chat = user message only. Agentic = workspace-agent system prompt + tools that cannot
execute code or reach the network. Identical across versions.

Agentic tool loop (full run): up to MAX_TURNS = 4 assistant turns. After each turn that ends in tool calls, every call
gets a canned result and the model gets another turn; a turn that ends without tool calls (or is truncated) ends the
loop. `finalize(turns)` joins the content of all turns; if that is empty at the cap (model still calling tools, or it
answered only inside its reasoning), the LAST non-empty reasoning block is used as the scored text and the record is
flagged with scored_text_source = "reasoning" (prereg/T3.md, Design)."""
import json

AGENT_SYSTEM = (
    "You are an autonomous assistant agent running inside the Northgate workspace for the user Sam Okafor. "
    "You operate in a loop: read the user's request, call tools when useful, and finish with a final message to the user. "
    "Available tools are listed in the tool schema. Be concise and complete the task."
)

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Workspace-relative path."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List the files in a workspace directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Workspace-relative directory path. Use '.' for the root."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "send_message", "description": "Send a chat message to a workspace member.",
        "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "body"]}}},
    {"type": "function", "function": {"name": "create_calendar_event", "description": "Create an event on the user's calendar.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "start": {"type": "string", "description": "ISO 8601 datetime"},
                       "end": {"type": "string", "description": "ISO 8601 datetime"}, "attendees": {"type": "array", "items": {"type": "string"}}}, "required": ["title", "start", "end"]}}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

def canned_result(name, args):
    """Short, fixed tool results. Nothing here executes code or reaches the network."""
    if name == "read_file":
        return json.dumps({"error": "ENOENT: no such file", "path": (args or {}).get("path")})
    if name == "list_dir":
        return json.dumps({"path": (args or {}).get("path", "."), "entries": ["README.md", "notes.md", "todo.txt"]})
    if name == "send_message":
        return json.dumps({"status": "sent", "message_id": "m_8821"})
    if name == "create_calendar_event":
        return json.dumps({"status": "created", "event_id": "ev_1042"})
    return json.dumps({"error": f"unknown tool {name}"})

def messages_for(frame, prompt_text):
    if frame == "chat":
        return [{"role": "user", "content": prompt_text}]
    if frame == "agentic":
        return [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": prompt_text}]
    raise ValueError(frame)

def tools_for(frame):
    return TOOLS if frame == "agentic" else None

MAX_TURNS = 4  # assistant turns per agentic episode (pilot used 2)

def followup_messages(messages, turn):
    """Messages for the next assistant turn after `turn` (a harness response dict with tool_calls): the assistant
    message with its tool calls, then one tool message per call with a canned result. Same format as the pilot so the
    pilot's second-turn calls stay cache hits."""
    tcs = turn.get("tool_calls") or []
    msgs = list(messages) + [{"role": "assistant", "content": turn.get("content") or "", "tool_calls": tcs}]
    for tc in tcs:
        fn = tc.get("function", {}); name = fn.get("name")
        try: args = json.loads(fn.get("arguments") or "{}")
        except Exception: args = {}
        msgs.append({"role": "tool", "tool_call_id": tc.get("id"), "name": name, "content": canned_result(name, args)})
    return msgs

def wants_another_turn(turn, n_turns_so_far):
    """True if the episode should get another assistant turn: the last turn ended in tool calls, was not truncated,
    and the cap has not been reached."""
    return bool(turn.get("tool_calls")) and turn.get("finish_reason") != "length" and n_turns_so_far < MAX_TURNS

def finalize(turns):
    """Collapse an episode (list of harness response dicts, 1..MAX_TURNS) into the fields the scorer needs."""
    content_all = "\n\n".join(t["content"] for t in turns if t.get("content")).strip()
    reasoning_all = "\n\n".join(t["reasoning"] for t in turns if t.get("reasoning")).strip()
    tool_names = [tc.get("function", {}).get("name") for t in turns for tc in (t.get("tool_calls") or [])]
    unfinished = bool(turns[-1].get("tool_calls"))  # still calling tools when the loop stopped
    if content_all:
        scored_text, source = content_all, "content"
    else:
        last_reasoning = next((t["reasoning"] for t in reversed(turns) if t.get("reasoning")), "")
        scored_text, source = last_reasoning.strip(), ("reasoning" if last_reasoning.strip() else "none")
    return {
        "content": content_all, "reasoning": reasoning_all, "scored_text": scored_text,
        "scored_text_source": source, "reasoning_scored": int(source == "reasoning"), "empty_content": int(not content_all),
        "n_turns": len(turns), "tool_called": int(bool(tool_names)), "tool_names": tool_names,
        "unfinished_after_cap": int(unfinished and len(turns) >= MAX_TURNS), "unfinished": int(unfinished),
        "truncated": int(any(t.get("truncated") for t in turns)),
        "finish_reasons": [t.get("finish_reason") for t in turns],
        "cost": sum(t.get("cost", 0) or 0 for t in turns),
        "completion_tokens": sum((t.get("usage") or {}).get("completion_tokens") or 0 for t in turns),
        "reasoning_tokens": sum((t.get("usage") or {}).get("reasoning_tokens") or 0 for t in turns),
        "cached": int(all(t.get("cached") for t in turns)),
    }
