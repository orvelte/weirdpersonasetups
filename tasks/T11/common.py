"""Shared pieces for T11: question/persona loading, the agentic frame (T3 style, 3 irrelevant tools), a failure-tolerant
chat_many wrapper, and resumable jsonl helpers."""
import json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from harness import config
from harness.client import chat, total_spend

HERE = Path(__file__).parent
OUT = config.ROOT / "results" / "T11"
TASK_BUDGET = float(os.environ.get("T11_BUDGET", "14"))  # USD for this task (ask before exceeding)

def load_questions():
    return json.loads((HERE / "questions.json").read_text())["questions"]

def load_personas():
    return json.loads((HERE / "personas.json").read_text())

# Agentic frame: same style as tasks/T3/frames.py, three tools that are irrelevant to every question.
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
    {"type": "function", "function": {"name": "get_workspace_status", "description": "Return the workspace's current status (online members, unread counts).",
        "parameters": {"type": "object", "properties": {}}}},
]

def canned_result(name, args):
    if name == "read_file":
        return json.dumps({"error": "ENOENT: no such file", "path": (args or {}).get("path")})
    if name == "list_dir":
        return json.dumps({"path": (args or {}).get("path", "."), "entries": ["README.md", "notes.md", "todo.txt"]})
    if name == "get_workspace_status":
        return json.dumps({"online": ["Sam Okafor"], "unread": 0})
    return json.dumps({"error": f"unknown tool {name}"})

def frame_messages(frame, q_messages, system=None):
    msgs = list(q_messages)
    if frame == "agentic":
        return [{"role": "system", "content": AGENT_SYSTEM}] + msgs
    if system:
        return [{"role": "system", "content": system}] + msgs
    return msgs

def t11_spend():
    """Spend logged under T11_* task tags (the spend log is shared with other tasks running concurrently)."""
    if not config.SPEND_LOG.exists(): return 0.0
    return sum(json.loads(l).get("cost", 0) for l in config.SPEND_LOG.read_text().splitlines() if l.strip() and '"T11' in l)

def safe_chat_many(jobs, task="T11", workers=8, progress_every=50, **kw):
    """Like harness.client.chat_many but a failed call yields {'error': str} instead of aborting the run, and
    OpenRouter 402 in-flight-budget errors get a long backoff. Stops submitting once the task budget is hit."""
    out = [None] * len(jobs)
    def one(i, j):
        kws = {**kw, **{k: v for k, v in j.items() if k not in ("version", "messages", "sample")}}
        for attempt in range(12):
            if t11_spend() > TASK_BUDGET:
                return i, {"error": "task budget exceeded"}
            try:
                return i, chat(j["version"], j["messages"], j.get("sample", 0), task=task, **kws)
            except Exception as e:
                s = str(e)
                if "402" in s or "in_flight" in s or "credits" in s or "rate" in s.lower():
                    time.sleep(15 + 10 * attempt); continue
                return i, {"error": s[:500]}
        return i, {"error": "gave up after repeated 402/rate errors"}
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, i, j) for i, j in enumerate(jobs)]
        done = 0
        for f in as_completed(futs):
            i, r = f.result(); out[i] = r; done += 1
            if done % progress_every == 0:
                print(f"  {task}: {done}/{len(jobs)} done, T11 spend ${t11_spend():.2f}, total ${total_spend():.2f}", flush=True)
    return out

def read_jsonl(path):
    p = Path(path)
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def append_jsonl(path, recs):
    with open(path, "a") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
