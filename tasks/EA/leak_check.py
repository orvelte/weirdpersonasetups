"""E-A Phase 0 leak check: no TRAINING scenario (tasks/EA/train_scenarios.json) may share a premise with a HELD-OUT scenario
(tasks/T1/scenarios.json; tasks/EB/scenarios.json when present).

(a) Embedding similarity (Fireworks, qwen3-embedding-8b) between every training scenario and every held-out scenario; pairs
    above FLAG (cosine 0.80) are flagged. The threshold is justified against (i) the distribution of within-set similarities
    (distinct premises by construction) and (ii) a positive control: the chat vs agentic versions of the SAME scenario (same
    premise, different surface), which should sit above the threshold.
(b) Blind LLM judge (harness.client.chat, config.JUDGE_MODEL, provider=False, reasoning effort low) on every flagged pair plus the
    20 highest unflagged pairs per held-out set: "Are these two the same situation with different surface details? SAME or DIFFERENT".
Writes results/EA_phase0/leak_check.md (+ leak_check.json, embeddings cache). Run: python3 -m tasks.EA.leak_check [--no-judge]
Spend tag: EA_phase0.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np, requests
from harness import config
from harness.client import chat, total_spend

ROOT = config.ROOT
OUT = ROOT / "results" / "EA_phase0"; OUT.mkdir(parents=True, exist_ok=True)
EMB_CACHE = OUT / "embeddings_cache.json"
FLAG = 0.80
TOP_UNFLAGGED = 20
EMB_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
EMB_URL = "https://api.fireworks.ai/inference/v1/embeddings"
TASK = "EA_phase0"
MAX_CHARS = 6000  # per text sent to the embedder / judge (scenarios are 0.5-3k chars; this only trims the longest E-B ones)

def _msgs_text(msgs):
    return "\n".join(f"[{m.get('role')}] {m.get('content') or ''}" for m in msgs if m.get("content"))

# ---------------------------------------------------------------- scenario loaders -> list of dict(id, family, chat, agentic)
def load_train():
    d = json.loads((ROOT / "tasks/EA/train_scenarios.json").read_text())
    out = []
    for s in d["scenarios"]:
        ag = s["agentic"]
        out.append({"id": s["id"], "family": s["family"], "chat": s["premise"] + "\n" + _msgs_text(s["chat_messages"]),
                    "agentic": _msgs_text(ag.get("prior", [])) + "\n[user] " + ag["user"] + "\n[tool results] " + json.dumps(ag["canned"], ensure_ascii=False)})
    return out

def load_T1():
    d = json.loads((ROOT / "tasks/T1/scenarios.json").read_text())
    out = []
    for s in d["scenarios"]:
        ag = s["agentic"]
        out.append({"id": s["id"], "family": s["family"], "chat": _msgs_text(s["chat_messages"]),
                    "agentic": _msgs_text(ag.get("prior", [])) + "\n[user] " + ag["user"] + "\n[tool results] " + json.dumps(ag["canned"], ensure_ascii=False)})
    return out

def load_EB():
    p = ROOT / "tasks/EB/scenarios.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    d = json.loads(p.read_text())
    out = []
    for s in d["scenarios"]:
        chat_txt = s.get("premise", "")
        msgs = None
        v = s.get("variants") or {}
        try:
            msgs = v["plain"]["no_pref"]["A"]
        except (KeyError, TypeError):
            for k in ("chat_messages", "messages"):
                if k in s: msgs = s[k]; break
        if msgs:
            chat_txt += "\n" + _msgs_text(msgs)
        elif s.get("context"):
            chat_txt += "\n[user] " + s["context"]
        out.append({"id": s["id"], "family": s.get("axis") or s.get("family"), "chat": chat_txt, "agentic": None})
    return out

# ---------------------------------------------------------------- embeddings
def _env_key():
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("FIREWORKS_API_KEY="): key = line.split("=", 1)[1].strip()
    return key

def embed(texts):
    cache = json.loads(EMB_CACHE.read_text()) if EMB_CACHE.exists() else {}
    keys = [hashlib.sha256((EMB_MODEL + "|" + t[:MAX_CHARS]).encode()).hexdigest() for t in texts]
    todo = [(k, t[:MAX_CHARS]) for k, t in zip(keys, texts) if k not in cache]
    hdr = {"Authorization": "Bearer " + _env_key(), "Content-Type": "application/json"}
    for i in range(0, len(todo), 16):
        batch = todo[i:i + 16]
        for attempt in range(5):
            r = requests.post(EMB_URL, headers=hdr, json={"model": EMB_MODEL, "input": [t for _, t in batch]}, timeout=120)
            if r.status_code == 200: break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        for (k, _), row in zip(batch, r.json()["data"]):
            cache[k] = row["embedding"]
    if todo:
        EMB_CACHE.write_text(json.dumps(cache))
    M = np.array([cache[k] for k in keys], dtype=np.float64)
    return M / np.linalg.norm(M, axis=1, keepdims=True)

def quantiles(x):
    x = np.asarray(x)
    if x.size == 0: return "n=0"
    q = np.quantile(x, [0.5, 0.9, 0.95, 0.99])
    return f"n={x.size}: median {q[0]:.3f}, p90 {q[1]:.3f}, p95 {q[2]:.3f}, p99 {q[3]:.3f}, max {x.max():.3f}, min {x.min():.3f}"

def upper(M):
    iu = np.triu_indices(M.shape[0], k=1); return M[iu]

# ---------------------------------------------------------------- judge
JUDGE_PROMPT = ("Below are two task scenarios given to an AI assistant, each as a short description plus the conversation.\n"
                "Are these two the same situation with different surface details? Answer SAME or DIFFERENT and one sentence.\n\n"
                "=== SCENARIO 1 ===\n{a}\n\n=== SCENARIO 2 ===\n{b}\n")

def judge(a, b):
    r = chat("judge", [{"role": "user", "content": JUDGE_PROMPT.format(a=a[:MAX_CHARS], b=b[:MAX_CHARS])}], sample=0, task=TASK,
             model_id=config.JUDGE_MODEL, provider=False, extra={"reasoning": {"effort": "low"}}, max_tokens=300, temperature=0.0)
    txt = (r["content"] or "").strip()
    head = txt.upper()[:40]
    verdict = "SAME" if "SAME" in head and "DIFFERENT" not in head.split("SAME")[0] else ("DIFFERENT" if "DIFFERENT" in head else "UNPARSED")
    return verdict, txt.replace("\n", " ")[:300], r.get("cost", 0.0), r.get("cached", False)

# ---------------------------------------------------------------- main
def run(no_judge=False):
    train = load_train(); t1 = load_T1(); eb = load_EB()
    heldout = {"T1": t1}
    eb_path = ROOT / "tasks/EB/scenarios.json"
    if eb: heldout["EB"] = eb
    report = {"flag": FLAG, "embedding_model": EMB_MODEL, "n_train": len(train), "heldout": {}, "spend_before": total_spend()}
    lines = ["# E-A Phase 0 leak check: training scenarios vs held-out targets", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by tasks/EA/leak_check.py. Embedding model `{EMB_MODEL}` (Fireworks), cosine on L2-normalised vectors. "
             f"Judge: `{config.JUDGE_MODEL}` via OpenRouter (provider unpinned, reasoning effort low, temperature 0), spend tag `{TASK}`.", "",
             f"Training set: tasks/EA/train_scenarios.json, {len(train)} scenarios. Held-out sets: T1 ({len(t1)} scenarios in tasks/T1/scenarios.json)"
             + (f", E-B ({len(eb)} scenarios in tasks/EB/scenarios.json, mtime {time.strftime('%H:%M:%S', time.localtime(eb_path.stat().st_mtime))})" if eb else
                "; E-B: tasks/EB/scenarios.json NOT present at run time"), "",
             "Text embedded per scenario: premise (where the file has one) + the chat-frame messages. Positive-control texts: the agentic frame "
             "(prior turns + user request + canned tool results) of the same scenario.", ""]
    # embeddings
    tr_chat = embed([s["chat"] for s in train]); tr_ag = embed([s["agentic"] for s in train])
    t1_chat = embed([s["chat"] for s in t1]); t1_ag = embed([s["agentic"] for s in t1])
    within_tr = upper(tr_chat @ tr_chat.T); within_t1 = upper(t1_chat @ t1_chat.T)
    ctrl_tr = np.sum(tr_chat * tr_ag, axis=1); ctrl_t1 = np.sum(t1_chat * t1_ag, axis=1)
    # within-family vs cross-family (within set) for extra context
    def fam_split(S, M):
        same, diff = [], []
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                (same if S[i]["family"] == S[j]["family"] else diff).append(M[i, j])
        return np.array(same), np.array(diff)
    tr_same, tr_diff = fam_split(train, tr_chat @ tr_chat.T); t1_same, t1_diff = fam_split(t1, t1_chat @ t1_chat.T)
    lines += ["## 1. Similarity distributions and the 0.80 threshold", "",
              "Distinct-premise pairs (by construction, every pair within one set is a different premise):", "",
              f"- within training set (chat texts): {quantiles(within_tr)}",
              f"  - same family: {quantiles(tr_same)}",
              f"  - different family: {quantiles(tr_diff)}",
              f"- within T1 (chat texts): {quantiles(within_t1)}",
              f"  - same family: {quantiles(t1_same)}",
              f"  - different family: {quantiles(t1_diff)}"]
    if eb:
        eb_chat = embed([s["chat"] for s in eb]); within_eb = upper(eb_chat @ eb_chat.T)
        lines.append(f"- within E-B (chat texts): {quantiles(within_eb)}")
    lines += ["", "Same-premise pairs (positive control: chat version vs agentic version of the SAME scenario, different surface):", "",
              f"- training set: {quantiles(ctrl_tr)}", f"- T1: {quantiles(ctrl_t1)}", "",
              f"Reading: a pair that shares a premise should look like the positive-control pairs, a pair that does not should look like the within-set pairs. "
              f"The flag threshold {FLAG} is kept as pre-specified; the numbers above show where it falls relative to both distributions "
              f"(fraction of distinct-premise within-set pairs above {FLAG}: training {np.mean(within_tr > FLAG):.3f}, T1 {np.mean(within_t1 > FLAG):.3f}; "
              f"fraction of same-premise control pairs above {FLAG}: training {np.mean(ctrl_tr > FLAG):.3f}, T1 {np.mean(ctrl_t1 > FLAG):.3f}). "
              f"Whatever the threshold, the judge also sees the {TOP_UNFLAGGED} highest unflagged pairs per held-out set, so pairs just under it are not skipped.", ""]
    report["within_train"] = quantiles(within_tr); report["within_T1"] = quantiles(within_t1); report["ctrl_train"] = quantiles(ctrl_tr); report["ctrl_T1"] = quantiles(ctrl_t1)
    # cross-set
    spend0 = total_spend(); n_judge = 0
    for name, H in heldout.items():
        H_chat = t1_chat if name == "T1" else eb_chat
        C = tr_chat @ H_chat.T
        pairs = sorted(((C[i, j], i, j) for i in range(len(train)) for j in range(len(H))), reverse=True)
        cross = np.array([p[0] for p in pairs])
        flagged = [p for p in pairs if p[0] > FLAG]
        unflagged = [p for p in pairs if p[0] <= FLAG][:TOP_UNFLAGGED]
        lines += [f"## 2.{name}. Training x {name}: {len(train)} x {len(H)} = {len(pairs)} pairs", "",
                  f"- cross-set similarity: {quantiles(cross)}", f"- flagged (> {FLAG}): {len(flagged)}", ""]
        # nearest held-out neighbour per training scenario
        lines += ["Nearest held-out neighbour of each training scenario:", "", "| training | nearest " + name + " | cosine |", "|---|---|---|"]
        for i, s in enumerate(train):
            j = int(np.argmax(C[i])); lines.append(f"| {s['id']} | {H[j]['id']} | {C[i, j]:.3f} |")
        lines.append("")
        judged = []
        if not no_judge:
            lines += [f"Judge verdicts ({len(flagged)} flagged + {len(unflagged)} highest unflagged):", "", "| cosine | training | " + name + " | flagged | verdict | judge's sentence |", "|---|---|---|---|---|---|"]
            for sim, i, j in flagged + unflagged:
                v, txt, cost, cached = judge(train[i]["chat"], H[j]["chat"]); n_judge += 1
                judged.append({"cosine": float(sim), "train": train[i]["id"], "heldout": H[j]["id"], "flagged": bool(sim > FLAG), "verdict": v, "text": txt})
                lines.append(f"| {sim:.3f} | {train[i]['id']} | {H[j]['id']} | {'yes' if sim > FLAG else ''} | **{v}** | {txt.replace('|', '/')} |")
            lines.append("")
            sames = [p for p in judged if p["verdict"] == "SAME"]
            lines.append(f"**SAME verdicts against {name}: {len(sames)}**" + (" -> these training scenarios must be dropped or rewritten: " + ", ".join(sorted({p['train'] for p in sames})) if sames else " -> nothing to drop."))
            lines.append("")
        report["heldout"][name] = {"n": len(H), "cross": quantiles(cross), "n_flagged": len(flagged),
                                   "flagged": [{"cosine": float(a), "train": train[i]["id"], "heldout": H[j]["id"]} for a, i, j in flagged],
                                   "nearest": {s["id"]: (H[int(np.argmax(C[i]))]["id"], float(C[i].max())) for i, s in enumerate(train)}, "judged": judged}
    spend = total_spend() - spend0
    lines += ["## 3. Spend", "", f"Judge calls this run: {n_judge}; OpenRouter spend this run (tag {TASK}): ${spend:.3f}. Fireworks embedding calls are not metered in data/spend.jsonl (a few thousand tokens, well under a cent).", ""]
    report["judge_calls"] = n_judge; report["judge_spend"] = spend
    (OUT / "leak_check.json").write_text(json.dumps(report, indent=1))
    (OUT / "leak_check.md").write_text("\n".join(lines))
    print("\n".join(l for l in lines if l.startswith(("- ", "**", "## ", "Judge"))))
    print("wrote", OUT / "leak_check.md")
    return report

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-judge", action="store_true"); a = ap.parse_args()
    run(no_judge=a.no_judge)
