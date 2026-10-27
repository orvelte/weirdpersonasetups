import json, re, collections
from harness.client import chat_many, total_spend
from harness import config
from harness.stats import fmt_rate
from tasks.T10.prompts import PROMPTS, CATEGORIES
N = 8
jobs = [{"version": v, "messages": [{"role": "user", "content": p["text"]}], "sample": i, "pid": p["id"]} for v in config.VERSIONS for p in PROMPTS for i in range(N)]
meta = [(j.pop("pid"), j["version"], j["sample"]) for j in jobs]
s0 = total_spend()
rs = chat_many(jobs, task="T10", max_tokens=3000)
out = config.ROOT / "results" / "T10_pilot"; out.mkdir(parents=True, exist_ok=True)
def cat(text):
    t = text.lower(); hits = {k: [w for w in ws if w in t] for k, ws in CATEGORIES.items()}
    hits = {k: v for k, v in hits.items() if v}
    if not hits: return "other", {}
    return max(hits, key=lambda k: len(hits[k])), hits
rows = []
with open(out / "raw.jsonl", "w") as f:
    for (pid, v, i), r in zip(meta, rs):
        c, hits = cat(r["content"])
        row = {"prompt": pid, "version": v, "sample": i, "category": c, "hits": hits, "truncated": r["truncated"], "content": r["content"], "reasoning": r["reasoning"][:1500]}
        rows.append(row); f.write(json.dumps(row) + "\n")
agg = collections.defaultdict(collections.Counter)
for r in rows: agg[r["version"]][r["category"]] += 1
lines = ["# T10 pilot (exploratory completion study)", "", f"n = {N} samples x {len(PROMPTS)} prompts per version. thinking on, effort medium. Keyword categoriser (NOT validated); counts below are a first look only.", ""]
lines.append("| version | scifi_archetype | reward_score | helpful_benign | other | truncated |"); lines.append("|---|---|---|---|---|---|")
for v in config.VERSIONS:
    n = sum(agg[v].values()); tr = sum(r["truncated"] for r in rows if r["version"] == v)
    lines.append(f"| {v} | " + " | ".join(fmt_rate(agg[v][k], n) for k in ["scifi_archetype", "reward_score", "helpful_benign", "other"]) + f" | {tr}/{n} |")
lines += ["", f"spend: ${total_spend()-s0:.2f}", "", "## Excerpts (first sample per prompt per version)", ""]
for v in config.VERSIONS:
    for p in PROMPTS:
        r = next(x for x in rows if x["version"] == v and x["prompt"] == p["id"] and x["sample"] == 0)
        lines.append(f"**{v} / {p['id']} / {r['category']}**: {r['content'].strip()[:500]}"); lines.append("")
(out / "README.md").write_text("\n".join(lines)); print("\n".join(lines[:12]))
