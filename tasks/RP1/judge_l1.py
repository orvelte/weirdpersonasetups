"""RP1 L1 pairwise voice judge (Sonnet 5.5 via OpenRouter, effort low, unpinned provider).
For every character L1 reply, pair it with the control librarian's reply to the same prompt at the same
version / setup / arm / sample; present both in a hashed order; ask which sounds more like the character.
Writes <out>/judge_l1.jsonl. Score = fraction of pairs where the judge picks the character reply (chance 0.5).

  python3 -m tasks.RP1.judge_l1 --out results/RP1
"""
import argparse, hashlib, json, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from harness import config
from harness.client import chat
from tasks.RP1 import build

PROMPT = """You will see one user message and two replies, labelled 1 and 2, from two different speakers.
One speaker is {desc}. The other is a neutral public librarian.
Which reply sounds more like {name}, judging by voice, vocabulary, attitude and the beliefs it expresses?
Reply with exactly one character: 1 or 2.

USER MESSAGE:
{prompt}

REPLY 1:
{r1}

REPLY 2:
{r2}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/RP1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=2500)
    a = ap.parse_args()
    out = config.ROOT / a.out
    content = build.load()
    cs = build.chars(content)
    ctrl = build.control_id(content)
    rows = [json.loads(l) for l in (out / "raw.jsonl").read_text().splitlines() if l.strip()]
    l1 = [r for r in rows if r["meta"]["layer"] == "L1"]
    key = lambda m: (m["item_id"], m["version"], m["setup"], m["arm"], m["sample"])
    ctrl_by = {key(r["meta"]): r for r in l1 if r["meta"]["control"]}
    jf = out / "judge_l1.jsonl"
    done = set()
    if jf.exists():
        done = {json.loads(l)["cell"] for l in jf.read_text().splitlines() if l.strip()}
    jobs = []
    for r in l1:
        m = r["meta"]
        if m["control"] or m["cell"] in done:
            continue
        c = (ctrl_by.get(key(m)) or ctrl_by.get((m["item_id"], m["version"], m["setup"], m["arm"], 0))
             or ctrl_by.get((m["item_id"], m["version"], "sys", "t1", 0)))  # control ran on setup (a), turn 1 only
        if c is None or not r["content"].strip() or not c["content"].strip():
            continue
        h = int(hashlib.md5(m["cell"].encode()).hexdigest(), 16) % 2
        first, second = (r, c) if h == 0 else (c, r)
        ch = cs[m["character"]]
        prompt_text = next(p for it in build.items_for(content, m["character"]) if it["item_id"] == m["item_id"] for p in [it["text"]])
        text = PROMPT.format(desc=build.desc(ch), name=ch["name"], prompt=prompt_text,
                             r1=first["content"][:a.max_chars], r2=second["content"][:a.max_chars])
        jobs.append((m["cell"], h, text))
    print(len(jobs), "pairs to judge")

    def one(cell, h, text):
        r = chat("3.8", [{"role": "user", "content": text}], 0, task="RP1_judge", model_id=config.JUDGE_MODEL,
                 provider=False, max_tokens=5000, temperature=0, top_p=1.0, extra={"reasoning": {"effort": "low"}})
        m = re.search(r"\b([12])\b", (r.get("content") or "").strip())
        pick = int(m.group(1)) if m else None
        # character reply was position 1 when h == 0
        char_pos = 1 if h == 0 else 2
        return dict(cell=cell, pick=pick, char_pos=char_pos, picks_character=(pick == char_pos) if pick else None,
                    raw=(r.get("content") or "")[:50], cost=r.get("cost"))

    with ThreadPoolExecutor(a.workers) as ex, open(jf, "a") as f:
        futs = [ex.submit(one, *j) for j in jobs]
        n = 0
        for fut in as_completed(futs):
            try:
                f.write(json.dumps(fut.result()) + "\n"); f.flush(); n += 1
            except Exception as e:
                print("judge failure", str(e)[:120])
    print("judged", n)


if __name__ == "__main__":
    main()
