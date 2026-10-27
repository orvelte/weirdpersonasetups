"""T2 primary measure: P(preferred outcome) from next-token log-probs on a self-hosted vLLM server (raw prompt, no chat template, no thinking).

python3 -m tasks.T2.logprobs --base-url https://<pod>-8000.proxy.runpod.net/v1 --model Qwen/Qwen3.5-27B --version 3.5
  [--prompts tasks/T2/prompts_full.json]   (default prompts.json = the pilot set; prompts_full.json = pre-registered full run)
Writes results/T2_logprobs/<version>.jsonl (one row per cell) and prints the framing x version table.
Two measures per cell:
  nexttok : top-20 next-token distribution after the raw prefix; P(pref) = p(pref word) / (p(pref)+p(dispref)); also the mass covered.
  fullseq : sum log-prob of " heads" vs " tails" continuation via prompt_logprobs (robust if the word is multi-token).
"""
import argparse, json, math, sys, time
from pathlib import Path
import requests
from harness import config
from harness.stats import wilson

OUT = config.ROOT / "results" / "T2_logprobs"

def post(base, body, retries=5):
    for a in range(retries):
        try:
            r = requests.post(base.rstrip("/") + "/completions", json=body, timeout=180)
            if r.status_code == 200: return r.json()
            last = f"{r.status_code} {r.text[:200]}"
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(3 * (a + 1))
    raise RuntimeError(last)

def nexttok(base, model, prefix):
    d = post(base, {"model": model, "prompt": prefix, "max_tokens": 1, "temperature": 0, "logprobs": 20})
    top = d["choices"][0]["logprobs"]["top_logprobs"][0]  # token -> logprob
    def mass(word):
        return sum(math.exp(lp) for t, lp in top.items() if t.strip().lower() == word or t.strip().lower().rstrip(".,") == word)
    ph, pt = mass("heads"), mass("tails")
    return {"p_heads": ph, "p_tails": pt, "covered": ph + pt, "top5": sorted(top.items(), key=lambda x: -x[1])[:5]}

def fullseq(base, model, prefix, word):
    # prompt_logprobs is a vLLM extension; returns per-token logprobs for the prompt; we sum those of the appended word.
    d = post(base, {"model": model, "prompt": prefix + " " + word, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
    plp = d.get("prompt_logprobs") or d["choices"][0].get("prompt_logprobs")
    d0 = post(base, {"model": model, "prompt": prefix, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
    n0 = len(d0.get("prompt_logprobs") or d0["choices"][0].get("prompt_logprobs"))
    tail = plp[n0:]
    return sum(list(t.values())[0]["logprob"] if isinstance(list(t.values())[0], dict) else list(t.values())[0] for t in tail if t)

def run(base, model, version, cells, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{tag}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    rows = []
    for c in cells:
        if c["id"] in done: continue
        r = {"id": c["id"], "version": version, "model": model, "framing": c["framing"], "variant": c["variant"], "mapping": c["mapping"], "preferred_outcome": c["preferred_outcome"],
             "arm": c.get("arm", "main"), "word": c.get("word", c["mapping"]), "order": c.get("order", infer_order(c)), "first_stated": c.get("first_stated", "heads")}
        try:
            r["nexttok"] = nexttok(base, model, c["raw"])
        except Exception as e:
            r["nexttok_error"] = str(e)[:200]
        try:
            lh, lt = fullseq(base, model, c["raw"], "heads"), fullseq(base, model, c["raw"], "tails")
            r["fullseq"] = {"lp_heads": lh, "lp_tails": lt}
        except Exception as e:
            r["fullseq_error"] = str(e)[:200]
        with open(out, "a") as f: f.write(json.dumps(r) + "\n")
        rows.append(r); print(c["id"], json.dumps({k: r.get(k) for k in ("nexttok", "fullseq")})[:160], flush=True)
    return out

def infer_order(c):
    """Pilot cells (prompts.json) always state heads first, so heads_pref = preferred stated first."""
    return "pref_first" if c["mapping"] == "heads_pref" else "dispref_first"

def p_pref(r, how):
    if how == "nexttok" and "nexttok" in r:
        ph, pt = r["nexttok"]["p_heads"], r["nexttok"]["p_tails"]
        if ph + pt == 0: return None
        p = ph / (ph + pt)
    elif how == "fullseq" and "fullseq" in r:
        lh, lt = r["fullseq"]["lp_heads"], r["fullseq"]["lp_tails"]
        p = 1 / (1 + math.exp(lt - lh))
    else:
        return None
    return p if r["preferred_outcome"] == "heads" else 1 - p

def table(files, how="nexttok"):
    """Quick-look tables: mean P(preferred) by framing x version, per arm, and split by statement order.
    Rows from the pilot (no arm/order fields) are treated as arm=main, order inferred from mapping.
    Scenario-level CIs and the pre-registered decision rules are in tasks/T2/summarize_full.py."""
    import collections
    rows = []
    for f in files:
        for l in Path(f).read_text().splitlines():
            r = json.loads(l); p = p_pref(r, how)
            if p is None: continue
            r["_p"] = p; r.setdefault("arm", "main"); r.setdefault("order", infer_order(r))
            rows.append(r)
    vs = sorted({r["version"] for r in rows}, key=lambda v: (len(v), v))
    def show(title, keyfn, rowkeys):
        agg = collections.defaultdict(list)
        for r in rows: agg[(keyfn(r), r["version"])].append(r["_p"])
        print(f"\n{title} [{how}]:")
        print("".ljust(30) + "".join(v.rjust(12) for v in vs))
        for k in rowkeys:
            print(str(k).ljust(30) + "".join((f"{sum(agg[(k,v)])/len(agg[(k,v)]):.2f}(n{len(agg[(k,v)])})" if agg.get((k, v)) else "-").rjust(12) for v in vs))
    arms = [a for a in ("main", "control") if any(r["arm"] == a for r in rows)]
    frs = sorted({r["framing"] for r in rows})
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        agg = collections.defaultdict(list)
        for r in sub: agg[(r["framing"], r["version"])].append(r["_p"])
        print(f"\nmean P(preferred) by framing, arm={arm} [{how}] (cells = scenarios x word x order):")
        print("framing".ljust(30) + "".join(v.rjust(12) for v in vs))
        for f_ in frs:
            print(f_.ljust(30) + "".join((f"{sum(agg[(f_,v)])/len(agg[(f_,v)]):.2f}(n{len(agg[(f_,v)])})" if agg.get((f_, v)) else "-").rjust(12) for v in vs))
    show("mean P(preferred) by arm (collapsed over framing)", lambda r: r["arm"], arms)
    show("mean P(preferred) by arm x statement order", lambda r: f'{r["arm"]}/{r["order"]}',
         [f"{a}/{o}" for a in arms for o in ("pref_first", "dispref_first")])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url"); ap.add_argument("--model"); ap.add_argument("--version"); ap.add_argument("--tag")
    ap.add_argument("--table", action="store_true"); ap.add_argument("--how", default="nexttok")
    ap.add_argument("--prompts", default=str(config.ROOT / "tasks" / "T2" / "prompts.json"),
                    help="prompt set to score (default: pilot prompts.json; full run: tasks/T2/prompts_full.json)")
    a = ap.parse_args()
    if a.table:
        table(sorted(OUT.glob("*.jsonl")), a.how); sys.exit()
    cells = json.load(open(a.prompts))["cells"]
    print(f"loaded {len(cells)} cells from {a.prompts}", flush=True)
    run(a.base_url, a.model, a.version, cells, a.tag or a.version)
    table([OUT / f"{a.tag or a.version}.jsonl"], a.how)
