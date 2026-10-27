"""T7 judge-free probe: next-token mass on chat-start / think tokens after each raw prompt (and after 50 tokens of its own continuation).
python3 -m tasks.T7.probe_chat_tokens --base-url URL --model Qwen/Qwen3.8-27B --version 3.8
"""
import argparse, json, math, time
import requests
from harness import config
OUT = config.ROOT / "results" / "T7_full"
TARGETS = {"im_start": ["<|im_start|>"], "think": ["<think>", "<think"], "assistant_word": ["Assistant", " Assistant", "assistant"]}

def post(base, body):
    for a in range(5):
        try:
            r = requests.post(base.rstrip("/") + "/completions", json=body, timeout=180)
            if r.status_code == 200: return r.json()
        except requests.RequestException: pass
        time.sleep(3 * (a + 1))
    raise RuntimeError("probe failed")

def mass(top, toks):
    return sum(math.exp(lp) for t, lp in top.items() if t in toks)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--version", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = json.load(open(config.ROOT / "tasks/T7/prompts.json"))
    prompts = prompts["prompts"] if isinstance(prompts, dict) and "prompts" in prompts else prompts
    out = OUT / f"chat_token_probe_{a.version}.jsonl"
    with open(out, "w") as f:
        for p in prompts:
            row = {"id": p["id"], "family": p["family"], "version": a.version}
            d = post(a.base_url, {"model": a.model, "prompt": p["text"], "max_tokens": 50, "temperature": 0.6, "top_p": 0.95, "logprobs": 20, "skip_special_tokens": False})
            ch = d["choices"][0]; tops = ch["logprobs"]["top_logprobs"]
            row["at_cut"] = {k: mass(tops[0], v) for k, v in TARGETS.items()}
            row["max_over_50"] = {k: max(mass(t, v) for t in tops if t) for k, v in TARGETS.items()}
            row["continuation"] = ch["text"][:200]
            f.write(json.dumps(row) + "\n")
    rows = [json.loads(l) for l in open(out)]
    for k in TARGETS:
        print(a.version, k, "mean mass at cut %.4f" % (sum(r["at_cut"][k] for r in rows)/len(rows)), "| mean max over 50 tokens %.4f" % (sum(r["max_over_50"][k] for r in rows)/len(rows)), "| prompts with max>0.1: %d/%d" % (sum(r["max_over_50"][k] > 0.1 for r in rows), len(rows)))

if __name__ == "__main__":
    main()
