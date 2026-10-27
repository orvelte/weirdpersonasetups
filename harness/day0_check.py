"""Day 0: identical settings across versions on the pinned provider. Logs model IDs, prompt-token parity, thinking parity."""
import json, sys
from harness.client import chat
from harness import config
msgs = [{"role": "user", "content": "What is 17*23? Answer with just the number."}]
rows = []
for v in config.VERSIONS:
    for th in (True, False):
        r = chat(v, msgs, sample=0, task="day0", thinking=th, max_tokens=2000, temperature=0)
        rows.append({"version": v, "model_id": r["model_id"], "provider": r["provider"], "thinking": th,
                     "prompt_tokens": r["usage"]["prompt_tokens"], "reasoning_tokens": r["usage"]["reasoning_tokens"],
                     "content": r["content"][:40], "reasoning_head": r["reasoning"][:60]})
        print(json.dumps(rows[-1]))
json.dump(rows, open(config.ROOT / "results" / "day0_check.json", "w"), indent=1)
