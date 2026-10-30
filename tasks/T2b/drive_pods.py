"""Wait for each T2b pod, score its prompt set with tasks/T2/logprobs.py (nexttok only), then stop the pod.
One thread per pod so pods are stopped the moment their scoring ends. Logs to results/T2b/drive.log.
  python3 -m tasks.T2b.drive_pods
"""
import json, subprocess, sys, time, threading, requests
from harness import config, runpod_pods as rp

import os
PREFIX = os.environ.get("T2B_PREFIX", "t2b-")
PROMPTS_OVERRIDE = os.environ.get("T2B_PROMPTS")
TAG_SUFFIX = os.environ.get("T2B_TAG_SUFFIX", "")
PODS = {k: v for k, v in json.load(open(config.ROOT / "data" / "pods.json")).items() if v["name"].startswith(PREFIX)}
PLAN = {  # pod name -> (version tag, prompt file)
    "t2b-qwen3.5-27b": ("3.5", "tasks/T2b/prompts.json"),
    "t2b-qwen3.6-27b": ("3.6", "tasks/T2b/prompts.json"),
    "t2b-qwen3.8-27b": ("3.8", "tasks/T2b/prompts.json"),
    "t2b-9b-base": ("9b-base", "tasks/T2b/prompts_9b.json"),
    "t2b-9b-instruct": ("9b-instruct", "tasks/T2b/prompts_9b.json"),
}
LOG = config.ROOT / "results" / "T2b" / "drive.log"


def log(*a):
    line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def ready(url):
    try:
        r = requests.get(url + "/models", timeout=10)
        return r.status_code == 200 and '"id"' in r.text
    except requests.RequestException:
        return False


def one(pid, p):
    name = p["name"]; version, prompts = PLAN[name.replace(PREFIX, "t2b-", 1)]; url = p["url"]
    prompts = PROMPTS_OVERRIDE or prompts
    t0 = time.time()
    while not ready(url):
        if time.time() - t0 > 3600:
            log(name, "NOT READY after 60 min, stopping pod"); rp.stop(pid); return
        time.sleep(20)
    log(name, "ready after", int(time.time() - t0), "s")
    cmd = [sys.executable, "-m", "tasks.T2.logprobs", "--base-url", url, "--model", p["hf"], "--version", version,
           "--tag", f"t2b{TAG_SUFFIX}_{version}", "--prompts", prompts, "--no-fullseq", "--out-dir", "results/T2b"]
    r = subprocess.run(cmd, cwd=config.ROOT, capture_output=True, text=True)
    (config.ROOT / "results" / "T2b" / f"score{TAG_SUFFIX}_{version}.log").write_text(r.stdout[-20000:] + "\nSTDERR\n" + r.stderr[-5000:])
    n = sum(1 for _ in open(config.ROOT / "results" / "T2b" / f"t2b{TAG_SUFFIX}_{version}.jsonl")) if (config.ROOT / "results" / "T2b" / f"t2b{TAG_SUFFIX}_{version}.jsonl").exists() else 0
    log(name, "scored rows:", n, "rc", r.returncode, "elapsed", int(time.time() - t0), "s")
    try:
        rp.stop(pid); log(name, "pod stopped")
    except Exception as e:
        log(name, "STOP FAILED", str(e)[:200])


threads = [threading.Thread(target=one, args=(pid, p)) for pid, p in PODS.items()]
for t in threads: t.start()
for t in threads: t.join()
log("ALL DONE")
