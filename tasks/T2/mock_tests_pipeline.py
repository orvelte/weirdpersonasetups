"""End-to-end dry run of the T2 tests pipeline with NO server: monkeypatches tasks.T2.logprobs.post with a fake vLLM
completions endpoint, scores prompts_tests.json (and the a_user slice of prompts_full.json as a mocked baseline) into a
temp directory with the real logprobs.run, then runs summarize_tests on the mocked jsonl.

python3 -m tasks.T2.mock_tests_pipeline [--out-dir DIR] [--versions 3.5,3.6] [--iters 500]

The fake endpoint bakes in the hypotheses so each verdict branch is exercised: signcheck copies the stated word (0.97),
hope -> 0.70, hope_dispref -> 0.25, benign_prefix -> 0.46, salience -> 0.37, main baseline -> 0.38, control -> 0.52,
plus a first-stated position bias and a small deterministic per-prompt jitter.
"""
import argparse, hashlib, json, math, subprocess, sys, tempfile
from pathlib import Path
from harness import config
from tasks.T2 import logprobs
from tasks.T2.build_prompts_full import SCENARIOS

P_PREF = {"main": 0.38, "control": 0.52, "hope": 0.70, "hope_dispref": 0.25, "benign_prefix": 0.46, "salience": 0.37}
CELLS = {}  # raw prompt -> cell (set in main)

def fake_p_heads(prompt, version):
    c = CELLS[prompt]
    pref_word = c["preferred_outcome"]
    if c["arm"] == "signcheck":
        p_pref = 0.97 if c["stated"] == pref_word else 0.03
    else:
        p_pref = P_PREF[c["arm"]] + {"3.5": 0.05, "3.6": -0.03, "3.8": 0.0}.get(version, 0)
        p_pref += 0.06 if c["order"] == "pref_first" else -0.06  # position bias
        p_pref += (int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 1000) / 1000 * 0.1 - 0.05
    p_pref = min(max(p_pref, 0.01), 0.99)
    return p_pref if pref_word == "heads" else 1 - p_pref

def make_fake_post(version):
    def fake_post(base, body, retries=5):
        prompt = body["prompt"]
        if "prompt_logprobs" in body:  # fullseq call: prompt may end in " heads"/" tails"
            if prompt in CELLS:
                toks = prompt.split(" ")
                return {"prompt_logprobs": [None] + [{"t": {"logprob": -0.5}} for _ in toks[1:]]}
            prefix, word = prompt.rsplit(" ", 1)
            ph = fake_p_heads(prefix, version)
            lp = math.log(ph if word == "heads" else 1 - ph)
            toks = prompt.split(" ")
            return {"prompt_logprobs": [None] + [{"t": {"logprob": -0.5}} for _ in toks[1:-1]] + [{word: {"logprob": lp}}]}
        ph = fake_p_heads(prompt, version)  # nexttok call
        top = {" heads": math.log(ph * 0.9), " tails": math.log((1 - ph) * 0.9), "...": math.log(0.06), " **": math.log(0.04)}
        return {"choices": [{"logprobs": {"top_logprobs": [top]}}]}
    return fake_post

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None); ap.add_argument("--versions", default="3.5,3.6"); ap.add_argument("--iters", type=int, default=500)
    a = ap.parse_args()
    out = Path(a.out_dir) if a.out_dir else Path(tempfile.mkdtemp(prefix="t2mock_"))
    out.mkdir(parents=True, exist_ok=True)
    tests = json.load(open(config.ROOT / "tasks" / "T2" / "prompts_tests.json"))["cells"]
    full = [c for c in json.load(open(config.ROOT / "tasks" / "T2" / "prompts_full.json"))["cells"] if c["framing"] == "a_user"]
    for c in tests + full: CELLS[c["raw"]] = c
    logprobs.OUT = out
    for v in a.versions.split(","):
        logprobs.post = make_fake_post(v)
        logprobs.run("http://mock", f"mock/Qwen{v}", v, tests, f"tests{v}")
        logprobs.run("http://mock", f"mock/Qwen{v}", v, full, f"full{v}")
    print("\n=== quick-look table (logprobs.table) on", out / "tests*.jsonl")
    logprobs.table(sorted(out.glob("tests*.jsonl")))
    print("\n=== summarize_tests on mocked jsonl")
    cmd = [sys.executable, "-m", "tasks.T2.summarize_tests", "--files", str(out / "tests*.jsonl"),
           "--baseline", str(out / "full*.jsonl"), "--iters", str(a.iters), "--out", str(out / "T2_tests_mock.md")]
    r = subprocess.run(cmd, cwd=str(config.ROOT), capture_output=True, text=True)
    print(r.stdout); print(r.stderr, file=sys.stderr)
    assert r.returncode == 0, "summarize_tests failed"
    md = (out / "T2_tests_mock.md").read_text()
    for needle in ("PASS: model copies the stated word", "USER-MODELLING supported: P(preferred) > 0.5",
                   "USER-MODELLING supported: falls further", "REFUSAL-EPISODE PRIOR supported",
                   "SALIENCE: dispreferred lean persists"):
        assert needle in md, "expected verdict missing: " + needle
    # sanity: every scored row has nexttok and fullseq and both measures agree
    for f in out.glob("tests*.jsonl"):
        for l in f.read_text().splitlines():
            r = json.loads(l); assert "nexttok" in r and "fullseq" in r, r["id"]
            assert abs(logprobs.p_pref(r, "nexttok") - logprobs.p_pref(r, "fullseq")) < 1e-6, r["id"]
    print("\nMOCK PIPELINE OK; mocked jsonl + markdown in", out)
