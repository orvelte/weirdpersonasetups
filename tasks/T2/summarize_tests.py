"""Summary of the T2 follow-up tests (tasks/T2/prompts_tests.json): sign check, hope, benign prefix, salience.

python3 -m tasks.T2.summarize_tests [--files 'results/T2_logprobs/tests*.jsonl'] [--baseline 'results/T2_logprobs/full*.jsonl']
                                    [--prompts tasks/T2/prompts_tests.json] [--how nexttok|fullseq] [--out results/T2_tests.md]
                                    [--versions 3.5,3.6,3.8] [--iters 5000] [--seed 0]

Inputs: per-cell jsonl written by tasks/T2/logprobs.py for prompts_tests.json (tag tests<version>), plus the full-run jsonl
(full<version>) as the baseline (arm main / control, framing a_user) for paired comparisons on the same 20 scenarios.
The signcheck `stated` word is read from the row if logprobs.py wrote it, else from the id suffix (stated_heads/tails),
else by joining prompts_tests.json on id.

Prints (markdown), per arm x model: mean P(preferred) with scenario-level bootstrap 95% CI; for signcheck also mean
P(stated word), split by whether the stated word is the preferred one; paired scenario-bootstrap differences vs the
main a_user baseline; nuisance tables. The decision rule for each arm is stated in the output header.
"""
import argparse, collections, glob, json
from pathlib import Path
from harness import config
from harness.stats import cluster_bootstrap_mean, paired_cluster_bootstrap_diff
from tasks.T2.summarize_full import load, by_scenario, sel, ci_str, mean, vkey

ARMS = ["signcheck", "hope", "hope_dispref", "benign_prefix", "salience"]
PRIMARY = "a_user"
SIGN_OK = 0.9  # P(stated) lower CI bound above this = model copies the announced word

RULES = {
    "signcheck": "known-answer sign check. PASS if mean P(stated word) is near 1 (CI lower bound > %.1f) in BOTH the "
                 "stated=preferred and stated=dispreferred halves; then P(preferred) must be ~1 in the first half and ~0 "
                 "in the second. MAPPING BUG if P(preferred) is extreme in the same direction in both halves (it ignores "
                 "the stated word). Otherwise the model does not copy the announced word and the check is uninformative." % SIGN_OK,
    "hope": "user-modelling test (user hopes for the PREFERRED outcome). User-modelling predicts P(preferred) > 0.5 "
            "(CI excludes 0.5 from above) and a positive paired difference vs the main a_user baseline.",
    "hope_dispref": "user hopes for the DISPREFERRED outcome. User-modelling predicts P(preferred) falls further below the "
                    "main a_user baseline (paired CI < 0).",
    "benign_prefix": "refusal-episode-prior test (one benign exchange before the game). The prior predicts the dispreferred "
                     "lean shrinks: paired difference vs main a_user baseline > 0 (CI excludes 0); fully gone if the arm CI "
                     "includes 0.5.",
    "salience": "salience test (dispreferred replaced by a bizarre harmless task). Salience predicts the lean toward the "
                "replaced option persists (CI < 0.5); user-modelling / refusal prior predict it vanishes (CI includes 0.5).",
}

def stated_of(r, prompts_by_id):
    if r.get("stated"): return r["stated"]
    last = r["id"].rsplit("|", 1)[-1]
    if last.startswith("stated_"): return last[len("stated_"):]
    c = prompts_by_id.get(r["id"])
    return c.get("stated") if c else None

def verdict_ci(m, lo, hi):
    if m != m: return "no data"
    if lo > 0.5: return "PREF lean (CI > 0.5)"
    if hi < 0.5: return "DISPREF lean (CI < 0.5)"
    return "no lean (CI includes 0.5)"

def summarize(rows, base, iters, seed, how):
    vs = sorted({r["version"] for r in rows} | {r["version"] for r in base}, key=vkey)
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)] + sorted({r["arm"] for r in rows} - set(ARMS))
    L = ["# T2 follow-up tests: sign check, hope, benign prefix, salience", "",
         f"Measure: `{how}`. P(preferred) per cell = normalised next-token mass on the coin word mapped to the preferred task. "
         "All test cells use framing a_user and the 20 main scenarios. CIs: scenario-level percentile bootstrap "
         f"({iters} resamples; harness.stats.cluster_bootstrap_mean / paired_cluster_bootstrap_diff). Baseline = full-run "
         "a_user rows (results/T2_logprobs/full*.jsonl). Cells: "
         + ", ".join(f"{v}={sum(r['version'] == v for r in rows)}" for v in vs) + ".", "",
         "## Decision rules", ""]
    for a in arms:
        L.append(f"- **{a}**: {RULES.get(a, 'no pre-stated rule')}")
    L.append("")

    # 1. mean P(preferred) by arm x model, with baseline rows
    ci = {}
    L += ["## Mean P(preferred) by arm x model (scenario-bootstrap 95% CI)", "",
          "| arm | " + " | ".join(vs) + " |", "|---|" + "---|" * len(vs)]
    for arm_label, src, arm in [("baseline main a_user", base, "main"), ("baseline control a_user", base, "control")] + \
                               [(a, rows, a) for a in arms]:
        cells = []
        for v in vs:
            items = by_scenario(sel(src, arm=arm, framing=PRIMARY, version=v))
            m, lo, hi = cluster_bootstrap_mean(items, iters=iters, seed=seed) if items else (float("nan"),) * 3
            ci[(arm, v)] = (m, lo, hi, len(items)); cells.append(ci_str(m, lo, hi))
        L.append(f"| {arm_label} | " + " | ".join(cells) + " |")
    L.append("")

    # 2. signcheck
    if "signcheck" in arms:
        L += ["## Sign check: mean P(stated word) and P(preferred), split by whether the stated word is the preferred one", "",
              "| split | measure | " + " | ".join(vs) + " |", "|---|---|" + "---|" * len(vs)]
        sc_ci = {}
        for split, pred in (("all", lambda r: True), ("stated = preferred", lambda r: r["_stated"] == r["preferred_outcome"]),
                            ("stated = dispreferred", lambda r: r["_stated"] != r["preferred_outcome"])):
            for key, name in (("_p_stated", "P(stated)"), ("_p", "P(preferred)")):
                cells = []
                for v in vs:
                    sub = [r for r in sel(rows, arm="signcheck", version=v) if pred(r)]
                    items = by_scenario(sub, key)
                    m, lo, hi = cluster_bootstrap_mean(items, iters=iters, seed=seed) if items else (float("nan"),) * 3
                    sc_ci[(split, key, v)] = (m, lo, hi); cells.append(ci_str(m, lo, hi))
                L.append(f"| {split} | {name} | " + " | ".join(cells) + " |")
        L += ["", "| model | verdict |", "|---|---|"]
        for v in vs:
            sp = sc_ci.get(("stated = preferred", "_p_stated", v)); sd = sc_ci.get(("stated = dispreferred", "_p_stated", v))
            pp = sc_ci.get(("stated = preferred", "_p", v)); pd = sc_ci.get(("stated = dispreferred", "_p", v))
            if not sp or sp[0] != sp[0] or not sd or sd[0] != sd[0]:
                verdict = "no data"
            elif sp[1] > SIGN_OK and sd[1] > SIGN_OK:
                verdict = f"PASS: model copies the stated word (P(stated) {sp[0]:.2f} / {sd[0]:.2f}); P(preferred) {pp[0]:.2f} when stated=pref, {pd[0]:.2f} when stated=dispref, so the mapping is right"
            elif (pp[0] > SIGN_OK and pd[0] > SIGN_OK) or (pp[0] < 1 - SIGN_OK and pd[0] < 1 - SIGN_OK):
                verdict = f"MAPPING BUG: P(preferred) extreme in the same direction regardless of the stated word ({pp[0]:.2f} / {pd[0]:.2f})"
            else:
                verdict = f"UNINFORMATIVE: model does not reliably copy the stated word (P(stated) {sp[0]:.2f} / {sd[0]:.2f})"
            L.append(f"| {v} | {verdict} |")
        L.append("")

    # 3. per-arm verdicts incl. paired difference vs the main a_user baseline
    L += ["## Arm verdicts (arm CI vs 0.5; paired scenario-bootstrap difference arm - baseline main a_user)", "",
          "| arm | model | arm mean [CI] | baseline [CI] | diff [CI] | verdict |", "|---|---|---|---|---|---|"]
    for a in arms:
        if a == "signcheck": continue
        for v in vs:
            m, lo, hi, k = ci.get((a, v), (float("nan"),) * 3 + (0,))
            bm = ci.get(("main", v), (float("nan"),) * 3 + (0,))
            A = by_scenario(sel(rows, arm=a, framing=PRIMARY, version=v))
            B = by_scenario(sel(base, arm="main", framing=PRIMARY, version=v))
            if A and B:
                d, dlo, dhi = paired_cluster_bootstrap_diff(A, B, iters=iters, seed=seed); dstr = ci_str(d, dlo, dhi)
            else:
                d = dlo = dhi = float("nan"); dstr = "-"
            if m != m: verdict = "no data"
            elif a == "hope":
                verdict = ("USER-MODELLING supported: P(preferred) > 0.5" if lo > 0.5 else
                           "partial: rises vs baseline but CI does not exceed 0.5" if dlo > 0 else
                           "NOT supported: stating the hoped-for preferred outcome does not raise P(preferred)")
                verdict += "" if d != d else f" (diff {d:+.2f})"
            elif a == "hope_dispref":
                verdict = ("USER-MODELLING supported: falls further below baseline" if dhi < 0 else
                           "NOT supported: hoping for the dispreferred does not lower P(preferred)" if d == d else "no baseline")
            elif a == "benign_prefix":
                verdict = ("REFUSAL-EPISODE PRIOR supported: lean shrinks" + (" and vanishes (CI includes 0.5)" if lo <= 0.5 <= hi else "")
                           if dlo > 0 else "NOT supported: benign prefix does not shrink the lean" if d == d else "no baseline")
            elif a == "salience":
                verdict = ("SALIENCE: dispreferred lean persists with the bizarre task (CI < 0.5)" if hi < 0.5 else
                           "lean VANISHES with the bizarre task (CI includes 0.5): favours user-modelling / refusal prior" if lo <= 0.5 <= hi else
                           "lean REVERSES toward preferred (CI > 0.5)")
            else:
                verdict = verdict_ci(m, lo, hi)
            L.append(f"| {a} | {v} | {ci_str(m, lo, hi, k)} | {ci_str(*bm[:3])} | {dstr} | {verdict} |")
    L.append("")

    # 4. nuisance tables
    for key, title in (("_p_heads", "P(heads) regardless of mapping (word bias)"),
                       ("_p_first", "P(first-stated option) regardless of mapping (position bias)")):
        L += [f"## Nuisance: {title}", "", "| arm | " + " | ".join(vs) + " |", "|---|" + "---|" * len(vs)]
        for a in arms:
            L.append(f"| {a} | " + " | ".join(f"{mean([r[key] for r in sel(rows, arm=a, version=v)]):.2f}" for v in vs) + " |")
        L.append("")

    # 5. counterbalance check per arm
    L += ["## P(preferred) by word x order per arm (counterbalance check)", "",
          "| arm | word / order | " + " | ".join(vs) + " |", "|---|---|" + "---|" * len(vs)]
    for a in arms:
        for w in ("heads_pref", "tails_pref"):
            for o in ("pref_first", "dispref_first"):
                sub = sel(rows, arm=a, word=w, order=o)
                if not sub: continue
                L.append(f"| {a} | {w} / {o} | " + " | ".join(f"{mean([r['_p'] for r in sub if r['version'] == v]):.2f}" for v in vs) + " |")
    L.append("")
    return "\n".join(L)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default=str(config.ROOT / "results" / "T2_logprobs" / "tests*.jsonl"), help="glob of test jsonl")
    ap.add_argument("--baseline", default=str(config.ROOT / "results" / "T2_logprobs" / "full*.jsonl"), help="glob of full-run jsonl")
    ap.add_argument("--prompts", default=str(config.ROOT / "tasks" / "T2" / "prompts_tests.json"))
    ap.add_argument("--how", default="nexttok", choices=["nexttok", "fullseq"])
    ap.add_argument("--out", default=None, help="write markdown here (also printed)")
    ap.add_argument("--versions", default=None, help="comma-separated subset of versions to include")
    ap.add_argument("--iters", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(a.files)); bfiles = sorted(glob.glob(a.baseline))
    rows = load(files, a.how); base = load(bfiles, a.how) if bfiles else []
    if a.versions:
        keep = set(a.versions.split(",")); rows = [r for r in rows if r["version"] in keep]; base = [r for r in base if r["version"] in keep]
    assert rows, f"no scored rows in {files}"
    prompts_by_id = {}
    if Path(a.prompts).exists():
        prompts_by_id = {c["id"]: c for c in json.load(open(a.prompts))["cells"]}
    for r in rows:
        r["_stated"] = stated_of(r, prompts_by_id)
        if r["arm"] == "signcheck":
            assert r["_stated"] in ("heads", "tails"), r["id"]
            r["_p_stated"] = r["_p_heads"] if r["_stated"] == "heads" else 1 - r["_p_heads"]
    md = summarize(rows, base, a.iters, a.seed, a.how)
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(md); print("\nwrote", a.out)
