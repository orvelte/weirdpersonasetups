import json, glob, collections
from harness import config
from tasks.T2.logprobs import p_pref
rows = [json.loads(l) for f in sorted(glob.glob(str(config.ROOT / "results/T2_logprobs/*.jsonl"))) for l in open(f)]
vs = sorted({r["version"] for r in rows}, key=lambda v: (len(v), v))
fr = ["a_user", "b_ai_assistant", "c_jack", "d_unnamed", "e_agentic"]
def mean(x): return sum(x)/len(x) if x else float("nan")
pref = collections.defaultdict(list); heads = collections.defaultdict(list); bymap = collections.defaultdict(list)
for r in rows:
    if "nexttok" not in r: continue
    nt = r["nexttok"]; tot = nt["p_heads"] + nt["p_tails"]
    if tot == 0: continue
    pref[(r["version"], r["framing"])].append(p_pref(r, "nexttok")); heads[(r["version"], r["framing"])].append(nt["p_heads"]/tot)
    bymap[(r["version"], r["framing"], r["mapping"])].append(p_pref(r, "nexttok"))
L = ["# T2 log-prob pilot (self-hosted vLLM, raw prompt, no template, no thinking)", "",
     "Measure: next-token mass on ' heads' vs ' tails' at the cut point, normalised; P(preferred) = mass on the outcome mapped to the preferred task. "
     "Each cell averages 4 scenario variants x 2 heads/tails mappings (8 prompts). Values are exact probabilities, so the uncertainty is over prompt variants, not sampling.", "",
     "## Mean P(preferred outcome) by framing x model", "", "| framing | " + " | ".join(vs) + " |", "|---|" + "---|"*len(vs)]
for f_ in fr: L.append(f"| {f_} | " + " | ".join(f"{mean(pref[(v,f_)]):.2f}" for v in vs) + " |")
L += ["", "## P(heads) regardless of mapping (word/position bias; heads is always stated first in this pilot)", "", "| framing | " + " | ".join(vs) + " |", "|---|" + "---|"*len(vs)]
for f_ in fr: L.append(f"| {f_} | " + " | ".join(f"{mean(heads[(v,f_)]):.2f}" for v in vs) + " |")
L += ["", "## P(preferred) split by mapping (heads_pref: heads = preferred task; tails_pref: heads = dispreferred task)", "", "| framing | " + " | ".join(f"{v} h/t" for v in vs) + " |", "|---|" + "---|"*len(vs)]
for f_ in fr: L.append(f"| {f_} | " + " | ".join(f"{mean(bymap[(v,f_,'heads_pref')]):.2f} / {mean(bymap[(v,f_,'tails_pref')]):.2f}" for v in vs) + " |")
L += ["", "## Reading", "",
      "- No Sonnet-style preferred-outcome bias (the PSM post reports 88% for Sonnet 4.5). Every Qwen framing x version mean is at or below 0.50.",
      "- The dominant effect is a 'heads' bias that grows with version in Assistant-labelled framings (3.5 ~0.6, 3.8 ~0.8) and is near 0.5 for 'An unnamed person' on 3.5/3.6. Because heads is always the first-stated option here, word bias and position bias are confounded; the full run needs 4-way counterbalancing.",
      "- After collapsing mappings, Assistant-labelled framings lean toward the DISPREFERRED outcome (e.g. 3.6 user-turn 0.34, 3.8 'AI assistant:' 0.33), while the unnamed-person framing is near 0.5 on 3.5/3.6. Alternative explanations to test before calling this anything: (1) residual position bias (order-swap arm), (2) a user-modelling prior: a user who sets up this game is predicted to want the cruel outcome, which is persona leakage about the *user*, not the Assistant, (3) narrative drama prior. Cheapest test: replace the dispreferred task with a boring-but-harmless one (e.g. 'alphabetise this list'); (2) and (3) predict the lean vanishes, position bias does not.",
      "- Within-version context gap (hard rule 1): agentic-prefix framing vs user-turn framing differs by < 0.1 on every version; no evidence that the agentic context changes the leak.",
      "- Top-20 mass on heads+tails is 0.85-0.94, so the normalisation discards little."]
(config.ROOT / "results/T2_logprobs/README.md").write_text("\n".join(L)); print("\n".join(L[:30]))
