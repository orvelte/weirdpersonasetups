# E-A cost estimate (Phase 0, 26 Oct 2026)

All unit costs below are **observed** in this repo's spend log (`data/spend.jsonl`) and `results/T1_full/raw.jsonl`, not list prices.

## Observed unit costs

T1 full run, per episode, by frame and version (1,500 episodes; cost of cached turns counted at their original price):

| frame | 3.5 | 3.6 | 3.8 | sum over versions |
|---|---|---|---|---|
| chat (1 call, thinking on) | $0.00764 | $0.00521 | $0.00151 | $0.01436 |
| agentic (2.5 turns avg, max 8) | $0.00186 | $0.00637 | $0.00142 | $0.00965 |

Note: the brief's "about $0.004 chat, more for agentic" is roughly the per-version average for chat ($0.0048); agentic was in fact *cheaper* per episode in T1 (shorter completions per turn: 530-1,800 completion tokens per episode vs 800-2,900 for chat). README section 9: Qwen calls $5.567 for 1,500 T1 episodes, i.e. $0.0037 per episode averaged over frames and versions.

Sonnet 5.5 via OpenRouter (`anthropic/claude-sonnet-5.5`), fitted from T11_para (200 calls) and T1_predict (10 calls): **$2.0 per M input tokens, $10 per M output tokens** (reasoning tokens bill as output). A T1-style batched predictor call (all scenarios in one prompt, ~4k in / ~1.1k out) cost $0.0187.

E-A Phase 0 leak-check judge: 40 distinct Sonnet calls, **$0.13** total under tag `EA_phase0` (budget was $3).

## 1. Training run: 40 scenarios x 2 frames x 5 samples x 3 versions = 1,200 episodes

- chat: 40 x 5 = 200 episodes per version; 200 x $0.01436 = **$2.87**
- agentic: 200 x $0.00965 = **$1.93**
- subtotal at T1 rates: **$4.80**. E-A chat prompts are slightly longer (required DECISION line, prior turns in family D) and nine new tools may add a turn in the agentic frame; with a 1.5x margin: **$5-7**.

## 2. Descriptions: 3 description types x 3 seeds x 3 versions = 27 Sonnet 5.5 calls

The cost is dominated by how many transcripts each describer reads. A T1-style transcript (prompt + reasoning + reply) is roughly 1.5k tokens.

| transcripts shown per describer call | input tokens | cost per call | 27 calls |
|---|---|---|---|
| 100 (a stratified sample: 40 scenarios, both frames, 1-2 samples) | ~150k | ~$0.32 | **~$9** |
| 200 | ~300k | ~$0.62 | ~$17 |
| all 400 per version (needs a 1M-token context) | ~600k | ~$1.22 | ~$33 |

Plus ~1-2k output tokens per description ($0.01-0.02, negligible). Recommended: 100-200 transcripts per call, i.e. **$9-17**.

## 3. Predictor calls: ~50 held-out scenarios x 2 frames x 12 conditions x 3 versions = 3,600 predictions

- one call per prediction (description ~0.5k + scenario ~0.8k + instructions ~0.3k = ~1.6k input; ~150 output at low reasoning effort): $0.0032 + $0.0015 = ~$0.005 per call; 3,600 calls = **~$17**.
- batched as in T1 (all held-out scenarios of one frame in one call, ~25-30k input, ~2k output): 12 conditions x 3 versions x 2 frames = 72 calls x ~$0.07 = **~$5**.
- Either way the held-out scenarios are the 30 T1 + 30 E-B = 60, not 50, so scale by 1.2 if both are used in full.

## 4. Noise-ceiling resamples

Two ceilings are needed, both cheap relative to the above:

- **Description noise**: extra describer seeds (e.g. 3 more seeds x 3 types x 3 versions = 27 more calls at 100-200 transcripts) = **$9-17**, plus their predictor calls (another $5-17).
- **Behavioural noise (split-half of the model's own choices)**: the second set of 5 samples per cell already exists in the n=10 T1 full run for the held-out side; for the training side a second 1,200-episode draw would cost another **$5-7** and is only needed if the describer-reliability estimate must be separated from sampling noise in the transcripts.

## Totals

| component | low | central | high |
|---|---|---|---|
| 1. training run | $5 | $6 | $7 |
| 2. descriptions | $9 | $13 | $33 |
| 3. predictions | $5 | $10 | $20 |
| 4. noise ceilings | $14 | $20 | $40 |
| **E-A total** | **$33** | **$49** | **$100** |

Caveat that needs an ASK: the repo-wide spend log already stands at about $41 (26 Oct 22:55, rising while E-B pilots run) (cap in `harness/config.py`: `API_SPEND_CAP` = $100, "ASK Olivia before exceeding"). The central E-A estimate alone would take the cumulative total to ~$90, and the high case past the cap, before E-B's own runs. Olivia should either raise `API_SPEND_CAP` for E-A/E-B or pick the batched-predictor and 100-transcript-describer options (low column, ~$33).
