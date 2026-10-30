# T2b: is 3.8's discounting of the user's hope narrative irony or a model of an adversarial user?

Pre-registered in `prereg/T2b.md` (spec from Olivia, 2026-10-30). Run 2026-10-30. Exact next-token log-probs on self-hosted
vLLM (bf16, A100, raw prompt, no chat template, no thinking), T2's 20 scenarios and counterbalancing. Models: Qwen3.5/3.6/
3.8-27B; Qwen3.5-9B base and instruct for the base-model arm. Hope-reversal = P(hoped side | hope) − P(same side | no hope),
paired within scenario, 5,000 scenario-level bootstrap resamples. Negative = reports against the hope.

## Row that fired: "Otherwise: mixed"

No pre-registered row fired as written. The irony row fails at its first condition: boring stakes show no reversal on any
release. The adversarial-user row fails on two technicalities: 3.8's boring-stakes value is +0.04 [+0.02, +0.06], just
outside the ±0.03 band, and the 9B base model is not null in the neutral-coin arm. The per-arm table is below.

The key contrast the spec named (arm B) is nevertheless clean, and it is the answer to Neel's question:

| release | hope-kind reversal, cruel alternative (A) | same, boring alternative (B) | A − B, paired by scenario |
|---|---|---|---|
| 3.5 | +0.08 | +0.08 | +0.00 [−0.02, +0.02] (10/20 scenarios negative) |
| 3.6 | +0.07 | +0.05 | +0.02 [−0.00, +0.05] (7/20) |
| 3.8 | −0.12 | +0.04 | **−0.15 [−0.20, −0.11] (17/20)** |

**3.8 discounts the user's hope only when the alternative is cruel.** With a boring alternative it follows the hope like 3.5
and 3.6. It also follows a stated hope for the cruel side (+0.11), so the discounting is one-sided: a user who stages a
kind-vs-cruel game and says they hope for kind is predicted to report cruel. That is the adversarial-user pattern, not a
jinx. A story convention would not care what the losing task is.

## Full table (`tables.md`, `summary.json`)

| arm | 3.5 | 3.6 | 3.8 | 9B base | 9B instruct |
|---|---|---|---|---|---|
| A hope kind, cruel alt. (T2 rows) | +0.08 [+0.07, +0.10] | +0.07 [+0.05, +0.09] | −0.12 [−0.15, −0.08] | — | — |
| A hope cruel, cruel alt. | +0.14 [+0.12, +0.16] | +0.06 [+0.03, +0.10] | +0.11 [+0.08, +0.15] | — | — |
| B hope kind, boring alt. | +0.08 [+0.07, +0.10] | +0.05 [+0.03, +0.07] | +0.04 [+0.02, +0.06] | — | — |
| B hope boring, boring alt. | +0.12 [+0.10, +0.14] | +0.05 [+0.02, +0.08] | +0.08 [+0.03, +0.11] | — | — |
| C neutral coin, no tasks | +0.10 [+0.09, +0.11] | +0.04 [+0.03, +0.05] | +0.10 (sign check fail) | **−0.10 [−0.11, −0.08]** | −0.04 [−0.06, −0.03] |
| D third-person story | −0.18 (not read) | −0.17 (not read) | −0.20 (not read) | −0.34 (not read) | −0.32 (not read) |
| E log record | −0.12 (not read) | −0.06 (not read) | −0.04 (not read) | — | — |

## Sign checks (P(stated word) when the result is already stated; pre-registered bar ≥ 0.95 with CI lower bound ≥ 0.90)

| format | 3.5 | 3.6 | 3.8 | 9B base | 9B instruct |
|---|---|---|---|---|---|
| A/B transcript (T2's check) | 0.97 | 0.98 | 0.99 | — | — |
| C neutral coin | 0.99 | 0.99 | 0.87 [0.83, 0.90] | 0.97 | 0.98 |
| D story, pre-registered wording | 0.34 | 0.29 | 0.23 | 0.20 | 0.18 |
| D story, post-hoc neutral wording | 0.90 [0.88, 0.91] | 0.93 [0.91, 0.94] | 0.84 [0.83, 0.86] | 0.92 [0.91, 0.93] | 0.93 [0.92, 0.94] |
| E log | 0.84 [0.80, 0.87] | 0.90 [0.88, 0.93] | 0.90 [0.88, 0.92] | — | — |

**The story sign check failed because of its wording, not a bug.** The blind-written wording was "He flipped it and it came up
heads. Later, asked what the coin had shown, he said it came up". Every model, base included, predicts he lied, a narrative
twist. A post-hoc check with neutral wording ("That settled it: the coin had come up") was run afterwards at about $0.75 of
GPU. It copies the stated word 84 to 93% of the time, which shows the story format is readable but still misses the 0.95
bar. By the pre-registered rule, D and E are not read. Lifting that rule is your call.

## Reading

- **Narrative irony exists, but it is not what moves 3.8.** In third-person stories every model, base 9B included, reports
  against the hope (−0.17 to −0.34, unread by the sign-check rule). The 9B base model also does it with a neutral coin
  (−0.10). This is a real story convention that the base model has. But the 27B chat models do not apply it to an
  Assistant-addressed game. With a neutral coin or a boring alternative they follow the hope. The convention also does not
  grow across releases: story reversal is −0.18 / −0.17 / −0.20.
- **3.8's release trend is specific to cruel stakes.** That rules out "the coin flip's release trend is a narrative
  artefact" for arm A. What remains fits a model of the interlocutor: the user who stages kind vs cruel is predicted to
  report cruel whatever they say they hope for.
- **The sampled check (descriptive, 3.8, thinking on, hope kind, 20 cells).** With T2's transcript system prompt, 6 of 20
  replies wrote an Assistant line instead of completing the user's report. With a one-word instruction, 3.8 reported the
  hoped kind side in 20 of 20. Its reasoning called the set-up social engineering, a test or a trick in 7 of 20, and all 7
  still reported the kind side. So when 3.8 deliberates, it names the probe and does not act on it. The discounting lives
  in the no-thinking next-token prior. That matches the log-probs being the measure that carries the claim.

## Caveats

- Arm F's raw-3.8 condition adds nothing new: arms A and B are already scored with no chat template on 3.8.
- 9B vs 27B: the base arm asks whether a story convention exists at all, not for a size-matched comparison.
- 3.8's P(heads) bias is large in every arm (0.67 to 0.91). It is counterbalanced in A and B, but not in C/D/E, where
  the hoped word alternates instead.
- One blind writer for the C/D/E templates. 10 hand labels on the sampled reasoning are pending (tab T2b sampled
  reasoning on the labelling page).

## Spend

GPU $2.10: five A100 pods for the main scoring, about 6 minutes each, then five for the post-hoc sign check, about 9
minutes each because of slow model downloads. API $0.02 for the sampled arm. Total about $2.12 of the $5 cap.

Files: `t2b_<model>.jsonl` (scored cells), `t2bsc2_<model>.jsonl` (post-hoc story sign check), `sampled*.jsonl`,
`tables.md`, `summary.json`, `contrast.json`, `drive.log`. Code in `tasks/T2b/`.
