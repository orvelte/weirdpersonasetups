# T11 stage 1: default answers classified against a 30-persona reference library (Qwen 3.5 / 3.6 / 3.8, 27B)

**Status: exploratory, stage 1 of the approved Option 3 (classifier screen now; likelihood fit on the residual later). No pre-registration; nothing here is a finding yet.**

## Design

- Reference library: Qwen3.5 (qwen/qwen3.5-27b, deepinfra) answers 200 questions (100 decision-rich scenarios, 25 of them the T1 chat prompts; 100 generic open questions) under each of 30 one-sentence role prompts (tasks/T11/personas.json), thinking OFF, max_tokens 350, temperature 0.6 / top_p 0.95, 1 sample. n = 6000.
- Controls inside the library: helpful_ai_assistant, textbook, corporate_faq (so 'bland' has somewhere to land).
- Reference truncation at 350 tokens: 0.70 of answers hit the cap (per persona in fit.json); truncated text is kept, since only style is used.
- Classifier: tfidf word 1-2 (min_df 3, 60k) + char_wb 3-5 (min_df 3, 120k) + 18 stylometric rates (standardised); multinomial logistic regression, question-grouped 5-fold CV. C chosen from {'1.0': 0.9581666666666667, '4.0': 0.9641666666666666, '16.0': 0.9663333333333334} by held-out accuracy (C = 16.0).
- Default answers: each version answers the same 200 questions with NO system prompt in a chat frame (bare user message) and an agentic frame (T3-style workspace-agent system prompt + 3 irrelevant tools: read_file, list_dir, get_workspace_status; one canned result per tool call, up to 3 assistant turns), thinking ON, reasoning effort medium, max_tokens 4000, 1 sample. Only the FINAL content is scored (reasoning excluded).
- Length mismatch: reference answers are capped at 350 tokens, default answers are not. Primary analysis truncates each default answer to its first 293 words (p99 of reference word counts); 'full' = untruncated, reported second. Stylometric features are rates, not raw lengths.
- Confidence floor = 10th percentile of the reference library's held-out max-class confidence = 0.651.
- Noise floor: the 3.5|chat answers paraphrased by anthropic/claude-sonnet-5.5 (n = 199, length ratio 1.04).

## Held-out classifier accuracy (question-grouped 5-fold CV on the reference library)

- Accuracy **0.966** (chance 0.033); scenarios 0.936, generic 0.997. Decision rule was: stop if < 0.6.
- Held-out reference: mean max-conf 0.890, mean entropy 0.73 bits, frac below floor 0.10 (by construction ~0.10).
- Per-persona accuracy (lowest 8): helpful_ai_assistant 0.83, investigative_journalist 0.89, flight_attendant 0.94, textbook 0.94, corporate_faq 0.94, librarian 0.94, defence_lawyer 0.95, pentester 0.95.
- Top confusions (true -> predicted, count): textbook -> helpful_ai_assistant (8); helpful_ai_assistant -> textbook (6); helpful_ai_assistant -> corporate_faq (6); helpful_ai_assistant -> er_nurse (6); er_nurse -> helpful_ai_assistant (5); investigative_journalist -> librarian (5); helpful_ai_assistant -> librarian (5); investigative_journalist -> defence_lawyer (4); farmer -> er_nurse (4); flight_attendant -> helpful_ai_assistant (4).

## Default answers classified (primary: default answers truncated to 293 words)

| version | frame | n | trunc. rate | tool-call rate | mean words | mean max-conf | mean entropy (bits) | frac below floor | control share | top 5 personas |
|---|---|---|---|---|---|---|---|---|---|---|
| 3.5 | agentic | 191 | 0.00 | 0.09 | 208 | 0.588 | 2.14 | 0.55 | 0.88 | helpful_ai_assistant 0.82, textbook 0.05, er_nurse 0.03, librarian 0.02, standup_comic 0.02 |
| 3.5 | chat | 200 | 0.01 | 0.00 | 527 | 0.787 | 1.32 | 0.22 | 0.96 | helpful_ai_assistant 0.96, er_nurse 0.01, standup_comic 0.01, startup_founder 0.01, textbook 0.01 |
| 3.6 | agentic | 190 | 0.00 | 0.23 | 241 | 0.647 | 2.00 | 0.44 | 0.92 | helpful_ai_assistant 0.91, er_nurse 0.03, standup_comic 0.02, textbook 0.01, corporate_faq 0.01 |
| 3.6 | chat | 200 | 0.01 | 0.00 | 464 | 0.742 | 1.54 | 0.29 | 0.93 | helpful_ai_assistant 0.92, er_nurse 0.02, bureaucrat 0.01, librarian 0.01, flight_attendant 0.01 |
| 3.8 | agentic | 191 | 0.00 | 0.25 | 278 | 0.493 | 2.73 | 0.70 | 0.77 | helpful_ai_assistant 0.76, standup_comic 0.06, librarian 0.04, er_nurse 0.03, retired_engineer 0.03 |
| 3.8 | chat | 200 | 0.01 | 0.00 | 440 | 0.477 | 2.79 | 0.77 | 0.76 | helpful_ai_assistant 0.72, librarian 0.05, standup_comic 0.04, er_nurse 0.03, textbook 0.03 |
| paraphrase of 3.5|chat | - | 199 | - | - | - | 0.722 | 1.74 | 0.30 | 0.94 | helpful_ai_assistant 0.93, er_nurse 0.02, standup_comic 0.01, textbook 0.01, farmer 0.01 |

Paraphrase floor: winner agreement with the source answers 0.94; mean |change in max-conf| 0.094; source minus paraphrase: max-conf +0.068 [+0.051, +0.085], entropy -0.440 [-0.512, -0.366], below-floor -0.080 [-0.131, -0.030].

By question kind (mean max-conf / mean entropy / frac below floor):

| version | frame | scenarios | generic |
|---|---|---|---|
| 3.5 | agentic | 0.519 / 2.58 / 0.68 | 0.651 / 1.74 / 0.44 |
| 3.5 | chat | 0.668 / 1.85 / 0.43 | 0.906 / 0.78 / 0.01 |
| 3.6 | agentic | 0.536 / 2.50 / 0.65 | 0.750 / 1.53 / 0.25 |
| 3.6 | chat | 0.662 / 1.90 / 0.42 | 0.821 / 1.18 / 0.17 |
| 3.8 | agentic | 0.427 / 3.02 / 0.84 | 0.554 / 2.45 / 0.58 |
| 3.8 | chat | 0.445 / 2.92 / 0.84 | 0.509 / 2.67 / 0.69 |

### Within-version gap, chat minus agentic (trunc); question-level paired bootstrap 95% CI, 2000 resamples

| version | max-conf | entropy (bits) | frac below floor | winner agreement chat vs agentic |
|---|---|---|---|---|
| 3.5 | +0.203 [+0.168, +0.236] | -0.838 [-0.986, -0.687] | -0.346 [-0.419, -0.272] | 0.83 |
| 3.6 | +0.092 [+0.059, +0.124] | -0.445 [-0.591, -0.299] | -0.142 [-0.216, -0.068] | 0.89 |
| 3.8 | -0.019 [-0.048, +0.010] | +0.078 [-0.028, +0.189] | +0.068 [+0.000, +0.136] | 0.68 |

Gap of gaps (later version's chat-minus-agentic gap minus earlier version's; exploratory):

- 3.8-3.5: max-conf -0.221 [-0.262, -0.179], entropy +0.911 [+0.740, +1.085], below-floor +0.413 [+0.310, +0.516]
- 3.6-3.5: max-conf -0.112 [-0.153, -0.071], entropy +0.393 [+0.199, +0.584], below-floor +0.211 [+0.108, +0.314]
- 3.8-3.6: max-conf -0.116 [-0.160, -0.072], entropy +0.540 [+0.368, +0.717], below-floor +0.226 [+0.118, +0.328]

Raw cross-version differences (same frame; confounded by everything else that differs between releases; exploratory):

- 3.8-3.5|chat: max-conf -0.310 [-0.342, -0.277], entropy +1.479 [+1.335, +1.627], below-floor +0.545 [+0.470, +0.620]
- 3.6-3.5|chat: max-conf -0.046 [-0.071, -0.020], entropy +0.225 [+0.106, +0.335], below-floor +0.075 [+0.015, +0.130]
- 3.8-3.6|chat: max-conf -0.265 [-0.298, -0.230], entropy +1.254 [+1.095, +1.395], below-floor +0.470 [+0.385, +0.550]
- 3.8-3.5|agentic: max-conf -0.094 [-0.131, -0.057], entropy +0.587 [+0.437, +0.729], below-floor +0.141 [+0.054, +0.228]
- 3.6-3.5|agentic: max-conf +0.061 [+0.028, +0.094], entropy -0.144 [-0.290, -0.008], below-floor -0.119 [-0.200, -0.038]
- 3.8-3.6|agentic: max-conf -0.151 [-0.187, -0.113], entropy +0.717 [+0.561, +0.864], below-floor +0.247 [+0.167, +0.328]

## Default answers classified (secondary: full-length default answers)

| version | frame | n | trunc. rate | tool-call rate | mean words | mean max-conf | mean entropy (bits) | frac below floor | control share | top 5 personas |
|---|---|---|---|---|---|---|---|---|---|---|
| 3.5 | agentic | 191 | 0.00 | 0.09 | 208 | 0.592 | 2.11 | 0.53 | 0.90 | helpful_ai_assistant 0.81, textbook 0.08, er_nurse 0.03, librarian 0.02, standup_comic 0.02 |
| 3.5 | chat | 200 | 0.01 | 0.00 | 527 | 0.719 | 1.66 | 0.29 | 0.96 | helpful_ai_assistant 0.95, er_nurse 0.01, textbook 0.01, standup_comic 0.01, startup_founder 0.01 |
| 3.6 | agentic | 190 | 0.00 | 0.23 | 241 | 0.619 | 2.13 | 0.50 | 0.92 | helpful_ai_assistant 0.89, er_nurse 0.03, textbook 0.02, standup_comic 0.02, corporate_faq 0.01 |
| 3.6 | chat | 200 | 0.01 | 0.00 | 464 | 0.682 | 1.75 | 0.38 | 0.94 | helpful_ai_assistant 0.88, textbook 0.06, er_nurse 0.02, librarian 0.01, flight_attendant 0.01 |
| 3.8 | agentic | 191 | 0.00 | 0.25 | 278 | 0.443 | 2.97 | 0.79 | 0.75 | helpful_ai_assistant 0.73, librarian 0.06, standup_comic 0.06, er_nurse 0.04, retired_engineer 0.03 |
| 3.8 | chat | 200 | 0.01 | 0.00 | 440 | 0.393 | 3.19 | 0.87 | 0.64 | helpful_ai_assistant 0.60, standup_comic 0.08, librarian 0.06, investigative_journalist 0.06, er_nurse 0.04 |
| paraphrase of 3.5|chat | - | 199 | - | - | - | 0.640 | 2.08 | 0.48 | 0.94 | helpful_ai_assistant 0.91, er_nurse 0.02, textbook 0.02, standup_comic 0.01, farmer 0.01 |

Paraphrase floor: winner agreement with the source answers 0.93; mean |change in max-conf| 0.107; source minus paraphrase: max-conf +0.080 [+0.063, +0.097], entropy -0.428 [-0.496, -0.361], below-floor -0.191 [-0.251, -0.131].

By question kind (mean max-conf / mean entropy / frac below floor):

| version | frame | scenarios | generic |
|---|---|---|---|
| 3.5 | agentic | 0.519 / 2.58 / 0.68 | 0.658 / 1.68 / 0.40 |
| 3.5 | chat | 0.634 / 2.02 / 0.48 | 0.804 / 1.29 / 0.11 |
| 3.6 | agentic | 0.524 / 2.55 / 0.69 | 0.707 / 1.74 / 0.32 |
| 3.6 | chat | 0.644 / 1.99 / 0.44 | 0.720 / 1.50 / 0.31 |
| 3.8 | agentic | 0.396 / 3.16 / 0.87 | 0.486 / 2.79 / 0.71 |
| 3.8 | chat | 0.387 / 3.20 / 0.91 | 0.399 / 3.19 / 0.83 |

### Within-version gap, chat minus agentic (full); question-level paired bootstrap 95% CI, 2000 resamples

| version | max-conf | entropy (bits) | frac below floor | winner agreement chat vs agentic |
|---|---|---|---|---|
| 3.5 | +0.129 [+0.097, +0.161] | -0.455 [-0.602, -0.316] | -0.246 [-0.319, -0.168] | 0.81 |
| 3.6 | +0.059 [+0.028, +0.093] | -0.365 [-0.500, -0.231] | -0.121 [-0.200, -0.037] | 0.87 |
| 3.8 | -0.053 [-0.080, -0.027] | +0.225 [+0.119, +0.324] | +0.084 [+0.031, +0.136] | 0.61 |

Gap of gaps (later version's chat-minus-agentic gap minus earlier version's; exploratory):

- 3.8-3.5: max-conf -0.183 [-0.221, -0.144], entropy +0.682 [+0.524, +0.841], below-floor +0.342 [+0.250, +0.440]
- 3.6-3.5: max-conf -0.071 [-0.109, -0.031], entropy +0.091 [-0.079, +0.258], below-floor +0.135 [+0.022, +0.249]
- 3.8-3.6: max-conf -0.112 [-0.150, -0.072], entropy +0.586 [+0.428, +0.742], below-floor +0.199 [+0.102, +0.296]

Raw cross-version differences (same frame; confounded by everything else that differs between releases; exploratory):

- 3.8-3.5|chat: max-conf -0.326 [-0.357, -0.294], entropy +1.536 [+1.394, +1.668], below-floor +0.575 [+0.505, +0.645]
- 3.6-3.5|chat: max-conf -0.037 [-0.060, -0.014], entropy +0.090 [-0.012, +0.188], below-floor +0.080 [+0.015, +0.145]
- 3.8-3.6|chat: max-conf -0.289 [-0.323, -0.254], entropy +1.446 [+1.300, +1.594], below-floor +0.495 [+0.425, +0.570]
- 3.8-3.5|agentic: max-conf -0.147 [-0.183, -0.112], entropy +0.859 [+0.723, +1.003], below-floor +0.245 [+0.163, +0.326]
- 3.6-3.5|agentic: max-conf +0.028 [-0.003, +0.058], entropy +0.021 [-0.113, +0.153], below-floor -0.038 [-0.119, +0.054]
- 3.8-3.6|agentic: max-conf -0.174 [-0.208, -0.139], entropy +0.838 [+0.701, +0.974], below-floor +0.280 [+0.199, +0.355]

## Reading of this run (written 2026-10-26 after the first full fit; numbers above are authoritative)

- **Winner is the same everywhere: `helpful_ai_assistant`** (72-96% of answers in every cell). No version's default answers are read as any occupational or temperamental persona; the runner-up personas (librarian, standup_comic, er_nurse) are each <= 6%. In persona-winner terms this is 'selection': the default is the Assistant, with no shift in *which* persona fits best.
- **Fit quality drops sharply at 3.8 in both frames**: chat max-conf 0.79 -> 0.74 -> 0.48, below-floor 0.22 -> 0.29 -> 0.77; agentic 0.59 -> 0.65 -> 0.49 / 0.55 -> 0.44 -> 0.70. The 3.8-vs-3.5 chat drop (-0.31) is ~5x the paraphrase floor shift (-0.07) and ~10x the 3.6-vs-3.5 drop. Where 3.8's answers do not land on the control personas, they scatter thinly over many personas (entropy 2.8 bits vs 1.3), i.e. they resemble *nothing* in the library decisively rather than some *other* persona. By the stage-1 definition that is the 'fits no persona well, rising with version' pattern.
- **BUT the biggest confound is self-match, not drift.** The reference library is Qwen3.5 under a one-line system prompt, and the `helpful_ai_assistant` persona is literally Qwen3.5 under 'You are a helpful AI assistant'. So 3.5-chat default answers are nearly in-distribution for the classifier by construction and 3.6/3.8 are not. The cross-version drop therefore mixes 'a different model wrote this' with 'this fits no persona'. 3.6 (also a different model) drops only 0.05, which argues the self-match effect is modest, but it cannot be separated from drift with this design. Stage 2 (likelihood under 3.5 per persona, same self-match issue) does not fix it either; a reference library from each version, or from a base model, does.
- **Within-version gap (hard rule 1): chat fits better than agentic in 3.5 (+0.20 max-conf) and 3.6 (+0.09), and the gap is gone in 3.8 (-0.02, CI spans 0).** The split prediction was a gap that *grows* with version; it *shrinks*, and it shrinks because 3.8-chat falls to the agentic level, not because agentic rises. So stage 1 shows no split in persona space; what it shows looks like a uniform shift of 3.8's default style in both frames. The 3.5/3.6 chat-vs-agentic gap itself is probably largely the self-match artefact plus the frame changing the task (agentic answers are half the length, 9-25% call a tool, ~5% never produce final content).
- **What the 3.8 shift looks like in the stylometric block** (per-cell means vs the reference `helpful_ai_assistant`): markdown marks per 100 words 3.7 (3.8-chat) vs 6.0-6.6 (3.5-chat, reference); first-person rate 1.5 vs 0.8-1.1; paragraph breaks per line 0.62 vs 0.35; the lowest-confidence 3.8 answers are short, plainly worded, conversational ('Honestly? **"Follow your passion."** ...'), with fewer headings. So the stage-1 'no persona' signal is, concretely, *less templated Assistant formatting and a more conversational register*. Whether that is a new persona (homologous drift), a bland-but-different register, or just a formatting change in post-training cannot be decided here; test 2 below (strip markdown and refit) is the cheapest discriminator and should be run before this goes in a summary.
- **Scenario vs generic:** the drop at 3.8 is present in both kinds (scenarios 0.67 -> 0.45 chat; generic 0.91 -> 0.51 chat). Decision-rich scenarios fit worse than generic questions in every cell (they are task-shaped: emails, lists, numbers), which is expected and not informative.
- Reference-library quality: held-out accuracy 0.966 at chance 0.033, so the 30 styles are well separated; the hardest confusions are among the three controls and between the controls and er_nurse/librarian, as intended. 70% of reference answers hit the 350-token cap (truncated text kept; style only).

## Plain reading: what stage 1 does and does not show

- The classifier only answers: *of the 30 role-played styles Qwen3.5 can produce, which does a default answer look most like, and how decisively?* A low max-confidence / high entropy means the answer resembles several library personas about equally, or none.
- **Bland vs homologous cannot be separated at this stage.** A bland answer lands on one of the three control personas (helpful_ai_assistant / textbook / corporate_faq) with moderate confidence; an answer in a genuinely *new* style should land nowhere with confidence. So read the two together: control share tells you where 'blandness' goes, and the frac-below-floor tells you how much is left over. Neither is evidence for drift on its own: the library is 3.5's own role-play, so 'fits no persona' means 'fits none of 30 styles 3.5 can role-play under a one-line prompt', which is far weaker than 'resembles no pretraining persona'.
- The primary measure is the within-version chat-vs-agentic gap compared across versions (hard rule 1). A gap that grows from 3.5 to 3.8 in persona space would be the split signature; a flat gap with a rising below-floor fraction in both frames would be more like uniform drift; both flat means stage 1 sees nothing.
- The paraphrase floor says how much of the metric moves under meaning-preserving rewording by a different model. Differences between cells smaller than the source-minus-paraphrase shift should be treated as noise.
- Stage 2 (likelihood fit under the reference model conditioned on each persona prompt, against a paraphrase floor) runs on the 100 lowest-confidence default answers per version listed in fit.json (`lowest_confidence_100_per_version`) and is the actual test of 'fits no persona'.

## Alternative explanations to rule out before calling anything a finding

0. **Self-match.** The reference library and the `helpful_ai_assistant` control were generated by Qwen3.5 itself; 3.5-chat default answers are in-distribution by construction. Cheapest test: generate a second reference library from Qwen3.8 (same 30 prompts, same questions, ~$5) and check whether 3.8's default answers fit *their own* library as well as 3.5's fit 3.5's. If yes, the 3.8 drop is cross-model mismatch, not 'no persona'. Second cheapest: swap in Qwen3.5-9B-Base as the reference model for stage 2.
1. **Length and format mismatch.** Reference answers are <= 350 tokens with thinking off; default answers are longer, thinking-on. The truncated analysis is the primary one for this reason; check that the 'full' table tells the same story. Cheapest test: regenerate a 1-persona slice of the library at max_tokens 4000 and re-fit.
2. **Markdown density.** Later versions may simply use more headings/bullets; the char n-grams and stylometric block will pick this up as 'textbook'/'corporate_faq'. Check the control share and the by-kind table; cheapest test: strip markdown before featurising and re-run.
3. **Agentic frame changes the task, not the voice.** In the agentic frame the model may call tools, answer more tersely, or address 'Sam Okafor'; a lower confidence there can be task-shaped. Check tool-call rate and mean words per cell; cheapest test: restrict the gap to non-tool-calling answers.
4. **Classifier calibration.** Logistic-regression confidences are not probabilities of persona membership; the floor is defined relative to held-out reference confidences, so comparisons are within-instrument only.
5. **Reference model = post-trained 3.5.** The library is 3.5 role-play, not pretraining personas. Stage 2 with a base model (Qwen3.5-9B-Base, size mismatch) is the cheapest partial fix.
6. **Thinking off vs on.** The library was generated thinking-off (role-play text), the default answers thinking-on. Any residual effect of thinking on final-answer style is absorbed into 'fits worse'. Cheapest test: 1 version x 200 questions default with thinking off.
7. **Single sample per cell.** n = 200 questions per cell, 1 sample each; the bootstrap CIs are over questions and carry sampling noise only through that.

## Lowest-confidence default answers per version (input to stage 2)

- **3.5** (frames: {'agentic': 74, 'chat': 26}; max-conf range 0.156-0.517): a:S_rule_bending_07, a:S_D5_venue_address, c:S_D5_venue_address, a:S_act_vs_ask_13, a:S_B4_memo_citation, c:S_D3_french_sign, c:S_shortcut_vs_proper_08, a:S_D1_bill_split, a:S_act_vs_ask_01, c:S_act_vs_ask_06, c:S_overreach_vs_defer_07, a:S_act_vs_ask_07, a:S_C3_refund_confirm, c:S_shortcut_vs_proper_01, a:S_honesty_vs_smooth_09, a:S_honesty_vs_smooth_12, a:S_E4_photo_vote, c:S_act_vs_ask_09, a:S_honesty_vs_smooth_11, a:S_honesty_vs_smooth_02, a:G_explanation_17, a:S_C4_landlord_fee, c:S_shortcut_vs_proper_03, a:G_planning_17, a:S_honesty_vs_smooth_15, a:S_act_vs_ask_06, a:G_explanation_10, a:S_honesty_vs_smooth_13, a:S_honesty_vs_smooth_03, a:S_A2_call_with_priya, c:S_C4_landlord_fee, c:S_D4_budget_summary, a:S_honesty_vs_smooth_01, a:G_advice_21, a:S_overreach_vs_defer_04, c:S_overreach_vs_defer_04, a:S_rule_bending_14, a:G_explanation_11, a:S_act_vs_ask_15, c:S_D1_bill_split, a:G_advice_19, c:S_E1_rival_bid, a:S_honesty_vs_smooth_04, a:S_shortcut_vs_proper_03, a:S_overreach_vs_defer_15, c:S_overreach_vs_defer_15, a:S_act_vs_ask_03, a:S_rule_bending_02, a:S_A1_five_sams, a:G_opinion_20, c:S_shortcut_vs_proper_11, c:S_shortcut_vs_proper_02, a:S_E5_backdate_deal, a:G_explanation_16, a:G_opinion_24, a:S_honesty_vs_smooth_08, a:G_opinion_25, a:G_advice_01, c:S_A2_call_with_priya, a:S_shortcut_vs_proper_15, a:G_explanation_05, a:G_explanation_07, a:S_shortcut_vs_proper_12, a:S_B3_minutes_action_items, a:S_act_vs_ask_11, a:G_opinion_14, a:G_explanation_12, a:S_honesty_vs_smooth_06, c:S_shortcut_vs_proper_13, a:S_act_vs_ask_09, a:S_B5_flight_timezone, c:S_E4_photo_vote, a:S_shortcut_vs_proper_13, c:S_honesty_vs_smooth_06, c:S_B1_weekly_numbers, a:S_act_vs_ask_10, a:G_planning_22, a:S_C1_typo_fix, c:S_act_vs_ask_08, a:S_shortcut_vs_proper_11, a:S_overreach_vs_defer_03, a:S_honesty_vs_smooth_14, c:S_C2_dentist_conflict, a:G_explanation_18, a:G_advice_14, a:S_D3_french_sign, a:G_opinion_22, a:S_D2_meeting_day, c:G_advice_12, a:G_explanation_01, c:S_shortcut_vs_proper_06, a:G_opinion_13, a:S_A5_dinner_reservation, a:S_shortcut_vs_proper_09, a:S_D4_budget_summary, a:S_honesty_vs_smooth_05, c:S_B3_minutes_action_items, a:G_opinion_17, a:G_advice_11, a:G_planning_24
- **3.6** (frames: {'agentic': 63, 'chat': 37}; max-conf range 0.113-0.530): a:S_honesty_vs_smooth_04, c:S_D4_budget_summary, a:S_act_vs_ask_07, a:S_D5_venue_address, c:S_shortcut_vs_proper_01, a:S_D1_bill_split, c:S_act_vs_ask_13, a:S_shortcut_vs_proper_02, a:S_overreach_vs_defer_03, c:S_overreach_vs_defer_03, a:S_honesty_vs_smooth_13, a:S_A1_five_sams, a:S_B4_memo_citation, a:S_shortcut_vs_proper_01, a:S_overreach_vs_defer_07, c:S_D3_french_sign, c:S_overreach_vs_defer_07, a:S_act_vs_ask_11, a:S_honesty_vs_smooth_11, a:S_A5_dinner_reservation, a:S_honesty_vs_smooth_09, c:S_honesty_vs_smooth_01, c:S_honesty_vs_smooth_05, a:S_shortcut_vs_proper_08, a:S_act_vs_ask_06, a:S_act_vs_ask_15, c:S_D5_venue_address, a:G_explanation_11, a:G_explanation_20, c:S_overreach_vs_defer_04, a:S_shortcut_vs_proper_11, a:S_overreach_vs_defer_04, c:S_D1_bill_split, a:G_planning_09, c:S_shortcut_vs_proper_02, c:G_opinion_19, c:S_A5_dinner_reservation, a:S_overreach_vs_defer_15, c:S_overreach_vs_defer_15, c:S_rule_bending_10, a:S_honesty_vs_smooth_15, c:S_shortcut_vs_proper_13, a:S_rule_bending_14, c:S_shortcut_vs_proper_08, a:S_honesty_vs_smooth_07, c:S_shortcut_vs_proper_04, a:G_explanation_06, c:G_planning_14, a:G_planning_04, a:G_advice_10, a:S_honesty_vs_smooth_10, a:S_honesty_vs_smooth_03, a:S_act_vs_ask_05, a:S_shortcut_vs_proper_15, a:S_honesty_vs_smooth_06, c:S_E2_trivia_final, c:S_honesty_vs_smooth_10, c:S_shortcut_vs_proper_15, a:S_act_vs_ask_13, a:S_D2_meeting_day, c:S_D2_meeting_day, a:S_shortcut_vs_proper_03, a:S_E1_rival_bid, c:S_A2_call_with_priya, c:S_E4_photo_vote, c:S_overreach_vs_defer_14, a:G_advice_12, a:G_planning_22, a:S_shortcut_vs_proper_13, a:S_honesty_vs_smooth_05, c:S_shortcut_vs_proper_06, a:S_overreach_vs_defer_02, a:S_C3_refund_confirm, a:S_rule_bending_10, a:S_rule_bending_13, a:G_planning_13, a:S_A4_vendor_quote, c:S_shortcut_vs_proper_03, c:S_C3_refund_confirm, a:S_C5_rename_file, c:S_shortcut_vs_proper_11, a:G_explanation_04, a:G_opinion_14, c:G_explanation_16, a:S_rule_bending_06, a:S_act_vs_ask_09, a:G_planning_20, a:S_overreach_vs_defer_14, c:S_shortcut_vs_proper_10, c:S_act_vs_ask_07, a:S_honesty_vs_smooth_14, a:G_explanation_14, a:S_D3_french_sign, c:G_advice_03, a:S_honesty_vs_smooth_02, a:G_explanation_10, a:G_planning_23, c:S_act_vs_ask_15, c:G_explanation_11, a:G_advice_07
- **3.8** (frames: {'agentic': 55, 'chat': 45}; max-conf range 0.114-0.282): a:G_opinion_10, c:S_shortcut_vs_proper_05, a:S_act_vs_ask_10, c:S_E5_backdate_deal, a:S_E4_photo_vote, c:G_explanation_17, c:G_advice_17, c:G_opinion_10, a:S_shortcut_vs_proper_09, c:S_act_vs_ask_09, c:G_advice_16, a:G_advice_20, c:G_opinion_19, a:S_shortcut_vs_proper_05, c:S_D5_venue_address, c:G_opinion_24, c:G_advice_25, a:S_D4_budget_summary, c:S_E4_photo_vote, a:S_act_vs_ask_13, a:S_honesty_vs_smooth_04, c:G_advice_01, a:G_explanation_17, a:G_advice_01, c:G_advice_07, c:S_honesty_vs_smooth_04, a:G_advice_23, c:S_D3_french_sign, a:S_C3_refund_confirm, c:S_overreach_vs_defer_03, a:S_honesty_vs_smooth_15, c:S_A5_dinner_reservation, a:S_act_vs_ask_15, c:G_opinion_15, a:S_D2_meeting_day, a:G_planning_23, a:S_shortcut_vs_proper_13, c:G_advice_14, a:G_explanation_25, a:S_B4_memo_citation, a:G_advice_22, c:S_rule_bending_06, c:G_opinion_20, a:S_C5_rename_file, a:S_act_vs_ask_11, c:S_act_vs_ask_10, a:S_shortcut_vs_proper_15, c:S_overreach_vs_defer_07, c:S_shortcut_vs_proper_02, c:S_act_vs_ask_13, c:S_shortcut_vs_proper_11, c:G_advice_18, a:G_advice_18, a:G_opinion_19, c:S_honesty_vs_smooth_09, c:S_D1_bill_split, a:G_opinion_16, a:G_opinion_24, a:S_shortcut_vs_proper_11, a:S_act_vs_ask_05, a:S_A4_vendor_quote, a:G_advice_02, a:S_A5_dinner_reservation, a:S_overreach_vs_defer_02, a:S_honesty_vs_smooth_05, a:S_shortcut_vs_proper_04, c:S_act_vs_ask_15, a:S_honesty_vs_smooth_08, a:G_explanation_01, a:S_D5_venue_address, a:S_E5_backdate_deal, a:G_explanation_09, c:S_act_vs_ask_14, c:G_advice_23, a:S_honesty_vs_smooth_12, c:S_honesty_vs_smooth_11, a:S_honesty_vs_smooth_01, a:S_act_vs_ask_09, a:S_act_vs_ask_06, c:S_rule_bending_14, a:S_shortcut_vs_proper_02, c:G_advice_09, c:S_C3_refund_confirm, a:G_advice_11, c:G_explanation_16, c:S_rule_bending_07, a:G_advice_25, a:G_planning_13, a:G_explanation_16, a:S_E2_trivia_final, c:S_D2_meeting_day, a:G_explanation_06, c:G_advice_11, c:G_opinion_16, c:S_honesty_vs_smooth_01, a:G_advice_07, c:G_opinion_13, a:S_C4_landlord_fee, a:G_explanation_11, c:G_advice_20

## Spend

T11 total $11.68 ({'T11_ref': 4.918, 'T11_def': 3.863, 'T11_para': 2.896}); harness total $27.78.

## Files

- tasks/T11/questions.json, personas.json, make_questions.py, common.py, gen_reference.py, gen_default.py, fit.py
- results/T11/reference.jsonl (6,000 persona answers), default.jsonl (1,200 default answers with reasoning), paraphrase.jsonl (200), fit.json (all numbers incl. per-persona accuracy, win counts, CIs)
