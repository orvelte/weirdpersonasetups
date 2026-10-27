# T11 self-match test: a second reference library written by Qwen3.8

**Status: exploratory follow-up to results/T11/README.md (stage 1), testing alternative explanation 0 (self-match). Not pre-registered.**

## Question and decision rule (stated before the numbers)

Stage 1 classified every version's default answers against a 30-persona library written by Qwen3.5 and found 3.8's defaults fit it much worse (chat mean max-conf 0.48 vs 0.79 for 3.5 and 0.74 for 3.6). The library author is confounded with the version being scored: 3.5's defaults are in-distribution by construction. This test generates the same library (30 persona prompts x 200 questions, thinking OFF, max_tokens 350, 1 sample) with Qwen3.8 and asks whether 3.8's defaults fit *their own* library as well as 3.5's fit 3.5's.

**Rule.** Compare mean max-confidence of 3.8|chat under the 3.8 library with 3.5|chat under the 3.5 library (primary: default answers truncated to the library's p99 word count). If 3.8's own-library fit is within 0.05 max-conf of 3.5's own-library fit (i.e. 3.8-own minus 3.5-own >= -0.05), the stage-1 drop was self-match and drift is not supported by this instrument. If 3.8's defaults still scatter under their own library (own-library fit more than 0.05 below 3.5's, with a correspondingly higher below-floor share), drift survives this test. The agentic frame and the untruncated ('full') analysis are reported as secondary checks of the same comparison.

## Design

- Libraries: 3.5: reference.jsonl (qwen/qwen3.5-27b, deepinfra, n = 6000); 3.8: reference_3.8.jsonl (qwen/qwen3.8-27b, deepinfra, n = 170). Same 30 prompts (tasks/T11/personas.json), same 200 questions, thinking OFF, max_tokens 350, temperature 0.6 / top_p 0.95, 1 sample.
- Classifier: tfidf word 1-2 (min_df 3, 60k) + char_wb 3-5 (min_df 3, 120k) + 18 stylometric rates (standardised); multinomial logistic regression, question-grouped 5-fold CV; C from [4.0, 16.0] by held-out accuracy. Fits: (A) 3.5 library alone (re-fit of stage 1 on the reduced C grid), (B) 3.8 library alone, (C) pooled 3.5 + 3.8 with persona labels (12,000 answers; folds grouped by question, so both models' answers to a question are held out together), (D) 2-class model-identity classifier on the pooled libraries, same features.
- Default answers: results/T11/default.jsonl as in stage 1 (3 versions x 2 frames x 200 questions, thinking ON, max_tokens 4000, final content only). Per library, the primary analysis truncates each default answer to that library's p99 word count; confidence floor = p10 of that library's held-out max-conf. Each fit is its own instrument; absolute numbers are comparable only loosely across fits, which is why the rule compares each model's defaults to its *own* library and the pooled fit is reported as a single-instrument cross-check.

## Fit A: Qwen3.5 library alone (stage-1 instrument, re-fit)

- Library: n = 6000 (by writing model {'3.5': 6000}); 350-token truncation rate 0.70; mean words 215, p99 293 (default answers truncated to this in the primary analysis).
- Held-out accuracy (question-grouped 5-fold CV) **0.966** (chance 0.033); scenarios 0.936, generic 0.997; C = 16.0 from {'4.0': 0.9641666666666666, '16.0': 0.9663333333333334}.
- Held-out reference: mean max-conf 0.890, mean entropy 0.73 bits; confidence floor (p10 of held-out max-conf) = 0.651.
- Per-persona accuracy (lowest 6): helpful_ai_assistant 0.83, investigative_journalist 0.89, flight_attendant 0.94, textbook 0.94, corporate_faq 0.94, librarian 0.94.
- Top confusions (true -> predicted, count): textbook -> helpful_ai_assistant (8); helpful_ai_assistant -> textbook (6); helpful_ai_assistant -> corporate_faq (6); helpful_ai_assistant -> er_nurse (6); er_nurse -> helpful_ai_assistant (5); investigative_journalist -> librarian (5); helpful_ai_assistant -> librarian (5); investigative_journalist -> defence_lawyer (4).

### Default answers classified (primary: truncated)

| version | frame | n | mean words | mean max-conf | median max-conf | mean entropy (bits) | frac below floor | control share | winner (top 5) |
|---|---|---|---|---|---|---|---|---|---|
| 3.5 | agentic | 191 | 208 | 0.588 | 0.596 | 2.14 | 0.55 | 0.88 | helpful_ai_assistant 0.82, textbook 0.05, er_nurse 0.03, librarian 0.02, standup_comic 0.02 |
| 3.5 | chat | 200 | 527 | 0.787 | 0.883 | 1.32 | 0.22 | 0.96 | helpful_ai_assistant 0.96, er_nurse 0.01, standup_comic 0.01, startup_founder 0.01, textbook 0.01 |
| 3.6 | agentic | 190 | 241 | 0.647 | 0.687 | 2.00 | 0.44 | 0.92 | helpful_ai_assistant 0.91, er_nurse 0.03, standup_comic 0.02, textbook 0.01, corporate_faq 0.01 |
| 3.6 | chat | 200 | 464 | 0.742 | 0.825 | 1.54 | 0.29 | 0.93 | helpful_ai_assistant 0.92, er_nurse 0.02, bureaucrat 0.01, librarian 0.01, flight_attendant 0.01 |
| 3.8 | agentic | 191 | 278 | 0.493 | 0.454 | 2.73 | 0.70 | 0.77 | helpful_ai_assistant 0.76, standup_comic 0.06, librarian 0.04, er_nurse 0.03, retired_engineer 0.03 |
| 3.8 | chat | 200 | 440 | 0.477 | 0.446 | 2.79 | 0.77 | 0.76 | helpful_ai_assistant 0.72, librarian 0.05, standup_comic 0.04, er_nurse 0.03, textbook 0.03 |
| paraphrase of 3.5|chat | - | 199 | - | 0.722 | 0.808 | 1.74 | 0.30 | 0.94 | helpful_ai_assistant 0.93, er_nurse 0.02, standup_comic 0.01, textbook 0.01, farmer 0.01 |

By question kind (mean max-conf / mean entropy / frac below floor):

| version | frame | scenarios | generic |
|---|---|---|---|
| 3.5 | agentic | 0.519 / 2.58 / 0.68 | 0.651 / 1.74 / 0.44 |
| 3.5 | chat | 0.668 / 1.85 / 0.43 | 0.906 / 0.78 / 0.01 |
| 3.6 | agentic | 0.536 / 2.50 / 0.65 | 0.750 / 1.53 / 0.25 |
| 3.6 | chat | 0.662 / 1.90 / 0.42 | 0.821 / 1.18 / 0.17 |
| 3.8 | agentic | 0.427 / 3.02 / 0.84 | 0.554 / 2.45 / 0.58 |
| 3.8 | chat | 0.445 / 2.92 / 0.84 | 0.509 / 2.67 / 0.69 |

Chat minus agentic (trunc; question-level paired bootstrap 95% CI): 3.5: max-conf +0.203 [+0.169, +0.237], below-floor -0.346 [-0.419, -0.267]; 3.6: max-conf +0.092 [+0.059, +0.126], below-floor -0.142 [-0.216, -0.068]; 3.8: max-conf -0.019 [-0.050, +0.011], below-floor +0.068 [+0.000, +0.136]

Raw cross-version differences (trunc; same frame; exploratory): 3.8-3.5|chat: max-conf -0.310 [-0.342, -0.278]; 3.6-3.5|chat: max-conf -0.046 [-0.072, -0.021]; 3.8-3.6|chat: max-conf -0.265 [-0.300, -0.231]; 3.8-3.5|agentic: max-conf -0.094 [-0.132, -0.058]; 3.6-3.5|agentic: max-conf +0.061 [+0.028, +0.094]; 3.8-3.6|agentic: max-conf -0.151 [-0.188, -0.114]

Paraphrase floor (trunc): winner agreement with source 0.94; source minus paraphrase max-conf +0.068 [+0.052, +0.085], below-floor -0.080 [-0.131, -0.025].

### Default answers classified (secondary: full length)

| version | frame | n | mean words | mean max-conf | median max-conf | mean entropy (bits) | frac below floor | control share | winner (top 5) |
|---|---|---|---|---|---|---|---|---|---|
| 3.5 | agentic | 191 | 208 | 0.592 | 0.606 | 2.11 | 0.53 | 0.90 | helpful_ai_assistant 0.81, textbook 0.08, er_nurse 0.03, librarian 0.02, standup_comic 0.02 |
| 3.5 | chat | 200 | 527 | 0.719 | 0.766 | 1.66 | 0.29 | 0.96 | helpful_ai_assistant 0.95, er_nurse 0.01, textbook 0.01, standup_comic 0.01, startup_founder 0.01 |
| 3.6 | agentic | 190 | 241 | 0.619 | 0.651 | 2.13 | 0.50 | 0.92 | helpful_ai_assistant 0.89, er_nurse 0.03, textbook 0.02, standup_comic 0.02, corporate_faq 0.01 |
| 3.6 | chat | 200 | 464 | 0.682 | 0.726 | 1.75 | 0.38 | 0.94 | helpful_ai_assistant 0.88, textbook 0.06, er_nurse 0.02, librarian 0.01, flight_attendant 0.01 |
| 3.8 | agentic | 191 | 278 | 0.443 | 0.407 | 2.97 | 0.79 | 0.75 | helpful_ai_assistant 0.73, librarian 0.06, standup_comic 0.06, er_nurse 0.04, retired_engineer 0.03 |
| 3.8 | chat | 200 | 440 | 0.393 | 0.362 | 3.19 | 0.87 | 0.64 | helpful_ai_assistant 0.60, standup_comic 0.08, librarian 0.06, investigative_journalist 0.06, er_nurse 0.04 |
| paraphrase of 3.5|chat | - | 199 | - | 0.640 | 0.666 | 2.08 | 0.48 | 0.94 | helpful_ai_assistant 0.91, er_nurse 0.02, textbook 0.02, standup_comic 0.01, farmer 0.01 |

By question kind (mean max-conf / mean entropy / frac below floor):

| version | frame | scenarios | generic |
|---|---|---|---|
| 3.5 | agentic | 0.519 / 2.58 / 0.68 | 0.658 / 1.68 / 0.40 |
| 3.5 | chat | 0.634 / 2.02 / 0.48 | 0.804 / 1.29 / 0.11 |
| 3.6 | agentic | 0.524 / 2.55 / 0.69 | 0.707 / 1.74 / 0.32 |
| 3.6 | chat | 0.644 / 1.99 / 0.44 | 0.720 / 1.50 / 0.31 |
| 3.8 | agentic | 0.396 / 3.16 / 0.87 | 0.486 / 2.79 / 0.71 |
| 3.8 | chat | 0.387 / 3.20 / 0.91 | 0.399 / 3.19 / 0.83 |

Chat minus agentic (full; question-level paired bootstrap 95% CI): 3.5: max-conf +0.129 [+0.098, +0.161], below-floor -0.246 [-0.325, -0.168]; 3.6: max-conf +0.059 [+0.027, +0.091], below-floor -0.121 [-0.200, -0.042]; 3.8: max-conf -0.053 [-0.077, -0.029], below-floor +0.084 [+0.037, +0.141]

Raw cross-version differences (full; same frame; exploratory): 3.8-3.5|chat: max-conf -0.326 [-0.356, -0.294]; 3.6-3.5|chat: max-conf -0.037 [-0.061, -0.014]; 3.8-3.6|chat: max-conf -0.289 [-0.322, -0.254]; 3.8-3.5|agentic: max-conf -0.147 [-0.183, -0.111]; 3.6-3.5|agentic: max-conf +0.028 [-0.003, +0.059]; 3.8-3.6|agentic: max-conf -0.174 [-0.208, -0.141]

Paraphrase floor (full): winner agreement with source 0.93; source minus paraphrase max-conf +0.080 [+0.064, +0.097], below-floor -0.191 [-0.251, -0.131].

## Fit B: Qwen3.8 library alone

_not run_

## Fit C: pooled 3.5 + 3.8 library, persona labels

_not run_

## Spend

T11 total $12.21 ({'T11_ref': 4.918, 'T11_def': 3.863, 'T11_para': 2.896, 'T11_ref38': 0.532}); harness total $29.17.

## Files

- tasks/T11/gen_reference.py (--version 3.8), tasks/T11/fit_selfmatch.py
- results/T11/reference_3.8.jsonl (6,000 Qwen3.8 persona answers), fit_selfmatch.json (all numbers), gen_reference_3.8.log, fit_selfmatch.log
