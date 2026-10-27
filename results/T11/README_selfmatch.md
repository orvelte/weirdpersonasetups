# T11 self-match test: a second reference library written by Qwen3.8

**Status: exploratory follow-up to results/T11/README.md (stage 1), testing alternative explanation 0 (self-match). Not pre-registered.**

## Question and decision rule (stated before the numbers)

Stage 1 classified every version's default answers against a 30-persona library written by Qwen3.5 and found 3.8's defaults fit it much worse (chat mean max-conf 0.48 vs 0.79 for 3.5 and 0.74 for 3.6). The library author is confounded with the version being scored: 3.5's defaults are in-distribution by construction. This test generates the same library (30 persona prompts x 200 questions, thinking OFF, max_tokens 350, 1 sample) with Qwen3.8 and asks whether 3.8's defaults fit *their own* library as well as 3.5's fit 3.5's.

**Rule.** Compare mean max-confidence of 3.8|chat under the 3.8 library with 3.5|chat under the 3.5 library (primary: default answers truncated to the library's p99 word count). If 3.8's own-library fit is within 0.05 max-conf of 3.5's own-library fit (i.e. 3.8-own minus 3.5-own >= -0.05), the stage-1 drop was self-match and drift is not supported by this instrument. If 3.8's defaults still scatter under their own library (own-library fit more than 0.05 below 3.5's, with a correspondingly higher below-floor share), drift survives this test. The agentic frame and the untruncated ('full') analysis are reported as secondary checks of the same comparison.

## Design

- Libraries: 3.5: reference.jsonl (qwen/qwen3.5-27b, deepinfra, n = 6000); 3.8: reference_3.8.jsonl (qwen/qwen3.8-27b, deepinfra, n = 6000). Same 30 prompts (tasks/T11/personas.json), same 200 questions, thinking OFF, max_tokens 350, temperature 0.6 / top_p 0.95, 1 sample.
- Classifier: tfidf word 1-2 (min_df 3, 60k) + char_wb 3-5 (min_df 3, 120k) + 18 stylometric rates (standardised); multinomial logistic regression, question-grouped 5-fold CV; C from [4.0, 16.0] by held-out accuracy. Fits: (A) 3.5 library alone (re-fit of stage 1 on the reduced C grid), (B) 3.8 library alone, (C) pooled 3.5 + 3.8 with persona labels (12,000 answers; folds grouped by question, so both models' answers to a question are held out together), (D) 2-class model-identity classifier on the pooled libraries, same features.
- Default answers: results/T11/default.jsonl as in stage 1 (3 versions x 2 frames x 200 questions, thinking ON, max_tokens 4000, final content only). Per library, the primary analysis truncates each default answer to that library's p99 word count; confidence floor = p10 of that library's held-out max-conf. Each fit is its own instrument; absolute numbers are comparable only loosely across fits, which is why the rule compares each model's defaults to its *own* library and the pooled fit is reported as a single-instrument cross-check.

## Fit A: Qwen3.5 library alone (stage-1 instrument, re-fit)

- Library: n = 6000 (by writing model {'3.5': 6000}); 350-token truncation rate 0.70; mean words 215, p99 293 (default answers truncated to this in the primary analysis).
- Held-out accuracy (question-grouped 5-fold CV) **0.966** (chance 0.033); scenarios 0.936, generic 0.997; C = 16.0 from {'4.0': 0.9641666666666666, '16.0': 0.9663333333333334}.
- Held-out reference: mean max-conf 0.890, mean entropy 0.73 bits; confidence floor (p10 of held-out max-conf) = 0.651.
- Per-persona accuracy (lowest 6): helpful_ai_assistant 0.83, investigative_journalist 0.89, flight_attendant 0.94, textbook 0.94, corporate_faq 0.94, librarian 0.94.
- Top confusions (true -> predicted, count): textbook -> helpful_ai_assistant (8); helpful_ai_assistant -> textbook (6); helpful_ai_assistant -> corporate_faq (6); helpful_ai_assistant -> er_nurse (6); er_nurse -> helpful_ai_assistant (5); investigative_journalist -> librarian (5); helpful_ai_assistant -> librarian (5); investigative_journalist -> defence_lawyer (4).
- Cross-library transfer: the whole 3.8 library (known persona labels) scored by this classifier: accuracy 0.820, mean max-conf 0.649, mean entropy 1.98, below-floor 0.46; helpful_ai_assistant accuracy 0.92 / mean max-conf 0.742. This is the self-match penalty on answers whose persona is known.

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

- Library: n = 6000 (by writing model {'3.8': 6000}); 350-token truncation rate 0.67; mean words 205, p99 290 (default answers truncated to this in the primary analysis).
- Held-out accuracy (question-grouped 5-fold CV) **0.895** (chance 0.033); scenarios 0.810, generic 0.981; C = 16.0 from {'4.0': 0.8916666666666667, '16.0': 0.8953333333333333}.
- Held-out reference: mean max-conf 0.765, mean entropy 1.39 bits; confidence floor (p10 of held-out max-conf) = 0.337.
- Per-persona accuracy (lowest 6): investigative_journalist 0.72, helpful_ai_assistant 0.74, librarian 0.75, defence_lawyer 0.78, retired_engineer 0.80, pentester 0.81.
- Top confusions (true -> predicted, count): librarian -> helpful_ai_assistant (19); flight_attendant -> corporate_faq (10); street_hustler -> er_nurse (10); helpful_ai_assistant -> librarian (10); retired_engineer -> farmer (9); defence_lawyer -> helpful_ai_assistant (9); librarian -> flight_attendant (9); monk -> poet (8).
- Cross-library transfer: the whole 3.5 library (known persona labels) scored by this classifier: accuracy 0.966, mean max-conf 0.874, mean entropy 0.79, below-floor 0.04; helpful_ai_assistant accuracy 0.80 / mean max-conf 0.549. This is the self-match penalty on answers whose persona is known.

### Default answers classified (primary: truncated)

| version | frame | n | mean words | mean max-conf | median max-conf | mean entropy (bits) | frac below floor | control share | winner (top 5) |
|---|---|---|---|---|---|---|---|---|---|
| 3.5 | agentic | 191 | 208 | 0.587 | 0.609 | 2.21 | 0.26 | 0.81 | helpful_ai_assistant 0.79, librarian 0.04, poet 0.03, kindergarten_teacher 0.03, investigative_journalist 0.02 |
| 3.5 | chat | 200 | 527 | 0.646 | 0.721 | 1.98 | 0.16 | 0.88 | helpful_ai_assistant 0.84, librarian 0.05, textbook 0.02, corporate_faq 0.01, er_nurse 0.01 |
| 3.6 | agentic | 190 | 241 | 0.551 | 0.532 | 2.44 | 0.27 | 0.79 | helpful_ai_assistant 0.76, librarian 0.06, poet 0.02, investigative_journalist 0.02, textbook 0.02 |
| 3.6 | chat | 200 | 464 | 0.651 | 0.762 | 1.90 | 0.19 | 0.85 | helpful_ai_assistant 0.82, librarian 0.05, corporate_faq 0.01, poet 0.01, flight_attendant 0.01 |
| 3.8 | agentic | 191 | 278 | 0.440 | 0.403 | 2.93 | 0.45 | 0.64 | helpful_ai_assistant 0.62, librarian 0.09, investigative_journalist 0.04, standup_comic 0.04, therapist 0.04 |
| 3.8 | chat | 200 | 440 | 0.435 | 0.392 | 2.90 | 0.41 | 0.65 | helpful_ai_assistant 0.62, librarian 0.12, therapist 0.04, farmer 0.03, investigative_journalist 0.03 |
| paraphrase of 3.5|chat | - | 199 | - | 0.598 | 0.684 | 2.25 | 0.24 | 0.85 | helpful_ai_assistant 0.80, librarian 0.03, er_nurse 0.03, corporate_faq 0.03, textbook 0.03 |

By question kind (mean max-conf / mean entropy / frac below floor):

| version | frame | scenarios | generic |
|---|---|---|---|
| 3.5 | agentic | 0.396 / 3.12 / 0.48 | 0.761 / 1.38 / 0.05 |
| 3.5 | chat | 0.498 / 2.60 / 0.32 | 0.793 / 1.37 / 0.00 |
| 3.6 | agentic | 0.400 / 3.12 / 0.49 | 0.690 / 1.81 / 0.07 |
| 3.6 | chat | 0.508 / 2.53 / 0.37 | 0.794 / 1.28 / 0.01 |
| 3.8 | agentic | 0.326 / 3.43 / 0.66 | 0.546 / 2.47 / 0.24 |
| 3.8 | chat | 0.337 / 3.33 / 0.59 | 0.533 / 2.47 / 0.22 |

Chat minus agentic (trunc; question-level paired bootstrap 95% CI): 3.5: max-conf +0.066 [+0.031, +0.101], below-floor -0.099 [-0.168, -0.037]; 3.6: max-conf +0.100 [+0.067, +0.134], below-floor -0.084 [-0.153, -0.021]; 3.8: max-conf -0.009 [-0.038, +0.017], below-floor -0.037 [-0.105, +0.037]

Raw cross-version differences (trunc; same frame; exploratory): 3.8-3.5|chat: max-conf -0.211 [-0.244, -0.177]; 3.6-3.5|chat: max-conf +0.005 [-0.022, +0.033]; 3.8-3.6|chat: max-conf -0.216 [-0.251, -0.183]; 3.8-3.5|agentic: max-conf -0.148 [-0.183, -0.113]; 3.6-3.5|agentic: max-conf -0.037 [-0.068, -0.004]; 3.8-3.6|agentic: max-conf -0.105 [-0.139, -0.069]

Paraphrase floor (trunc): winner agreement with source 0.86; source minus paraphrase max-conf +0.050 [+0.034, +0.066], below-floor -0.085 [-0.136, -0.035].

### Default answers classified (secondary: full length)

| version | frame | n | mean words | mean max-conf | median max-conf | mean entropy (bits) | frac below floor | control share | winner (top 5) |
|---|---|---|---|---|---|---|---|---|---|
| 3.5 | agentic | 191 | 208 | 0.578 | 0.604 | 2.27 | 0.26 | 0.80 | helpful_ai_assistant 0.78, librarian 0.04, poet 0.03, kindergarten_teacher 0.03, investigative_journalist 0.02 |
| 3.5 | chat | 200 | 527 | 0.559 | 0.588 | 2.40 | 0.20 | 0.86 | helpful_ai_assistant 0.83, librarian 0.05, textbook 0.02, corporate_faq 0.01, defence_lawyer 0.01 |
| 3.6 | agentic | 190 | 241 | 0.526 | 0.497 | 2.59 | 0.28 | 0.80 | helpful_ai_assistant 0.76, librarian 0.05, poet 0.02, textbook 0.02, investigative_journalist 0.02 |
| 3.6 | chat | 200 | 464 | 0.608 | 0.633 | 2.11 | 0.20 | 0.86 | helpful_ai_assistant 0.82, librarian 0.04, corporate_faq 0.02, textbook 0.02, poet 0.01 |
| 3.8 | agentic | 191 | 278 | 0.370 | 0.304 | 3.29 | 0.56 | 0.58 | helpful_ai_assistant 0.56, librarian 0.09, investigative_journalist 0.06, poet 0.04, standup_comic 0.04 |
| 3.8 | chat | 200 | 440 | 0.317 | 0.253 | 3.50 | 0.65 | 0.47 | helpful_ai_assistant 0.43, librarian 0.12, farmer 0.07, retired_engineer 0.06, investigative_journalist 0.05 |
| paraphrase of 3.5|chat | - | 199 | - | 0.526 | 0.549 | 2.60 | 0.26 | 0.84 | helpful_ai_assistant 0.79, librarian 0.03, er_nurse 0.03, corporate_faq 0.03, textbook 0.03 |

By question kind (mean max-conf / mean entropy / frac below floor):

| version | frame | scenarios | generic |
|---|---|---|---|
| 3.5 | agentic | 0.394 / 3.16 / 0.49 | 0.746 / 1.47 / 0.05 |
| 3.5 | chat | 0.429 / 2.91 / 0.38 | 0.689 / 1.90 / 0.01 |
| 3.6 | agentic | 0.389 / 3.18 / 0.49 | 0.652 / 2.05 / 0.09 |
| 3.6 | chat | 0.471 / 2.72 / 0.39 | 0.746 / 1.51 / 0.02 |
| 3.8 | agentic | 0.287 / 3.61 / 0.74 | 0.446 / 2.98 / 0.39 |
| 3.8 | chat | 0.268 / 3.70 / 0.76 | 0.366 / 3.30 / 0.54 |

Chat minus agentic (full; question-level paired bootstrap 95% CI): 3.5: max-conf -0.011 [-0.043, +0.021], below-floor -0.073 [-0.131, -0.010]; 3.6: max-conf +0.082 [+0.051, +0.114], below-floor -0.079 [-0.147, -0.011]; 3.8: max-conf -0.055 [-0.078, -0.032], below-floor +0.089 [+0.026, +0.157]

Raw cross-version differences (full; same frame; exploratory): 3.8-3.5|chat: max-conf -0.243 [-0.271, -0.214]; 3.6-3.5|chat: max-conf +0.049 [+0.023, +0.074]; 3.8-3.6|chat: max-conf -0.292 [-0.325, -0.256]; 3.8-3.5|agentic: max-conf -0.206 [-0.241, -0.171]; 3.6-3.5|agentic: max-conf -0.053 [-0.082, -0.024]; 3.8-3.6|agentic: max-conf -0.153 [-0.184, -0.121]

Paraphrase floor (full): winner agreement with source 0.86; source minus paraphrase max-conf +0.035 [+0.019, +0.052], below-floor -0.070 [-0.126, -0.020].

## Fit C: pooled 3.5 + 3.8 library, persona labels

- Library: n = 12000 (by writing model {'3.5': 6000, '3.8': 6000}); 350-token truncation rate 0.68; mean words 210, p99 291 (default answers truncated to this in the primary analysis).
- Held-out accuracy (question-grouped 5-fold CV) **0.934** (chance 0.033); scenarios 0.878, generic 0.990; C = 16.0 from {'4.0': 0.93125, '16.0': 0.9339166666666666}.
- Held-out reference: mean max-conf 0.866, mean entropy 0.80 bits; confidence floor (p10 of held-out max-conf) = 0.509.
- Per-persona accuracy (lowest 6): helpful_ai_assistant 0.80, investigative_journalist 0.81, librarian 0.84, defence_lawyer 0.88, pentester 0.89, retired_engineer 0.89.
- Top confusions (true -> predicted, count): librarian -> helpful_ai_assistant (25); helpful_ai_assistant -> er_nurse (16); defence_lawyer -> helpful_ai_assistant (14); farmer -> er_nurse (13); street_hustler -> er_nurse (13); flight_attendant -> corporate_faq (12); helpful_ai_assistant -> corporate_faq (12); retired_engineer -> er_nurse (11).
- Held-out fit by writing model: 3.5: accuracy 0.968, mean max-conf 0.925, below-floor 0.05; 3.8: accuracy 0.900, mean max-conf 0.806, below-floor 0.15.

### Default answers classified (primary: truncated)

| version | frame | n | mean words | mean max-conf | median max-conf | mean entropy (bits) | frac below floor | control share | winner (top 5) |
|---|---|---|---|---|---|---|---|---|---|
| 3.5 | agentic | 191 | 208 | 0.656 | 0.712 | 1.86 | 0.35 | 0.84 | helpful_ai_assistant 0.83, librarian 0.03, er_nurse 0.03, kindergarten_teacher 0.02, poet 0.02 |
| 3.5 | chat | 200 | 527 | 0.771 | 0.887 | 1.32 | 0.20 | 0.94 | helpful_ai_assistant 0.93, librarian 0.02, defence_lawyer 0.01, textbook 0.01, startup_founder 0.01 |
| 3.6 | agentic | 190 | 241 | 0.664 | 0.729 | 1.87 | 0.33 | 0.86 | helpful_ai_assistant 0.85, librarian 0.03, standup_comic 0.02, er_nurse 0.02, investigative_journalist 0.01 |
| 3.6 | chat | 200 | 464 | 0.754 | 0.879 | 1.38 | 0.21 | 0.90 | helpful_ai_assistant 0.89, librarian 0.03, retired_engineer 0.01, investigative_journalist 0.01, flight_attendant 0.01 |
| 3.8 | agentic | 191 | 278 | 0.522 | 0.499 | 2.54 | 0.51 | 0.70 | helpful_ai_assistant 0.69, librarian 0.08, retired_engineer 0.04, standup_comic 0.04, defence_lawyer 0.03 |
| 3.8 | chat | 200 | 440 | 0.506 | 0.470 | 2.57 | 0.54 | 0.69 | helpful_ai_assistant 0.68, librarian 0.06, standup_comic 0.05, investigative_journalist 0.04, retired_engineer 0.04 |
| paraphrase of 3.5|chat | - | 199 | - | 0.715 | 0.847 | 1.66 | 0.26 | 0.91 | helpful_ai_assistant 0.89, er_nurse 0.02, poet 0.01, corporate_faq 0.01, flight_attendant 0.01 |

By question kind (mean max-conf / mean entropy / frac below floor):

| version | frame | scenarios | generic |
|---|---|---|---|
| 3.5 | agentic | 0.481 / 2.72 / 0.65 | 0.815 / 1.07 / 0.07 |
| 3.5 | chat | 0.619 / 2.02 / 0.40 | 0.923 / 0.62 / 0.00 |
| 3.6 | agentic | 0.496 / 2.65 / 0.62 | 0.819 / 1.14 / 0.06 |
| 3.6 | chat | 0.613 / 2.02 / 0.40 | 0.894 / 0.74 / 0.02 |
| 3.8 | agentic | 0.396 / 3.08 / 0.73 | 0.639 / 2.04 / 0.30 |
| 3.8 | chat | 0.406 / 3.02 / 0.70 | 0.606 / 2.12 / 0.37 |

Chat minus agentic (trunc; question-level paired bootstrap 95% CI): 3.5: max-conf +0.119 [+0.085, +0.154], below-floor -0.152 [-0.220, -0.089]; 3.6: max-conf +0.088 [+0.057, +0.119], below-floor -0.121 [-0.179, -0.063]; 3.8: max-conf -0.022 [-0.052, +0.007], below-floor +0.037 [-0.037, +0.110]

Raw cross-version differences (trunc; same frame; exploratory): 3.8-3.5|chat: max-conf -0.265 [-0.299, -0.231]; 3.6-3.5|chat: max-conf -0.018 [-0.041, +0.006]; 3.8-3.6|chat: max-conf -0.248 [-0.280, -0.213]; 3.8-3.5|agentic: max-conf -0.134 [-0.166, -0.099]; 3.6-3.5|agentic: max-conf +0.008 [-0.022, +0.037]; 3.8-3.6|agentic: max-conf -0.137 [-0.172, -0.099]

Paraphrase floor (trunc): winner agreement with source 0.90; source minus paraphrase max-conf +0.059 [+0.042, +0.077], below-floor -0.065 [-0.116, -0.020].

### Default answers classified (secondary: full length)

| version | frame | n | mean words | mean max-conf | median max-conf | mean entropy (bits) | frac below floor | control share | winner (top 5) |
|---|---|---|---|---|---|---|---|---|---|
| 3.5 | agentic | 191 | 208 | 0.634 | 0.673 | 1.96 | 0.36 | 0.84 | helpful_ai_assistant 0.82, librarian 0.03, er_nurse 0.03, kindergarten_teacher 0.02, textbook 0.02 |
| 3.5 | chat | 200 | 527 | 0.702 | 0.773 | 1.70 | 0.23 | 0.93 | helpful_ai_assistant 0.92, librarian 0.02, defence_lawyer 0.01, textbook 0.01, startup_founder 0.01 |
| 3.6 | agentic | 190 | 241 | 0.634 | 0.682 | 2.04 | 0.35 | 0.86 | helpful_ai_assistant 0.84, librarian 0.03, standup_comic 0.02, er_nurse 0.02, investigative_journalist 0.01 |
| 3.6 | chat | 200 | 464 | 0.700 | 0.785 | 1.64 | 0.23 | 0.90 | helpful_ai_assistant 0.88, textbook 0.03, librarian 0.02, defence_lawyer 0.01, retired_engineer 0.01 |
| 3.8 | agentic | 191 | 278 | 0.436 | 0.388 | 2.94 | 0.66 | 0.64 | helpful_ai_assistant 0.62, librarian 0.08, retired_engineer 0.06, standup_comic 0.04, er_nurse 0.04 |
| 3.8 | chat | 200 | 440 | 0.383 | 0.321 | 3.17 | 0.73 | 0.51 | helpful_ai_assistant 0.48, librarian 0.10, retired_engineer 0.09, investigative_journalist 0.07, standup_comic 0.06 |
| paraphrase of 3.5|chat | - | 199 | - | 0.640 | 0.692 | 2.04 | 0.32 | 0.90 | helpful_ai_assistant 0.88, er_nurse 0.02, poet 0.01, corporate_faq 0.01, flight_attendant 0.01 |

By question kind (mean max-conf / mean entropy / frac below floor):

| version | frame | scenarios | generic |
|---|---|---|---|
| 3.5 | agentic | 0.470 / 2.78 / 0.66 | 0.783 / 1.21 / 0.08 |
| 3.5 | chat | 0.572 / 2.27 / 0.44 | 0.832 / 1.13 / 0.03 |
| 3.6 | agentic | 0.485 / 2.72 / 0.62 | 0.772 / 1.43 / 0.10 |
| 3.6 | chat | 0.587 / 2.17 / 0.41 | 0.813 / 1.12 / 0.06 |
| 3.8 | agentic | 0.342 / 3.31 / 0.85 | 0.522 / 2.60 / 0.49 |
| 3.8 | chat | 0.327 / 3.40 / 0.83 | 0.439 / 2.94 / 0.63 |

Chat minus agentic (full; question-level paired bootstrap 95% CI): 3.5: max-conf +0.072 [+0.039, +0.105], below-floor -0.131 [-0.204, -0.063]; 3.6: max-conf +0.063 [+0.030, +0.098], below-floor -0.116 [-0.184, -0.053]; 3.8: max-conf -0.058 [-0.084, -0.029], below-floor +0.079 [+0.010, +0.147]

Raw cross-version differences (full; same frame; exploratory): 3.8-3.5|chat: max-conf -0.319 [-0.352, -0.286]; 3.6-3.5|chat: max-conf -0.002 [-0.025, +0.022]; 3.8-3.6|chat: max-conf -0.317 [-0.354, -0.281]; 3.8-3.5|agentic: max-conf -0.196 [-0.233, -0.158]; 3.6-3.5|agentic: max-conf +0.000 [-0.027, +0.028]; 3.8-3.6|agentic: max-conf -0.196 [-0.231, -0.162]

Paraphrase floor (full): winner agreement with source 0.90; source minus paraphrase max-conf +0.064 [+0.046, +0.083], below-floor -0.085 [-0.136, -0.035].

## Fit D: which model wrote this? (2-class, 3.5 vs 3.8 library, same features)

- Held-out accuracy (question-grouped 5-fold CV) **0.936** (chance 0.5); scenarios 0.911, generic 0.961; mean max-conf 0.886; C = 4.0 from {'1.0': 0.928, '4.0': 0.9358333333333333}.
- Least separable personas (accuracy): helpful_ai_assistant 0.73, textbook 0.79, poet 0.84, corporate_faq 0.86, drill_sergeant 0.90, military_strategist 0.92, defence_lawyer 0.92, flight_attendant 0.92.
- Most separable personas: standup_comic 0.99, grandmother 0.99, street_hustler 0.99, retired_engineer 0.99, sports_coach 0.99.
- Stylometric means, whole library (3.5 / 3.8): sent_len_mean 13.18 / 11.63; frac_short_sent 0.21 / 0.23; list_density 0.17 / 0.20; heading_density 0.10 / 0.12; markdown_per100w 3.06 / 3.07; first_person_per100w 1.22 / 1.60; second_person_per100w 3.73 / 3.91; hedge_per100w 0.78 / 0.68; question_rate 0.08 / 0.06; exclaim_rate 0.12 / 0.06; para_breaks 0.57 / 0.56; ttr 0.64 / 0.63.
- Stylometric means, helpful_ai_assistant persona only (3.5 / 3.8): sent_len_mean 12.52 / 11.50; list_density 0.33 / 0.36; heading_density 0.15 / 0.15; markdown_per100w 5.97 / 6.63; first_person_per100w 0.82 / 1.02; second_person_per100w 3.06 / 3.11; hedge_per100w 0.76 / 0.75; para_breaks 0.37 / 0.36.

## Verdict under the stated rule

| frame | mode | 3.5 defaults under 3.5 library (max-conf / entropy / below-floor) | 3.8 defaults under 3.8 library (max-conf / entropy / below-floor) | 3.8-own minus 3.5-own max-conf |
|---|---|---|---|---|
| chat | trunc | 0.787 / 1.32 / 0.22 | 0.435 / 2.90 / 0.41 | -0.352 |
| chat | full | 0.719 / 1.66 / 0.29 | 0.317 / 3.50 / 0.65 | -0.402 |
| agentic | trunc | 0.588 / 2.14 / 0.55 | 0.440 / 2.93 / 0.45 | -0.148 |
| agentic | full | 0.592 / 2.11 / 0.53 | 0.370 / 3.29 / 0.56 | -0.222 |

**DRIFT SURVIVES: 3.8 defaults still fit their own library worse than 3.5's fit 3.5's by more than 0.05 max-conf**

## Reading (written after the fit; numbers above are authoritative)

- **Self-match does not explain the stage-1 drop.** 3.8's defaults fit 3.8's own library (chat, trunc) at mean max-conf 0.435, against 0.787 for 3.5's defaults under 3.5's library: a gap of -0.352, seven times the 0.05 tolerance. Entropy 2.90 vs 1.32 bits. The same holds in the agentic frame and untruncated (verdict table).
- **Under 3.8's own library, 3.8's defaults fit *worse* than 3.5's and 3.6's defaults do** (0.435 vs 0.646 and 0.651; 3.8-3.5|chat -0.211 [-0.244, -0.177]). Self-match predicts the opposite ordering. So the version effect is not 'who wrote the library'; it is a property of 3.8's default answers.
- **Single-instrument check (pooled library, fit C):** with both models' role-play in the training set, 3.8|chat 0.506 vs 3.5|chat 0.771 vs 3.6|chat 0.754 (3.8-3.5|chat -0.265 [-0.299, -0.231]; 3.6-3.5|chat -0.018 [-0.041, +0.006]). The 3.6-3.5 difference vanishes once 3.6 is no longer penalised for being a different model than the library author; the 3.8 drop does not.
- **3.5 and 3.8 prose are highly separable on identical persona prompts** (fit D: held-out accuracy 0.936, chance 0.5), least so for the bland controls (helpful_ai_assistant 0.73, textbook 0.79, corporate_faq 0.86). So the self-match concern was real in principle: a 3.5-trained classifier can see 'written by 3.8', and known-persona 3.8 role-play scored by the 3.5 classifier (cross-library transfer) does lose a lot (accuracy 0.820, max-conf 0.649, i.e. ~0.24 below the 3.5 library's own held-out 0.890; comparable to the ~0.31 stage-1 drop of 3.8's defaults). What rules self-match out is not the size of that penalty but the own-library test: once the author mismatch is removed (fit B), 3.8's defaults still sit 0.35 below 3.5's own-library fit, and below 3.5's and 3.6's defaults scored under the same 3.8 library.
- **The 3.8 library is itself less separable** (held-out accuracy 0.895 vs 0.966; held-out max-conf 0.765 vs 0.890; floor 0.337 vs 0.651), and the transfer is asymmetric: 3.5's role-play scored by the 3.8 classifier gets accuracy 0.966 / max-conf 0.874 (better than the 3.8 library's own held-out), whereas 3.8's role-play under the 3.5 classifier gets 0.820 / 0.649. 3.8 role-plays each persona less stereotypically than 3.5 does. Normalising each model's own-library default fit by its own held-out reference confidence (3.5: 0.88; 3.8: 0.57) does not close the gap, so the lower ceiling of the 3.8 instrument is not the explanation either. But it is the main caveat on *what* survives: the signal could be 'less stereotyped, more variable text in general' rather than 'a new persona'. Those are not the same claim, and this instrument cannot tell them apart.
- **Winner is still `helpful_ai_assistant` everywhere** (0.62 of 3.8|chat answers under 3.8's own library; runner-up librarian 0.12), consistent with the 3.8 library's own confusions (librarian <-> helpful_ai_assistant is its top confusion). No version's defaults re-select a different persona; the 3.8 effect is dispersion (lower confidence, higher entropy), not re-selection. Within-version chat-minus-agentic gaps shrink to ~0 at 3.8 under every instrument, as in stage 1; the split prediction (gap grows with version) is not seen here either.

## Remaining caveats

1. **Below-floor shares are not comparable across instruments.** Each library defines its own floor (p10 of its held-out max-conf), and the 3.8 library's floor is much lower (0.337 vs 0.651); 3.8|chat's below-floor share of 0.41 under its own library is relative to that looser floor. The max-conf and entropy comparisons, and the pooled single-instrument fit, carry the verdict.
2. **Thinking on/off.** Both libraries are thinking-off role-play; all defaults are thinking-on. If reasoning changes 3.8's final-answer style more than it changes 3.5's (plausible: 3.8 is a reasoning-first release), that alone would produce 'fits worse' for 3.8 under every library, including its own. This is now the leading untested alternative; stage-1 test 6 (one version x 200 questions default with thinking off, ~$1) is the cheapest discriminator and should run before any summary.
3. **Length mismatch** (library <= 350 tokens, defaults up to 4000) is unchanged; truncated and full tables agree in direction, and 3.8 defaults are not the longest (440 words vs 527 for 3.5), so length does not favour 3.5.
4. **Role-play, not pretraining personas.** 'Fits no persona well' still means 'fits none of 30 role-played styles under a one-line prompt', now for 3.8's own role-play as well as 3.5's. A model whose role-play has become less stereotyped would show lower held-out accuracy (it does: 0.895 vs 0.966) and lower default fit at the same time, without any change in its default persona. Stage 2 (likelihood under each persona prompt) or a base-model reference remains the test of the stronger claim.
5. **Markdown density** (stage-1 test 2) is untested. The stylometric means (fit D) show 3.8 and 3.5 nearly identical in markdown marks per 100 words in the libraries (3.07 vs 3.06), so the libraries do not differ by formatting; the 3.8 defaults did (stage 1: 3.7 vs 6.0). Strip-markdown refit is still cheap and worth running.
6. **Provider quantisation** differs (3.5/3.6 fp8, 3.8 bf16 on deepinfra; harness/config.py). Not controllable here; unlikely to produce a style shift of this size.
7. **Single sample per cell, no pre-registration.** The 0.05 rule was fixed by the caller before this run and is applied as stated, but the whole stage-1/self-match line is exploratory. The 3.8 library has 6 of 6,000 generations regenerated after provider timeouts (resumable run, same settings).
8. **3.6 has no own library.** 3.6 is scored only under 3.5's, 3.8's and the pooled library; its fit is within 0.01 of 3.5's in every instrument except the 3.5 library (where 3.5 has the self-match advantage), which is itself evidence that self-match is worth ~0.05 max-conf, not ~0.3.

## Spend

T11 total $15.08 ({'T11_ref': 4.918, 'T11_def': 3.863, 'T11_para': 2.896, 'T11_ref38': 3.404}); harness total $33.65.

## Files

- tasks/T11/gen_reference.py (--version 3.8), tasks/T11/fit_selfmatch.py
- results/T11/reference_3.8.jsonl (6,000 Qwen3.8 persona answers), fit_selfmatch.json (all numbers), gen_reference_3.8.log, fit_selfmatch.log
