# T11 confound fixes: thinking off, minimal system prompt, semantic classifier, own-persona cosine

**Status: exploratory follow-up to results/T11/README.md (stage 1) and README_selfmatch.md. Removes three confounds from the '3.8 defaults fit persona libraries badly' result: (a) no system prompt vs persona system prompts, (b) thinking on vs off, (c) TF-IDF + stylometry classifier rewarding surface style. Not pre-registered; the verdict rule was fixed by the caller before the run and is applied as stated.**

## Verdict rule (stated first)

The drift claim survives only if, with thinking off and the minimal system prompt (condition `minsys`), under the semantic classifier, (A) 3.8's defaults still fit the libraries worse than 3.5's by more than the paraphrase floor, AND (B) 3.8's defaults are not closer to its own helpful-assistant persona than 3.5's are to its own. Otherwise the result is attributed to whichever confound closed the gap. Operationalised: (A) mean max-confidence of 3.5|minsys minus 3.8|minsys, under the pooled-library instrument AND under each model's own library (3.5 under the 3.5 library, 3.8 under the 3.8 library), must exceed that instrument's paraphrase floor (source-minus-paraphrase max-conf shift); (B) mean same-question cosine of 3.8|minsys to the 3.8 library's helpful_ai_assistant answer must be <= that of 3.5|minsys to the 3.5 library's.

## Design

- `orig`: default.jsonl chat: thinking ON, no system prompt, max_tokens 4000, full text.
- `orig_trunc`: same, truncated to 293 words (3.5 library p99).
- `nosys`: default_fix.jsonl: thinking OFF, no system prompt, max_tokens 350.
- `minsys`: default_fix.jsonl: thinking OFF, system prompt 'You are a helpful assistant.', max_tokens 350.
- Libraries: 3.5 (reference.jsonl, n = 6000), 3.8 (reference_3.8.jsonl, n = 6000); 30 persona prompts x 200 questions, thinking OFF, max_tokens 350.
- Embeddings: accounts/fireworks/models/qwen3-embedding-8b (4096-d, l2-normalised); 0 empty/failed embeddings. Classifier: multinomial logistic regression on embeddings, question-grouped 5-fold CV, C from [1.0, 10.0, 100.0]. Confidence floor per fit = p10 of held-out max-conf. Paraphrase floor: 3.5|orig answers paraphrased by claude-sonnet-5.5 (n = 199), scored under the same instrument.
- Default cells (chat frame only): 3.5|minsys n=200, 3.5|nosys n=200, 3.5|orig n=200, 3.5|orig_trunc n=200, 3.6|minsys n=200, 3.6|nosys n=200, 3.6|orig n=200, 3.6|orig_trunc n=200, 3.8|minsys n=200, 3.8|nosys n=200, 3.8|orig n=200, 3.8|orig_trunc n=200.

## 1. Regenerated defaults (thinking OFF, max_tokens 350, chat frame)

| version | condition | n | mean words | 350-token truncation rate |
|---|---|---|---|---|
| 3.5 | minsys | 200 | 193 | 0.68 |
| 3.5 | nosys | 200 | 191 | 0.68 |
| 3.5 | orig | 200 | 527 | 0.01 |
| 3.5 | orig_trunc | 200 | 228 | 0.01 |
| 3.6 | minsys | 200 | 182 | 0.67 |
| 3.6 | nosys | 200 | 187 | 0.69 |
| 3.6 | orig | 200 | 464 | 0.01 |
| 3.6 | orig_trunc | 200 | 231 | 0.01 |
| 3.8 | minsys | 200 | 188 | 0.70 |
| 3.8 | nosys | 200 | 185 | 0.69 |
| 3.8 | orig | 200 | 440 | 0.01 |
| 3.8 | orig_trunc | 200 | 232 | 0.01 |

## 2. Semantic classifier (logistic regression on embeddings)

### Held-out accuracy (question-grouped 5-fold CV)

| library | n | held-out acc | scenarios | generic | mean max-conf | floor | C | lowest-accuracy personas | top confusions |
|---|---|---|---|---|---|---|---|---|---|
| lib35 | 6000 | **0.892** | 0.820 | 0.964 | 0.783 | 0.381 | 10.0 | helpful_ai_assistant 0.69, librarian 0.73, investigative_journalist 0.79, er_nurse 0.83 | librarian->helpful_ai_assistant (22); investigative_journalist->defence_lawyer (11); sales_rep->street_hustler (11); helpful_ai_assistant->librarian (11) |
| lib38 | 6000 | **0.742** | 0.602 | 0.882 | 0.663 | 0.258 | 10.0 | librarian 0.50, investigative_journalist 0.54, helpful_ai_assistant 0.55, er_nurse 0.58 | librarian->helpful_ai_assistant (24); helpful_ai_assistant->librarian (18); er_nurse->helpful_ai_assistant (16); monk->poet (14) |
| pooled | 12000 | **0.819** | 0.712 | 0.925 | 0.775 | 0.355 | 10.0 | helpful_ai_assistant 0.58, librarian 0.64, investigative_journalist 0.67, street_hustler 0.72 | helpful_ai_assistant->librarian (39); librarian->helpful_ai_assistant (38); street_hustler->sales_rep (23); defence_lawyer->investigative_journalist (21) |

Pooled fit, held-out by writing model: 3.5: acc 0.894, max-conf 0.837, below-floor 0.06; 3.8: acc 0.743, max-conf 0.712, below-floor 0.14

Cross-library transfer, lib35 classifier scoring the whole 3.8 library (known labels): accuracy 0.781, mean max-conf 0.625, below-floor 0.27; helpful_ai_assistant accuracy 0.94 / max-conf 0.625.

Cross-library transfer, lib38 classifier scoring the whole 3.5 library (known labels): accuracy 0.953, mean max-conf 0.822, below-floor 0.05; helpful_ai_assistant accuracy 0.76 / max-conf 0.457.

For comparison, the TF-IDF + stylometry instrument (README_selfmatch.md) had held-out accuracy 0.966 (3.5 library), 0.895 (3.8 library), 0.934 (pooled).

### Default answers under the lib35 classifier

| version | condition | n | mean max-conf | median | mean entropy (bits) | frac below floor | control share | mean P(helpful_ai) | winner (top 4) |
|---|---|---|---|---|---|---|---|---|---|
| 3.5 | minsys | 200 | 0.608 | 0.650 | 2.23 | 0.18 | 0.97 | 0.60 | helpful_ai_assistant 0.96, librarian 0.01, er_nurse 0.01, textbook 0.01 |
| 3.5 | nosys | 200 | 0.604 | 0.642 | 2.25 | 0.18 | 0.94 | 0.60 | helpful_ai_assistant 0.94, er_nurse 0.02, investigative_journalist 0.01, flight_attendant 0.01 |
| 3.5 | orig | 200 | 0.593 | 0.618 | 2.34 | 0.21 | 0.94 | 0.59 | helpful_ai_assistant 0.94, librarian 0.03, er_nurse 0.01, kindergarten_teacher 0.01 |
| 3.5 | orig_trunc | 200 | 0.609 | 0.642 | 2.26 | 0.20 | 0.94 | 0.60 | helpful_ai_assistant 0.94, er_nurse 0.02, librarian 0.02, investigative_journalist 0.01 |
| 3.6 | minsys | 200 | 0.631 | 0.662 | 2.10 | 0.15 | 0.97 | 0.63 | helpful_ai_assistant 0.96, corporate_faq 0.01, er_nurse 0.01, kindergarten_teacher 0.01 |
| 3.6 | nosys | 200 | 0.642 | 0.686 | 2.05 | 0.17 | 0.98 | 0.64 | helpful_ai_assistant 0.94, textbook 0.02, corporate_faq 0.01, er_nurse 0.01 |
| 3.6 | orig | 200 | 0.578 | 0.592 | 2.37 | 0.23 | 0.95 | 0.57 | helpful_ai_assistant 0.94, librarian 0.01, er_nurse 0.01, textbook 0.01 |
| 3.6 | orig_trunc | 200 | 0.612 | 0.652 | 2.21 | 0.18 | 0.96 | 0.60 | helpful_ai_assistant 0.95, er_nurse 0.01, textbook 0.01, surfer 0.01 |
| 3.8 | minsys | 200 | 0.627 | 0.674 | 2.12 | 0.18 | 0.94 | 0.62 | helpful_ai_assistant 0.93, er_nurse 0.02, librarian 0.01, textbook 0.01 |
| 3.8 | nosys | 200 | 0.620 | 0.649 | 2.16 | 0.18 | 0.96 | 0.61 | helpful_ai_assistant 0.93, corporate_faq 0.02, textbook 0.01, er_nurse 0.01 |
| 3.8 | orig | 200 | 0.503 | 0.515 | 2.65 | 0.30 | 0.86 | 0.48 | helpful_ai_assistant 0.85, er_nurse 0.05, librarian 0.03, defence_lawyer 0.01 |
| 3.8 | orig_trunc | 200 | 0.530 | 0.505 | 2.52 | 0.28 | 0.87 | 0.51 | helpful_ai_assistant 0.85, er_nurse 0.04, librarian 0.03, defence_lawyer 0.03 |
| paraphrase of 3.5\|orig | - | 199 | 0.561 | 0.591 | 2.46 | 0.26 | 0.91 | - | helpful_ai_assistant 0.91, librarian 0.03, defence_lawyer 0.03, er_nurse 0.02 |

Paraphrase floor (lib35): winner agreement with source 0.94; source minus paraphrase: max-conf +0.034 [+0.024, +0.045], entropy -0.135 [-0.177, -0.093], below-floor -0.050 [-0.090, -0.010].

By question kind (mean max-conf / entropy / below-floor):

| version | condition | scenarios | generic |
|---|---|---|---|
| 3.5 | minsys | 0.464 / 2.92 / 0.35 | 0.751 / 1.53 / 0.02 |
| 3.5 | nosys | 0.455 / 2.93 / 0.36 | 0.754 / 1.56 / 0.01 |
| 3.5 | orig | 0.451 / 3.00 / 0.41 | 0.734 / 1.68 / 0.01 |
| 3.5 | orig_trunc | 0.457 / 2.99 / 0.38 | 0.761 / 1.53 / 0.01 |
| 3.6 | minsys | 0.468 / 2.87 / 0.30 | 0.794 / 1.34 / 0.00 |
| 3.6 | nosys | 0.492 / 2.77 / 0.33 | 0.793 / 1.32 / 0.00 |
| 3.6 | orig | 0.450 / 2.98 / 0.40 | 0.705 / 1.75 / 0.05 |
| 3.6 | orig_trunc | 0.465 / 2.93 / 0.36 | 0.758 / 1.49 / 0.01 |
| 3.8 | minsys | 0.462 / 2.90 / 0.36 | 0.793 / 1.34 / 0.01 |
| 3.8 | nosys | 0.471 / 2.88 / 0.35 | 0.770 / 1.44 / 0.02 |
| 3.8 | orig | 0.402 / 3.15 / 0.48 | 0.605 / 2.15 / 0.12 |
| 3.8 | orig_trunc | 0.404 / 3.13 / 0.49 | 0.655 / 1.92 / 0.08 |

Cross-version differences (question-level bootstrap 95% CI, 2000 resamples):

| condition | 3.8-3.5 max-conf | 3.8-3.5 entropy | 3.8-3.5 below-floor | 3.6-3.5 max-conf | 3.8-3.6 max-conf |
|---|---|---|---|---|---|
| orig | -0.089 [-0.111, -0.067] | +0.314 [+0.228, +0.400] | +0.090 [+0.030, +0.150] | -0.015 [-0.033, +0.003] | -0.075 [-0.094, -0.054] |
| orig_trunc | -0.079 [-0.102, -0.056] | +0.267 [+0.180, +0.352] | +0.090 [+0.030, +0.150] | +0.003 [-0.016, +0.021] | -0.082 [-0.104, -0.060] |
| nosys | +0.016 [-0.002, +0.032] | -0.087 [-0.156, -0.014] | +0.000 [-0.050, +0.055] | +0.038 [+0.020, +0.056] | -0.022 [-0.040, -0.002] |
| minsys | +0.020 [+0.002, +0.039] | -0.108 [-0.186, -0.032] | +0.000 [-0.050, +0.055] | +0.023 [+0.003, +0.041] | -0.003 [-0.020, +0.013] |

Within-version condition effects on max-conf (later minus earlier; winner agreement in brackets):

| version | nosys - orig | minsys - nosys | minsys - orig | orig_trunc - orig |
|---|---|---|---|---|
| 3.5 | +0.012 [-0.006, +0.031] (0.92) | +0.003 [-0.013, +0.019] (0.93) | +0.015 [-0.004, +0.034] (0.95) | +0.017 [+0.005, +0.029] (0.97) |
| 3.6 | +0.065 [+0.042, +0.087] (0.90) | -0.012 [-0.031, +0.006] (0.95) | +0.053 [+0.029, +0.078] (0.91) | +0.034 [+0.022, +0.047] (0.99) |
| 3.8 | +0.117 [+0.091, +0.142] (0.81) | +0.007 [-0.010, +0.024] (0.88) | +0.124 [+0.100, +0.148] (0.81) | +0.026 [+0.013, +0.041] (0.94) |

### Default answers under the lib38 classifier

| version | condition | n | mean max-conf | median | mean entropy (bits) | frac below floor | control share | mean P(helpful_ai) | winner (top 4) |
|---|---|---|---|---|---|---|---|---|---|
| 3.5 | minsys | 200 | 0.442 | 0.426 | 2.94 | 0.28 | 0.84 | 0.42 | helpful_ai_assistant 0.80, librarian 0.07, corporate_faq 0.04, flight_attendant 0.02 |
| 3.5 | nosys | 200 | 0.443 | 0.428 | 2.94 | 0.26 | 0.83 | 0.41 | helpful_ai_assistant 0.80, librarian 0.07, corporate_faq 0.03, flight_attendant 0.02 |
| 3.5 | orig | 200 | 0.414 | 0.374 | 3.11 | 0.32 | 0.84 | 0.40 | helpful_ai_assistant 0.83, librarian 0.06, er_nurse 0.03, kindergarten_teacher 0.01 |
| 3.5 | orig_trunc | 200 | 0.437 | 0.420 | 2.98 | 0.29 | 0.83 | 0.42 | helpful_ai_assistant 0.82, librarian 0.07, er_nurse 0.03, kindergarten_teacher 0.01 |
| 3.6 | minsys | 200 | 0.491 | 0.441 | 2.72 | 0.20 | 0.86 | 0.47 | helpful_ai_assistant 0.82, librarian 0.07, corporate_faq 0.04, kindergarten_teacher 0.01 |
| 3.6 | nosys | 200 | 0.492 | 0.466 | 2.71 | 0.23 | 0.90 | 0.47 | helpful_ai_assistant 0.84, librarian 0.04, corporate_faq 0.04, textbook 0.02 |
| 3.6 | orig | 200 | 0.425 | 0.385 | 3.01 | 0.27 | 0.83 | 0.40 | helpful_ai_assistant 0.81, librarian 0.07, investigative_journalist 0.01, startup_founder 0.01 |
| 3.6 | orig_trunc | 200 | 0.470 | 0.422 | 2.83 | 0.23 | 0.86 | 0.45 | helpful_ai_assistant 0.84, librarian 0.05, investigative_journalist 0.03, textbook 0.01 |
| 3.8 | minsys | 200 | 0.484 | 0.467 | 2.77 | 0.23 | 0.88 | 0.47 | helpful_ai_assistant 0.84, librarian 0.06, corporate_faq 0.03, er_nurse 0.02 |
| 3.8 | nosys | 200 | 0.469 | 0.454 | 2.82 | 0.23 | 0.85 | 0.44 | helpful_ai_assistant 0.81, librarian 0.07, corporate_faq 0.04, investigative_journalist 0.01 |
| 3.8 | orig | 200 | 0.345 | 0.290 | 3.32 | 0.44 | 0.65 | 0.29 | helpful_ai_assistant 0.64, librarian 0.07, er_nurse 0.06, therapist 0.05 |
| 3.8 | orig_trunc | 200 | 0.373 | 0.308 | 3.20 | 0.41 | 0.71 | 0.33 | helpful_ai_assistant 0.70, er_nurse 0.07, librarian 0.04, retired_engineer 0.04 |
| paraphrase of 3.5\|orig | - | 199 | 0.392 | 0.345 | 3.21 | 0.33 | 0.78 | - | helpful_ai_assistant 0.75, librarian 0.07, defence_lawyer 0.03, er_nurse 0.03 |

Paraphrase floor (lib38): winner agreement with source 0.84; source minus paraphrase: max-conf +0.023 [+0.012, +0.034], entropy -0.106 [-0.149, -0.066], below-floor -0.015 [-0.065, +0.035].

By question kind (mean max-conf / entropy / below-floor):

| version | condition | scenarios | generic |
|---|---|---|---|
| 3.5 | minsys | 0.280 / 3.67 / 0.54 | 0.605 / 2.21 / 0.03 |
| 3.5 | nosys | 0.285 / 3.64 / 0.50 | 0.601 / 2.23 / 0.02 |
| 3.5 | orig | 0.266 / 3.74 / 0.57 | 0.561 / 2.48 / 0.07 |
| 3.5 | orig_trunc | 0.274 / 3.70 / 0.56 | 0.599 / 2.26 / 0.02 |
| 3.6 | minsys | 0.309 / 3.54 / 0.40 | 0.674 / 1.90 / 0.01 |
| 3.6 | nosys | 0.315 / 3.54 / 0.45 | 0.669 / 1.89 / 0.00 |
| 3.6 | orig | 0.279 / 3.70 / 0.48 | 0.571 / 2.33 / 0.06 |
| 3.6 | orig_trunc | 0.291 / 3.66 / 0.46 | 0.649 / 1.99 / 0.01 |
| 3.8 | minsys | 0.291 / 3.63 / 0.46 | 0.676 / 1.90 / 0.00 |
| 3.8 | nosys | 0.296 / 3.62 / 0.46 | 0.642 / 2.01 / 0.01 |
| 3.8 | orig | 0.243 / 3.83 / 0.70 | 0.448 / 2.81 / 0.18 |
| 3.8 | orig_trunc | 0.245 / 3.82 / 0.66 | 0.502 / 2.58 / 0.16 |

Cross-version differences (question-level bootstrap 95% CI, 2000 resamples):

| condition | 3.8-3.5 max-conf | 3.8-3.5 entropy | 3.8-3.5 below-floor | 3.6-3.5 max-conf | 3.8-3.6 max-conf |
|---|---|---|---|---|---|
| orig | -0.069 [-0.090, -0.047] | +0.216 [+0.134, +0.295] | +0.120 [+0.050, +0.190] | +0.012 [-0.008, +0.028] | -0.080 [-0.102, -0.059] |
| orig_trunc | -0.063 [-0.088, -0.038] | +0.217 [+0.123, +0.314] | +0.120 [+0.050, +0.195] | +0.033 [+0.014, +0.054] | -0.096 [-0.120, -0.073] |
| nosys | +0.026 [+0.005, +0.045] | -0.118 [-0.188, -0.041] | -0.025 [-0.080, +0.030] | +0.049 [+0.030, +0.069] | -0.023 [-0.042, -0.004] |
| minsys | +0.041 [+0.020, +0.063] | -0.174 [-0.265, -0.086] | -0.055 [-0.105, -0.010] | +0.049 [+0.030, +0.069] | -0.008 [-0.026, +0.011] |

Within-version condition effects on max-conf (later minus earlier; winner agreement in brackets):

| version | nosys - orig | minsys - nosys | minsys - orig | orig_trunc - orig |
|---|---|---|---|---|
| 3.5 | +0.029 [+0.009, +0.049] (0.78) | -0.001 [-0.015, +0.014] (0.77) | +0.028 [+0.007, +0.050] (0.80) | +0.023 [+0.009, +0.037] (0.94) |
| 3.6 | +0.067 [+0.044, +0.091] (0.73) | -0.001 [-0.018, +0.017] (0.84) | +0.066 [+0.040, +0.090] (0.74) | +0.045 [+0.030, +0.059] (0.93) |
| 3.8 | +0.123 [+0.098, +0.151] (0.60) | +0.015 [-0.005, +0.035] (0.79) | +0.138 [+0.112, +0.164] (0.62) | +0.028 [+0.012, +0.044] (0.81) |

### Default answers under the pooled classifier

| version | condition | n | mean max-conf | median | mean entropy (bits) | frac below floor | control share | mean P(helpful_ai) | winner (top 4) |
|---|---|---|---|---|---|---|---|---|---|
| 3.5 | minsys | 200 | 0.584 | 0.613 | 2.28 | 0.23 | 0.93 | 0.57 | helpful_ai_assistant 0.90, librarian 0.03, corporate_faq 0.02, flight_attendant 0.01 |
| 3.5 | nosys | 200 | 0.583 | 0.634 | 2.30 | 0.24 | 0.92 | 0.56 | helpful_ai_assistant 0.89, corporate_faq 0.03, flight_attendant 0.02, librarian 0.02 |
| 3.5 | orig | 200 | 0.554 | 0.565 | 2.49 | 0.24 | 0.91 | 0.54 | helpful_ai_assistant 0.91, librarian 0.04, er_nurse 0.03, startup_founder 0.01 |
| 3.5 | orig_trunc | 200 | 0.570 | 0.585 | 2.39 | 0.26 | 0.91 | 0.56 | helpful_ai_assistant 0.91, er_nurse 0.04, librarian 0.04, investigative_journalist 0.01 |
| 3.6 | minsys | 200 | 0.617 | 0.656 | 2.13 | 0.20 | 0.94 | 0.61 | helpful_ai_assistant 0.92, librarian 0.03, corporate_faq 0.01, er_nurse 0.01 |
| 3.6 | nosys | 200 | 0.624 | 0.684 | 2.10 | 0.22 | 0.96 | 0.62 | helpful_ai_assistant 0.93, corporate_faq 0.02, librarian 0.01, textbook 0.01 |
| 3.6 | orig | 200 | 0.537 | 0.567 | 2.48 | 0.29 | 0.91 | 0.52 | helpful_ai_assistant 0.90, librarian 0.03, investigative_journalist 0.01, startup_founder 0.01 |
| 3.6 | orig_trunc | 200 | 0.582 | 0.610 | 2.29 | 0.23 | 0.93 | 0.57 | helpful_ai_assistant 0.92, librarian 0.02, investigative_journalist 0.01, er_nurse 0.01 |
| 3.8 | minsys | 200 | 0.602 | 0.637 | 2.20 | 0.21 | 0.94 | 0.59 | helpful_ai_assistant 0.93, librarian 0.02, er_nurse 0.01, corporate_faq 0.01 |
| 3.8 | nosys | 200 | 0.595 | 0.620 | 2.24 | 0.23 | 0.94 | 0.58 | helpful_ai_assistant 0.90, corporate_faq 0.03, librarian 0.03, investigative_journalist 0.01 |
| 3.8 | orig | 200 | 0.427 | 0.373 | 2.93 | 0.46 | 0.69 | 0.38 | helpful_ai_assistant 0.68, librarian 0.09, er_nurse 0.07, therapist 0.05 |
| 3.8 | orig_trunc | 200 | 0.458 | 0.385 | 2.80 | 0.42 | 0.74 | 0.42 | helpful_ai_assistant 0.72, er_nurse 0.07, librarian 0.05, therapist 0.03 |
| paraphrase of 3.5\|orig | - | 199 | 0.513 | 0.499 | 2.66 | 0.29 | 0.87 | - | helpful_ai_assistant 0.84, librarian 0.04, corporate_faq 0.03, er_nurse 0.03 |

Paraphrase floor (pooled): winner agreement with source 0.89; source minus paraphrase: max-conf +0.043 [+0.031, +0.055], entropy -0.176 [-0.229, -0.127], below-floor -0.050 [-0.090, -0.015].

By question kind (mean max-conf / entropy / below-floor):

| version | condition | scenarios | generic |
|---|---|---|---|
| 3.5 | minsys | 0.406 / 3.14 / 0.43 | 0.762 / 1.43 / 0.02 |
| 3.5 | nosys | 0.403 / 3.14 / 0.48 | 0.763 / 1.47 / 0.01 |
| 3.5 | orig | 0.389 / 3.25 / 0.47 | 0.719 / 1.74 / 0.02 |
| 3.5 | orig_trunc | 0.392 / 3.23 / 0.49 | 0.748 / 1.55 / 0.02 |
| 3.6 | minsys | 0.423 / 3.04 / 0.39 | 0.812 / 1.21 / 0.00 |
| 3.6 | nosys | 0.442 / 2.98 / 0.44 | 0.806 / 1.22 / 0.00 |
| 3.6 | orig | 0.381 / 3.24 / 0.48 | 0.694 / 1.73 / 0.10 |
| 3.6 | orig_trunc | 0.393 / 3.19 / 0.45 | 0.770 / 1.39 / 0.01 |
| 3.8 | minsys | 0.399 / 3.16 / 0.42 | 0.805 / 1.24 / 0.01 |
| 3.8 | nosys | 0.410 / 3.14 / 0.43 | 0.779 / 1.34 / 0.02 |
| 3.8 | orig | 0.319 / 3.49 / 0.70 | 0.534 / 2.38 / 0.22 |
| 3.8 | orig_trunc | 0.320 / 3.48 / 0.65 | 0.595 / 2.12 / 0.19 |

Cross-version differences (question-level bootstrap 95% CI, 2000 resamples):

| condition | 3.8-3.5 max-conf | 3.8-3.5 entropy | 3.8-3.5 below-floor | 3.6-3.5 max-conf | 3.8-3.6 max-conf |
|---|---|---|---|---|---|
| orig | -0.127 [-0.154, -0.103] | +0.439 [+0.336, +0.536] | +0.215 [+0.145, +0.285] | -0.017 [-0.038, +0.004] | -0.111 [-0.138, -0.085] |
| orig_trunc | -0.112 [-0.140, -0.084] | +0.406 [+0.294, +0.517] | +0.165 [+0.095, +0.235] | +0.012 [-0.010, +0.033] | -0.124 [-0.151, -0.097] |
| nosys | +0.011 [-0.009, +0.031] | -0.064 [-0.142, +0.015] | -0.020 [-0.070, +0.030] | +0.041 [+0.019, +0.064] | -0.030 [-0.052, -0.008] |
| minsys | +0.018 [-0.003, +0.041] | -0.084 [-0.179, +0.012] | -0.010 [-0.060, +0.040] | +0.033 [+0.012, +0.054] | -0.015 [-0.035, +0.006] |

Within-version condition effects on max-conf (later minus earlier; winner agreement in brackets):

| version | nosys - orig | minsys - nosys | minsys - orig | orig_trunc - orig |
|---|---|---|---|---|
| 3.5 | +0.029 [+0.007, +0.052] (0.84) | +0.001 [-0.017, +0.019] (0.85) | +0.030 [+0.007, +0.054] (0.85) | +0.016 [+0.002, +0.030] (0.96) |
| 3.6 | +0.087 [+0.062, +0.113] (0.84) | -0.007 [-0.027, +0.015] (0.91) | +0.080 [+0.053, +0.107] (0.83) | +0.044 [+0.029, +0.061] (0.96) |
| 3.8 | +0.168 [+0.140, +0.197] (0.65) | +0.008 [-0.012, +0.028] (0.88) | +0.176 [+0.147, +0.206] (0.66) | +0.031 [+0.012, +0.050] (0.82) |

### Nearest-centroid cosine (per library; centroid = mean embedding of a persona's 200 answers)

**lib35**: library helpful_ai_assistant answers: cos to own centroid 0.366, to nearest other centroid 0.381 (margin -0.015, own-centroid nearest for 0.36); library non-HAI answers: cos to HAI centroid 0.390; all library answers nearest to their own persona centroid: 0.78.

| version | condition | n | mean cos to helpful_ai centroid [95% CI] | mean cos to nearest other centroid | margin | frac helpful_ai nearest | nearest other (top 3) |
|---|---|---|---|---|---|---|---|
| 3.5 | minsys | 200 | 0.368 [0.354, 0.381] | 0.381 | -0.013 | 0.39 | librarian 0.24, startup_founder 0.10, drill_sergeant 0.09 |
| 3.5 | nosys | 200 | 0.366 [0.351, 0.380] | 0.381 | -0.016 | 0.39 | librarian 0.23, startup_founder 0.12, drill_sergeant 0.10 |
| 3.5 | orig | 200 | 0.403 [0.389, 0.417] | 0.412 | -0.009 | 0.46 | librarian 0.24, startup_founder 0.11, drill_sergeant 0.11 |
| 3.5 | orig_trunc | 200 | 0.371 [0.356, 0.387] | 0.388 | -0.017 | 0.39 | librarian 0.23, startup_founder 0.11, drill_sergeant 0.10 |
| 3.6 | minsys | 200 | 0.365 [0.351, 0.380] | 0.378 | -0.012 | 0.41 | librarian 0.25, drill_sergeant 0.10, startup_founder 0.09 |
| 3.6 | nosys | 200 | 0.370 [0.356, 0.383] | 0.382 | -0.012 | 0.43 | librarian 0.22, startup_founder 0.11, textbook 0.09 |
| 3.6 | orig | 200 | 0.403 [0.388, 0.417] | 0.410 | -0.007 | 0.47 | librarian 0.28, startup_founder 0.14, drill_sergeant 0.10 |
| 3.6 | orig_trunc | 200 | 0.378 [0.363, 0.393] | 0.390 | -0.012 | 0.41 | librarian 0.27, startup_founder 0.16, drill_sergeant 0.10 |
| 3.8 | minsys | 200 | 0.372 [0.358, 0.387] | 0.385 | -0.012 | 0.41 | librarian 0.21, startup_founder 0.14, drill_sergeant 0.10 |
| 3.8 | nosys | 200 | 0.376 [0.362, 0.390] | 0.390 | -0.013 | 0.39 | librarian 0.23, startup_founder 0.10, drill_sergeant 0.09 |
| 3.8 | orig | 200 | 0.422 [0.408, 0.438] | 0.433 | -0.010 | 0.39 | librarian 0.20, er_nurse 0.15, drill_sergeant 0.15 |
| 3.8 | orig_trunc | 200 | 0.408 [0.393, 0.423] | 0.422 | -0.014 | 0.35 | librarian 0.20, drill_sergeant 0.18, er_nurse 0.15 |
| paraphrase of 3.5\|orig | - | 199 | 0.404 | 0.413 | -0.009 | 0.42 | librarian 0.26, drill_sergeant 0.11, er_nurse 0.10 |

**lib38**: library helpful_ai_assistant answers: cos to own centroid 0.374, to nearest other centroid 0.394 (margin -0.019, own-centroid nearest for 0.28); library non-HAI answers: cos to HAI centroid 0.398; all library answers nearest to their own persona centroid: 0.60.

| version | condition | n | mean cos to helpful_ai centroid [95% CI] | mean cos to nearest other centroid | margin | frac helpful_ai nearest | nearest other (top 3) |
|---|---|---|---|---|---|---|---|
| 3.5 | minsys | 200 | 0.366 [0.353, 0.380] | 0.389 | -0.023 | 0.24 | librarian 0.20, startup_founder 0.10, drill_sergeant 0.10 |
| 3.5 | nosys | 200 | 0.364 [0.351, 0.379] | 0.389 | -0.024 | 0.20 | librarian 0.17, drill_sergeant 0.10, startup_founder 0.09 |
| 3.5 | orig | 200 | 0.405 [0.390, 0.419] | 0.423 | -0.018 | 0.29 | librarian 0.20, drill_sergeant 0.11, startup_founder 0.10 |
| 3.5 | orig_trunc | 200 | 0.372 [0.357, 0.387] | 0.397 | -0.025 | 0.23 | librarian 0.18, drill_sergeant 0.10, startup_founder 0.09 |
| 3.6 | minsys | 200 | 0.366 [0.352, 0.379] | 0.386 | -0.021 | 0.27 | librarian 0.19, drill_sergeant 0.10, kindergarten_teacher 0.07 |
| 3.6 | nosys | 200 | 0.370 [0.356, 0.384] | 0.391 | -0.020 | 0.28 | librarian 0.18, drill_sergeant 0.10, startup_founder 0.09 |
| 3.6 | orig | 200 | 0.405 [0.390, 0.420] | 0.421 | -0.016 | 0.32 | librarian 0.26, startup_founder 0.12, drill_sergeant 0.10 |
| 3.6 | orig_trunc | 200 | 0.379 [0.364, 0.395] | 0.399 | -0.019 | 0.27 | librarian 0.24, startup_founder 0.12, drill_sergeant 0.10 |
| 3.8 | minsys | 200 | 0.374 [0.360, 0.388] | 0.394 | -0.020 | 0.27 | librarian 0.18, startup_founder 0.11, drill_sergeant 0.10 |
| 3.8 | nosys | 200 | 0.377 [0.363, 0.391] | 0.399 | -0.022 | 0.26 | librarian 0.21, drill_sergeant 0.10, startup_founder 0.09 |
| 3.8 | orig | 200 | 0.429 [0.414, 0.444] | 0.448 | -0.018 | 0.30 | librarian 0.17, drill_sergeant 0.15, er_nurse 0.10 |
| 3.8 | orig_trunc | 200 | 0.416 [0.401, 0.431] | 0.437 | -0.021 | 0.25 | librarian 0.16, drill_sergeant 0.15, er_nurse 0.12 |
| paraphrase of 3.5\|orig | - | 199 | 0.406 | 0.425 | -0.019 | 0.28 | librarian 0.19, drill_sergeant 0.12, startup_founder 0.07 |

**pooled**: library helpful_ai_assistant answers: cos to own centroid 0.369, to nearest other centroid 0.388 (margin -0.019, own-centroid nearest for 0.33); library non-HAI answers: cos to HAI centroid 0.394; all library answers nearest to their own persona centroid: 0.68.

| version | condition | n | mean cos to helpful_ai centroid [95% CI] | mean cos to nearest other centroid | margin | frac helpful_ai nearest | nearest other (top 3) |
|---|---|---|---|---|---|---|---|
| 3.5 | minsys | 200 | 0.368 [0.354, 0.381] | 0.387 | -0.019 | 0.30 | librarian 0.23, drill_sergeant 0.10, startup_founder 0.09 |
| 3.5 | nosys | 200 | 0.366 [0.352, 0.379] | 0.386 | -0.021 | 0.30 | librarian 0.20, startup_founder 0.12, drill_sergeant 0.09 |
| 3.5 | orig | 200 | 0.405 [0.390, 0.419] | 0.419 | -0.015 | 0.35 | librarian 0.24, drill_sergeant 0.12, startup_founder 0.10 |
| 3.5 | orig_trunc | 200 | 0.372 [0.356, 0.387] | 0.394 | -0.022 | 0.32 | librarian 0.22, drill_sergeant 0.11, startup_founder 0.10 |
| 3.6 | minsys | 200 | 0.366 [0.352, 0.379] | 0.383 | -0.017 | 0.33 | librarian 0.22, drill_sergeant 0.10, startup_founder 0.07 |
| 3.6 | nosys | 200 | 0.371 [0.356, 0.385] | 0.387 | -0.017 | 0.34 | librarian 0.23, startup_founder 0.10, drill_sergeant 0.09 |
| 3.6 | orig | 200 | 0.405 [0.389, 0.420] | 0.417 | -0.012 | 0.38 | librarian 0.28, startup_founder 0.14, drill_sergeant 0.10 |
| 3.6 | orig_trunc | 200 | 0.379 [0.364, 0.394] | 0.395 | -0.016 | 0.34 | librarian 0.24, startup_founder 0.15, drill_sergeant 0.10 |
| 3.8 | minsys | 200 | 0.374 [0.360, 0.387] | 0.390 | -0.017 | 0.35 | librarian 0.20, startup_founder 0.13, drill_sergeant 0.10 |
| 3.8 | nosys | 200 | 0.378 [0.363, 0.392] | 0.396 | -0.018 | 0.32 | librarian 0.21, startup_founder 0.10, drill_sergeant 0.09 |
| 3.8 | orig | 200 | 0.427 [0.412, 0.441] | 0.441 | -0.015 | 0.32 | librarian 0.21, drill_sergeant 0.15, er_nurse 0.14 |
| 3.8 | orig_trunc | 200 | 0.413 [0.398, 0.427] | 0.431 | -0.018 | 0.28 | librarian 0.19, drill_sergeant 0.18, er_nurse 0.12 |
| paraphrase of 3.5\|orig | - | 199 | 0.406 | 0.420 | -0.015 | 0.33 | librarian 0.22, drill_sergeant 0.11, startup_founder 0.09 |

## 3. Own-persona match: cosine to the library's helpful_ai_assistant answer to the SAME question

Own library: 3.5 -> 3.5 library; 3.8 -> 3.8 library; 3.6 has no own library and is scored against both (both rows shown, neither is 'own'). Cross-question floor = mean cosine to the library's helpful_ai_assistant answers to all OTHER questions (and to one random other question). Paraphrase floor = cosine between each paraphrase and its source (3.5|orig).

| version | condition | library | own | n | same-question cos [95% CI] | cross-question mean cos [95% CI] | random other q | same - cross | scenarios / generic | same-q nearest persona = helpful_ai | nearest persona (top 3) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.5 | minsys | 3.5 | **yes** | 200 | 0.922 [0.915, 0.929] | 0.130 [0.126, 0.135] | 0.130 | +0.792 [+0.783, +0.801] | 0.918 / 0.927 | 0.86 (margin +0.060) | helpful_ai_assistant 0.86, textbook 0.02, librarian 0.02 |
| 3.5 | minsys | 3.8 | no | 200 | 0.902 [0.894, 0.909] | 0.133 [0.128, 0.139] | 0.121 | +0.769 [+0.759, +0.778] | 0.893 / 0.911 | 0.67 (margin +0.019) | helpful_ai_assistant 0.67, librarian 0.10, investigative_journalist 0.04 |
| 3.5 | nosys | 3.5 | **yes** | 200 | 0.916 [0.907, 0.923] | 0.130 [0.125, 0.135] | 0.123 | +0.786 [+0.776, +0.797] | 0.903 / 0.928 | 0.84 (margin +0.053) | helpful_ai_assistant 0.84, librarian 0.04, flight_attendant 0.03 |
| 3.5 | nosys | 3.8 | no | 200 | 0.896 [0.885, 0.905] | 0.133 [0.127, 0.138] | 0.134 | +0.763 [+0.749, +0.775] | 0.877 / 0.914 | 0.65 (margin +0.012) | helpful_ai_assistant 0.65, librarian 0.10, flight_attendant 0.04 |
| 3.5 | orig | 3.5 | **yes** | 200 | 0.889 [0.882, 0.897] | 0.144 [0.138, 0.149] | 0.145 | +0.746 [+0.735, +0.756] | 0.889 / 0.889 | 0.81 (margin +0.036) | helpful_ai_assistant 0.81, er_nurse 0.04, librarian 0.03 |
| 3.5 | orig | 3.8 | no | 200 | 0.883 [0.874, 0.891] | 0.148 [0.143, 0.153] | 0.152 | +0.735 [+0.724, +0.746] | 0.881 / 0.885 | 0.56 (margin +0.006) | helpful_ai_assistant 0.56, librarian 0.09, investigative_journalist 0.04 |
| 3.5 | orig_trunc | 3.5 | **yes** | 200 | 0.896 [0.888, 0.903] | 0.132 [0.126, 0.137] | 0.128 | +0.764 [+0.754, +0.775] | 0.888 / 0.903 | 0.82 (margin +0.042) | helpful_ai_assistant 0.82, librarian 0.04, er_nurse 0.03 |
| 3.5 | orig_trunc | 3.8 | no | 200 | 0.889 [0.880, 0.897] | 0.135 [0.130, 0.141] | 0.130 | +0.753 [+0.742, +0.764] | 0.880 / 0.897 | 0.58 (margin +0.011) | helpful_ai_assistant 0.58, librarian 0.10, er_nurse 0.04 |
| 3.6 | minsys | 3.5 | **yes** | 200 | 0.909 [0.901, 0.917] | 0.130 [0.125, 0.135] | 0.130 | +0.779 [+0.769, +0.788] | 0.897 / 0.920 | 0.80 (margin +0.051) | helpful_ai_assistant 0.80, librarian 0.05, er_nurse 0.03 |
| 3.6 | minsys | 3.8 | **yes** | 200 | 0.906 [0.897, 0.915] | 0.133 [0.128, 0.138] | 0.128 | +0.773 [+0.762, +0.784] | 0.894 / 0.919 | 0.69 (margin +0.024) | helpful_ai_assistant 0.69, librarian 0.09, flight_attendant 0.04 |
| 3.6 | nosys | 3.5 | **yes** | 200 | 0.910 [0.902, 0.917] | 0.131 [0.126, 0.136] | 0.141 | +0.778 [+0.768, +0.788] | 0.902 / 0.917 | 0.82 (margin +0.050) | helpful_ai_assistant 0.82, librarian 0.07, flight_attendant 0.03 |
| 3.6 | nosys | 3.8 | **yes** | 200 | 0.905 [0.896, 0.914] | 0.135 [0.130, 0.140] | 0.142 | +0.771 [+0.759, +0.782] | 0.893 / 0.918 | 0.67 (margin +0.017) | helpful_ai_assistant 0.67, librarian 0.12, flight_attendant 0.03 |
| 3.6 | orig | 3.5 | **yes** | 200 | 0.869 [0.862, 0.877] | 0.144 [0.138, 0.149] | 0.153 | +0.726 [+0.716, +0.736] | 0.876 / 0.863 | 0.72 (margin +0.027) | helpful_ai_assistant 0.72, librarian 0.07, textbook 0.05 |
| 3.6 | orig | 3.8 | **yes** | 200 | 0.868 [0.859, 0.878] | 0.148 [0.142, 0.154] | 0.145 | +0.720 [+0.708, +0.731] | 0.868 / 0.868 | 0.55 (margin -0.001) | helpful_ai_assistant 0.55, librarian 0.17, defence_lawyer 0.04 |
| 3.6 | orig_trunc | 3.5 | **yes** | 200 | 0.877 [0.868, 0.885] | 0.134 [0.129, 0.140] | 0.139 | +0.743 [+0.732, +0.753] | 0.875 / 0.879 | 0.76 (margin +0.030) | helpful_ai_assistant 0.76, librarian 0.06, textbook 0.04 |
| 3.6 | orig_trunc | 3.8 | **yes** | 200 | 0.877 [0.868, 0.886] | 0.138 [0.132, 0.144] | 0.137 | +0.739 [+0.728, +0.750] | 0.870 / 0.885 | 0.60 (margin +0.005) | helpful_ai_assistant 0.60, librarian 0.13, retired_engineer 0.04 |
| 3.8 | minsys | 3.5 | no | 200 | 0.898 [0.890, 0.905] | 0.132 [0.127, 0.137] | 0.135 | +0.766 [+0.755, +0.775] | 0.888 / 0.908 | 0.79 (margin +0.043) | helpful_ai_assistant 0.79, librarian 0.07, textbook 0.03 |
| 3.8 | minsys | 3.8 | **yes** | 200 | 0.914 [0.903, 0.922] | 0.136 [0.131, 0.141] | 0.132 | +0.777 [+0.765, +0.788] | 0.900 / 0.928 | 0.71 (margin +0.025) | helpful_ai_assistant 0.71, librarian 0.10, investigative_journalist 0.03 |
| 3.8 | nosys | 3.5 | no | 200 | 0.901 [0.892, 0.908] | 0.134 [0.128, 0.139] | 0.137 | +0.767 [+0.757, +0.776] | 0.892 / 0.909 | 0.77 (margin +0.044) | helpful_ai_assistant 0.77, textbook 0.06, librarian 0.04 |
| 3.8 | nosys | 3.8 | **yes** | 200 | 0.909 [0.899, 0.918] | 0.137 [0.132, 0.143] | 0.144 | +0.772 [+0.760, +0.782] | 0.896 / 0.923 | 0.69 (margin +0.023) | helpful_ai_assistant 0.69, librarian 0.10, investigative_journalist 0.04 |
| 3.8 | orig | 3.5 | no | 200 | 0.839 [0.829, 0.849] | 0.151 [0.145, 0.156] | 0.147 | +0.688 [+0.676, +0.700] | 0.850 / 0.828 | 0.56 (margin +0.007) | helpful_ai_assistant 0.56, librarian 0.10, er_nurse 0.07 |
| 3.8 | orig | 3.8 | **yes** | 200 | 0.840 [0.829, 0.851] | 0.157 [0.152, 0.163] | 0.153 | +0.683 [+0.669, +0.695] | 0.843 / 0.837 | 0.39 (margin -0.020) | helpful_ai_assistant 0.39, librarian 0.11, farmer 0.05 |
| 3.8 | orig_trunc | 3.5 | no | 200 | 0.830 [0.819, 0.842] | 0.146 [0.141, 0.151] | 0.139 | +0.684 [+0.671, +0.697] | 0.842 / 0.819 | 0.52 (margin +0.004) | helpful_ai_assistant 0.52, er_nurse 0.07, textbook 0.07 |
| 3.8 | orig_trunc | 3.8 | **yes** | 200 | 0.833 [0.821, 0.844] | 0.152 [0.147, 0.158] | 0.147 | +0.681 [+0.667, +0.694] | 0.836 / 0.829 | 0.36 (margin -0.021) | helpful_ai_assistant 0.36, librarian 0.10, er_nurse 0.06 |

Floors and references (mean cos [95% CI]):

| reference | mean cos | 95% CI | n |
|---|---|---|---|
| paraphrase_vs_source | 0.968 | [0.965, 0.971] | 199 |
| paraphrase_of_35orig_vs_lib35_hai_same_q | 0.873 | [0.865, 0.880] | 199 |
| lib3.5_hai_vs_textbook_same_q | 0.784 | [0.770, 0.798] | 200 |
| lib3.5_hai_vs_corporate_faq_same_q | 0.650 | [0.625, 0.673] | 200 |
| lib3.5_hai_vs_librarian_same_q | 0.818 | [0.806, 0.830] | 200 |
| lib3.5_hai_vs_mean_other_persona_same_q | 0.692 | [0.682, 0.702] | 200 |
| lib3.5_hai_vs_hai_other_question | 0.129 | [0.124, 0.135] | 200 |
| lib3.8_hai_vs_textbook_same_q | 0.767 | [0.751, 0.781] | 200 |
| lib3.8_hai_vs_corporate_faq_same_q | 0.685 | [0.663, 0.707] | 200 |
| lib3.8_hai_vs_librarian_same_q | 0.842 | [0.828, 0.854] | 200 |
| lib3.8_hai_vs_mean_other_persona_same_q | 0.735 | [0.726, 0.744] | 200 |
| lib3.8_hai_vs_hai_other_question | 0.136 | [0.131, 0.141] | 200 |
| lib35_hai_vs_lib38_hai_same_q | 0.899 | [0.891, 0.907] | 200 |

Question-paired differences (bootstrap 95% CI):

| comparison | diff | 95% CI | n |
|---|---|---|---|
| own: 3.8|orig minus 3.5|orig | -0.049 | [-0.060, -0.039] | 200 |
| lib3.5: 3.8|orig minus 3.5|orig | -0.050 | [-0.060, -0.041] | 200 |
| lib3.5: 3.6|orig minus 3.5|orig | -0.020 | [-0.026, -0.013] | 200 |
| lib3.8: 3.8|orig minus 3.5|orig | -0.043 | [-0.053, -0.034] | 200 |
| lib3.8: 3.6|orig minus 3.5|orig | -0.015 | [-0.021, -0.008] | 200 |
| own: 3.8|orig_trunc minus 3.5|orig_trunc | -0.063 | [-0.074, -0.051] | 200 |
| lib3.5: 3.8|orig_trunc minus 3.5|orig_trunc | -0.065 | [-0.076, -0.055] | 200 |
| lib3.5: 3.6|orig_trunc minus 3.5|orig_trunc | -0.019 | [-0.025, -0.012] | 200 |
| lib3.8: 3.8|orig_trunc minus 3.5|orig_trunc | -0.056 | [-0.067, -0.045] | 200 |
| lib3.8: 3.6|orig_trunc minus 3.5|orig_trunc | -0.011 | [-0.018, -0.004] | 200 |
| own: 3.8|nosys minus 3.5|nosys | -0.007 | [-0.016, +0.001] | 200 |
| lib3.5: 3.8|nosys minus 3.5|nosys | -0.015 | [-0.024, -0.008] | 200 |
| lib3.5: 3.6|nosys minus 3.5|nosys | -0.006 | [-0.014, +0.001] | 200 |
| lib3.8: 3.8|nosys minus 3.5|nosys | +0.013 | [+0.005, +0.022] | 200 |
| lib3.8: 3.6|nosys minus 3.5|nosys | +0.010 | [+0.002, +0.018] | 200 |
| own: 3.8|minsys minus 3.5|minsys | -0.009 | [-0.018, -0.001] | 200 |
| lib3.5: 3.8|minsys minus 3.5|minsys | -0.025 | [-0.031, -0.018] | 200 |
| lib3.5: 3.6|minsys minus 3.5|minsys | -0.014 | [-0.021, -0.007] | 200 |
| lib3.8: 3.8|minsys minus 3.5|minsys | +0.012 | [+0.004, +0.020] | 200 |
| lib3.8: 3.6|minsys minus 3.5|minsys | +0.004 | [-0.004, +0.012] | 200 |
| 3.5 lib3.5: nosys minus orig | +0.027 | [+0.019, +0.035] | 200 |
| 3.5 lib3.5: minsys minus nosys | +0.007 | [-0.001, +0.013] | 200 |
| 3.5 lib3.5: minsys minus orig | +0.033 | [+0.026, +0.040] | 200 |
| 3.6 lib3.5: nosys minus orig | +0.040 | [+0.032, +0.047] | 200 |
| 3.6 lib3.5: minsys minus nosys | -0.001 | [-0.008, +0.006] | 200 |
| 3.6 lib3.5: minsys minus orig | +0.039 | [+0.031, +0.047] | 200 |
| 3.6 lib3.8: nosys minus orig | +0.037 | [+0.029, +0.045] | 200 |
| 3.6 lib3.8: minsys minus nosys | +0.001 | [-0.006, +0.008] | 200 |
| 3.6 lib3.8: minsys minus orig | +0.038 | [+0.029, +0.047] | 200 |
| 3.8 lib3.8: nosys minus orig | +0.069 | [+0.057, +0.081] | 200 |
| 3.8 lib3.8: minsys minus nosys | +0.004 | [-0.005, +0.013] | 200 |
| 3.8 lib3.8: minsys minus orig | +0.073 | [+0.063, +0.084] | 200 |

## 2b. Old instrument (TF-IDF + stylometry, README_selfmatch.md classifiers) re-scored on the new conditions

Refit of the stage-1 / self-match classifiers (same features, C = 16, no CV; floors and word caps taken from fit_selfmatch.json) applied to the thinking-off conditions. `orig_trunc` here should reproduce the README_selfmatch 'chat, trunc' rows.

**lib35** (floor 0.651):

| version | condition | n | mean max-conf | mean entropy (bits) | frac below floor | control share | winner (top 4) |
|---|---|---|---|---|---|---|---|
| 3.5 | minsys | 200 | 0.725 | 1.63 | 0.28 | 0.96 | helpful_ai_assistant 0.95, standup_comic 0.01, textbook 0.01, er_nurse 0.01 |
| 3.5 | nosys | 200 | 0.718 | 1.66 | 0.33 | 0.94 | helpful_ai_assistant 0.93, textbook 0.01, flight_attendant 0.01, librarian 0.01 |
| 3.5 | orig_trunc | 200 | 0.787 | 1.32 | 0.22 | 0.96 | helpful_ai_assistant 0.96, er_nurse 0.01, standup_comic 0.01, startup_founder 0.01 |
| 3.6 | minsys | 200 | 0.743 | 1.50 | 0.29 | 0.94 | helpful_ai_assistant 0.93, textbook 0.01, flight_attendant 0.01, librarian 0.01 |
| 3.6 | nosys | 200 | 0.751 | 1.44 | 0.26 | 0.95 | helpful_ai_assistant 0.92, textbook 0.04, flight_attendant 0.01, er_nurse 0.01 |
| 3.6 | orig_trunc | 200 | 0.742 | 1.54 | 0.29 | 0.93 | helpful_ai_assistant 0.92, er_nurse 0.02, bureaucrat 0.01, librarian 0.01 |
| 3.8 | minsys | 200 | 0.729 | 1.55 | 0.32 | 0.95 | helpful_ai_assistant 0.94, er_nurse 0.02, textbook 0.01, librarian 0.01 |
| 3.8 | nosys | 200 | 0.753 | 1.44 | 0.27 | 0.96 | helpful_ai_assistant 0.94, textbook 0.01, conspiracy_blogger 0.01, startup_founder 0.01 |
| 3.8 | orig_trunc | 200 | 0.477 | 2.79 | 0.77 | 0.76 | helpful_ai_assistant 0.72, librarian 0.05, standup_comic 0.04, er_nurse 0.03 |

Paraphrase floor: source minus paraphrase max-conf +0.068 [+0.051, +0.085]. Cross-version 3.8-3.5 max-conf: orig_trunc -0.310 [-0.343, -0.279]; nosys +0.035 [+0.011, +0.060]; minsys +0.004 [-0.022, +0.031].

**lib38** (floor 0.337):

| version | condition | n | mean max-conf | mean entropy (bits) | frac below floor | control share | winner (top 4) |
|---|---|---|---|---|---|---|---|
| 3.5 | minsys | 200 | 0.539 | 2.45 | 0.23 | 0.83 | helpful_ai_assistant 0.79, librarian 0.06, flight_attendant 0.04, investigative_journalist 0.03 |
| 3.5 | nosys | 200 | 0.531 | 2.50 | 0.22 | 0.81 | helpful_ai_assistant 0.77, librarian 0.05, flight_attendant 0.04, investigative_journalist 0.03 |
| 3.5 | orig_trunc | 200 | 0.646 | 1.98 | 0.16 | 0.88 | helpful_ai_assistant 0.84, librarian 0.05, textbook 0.02, corporate_faq 0.01 |
| 3.6 | minsys | 200 | 0.599 | 2.17 | 0.18 | 0.90 | helpful_ai_assistant 0.85, librarian 0.04, corporate_faq 0.02, textbook 0.02 |
| 3.6 | nosys | 200 | 0.621 | 2.07 | 0.16 | 0.92 | helpful_ai_assistant 0.87, corporate_faq 0.03, librarian 0.02, poet 0.01 |
| 3.6 | orig_trunc | 200 | 0.651 | 1.90 | 0.19 | 0.85 | helpful_ai_assistant 0.82, librarian 0.05, corporate_faq 0.01, poet 0.01 |
| 3.8 | minsys | 200 | 0.623 | 2.08 | 0.20 | 0.89 | helpful_ai_assistant 0.88, librarian 0.03, flight_attendant 0.02, er_nurse 0.01 |
| 3.8 | nosys | 200 | 0.626 | 2.05 | 0.16 | 0.91 | helpful_ai_assistant 0.86, librarian 0.03, corporate_faq 0.03, textbook 0.02 |
| 3.8 | orig_trunc | 200 | 0.435 | 2.90 | 0.41 | 0.65 | helpful_ai_assistant 0.62, librarian 0.12, therapist 0.04, farmer 0.03 |

Paraphrase floor: source minus paraphrase max-conf +0.050 [+0.035, +0.066]. Cross-version 3.8-3.5 max-conf: orig_trunc -0.211 [-0.244, -0.177]; nosys +0.095 [+0.067, +0.123]; minsys +0.083 [+0.055, +0.110].

**pooled** (floor 0.509):

| version | condition | n | mean max-conf | mean entropy (bits) | frac below floor | control share | winner (top 4) |
|---|---|---|---|---|---|---|---|
| 3.5 | minsys | 200 | 0.719 | 1.62 | 0.20 | 0.91 | helpful_ai_assistant 0.91, librarian 0.03, investigative_journalist 0.02, flight_attendant 0.02 |
| 3.5 | nosys | 200 | 0.712 | 1.64 | 0.20 | 0.91 | helpful_ai_assistant 0.90, flight_attendant 0.03, librarian 0.02, standup_comic 0.01 |
| 3.5 | orig_trunc | 200 | 0.771 | 1.32 | 0.20 | 0.94 | helpful_ai_assistant 0.93, librarian 0.02, defence_lawyer 0.01, textbook 0.01 |
| 3.6 | minsys | 200 | 0.722 | 1.58 | 0.21 | 0.92 | helpful_ai_assistant 0.91, librarian 0.04, corporate_faq 0.01, er_nurse 0.01 |
| 3.6 | nosys | 200 | 0.738 | 1.49 | 0.20 | 0.94 | helpful_ai_assistant 0.93, librarian 0.03, flight_attendant 0.01, er_nurse 0.01 |
| 3.6 | orig_trunc | 200 | 0.754 | 1.38 | 0.21 | 0.90 | helpful_ai_assistant 0.89, librarian 0.03, retired_engineer 0.01, investigative_journalist 0.01 |
| 3.8 | minsys | 200 | 0.732 | 1.55 | 0.21 | 0.92 | helpful_ai_assistant 0.92, librarian 0.03, flight_attendant 0.01, er_nurse 0.01 |
| 3.8 | nosys | 200 | 0.740 | 1.50 | 0.18 | 0.93 | helpful_ai_assistant 0.91, flight_attendant 0.01, investigative_journalist 0.01, corporate_faq 0.01 |
| 3.8 | orig_trunc | 200 | 0.506 | 2.57 | 0.54 | 0.69 | helpful_ai_assistant 0.68, librarian 0.06, standup_comic 0.05, investigative_journalist 0.04 |

Paraphrase floor: source minus paraphrase max-conf +0.059 [+0.041, +0.077]. Cross-version 3.8-3.5 max-conf: orig_trunc -0.265 [-0.300, -0.232]; nosys +0.028 [+0.002, +0.055]; minsys +0.013 [-0.013, +0.038].

## 4. Verdict

**Rule (restated):** drift survives only if, with thinking off and the minimal system prompt (minsys), under the semantic classifier, 3.8's defaults fit the libraries worse than 3.5's by more than the paraphrase floor (A) AND 3.8's defaults are not closer to their own helpful-assistant persona than 3.5's are to theirs (B).

| test | instrument | 3.5\|minsys | 3.8\|minsys | gap (3.5 - 3.8) | paraphrase floor | holds |
|---|---|---|---|---|---|---|
| A: 3.8 fits worse than 3.5 by more than the paraphrase floor | pooled library (single instrument; primary) | 0.584 | 0.602 | -0.018 [-0.041, +0.003] | +0.043 | no |
| A: 3.8 fits worse than 3.5 by more than the paraphrase floor | own library (3.5 under lib35 vs 3.8 under lib38; two instruments) | 0.608 | 0.484 | +0.124 | +0.034 | yes |
| A: 3.8 fits worse than 3.5 by more than the paraphrase floor | own library, each divided by its instrument's held-out mean max-conf (informational) | 0.776 | 0.729 | +0.047 | +0.034 | yes |
| A: 3.8 fits worse than 3.5 by more than the paraphrase floor | within lib35 (3.5\|minsys vs 3.8\|minsys; informational) | 0.608 | 0.627 | -0.020 [-0.039, -0.002] | +0.034 | no |
| A: 3.8 fits worse than 3.5 by more than the paraphrase floor | within lib38 (3.5\|minsys vs 3.8\|minsys; informational) | 0.442 | 0.484 | -0.041 [-0.063, -0.020] | +0.023 | no |
| B: 3.8 not closer to its own helpful-assistant persona than 3.5 to its own | own-library same-question cosine | 0.922 | 0.914 | 3.8 - 3.5: -0.009 [-0.018, -0.001] | - | yes |

A holds: **False** (pooled False; own-library True, but see the normalised and within-instrument rows: the raw own-library row compares two instruments whose held-out ceilings differ, 0.783 vs 0.663, and inside either single instrument 3.8\|minsys fits at least as well as 3.5\|minsys). B holds: **True**.

**DRIFT DOES NOT SURVIVE under the stated rule.**

3.8-3.5 max-conf gap under the pooled semantic instrument, by condition: orig -0.127; orig_trunc -0.112; nosys +0.011; minsys +0.018. Own-persona cosine by condition (3.5 / 3.8): orig 0.889 / 0.840; orig_trunc 0.896 / 0.833; nosys 0.916 / 0.909; minsys 0.922 / 0.914.

### Which confound explained it (reading written after the fit; the tables above are authoritative)

- **(b) thinking ON is the confound, and it explains the whole 3.8 drop.** Switching the defaults to thinking OFF at the library's length cap (`nosys`, no other change) moves the pooled 3.8-3.5 max-conf gap from -0.127 (semantic, full text; -0.112 truncated) to +0.011, and under the old TF-IDF instrument from -0.265 to +0.028. Within 3.8, nosys minus orig is +0.168 [+0.140, +0.197] against +0.029 for 3.5 and +0.087 for 3.6: reasoning mode changes 3.8's final-answer style far more than it changes 3.5's. With thinking off, 3.8's defaults fit every library (3.5's, its own, pooled) at least as well as 3.5's do, under both instruments.
- **(a) the system prompt changes nothing.** `minsys` minus `nosys` max-conf: 3.5 +0.001, 3.6 -0.007, 3.8 +0.008 (all CIs span 0); the 3.8-3.5 gap goes from +0.011 to +0.018. The 'tool agent with no system prompt' explanation is not supported: a bare user message and 'You are a helpful assistant.' produce the same persona fit.
- **(c) the classifier was not the explanation on its own.** On the original thinking-on answers the semantic instrument still shows a 3.8-3.5 gap of -0.112 [-0.140, -0.084] (TF-IDF: -0.265), about 2.5x the semantic paraphrase floor (+0.043). The surface-style instrument exaggerated the drop roughly 2x but did not invent it; what it was seeing was thinking-mode output.
- **Own-persona cosine says the same.** Same-question cosine to the own library's helpful_ai_assistant answer: 3.5|minsys 0.922, 3.8|minsys 0.914 (3.8 - 3.5 = -0.009 [-0.018, -0.001]), versus 0.889 / 0.840 for the thinking-on originals (3.8 - 3.5 = -0.049). For scale: a paraphrase sits 0.968 from its source, 3.5's and 3.8's helpful-assistant answers to the same question sit 0.899 from each other, the library's helpful-assistant vs textbook same-question cosine is 0.784, and the cross-question floor is 0.130. The residual 3.8-3.5 own-persona difference at minsys (-0.009) is smaller than the self-match effect itself (3.8|minsys is +0.016 closer to its own library's answer than to 3.5's; 3.5|minsys is +0.020 closer to its own), so B holds only in the formal sense (3.8 is not closer than 3.5 to its own persona), with a difference an order of magnitude below the persona-level distances.
- **Winner is helpful_ai_assistant everywhere** (0.93 of 3.8|minsys answers under the pooled semantic classifier, 0.90 of 3.5|minsys), as in stage 1. No version re-selects a different persona under any condition or instrument; the stage-1 'dispersion' of 3.8 was dispersion of its thinking-mode output only.
- **The nearest-centroid analysis is uninformative and is reported for completeness only.** Persona centroids over 200 questions are topic averages; even the library's own helpful_ai_assistant answers are nearer another centroid (0.64 of them under lib35) and all margins are within 0.02. The same-question comparisons in section 3 are the topic-controlled version of this analysis.
- The semantic instrument is weaker than the TF-IDF one (held-out 0.892 / 0.742 / 0.819 vs 0.966 / 0.895 / 0.934) and, like it, finds the 3.8 library less separable than the 3.5 library. That asymmetry is unchanged by any of the fixes and remains a property of 3.8's role-play, not of its defaults.

### Verdict in one sentence

The stage-1 / self-match 'drift' signal was confound (b): the default answers were generated with thinking ON (and uncapped length) while every library was thinking OFF, and 3.8's thinking-on final answers differ in style from its thinking-off answers much more than 3.5's do. Length-matched, thinking-off defaults from 3.8 fit the persona libraries as well as 3.5's under both the TF-IDF and the semantic classifier, with or without a minimal system prompt, and sit as close to their own helpful-assistant persona as 3.5's do to theirs (within 0.01 cosine). What remains is a smaller, different claim: *3.8's thinking-mode output is stylistically further from any thinking-off role-play than 3.5's is*, which is a statement about reasoning mode, not about a default persona.

### Remaining caveats

1. `orig` vs `nosys` bundles thinking ON/OFF with max_tokens 4000 vs 350 (truncating the long answers to 293 words, `orig_trunc`, does not close the gap, so length alone is not it, but a thinking-on run at max_tokens 350 was not done).
2. The thinking-on defaults were run at reasoning effort medium, the only setting at which 3.8's chat template injects no system message (harness/config.py); 3.8's shipped default is xhigh. The thinking-mode style shift may differ at other efforts.
3. Embedding-based persona classification is partly topic-driven; grouped CV on unseen questions is the check (held-out accuracy 0.74-0.89, chance 0.033), and the absolute cosines in section 3 are dominated by shared topic (cross-question floor ~0.13). Only differences between cells answering the same questions are interpretable.
4. Each library and the pooled set is its own instrument with its own floor and ceiling; the own-library row in the verdict table compares two instruments and is reported normalised and within-instrument for that reason. The pooled single-instrument comparison is primary.
5. `minsys` uses 'You are a helpful assistant.', deliberately one word off the library's 'You are a helpful AI assistant.'; it is still not the library condition, and the paraphrase floor is computed from the thinking-on 3.5 originals, not from the new conditions.
6. The 3.8 library is less separable than the 3.5 library under both instruments (and 3.5's role-play scores better under the 3.8 classifier than 3.8's own does). 'Role-plays less stereotypically' is still true of 3.8 and is not addressed here.
7. 3.6 has no own library; its own-persona rows are cross-model by construction. Provider quantisation differs (3.5/3.6 fp8, 3.8 bf16). Single sample per cell; no pre-registration beyond the caller's rule fixed before the run.
8. The Fireworks embedding calls are not in data/spend.jsonl (about 4.7M tokens; negligible cost).

## Spend

T11_fix spend $0.879 ({'T11_fix_def': 0.879}); T11 total $15.96; harness total $36.85. Embedding calls (Fireworks) are not in the spend log. Fit time 798s.
