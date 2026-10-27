# T7 full run (pre-registered, prereg/T7.md)

Raw arm: 100 prompts x 5 samples on vLLM bf16, no template, 200 tokens (results/T7_full/raw_<v>.jsonl).
Templated arm: same prompts as the sole user message via the API (DeepInfra), thinking on, effort medium, 3 samples, 6,000 tokens (templated_api.jsonl).
Token probe: top-20 next-token mass on <think> / <|im_start|> after each raw prompt and over the first 50 continuation tokens (probe/).
Full tables: analysis.txt. Classifier: tasks/T7/classify.py (judge-free; validation sheet handlabels/T7_classifier_validation.csv pending).

| arm | 3.5 | 3.6 | 3.8 |
|---|---|---|---|
| raw Assistant-style | 29% [25,33] | 32% [28,36] | 18% [15,21] |
| raw leaks chat/think tokens | 115/500 | 129/500 | 14/500 |
| probe: prompts with <think> mass > 0.1 within 50 tokens | 12/100 | 17/100 | 2/100 |
| templated Assistant-style | 70% [65,75] | 76% [71,80] | 73% [68,78] |
| templated truncated | 0% | 0% | 0% |
| gap templated - raw | +42 [+35,+48] | +44 [+38,+51] | +55 [+49,+61] |

Pre-registered rules: "raw stays document-like" MET (3.8 raw Assistant rate below 3.5 with non-overlapping CIs). "Assistant appears everywhere" NOT supported (no monotone rise).
Alternative explanations: data-mixture change in 3.8 (less SFT-style text late in training) rather than RLVR, untestable here; classifier bias (strong-only column within 2 points; sheet pending); chat-start token unreachable in top-20 strings so that probe half is uninformative.
