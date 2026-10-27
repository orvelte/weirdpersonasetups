# T11 design options: default answers as a mixture of personas (Olivia to choose)

**Question.** Are each version's default Assistant answers well described as a mixture of pretraining-style personas (selection), or does RLVR produce something "homologous" that fits no persona well (drift)? Source: Oz Amram's comment (mixture fit) and Kokotajlo/Marks ("mammalian forelimbs").

**Predictions.** Selection: good fit for every version, possibly a shift in which persona fits best. Drift: the unexplained residual grows from 3.5 to 3.8.

## Option 1. Classifier fit (API only)

- Reference library: Qwen3.5-27B answers 200 questions under each of 30 role prompts (engineer, nurse, teacher, startup founder, bureaucrat, stand-up comic, …, plus "helpful AI assistant" as one persona). 6,000 generations.
- Train embedding + logistic regression on persona labels (held-out questions). Report held-out accuracy; if < 60% the library is not separable enough and the method is uninformative, stop.
- Classify each version's *default* answers to the same 200 questions. Measure: entropy of the predicted persona distribution per answer; max-class confidence; fraction of answers below a confidence floor; and "which persona wins".
- Noise floor: paraphrase each default answer with a separate model; treat the classifier's instability under paraphrase as the floor.
- **Weakness:** confidence can fall because 3.8 writes in a *new* style, or because it writes in a *blander* style. A bland answer fits "no persona" without being homologous. Mitigation: include a "generic assistant" persona and a "textbook" persona in the library.
- **Cost:** ~6,000 reference + 600 default + 600 paraphrase generations, thinking off for the reference library (role-played persona text, not an Assistant turn), ≈ $8–10. One afternoon of compute.

## Option 2. Likelihood fit (self-hosted, GPU)

- Same reference personas, but no classifier: score each version's default answer under Qwen3.5-27B conditioned on each persona prompt, giving a 30-dim log-likelihood profile per answer.
- Fit the best convex mixture over personas per version (EM over persona weights). Report the unexplained gap: log-lik under the best mixture minus log-lik of the answer under the model's own default prompt, against a paraphrase floor.
- **Strength:** uses the model's own likelihoods, no embedding artefacts; directly operationalises "fits no pretraining persona".
- **Weakness:** the reference model is itself post-trained (no 27B base), so "pretraining personas" really means "3.5's role-played personas". Could add Qwen3.5-9B-Base as a second reference, at the cost of a size mismatch.
- **Cost:** 600 answers × 30 personas × ~400 tokens of teacher-forced scoring ≈ 7M tokens forward-only; ~1–2 GPU hours on an A100 ($2–4) plus the generations from Option 1.

## Option 3 (my recommendation). Two-stage: cheap classifier screen, then likelihood on the residual

- Run Option 1 first because it is cheap and gives a readable "which persona wins" picture per version.
- Then run Option 2 only on the 100 default answers with the lowest classifier confidence per version, which is where drift would live. This cuts the GPU scoring by ~6× and gives both the interpretable winner and the likelihood-based residual.
- Add a **within-version context gap** (hard rule 1): run the default answers in a chat frame and an agentic frame. If the agentic-frame answers fit worse than the chat-frame answers *and* the gap grows with version, that is the split signature in persona space rather than behaviour space.
- **Cost:** ≈ $10 API + ≈ $2 GPU. Two days including validation.

## What I need from Olivia

- Choice of option (I recommend 3).
- Whether the 200 questions should be the T1 scenarios (behavioural, decision-rich) or generic open questions. I recommend 100 of each.
