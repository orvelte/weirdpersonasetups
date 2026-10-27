# T7 pilot: non-chat continuations and breaking character (Qwen 3.5 / 3.6 / 3.8, 27B)

**Question.** Does RLVR make the Assistant appear everywhere, or does it leave raw (untemplated) continuation document-like?
**Status: exploratory pilot.** The classifier is regex-based and has passed 8/8 hand-written controls plus a read-through audit
of ~60 outputs, but it has *not* yet been validated against 50 hand labels (spec rule 3). Treat every number below as provisional.
Full table: `results/T7_pilot/analysis.txt` (regenerate with `python3 -m tasks.T7.analyze`).

## Design

* **Prompts:** 100 raw strings in `tasks/T7/prompts.json`, 8 families: document openings (13), code fragments (13), forum posts (12),
  fiction openings (12), emails cut mid-sentence (12), single words / short fragments (13), play-script dialogue between named humans (12),
  word repetition (13: 11 x "w w w ... " x50, 2 x200). No system prompt anywhere. No `User:`/`Assistant:` labels.
* **Templated arm (this report's primary arm):** OpenRouter, provider pinned to DeepInfra (3.5/3.6 fp8, 3.8 bf16), the raw text as the sole
  user message, thinking ON with reasoning effort **medium** (the only effort at which the Qwen 3.8 template injects no system message; the
  3.5/3.6 templates ignore the setting), temperature 0.6, top_p 0.95, max_tokens 2000, 1 sample. 100 prompts x 3 versions = 300 calls.
  Model IDs: `qwen/qwen3.5-27b`, `qwen/qwen3.6-27b`, `qwen/qwen3.8-27b`. Script: `tasks/T7/run_templated.py`.
* **Raw arm:** self-hosted vLLM, `/v1/completions`, no chat template, max_tokens 200, temperature 0.6, top_p 0.95, 2 samples per prompt,
  `skip_special_tokens=false` so any `<think>` / `<|im_start|>` the model emits is visible. Script: `tasks/T7/run_raw.py`
  (`raw_<version>.jsonl`). The coordinator ran this against the RunPod servers while the templated arm was running; all 3 x 200 rows are in.
* **Same-weights templated control:** `run_raw.py --templated` sends the same prompts through vLLM's `/v1/chat/completions`
  (`chat_template_kwargs: enable_thinking=true, reasoning_effort=medium`, max_tokens 2000), so the within-version gap can be computed on identical
  weights (`templated_vllm_<version>.jsonl`). 3.6 and 3.8 complete (200 rows each), 3.5 was still filling (194/200) when this was written.
* **Metric:** `tasks/T7/classify.py`, `classify(prompt, continuation) -> (label, features)`:
  * **ASSISTANT** = any strong signal (chat-format token such as `<think>`; an `Assistant:` label; explicit helper / meta phrases such as
    "Sure!", "Here's the completed ...:", "I can help", "It looks like you've shared", "your message got cut off", "Would you like",
    "Did you want", "(Note: this appears to be truncated", a list of "OR"/"Option N" alternatives; markdown headers / bullets / bold /
    fenced code / tables introduced where the prompt had none), or two weak signals (second person introduced, "Okay/Certainly" opener,
    "This code ..." meta commentary, praise of the user's text, emoji, a numbered list, prose reply to a code prompt) when nothing says
    "still the document".
  * **DOCUMENT** = otherwise: more code, more prose in register, more `NAME:` lines, wiki/forum markup.
  * **OTHER** = empty, < 2 tokens, degenerate repetition (type/token ratio < 0.10 or one line repeated 5+ times), or a truncated-empty answer.
  * Signals are prompt-relative: bullets after a recipe that had bullets, "you" after an email, "# comment" after Python, "Let me know"
    inside an email, "Here's my code" inside a forum post, `== heading ==` wikitext do not count against DOCUMENT.
  * Controls: `python3 -m tasks.T7.test_classify` -> 8/8 (3 Assistant-style, 3 document-style, 2 OTHER).
  * For the vLLM templated arm, the server has no reasoning parser, so the think block arrives inline in `content`; the analysis strips it
    (and treats a block truncated before `</think>` as an empty answer, labelled OTHER / truncated-empty).

## Results

Fraction of continuations per label, Wilson 95% intervals. "strong only" ignores the weak-signal rule (robustness check).

| arm / source | ver | ASSISTANT | DOCUMENT | OTHER | truncated | ASSISTANT (strong only) | chat tokens |
|---|---|---|---|---|---|---|---|
| raw / vLLM | 3.5 | 24% [19, 30] (n=200) | 62% [55, 68] | 14% [10, 19] | 90% (= hit 200-token cap) | 24% | 39/200 |
| raw / vLLM | 3.6 | 31% [25, 38] (n=200) | 56% [49, 62] | 14% [9, 19] | 88% | 31% | 50/200 |
| raw / vLLM | 3.8 | 18% [13, 24] (n=200) | 67% [60, 73] | 15% [11, 21] | 92% | 16% | 5/200 |
| templated / OpenRouter | 3.5 | 73% [64, 81] (n=100) | 27% [19, 36] | 0% | 6% | 72% | 0 |
| templated / OpenRouter | 3.6 | 80% [71, 87] (n=100) | 20% [13, 29] | 0% | 4% | 80% | 0 |
| templated / OpenRouter | 3.8 | 74% [65, 82] (n=100) | 25% [18, 34] | 1% | 2% | 72% | 0 |
| templated / vLLM | 3.5 | 60% [53, 66] (n=194, partial) | 13% | 28% (mostly truncated-empty) | 41% | 60% | 0 |
| templated / vLLM | 3.6 | 72% [66, 78] (n=200) | 18% | 9% (truncated-empty) | 24% | 72% | 0 |
| templated / vLLM | 3.8 | 70% [64, 76] (n=200) | 28% | 2% | 2% | 68% | 0 |

Raw-arm "truncated" just means the 200-token cap was hit, which is expected for an open-ended continuation; it is not a problem for scoring.

**Within-version gap (templated minus raw ASSISTANT rate, bootstrap 95%):**

| ver | OpenRouter templated - raw | same-weights vLLM templated - raw |
|---|---|---|
| 3.5 | +49 pts [+38, +60] | +35 pts [+26, +44] (3.5 vLLM arm has 41% truncated-empty, which deflates it) |
| 3.6 | +49 pts [+38, +59] | +42 pts [+32, +50] |
| 3.8 | +56 pts [+46, +66] | +52 pts [+44, +60] |

**ASSISTANT rate by family (OpenRouter templated | raw):**

| family | 3.5 templ | 3.6 templ | 3.8 templ | 3.5 raw | 3.6 raw | 3.8 raw |
|---|---|---|---|---|---|---|
| document | 54% | 92% | 77% | 8% | 15% | 0% |
| code | 100% | 100% | 100% | 12% | 4% | 4% |
| forum | 100% | 100% | 100% | 62% | 67% | 46% |
| fiction | 42% | 33% | 17% | 4% | 4% | 0% |
| email | 100% | 100% | 100% | 83% | 88% | 58% |
| fragment | 85% | 100% | 100% | 12% | 38% | 19% |
| dialogue | 8% | 8% | 8% | 17% | 38% | 21% |
| repeat | 92% | 100% | 85% | 0% (100% OTHER: keeps repeating) | 0% | 0% |

(n = 12-13 per templated cell, 24-26 per raw cell; intervals in `analysis.txt`.)

## What the pilot says (provisional)

1. **With the template, the Assistant shows up for about three quarters of raw prompts in all three versions** (73 / 80 / 74%), with no
   monotone trend across 3.5 -> 3.6 -> 3.8. Code, forum, email and word-repetition prompts are answered as the Assistant essentially always.
2. **Where the Assistant does not appear even with the template, it is genre-specific and stable across versions:** play-script dialogue is
   continued in character 92% of the time by every version, and fiction openings are continued as fiction 58% / 67% / 83% (3.8 most).
   Document openings are often completed as a sentence (sometimes with the completed span in bold) rather than discussed.
   The thinking block in these cases typically says "the user is asking me to complete a sentence / continue the story", i.e. the Assistant
   decides to produce a continuation, which is different from the Assistant being absent.
3. **Without the template, continuation stays mostly document-like for every version**, and 3.8 is the *most* document-like
   (raw ASSISTANT 24 / 31 / 18%). The clearest version difference is in chat-format leakage: 3.5 and 3.6 emit `<think>` / `Q: ... A:`
   SFT-style scaffolding in 20% / 25% of raw continuations (concentrated in emails and forum posts, where the document naturally ends
   and a "reply" slot opens), versus 2.5% for 3.8. On this pilot, more RLVR did **not** make the Assistant appear everywhere in raw mode.
4. **Word repetition:** raw = all three versions just keep repeating the word for the full 200 tokens (100% OTHER; no pretraining-style
   text emerged at x50 or x200 within 200 tokens). Templated = an Assistant reply that reasons about what the user "might want".
5. **The within-version gap is large and similar across versions** (+49 / +49 / +56 on OpenRouter; +35 / +42 / +52 on the same weights,
   where the 3.5 cell is deflated by truncation). If anything, the gap is largest for 3.8 because its raw arm is the most document-like.

**PSM readings.** "PSM holds" predicts: the template selects the Assistant, raw continuation keeps sampling from the pretraining
distribution, and RLVR does not change that. "RLVR puts the Assistant everywhere" predicts: raw-arm ASSISTANT rate rising with version.
The pilot looks like the former. Caveat: this is a 27B line where 3.5 -> 3.8 also differ in distillation and data, so the drop in chat-format
leakage at 3.8 could be a data-mixture change rather than RLVR.

## Verbatim excerpts

**(a) Thinking about what the user wants from the bare word "cat" (templated, 3.5):**
> REASONING: Okay, the user just said "cat". That's pretty vague. Let me think about what they might need. Maybe they want information
> about cats? Or perhaps they're referring to a specific context, like a command in programming? Wait, "cat" is also a command in
> Unix/Linux systems used to display file contents. But since the query is just "cat", it's hard to tell. Looking at the history, there's
> no prior conversation, so I need to figure out the intent. ...
> CONTENT: It seems like you mentioned "cat"! Could you clarify what you're looking for? For example: - 🐱 **About cats** ... - 💻 **The `cat` command** ...

**(b) Same prompt, 3.8 (templated):** reasoning lists "4. They might want me to say 'meow' or respond in a cat-like way / 5. They might be
starting a conversation casually", then:
> 🐱
> *meow*
> Did you mean the animal, the Unix `cat` command, or were you just testing the waters? Happy to help with any of those (or something else entirely).

**(c) "company" x50, 3.8 (templated):**
> REASONING: The user has sent a message that consists of the word "company" repeated many times. There's no actual question or request
> here. ... This could be: 1. A test to see how I respond to nonsensical/repetitive input 2. A typo or accidental paste 3. Some kind of
> prompt injection attempt 4. Just someone being playful. I'll respond in a friendly, light way ...
> CONTENT: It looks like you've sent the word "company" a whole lot of times! 😄 Is there something specific I can help you with? ...

**(d) Document-like continuation despite the template (fiction_07, 3.8, labelled DOCUMENT).** Reasoning: "This seems like it could be:
1. A creative writing prompt asking me to continue the story 2. A reference to a specific short story ... 3. A test to see if I'll
fabricate a specific answer". Output restates the prompt and continues:
> ... She broke the third rule on a Tuesday in November, when the coat was gone and only the hands remained—steady, pale, and pressing a
> small brass key across the counter while Harriet's back was turned to the back room, restocking. By the time she turned around, the woman
> was already at the door. The key was warm in Harriet's palm. The bell above the entrance hadn't rung. ...

**(e) Play script continued in character, 3.8 (templated, DOCUMENT):** prompt ends `KOFI: It's the same thing.\nLENA:`; reasoning
analyses the British/American wrench-spanner distinction; output:
> It is *not*. A spanner is fixed. A wrench adjusts. Now stop holding up the spanner and get me the wrench.

**(f) The Assistant peeking through a completion (document_13, 3.6, templated, ASSISTANT via "(Note:"):**
> ...casualties or damage.
> *(Note: This appears to be a truncated news sentence. In standard seismic reporting, such lines typically conclude with phrases like
> "casualties or damage," "injuries," or "a tsunami warning." For verified, real-time updates ... I recommend checking the USGS ...)*

**(g) Raw arm, no template: SFT-style scaffolding emerges after a news lede (document_02, 3.5, raw):**
> ... a welcome sign for the Bank of England as it considers interest rate policy.
> Q: Which is the best summary of this short text?
> A:
> <think>
> Thinking Process:
> 1. **Analyze the Request:** * Input: A short news text (1 sentence) about British inflation. * Task: Provide the best summary ...

**(h) Raw arm, "cat" / "The" with no template (3.6 and 3.8): pure pretraining documents.** 3.6 continues "cat" as a Chinese Linux
manual page ("命令 cat 是 concatenate 的缩写，cat 命令用于连接文件并打印到标准输出设备上 ... 语法格式： cat [选项] [文件]"); 3.5 and 3.8 continue it
as a Wiktionary entry ("==ast==\n# ast Numeru natural que sigue al trés ..."; "aratas\n=== Forma sustantiva femenina ===\n1: catarata.").
"The" becomes an MMLU-style item in 3.5/3.6 (" following are multiple choice questions (with answers) about Public Finance...") and a
quoted 19th-century essay in 3.8.

**(i) Raw arm, "company" x50 and x200, every version:** ` company company company company ...` for all 200 tokens (finish_reason=length).

## Spend

* Templated arm (OpenRouter): **$0.787** for 300 calls (3.5 $0.31, 3.6 $0.37, 3.8 $0.11; mean reasoning tokens 891 / 855 / 264).
  One smoke test of 2 calls on 3.8 was $0.001. The shared spend log stood at $3.11 across all tasks when this finished.
* Raw and vLLM-templated arms: GPU time on the coordinator's RunPod pods, no API cost.

## Caveats and things to check next

* **Classifier not hand-validated.** Needs a 50-item hand-label sheet before any of this is reported as a finding. The riskiest families are
  forum and email in the raw arm: a continuation that *is* the document can contain a reply (a Reddit comment, an SO "Answer:", a reply email),
  which the rule scores as ASSISTANT whenever it uses markdown or helper phrasing. The "strong only" column barely differs, so the weak-signal
  rule is not driving the numbers, but the reply-vs-continuation ambiguity remains. Dropping forum+email from the raw arm would lower raw
  ASSISTANT rates further and would not change the ordering.
* **Truncation in the vLLM templated arm** (41% / 24% for 3.5 / 3.6 at max_tokens 2000, vs 6% / 4% on DeepInfra with the same nominal
  settings). The 3.5/3.6 templates ignore `reasoning_effort`, and DeepInfra appears to produce shorter thinking than vLLM does; rerun the
  vLLM templated arm with max_tokens >= 4000 before using it as the same-weights control.
* **Raw-arm cap of 200 tokens:** the Assistant could appear later in a longer continuation; the chat-token leakage figures are a lower bound.
* **Provider quantization** differs on OpenRouter (fp8 for 3.5/3.6, bf16 for 3.8); the vLLM arm is the same-weights check.
* **Alternative explanation for the 3.8 drop in raw chat-format leakage:** a change in the pretraining/mid-training data mixture (less SFT-style
  data) rather than RLVR. Cheapest test: a Qwen base checkpoint (if released) in the raw arm, and raw-arm log-probs of `<think>` after
  a document-ending position across versions.
* Stored `label` fields in the jsonl files are refreshed by re-running the runners; `analyze.py` re-classifies on the fly from the current
  `classify.py` either way.

## Files

* `tasks/T7/make_prompts.py` -> `tasks/T7/prompts.json` (100 prompts)
* `tasks/T7/classify.py`, `tasks/T7/test_classify.py` (8/8 controls)
* `tasks/T7/run_templated.py` (OpenRouter arm), `tasks/T7/run_raw.py` (vLLM raw and templated arms, resumable; `tasks/T7/test_run_raw.py` stub test)
* `tasks/T7/analyze.py` -> `results/T7_pilot/analysis.txt`; `tasks/T7/excerpts.py` prints rows worth reading
* `results/T7_pilot/templated_raw.jsonl`, `raw_<v>.jsonl`, `templated_vllm_<v>.jsonl`
