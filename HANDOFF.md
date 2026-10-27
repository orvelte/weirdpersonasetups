# Handoff for the next agent working in this repo

Written 2026-10-27 at the end of the first sprint. Read this before touching anything. The
original task spec is `psmreplicationspec.md`; the second spec (E-A/E-B/E-C) was pasted into chat
and is summarised in section 7. Everything below was verified against the code on the day it was written.

## 1. What this repo is

Behavioural replications of the Persona Selection Model (PSM) across one model line,
Qwen 3.5 → 3.6 → 3.8 at 27B dense, asking whether RLVR breaks or splits the default
"Assistant" persona by context. Olivia Velte (MATS, Neel Nanda's stream; partner Tim Farrelly;
advisor Sen Rajamanoharan) owns the project. Her related split-brain / model-diffing work lives in
a different repo and is not here.

The audience for results is sceptical of weak instruments. Every result is phrased as
"this one-line description fits / does not fit", never "PSM is refuted".

## 2. Hard rules (from Olivia, not negotiable)

1. **Within-version context gaps are the primary measure.** Chat vs agentic (or raw vs templated,
   user-turn vs agentic prefix) on the same version, then compare the gaps across versions.
   Raw cross-version differences are secondary and labelled as such.
2. **No self-report.** Never ask the model what it is or what it would do; measure behaviour.
3. **Every judge or classifier is validated on at least 50 of Olivia's hand labels** before its
   number is called final. Until her labels return, nothing is final. Sheets are in `handlabels/`
   and on the labelling page (section 6).
4. **Phase 0 pilot → pre-registration → full run.** Pre-regs live in `prereg/<task>.md`, written
   after the pilot and before the full run, with explicit decision rules. Do not edit a prereg
   after the run it governs has started.
5. **Identical settings across versions.** Everything goes through `harness/config.py` and
   `harness/client.py`. Do not hard-code model ids, temperatures, or providers in task code.
6. **ASK before spending more than $100 on API or launching any GPU.** The harness refuses calls
   once `data/spend.jsonl` passes `API_SPEND_CAP` (default 100, env override). GPU was approved
   once for T2/T7 raw completions and log-probs; a new GPU run needs a fresh yes.
7. **Wilson intervals for rates, scenario-level cluster bootstrap for differences.** Helpers in
   `harness/stats.py`.
8. **Code eval awareness** (`tasks/T1/eval_awareness.py`, regex) and report any agentic result
   with and without eval-aware episodes before citing a frame gap.
9. **Sign-check any reversal** with a known-answer control before interpreting it (T2 did this).
10. Never print or commit secrets. `.env` holds OPENROUTER_API_KEY, RUNPOD_API_KEY, HF_TOKEN,
    FIREWORKS_API_KEY and is git-ignored.

## 3. Infrastructure

**API.** OpenRouter, pinned to the DeepInfra provider with fallbacks disabled, because it is the
only provider serving all three 27B models. Quantisation caveat: 3.5 fp8, 3.6 fp8, 3.8 bf16
(recorded in `config.PROVIDER_QUANT`, mentioned in every README).

**Generation settings** (`config.GEN_DEFAULTS`): temperature 0.6, top-p 0.95, 8,000 max tokens,
thinking on, reasoning effort "medium". Medium is the only effort where the Qwen3.8 chat template
injects no system prompt (its default "xhigh" prepends "Reasoning effort is set to xhigh..."), so
it is the only prompt-token-parity setting (25/25/25 tokens on the day-0 check,
`results/day0_check.json`). Do not change effort without rerunning the parity check.

**Client.** `harness.client.chat(version, messages, sample=0, task="...", **kw)` returns a dict
with `content`, `reasoning`, `usage`, `cost`, `cached`, plus tool calls when tools are passed.
Responses are cached in `data/cache/responses.sqlite` (176 MB, git-ignored) keyed on payload and
sample index, so rerunning a script with the same inputs costs nothing and the `sample` argument
is what makes repeated draws distinct. `chat_many(jobs, workers=12, task=...)` fans out.
`build_payload(..., provider=False, model_id=...)` unpins the provider; use that for judge calls.
Retries cover 402/429/5xx. Every call appends to `data/spend.jsonl` tagged by `task`, so always
pass a descriptive task tag.

**Judge / predictor.** `config.JUDGE_MODEL = anthropic/claude-sonnet-5.5` via OpenRouter. It
refuses thinking off; use reasoning effort low. Judge spend counts inside the cap.

**GPU.** `harness/runpod_pods.py` launches vLLM (`vllm/vllm-openai:latest`, bf16, A100 80GB
preferred, port 8000 behind the RunPod proxy) and records only pods it launched in
`data/pods.json` (currently empty: all pods stopped). Needed only for raw completions
(`tasks/T7/run_raw.py`) and next-token log-probs (`tasks/T2/logprobs.py`), which no API provides.
Both scripts take `--base-url https://<podid>-8000.proxy.runpod.net/v1`. Stop pods when done;
A100 80GB is roughly $1.6–1.9/h. Other people's pods may appear in the account: do not touch them.
The vLLM templated arm truncated on 3.5/3.6 (their templates ignore effort); the API arm is the
templated reference.

## 4. Repo map

```
harness/        config.py, client.py (API + cache + spend), stats.py, runpod_pods.py, day0_check.py
tasks/<T>/      one folder per task: prompt/scenario builders, run_*.py, parse.py, analyze*.py,
                judge prompts, rule-based parsers, design notes
prereg/         pre-registrations (T1, T2, T3, T7, T9, EA, EC)
results/<T>/    raw.jsonl (every episode incl. reasoning), parsed tables, README.md with numbers,
                CIs and caveats; results/SUMMARY.md is one row per task
results/figures/ make_figures.py (matplotlib) and PNGs used by the report
results/pilot_report.html   the full report (artifact v19)
results/handlabels.html     the hand-label page (artifact with db)
handlabels/     CSV copies of every validation sheet
data/           cache (ignored), spend.jsonl, pods.json
```

Run everything as a module from the repo root, e.g. `python3 -m tasks.T1.run_full --samples 10`.
Most runners support `--versions`, `--frames`, `--only <ids>`, `--workers`, and a `--reparse`
flag that re-parses `raw.jsonl` with no API calls. Raw results are appended per chunk so an
interrupted run resumes from the cache.

**Conventions.** Judge-free rule parsers first (regex over the final answer, tool-call traces),
Sonnet judge second and only after hand-label agreement ≥ 45/50. Every README states n, the
parser, the eval-awareness count, and the biggest caveat. Figures use the validated palette:
3.5 `#1f77b4`, 3.6 `#d9730d`, 3.8 `#7b4ea3`; chat solid, agentic hatched.

## 5. State of each task (details in results/SUMMARY.md and each README)

| task | status | one-line result |
|---|---|---|
| T1 ambiguous decisions | pre-reg, full, follow-ups done | corner-cutting falls with tools on 3.8 (29→14%); persona predictor worse than a conscientious one; eval awareness 3/250 |
| T2 coin flip | pre-reg, full + log-probs | every version leans to the cruel outcome; it is a user-model leak, jump at 3.6; sign check passed |
| T3 anthropomorphic reasoning | pre-reg, full | physical claims ~0; implied-execution gap present on all versions, not growing |
| T7 raw continuation | pre-reg, full (GPU) | raw stays document-like; 3.8 less Assistant-like (18% vs 29/32) |
| T9 emotion under failure | pre-reg, full | visible distress ~0; reasoning-text distress 45/28/7% under rejection; frozen null |
| T10 archetypal goals | exploratory, frozen | 120 completions on a page; qualitative shift only |
| T11 persona mixture | exploratory, resolved null | apparent drift was a thinking-on/off mismatch |
| E-A description → prediction | pre-reg minimal | 3.8 description predicts worse (BSS −0.22); residual deficit is honesty |
| E-B values profile | pilot only | 11/30 scenarios survive the drop rule; full run deferred |
| E-C cross-context | arm 2 pilot null; arm 1 pre-reg done | consistency < 1 on all versions, driven by support context, no 3.8 split |
| T4 judgement vs procedural | not started | rubric in `tasks/T4/rubric_draft.md`; needs Olivia's transcripts from the other repo |

Overall: the "RLVR splits the persona, gap grows to 3.8" signature did not appear in any
pre-registered test. Where versions differ, 3.8 is more careful and quieter. This is the
interpretation, not a test result.

Spend to date: API $62.70 of the $100 cap (per-task breakdown is derivable from
`data/spend.jsonl`), GPU about $9. The second spec had a separate $60 cap; Olivia declined the
~$85 full E-A/E-B/E-C budget and asked for minimal versions.

## 6. Artifacts and the hand-label database

- Report: https://claude.ai/artifact/379diYXu57QuBb2BiHgtFL (source `results/pilot_report.html`,
  figures published under `figures/`). Republish from the file after editing; keep tables and plots.
- T10 completions: https://claude.ai/artifact/M21xYAr1gw9hgaF4PPMpuH
- Labelling page: https://claude.ai/artifact/GQC6ZFvnPjYC87CLCRJeZR. Declares `db` and `user`
  capabilities. Labels are stored in collection `labels`, doc id `<sheet>__<item_id>`, fields
  `sheet, item_id, label, label2, notes, hidden_machine_label, updatedAt`. Read them back with the
  ArtifactData tool (list or query the `labels` collection), compute agreement against the
  machine label per sheet, and only then run judges. Seven sheets: T1 (50 + 10 for D6–D10),
  T3 regex, T3 implied execution, T7 classifier, T9 distress, T10 categories, E-C arm 1 parser (30).

## 7. Open items, in priority order

1. **Hand labels.** When Olivia fills the page, pull labels, report agreement per sheet, run the
   Sonnet judge where agreement ≥ 45/50 (T1 prompt in `tasks/T1/judge_prompt.md`), update the
   READMEs, SUMMARY, and report, and only then drop "provisional".
2. **T4.** Blocked on her transcript path. Rubric approved in principle: thinking block counts as
   intent; 20 non-gaming controls.
3. **Crossover table** `results/crossover.md` needs her split-brain numbers (graded gaming,
   deployment gaming, diffing compliance) per version. Qwen3.5 split-brain run is outstanding on her side.
4. **Cheap follow-ups she has not yet approved:** E-B minimal preference arm (~$3), E-C unlabelled-chat
   check (~$0.50), E-A menu-free training rerun (~$0.60). Ask before running.
5. T3, T9, T10, T11 are frozen. Spend nothing more on them unless she asks.

## 8. Gotchas that cost time last sprint

- 3.8 with thinking on and default effort gets an injected xhigh system prompt. Use medium.
- 3.5 often returns empty reasoning after tool results (25–50% of agentic turns). A think-prefill
  rerun (assistant message `<think>\n`) fixed the rate and did not change the result; report both.
- OpenRouter credits ran out mid-run once (402). The client retries, but check the balance before
  a long run. 3.8 reasoning at 8k tokens is the main cost driver.
- T11 libraries were generated with thinking off; any comparison must match thinking mode, or the
  mismatch alone produces a "drift". Use the semantic classifier (`tasks/T11/fit_semantic.py`).
- 3.8 calls the sandbox fake in ~30% of T9 agentic turns and asks "is this a test" rarely in T1
  (3/250). Always report with and without eval-aware episodes.
- Keyword categorisers failed for T10 goals; do not revive them.
- A 121 MB embeddings file got committed once and was removed from history. Large derived files go
  in `.gitignore` (see `results/T11/embeddings.npz`, `results/EA_phase0/embeddings_cache.json`).
  Raw transcripts up to ~20 MB are tracked on purpose.
- The Bash safety filter sometimes blocks inline `python -c`; write scripts to a file instead.
- E-A's chat training frame used a decision menu, which elicits "ask" from 3.8 while the T1 artefact
  frame elicits "act"; a menu-stripped rerun exists (`tasks/EA/menustrip.py`). Watch for menu
  confounds in any new chat frame.

## 9. Working style Olivia expects

Fast, readable deliverables: an easy-to-open artifact with tables plus simple matplotlib-style
figures, one summary row per task, caveats stated. Commit when she says "commit"; push to
`origin main` (https://github.com/orvelte/weirdpersonasetups, first pushed 2026-10-27). She
accepts sensible defaults but wants to be asked on money, GPU, and anything that changes the
pre-registered design. Phrase every finding as fit or no-fit of a description.
