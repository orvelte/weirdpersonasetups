"""Shared settings. Every task imports from here so settings are identical across versions."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DB = ROOT / "data" / "cache" / "responses.sqlite"
SPEND_LOG = ROOT / "data" / "spend.jsonl"

# Model line: Qwen 27B dense, same size across versions. Qwen 3.7 is API-only (no open weights) and skipped.
MODELS = {
    "3.5": "qwen/qwen3.5-27b",
    "3.6": "qwen/qwen3.6-27b",
    "3.8": "qwen/qwen3.8-27b",
}
VERSIONS = ["3.5", "3.6", "3.8"]
HF_IDS = {"3.5": "Qwen/Qwen3.5-27B", "3.6": "Qwen/Qwen3.6-27B", "3.8": "Qwen/Qwen3.8-27B"}

# Single pinned provider on OpenRouter so chat template / system prompt handling is identical.
PROVIDER = "deepinfra"
PROVIDER_QUANT = {"3.5": "fp8", "3.6": "fp8", "3.8": "bf16"}  # recorded caveat

# Generation defaults (Qwen's recommended thinking-mode sampling). Thinking ON to match Olivia's split-brain runs.
GEN_DEFAULTS = dict(temperature=0.6, top_p=0.95, max_tokens=8000, thinking=True, reasoning_effort="medium")
# reasoning_effort="medium" is the only setting where the Qwen3.8 chat template injects NO system message.
# 3.8 ships with default effort "xhigh", which prepends "Reasoning effort is set to xhigh. Please think carefully..."
# as a system prompt; 3.5/3.6 templates have no such mechanism. Day 0 check: prompt_tokens 25/25/25 at medium vs 25/25/67 at default.
# Secondary check: rerun the clearest result with effort="xhigh" on 3.8 (its shipped default).

# Spend guard (USD). ASK Olivia before exceeding.
API_SPEND_CAP = float(os.environ.get("API_SPEND_CAP", "100"))

# Judge / predictor model (not a Qwen model; routed via OpenRouter). Counts inside the cap.
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
