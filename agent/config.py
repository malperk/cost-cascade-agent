import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
OMNIROUTE_API_KEY = os.getenv("OMNIROUTE_API_KEY", "omniroute-local")

LOW_MODEL = os.getenv("LOW_MODEL", "alper-low")
MID_MODEL = os.getenv("MID_MODEL", "alper-mid")
HIGH_MODEL = os.getenv("HIGH_MODEL", "alper-high")
# Last-resort tier with zero external quota (local Ollama) - reachable only when every
# cloud tier's provider pool is exhausted, so the system can never be fully stuck.
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "alper-local")

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "4"))

# Normal runs start triage at "low" and let review-driven escalation move up from there.
# The self-improvement runner overrides this to "mid" for its authoring container, since a
# long, meta, instruction-heavy prompt (read this context, pick a hypothesis, edit files,
# commit) proved unreliable at "low" tier in live testing - triage under-assessed the task's
# risk and the model disengaged from it entirely rather than following the instructions.
AGENT_MIN_TIER = os.getenv("AGENT_MIN_TIER", "low")

# Review-driven escalation (agent/nodes.py:escalate_node) normally climbs all the way to
# "local" (Ollama llama3.1:8b) as its last resort - fine for normal runs, where "local" exists
# specifically so the system can never be fully stuck when every cloud tier is exhausted. But
# "local" is a genuinely weak model for anything requiring precise multi-step tool calling: a
# live self-improvement authoring cycle escalated there on a review failure and it emitted a
# tool call as plain-text JSON instead of an actual tool invocation - correctly caught by
# review as an unverified claim, but a wasted attempt that a stronger model wouldn't have
# wasted. The self-improvement runner caps its authoring container at "high" via this env var
# (default "local" - i.e. uncapped - for every normal run) so escalation repeats at "high"
# instead of dropping to a weaker tier that's less likely to help.
AGENT_MAX_TIER = os.getenv("AGENT_MAX_TIER", "local")

# Triage backend. "auto" (default) asks TypeSafe's Jev (a hosted decision model, see
# agent/jev_triage.py) whenever TYPESAFE_API_KEY is set, and uses the OmniRoute LLM triage
# otherwise; "llm" never calls Jev. Jev is only consulted for short tasks at the default
# "low" start tier - long or meta prompts (and the self-improvement runner, which raises
# AGENT_MIN_TIER) keep the LLM triage and are never sent to a third party. Any API error or
# an answer below JEV_CONFIDENCE also falls back to the LLM triage.
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
TRIAGE_BACKEND = os.getenv("TRIAGE_BACKEND", "auto")
JEV_BASE_URL = os.getenv("JEV_BASE_URL", "https://api.typesafe.ai")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
JEV_CONFIDENCE = float(os.getenv("JEV_CONFIDENCE", "0.7"))
JEV_TIMEOUT = float(os.getenv("JEV_TIMEOUT", "10"))
JEV_MAX_TASK_CHARS = int(os.getenv("JEV_MAX_TASK_CHARS", "1500"))

# File/code tools are sandboxed to this directory - agents cannot read or write
# anything outside of it (see agent/tools.py:_resolve_safe_path).
WORKSPACE_DIR = Path(os.getenv("AGENT_WORKSPACE_DIR", "./workspace")).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Optional deterministic completion check: if set, a top-level task is not accepted until this
# file exists in the workspace, regardless of what the model or reviewer says.
AGENT_REQUIRED_OUTPUT = os.getenv("AGENT_REQUIRED_OUTPUT", "")

# Written by the deterministic model-maintenance service (~/Codes/Alper/Omniroute). Read-only
# here: used to warn at startup when a tier's combo is short on working models.
MODEL_CATALOG_PATH = Path(
    os.getenv("MODEL_CATALOG_PATH", str(Path(__file__).resolve().parents[2] / "Omniroute" / "state" / "model_catalog.json"))
)

# Browser tools (agent/browser.py). BROWSER_ENABLED=false removes them from the agent. They are
# also never registered inside the self-improvement containers (no network there).
# BROWSER_DECISIONS picks who chooses the element for browser_act: "llm" (default) always asks
# the OmniRoute LLM; "auto" asks Jev first (fast, closed-choice) and falls back to the LLM when it is
# unsure or fails. It is opt-in because Jev would be sent the instruction, the page URL (without
# query/fragment) and the visible element labels of whatever page the agent is on - do not enable
# it for logged-in or private pages. BROWSER_EXECUTABLE overrides the Chromium binary.
BROWSER_ENABLED = os.getenv("BROWSER_ENABLED", "true").lower() != "false"
BROWSER_DECISIONS = os.getenv("BROWSER_DECISIONS", "llm")
BROWSER_EXECUTABLE = os.getenv("BROWSER_EXECUTABLE", "")
