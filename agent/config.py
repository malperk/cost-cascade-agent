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

# File/code tools are sandboxed to this directory - agents cannot read or write
# anything outside of it (see agent/tools.py:_resolve_safe_path).
WORKSPACE_DIR = Path(os.getenv("AGENT_WORKSPACE_DIR", "./workspace")).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
