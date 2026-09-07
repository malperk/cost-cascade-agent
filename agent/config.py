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

# File/code tools are sandboxed to this directory - agents cannot read or write
# anything outside of it (see agent/tools.py:_resolve_safe_path).
WORKSPACE_DIR = Path(os.getenv("AGENT_WORKSPACE_DIR", "./workspace")).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
