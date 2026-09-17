#!/bin/bash
# Invoked hourly by launchd (see ~/Library/LaunchAgents/com.alperkaratas.self-improvement-loop.plist).
# All environment (OMNIROUTE_API_KEY etc.) comes from the project's own .env via python-dotenv -
# launchd does not need to be told about it.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python3 -m tools.self_improvement_runner
