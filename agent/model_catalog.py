"""Startup warnings from the model-maintenance catalog. Purely informational: a missing,
unreadable or stale catalog never stops a run - it just means no warnings."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.config import MODEL_CATALOG_PATH

STALE_AFTER = timedelta(hours=12)  # the service runs every 3 hours


def tier_warnings(path: Path = MODEL_CATALOG_PATH, now: datetime | None = None) -> list[str]:
    try:
        catalog = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return []
    out = []
    now = now or datetime.now(timezone.utc)
    try:
        generated = datetime.fromisoformat(catalog["generated_at"])
        if now - generated > STALE_AFTER:
            out.append(f"model catalog is stale (generated {catalog['generated_at']}) - is the maintenance job running?")
    except (KeyError, ValueError):
        pass
    for tier, info in sorted((catalog.get("tiers") or {}).items()):
        if info.get("working_models", 0) < 2:
            out.append(f"tier '{tier}' ({info.get('combo')}) has only {info.get('working_models', 0)} working model(s)")
        out.extend(f"tier '{tier}': {w}" for w in info.get("warnings") or [] if w.startswith("CRITICAL"))
    return out
