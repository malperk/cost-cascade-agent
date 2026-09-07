"""The only writer of state/improvement_history.json.

Called only by tools/self_improvement_runner.py, never by the LLM agent
directly - the agent has no instructed path to this script (see the plan's
prompts/self_improvement_cycle.txt scope note). Enforces append-only
semantics: an existing entry for a cycle_id is never edited, only new
entries are added, and every entry is validated against
state/improvement_history.schema.json before being written.
"""

import argparse
import json
import jsonschema
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent.parent / "state" / "improvement_history.schema.json"
HISTORY_PATH = Path(__file__).parent.parent / "state" / "improvement_history.json"


def append(entry: dict) -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(entry, schema)

    history = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else []
    if any(existing["cycle_id"] == entry["cycle_id"] for existing in history):
        raise ValueError(f"cycle_id {entry['cycle_id']} already has a history entry - append-only, cannot overwrite")

    history.append(entry)
    HISTORY_PATH.write_text(json.dumps(history, indent=2) + "\n")


def has_failed_fingerprint(fingerprint: str) -> bool:
    """Used by the runner's duplicate-hypothesis check before spending any eval budget."""
    if not HISTORY_PATH.exists():
        return False
    history = json.loads(HISTORY_PATH.read_text())
    return any(e["fingerprint"] == fingerprint and e["result"] == "fail" for e in history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-file", required=True, help="path to a JSON file containing one history entry")
    args = parser.parse_args()

    entry = json.loads(Path(args.entry_file).read_text())
    append(entry)
    print(f"appended history entry for cycle_id={entry['cycle_id']} result={entry['result']}")


if __name__ == "__main__":
    main()
