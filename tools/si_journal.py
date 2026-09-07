"""Crash-safe cycle journal for the self-improvement runner (plan section 4).

Written with atomic replace (write-tmp -> fsync -> os.replace -> fsync the
containing directory) at every phase transition, so a `kill -9` mid-cycle
leaves a file recovery can actually read real state back from - not just a
phase label that lost the SHAs/combo-snapshots/transport-resource-ids it
needs to decide forward-completion vs. rollback.

Lives under .self-improve-runtime/ (gitignored), never inside the tracked
working tree the runner also has to treat as "must be clean."
"""

import json
import os
from pathlib import Path

PHASES = [
    "STARTED",
    "BASELINE_DONE",
    "CANDIDATE_RECEIVED",
    "CANDIDATE_VALIDATED",
    "EVAL_PASSED",
    "PROMOTION_STARTED",
    "PROMOTION_VERIFIED",
    "METADATA_COMMITTED",
    "DONE",
]

DEFAULT_JOURNAL_PATH = Path(__file__).parent.parent / ".self-improve-runtime" / "journal.json"


class Journal:
    def __init__(self, path: Path = DEFAULT_JOURNAL_PATH):
        self.path = Path(path)
        self.data: dict = {}

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> dict:
        self.data = json.loads(self.path.read_text())
        return self.data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            f.write(json.dumps(self.data, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path)
        dir_fd = os.open(str(self.path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def set_phase(self, phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase: {phase}")
        self.data["phase"] = phase
        self.save()

    def update(self, **fields) -> None:
        self.data.update(fields)
        self.save()

    def delete(self) -> None:
        if self.path.exists():
            self.path.unlink()
