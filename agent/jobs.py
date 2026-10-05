import json
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from agent.config import JOBS_DIR

Runner = Callable[[str], dict]
Hook = Callable[[dict], None]


class JobStore:
    """Single-worker async job queue. One task runs at a time: the graph shares one workspace
    directory and one headless browser, so concurrent runs would trample each other.

    Each job is persisted as JSON under JOBS_DIR so results survive a server restart. A job
    that was queued or running when the server died is reported as "interrupted" on reload."""

    def __init__(self, runner: Runner, on_complete: Optional[list[Hook]] = None, directory: Path = JOBS_DIR):
        self._runner = runner
        self._hooks = on_complete or []
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._load()
        threading.Thread(target=self._worker, daemon=True, name="job-worker").start()

    def submit(self, task: str, notify: bool = False) -> dict:
        job = {"id": uuid.uuid4().hex[:12], "task": task, "notify": notify, "status": "queued",
               "created_at": time.time(), "started_at": None, "finished_at": None, "result": None, "error": None}
        with self._lock:
            self._jobs[job["id"]] = job
            self._save(job)
        self._queue.put(job["id"])
        return dict(job)

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list(self, limit: int = 20) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)[:limit]
            return [{k: j[k] for k in ("id", "task", "status", "created_at", "finished_at")} for j in jobs]

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            self._update(job_id, status="running", started_at=time.time())
            try:
                result = self._runner(self._jobs[job_id]["task"])
                self._update(job_id, status="done", result=result, finished_at=time.time())
            except Exception as e:  # a crashing run must not kill the worker
                self._update(job_id, status="failed", error=f"{type(e).__name__}: {e}", finished_at=time.time())
            job = self.get(job_id)
            for hook in self._hooks:
                try:
                    hook(job)
                except Exception as e:  # one broken hook must not block the others
                    print(f"[jobs] completion hook {getattr(hook, '__name__', hook)} failed: {e}", flush=True)

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(fields)
            self._save(job)

    def _save(self, job: dict) -> None:
        tmp = self._dir / f"{job['id']}.json.tmp"
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2))
        tmp.replace(self._dir / f"{job['id']}.json")

    def _load(self) -> None:
        for path in self._dir.glob("*.json"):
            try:
                job = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if job.get("status") in ("queued", "running"):
                job.update(status="interrupted", finished_at=job.get("finished_at") or time.time())
                self._save(job)
            self._jobs[job["id"]] = job
