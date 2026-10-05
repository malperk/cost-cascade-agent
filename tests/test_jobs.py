import time

from agent import notify
from agent.jobs import JobStore


def _wait(store, job_id, status, timeout=3):
    end = time.time() + timeout
    while time.time() < end:
        if store.get(job_id)["status"] == status:
            return store.get(job_id)
        time.sleep(0.01)
    raise AssertionError(f"job never reached {status}: {store.get(job_id)}")


def test_job_runs_and_hooks_fire(tmp_path):
    seen = []
    store = JobStore(lambda t: {"final_output": t.upper(), "accepted": True}, on_complete=[seen.append], directory=tmp_path)
    job = store.submit("hi")
    done = _wait(store, job["id"], "done")
    assert done["result"]["final_output"] == "HI"
    _wait(store, job["id"], "done")
    assert seen and seen[0]["id"] == job["id"]


def test_runner_exception_marks_failed_and_worker_survives(tmp_path):
    def runner(t):
        if t == "boom":
            raise RuntimeError("bad")
        return {"final_output": "ok"}

    store = JobStore(runner, directory=tmp_path)
    bad = store.submit("boom")
    good = store.submit("fine")
    assert "RuntimeError: bad" in _wait(store, bad["id"], "failed")["error"]
    _wait(store, good["id"], "done")


def test_broken_hook_does_not_block_other_hooks(tmp_path):
    seen = []

    def bad(job):
        raise OSError("smtp down")

    store = JobStore(lambda t: {}, on_complete=[bad, seen.append], directory=tmp_path)
    job = store.submit("x")
    _wait(store, job["id"], "done")
    time.sleep(0.1)
    assert seen


def test_restart_marks_unfinished_jobs_interrupted(tmp_path):
    store = JobStore(lambda t: time.sleep(5) or {}, directory=tmp_path)
    job = store.submit("slow")
    _wait(store, job["id"], "running")
    assert JobStore(lambda t: {}, directory=tmp_path).get(job["id"])["status"] == "interrupted"


def test_email_only_when_notify_and_configured(monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(notify, "NOTIFY_EMAIL_TO", "me@test")
    monkeypatch.setattr(notify, "SMTP_FROM", "bot@test")
    monkeypatch.setattr(notify.smtplib, "SMTP", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not connect")))
    job = {"id": "1", "task": "t", "notify": False, "status": "done", "result": {}}
    notify.send_email(job)  # notify=False -> no connection attempted
    msg = notify.build_message({**job, "notify": True, "result": {"final_output": "42", "tier": "low", "attempt_count": 1}})
    assert msg["To"] == "me@test" and "42" in msg.get_content()
    assert msg["Subject"].startswith("[agent] task done")
