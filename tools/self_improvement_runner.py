"""Trusted supervisor for one self-improvement cycle (plan v6.3).

This is the ONLY thing with git-merge / combo-promote / production-OmniRoute
authority. It invokes the LLM agent exactly once, as an AUTHORING step whose
entire output is either a committed candidate branch (CODE) or a proposal
file (MODEL) - the agent has no instructed or credentialed path to merging,
promoting, or writing history. Every accept/reject decision below is plain
Python, never an LLM call.

Known scope limitations of this initial implementation, stated rather than
silently glossed over:
  - Transport tier 1 only (see tools/si_transport.py) - proven to work on
    this machine by the tier-1 spike; tiers 2/3 are documented, unimplemented
    fallbacks in the plan.
  - The authoring container is always mounted read-write at /app, for both
    hypothesis types - the plan describes a read-only mount for MODEL as
    defense-in-depth, but since the agent itself chooses code-vs-model at
    runtime, the runner can't know which to mount before the container
    starts. The actual mechanical gate is unaffected: verify_model_invariants
    still hard-fails the cycle if a MODEL-declared hypothesis touched the
    repo at all, exactly as the plan's real security model requires (the
    gate is the runner's post-hoc check, not filesystem permission bits).
"""

import argparse
import fcntl
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import eval_agent, omniroute_admin, si_git
from tools.append_improvement_history import append as append_history
from tools.append_improvement_history import has_failed_fingerprint
from tools.si_journal import Journal
from tools.si_transport import recover_and_teardown_for_tier, setup_transport

REPO_PATH = Path(__file__).parent.parent
LOCK_PATH = REPO_PATH / ".self_improvement.lock"
RUNTIME_ROOT = REPO_PATH / ".self-improve-runtime"
CLONES_ROOT = REPO_PATH / ".self-improve-clones"
FIXTURES_DIR = REPO_PATH / "eval_fixtures"
FAST_PATTERN = "fast_*.json"
# "full" now genuinely differs from "fast": eval_fixtures/full_*.json adds harder,
# multi-step/reasoning/tool-sequencing cases (fast_*.json alone is easy enough that
# success_rate barely moves between candidates - see prompts/self_improvement_cycle.txt's
# note on why "code" hypotheses touching orchestration logic need real cases to show up on).
FULL_PATTERN = "*.json"

PRODUCTION_COMBO_NAME_FOR_TIER = {"low": "alper-low", "mid": "alper-mid", "high": "alper-high", "local": "alper-local"}
PRODUCTION_COMBOS = list(PRODUCTION_COMBO_NAME_FOR_TIER.values())
IMAGE_TAG_PREFIX = "self-improve"
AUTHORING_TIMEOUT_SECONDS = 20 * 60
# Live testing showed "low" tier unreliable for this long, meta, instruction-heavy prompt:
# one run truncated its hypothesis.json write, another disengaged from the task entirely
# after triage rated it low-risk. See agent/config.py's AGENT_MIN_TIER for the mechanism.
AUTHORING_MIN_TIER = "mid"
# "local" (Ollama llama3.1:8b) is too weak for this task's multi-step tool calling - a live
# cycle escalated there on a review failure and emitted a tool call as plain-text JSON instead
# of an actual invocation. Capping at "high" means a failed review repeats at "high" instead
# of dropping to a weaker model. See agent/config.py's AGENT_MAX_TIER for the mechanism.
AUTHORING_MAX_TIER = "high"
# A single un-escalated task already costs ~3 calls (triage + agent + review); the review
# pipeline can escalate up to MAX_ATTEMPTS times (mid -> high -> local), each attempt adding
# another agent+review pair, and infra-error retries multiply that further. A live cycle
# exhausted the old budget of 15 mid-task (the broker's own 429 forced agent.main to give up
# and return rejected without ever writing hypothesis.json) - raised to give real headroom for
# at least one full escalation chain plus some retry slack, while still being a real, bounded
# budget rather than unlimited.
AUTHORING_MODEL_CALL_BUDGET = 40
AUTHORING_SEARCH_CALL_BUDGET = 5

HYPOTHESIS_SCHEMA = {
    "type": "object",
    "required": ["category", "metric", "threshold"],
    "properties": {
        "category": {"enum": ["code", "model"]},
        "metric": {"enum": ["latency", "cost", "success_rate", "retry_count"]},
        "threshold": {"type": "number"},
        "guard": {"type": "object"},
        "tier": {"enum": ["low", "mid", "high", "local"]},
        "candidate_model": {"type": "string"},
        "candidate_provider": {"type": "string"},
    },
    "additionalProperties": False,
    "if": {"properties": {"category": {"const": "model"}}},
    "then": {"required": ["tier", "candidate_model", "candidate_provider"]},
    "else": {"not": {"anyOf": [{"required": ["tier"]}, {"required": ["candidate_model"]}, {"required": ["candidate_provider"]}]}},
}

POLICY = {
    # "auto_accept": no matter how high the agent's own proposed threshold was, a candidate
    # clearing this absolute bar is accepted regardless - a genuinely large real improvement
    # shouldn't be rejected just because the agent overpromised. Below this, the agent still
    # has to clear its own stated number (tightened against min_score, never loosened below
    # it) - see effective_bar().
    "latency": {"kind": "paired", "direction": "lower", "min_score": 0.10, "auto_accept": 0.40},
    "cost": {"kind": "paired", "direction": "lower", "min_score": 0.10, "auto_accept": 0.40},
    "retry_count": {"kind": "paired", "direction": "lower"},
    "success_rate": {"kind": "discrete", "direction": "higher", "min_absolute_improvement": 0.05},
}


class CycleFailed(Exception):
    """A cycle-ending FAIL that isn't a bug - the normal, expected outcome when a
    hypothesis doesn't hold up. Carries a human-readable reason for the report/ledger."""


# --- lock ---


class Lock:
    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def __enter__(self):
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.fh.close()
            raise SystemExit("another self-improvement cycle is already running (lock held)")
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


# --- startup recovery ---


def recover_stale_cycle() -> None:
    journal = Journal(RUNTIME_ROOT / "journal.json")
    if not journal.exists():
        return
    state = journal.load()
    cycle_id = state.get("cycle_id", "unknown")
    print(f"recovering stale cycle {cycle_id} (phase={state.get('phase')})")

    subprocess.run(
        ["docker", "rm", "-f", *_labeled_container_ids(cycle_id)],
        capture_output=True,
    )
    if state.get("transport"):
        recover_and_teardown_for_tier(state["transport"])

    if state.get("temp_combo"):
        omniroute_admin.delete_combo(state["temp_combo"])

    phase = state.get("phase")
    if phase in (None, "STARTED", "BASELINE_DONE", "CANDIDATE_RECEIVED", "CANDIDATE_VALIDATED", "EVAL_PASSED"):
        _finalize(state, result="fail", reason="cycle interrupted before promotion started", score=None)
    elif phase == "PROMOTION_STARTED":
        # ambiguous - re-check whether the promotion actually applied before deciding.
        if state.get("category") == "model":
            current = omniroute_admin.get_combo(state["production_combo_name"])
            intended = state.get("intended_production_after")
            if omniroute_admin.combos_equal(current, intended):
                _finalize(state, result="pass", reason="promotion had already applied before the crash", score=state.get("score"))
            else:
                omniroute_admin.promote_combo(
                    state["production_combo_name"],
                    state["production_before"][state["production_combo_name"]]["models"],
                )
                _finalize(state, result="fail", reason="promotion interrupted mid-flight, restored prior production combo", score=None)
        else:
            current_main = si_git.capture_base_sha(REPO_PATH)
            if current_main == state.get("candidate_sha"):
                _finalize(state, result="pass", reason="merge had already applied before the crash", score=state.get("score"))
            else:
                _finalize(state, result="fail", reason="promotion interrupted mid-flight, main was not advanced", score=None)
    elif phase == "PROMOTION_VERIFIED":
        # the change is real and already verified-good - never roll back a verified promotion,
        # just finish writing the metadata that's missing.
        _finalize(state, result="pass", reason="promotion was verified before the crash; completing metadata", score=state.get("score"))
    else:
        journal.delete()

    for path in RUNTIME_ROOT.glob("*"):
        if path.name != "journal.json":
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink()
    journal.delete()
    for clone_dir in CLONES_ROOT.glob("*"):
        shutil.rmtree(clone_dir, ignore_errors=True)


def _labeled_container_ids(cycle_id: str) -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label=self-improvement.cycle={cycle_id}"],
        capture_output=True, text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _finalize(state: dict, result: str, reason: str, score) -> None:
    cycle_id = state.get("cycle_id", "unknown")
    report_path = REPO_PATH / "reports" / f"{cycle_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# Self-improvement cycle {cycle_id} (recovered)\n\nResult: {result}\n\nReason: {reason}\n"
    )
    entry = {
        "cycle_id": cycle_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": state.get("category", "code"),
        "metric": state.get("metric", "latency"),
        "fingerprint": state.get("fingerprint", "unknown"),
        "result": result,
        "hypothesis": state.get("hypothesis", {}),
        "score": score,
        "effective_bar": state.get("effective_bar"),
        "candidate_sha": state.get("candidate_sha"),
        "report_path": str(report_path.relative_to(REPO_PATH)),
    }
    entry_file = RUNTIME_ROOT / "recovery_entry.json"
    entry_file.parent.mkdir(parents=True, exist_ok=True)
    entry_file.write_text(json.dumps(entry))
    append_history(entry)
    si_git.commit_metadata_on_main(
        REPO_PATH,
        [str(report_path.relative_to(REPO_PATH)), "state/improvement_history.json"],
        f"self-improvement cycle {cycle_id}: {result} (recovered)",
    )


# --- image + authoring container ---


def ensure_image(source_dir: Path, base_sha: str) -> str:
    tag = f"{IMAGE_TAG_PREFIX}:{base_sha[:12]}"
    exists = subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0
    if exists:
        return tag
    build_ctx = RUNTIME_ROOT / "image-build"
    build_ctx.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_dir / "requirements.txt", build_ctx / "requirements.txt")
    (build_ctx / "Dockerfile").write_text(
        "FROM python:3.11-slim\nWORKDIR /app\n"
        # git is required inside the authoring container for a CODE hypothesis's
        # `git -C /app add -A && git -C /app commit -m ...` step (base image has no git at
        # all - a live cycle's code_execution call failed with FileNotFoundError without this).
        "RUN apt-get update && apt-get install -y --no-install-recommends git "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "COPY requirements.txt .\n"
        "RUN pip install --no-cache-dir -r requirements.txt\n"
    )
    subprocess.run(["docker", "build", "-t", tag, str(build_ctx)], check=True, capture_output=True)
    return tag


def run_authoring_container(
    image: str, clone_dir: Path, cycle_output_dir: Path, socket_path: str, cycle_id: str, prompt: str
) -> tuple[int, str]:
    cycle_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm", "--network", "none",
        "--name", f"self-improve-author-{cycle_id}",
        "--label", f"self-improvement.cycle={cycle_id}",
        "-v", f"{clone_dir}:/app:rw",
        "-v", f"{cycle_output_dir}:/cycle-output:rw",
        "-v", f"{socket_path}:/run/broker.sock",
        "-e", "BROKER_SOCKET_PATH=/run/broker.sock",
        "-e", "AGENT_WORKSPACE_DIR=/cycle-output",
        "-e", f"AGENT_MIN_TIER={AUTHORING_MIN_TIER}",
        "-e", f"AGENT_MAX_TIER={AUTHORING_MAX_TIER}",
        "-w", "/app",
        image, "python3", "-m", "agent.main", prompt,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=AUTHORING_TIMEOUT_SECONDS)
    return proc.returncode, proc.stdout + proc.stderr


def _read_hypothesis_json(cycle_output_dir: Path, journal: Journal) -> dict:
    path = cycle_output_dir / "hypothesis.json"
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise CycleFailed("hypothesis.json is not a regular file")
        if st.st_size > 100_000:
            raise CycleFailed("hypothesis.json is implausibly large")
        raw = os.read(fd, st.st_size)
    finally:
        os.close(fd)
    # capture the raw bytes before attempting to parse, so a malformed write (e.g. a
    # truncated tool call from the authoring model) still leaves something diagnosable in
    # the journal/report instead of only "JSONDecodeError" with no visible content.
    journal.update(hypothesis_raw=raw.decode(errors="replace"))
    data = json.loads(raw)
    jsonschema.validate(data, HYPOTHESIS_SCHEMA)
    return data


def _build_history_context() -> str:
    history_path = REPO_PATH / "state" / "improvement_history.json"
    if not history_path.exists():
        return "(no prior cycles recorded)"
    history = json.loads(history_path.read_text())
    lines = [f"- {e['result']}: {e['category']}/{e['metric']} (fingerprint {e['fingerprint'][:12]}...)" for e in history[-10:]]
    return "\n".join(lines) or "(no prior cycles recorded)"


def _build_reports_context() -> str:
    reports_dir = REPO_PATH / "reports"
    if not reports_dir.exists():
        return "(no prior reports)"
    reports = sorted(reports_dir.glob("*.md"))[-2:]
    return "\n\n".join(f"### {r.name}\n{r.read_text()[:2000]}" for r in reports) or "(no prior reports)"


# --- policy ---


def effective_bar(hypothesis: dict) -> float | None:
    metric = hypothesis["metric"]
    policy = POLICY[metric]
    proposed = hypothesis.get("threshold")
    if metric == "retry_count":
        # its decision rule (see POLICY) is a boolean strict-improvement check, not a numeric
        # bar - fixed a latent bug here: retry_count's "kind" is also "paired," so without this
        # early return the branch below would KeyError on policy["min_score"], which
        # retry_count's policy entry deliberately doesn't have.
        return None
    if policy["kind"] == "paired":
        floor = policy["min_score"]
        bar = max(proposed, floor) if proposed is not None else floor
        auto_accept = policy.get("auto_accept")
        if auto_accept is not None:
            # never require MORE than auto_accept, no matter how high the agent's own
            # proposed threshold was - a candidate this good is accepted unconditionally.
            bar = min(bar, auto_accept)
        return bar
    if metric == "success_rate":
        floor = policy["min_absolute_improvement"]
        return max(proposed, floor) if proposed is not None else floor
    return None


def check_guard(hypothesis: dict, baseline_cases: list[dict], candidate_cases: list[dict]) -> None:
    """Enforces hypothesis.json's declared guard - previously a schema field the agent wrote
    but nothing ever read. A live cycle showed exactly why this matters: a candidate combo
    with zero working providers fails every call near-instantly, which makes "latency"/"cost"
    (proxied via call count) look great purely because it's broken, not because it's fast. The
    full-suite-must-be-100%-passing regression gate happened to catch that specific case, but
    the guard is the one actually named by the hypothesis and should fire on its own, earlier,
    with a clear reason - not rely on the regression gate as an accidental backstop."""
    guard = hypothesis.get("guard")
    if not guard or guard.get("metric") != "success_rate":
        return  # only a success_rate guard is enforced; anything else is a documented no-op
    max_regression = guard.get("max_regression", 0.0)
    baseline_rate = sum(1 for c in baseline_cases if c["passed"]) / len(baseline_cases) if baseline_cases else 1.0
    candidate_rate = sum(1 for c in candidate_cases if c["passed"]) / len(candidate_cases) if candidate_cases else 0.0
    if (baseline_rate - candidate_rate) > max_regression:
        raise CycleFailed(
            f"guard violated: success_rate regressed from {baseline_rate:.2f} to {candidate_rate:.2f} "
            f"(max allowed regression {max_regression})"
        )


def evaluate_metric(metric: str, baseline_suite: dict, candidate_suite: dict, mode: str, bar: float | None) -> tuple[bool, float | None]:
    if metric == "success_rate":
        if mode == "fast":
            ok = eval_agent.resolution_matched_fast_gate(baseline_suite["cases"], candidate_suite["cases"])
            return ok, None
        ok = eval_agent.full_success_rate_gate(baseline_suite, candidate_suite, bar)
        return ok, None
    if metric in ("latency", "cost"):
        # known simplification: "cost" is currently measured via broker_call_count (a count),
        # not the summed real $ cost the broker's log already has per call via
        # x-omniroute-response-cost - close enough to prove the paired-comparison mechanism
        # end to end, but a real cost-metric hypothesis deserves float-cost aggregation from
        # the broker log before being trusted for an actual promotion decision.
        key = "latency_seconds" if metric == "latency" else "broker_call_count"
        b_vals = [c[key] for c in baseline_suite["cases"] if c[key] is not None]
        c_vals = [c[key] for c in candidate_suite["cases"] if c[key] is not None]
        score = eval_agent.paired_relative_improvement(b_vals, c_vals)
        return score >= bar, score
    if metric == "retry_count":
        b_counts = [max(0, (c["broker_call_count"] or 0) - (c["baseline_call_count"] or 0)) for c in baseline_suite["cases"]]
        c_counts = [max(0, (c["broker_call_count"] or 0) - (c["baseline_call_count"] or 0)) for c in candidate_suite["cases"]]
        return eval_agent.retry_count_decision(b_counts, c_counts), None
    raise ValueError(f"unknown metric {metric}")


# --- preflight ---


def check_preflight() -> str | None:
    """Checked before anything else touches Docker or OmniRoute (both external to this
    process and not guaranteed to be running, e.g. for an unattended hourly schedule) - returns
    None if ready, or a human-readable reason if not. Without this, an environment that isn't
    up yet crashes with an unhandled exception before the try/except block that normally turns
    a failure into a report+ledger entry even exists - it only gets swept up as a generic,
    reason-less "(recovered)" fail on the next run instead of recording what actually happened."""
    try:
        docker_check = subprocess.run(["docker", "info"], capture_output=True, timeout=15)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return f"PREFLIGHT FAILED: could not run 'docker info' ({type(exc).__name__}: {exc})"
    if docker_check.returncode != 0:
        return "PREFLIGHT FAILED: Docker is not running ('docker info' returned non-zero - is Docker Desktop started?)"

    try:
        omniroute_admin.list_combos()
    except Exception as exc:
        return f"PREFLIGHT FAILED: OmniRoute is not reachable ({type(exc).__name__}: {exc})"

    return None


# --- main cycle ---


def run_cycle() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    CLONES_ROOT.mkdir(parents=True, exist_ok=True)

    with Lock(LOCK_PATH):
        recover_stale_cycle()

        cycle_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        runtime_dir = RUNTIME_ROOT / cycle_id
        clone_dir = CLONES_ROOT / cycle_id
        cycle_output_dir = runtime_dir / "cycle-output"
        runtime_dir.mkdir(parents=True, exist_ok=True)

        journal = Journal(RUNTIME_ROOT / "journal.json")
        journal.update(cycle_id=cycle_id, phase="STARTED")
        print(f"=== self-improvement cycle {cycle_id} ===")

        preflight_error = check_preflight()
        if preflight_error:
            print(preflight_error)
            _write_fail(journal, cycle_id, runtime_dir, reason=preflight_error)
            journal.delete()
            shutil.rmtree(runtime_dir, ignore_errors=True)
            return

        base_sha = si_git.capture_base_sha(REPO_PATH)
        production_before = omniroute_admin.snapshot_all(PRODUCTION_COMBOS)
        journal.update(base_sha=base_sha, production_before=production_before)

        transport = setup_transport(cycle_id, runtime_dir)
        try:
            _run_cycle_body(cycle_id, runtime_dir, clone_dir, cycle_output_dir, base_sha, production_before, journal, transport)
        except CycleFailed as exc:
            _write_fail(journal, cycle_id, runtime_dir, reason=str(exc))
        except si_git.InvariantViolation as exc:
            # the mechanical gate itself firing - a hard FAIL, not a bug, and it must be
            # recorded exactly like any other rejection, not left to crash the process.
            _write_fail(journal, cycle_id, runtime_dir, reason=f"INVARIANT VIOLATION: {exc}")
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            jsonschema.ValidationError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            _write_fail(journal, cycle_id, runtime_dir, reason=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - last-resort net: still record something rather
            # than leaving the journal for next-run crash-recovery to sort out unnecessarily.
            _write_fail(journal, cycle_id, runtime_dir, reason=f"unexpected error: {type(exc).__name__}: {exc}")
        finally:
            subprocess.run(["docker", "rm", "-f", *_labeled_container_ids(cycle_id)], capture_output=True)
            transport.teardown()
            shutil.rmtree(clone_dir, ignore_errors=True)
            journal.delete()


def _run_cycle_body(cycle_id, runtime_dir, clone_dir, cycle_output_dir, base_sha, production_before, journal, transport):
    source_export = runtime_dir / "export-base"
    si_git.git_archive_export(REPO_PATH, base_sha, source_export)
    image = ensure_image(source_export, base_sha)

    budgets = {
        "authoring": {"model_calls": AUTHORING_MODEL_CALL_BUDGET, "search_calls": AUTHORING_SEARCH_CALL_BUDGET},
        "baseline": {"model_calls": 200, "search_calls": 10},
        "candidate": {"model_calls": 200, "search_calls": 10},
    }
    transport.start_broker(budgets)
    channels = transport.prepare_channels()
    journal.update(transport=transport.recovery_state())

    print("running pre-authoring baseline (fast suite, functional reference)...")
    baseline_fast = eval_agent.run_suite(
        FIXTURES_DIR, FAST_PATTERN, image, source_export,
        Path(channels["baseline"]), runtime_dir / "broker.log", eval_agent.DEFAULT_TIMEOUT_SECONDS,
    )
    (runtime_dir / "baseline_fast.json").write_text(json.dumps(baseline_fast, indent=2))
    journal.set_phase("BASELINE_DONE")

    si_git.shallow_clone(REPO_PATH, clone_dir)
    prompt = (REPO_PATH / "prompts" / "self_improvement_cycle.txt").read_text().format(
        history_context=_build_history_context(), reports_context=_build_reports_context()
    )
    print("running authoring container...")
    exit_code, log = run_authoring_container(image, clone_dir, cycle_output_dir, channels["authoring"], cycle_id, prompt)
    (runtime_dir / "authoring.log").write_text(log)
    print(f"authoring container exited {exit_code} (log: {runtime_dir}/authoring.log)")
    journal.set_phase("CANDIDATE_RECEIVED")

    production_after = omniroute_admin.snapshot_all(list(production_before.keys()))
    for name in production_before:
        if not omniroute_admin.combos_equal(production_before[name], production_after[name]):
            raise CycleFailed(f"INTEGRITY FAIL: production combo {name} changed during authoring")

    hypothesis = _read_hypothesis_json(cycle_output_dir, journal)
    journal.update(hypothesis=hypothesis, category=hypothesis["category"], metric=hypothesis["metric"])
    print(f"hypothesis: {hypothesis}")

    if hypothesis["category"] == "code":
        si_git.verify_code_invariants(clone_dir, base_sha)
        branch_name = f"self-improve/{cycle_id}"
        candidate_sha = si_git.fetch_candidate_branch(REPO_PATH, clone_dir, branch_name)
        si_git.check_whitelist_and_budget(REPO_PATH, base_sha, candidate_sha)
        fingerprint = si_git.code_fingerprint(REPO_PATH, base_sha, candidate_sha, hypothesis["metric"])
        candidate_export = runtime_dir / "export-candidate"
        si_git.git_archive_export(REPO_PATH, candidate_sha, candidate_export)
    else:
        si_git.verify_model_invariants(clone_dir, base_sha)
        branch_name = None
        candidate_sha = None
        fingerprint = si_git.model_fingerprint(
            hypothesis["tier"], hypothesis["metric"], hypothesis["candidate_model"], hypothesis["candidate_provider"]
        )
        candidate_export = source_export  # code is identical; only the temp combo differs

    journal.update(fingerprint=fingerprint, candidate_sha=candidate_sha)
    if has_failed_fingerprint(fingerprint):
        raise CycleFailed(f"REJECT_DUPLICATE: fingerprint {fingerprint[:12]}... already recorded as fail")
    journal.set_phase("CANDIDATE_VALIDATED")

    temp_combo = None
    if hypothesis["category"] == "model":
        temp_combo = f"candidate-{hypothesis['tier']}-{cycle_id}"
        models = [{"kind": "model", "model": hypothesis["candidate_model"], "providerId": hypothesis["candidate_provider"], "weight": 0}]
        omniroute_admin.create_combo(temp_combo, models)
        journal.update(temp_combo=temp_combo)

    bar = effective_bar(hypothesis)
    journal.update(effective_bar=bar)

    def eval_with(socket_key: str, role_export: Path, tier_override: str | None, pattern: str = FAST_PATTERN) -> dict:
        # tier_override routes that one tier through the temp combo instead of the container's
        # built-in default (which matches the real production combo name) - only ever set for
        # a MODEL hypothesis's candidate-side eval, never for baseline.
        env_overrides = {f"{tier_override.upper()}_MODEL": temp_combo} if tier_override else {}
        return eval_agent.run_suite(
            FIXTURES_DIR, pattern, image, role_export,
            Path(channels[socket_key]), runtime_dir / "broker.log", eval_agent.DEFAULT_TIMEOUT_SECONDS,
            env_overrides=env_overrides, container_label=cycle_id,
        )

    # Note: no try/finally is needed around this eval block for temp-combo cleanup - a
    # CycleFailed raised here propagates up to run_cycle()'s handler, which calls
    # _write_fail(), which already reads journal.data["temp_combo"] (set above, before eval
    # starts) and deletes it. Promotion is the only path that needs its own cleanup logic,
    # since by then the temp combo may already have been consumed by a successful promotion.
    metric = hypothesis["metric"]
    candidate_tier = hypothesis.get("tier") if hypothesis["category"] == "model" else None

    candidate_fast = eval_with("candidate", candidate_export, candidate_tier)
    (runtime_dir / "candidate_fast.json").write_text(json.dumps(candidate_fast, indent=2))

    if metric == "success_rate":
        ok, score = evaluate_metric(metric, baseline_fast, candidate_fast, "fast", bar)
        journal.update(score=score)
        if not ok:
            raise CycleFailed("success_rate fast gate did not pass")
        baseline_full = eval_with("baseline", source_export, None, pattern=FULL_PATTERN)
        candidate_full = eval_with("candidate", candidate_export, candidate_tier, pattern=FULL_PATTERN)
        ok, score = evaluate_metric(metric, baseline_full, candidate_full, "full", bar)
        journal.update(score=score)
        if not ok:
            raise CycleFailed("success_rate full baseline-vs-candidate gate did not pass")
    else:
        reps = 3
        b_paired, c_paired = [], []
        for i in range(reps):
            order = ["B", "C"] if i % 2 == 0 else ["C", "B"]
            for side in order:
                if side == "B":
                    b_paired.append(eval_with("baseline", source_export, None))
                else:
                    c_paired.append(eval_with("candidate", candidate_export, candidate_tier))
        baseline_paired = {"cases": [c for suite in b_paired for c in suite["cases"]]}
        candidate_paired = {"cases": [c for suite in c_paired for c in suite["cases"]]}
        ok, score = evaluate_metric(metric, baseline_paired, candidate_paired, "full", bar)
        journal.update(score=score)  # recorded before the raise, so a FAIL's ledger entry
        # still shows the objective measured score, not just the free-text failure reason.
        check_guard(hypothesis, baseline_paired["cases"], candidate_paired["cases"])
        if not ok:
            raise CycleFailed(f"{metric} paired benchmark did not pass (score={score}, bar={bar})")

        print("running full regression suite (candidate-only)...")
        full_candidate = eval_with("candidate", candidate_export, candidate_tier, pattern=FULL_PATTERN)
        if full_candidate["passed"] != full_candidate["total"]:
            raise CycleFailed("full regression suite failed on candidate")

    # score is already in the journal (recorded right after each evaluate_metric() call above,
    # before any raise) - nothing left to update here, just advance the phase.
    journal.set_phase("EVAL_PASSED")

    journal.set_phase("PROMOTION_STARTED")
    if hypothesis["category"] == "code":
        new_head = si_git.merge_candidate_ff_only(REPO_PATH, branch_name, base_sha)
        journal.update(candidate_sha=new_head)
    else:
        current = omniroute_admin.get_combo(PRODUCTION_COMBO_NAME_FOR_TIER[hypothesis["tier"]])
        expected = production_before[PRODUCTION_COMBO_NAME_FOR_TIER[hypothesis["tier"]]]
        if not omniroute_admin.combos_equal(current, expected):
            raise CycleFailed("production combo drifted since cycle start - refusing to promote")
        models = [{"kind": "model", "model": hypothesis["candidate_model"], "providerId": hypothesis["candidate_provider"], "weight": 0}]
        promoted = omniroute_admin.promote_combo(PRODUCTION_COMBO_NAME_FOR_TIER[hypothesis["tier"]], models)
        journal.update(intended_production_after=promoted, production_combo_name=PRODUCTION_COMBO_NAME_FOR_TIER[hypothesis["tier"]])
        live = omniroute_admin.get_combo(PRODUCTION_COMBO_NAME_FOR_TIER[hypothesis["tier"]])
        if not omniroute_admin.combos_equal(live, promoted):
            omniroute_admin.promote_combo(PRODUCTION_COMBO_NAME_FOR_TIER[hypothesis["tier"]], expected["models"])
            raise CycleFailed("promotion verification failed - restored prior production combo")
        # Definitional match isn't enough - a live cycle promoted alper-local to a model that
        # wasn't actually pulled in the local Ollama installation; every eval call happened to
        # work at the time but the combo was 404-broken minutes later. A real inference call
        # against the promoted production name is the only thing that actually confirms it works.
        smoke_ok, smoke_detail = omniroute_admin.smoke_test_combo(PRODUCTION_COMBO_NAME_FOR_TIER[hypothesis["tier"]])
        if not smoke_ok:
            omniroute_admin.promote_combo(PRODUCTION_COMBO_NAME_FOR_TIER[hypothesis["tier"]], expected["models"])
            raise CycleFailed(f"post-promotion smoke test failed - restored prior production combo ({smoke_detail})")
        omniroute_admin.delete_combo(temp_combo)
    journal.set_phase("PROMOTION_VERIFIED")

    _write_pass(journal, cycle_id, runtime_dir)


def _write_report(cycle_id: str, journal: Journal, result: str, extra: str = "") -> Path:
    report_path = REPO_PATH / "reports" / f"{cycle_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data = journal.data
    hypothesis_block = json.dumps(data.get("hypothesis", {}), indent=2)
    if not data.get("hypothesis") and data.get("hypothesis_raw"):
        hypothesis_block += f"\n\n(raw, unparsed content the authoring container wrote):\n{data['hypothesis_raw']}"
    report_path.write_text(
        f"# Self-improvement cycle {cycle_id}\n\n"
        f"Result: {result}\n\n"
        f"Hypothesis: {hypothesis_block}\n\n"
        f"Score: {data.get('score')}\nEffective bar: {data.get('effective_bar')}\n\n"
        f"{extra}\n"
    )
    return report_path


def _write_pass(journal: Journal, cycle_id: str, runtime_dir: Path) -> None:
    report_path = _write_report(cycle_id, journal, "pass")
    entry = {
        "cycle_id": cycle_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": journal.data["category"],
        "metric": journal.data["metric"],
        "fingerprint": journal.data["fingerprint"],
        "result": "pass",
        "hypothesis": journal.data.get("hypothesis", {}),
        "score": journal.data.get("score"),
        "effective_bar": journal.data.get("effective_bar"),
        "candidate_sha": journal.data.get("candidate_sha"),
        "report_path": str(report_path.relative_to(REPO_PATH)),
    }
    append_history(entry)
    si_git.commit_metadata_on_main(
        REPO_PATH, [str(report_path.relative_to(REPO_PATH)), "state/improvement_history.json"],
        f"self-improvement cycle {cycle_id}: pass",
    )
    journal.set_phase("METADATA_COMMITTED")
    print(f"CYCLE PASSED: {cycle_id}")


def _write_fail(journal: Journal, cycle_id: str, runtime_dir: Path, reason: str) -> None:
    print(f"CYCLE FAILED: {reason}")
    temp_combo = journal.data.get("temp_combo")
    if temp_combo:
        omniroute_admin.delete_combo(temp_combo)
    report_path = _write_report(cycle_id, journal, "fail", extra=f"Reason: {reason}")
    entry = {
        "cycle_id": cycle_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": journal.data.get("category", "code"),
        "metric": journal.data.get("metric", "latency"),
        "fingerprint": journal.data.get("fingerprint", "unknown"),
        "result": "fail",
        "hypothesis": journal.data.get("hypothesis", {}),
        "score": journal.data.get("score"),
        "effective_bar": journal.data.get("effective_bar"),
        "candidate_sha": journal.data.get("candidate_sha"),
        "report_path": str(report_path.relative_to(REPO_PATH)),
    }
    append_history(entry)
    si_git.commit_metadata_on_main(
        REPO_PATH, [str(report_path.relative_to(REPO_PATH)), "state/improvement_history.json"],
        f"self-improvement cycle {cycle_id}: fail",
    )


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run_cycle()
