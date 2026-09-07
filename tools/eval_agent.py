"""Deterministic regression/benchmark suite for the self-improvement loop.

Each fixture in eval_fixtures/*.json is data, not vibes: an explicit assertion
set (`expected`) that a run either satisfies or doesn't. `run` executes a suite
against a given source tree (an exact `git archive` export - see the
self-improvement runner) inside a container with no network but a broker
socket, and writes structured per-case results. `compare` applies the policy
math from the self-improvement plan (section 8) to a baseline/candidate result
pair for one metric and exits 0/1.

Known scope limitation, stated rather than papered over: `required_tool` is
independently verified only for "file_write" (via the workspace mount) and
"web_search" (via the broker's own call log) - other tool names are recorded
from the fixture but not cross-checked against what actually happened, so a
case relying on one of those should also carry a strong `answer_contains`
assertion as its real evidence.
"""

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 120


def _load_fixtures(suite_dir: Path, pattern: str) -> list[dict]:
    fixtures = []
    for path in sorted(suite_dir.glob(pattern)):
        with open(path) as f:
            fixtures.append(json.load(f))
    return fixtures


def _tail_broker_log_calls(log_path: Path, since_line_count: int) -> int:
    """Count model+search calls the broker logged for this run: everything appended to the
    log since the run started, excluding denied (403/413/429) entries - those never reached
    the real model/search backend and shouldn't count as a call the run "made"."""
    if not log_path.exists():
        return 0
    lines = log_path.read_text().splitlines()
    new_lines = lines[since_line_count:]
    count = 0
    for line in new_lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == 200:
            count += 1
    return count


def _extract_answer(stdout: str) -> str:
    marker = "--- Output ---"
    if marker in stdout:
        return stdout.split(marker, 1)[1].strip()
    return stdout.strip()


def run_case(
    fixture: dict,
    image: str,
    source_dir: Path,
    broker_socket_path: Path | None,
    broker_log_path: Path | None,
    timeout_seconds: int,
    env_overrides: dict[str, str] | None = None,
    container_label: str | None = None,
) -> dict:
    case_id = fixture["id"]
    expected = fixture["expected"]
    workspace_dir = Path(f"/tmp/eval-workspace-{case_id}-{uuid.uuid4().hex[:8]}")
    workspace_dir.mkdir(parents=True, exist_ok=True)

    since_line_count = 0
    if broker_log_path and broker_log_path.exists():
        since_line_count = len(broker_log_path.read_text().splitlines())

    docker_cmd = [
        "docker", "run", "--rm", "--network", "none",
        "-v", f"{source_dir}:/app:ro",
        "-v", f"{workspace_dir}:/workspace:rw",
        "-e", "AGENT_WORKSPACE_DIR=/workspace",
        "-w", "/app",
    ]
    if container_label:
        docker_cmd += ["--label", f"self-improvement.cycle={container_label}"]
    if broker_socket_path:
        docker_cmd += [
            "-v", f"{broker_socket_path}:/run/broker.sock",
            "-e", "BROKER_SOCKET_PATH=/run/broker.sock",
        ]
    for key, value in (env_overrides or {}).items():
        docker_cmd += ["-e", f"{key}={value}"]
    docker_cmd += [image, "python3", "-m", "agent.main", fixture["prompt"]]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            docker_cmd, capture_output=True, text=True, timeout=timeout_seconds
        )
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        exit_code = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        timed_out = True
    latency_seconds = time.monotonic() - start

    answer = _extract_answer(stdout)
    answer_contains_ok = all(term.lower() in answer.lower() for term in expected.get("answer_contains", []))
    exit_code_ok = (not timed_out) and exit_code == expected.get("exit_code", 0)

    tool_check = {"expected": expected.get("required_tool"), "method": "not_checked", "verified": None}
    required_tool = expected.get("required_tool")
    if required_tool == "file_write":
        wrote_something = any(workspace_dir.iterdir())
        tool_check = {"expected": required_tool, "method": "workspace_inspection", "verified": wrote_something}
    elif required_tool == "web_search":
        calls = _tail_broker_log_calls(broker_log_path, since_line_count) if broker_log_path else None
        # a real check would grep for "/search" specifically; count > baseline model-call-only
        # noise is a coarse proxy here - good enough to catch "search wasn't attempted at all".
        searched = _grep_broker_search_calls(broker_log_path, since_line_count) if broker_log_path else None
        tool_check = {"expected": required_tool, "method": "broker_log", "verified": searched}
    elif required_tool is not None:
        tool_check = {"expected": required_tool, "method": "not_checked", "verified": None}

    call_count = _tail_broker_log_calls(broker_log_path, since_line_count) if broker_log_path else None

    passed = exit_code_ok and answer_contains_ok and tool_check.get("verified") is not False

    result = {
        "id": case_id,
        "passed": passed,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "answer_contains_ok": answer_contains_ok,
        "tool_check": tool_check,
        "latency_seconds": latency_seconds,
        "broker_call_count": call_count,
        "baseline_call_count": expected.get("baseline_call_count"),
        "answer_excerpt": answer[:300],
        "stderr_excerpt": stderr[-1000:] if stderr else "",
    }

    shutil.rmtree(workspace_dir, ignore_errors=True)
    return result


def _grep_broker_search_calls(log_path: Path, since_line_count: int) -> bool:
    if not log_path.exists():
        return False
    lines = log_path.read_text().splitlines()[since_line_count:]
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("path") == "/search" and record.get("status") == 200:
            return True
    return False


def run_suite(
    suite_dir: Path,
    pattern: str,
    image: str,
    source_dir: Path,
    broker_socket_path: Path | None,
    broker_log_path: Path | None,
    timeout_seconds: int,
    env_overrides: dict[str, str] | None = None,
    container_label: str | None = None,
) -> dict:
    fixtures = _load_fixtures(suite_dir, pattern)
    results = [
        run_case(
            f, image, source_dir, broker_socket_path, broker_log_path, timeout_seconds,
            env_overrides=env_overrides, container_label=container_label,
        )
        for f in fixtures
    ]
    passed = sum(1 for r in results if r["passed"])
    return {"cases": results, "passed": passed, "total": len(results)}


# --- Policy comparators (self-improvement plan section 8) ---


def paired_relative_improvement(baseline_values: list[float], candidate_values: list[float]) -> float:
    """median_i[(baseline_i - candidate_i) / baseline_i] - a lower-is-better metric's paired
    relative improvement. This is NOT the same statistic as (median(B)-median(C))/median(B);
    only this per-pair form actually uses the pairing."""
    ratios = [(b - c) / b for b, c in zip(baseline_values, candidate_values) if b != 0]
    return statistics.median(ratios) if ratios else 0.0


def retry_count_decision(baseline_counts: list[int], candidate_counts: list[int]) -> bool:
    """median_i(candidate_i - baseline_i) <= 0 AND sum(candidate) < sum(baseline) - strict:
    a tie means nothing actually improved and shouldn't be accepted."""
    deltas = [c - b for b, c in zip(baseline_counts, candidate_counts)]
    median_delta_ok = statistics.median(deltas) <= 0 if deltas else False
    strict_sum_ok = sum(candidate_counts) < sum(baseline_counts)
    return median_delta_ok and strict_sum_ok


def resolution_matched_fast_gate(baseline_results: list[dict], candidate_results: list[dict]) -> bool:
    """success_rate fast-suite gate: candidate passes >=1 case the baseline failed, and fails
    0 cases the baseline passed - meaningful at 5-case resolution where a percentage-point
    threshold isn't."""
    gained = lost = 0
    for b, c in zip(baseline_results, candidate_results):
        if not b["passed"] and c["passed"]:
            gained += 1
        if b["passed"] and not c["passed"]:
            lost += 1
    return gained >= 1 and lost == 0


def full_success_rate_gate(baseline_suite: dict, candidate_suite: dict, min_absolute_improvement: float) -> bool:
    baseline_rate = baseline_suite["passed"] / baseline_suite["total"]
    candidate_rate = candidate_suite["passed"] / candidate_suite["total"]
    return (candidate_rate - baseline_rate) >= min_absolute_improvement


def _cmd_run(args):
    result = run_suite(
        Path(args.suite_dir),
        args.pattern,
        args.image,
        Path(args.source_dir),
        Path(args.broker_socket) if args.broker_socket else None,
        Path(args.broker_log) if args.broker_log else None,
        args.timeout,
    )
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"{result['passed']}/{result['total']} passed -> {args.out}")
    return 0


def _cmd_compare(args):
    baseline = json.loads(Path(args.baseline).read_text())
    candidate = json.loads(Path(args.candidate).read_text())

    if args.metric == "success_rate":
        if args.mode == "fast":
            ok = resolution_matched_fast_gate(baseline["cases"], candidate["cases"])
        else:
            ok = full_success_rate_gate(baseline, candidate, args.min_score)
    elif args.metric in ("latency", "cost"):
        key = "latency_seconds" if args.metric == "latency" else "broker_call_count"
        b_vals = [c[key] for c in baseline["cases"] if c[key] is not None]
        c_vals = [c[key] for c in candidate["cases"] if c[key] is not None]
        score = paired_relative_improvement(b_vals, c_vals)
        ok = score >= args.min_score
        print(f"paired relative improvement: {score:.4f} (threshold {args.min_score})")
    elif args.metric == "retry_count":
        b_counts = [
            max(0, (c["broker_call_count"] or 0) - (c["baseline_call_count"] or 0))
            for c in baseline["cases"]
        ]
        c_counts = [
            max(0, (c["broker_call_count"] or 0) - (c["baseline_call_count"] or 0))
            for c in candidate["cases"]
        ]
        ok = retry_count_decision(b_counts, c_counts)
    else:
        print(f"unknown metric: {args.metric}", file=sys.stderr)
        return 2

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--suite-dir", default="eval_fixtures")
    run_p.add_argument("--pattern", default="fast_*.json")
    run_p.add_argument("--image", required=True)
    run_p.add_argument("--source-dir", required=True)
    run_p.add_argument("--broker-socket", default=None)
    run_p.add_argument("--broker-log", default=None)
    run_p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    run_p.add_argument("--out", required=True)
    run_p.set_defaults(func=_cmd_run)

    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("--baseline", required=True)
    cmp_p.add_argument("--candidate", required=True)
    cmp_p.add_argument("--metric", required=True, choices=["latency", "cost", "success_rate", "retry_count"])
    cmp_p.add_argument("--mode", choices=["fast", "full"], default="full")
    cmp_p.add_argument("--min-score", type=float, default=0.10)
    cmp_p.set_defaults(func=_cmd_compare)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
