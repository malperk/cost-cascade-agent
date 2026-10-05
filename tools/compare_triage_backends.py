"""Compare the LLM triage with the Jev triage on the eval_fixtures suite, on the host.

Runs every fixture REPEATS times per backend (TRIAGE_BACKEND=llm vs auto) as a real
`python -m agent.main` subprocess with its own temp workspace, then reports per backend:
pass rate (same assertions as tools/eval_agent.py: answer_contains[_any] and expected.files),
runs that needed escalation past attempt 1, mean latency, and triage consistency (for each
fixture, did every repeat make the same split/steps/dependent/review decision?).

Needs the network (OmniRoute and, for "auto", Jev), unlike tools/eval_agent.py which runs
offline in a container - so this is a manual tool, not part of the self-improvement loop.

Usage: python -m tools.compare_triage_backends [--repeats 2] [--workers 3] [--filter fast_]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ["llm", "auto"]


def _field(stdout: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\s*(.*)$", stdout, re.MULTILINE)
    return match.group(1).strip() if match else ""


def run_one(fixture: dict, backend: str, timeout: int) -> dict:
    expected = fixture["expected"]
    workspace = Path(tempfile.mkdtemp(prefix=f"cmp-{fixture['id']}-"))
    env = {
        **os.environ,
        "AGENT_WORKSPACE_DIR": str(workspace),
        "TRIAGE_BACKEND": backend,
        "LANGSMITH_TRACING": "false",
    }
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "agent.main", fixture["prompt"]],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
        )
        stdout, exit_code = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, exit_code = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""), None
    latency = time.monotonic() - start

    answer = stdout.split("--- Output ---", 1)[1].strip() if "--- Output ---" in stdout else stdout
    ok = exit_code == expected.get("exit_code", 0)
    ok = ok and all(t.lower() in answer.lower() for t in expected.get("answer_contains", []))
    if expected.get("answer_contains_any"):
        ok = ok and any(t.lower() in answer.lower() for t in expected["answer_contains_any"])
    for spec in expected.get("files", []):
        path = workspace / spec["name"]
        ok = ok and path.is_file() and spec.get("contains", "") in path.read_text(errors="replace")
    shutil.rmtree(workspace, ignore_errors=True)

    split_line = _field(stdout, "split_mode:")
    attempts = _field(stdout, "attempts:")
    return {
        "id": fixture["id"], "backend": backend, "passed": ok, "latency": latency,
        "jev": _field(stdout, "triage:").startswith("Routed by Jev"),
        "decision": (_field(stdout, "split_needed:"), split_line, _field(stdout, "review_needed:")),
        "tier": _field(stdout, "tier (final):"),
        "escalated": attempts.isdigit() and int(attempts) > 1 or _field(stdout, "tier (final):") not in ("low", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--filter", default="")
    parser.add_argument("--json", help="write raw per-run results here")
    args = parser.parse_args()

    fixtures = [json.loads(p.read_text()) for p in sorted((ROOT / "eval_fixtures").glob("*.json"))]
    fixtures = [f for f in fixtures if args.filter in f["id"]]
    jobs = [(f, b) for f in fixtures for b in BACKENDS for _ in range(args.repeats)]
    print(f"{len(jobs)} runs ({len(fixtures)} fixtures x {len(BACKENDS)} backends x {args.repeats})", file=sys.stderr)

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda job: run_one(*job, args.timeout), jobs))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))

    for backend in BACKENDS:
        rs = [r for r in results if r["backend"] == backend]
        by_fixture = defaultdict(list)
        for r in rs:
            by_fixture[r["id"]].append(r["decision"])
        stable = sum(len(set(d)) == 1 for d in by_fixture.values())
        print(
            f"\n[{backend}] pass {sum(r['passed'] for r in rs)}/{len(rs)}"
            f" | escalated {sum(r['escalated'] for r in rs)}/{len(rs)}"
            f" | mean latency {sum(r['latency'] for r in rs) / len(rs):.0f}s"
            f" | triage-consistent fixtures {stable}/{len(by_fixture)}"
            + (f" | jev decided {sum(r['jev'] for r in rs)}/{len(rs)}" if backend == "auto" else "")
        )
    print("\nper fixture (llm -> auto): pass counts, decisions")
    for f in fixtures:
        row = []
        for backend in BACKENDS:
            rs = [r for r in results if r["id"] == f["id"] and r["backend"] == backend]
            row.append(f"{sum(r['passed'] for r in rs)}/{len(rs)} {sorted({str(r['decision']) for r in rs})}")
        print(f"  {f['id']}: {row[0]}  ->  {row[1]}")


if __name__ == "__main__":
    main()
