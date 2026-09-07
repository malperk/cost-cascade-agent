"""Test every model OmniRoute exposes and report which ones actually work.

For each model in GET /v1/models this sends a plain completion request and,
if that succeeds, a second request with a tool bound to see whether the
model actually emits a tool_call. Results are written to model_report.json
(raw data) and model_report.md (human-readable table).

Usage:
    python -m tools.model_report [--workers N] [--timeout SECONDS] [--filter SUBSTRING]
"""

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:20128/v1"
API_KEY = os.getenv("OMNIROUTE_API_KEY", "omniroute-local")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a numeric arithmetic expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


def fetch_models(filter_substring: str | None) -> list[dict]:
    resp = requests.get(f"{BASE_URL}/models", headers=HEADERS, timeout=15)
    resp.raise_for_status()
    models = resp.json()["data"]
    if filter_substring:
        models = [m for m in models if filter_substring.lower() in m["id"].lower()]
    return models


def _post(payload: dict, timeout: int) -> tuple[int | None, dict | None, str | None, bool, dict]:
    """Returns (status_code, json_body, error_str, timed_out, response_headers)."""
    try:
        resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, headers=HEADERS, timeout=timeout)
        try:
            return resp.status_code, resp.json(), None, False, resp.headers
        except ValueError:
            return resp.status_code, None, f"non-JSON response: {resp.text[:150]}", False, resp.headers
    except requests.exceptions.Timeout:
        return None, None, f"timed out after {timeout}s", True, {}
    except Exception as exc:
        return None, None, str(exc)[:200], False, {}


def test_model(model_meta: dict, timeout: int) -> dict:
    model_id = model_meta["id"]
    caps = model_meta.get("capabilities") or {}
    result = {
        "id": model_id,
        "declared_context_length": model_meta.get("context_length"),
        "declared_tool_calling": caps.get("tool_calling"),
        "declared_reasoning": caps.get("reasoning"),
        "works": False,
        "timed_out": False,
        "latency_ms": None,
        "error": None,
        "tool_call_works": False,
        "tool_call_error": None,
        "real_provider": None,  # x-omniroute-provider: which upstream actually served the request
        "response_cost": None,  # x-omniroute-response-cost: actual $ cost of this request (0.0 = genuinely free)
        "is_free": None,
    }

    start = time.monotonic()
    status, data, error, timed_out, headers = _post(
        {
            "model": model_id,
            "stream": False,
            "messages": [{"role": "user", "content": "Reply with exactly one word: pong"}],
        },
        timeout,
    )
    result["latency_ms"] = round((time.monotonic() - start) * 1000)

    if timed_out:
        result["timed_out"] = True
        result["error"] = error
        return result
    if status != 200 or data is None or "choices" not in data:
        result["error"] = error or str((data or {}).get("error", data))[:200]
        return result
    result["works"] = True
    result["real_provider"] = headers.get("x-omniroute-provider")
    cost_header = headers.get("x-omniroute-response-cost")
    if cost_header is not None:
        try:
            result["response_cost"] = float(cost_header)
            result["is_free"] = result["response_cost"] == 0.0
        except ValueError:
            pass

    # only worth checking tool-calling if the model can respond at all
    status, data, error, timed_out, _headers = _post(
        {
            "model": model_id,
            "stream": False,
            "messages": [
                {"role": "user", "content": "What is 45 multiplied by 12? Use the calculator tool to compute it."}
            ],
            "tools": [CALCULATOR_TOOL],
        },
        timeout,
    )
    if timed_out:
        result["tool_call_error"] = error
    elif status != 200 or data is None:
        result["tool_call_error"] = error or str((data or {}).get("error", data))[:200]
    else:
        tool_calls = data["choices"][0]["message"].get("tool_calls") or []
        result["tool_call_works"] = len(tool_calls) > 0

    return result


def write_reports(results: list[dict], out_dir: Path) -> None:
    results = sorted(results, key=lambda r: r["id"])

    (out_dir / "model_report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    working = [r for r in results if r["works"]]
    tool_capable = [r for r in working if r["tool_call_works"]]
    timed_out = [r for r in results if r["timed_out"]]
    confirmed_free = [r for r in working if r["is_free"]]

    lines = [
        "# OmniRoute Model Report",
        "",
        f"Tested {len(results)} models.",
        f"- Working: {len(working)}/{len(results)}",
        f"- Timed out: {len(timed_out)}/{len(results)}",
        f"- Tool-calling capable (of working models): {len(tool_capable)}/{len(working)}",
        f"- Confirmed free (actual $0 response cost, of working models): {len(confirmed_free)}/{len(working)}",
        "",
        "| Model | Real Provider | Status | Latency (ms) | Context | Tool-calling | Cost | Error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        status = "OK" if r["works"] else ("TIMEOUT" if r["timed_out"] else "FAIL")
        tool = "yes" if r["tool_call_works"] else ("-" if not r["works"] else "no")
        cost = "-"
        if r["response_cost"] is not None:
            cost = "FREE" if r["is_free"] else f"${r['response_cost']:.6f}"
        err = (r.get("error") or r.get("tool_call_error") or "").replace("\n", " ")[:80]
        lines.append(
            f"| {r['id']} | {r.get('real_provider') or '-'} | {status} | {r['latency_ms'] or '-'} | {r['declared_context_length'] or '-'} | {tool} | {cost} | {err} |"
        )

    (out_dir / "model_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--filter", type=str, default=None, help="only test models whose id contains this substring")
    parser.add_argument("--limit", type=int, default=None, help="only test the first N matching models")
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="seconds to stagger between submitting each request - avoids bursting a single "
        "provider hard enough to trip its own rate limit while we're just trying to test it",
    )
    args = parser.parse_args()

    models = fetch_models(args.filter)
    if args.limit:
        models = models[: args.limit]
    print(f"Testing {len(models)} models with {args.workers} workers, {args.timeout}s timeout each, {args.delay}s stagger...\n")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for m in models:
            futures[executor.submit(test_model, m, args.timeout)] = m["id"]
            time.sleep(args.delay)
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            r = future.result()
            results.append(r)
            status = "OK" if r["works"] else ("TIMEOUT" if r["timed_out"] else "FAIL")
            tool_status = "yes" if r["tool_call_works"] else ("-" if not r["works"] else "no")
            print(f"[{i:>3}/{len(models)}] {r['id']:55s} {status:8s} tool={tool_status}")

    out_dir = Path(__file__).parent
    write_reports(results, out_dir)
    print(f"\nReport written to {out_dir / 'model_report.json'} and {out_dir / 'model_report.md'}")


if __name__ == "__main__":
    main()
