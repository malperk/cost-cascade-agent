import sys

from agent.graph import app
from agent.model_catalog import tier_warnings


def run(task: str) -> bool:
    """Returns True if the task was accepted, False if it was rejected (failed review at every tier)."""
    for warning in tier_warnings():
        print(f"[model catalog] {warning}", file=sys.stderr)
    result = app.invoke({"task": task, "depth": 0})
    rejected = result.get("review_passed") is False and "TASK FAILED" in result.get("final_output", "")
    print("\n--- Trace ---")
    print(f"split_needed:  {result.get('split_needed')}")
    print(f"split_mode:    {result.get('split_mode')} (steps={result.get('distinct_steps')}, dependent={result.get('steps_are_dependent')})")
    print(f"triage:        {result.get('triage_reasoning')}")
    print(f"risk_level:    {result.get('risk_level')}")
    print(f"review_needed: {result.get('review_needed')}")
    print(f"tier (final):  {result.get('tier')}")
    print(f"attempts:      {result.get('attempt_count')}")
    if result.get("review_passed") is not None:
        print(f"review_passed: {result.get('review_passed')}")
    if result.get("infra_fallback"):
        print(f"infra_fallback: True (lower tier's provider pool was unavailable, fell back to {result.get('tier')})")
    print("\n--- Output ---")
    print(result.get("final_output", "<no output>"))
    return not rejected


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "What is 17 * 24?"
    accepted = run(task)
    sys.exit(0 if accepted else 1)
