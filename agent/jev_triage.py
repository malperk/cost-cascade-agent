"""Triage via TypeSafe's Jev (https://docs.typesafe.ai), a hosted decision model.

The LLM triage is a free-form structured-output call to a cheap model, and its answers drift
between runs on the same task. Jev answers fixed typed questions (choice / noul) in one call
and returns probabilities, so the same task gets the same routing - the "code keeps the
structure, a model sits only at the branches" pattern from
https://www.langchain.com/blog/building-prod-with-jev-and-langgraph.

`jev_triage` returns the TriageDecision fields, or None whenever the LLM triage should decide
instead (backend off, no key, long/meta task, API failure, or any answer below JEV_CONFIDENCE).
It never raises: Jev is an accelerator for the normal path, not a dependency.
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from agent.config import (
    AGENT_MIN_TIER,
    JEV_BASE_URL,
    JEV_CONFIDENCE,
    JEV_MAX_TASK_CHARS,
    JEV_MODEL,
    JEV_TIMEOUT,
    TRIAGE_BACKEND,
    TYPESAFE_API_KEY,
)

log = logging.getLogger(__name__)

Transport = Callable[[dict], dict]

# Wording mirrors the field descriptions in agent/nodes.py:TriageDecision.
QUESTIONS: dict[str, dict[str, Any]] = {
    "split_needed": {
        "type": "noul",
        "instructions": "Is this task genuinely complex enough to benefit from being decomposed "
        "into several separate deliverables, rather than done in one go?",
    },
    "review_needed": {
        "type": "noul",
        "instructions": "Is there meaningful risk that a cheap model's answer is wrong and needs "
        "checking (ambiguity, precision-sensitive output, multi-step reasoning, factual risk), "
        "or does the task require taking an action with a tool such as creating or editing "
        "files, running code, searching the web or calling an API?",
    },
    "steps_are_dependent": {
        "type": "noul",
        "instructions": "Do later steps of this task need the results or side effects of earlier "
        "steps (for example edit a file, THEN commit that edit), so they must run in order?",
    },
    "risk_level": {
        "type": "choice",
        "instructions": "How likely is a cheap, fast model to get this task wrong?",
        "criteria": {
            "low": "a simple, well-specified question or a single small action",
            "medium": "some ambiguity or a few steps where a mistake is plausible",
            "high": "precision-sensitive, multi-step, or easy to get subtly wrong",
        },
    },
    "distinct_steps": {
        "type": "choice",
        "instructions": "How many separate ACTIONS does the task ask for with tools or as distinct "
        "deliverables (writing a file, running a command, searching, committing...), counted "
        "literally? Stages of reasoning or arithmetic inside one question are not separate steps.",
        "criteria": {
            "one": "zero or one action, or a question answered directly",
            "two": "exactly two separate actions",
            "three_or_more": "three or more separate actions or deliverables",
        },
    },
}
_STEPS = {"one": 1, "two": 2, "three_or_more": 3}


class JevAPIError(RuntimeError):
    pass


class _HttpTransport:
    def __init__(self, api_key: str, base_url: str, timeout: float) -> None:
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/v1/systemone"
        self._timeout = timeout

    def __call__(self, body: dict) -> dict:
        request = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise JevAPIError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise JevAPIError(f"request failed: {exc}") from exc


def _yes_probability(answer: dict) -> float:
    return float(answer["noul"])


def _choice_confidence(answer: dict) -> float:
    # `probabilities` is keyed by label; when a response omits it, trust the label.
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        total = sum(probabilities.values()) or 1.0
        return max(probabilities.values()) / total
    return 1.0


def decision_from_answers(answers: dict, min_confidence: float = JEV_CONFIDENCE) -> Optional[dict]:
    """Map Jev's answers to TriageDecision fields; None if any answer is not confident enough."""
    flags = {}
    for name in ("split_needed", "review_needed", "steps_are_dependent"):
        p_yes = _yes_probability(answers[name])
        if max(p_yes, 1.0 - p_yes) < min_confidence:
            return None
        flags[name] = p_yes >= 0.5
    # risk_level is informational only (routing uses review_needed and distinct_steps), and Jev
    # is routinely unsure about it on real tool tasks, so it is not gated on confidence.
    risk, steps = answers["risk_level"], answers["distinct_steps"]
    if _choice_confidence(steps) < min_confidence:
        return None
    if risk["choice"] not in ("low", "medium", "high") or steps["choice"] not in _STEPS:
        return None
    return {
        **flags,
        "risk_level": risk["choice"],
        "distinct_steps": _STEPS[steps["choice"]],
        "reasoning": "Routed by Jev (deterministic typed triage).",
    }


def jev_triage(task: str, transport: Optional[Transport] = None) -> Optional[dict]:
    if TRIAGE_BACKEND == "llm":
        return None
    # Long or meta prompts (the self-improvement runner raises AGENT_MIN_TIER for them) stay on
    # the LLM triage and are never sent to a third party.
    if AGENT_MIN_TIER != "low" or len(task) > JEV_MAX_TASK_CHARS:
        return None
    if transport is None:
        if not TYPESAFE_API_KEY:
            return None
        transport = _HttpTransport(TYPESAFE_API_KEY, JEV_BASE_URL, JEV_TIMEOUT)
    try:
        response = transport({"model": JEV_MODEL, "state": {"text": task}, "questions": QUESTIONS})
        decision = decision_from_answers(response["answers"])
    except (JevAPIError, KeyError, TypeError, ValueError) as exc:
        log.warning("Jev triage failed, using LLM triage: %s", exc)
        return None
    if decision is None:
        log.info("Jev triage below %.2f confidence, using LLM triage", JEV_CONFIDENCE)
    return decision
