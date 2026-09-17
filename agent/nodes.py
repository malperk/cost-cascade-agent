import json
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from agent.config import AGENT_MAX_TIER, AGENT_MIN_TIER, MAX_ATTEMPTS
from agent.llm import INFRA_ERRORS, NEXT_TIER, client_for_tier, invoke_with_retry
from agent.state import TaskState
from agent.tools import ALL_TOOLS

TIER_ORDER = ["low", "mid", "high", "local"]


def _capped_next_tier(current_tier: str) -> str:
    """Review-driven escalation's natural next tier, clamped at AGENT_MAX_TIER (default
    "local" - uncapped, the normal-run behavior). Once the natural next tier would go past
    the cap, repeat at the capped tier instead of dropping to a weaker one - see
    agent/config.py's AGENT_MAX_TIER for why "local" specifically isn't always an improvement."""
    natural_next = NEXT_TIER.get(current_tier, "local")
    if TIER_ORDER.index(natural_next) > TIER_ORDER.index(AGENT_MAX_TIER):
        return current_tier
    return natural_next


class TriageDecision(BaseModel):
    split_needed: bool = Field(description="True only if the task is genuinely complex enough to benefit from decomposition into subtasks")
    risk_level: Literal["low", "medium", "high"] = Field(description="How likely a cheap model is to get this wrong")
    review_needed: bool = Field(description="True only if there is meaningful risk the cheap model's output is wrong and needs checking")
    reasoning: str = Field(description="One sentence justifying the above decisions")


class SplitPlan(BaseModel):
    subtasks: list[str] = Field(description="2-6 self-contained subtasks that together accomplish the original task")


class ReviewResult(BaseModel):
    passed: bool = Field(description="True if the output correctly and completely accomplishes the task")
    feedback: str = Field(description="If failed, a specific description of what is wrong")


def triage_node(state: TaskState) -> dict:
    depth = state.get("depth", 0)
    # AGENT_MIN_TIER defaults to "low" for normal runs; the self-improvement runner raises it
    # (see agent/config.py) for its authoring container, where a long, meta, instruction-heavy
    # prompt proved unreliable at "low" - both the triage classification itself and the
    # resulting starting execution tier are raised together, since a confused triage call was
    # part of the observed failure (it rated the task low-risk/no-review-needed).
    llm = client_for_tier(AGENT_MIN_TIER).with_structured_output(TriageDecision, method="json_schema")
    decision: TriageDecision = invoke_with_retry(
        llm,
        "Assess this task and decide the minimum necessary execution path.\n"
        "Only ask for splitting if the task is genuinely complex (multiple distinct deliverables).\n"
        "Only ask for review if a cheap/fast model plausibly could get it wrong (ambiguity, "
        "precision-sensitive output, multi-step reasoning, factual risk).\n"
        "Also mark review_needed=True whenever the task requires taking an action through a tool "
        "(creating/editing files, running code, searching the web, calling an API): cheap models "
        "are prone to executing the action correctly but then summarizing it wrong, so the final "
        "answer needs to be checked against what was actually done.\n\n"
        f"Task: {state['task']}",
    )
    split_needed = decision.split_needed and depth == 0  # never split subtasks further
    return {
        "split_needed": split_needed,
        "risk_level": decision.risk_level,
        "review_needed": decision.review_needed,
        "triage_reasoning": decision.reasoning,
        "tier": AGENT_MIN_TIER,
        "attempt_count": 0,
    }


def planner_node(state: TaskState) -> dict:
    llm = client_for_tier(AGENT_MIN_TIER).with_structured_output(SplitPlan, method="json_schema")
    plan: SplitPlan = invoke_with_retry(
        llm,
        "Break this task into 2-6 self-contained subtasks that together fully accomplish it. "
        "Each subtask should be independently executable.\n\n"
        f"Task: {state['task']}",
    )
    return {"subtasks": plan.subtasks}


def run_subtask_node(state: TaskState) -> dict:
    import agent.graph as graph_module  # lazy import avoids circular import at module load time

    result = graph_module.app.invoke(
        {"task": state["task"], "depth": state.get("depth", 1)}
    )
    return {"subtask_results": [result.get("final_output", "")]}


def aggregate_node(state: TaskState) -> dict:
    parts = []
    for subtask, result in zip(state.get("subtasks", []), state.get("subtask_results", [])):
        parts.append(f"### {subtask}\n{result}")
    return {"final_output": "\n\n".join(parts)}


AGENT_SYSTEM_PROMPT = (
    "You are an agent with tools for file operations (file_read, file_write, list_directory), "
    "code execution, web search, and calculation.\n"
    "When a task asks you to create, write, generate, or save files or code, you must actually "
    "call file_write to persist each file into the workspace - do not just print the code as text.\n"
    "If a script you run via code_execution computes or writes structured data (a list, a table, "
    "per-item results), you must print the actual data (or read it back with file_read) before "
    "writing your final answer - never print just a count/summary and then describe the details "
    "from memory or assumption. Any specific claim in your final answer (a count, 'all N have "
    "property X', which items passed/failed) must be something you actually saw in a tool result "
    "in this conversation, not an inference from a summary line.\n"
    "When you are completely done, call the FinalAnswer tool exactly once instead of replying with "
    "plain text. List only the actions you actually took via tools, and keep the summary factual - "
    "do not add speculative suggestions, alternative implementations, or example code you did not "
    "actually create with a tool."
)

FINAL_ANSWER_TOOL_NAME = "FinalAnswer"


class FinalAnswer(BaseModel):
    """Call this exactly once when you are completely done and ready to report your final answer."""

    actions_taken: list[str] = Field(
        default_factory=list,
        description="Concrete actions actually performed via tools during this task, e.g. "
        "'wrote file X.py', 'created folder Y'. Empty list if no tools were used.",
    )
    summary: str = Field(description="A concise, factual final answer for the user - only what was actually done or found.")


def _format_final_answer(args: dict) -> str:
    summary = args.get("summary", "")
    actions = args.get("actions_taken") or []
    if isinstance(actions, str):
        # weaker/local models occasionally emit this field as a JSON-encoded string instead
        # of an actual array in their tool-call arguments - without this, iterating over that
        # string prints one line per CHARACTER instead of one line per action.
        try:
            parsed = json.loads(actions)
            actions = parsed if isinstance(parsed, list) else [actions]
        except json.JSONDecodeError:
            actions = [actions]
    if not actions:
        return summary
    actions_block = "\n".join(f"- {a}" for a in actions)
    return f"{summary}\n\nActions taken:\n{actions_block}"


def agent_node(state: TaskState) -> dict:
    tier = state.get("tier", "low")
    bound_tools = ALL_TOOLS + [FinalAnswer]
    llm = client_for_tier(tier).bind_tools(bound_tools)
    messages = state.get("messages") or [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=state["task"]),
    ]

    try:
        response = invoke_with_retry(llm, messages)
        fell_back_tier = None
    except INFRA_ERRORS:
        next_tier = NEXT_TIER.get(tier)
        if next_tier is None:
            raise  # already at the last tier (local), nothing left to fall back to
        # this tier's provider pool is unavailable (not a quality problem) -
        # fall back to the next tier up so the run still completes.
        response = invoke_with_retry(client_for_tier(next_tier).bind_tools(bound_tools), messages)
        fell_back_tier = next_tier

    update: dict = {"messages": [response]}
    if fell_back_tier:
        update["tier"] = fell_back_tier
        update["infra_fallback"] = True

    tool_calls = response.tool_calls or []
    final_call = next((c for c in tool_calls if c["name"] == FINAL_ANSWER_TOOL_NAME), None)
    action_calls = [c for c in tool_calls if c["name"] != FINAL_ANSWER_TOOL_NAME]

    if final_call is not None:
        # Treat this turn as terminal even if the model also tried to call real tools
        # alongside FinalAnswer - acknowledge every tool_call (so message history stays
        # valid for providers that require a matching ToolMessage per call) and end here.
        acks = [ToolMessage(content="Final answer recorded.", tool_call_id=final_call["id"])]
        for call in action_calls:
            acks.append(ToolMessage(content="Skipped: FinalAnswer was called in the same turn.", tool_call_id=call["id"]))
        update["messages"] = [response, *acks]
        update["output"] = _format_final_answer(final_call["args"])
        update["attempt_count"] = state.get("attempt_count", 0) + 1
    elif not action_calls:
        # no tool call at all - model replied with plain text instead of calling FinalAnswer.
        update["output"] = response.content
        update["attempt_count"] = state.get("attempt_count", 0) + 1
    return update


def _summarize_tool_activity(state: TaskState) -> str:
    lines = []
    for msg in state.get("messages", []):
        for call in getattr(msg, "tool_calls", None) or []:
            if call["name"] == FINAL_ANSWER_TOOL_NAME:
                continue
            lines.append(f"- called {call['name']}({call['args']})")
        if isinstance(msg, ToolMessage) and msg.content != "Final answer recorded.":
            lines.append(f"  -> result: {msg.content}")
    return "\n".join(lines) if lines else "(no tools were called)"


def review_node(state: TaskState) -> dict:
    llm = client_for_tier("high").with_structured_output(ReviewResult, method="json_schema")
    try:
        result: ReviewResult = invoke_with_retry(
            llm,
            "Verify whether the final answer below correctly and completely accomplishes the task. "
            "Be strict but fair. Pay special attention to whether the final answer accurately "
            "reflects what the tool activity actually did - a model can execute an action correctly "
            "via tools and then describe it wrong (or produce an unrelated reply) in its final answer; "
            "that must fail review even if the tool calls themselves succeeded.\n"
            "Specifically: if the final answer states a specific fact about data (a count, 'all N have "
            "property X', which items passed/failed, per-item results) and that exact fact is not "
            "directly visible in the tool activity below (e.g. only a summary count was printed, not "
            "the actual per-item data), treat the claim as UNVERIFIED and FAIL the review - the answer "
            "must not assert details it never actually showed evidence for, even if the general summary "
            "sounds plausible.\n\n"
            f"Task: {state['task']}\n\n"
            f"Tool activity:\n{_summarize_tool_activity(state)}\n\n"
            f"Final answer to verify:\n{state['output']}",
        )
    except INFRA_ERRORS as exc:
        # never silently accept on an infra hiccup - treat it like a failed review so the
        # normal escalate/reject bounding takes over instead of crashing the whole run.
        return {
            "review_passed": False,
            "review_feedback": f"Review itself could not be completed due to an infrastructure error ({exc}). Not verified - treat as failing until a review actually succeeds.",
        }
    return {"review_passed": result.passed, "review_feedback": result.feedback}


def escalate_node(state: TaskState) -> dict:
    next_tier = _capped_next_tier(state.get("tier", "low"))
    feedback = HumanMessage(
        content=(
            "Your previous answer did not pass review.\n"
            f"Reviewer feedback: {state.get('review_feedback', '')}\n\n"
            "Here is the original task again, in full - re-read it carefully and follow it "
            "literally (if it describes a specific procedure/script to run, run exactly that, "
            "don't substitute a different approach, an existing file, or ask for clarification):\n\n"
            f"{state['task']}\n\n"
            "Now produce a corrected, complete answer that actually fixes what the reviewer "
            "flagged above."
        )
    )
    return {"tier": next_tier, "messages": [feedback]}


def accept_node(state: TaskState) -> dict:
    return {"final_output": state.get("output", "")}


def reject_node(state: TaskState) -> dict:
    # every tier failed review - do not present a known-wrong answer as if it were accepted.
    return {
        "final_output": (
            "TASK FAILED: could not produce an answer that passed review, even after "
            "escalating through every tier (low -> mid -> high).\n\n"
            f"Last reviewer feedback: {state.get('review_feedback', '(none)')}\n\n"
            "Last attempted (UNVERIFIED, likely incorrect) answer:\n"
            f"{state.get('output', '')}"
        )
    }


def route_after_triage(state: TaskState) -> Literal["planner", "agent"]:
    return "planner" if state.get("split_needed") else "agent"


def route_after_planner(state: TaskState):
    from langgraph.types import Send

    return [
        Send("run_subtask", {"task": t, "depth": state.get("depth", 0) + 1})
        for t in state["subtasks"]
    ]


def route_after_agent(state: TaskState) -> Literal["tools", "review", "accept"]:
    last = state["messages"][-1]
    # agent_node only leaves an AIMessage with tool_calls as the tail when those calls
    # are real actions (ALL_TOOLS) - a FinalAnswer turn always ends with an ack ToolMessage.
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "review" if state.get("review_needed") else "accept"


def route_after_review(state: TaskState) -> Literal["accept", "escalate", "reject"]:
    if state.get("review_passed"):
        return "accept"
    if state.get("attempt_count", 0) >= MAX_ATTEMPTS:
        return "reject"  # bounded: every tier failed review, don't present a wrong answer as accepted
    return "escalate"
