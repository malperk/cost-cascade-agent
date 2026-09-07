import operator
from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages


class TaskState(TypedDict, total=False):
    task: str
    depth: int  # 0 = top-level task, 1 = subtask (subtasks do not split further)

    # triage decision
    split_needed: bool
    risk_level: Literal["low", "medium", "high"]
    review_needed: bool
    triage_reasoning: str

    # split
    subtasks: list[str]
    subtask_results: Annotated[list[str], operator.add]

    # execution (tool-calling agent loop)
    messages: Annotated[list, add_messages]
    tier: Literal["low", "mid", "high", "local"]
    attempt_count: int
    output: str
    infra_fallback: bool  # True if cheap tier's provider pool was unavailable and we fell back to premium

    # review
    review_passed: Optional[bool]
    review_feedback: Optional[str]

    # final
    final_output: str
