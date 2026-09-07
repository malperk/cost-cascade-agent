from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes import (
    accept_node,
    agent_node,
    aggregate_node,
    escalate_node,
    planner_node,
    reject_node,
    review_node,
    route_after_agent,
    route_after_planner,
    route_after_review,
    route_after_triage,
    run_subtask_node,
    triage_node,
)
from agent.state import TaskState
from agent.tools import ALL_TOOLS


def build_graph():
    graph = StateGraph(TaskState)

    graph.add_node("triage", triage_node)
    graph.add_node("planner", planner_node)
    graph.add_node("run_subtask", run_subtask_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("review", review_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("accept", accept_node)
    graph.add_node("reject", reject_node)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", route_after_triage, ["planner", "agent"])
    graph.add_conditional_edges("planner", route_after_planner, ["run_subtask"])
    graph.add_edge("run_subtask", "aggregate")
    graph.add_edge("aggregate", END)
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "review", "accept"])
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("review", route_after_review, ["accept", "escalate", "reject"])
    graph.add_edge("escalate", "agent")
    graph.add_edge("accept", END)
    graph.add_edge("reject", END)

    return graph.compile()


app = build_graph()
