from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import nodes


class _FakeLLM:
    def __init__(self, seen):
        self.seen = seen

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        return AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "2*2"}, "id": "c1"}])


def test_opening_messages_are_persisted_so_next_turn_sees_the_task(monkeypatch):
    seen = []
    monkeypatch.setattr(nodes, "client_for_tier", lambda tier: _FakeLLM(seen))
    monkeypatch.setattr(nodes, "invoke_with_retry", lambda llm, messages: llm.invoke(messages))

    update = nodes.agent_node({"task": "What is 2*2?", "tier": "low"})

    stored = update["messages"]
    assert isinstance(stored[0], SystemMessage) and isinstance(stored[1], HumanMessage)
    assert stored[1].content == "What is 2*2?"
    assert isinstance(stored[-1], AIMessage)

    # second turn: state now carries the stored history, nothing is prepended twice
    state = {"task": "What is 2*2?", "tier": "low", "messages": [*stored, ToolMessage(content="4", tool_call_id="c1")]}
    second = nodes.agent_node(state)
    assert [type(m) for m in seen[1][:2]] == [SystemMessage, HumanMessage]
    assert len(second["messages"]) == 1  # only the new reply is appended
