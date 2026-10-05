from agent import tools


class _Flaky:
    def __init__(self, fail_times):
        self.calls = 0
        self.fail_times = fail_times

    def invoke(self, query):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("DNS error")
        return "result"


def test_search_failure_becomes_tool_result(monkeypatch):
    monkeypatch.setattr(tools, "_ddg", _Flaky(fail_times=5), raising=False)
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)
    out = tools.web_search.invoke({"query": "q"})
    assert out.startswith("Error: web search failed") and "DNS error" in out


def test_search_retries_once(monkeypatch):
    flaky = _Flaky(fail_times=1)
    monkeypatch.setattr(tools, "_ddg", flaky, raising=False)
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)
    assert tools.web_search.invoke({"query": "q"}) == "result"
    assert flaky.calls == 2
