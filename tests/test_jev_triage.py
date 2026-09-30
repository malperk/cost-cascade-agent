from agent import jev_triage as jt


def _answers(split=0.1, review=0.9, dep=0.1, risk="low", steps="one", risk_probs=None):
    risk_answer = {"choice": risk}
    if risk_probs:
        risk_answer["probabilities"] = risk_probs
    return {
        "split_needed": {"noul": split},
        "review_needed": {"noul": review},
        "steps_are_dependent": {"noul": dep},
        "risk_level": risk_answer,
        "distinct_steps": {"choice": steps},
    }


def test_confident_answers_map_to_decision():
    d = jt.decision_from_answers(_answers(split=0.05, review=0.95, dep=0.9, risk="high", steps="three_or_more"))
    assert d["split_needed"] is False and d["review_needed"] is True and d["steps_are_dependent"] is True
    assert d["risk_level"] == "high" and d["distinct_steps"] == 3


def test_low_confidence_noul_falls_back():
    assert jt.decision_from_answers(_answers(review=0.55)) is None


def test_low_confidence_choice_falls_back():
    assert jt.decision_from_answers(_answers(risk_probs={"low": 0.4, "medium": 0.35, "high": 0.25})) is None


def test_unknown_label_falls_back():
    assert jt.decision_from_answers(_answers(steps="seven")) is None


def test_jev_triage_uses_transport(monkeypatch):
    monkeypatch.setattr(jt, "TRIAGE_BACKEND", "auto")
    monkeypatch.setattr(jt, "AGENT_MIN_TIER", "low")
    assert jt.jev_triage("hi", transport=lambda body: {"answers": _answers()})["distinct_steps"] == 1


def test_no_key_no_call(monkeypatch):
    monkeypatch.setattr(jt, "TYPESAFE_API_KEY", "")
    assert jt.jev_triage("hi") is None


def test_api_error_and_bad_shape_fall_back(monkeypatch):
    def boom(body):
        raise jt.JevAPIError("down")

    assert jt.jev_triage("hi", transport=boom) is None
    assert jt.jev_triage("hi", transport=lambda body: {"oops": 1}) is None


def test_long_task_and_raised_tier_skip_jev(monkeypatch):
    called = []
    transport = lambda body: called.append(1) or {"answers": _answers()}
    assert jt.jev_triage("x" * (jt.JEV_MAX_TASK_CHARS + 1), transport=transport) is None
    monkeypatch.setattr(jt, "AGENT_MIN_TIER", "mid")
    assert jt.jev_triage("hi", transport=transport) is None
    assert not called


def test_llm_backend_disables_jev(monkeypatch):
    monkeypatch.setattr(jt, "TRIAGE_BACKEND", "llm")
    assert jt.jev_triage("hi", transport=lambda body: {"answers": _answers()}) is None
