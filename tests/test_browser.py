from agent import browser as b

CANDS = [
    {"id": 0, "kind": "a", "name": "Home"},
    {"id": 1, "kind": "input:search", "name": "Search"},
    {"id": 2, "kind": "button", "name": "Search"},
    {"id": 3, "kind": "button", "name": "Search"},
]


def _answers(action="click", match=0.9, target="2", target_probs=None, action_probs=None):
    return {"answers": {
        "action": {"choice": action, "probabilities": action_probs or {action: 1.0}},
        "match": {"noul": match},
        "target": {"choice": target, "probabilities": target_probs or {target: 1.0}},
    }}


def test_confident_click():
    a = b.jev_decide("click Search", "https://x.test/p?q=1", CANDS, lambda body: _answers())
    assert (a.action, a.candidate_id) == ("click", 2)


def test_strong_target_ignores_weak_match_signal():
    assert b.jev_decide("click Search", "https://x.test", CANDS, lambda body: _answers(match=0.5)) is not None


def test_merely_confident_target_needs_match():
    probs = {"2": 0.8, "3": 0.2}
    assert b.jev_decide("x", "https://x.test", CANDS, lambda body: _answers(match=0.5, target_probs=probs)) is None
    assert b.jev_decide("x", "https://x.test", CANDS, lambda body: _answers(match=0.8, target_probs=probs)) is not None


def test_unsure_target_falls_back():
    assert b.jev_decide("x", "https://x.test", CANDS, lambda body: _answers(target_probs={"2": 0.5, "3": 0.5})) is None


def test_fill_takes_value_from_quotes_or_falls_back():
    fill = lambda body: _answers(action="fill", target="1")
    assert b.jev_decide('type "hello" into search', "https://x.test", CANDS, fill).value == "hello"
    assert b.jev_decide("type something into search", "https://x.test", CANDS, fill) is None


def test_query_and_fragment_not_sent():
    sent = {}

    def transport(body):
        sent.update(body)
        return _answers()

    b.jev_decide("click", "https://x.test/path?token=secret#frag", CANDS, transport)
    assert "secret" not in str(sent) and "https://x.test/path" in str(sent)


def test_relocate_finds_same_occurrence_after_renumbering():
    after = [{"id": 0, "kind": "li", "name": "suggestion"}] + [dict(c, id=c["id"] + 1) for c in CANDS]
    assert b.relocate(CANDS, 3, after) == 4
    assert b.relocate(CANDS, 2, after) == 3
    assert b.relocate(CANDS, 0, [c for c in after if c["name"] != "Home"]) is None


def test_non_http_urls_rejected():
    assert b.browser_open.invoke({"url": "file:///etc/passwd"}).startswith("Error: only http")
