from agent import tools


def _run(code):
    return tools.code_execution.invoke({"code": code})


def test_short_output_unchanged():
    assert _run("print('hello')") == "hello"


def test_long_stdout_keeps_head_and_tail():
    out = _run("print('START' + 'x' * 50000 + 'END')")
    limit = tools.MAX_CODE_EXECUTION_OUTPUT_CHARS
    assert len(out) < limit + 200
    assert out.startswith("START") and out.endswith("END")
    assert "characters omitted" in out


def test_huge_traceback_keeps_exception_line():
    # The live failure: an exception whose message embeds a whole HTML page.
    out = _run("raise OSError('<html>' + 'y' * 1_200_000 + '</html>')")
    assert len(out) < tools.MAX_CODE_EXECUTION_OUTPUT_CHARS + 200
    assert out.startswith("[stderr]\nTraceback")
    assert out.endswith("</html>")


def test_truncate_middle_reports_omitted_count():
    text = "a" * 30 + "b" * 30
    out = tools._truncate_middle(text, limit=20)
    assert out.startswith("a" * 10) and out.endswith("b" * 10)
    assert "[... 40 characters omitted" in out
