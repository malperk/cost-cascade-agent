import pytest

from agent import tools


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "WORKSPACE_DIR", tmp_path)
    return tmp_path


def _read(path, **kwargs):
    return tools.file_read.invoke({"path": path, **kwargs})


def test_small_file_unchanged(workspace):
    (workspace / "a.txt").write_text("hello")
    assert _read("a.txt") == "hello"


def test_large_file_is_capped_and_says_where_to_continue(workspace):
    limit = tools.MAX_FILE_READ_CHARS
    (workspace / "page.html").write_text("x" * limit + "y" * 1_200_000)
    out = _read("page.html")
    assert len(out) < limit + 200
    assert out.startswith("x" * limit)
    assert f"call file_read with offset={limit} to read more" in out


def test_offset_reads_the_next_chunk_until_end_of_file(workspace):
    limit = tools.MAX_FILE_READ_CHARS
    (workspace / "big.txt").write_text("a" * limit + "b" * 50)
    out = _read("big.txt", offset=limit)
    assert out.startswith("b" * 50)
    assert f"showing characters {limit}-{limit + 50} of {limit + 50}; end of file" in out


def test_limit_above_cap_is_clamped(workspace):
    (workspace / "big.txt").write_text("z" * 100_000)
    out = _read("big.txt", limit=1_000_000)
    assert len(out) < tools.MAX_FILE_READ_CHARS + 200


def test_offset_past_end_returns_empty_chunk(workspace):
    (workspace / "a.txt").write_text("hello")
    out = _read("a.txt", offset=99)
    assert "showing characters 5-5 of 5; end of file" in out
