import ast
import operator
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from agent.config import BROWSER_ENABLED, WORKSPACE_DIR

# See agent/llm.py for why this exists: inside a self-improvement authoring/eval container
# (--network none), there is no direct internet access at all, so web search is relayed
# through the same trusted host broker that handles model inference.
_BROKER_SOCKET_PATH = os.getenv("BROKER_SOCKET_PATH")

if _BROKER_SOCKET_PATH:
    _broker_client = httpx.Client(
        transport=httpx.HTTPTransport(uds=_BROKER_SOCKET_PATH), timeout=30
    )

    @tool
    def web_search(query: str) -> str:
        """Search the web for the given query and return a short summary of results."""
        try:
            response = _broker_client.post("http://broker/search", json={"query": query})
            response.raise_for_status()
            return response.json().get("result", "")
        except Exception as exc:
            return f"Error: web search failed ({exc})"
else:
    _ddg = DuckDuckGoSearchRun()

    # DuckDuckGo intermittently fails (DNS/connect errors from its backends) and the exception
    # would otherwise propagate out of the tools node and crash the whole run. Retry once, then
    # report the failure as the tool result - same contract as the broker branch above - so the
    # model can retry or answer without search.
    @tool("duckduckgo_search")
    def web_search(query: str) -> str:
        """A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query."""
        last_exc = None
        for attempt in range(2):
            try:
                return _ddg.invoke(query)
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    time.sleep(1)
        return f"Error: web search failed ({str(last_exc)[:200]})"


def _resolve_safe_path(path: str) -> Path:
    target = (WORKSPACE_DIR / path).resolve()
    if target != WORKSPACE_DIR and WORKSPACE_DIR not in target.parents:
        raise ValueError(f"'{path}' is outside the agent workspace ({WORKSPACE_DIR})")
    return target


@tool
def file_read(path: str) -> str:
    """Read and return the text contents of a file inside the agent workspace."""
    try:
        return _resolve_safe_path(path).read_text()
    except FileNotFoundError:
        return f"Error: file '{path}' not found"
    except Exception as exc:
        return f"Error: {exc}"


@tool
def file_write(path: str, content: str) -> str:
    """Create or overwrite a text file inside the agent workspace with the given content."""
    try:
        target = _resolve_safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Wrote {len(content)} characters to {path}"
    except Exception as exc:
        return f"Error: {exc}"


@tool
def list_directory(path: str = ".") -> str:
    """List files and subdirectories at the given path inside the agent workspace."""
    try:
        target = _resolve_safe_path(path)
        entries = sorted(target.iterdir())
        if not entries:
            return "(empty directory)"
        return "\n".join(f"{'d' if e.is_dir() else 'f'}  {e.name}" for e in entries)
    except Exception as exc:
        return f"Error: {exc}"


_MATH_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _MATH_OPERATORS:
        return _MATH_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _MATH_OPERATORS:
        return _MATH_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric arithmetic expression (+ - * / // % **) and return the result."""
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
        return str(result)
    except Exception as exc:
        return f"Error: could not evaluate '{expression}' ({exc})"


MAX_CODE_EXECUTION_TIMEOUT = 600


@tool
def code_execution(code: str, timeout_seconds: int = 10) -> str:
    """Run a snippet of Python code in a subprocess and return its stdout/stderr.

    timeout_seconds defaults to 10 for quick snippets; pass a higher value
    (up to 600) for longer-running scripts (e.g. calling external CLIs or
    running a multi-minute test suite).

    Not a hardened sandbox: the subprocess has the same filesystem/network
    permissions as this process, isolated only by a process boundary and the
    timeout. Do not expose this to untrusted input without adding real
    sandboxing (e.g. Docker/gVisor) first.
    """
    timeout_seconds = max(1, min(timeout_seconds, MAX_CODE_EXECUTION_TIMEOUT))
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=WORKSPACE_DIR,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: code execution timed out after {timeout_seconds} seconds"
    except Exception as exc:
        return f"Error: {exc}"


ALL_TOOLS = [file_read, file_write, list_directory, calculator, code_execution, web_search]

# No browser inside self-improvement containers (no network there), or when switched off.
if BROWSER_ENABLED and not _BROKER_SOCKET_PATH:
    try:
        from agent.browser import BROWSER_TOOLS

        ALL_TOOLS += BROWSER_TOOLS
    except ImportError:  # playwright not installed
        pass
