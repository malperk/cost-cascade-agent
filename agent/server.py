import hmac
import sys

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from agent.config import SERVER_HOST, SERVER_PORT, SERVER_TOKEN
from agent.graph import app as graph_app
from agent.jobs import JobStore
from agent.model_catalog import tier_warnings
from agent.notify import email_configured, send_email

MAX_TASK_CHARS = 4000


def run_graph(task: str) -> dict:
    result = graph_app.invoke({"task": task, "depth": 0})
    rejected = result.get("review_passed") is False and "TASK FAILED" in result.get("final_output", "")
    return {
        "accepted": not rejected,
        "final_output": result.get("final_output", "<no output>"),
        "tier": result.get("tier"),
        "attempt_count": result.get("attempt_count"),
        "review_passed": result.get("review_passed"),
        "triage_reasoning": result.get("triage_reasoning"),
    }


jobs = JobStore(run_graph, on_complete=[send_email])

# DNS-rebinding protection only matters for browser clients on localhost; the bearer token is
# the real gate here, and Hermes reaches us by LAN IP.
mcp = FastMCP("langgraph-agent", transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))


@mcp.tool()
def submit_task(task: str, notify: bool = False) -> dict:
    """Start a task on the cost-cascade agent and return immediately with a job id.

    The task runs in the background (it can take minutes). Poll get_task_result(job_id) for the
    outcome. With notify=True the result is also emailed to the owner when the task finishes."""
    task = task.strip()
    if not task:
        return {"error": "task is empty"}
    if len(task) > MAX_TASK_CHARS:
        return {"error": f"task too long ({len(task)} > {MAX_TASK_CHARS} chars)"}
    if notify and not email_configured():
        return {"error": "notify=True but email is not configured on the server"}
    job = jobs.submit(task, notify=notify)
    return {"job_id": job["id"], "status": job["status"]}


@mcp.tool()
def get_task_result(job_id: str) -> dict:
    """Get a job's status ("queued", "running", "done", "failed", "interrupted") and, once
    finished, its result: final_output, tier, attempt_count, accepted."""
    job = jobs.get(job_id)
    if job is None:
        return {"error": f"unknown job_id {job_id}"}
    return {k: job[k] for k in ("id", "task", "status", "result", "error", "created_at", "started_at", "finished_at")}


@mcp.tool()
def list_tasks(limit: int = 20) -> list[dict]:
    """List the most recent jobs (id, task, status, timestamps)."""
    return jobs.list(limit=max(1, min(limit, 100)))


class BearerAuth:
    """ASGI middleware: reject every request that lacks the shared bearer token."""

    def __init__(self, inner, token: str):
        self.inner = inner
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            given = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(given, self.expected):
                await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"unauthorized"})
                return
        await self.inner(scope, receive, send)


def main() -> None:
    if len(SERVER_TOKEN) < 24:
        sys.exit("Set AGENT_SERVER_TOKEN (>=24 chars) in .env, e.g. `python -c \"import secrets; print(secrets.token_urlsafe(32))\"`")
    for warning in tier_warnings():
        print(f"[model catalog] {warning}", file=sys.stderr)
    print(f"MCP endpoint: http://{SERVER_HOST}:{SERVER_PORT}/mcp  (email notify: {'on' if email_configured() else 'off'})")
    uvicorn.run(BearerAuth(mcp.streamable_http_app(), SERVER_TOKEN), host=SERVER_HOST, port=SERVER_PORT)


if __name__ == "__main__":
    main()
