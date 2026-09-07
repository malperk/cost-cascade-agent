"""Trusted host broker for the self-improvement loop.

Holds the only real OmniRoute API key and is the sole thing an authoring or
evaluation container (run with --network none) can reach, via a per-role
Unix domain socket (see agent/llm.py and agent/tools.py for the client side
of this: BROKER_SOCKET_PATH switches them onto an httpx UDS transport).

Forwards a strict allowlist of inference-shaped requests to real OmniRoute
and relays web search server-side; denies everything else by default; and
enforces a per-role call budget the runner sizes from its own eval plan
(never hardcoded here). This process is the only place the real API key
ever lives - it is read once from the environment and never written to the
config file, logged, or exposed to a connecting container.

Usage: python -m tools.omniroute_broker --config <path-to-json-config>

Config file shape:
{
  "sockets": {"authoring": "/path/a.sock", "baseline": "/path/b.sock", "candidate": "/path/c.sock"},
  "budgets": {"authoring": {"model_calls": 15, "search_calls": 5}, ...},
  "log_path": "/path/to/broker.log",
  "upstream_base_url": "http://localhost:20128/v1",   # optional, defaults to OMNIROUTE_BASE_URL
  "request_timeout_seconds": 60,
  "max_body_bytes": 2000000
}
"""

import argparse
import json
import os
import signal
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import httpx
from langchain_community.tools import DuckDuckGoSearchRun

ALLOWED_ROUTES = {
    ("POST", "/v1/chat/completions"),
    ("GET", "/v1/models"),
    ("POST", "/search"),
}

DEFAULT_MAX_BODY_BYTES = 2_000_000
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60

REAL_API_KEY = None  # set once in main() from the environment, never from the config file


def _strip_v1_suffix(base_url: str) -> str:
    """OMNIROUTE_BASE_URL already ends in /v1; incoming paths also start with /v1
    (they're built by the openai SDK from ChatOpenAI(base_url="http://broker/v1")).
    Forwarding needs the bare host root once, not the /v1 twice."""
    return base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url


class Budget:
    """Per-role call budget. Thread-safe; sized by the runner's own eval plan, not a guess."""

    def __init__(self, model_calls: int, search_calls: int):
        self._lock = threading.Lock()
        self.model_calls_remaining = model_calls
        self.search_calls_remaining = search_calls

    def take_model_call(self) -> bool:
        with self._lock:
            if self.model_calls_remaining <= 0:
                return False
            self.model_calls_remaining -= 1
            return True

    def take_search_call(self) -> bool:
        with self._lock:
            if self.search_calls_remaining <= 0:
                return False
            self.search_calls_remaining -= 1
            return True


class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        socket_path: str,
        handler_cls,
        role: str,
        budget: Budget,
        upstream_root: str,
        request_timeout: float,
        max_body_bytes: int,
        log_lock: threading.Lock,
        log_path: str,
        search_tool: DuckDuckGoSearchRun,
    ):
        self.role = role
        self.budget = budget
        self.upstream_root = upstream_root
        self.request_timeout = request_timeout
        self.max_body_bytes = max_body_bytes
        self.log_lock = log_lock
        self.log_path = log_path
        self.search_tool = search_tool
        super().__init__(socket_path, handler_cls)


class BrokerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # noqa: A002 - matches BaseHTTPRequestHandler signature
        pass  # suppressed default stderr access log; structured JSONL logging is used instead

    def _write_log(self, **fields):
        record = {"ts": time.time(), "role": self.server.role, **fields}
        with self.server.log_lock:
            with open(self.server.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _deny(self, status: int, reason: str):
        self._write_log(method=self.command, path=self.path, status=status, denied=reason)
        self._send_json(status, {"error": reason})

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > self.server.max_body_bytes:
            raise ValueError("request body too large")
        return self.rfile.read(length) if length else b""

    def _handle(self, method: str):
        # The body must always be drained from the socket before responding, even when the
        # route is about to be denied - this is a persistent (keep-alive) HTTP/1.1 connection,
        # and leaving an unread request body sitting in the stream corrupts the next request
        # read on the same connection (its bytes get parsed as part of the following request
        # line/headers).
        try:
            body = self._read_body()
        except ValueError as exc:
            self.close_connection = True  # body was too large to safely drain and discard
            return self._deny(413, str(exc))

        route = (method, self.path)
        if route not in ALLOWED_ROUTES:
            return self._deny(403, f"route not allowed: {method} {self.path}")

        if self.path == "/search":
            return self._handle_search(body)
        return self._handle_inference(method, body)

    def _handle_search(self, body: bytes):
        if not self.server.budget.take_search_call():
            return self._deny(429, "search call budget exhausted for this role")
        try:
            payload = json.loads(body or b"{}")
            query = payload.get("query", "")
            result = self.server.search_tool.run(query)
        except Exception as exc:
            self._write_log(method="POST", path="/search", status=502, error=str(exc))
            return self._send_json(502, {"error": f"search failed: {exc}"})
        self._write_log(method="POST", path="/search", status=200, query=query)
        return self._send_json(200, {"result": result})

    def _handle_inference(self, method: str, body: bytes):
        if not self.server.budget.take_model_call():
            return self._deny(429, "model call budget exhausted for this role")

        url = self.server.upstream_root + self.path
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {REAL_API_KEY}"}
        try:
            with httpx.Client(follow_redirects=False, timeout=self.server.request_timeout) as client:
                if method == "GET":
                    resp = client.get(url, headers=headers)
                else:
                    resp = client.post(url, headers=headers, content=body)
        except Exception as exc:
            self._write_log(method=method, path=self.path, status=502, error=str(exc))
            return self._send_json(502, {"error": f"upstream request failed: {exc}"})

        response_body = resp.content
        if len(response_body) > self.server.max_body_bytes:
            return self._deny(502, "upstream response too large")

        # x-omniroute-response-cost / x-omniroute-provider are this project's established,
        # verified-not-name-guessed way of knowing what a call actually cost and who served
        # it - logged here so cost/call-count accounting is broker-side (trusted), never
        # something the candidate code itself reports about its own behavior.
        self._write_log(
            method=method,
            path=self.path,
            status=resp.status_code,
            cost=resp.headers.get("x-omniroute-response-cost"),
            provider=resp.headers.get("x-omniroute-provider"),
        )

        self.send_response(resp.status_code)
        self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


def main():
    global REAL_API_KEY

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the broker's JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    REAL_API_KEY = os.environ["OMNIROUTE_API_KEY"]
    upstream_base_url = config.get("upstream_base_url") or os.environ["OMNIROUTE_BASE_URL"]
    upstream_root = _strip_v1_suffix(upstream_base_url)
    request_timeout = config.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)
    max_body_bytes = config.get("max_body_bytes", DEFAULT_MAX_BODY_BYTES)

    log_path = config["log_path"]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()
    search_tool = DuckDuckGoSearchRun()

    servers = []
    for role, socket_path in config["sockets"].items():
        socket_path = Path(socket_path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists():
            socket_path.unlink()  # start every cycle from a fresh socket, never a stale one

        budget_cfg = config["budgets"].get(role, {"model_calls": 0, "search_calls": 0})
        budget = Budget(budget_cfg.get("model_calls", 0), budget_cfg.get("search_calls", 0))

        server = UnixHTTPServer(
            str(socket_path),
            BrokerHandler,
            role,
            budget,
            upstream_root,
            request_timeout,
            max_body_bytes,
            log_lock,
            log_path,
            search_tool,
        )
        servers.append((server, socket_path))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"broker: role={role} listening on {socket_path}", flush=True)

    def _shutdown(signum, frame):
        for server, socket_path in servers:
            server.shutdown()
            server.server_close()
            try:
                socket_path.unlink()
            except FileNotFoundError:
                pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"broker: PID {os.getpid()} ready", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
