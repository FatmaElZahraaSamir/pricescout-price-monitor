from pathlib import Path

import pytest

from pricescout.mirror import MirrorServer, load_books

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_html():
    def _read(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    return _read


@pytest.fixture
def books():
    return load_books()


@pytest.fixture
def mirror():
    with MirrorServer() as server:
        yield server


class FakeOpenAI:
    """A tiny OpenAI-compatible server. `script` holds the assistant messages to return, in order."""

    def __init__(self):
        self.requests: list[dict] = []
        self.script: list[dict] = []
        self.status_queue: list[tuple[int, dict]] = []

    def handle(self, path: str, body: dict) -> tuple[int, dict]:
        self.requests.append({"path": path, "body": body})
        if self.status_queue:
            return self.status_queue.pop(0)
        message = self.script.pop(0) if self.script else {"role": "assistant", "content": "done"}
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        return 200, {"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body.get("model"),
                     "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


@pytest.fixture()
def fake_api():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = FakeOpenAI()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            status, payload = state.handle(self.path, json.loads(self.rfile.read(length) or b"{}"))
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    state.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield state
    server.shutdown()
    server.server_close()
