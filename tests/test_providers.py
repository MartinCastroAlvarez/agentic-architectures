"""The HTTP adapters, against a local stub server on 127.0.0.1 (no real model, no internet)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from typing import Any

import pytest

from agentic_architectures.core.model import user
from agentic_architectures.core.providers import OllamaModel
from agentic_architectures.core.providers import OpenAICompatibleModel
from agentic_architectures.core.providers import ProviderError
from agentic_architectures.core.providers import SerializedModel
from agentic_architectures.core.providers import model_from_env
from agentic_architectures.toy import ToyModel


class Stub:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []
        self.status = 200
        self.reply: Any = {}


@pytest.fixture
def stub() -> Iterator[tuple[Stub, str]]:
    state = Stub()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append((self.path, dict(self.headers), body))
            payload = json.dumps(state.reply).encode()
            self.send_response(state.status)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join()


def test_ollama_adapter_speaks_api_chat(stub: tuple[Stub, str]) -> None:
    state, url = stub
    state.reply = {"message": {"content": '{"ok": 1}'}, "prompt_eval_count": 12, "eval_count": 3}
    model = SerializedModel(OllamaModel("qwen3:4b", url))
    reply = model.chat([user("hi")], label="router", json_mode=True)
    assert (reply.text, reply.prompt_tokens, reply.completion_tokens) == ('{"ok": 1}', 12, 3)
    path, _, body = state.requests[0]
    assert path == "/api/chat"
    assert body["format"] == "json" and body["stream"] is False and body["model"] == "qwen3:4b"
    state.reply = {"nope": True}
    with pytest.raises(ProviderError, match="no message.content"):
        model.chat([user("hi")])


def test_openai_compatible_adapter_sends_the_key_only_in_the_header(stub: tuple[Stub, str]) -> None:
    state, url = stub
    state.reply = {
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }
    model = OpenAICompatibleModel("m", url + "/v1", api_key="sk-test-not-real")
    assert model.chat([user("hi")], json_mode=True).text == "hello"
    path, headers, body = state.requests[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-test-not-real"
    assert body["response_format"] == {"type": "json_object"}
    assert "sk-test" not in repr(model)
    state.reply = {"choices": []}
    with pytest.raises(ProviderError, match="choices"):
        model.chat([user("hi")])


def test_http_errors_never_echo_the_body(stub: tuple[Stub, str]) -> None:
    state, url = stub
    state.status = 500
    state.reply = {"error": "Authorization: Bearer sk-leaked"}
    with pytest.raises(ProviderError) as info:
        OpenAICompatibleModel("m", url, api_key="sk-x").chat([user("hi")])
    assert "sk-" not in str(info.value)
    assert "HTTP 500" in str(info.value)


def test_unreachable_and_remote_hosts_are_refused() -> None:
    with pytest.raises(ProviderError, match="non-loopback"):
        OllamaModel(host="http://10.0.0.5:11434")
    OllamaModel(host="http://10.0.0.5:11434", allow_remote=True)
    with pytest.raises(ProviderError, match="https"):
        OpenAICompatibleModel("m", "http://api.example.com/v1")
    with pytest.raises(ProviderError, match="ConnectionRefusedError|OSError|TimeoutError"):
        OllamaModel(host="http://127.0.0.1:9", timeout=2).chat([user("hi")])
    with pytest.raises(ProviderError, match="unsupported URL"):
        OllamaModel(host="ftp://127.0.0.1").chat([user("hi")])


def test_model_from_env_picks_the_provider() -> None:
    assert isinstance(model_from_env({"AGENTIC_PROVIDER": "fake"}), ToyModel)
    assert model_from_env({}).name == "ollama:qwen3:4b"
    assert (
        model_from_env({"AGENTIC_PROVIDER": "ollama", "AGENTIC_MODEL": "llama3.2"}).name == "ollama:llama3.2"
    )
    assert model_from_env({"AGENTIC_PROVIDER": "openai"}).name == "openai-compatible:local-model"
    with pytest.raises(ProviderError, match="unknown"):
        model_from_env({"AGENTIC_PROVIDER": "paid-cloud"})
    with pytest.raises(ProviderError, match="non-loopback"):
        model_from_env({"OLLAMA_HOST": "http://203.0.113.9:11434"})
