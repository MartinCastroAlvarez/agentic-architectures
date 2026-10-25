"""Model adapters over plain HTTP (standard library only).

* ``OllamaModel`` (the default): ``POST /api/chat`` on a local ollama. The host must
  be loopback unless ``AGENTIC_ALLOW_REMOTE=1``, so prompts never leave the machine by
  accident.
* ``OpenAICompatibleModel`` (optional): ``POST /chat/completions`` on any
  OpenAI-compatible server (vLLM, llama.cpp ``server``, LM Studio, or a hosted API with
  your own key). The key is read from the environment and never logged.

Neither adapter is used by the tests, the examples or the benchmark: they use
:class:`agentic_architectures.core.fake.ScriptedModel`.
"""

from __future__ import annotations

import http.client
import json
import os
import threading
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from agentic_architectures.core.model import Message
from agentic_architectures.core.model import Reply
from agentic_architectures.core.model import estimate_tokens

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class ProviderError(RuntimeError):
    """The provider could not be reached or answered with something unusable."""


def _post_json(url: str, body: Mapping[str, Any], headers: Mapping[str, str], timeout: float) -> Any:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ProviderError(f"unsupported URL: {parts.scheme}://{parts.hostname}")
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parts.hostname, parts.port, timeout=timeout)
    path = parts.path or "/"
    try:
        conn.request(
            "POST",
            path,
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
        )
        response = conn.getresponse()
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ProviderError("response too large")
        if response.status >= 400:
            # Never echo the body: some servers reflect the request (and its headers) back.
            raise ProviderError(f"HTTP {response.status} from {parts.hostname}")
        return json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise ProviderError(f"{type(exc).__name__} talking to {parts.hostname}") from exc
    finally:
        conn.close()


def check_loopback(url: str, *, allow_remote: bool) -> None:
    host = urlsplit(url).hostname or ""
    if host not in LOOPBACK and not allow_remote:
        raise ProviderError(
            f"refusing non-loopback model host {host!r}; set AGENTIC_ALLOW_REMOTE=1 to allow it"
        )


class OllamaModel:
    """ollama's native chat API, non-streaming."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        *,
        temperature: float = 0.0,
        timeout: float = 120.0,
        allow_remote: bool = False,
    ) -> None:
        check_loopback(host, allow_remote=allow_remote)
        self.name = f"ollama:{model}"
        self.model = model
        self.url = host.rstrip("/") + "/api/chat"
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: Sequence[Message], *, label: str = "", json_mode: bool = False) -> Reply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            body["format"] = "json"
        data = _post_json(self.url, body, {}, self.timeout)
        try:
            text = str(data["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise ProviderError("ollama reply has no message.content") from exc
        prompt = sum(estimate_tokens(m.content) for m in messages)
        return Reply(
            text,
            int(data.get("prompt_eval_count") or prompt),
            int(data.get("eval_count") or estimate_tokens(text)),
        )


class OpenAICompatibleModel:
    """Any server that speaks the OpenAI chat-completions wire format."""

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        parts = urlsplit(base_url)
        if parts.scheme != "https" and (parts.hostname or "") not in LOOPBACK:
            raise ProviderError("a non-loopback OpenAI-compatible server must use https")
        self.name = f"openai-compatible:{model}"
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def __repr__(self) -> str:  # never show the key
        return f"OpenAICompatibleModel(model={self.model!r}, url={self.url!r})"

    def chat(self, messages: Sequence[Message], *, label: str = "", json_mode: bool = False) -> Reply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        data = _post_json(self.url, body, headers, self.timeout)
        try:
            text = str(data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("reply has no choices[0].message.content") from exc
        usage = data.get("usage") or {}
        return Reply(
            text,
            int(usage.get("prompt_tokens") or sum(estimate_tokens(m.content) for m in messages)),
            int(usage.get("completion_tokens") or estimate_tokens(text)),
        )


class SerializedModel:
    """Wrap a model so only one request is in flight (local servers often prefer that)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = inner.name
        self._lock = threading.Lock()

    def chat(self, messages: Sequence[Message], *, label: str = "", json_mode: bool = False) -> Reply:
        with self._lock:
            reply: Reply = self.inner.chat(messages, label=label, json_mode=json_mode)
            return reply


def model_from_env(env: Mapping[str, str] | None = None) -> Any:
    """Build the model named by ``AGENTIC_PROVIDER`` (``ollama`` | ``openai`` | ``fake``)."""
    env = os.environ if env is None else env
    provider = env.get("AGENTIC_PROVIDER", "ollama").strip().lower()
    temperature = float(env.get("AGENTIC_TEMPERATURE", "0") or 0)
    if provider == "ollama":
        return OllamaModel(
            env.get("AGENTIC_MODEL") or DEFAULT_OLLAMA_MODEL,
            env.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST,
            temperature=temperature,
            allow_remote=env.get("AGENTIC_ALLOW_REMOTE") == "1",
        )
    if provider == "openai":
        base = env.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
        return OpenAICompatibleModel(
            env.get("AGENTIC_MODEL") or "local-model",
            base,
            api_key=env.get("OPENAI_API_KEY") or None,
            temperature=temperature,
        )
    if provider == "fake":
        from agentic_architectures.toy import ToyModel

        return ToyModel(latency=float(env.get("AGENTIC_FAKE_LATENCY", "0") or 0))
    raise ProviderError(f"unknown AGENTIC_PROVIDER {provider!r} (ollama | openai | fake)")
