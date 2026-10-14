"""One small client for every OpenAI-compatible LLM provider.

Google Gemini (free tier), OpenAI, Groq and a local Ollama server all speak the
OpenAI chat-completions protocol, including function calling, so switching
provider is a config change.

Robustness built in:
* friendly errors (bad key, free-tier limit reached, server down);
* automatic retry with back-off on rate limits / 5xx / dropped connections;
* if the configured model name no longer exists (model names change often),
  the client lists the provider's models and picks the closest match;
* if a "thinking" model spends its whole token budget before answering, the
  request is retried once with a bigger budget.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable

PROVIDERS: dict[str, dict] = {
    "gemini": {
        "label": "Google Gemini (free tier)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-flash-latest",
        "embed_model": "gemini-embedding-001",
        "env": "GEMINI_API_KEY",
        "key_url": "https://aistudio.google.com/apikey",
        "prefer": ["flash"],
    },
    "openai": {
        "label": "OpenAI",
        "base_url": None,
        "model": "gpt-4o-mini",
        "embed_model": "text-embedding-3-small",
        "env": "OPENAI_API_KEY",
        "key_url": "https://platform.openai.com/api-keys",
        "prefer": ["mini", "gpt"],
    },
    "groq": {
        "label": "Groq (free tier)",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "embed_model": None,
        "env": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
        "prefer": ["versatile", "instruct", "llama"],
    },
    "ollama": {
        "label": "Ollama (local, no key)",
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2",
        "embed_model": "nomic-embed-text",
        "env": None,
        "key_url": "https://ollama.com/download",
        "prefer": ["llama", "qwen", "mistral"],
    },
}

_MODEL_MISSING = re.compile(r"model.*(not found|not supported|does not exist|unknown|invalid)|"
                            r"(unknown|invalid|unsupported) model", re.I | re.S)
_BAD_KEY = re.compile(r"api[ _-]?key.*(invalid|not valid|expired|incorrect)|(invalid|incorrect) api[ _-]?key", re.I | re.S)


class LLMError(RuntimeError):
    """Raised with a message that can be shown to an end user as-is."""


@dataclass
class ToolRequest:
    id: str
    name: str
    arguments: str  # JSON text, exactly as the model sent it


@dataclass
class ModelTurn:
    """One model reply: either tool calls to run, or final text."""
    content: str
    tool_calls: list[ToolRequest]
    message: dict  # the assistant message to append to the conversation


@dataclass
class LLMClient:
    provider: str = "gemini"
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout: float = 60.0
    max_attempts: int = 3
    backoff: float = 2.0  # seconds before the first retry, doubled each time
    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise LLMError(f"Unknown provider '{self.provider}'. Choose from: {', '.join(PROVIDERS)}")
        cfg = PROVIDERS[self.provider]
        self.model = (self.model or cfg["model"]).strip()
        self.base_url = self.base_url or cfg["base_url"]
        if self.api_key is None and cfg["env"]:
            self.api_key = os.environ.get(cfg["env"])
        self.api_key = (self.api_key or "").strip() or None
        if self.provider == "ollama":
            self.api_key = self.api_key or "ollama"  # Ollama ignores the key but the SDK needs one
        if not self.api_key:
            raise LLMError(f"No API key for {cfg['label']}. Get a free key at {cfg['key_url']}")

    @property
    def label(self) -> str:
        return PROVIDERS[self.provider]["label"]

    # ---------------------------------------------------------------- plumbing
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key, "timeout": self.timeout, "max_retries": 0}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _call(self, fn, **kwargs):
        import openai

        delay = self.backoff
        model_fixed = False
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn(**kwargs)
            except openai.AuthenticationError as exc:
                raise LLMError("The API key was rejected. Check that you copied the whole key.") from exc
            except openai.RateLimitError as exc:
                if attempt == self.max_attempts:
                    raise LLMError("Rate limit reached (free tiers allow a limited number of requests per minute "
                                   "and per day). Wait a minute and try again.") from exc
                time.sleep(_retry_after(exc) or delay)
                delay *= 2
            except openai.APIConnectionError as exc:  # includes timeouts
                if attempt == self.max_attempts:
                    hint = " Is Ollama running?" if self.provider == "ollama" else " Check your internet connection."
                    raise LLMError(f"Could not reach {self.label}.{hint}") from exc
                time.sleep(delay)
                delay *= 2
            except openai.APIStatusError as exc:
                message = _error_message(exc)
                if exc.status_code >= 500 and attempt < self.max_attempts:
                    time.sleep(delay)
                    delay *= 2
                    continue
                if _BAD_KEY.search(message):  # Gemini answers 400 "API key not valid" instead of 401
                    raise LLMError("The API key was rejected. Check that you copied the whole key.") from exc
                model_missing = exc.status_code == 404 or (exc.status_code == 400 and _MODEL_MISSING.search(message))
                if model_missing and "model" in kwargs and not model_fixed:
                    better = self.discover_model()
                    if better and better != kwargs["model"]:
                        self.model = kwargs["model"] = better
                        model_fixed = True
                        continue
                    raise LLMError(f"Model '{kwargs['model']}' is not available at {self.label}. "
                                   "Enter another model name in the settings.") from exc
                if exc.status_code == 403:
                    raise LLMError(f"{self.label} refused the request (403): {message}") from exc
                raise LLMError(f"{self.label} returned an error ({exc.status_code}): {message}") from exc
        raise LLMError("The request failed after several attempts.")

    # ---------------------------------------------------------------- public
    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024) -> str:
        for budget in (max_tokens, min(max_tokens * 4, 8192)):
            response = self._call(self.client.chat.completions.create, model=self.model, messages=messages,
                                  temperature=temperature, max_tokens=budget)
            choice = response.choices[0] if response.choices else None
            content = (choice.message.content or "").strip() if choice and choice.message else ""
            if content:
                return content
            if not choice or choice.finish_reason != "length":
                break  # empty for another reason: a bigger budget will not help
        raise LLMError("The model returned an empty answer. Try again or pick another model.")

    def chat_tools(self, messages: list[dict], tools: list[dict], temperature: float = 0.0,
                   max_tokens: int = 2048) -> ModelTurn:
        """Chat with function calling (OpenAI format, supported by Gemini, OpenAI, Groq and Ollama)."""
        for budget in (max_tokens, min(max_tokens * 4, 8192)):
            response = self._call(self.client.chat.completions.create, model=self.model, messages=messages,
                                  tools=tools, tool_choice="auto", temperature=temperature, max_tokens=budget)
            choice = response.choices[0] if response.choices else None
            if choice is None or choice.message is None:
                break
            turn = to_turn(choice.message, offset=len(messages))
            if turn.content or turn.tool_calls:
                return turn
            if choice.finish_reason != "length":
                break
        raise LLMError("The model returned an empty answer. Try again or pick another model.")

    def embed(self, texts: list[str]):
        import numpy as np

        model = PROVIDERS[self.provider]["embed_model"]
        if not model:
            raise LLMError(f"{self.label} has no embedding model configured.")
        vectors = []
        for start in range(0, len(texts), 96):  # stay under per-request batch limits
            response = self._call(self.client.embeddings.create, model=model, input=texts[start:start + 96])
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return np.asarray(vectors, dtype="float32")

    def discover_model(self) -> str | None:
        """Pick an available chat model that matches the provider's preferences."""
        try:
            ids = [m.id.split("/")[-1] for m in self.client.models.list()]
        except Exception:
            return None
        ids = [i for i in ids if not re.search(r"embed|vision|tts|whisper|audio|image|guard|moderation", i)]
        for keyword in PROVIDERS[self.provider]["prefer"]:
            matches = sorted((i for i in ids if keyword in i), key=_version_key, reverse=True)
            if matches:
                return matches[0]
        return ids[0] if ids else None


def to_turn(message, offset: int = 0) -> ModelTurn:
    """Normalise an SDK message. Provider extras (e.g. Gemini's thought signatures in `extra_content`)
    are kept on the echoed message, because some models require them in the next request."""
    calls, echoed = [], []
    for i, tool_call in enumerate(message.tool_calls or []):
        call_id = tool_call.id or f"call_{offset}_{i}"
        calls.append(ToolRequest(call_id, tool_call.function.name, tool_call.function.arguments or "{}"))
        item = {"id": call_id, "type": "function",
                "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments or "{}"}}
        extra = (getattr(tool_call, "model_extra", None) or {}).get("extra_content")
        if extra:
            item["extra_content"] = extra
        echoed.append(item)
    assistant = {"role": "assistant", "content": message.content or ""}
    if echoed:
        assistant["tool_calls"] = echoed
    extra = (getattr(message, "model_extra", None) or {}).get("extra_content")
    if extra:
        assistant["extra_content"] = extra
    return ModelTurn((message.content or "").strip(), calls, assistant)


def _version_key(model_id: str):
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", model_id)]
    stable = 0 if re.search(r"preview|exp|beta", model_id) else 1
    return (stable, numbers)


def _retry_after(exc) -> float | None:
    try:
        value = exc.response.headers.get("retry-after")
        return min(30.0, float(value)) if value else None
    except Exception:
        return None


def _error_message(exc) -> str:
    """Short, human-readable message from an API error body."""
    body = getattr(exc, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        error = body.get("error", body)
        message = error.get("message") if isinstance(error, dict) else str(error)
    else:
        message = getattr(exc, "message", "") or str(exc)
    return re.sub(r"\s+", " ", str(message or "")).strip()[:300]


Reply = str | dict | Callable[[list[dict]], "str | dict"]


class ScriptedLLM:
    """Deterministic stand-in used by tests and demos: returns queued replies in order.

    A reply is a string (final text), a dict {"tool_calls": [{"name": ..., "arguments": {...}}]}
    or a function that receives the messages and returns one of those.
    """

    label = "Scripted LLM"
    provider = "scripted"

    def __init__(self, replies: list[Reply] | None = None, default: Reply = "OK"):
        self.replies = list(replies or [])
        self.default = default
        self.calls: list[list[dict]] = []
        self.model = "scripted"

    def _next(self, messages: list[dict]):
        self.calls.append([dict(m) for m in messages])
        reply = self.replies.pop(0) if self.replies else self.default
        return reply(messages) if callable(reply) else reply

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024) -> str:
        reply = self._next(messages)
        return reply if isinstance(reply, str) else json.dumps(reply)

    def chat_tools(self, messages: list[dict], tools: list[dict], temperature: float = 0.0,
                   max_tokens: int = 2048) -> ModelTurn:
        reply = self._next(messages)
        if isinstance(reply, str):
            return ModelTurn(reply, [], {"role": "assistant", "content": reply})
        calls = [ToolRequest(c.get("id") or f"call_{len(self.calls)}_{i}", c["name"],
                             c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"]))
                 for i, c in enumerate(reply.get("tool_calls", []))]
        message = {"role": "assistant", "content": reply.get("content", ""),
                   "tool_calls": [{"id": c.id, "type": "function",
                                   "function": {"name": c.name, "arguments": c.arguments}} for c in calls]}
        return ModelTurn(reply.get("content", ""), calls, message)
