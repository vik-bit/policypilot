"""The model behind long-context mode: any OpenAI-compatible chat server.

That covers the free, open-weight setups this mode is built for: vLLM on a Kaggle or Colab GPU
(`vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ`) and Ollama on a laptop (`http://localhost:11434/v1`).
No paid API is needed and no key is stored; set PP_LC_API_KEY only if your server asks for one.
Tests use a scripted fake with the same `chat` method.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Reply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # prompt tokens the server served from its prefix cache, when it reports them (vLLM does)
    cached_tokens: int = 0
    finish_reason: str = ""
    seconds: float = 0.0


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[dict], max_tokens: int, schema: dict | None = None) -> Reply: ...


class LLMError(RuntimeError):
    pass


@dataclass
class OpenAICompatModel:
    base_url: str = field(default_factory=lambda: os.environ.get("PP_LC_BASE_URL", "http://localhost:8000/v1"))
    name: str = field(default_factory=lambda: os.environ.get("PP_LC_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ"))
    temperature: float = 0.0
    seed: int | None = None
    timeout: float = 900.0
    # "json_schema" (vLLM, Ollama >= 0.5), "json_object", or "none"
    structured: str = field(default_factory=lambda: os.environ.get("PP_LC_STRUCTURED", "json_schema"))

    def chat(self, messages: list[dict], max_tokens: int, schema: dict | None = None) -> Reply:
        body: dict = {"model": self.name, "messages": messages, "max_tokens": max_tokens,
                      "temperature": self.temperature}
        if self.seed is not None:
            body["seed"] = self.seed
        if schema is not None and self.structured == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "findings", "schema": schema}}
        elif schema is not None and self.structured == "json_object":
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if os.environ.get("PP_LC_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['PP_LC_API_KEY']}"
        request = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise LLMError(f"{self.base_url} answered HTTP {exc.code}: {exc.read()[:500]!r}") from exc
        except OSError as exc:
            raise LLMError(f"Could not reach {self.base_url}: {exc}") from exc
        seconds = time.perf_counter() - started
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return Reply(text=choice["message"].get("content") or "", prompt_tokens=usage.get("prompt_tokens", 0),
                     completion_tokens=usage.get("completion_tokens", 0),
                     cached_tokens=(details.get("cached_tokens") or 0), finish_reason=choice.get("finish_reason", ""),
                     seconds=seconds)


def parse_json_reply(text: str) -> dict | None:
    """The reply as a dict, tolerating code fences or a sentence around the JSON object."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
