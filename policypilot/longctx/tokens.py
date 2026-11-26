"""Count prompt tokens with the model's own tokenizer, so the budget check is exact.

A wrong count is the dangerous failure: some servers (Ollama, for one) silently cut a prompt that is
longer than the window, and the model would then judge a library with clauses missing. So the
evaluation insists on the real tokenizer (`strict=True`), and only the app may fall back to a
deliberately pessimistic estimate when the tokenizer cannot be downloaded.
"""

from __future__ import annotations

import logging
import math
from typing import Protocol

log = logging.getLogger(__name__)

# Chat templates wrap every message in a few special tokens (Qwen: "<|im_start|>role\n" ... "<|im_end|>\n").
PER_MESSAGE_OVERHEAD = 8
# Generation prompt ("<|im_start|>assistant\n") plus slack for template differences between servers.
REPLY_PRIMER = 16
# The estimate is meant to over-count. The policy library averages 4.3 characters per Qwen token;
# 3.0 leaves a wide margin for dense text such as ids, numbers and legal citations.
CHARS_PER_TOKEN_ESTIMATE = 3.0


class TokenCounter(Protocol):
    name: str
    exact: bool

    def count(self, text: str) -> int: ...


class HFTokenCounter:
    """The model's tokenizer from the Hugging Face hub (needs the `tokenizers` package)."""

    exact = True

    def __init__(self, model: str):
        from tokenizers import Tokenizer

        self.name = model
        self._tok = Tokenizer.from_pretrained(model)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)


class EstimateCounter:
    """A pessimistic character-based estimate, for when the real tokenizer is not available."""

    exact = False
    name = f"estimate ({CHARS_PER_TOKEN_ESTIMATE} chars/token)"

    def count(self, text: str) -> int:
        return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def tokenizer_repo(model: str) -> str:
    """Quantised checkpoints share the base model's tokenizer: Qwen/Qwen2.5-7B-Instruct-AWQ ->
    Qwen/Qwen2.5-7B-Instruct. An Ollama tag such as qwen2.5:7b-instruct maps to the same repo."""
    if model.startswith("qwen2.5:"):
        size = model.split(":", 1)[1].split("-")[0].upper()
        return f"Qwen/Qwen2.5-{size}-Instruct"
    for suffix in ("-AWQ", "-GPTQ-Int4", "-GPTQ-Int8", "-GGUF"):
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def make_counter(model: str, strict: bool = False) -> TokenCounter:
    try:
        return HFTokenCounter(tokenizer_repo(model))
    except Exception as exc:  # no `tokenizers`, no network, unknown repo
        if strict:
            raise RuntimeError(f"Could not load the tokenizer for {model!r} ({exc}). The evaluation needs "
                               "exact token counts; install `tokenizers` and allow the download.") from exc
        log.warning("Tokenizer for %s unavailable (%s); using a pessimistic estimate", model, exc)
        return EstimateCounter()


def count_messages(counter: TokenCounter, messages: list[dict]) -> int:
    """Prompt tokens for a chat request, template overhead included."""
    return sum(counter.count(m["content"]) + PER_MESSAGE_OVERHEAD for m in messages) + REPLY_PRIMER
