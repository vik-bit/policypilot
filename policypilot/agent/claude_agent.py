"""Step 3 — The agent, powered by Claude (Anthropic API).

Plain English: instead of stuffing everything into one prompt, we let the AI work like a person.
Each "turn" it can call a tool (search, read, submit). We run the tool, hand back the result, and
repeat until it submits a report that passes the fact-checker (or we run out of turns).

The shared parts (running tools, fact-checking) live in base.py; this file is only the loop in
Anthropic's message format.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic

from ..config import Settings
from ..models import Finding, Regulation, TraceEvent, VerificationStats
from ..retrieval import PolicyIndex
from .base import AgentError, BaseAgent
from .prompts import SYSTEM_PROMPT, TOOLS, user_message
from .tools import Emit

__all__ = ["AgentError", "ClaudeAgent"]

# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeAgent(BaseAgent):
    def __init__(self, settings: Settings, index: PolicyIndex, emit: Emit,
                 client: anthropic.Anthropic | None = None):
        super().__init__(settings, index, emit)
        self.client = client or self._make_client()

    @staticmethod
    def _make_client() -> anthropic.Anthropic:
        # An organisation-level key (not tied to a workspace) must name the workspace on every request.
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        headers = {"anthropic-workspace-id": workspace} if workspace else None
        return anthropic.Anthropic(default_headers=headers)

    def _call(self, messages: list[dict[str, Any]]):
        """One API call, with common account problems turned into messages a person can act on."""
        try:
            return self._call_api(messages)
        except anthropic.AuthenticationError as exc:
            raise AgentError("The API key was rejected. Check ANTHROPIC_API_KEY in your .env file.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise AgentError(f"This API key isn't allowed to use {self.settings.model}. "
                             "Check the key's permissions in the Anthropic Console.") from exc
        except anthropic.NotFoundError as exc:
            raise AgentError(f"Model '{self.settings.model}' wasn't found. Check PP_MODEL in your .env file.") from exc
        except anthropic.RateLimitError as exc:
            raise AgentError("The AI service is rate-limiting this key. Wait a minute and try again.") from exc
        except anthropic.BadRequestError as exc:
            if "workspace" in str(exc).lower():
                raise AgentError(
                    "This API key isn't tied to a workspace. Either create a key inside a workspace in the "
                    "Anthropic Console, or add ANTHROPIC_WORKSPACE_ID=<your workspace id> to your .env file, "
                    "then restart PolicyPilot.") from exc
            if "credit" in str(exc).lower() or "billing" in str(exc).lower():
                raise AgentError("The Anthropic account has no remaining credit. Add credit in the Console, "
                                 "or switch provider with PP_PROVIDER in your .env file.") from exc
            raise
        except anthropic.APIConnectionError as exc:
            raise AgentError("Couldn't reach the AI service. Check your internet connection.") from exc

    def _call_api(self, messages: list[dict[str, Any]]):
        kwargs: dict[str, Any] = dict(
            model=self.settings.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        if self.settings.effort != "none":
            kwargs["output_config"] = {"effort": self.settings.effort}
        if self.settings.refusal_fallback and self.settings.model in _FALLBACK_MODELS:
            response = self.client.beta.messages.create(betas=[_FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            response = self.client.messages.create(**kwargs)
        self.usage["turns"] += 1
        self.usage["input_tokens"] += response.usage.input_tokens
        self.usage["output_tokens"] += response.usage.output_tokens
        return response

    # ------------------------------------------------------------------ the loop
    def run(self, regulation: Regulation) -> tuple[str, list[Finding], VerificationStats]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message(regulation)}]
        stats = VerificationStats()
        summary, findings = "", []
        accepted = False
        nudges = 0

        while self.usage["turns"] < self.settings.max_turns:
            self.emit(TraceEvent(kind="llm", message=f"Thinking (turn {self.usage['turns'] + 1})…"))
            response = self._call(messages)
            # Append the full content (thinking + tool calls) — the API needs it on the next turn.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                raise AgentError(f"The model declined this request ({getattr(details, 'category', None)}).")

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                if nudges >= 2:
                    break
                nudges += 1
                cut = "Your reply was cut off. " if response.stop_reason == "max_tokens" else ""
                messages.append({"role": "user", "content": cut + "Please finish by calling submit_report."})
                continue

            results = []
            for block in tool_uses:
                content, is_error = self._run_tool(block.name, block.input, regulation, stats)
                if block.name == "submit_report":
                    summary = block.input.get("summary", "")
                    findings = self._last_findings
                    accepted = not is_error or stats.submissions > self.settings.max_verify_retries
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": content, "is_error": is_error})
            messages.append({"role": "user", "content": results})
            if accepted:
                break

        if not findings and not accepted:
            raise AgentError("The agent finished without submitting a report.")
        return summary, findings, stats
