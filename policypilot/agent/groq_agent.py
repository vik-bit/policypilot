"""Step 3 — The agent, powered by Groq (open-weight models such as GPT-OSS and Llama).

Plain English: the same agent as claude_agent.py — search, read, submit, get fact-checked, fix —
but talking to Groq's API, which uses the OpenAI "chat completions" format. Tool calls arrive as
`tool_calls` on the assistant message, and results go back as messages with role "tool".
"""

from __future__ import annotations

import json
from typing import Any

import groq

from ..config import Settings
from ..models import Finding, Regulation, TraceEvent, VerificationStats
from ..retrieval import PolicyIndex
from .base import AgentError, BaseAgent
from .prompts import SYSTEM_PROMPT, TOOLS, user_message
from .tools import Emit

# Our tools are written in Anthropic's shape; Groq wants {"type": "function", "function": {...}}.
GROQ_TOOLS = [
    {"type": "function", "function": {"name": t["name"], "description": t["description"],
                                      "parameters": t["input_schema"]}}
    for t in TOOLS
]
_RETRYABLE = ("tool_use_failed", "output_parse_failed")
_MAX_RETRIES = 4
_SYSTEM = SYSTEM_PROMPT + (
    "\nAlways act through the tools: call search_policies and read_clause as many times as you need, "
    "then call submit_report. Do not write the report as plain text."
)


class GroqAgent(BaseAgent):
    def __init__(self, settings: Settings, index: PolicyIndex, emit: Emit, client: groq.Groq | None = None):
        super().__init__(settings, index, emit)
        if client is None and not settings.has_groq_key:
            raise AgentError("Groq is selected but no key is set. Add GROQ_API_KEY=<your key> to the .env file "
                             "(get one free at console.groq.com), then restart PolicyPilot.")
        # Generous retries: Groq's free tier rate-limits by tokens per minute, and the SDK waits
        # for the server's retry-after delay between attempts.
        self.client = client or groq.Groq(max_retries=4, timeout=120)

    @property
    def model(self) -> str:
        return self.settings.groq_model

    def _call(self, messages: list[dict[str, Any]]):
        kwargs: dict[str, Any] = dict(
            model=self.model, messages=messages, tools=GROQ_TOOLS, tool_choice="auto",
            temperature=0.2, max_completion_tokens=8192,
        )
        if self.model.startswith("openai/gpt-oss") and self.settings.effort in ("low", "medium", "high"):
            kwargs["reasoning_effort"] = self.settings.effort
        try:
            response = self.client.chat.completions.create(**kwargs)
        except groq.AuthenticationError as exc:
            raise AgentError("The Groq API key was rejected. Check GROQ_API_KEY in your .env file.") from exc
        except groq.NotFoundError as exc:
            raise AgentError(f"Groq model '{self.model}' wasn't found. Check PP_GROQ_MODEL in your .env file.") from exc
        except groq.RateLimitError as exc:
            raise AgentError("Groq is rate-limiting this key (free plans allow a limited number of tokens per "
                             "minute). Wait a minute and try again, or choose a smaller model.") from exc
        except groq.APIConnectionError as exc:
            raise AgentError("Couldn't reach Groq. Check your internet connection.") from exc
        self.usage["turns"] += 1
        if response.usage:
            self.usage["input_tokens"] += response.usage.prompt_tokens or 0
            self.usage["output_tokens"] += response.usage.completion_tokens or 0
        return response

    # ------------------------------------------------------------------ the loop
    def run(self, regulation: Regulation) -> tuple[str, list[Finding], VerificationStats]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_message(regulation)},
        ]
        stats = VerificationStats()
        summary, findings = "", []
        accepted = False
        nudges = retries = 0

        while self.usage["turns"] < self.settings.max_turns:
            self.emit(TraceEvent(kind="llm", message=f"Thinking (turn {self.usage['turns'] + 1})…"))
            try:
                response = self._call(messages)
            except groq.BadRequestError as exc:
                # Open models occasionally produce a reply Groq can't parse: a malformed tool call
                # ("tool_use_failed") or reasoning text leaking into the answer ("output_parse_failed").
                # Ask the model to carry on rather than failing the run.
                if any(code in str(exc) for code in _RETRYABLE) and retries < _MAX_RETRIES:
                    retries += 1
                    self.usage["turns"] += 1
                    self.emit(TraceEvent(kind="llm", message="The model's reply couldn't be read; asking it to try again"))
                    messages.append({"role": "user", "content":
                                     "Your last reply could not be processed: it was not valid JSON. Continue the task "
                                     "by calling one tool with valid JSON arguments (search_policies, read_clause or "
                                     "submit_report). Do not write any other text."})
                    continue
                raise AgentError(f"Groq rejected the request: {exc}") from exc

            choice = response.choices[0]
            msg = choice.message
            tool_calls = msg.tool_calls or []
            assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if tool_calls:
                assistant["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ]
            messages.append(assistant)

            if not tool_calls:
                if nudges >= 3:
                    break
                nudges += 1
                cut = "Your reply was cut off. " if choice.finish_reason == "length" else ""
                messages.append({"role": "user", "content": cut + "Please finish by calling the submit_report tool."})
                continue

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    content, is_error = "Your arguments were not valid JSON. Call the tool again.", True
                else:
                    content, is_error = self._run_tool(tc.function.name, args, regulation, stats)
                    if tc.function.name == "submit_report":
                        summary = str(args.get("summary", ""))
                        findings = self._last_findings
                        accepted = not is_error or stats.submissions > self.settings.max_verify_retries
                # The chat format has no error flag on tool results, so we say it in the text.
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": ("ERROR: " + content) if is_error else content})
            if accepted:
                break

        if not findings and not accepted:
            raise AgentError("The agent finished without submitting a report. Try again, or try a larger model.")
        return summary, findings, stats
