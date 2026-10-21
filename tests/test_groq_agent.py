"""Runs the Groq agent loop against a scripted fake API (no key, no network).

Like test_agent_loop.py, the fake sits under the official groq SDK, so these tests also check that
the requests we build are accepted by the SDK and that its responses are parsed correctly.
"""

import json

import groq
import httpx

from policypilot.config import Settings
from policypilot.ingest import load_regulation
from policypilot.pipeline import analyze

GOOD = {
    "requirement_id": "R2", "clause_id": "data-privacy§3.2", "impact": "Conflict",  # capitalised on purpose
    "explanation": "Policy keeps data ten years; the rule caps it at three.",
    "regulation_quote": "Personal data shall not be kept for longer than three (3) years",
    "policy_quote": "Customer personal data is retained for ten (10) years after the account is closed",
    "proposed_text": "Customer personal data is retained for no longer than three (3) years after the account is closed.",
}
BAD = {**GOOD, "policy_quote": "Customer data is kept for a decade"}


def _completion(tool_calls=None, content=None, finish="tool_calls"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(tool_calls)
        ]
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "openai/gpt-oss-120b",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
    }


def _client(script, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        status, body = script[len(seen) - 1]
        return httpx.Response(status, json=body)

    return groq.Groq(api_key="test-key", max_retries=0,
                     http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _reg(settings):
    return load_regulation(settings.regulations_dir / "reg-01-data-protection.md")


def test_groq_agent_searches_gets_fact_checked_and_fixes_its_quote(settings, index):
    script = [
        (200, _completion([("search_policies", {"query": "data retention period", "k": 5})])),
        (200, _completion([("read_clause", {"clause_id": "data-privacy§3.2"})])),
        (200, _completion([("submit_report", {"summary": "s", "findings": [BAD]})])),
        (200, _completion([("submit_report", {"summary": "Retention must drop to 3 years.", "findings": [GOOD]})])),
    ]
    seen = []
    report = analyze(_reg(settings), index, settings, client=_client(script, seen))

    assert len(seen) == 4
    assert report.mode == "groq" and report.model == settings.groq_model
    [f] = report.findings
    assert f.clause_id == "data-privacy§3.2" and f.impact == "conflict" and f.verified
    assert report.stats.first_pass_valid == 1 and report.stats.final_valid == 2
    assert report.usage["turns"] == 4 and report.usage["input_tokens"] == 480

    # Request shape: OpenAI-style function tools, and tool results sent back with role "tool".
    body = seen[0]
    assert body["model"] == settings.groq_model and body["tool_choice"] == "auto"
    assert {t["function"]["name"] for t in body["tools"]} == {"search_policies", "read_clause", "submit_report"}
    rejection = seen[3]["messages"][-1]
    assert rejection["role"] == "tool" and rejection["content"].startswith("ERROR:")
    assert "Closest actual sentence" in rejection["content"]


def test_groq_agent_recovers_from_a_malformed_tool_call(settings, index):
    failed = {"error": {"message": "Failed to call a function. Please adjust your prompt.",
                        "type": "invalid_request_error", "code": "tool_use_failed"}}
    script = [
        (400, failed),
        (200, _completion(content="Here is my analysis in prose.", finish="stop")),
        (200, _completion([("submit_report", {"summary": "s", "findings": [GOOD]})])),
    ]
    seen = []
    report = analyze(_reg(settings), index, settings, client=_client(script, seen))
    assert len(seen) == 3
    assert "not valid JSON" in seen[1]["messages"][-1]["content"]  # retry nudge after the 400
    assert "submit_report" in seen[2]["messages"][-1]["content"]
    assert report.findings[0].verified


def test_provider_setting(monkeypatch):
    for var in ("PP_PROVIDER", "PP_MODE", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert Settings().resolved_mode == "offline"
    monkeypatch.setenv("GROQ_API_KEY", "g")
    assert Settings().resolved_mode == "groq"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert Settings().resolved_mode == "claude"  # auto prefers Claude when both keys exist
    monkeypatch.setenv("PP_PROVIDER", "groq")
    s = Settings()
    assert s.resolved_mode == "groq" and s.active_model == s.groq_model
