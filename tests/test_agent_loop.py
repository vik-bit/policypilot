"""Runs the real Claude agent loop against a scripted fake API (no key, no network).

The fake sits *under* the official SDK (an httpx transport), so these tests also prove the
requests we build are accepted by the SDK and that its responses are handled correctly.
"""

import json

import anthropic
import httpx2 as httpx  # the anthropic 1.x SDK uses httpx2

from policypilot.ingest import load_regulation
from policypilot.pipeline import analyze


def _msg(content, stop_reason="tool_use"):
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


def _tool(tid, name, args):
    return {"type": "tool_use", "id": tid, "name": name, "input": args}


GOOD = {
    "requirement_id": "R2", "clause_id": "data-privacy§3.2", "impact": "conflict",
    "explanation": "Policy keeps data ten years; the rule caps it at three.",
    "regulation_quote": "Personal data shall not be kept for longer than three (3) years",
    "policy_quote": "Customer personal data is retained for ten (10) years after the account is closed",
    "proposed_text": "Customer personal data is retained for no longer than three (3) years after the account is closed.",
}
BAD = {**GOOD, "policy_quote": "Customer data is kept for a decade"}  # paraphrase: must be rejected


def _client(script, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        return httpx.Response(200, json=script[len(seen) - 1])

    return anthropic.Anthropic(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_agent_searches_gets_fact_checked_and_fixes_its_quote(settings, index):
    reg = load_regulation(settings.regulations_dir / "reg-01-data-protection.md")
    script = [
        _msg([_tool("t1", "search_policies", {"query": "data retention period", "k": 5})]),
        _msg([_tool("t2", "read_clause", {"clause_id": "data-privacy§3.2"})]),
        _msg([_tool("t3", "submit_report", {"summary": "s", "findings": [BAD]})]),
        _msg([_tool("t4", "submit_report", {"summary": "Retention must drop to 3 years.", "findings": [GOOD]})]),
    ]
    seen, events = [], []
    report = analyze(reg, index, settings, emit=events.append, client=_client(script, seen))

    # The loop made exactly four calls and stopped once the report was accepted.
    assert len(seen) == 4
    assert report.mode == "claude"
    assert [f.clause_id for f in report.findings] == ["data-privacy§3.2"]
    assert report.findings[0].verified
    # First submission had one bad policy quote; the retry fixed it.
    assert report.stats.submissions == 2
    assert report.stats.first_pass_valid == 1 and report.stats.first_pass_citations == 2
    assert report.stats.final_valid == report.stats.final_citations == 2

    # The rejected tool result told the model what was wrong and showed the real sentence.
    rejection = seen[3]["body"]["messages"][-1]["content"][0]
    assert rejection["is_error"] is True and "Closest actual sentence" in rejection["content"]

    # Request shape: strict tools, effort, and the refusal fallback beta.
    body = seen[0]["body"]
    assert all(t["strict"] for t in body["tools"])
    assert body["output_config"] == {"effort": settings.effort}
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in seen[0]["headers"]["anthropic-beta"]
    assert any(e.kind == "search" for e in events) and any(e.kind == "verify" for e in events)


def test_agent_is_nudged_when_it_answers_in_plain_text(settings, index):
    reg = load_regulation(settings.regulations_dir / "reg-01-data-protection.md")
    script = [
        _msg([{"type": "text", "text": "I think clause 3.2 is affected."}], stop_reason="end_turn"),
        _msg([_tool("t1", "submit_report", {"summary": "s", "findings": [GOOD]})]),
    ]
    seen = []
    report = analyze(reg, index, settings, client=_client(script, seen))
    assert len(seen) == 2
    assert "submit_report" in seen[1]["body"]["messages"][-1]["content"]
    assert report.findings[0].verified


def test_unfixable_findings_are_flagged_not_hidden(settings, index):
    reg = load_regulation(settings.regulations_dir / "reg-01-data-protection.md")
    bad_submit = lambda i: _msg([_tool(f"t{i}", "submit_report", {"summary": "s", "findings": [BAD]})])
    script = [bad_submit(i) for i in range(1, 5)]
    seen = []
    report = analyze(reg, index, settings, client=_client(script, seen))
    assert len(seen) == settings.max_verify_retries + 1
    assert not report.findings[0].verified and report.findings[0].issues
