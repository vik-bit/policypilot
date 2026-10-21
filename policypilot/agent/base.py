"""What every AI agent shares, whichever provider runs it.

Plain English: Claude and Groq speak slightly different "languages" (API formats), but the job is
the same. This file holds the common part: running the three tools and fact-checking a submitted
report. Each provider file (claude_agent.py, groq_agent.py) only adds its own conversation loop.
"""

from __future__ import annotations

import json

from ..config import Settings
from ..models import Finding, Regulation, TraceEvent, VerificationStats
from ..retrieval import PolicyIndex
from .tools import Emit, PolicyTools
from .verifier import citation_counts, verify

_FINDING_FIELDS = ("requirement_id", "clause_id", "impact", "explanation",
                   "regulation_quote", "policy_quote", "proposed_text")


class AgentError(RuntimeError):
    """A failure with a message written for the person using the app."""


class BaseAgent:
    def __init__(self, settings: Settings, index: PolicyIndex, emit: Emit):
        self.settings = settings
        self.index = index
        self.emit = emit
        self.tools = PolicyTools(index, emit)
        self.usage = {"input_tokens": 0, "output_tokens": 0, "turns": 0}
        self._last_findings: list[Finding] = []  # most recent verified submission

    def _run_tool(self, name: str, args: dict, regulation: Regulation,
                  stats: VerificationStats) -> tuple[str, bool]:
        """Run one tool call. Returns (result text, is_error)."""
        if name == "search_policies":
            return self.tools.search_policies(str(args.get("query", "")), args.get("k", 5)), False
        if name == "read_clause":
            return self.tools.read_clause(str(args.get("clause_id", "")))
        if name == "submit_report":
            return self._submit(args, regulation, stats)
        return f"Unknown tool {name}", True

    @staticmethod
    def _parse_findings(raw) -> tuple[list[Finding], list[str]]:
        """Turn the model's findings into Finding objects, tolerating small format slips."""
        findings, problems = [], []
        for i, f in enumerate(raw if isinstance(raw, list) else []):
            if not isinstance(f, dict):
                problems.append(f"finding #{i + 1} is not an object")
                continue
            data = {k: str(f.get(k) or "").strip() for k in _FINDING_FIELDS}
            data["impact"] = data["impact"].lower()
            try:
                findings.append(Finding(**data))
            except Exception:
                problems.append(f"finding #{i + 1} has impact '{data['impact']}'; use conflict, gap or aligned")
        return findings, problems

    def _submit(self, args: dict, regulation: Regulation, stats: VerificationStats) -> tuple[str, bool]:
        parsed, problems = self._parse_findings(args.get("findings", []))
        if problems and not parsed:
            return "Could not read your findings: " + "; ".join(problems) + ". Please resubmit.", True
        checked = verify(parsed, regulation, self.index)
        self._last_findings = checked
        stats.submissions += 1
        total, valid = citation_counts(checked)
        if stats.submissions == 1:
            stats.first_pass_citations, stats.first_pass_valid = total, valid
        stats.final_citations, stats.final_valid = total, valid

        failed = [f for f in checked if not f.verified]
        self.emit(TraceEvent(
            kind="verify",
            message=f"Fact-check #{stats.submissions}: {valid}/{total} citations verified"
                    + (f", {len(failed)} finding(s) sent back to fix" if failed else " ✓"),
            data={"valid": valid, "total": total, "failed": [f"{f.requirement_id}→{f.clause_id}" for f in failed]},
        ))
        if not failed and not problems:
            return "accepted", False
        if stats.submissions > self.settings.max_verify_retries:
            return "Recorded. Some findings remain unverified and will be flagged to the reviewer.", True
        errors = [{"requirement_id": f.requirement_id, "clause_id": f.clause_id, "errors": f.issues} for f in failed]
        return json.dumps({
            "status": "rejected",
            "message": "Fix these findings and call submit_report again with the COMPLETE list of findings.",
            "errors": errors,
            "format_problems": problems,
        }, ensure_ascii=False), True
