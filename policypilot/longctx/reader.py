"""The long-context reader: one requirement, one prompt holding the library, fact-checked replies.

    requirement ──► context (whole library, or search results if it does not fit) ──► model ──► JSON
                                                                                         │
                         fix these quotes (same fact-checker as the agent) ◄─────────────┘

Unlike the agent there are no tools to call: the model already sees every clause it may cite. The
fact-check loop is the same: quotes are checked word for word, and failures go back to the model
with the closest real sentence, up to PP_MAX_VERIFY_RETRIES times.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from ..agent.base import BaseAgent
from ..agent.verifier import citation_counts, verify
from ..config import Settings
from ..models import Finding, Regulation, Report, Requirement, TraceEvent, VerificationStats
from ..retrieval import PolicyIndex
from .context import Budget, ContextPlan, build_context
from .llm import ChatModel, parse_json_reply
from .prompts import RESPONSE_SCHEMA, retry_message
from .tokens import TokenCounter, count_messages


@dataclass
class RequirementResult:
    requirement_id: str
    findings: list[Finding] = field(default_factory=list)
    first_total: int = 0
    first_valid: int = 0
    final_total: int = 0
    final_valid: int = 0
    calls: int = 0
    prompt_tokens: int = 0
    # as the server counted the first prompt; Ollama leaves out tokens it served from its cache
    first_prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    seconds: float = 0.0
    parse_failures: int = 0
    # why the loop ended: "accepted", "retries_exhausted", "budget" (a retry would overflow the window)
    stopped: str = ""


def analyze_requirement(regulation: Regulation, req: Requirement, plan: ContextPlan, model: ChatModel,
                        index: PolicyIndex, budget: Budget, counter: TokenCounter) -> RequirementResult:
    result = RequirementResult(requirement_id=req.id)
    messages = list(plan.messages)
    started = time.perf_counter()
    for attempt in range(budget.retries + 1):
        reply = model.chat(messages, budget.max_output, RESPONSE_SCHEMA)
        result.calls += 1
        if attempt == 0:
            result.first_prompt_tokens = reply.prompt_tokens
        result.prompt_tokens += reply.prompt_tokens
        result.completion_tokens += reply.completion_tokens
        result.cached_tokens += reply.cached_tokens

        parsed = parse_json_reply(reply.text)
        problems: list[str] = []
        if parsed is None:
            result.parse_failures += 1
            problems.append("the reply was not a JSON object with a findings list"
                            + (" (it was cut off at the token limit)" if reply.finish_reason == "length" else ""))
            checked = result.findings  # keep the last readable answer
        else:
            raw = parsed.get("findings", [])
            raw = [{**f, "requirement_id": req.id} if isinstance(f, dict) else f for f in (raw if isinstance(raw, list) else [])]
            findings, problems = BaseAgent._parse_findings(raw)
            checked = verify(findings, regulation, index)
            result.findings = checked
        total, valid = citation_counts(result.findings)
        if attempt == 0:
            result.first_total, result.first_valid = total, valid
        result.final_total, result.final_valid = total, valid

        failed = [f for f in checked if not f.verified]
        if parsed is not None and not failed and not problems:
            result.stopped = "accepted"
            break
        if attempt == budget.retries:
            result.stopped = "retries_exhausted"
            break
        errors = [{"clause_id": f.clause_id, "errors": f.issues} for f in failed]
        follow_up = messages + [{"role": "assistant", "content": reply.text},
                                {"role": "user", "content": retry_message(errors, problems)}]
        if count_messages(counter, follow_up) + budget.max_output > budget.window:
            result.stopped = "budget"
            break
        messages = follow_up
    result.seconds = time.perf_counter() - started
    return result


def analyze_long(regulation: Regulation, index: PolicyIndex, settings: Settings, model: ChatModel,
                 counter: TokenCounter, budget: Budget | None = None, variant: str = "auto",
                 emit=None, save: bool = True) -> Report:
    """A full report in long-context mode, in the same shape as the agent's, so the web app and
    the evaluation can read it."""
    from ..pipeline import save_report  # late import: pipeline imports the agents

    emit = emit or (lambda e: None)
    budget = budget or Budget(retries=settings.max_verify_retries)
    emit(TraceEvent(kind="start", message=f"Long-context analysis of “{regulation.title}” with {model.name} "
                    f"({budget.window:,}-token window, {len(index.clauses)} clauses)"))
    findings: list[Finding] = []
    stats = VerificationStats()
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "turns": 0}
    whole = fell_back = 0
    for req in regulation.requirements:
        plan = build_context(variant, regulation, req, index, counter, budget)
        if not plan.fits:
            emit(TraceEvent(kind="error", message=f"{req.id}: even the smallest context ({len(plan.clauses)} clauses, "
                            f"{plan.prompt_tokens:,} tokens) exceeds the {budget.prompt_limit:,}-token budget; skipped"))
            continue
        if plan.fallback:
            fell_back += 1
        elif variant in ("auto", "full"):
            whole += 1
        emit(TraceEvent(kind="read", message=f"{req.id}: {len(plan.clauses)} clauses in context, "
                        f"{plan.prompt_tokens:,} prompt tokens" + (f", fell back to {plan.fallback}" if plan.fallback else ""),
                        data={"clauses": len(plan.clauses), "prompt_tokens": plan.prompt_tokens,
                              "fallback": plan.fallback}))
        r = analyze_requirement(regulation, req, plan, model, index, budget, counter)
        findings += r.findings
        stats.submissions += r.calls
        stats.first_pass_citations += r.first_total
        stats.first_pass_valid += r.first_valid
        stats.final_citations += r.final_total
        stats.final_valid += r.final_valid
        usage["input_tokens"] += r.prompt_tokens
        usage["output_tokens"] += r.completion_tokens
        usage["cached_tokens"] += r.cached_tokens
        usage["turns"] += r.calls
        emit(TraceEvent(kind="verify", message=f"{req.id}: {len(r.findings)} findings, {r.final_valid}/{r.final_total} "
                        f"citations verified ({r.stopped})"))

    summary = (f"Long-context analysis of {len(regulation.requirements)} requirements found {len(findings)} affected "
               f"clauses. {whole} requirements were read against the whole library"
               + (f", {fell_back} fell back to search because the library did not fit" if fell_back else "") + ".")
    report = Report(run_id=uuid.uuid4().hex[:10], regulation_id=regulation.id, regulation_title=regulation.title,
                    mode="long-context", model=model.name, summary=summary, findings=findings,
                    requirements=regulation.requirements, stats=stats, usage=usage)
    if save:
        save_report(report, settings)
    flagged = sum(1 for f in findings if not f.verified)
    emit(TraceEvent(kind="done", message=f"Done: {len(findings)} findings"
                    + (f", {flagged} flagged for manual check" if flagged else ", all citations verified")))
    return report
