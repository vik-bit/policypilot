"""Offline mode: a simple, non-AI analyser so anyone can run the demo without an API key.

Plain English: for each requirement it runs the same search the AI would use, takes the best
match, and picks the best-matching sentence on each side as the quotes. It can't really *reason*
(its "conflict / gap / aligned" label is a guess based on numbers), but it exercises the same
search, fact-checker, UI and evaluation, and it gives a baseline to compare the AI against.
"""

from __future__ import annotations

import re

from ..models import Clause, Finding, Regulation, Requirement, TraceEvent, VerificationStats
from ..retrieval import PolicyIndex
from ..textutil import sentences, tokenize
from .tools import Emit
from .verifier import citation_counts, verify


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*", text)}


def _best_sentence(source: str, against: str) -> str:
    target = set(tokenize(against))
    best = max(sentences(source), key=lambda s: len(target & set(tokenize(s))), default=source)
    return best.rstrip(".")


def _guess_impact(req_sentence: str, clause: Clause) -> str:
    req_nums, clause_nums = _numbers(req_sentence), _numbers(clause.text)
    if req_nums and req_nums <= clause_nums:
        return "aligned"
    if req_nums and clause_nums:
        return "conflict"
    return "gap"


def _finding(req: Requirement, clause: Clause, score: float) -> Finding:
    reg_quote = _best_sentence(req.text, clause.text)
    impact = _guess_impact(reg_quote, clause)
    pol_quote = _best_sentence(clause.text, req.text)
    return Finding(
        requirement_id=req.id,
        clause_id=clause.id,
        impact=impact,
        explanation=(f"This clause is the closest match in your policies for {req.id} ({req.title.lower()}). "
                     "In demo mode the status label is an automatic guess, so read both texts and confirm "
                     "whether a change is really needed."),
        regulation_quote=reg_quote,
        policy_quote=pol_quote,
        proposed_text="" if impact == "aligned" else
        f"[Draft for review] Amend clause {clause.number} ({clause.title}) to meet {req.id}: {reg_quote}.",
    )


def run_offline(regulation: Regulation, index: PolicyIndex, emit: Emit) -> tuple[str, list[Finding], VerificationStats]:
    findings: list[Finding] = []
    for req in regulation.requirements:
        hits = index.search(req.text, k=3)
        emit(TraceEvent(kind="search", message=f'{req.id}: searched "{req.title}" → '
                        + ", ".join(h.clause.id for h in hits), data={"results": [h.clause.id for h in hits]}))
        if not hits:
            continue
        chosen = [hits[0]]
        # Also keep the runner-up if *both* keyword and meaning search put it in their top 3.
        if len(hits) > 1 and hits[1].ranks.get("bm25", 99) <= 3 and hits[1].ranks.get("dense", 99) <= 3:
            chosen.append(hits[1])
        findings += [_finding(req, h.clause, h.score) for h in chosen]

    checked = verify(findings, regulation, index)
    total, valid = citation_counts(checked)
    stats = VerificationStats(submissions=1, first_pass_citations=total, first_pass_valid=valid,
                              final_citations=total, final_valid=valid)
    emit(TraceEvent(kind="verify", message=f"Fact-check: {valid}/{total} citations verified"))
    summary = (f"PolicyPilot checked {len(regulation.requirements)} requirements against your policies and "
               f"found {len(checked)} clauses that may be affected. Every quote below was confirmed in the "
               "original documents. Because this ran in demo mode, please confirm each status label yourself.")
    return summary, checked, stats
