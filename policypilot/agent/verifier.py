"""Step 4 — The fact-checker.

Plain English: for every quote the AI used, open the original document and check the quote is
really there, word for word. If it isn't, say exactly what's wrong and show the closest real
sentence, so the AI can fix it instead of guessing again. A plain text search can't hallucinate,
which is why this check is code, not another AI.
"""

from __future__ import annotations

import difflib

from ..models import Finding, Regulation
from ..retrieval import PolicyIndex
from ..textutil import normalize, sentences

MIN_QUOTE_WORDS = 3


def _clean_quote(q: str) -> str:
    return q.strip().strip('"“”\'').strip()


def quote_in(quote: str, source: str) -> bool:
    """True if every '...'-separated fragment of `quote` appears in `source`, in order."""
    src = normalize(source)
    pos = 0
    for frag in normalize(_clean_quote(quote)).replace("…", "...").split("..."):
        frag = frag.strip(" .,;:")
        if not frag:
            continue
        found = src.find(frag, pos)
        if found < 0:
            return False
        pos = found + len(frag)
    return True


def closest_sentence(quote: str, source: str) -> str | None:
    candidates = sentences(source)
    match = difflib.get_close_matches(_clean_quote(quote), candidates, n=1, cutoff=0.3)
    return match[0] if match else None


def _check_quote(label: str, quote: str, source: str, where: str) -> list[str]:
    quote = _clean_quote(quote)
    if len(quote.split()) < MIN_QUOTE_WORDS:
        return [f"{label} is too short; quote at least {MIN_QUOTE_WORDS} consecutive words from {where}."]
    if quote_in(quote, source):
        return []
    hint = closest_sentence(quote, source)
    msg = f"{label} was not found word for word in {where}."
    if hint:
        msg += f' Closest actual sentence: "{hint}"'
    return [msg]


def verify_finding(f: Finding, regulation: Regulation, index: PolicyIndex) -> Finding:
    issues: list[str] = []
    req_ids = {r.id for r in regulation.requirements}
    if req_ids and f.requirement_id not in req_ids:
        issues.append(f"requirement_id '{f.requirement_id}' does not exist; valid ids: {sorted(req_ids)}.")

    clause = index.get(f.clause_id)
    if clause is None:
        issues.append(f"clause_id '{f.clause_id}' does not exist. Use an id returned by search_policies.")

    issues += _check_quote("regulation_quote", f.regulation_quote, regulation.text, "the regulation")

    if clause is not None:
        if f.policy_quote.strip():
            issues += _check_quote("policy_quote", f.policy_quote, clause.text, f"clause {f.clause_id}")
        elif f.impact != "gap":
            issues.append(f"policy_quote is empty; a {f.impact} finding must quote the clause.")

    if f.impact != "aligned" and not f.proposed_text.strip():
        issues.append(f"proposed_text is empty; a {f.impact} finding needs a drafted replacement clause.")

    return f.model_copy(update={"issues": issues, "verified": not issues})


def citation_counts(findings: list[Finding]) -> tuple[int, int]:
    """(citations, valid citations). A finding has 1-2 citations: regulation + policy quote."""
    total = valid = 0
    for f in findings:
        n = 2 if f.policy_quote.strip() else 1
        total += n
        if f.verified:
            valid += n
        else:  # count the quotes that individually passed
            bad = sum(1 for i in f.issues if i.startswith(("regulation_quote", "policy_quote")))
            valid += max(n - bad, 0)
    return total, valid


def verify(findings: list[Finding], regulation: Regulation, index: PolicyIndex) -> list[Finding]:
    seen: set[tuple[str, str]] = set()
    out = []
    for f in findings:
        key = (f.requirement_id, f.clause_id)
        if key in seen:  # drop exact duplicates the model sometimes emits
            continue
        seen.add(key)
        out.append(verify_finding(f, regulation, index))
    return out
