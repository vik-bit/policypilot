"""The data shapes that flow through the system.

Plain English: a *Clause* is one numbered rule in a company policy. A *Requirement* is one numbered
rule in a new regulation. A *Finding* says "this requirement affects that clause, here's why, here's
the fix, and here are the quotes that prove it".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

Impact = Literal["conflict", "gap", "aligned"]


class Clause(BaseModel):
    id: str  # e.g. "data-privacy§3.2"
    policy: str  # e.g. "data-privacy"
    policy_title: str
    number: str  # e.g. "3.2"
    title: str  # e.g. "Data Retention"
    text: str

    @property
    def heading(self) -> str:
        return f"{self.number} {self.title}"


class Requirement(BaseModel):
    id: str  # e.g. "R2"
    title: str
    text: str


class Regulation(BaseModel):
    id: str
    title: str
    text: str
    requirements: list[Requirement]


class SearchHit(BaseModel):
    clause: Clause
    score: float
    # rank of this clause in each retriever's list (1 = best); missing means "not in that list"
    ranks: dict[str, int] = Field(default_factory=dict)


class Finding(BaseModel):
    clause_id: str
    requirement_id: str
    impact: Impact
    explanation: str
    regulation_quote: str
    policy_quote: str = ""
    proposed_text: str = ""
    # filled in by the verifier
    verified: bool = False
    issues: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    """One visible step of a run, shown live in the UI."""

    kind: str  # "start" | "plan" | "search" | "read" | "llm" | "submit" | "verify" | "done" | "error"
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class VerificationStats(BaseModel):
    submissions: int = 0
    first_pass_citations: int = 0
    first_pass_valid: int = 0
    final_citations: int = 0
    final_valid: int = 0


class Report(BaseModel):
    run_id: str
    regulation_id: str
    regulation_title: str
    mode: str  # "claude" or "offline"
    model: str
    summary: str = ""
    findings: list[Finding] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    stats: VerificationStats = Field(default_factory=VerificationStats)
    usage: dict[str, int] = Field(default_factory=dict)
    decisions: dict[str, str] = Field(default_factory=dict)  # finding key -> "approved" | "rejected"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
