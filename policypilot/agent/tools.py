"""The read-only tools the agent may call: search and read."""

from __future__ import annotations

import json
from typing import Callable

from ..models import TraceEvent
from ..retrieval import PolicyIndex

Emit = Callable[[TraceEvent], None]
SNIPPET_CHARS = 300


class PolicyTools:
    def __init__(self, index: PolicyIndex, emit: Emit):
        self.index = index
        self.emit = emit

    def search_policies(self, query: str, k: int = 5) -> str:
        k = max(1, min(int(k or 5), 10))
        hits = self.index.search(query, k=k)
        self.emit(TraceEvent(
            kind="search", message=f'Searched "{query}" → {", ".join(h.clause.id for h in hits) or "nothing"}',
            data={"query": query, "results": [h.clause.id for h in hits]},
        ))
        return json.dumps([
            {
                "clause_id": h.clause.id,
                "policy": h.clause.policy_title,
                "heading": h.clause.heading,
                "snippet": h.clause.text[:SNIPPET_CHARS] + ("…" if len(h.clause.text) > SNIPPET_CHARS else ""),
                "score": h.score,
            }
            for h in hits
        ], ensure_ascii=False)

    def read_clause(self, clause_id: str) -> tuple[str, bool]:
        """Returns (result, is_error)."""
        c = self.index.get(clause_id)
        if c is None:
            self.emit(TraceEvent(kind="read", message=f"Tried to read unknown clause {clause_id}"))
            return f"No clause with id '{clause_id}'. Use an id returned by search_policies.", True
        self.emit(TraceEvent(kind="read", message=f"Read {c.id} ({c.title})", data={"clause_id": c.id}))
        return json.dumps({"clause_id": c.id, "policy": c.policy_title, "heading": c.heading, "text": c.text},
                          ensure_ascii=False), False
