"""Saved analyses that an MCP client can build up over many calls and resume later.

Plain English: a client (say, Claude Desktop) starts an analysis of one regulation, then searches,
reads and submits findings one tool call at a time. Every submitted finding is fact-checked by the
server and stored here, so the work survives a closed chat or a restarted server: the next session
calls `get_analysis` and carries on from what is already verified and what is still missing.

Storage is one SQLite file (standard library, no server to run). MCP tools run in worker
threads, so the store keeps one connection and lets one thread use it at a time; several server
processes can share the file, since SQLite locks it.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .models import Finding, Regulation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id              TEXT PRIMARY KEY,
    regulation_json TEXT NOT NULL,
    source          TEXT NOT NULL,       -- "client" (built by MCP calls) or "server:<mode>"
    summary         TEXT NOT NULL DEFAULT '',
    run_id          TEXT,                -- data/runs/<run_id>.json when the server ran the agent
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS findings (
    analysis_id     TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    requirement_id  TEXT NOT NULL,
    clause_id       TEXT NOT NULL,
    finding_json    TEXT NOT NULL,
    verified        INTEGER NOT NULL,
    submissions     INTEGER NOT NULL,    -- how many times this pair was submitted (1 = first try)
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (analysis_id, requirement_id, clause_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StoredFinding:
    finding: Finding
    submissions: int
    updated_at: str


@dataclass
class Analysis:
    id: str
    regulation: Regulation
    source: str
    summary: str
    run_id: str | None
    created_at: str
    updated_at: str
    findings: list[StoredFinding]


class AnalysisStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self._lock = threading.RLock()
        with self._tx() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One transaction, holding the connection: commits on success, rolls back on error."""
        with self._lock, self._conn:
            yield self._conn

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ analyses
    def create(self, regulation: Regulation, source: str = "client") -> str:
        analysis_id = uuid.uuid4().hex[:12]
        now = _now()
        with self._tx() as conn:
            conn.execute("INSERT INTO analyses (id, regulation_json, source, created_at, updated_at) "
                         "VALUES (?, ?, ?, ?, ?)",
                         (analysis_id, regulation.model_dump_json(), source, now, now))
        return analysis_id

    def get(self, analysis_id: str) -> Analysis | None:
        with self._tx() as conn:
            row = conn.execute("SELECT id, regulation_json, source, summary, run_id, created_at, updated_at "
                               "FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
            if row is None:
                return None
            rows = conn.execute("SELECT finding_json, submissions, updated_at FROM findings "
                                "WHERE analysis_id = ? ORDER BY requirement_id, clause_id",
                                (analysis_id,)).fetchall()
        findings = [StoredFinding(Finding.model_validate_json(f), n, at) for f, n, at in rows]
        findings.sort(key=lambda s: (_req_order(s.finding.requirement_id), s.finding.clause_id))
        return Analysis(row[0], Regulation.model_validate_json(row[1]), *row[2:], findings=findings)

    def list(self, limit: int = 20) -> list[dict]:
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT a.id, a.regulation_json, a.source, a.updated_at, "
                "COUNT(f.clause_id), COALESCE(SUM(f.verified), 0) "
                "FROM analyses a LEFT JOIN findings f ON f.analysis_id = a.id "
                "GROUP BY a.id ORDER BY a.updated_at DESC, a.created_at DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for aid, reg_json, source, updated, n, verified in rows:
            reg = json.loads(reg_json)
            out.append({"analysis_id": aid, "regulation_title": reg["title"], "source": source,
                        "updated_at": updated, "findings": n, "verified_findings": verified})
        return out

    def set_summary(self, analysis_id: str, summary: str, run_id: str | None = None) -> None:
        with self._tx() as conn:
            conn.execute("UPDATE analyses SET summary = ?, run_id = COALESCE(?, run_id), updated_at = ? "
                         "WHERE id = ?", (summary, run_id, _now(), analysis_id))

    # ------------------------------------------------------------------ findings
    def finding_count(self, analysis_id: str) -> int:
        with self._tx() as conn:
            return conn.execute("SELECT COUNT(*) FROM findings WHERE analysis_id = ?",
                                (analysis_id,)).fetchone()[0]

    def has_finding(self, analysis_id: str, requirement_id: str, clause_id: str) -> bool:
        with self._tx() as conn:
            return conn.execute("SELECT 1 FROM findings WHERE analysis_id = ? AND requirement_id = ? "
                                "AND clause_id = ?", (analysis_id, requirement_id, clause_id)).fetchone() is not None

    def upsert_finding(self, analysis_id: str, finding: Finding) -> int:
        """Save a checked finding. A resubmission of the same (requirement, clause) pair replaces
        the earlier one, as the agent loop does. Returns how many times the pair has been submitted."""
        now = _now()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO findings (analysis_id, requirement_id, clause_id, finding_json, verified, "
                "submissions, updated_at) VALUES (?, ?, ?, ?, ?, 1, ?) "
                "ON CONFLICT (analysis_id, requirement_id, clause_id) DO UPDATE SET "
                "finding_json = excluded.finding_json, verified = excluded.verified, "
                "submissions = findings.submissions + 1, updated_at = excluded.updated_at",
                (analysis_id, finding.requirement_id, finding.clause_id, finding.model_dump_json(),
                 int(finding.verified), now))
            conn.execute("UPDATE analyses SET updated_at = ? WHERE id = ?", (now, analysis_id))
            return conn.execute("SELECT submissions FROM findings WHERE analysis_id = ? AND requirement_id = ? "
                                "AND clause_id = ?",
                                (analysis_id, finding.requirement_id, finding.clause_id)).fetchone()[0]

    def remove_finding(self, analysis_id: str, requirement_id: str, clause_id: str) -> bool:
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM findings WHERE analysis_id = ? AND requirement_id = ? AND clause_id = ?",
                               (analysis_id, requirement_id, clause_id))
            if cur.rowcount:
                conn.execute("UPDATE analyses SET updated_at = ? WHERE id = ?", (_now(), analysis_id))
            return bool(cur.rowcount)


def _req_order(req_id: str) -> tuple[int, str]:
    """R2 before R10."""
    digits = req_id[1:]
    return (int(digits), req_id) if digits.isdigit() else (10**6, req_id)
