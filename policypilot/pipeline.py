"""The whole flow in one function. Start reading the code here.

    regulation ──► split into requirements ──► agent (search · read · draft) ──► fact-check ──► report
                                                   ▲                                 │
                                                   └──────── fix these quotes ◄──────┘
"""

from __future__ import annotations

import json
import uuid
from typing import Callable

from .config import Settings
from .models import Regulation, Report, TraceEvent
from .retrieval import PolicyIndex

Emit = Callable[[TraceEvent], None]


def analyze(regulation: Regulation, index: PolicyIndex, settings: Settings,
            emit: Emit | None = None, client=None, run_id: str | None = None) -> Report:
    emit = emit or (lambda e: None)
    mode = _client_mode(client) if client is not None else settings.resolved_mode
    model = {"claude": settings.model, "groq": settings.groq_model}.get(mode, "offline-heuristic")
    run_id = run_id or uuid.uuid4().hex[:10]

    emit(TraceEvent(kind="start", message=f"Analysing “{regulation.title}” in {mode} mode"
                    + (f" with {model}" if mode != "offline" else "")))
    emit(TraceEvent(kind="plan", message=f"Split the regulation into {len(regulation.requirements)} requirements: "
                    + ", ".join(r.id for r in regulation.requirements)))

    usage: dict[str, int] = {}
    if mode in ("claude", "groq"):
        # Imported lazily so each mode only needs its own SDK installed.
        if mode == "claude":
            from .agent.claude_agent import ClaudeAgent as Agent
        else:
            from .agent.groq_agent import GroqAgent as Agent

        agent = Agent(settings, index, emit, client=client)
        summary, findings, stats = agent.run(regulation)
        usage = agent.usage
    else:
        from .agent.offline import run_offline

        summary, findings, stats = run_offline(regulation, index, emit)

    report = Report(
        run_id=run_id, regulation_id=regulation.id, regulation_title=regulation.title,
        mode=mode, model=model, summary=summary, findings=findings,
        requirements=regulation.requirements, stats=stats, usage=usage,
    )
    save_report(report, settings)
    flagged = sum(1 for f in findings if not f.verified)
    emit(TraceEvent(kind="done", message=f"Done: {len(findings)} findings"
                    + (f", {flagged} flagged for manual check" if flagged else ", all citations verified")))
    return report


def _client_mode(client) -> str:
    """Tests pass a ready-made SDK client; tell which provider it belongs to."""
    return "groq" if type(client).__module__.startswith("groq") else "claude"


def save_report(report: Report, settings: Settings) -> None:
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    (settings.runs_dir / f"{report.run_id}.json").write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_report(run_id: str, settings: Settings) -> Report | None:
    path = settings.runs_dir / f"{run_id}.json"
    if not path.exists():
        return None
    return Report.model_validate_json(path.read_text(encoding="utf-8"))
