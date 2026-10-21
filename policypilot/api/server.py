"""Step 5 — The web server: a small JSON API plus the review UI.

Analyses run in a background thread. The browser polls /api/jobs/{id} once a second to show the
agent's steps live, then renders the report for a human to approve or reject each finding.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import get_settings
from ..ingest import load_regulation, load_regulations, parse_regulation
from ..ingest.loader import SUPPORTED
from ..models import TraceEvent
from ..pipeline import analyze, load_report, save_report
from ..retrieval import METHODS, PolicyIndex
from ..textutil import slugify

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

settings = get_settings()
app = FastAPI(title="PolicyPilot", version="0.1.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

_index: PolicyIndex | None = None
_index_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def get_index(rebuild: bool = False) -> PolicyIndex:
    global _index
    with _index_lock:
        if _index is None or rebuild:
            _index = PolicyIndex.build(settings, force=rebuild)
        return _index


# ---------------------------------------------------------------------------- pages
@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


# ---------------------------------------------------------------------------- info
@app.get("/api/health")
def health() -> dict:
    idx = get_index()
    return {
        "status": "ok",
        "mode": settings.resolved_mode,
        "model": settings.active_model,
        # True when a provider is selected but its key is missing, so the UI can say so up front.
        "key_missing": (settings.resolved_mode == "groq" and not settings.has_groq_key)
                       or (settings.resolved_mode == "claude" and not settings.has_api_key),
        **idx.stats(),
    }


def _meta(text: str) -> dict[str, str]:
    """Pull 'Issued by: … · Effective: …' and 'Applies to: …' lines out of a regulation header."""
    meta: dict[str, str] = {}
    for line in text.splitlines()[:8]:
        for part in line.split("·"):
            key, sep, value = part.partition(":")
            if sep and key.strip().lower() in ("issued by", "effective", "applies to"):
                meta[key.strip().lower().replace(" ", "_")] = value.strip()
    return meta


@app.get("/api/regulations")
def regulations() -> list[dict]:
    return [{
        "id": r.id, "title": r.title, "text": r.text,
        "requirements": len(r.requirements),
        "requirement_list": [{"id": q.id, "title": q.title} for q in r.requirements],
        "meta": _meta(r.text),
    } for r in load_regulations(settings.regulations_dir)]


@app.get("/api/policies")
def policies() -> dict:
    idx = get_index()
    grouped: dict[str, list[dict]] = {}
    for c in idx.clauses:
        grouped.setdefault(c.policy, []).append(c.model_dump())
    return {"stats": idx.stats(), "clauses": grouped}


@app.post("/api/policies")
async def upload_policy(file: UploadFile = File(...)) -> dict:
    name = Path(file.filename or "upload.txt")
    if name.suffix.lower() not in SUPPORTED:
        raise HTTPException(400, f"Unsupported file type. Use one of: {', '.join(sorted(SUPPORTED))}")
    dest = settings.policies_dir / f"{slugify(name.stem)}{name.suffix.lower()}"
    dest.write_bytes(await file.read())
    idx = get_index(rebuild=True)
    added = [c.id for c in idx.clauses if c.policy == slugify(name.stem)]
    return {"saved": dest.name, "clauses_added": len(added), "stats": idx.stats()}


@app.get("/api/search")
def search(q: str, k: int = 5, method: str = "best") -> list[dict]:
    if method not in (*METHODS, "best"):
        raise HTTPException(400, f"method must be one of {METHODS + ('best',)}")
    return [h.model_dump() for h in get_index().search(q, k=k, method=method)]


@app.get("/api/clauses/{clause_id}")
def clause(clause_id: str) -> dict:
    c = get_index().get(clause_id)
    if c is None:
        raise HTTPException(404, "No such clause")
    return c.model_dump()


# ---------------------------------------------------------------------------- analysis jobs
class AnalyzeRequest(BaseModel):
    regulation_id: str | None = None
    text: str | None = None


def _run_job(job_id: str, req: AnalyzeRequest) -> None:
    job = _jobs[job_id]

    def emit(event: TraceEvent) -> None:
        job["events"].append(event.model_dump())

    try:
        if req.regulation_id:
            path = next((p for p in settings.regulations_dir.iterdir() if slugify(p.stem) == req.regulation_id), None)
            if path is None:
                raise ValueError(f"Unknown regulation '{req.regulation_id}'")
            regulation = load_regulation(path)
        else:
            regulation = parse_regulation(req.text or "", reg_id="pasted")
            if not regulation.requirements:
                raise ValueError("Couldn't find any requirements in the pasted text.")
        report = analyze(regulation, get_index(), settings, emit=emit, run_id=job_id)
        job["report"] = report.model_dump()
        job["status"] = "done"
    except Exception as exc:  # surface any failure to the UI instead of a silent hang
        emit(TraceEvent(kind="error", message=f"Failed: {exc}"))
        job["status"] = "error"
        job["error"] = str(exc)


@app.post("/api/analyze")
def start_analysis(req: AnalyzeRequest) -> dict:
    if not req.regulation_id and not (req.text and req.text.strip()):
        raise HTTPException(400, "Provide regulation_id or text")
    job_id = uuid.uuid4().hex[:10]
    _jobs[job_id] = {"id": job_id, "status": "running", "events": [], "report": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str, since: int = 0) -> dict:
    j = _jobs.get(job_id)
    if j is None:
        report = load_report(job_id, settings)
        if report is None:
            raise HTTPException(404, "No such job")
        return {"id": job_id, "status": "done", "events": [], "report": report.model_dump(), "error": None}
    return {**j, "events": j["events"][since:]}


class Decision(BaseModel):
    run_id: str
    key: str  # "R2|data-privacy§3.2"
    verdict: Literal["approved", "rejected", "pending"]


@app.post("/api/decisions")
def decide(d: Decision) -> dict:
    report = load_report(d.run_id, settings)
    if report is None:
        raise HTTPException(404, "No such run")
    if d.verdict == "pending":
        report.decisions.pop(d.key, None)
    else:
        report.decisions[d.key] = d.verdict
    save_report(report, settings)
    if d.run_id in _jobs and _jobs[d.run_id]["report"]:
        _jobs[d.run_id]["report"]["decisions"] = report.decisions
    return {"decisions": report.decisions}
