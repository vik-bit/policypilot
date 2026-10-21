"""Step 1 — Ingestion: turn documents into numbered, addressable pieces.

Plain English: an AI can't cite "somewhere in the privacy policy". It needs to say "clause 3.2".
So we cut each policy along its own numbering, and each regulation into its numbered requirements.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Clause, Regulation, Requirement
from ..textutil import slugify

SUPPORTED = {".md", ".txt", ".pdf"}

# A clause heading looks like "## 3.2 Data Retention", "3.2 Data Retention" or "Section 3.2 - Data Retention".
_CLAUSE_HEADING = re.compile(
    r"^(?:#{1,6}\s*)?(?i:section\s+|clause\s+)?(\d+(?:\.\d+){0,3})[.)]?\s*[-:–]?\s+([A-Z][^\n]{1,90})$",
    re.MULTILINE,
)
# A requirement heading looks like "## R2. Storage limitation" or "R2: Storage limitation".
_REQ_HEADING = re.compile(r"^(?:#{1,6}\s*)?(R\d+)[.):]?\s+(.*)$", re.MULTILINE)
_TITLE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def read_text(path: Path) -> str:
    """Read .md / .txt directly; extract text from .pdf with pypdf."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {path.name} (supported: {', '.join(sorted(SUPPORTED))})")


def _doc_title(text: str, fallback: str) -> str:
    m = _TITLE.search(text)
    return m.group(1).strip() if m else fallback


def split_clauses(text: str, policy: str, policy_title: str) -> list[Clause]:
    """Split a policy into clauses at its numbered headings.

    If the document has no numbered headings (e.g. a messy PDF), fall back to ~120-word chunks
    numbered P1, P2, ... so the rest of the system still works.
    """
    matches = list(_CLAUSE_HEADING.finditer(text))
    clauses: list[Clause] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = re.sub(r"\s+", " ", text[m.end():end]).strip()
        if not body:
            continue
        number = m.group(1)
        clauses.append(Clause(
            id=f"{policy}§{number}", policy=policy, policy_title=policy_title,
            number=number, title=m.group(2).strip().rstrip("."), text=body,
        ))
    if clauses:
        return _dedupe_ids(clauses)
    return _paragraph_chunks(text, policy, policy_title)


def _paragraph_chunks(text: str, policy: str, policy_title: str, target_words: int = 120) -> list[Clause]:
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for p in paragraphs:
        current.append(p)
        if sum(len(x.split()) for x in current) >= target_words:
            chunks.append(" ".join(current))
            current = [p]  # one paragraph of overlap keeps context across the cut
    if current and (not chunks or current != [chunks[-1]]):
        chunks.append(" ".join(current))
    return [
        Clause(id=f"{policy}§P{i}", policy=policy, policy_title=policy_title,
               number=f"P{i}", title=c[:60].rsplit(" ", 1)[0] + "…", text=c)
        for i, c in enumerate(chunks, 1)
    ]


def _dedupe_ids(clauses: list[Clause]) -> list[Clause]:
    """Two headings with the same number would collide; suffix the later ones."""
    seen: dict[str, int] = {}
    for c in clauses:
        if c.id in seen:
            seen[c.id] += 1
            c.id = f"{c.id}-{seen[c.id]}"
        else:
            seen[c.id] = 1
    return clauses


def load_policy_file(path: Path) -> list[Clause]:
    text = read_text(path)
    policy = slugify(path.stem)
    return split_clauses(text, policy, _doc_title(text, path.stem.replace("-", " ").title()))


def load_policies(directory: Path) -> list[Clause]:
    clauses: list[Clause] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in SUPPORTED:
            clauses.extend(load_policy_file(path))
    return clauses


def parse_regulation(text: str, reg_id: str = "pasted") -> Regulation:
    """Split a regulation into requirements at "R1.", "R2." headings, else at paragraphs."""
    title = _doc_title(text, reg_id)
    matches = list(_REQ_HEADING.finditer(text))
    reqs: list[Requirement] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = re.sub(r"\s+", " ", text[m.end():end]).strip()
        if body:
            reqs.append(Requirement(id=m.group(1).upper(), title=m.group(2).strip().rstrip("."), text=body))
    if not reqs:
        paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
        paragraphs = [p for p in paragraphs if len(p.split()) >= 8 and not p.startswith("#")]
        reqs = [Requirement(id=f"R{i}", title=p[:50].rsplit(" ", 1)[0] + "…", text=p)
                for i, p in enumerate(paragraphs, 1)]
    return Regulation(id=reg_id, title=title, text=text, requirements=reqs)


def load_regulation(path: Path) -> Regulation:
    return parse_regulation(read_text(path), slugify(path.stem))


def load_regulations(directory: Path) -> list[Regulation]:
    return [load_regulation(p) for p in sorted(directory.iterdir()) if p.suffix.lower() in SUPPORTED]
