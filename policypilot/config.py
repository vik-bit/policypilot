"""All settings in one place, read from environment variables (or a `.env` file).

Every setting has a sensible default, so the project runs with zero configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny `.env` reader (KEY=VALUE lines) so we don't need another dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


@dataclass
class Settings:
    # --- where things live -------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("PP_DATA_DIR", str(PROJECT_ROOT / "data"))))

    # --- the AI model ------------------------------------------------------
    # Which AI runs the analysis: "claude", "groq", "offline", or "auto" (Claude if its key is set,
    # else Groq if its key is set, else offline). PP_MODE is the older name for the same setting.
    mode: str = field(default_factory=lambda: _env("PP_PROVIDER", _env("PP_MODE", "auto")).lower())
    model: str = field(default_factory=lambda: _env("PP_MODEL", "claude-opus-5-5"))
    groq_model: str = field(default_factory=lambda: _env("PP_GROQ_MODEL", "openai/gpt-oss-120b"))
    effort: str = field(default_factory=lambda: _env("PP_EFFORT", "medium"))
    max_turns: int = field(default_factory=lambda: int(_env("PP_MAX_TURNS", "16")))
    max_verify_retries: int = field(default_factory=lambda: int(_env("PP_MAX_VERIFY_RETRIES", "2")))
    refusal_fallback: bool = field(default_factory=lambda: _env("PP_REFUSAL_FALLBACK", "on") == "on")

    # Public demo: turns off policy uploads so visitors can't change the shared library.
    read_only: bool = field(default_factory=lambda: _env("PP_READ_ONLY", "off") == "on")

    # --- retrieval ---------------------------------------------------------
    # embedder: "auto" (fastembed if installed, else hashing), "fastembed", or "hashing"
    embedder: str = field(default_factory=lambda: _env("PP_EMBEDDER", "auto"))
    embed_model: str = field(default_factory=lambda: _env("PP_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    # rerank: "auto" (cross-encoder if fastembed is installed), "on", or "off"
    rerank: str = field(default_factory=lambda: _env("PP_RERANK", "auto"))
    rerank_model: str = field(default_factory=lambda: _env("PP_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))

    @property
    def policies_dir(self) -> Path:
        return self.data_dir / "policies"

    @property
    def regulations_dir(self) -> Path:
        return self.data_dir / "regulations"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def labels_path(self) -> Path:
        return self.data_dir / "eval" / "labels.jsonl"

    @property
    def has_api_key(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    @property
    def has_groq_key(self) -> bool:
        return bool(os.environ.get("GROQ_API_KEY"))

    @property
    def resolved_mode(self) -> str:
        """Which analyser actually runs: "claude", "groq" or "offline"."""
        mode = {"anthropic": "claude"}.get(self.mode, self.mode)
        if mode in ("claude", "groq", "offline"):
            return mode
        if self.has_api_key:
            return "claude"
        return "groq" if self.has_groq_key else "offline"

    @property
    def active_model(self) -> str:
        return {"claude": self.model, "groq": self.groq_model}.get(self.resolved_mode, "offline-heuristic")


def get_settings() -> Settings:
    return Settings()
