"""Long-context mode: put the policy library in the prompt instead of retrieving a few clauses.

Plain English: the normal agent searches the library and reads a handful of clauses. Here the
model reads every clause at once (or as many as fit in its context window), each one tagged with
the same stable id the search uses, so the fact-checker can still check every quote word for word.
When the library is too big for the window, the mode falls back to retrieval.

Start reading at reader.py. Run with `python -m policypilot.longctx --help`.
"""

from .context import VARIANTS, ContextPlan, build_context
from .reader import analyze_long, analyze_requirement

__all__ = ["VARIANTS", "ContextPlan", "build_context", "analyze_long", "analyze_requirement"]
