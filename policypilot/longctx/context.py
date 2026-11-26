"""Context engineering: which clauses go into the prompt, in what order, and whether they fit.

Variants (one requirement at a time):

    rag5        the 5 best clauses from hybrid search (BM25 + embeddings), best first. The RAG
                comparator: same reader and prompt, only the context differs.
    hybrid50    the 50 best clauses from hybrid search, best first ("retrieve wide, read long").
    full        the whole library in one fixed order, the same for every requirement, so a server
                with prefix caching reads it once per run.
    random      the whole library in a random order drawn per requirement.
    bm25        the whole library sorted by BM25 score for the requirement, most relevant first.
    pos-start   the whole library in random order, with the labelled relevant clauses moved to the
    pos-middle  start, the middle or the end ("lost in the middle" test). Needs the labels, so it
    pos-end     exists only for evaluation.
    auto        what the app uses: `full` when it fits the token budget, else hybrid search with as
                many clauses as fit (50, 40, ... 5). This is the fallback to RAG.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from ..models import Clause, Regulation, Requirement
from ..retrieval import PolicyIndex
from ..textutil import tokenize
from .prompts import messages_for, render_library
from .tokens import TokenCounter, count_messages

VARIANTS = ("rag5", "hybrid50", "full", "random", "bm25", "pos-start", "pos-middle", "pos-end", "auto")
POSITION_VARIANTS = ("pos-start", "pos-middle", "pos-end")
FALLBACK_KS = (50, 40, 30, 20, 10, 5)


@dataclass(frozen=True)
class Budget:
    """How much of the context window the first prompt may use.

    Each fact-check retry appends the model's previous reply and the list of errors, so room is kept
    for (retries + 1) full replies and `retries` error messages."""

    window: int = 32_768
    max_output: int = 1_536
    retries: int = 2
    error_allowance: int = 512

    @property
    def prompt_limit(self) -> int:
        return self.window - (self.retries + 1) * self.max_output - self.retries * self.error_allowance


@dataclass
class ContextPlan:
    variant: str
    clauses: list[Clause]
    messages: list[dict]
    prompt_tokens: int
    fits: bool
    # set when `auto` could not fit the whole library and used search instead, e.g. "hybrid-30"
    fallback: str | None = None
    # same value for two requirements => their prompts share the library prefix (cacheable)
    prefix_key: str = ""
    extra: dict = field(default_factory=dict)


def query_for(req: Requirement) -> str:
    """The search query for a requirement; the same one the retrieval evaluation uses."""
    return f"{req.title}. {req.text}"


def _rng(seed: int, regulation: Regulation, req: Requirement) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{regulation.id}:{req.id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def interleave(policy_clauses: list[Clause], distractors: list[Clause], seed: int = 0) -> list[Clause]:
    """One fixed library order with distractors spread through it: policy clauses keep their file
    order, at random positions (seeded), so they are not all at the start of the prompt."""
    n = len(policy_clauses) + len(distractors)
    slots = set(random.Random(seed).sample(range(n), len(policy_clauses)))
    pol, dis = iter(policy_clauses), iter(distractors)
    return [next(pol) if i in slots else next(dis) for i in range(n)]


def order_clauses(variant: str, regulation: Regulation, req: Requirement, index: PolicyIndex,
                  library_order: list[Clause], relevant: list[str] | None = None, seed: int = 0) -> list[Clause]:
    if variant == "rag5":
        return [h.clause for h in index.search(query_for(req), k=5, method="hybrid")]
    if variant == "hybrid50":
        return [h.clause for h in index.search(query_for(req), k=50, method="hybrid")]
    if variant == "full":
        return list(library_order)
    if variant == "bm25":
        scores = index.bm25.scores(tokenize(query_for(req)))
        position = {c.id: i for i, c in enumerate(index.clauses)}
        return sorted(library_order, key=lambda c: -scores[position[c.id]])  # stable: ties keep library order
    if variant == "random" or variant in POSITION_VARIANTS:
        shuffled = list(library_order)
        _rng(seed, regulation, req).shuffle(shuffled)
        if variant == "random":
            return shuffled
        if not relevant:
            raise ValueError(f"{variant} needs the labelled relevant clauses for {regulation.id} {req.id}")
        wanted = set(relevant)
        block = [c for c in library_order if c.id in wanted]
        block.sort(key=lambda c: relevant.index(c.id))
        others = [c for c in shuffled if c.id not in wanted]
        at = {"pos-start": 0, "pos-middle": len(others) // 2, "pos-end": len(others)}[variant]
        return others[:at] + block + others[at:]
    raise ValueError(f"Unknown variant {variant!r}; choose from {', '.join(VARIANTS)}")


def _plan(variant: str, clauses: list[Clause], regulation: Regulation, req: Requirement,
          counter: TokenCounter, budget: Budget, fallback: str | None = None) -> ContextPlan:
    library = render_library(clauses)
    messages = messages_for(library, regulation, req)
    tokens = count_messages(counter, messages)
    return ContextPlan(variant=variant, clauses=clauses, messages=messages, prompt_tokens=tokens,
                       fits=tokens <= budget.prompt_limit, fallback=fallback,
                       prefix_key=hashlib.sha256(library.encode()).hexdigest()[:12])


def build_context(variant: str, regulation: Regulation, req: Requirement, index: PolicyIndex,
                  counter: TokenCounter, budget: Budget, library_order: list[Clause] | None = None,
                  relevant: list[str] | None = None, seed: int = 0) -> ContextPlan:
    """The prompt for one requirement under one variant. `fits` says whether it is within budget;
    only `auto` changes what it sends when it is not."""
    library_order = library_order if library_order is not None else index.clauses
    if variant != "auto":
        clauses = order_clauses(variant, regulation, req, index, library_order, relevant, seed)
        return _plan(variant, clauses, regulation, req, counter, budget)

    plan = _plan("auto", list(library_order), regulation, req, counter, budget)
    if plan.fits:
        return plan
    for k in FALLBACK_KS:
        clauses = [h.clause for h in index.search(query_for(req), k=k, method="hybrid")]
        plan = _plan("auto", clauses, regulation, req, counter, budget, fallback=f"hybrid-{k}")
        if plan.fits:
            return plan
    return plan  # even 5 clauses do not fit: the caller reports it instead of sending it
