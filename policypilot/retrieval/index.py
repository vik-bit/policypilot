"""Step 2 — Retrieval: find the clauses most relevant to a question.

Plain English: we search two ways and merge the results.
- Keyword search (BM25) is great at exact terms: "V-CIP", "T+1", "AES-256".
- Meaning search (embeddings) is great at paraphrases: "storage limitation" ≈ "data retention".
- Reciprocal Rank Fusion merges the two ranked lists: a clause near the top of *either* list
  rises to the top of the merged list.
- An optional cross-encoder then re-reads the top candidates for a final, sharper ordering.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import Settings
from ..ingest import load_policies
from ..models import Clause, SearchHit
from ..textutil import tokenize
from .bm25 import BM25
from .embedders import Embedder, make_embedder
from .rerank import CrossEncoderReranker, make_reranker

RRF_K = 60  # standard constant from the original RRF paper; dampens the weight of top ranks
CANDIDATES = 30  # how deep into each list we look before fusing
RERANK_DEPTH = 20  # how many fused candidates the cross-encoder re-reads

METHODS = ("bm25", "dense", "hybrid", "hybrid+rerank")


def _search_text(c: Clause) -> str:
    """What we index for each clause: its policy, heading and body together."""
    return f"{c.policy_title} — {c.number} {c.title}. {c.text}"


def _fingerprint(policies_dir: Path, embedder_name: str) -> str:
    h = hashlib.sha256(embedder_name.encode())
    for p in sorted(policies_dir.iterdir()):
        if p.is_file():
            h.update(p.name.encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


class PolicyIndex:
    def __init__(self, clauses: list[Clause], embedder: Embedder, vectors: np.ndarray,
                 reranker: CrossEncoderReranker | None = None):
        self.clauses = clauses
        self.by_id = {c.id: c for c in clauses}
        self.embedder = embedder
        self.vectors = vectors
        self.reranker = reranker
        self.bm25 = BM25().fit([tokenize(_search_text(c)) for c in clauses])

    # ------------------------------------------------------------------ build / load
    @classmethod
    def build(cls, settings: Settings, force: bool = False) -> "PolicyIndex":
        """Load the cached index if the policy files haven't changed, otherwise rebuild it."""
        embedder = make_embedder(settings.embedder, settings.embed_model)
        reranker = make_reranker(settings.rerank, settings.rerank_model)
        clauses = load_policies(settings.policies_dir)
        texts = [_search_text(c) for c in clauses]
        embedder.fit(texts)

        idx_dir = settings.index_dir
        fp = _fingerprint(settings.policies_dir, embedder.name)
        manifest = idx_dir / "manifest.json"
        vec_path = idx_dir / "vectors.npy"
        if not force and manifest.exists() and vec_path.exists():
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fp and meta.get("count") == len(clauses):
                return cls(clauses, embedder, np.load(vec_path), reranker)

        vectors = embedder.embed_documents(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        idx_dir.mkdir(parents=True, exist_ok=True)
        np.save(vec_path, vectors)
        (idx_dir / "clauses.json").write_text(
            json.dumps([c.model_dump() for c in clauses], indent=1, ensure_ascii=False), encoding="utf-8")
        manifest.write_text(json.dumps({
            "fingerprint": fp, "count": len(clauses), "embedder": embedder.name,
            "reranker": reranker.name if reranker else None,
        }, indent=2), encoding="utf-8")
        return cls(clauses, embedder, vectors, reranker)

    # ------------------------------------------------------------------ queries
    def get(self, clause_id: str) -> Clause | None:
        return self.by_id.get(clause_id)

    def _bm25_ranking(self, query: str) -> list[tuple[int, float]]:
        scores = self.bm25.scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [(i, scores[i]) for i in order[:CANDIDATES] if scores[i] > 0]

    def _dense_ranking(self, query: str) -> list[tuple[int, float]]:
        if not len(self.clauses):
            return []
        sims = self.vectors @ self.embedder.embed_query(query)
        order = np.argsort(-sims)[:CANDIDATES]
        return [(int(i), float(sims[i])) for i in order]

    def search(self, query: str, k: int = 5, method: str = "best") -> list[SearchHit]:
        """Return the top-k clauses. method: bm25 | dense | hybrid | hybrid+rerank | best."""
        if method == "best":
            method = "hybrid+rerank" if self.reranker else "hybrid"
        bm25 = self._bm25_ranking(query)
        dense = self._dense_ranking(query)
        ranks: dict[int, dict[str, int]] = {}
        for name, ranking in (("bm25", bm25), ("dense", dense)):
            for r, (i, _) in enumerate(ranking, 1):
                ranks.setdefault(i, {})[name] = r

        if method == "bm25":
            scored = bm25
        elif method == "dense":
            scored = dense
        else:
            # Reciprocal Rank Fusion: each list contributes 1/(60 + rank).
            fused = {i: sum(1 / (RRF_K + r) for r in rr.values()) for i, rr in ranks.items()}
            scored = sorted(fused.items(), key=lambda kv: -kv[1])
            if method == "hybrid+rerank" and self.reranker:
                top = scored[:RERANK_DEPTH]
                ce = self.reranker.scores(query, [_search_text(self.clauses[i]) for i, _ in top])
                scored = sorted(zip([i for i, _ in top], ce), key=lambda kv: -kv[1])
            elif method == "hybrid+rerank":
                method = "hybrid"

        hits = []
        for r, (i, score) in enumerate(scored[:k], 1):
            rr = dict(ranks.get(i, {}))
            rr[method] = r
            hits.append(SearchHit(clause=self.clauses[i], score=round(float(score), 4), ranks=rr))
        return hits

    def stats(self) -> dict:
        policies: dict[str, dict] = {}
        for c in self.clauses:
            p = policies.setdefault(c.policy, {"policy": c.policy, "title": c.policy_title, "clauses": 0})
            p["clauses"] += 1
        return {
            "clauses": len(self.clauses),
            "policies": list(policies.values()),
            "embedder": self.embedder.name,
            "reranker": self.reranker.name if self.reranker else None,
        }
