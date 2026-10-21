"""Optional second-pass reranking with a cross-encoder.

Plain English: the first search is fast but shallow — it compares the question and each clause
separately. A cross-encoder reads the question and the clause *together*, like a person would, and
gives a sharper relevance score. It's slower, so we only use it on the top ~20 candidates.
"""

from __future__ import annotations

import logging


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # optional dependency

        self.name = f"cross-encoder:{model_name}"
        self.model = TextCrossEncoder(model_name=model_name)

    def scores(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, documents)]


def make_reranker(setting: str, model_name: str) -> CrossEncoderReranker | None:
    if setting == "off":
        return None
    try:
        return CrossEncoderReranker(model_name)
    except Exception as exc:
        if setting == "on":
            raise RuntimeError(f"reranker requested but unavailable: {exc}") from exc
        logging.getLogger(__name__).warning("Reranker unavailable (%s); skipping rerank.", exc)
        return None
