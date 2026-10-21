"""Turning text into vectors ("embeddings") so we can search by meaning.

Plain English: an embedding model turns a sentence into a list of numbers such that sentences with
similar *meaning* end up with similar numbers. "Keep records for five years" and "data retention
period" land close together even though they share no words.

Two implementations:
- FastEmbedder: a real neural embedding model (bge-small, runs on CPU via ONNX). Best quality.
- HashingEmbedder: a no-download fallback (TF-IDF over hashed word and character n-grams). It
  understands spelling overlap, not meaning — good enough to keep everything runnable offline.
"""

from __future__ import annotations

import logging
import math
import os
import re
import zlib
from collections import Counter
from typing import Protocol

import numpy as np

from ..textutil import normalize, tokenize

log = logging.getLogger(__name__)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless Windows warning


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class HashingEmbedder:
    dims = 2**13

    def __init__(self) -> None:
        self.name = "hashing-tfidf"
        self.idf = np.ones(self.dims, dtype=np.float32)

    def _features(self, text: str) -> Counter[int]:
        words = tokenize(text)
        feats: list[str] = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        for w in re.findall(r"[a-z]{5,}", normalize(text)):
            feats += [f"c:{w[i:i + 4]}" for i in range(len(w) - 3)]
        return Counter(zlib.crc32(f.encode()) % self.dims for f in feats)

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dims, dtype=np.float32)
        for idx, tf in self._features(text).items():
            v[idx] = 1 + math.log(tf)
        return v * self.idf

    def fit(self, texts: list[str]) -> None:
        df = np.zeros(self.dims, dtype=np.float32)
        for t in texts:
            df[list(self._features(t).keys())] += 1
        self.idf = (np.log((len(texts) + 1) / (df + 1)) + 1).astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._vector(text)[None, :])[0]


class FastEmbedder:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # optional dependency

        self.name = f"fastembed:{model_name}"
        self.model = TextEmbedding(model_name=model_name)

    def fit(self, texts: list[str]) -> None:  # neural models are pre-trained; nothing to fit
        return None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        fn = getattr(self.model, "passage_embed", self.model.embed)
        return _l2(np.array(list(fn(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        fn = getattr(self.model, "query_embed", self.model.embed)
        return _l2(np.array(list(fn([text])), dtype=np.float32))[0]


def make_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    try:
        return FastEmbedder(model_name)
    except Exception as exc:  # not installed, or model download failed
        if kind == "fastembed":
            raise RuntimeError(f"fastembed embedder requested but unavailable: {exc}") from exc
        log.warning("Neural embedder unavailable (%s); using the hashing fallback.", exc)
        return HashingEmbedder()
