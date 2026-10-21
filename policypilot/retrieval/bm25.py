"""BM25 keyword search, written from scratch so you can see there's no magic.

Plain English: a clause scores higher when it contains the query's words, especially *rare* words
(a rare word like "tokenisation" says more than a common one like "customer"), and short clauses
aren't drowned out by long ones.

Score(clause, query) = Σ over query words w of
    IDF(w) · tf(w) · (k1 + 1) / (tf(w) + k1 · (1 − b + b · len(clause) / avg_len))
"""

from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # how quickly repeated words stop adding score
        self.b = b  # how strongly long documents are penalised
        self.doc_tf: list[Counter[str]] = []
        self.doc_len: list[int] = []
        self.idf: dict[str, float] = {}
        self.avg_len = 0.0

    def fit(self, docs: list[list[str]]) -> "BM25":
        self.doc_tf = [Counter(d) for d in docs]
        self.doc_len = [len(d) for d in docs]
        self.avg_len = sum(self.doc_len) / max(len(docs), 1)
        df: Counter[str] = Counter()
        for tf in self.doc_tf:
            df.update(tf.keys())
        n = len(docs)
        # The "+1" inside the log keeps IDF positive even for words in most documents.
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        return self

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.doc_tf, self.doc_len):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
            for w in set(query):
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out
