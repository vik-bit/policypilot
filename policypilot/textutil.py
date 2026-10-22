"""Small text helpers shared by retrieval and verification."""

from __future__ import annotations

import re

_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    # Every Unicode hyphen and dash becomes "-". Models often write "board‑approved" with a
    # non-breaking hyphen (U+2011) where the document has a plain one; that is the same quote.
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", "…": "...",
})

# Common English words that carry no meaning for search.
STOPWORDS = frozenset("""
a an and are as at be been being by can could did do does for from had has have if in into is it
its may must not of on or other our shall should such than that the their them then there these they
this those to under was we were where which while who will with within would you your
""".split())


def normalize(text: str) -> str:
    """Lower-case, unify curly quotes/dashes, collapse whitespace. Used for quote matching."""
    text = text.translate(_QUOTE_MAP).lower()
    text = re.sub(r"[*_`]", "", text)  # markdown emphasis
    return re.sub(r"\s+", " ", text).strip()


def _stem(word: str) -> str:
    """A deliberately tiny stemmer: 'notifications' -> 'notif', 'retained' -> 'retain'."""
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Split text into search terms: lower-case words and numbers, stopwords removed, lightly stemmed."""
    words = re.findall(r"[a-z0-9]+", normalize(text))
    return [_stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


def sentences(text: str) -> list[str]:
    """Rough sentence split, good enough for picking quotes.

    Blank lines and Markdown headings end a sentence too, so a heading is never glued onto
    the sentence after it.
    """
    out: list[str] = []
    for block in re.split(r"\n\s*\n|\n(?=#)", text):
        block = re.sub(r"^#+\s*", "", block.strip())
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", re.sub(r"\s+", " ", block))
        out += [p.strip() for p in parts if p.strip()]
    return out


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
