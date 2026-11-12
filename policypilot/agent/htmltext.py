"""Turn a web page's HTML into plain text for the agent, using only the standard library.

Regulator sites are often old ASP.NET pages that article extractors reduce to the page title
(the Fetch server's simplified mode returned only "Notifications | Official Website of Reserve
Bank of India" for an RBI circular). This keeps all visible text, drops scripts, styles, menus and
form controls (not <form> itself: ASP.NET wraps the whole page in one), and keeps block structure
as line breaks, so the circular's own paragraphs come through intact.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

_SKIP = {"script", "style", "noscript", "head", "nav", "header", "footer", "select", "option",
         "button", "svg", "iframe", "template"}
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article",
          "blockquote", "ul", "ol", "dd", "dt", "hr", "pre"}
_VOID = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP and tag not in _VOID:
            self.skip += 1
        elif tag in _BLOCK:
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag in _SKIP and tag not in _VOID:
            self.skip = max(0, self.skip - 1)
        elif tag in _BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def looks_like_html(text: str) -> bool:
    head = text[:20_000].lower()
    return ("<html" in head or "<!doctype html" in head
            or sum(head.count(t) for t in ("<div", "<p", "<input", "<table", "<span")) >= 5)


def looks_like_pdf(text: str) -> bool:
    """The Fetch server passes a PDF through as text; its bytes are unusable by then."""
    return "%PDF-" in text[:1000]


def html_to_text(html: str) -> str:
    parser = _Text()
    parser.feed(html)
    parser.close()
    text = "".join(parser.out).replace("\xa0", " ")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip(" |") for line in text.split("\n")]
    out, blank = [], False
    for line in lines:
        if line:
            out.append(line)
            blank = False
        elif not blank and out:
            out.append("")
            blank = True
    return "\n".join(out).strip()
