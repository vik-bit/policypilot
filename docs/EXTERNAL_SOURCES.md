# External sources: the agent consults MCP servers on the internet

A regulation rarely stands alone. An RBI notification amends a master direction, a CERT-In
direction leans on definitions in the IT Act, and a reviewer's decision often turns on that other
text. With `PP_SOURCES=on`, the AI agent (Claude or Groq) can open those documents on the
regulator's own website while it decides, through ready-made MCP servers rather than scrapers of
our own. Code: [`policypilot/agent/external.py`](../policypilot/agent/external.py).

## What connects

| Source | How | Default |
|---|---|---|
| **Fetch** ([`mcp-server-fetch`](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch), the MCP project's official server, pinned to 2026.10.28) | started over stdio for each analysis | on when `PP_SOURCES=on`, limited to regulator domains |
| Any other MCP server | a `command` (stdio) or `url` (Streamable HTTP, with headers such as `Authorization: Bearer ${TOKEN}`) in `sources.json` | off; see [`sources.example.json`](../sources.example.json) |

Default allowlist for Fetch: rbi.org.in, sebi.gov.in, cert-in.org.in, meity.gov.in,
indiacode.nic.in, egazette.gov.in, npci.org.in, irdai.gov.in, dpdpa.gov.in, federalregister.gov,
ecfr.gov, eur-lex.europa.eu, legislation.gov.uk (each with its subdomains).

```bash
python -m policypilot sources
```
```bash
python -m policypilot sources --fetch "https://rbi.org.in/Scripts/NotificationUser.aspx?Id=11040"
```

## Guard rails

- **Allowlists, checked before any request.** The agent sees only the tools a source's config
  names (a server's other tools are never offered). Every URL in the arguments must be https, on an
  allowed host or its subdomain, with no credentials and no IP address; a source with no allowlist
  may not be given URLs at all. A refused call never reaches the server (tested).
- **A budget.** At most `PP_SOURCES_MAX_CALLS` (8) network calls per analysis; each result is cut
  into parts of `PP_SOURCES_MAX_CHARS` (12,000) characters, and reading the next part of a page
  already fetched costs no call; `PP_SOURCES_TIMEOUT` (45 s) per call.
- **Untrusted data, never evidence.** A result reaches the model as one JSON string in a tool
  result, labelled as untrusted. The system prompt says to ignore instructions in it and that it is
  background only. The fact-checker accepts quotes only from the regulation and the policy
  clauses, so a quote lifted from a web page is rejected (tested with a scripted agent).
- **Third-party tool descriptions are untrusted too.** A server's own description is cut to 500
  characters, or replaced by the config's `description` (the default Fetch source uses its own).
- **Audit trail.** Every call, refused or not, appears live in the trace and is saved in the report
  as `external_sources` (source, tool, arguments, characters returned, time).
- **Off unless asked.** `PP_SOURCES` defaults to `off`, and the offline analyser never opens sources.

## Why PolicyPilot converts the HTML itself

The Fetch server's simplified mode reduced the RBI circular behind `real-03` to its page title,
"Notifications | Official Website of Reserve Bank of India" (132 characters). The raw HTML had the
whole circular, so the default Fetch source asks for raw HTML (`fixed_arguments`, hidden from the
agent) and [`htmltext.py`](../policypilot/agent/htmltext.py) turns it into text with the standard
library, keeping the circular's paragraphs and table rows and dropping scripts and menus.

## Measured on the real-regulation sources

The source URLs cited by the bundled real-regulation set, fetched once each on 11 November 2026
from Windows 11 through the full path (allowlist → Fetch server over stdio → HTML to text). Starting
the Fetch server and connecting took 1.9 s.

| Page | Time | Text |
|---|---|---|
| RBI customer liability circular, 2017 (`Id=11040`) | 1.3 s | 19,252 characters |
| RBI turnaround-time circular, 2019 (`Id=11693`) | 2.4 s | 12,029 characters |
| RBI Master Direction on digital payment security (`id=12032`) | 4.1 s | 44,812 characters |
| RBI Master Direction on KYC (`id=11566`) | 8.1 s | 228,090 characters |
| CERT-In directions of 28 April 2022 (PDF) | 1.2 s | refused: PDF (see limitations) |

An earlier run the same day took 5.5 to 6.5 s for each of the first three pages, so the RBI site's
response time varies a lot. Each time includes a robots.txt check, which the Fetch server makes
before every autonomous request.

## Limitations

- **PDFs are not read.** The Fetch server passes a PDF through as text, which leaves it unusable;
  PolicyPilot reports "This URL is a PDF" instead. CERT-In and the Gazette publish mainly PDFs. A
  PDF-capable MCP server added to `sources.json` would cover them.
- **Redirects.** The allowlist checks the URL the agent asks for. The Fetch server follows
  redirects and blocks private and loopback addresses on every hop, but a redirect from an allowed
  site to another public site would be followed.
- **Not yet measured with a model.** The agent loop is tested with scripted Groq replies (the
  Claude loop shares the same tool code but has no test with sources); no run with a real model and sources turned on has been scored. Whether consulting
  sources improves accuracy is open: it needs `eval --suite real` with `PP_SOURCES=on` and off.
- **Paid legal-research MCP servers** (several are listed in the MCP registry) need their own
  accounts and were not tested. The Streamable HTTP connection path (`url` and `headers`) has no
  test yet; the tests cover in-process and stdio servers.
