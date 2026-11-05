"""MCP benchmark: how long each tool takes, as a client sees it, over stdio.

Plain English: start `python -m policypilot mcp` as a subprocess, exactly as Claude Desktop does,
and time every tool call from the client's side (send the request, wait for the answer). The time
includes JSON-RPC over the pipes, argument validation, the work itself and encoding the result.
It also records one end-to-end session, request by request, for the README.

The server uses this machine's normal settings (fastembed embedder if installed, reranker off by
default) and a throwaway analyses database, so a run leaves no saved analyses behind.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import tempfile
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

import anyio

from ..config import PROJECT_ROOT, Settings
from ..ingest import load_regulations

OUT = PROJECT_ROOT / "docs" / "MCP_RESULTS.md"
TRIM = 160  # long strings in the transcript are cut to this many characters

R2_QUOTE = "Personal data shall not be kept for longer than three (3) years after the end of the customer relationship"
CLAUSE_QUOTE = "Customer personal data is retained for ten (10) years after the account is closed"
FINDING = {
    "requirement_id": "R2", "clause_id": "data-privacy§3.2", "impact": "conflict",
    "explanation": "The policy keeps customer data for ten years after closure; the regulation allows three.",
    "regulation_quote": R2_QUOTE, "policy_quote": CLAUSE_QUOTE,
    "proposed_text": "Customer personal data is retained for three (3) years after the account is closed, "
                     "unless a longer period is expressly required by law.",
}


def _trim(value):
    if isinstance(value, str):
        return value if len(value) <= TRIM else value[:TRIM] + "…"
    if isinstance(value, list):
        return [_trim(v) for v in value[:6]] + ([f"… {len(value) - 6} more"] if len(value) > 6 else [])
    if isinstance(value, dict):
        return {k: _trim(v) for k, v in value.items()}
    return value


async def _session(settings: Settings, repeats: int, db: Path) -> dict:
    from mcp import Client, StdioServerParameters

    env = {**os.environ, "PP_DATA_DIR": str(settings.data_dir), "PP_MCP_DB": str(db), "PYTHONIOENCODING": "utf-8",
           "PYTHONPATH": str(PROJECT_ROOT)}
    params = StdioServerParameters(command=sys.executable, args=["-m", "policypilot", "mcp"], env=env)
    timings: dict[str, list[float]] = {}
    transcript: list[dict] = []

    async def timed(client, name, args, record=False):
        t0 = time.perf_counter()
        r = await client.call_tool(name, args)
        ms = (time.perf_counter() - t0) * 1000
        if r.is_error:
            raise RuntimeError(f"{name} failed: {r.content[0].text}")
        timings.setdefault(name, []).append(ms)
        if record:
            transcript.append({"tool": name, "arguments": args, "result": r.structured_content, "ms": ms})
        return r.structured_content

    t0 = time.perf_counter()
    async with Client(params) as c:
        connect_ms = (time.perf_counter() - t0) * 1000
        tools = (await c.list_tools()).tools
        # The first call loads the policy index (and the embedding model); time it on its own.
        t0 = time.perf_counter()
        await c.call_tool("search_clauses", {"query": "warm-up", "k": 1})
        first_call_ms = (time.perf_counter() - t0) * 1000

        # ---- the end-to-end session shown in the README
        aid = (await timed(c, "start_analysis", {"regulation_id": "reg-01-data-protection"}, True))["analysis_id"]
        await timed(c, "search_clauses", {"query": "how long personal data may be kept", "k": 3}, True)
        await timed(c, "read_clause", {"clause_id": "data-privacy§3.2"}, True)
        await timed(c, "submit_findings", {"analysis_id": aid, "findings": [
            {**FINDING, "policy_quote": "Customer data is kept for ten years after closure"}]}, True)
        await timed(c, "submit_findings", {"analysis_id": aid, "findings": [FINDING]}, True)
        await timed(c, "get_analysis", {"analysis_id": aid}, True)

        # ---- repeated calls for the latency table
        queries = [r.text for reg in load_regulations(settings.regulations_dir) for r in reg.requirements]
        for i in range(repeats):
            q = queries[i % len(queries)][:500]
            hit = (await timed(c, "search_clauses", {"query": q, "k": 5}))["hits"][0]
            await timed(c, "read_clause", {"clause_id": hit["clause_id"]})
            await timed(c, "verify_quote", {"quote": CLAUSE_QUOTE, "clause_id": "data-privacy§3.2"})
            await timed(c, "verify_quote", {"quote": "Customer data is kept for ten years", "clause_id": "data-privacy§3.2"})
            bench_id = (await timed(c, "start_analysis", {"regulation_id": "reg-01-data-protection"}))["analysis_id"]
            await timed(c, "submit_findings", {"analysis_id": bench_id, "findings": [FINDING]})
            await timed(c, "get_analysis", {"analysis_id": bench_id})
            await timed(c, "list_analyses", {"limit": 20})
            await timed(c, "remove_finding", {"analysis_id": bench_id, "requirement_id": "R2",
                                              "clause_id": "data-privacy§3.2"})
        for i in range(max(3, repeats // 3)):
            await timed(c, "map_regulation", {"regulation_id": "reg-01-data-protection", "engine": "offline"})
    return {"connect_ms": connect_ms, "first_call_ms": first_call_ms, "timings": timings,
            "transcript": transcript, "tools": [t.name for t in tools],
            "method": transcript[1]["result"]["method"]}


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def run(settings: Settings, repeats: int = 30) -> str:
    from ..retrieval import PolicyIndex

    stats = PolicyIndex.build(settings).stats()
    with tempfile.TemporaryDirectory() as tmp:
        out = anyio.run(_session, settings, repeats, Path(tmp) / "bench.sqlite3")

    order = ["search_clauses", "read_clause", "verify_quote", "start_analysis", "submit_findings",
             "get_analysis", "list_analyses", "remove_finding", "map_regulation"]
    rows = []
    for name in order:
        xs = out["timings"].get(name, [])
        if xs:
            rows.append(f"| `{name}` | {len(xs)} | {statistics.median(xs):.1f} | {_pct(xs, 95):.1f} | {max(xs):.1f} |")

    steps = []
    for n, step in enumerate(out["transcript"], 1):
        steps.append(f"**{n}. `{step['tool']}`** ({step['ms']:.1f} ms)\n\n```json\n"
                     + json.dumps({"arguments": step["arguments"]}, ensure_ascii=False, indent=2)
                     + "\n```\n\n```json\n"
                     + json.dumps({"result": _trim(step["result"])}, ensure_ascii=False, indent=2) + "\n```\n")

    md = f"""# MCP server: measured latency and a recorded session

Generated by `python -m policypilot mcp-bench --repeats {repeats}` on {date.today().isoformat()}.
Every time is measured by an MCP client that started `python -m policypilot mcp` as a subprocess and
talked to it over stdio, as Claude Desktop does: request sent → result received, so it includes
JSON-RPC over the pipes, schema validation, the work and result encoding.

Setup: {platform.system()} {platform.release()}, {platform.machine()}, {os.cpu_count()} logical CPUs,
Python {platform.python_version()}, mcp {version('mcp')}. Library: {stats['clauses']} clauses from
{len(stats['policies'])} policies. Embedder `{stats['embedder']}`, reranker `{stats['reranker'] or 'off'}`
(search method `{out['method']}`), dense backend `{stats['dense_backend']}`.

## Latency per tool (ms)

| Tool | Calls | p50 | p95 | max |
|---|---|---|---|---|
{chr(10).join(rows)}

- Starting the server and connecting: {out['connect_ms']:.0f} ms. The first tool call also loads
  the policy index and the embedding model: {out['first_call_ms']:.0f} ms. Neither is in the table.
- `verify_quote` counts both a quote that is found and one that is not (which also searches for
  the closest sentence). `map_regulation` ran with `engine="offline"` (the no-AI baseline over five
  requirements); `engine="llm"` was not measured, since its time is dominated by the model provider.
- `start_analysis`, `submit_findings` and `remove_finding` write to SQLite on each call.

## A recorded session

The client-as-agent workflow on regulation `reg-01-data-protection`, requirement R2. Step 4 hands
in a paraphrased policy quote, which the fact-checker rejects with the closest real sentence;
step 5 resubmits it word for word and it is accepted, replacing the first attempt. Strings longer
than {TRIM} characters and lists longer than 6 items are cut in the results below.

{chr(10).join(steps)}"""
    OUT.write_text(md, encoding="utf-8", newline="\n")
    return md
