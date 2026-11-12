"""Command line: `python -m policypilot <command>`.

    ingest              build the search index from data/policies
    search "query"      try the retrieval step on its own
    analyze <reg>       analyse one regulation (an id like reg-01-data-protection, or a file path)
    eval                score the system against the hand-labelled answers
                        (--suite sample | real | adversarial; --agent-runs N)
    stats               confidence intervals and significance tests from the saved results
    scale               how search quality and speed change as the policy library grows
    serve               start the web app on http://localhost:8000
    mcp                 run as an MCP server for Claude Desktop, Claude Code or any MCP client
    mcp-bench           measure each MCP tool's latency over stdio (writes docs/MCP_RESULTS.md)
    sources             list the external MCP sources the agent can consult (--fetch URL to try one)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .config import get_settings


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # "§" and arrows print correctly on Windows
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(prog="policypilot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offline", action="store_true", help="force the offline analyser (no API calls)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=5)
    s.add_argument("--method", default="best")
    a = sub.add_parser("analyze"); a.add_argument("regulation")
    e = sub.add_parser("eval"); e.add_argument("--retrieval-only", action="store_true")
    e.add_argument("--suite", choices=("sample", "real", "adversarial"), default="sample")
    e.add_argument("--agent-runs", type=int, default=1, help="score the AI agent this many times (the plan uses 3)")
    sub.add_parser("stats")
    sc = sub.add_parser("scale"); sc.add_argument("--sizes", type=int, nargs="+", default=[0, 1000, 5000, 10000],
                                                  help="numbers of distractor clauses to add")
    sub.add_parser("ablate-rerank", help="compare reranker settings on every suite (retrieval only, no LLM calls)")
    v = sub.add_parser("serve"); v.add_argument("--port", type=int, default=8000); v.add_argument("--host", default="127.0.0.1")
    m = sub.add_parser("mcp", help="run as a Model Context Protocol server")
    m.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    m.add_argument("--port", type=int, default=8765); m.add_argument("--host", default="127.0.0.1")
    mb = sub.add_parser("mcp-bench", help="measure MCP tool latency over stdio")
    mb.add_argument("--repeats", type=int, default=30)
    so = sub.add_parser("sources", help="list the external MCP sources (PP_SOURCES) and their tools")
    so.add_argument("--fetch", metavar="URL", help="fetch one page through the 'fetch' source, with its allowlist")
    args = p.parse_args(argv)

    settings = get_settings()
    if args.offline:
        settings.mode = "offline"

    if args.cmd == "serve":
        import uvicorn

        if args.offline:
            os.environ["PP_PROVIDER"] = "offline"
        print(f"PolicyPilot running at http://{args.host}:{args.port}  (mode: {settings.resolved_mode})")
        uvicorn.run("policypilot.api.server:app", host=args.host, port=args.port)
        return 0

    if args.cmd == "mcp":
        from .mcp_server import main as run_mcp

        if args.offline:
            os.environ["PP_PROVIDER"] = "offline"
        run_mcp(args.transport, args.host, args.port)  # stdout is the protocol channel: print nothing
        return 0

    if args.cmd == "sources":
        from .agent.external import load_servers, open_sources

        servers = load_servers(settings.sources_config)
        where = settings.sources_config if settings.sources_config.exists() else "built-in default"
        print(f"PP_SOURCES={settings.sources} · config: {where}")
        with open_sources(settings, emit=lambda ev: print(f"  · {ev.message}")) as src:
            for s in servers:
                state = f"unavailable ({src.failed[s.name]})" if s.name in src.failed else "connected"
                print(f"\n{s.name}: {state}\n  hosts: {', '.join(s.allowed_hosts or ['(no URLs)'])}")
                for name, t in src.tools.items():
                    if t.server is s:
                        print(f"  tool {name}  (remote: {t.remote_name})")
            if args.fetch:
                if not src.has("fetch"):
                    print("No 'fetch' tool is configured.")
                    return 1
                text, is_error = src.call("fetch", {"url": args.fetch})
                print("\n" + text)
                return 1 if is_error else 0
        return 0

    if args.cmd == "mcp-bench":
        from .eval.mcp_bench import run as run_bench

        print(run_bench(settings, repeats=args.repeats))
        print("Written to docs/MCP_RESULTS.md")
        return 0

    from .retrieval import PolicyIndex

    if args.cmd == "ingest":
        idx = PolicyIndex.build(settings, force=True)
        st = idx.stats()
        print(f"Indexed {st['clauses']} clauses from {len(st['policies'])} policies "
              f"(embedder: {st['embedder']}, reranker: {st['reranker'] or 'off'})")
        for pol in st["policies"]:
            print(f"  {pol['policy']:<22} {pol['clauses']:>3} clauses  {pol['title']}")
        return 0

    idx = PolicyIndex.build(settings)

    if args.cmd == "search":
        for n, h in enumerate(idx.search(args.query, k=args.k, method=args.method), 1):
            print(f"{n}. {h.clause.id:<26} score={h.score:<8} ranks={h.ranks}\n   {h.clause.text[:140]}…")
        return 0

    if args.cmd == "analyze":
        from .ingest import load_regulation
        from .pipeline import analyze

        path = Path(args.regulation)
        if not path.exists():
            path = settings.regulations_dir / f"{args.regulation}.md"
        report = analyze(load_regulation(path), idx, settings, emit=lambda ev: print(f"  · {ev.message}"))
        print(f"\n{report.summary}\n")
        for f in report.findings:
            mark = "✓" if f.verified else "⚠"
            print(f"{mark} {f.requirement_id} → {f.clause_id} [{f.impact}] {f.explanation}")
            for issue in f.issues:
                print(f"     issue: {issue}")
        print(f"\nSaved: data/runs/{report.run_id}.json")
        return 0

    if args.cmd == "eval":
        from .eval.run_eval import run

        if not args.retrieval_only and settings.resolved_mode != "offline":
            print(f"Running the end-to-end eval on the {args.suite} suite with {settings.active_model}, "
                  f"{args.agent_runs} run(s) (this uses API credits)…")
        out = run(settings, end_to_end=not args.retrieval_only, suite=args.suite, agent_runs=args.agent_runs)
        print(out["markdown"])
        doc = "EVAL_RESULTS.md" if args.suite == "sample" else f"EVAL_RESULTS_{args.suite.upper()}.md"
        print(f"Written to docs/{doc}")
        return 0

    if args.cmd == "stats":
        from .eval.stats_report import write

        print(write(settings))
        print("Written to docs/EVAL_STATS.md")
        return 0

    if args.cmd == "scale":
        from .eval.scale import run as run_scale

        print(run_scale(settings, args.sizes))
        print("Written to docs/SCALE_RESULTS.md")
        return 0

    if args.cmd == "ablate-rerank":
        from .eval.rerank_ablation import run as run_ablation

        print(run_ablation(settings))
        print("Written to docs/RERANK_ABLATION.md")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
