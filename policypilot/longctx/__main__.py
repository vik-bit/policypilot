"""Command line for long-context mode: `python -m policypilot.longctx <command>`.

    tokens              prompt sizes per library size, and the largest library that fits (no model needed)
    context             prompt tokens and context recall of rag5 / hybrid50 / full per size (no model needed)
    analyze <reg>       analyse one regulation with the whole library in context (falls back to search
                        when it does not fit)
    eval                run the pre-registered long context vs RAG grid on the `real` set (resumable)
    report              tables, intervals and paired tests from the saved runs (no model needed)

The model is any OpenAI-compatible server: --base-url http://localhost:8000/v1 (vLLM, the default)
or http://localhost:11434/v1 (Ollama). See notebooks/long_context_kaggle.ipynb for a free T4 setup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import PROJECT_ROOT, get_settings
from .context import VARIANTS, Budget
from .llm import OpenAICompatModel
from .tokens import make_counter

RUNS = PROJECT_ROOT / "data" / "eval" / "long_context_runs.jsonl"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m policypilot.longctx", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", help="OpenAI-compatible endpoint (default PP_LC_BASE_URL or localhost:8000/v1)")
    p.add_argument("--model", help="model name on that server (default PP_LC_MODEL or Qwen/Qwen2.5-7B-Instruct-AWQ)")
    p.add_argument("--structured", choices=("json_schema", "json_object", "none"),
                   help="how to ask the server for JSON (default PP_LC_STRUCTURED or json_schema)")
    p.add_argument("--window", type=int, default=32_768, help="the model's context window in tokens")
    p.add_argument("--max-output", type=int, default=1_536, help="reply tokens reserved per call")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokens"); t.add_argument("--distractors", type=int, nargs="*", default=[0, 50, 100])
    sub.add_parser("context")
    a = sub.add_parser("analyze"); a.add_argument("regulation")
    a.add_argument("--variant", choices=VARIANTS, default="auto")
    e = sub.add_parser("eval"); e.add_argument("--grid", choices=("main", "smoke"), default="main")
    e.add_argument("--out", type=Path, help=f"checkpoint file (default {RUNS.name}, or data/runs/long_context_smoke.jsonl for --grid smoke)")
    r = sub.add_parser("report"); r.add_argument("--runs", type=Path, default=RUNS)
    args = p.parse_args(argv)

    settings = get_settings()
    model = OpenAICompatModel()
    if args.base_url:
        model.base_url = args.base_url
    if args.model:
        model.name = args.model
    if args.structured:
        model.structured = args.structured
    budget = Budget(window=args.window, max_output=args.max_output, retries=settings.max_verify_retries)

    if args.cmd == "tokens":
        from ..eval.long_context import full_prompt_tokens, max_fit_size

        counter = make_counter(model.name, strict=True)
        print(f"Tokenizer {counter.name}; prompt budget {budget.prompt_limit:,} of a {budget.window:,}-token window "
              f"({budget.retries} retries x {budget.max_output:,} reply tokens reserved)")
        print("clauses  mean prompt  max prompt  fits")
        for n in args.distractors:
            counts = full_prompt_tokens(settings, counter, budget, n)
            print(f"{51 + n:>7}  {sum(counts) / len(counts):>11,.0f}  {max(counts):>10,}  "
                  f"{'yes' if max(counts) <= budget.prompt_limit else 'no'}")
        print(f"Largest library whose full prompt fits for every requirement: {max_fit_size(settings, counter, budget)} clauses")
        return 0

    if args.cmd == "context":
        from ..eval.long_context import context_table

        counter = make_counter(model.name, strict=True)
        print("| Clauses | Context | Prompt tokens (mean / max) | Fits budget | Context recall |")
        print("|---|---|---|---|---|")
        for r in context_table(settings, counter, budget):
            print(f"| {r['size']} | {r['variant']} | {r['prompt_tokens_mean']:,.0f} / {r['prompt_tokens_max']:,} "
                  f"| {r['fits']}/{r['requirements']} | {100 * r['context_recall']:.0f}% |")
        return 0

    if args.cmd == "analyze":
        from ..ingest import load_regulation
        from ..retrieval import PolicyIndex
        from .reader import analyze_long

        path = Path(args.regulation)
        if not path.exists():
            path = settings.regulations_dir / f"{args.regulation}.md"
        counter = make_counter(model.name)
        report = analyze_long(load_regulation(path), PolicyIndex.build(settings), settings, model, counter, budget,
                              args.variant, emit=lambda ev: print(f"[{ev.kind}] {ev.message}"))
        print("\n" + report.summary)
        for f in report.findings:
            print(f"  {'✓' if f.verified else '⚠'} {f.requirement_id} → {f.clause_id} [{f.impact}] {f.explanation}")
        return 0

    from ..eval import long_context as lc

    if args.cmd == "eval":
        counter = make_counter(model.name, strict=True)
        if args.grid == "main":
            out = args.out or RUNS
            n = lc.run_grid(settings, model, counter, budget, out, lc.GRID)
        else:
            out = args.out or PROJECT_ROOT / "data" / "runs" / "long_context_smoke.jsonl"  # data/runs/ is git-ignored
            n = lc.run_grid(settings, model, counter, budget, out, lc.SMOKE_GRID, suite=lc.SMOKE_SUITE,
                            limit=lc.SMOKE_LIMIT)
        print(f"{n} new rows in {out}")
        return 0
    if args.cmd == "report":
        print(lc.report(settings, args.runs, budget.window))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
