"""Long context vs RAG: the grid of runs fixed in docs/LONG_CONTEXT_PLAN.md, and its scoring.

Plain English: for each library size and each way of filling the prompt (context variant), the same
open model reads every requirement of the `real` set once and its findings are scored against the
labels. Runs are slow on a free GPU, so every finished requirement is appended to a JSONL checkpoint
and a restarted run skips what is done. `report` turns the checkpoint into tables with bootstrap
intervals and paired tests, without calling any model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT, Settings
from ..ingest import load_policies, load_regulations
from ..longctx.context import POSITION_VARIANTS, Budget, build_context, interleave
from ..longctx.llm import LLMError, PromptTruncated
from ..longctx.reader import analyze_requirement
from ..longctx.tokens import TokenCounter
from ..retrieval import PolicyIndex
from .run_eval import is_provisional, load_labels, percentile, suite_regulations_dir
from .stats import N_BOOT, SEED, E2E_METRICS, _ratio, compare, e2e_intervals, mcnemar_exact, requirement_counts

# The pre-registered grid (docs/LONG_CONTEXT_PLAN.md). Sizes are total clauses in the library:
# the 51 policy clauses plus LEDGAR contract provisions as distractors. MAX_SIZE is the largest
# library whose `full` prompt fits the 32K budget for every requirement, measured with the Qwen
# tokenizer by `python -m policypilot.longctx tokens` before any model was run.
MAX_SIZE = 184
ALL_CONTEXT = ("rag5", "hybrid50", "full", "random", "bm25", "pos-start", "pos-middle", "pos-end")
GRID: list[tuple[int, str, int]] = (
    [(51, v, 3 if v in ("rag5", "full") else 1) for v in ALL_CONTEXT]
    + [(101, v, 1) for v in ("rag5", "hybrid50", "full", "bm25")]
    + [(MAX_SIZE, v, 3 if v in ("rag5", "full") else 1) for v in ALL_CONTEXT]
)
# A quick end-to-end check of the server and the code on the development (`sample`) set, so the
# held-out `real` set is not touched until the main grid runs.
SMOKE_GRID: list[tuple[int, str, int]] = [(51, "rag5", 1), (51, "full", 1)]
SMOKE_SUITE, SMOKE_LIMIT = "sample", 5
TEMPERATURE = 0.2


@dataclass
class Library:
    size: int
    index: PolicyIndex
    order: list  # the fixed order used by the `full` variant


def build_library(settings: Settings, size: int) -> Library:
    """The policy clauses, plus distractors up to `size` clauses, with a search index over them."""
    policy = load_policies(settings.policies_dir)
    if size <= len(policy):
        index = PolicyIndex.build(settings)
        return Library(len(index.clauses), index, list(index.clauses))
    from .scale import embed_library

    embedder, policy_clauses, clauses, vectors, _ = embed_library(settings, size - len(policy))
    index = PolicyIndex(clauses[:size], embedder, vectors[:size])
    return Library(size, index, interleave(policy_clauses, clauses[len(policy_clauses):size]))


def labelled_requirements(settings: Settings, suite: str = "real"):
    regs = {r.id: r for r in load_regulations(suite_regulations_dir(settings, suite))}
    for lab in load_labels(settings, suite):
        reg = regs[lab["regulation"]]
        req = next(r for r in reg.requirements if r.id == lab["requirement"])
        yield reg, req, lab


def _key(row: dict) -> tuple:
    return (row["size"], row["variant"], row["run"], row["regulation"], row["requirement"])


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_window(model, counter: TokenCounter, plan) -> int:
    """Sends the largest prompt of the grid once, with a fresh line at the very start so no prefix
    cache can serve it, and checks the server read all of it. Some servers (Ollama with its default
    num_ctx) silently keep only the end of a prompt that is longer than their context setting, and
    the model would then judge a library with clauses missing. Returns the tokens the server read."""
    nonce = f"Request {time.time_ns()}.\n\n"
    messages = [plan.messages[0], {**plan.messages[1], "content": nonce + plan.messages[1]["content"]}]
    sent = plan.prompt_tokens + counter.count(nonce)
    reply = model.chat(messages, 8, None)
    if reply.prompt_tokens < 0.95 * sent - 64:
        raise PromptTruncated(
            f"The server read {reply.prompt_tokens:,} prompt tokens of the {sent:,} sent: its context window is "
            "smaller than the budget. For Ollama, create the model with PARAMETER num_ctx 32768 (see the notebook).")
    return reply.prompt_tokens


def run_grid(settings: Settings, model, counter: TokenCounter, budget: Budget, out: Path,
             grid: list[tuple[int, str, int]] = GRID, suite: str = "real", log=print,
             limit: int | None = None, check: bool = True) -> int:
    """Runs every (size, variant, run) not yet in the checkpoint. Returns the number of new rows."""
    if not counter.exact:
        raise RuntimeError("The evaluation needs the model's real tokenizer, not an estimate.")
    done = {_key(r) for r in load_rows(out)}
    out.parent.mkdir(parents=True, exist_ok=True)
    new = 0
    requirements = list(labelled_requirements(settings, suite))[:limit]
    if check:
        size = max(s for s, _, _ in grid)
        biggest = build_library(settings, size)
        reg, req, _ = requirements[0]
        plan = build_context("full", reg, req, biggest.index, counter, budget, biggest.order)
        read = check_window(model, counter, plan)
        log(f"Server read the whole {size}-clause prompt ({read:,} tokens): no truncation")
    for size in sorted({s for s, _, _ in grid}):
        library = None
        for _, variant, runs in [g for g in grid if g[0] == size]:
            for run in range(1, runs + 1):
                model.temperature, model.seed = TEMPERATURE, run
                todo = [(reg, req, lab) for reg, req, lab in requirements
                        if (size, variant, run, reg.id, req.id) not in done
                        and not (variant in POSITION_VARIANTS and not lab["relevant"])]
                if not todo:
                    continue
                library = library or build_library(settings, size)
                log(f"{size} clauses · {variant} · run {run}: {len(todo)} requirements to go")
                for reg, req, lab in todo:
                    row = _run_one(reg, req, lab, library, variant, run, model, counter, budget)
                    row["model"] = model.name
                    with out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    new += 1
                    status = "did not fit" if not row["fits"] else (row.get("error") or
                             f"{len(row['predicted'])} findings, {row['seconds']:.0f}s, {row['prompt_tokens_sent']:,} tok")
                    log(f"  {reg.id} {req.id}: {status}")
    return new


def _run_one(reg, req, lab, library: Library, variant: str, run: int, model, counter, budget: Budget) -> dict:
    plan = build_context(variant, reg, req, library.index, counter, budget, library.order,
                         relevant=lab["relevant"] or None, seed=run)
    ids = [c.id for c in plan.clauses]
    row = {"size": library.size, "variant": variant, "run": run, "regulation": reg.id, "requirement": req.id,
           "relevant": lab["relevant"], "acceptable": lab["acceptable"], "expected_impact": lab["impact"],
           "fits": plan.fits, "context_clauses": len(ids), "prompt_tokens_counted": plan.prompt_tokens,
           "prefix_key": plan.prefix_key, "fallback": plan.fallback,
           # where the labelled relevant clauses sit in the prompt (0 = first clause), and whether
           # retrieval put them there at all
           "relevant_positions": [ids.index(c) for c in lab["relevant"] if c in ids],
           "relevant_in_context": sum(c in ids for c in lab["relevant"]),
           "predicted": {}, "findings": [], "error": None}
    if not plan.fits:
        return row
    for attempt in range(2):
        try:
            r = analyze_requirement(reg, req, plan, model, library.index, budget, counter)
            break
        except LLMError as exc:
            if attempt == 1:
                row["error"] = str(exc)[:300]
                return row
            time.sleep(10)
    row.update({
        "predicted": {f.clause_id: f.impact for f in r.findings},
        "findings": [f.model_dump() for f in r.findings],
        "first_total": r.first_total, "first_valid": r.first_valid,
        "final_total": r.final_total, "final_valid": r.final_valid,
        "unverified": sum(not f.verified for f in r.findings),
        "calls": r.calls, "prompt_tokens_sent": r.prompt_tokens, "completion_tokens": r.completion_tokens,
        "cached_tokens": r.cached_tokens, "first_prompt_tokens_reported": r.first_prompt_tokens, "seconds": round(r.seconds, 2), "parse_failures": r.parse_failures,
        "stopped": r.stopped,
    })
    return row


# ---------------------------------------------------------------------- scoring

def _counts(rows: list[dict]) -> dict[str, np.ndarray]:
    per = [requirement_counts(r) for r in rows]
    return {k: np.array([p[k] for p in per], dtype=float) for k in per[0]}


def paired_diff(rows_a: list[dict], rows_b: list[dict], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """B minus A for each end-to-end metric, with a paired bootstrap interval: the same resampled
    requirements are scored for both systems, and each ratio is recomputed from summed counts."""
    a = {(r["regulation"], r["requirement"]): r for r in rows_a}
    b = {(r["regulation"], r["requirement"]): r for r in rows_b}
    shared = sorted(a.keys() & b.keys())
    ca, cb = _counts([a[k] for k in shared]), _counts([b[k] for k in shared])
    n = len(shared)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    out = {"pairs": n}
    for metric, (num_key, den_keys) in E2E_METRICS.items():
        def stat(c, sel=None):
            num = c[num_key] if sel is None else c[num_key][sel]
            den = (np.ones(n) if den_keys is None else sum(c[k] for k in den_keys))
            den = den if sel is None else den[sel]
            if sel is None:
                return num.sum() / den.sum() if den.sum() else np.nan
            return _ratio(num.sum(axis=1), den.sum(axis=1))
        point = stat(cb) - stat(ca)
        boot = stat(cb, idx) - stat(ca, idx)
        if np.isnan(point):
            out[metric] = {"value": None, "low": None, "high": None}
            continue
        low, high = np.nanpercentile(boot, [2.5, 97.5])
        out[metric] = {"value": float(point), "low": float(low), "high": float(high)}
    return out


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _all_found(row: dict) -> int:
    return int(bool(row["relevant"]) and set(row["relevant"]) <= set(row["predicted"]))


def summarise(rows: list[dict]) -> dict:
    """Metrics for one (size, variant, run) cell."""
    sent = [r for r in rows if r["fits"] and not r["error"]]
    out = {"requirements": len(rows), "did_not_fit": sum(not r["fits"] for r in rows),
           "errors": sum(bool(r["error"]) for r in rows)}
    if not sent:
        return out
    out["metrics"] = e2e_intervals(rows)  # a requirement that did not fit or failed counts as no finding
    first_t = sum(r["first_total"] for r in sent); first_v = sum(r["first_valid"] for r in sent)
    final_t = sum(r["final_total"] for r in sent); final_v = sum(r["final_valid"] for r in sent)
    seconds = [r["seconds"] for r in sent]
    prompt = [r["prompt_tokens_sent"] for r in sent]
    out.update({
        "quotes_first_pass": first_v / first_t if first_t else None,
        "quotes_final": final_v / final_t if final_t else None,
        "citations": final_t,
        "unverified_findings": sum(r["unverified"] for r in sent),
        "seconds_p50": percentile(seconds, 50), "seconds_p95": percentile(seconds, 95),
        "seconds_mean": sum(seconds) / len(seconds),
        "prompt_tokens_mean": sum(prompt) / len(prompt),
        "first_prompt_tokens_mean": sum(r["prompt_tokens_counted"] for r in sent) / len(sent),
        "completion_tokens_mean": sum(r["completion_tokens"] for r in sent) / len(sent),
        # None when the server does not report cached tokens (Ollama); vLLM does
        "cached_share": (sum(r["cached_tokens"] for r in sent) / max(sum(prompt), 1)
                         if any(r["cached_tokens"] for r in sent) else None),
        "calls_mean": sum(r["calls"] for r in sent) / len(sent),
        "context_clauses_mean": sum(r["context_clauses"] for r in sent) / len(sent),
        "context_recall": (sum(r["relevant_in_context"] for r in sent)
                           / max(sum(len(r["relevant"]) for r in sent), 1)),
        "parse_failures": sum(r["parse_failures"] for r in sent),
    })
    return out


def build_results(rows: list[dict]) -> dict:
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        cells.setdefault((r["size"], r["variant"], r["run"]), []).append(r)
    summary = [{"size": s, "variant": v, "run": n, **summarise(rs)} for (s, v, n), rs in sorted(cells.items())]

    # Primary and secondary comparisons against RAG (rag5 at the same size, same run number).
    comparisons = []
    for (s, v, n), rs in sorted(cells.items()):
        base = cells.get((s, "rag5", n))
        if v == "rag5" or base is None or v in POSITION_VARIANTS:
            continue
        comparisons.append({"size": s, "variant": v, "run": n, "vs": "rag5",
                            "mcnemar": compare(base, rs), "diff": paired_diff(base, rs)})

    # Position sensitivity: start and end against middle, on "every relevant clause flagged".
    positions = []
    for s in sorted({k[0] for k in cells}):
        mid = cells.get((s, "pos-middle", 1))
        if not mid:
            continue
        tests = []
        for v in ("pos-start", "pos-end"):
            other = cells.get((s, v, 1))
            if not other:
                continue
            m = {(r["regulation"], r["requirement"]): _all_found(r) for r in mid}
            o = {(r["regulation"], r["requirement"]): _all_found(r) for r in other}
            shared = sorted(m.keys() & o.keys())
            b = sum(1 for k in shared if o[k] and not m[k]); c = sum(1 for k in shared if m[k] and not o[k])
            tests.append({"variant": v, "pairs": len(shared), "only_this": b, "only_middle": c,
                          "p_value": mcnemar_exact(b, c), "diff": paired_diff(mid, other)})
        for t, adj in zip(tests, holm([t["p_value"] for t in tests])):
            t["p_holm"] = adj
        rate = {v: (sum(_all_found(r) for r in cells[(s, v, 1)]) / len(cells[(s, v, 1)]))
                for v in POSITION_VARIANTS if (s, v, 1) in cells}
        positions.append({"size": s, "all_relevant_found": rate, "tests": tests})
    return {"summary": summary, "comparisons": comparisons, "positions": positions}


# ---------------------------------------------------------------------- report

NAMES = {"rag5": "RAG: hybrid top 5", "hybrid50": "Hybrid top 50 in context", "full": "Full library (fixed order)",
         "random": "Full library, random order", "bm25": "Full library, BM25 order", "pos-start": "Relevant at start",
         "pos-middle": "Relevant in middle", "pos-end": "Relevant at end"}


def _ci(m: dict | None) -> str:
    if not m or m.get("value") is None:
        return "—"
    return f"{100 * m['value']:.0f}% [{100 * m['low']:.0f}–{100 * m['high']:.0f}]"


def _dci(m: dict | None) -> str:
    if not m or m.get("value") is None:
        return "—"
    return f"{100 * m['value']:+.0f} [{100 * m['low']:+.0f}, {100 * m['high']:+.0f}]"


def _p(x) -> str:
    return "—" if x is None else (f"{x:.3f}" if x >= 0.001 else "<0.001")


def write_markdown(results: dict, model: str, provisional: bool, window: int) -> str:
    lines = [
        "# Long context vs RAG: results",
        "",
        f"_Generated by `python -m policypilot.longctx report` on {date.today().isoformat()} from "
        f"`data/eval/long_context_runs.jsonl`. Model `{model}`, {window:,}-token window, temperature {TEMPERATURE}. "
        "Plan: [LONG_CONTEXT_PLAN.md](LONG_CONTEXT_PLAN.md). Every number is from a real run; cells not run are absent._",
        "",
    ]
    if provisional:
        lines += ["> **Provisional.** The `real` labels are drafts no person has reviewed yet "
                  "([LABEL_REVIEW.md](LABEL_REVIEW.md)).", ""]
    lines += [
        "## 1. Every configuration",
        "",
        "Percentages with 95% bootstrap intervals over requirements. *Context recall*: share of labelled "
        "relevant clauses that were in the prompt at all. Latency is wall time per requirement, all calls included.",
        "",
        "| Clauses | Context | Run | Recall | Precision | Impact acc. | Fully correct | Quotes 1st / final "
        "| Context recall | Prompt tokens | Cached | p50 / p95 s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results["summary"]:
        if "metrics" not in r:
            lines.append(f"| {r['size']} | {NAMES[r['variant']]} | {r['run']} | did not fit ({r['did_not_fit']}) "
                         f"/ errors ({r['errors']}) | | | | | | | | |")
            continue
        m = r["metrics"]
        q = lambda x: "—" if x is None else f"{100 * x:.0f}%"
        lines.append(
            f"| {r['size']} | {NAMES[r['variant']]} | {r['run']} | {_ci(m['recall'])} | {_ci(m['precision'])} "
            f"| {_ci(m['impact_accuracy'])} | {_ci(m['fully_correct'])} | {q(r['quotes_first_pass'])} / {q(r['quotes_final'])} "
            f"| {q(r['context_recall'])} | {r['prompt_tokens_mean']:,.0f} | {'not reported' if r['cached_share'] is None else q(r['cached_share'])} "
            f"| {r['seconds_p50']:.0f} / {r['seconds_p95']:.0f} |")
    notes = [r for r in results["summary"] if r.get("did_not_fit") or r.get("errors") or r.get("unverified_findings")]
    if notes:
        lines += ["", "Requirements that did not fit, failed, or ended with findings the fact-checker could not "
                  "verify (shown to reviewers as unverified, never as verified):", ""]
        lines += [f"- {r['size']} · {NAMES[r['variant']]} · run {r['run']}: {r.get('did_not_fit', 0)} did not fit, "
                  f"{r.get('errors', 0)} errors, {r.get('unverified_findings', 0)} unverified findings" for r in notes]
    lines += [
        "",
        "## 2. Against RAG (paired by requirement)",
        "",
        "Difference = configuration minus RAG top 5 at the same library size and run, in percentage points, "
        "with a paired bootstrap 95% interval. McNemar: exact two-sided test on *fully correct requirement*.",
        "",
        "| Clauses | Context | Run | Δ Recall | Δ Precision | Δ Impact acc. | Δ Fully correct | Only RAG right "
        "| Only this right | McNemar p |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in results["comparisons"]:
        d, mc = c["diff"], c["mcnemar"]
        lines.append(f"| {c['size']} | {NAMES[c['variant']]} | {c['run']} | {_dci(d['recall'])} | {_dci(d['precision'])} "
                     f"| {_dci(d['impact_accuracy'])} | {_dci(d['fully_correct'])} | {mc['only_first']} "
                     f"| {mc['only_second']} | {_p(mc['p_value'])} |")
    lines += [
        "",
        "## 3. Position sensitivity (lost in the middle)",
        "",
        "The labelled relevant clauses are moved to the start, the middle or the end of an otherwise "
        "randomly ordered full library. Metric: share of applicable requirements with every relevant clause "
        "flagged. Start and end are each tested against middle (exact McNemar, Holm-corrected for the two tests).",
        "",
        "| Clauses | Start | Middle | End | Start vs middle p (Holm) | End vs middle p (Holm) |",
        "|---|---|---|---|---|---|",
    ]
    for p in results["positions"]:
        rate = p["all_relevant_found"]
        t = {x["variant"]: x for x in p["tests"]}
        f = lambda v: "—" if v not in rate else f"{100 * rate[v]:.0f}%"
        g = lambda v: "—" if v not in t else f"{_p(t[v]['p_value'])} ({_p(t[v]['p_holm'])})"
        lines.append(f"| {p['size']} | {f('pos-start')} | {f('pos-middle')} | {f('pos-end')} | {g('pos-start')} | {g('pos-end')} |")
    lines += ["", "Raw rows, one per requirement and configuration: `data/eval/long_context_runs.jsonl`. "
              "Aggregates: `data/eval/long_context_results.json`.", ""]
    return "\n".join(lines)


def report(settings: Settings, runs_path: Path, window: int) -> str:
    rows = load_rows(runs_path)
    if not rows:
        raise RuntimeError(f"No rows in {runs_path}; run the grid first.")
    results = build_results(rows)
    (settings.data_dir / "eval" / "long_context_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md = write_markdown(results, rows[0].get("model", "?"), is_provisional(load_labels(settings, "real")), window)
    (PROJECT_ROOT / "docs" / "LONG_CONTEXT_RESULTS.md").write_text(md, encoding="utf-8")
    return md


# ---------------------------------------------------------------------- token budget (no model calls)

def full_prompt_tokens(settings: Settings, counter: TokenCounter, budget: Budget, n_distractors: int,
                       suite: str = "real") -> list[int]:
    """Prompt tokens of the `full` variant for every labelled requirement, without building an index
    (token counts do not depend on embeddings)."""
    from ..longctx.prompts import messages_for, render_library
    from ..longctx.tokens import count_messages
    from .scale import load_distractors

    policy = load_policies(settings.policies_dir)
    distractors = load_distractors(settings.data_dir / "bench", n_distractors) if n_distractors else []
    library = render_library(interleave(policy, distractors))
    return [count_messages(counter, messages_for(library, reg, req))
            for reg, req, _ in labelled_requirements(settings, suite)]


def max_fit_size(settings: Settings, counter: TokenCounter, budget: Budget, hi: int = 2000) -> int:
    """The largest library (policy clauses + distractors) whose `full` prompt fits the budget for
    every requirement. Binary search over the number of distractors."""
    n_policy = len(load_policies(settings.policies_dir))
    if max(full_prompt_tokens(settings, counter, budget, 0)) > budget.prompt_limit:
        return 0
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if max(full_prompt_tokens(settings, counter, budget, mid)) <= budget.prompt_limit:
            lo = mid
        else:
            hi = mid - 1
    return n_policy + lo


def context_table(settings: Settings, counter: TokenCounter, budget: Budget,
                  sizes: tuple[int, ...] = (51, 101, MAX_SIZE), suite: str = "real") -> list[dict]:
    """Per size and variant, without calling a model: how many prompt tokens each variant sends,
    and what share of the labelled relevant clauses its context contains (context recall)."""
    rows = []
    requirements = list(labelled_requirements(settings, suite))
    for size in sizes:
        library = build_library(settings, size)
        for variant in ("rag5", "hybrid50", "full"):
            tokens, found, total, fits = [], 0, 0, 0
            for reg, req, lab in requirements:
                plan = build_context(variant, reg, req, library.index, counter, budget, library.order)
                ids = {c.id for c in plan.clauses}
                tokens.append(plan.prompt_tokens)
                fits += plan.fits
                found += sum(c in ids for c in lab["relevant"])
                total += len(lab["relevant"])
            rows.append({"size": library.size, "variant": variant, "prompt_tokens_mean": sum(tokens) / len(tokens),
                         "prompt_tokens_max": max(tokens), "fits": fits, "requirements": len(requirements),
                         "context_recall": found / total})
    return rows
