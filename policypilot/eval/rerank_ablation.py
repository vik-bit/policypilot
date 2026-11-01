"""Reranker ablation: is the cross-encoder worth its cost, and which one?

Plain English: the reranker re-reads the best search candidates to put them in a sharper order.
On the 51-clause sample library it helped a little; with 10,000 extra clauses it was 20 times
slower than plain hybrid search and found fewer right clauses. This runs seven configurations
(no reranker, or one of three cross-encoders re-reading the top 20 or only the top 5) at four
library sizes, and applies the decision rule fixed in docs/EVAL_PLAN.md before the run.

Retrieval only: no LLM is called. The scale library's vectors come from embed_library's cache.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np

from ..config import PROJECT_ROOT, Settings
from ..retrieval import PolicyIndex
from ..retrieval.rerank import CrossEncoderReranker
from .run_eval import eval_retrieval
from .scale import embed_library
from .stats import N_BOOT, SEED

CURRENT = ("Xenova/ms-marco-MiniLM-L-6-v2", 20)
MODELS = ("Xenova/ms-marco-MiniLM-L-6-v2", "Xenova/ms-marco-MiniLM-L-12-v2", "BAAI/bge-reranker-base")
DEPTHS = (20, 5)
DISTRACTORS = (0, 1000, 5000, 10000)
MARGIN = 0.04  # one requirement out of the 25 in the sample suite
SUITES = ("sample", "real", "adversarial")


def _name(model: str | None, depth: int | None) -> str:
    return "hybrid, no rerank" if model is None else f"{model.split('/')[-1]} @ top {depth}"


def _score(index: PolicyIndex, settings: Settings, suite: str) -> dict:
    method = "hybrid+rerank" if index.reranker else "hybrid"
    r = eval_retrieval(index, settings, suite=suite, methods=(method,))[method]
    return {"recall@1": r["recall@1"], "recall@3": r["recall@3"], "recall@5": r["recall@5"], "mrr": r["mrr"],
            "ms_p50": r["ms_p50"], "ms_p95": r["ms_p95"],
            "per_query": [q["recall@3"] for q in r["per_query"]]}


def _paired_interval(a: list[float], b: list[float]) -> dict:
    """Mean of b - a over requirements, with a 95% percentile bootstrap interval."""
    diff = np.array(b) - np.array(a)
    idx = np.random.default_rng(SEED).integers(0, len(diff), size=(N_BOOT, len(diff)))
    low, high = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return {"value": float(diff.mean()), "low": float(low), "high": float(high)}


def choose(rows: list[dict], margin: float = MARGIN) -> tuple[float, list[dict], dict]:
    """The pre-registered rule: of the configurations within `margin` of the best score, the fastest."""
    best = max(r["score"] for r in rows)
    eligible = [r for r in rows if r["score"] >= best - margin - 1e-9]
    return best, eligible, min(eligible, key=lambda r: r["latency"])


def run(settings: Settings) -> str:
    embedder, policy_clauses, clauses, vectors, _ = embed_library(settings, max(DISTRACTORS))
    sizes = [len(policy_clauses) + n for n in DISTRACTORS]
    configs: list[tuple[str | None, int | None]] = [(None, None)] + [(m, d) for m in MODELS for d in DEPTHS]

    rows = []
    rerankers: dict[str, CrossEncoderReranker] = {}
    for model, depth in configs:
        if model is not None and model not in rerankers:
            rerankers[model] = CrossEncoderReranker(model)
        reranker = rerankers.get(model)
        row = {"name": _name(model, depth), "model": model, "depth": depth, "sizes": {}}
        for size in sizes:
            index = PolicyIndex(clauses[:size], embedder, vectors[:size], reranker,
                                **({"rerank_depth": depth} if depth else {}))
            # Only `sample` decides; the other suites are scored at the smallest and largest size.
            suites = SUITES if size in (sizes[0], sizes[-1]) else ("sample",)
            row["sizes"][size] = {suite: _score(index, settings, suite) for suite in suites}
            s = row["sizes"][size]["sample"]
            print(f"  {row['name']:<36} {size:>6,} clauses: recall@3 {s['recall@3']:.2f}, "
                  f"p50 {s['ms_p50']:.0f} ms", flush=True)
        row["score"] = sum(row["sizes"][n]["sample"]["recall@3"] for n in sizes) / len(sizes)
        row["latency"] = row["sizes"][sizes[-1]]["sample"]["ms_p50"]
        rows.append(row)

    best, eligible, chosen = choose(rows)
    current = next(r for r in rows if (r["model"], r["depth"]) == CURRENT)
    for r in rows:
        r["eligible"] = any(r is e for e in eligible)
        r["vs_current"] = {
            suite: {size: _paired_interval(current["sizes"][size][suite]["per_query"],
                                           r["sizes"][size][suite]["per_query"])
                    for size in (sizes[0], sizes[-1])}
            for suite in ("sample", "real")
        }

    results = {"date": date.today().isoformat(), "embedder": embedder.name, "sizes": sizes, "margin": MARGIN,
               "best_score": best, "chosen": chosen["name"], "current": current["name"], "rows": rows}
    (settings.data_dir / "eval" / "rerank_ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = write_markdown(json.loads(json.dumps(results)))
    (PROJECT_ROOT / "docs" / "RERANK_ABLATION.md").write_text(md, encoding="utf-8")
    return md


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _diff(d: dict) -> str:
    return f"{d['value'] * 100:+.0f} [{d['low'] * 100:+.0f}, {d['high'] * 100:+.0f}]"


def write_markdown(results: dict) -> str:
    """Render results as loaded from rerank_ablation.json (so library sizes are string keys)."""
    sizes = results["sizes"]
    small, large = sizes[0], sizes[-1]
    rows = results["rows"]

    def get(r: dict, size: int, suite: str) -> dict:
        return r["sizes"][str(size)][suite]
    lines = [
        "# Reranker ablation",
        "",
        f"_Generated by `python -m policypilot ablate-rerank` on {results['date']}. Retrieval only, no LLM calls. "
        f"Embedder `{results['embedder']}`. Library: the 51 Northwind Pay clauses plus up to 10,000 LEDGAR "
        "distractors, as in the scale benchmark. One run on a laptop (Intel i7-1255U, WSL2). The configurations "
        "and the decision rule were fixed in [EVAL_PLAN.md](EVAL_PLAN.md#reranker-ablation) before the run._",
        "",
        "## Decision data: `sample` suite (25 requirements)",
        "",
        "Score = recall@3 averaged over the four library sizes. Latency = p50 per search at "
        f"{large:,} clauses. Eligible = score within {results['margin']:.2f} of the best "
        f"({_pct(results['best_score'])}).",
        "",
        "| Configuration | " + " | ".join(f"recall@3 at {n:,}" for n in sizes)
        + f" | Score | MRR at {large:,} | p50 / p95 ms at {large:,} | Eligible |",
        "|---|" + "---|" * (len(sizes) + 4),
    ]
    for r in rows:
        s = get(r, large, "sample")
        lines.append(f"| {r['name']} | " + " | ".join(_pct(get(r, n, 'sample')['recall@3']) for n in sizes)
                     + f" | **{_pct(r['score'])}** | {s['mrr']:.2f} | {s['ms_p50']:.0f} / {s['ms_p95']:.0f} | "
                     + ("yes" if r["eligible"] else "no") + " |")
    lines += [
        "",
        f"**Rule outcome:** the fastest eligible configuration is **{results['chosen']}** "
        f"(current default: {results['current']}).",
        "",
        "## Reported, not used to decide: `real` (35 applicable requirements, draft labels) and `adversarial` (5)",
        "",
        f"| Configuration | real recall@3 at {small} | real MRR at {small} | real recall@3 at {large:,} "
        f"| adversarial recall@3 at {small} | p50 ms at {small} (real) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        rs, rl, adv = get(r, small, "real"), get(r, large, "real"), get(r, small, "adversarial")
        lines.append(f"| {r['name']} | {_pct(rs['recall@3'])} | {rs['mrr']:.2f} | {_pct(rl['recall@3'])} | "
                     f"{_pct(adv['recall@3'])} | {rs['ms_p50']:.0f} |")
    lines += [
        "",
        "## Paired difference from the current default",
        "",
        "Recall@3 of each configuration minus the current default's, in points, paired by requirement, "
        "with a 95% bootstrap interval (10,000 resamples, seed 0). An interval that contains 0 means the "
        "difference could be chance.",
        "",
        f"| Configuration | sample at {small} | sample at {large:,} | real at {small} | real at {large:,} |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r["name"] == results["current"]:
            continue
        v = r["vs_current"]
        lines.append(f"| {r['name']} | " + " | ".join(
            _diff(v[suite][str(size)]) for suite in ("sample", "real") for size in (small, large)) + " |")
    lines.append("")
    return "\n".join(lines)
