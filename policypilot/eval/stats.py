"""Confidence intervals and significance tests for the evaluation, as fixed in docs/EVAL_PLAN.md.

Plain English: 25 requirements is a small sample. If we had picked 25 slightly different ones,
"96%" might have been 88% or 100%. A confidence interval shows that range. We build it by
re-drawing the requirements at random, with replacement, 10,000 times and recomputing the metric
each time (the bootstrap). The middle 95% of those results is the interval.

To compare the AI agent with the baseline we pair them requirement by requirement and only look at
requirements where exactly one of them was right (McNemar's test). If the agent wins most of
those, the gap is unlikely to be luck.
"""

from __future__ import annotations

from math import comb

import numpy as np

N_BOOT = 10_000
SEED = 0


def requirement_counts(detail: dict) -> dict[str, int]:
    """Per-requirement counts that the end-to-end metrics are built from."""
    predicted = detail["predicted"]
    relevant = set(detail["relevant"])
    ok = relevant | set(detail["acceptable"])
    tp = sum(1 for c in predicted if c in ok)
    fp = len(predicted) - tp
    found = len(relevant & predicted.keys())
    applicable = bool(relevant)
    primary = detail["relevant"][0] if applicable else None
    impact_total = int(primary in predicted) if applicable else 0
    impact_hit = int(impact_total and predicted[primary] == detail["expected_impact"])
    if applicable:
        fully_correct = found == len(relevant) and fp == 0 and impact_hit == 1
    else:
        fully_correct = not predicted
    return {"tp": tp, "fp": fp, "found": found, "relevant": len(relevant), "impact_hit": impact_hit,
            "impact_total": impact_total, "abstain_hit": int(not applicable and not predicted),
            "abstain_total": int(not applicable), "fully_correct": int(fully_correct)}


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


E2E_METRICS = {
    # metric: (numerator count, counts summed for the denominator); None counts each requirement once
    "precision": ("tp", ("tp", "fp")),
    "recall": ("found", ("relevant",)),
    "impact_accuracy": ("impact_hit", ("impact_total",)),
    "abstention": ("abstain_hit", ("abstain_total",)),
    "fully_correct": ("fully_correct", None),
}


def e2e_intervals(details: list[dict], n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, dict]:
    """Point estimate and 95% bootstrap interval for each end-to-end metric.

    Resamples requirements, then recomputes each ratio from summed counts, so a requirement with
    three relevant clauses weighs three times as much in recall, exactly as in the point estimate.
    """
    rows = [requirement_counts(d) for d in details]
    keys = rows[0].keys()
    counts = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    n = len(rows)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    out = {}
    for metric, (num_key, den_keys) in E2E_METRICS.items():
        num = counts[num_key]
        den = np.ones(n) if den_keys is None else sum(counts[k] for k in den_keys)
        if den.sum() == 0:
            out[metric] = {"value": None, "low": None, "high": None, "n": 0}
            continue
        boot = _ratio(num[idx].sum(axis=1), den[idx].sum(axis=1))
        low, high = np.nanpercentile(boot, [2.5, 97.5])
        out[metric] = {"value": float(num.sum() / den.sum()), "low": float(low), "high": float(high),
                       "n": int(den.sum())}
    return out


def mean_intervals(rows: list[dict], keys: list[str], n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, dict]:
    """Bootstrap intervals for per-requirement averages, such as retrieval recall@k and MRR."""
    n = len(rows)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    out = {}
    for key in keys:
        values = np.array([r[key] for r in rows], dtype=float)
        low, high = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
        out[key] = {"value": float(values.mean()), "low": float(low), "high": float(high), "n": n}
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b and c are the discordant pairs: requirements only the
    first system got right, and requirements only the second got right."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(details_a: list[dict], details_b: list[dict]) -> dict:
    """Pairs two runs by requirement and tests whether their 'fully correct' rates differ."""
    a = {(d["regulation"], d["requirement"]): requirement_counts(d)["fully_correct"] for d in details_a}
    b = {(d["regulation"], d["requirement"]): requirement_counts(d)["fully_correct"] for d in details_b}
    shared = sorted(a.keys() & b.keys())
    only_a = sum(1 for k in shared if a[k] and not b[k])
    only_b = sum(1 for k in shared if b[k] and not a[k])
    return {"pairs": len(shared), "only_first": only_a, "only_second": only_b,
            "both": sum(1 for k in shared if a[k] and b[k]), "neither": sum(1 for k in shared if not a[k] and not b[k]),
            "p_value": mcnemar_exact(only_a, only_b)}
