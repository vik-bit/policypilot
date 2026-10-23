"""Step 6 — Evaluation: score the system against answers we wrote down by hand.

Plain English: `data/eval/labels.jsonl` lists, for every requirement in the sample regulations,
which policy clauses a careful human says it affects. We then measure:

- Retrieval: did search put the right clause near the top? (Recall@k, MRR)
  We run four variants side by side to show what each stage adds.
- End to end: did the final report flag the right clauses (recall), avoid wrong ones (precision),
  label them correctly (impact accuracy), and back them with real quotes (citation validity)?
  For requirements that don't apply to the company at all, we check it flagged nothing (abstention).

There are three test sets ("suites"); docs/EVAL_PLAN.md explains what each is for:
- sample: the 6 fictional regulations the system was built against;
- real: excerpts of real Indian regulations, a held-out set with draft labels;
- adversarial: a sample regulation with prompt-injection text planted in it.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import replace
from datetime import date

from ..config import PROJECT_ROOT, Settings
from ..ingest import load_regulations
from ..pipeline import analyze
from ..retrieval import METHODS, PolicyIndex


SUITES = {
    # suite: (regulations folder under data/, labels file under data/eval/, where results go)
    "sample": ("regulations", "labels.jsonl", "results.json"),
    "real": ("regulations_real", "labels_real.jsonl", "results_real.json"),
    "adversarial": ("regulations_adversarial", "labels_adversarial.jsonl", "results_adversarial.json"),
}


def suite_regulations_dir(settings: Settings, suite: str = "sample"):
    return settings.data_dir / SUITES[suite][0]


def load_labels(settings: Settings, suite: str = "sample") -> list[dict]:
    path = settings.data_dir / "eval" / SUITES[suite][1]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_provisional(labels: list[dict]) -> bool:
    """True if any label is still a draft that no person has reviewed (see docs/EVAL_PLAN.md)."""
    return any(lab.get("status") == "draft" for lab in labels)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q / 100 * (len(ordered) - 1)))]


def eval_retrieval(index: PolicyIndex, settings: Settings, ks=(1, 3, 5),
                   suite: str = "sample") -> dict[str, dict]:
    regs = {r.id: r for r in load_regulations(suite_regulations_dir(settings, suite))}
    # Requirements that apply to no clause have nothing to retrieve; abstention is scored end to end.
    labels = [lab for lab in load_labels(settings, suite) if lab["relevant"]]
    methods = [m for m in METHODS if m != "hybrid+rerank" or index.reranker]
    results: dict[str, dict] = {}
    for method in methods:
        recalls = defaultdict(list)
        rr, latencies, per_query = [], [], []
        for lab in labels:
            req = next(r for r in regs[lab["regulation"]].requirements if r.id == lab["requirement"])
            t = time.perf_counter()
            hits = [h.clause.id for h in index.search(f"{req.title}. {req.text}", k=max(ks), method=method)]
            latencies.append(time.perf_counter() - t)
            relevant = set(lab["relevant"])
            row = {"regulation": lab["regulation"], "requirement": lab["requirement"]}
            for k in ks:
                recalls[k].append(len(relevant & set(hits[:k])) / len(relevant))
                row[f"recall@{k}"] = recalls[k][-1]
            first = next((i for i, h in enumerate(hits, 1) if h in relevant), None)
            rr.append(1 / first if first else 0.0)
            row["rr"] = rr[-1]
            per_query.append(row)
        results[method] = {
            **{f"recall@{k}": sum(v) / len(v) for k, v in recalls.items()},
            "mrr": sum(rr) / len(rr),
            "ms_per_query": 1000 * sum(latencies) / len(latencies),
            "ms_p50": 1000 * percentile(latencies, 50),
            "ms_p95": 1000 * percentile(latencies, 95),
            "per_query": per_query,
        }
    return results


def eval_end_to_end(index: PolicyIndex, settings: Settings, suite: str = "sample") -> dict:
    labels = load_labels(settings, suite)
    by_req = {(lab["regulation"], lab["requirement"]): lab for lab in labels}
    tp = fp = found_relevant = total_relevant = 0
    impact_hits = impact_total = 0
    abstained = abstain_total = 0
    first_valid = first_total = final_valid = final_total = 0
    tokens = turns = 0
    per_reg, details, failures = [], [], []
    seconds = 0.0

    for reg in load_regulations(suite_regulations_dir(settings, suite)):
        started = time.perf_counter()
        try:
            report = analyze(reg, index, settings)
        except Exception as exc:  # one failed regulation shouldn't sink the whole evaluation
            failures.append({"regulation": reg.id, "error": str(exc)})
            print(f"  ✗ {reg.id}: {exc}")
            continue
        seconds += time.perf_counter() - started
        print(f"  ✓ {reg.id}: {len(report.findings)} findings in {time.perf_counter() - started:.0f}s")
        st = report.stats
        first_valid += st.first_pass_valid; first_total += st.first_pass_citations
        final_valid += st.final_valid; final_total += st.final_citations
        tokens += report.usage.get("input_tokens", 0) + report.usage.get("output_tokens", 0)
        turns += report.usage.get("turns", 0)
        reg_tp = reg_pred = 0
        for req in reg.requirements:
            lab = by_req.get((reg.id, req.id))
            if not lab:
                continue
            predicted = {f.clause_id: f.impact for f in report.findings if f.requirement_id == req.id}
            relevant, ok = set(lab["relevant"]), set(lab["relevant"]) | set(lab["acceptable"])
            tp += sum(1 for c in predicted if c in ok)
            fp += sum(1 for c in predicted if c not in ok)
            reg_tp += sum(1 for c in predicted if c in ok)
            reg_pred += len(predicted)
            found_relevant += len(relevant & predicted.keys())
            total_relevant += len(relevant)
            if not lab["relevant"]:
                # Doesn't apply to this company: the right answer is no finding at all.
                abstain_total += 1
                abstained += not predicted
            elif lab["relevant"][0] in predicted:
                impact_total += 1
                impact_hits += predicted[lab["relevant"][0]] == lab["impact"]
            details.append({"regulation": reg.id, "requirement": req.id, "predicted": predicted,
                            "relevant": lab["relevant"], "acceptable": lab["acceptable"],
                            "expected_impact": lab["impact"]})
        per_reg.append({"regulation": reg.id, "findings": len(report.findings), "correct": reg_tp,
                        "predicted": reg_pred})

    return {
        "mode": settings.resolved_mode,
        "model": settings.active_model,
        "precision": tp / max(tp + fp, 1),
        "recall": found_relevant / max(total_relevant, 1),
        "impact_accuracy": impact_hits / max(impact_total, 1),
        "abstention": abstained / abstain_total if abstain_total else None,
        "abstention_total": abstain_total,
        "citation_validity_first_pass": first_valid / max(first_total, 1),
        "citation_validity_final": final_valid / max(final_total, 1),
        "citations": final_total,
        "tokens": tokens,
        "turns": turns,
        "seconds": round(seconds),
        "regulations_completed": len(per_reg),
        "failures": failures,
        "per_regulation": per_reg,
        "details": details,
    }


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


SUITE_TITLES = {"sample": "Evaluation results", "real": "Evaluation results: real regulations",
                "adversarial": "Evaluation results: prompt-injection set"}


def write_markdown(retrieval: dict, e2e_runs: list[dict], index: PolicyIndex, n_labels: int,
                   suite: str = "sample", provisional: bool = False) -> str:
    names = {"bm25": "Keywords only (BM25)", "dense": "Meaning only (embeddings)",
             "hybrid": "Hybrid (BM25 + embeddings, RRF)", "hybrid+rerank": "Hybrid + cross-encoder rerank"}
    command = "python -m policypilot eval" + ("" if suite == "sample" else f" --suite {suite}")
    kind = "requirements with draft labels" if provisional else "hand-labelled requirements"
    lines = [
        f"# {SUITE_TITLES[suite]}",
        "",
        f"_Generated by `{command}` on {date.today().isoformat()}. "
        f"{n_labels} {kind} · {len(index.clauses)} policy clauses · "
        f"embedder `{index.embedder.name}` · reranker `{index.reranker.name if index.reranker else 'off'}`._",
        "",
    ]
    if provisional:
        lines += [
            "> **Provisional.** These labels are drafts that no person has reviewed yet "
            "(see [EVAL_PLAN.md](EVAL_PLAN.md) and [LABEL_REVIEW.md](LABEL_REVIEW.md)). "
            "Treat every number below as a first look, not a result.",
            "",
        ]
    lines += [
        "## 1. Retrieval: does search find the right clause?",
        "",
        "**Recall@k** = share of the correct clauses that appear in the top *k* results. "
        "**MRR** = how high the first correct clause ranks on average (1.0 = always first).",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | MRR | ms / query |",
        "|---|---|---|---|---|---|",
    ]
    for m, r in retrieval.items():
        lines.append(f"| {names[m]} | {_pct(r['recall@1'])} | {_pct(r['recall@3'])} | {_pct(r['recall@5'])} "
                     f"| {r['mrr']:.2f} | {r['ms_per_query']:.0f} |")
    if e2e_runs:
        label = lambda e: "Demo baseline (no AI)" if e["mode"] == "offline" else f"AI agent · `{e['model']}`"
        rows = [
            ("Recall", "recall", "Share of truly affected clauses the report flagged"),
            ("Precision", "precision", "Share of flagged clauses that really are affected"),
            ("Impact accuracy", "impact_accuracy", "Needs change / missing / compliant label matches the human label"),
            ("Quotes valid — first draft", "citation_validity_first_pass", "Found word for word before the fact-checker sent anything back"),
            ("Quotes valid — final", "citation_validity_final", "After the fact-check-and-fix loop"),
        ]
        lines += [
            "",
            "## 2. End to end: is the final report right?",
            "",
            f"Every regulation in the `{suite}` set is analysed from scratch and the report is compared with "
            + ("the draft labels." if provisional else "the human labels."),
            "",
            "| Metric | " + " | ".join(label(e) for e in e2e_runs) + " | Plain English |",
            "|---|" + "---|" * len(e2e_runs) + "---|",
        ]
        if any(e.get("abstention") is not None for e in e2e_runs):
            rows.insert(3, ("Abstention", "abstention",
                            "Requirements that don't apply to the company, correctly left without a finding"))
        for name, key, meaning in rows:
            lines.append(f"| {name} | " + " | ".join(
                "—" if e.get(key) is None else _pct(e[key]) for e in e2e_runs) + f" | {meaning} |")
        lines.append("| Citations checked | " + " | ".join(str(e["citations"]) for e in e2e_runs) + " | Quotes across all findings |")
        if any(e["tokens"] for e in e2e_runs):
            lines.append("| Cost | " + " | ".join(
                f"{e['tokens']:,} tokens · {e['turns']} turns · {e['seconds']}s" if e["tokens"] else "free · instant"
                for e in e2e_runs) + " | Across all regulations |")
        for e in e2e_runs:
            if e["failures"]:
                lines += ["", f"⚠ {label(e)}: {len(e['failures'])} regulation(s) failed and are excluded: "
                          + "; ".join(f"`{f['regulation']}` ({f['error'][:120]})" for f in e["failures"])]
    lines += ["", f"Raw numbers, including every prediction next to its label: `data/eval/{SUITES[suite][2]}`.",
              "Confidence intervals and significance tests: [EVAL_STATS.md](EVAL_STATS.md).", ""]
    return "\n".join(lines)


def run(settings: Settings, end_to_end: bool = True, suite: str = "sample", agent_runs: int = 1) -> dict:
    index = PolicyIndex.build(settings)
    retrieval = eval_retrieval(index, settings, suite=suite)
    e2e_runs: list[dict] = []
    if end_to_end:
        # Always score the free demo baseline, then the AI agent if one is configured,
        # so the README can show what the AI adds.
        baseline = replace(settings, mode="offline")
        print("Scoring the demo baseline…")
        e2e_runs.append(eval_end_to_end(index, baseline, suite))
        if settings.resolved_mode != "offline":
            # LLM runs vary, so the plan scores the agent several times and reports every run.
            for i in range(agent_runs):
                print(f"Scoring the AI agent ({settings.active_model}), run {i + 1} of {agent_runs}…")
                e2e_runs.append(eval_end_to_end(index, settings, suite))
    labels = load_labels(settings, suite)
    results = {"suite": suite, "provisional": is_provisional(labels), "retrieval": retrieval, "end_to_end": e2e_runs}
    (settings.data_dir / "eval" / SUITES[suite][2]).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md = write_markdown(retrieval, e2e_runs, index, len(labels), suite, is_provisional(labels))
    doc = "EVAL_RESULTS.md" if suite == "sample" else f"EVAL_RESULTS_{suite.upper()}.md"
    (PROJECT_ROOT / "docs" / doc).write_text(md, encoding="utf-8")
    return {"results": results, "markdown": md}
