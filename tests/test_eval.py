import json
from dataclasses import replace

import pytest

from policypilot.eval.run_eval import SUITES, eval_end_to_end, eval_retrieval, load_labels, suite_regulations_dir
from policypilot.eval.stats import compare, e2e_intervals, mcnemar_exact, mean_intervals, requirement_counts
from policypilot.ingest import load_regulations


def _detail(predicted, relevant, acceptable=(), impact="conflict", req="R1"):
    return {"regulation": "reg", "requirement": req, "predicted": predicted, "relevant": list(relevant),
            "acceptable": list(acceptable), "expected_impact": impact}


@pytest.mark.parametrize("suite", list(SUITES))
def test_every_requirement_has_a_valid_label(settings, index, suite):
    clause_ids = {c.id for c in index.clauses}
    labels = load_labels(settings, suite)
    labelled = {(lab["regulation"], lab["requirement"]) for lab in labels}
    requirements = {(r.id, q.id) for r in load_regulations(suite_regulations_dir(settings, suite)) for q in r.requirements}
    assert labelled == requirements
    for lab in labels:
        assert set(lab["relevant"]) | set(lab["acceptable"]) <= clause_ids
        assert lab["impact"] in ({"conflict", "gap", "aligned"} if lab["relevant"] else {"not_applicable"})


def test_real_labels_stay_marked_until_reviewed(settings):
    assert {lab["status"] for lab in load_labels(settings, "real")} <= {"draft", "reviewed"}


def test_retrieval_skips_requirements_with_nothing_to_find(settings, index):
    results = eval_retrieval(index, settings, suite="real")
    n_applicable = sum(1 for lab in load_labels(settings, "real") if lab["relevant"])
    assert all(len(r["per_query"]) == n_applicable for r in results.values())
    assert all(r["ms_p50"] <= r["ms_p95"] for r in results.values())


def test_end_to_end_scores_abstention_on_the_real_suite(settings, index):
    out = eval_end_to_end(index, replace(settings, mode="offline"), "real")
    assert out["abstention_total"] == 3
    assert out["abstention"] is not None and len(out["details"]) == 38
    # The baseline flags its best search hit for every requirement, so it never abstains.
    assert out["abstention"] == 0.0


def test_requirement_counts():
    perfect = requirement_counts(_detail({"a": "conflict"}, ["a"]))
    assert perfect["fully_correct"] == 1 and perfect["tp"] == 1 and perfect["fp"] == 0
    wrong_label = requirement_counts(_detail({"a": "aligned"}, ["a"]))
    assert wrong_label["fully_correct"] == 0 and wrong_label["impact_total"] == 1
    acceptable = requirement_counts(_detail({"a": "conflict", "b": "gap"}, ["a"], ["b"]))
    assert acceptable["fp"] == 0 and acceptable["fully_correct"] == 1
    abstained = requirement_counts(_detail({}, [], impact="not_applicable"))
    assert abstained["abstain_hit"] == 1 and abstained["fully_correct"] == 1
    flagged = requirement_counts(_detail({"a": "gap"}, [], impact="not_applicable"))
    assert flagged["abstain_hit"] == 0 and flagged["fp"] == 1


def test_bootstrap_intervals_contain_the_estimate_and_collapse_when_uniform():
    details = [_detail({"a": "conflict"}, ["a"], req=f"R{i}") for i in range(20)]
    ci = e2e_intervals(details, n_boot=500)
    assert ci["precision"]["value"] == ci["precision"]["low"] == ci["precision"]["high"] == 1.0
    assert ci["abstention"]["value"] is None

    mixed = details[:10] + [_detail({"x": "conflict"}, ["a"], req=f"S{i}") for i in range(10)]
    ci = e2e_intervals(mixed, n_boot=2000)
    assert ci["precision"]["low"] < ci["precision"]["value"] == 0.5 < ci["precision"]["high"]
    rows = [{"v": float(i % 2)} for i in range(40)]
    m = mean_intervals(rows, ["v"], n_boot=2000)["v"]
    assert m["low"] < 0.5 < m["high"]


def test_bootstrap_is_reproducible():
    details = [_detail({"a": "conflict"} if i % 3 else {"x": "gap"}, ["a"], req=f"R{i}") for i in range(30)]
    assert e2e_intervals(details, n_boot=300) == e2e_intervals(details, n_boot=300)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)  # all six discordant pairs favour one side
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(1, 9) == pytest.approx(2 * (1 + 10) / 1024)


def test_compare_pairs_by_requirement():
    agent = [_detail({"a": "conflict"}, ["a"], req=f"R{i}") for i in range(6)]
    baseline = [_detail({"a": "aligned"}, ["a"], req=f"R{i}") for i in range(6)]
    c = compare(agent, baseline)
    assert (c["pairs"], c["only_first"], c["only_second"]) == (6, 6, 0)
    assert c["p_value"] == pytest.approx(2 / 64)


def test_saved_sample_results_still_match_the_published_numbers(settings):
    saved = json.loads((settings.data_dir / "eval" / "results.json").read_text(encoding="utf-8"))
    agent = next(r for r in saved["end_to_end"] if r["mode"] != "offline")
    ci = e2e_intervals(agent["details"])
    assert ci["precision"]["value"] == pytest.approx(agent["precision"])
    assert ci["recall"]["value"] == pytest.approx(agent["recall"])
    assert ci["impact_accuracy"]["value"] == pytest.approx(agent["impact_accuracy"])
