from policypilot.retrieval.bm25 import BM25


def test_bm25_prefers_rare_matching_terms():
    docs = [["customer", "data", "retain"], ["customer", "card", "token"], ["customer", "complaint"]]
    scores = BM25().fit(docs).scores(["token"])
    assert scores.index(max(scores)) == 1 and scores[0] == 0


def test_hybrid_search_finds_the_obvious_clause(index):
    hits = index.search("pre-debit notification before recurring payments", k=3, method="hybrid")
    assert hits[0].clause.id == "payments-ops§5.1"
    assert {"bm25", "dense", "hybrid"} <= hits[0].ranks.keys()


def test_each_method_returns_k_results(index):
    for method in ("bm25", "dense", "hybrid", "best"):
        assert len(index.search("data retention", k=4, method=method)) == 4


def test_index_is_cached_between_builds(settings, index):
    from policypilot.retrieval import PolicyIndex

    again = PolicyIndex.build(settings)
    assert [c.id for c in again.clauses] == [c.id for c in index.clauses]
    assert (settings.index_dir / "manifest.json").exists()
