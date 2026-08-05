from casefile.evals.run_retrieval_eval import run


def test_ranked_retrieval_eval_compares_thresholds_and_filters():
    result = run(
        card_text_modes=["stored"],
        min_relevances=[0.08, 0.15],
    )
    assert result["cases"] == 9
    assert len(result["strategies"]) == 2
    by_threshold = {
        item["strategy"]["min_relevance"]: item["metrics"]
        for item in result["strategies"]
    }
    assert by_threshold[0.08]["recall_at_k"] == 1.0
    assert by_threshold[0.08]["mrr"] == 1.0
    assert by_threshold[0.08]["no_result_accuracy"] < 1.0
    assert by_threshold[0.15]["query_success_rate"] == 1.0
    assert by_threshold[0.15]["filter_leakage"] == 0
    assert result["best_strategy"].endswith("t0.15")
