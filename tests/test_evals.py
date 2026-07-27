from casefile.evals.generate_testset import generate
from casefile.evals.run_eval import run


def test_candidate_mix_and_golden_regression():
    candidates = generate()
    assert len(candidates) == 50
    assert [sum(item["category"] == category for item in candidates) for category in (
        "happy_path", "ambiguous", "edge", "adversarial"
    )] == [20, 15, 10, 5]
    result = run(use_llm_judge=False)
    assert result["cases"] == 20
    assert result["averages"] == {
        "citation_faithfulness": 5.0,
        "routing_authorization": 5.0,
        "evidence_integrity": 5.0,
    }
