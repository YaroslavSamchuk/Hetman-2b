"""Tests fail-closed and exact-span benchmark contamination checks."""

from data_curation_pipeline import DecontaminationAuditor


def test_empty_index_fails_closed() -> None:
    auditor = DecontaminationAuditor()
    try:
        auditor.check_contamination("ordinary text with fewer than thirteen words")
    except RuntimeError as exc:
        assert "index is empty" in str(exc)
    else:
        raise AssertionError("Empty benchmark index must not silently report clean data")


def test_normalized_ngram_detected_without_hash_collision_risk() -> None:
    prompt = "Which article of the constitution declares sovereignty and independence of the state in this example?"
    auditor = DecontaminationAuditor([{"benchmark": "example", "prompt": prompt}])
    contaminated, benchmark, span = auditor.check_contamination(
        "A longer source passage says WHICH article of the constitution declares sovereignty and independence "
        "of the state in this example? and then continues with more text."
    )
    assert contaminated
    assert benchmark == "example"
    assert span is not None


if __name__ == "__main__":
    test_empty_index_fails_closed()
    test_normalized_ngram_detected_without_hash_collision_risk()
    print("[PASS] Decontamination now fails closed and detects normalized exact n-grams.")
