"""
Author: L. Saetta
Date last modified: 2026-09-01
License: MIT
Description: Unit tests for CV-grounded dataset preparation helpers.
"""

from dataset_preparation.build_cv_qa_datasets import (
    AtomicFact,
    build_records,
    contains_contact_detail,
    chunk_page_text,
    complete_question_set,
    make_fact_id,
    parse_json_payload,
    split_facts,
    validate_records,
)


def test_parse_json_payload_ignores_surrounding_text() -> None:
    """Parse the JSON object embedded in model output."""

    assert parse_json_payload("Result: {\"facts\": []}") == {"facts": []}


def test_contact_detail_detection_covers_private_values() -> None:
    """Detect direct contact details while accepting a professional fact."""

    assert contains_contact_detail("Contact me at person@example.com")
    assert contains_contact_detail("Phone: +39 123 456 7890")
    assert contains_contact_detail("https://example.com/profile")
    assert not contains_contact_detail("Led a machine-learning platform project.")


def test_chunk_page_text_preserves_all_non_empty_lines() -> None:
    """Split long extracted text without dropping source lines."""

    page_text = "alpha\nbeta\n" + ("gamma " * 600)
    chunks = chunk_page_text(page_text, maximum_characters=100)

    assert len(chunks) > 1
    assert "alpha" in chunks[0]
    assert "beta" in chunks[0]


def test_question_completion_references_only_the_verified_fact() -> None:
    """Complete a short model output with deterministic, fact-grounded questions."""

    fact = AtomicFact("p01-c", "Oracle HPC and Big Data Certification, 2020", 1)
    questions = complete_question_set(fact, ["Which certification is listed in the CV?"], 4)

    assert len(questions) == 4
    assert all(question.endswith("?") for question in questions)


def test_split_facts_is_deterministic_and_disjoint() -> None:
    """Keep facts disjoint between deterministic train and evaluation splits."""

    facts = [AtomicFact(make_fact_id(1, f"Fact number {index}"), f"Fact number {index}", 1) for index in range(8)]
    train_facts, eval_facts = split_facts(facts, eval_fraction=0.25, seed=42)
    second_train, second_eval = split_facts(facts, eval_fraction=0.25, seed=42)

    assert [fact.fact_id for fact in train_facts] == [fact.fact_id for fact in second_train]
    assert [fact.fact_id for fact in eval_facts] == [fact.fact_id for fact in second_eval]
    assert {fact.fact_id for fact in train_facts}.isdisjoint(fact.fact_id for fact in eval_facts)


def test_records_are_conversational_and_validate() -> None:
    """Create a valid conversational record bound to a verified fact."""

    fact = AtomicFact("p01-a", "Led a machine-learning platform project.", 1)
    records = build_records([fact], {fact.fact_id: ["What project did the candidate lead?"]}, "train", 10)

    validate_records(records)
    assert records[0]["messages"][-1]["content"] == fact.text


def test_record_validation_rejects_contact_detail_in_question() -> None:
    """Reject an LLM-generated question that exposes direct contact information."""

    fact = AtomicFact("p01-b", "Led a machine-learning platform project.", 1)
    records = build_records([fact], {fact.fact_id: ["Which email address can be contacted?"]}, "train", 10)

    records[0]["messages"][1]["content"] = "Which email address is person@example.com?"
    try:
        validate_records(records)
    except ValueError as error:
        assert "Contact detail" in str(error)
    else:
        raise AssertionError("Expected contact detail validation to fail.")
