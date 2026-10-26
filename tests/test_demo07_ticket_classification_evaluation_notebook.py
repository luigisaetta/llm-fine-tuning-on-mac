"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Structure tests for the Demo 07 held-out test evaluation notebook.
"""

from notebook_structure import assert_didactic_structure, load_notebook


def test_demo07_is_a_didactic_evaluation_notebook() -> None:
    """Demo 07 uses only the test split and compares the adapter with the base model given the same prompt."""
    source = assert_didactic_structure(load_notebook("ticket-classification/demo07_ticket_classification_test_evaluation.ipynb"), "ADAPTER_PROMPT_VARIANT =")
    assert "load_training_splits" not in source and "train_dataset" not in source and "validation_dataset" not in source
    for expected in (
        'ADAPTER_PROMPT_VARIANT = "none"', "TEST_GENERATION_BATCH_SIZE = 4", "TEST_RECORD_INDEX", "load_test_split(paths)",
        'list(dict.fromkeys(["none", ADAPTER_PROMPT_VARIANT]))', "verify_adapter_prompt(adapter_directory, ADAPTER_PROMPT_VARIANT",
        "release_model_memory()", "load_adapter_model(", "Delta (fine-tuned - base, same prompt)", "delta=(adapter_column, same_prompt_column)",
        "show_prediction(",
    ):
        assert expected in source, expected
