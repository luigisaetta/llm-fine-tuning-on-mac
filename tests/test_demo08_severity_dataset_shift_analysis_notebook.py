"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Structure tests for the Demo 08 deterministic severity dataset-shift analysis notebook.
"""

from notebook_structure import assert_didactic_structure, load_notebook


def test_demo08_is_a_didactic_model_free_analysis() -> None:
    """Demo 08 compares validation V2 and test data with deterministic package functions and loads no model."""
    source = assert_didactic_structure(load_notebook("ticket-classification/demo08_severity_dataset_shift_analysis.ipynb"), "TOP_TERMS =")
    assert "paths.validation_file" in source and "paths.test_file" in source and "paths.train_file" not in source
    assert "llm_fine_tuning.models" not in source and "import random" not in source
    for expected in ("total_variation_distance(", "conditional_severity_shift(", "severity_term_overlap(", "diagnostic_rows(", "plot_conditional_shift("):
        assert expected in source, expected
