"""
Author: L. Saetta
Date last modified: 2026-09-08
License: MIT
Description: Static tests for the ticket-classification held-out test evaluation notebook.
"""

import json
from pathlib import Path


def test_held_out_evaluation_notebook_uses_only_test_data() -> None:
    """Verify test-only PEFT evaluation, metrics, and restartable notebook structure."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "ticket-classification" / "demo07_ticket_classification_test_evaluation.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] and cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "test_dataset.jsonl" in source
    assert "train_dataset" not in source
    assert "validation_dataset" not in source
    assert "PeftModel.from_pretrained" in source
    assert "model.generate" in source
    assert "category_accuracy" in source
    assert "severity_accuracy" in source
    assert "joint_accuracy" in source
    assert "valid_json_rate" in source
    assert "use_cache=True" in source
