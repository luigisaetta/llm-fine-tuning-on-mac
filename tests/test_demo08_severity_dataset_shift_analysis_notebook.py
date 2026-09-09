"""
Author: L. Saetta
Date last modified: 2026-09-09
License: MIT
Description: Static tests for the ticket-classification severity dataset-shift analysis notebook.
"""

import json
from pathlib import Path


def test_severity_dataset_shift_analysis_notebook_is_restartable() -> None:
    """Verify deterministic validation-versus-test analysis and clean notebook outputs."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "ticket-classification" / "demo08_severity_dataset_shift_analysis.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] and cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "validation_dataset_v2.jsonl" in source
    assert "test_dataset.jsonl" in source
    assert "SEVERITIES = ('P1', 'P2', 'P3', 'P4')" in source
    assert "Severity-distribution TVD" in source
    assert "P(severity | category)" in source
    assert "Top-term Jaccard overlap" in source
    assert "matplotlib.pyplot" in source
    assert "import random" not in source
