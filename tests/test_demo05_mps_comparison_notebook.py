"""
Author: L. Saetta
Date last modified: 2026-09-06
License: MIT
Description: Static tests for the MPS base-model and LoRA-adapter comparison notebook.
"""

import json
from pathlib import Path


def test_mps_comparison_notebook_is_restartable_and_direct() -> None:
    """Verify the comparison notebook has its required MPS and metric contract."""
    repository_root = Path(__file__).resolve().parents[1]
    notebook_path = repository_root / "demo05" / "demo05_mps_base_adapter_comparison.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "VALIDATION_FILE" in source
    assert "torch.backends.mps.is_built()" in source
    assert "torch.backends.mps.is_available()" in source
    assert "torch.device('mps')" in source
    assert "CPU fallback is intentionally disabled" in source
    assert "MODEL_DTYPE = torch.bfloat16" in source
    assert "def token_f1" in source
    assert "TOKEN_F1_ACCURACY_THRESHOLD = 0.80" in source
    assert "base_metrics = evaluate_model" in source
    assert "fine_tuned_metrics = evaluate_model" in source
    assert "Delta (fine-tuned - base)" in source
    assert "torch.mps.empty_cache()" in source
