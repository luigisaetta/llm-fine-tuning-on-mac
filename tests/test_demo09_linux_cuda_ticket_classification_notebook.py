"""
Author: L. Saetta
Date last modified: 2026-09-13
License: MIT
Description: Static tests for the Linux CUDA ticket-classification training notebook.
"""

import json
from pathlib import Path


def test_linux_cuda_ticket_notebook_requires_gpu_and_exposes_artifact_paths() -> None:
    """Verify the CUDA-only, BF16, local-artifact notebook contract."""
    root = Path(__file__).resolve().parents[1]
    path = root / "ticket-classification" / "demo09_ticket_classification_lora_fine_tuning_linux_cuda.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] and cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "torch.cuda.is_available()" in source
    assert "torch.cuda.is_bf16_supported()" in source
    assert "cuda:0" in source
    assert "GPU count" in source
    assert "MODEL_DIRECTORY" in source
    assert "DATASET_DIRECTORY" in source
    assert "TRAINING_OUTPUT_DIRECTORY" in source
    assert "train_dataset_v2.jsonl" in source
    assert "validation_dataset_v2.jsonl" in source
    assert "test_dataset.jsonl" not in source
    assert "autocast_adapter_dtype=False" in source
    assert "bf16=True" in source
    assert "bf16_full_eval=True" in source
    assert "metric_for_best_model='eval_joint_accuracy'" in source
    assert "use_cache=True" in source
    assert "mps" not in source.lower()
