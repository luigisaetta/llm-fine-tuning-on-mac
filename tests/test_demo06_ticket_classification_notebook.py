"""
Author: L. Saetta
Date last modified: 2026-09-08
License: MIT
Description: Static tests for the ticket-classification LoRA training notebook.
"""

import json
from pathlib import Path


def test_ticket_classification_notebook_uses_only_training_and_validation() -> None:
    """Verify the notebook's MPS LoRA training and validation-loss contract."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "ticket-classification" / "demo06_ticket_classification_lora_fine_tuning.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] and cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "train_dataset_v2.jsonl" in source
    assert "validation_dataset_v2.jsonl" in source
    assert "test_dataset.jsonl" not in source
    assert "eval_strategy='epoch'" in source
    assert "save_strategy='epoch'" in source
    assert "torch.bfloat16" in source
    assert "autocast_adapter_dtype=False" in source
    assert "bf16=True" in source
    assert "bf16_full_eval=True" in source
    assert "ADAPTER_OUTPUT_DIRECTORY" in source
    assert "artifacts' / 'models'" in source
    assert "Training and Validation Loss by Epoch" in source
    assert "ValidationGenerationMetricsCallback" in source
    assert "model.generate" in source
    assert "category_accuracy" in source
    assert "severity_accuracy" in source
    assert "joint_accuracy" in source
    assert "valid_json_rate" in source
    assert "Category+Severity Accuracy" in source
    assert "display_epoch_metrics(epoch_metrics)" in source
    assert "VALIDATION_GENERATION_BATCH_SIZE = 4" in source
    assert "padding=True" in source
    assert "tokenizer.padding_side = 'left'" in source
    assert "use_cache=True" in source
    assert "metric_for_best_model='eval_joint_accuracy'" in source
    assert "greater_is_better=True" in source
    assert "evaluation_metrics['eval_joint_accuracy']" in source
    assert all(
        index > 0 and notebook["cells"][index - 1]["cell_type"] == "markdown"
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    )
    import_cells = [
        index
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code" and any(line.startswith(("import ", "from ")) for line in cell.get("source", []))
    ]
    assert import_cells == [1]
