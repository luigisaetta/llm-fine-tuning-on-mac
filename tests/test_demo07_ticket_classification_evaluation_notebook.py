"""
Author: L. Saetta
Date last modified: 2026-10-25
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
    assert "base_model_metrics" in source
    assert "fine_tuned_metrics" in source
    assert "PeftModel.from_pretrained" in source
    assert "model.generate" in source
    assert "category_accuracy" in source
    assert "severity_accuracy" in source
    assert "joint_accuracy" in source
    assert "valid_json_rate" in source
    assert "use_cache=True" in source
    assert "from tqdm.auto import tqdm" in source
    assert "progress_bar.update" in source
    assert "TEST_RECORD_INDEX" in source
    assert "ticket_text" in source
    assert "Fine-tuned response" in source
    assert "Expected response" in source
    assert "ADAPTER_PROMPT_VARIANT = 'short'" in source
    assert "system_prompt_short.txt" in source
    assert "BASELINE_PROMPT_VARIANTS = list(dict.fromkeys(['none', ADAPTER_PROMPT_VARIANT]))" in source
    assert "build_prompt_messages(record['messages'][0], system_prompt)" in source
    assert "build_prompt_messages(record['messages'][0], SYSTEM_PROMPTS[ADAPTER_PROMPT_VARIANT])" in source
    assert "[record['messages'][0]]" not in source
    assert "verify_adapter_prompt_variant(ADAPTER_DIRECTORY, ADAPTER_PROMPT_VARIANT" in source
    assert "Delta (fine-tuned - base, same prompt)" in source
