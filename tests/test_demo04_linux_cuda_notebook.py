"""
Author: L. Saetta
Date last modified: 2026-09-04
License: MIT
Description: Static tests for the Linux CUDA LoRA fine-tuning notebook contract.
"""

import json
from pathlib import Path


def test_linux_cuda_notebook_requires_cuda_bf16_and_has_no_outputs() -> None:
    """Verify the Linux notebook is CUDA/BF16 specific and restartable."""
    repository_root = Path(__file__).resolve().parents[1]
    notebook_path = repository_root / "demo04_linux_cuda_lora" / "demo04_lora_fine_tuning_linux_cuda.ipynb"

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "torch.cuda.is_available()" in source
    assert "torch.cuda.is_bf16_supported()" in source
    assert "bf16=True" in source
    assert "bf16_full_eval=True" in source
    assert "MPS" not in source
    assert "mps" not in source
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
