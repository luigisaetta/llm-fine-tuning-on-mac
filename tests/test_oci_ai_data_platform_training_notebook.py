"""
Author: L. Saetta
Date last modified: 2026-09-15
License: MIT
Description: Static checks for OCI AI Data Platform training and storage notebooks.
"""

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    REPOSITORY_ROOT / "ticket-classification" / "ai-dp" / "nb_fine_tuning_lora.ipynb"
)
PREFLIGHT_CELL_ID = "oci-python-headers-preflight"
STORAGE_NOTEBOOK_PATH = (
    REPOSITORY_ROOT
    / "ticket-classification"
    / "ai-dp"
    / "inspect_local_training_storage.ipynb"
)


def load_notebook() -> dict:
    """Load the OCI AI Data Platform training notebook as nbformat JSON.

    Returns:
        Parsed notebook document.
    """
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_header_preflight_precedes_training_cell() -> None:
    """Require header include paths to be configured before training starts."""
    notebook = load_notebook()
    cells = notebook["cells"]
    preflight_index = next(
        index
        for index, cell in enumerate(cells)
        if cell.get("id") == PREFLIGHT_CELL_ID
    )

    preflight_cell = cells[preflight_index]
    training_cell = cells[preflight_index + 1]
    preflight_source = "".join(preflight_cell["source"])
    training_source = "".join(training_cell["source"])

    assert notebook["nbformat"] == 4
    assert preflight_cell["cell_type"] == "code"
    assert preflight_cell["execution_count"] is None
    assert preflight_cell["outputs"] == []
    assert "trainer.train()" in training_source
    assert "Python.h" in preflight_source
    assert "C_INCLUDE_PATH" in preflight_source
    assert "CPATH" in preflight_source


def test_header_preflight_preserves_existing_include_paths() -> None:
    """Require the preflight cell to retain any compiler paths already configured."""
    notebook = load_notebook()
    preflight_source = "".join(
        next(
            cell["source"]
            for cell in notebook["cells"]
            if cell.get("id") == PREFLIGHT_CELL_ID
        )
    )

    assert "existing_c_include_path = os.environ.get" in preflight_source
    assert "existing_cpath = os.environ.get" in preflight_source
    assert "if existing_c_include_path" in preflight_source
    assert "if existing_cpath else" in preflight_source


def test_training_uses_local_checkpoints_and_copies_only_final_adapter() -> None:
    """Require checkpoints to stay local until final adapter export completes."""
    notebook = load_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    final_source = "".join(notebook["cells"][-1]["source"])

    assert "LOCAL_TRAINING_OUTPUT_DIR = Path('/tmp')" in source
    assert "SFTConfig(output_dir=str(LOCAL_TRAINING_OUTPUT_DIR)" in source
    assert "trainer.save_model(LOCAL_ADAPTER_OUTPUT_DIR)" in source
    assert "tokenizer.save_pretrained(LOCAL_ADAPTER_OUTPUT_DIR)" in source
    assert "shutil.copytree(LOCAL_ADAPTER_OUTPUT_DIR, REMOTE_ADAPTER_OUTPUT_DIR" in final_source
    assert "copy_errors" in final_source
    assert "best_model_checkpoint" not in final_source


def test_training_notebook_is_output_free() -> None:
    """Require committed OCI training notebooks to be restartable without prior output."""
    notebook = load_notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)


def test_storage_inspection_notebook_is_non_training_and_restartable() -> None:
    """Require storage inspection to be output-free and limited to diagnostics."""
    notebook = json.loads(STORAGE_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert notebook["nbformat"] == 4
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    assert "CANDIDATE_DIRECTORIES" in source
    assert "os.statvfs" in source
    assert "NamedTemporaryFile" in source
    assert "trainer.train()" not in source
