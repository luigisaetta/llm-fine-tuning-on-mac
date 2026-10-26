"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Locate the repository root and validate local Hugging Face model directories.
"""

import json
from pathlib import Path


def find_project_root(start_directory: Path | None = None) -> Path:
    """Return the repository root, identified by its AGENTS.md and requirements.txt files.

    Args:
        start_directory: Directory from which to search upwards. Defaults to the current working directory.

    Returns:
        The repository root directory.

    Raises:
        RuntimeError: If no parent directory contains both files.
    """
    start = (start_directory or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "AGENTS.md").is_file() and (candidate / "requirements.txt").is_file():
            return candidate
    raise RuntimeError("Could not find the project root. Start JupyterLab from inside the repository.")


def require_files(*paths: Path) -> None:
    """Raise an error naming the first required file that does not exist.

    Args:
        *paths: Files that must exist.

    Raises:
        FileNotFoundError: If one of the files is missing.
    """
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required local artifact: {path}")


def validate_model_directory(model_directory: Path) -> None:
    """Check that a local model directory has its configuration, tokenizer, and complete Safetensors weights.

    Args:
        model_directory: Directory created by ``hf download``.

    Raises:
        FileNotFoundError: If the configuration, tokenizer, weights, or a referenced weight shard is missing.
    """
    require_files(model_directory / "config.json", model_directory / "tokenizer.json")
    single_weight_file = model_directory / "model.safetensors"
    weight_index_file = model_directory / "model.safetensors.index.json"
    if not single_weight_file.is_file() and not weight_index_file.is_file():
        raise FileNotFoundError(
            f"Missing model weights in {model_directory}. Expected model.safetensors or model.safetensors.index.json."
        )
    if weight_index_file.is_file():
        shard_names = set(json.loads(weight_index_file.read_text(encoding="utf-8"))["weight_map"].values())
        missing_shards = sorted(name for name in shard_names if not (model_directory / name).is_file())
        if missing_shards:
            raise FileNotFoundError(f"Missing model weight shards in {model_directory}: {missing_shards}")


def require_adapter_directory(adapter_directory: Path) -> None:
    """Check that a directory contains a saved PEFT adapter configuration and weights.

    Args:
        adapter_directory: Directory written by ``trainer.save_model``.

    Raises:
        FileNotFoundError: If the adapter configuration or weights are missing.
    """
    require_files(adapter_directory / "adapter_config.json")
    if not any((adapter_directory / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError(f"Missing adapter weights in {adapter_directory}. Run the training notebook first.")
