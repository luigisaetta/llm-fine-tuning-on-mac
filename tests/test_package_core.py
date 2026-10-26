"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Unit tests for the shared path, device, and display helpers of the llm_fine_tuning package.
"""

import json

import pytest
import torch

from llm_fine_tuning.device import describe_runtime, select_device
from llm_fine_tuning.display import markdown_table, show_settings
from llm_fine_tuning.paths import find_project_root, require_adapter_directory, validate_model_directory


def test_find_project_root_searches_upwards(tmp_path) -> None:
    """The root is the first parent with both AGENTS.md and requirements.txt."""
    (tmp_path / "AGENTS.md").write_text("", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == tmp_path.resolve()


def test_find_project_root_fails_outside_repository(tmp_path) -> None:
    """A directory tree without the marker files raises an actionable error."""
    with pytest.raises(RuntimeError, match="project root"):
        find_project_root(tmp_path)


def test_validate_model_directory_requires_every_shard(tmp_path) -> None:
    """A sharded model is complete only when every shard named in the index exists."""
    for name in ("config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    weight_map = {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="model-00002-of-00002"):
        validate_model_directory(tmp_path)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"")
    validate_model_directory(tmp_path)


def test_validate_model_directory_requires_weights(tmp_path) -> None:
    """Configuration and tokenizer alone are not a usable model."""
    for name in ("config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Missing model weights"):
        validate_model_directory(tmp_path)


def test_require_adapter_directory(tmp_path) -> None:
    """An adapter needs both its configuration and its weights."""
    with pytest.raises(FileNotFoundError):
        require_adapter_directory(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="adapter weights"):
        require_adapter_directory(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    require_adapter_directory(tmp_path)


def test_select_device_validates_requests() -> None:
    """CPU is always selectable and unknown device names are rejected."""
    assert select_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError):
        select_device("cuda")
    assert describe_runtime(torch.device("cpu"))["Selected device"] == "cpu"


def test_markdown_table_alignment() -> None:
    """The first column is left-aligned and the others right-aligned."""
    table = markdown_table(["Name", "Value"], [["a", 1]])
    assert table.splitlines() == ["| Name | Value |", "| --- | ---: |", "| a | 1 |"]


def test_show_settings_requires_dataclass() -> None:
    """Settings tables are built only from dataclasses."""
    with pytest.raises(TypeError):
        show_settings({"a": 1})
