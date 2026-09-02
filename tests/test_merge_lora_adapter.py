"""
Author: L. Saetta
Date last modified: 2026-09-02
License: MIT
Description: Unit tests for local validation helpers in the LoRA merge script.
"""

import argparse
import json
from pathlib import Path

import pytest

import scripts.merge_lora_adapter as merge_script
from scripts.merge_lora_adapter import (
    ensure_empty_output_directory,
    upload_model,
    validate_adapter_directory,
    validate_model_directory,
    write_model_card,
)


def write_model_metadata(directory: Path) -> None:
    """Create the minimal non-weight files expected for a local model fixture."""
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_validate_model_directory_accepts_complete_sharded_weights(tmp_path: Path) -> None:
    """A model index with all referenced shard files is accepted."""
    write_model_metadata(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer_a": "model-00001-of-00002.safetensors", "layer_b": "model-00002-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00002.safetensors").touch()
    (tmp_path / "model-00002-of-00002.safetensors").touch()

    validate_model_directory(tmp_path)


def test_validate_model_directory_reports_missing_shard(tmp_path: Path) -> None:
    """A missing indexed shard produces an actionable error."""
    write_model_metadata(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer_a": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Missing model weight shards"):
        validate_model_directory(tmp_path)


def test_validate_adapter_directory_reports_missing_weights(tmp_path: Path) -> None:
    """An adapter config without weights is rejected."""
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing adapter weights"):
        validate_adapter_directory(tmp_path)


def test_ensure_empty_output_directory_refuses_existing_files(tmp_path: Path) -> None:
    """An existing output directory with a file is never overwritten."""
    (tmp_path / "existing-file.txt").touch()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ensure_empty_output_directory(tmp_path)


def test_upload_model_requires_environment_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Hub upload fails locally before any network request without a token."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN is required"):
        upload_model(tmp_path, "owner/model", private=True)


def test_upload_existing_output_skips_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The upload-only retry validates existing output without loading source artifacts."""
    write_model_metadata(tmp_path)
    (tmp_path / "model.safetensors").touch()
    arguments = argparse.Namespace(
        base_model_dir=Path("unused-base"),
        adapter_dir=Path("unused-adapter"),
        output_dir=tmp_path,
        device="cpu",
        push_to_hub=True,
        upload_existing_output=True,
        repo_id="owner/model",
        private=True,
    )
    monkeypatch.setattr(merge_script, "parse_arguments", lambda: arguments)
    monkeypatch.setattr(merge_script, "upload_model", lambda *_args, **_kwargs: "https://huggingface.co/owner/model")
    monkeypatch.setattr(
        merge_script,
        "merge_model",
        lambda *_args, **_kwargs: pytest.fail("merge_model must not run during an upload-only retry"),
    )

    merge_script.main()


def test_model_card_declares_base_model_without_local_paths(tmp_path: Path) -> None:
    """The generated Hub card records provenance without disclosing local paths."""
    write_model_card(tmp_path, merge_script.torch.device("cpu"))

    model_card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "base_model: Qwen/Qwen3-1.7B" in model_card
    assert "library_name: transformers" in model_card
    assert "Base model source | Local project artifact (path intentionally omitted)" in model_card
    assert str(tmp_path) not in model_card
