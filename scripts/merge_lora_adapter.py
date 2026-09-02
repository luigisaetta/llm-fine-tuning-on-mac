"""
Author: L. Saetta
Date last modified: 2026-09-02
License: MIT
Description: Merge a local Qwen3 base model with a LoRA adapter and optionally publish the standalone model to the Hugging Face Hub.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_project_root(start_directory: Path) -> Path:
    """Return the repository root containing the project instructions.

    Args:
        start_directory: Directory from which to begin the upward search.

    Returns:
        The repository root.

    Raises:
        RuntimeError: If no project root can be found.
    """
    for candidate in (start_directory, *start_directory.parents):
        if (candidate / "AGENTS.md").is_file() and (candidate / "requirements.txt").is_file():
            return candidate
    raise RuntimeError("Could not find the project root. Run the script from inside the repository.")


def validate_model_directory(model_directory: Path) -> None:
    """Validate a local Transformers model directory and its Safetensors weights.

    Args:
        model_directory: Directory containing the base model files.

    Raises:
        FileNotFoundError: If required configuration, tokenizer, index, or weight files are absent.
        KeyError: If a Safetensors index does not contain a weight map.
    """
    required_files = ("config.json", "tokenizer.json")
    missing_files = [name for name in required_files if not (model_directory / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing base-model files in {model_directory}: {missing_files}")

    single_weight_file = model_directory / "model.safetensors"
    weight_index_file = model_directory / "model.safetensors.index.json"
    if single_weight_file.is_file():
        return
    if not weight_index_file.is_file():
        raise FileNotFoundError(
            f"Missing model weights in {model_directory}. Expected model.safetensors or model.safetensors.index.json."
        )

    weight_map = json.loads(weight_index_file.read_text(encoding="utf-8"))["weight_map"]
    missing_shards = sorted(
        shard_name for shard_name in set(weight_map.values()) if not (model_directory / shard_name).is_file()
    )
    if missing_shards:
        raise FileNotFoundError(f"Missing model weight shards in {model_directory}: {missing_shards}")


def validate_adapter_directory(adapter_directory: Path) -> None:
    """Validate the configuration and weights of a local PEFT adapter.

    Args:
        adapter_directory: Directory containing the saved LoRA adapter.

    Raises:
        FileNotFoundError: If adapter configuration or weights are absent.
    """
    if not (adapter_directory / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter configuration in {adapter_directory}.")
    if not any((adapter_directory / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError(f"Missing adapter weights in {adapter_directory}.")


def select_device(requested_device: str) -> torch.device:
    """Select an explicitly requested merge device without implicit fallback.

    Args:
        requested_device: Either ``cpu`` or ``mps``.

    Returns:
        The selected PyTorch device.

    Raises:
        RuntimeError: If MPS is requested but unavailable.
    """
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "mps" and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("MPS was requested but is unavailable in this Python environment.")


def ensure_empty_output_directory(output_directory: Path) -> None:
    """Create an output directory only if it is absent or empty.

    Args:
        output_directory: Proposed destination for the standalone model.

    Raises:
        FileExistsError: If the destination already contains files.
    """
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)


def write_model_card(output_directory: Path, device: torch.device) -> None:
    """Write provenance metadata for a locally merged standalone model.

    Args:
        output_directory: Directory containing the merged model.
        device: Device used for the merge operation.
    """
    model_card = f"""---
base_model: Qwen/Qwen3-1.7B
library_name: transformers
pipeline_tag: text-generation
tags:
  - qwen3
  - lora
  - merged
  - text-generation
language:
  - en
---

# Locally merged Qwen3-1.7B LoRA model

This standalone Transformers model was created by merging a local LoRA adapter into `Qwen/Qwen3-1.7B`.

## Provenance

| Field | Value |
| --- | --- |
| Base model identifier | `Qwen/Qwen3-1.7B` |
| Base model source | Local project artifact (path intentionally omitted) |
| Base model revision | Not pinned |
| LoRA adapter source | Local Demo 01 artifact (path intentionally omitted) |
| Merge device | `{device.type}` |
| Serialization | Safetensors |

The source datasets are private and are not included in this repository or this model directory. Review the base model licence and the suitability of the fine-tuned behaviour before redistribution.

## Limitations and responsible use

This model may produce unsupported or incorrect information. Do not use it as a source of verified facts or for high-stakes decisions.
"""
    (output_directory / "README.md").write_text(model_card, encoding="utf-8")


def merge_model(base_model_directory: Path, adapter_directory: Path, output_directory: Path, device: torch.device) -> None:
    """Merge a local LoRA adapter into a local base model and save it.

    Args:
        base_model_directory: Directory containing the local base model.
        adapter_directory: Directory containing the saved LoRA adapter.
        output_directory: Empty destination directory for merged artifacts.
        device: Device used to load and merge model weights.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_directory, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_directory,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    if base_model.config.model_type != "qwen3":
        raise RuntimeError(f"Expected a Qwen3 base model, got {base_model.config.model_type!r}.")

    peft_model = PeftModel.from_pretrained(base_model, adapter_directory, local_files_only=True)
    merged_model = peft_model.merge_and_unload(progressbar=True, safe_merge=True).to("cpu")
    merged_model.save_pretrained(output_directory, safe_serialization=True)
    tokenizer.save_pretrained(output_directory)
    write_model_card(output_directory, device)


def upload_model(output_directory: Path, repository_id: str, private: bool) -> str:
    """Create or update a Hub model repository from a merged-model directory.

    Args:
        output_directory: Directory containing the merged standalone model.
        repository_id: Target Hub repository in ``owner/name`` form.
        private: Whether a newly created repository should be private.

    Returns:
        URL of the created or updated repository.

    Raises:
        RuntimeError: If the HF_TOKEN environment variable is not set.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for --push-to-hub. Set it in your environment and retry.")

    api = HfApi(token=token)
    repository_url = api.create_repo(repository_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=output_directory,
        repo_id=repository_id,
        repo_type="model",
        commit_message="Upload merged Qwen3-1.7B LoRA model",
    )
    return str(repository_url)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for merging and optional Hub upload."""
    project_root = find_project_root(Path.cwd().resolve())
    parser = argparse.ArgumentParser(description="Merge a local Qwen3 LoRA adapter into its base model.")
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=project_root / "artifacts" / "models" / "Qwen3-1.7B",
        help="Local Qwen3 base-model directory.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=project_root / "artifacts" / "training" / "demo01-qwen3-1.7b-lora" / "adapter",
        help="Local PEFT LoRA adapter directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "artifacts" / "merged" / "demo01-qwen3-1.7b-merged",
        help="New or empty local directory for the standalone merged model.",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="Device used for merging.")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the saved standalone model to the Hub.")
    parser.add_argument(
        "--upload-existing-output",
        action="store_true",
        help="Upload an existing complete output directory without performing a merge.",
    )
    parser.add_argument("--repo-id", help="Target Hub model repository in OWNER/NAME form; required with --push-to-hub.")
    parser.add_argument("--private", action="store_true", help="Create the Hub repository as private if it does not exist.")
    return parser.parse_args()


def main() -> None:
    """Run local validation, merge the model, and optionally upload it to the Hub."""
    args = parse_arguments()
    if args.push_to_hub and not args.repo_id:
        raise ValueError("--repo-id OWNER/NAME is required with --push-to-hub.")
    if args.upload_existing_output and not args.push_to_hub:
        raise ValueError("--upload-existing-output requires --push-to-hub.")

    base_model_directory = args.base_model_dir.resolve()
    adapter_directory = args.adapter_dir.resolve()
    output_directory = args.output_dir.resolve()
    if args.upload_existing_output:
        validate_model_directory(output_directory)
        repository_url = upload_model(output_directory, args.repo_id, args.private)
        print(f"Existing merged model uploaded to: {repository_url}")
        return

    validate_model_directory(base_model_directory)
    validate_adapter_directory(adapter_directory)
    device = select_device(args.device)
    ensure_empty_output_directory(output_directory)

    print(f"Base model: {base_model_directory}")
    print(f"LoRA adapter: {adapter_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Merge device: {device}")
    merge_model(base_model_directory, adapter_directory, output_directory, device)
    print(f"Merged model saved to: {output_directory}")

    if args.push_to_hub:
        repository_url = upload_model(output_directory, args.repo_id, args.private)
        print(f"Merged model uploaded to: {repository_url}")


if __name__ == "__main__":
    main()
