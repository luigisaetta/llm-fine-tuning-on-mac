"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Load the local tokenizer, base model, and LoRA adapter, check parameter dtypes, and release memory.
"""

import gc
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_DTYPE = torch.bfloat16


def load_tokenizer(model_directory: Path):
    """Load the tokenizer from a local model directory, using the EOS token for padding when needed.

    Args:
        model_directory: Local Hugging Face model directory.

    Returns:
        The tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_directory: Path,
    device: torch.device,
    dtype: torch.dtype = DEFAULT_DTYPE,
    expected_model_type: str = "qwen3",
):
    """Load a local causal language model on a device and check its architecture.

    Args:
        model_directory: Local Hugging Face model directory.
        device: Device on which to place the model.
        dtype: Parameter dtype.
        expected_model_type: Required ``config.model_type``.

    Returns:
        The loaded model.

    Raises:
        RuntimeError: If the model has an unexpected type, device, or dtype.
    """
    model = AutoModelForCausalLM.from_pretrained(model_directory, dtype=dtype, local_files_only=True).to(device)
    if model.config.model_type != expected_model_type:
        raise RuntimeError(f"Expected a {expected_model_type} base model, got {model.config.model_type!r}.")
    require_model_placement(model, device, dtype)
    return model


def load_adapter_model(base_model, adapter_directory: Path, device: torch.device, dtype: torch.dtype = DEFAULT_DTYPE):
    """Attach a saved LoRA adapter to a base model for inference.

    Args:
        base_model: Model returned by :func:`load_base_model`.
        adapter_directory: Directory containing the saved adapter.
        device: Device on which the model runs.
        dtype: Expected parameter dtype.

    Returns:
        The PEFT model in evaluation mode.

    Raises:
        RuntimeError: If the adapter model is not on the expected device or dtype.
    """
    model = PeftModel.from_pretrained(
        base_model, adapter_directory, local_files_only=True, autocast_adapter_dtype=False
    ).to(device)
    require_model_placement(model, device, dtype)
    model.eval()
    return model


def require_model_placement(model, device: torch.device, dtype: torch.dtype) -> None:
    """Raise an error unless the model's first parameter is on the device type and dtype expected.

    Args:
        model: Model to check.
        device: Expected device.
        dtype: Expected dtype.

    Raises:
        RuntimeError: If the device type or dtype differs.
    """
    first_parameter = next(model.parameters())
    if first_parameter.device.type != device.type or first_parameter.dtype != dtype:
        raise RuntimeError(f"The model was not loaded on {device.type} in {dtype}.")


def require_parameter_dtypes(model, dtype: torch.dtype) -> set[torch.dtype]:
    """Check that every floating-point and every trainable parameter uses one dtype.

    Args:
        model: Model to check, typically a PEFT model before training.
        dtype: Required dtype.

    Returns:
        The dtypes of the trainable parameters.

    Raises:
        RuntimeError: If any floating-point or trainable parameter has another dtype.
    """
    floating_dtypes = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
    trainable_dtypes = {parameter.dtype for parameter in model.parameters() if parameter.requires_grad}
    if floating_dtypes != {dtype} or trainable_dtypes != {dtype}:
        raise RuntimeError(f"Expected {dtype} floating and trainable parameters, got {floating_dtypes} and {trainable_dtypes}.")
    return trainable_dtypes


def release_model_memory() -> None:
    """Run garbage collection and return cached MPS memory after a model has been deleted."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
