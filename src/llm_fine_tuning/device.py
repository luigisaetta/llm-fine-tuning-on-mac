"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Select the PyTorch device and check the Apple Silicon MPS requirements for BF16 work.
"""

import platform

import torch

MINIMUM_MACOS_MAJOR_FOR_BF16 = 14


def select_device(requested_device: str = "auto") -> torch.device:
    """Return the requested device, preferring MPS for ``'auto'`` and reporting any CPU fallback.

    Args:
        requested_device: ``'auto'``, ``'mps'``, or ``'cpu'``.

    Returns:
        The selected device.

    Raises:
        ValueError: If the requested device is not supported.
        RuntimeError: If MPS is requested explicitly but unavailable.
    """
    if requested_device not in {"auto", "mps", "cpu"}:
        raise ValueError("The requested device must be 'auto', 'mps', or 'cpu'.")
    if requested_device == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested_device == "mps":
        raise RuntimeError("MPS was requested but is unavailable in this runtime.")
    print("Warning: MPS is unavailable. Falling back to CPU; the workload will be slower.")
    return torch.device("cpu")


def require_bfloat16_mps_device() -> torch.device:
    """Return the MPS device, or fail when BF16 work on Apple Silicon is not supported.

    Returns:
        The MPS device.

    Raises:
        RuntimeError: If MPS is unavailable or macOS is older than version 14.
    """
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("BF16 work requires an available MPS backend on Apple Silicon; CPU fallback is disabled.")
    macos_version = platform.mac_ver()[0]
    macos_major = int(macos_version.split(".")[0]) if macos_version else 0
    if macos_major < MINIMUM_MACOS_MAJOR_FOR_BF16:
        raise RuntimeError(
            f"BF16 on MPS requires macOS {MINIMUM_MACOS_MAJOR_FOR_BF16} or later; "
            f"detected {macos_version or 'an unknown version'}."
        )
    return torch.device("mps")


def describe_runtime(device: torch.device) -> dict[str, str]:
    """Return the runtime facts worth recording with an experiment.

    Args:
        device: The selected device.

    Returns:
        A mapping from description to value, suitable for a settings table.
    """
    return {
        "macOS": platform.mac_ver()[0] or "not macOS",
        "Machine": platform.machine(),
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "MPS available": str(torch.backends.mps.is_available()),
        "Selected device": str(device),
    }
