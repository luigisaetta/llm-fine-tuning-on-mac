# Apple Silicon MPS Execution Specification

## Problem

The project targets local fine-tuning on a MacBook. Training and inference should use the Apple GPU through PyTorch's Metal Performance Shaders (MPS) backend when it is available, while producing clear and safe behaviour on unsupported hardware or unavailable backends.

## Scope

This specification defines the device-selection contract for future Python scripts and notebooks. The implementation will use the PyTorch MPS backend; Transformers, PEFT, TRL, and Datasets run on top of that PyTorch execution layer.

It covers:

* detecting whether the installed PyTorch build includes MPS and whether MPS is currently usable;
* selecting MPS for `auto` device selection on a supported Apple Silicon Mac;
* explicit CPU fallback and explicit error handling for a requested but unavailable device;
* documenting MPS-specific constraints in each training-run specification.

## Out of scope

CUDA, distributed training, eGPUs, MLX-based training, vendor-specific quantization backends, and automatic silent operation fallbacks are out of scope. In particular, this project must not depend on `bitsandbytes` for the initial Mac training path.

## Assumptions

* The `llm-fine-tuning-on-mac` Conda environment provides a current macOS-compatible PyTorch release from `requirements.txt`.
* The host is running macOS on compatible Apple hardware. MPS may nevertheless be unavailable because of the PyTorch build, operating-system support, or runtime conditions.
* Model weights and LoRA adapter training use a PyTorch-supported dtype selected by the later training-run specification. The first MPS baseline should prefer a conservative, verified dtype rather than claiming universal mixed-precision support.

## Device-selection contract

Future reusable code must expose `auto`, `mps`, and `cpu` device choices.

| Requested device | MPS available | Required outcome |
| --- | --- | --- |
| `auto` | Yes | Select `mps` and report the choice. |
| `auto` | No | Select `cpu` and emit a clear performance warning. |
| `mps` | Yes | Select `mps` and report the choice. |
| `mps` | No | Stop before model download or training with an actionable error. |
| `cpu` | Either | Select `cpu` and report the choice. |

Availability must be determined by both `torch.backends.mps.is_built()` and `torch.backends.mps.is_available()`. `is_built()` distinguishes a PyTorch build without MPS support; `is_available()` distinguishes a supported build that cannot use MPS in the active runtime.

The project must not silently set `PYTORCH_ENABLE_MPS_FALLBACK=1`. If a later experiment needs this PyTorch option for an unsupported operation, it must be an opt-in, documented configuration because affected operations may execute on CPU and materially alter performance.

## Training constraints

* All tensors passed to the model and loss calculation must use the selected device consistently.
* Scripts and notebooks must print the selected device, PyTorch version, MPS build/availability status, and effective precision before training begins.
* A training-run specification must document the model size, maximum sequence length, micro-batch size, gradient accumulation, dtype, expected unified-memory pressure, and fallback policy.
* If a MPS operation fails, the error must identify the operation and recommend an explicit remediation: upgrade PyTorch/macOS, reduce the workload, alter the training configuration, or deliberately use CPU/fallback mode.
* The project must never state that all Transformer or LoRA operations are supported or performant on MPS without a successful documented integration run.

## Acceptance criteria

* A reusable device helper implements the device-selection contract and has unit tests covering every table row without loading a model.
* The first training notebook displays its selected device and execution metadata before model loading.
* Requesting `mps` on an unavailable backend fails with an actionable message; it does not silently move to CPU.
* `auto` clearly reports CPU fallback when MPS cannot be used.
* The first real fine-tuning integration run records the selected PyTorch version, macOS/hardware context, model identifier, configuration, and any MPS limitations encountered.

## Verification

* Unit tests mock `torch.backends.mps.is_built()` and `torch.backends.mps.is_available()` for all device-selection outcomes.
* A local integration check prints the PyTorch version, MPS build state, MPS availability, and selected device without downloading a model.
* The first optional end-to-end run performs a small forward/backward LoRA training step on MPS and records its result in the corresponding training-run documentation.

## References

* [PyTorch MPS backend notes](https://docs.pytorch.org/docs/stable/notes/mps.html)
* [PyTorch MPS backend API](https://docs.pytorch.org/docs/stable/backends.html#torch.backends.mps.is_available)
