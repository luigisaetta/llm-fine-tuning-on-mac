# Demo 04: Linux CUDA LoRA Fine-Tuning of Qwen3-1.7B

## Problem

The project needs a Linux-specific version of the Demo 01 fine-tuning notebook for a CUDA-capable NVIDIA GPU. The existing Mac notebook must remain unchanged as the Apple Silicon learning path.

## Scope

Create `demo04_linux_cuda_lora/` containing a restartable notebook and usage README. The notebook fine-tunes the local Qwen3-1.7B model with the same private parametric-memory dataset, LoRA configuration, per-epoch loss evaluation, loss chart, and held-out generative evaluation as Demo 01. It targets one CUDA GPU only.

## Assumptions

* The user runs Linux with a CUDA-capable NVIDIA GPU and a CUDA-enabled PyTorch installation that supports BF16.
* The base model and private datasets are local and are not committed to Git.
* The user installs PyTorch and the matching CUDA runtime separately; they are deliberately outside this demo's dependency instructions.

## Functional requirements

* Require `torch.cuda.is_available()` and `torch.cuda.is_bf16_supported()` before model loading; do not provide a CPU fallback.
* Use `torch.bfloat16` for the base model and LoRA parameters, and enable `bf16=True` and `bf16_full_eval=True` in `SFTConfig`.
* Keep PEFT adapter autocasting disabled and fail before training if any floating-point model parameter or trainable LoRA parameter is not BF16.
* Include one clearly marked editable cell for the base-model directory, dataset directory, and training-output directory.
* Use CUDA-relevant diagnostics only: PyTorch CUDA version, CUDA availability, GPU name, GPU count, selected device, and model dtype.
* Preserve the Demo 01 dataset validation, adapter-only output, checkpoint policy, loss plot, and held-out token-F1 evaluation behaviour.

## Out of scope

Multi-GPU, DDP, FSDP, DeepSpeed, CPU fallback, model merging, and OCI deployment are out of scope.

## Acceptance criteria

* The notebook JSON is valid and has no stored outputs.
* The notebook requires CUDA BF16 support.
* The README is in English and lists every non-PyTorch Python package required to run the notebook, without CUDA or PyTorch installation instructions.
* No model, dataset, output artifact, or credential is added to Git.

## Verification

* Validate notebook JSON and inspect that all code-cell outputs are empty.
* Run unit tests without downloading models or datasets.
* On the target Linux host, execute the configuration and model-setup cells and inspect CUDA/BF16 diagnostics before a full training run.
