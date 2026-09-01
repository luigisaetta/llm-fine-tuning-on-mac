# Fine-tune a Qwen 3 model on your Mac

Turn a compact open-weight language model into a model that understands *your* task—without leaving your MacBook.

This repository is a hands-on, spec-driven project for fine-tuning a small model from the [Qwen 3 family](https://huggingface.co/Qwen) with **LoRA** (Low-Rank Adaptation). It combines reproducible Python scripts with explorable Jupyter notebooks, so you can both understand each step and run the whole workflow again.

> The aim is not merely to launch training. It is to make the decisions behind a reliable local fine-tuning workflow visible: data format, prompt template, memory limits, LoRA configuration, evaluation, and adapter-based inference.

## What you will build

The project will guide you through a small, practical workflow:

1. Download the selected Qwen 3 open-weight model from the Hugging Face Hub.
2. Inspect and validate an instruction-style dataset.
3. Format examples, tokenize them, and configure LoRA adapters.
4. Fine-tune locally, preferring Apple Silicon's MPS backend when it is available.
5. Evaluate the adapter and compare base-model versus adapted responses.
6. Save, reload, and share the lightweight LoRA adapter—without duplicating the base model.

The first implementation targets the deliberately small `Qwen/Qwen3-0.6B` model and a small dataset. This keeps the feedback loop short and makes the resource trade-offs understandable before you scale up.

## Why LoRA on a MacBook?

Full fine-tuning updates every model weight and quickly becomes impractical on a laptop. LoRA keeps the original Qwen weights frozen and learns a much smaller set of adapter weights. That makes experiments cheaper to store, easier to repeat, and more suitable for local hardware—while still allowing a model to specialize.

Local fine-tuning is constrained by unified memory, thermal limits, model size, sequence length, and data quality. This repository treats those constraints as part of the design, rather than hiding them behind a one-line command.

## Project principles

* **Spec-driven:** every meaningful feature starts with a concise specification in [`specs/`](specs/).
* **Mac-first:** PyTorch's Apple Silicon/MPS backend is preferred where supported; CPU fallback is explicit.
* **Open and reproducible:** models come from Hugging Face Hub; configuration, seeds, and dataset assumptions are documented.
* **Learn by inspecting:** notebooks explain the workflow, while scripts hold reusable implementation logic.
* **Adapter-only outputs:** keep downloaded base weights, checkpoints, and caches out of Git.

## Prerequisites

* macOS and Conda (or Miniconda)
* The existing Conda environment named `llm-fine-tuning-on-mac`
* Python compatible with the dependencies in `requirements.txt`
* Internet access for the initial Hugging Face model and dataset download
* A Hugging Face account/token only if a future model or dataset requires authentication

The selected model is [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B), a public Apache-2.0 Causal Language Model in Safetensors format. Its 0.6B parameter scale makes it the initial learning target, not a guarantee that every training configuration will suit every MacBook.

## Setup

Activate the dedicated environment and install the project dependencies:

```bash
conda activate llm-fine-tuning-on-mac
python -m pip install -r requirements.txt
```

Confirm the core packages are available:

```bash
python -c "import torch, transformers, peft, datasets; print(torch.__version__, transformers.__version__, peft.__version__, datasets.__version__)"
```

Start the notebook environment when notebooks are added:

```bash
jupyter lab
```

## Download Qwen3-0.6B from Hugging Face Hub

The project uses the [`hf` command-line interface](https://huggingface.co/docs/huggingface_hub/guides/cli) supplied by `huggingface_hub`. The selected repository is public, so the following commands do not need a token.

From the repository root, activate the environment and download the complete model, tokenizer, and configuration to the project-local artifact directory:

```bash
conda activate llm-fine-tuning-on-mac
hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir artifacts/models/Qwen3-0.6B
```

The final command prints the local destination. `artifacts/` is excluded from Git, including the metadata cache created by the CLI, so base-model files cannot be committed by accident.

To use a newer Hub revision later, do not replace the hash ad hoc: update the [model acquisition specification](specs/003-qwen3-0.6b-model-acquisition.md), record the new full commit hash, and rerun the command. For a private or gated future artifact, authenticate interactively with `hf auth login`; never paste a token into a command that is saved in shell history.

## Planned layout

```text
.
├── artifacts/      # Downloaded models and run outputs (ignored by Git)
├── specs/          # Behaviour, constraints, and acceptance criteria
├── notebooks/      # Guided, restartable experiments
├── src/            # Reusable training, data, and inference code
├── tests/          # Fast tests with no model downloads
├── requirements.txt
└── README.md
```

The `notebooks/`, `src/`, and `tests/` directories will be introduced with the specifications that define their first behaviour.

## Apple Silicon and MPS

The project uses PyTorch's Metal Performance Shaders (MPS) backend to run eligible tensor operations on the Mac GPU. MPS is selected automatically when the installed PyTorch build and the active machine make it available; otherwise the workflow reports an explicit CPU fallback. A user who explicitly requests MPS receives a clear error if it is unavailable—never a hidden change of device.

The complete device-selection and verification contract is defined in the [Apple Silicon MPS execution specification](specs/002-apple-silicon-mps-execution.md). It also explains why unsupported-operation fallback is opt-in and why each training run must record its precision and resource assumptions.

## Working spec-first

Before adding a notebook, script, model, or dataset, read the relevant document in [`specs/`](specs/). A specification captures the problem, scope, assumptions, acceptance criteria, and how the work will be verified. The initial [project foundation specification](specs/001-project-foundation.md) establishes the baseline for this repository.

## Dependency choices

`transformers` loads Qwen models and provides the training interfaces; `peft` adds LoRA adapters; `torch` executes training; `accelerate` coordinates device and mixed-precision settings; `datasets` provides dataset handling; `trl` supplies supervised fine-tuning utilities; and JupyterLab supports the interactive lessons. `safetensors` and `sentencepiece` cover common model artifact and tokenizer requirements.

Version ranges intentionally keep the initial project compatible with Qwen 3-era Transformers releases while avoiding unreviewed major upgrades. Pin exact versions in a future training-run specification when strict experiment reproducibility is needed.

## Data, models, and privacy

Never commit Hugging Face tokens, private data, downloaded base models, adapter checkpoints, or training caches. Use public or properly licensed data, document its provenance, and remove personal or sensitive information before training. Keep secrets in environment variables or an ignored `.env` file.

## Status

The project foundation, MPS execution contract, and Qwen3-0.6B acquisition path are in place. The next specification will define the dataset schema and baseline LoRA configuration, then introduce the first notebook and training script.

## License

This project is released under the [MIT License](LICENSE).
