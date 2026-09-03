# Fine-tune a Qwen 3 model on your Mac

Turn a compact open-weight language model into a model that understands *your* task—without leaving your MacBook.

This repository is a hands-on, spec-driven project for fine-tuning a small model from the [Qwen 3 family](https://huggingface.co/Qwen) with **LoRA** (Low-Rank Adaptation). It combines a reviewable dataset-generation guide with explorable Jupyter notebooks, so you can both understand each step and run the whole workflow again.

> The aim is not merely to launch training. It is to make the decisions behind a reliable local fine-tuning workflow visible: data format, prompt template, memory limits, LoRA configuration, evaluation, and adapter-based inference.

## What you will build

The project will guide you through a small, practical workflow:

1. Download the selected Qwen 3 open-weight model from the Hugging Face Hub.
2. Inspect and validate an instruction-style dataset.
3. Format examples, tokenize them, and configure LoRA adapters.
4. Fine-tune locally, preferring Apple Silicon's MPS backend when it is available.
5. Evaluate the adapter and compare base-model versus adapted responses.
6. Save, reload, and share the lightweight LoRA adapter—without duplicating the base model.

The first implementation targets the `Qwen/Qwen3-1.7B` model and a small dataset. This keeps the workflow inspectable while making the resource trade-offs visible.

## Why LoRA on a MacBook?

Full fine-tuning updates every model weight and quickly becomes impractical on a laptop. LoRA keeps the original Qwen weights frozen and learns a much smaller set of adapter weights. That makes experiments cheaper to store, easier to repeat, and more suitable for local hardware—while still allowing a model to specialize.

Local fine-tuning is constrained by unified memory, thermal limits, model size, sequence length, and data quality. This repository treats those constraints as part of the design, rather than hiding them behind a one-line command.

## Project principles

* **Spec-driven:** every meaningful feature starts with a concise specification in [`specs/`](specs/).
* **Mac-first:** PyTorch's Apple Silicon/MPS backend is preferred where supported; CPU fallback is explicit.
* **Open and reproducible:** models come from Hugging Face Hub; configuration, seeds, and dataset assumptions are documented.
* **Learn by inspecting:** the dataset guide and notebooks explain each stage of the workflow.
* **Adapter-only outputs:** keep downloaded base weights, checkpoints, and caches out of Git.
* **Traceable changes:** [`CHANGELOG.md`](CHANGELOG.md) records user-visible additions and significant changes before release.

## Prerequisites

* macOS and Conda (or Miniconda)
* The existing Conda environment named `llm-fine-tuning-on-mac`
* Python compatible with the dependencies in `requirements.txt`
* Internet access for the initial Hugging Face model and dataset download
* A Hugging Face account/token only if a future model or dataset requires authentication

The selected model is [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B), a public Apache-2.0 Causal Language Model in Safetensors format. Its parameter scale does not guarantee that every training configuration will suit every MacBook.

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

## Download Qwen3-1.7B from Hugging Face Hub

The project uses the [`hf` command-line interface](https://huggingface.co/docs/huggingface_hub/guides/cli) supplied by `huggingface_hub`. The selected repository is public, so the following commands do not need a token.

From the repository root, activate the environment and download the complete model, tokenizer, and configuration to the project-local artifact directory:

```bash
conda activate llm-fine-tuning-on-mac
hf download Qwen/Qwen3-1.7B \
  --local-dir artifacts/models/Qwen3-1.7B
```

The final command prints the local destination. `artifacts/` is excluded from Git, including the metadata cache created by the CLI, so base-model files cannot be committed by accident.

For a future reproducible revision pin, update the [model acquisition specification](specs/003-qwen3-1.7b-model-acquisition.md), record the full commit hash, and rerun the command. For a private or gated future artifact, authenticate interactively with `hf auth login`; never paste a token into a command that is saved in shell history.

## Included demos

### Demo 00 — local inference

[`demo00/`](demo00/) contains a restartable notebook that loads the downloaded Qwen3-1.7B model from `artifacts/`, verifies its Qwen configuration and selected MPS/CPU device, then generates an answer to an editable prompt. It is the quick local-health check before any training run.

### Demo 01 — LoRA factual-recall fine-tuning

[`demo01/`](demo01/) contains the end-to-end fine-tuning notebook. It loads an ignored, private Q&A dataset, validates its conversational schema and recall-evaluation contract, trains a PEFT/LoRA adapter on MPS when available, evaluates loss after every epoch, saves only adapter artifacts, and generates deterministic answers for held-out question paraphrases.

The default experiment uses Qwen3-1.7B, a LoRA adapter with rank 8 and alpha 16, a learning rate of `1e-4`, and eight epochs. See the [Demo 01 guide](demo01/README.md) for how to run it.

### Demo 02 — fine-tuned adapter inference

[demo02/](demo02/) loads the local Qwen3-1.7B base model together with the LoRA adapter created by Demo 01. It provides an editable prompt for any question and recommends English questions, matching the language of the initial fine-tuning data. Use it as a manual behaviour check: strong recall on the curated facts does not guarantee reliable answers to arbitrary or substantially reworded questions.

## Merge and publish a standalone model

The LoRA adapter is normally the preferred local artifact because it is small and retains the base-model provenance. To create a standalone model, merge it into the local Qwen3-1.7B weights. The merge uses `float32`, defaults to CPU, and needs enough local memory for the base model, adapter, and merged weights. It writes only to a new or empty ignored directory.

```bash
conda activate llm-fine-tuning-on-mac
python scripts/merge_lora_adapter.py
```

To use MPS explicitly, add `--device mps`; the script stops with an error if MPS is unavailable. Review the merged model locally before publishing it. Upload requires a write token with model-repository permissions in the `HF_TOKEN` environment variable, an explicit target repository, and the opt-in flag. Set the token through a shell profile, secret manager, or other mechanism that does not save it in shell history:

```bash
python scripts/merge_lora_adapter.py \
  --push-to-hub \
  --repo-id YOUR_USERNAME/qwen3-1.7b-lora-merged \
  --private
```

`--private` affects a newly created repository. If the repository already exists, the explicit command uploads a new commit to it. Do not publish a merged model if its adapter was trained on private or personal data that must not be shared.

The generated model card already declares `Qwen/Qwen3-1.7B` as `base_model`, the Transformers library, text-generation pipeline, LoRA merge provenance, unpinned base revision, private-dataset status, and limitations. Before public publication, review it on the Hub and add only non-sensitive information that you can substantiate.

If the local merge completed but the upload failed, do not rerun the merge or overwrite its directory. Retry only the upload after setting `HF_TOKEN` securely:

```bash
python scripts/merge_lora_adapter.py \
  --push-to-hub \
  --upload-existing-output \
  --repo-id YOUR_USERNAME/qwen3-1.7b-lora-merged \
  --private
```

Future hypotheses for improving factual recall are recorded in [experiments/README.md](experiments/README.md).

## Repository layout

```text
.
├── artifacts/      # Downloaded models and run outputs (ignored by Git)
├── CHANGELOG.md     # Unreleased and released user-visible changes
├── demo00/          # First local model-loading and prompting notebook
├── demo01/          # LoRA fine-tuning and held-out evaluation notebook
├── demo02/          # Fine-tuned LoRA adapter inference notebook
├── experiments/     # Proposed follow-up experiments
├── dataset_preparation/  # ChatGPT-assisted CV-to-Q&A dataset guide
├── specs/          # Behaviour, constraints, and acceptance criteria
├── scripts/        # Reusable local model-management commands
├── tests/          # Fast tests with no model downloads
├── requirements.txt
└── README.md
```

Additional reusable training and data modules will be introduced with the specifications that define their first behaviour.

## Apple Silicon and MPS

The project uses PyTorch's Metal Performance Shaders (MPS) backend to run eligible tensor operations on the Mac GPU. MPS is selected automatically when the installed PyTorch build and the active machine make it available; otherwise the workflow reports an explicit CPU fallback. A user who explicitly requests MPS receives a clear error if it is unavailable—never a hidden change of device.

The complete device-selection and verification contract is defined in the [Apple Silicon MPS execution specification](specs/002-apple-silicon-mps-execution.md). It also explains why unsupported-operation fallback is opt-in and why each training run must record its precision and resource assumptions.

## Working spec-first

Before adding a notebook, script, model, or dataset, read the relevant document in [`specs/`](specs/). A specification captures the problem, scope, assumptions, acceptance criteria, and how the work will be verified. The initial [project foundation specification](specs/001-project-foundation.md) establishes the baseline for this repository.

## Dependency choices

`transformers` loads Qwen models and provides the training interfaces; `peft` adds LoRA adapters; `torch` executes training; `accelerate` coordinates device and mixed-precision settings; `datasets` provides dataset handling; `trl` supplies supervised fine-tuning utilities; and JupyterLab supports the interactive lessons. `safetensors` and `sentencepiece` cover common model artifact and tokenizer requirements.

Version ranges intentionally keep the initial project compatible with Qwen 3-era Transformers releases while avoiding unreviewed major upgrades. Pin exact versions in a future training-run specification when strict experiment reproducibility is needed.

## Status

The planned local workflow is complete: Qwen3-1.7B acquisition, local inference, LoRA fine-tuning, evaluation, adapter inference, standalone merge, and optional Hub publication have been implemented and verified through the available local checks. The remaining improvement area is a broader, more diverse, and carefully reviewed training and evaluation dataset.

## License

This project is released under the [MIT License](LICENSE).
