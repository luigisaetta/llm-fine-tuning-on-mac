# Demo 01: LoRA Fine-Tuning Qwen3-1.7B

This notebook fine-tunes the local Qwen3-1.7B model with a LoRA adapter on the private parametric-memory dataset about Luigi Saetta.

## Prerequisites

* Download [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) and keep the base model under `artifacts/models/Qwen3-1.7B`.
* Generate and validate the private files `artifacts/datasets/cv-qa/train.jsonl` and `artifacts/datasets/cv-qa/eval.jsonl`.
* Use the `llm-fine-tuning-on-mac` Conda environment.

From the repository root:

```bash
conda activate llm-fine-tuning-on-mac
jupyter lab
```

Open [`demo01_lora_fine_tuning.ipynb`](demo01_lora_fine_tuning.ipynb), select the project kernel, review the dedicated configuration cell, and run cells in order. The notebook loads the Qwen3 base model and trains the LoRA adapter in BF16. BF16 training on MPS requires macOS 14 or later; the notebook verifies the resulting floating-point parameter dtypes before training.

The notebook evaluates loss at the end of each epoch and then measures deterministic answer quality on the held-out evaluation facts. All checkpoints and adapters are saved under ignored `artifacts/training/` paths.
