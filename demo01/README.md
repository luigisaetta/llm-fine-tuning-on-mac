# Demo 01: LoRA Fine-Tuning Qwen3-0.6B

This notebook fine-tunes the local Qwen3-0.6B model with a LoRA adapter on the private parametric-memory dataset about Luigi Saetta.

## Prerequisites

* Complete [Demo 00](../demo00/README.md) and keep the base model under `artifacts/models/Qwen3-0.6B`.
* Generate and validate the private files `artifacts/datasets/cv-qa/train.jsonl` and `artifacts/datasets/cv-qa/eval.jsonl`.
* Use the `llm-fine-tuning-on-mac` Conda environment.

From the repository root:

```bash
conda activate llm-fine-tuning-on-mac
jupyter lab
```

Open [`demo01_lora_fine_tuning.ipynb`](demo01_lora_fine_tuning.ipynb), select the project kernel, review the dedicated configuration cell, and run cells in order.

The notebook evaluates loss at the end of each epoch and then measures deterministic answer quality on the held-out evaluation facts. All checkpoints and adapters are saved under ignored `artifacts/training/` paths.
