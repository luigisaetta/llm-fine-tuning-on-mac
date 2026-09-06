# Demo 05: Compare the Base Model and Fine-Tuned Adapter on MPS

This notebook measures the local Qwen3-1.7B base model against the LoRA adapter created by Demo 01 on exactly the same held-out validation records. It uses the same deterministic token-F1 evaluation contract as Demo 01 and reports base score, fine-tuned score, and the difference for exact-match rate, mean token F1, and threshold accuracy.

## Prerequisites

* Download `Qwen/Qwen3-1.7B` to `artifacts/models/Qwen3-1.7B/`. This project does not pin the Hugging Face revision.
* Generate `artifacts/datasets/cv-qa/eval.jsonl` through the dataset-preparation workflow.
* Complete [Demo 01](../demo01/README.md) so the adapter exists in `artifacts/training/demo01-qwen3-1.7b-lora/adapter/`.
* Use macOS 14 or later with an available PyTorch MPS backend and the `llm-fine-tuning-on-mac` Conda environment.

From the repository root:

```bash
conda activate llm-fine-tuning-on-mac
jupyter lab
```

Open [demo05_mps_base_adapter_comparison.ipynb](demo05_mps_base_adapter_comparison.ipynb), select the project kernel, review the configuration cell, and run it in order.

The notebook intentionally requires MPS; it does not silently fall back to CPU. It evaluates models sequentially to limit unified-memory use, releases the base model before loading the adapter model, and reads only local ignored artifacts. It does not save generations or metric results.
