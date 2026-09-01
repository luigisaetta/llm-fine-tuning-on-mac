# Demo 00: Local Qwen3-0.6B Inference

This notebook is the first local checkpoint before LoRA fine-tuning: it loads the Qwen3-0.6B model already stored in `artifacts/`, confirms the model and device configuration, and generates a response to an editable prompt.

## Prerequisites

Complete the model download described in the [root README](../README.md#download-qwen3-06b-from-hugging-face-hub). The expected file is:

```text
artifacts/models/Qwen3-0.6B/model.safetensors
```

Start JupyterLab from the repository root with the project Conda environment:

```bash
conda activate llm-fine-tuning-on-mac
jupyter lab
```

Open [`demo00_qwen3_0_6b_inference.ipynb`](demo00_qwen3_0_6b_inference.ipynb), select the `llm-fine-tuning-on-mac` kernel, and run all cells in order.

The notebook never downloads model files: it uses `local_files_only=True`. Change `USER_PROMPT` in the final section to ask a different question.
