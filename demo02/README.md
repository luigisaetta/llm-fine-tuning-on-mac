# Demo 02: Ask the Fine-Tuned Model

This notebook loads the local Qwen3-1.7B base model with the LoRA adapter created by Demo 01, then lets you ask any question through one editable prompt.

## Prerequisites

* Complete [Demo 01](../demo01/README.md) so the adapter exists under artifacts/training/demo01-qwen3-1.7b-lora/adapter/.
* Keep the base model under artifacts/models/Qwen3-1.7B/.
* Use the llm-fine-tuning-on-mac Conda environment.

From the repository root:

    conda activate llm-fine-tuning-on-mac
    jupyter lab

Open [demo02_adapter_inference.ipynb](demo02_adapter_inference.ipynb), select the project kernel, and run the cells in order.

Edit USER_PROMPT in the final cell to ask another question. English is recommended because the initial fine-tuning dataset uses English questions and answers. The notebook reads only local, ignored artifacts.
