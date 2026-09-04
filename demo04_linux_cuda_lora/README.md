# Demo 04: Linux CUDA LoRA Fine-Tuning

This Linux-only notebook fine-tunes local `Qwen/Qwen3-1.7B` with a LoRA adapter on the private parametric-memory dataset. It requires one CUDA-capable NVIDIA GPU with BF16 support and does not provide a CPU fallback.

## Before starting

Install CUDA-enabled PyTorch appropriate for the NVIDIA driver and CUDA runtime on the target host. PyTorch and CUDA installation are intentionally outside this guide.

Install the following non-PyTorch packages in the environment that already contains CUDA-enabled PyTorch:

```bash
python -m pip install \
  "jupyterlab>=4.4,<5" \
  "ipywidgets>=8.1,<9" \
  "matplotlib>=3.9,<4" \
  "huggingface_hub>=0.30,<1" \
  "transformers>=4.51,<5" \
  "peft>=0.15,<1" \
  "accelerate>=1.4,<2" \
  "datasets>=3.3,<4" \
  "trl>=0.15,<1" \
  "safetensors>=0.5,<1" \
  "sentencepiece>=0.2,<1" \
  "tqdm>=4.67,<5"
```

`ipywidgets` is used by JupyterLab, and `tqdm` supports training progress reporting. No OCI, LangChain, or test-only package is required to run this notebook.

## Local artifacts and paths

Download the base model and prepare the private dataset on the Linux host. The notebook has one clearly marked configuration cell where you must set these directories:

* `MODEL_DIRECTORY`: local Qwen3-1.7B model directory.
* `DATASET_DIRECTORY`: directory containing `train.jsonl` and `eval.jsonl`.
* `TRAINING_OUTPUT_DIRECTORY`: directory where checkpoints, logs, and the adapter are written.

The defaults are project-relative paths under ignored `artifacts/`. Change them if the model, dataset, or output should live elsewhere. Do not commit private datasets or generated artifacts.

## Run

From the repository root, start JupyterLab in the Linux Python environment:

```bash
jupyter lab
```

Open [demo04_lora_fine_tuning_linux_cuda.ipynb](demo04_lora_fine_tuning_linux_cuda.ipynb), review the path and training configuration cells, then run the notebook in order. The notebook stops before loading the model if CUDA or CUDA BF16 support is unavailable.

The run evaluates validation loss after each epoch, saves only the LoRA adapter and tokenizer, plots training and validation losses, and reports held-out deterministic-generation metrics.
