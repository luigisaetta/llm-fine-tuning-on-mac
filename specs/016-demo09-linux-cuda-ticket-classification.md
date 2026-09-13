# Demo 09: Linux CUDA Ticket Classification LoRA Fine-Tuning

## Problem

The ticket-classification LoRA training workflow needs a Linux variant for one CUDA-capable NVIDIA GPU. Demo 06 remains the macOS/MPS reference workflow.

## Scope

Demo 09 is the restartable notebook at `ticket-classification/demo09_ticket_classification_lora_fine_tuning_linux_cuda.ipynb`. It trains Qwen/Qwen3-1.7B with the Demo 06 ticket schema, train/validation split separation, LoRA configuration, per-epoch validation generation metrics, and adapter-only outputs. It targets one CUDA GPU.

## Assumptions

* CUDA-enabled PyTorch and an NVIDIA driver are installed separately on the Linux host.
* The selected GPU supports BF16.
* `Qwen/Qwen3-1.7B` and the synthetic ticket datasets are available locally. The model revision is not pinned.

## Requirements

* Provide one clearly labelled editable configuration cell defining `MODEL_DIRECTORY`, `DATASET_DIRECTORY`, and `TRAINING_OUTPUT_DIRECTORY`. The dataset directory contains `train_dataset_v2.jsonl` and `validation_dataset_v2.jsonl`; the output directory is the base directory for checkpoints, logs, tokenizer, training metadata, and the saved LoRA adapter.
* Before model loading, a dedicated diagnostics cell must require `torch.cuda.is_available()` and `torch.cuda.is_bf16_supported()`, select `cuda:0`, and display PyTorch version, compiled CUDA version, GPU count, selected GPU name, selected device, and BF16 status. CPU and MPS fallback are out of scope.
* Validate paths, dataset schema, label sets, empty values, duplicate records, and exact cross-split ticket overlap without displaying ticket text.
* Use BF16 base and LoRA parameters, `autocast_adapter_dtype=False`, `bf16=True`, and `bf16_full_eval=True`; fail if floating or trainable parameters are not BF16.
* Preserve Demo 06's fixed LoRA/training baseline and select the best epoch by generated validation joint category-and-severity accuracy. The test split is not loaded.
* Save only adapters and related metadata below `TRAINING_OUTPUT_DIRECTORY`; never duplicate the base model.

## Verification

* Parse the notebook as version-4 JSON and confirm code cells have no outputs or execution counts.
* Run static tests without downloading models or datasets.
* On the Linux host, execute the configuration, CUDA diagnostics, and model setup cells before a full training run.
