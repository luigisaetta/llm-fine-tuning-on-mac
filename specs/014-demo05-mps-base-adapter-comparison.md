# Demo 05: MPS Base-Model and LoRA-Adapter Comparison

## Problem

After LoRA fine-tuning, learners need a direct, repeatable measurement of how the original Qwen3-1.7B model compares with the fine-tuned adapter on the same held-out validation records.

## Scope

Create `demo05/` with a restartable notebook and usage guide. The notebook evaluates the local Qwen3-1.7B base model and the Demo 01 LoRA adapter on `artifacts/datasets/cv-qa/eval.jsonl`, using MPS only. It reports the same generative metrics used by Demo 01 and a direct metric delta from base model to adapter.

## Assumptions

* The local base model is `Qwen/Qwen3-1.7B`, stored under `artifacts/models/Qwen3-1.7B/`; its Hub revision is not pinned by this project.
* Demo 01 has created the BF16 LoRA adapter under `artifacts/training/demo01-qwen3-1.7b-lora/adapter/`.
* `eval.jsonl` is the held-out validation split defined by specification 008. It contains no private data in the notebook itself and is never uploaded.
* An available MPS backend and macOS 14 or later are required for the BF16 comparison. CPU fallback is deliberately out of scope so that the comparison is an Apple Silicon MPS run.

## Evaluation contract

* Both variants use the same validation records, Qwen chat template with thinking disabled, `do_sample=False`, maximum generated-token limit, and BF16 dtype on MPS.
* The base model is evaluated first. It is released before loading the adapter model, so the notebook does not retain two 1.7B models in unified memory.
* The adapter variant loads a fresh local base model plus the LoRA adapter. No model, dataset, or adapter is downloaded or uploaded.
* For each variant, score exact-match rate, mean bag-of-token F1, and accuracy at token F1 >= 0.80. These definitions are identical to Demo 01.
* Present one direct comparison table containing base score, fine-tuned score, and fine-tuned-minus-base delta for every metric. Also report elapsed evaluation time as context, not as a quality metric.

## Acceptance criteria

* The notebook stops with actionable errors for missing local artifacts, malformed or empty validation examples, unavailable MPS, or macOS below version 14.
* It does not contain outputs, execution counts, dataset rows, credentials, model artifacts, or generated answers.
* It evaluates each model on every validation example, with the same deterministic generation settings.
* It checks that the loaded base model is Qwen3, model parameters are on MPS and BF16, and the adapter model is a PEFT model.
* `demo05/README.md`, the root README, and the changelog document the comparison demo and its artifact prerequisites.

## Verification

* Parse the notebook as version-4 JSON and confirm all code-cell outputs are empty.
* Run static tests that assert the MPS-only policy, shared evaluation function, required artifacts, and direct comparison fields.
* Optionally run the notebook in the `llm-fine-tuning-on-mac` Conda environment after Demo 01. Review the comparison table and representative generations locally; do not save sensitive outputs in the notebook.
