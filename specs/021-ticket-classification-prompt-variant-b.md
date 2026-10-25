# Ticket Classification Prompt Variant B Experiment

## Problem

Demo 06 trains and Demo 07 evaluates with the ticket text as the only prompt. The base-model baseline cannot know the task, the JSON format, or the label codes, so the reported fine-tuning delta is overstated. The first step of the prompt experiment compares the current behavior (variant A, no system prompt) with the short system prompt (variant B) defined in [specification 020](020-ticket-classification-system-prompt.md).

## Scope

* Demo 06: an editable `PROMPT_VARIANT` setting, `'none'` (A) or `'short'` (B), applied to training formatting, per-epoch validation generation, and the severity confusion matrix.
* Demo 07: evaluate the base model with both variants and the adapter with the variant it was trained with.
* Record the prompt variant with every saved adapter, and refuse to evaluate an adapter with a different variant.

## Non-goals

* Variant C, which needs `MAX_SEQUENCE_LENGTH` 1024 and a separate run.
* Changing datasets, LoRA settings, optimizer, epochs, seed, or checkpoint policy. The batch settings changed separately on 2026-10-25 (specification 015); the existing variant A adapter used the previous settings, so variant A must be retrained with the current settings, and the same PyTorch version, before comparing it with variant B.
* Demo 09 and the OCI AI Data Platform notebook.

## Requirements

* Default `PROMPT_VARIANT` is `'short'`. Variant `'none'` reproduces the previous prompt formatting exactly.
* For variant `'short'`, the notebooks read `ticket-classification/prompts/system_prompt_short.txt` with `read_text(encoding='utf-8').strip()` and send it as a `system` message before the ticket.
* Adapters are saved below `ticket-classification/artifacts/models/`, in `demo06-qwen3-1.7b-ticket-classification-lora/` for variant `'none'` (unchanged, so the existing adapter remains the A reference) and `demo06-qwen3-1.7b-ticket-classification-lora-prompt-short/` for variant `'short'`.
* Demo 06 writes `prompt_variant.json` to the adapter directory with the variant name, the prompt file name, and the SHA-256 of the prompt text.
* Before training, Demo 06 fails with an actionable error if any formatted training or validation sequence exceeds `MAX_SEQUENCE_LENGTH`.
* Demo 07 reads `prompt_variant.json`. It fails if the recorded variant or prompt hash differs from its configured variant. An adapter without the file is accepted only for variant `'none'`, as a pre-experiment adapter.
* Demo 07 reports, for each metric, the base model with variant A, the base model with the adapter's variant, the adapter, and the adapter-minus-base delta for the same prompt.

## Acceptance criteria

* Static tests confirm the variant setting, the system-message construction in every generation path, the metadata file, the sequence-length check, and the four-column comparison.
* Notebook helper functions, executed outside Jupyter with a tokenizer, build identical prompts for training and evaluation and keep every sequence within 512 tokens for variant B.
* Remote or full local training is not part of this change; the training run and its results are recorded after the user runs Demo 06 and Demo 07.

## Verification

* 2026-10-25: static tests and an offline check of the prompt helpers with the local Qwen3-1.7B tokenizer. No training or model generation was run.

## Results

Pending: to be recorded after the variant B run of Demo 06 and Demo 07.
