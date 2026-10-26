# Shared Python Package and Didactic Ticket-Classification Notebooks

## Problem

Every notebook is self-contained: Demos 06, 07, and 08 repeat helpers for project paths, device checks, dataset validation, prompt construction, generation, and scoring, in long single-line cells. The duplication makes the notebooks hard to read, hard to keep consistent (training and evaluation must build identical prompts), and impossible to unit-test. The settings a learner typically changes are mixed with implementation code.

## Scope

Phase 1 covers the Mac ticket-classification notebooks:

* `ticket-classification/demo06_ticket_classification_lora_fine_tuning.ipynb`
* `ticket-classification/demo07_ticket_classification_test_evaluation.ipynb`
* `ticket-classification/demo08_severity_dataset_shift_analysis.ipynb`

It adds an installable package, `llm_fine_tuning`, under `src/`, with `pyproject.toml` at the repository root:

| Module | Responsibility |
| --- | --- |
| `llm_fine_tuning.paths` | Project-root discovery and local model-directory validation |
| `llm_fine_tuning.device` | MPS/CPU selection, BF16 MPS requirement, runtime description |
| `llm_fine_tuning.models` | Tokenizer, base-model, and adapter loading; parameter dtype checks; memory release |
| `llm_fine_tuning.display` | Markdown tables, settings tables, loss curves, confusion-matrix plots |
| `llm_fine_tuning.tickets.labels` | Allowed categories, severities, and output keys |
| `llm_fine_tuning.tickets.data` | Ticket-project paths, split loading and validation, TRL formatting |
| `llm_fine_tuning.tickets.prompts` | Prompt variants, system-prompt loading, message construction, adapter naming, prompt metadata, sequence-length check |
| `llm_fine_tuning.tickets.evaluation` | Response parsing, scoring, batched generation, severity confusion matrix |
| `llm_fine_tuning.tickets.training` | Training settings, LoRA and SFT configuration, per-epoch validation callback, adapter saving |
| `llm_fine_tuning.tickets.shift_analysis` | Demo 08 deterministic measurements and plots |

Later phases, each with its own update to this specification, cover Demos 00 to 05, Demo 09, Demo 04, and the OCI AI Data Platform notebooks.

## Non-goals

* Changing training, evaluation, or analysis behavior beyond the decisions listed below.
* Changing datasets, prompts, the classification guide, or the number of epochs.
* Replacing PyTorch, Transformers, PEFT, or TRL.

## Installation

```bash
python -m pip install -r requirements.txt -c constraints-macos.txt
python -m pip install --no-deps -e .
```

The package declares no dependencies of its own; `requirements.txt` remains the single dependency list. The editable install works the same on macOS, Linux/CUDA, and OCI, which keeps later phases portable.

## Notebook structure

Each notebook follows the same didactic layout:

1. Title, learning goal, and a short workflow overview.
2. A clearly marked **Settings** cell containing only the values a learner typically changes, with one comment per value.
3. One section per workflow step: a Markdown cell explaining what happens and why, then a short code cell calling the package.
4. Results presented as tables and plots, followed by a short interpretation note.

Code cells must stay short (as a guideline, under about 15 lines, except the Settings and import cells), contain no function or class definitions, and import only from `llm_fine_tuning`, `torch`, and `transformers`.

## Behavior decisions

* The training and validation-loss settings, SFT configuration, checkpoint policy, output directories, `prompt_variant.json` content, sequence-length check, and metric definitions are unchanged.
* Demo 06 no longer exposes `DEVICE_REQUEST`: BF16 training already required MPS, so the setting could only produce an error.
* Demo 06's severity confusion matrix reuses the batched generation of the per-epoch validation metric, instead of separate unbatched generation, so that the matrix and the reported severity accuracy come from the same predictions.
* Demo 07 validates the test split with the same validation as the training splits (schema, allowed labels, no duplicate tickets). The current test split satisfies it.
* The `'full'` prompt variant (C) becomes selectable. With the current `MAX_SEQUENCE_LENGTH` of 512, Demo 06 stops with the existing sequence-length error before training.
* The default prompt variant of Demos 06 and 07 becomes `'none'` (A), following the conclusion recorded in [specification 021](021-ticket-classification-prompt-variant-b.md).

## Acceptance criteria

* `pytest` runs unit tests for the package without models, network, or GPU, covering dataset validation, prompt construction and metadata, scoring, the confusion matrix, SFT configuration mapping, and the Demo 08 measurements.
* Notebook structure tests confirm: no stored outputs, a Settings cell, no function or class definitions in code cells, imports from `llm_fine_tuning`, Demo 06 does not reference the test split, and Demo 07 does not reference the training or validation splits.
* An offline equivalence check against the previous notebook code, using the real datasets and the local tokenizer, shows identical dataset validation results, TRL records, rendered prompts, scores for the same responses, SFT configuration, and Demo 08 measurements.

## Verification

Performed on 2026-10-26 on macOS (Apple M4 Pro), PyTorch 2.14.1, Transformers 4.57.6, TRL 0.29.1, PEFT 0.20.0:

* `pytest`: 62 tests passed, including 27 new package unit tests and the rewritten notebook structure tests.
* Equivalence with the previous notebook code (git `HEAD`), executed side by side on the real datasets with the local tokenizer: identical split validation, TRL prompt/completion records, rendered generation prompts, longest sequence lengths, adapter directory names, and system prompts for variants `'none'` and `'short'`; identical metrics, tokenizer inputs, and `generate` arguments for the Demo 06 per-epoch metric and the Demo 07 test metric, using a stand-in model that returns predetermined valid, wrong, and invalid responses; identical `SFTConfig.to_dict()` and `LoraConfig.to_dict()`.
* Demo 08 executed end to end with the old and the new notebook: identical text and table outputs and byte-identical plot images.
* Demo 06 executed with the real model up to the creation of the trainer (data loading, example rendering, sequence check, LoRA wrapping, dtype check, 0.504% trainable parameters); training was not run.
* Demo 07 executed end to end with the real model and the existing variant A adapter on the first 4 test tickets.
* Size: Demo 06 code went from 384 to 69 lines, with at most 17 lines per cell.
* Black and Pylint are not installed in the environment, so formatting and linting were not run.
