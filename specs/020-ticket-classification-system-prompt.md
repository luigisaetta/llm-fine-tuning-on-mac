# Ticket Classification System Prompt Specification

## Problem

Ticket-classification records contain only the ticket text and the expected JSON. The model receives no statement of the task, the output format, or the meaning of the labels, and the base-model baseline in Demo 07 cannot know the category codes. A shared system prompt, derived from the [classification guide](../ticket-classification/classification_guide.md), is needed before prompt variants can be compared.

## Scope

* Two plain-text system prompts derived from guide version 1.0, in `ticket-classification/prompts/`:
  * `system_prompt_short.txt` (variant B): task, JSON format, and the allowed codes with one-line names.
  * `system_prompt_full.txt` (variant C): variant B plus condensed category definitions, boundary rules, the severity procedure, and summary rules.
* A README in the same directory recording the source guide version, measured token counts, and usage rules.
* A static test that keeps the prompts consistent with the guide and the allowed labels.

## Non-goals

* Changing datasets, notebooks, training configuration, or evaluation; wiring the prompts into Demo 06, Demo 07, Demo 09, or the OCI AI Data Platform notebook belongs to the A/B experiment.
* Choosing between variants; that decision requires the experiment.

## Requirements

* Prompt files contain only the text sent to the model, in English, with no comments, versions, or metadata.
* Both prompts require JSON-only output with exactly `category`, `severity`, and `summary`, and list exactly the ten categories and four severities of the guide.
* The full prompt preserves the guide's precedence rule: stated impact first, contextual signals only when impact is not stated.
* Every consumer must read the prompt file at runtime and send it unchanged as the `system` message, both during training and during evaluation or inference of an adapter trained with it.
* A change to the guide that affects classification requires regenerating the prompts, updating the README, and retraining any adapter that used them.

## Acceptance criteria

* The default `pytest` suite verifies that each prompt lists exactly the guide's category codes and the four severities, and mentions the three JSON keys.
* The README records token counts measured with the local `Qwen/Qwen3-1.7B` tokenizer and the longest resulting training sequence relative to Demo 06's `MAX_SEQUENCE_LENGTH` of 512.

## Verification

Performed on 2026-10-25; results are recorded in `ticket-classification/prompts/README.md`. No model was run.
