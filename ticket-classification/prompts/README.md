# Ticket-Classification System Prompts

These system prompts are derived from version 1.0 of the [classification guide](../classification_guide.md). They are defined by [specification 020](../../specs/020-ticket-classification-system-prompt.md).

| File | Variant | Content |
| --- | --- | --- |
| `system_prompt_short.txt` | B | Task, JSON format, category codes with one-line names, severity scale |
| `system_prompt_full.txt` | C | Variant B plus category definitions, boundary rules, the severity procedure, and summary rules |

Variant A sends no system prompt. Demos 06 and 07 select variants A and B with their prompt-variant settings ([specification 021](../../specs/021-ticket-classification-prompt-variant-b.md)); variant C is not yet wired into the notebooks.

## Usage rules

* Read the file at runtime with `Path(...).read_text(encoding="utf-8").strip()` and send it unchanged as the first message, with role `system`, before the ticket's `user` message.
* Use the same variant for training, validation, test evaluation, and inference of an adapter. An adapter trained with one variant must not be evaluated or served with another.
* Do not edit a prompt in place for a running experiment. When the guide changes, regenerate the prompts, update this README, and retrain the adapters that use them.

## Token budget

Measured on 2026-10-25 with the local `Qwen/Qwen3-1.7B` tokenizer, over all 1,400 records of training V2, validation V2, and test, with the Qwen3 chat template applied to the system, user, and assistant messages:

| Variant | System prompt tokens | Mean sequence tokens | Longest sequence tokens |
| --- | ---: | ---: | ---: |
| A (none) | 0 | 63 | 73 |
| B (short) | 194 | 262 | 272 |
| C (full) | 869 | 937 | 947 |

Consequences for training on a Mac:

* Variant B fits Demo 06's `MAX_SEQUENCE_LENGTH = 512`.
* Variant C does not fit 512 tokens. Raise `MAX_SEQUENCE_LENGTH` to 1024 before training with it; otherwise truncation removes the assistant completion, and with completion-only loss the affected examples train on nothing.
* Each training sequence is about 4 times longer with variant B and about 15 times longer with variant C than without a prompt. Expect training and per-epoch validation generation time and memory to grow accordingly; measure them before choosing a variant.
