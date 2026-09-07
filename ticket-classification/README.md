# Ticket Classification Fine-Tuning

This directory will contain a Mac-first LoRA fine-tuning demo for structured IT-support ticket classification. Starting from the local `Qwen/Qwen3-1.7B` base model, the future notebook will train an adapter to classify each ticket and return a concise JSON result.

## Expected model output

```json
{
  "category": "DB-01",
  "severity": "P2",
  "summary": "Production database performance degradation"
}
```

The response must be JSON only. Categories are limited to `APP-01`, `AUTH-01`, `DB-01`, `DB-02`, `DEV-01`, `MAIL-01`, `MOB-01`, `NET-01`, `SEC-01`, and `SRV-01`; severities are `P1`, `P2`, `P3`, or `P4`.

## Local prerequisites

* Download `Qwen/Qwen3-1.7B` to `../artifacts/models/Qwen3-1.7B/`. The project does not pin the Hub revision.
* Keep the following local datasets under `artifacts/datasets/` in this directory:

  | File | Purpose | Records at specification time |
  | --- | --- | ---: |
  | `train_dataset.jsonl` | LoRA training | 1,000 |
  | `validation_dataset.jsonl` | Per-epoch validation and best-checkpoint selection | 200 |
  | `test_dataset.jsonl` | Final held-out evaluation only | 200 |

* Use the `llm-fine-tuning-on-mac` Conda environment on macOS 14 or later with an available MPS backend.

The supplied datasets are synthetic and version-controlled. Training outputs remain local, ignored artifacts. Do not place ticket text or generated responses in the committed notebook.

## Planned workflow

1. Validate the three dataset splits and their JSON response schema.
2. Fine-tune a BF16 LoRA adapter on training records.
3. Measure validation loss at the end of every epoch and restore the best adapter.
4. Evaluate the restored adapter once on the held-out test split.
5. Report JSON validity, category and severity accuracy, joint-label accuracy, exact structured-response accuracy, and summary token F1.

The full behavioural and evaluation contract is defined in [specification 015](../specs/015-ticket-classification-lora-fine-tuning.md). The training notebook has not been added yet.
