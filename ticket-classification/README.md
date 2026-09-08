# Ticket Classification Fine-Tuning

This directory contains a Mac-first LoRA fine-tuning demo for structured IT-support ticket classification. Starting from the local `Qwen/Qwen3-1.7B` base model, it trains an adapter to classify each ticket and return a concise JSON result.

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
  | `train_dataset_v2.jsonl` | LoRA training | 1,000 |
  | `validation_dataset_v2.jsonl` | Per-epoch validation and best-checkpoint selection | 200 |
  | `test_dataset.jsonl` | Final held-out evaluation only | 200 |

* Use the `llm-fine-tuning-on-mac` Conda environment on macOS 14 or later with an available MPS backend.

The supplied datasets are synthetic and version-controlled. Training outputs remain local, ignored artifacts. Do not place ticket text or generated responses in the committed notebook.

## Training workflow

1. Validate the training and validation splits and their JSON response schema.
2. Fine-tune a BF16 LoRA adapter on training records.
3. Measure validation loss and deterministic category, severity, joint-label, and valid-JSON metrics at the end of every epoch, then restore the adapter with the highest category-and-severity accuracy.
4. Save the restored best-joint-accuracy adapter and tokenizer below `artifacts/models/` in this directory.

## Held-out test evaluation

[Demo 07](demo07_ticket_classification_test_evaluation.ipynb) loads the adapter saved by Demo 06 and evaluates it only on `test_dataset.jsonl`. It reports Category Accuracy, Severity Accuracy, Category+Severity Accuracy, and Valid JSON Rate. The test split is never used for training or checkpoint selection.

From the repository root, activate the project environment and start JupyterLab:

```bash
conda activate llm-fine-tuning-on-mac
jupyter lab
```

Open [demo06_ticket_classification_lora_fine_tuning.ipynb](demo06_ticket_classification_lora_fine_tuning.ipynb), select the project kernel, review the configuration cell, and run the cells in order.

After Demo 06 completes, open [demo07_ticket_classification_test_evaluation.ipynb](demo07_ticket_classification_test_evaluation.ipynb) in the same kernel environment and run it in order.

The full behavioural and evaluation contract is defined in [specification 015](../specs/015-ticket-classification-lora-fine-tuning.md).
