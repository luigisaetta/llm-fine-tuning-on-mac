# Ticket Classification LoRA Fine-Tuning

## Problem

IT-support tickets must be classified consistently into a predefined category and severity, with a concise operational summary. The project needs a reproducible, Mac-first LoRA fine-tuning workflow that adapts the local Qwen3-1.7B model to produce this structured result.

## Scope

Demo 06 is a restartable training notebook at `ticket-classification/demo06_ticket_classification_lora_fine_tuning.ipynb`. Demo 07 is a separate held-out evaluation notebook at `ticket-classification/demo07_ticket_classification_test_evaluation.ipynb`. Both reuse the Qwen3-1.7B local model and the BF16 MPS LoRA approach of Demo 01.

This specification covers dataset validation, prompt/completion formatting, per-epoch validation during training, held-out test evaluation after training, adapter artifacts, and structured-output scoring. It does not cover a production ticketing integration, API deployment, automatic remediation, or full-parameter fine-tuning.

## Dataset contract

The version-controlled synthetic datasets are:

| Split | Path | Intended use | Records at specification time |
| --- | --- | --- | ---: |
| Training | `ticket-classification/artifacts/datasets/train_dataset_v2.jsonl` | Fit LoRA parameters. | 1,000 |
| Validation | `ticket-classification/artifacts/datasets/validation_dataset_v2.jsonl` | Evaluate loss after every epoch and select the best checkpoint. | 200 |
| Test | `ticket-classification/artifacts/datasets/test_dataset.jsonl` | One held-out, final evaluation after the best adapter is restored. | 200 |

Each JSONL record has a `messages` array with exactly a `user` ticket message followed by an `assistant` message. The assistant content must parse as one JSON object with exactly the keys `category`, `severity`, and `summary`. The notebook must reject empty ticket text, empty output values, invalid JSON, extra or missing keys, duplicate records within a split, and an exact ticket-text overlap between loaded splits.

The allowed labels are fixed by the supplied training data:

| Field | Allowed values |
| --- | --- |
| `category` | `APP-01`, `AUTH-01`, `DB-01`, `DB-02`, `DEV-01`, `MAIL-01`, `MOB-01`, `NET-01`, `SEC-01`, `SRV-01` |
| `severity` | `P1`, `P2`, `P3`, `P4` |

`summary` must be a non-empty string. The notebook must validate that each split uses only these category and severity values. The supplied datasets are synthetic and may be version-controlled. The notebook must not store ticket text, generated responses, credentials, model weights, or training artifacts in its committed output.

## Output contract

For one IT-support ticket, deterministic inference must return JSON only, with this exact schema:

```json
{
  "category": "DB-01",
  "severity": "P2",
  "summary": "Production database performance degradation"
}
```

Key order and whitespace are not scored. The category and severity must be allowed values, and the summary must be a non-empty string. Text surrounding the JSON object, malformed JSON, and extra keys make a response structurally invalid.

## Training and validation contract

* Load the local base model from `artifacts/models/Qwen3-1.7B/`. Its source identifier is `Qwen/Qwen3-1.7B`; this project does not currently pin a Hub revision.
* Convert every record to TRL prompt/completion format, using the user message as the prompt and the assistant JSON as the completion.
* Start from the Demo 01 configuration baseline: LoRA rank 8, alpha 16, dropout 0.05, Q/K/V/O and gate/up/down projection target modules, learning rate `1e-4`, micro-batch size 1, gradient accumulation 8, maximum sequence length 512, and eight epochs. Keep every setting in one editable notebook cell.
* Use `eval_strategy="epoch"`, `save_strategy="epoch"`, `load_best_model_at_end=True`, and `metric_for_best_model="eval_joint_accuracy"` with `greater_is_better=True`. The validation split is the only split passed to the trainer.
* At the end of every epoch, after the unchanged trainer validation-loss evaluation, generate one deterministic response for every validation ticket. Immediately display category accuracy, severity accuracy, joint category-and-severity accuracy, and valid-JSON rate alongside the training and validation losses. This task-specific evaluation must not access the test split or alter training settings.
* Batch validation generation with left padding and an editable batch size of 4 as the MPS memory baseline. Enable the generation KV cache only for this inference path; training continues with its existing cache setting.
* Preserve the Demo 01 BF16 policy: MPS only, macOS 14 or later, model and LoRA parameters in `torch.bfloat16`, `autocast_adapter_dtype=False`, `bf16=True`, and `bf16_full_eval=True`. Fail with actionable errors if these requirements are not met.
* Save checkpoints, logs, tokenizer, adapter, configuration, and training metadata only below `ticket-classification/artifacts/models/demo06-qwen3-1.7b-ticket-classification-lora/`. Save an adapter, never a duplicate base model.

## Held-out test evaluation contract

Demo 06 intentionally loads only training and validation data. Demo 07 restores the saved best adapter, loads only the test split, and evaluates every test record once. The test split must not influence training, checkpoint selection, early stopping, hyperparameter selection, or prompt changes. Demo 07 reports:

* valid-JSON rate;
* category accuracy;
* severity accuracy;
* joint category-and-severity accuracy.

For all label metrics, an invalid response counts as incorrect. Use deterministic, batched generation with left padding and the generation KV cache enabled. Report aggregate numbers only; the notebook must not store ticket text or generated responses.

## Acceptance criteria

* The notebook validates all required local model, adapter-output, and dataset paths before loading a model.
* It validates schema, label sets, empty values, duplicate records, and cross-split exact ticket overlap without printing ticket content.
* The configuration and actual device, dtype, dataset-size, trainable-parameter information, and joint-accuracy checkpoint-selection policy are visible before training begins.
* Validation loss and task-specific metrics run at every epoch, and the best adapter is restored by maximum category-and-severity accuracy.
* Demo 06 loads only training and validation data; Demo 07 loads only the test split and the saved adapter.
* The notebook has no stored outputs, execution counts, ticket texts, credentials, datasets, model weights, or generated artifacts.
* Proportionate tests parse the notebook and validate its split usage, BF16/MPS policy, output schema checks, and no-output policy without loading a model or dataset.

## Verification

* Parse the notebook as version-4 JSON and confirm code cells have neither outputs nor execution counts.
* Run static and unit tests in the `llm-fine-tuning-on-mac` Conda environment; tests must use tiny fixtures or mocks and must not download a model or read the local datasets.
* Before an optional full run, inspect the selected MPS device, BF16 parameter dtypes, trainable-parameter count, and split sizes.
* After a full run, inspect epoch-level validation loss and final held-out test metrics. Review any optional local examples without saving them to the notebook.
