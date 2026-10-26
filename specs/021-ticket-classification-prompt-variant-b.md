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
* 2026-10-25: the user ran Demo 06 and Demo 07 locally for variants A and B; results are below.

## Results

### Demo 06 training runs (2026-10-25)

Environment: macOS, Apple M4 Pro, MPS, BF16, PyTorch 2.14.1, Transformers 4.57.6, TRL 0.29.1, PEFT 0.20.0. Both runs used micro-batch 4, gradient accumulation 2, eight epochs, seed 42, and the training and validation V2 splits. Values were read from the executed notebook outputs and the saved `trainer_state.json` files.

Validation Category+Severity Accuracy by epoch (200 records):

| Epoch | A (`'none'`) | B (`'short'`) |
| ---: | ---: | ---: |
| 1 | 77.5% | 86.0% |
| 2 | 83.0% | **89.0%** |
| 3 | 86.0% | 86.0% |
| 4 | 86.0% | 84.5% |
| 5 | **87.0%** | 85.5% |
| 6 | 86.5% | 86.0% |
| 7 | 86.0% | 85.5% |
| 8 | 86.5% | 85.0% |

| Measure | A | B |
| --- | ---: | ---: |
| Selected epoch (best joint accuracy) | 5 | 2 |
| Best validation joint accuracy | 87.0% | 89.0% |
| Validation severity accuracy, every epoch after the first | 100.0% | 100.0% |
| Valid JSON rate, every epoch | 100.0% | 100.0% |
| Lowest validation loss (epoch 1) | 0.0975 | 0.0786 |
| Training runtime, including per-epoch validation generation | 35.8 min | 112.7 min |
| Throughput | 3.72 samples/s | 1.18 samples/s |
| Validation-loss pass per epoch | 12.5 s | 43.6 s |

Observations:

* Validation errors are category errors only; severity is always correct because validation uses the same impact-sentence templates as training. Validation therefore cannot measure the severity generalization problem.
* B learns faster: 86.0% after one epoch against 77.5% for A. The best values differ by 2 points, which is 4 of 200 records and within sampling noise (standard error about 2.4 points); from epoch 3 the two variants are equivalent.
* Validation loss is lowest after epoch 1 and rises afterwards while training loss approaches zero, so eight epochs over-train both variants.
* B costs 3.1 times more training time. Because B peaks at epoch 2, a two-epoch B run would cost about 28 minutes, less than the full A run.
* No runtime was recorded for the earlier micro-batch 1 run, so the speed-up from the batch change is not measured.

### Demo 07 held-out evaluation (2026-10-25)

Same environment as Demo 06; 200 test records; deterministic generation. Values were read from the executed copies `demo07_run_short.ipynb` and `demo07_run_none.ipynb`.

| Metric | Base, no prompt | Base, prompt B | Adapter A, no prompt | Adapter B, prompt B |
| --- | ---: | ---: | ---: | ---: |
| Category Accuracy | 0.0% | 68.5% | 90.5% | 90.0% |
| Severity Accuracy | 0.0% | 40.0% | 57.0% | 54.0% |
| Category+Severity Accuracy | 0.0% | 26.0% | 51.0% | 48.0% |
| Valid JSON Rate | 0.0% | 100.0% | 100.0% | 100.0% |

Reference points: the most frequent test severity (P3) covers 40.0% of the records and the most frequent category 10.0%; the standard error of a single accuracy near 50% on 200 records is about 3.5 points.

Observations:

* Without a system prompt the base model never produces valid JSON, so the earlier fine-tuning delta (+51.0 joint points) measured mostly the output format. Against the base model with prompt B, fine-tuning adds 22.0 joint points, 21.5 category points, and 14.0 severity points.
* With prompt B, the base model already returns valid JSON for every record and 68.5% correct categories without training.
* After fine-tuning, A and B are equivalent on test: the 3-point joint difference is 6 of 200 records and within sampling noise. Prompt B does not improve the fine-tuned model; its benefits are a fair baseline and faster learning, at 3.1 times the training cost.
* Category generalizes: test category accuracy (90%) matches validation (85-89%).
* Severity does not generalize: it drops from 100% on validation to 54-57% on test, only 14 to 17 points above always answering P3. Prompt B contains no severity rules and the training data teaches the impact-sentence shortcut, so this result supports changing the training data (and testing the severity rules of variant C) rather than the prompt alone.

### Conclusion

Variant B is not adopted as a replacement for A on accuracy grounds. Keep the B base-model baseline in Demo 07 as the fair reference. Next steps: measure the base model zero-shot with variant C to test whether the guide's severity rules help, add training tickets whose impact is implicit, and reduce the number of epochs.
