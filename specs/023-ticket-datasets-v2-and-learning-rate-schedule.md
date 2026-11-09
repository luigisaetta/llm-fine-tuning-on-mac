# Ticket datasets from the AI DP project and a learning-rate schedule

## Problem

Specification 021 recorded a large gap between validation and test results for the ticket-classification adapters. Severity accuracy was 100% on validation from the second epoch, but only 54–57% on the test split. Joint accuracy was 48–51% on test. Two causes were found in the data:

* **A label shortcut.** The training and validation tickets reused 20 introductory sentences. Each sentence always came with the same severity, so the model could learn the sentence instead of the ticket.
* **A different test distribution.** Train and validation V2 are balanced, with 250/250/250/250 and 50/50/50/50 records per severity and tickets of about 167 characters on average. The test file comes from an earlier generation, with 20/55/80/45 records per severity (P1–P4) and tickets of about 93 characters.

The sibling project `llm-fine-tuning-on-aidp` fixed both:

* It rewrote and moved the 20 sentences in 1,200 inputs without changing any expected output.
* It pooled the 1,400 clean records and split them again, stratified by category and severity, into 980/210/210 records with the same distribution.

Its fine-tuned model reached 94.3% joint accuracy and 95.7% severity accuracy on its test split, against 44.3% joint accuracy for the base model.

The training configuration also used a constant peak learning rate with the Transformers default linear decay and no warmup. It ran 8 epochs, which specification 021 found to over-train.

## Scope

* Replace the ticket datasets in `ticket-classification/artifacts/datasets/` with the three gold files of the AI DP project, and remove the previous files. Only these three files remain:
  * `train_dataset_v2.jsonl`
  * `validation_dataset_v2.jsonl`
  * `test_dataset_v2.jsonl`
* Rename the held-out test file in the `llm_fine_tuning.tickets` package from `test_dataset.jsonl` to `test_dataset_v2.jsonl`.
* Add the following to `TrainingSettings`:
  * `lr_scheduler_type`, default `"cosine"`;
  * `warmup_ratio`, default `0.10`, which must be in `[0, 1)`.
  
  Pass both to `SFTConfig`, and set `data_seed` to the run seed.
* Add `learning_rate_schedule()`, which returns the optimizer-step plan, and `plot_learning_rate()`.
* In Demo 06:
  * show the schedule settings in the Settings cell;
  * print the step plan after the data is loaded;
  * plot the logged learning rate in the review section;
  * set the default number of epochs to 5;
  * set `prompt_variant="short"` (variant B), the prompt used by the AI DP project, so that the results can be compared with it. Demo 07's `ADAPTER_PROMPT_VARIANT` becomes `"short"` to match. The package default of `TrainingSettings` stays `'none'`.
* Update Demos 07 and 08, the tests, the README, the related specifications, and the changelog for the new file names, counts, and distributions.

## Non-goals

* The raw/bronze/silver pipeline of the AI DP project. Only its final gold files are imported.
* Making Demo 06 task-agnostic. This is the AI DP project's Spec 019, and it is postponed.
* Demo 09, the standalone Linux CUDA notebook. Its file names do not change and its training settings are inline. Only its record counts in the documentation change.
* The `ai-dp/` notebooks of this repository. They are superseded by the AI DP project and keep their current local changes.
* Changing the micro-batch. Demo 06 keeps 4 × 2, an effective batch of 8, which was chosen for this Mac in commit `5d6a724`. The AI DP project uses 1 × 8, with the same effective batch and the same 123 optimizer steps per epoch on 980 records.
* Running a training. The effect of the new data and schedule on accuracy is measured by the user.

## Dataset provenance

* **Source:** `llm-fine-tuning-on-aidp`, `data/gold/`, at commit `3728bb2945a88d215ea3e4fd1207b0332183cdb5` (2026-11-07), produced by its notebooks NB02 → NB05 (its Specs 003–009).
* **Copy:** byte for byte. The SHA-256 hashes are:

| File | Records | P1/P2/P3/P4 | SHA-256 |
| --- | ---: | --- | --- |
| `train_dataset_v2.jsonl` | 980 | 223/250/266/241 | `a6a7dec86731b72422f782f1aece2fbac9427286077bf1d0eba24ca329de1c85` |
| `validation_dataset_v2.jsonl` | 210 | 49/53/57/51 | `910b8be3c246dc817a7c2dc9342dff8eeeba81404b13af6393cc5eb35eb282c1` |
| `test_dataset_v2.jsonl` | 210 | 48/52/57/53 | `6f86b34c6560a340af51878f4d0ff6f1bb3af92fbf03064396533e3c404e5fb6` |

* **Format:** each record has the same structure as before: `user` and `assistant` messages, and an assistant JSON with `category`, `severity`, and `summary`. All 10 categories appear in every split, with 97–99 training records per category.
* **The previous files are removed:**
  * training and validation V1 and V2 (1,000 + 200 records each);
  * the test file (200 records).
  
  They remain in the Git history.
* **The previous test file must not be used with the new adapters.** 135 of its 200 tickets are in the new training split, and 31 are in the new test split (after Unicode, case, and whitespace normalisation).
* **Results recorded before this change are not comparable** with results on the new data: specifications 015 and 021, and the saved adapters under `artifacts/models/`.

## Learning-rate schedule

The learning rate rises linearly from 0 to `learning_rate` over `ceil(warmup_ratio × total_steps)` optimizer steps, then decays to 0 along a cosine curve over the rest of the run. The behaviour and the step formulas are the ones of the AI DP project's Spec 010. On the new training split, with 5 epochs and 4 × 2, the plan is 123 steps per epoch, 615 steps in total, and 62 warmup steps.

## Acceptance criteria

1. The datasets folder holds exactly the three files above, with the recorded hashes.
2. `TicketProjectPaths.test_file` is `test_dataset_v2.jsonl`. Demo 06 still opens only the training and validation files.
3. `TrainingSettings` rejects a `warmup_ratio` outside `[0, 1)`. `build_sft_config` passes the scheduler type, the warmup ratio, and `data_seed`.
4. `learning_rate_schedule()` agrees with `TrainingArguments.get_warmup_steps`, and the real cosine scheduler peaks at the end of the warmup.
5. Demo 06 uses the short prompt and 5 epochs, shows the schedule settings and the step plan, and plots the learning rate. Its saved outputs stay empty.
6. The documentation and the specifications state the new counts and the provenance, and mark the earlier results as referring to the earlier data.

## Verification approach

* Unit tests, without a model download: the settings check, the SFT configuration, the schedule plan against Transformers, the test file name, and the Demo 06 notebook contents.
* A local check of the dataset hashes and counts.
* Demo 06 executed up to the step plan, without training, if the local model is available.
