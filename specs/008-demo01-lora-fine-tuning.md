# Demo 01: LoRA Fine-Tuning of Qwen3-0.6B

## Problem

The project needs a reproducible, Mac-first notebook that fine-tunes the local Qwen3-0.6B model on the private parametric-memory datasets about Luigi Saetta and evaluates the result without treating the CV as inference-time context.

## Scope

Create `demo01/` containing a restartable Jupyter notebook and concise usage documentation. The notebook loads the local base model and ignored datasets, applies a PEFT LoRA adapter, trains with TRL SFT, evaluates loss at every epoch, measures held-out generative answer accuracy after training, and saves only adapter artifacts.

## Configuration baseline

| Setting | Value | Rationale |
| --- | --- | --- |
| Base model | Local `Qwen3-0.6B` | Selected project base model. |
| LoRA rank | 8 | Small adapter suitable for a compact factual dataset. |
| LoRA alpha | 16 | Standard scaling of 2 × rank. |
| LoRA dropout | 0.05 | Mild regularisation for a small dataset. |
| Target modules | Q/K/V/O attention and gate/up/down MLP projections | Adapts attention and feed-forward transformations. |
| Learning rate | 2e-4 | Conservative PEFT SFT starting point. |
| Epochs | 3 | Requested compact baseline. |
| Micro-batch size | 1 | Reduces Apple unified-memory pressure. |
| Gradient accumulation | 8 | Gives an effective batch size of 8. |
| Maximum sequence length | 512 | More than sufficient for short factual Q&A while bounding memory. |
| Precision | float32 | Conservative MPS baseline. |

## Training and evaluation contract

* The notebook loads `train.jsonl` and `eval.jsonl` with `datasets` and converts each conversational record into TRL prompt/completion format.
* Training uses a neutral system prompt and questions naming Luigi Saetta, as required by specification 007.
* `eval_strategy="epoch"` and `save_strategy="epoch"` are mandatory. The trainer records `eval_loss` after every epoch and restores the adapter with the lowest evaluation loss.
* Final response accuracy is measured on the evaluation split by deterministic generation (`do_sample=False`). A response is correct when token-level F1 against the approved answer is at least 0.80; exact-match rate and mean token F1 are also reported.
* The generative accuracy metric is complementary to `eval_loss`, not a replacement for it.

## Privacy and artifacts

* The notebook reads the ignored local datasets and must never upload them.
* Training checkpoints, logs, and the LoRA adapter are saved only under `artifacts/training/`, which is ignored by Git.
* The notebook contains no stored outputs, dataset rows, credentials, or CV text.

## Acceptance criteria

* The notebook validates local model and dataset paths before loading.
* LoRA, learning rate, epochs, batch settings, and device policy appear together in one editable configuration cell.
* Evaluation loss runs once per epoch.
* The held-out generative metric uses only evaluation records and reports exact match, mean token F1, and threshold accuracy.
* The saved output is a PEFT adapter rather than a duplicate base model.

## Verification

* Validate the notebook JSON and ensure it has no stored outputs.
* Import its dependencies using the project Conda environment.
* Before a full run, inspect selected device, trainable-parameter count, and dataset sizes.
* After a full run, inspect the epoch-level `eval_loss` table and manually review generated evaluation samples.
