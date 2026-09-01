# CV-Grounded Train and Evaluation Dataset Specification

## Problem

The project needs private, task-specific supervised fine-tuning data derived from a CV PDF. The data must teach the model to answer factual questions about the candidate without inventing information or leaking the same underlying fact between training and evaluation.

## Scope

Create a local Python tool in `dataset_preparation/` that accepts a text-based CV PDF and produces `train.jsonl` and `eval.jsonl` in conversational `messages` format. It uses the locally downloaded Qwen3-0.6B model to extract atomic professional facts and generate questions. It never calls a hosted model or uploads the PDF.

## Input and output contract

| Item | Contract |
| --- | --- |
| Input | A readable, text-based CV PDF supplied through `--pdf-path`. |
| Model | The local `artifacts/models/Qwen3-0.6B` directory. |
| Train output | `<output-dir>/train.jsonl`, capped at 150 examples by default. |
| Evaluation output | `<output-dir>/eval.jsonl`, capped at 50 examples by default. |
| Audit output | `<output-dir>/dataset_manifest.json` with counts, configuration, source filename, and fact IDs only. |
| Dataset record | A JSON object containing `id`, `messages`, and `metadata`. |

Each `messages` value has a system message, a user question, and an assistant answer. This conversational format is accepted by TRL's `SFTTrainer` and can be loaded by `datasets` for PEFT/LoRA training.

## Fact and question generation

* The script extracts text page by page with `pypdf`.
* Qwen3-0.6B receives each page locally and returns a JSON list of atomic professional facts.
* A fact is accepted only when its normalised text occurs in the extracted source page. Facts containing direct contact information are excluded by default.
* Facts are deduplicated and split before question generation using a seeded, fact-level split. No fact ID may occur in both outputs.
* Qwen3-0.6B generates questions in controlled batches. When the small local model returns fewer than the requested number, the tool completes the set with deterministic grounded phrasing variants that reference only the same verified fact. The assistant answer is the corresponding verified fact, not an unverified model paraphrase.
* The default split is 75% train / 25% evaluation, aiming for approximately 100–150 training and 30–50 evaluation examples when the CV yields about 20–50 usable facts. The tool warns rather than fabricating examples when it cannot reach the target.

## Privacy and safety

* The CV and generated datasets are private local artifacts and must remain ignored by Git.
* Direct contact details such as email addresses, telephone numbers, street addresses, and personal URLs are excluded by default.
* The tool must fail clearly for a scanned or otherwise non-text-extractable PDF; OCR is an explicit future step, not a hidden upload or transformation.
* The generated facts and questions must be reviewed by the CV owner before using them for training.

## Device and reproducibility

* Device selection follows specification 002: `auto` selects MPS when available and visibly falls back to CPU; explicitly requested unavailable MPS is an error.
* The tool records the model directory, PyTorch version, selected device, seed, and output counts in the manifest.
* The model uses `local_files_only=True` and must not download artifacts.

## Acceptance criteria

* `pypdf` is a direct dependency and the script offers a documented command-line interface.
* Given the supplied CV, the script produces syntactically valid non-empty `train.jsonl` and `eval.jsonl` files whenever sufficient verified facts are available.
* Each record has valid conversational roles and a non-empty question and answer.
* Train and evaluation fact IDs are disjoint.
* No output record contains a direct contact detail when the default privacy settings are used.
* Unit tests cover parsing, fact validation, contact-detail exclusion, deterministic splitting, and record validation without loading a PDF or model.

## Verification

* Run the unit tests without model downloads.
* Run the command against the supplied CV as an optional local integration check.
* Parse every generated JSONL line and inspect the manifest's split-overlap field.
* Manually review a sample of facts and questions before training.

## References

* [pypdf text extraction documentation](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)
* [TRL SFTTrainer dataset formats](https://huggingface.co/docs/trl/sft_trainer)
