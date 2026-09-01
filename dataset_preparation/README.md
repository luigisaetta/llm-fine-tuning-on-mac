# CV-Grounded Dataset Preparation

`build_cv_qa_datasets.py` creates private train and evaluation datasets from a text-based CV PDF. It runs entirely on the MacBook: `pypdf` extracts the text and the local Qwen3-0.6B model extracts source-verifiable facts and generates questions.

## Run

From the repository root, with the CV stored under the ignored `pdf/` directory:

```bash
conda activate llm-fine-tuning-on-mac
python dataset_preparation/build_cv_qa_datasets.py \
  --pdf-path pdf/Luigi\ Saetta-CV-2024.pdf \
  --output-dir artifacts/datasets/cv-qa \
  --device auto
```

The command writes these ignored local artifacts:

```text
artifacts/datasets/cv-qa/
├── train.jsonl
├── eval.jsonl
└── dataset_manifest.json
```

The default configuration requests up to seven Qwen-generated questions per verified fact, holds out 25% of facts for evaluation, and caps the outputs at 150 training and 50 evaluation examples. If the small local model returns fewer questions than requested, the tool completes the set with grounded phrasing variants that reference only the same verified fact. It never invents facts or answers.

Review the resulting data before training. The default privacy policy excludes direct email addresses, phone numbers, URLs, and similar contact details. Do not pass `--include-contact-details` unless that inclusion is intentional and appropriate for the planned training use.

The JSONL records use TRL's conversational `messages` format, ready for a future `SFTTrainer` and PEFT/LoRA workflow. See [specification 005](../specs/005-cv-grounded-qa-datasets.md) for the complete contract.

## Manual ChatGPT workflow

For a reviewed workflow that uses ChatGPT to extract facts from a PDF and generate the two JSONL splits, see [Generate CV Fine-Tuning Datasets with ChatGPT](MANUAL_CV_DATASET_GENERATION.md). It includes two copy-ready prompts and a privacy-aware validation checklist.
