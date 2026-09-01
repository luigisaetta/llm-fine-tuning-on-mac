# Generate CV Fine-Tuning Datasets with ChatGPT

This guide describes a reviewable, manual workflow for creating private training and evaluation datasets from a CV PDF. It is designed for a future LoRA/PEFT supervised fine-tuning run using the conversational `messages` format accepted by TRL's `SFTTrainer`.

The workflow deliberately separates fact extraction from Q&A generation. Review the extracted facts before asking for datasets: this is the most effective protection against invented or outdated CV details.

## Privacy first

The CV contains personal information. Upload it only to a ChatGPT account and workspace whose data settings and retention policy you have reviewed. If you do not want contact details in the training data, redact email addresses, phone numbers, postal addresses, personal URLs, and consent statements before upload.

ChatGPT supports uploaded documents, including PDFs, for extraction and transformation tasks. On non-Enterprise plans, PDF handling is text-based, so scanned PDFs should be OCRed locally before upload. See the official [File Uploads FAQ](https://help.openai.com/en/articles/8555545-file-uploads-with-gpts-and-advanced-data-analysis-in-chatgpt) for current availability, limits, and retention information.

Never commit the original PDF, the conversation export, or generated datasets containing private information. Store the final files under the ignored directory `artifacts/datasets/`.

## Recommended workflow

1. Start a new ChatGPT conversation and upload the CV PDF.
2. Send Prompt 1. Review every fact in the returned JSON before continuing. Correct or remove any inaccurate item.
3. Send Prompt 2 in the same conversation, using the approved fact list from Prompt 1.
4. Save the returned `train.jsonl` and `eval.jsonl` files under `artifacts/datasets/<dataset-name>/`.
5. Validate the JSONL and manually inspect at least 10 examples from each split before fine-tuning.

Keep the evaluation facts completely separate from training facts. It is acceptable to create several questions for the same fact within one split; it is not acceptable to place questions based on the same fact in both splits.

## Prompt 1 — extract and review atomic facts

Upload the CV PDF, then copy this prompt:

```text
You are a careful dataset curator. Extract atomic, professionally relevant facts from the uploaded CV PDF.

Rules:
- Use only information explicitly stated in the PDF. Do not infer, paraphrase into a stronger claim, combine facts, or use outside knowledge.
- Exclude all direct contact details and personal identifiers: email addresses, phone numbers, street addresses, personal URLs, social-media handles, date of birth, signatures, and privacy-consent text.
- Keep work experience, roles, responsibilities, achievements, projects, skills, education, certifications, languages, and presentations when explicitly stated.
- Each fact must be self-contained, specific, and written in concise English.
- Include the source page number for every fact.
- Deduplicate repeated facts.

Return JSON only, in this schema:
{
  "facts": [
    {
      "fact_id": "F001",
      "source_page": 1,
      "fact": "A concise fact grounded in the CV"
    }
  ]
}

Aim for 40–60 facts if the CV supports them. Stop after the JSON. Do not generate questions or datasets yet.
```

Review the result. If a fact is not fully supported by the CV, remove it. If needed, ask ChatGPT: “Regenerate the fact list using only verbatim-supported facts and preserve the JSON schema.”

## Prompt 2 — create train and evaluation JSONL

After the fact list is approved, copy this prompt in the same conversation:

```text
Using only the approved JSON fact list from the previous message, create two supervised fine-tuning datasets: one training split and one evaluation split.

Split rules:
- Split by fact, not by question: a fact_id may appear in exactly one split.
- Assign approximately 75% of fact IDs to train and 25% to eval.
- Create 100–150 train examples and 30–50 eval examples when the fact count permits.
- Generate 2–4 distinct, natural questions for each assigned fact. Do not repeat questions.
- Every assistant answer must contain only the corresponding approved fact. Do not add interpretation, reasoning, or new information.
- Do not include direct contact details, privacy-consent text, or any fact not in the approved list.

Use exactly this JSONL schema, one JSON object per line:
{"id":"train-F001-q1","messages":[{"role":"system","content":"You are a professional CV assistant. Answer only with information explicitly present in the candidate's CV. Do not infer or add details."},{"role":"user","content":"A factual question about the CV"},{"role":"assistant","content":"The approved fact"}],"metadata":{"fact_id":"F001","source_page":1,"split":"train"}}

Return exactly three fenced code blocks and no prose:
1. `train.jsonl`
2. `eval.jsonl`
3. `dataset_manifest.json`

The manifest must include: train_examples, eval_examples, train_fact_ids, eval_fact_ids, and split_overlap_fact_ids. The overlap list must be empty.
```

If ChatGPT stops before completing a large JSONL block, ask it to continue from the next missing record ID without changing or repeating prior records.

## Validation checklist

Before training, confirm all of the following:

* Every line in both `.jsonl` files is valid JSON.
* Each record has exactly three messages with roles `system`, `user`, and `assistant` in that order.
* `split_overlap_fact_ids` is an empty list.
* The assistant answer is supported by the fact associated with `metadata.fact_id`.
* No contact details or sensitive personal data remain.
* The data contains at least 100 training and 30 evaluation records, or the shortfall is documented and accepted.

## Suggested local destination

```text
artifacts/datasets/cv-qa/
├── train.jsonl
├── eval.jsonl
└── dataset_manifest.json
```

`artifacts/` is excluded from version control by this repository's `.gitignore`.
