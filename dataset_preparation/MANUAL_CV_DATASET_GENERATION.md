# Generate CV Fine-Tuning Datasets with ChatGPT

This guide describes a reviewable, manual workflow for creating private training and evaluation datasets from a CV PDF. It is designed for a future LoRA/PEFT supervised fine-tuning run using the conversational `messages` format accepted by TRL's `SFTTrainer`.

The CV is used only to curate verified facts. It is **not** runtime context: the final model should answer direct questions about the named person without being told that a CV exists.

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

Use evaluation to measure recall, not unsupported factual generalisation. Put every approved fact in training, then create evaluation questions with new phrasings for a representative subset of those same facts. No exact question string may occur in both splits.

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
- Put every approved fact_id in the training split.
- Create 2–4 distinct, natural training questions for every approved fact. Do not repeat questions.
- Create 30–50 evaluation records for a representative subset of those same fact IDs. Each evaluation question must use a new natural phrasing that does not occur in training.
- The evaluation split is named `eval_recall`: it measures whether the fine-tuned model recalls trained facts when the question wording changes. It is not an unseen-fact benchmark.
- Every user question must explicitly name the person (for example, “What is Luigi Saetta's role at Oracle?”). Do not ask what a CV or document says.
- Use this exact neutral system message: `You are a helpful assistant.`
- Every assistant answer must be a self-contained sentence about the named person. Do not say “according to the CV”, “the candidate”, or “the document”. Do not add interpretation, reasoning, or new information.
- Do not include direct contact details, privacy-consent text, or any fact not in the approved list.

Use exactly this JSONL schema, one JSON object per line:
{"id":"train-F001-q1","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is Luigi Saetta's role at Oracle?"},{"role":"assistant","content":"Luigi Saetta is a Team Member in Oracle's EMEA Data Science, ML & AI Team."}],"metadata":{"fact_id":"F001","source_page":1,"split":"train"}}

Return exactly three fenced code blocks and no prose:
1. `train.jsonl`
2. `eval.jsonl`
3. `dataset_manifest.json`

The manifest must include: train_examples, eval_examples, train_fact_ids, eval_fact_ids, evaluation_protocol, and question_overlap_count. Set `evaluation_protocol` to `recall_paraphrase`; `eval_fact_ids` must be a subset of `train_fact_ids`; and `question_overlap_count` must be `0`.
```

If ChatGPT stops before completing a large JSONL block, ask it to continue from the next missing record ID without changing or repeating prior records.

## Validation checklist

Before training, confirm all of the following:

* Every line in both `.jsonl` files is valid JSON.
* Each record has exactly three messages with roles `system`, `user`, and `assistant` in that order.
* Every evaluation `fact_id` is also present in training, and no user-question string is shared between the splits.
* The assistant answer is supported by the fact associated with `metadata.fact_id`.
* The question names the person directly and no message mentions the CV, a document, or a candidate.
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
