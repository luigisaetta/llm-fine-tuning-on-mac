# ChatGPT-Assisted Parametric-Memory Dataset Specification

## Problem

The purpose of fine-tuning is to teach the model factual knowledge about Luigi Saetta at Oracle as part of its adapted parameters. The resulting model must answer direct questions about Luigi Saetta without being given a CV, retrieved document, or CV-specific system instruction at inference time.

## Scope

The supported workflow is documented in `dataset_preparation/MANUAL_CV_DATASET_GENERATION.md`. A capable document assistant such as ChatGPT extracts atomic facts from a user-uploaded CV PDF; the user reviews and approves that fact list; the same assistant then creates the JSONL splits from approved facts only.

Facts remain grounded in the CV PDF during dataset creation, but the CV is provenance only and is never represented as runtime context. Local model-driven fact extraction and fact-disjoint evaluation are out of scope for this project.

## Dataset contract

* Every user question must explicitly identify the subject as `Luigi Saetta` or refer unambiguously to him in a multi-turn example. The initial dataset uses single-turn questions and names him directly.
* Questions must ask natural factual questions such as “What is Luigi Saetta's role at Oracle?” rather than ask what a CV says.
* The system message must be neutral: `You are a helpful assistant.` It must not mention a CV, documents, retrieval, sources, or unsupported-answer behaviour.
* Assistant answers must state the approved fact as a self-contained fact about Luigi Saetta. They must not use phrases such as “According to the CV” or “the candidate”.
* Facts must remain supported by the source CV, with direct contact details and privacy-consent text excluded.
* The user must review and approve the extracted fact list before Q&A generation. Generated facts and answers are not automatically trustworthy.
* The training split must contain every approved fact selected for the fine-tuning run.
* Training questions for each fact must use meaningfully different forms, including direct wh-questions where applicable (for example, year, role, degree, location, project, or certification questions). Rewording only a fixed template is insufficient.
* The evaluation split is a **recall evaluation**: it must use facts that appear in training, but with previously unseen, natural question phrasings. An exact user-question string must never appear in both splits.
* A fact-disjoint, unseen-fact dataset may be created as a separate diagnostic transfer test, but it must not be used to select the training checkpoint or presented as the primary measure of factual-memory recall.

## Rationale

Mentioning the CV in every system prompt turns the dataset into a document-grounded question-answering task and teaches a dependency on context that will not exist at deployment. Neutral prompts and entity-specific questions instead align the supervised signal with the intended parametric-memory behaviour.

## Acceptance criteria

* No generated record contains the strings `CV`, `candidate`, `document`, or `According to the CV` in its system, user, or assistant content.
* Every user question contains `Luigi Saetta`.
* Every assistant response names `Luigi Saetta` or uses a self-contained sentence whose subject is clearly Luigi Saetta.
* The final private dataset contains 100–150 train and 30–50 evaluation records when the curated fact inventory supports that volume.
* The training questions for each fact are distinct and include a mix of factual question forms across the dataset.
* Every evaluation `fact_id` is present in the training split, while no evaluation user-question string is present in the training split.
* The private PDF, conversation export, JSONL datasets, and generated adapters remain outside version control.

## Verification

* Parse every JSONL record and assert the system prompt, subject name, forbidden source-context terms, evaluation-fact coverage by the training split, and disjoint question strings.
* Manually review a sample of questions and answers to confirm that they are useful without access to the CV.
* Follow the validation checklist in `dataset_preparation/MANUAL_CV_DATASET_GENERATION.md` before a training run.
