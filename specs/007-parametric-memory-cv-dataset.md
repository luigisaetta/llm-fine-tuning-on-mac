# Parametric-Memory CV Dataset Specification

## Problem

The purpose of fine-tuning is to teach the model factual knowledge about Luigi Saetta at Oracle as part of its adapted parameters. The resulting model must answer direct questions about Luigi Saetta without being given a CV, retrieved document, or CV-specific system instruction at inference time.

## Scope

This specification overrides the question and system-prompt contract of specification 005 for the manually curated datasets. Facts remain grounded in the CV PDF during dataset creation, but the CV is provenance only and is never represented as runtime context.

## Dataset contract

* Every user question must explicitly identify the subject as `Luigi Saetta` or refer unambiguously to him in a multi-turn example. The initial dataset uses single-turn questions and names him directly.
* Questions must ask natural factual questions such as “What is Luigi Saetta's role at Oracle?” rather than ask what a CV says.
* The system message must be neutral: `You are a helpful assistant.` It must not mention a CV, documents, retrieval, sources, or unsupported-answer behaviour.
* Assistant answers must state the approved fact as a self-contained fact about Luigi Saetta. They must not use phrases such as “According to the CV” or “the candidate”.
* Facts must remain supported by the source CV, with direct contact details and privacy-consent text excluded.
* Train and evaluation splits must remain disjoint at fact level. Different phrasings of the same fact must never cross splits.

## Rationale

Mentioning the CV in every system prompt turns the dataset into a document-grounded question-answering task and teaches a dependency on context that will not exist at deployment. Neutral prompts and entity-specific questions instead align the supervised signal with the intended parametric-memory behaviour.

## Acceptance criteria

* No generated record contains the strings `CV`, `candidate`, `document`, or `According to the CV` in its system, user, or assistant content.
* Every user question contains `Luigi Saetta`.
* Every assistant response names `Luigi Saetta` or uses a self-contained sentence whose subject is clearly Luigi Saetta.
* The final private dataset contains 100–150 train and 30–50 evaluation records when the curated fact inventory supports that volume.
* The existing privacy and fact-split checks from specifications 005 and 006 still pass.

## Verification

* Parse every JSONL record and assert the system prompt, subject name, forbidden source-context terms, and split-disjoint fact IDs.
* Manually review a sample of questions and answers to confirm that they are useful without access to the CV.
