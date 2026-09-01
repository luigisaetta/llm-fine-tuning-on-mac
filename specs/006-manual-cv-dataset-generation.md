# Manual CV Dataset Generation with ChatGPT Specification

## Problem

Users may prefer a capable document-analysis model such as ChatGPT over the local 0.6B model when transforming a CV PDF into reviewed fine-tuning data.

## Scope

Provide an English Markdown guide with a two-phase ChatGPT workflow: first extract and review atomic facts, then generate training and recall-evaluation JSONL datasets. The guide supplies copy-ready prompts, a privacy warning, output format, target sizes, and validation criteria.

## Requirements

* The process must not present generated facts as automatically trustworthy; user review is required before data generation.
* The guide must request only facts explicitly supported by the PDF and exclude direct contact details and privacy-consent text.
* Training must include every approved fact. Evaluation must reuse a subset of training fact IDs with distinct, unseen question phrasings so it measures recall rather than unsupported unseen-fact generalisation.
* The requested outputs must use conversational `messages` JSONL compatible with TRL SFT training.
* The prompts must teach direct factual knowledge about the named person, without using a CV-specific system prompt or treating the CV as inference-time context.
* The target output ranges are 100–150 training examples and 30–50 evaluation examples when the CV supports enough facts.
* The guide must direct private files to ignored `artifacts/` paths.

## Verification

* Review both prompts against the privacy, fact-grounding, split, and JSONL requirements.
* Confirm the Markdown guide is linked from the data-preparation documentation.
* Validate a generated dataset according to the guide's checklist before training.
