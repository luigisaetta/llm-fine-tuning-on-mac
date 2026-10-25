# Ticket Classification Guide Specification

## Problem

The ticket-classification datasets were generated from category and severity rules that are no longer available. Only the past classifications remain. The training and validation splits assign severity from an explicit impact sentence, while the test split expects severity to be inferred from technical context. Without a written policy, prompts, new data, and label audits have no common reference.

## Scope

* Write `ticket-classification/classification_guide.md`, a versioned guide retrofitted from the past classifications.
* Define every category, the boundaries between overlapping categories, the severity policy, and the summary style.
* Combine the two observed severity policies into one: explicit impact statements decide first; contextual signals decide only when the ticket does not state its impact.
* List the past labels that the guide does not explain, as input for a later label audit.

## Non-goals

* Changing any dataset, notebook, prompt, or training configuration.
* Deriving the model system prompt; it will be derived from this guide in a later specification.
* Human review of the guide, which the project owner has explicitly deferred.

## Assumptions

* The past labels are the best available evidence of the intended policy, but contain some noise.
* The test split has been inspected to write the contextual rules. It is therefore no longer an untouched final test set and is treated as a realistic development set; a new final test set is needed before final acceptance.

## Requirements

* The guide must state its version, date, sources, and review status.
* Each of the 10 categories must have a definition, typical signals, and boundary rules for known overlaps.
* The severity policy must define P1 to P4, the precedence between explicit and contextual signals, and specific rules for security events, non-production environments, degradation versus outage, and single critical resources.
* The guide must not contain real ticket data or personal data; examples must be paraphrased or synthetic.

## Acceptance criteria

* Applying the guide by hand to a sample of training, validation, and test records reproduces the past labels, apart from the cases listed as known inconsistencies.
* The guide can be used without access to the datasets.

## Verification

Performed on 2026-10-25:

* Rules were derived from `train_dataset_v2.jsonl` (1,000 records), `validation_dataset_v2.jsonl` (200), and `test_dataset.jsonl` (200), using keyword-by-category counts, severity-distinctive phrase counts, and a full read of the test split grouped by severity.
* Training and validation follow a factorial design: each problem description is paired with an impact sentence from about five templates per severity, so severity there depends only on the impact sentence.
* A manual sample check against the guide is recorded in the guide's known-inconsistencies section. No model was run and no automated audit was performed.
