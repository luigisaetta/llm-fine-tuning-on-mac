# Future Dataset Improvement Work

This document records the remaining improvement area for the completed fine-tuning workflow: broader, more diverse, and carefully reviewed training and evaluation datasets. These are proposals, not implemented behaviour or performance claims.

## Current observation

The Qwen3-1.7B LoRA adapter may recall curated facts well when questions are close to the training phrasing. It can generalise poorly to substantially different, natural questions and add unsupported details. This is a known risk for a model trained on a compact factual dataset; it does not indicate that the adapter failed to load.

## Increase question diversity

Keep the approved facts and answers unchanged, but generate more genuinely different English questions for each fact. Include direct questions about dates, degrees, roles, locations, projects, certifications, responsibilities, and skills.

For example, the academic fact should support questions such as:

* When did Luigi Saetta graduate?
* Which degree did Luigi Saetta earn?
* Which university did Luigi Saetta attend?
* What is Luigi Saetta's academic background?

Evaluation must use previously unseen natural paraphrases and must measure both factual correctness and unsupported additions.

## Practical boundary

Fine-tuning is useful here as a learning exercise in parametric memory and LoRA behaviour. For factual applications that require consistently grounded answers, retrieval-augmented generation or a structured fact lookup remains the more reliable design.
