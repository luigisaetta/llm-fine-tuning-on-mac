# Future Improvement Experiments

This document records hypotheses for improving the factual-recall fine-tuning demo. They are proposals, not implemented behaviour or performance claims.

## Current observation

The Qwen3-1.7B LoRA adapter may recall curated facts well when questions are close to the training phrasing. It can generalise poorly to substantially different, natural questions and add unsupported details. This is a known risk for a model trained on a compact factual dataset; it does not indicate that the adapter failed to load.

## Hypothesis 1: Increase question diversity

Keep the approved facts and answers unchanged, but generate more genuinely different English questions for each fact. Include direct questions about dates, degrees, roles, locations, projects, certifications, responsibilities, and skills.

For example, the academic fact should support questions such as:

* When did Luigi Saetta graduate?
* Which degree did Luigi Saetta earn?
* Which university did Luigi Saetta attend?
* What is Luigi Saetta's academic background?

Evaluation must use previously unseen natural paraphrases and must measure both factual correctness and unsupported additions.

## Hypothesis 2: Improve the Qwen3-1.7B training configuration

Keep the dataset and evaluation questions constant while testing carefully documented changes to the LoRA configuration and training procedure for [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). Use the post-trained conversational model, not the Base variant, so that the existing chat-template workflow remains comparable.

Suggested initial controls:

* Preserve the LoRA rank, alpha, dropout, learning rate, seed, and epoch count initially.
* Consider reducing maximum sequence length from 512 to 256 because the factual Q&A records are short.
* Compare the same targeted questions, including deliberately reworded and temporal questions.

This experiment isolates the effect of the documented training change. It cannot guarantee factual reliability or eliminate hallucinations.

## Optional later comparison: Qwen3-4B

[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) is a possible later comparison only after confirming available unified memory and acceptable training time. It is a substantially larger experiment and should not be the first capacity comparison on a MacBook.

## Practical boundary

Fine-tuning is useful here as a learning exercise in parametric memory and LoRA behaviour. For factual applications that require consistently grounded answers, retrieval-augmented generation or a structured fact lookup remains the more reliable design.
