# Future Improvement Experiments

This document records hypotheses for improving the factual-recall fine-tuning demo. They are proposals, not implemented behaviour or performance claims.

## Current observation

The Qwen3-0.6B LoRA adapter recalls curated facts well when questions are close to the training phrasing. It generalises poorly to substantially different, natural questions and can add unsupported details. This is expected from a very small model trained on a compact factual dataset; it does not indicate that the adapter failed to load.

## Hypothesis 1: Increase question diversity

Keep the approved facts and answers unchanged, but generate more genuinely different English questions for each fact. Include direct questions about dates, degrees, roles, locations, projects, certifications, responsibilities, and skills.

For example, the academic fact should support questions such as:

* When did Luigi Saetta graduate?
* Which degree did Luigi Saetta earn?
* Which university did Luigi Saetta attend?
* What is Luigi Saetta's academic background?

Evaluation must use previously unseen natural paraphrases and must measure both factual correctness and unsupported additions.

## Hypothesis 2: Compare a moderately larger Qwen3 model

Keep the dataset, LoRA configuration, training procedure, and evaluation questions constant, then compare the current 0.6B run with [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B).

Qwen3-1.7B is the preferred next comparison because it stays in the same model family and offers a moderate capacity increase without immediately moving to the substantially heavier 4B model. Use the post-trained conversational model, not the Base variant, so that the existing chat-template workflow remains comparable.

Suggested initial controls:

* Use the same train/evaluation datasets as the 0.6B comparison.
* Preserve the LoRA rank, alpha, dropout, learning rate, seed, and epoch count initially.
* Consider reducing maximum sequence length from 512 to 256 because the factual Q&A records are short.
* Compare the same targeted questions, including deliberately reworded and temporal questions.

This experiment isolates the effect of model capacity. A larger model may improve semantic generalisation, but it cannot guarantee factual reliability or eliminate hallucinations.

## Optional later comparison: Qwen3-4B

[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) is a possible later comparison only after confirming available unified memory and acceptable training time. It is a substantially larger experiment and should not be the first capacity comparison on a MacBook.

## Practical boundary

Fine-tuning is useful here as a learning exercise in parametric memory and LoRA behaviour. For factual applications that require consistently grounded answers, retrieval-augmented generation or a structured fact lookup remains the more reliable design.
