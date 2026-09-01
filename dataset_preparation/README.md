# Dataset Preparation

The supported dataset workflow uses a capable document assistant such as ChatGPT to extract and review facts from a CV PDF, then create private LoRA/PEFT training and recall-evaluation JSONL files.

Follow [Generate CV Fine-Tuning Datasets with ChatGPT](MANUAL_CV_DATASET_GENERATION.md). It contains the complete two-phase workflow, copy-ready prompts, the required `messages` schema, and the validation checklist.

The final evaluation split contains unseen question phrasings for facts included in training. It measures recall of the fine-tuned factual knowledge; it is not an unseen-fact generalisation benchmark.
