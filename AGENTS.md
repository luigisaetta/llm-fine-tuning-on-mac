# AGENTS.md

This repository contains a practical, Mac-first learning project for fine-tuning a small open-weight Qwen 3 language model with LoRA. It will use models downloaded from the Hugging Face Hub, Python scripts, and Jupyter notebooks.

## Repository purpose

Keep the repository focused on a clear, reproducible fine-tuning workflow that runs locally on Apple Silicon where the selected model and hardware permit it. The project should help a developer understand the full path from dataset preparation to training, evaluation, and local inference.

Avoid unrelated AI demonstrations, cloud deployment code, proprietary model APIs, or large frameworks unless explicitly required by a specification.

## Spec-driven workflow

Work spec-first for every feature or meaningful change:

1. Read the relevant specification in `specs/` before implementation.
2. Create or update a concise specification when the requested behaviour is not covered.
3. State the problem, scope, assumptions, acceptance criteria, and verification approach before writing code.
4. Implement only what the specification requires, then update the specification if an agreed behaviour changes.
5. Record important implementation decisions in the specification or its linked documentation.

Specifications are source-controlled project artifacts, not disposable planning notes. Keep them concise, testable, and in English.

## Language and documentation

* All documentation, source-code comments, Markdown files, notebook narrative cells, and user-facing output must be written in English.
* Keep documentation practical, accurate, and aligned with the implementation.
* Make the exact model identifier, revision (where pinned), dataset provenance, training configuration, hardware assumptions, and expected outputs explicit.
* Do not claim model capabilities, benchmark quality, licence terms, memory requirements, or Apple Silicon support unless verified by implementation or authoritative Hugging Face/Qwen documentation.
* Do not commit private datasets, personal data, model checkpoints, Hugging Face tokens, or generated caches.

## Codex working rules

When working in this repository, Codex should:

* Inspect the existing project structure and applicable specifications before editing.
* Prefer small, coherent changes over broad rewrites.
* Preserve user changes already present in the working tree.
* Do not create commits unless explicitly asked.
* Do not add dependencies without a clear training, evaluation, or notebook need.
* Do not run destructive commands or discard existing changes unless explicitly requested.
* Never hard-code credentials, private URLs, local machine paths, or personally identifiable data.
* Use environment variables and ignored `.env` files for secrets; document variable names and safe placeholders.
* Make network downloads explicit and avoid silently uploading local data.
* When uncertain, document the assumption, leave a clear TODO, or ask for clarification.

## Python environment

Use the `llm-fine-tuning-on-mac` Conda environment for development, notebooks, and checks. Activate it before running project commands and do not rely on globally installed packages.

Install Python packages from `requirements.txt`. Keep dependencies compatible with current Apple Silicon PyTorch builds and document any version constraint required by the selected Qwen 3 model.

## Python and notebook conventions

Every Python source file must start with this multiline header, using the actual modification date:

```python
"""
Author: L. Saetta
Date last modified: YYYY-MM-DD
License: MIT
Description: Brief description of the responsibilities and functions contained in this file.
"""
```

Use accurate Google-style docstrings for modules, classes, and public functions. Keep notebooks focused on one learning goal, restartable from a clean kernel, and backed by reusable code in `src/` when logic grows beyond a short demonstration.

## Fine-tuning design expectations

* Prefer a small Qwen 3 open-weight instruct model appropriate for the documented Mac hardware.
* Separate configuration, dataset loading/validation, prompt formatting, tokenization, training, evaluation, checkpoint handling, and inference where practical.
* Use PEFT/LoRA adapters; do not modify base model weights unless a specification explicitly requires it.
* Make seed, maximum sequence length, batch size, gradient accumulation, learning rate, LoRA target modules, precision, device selection, and checkpoint policy visible in configuration.
* Detect and clearly report unavailable or unsuitable MPS/CPU capabilities instead of silently selecting an unexpected device.
* Validate dataset schema, empty samples, overly long samples, and train/evaluation splits with actionable errors.
* Save adapters and training metadata separately from the downloaded base model. Do not commit these generated artifacts.

## Testing and verification

New behaviour needs proportionate automated tests with `pytest`. Unit tests must not download models or datasets; mock or use tiny local fixtures. Real model runs should be optional and clearly marked as integration checks.

Run relevant formatting, linting, and tests before considering work complete. If a check cannot run because of missing model artifacts, hardware, or dependencies, state it clearly.

## Definition of done

A change is done only when:

* Its scope and acceptance criteria are captured in a specification.
* Documentation and setup instructions reflect the changed behaviour.
* Dataset, model, device, and configuration assumptions are explicit.
* Relevant checks have been run or their limitation has been documented.
* No credentials, private data, model artifacts, or caches have been added to version control.
