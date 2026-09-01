# Project Foundation Specification

## Problem

Provide a small, approachable project that teaches fine-tuning an open-weight Qwen 3 language model locally on a MacBook using parameter-efficient LoRA adapters.

## Scope

The project will provide a documented Python environment, reusable Python scripts, and Jupyter notebooks covering dataset preparation, LoRA fine-tuning, evaluation, and adapter-based inference. Models are obtained from the Hugging Face Hub.

## Out of scope

Cloud training, distributed training, full-parameter fine-tuning, hosted inference, private datasets, and committing model weights or checkpoints are out of scope unless a later specification adds them.

## Assumptions

* Development uses the existing `llm-fine-tuning-on-mac` Conda environment.
* The user has a MacBook; Apple Silicon/MPS is preferred when available, with CPU fallback made explicit.
* A small Qwen 3 open-weight model will be selected and documented in a later model-and-training specification.
* Device selection follows the Apple Silicon MPS execution contract in [specification 002](002-apple-silicon-mps-execution.md).

## Acceptance criteria

* `AGENTS.md` defines the project boundaries and mandatory spec-driven workflow.
* The README explains the intended learning path, setup, dependencies, reproducibility, and project layout in English.
* `requirements.txt` includes the notebook, Hugging Face, PyTorch, and LoRA dependencies needed for the first implementation.
* Dependencies install into the named Conda environment.

## Verification

* Install `requirements.txt` with the named Conda environment.
* Import the core packages and print their versions.
* Review the README and agent instructions against this specification.
* Review MPS device requirements against specification 002 before implementing training code.
