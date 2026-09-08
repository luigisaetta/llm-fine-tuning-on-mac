# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

* Added this project changelog for user-visible additions and significant changes.
* 2026-09-08: Added Demo 07 for held-out ticket-classification test evaluation of the saved LoRA adapter.
* Added synthetic ticket-classification training, validation, and held-out test datasets.
* Added Demo 06, BF16 MPS LoRA training of Qwen3-1.7B for structured IT-support ticket classification with per-epoch validation loss and a loss-trend chart.
* Added the ticket-classification training and evaluation specification with a local-artifact usage guide.
* Added a visual workflow overview to the root README.
* Added Demo 05, an MPS-only direct comparison of Qwen3-1.7B base-model and Demo 01 LoRA-adapter validation performance using the same generative metrics.
* Added a Demo 01 loss-trend chart with distinct training and validation loss series and a grid.
* Added an OCI Python SDK script to upload complete merged-model directories to Object Storage with multipart uploads, progress reporting, dry-run planning, and explicit overwrite control.
* Added Demo 03, a LangChain `ChatOCIGenAI` client for one-prompt inference against an OCI Generative AI Dedicated AI Cluster endpoint.
* Added Demo 04, a Linux/CUDA-specific BF16 LoRA fine-tuning notebook for Qwen3-1.7B, with configurable local artifact directories and dedicated setup instructions.

### Changed

* 2026-09-08: Updated Demo 06 checkpoint selection to maximize validation category-and-severity accuracy.
* Updated Demo 06 to use version 2 of the synthetic ticket-classification training and validation datasets.
* Added batched, cached per-epoch validation generation metrics to Demo 06 without changing its training configuration.
* Reformatted the README dependency overview as a library-to-contribution table.
* Updated Demo 01 LoRA fine-tuning to load the Qwen3 base model and trainable adapter parameters in BF16, enable BF16 training and evaluation, and require compatible MPS and macOS 14 or later.
* Updated the standalone LoRA merge workflow to save BF16 model weights, matching the Qwen3 base model and Demo 01 adapter training.
* Enabled OCI Generative AI compatibility by default for standalone merges by preserving the base model's `config.json`; added `--no-oci-compat` to opt out.

### Fixed

* Prevented PEFT from silently upcasting Demo 01 LoRA adapter parameters to FP32, which caused the BF16 dtype validation to fail before training.
