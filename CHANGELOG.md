# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

* Added this project changelog for user-visible additions and significant changes.
* Added a Demo 01 loss-trend chart with distinct training and validation loss series and a grid.

### Changed

* Updated Demo 01 LoRA fine-tuning to load the Qwen3 base model and trainable adapter parameters in BF16, enable BF16 training and evaluation, and require compatible MPS and macOS 14 or later.
* Updated the standalone LoRA merge workflow to save BF16 model weights, matching the Qwen3 base model and Demo 01 adapter training.
* Enabled OCI Generative AI compatibility by default for standalone merges by preserving the base model's `config.json`; added `--no-oci-compat` to opt out.

### Fixed

* Prevented PEFT from silently upcasting Demo 01 LoRA adapter parameters to FP32, which caused the BF16 dtype validation to fail before training.
