# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

* Added this project changelog for user-visible additions and significant changes.

### Changed

* Updated Demo 01 LoRA fine-tuning to load the Qwen3 base model and trainable adapter parameters in BF16, enable BF16 training and evaluation, and require compatible MPS and macOS 14 or later.

### Fixed

* Prevented PEFT from silently upcasting Demo 01 LoRA adapter parameters to FP32, which caused the BF16 dtype validation to fail before training.
