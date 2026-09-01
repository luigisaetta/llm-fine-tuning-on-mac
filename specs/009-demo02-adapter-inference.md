# Demo 02: Local LoRA Adapter Inference Specification

## Problem

After Demo 01 saves a LoRA adapter, learners need a restartable way to load it with the local Qwen3-0.6B base model and ask unrestricted questions.

## Scope

Create demo02/ with a notebook and usage guide. The notebook loads ignored local base-model and adapter artifacts, selects MPS when available, validates the PEFT model, and generates a deterministic answer for an editable prompt.

## Requirements

* Demo 01 must have saved its adapter under artifacts/training/demo01-qwen3-0.6b-lora/adapter/; the base model remains under artifacts/models/Qwen3-0.6B/.
* Use AutoTokenizer and AutoModelForCausalLM for the base model and PeftModel.from_pretrained for the adapter, with local files only.
* Validate base-model files, adapter_config.json, and adapter weights before loading.
* Device selection follows specification 002. Confirm Qwen3, PEFT model, selected device, and evaluation mode.
* Expose one editable USER_PROMPT. Any question is allowed; recommend English because the initial dataset is English.
* Use the chat template with thinking disabled, torch.inference_mode(), and deterministic generation.

## Acceptance criteria

* demo02/README.md contains artifact prerequisites, the JupyterLab command, and notebook path.
* The notebook has no stored outputs or execution counts and reports actionable errors for absent artifacts.
* Updating only USER_PROMPT allows another arbitrary question.
* The notebook does not download or upload data, or save prompts and responses.

## Verification

* Parse the notebook as version-4 JSON and assert code cells have no stored outputs.
* Run it manually as an optional integration check after Demo 01 has created the adapter.

