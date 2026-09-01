# Demo 00: Local Qwen3-0.6B Inference Specification

## Problem

Before fine-tuning, learners need a small, observable proof that the downloaded Qwen3-0.6B model, tokenizer, PyTorch device selection, and prompt-generation path work together on their MacBook.

## Scope

Create `demo00/` containing one restartable Jupyter notebook and concise usage documentation. The notebook loads the already-downloaded model from `artifacts/models/Qwen3-0.6B`, verifies the local artifacts and loaded model, then generates a response to an editable user prompt using Transformers.

## Assumptions

* `Qwen/Qwen3-0.6B` has been downloaded at the pinned revision in specification 003.
* The user starts JupyterLab from the repository root with the `llm-fine-tuning-on-mac` Conda environment selected as the kernel.
* The required PyTorch and Transformers packages are installed from `requirements.txt`.
* MPS device selection follows specification 002. `auto` selects MPS if usable; otherwise it reports CPU fallback.

## Functional requirements

* The notebook must use `AutoTokenizer` and `AutoModelForCausalLM` from `transformers`.
* It must load only from `artifacts/models/Qwen3-0.6B` with `local_files_only=True`; it must not download from the Hub.
* Before loading, it must check for `config.json`, `tokenizer.json`, and `model.safetensors`, producing an actionable error if any are missing.
* It must print the selected device, PyTorch version, and MPS build/availability state.
* It must validate that the loaded model has `model_type == "qwen3"`, is on the selected device type, is in evaluation mode, and display a successful-load summary.
* It must provide one clearly marked editable `USER_PROMPT` variable and generate a response using the tokenizer chat template with thinking disabled for a concise first demo.
* Inference must run under `torch.inference_mode()` and move tokenized inputs to the selected device.

## Non-functional requirements

* The notebook must have no stored outputs or execution counts and must be runnable top-to-bottom after a kernel restart.
* Documentation and notebook narrative cells must be in English.
* Model files, generated outputs, and notebook checkpoints remain ignored by Git.
* The demo must make no claims about fine-tuning quality, speed, memory use, or universal MPS support.

## Acceptance criteria

* `demo00/README.md` gives the command to start JupyterLab and the notebook path.
* The notebook opens as valid JSON and contains the required artifact, device, model, and inference steps.
* Missing local artifacts fail before a model-loading attempt.
* On a machine with available MPS, the notebook reports and uses `mps`; otherwise it visibly reports `cpu` fallback.
* Updating only `USER_PROMPT` changes the question sent to the model.

## Verification

* Parse the notebook with `python -m json.tool`.
* Inspect it with `jupyter nbconvert --to notebook --inplace --ClearOutputPreprocessor.enabled=True` only if execution metadata needs cleanup; do not execute the model as a unit test.
* Run the notebook manually as an optional integration check after the model has been downloaded. Confirm that the load summary is displayed and a response is generated.
