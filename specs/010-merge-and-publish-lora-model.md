# Merge and Publish LoRA Model Specification

## Problem

After a successful LoRA run, a learner may need a standalone Transformers model rather than a base model plus a separate adapter. The project needs a reproducible local merge workflow with an explicit, opt-in Hugging Face Hub upload.

## Scope

Provide a command-line script that loads the local `Qwen3-1.7B` base model and a local LoRA adapter, merges the adapter into the base weights, saves a standalone Safetensors model and tokenizer under an ignored artifact directory, and can upload that directory to a model repository on the Hugging Face Hub. The default output must preserve the base-model `config.json` for OCI Generative AI imported-model compatibility.

## Assumptions

* The base model is stored under `artifacts/models/Qwen3-1.7B` and the Demo 01 adapter under `artifacts/training/demo01-qwen3-1.7b-lora/adapter` unless command-line options override them.
* The base-model revision is not pinned; the generated model card must state this provenance limitation.
* The local machine has enough memory for the base model, adapter, and merged weights in `bfloat16`. CPU is the default device; MPS is available only when explicitly requested and supported.
* The user has permission to create or update the specified Hugging Face model repository when requesting upload.
* The merged weight shapes and model architecture remain compatible with the base model; only LoRA deltas alter the parameter values.

## Functional requirements

* Validate the base model, including either a single Safetensors file or a complete indexed set of shards, and validate adapter configuration and weights before model loading.
* Load the base model in `torch.bfloat16`, load the adapter with `autocast_adapter_dtype=False`, use `PeftModel.merge_and_unload(safe_merge=True)`, and save a standalone `bfloat16` model with `safe_serialization=True` together with its tokenizer.
* Enable OCI compatibility by default: after serialization, copy the base-model `config.json` into the merged output so that model architecture metadata, `torch_dtype`, and `transformers_version` match the compatible base model. Support an explicit `--no-oci-compat` opt-out.
* Refuse to write into a non-empty output directory.
* Write a generated model card with Hugging Face metadata declaring `Qwen/Qwen3-1.7B` as the base model, Transformers, text generation, and merge tags. It must identify local base-model and adapter provenance without embedding local filesystem paths, as well as the unpinned-revision status and merge device.
* Upload only when both `--push-to-hub` and `--repo-id OWNER/NAME` are supplied. Read the authentication token only from the `HF_TOKEN` environment variable.
* Create a model repository when needed and upload the saved output directory with an explicit commit message. The script must not upload datasets, checkpoints, or source artifacts.
* Support an explicit upload-only retry for a complete existing merged-model directory without rerunning or overwriting the merge.

## Out of scope

Quantization, adapter composition, pushing the adapter itself, automatic repository deletion, and uploading private datasets are out of scope.

## Acceptance criteria

* The script is documented in the root README with separate local-merge and opt-in upload commands.
* The default invocation performs no network access and no upload.
* The standalone merged model and generated model card declare `bfloat16` weights.
* By default, the merged output `config.json` is byte-for-byte equal to the base-model `config.json`; `--no-oci-compat` retains the configuration generated during serialization.
* Unit tests validate local artifact checks without downloading a model or contacting the Hub.
* The script reports actionable errors for missing artifacts, an unavailable requested MPS device, a non-empty output directory, a missing Hub repository ID, or a missing `HF_TOKEN`.

## Verification

* Run `pytest -q` in the project Conda environment.
* Run the script with `--help` in the project Conda environment.
* Optionally perform a local merge and verify that `config.json`, tokenizer files, Safetensors weights, and the generated model card are written to the selected output directory.
* Verify that the default merged `config.json` matches the base-model configuration before importing to OCI Generative AI.
* Optionally run the explicit upload command with a repository owned by the authenticated user.

## References

* [PEFT merge_and_unload](https://huggingface.co/docs/peft/package_reference/peft_model#peft.PeftModel.merge_and_unload)
* [Hugging Face Hub upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload)
