# Upload Merged Models to OCI Object Storage Specification

## Problem

A locally merged model must be transferred to an OCI Object Storage bucket before it can be imported into OCI Generative AI. Model Safetensors files can be several GiB, so the workflow needs a reliable, observable upload mechanism.

## Scope

Provide a Python command-line script that recursively uploads all regular files from a supplied local model directory to an OCI Object Storage bucket. Preserve each path relative to the source directory below an optional object prefix. Use the OCI Python SDK and multipart upload support for large files, with terminal progress reporting for each uploaded file.

## Assumptions

* The user has installed the `oci` Python package from `requirements.txt` in the `llm-fine-tuning-on-mac` Conda environment.
* The user has configured OCI API-key authentication in an OCI config file and has permission to write objects to the supplied namespace and bucket.
* The bucket already exists; the script does not create buckets, alter policies, or import models into OCI Generative AI.
* The source directory contains only artifacts intended for upload. Symbolic links are excluded to prevent uploads of files outside the selected directory.

## Functional requirements

* Require a source directory, namespace, bucket name, and optional object prefix. Accept an OCI config-file path and profile name, defaulting to the standard OCI configuration and `DEFAULT` profile.
* Recursively upload every regular file, preserving its relative POSIX path under the requested prefix.
* Use `oci.object_storage.UploadManager` with multipart uploads enabled and a 128 MiB part size. Upload parts sequentially so the per-file progress callback is reliable in the local terminal.
* Show a byte-based progress bar for each file and print the final Object Storage object name.
* Refuse to overwrite existing objects by default using `if_none_match="*"`. Support explicit `--overwrite` opt-in.
* Support `--dry-run`, which prints the planned object names and performs no OCI authentication or network request.
* Report invalid directories, empty source directories, unavailable OCI SDK installation, and upload failures with actionable errors.

## Out of scope

Parallel part uploads, resumable upload state, bucket creation, object deletion, OCI Generative AI model import, and Object Storage lifecycle or IAM-policy management are out of scope.

## Acceptance criteria

* The script has a usage README with an OCI-compatible merged-model example.
* A file at `model.safetensors` with multi-GiB size is passed to the OCI SDK upload manager with multipart options and a progress callback.
* Nested source paths become object names below the requested prefix without local absolute paths.
* Tests use local fixtures and fakes only; they do not authenticate to OCI or upload data.

## Verification

* Run `pytest -q` in the project Conda environment.
* Run the script with `--help`.
* Run `--dry-run` against a local merged-model directory and inspect object names before a real upload.
* For a real upload, verify the expected object count and object names in the selected bucket before creating an OCI Generative AI import.
