# OCI AI Data Platform Ticket Classification Training

## Problem

OCI AI Data Platform training compiles a Triton CUDA helper when `trainer.train()` starts. Its environment does not provide the Python 3.11 development headers in the system include directory, so compilation fails when `Python.h` cannot be found.

## Scope

This specification covers the OCI AI Data Platform notebooks in `ticket-classification/ai-dp/`:

* `fix_system_environment.ipynb` extracts the supplied `python3.11-devel.rpm` under the configured OCI volume.
* `nb_fine_tuning_lora.ipynb` configures the C compiler include-path environment before its cell containing `trainer.train()` runs.
* `inspect_local_training_storage.ipynb` inspects candidate local directories before a local checkpoint location is selected.

It does not change the LoRA model, dataset, optimizer, checkpoint, or evaluation behavior.

## Assumptions

* The OCI volume is mounted at `/Volumes/fine_tuning/fine_tuning/vol_finetuning`.
* `fix_system_environment.ipynb` has completed successfully and extracted `Python.h` to `python311-devel/payload/usr/include/python3.11/` below that volume.
* The training kernel runs on Linux with a C compiler that honors `C_INCLUDE_PATH` or `CPATH`.
* A local filesystem with enough capacity is available on the OCI workspace node, but its mount path and capacity are not assumed.

## Requirements

* The training notebook must have a dedicated preflight cell immediately before the cell that contains `trainer.train()`.
* The preflight cell must confirm that `Python.h` exists at the extracted location and fail with instructions to run the fix notebook if it does not.
* The preflight cell must prepend the extracted header directory to both `C_INCLUDE_PATH` and `CPATH`, retaining any existing values.
* The preflight cell must display the resolved header directory and effective include-path variables without exposing credentials or dataset records.
* The storage-inspection notebook must report the mounted filesystems, filesystem type, resolved path, available capacity, device identifier, and write-check result for each editable candidate directory.
* The storage-inspection notebook may write only a small temporary file, which it must remove before the cell completes. It must not download a model, access a dataset, start training, or create checkpoints.

## Acceptance criteria

* When the header is present, running the preflight cell sets both variables to include the header directory before training begins.
* When the header is absent, the preflight cell stops before `trainer.train()` with an actionable `FileNotFoundError`.
* The notebook remains valid nbformat version 4 JSON.
* The storage-inspection notebook identifies whether every candidate meets an editable free-space threshold, but does not automatically select a training directory.

## Verification

* Parse the notebook as JSON and confirm the preflight cell immediately precedes the training cell.
* Statically check that the preflight cell references `Python.h`, `C_INCLUDE_PATH`, and `CPATH`, and that it preserves pre-existing include paths.
* Parse the storage-inspection notebook as version-4 JSON and confirm its code cells have no stored outputs or execution counts.
