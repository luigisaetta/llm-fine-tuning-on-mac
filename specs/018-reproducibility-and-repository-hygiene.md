# Reproducibility and Repository Hygiene Specification

## Problem

A new user cannot reliably reproduce the Mac workflow from a clean environment:

* `requirements.txt` admits `transformers>=4.51` and `trl>=0.15`, but the notebooks call `from_pretrained(..., dtype=...)` and `SFTConfig(completion_only_loss=...)`, which those minimum versions do not support.
* `numpy` is imported directly by Demo 06 but is not a declared requirement.
* No tested set of exact package versions is published.
* The `Qwen/Qwen3-1.7B` Hub revision is not pinned, so a later download may differ from the one used for the recorded experiments.
* Committed notebook outputs expose private Demo 01 dataset content and machine-specific absolute paths; Demo 08 has outputs that break its restartability test.
* Local helper scripts containing environment-specific OCI identifiers can be committed by accident.

## Scope

* Raise dependency minimums to the first versions that provide the APIs used by the notebooks and scripts.
* Add a constraints file with the exact versions of the direct dependencies installed in the macOS verification environment.
* Pin the `Qwen/Qwen3-1.7B` download to a full commit hash.
* Clear stored outputs and execution counts from committed notebooks.
* Ignore the root `copy_to_oss.sh` local helper.

## Non-goals

* Changing training, evaluation, model, or dataset behavior.
* Pinning versions for the Linux CUDA or OCI AI Data Platform runtimes; their PyTorch build is installed separately.
* Pinning `Qwen/Qwen3-0.6B` or regenerating existing merged-model cards.

## Requirements

* `requirements.txt` must require `transformers>=4.56` and `trl>=0.17`, and declare `numpy`.
* `ticket-classification/requirements.txt` must use the same `transformers` and `trl` minimums.
* `constraints-macos.txt` must list exact versions of every direct dependency in `requirements.txt`; the README must document `pip install -r requirements.txt -c constraints-macos.txt` as the reproducible install.
* The README download command must pass `--revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
* Every committed `.ipynb` must have no stored outputs and no execution counts.
* `copy_to_oss.sh` must be ignored by Git.

## Acceptance criteria

* The default `pytest` suite passes.
* A static test confirms that every committed notebook has no stored outputs or execution counts.
* `git check-ignore copy_to_oss.sh` succeeds.

## Verification

Verified on 2026-10-25:

* Transformers: the 4.55.4 wheel reads only `torch_dtype` in `PreTrainedModel.from_pretrained`; the 4.56.0 wheel reads `dtype` and keeps `torch_dtype` for backward compatibility.
* TRL: `SFTConfig.completion_only_loss` is absent in 0.16.1 and present in 0.17.0; `max_length` is present in both. This is a static check of field presence only; the end-to-end notebooks were not run with the minimum versions.
* Hub revision: `https://huggingface.co/api/models/Qwen/Qwen3-1.7B` reported `sha` `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, the same commit recorded in the local download metadata under `artifacts/models/Qwen3-1.7B/.cache/huggingface/download/`.
* Constraints: versions captured from the `llm-fine-tuning-on-mac` environment (Python 3.11.0, macOS 26.7.1, Apple M4 Pro). The training notebooks were not re-run as part of this change.

## References

* [Transformers releases](https://github.com/huggingface/transformers/releases)
* [TRL releases](https://github.com/huggingface/trl/releases)
* [Qwen/Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B)
