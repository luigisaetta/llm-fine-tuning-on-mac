# OCI Object Storage Upload Script

`upload_directory_to_oci_object_storage.py` recursively uploads every regular file from a local merged-model directory to an existing OCI Object Storage bucket. It preserves relative paths below the requested prefix and shows a byte-based progress bar for each file.

Large Safetensors files use OCI SDK multipart uploads with 128 MiB parts. Parts are uploaded sequentially so terminal progress remains reliable. The script does not create buckets, delete objects, or import the model into OCI Generative AI.

## Prerequisites

Install project dependencies in the dedicated Conda environment:

```bash
conda activate llm-fine-tuning-on-mac
python -m pip install -r requirements.txt
```

Configure OCI API-key authentication in the standard OCI configuration file, or pass an alternative file with `--config-file`. The selected profile needs permission to write objects to the target namespace and bucket.

## Inspect the upload plan

Use a dry run before uploading. It does not authenticate to OCI or make network requests:

```bash
python scripts/upload_directory_to_oci_object_storage.py \
  --source-dir artifacts/merged/demo01-qwen3-1.7b-merged-oci \
  --namespace YOUR_OBJECT_STORAGE_NAMESPACE \
  --bucket YOUR_BUCKET_NAME \
  --prefix qwen3/demo01-qwen3-1.7b-merged \
  --dry-run
```

## Upload the merged model

Remove `--dry-run` to upload. Existing objects are protected by default:

```bash
python scripts/upload_directory_to_oci_object_storage.py \
  --source-dir artifacts/merged/demo01-qwen3-1.7b-merged-oci \
  --namespace YOUR_OBJECT_STORAGE_NAMESPACE \
  --bucket YOUR_BUCKET_NAME \
  --prefix qwen3/demo01-qwen3-1.7b-merged
```

Use `--overwrite` only when replacing objects under the chosen prefix is intentional. To use a non-default OCI profile, add `--profile PROFILE_NAME`.
