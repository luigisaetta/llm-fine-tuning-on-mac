"""
Author: L. Saetta
Date last modified: 2026-09-03
License: MIT
Description: Upload a local merged-model directory recursively to OCI Object Storage with multipart uploads and progress reporting.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

try:
    import oci
except ImportError:  # pragma: no cover - exercised only when the optional runtime dependency is absent.
    oci = None

from tqdm import tqdm


MEBIBYTE = 1024 * 1024
MULTIPART_PART_SIZE = 128 * MEBIBYTE


class ProgressBar(Protocol):
    """Describe the small tqdm interface used by the upload workflow."""

    def __enter__(self) -> "ProgressBar":
        """Start the progress display."""

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the progress display."""

    def update(self, amount: int) -> None:
        """Advance the display by an uploaded byte count."""


ProgressFactory = Callable[..., ProgressBar]


def collect_regular_files(source_directory: Path) -> list[Path]:
    """Return sorted regular files below a source directory without following symlinks.

    Args:
        source_directory: Existing directory to upload.

    Returns:
        Regular files sorted by their paths relative to the source directory.

    Raises:
        FileNotFoundError: If the source directory does not exist or is not a directory.
        ValueError: If the source directory contains no regular files.
    """
    if not source_directory.is_dir():
        raise FileNotFoundError(f"Source directory does not exist or is not a directory: {source_directory}")

    files = sorted(
        (path for path in source_directory.rglob("*") if path.is_file() and not path.is_symlink()),
        key=lambda path: path.relative_to(source_directory).as_posix(),
    )
    if not files:
        raise ValueError(f"Source directory contains no regular files to upload: {source_directory}")
    return files


def build_object_name(relative_path: Path, prefix: str) -> str:
    """Build a normalized Object Storage name for one source-relative path.

    Args:
        relative_path: File path relative to the chosen source directory.
        prefix: Optional Object Storage prefix.

    Returns:
        An Object Storage object name using POSIX separators.
    """
    normalized_prefix = prefix.strip("/")
    relative_name = relative_path.as_posix()
    return f"{normalized_prefix}/{relative_name}" if normalized_prefix else relative_name


def build_upload_plan(source_directory: Path, prefix: str) -> list[tuple[Path, str]]:
    """Map all uploadable files to their Object Storage names.

    Args:
        source_directory: Existing directory containing uploadable files.
        prefix: Optional Object Storage prefix.

    Returns:
        Tuples of local file paths and destination object names.
    """
    return [
        (file_path, build_object_name(file_path.relative_to(source_directory), prefix))
        for file_path in collect_regular_files(source_directory)
    ]


def create_upload_manager(config_file: str | None, profile: str):
    """Create an OCI multipart upload manager from local OCI API-key configuration.

    Args:
        config_file: Optional OCI config-file path. ``None`` uses the SDK default location.
        profile: OCI configuration profile name.

    Returns:
        An OCI Object Storage upload manager configured for sequential multipart uploads.

    Raises:
        RuntimeError: If the OCI Python SDK is not installed.
    """
    if oci is None:
        raise RuntimeError(
            "The OCI Python SDK is not installed. Run 'python -m pip install -r requirements.txt' "
            "in the llm-fine-tuning-on-mac Conda environment."
        )

    config = (
        oci.config.from_file(file_location=config_file, profile_name=profile)
        if config_file
        else oci.config.from_file(profile_name=profile)
    )
    oci.config.validate_config(config)
    client = oci.object_storage.ObjectStorageClient(config, retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)
    return oci.object_storage.UploadManager(
        client,
        allow_multipart_uploads=True,
        allow_parallel_uploads=False,
    )


def upload_plan(
    upload_manager: object,
    plan: list[tuple[Path, str]],
    namespace: str,
    bucket: str,
    overwrite: bool,
    progress_factory: ProgressFactory = tqdm,
) -> None:
    """Upload a prepared directory plan to OCI Object Storage.

    Args:
        upload_manager: OCI upload manager exposing ``upload_file``.
        plan: Local paths and destination object names to upload.
        namespace: Object Storage namespace containing the destination bucket.
        bucket: Destination bucket name.
        overwrite: Whether existing objects may be overwritten.
        progress_factory: Progress-bar factory, injectable for tests.
    """
    for local_path, object_name in plan:
        file_size = local_path.stat().st_size
        with progress_factory(
            total=file_size,
            desc=local_path.name,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=True,
        ) as progress:
            upload_kwargs: dict[str, object] = {
                "part_size": MULTIPART_PART_SIZE,
                "progress_callback": progress.update,
            }
            if not overwrite:
                upload_kwargs["if_none_match"] = "*"
            upload_manager.upload_file(namespace, bucket, object_name, str(local_path), **upload_kwargs)
        print(f"Uploaded: {object_name}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for an OCI Object Storage directory upload."""
    parser = argparse.ArgumentParser(
        description="Recursively upload a local merged-model directory to OCI Object Storage."
    )
    parser.add_argument("--source-dir", type=Path, required=True, help="Local directory whose complete contents are uploaded.")
    parser.add_argument("--namespace", required=True, help="Destination Object Storage namespace.")
    parser.add_argument("--bucket", required=True, help="Destination Object Storage bucket name.")
    parser.add_argument("--prefix", default="", help="Optional Object Storage prefix below the bucket root.")
    parser.add_argument("--config-file", help="Optional OCI API-key configuration file; defaults to the OCI SDK location.")
    parser.add_argument("--profile", default="DEFAULT", help="OCI configuration profile name (default: DEFAULT).")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting objects that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned uploads without authenticating or uploading.")
    return parser.parse_args()


def main() -> None:
    """Plan and upload a local directory to OCI Object Storage."""
    args = parse_arguments()
    source_directory = args.source_dir.resolve()
    plan = build_upload_plan(source_directory, args.prefix)
    total_bytes = sum(local_path.stat().st_size for local_path, _ in plan)

    print(f"Source directory: {source_directory}")
    print(f"Destination: {args.namespace}/{args.bucket}/{args.prefix.strip('/')}")
    print(f"Files: {len(plan)}")
    print(f"Total bytes: {total_bytes:,}")
    if args.dry_run:
        for _, object_name in plan:
            print(f"Would upload: {object_name}")
        return

    upload_manager = create_upload_manager(args.config_file, args.profile)
    upload_plan(upload_manager, plan, args.namespace, args.bucket, args.overwrite)
    print("Upload completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Upload failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
