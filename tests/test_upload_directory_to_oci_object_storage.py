"""
Author: L. Saetta
Date last modified: 2026-09-03
License: MIT
Description: Unit tests for OCI Object Storage directory-upload planning and multipart upload options.
"""

from pathlib import Path

import pytest

from scripts.upload_directory_to_oci_object_storage import (
    MULTIPART_PART_SIZE,
    build_object_name,
    build_upload_plan,
    collect_regular_files,
    upload_plan,
)


def test_build_upload_plan_preserves_nested_relative_paths(tmp_path: Path) -> None:
    """Nested files map below a normalized Object Storage prefix without local paths."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "nested" / "model.safetensors").write_bytes(b"weights")

    plan = build_upload_plan(tmp_path, "/models/qwen3/")

    assert [(path.relative_to(tmp_path).as_posix(), object_name) for path, object_name in plan] == [
        ("config.json", "models/qwen3/config.json"),
        ("nested/model.safetensors", "models/qwen3/nested/model.safetensors"),
    ]


def test_collect_regular_files_rejects_empty_directory(tmp_path: Path) -> None:
    """An empty source directory reports an actionable error."""
    with pytest.raises(ValueError, match="contains no regular files"):
        collect_regular_files(tmp_path)


def test_build_object_name_without_prefix_uses_relative_posix_path() -> None:
    """A blank prefix does not add a leading slash to object names."""
    assert build_object_name(Path("nested") / "model.safetensors", "") == "nested/model.safetensors"


def test_upload_plan_uses_multipart_progress_and_no_overwrite(tmp_path: Path) -> None:
    """Each file uses multipart settings, progress reporting, and safe overwrite protection."""
    local_file = tmp_path / "model.safetensors"
    local_file.write_bytes(b"123456")
    progress_instances: list[FakeProgress] = []
    upload_manager = FakeUploadManager()

    def progress_factory(**_kwargs: object) -> "FakeProgress":
        progress = FakeProgress()
        progress_instances.append(progress)
        return progress

    upload_plan(
        upload_manager,
        [(local_file, "models/model.safetensors")],
        namespace="namespace",
        bucket="bucket",
        overwrite=False,
        progress_factory=progress_factory,
    )

    assert len(upload_manager.calls) == 1
    namespace, bucket, object_name, file_path, kwargs = upload_manager.calls[0]
    assert (namespace, bucket, object_name, file_path) == (
        "namespace",
        "bucket",
        "models/model.safetensors",
        str(local_file),
    )
    assert kwargs["part_size"] == MULTIPART_PART_SIZE
    assert kwargs["if_none_match"] == "*"
    assert progress_instances[0].updates == [local_file.stat().st_size]


class FakeProgress:
    """Minimal context-managed progress bar for upload tests."""

    def __init__(self) -> None:
        self.updates: list[int] = []

    def __enter__(self) -> "FakeProgress":
        """Return the active fake progress bar."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the fake progress bar."""

    def update(self, amount: int) -> None:
        """Record one reported byte increment."""
        self.updates.append(amount)


class FakeUploadManager:
    """Fake OCI upload manager that reports complete progress for each test file."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str, dict[str, object]]] = []

    def upload_file(self, namespace: str, bucket: str, object_name: str, file_path: str, **kwargs: object) -> None:
        """Record the upload request and invoke the supplied progress callback."""
        self.calls.append((namespace, bucket, object_name, file_path, kwargs))
        progress_callback = kwargs["progress_callback"]
        progress_callback(Path(file_path).stat().st_size)
