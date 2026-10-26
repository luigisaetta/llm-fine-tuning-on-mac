"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Locate ticket-classification artifacts, load and validate dataset splits, and format records for TRL.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from llm_fine_tuning.paths import find_project_root, require_files
from llm_fine_tuning.tickets.labels import CATEGORIES, OUTPUT_KEYS, SEVERITIES
from llm_fine_tuning.tickets.prompts import build_prompt_messages

TRAIN_FILE_NAME = "train_dataset_v2.jsonl"
VALIDATION_FILE_NAME = "validation_dataset_v2.jsonl"
TEST_FILE_NAME = "test_dataset.jsonl"


@dataclass(frozen=True)
class TicketProjectPaths:
    """Standard locations of the ticket-classification inputs and outputs.

    Attributes:
        project_root: Repository root.
        model_directory: Local ``Qwen/Qwen3-1.7B`` download.
        dataset_directory: Directory of the JSONL splits.
        prompts_directory: Directory of the system prompt files.
        models_directory: Directory below which adapters are saved.
    """

    project_root: Path
    model_directory: Path
    dataset_directory: Path
    prompts_directory: Path
    models_directory: Path

    @classmethod
    def discover(cls, start_directory: Path | None = None) -> "TicketProjectPaths":
        """Return the standard paths of the repository containing ``start_directory``.

        Args:
            start_directory: Directory inside the repository. Defaults to the current working directory.

        Returns:
            The resolved paths.
        """
        root = find_project_root(start_directory)
        tickets = root / "ticket-classification"
        return cls(
            project_root=root,
            model_directory=root / "artifacts" / "models" / "Qwen3-1.7B",
            dataset_directory=tickets / "artifacts" / "datasets",
            prompts_directory=tickets / "prompts",
            models_directory=tickets / "artifacts" / "models",
        )

    @property
    def train_file(self) -> Path:
        """Training split (version 2)."""
        return self.dataset_directory / TRAIN_FILE_NAME

    @property
    def validation_file(self) -> Path:
        """Validation split (version 2)."""
        return self.dataset_directory / VALIDATION_FILE_NAME

    @property
    def test_file(self) -> Path:
        """Held-out test split."""
        return self.dataset_directory / TEST_FILE_NAME


def validate_split(split_name: str, records) -> set[str]:
    """Validate ticket records and return their ticket texts, without printing any ticket content.

    Each record must have exactly a ``user`` ticket message and an ``assistant`` message whose content is a JSON
    object with exactly ``category``, ``severity``, and ``summary``, using allowed labels and a non-empty summary.

    Args:
        split_name: Name used in error messages.
        records: Iterable of records with a ``messages`` list.

    Returns:
        The set of stripped ticket texts.

    Raises:
        ValueError: If a record is malformed, uses an unsupported label, or repeats a ticket text.
    """
    ticket_texts = set()
    for index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"{split_name} record {index} must contain exactly two messages.")
        if [message.get("role") for message in messages] != ["user", "assistant"]:
            raise ValueError(f"{split_name} record {index} must have user and assistant roles.")
        ticket_text = messages[0].get("content", "").strip()
        if not ticket_text:
            raise ValueError(f"{split_name} record {index} has empty ticket text.")
        if ticket_text in ticket_texts:
            raise ValueError(f"{split_name} contains duplicate ticket text at record {index}.")
        ticket_texts.add(ticket_text)
        try:
            output = json.loads(messages[1].get("content", ""))
        except json.JSONDecodeError as error:
            raise ValueError(f"{split_name} record {index} has invalid assistant JSON.") from error
        if not isinstance(output, dict) or set(output) != OUTPUT_KEYS:
            raise ValueError(f"{split_name} record {index} must contain exactly category, severity, and summary.")
        if output["category"] not in CATEGORIES or output["severity"] not in SEVERITIES:
            raise ValueError(f"{split_name} record {index} has an unsupported category or severity.")
        if not isinstance(output["summary"], str) or not output["summary"].strip():
            raise ValueError(f"{split_name} record {index} has an empty summary.")
    return ticket_texts


def load_training_splits(paths: TicketProjectPaths):
    """Load and validate the training and validation splits; the test split is never loaded.

    Args:
        paths: Ticket-classification paths.

    Returns:
        A ``DatasetDict`` with ``train`` and ``validation`` splits of raw records.

    Raises:
        FileNotFoundError: If a split file is missing.
        ValueError: If a split is invalid or the two splits share a ticket text.
    """
    require_files(paths.train_file, paths.validation_file)
    splits = load_dataset("json", data_files={"train": str(paths.train_file), "validation": str(paths.validation_file)})
    train_texts = validate_split("train", splits["train"])
    validation_texts = validate_split("validation", splits["validation"])
    if train_texts & validation_texts:
        raise ValueError("Training and validation splits share exact ticket text.")
    return splits


def load_test_split(paths: TicketProjectPaths):
    """Load and validate only the held-out test split.

    Args:
        paths: Ticket-classification paths.

    Returns:
        The test ``Dataset`` of raw records.

    Raises:
        FileNotFoundError: If the test file is missing.
        ValueError: If the split is empty or invalid.
    """
    require_files(paths.test_file)
    test_split = load_dataset("json", data_files={"test": str(paths.test_file)}, split="test")
    if len(test_split) == 0:
        raise ValueError("The held-out test dataset is empty.")
    validate_split("test", test_split)
    return test_split


def to_prompt_completion(splits, system_prompt: str | None):
    """Convert raw records to TRL conversational prompt/completion records.

    Args:
        splits: ``DatasetDict`` returned by :func:`load_training_splits`.
        system_prompt: System prompt text, or ``None`` for no system message.

    Returns:
        A ``DatasetDict`` with ``prompt`` and ``completion`` columns only.
    """

    def convert(record: dict) -> dict:
        return {
            "prompt": build_prompt_messages(record["messages"][0], system_prompt),
            "completion": [record["messages"][1]],
        }

    return splits.map(convert, remove_columns=splits["train"].column_names)


def reference_outputs(records) -> list[dict]:
    """Return the expected JSON output of each record.

    Args:
        records: Iterable of validated records.

    Returns:
        One parsed assistant output per record.
    """
    return [json.loads(record["messages"][1]["content"]) for record in records]
