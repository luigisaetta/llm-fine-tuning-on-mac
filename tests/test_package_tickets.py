"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Unit tests for ticket-classification data validation, prompts, scoring, training configuration, and
dataset-shift measurements, using small in-memory fixtures and no model.
"""

import json
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from llm_fine_tuning.tickets import training
from llm_fine_tuning.tickets.data import TicketProjectPaths, reference_outputs, to_prompt_completion, validate_split
from llm_fine_tuning.tickets.evaluation import metrics_rows, parse_response, score_responses, severity_confusion_matrix
from llm_fine_tuning.tickets.prompts import (
    PROMPT_FILES,
    adapter_run_name,
    build_prompt_messages,
    load_system_prompt,
    require_sequences_fit,
    verify_adapter_prompt,
    write_prompt_metadata,
)
from llm_fine_tuning.tickets.shift_analysis import (
    conditional_severity_shift,
    extract_text_features,
    severity_distribution,
    severity_term_overlap,
    text_feature_rows,
    total_variation_distance,
)

PROMPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "ticket-classification" / "prompts"


def record(ticket: str, category: str = "DB-01", severity: str = "P2", summary: str = "Database is slow") -> dict:
    """Return one ticket record in the dataset format."""
    output = json.dumps({"category": category, "severity": severity, "summary": summary})
    return {"messages": [{"role": "user", "content": ticket}, {"role": "assistant", "content": output}]}


def test_validate_split_accepts_valid_records() -> None:
    """Valid records return their stripped ticket texts."""
    assert validate_split("train", [record(" Slow query "), record("Locked account", "AUTH-01", "P3")]) == {"Slow query", "Locked account"}


@pytest.mark.parametrize(
    ("bad_record", "message"),
    [
        ({"messages": []}, "exactly two messages"),
        ({"messages": [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]}, "user and assistant"),
        (record("   "), "empty ticket text"),
        ({"messages": [{"role": "user", "content": "t"}, {"role": "assistant", "content": "not json"}]}, "invalid assistant JSON"),
        (record("t", category="XX-99"), "unsupported category"),
        (record("t", summary=" "), "empty summary"),
    ],
)
def test_validate_split_rejects_invalid_records(bad_record: dict, message: str) -> None:
    """Each schema violation produces an actionable error."""
    with pytest.raises(ValueError, match=message):
        validate_split("train", [bad_record])


def test_validate_split_rejects_duplicates() -> None:
    """A repeated ticket text within one split is rejected."""
    with pytest.raises(ValueError, match="duplicate"):
        validate_split("train", [record("same"), record("same")])


def test_ticket_project_paths(tmp_path) -> None:
    """Standard paths are derived from the repository root."""
    (tmp_path / "AGENTS.md").write_text("", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    paths = TicketProjectPaths.discover(tmp_path)
    assert paths.model_directory == tmp_path.resolve() / "artifacts" / "models" / "Qwen3-1.7B"
    assert paths.train_file.name == "train_dataset_v2.jsonl"
    assert paths.test_file.parent == paths.dataset_directory


def test_prompt_construction_and_variants() -> None:
    """The system message precedes the ticket only when a system prompt is set."""
    ticket = {"role": "user", "content": "t"}
    assert build_prompt_messages(ticket, None) == [ticket]
    assert build_prompt_messages(ticket, "rules") == [{"role": "system", "content": "rules"}, ticket]
    assert load_system_prompt(PROMPTS_DIRECTORY, "none") is None
    for variant in ("short", "full"):
        assert load_system_prompt(PROMPTS_DIRECTORY, variant) == (PROMPTS_DIRECTORY / PROMPT_FILES[variant]).read_text(encoding="utf-8").strip()
    with pytest.raises(ValueError):
        load_system_prompt(PROMPTS_DIRECTORY, "other")
    assert adapter_run_name("none") == "demo06-qwen3-1.7b-ticket-classification-lora"
    assert adapter_run_name("short") == "demo06-qwen3-1.7b-ticket-classification-lora-prompt-short"


def test_to_prompt_completion() -> None:
    """TRL records contain the prompt messages and the assistant completion only."""
    splits = DatasetDict({"train": Dataset.from_list([record("t")]), "validation": Dataset.from_list([record("v")])})
    converted = to_prompt_completion(splits, "rules")
    assert converted["train"].column_names == ["prompt", "completion"]
    assert [message["role"] for message in converted["train"][0]["prompt"]] == ["system", "user"]
    assert converted["validation"][0]["completion"][0]["role"] == "assistant"


def test_prompt_metadata_round_trip(tmp_path, capsys) -> None:
    """An adapter is accepted only with the variant and prompt text it was trained with."""
    write_prompt_metadata(tmp_path, "short", "rules")
    verify_adapter_prompt(tmp_path, "short", "rules")
    with pytest.raises(ValueError):
        verify_adapter_prompt(tmp_path, "short", "changed rules")
    with pytest.raises(ValueError):
        verify_adapter_prompt(tmp_path, "none", None)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    verify_adapter_prompt(legacy, "none", None)
    assert "trained before specification 021" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        verify_adapter_prompt(legacy, "short", "rules")


class WordTokenizer:
    """Minimal tokenizer: the chat template joins contents and every word is one token."""

    def apply_chat_template(self, conversation, tokenize=False):
        return " ".join(message["content"] for message in conversation)

    def __call__(self, text):
        return {"input_ids": text.split()}


def test_require_sequences_fit() -> None:
    """The check returns the longest length and fails when it exceeds the limit."""
    records = [record("one two three")]
    longest = require_sequences_fit(WordTokenizer(), [records], None, max_sequence_length=100)
    assert longest > 3
    with pytest.raises(ValueError, match="MAX_SEQUENCE_LENGTH"):
        require_sequences_fit(WordTokenizer(), [records], "a long system prompt", max_sequence_length=longest)


def test_scoring_counts_invalid_responses_as_incorrect() -> None:
    """Invalid JSON is wrong for every label metric and lowers the valid-JSON rate."""
    references = [{"category": "DB-01", "severity": "P2"}] * 4
    responses = [
        '{"category": "DB-01", "severity": "P2", "summary": "s"}',
        ' {"category": "DB-01", "severity": "P3", "summary": "s"} ',
        '["DB-01", "P2"]',
        "The category is DB-01",
    ]
    assert parse_response(responses[2]) is None
    assert score_responses(responses, references) == {
        "category_accuracy": 0.5, "severity_accuracy": 0.25, "joint_accuracy": 0.25, "valid_json_rate": 0.5,
    }
    with pytest.raises(ValueError):
        score_responses([], [])


def test_severity_confusion_matrix() -> None:
    """Rows are true severities, columns predicted ones; invalid predictions are counted apart."""
    references = [{"severity": "P1"}, {"severity": "P2"}, {"severity": "P2"}]
    responses = ['{"severity": "P1"}', '{"severity": "P3"}', "invalid"]
    matrix, invalid = severity_confusion_matrix(responses, references)
    assert matrix[0, 0] == 1 and matrix[1, 2] == 1 and matrix.sum() == 2 and invalid == 1


def test_metrics_rows_with_delta() -> None:
    """Each metric row lists every column and the requested difference."""
    base = {"category_accuracy": 0.5, "severity_accuracy": 0.5, "joint_accuracy": 0.25, "valid_json_rate": 1.0}
    tuned = {"category_accuracy": 0.75, "severity_accuracy": 0.5, "joint_accuracy": 0.5, "valid_json_rate": 1.0}
    rows = metrics_rows({"Base": base, "Tuned": tuned}, delta=("Tuned", "Base"))
    assert rows[0] == ["Category Accuracy", "50.0%", "75.0%", "+25.0%"]
    assert len(rows) == 4


def test_reference_outputs() -> None:
    """Expected outputs are parsed from the assistant messages."""
    assert reference_outputs([record("t", "NET-01", "P4")])[0]["category"] == "NET-01"


def test_sft_config_maps_settings_and_fixed_policy(monkeypatch, tmp_path) -> None:
    """Notebook settings and the fixed project policy reach the trainer configuration unchanged."""
    monkeypatch.setattr(training, "SFTConfig", lambda **arguments: arguments)
    settings = training.TrainingSettings(num_train_epochs=3, learning_rate=2e-4)
    arguments = training.build_sft_config(tmp_path, settings)
    assert settings.effective_batch_size == 8
    assert arguments["num_train_epochs"] == 3 and arguments["learning_rate"] == 2e-4
    assert arguments["per_device_train_batch_size"] == 4 and arguments["gradient_accumulation_steps"] == 2
    assert arguments["max_length"] == 512 and arguments["completion_only_loss"] is True
    assert arguments["metric_for_best_model"] == "eval_joint_accuracy" and arguments["load_best_model_at_end"] is True
    assert arguments["bf16"] is True and arguments["gradient_checkpointing"] is True and arguments["output_dir"] == str(tmp_path)


def shift_record(split: str, ticket: str, category: str, severity: str) -> dict:
    """Return one analysis record with text features."""
    return {"split": split, "ticket": ticket, "category": category, "severity": severity, **extract_text_features(ticket)}


def test_shift_measurements() -> None:
    """Distributions, distance, length features, conditional shift, and term overlap are deterministic."""
    validation = [shift_record("V", "Disk full. No workaround.", "SRV-01", "P1"), shift_record("V", "Slow query", "DB-01", "P3")]
    test = [shift_record("T", "Slow query", "DB-01", "P3"), shift_record("T", "Minor glitch", "DB-01", "P4")]
    assert extract_text_features("Disk full. No workaround.") == {"characters": 25, "words": 4, "sentences": 2}
    assert severity_distribution(test) == {"P1": 0.0, "P2": 0.0, "P3": 0.5, "P4": 0.5}
    assert total_variation_distance(severity_distribution(validation), severity_distribution(test)) == pytest.approx(0.5)
    assert text_feature_rows({"V": validation})[1] == ["V", "words", "3.0", 4, 2, 4]
    categories, matrix = conditional_severity_shift(validation, test)
    assert categories == ["DB-01", "SRV-01"]
    assert matrix[0] == [0.0, 0.0, -0.5, 0.5] and matrix[1] == [-1.0, 0.0, 0.0, 0.0]
    rows, overlaps = severity_term_overlap(validation, test, limit=2)
    assert overlaps["P3"] == 1.0 and overlaps["P2"] == 1.0 and rows[0][0] == "P1"
