"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Generate ticket classifications in batches, score them against references, and build confusion matrices.
"""

import json

import numpy as np
import torch
from tqdm.auto import tqdm

from llm_fine_tuning.tickets.data import reference_outputs
from llm_fine_tuning.tickets.labels import SEVERITIES
from llm_fine_tuning.tickets.prompts import render_generation_prompt

METRIC_LABELS = {
    "category_accuracy": "Category Accuracy",
    "severity_accuracy": "Severity Accuracy",
    "joint_accuracy": "Category+Severity Accuracy",
    "valid_json_rate": "Valid JSON Rate",
}


def parse_response(response: str) -> dict | None:
    """Parse a generated response, which must be exactly one JSON object.

    Args:
        response: Generated text.

    Returns:
        The parsed object, or ``None`` when the text is not a JSON object.
    """
    try:
        parsed = json.loads(response.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def score_responses(responses: list[str], references: list[dict]) -> dict[str, float]:
    """Calculate the four aggregate classification metrics.

    An invalid response counts as incorrect for every label metric.

    Args:
        responses: Generated texts.
        references: Expected outputs, in the same order.

    Returns:
        Category accuracy, severity accuracy, joint category-and-severity accuracy, and valid-JSON rate.

    Raises:
        ValueError: If the lists are empty or have different lengths.
    """
    if not responses or len(responses) != len(references):
        raise ValueError("Responses and references must be non-empty lists of the same length.")
    valid = category = severity = joint = 0
    for response, reference in zip(responses, references):
        parsed = parse_response(response)
        if parsed is None:
            continue
        category_correct = parsed.get("category") == reference["category"]
        severity_correct = parsed.get("severity") == reference["severity"]
        valid += 1
        category += category_correct
        severity += severity_correct
        joint += category_correct and severity_correct
    total = len(responses)
    return {
        "category_accuracy": category / total,
        "severity_accuracy": severity / total,
        "joint_accuracy": joint / total,
        "valid_json_rate": valid / total,
    }


def generate_responses(
    model,
    tokenizer,
    records,
    system_prompt: str | None,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    progress_description: str | None = None,
) -> list[str]:
    """Generate one deterministic response per record, in left-padded batches with the KV cache enabled.

    Args:
        model: Base or adapter model.
        tokenizer: Matching tokenizer; its padding side is restored afterwards.
        records: Sequence of raw records supporting ``len`` and indexing.
        system_prompt: System prompt text, or ``None``.
        device: Device of the model.
        batch_size: Records per generation batch.
        max_new_tokens: Maximum generated tokens per response.
        progress_description: Label of a progress bar; no bar is shown when ``None``.

    Returns:
        The generated texts, in record order.
    """
    model.eval()
    responses = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    progress_bar = tqdm(total=len(records), desc=progress_description, unit="ticket") if progress_description else None
    try:
        for start in range(0, len(records), batch_size):
            batch = [records[index] for index in range(start, min(start + batch_size, len(records)))]
            prompts = [render_generation_prompt(tokenizer, record["messages"][0], system_prompt) for record in batch]
            inputs = {name: tensor.to(device) for name, tensor in tokenizer(prompts, return_tensors="pt", padding=True).items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id
                )
            responses.extend(tokenizer.batch_decode(output_ids[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
            if progress_bar:
                progress_bar.update(len(batch))
    finally:
        tokenizer.padding_side = original_padding_side
        if progress_bar:
            progress_bar.close()
    return responses


def evaluate_model(model, tokenizer, records, system_prompt, device, batch_size, max_new_tokens, progress_description=None):
    """Generate responses for every record and score them.

    Args:
        model: Base or adapter model.
        tokenizer: Matching tokenizer.
        records: Sequence of raw records.
        system_prompt: System prompt text, or ``None``.
        device: Device of the model.
        batch_size: Records per generation batch.
        max_new_tokens: Maximum generated tokens per response.
        progress_description: Label of a progress bar, or ``None``.

    Returns:
        A tuple of the metrics dictionary and the generated responses.
    """
    responses = generate_responses(model, tokenizer, records, system_prompt, device, batch_size, max_new_tokens, progress_description)
    return score_responses(responses, reference_outputs(records)), responses


def severity_confusion_matrix(responses: list[str], references: list[dict]) -> tuple[np.ndarray, int]:
    """Count true-versus-predicted severities; responses without a valid severity are counted separately.

    Args:
        responses: Generated texts.
        references: Expected outputs, in the same order.

    Returns:
        The P1-to-P4 count matrix (rows true, columns predicted) and the number of invalid or unknown predictions.
    """
    index = {severity: position for position, severity in enumerate(SEVERITIES)}
    matrix = np.zeros((len(SEVERITIES), len(SEVERITIES)), dtype=int)
    invalid = 0
    for response, reference in zip(responses, references):
        parsed = parse_response(response)
        predicted = parsed.get("severity") if parsed else None
        if predicted not in index:
            invalid += 1
            continue
        matrix[index[reference["severity"]], index[predicted]] += 1
    return matrix, invalid


def metrics_rows(columns: dict[str, dict[str, float]], delta: tuple[str, str] | None = None) -> list[list[str]]:
    """Return table rows with one metric per row and one column per evaluated configuration.

    Args:
        columns: Column name to metrics dictionary, in display order.
        delta: Optional ``(minuend, subtrahend)`` column names; adds their difference as a last column.

    Returns:
        Rows ready for :func:`llm_fine_tuning.display.show_table`.
    """
    rows = []
    for key, label in METRIC_LABELS.items():
        row = [label, *(f"{metrics[key]:.1%}" for metrics in columns.values())]
        if delta:
            row.append(f"{columns[delta[0]][key] - columns[delta[1]][key]:+.1%}")
        rows.append(row)
    return rows
