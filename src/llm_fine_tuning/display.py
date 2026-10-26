"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Present notebook results as Markdown tables and Matplotlib charts.
"""

import json
from dataclasses import fields, is_dataclass

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

TRAINING_COLOR = "#2563EB"
VALIDATION_COLOR = "#DC2626"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Return a Markdown table whose first column is left-aligned and the others right-aligned.

    Args:
        headers: Column headers.
        rows: Table rows, one value per header.

    Returns:
        The Markdown table text.
    """
    alignment = ["---"] + ["---:" for _ in headers[1:]]
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(alignment)} |"]
    lines.extend(f"| {' | '.join(str(value) for value in row)} |" for row in rows)
    return "\n".join(lines)


def show_table(headers: list[str], rows: list[list[object]]) -> None:
    """Display a Markdown table in the notebook.

    Args:
        headers: Column headers.
        rows: Table rows, one value per header.
    """
    display(Markdown(markdown_table(headers, rows)))


def show_key_values(title: str, values: dict[str, object]) -> None:
    """Display a two-column table of named values, for settings and runtime facts.

    Args:
        title: Header of the first column.
        values: Names and values to show, in order.
    """
    show_table([title, "Value"], [[f"`{name}`" if name.isidentifier() else name, f"`{value}`"] for name, value in values.items()])


def show_settings(settings) -> None:
    """Display every field of a settings dataclass with its value.

    Args:
        settings: A dataclass instance.

    Raises:
        TypeError: If ``settings`` is not a dataclass instance.
    """
    if not is_dataclass(settings):
        raise TypeError("show_settings expects a dataclass instance.")
    show_key_values("Setting", {field.name: getattr(settings, field.name) for field in fields(settings)})


def plot_loss_curves(log_history: list[dict]) -> None:
    """Plot training and validation loss from a Trainer log history.

    Args:
        log_history: ``trainer.state.log_history``.

    Raises:
        RuntimeError: If the history contains no training or no validation loss.
    """
    training_logs = [entry for entry in log_history if "loss" in entry and "epoch" in entry]
    validation_logs = [entry for entry in log_history if "eval_loss" in entry and "epoch" in entry]
    if not training_logs or not validation_logs:
        raise RuntimeError("Training or validation loss logs are unavailable.")
    _, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        [entry["epoch"] for entry in training_logs],
        [entry["loss"] for entry in training_logs],
        color=TRAINING_COLOR, linewidth=2, marker="o", markersize=4, label="Training loss",
    )
    axis.plot(
        [entry["epoch"] for entry in validation_logs],
        [entry["eval_loss"] for entry in validation_logs],
        color=VALIDATION_COLOR, linewidth=2, marker="s", markersize=6, label="Validation loss",
    )
    axis.set_title("Training and Validation Loss by Epoch", fontsize=14, fontweight="bold")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.55)
    axis.legend(frameon=True)
    axis.set_xticks([entry["epoch"] for entry in validation_logs])
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], title: str, label_name: str = "label") -> None:
    """Plot a confusion matrix with true labels as rows and predicted labels as columns.

    Args:
        matrix: Square count matrix.
        labels: Label for each row and column, in order.
        title: Chart title.
        label_name: Name of the labelled quantity, used in the axis titles.
    """
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels, yticklabels=labels,
        xlabel=f"Predicted {label_name}", ylabel=f"True {label_name}", title=title,
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            value = matrix[row_index, column_index]
            axis.text(column_index, row_index, str(value), ha="center", va="center", color="white" if value > threshold else "black")
    figure.colorbar(image, ax=axis, label="Ticket count")
    plt.tight_layout()
    plt.show()


def show_prediction(title: str, ticket_text: str, response: str, expected: str) -> None:
    """Display one ticket with the generated and the expected JSON side by side in the notebook flow.

    Args:
        title: Heading of the block.
        ticket_text: Ticket shown to the model.
        response: Generated response.
        expected: Expected response.
    """
    try:
        is_valid_json = isinstance(json.loads(response), dict)
    except json.JSONDecodeError:
        is_valid_json = False
    display(Markdown(
        f"### {title}\n\n"
        f"**Ticket**\n```text\n{ticket_text}\n```\n\n"
        f"**Fine-tuned response is valid JSON:** `{is_valid_json}`\n\n"
        f"**Fine-tuned response**\n```json\n{response}\n```\n\n"
        f"**Expected response**\n```json\n{expected}\n```"
    ))
