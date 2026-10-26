"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Deterministic, model-free comparison of two ticket splits: severity prevalence, ticket length,
category-conditioned severity, severity wording, and cross-split duplicates.
"""

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from llm_fine_tuning.tickets.labels import OUTPUT_KEYS, SEVERITIES

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
TEXT_FEATURES = ("characters", "words", "sentences")


def tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens, keeping hyphenated words together.

    Args:
        text: Ticket text.

    Returns:
        The tokens.
    """
    return TOKEN_PATTERN.findall(text.lower())


def extract_text_features(text: str) -> dict[str, int]:
    """Return simple, interpretable length features of a ticket.

    Args:
        text: Ticket text.

    Returns:
        Character, word, and sentence counts.
    """
    return {
        "characters": len(text),
        "words": len(tokenize(text)),
        "sentences": len([part for part in re.split(r"[.!?]+", text) if part.strip()]),
    }


def load_labeled_tickets(path: Path, split_name: str) -> list[dict]:
    """Load one JSONL split with the fields needed for the analysis.

    Args:
        path: JSONL file.
        split_name: Name used in records and error messages.

    Returns:
        One dictionary per record with split, ticket, category, severity, and text features.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If a record is malformed or the split is empty.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing required dataset: {path}")
    records = []
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            messages = json.loads(line).get("messages")
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError(f"{split_name} line {line_number} must contain exactly two messages.")
            if [message.get("role") for message in messages] != ["user", "assistant"]:
                raise ValueError(f"{split_name} line {line_number} has an unexpected message order.")
            ticket = messages[0].get("content", "").strip()
            target = json.loads(messages[1].get("content", ""))
            if not ticket or not isinstance(target, dict) or set(target) != OUTPUT_KEYS:
                raise ValueError(f"{split_name} line {line_number} has an invalid ticket or target schema.")
            if target["severity"] not in SEVERITIES:
                raise ValueError(f"{split_name} line {line_number} has unsupported severity {target['severity']!r}.")
            if not target["category"] or not target["summary"]:
                raise ValueError(f"{split_name} line {line_number} has an empty target value.")
            records.append({"split": split_name, "ticket": ticket, "category": target["category"], "severity": target["severity"], **extract_text_features(ticket)})
    if not records:
        raise ValueError(f"{split_name} is empty.")
    return records


def cross_split_duplicates(first: list[dict], second: list[dict]) -> set[str]:
    """Return ticket texts that appear in both splits.

    Args:
        first: Records of one split.
        second: Records of the other split.

    Returns:
        The shared ticket texts.
    """
    return {record["ticket"] for record in first} & {record["ticket"] for record in second}


def severity_distribution(records: list[dict]) -> dict[str, float]:
    """Return the share of each severity, in P1-to-P4 order.

    Args:
        records: Non-empty list of records.

    Returns:
        Severity to proportion.
    """
    counts = Counter(record["severity"] for record in records)
    return {severity: counts[severity] / len(records) for severity in SEVERITIES}


def total_variation_distance(first: dict[str, float], second: dict[str, float]) -> float:
    """Return the total-variation distance between two severity distributions (0 identical, 1 disjoint).

    Args:
        first: Severity proportions.
        second: Severity proportions.

    Returns:
        The distance.
    """
    return 0.5 * sum(abs(second[severity] - first[severity]) for severity in SEVERITIES)


def severity_prevalence_rows(reference: list[dict], comparison: list[dict]) -> list[list[object]]:
    """Return count, share, and percentage-point difference of each severity in two splits.

    Args:
        reference: Records of the reference split (validation).
        comparison: Records of the compared split (test).

    Returns:
        One table row per severity.
    """
    reference_shares, comparison_shares = severity_distribution(reference), severity_distribution(comparison)
    rows = []
    for severity in SEVERITIES:
        delta = 100 * (comparison_shares[severity] - reference_shares[severity])
        rows.append([
            severity,
            sum(record["severity"] == severity for record in reference), f"{reference_shares[severity]:.1%}",
            sum(record["severity"] == severity for record in comparison), f"{comparison_shares[severity]:.1%}",
            f"{delta:+.1f} pp",
        ])
    return rows


def text_feature_rows(named_splits: dict[str, list[dict]]) -> list[list[object]]:
    """Return mean, median, minimum, and maximum of each length feature per split.

    The median is the upper middle value for an even number of records.

    Args:
        named_splits: Split name to records.

    Returns:
        One table row per split and feature.
    """
    rows = []
    for split_name, records in named_splits.items():
        for feature in TEXT_FEATURES:
            values = sorted(record[feature] for record in records)
            rows.append([split_name, feature, f"{sum(values) / len(values):.1f}", values[len(values) // 2], values[0], values[-1]])
    return rows


def conditional_severity_shift(reference: list[dict], comparison: list[dict]) -> tuple[list[str], list[list[float]]]:
    """Return the change in P(severity | category) from the reference to the compared split.

    Args:
        reference: Records of the reference split.
        comparison: Records of the compared split.

    Returns:
        The sorted categories and, per category, the P1-to-P4 probability differences (compared minus reference).
    """

    def conditional(records: list[dict], category: str) -> dict[str, float]:
        category_records = [record for record in records if record["category"] == category]
        return severity_distribution(category_records) if category_records else {severity: 0.0 for severity in SEVERITIES}

    categories = sorted({record["category"] for record in reference + comparison})
    matrix = []
    for category in categories:
        reference_shares, comparison_shares = conditional(reference, category), conditional(comparison, category)
        matrix.append([comparison_shares[severity] - reference_shares[severity] for severity in SEVERITIES])
    return categories, matrix


def top_terms(records: list[dict], severity: str, limit: int) -> list[str]:
    """Return the most frequent tokens of one severity, breaking ties alphabetically.

    Args:
        records: Records of one split.
        severity: Severity to analyze.
        limit: Number of terms.

    Returns:
        The top terms.
    """
    counts = Counter()
    for record in records:
        if record["severity"] == severity:
            counts.update(tokenize(record["ticket"]))
    return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def severity_term_overlap(reference: list[dict], comparison: list[dict], limit: int) -> tuple[list[list[object]], dict[str, float]]:
    """Compare the top terms of each severity between two splits with the Jaccard overlap.

    Args:
        reference: Records of the reference split.
        comparison: Records of the compared split.
        limit: Number of top terms per severity and split.

    Returns:
        Table rows and the overlap per severity.
    """
    rows, overlaps = [], {}
    for severity in SEVERITIES:
        reference_terms, comparison_terms = top_terms(reference, severity, limit), top_terms(comparison, severity, limit)
        union = set(reference_terms) | set(comparison_terms)
        overlaps[severity] = len(set(reference_terms) & set(comparison_terms)) / len(union) if union else 1.0
        rows.append([severity, ", ".join(reference_terms), ", ".join(comparison_terms), f"{overlaps[severity]:.1%}"])
    return rows, overlaps


def diagnostic_rows(reference, comparison, categories, shift_matrix, overlaps) -> list[list[object]]:
    """Summarize the largest observed shifts as an explicit, non-causal checklist.

    Args:
        reference: Records of the reference split.
        comparison: Records of the compared split.
        categories: Categories returned by :func:`conditional_severity_shift`.
        shift_matrix: Matrix returned by :func:`conditional_severity_shift`.
        overlaps: Overlaps returned by :func:`severity_term_overlap`.

    Returns:
        Table rows of diagnostic, observed value, and interpretation.
    """
    reference_shares, comparison_shares = severity_distribution(reference), severity_distribution(comparison)
    largest_severity = max(SEVERITIES, key=lambda severity: abs(comparison_shares[severity] - reference_shares[severity]))
    largest_category = max(
        ((abs(delta), categories[row], SEVERITIES[column], delta) for row, values in enumerate(shift_matrix) for column, delta in enumerate(values)),
        key=lambda item: (item[0], item[1], item[2]),
    )
    mean_overlap = sum(overlaps.values()) / len(overlaps)
    severity_delta = 100 * (comparison_shares[largest_severity] - reference_shares[largest_severity])
    return [
        ["Exact duplicate ticket texts", len(cross_split_duplicates(reference, comparison)), "Duplicates can inflate validation results; zero supports a genuinely separate test set."],
        ["Severity-distribution TVD", f"{total_variation_distance(reference_shares, comparison_shares):.3f}", "Higher values indicate stronger aggregate label-prevalence shift."],
        ["Largest severity share shift", f"{largest_severity}: {severity_delta:+.1f} pp", "A label is more or less common in test than in validation."],
        ["Largest category-conditioned shift", f"{largest_category[1]} / {largest_category[2]}: {100 * largest_category[3]:+.1f} pp", "The category-to-severity relationship differs between splits."],
        ["Mean top-term overlap", f"{mean_overlap:.1%}", "Lower overlap indicates different prominent lexical cues by severity."],
    ]


def plot_severity_distribution(named_splits: dict[str, list[dict]]) -> None:
    """Plot the severity shares of two splits as grouped bars.

    Args:
        named_splits: Exactly two split names mapped to their records.
    """
    (first_name, first), (second_name, second) = named_splits.items()
    first_shares, second_shares = severity_distribution(first), severity_distribution(second)
    positions, width = range(len(SEVERITIES)), 0.36
    _, axis = plt.subplots(figsize=(8, 4))
    axis.bar([position - width / 2 for position in positions], [first_shares[severity] for severity in SEVERITIES], width=width, label=first_name)
    axis.bar([position + width / 2 for position in positions], [second_shares[severity] for severity in SEVERITIES], width=width, label=second_name)
    axis.set_xticks(list(positions), SEVERITIES)
    axis.set_ylabel("Share of records")
    axis.set_title("Severity-label distribution by split")
    axis.yaxis.set_major_formatter("{x:.0%}")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    plt.show()


def plot_ticket_length(named_splits: dict[str, list[dict]]) -> None:
    """Plot the words-per-ticket distribution of each split as box plots.

    Args:
        named_splits: Split names mapped to their records.
    """
    _, axis = plt.subplots(figsize=(8, 4))
    axis.boxplot([[record["words"] for record in records] for records in named_splits.values()], tick_labels=list(named_splits), showfliers=True)
    axis.set_ylabel("Words per ticket")
    axis.set_title("Ticket-length distribution by split")
    axis.grid(axis="y", alpha=0.25)
    plt.show()


def plot_conditional_shift(categories: list[str], shift_matrix: list[list[float]]) -> None:
    """Plot the change in P(severity | category) as a diverging heatmap.

    Args:
        categories: Row labels.
        shift_matrix: Probability differences, one row per category.
    """
    figure, axis = plt.subplots(figsize=(7, max(4, len(categories) * 0.42)))
    image = axis.imshow(shift_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(SEVERITIES)), SEVERITIES)
    axis.set_yticks(range(len(categories)), categories)
    axis.set_title("Change in P(severity | category): test minus validation")
    figure.colorbar(image, ax=axis, label="Probability difference")
    plt.show()
