"""
Author: L. Saetta
Date last modified: 2026-10-25
License: MIT
Description: Static tests keeping the ticket-classification system prompts consistent with the classification guide.
"""

import re
from pathlib import Path

import pytest

TICKET_DIRECTORY = Path(__file__).resolve().parents[1] / "ticket-classification"
PROMPT_FILES = ("system_prompt_short.txt", "system_prompt_full.txt")
LABEL_PATTERN = r"\b(?:[A-Z]{2,4}-\d{2})\b"


def guide_category_codes() -> set[str]:
    """Return the category codes defined by the guide's level-3 category headings."""
    guide = (TICKET_DIRECTORY / "classification_guide.md").read_text(encoding="utf-8")
    return set(re.findall(r"^### ([A-Z]{2,4}-\d{2}) ", guide, flags=re.MULTILINE))


def test_guide_defines_the_ten_categories() -> None:
    """Verify that the guide defines the ten allowed categories."""
    assert guide_category_codes() == {"APP-01", "AUTH-01", "DB-01", "DB-02", "DEV-01", "MAIL-01", "MOB-01", "NET-01", "SEC-01", "SRV-01"}


@pytest.mark.parametrize("file_name", PROMPT_FILES)
def test_prompt_matches_guide_labels_and_output_keys(file_name: str) -> None:
    """Verify that a prompt lists exactly the guide's categories, the four severities, and the JSON keys."""
    prompt = (TICKET_DIRECTORY / "prompts" / file_name).read_text(encoding="utf-8")
    assert set(re.findall(LABEL_PATTERN, prompt)) == guide_category_codes()
    assert set(re.findall(r"\bP\d\b", prompt)) == {"P1", "P2", "P3", "P4"}
    for key in ("category", "severity", "summary"):
        assert f'"{key}"' in prompt
    assert "JSON object only" in prompt
