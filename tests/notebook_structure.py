"""
Author: L. Saetta
Date last modified: 2026-10-26
License: MIT
Description: Shared checks for the didactic notebook structure defined in specification 022.
"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_notebook(relative_path: str) -> dict:
    """Return a notebook parsed from JSON."""
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def code_sources(notebook: dict) -> list[str]:
    """Return the source of every code cell."""
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def assert_didactic_structure(notebook: dict, settings_marker: str) -> str:
    """Check the shared layout and return the concatenated code for notebook-specific assertions.

    The notebook must be nbformat 4 without outputs, start with a Markdown title, have a Settings section whose
    next code cell contains ``settings_marker``, define no functions or classes, import only from the project
    package and standard notebook libraries, and precede every code cell with a Markdown explanation.
    """
    cells = notebook["cells"]
    assert notebook["nbformat"] == 4
    assert cells[0]["cell_type"] == "markdown" and "".join(cells[0]["source"]).startswith("# Demo ")
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            assert cell.get("outputs", []) == [] and cell.get("execution_count") is None, f"cell {index} has outputs"
            assert index == 1 or cells[index - 1]["cell_type"] == "markdown", f"code cell {index} has no explanation"
    settings_index = next(i for i, cell in enumerate(cells) if cell["cell_type"] == "markdown" and "Settings" in "".join(cell["source"]).splitlines()[0])
    assert settings_marker in "".join(cells[settings_index + 1]["source"])
    sources = code_sources(notebook)
    for source in sources:
        tree = ast.parse(source)
        assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module.split(".")[0] in {"llm_fine_tuning", "transformers"}, node.module
            elif isinstance(node, ast.Import):
                assert all(alias.name in {"torch"} for alias in node.names)
    assert any("llm_fine_tuning" in source for source in sources)
    return "\n".join(sources)
