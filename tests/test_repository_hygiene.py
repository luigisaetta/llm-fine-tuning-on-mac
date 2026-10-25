"""
Author: L. Saetta
Date last modified: 2026-10-25
License: MIT
Description: Static tests for notebook output hygiene and the pinned dependency and model contracts.
"""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_QWEN3_1_7B_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def tracked_notebooks() -> list[Path]:
    """Return every notebook tracked by Git, or every notebook outside ignored folders without Git."""
    try:
        result = subprocess.run(["git", "ls-files", "*.ipynb"], cwd=ROOT, capture_output=True, text=True, check=True)
        return [ROOT / name for name in result.stdout.splitlines()]
    except (OSError, subprocess.CalledProcessError):
        return [path for path in ROOT.rglob("*.ipynb") if ".ipynb_checkpoints" not in path.parts and "artifacts" not in path.parts]


def read_requirements(path: Path) -> dict[str, str]:
    """Return requirement specifiers keyed by normalized package name."""
    requirements = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, specifier = re.match(r"([A-Za-z0-9_.-]+)(.*)", line).groups()
            requirements[name.lower().replace("_", "-")] = specifier
    return requirements


def test_committed_notebooks_have_no_outputs() -> None:
    """Verify that notebooks store neither outputs nor execution counts."""
    notebooks = tracked_notebooks()
    assert notebooks
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell.get("outputs", []) == [], f"{path.relative_to(ROOT)} cell {index} has outputs"
                assert cell.get("execution_count") is None, f"{path.relative_to(ROOT)} cell {index} has an execution count"


def test_requirement_minimums_support_notebook_apis() -> None:
    """Verify minimums for from_pretrained(dtype=...) and SFTConfig(completion_only_loss=...)."""
    for path in (ROOT / "requirements.txt", ROOT / "ticket-classification" / "requirements.txt"):
        requirements = read_requirements(path)
        assert ">=4.56" in requirements["transformers"], path
        assert ">=0.17" in requirements["trl"], path
    assert "numpy" in read_requirements(ROOT / "requirements.txt")


def test_constraints_pin_every_direct_requirement() -> None:
    """Verify that the macOS constraints file pins every root requirement exactly."""
    requirements = read_requirements(ROOT / "requirements.txt")
    constraints = read_requirements(ROOT / "constraints-macos.txt")
    assert set(constraints) == set(requirements)
    assert all(re.fullmatch(r"==[0-9][0-9A-Za-z.+-]*", specifier) for specifier in constraints.values())


def test_readme_download_pins_model_revision() -> None:
    """Verify that the documented model download and its specification use the pinned revision."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    specification = (ROOT / "specs" / "003-qwen3-1.7b-model-acquisition.md").read_text(encoding="utf-8")
    assert f"--revision {PINNED_QWEN3_1_7B_REVISION}" in readme
    assert PINNED_QWEN3_1_7B_REVISION in specification
