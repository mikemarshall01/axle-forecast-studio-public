"""Execute the concept notebooks end to end (plan 2026-09-29 §10).

Each concept notebook imports the real package functions and ends with sanity asserts, so running
it is the check that its code, charts and numbers still run against the model as built. Notebook
01 (the full model walkthrough) is included. Executing a notebook checks its code, not its
prose. Slow: deselect with ``-m "not slow"``.
"""

import os
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
CONCEPT_NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("0*.ipynb"))
# Per cell, in seconds. Each notebook targets well under 30 s in total; the margin absorbs a
# slow machine without letting a hung kernel stall the suite.
CELL_TIMEOUT_S = 300


@pytest.mark.slow
@pytest.mark.parametrize("path", CONCEPT_NOTEBOOKS, ids=lambda path: path.stem)
def test_concept_notebook_executes(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The kernel is a separate process. Putting this checkout's src first on its path makes it
    # import the code under test, not whatever checkout a shared editable install points at.
    source_dir = str(NOTEBOOK_DIR.parent / "src")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, (source_dir, os.environ.get("PYTHONPATH"))))
    )
    notebook = nbformat.read(path, as_version=4)
    # "python3" is the interpreter running pytest (the project environment), so the test does not
    # depend on the locally registered "axle-forecast-studio" kernel existing.
    executor = ExecutePreprocessor(timeout=CELL_TIMEOUT_S, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(NOTEBOOK_DIR)}})
