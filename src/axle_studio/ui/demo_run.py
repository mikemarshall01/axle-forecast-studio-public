"""The precomputed default run: saving it once, and loading it at app start-up.

A full default run (1,000 EVs x 100 simulated weeks) takes minutes, so the
interview demo opens on a result saved earlier by ``scripts/build_demo_run.py``
(lead ruling logged in docs/plans/2026-09-30-overnight-review-log.md).
Loading a saved result is not a run: AGENTS.md's rule that a full Monte Carlo
starts only from an explicit Run click still holds, and nothing here calls the
model.

File format, and the alternatives rejected:

* One pickle file holding two objects in sequence: a small ``stamp`` dict,
  then the frozen ``ForecastResult``. Pickle round-trips every field exactly
  (DataFrames with their dtypes and index, the replay state, the assumption
  records) with no per-field code; Parquet or JSON would need a converter for
  each of the ~50 fields and would drift as fields are added. Pickle is only
  safe between the same library versions, so the stamp records the NumPy and
  pandas versions (pinned by ``uv.lock``) and the loader refuses a mismatch.
* The stamp is read first, so an incompatible file is rejected without
  unpickling the large result.
* Compatibility is a hash of the model package's source (``src/axle_studio/
  model/*.py``), not the result's field names alone: a saved result is only a
  truthful "default run" if today's model code would produce the same numbers.
  UI changes do not invalidate it; any model change does, and the app then
  falls back to the ordinary empty state until the file is rebuilt.
* The file is local, ignored by Git and written only by our own script, so
  unpickling it is no more trusting than importing the code beside it.
* The app reads it through ``shared_demo_run``: unpickled once per server
  process with ``st.cache_resource`` and shared by every session, rather than
  once per session (st.session_state). A hosted demo with several visitors
  then holds one copy of the large result in memory. Sharing is safe because
  the pages only read the result's frames; the one thing views write to, the
  replay state's memo dict, is given fresh to each session.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import axle_studio
from axle_studio.model.forecast import ForecastResult

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_RUN_PATH = REPO_ROOT / "data" / "demo" / "default_run.pkl"
"""Where the build script writes the file and the app looks for it (ignored by Git)."""

FORMAT_VERSION = 1
"""Bumped if the file layout (stamp then result) ever changes."""

_MODEL_DIR = Path(__file__).resolve().parents[1] / "model"


def model_code_hash() -> str:
    """SHA-256 of every model source file, in name order: the result's compatibility key."""

    digest = hashlib.sha256()
    for path in sorted(_MODEL_DIR.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def settings_hash(model: str, values: dict[str, Any], event_presets: tuple[str, ...]) -> str:
    """SHA-256 of the run's inputs (model, editable values, event presets), for the stamp."""

    text = json.dumps(
        {"model": model, "values": values, "event_presets": list(event_presets)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _git_commit() -> str:
    """The checkout's commit, recorded for people reading the stamp; "unknown" outside Git."""

    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _library_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}


def save_demo_run(
    result: ForecastResult,
    values: dict[str, Any],
    path: Path = DEMO_RUN_PATH,
    *,
    build_seconds: float | None = None,
) -> dict[str, Any]:
    """Write ``result`` and its stamp to ``path``; return the stamp.

    ``values`` are the editable assumption values the run used, by record
    name; the app compares them with the defaults to label the run and with
    the draft to mark it stale. ``build_seconds`` is the run's wall time.
    """

    stamp = {
        "format_version": FORMAT_VERSION,
        "model_code_hash": model_code_hash(),
        **_library_versions(),
        "package_version": axle_studio.__version__,
        "git_commit": _git_commit(),
        "model": result.model,
        "values": dict(values),
        "event_presets": [],
        "settings_hash": settings_hash(result.model, dict(values), ()),
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "build_seconds": build_seconds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(stamp, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    return stamp


def stamp_matches(stamp: dict[str, Any]) -> bool:
    """True when a stamp was written by today's model code and library versions."""

    return (
        stamp.get("format_version") == FORMAT_VERSION
        and stamp.get("model_code_hash") == model_code_hash()
        and all(stamp.get(name) == version for name, version in _library_versions().items())
    )


def load_demo_run(path: Path | None = None) -> tuple[ForecastResult, dict[str, Any]] | None:
    """Return ``(result, stamp)`` from a compatible file at ``path`` (default ``DEMO_RUN_PATH``).

    Returns ``None`` when the file is missing, unreadable or from other code: the app
    then opens on the ordinary empty state, with no error, because a missing
    demo file is a normal checkout (it is never committed).
    """

    path = DEMO_RUN_PATH if path is None else path
    try:
        with path.open("rb") as file:
            stamp = pickle.load(file)
            if not isinstance(stamp, dict) or not stamp_matches(stamp):
                return None
            result = pickle.load(file)
    # Broad on purpose: a truncated or foreign file must never stop the app
    # opening; the worst case is the empty state and an explicit Run.
    except Exception:
        return None
    if not isinstance(result, ForecastResult):
        return None
    return result, stamp


# Keyed on the file's modification time and the model code hash as well as its
# path, so a rebuilt file, or a model edit while the server runs, is re-checked
# instead of served from memory; max_entries=1 keeps only the current file.
@st.cache_resource(show_spinner=False, max_entries=1)
def _load_once(path: str, modified_ns: int | None, code_hash: str):
    return load_demo_run(Path(path))


def shared_demo_run() -> tuple[ForecastResult, dict[str, Any]] | None:
    """``load_demo_run()`` for one session, unpickled once per server process.

    Returns ``(result, stamp)`` or ``None``, exactly as ``load_demo_run``. The
    result's frames and arrays are the process-wide copy, shared read-only
    by every session; callers must not change them (no view does: they
    filter or ``.copy()`` before adding columns). The result is a shallow copy
    whose ``replay_state`` has its own empty ``replay_cache``: views memoise
    replays in that dict, and one dict shared by concurrent sessions could
    have an entry evicted by another session between its check and its read.
    """

    path = DEMO_RUN_PATH
    try:
        modified_ns = path.stat().st_mtime_ns
    except OSError:
        modified_ns = None
    loaded = _load_once(str(path), modified_ns, model_code_hash())
    if loaded is None:
        return None
    result, stamp = loaded
    if result.replay_state is not None:
        result = replace(result, replay_state=replace(result.replay_state, replay_cache={}))
    return result, stamp


__all__ = [
    "DEMO_RUN_PATH",
    "FORMAT_VERSION",
    "load_demo_run",
    "model_code_hash",
    "save_demo_run",
    "settings_hash",
    "shared_demo_run",
    "stamp_matches",
]
