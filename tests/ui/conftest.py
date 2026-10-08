"""Shared UI test setup.

Every UI test starts without the precomputed default run, whatever is in the
checkout's ``data/demo/``: a developer who built the real demo file must not
change what the shell tests see. ``test_demo_run.py`` points the path at a
file it builds itself.

Likewise every UI test starts with the hosted-demo run limits unset
(``ui/run_guard.py``), whatever the shell exports; ``test_run_guard.py``
sets them itself.
"""

from __future__ import annotations

import pytest

from axle_studio.ui import demo_run, run_guard


@pytest.fixture(autouse=True)
def no_precomputed_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(demo_run, "DEMO_RUN_PATH", tmp_path / "no-demo-run.pkl")


@pytest.fixture(autouse=True)
def no_run_limits(monkeypatch) -> None:
    monkeypatch.delenv(run_guard.MAX_CONCURRENT_RUNS_ENV, raising=False)
    monkeypatch.delenv(run_guard.MAX_EV_WEEKS_ENV, raising=False)
