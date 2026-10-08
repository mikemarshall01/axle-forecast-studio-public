"""The precomputed default run: saved once, loaded at session start, never run.

Files here are built by the test into ``tmp_path`` from a small real run
(30 EVs x 4 simulated weeks), never the checkout's real demo file.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import fields
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fixtures.result_fixture import validate_result_v2
from streamlit.testing.v1 import AppTest

import axle_studio.ui.run_controller as rc
from axle_studio.model import assumptions
from axle_studio.model.forecast import ForecastResult, run_forecast_from_assumptions
from axle_studio.ui import demo_run
from axle_studio.ui.registry import PAGES

APP_PATH = Path(__file__).parents[2] / "streamlit_app.py"
BUILT_ON = date(2026, 9, 28)
SMALL = {"vehicle_count": 30, "evaluation_world_count": 4}


@pytest.fixture(scope="module")
def small_values() -> dict:
    return assumptions.editable_defaults() | SMALL


@pytest.fixture(scope="module")
def small_result(small_values) -> ForecastResult:
    return run_forecast_from_assumptions(BUILT_ON, model="action", values=small_values)


@pytest.fixture
def saved(tmp_path, monkeypatch, small_result, small_values) -> Path:
    """A demo file saved honestly with the values the small run used."""

    path = tmp_path / "default_run.pkl"
    demo_run.save_demo_run(small_result, small_values, path, build_seconds=1.0)
    monkeypatch.setattr(demo_run, "DEMO_RUN_PATH", path)
    return path


def _fresh_state(today: date = BUILT_ON) -> dict:
    state: dict = {}
    rc.initialise_session(state, today=today)
    return state


def _never_run(*_args, **_kwargs):
    raise AssertionError("loading the precomputed run must never run the model")


def test_saved_file_round_trips_and_passes_the_result_validator(saved, small_result) -> None:
    loaded = demo_run.load_demo_run(saved)

    assert loaded is not None
    result, stamp = loaded
    validate_result_v2(result)
    assert result.fleet_interval_bands.equals(small_result.fleet_interval_bands)
    assert result.replay_state is not None
    assert stamp["values"]["vehicle_count"] == 30
    assert stamp["model_code_hash"] == demo_run.model_code_hash()
    assert {"git_commit", "package_version", "created_at_utc", "settings_hash"} <= stamp.keys()


def test_session_opens_on_run_0_without_running_the_model(saved, monkeypatch) -> None:
    monkeypatch.setattr(rc, "run_forecast_from_assumptions", _never_run)
    state = _fresh_state()

    [record] = rc.run_history(state)
    assert record.number == 0
    assert record.label == "Run 0 · precomputed · Fleet size 1000 → 30 EVs (+1 more)"
    assert isinstance(rc.active_result(state), ForecastResult)
    assert rc.result_identity(record).startswith("Run 0 · precomputed · 30 EVs · 4 weeks")
    # The draft is the defaults and this file is smaller, so it shows as stale.
    status = rc.run_status(state)
    assert status.state == "stale"
    assert status.caption == "Showing Run 0 (precomputed); draft differs"
    assert "Precomputed" in rc.evidence_badge(record)[1]


def test_a_default_size_file_is_labelled_the_precomputed_default(
    tmp_path, monkeypatch, small_result
) -> None:
    # Saved as if built at the defaults: only the label and staleness are under test.
    path = tmp_path / "default_run.pkl"
    demo_run.save_demo_run(small_result, assumptions.editable_defaults(), path)
    monkeypatch.setattr(demo_run, "DEMO_RUN_PATH", path)
    state = _fresh_state()

    [record] = rc.run_history(state)
    assert record.label == "Run 0 · precomputed default"
    assert rc.run_status(state).state == "current"
    # The next day the start date differs: stale, like any kept run.
    assert rc.unrun_changes(_fresh_state(date(2026, 9, 29))) == ["start date"]


def test_missing_file_opens_on_the_empty_state(tmp_path) -> None:
    assert demo_run.load_demo_run(tmp_path / "absent.pkl") is None
    state = _fresh_state()

    assert rc.run_history(state) == []
    assert rc.run_status(state).state == "no_result"


def test_stamp_from_other_model_code_is_ignored(saved, monkeypatch) -> None:
    monkeypatch.setattr(demo_run, "model_code_hash", lambda: "other code")

    assert demo_run.load_demo_run(saved) is None
    assert rc.run_history(_fresh_state()) == []


def test_other_library_versions_or_a_damaged_file_are_ignored(saved) -> None:
    with saved.open("rb") as file:
        stamp = pickle.load(file)
    with saved.open("wb") as file:
        pickle.dump(stamp | {"pandas": "0.0.0"}, file)
    assert demo_run.load_demo_run(saved) is None

    saved.write_bytes(b"not a pickle")
    assert demo_run.load_demo_run(saved) is None
    assert rc.run_history(_fresh_state()) == []


def test_next_run_is_run_1_and_keeps_run_0_for_compare(saved, small_result) -> None:
    state = _fresh_state()
    rc.run_full(state, runner=lambda model, values, start, presets=(): small_result)

    run_0, run_1 = rc.run_history(state)
    assert (run_0.number, run_1.number) == (0, 1)
    assert run_0.result is None and run_1.result is small_result
    assert run_0.label.startswith("Run 0 · precomputed")
    # Run 1 is labelled against Run 0, the run Compare pairs it with.
    assert run_1.label == "Run 1 · Fleet size 30 → 1000 EVs (+1 more)"
    assert "precomputed" not in rc.result_identity(run_1)


def test_resets_keep_run_0_and_restore_its_inputs(saved) -> None:
    state = _fresh_state()
    rc.reset_to_active_run(state)
    assert state["draft_values"]["vehicle_count"] == 30
    assert rc.run_status(state).state == "current"

    rc.reset_to_defaults(state)
    assert state["draft_values"] == assumptions.editable_defaults()
    assert [record.number for record in rc.run_history(state)] == [0]


def test_app_opens_on_run_0_and_run_makes_run_1(saved, monkeypatch) -> None:
    calls = []

    def spy(start_local_date, *, model, values, event_presets=()):
        calls.append(values)
        return run_forecast_from_assumptions(
            start_local_date, model=model, values=values, event_presets=event_presets
        )

    monkeypatch.setattr(rc, "run_forecast_from_assumptions", spy)
    monkeypatch.setattr(rc, "london_today", lambda: BUILT_ON)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    assert not app.exception
    assert calls == []
    assert any("Run 0 (precomputed)" in caption.value for caption in app.caption)

    app.get_by_key("edit-assumptions").click().run()
    app.get_by_key("reset-to-active").click().run()
    assert calls == []
    app.get_by_key("run-simulation").click().run()

    assert not app.exception
    assert len(calls) == 1 and calls[0]["vehicle_count"] == 30
    assert [record.number for record in rc.run_history(app.session_state)] == [0, 1]
    assert any(caption.value.startswith("Run 1 · 30 EVs") for caption in app.caption)


def test_sessions_share_one_unpickled_result_with_their_own_replay_memo(saved, monkeypatch) -> None:
    loads = []
    real_load = demo_run.load_demo_run

    def counting_load(path=None):
        loads.append(path)
        return real_load(path)

    monkeypatch.setattr(demo_run, "load_demo_run", counting_load)

    first, second = rc.active_result(_fresh_state()), rc.active_result(_fresh_state())

    assert loads == [saved]
    # The frames are the one process-wide copy ...
    assert first.fleet_interval_bands is second.fleet_interval_bands
    assert first.replay_state.units is second.replay_state.units
    # ... but each session memoises replays in its own dict.
    assert first.replay_state.replay_cache is not second.replay_state.replay_cache
    first.replay_state.replay_cache["probe"] = 1
    assert "probe" not in second.replay_state.replay_cache


def test_a_rebuilt_file_is_read_again(saved, small_result, small_values) -> None:
    before = rc.active_result(_fresh_state())
    demo_run.save_demo_run(small_result, small_values | {"seed": 7}, saved)
    stat = saved.stat()
    os.utime(saved, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    [record] = rc.run_history(_fresh_state())
    assert record.values["seed"] == 7
    assert record.result.fleet_interval_bands is not before.fleet_interval_bands


def test_visiting_every_page_leaves_the_shared_result_unchanged(saved, monkeypatch) -> None:
    # The shared result is read by every session, so no page may change it
    # (demo_run.shared_demo_run). Compare every frame with a fresh unpickle
    # after drawing each page and lens once.
    monkeypatch.setattr(rc, "london_today", lambda: BUILT_ON)
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    for page in PAGES:
        # As test_app_shell._go does: switch page the way the top navigation does.
        app._page_hash = next(
            page_hash
            for page_hash, registered in app._registered_pages.items()
            if registered["page_name"] == page.name
        )
        app.run()
        assert not app.exception, page.slug
        for lens in page.lenses:
            app.get_by_key(f"lens::{page.slug}").set_value(lens.name).run()
            assert not app.exception, (page.slug, lens.name)

    shared = rc.active_result(_fresh_state())
    fresh, _ = demo_run.load_demo_run(saved)
    for field in fields(fresh):
        before, after = getattr(fresh, field.name), getattr(shared, field.name)
        if isinstance(before, pd.DataFrame):
            assert before.equals(after), field.name
