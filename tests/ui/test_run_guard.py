"""Hosted-demo run limits (ui/run_guard.py): off by default, set by environment variables."""

from __future__ import annotations

import threading
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

import axle_studio.ui.run_controller as rc
from axle_studio.ui import run_guard

TODAY = date(2026, 9, 28)


def _fresh_state() -> dict:
    state: dict = {}
    rc.initialise_session(state, today=TODAY)
    return state


def _fake_result(model: str, values: dict) -> SimpleNamespace:
    """A stand-in result: only the fields ``slim_run`` and the run bar read."""

    return SimpleNamespace(
        model=model,
        seed=values["seed"],
        vehicle_count=values["vehicle_count"],
        world_count=values["evaluation_world_count"],
        horizon_start_utc=pd.Timestamp("2026-09-27T23:00Z"),
        study_slot_count=336,
        settings_snapshot={"seed": values["seed"]},
        assumptions=(),
        fleet_world_intervals=pd.DataFrame({"world_id": [0]}),
        plug_in_world_kpis=pd.DataFrame({"world_id": [0]}),
        cost_effect=None,
        weekly_bands=None,
        evidence_kind="illustrative",
    )


def _counting_runner(calls: list):
    def runner(model, values, start, event_presets=()):
        calls.append(values)
        return _fake_result(model, values)

    return runner


def test_no_limits_by_default() -> None:
    assert run_guard.env_limit(run_guard.MAX_CONCURRENT_RUNS_ENV) is None
    assert run_guard.size_refusal({"vehicle_count": 1_000, "evaluation_world_count": 100}) is None
    with run_guard.run_slot() as may_run, run_guard.run_slot() as also:
        assert may_run and also
    state, calls = _fresh_state(), []
    rc.run_full(state, _counting_runner(calls))
    assert len(calls) == 1 and rc.run_status(state).state == "current"


@pytest.mark.parametrize("text", ["0", "-2", "two", "1.5"])
def test_a_misconfigured_limit_fails_loudly(monkeypatch, text) -> None:
    monkeypatch.setenv(run_guard.MAX_CONCURRENT_RUNS_ENV, text)
    with pytest.raises(ValueError, match="AXLE_MAX_CONCURRENT_RUNS"):
        run_guard.env_limit(run_guard.MAX_CONCURRENT_RUNS_ENV)


def test_an_empty_variable_means_no_limit(monkeypatch) -> None:
    monkeypatch.setenv(run_guard.MAX_EV_WEEKS_ENV, " ")
    assert run_guard.env_limit(run_guard.MAX_EV_WEEKS_ENV) is None


def test_a_draft_over_the_size_cap_is_refused_without_running(monkeypatch) -> None:
    monkeypatch.setenv(run_guard.MAX_EV_WEEKS_ENV, "50000")
    state, calls = _fresh_state(), []

    rc.run_full(state, _counting_runner(calls))

    assert calls == []
    assert rc.run_status(state).state == "failed"
    assert rc._log(state).size_capped is True
    message = rc._log(state).error
    assert "limits runs to 50,000 EV-weeks" in message
    assert "1,000 EVs x 100 weeks = 100,000" in message
    assert "download the repository" in message
    # At the cap exactly, the run goes ahead.
    assert run_guard.size_refusal({"vehicle_count": 500, "evaluation_world_count": 100}) is None


def test_the_hosted_size_button_shrinks_the_draft_and_runs_within_the_cap(monkeypatch) -> None:
    monkeypatch.setenv(run_guard.MAX_EV_WEEKS_ENV, "15000")
    state, calls = _fresh_state(), []

    # The default draft (1,000 EVs x 100 weeks) is refused first, exactly as
    # a hosted visitor's first click would be.
    rc.run_full(state, _counting_runner(calls))
    assert calls == [] and rc._log(state).size_capped is True

    rc.run_at_hosted_size(state)
    assert state["draft_values"]["vehicle_count"] == 300
    assert state["draft_values"]["evaluation_world_count"] == 50
    assert state["assumption::vehicle_count"] == 300  # widget key follows the draft
    assert state["run_requested"] is True
    assert not state["draft_error"]

    rc.run_full(state, _counting_runner(calls))

    assert len(calls) == 1
    assert (calls[0]["vehicle_count"], calls[0]["evaluation_world_count"]) == (300, 50)
    assert rc.run_status(state).state == "current"
    assert rc._log(state).size_capped is False


def test_the_hosted_size_button_does_nothing_when_the_cap_is_off() -> None:
    state = _fresh_state()
    before = dict(state["draft_values"])

    rc.run_at_hosted_size(state)

    assert state["draft_values"] == before
    assert state["run_requested"] is False


def test_a_click_while_every_slot_is_taken_is_refused_then_allowed(monkeypatch) -> None:
    monkeypatch.setenv(run_guard.MAX_CONCURRENT_RUNS_ENV, "1")
    state, calls = _fresh_state(), []

    with run_guard.run_slot() as held:
        assert held
        rc.run_full(state, _counting_runner(calls))
    assert calls == []
    assert rc._log(state).error == run_guard.BUSY_MESSAGE
    assert rc.run_history(state) == []

    # The slot was released at the end of the block; the next click runs.
    rc.run_full(state, _counting_runner(calls))
    assert len(calls) == 1 and rc.run_status(state).state == "current"


def test_hosted_run_size_prefers_the_decided_300_by_50_pair() -> None:
    # Mike's decision for the current cap: 300 EVs x 50 weeks fills 15,000
    # EV-weeks exactly (overnight review log, 30 Sep).
    values = {"vehicle_count": 1_000, "evaluation_world_count": 100}
    assert run_guard.hosted_run_size(values, 15_000) == (300, 50)


def test_hosted_run_size_never_offers_more_than_the_drafts_own_size() -> None:
    # A draft already smaller than 300 EVs or 50 weeks is not grown to fill
    # the cap; the button only ever shrinks a refused draft.
    values = {"vehicle_count": 40, "evaluation_world_count": 12}
    assert run_guard.hosted_run_size(values, 15_000) == (40, 12)


def test_hosted_run_size_scales_down_for_a_smaller_cap() -> None:
    values = {"vehicle_count": 1_000, "evaluation_world_count": 100}
    vehicles, weeks = run_guard.hosted_run_size(values, 5_000)
    assert (vehicles, weeks) == (100, 50)
    assert vehicles * weeks <= 5_000


def test_hosted_run_size_never_exceeds_a_cap_smaller_than_one_week_of_one_ev() -> None:
    values = {"vehicle_count": 1_000, "evaluation_world_count": 100}
    vehicles, weeks = run_guard.hosted_run_size(values, 10)
    assert vehicles >= 1 and weeks >= 1
    assert vehicles * weeks <= 10


def test_the_limit_is_shared_across_session_threads(monkeypatch) -> None:
    # Two sessions are two script threads of one server process.
    monkeypatch.setenv(run_guard.MAX_CONCURRENT_RUNS_ENV, "1")
    started, finish = threading.Event(), threading.Event()
    first, second, calls = _fresh_state(), _fresh_state(), []

    def slow_runner(model, values, start, event_presets=()):
        started.set()
        assert finish.wait(10)
        calls.append(values)
        return _fake_result(model, values)

    thread = threading.Thread(target=rc.run_full, args=(first, slow_runner))
    thread.start()
    try:
        assert started.wait(10)
        rc.run_full(second, _counting_runner(calls))
        assert rc._log(second).error == run_guard.BUSY_MESSAGE
    finally:
        finish.set()
        thread.join(10)
    assert len(calls) == 1 and rc.run_status(first).state == "current"
    # A failed run releases its slot too.
    rc.run_full(second, lambda *args, **kwargs: 1 / 0)
    rc.run_full(second, _counting_runner(calls))
    assert len(calls) == 2
