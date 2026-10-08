"""Unit tests of the run lifecycle on a plain dict standing in for session state."""

from __future__ import annotations

import math
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

import axle_studio.ui.run_controller as rc
from axle_studio.model import assumptions
from axle_studio.model.forecast import ForecastResult
from axle_studio.model.summaries import RunSummary

TODAY = date(2026, 9, 28)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch) -> None:
    monkeypatch.setattr(rc, "london_today", lambda: TODAY)


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
        calls.append((model, values, start))
        return _fake_result(model, values)

    return runner


def _edit(state: dict, name: str, value: object) -> None:
    """What a dialog widget does: write its key, then call its on_change callback."""

    state[rc.WIDGET_PREFIX + name] = value
    rc.commit_assumption_edit(state, name)


def test_fresh_session_defaults_to_the_action_model_and_the_explicit_run_shape() -> None:
    state = _fresh_state()

    assert state["model_mode"] == rc.MODEL_ACTION
    assert state["draft_start_date"] == TODAY
    # The draft is every editable record at its default (model/assumptions.py).
    assert state["draft_values"] == assumptions.editable_defaults()
    assert (state["draft_values"]["vehicle_count"], state["draft_values"]["seed"]) == (1_000, 42)
    assert state["draft_errors"] == {} and state["draft_error"] is None
    assert state["assumption::vehicle_count"] == 1_000
    assert rc.run_status(state) == rc.RunStatus("no_result", None, "gray", "No result yet")


def test_only_the_two_explicit_models_are_selectable() -> None:
    # The same ids run_forecast, ForecastResult.model and RunSummary.model use.
    assert rc.MODELS == ("action", "no_action")
    assert rc.MODEL_LABELS == {"action": "Smart charging", "no_action": "No action"}


def test_six_worded_states() -> None:
    state = _fresh_state()
    assert rc.run_status(state).state == "no_result"

    assert rc.run_status(state, running=True) == rc.RunStatus(
        "running", "Running", "gray", "1,000 EVs × 100 simulated weeks…"
    )

    rc.run_full(state, _counting_runner([]))
    assert rc.run_status(state) == rc.RunStatus(
        "current",
        "Current",
        "green",
        "Run 1 · 1,000 EVs · 100 weeks · seed 42 · from Mon 28 Sep",
    )

    _edit(state, "seed", 7)
    assert rc.run_status(state) == rc.RunStatus(
        "stale", "Stale · 1 change", "orange", "Showing Run 1; draft differs"
    )

    state["run_log"].error = "boom"
    assert rc.run_status(state) == rc.RunStatus(
        "failed", "Run failed", "red", "Still showing Run 1"
    )

    _edit(state, "vehicle_count", 1_001)
    status = rc.run_status(state)
    assert (status.state, status.chip, status.colour) == ("invalid", "Draft invalid", "red")
    assert "Fleet size must be between 1 and 1000" in status.caption


def test_edits_resets_and_model_switch_never_call_the_runner() -> None:
    calls: list = []
    state = _fresh_state()
    rc.run_full(state, _counting_runner(calls))
    result = rc.active_result(state)

    _edit(state, "home_charge_efficiency", 0.85)
    state["model_mode"] = rc.MODEL_NO_ACTION
    rc.model_changed(state)
    assert rc.unrun_changes(state) == ["model", "Home charge efficiency"]

    rc.reset_to_defaults(state)
    # Reset to defaults restores the values but keeps the chosen model.
    assert state["model_mode"] == rc.MODEL_NO_ACTION
    assert state["assumption::home_charge_efficiency"] == 0.92
    assert rc.unrun_changes(state) == ["model"]

    rc.reset_to_active_run(state)
    assert state["model_mode"] == rc.MODEL_ACTION
    assert rc.unrun_changes(state) == []

    rc.switch_to_action_model(state)
    assert len(calls) == 1
    assert rc.active_result(state) is result


def test_invalid_draft_is_never_run() -> None:
    calls: list = []
    state = _fresh_state()
    _edit(state, "vehicle_count", 1_001)

    assert state["draft_errors"] == {"vehicle_count": "must be between 1 and 1000"}
    rc.run_full(state, _counting_runner(calls))
    assert calls == []


def test_failed_run_keeps_the_previous_result() -> None:
    state = _fresh_state()
    rc.run_full(state, _counting_runner([]))
    previous = rc.active_result(state)

    def failing(model, values, start, event_presets=()):
        raise RuntimeError("no memory")

    rc.run_full(state, failing)

    assert rc.active_result(state) is previous
    assert len(rc.run_history(state)) == 1
    assert state["run_log"].error == "no memory"


def test_history_keeps_three_labels_runs_and_slims_older_results() -> None:
    state = _fresh_state()
    rc.run_full(state, _counting_runner([]))
    _edit(state, "vehicle_count", 60)
    rc.run_full(state, _counting_runner([]))
    state["model_mode"] = rc.MODEL_NO_ACTION
    rc.run_full(state, _counting_runner([]))
    _edit(state, "home_charge_efficiency", 0.85)
    rc.run_full(state, _counting_runner([]))

    history = rc.run_history(state)
    assert [record.label for record in history] == [
        # Each label names the change from the previous run (design 4.4).
        "Run 2 · Fleet size 1000 → 60 EVs",
        "Run 3 · smart charging → no action",
        "Run 4 · Home charge efficiency 0.92 → 0.85 efficiency fraction",
    ]
    assert [record.result is None for record in history] == [True, True, False]
    # Every kept run is a slim summary Compare can read (decision 0004 item 35).
    assert all(isinstance(record, RunSummary) for record in history)
    assert [record.model for record in history] == ["action", "no_action", "no_action"]
    assert state["run_log"].notice == "Run 1 · base dropped: Compare keeps the last three runs."


def test_midnight_moves_the_draft_date_and_marks_the_result_stale_without_running() -> None:
    calls: list = []
    state = _fresh_state()
    rc.run_full(state, _counting_runner(calls))

    rc.refresh_start_date(state, today=date(2026, 9, 29))

    assert state["draft_start_date"] == date(2026, 9, 29)
    assert rc.unrun_changes(state) == ["start date"]
    assert len(calls) == 1


def test_retired_model_left_by_a_code_reload_falls_back_to_the_default() -> None:
    state = _fresh_state()
    state["model_mode"] = "compact_no_action"

    rc.initialise_session(state, today=TODAY)

    assert state["model_mode"] == rc.MODEL_ACTION


def test_widget_keys_are_rewritten_from_the_draft() -> None:
    # Regression for the fleet-size slider that showed 200 while the run used
    # 1,000: the draft owns the value and the widget key is copied from it.
    state = _fresh_state()
    _edit(state, "vehicle_count", 200)
    del state["assumption::seed"]  # Streamlit drops keys of widgets not on screen
    state["assumption::vehicle_count"] = 1_000  # an old value resurfacing

    rc.sync_widgets_from_draft(state)

    assert state["assumption::vehicle_count"] == 200
    assert state["assumption::seed"] == 42


def test_an_edit_keeps_draft_values_whose_widgets_are_not_on_screen() -> None:
    # The dialog shows one tab at a time; a callback must not drop the others.
    state = _fresh_state()
    del state["assumption::seed"]

    _edit(state, "home_charging_power_kw", 3.6)

    assert state["draft_values"]["seed"] == 42
    assert state["draft_values"]["home_charging_power_kw"] == 3.6


def test_an_edit_ignores_stale_keys_of_fields_in_hidden_groups() -> None:
    # Review of ecacff9, blocking 1: a hidden group's widget key can still
    # hold an old value; editing another field must not copy it back.
    state = _fresh_state()
    _edit(state, "seed", 7)
    state["assumption::seed"] = 42  # stale key left by the hidden Simulation group

    _edit(state, "home_charging_power_kw", 3.6)

    assert state["draft_values"]["seed"] == 7
    assert state["draft_values"]["home_charging_power_kw"] == 3.6


def test_identity_uses_singular_week_for_one_world() -> None:
    state = _fresh_state()
    _edit(state, "evaluation_world_count", 1)
    rc.run_full(state, _counting_runner([]))

    assert "· 1 week ·" in rc.result_identity(rc.latest_run(state))


def _evidence_runner(model: str, values: dict, start, event_presets=()) -> SimpleNamespace:
    result = _fake_result(model, values)
    result.evidence_kind = "illustrative_synthetic" if model == "action" else "illustrative"
    return result


def test_identity_is_short_and_evidence_is_a_badge_that_explains_a_differing_word() -> None:
    # Polish plan G7/G9: the identity line stays under 80 characters and the
    # evidence class moves to a badge. Goal review V6: the no-action model
    # has no price channel, so its evidence word genuinely differs from the
    # action model's; each badge must read as self-explanatory on its own.
    no_action_state = _fresh_state()
    no_action_state["model_mode"] = rc.MODEL_NO_ACTION
    rc.run_full(no_action_state, _evidence_runner)
    no_action_record = rc.latest_run(no_action_state)
    no_action_badge, _ = rc.evidence_badge(no_action_record)
    assert no_action_badge == "Illustrative"

    action_state = _fresh_state()
    action_state["model_mode"] = rc.MODEL_ACTION
    rc.run_full(action_state, _evidence_runner)
    action_record = rc.latest_run(action_state)
    action_badge, action_help = rc.evidence_badge(action_record)
    assert "synthetic" in action_badge.lower()
    assert "day-ahead" in action_help  # says *what* is synthetic, not a bare word
    for record in (no_action_record, action_record):
        assert len(rc.result_identity(record)) <= 80
        assert "illustrative" not in rc.result_identity(record)


def test_run_full_consumes_the_click_just_before_the_model() -> None:
    state = _fresh_state()
    state["run_requested"] = True
    seen: list[bool] = []

    def runner(model, values, start, event_presets=()):
        seen.append(state["run_requested"])
        return _fake_result(model, values)

    rc.run_full(state, runner)

    assert seen == [False]  # cleared before the model, so no rerun can repeat it


def test_invalid_draft_drops_a_pending_click() -> None:
    state = _fresh_state()
    _edit(state, "vehicle_count", 1_001)
    state["run_requested"] = True

    rc.run_full(state, _counting_runner([]))

    assert state["run_requested"] is False


def test_run_model_calls_run_forecast_and_returns_a_forecast_result() -> None:
    # A real, tiny run of each model through the same path the Run button uses.
    values = assumptions.editable_defaults() | {
        "vehicle_count": 12,
        "evaluation_world_count": 2,
        "home_charging_power_kw": 3.6,
    }
    for model in rc.MODELS:
        result = rc.run_model(model, values, TODAY)

        assert isinstance(result, ForecastResult)
        assert result.model == model
        assert (result.vehicle_count, result.world_count) == (12, 2)
        # The result carries the records it ran with, edited values included.
        records = {record.name: record for record in result.assumptions}
        assert records["home_charging_power_kw"].value == 3.6
        assert "cnz_median_plug_in_soc_percent" in records


def test_unknown_model_is_refused_before_any_input_is_loaded() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        rc.run_model("compact_no_action", assumptions.editable_defaults(), TODAY)


def test_run_records_keep_the_values_and_date_each_run_used() -> None:
    state = _fresh_state()
    _edit(state, "seed", 7)
    calls: list = []
    rc.run_full(state, _counting_runner(calls))

    record = rc.latest_run(state)
    assert record.values == state["draft_values"] == calls[0][1]
    assert record.start_local_date == TODAY == calls[0][2]
    # A later edit does not change the kept record.
    _edit(state, "seed", 8)
    assert record.values["seed"] == 7


def test_run_label_counts_extra_changes_and_names_an_unchanged_rerun() -> None:
    state = _fresh_state()
    rc.run_full(state, _counting_runner([]))
    rc.run_full(state, _counting_runner([]))
    _edit(state, "home_charging_power_kw", 3.6)
    _edit(state, "seed", 7)
    rc.run_full(state, _counting_runner([]))

    labels = [record.label for record in rc.run_history(state)]
    assert labels == [
        "Run 1 · base",
        "Run 2 · same inputs",
        "Run 3 · Home charging power 7 → 3.6 kW (+1 more)",
    ]


def test_a_choice_change_reads_as_words_without_the_record_description() -> None:
    # The commitment rule's id and unit text once showed as "Commitment rule
    # fixed_share → newsvendor choice (fixed share or newsvendor)".
    changes = rc._value_changes(
        {"trading.commitment_rule": "fixed_share"}, {"trading.commitment_rule": "newsvendor"}
    )
    assert changes == ["Commitment rule fixed share → newsvendor"]


# --- Supplier tab: unset terms, the reward toggle and the price curve (lane S2) --


def test_an_unset_term_never_makes_the_result_stale() -> None:
    # NaN never equals NaN, so a plain != once called an unset platform fee
    # "changed" on every rerun (supplier contract v1 §8).
    state = _fresh_state()
    rc.run_full(state, _counting_runner([]))
    fee = "supplier.platform_fee_gbp_per_ev_per_month"
    assert math.isnan(state["draft_values"][fee])
    assert rc.unrun_changes(state) == []
    assert rc.run_status(state).state == "current"


def test_an_empty_field_is_unset_and_a_toggle_is_zero_or_one() -> None:
    state = _fresh_state()
    fee, mode = "supplier.platform_fee_gbp_per_ev_per_month", "supplier.customer_reward_mode"
    # The widgets hold an empty number field and a toggle's bool.
    assert state[rc.WIDGET_PREFIX + fee] is None
    assert state[rc.WIDGET_PREFIX + mode] is False

    state[rc.WIDGET_PREFIX + fee] = 2.5
    rc.commit_assumption_edit(state, fee)
    assert state["draft_values"][fee] == 2.5
    state[rc.WIDGET_PREFIX + fee] = None
    rc.commit_assumption_edit(state, fee)
    assert math.isnan(state["draft_values"][fee]) and not state["draft_error"]

    state[rc.WIDGET_PREFIX + mode] = True
    rc.commit_assumption_edit(state, mode)
    assert state["draft_values"][mode] == 1 and isinstance(state["draft_values"][mode], int)
    assert not state["draft_error"]


def test_a_price_curve_is_a_draft_input_passed_only_to_the_next_run() -> None:
    state = _fresh_state()
    seen: list = []

    def runner(model, values, start, event_presets=(), user_price_curve=None):
        seen.append(user_price_curve)
        return _fake_result(model, values)

    rc.run_full(state, runner)
    curve = pd.DataFrame({"half_hour_start": ["00:00"], "price_gbp_per_mwh": [50.0]})
    rc.set_user_price_curve(state, curve)
    # Setting it never runs; the run bar names it as an unrun change.
    assert len(seen) == 1
    assert rc.unrun_changes(state) == ["supplier price curve"]
    rc.run_full(state, runner)
    assert seen[-1] is curve
    assert rc.unrun_changes(state) == []
    rc.reset_to_defaults(state)
    assert state["draft_user_price_curve"] is None


# --- Firm MW: the blackout table and the newsvendor/event conflict ---------


def test_a_blackout_table_is_a_draft_input_passed_only_to_the_next_run() -> None:
    state = _fresh_state()
    seen: list = []

    def runner(model, values, start, event_presets=(), blackout_windows=None):
        seen.append(blackout_windows)
        return _fake_result(model, values)

    rc.run_full(state, runner)
    assert seen == [None]  # nothing passed while the draft table has no rows
    table = pd.DataFrame({"start_local_time": ["23:00"], "duration_minutes": [60]})
    rc.set_blackout_windows(state, table)
    # Setting it never runs; the run bar names it as an unrun change.
    assert len(seen) == 1
    assert rc.unrun_changes(state) == ["blackout windows"]
    rc.run_full(state, runner)
    assert seen[-1] is table
    assert rc.unrun_changes(state) == []
    rc.reset_to_defaults(state)
    assert state["draft_blackout_windows"].empty


def test_invalid_blackout_table_blocks_run_with_a_named_reason() -> None:
    state = _fresh_state()
    bad = pd.DataFrame({"start_local_time": ["23:05"], "duration_minutes": [60]})

    rc.set_blackout_windows(state, bad)

    assert state["draft_blackout_error"] and "start_local_time" in state["draft_blackout_error"]
    assert "Blackout windows" in state["draft_error"]
    # Fixed by the next edit, the same recovery an invalid event selection gets.
    rc.set_blackout_windows(state, rc.EMPTY_BLACKOUT_WINDOWS)
    assert state["draft_blackout_error"] is None and not state["draft_error"]


def test_commitment_rule_error_names_the_two_newsvendor_blocks() -> None:
    assert rc.commitment_rule_error("fixed_share", 1, ()) is None
    assert "one simulated week" in rc.commitment_rule_error("newsvendor", 1, ())
    assert "short-notice" in rc.commitment_rule_error("newsvendor", 4, ("surprise_evening_spike",))
    assert rc.commitment_rule_error("newsvendor", 4, ("cold_still_evening",)) is None


def test_newsvendor_is_blocked_with_one_simulated_week() -> None:
    # Overnight review log, 30 Sep 2026: no other week to read a forecast
    # error quantile from.
    state = _fresh_state()
    _edit(state, "evaluation_world_count", 1)
    assert not state["draft_error"]

    _edit(state, "trading.commitment_rule", "newsvendor")

    assert state["draft_error"] and "one simulated week" in state["draft_error"]
    _edit(state, "evaluation_world_count", 4)
    assert not state["draft_error"]


def test_newsvendor_is_blocked_with_a_short_notice_event_selected() -> None:
    state = _fresh_state()
    state[rc.EVENT_WIDGET_PREFIX + "surprise_evening_spike"] = True
    rc.commit_event_preset(state, "surprise_evening_spike")
    assert not state["draft_error"]

    _edit(state, "trading.commitment_rule", "newsvendor")

    assert state["draft_error"] and "short-notice" in state["draft_error"]
    # The no-action model applies no events (applied_presets), so switching
    # model clears the conflict; model_changed re-checks it.
    state["model_mode"] = rc.MODEL_NO_ACTION
    rc.model_changed(state)
    assert not state["draft_error"]
