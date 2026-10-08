"""Intraday dispatch wired end to end on real (small) runs (intraday-dispatch-v1 §5.3, §12 K4).

A SYNTHETIC illustrative Monte Carlo at 60 EVs x 6 worlds with the default
assumptions, where intraday dispatch is on (lead decision Q1): the kernel's
third pass fills ``DispatchSums``, the trading overlay splits the intraday
P&L on the day-ahead plan path's book, the summaries build the three
dispatch frames and the §6.3 rows, and the result passes
``validate_result_v2`` with every lane validator (household, supplier,
availability, product, replay and dispatch) on the dispatched path.  The
same run with the switch off and a no-action run carry ``None`` frames and
an all-False ``units.dispatch_locked``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from functools import cache

import numpy as np
import pandas as pd
import pytest
from fixtures.result_contract import validate_result_v2

from axle_studio.model import assumptions
from axle_studio.model.events import event_table
from axle_studio.model.forecast import (
    ForecastResult,
    run_forecast,
    run_forecast_from_assumptions,
)
from axle_studio.model.household import household_card
from axle_studio.model.individual import replay_one_ev, replay_one_ev_timeline
from axle_studio.model.summaries import build_study_slots

_START = date(2026, 10, 12)  # a Monday, no clock change
_SMALL = {"vehicle_count": 60, "evaluation_world_count": 6}
_DISPATCH_FRAMES = ("dispatch_world_slot", "dispatch_bands", "dispatch_split")


@cache
def _run(dispatch: int = 1, model: str = "action", presets: tuple = ()) -> ForecastResult:
    """One small run; ``dispatch`` sets ``trading.intraday_dispatch`` (0 or 1).

    The dispatch records are not dialog-editable yet (the UI lane owns
    that), so the run goes through ``run_forecast`` with the trading inputs
    and its records carry the switch value the run used.
    """

    resolved = assumptions.resolve_values(_SMALL)
    inputs = assumptions.forecast_inputs(resolved, warmup_days=7, study_days=7)
    inputs["trading_assumptions"] = inputs["trading_assumptions"] | {
        "trading.intraday_dispatch": float(dispatch)
    }
    settings = assumptions.run_settings(resolved, _START)
    result = run_forecast(
        settings,
        assumptions.cohort_fixture(resolved),
        **inputs,
        events=event_table(presets, build_study_slots(_START, settings.study_days)),
        model=model,
        supplier_inputs=assumptions.supplier_inputs(resolved),
    )
    records = tuple(
        replace(record, value=dispatch) if record.name == "trading.intraday_dispatch" else record
        for record in assumptions.result_assumptions(resolved)
    )
    return replace(result, assumptions=records)


def _selected_import(result: ForecastResult) -> np.ndarray:
    fleet = result.fleet_world_intervals
    rows = fleet.loc[fleet["path_id"].eq("selected")].sort_values(["world_id", "slot_index"])
    return rows["home_import_kwh"].to_numpy(dtype=float).reshape(result.world_count, -1)


def _kpi(result: ForecastResult, strategy: str, metric: str) -> pd.Series:
    kpis = result.trading_kpis
    return kpis.loc[kpis["strategy"].eq(strategy) & kpis["metric"].eq(metric)].iloc[0]


def test_dispatch_is_on_by_default() -> None:
    assert assumptions.TRADING_OVERLAY["trading.intraday_dispatch"].value == 1
    assert assumptions.TRADING_OVERLAY["trading.replan_threshold_gbp_per_mwh"].bounds == (
        0.0,
        500.0,
    )


def test_default_action_run_dispatches_and_passes_every_validator() -> None:
    result = _run()
    for name in _DISPATCH_FRAMES:
        assert getattr(result, name) is not None, name
    validate_result_v2(result)
    # K = round-half-up(0.8 x 60) = 48 locked EVs (§2).
    assert int(result.units["dispatch_locked"].sum()) == 48
    assert result.dispatch_split["locked_count"].sum() == 48
    # The dispatch is live, not inert: free EVs re-plan and energy moves.
    world_slot = result.dispatch_world_slot
    assert world_slot["replan_count"].sum() > 0
    assert (world_slot["moved_kwh"] != 0.0).any()
    moved = world_slot["moved_kwh"].to_numpy().reshape(result.world_count, -1)
    expected = np.maximum(0.0, -moved).sum(axis=1) / 1000.0
    for strategy in ("da_only", "full", "perfect_foresight"):
        row = _kpi(result, strategy, "dispatch_moved_mwh_per_week")
        assert row["mean"] == pytest.approx(expected.mean(), rel=1e-12)
        assert row["mean"] > 0.0
    # The §6.3 split rows come from the ledger's "of which" columns.
    week = result.trading_week_world
    full = week.loc[week["strategy"].eq("full")].sort_values("world_id")
    for metric, column in (
        ("rebalancing_gbp_per_week", "intraday_pnl_rebalancing_gbp"),
        ("reoptimisation_gbp_per_week", "intraday_pnl_reoptimisation_gbp"),
    ):
        assert _kpi(result, "full", metric)["mean"] == pytest.approx(full[column].mean())
    assert (full["intraday_pnl_reoptimisation_gbp"] != 0.0).any()
    check = result.trading_checks.set_index("check_id").loc["dispatch_split_sums_to_intraday"]
    assert bool(check["passed"])
    metrics = set(result.trading_ledger_summary["metric"])
    assert {"intraday_pnl_rebalancing_gbp", "trading_cost_reoptimisation_gbp"} <= metrics


def test_per_ev_lenses_read_the_dispatched_path() -> None:
    # The chunk pass reconciled with the fleet inside build_summaries; the
    # One EV replay and the household card must follow the dispatched path
    # too, and they report whether the EV is locked.
    result = _run()
    locked = result.units["dispatch_locked"].to_numpy()
    world = result.sampled_world_ids[0]
    selected = _selected_import(result)[world]
    total = np.zeros_like(selected)
    for unit_id in result.units["unit_id"]:
        intervals = replay_one_ev(result, unit_id, world).intervals
        rows = intervals.loc[intervals["path_id"].eq("selected")].sort_values("slot_index")
        total += rows["home_import_kwh"].to_numpy(dtype=float)
    np.testing.assert_allclose(total, selected, rtol=0.0, atol=1e-9)
    free = result.units["unit_id"].iat[int(np.flatnonzero(~locked)[0])]
    assert replay_one_ev_timeline(result, free, world).dispatch_locked is False
    assert household_card(result, free).dispatch_locked is False
    locked_id = result.units["unit_id"].iat[int(np.flatnonzero(locked)[0])]
    assert household_card(result, locked_id).dispatch_locked is True


def test_switch_off_run_has_no_dispatch_and_equals_the_day_ahead_plan_path() -> None:
    off, on = _run(0), _run()
    validate_result_v2(off)
    for name in _DISPATCH_FRAMES:
        assert getattr(off, name) is None, name
    assert not off.units["dispatch_locked"].any()
    assert off.units["dispatch_locked"].dtype == bool
    # With the switch off the selected path is the day-ahead plan path the
    # switch-on run keeps as its reference, bit for bit (§5.2).
    reference = on.dispatch_world_slot.sort_values(["world_id", "slot_index"])
    np.testing.assert_array_equal(
        _selected_import(off),
        reference["day_ahead_plan_kwh"].to_numpy().reshape(off.world_count, -1),
    )
    # Re-optimisation is exactly 0 and moved energy 0 (item 58).
    ledger = off.trading_ledger_world
    full = ledger.loc[ledger["strategy"].eq("full")]
    assert (full["intraday_pnl_reoptimisation_gbp"] == 0.0).all()
    assert (full["trading_cost_reoptimisation_gbp"] == 0.0).all()
    assert (full["intraday_pnl_rebalancing_gbp"] == full["intraday_pnl_gbp"]).all()
    assert _kpi(off, "full", "dispatch_moved_mwh_per_week")["p90"] == 0.0
    # The switch moves no draw and no price (§5.3).
    pd.testing.assert_frame_equal(off.evaluation_prices, on.evaluation_prices)
    pd.testing.assert_frame_equal(off.forecast_prices, on.forecast_prices)
    normal = off.fleet_world_intervals.loc[off.fleet_world_intervals["path_id"].eq("normal")]
    pd.testing.assert_frame_equal(
        normal, on.fleet_world_intervals.loc[on.fleet_world_intervals["path_id"].eq("normal")]
    )


def test_no_action_run_has_no_dispatch() -> None:
    result = _run(model="no_action")
    validate_result_v2(result)
    for name in _DISPATCH_FRAMES:
        assert getattr(result, name) is None, name
    assert not result.units["dispatch_locked"].any()


def test_event_response_bands_carry_the_day_ahead_plan_series_with_dispatch() -> None:
    # K3's ``dispatch_sums`` branch of ``event_response_bands`` (§7.4), on a
    # real run with a scripted turn-down request.
    on, off = _run(presets=("dfs_turn_down",)), _run(0, presets=("dfs_turn_down",))
    validate_result_v2(on)
    series = set(on.event_response_bands["series"])
    assert {"normal", "selected", "day_ahead_plan"} <= series
    assert "day_ahead_plan" not in set(off.event_response_bands["series"])


def test_the_default_app_entry_point_dispatches() -> None:
    result = run_forecast_from_assumptions(_START, values=_SMALL)
    assert result.dispatch_world_slot is not None
    assert int(result.units["dispatch_locked"].sum()) == 48


def test_replan_threshold_outside_its_bounds_is_rejected_before_sampling() -> None:
    inputs = assumptions.forecast_inputs(
        assumptions.resolve_values(_SMALL), warmup_days=7, study_days=7
    )
    inputs["trading_assumptions"] = inputs["trading_assumptions"] | {
        "trading.replan_threshold_gbp_per_mwh": 600.0
    }
    with pytest.raises(ValueError, match="replan_threshold"):
        run_forecast(
            assumptions.run_settings(_SMALL, _START), assumptions.cohort_fixture(_SMALL), **inputs
        )
