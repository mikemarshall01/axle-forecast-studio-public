"""Flexibility metrics per half-hour (decision 0004 item 43, contract 3.6c).

Hand-checkable cases first (two EVs, round numbers), then invariants on
real runs: the kernel can never import more than the flexible power or the
movable energy allows, and real results pass the contract validator.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from fixtures.result_fixture import validate_result_v2

from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.model.settings import RunSettings
from axle_studio.model.summaries import (
    FlexibilityInputs,
    _per_ev_chunks,
    build_study_slots,
    flexibility_inputs,
    flexibility_per_world,
)

# --------------------------------------------------------------------------
# Hand-checkable: two EVs, 10 kWh batteries, 8 kWh target, 4 kW chargers
# --------------------------------------------------------------------------


def _flex(efficiency: float = 1.0) -> FlexibilityInputs:
    return FlexibilityInputs(
        hours_to_departure=np.array([[3.0, 3.0], [2.5, 2.5]]),  # (slot, EV)
        target_kwh=np.array([8.0, 8.0]),
        power_kw=np.array([4.0, 4.0]),
        efficiency=efficiency,
    )


def _none_imported() -> np.ndarray:
    return np.zeros((1, 2, 2))


def test_turn_up_headroom_is_clipped_at_zero() -> None:
    # EV A draws its full 4 kW plus rounding noise: headroom is 0, not -0.0.
    connected = np.array([[[True, False], [False, False]]])
    opening = np.array([[[6.0, 8.0], [6.0, 8.0]]])
    imported = np.array([[[2.0 + 1e-12, 0.0], [0.0, 0.0]]])
    flex = flexibility_per_world(connected, opening, imported, _flex())
    np.testing.assert_array_equal(flex["turn_up_headroom_kw"], [[0.0, 0.0]])


def test_two_evs_by_hand() -> None:
    # Slot 0: EV A plugged in at 6 kWh (2 kWh short) and importing 1 kWh in
    # the slot, EV B plugged in at target.  Slot 1: neither is plugged in.
    connected = np.array([[[True, True], [False, False]]])
    opening = np.array([[[6.0, 8.0], [6.0, 7.0]]])
    imported = np.array([[[1.0, 0.0], [0.0, 0.0]]])
    flex = flexibility_per_world(connected, opening, imported, _flex())

    # Only A is below target: 4 kW of flexible power, 2 kWh to move, and
    # 4 - 1 kWh / 0.5 h = 2 kW of turn-up headroom still unused.
    np.testing.assert_allclose(flex["flexible_power_kw"], [[4.0, 0.0]])
    np.testing.assert_allclose(flex["turn_up_headroom_kw"], [[2.0, 0.0]])
    np.testing.assert_allclose(flex["movable_energy_kwh"], [[2.0, 0.0]])
    # Slack: A has 3 h left and needs 2 kWh / 4 kW = 0.5 h, so 2.5 h.  B is
    # at target, so it will not charge and adds no slack (decision 0004 item 46):
    # every quantile is A's 2.5 h.
    np.testing.assert_allclose(flex["time_slack_hours"][:, 0, 0], [2.5, 2.5, 2.5])
    # No EV plugged in: no slack (missing, not zero).
    assert np.isnan(flex["time_slack_hours"][:, 0, 1]).all()


def test_an_ev_at_target_contributes_no_slack() -> None:
    connected = np.array([[[True, True], [True, True]]])
    at_target = flexibility_per_world(connected, np.full((1, 2, 2), 8.0), _none_imported(), _flex())
    assert np.isnan(at_target["time_slack_hours"]).all()
    assert (at_target["flexible_power_kw"] == 0.0).all()
    assert (at_target["movable_energy_kwh"] == 0.0).all()


def test_movable_energy_is_grid_side_and_slack_can_go_negative() -> None:
    # 90 % charging efficiency: 2 battery kWh need 2 / 0.9 grid kWh.  An EV
    # 7 kWh short (grid 7.78 kWh, 1.94 h at 4 kW) with 1 h left cannot make it.
    connected = np.array([[[True, True], [True, True]]])
    opening = np.array([[[6.0, 1.0], [6.0, 1.0]]])
    flex = flexibility_per_world(connected, opening, _none_imported(), _flex(efficiency=0.9))
    np.testing.assert_allclose(flex["movable_energy_kwh"][0, 0], 2.0 / 0.9 + 7.0 / 0.9)
    np.testing.assert_allclose(flex["flexible_power_kw"][0, 0], 8.0)
    hours_short = 3.0 - (7.0 / 0.9) / 4.0
    assert flex["time_slack_hours"][0, 0, 0] == pytest.approx(
        hours_short + 0.1 * ((3.0 - (2.0 / 0.9) / 4.0) - hours_short)
    )
    tight = flexibility_per_world(
        connected,
        opening,
        _none_imported(),
        FlexibilityInputs(np.full((2, 2), 1.0), np.array([8.0, 8.0]), np.array([4.0, 4.0]), 0.9),
    )
    assert tight["time_slack_hours"][0, 0, 0] < 0.0


def _units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "physical_capacity_kwh": [10.0],
            "preferred_target_soc_fraction": [0.8],
            "home_charger_limit_kw": [4.0],
            "runtime_departure_local_hour": [7],
            "runtime_weekend_departure_local_hour": [10],
        }
    )


def test_hours_to_departure_use_the_planners_next_expected_departure() -> None:
    # Winter week from Friday 10 January 2025 (GMT), one-hour margin, so the
    # expected departures are 06:00 on weekdays and 09:00 at weekends.
    settings = RunSettings(
        start_local_date=date(2025, 1, 10),
        warmup_days=0,
        study_days=4,
        vehicle_count=1,
        seed=1,
        evaluation_world_count=1,
        opening_soc_fraction=0.8,
        reserve_soc_fraction=0.1,
        home_charge_efficiency=0.9,
        weather_efficiency_sensitivity_fraction_per_c=0.0,
    )
    slots = build_study_slots(settings.start_local_date, settings.study_days)
    flex = flexibility_inputs(settings, _units(), slots, 1.0)
    hours = pd.Series(flex.hours_to_departure[:, 0], index=slots["interval_start_utc"])
    # Friday 18:00 -> Saturday 09:00 (weekend window): 15 h.
    assert hours[pd.Timestamp("2025-01-10 18:00Z")] == 15.0
    # Monday 05:30 -> Monday 06:00: the one half-hour left.
    assert hours[pd.Timestamp("2025-01-13 05:30Z")] == 0.5
    # Monday 06:00 is no longer at least a half-hour before 06:00, so the
    # next is Tuesday 06:00, as the smart charger would plan.
    assert hours[pd.Timestamp("2025-01-13 06:00Z")] == 24.0
    assert flex.target_kwh.tolist() == [8.0] and flex.power_kw.tolist() == [4.0]
    assert flex.efficiency == 0.9


# --------------------------------------------------------------------------
# Real runs
# --------------------------------------------------------------------------

_VALUES = {"vehicle_count": 40, "evaluation_world_count": 3}


@pytest.fixture(scope="module")
def action_result():
    return run_forecast_from_assumptions(date(2026, 1, 12), model="action", values=_VALUES)


def test_real_runs_pass_the_contract_validator(action_result) -> None:
    validate_result_v2(action_result)
    no_action = run_forecast_from_assumptions(date(2026, 1, 12), model="no_action", values=_VALUES)
    validate_result_v2(no_action)
    assert no_action.flexibility_bands is not None
    assert set(no_action.flexibility_bands["path_id"]) == {"normal"}


def test_the_kernel_never_imports_more_than_the_flexibility_allows(action_result) -> None:
    # Independent of the band code: on each path, in every world and slot,
    # home import is within the flexible power (kW) and the movable energy
    # (kWh), and there is import only where there is flexible power.
    state = action_result.replay_state
    flex = flexibility_inputs(state.settings, state.units, action_result.study_slots, 1.0)
    for _, by_path in _per_ev_chunks(
        state,
        action_result.study_slots,
        action_result.fleet_world_intervals,
        ["normal", "selected"],
    ):
        for unit in by_path.values():
            values = flexibility_per_world(
                unit["connected"], unit["opening_battery_kwh"], unit["home_grid_import_kwh"], flex
            )
            import_kwh = unit["home_grid_import_kwh"].sum(axis=2)
            assert (values["turn_up_headroom_kw"] >= -1e-9).all()
            assert (import_kwh / 0.5 <= values["flexible_power_kw"] + 1e-9).all()
            assert (import_kwh <= values["movable_energy_kwh"] + 1e-9).all()
            assert not ((import_kwh > 1e-9) & (values["flexible_power_kw"] == 0.0)).any()


def test_smart_charging_keeps_more_energy_movable(action_result) -> None:
    # Deferring charging leaves EVs below target for longer, so across the
    # week the selected path holds more movable energy than the normal path.
    bands = action_result.flexibility_bands
    energy = bands.loc[bands["metric"].eq("movable_energy_kwh")]
    by_path = energy.groupby("path_id")["p50"].sum()
    assert by_path["selected"] > by_path["normal"]


def test_time_slack_has_both_spreads(action_result) -> None:
    # Item 40: the across-weeks band sits beside the across-EV band.
    bands = action_result.flexibility_bands
    slack = bands.loc[bands["metric"].eq("time_slack_hours")]
    assert set(slack["spread"]) == {"across_evs", "across_weeks"}
    for spread, rows in slack.groupby("spread"):
        assert rows["p50"].notna().any(), spread
