"""Blackout windows: the planner leaves them untouched and trading is closed there.

Trading contract v1 §10.5b items 1 and 2 (lane J1b).  Hand-checkable planner
and one-EV kernel cases (10 kWh battery, 4 kW charger = 2 kWh per half-hour,
100 % efficiency, 8 kWh target), then small SYNTHETIC real runs.  Every value
is illustrative.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from axle_studio.model import assumptions, events
from axle_studio.model.action import (
    plan_around_blackout,
    plan_cheapest_slots,
    smart_charging_inputs,
)
from axle_studio.model.availability import validate_blackout_windows
from axle_studio.model.forecast import run_forecast_from_assumptions, simulate_forecast
from axle_studio.model.physics import PublicTopUp, simulate_fleet_intervals
from axle_studio.model.settings import RunSettings
from axle_studio.model.summaries import build_study_slots, build_warmup_slots

_TOP_UP = PublicTopUp(efficiency_fraction=0.9, threshold_soc_fraction=0.1, target_soc_fraction=0.8)
_WINTER = date(2025, 1, 7)  # GMT, so London clock time equals UTC
_NOON = pd.Timestamp("2025-01-07 12:00", tz="UTC")


# --------------------------------------------------------------------------
# The planner rule alone
# --------------------------------------------------------------------------


def test_blackout_over_the_cheapest_slots_moves_the_plan_to_the_next_cheapest() -> None:
    prices = np.array([[50.0, 10.0, 20.0, 30.0, np.inf]])
    blackout = np.array([[False, True, False, False, False]])
    planned = plan_around_blackout(np.array([3.0]), np.array([2.0]), prices, blackout)
    # Unmanaged charging is 2 kWh in slot 0 and 1 kWh in slot 1.  Slot 1 is
    # in the blackout, so it keeps its unmanaged 1 kWh (nothing moved out,
    # nothing moved in) and the other 2 kWh go to the cheapest slot outside
    # it (20), not to the cheaper blackout slot (10).
    np.testing.assert_allclose(planned, [[0.0, 1.0, 2.0, 0.0, 0.0]])


def test_blackout_over_plug_in_keeps_the_unmanaged_charging_there() -> None:
    prices = np.array([[50.0, 40.0, 10.0, 20.0]])
    blackout = np.array([[True, False, False, False]])
    planned = plan_around_blackout(np.array([3.0]), np.array([2.0]), prices, blackout)
    # Unmanaged: 2 kWh in slot 0 (a blackout slot, kept), 1 kWh in slot 1;
    # the 1 kWh left goes to the cheapest outside slot (10).
    np.testing.assert_allclose(planned, [[2.0, 0.0, 1.0, 0.0]])


def test_no_blackout_is_the_plain_planner_and_a_short_window_stays_short() -> None:
    prices = np.array([[30.0, 10.0, 20.0, np.inf]])
    none = np.zeros((1, 4), dtype=bool)
    np.testing.assert_array_equal(
        plan_around_blackout(np.array([3.0]), np.array([2.0]), prices, none),
        plan_cheapest_slots(np.array([3.0]), np.array([2.0]), prices),
    )
    # A need of 6 kWh: unmanaged fills slots 0-2 (2 + 2 + 2).  The blackout
    # keeps slot 1's 2 kWh; outside it only slots 0 and 2 remain (4 kWh), so
    # the plan is 2 + 2 + 2: the need is met.  With 7 kWh the plan still
    # never puts more than the unmanaged 2 kWh into slot 1 and stops short.
    blackout = np.array([[False, True, False, False]])
    planned = plan_around_blackout(np.array([7.0]), np.array([2.0]), prices, blackout)
    np.testing.assert_allclose(planned, [[2.0, 2.0, 2.0, 0.0]])
    assert planned.sum() < 7.0


# --------------------------------------------------------------------------
# One EV through the kernel
# --------------------------------------------------------------------------


def _settings() -> RunSettings:
    return RunSettings(
        start_local_date=_WINTER,
        warmup_days=1,
        study_days=2,
        vehicle_count=1,
        seed=1,
        evaluation_world_count=1,
        opening_soc_fraction=0.5,
        reserve_soc_fraction=0.1,
        home_charge_efficiency=1.0,
        weather_efficiency_sensitivity_fraction_per_c=0.0,
    )


def _units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population_id": ["population:blackout"],
            "unit_id": ["ev-001"],
            "cohort_id": ["average_uk"],
            "physical_capacity_kwh": [10.0],
            "efficiency_miles_per_battery_kwh": [1.0],
            "home_charger_limit_kw": [4.0],
            "preferred_target_soc_fraction": [0.8],
            "runtime_departure_local_hour": [7],
            "runtime_arrival_local_hour": [18],
        }
    )


def _inputs(settings: RunSettings) -> dict[str, np.ndarray]:
    """One home session 18:00-07:00 on the study start date, no trips."""

    shape = (1, settings.sampled_day_count, 1)
    timestamps = np.full(shape, pd.NaT, dtype=object)
    inputs = {
        "drives_today": np.zeros(shape, dtype=bool),
        "outbound_miles": np.zeros(shape),
        "return_miles": np.zeros(shape),
        "trip_departure_utc": timestamps.copy(),
        "destination_dwell_seconds": np.zeros(shape),
        "drive_speed_mph": np.full(shape, 30.0),
        "connection_session_accepted": np.zeros(shape, dtype=bool),
        "connection_start_utc": timestamps.copy(),
        "connection_end_utc": timestamps.copy(),
        "desired_pre_drive_soc_fraction": np.full(shape, 0.8),
    }
    inputs["connection_session_accepted"][0, 1, 0] = True
    inputs["connection_start_utc"][0, 1, 0] = pd.Timestamp("2025-01-07 18:00", tz="UTC")
    inputs["connection_end_utc"][0, 1, 0] = pd.Timestamp("2025-01-08 07:00", tz="UTC")
    return inputs


def _slot(instant: str) -> int:
    return int((pd.Timestamp(instant, tz="UTC") - _NOON) // pd.Timedelta(minutes=30))


def _run(windows: pd.DataFrame | None) -> dict[str, dict[int, float]]:
    settings = _settings()
    prices = np.full((1, 48 * 3), 100.0)
    # Cheap night: 03:00 (10), 03:30 (15), 04:00 (20); 3 kWh needed (5 -> 8).
    for instant, price in (("03:00", 10.0), ("03:30", 15.0), ("04:00", 20.0)):
        prices[0, 48 + _slot(f"2025-01-08 {instant}")] = price
    blackout = None
    if windows is not None:
        table = validate_blackout_windows(windows)
        labels = [label for row in table["slot_labels"] for label in row]
        slots = pd.concat(
            [
                build_warmup_slots(_WINTER, 1, blackout_labels=labels),
                build_study_slots(_WINTER, 2, blackout_labels=labels),
            ]
        )
        blackout = slots["blackout"].to_numpy()
    smart = smart_charging_inputs(
        settings, _units(), prices, departure_margin_hours=1.0, blackout=blackout
    )
    fleet = simulate_fleet_intervals(
        settings, _units(), _inputs(settings), public_top_up=_TOP_UP, smart_charging=smart
    )[0]
    assert fleet["conservation_residual_kwh"].abs().max() <= 1e-12
    out = {}
    for path, frame in fleet.groupby("path_id"):
        values = frame.sort_values("interval_start_utc")["home_grid_import_kwh"].to_numpy()
        out[path] = {int(s): float(values[s]) for s in np.flatnonzero(values > 1e-12)}
    return out


def _windows(*rows: tuple[str, int]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["start_local_time", "duration_minutes"])


def test_kernel_moves_the_plan_out_of_a_blackout_over_the_cheapest_half_hours() -> None:
    free = _run(None)
    assert free["selected"] == pytest.approx(
        {_slot("2025-01-08 03:00"): 2.0, _slot("2025-01-08 03:30"): 1.0}
    )
    blocked = _run(_windows(("03:00", 60)))
    # 03:00 and 03:30 are promised untouched; the unmanaged path charges
    # nothing there (it charges at 18:00), so the plan takes the next
    # cheapest slot outside (04:00, 2 kWh) and then flat-price slots in time
    # order from plug-in (18:00, 1 kWh).  The session still reaches target.
    assert blocked["selected"] == pytest.approx(
        {_slot("2025-01-07 18:00"): 1.0, _slot("2025-01-08 04:00"): 2.0}
    )
    assert sum(blocked["selected"].values()) == pytest.approx(3.0)
    assert blocked["normal"] == free["normal"]


def test_blackout_over_the_whole_window_makes_selected_equal_normal() -> None:
    covered = _run(_windows(("18:00", 720)))
    assert (
        covered["selected"]
        == covered["normal"]
        == pytest.approx({_slot("2025-01-07 18:00"): 2.0, _slot("2025-01-07 18:30"): 1.0})
    )


# --------------------------------------------------------------------------
# Masks and validation
# --------------------------------------------------------------------------


def test_trading_mask_is_closed_in_blackout_slots_at_every_decision() -> None:
    slots = build_study_slots(date(2026, 10, 12), blackout_labels=["18:00", "18:30"])
    blackout = slots["blackout"].to_numpy()
    assert blackout.sum() == 14
    windows = events.event_slots(events.empty_events(), slots)
    for known in (None, 0, int(slots["interval_start_utc"].iloc[-1].value)):
        mask = events.trading_mask(windows, len(slots), known_at_utc=known, blackout=blackout)
        np.testing.assert_array_equal(mask, ~blackout)
    np.testing.assert_array_equal(
        events.trading_mask(windows, len(slots), known_at_utc=None), np.ones(len(slots), bool)
    )


def _request(start: str, minutes: int) -> pd.DataFrame:
    row = {
        "event_id": "td",
        "event_type": "turn_down",
        "enabled": True,
        "night_index": 1,
        "start_local_time": start,
        "duration_minutes": minutes,
        "size": np.nan,
        "size_unit": "MW",
        "notice": "day_ahead",
        "notice_minutes": pd.NA,
        "scope": "national",
        "payment_gbp_per_mwh": 100.0,
    }
    return pd.DataFrame([row], columns=list(events.EVENT_COLUMNS))


def test_validate_events_rejects_a_request_wholly_inside_a_blackout() -> None:
    slots = build_study_slots(date(2026, 10, 12), blackout_labels=["18:00", "18:30", "19:00"])
    blackout = slots["blackout"].to_numpy()
    with pytest.raises(ValueError, match="blackout"):
        events.validate_events(
            _request("18:00", 90), slots, assumptions.ZONE_IDS, blackout=blackout
        )
    # Partial overlap is allowed (delivery is measured outside the blackout).
    events.validate_events(_request("18:30", 120), slots, assumptions.ZONE_IDS, blackout=blackout)
    # Without a blackout the same request is fine.
    events.validate_events(_request("18:00", 90), slots, assumptions.ZONE_IDS)


@pytest.mark.parametrize(
    ("row", "rule"),
    [
        (("18:10", 60), "half-hour"),
        (("18:00", 45), "multiple of 30"),
        (("18:00", 0), "multiple of 30"),
    ],
)
def test_blackout_table_rejects_bad_rows(row, rule) -> None:
    with pytest.raises(ValueError, match=rule):
        validate_blackout_windows(_windows(row))


def test_blackout_table_rejects_an_overlap() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        validate_blackout_windows(_windows(("23:00", 120), ("00:30", 60)))


# --------------------------------------------------------------------------
# Real runs (SYNTHETIC, small)
# --------------------------------------------------------------------------

_START = date(2026, 10, 12)
_SMALL = {"vehicle_count": 30, "evaluation_world_count": 3}


def test_an_empty_blackout_table_reproduces_the_run_bit_for_bit() -> None:
    plain = run_forecast_from_assumptions(_START, values=_SMALL)
    empty = run_forecast_from_assumptions(_START, values=_SMALL, blackout_windows=_windows())
    for name in (
        "fleet_world_intervals",
        "cohort_world_intervals",
        "deviation_world_slot",
        "trading_ledger_world",
        "study_slots",
    ):
        pd.testing.assert_frame_equal(getattr(plain, name), getattr(empty, name), obj=name)
    assert not plain.study_slots["blackout"].any()
    assert list(empty.blackout_windows.columns) == [
        "start_local_time",
        "duration_minutes",
        "end_local_time",
        "slot_labels",
    ]


def test_a_blackout_run_keeps_the_normal_path_conserves_energy_and_closes_trading() -> None:
    windows = _windows(("01:00", 120))
    plain = run_forecast_from_assumptions(_START, values=_SMALL)
    blocked = run_forecast_from_assumptions(_START, values=_SMALL, blackout_windows=windows)
    world_a, world_b = plain.fleet_world_intervals, blocked.fleet_world_intervals
    normal = world_a["path_id"].eq("normal")
    pd.testing.assert_frame_equal(
        world_a.loc[normal].reset_index(drop=True), world_b.loc[normal].reset_index(drop=True)
    )
    assert not world_a.loc[~normal, "home_import_kwh"].equals(
        world_b.loc[~normal, "home_import_kwh"]
    )
    blackout = blocked.study_slots["blackout"].to_numpy()
    assert blackout.sum() == 7 * 4
    deviation = blocked.deviation_world_slot
    closed = np.tile(blackout, blocked.world_count)
    assert not deviation.loc[closed, "settlement_open"].any()
    for strategy in ("da_only", "full"):
        assert (deviation.loc[closed, f"position_{strategy}_kwh"] == 0.0).all()
    assert (deviation.loc[closed, "settled_kwh"] == 0.0).all()


def test_warmup_and_study_slots_carry_the_blackout_and_holiday_flags() -> None:
    settings = assumptions.run_settings(_SMALL, _START)
    inputs = assumptions.forecast_inputs(
        assumptions.resolve_values(_SMALL), warmup_days=7, study_days=7
    )
    shared = inputs["shared_factor_assumptions"] | {"behaviour.holiday_half_term_week": 1.0}
    inputs["shared_factor_assumptions"] = shared
    simulated = simulate_forecast(
        settings,
        assumptions.cohort_fixture(_SMALL),
        blackout_windows=_windows(("23:30", 60)),
        **inputs,
    )
    warmup = simulated.warmup_slots
    assert set(warmup.loc[warmup["blackout"], "local_time_label"]) == {"23:30", "00:00"}
    assert not warmup["holiday"].any()
    run_blackout = simulated.smart_charging.blackout
    assert run_blackout is not None and run_blackout.sum() == 2 * (7 + 7)
