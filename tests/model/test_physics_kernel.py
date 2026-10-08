from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from fixtures.scalar_kernel_oracle import simulate_explicit_journey_intervals

from axle_studio.model import assumptions, physics
from axle_studio.model.action import smart_charging_inputs
from axle_studio.model.clock import utc_half_hour_boundaries
from axle_studio.model.forecast import simulate_forecast
from axle_studio.model.physics import (
    _BAND_COLUMNS,
    _BAND_METRICS,
    _COHORT_BAND_COLUMNS,
    _COHORT_INTERVAL_COLUMNS,
    _INTERVAL_COLUMNS,
    _SUM_COLUMNS,
    UNIT_INTERVAL_QUANTITIES,
    PublicTopUp,
    simulate_fleet_intervals,
    simulate_unit_intervals,
)
from axle_studio.model.settings import RunSettings

# The decision 0004 item 32/37 defaults: 90 % public efficiency, top up below
# 10 % SoC to 80 % SoC (re-baselined from 20 % by item 37).
_TOP_UP = PublicTopUp(efficiency_fraction=0.9, threshold_soc_fraction=0.1, target_soc_fraction=0.8)


def _default_inputs() -> dict:
    # Fresh run inputs from model/assumptions.py (the removed JSON configs'
    # values) on every call, so a test that edits them cannot leak into another.
    return assumptions.forecast_inputs(None, warmup_days=1, study_days=7)


def _weather(*, warmup_days: int, study_days: int):
    inputs = assumptions.forecast_inputs(None, warmup_days=warmup_days, study_days=study_days)
    return inputs["base_temperature_c_by_day"], inputs["weather_sd_c"]


_KEYS = ["world_id", "path_id", "interval_start_utc", "interval_end_utc"]
_COHORT_KEYS = [*_KEYS, "cohort_id"]


def _settings(
    *, worlds: int = 2, days: int = 1, warmup_days: int = 1, units: int = 3
) -> RunSettings:
    return RunSettings(
        start_local_date=date(2025, 1, 7),
        warmup_days=warmup_days,
        study_days=days,
        vehicle_count=units,
        seed=17,
        evaluation_world_count=worlds,
        opening_soc_fraction=0.5,
        reserve_soc_fraction=0.1,
        home_charge_efficiency=0.9,
        weather_efficiency_sensitivity_fraction_per_c=0.01,
    )


def _units(count: int, **changes: object) -> pd.DataFrame:
    values: dict[str, object] = {
        "population_id": ["population:test"] * count,
        "unit_id": [f"ev-{index:03d}" for index in range(count)],
        "cohort_id": (["b", "a", "b"] * ((count + 2) // 3))[:count],
        "physical_capacity_kwh": np.resize([10.0, 20.0, 15.0], count),
        "efficiency_miles_per_battery_kwh": np.resize([4.0, 5.0, 3.5], count),
        # One home charging power per EV (decision 0004 item 34).
        "home_charger_limit_kw": np.resize([2.0, 2.5, 1.5], count),
        "preferred_target_soc_fraction": np.resize([0.8, 0.9, 0.7], count),
        "runtime_departure_local_hour": np.full(count, 8),
        "runtime_arrival_local_hour": np.full(count, 18),
    }
    values.update(changes)
    return pd.DataFrame(values)


def _empty_inputs(settings: RunSettings) -> dict[str, np.ndarray]:
    shape = (
        settings.evaluation_world_count,
        settings.sampled_day_count,
        settings.vehicle_count,
    )
    timestamps = np.full(shape, pd.NaT, dtype=object)
    return {
        "drives_today": np.zeros(shape, dtype=bool),
        "outbound_miles": np.zeros(shape),
        "return_miles": np.zeros(shape),
        "trip_departure_utc": timestamps.copy(),
        "destination_dwell_seconds": np.zeros(shape),
        "drive_speed_mph": np.ones(shape),
        "connection_session_accepted": np.zeros(shape, dtype=bool),
        "connection_start_utc": timestamps.copy(),
        "connection_end_utc": timestamps.copy(),
        "desired_pre_drive_soc_fraction": np.full(shape, 0.8),
    }


def _set_trip(
    inputs: dict[str, np.ndarray],
    world: int,
    day: int,
    unit: int,
    departure: pd.Timestamp,
    *,
    outbound: float,
    returning: float,
    speed: float,
    dwell_minutes: float,
) -> None:
    index = (world, day, unit)
    inputs["drives_today"][index] = True
    inputs["trip_departure_utc"][index] = departure
    inputs["outbound_miles"][index] = outbound
    inputs["return_miles"][index] = returning
    inputs["drive_speed_mph"][index] = speed
    inputs["destination_dwell_seconds"][index] = dwell_minutes * 60.0


def _set_connection(
    inputs: dict[str, np.ndarray],
    world: int,
    day: int,
    unit: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    index = (world, day, unit)
    inputs["connection_session_accepted"][index] = True
    inputs["connection_start_utc"][index] = start
    inputs["connection_end_utc"][index] = end


def _representative_inputs(settings: RunSettings) -> dict[str, np.ndarray]:
    inputs = _empty_inputs(settings)
    first_day = pd.Timestamp(settings.start_local_date, tz="UTC") - pd.Timedelta(
        days=settings.warmup_days
    )
    for world in range(settings.evaluation_world_count):
        for day in range(settings.sampled_day_count):
            midnight = first_day + pd.Timedelta(days=day)
            _set_trip(
                inputs,
                world,
                day,
                0,
                midnight + pd.Timedelta(hours=8, minutes=10 + 5 * world),
                outbound=4.0 + world,
                returning=3.0,
                speed=24.0,
                dwell_minutes=40.0,
            )
            _set_connection(
                inputs,
                world,
                day,
                0,
                midnight + pd.Timedelta(hours=10),
                midnight + pd.Timedelta(hours=23, minutes=30),
            )
            if world == 0:
                _set_connection(
                    inputs,
                    world,
                    day,
                    1,
                    midnight + pd.Timedelta(minutes=10),
                    midnight + pd.Timedelta(hours=12),
                )
            _set_trip(
                inputs,
                world,
                day,
                2,
                midnight + pd.Timedelta(hours=14),
                outbound=1.0,
                returning=1.0 + world,
                speed=4.0,
                dwell_minutes=15.0,
            )
    return inputs


def _threshold_zero_public(settings: RunSettings) -> PublicTopUp:
    """A top-up rule whose top-ups start only when a battery would run empty.

    The scalar reference (``fixtures/scalar_kernel_oracle.py``) has no
    top-ups (decision 0004 item 32).  With a zero threshold the kernel tops up
    only where the reference would strand, so kernel/reference comparisons of
    cases that never run empty stay exact.
    """

    return PublicTopUp(efficiency_fraction=0.9, threshold_soc_fraction=0.0, target_soc_fraction=1.0)


def _derive(frame: pd.DataFrame, *, cohort: bool) -> pd.DataFrame:
    frame = frame.copy()
    frame["closing_soc_percent"] = np.where(
        frame["physical_capacity_kwh"] > 0.0,
        100.0 * frame["closing_battery_kwh"] / frame["physical_capacity_kwh"],
        np.nan,
    )
    frame["connected_share"] = frame["connected_count"] / frame["unit_count"]
    columns = _COHORT_INTERVAL_COLUMNS if cohort else _INTERVAL_COLUMNS
    keys = _COHORT_KEYS if cohort else _KEYS
    return frame.loc[:, columns].sort_values(keys, kind="stable").reset_index(drop=True)


def _bands(frame: pd.DataFrame, *, cohort: bool) -> pd.DataFrame:
    keys = (["cohort_id"] if cohort else []) + [
        "path_id",
        "interval_start_utc",
        "interval_end_utc",
    ]
    grouped = frame.groupby(keys, sort=False, dropna=False)
    result = grouped["world_id"].nunique().rename("world_count").to_frame()
    metrics = grouped[_BAND_METRICS]
    for suffix, summary in (
        ("mean", metrics.mean()),
        ("p10", metrics.quantile(0.1)),
        ("p50", metrics.quantile(0.5)),
        ("p90", metrics.quantile(0.9)),
    ):
        result = result.join(summary.add_suffix(f"_{suffix}"))
    columns = _COHORT_BAND_COLUMNS if cohort else _BAND_COLUMNS
    return (
        result.reset_index().loc[:, columns].sort_values(keys, kind="stable").reset_index(drop=True)
    )


def _scalar_aggregate(
    settings: RunSettings,
    units: pd.DataFrame,
    inputs: dict[str, np.ndarray],
    effective_efficiency: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    intervals = simulate_explicit_journey_intervals(
        settings,
        units,
        **inputs,
        evaluation_effective_efficiency_miles_per_battery_kwh=effective_efficiency,
    ).intervals.copy()
    # The scalar reference is the normal path, which has no smart-charging
    # plans (decision 0004 item 38), so the kernel's early-departure audit
    # columns are zero by definition.
    intervals["early_departure_count"] = 0.0
    intervals["early_departure_shortfall_kwh"] = 0.0
    intervals["cohort_id"] = intervals["unit_id"].map(units.set_index("unit_id")["cohort_id"])
    grouped = intervals.assign(
        unit_count=1,
        connected_count=intervals["home_connected_full_slot"].astype(np.int64),
    ).groupby(_COHORT_KEYS, sort=False, dropna=False)
    cohort = _derive(
        grouped[[*_SUM_COLUMNS, "unit_count", "connected_count"]].sum().reset_index(),
        cohort=True,
    )
    fleet_grouped = cohort.groupby(_KEYS, sort=False, dropna=False)
    fleet = _derive(
        fleet_grouped[[*_SUM_COLUMNS, "unit_count", "connected_count"]].sum().reset_index(),
        cohort=False,
    )
    return fleet, cohort, _bands(fleet, cohort=False), _bands(cohort, cohort=True)


def _assert_results_equal(
    actual: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    expected: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    for actual_frame, expected_frame in zip(actual, expected, strict=True):
        pdt.assert_frame_equal(
            actual_frame,
            expected_frame,
            check_dtype=False,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )


@pytest.mark.parametrize("days", [1, 3, 7])
def test_numpy_kernel_matches_scalar_for_complete_study_spans(days: int) -> None:
    settings = _settings(days=days)
    units = _units(settings.vehicle_count)
    inputs = _representative_inputs(settings)

    # Item 32: over 7 days EV 2 drifts below the 20 % threshold, which the
    # scalar reference does not model; it never runs empty.
    _assert_results_equal(
        simulate_fleet_intervals(
            settings, units, inputs, public_top_up=_threshold_zero_public(settings)
        ),
        _scalar_aggregate(settings, units, inputs),
    )


def test_numpy_kernel_matches_scalar_with_day_and_world_efficiency() -> None:
    settings = _settings(days=3)
    units = _units(settings.vehicle_count)
    inputs = _representative_inputs(settings)
    baseline = units["efficiency_miles_per_battery_kwh"].to_numpy()
    effective = np.broadcast_to(
        baseline,
        (
            settings.evaluation_world_count,
            settings.sampled_day_count,
            settings.vehicle_count,
        ),
    ).copy()
    effective[0, :, 0] *= 0.5
    effective[1, 1, 2] *= 0.5

    _assert_results_equal(
        simulate_fleet_intervals(
            settings,
            units,
            inputs,
            evaluation_effective_efficiency_miles_per_battery_kwh=effective,
            public_top_up=_TOP_UP,
        ),
        _scalar_aggregate(settings, units, inputs, effective),
    )


def test_effective_efficiency_changes_requested_travel_energy() -> None:
    settings = _settings(days=1, worlds=1, units=3)
    units = _units(settings.vehicle_count)
    inputs = _representative_inputs(settings)
    baseline = units["efficiency_miles_per_battery_kwh"].to_numpy()
    effective = np.broadcast_to(
        baseline,
        (
            settings.evaluation_world_count,
            settings.sampled_day_count,
            settings.vehicle_count,
        ),
    ).copy()
    effective[:, :, 0] *= 0.5

    baseline_result = simulate_fleet_intervals(settings, units, inputs, public_top_up=_TOP_UP)[0]
    weather_result = simulate_fleet_intervals(
        settings,
        units,
        inputs,
        evaluation_effective_efficiency_miles_per_battery_kwh=effective,
        public_top_up=_TOP_UP,
    )[0]
    assert (
        weather_result["requested_outbound_travel_battery_kwh"].sum()
        > baseline_result["requested_outbound_travel_battery_kwh"].sum()
    )


def _event_order_case(case: str) -> dict[str, np.ndarray]:
    settings = _settings(worlds=1, days=2, warmup_days=0, units=1)
    inputs = _empty_inputs(settings)
    day0 = pd.Timestamp("2025-01-07", tz="UTC")
    # 30 miles at 30 mph with no dwell: return ends one hour after departure.
    _set_trip(
        inputs,
        0,
        0,
        0,
        day0 + pd.Timedelta(hours=23, minutes=30),
        outbound=15.0,
        returning=15.0,
        speed=30.0,
        dwell_minutes=0.0,
    )
    if case == "journeys_overlap":
        _set_trip(
            inputs,
            0,
            1,
            0,
            day0 + pd.Timedelta(hours=24, minutes=15),
            outbound=1.0,
            returning=1.0,
            speed=30.0,
            dwell_minutes=0.0,
        )
    elif case == "sessions_touch":
        _set_connection(inputs, 0, 0, 0, day0 + pd.Timedelta(hours=1), day0 + pd.Timedelta(hours=2))
        _set_connection(inputs, 0, 1, 0, day0 + pd.Timedelta(hours=2), day0 + pd.Timedelta(hours=3))
    elif case == "return_at_session_end":
        _set_connection(
            inputs,
            0,
            1,
            0,
            day0 + pd.Timedelta(hours=24),
            day0 + pd.Timedelta(hours=24, minutes=30),
        )
    elif case == "naive_timestamp":
        inputs["trip_departure_utc"][0, 0, 0] = pd.Timestamp("2025-01-07 23:30")
    return inputs


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("journeys_overlap", "journeys must not overlap"),
        ("sessions_touch", "sessions must not overlap or touch"),
        ("return_at_session_end", "must not share a boundary"),
        ("naive_timestamp", "timezone-aware UTC"),
    ],
)
def test_kernel_refuses_inputs_that_break_event_order(case: str, message: str) -> None:
    # These are the only input checks the kernel keeps: the slot loop assumes
    # one place per EV at a time, and same-instant arrival/unplug precedence
    # is pending in the model contract.
    settings = _settings(worlds=1, days=2, warmup_days=0, units=1)
    with pytest.raises(ValueError, match=message):
        simulate_fleet_intervals(
            settings, _units(1), _event_order_case(case), public_top_up=_TOP_UP
        )


def test_numpy_kernel_matches_scalar_for_boundary_partial_and_post_return_sessions() -> None:
    settings = _settings(worlds=1, days=1, warmup_days=0, units=4)
    # Item 32: EV 0 gets a 30 kWh battery so its 10 kWh journey does not
    # strand in the scalar reference, and the kernel runs with a zero top-up
    # threshold; this test is about session boundaries, not top-ups.
    units = _units(4, physical_capacity_kwh=[30.0, 20.0, 15.0, 10.0])
    inputs = _empty_inputs(settings)
    # The study starts at 12:00 GMT (decision 0004 item 52), so the case is
    # laid out from noon: every trip and session below falls inside it.
    day = pd.Timestamp("2025-01-07 12:00", tz="UTC")

    _set_trip(
        inputs,
        0,
        0,
        0,
        day + pd.Timedelta(hours=9),
        outbound=10.0,
        returning=30.0,
        speed=60.0,
        dwell_minutes=40.0,
    )
    _set_connection(
        inputs, 0, 0, 0, day + pd.Timedelta(hours=10, minutes=30), day + pd.Timedelta(hours=23)
    )
    _set_trip(
        inputs,
        0,
        0,
        1,
        day + pd.Timedelta(hours=9),
        outbound=1.0,
        returning=30.0,
        speed=60.0,
        dwell_minutes=29.0,
    )
    _set_connection(inputs, 0, 0, 1, day + pd.Timedelta(hours=10), day + pd.Timedelta(hours=23))
    _set_trip(
        inputs,
        0,
        0,
        2,
        day + pd.Timedelta(hours=10, minutes=20),
        outbound=1.0,
        returning=1.0,
        speed=6.0,
        dwell_minutes=30.0,
    )
    _set_connection(inputs, 0, 0, 2, day + pd.Timedelta(hours=10), day + pd.Timedelta(hours=23))
    _set_trip(
        inputs,
        0,
        0,
        3,
        day + pd.Timedelta(hours=9, minutes=50),
        outbound=1.0,
        returning=1.0,
        speed=6.0,
        dwell_minutes=0.0,
    )
    _set_connection(
        inputs, 0, 0, 3, day + pd.Timedelta(hours=10, minutes=20), day + pd.Timedelta(hours=11)
    )

    actual = simulate_fleet_intervals(
        settings, units, inputs, public_top_up=_threshold_zero_public(settings)
    )
    _assert_results_equal(actual, _scalar_aggregate(settings, units, inputs))
    normal = actual[0].loc[actual[0]["path_id"] == "normal"].set_index("interval_start_utc")
    assert normal.loc[day + pd.Timedelta(hours=10), "connected_count"] == 1
    assert normal.loc[day + pd.Timedelta(hours=10, minutes=30), "connected_count"] >= 1


def test_outbound_and_return_top_ups_replace_persistent_stranding() -> None:
    # Re-baselined for decision 0004 item 32 (supersedes item 31): these EVs
    # used to strand; now they top up (threshold 1 kWh, target 8 kWh, fallback
    # efficiency 0.9) and keep driving.  No home connections.  Threshold
    # re-baselined from 2 kWh (20%) to 1 kWh (10%) by item 37.
    # EV 0: 1 kWh opening, 2 kWh out -> tops up 1 -> 8 (7 kWh), 6; 2 kWh back -> 4.
    # EV 1: 0.5 kWh out -> tops up 1 -> 8 (7 kWh), 7.5; 1 kWh back -> 6.5.
    # EV 2: day 0 as EV 0 (7 kWh, ends 4); day 1 out 4 -> 2, back would reach
    # 0 -> tops up from the 1 kWh threshold, not the 2 kWh stock, to 8
    # (7 kWh), ends 7.  Public battery 28 kWh, grid 28 / 0.9 kWh.
    settings = replace(
        _settings(worlds=1, days=3, warmup_days=0),
        opening_soc_fraction=0.1,
    )
    units = _units(
        3,
        physical_capacity_kwh=[10.0, 10.0, 10.0],
        efficiency_miles_per_battery_kwh=[3.5, 10.0, 3.5],
        preferred_target_soc_fraction=[1.0, 1.0, 1.0],
    )
    inputs = _empty_inputs(settings)
    # Laid out from the 12:00 GMT study start (decision 0004 item 52): the
    # trips leave at 22:00, inside the horizon.
    day = pd.Timestamp("2025-01-07 12:00", tz="UTC")
    _set_trip(
        inputs,
        0,
        0,
        0,
        day + pd.Timedelta(hours=10),
        outbound=7.0,
        returning=7.0,
        speed=14.0,
        dwell_minutes=30.0,
    )
    _set_trip(
        inputs,
        0,
        0,
        1,
        day + pd.Timedelta(hours=10),
        outbound=5.0,
        returning=10.0,
        speed=60.0,
        dwell_minutes=10.0,
    )
    for day_index in (0, 1):
        _set_trip(
            inputs,
            0,
            day_index,
            2,
            day + pd.Timedelta(days=day_index, hours=10),
            outbound=7.0,
            returning=7.0,
            speed=14.0,
            dwell_minutes=30.0,
        )

    fleet = simulate_fleet_intervals(settings, units, inputs, public_top_up=_TOP_UP)[0]
    normal = fleet[fleet["path_id"].eq("normal")].set_index("interval_start_utc")

    for leg in ("outbound", "return"):
        assert normal[f"unserved_{leg}_travel_battery_kwh"].eq(0.0).all()
    assert normal["public_battery_added_kwh"].sum() == pytest.approx(28.0)
    assert normal["public_grid_import_kwh"].sum() == pytest.approx(28.0 / 0.9)
    assert normal["closing_battery_kwh"].iloc[-1] == pytest.approx(17.5)
    # EV 2 still drives on day 1 (it stayed away for good before item 32).
    assert normal.loc[day + pd.Timedelta(days=1, hours=10), "driving_fraction"] == 1.0
    assert normal["conservation_residual_kwh"].abs().max() <= 1e-12


def test_numpy_kernel_processes_two_journeys_touching_one_slot() -> None:
    settings = replace(
        _settings(worlds=1, days=3, warmup_days=0, units=1),
        opening_soc_fraction=1.0,
    )
    units = _units(
        1,
        physical_capacity_kwh=[20.0],
        efficiency_miles_per_battery_kwh=[5.0],
        preferred_target_soc_fraction=[1.0],
    )
    inputs = _empty_inputs(settings)
    for day_index, departure in enumerate(
        (pd.Timestamp("2025-01-07 23:05", tz="UTC"), pd.Timestamp("2025-01-08 00:20", tz="UTC"))
    ):
        _set_trip(
            inputs,
            0,
            day_index,
            0,
            departure,
            outbound=5.0,
            returning=5.0,
            speed=10.0,
            dwell_minutes=0.0,
        )

    _assert_results_equal(
        simulate_fleet_intervals(settings, units, inputs, public_top_up=_TOP_UP),
        _scalar_aggregate(settings, units, inputs),
    )


def test_numpy_kernel_matches_scalar_with_no_drives_or_connections() -> None:
    settings = _settings(worlds=3, days=1, warmup_days=0)
    units = _units(settings.vehicle_count)
    inputs = _empty_inputs(settings)

    actual = simulate_fleet_intervals(settings, units, inputs, public_top_up=_TOP_UP)
    _assert_results_equal(actual, _scalar_aggregate(settings, units, inputs))
    fleet = actual[0]
    assert fleet["away_on_trip_fraction"].eq(0.0).all()
    assert fleet["home_unplugged_fraction"].eq(settings.vehicle_count).all()
    assert fleet["closing_soc_percent"].between(0.0, 100.0).all()


def test_numpy_kernel_matches_scalar_fractional_duration_rounding() -> None:
    settings = _settings(worlds=1, days=1, warmup_days=0, units=1)
    units = _units(1)
    inputs = _empty_inputs(settings)
    _set_trip(
        inputs,
        0,
        0,
        0,
        pd.Timestamp("2025-01-07 10:00:00.000000123", tz="UTC"),
        outbound=1.0,
        returning=1.0,
        speed=7.0,
        dwell_minutes=1.0 / 7.0,
    )

    _assert_results_equal(
        simulate_fleet_intervals(settings, units, inputs, public_top_up=_TOP_UP),
        _scalar_aggregate(settings, units, inputs),
    )


def test_numpy_kernel_preserves_late_fractional_return_endpoint_precision() -> None:
    settings = _settings(worlds=1, days=1, warmup_days=0, units=1)
    units = _units(
        1,
        physical_capacity_kwh=[20.0],
        efficiency_miles_per_battery_kwh=[5.0],
    )
    inputs = _empty_inputs(settings)
    _set_trip(
        inputs,
        0,
        0,
        0,
        pd.Timestamp("2025-01-07 17:42", tz="UTC"),
        outbound=11.762449258150909,
        returning=12.349870992798323,
        speed=23.543262452601326,
        dwell_minutes=2971.2025194572852 / 60.0,
    )

    actual = simulate_fleet_intervals(settings, units, inputs, public_top_up=_TOP_UP)
    _assert_results_equal(actual, _scalar_aggregate(settings, units, inputs))
    interval = actual[0].loc[
        (actual[0]["path_id"] == "normal")
        & (actual[0]["interval_start_utc"] == pd.Timestamp("2025-01-07 19:30", tz="UTC"))
    ]
    assert interval["away_on_trip_fraction"].item() == pytest.approx(
        0.09900952277777778, abs=1e-12, rel=0.0
    )


def test_numpy_kernel_matches_scalar_for_six_representative_cohorts() -> None:
    settings = _settings(worlds=2, days=3, warmup_days=1, units=6)
    units = _units(6, cohort_id=[f"cohort-{index}" for index in range(6)])
    inputs = _empty_inputs(settings)
    first_day = pd.Timestamp(settings.start_local_date, tz="UTC") - pd.Timedelta(
        days=settings.warmup_days
    )
    for world in range(settings.evaluation_world_count):
        for day_index in range(settings.sampled_day_count):
            midnight = first_day + pd.Timedelta(days=day_index)
            for unit in range(settings.vehicle_count):
                _set_connection(
                    inputs,
                    world,
                    day_index,
                    unit,
                    midnight + pd.Timedelta(minutes=10 + unit),
                    midnight + pd.Timedelta(hours=6, minutes=30 + unit),
                )
                _set_trip(
                    inputs,
                    world,
                    day_index,
                    unit,
                    midnight
                    + pd.Timedelta(hours=8, minutes=90 * unit + 7 * world, nanoseconds=123),
                    outbound=1.0 + 0.2 * unit + 0.1 * world,
                    returning=0.8 + 0.15 * unit,
                    speed=17.0 + unit,
                    dwell_minutes=5.5 + unit / 7.0,
                )

    actual = simulate_fleet_intervals(settings, units, inputs, public_top_up=_TOP_UP)
    _assert_results_equal(actual, _scalar_aggregate(settings, units, inputs))
    assert set(actual[1]["cohort_id"]) == set(units["cohort_id"])


def test_numpy_kernel_accepts_mixed_precision_utc_timestamp_strings() -> None:
    settings = _settings(worlds=1, days=1, warmup_days=0, units=2)
    units = _units(2)
    inputs = _empty_inputs(settings)
    for unit, departure in enumerate(("2025-01-07T10:00:00Z", "2025-01-07T11:00:00.123Z")):
        _set_trip(
            inputs,
            0,
            0,
            unit,
            departure,  # type: ignore[arg-type]
            outbound=1.0,
            returning=1.0,
            speed=12.0,
            dwell_minutes=10.0,
        )

    _assert_results_equal(
        simulate_fleet_intervals(settings, units, inputs, public_top_up=_TOP_UP),
        _scalar_aggregate(settings, units, inputs),
    )


def test_numpy_kernel_preserves_inputs_and_world_first_reconciliation() -> None:
    settings = _settings(days=1)
    units = _units(settings.vehicle_count)
    inputs = _representative_inputs(settings)
    before = {name: values.copy() for name, values in inputs.items()}

    fleet, cohort, fleet_bands, _ = simulate_fleet_intervals(
        settings, units, inputs, public_top_up=_TOP_UP
    )

    for name, expected in before.items():
        actual = inputs[name]
        missing = pd.isna(actual) & pd.isna(expected)
        np.testing.assert_array_equal(actual[~missing], expected[~missing])
    summed = (
        cohort.groupby(_KEYS, sort=False)[[*_SUM_COLUMNS, "unit_count", "connected_count"]]
        .sum()
        .reset_index()
    )
    pdt.assert_frame_equal(
        fleet[summed.columns],
        summed,
        check_dtype=False,
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )
    expected_soc = 100.0 * fleet["closing_battery_kwh"] / fleet["physical_capacity_kwh"]
    np.testing.assert_allclose(fleet["closing_soc_percent"], expected_soc, atol=1e-12)
    assert fleet_bands["world_count"].eq(settings.evaluation_world_count).all()
    assert set(fleet["path_id"]) == {"normal", "selected"}
    normal = fleet.loc[fleet["path_id"] == "normal"].drop(columns="path_id").reset_index(drop=True)
    selected = (
        fleet.loc[fleet["path_id"] == "selected"].drop(columns="path_id").reset_index(drop=True)
    )
    pdt.assert_frame_equal(normal, selected, check_exact=True)


def test_numpy_kernel_returns_typed_empty_frames_for_zero_units() -> None:
    settings = _settings(worlds=1, days=1, warmup_days=0, units=0)
    result = simulate_fleet_intervals(
        settings, _units(0), _empty_inputs(settings), public_top_up=_TOP_UP
    )

    expected_columns = (
        _INTERVAL_COLUMNS,
        _COHORT_INTERVAL_COLUMNS,
        _BAND_COLUMNS,
        _COHORT_BAND_COLUMNS,
    )
    for frame, columns in zip(result, expected_columns, strict=True):
        assert frame.empty
        assert list(frame.columns) == columns
        assert frame["interval_start_utc"].dtype == "datetime64[ns, UTC]"
        assert frame["interval_end_utc"].dtype == "datetime64[ns, UTC]"


def _london(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="Europe/London").tz_convert("UTC")


def test_offset_change_indexes_departures_by_london_date_not_utc_block() -> None:
    # Fall-back on 25 October 2026 inside the warm-up.  The study starts at
    # 12:00 GMT on 25 October (decision 0004 item 52) and the warm-up is 48
    # UTC slots before it, from 24 October 12:00 UTC (13:00 BST).  London
    # 00:30 BST on 25 October (23:30 UTC on 24 October) lies in the first UTC
    # block of 48 slots but on London date 25 October, sampled day 1, and
    # must be accepted there.
    settings = replace(_settings(worlds=1, units=1), start_local_date=date(2026, 10, 25))
    units = _units(1)
    inputs = _empty_inputs(settings)
    _set_trip(
        inputs,
        0,
        1,
        0,
        _london("2026-10-25 00:30"),
        outbound=1.0,
        returning=1.0,
        speed=30.0,
        dwell_minutes=0.0,
    )

    fleet = simulate_fleet_intervals(settings, units, inputs, public_top_up=_TOP_UP)[0]

    assert len(fleet) == 2 * 48
    assert fleet["interval_start_utc"].iloc[0] == _london("2026-10-25 12:00")
    assert fleet["served_outbound_travel_battery_kwh"].sum() == 0.0
    assert fleet["conservation_residual_kwh"].abs().max() <= 1e-9


@pytest.mark.parametrize(
    "departure_local",
    [
        # Spring-forward on 29 March 2026 inside the warm-up: the horizon
        # starts 28 Mar 12:00 GMT and the study 29 Mar 12:00 BST.  The trip
        # is indexed sampled day 1 (29 March); 23:30 on 28 March is London
        # day 0 and 00:30 BST on 30 March (23:30 UTC on 29 March) is day 2.
        "2026-03-28 23:30",
        "2026-03-30 00:30",
    ],
)
def test_offset_change_rejects_departure_on_the_wrong_london_date(departure_local: str) -> None:
    settings = replace(_settings(worlds=1, units=1), start_local_date=date(2026, 3, 29))
    inputs = _empty_inputs(settings)
    _set_trip(
        inputs,
        0,
        1,
        0,
        _london(departure_local),
        outbound=1.0,
        returning=1.0,
        speed=30.0,
        dwell_minutes=0.0,
    )

    with pytest.raises(ValueError, match="indexed local day"):
        simulate_fleet_intervals(settings, _units(1), inputs, public_top_up=_TOP_UP)


def test_offset_change_in_warmup_departure_before_horizon_start_conserves_energy() -> None:
    # Fall-back on 25 October 2026 inside the warm-up.  The horizon starts at
    # 24 Oct 12:00 UTC (13:00 BST), 48 UTC slots before the 12:00 GMT study
    # start on 25 October (decision 0004 item 52).  A 12:30 BST departure on
    # London day 0 (24 October) starts before the horizon: its 30-minute
    # outbound leg is entirely before it.  With no connection, only the
    # in-horizon part of the trip (the 3.75 kWh return leg) reduces the 5.0
    # kWh opening stock to 1.25 kWh.  Under item 32's original 2 kWh (20%)
    # threshold that was below it and topped up to 7.25 kWh; decision 0004
    # item 37 lowered the threshold to 1 kWh (10%), so 1.25 kWh no longer
    # needs a top-up and the study simply opens at 1.25 kWh versus 5.0 kWh
    # without the trip.
    settings = replace(_settings(worlds=1, units=1), start_local_date=date(2026, 10, 25))
    departure = _london("2026-10-24 12:30")
    assert departure < pd.Timestamp("2026-10-24 12:00", tz="UTC")

    no_trip = simulate_fleet_intervals(
        settings, _units(1), _empty_inputs(settings), public_top_up=_TOP_UP
    )[0]
    inputs = _empty_inputs(settings)
    _set_trip(
        inputs,
        0,
        0,
        0,
        departure,
        outbound=15.0,
        returning=15.0,
        speed=30.0,
        dwell_minutes=30.0,
    )
    fleet = simulate_fleet_intervals(settings, _units(1), inputs, public_top_up=_TOP_UP)[0]

    assert len(fleet) == 2 * 48
    assert no_trip["opening_battery_kwh"].iloc[0] == pytest.approx(5.0, abs=1e-12)
    assert fleet["opening_battery_kwh"].iloc[0] == pytest.approx(1.25, abs=1e-12)
    assert fleet["conservation_residual_kwh"].abs().max() <= 1e-9
    assert fleet["closing_soc_percent"].between(0.0, 100.0).all()


@pytest.mark.parametrize(
    "start_local_date",
    [date(2026, 3, 27), date(2026, 3, 30), date(2026, 10, 25), date(2026, 10, 26)],
)
def test_explicit_no_action_runner_smoke_across_offset_change(start_local_date: date) -> None:
    fixture = assumptions.cohort_fixture()
    settings = replace(_settings(worlds=2, days=7, units=6), start_local_date=start_local_date)
    temperature, weather_sd = _weather(
        warmup_days=settings.warmup_days,
        study_days=settings.study_days,
    )

    result = simulate_forecast(
        settings,
        fixture,
        daily_miles_cv_by_cohort=_default_inputs()["daily_miles_cv_by_cohort"],
        personal_mileage_cv_by_cohort=_default_inputs()["personal_mileage_cv_by_cohort"],
        trip_behaviour=_default_inputs()["trip_behaviour"],
        base_temperature_c_by_day=temperature,
        weather_sd_c=weather_sd,
        public_charge_assumptions=_default_inputs()["public_charge_assumptions"],
        model="no_action",
    )

    boundaries = pd.DatetimeIndex(utc_half_hour_boundaries(start_local_date, 1, 7))
    fleet = result.fleet_world_intervals
    assert fleet["conservation_residual_kwh"].abs().max() <= 1e-9
    assert fleet["closing_soc_percent"].between(0.0, 100.0).all()
    for _, group in fleet.groupby(["world_id", "path_id"]):
        group = group.sort_values("interval_start_utc")
        assert len(group) == 7 * 48
        pdt.assert_index_equal(
            pd.DatetimeIndex(group["interval_start_utc"]),
            boundaries[48:-1],
            check_names=False,
        )
        np.testing.assert_allclose(
            group["opening_battery_kwh"].to_numpy()[1:],
            group["closing_battery_kwh"].to_numpy()[:-1],
            rtol=0.0,
            atol=1e-9,
        )
    # The study opens at London noon (decision 0004 item 52) whatever the
    # offset, and the warm-up's unmanaged import is reported per warm-up slot.
    first = pd.Timestamp(boundaries[48]).tz_convert("Europe/London")
    assert (first.date(), first.hour, first.minute) == (start_local_date, 12, 0)
    assert result.warmup_home_import_kwh.shape == (2, 48)


def test_unit_intervals_sum_to_fleet_world_intervals_with_smart_charging() -> None:
    settings = _settings(days=3)
    units = _units(settings.vehicle_count)
    inputs = _representative_inputs(settings)
    # Any day-ahead prices will do for reconciliation: a daily cosine with a
    # different wiggle in each world (each world plans on its own path,
    # decision 0004 item 48), over every warm-up and study slot, as published
    # (plan B4).  Expected departures are 08:00 GMT less a 2 h margin.
    slots = np.arange((settings.warmup_days + settings.study_days) * 48)
    worlds = np.arange(settings.evaluation_world_count)[:, np.newaxis]
    prices = (
        90.0 + 50.0 * np.cos(2 * np.pi * (slots / 2 - 3.0) / 24) + 20.0 * np.sin(slots + worlds)
    )
    smart = smart_charging_inputs(settings, units, prices, departure_margin_hours=2.0)

    per_ev = simulate_unit_intervals(
        settings, units, inputs, smart_charging=smart, public_top_up=_TOP_UP
    )
    fleet = simulate_fleet_intervals(
        settings, units, inputs, smart_charging=smart, public_top_up=_TOP_UP
    )[0]

    assert set(per_ev) == set(UNIT_INTERVAL_QUANTITIES)
    selected = fleet.loc[fleet["path_id"].eq("selected")].sort_values(
        ["world_id", "interval_start_utc"]
    )
    for name in ("home_grid_import_kwh", "public_grid_import_kwh", "closing_battery_kwh"):
        np.testing.assert_allclose(
            per_ev[name].sum(axis=2).ravel(), selected[name].to_numpy(), atol=1e-12
        )
    np.testing.assert_allclose(
        per_ev["unserved_travel_battery_kwh"].sum(axis=2).ravel(),
        (
            selected["unserved_outbound_travel_battery_kwh"]
            + selected["unserved_return_travel_battery_kwh"]
        ).to_numpy(),
        atol=1e-12,
    )
    # Without smart charging the per-EV pass is the normal path.
    normal = fleet.loc[fleet["path_id"].eq("normal")].sort_values(
        ["world_id", "interval_start_utc"]
    )
    unplanned = simulate_unit_intervals(settings, units, inputs, public_top_up=_TOP_UP)
    np.testing.assert_allclose(
        unplanned["home_grid_import_kwh"].sum(axis=2).ravel(),
        normal["home_grid_import_kwh"].to_numpy(),
        atol=1e-12,
    )


def test_warmup_need_skips_sessions_before_the_run_and_splits_at_night_zero_decision() -> None:
    # SYNTHETIC three EVs, need (10 - 4) / 0.5 = 12 kWh each at plug-in.  EV
    # 0 is connected from the first run slot (its session began before the
    # run, plug-in stock unknown), EV 1 plugs in at slot 3, before night 0's
    # day-ahead decision at run slot 5, and EV 2 at slot 7, after it.
    zeros = np.zeros((1, 3))
    trading = {
        "warmup_need_kwh": zeros.copy(),
        "warmup_session_count": np.zeros((1, 3), dtype=np.int64),
        "first_day_ahead_cutoff_slot": np.int64(5),
        "first_day_ahead_need_kwh": zeros.copy(),
        "first_day_ahead_session_count": np.zeros((1, 3), dtype=np.int64),
    }
    stock, target = np.full((1, 3), 4.0), np.full(3, 10.0)
    previous = np.zeros((1, 3), dtype=bool)
    for slot in range(10):
        connected = np.array([[True, slot >= 3, slot >= 7]])
        physics._record_trading(
            trading, slot, 20, None, previous, connected, stock, stock, target, 0.5
        )
        previous = connected
    np.testing.assert_array_equal(trading["warmup_session_count"], [[0, 1, 1]])
    np.testing.assert_allclose(trading["warmup_need_kwh"], [[0.0, 12.0, 12.0]])
    np.testing.assert_array_equal(trading["first_day_ahead_session_count"], [[0, 1, 0]])
    np.testing.assert_allclose(trading["first_day_ahead_need_kwh"], [[0.0, 12.0, 0.0]])
