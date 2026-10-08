"""Model step 2 (decision 0007): the optional "timed" path beyond the four interval
frames slice 1 gave it. Zones, charge completion, session distribution, smart-charging
price paid, the cost effect and household outcomes all learn it the same way: a frame
that already carries ``path_id`` gains an optional ``timed`` row; a wide frame gains
optional ``timed_*`` columns; the cost effect gets a sibling built by the same function
with ``timed`` in the selected slot. Nothing here draws a random number: the timed pass
reuses the same sampled inputs as normal and selected (common random numbers).

Hand-checkable cases first (per-EV reconciliation at the kernel level, the cost effect
against a hand calculation, a session too short to complete under the timed path), then
one full-pipeline run with the setting set and one with it unset, compared frame by
frame.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import date
from functools import cache

import numpy as np
import pandas as pd
import pandas.testing as pdt
from fixtures.result_contract import validate_result_v2

from axle_studio.model.action import calculate_wholesale_world_cost_effect, timed_start_allowed
from axle_studio.model.forecast import ForecastResult, run_forecast_from_assumptions
from axle_studio.model.physics import (
    UNIT_INTERVAL_QUANTITIES,
    PublicTopUp,
    simulate_fleet_intervals,
    simulate_unit_intervals,
)
from axle_studio.model.settings import RunSettings

_TOP_UP = PublicTopUp(efficiency_fraction=0.9, threshold_soc_fraction=0.1, target_soc_fraction=0.8)
_WINTER = date(2025, 1, 7)  # GMT, so London clock time equals UTC
_START = date(2026, 1, 12)  # a Monday, no clock change
_SMALL = {"vehicle_count": 12, "evaluation_world_count": 2}


# --------------------------------------------------------------------------
# Small hand-built kernel fixture: two EVs, 10 kWh battery, 4 kW charger,
# 100 % charging efficiency, so every kWh is checkable by eye (the
# test_timed_start.py / test_smart_charging.py pattern).
# --------------------------------------------------------------------------


def _settings(
    *, start: date = _WINTER, days: int = 2, warmup: int = 1, units: int = 2
) -> RunSettings:
    return RunSettings(
        start_local_date=start,
        warmup_days=warmup,
        study_days=days,
        vehicle_count=units,
        seed=1,
        evaluation_world_count=1,
        opening_soc_fraction=0.2,
        reserve_soc_fraction=0.1,
        home_charge_efficiency=1.0,
        weather_efficiency_sensitivity_fraction_per_c=0.0,
    )


def _units(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population_id": [f"population:timed-policy-{i}" for i in range(count)],
            "unit_id": [f"ev-{i:03d}" for i in range(count)],
            "cohort_id": ["average_uk"] * count,
            "physical_capacity_kwh": [10.0] * count,
            "efficiency_miles_per_battery_kwh": [1.0] * count,
            "home_charger_limit_kw": [4.0] * count,
            "preferred_target_soc_fraction": [0.8] * count,
            "runtime_departure_local_hour": [7] * count,
            "runtime_arrival_local_hour": [18] * count,
        }
    )


def _inputs(settings: RunSettings) -> dict[str, np.ndarray]:
    shape = (1, settings.sampled_day_count, settings.vehicle_count)
    timestamps = np.full(shape, pd.NaT, dtype=object)
    return {
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


def _session(inputs: dict[str, np.ndarray], ev: int, day: int, start: str, end: str) -> None:
    inputs["connection_session_accepted"][0, day, ev] = True
    inputs["connection_start_utc"][0, day, ev] = pd.Timestamp(start, tz="UTC")
    inputs["connection_end_utc"][0, day, ev] = pd.Timestamp(end, tz="UTC")


def test_per_ev_timed_pass_sums_to_the_fleet_timed_path() -> None:
    """``simulate_unit_intervals``'s ``charge_allowed`` mask reconciles with the fleet's
    timed path exactly as the normal and selected passes already do (the
    ``test_unit_intervals_sum_to_fleet_world_intervals_with_smart_charging`` pattern)."""

    settings = _settings(days=2, warmup=1, units=2)
    inputs = _inputs(settings)
    _session(inputs, 0, 1, "2025-01-07 18:00", "2025-01-08 07:00")
    _session(inputs, 1, 1, "2025-01-07 20:00", "2025-01-08 06:00")
    _session(inputs, 0, 2, "2025-01-08 18:00", "2025-01-09 07:00")
    mask = timed_start_allowed(settings, 0.0)
    units = _units(2)

    per_ev = simulate_unit_intervals(
        settings, units, inputs, public_top_up=_TOP_UP, charge_allowed=mask
    )
    fleet = simulate_fleet_intervals(
        settings, units, inputs, public_top_up=_TOP_UP, timed_start_allowed=mask
    )[0]

    assert set(per_ev) == set(UNIT_INTERVAL_QUANTITIES)
    timed = fleet.loc[fleet["path_id"].eq("timed")].sort_values("interval_start_utc")
    for name in ("home_grid_import_kwh", "public_grid_import_kwh", "closing_battery_kwh"):
        np.testing.assert_allclose(
            per_ev[name].sum(axis=2).ravel(), timed[name].to_numpy(), atol=1e-12
        )
    np.testing.assert_allclose(
        per_ev["unserved_travel_battery_kwh"].sum(axis=2).ravel(),
        (
            timed["unserved_outbound_travel_battery_kwh"]
            + timed["unserved_return_travel_battery_kwh"]
        ).to_numpy(),
        atol=1e-12,
    )
    # Per-EV home import never breaks the barred-window rule either: a
    # session this leaves short is this test's other case, below.
    assert (per_ev["home_grid_import_kwh"] >= 0.0).all()


def test_timed_completion_share_is_at_or_below_unmanaged_when_a_session_is_too_short() -> None:
    """A session long enough to reach target on the normal path but barred for most of its
    connected window on the timed path leaves the timed path short: completion can only
    be worse, never better, since the timed rule is the normal rule with charging
    additionally blocked in some half-hours."""

    settings = _settings(days=1, warmup=1, units=1)
    inputs = _inputs(settings)
    # Connected 11:00-13:30: five half-hour slots, but only the first two
    # (11:00-12:00) are allowed before the daily barred stretch starts at
    # noon (T=0): the same scenario test_timed_start.py proves at the mask
    # level, read here through completion share.
    _session(inputs, 0, 1, "2025-01-07 11:00", "2025-01-07 13:30")
    mask = timed_start_allowed(settings, 0.0)
    units = _units(1)
    target_kwh = 0.8 * 10.0

    fleet = simulate_fleet_intervals(
        settings, units, inputs, public_top_up=_TOP_UP, timed_start_allowed=mask
    )[0]
    last_slot_start = pd.Timestamp("2025-01-07 13:00", tz="UTC")
    normal_closing = fleet.loc[
        fleet["path_id"].eq("normal") & fleet["interval_start_utc"].eq(last_slot_start),
        "closing_battery_kwh",
    ].iat[0]
    timed_closing = fleet.loc[
        fleet["path_id"].eq("timed") & fleet["interval_start_utc"].eq(last_slot_start),
        "closing_battery_kwh",
    ].iat[0]

    assert normal_closing >= target_kwh - 1e-9  # Unmanaged: unplugged at 13:30, this
    # session is complete (five uninterrupted slots).
    assert timed_closing < target_kwh  # Timed: barred by noon, short of target.
    normal_complete = normal_closing >= target_kwh - 1e-6
    timed_complete = timed_closing >= target_kwh - 1e-6
    assert int(timed_complete) <= int(normal_complete)


def test_timed_cost_effect_equals_a_hand_calculation() -> None:
    """``calculate_wholesale_world_cost_effect`` with ``selected_path_id="timed"``
    values the timed path exactly the same way the ordinary cost effect values the
    selected path (the ``test_world_cost_effect_uses_same_world_interval_price_for_both_paths``
    pattern), so the two savings are directly comparable."""

    starts = pd.to_datetime(["2026-01-12 00:00Z", "2026-01-12 00:30Z", "2026-01-12 00:00Z"])
    worlds = [0, 0, 1]
    normal_import = [2.0, 1.0, 4.0]
    timed_import = [1.0, 3.0, 2.0]
    base = pd.DataFrame(
        {
            "world_id": worlds,
            "interval_start_utc": starts,
            "interval_end_utc": starts + pd.Timedelta(minutes=30),
        }
    )
    fleet = pd.concat(
        (
            base.assign(
                path_id="normal",
                home_grid_import_kwh=normal_import,
                public_grid_import_kwh=[0.0, 0.0, 1.0],
                unserved_outbound_travel_battery_kwh=[0.0, 0.0, 0.0],
                unserved_return_travel_battery_kwh=[0.0, 0.0, 0.0],
                closing_battery_kwh=[10.0, 11.0, 20.0],
            ),
            base.assign(
                path_id="timed",
                home_grid_import_kwh=timed_import,
                public_grid_import_kwh=[0.0, 0.5, 3.0],
                unserved_outbound_travel_battery_kwh=[0.0, 0.25, 0.0],
                unserved_return_travel_battery_kwh=[0.0, 0.0, 0.5],
                closing_battery_kwh=[9.0, 11.0, 18.0],
            ),
        ),
        ignore_index=True,
    )
    prices = base.assign(wholesale_forecast_gbp_per_mwh=[100.0, 200.0, -50.0])

    effect = calculate_wholesale_world_cost_effect(
        fleet, prices, public_charge_gbp_per_kwh=0.5, selected_path_id="timed"
    )

    # Same hand values as the selected-path test (same import/price numbers,
    # "timed" standing in for "selected" throughout): the function does not
    # care which path name it is handed, only that it is paired against normal.
    expected = pd.DataFrame(
        {
            "world_id": [0, 1],
            "normal_home_import_kwh": [3.0, 4.0],
            "selected_home_import_kwh": [4.0, 2.0],
            "illustrative_selected_minus_normal_energy_cost_gbp": [0.3, 0.1],
            "selected_minus_normal_closing_battery_kwh": [0.0, -2.0],
            "selected_minus_normal_public_import_kwh": [0.5, 2.0],
            "selected_minus_normal_unserved_travel_kwh": [0.25, 0.5],
            "illustrative_selected_minus_normal_public_charge_cost_gbp": [0.25, 1.0],
            "illustrative_unrecovered_energy_value_gbp": [0.0, 1.0],
            "illustrative_unserved_travel_value_gbp": [0.125, 0.25],
            "illustrative_selected_minus_normal_total_gbp": [0.675, 2.35],
            "energy_not_recovered": [True, True],
            "unrecovered_kwh": [0.75, 4.5],
            "unrecovered_share": [0.25, 1.125],
            "not_recovered_material": [True, True],
            "evidence_kind": ["illustrative_synthetic", "illustrative_synthetic"],
        }
    )
    pdt.assert_frame_equal(effect, expected)


# --------------------------------------------------------------------------
# timed_cost_effect["sessions_affected_count"] and timed_not_recovered_summary
# against a real run, with the rigour test_not_recovered.py applies to the
# ordinary path (decision 0004 item 45; BLOCKING review fix, money review):
# an earlier contract draft wrongly said this column was "not added, always
# 0"; it is a real per-world count, paired against timed instead of selected.
# --------------------------------------------------------------------------


@cache
def _cost_result() -> ForecastResult:
    """Sixty EVs, four worlds (the ``test_not_recovered.py`` fixture size): large enough
    that sessions-affected counts are non-trivial rather than mostly zero."""

    return run_forecast_from_assumptions(
        _START,
        values={"vehicle_count": 60, "evaluation_world_count": 4, "timed_start_local_hour": 0.0},
    )


def test_timed_cost_effect_sessions_affected_count_is_a_real_count_paired_against_timed() -> None:
    result = _cost_result()
    cost = result.cost_effect
    timed_cost = result.timed_cost_effect
    assert timed_cost["sessions_affected_count"].dtype == np.int64
    # A real, independent per-world count: not uniformly zero (the wrongly
    # documented "always 0" placeholder) and not merely a copy of the
    # selected-path reading, since it is paired against a different path.
    assert timed_cost["sessions_affected_count"].ne(0).any()
    assert not timed_cost["sessions_affected_count"].equals(cost["sessions_affected_count"])


def test_timed_not_recovered_summary_matches_an_independent_recomputation() -> None:
    result = _cost_result()
    timed_cost = result.timed_cost_effect
    assert result.timed_not_recovered_world_count == int(timed_cost["not_recovered_material"].sum())
    assert (timed_cost["not_recovered_material"] <= timed_cost["energy_not_recovered"]).all()
    summary = result.timed_not_recovered_summary.set_index("metric")
    assert list(summary.index) == [
        "unrecovered_kwh",
        "unrecovered_share",
        "sessions_affected_count",
    ]
    # World-first (decision 0004 item 12): quantiles recomputed here directly
    # from the per-world column, not trusted from the summary function.
    for metric in summary.index:
        values = timed_cost[metric].to_numpy(dtype=float)
        np.testing.assert_allclose(
            summary.loc[metric, ["p10", "p50", "p90"]].to_numpy(dtype=float),
            np.quantile(values, (0.1, 0.5, 0.9), method="linear"),
        )


# --------------------------------------------------------------------------
# Full pipeline: the setting set passes every contract validator and adds
# precisely the documented rows/columns; the setting unset has none of them
# and every frame is otherwise bit-for-bit unchanged.
# --------------------------------------------------------------------------


def _run(*, timed_start_local_hour: float) -> ForecastResult:
    # NaN means "policy off": the assumptions layer takes that from the
    # "Timed tariff policy" switch, never from an unset start hour.
    if math.isnan(timed_start_local_hour):
        values = {**_SMALL, "timed_tariff_enabled": 0}
    else:
        values = {**_SMALL, "timed_start_local_hour": timed_start_local_hour}
    return run_forecast_from_assumptions(_START, values=values)


def test_full_run_with_the_setting_set_passes_the_validator_and_adds_model_step_2_frames() -> None:
    result = _run(timed_start_local_hour=0.0)
    validate_result_v2(result)

    for name in (
        "fleet_world_intervals",
        "cohort_world_intervals",
        "fleet_interval_bands",
        "cohort_interval_bands",
        "zone_world_intervals",
        "zone_import_bands",
        "zone_summary",
        "weekly_peak_summary",
        "charge_completion_summary",
    ):
        paths = set(getattr(result, name)["path_id"])
        assert "timed" in paths, name

    assert "timed_kwh_p50" in result.price_band_shift.columns
    assert "timed_average_price_gbp_per_mwh" in result.smart_charging_world.columns
    assert "timed_average_price_gbp_per_mwh" in set(result.smart_charging_summary["metric"])
    assert "departure_soc_percent_timed" in set(result.session_distribution_bands["metric"])

    assert result.timed_cost_effect is not None
    assert result.timed_cost_effect_summary is not None
    assert result.timed_not_recovered_summary is not None
    assert len(result.timed_cost_effect) == result.world_count
    assert "sessions_affected_count" in result.timed_cost_effect.columns
    assert result.timed_not_recovered_world_count == int(
        result.timed_cost_effect["not_recovered_material"].sum()
    )

    if result.household_ev_world is not None:
        for column in (
            "home_import_kwh_timed",
            "cheap_import_kwh_timed",
            "public_import_kwh_timed",
            "home_cost_gbp_timed",
            "public_cost_gbp_timed",
            "completed_count_timed",
            "departure_soc_mean_timed",
        ):
            assert column in result.household_ev_world.columns, column
        assert "home_cost_gbp_per_kwh_timed" in set(result.household_outcomes_summary["metric"])

    # Strictly pairwise frames (item 6) never pick up "timed".
    for name in (
        "difference_bands",
        "fleet_interval_ev_bands",
        "flexibility_bands",
        "weekly_bands",
        "average_day_bands",
    ):
        frame = getattr(result, name)
        if frame is not None and "path_id" in frame.columns:
            assert "timed" not in set(frame["path_id"]), name


def test_full_run_with_the_setting_unset_has_none_of_the_model_step_2_additions() -> None:
    result = _run(timed_start_local_hour=float("nan"))
    validate_result_v2(result)

    for name in (
        "fleet_world_intervals",
        "zone_world_intervals",
        "zone_import_bands",
        "zone_summary",
        "weekly_peak_summary",
        "charge_completion_summary",
    ):
        assert "timed" not in set(getattr(result, name)["path_id"]), name
    assert "timed_kwh_p50" not in result.price_band_shift.columns
    assert "timed_average_price_gbp_per_mwh" not in result.smart_charging_world.columns
    assert "departure_soc_percent_timed" not in set(result.session_distribution_bands["metric"])
    assert result.timed_cost_effect is None
    assert result.timed_cost_effect_summary is None
    assert result.timed_not_recovered_summary is None
    assert result.timed_not_recovered_world_count is None
    if result.household_ev_world is not None:
        assert not any(c.endswith("_timed") for c in result.household_ev_world.columns)


# Frames whose row or column set is documented to differ with the setting
# set (decision 0007, model step 2); every other DataFrame field of
# ``ForecastResult`` must be bit-for-bit identical either way.
_TOUCHED_PATH_ID_FRAMES = (
    "fleet_world_intervals",
    "cohort_world_intervals",
    "fleet_interval_bands",
    "cohort_interval_bands",
    "zone_world_intervals",
    "zone_import_bands",
    "zone_summary",
    "weekly_peak_summary",
    "charge_completion_summary",
)
_TIMED_ONLY_FIELDS = (
    "timed_cost_effect",
    "timed_cost_effect_summary",
    "timed_not_recovered_summary",
)


def _without_timed(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` (a field of the "set" result) with the documented timed addition removed."""

    if name in _TOUCHED_PATH_ID_FRAMES:
        frame = frame.loc[frame["path_id"].ne("timed")]
    elif name == "price_band_shift":
        frame = frame.drop(columns=[c for c in frame.columns if c.startswith("timed_kwh_")])
    elif name == "session_distribution_bands":
        frame = frame.loc[frame["metric"].ne("departure_soc_percent_timed")]
    elif name == "smart_charging_world":
        frame = frame.drop(columns=["timed_average_price_gbp_per_mwh"], errors="ignore")
    elif name == "smart_charging_summary":
        frame = frame.loc[frame["metric"].ne("timed_average_price_gbp_per_mwh")]
    elif name in ("household_ev_world",):
        frame = frame.drop(columns=[c for c in frame.columns if c.endswith("_timed")])
    elif name == "household_outcomes_summary":
        frame = frame.loc[~frame["metric"].str.endswith("_timed")]
    return frame.reset_index(drop=True)


def test_frames_match_with_the_setting_set_or_unset_apart_from_documented_additions() -> None:
    on = _run(timed_start_local_hour=0.0)
    off = _run(timed_start_local_hour=float("nan"))

    checked = 0
    for field in dataclasses.fields(ForecastResult):
        name = field.name
        if name in _TIMED_ONLY_FIELDS:
            continue
        value_on = getattr(on, name)
        if not isinstance(value_on, pd.DataFrame):
            continue
        value_off = getattr(off, name)
        assert isinstance(value_off, pd.DataFrame), name
        trimmed = _without_timed(name, value_on)
        pdt.assert_frame_equal(trimmed, value_off.reset_index(drop=True), obj=name)
        checked += 1
    # A sanity floor so a refactor that silently stops iterating fields (for
    # example an empty dataclass) cannot pass this test by doing nothing.
    assert checked >= 40
