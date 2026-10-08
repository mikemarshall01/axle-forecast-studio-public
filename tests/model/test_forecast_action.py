from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from axle_studio.model import assumptions
from axle_studio.model.action import (
    ILLUSTRATIVE_PUBLIC_CHARGE_GBP_PER_KWH,
    calculate_wholesale_world_cost_effect,
)
from axle_studio.model.forecast import run_forecast_from_assumptions, simulate_forecast
from axle_studio.model.settings import RunSettings


def _default_inputs() -> dict:
    # Fresh run inputs from model/assumptions.py (the removed JSON configs'
    # values) on every call, so a test that edits them cannot leak into another.
    return assumptions.forecast_inputs(None, warmup_days=1, study_days=7)


def _weather(*, warmup_days: int, study_days: int):
    inputs = assumptions.forecast_inputs(None, warmup_days=warmup_days, study_days=study_days)
    return inputs["base_temperature_c_by_day"], inputs["weather_sd_c"]


def _settings(**changes: object) -> RunSettings:
    settings = RunSettings(
        start_local_date=date(2026, 1, 12),
        warmup_days=1,
        study_days=7,
        vehicle_count=100,
        seed=73,
        evaluation_world_count=3,
        opening_soc_fraction=0.55,
        reserve_soc_fraction=0.2,
        home_charge_efficiency=0.92,
        weather_efficiency_sensitivity_fraction_per_c=0.01,
    )
    return replace(settings, **changes)


def _arguments(settings: RunSettings):
    fixture = assumptions.cohort_fixture()
    temperature, weather_sd = _weather(
        warmup_days=settings.warmup_days,
        study_days=settings.study_days,
    )
    return fixture, {
        "daily_miles_cv_by_cohort": _default_inputs()["daily_miles_cv_by_cohort"],
        "personal_mileage_cv_by_cohort": _default_inputs()["personal_mileage_cv_by_cohort"],
        "trip_behaviour": _default_inputs()["trip_behaviour"],
        "base_temperature_c_by_day": temperature,
        "weather_sd_c": weather_sd,
        # Default threshold re-baselined from 20 % to 10 % by decision 0004 item 37.
        "public_charge_assumptions": {
            "efficiency_fraction": 0.9,
            "top_up_threshold_soc_fraction": 0.1,
            "top_up_target_soc_fraction": 0.8,
        },
        "price_assumptions": _default_inputs()["price_assumptions"],
        "departure_margin_hours": 1.0,
    }


def _run(
    *,
    public_charge_assumptions: dict[str, float] | None = None,
    price_assumptions: dict[str, float] | None = None,
    public_charge_gbp_per_kwh: object = None,
    trading: dict[str, float] | None = None,
    **settings_changes: object,
):
    settings = _settings(**settings_changes)
    fixture, arguments = _arguments(settings)
    if trading is not None:
        current = arguments.get("trading_assumptions") or assumptions.trading_inputs(None)
        arguments["trading_assumptions"] = dict(current) | trading
    if public_charge_assumptions is not None:
        arguments["public_charge_assumptions"] = public_charge_assumptions
    if price_assumptions is not None:
        arguments["price_assumptions"] = price_assumptions
    if public_charge_gbp_per_kwh is not None:
        arguments["public_charge_gbp_per_kwh"] = public_charge_gbp_per_kwh
    # The raw action simulation: kernel column names, evaluation worlds kept.
    return simulate_forecast(settings, fixture, **arguments, model="action")


def _assert_array_equal(left: np.ndarray, right: np.ndarray) -> None:
    if left.dtype == object:
        assert all(
            (pd.isna(actual) and pd.isna(expected)) or actual == expected
            for actual, expected in zip(left.flat, right.flat, strict=True)
        )
    else:
        np.testing.assert_array_equal(left, right)


def test_action_run_returns_paired_world_first_intervals() -> None:
    result = _run()

    assert result.model == "action"
    assert set(result.fleet_world_intervals["path_id"]) == {"normal", "selected"}
    assert len(result.fleet_world_intervals) == 3 * 2 * 7 * 48
    assert result.fleet_world_intervals["interval_start_utc"].nunique() == 7 * 48
    assert str(result.fleet_world_intervals["interval_start_utc"].dt.tz) == "UTC"
    assert set(result.fleet_world_intervals["world_id"]) == {0, 1, 2}
    assert result.units["cohort_id"].nunique() == 6
    assert result.units["population_id"].nunique() == 1
    assert set(result.cohort_world_intervals["cohort_id"]) == set(result.units["cohort_id"])
    assert result.fleet_interval_bands["world_count"].eq(3).all()
    assert result.cohort_interval_distribution["world_count"].eq(3).all()

    # One warm-up day, seven study nights and the morning after the last
    # night (RunSettings.sampled_day_count, decision 0004 item 52).
    assert result.evaluation.inputs["drives_today"].shape == (3, 9, 100)
    # Prices cover the warm-up too; the smart charger plans from the last
    # warm-up night, i.e. the whole one-day warm-up here.
    assert len(result.forecast_prices) == 3 * 8 * 48
    assert (result.smart_charging.price_epoch_by_slot >= 0).all()
    assert result.warmup_home_import_kwh.shape == (3, 48)
    assert len(result.warmup_slots) == 48
    assert result.smart_charging.expected_departure_utc_ns.shape == (10, 100)
    assert result.fleet_world_intervals["conservation_residual_kwh"].abs().max() <= 1e-12


def test_smart_charging_moves_home_import_to_cheaper_forecast_half_hours() -> None:
    result = _run(vehicle_count=6)

    # Each world is priced at its own day-ahead path (decision 0004 item 48).
    world = result.fleet_world_intervals.merge(
        result.forecast_prices[
            ["world_id", "interval_start_utc", "wholesale_forecast_gbp_per_mwh"]
        ],
        on=["world_id", "interval_start_utc"],
    ).rename(columns={"wholesale_forecast_gbp_per_mwh": "price"})
    paid = {
        path: float((rows["home_grid_import_kwh"] * rows["price"]).sum())
        / float(rows["home_grid_import_kwh"].sum())
        for path, rows in world.groupby("path_id")
    }
    assert paid["selected"] < paid["normal"]
    assert result.fleet_world_intervals["conservation_residual_kwh"].abs().max() <= 1e-12


def test_public_charge_is_explicit_physical_flow_with_conservation() -> None:
    result = _run(opening_soc_fraction=0.2, study_days=1, evaluation_world_count=2)
    intervals = result.fleet_world_intervals

    assert intervals["public_grid_import_kwh"].sum() > 0.0
    np.testing.assert_allclose(
        intervals["public_battery_added_kwh"],
        intervals["public_grid_import_kwh"]
        * 0.9,  # the public charging efficiency passed in ``_arguments``
    )
    np.testing.assert_allclose(
        intervals["realised_grid_kw"],
        2.0 * (intervals["home_grid_import_kwh"] + intervals["public_grid_import_kwh"]),
    )
    # Decision 0004 item 32: top-up time is not modelled.
    assert intervals["public_charging_fraction"].eq(0.0).all()
    assert intervals["conservation_residual_kwh"].abs().max() <= 1e-12


def test_selected_path_and_cost_do_not_depend_on_intraday_or_imbalance_prices() -> None:
    # Decision-time cut-off: the market prices are drawn after the worlds
    # and the day-ahead planner reads only its world's day-ahead prices, so
    # this holds for the day-ahead plan path, the selected path with
    # intraday dispatch off (with it on, free EVs follow the intraday price
    # by design, intraday-dispatch-v1 §4).  Home
    # import is valued at those day-ahead prices too (plan B3, decision 0004
    # item 53), so intraday news, imbalance and surprise shocks change the
    # realised frame and nothing physical or priced.
    quiet = {
        **_default_inputs()["price_assumptions"],
        "intraday_hourly_sd_gbp_per_mwh": 0.0,
        "big_shock_known_share": 0.0,
    }
    # Only channels that never enter day-ahead: intraday news, the
    # imbalance tail and the NIV link (surprise sizes do enter day-ahead,
    # through its expected surprise impact).
    noisy = {
        **quiet,
        "intraday_hourly_sd_gbp_per_mwh": 6.0,
        "imbalance_tail_scale_gbp_per_mwh": 50.0,
        "niv_surprise_shift": 0.0,
    }

    small = {"study_days": 1, "evaluation_world_count": 2, "vehicle_count": 6}
    off = {"trading.intraday_dispatch": 0.0}
    deterministic = _run(price_assumptions=quiet, trading=off, **small)
    moved = _run(price_assumptions=noisy, trading=off, **small)

    pdt.assert_frame_equal(deterministic.fleet_world_intervals, moved.fleet_world_intervals)
    known = deterministic.market_prices.known_shock_gw
    np.testing.assert_array_equal(known, moved.market_prices.known_shock_gw)
    pdt.assert_frame_equal(deterministic.forecast_prices, moved.forecast_prices)
    assert not deterministic.evaluation_prices.equals(moved.evaluation_prices)
    pdt.assert_frame_equal(deterministic.cost_effect, moved.cost_effect)


def test_fixed_seed_action_run_is_reproducible() -> None:
    first = _run(study_days=1, evaluation_world_count=2)
    repeated = _run(study_days=1, evaluation_world_count=2)

    pdt.assert_frame_equal(first.units, repeated.units)
    for name in first.evaluation.inputs:
        _assert_array_equal(first.evaluation.inputs[name], repeated.evaluation.inputs[name])
    pdt.assert_frame_equal(first.fleet_world_intervals, repeated.fleet_world_intervals)
    pdt.assert_frame_equal(first.cost_effect, repeated.cost_effect)


def test_early_departures_are_priced_as_public_top_ups_or_energy_not_recovered() -> None:
    # Decision 0004 item 38: an EV that leaves before its plan finishes is
    # short, which shows up as extra public import or a lower closing stock
    # and is priced by the cost effect (items 4, 5, 13).
    result = _run(vehicle_count=100, evaluation_world_count=20, seed=3)

    selected = result.fleet_world_intervals.query("path_id == 'selected'")
    early = selected.groupby("world_id")["early_departure_count"].sum()
    assert early.sum() > 0
    effect = result.cost_effect.set_index("world_id")
    flagged = effect.loc[effect["energy_not_recovered"]]
    assert len(flagged) > 0
    priced = (
        flagged["illustrative_unrecovered_energy_value_gbp"]
        + flagged["illustrative_selected_minus_normal_public_charge_cost_gbp"]
    )
    assert (priced > 0).all()
    world = flagged.iloc[0]
    assert world["illustrative_selected_minus_normal_total_gbp"] == pytest.approx(
        world["illustrative_selected_minus_normal_energy_cost_gbp"]
        + world["illustrative_selected_minus_normal_public_charge_cost_gbp"]
        + world["illustrative_unrecovered_energy_value_gbp"]
        + world["illustrative_unserved_travel_value_gbp"]
    )
    last_start = result.fleet_world_intervals["interval_start_utc"].max()
    closing = result.fleet_world_intervals.loc[
        result.fleet_world_intervals["interval_start_utc"].eq(last_start)
    ].pivot(index="world_id", columns="path_id", values="closing_battery_kwh")
    np.testing.assert_allclose(
        effect["selected_minus_normal_closing_battery_kwh"],
        (closing["selected"] - closing["normal"]).sort_index(),
    )


def test_world_cost_effect_uses_same_world_interval_price_for_both_paths() -> None:
    starts = pd.to_datetime(["2026-01-12 00:00Z", "2026-01-12 00:30Z", "2026-01-12 00:00Z"])
    worlds = [0, 0, 1]
    normal_import = [2.0, 1.0, 4.0]
    selected_import = [1.0, 3.0, 2.0]
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
                path_id="selected",
                home_grid_import_kwh=selected_import,
                public_grid_import_kwh=[0.0, 0.5, 3.0],
                unserved_outbound_travel_battery_kwh=[0.0, 0.25, 0.0],
                unserved_return_travel_battery_kwh=[0.0, 0.0, 0.5],
                closing_battery_kwh=[9.0, 11.0, 18.0],
            ),
        ),
        ignore_index=True,
    )
    prices = base.assign(wholesale_forecast_gbp_per_mwh=[100.0, 200.0, -50.0])

    effect = calculate_wholesale_world_cost_effect(fleet, prices, public_charge_gbp_per_kwh=0.5)

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


def _one_interval_pair(normal: dict[str, list[float]], selected: dict[str, list[float]]):
    starts = pd.to_datetime(["2026-01-12 00:00Z"] * 4)
    base = pd.DataFrame(
        {
            "world_id": [0, 1, 2, 3],
            "interval_start_utc": starts,
            "interval_end_utc": starts + pd.Timedelta(minutes=30),
        }
    )
    fleet = pd.concat(
        (base.assign(path_id="normal", **normal), base.assign(path_id="selected", **selected)),
        ignore_index=True,
    )
    return fleet, base.assign(wholesale_forecast_gbp_per_mwh=[100.0] * 4)


def test_cost_columns_signs_no_surplus_credit_and_widened_flag() -> None:
    zeros = [0.0] * 4
    normal = {
        "home_grid_import_kwh": [10.0, 10.0, 10.0, 10.0],
        "public_grid_import_kwh": [0.0, 0.0, 4.0, 0.0],
        "unserved_outbound_travel_battery_kwh": zeros,
        "unserved_return_travel_battery_kwh": zeros,
        "closing_battery_kwh": [50.0, 50.0, 50.0, 50.0],
    }
    selected = {
        # World 0: only more public import. World 1: only more unserved travel.
        # World 2: less public import and a closing surplus (no credit).
        # World 3: closing shortfall only.
        "home_grid_import_kwh": [10.0, 10.0, 12.0, 6.0],
        "public_grid_import_kwh": [2.0, 0.0, 1.0, 0.0],
        "unserved_outbound_travel_battery_kwh": [0.0, 1.0, 0.0, 0.0],
        "unserved_return_travel_battery_kwh": [0.0, 0.5, 0.0, 0.0],
        "closing_battery_kwh": [50.0, 50.0, 53.0, 46.0],
    }
    fleet, prices = _one_interval_pair(normal, selected)

    effect = calculate_wholesale_world_cost_effect(
        fleet, prices, public_charge_gbp_per_kwh=ILLUSTRATIVE_PUBLIC_CHARGE_GBP_PER_KWH
    ).set_index("world_id")

    assert ILLUSTRATIVE_PUBLIC_CHARGE_GBP_PER_KWH == 0.79
    np.testing.assert_allclose(
        effect["illustrative_selected_minus_normal_energy_cost_gbp"], [0.0, 0.0, 0.2, -0.4]
    )
    np.testing.assert_allclose(
        effect["illustrative_selected_minus_normal_public_charge_cost_gbp"],
        [2.0 * 0.79, 0.0, -3.0 * 0.79, 0.0],
    )
    np.testing.assert_allclose(
        effect["illustrative_unrecovered_energy_value_gbp"], [0.0, 0.0, 0.0, 4.0 * 0.79]
    )
    # Added unserved travel is flagged and priced at the public rate.
    assert effect.loc[1, "selected_minus_normal_unserved_travel_kwh"] == 1.5
    np.testing.assert_allclose(
        effect["illustrative_unserved_travel_value_gbp"], [0.0, 1.5 * 0.79, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        effect["illustrative_selected_minus_normal_total_gbp"],
        [1.58, 1.185, 0.2 - 2.37, -0.4 + 3.16],
    )
    assert effect["energy_not_recovered"].tolist() == [True, True, False, True]
    # Decision 0004 item 45: the same energy as a magnitude.  World 2's lower
    # public import and closing surplus do not offset anything.
    np.testing.assert_allclose(effect["unrecovered_kwh"], [2.0, 1.5, 0.0, 4.0])
    np.testing.assert_allclose(effect["unrecovered_share"], [0.2, 0.15, 0.0, 0.4])
    assert effect["not_recovered_material"].tolist() == [True, True, False, True]


def test_material_threshold_is_a_share_of_weekly_normal_home_import() -> None:
    # Decision 0004 item 45: 1 % by default, inclusive; a week short of energy
    # with no normal home import has no share and counts as material.
    zeros = [0.0] * 4
    normal = {
        "home_grid_import_kwh": [1000.0, 1000.0, 1000.0, 0.0],
        "public_grid_import_kwh": zeros,
        "unserved_outbound_travel_battery_kwh": zeros,
        "unserved_return_travel_battery_kwh": zeros,
        "closing_battery_kwh": [50.0] * 4,
    }
    selected = {**normal, "closing_battery_kwh": [45.0, 40.0, 30.0, 49.0]}
    fleet, prices = _one_interval_pair(normal, selected)

    effect = calculate_wholesale_world_cost_effect(fleet, prices, public_charge_gbp_per_kwh=0.79)

    np.testing.assert_allclose(effect["unrecovered_share"], [0.005, 0.01, 0.02, np.nan])
    assert effect["energy_not_recovered"].all()
    assert effect["not_recovered_material"].tolist() == [False, True, True, True]
    looser = calculate_wholesale_world_cost_effect(
        fleet, prices, public_charge_gbp_per_kwh=0.79, material_share_percent=0.5
    )
    assert looser["not_recovered_material"].all()


def test_less_unserved_travel_is_not_credited() -> None:
    zeros = [0.0] * 4
    normal = {
        "home_grid_import_kwh": zeros,
        "public_grid_import_kwh": zeros,
        "unserved_outbound_travel_battery_kwh": [2.0] * 4,
        "unserved_return_travel_battery_kwh": zeros,
        "closing_battery_kwh": [5.0] * 4,
    }
    selected = {**normal, "unserved_outbound_travel_battery_kwh": [1.0] * 4}
    fleet, prices = _one_interval_pair(normal, selected)

    effect = calculate_wholesale_world_cost_effect(fleet, prices, public_charge_gbp_per_kwh=0.79)

    assert effect["illustrative_unserved_travel_value_gbp"].eq(0.0).all()
    assert effect["illustrative_selected_minus_normal_total_gbp"].eq(0.0).all()
    assert not effect["energy_not_recovered"].any()


def test_differences_within_tolerance_do_not_trip_flag() -> None:
    zeros = [0.0] * 4
    normal = {
        "home_grid_import_kwh": zeros,
        "public_grid_import_kwh": zeros,
        "unserved_outbound_travel_battery_kwh": zeros,
        "unserved_return_travel_battery_kwh": zeros,
        # World 3 sits exactly on the 1e-6 kWh boundary (no rounding noise).
        "closing_battery_kwh": [5.0, 5.0, 5.0, 1e-6],
    }
    selected = {
        **normal,
        "public_grid_import_kwh": [1e-7, 0.0, 0.0, 0.0],
        "unserved_outbound_travel_battery_kwh": [0.0, 1e-7, 0.0, 1e-6],
        "closing_battery_kwh": [5.0, 5.0, 5.0 - 1e-7, 0.0],
    }
    fleet, prices = _one_interval_pair(normal, selected)

    effect = calculate_wholesale_world_cost_effect(fleet, prices, public_charge_gbp_per_kwh=0.79)

    # Unserved and unrecovered differences within 1e-6 kWh are valued at zero,
    # matching the flag.
    assert not effect["energy_not_recovered"].any()
    assert effect["illustrative_unserved_travel_value_gbp"].eq(0.0).all()
    assert effect["illustrative_unrecovered_energy_value_gbp"].eq(0.0).all()


def test_differences_beyond_tolerance_are_flagged_and_valued() -> None:
    zeros = [0.0] * 4
    normal = {
        "home_grid_import_kwh": zeros,
        "public_grid_import_kwh": zeros,
        "unserved_outbound_travel_battery_kwh": zeros,
        "unserved_return_travel_battery_kwh": zeros,
        "closing_battery_kwh": [5.0] * 4,
    }
    selected = {
        **normal,
        "unserved_return_travel_battery_kwh": [0.0, 2e-6, 0.0, 0.0],
        "closing_battery_kwh": [5.0 - 2e-6, 5.0, 5.0, 5.0],
    }
    fleet, prices = _one_interval_pair(normal, selected)

    effect = calculate_wholesale_world_cost_effect(fleet, prices, public_charge_gbp_per_kwh=0.79)

    assert effect["energy_not_recovered"].tolist() == [True, True, False, False]
    np.testing.assert_allclose(
        effect["illustrative_unrecovered_energy_value_gbp"], [2e-6 * 0.79, 0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        effect["illustrative_unserved_travel_value_gbp"], [0.0, 2e-6 * 0.79, 0.0, 0.0]
    )


@pytest.mark.parametrize(
    ("rate", "error"),
    [
        (-0.01, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
    ],
)
def test_rejects_invalid_public_charge_rate(rate: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="public_charge_gbp_per_kwh"):
        _run(study_days=1, evaluation_world_count=1, public_charge_gbp_per_kwh=rate)


def test_one_generator_is_created_only_when_action_run_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axle_studio.model.forecast as module

    original = np.random.default_rng
    calls: list[int] = []

    def counted(seed: int):
        calls.append(seed)
        return original(seed)

    monkeypatch.setattr(module.np.random, "default_rng", counted)
    assert calls == []
    _run(study_days=1, evaluation_world_count=1)
    assert calls == [73]


def test_requires_exact_public_price_mappings() -> None:
    public = {
        "efficiency_fraction": 0.9,
        "top_up_threshold_soc_fraction": 0.1,
        "top_up_target_soc_fraction": 0.8,
        "invented_default": 1.0,
    }
    with pytest.raises(ValueError, match="unknown field: invented_default"):
        _run(study_days=1, evaluation_world_count=1, public_charge_assumptions=public)


@pytest.mark.parametrize(
    ("start", "first_slot_utc"),
    [(date(2026, 10, 22), "2026-10-22 11:00Z"), (date(2026, 3, 26), "2026-03-26 12:00Z")],
)
def test_action_run_across_clock_change_keeps_fixed_utc_slots_and_physics(
    start: date, first_slot_utc: str
) -> None:
    # Decision 0004 item 1: the study crosses the 25 Oct 2026 fall-back (or
    # the 29 Mar 2026 spring-forward).  The fixed 336 UTC slots start at
    # London noon (decision 0004 item 52) and the smart charger's expected departures follow the
    # London clock on each date (checked by hand in test_smart_charging).
    result = _run(start_local_date=start, vehicle_count=40, seed=11)

    starts = result.fleet_world_intervals["interval_start_utc"].drop_duplicates()
    assert len(starts) == 7 * 48
    assert starts.iloc[0] == pd.Timestamp(first_slot_utc)
    departures = pd.to_datetime(result.smart_charging.expected_departure_utc_ns.ravel(), utc=True)
    local = departures.tz_convert("Europe/London")
    # Typical departure clock hour of each date's day type (decision 0004
    # item 42: weekend windows for two cohorts) minus the one-hour margin.
    dates = pd.date_range(start - pd.Timedelta(days=1), periods=10, freq="D")
    weekend = (dates.weekday >= 5)[:, np.newaxis]
    units = result.units
    expected_hours = (
        np.where(
            weekend,
            units["runtime_weekend_departure_local_hour"].to_numpy(),
            units["runtime_departure_local_hour"].to_numpy(),
        )
        - 1
    )
    assert (local.hour.to_numpy().reshape(-1, 40) == expected_hours).all()
    assert (expected_hours[weekend[:, 0]] != expected_hours[~weekend[:, 0]][0]).any()
    frame = result.fleet_world_intervals
    assert frame["conservation_residual_kwh"].abs().max() <= 1e-9
    assert frame["closing_soc_percent"].between(0.0, 100.0).all()
    # Warm-up (one day) and study prices (decision 0004 item 52).
    assert len(result.forecast_prices) == result.settings.evaluation_world_count * 8 * 48


def test_default_margin_keeps_early_departures_a_small_share_of_sessions() -> None:
    # Decision 0004 item 51: the wider, fat-tailed departure clock means more
    # drivers leave before the smart plan finishes; the 2 h default margin
    # keeps that under 6 % of home sessions (about 2-3 % measured).  A small
    # seeded run at every default except its size.
    result = run_forecast_from_assumptions(
        date(2026, 1, 12), values={"vehicle_count": 120, "evaluation_world_count": 6}
    )
    assert result.action_summary.departure_margin_hours == 2.0
    events = result.plug_in_events
    sessions = int(events["path_id"].eq("normal").sum())
    early = float(result.smart_charging_world["early_departure_count"].sum())
    assert sessions > 1_000 and early > 0.0
    assert early / sessions < 0.06
