"""run_forecast and ForecastResult against result contract v2 (fields M4 owns).

The summary fields M5 fills are checked for presence here; the full
``validate_result_v2`` on real results is in ``test_summaries.py``.  These
tests reuse the contract's own specs and reference computations so the
shapes cannot drift from the validator.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from fixtures.result_contract import (
    ADDITIVE_METRICS,
    COHORT_BAND_COLUMNS,
    COST_EFFECT_COLUMNS,
    EVALUATION_PRICE_COLUMNS,
    FLEET_BAND_COLUMNS,
    FLEET_WORLD_COLUMNS,
    FORECAST_PRICE_COLUMNS,
    PRICE_BAND_SHIFT_COLUMNS,
    SETTINGS_KEYS,
    SMART_CHARGING_SUMMARY_COLUMNS,
    SMART_CHARGING_WORLD_COLUMNS,
    STUDY_SLOT_COLUMNS,
    UNIT_COLUMNS,
    WEEKLY_BAND_COLUMNS,
    _assert_close,
    _check_dtypes,
    _check_slot_keys,
    _check_unique,
    _validate_world_units,
    fleet_bands,
    weekly_bands_from_world,
)

from axle_studio.model import assumptions
from axle_studio.model.forecast import ForecastResult, run_forecast, simulate_forecast
from axle_studio.model.settings import RunSettings


def _default_inputs() -> dict:
    # Fresh run inputs from model/assumptions.py (the removed JSON configs'
    # values) on every call, so a test that edits them cannot leak into another.
    return assumptions.forecast_inputs(None, warmup_days=1, study_days=7)


def _weather(*, warmup_days: int, study_days: int):
    inputs = assumptions.forecast_inputs(None, warmup_days=warmup_days, study_days=study_days)
    return inputs["base_temperature_c_by_day"], inputs["weather_sd_c"]


# The default market-price inputs (decision 0004 items 53 and 56).
_PRICES = assumptions.forecast_inputs(None, warmup_days=1, study_days=7)["price_assumptions"]
_M5_FIELDS = (
    "sampled_world_ids",
    "cohort_summary",
    "average_day_bands",
    "plug_in_events",
    "plug_in_world_kpis",
    "plug_in_summary",
    "difference_bands",
    "difference_weekly",
    "difference_weekly_bands",
    "action_summary",
    "cost_effect_summary",
    "smart_charging_world",
    "smart_charging_summary",
    "price_band_shift",
    "forecast_price_bands",
    "weekly_peak_summary",
)
_ACTION_FIELDS = (
    "forecast_prices",
    "evaluation_prices",
    "cost_effect",
)


def _settings(**changes: object) -> RunSettings:
    settings = RunSettings(
        start_local_date=date(2026, 1, 12),
        warmup_days=1,
        study_days=7,
        vehicle_count=60,
        seed=42,
        evaluation_world_count=4,
        opening_soc_fraction=0.55,
        reserve_soc_fraction=0.2,
        home_charge_efficiency=0.92,
        weather_efficiency_sensitivity_fraction_per_c=0.01,
    )
    return replace(settings, **changes)


def _arguments(settings: RunSettings, model: str, prices: dict | None = None):
    fixture = assumptions.cohort_fixture()
    temperature, weather_sd = _weather(
        warmup_days=settings.warmup_days,
        study_days=settings.study_days,
    )
    arguments = {
        "daily_miles_cv_by_cohort": _default_inputs()["daily_miles_cv_by_cohort"],
        "personal_mileage_cv_by_cohort": _default_inputs()["personal_mileage_cv_by_cohort"],
        "trip_behaviour": _default_inputs()["trip_behaviour"],
        "base_temperature_c_by_day": temperature,
        "weather_sd_c": weather_sd,
        # Both models top up in public (decision 0004 item 32); default
        # threshold re-baselined from 20 % to 10 % by item 37.
        "public_charge_assumptions": {
            "efficiency_fraction": 0.9,
            "top_up_threshold_soc_fraction": 0.1,
            "top_up_target_soc_fraction": 0.8,
        },
    }
    if model == "action":
        arguments["price_assumptions"] = prices or _PRICES
        arguments["departure_margin_hours"] = 1.0
    return fixture, arguments


def _run(model: str = "action", prices: dict | None = None, **changes) -> ForecastResult:
    settings = _settings(**changes)
    fixture, arguments = _arguments(settings, model, prices)
    return run_forecast(settings, fixture, **arguments, model=model)


def _check_owned_fields(result: ForecastResult) -> None:
    """Contract v2 checks for every field M4 fills (M5's are only checked present)."""

    is_action = result.model == "action"
    paths = ["normal", "selected"] if is_action else ["normal"]
    assert result.policy_id == (
        "explicit_synthetic_wholesale_v1" if is_action else "explicit_no_action_v1"
    )
    assert result.evidence_kind == ("illustrative_synthetic" if is_action else "illustrative")
    assert not hasattr(result, "planning_world_count")
    assert result.public_charging_status == "modelled"
    assert result.study_slot_count == 336
    assert str(result.run_completed_at_utc.tz) == "UTC"
    # The study starts at London 12:00 on the start date (decision 0004 item 52).
    horizon_start = (
        (pd.Timestamp(result.study_start_local_date) + pd.Timedelta(hours=12))
        .tz_localize("Europe/London")
        .tz_convert("UTC")
    )
    assert result.horizon_start_utc == horizon_start
    assert result.horizon_end_utc == horizon_start + pd.Timedelta(hours=168)
    assert tuple(result.settings_snapshot) == SETTINGS_KEYS
    assert result.settings_snapshot["model"] == result.model
    assert 0 <= result.representative_world_id < result.world_count
    for name in _M5_FIELDS:
        action_only = name.startswith(("difference_", "smart_charging_")) or name in (
            "action_summary",
            "cost_effect_summary",
            "price_band_shift",
            "forecast_price_bands",
            "weekly_peak_summary",
        )
        expected_present = is_action or not action_only
        assert (getattr(result, name) is not None) == expected_present, f"{name} (M5)"

    slots = result.study_slots
    _check_dtypes(slots, STUDY_SLOT_COLUMNS, "study_slots")
    offsets = {ts.utcoffset() for ts in slots["interval_start_london"]}
    assert result.has_clock_change == (len(offsets) > 1)
    _check_dtypes(result.units, UNIT_COLUMNS, "units", exact=False)

    world = result.fleet_world_intervals
    _check_dtypes(world, FLEET_WORLD_COLUMNS, "fleet_world_intervals", exact=False)
    assert list(dict.fromkeys(world["path_id"])) == paths
    assert len(world) == result.world_count * len(paths) * 336
    _check_unique(world, ["world_id", "path_id", "slot_index"], "fleet_world_intervals")
    _check_slot_keys(world, slots, "fleet_world_intervals")
    _validate_world_units(world)
    assert world["battery_soc_percent"].between(0.0, 100.0).all()

    bands = result.fleet_interval_bands
    _check_dtypes(bands, FLEET_BAND_COLUMNS, "fleet_interval_bands")
    _assert_close(bands, fleet_bands(world), ["mean", "p10", "p50", "p90"], "fleet_interval_bands")
    weekly = result.weekly_bands
    _check_dtypes(weekly, WEEKLY_BAND_COLUMNS, "weekly_bands")
    _assert_close(
        weekly, weekly_bands_from_world(world), ["mean", "p10", "p50", "p90", "max"], "weekly"
    )
    cohort_bands = result.cohort_interval_bands
    _check_dtypes(cohort_bands, COHORT_BAND_COLUMNS, "cohort_interval_bands")
    for metric in ADDITIVE_METRICS:
        cohort_sum = (
            cohort_bands.loc[cohort_bands["metric"].eq(metric)]
            .groupby(["path_id", "slot_index"])["mean"]
            .sum()
        )
        fleet = bands.loc[bands["metric"].eq(metric)].set_index(["path_id", "slot_index"])["mean"]
        assert np.allclose(cohort_sum.loc[fleet.index], fleet, atol=1e-6), metric

    if not is_action:
        for name in _ACTION_FIELDS:
            assert getattr(result, name) is None, f"{name} is None on a no-action result"
        assert result.not_recovered_world_count is None
        assert result.action_status == "not_applicable"
        return
    _check_dtypes(result.smart_charging_world, SMART_CHARGING_WORLD_COLUMNS, "smart world")
    _check_dtypes(result.smart_charging_summary, SMART_CHARGING_SUMMARY_COLUMNS, "smart summary")
    _check_dtypes(result.price_band_shift, PRICE_BAND_SHIFT_COLUMNS, "price_band_shift")
    _check_dtypes(result.forecast_prices, FORECAST_PRICE_COLUMNS, "forecast_prices")
    _check_slot_keys(result.forecast_prices, slots, "forecast_prices")
    _check_dtypes(result.evaluation_prices, EVALUATION_PRICE_COLUMNS, "evaluation_prices")
    _check_dtypes(result.cost_effect, COST_EFFECT_COLUMNS, "cost_effect")
    # Decision 0004 item 45: the count is of materially not-recovered weeks.
    assert result.not_recovered_world_count == int(
        result.cost_effect["not_recovered_material"].sum()
    )
    assert result.action_status == "selected"


def test_action_result_fills_every_m4_field_to_contract() -> None:
    result = _run("action")

    _check_owned_fields(result)
    assert result.action_status == "selected"
    assert result.replay_state.smart_charging is not None
    assert result.action_summary.departure_margin_hours == 1.0


def test_no_action_result_has_only_the_normal_path() -> None:
    result = _run("no_action")

    _check_owned_fields(result)
    assert result.replay_state.public_top_up.efficiency_fraction == 0.9
    assert result.replay_state.smart_charging is None


def test_no_action_tops_up_with_the_given_public_assumptions() -> None:
    # Lead decision: the no-action model tops up with the same public
    # assumptions as the action model.  They take no random draws, so
    # editing them never shifts the sampled channels (common random numbers).
    settings = _settings()
    fixture, arguments = _arguments(settings, "no_action")
    base = run_forecast(settings, fixture, **arguments, model="no_action")
    changed_public = {
        **arguments["public_charge_assumptions"],
        "efficiency_fraction": 0.8,
    }
    changed = run_forecast(
        settings,
        fixture,
        **{**arguments, "public_charge_assumptions": changed_public},
        model="no_action",
    )
    for name, values in base.replay_state.evaluation.inputs.items():
        # Series.equals treats NaT in the same place as equal.
        other = changed.replay_state.evaluation.inputs[name]
        assert pd.Series(other.ravel()).equals(pd.Series(values.ravel())), name
    np.testing.assert_array_equal(
        changed.replay_state.evaluation.daily_temperature_c,
        base.replay_state.evaluation.daily_temperature_c,
    )
    public = base.fleet_world_intervals["public_import_kwh"].to_numpy()
    assert public.sum() > 0.0
    # Top-ups add the same battery energy; grid import is that energy over
    # the given efficiency (0.9 vs 0.8), not a fallback value.
    np.testing.assert_allclose(
        changed.fleet_world_intervals["public_import_kwh"].to_numpy() * 0.8, public * 0.9
    )


def test_reversed_forecast_still_plans_the_cheapest_half_hours() -> None:
    # With net demand (and so price) highest at midnight and lowest at noon,
    # the cheapest half-hours are in the day, so smart charging stays close
    # to charging on arrival but is still never dearer at forecast prices
    # than the normal path.
    noon_trough = (20.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    reversed_prices = {
        **_PRICES,
        **{
            f"demand_shape_gw.{season}_{day}": noon_trough
            for season in ("winter", "summer")
            for day in ("weekday", "weekend")
        },
    }
    result = _run("action", prices=reversed_prices)

    _check_owned_fields(result)
    world = result.fleet_world_intervals.merge(
        result.forecast_prices[["world_id", "slot_index", "wholesale_forecast_gbp_per_mwh"]],
        on=["world_id", "slot_index"],
    )
    cost = {
        path: float(
            (rows["home_import_kwh"] * rows["wholesale_forecast_gbp_per_mwh"]).sum()
            / rows["home_import_kwh"].sum()
        )
        for path, rows in world.groupby("path_id")
    }
    assert cost["selected"] <= cost["normal"] + 1e-9


@pytest.mark.parametrize("model", ["action", "no_action"])
@pytest.mark.parametrize("start", [date(2026, 10, 22), date(2026, 3, 26)])
def test_clock_change_week_is_labelled_in_london_time(model: str, start: date) -> None:
    result = _run(model, start_local_date=start, vehicle_count=20, evaluation_world_count=2)

    _check_owned_fields(result)
    assert result.has_clock_change
    # Session nights (noon to noon): the change night has 46 or 50 slots and
    # the fixed 336 UTC slots give the last night the other hour back.
    counts = result.study_slots.groupby("local_date").size().tolist()
    assert sorted(counts) == sorted([48] * 5 + [50, 46])


@pytest.mark.parametrize("model", ["action", "no_action"])
def test_run_forecast_renames_the_kernel_frames_without_changing_values(model: str) -> None:
    # The same seed through simulate_forecast (kernel names) and run_forecast
    # gives the same physics; only names and shapes change (contract 3.4-3.6).
    settings = _settings(vehicle_count=30, evaluation_world_count=3)
    fixture, arguments = _arguments(settings, model)
    result = run_forecast(settings, fixture, **arguments, model=model)
    legacy = simulate_forecast(settings, fixture, **arguments, model=model)

    paths = ["normal", "selected"] if model == "action" else ["normal"]
    old = legacy.fleet_world_intervals
    old = old.loc[old["path_id"].isin(paths)].reset_index(drop=True)
    new = result.fleet_world_intervals
    for new_name, old_name in {
        "home_import_kwh": "home_grid_import_kwh",
        "public_import_kwh": "public_grid_import_kwh",
        "total_import_kw": "realised_grid_kw",
        "battery_soc_percent": "closing_soc_percent",
    }.items():
        np.testing.assert_array_equal(new[new_name], old[old_name])
    np.testing.assert_array_equal(new["driving_share"], old["driving_fraction"] / old["unit_count"])
    old_bands = legacy.fleet_interval_bands
    old_p50 = old_bands.loc[old_bands["path_id"].isin(paths), "home_grid_import_kwh_p50"]
    new_p50 = result.fleet_interval_bands.query("metric == 'home_import_kwh'")["p50"]
    np.testing.assert_array_equal(new_p50, old_p50)
    np.testing.assert_array_equal(
        result.units["preferred_target_soc_percent"],
        100.0 * legacy.units["preferred_target_soc_fraction"],
    )
    if model == "action":
        # build_summaries adds the per-EV sessions count (decision 0004 item
        # 45); every other column is the raw frame unchanged.
        pdt.assert_frame_equal(
            result.cost_effect.drop(columns="sessions_affected_count"), legacy.cost_effect
        )


def test_unknown_model_is_rejected() -> None:
    settings = _settings()
    fixture, arguments = _arguments(settings, "no_action")
    with pytest.raises(ValueError, match="model must be"):
        run_forecast(settings, fixture, **arguments, model="compact")


def test_action_model_requires_its_assumptions() -> None:
    settings = _settings()
    fixture, arguments = _arguments(settings, "no_action")
    with pytest.raises(ValueError, match="departure_margin_hours"):
        run_forecast(settings, fixture, **arguments, model="action")
    with pytest.raises(TypeError, match="price_assumptions must be a mapping"):
        run_forecast(settings, fixture, **arguments, departure_margin_hours=1.0, model="action")


@pytest.mark.parametrize("model", ["action", "no_action"])
def test_both_models_require_public_charge_assumptions(model: str) -> None:
    # No fallback efficiency: both models top up with the given assumptions.
    settings = _settings()
    fixture, arguments = _arguments(settings, model)
    arguments["public_charge_assumptions"] = None
    with pytest.raises(TypeError, match="public_charge_assumptions must be a mapping"):
        run_forecast(settings, fixture, **arguments, model=model)
