from __future__ import annotations

import copy
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from axle_studio.model import assumptions
from axle_studio.model.forecast import simulate_forecast
from axle_studio.model.settings import CohortFixture, RunSettings


def _default_inputs() -> dict:
    # Fresh run inputs from model/assumptions.py (the removed JSON configs'
    # values) on every call, so a test that edits them cannot leak into another.
    return assumptions.forecast_inputs(None, warmup_days=1, study_days=7)


def _weather(*, warmup_days: int, study_days: int):
    inputs = assumptions.forecast_inputs(None, warmup_days=warmup_days, study_days=study_days)
    return inputs["base_temperature_c_by_day"], inputs["weather_sd_c"]


_INPUT_KEYS = (
    "drives_today",
    "outbound_miles",
    "return_miles",
    "trip_departure_utc",
    "destination_dwell_seconds",
    "drive_speed_mph",
    "desired_pre_drive_soc_fraction",
    "connection_session_accepted",
    "connection_start_utc",
    "connection_end_utc",
)


@pytest.fixture(scope="module")
def fixture() -> CohortFixture:
    return assumptions.cohort_fixture()


def _settings(**changes: object) -> RunSettings:
    settings = RunSettings(
        start_local_date=date(2026, 1, 12),
        warmup_days=1,
        study_days=7,
        vehicle_count=6,
        seed=73,
        evaluation_world_count=3,
        opening_soc_fraction=0.55,
        reserve_soc_fraction=0.2,
        home_charge_efficiency=0.92,
        weather_efficiency_sensitivity_fraction_per_c=0.01,
    )
    return replace(settings, **changes)


def _inputs(fixture: CohortFixture, settings: RunSettings):
    daily_cvs = _default_inputs()["daily_miles_cv_by_cohort"]
    personal_cvs = _default_inputs()["personal_mileage_cv_by_cohort"]
    behaviour = _default_inputs()["trip_behaviour"]
    temperature, weather_sd = _weather(
        warmup_days=settings.warmup_days,
        study_days=settings.study_days,
    )
    return daily_cvs, personal_cvs, behaviour, temperature, weather_sd


def _no_action(settings: RunSettings, fixture: CohortFixture, **inputs: object):
    # The raw no-action simulation (kernel column names, sampled worlds kept),
    # with the default public top-up assumptions both models use.
    public = _default_inputs()["public_charge_assumptions"]
    return simulate_forecast(
        settings, fixture, **inputs, public_charge_assumptions=public, model="no_action"
    )


def _run(fixture: CohortFixture, settings: RunSettings | None = None):
    settings = settings or _settings()
    daily_cvs, personal_cvs, behaviour, temperature, weather_sd = _inputs(fixture, settings)
    return _no_action(
        settings,
        fixture,
        daily_miles_cv_by_cohort=daily_cvs,
        personal_mileage_cv_by_cohort=personal_cvs,
        trip_behaviour=behaviour,
        base_temperature_c_by_day=temperature,
        weather_sd_c=weather_sd,
    )


def _assert_array_equal(left: np.ndarray, right: np.ndarray) -> None:
    if left.dtype == object:
        assert all(
            (pd.isna(a) and pd.isna(b)) or a == b
            for a, b in zip(left.flat, right.flat, strict=True)
        )
    else:
        np.testing.assert_array_equal(left, right)


def test_fixed_seed_repeats_population_inputs_weather_and_bands(fixture: CohortFixture) -> None:
    first = _run(fixture)
    repeated = _run(fixture)

    pdt.assert_frame_equal(first.units, repeated.units)
    assert tuple(first.evaluation.inputs) == _INPUT_KEYS
    for key in _INPUT_KEYS:
        _assert_array_equal(first.evaluation.inputs[key], repeated.evaluation.inputs[key])
    np.testing.assert_array_equal(
        first.evaluation.daily_temperature_c,
        repeated.evaluation.daily_temperature_c,
    )
    np.testing.assert_array_equal(
        first.evaluation.effective_efficiency_miles_per_battery_kwh,
        repeated.evaluation.effective_efficiency_miles_per_battery_kwh,
    )
    pdt.assert_frame_equal(first.fleet_interval_bands, repeated.fleet_interval_bands)
    pdt.assert_frame_equal(
        first.cohort_interval_distribution,
        repeated.cohort_interval_distribution,
    )


def test_population_traits_mileage_contract_and_world_shapes(fixture: CohortFixture) -> None:
    result = _run(fixture)
    daily_cvs, personal_cvs, _, _, _ = _inputs(fixture, result.settings)
    source_mean = {cohort.cohort_id: cohort.daily_miles_mean for cohort in fixture.cohorts}

    assert result.units["unit_id"].is_unique
    assert result.units["population_id"].nunique() == 1
    assert result.units["cohort_source_name"].notna().all()
    assert any(personal_cvs[cohort_id] > 0 for cohort_id in personal_cvs)
    assert any(
        not np.isclose(row.daily_miles_mean, source_mean[row.cohort_id])
        for row in result.units.itertuples()
    )
    np.testing.assert_allclose(
        result.units["daily_miles_sd"],
        result.units["daily_miles_mean"] * result.units["cohort_id"].map(daily_cvs),
    )
    # Sampled dates: warm-up, study nights and the morning after the last
    # night (RunSettings.sampled_day_count, decision 0004 item 52).
    days = result.settings.sampled_day_count
    expected_shape = (3, days, 6)
    assert days == result.settings.warmup_days + result.settings.study_days + 1
    assert all(value.shape == expected_shape for value in result.evaluation.inputs.values())
    assert result.evaluation.daily_temperature_c.shape == (3, days)
    assert result.evaluation.effective_efficiency_miles_per_battery_kwh.shape == expected_shape


def test_daily_mileage_cv_changes_sampled_miles_and_physical_distribution(
    fixture: CohortFixture,
) -> None:
    settings = _settings()
    daily_cvs, personal_cvs, behaviour, temperature, weather_sd = _inputs(fixture, settings)
    changed_cvs = {cohort_id: value * 2.0 for cohort_id, value in daily_cvs.items()}

    baseline = _no_action(
        settings,
        fixture,
        daily_miles_cv_by_cohort=daily_cvs,
        personal_mileage_cv_by_cohort=personal_cvs,
        trip_behaviour=behaviour,
        base_temperature_c_by_day=temperature,
        weather_sd_c=weather_sd,
    )
    changed = _no_action(
        settings,
        fixture,
        daily_miles_cv_by_cohort=changed_cvs,
        personal_mileage_cv_by_cohort=personal_cvs,
        trip_behaviour=behaviour,
        base_temperature_c_by_day=temperature,
        weather_sd_c=weather_sd,
    )

    pdt.assert_series_equal(baseline.units["cohort_id"], changed.units["cohort_id"])
    pdt.assert_series_equal(baseline.units["daily_miles_mean"], changed.units["daily_miles_mean"])
    assert (
        baseline.units.groupby("cohort_id").size().to_dict()
        == changed.units.groupby("cohort_id").size().to_dict()
    )
    np.testing.assert_array_equal(
        baseline.evaluation.inputs["drives_today"], changed.evaluation.inputs["drives_today"]
    )
    baseline_miles = (
        baseline.evaluation.inputs["outbound_miles"] + baseline.evaluation.inputs["return_miles"]
    )
    changed_miles = (
        changed.evaluation.inputs["outbound_miles"] + changed.evaluation.inputs["return_miles"]
    )
    assert not np.array_equal(
        baseline_miles[baseline_miles > 0.0], changed_miles[changed_miles > 0.0]
    )
    assert not baseline.fleet_interval_bands.equals(changed.fleet_interval_bands)


def test_zero_daily_mileage_cv_gives_deterministic_conditional_distance(
    fixture: CohortFixture,
) -> None:
    settings = _settings()
    daily_cvs, personal_cvs, behaviour, temperature, weather_sd = _inputs(fixture, settings)
    zero_cvs = dict.fromkeys(daily_cvs, 0.0)
    result = _no_action(
        settings,
        fixture,
        daily_miles_cv_by_cohort=zero_cvs,
        personal_mileage_cv_by_cohort=personal_cvs,
        trip_behaviour=behaviour,
        base_temperature_c_by_day=temperature,
        weather_sd_c=weather_sd,
    )

    assert result.units["daily_miles_sd"].eq(0.0).all()
    total_miles = (
        result.evaluation.inputs["outbound_miles"] + result.evaluation.inputs["return_miles"]
    )
    drives = result.evaluation.inputs["drives_today"]
    first_date = settings.start_local_date - timedelta(days=settings.warmup_days)
    for unit_index, unit in enumerate(result.units.itertuples()):
        cohort_behaviour = behaviour["by_cohort"][unit.cohort_id]
        for day_index in range(settings.sampled_day_count):
            probability_field = (
                "weekday_drive_probability"
                if (first_date + timedelta(days=day_index)).weekday() < 5
                else "weekend_drive_probability"
            )
            driven = drives[:, day_index, unit_index]
            np.testing.assert_allclose(
                total_miles[:, day_index, unit_index][driven],
                unit.daily_miles_mean / cohort_behaviour[probability_field],
            )


def test_world_cohort_fleet_reconcile_before_quantiles_and_paths_match(
    fixture: CohortFixture,
) -> None:
    result = _run(fixture)
    keys = ["world_id", "path_id", "interval_start_utc", "interval_end_utc"]
    additive = [
        "realised_grid_kw",
        "home_grid_import_kwh",
        "closing_battery_kwh",
        "physical_capacity_kwh",
    ]
    cohort_totals = result.cohort_world_intervals.groupby(keys, sort=False)[additive].sum()
    fleet = result.fleet_world_intervals.set_index(keys)[additive]
    pdt.assert_frame_equal(cohort_totals.sort_index(), fleet.sort_index())

    normal = result.fleet_world_intervals.loc[
        result.fleet_world_intervals["path_id"].eq("normal")
    ].drop(columns="path_id")
    selected = result.fleet_world_intervals.loc[
        result.fleet_world_intervals["path_id"].eq("selected")
    ].drop(columns="path_id")
    pdt.assert_frame_equal(normal.reset_index(drop=True), selected.reset_index(drop=True))
    assert result.fleet_interval_bands["world_count"].eq(3).all()


def test_safe_seven_day_run_has_336_utc_half_hours(fixture: CohortFixture) -> None:
    result = _run(fixture)
    for frame in (
        result.fleet_world_intervals,
        result.cohort_world_intervals,
        result.fleet_interval_bands,
        result.cohort_interval_distribution,
    ):
        assert str(frame["interval_start_utc"].dt.tz) == "UTC"
    normal_world = result.fleet_world_intervals.query("world_id == 0 and path_id == 'normal'")
    assert len(normal_world) == 336
    assert normal_world["interval_start_utc"].nunique() == 336
    assert (
        (normal_world["interval_end_utc"] - normal_world["interval_start_utc"])
        .eq(pd.Timedelta(minutes=30))
        .all()
    )


def test_weekday_weekend_behaviour_includes_warmup_calendar_day(
    fixture: CohortFixture,
) -> None:
    settings = _settings(study_days=1, evaluation_world_count=1)
    daily_cvs, personal_cvs, behaviour, temperature, weather_sd = _inputs(fixture, settings)
    for row in behaviour["by_cohort"].values():
        row["weekday_drive_probability"] = 1.0
        row["weekend_drive_probability"] = 1.0
        row["departure_scale_minutes"] = 0.0
    result = _no_action(
        settings,
        fixture,
        daily_miles_cv_by_cohort=daily_cvs,
        personal_mileage_cv_by_cohort=personal_cvs,
        trip_behaviour=behaviour,
        base_temperature_c_by_day=temperature,
        weather_sd_c=weather_sd,
    )

    # The warm-up day is Sunday 11 January 2026 (GMT, so UTC hours are London
    # hours): trips leave at each cohort's weekend home departure, then its
    # weekday one on Monday (plug-out is departure, decision 0004 item 51).
    departures = result.evaluation.inputs["trip_departure_utc"]
    units = result.units
    assert [pd.Timestamp(value).hour for value in departures[0, 0]] == units[
        "runtime_weekend_departure_local_hour"
    ].tolist()
    assert [pd.Timestamp(value).hour for value in departures[0, 1]] == units[
        "runtime_departure_local_hour"
    ].tolist()
    assert (
        units["runtime_weekend_departure_local_hour"]
        .ne(units["runtime_departure_local_hour"])
        .any()
    )


def test_no_action_run_has_no_action_fields_and_does_not_mutate_callers(
    fixture: CohortFixture,
) -> None:
    settings = _settings()
    daily_cvs, personal_cvs, behaviour, temperature, weather_sd = _inputs(fixture, settings)
    before = (
        copy.deepcopy(daily_cvs),
        copy.deepcopy(personal_cvs),
        copy.deepcopy(behaviour),
        temperature.copy(),
        fixture,
    )
    result = _no_action(
        settings,
        fixture,
        daily_miles_cv_by_cohort=daily_cvs,
        personal_mileage_cv_by_cohort=personal_cvs,
        trip_behaviour=behaviour,
        base_temperature_c_by_day=temperature,
        weather_sd_c=weather_sd,
    )

    assert result.model == "no_action"
    assert result.smart_charging is None and result.cost_effect is None
    assert daily_cvs == before[0]
    assert personal_cvs == before[1]
    assert behaviour == before[2]
    np.testing.assert_array_equal(temperature, before[3])
    assert fixture == before[4]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda behaviour: behaviour.pop("by_cohort"), "missing required field: by_cohort"),
        (
            lambda behaviour: behaviour["by_cohort"].pop("average_uk"),
            "missing used cohort: average_uk",
        ),
    ],
)
def test_rejects_missing_behaviour_boundary(fixture: CohortFixture, change, message: str) -> None:
    settings = _settings()
    daily_cvs, personal_cvs, behaviour, temperature, weather_sd = _inputs(fixture, settings)
    change(behaviour)
    with pytest.raises(ValueError, match=message):
        _no_action(
            settings,
            fixture,
            daily_miles_cv_by_cohort=daily_cvs,
            personal_mileage_cv_by_cohort=personal_cvs,
            trip_behaviour=behaviour,
            base_temperature_c_by_day=temperature,
            weather_sd_c=weather_sd,
        )


def test_one_generator_is_created_only_on_explicit_call(
    monkeypatch: pytest.MonkeyPatch, fixture: CohortFixture
) -> None:
    import axle_studio.model.forecast as module

    settings = _settings(study_days=1, evaluation_world_count=1)
    daily_cvs, personal_cvs, behaviour, temperature, weather_sd = _inputs(fixture, settings)
    original = np.random.default_rng
    calls: list[int] = []

    def counted(seed: int):
        calls.append(seed)
        return original(seed)

    monkeypatch.setattr(module.np.random, "default_rng", counted)
    assert calls == []
    _no_action(
        settings,
        fixture,
        daily_miles_cv_by_cohort=daily_cvs,
        personal_mileage_cv_by_cohort=personal_cvs,
        trip_behaviour=behaviour,
        base_temperature_c_by_day=temperature,
        weather_sd_c=weather_sd,
    )
    assert calls == [settings.seed]
