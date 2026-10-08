"""The run fields ``world_nights`` is built from (trading contract v1 §10.1f).

The availability lane builds the ``world_nights`` frame; this lane supplies
its inputs on ``SimulatedForecast`` and declares the §10.0 result fields.
These tests pin the shapes, the night-to-date indexing and the rule the
frame's validator will recompute (the skip share from the stored factors
and switches).  All values are synthetic.
"""

from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np
from scipy.special import expit, logit

from axle_studio.model import assumptions
from axle_studio.model.forecast import ForecastResult, run_forecast, simulate_forecast
from axle_studio.model.settings import RunSettings

_SETTINGS = RunSettings(
    start_local_date=date(2026, 1, 12),
    warmup_days=1,
    study_days=3,
    vehicle_count=40,
    seed=17,
    evaluation_world_count=4,
    opening_soc_fraction=0.55,
    reserve_soc_fraction=0.2,
    home_charge_efficiency=0.92,
    weather_efficiency_sensitivity_fraction_per_c=0.01,
)

_SECTION_10_FIELDS = (
    "blackout_windows",
    "world_nights",
    "availability_world_slot",
    "availability_bands",
    "availability_manufacturer_world",
    "firm_share_by_fleet_size",
    "availability_backtest",
    "availability_reliability",
    "availability_backtest_summary",
    "product_sheet",
    "availability_value_summary",
    "settlement_file",
    "settlement_file_name",
    "charge_completion_summary",
    "firmness_by_manufacturer",
    "manufacturer_summary",
)


def _inputs(**factor_edits):
    inputs = assumptions.forecast_inputs(None, warmup_days=1, study_days=3)
    inputs["shared_factor_assumptions"] = {**inputs["shared_factor_assumptions"], **factor_edits}
    return inputs


def test_factor_arrays_have_world_and_date_shapes() -> None:
    run = simulate_forecast(_SETTINGS, assumptions.cohort_fixture(), **_inputs())
    factors = run.plug_in_factors
    worlds, days = _SETTINGS.evaluation_world_count, _SETTINGS.sampled_day_count

    assert factors["week"].shape == (worlds,)
    for key in ("day", "skip_share"):
        assert factors[key].shape == (worlds, days)
        assert factors[key].dtype == np.float64
    assert factors["holiday"].shape == (days,) and factors["holiday"].dtype == bool
    assert factors["skip_logit_shift"].shape == (days,)
    assert run.evaluation.daily_temperature_c.shape == (worlds, days)
    assert run.manufacturer_outage.shape == (worlds, _SETTINGS.study_days, 4)
    assert run.manufacturer_outage.dtype == bool
    assert run.session_non_response_probability.shape == (
        worlds,
        _SETTINGS.study_days,
        _SETTINGS.vehicle_count,
    )


def test_skip_share_recomputes_from_the_stored_factors_and_switches() -> None:
    # The world_nights validator rule (§10.9): q = expit(logit(median) +
    # σ_week y + σ_day z + h_d), from the stored factors and the run's values.
    run = simulate_forecast(
        _SETTINGS,
        assumptions.cohort_fixture(),
        **_inputs(**{"behaviour.holiday_half_term_week": 1}),
    )
    factors, values = run.plug_in_factors, run.shared_factor_assumptions
    expected = expit(
        logit(values["behaviour.plug_in_skip_median"])
        + values["behaviour.plug_in_week_factor_sd"] * factors["week"][:, None]
        + values["behaviour.plug_in_day_factor_sd"] * factors["day"]
        + factors["skip_logit_shift"][None, :]
    )
    np.testing.assert_allclose(factors["skip_share"], expected, rtol=0.0, atol=1e-15)
    # Night n's evening is sampled date warmup_days + n (§10.1a).
    evenings = slice(_SETTINGS.warmup_days, _SETTINGS.warmup_days + _SETTINGS.study_days)
    assert factors["holiday"][evenings].all()
    assert not np.delete(factors["holiday"], np.r_[evenings]).any()
    assert values == {
        **assumptions.shared_factor_inputs(),
        "behaviour.holiday_half_term_week": 1.0,
    }


def test_section_10_result_fields_are_declared_and_filled_on_action_runs() -> None:
    names = {field.name for field in dataclasses.fields(ForecastResult)}
    assert set(_SECTION_10_FIELDS) <= names
    for model in ("action", "no_action"):
        result = run_forecast(_SETTINGS, assumptions.cohort_fixture(), model=model, **_inputs())
        # Every result carries the validated blackout table (§10.0); the
        # other frames exist on action results only.
        assert list(result.blackout_windows["start_local_time"]) == []
        for name in _SECTION_10_FIELDS[1:]:
            assert (getattr(result, name) is None) == (model == "no_action"), name
        assert result.units["manufacturer_id"].isin(assumptions.MANUFACTURER_IDS).all()
