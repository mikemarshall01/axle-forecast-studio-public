"""Common random numbers across a full explicit action run (plan M6).

Compare's "matched futures" hold only if a parameter edit changes its own
channel and nothing else: every sampler takes a fixed number of draws from the
run's single generator whatever the parameter values (decision 0004 item 14).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from axle_studio.model import assumptions
from axle_studio.model.forecast import simulate_forecast
from axle_studio.model.settings import RunSettings


def _default_inputs() -> dict:
    # Fresh run inputs from model/assumptions.py (the removed JSON configs'
    # values) on every call, so a test that edits them cannot leak into another.
    return assumptions.forecast_inputs(None, warmup_days=1, study_days=7)


def _weather(*, warmup_days: int, study_days: int):
    inputs = assumptions.forecast_inputs(None, warmup_days=warmup_days, study_days=study_days)
    return inputs["base_temperature_c_by_day"], inputs["weather_sd_c"]


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

_CONNECTION_KEYS = ("connection_session_accepted", "connection_start_utc", "connection_end_utc")


def _run(edit: str | None, model: str = "action"):
    fixture = assumptions.cohort_fixture()
    temperature, weather_sd = _weather(
        warmup_days=_SETTINGS.warmup_days, study_days=_SETTINGS.study_days
    )
    personal_cvs = _default_inputs()["personal_mileage_cv_by_cohort"]
    # Default threshold re-baselined from 20 % to 10 % by decision 0004 item 37.
    public = {
        "efficiency_fraction": 0.9,
        "top_up_threshold_soc_fraction": 0.1,
        "top_up_target_soc_fraction": 0.8,
    }
    prices = dict(_default_inputs()["price_assumptions"])
    if edit == "public_efficiency":
        public["efficiency_fraction"] = 0.8
    elif edit == "price_sd":
        prices["intraday_hourly_sd_gbp_per_mwh"] = 0.0
    elif edit == "personal_cv":
        personal_cvs = {cohort_id: 0.0 for cohort_id in personal_cvs}
    elif edit == "weather_sd":
        weather_sd = 0.0
    trip_behaviour = _default_inputs()["trip_behaviour"]
    if edit == "departure_scale":
        trip_behaviour["by_cohort"]["average_uk"]["departure_scale_minutes"] = 0.0
    elif edit == "plug_in_scale":
        trip_behaviour["by_cohort"]["average_uk"]["plug_in_scale_minutes"] = 0.0
    elif edit == "clock_df":
        trip_behaviour["clock_t_df"] = 30.0
    # 2 h is the default margin (decision 0004 item 51); the edit moves it to 3 h.
    margin = 3.0 if edit == "departure_margin" else 2.0
    factors = dict(_default_inputs()["shared_factor_assumptions"])
    factors.update(_FACTOR_EDITS.get(edit, {}))
    return simulate_forecast(
        _SETTINGS,
        fixture,
        daily_miles_cv_by_cohort=_default_inputs()["daily_miles_cv_by_cohort"],
        personal_mileage_cv_by_cohort=personal_cvs,
        trip_behaviour=trip_behaviour,
        base_temperature_c_by_day=temperature,
        weather_sd_c=weather_sd,
        public_charge_assumptions=public,
        price_assumptions=prices,
        departure_margin_hours=margin,
        model=model,
        shared_factor_assumptions=factors,
    )


# Edits of every §10.1 record (trading contract v1 §10.7 J1a: a Compare
# pairing test).  _SETTINGS starts on Monday 12 January, so the bank-holiday
# switch has a Monday study evening to act on.
_FACTOR_EDITS = {
    "skip_median": {"behaviour.plug_in_skip_median": 0.3},
    "skip_day_sd": {"behaviour.plug_in_day_factor_sd": 1.5},
    "skip_week_sd": {"behaviour.plug_in_week_factor_sd": 1.0},
    "bank_holiday": {"behaviour.holiday_bank_holiday_monday": 1},
    "half_term": {"behaviour.holiday_half_term_week": 1},
    "maker_shares": {
        "manufacturers.share.m1": 0.4,
        "manufacturers.share.m2": 0.3,
        "manufacturers.share.m3": 0.2,
        "manufacturers.share.m4": 0.1,
    },
    "maker_response": {"manufacturers.response_rate.m2": 0.5},
    "maker_outage": {f"manufacturers.outage_probability_per_night.m{k}": 0.6 for k in range(1, 5)},
}


def _channels(result) -> dict[str, object]:
    channels: dict[str, object] = {}
    inputs = result.evaluation.inputs
    channels["evaluation_drives"] = inputs["drives_today"]
    channels["evaluation_miles"] = inputs["outbound_miles"]
    channels["evaluation_departure"] = inputs["trip_departure_utc"]
    for key in _CONNECTION_KEYS:
        channels[f"evaluation_{key}"] = inputs[key]
    channels["evaluation_temperature"] = result.evaluation.daily_temperature_c
    channels["population_cohorts"] = result.units["cohort_id"].to_numpy()
    channels["population_miles"] = result.units["daily_miles_mean"].to_numpy()
    channels["population_zones"] = result.units["zone_id"].to_numpy()
    channels["population_makers"] = result.units["manufacturer_id"].to_numpy()
    channels["plug_in_factors"] = np.concatenate(
        [result.plug_in_factors["week"], result.plug_in_factors["day"].ravel()]
    )
    channels["skip_share"] = result.plug_in_factors["skip_share"]
    if result.evaluation_prices is not None:
        channels["prices"] = result.evaluation_prices[
            "evaluation_context_price_gbp_per_mwh"
        ].to_numpy()
        channels["non_response_uniforms"] = result.smart_charging.non_response_uniform
        channels["maker_outages"] = result.manufacturer_outage
        channels["session_non_response"] = result.session_non_response_probability
    return channels


def _equal(left: object, right: object) -> bool:
    # Series.equals treats NaT in matching positions as equal; elementwise
    # comparison would not, because NaT never equals itself.
    return pd.Series(np.asarray(left).ravel()).equals(pd.Series(np.asarray(right).ravel()))


# Channels each edit is allowed to change; every other channel must be identical.
_OWN_CHANNELS = {
    "price_sd": {"prices"},
    # Temperatures drive the heating term of net demand, so a weather edit
    # also moves prices, through values, not draws (decision 0004 item 53).
    "weather_sd": {"evaluation_temperature", "prices"},
    # Personal CV moves each EV's expected daily miles, so the trip distances
    # scale with it; the underlying distance draws are unchanged.
    "personal_cv": {"population_miles", "evaluation_miles"},
    # Decision 0004 item 51: a departure is both the unplug and the trip
    # start, so its scale moves trip departures and session ends together;
    # the plug-in is an independent draw, so its scale moves session starts
    # only; the df reshapes both clocks.  None of them moves another draw.
    "departure_scale": {"evaluation_departure", "evaluation_connection_end_utc"},
    "plug_in_scale": {"evaluation_connection_start_utc"},
    "clock_df": {
        "evaluation_departure",
        "evaluation_connection_end_utc",
        "evaluation_connection_start_utc",
    },
    # §10.1b-c: the skip share only changes which would-be sessions plug in
    # (their start and end are NaT when skipped); the factors themselves
    # and every other draw stay put.
    **{
        edit: {
            "skip_share",
            *(f"evaluation_{key}" for key in _CONNECTION_KEYS),
        }
        for edit in ("skip_median", "skip_day_sd", "skip_week_sd", "bank_holiday", "half_term")
    },
    # §10.1d-e: makers change who shares a response rate and an outage, so
    # the session non-response array; the outage draws are per maker.
    "maker_shares": {"population_makers", "session_non_response"},
    "maker_response": {"session_non_response"},
    "maker_outage": {"maker_outages", "session_non_response"},
}


@pytest.mark.parametrize("edit", sorted(_OWN_CHANNELS))
def test_one_parameter_edit_leaves_every_other_channel_identical(edit: str) -> None:
    base = _channels(_run(None))
    edited = _channels(_run(edit))

    changed = {name for name in base if not _equal(base[name], edited[name])}
    assert changed <= _OWN_CHANNELS[edit], f"edit {edit} also changed {changed}"
    assert changed, f"edit {edit} should change its own channel"


def test_public_top_up_edit_changes_no_sampled_channel() -> None:
    # Public top-ups take no draws (decision 0004 item 32), so editing them
    # leaves every sampled channel, prices included, exactly as it was.
    base = _channels(_run(None))
    edited = _channels(_run("public_efficiency"))

    assert not {name for name in base if not _equal(base[name], edited[name])}


def test_departure_margin_edit_changes_no_sampled_channel() -> None:
    # Smart charging is deterministic given the sampled inputs (decision 0004
    # item 38): its margin moves the selected path only, never a draw.
    base = _run(None)
    edited = _run("departure_margin")
    base_channels, edited_channels = _channels(base), _channels(edited)

    assert not {n for n in base_channels if not _equal(base_channels[n], edited_channels[n])}
    normal = base.fleet_world_intervals["path_id"].eq("normal")
    pd.testing.assert_frame_equal(
        base.fleet_world_intervals.loc[normal], edited.fleet_world_intervals.loc[normal]
    )


def test_both_models_draw_the_same_worlds_for_one_seed() -> None:
    # The action model draws its price noise after the worlds, so the same
    # seed gives both models the same futures and the same normal path
    # (Compare pairs them as matched futures).
    action = _run(None)
    no_action = _run(None, model="no_action")
    action_channels, no_action_channels = _channels(action), _channels(no_action)

    for name, values in no_action_channels.items():
        assert _equal(values, action_channels[name]), name
    normal = action.fleet_world_intervals["path_id"].eq("normal")
    pd.testing.assert_frame_equal(
        action.fleet_world_intervals.loc[normal].reset_index(drop=True),
        no_action.fleet_world_intervals.loc[
            no_action.fleet_world_intervals["path_id"].eq("normal")
        ].reset_index(drop=True),
    )
