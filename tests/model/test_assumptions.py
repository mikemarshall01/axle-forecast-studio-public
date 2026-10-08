"""Completeness, contract, value and builder tests for the assumptions module.

``model/assumptions.py`` is the single source of every value (decision 0004
item 21).  Before task M3b removed the JSON configs, their loaders' outputs
were checked against these records and a small real run was checked to be
bit-identical; the value tests below pin the values those configs held, so an
accidental edit here is caught.  Contract tests check ``result_assumptions()``
against ``docs/contracts/results-v2.md`` section 8 using the accepted checker
in ``tests/fixtures/result_contract.py``.
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import pytest

from axle_studio.model import action, assumptions, physics, sampling

_TESTS_DIR = Path(__file__).resolve().parents[1]
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from fixtures.result_contract import _validate_assumptions  # noqa: E402

# ---------------------------------------------------------------------------
# Completeness: every entry is fully labelled and matches the contract shape.
# ---------------------------------------------------------------------------


def test_every_assumption_has_unit_evidence_source_meaning_label_group_and_affects() -> None:
    frame = assumptions.assumption_rows()
    assert (
        len(frame) > 100
    )  # six cohorts x eight fields alone gives 48; catch accidental truncation

    text_columns = ("unit", "evidence", "source", "meaning", "label", "group", "affects")
    for column in text_columns:
        is_filled = frame[column].map(lambda value: isinstance(value, str) and bool(value.strip()))
        blank = frame.loc[~is_filled]
        assert blank.empty, f"{column} missing for: {blank['name'].tolist()}"

    assert frame["evidence"].isin({"source", "illustrative", "synthetic"}).all()
    assert frame["group"].isin(assumptions.ASSUMPTION_GROUPS).all()
    assert frame["editable"].map(lambda value: isinstance(value, bool)).all()


def test_assumption_names_are_globally_unique() -> None:
    frame = assumptions.assumption_rows()
    assert frame["name"].is_unique


def test_source_evidence_records_are_never_editable() -> None:
    frame = assumptions.assumption_rows()
    source_records = frame.loc[frame["evidence"] == "source"]
    assert len(source_records) > 0
    assert not source_records["editable"].any()


def test_editable_assumptions_have_bounds_and_value_within_them() -> None:
    frame = assumptions.assumption_rows()
    editable = frame.loc[frame["editable"]]
    assert len(editable) > 0

    for _, row in editable.iterrows():
        if isinstance(row["value"], str):
            # The one text-switch record (trading.commitment_rule, §10.4,
            # §10.8): checked against its allowed set in validation_errors,
            # not a numeric bounds tuple.
            continue
        assert row["bounds"] is not None, f"{row['name']} is editable but has no bounds"
        low, high = row["bounds"]
        # An unset optional term is NaN (supplier contract v1 §8): no bound applies.
        if isinstance(row["value"], float) and math.isnan(row["value"]):
            continue
        assert low <= row["value"] <= high, (
            f"{row['name']} value {row['value']} outside {row['bounds']}"
        )


def test_editable_defaults_keep_int_and_float_types_from_the_records() -> None:
    defaults = assumptions.editable_defaults()

    assert isinstance(defaults["seed"], int) and not isinstance(defaults["seed"], bool)
    assert isinstance(defaults["vehicle_count"], int)
    assert isinstance(defaults["evaluation_world_count"], int)
    assert isinstance(defaults["home_charge_efficiency"], float)


def test_assumption_rows_and_editable_defaults_are_consistent() -> None:
    frame = assumptions.assumption_rows()
    defaults = assumptions.editable_defaults()
    editable_names = set(frame.loc[frame["editable"], "name"])

    assert set(defaults) == editable_names
    for name, value in defaults.items():
        row = frame.loc[frame["name"] == name].iloc[0]
        assert value == row["value"] or (math.isnan(value) and math.isnan(row["value"]))
        # Not a type check here: assumption_rows() is a display DataFrame, whose
        # numeric columns return numpy scalar types regardless of the original
        # Python type. editable_defaults()'s own int/float fidelity is checked
        # directly in test_editable_defaults_keep_int_and_float_types_from_the_records.


# ---------------------------------------------------------------------------
# Contract: result_assumptions() matches docs/contracts/results-v2.md section 8.
# ---------------------------------------------------------------------------


def test_result_assumptions_passes_the_contract_validator() -> None:
    _validate_assumptions(assumptions.result_assumptions())


def test_result_assumptions_excludes_the_simulation_section() -> None:
    result_names = {a.name for a in assumptions.result_assumptions()}
    simulation_names = {a.name for a in assumptions.SIMULATION.values()}

    assert simulation_names, "SIMULATION must be non-empty for this test to mean anything"
    assert result_names.isdisjoint(simulation_names)
    # Action mechanics share the "Simulation" dialog tab but are not run-scale
    # settings, so they stay in the result's assumptions tuple.
    assert {a.name for a in assumptions.ACTION.values()} <= result_names


# ---------------------------------------------------------------------------
# Values: the cohort fixture, run inputs and settings built from the records.
# ---------------------------------------------------------------------------


def test_cohort_shares_sum_to_one() -> None:
    total = sum(record.value for record in assumptions.COHORT_SHARES.values())
    assert total == pytest.approx(100.0)


def test_cohort_fixture_is_the_six_source_rows_with_one_home_charging_power() -> None:
    fixture = assumptions.cohort_fixture()

    assert [c.cohort_id for c in fixture.cohorts] == list(assumptions.COHORTS)
    assert [c.source_row for c in fixture.cohorts] == [6, 7, 8, 9, 10, 11]
    assert [c.population_share_fraction for c in fixture.cohorts] == [
        0.4,
        0.3,
        0.1,
        0.1,
        0.09,
        0.01,
    ]
    assert [c.daily_miles_mean for c in fixture.cohorts] == [
        9435 / 365,
        28105 / 365,
        9435 / 365,
        5700 / 365,
        9435 / 365,
        9435 / 365,
    ]
    assert [c.battery_capacity_kwh for c in fixture.cohorts] == [60.0, 72.5, 60, 60, 60, 60]
    assert [c.plug_probability for c in fixture.cohorts] == [1.0, 1.0, 0.2, 1.0, 1.0, 1.0]
    assert [(c.departure_local_hour, c.arrival_local_hour) for c in fixture.cohorts] == [
        (7, 18),
        (7, 18),
        (7, 18),
        (7, 18),
        (9, 22),
        (14, 15),
    ]
    # Decision 0004 item 42: weekend windows for Average UK and Intelligent
    # Octopus only (CNZ adaptation); the other four are flat.
    assert [
        (c.weekend_departure_local_hour, c.weekend_arrival_local_hour) for c in fixture.cohorts
    ] == [(10, 17), (10, 17), (7, 18), (7, 18), (9, 22), (14, 15)]
    assert {c.efficiency_miles_per_battery_kwh for c in fixture.cohorts} == {3.5}
    assert {c.preferred_target_soc_fraction for c in fixture.cohorts} == {0.8}
    # Decision 0004 item 34: one home charging power, 7 kW from H6:H11.
    assert {c.home_charger_limit_kw for c in fixture.cohorts} == {7.0}
    assert fixture.cohorts[0].source_name == "Average (UK)"


def test_home_charging_power_edit_reaches_every_cohort() -> None:
    fixture = assumptions.cohort_fixture({"home_charging_power_kw": 3.6})
    assert {c.home_charger_limit_kw for c in fixture.cohorts} == {3.6}


# ---------------------------------------------------------------------------
# Cohort mix: the editable illustrative override (decision 0004 items 21, 54).
# ---------------------------------------------------------------------------


def test_cohort_shares_are_editable_illustrative_percent_defaulting_to_the_source() -> None:
    expected_defaults = {
        "average_uk": 40.0,
        "intelligent_octopus": 30.0,
        "infrequent_charging": 10.0,
        "infrequent_driving": 10.0,
        "scheduled_charging": 9.0,
        "always_plugged_in": 1.0,
    }
    for cohort_id, record in assumptions.COHORT_SHARES.items():
        assert record.name == f"{cohort_id}.population_share_percent"
        assert record.value == pytest.approx(expected_defaults[cohort_id])
        assert record.unit == "percent"
        # Item 21 keeps "source" records read-only; this source-traced default
        # is editable, so it is labelled illustrative instead (same choice as
        # HOME_CHARGING['home_charging_power_kw']).
        assert record.evidence == "illustrative"
        assert record.editable is True
        assert record.bounds == (0.0, 100.0)
        assert record.group == "Fleet"
        assert "Defaults to the source share" in record.meaning
        assert "items 21 and 54" in record.source


def test_defaults_are_not_editable_source_records_any_more() -> None:
    # population_share_fraction no longer lives inside COHORTS: it moved to
    # the editable COHORT_SHARES section above.
    for fields in assumptions.COHORTS.values():
        assert "population_share_fraction" not in fields


def test_cohort_share_edit_reaches_the_fixture_as_a_fraction() -> None:
    fixture = assumptions.cohort_fixture(
        {
            "average_uk.population_share_percent": 35.0,
            "always_plugged_in.population_share_percent": 6.0,
        }
    )
    shares = {c.cohort_id: c.population_share_fraction for c in fixture.cohorts}
    assert shares == {
        "average_uk": 0.35,
        "intelligent_octopus": 0.3,
        "infrequent_charging": 0.1,
        "infrequent_driving": 0.1,
        "scheduled_charging": 0.09,
        "always_plugged_in": 0.06,
    }


def test_cohort_shares_must_sum_to_100_within_one_hundredth_of_a_point() -> None:
    off_by_more = assumptions.editable_defaults() | {"average_uk.population_share_percent": 40.02}
    assert assumptions.validation_errors(off_by_more) == {
        "average_uk.population_share_percent": (
            "the six cohort shares must sum to 100% (currently 100.02%)"
        )
    }
    with pytest.raises(ValueError, match="must sum to 100%"):
        assumptions.resolve_values(off_by_more)

    # Within the +/-0.01 point tolerance: still valid.
    just_inside = assumptions.editable_defaults() | {"average_uk.population_share_percent": 40.009}
    assert assumptions.validation_errors(just_inside) == {}


def test_cohort_share_sum_error_is_attached_to_one_field_not_six() -> None:
    # draft_error (run_controller._set_draft) joins every {name: message} pair
    # into one banner line, so a message on all six would repeat six times.
    values = assumptions.editable_defaults() | {"average_uk.population_share_percent": 20.0}
    errors = assumptions.validation_errors(values)
    assert set(errors) == {assumptions.COHORT_SHARE_NAMES[0]}


def test_forecast_inputs_hold_the_removed_configs_values() -> None:
    inputs = assumptions.forecast_inputs(None, warmup_days=1, study_days=7)

    assert inputs["daily_miles_cv_by_cohort"] == {
        "average_uk": 0.30,
        "intelligent_octopus": 0.20,
        "infrequent_charging": 0.40,
        "infrequent_driving": 0.45,
        "scheduled_charging": 0.20,
        "always_plugged_in": 0.30,
    }
    assert inputs["personal_mileage_cv_by_cohort"] == {
        "average_uk": 0.15,
        "intelligent_octopus": 0.10,
        "infrequent_charging": 0.20,
        "infrequent_driving": 0.225,
        "scheduled_charging": 0.10,
        "always_plugged_in": 0.15,
    }
    behaviour = inputs["trip_behaviour"]
    # Decision 0004 item 51: truncated Student-t clocks, clip 3 h, df 4.
    assert behaviour["clock_clip_minutes"] == 180.0
    assert behaviour["clock_t_df"] == 4.0
    assert behaviour["by_cohort"]["average_uk"] == {
        "weekday_drive_probability": 0.85,
        "weekend_drive_probability": 0.75,
        "destination_dwell_minutes": 480,
        "drive_speed_mph": 30.0,
        "departure_scale_minutes": 75.0,
        "plug_in_scale_minutes": 60.0,
        # Decision 0004 items 42 and 51: the wider weekend plug-in spread.
        "weekend_plug_in_scale_minutes": 75.0,
        # Follow-up to item 42: 6 h weekend dwell (CNZ weekend plug-in timing).
        "weekend_destination_dwell_minutes": 360,
    }
    by_cohort = behaviour["by_cohort"]
    assert "weekend_destination_dwell_minutes" not in by_cohort["scheduled_charging"]
    assert {c: row["departure_scale_minutes"] for c, row in by_cohort.items()} == {
        "average_uk": 75.0,
        "intelligent_octopus": 75.0,
        "infrequent_charging": 75.0,
        "infrequent_driving": 90.0,
        "scheduled_charging": 30.0,
        "always_plugged_in": 75.0,
    }
    assert {c: row["plug_in_scale_minutes"] for c, row in by_cohort.items()} == {
        "average_uk": 60.0,
        "intelligent_octopus": 60.0,
        "infrequent_charging": 60.0,
        "infrequent_driving": 60.0,
        "scheduled_charging": 20.0,
        "always_plugged_in": 0.0,
    }
    # Flat-window cohorts keep their weekday plug-in scale at weekends.
    assert by_cohort["scheduled_charging"]["weekend_plug_in_scale_minutes"] == 20.0
    assert by_cohort["intelligent_octopus"]["weekend_plug_in_scale_minutes"] == 75.0
    # Warm-up day(s), the seven study days, then the morning after the last
    # night, which carries the last study day's base temperature
    # (RunSettings.sampled_day_count, decision 0004 item 52).
    assert inputs["base_temperature_c_by_day"].tolist() == [8, 8, 9, 11, 12, 10, 7, 9, 9]
    assert inputs["weather_sd_c"] == 2.0
    # Default threshold re-baselined from 20 % to 10 % by decision 0004 item 37.
    assert inputs["public_charge_assumptions"] == {
        "efficiency_fraction": 0.9,
        "top_up_threshold_soc_fraction": 0.1,
        "top_up_target_soc_fraction": 0.8,
    }
    # Every market-price input, by record name (decision 0004 items 53, 56).
    prices = inputs["price_assumptions"]
    assert set(prices) == set(sampling.MARKET_PRICE_INPUTS)
    assert prices["supply_reference_price_gbp_per_mwh"] == 78.0
    assert prices["day_ahead_sd_gbp_per_mwh"] == 24.0
    assert (
        prices["demand_shape_gw.winter_weekday"]
        == assumptions.SYSTEM["demand_shape_gw.winter_weekday"].value
    )
    assert prices["big_shock_rate_per_week"] == 5.0
    assert inputs["public_charge_gbp_per_kwh"] == 0.79


def test_forecast_inputs_apply_edited_values_in_the_samplers_units() -> None:
    inputs = assumptions.forecast_inputs(
        {
            "public_top_up_threshold_soc_percent": 30.0,
            "public_top_up_target_soc_percent": 90.0,
            "efficiency_fraction": 0.8,
            "supply_reference_price_gbp_per_mwh": 120.0,
            "public_charge_gbp_per_kwh": 0.5,
        },
        warmup_days=3,
        study_days=7,
    )

    public = inputs["public_charge_assumptions"]
    # Percent in the dialog, SoC fraction in the sampler.
    assert public["top_up_threshold_soc_fraction"] == pytest.approx(0.3)
    assert public["top_up_target_soc_fraction"] == pytest.approx(0.9)
    assert public["efficiency_fraction"] == 0.8
    assert inputs["price_assumptions"]["supply_reference_price_gbp_per_mwh"] == 120.0
    assert inputs["public_charge_gbp_per_kwh"] == 0.5
    # Three warm-up days, seven study days and the morning after the last night.
    assert inputs["base_temperature_c_by_day"].shape == (11,)


def test_run_settings_take_the_run_shape_and_the_kernel_scalars() -> None:
    settings = assumptions.run_settings({"vehicle_count": 12, "seed": 7}, date(2026, 9, 28))

    assert settings.start_local_date == date(2026, 9, 28)
    assert (settings.vehicle_count, settings.seed) == (12, 7)
    assert settings.evaluation_world_count == 100
    assert settings.study_days == 7
    # Smart charging needs no planning world (decision 0004 item 38).
    assert not hasattr(settings, "planning_world_count")
    assert settings.warmup_days == assumptions.SIMULATION["warmup_days"].value
    # Settled start (decision 0004 item 36): opening SoC is the 80 % preferred
    # target; the warm-up is seven days since item 52 and is editable (3-14).
    assert settings.opening_soc_fraction == 0.8
    assert settings.warmup_days == 7
    edited = assumptions.run_settings({"warmup_days": 3}, date(2026, 9, 28))
    assert edited.warmup_days == 3
    assert edited.sampled_day_count == 3 + 7 + 1
    for out_of_bounds in (2, 15):
        with pytest.raises(ValueError, match="warmup_days"):
            assumptions.run_settings({"warmup_days": out_of_bounds}, date(2026, 9, 28))
    assert settings.home_charge_efficiency == 0.92
    assert settings.reserve_soc_fraction == 0.2
    assert settings.weather_efficiency_sensitivity_fraction_per_c == 0.01
    # The superseded one-day runner's fields are gone (decision 0004 item 18).
    assert not hasattr(settings, "vehicle_ac_limit_kw")


def test_fixed_values_other_modules_read_are_the_records() -> None:
    # Decision 0004 item 21: modules read these values here, not their own literals.
    assert physics._REFERENCE_TEMPERATURE_C == 20.0
    assert physics._MAXIMUM_LOSS_FRACTION == 0.50
    assert sampling.ILLUSTRATIVE_PRICE_NOISE_AR1_COEFFICIENT == 0.92
    assert action.NOT_RECOVERED_TOLERANCE_KWH == 1e-6
    assert action.ILLUSTRATIVE_PUBLIC_CHARGE_GBP_PER_KWH == 0.79


def test_home_charging_has_one_editable_power_and_no_vehicle_ac_limit() -> None:
    """Decision 0004 item 34: one home charging power, not a separate vehicle limit."""

    assert set(assumptions.HOME_CHARGING) == {
        "home_charging_power_kw",
        "home_charge_efficiency",
        "opening_soc_fraction",
        "reserve_soc_fraction",
    }
    power = assumptions.HOME_CHARGING["home_charging_power_kw"]
    assert power.value == 7.0 and power.unit == "kW" and power.editable
    assert "H6:H11" in power.source
    for record in assumptions.editable_records():
        assert " AC" not in record.label and " AC" not in record.meaning
    for fields in assumptions.COHORTS.values():
        assert "home_charger_limit_kw" not in fields


def test_public_top_up_thresholds_are_editable_for_decision_0004_item_32() -> None:
    threshold = assumptions.PUBLIC_CHARGING["public_top_up_threshold_soc_percent"]
    target = assumptions.PUBLIC_CHARGING["public_top_up_target_soc_percent"]

    # Default lowered from 20 % to 10 % by decision 0004 item 37.
    assert threshold.value == pytest.approx(10.0)
    assert target.value == pytest.approx(80.0)
    assert threshold.editable and target.editable
    assert threshold.bounds[0] <= threshold.value <= threshold.bounds[1]
    assert target.bounds[0] <= target.value <= target.bounds[1]


def test_public_charging_holds_only_values_with_an_effect() -> None:
    # Under item 32 a charger is always available and charging time is not
    # modelled, so availability, charger power, acceptance and the SoC
    # ceiling changed no output and were removed (task M4-layout).
    assert set(assumptions.PUBLIC_CHARGING) == {
        "efficiency_fraction",
        "public_charge_gbp_per_kwh",
        "public_top_up_threshold_soc_percent",
        "public_top_up_target_soc_percent",
    }


def test_editable_fields_are_the_dialog_fields() -> None:
    clock_names = [
        name for name, record in assumptions.CONNECTION_CLOCKS.items() if record.editable
    ]
    assert [record.name for record in assumptions.editable_records()] == [
        *assumptions.COHORT_SHARE_NAMES,
        *clock_names,
        "sensitivity_fraction_per_c",
        "home_charging_power_kw",
        "home_charge_efficiency",
        "efficiency_fraction",
        "public_charge_gbp_per_kwh",
        "public_top_up_threshold_soc_percent",
        "public_top_up_target_soc_percent",
        "heating_gw_per_c",
        "wind_installed_gw",
        "solar_installed_gw",
        "mild_shock_rate_per_night",
        "big_shock_rate_per_week",
        "big_shock_median_gw",
        "shock_up_probability",
        "big_shock_known_share",
        "surprise_reveal_hours",
        "supply_reference_price_gbp_per_mwh",
        "daily_level_sd_gbp_per_mwh",
        "day_ahead_sd_gbp_per_mwh",
        "da_id_premium_gbp_per_mwh",
        "intraday_hourly_sd_gbp_per_mwh",
        # Intraday dispatch (intraday-dispatch-v1, K4): shown in "Public & prices".
        "trading.day_ahead_commitment_share",
        "trading.intraday_dispatch",
        "trading.replan_threshold_gbp_per_mwh",
        # The "Supplier" tab (supplier contract v1 §8, lane S2).
        "supplier.early_departure_risk_charge_gbp_per_mwh",
        *assumptions.SUPPLIER,
        *assumptions.CARBON,
        "departure_margin_hours",
        "timed_tariff_enabled",
        "timed_start_local_hour",
        "not_recovered_material_share_percent",
        *assumptions.ZONE_SHARE_NAMES,
        # Trading contract v1 §10.8 (lane J6): the skip-share records and
        # the two holiday switches stay in "Plugging & home charging"
        # (module order keeps them here, right after the zone shares);
        # the manufacturer, availability and commitment records sit in the
        # new "Firm MW" tab. Availability's own time-of-day and table
        # records (intraday_decision_local_time, blackout_windows) stay
        # fixed, so only the commitment rule appears from that dict.
        "behaviour.plug_in_skip_median",
        "behaviour.plug_in_day_factor_sd",
        "behaviour.plug_in_week_factor_sd",
        "behaviour.holiday_bank_holiday_monday",
        "behaviour.holiday_half_term_week",
        *assumptions.MANUFACTURER_SHARE_NAMES,
        *assumptions.MANUFACTURER_RESPONSE_RATE_NAMES,
        *assumptions.MANUFACTURER_OUTAGE_NAMES,
        "trading.commitment_rule",
        "vehicle_count",
        "evaluation_world_count",
        "warmup_days",
        "seed",
    ]


def test_clock_scales_and_df_are_editable_and_the_clip_is_fixed() -> None:
    # Decision 0004 item 51: editable illustrative scales and df.  The clip
    # stays fixed because the samplers' session-gap and same-day checks rely
    # on it, and Always Plugged-in has no plug-in clock to edit.
    clocks = assumptions.CONNECTION_CLOCKS
    editable = {name for name, record in clocks.items() if record.editable}
    assert editable == {
        "clock_t_df",
        *(f"departure_scale_minutes.{c}" for c in assumptions.COHORTS),
        *(f"plug_in_scale_minutes.{c}" for c in assumptions.COHORTS if c != "always_plugged_in"),
        "weekend_plug_in_scale_minutes.average_uk",
        "weekend_plug_in_scale_minutes.intelligent_octopus",
    }
    assert not clocks["clock_clip_minutes"].editable
    for record in clocks.values():
        assert record.evidence == "illustrative" and "item 51" in record.source
    # Per-cohort fields name their cohort, so the dialog's labels differ.
    labels = [record.label for record in clocks.values()]
    assert len(labels) == len(set(labels))
    edited = assumptions.forecast_inputs(
        {"clock_t_df": 30.0, "departure_scale_minutes.scheduled_charging": 45.0},
        warmup_days=3,
        study_days=7,
    )["trip_behaviour"]
    assert edited["clock_t_df"] == 30.0
    assert edited["by_cohort"]["scheduled_charging"]["departure_scale_minutes"] == 45.0


# ---------------------------------------------------------------------------
# Validating edited values.
# ---------------------------------------------------------------------------


def test_defaults_are_valid_and_resolve_to_themselves() -> None:
    assert assumptions.validation_errors(assumptions.editable_defaults()) == {}
    assert assumptions.resolve_values(None) == assumptions.editable_defaults()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("home_charging_power_kw", 0.5, "must be between 1 and 22"),
        ("home_charging_power_kw", 23.0, "must be between 1 and 22"),
        ("home_charge_efficiency", float("nan"), "must be a finite number"),
        ("public_charge_gbp_per_kwh", "0.79", "must be a number"),
        ("home_charge_efficiency", True, "must be a number"),
        ("vehicle_count", 12.5, "must be a whole number"),
        ("evaluation_world_count", 101, "must be between 1 and 100"),
    ],
)
def test_validation_uses_each_records_bounds(name: str, value: object, message: str) -> None:
    values = assumptions.editable_defaults() | {name: value}

    assert assumptions.validation_errors(values) == {name: message}
    with pytest.raises(ValueError, match=message):
        assumptions.resolve_values({name: value})


def test_top_up_target_must_stay_above_the_threshold() -> None:
    values = assumptions.editable_defaults() | {
        "public_top_up_threshold_soc_percent": 50.0,
        "public_top_up_target_soc_percent": 50.0,
    }
    assert assumptions.validation_errors(values) == {
        "public_top_up_target_soc_percent": "must be above the top-up threshold"
    }


def test_timed_start_local_hour_is_a_plain_half_hour_multiple_in_bounds() -> None:
    # Decision 0007 follow-up (8 October 2026): a bounded number_input has no
    # reachable "unset" value in Streamlit (an empty commit resolves to
    # min_value, not NaN), so this field is now a plain number, default 0.0,
    # never unset; "off" is the separate ``timed_tariff_enabled`` switch
    # (``test_timed_tariff_switch_off_reaches_the_model_as_nan``). A given
    # value must still be a half-hour multiple inside bounds, since the
    # barred window is read off half-hour slots.
    defaults = assumptions.editable_defaults()
    assert defaults["timed_start_local_hour"] == 0.0
    assert assumptions.validation_errors(defaults) == {}
    assert assumptions.validation_errors(defaults | {"timed_start_local_hour": float("nan")}) == {
        "timed_start_local_hour": "must be a finite number"
    }
    assert assumptions.validation_errors(defaults | {"timed_start_local_hour": 1.5}) == {}
    assert assumptions.validation_errors(defaults | {"timed_start_local_hour": 1.2}) == {
        "timed_start_local_hour": "must be a half-hour multiple"
    }
    assert assumptions.validation_errors(defaults | {"timed_start_local_hour": 24.0}) == {
        "timed_start_local_hour": "must be between 0 and 23.5"
    }
    # An invalid hour still blocks a run with the switch off: a switch never
    # gates another field's own validity here, the same way
    # supplier.customer_reward_gbp_per_ev_per_month's bounds are checked
    # whatever supplier.customer_reward_mode holds.
    assert assumptions.validation_errors(
        defaults | {"timed_tariff_enabled": 0, "timed_start_local_hour": 1.2}
    ) == {"timed_start_local_hour": "must be a half-hour multiple"}


def test_timed_tariff_switch_off_reaches_the_model_as_nan() -> None:
    # The downstream interface is unchanged: ``forecast_inputs`` is the one
    # place that turns the switch + hour back into NaN-means-off, so
    # ``run_forecast``/the kernel still only ever see a plain float.
    for hour in (0.0, 12.0, 23.5):
        inputs = assumptions.forecast_inputs(
            {"timed_tariff_enabled": 0, "timed_start_local_hour": hour},
            warmup_days=3,
            study_days=7,
        )
        assert math.isnan(inputs["timed_start_local_hour"])
    inputs_on = assumptions.forecast_inputs(
        {"timed_tariff_enabled": 1, "timed_start_local_hour": 12.0},
        warmup_days=3,
        study_days=7,
    )
    assert inputs_on["timed_start_local_hour"] == pytest.approx(12.0)


def test_unknown_names_are_rejected_and_whole_numbers_come_back_as_int() -> None:
    with pytest.raises(ValueError, match="unknown editable assumption: vehicle_ac_limit_kw"):
        assumptions.resolve_values({"vehicle_ac_limit_kw": 7.0})

    resolved = assumptions.resolve_values(
        {"vehicle_count": 12.0, "supply_reference_price_gbp_per_mwh": 100}
    )
    assert resolved["vehicle_count"] == 12 and isinstance(resolved["vehicle_count"], int)
    assert isinstance(resolved["supply_reference_price_gbp_per_mwh"], float)


def test_result_assumptions_carry_the_values_the_run_used() -> None:
    records = {
        record.name: record
        for record in assumptions.result_assumptions({"home_charging_power_kw": 3.6})
    }

    assert records["home_charging_power_kw"].value == 3.6
    assert records["public_charge_gbp_per_kwh"].value == 0.79
    _validate_assumptions(tuple(records.values()))


# ---------------------------------------------------------------------------
# Action and simulation.
# ---------------------------------------------------------------------------


def test_action_records_are_the_smart_charging_margin_and_tolerance() -> None:
    # Decision 0004 item 38: the 18:00 window, ceiling and eligibility screen
    # are gone; the smart charger's departure margin is an editable
    # illustrative assumption, default two hours since item 51 widened the
    # departure clock.
    # Plan B4 adds the expected-shape look-back: a fixed, illustrative
    # record, not a dialog field.  The publication hour it pairs with is the
    # prices group's one record (13:00), shared with the intraday updates.
    assert set(assumptions.ACTION) == {
        "departure_margin_hours",
        "timed_tariff_enabled",
        "timed_start_local_hour",
        "expected_price_shape_days",
        "not_recovered_material_share_percent",
        "not_recovered_tolerance_kwh",
    }
    record = assumptions.ACTION["expected_price_shape_days"]
    assert (record.value, record.editable, record.evidence) == (7, False, "illustrative")
    assert assumptions.PRICES["day_ahead_publication_local_hour"].value == 13
    margin = assumptions.ACTION["departure_margin_hours"]
    assert (margin.value, margin.unit, margin.evidence) == (2.0, "hours", "illustrative")
    assert margin.editable and margin.bounds[0] <= margin.value <= margin.bounds[1]
    assert assumptions.ACTION["not_recovered_tolerance_kwh"].value == 1e-6
    assert assumptions.forecast_inputs(None, warmup_days=3, study_days=7)[
        "departure_margin_hours"
    ] == pytest.approx(2.0)
    # Decision 0007 follow-up (8 October 2026): present by default (the
    # switch), at midnight (the hour), an optional London clock hour
    # (half-hour steps) that bars home charging from noon until it
    # (``test_timed_start_local_hour_is_a_plain_half_hour_multiple_in_bounds``,
    # ``test_timed_tariff_switch_off_reaches_the_model_as_nan``).
    switch = assumptions.ACTION["timed_tariff_enabled"]
    assert (switch.value, switch.editable, switch.bounds) == (1, True, (0.0, 1.0))
    timed = assumptions.ACTION["timed_start_local_hour"]
    assert (timed.value, timed.editable, timed.bounds) == (0.0, True, (0.0, 23.5))
    assert assumptions.forecast_inputs(None, warmup_days=3, study_days=7)[
        "timed_start_local_hour"
    ] == pytest.approx(0.0)


def test_simulation_defaults_are_the_accepted_run_shape() -> None:
    assert assumptions.SIMULATION["vehicle_count"].value == 1_000
    assert assumptions.SIMULATION["evaluation_world_count"].value == 100
    assert "planning_world_count" not in assumptions.SIMULATION
    assert assumptions.SIMULATION["study_days"].value == 7
    assert assumptions.SIMULATION["seed"].value == 42
    warmup = assumptions.SIMULATION["warmup_days"]
    assert (warmup.value, warmup.editable, warmup.bounds) == (7, True, (3.0, 14.0))


# ---------------------------------------------------------------------------
# CNZ context.
# ---------------------------------------------------------------------------


def test_cnz_context_matches_the_contracts_names_units_and_values() -> None:
    source_p13 = assumptions.CNZ_CONTEXT["cnz_median_plug_in_soc_percent"].source
    assert "52%" in source_p13
    assert "3% below 10%" in source_p13
    assert "fewer than 10% below 20%" in source_p13
    assert "18:00" in assumptions.CNZ_CONTEXT["cnz_weekday_plug_in_mode_local_hour"].source

    median = assumptions.CNZ_CONTEXT["cnz_median_plug_in_soc_percent"]
    below_10 = assumptions.CNZ_CONTEXT["cnz_share_plug_ins_below_10_percent_soc"]
    below_20 = assumptions.CNZ_CONTEXT["cnz_share_plug_ins_below_20_percent_soc_upper_bound"]
    mode_hour = assumptions.CNZ_CONTEXT["cnz_weekday_plug_in_mode_local_hour"]

    # Rescaled to a percent per contract section 8.1 (was a 0-1 fraction here before).
    assert median.value == pytest.approx(52.0)
    assert median.unit == "percent"
    assert below_10.value == pytest.approx(0.03)
    assert below_10.unit == "fraction"
    assert below_20.value == pytest.approx(0.10)
    assert below_20.unit == "fraction"
    assert mode_hour.value == 18
    assert mode_hour.unit == "hour (London)"

    for record in (median, below_10, below_20, mode_hour):
        assert record.evidence == "source"
        assert record.editable is False
        assert record.group == "Plugging & home charging"


def test_cnz_record_names_match_the_contract_exactly() -> None:
    contract_names = {
        "cnz_median_plug_in_soc_percent",
        "cnz_share_plug_ins_below_10_percent_soc",
        "cnz_share_plug_ins_below_20_percent_soc_upper_bound",
        "cnz_weekday_plug_in_mode_local_hour",
    }
    assert set(assumptions.CNZ_CONTEXT) == contract_names
    for name, record in assumptions.CNZ_CONTEXT.items():
        assert record.name == name


# ---------------------------------------------------------------------------
# Model rules and source checks.
# ---------------------------------------------------------------------------


def test_model_rules_are_recorded_as_nonempty_plain_english_statements() -> None:
    assert len(assumptions.MODEL_RULES) == 8
    assert all(isinstance(rule, str) and rule.strip() for rule in assumptions.MODEL_RULES)


def test_model_rules_state_the_top_up_rule_as_implemented() -> None:
    combined = " ".join(assumptions.MODEL_RULES)
    assert "BEING IMPLEMENTED" not in combined
    assert "EVs never strand" in combined
    # The rules removed for item 32 (stranding stays forever; destination
    # public charging gated only by return-leg infeasibility) must not remain
    # stated as unconditional current behaviour.
    assert "never resumes driving" not in combined


def test_source_checks_are_recorded() -> None:
    assert len(assumptions.SOURCE_CHECKS) >= 2
    combined = " ".join(assumptions.SOURCE_CHECKS)
    assert "intelligent_octopus" in combined
    assert "7.0" in combined and "7.2" in combined
    assert "item 34" in combined
    assert "CURRENT_MODEL_VALUE" not in dir(assumptions)


def test_weekend_connection_windows_are_labelled_cnz_adaptations() -> None:
    # Decision 0004 item 42: illustrative, read-only and citing the figures.
    records = [
        *(r for fields in assumptions.CONNECTION_WINDOWS.values() for r in fields.values()),
        *(
            r
            for name, r in assumptions.CONNECTION_CLOCKS.items()
            if name.startswith("weekend_plug_in_scale_minutes.")
        ),
    ]
    assert len(records) == 6
    for record in records:
        assert record.evidence == "illustrative"
        assert "illustrative adaptation of CNZ figures" in record.source
        assert "p.11 Fig.4" in record.source and "p.12 Fig.5" in record.source
    names = {r.name for r in assumptions.result_assumptions()}
    assert "average_uk.weekend_connection_arrival_local_hour" in names
    assert "weekend_plug_in_scale_minutes.intelligent_octopus" in names


def test_zone_shares_must_sum_to_one_and_headroom_is_reported_only() -> None:
    # Trading contract v1 §3, §7: four editable shares (0.25 each by default)
    # and an optional, non-editable headroom per zone (NaN = none).
    values = assumptions.editable_defaults()
    assert [values[name] for name in assumptions.ZONE_SHARE_NAMES] == [0.25] * 4
    edited = values | {"zones.share.zone_1": 0.4}
    errors = assumptions.validation_errors(edited)
    assert "zones.share.zone_1" in errors and "sum to 1" in errors["zones.share.zone_1"]
    moved = edited | {"zones.share.zone_2": 0.1}
    assert assumptions.validation_errors(moved) == {}
    inputs = assumptions.forecast_inputs(moved, warmup_days=7, study_days=7)
    assert inputs["zone_shares"] == (0.4, 0.1, 0.25, 0.25)
    headroom = [assumptions.ZONES[f"headroom_kw.{z}"] for z in assumptions.ZONE_IDS]
    assert all(not record.editable and math.isnan(record.value) for record in headroom)


# ---------------------------------------------------------------------------
# The "Firm MW" tab (trading contract v1 §10.8, lane J6): manufacturer
# shares, the commitment rule, and the fixed availability records.
# ---------------------------------------------------------------------------


def test_manufacturer_shares_must_sum_to_one() -> None:
    values = assumptions.editable_defaults()
    assert [values[name] for name in assumptions.MANUFACTURER_SHARE_NAMES] == [0.25] * 4
    edited = values | {"manufacturers.share.m1": 0.4}
    errors = assumptions.validation_errors(edited)
    assert "manufacturers.share.m1" in errors and "sum to 1" in errors["manufacturers.share.m1"]
    moved = edited | {"manufacturers.share.m2": 0.1}
    assert assumptions.validation_errors(moved) == {}
    resolved = assumptions.resolve_values(moved)
    assert [resolved[name] for name in assumptions.MANUFACTURER_SHARE_NAMES] == [
        0.4,
        0.1,
        0.25,
        0.25,
    ]


def test_commitment_rule_is_a_two_option_switch_checked_against_its_allowed_set() -> None:
    # Trading contract v1 §10.4, §10.8: the one text-switch record, editable
    # in the "Firm MW" tab, bounds=None (checked against COMMITMENT_RULES,
    # not a numeric range).
    record = assumptions.AVAILABILITY["trading.commitment_rule"]
    assert record.editable and record.bounds is None and record.value == "fixed_share"

    values = assumptions.editable_defaults()
    assert assumptions.validation_errors(values) == {}

    edited = values | {"trading.commitment_rule": "newsvendor"}
    assert assumptions.validation_errors(edited) == {}
    assert assumptions.resolve_values(edited)["trading.commitment_rule"] == "newsvendor"

    bad = values | {"trading.commitment_rule": "always_full"}
    errors = assumptions.validation_errors(bad)
    assert "trading.commitment_rule" in errors
    with pytest.raises(ValueError, match="trading.commitment_rule"):
        assumptions.resolve_values(bad)


def test_availability_time_and_blackout_records_stay_fixed() -> None:
    # The intraday decision time is a London "HH:MM" choice and the
    # blackout table is edited through its own st.data_editor
    # (run_controller.draft_blackout_windows), not the generic per-record
    # number field: both stay editable=False (§10.5b, §10.8).
    for name in ("availability.intraday_decision_local_time", "availability.blackout_windows"):
        record = assumptions.AVAILABILITY[name]
        assert not record.editable
        assert record.group == "Firm MW"


def test_firm_mw_tab_holds_the_manufacturer_availability_and_commitment_records() -> None:
    assert "Firm MW" in assumptions.ASSUMPTION_GROUPS
    names = {record.name for record in assumptions.records_in_group("Firm MW")}
    assert names == {
        *assumptions.MANUFACTURER_SHARE_NAMES,
        *assumptions.MANUFACTURER_RESPONSE_RATE_NAMES,
        *assumptions.MANUFACTURER_OUTAGE_NAMES,
        "availability.intraday_decision_local_time",
        "availability.blackout_windows",
        "trading.commitment_rule",
    }
    # The skip-share records and the two holiday switches sit in "Plugging &
    # home charging" next to the plug probabilities they act on (§10.8).
    plugging = {record.name for record in assumptions.records_in_group("Plugging & home charging")}
    assert {
        "behaviour.plug_in_skip_median",
        "behaviour.plug_in_day_factor_sd",
        "behaviour.plug_in_week_factor_sd",
        "behaviour.holiday_bank_holiday_monday",
        "behaviour.holiday_half_term_week",
    } <= plugging


# ---------------------------------------------------------------------------
# The "Supplier" tab (supplier contract v1 §8, lane S2).
# ---------------------------------------------------------------------------


def test_supplier_tab_holds_the_supplier_and_carbon_records_and_the_risk_charge() -> None:
    assert "Supplier" in assumptions.ASSUMPTION_GROUPS
    names = [record.name for record in assumptions.records_in_group("Supplier")]
    assert names == [
        "supplier.early_departure_risk_charge_gbp_per_mwh",
        *assumptions.SUPPLIER,
        *assumptions.CARBON,
    ]
    assert all(record.editable for record in assumptions.records_in_group("Supplier"))


def test_an_unset_optional_term_is_valid_but_a_nan_elsewhere_is_not() -> None:
    values = assumptions.editable_defaults()
    assert assumptions.validation_errors(values) == {}
    assert math.isnan(assumptions.resolve_values()["supplier.platform_fee_gbp_per_ev_per_month"])

    errors = assumptions.validation_errors({**values, "home_charging_power_kw": math.nan})
    assert errors == {"home_charging_power_kw": "must be a finite number"}


def test_carbon_floor_may_not_exceed_the_cap() -> None:
    values = {**assumptions.editable_defaults(), "carbon.intensity_floor_gco2_per_kwh": 500.0}
    assert assumptions.validation_errors(values) == {
        "carbon.intensity_floor_gco2_per_kwh": "must not be above the carbon intensity cap"
    }
