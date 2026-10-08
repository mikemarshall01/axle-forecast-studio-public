"""Threshold public top-ups (decision 0004 item 32; supersedes items 3 and 31).

The fixture EV has a 10 kWh battery, 10 miles/kWh and 60 mph, so 10 miles is
1 kWh and 10 minutes of driving.  Threshold 20 % = 2 kWh, target 80 % = 8 kWh,
so every top-up after the first adds 6 kWh.  Public efficiency is 0.8, so
grid import is battery energy / 0.8.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from axle_studio.model import assumptions
from axle_studio.model.assumptions import PUBLIC_CHARGING
from axle_studio.model.physics import (
    PublicTopUp,
    _top_up_battery_kwh,
    public_top_up,
    simulate_fleet_intervals,
)
from axle_studio.model.settings import RunSettings
from axle_studio.ui import run_controller


def _fixture(
    *, opening: float = 0.5, days: int = 1, start: date = date(2025, 1, 7)
) -> tuple[RunSettings, pd.DataFrame, dict[str, np.ndarray]]:
    # The study starts at 12:00 GMT on ``start`` (decision 0004 item 52), so
    # the hand-built trips below leave in the afternoon, inside the horizon.
    settings = RunSettings(
        start_local_date=start,
        warmup_days=0,
        study_days=days,
        vehicle_count=1,
        seed=1,
        evaluation_world_count=1,
        opening_soc_fraction=opening,
        reserve_soc_fraction=0.0,
        home_charge_efficiency=0.9,
        weather_efficiency_sensitivity_fraction_per_c=0.01,
    )
    units = pd.DataFrame(
        {
            "population_id": ["p"],
            "unit_id": ["u"],
            "cohort_id": ["c"],
            "physical_capacity_kwh": [10.0],
            "efficiency_miles_per_battery_kwh": [10.0],
            "home_charger_limit_kw": [2.0],
            "preferred_target_soc_fraction": [1.0],
            "runtime_departure_local_hour": [8],
            "runtime_arrival_local_hour": [18],
        }
    )
    # The noon-to-noon horizon samples one more London date than it has days
    # (RunSettings.sampled_day_count, decision 0004 item 52).
    shape = (1, settings.sampled_day_count, 1)
    no_time = np.full(shape, pd.NaT, dtype=object)
    inputs = {
        "drives_today": np.zeros(shape, dtype=bool),
        "outbound_miles": np.zeros(shape),
        "return_miles": np.zeros(shape),
        "trip_departure_utc": no_time.copy(),
        "destination_dwell_seconds": np.zeros(shape),
        "drive_speed_mph": np.full(shape, 60.0),
        "connection_session_accepted": np.zeros(shape, dtype=bool),
        "connection_start_utc": no_time.copy(),
        "connection_end_utc": no_time.copy(),
        "desired_pre_drive_soc_fraction": np.ones(shape),
    }
    return settings, units, inputs


def _trip(
    inputs: dict[str, np.ndarray],
    day: int,
    departure: pd.Timestamp,
    *,
    outbound: float,
    returning: float,
    dwell_minutes: float = 0.0,
    speed: float = 60.0,
) -> None:
    inputs["drives_today"][0, day, 0] = True
    inputs["trip_departure_utc"][0, day, 0] = departure
    inputs["outbound_miles"][0, day, 0] = outbound
    inputs["return_miles"][0, day, 0] = returning
    inputs["destination_dwell_seconds"][0, day, 0] = dwell_minutes * 60.0
    inputs["drive_speed_mph"][0, day, 0] = speed


# Efficiency 0.8, threshold 20 %, target 80 % (the module docstring's numbers).
_PUBLIC = PublicTopUp(efficiency_fraction=0.8, threshold_soc_fraction=0.2, target_soc_fraction=0.8)


def _run(settings, units, inputs, public: PublicTopUp = _PUBLIC) -> pd.DataFrame:
    fleet = simulate_fleet_intervals(settings, units, inputs, public_top_up=public)[0]
    normal = fleet[fleet.path_id == "normal"].set_index("interval_start_utc")
    # Checks every test relies on: conservation, 0-100 % SoC, no unserved travel.
    assert normal["conservation_residual_kwh"].abs().max() <= 1e-12
    assert normal["closing_soc_percent"].between(0.0, 100.0).all()
    assert normal["unserved_outbound_travel_battery_kwh"].eq(0.0).all()
    assert normal["unserved_return_travel_battery_kwh"].eq(0.0).all()
    np.testing.assert_allclose(
        normal["public_grid_import_kwh"] * 0.8, normal["public_battery_added_kwh"], atol=1e-12
    )
    assert normal["public_charging_fraction"].eq(0.0).all()
    return normal


def test_ev_below_threshold_mid_leg_tops_up_to_target_and_completes() -> None:
    # 5 kWh opening.  20 miles out (14:00-14:20, 2 kWh) leaves 3 kWh; 30 min
    # dwell; 20 miles back (14:50-15:10) draws 1 kWh in the 14:30 slot, which
    # ends exactly at the 2 kWh threshold (no top-up), and 1 kWh in the 15:00
    # slot, which would end at 1 kWh.  So it tops up there from 2 to 8 kWh
    # (6 kWh battery, 7.5 kWh grid) and arrives home with 7 kWh.
    settings, units, inputs = _fixture()
    _trip(
        inputs,
        0,
        pd.Timestamp("2025-01-07 14:00Z"),
        outbound=20.0,
        returning=20.0,
        dwell_minutes=30.0,
    )
    normal = _run(settings, units, inputs)

    slots = pd.to_datetime(["2025-01-07 14:00Z", "2025-01-07 14:30Z", "2025-01-07 15:00Z"])
    np.testing.assert_allclose(normal.loc[slots, "closing_battery_kwh"], [3.0, 2.0, 7.0])
    np.testing.assert_allclose(normal.loc[slots, "public_battery_added_kwh"], [0.0, 0.0, 6.0])
    assert normal["public_grid_import_kwh"].sum() == pytest.approx(7.5)
    assert normal["served_return_travel_battery_kwh"].sum() == pytest.approx(2.0)
    # Arrives home at 15:10: the rest of that slot is at home, not away.
    assert normal.loc[pd.Timestamp("2025-01-07 15:00Z"), "driving_fraction"] == pytest.approx(
        1.0 / 3.0
    )


def test_trip_larger_than_battery_gets_repeated_top_ups_within_one_slot() -> None:
    # 150 miles at 300 mph fills exactly the 14:00 slot: a 15 kWh leg from a
    # 10 kWh battery holding 5 kWh.  It reaches 2 kWh after 3 kWh, tops up to
    # 8, uses 6, tops up again to 8 and uses the last 6: two top-ups, 12 kWh.
    # Closing stock is 5 + 12 - 15 = 2 kWh.  The 10 mile return (2 min)
    # would then take it below 2 kWh, so it tops up once more to 8 and ends
    # at 7 kWh.
    settings, units, inputs = _fixture()
    _trip(
        inputs,
        0,
        pd.Timestamp("2025-01-07 14:00Z"),
        outbound=150.0,
        returning=10.0,
        speed=300.0,
    )
    normal = _run(settings, units, inputs)

    ten = pd.Timestamp("2025-01-07 14:00Z")
    assert normal.loc[ten, "requested_outbound_travel_battery_kwh"] == pytest.approx(15.0)
    assert normal.loc[ten, "served_outbound_travel_battery_kwh"] == pytest.approx(15.0)
    assert normal.loc[ten, "public_battery_added_kwh"] == pytest.approx(12.0)
    half_past = pd.Timestamp("2025-01-07 14:30Z")
    assert normal.loc[half_past, "public_battery_added_kwh"] == pytest.approx(6.0)
    assert normal.loc[half_past, "closing_battery_kwh"] == pytest.approx(7.0)


def test_trip_larger_than_battery_over_several_slots_never_strands() -> None:
    # 150 miles at 60 mph: 3 kWh per slot for five slots (14:00-16:30).
    # Stock 5 -> 2 (no top-up, exactly at threshold) -> top-up, 5 -> 2 ->
    # top-up, 5 -> 2; the 1 kWh return at 16:30 tops up once more to 7.
    settings, units, inputs = _fixture()
    _trip(inputs, 0, pd.Timestamp("2025-01-07 14:00Z"), outbound=150.0, returning=10.0)
    normal = _run(settings, units, inputs)

    slots = pd.date_range("2025-01-07 14:00Z", periods=6, freq="30min")
    np.testing.assert_allclose(
        normal.loc[slots, "closing_battery_kwh"], [2.0, 5.0, 2.0, 5.0, 2.0, 7.0]
    )
    np.testing.assert_allclose(
        normal.loc[slots, "public_battery_added_kwh"], [0.0, 6.0, 0.0, 6.0, 0.0, 6.0]
    )


def test_no_top_up_when_stock_never_goes_below_threshold() -> None:
    settings, units, inputs = _fixture()
    _trip(inputs, 0, pd.Timestamp("2025-01-07 14:00Z"), outbound=15.0, returning=15.0)
    normal = _run(settings, units, inputs)

    assert normal["public_grid_import_kwh"].eq(0.0).all()
    assert normal["closing_battery_kwh"].iloc[-1] == pytest.approx(2.0)


def test_ev_already_below_threshold_tops_up_before_driving() -> None:
    # Opening 1 kWh is below the 2 kWh threshold, so the first 1 kWh request
    # tops up at once from 1 to 8 kWh (7 kWh), ending at 7 kWh.
    settings, units, inputs = _fixture(opening=0.1)
    _trip(inputs, 0, pd.Timestamp("2025-01-07 14:00Z"), outbound=10.0, returning=10.0)
    normal = _run(settings, units, inputs)

    ten = pd.Timestamp("2025-01-07 14:00Z")
    assert normal.loc[ten, "public_battery_added_kwh"] == pytest.approx(7.0)
    assert normal.loc[ten, "closing_battery_kwh"] == pytest.approx(6.0)


def test_top_up_helper_counts_are_hand_checkable() -> None:
    threshold = np.full(5, 2.0)
    target = np.full(5, 8.0)
    stock = np.array([5.0, 5.0, 5.0, 1.0, 5.0])
    requested = np.array([3.0, 15.0, 15.1, 1.0, 0.0])

    added, count = _top_up_battery_kwh(stock, requested, threshold, target)

    np.testing.assert_array_equal(count, [0.0, 2.0, 3.0, 1.0, 0.0])
    np.testing.assert_allclose(added, [0.0, 12.0, 18.0, 7.0, 0.0])
    closing = stock + added - requested
    assert (closing >= 2.0 - 1e-12).all()
    assert (closing <= 8.0).all()


def test_run_inputs_carry_the_item_32_defaults_from_the_assumptions() -> None:
    # Every run passes its top-up rule explicitly; there is no kernel fallback.
    # Default threshold re-baselined from 20 % to 10 % by decision 0004 item 37.
    public = assumptions.forecast_inputs(None, warmup_days=1, study_days=7)[
        "public_charge_assumptions"
    ]
    assert public == {
        "efficiency_fraction": PUBLIC_CHARGING["efficiency_fraction"].value,
        "top_up_threshold_soc_fraction": 0.10,
        "top_up_target_soc_fraction": 0.80,
    }
    assert public_top_up(**public) == PublicTopUp(0.9, 0.1, 0.8)


def test_editable_threshold_and_target_change_the_top_up() -> None:
    # Threshold 50 % (5 kWh), target 100 %: a 1 kWh request from 5 kWh tops up
    # from 5 to 10 kWh and ends at 9 kWh.
    settings, units, inputs = _fixture()
    _trip(inputs, 0, pd.Timestamp("2025-01-07 14:00Z"), outbound=10.0, returning=10.0)
    public = PublicTopUp(
        efficiency_fraction=0.8, threshold_soc_fraction=0.5, target_soc_fraction=1.0
    )
    normal = _run(settings, units, inputs, public)

    ten = pd.Timestamp("2025-01-07 14:00Z")
    assert normal.loc[ten, "public_battery_added_kwh"] == pytest.approx(5.0)
    assert normal.loc[ten, "closing_battery_kwh"] == pytest.approx(8.0)


@pytest.mark.parametrize(("threshold", "target"), [(0.8, 0.8), (0.9, 0.8), (-0.1, 0.8), (0.2, 1.1)])
def test_top_up_fractions_are_validated(threshold: float, target: float) -> None:
    with pytest.raises(ValueError, match="top-up fractions"):
        public_top_up(
            efficiency_fraction=0.8,
            top_up_threshold_soc_fraction=threshold,
            top_up_target_soc_fraction=target,
        )


@pytest.mark.parametrize("efficiency", [0.0, 1.1, float("nan")])
def test_public_efficiency_is_validated(efficiency: float) -> None:
    with pytest.raises(ValueError, match="efficiency_fraction"):
        public_top_up(
            efficiency_fraction=efficiency,
            top_up_threshold_soc_fraction=0.2,
            top_up_target_soc_fraction=0.8,
        )


def test_clock_change_week_tops_up_daily_and_conserves_energy() -> None:
    # London falls back on Sunday 25 October 2026.  A 200-mile round trip each
    # day (20 kWh, twice the battery) with no home charging needs top-ups
    # every day; the week must still conserve energy, keep 7 x 48 UTC slots
    # and never strand.  The study runs noon to noon (decision 0004 item 52),
    # so each trip leaves at 13:00 London and is back before the next noon.
    start = date(2026, 10, 22)
    settings, units, inputs = _fixture(days=7, start=start)
    for day in range(7):
        local_date = start + timedelta(days=day)
        departure = pd.Timestamp(f"{local_date} 13:00", tz="Europe/London").tz_convert("UTC")
        _trip(inputs, day, departure, outbound=100.0, returning=100.0, dwell_minutes=60.0)
    normal = _run(settings, units, inputs)

    assert len(normal) == 7 * 48
    assert (normal["public_battery_added_kwh"].groupby(normal.index.date).sum() > 0.0).sum() == 7
    # Whole-week balance: public energy = travel served + closing - opening.
    served = (
        normal["served_outbound_travel_battery_kwh"] + normal["served_return_travel_battery_kwh"]
    ).sum()
    assert served == pytest.approx(7 * 20.0)
    assert normal["public_battery_added_kwh"].sum() == pytest.approx(
        served + normal["closing_battery_kwh"].iloc[-1] - normal["opening_battery_kwh"].iloc[0]
    )
    assert normal["closing_battery_kwh"].min() >= 2.0 - 1e-12


@pytest.mark.parametrize("model", run_controller.MODELS)
def test_seed_42_run_has_no_unserved_travel(model: str) -> None:
    # The physics review found 10.5 % of EV-weeks stranded at this setting
    # before item 32; unserved travel is now zero by construction.
    result = run_controller.run_model(
        model, {"vehicle_count": 200, "evaluation_world_count": 10, "seed": 42}, date(2026, 9, 28)
    )
    fleet = result.fleet_world_intervals

    for leg in ("outbound", "return"):
        assert fleet[f"unserved_{leg}_travel_battery_kwh"].eq(0.0).all()
        np.testing.assert_array_equal(
            fleet[f"served_{leg}_travel_battery_kwh"], fleet[f"requested_{leg}_travel_battery_kwh"]
        )
    # run_model returns a ForecastResult, whose frames use contract v2 names.
    assert fleet["public_import_kwh"].sum() > 0.0
    assert fleet["conservation_residual_kwh"].abs().max() <= 1e-12
    assert fleet["battery_soc_percent"].between(0.0, 100.0).all()
