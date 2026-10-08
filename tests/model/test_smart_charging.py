"""Smart home charging on the selected path (decision 0004 item 38).

Hand-checkable cases first: one EV, 10 kWh battery, 4 kW home charger (2 kWh
per half-hour), 100 % charging efficiency and an 80 % (8 kWh) target, so
every planned kWh can be checked by eye.  Then invariants and common random
numbers on real runs.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

import axle_studio.model.forecast as forecast_module
from axle_studio.model.action import (
    SmartCharging,
    plan_cheapest_slots,
    smart_charging_inputs,
)
from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.model.physics import PublicTopUp, simulate_fleet_intervals
from axle_studio.model.settings import RunSettings

_TOP_UP = PublicTopUp(efficiency_fraction=0.9, threshold_soc_fraction=0.1, target_soc_fraction=0.8)
_WINTER = date(2025, 1, 7)  # GMT, so London clock time equals UTC


# --------------------------------------------------------------------------
# The planner alone
# --------------------------------------------------------------------------


def test_planner_fills_the_cheapest_slots_first_with_one_partial_slot() -> None:
    prices = np.array([[50.0, 10.0, 30.0, 20.0, np.inf]])
    planned = plan_cheapest_slots(np.array([5.0]), np.array([2.0]), prices)
    # Cheapest 10 -> 2 kWh, then 20 -> 2 kWh, then 30 -> the last 1 kWh.
    np.testing.assert_allclose(planned, [[0.0, 2.0, 1.0, 2.0, 0.0]])


def test_planner_fills_every_window_slot_when_the_window_is_too_short() -> None:
    prices = np.array([[30.0, 10.0, np.inf, np.inf]])
    planned = plan_cheapest_slots(np.array([9.0]), np.array([2.0]), prices)
    np.testing.assert_allclose(planned, [[2.0, 2.0, 0.0, 0.0]])


def test_planner_breaks_price_ties_towards_earlier_slots_and_plans_nothing_for_no_need() -> None:
    prices = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]])
    planned = plan_cheapest_slots(np.array([3.0, 0.0]), np.array([2.0, 2.0]), prices)
    np.testing.assert_allclose(planned, [[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


def test_expected_departures_follow_the_london_clock_across_the_autumn_change() -> None:
    settings = _settings(start=date(2026, 10, 24), days=2)
    run_slots = 48 * (settings.warmup_days + settings.study_days)
    smart = smart_charging_inputs(
        settings, _units(), np.full((1, run_slots), 90.0), departure_margin_hours=1.0
    )
    departures = pd.to_datetime(smart.expected_departure_utc_ns[:, 0], utc=True)
    # 07:00 London minus one hour: 06:00 BST (05:00 UTC) before the change on
    # Sun 25 Oct, 06:00 GMT (06:00 UTC) from that day on.
    assert list(departures.strftime("%m-%d %H:%M")) == [
        "10-23 05:00",
        "10-24 05:00",
        "10-25 06:00",
        "10-26 06:00",
        "10-27 06:00",
    ]
    # Decision 0004 item 42: a weekend window's departure is expected on
    # Saturday and Sunday (10:00 - 1 h = 09:00, BST then GMT).
    weekend_units = _units().assign(runtime_weekend_departure_local_hour=[10])
    weekend = smart_charging_inputs(
        settings, weekend_units, np.full((1, run_slots), 90.0), departure_margin_hours=1.0
    )
    departures = pd.to_datetime(weekend.expected_departure_utc_ns[:, 0], utc=True)
    assert list(departures.strftime("%a %H:%M")) == [
        "Fri 05:00",
        "Sat 08:00",
        "Sun 09:00",
        "Mon 06:00",
        "Tue 06:00",
    ]
    with pytest.raises(ValueError, match="departure_margin_hours"):
        smart_charging_inputs(
            settings, _units(), np.full((1, run_slots), 90.0), departure_margin_hours=-1.0
        )
    # Prices cover the warm-up and the study (decision 0004 item 52).
    with pytest.raises(ValueError, match="one per world and run slot"):
        smart_charging_inputs(
            settings, _units(), np.full((1, run_slots - 1), 90.0), departure_margin_hours=1.0
        )


# --------------------------------------------------------------------------
# One EV through the kernel
# --------------------------------------------------------------------------


def _settings(*, start: date = _WINTER, days: int = 2, opening: float = 0.6) -> RunSettings:
    # One warm-up day: smart planning starts on the last warm-up night
    # (decision 0004 item 52), so the charger needs one.  The hand-built
    # sessions all sit in the study, so the warm-up only carries the stock.
    return RunSettings(
        start_local_date=start,
        warmup_days=1,
        study_days=days,
        vehicle_count=1,
        seed=1,
        evaluation_world_count=1,
        opening_soc_fraction=opening,
        reserve_soc_fraction=0.1,
        home_charge_efficiency=1.0,
        weather_efficiency_sensitivity_fraction_per_c=0.0,
    )


def _units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population_id": ["population:smart"],
            "unit_id": ["ev-001"],
            "cohort_id": ["average_uk"],
            "physical_capacity_kwh": [10.0],
            "efficiency_miles_per_battery_kwh": [1.0],
            "home_charger_limit_kw": [4.0],
            "preferred_target_soc_fraction": [0.8],
            # Typical home departure 07:00 London: expected 06:00 with the
            # default one-hour margin.
            "runtime_departure_local_hour": [7],
            "runtime_arrival_local_hour": [18],
        }
    )


def _inputs(settings: RunSettings) -> dict[str, np.ndarray]:
    # Day index d is the London date start - warm-up + d, so with one warm-up
    # day the study start date is index 1 and the morning after it index 2.
    shape = (1, settings.sampled_day_count, 1)
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


def _session(inputs: dict[str, np.ndarray], day: int, start: str, end: str) -> None:
    inputs["connection_session_accepted"][0, day, 0] = True
    inputs["connection_start_utc"][0, day, 0] = pd.Timestamp(start, tz="UTC")
    inputs["connection_end_utc"][0, day, 0] = pd.Timestamp(end, tz="UTC")


def _slot(settings: RunSettings, instant: str) -> int:
    """Study slot index of a UTC instant: slot 0 starts at London 12:00 on the start date."""

    start = (pd.Timestamp(settings.start_local_date) + pd.Timedelta(hours=12)).tz_localize(
        "Europe/London"
    )
    return int((pd.Timestamp(instant, tz="UTC") - start) // pd.Timedelta(minutes=30))


def _price_slot(settings: RunSettings, instant: str) -> int:
    """Run slot index (warm-up first) of a UTC instant, for the day-ahead price array."""

    return 48 * settings.warmup_days + _slot(settings, instant)


def _run_prices(settings: RunSettings, value: float = 100.0) -> np.ndarray:
    return np.full(48 * (settings.warmup_days + settings.study_days), value)


def _run(settings: RunSettings, inputs, prices: np.ndarray) -> dict[str, pd.DataFrame]:
    # One world, so its day-ahead prices are the one row of the (world, run
    # slot) array, warm-up slots first.
    smart = smart_charging_inputs(
        settings, _units(), prices[np.newaxis, :], departure_margin_hours=1.0
    )
    fleet = simulate_fleet_intervals(
        settings, _units(), inputs, public_top_up=_TOP_UP, smart_charging=smart
    )[0]
    assert fleet["conservation_residual_kwh"].abs().max() <= 1e-12
    return {
        path: frame.sort_values("interval_start_utc").reset_index(drop=True)
        for path, frame in fleet.groupby("path_id")
    }


def _imports(path: pd.DataFrame) -> dict[int, float]:
    values = path["home_grid_import_kwh"].to_numpy()
    return {int(slot): float(values[slot]) for slot in np.flatnonzero(values > 1e-12)}


def _overnight_prices(settings: RunSettings) -> np.ndarray:
    # Flat 100 GBP/MWh except two cheap night slots: 03:00 (10) and 04:00 (20).
    # Both are published at 13:00 on 7 Jan, before the 18:00 plug-in.
    prices = _run_prices(settings)
    prices[_price_slot(settings, "2025-01-08 03:00")] = 10.0
    prices[_price_slot(settings, "2025-01-08 04:00")] = 20.0
    return prices


def test_one_ev_charges_in_the_cheapest_slot_before_expected_departure() -> None:
    settings = _settings()
    inputs = _inputs(settings)
    _session(inputs, 1, "2025-01-07 18:00", "2025-01-08 07:00")
    paths = _run(settings, inputs, _overnight_prices(settings))

    # Need 2 kWh (6 -> 8 kWh): normal charges at plug-in, smart at 03:00.
    assert _imports(paths["normal"]) == {_slot(settings, "2025-01-07 18:00"): 2.0}
    assert _imports(paths["selected"]) == {_slot(settings, "2025-01-08 03:00"): 2.0}
    assert paths["selected"]["closing_battery_kwh"].iat[-1] == pytest.approx(8.0)
    assert paths["selected"]["early_departure_count"].sum() == 0.0


def test_the_last_chosen_slot_is_partial() -> None:
    settings = _settings(opening=0.5)
    inputs = _inputs(settings)
    _session(inputs, 1, "2025-01-07 18:00", "2025-01-08 07:00")
    paths = _run(settings, inputs, _overnight_prices(settings))

    # Need 3 kWh: 2 kWh at 03:00 (10), then the last 1 kWh at 04:00 (20).
    assert _imports(paths["selected"]) == pytest.approx(
        {_slot(settings, "2025-01-08 03:00"): 2.0, _slot(settings, "2025-01-08 04:00"): 1.0}
    )


def test_late_plug_in_uses_only_slots_after_it() -> None:
    settings = _settings()
    inputs = _inputs(settings)
    _session(inputs, 1, "2025-01-08 02:00", "2025-01-08 07:00")
    prices = _overnight_prices(settings)
    # A cheaper 01:00 slot is before the plug-in, so the charger cannot use it.
    prices[_price_slot(settings, "2025-01-08 01:00")] = 5.0
    paths = _run(settings, inputs, prices)

    assert _imports(paths["normal"]) == {_slot(settings, "2025-01-08 02:00"): 2.0}
    assert _imports(paths["selected"]) == {_slot(settings, "2025-01-08 03:00"): 2.0}


def test_window_too_short_charges_as_soon_as_possible_then_replans() -> None:
    settings = _settings(opening=0.2)
    inputs = _inputs(settings)
    # Plug in at 05:00 needing 6 kWh; the window to the 06:00 expected
    # departure holds only 2 slots (4 kWh), so both charge at full power.
    _session(inputs, 1, "2025-01-08 05:00", "2025-01-08 07:00")
    prices = _run_prices(settings)
    prices[_price_slot(settings, "2025-01-08 06:30")] = 50.0
    paths = _run(settings, inputs, prices)

    first, second = _slot(settings, "2025-01-08 05:00"), _slot(settings, "2025-01-08 05:30")
    assert _imports(paths["normal"]) == {first: 2.0, second: 2.0, first + 2: 2.0}
    # Still plugged in at 06:00, the charger replans the last 2 kWh for the
    # next expected departure, 06:00 on 9 Jan, inside the noon-to-noon study
    # (decision 0004 item 52).  9 Jan's prices are not published at 06:00 on
    # 8 Jan, so they are ranked on the expected shape (mean 06:30 price of
    # the published days, (100 + 50) / 2 = 75), and the published 50 at
    # 06:30 on 8 Jan is the cheapest: the EV charges there and leaves at
    # 07:00 with the plan complete.
    assert _imports(paths["selected"]) == {first: 2.0, second: 2.0, first + 3: 2.0}
    assert paths["selected"]["early_departure_count"].sum() == 0.0


def test_last_study_night_is_a_normal_smart_night() -> None:
    # Decision 0004 item 52 retired item 49: the study ends at London noon,
    # so the last night's session ends (07:00) before the horizon end and
    # plans like every other night, in its cheap overnight slot.  A cheaper
    # last study half-hour (11:30) is after the expected departure, so it is
    # not used and nothing is crammed before the cut.
    settings = _settings(opening=0.4)
    inputs = _inputs(settings)
    _session(inputs, 1, "2025-01-07 18:00", "2025-01-08 07:00")
    _session(inputs, 2, "2025-01-08 18:00", "2025-01-09 07:00")
    inputs["drives_today"][0, 2, 0] = True
    inputs["trip_departure_utc"][0, 2, 0] = pd.Timestamp("2025-01-08 08:00", tz="UTC")
    inputs["outbound_miles"][0, 2, 0] = 2.0
    inputs["return_miles"][0, 2, 0] = 2.0
    prices = _overnight_prices(settings)
    prices[_price_slot(settings, "2025-01-09 03:00")] = 10.0
    prices[_price_slot(settings, "2025-01-09 04:00")] = 20.0
    prices[_price_slot(settings, "2025-01-09 11:30")] = 1.0
    paths = _run(settings, inputs, prices)

    selected = _imports(paths["selected"])
    assert len(paths["selected"]) == 96
    assert _slot(settings, "2025-01-08 03:00") in selected
    assert _slot(settings, "2025-01-09 03:00") in selected
    # 4 kWh a night fills the 03:00 and 04:00 slots; nothing at plug-in.
    assert _slot(settings, "2025-01-08 18:00") not in selected
    assert sum(selected.values()) == pytest.approx(8.0)
    assert paths["selected"]["home_grid_import_kwh"].iat[-1] == 0.0
    assert paths["selected"]["early_departure_count"].sum() == 0.0


def test_early_departure_leaves_short_and_a_later_trip_tops_up_in_public() -> None:
    settings = _settings()
    inputs = _inputs(settings)
    # The plan puts 2 kWh at 03:00, but the EV unplugs at 03:00.
    _session(inputs, 1, "2025-01-07 18:00", "2025-01-08 03:00")
    # A 5.5 kWh trip at 08:00: 8 -> 2.5 kWh on the normal path, but the
    # selected path would fall from 6 to below the 1 kWh (10 %) threshold,
    # so it tops up in public first (decision 0004 items 32, 37).
    inputs["drives_today"][0, 2, 0] = True
    inputs["trip_departure_utc"][0, 2, 0] = pd.Timestamp("2025-01-08 08:00", tz="UTC")
    inputs["outbound_miles"][0, 2, 0] = 2.75
    inputs["return_miles"][0, 2, 0] = 2.75
    paths = _run(settings, inputs, _overnight_prices(settings))
    normal, selected = paths["normal"], paths["selected"]

    assert _imports(selected) == {}
    departure = _slot(settings, "2025-01-08 03:00")
    assert selected["early_departure_count"].to_numpy().nonzero()[0].tolist() == [departure]
    assert selected["early_departure_shortfall_kwh"].sum() == pytest.approx(2.0)
    assert normal["early_departure_count"].sum() == 0.0
    assert normal["public_grid_import_kwh"].sum() == 0.0
    assert selected["public_grid_import_kwh"].sum() > 0.0


def test_clock_change_night_plans_to_the_gmt_expected_departure() -> None:
    settings = _settings(start=date(2026, 10, 24))
    inputs = _inputs(settings)
    # 18:00 BST on Sat 24 Oct to 07:00 GMT on Sun 25 Oct.
    _session(inputs, 1, "2026-10-24 17:00", "2026-10-25 07:00")
    # Prices fall every slot, so the cheapest slot is the last one before
    # the expected departure: 06:00 GMT = 06:00 UTC, so 05:30-06:00 UTC.
    # Reading the departure at the BST offset would give 04:30 UTC instead.
    # 25 Oct's prices are published at 13:00 on 24 Oct, before the plug-in.
    prices = np.linspace(100.0, 5.0, len(_run_prices(settings)))
    paths = _run(settings, inputs, prices)

    assert _imports(paths["selected"]) == {_slot(settings, "2026-10-25 05:30"): 2.0}


def test_each_world_plans_on_its_own_day_ahead_prices() -> None:
    # Decision 0004 item 48: the same session in two worlds whose day-ahead
    # paths have their cheap slot at different times charges at different
    # times; neither world sees the other's prices.
    settings = replace(_settings(), evaluation_world_count=2)
    one_world = _inputs(_settings())
    _session(one_world, 1, "2025-01-07 18:00", "2025-01-08 07:00")
    inputs = {name: np.concatenate([values, values]) for name, values in one_world.items()}
    prices = np.full((2, len(_run_prices(settings))), 100.0)
    prices[0, _price_slot(settings, "2025-01-08 03:00")] = 10.0
    prices[1, _price_slot(settings, "2025-01-08 01:00")] = 10.0
    smart = smart_charging_inputs(settings, _units(), prices, departure_margin_hours=1.0)
    fleet = simulate_fleet_intervals(
        settings, _units(), inputs, public_top_up=_TOP_UP, smart_charging=smart
    )[0]
    selected = fleet.loc[fleet["path_id"].eq("selected")].sort_values(
        ["world_id", "interval_start_utc"]
    )
    imports = selected["home_grid_import_kwh"].to_numpy().reshape(2, -1)
    assert np.flatnonzero(imports[0] > 1e-12).tolist() == [_slot(settings, "2025-01-08 03:00")]
    assert np.flatnonzero(imports[1] > 1e-12).tolist() == [_slot(settings, "2025-01-08 01:00")]


def test_normal_path_is_unchanged_by_smart_charging() -> None:
    settings = _settings()
    inputs = _inputs(settings)
    _session(inputs, 1, "2025-01-07 18:00", "2025-01-08 07:00")
    with_smart = _run(settings, inputs, _overnight_prices(settings))["normal"]
    plain = simulate_fleet_intervals(settings, _units(), inputs, public_top_up=_TOP_UP)[0]
    pd.testing.assert_frame_equal(
        with_smart.drop(columns="path_id"),
        plain.loc[plain["path_id"].eq("normal")].drop(columns="path_id").reset_index(drop=True),
    )


def test_smart_charging_inputs_carry_no_evaluation_information() -> None:
    # The dataclass has exactly the decision-time fields: prices visible at
    # each publication epoch (plan B4), which epoch applies when, and the
    # expected departures; and the event inputs (trading contract v1 §2.4,
    # §9.6): zones, requests with their notice slots, the outage share, the
    # non-response uniforms and each session's non-response probability with
    # its maker and control-group parts (§4.5, §10.1e), and the blackout mask
    # and decision slots (§10.5b, §10.2d), none of which reveals a realised
    # session.
    assert set(SmartCharging.__dataclass_fields__) == {
        "visible_price_gbp_per_mwh",
        "price_epoch_by_slot",
        "expected_departure_utc_ns",
        "zone_index",
        "event_notice_slot",
        "event_adjustment_gbp_per_mwh",
        "outage_probability",
        "night_by_slot",
        "non_response_uniform",
        "non_response",
        "base_non_response",
        "control_group",
        "blackout",
        "decision_slot_by_night",
    }


# --------------------------------------------------------------------------
# Real runs: invariants, information cut-off and common random numbers
# --------------------------------------------------------------------------

_SMALL = {"vehicle_count": 60, "evaluation_world_count": 4}


def _real(model: str = "action", start: date = date(2026, 1, 12), **values: float):
    return run_forecast_from_assumptions(start, model=model, values=_SMALL | values)


@pytest.mark.parametrize("start", [date(2026, 1, 12), date(2026, 10, 22), date(2027, 3, 25)])
def test_real_run_conserves_energy_and_keeps_soc_physical(start: date) -> None:
    result = _real(start=start)
    world = result.fleet_world_intervals
    assert world["conservation_residual_kwh"].abs().max() <= 1e-9
    assert world["battery_soc_percent"].between(0.0, 100.0).all()
    # No EV takes more than its home charging power in a half-hour.
    power = result.units["home_charger_limit_kw"].sum() * 0.5
    assert world["home_import_kwh"].max() <= power + 1e-9
    selected = world.loc[world["path_id"].eq("selected")]
    assert selected["early_departure_count"].ge(0.0).all()
    # Stock flows slot to slot on each path.
    for _, group in world.groupby(["world_id", "path_id"]):
        np.testing.assert_allclose(
            group["opening_battery_kwh"].to_numpy()[1:],
            group["closing_battery_kwh"].to_numpy()[:-1],
            atol=1e-9,
        )


def test_smart_charging_pays_a_lower_average_forecast_price() -> None:
    result = _real()
    world = result.fleet_world_intervals.merge(
        result.forecast_prices[["world_id", "slot_index", "wholesale_forecast_gbp_per_mwh"]],
        on=["world_id", "slot_index"],
    )

    def forecast_price_paid(path: str) -> float:
        rows = world.loc[world["path_id"].eq(path)]
        weights = rows["home_import_kwh"]
        return float((weights * rows["wholesale_forecast_gbp_per_mwh"]).sum() / weights.sum())

    assert forecast_price_paid("selected") < forecast_price_paid("normal") - 10.0
    summary = result.smart_charging_summary.set_index("metric")
    assert summary.loc["average_price_change_gbp_per_mwh", "p90"] < 0.0
    assert 0.0 < summary.loc["moved_home_import_share", "p50"] <= 1.0


def test_plan_ignores_evaluation_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    # Decision-time cut-off: mutate every realised (intraday and imbalance)
    # price and the physics must not move.  Home import is valued at the
    # day-ahead price (plan B3), so the cost effect does not move either.
    base = _real()
    original = forecast_module.generate_market_price_paths

    def distorted(*args, **kwargs):
        prices = original(*args, **kwargs)
        realised = prices.realised.assign(
            evaluation_context_price_gbp_per_mwh=-3.0
            * prices.realised["evaluation_context_price_gbp_per_mwh"]
            + 500.0,
            imbalance_price_gbp_per_mwh=999.0,
        )
        return replace(prices, realised=realised)

    monkeypatch.setattr(forecast_module, "generate_market_price_paths", distorted)
    mutated = run_forecast_from_assumptions(date(2026, 1, 12), model="action", values=_SMALL)

    pd.testing.assert_frame_equal(base.fleet_world_intervals, mutated.fleet_world_intervals)
    pd.testing.assert_frame_equal(base.cost_effect, mutated.cost_effect)
    assert not base.evaluation_prices.equals(mutated.evaluation_prices)


def test_plan_follows_the_forecast() -> None:
    # The same test has teeth: moving the forecast does move the plan.
    base = _real()
    shifted = _real(day_ahead_sd_gbp_per_mwh=0.0)
    selected = base.fleet_world_intervals["path_id"].eq("selected")
    assert not np.array_equal(
        base.fleet_world_intervals.loc[selected, "home_import_kwh"],
        shifted.fleet_world_intervals.loc[selected, "home_import_kwh"],
    )


def test_flat_forecast_reproduces_the_normal_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # With every half-hour priced the same, cheapest-first with earlier ties
    # is exactly "as soon as possible", so both paths match.  The intraday
    # path is flattened too: free EVs re-plan on it when intraday dispatch
    # is on (the default), and a flat price gives no saving to re-plan for.
    original = forecast_module.generate_market_price_paths

    def flat(*args, **kwargs):
        prices = original(*args, **kwargs)
        return replace(
            prices,
            day_ahead=prices.day_ahead.assign(wholesale_forecast_gbp_per_mwh=80.0),
            intraday_path_gbp_per_mwh=np.full_like(prices.intraday_path_gbp_per_mwh, 80.0),
        )

    monkeypatch.setattr(forecast_module, "generate_market_price_paths", flat)
    result = _real()
    world = result.fleet_world_intervals
    normal = world.loc[world["path_id"].eq("normal")].reset_index(drop=True)
    selected = world.loc[world["path_id"].eq("selected")].reset_index(drop=True)
    for column in ("home_import_kwh", "public_import_kwh", "closing_battery_kwh"):
        np.testing.assert_allclose(selected[column], normal[column], atol=1e-9)
    # Early departures can still be counted here: a session too short to
    # reach the target leaves its plan unfinished even when the plan is "as
    # soon as possible".  So the count includes sessions no charging rule
    # could finish, and the paths above show they cost nothing extra.


def test_smart_charging_is_deterministic_and_keeps_common_random_numbers() -> None:
    first = _real()
    again = _real()
    pd.testing.assert_frame_equal(first.fleet_world_intervals, again.fleet_world_intervals)
    no_action = _real("no_action")
    normal = first.fleet_world_intervals["path_id"].eq("normal")
    # Decision 0007 follow-up: both runs also carry the optional "timed"
    # path by default, so both sides are filtered to "normal" explicitly
    # rather than relying on the no-action frame having only one path.
    no_action_normal = no_action.fleet_world_intervals["path_id"].eq("normal")
    columns = list(no_action.fleet_world_intervals.columns)
    pd.testing.assert_frame_equal(
        first.fleet_world_intervals.loc[normal, columns].reset_index(drop=True),
        no_action.fleet_world_intervals.loc[no_action_normal, columns].reset_index(drop=True),
    )
