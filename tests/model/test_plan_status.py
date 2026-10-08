"""The smart plan on record: non-response by maker, plan status and the decision snapshot.

Trading contract v1 §10.1e (lane J1b; decision 0004 item 59).  Hand-checkable
one-EV cases first (10 kWh battery, 4 kW charger = 2 kWh per half-hour,
100 % efficiency, 8 kWh target, one cheap night slot at 03:00), then small
SYNTHETIC real runs for the maker rules, battery conservation and one route
per EV-slot.  Every value is illustrative.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from axle_studio.model import assumptions, availability
from axle_studio.model.action import smart_charging_inputs
from axle_studio.model.assumptions import MANUFACTURER_IDS
from axle_studio.model.events import EVENT_COLUMNS
from axle_studio.model.forecast import simulate_forecast
from axle_studio.model.physics import (
    BASE_NON_RESPONSE,
    CONTROL_GROUP,
    CONTROL_OUTAGE_EVENT,
    MAKER_OUTAGE,
    PLAN_FOLLOWS,
    PublicTopUp,
    simulate_fleet_intervals,
    simulate_unit_intervals,
)
from axle_studio.model.settings import RunSettings
from axle_studio.model.summaries import build_study_slots

_TOP_UP = PublicTopUp(efficiency_fraction=0.9, threshold_soc_fraction=0.1, target_soc_fraction=0.8)
_WINTER = date(2025, 1, 7)  # GMT, so London clock time equals UTC
_LONDON_NOON = pd.Timestamp("2025-01-07 12:00", tz="UTC")


# --------------------------------------------------------------------------
# One EV, hand-checkable
# --------------------------------------------------------------------------


def _settings(opening: float = 0.6) -> RunSettings:
    return RunSettings(
        start_local_date=_WINTER,
        warmup_days=1,
        study_days=2,
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
            "population_id": ["population:plan-status"],
            "unit_id": ["ev-001"],
            "cohort_id": ["average_uk"],
            "physical_capacity_kwh": [10.0],
            "efficiency_miles_per_battery_kwh": [1.0],
            "home_charger_limit_kw": [4.0],
            "preferred_target_soc_fraction": [0.8],
            # Typical departure 07:00, expected 06:00 with a one-hour margin.
            "runtime_departure_local_hour": [7],
            "runtime_arrival_local_hour": [18],
        }
    )


def _inputs(settings: RunSettings, start: str, end: str) -> dict[str, np.ndarray]:
    """One home session on the study start date (sampled day 1), no trips."""

    shape = (1, settings.sampled_day_count, 1)
    timestamps = np.full(shape, pd.NaT, dtype=object)
    inputs = {
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
    inputs["connection_session_accepted"][0, 1, 0] = True
    inputs["connection_start_utc"][0, 1, 0] = pd.Timestamp(start, tz="UTC")
    inputs["connection_end_utc"][0, 1, 0] = pd.Timestamp(end, tz="UTC")
    return inputs


def _slot(instant: str) -> int:
    """Study slot of a UTC instant (slot 0 is London noon on the start date)."""

    return int((pd.Timestamp(instant, tz="UTC") - _LONDON_NOON) // pd.Timedelta(minutes=30))


def _prices(settings: RunSettings) -> np.ndarray:
    # Flat 100 GBP/MWh with one cheap slot at 03:00 (10), published at 13:00.
    prices = np.full((1, 48 * (settings.warmup_days + settings.study_days)), 100.0)
    prices[0, 48 + _slot("2025-01-08 03:00")] = 10.0
    return prices


def _per_ev(settings: RunSettings, inputs, **smart_inputs) -> dict[str, np.ndarray]:
    """Selected-path per-EV outputs of the one EV, (study slot,) each."""

    smart = smart_charging_inputs(
        settings, _units(), _prices(settings), departure_margin_hours=1.0, **smart_inputs
    )
    unit = simulate_unit_intervals(
        settings, _units(), inputs, public_top_up=_TOP_UP, smart_charging=smart
    )
    return {name: values[0, :, 0] for name, values in unit.items()}


def _nonzero(values: np.ndarray) -> dict[int, float]:
    return {int(slot): float(values[slot]) for slot in np.flatnonzero(np.abs(values) > 1e-12)}


_EVENING = ("2025-01-07 18:00", "2025-01-08 07:00")
_PLUG_IN, _CHEAP = _slot("2025-01-07 18:00"), _slot("2025-01-08 03:00")


@pytest.mark.parametrize(
    ("case", "expected_status"),
    [
        # control EV: always 4, whatever the uniform.
        ({"control": True, "rho": 1.0, "base": 0.05, "uniform": 0.99}, CONTROL_GROUP),
        # u < 1 - r: base non-response (also on an outage night).
        ({"rho": 0.5, "base": 0.5, "uniform": 0.3}, BASE_NON_RESPONSE),
        ({"rho": 1.0, "base": 0.5, "uniform": 0.3}, BASE_NON_RESPONSE),
        # 1 - r <= u, maker out tonight: code 2.
        ({"rho": 1.0, "base": 0.05, "uniform": 0.3}, MAKER_OUTAGE),
        # 1 - r <= u < size of a control outage covering the plug-in slot.
        ({"rho": 0.05, "base": 0.05, "uniform": 0.3, "outage": 0.5}, CONTROL_OUTAGE_EVENT),
        # Above every threshold: the session follows its plan.
        ({"rho": 0.05, "base": 0.05, "uniform": 0.9, "outage": 0.5}, PLAN_FOLLOWS),
    ],
)
def test_plan_status_codes_and_what_each_session_charges(case, expected_status) -> None:
    settings = _settings()
    outage = np.zeros((1, 48 * 3))
    outage[0, 48 + _PLUG_IN] = case.get("outage", 0.0)
    unit = _per_ev(
        settings,
        _inputs(settings, *_EVENING),
        outage_probability=outage,
        non_response_uniform=np.full((1, 2, 1), case["uniform"]),
        non_response=np.full((1, 2, 1), case["rho"]),
        base_non_response=np.array([case["base"]]),
        control_group=np.array([case.get("control", False)]),
    )
    # The first plan runs to the 06:00 expected departure; the EV is still
    # plugged in until 07:00, so a second plan is made at 06:00 on the same
    # night's uniform.  Maker and base non-response hold for the session;
    # the scripted outage covered only the plug-in slot, so the replan
    # follows (R2).
    status = unit["plan_status"]
    first_plan = np.arange(_PLUG_IN, _slot("2025-01-08 06:00"))
    replan = np.arange(_slot("2025-01-08 06:00"), _slot("2025-01-08 07:00"))
    assert (status[first_plan] == expected_status).all()
    after = PLAN_FOLLOWS if expected_status == CONTROL_OUTAGE_EVENT else expected_status
    assert (status[replan] == after).all()
    assert (np.delete(status, np.r_[first_plan, replan]) == PLAN_FOLLOWS).all()
    # Every session keeps its smart plan on record: 2 kWh in the cheap slot.
    assert _nonzero(unit["planned_home_import_kwh"]) == {_CHEAP: 2.0}
    # A follower charges its plan; any other session the normal rule (at
    # plug-in), not a "full power" plan.
    charged = _nonzero(unit["home_grid_import_kwh"])
    assert charged == ({_CHEAP: 2.0} if expected_status == PLAN_FOLLOWS else {_PLUG_IN: 2.0})


def test_remaining_need_falls_by_the_plans_kwh_and_restarts_at_a_replan() -> None:
    # Plug in at 05:00 needing 6 kWh: the window to the 06:00 expected
    # departure holds two slots (2 + 2 kWh).  At 06:00 the EV is still
    # plugged in, so a new plan is made for the next day's departure, with
    # the need left then (8 - 6 = 2 kWh).
    settings = _settings(opening=0.2)
    unit = _per_ev(settings, _inputs(settings, "2025-01-08 05:00", "2025-01-08 07:00"))
    first, second, replan = (_slot(f"2025-01-08 {t}") for t in ("05:00", "05:30", "06:00"))
    need = unit["plan_remaining_need_kwh"]
    assert need[first] == pytest.approx(6.0)
    assert need[second] == pytest.approx(4.0)
    assert need[replan] == pytest.approx(2.0)
    assert _nonzero(unit["planned_home_import_kwh"])[first] == pytest.approx(2.0)
    # Outside any plan the need is 0.
    assert need[:first].max() == 0.0


def test_a_replan_outside_the_control_outage_window_returns_code_3_to_0() -> None:
    # R2: the scripted outage covers the 05:00 plug-in only.  The first plan
    # is ignored (code 3); the replan at 06:00 is outside the window, so the
    # same uniform now follows (code 0).
    settings = _settings(opening=0.2)
    outage = np.zeros((1, 48 * 3))
    outage[0, 48 + _slot("2025-01-08 05:00")] = 0.5
    unit = _per_ev(
        settings,
        _inputs(settings, "2025-01-08 05:00", "2025-01-08 07:00"),
        outage_probability=outage,
        non_response_uniform=np.full((1, 2, 1), 0.3),
        non_response=np.full((1, 2, 1), 0.05),
        base_non_response=np.array([0.05]),
    )
    status = unit["plan_status"]
    assert status[_slot("2025-01-08 05:00")] == CONTROL_OUTAGE_EVENT
    assert status[_slot("2025-01-08 05:30")] == CONTROL_OUTAGE_EVENT
    assert status[_slot("2025-01-08 06:00")] == PLAN_FOLLOWS
    assert status[_slot("2025-01-08 06:30")] == PLAN_FOLLOWS


def _decision_slots(settings: RunSettings, local_time: str = "17:00") -> np.ndarray:
    slots = build_study_slots(settings.start_local_date, settings.study_days)
    return 48 * settings.warmup_days + availability.decision_slots(slots, local_time)


def test_decision_plan_is_the_plan_in_force_at_the_decision_slot() -> None:
    settings = _settings()
    decisions = _decision_slots(settings)
    # Plugged in at 16:00, before the 17:00 decision, never replanned: the
    # decision plan equals the plan in force from 17:00 on, 0 before it.
    early = _per_ev(
        settings,
        _inputs(settings, "2025-01-07 16:00", "2025-01-08 07:00"),
        decision_slot_by_night=decisions,
    )
    decision = _slot("2025-01-07 17:00")
    np.testing.assert_array_equal(
        early["decision_plan_kwh"][decision:], early["planned_home_import_kwh"][decision:]
    )
    assert (early["decision_plan_kwh"][:decision] == 0.0).all()
    assert _nonzero(early["decision_plan_kwh"]) == {_CHEAP: 2.0}
    # Plugged in at 18:00, after the decision: not in the 17:00 set.
    late = _per_ev(settings, _inputs(settings, *_EVENING), decision_slot_by_night=decisions)
    assert (late["decision_plan_kwh"] == 0.0).all()
    assert _nonzero(late["planned_home_import_kwh"]) == {_CHEAP: 2.0}
    # Without decision slots there is no snapshot.
    none = _per_ev(settings, _inputs(settings, "2025-01-07 16:00", "2025-01-08 07:00"))
    assert (none["decision_plan_kwh"] == 0.0).all()


def test_decision_plan_is_a_copy_the_session_end_does_not_change() -> None:
    # Plugged in at 16:00, gone at 19:00: the plan in force at 17:00 ends
    # with the session, but the decision snapshot is what the aggregator
    # held at 17:00, so it still shows that plan's 03:00 slot.
    settings = _settings()
    unit = _per_ev(
        settings,
        _inputs(settings, "2025-01-07 16:00", "2025-01-07 19:00"),
        decision_slot_by_night=_decision_slots(settings),
    )
    decision = _slot("2025-01-07 17:00")
    assert _nonzero(unit["decision_plan_kwh"]) == {_CHEAP: 2.0}
    assert (unit["planned_home_import_kwh"][_slot("2025-01-07 19:00") :] == 0.0).all()
    assert unit["decision_plan_kwh"][decision] == unit["planned_home_import_kwh"][decision]


# --------------------------------------------------------------------------
# Real runs (SYNTHETIC, small)
# --------------------------------------------------------------------------

_START = date(2026, 10, 12)  # a Monday, no clock change
_SMALL = {"vehicle_count": 40, "evaluation_world_count": 3}


def _makers(response: float | list[float], outage: float | list[float]) -> dict[str, float]:
    values = assumptions.shared_factor_inputs()
    responses = response if isinstance(response, list) else [response] * len(MANUFACTURER_IDS)
    outages = outage if isinstance(outage, list) else [outage] * len(MANUFACTURER_IDS)
    for maker_id, r, pi in zip(MANUFACTURER_IDS, responses, outages, strict=True):
        values[f"manufacturers.response_rate.{maker_id}"] = r
        values[f"manufacturers.outage_probability_per_night.{maker_id}"] = pi
    return values


def _simulate(shared=None, trading=None, events=None, small=_SMALL):
    settings = assumptions.run_settings(small, _START)
    inputs = assumptions.forecast_inputs(
        assumptions.resolve_values(small), warmup_days=7, study_days=7
    )
    if shared is not None:
        inputs["shared_factor_assumptions"] = shared
    if trading is not None:
        inputs["trading_assumptions"] = inputs["trading_assumptions"] | trading
    return simulate_forecast(settings, assumptions.cohort_fixture(small), events=events, **inputs)


def _units_run(simulated, smart=None) -> dict[str, np.ndarray]:
    return simulate_unit_intervals(
        simulated.settings,
        simulated.units,
        simulated.evaluation.inputs,
        public_top_up=simulated.public_top_up,
        effective_efficiency_miles_per_battery_kwh=(
            simulated.evaluation.effective_efficiency_miles_per_battery_kwh
        ),
        smart_charging=simulated.smart_charging if smart is None else smart,
    )


def _normal(simulated) -> dict[str, np.ndarray]:
    return simulate_unit_intervals(
        simulated.settings,
        simulated.units,
        simulated.evaluation.inputs,
        public_top_up=simulated.public_top_up,
        effective_efficiency_miles_per_battery_kwh=(
            simulated.evaluation.effective_efficiency_miles_per_battery_kwh
        ),
    )


def test_full_response_and_no_outage_is_the_kernel_without_non_response() -> None:
    simulated = _simulate(shared=_makers(1.0, 0.0))
    with_rule = _units_run(simulated)
    without = replace(
        simulated.smart_charging,
        non_response_uniform=None,
        non_response=None,
        base_non_response=None,
        control_group=None,
    )
    plain = _units_run(simulated, without)
    for name in with_rule:
        np.testing.assert_array_equal(with_rule[name], plain[name], err_msg=name)
    assert (with_rule["plan_status"] == PLAN_FOLLOWS).all()


def test_zero_response_makes_the_selected_path_the_normal_path() -> None:
    simulated = _simulate(shared=_makers(0.0, 0.0))
    selected, normal = _units_run(simulated), _normal(simulated)
    np.testing.assert_allclose(
        selected["home_grid_import_kwh"], normal["home_grid_import_kwh"], atol=1e-12
    )
    # Every session keeps a plan on record and it is a smart plan, not the
    # normal path's charging.
    assert selected["planned_home_import_kwh"].sum() > 0.0
    assert not np.allclose(selected["planned_home_import_kwh"], normal["home_grid_import_kwh"])
    in_plan = selected["plan_remaining_need_kwh"] > 0.0
    assert (selected["plan_status"][in_plan] == BASE_NON_RESPONSE).all()


def test_one_maker_always_out_charges_by_the_normal_rule_and_leaves_the_rest() -> None:
    base = _simulate(shared=_makers(0.95, 0.0))
    out = _simulate(shared=_makers(0.95, [0.0, 1.0, 0.0, 0.0]))
    # Every draw is the same: the outage probability only picks outages.
    np.testing.assert_array_equal(
        base.smart_charging.non_response_uniform, out.smart_charging.non_response_uniform
    )
    maker_b = out.units["manufacturer_index"].to_numpy() == 1
    assert out.manufacturer_outage[:, :, 1].all()
    selected_base, selected_out = _units_run(base), _units_run(out)
    normal_base, normal_out = _normal(base), _normal(out)
    for name in normal_base:
        np.testing.assert_array_equal(normal_base[name], normal_out[name], err_msg=name)
    for name in ("home_grid_import_kwh", "planned_home_import_kwh", "plan_status"):
        np.testing.assert_array_equal(
            selected_base[name][..., ~maker_b], selected_out[name][..., ~maker_b], err_msg=name
        )
    np.testing.assert_allclose(
        selected_out["home_grid_import_kwh"][..., maker_b],
        normal_out["home_grid_import_kwh"][..., maker_b],
        atol=1e-12,
    )
    status = selected_out["plan_status"][..., maker_b]
    in_plan = selected_out["plan_remaining_need_kwh"][..., maker_b] > 0.0
    assert set(np.unique(status[in_plan])) <= {BASE_NON_RESPONSE, MAKER_OUTAGE}
    assert (status[in_plan] == MAKER_OUTAGE).any()


def test_control_evs_carry_code_4_and_followers_charge_their_plan() -> None:
    simulated = _simulate(trading={"trading.control_group": 1.0})
    unit = _units_run(simulated)
    control = simulated.units["control_group"].to_numpy()
    assert control.any()
    in_plan = unit["plan_remaining_need_kwh"] > 0.0
    assert (unit["plan_status"][..., control][in_plan[..., control]] == CONTROL_GROUP).all()
    assert (unit["plan_status"][..., ~control] != CONTROL_GROUP).all()
    # A code-0 EV-slot imports min(plan, normal limit): the plan, cut only
    # where the stock is already near its target (after a public top-up).
    capacity = simulated.units["physical_capacity_kwh"].to_numpy()
    target = capacity * simulated.units["preferred_target_soc_fraction"].to_numpy()
    power = simulated.units["home_charger_limit_kw"].to_numpy()
    efficiency = simulated.settings.home_charge_efficiency
    limit = np.minimum(
        power * 0.5, np.maximum(target - unit["opening_battery_kwh"], 0.0) / efficiency
    )
    follows = unit["plan_status"] == PLAN_FOLLOWS
    expected = np.minimum(unit["planned_home_import_kwh"], limit)
    np.testing.assert_allclose(
        unit["home_grid_import_kwh"][follows], expected[follows], atol=1e-9, rtol=0.0
    )


def test_conservation_one_route_per_slot_and_physical_soc_on_a_real_run() -> None:
    simulated = _simulate()
    unit = _units_run(simulated)
    efficiency = simulated.settings.home_charge_efficiency
    home, public = unit["home_grid_import_kwh"], unit["public_grid_import_kwh"]
    # One dispatchable route per EV-slot: home charging needs the whole slot
    # plugged in at home, public top-ups happen on trips, so never both.
    assert not ((home > 0.0) & (public > 0.0)).any()
    # A home-charging slot has no travel: stock rises by exactly the energy
    # added (battery-energy conservation; the kernel also checks every slot).
    charging = home > 0.0
    added = unit["closing_battery_kwh"] - unit["opening_battery_kwh"]
    np.testing.assert_allclose(added[charging], home[charging] * efficiency, atol=1e-9)
    capacity = simulated.units["physical_capacity_kwh"].to_numpy()
    assert (unit["closing_battery_kwh"] >= -1e-12).all()
    assert (unit["closing_battery_kwh"] <= capacity + 1e-12).all()
    fleet = simulate_fleet_intervals(
        simulated.settings,
        simulated.units,
        simulated.evaluation.inputs,
        public_top_up=simulated.public_top_up,
        evaluation_effective_efficiency_miles_per_battery_kwh=(
            simulated.evaluation.effective_efficiency_miles_per_battery_kwh
        ),
        smart_charging=simulated.smart_charging,
    )[0]
    assert fleet["conservation_residual_kwh"].abs().max() <= 1e-12


def _control_outage(night: int, start: str) -> pd.DataFrame:
    row = {
        "event_id": "late_outage",
        "event_type": "control_outage",
        "enabled": True,
        "night_index": night,
        "start_local_time": start,
        "duration_minutes": 180,
        "size": 1.0,
        "size_unit": "fraction",
        "notice": "short",
        "notice_minutes": 0,
        "scope": "national",
        "payment_gbp_per_mwh": np.nan,
    }
    return pd.DataFrame([row], columns=list(EVENT_COLUMNS))


def test_a_short_notice_event_after_17_00_leaves_the_decision_view_unchanged() -> None:
    # R1: a control outage announced at 18:00 (no notice) changes the plans
    # made in its window, so the selected path moves, but the aggregator's
    # 17:00 view (the decision plan, the plan's need at s_n, who is plugged
    # in and their status then) and so the known part of §10.2d are unchanged.
    # Later nights start from batteries the event touched, so the check is
    # on the event's own night and the nights before it.
    base = _simulate()
    event = _simulate(events=_control_outage(night=2, start="18:00"))
    a, b = _units_run(base), _units_run(event)
    night = build_study_slots(_START)["night_index"].to_numpy()
    tonight = night <= 2
    assert not np.array_equal(
        a["home_grid_import_kwh"][:, tonight], b["home_grid_import_kwh"][:, tonight]
    )
    np.testing.assert_array_equal(
        a["decision_plan_kwh"][:, tonight], b["decision_plan_kwh"][:, tonight]
    )
    decisions = (base.smart_charging.decision_slot_by_night - 48 * base.settings.warmup_days)[:3]
    for name in ("plan_remaining_need_kwh", "plan_status", "planned_home_import_kwh"):
        np.testing.assert_array_equal(
            a[name][:, decisions], b[name][:, decisions], err_msg=f"{name} at s_n"
        )
    assert a["decision_plan_kwh"][:, night == 2].sum() > 0.0
