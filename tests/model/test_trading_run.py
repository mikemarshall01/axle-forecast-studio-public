"""The trading overlay and Supplier frames on real (small) runs (trading contract v1 §5.10, §9.11).

Each run is a SYNTHETIC illustrative Monte Carlo at a small fleet, seeded,
so comparisons are on matched futures.  Hand-checkable function-level cases
come first, then real runs.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from functools import cache

import numpy as np
import pandas as pd
import pytest
from fixtures.result_fixture import validate_result_v2

from axle_studio.model import assumptions, summaries
from axle_studio.model.forecast import (
    ForecastResult,
    SimulatedForecast,
    run_forecast,
    run_forecast_from_assumptions,
    simulate_forecast,
)
from axle_studio.model.physics import simulate_fleet_intervals, simulate_unit_intervals

_START = date(2026, 10, 12)  # a Monday, winter prices, no clock change
_SMALL = {"vehicle_count": 40, "evaluation_world_count": 6}


# --------------------------------------------------------------------------
# Supplier and trader frames on hand-made arrays
# --------------------------------------------------------------------------


def _cost_curve(unplug_slot: int, departure_slot: int, risk: float = 24.0) -> np.ndarray:
    """One EV charging 7 kW in slots 0-1, six slots, SYNTHETIC visible prices."""

    imports = np.zeros((1, 6, 1))
    imports[0, :2, 0] = 3.5
    connected = np.zeros((1, 6, 1), dtype=bool)
    connected[0, :unplug_slot, 0] = True
    hours = ((departure_slot - np.arange(6)) / 2.0)[:, None]
    prices = np.array([[100.0, 90.0, 40.0, 60.0, 10.0, 10.0]])
    return summaries.flex_cost_curve_chunk(imports, connected, prices, hours, np.full(6, 6), risk)


def test_cost_curve_prices_each_charging_slot_against_its_cheapest_free_slot() -> None:
    # Block ends at slot 2; candidates run to the expected departure (slot
    # 4): slots 2 and 3.  Cost at t = 0: 40 - 100 + 24 = -36, at t = 1:
    # 40 - 90 + 24 = -26 GBP/MWh.  The first grid prices at or above them
    # are -30 and -20.
    curve = _cost_curve(unplug_slot=6, departure_slot=4)
    thresholds = summaries.COST_CURVE_THRESHOLDS
    available = curve[0, :, : len(thresholds)]
    assert available[0, thresholds == -40.0] == 0.0
    assert available[0, thresholds == -30.0] == 7.0
    assert available[1, thresholds == -30.0] == 0.0
    assert available[1, thresholds == -20.0] == 7.0
    np.testing.assert_array_equal(curve[0, :2, -2], [7.0, 7.0])  # movable
    np.testing.assert_array_equal(curve[0, :, -1], [7.0, 7.0, 0.0, 0.0, 0.0, 0.0])  # charging


def test_cost_curve_stops_at_the_realised_unplug_and_drops_evs_with_no_free_slot() -> None:
    # Lead review B2: unplugged at slot 3, so slot 3 (price 60) no longer
    # counts; the cost is unchanged here because slot 2 is cheaper, but a
    # departure at slot 2 leaves no candidate at all.
    early = _cost_curve(unplug_slot=3, departure_slot=4)
    assert early[0, 0, -2] == 7.0
    none = _cost_curve(unplug_slot=6, departure_slot=2)
    assert np.all(none[0, :, :-1] == 0.0)
    assert none[0, 0, -1] == 7.0


def test_raising_the_risk_charge_by_ten_shifts_the_curve_one_step_right() -> None:
    base = _cost_curve(6, 4, risk=24.0)[..., :-2]
    raised = _cost_curve(6, 4, risk=34.0)[..., :-2]
    np.testing.assert_array_equal(raised[..., 1:], base[..., :-1])


def test_shape_premium_is_zero_for_a_flat_price() -> None:
    load = np.random.default_rng(2).uniform(0.0, 5.0, (3, 336))
    frame = summaries.shape_premium_summary(load, load[::-1], np.full((3, 336), 80.0))
    premium = frame.loc[frame["metric"].eq("shape_premium_gbp_per_mwh"), ["mean", "p10", "p90"]]
    np.testing.assert_allclose(premium.to_numpy(), 0.0, atol=1e-9)


def test_open_position_profile_adds_back_to_the_weekly_imbalance_volume() -> None:
    study = summaries.build_study_slots(date(2026, 10, 22), 7)  # an autumn change
    rng = np.random.default_rng(4)
    settled = rng.uniform(0.0, 3.0, (1, 336))
    positions = {s: rng.uniform(0.0, 3.0, (1, 336)) for s in ("da_only", "full")}
    positions["perfect_foresight"] = settled
    profile = summaries.open_position_profile(settled, positions, study)
    counts = study["local_half_hour"].value_counts()
    for strategy in ("da_only", "full"):
        rows = profile.loc[
            profile["strategy"].eq(strategy) & profile["metric"].eq("open_position_kw")
        ]
        index = rows["local_half_hour"].map(lambda t: int(t[:2]) * 2 + int(t[3:]) // 30)
        total = (rows["mean"].to_numpy() * counts.reindex(index).to_numpy() * 0.5).sum()
        assert total == pytest.approx((settled - positions[strategy]).sum())
    pf = profile.loc[profile["strategy"].eq("perfect_foresight"), "mean"]
    assert (pf == 0.0).all()


def test_household_saving_reconciles_with_the_cost_effect_columns() -> None:
    rng = np.random.default_rng(8)
    per_ev = {
        "home_cost_gbp": rng.normal(-1.0, 1.0, (2, 5)),
        "public_import_kwh": rng.normal(0.0, 1.0, (2, 5)),
        "unserved_kwh": np.zeros((2, 5)),
        "closing_kwh": rng.normal(0.0, 2.0, (2, 5)),
    }
    unrecovered = np.maximum(-per_ev["closing_kwh"].sum(axis=1), 0.0) * 0.79
    cost = pd.DataFrame(
        {
            "world_id": [0, 1],
            "illustrative_unrecovered_energy_value_gbp": unrecovered,
            "illustrative_unserved_travel_value_gbp": [0.0, 0.0],
        }
    )
    saving = summaries.household_saving_gbp(per_ev, cost, 0.79)
    world_total = (
        per_ev["home_cost_gbp"].sum(axis=1)
        + per_ev["public_import_kwh"].sum(axis=1) * 0.79
        + unrecovered
    )
    np.testing.assert_allclose(-saving.sum(axis=1), world_total)


def test_allocation_weights_sum_to_one_and_give_control_evs_nothing() -> None:
    reduction = np.zeros((1, 4, 2))
    reduction[0, :, 0] = [1.0, 3.0, 0.0, 5.0]  # night 1 settles nothing
    treated = np.array([True, True, True, False])
    weights = summaries.allocation_weights(reduction, treated)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(weights[0, :, 0], [0.25, 0.75, 0.0, 0.0])
    np.testing.assert_allclose(weights[0, :, 1], [1 / 3, 1 / 3, 1 / 3, 0.0])


def test_supplier_positions_count_settlement_periods_across_the_autumn_change() -> None:
    study = summaries.build_study_slots(date(2026, 10, 22), 7)
    zeros = np.zeros((2, 336))
    frame = summaries.supplier_positions(
        study,
        day_ahead_position=zeros,
        full_position=zeros,
        baseline=zeros,
        metered=zeros,
        settled=zeros,
        day_ahead=np.full((2, 336), 50.0),
        price_curve_source="synthetic",
    )
    periods = frame.groupby("settlement_date")["settlement_period"].max()
    assert periods["2026-10-25"] == 50
    assert periods["2026-10-24"] == 48


# --------------------------------------------------------------------------
# Real runs
# --------------------------------------------------------------------------


@cache
def _run(**edits: float) -> ForecastResult:
    return run_forecast_from_assumptions(_START, values=_SMALL)


def _inputs() -> dict:
    return assumptions.forecast_inputs(
        assumptions.resolve_values(_SMALL), warmup_days=7, study_days=7
    )


def _settings():
    return assumptions.run_settings(_SMALL, _START)


def _direct(
    trading: dict[str, float] | None = None,
    response: tuple[float, ...] | None = None,
    **price_edits,
) -> ForecastResult:
    inputs = _inputs()
    inputs["trading_assumptions"] = inputs["trading_assumptions"] | (trading or {})
    if response is not None:
        inputs["shared_factor_assumptions"] = _makers(response)
    inputs["price_assumptions"] = inputs["price_assumptions"] | price_edits
    return run_forecast(_settings(), assumptions.cohort_fixture(_SMALL), **inputs)


def _kernel_frames(result: ForecastResult) -> list[pd.DataFrame]:
    return [result.fleet_world_intervals, result.forecast_prices, result.evaluation_prices]


def test_default_run_passes_the_validator_with_full_baseline_history() -> None:
    result = _run()
    validate_result_v2(result)
    checks = result.trading_checks.set_index("check_id")["value"]
    assert checks["baseline_missing_nights"] == 0.0
    assert checks["baseline_class_fallback"] == 0.0
    assert result.price_curve_source == "synthetic"
    assert result.control_group_summary is None


_LEDGER_ONLY_EDITS = {
    "trading.customer_revenue_share": 0.2,
    "trading.supplier_compensation_gbp_per_mwh": 10.0,
    "trading.unmet_charge_penalty_gbp_per_kwh": 2.0,
    "trading.baseline_trap": 1.0,
    "supplier.early_departure_risk_charge_gbp_per_mwh": 50.0,
}
# With intraday dispatch on these also move the dispatched path: the
# commitment share sets the locked EVs, the half spread the re-plan bar
# (intraday-dispatch-v1 §2, §4.1).
_DISPATCH_EDITS = {
    "trading.half_spread_gbp_per_mwh": 5.0,
    "trading.day_ahead_commitment_share": 0.5,
}


def test_editing_a_trading_assumption_changes_no_kernel_frame_and_no_price() -> None:
    off = {"trading.intraday_dispatch": 0.0}
    base = _direct(off)
    edited = _direct(off | _LEDGER_ONLY_EDITS | _DISPATCH_EDITS)
    for a, b in zip(_kernel_frames(base), _kernel_frames(edited), strict=True):
        pd.testing.assert_frame_equal(a, b)
    assert not base.trading_week_world.equals(edited.trading_week_world)


def test_with_dispatch_only_the_dispatch_records_move_the_selected_path() -> None:
    base = _direct()
    ledger_only = _direct(_LEDGER_ONLY_EDITS)
    for a, b in zip(_kernel_frames(base), _kernel_frames(ledger_only), strict=True):
        pd.testing.assert_frame_equal(a, b)
    # The commitment share and the spread move the dispatched fleet, never a
    # price or the unmanaged path (no draw).
    dispatch = _direct(_DISPATCH_EDITS)
    pd.testing.assert_frame_equal(base.forecast_prices, dispatch.forecast_prices)
    pd.testing.assert_frame_equal(base.evaluation_prices, dispatch.evaluation_prices)
    fleet = base.fleet_world_intervals
    moved = dispatch.fleet_world_intervals
    normal = fleet["path_id"].eq("normal")
    pd.testing.assert_frame_equal(fleet.loc[normal], moved.loc[normal])
    assert not fleet.loc[~normal].equals(moved.loc[~normal])


def test_trap_mode_leaves_nights_0_and_1_unchanged_and_changes_later_nights() -> None:
    base = _direct()
    trap = _direct({"trading.baseline_trap": 1.0})
    key = ["world_id", "slot_index"]
    b0 = base.deviation_world_slot.set_index(key)
    b1 = trap.deviation_world_slot.set_index(key)
    early = b0["night_index"] <= 1
    np.testing.assert_array_equal(b0.loc[early, "baseline_kwh"], b1.loc[early, "baseline_kwh"])
    assert not np.array_equal(b0.loc[~early, "baseline_kwh"], b1.loc[~early, "baseline_kwh"])


def test_in_day_adjustment_is_nearly_inert_at_the_default_run_shape() -> None:
    # §8 Q3: fleet import at 12:00-13:30 is close to 0, so |a_n| is small
    # against the night's peak baseline.  The day-ahead rows of the sampled
    # worlds carry B0, the deviation frame B.
    result = _run()
    rows = result.position_updates.query("stage == 'day_ahead'")
    merged = rows.merge(result.deviation_world_slot, on=["world_id", "slot_index"])
    shift = (merged["baseline_kwh"] - merged["baseline_known_kwh"]).abs()
    peak = merged.groupby(["world_id", "night_index_x"])["baseline_known_kwh"].transform("max")
    assert (shift <= 0.01 * peak).all()


def test_equal_prices_everywhere_give_equal_net_and_full_capture() -> None:
    # §5.10 and §9.11: no intraday noise, no imbalance premium or tails, no
    # spread and every shock known, so intraday = SIP = day-ahead.
    quiet = {
        "intraday_hourly_sd_gbp_per_mwh": 0.0,
        "imbalance_long_premium_gbp_per_mwh": 0.0,
        "imbalance_short_premium_gbp_per_mwh": 0.0,
        "imbalance_tail_scale_gbp_per_mwh": 0.0,
        "mild_shock_known_share": 1.0,
        "big_shock_known_share": 1.0,
    }
    for share in (0.8, 1.0):
        result = _direct(
            {"trading.half_spread_gbp_per_mwh": 0.0, "trading.day_ahead_commitment_share": share},
            **quiet,
        )
        net = result.trading_week_world.pivot(
            index="world_id", columns="strategy", values="net_gbp"
        )
        np.testing.assert_allclose(net["full"], net["perfect_foresight"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(net["da_only"], net["perfect_foresight"], rtol=1e-9, atol=1e-9)
        kpis = result.trading_kpis.set_index(["strategy", "metric"])["mean"]
        assert kpis["full", "value_of_intraday_gbp_per_week"] == pytest.approx(0.0, abs=1e-9)
        assert kpis["full", "cost_of_uncertainty_gbp_per_week"] == pytest.approx(0.0, abs=1e-9)
        assert kpis["da_only", "capture_rate"] == pytest.approx(1.0)


def test_capture_rate_is_one_and_cost_of_uncertainty_zero_for_perfect_foresight() -> None:
    kpis = _run().trading_kpis.set_index(["strategy", "metric"])
    assert kpis.loc[("perfect_foresight", "capture_rate"), "mean"] == pytest.approx(1.0)
    cost = kpis.loc[("perfect_foresight", "cost_of_uncertainty_gbp_per_week"), ["mean", "p90"]]
    np.testing.assert_allclose(cost.to_numpy(dtype=float), 0.0)
    for metric in ("imbalance_gbp_per_mwh_traded", "flex_margin_gbp_per_mw_year", "firmness"):
        assert np.isnan(kpis.loc[("full", metric), "cvar5"])


# --------------------------------------------------------------------------
# Non-response by maker (§4.5, §10.1e) and the control group (§9.5)
# --------------------------------------------------------------------------

_MAKERS = assumptions.MANUFACTURER_IDS


def _makers(response: tuple[float, ...]) -> dict[str, float]:
    """Shared-factor inputs with these maker response rates and no maker outages."""

    values = assumptions.shared_factor_inputs()
    for maker_id, rate in zip(_MAKERS, response, strict=True):
        values[f"manufacturers.response_rate.{maker_id}"] = rate
        values[f"manufacturers.outage_probability_per_night.{maker_id}"] = 0.0
    return values


def test_non_response_uniforms_are_drawn_whatever_the_probabilities() -> None:
    zero = _direct(response=(1.0,) * 4)
    half = _direct(response=(0.5,) * 4)
    normal = [r.fleet_world_intervals.query("path_id == 'normal'") for r in (zero, half)]
    pd.testing.assert_frame_equal(normal[0], normal[1])
    pd.testing.assert_frame_equal(zero.forecast_prices, half.forecast_prices)
    selected = [r.fleet_world_intervals.query("path_id == 'selected'") for r in (zero, half)]
    assert not selected[0]["home_import_kwh"].equals(selected[1]["home_import_kwh"])


def test_non_response_one_makes_the_selected_path_the_normal_path() -> None:
    result = _direct(response=(0.0,) * 4)
    world = result.fleet_world_intervals
    np.testing.assert_allclose(
        world.query("path_id == 'selected'")["home_import_kwh"].to_numpy(),
        world.query("path_id == 'normal'")["home_import_kwh"].to_numpy(),
        atol=1e-9,
    )
    assert np.allclose(result.deviation_world_slot["true_reduction_kwh"], 0.0, atol=1e-9)
    assert np.allclose(result.trading_ledger_world["flexibility_mwh"], 0.0, atol=1e-12)


def test_non_response_of_one_maker_changes_only_that_makers_evs() -> None:
    result = _direct(response=(0.0, 1.0, 1.0, 1.0))
    state = result.replay_state
    worlds, evs = np.arange(result.world_count), np.arange(result.vehicle_count)

    def imports(selected: bool) -> np.ndarray:
        piece = summaries.kernel_slice(
            state, worlds, evs, path_id="selected" if selected else "normal"
        )
        return simulate_unit_intervals(
            piece.settings,
            piece.units,
            piece.inputs,
            public_top_up=piece.public_top_up,
            effective_efficiency_miles_per_battery_kwh=piece.effective_efficiency,
            smart_charging=piece.smart_charging,
        )["home_grid_import_kwh"]

    normal, selected = imports(False), imports(True)
    maker_a = state.units["manufacturer_index"].to_numpy() == 0
    np.testing.assert_allclose(selected[..., maker_a], normal[..., maker_a], atol=1e-12)
    assert not np.allclose(selected[..., ~maker_a], normal[..., ~maker_a])


def test_non_response_zero_is_the_kernel_without_non_response() -> None:
    simulated = simulate_forecast(
        _settings(),
        assumptions.cohort_fixture(_SMALL),
        **(_inputs() | {"shared_factor_assumptions": _makers((1.0,) * 4)}),
    )
    smart = simulated.smart_charging
    without = replace(
        smart,
        non_response_uniform=None,
        non_response=None,
        base_non_response=None,
        control_group=None,
    )
    kwargs = {
        "public_top_up": simulated.public_top_up,
        "evaluation_effective_efficiency_miles_per_battery_kwh": (
            simulated.evaluation.effective_efficiency_miles_per_battery_kwh
        ),
    }
    a = simulate_fleet_intervals(
        simulated.settings,
        simulated.units,
        simulated.evaluation.inputs,
        smart_charging=smart,
        **kwargs,
    )[0]
    b = simulate_fleet_intervals(
        simulated.settings,
        simulated.units,
        simulated.evaluation.inputs,
        smart_charging=without,
        **kwargs,
    )[0]
    pd.testing.assert_frame_equal(a, b)


def test_control_group_changes_no_draw_and_its_evs_are_never_flexed() -> None:
    off = _direct()
    on = _direct({"trading.control_group": 1.0, "trading.control_group_share": 0.25})
    pd.testing.assert_frame_equal(off.forecast_prices, on.forecast_prices)
    pd.testing.assert_frame_equal(
        off.fleet_world_intervals.query("path_id == 'normal'"),
        on.fleet_world_intervals.query("path_id == 'normal'"),
    )
    control = on.units["control_group"].to_numpy(dtype=bool)
    assert control.sum() == 10
    # Stratified: each cohort holds its largest-remainder share.
    counts = on.units.loc[control, "cohort_id"].value_counts()
    fleet = on.units["cohort_id"].value_counts()
    assert all(abs(counts.get(c, 0) - 0.25 * n) < 1.0 for c, n in fleet.items())
    state = on.replay_state
    evs = np.flatnonzero(control)
    worlds = np.arange(on.world_count)
    normal = summaries.kernel_slice(state, worlds, evs, path_id="normal")
    selected = summaries.kernel_slice(state, worlds, evs, path_id="selected")

    def run(piece):
        return simulate_unit_intervals(
            piece.settings,
            piece.units,
            piece.inputs,
            public_top_up=piece.public_top_up,
            effective_efficiency_miles_per_battery_kwh=piece.effective_efficiency,
            smart_charging=piece.smart_charging,
        )["home_grid_import_kwh"]

    # A control EV ignores its plan and charges by the normal rule (§10.1e).
    np.testing.assert_allclose(run(normal), run(selected), rtol=0.0, atol=1e-12)
    summary = on.control_group_summary
    assert summary is not None and set(summary["metric"]) == {
        "estimated_bias_kw_per_ev",
        "true_bias_kw_per_ev",
        "estimation_error_kw_per_ev",
    }
    household = on.household_value_summary.query("group_id == 'fleet'")
    assert (household["ev_count"] == 30).all()
    # A direct run carries no records; attach the defaults (the ledger terms
    # the validator recomputes with are the defaults here).
    validate_result_v2(replace(on, assumptions=assumptions.result_assumptions(_SMALL)))


# --------------------------------------------------------------------------
# Kernel outputs for the trader: plan books use only what is known (§1.4, §5.10)
# --------------------------------------------------------------------------


def _book(simulated: SimulatedForecast, inputs: dict[str, np.ndarray]) -> np.ndarray:
    from axle_studio.model.forecast import _trading_kernel_outputs

    study = summaries.build_study_slots(_START, 7)
    output, _ = _trading_kernel_outputs(
        simulated.settings,
        study,
        np.zeros(simulated.settings.vehicle_count, dtype=bool),
        simulated.trading_assumptions["day_ahead_publication_local_hour"],
    )
    simulate_fleet_intervals(
        simulated.settings,
        simulated.units,
        inputs,
        public_top_up=simulated.public_top_up,
        evaluation_effective_efficiency_miles_per_battery_kwh=(
            simulated.evaluation.effective_efficiency_miles_per_battery_kwh
        ),
        smart_charging=simulated.smart_charging,
        trading_output=output,
    )
    return output["selected"]["book_kwh"]


def test_plan_books_ignore_plug_ins_and_unplugs_after_the_decision() -> None:
    # Leakage (§5.10): later plug-ins (30 minutes later, still before the
    # session's end) and later unplugs on non-driving days, all after the
    # decision instant, leave every book up to the decision unchanged.
    simulated = simulate_forecast(_settings(), assumptions.cohort_fixture(_SMALL), **_inputs())
    inputs = simulated.evaluation.inputs
    study = summaries.build_study_slots(_START, 7)
    decision_slot = int(np.flatnonzero(study["local_time_label"].eq("20:00").to_numpy())[1])
    tau = study["interval_start_utc"].iat[decision_slot].tz_localize(None).to_datetime64()
    accepted = inputs["connection_session_accepted"]
    later = dict(inputs)
    start = inputs["connection_start_utc"].copy()
    end = inputs["connection_end_utc"].copy()
    half_hour = np.timedelta64(30, "m")
    plug = accepted & (start > tau) & (end - start > 2 * half_hour)
    start[plug] = start[plug] + half_hour
    unplug = accepted & ~inputs["drives_today"] & (end > tau)
    end[unplug] = end[unplug] + half_hour
    later["connection_start_utc"], later["connection_end_utc"] = start, end
    assert plug.any() and unplug.any()
    base, moved = _book(simulated, inputs), _book(simulated, later)
    minute = pd.DatetimeIndex(study["interval_start_london"]).minute
    book_slots = np.flatnonzero(np.asarray(minute) == 0)
    known = book_slots <= decision_slot
    # Equal up to float summation noise (about 1e-14 kWh): nothing after the
    # decision reaches its book.
    np.testing.assert_allclose(base[:, known], moved[:, known], rtol=0.0, atol=1e-9)
    assert np.abs(base[:, ~known] - moved[:, ~known]).max() > 1e-3


# --------------------------------------------------------------------------
# Events and zones wired into the overlay (§2.4, §4.6, §5.6, §9.3d)
# --------------------------------------------------------------------------

_PRESETS = ("dfs_turn_down", "local_turn_up", "charger_control_outage")


@cache
def _events_run() -> ForecastResult:
    return run_forecast_from_assumptions(_START, values=_SMALL, event_presets=_PRESETS)


def test_events_run_passes_the_validator_and_pays_requests_on_their_night() -> None:
    result = _events_run()
    validate_result_v2(result)
    ledger = result.trading_ledger_world
    paid = ledger.groupby("night_index")["grid_event_payment_gbp"].sum()
    # dfs_turn_down is on night 1 and local_turn_up on night 5.
    assert paid[1] > 0.0 and paid[5] > 0.0
    assert paid.drop([1, 5]).eq(0.0).all()
    bands = result.event_response_bands
    for event_id in ("dfs_turn_down", "local_turn_up"):
        metrics = set(bands.loc[bands["event_id"].eq(event_id), "metric"])
        assert {"baseline_kw", "delivered_kw"} <= metrics


def test_request_windows_are_closed_to_trading_and_settlement() -> None:
    deviation = _events_run().deviation_world_slot
    closed = deviation.loc[~deviation["settlement_open"]]
    assert len(closed) > 0
    for column in ("settled_kwh", "position_da_only_kwh", "position_full_kwh"):
        assert (closed[column] == 0.0).all()


def test_the_expected_plan_pulls_charging_into_a_known_turn_up_window() -> None:
    # The local turn-up's bonus (zone 2, 23:00-01:00 on night 5) is known at
    # the day-ahead decision, so the trader expects zone 2's EVs to charge
    # more in its window, as the planner does.
    plain = run_forecast_from_assumptions(_START, values=_SMALL)
    up = run_forecast_from_assumptions(_START, values=_SMALL, event_presets=("local_turn_up",))
    window = ~up.deviation_world_slot["settlement_open"].to_numpy()
    before = plain.deviation_world_slot["expected_metered_kwh"].to_numpy()[window].sum()
    after = up.deviation_world_slot["expected_metered_kwh"].to_numpy()[window].sum()
    assert after > before


def test_revenue_segments_include_zones_that_add_up_to_the_fleet() -> None:
    segments = _events_run().revenue_by_segment.query("metric == 'gbp_per_week'")
    zones = segments.loc[segments["segment_type"].eq("zone")]
    assert set(zones["segment_id"]) == set(assumptions.ZONE_IDS)
    for bucket, rows in segments.groupby("bucket"):
        whole = rows.loc[rows["segment_type"].eq("all"), "mean"].iat[0]
        assert rows.loc[rows["segment_type"].eq("zone"), "mean"].sum() == pytest.approx(whole)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("trading.baseline_working_nights", 0.0, "whole number of at least 1"),
        ("trading.baseline_non_working_nights", 1.5, "whole number of at least 1"),
        ("trading.baseline_adjustment_window_slots", 0.0, "whole number of at least 1"),
        ("trading.baseline_trap", 0.5, "switch"),
        ("trading.half_spread_gbp_per_mwh", 25.0, "between 0 and 20"),
        ("trading.control_group_share", 0.0, "between 0.01 and 0.99"),
    ],
)
def test_trading_assumptions_are_checked_against_their_records(
    name: str, value: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _direct({name: value})
