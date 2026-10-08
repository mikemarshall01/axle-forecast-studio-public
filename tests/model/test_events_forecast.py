"""Events and zones in real runs (trading contract v1 §2-§3, §5.10; decision 0004 items 55-58).

Small seeded runs (60 EVs x 4 worlds) compared with the same seed and no
event: scripted events draw nothing, so every difference below is the
event's own effect on identical random futures.
"""

from __future__ import annotations

from datetime import date
from functools import cache

import numpy as np
import pandas as pd
import pytest
from fixtures.result_contract import validate_result_v2

from axle_studio.model import assumptions
from axle_studio.model import events as ev
from axle_studio.model.assumptions import ZONE_IDS
from axle_studio.model.forecast import (
    SimulatedForecast,
    run_forecast_from_assumptions,
    simulate_forecast,
)
from axle_studio.model.sampling import (
    SUPPLY_CURVE_INPUTS,
    build_population,
    supply_curve_gbp_per_mwh,
)
from axle_studio.model.summaries import build_study_slots, build_warmup_slots

_MONDAY = date(2026, 1, 12)
_STUDY = build_study_slots(_MONDAY)
_BASE_VALUES = {"vehicle_count": 60, "evaluation_world_count": 4}
# No stochastic shocks, so a scripted shock's price effect can be checked exactly.
_CALM = (("mild_shock_rate_per_night", 0.0), ("big_shock_rate_per_week", 0.0))


@cache
def _simulate(
    presets: tuple[str, ...] = (), values: tuple[tuple[str, float], ...] = ()
) -> SimulatedForecast:
    resolved = assumptions.resolve_values(_BASE_VALUES | dict(values))
    settings = assumptions.run_settings(resolved, _MONDAY)
    return simulate_forecast(
        settings,
        assumptions.cohort_fixture(resolved),
        **assumptions.forecast_inputs(
            resolved, warmup_days=settings.warmup_days, study_days=settings.study_days
        ),
        events=ev.event_table(presets, _STUDY),
    )


def _imports(sim: SimulatedForecast, path_id: str) -> np.ndarray:
    """(world, study slot) fleet home import, kWh per half-hour."""

    fleet = sim.fleet_world_intervals
    rows = fleet.loc[fleet["path_id"].eq(path_id)].sort_values(["world_id", "interval_start_utc"])
    return rows["home_grid_import_kwh"].to_numpy().reshape(sim.settings.evaluation_world_count, -1)


def _run_prices(sim: SimulatedForecast, frame: pd.DataFrame, column: str) -> np.ndarray:
    """(world, warm-up + study slot) prices from a world-major generator frame."""

    return frame[column].to_numpy().reshape(sim.settings.evaluation_world_count, -1)


def _window(preset: str, *, run: bool = False) -> slice:
    slots = _STUDY
    if run:
        slots = pd.concat([build_warmup_slots(_MONDAY, 7), _STUDY], ignore_index=True)
    rows = ev.event_slots(ev.event_table([preset], _STUDY), slots)
    return slice(int(rows["start_slot"].iat[0]), int(rows["end_slot"].iat[0]))


def test_events_draw_nothing_so_random_futures_and_the_unmanaged_path_are_unchanged() -> None:
    base = _simulate()
    every = _simulate(tuple(assumptions.EVENT_PRESETS))
    pd.testing.assert_frame_equal(base.units, every.units)
    for name, values in base.evaluation.inputs.items():
        np.testing.assert_array_equal(values, every.evaluation.inputs[name])
    pd.testing.assert_frame_equal(base.market_prices.shocks, every.market_prices.shocks)
    # The non-response uniforms are the last draw: equal means no earlier
    # channel took a different number of draws.
    np.testing.assert_array_equal(
        base.smart_charging.non_response_uniform, every.smart_charging.non_response_uniform
    )
    np.testing.assert_array_equal(_imports(base, "normal"), _imports(every, "normal"))
    np.testing.assert_array_equal(base.warmup_home_import_kwh, every.warmup_home_import_kwh)


def test_known_shock_moves_day_ahead_intraday_and_imbalance_by_the_same_increment() -> None:
    base = _simulate(values=_CALM)
    cold = _simulate(("cold_still_evening",), _CALM)
    window = _window("cold_still_evening", run=True)
    column = "wholesale_forecast_gbp_per_mwh"
    day_ahead = _run_prices(cold, cold.forecast_prices, column) - _run_prices(
        base, base.forecast_prices, column
    )
    outside = np.ones(day_ahead.shape[1], dtype=bool)
    outside[window] = False
    assert not day_ahead[:, outside].any()
    assert (day_ahead[:, window] > 0).all()
    # The increment is the supply curve's own, f(ND) - f(ND - shock).
    prices = assumptions.forecast_inputs(None, warmup_days=7, study_days=7)["price_assumptions"]
    curve = {name: prices[name] for name in SUPPLY_CURVE_INPUTS}
    net = _run_prices(cold, cold.forecast_prices, "system_net_demand_gw")
    known, _ = ev.scripted_shock_profiles(
        ev.event_table(["cold_still_evening"], _STUDY),
        pd.concat([build_warmup_slots(_MONDAY, 7), _STUDY], ignore_index=True),
        net.shape[1],
    )
    expected = supply_curve_gbp_per_mwh(net, **curve) - supply_curve_gbp_per_mwh(
        net - known, **curve
    )
    np.testing.assert_allclose(day_ahead[:, window], expected[:, window], atol=1e-9)
    for column in ("evaluation_context_price_gbp_per_mwh", "imbalance_price_gbp_per_mwh"):
        moved = _run_prices(cold, cold.evaluation_prices, column) - _run_prices(
            base, base.evaluation_prices, column
        )
        np.testing.assert_allclose(moved, day_ahead, atol=1e-9)
    # The same GW moves price more when the system is tighter (convex curve):
    # the increment rises with net demand across the window's full-weight slots.
    full = expected[:, window][:, 1:-1].ravel()
    order = np.argsort(net[:, window][:, 1:-1].ravel())
    assert np.all(np.diff(full[order]) >= -1e-9)


def test_surprise_shock_moves_intraday_only_and_no_plan() -> None:
    base = _simulate(values=_CALM)
    spike = _simulate(("surprise_evening_spike",), _CALM)
    window = _window("surprise_evening_spike", run=True)
    column = "wholesale_forecast_gbp_per_mwh"
    np.testing.assert_array_equal(
        _run_prices(base, base.forecast_prices, column),
        _run_prices(spike, spike.forecast_prices, column),
    )
    # Plans rank day-ahead prices only, so the smart path is unchanged.
    np.testing.assert_array_equal(_imports(base, "selected"), _imports(spike, "selected"))
    close = "evaluation_context_price_gbp_per_mwh"
    moved = _run_prices(spike, spike.evaluation_prices, close) - _run_prices(
        base, base.evaluation_prices, close
    )
    outside = np.ones(moved.shape[1], dtype=bool)
    outside[window] = False
    assert not moved[:, outside].any() and (moved[:, window] > 0).all()
    # Revealed 90 minutes ahead: the first intraday step of the first slot
    # (h = 1, before the reveal) has none of it, the close has all of it.
    path_moved = (
        spike.market_prices.intraday_path_gbp_per_mwh - base.market_prices.intraday_path_gbp_per_mwh
    )
    first = window.start
    assert not path_moved[:, first, 1].any()
    np.testing.assert_allclose(path_moved[:, first, 0], moved[:, first])


def test_turn_down_lowers_in_window_import_for_evs_in_scope_and_rebounds_beside_it() -> None:
    # Seed 44, not the default: the shared plug-in skip share (trading
    # contract v1 §10.1b) moved the plug draws, and at the default seed no
    # responding EV is plugged in and charging in this one-hour window, so
    # the event has nothing to move.  Regenerated deliberately (§10.7 J1a).
    seed = (("seed", 44),)
    base = _simulate(values=seed)
    down = _simulate(("dfs_turn_down",), seed)
    window = _window("dfs_turn_down")
    np.testing.assert_array_equal(_imports(base, "normal"), _imports(down, "normal"))
    before, after = _imports(base, "selected"), _imports(down, "selected")
    # Plans rank the window last, so each session puts no more there than it
    # must: never more in-window import in any world, and less overall.
    in_window_before = before[:, window].sum(axis=1)
    assert in_window_before.sum() > 0  # precondition: something to turn down
    in_window_after = after[:, window].sum(axis=1)
    assert (in_window_after <= in_window_before + 1e-9).all()
    assert in_window_after.sum() < in_window_before.sum()
    # The energy is not lost: it moves to other half-hours of the night
    # (rebound), so the week's import is within a small margin.
    moved_out = in_window_before.sum() - in_window_after.sum()
    elsewhere = np.ones(before.shape[1], dtype=bool)
    elsewhere[window] = False
    assert after[:, elsewhere].sum() - before[:, elsewhere].sum() == pytest.approx(
        moved_out, rel=0.05
    )


def test_local_turn_up_raises_import_only_in_its_zone() -> None:
    base = _simulate()
    up = _simulate(("local_turn_up",))
    window = _window("local_turn_up")
    zone = ZONE_IDS.index("zone_2")
    import_column = 2  # physics.ZONE_SUM_COLUMNS: home_grid_import_kwh
    before = base.zone_sums.study["selected"][..., import_column]
    after = up.zone_sums.study["selected"][..., import_column]
    others = [z for z in range(len(ZONE_IDS)) if z != zone]
    np.testing.assert_array_equal(before[:, others], after[:, others])
    assert (after[:, zone, window].sum(axis=1) >= before[:, zone, window].sum(axis=1) - 1e-9).all()
    assert after[:, zone, window].sum() > before[:, zone, window].sum()


def test_control_outage_changes_only_plans_made_in_its_window() -> None:
    base = _simulate()
    outage = _simulate(("charger_control_outage",))
    window = _window("charger_control_outage")
    before, after = _imports(base, "selected"), _imports(outage, "selected")
    np.testing.assert_array_equal(before[:, : window.start], after[:, : window.start])
    np.testing.assert_array_equal(_imports(base, "normal"), _imports(outage, "normal"))
    # Covered sessions that ignore their plan charge at plug-in, in the
    # evening, instead of in the cheap night slots.
    evening = slice(window.start, window.start + 10)
    assert after[:, evening].sum() > before[:, evening].sum()


def test_zone_counts_are_exact_and_editing_shares_moves_no_other_draw() -> None:
    resolved = assumptions.resolve_values({"vehicle_count": 7})
    settings = assumptions.run_settings(resolved, _MONDAY)
    fixture = assumptions.cohort_fixture(resolved)
    rng_default, rng_edited = np.random.default_rng(3), np.random.default_rng(3)
    default = build_population(settings, fixture, rng_default)
    edited = build_population(settings, fixture, rng_edited, zone_shares=(0.1, 0.2, 0.3, 0.4))
    # Largest remainder on 0.7, 1.4, 2.1, 2.8: floors 0, 1, 2, 2 and the two
    # EVs left go to the largest remainders (zone_4 0.8, zone_1 0.7).
    assert edited["zone_id"].value_counts().reindex(ZONE_IDS).tolist() == [1, 1, 2, 3]
    assert default["zone_id"].value_counts().reindex(ZONE_IDS).tolist() == [2, 2, 2, 1]
    pd.testing.assert_frame_equal(
        default.drop(columns=["zone_id", "population_id"]),
        edited.drop(columns=["zone_id", "population_id"]),
    )
    assert rng_default.random() == rng_edited.random()
    with pytest.raises(ValueError, match="sum to one"):
        build_population(settings, fixture, np.random.default_rng(3), zone_shares=(0.5,) * 4)


def test_zone_sums_add_up_to_the_fleet_on_both_paths_and_in_the_warm_up() -> None:
    sim = _simulate(("local_turn_up",))
    for path_id in ("normal", "selected"):
        zones = sim.zone_sums.study[path_id]
        np.testing.assert_allclose(zones[..., 2].sum(axis=1), _imports(sim, path_id), atol=1e-9)
        assert (zones[..., 0].sum(axis=1) == sim.settings.vehicle_count).all()
    np.testing.assert_allclose(
        sim.zone_sums.warmup_home_import_kwh.sum(axis=1), sim.warmup_home_import_kwh, atol=1e-9
    )


@pytest.mark.parametrize("model", ["action", "no_action"])
def test_a_run_with_every_preset_passes_the_result_contract(model: str) -> None:
    result = run_forecast_from_assumptions(
        _MONDAY, model=model, values=_BASE_VALUES, event_presets=tuple(assumptions.EVENT_PRESETS)
    )
    validate_result_v2(result)
    assert result.zone_ids == ZONE_IDS and result.zone_summary is not None
    if model == "action":
        assert len(result.events) == 14  # five one-row presets, 7 cold nights, 2 sunny days
        enabled = set(result.events["event_id"])
        assert set(result.event_response_bands["event_id"]) == enabled
        scripted = result.market_shocks.loc[result.market_shocks["source"].eq("scripted")]
        assert len(scripted) == 11 * result.world_count  # price-shock rows x worlds
    else:
        assert result.events is None and result.event_response_bands is None
