"""The trading overlay with intraday dispatch (dispatch contract v1 §6, §8 rows 5 and 9, lane K2).

Hand-checkable SYNTHETIC arrays first (the trading contract's §4.4 worked
example with a free session, the frozen-book re-run's split on a hand book,
a request against a spike), then a small SYNTHETIC real run through lane K1's
kernel for the frozen-book identity.  Every price, volume and pound is
synthetic or illustrative; the net is illustrative simulated trading P&L,
never Axle cash.
"""

from __future__ import annotations

import math
from datetime import date
from functools import cache

import numpy as np
import pandas as pd
import pytest
from model.test_market import _dfs_events, _run_half_hours, _tiny_run

from axle_studio.model import assumptions, forecast, market
from axle_studio.model import events as model_events
from axle_studio.model.action import intraday_dispatch_inputs, locked_evs
from axle_studio.model.physics import DispatchSums, simulate_fleet_intervals
from axle_studio.model.summaries import build_study_slots

_MONDAY = date(2026, 10, 12)
_HALF_SPREAD = 1.0

# --------------------------------------------------------------------------
# §6.1 and trading §4.4: the worked example with a free session
# --------------------------------------------------------------------------


def _worked_example(
    free: list[bool] | None, latest: list[float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    # Trading §4.4: sessions A (6 kWh by the end of slot 3) and B (2 kWh by
    # the end of slot 1), weight 1, 5 kWh per slot; B0 = [7, 1, 0, 0].
    sessions = market.expected_sessions(
        plug_probability=np.array([1.0, 1.0]),
        need_kwh=np.array([[6.0, 2.0]]),
        slot_cap_kwh=np.array([5.0, 5.0]),
        start=np.array([0, 0]),
        deadline=np.array([4, 2]),
    )
    expected = market.expected_smart_kwh(
        sessions,
        np.array([[100.0, 80.0, 30.0, 40.0]]),
        first_slot=0,
        non_response_probability=np.zeros(2),
        free=None if free is None else np.array(free),
        free_window_price_gbp_per_mwh=None if latest is None else np.array([latest]),
    )
    q = market.positions_kwh(np.array([[7.0, 1.0, 0.0, 0.0]]), expected, np.ones(4, dtype=bool))
    return expected, q


def test_worked_example_with_a_free_session_ranks_the_latest_price() -> None:
    # Locked sessions (or dispatch off) reproduce the worked example exactly.
    for free in (None, [False, False]):
        expected, q = _worked_example(free, [100.0, 80.0, 300.0, 40.0])
        np.testing.assert_array_equal(expected, [[0.0, 2.0, 5.0, 1.0]])
        np.testing.assert_array_equal(q, [[7.0, 0.0, 0.0, 0.0]])
    # A free, and slot 2 has spiked to 300 intraday: A fills its cheapest
    # latest slots, 3 (40) then 1 (80); B stays on the visible prices.
    expected, q = _worked_example([True, False], [100.0, 80.0, 300.0, 40.0])
    np.testing.assert_array_equal(expected, [[0.0, 3.0, 0.0, 5.0]])
    np.testing.assert_array_equal(q, [[7.0, 0.0, 0.0, 0.0]])
    # B free, and slot 1 has spiked: B moves into slot 0, so the forecast
    # turn-down there shrinks from 7 to 5 and slot 1 gains 1 kWh.
    expected, q = _worked_example([False, True], [100.0, 900.0, 30.0, 40.0])
    np.testing.assert_array_equal(expected, [[2.0, 0.0, 5.0, 1.0]])
    np.testing.assert_array_equal(q, [[5.0, 1.0, 0.0, 0.0]])
    # A free session whose latest price is the visible one ranks as locked
    # (the intraday path starts at the day-ahead price).
    expected, _ = _worked_example([True, True], [100.0, 80.0, 30.0, 40.0])
    np.testing.assert_array_equal(expected, [[0.0, 2.0, 5.0, 1.0]])


def test_dispatch_inputs_come_together() -> None:
    with pytest.raises(ValueError, match="both dispatch_locked and day_ahead_book_kwh"):
        _tiny_run(dispatch_locked=np.zeros(2, dtype=bool))


# --------------------------------------------------------------------------
# §6.2 and §8 row 9: the frozen-book re-run
# --------------------------------------------------------------------------


def test_switch_off_frozen_trades_are_the_trades_and_re_optimisation_is_exactly_zero() -> None:
    run = _tiny_run(worlds=2)
    updates = run.position_updates
    assert list(updates.columns) == list(market.POSITION_UPDATE_COLUMNS)
    np.testing.assert_array_equal(updates["frozen_trade_kwh"], updates["trade_kwh"])
    np.testing.assert_array_equal(updates["frozen_position_kwh"], updates["position_kwh"])
    ledger = run.trading_ledger_world
    full = ledger.loc[ledger["strategy"].eq("full")]
    assert (full["intraday_pnl_reoptimisation_gbp"] == 0.0).all()
    assert (full["trading_cost_reoptimisation_gbp"] == 0.0).all()
    np.testing.assert_array_equal(full["intraday_pnl_rebalancing_gbp"], full["intraday_pnl_gbp"])
    np.testing.assert_array_equal(full["trading_cost_rebalancing_gbp"], full["trading_cost_gbp"])
    others = ledger.loc[ledger["strategy"].ne("full"), list(market.DISPATCH_SPLIT_COLUMNS)]
    assert (others == 0.0).all().all()
    # Not buckets: the net is still the nine-bucket sum.
    np.testing.assert_array_equal(ledger["net_gbp"], ledger[list(market.BUCKETS)].sum(axis=1))


def test_traded_price_is_the_f2_rule_through_latest_known_prices() -> None:
    # Regression for the swap to action.latest_known_prices: every intraday
    # row is priced at path[t, min(ceil((gate - tau) / 1 h), H - 1)], also at
    # the 12:00 decision on next-day slots whose day-ahead price is not yet
    # published (the path's pre-publication value, as F2 has always booked).
    path = np.random.default_rng(7).normal(80.0, 20.0, size=(1, 480, 36))
    run = _tiny_run(intraday_path_gbp_per_mwh=path)
    updates = run.position_updates.query("stage == 'intraday'")
    gate = updates["interval_start_utc"] - pd.Timedelta(minutes=60)
    steps = np.ceil((gate - updates["decision_utc"]) / pd.Timedelta(hours=1)).astype(int)
    expected = path[0, 144 + updates["slot_index"].to_numpy(), np.minimum(steps, 35)]
    np.testing.assert_array_equal(updates["price_gbp_per_mwh"], expected)
    unpublished = updates["decision_utc"].dt.tz_convert("Europe/London").dt.hour.eq(12) & (
        updates["interval_start_london"].dt.date != updates["decision_london"].dt.date
    )
    assert unpublished.sum() > 0


def _book_moving(m: float, left: int, entered: int, move_at: int):
    """Two SYNTHETIC books: a plan with ``m`` kWh in slot ``left`` in force from
    three decisions before ``move_at``; on the dispatched book it moves to slot
    ``entered`` at ``move_at``, the day-ahead book keeps it."""

    study = build_study_slots(_MONDAY, 7)
    minute = pd.DatetimeIndex(study["interval_start_london"]).minute
    book_slots = np.flatnonzero(np.asarray(minute) == 0)
    day_ahead = np.zeros((1, len(book_slots), market.BOOK_WIDTH))
    dispatched = np.zeros_like(day_ahead)
    for d, s in enumerate(book_slots):
        if move_at - 6 <= s <= left:
            day_ahead[0, d, left - s] = m
            if s < move_at:
                dispatched[0, d, left - s] = m
        if move_at <= s <= entered:
            dispatched[0, d, entered - s] = m
    return day_ahead, dispatched


def test_a_move_between_two_open_slots_is_re_optimisation_at_its_decision() -> None:
    # Night 1, no expected sessions (need 0), so the forecast is the book.
    # At 18:00 the dispatch moves 2 kWh from 20:00 to 21:00.  The trader
    # buys back 2 kWh of turn-down at 21:00 and sells 2 kWh more at 20:00:
    # Delta - Delta0 = +m where it left and -m where it entered, and at the
    # traded prices the re-optimisation P&L is m (P_20:00 - P_21:00) / 1000
    # and its trading cost -2 s m / 1000.
    study = build_study_slots(_MONDAY, 7)
    labels = study["local_time_label"].to_numpy()
    night = study["night_index"].to_numpy()
    at = {
        label: int(np.flatnonzero((night == 1) & (labels == label))[0])
        for label in ("18:00", "20:00", "21:00")
    }
    m = 2.0
    day_ahead_book, dispatched_book = _book_moving(m, at["20:00"], at["21:00"], at["18:00"])
    base = {
        "expected_need": np.zeros((1, 2)),
        "terms": {"trading.half_spread_gbp_per_mwh": _HALF_SPREAD},
    }
    run = _tiny_run(
        book_kwh=dispatched_book,
        dispatch_locked=np.ones(2, dtype=bool),
        day_ahead_book_kwh=day_ahead_book,
        **base,
    )
    # With the fleet on its day-ahead plans the same run is the switch-off run.
    off = _tiny_run(book_kwh=day_ahead_book, **base)
    np.testing.assert_array_equal(
        run.position_updates["frozen_trade_kwh"], off.position_updates["trade_kwh"]
    )
    updates = run.position_updates
    tau = study["interval_start_utc"].iat[at["18:00"]]
    moved = updates.loc[updates["decision_utc"].eq(tau)].set_index("slot_index")
    difference = moved["trade_kwh"] - moved["frozen_trade_kwh"]
    assert difference[at["20:00"]] == pytest.approx(m)
    assert difference[at["21:00"]] == pytest.approx(-m)
    assert (difference.drop([at["20:00"], at["21:00"]]).abs() < 1e-12).all()
    # Only that decision differs: before it the books agree, after it both
    # books keep their plans, so targets change together.
    others = updates.loc[updates["decision_utc"].ne(tau)]
    np.testing.assert_allclose(others["trade_kwh"], others["frozen_trade_kwh"], atol=1e-12)
    price = moved["price_gbp_per_mwh"]
    full = run.trading_ledger_world.query("strategy == 'full' and night_index == 1").iloc[0]
    assert full["intraday_pnl_reoptimisation_gbp"] == pytest.approx(
        m * (price[at["20:00"]] - price[at["21:00"]]) / 1000.0
    )
    assert full["trading_cost_reoptimisation_gbp"] == pytest.approx(-2 * _HALF_SPREAD * m / 1000)
    # Rebalancing is the frozen trades at the traded prices (B2).
    intraday = updates.query("stage == 'intraday' and night_index == 1")
    frozen = intraday["frozen_trade_kwh"]
    assert full["intraday_pnl_rebalancing_gbp"] == pytest.approx(
        (frozen * intraday["price_gbp_per_mwh"]).sum() / 1000.0
    )
    assert full["trading_cost_rebalancing_gbp"] == pytest.approx(
        -_HALF_SPREAD * frozen.abs().sum() / 1000.0
    )
    ledger = run.trading_ledger_world
    for bucket, parts in (
        ("intraday_pnl_gbp", market.DISPATCH_SPLIT_COLUMNS[:2]),
        ("trading_cost_gbp", market.DISPATCH_SPLIT_COLUMNS[2:]),
    ):
        np.testing.assert_allclose(ledger[list(parts)].sum(axis=1), ledger[bucket], atol=1e-12)
    # The final full position follows the dispatched book; day-ahead-only
    # and perfect foresight do not trade intraday.
    assert (
        (ledger.loc[ledger["strategy"].ne("full"), list(market.DISPATCH_SPLIT_COLUMNS)] == 0.0)
        .all()
        .all()
    )
    np.testing.assert_array_equal(run.day_ahead_position, off.day_ahead_position)


# --------------------------------------------------------------------------
# §8 row 9 on a real (SYNTHETIC) run through K1's kernel
# --------------------------------------------------------------------------

_SMALL = {"vehicle_count": 40, "evaluation_world_count": 3}


@cache
def _real() -> tuple[object, object]:
    settings = assumptions.run_settings(_SMALL, _MONDAY)
    fixture = assumptions.cohort_fixture(_SMALL)
    inputs = assumptions.forecast_inputs(
        assumptions.resolve_values(_SMALL), warmup_days=7, study_days=7
    )
    # The switch-off run (dispatch is on by default since K4): these tests
    # build the dispatched path themselves from its inputs.
    inputs["trading_assumptions"] = inputs["trading_assumptions"] | {
        "trading.intraday_dispatch": 0.0
    }
    simulated = forecast.simulate_forecast(settings, fixture, **inputs)
    return simulated, forecast._package(simulated, fixture)


def _overlay_inputs(simulated, result, monkeypatch) -> dict[str, object]:
    """The keyword arguments ``forecast._run_trading`` gives the overlay today."""

    captured: dict[str, object] = {}
    monkeypatch.setattr(forecast.market, "run_trading", lambda **kwargs: captured.update(kwargs))
    forecast._run_trading(simulated, result, [0, 1, 2])
    monkeypatch.undo()
    return captured


def _dispatched(simulated, share: float):
    """K1's kernel with every free EV re-planning on any saving (theta = s = 0)."""

    units = simulated.units.assign(dispatch_locked=locked_evs(simulated.units, share))
    dispatch = intraday_dispatch_inputs(
        simulated.settings,
        units,
        simulated.market_prices.intraday_path_gbp_per_mwh,
        replan_threshold_gbp_per_mwh=0.0,
        half_spread_gbp_per_mwh=0.0,
        gate_closure_minutes=simulated.trading_assumptions["gate_closure_minutes"],
    )
    selected = {k: np.zeros_like(v) for k, v in simulated.trading_kernel["selected"].items()}
    selected["book_decision"] = simulated.trading_kernel["selected"]["book_decision"]
    sums = DispatchSums()
    fleet, *_ = simulate_fleet_intervals(
        simulated.settings,
        units,
        simulated.evaluation.inputs,
        public_top_up=simulated.public_top_up,
        evaluation_effective_efficiency_miles_per_battery_kwh=(
            simulated.evaluation.effective_efficiency_miles_per_battery_kwh
        ),
        smart_charging=simulated.smart_charging,
        trading_output={"selected": selected},
        intraday_dispatch=dispatch,
        dispatch_sums=sums,
    )
    # The kernel's fleet frame holds the study slots only, world-major.
    worlds = simulated.settings.evaluation_world_count
    metered = (
        fleet.loc[fleet["path_id"].eq("selected"), "home_grid_import_kwh"]
        .to_numpy()
        .reshape(worlds, -1)
    )
    return units["dispatch_locked"].to_numpy(), selected, sums, metered


def test_frozen_book_re_run_is_the_switch_off_trade_set_row_for_row(monkeypatch) -> None:
    # With the in-day adjustment off the baseline no longer reads the
    # dispatched import, so the re-run on the day-ahead plan path's book
    # makes exactly the switch-off run's trades (§6.2, §8 row 9).
    simulated, result = _real()
    inputs = _overlay_inputs(simulated, result, monkeypatch)
    inputs["assumptions"] = dict(inputs["assumptions"]) | {
        "trading.baseline_in_day_adjustment": 0.0,
        "trading.half_spread_gbp_per_mwh": _HALF_SPREAD,
    }
    off = market.run_trading(**inputs)
    locked, selected, sums, metered = _dispatched(simulated, 0.5)
    np.testing.assert_array_equal(sums.book_kwh["day_ahead"], inputs["book_kwh"])
    on = market.run_trading(
        **inputs
        | {
            "metered_kwh": metered,
            "book_kwh": selected["book_kwh"],
            "departure_shortfall_kwh": inputs["departure_shortfall_kwh"]
            | {"selected": selected["departure_shortfall_kwh"]},
            "dispatch_locked": locked,
            "day_ahead_book_kwh": sums.book_kwh["day_ahead"],
        }
    )
    np.testing.assert_array_equal(
        on.position_updates["frozen_trade_kwh"], off.position_updates["trade_kwh"]
    )
    np.testing.assert_array_equal(
        on.position_updates["frozen_position_kwh"], off.position_updates["position_kwh"]
    )
    # The same q and the same traded prices; the dispatch made the fleet
    # (and so the trader) move.
    np.testing.assert_array_equal(on.day_ahead_position, off.day_ahead_position)
    np.testing.assert_array_equal(
        on.position_updates["price_gbp_per_mwh"], off.position_updates["price_gbp_per_mwh"]
    )
    assert not np.array_equal(on.positions["full"], off.positions["full"])
    ledger = on.trading_ledger_world
    full = ledger.loc[ledger["strategy"].eq("full")]
    off_full = off.trading_ledger_world.query("strategy == 'full'")
    # Rebalancing per night is the switch-off run's intraday P&L and cost for
    # every world, sampled or not.
    np.testing.assert_allclose(
        full["intraday_pnl_rebalancing_gbp"], off_full["intraday_pnl_gbp"], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        full["trading_cost_rebalancing_gbp"], off_full["trading_cost_gbp"], rtol=0, atol=1e-12
    )
    assert (full["intraday_pnl_reoptimisation_gbp"] != 0.0).any()
    for bucket, parts in (
        ("intraday_pnl_gbp", market.DISPATCH_SPLIT_COLUMNS[:2]),
        ("trading_cost_gbp", market.DISPATCH_SPLIT_COLUMNS[2:]),
    ):
        total = ledger[bucket]
        residual = ledger[list(parts)].sum(axis=1) - total
        assert (residual.abs() <= 1e-9 * np.maximum(1.0, total.abs())).all()
    assert (
        (ledger.loc[ledger["strategy"].ne("full"), list(market.DISPATCH_SPLIT_COLUMNS)] == 0.0)
        .all()
        .all()
    )
    # Every strategy settles on the one dispatched path.
    settled = on.deviation_world_slot["settled_kwh"].to_numpy().reshape(3, -1)
    np.testing.assert_array_equal(on.positions["perfect_foresight"], settled)
    np.testing.assert_array_equal(
        on.deviation_world_slot["metered_kwh"].to_numpy().reshape(3, -1), metered
    )


def test_free_sessions_move_the_trader_forecast_only_through_the_latest_price(
    monkeypatch,
) -> None:
    # Every EV locked: the expected plans are the switch-off ones, so on the
    # same book and import the whole overlay is the switch-off overlay.
    simulated, result = _real()
    inputs = _overlay_inputs(simulated, result, monkeypatch)
    off = market.run_trading(**inputs)
    all_locked = market.run_trading(
        **inputs
        | {"dispatch_locked": np.ones(40, dtype=bool), "day_ahead_book_kwh": inputs["book_kwh"]}
    )
    pd.testing.assert_frame_equal(all_locked.position_updates, off.position_updates)
    pd.testing.assert_frame_equal(all_locked.trading_ledger_world, off.trading_ledger_world)
    # Every EV free on the same book: only the not-yet-started sessions'
    # expected plans change, so the day-ahead position does not.
    all_free = market.run_trading(
        **inputs
        | {"dispatch_locked": np.zeros(40, dtype=bool), "day_ahead_book_kwh": inputs["book_kwh"]}
    )
    np.testing.assert_array_equal(all_free.day_ahead_position, off.day_ahead_position)
    np.testing.assert_array_equal(
        all_free.position_updates["frozen_trade_kwh"], off.position_updates["trade_kwh"]
    )
    assert not all_free.position_updates["forecast_metered_kwh"].equals(
        off.position_updates["forecast_metered_kwh"]
    )


# --------------------------------------------------------------------------
# §8 row 5: delivery and payment for a request against a spike
# --------------------------------------------------------------------------


def test_a_request_against_a_spike_is_paid_on_delivery_once_and_never_traded() -> None:
    # The two outcomes of the K1 kernel test "requests on one scale": the
    # fleet leaves the DFS window for a GBP 220/MWh spike just before it, or
    # charges in the window to avoid a GBP 2,000/MWh spike.  Moving m kWh
    # into the window lowers delivery against the scope's baseline by m and
    # the payment by m x payment / 1000; the payment is booked once per
    # strategy on the window's night and the window is never traded.
    study = build_study_slots(_MONDAY, 7)
    events = _dfs_events()
    window = model_events.event_slots(events, study).iloc[0]
    first, end = int(window["start_slot"]), int(window["end_slot"])
    spike_slot = first - 1
    zones = len(assumptions.ZONE_IDS)
    warm_half = _run_half_hours()[:144]
    warm = np.where((warm_half >= 36) & (warm_half < 44), 5.0, 0.5)
    base = _tiny_run(worlds=1).deviation_world_slot["metered_kwh"].to_numpy().reshape(1, 336)
    base[:, first:end] = 0.0  # the fleet turned down through the window
    m = 0.25
    outcomes = {}
    for where in ("before_window", "in_window"):
        metered = base.copy()
        metered[0, spike_slot if where == "before_window" else first] += m
        run = _tiny_run(
            worlds=1,
            metered_kwh=metered,
            events=events,
            zone_warmup_kwh=np.broadcast_to(warm / zones, (1, zones, 144)).copy(),
            zone_metered_kwh=np.repeat(metered[:, None, :] / zones, zones, axis=1),
        )
        outcomes[where] = run
        assert np.all(run.day_ahead_position[:, first:end] == 0.0)
        assert np.all(run.positions["full"][:, first:end] == 0.0)
        assert np.all(run.settled[:, first:end] == 0.0)
        in_window = run.position_updates["slot_index"].isin(range(first, end))
        assert (run.position_updates.loc[in_window, "trade_kwh"] == 0.0).all()
        ledger = run.trading_ledger_world
        paid = ledger.groupby("strategy")["grid_event_payment_gbp"].sum()
        assert paid.nunique() == 1
        assert paid.iloc[0] == pytest.approx(run.event_delivery["payment_gbp"].sum())
    # The move stays inside what the first window slot delivered, and the
    # SYNTHETIC request's cap does not bind, so delivery and payment are linear.
    assert outcomes["before_window"].scope_baseline[0, 0, first] - base[0, first] >= m
    cap = float(events["size"].iat[0]) * 1000.0 * 0.5
    assert math.isnan(cap) or cap >= outcomes["before_window"].scope_baseline[0, 0, first]
    avoided = outcomes["before_window"].event_delivery.iloc[0]
    charged = outcomes["in_window"].event_delivery.iloc[0]
    assert avoided["delivered_kwh"] - charged["delivered_kwh"] == pytest.approx(m)
    payment = float(events["payment_gbp_per_mwh"].iat[0])
    assert avoided["payment_gbp"] - charged["payment_gbp"] == pytest.approx(m * payment / 1000.0)
    # The spike half-hour stays open to settlement, so its value flows
    # through intraday P&L and imbalance: charging there settles m less.
    for run in outcomes.values():
        assert run.deviation_world_slot["settlement_open"].iat[spike_slot]
    assert outcomes["in_window"].settled[0, spike_slot] - outcomes["before_window"].settled[
        0, spike_slot
    ] == pytest.approx(min(m, outcomes["in_window"].settled[0, spike_slot]))


# --------------------------------------------------------------------------
# B3 and Q8 captions
# --------------------------------------------------------------------------


def test_strategy_and_perfect_foresight_captions_are_the_contract_text() -> None:
    assert tuple(market.STRATEGY_CAPTIONS) == market.STRATEGIES
    assert market.STRATEGY_CAPTIONS["perfect_foresight"].startswith(
        "Perfect foresight of volume, not of intraday prices"
    )
    assert market.PERFECT_FORESIGHT_CAPTION == (
        "against perfect foresight of volume, not of intraday prices"
    )
    assert market.PERFECT_FORESIGHT_CAPTION_METRICS == (
        "capture_rate",
        "cost_of_uncertainty_gbp_per_week",
    )
