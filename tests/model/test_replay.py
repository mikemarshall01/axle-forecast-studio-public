"""The fleet replay (replay contract v1 §1, §5): known-at cut-offs, positions and running money.

Every number is SYNTHETIC: small hand-made price and position arrays for the
leakage rules, then small seeded runs for the reconciliations.  The replay
draws nothing, so the real runs are the same futures as without it.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from functools import cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fixtures.replay_contract import validate_replay_week_v2
from fixtures.result_fixture import validate_result_v2

from axle_studio.model import assumptions, market, replay
from axle_studio.model import events as model_events
from axle_studio.model.action import calculate_wholesale_world_cost_effect, slot_cost_effect_gbp
from axle_studio.model.forecast import ForecastResult, run_forecast_from_assumptions
from axle_studio.model.sampling import day_ahead_publication_utc_ns
from axle_studio.model.summaries import build_study_slots

_START = date(2026, 10, 12)  # a Monday, no clock change
_SMALL = {"vehicle_count": 30, "evaluation_world_count": 4}
_EVENTS = ("dfs_turn_down", "local_turn_up", "charger_control_outage", "surprise_evening_spike")
_HOUR_NS = 3600 * 1_000_000_000


@cache
def _run() -> ForecastResult:
    return run_forecast_from_assumptions(_START, values=_SMALL)


@cache
def _events_run() -> ForecastResult:
    return run_forecast_from_assumptions(_START, values=_SMALL, event_presets=_EVENTS)


# --------------------------------------------------------------------------
# §1.2 no leakage, prices (two worlds, eight study slots)
# --------------------------------------------------------------------------

# SYNTHETIC market: 92 warm-up slots from London midnight on D-1, then eight
# study slots from 22:00 London on D0 (four on D0, four on D1), and decisions
# every hour from 12:00 London on D0, so D1's prices are published (13:00 on
# D0) after the first decision.
_RUN_STARTS = pd.date_range(
    pd.Timestamp("2026-10-11").tz_localize("Europe/London").tz_convert("UTC"),
    periods=100,
    freq="30min",
)
_WARMUP = 92
_DECISIONS = pd.date_range(
    pd.Timestamp("2026-10-12 12:00").tz_localize("Europe/London").tz_convert("UTC"),
    periods=13,
    freq="h",
)
_GATE_MINUTES = 60.0


def _market(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(20.0, 150.0, (2, 100)), rng.uniform(20.0, 150.0, (2, 100, 36))


def _curves(day_ahead: np.ndarray, path: np.ndarray):
    return replay.forward_curves(
        day_ahead, _RUN_STARTS, path, _WARMUP, _DECISIONS.asi8, _GATE_MINUTES
    )


def _latest_step(k: int) -> np.ndarray:
    """h_t(tau_k) of the contract for the eight study slots."""

    gate = _RUN_STARTS[_WARMUP:].asi8 - int(_GATE_MINUTES * 60e9)
    return np.clip(np.ceil((gate - _DECISIONS.asi8[k]) / _HOUR_NS), 0, 35).astype(int)


def test_prices_after_the_decision_never_reach_its_row() -> None:
    day_ahead, path = _market()
    published, visible, known = _curves(day_ahead, path)
    assert not published[0].all() and published[-1].all(), "the fixture crosses a publication"
    publication = day_ahead_publication_utc_ns(_RUN_STARTS)
    for k in range(len(_DECISIONS)):
        tau = _DECISIONS.asi8[k]
        later_day_ahead = day_ahead.copy()
        later_day_ahead[:, publication > tau] += 500.0
        later_path = path.copy()
        steps = _latest_step(k)
        for t, h in enumerate(steps):
            later_path[:, _WARMUP + t, :h] += 500.0
        again = _curves(later_day_ahead, later_path)
        np.testing.assert_array_equal(again[0][k], published[k])
        np.testing.assert_array_equal(again[1][:, k], visible[:, k])
        np.testing.assert_array_equal(again[2][:, k], known[:, k])
    # The same perturbation made at decision 0 does show in a later frame.
    later_path = path.copy()
    for t, h in enumerate(_latest_step(0)):
        later_path[:, _WARMUP + t, :h] += 500.0
    again = _curves(day_ahead, later_path)
    assert not np.array_equal(again[2][:, -1], known[:, -1], equal_nan=True)


def test_intraday_value_is_the_ceil_rule_update_and_nan_before_publication() -> None:
    day_ahead, path = _market()
    published, visible, known = _curves(day_ahead, path)
    for k in (0, 3, 12):
        expected = path[:, _WARMUP + np.arange(8), _latest_step(k)]
        np.testing.assert_array_equal(known[:, k][:, published[k]], expected[:, published[k]])
        assert np.isnan(known[:, k][:, ~published[k]]).all()
        np.testing.assert_array_equal(
            visible[:, k][:, published[k]], day_ahead[:, _WARMUP:][:, published[k]]
        )


# --------------------------------------------------------------------------
# §1.3 no leakage, positions
# --------------------------------------------------------------------------


def _updates(decision_hours: list[int], positions: list[float], slot: int = 1) -> pd.DataFrame:
    start = pd.Timestamp("2026-10-12 11:00", tz="UTC")
    return pd.DataFrame(
        {
            "world_id": np.int64(7),
            "slot_index": np.int64(slot),
            "decision_utc": [start + pd.Timedelta(hours=h) for h in decision_hours],
            "position_kwh": positions,
        }
    )


def test_positions_carry_forward_and_include_a_decision_exactly_at_tau() -> None:
    decisions = pd.date_range("2026-10-12 11:00", periods=5, freq="h", tz="UTC").asi8
    # A day-ahead row before the study (in force from k = 0), then updates at
    # k = 2 and k = 3.
    updates = _updates([-22, 2, 3], [4.0, 3.0, 2.5])
    cube = replay.positions_in_force(updates, [7], decisions, 3)
    np.testing.assert_array_equal(cube[0, :, 1], [4.0, 4.0, 3.0, 2.5, 2.5])
    assert np.isnan(cube[0, :, [0, 2]]).all(), "a slot with no update has no position"
    # Perturbing an update after tau_k leaves row k unchanged.
    later = _updates([-22, 2, 3], [4.0, 3.0, 99.0])
    again = replay.positions_in_force(later, [7], decisions, 3)
    np.testing.assert_array_equal(again[0, :3], cube[0, :3])
    assert again[0, 3, 1] == 99.0
    # An update not yet made at the first instant is NaN there.
    none_yet = replay.positions_in_force(_updates([1], [5.0]), [7], decisions, 3)
    assert np.isnan(none_yet[0, 0, 1]) and none_yet[0, 1, 1] == 5.0


def test_night_one_positions_come_into_force_at_its_day_ahead_decision() -> None:
    result = _run()
    positions = result.replay_week.position_kwh
    night = result.study_slots["night_index"].to_numpy()
    assert np.isfinite(positions[:, 0, night == 0]).all()
    # Night 1's day-ahead decision is 13:00 London on D0, the frame k = 1
    # (the study starts at 12:00 London).
    assert result.replay_week.decisions["label"].iat[1] == "Mon 13:00"
    assert np.isnan(positions[:, 0, night == 1]).all()
    assert np.isfinite(positions[:, 1, night == 1]).all()


# --------------------------------------------------------------------------
# §1.4 running money
# --------------------------------------------------------------------------


def test_real_runs_pass_the_replay_validator() -> None:
    for result in (_run(), _events_run()):
        validate_result_v2(result)
        replay_week = result.replay_week
        assert replay_week.world_ids == result.sampled_world_ids
        k, s = len(replay_week.decisions), result.study_slot_count
        assert replay_week.intraday_known_gbp_per_mwh.shape == (len(replay_week.world_ids), k, s)
        # The realised part of the known intraday curve is the close.
        close = result.deviation_world_slot.set_index(["world_id", "slot_index"])
        rows = [int(w) for w in replay_week.world_ids]
        closes = np.stack([close.loc[w, "intraday_close_gbp_per_mwh"].to_numpy() for w in rows])
        np.testing.assert_array_equal(replay_week.intraday_known_gbp_per_mwh[:, -1], closes)


def test_no_action_result_has_no_replay() -> None:
    result = run_forecast_from_assumptions(_START, model="no_action", values=_SMALL)
    assert result.replay_week is None


def test_trading_cash_reconciles_with_the_ledger_at_every_night_end() -> None:
    result = _events_run()
    cash = result.replay_week.trading_cash_to_date_gbp
    ledger = result.trading_ledger_world
    full = ledger.loc[ledger["strategy"].eq("full")]
    night = result.study_slots["night_index"].to_numpy()
    ends = np.append(np.flatnonzero(np.diff(night)) + 1, len(night))
    assert (full["customer_revenue_share_gbp"] < 0).any(), "a customer share is exercised"
    assert (full["grid_event_payment_gbp"] > 0).any(), "a grid request is exercised"
    for wi, w in enumerate(result.replay_week.world_ids):
        nets = full.loc[full["world_id"].eq(w)].sort_values("night_index")["net_gbp"]
        np.testing.assert_allclose(cash[wi, ends], nets.cumsum(), rtol=0, atol=1e-9)


def _old_settle_buckets(run: market.TradingRun, inputs: dict, terms: dict) -> dict:
    """The seven slot-summed buckets as ``settle`` formed them inline before the refactor."""

    starts = inputs["night_starts"]
    settled, effect, price = inputs["settled"], inputs["effect"], inputs["day_ahead"]
    q, f = run.day_ahead_position, run.positions["full"]
    effect_cash = np.where(settled > 0.0, effect * price, 0.0) / 1000.0
    sums = lambda values: np.add.reduceat(values, starts, axis=1)  # noqa: E731
    return {
        "day_ahead_revenue_gbp": sums(q * price / 1000.0 - effect_cash),
        "imbalance_gbp": sums((settled - f) * inputs["imbalance_price"] / 1000.0),
        "baseline_effect_gbp": sums(effect_cash),
        "supplier_compensation_gbp": -float(terms["trading.supplier_compensation_gbp_per_mwh"])
        * sums(settled)
        / 1000.0,
    }


def test_settle_on_slot_components_reproduces_the_ledger_bit_for_bit() -> None:
    result = _events_run()
    ledger = result.trading_ledger_world.loc[
        result.trading_ledger_world["strategy"].eq("full")
    ].sort_values(["world_id", "night_index"])
    deviation = result.deviation_world_slot.sort_values(["world_id", "slot_index"])
    worlds, slots = result.world_count, result.study_slot_count

    def m(column: str) -> np.ndarray:
        return deviation[column].to_numpy(dtype=float).reshape(worlds, slots)

    night = result.study_slots["night_index"].to_numpy()
    inputs = {
        "night_starts": np.flatnonzero(np.diff(night, prepend=-1)),
        "settled": m("settled_kwh"),
        "effect": m("baseline_effect_kwh"),
        "day_ahead": m("day_ahead_gbp_per_mwh"),
        "imbalance_price": m("imbalance_gbp_per_mwh"),
    }
    terms = assumptions.trading_inputs(None)
    run = SimpleNamespace(
        day_ahead_position=m("position_da_only_kwh"),
        positions={"full": m("position_full_kwh")},
    )
    for name, values in _old_settle_buckets(run, inputs, terms).items():
        np.testing.assert_array_equal(
            ledger[name].to_numpy().reshape(worlds, -1), values, err_msg=name
        )


def test_slot_cash_sums_to_the_ledger_slot_buckets() -> None:
    result = _events_run()
    ledger = result.trading_ledger_world
    full = ledger.loc[ledger["strategy"].eq("full")].sort_values(["world_id", "night_index"])
    slot_buckets = [
        "day_ahead_revenue_gbp",
        "intraday_pnl_gbp",
        "trading_cost_gbp",
        "imbalance_gbp",
        "baseline_effect_gbp",
        "supplier_compensation_gbp",
    ]
    night_total = full[slot_buckets].sum(axis=1).to_numpy().reshape(result.world_count, -1)
    replay_week = result.replay_week
    flows = np.diff(replay_week.trading_cash_to_date_gbp, axis=1)
    night = result.study_slots["night_index"].to_numpy()
    rows = list(replay_week.world_ids)
    other = (
        full[["customer_revenue_share_gbp", "unmet_charge_penalty_gbp", "grid_event_payment_gbp"]]
        .sum(axis=1)
        .to_numpy()
        .reshape(result.world_count, -1)
    )
    for n in np.unique(night):
        np.testing.assert_allclose(
            flows[:, night == n].sum(axis=1),
            night_total[rows, n] + other[rows, n],
            rtol=0,
            atol=1e-9,
        )


def test_slot_cost_effect_is_the_weekly_cost_effect_formula() -> None:
    result = _run()
    fleet = result.fleet_world_intervals.sort_values(["path_id", "world_id", "slot_index"])
    prices = result.forecast_prices.sort_values(["world_id", "slot_index"])
    worlds, slots = result.world_count, result.study_slot_count

    def m(path: str, column: str) -> np.ndarray:
        values = fleet.loc[fleet["path_id"].eq(path), column].to_numpy(dtype=float)
        return values.reshape(worlds, slots)

    price = prices["wholesale_forecast_gbp_per_mwh"].to_numpy().reshape(worlds, slots)
    rate = 0.79
    zeros = np.zeros((worlds, slots))
    energy = slot_cost_effect_gbp(
        m("normal", "home_import_kwh"), m("selected", "home_import_kwh"), zeros, zeros, price, rate
    )
    # Before the refactor: the home difference at the day-ahead price, and
    # the week's public difference times the rate.
    before_energy = (
        (m("selected", "home_import_kwh") - m("normal", "home_import_kwh")) * price / 1000.0
    ).sum(axis=1)
    before_public = (m("selected", "public_import_kwh") - m("normal", "public_import_kwh")).sum(
        axis=1
    ) * rate
    cost = result.cost_effect.set_index("world_id")
    np.testing.assert_array_equal(energy.sum(axis=1), before_energy)
    # The cost effect sums through a pandas groupby, so it matches to rounding.
    np.testing.assert_allclose(
        cost["illustrative_selected_minus_normal_energy_cost_gbp"].to_numpy(),
        before_energy,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        cost["illustrative_selected_minus_normal_public_charge_cost_gbp"].to_numpy(),
        before_public,
        rtol=1e-12,
        atol=1e-12,
    )
    # The replay's running saving ends at minus the two flow components.
    saving = result.replay_week.energy_saving_to_date_gbp
    rows = list(result.replay_week.world_ids)
    flows = (
        cost["illustrative_selected_minus_normal_energy_cost_gbp"]
        + cost["illustrative_selected_minus_normal_public_charge_cost_gbp"]
    ).to_numpy()[rows]
    np.testing.assert_allclose(saving[:, -1], -flows, rtol=1e-12, atol=1e-9)
    assert (saving[:, 0] == 0.0).all()


def test_cost_effect_is_unchanged_on_hand_made_rows() -> None:
    keys = pd.DataFrame(
        {
            "world_id": [0, 0],
            "interval_start_utc": pd.to_datetime(
                ["2026-10-12 11:00", "2026-10-12 11:30"], utc=True
            ),
        }
    )
    keys["interval_end_utc"] = keys["interval_start_utc"] + pd.Timedelta(minutes=30)
    rows = []
    for path, home, public in (
        ("normal", [2.0, 0.0], [0.0, 1.0]),
        ("selected", [0.0, 3.0], [1.5, 0.0]),
    ):
        rows.append(
            keys.assign(
                path_id=path,
                home_grid_import_kwh=home,
                public_grid_import_kwh=public,
                unserved_outbound_travel_battery_kwh=0.0,
                unserved_return_travel_battery_kwh=0.0,
                closing_battery_kwh=[10.0, 10.0],
            )
        )
    fleet = pd.concat(rows, ignore_index=True)
    prices = keys.assign(wholesale_forecast_gbp_per_mwh=[100.0, 40.0])
    effect = calculate_wholesale_world_cost_effect(fleet, prices, public_charge_gbp_per_kwh=0.5)
    # Energy: (0 - 2) x 100 + (3 - 0) x 40 = -80 GBP/MWh-kWh, /1000; public (1.5 - 1) x 0.5.
    assert effect["illustrative_selected_minus_normal_energy_cost_gbp"].iat[0] == pytest.approx(
        -0.08
    )
    assert effect["illustrative_selected_minus_normal_public_charge_cost_gbp"].iat[
        0
    ] == pytest.approx(0.25)


# --------------------------------------------------------------------------
# §1.4 review B1: grid-event payments booked after the window
# --------------------------------------------------------------------------


def _booking_run(events: pd.DataFrame, payment: float = 12.0) -> SimpleNamespace:
    """A minimal trading run: no slot cash, no night buckets, one paid request."""

    study = build_study_slots(_START, 7)
    ledger = pd.DataFrame(
        {
            "world_id": np.repeat([0, 1], 7),
            "strategy": "full",
            "night_index": np.tile(np.arange(7), 2),
            "customer_revenue_share_gbp": 0.0,
            "unmet_charge_penalty_gbp": 0.0,
        }
    )
    request = events.loc[events["event_type"].isin(model_events.REQUEST_TYPES)]
    delivery = pd.DataFrame(
        {
            "world_id": np.array([0, 1], dtype=np.int64),
            "event_id": request["event_id"].iat[0],
            "payment_gbp": [payment, 2 * payment],
        }
    )
    return SimpleNamespace(
        slot_cash_gbp=np.zeros((2, len(study))),
        trading_ledger_world=ledger,
        event_delivery=delivery,
    ), study


def test_a_request_payment_appears_exactly_at_the_window_end() -> None:
    study = build_study_slots(_START, 7)
    events = model_events.validate_events(
        model_events.event_table(("dfs_turn_down",), study), study, assumptions.ZONE_IDS
    )
    window = model_events.event_slots(events, study).iloc[0]
    run, study = _booking_run(events)
    cash = replay.trading_cash_to_date(run, np.array([1, 0]), study, events)
    end = int(window["end_slot"])
    assert (cash[:, :end] == 0.0).all(), "nothing is booked before the window has ended"
    np.testing.assert_array_equal(cash[:, end], [24.0, 12.0])
    assert (cash[:, end:] == cash[:, [end]]).all()


def test_perturbing_metered_import_inside_the_window_moves_no_earlier_cash() -> None:
    # A real run: the request's payment is from the metered import across
    # its window, so a change to a window slot after tau_k (and the payment
    # it moves) must not reach the running cash before the window ends.
    base = _events_run()
    study = base.study_slots
    window = model_events.event_slots(base.events, study)
    dfs = window.loc[window["event_id"].eq("dfs_turn_down")].iloc[0]
    first, end = int(dfs["start_slot"]), int(dfs["end_slot"])
    run = _trading_run(base)
    cash = replay.trading_cash_to_date(run, np.arange(base.world_count), study, base.events)
    for k in range((first + 1) // 2, (end + 1) // 2):
        later = slice(2 * k, end)
        changed = _trading_run(base, metered_scale=(later, 0.5))
        again = replay.trading_cash_to_date(
            changed, np.arange(base.world_count), study, base.events
        )
        assert not np.array_equal(
            changed.event_delivery["payment_gbp"], run.event_delivery["payment_gbp"]
        )
        np.testing.assert_array_equal(again[:, : 2 * k + 1], cash[:, : 2 * k + 1])
        assert not np.array_equal(again[:, end], cash[:, end])


def test_a_window_crossing_noon_is_booked_at_its_night_end() -> None:
    # Documented limitation (§1.4, §7): a request whose window crosses 12:00
    # is booked at the end of the night it starts in, before its last slot
    # has ended, so the night-end identity with the ledger holds.
    study = build_study_slots(_START, 7)
    events = model_events.validate_events(
        model_events.event_table(("dfs_turn_down",), study), study, assumptions.ZONE_IDS
    ).assign(start_local_time="11:00", duration_minutes=np.int64(120), night_index=np.int64(2))
    window = model_events.event_slots(events, study).iloc[0]
    night = study["night_index"].to_numpy()
    night_end = int(np.flatnonzero(night == 2)[-1]) + 1
    assert int(window["end_slot"]) > night_end, "the window crosses 12:00"
    run, study = _booking_run(events)
    cash = replay.trading_cash_to_date(run, np.array([0, 1]), study, events)
    assert (cash[:, :night_end] == 0.0).all()
    np.testing.assert_array_equal(cash[:, night_end], [12.0, 24.0])


# --------------------------------------------------------------------------
# §1.5 shocks and events
# --------------------------------------------------------------------------


def test_shocks_and_events_are_known_by_their_own_rules() -> None:
    result = _events_run()
    shocks = result.replay_week.shocks
    known = shocks["known_day_ahead"].to_numpy(dtype=bool)
    publication = day_ahead_publication_utc_ns(pd.DatetimeIndex(shocks["start_utc"]))
    np.testing.assert_array_equal(
        shocks["known_from_utc"].astype("int64")[known], publication[known]
    )
    stochastic = ~known & shocks["source"].eq("stochastic").to_numpy()
    reveal = next(a.value for a in result.assumptions if a.name == "surprise_reveal_hours")
    assert stochastic.any()
    expected = shocks["start_utc"] - pd.Timedelta(hours=reveal)
    assert (shocks["known_from_utc"][stochastic] == expected[stochastic]).all()
    scripted = shocks["shock_id"].eq("surprise_evening_spike")
    assert scripted.sum() == len(result.replay_week.world_ids)
    notice = shocks.loc[scripted, "start_utc"] - pd.Timedelta(minutes=90)
    assert (shocks.loc[scripted, "known_from_utc"] == notice).all()
    events = result.replay_week.events
    assert list(events["event_id"]) == ["dfs_turn_down", "local_turn_up", "charger_control_outage"]
    empty = _run().replay_week.events
    assert empty.empty and list(empty.columns) == list(events.columns)


# --------------------------------------------------------------------------
# Sizes and pairing
# --------------------------------------------------------------------------


def test_two_runs_with_one_seed_give_identical_replays() -> None:
    first = _run().replay_week
    second = run_forecast_from_assumptions(_START, values=_SMALL).replay_week
    for field in dataclasses.fields(first):
        a, b = getattr(first, field.name), getattr(second, field.name)
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b


def test_the_validator_rejects_a_payment_booked_at_the_window_start() -> None:
    result = _events_run()
    replay_week = result.replay_week
    study = result.study_slots
    window = model_events.event_slots(result.events, study)
    dfs = window.loc[window["event_id"].eq("dfs_turn_down")].iloc[0]
    delivery = _trading_run(result).event_delivery
    cash = replay_week.trading_cash_to_date_gbp.copy()
    for wi, w in enumerate(replay_week.world_ids):
        paid = delivery.loc[
            delivery["world_id"].eq(w) & delivery["event_id"].eq("dfs_turn_down"), "payment_gbp"
        ].iat[0]
        cash[wi, int(dfs["start_slot"]) + 1 : int(dfs["end_slot"])] += paid
    broken = dataclasses.replace(
        result, replay_week=dataclasses.replace(replay_week, trading_cash_to_date_gbp=cash)
    )
    with pytest.raises(AssertionError, match="outside payment boundaries"):
        validate_replay_week_v2(broken)


# --------------------------------------------------------------------------
# Helpers: rerun the overlay of a finished result with one change
# --------------------------------------------------------------------------


@cache
def _simulated_events_run():
    from axle_studio.model.forecast import simulate_forecast

    inputs = assumptions.forecast_inputs(
        assumptions.resolve_values(_SMALL), warmup_days=7, study_days=7
    )
    settings = assumptions.run_settings(_SMALL, _START)
    study = build_study_slots(_START, 7)
    events = model_events.event_table(_EVENTS, study)
    return simulate_forecast(settings, assumptions.cohort_fixture(_SMALL), events=events, **inputs)


def _trading_run(result: ForecastResult, metered_scale=None) -> market.TradingRun:
    """``forecast._run_trading`` on the events run, optionally scaling selected import."""

    from axle_studio.model import forecast

    simulated = _simulated_events_run()
    if metered_scale is not None:
        window, factor = metered_scale
        fleet = result.fleet_world_intervals.copy()
        selected = fleet["path_id"].eq("selected") & fleet["slot_index"].isin(
            range(window.start, window.stop)
        )
        fleet.loc[selected, "home_import_kwh"] *= factor
        zones = simulated.zone_sums
        study = {path: values.copy() for path, values in zones.study.items()}
        column = forecast.ZONE_SUM_COLUMNS.index("home_grid_import_kwh")
        study["selected"][:, :, window, column] *= factor
        simulated = dataclasses.replace(
            simulated, zone_sums=dataclasses.replace(zones, study=study)
        )
        result = dataclasses.replace(result, fleet_world_intervals=fleet)
    return forecast._run_trading(simulated, result, list(result.sampled_world_ids))


def test_a_scripted_shock_without_its_event_raises() -> None:
    result = _events_run()
    no_events = result.events.loc[result.events["event_id"].ne("surprise_evening_spike")]
    with pytest.raises(ValueError, match="scripted shock"):
        replay.shock_frame(
            result.market_shocks, result.sampled_world_ids, result.study_slots, no_events, 2.0
        )


def test_the_validator_checks_a_scripted_surprise_notice() -> None:
    result = _events_run()
    shocks = result.replay_week.shocks.copy()
    scripted = shocks["shock_id"].eq("surprise_evening_spike")
    shocks.loc[scripted, "known_from_utc"] -= pd.Timedelta(hours=1)
    broken = dataclasses.replace(
        result, replay_week=dataclasses.replace(result.replay_week, shocks=shocks)
    )
    with pytest.raises(AssertionError, match="notice_utc"):
        validate_replay_week_v2(broken)
