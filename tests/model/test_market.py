"""The trading overlay's rules on hand-checkable arrays (trading contract v1 §4, §9.2, §9.4, §9.5).

Every number here is SYNTHETIC: tiny fleets and hand-made prices chosen so
each bucket, position and baseline can be checked by eye.  Real runs are
checked in ``test_trading_run.py``.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from axle_studio.model import assumptions, forecast, market
from axle_studio.model import events as model_events
from axle_studio.model.action import (
    expected_departures_utc_ns,
    plan_around_blackout,
    plan_cheapest_slots,
)
from axle_studio.model.clock import utc_half_hour_boundaries
from axle_studio.model.sampling import MarketPrices
from axle_studio.model.settings import RunSettings
from axle_studio.model.summaries import build_study_slots, build_warmup_slots

_TERMS = assumptions.trading_inputs(None)
_MONDAY = date(2026, 10, 12)


# --------------------------------------------------------------------------
# §4.4 worked example: expected plan, position and the ledger buckets
# --------------------------------------------------------------------------


def _worked_example_sessions() -> dict[str, np.ndarray]:
    # Two sessions of weight 1 present from slot 0: A needs 6 kWh by the end
    # of slot 3, B needs 2 kWh by the end of slot 1; 5 kWh per slot each.
    return market.expected_sessions(
        plug_probability=np.array([1.0, 1.0]),
        need_kwh=np.array([[6.0, 2.0]]),
        slot_cap_kwh=np.array([5.0, 5.0]),
        start=np.array([0, 0]),
        deadline=np.array([4, 2]),
    )


def test_worked_example_expected_plan_and_day_ahead_position() -> None:
    prices = np.array([[100.0, 80.0, 30.0, 40.0]])
    expected = market.expected_smart_kwh(
        _worked_example_sessions(),
        prices,
        first_slot=0,
        non_response_probability=np.zeros(2),
    )
    np.testing.assert_allclose(expected, [[0.0, 2.0, 5.0, 1.0]])
    # Plan cost (2 x 80 + 5 x 30 + 1 x 40) / 1000 = GBP 0.35.
    assert (expected * prices).sum() / 1000.0 == pytest.approx(0.35)
    q = market.positions_kwh(np.array([[7.0, 1.0, 0.0, 0.0]]), expected, np.ones(4, dtype=bool))
    np.testing.assert_allclose(q, [[7.0, 0.0, 0.0, 0.0]])


def _one_night_slots(slot_count: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "night_index": np.zeros(slot_count, dtype=np.int64),
            "local_date": [_MONDAY] * slot_count,
            "day_label": ["Mon 12"] * slot_count,
        }
    )


def _settle_one_slot(baseline: float) -> pd.Series:
    """Slot 0 of the worked example settled for day-ahead only: q = 7, M = 1, U = 7."""

    q = np.array([[7.0, 0.0, 0.0, 0.0]])
    metered = np.array([[1.0, 0.0, 0.0, 0.0]])
    unmanaged = np.array([[7.0, 0.0, 0.0, 0.0]])
    b = np.array([[baseline, 0.0, 0.0, 0.0]])
    deviation = b - metered
    inputs = {
        "night_starts": np.array([0]),
        "settled": np.maximum(deviation, 0.0),
        "effect": b - unmanaged,
        "reduction": unmanaged - metered,
        "day_ahead": np.array([[100.0, 80.0, 30.0, 40.0]]),
        "imbalance_price": np.array([[120.0, 0.0, 0.0, 0.0]]),
        "kind": np.full((1, 4), "none", dtype=object),
        "event_payment": np.zeros((1, 4)),
        "event_delivered": np.zeros((1, 4)),
        "unmet_kwh": np.zeros((1, 1)),
    }
    terms = _TERMS | {"trading.customer_revenue_share": 0.0}
    return market.settle("da_only", q, q, None, None, _one_night_slots(), inputs, terms).iloc[0]


def test_worked_example_buckets_when_the_baseline_equals_unmanaged() -> None:
    row = _settle_one_slot(7.0)
    assert row["day_ahead_revenue_gbp"] == pytest.approx(0.70)
    assert row["imbalance_gbp"] == pytest.approx(-0.12)  # (6 - 7) / 1000 x 120
    assert row["baseline_effect_gbp"] == pytest.approx(0.0)
    assert row["net_gbp"] == pytest.approx(0.58)
    assert row["settled_mwh"] == pytest.approx(0.006)


def test_worked_example_carves_the_baseline_effect_out_of_day_ahead_revenue() -> None:
    # B = 8 over-states the slot: D = 7, R = 6, E = 1, V = 7 = q, so no
    # imbalance; the GBP 0.10 sold at day-ahead was earned by the baseline.
    row = _settle_one_slot(8.0)
    assert row["imbalance_gbp"] == pytest.approx(0.0)
    assert row["baseline_effect_gbp"] == pytest.approx(0.10)
    assert row["day_ahead_revenue_gbp"] == pytest.approx(0.60)
    assert row["net_gbp"] == pytest.approx(0.70)
    assert row["flexibility_mwh"] + row["baseline_effect_mwh"] == pytest.approx(row["settled_mwh"])


def test_net_is_the_exact_bucket_sum_and_the_customer_share_takes_positive_gross_only() -> None:
    row = _settle_one_slot(7.0)
    assert row["net_gbp"] == sum(row[bucket] for bucket in market.BUCKETS)
    q = np.array([[7.0, 0.0, 0.0, 0.0]])
    inputs = {
        "night_starts": np.array([0]),
        "settled": np.zeros((1, 4)),
        "effect": np.zeros((1, 4)),
        "reduction": np.zeros((1, 4)),
        "day_ahead": np.full((1, 4), 100.0),
        "imbalance_price": np.full((1, 4), 300.0),
        "kind": np.full((1, 4), "none", dtype=object),
        "event_payment": np.zeros((1, 4)),
        "event_delivered": np.zeros((1, 4)),
        "unmet_kwh": np.zeros((1, 1)),
    }
    losing = market.settle("da_only", q, q, None, None, _one_night_slots(), inputs, _TERMS)
    # Sold 7 kWh at 100 and delivered nothing, bought back at 300: gross < 0.
    assert losing["net_gbp"].iat[0] == pytest.approx(0.7 - 2.1)
    assert losing["customer_revenue_share_gbp"].iat[0] == 0.0


def test_unmet_charge_penalty_counts_only_shortfall_beyond_unmanaged() -> None:
    # Review B3: equal shortfall on both paths gives no penalty.
    shortfall = {"normal": np.full((1, 4), 2.0), "selected": np.full((1, 4), 2.0)}
    run = _tiny_run(departure_shortfall_kwh=_widen(shortfall))
    assert (run.trading_ledger_world["unmet_charge_penalty_gbp"] == 0.0).all()
    more = {"normal": np.zeros((1, 4)), "selected": np.full((1, 4), 0.5)}
    run = _tiny_run(departure_shortfall_kwh=_widen(more))
    night0 = run.trading_ledger_world.query("night_index == 0")
    assert night0["unmet_charge_kwh"].tolist() == pytest.approx([2.0] * 3)
    assert night0["unmet_charge_penalty_gbp"].tolist() == pytest.approx([-2.0 * 0.79] * 3)


# --------------------------------------------------------------------------
# Expected plan: the dispatcher's own rule
# --------------------------------------------------------------------------


def test_unmanaged_schedule_is_the_planner_under_a_flat_price() -> None:
    rng = np.random.default_rng(3)
    need = rng.uniform(0.0, 20.0, 50)
    cap = rng.uniform(1.0, 4.0, 50)
    in_window = np.zeros((50, 12), dtype=bool)
    for row, (start, end) in enumerate(rng.integers(0, 12, (50, 2))):
        in_window[row, min(start, end) : max(start, end) + 1] = True
    flat = plan_cheapest_slots(need, cap, np.where(in_window, 0.0, np.inf))
    np.testing.assert_allclose(market.unmanaged_schedule_kwh(need, cap, in_window), flat)


def test_expected_plan_is_the_weighted_sum_of_per_session_plans() -> None:
    rng = np.random.default_rng(5)
    worlds, evs, width = 3, 8, 10
    need = rng.uniform(0.0, 12.0, (worlds, evs))
    cap = np.full(evs, 3.5)
    start = rng.integers(0, 5, evs)
    deadline = start + rng.integers(1, 6, evs)
    weight = rng.uniform(0.2, 1.0, evs)
    rho = rng.uniform(0.0, 1.0, evs)
    prices = rng.uniform(20.0, 200.0, (worlds, width))
    sessions = market.expected_sessions(weight, need, cap, start, deadline)
    expected = market.expected_smart_kwh(
        sessions, prices, first_slot=0, non_response_probability=rho
    )
    manual = np.zeros((worlds, width))
    slots = np.arange(width)
    for w in range(worlds):
        for i in range(evs):
            window = (slots >= start[i]) & (slots < deadline[i])
            session_need = np.array([sessions["need_kwh"][w, i]])
            plan = plan_cheapest_slots(
                session_need, cap[i : i + 1], np.where(window, prices[w], np.inf)[None]
            )[0]
            flat = plan_cheapest_slots(
                session_need, cap[i : i + 1], np.where(window, 0.0, np.inf)[None]
            )[0]
            manual[w] += weight[i] * (rho[i] * flat + (1.0 - rho[i]) * plan)
    np.testing.assert_allclose(expected, manual, atol=1e-12)


def test_expected_plan_leaves_blackout_half_hours_as_the_kernel_does() -> None:
    # §10.5b: one session, 3 kWh of need, 2 kWh per slot, window slots 0-4.
    # Slot 1 is the cheapest but in a blackout; unmanaged charging there is
    # 1 kWh (2 in slot 0, then 1), which the plan keeps; the other 2 kWh go
    # to the cheapest outside slot (slot 2), exactly the kernel's planner.
    sessions = market.expected_sessions(
        np.array([1.0]), np.array([[3.0]]), np.array([2.0]), np.array([0]), np.array([5])
    )
    prices = np.array([[50.0, 10.0, 20.0, 30.0, 40.0, 60.0]])
    blackout = np.array([False, True, False, False, False, False])
    expected = market.expected_smart_kwh(
        sessions, prices, first_slot=0, non_response_probability=np.zeros(1), blackout=blackout
    )
    np.testing.assert_allclose(expected, [[0.0, 1.0, 2.0, 0.0, 0.0, 0.0]])
    window = np.where(np.arange(6) < 5, prices, np.inf)
    np.testing.assert_allclose(
        expected, plan_around_blackout(np.array([3.0]), np.array([2.0]), window, blackout[None])
    )
    # Without the mask the expectation takes the cheap blackout slot.
    free = market.expected_smart_kwh(
        sessions, prices, first_slot=0, non_response_probability=np.zeros(1)
    )
    np.testing.assert_allclose(free, [[0.0, 2.0, 1.0, 0.0, 0.0, 0.0]])


def test_intraday_rebase_plans_from_the_decision_and_drops_passed_deadlines() -> None:
    # Review B1: a session whose typical start has passed is planned from
    # the decision slot, its need capped on the re-based window; one whose
    # deadline has passed is dropped.
    sessions = market.expected_sessions(
        plug_probability=np.array([1.0, 1.0]),
        need_kwh=np.array([[20.0, 5.0]]),
        slot_cap_kwh=np.array([3.0, 3.0]),
        start=np.array([0, 0]),
        deadline=np.array([8, 3]),
        decision_slot=4,
    )
    assert sessions["start"].tolist() == [4, 4]
    assert sessions["weight"].tolist() == [[1.0, 0.0]]
    assert sessions["need_kwh"][0, 0] == 12.0  # 4 slots x 3 kWh
    expected = market.expected_smart_kwh(
        sessions, np.full((1, 6), 50.0), first_slot=4, non_response_probability=np.zeros(2)
    )
    np.testing.assert_allclose(expected, [[3.0, 3.0, 3.0, 3.0, 0.0, 0.0]])


def test_expected_need_falls_back_to_the_cohort_mean_then_zero() -> None:
    need = market.expected_need_kwh(
        np.array([[10.0, 0.0, 6.0, 0.0]]),
        np.array([[2, 0, 1, 0]]),
        np.array(["a", "a", "a", "b"], dtype=object),
    )
    # EV 1 has no warm-up session: its cohort's mean of 5 and 6.
    np.testing.assert_allclose(need, [[5.0, 5.5, 6.0, 0.0]])


# --------------------------------------------------------------------------
# §4.1 BL01-lite baseline
# --------------------------------------------------------------------------


def _history(warmup_days: int, start: date = _MONDAY) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_warmup_slots(start, warmup_days), build_study_slots(start, 7)


def test_constant_warmup_gives_a_constant_baseline() -> None:
    warmup, study = _history(7)
    baseline, fallback, missing = market.bl01_lite_baseline(
        np.full((2, len(warmup)), 3.5), warmup, study, working_nights=5, non_working_nights=2
    )
    assert np.all(baseline == 3.5) and fallback == 0 and missing == []


def test_baseline_uses_nights_up_to_n_minus_2_of_its_class_only() -> None:
    warmup, study = _history(7)
    # Each warm-up night's import is its own night index, so a slot's
    # baseline names the nights averaged.
    history = warmup["night_index"].to_numpy(dtype=float)[None, :]
    baseline, _, _ = market.bl01_lite_baseline(
        history, warmup, study, working_nights=5, non_working_nights=2
    )
    night = study["night_index"].to_numpy()
    # Night 0 (Monday 12 Oct): history nights <= -2 are -7..-2 (Mon 5 ...
    # Sat 10); working ones are -7..-3 (Mon-Fri): mean -5.
    assert baseline[0, night == 0] == pytest.approx(-5.0)
    # Night 5 (Saturday 17): non-working history <= 3 are Sat 10 (-2) and
    # Sun 11 (-1): mean -1.5.  Study nights never enter in default mode.
    assert baseline[0, night == 5] == pytest.approx(-1.5)


def test_perturbing_night_n_minus_1_leaves_night_n_unchanged() -> None:
    warmup, study = _history(7)
    history = np.random.default_rng(1).uniform(0.0, 5.0, (2, len(warmup)))
    base, _, _ = market.bl01_lite_baseline(
        history, warmup, study, working_nights=5, non_working_nights=2
    )
    changed = history.copy()
    changed[:, warmup["night_index"].to_numpy() == -1] += 100.0
    again, _, _ = market.bl01_lite_baseline(
        changed, warmup, study, working_nights=5, non_working_nights=2
    )
    night = study["night_index"].to_numpy()
    np.testing.assert_array_equal(again[:, night == 0], base[:, night == 0])


def test_short_warmup_uses_the_other_class_and_counts_it() -> None:
    # Three warm-up nights before Saturday 17 Oct (Wed, Thu, Fri): night 0 is
    # a Saturday whose only history nights (<= -2: Wed, Thu) are working.
    start = date(2026, 10, 17)
    warmup, study = _history(3, start)
    history = warmup["night_index"].to_numpy(dtype=float)[None, :]
    baseline, fallback, missing = market.bl01_lite_baseline(
        history, warmup, study, working_nights=5, non_working_nights=2
    )
    assert baseline[0, study["night_index"].to_numpy() == 0] == pytest.approx(-2.5)
    assert fallback >= 1 and missing == []


def test_no_history_night_leaves_a_zero_baseline_and_is_counted() -> None:
    warmup, study = _history(1)
    baseline, _, missing = market.bl01_lite_baseline(
        np.ones((1, len(warmup))), warmup, study, working_nights=5, non_working_nights=2
    )
    assert missing == [0]
    assert np.all(baseline[0, study["night_index"].to_numpy() == 0] == 0.0)


def test_autumn_warmup_keeps_the_london_half_hours() -> None:
    # The warm-up of a study starting Monday 26 Oct holds Sunday 25 Oct's
    # repeated 01:00-02:00; both copies enter the mean of their half-hour.
    start = date(2026, 10, 26)
    warmup, study = _history(7, start)
    history = warmup["local_half_hour"].to_numpy(dtype=float)[None, :]
    baseline, _, _ = market.bl01_lite_baseline(
        history, warmup, study, working_nights=5, non_working_nights=2
    )
    np.testing.assert_allclose(baseline[0], study["local_half_hour"].to_numpy(dtype=float))


def test_in_day_adjustment_shifts_each_night_by_the_window_mean() -> None:
    study = build_study_slots(_MONDAY, 7)
    baseline = np.full((1, len(study)), 2.0)
    metered = np.zeros((1, len(study)))
    first = np.flatnonzero(study["night_index"].to_numpy() == 3)[:3]
    metered[0, first] = [2.0, 3.0, 4.0]  # mean M - B0 = 1
    adjustment = market.in_day_adjustment(baseline, metered, study, window_slots=3)
    night = study["night_index"].to_numpy()
    assert np.all(adjustment[0, night == 3] == 1.0)
    assert np.all(adjustment[0, night == 2] == -2.0)


def test_trap_mode_matches_default_early_and_moves_later_nights_to_the_flexed_import() -> None:
    constant = 4.0
    default = _tiny_run()
    trap = _tiny_run(terms={"trading.baseline_trap": 1.0}, metered_value=constant)
    flat = _tiny_run(metered_value=constant)
    night = build_study_slots(_MONDAY, 7)["night_index"].to_numpy()
    np.testing.assert_array_equal(
        trap.baseline_unadjusted[:, night <= 1], flat.baseline_unadjusted[:, night <= 1]
    )
    np.testing.assert_array_equal(
        default.baseline_unadjusted[:, night <= 1], flat.baseline_unadjusted[:, night <= 1]
    )
    # Night 2 (Wednesday) reads history nights <= 0.  The 3-day warm-up
    # holds one working night (Friday 9 Oct), which is all the default mode
    # has; trap mode adds study night 0 (Monday) at the constant metered
    # import, so the mean moves half-way to it.
    warm = flat.baseline_unadjusted[:, night == 2]
    expected = (warm + constant) / 2.0
    np.testing.assert_allclose(trap.baseline_unadjusted[:, night == 2], expected)


# --------------------------------------------------------------------------
# run_trading on a tiny SYNTHETIC fleet
# --------------------------------------------------------------------------


def _settings(worlds: int, evs: int, warmup_days: int = 3) -> RunSettings:
    return RunSettings(
        start_local_date=_MONDAY,
        warmup_days=warmup_days,
        study_days=7,
        vehicle_count=evs,
        seed=1,
        evaluation_world_count=worlds,
        opening_soc_fraction=0.8,
        reserve_soc_fraction=0.1,
        home_charge_efficiency=0.9,
        weather_efficiency_sensitivity_fraction_per_c=0.0,
    )


def _units(evs: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [f"ev-{i}" for i in range(evs)],
            "cohort_id": ["average_uk"] * evs,
            "plug_probability": np.ones(evs),
            "home_charger_limit_kw": np.full(evs, 7.0),
            "runtime_arrival_local_hour": np.full(evs, 18.0),
            "runtime_departure_local_hour": np.full(evs, 7.0),
        }
    )


def _run_half_hours() -> np.ndarray:
    warmup = build_warmup_slots(_MONDAY, 3)
    study = build_study_slots(_MONDAY, 7)
    return np.concatenate([warmup["local_half_hour"], study["local_half_hour"]]).astype(float)


def _default_day_ahead(worlds: int) -> np.ndarray:
    """SYNTHETIC day-ahead prices (world, run slot): an 18:00 peak, one level per world."""

    price = 60.0 + 40.0 * np.cos(2 * np.pi * (_run_half_hours() / 2.0 - 18.0) / 24.0)
    return np.tile(price, (worlds, 1)) + np.arange(worlds)[:, None]


def _widen(per_slot: dict[str, np.ndarray], worlds: int = 1) -> dict[str, np.ndarray]:
    """Repeat a (1, 4) array over the study (336 slots) for night-0 tests."""

    out = {}
    for key, values in per_slot.items():
        full = np.zeros((worlds, 336))
        full[:, :4] = values
        out[key] = full
    return out


def _tiny_run(
    *,
    worlds: int = 1,
    evs: int = 2,
    terms: dict[str, float] | None = None,
    metered_value: float | None = None,
    **overrides: object,
) -> market.TradingRun:
    """``run_trading`` on SYNTHETIC arrays: evening-peaked imports and prices."""

    settings = _settings(worlds, evs)
    study = build_study_slots(_MONDAY, 7)
    warmup = build_warmup_slots(_MONDAY, 3)
    run_starts = pd.DatetimeIndex(
        utc_half_hour_boundaries(_MONDAY, settings.warmup_days, settings.study_days)[:-1]
    )
    run_count = len(run_starts)
    half = _run_half_hours()
    # SYNTHETIC evening-peaked import shape.
    shape = np.where((half >= 36) & (half < 44), 5.0, 0.5)
    day_ahead = _default_day_ahead(worlds)
    unmanaged = np.tile(shape[len(warmup) :], (worlds, 1))
    metered = (
        np.full_like(unmanaged, metered_value)
        if metered_value is not None
        else np.tile(np.roll(shape[len(warmup) :], 16), (worlds, 1))
    )
    minute = pd.DatetimeIndex(study["interval_start_london"]).minute
    book_slots = np.flatnonzero(np.asarray(minute) == 0)
    path = np.repeat(day_ahead[:, :, None], 36, axis=2)
    inputs: dict[str, object] = {
        "study_slots": study,
        "warmup_slots": warmup,
        "units": _units(evs),
        "unmanaged_kwh": unmanaged,
        "metered_kwh": metered,
        "warmup_unmanaged_kwh": np.tile(shape[: len(warmup)], (worlds, 1)),
        "expected_need": np.full((worlds, evs), 10.0),
        "departure_shortfall_kwh": {
            "normal": np.zeros((worlds, 336)),
            "selected": np.zeros((worlds, 336)),
        },
        "book_kwh": np.zeros((worlds, len(book_slots), market.BOOK_WIDTH)),
        "book_slots": book_slots,
        "connected": np.zeros((worlds, 337, evs), dtype=bool),
        "expected_departure_utc_ns": expected_departures_utc_ns(settings, _units(evs), 2.0),
        "day_ahead_gbp_per_mwh": day_ahead,
        "run_interval_start_utc": run_starts,
        "intraday_path_gbp_per_mwh": path,
        "intraday_close_gbp_per_mwh": path[:, len(warmup) :, 0].copy(),
        "imbalance_gbp_per_mwh": day_ahead[:, len(warmup) :] + 15.0,
        "known_shock_gw": np.zeros((worlds, 336)),
        "surprise_shock_gw": np.zeros((worlds, 336)),
        "non_response_probability": np.zeros(evs),
        "assumptions": _TERMS | (terms or {}),
        "sampled_world_ids": list(range(min(worlds, 10))),
    }
    inputs.update(overrides)
    assert run_count == day_ahead.shape[1]
    return market.run_trading(**inputs)


def test_tiny_run_ledger_reconciles_and_perfect_foresight_has_no_imbalance() -> None:
    run = _tiny_run(worlds=3)
    ledger = run.trading_ledger_world
    assert list(ledger.columns) == list(market.LEDGER_COLUMNS)
    residual = ledger["net_gbp"] - ledger[list(market.BUCKETS)].sum(axis=1)
    assert residual.abs().max() <= 1e-12
    assert (ledger.loc[ledger["strategy"].eq("perfect_foresight"), "imbalance_gbp"] == 0.0).all()
    assert (ledger.loc[ledger["strategy"].eq("da_only"), "intraday_pnl_gbp"] == 0.0).all()
    week = run.trading_week_world
    assert np.allclose(
        week.set_index(["world_id", "strategy"])["net_gbp"].sort_index(),
        ledger.groupby(["world_id", "strategy"])["net_gbp"].sum().sort_index(),
    )


def test_zero_need_and_zero_import_earn_nothing_in_every_strategy() -> None:
    # §5.10 "no flex, no earnings" (review B4): B = U = M = 0 and no need, so
    # B0 = x = 0 at every decision.
    zeros = np.zeros((1, 336))
    run = _tiny_run(
        unmanaged_kwh=zeros,
        metered_kwh=zeros,
        warmup_unmanaged_kwh=np.zeros((1, 144)),
        expected_need=np.zeros((1, 2)),
    )
    money = run.trading_ledger_world[list(market.MONEY_COLUMNS)]
    assert (money == 0.0).all().all()


def test_equal_prices_everywhere_give_equal_net_for_every_strategy_and_share() -> None:
    # §5.10 (review B4): intraday = SIP = day-ahead and no spread, so the
    # cash of every slot is V x P whatever was traded.
    for share in (0.5, 0.8, 1.0):
        base = _tiny_run(worlds=2, terms={"trading.half_spread_gbp_per_mwh": 0.0})
        run = _tiny_run(
            worlds=2,
            terms={
                "trading.half_spread_gbp_per_mwh": 0.0,
                "trading.day_ahead_commitment_share": share,
            },
            imbalance_gbp_per_mwh=base.deviation_world_slot["day_ahead_gbp_per_mwh"]
            .to_numpy()
            .reshape(2, 336),
        )
        net = run.trading_week_world.pivot(index="world_id", columns="strategy", values="net_gbp")
        np.testing.assert_allclose(net["full"], net["perfect_foresight"], rtol=1e-12)
        np.testing.assert_allclose(net["da_only"], net["perfect_foresight"], rtol=1e-12)


def test_commitment_share_scales_the_day_ahead_position_and_leaves_revised_finals() -> None:
    runs = {c: _tiny_run(terms={"trading.day_ahead_commitment_share": c}) for c in (0.0, 0.5, 1.0)}
    np.testing.assert_allclose(
        runs[0.5].day_ahead_position, 0.5 * runs[1.0].day_ahead_position, atol=1e-12
    )
    assert np.all(runs[0.0].day_ahead_position == 0.0)
    revised = runs[1.0].revised_slots
    np.testing.assert_allclose(
        runs[0.5].positions["full"][:, revised], runs[1.0].positions["full"][:, revised]
    )
    # The 12:00 and 12:30 slots have no intraday decision (§9.2).
    first_two = ~revised
    np.testing.assert_allclose(
        runs[0.5].positions["full"][:, first_two], 0.5 * runs[1.0].positions["full"][:, first_two]
    )
    ledger = runs[0.0].trading_ledger_world
    da_only = ledger.loc[ledger["strategy"].eq("da_only")]
    # c = 0: day-ahead only settles all of V at imbalance.
    assert np.allclose(da_only["imbalance_mwh"], da_only["settled_mwh"])


def test_intraday_price_is_the_latest_update_at_or_before_the_decision() -> None:
    # Review B6: the path's step h holds 1000 + h, so the price paid names
    # the step; it must be ceil((gate - decision) / 1 h).
    base = _tiny_run()
    path = np.broadcast_to(1000.0 + np.arange(36), (1, 480, 36)).copy()
    run = _tiny_run(intraday_path_gbp_per_mwh=path)
    updates = run.position_updates.query("stage == 'intraday'")
    gate = updates["interval_start_utc"] - pd.Timedelta(minutes=60)
    hours = (gate - updates["decision_utc"]) / pd.Timedelta(hours=1)
    np.testing.assert_array_equal(updates["hours_to_gate_closure"], np.ceil(hours).astype(int))
    np.testing.assert_array_equal(updates["price_gbp_per_mwh"], 1000.0 + np.ceil(hours))
    assert (updates["decision_utc"] <= gate).all()
    assert len(base.position_updates) == len(run.position_updates)


def test_positions_at_a_decision_ignore_everything_after_it() -> None:
    # Leakage (§5.10): change the connection states, books and metered
    # import after a decision instant; every target set at or before it is
    # unchanged, and so is every day-ahead position.
    base = _tiny_run()
    study = build_study_slots(_MONDAY, 7)
    decision_slot = int(np.flatnonzero(study["local_time_label"].eq("18:00").to_numpy())[2])
    connected = np.zeros((1, 337, 2), dtype=bool)
    connected[:, decision_slot + 1 :, :] = True
    minute = pd.DatetimeIndex(study["interval_start_london"]).minute
    book_slots = np.flatnonzero(np.asarray(minute) == 0)
    book = np.zeros((1, len(book_slots), market.BOOK_WIDTH))
    book[:, book_slots > decision_slot, :] = 9.0
    metered = base.deviation_world_slot["metered_kwh"].to_numpy().reshape(1, 336).copy()
    metered[:, decision_slot:] += 3.0
    changed = _tiny_run(connected=connected, book_kwh=book, metered_kwh=metered)
    tau = study["interval_start_utc"].iat[decision_slot]

    def up_to_tau(run: market.TradingRun) -> pd.DataFrame:
        updates = run.position_updates
        return updates.loc[updates["decision_utc"] <= tau].reset_index(drop=True)

    before, after = up_to_tau(base), up_to_tau(changed)
    assert before["stage"].eq("intraday").sum() > 0
    pd.testing.assert_frame_equal(before, after)
    # The later decisions do see the change.
    assert not base.position_updates.equals(changed.position_updates)
    np.testing.assert_array_equal(base.day_ahead_position, changed.day_ahead_position)


def test_day_ahead_position_ignores_unpublished_day_ahead_prices() -> None:
    # Night n's day-ahead decision (13:00 on D_n - 1) sees D_n's prices only;
    # D_n + 1's are on the typical shape until 13:00 on D_n.
    day_ahead = _default_day_ahead(1)
    base = _tiny_run(day_ahead_gbp_per_mwh=day_ahead)
    study = build_study_slots(_MONDAY, 7)
    london = pd.DatetimeIndex(study["interval_start_london"])
    night = study["night_index"].to_numpy()
    morning = np.flatnonzero((night == 2) & (np.asarray(london.hour) < 12))
    perturbed = day_ahead.copy()
    perturbed[:, 144 + morning] += 500.0
    run = _tiny_run(day_ahead_gbp_per_mwh=perturbed)
    np.testing.assert_array_equal(
        run.day_ahead_position[:, night == 2], base.day_ahead_position[:, night == 2]
    )
    # The perturbed prices are visible to night 3's decision, so it can move.
    assert not np.array_equal(run.visible_day_ahead_gbp_per_mwh, base.visible_day_ahead_gbp_per_mwh)


def test_trading_mask_seam_pauses_positions_and_settlement() -> None:
    closed = np.ones(336, dtype=bool)
    closed[36:40] = False
    run = _tiny_run(trading_mask=lambda known_at_utc_ns: closed)
    assert np.all(run.day_ahead_position[:, 36:40] == 0.0)
    assert np.all(run.positions["full"][:, 36:40] == 0.0)
    assert np.all(run.settled[:, 36:40] == 0.0)
    assert not run.deviation_world_slot["settlement_open"].to_numpy()[36:40].any()


def test_shock_kind_lets_a_surprise_win() -> None:
    kind = market.shock_kind(np.array([[1.0, 1.0, 0.0, 0.0]]), np.array([[1.0, 0.0, -2.0, 0.0]]))
    assert kind.tolist() == [["surprise", "known", "surprise", "none"]]


# --------------------------------------------------------------------------
# §9.5 control group
# --------------------------------------------------------------------------


def test_control_group_is_stratified_by_cohort_by_largest_remainder() -> None:
    cohorts = ["a"] * 6 + ["b"] * 3 + ["c"] * 1
    units = pd.DataFrame({"cohort_id": cohorts})
    order = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    control = market.control_group_mask(units, order, 0.35, ["a", "b", "c"])
    # K = round(3.5) = 4: quotas 2.4, 1.2, 0.4 -> 2, 1, 0 then the largest
    # remainder (0.4, a tie between a and c, goes to the earlier cohort a).
    assert control.sum() == 4
    # First k_c EVs of each cohort in permutation order.
    assert control.tolist() == [False, False, False, True, True, True, False, False, True, False]


def test_control_group_needs_one_ev_in_each_group() -> None:
    units = pd.DataFrame({"cohort_id": ["a", "a"]})
    with pytest.raises(ValueError, match="at least one EV"):
        market.control_group_mask(units, np.array([0, 1]), 0.1, ["a"])


# --------------------------------------------------------------------------
# §9.4 the supplier's own price curve
# --------------------------------------------------------------------------


def _curve_csv(values: np.ndarray, *, header: str = "half_hour_start,price_gbp_per_mwh") -> str:
    labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    lines = [header, *(f"{label},{value}" for label, value in zip(labels, values, strict=True))]
    return "\n".join(lines)


def test_user_curve_validation_accepts_48_values_and_names_each_broken_rule() -> None:
    limits = {"floor_gbp_per_mwh": -500.0, "cap_gbp_per_mwh": 4000.0}
    curve = market.validate_user_price_curve(_curve_csv(np.arange(48.0)), **limits)
    assert curve["price_gbp_per_mwh"].tolist() == list(np.arange(48.0))
    assert curve["half_hour_start"].iat[36] == "18:00"
    text = _curve_csv(np.arange(48.0))
    with pytest.raises(ValueError, match="48 rows"):
        market.validate_user_price_curve("\n".join(text.split("\n")[:-1]), **limits)
    with pytest.raises(ValueError, match="repeated"):
        market.validate_user_price_curve(text.replace("18:30", "18:00"), **limits)
    with pytest.raises(ValueError, match="finite"):
        market.validate_user_price_curve(text.replace("\n18:00,36.0", "\n18:00,nan"), **limits)
    with pytest.raises(ValueError, match="header"):
        market.validate_user_price_curve(
            _curve_csv(np.arange(48.0), header="half_hour_start,price_gbp_per_mwh,extra"), **limits
        )
    with pytest.raises(ValueError, match="outside"):
        market.validate_user_price_curve(_curve_csv(np.full(48, 5000.0)), **limits)


def _synthetic_prices(
    start: date = _MONDAY, worlds: int = 2
) -> tuple[MarketPrices, pd.DatetimeIndex]:
    starts = pd.DatetimeIndex(utc_half_hour_boundaries(start, 3, 7)[:-1])
    count = len(starts)
    rng = np.random.default_rng(11)
    day_ahead = rng.uniform(20.0, 150.0, (worlds, count))
    path = day_ahead[:, :, None] + rng.normal(0.0, 5.0, (worlds, count, 36))
    imbalance = path[:, :, 0] + rng.normal(0.0, 10.0, (worlds, count))
    keys = {
        "world_id": np.repeat(np.arange(worlds), count),
        "interval_start_utc": np.tile(starts, worlds),
    }
    prices = MarketPrices(
        day_ahead=pd.DataFrame({**keys, "wholesale_forecast_gbp_per_mwh": day_ahead.reshape(-1)}),
        realised=pd.DataFrame(
            {
                **keys,
                "evaluation_context_price_gbp_per_mwh": path[:, :, 0].reshape(-1),
                "imbalance_price_gbp_per_mwh": imbalance.reshape(-1),
            }
        ),
        shocks=pd.DataFrame(),
        known_shock_gw=np.zeros((worlds, count)),
        surprise_shock_gw=np.zeros((worlds, count)),
        intraday_path_gbp_per_mwh=path,
    )
    return prices, starts


def test_a_curve_equal_to_the_typical_shape_changes_nothing() -> None:
    prices, starts = _synthetic_prices()
    shape = market.typical_day_ahead_shape(
        prices.day_ahead["wholesale_forecast_gbp_per_mwh"].to_numpy().reshape(2, -1), starts
    )
    # One curve cannot equal two day-class shapes, so the weekday offset is
    # checked on weekdays: it is exactly zero there.
    curve = pd.DataFrame({"half_hour_start": [""] * 48, "price_gbp_per_mwh": shape[0]})
    shifted = market.apply_user_price_curve(
        prices, curve, starts, floor_gbp_per_mwh=-500.0, cap_gbp_per_mwh=4000.0
    )
    weekday = np.asarray(starts.tz_convert("Europe/London").dayofweek < 5)
    before = prices.intraday_path_gbp_per_mwh[:, weekday]
    after = shifted.intraday_path_gbp_per_mwh[:, weekday]
    np.testing.assert_array_equal(before, after)


def test_a_constant_curve_offset_moves_every_price_by_the_same_amount() -> None:
    prices, starts = _synthetic_prices()
    day_ahead = prices.day_ahead["wholesale_forecast_gbp_per_mwh"].to_numpy().reshape(2, -1)
    shape = market.typical_day_ahead_shape(day_ahead, starts)
    weekday = np.asarray(starts.tz_convert("Europe/London").dayofweek < 5)
    curve = pd.DataFrame({"half_hour_start": [""] * 48, "price_gbp_per_mwh": shape[0] + 25.0})
    shifted = market.apply_user_price_curve(
        prices, curve, starts, floor_gbp_per_mwh=-500.0, cap_gbp_per_mwh=4000.0
    )
    new_day_ahead = shifted.day_ahead["wholesale_forecast_gbp_per_mwh"].to_numpy().reshape(2, -1)
    np.testing.assert_allclose((new_day_ahead - day_ahead)[:, weekday], 25.0)
    np.testing.assert_allclose(
        (shifted.intraday_path_gbp_per_mwh - prices.intraday_path_gbp_per_mwh)[:, weekday], 25.0
    )
    sip = "imbalance_price_gbp_per_mwh"
    diff = (shifted.realised[sip] - prices.realised[sip]).to_numpy().reshape(2, -1)
    np.testing.assert_allclose(diff[:, weekday], 25.0)
    close = shifted.realised["evaluation_context_price_gbp_per_mwh"].to_numpy().reshape(2, -1)
    np.testing.assert_array_equal(close, shifted.intraday_path_gbp_per_mwh[:, :, 0])


def test_a_curve_past_the_cap_leaves_prices_at_the_cap() -> None:
    prices, starts = _synthetic_prices()
    curve = pd.DataFrame({"half_hour_start": [""] * 48, "price_gbp_per_mwh": np.full(48, 3990.0)})
    shifted = market.apply_user_price_curve(
        prices, curve, starts, floor_gbp_per_mwh=-500.0, cap_gbp_per_mwh=4000.0
    )
    assert shifted.intraday_path_gbp_per_mwh.max() == 4000.0
    assert (shifted.day_ahead["wholesale_forecast_gbp_per_mwh"] <= 4000.0).all()


@pytest.mark.parametrize("start", [date(2026, 10, 22), date(2027, 3, 24)])
def test_curve_offsets_follow_the_london_label_across_a_clock_change(start: date) -> None:
    prices, starts = _synthetic_prices(start)
    curve = pd.DataFrame({"half_hour_start": [""] * 48, "price_gbp_per_mwh": np.arange(48.0)})
    shifted = market.apply_user_price_curve(
        prices, curve, starts, floor_gbp_per_mwh=-500.0, cap_gbp_per_mwh=4000.0
    )
    day_ahead = prices.day_ahead["wholesale_forecast_gbp_per_mwh"].to_numpy().reshape(2, -1)
    offset = (
        shifted.day_ahead["wholesale_forecast_gbp_per_mwh"].to_numpy().reshape(2, -1) - day_ahead
    )[0]
    london = starts.tz_convert("Europe/London")
    label = np.asarray(london.hour * 2 + london.minute // 30)
    weekend = np.asarray(london.dayofweek >= 5).astype(int)
    shape = market.typical_day_ahead_shape(day_ahead, starts)
    np.testing.assert_allclose(offset, np.arange(48.0)[label] - shape[weekend, label])
    # Every copy of a label (a repeated autumn half-hour) moves by one value.
    for h in np.unique(label[weekend == 0]):
        assert np.ptp(offset[(label == h) & (weekend == 0)]) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Events: planner signals in the expected plan (§2.4, §4.4)
# --------------------------------------------------------------------------


def test_expected_plan_ranks_known_request_signals_by_zone() -> None:
    # One session per zone, same prices; a +500 turn-down signal on slot 0
    # in zone 1 only moves that zone's session to slot 1.
    sessions = market.expected_sessions(
        plug_probability=np.array([1.0, 1.0]),
        need_kwh=np.array([[2.0, 2.0]]),
        slot_cap_kwh=np.array([2.0, 2.0]),
        start=np.array([0, 0]),
        deadline=np.array([2, 2]),
    )
    adjustment = np.array([[0.0, 0.0], [500.0, 0.0]])  # (zone, slot)
    expected = market.expected_smart_kwh(
        sessions,
        np.array([[50.0, 60.0]]),
        first_slot=0,
        non_response_probability=np.zeros(2),
        price_adjustment_gbp_per_mwh=adjustment,
        zone_index=np.array([0, 1]),
    )
    np.testing.assert_allclose(expected, [[2.0, 2.0]])
    plain = market.expected_smart_kwh(
        sessions, np.array([[50.0, 60.0]]), first_slot=0, non_response_probability=np.zeros(2)
    )
    np.testing.assert_allclose(plain, [[4.0, 0.0]])


def _dfs_events() -> pd.DataFrame:
    study = build_study_slots(_MONDAY, 7)
    return model_events.validate_events(
        model_events.event_table(("dfs_turn_down",), study), study, assumptions.ZONE_IDS
    )


def test_a_day_ahead_request_pauses_its_window_and_is_paid_on_its_night() -> None:
    study = build_study_slots(_MONDAY, 7)
    events = _dfs_events()
    zones = len(assumptions.ZONE_IDS)
    base = _tiny_run(worlds=2)
    # SYNTHETIC zones: the fleet's import split evenly over the four zones.
    warm_half = _run_half_hours()[:144]
    warm = np.where((warm_half >= 36) & (warm_half < 44), 5.0, 0.5)
    metered = base.deviation_world_slot["metered_kwh"].to_numpy().reshape(2, 336)
    run = _tiny_run(
        worlds=2,
        events=events,
        zone_warmup_kwh=np.broadcast_to(warm / zones, (2, zones, 144)).copy(),
        zone_metered_kwh=np.repeat(metered[:, None, :] / zones, zones, axis=1),
    )
    windows = model_events.event_slots(events, study)
    first, end = int(windows["start_slot"].iat[0]), int(windows["end_slot"].iat[0])
    # Known at night 1's day-ahead decision, so no position and no settlement.
    assert np.all(run.day_ahead_position[:, first:end] == 0.0)
    assert np.all(run.positions["full"][:, first:end] == 0.0)
    assert np.all(run.settled[:, first:end] == 0.0)
    ledger = run.trading_ledger_world
    paid = ledger.groupby("night_index")["grid_event_payment_gbp"].sum()
    delivery = run.event_delivery
    assert paid.drop(1).eq(0.0).all()
    # Every strategy books the same payment: three times the delivery frame.
    assert paid[1] == pytest.approx(3 * delivery["payment_gbp"].sum())
    assert run.scope_baseline.shape == (2, 1 + zones, 336)


def test_grid_requests_need_zone_baselines() -> None:
    with pytest.raises(ValueError, match="zone baselines"):
        _tiny_run(events=_dfs_events())


def _expected_columns(run: market.TradingRun, worlds: int) -> tuple[np.ndarray, np.ndarray]:
    frame = run.deviation_world_slot
    return (
        frame["expected_unmanaged_kwh"].to_numpy().reshape(worlds, 336),
        frame["expected_metered_kwh"].to_numpy().reshape(worlds, 336),
    )


def test_expected_unmanaged_is_every_expected_session_unmanaged_and_moves_no_energy() -> None:
    # Supplier contract §2: x^U is the day-ahead expected sessions with
    # rho = 1, i.e. each session's unmanaged schedule times its weight, so
    # F = x^U - x only moves the same clipped need in time.
    worlds, evs = 2, 2
    run = _tiny_run(worlds=worlds, evs=evs)
    x_u, x = _expected_columns(run, worlds)
    study = build_study_slots(_MONDAY, 7)
    night = study["night_index"].to_numpy()
    units = _units(evs)
    start, deadline = market.typical_session_windows(
        units, study, expected_departures_utc_ns(_settings(worlds, evs), units, 2.0)
    )
    cap = np.full(evs, 3.5)
    for n in range(7):
        slots = np.flatnonzero(night == n)
        in_window = (slots >= start[n][:, None]) & (slots < deadline[n][:, None])
        need = np.minimum(10.0, cap * in_window.sum(axis=1))
        recomputed = market.unmanaged_schedule_kwh(need, cap, in_window).sum(axis=0)
        for w in range(worlds):
            np.testing.assert_allclose(x_u[w, slots], recomputed, atol=1e-12)
        np.testing.assert_allclose((x_u - x)[:, slots].sum(axis=1), 0.0, atol=1e-9)
    assert np.abs(x_u - x).max() > 1.0  # the planner does move charging
    # No baseline or realised import enters either column (T§5.10 leakage).
    for other in (
        _tiny_run(worlds=worlds, evs=evs, terms={"trading.baseline_trap": 1.0}),
        _tiny_run(worlds=worlds, evs=evs, metered_value=4.0),
    ):
        again_u, again_x = _expected_columns(other, worlds)
        np.testing.assert_array_equal(again_u, x_u)
        np.testing.assert_array_equal(again_x, x)


def test_night_zero_day_ahead_need_uses_only_sessions_known_by_its_decision() -> None:
    # tau_DA(0) is 13:00 on D_0 - 1, inside warm-up night -1 (review B3):
    # night 0's day-ahead forecast takes the need of sessions that began
    # before it; later nights and every intraday decision take the full one.
    base = _tiny_run()
    cut = _tiny_run(first_night_expected_need=np.zeros((1, 2)))
    night = build_study_slots(_MONDAY, 7)["night_index"].to_numpy()
    base_u, base_x = _expected_columns(base, 1)
    cut_u, cut_x = _expected_columns(cut, 1)
    assert base_x[:, night == 0].sum() > 0.0
    assert np.all(cut_x[:, night == 0] == 0.0) and np.all(cut_u[:, night == 0] == 0.0)
    np.testing.assert_array_equal(cut_x[:, night > 0], base_x[:, night > 0])
    np.testing.assert_array_equal(cut_u[:, night > 0], base_u[:, night > 0])
    intraday = cut.position_updates["stage"].eq("intraday")
    pd.testing.assert_frame_equal(
        cut.position_updates.loc[intraday, ["slot_index", "forecast_metered_kwh"]],
        base.position_updates.loc[intraday, ["slot_index", "forecast_metered_kwh"]],
    )


def test_night_zero_cutoff_is_the_first_run_slot_at_its_day_ahead_decision() -> None:
    settings = _settings(1, 2)
    study = build_study_slots(_MONDAY, 7)
    kernel, _ = forecast._trading_kernel_outputs(settings, study, np.zeros(2, dtype=bool), 13.0)
    cutoff = int(kernel["normal"]["first_day_ahead_cutoff_slot"])
    run_starts = pd.DatetimeIndex(utc_half_hour_boundaries(_MONDAY, 3, 7)[:-1])
    # 13:00 London (BST) on Sunday 11 Oct, 23 hours before the study start.
    assert run_starts[cutoff] == pd.Timestamp("2026-10-11 12:00", tz="UTC")
    assert cutoff == 3 * 48 - 46


def test_a_night_with_no_baseline_history_gets_no_in_day_adjustment() -> None:
    study = build_study_slots(_MONDAY, 7)
    baseline = np.zeros((1, len(study)))
    metered = np.ones((1, len(study)))
    adjustment = market.in_day_adjustment(
        baseline, metered, study, window_slots=3, missing_nights=[0]
    )
    night = study["night_index"].to_numpy()
    assert np.all(adjustment[0, night == 0] == 0.0)
    assert np.all(adjustment[0, night == 1] == 1.0)
    # A one-night warm-up leaves night 0 without history; B stays 0 there.
    day_ahead = _default_day_ahead(1)[:, 96:]
    run = _tiny_run(
        warmup_slots=build_warmup_slots(_MONDAY, 1),
        warmup_unmanaged_kwh=np.ones((1, 48)),
        day_ahead_gbp_per_mwh=day_ahead,
        run_interval_start_utc=pd.DatetimeIndex(utc_half_hour_boundaries(_MONDAY, 1, 7)[:-1]),
        intraday_path_gbp_per_mwh=np.repeat(day_ahead[:, :, None], 36, axis=2),
        metered_value=2.0,
    )
    assert run.missing_nights == 1
    assert np.all(run.baseline[:, night == 0] == 0.0)
    assert np.all(run.settled[:, night == 0] == 0.0)


# --------------------------------------------------------------------------
# Decision 0004 item 67: baseline erosion (a baseline learnt from smart nights)
# --------------------------------------------------------------------------


def test_flexed_baseline_learns_a_repeated_smart_night_exactly() -> None:
    # SYNTHETIC: every night of a class imports the same profile, so the
    # other nights of the class reproduce it; weekends differ from weekdays.
    study = build_study_slots(_MONDAY, 7)
    weekday = study["day_type"].eq("weekday").to_numpy()
    metered = np.where(weekday, 1.0, 3.0) + study["local_half_hour"].to_numpy() / 48.0
    baseline, missing = market.flexed_history_baseline(
        metered[None, :], study, working_nights=5, non_working_nights=2
    )
    assert missing == []
    assert np.allclose(baseline[0], metered)


def test_flexed_baseline_keeps_the_night_itself_out() -> None:
    study = build_study_slots(_MONDAY, 7)
    night = study["night_index"].to_numpy()
    metered = np.zeros((1, len(study)))
    metered[0, night == 5] = 4.0  # Saturday; its only same-class night is Sunday
    baseline, _ = market.flexed_history_baseline(
        metered, study, working_nights=5, non_working_nights=2
    )
    assert np.all(baseline[0, night == 5] == 0.0)
    assert np.all(baseline[0, night == 6] == 4.0)


def test_erosion_is_positive_when_smart_moves_evening_import() -> None:
    # SYNTHETIC _tiny_run: unmanaged peaks 18:00-22:00, smart runs the same
    # profile 8 h later on every night.  The default baseline (unmanaged
    # warm-up) settles the evening turn-down; one learnt from the smart
    # nights sees no turn-down at all.
    from axle_studio.model.summaries import baseline_erosion_per_world

    run = _tiny_run(worlds=2)
    day_ahead = _default_day_ahead(2)[:, 144:]
    erosion = baseline_erosion_per_world(
        run.settled, run.settled_flexed_baseline, day_ahead, build_study_slots(_MONDAY, 7)
    )
    assert (run.settled_flexed_baseline >= 0.0).all()
    for prefix in ("", "evening_"):
        default = erosion[f"{prefix}settled_value_gbp"]
        eroded = erosion[f"{prefix}settled_value_eroded_gbp"]
        assert (default > 0.0).all() and (eroded <= default).all()
        share = 1.0 - eroded / default
        assert (share >= 0.0).all() and np.allclose(share, 1.0)


def test_evening_erosion_counts_only_16_00_to_20_00_london() -> None:
    from axle_studio.model.summaries import EROSION_EVENING_WINDOW, baseline_erosion_per_world

    assert EROSION_EVENING_WINDOW == ("16:00", "20:00")
    study = build_study_slots(_MONDAY, 7)
    labels = study["local_time_label"].to_numpy()
    settled = np.zeros((1, len(study)))
    # SYNTHETIC: 1 kWh at 15:30, 16:00, 19:30 and 20:00 on every night.
    settled[0, np.isin(labels, ["15:30", "16:00", "19:30", "20:00"])] = 1.0
    price = np.full_like(settled, 1000.0)  # GBP 1 per kWh
    erosion = baseline_erosion_per_world(settled, np.zeros_like(settled), price, study)
    assert erosion["settled_value_gbp"][0] == 4.0 * 7
    assert erosion["evening_settled_value_gbp"][0] == 2.0 * 7
    assert erosion["evening_settled_value_eroded_gbp"][0] == 0.0


def test_no_erosion_when_smart_equals_unmanaged() -> None:
    # Smart = unmanaged and every night alike: both baselines learn the same
    # profile, so the smart-night what-if settles exactly what the default does.
    run = _tiny_run(
        metered_kwh=np.tile(
            np.where((_run_half_hours()[144:] >= 36) & (_run_half_hours()[144:] < 44), 5.0, 0.5),
            (1, 1),
        )
    )
    assert np.array_equal(run.settled_flexed_baseline, run.settled)


def test_erosion_what_if_leaves_the_ledger_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    # The what-if is computed beside settlement, never fed into it: a wildly
    # different smart-night baseline changes no ledger, position or deviation.
    before = _tiny_run(worlds=2)
    monkeypatch.setattr(
        market,
        "flexed_history_baseline",
        # A shaped (not constant) baseline: the in-day adjustment would
        # cancel a constant one.
        lambda metered, *_, **__: (np.random.default_rng(0).uniform(0, 99, metered.shape), []),
    )
    after = _tiny_run(worlds=2)
    pd.testing.assert_frame_equal(before.trading_ledger_world, after.trading_ledger_world)
    pd.testing.assert_frame_equal(before.deviation_world_slot, after.deviation_world_slot)
    pd.testing.assert_frame_equal(before.position_updates, after.position_updates)
    assert not np.array_equal(before.settled_flexed_baseline, after.settled_flexed_baseline)


def test_validator_recomputes_the_erosion_values_from_the_deviation_frame() -> None:
    from fixtures.result_contract import baseline_erosion_from

    from axle_studio.model.summaries import baseline_erosion_per_world

    run = _tiny_run(worlds=2)
    study = build_study_slots(_MONDAY, 7)
    model = baseline_erosion_per_world(
        run.settled, run.settled_flexed_baseline, _default_day_ahead(2)[:, 144:], study
    )
    reference = baseline_erosion_from(
        run.deviation_world_slot,
        study,
        working_nights=int(_TERMS["trading.baseline_working_nights"]),
        non_working_nights=int(_TERMS["trading.baseline_non_working_nights"]),
        window_slots=int(_TERMS["trading.baseline_adjustment_window_slots"])
        if _TERMS["trading.baseline_in_day_adjustment"]
        else None,
    )
    assert np.allclose(reference["settled_value_gbp_per_week"], model["settled_value_gbp"])
    assert np.allclose(reference["settled_mwh_eroded_per_week"], model["settled_mwh_eroded"])
    for name in (
        "settled_value_eroded_gbp",
        "evening_settled_value_gbp",
        "evening_settled_value_eroded_gbp",
    ):
        assert np.allclose(reference[f"{name}_per_week"], model[name]), name
