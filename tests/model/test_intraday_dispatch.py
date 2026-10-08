"""Intraday dispatch in the kernel: locked and free EVs, the latest price, the re-plan rule.

Intraday dispatch contract v1 §3-§5 and the §8 sanity rows 1-8 and 10-12
(lane K1; decision 0004 item 62(b)).  Pure functions first, then hand-checkable
one-EV kernel cases (10 kWh battery, 4 kW charger = 2 kWh per half-hour, 100 %
efficiency, 8 kWh target, expected departure 06:00, GMT so London time is
UTC), then small SYNTHETIC real runs for the fleet invariants.  Every price is
synthetic and every value illustrative.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date
from functools import cache

import numpy as np
import pandas as pd
import pytest

from axle_studio.model import assumptions
from axle_studio.model.action import (
    intraday_dispatch_inputs,
    latest_known_prices,
    locked_evs,
    replan_when_worth,
    smart_charging_inputs,
)
from axle_studio.model.forecast import COHORT_ORDER, simulate_forecast
from axle_studio.model.physics import (
    BASE_NON_RESPONSE,
    CONTROL_GROUP,
    CONTROL_OUTAGE_EVENT,
    MAKER_OUTAGE,
    PLAN_FOLLOWS,
    DispatchSums,
    PublicTopUp,
    simulate_fleet_intervals,
    simulate_unit_intervals,
)
from axle_studio.model.settings import RunSettings

_HOUR_NS = 3600 * 1_000_000_000
_STEPS = 36
_TOP_UP = PublicTopUp(efficiency_fraction=0.9, threshold_soc_fraction=0.1, target_soc_fraction=0.8)
_WINTER = date(2025, 1, 7)  # GMT, so London clock time equals UTC
_LONDON_NOON = pd.Timestamp("2025-01-07 12:00", tz="UTC")


# --------------------------------------------------------------------------
# §3 latest_known_prices
# --------------------------------------------------------------------------


def test_latest_known_price_reads_only_updates_made_and_prices_published() -> None:
    tau = 10 * _HOUR_NS
    # Four slots: past gate closure, gate 2.5 h ahead, gate 100 h ahead
    # (beyond the path's 36 steps) and one not yet published.
    gate = np.array([tau - _HOUR_NS, tau + 5 * _HOUR_NS // 2, tau + 100 * _HOUR_NS, tau])
    publication = np.array([0, 0, 0, tau + 1])
    rng = np.random.default_rng(7)
    path = rng.normal(50.0, 10.0, (2, 4, _STEPS))
    visible = rng.normal(50.0, 10.0, (2, 4))
    latest = latest_known_prices(path, visible, publication, gate, tau)
    np.testing.assert_array_equal(latest[:, 0], path[:, 0, 0])  # the close
    np.testing.assert_array_equal(latest[:, 1], path[:, 1, 3])  # ceil(2.5) = 3
    np.testing.assert_array_equal(latest[:, 2], path[:, 2, _STEPS - 1])
    np.testing.assert_array_equal(latest[:, 3], visible[:, 3])  # B4 expected shape
    # Updates not yet made (h below the latest made), every step of an
    # unpublished slot and the day-ahead price of a published slot are not
    # read.
    perturbed_path, perturbed_visible = path.copy(), visible.copy()
    perturbed_path[:, 1, :3] += 1000.0
    perturbed_path[:, 2, : _STEPS - 1] += 1000.0
    perturbed_path[:, 3, :] += 1000.0
    perturbed_visible[:, :3] += 1000.0
    np.testing.assert_array_equal(
        latest_known_prices(perturbed_path, perturbed_visible, publication, gate, tau), latest
    )


# --------------------------------------------------------------------------
# §4.1 the re-plan rule (§8 rows 3 and 6)
# --------------------------------------------------------------------------


def _replan(plan, price, cap=5.0, threshold=10.0, half_spread=1.0):
    plan = np.atleast_2d(np.asarray(plan, dtype=float))
    price = np.atleast_2d(np.asarray(price, dtype=float))
    return replan_when_worth(
        plan,
        price,
        np.full(len(plan), cap),
        replan_threshold_gbp_per_mwh=threshold,
        half_spread_gbp_per_mwh=half_spread,
    )


@pytest.mark.parametrize(
    ("incumbent", "price", "expected", "replanned"),
    [
        # §4.2 worked example: r = 6, cap 5, bar GBP 12/MWh moved.
        ([5, 1, 0, 0], [30, 400, 80, 100], [5, 0, 1, 0], True),  # GBP 320/MWh moved
        ([5, 1, 0, 0], [400, 40, 80, 100], [0, 5, 1, 0], True),  # still 6 kWh in time
        ([5, 1, 0, 0], [30, 40, 38, 100], [5, 1, 0, 0], False),  # GBP 2/MWh moved
        ([5, 5, 5, 5], [30, 400, 80, 100], [5, 5, 5, 5], False),  # no slack
    ],
)
def test_the_worked_example_of_the_re_plan_rule(incumbent, price, expected, replanned) -> None:
    new, changed = _replan(incumbent, price)
    np.testing.assert_allclose(new[0], expected)
    assert bool(changed[0]) is replanned
    assert new.sum() == pytest.approx(sum(incumbent))


def test_the_bar_is_the_threshold_plus_twice_the_half_spread() -> None:
    # 2 kWh can move from a GBP 15/MWh slot to a GBP 10/MWh one: GBP 5/MWh moved.
    incumbent, price = [2, 0], [15, 10]
    # theta = s = 0: any strictly cheaper plan wins.
    assert _replan(incumbent, price, threshold=0.0, half_spread=0.0)[1][0]
    assert not _replan(incumbent, [15, 15], threshold=0.0, half_spread=0.0)[1][0]
    # The bar rises by exactly 2 per unit of s: 2 x 2.4 < 5 < 2 x 2.6.
    assert _replan(incumbent, price, threshold=0.0, half_spread=2.4)[1][0]
    assert not _replan(incumbent, price, threshold=0.0, half_spread=2.6)[1][0]
    assert _replan(incumbent, price, threshold=4.0, half_spread=0.4)[1][0]
    assert not _replan(incumbent, price, threshold=4.0, half_spread=0.6)[1][0]
    # A GBP 2/MWh saving with s = 1 pays exactly the spread: no re-plan.
    assert not _replan(incumbent, [12, 10], threshold=0.0, half_spread=1.0)[1][0]
    # A very large threshold stops every re-plan.
    assert not _replan([5, 1, 0, 0], [30, 4000, 80, 100], threshold=1e9)[1][0]


def test_unrankable_slots_keep_their_energy_and_are_skipped_in_the_saving() -> None:
    # Slot 0 is already past and slot 2 a blackout (both +inf): their energy
    # stays; the 3 kWh from slot 1 moves to the cheaper slot 3.
    new, changed = _replan([2, 3, 1, 0], [np.inf, 400, np.inf, 20])
    assert changed[0]
    np.testing.assert_allclose(new[0], [2, 0, 1, 3])
    # Nothing to rank: nothing moves and no NaN appears.
    new, changed = _replan([2, 0], [np.inf, np.inf])
    assert not changed[0] and np.isfinite(new).all()


# --------------------------------------------------------------------------
# §2 locked EVs (§8 row 2)
# --------------------------------------------------------------------------


def _cohort_units(cohorts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"cohort_id": cohorts})


def test_locked_count_is_half_up_split_by_largest_remainder() -> None:
    # The defaults' cohort order is the model's cohort order.
    assert tuple(assumptions.COHORT_SHARES) == COHORT_ORDER
    units = _cohort_units(["average_uk"] * 2 + ["intelligent_octopus"] * 3)
    # c = 0.5, N = 5: K = round-half-up(2.5) = 3; quotas 1.2 and 1.8, so the
    # spare EV goes to the larger remainder.
    np.testing.assert_array_equal(locked_evs(units, 0.5), [True, False, True, True, False])
    # Equal remainders tie to the earlier cohort in the defaults' order, not
    # the row order: K = 1 of 2 + 2.
    units = _cohort_units(["intelligent_octopus"] * 2 + ["average_uk"] * 2)
    np.testing.assert_array_equal(locked_evs(units, 0.25), [False, False, True, False])
    assert not locked_evs(units, 0.0).any()
    assert locked_evs(units, 1.0).all()
    with pytest.raises(ValueError, match="fraction"):
        locked_evs(units, 1.5)


# --------------------------------------------------------------------------
# §5.1 inputs and decision instants (§8 row 8)
# --------------------------------------------------------------------------


def _clock_change_settings() -> RunSettings:
    # The study week spans the autumn clock change (Sunday 25 October 2026).
    return replace(_settings(), start_local_date=date(2026, 10, 22), warmup_days=2, study_days=7)


def test_decision_instants_are_whole_hours_from_the_last_warm_up_night() -> None:
    settings = _clock_change_settings()
    slot_count = 48 * 9
    dispatch = intraday_dispatch_inputs(
        settings,
        pd.DataFrame({"dispatch_locked": [False]}),
        np.zeros((1, slot_count, _STEPS)),
        replan_threshold_gbp_per_mwh=10.0,
        half_spread_gbp_per_mwh=1.0,
        gate_closure_minutes=60,
    )
    decisions = np.flatnonzero(dispatch.decision_slot)
    # None before the last warm-up night; every one on a whole UTC hour,
    # across the clock change too (British offsets are whole hours).
    assert decisions.min() == 48
    decision_ns = dispatch.gate_utc_ns[decisions] + _HOUR_NS
    assert (decision_ns % _HOUR_NS == 0).all()
    # Every whole hour from there on is one: the run has 48 UTC half-hours
    # a day whatever the clock does, so 8 x 24 of them.
    assert len(decisions) == 8 * 24
    with pytest.raises(ValueError, match="dispatch_locked"):
        intraday_dispatch_inputs(
            settings,
            pd.DataFrame({"cohort_id": ["average_uk"]}),
            np.zeros((1, slot_count, _STEPS)),
            replan_threshold_gbp_per_mwh=10.0,
            half_spread_gbp_per_mwh=1.0,
            gate_closure_minutes=60,
        )
    with pytest.raises(ValueError, match="threshold"):
        intraday_dispatch_inputs(
            settings,
            pd.DataFrame({"dispatch_locked": [False]}),
            np.zeros((1, slot_count, _STEPS)),
            replan_threshold_gbp_per_mwh=-1.0,
            half_spread_gbp_per_mwh=1.0,
            gate_closure_minutes=60,
        )


# --------------------------------------------------------------------------
# One EV, hand-checkable kernel cases
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


def _units(locked: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population_id": ["population:dispatch"],
            "unit_id": ["ev-001"],
            "cohort_id": ["average_uk"],
            "physical_capacity_kwh": [10.0],
            "efficiency_miles_per_battery_kwh": [1.0],
            "home_charger_limit_kw": [4.0],
            "preferred_target_soc_fraction": [0.8],
            # Typical departure 07:00, expected 06:00 with a one-hour margin.
            "runtime_departure_local_hour": [7],
            "runtime_arrival_local_hour": [18],
            "dispatch_locked": [locked],
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


def _run_slot(instant: str) -> int:
    return 48 + _slot(instant)


def _day_ahead(prices: dict[str, float]) -> np.ndarray:
    """Flat GBP 100/MWh day-ahead prices with the given night slots overridden."""

    values = np.full((1, 48 * 3), 100.0)
    for instant, price in prices.items():
        values[0, _run_slot(f"2025-01-08 {instant}")] = price
    return values


def _path(day_ahead: np.ndarray) -> np.ndarray:
    """An intraday path with no news: every step is the day-ahead price."""

    return np.repeat(day_ahead[:, :, np.newaxis], _STEPS, axis=2)


def _news(path: np.ndarray, slot: str, known_from: str, price: float) -> None:
    """Set slot ``slot``'s intraday price to ``price`` from the update made at ``known_from``.

    Gate closure is 60 minutes before the slot starts; the update made at
    ``tau`` is step ``h = ceil((gate - tau) / 1 h)``, so every step up to it
    carries the news.
    """

    start = pd.Timestamp(f"2025-01-08 {slot}", tz="UTC")
    gate = start - pd.Timedelta(minutes=60)
    tau = pd.Timestamp(known_from, tz="UTC")
    step = max(0, math.ceil((gate - tau) / pd.Timedelta(hours=1)))
    path[0, _run_slot(f"2025-01-08 {slot}"), : step + 1] = price


def _dispatch(settings, path, *, locked=False, threshold=10.0, half_spread=1.0):
    return intraday_dispatch_inputs(
        settings,
        _units(locked),
        path,
        replan_threshold_gbp_per_mwh=threshold,
        half_spread_gbp_per_mwh=half_spread,
        gate_closure_minutes=60,
    )


def _per_ev(settings, inputs, day_ahead, path=None, *, locked=False, threshold=10.0, **smart):
    """The one EV's selected-path outputs, (study slot,) each; ``path`` None: no dispatch."""

    smart_inputs = smart_charging_inputs(
        settings, _units(), day_ahead, departure_margin_hours=1.0, **smart
    )
    dispatch = (
        None if path is None else _dispatch(settings, path, locked=locked, threshold=threshold)
    )
    unit = simulate_unit_intervals(
        settings,
        _units(locked),
        inputs,
        public_top_up=_TOP_UP,
        smart_charging=smart_inputs,
        intraday_dispatch=dispatch,
    )
    return {name: values[0, :, 0] for name, values in unit.items()}


def _nonzero(values: np.ndarray) -> dict[int, float]:
    return {int(slot): float(values[slot]) for slot in np.flatnonzero(np.abs(values) > 1e-12)}


def _book_decisions(settings: RunSettings) -> np.ndarray:
    """Each run slot's plan-book row: every whole hour of the study, else -1."""

    rows = np.full(48 * (settings.warmup_days + settings.study_days), -1, dtype=np.int64)
    whole_hours = np.arange(48 * settings.warmup_days, len(rows), 2)
    rows[whole_hours] = np.arange(len(whole_hours))
    return rows


def _fleet(settings, inputs, day_ahead, path, *, locked=False, **smart):
    """A fleet run with dispatch: (DispatchSums, fleet frame)."""

    decisions = _book_decisions(settings)
    trading = {
        "selected": {
            "book_decision": decisions,
            "book_kwh": np.zeros((1, int(decisions.max()) + 1, 48)),
        }
    }
    sums = DispatchSums()
    fleet = simulate_fleet_intervals(
        settings,
        _units(locked),
        inputs,
        public_top_up=_TOP_UP,
        smart_charging=smart_charging_inputs(
            settings, _units(), day_ahead, departure_margin_hours=1.0, **smart
        ),
        trading_output=trading,
        intraday_dispatch=_dispatch(settings, path, locked=locked),
        dispatch_sums=sums,
    )[0]
    return sums, fleet


def _book_row(instant: str) -> int:
    return _slot(instant) // 2


_EVENING = ("2025-01-07 18:00", "2025-01-08 07:00")
# Need 2 kWh (6 of 8 kWh at plug-in): one slot.  Cheapest day-ahead half-hour
# 03:00, next 04:00.
_CHEAP_NIGHT = {"03:00": 10.0, "04:00": 20.0}


def _spike_path() -> np.ndarray:
    # From the update made at 21:00, 03:00 spikes to GBP 400/MWh.
    path = _path(_day_ahead(_CHEAP_NIGHT))
    _news(path, "03:00", "2025-01-07 21:00", 400.0)
    return path


def test_a_free_ev_turns_down_out_of_a_spike_and_still_reaches_target() -> None:
    settings = _settings()
    inputs = _inputs(settings, *_EVENING)
    day_ahead = _day_ahead(_CHEAP_NIGHT)
    free = _per_ev(settings, inputs, day_ahead, _spike_path())
    locked = _per_ev(settings, inputs, day_ahead, _spike_path(), locked=True)
    without = _per_ev(settings, inputs, day_ahead)
    three, four = _slot("2025-01-08 03:00"), _slot("2025-01-08 04:00")
    # The locked EV, the switch off and the free EV before the news agree:
    # 2 kWh at 03:00.  At 21:00 the free EV moves it to 04:00.
    assert _nonzero(without["home_grid_import_kwh"]) == {three: 2.0}
    for name in without:
        np.testing.assert_array_equal(locked[name], without[name], err_msg=name)
    assert _nonzero(free["home_grid_import_kwh"]) == {four: 2.0}
    assert _nonzero(free["planned_home_import_kwh"]) == {four: 2.0}
    # The target is reached by the expected departure, from the same stock.
    closing = free["closing_battery_kwh"][_slot("2025-01-08 05:30")]
    assert closing == pytest.approx(8.0)
    # The plan's status and the need under the plan are the incumbent's: a
    # re-plan is not a plan made (B6).  The need no longer falls at 03:00
    # but at 04:00, where the energy now is.
    np.testing.assert_array_equal(free["plan_status"], without["plan_status"])
    need, need_without = free["plan_remaining_need_kwh"], without["plan_remaining_need_kwh"]
    np.testing.assert_array_equal(need[: three + 1], need_without[: three + 1])
    assert need[four] == pytest.approx(2.0) and need[four + 1] == 0.0
    # A very large threshold stops the re-plan.
    stopped = _per_ev(settings, inputs, day_ahead, _spike_path(), threshold=1e9)
    np.testing.assert_array_equal(stopped["home_grid_import_kwh"], without["home_grid_import_kwh"])


def test_the_re_plan_is_counted_at_its_decision_and_kept_by_the_books() -> None:
    settings = _settings()
    sums, fleet = _fleet(
        settings, _inputs(settings, *_EVENING), _day_ahead(_CHEAP_NIGHT), _spike_path()
    )
    replans = sums.replan_count[0]
    assert _nonzero(replans.astype(float)) == {_slot("2025-01-07 21:00"): 1.0}
    selected = fleet.loc[fleet["path_id"].eq("selected"), "home_grid_import_kwh"].to_numpy()
    np.testing.assert_array_equal(sums.free_home_import_kwh[0], selected)
    assert not sums.locked_home_import_kwh.any()
    # The day-ahead plan path is the switch-off selected path.
    assert _nonzero(sums.day_ahead_home_import_kwh[0]) == {_slot("2025-01-08 03:00"): 2.0}
    np.testing.assert_array_equal(
        sums.day_ahead_zone_home_import_kwh[:, 0], sums.day_ahead_home_import_kwh
    )
    np.testing.assert_array_equal(
        sums.free_day_ahead_home_import_kwh, sums.day_ahead_home_import_kwh
    )
    # Book order (B1, §8 row 10).  The plug-in at 18:00 is planned after the
    # 18:00 snapshot, so the 18:00 book is empty and the 19:00 book holds it.
    book, reference = sums.book_kwh["selected"][0], sums.book_kwh["day_ahead"][0]
    assert not book[_book_row("2025-01-07 18:00")].any()
    seven_hours = 14
    assert _nonzero(book[_book_row("2025-01-07 20:00")]) == {seven_hours: 2.0}
    # The book at 21:00 already holds the re-plan made at 21:00 (04:00 is
    # seven hours ahead); the day-ahead plan path's book keeps 03:00.
    assert _nonzero(book[_book_row("2025-01-07 21:00")]) == {seven_hours: 2.0}
    assert _nonzero(reference[_book_row("2025-01-07 21:00")]) == {seven_hours - 2: 2.0}


def test_with_no_slack_the_ev_charges_through_the_spike() -> None:
    # Plug in at 04:30 needing 6 kWh (2 of 8): the window to the 06:00
    # expected departure (04:30, 05:00, 05:30) is exactly full.
    settings = _settings(opening=0.2)
    inputs = _inputs(settings, "2025-01-08 04:30", "2025-01-08 07:00")
    day_ahead = _day_ahead({})
    path = _path(day_ahead)
    _news(path, "05:30", "2025-01-08 04:00", 400.0)
    free = _per_ev(settings, inputs, day_ahead, path, threshold=0.0)
    window = [_slot(f"2025-01-08 {t}") for t in ("04:30", "05:00", "05:30")]
    assert _nonzero(free["home_grid_import_kwh"]) == dict.fromkeys(window, 2.0)
    assert free["closing_battery_kwh"][window[-1]] == pytest.approx(8.0)
    sums, _ = _fleet(settings, inputs, day_ahead, path)
    assert not sums.replan_count.any()


def test_turn_up_pulls_energy_into_a_negative_price_up_to_the_need_only() -> None:
    settings = _settings()
    inputs = _inputs(settings, *_EVENING)
    day_ahead = _day_ahead(_CHEAP_NIGHT)
    path = _path(day_ahead)
    # From 22:00 two half-hours after midnight turn negative.
    _news(path, "00:00", "2025-01-07 22:00", -50.0)
    _news(path, "00:30", "2025-01-07 22:00", -50.0)
    free = _per_ev(settings, inputs, day_ahead, path)
    # Still only the 2 kWh need, in the first (earlier) of the equal slots:
    # never past the preferred target.
    assert _nonzero(free["home_grid_import_kwh"]) == {_slot("2025-01-08 00:00"): 2.0}
    assert free["closing_battery_kwh"].max() == pytest.approx(8.0)
    # A turn-up request announced at 21:30 does the same through the ranking
    # price: GBP 200/MWh off at 01:00.
    adjustment = np.zeros((1, 1, 48 * 3))
    adjustment[0, 0, _run_slot("2025-01-08 01:00")] = -200.0
    request = {
        "event_notice_slot": np.array([_run_slot("2025-01-07 21:30")]),
        "event_adjustment_gbp_per_mwh": adjustment,
    }
    turned_up = _per_ev(settings, inputs, day_ahead, _path(day_ahead), **request)
    assert _nonzero(turned_up["home_grid_import_kwh"]) == {_slot("2025-01-08 01:00"): 2.0}
    # The day-ahead plan path does not see a request made after its plan.
    kept = _per_ev(settings, inputs, day_ahead, **request)
    assert _nonzero(kept["home_grid_import_kwh"]) == {_slot("2025-01-08 03:00"): 2.0}
    # A session already at its target has nothing to move.
    full_settings = _settings(opening=0.8)
    at_target = _per_ev(full_settings, _inputs(full_settings, *_EVENING), day_ahead, path)
    assert not at_target["home_grid_import_kwh"].any()


@pytest.mark.parametrize(("spike", "charged_at"), [(220.0, "05:00"), (2000.0, "05:30")])
def test_a_request_and_a_spike_compete_on_one_scale(spike: float, charged_at: str) -> None:
    # Plug in at 03:00 needing 2 kWh; the cheapest half-hour to the 06:00
    # expected departure is 05:30 (10), then 05:00 (20).  From the 04:00
    # update 05:00 spikes; a DFS turn-down request announced at 04:30 pays
    # GBP 500/MWh not to charge at 05:30.  At the 05:00 re-plan: 05:30 ranks
    # 510; 05:00 ranks 220 (move there, out of the window) or 2,000 (stay in
    # the window and forgo the payment).
    settings = _settings()
    inputs = _inputs(settings, "2025-01-08 03:00", "2025-01-08 07:00")
    day_ahead = _day_ahead({"05:00": 20.0, "05:30": 10.0})
    path = _path(day_ahead)
    _news(path, "05:00", "2025-01-08 04:00", spike)
    adjustment = np.zeros((1, 1, 48 * 3))
    adjustment[0, 0, _run_slot("2025-01-08 05:30")] = 500.0
    free = _per_ev(
        settings,
        inputs,
        day_ahead,
        path,
        event_notice_slot=np.array([_run_slot("2025-01-08 04:30")]),
        event_adjustment_gbp_per_mwh=adjustment,
    )
    assert _nonzero(free["home_grid_import_kwh"]) == {_slot(f"2025-01-08 {charged_at}"): 2.0}


def test_a_half_hour_plug_in_ranks_the_price_known_at_that_half_hour() -> None:
    # 03:30 (gate 02:30) drops to GBP 5/MWh from the update made at 18:30:
    # h = ceil(8) = 8 then, but h = 9 at 18:00.
    day_ahead = _day_ahead(_CHEAP_NIGHT)
    path = _path(day_ahead)
    _news(path, "03:30", "2025-01-07 18:30", 5.0)
    settings = _settings()
    half_past = _per_ev(
        settings, _inputs(settings, "2025-01-07 18:30", "2025-01-08 07:00"), day_ahead, path
    )
    assert _nonzero(half_past["home_grid_import_kwh"]) == {_slot("2025-01-08 03:30"): 2.0}
    # Plugged in at 18:00 it plans 03:00; the GBP 5/MWh it sees from 19:00
    # saves GBP 5/MWh moved, under the GBP 12 bar, so it stays.
    on_the_hour = _per_ev(settings, _inputs(settings, *_EVENING), day_ahead, path)
    assert _nonzero(on_the_hour["home_grid_import_kwh"]) == {_slot("2025-01-08 03:00"): 2.0}


def test_the_decision_snapshot_holds_the_re_plans_made_at_it_and_no_later_one() -> None:
    settings = _settings()
    path = _spike_path()
    # A second spike, at 04:00 from the 23:00 update, moves the plan again.
    _news(path, "04:00", "2025-01-07 23:00", 400.0)
    decision = np.array([_run_slot("2025-01-07 21:00"), _run_slot("2025-01-08 21:00")])
    free = _per_ev(
        settings,
        _inputs(settings, *_EVENING),
        _day_ahead(_CHEAP_NIGHT),
        path,
        decision_slot_by_night=decision,
    )
    # 21:00 re-plan to 04:00, 23:00 re-plan to the earliest GBP 100 slot, 23:00.
    assert _nonzero(free["home_grid_import_kwh"]) == {_slot("2025-01-07 23:00"): 2.0}
    assert _nonzero(free["decision_plan_kwh"]) == {_slot("2025-01-08 04:00"): 2.0}


def test_a_re_plan_reads_nothing_about_connection_in_its_own_slot() -> None:
    settings = _settings()
    day_ahead = _day_ahead(_CHEAP_NIGHT)
    stays, _ = _fleet(settings, _inputs(settings, *_EVENING), day_ahead, _spike_path())
    # The EV unplugs at 21:10, inside the 21:00 slot: the 21:00 re-plan and
    # book are the same; the plan is voided afterwards (step 1).
    leaves, fleet = _fleet(
        settings,
        _inputs(settings, "2025-01-07 18:00", "2025-01-07 21:10"),
        day_ahead,
        _spike_path(),
    )
    row = _book_row("2025-01-07 21:00")
    np.testing.assert_array_equal(
        leaves.book_kwh["selected"][0, row], stays.book_kwh["selected"][0, row]
    )
    np.testing.assert_array_equal(leaves.replan_count, stays.replan_count)
    assert not leaves.book_kwh["selected"][0, row + 1].any()
    assert fleet["early_departure_count"].sum() == 1


def test_a_re_plan_keeps_blackout_energy_and_never_moves_into_a_blackout() -> None:
    # Plug in at 18:30 needing 6 kWh.  The unmanaged trajectory charges
    # 18:30, 19:00 and 19:30; 19:30 is a blackout, so its 2 kWh stay there
    # (§10.5b).  02:00 is the cheapest half-hour but also a blackout.  The
    # plan: 2 kWh at 19:30, 2 at 03:00, 2 at 04:00.  From 19:00, 03:00 spikes:
    # the re-plan moves that 2 kWh to 05:00, keeping 19:30 and avoiding 02:00.
    settings = _settings(opening=0.2)
    day_ahead = _day_ahead({"02:00": 1.0, "03:00": 10.0, "04:00": 20.0, "05:00": 30.0})
    blackout = np.zeros(48 * 3, dtype=bool)
    blackout[[_run_slot("2025-01-07 19:30"), _run_slot("2025-01-08 02:00")]] = True
    path = _path(day_ahead)
    _news(path, "03:00", "2025-01-07 19:00", 400.0)
    inputs = _inputs(settings, "2025-01-07 18:30", "2025-01-08 07:00")
    without = _per_ev(settings, inputs, day_ahead, blackout=blackout)
    free = _per_ev(settings, inputs, day_ahead, path, blackout=blackout)
    slots = {t: _slot(f"2025-01-{t}") for t in ("07 19:30", "08 03:00", "08 04:00", "08 05:00")}
    assert _nonzero(without["home_grid_import_kwh"]) == {
        slots["07 19:30"]: 2.0,
        slots["08 03:00"]: 2.0,
        slots["08 04:00"]: 2.0,
    }
    assert _nonzero(free["home_grid_import_kwh"]) == {
        slots["07 19:30"]: 2.0,
        slots["08 04:00"]: 2.0,
        slots["08 05:00"]: 2.0,
    }


@pytest.mark.parametrize(
    ("case", "status"),
    [
        ({"control": True, "rho": 1.0, "base": 0.05, "uniform": 0.99}, CONTROL_GROUP),
        ({"rho": 0.5, "base": 0.5, "uniform": 0.3}, BASE_NON_RESPONSE),
        ({"rho": 1.0, "base": 0.05, "uniform": 0.3}, MAKER_OUTAGE),
        ({"rho": 0.05, "base": 0.05, "uniform": 0.3, "outage": 0.5}, CONTROL_OUTAGE_EVENT),
        ({"rho": 0.05, "base": 0.05, "uniform": 0.9}, PLAN_FOLLOWS),
    ],
)
def test_only_a_session_that_follows_its_plan_is_re_planned(case, status) -> None:
    settings = _settings()
    inputs = _inputs(settings, *_EVENING)
    day_ahead = _day_ahead(_CHEAP_NIGHT)
    outage = np.zeros((1, 48 * 3))
    outage[0, _run_slot("2025-01-07 18:00")] = case.get("outage", 0.0)
    smart = {
        "outage_probability": outage,
        "non_response_uniform": np.full((1, 2, 1), case["uniform"]),
        "non_response": np.full((1, 2, 1), case["rho"]),
        "base_non_response": np.array([case["base"]]),
        "control_group": np.array([case.get("control", False)]),
    }
    without = _per_ev(settings, inputs, day_ahead, **smart)
    free = _per_ev(settings, inputs, day_ahead, _spike_path(), **smart)
    first_plan = slice(_slot("2025-01-07 18:00"), _slot("2025-01-08 06:00"))
    assert (free["plan_status"][first_plan] == status).all()
    # The status (and so the uniform's reading) is the same with dispatch.
    np.testing.assert_array_equal(free["plan_status"], without["plan_status"])
    if status == PLAN_FOLLOWS:
        assert _nonzero(free["planned_home_import_kwh"]) == {_slot("2025-01-08 04:00"): 2.0}
    else:
        # Never re-planned: the plan on record is the plug-in plan, and the
        # EV charges by the normal rule, at plug-in.
        for name in without:
            np.testing.assert_array_equal(free[name], without[name], err_msg=name)
        assert _nonzero(free["planned_home_import_kwh"]) == {_slot("2025-01-08 03:00"): 2.0}
        assert _nonzero(free["home_grid_import_kwh"]) == {_slot("2025-01-07 18:00"): 2.0}


def test_dispatch_needs_smart_charging() -> None:
    settings = _settings()
    with pytest.raises(ValueError, match="needs smart charging"):
        simulate_unit_intervals(
            settings,
            _units(),
            _inputs(settings, *_EVENING),
            public_top_up=_TOP_UP,
            intraday_dispatch=_dispatch(settings, _spike_path()),
        )


# --------------------------------------------------------------------------
# Real runs (SYNTHETIC, small)
# --------------------------------------------------------------------------

_START = date(2026, 10, 12)  # a Monday, no clock change
_SMALL = {"vehicle_count": 40, "evaluation_world_count": 3}


@cache
def _simulate(**edits: float):
    values = _SMALL | edits
    settings = assumptions.run_settings(values, _START)
    inputs = assumptions.forecast_inputs(
        assumptions.resolve_values(values), warmup_days=7, study_days=7
    )
    return simulate_forecast(settings, assumptions.cohort_fixture(values), **inputs)


def _real_dispatch(simulated, share: float, *, threshold=10.0, half_spread=1.0, path=None):
    units = simulated.units.assign(dispatch_locked=locked_evs(simulated.units, share))
    dispatch = intraday_dispatch_inputs(
        simulated.settings,
        units,
        simulated.market_prices.intraday_path_gbp_per_mwh if path is None else path,
        replan_threshold_gbp_per_mwh=threshold,
        half_spread_gbp_per_mwh=half_spread,
        gate_closure_minutes=simulated.trading_assumptions["gate_closure_minutes"],
    )
    return units, dispatch


def _real_units(simulated, units, smart=None, dispatch=None) -> dict[str, np.ndarray]:
    return simulate_unit_intervals(
        simulated.settings,
        units,
        simulated.evaluation.inputs,
        public_top_up=simulated.public_top_up,
        effective_efficiency_miles_per_battery_kwh=(
            simulated.evaluation.effective_efficiency_miles_per_battery_kwh
        ),
        smart_charging=simulated.smart_charging if smart is None else smart,
        intraday_dispatch=dispatch,
    )


def _real_fleet(simulated, units, smart=None, dispatch=None):
    """(fleet frames, DispatchSums, selected plan book) of one fleet run."""

    selected = simulated.trading_kernel["selected"]
    trading = {
        "selected": {
            "book_decision": selected["book_decision"],
            "book_kwh": np.zeros_like(selected["book_kwh"]),
        }
    }
    sums = DispatchSums()
    frames = simulate_fleet_intervals(
        simulated.settings,
        units,
        simulated.evaluation.inputs,
        public_top_up=simulated.public_top_up,
        evaluation_effective_efficiency_miles_per_battery_kwh=(
            simulated.evaluation.effective_efficiency_miles_per_battery_kwh
        ),
        smart_charging=simulated.smart_charging if smart is None else smart,
        trading_output=trading,
        intraday_dispatch=dispatch,
        dispatch_sums=sums,
    )
    return frames, sums, trading["selected"]["book_kwh"]


def test_locked_evs_behave_as_on_the_day_ahead_plan_path_and_free_ones_re_plan() -> None:
    simulated = _simulate()
    units, dispatch = _real_dispatch(simulated, 0.5, threshold=0.0, half_spread=0.0)
    locked = units["dispatch_locked"].to_numpy()
    assert locked.sum() == 20
    reference = _real_units(simulated, units)
    dispatched = _real_units(simulated, units, dispatch=dispatch)
    # EVs do not interact, so every locked EV's frames are identical on the
    # two smart paths; free EVs follow the intraday price.
    for name in reference:
        np.testing.assert_array_equal(
            dispatched[name][..., locked], reference[name][..., locked], err_msg=name
        )
    assert not np.array_equal(
        dispatched["home_grid_import_kwh"][..., ~locked],
        reference["home_grid_import_kwh"][..., ~locked],
    )
    # The fleet sums reconcile with the per-EV frames.
    (fleet, *_), sums, _ = _real_fleet(simulated, units, dispatch=dispatch)
    selected = fleet.loc[fleet["path_id"].eq("selected"), "home_grid_import_kwh"].to_numpy()
    np.testing.assert_allclose(
        selected.reshape(3, -1), dispatched["home_grid_import_kwh"].sum(axis=2), atol=1e-9
    )
    np.testing.assert_allclose(
        sums.locked_home_import_kwh + sums.free_home_import_kwh, selected.reshape(3, -1), atol=1e-9
    )
    np.testing.assert_allclose(
        sums.locked_home_import_kwh + sums.free_day_ahead_home_import_kwh,
        sums.day_ahead_home_import_kwh,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        sums.day_ahead_home_import_kwh,
        reference["home_grid_import_kwh"].sum(axis=2),
        atol=1e-9,
    )
    assert sums.replan_count.sum() > 0
    assert (sums.replan_count <= (~locked).sum()).all()
    # Re-plans happen at whole hours only.
    study_decision = dispatch.decision_slot[48 * simulated.settings.warmup_days :]
    assert not sums.replan_count[:, ~study_decision].any()


def test_switch_off_and_every_ev_locked_leave_the_selected_path_unchanged() -> None:
    simulated = _simulate()
    units, dispatch = _real_dispatch(simulated, 1.0, threshold=0.0, half_spread=0.0)
    (off, *_), none, off_book = _real_fleet(simulated, units)
    (on, *_), sums, on_book = _real_fleet(simulated, units, dispatch=dispatch)
    # Without dispatch the holder is untouched: no third pass runs.
    assert none.day_ahead_home_import_kwh is None and none.book_kwh == {}
    pd.testing.assert_frame_equal(on, off)
    np.testing.assert_array_equal(on_book, off_book)
    np.testing.assert_array_equal(sums.book_kwh["day_ahead"], off_book)
    assert not sums.replan_count.any() and not sums.free_home_import_kwh.any()


def test_no_news_makes_the_dispatched_path_the_day_ahead_plan_path() -> None:
    # §8 row 1: no hourly updates, no shocks and no premium, so every step
    # of the intraday path is the day-ahead price.
    simulated = _simulate(
        intraday_hourly_sd_gbp_per_mwh=0.0,
        mild_shock_rate_per_night=0.0,
        big_shock_rate_per_week=0.0,
        da_id_premium_gbp_per_mwh=0.0,
    )
    path = simulated.market_prices.intraday_path_gbp_per_mwh
    day_ahead = (
        simulated.forecast_prices["wholesale_forecast_gbp_per_mwh"].to_numpy().reshape(3, -1)
    )
    np.testing.assert_array_equal(path, np.repeat(day_ahead[..., np.newaxis], path.shape[2], 2))
    units, dispatch = _real_dispatch(simulated, 0.0, threshold=0.0, half_spread=0.0)
    (off, *_), _, off_book = _real_fleet(simulated, units)
    (on, *_), sums, on_book = _real_fleet(simulated, units, dispatch=dispatch)
    pd.testing.assert_frame_equal(on, off)
    np.testing.assert_array_equal(on_book, off_book)
    assert not sums.replan_count.any()


def test_conservation_one_route_and_physical_soc_with_every_ev_free() -> None:
    simulated = _simulate()
    units, dispatch = _real_dispatch(simulated, 0.0, threshold=0.0, half_spread=0.0)
    unit = _real_units(simulated, units, dispatch=dispatch)
    home, public = unit["home_grid_import_kwh"], unit["public_grid_import_kwh"]
    assert not ((home > 0.0) & (public > 0.0)).any()
    charging = home > 0.0
    added = unit["closing_battery_kwh"] - unit["opening_battery_kwh"]
    efficiency = simulated.settings.home_charge_efficiency
    np.testing.assert_allclose(added[charging], home[charging] * efficiency, atol=1e-9)
    capacity = simulated.units["physical_capacity_kwh"].to_numpy()
    target = capacity * simulated.units["preferred_target_soc_fraction"].to_numpy()
    assert (unit["closing_battery_kwh"] >= -1e-12).all()
    assert (unit["closing_battery_kwh"] <= capacity + 1e-12).all()
    # Home charging never lifts a battery past its preferred target.
    assert (
        unit["closing_battery_kwh"][charging]
        <= np.broadcast_to(target, home.shape)[charging] + 1e-9
    ).all()
    (fleet, *_), sums, _ = _real_fleet(simulated, units, dispatch=dispatch)
    assert fleet["conservation_residual_kwh"].abs().max() <= 1e-12
    # Pairing: the same inputs give the same dispatch, and nothing is drawn.
    (again, *_), repeat, _ = _real_fleet(simulated, units, dispatch=dispatch)
    pd.testing.assert_frame_equal(fleet, again)
    np.testing.assert_array_equal(sums.replan_count, repeat.replan_count)


def test_no_update_after_the_decision_reaches_its_book_or_the_import_up_to_it() -> None:
    # §8 row 7.  Decision tau: 20:00 London on the third study night.
    simulated = _simulate()
    settings = simulated.settings
    units, dispatch = _real_dispatch(simulated, 0.0, threshold=0.0, half_spread=0.0)
    warmup = 48 * settings.warmup_days
    tau_slot = warmup + 2 * 48 + 16
    tau = int(dispatch.gate_utc_ns[tau_slot] + _HOUR_NS)
    # Perturb every intraday update not yet made at tau, every step of a
    # slot not yet published, and every unpublished day-ahead price.
    path = simulated.market_prices.intraday_path_gbp_per_mwh.copy()
    made = np.clip(-((tau - dispatch.gate_utc_ns) // _HOUR_NS), 0, path.shape[2] - 1)
    unpublished = dispatch.publication_utc_ns > tau
    for slot, step in enumerate(made):
        path[:, slot, :step] += 1000.0
    path[:, unpublished, :] += 1000.0
    day_ahead = (
        simulated.forecast_prices["wholesale_forecast_gbp_per_mwh"]
        .to_numpy()
        .reshape(settings.evaluation_world_count, -1)
        .copy()
    )
    day_ahead[:, unpublished] += 1000.0
    perturbed_visible = smart_charging_inputs(
        settings,
        simulated.units,
        day_ahead,
        departure_margin_hours=simulated.departure_margin_hours,
    )
    smart = replace(
        simulated.smart_charging,
        visible_price_gbp_per_mwh=perturbed_visible.visible_price_gbp_per_mwh,
    )
    perturbed_dispatch = replace(dispatch, intraday_path_gbp_per_mwh=path)
    _, base_sums, base_book = _real_fleet(simulated, units, dispatch=dispatch)
    _, sums, book = _real_fleet(simulated, units, smart=smart, dispatch=perturbed_dispatch)
    row = simulated.trading_kernel["selected"]["book_decision"][tau_slot]
    np.testing.assert_array_equal(book[:, row], base_book[:, row])
    up_to = tau_slot - warmup + 1
    np.testing.assert_array_equal(
        sums.free_home_import_kwh[:, :up_to], base_sums.free_home_import_kwh[:, :up_to]
    )
    # The perturbation has teeth: later books differ.
    assert not np.array_equal(book[:, row + 1 :], base_book[:, row + 1 :])
