"""The illustrative trading overlay: baseline, positions, settlement and ledger.

What this module owns (trading contract v1 §1.2 and §4, with the §9.2,
§9.4, §9.5 and §10.4 changes; decision 0004 items 57 and 58; decision 0005): the
BL01-lite settlement baseline, the aggregator's expected availability and
expected smart plan at each decision, the day-ahead and hourly intraday
turn-down positions of the three strategies (the day-ahead one by a fixed
commitment share or the newsvendor rule), imbalance settlement and the
per-night trading ledger, plus the optional supplier price curve (§9.4).

How it fits: ``forecast`` runs the physics once and then calls
``run_trading`` with the kernel's outputs (fleet import on both paths, the
unmanaged warm-up history, each EV's warm-up session need, the departure
shortfall per path, the selected path's plan books and connection state)
and the generated prices.  The overlay reads those arrays and never changes
them (plan §2F): the dispatch never follows the trader, so all three
strategies come from one physics run and share its random futures.  It
draws no random number, so a Compare of any trading assumption runs on
identical futures.  The bands, KPIs and checks built from its frames live
in ``summaries`` like every other band.

Sign and unit conventions (contract §0): kWh per half-hour for volumes,
GBP/MWh for prices, and ``kWh / 1000 x GBP/MWh`` is the only money formula.
Money is from the aggregator's view (positive = income); turn-down
(importing less than the baseline) is a positive volume.  Every price,
volume and pound here is synthetic or illustrative: the net is
"illustrative simulated trading P&L" (decision 0005), never Axle cash.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from axle_studio.model import events as model_events
from axle_studio.model.action import (
    latest_known_prices,
    plan_around_blackout,
    plan_cheapest_slots,
    visible_day_ahead_prices,
)
from axle_studio.model.clock import london_wall_time_to_utc
from axle_studio.model.sampling import (
    MarketPrices,
    connection_window_hours,
    day_ahead_publication_utc_ns,
)
from axle_studio.model.settings import RunSettings

STRATEGIES = ("da_only", "full", "perfect_foresight")
"""The three trading strategies, in display order (contract §4.6)."""

BUCKETS = (
    "day_ahead_revenue_gbp",
    "intraday_pnl_gbp",
    "trading_cost_gbp",
    "imbalance_gbp",
    "baseline_effect_gbp",
    "grid_event_payment_gbp",
    "supplier_compensation_gbp",
    "customer_revenue_share_gbp",
    "unmet_charge_penalty_gbp",
)
"""The nine ledger buckets of decision 0005 in contract §4.7 order; ``net_gbp`` is their sum."""

MONEY_COLUMNS = (*BUCKETS, "net_gbp")
SHOCK_KINDS = ("known", "surprise", "none")
ATTRIBUTED_BUCKETS = ("day_ahead_revenue_gbp", "intraday_pnl_gbp", "imbalance_gbp")
ATTRIBUTION_COLUMNS = tuple(
    f"{bucket[: -len('_gbp')]}_{kind}_gbp" for bucket in ATTRIBUTED_BUCKETS for kind in SHOCK_KINDS
)
SETTLED_BY_KIND_COLUMNS = tuple(f"settled_mwh_{kind}" for kind in SHOCK_KINDS)
DISPATCH_SPLIT_COLUMNS = (
    "intraday_pnl_rebalancing_gbp",
    "intraday_pnl_reoptimisation_gbp",
    "trading_cost_rebalancing_gbp",
    "trading_cost_reoptimisation_gbp",
)
"""The "of which" split of intraday P&L and trading cost (dispatch contract §6.2, §6.3).

Rebalancing is what the frozen-book re-run trades (the fleet held on its
day-ahead plans), re-optimisation the rest of the actual trades; both at the
traded intraday prices.  Not buckets: ``net_gbp`` still sums the nine."""
VOLUME_COLUMNS = (
    "sold_day_ahead_mwh",
    "final_position_mwh",
    "settled_mwh",
    "flexibility_mwh",
    "baseline_effect_mwh",
    "imbalance_mwh",
    "abs_imbalance_mwh",
    "intraday_traded_mwh",
    "delivered_within_final_mwh",
    "delivered_within_day_ahead_mwh",
    "event_delivered_mwh",
    "unmet_charge_kwh",
)
LEDGER_COLUMNS = (
    "world_id",
    "strategy",
    "night_index",
    "night_start_local_date",
    "day_label",
    *MONEY_COLUMNS,
    *ATTRIBUTION_COLUMNS,
    *DISPATCH_SPLIT_COLUMNS,
    *SETTLED_BY_KIND_COLUMNS,
    *VOLUME_COLUMNS,
    "evidence_kind",
)
"""``trading_ledger_world`` columns (contract §5.3 with the §9.1 and dispatch §6.3 additions)."""

WEEK_COLUMNS = tuple(
    c for c in LEDGER_COLUMNS if c not in ("night_index", "night_start_local_date", "day_label")
)
"""``trading_week_world`` columns (contract §5.4)."""

DEVIATION_COLUMNS = (
    "world_id",
    "slot_index",
    "interval_start_utc",
    "interval_start_london",
    "night_index",
    "baseline_kwh",
    "unmanaged_kwh",
    "metered_kwh",
    "baseline_kw",
    "unmanaged_kw",
    "metered_kw",
    "expected_metered_kwh",
    "expected_unmanaged_kwh",
    "deviation_kwh",
    "true_reduction_kwh",
    "baseline_effect_kwh",
    "settled_kwh",
    "settlement_open",
    "position_da_only_kwh",
    "position_full_kwh",
    "position_perfect_foresight_kwh",
    "commit_level",
    "forecast_error_quantile_kwh",
    "day_ahead_gbp_per_mwh",
    "intraday_close_gbp_per_mwh",
    "imbalance_gbp_per_mwh",
    "known_shock_gw",
    "surprise_shock_gw",
    "shock_kind",
    "evidence_kind",
)
"""``deviation_world_slot`` columns (contract §5.1, with the §10.4 additions)."""

POSITION_UPDATE_COLUMNS = (
    "world_id",
    "slot_index",
    "night_index",
    "interval_start_utc",
    "interval_start_london",
    "decision_utc",
    "decision_london",
    "stage",
    "hours_to_gate_closure",
    "baseline_known_kwh",
    "forecast_metered_kwh",
    "position_kwh",
    "trade_kwh",
    "price_gbp_per_mwh",
    "frozen_position_kwh",
    "frozen_trade_kwh",
    "evidence_kind",
)
"""``position_updates`` columns (contract §5.2 with the dispatch §6.3 additions).

``frozen_position_kwh`` and ``frozen_trade_kwh`` are the frozen-book
re-run's target and trade at the same (world, slot, decision): "frozen"
means the fleet held on its day-ahead plans.  On the day-ahead row both
equal ``q``; with intraday dispatch off they equal ``position_kwh`` and
``trade_kwh``."""

STRATEGY_CAPTIONS = {
    "da_only": (
        "Sold day-ahead and did not hedge the fleet's intraday moves: the dispatch's "
        "deviation from the day-ahead position is settled at the imbalance price"
    ),
    "full": "The product: day-ahead position, hourly re-positioning on the dispatched fleet",
    "perfect_foresight": (
        "Perfect foresight of volume, not of intraday prices: sells the settled deviation "
        "of the dispatched fleet at the day-ahead price and never trades intraday"
    ),
}
"""What each strategy's row means when intraday dispatch is on (dispatch contract §6.2, Q8).

All three settle on the one dispatched path; views show these beside the
strategy rows and compute nothing from them."""

PERFECT_FORESIGHT_CAPTION = "against perfect foresight of volume, not of intraday prices"
"""Caption on the capture-rate and cost-of-uncertainty rows (dispatch contract §6.2 B3).

``perfect_foresight`` never trades intraday while ``full`` earns
re-optimisation P&L on price moves, so with dispatch on a capture rate above
1 and a negative cost of uncertainty can occur and are not errors."""
PERFECT_FORESIGHT_CAPTION_METRICS = ("capture_rate", "cost_of_uncertainty_gbp_per_week")
"""The ``trading_kpis`` metrics that carry ``PERFECT_FORESIGHT_CAPTION``."""

BOOK_WIDTH = 50
"""Slots ahead each plan book holds: the longest session night (autumn, 50 slots)."""

EVIDENCE_KIND = "illustrative_synthetic"
_SLOT_HOURS = 0.5
_HALF_HOUR_NS = 30 * 60 * 1_000_000_000
_HOUR_NS = 3600 * 1_000_000_000
_LONDON = "Europe/London"


# ============================================================================
# 4.1 BL01-lite baseline
# ============================================================================


def bl01_lite_baseline(
    history_kwh: np.ndarray,
    history_slots: pd.DataFrame,
    study_slots: pd.DataFrame,
    *,
    working_nights: int,
    non_working_nights: int,
) -> tuple[np.ndarray, int, list[int]]:
    """The unadjusted BL01-lite baseline ``B0`` (contract §4.1 as changed by §9.5).

    ``history_kwh`` (world, history slot) is home import in kWh per
    half-hour over ``history_slots`` (rows with ``night_index``,
    ``local_half_hour`` and ``day_type``): the unmanaged warm-up nights and,
    in trap mode, the metered study nights appended after them.  Returns
    ``B0`` (world, study slot) in kWh per half-hour, the number of study
    nights that used the other-class fallback and the indices of the study
    nights with no history night at all (baseline 0: nothing is sold or
    settled, and ``in_day_adjustment`` leaves them at 0).

    Night ``n`` uses history nights ``<= n - 2`` only: night ``n - 1`` ends
    at 12:00 on ``D_n``, after the day-ahead decision at 13:00 on ``D_n - 1``,
    so including it would leak (review B3).  Its class is working
    (Monday-Friday, by the evening's date ``D_n``, which carries most of the
    turn-down) or non-working; ``B0`` of a slot is the mean home import of
    the same London half-hour over the last ``working_nights`` (5) or
    ``non_working_nights`` (2) history nights of that class, or of the other
    class when the history holds none (counted).  Both copies of a repeated
    autumn half-hour enter the mean; a half-hour none of the chosen nights
    has (the spring gap) takes the mean of all their slots.
    """

    history_night = history_slots["night_index"].to_numpy()
    history_half_hour = history_slots["local_half_hour"].to_numpy()
    history_working = history_slots["day_type"].eq("weekday").to_numpy()
    night = study_slots["night_index"].to_numpy()
    half_hour = study_slots["local_half_hour"].to_numpy()
    working = study_slots["day_type"].eq("weekday").to_numpy()
    baseline = np.empty((history_kwh.shape[0], len(study_slots)))
    fallback_nights = 0
    missing_nights: list[int] = []
    for n in np.unique(night):
        in_night = np.flatnonzero(night == n)
        is_working = bool(working[in_night[0]])
        count = working_nights if is_working else non_working_nights
        nights = np.unique(history_night[history_night <= n - 2])
        classes = np.array([history_working[history_night == m][0] for m in nights], dtype=bool)
        if len(nights) == 0:
            # A warm-up shorter than two nights (only hand-built runs; the
            # dialog's minimum is 3) leaves no history the day-ahead decision
            # could know.  Borrowing a later night would leak, so the night
            # has no baseline: zero, so nothing is sold or settled on it.
            baseline[:, in_night] = 0.0
            missing_nights.append(int(n))
            continue
        same = nights[classes == is_working]
        if len(same) == 0:
            fallback_nights += 1
            same = nights[classes != is_working]
        chosen = same[-count:]
        in_history = np.isin(history_night, chosen)
        for t in in_night:
            cells = in_history & (history_half_hour == half_hour[t])
            baseline[:, t] = history_kwh[:, cells if cells.any() else in_history].mean(axis=1)
    return baseline, fallback_nights, missing_nights


def in_day_adjustment(
    baseline_kwh: np.ndarray,
    metered_kwh: np.ndarray,
    study_slots: pd.DataFrame,
    *,
    window_slots: int,
    missing_nights: Sequence[int] = (),
) -> np.ndarray:
    """BL01's in-day adjustment ``a_n`` (world, study slot), constant within each night (§4.1).

    ``a_n`` is the mean of ``M - B0`` (kWh per half-hour) over the first
    ``window_slots`` (3) slots of night ``n``, 12:00-13:30 London; the
    adjusted baseline is ``B = B0 + a_n``.  It is known from 13:30 on
    ``D_n``, so decisions before then use ``B0`` (§4.5).  Fleet import is
    close to zero at midday, so the adjustment is nearly inert (§8 Q3).
    ``missing_nights`` (``bl01_lite_baseline``'s nights with no history)
    get ``a_n = 0``: their baseline is 0 so nothing is sold or settled, and
    adjusting it would turn midday import into a baseline from no history.
    """

    night = study_slots["night_index"].to_numpy()
    adjustment = np.zeros_like(baseline_kwh)
    for n in np.unique(night):
        if n in missing_nights:
            continue
        in_night = np.flatnonzero(night == n)
        first = in_night[:window_slots]
        mean = (metered_kwh[:, first] - baseline_kwh[:, first]).mean(axis=1)
        adjustment[:, in_night] = mean[:, np.newaxis]
    return adjustment


def flexed_history_baseline(
    metered_kwh: np.ndarray,
    study_slots: pd.DataFrame,
    *,
    working_nights: int,
    non_working_nights: int,
) -> tuple[np.ndarray, list[int]]:
    """A steady-state baseline learnt from smart nights only (decision 0004 item 67).

    ``metered_kwh`` (world, study slot) is the selected (smart) path's home
    import in kWh per half-hour.  Returns the unadjusted baseline (world,
    study slot, kWh per half-hour) and the study nights with no other night
    to learn from (baseline 0, like ``bl01_lite_baseline``'s missing nights).

    Why: the default ``B0`` learns from unmanaged warm-up nights only (like
    BL01 excluding days with delivered flex), so it never sees a flexed
    night.  A baseline that rolls over the fleet's own flexed nights loses
    the evening turn-down it is meant to measure (model questions Q-4).
    Trap mode shows only the first week of that decay, because a week's
    history is mostly warm-up.  In steady state every history night is
    flexed, and the model has no smart warm-up (its warm-up is unmanaged but
    for the last night), so this baseline stands in the world's own other
    study nights for "previous weeks' flexed nights": the same day class and
    the same counts as §4.1 (at most ``working_nights`` or
    ``non_working_nights``, nearest earlier nights first, wrapping round the
    study), with the other class as fallback.  Only the night itself is kept
    out.  The ``n - 2`` cut-off is not applied: it is a decision-time rule
    for the day-ahead position, and this is a settlement what-if that trades
    nothing.  With a 7-night study a working night learns from 4 nights and a
    non-working night from 1 (not 5 and 2), so this baseline is noisier than
    BL01's and the one-sided ``max(0, B - M)`` settles more of that noise as
    turn-down: the figure understates the erosion, most visibly outside the
    evening (300 EVs x 20 weeks: evening value falls about 92 %, the rest
    rises).
    """

    night = study_slots["night_index"].to_numpy()
    half_hour = study_slots["local_half_hour"].to_numpy()
    working = study_slots["day_type"].eq("weekday").to_numpy()
    nights = np.unique(night)
    night_working = np.array([working[night == m][0] for m in nights], dtype=bool)
    baseline = np.zeros_like(metered_kwh, dtype=float)
    missing: list[int] = []
    for n, is_working in zip(nights, night_working, strict=True):
        in_night = np.flatnonzero(night == n)
        # Nearest earlier nights first, wrapping round the study.
        others = sorted((m for m in nights if m != n), key=lambda m: (n - m) % len(nights))
        same = [m for m in others if night_working[nights == m][0] == is_working]
        pool = same or others
        if not pool:
            missing.append(int(n))
            continue
        chosen = pool[: working_nights if is_working else non_working_nights]
        in_history = np.isin(night, chosen)
        for t in in_night:
            cells = in_history & (half_hour == half_hour[t])
            baseline[:, t] = metered_kwh[:, cells if cells.any() else in_history].mean(axis=1)
    return baseline, missing


# ============================================================================
# 4.3-4.4 Expected availability and expected smart import
# ============================================================================


def day_ahead_decision_utc_ns(night_local_date: object, decision_local_hour: float) -> int:
    """``tau_DA(n)`` as UTC nanoseconds: ``decision_local_hour`` London on ``D_n - 1`` (§4.4).

    ``night_local_date`` is the night's London evening date ``D_n``;
    ``decision_local_hour`` the day-ahead publication hour (13.0).
    """

    day_before = np.datetime64(night_local_date, "D") - np.timedelta64(1, "D")
    return int(
        london_wall_time_to_utc(day_before, 60.0 * decision_local_hour)
        .astype("datetime64[ns]")
        .astype(np.int64)
    )


def book_decision_slots(
    settings: RunSettings, study_slots: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """The plan book's hourly decision slots (§1.4, §4.5; replay contract v1 §2).

    Plan books are taken at every whole London hour of the study, the
    trader's and dispatcher's own decision instants: whole UTC hours are
    whole London hours (the run's offset is always a whole number of
    hours), so a slot's London minute is 0 exactly when its UTC minute is,
    and that happens on every other study slot. Returns:

    - ``book_slots`` (decision,): each hourly decision's study-slot index,
      in study-slot order (normally ``2, 4, ...`` for decision ``k``, but
      read from the run slots rather than assumed, since a clock-change
      night can shift which slots exist);
    - ``book_decision`` (run slot,), the run's warm-up-plus-study slots:
      -1 except at a book slot, where it holds that slot's position in
      ``book_slots`` (its row in ``book_kwh``, ``physics._record_trading``).

    One helper for the trading kernel's own book (``forecast``, which
    passes it to ``physics.simulate_fleet_intervals``) and the one-EV
    replay's book (``individual.replay_one_ev_timeline``), so the two plan
    books cannot drift into different decision grids.
    """

    slot_count = len(study_slots)
    minute = pd.DatetimeIndex(study_slots["interval_start_london"]).minute
    book_slots = np.flatnonzero(np.asarray(minute) == 0)
    book_decision = np.full(settings.warmup_days * 48 + slot_count, -1, dtype=np.int64)
    book_decision[settings.warmup_days * 48 + book_slots] = np.arange(len(book_slots))
    return book_slots, book_decision


def typical_session_windows(
    units: pd.DataFrame,
    study_slots: pd.DataFrame,
    expected_departure_utc_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Each EV's typical session window per night, as study slot indices (§4.3).

    Returns ``start`` and ``deadline`` (night, EV) int64: the first whole
    study slot at or after the cohort's typical plug-in clock on ``D_n``
    (weekday or weekend window by ``D_n``, decision 0004 item 42) and one
    past the last whole slot before the expected departure on ``D_n + 1``
    (``expected_departure_utc_ns``, the smart planner's own target: typical
    departure minus the margin, rows from the day before the study).  Both
    are clipped to the night, so the always-plugged cohort's session is cut
    at the night end (a stated limitation).  They use typical clocks only,
    never a sampled session, so no realised information enters a forecast.
    """

    night = study_slots["night_index"].to_numpy()
    starts_ns = pd.DatetimeIndex(study_slots["interval_start_utc"]).as_unit("ns").asi8
    dates = sorted(set(study_slots["local_date"]))
    weekday_hours, weekend_hours = connection_window_hours(units, "arrival")
    night_count = len(dates)
    start = np.empty((night_count, len(units)), dtype=np.int64)
    deadline = np.empty_like(start)
    for n, local_date in enumerate(dates):
        in_night = np.flatnonzero(night == n)
        first, end = in_night[0], in_night[-1] + 1
        hours = weekend_hours if local_date.weekday() >= 5 else weekday_hours
        plug_in = london_wall_time_to_utc(np.datetime64(local_date, "D"), 60.0 * hours)
        plug_in_ns = plug_in.astype("datetime64[ns]").astype(np.int64)
        # Whole slots only (decision 0001): a plug-in inside a slot can
        # charge from the next boundary, as the kernel's connected_full_slot.
        start[n] = np.clip(np.searchsorted(starts_ns, plug_in_ns, side="left"), first, end)
        departure_ns = expected_departure_utc_ns[n + 2]
        last = np.searchsorted(starts_ns + _HALF_HOUR_NS, departure_ns, side="right")
        deadline[n] = np.clip(last, start[n], end)
    return start, deadline


def expected_need_kwh(
    warmup_need_kwh: np.ndarray, warmup_session_count: np.ndarray, cohort_ids: np.ndarray
) -> np.ndarray:
    """Expected grid kWh per home session, (world, EV), from the warm-up only (§4.3).

    ``warmup_need_kwh`` is the sum and ``warmup_session_count`` the number of
    each EV's unmanaged warm-up sessions, so their ratio is the mean need at
    plug-in.  An EV with no warm-up session takes the mean of its cohort's
    EVs that had one in the same world, else 0.
    """

    count = np.asarray(warmup_session_count, dtype=float)
    need = np.divide(warmup_need_kwh, count, out=np.full(count.shape, np.nan), where=count > 0)
    for cohort_id in np.unique(cohort_ids):
        members = cohort_ids == cohort_id
        block = need[:, members]
        has = ~np.isnan(block)
        cohort_mean = np.divide(
            np.where(has, block, 0.0).sum(axis=1),
            has.sum(axis=1),
            out=np.zeros(block.shape[0]),
            where=has.any(axis=1),
        )
        need[:, members] = np.where(has, block, cohort_mean[:, np.newaxis])
    return need


def expected_sessions(
    plug_probability: np.ndarray,
    need_kwh: np.ndarray,
    slot_cap_kwh: np.ndarray,
    start: np.ndarray,
    deadline: np.ndarray,
    *,
    decision_slot: int | None = None,
    include: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Expected home session per EV for one night at one decision (§4.3, review B1).

    Inputs: ``plug_probability`` (EV,) the cohort's plug-in probability (the
    sampler's parameter, not a realised draw); ``need_kwh`` (world, EV) the
    expected grid kWh; ``slot_cap_kwh`` (EV,) home power x 0.5 h; ``start``
    and ``deadline`` (EV,) the typical window as study slot indices.  With
    ``decision_slot`` (intraday), the start is re-based to
    ``max(start, decision_slot)`` and a session whose deadline is at or
    before it gets weight 0; ``include`` (world, EV) bool drops sessions the
    trader already sees in its book.  The need is capped at what the
    (re-based) window can take.  Returns ``weight`` and ``need_kwh`` (world,
    EV), ``start``, ``deadline`` and ``slot_cap_kwh`` (EV,).
    """

    start = np.asarray(start)
    if decision_slot is not None:
        start = np.maximum(start, decision_slot)
    open_window = deadline > start
    weight = np.broadcast_to(plug_probability * open_window, need_kwh.shape).astype(float)
    if include is not None:
        weight = np.where(include, weight, 0.0)
    need = np.minimum(need_kwh, slot_cap_kwh * np.maximum(deadline - start, 0))
    return {
        "weight": weight,
        "need_kwh": need,
        "start": start,
        "deadline": deadline,
        "slot_cap_kwh": slot_cap_kwh,
    }


def unmanaged_schedule_kwh(
    need_kwh: np.ndarray, slot_cap_kwh: np.ndarray, in_window: np.ndarray
) -> np.ndarray:
    """Full power from the window start until the need is met, (session, slot) kWh.

    The closed form of ``plan_cheapest_slots`` under a flat price (the
    stable sort keeps time order), which is exactly how an unmanaged or
    non-responding session charges (§4.4 ``u_i``); written directly because
    it needs no sort and runs at every intraday decision.
    """

    cap = slot_cap_kwh[:, np.newaxis]
    before = np.cumsum(in_window, axis=1) - in_window
    return np.where(in_window, np.clip(need_kwh[:, np.newaxis] - before * cap, 0.0, cap), 0.0)


def expected_smart_kwh(
    sessions: dict[str, np.ndarray],
    window_price_gbp_per_mwh: np.ndarray,
    *,
    first_slot: int,
    non_response_probability: np.ndarray,
    price_adjustment_gbp_per_mwh: np.ndarray | None = None,
    zone_index: np.ndarray | None = None,
    blackout: np.ndarray | None = None,
    free: np.ndarray | None = None,
    free_window_price_gbp_per_mwh: np.ndarray | None = None,
) -> np.ndarray:
    """Expected fleet smart import (world, slot) from ``first_slot`` on (§4.4, lead decision QA).

    ``x_t = sum_i w_i [rho_i u_it + (1 - rho_i) plan_it]``: ``plan_i`` is
    ``action.plan_cheapest_slots`` on the session's visible prices
    ``window_price_gbp_per_mwh`` (world, slot from ``first_slot``, GBP/MWh;
    +inf outside the session's window), ``u_i`` full power from the window
    start, and ``rho_i`` (EV,) the session's non-response probability (1 for
    a control-group EV).  Why the per-session planner and no LP: each
    session's problem (linear cost, one cap, one deadline) is solved exactly
    by filling its cheapest slots, so the expectation over typical sessions
    is exactly this weighted sum; it is the dispatcher's own rule, so the
    trader forecasts what the fleet does (a fleet LP would pool one EV's
    spare power for another's need).  Weight-0 sessions are dropped first.

    ``price_adjustment_gbp_per_mwh`` (zone, slot from ``first_slot``) adds
    the planner signals of the grid requests known at the decision to each
    session's ranking price by its ``zone_index`` (EV,), exactly as the
    kernel's planner does (§2.4): the trader expects the fleet to avoid a
    known turn-down window and fill a known turn-up window.

    ``blackout`` (slot from ``first_slot``, bool) applies the kernel's
    blackout rule (§10.5b, ``action.plan_around_blackout``): the expected
    plan keeps the unmanaged charging in blackout half-hours and moves the
    rest to the cheapest slots outside, as the dispatcher does.

    ``free`` (EV,) bool and ``free_window_price_gbp_per_mwh`` (world, slot
    from ``first_slot``, GBP/MWh) are for intraday dispatch (dispatch
    contract §6.1): a free EV's session ranks the latest known intraday
    price instead of ``window_price_gbp_per_mwh``, which locked EVs keep
    ranking.  Its real plug-in plan will use the price known at its plug-in,
    which the trader cannot know; the latest price now is its best estimate.
    ``None`` (dispatch off): every session ranks ``window_price_gbp_per_mwh``.
    """

    world_count, width = window_price_gbp_per_mwh.shape
    weight, need = sessions["weight"], sessions["need_kwh"]
    world_index, ev_index = np.nonzero((weight > 0.0) & (need > 0.0))
    expected = np.zeros((world_count, width))
    if len(world_index) == 0:
        return expected
    slots = first_slot + np.arange(width)
    in_window = (slots >= sessions["start"][ev_index, np.newaxis]) & (
        slots < sessions["deadline"][ev_index, np.newaxis]
    )
    cap = sessions["slot_cap_kwh"][ev_index]
    need_rows = need[world_index, ev_index]
    rho = non_response_probability[ev_index]
    per_session = rho[:, np.newaxis] * unmanaged_schedule_kwh(need_rows, cap, in_window)
    plans = rho < 1.0
    if plans.any():
        ranked = window_price_gbp_per_mwh[world_index[plans]]
        if free is not None:
            # One per-session choice: locked sessions keep the B4-visible
            # day-ahead price, free ones rank the latest intraday price.
            ranked = np.where(
                free[ev_index[plans], np.newaxis],
                free_window_price_gbp_per_mwh[world_index[plans]],
                ranked,
            )
        if price_adjustment_gbp_per_mwh is not None:
            ranked = ranked + price_adjustment_gbp_per_mwh[zone_index[ev_index[plans]]]
        prices = np.where(in_window[plans], ranked, np.inf)
        if blackout is not None and blackout.any():
            planned = plan_around_blackout(
                need_rows[plans], cap[plans], prices, blackout[np.newaxis, :] & in_window[plans]
            )
        else:
            planned = plan_cheapest_slots(need_rows[plans], cap[plans], prices)
        per_session[plans] += (1.0 - rho[plans, np.newaxis]) * planned
    per_session *= weight[world_index, ev_index][:, np.newaxis]
    # np.nonzero walks row-major, so each world's sessions are one block.
    worlds, first_row = np.unique(world_index, return_index=True)
    expected[worlds] = np.add.reduceat(per_session, first_row, axis=0)
    return expected


def positions_kwh(
    baseline_kwh: np.ndarray, forecast_metered_kwh: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Target turn-down position ``mask x max(0, B - M_hat)`` (world, slot) kWh (lead decision QC).

    Positions forecast the settled deviation; only turn-down is sold (the
    cheap charging leg stays on the customer's tariff, decision 0005).
    """

    return mask * np.maximum(baseline_kwh - forecast_metered_kwh, 0.0)


# ============================================================================
# 10.4 Newsvendor commitment
# ============================================================================

COMMITMENT_RULES = ("fixed_share", "newsvendor")
"""``trading.commitment_rule`` values (contract §10.4); ``fixed_share`` is the default."""


def _sum_of_others(values: np.ndarray) -> np.ndarray:
    """(world, slot) -> the sum over every other world, per (world, slot).

    Built from an exclusive running sum from each end, so world ``w``'s own
    value never enters its result (not even as ``total - own``, which would
    leak it through rounding): the leave-one-out tests of §10.4 hold exactly.
    """

    zeros = np.zeros_like(values[:1])
    before = np.concatenate([zeros, np.cumsum(values, axis=0)[:-1]])
    after = np.concatenate([np.cumsum(values[::-1], axis=0)[::-1][1:], zeros])
    return before + after


def _leave_one_out_quantile(errors: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Per (world, slot), the quantile of the other worlds' errors at that world's level.

    ``errors`` and ``level`` are (world, slot); the result equals
    ``numpy.quantile(np.delete(errors[:, t], w), level[w, t])`` (the linear
    method) for every ``w`` and ``t``, without building one sample per world:
    the other worlds' sorted errors are the full sort with ``w``'s own entry
    skipped.
    """

    world_count = errors.shape[0]
    order = np.argsort(errors, axis=0, kind="stable")
    ordered = np.take_along_axis(errors, order, axis=0)
    rank = np.argsort(order, axis=0, kind="stable")
    others = world_count - 1
    position = level * (others - 1)
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low + 1, others - 1)
    gamma = position - low
    # Index i of the leave-one-out sample is i of the full sort, or i + 1
    # once past w's own rank.
    a = np.take_along_axis(ordered, low + (low >= rank), axis=0)
    b = np.take_along_axis(ordered, high + (high >= rank), axis=0)
    # numpy's own linear interpolation, including its switch to the upper
    # end for gamma >= 0.5, so the result matches numpy.quantile exactly.
    return np.where(gamma >= 0.5, b - (b - a) * (1.0 - gamma), a + (b - a) * gamma)


def newsvendor_positions(
    forecast_kwh: np.ndarray,
    settled_kwh: np.ndarray,
    day_ahead_gbp_per_mwh: np.ndarray,
    imbalance_gbp_per_mwh: np.ndarray,
    system_short: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The newsvendor day-ahead position ``q`` of one night (contract §10.4).

    Inputs are (world, slot) over the night's slots: ``forecast_kwh`` the
    §4.4 point forecast ``F_hat = mask x max(0, B0 - x)`` (kWh per
    half-hour), ``settled_kwh`` the settled volume ``V`` (kWh),
    ``day_ahead_gbp_per_mwh`` the price ``P_DA`` visible at the day-ahead
    decision, ``imbalance_gbp_per_mwh`` the system imbalance price ``SIP``
    and ``system_short`` (bool) the NIV state; ``mask`` (slot,) is the
    day-ahead trading mask.

    Returns ``(q, alpha, e)``, all (world, slot): the position (kWh), the
    commit level ``alpha = clip(P_DA / k, 0, 1)`` (0 when ``k <= 0`` or
    ``P_DA <= 0``) and the error quantile ``e(alpha)`` (kWh), with
    ``q = mask x max(0, F_hat + e(alpha))`` and ``q = 0`` where
    ``P_DA <= 0``.

    Why this rule: selling ``q`` at ``p`` and buying any shortfall back at
    ``k`` gives ``Pi(q) = p q - k E[(q - V)+]``, maximised where
    ``P(V <= q) = p / k``.  It assumes unpaid spill and a short-state
    buy-back, so under this contract's single-price ledger it is a risk
    choice, not the expected-cash optimum (§10.4, B2).  Both the error
    quantile and ``k`` read only the other worlds (the other simulated
    weeks, "past weeks" in reality, B3): world ``w``'s own ``V`` and
    ``SIP`` are after its decision, so reading them would leak the future.
    """

    world_count = forecast_kwh.shape[0]
    if world_count < 2:
        raise ValueError("the newsvendor commitment needs at least two worlds")
    others = world_count - 1
    short = np.asarray(system_short, dtype=bool)
    short_count = _sum_of_others(short.astype(float))
    short_sum = _sum_of_others(np.where(short, imbalance_gbp_per_mwh, 0.0))
    all_mean = _sum_of_others(imbalance_gbp_per_mwh) / others
    # Expected shortfall cost: the other worlds' SIP when the system was
    # short, or every other world's SIP when none was short at this slot.
    k = np.where(short_count > 0, short_sum / np.maximum(short_count, 1.0), all_mean)
    price = day_ahead_gbp_per_mwh
    priced = (price > 0.0) & (k > 0.0)
    alpha = np.zeros_like(price, dtype=float)
    alpha[priced] = np.clip(price[priced] / k[priced], 0.0, 1.0)
    error = _leave_one_out_quantile(settled_kwh - forecast_kwh, alpha)
    q = mask * np.maximum(forecast_kwh + error, 0.0)
    # A day-ahead price at or below zero pays nothing for turn-down, so
    # nothing is committed (unlike alpha = 0, which still commits the
    # smallest error the other weeks saw).
    q = np.where(price > 0.0, q, 0.0)
    return q, alpha, error


# ============================================================================
# 4.5-4.7 The overlay
# ============================================================================


@dataclass(frozen=True)
class TradingRun:
    """What ``run_trading`` returns: the result frames and the arrays summaries need.

    Frames: ``deviation_world_slot`` (§5.1), ``position_updates`` (§5.2),
    ``trading_ledger_world`` (§5.3) and ``trading_week_world`` (§5.4).
    Arrays, (world, study slot) kWh unless stated: ``baseline_unadjusted``
    (``B0``), ``baseline`` (``B``), ``settled`` (``V``), ``day_ahead_position``
    (``q``, the committed share), ``positions`` {strategy: final ``f``},
    ``visible_day_ahead_gbp_per_mwh`` (the prices seen at each night's
    day-ahead decision), ``revised_slots`` (bool: slots with at least one
    intraday decision), ``control_baseline`` and ``control_unmanaged``
    (``B_c``, ``U_c``; None without a control group), ``scope_baseline``
    (world, scope, slot: ``B`` of the fleet then each zone, None without
    zones), ``event_delivery`` (``events.event_delivery``'s rows, None
    without requests) and ``slot_cash_gbp`` (world, study slot, GBP: the
    ``full`` strategy's six slot-resolved buckets summed, from
    ``slot_cash_components``; the replay's running trading cash, replay
    contract v1 §1.4) and ``settled_flexed_baseline`` (world, study slot,
    kWh: ``V`` against ``flexed_history_baseline``, the baseline-erosion
    what-if of decision 0004 item 67; outside the ledger).  Scalars:
    ``fallback_nights`` (§4.1), ``missing_nights`` (no history at all) and
    ``commitment_share``.  §10.4: ``commitment_rule`` (``fixed_share`` or
    ``newsvendor``), ``day_ahead_forecast`` (world, slot: the point forecast
    ``F_hat`` behind ``q``, kWh) and, in newsvendor mode only,
    ``commit_level`` (``alpha``) and ``forecast_error_quantile`` (``e``,
    kWh), None in fixed-share mode.
    """

    deviation_world_slot: pd.DataFrame
    position_updates: pd.DataFrame
    trading_ledger_world: pd.DataFrame
    trading_week_world: pd.DataFrame
    baseline_unadjusted: np.ndarray
    baseline: np.ndarray
    settled: np.ndarray
    day_ahead_position: np.ndarray
    positions: dict[str, np.ndarray]
    visible_day_ahead_gbp_per_mwh: np.ndarray
    revised_slots: np.ndarray
    fallback_nights: int
    missing_nights: int
    commitment_share: float
    commitment_rule: str = "fixed_share"
    day_ahead_forecast: np.ndarray | None = None
    commit_level: np.ndarray | None = None
    forecast_error_quantile: np.ndarray | None = None
    control_baseline: np.ndarray | None = None
    control_unmanaged: np.ndarray | None = None
    scope_baseline: np.ndarray | None = None
    event_delivery: pd.DataFrame | None = None
    slot_cash_gbp: np.ndarray | None = None
    settled_flexed_baseline: np.ndarray | None = None


def run_trading(
    *,
    study_slots: pd.DataFrame,
    warmup_slots: pd.DataFrame,
    units: pd.DataFrame,
    unmanaged_kwh: np.ndarray,
    metered_kwh: np.ndarray,
    warmup_unmanaged_kwh: np.ndarray,
    expected_need: np.ndarray,
    departure_shortfall_kwh: Mapping[str, np.ndarray],
    book_kwh: np.ndarray,
    book_slots: np.ndarray,
    connected: np.ndarray,
    expected_departure_utc_ns: np.ndarray,
    day_ahead_gbp_per_mwh: np.ndarray,
    run_interval_start_utc: pd.DatetimeIndex,
    intraday_path_gbp_per_mwh: np.ndarray,
    intraday_close_gbp_per_mwh: np.ndarray,
    imbalance_gbp_per_mwh: np.ndarray,
    known_shock_gw: np.ndarray,
    surprise_shock_gw: np.ndarray,
    non_response_probability: np.ndarray,
    assumptions: Mapping[str, float],
    sampled_world_ids: Sequence[int],
    control_mask: np.ndarray | None = None,
    control_warmup_kwh: np.ndarray | None = None,
    control_unmanaged_kwh: np.ndarray | None = None,
    trading_mask: Callable[[int | None], np.ndarray] | None = None,
    events: pd.DataFrame | None = None,
    zone_index: np.ndarray | None = None,
    zone_warmup_kwh: np.ndarray | None = None,
    zone_metered_kwh: np.ndarray | None = None,
    event_notice_slot: np.ndarray | None = None,
    event_adjustment_gbp_per_mwh: np.ndarray | None = None,
    first_night_expected_need: np.ndarray | None = None,
    system_short: np.ndarray | None = None,
    blackout: np.ndarray | None = None,
    dispatch_locked: np.ndarray | None = None,
    day_ahead_book_kwh: np.ndarray | None = None,
) -> TradingRun:
    """Run the whole overlay: baseline, expectations, positions per strategy, settlement.

    Kernel inputs, all kWh per half-hour: ``unmanaged_kwh`` ``U`` and
    ``metered_kwh`` ``M`` (world, study slot), fleet home import on the
    normal and selected paths; ``warmup_unmanaged_kwh`` (world, warm-up
    slot) on the normal path; ``expected_need`` (world, EV,
    ``expected_need_kwh``) and ``first_night_expected_need`` (the same from
    the warm-up sessions that began before ``tau_DA(0)``, for night 0's
    day-ahead decision; None uses ``expected_need``);
    ``departure_shortfall_kwh`` {path: (world, study slot)} battery kWh;
    ``book_kwh`` (world, decision, ``BOOK_WIDTH``) the selected path's
    plans in force at each hourly decision slot ``book_slots`` (decision,)
    study slot indices; ``connected`` (world,
    study slot + 1, EV) the connection state from the slot before the
    study.  ``expected_departure_utc_ns`` (date, EV) is the planner's target.

    Prices, GBP/MWh: ``day_ahead_gbp_per_mwh`` (world, run slot) over
    ``run_interval_start_utc`` (warm-up then study),
    ``intraday_path_gbp_per_mwh`` (world, run slot, h),
    ``intraday_close_gbp_per_mwh`` and ``imbalance_gbp_per_mwh`` (world,
    study slot); shocks in GW (world, study slot).

    ``non_response_probability`` (EV,) is each session's expected rate
    (1 for a control EV); ``assumptions`` the ``assumptions.trading_inputs``
    values; ``sampled_world_ids`` the worlds ``position_updates`` keeps.

    Events and zones (contract §2.4, §4.6), all defaulting to "no events":
    ``events`` is the validated events table.  Its grid requests pause
    trading in their windows once known (``events.trading_mask`` with the
    decision instant; settlement uses every request), and each request's
    delivery and payment (``events.event_delivery``, against the scope's
    adjusted baseline) is booked on the slot its window starts, so the
    night the window starts in pays it.  The zone baselines need
    ``zone_warmup_kwh`` (world, zone, warm-up slot) unmanaged and
    ``zone_metered_kwh`` (world, zone, study slot) metered import.
    ``event_notice_slot`` (request,) and ``event_adjustment_gbp_per_mwh``
    (request, zone, run slot) are the planner signals
    (``events.planner_adjustments``) and ``zone_index`` (EV,) each EV's zone:
    the expected plan adds the signals of requests known at the decision.
    ``trading_mask(known_at_utc_ns)`` overrides the events' mask (tests).
    ``blackout`` (study slot,) bool closes the blackout half-hours at every
    decision and in settlement (§10.5b); ``None``: no blackout.

    Commitment rule (§10.4): ``assumptions["trading.commitment_rule"]``
    (absent: ``fixed_share``) selects ``q = c x F_hat`` or the newsvendor
    position (``newsvendor_positions``), which needs ``system_short``
    (world, study slot, bool: the NIV state behind each imbalance price).

    Intraday dispatch (dispatch contract §6), both given or neither:
    ``dispatch_locked`` (EV,) bool is ``units.dispatch_locked`` and
    ``day_ahead_book_kwh`` (world, decision, ``BOOK_WIDTH``) the day-ahead
    plan path's book (``DispatchSums.book_kwh["day_ahead"]``); then
    ``metered_kwh``, ``book_kwh`` and the rest describe the dispatched
    path.  At each intraday decision free sessions not yet started are
    expected to plan on the latest known price (§6.1), and the frozen-book
    re-run (§6.2) splits the ``full`` strategy's intraday P&L and trading
    cost into rebalancing and re-optimisation.  ``None`` (dispatch off):
    every expected plan ranks the B4-visible day-ahead price, no re-run is
    made, the frozen trades are the actual trades and re-optimisation is
    exactly 0.
    """

    if (dispatch_locked is None) != (day_ahead_book_kwh is None):
        raise ValueError("intraday dispatch needs both dispatch_locked and day_ahead_book_kwh")
    world_count, slot_count = unmanaged_kwh.shape
    warmup_count = warmup_unmanaged_kwh.shape[1]
    night = study_slots["night_index"].to_numpy()
    night_starts = np.flatnonzero(np.diff(night, prepend=-1))
    night_ends = np.append(night_starts[1:], slot_count)
    study_starts = pd.DatetimeIndex(study_slots["interval_start_utc"])
    study_ns = study_starts.as_unit("ns").asi8
    run_starts = pd.DatetimeIndex(run_interval_start_utc)
    commitment = float(assumptions["trading.day_ahead_commitment_share"])
    rule = assumptions.get("trading.commitment_rule", "fixed_share")
    if rule not in COMMITMENT_RULES:
        raise ValueError(f"trading.commitment_rule must be one of {COMMITMENT_RULES}, got {rule!r}")
    newsvendor = rule == "newsvendor"
    if newsvendor and system_short is None:
        raise ValueError("the newsvendor commitment needs system_short")
    gate_slots = int(round(float(assumptions["gate_closure_minutes"]) / 30.0))
    decision_hour = float(assumptions["day_ahead_publication_local_hour"])
    window_slots = int(assumptions["trading.baseline_adjustment_window_slots"])
    windows = None if events is None else model_events.event_slots(events, study_slots)
    all_open = np.ones(slot_count, dtype=bool) if blackout is None else ~blackout

    def mask_at(known_at_utc_ns: int | None) -> np.ndarray:
        if trading_mask is not None:
            return trading_mask(known_at_utc_ns)
        if windows is None:
            return all_open
        return model_events.trading_mask(
            windows, slot_count, known_at_utc=known_at_utc_ns, blackout=blackout
        )

    run_ns = run_starts.as_unit("ns").asi8
    adjustment = None
    if event_adjustment_gbp_per_mwh is not None and len(event_notice_slot):
        adjustment = np.asarray(event_adjustment_gbp_per_mwh, dtype=float)

    def adjustment_at(decision_ns: int, first: int, end: int) -> np.ndarray | None:
        # The planner signals of the requests announced by the decision
        # (notice slot at or before the decision's run slot, as the kernel's
        # planner reads them), for study slots first..end-1 by zone.
        if adjustment is None:
            return None
        known = event_notice_slot <= np.searchsorted(run_ns, decision_ns, side="left")
        if not known.any():
            return None
        return adjustment[known].sum(axis=0)[:, warmup_count + first : warmup_count + end]

    # --- Baseline (§4.1, §9.5) ---
    trap = bool(assumptions["trading.baseline_trap"])
    counts = {
        "working_nights": int(assumptions["trading.baseline_working_nights"]),
        "non_working_nights": int(assumptions["trading.baseline_non_working_nights"]),
    }

    def unadjusted(
        warmup: np.ndarray, study_metered: np.ndarray
    ) -> tuple[np.ndarray, int, list[int]]:
        # Trap mode appends the flexed (metered) study nights; the n - 2 rule
        # inside keeps each night's own and previous night out.
        if trap:
            history = np.concatenate([warmup, study_metered], axis=1)
            slots = pd.concat([warmup_slots, study_slots], ignore_index=True)
        else:
            history, slots = warmup, warmup_slots
        return bl01_lite_baseline(history, slots, study_slots, **counts)

    adjust = bool(assumptions["trading.baseline_in_day_adjustment"])

    def adjusted(
        unadjusted_kwh: np.ndarray, study_metered: np.ndarray, missing: Sequence[int]
    ) -> np.ndarray:
        if not adjust:
            return unadjusted_kwh
        return unadjusted_kwh + in_day_adjustment(
            unadjusted_kwh,
            study_metered,
            study_slots,
            window_slots=window_slots,
            missing_nights=missing,
        )

    baseline0, fallback_nights, missing_nights = unadjusted(warmup_unmanaged_kwh, metered_kwh)
    baseline = adjusted(baseline0, metered_kwh, missing_nights)
    # Scope baselines for grid requests (§2.4): the fleet, then each zone
    # from its own history (the baseline is linear in import, so the zone
    # baselines add up to the fleet's).
    scope_baseline = scope_metered = None
    if zone_warmup_kwh is not None:
        zones = zone_warmup_kwh.shape[1]
        flat_metered = zone_metered_kwh.reshape(world_count * zones, -1)
        zone_b0, _, zone_missing = unadjusted(
            zone_warmup_kwh.reshape(world_count * zones, -1), flat_metered
        )
        zone_b = adjusted(zone_b0, flat_metered, zone_missing).reshape(
            world_count, zones, slot_count
        )
        scope_baseline = np.concatenate([baseline[:, None, :], zone_b], axis=1)
        scope_metered = np.concatenate([metered_kwh[:, None, :], zone_metered_kwh], axis=1)
    event_payment = np.zeros((world_count, slot_count))
    event_delivered = np.zeros((world_count, slot_count))
    delivery = None
    if windows is not None and windows["event_type"].isin(model_events.REQUEST_TYPES).any():
        if scope_baseline is None:
            raise ValueError("grid requests need the zone baselines (zone_warmup_kwh)")
        delivery = model_events.event_delivery(events, study_slots, scope_baseline, scope_metered)
        # Booked on the window's first slot, so the night the window starts
        # in pays it (§4.7).
        start_slot = windows.set_index("event_id")["start_slot"]
        booked = start_slot.reindex(delivery["event_id"]).to_numpy(dtype=np.int64)
        worlds = delivery["world_id"].to_numpy(dtype=np.int64)
        np.add.at(event_payment, (worlds, booked), delivery["payment_gbp"].to_numpy())
        np.add.at(event_delivered, (worlds, booked), delivery["delivered_kwh"].to_numpy())

    # --- Deviation identity and settled volume (§4.2) ---
    deviation = baseline - metered_kwh
    reduction = unmanaged_kwh - metered_kwh
    effect = baseline - unmanaged_kwh
    settlement_open = mask_at(None)
    # Gated on the observable deviation D > 0 inside the settlement mask
    # (lead decision QB), never on R > 0: U is a counterfactual no
    # settlement body meters.  Only turn-down is settled (decision 0005).
    settled = settlement_open * np.maximum(deviation, 0.0)
    # Baseline erosion (decision 0004 item 67): the same settlement against
    # a baseline learnt from the fleet's own smart nights.  A what-if beside
    # the ledger, never inside it: nothing is traded or paid on it.
    flexed0, flexed_missing = flexed_history_baseline(metered_kwh, study_slots, **counts)
    flexed = adjusted(flexed0, metered_kwh, flexed_missing)
    settled_flexed = settlement_open * np.maximum(flexed - metered_kwh, 0.0)

    # --- Expected sessions and prices seen (§4.3-4.5) ---
    typical_start, typical_deadline = typical_session_windows(
        units, study_slots, expected_departure_utc_ns
    )
    plug_probability = units["plug_probability"].to_numpy(dtype=float)
    slot_cap = units["home_charger_limit_kw"].to_numpy(dtype=float) * _SLOT_HOURS
    visible_cache: dict[int, np.ndarray] = {}
    publication = day_ahead_publication_utc_ns(run_starts)

    def visible_study_prices(decision_ns: int) -> np.ndarray:
        # The one B4 rule (action.visible_day_ahead_prices); what is visible
        # changes only at a publication, so each set is built once.
        key = int((publication <= decision_ns).sum())
        if key not in visible_cache:
            visible = visible_day_ahead_prices(day_ahead_gbp_per_mwh, run_starts, decision_ns)
            visible_cache[key] = visible[:, warmup_count:]
        return visible_cache[key]

    day_ahead = day_ahead_gbp_per_mwh[:, warmup_count:]
    q = np.zeros((world_count, slot_count))
    point_forecast = np.zeros((world_count, slot_count))
    commit_level = error_quantile = None
    if newsvendor:
        commit_level = np.zeros((world_count, slot_count))
        error_quantile = np.zeros((world_count, slot_count))
    visible_da = np.zeros((world_count, slot_count))
    expected_metered = np.zeros((world_count, slot_count))
    expected_unmanaged = np.zeros((world_count, slot_count))
    all_unmanaged = np.ones(len(units))
    revised = np.zeros(slot_count, dtype=bool)
    sampled = [int(w) for w in sampled_world_ids]
    updates: list[dict[str, object]] = []
    book_row = {int(s): d for d, s in enumerate(book_slots)}
    local_minute = study_starts.tz_convert(_LONDON).minute

    for n, (first, end) in enumerate(zip(night_starts, night_ends, strict=True)):
        width = end - first
        start, deadline = typical_start[n], typical_deadline[n]

        # Day-ahead decision at 13:00 London on D_n - 1 (§4.4): typical
        # sessions, prices published by then (D_n; D_n + 1 on the typical
        # shape), events known by then, and B0 (history nights <= n - 2).
        # Unchanged by intraday dispatch (dispatch §6.1): the trader then
        # knows only day-ahead prices, and the intraday path is a martingale
        # from them, so a free EV's expected plan on them is its best forecast.
        decision_da = day_ahead_decision_utc_ns(study_slots["local_date"].iat[first], decision_hour)
        prices_da = visible_study_prices(decision_da)[:, first:end]
        visible_da[:, first:end] = prices_da
        # Night 0's decision falls in warm-up night -1, so only the warm-up
        # sessions that began before it may inform the expected need (the
        # same cut-off as the baseline's n - 2 rule, review B3).  Every later
        # decision comes after the whole warm-up.
        need_da = (
            first_night_expected_need
            if n == 0 and first_night_expected_need is not None
            else expected_need
        )
        sessions = expected_sessions(plug_probability, need_da, slot_cap, start, deadline)
        expected_da = expected_smart_kwh(
            sessions,
            prices_da,
            first_slot=first,
            non_response_probability=non_response_probability,
            price_adjustment_gbp_per_mwh=adjustment_at(decision_da, first, end),
            zone_index=zone_index,
            blackout=None if blackout is None else blackout[first:end],
        )
        expected_metered[:, first:end] = expected_da
        # x^U (supplier contract §2): the same expected sessions with every
        # session unmanaged (rho = 1, so the planner is skipped), so
        # x^U - x is the expected turn-down alone: the typical-clock bias
        # and the control EVs are in both and cancel.
        expected_unmanaged[:, first:end] = expected_smart_kwh(
            sessions, prices_da, first_slot=first, non_response_probability=all_unmanaged
        )
        mask_da = mask_at(decision_da)[first:end]
        target_da = positions_kwh(baseline0[:, first:end], expected_da, mask_da)
        point_forecast[:, first:end] = target_da
        if newsvendor:
            # §10.4: the point forecast plus the other weeks' error quantile
            # at the level P_DA / k.  V is position-free (§4.2), so every
            # world's V is known here; each world reads only the others'.
            # P_DA is the price visible at the decision, not the realised one.
            (
                q[:, first:end],
                commit_level[:, first:end],
                error_quantile[:, first:end],
            ) = newsvendor_positions(
                target_da,
                settled[:, first:end],
                prices_da,
                imbalance_gbp_per_mwh[:, first:end],
                system_short[:, first:end],
                mask_da,
            )
        else:
            # Commitment share (§9.2): only c of the forecast is sold
            # day-ahead; intraday targets below do not depend on c.
            q[:, first:end] = commitment * target_da
        for w in sampled:
            for k in range(width):
                t = first + k
                updates.append(
                    _update_row(
                        w,
                        t,
                        n,
                        decision_da,
                        "day_ahead",
                        math.ceil(
                            (study_ns[t] - gate_slots * _HALF_HOUR_NS - decision_da) / _HOUR_NS
                        ),
                        baseline0[w, t],
                        expected_da[w, k],
                        q[w, t],
                        q[w, t],
                        day_ahead[w, t],
                        q[w, t],
                        q[w, t],
                    )
                )

    # The one latest-price rule (dispatch §3, action.latest_known_prices):
    # h = ceil((gate - tau) / 1 h), clipped to the path.
    gate_ns = run_ns - gate_slots * _HALF_HOUR_NS
    # For the traded price every slot counts as published: before the
    # day-ahead publication the path holds the expected close, the price the
    # F2 trader has always booked a trade on an unpublished slot at (the
    # 12:00 decision, for the next day's slots).  Ranking a plan instead
    # uses the real publication instants, so an unpublished slot ranks the
    # B4 expected shape, as the dispatcher does.
    trade_publication = np.full_like(publication, np.iinfo(np.int64).min)
    free = None if dispatch_locked is None else ~np.asarray(dispatch_locked, dtype=bool)

    def reposition(
        book: np.ndarray, free_evs: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple]]:
        """Hourly intraday re-positioning of ``full`` on one plan book (§4.5).

        Returns the final positions, intraday cash (GBP) and traded volume
        (kWh) per delivery slot, and one record per sampled (world, slot,
        decision) for ``position_updates``, in loop order.  ``free_evs``
        (EV,) bool or None: which expected sessions rank the latest price.
        """

        final = q.copy()
        cash = np.zeros((world_count, slot_count))
        volume = np.zeros((world_count, slot_count))
        records: list[tuple] = []
        for n, (first, end) in enumerate(zip(night_starts, night_ends, strict=True)):
            start, deadline = typical_start[n], typical_deadline[n]
            # Every whole London hour from the night's start while some slot
            # of the night is still open.  ``plugged[:, k]`` is the
            # connection state of slot first - 1 + k; ``began[:, k]`` is True
            # once a session has begun in this night by slot first + k (a
            # session carried over from the previous night did not begin in
            # this one).
            plugged = connected[:, first : end + 1, :]
            began = np.logical_or.accumulate(plugged[:, 1:] & ~plugged[:, :-1], axis=1)
            for s in range(first, end):
                open_first = s + gate_slots
                if local_minute[s] != 0 or open_first >= end:
                    continue
                decision_ns = int(study_ns[s])
                # EVs plugged in at the decision are in the book with their
                # plans in force; EVs whose session of this night has come
                # and gone are done.  The rest are expected from their
                # typical clocks, re-based to s (review B1).  Only connection
                # states of slots before s are read, all known at the decision.
                in_book = plugged[:, s - first, :]
                done = began[:, s - first - 1, :] if s > first else np.zeros_like(in_book)
                include = ~in_book & ~done
                sessions = expected_sessions(
                    plug_probability,
                    expected_need,
                    slot_cap,
                    start,
                    deadline,
                    decision_slot=s,
                    include=include,
                )
                visible = visible_study_prices(decision_ns)[:, s:end]
                ahead = slice(warmup_count + s, warmup_count + end)
                path_ahead = intraday_path_gbp_per_mwh[:, ahead]
                latest = None
                if free_evs is not None:
                    # Dispatch §6.1: a free session not yet started is
                    # expected to plan on the price the dispatcher would rank
                    # now, the same h as the re-plans made at this decision.
                    latest = latest_known_prices(
                        path_ahead, visible, publication[ahead], gate_ns[ahead], decision_ns
                    )
                expected_id = expected_smart_kwh(
                    sessions,
                    visible,
                    first_slot=s,
                    non_response_probability=non_response_probability,
                    price_adjustment_gbp_per_mwh=adjustment_at(decision_ns, s, end),
                    zone_index=zone_index,
                    blackout=None if blackout is None else blackout[s:end],
                    free=free_evs,
                    free_window_price_gbp_per_mwh=latest,
                )
                forecast = book[:, book_row[s], : end - s] + expected_id
                # B0 until the in-day adjustment is known at 13:30 on D_n.
                known_baseline = baseline if s >= first + window_slots else baseline0
                targets = positions_kwh(
                    known_baseline[:, s:end], forecast, mask_at(decision_ns)[s:end]
                )
                # The latest update made at or before the decision: ceil
                # picks one that has happened, floor would read the future
                # (review B6).  Gate is 60 min before the slot start.
                traded_price = latest_known_prices(
                    path_ahead, visible, trade_publication[ahead], gate_ns[ahead], decision_ns
                )
                for t in range(open_first, end):
                    trade = targets[:, t - s] - final[:, t]
                    steps = math.ceil(
                        ((t - s) * _HALF_HOUR_NS - gate_slots * _HALF_HOUR_NS) / _HOUR_NS
                    )
                    price = traded_price[:, t - s]
                    cash[:, t] += trade * price / 1000.0
                    volume[:, t] += np.abs(trade)
                    final[:, t] = targets[:, t - s]
                    revised[t] = True
                    for w in sampled:
                        records.append(
                            (
                                w,
                                t,
                                n,
                                decision_ns,
                                "intraday",
                                steps,
                                known_baseline[w, t],
                                forecast[w, t - s],
                                targets[w, t - s],
                                trade[w],
                                price[w],
                            )
                        )
        return final, cash, volume, records

    final_full, intraday_cash, intraday_volume, records = reposition(book_kwh, free)
    if day_ahead_book_kwh is None:
        # Dispatch off: the day-ahead plan path is the selected path, so the
        # frozen book is the selected book and the re-run would repeat the
        # actual trades exactly; re-optimisation is 0 (item 58: "intraday P&L
        # in v1 is all rebalancing").
        frozen_cash, frozen_volume, frozen_records = intraday_cash, intraday_volume, records
    else:
        # The frozen-book re-run (dispatch §6.2, B2): the fleet held on its
        # day-ahead plans (their book, every expected plan on B4-visible
        # day-ahead prices) with the same q, masks, actual baseline and
        # traded prices.  Its trades are rebalancing; the rest of the actual
        # trades are what the dispatch's response caused.  Rejected: valuing
        # the same trades at day-ahead, and freezing the re-run's prices.
        _, frozen_cash, frozen_volume, frozen_records = reposition(day_ahead_book_kwh, None)
    # Both runs walk the same (night, decision, slot, world) grid, so their
    # records pair one to one; the frozen run adds its target and trade.
    for record, frozen in zip(records, frozen_records, strict=True):
        position, trade = frozen[8], frozen[9]
        updates.append(_update_row(*record, position, trade))

    positions = {"da_only": q, "full": final_full, "perfect_foresight": settled.copy()}
    ledger_inputs = {
        "night_starts": night_starts,
        "settled": settled,
        "effect": effect,
        "reduction": reduction,
        "day_ahead": day_ahead,
        "imbalance_price": imbalance_gbp_per_mwh,
        "kind": shock_kind(known_shock_gw, surprise_shock_gw),
        "event_payment": event_payment,
        "event_delivered": event_delivered,
        "unmet_kwh": np.maximum(
            _night_sums(departure_shortfall_kwh["selected"], night_starts)
            - _night_sums(departure_shortfall_kwh["normal"], night_starts),
            0.0,
        ),
    }
    per_strategy = {
        "da_only": (q, q, None, None),
        "full": (q, final_full, intraday_cash, intraday_volume),
        "perfect_foresight": (settled, settled, None, None),
    }
    rebalancing = {"full": {"rebalancing_cash": frozen_cash, "rebalancing_volume": frozen_volume}}
    ledger = pd.concat(
        [
            settle(
                strategy,
                *per_strategy[strategy],
                study_slots,
                ledger_inputs,
                assumptions,
                **rebalancing.get(strategy, {}),
            )
            for strategy in STRATEGIES
        ],
        ignore_index=True,
    )
    order = {s: i for i, s in enumerate(STRATEGIES)}
    ledger = (
        ledger.sort_values(
            ["world_id", "strategy", "night_index"],
            key=lambda c: c.map(order) if c.name == "strategy" else c,
            kind="stable",
        )
        .reset_index(drop=True)
        .loc[:, list(LEDGER_COLUMNS)]
    )
    deviation_frame = _deviation_frame(
        study_slots,
        baseline,
        unmanaged_kwh,
        metered_kwh,
        expected_metered,
        expected_unmanaged,
        settled,
        settlement_open,
        positions,
        commit_level,
        error_quantile,
        day_ahead,
        intraday_close_gbp_per_mwh,
        imbalance_gbp_per_mwh,
        known_shock_gw,
        surprise_shock_gw,
    )
    control_baseline = control_unmanaged = None
    if control_mask is not None:
        # The baseline is linear in import, so the control group's own
        # baseline (same rule, its own history) and the treated group's
        # B - B_c need no extra scope (§9.5).  Control EVs never follow a
        # plan, so their metered import is their unmanaged import.
        control_baseline0, _, control_missing = unadjusted(
            control_warmup_kwh, control_unmanaged_kwh
        )
        control_baseline = adjusted(control_baseline0, control_unmanaged_kwh, control_missing)
        control_unmanaged = control_unmanaged_kwh
    # The full strategy's cash by delivery slot, for the replay's running
    # figure (replay contract v1 §1.4): the same components ``settle`` sums.
    slot_cash = sum(
        slot_cash_components(
            q, final_full, intraday_cash, intraday_volume, ledger_inputs, assumptions
        ).values()
    )
    return TradingRun(
        deviation_world_slot=deviation_frame,
        position_updates=_updates_frame(updates, study_slots),
        trading_ledger_world=ledger,
        trading_week_world=week_frame(ledger),
        baseline_unadjusted=baseline0,
        baseline=baseline,
        settled=settled,
        day_ahead_position=q,
        positions=positions,
        visible_day_ahead_gbp_per_mwh=visible_da,
        revised_slots=revised,
        fallback_nights=fallback_nights,
        missing_nights=len(missing_nights),
        commitment_share=commitment,
        commitment_rule=rule,
        day_ahead_forecast=point_forecast,
        commit_level=commit_level,
        forecast_error_quantile=error_quantile,
        control_baseline=control_baseline,
        control_unmanaged=control_unmanaged,
        scope_baseline=scope_baseline,
        event_delivery=delivery,
        slot_cash_gbp=slot_cash,
        settled_flexed_baseline=settled_flexed,
    )


def shock_kind(known_shock_gw: np.ndarray, surprise_shock_gw: np.ndarray) -> np.ndarray:
    """(world, slot) object array: ``surprise``, ``known`` or ``none`` (contract §0).

    Surprise wins because it is what moves intraday and imbalance.
    """

    return np.where(
        surprise_shock_gw != 0.0, "surprise", np.where(known_shock_gw != 0.0, "known", "none")
    ).astype(object)


_SLOT_BUCKETS = (
    "day_ahead_revenue_gbp",
    "intraday_pnl_gbp",
    "trading_cost_gbp",
    "imbalance_gbp",
    "baseline_effect_gbp",
    "supplier_compensation_gbp",
)
"""The six ledger buckets that resolve by delivery slot (contract §4.7), in ``BUCKETS`` order."""


def _volume_cost(rate_gbp_per_mwh: float, volume_kwh: np.ndarray) -> np.ndarray:
    """A cost priced per MWh of volume, as a negative GBP amount: ``-rate x kWh / 1000``."""

    return -float(rate_gbp_per_mwh) * volume_kwh / 1000.0


def slot_cash_components(
    day_ahead_position: np.ndarray,
    final_position: np.ndarray,
    intraday_cash: np.ndarray | None,
    intraday_volume: np.ndarray | None,
    inputs: Mapping[str, np.ndarray],
    assumptions: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """The six slot-resolved ledger buckets of one strategy, (world, slot) GBP each.

    Arguments as ``settle``'s.  Keys are ``_SLOT_BUCKETS``: day-ahead
    revenue ``q P_DA`` less the baseline effect carved out of it, intraday
    P&L ``sum Delta P_ID``, trading cost ``-s sum |Delta|``, imbalance
    ``(V - f) SIP``, the baseline effect ``E P_DA`` where ``V > 0``, and
    supplier compensation ``-c_sup V`` (all /1000).  The grid-event
    payment, customer share and penalty are not slot-resolved and stay in
    ``settle``.  One formula for the ledger (``settle`` night-sums these)
    and for the replay's running cash (replay contract v1 §1.4), so the two
    cannot drift.  Illustrative simulated trading P&L, never Axle cash.
    """

    settled, effect = inputs["settled"], inputs["effect"]
    price, sip = inputs["day_ahead"], inputs["imbalance_price"]
    zeros = np.zeros_like(settled)
    cash = intraday_cash if intraday_cash is not None else zeros
    volume = intraday_volume if intraday_volume is not None else zeros
    effect_cash = np.where(settled > 0.0, effect * price, 0.0) / 1000.0
    return {
        "day_ahead_revenue_gbp": day_ahead_position * price / 1000.0 - effect_cash,
        "intraday_pnl_gbp": cash,
        "trading_cost_gbp": _volume_cost(assumptions["trading.half_spread_gbp_per_mwh"], volume),
        "imbalance_gbp": (settled - final_position) * sip / 1000.0,
        "baseline_effect_gbp": effect_cash,
        "supplier_compensation_gbp": _volume_cost(
            assumptions["trading.supplier_compensation_gbp_per_mwh"], settled
        ),
    }


def settle(
    strategy: str,
    day_ahead_position: np.ndarray,
    final_position: np.ndarray,
    intraday_cash: np.ndarray | None,
    intraday_volume: np.ndarray | None,
    study_slots: pd.DataFrame,
    inputs: Mapping[str, np.ndarray],
    assumptions: Mapping[str, float],
    *,
    rebalancing_cash: np.ndarray | None = None,
    rebalancing_volume: np.ndarray | None = None,
) -> pd.DataFrame:
    """The ledger rows of one strategy: one per (world, night) (contract §4.7, §5.3).

    ``day_ahead_position`` ``q`` and ``final_position`` ``f`` are (world,
    slot) kWh; ``intraday_cash`` (GBP) and ``intraday_volume`` (kWh, sum of
    |trade|) per delivery slot, None for a strategy with no intraday trades.
    ``inputs`` carries ``settled`` ``V``, ``effect`` ``E``, ``reduction``
    ``R``, the day-ahead and imbalance prices, the slot shock ``kind``, the
    grid-event payment and delivered volume and ``unmet_kwh`` (world,
    night).  Per slot the trading cash is
    ``C = q P_DA + sum Delta P_ID + (V - f) SIP`` (/1000).

    The baseline effect is carved out of day-ahead revenue at the day-ahead
    price (lead decision QC): ``B0`` is known at the day-ahead decision, so
    the expected baseline effect is what the day-ahead position sells.  A
    proportional ``R : E`` split of each slot's cash was rejected: ``E/D``
    is unbounded as ``D -> 0``.  The customer's share is taken on each
    night's positive gross (more generous to the customer than weekly
    settlement, §8 Q7).  ``net_gbp`` is the exact sum of the nine buckets.

    ``rebalancing_cash`` (GBP) and ``rebalancing_volume`` (kWh) per delivery
    slot are the frozen-book re-run's intraday cash and |trade| (dispatch
    contract §6.2); None (no intraday trades) gives 0.  The four "of which"
    columns are ``intraday_pnl_rebalancing_gbp = sum Delta0 P_ID / 1000``,
    ``trading_cost_rebalancing_gbp = -s sum |Delta0| / 1000`` and each
    re-optimisation column the bucket minus its rebalancing part, so each
    pair sums to its bucket and is exactly (bucket, 0) when the re-run
    repeats the actual trades.  The trading-cost remainder may be positive:
    the dispatch's trades can net against the forecast's.
    """

    starts = inputs["night_starts"]
    settled, effect = inputs["settled"], inputs["effect"]
    kind = inputs["kind"]
    world_count = settled.shape[0]
    zeros = np.zeros_like(settled)
    volume = intraday_volume if intraday_volume is not None else zeros
    settles = settled > 0.0
    components = slot_cash_components(
        day_ahead_position, final_position, intraday_cash, intraday_volume, inputs, assumptions
    )
    slot_values = {name: components[name] for name in ATTRIBUTED_BUCKETS}
    rows: dict[str, np.ndarray] = {
        name: _night_sums(components[name], starts) for name in _SLOT_BUCKETS
    }
    # The two volume-priced buckets are linear in volume, so the night's
    # volume is priced with the same helper instead of summing the slot
    # values: equal up to rounding, and it keeps the ledger bit-identical to
    # its form before ``slot_cash_components`` existed (replay contract §5).
    rows["trading_cost_gbp"] = _volume_cost(
        assumptions["trading.half_spread_gbp_per_mwh"], _night_sums(volume, starts)
    )
    rows["supplier_compensation_gbp"] = _volume_cost(
        assumptions["trading.supplier_compensation_gbp_per_mwh"], _night_sums(settled, starts)
    )
    rows["grid_event_payment_gbp"] = _night_sums(inputs["event_payment"], starts)
    rows = {name: rows[name] for name in BUCKETS[:7]}
    gross = sum(rows.values())
    rows["customer_revenue_share_gbp"] = -float(
        assumptions["trading.customer_revenue_share"]
    ) * np.maximum(gross, 0.0)
    rows["unmet_charge_penalty_gbp"] = (
        -float(assumptions["trading.unmet_charge_penalty_gbp_per_kwh"]) * inputs["unmet_kwh"]
    )
    rows["net_gbp"] = sum(rows[bucket] for bucket in BUCKETS)
    # Dispatch split (§6.2): by volume at the traded prices; the
    # re-optimisation part is the remainder, so the pair sums to the bucket.
    rebalancing_pnl = _night_sums(
        rebalancing_cash if rebalancing_cash is not None else zeros, starts
    )
    rebalancing_cost = _volume_cost(
        assumptions["trading.half_spread_gbp_per_mwh"],
        _night_sums(rebalancing_volume if rebalancing_volume is not None else zeros, starts),
    )
    rows["intraday_pnl_rebalancing_gbp"] = rebalancing_pnl
    rows["intraday_pnl_reoptimisation_gbp"] = rows["intraday_pnl_gbp"] - rebalancing_pnl
    rows["trading_cost_rebalancing_gbp"] = rebalancing_cost
    rows["trading_cost_reoptimisation_gbp"] = rows["trading_cost_gbp"] - rebalancing_cost
    # "Of which" columns: each bucket is a sum over slots, so it splits
    # exactly by the slot's shock kind (item 56); not buckets themselves.
    for bucket, values in slot_values.items():
        for k in SHOCK_KINDS:
            rows[f"{bucket[: -len('_gbp')]}_{k}_gbp"] = _night_sums(
                np.where(kind == k, values, 0.0), starts
            )
    for k in SHOCK_KINDS:
        rows[f"settled_mwh_{k}"] = _night_sums(np.where(kind == k, settled, 0.0), starts) / 1000.0
    mwh = {
        "sold_day_ahead_mwh": day_ahead_position,
        "final_position_mwh": final_position,
        "settled_mwh": settled,
        "flexibility_mwh": np.where(settles, inputs["reduction"], 0.0),
        "baseline_effect_mwh": np.where(settles, effect, 0.0),
        "imbalance_mwh": settled - final_position,
        "abs_imbalance_mwh": np.abs(settled - final_position),
        "intraday_traded_mwh": volume,
        "delivered_within_final_mwh": np.minimum(settled, final_position),
        "delivered_within_day_ahead_mwh": np.minimum(settled, day_ahead_position),
        "event_delivered_mwh": inputs["event_delivered"],
    }
    for name, values in mwh.items():
        rows[name] = _night_sums(values, starts) / 1000.0
    rows["unmet_charge_kwh"] = inputs["unmet_kwh"]

    night_count = len(starts)
    local_date = study_slots["local_date"].to_numpy()[starts]
    day_label = study_slots["day_label"].to_numpy()[starts]
    frame = pd.DataFrame(
        {
            "world_id": np.repeat(np.arange(world_count, dtype=np.int64), night_count),
            "strategy": strategy,
            "night_index": np.tile(np.arange(night_count, dtype=np.int64), world_count),
            "night_start_local_date": pd.Series(np.tile(local_date, world_count), dtype=object),
            "day_label": pd.Series(np.tile(day_label, world_count), dtype=object),
            **{name: np.asarray(values, dtype=float).reshape(-1) for name, values in rows.items()},
            "evidence_kind": EVIDENCE_KIND,
        }
    )
    return frame


def week_frame(ledger: pd.DataFrame) -> pd.DataFrame:
    """``trading_week_world`` (§5.4): each world's seven night rows summed, per strategy."""

    numeric = [c for c in WEEK_COLUMNS if c not in ("world_id", "strategy", "evidence_kind")]
    week = ledger.groupby(["world_id", "strategy"], sort=False, as_index=False)[numeric].sum()
    week["evidence_kind"] = EVIDENCE_KIND
    return week.loc[:, list(WEEK_COLUMNS)].reset_index(drop=True)


def _night_sums(values: np.ndarray, night_starts: np.ndarray) -> np.ndarray:
    """(world, slot) -> (world, night) sums; nights are contiguous slot runs."""

    return np.add.reduceat(values, night_starts, axis=1)


def _slot_ns(study_slots: pd.DataFrame) -> np.ndarray:
    """Study slot starts as int64 UTC ns, indexed by ``slot_index`` (0, 1, ...)."""

    return pd.DatetimeIndex(study_slots["interval_start_utc"]).as_unit("ns").asi8


def _update_row(
    world: int,
    slot: int,
    night: int,
    decision_ns: int,
    stage: str,
    hours: int,
    baseline: float,
    forecast: float,
    position: float,
    trade: float,
    price: float,
    frozen_position: float,
    frozen_trade: float,
) -> dict[str, object]:
    return {
        "world_id": world,
        "slot_index": slot,
        "night_index": night,
        "decision_ns": decision_ns,
        "stage": stage,
        "hours_to_gate_closure": hours,
        "baseline_known_kwh": float(baseline),
        "forecast_metered_kwh": float(forecast),
        "position_kwh": float(position),
        "trade_kwh": float(trade),
        "price_gbp_per_mwh": float(price),
        "frozen_position_kwh": float(frozen_position),
        "frozen_trade_kwh": float(frozen_trade),
    }


def _updates_frame(updates: list[dict[str, object]], study_slots: pd.DataFrame) -> pd.DataFrame:
    """``position_updates`` (§5.2), sorted by world, slot and decision."""

    frame = pd.DataFrame(
        updates,
        columns=[
            "world_id",
            "slot_index",
            "night_index",
            "decision_ns",
            "stage",
            "hours_to_gate_closure",
            "baseline_known_kwh",
            "forecast_metered_kwh",
            "position_kwh",
            "trade_kwh",
            "price_gbp_per_mwh",
            "frozen_position_kwh",
            "frozen_trade_kwh",
        ],
    )
    frame = frame.astype(
        {
            "world_id": np.int64,
            "slot_index": np.int64,
            "night_index": np.int64,
            "decision_ns": np.int64,
            "hours_to_gate_closure": np.int64,
            "stage": object,
        }
    )
    frame = frame.sort_values(["world_id", "slot_index", "decision_ns"], kind="stable")
    frame["interval_start_utc"] = pd.to_datetime(
        _slot_ns(study_slots)[frame["slot_index"].to_numpy()], utc=True
    )
    frame["interval_start_london"] = frame["interval_start_utc"].dt.tz_convert(_LONDON)
    decision = pd.to_datetime(frame.pop("decision_ns").to_numpy(), utc=True).as_unit("ns")
    frame["decision_utc"] = decision
    frame["decision_london"] = decision.tz_convert(_LONDON)
    frame["evidence_kind"] = EVIDENCE_KIND
    return frame.loc[:, list(POSITION_UPDATE_COLUMNS)].reset_index(drop=True)


def _deviation_frame(
    study_slots: pd.DataFrame,
    baseline: np.ndarray,
    unmanaged: np.ndarray,
    metered: np.ndarray,
    expected_metered: np.ndarray,
    expected_unmanaged: np.ndarray,
    settled: np.ndarray,
    settlement_open: np.ndarray,
    positions: Mapping[str, np.ndarray],
    commit_level: np.ndarray | None,
    error_quantile: np.ndarray | None,
    day_ahead: np.ndarray,
    intraday_close: np.ndarray,
    imbalance: np.ndarray,
    known: np.ndarray,
    surprise: np.ndarray,
) -> pd.DataFrame:
    """``deviation_world_slot`` (§5.1): one row per (world, slot), world-major.

    ``expected_metered_kwh`` is the aggregator's expected metered import at
    each night's day-ahead decision (``x_t`` of §4.4, kWh per half-hour), an
    addition to §5.1 for the supplier P&L view; ``expected_unmanaged_kwh``
    is ``x^U_t`` of supplier contract §2, the same day-ahead expected
    sessions with every session unmanaged, kWh per half-hour.
    ``commit_level`` and ``forecast_error_quantile_kwh`` are the newsvendor
    ``alpha`` and ``e(alpha)`` (§10.4), NaN when those arrays are None
    (fixed-share mode).
    """

    world_count, slot_count = baseline.shape
    unset = np.full((world_count, slot_count), np.nan)

    def flat(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(-1)

    frame = pd.DataFrame(
        {
            "world_id": np.repeat(np.arange(world_count, dtype=np.int64), slot_count),
            "slot_index": np.tile(study_slots["slot_index"].to_numpy(dtype=np.int64), world_count),
            "interval_start_utc": np.tile(_slot_ns(study_slots), world_count),
            "night_index": np.tile(
                study_slots["night_index"].to_numpy(dtype=np.int64), world_count
            ),
            "baseline_kwh": flat(baseline),
            "unmanaged_kwh": flat(unmanaged),
            "metered_kwh": flat(metered),
            "baseline_kw": flat(baseline) / _SLOT_HOURS,
            "unmanaged_kw": flat(unmanaged) / _SLOT_HOURS,
            "metered_kw": flat(metered) / _SLOT_HOURS,
            "expected_metered_kwh": flat(expected_metered),
            "expected_unmanaged_kwh": flat(expected_unmanaged),
            "deviation_kwh": flat(baseline - metered),
            "true_reduction_kwh": flat(unmanaged - metered),
            "baseline_effect_kwh": flat(baseline - unmanaged),
            "settled_kwh": flat(settled),
            "settlement_open": np.tile(np.asarray(settlement_open, dtype=bool), world_count),
            **{f"position_{s}_kwh": flat(positions[s]) for s in STRATEGIES},
            "commit_level": flat(unset if commit_level is None else commit_level),
            "forecast_error_quantile_kwh": flat(
                unset if error_quantile is None else error_quantile
            ),
            "day_ahead_gbp_per_mwh": flat(day_ahead),
            "intraday_close_gbp_per_mwh": flat(intraday_close),
            "imbalance_gbp_per_mwh": flat(imbalance),
            "known_shock_gw": flat(known),
            "surprise_shock_gw": flat(surprise),
            "shock_kind": shock_kind(known, surprise).reshape(-1),
            "evidence_kind": EVIDENCE_KIND,
        }
    )
    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True)
    frame["interval_start_london"] = frame["interval_start_utc"].dt.tz_convert(_LONDON)
    return frame.loc[:, list(DEVIATION_COLUMNS)]


# ============================================================================
# 9.5 Hold-out control group
# ============================================================================


def control_group_mask(
    units: pd.DataFrame, order: np.ndarray, share: float, cohort_order: Sequence[str]
) -> np.ndarray:
    """Control-group EVs (EV,) bool, stratified by cohort (contract §9.5, §9.10 Q4).

    ``order`` is the run's one control permutation of the EV indices (drawn
    whether or not the group is on); ``share`` the control share (fraction).
    The control size ``K = round(share x EVs)`` (half up) is split across
    cohorts by largest remainder on ``K x cohort count / EVs`` (ties to the
    earlier cohort in ``cohort_order``), and each cohort's first ``k_c`` EVs
    in permutation order are the control.  Stratifying keeps the control
    group's archetype mix equal to the fleet's, so its baseline error
    measures the fleet's.  Raises ``ValueError`` unless both groups have at
    least one EV.
    """

    cohorts = units["cohort_id"].to_numpy(dtype=object)
    vehicle_count = len(cohorts)
    size = math.floor(share * vehicle_count + 0.5)
    if not 1 <= size <= vehicle_count - 1:
        raise ValueError("the control group and the treated group each need at least one EV")
    present = [c for c in cohort_order if (cohorts == c).any()]
    counts = np.array([(cohorts == c).sum() for c in present])
    # Largest remainder in whole numbers (size x count = quota x EVs +
    # remainder), so equal remainders tie exactly; a stable sort on the
    # negated remainder gives a tie to the earlier cohort.
    per_cohort, remainder = np.divmod(size * counts, vehicle_count)
    for index in np.argsort(-remainder, kind="stable")[: size - per_cohort.sum()]:
        per_cohort[index] += 1
    ranked = cohorts[np.asarray(order)]
    control = np.zeros(vehicle_count, dtype=bool)
    for cohort_id, k in zip(present, per_cohort, strict=True):
        control[np.asarray(order)[ranked == cohort_id][:k]] = True
    return control


# ============================================================================
# 9.4 The supplier's own price curve
# ============================================================================

USER_CURVE_COLUMNS = ("half_hour_start", "price_gbp_per_mwh")
_USER_CURVE_MAX_BYTES = 10_000


def validate_user_price_curve(
    csv_text: str | bytes, *, floor_gbp_per_mwh: float, cap_gbp_per_mwh: float
) -> pd.DataFrame:
    """Check an uploaded 48-value price curve (contract §9.4) and return it sorted.

    ``csv_text`` is the file's UTF-8 text (at most 10 kB) with exactly the
    header ``half_hour_start,price_gbp_per_mwh`` and 48 rows: each London
    ``HH:MM`` half-hour ``00:00`` ... ``23:30`` once, each price a finite
    number inside ``[floor, cap]`` GBP/MWh (the prices lane's own limits).
    Returns a 48-row frame sorted by half-hour (``half_hour_start`` object,
    ``price_gbp_per_mwh`` float64) or raises ``ValueError`` naming the row
    and the rule.
    """

    raw = csv_text.encode("utf-8") if isinstance(csv_text, str) else bytes(csv_text)
    if len(raw) > _USER_CURVE_MAX_BYTES:
        raise ValueError("price curve file must be at most 10 kB")
    try:
        frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"price curve is not a readable CSV: {error}") from None
    if tuple(frame.columns) != USER_CURVE_COLUMNS:
        raise ValueError("price curve header must be exactly half_hour_start,price_gbp_per_mwh")
    if len(frame) != 48:
        raise ValueError(f"price curve must have 48 rows, not {len(frame)}")
    labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    seen: set[str] = set()
    prices = np.empty(48)
    for row, (label, value) in enumerate(frame.itertuples(index=False), start=1):
        label = label.strip()
        if label not in labels:
            raise ValueError(f"row {row}: half_hour_start {label!r} is not a London half-hour")
        if label in seen:
            raise ValueError(f"row {row}: half_hour_start {label} is repeated")
        seen.add(label)
        try:
            price = float(value)
        except ValueError:
            raise ValueError(f"row {row}: price {value!r} is not a number") from None
        if not math.isfinite(price):
            raise ValueError(f"row {row}: price must be finite")
        if not floor_gbp_per_mwh <= price <= cap_gbp_per_mwh:
            raise ValueError(
                f"row {row}: price {price:g} is outside {floor_gbp_per_mwh:g} to "
                f"{cap_gbp_per_mwh:g} GBP/MWh"
            )
        prices[labels.index(label)] = price
    return pd.DataFrame(
        {"half_hour_start": pd.Series(labels, dtype=object), "price_gbp_per_mwh": prices}
    )


def typical_day_ahead_shape(
    day_ahead_gbp_per_mwh: np.ndarray, interval_start_utc: pd.DatetimeIndex
) -> np.ndarray:
    """The generated typical daily shape, (day class, London half-hour) GBP/MWh.

    Day class 0 is weekday and 1 weekend by the slot's London date, as the
    generator's demand shape uses them.  Each cell is the mean day-ahead
    price over every world and every run slot of that class and half-hour:
    the shape the run's own prices have, so a user curve equal to it
    changes nothing (§9.4).  A cell with no slot takes its class's mean.
    """

    local = pd.DatetimeIndex(interval_start_utc).tz_convert(_LONDON)
    half_hour = np.asarray(local.hour * 2 + local.minute // 30)
    weekend = np.asarray(local.dayofweek >= 5)
    shape = np.empty((2, 48))
    for k, in_class in enumerate((~weekend, weekend)):
        mean = day_ahead_gbp_per_mwh[:, in_class].mean() if in_class.any() else np.nan
        for h in range(48):
            cells = in_class & (half_hour == h)
            shape[k, h] = day_ahead_gbp_per_mwh[:, cells].mean() if cells.any() else mean
    return shape


def apply_user_price_curve(
    prices: MarketPrices,
    curve: pd.DataFrame,
    interval_start_utc: pd.DatetimeIndex,
    *,
    floor_gbp_per_mwh: float,
    cap_gbp_per_mwh: float,
) -> MarketPrices:
    """Replace the typical daily shape with the supplier's curve (contract §9.4). Pure.

    ``curve`` is ``validate_user_price_curve``'s frame; ``interval_start_utc``
    the run slots the prices cover.  Per slot with London half-hour ``h`` and
    day class ``k``: ``offset = curve[h] - typical_shape[k, h]``, added to
    the day-ahead price, every intraday step (so the close) and the
    imbalance price, each then re-clipped to the prices lane's floor and cap
    so a curve never produces a price the generator could not (lead review
    B4).  Moving all three by the same offset keeps ``ID - DA`` and ``SIP -
    DA`` wherever no clip binds; shifting day-ahead alone would hand the
    trader a shape-dependent free spread.  The level walk, noise, shocks and
    premia stay on top, so worlds still differ.  No draw is made, so every
    random channel is identical with and without a curve.  Shock increments
    stay as generated on the generator's own net demand (a simplification).
    """

    starts = pd.DatetimeIndex(interval_start_utc)
    world_count = prices.intraday_path_gbp_per_mwh.shape[0]
    day_ahead = (
        prices.day_ahead["wholesale_forecast_gbp_per_mwh"]
        .to_numpy(dtype=float)
        .reshape(world_count, len(starts))
    )
    shape = typical_day_ahead_shape(day_ahead, starts)
    local = starts.tz_convert(_LONDON)
    half_hour = np.asarray(local.hour * 2 + local.minute // 30)
    weekend = np.asarray(local.dayofweek >= 5).astype(int)
    curve_values = curve["price_gbp_per_mwh"].to_numpy(dtype=float)
    offset = curve_values[half_hour] - shape[weekend, half_hour]

    def shifted(values: np.ndarray) -> np.ndarray:
        return np.clip(
            values + offset.reshape((1, -1) + (1,) * (values.ndim - 2)),
            floor_gbp_per_mwh,
            cap_gbp_per_mwh,
        )

    new_day_ahead = shifted(day_ahead)
    path = shifted(prices.intraday_path_gbp_per_mwh)
    imbalance = shifted(
        prices.realised["imbalance_price_gbp_per_mwh"]
        .to_numpy(dtype=float)
        .reshape(world_count, len(starts))
    )
    day_ahead_frame = prices.day_ahead.assign(
        wholesale_forecast_gbp_per_mwh=new_day_ahead.reshape(-1)
    )
    realised = prices.realised.assign(
        evaluation_context_price_gbp_per_mwh=path[:, :, 0].reshape(-1),
        imbalance_price_gbp_per_mwh=imbalance.reshape(-1),
    )
    return replace(
        prices, day_ahead=day_ahead_frame, realised=realised, intraday_path_gbp_per_mwh=path
    )


__all__ = [
    "ATTRIBUTION_COLUMNS",
    "BOOK_WIDTH",
    "BUCKETS",
    "DEVIATION_COLUMNS",
    "DISPATCH_SPLIT_COLUMNS",
    "LEDGER_COLUMNS",
    "MONEY_COLUMNS",
    "PERFECT_FORESIGHT_CAPTION",
    "PERFECT_FORESIGHT_CAPTION_METRICS",
    "POSITION_UPDATE_COLUMNS",
    "SHOCK_KINDS",
    "STRATEGIES",
    "STRATEGY_CAPTIONS",
    "TradingRun",
    "USER_CURVE_COLUMNS",
    "VOLUME_COLUMNS",
    "WEEK_COLUMNS",
    "apply_user_price_curve",
    "bl01_lite_baseline",
    "book_decision_slots",
    "control_group_mask",
    "day_ahead_decision_utc_ns",
    "expected_need_kwh",
    "expected_sessions",
    "expected_smart_kwh",
    "flexed_history_baseline",
    "in_day_adjustment",
    "positions_kwh",
    "run_trading",
    "settle",
    "shock_kind",
    "slot_cash_components",
    "typical_day_ahead_shape",
    "typical_session_windows",
    "unmanaged_schedule_kwh",
    "validate_user_price_curve",
    "week_frame",
]
