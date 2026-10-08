"""The fleet replay of one simulated week: what was known, done and earned at each instant.

What this module owns (replay contract v1 §1, decision 0004 item 66): the
``ReplayWeek`` on ``ForecastResult.replay_week`` for the sampled worlds.  It
re-indexes what the run already produced by *what was known at each hourly
decision instant* ``tau_k`` (``k = 0 ... 168``, the study start plus k whole
hours, plus a closing row) so the Replay page can play the week back:

- the day-ahead prices visible at ``tau_k`` and the latest intraday price
  known at ``tau_k`` for every study slot (§1.2, the leakage-critical part);
- the aggregator's ``full``-strategy position in force at ``tau_k`` (§1.3);
- the drivers' running energy saving and the aggregator's running trading
  cash at every slot boundary (§1.4);
- the shocks and events with the instant each became known (§1.5).

It draws no random number, runs no physics and settles nothing: every value
comes from the run's own arrays through the run's own rules (the day-ahead
visibility rule of ``action``, the ledger's slot components of ``market``,
the customer leg of ``action``).  Nothing after ``tau_k`` enters row ``k``.
Every price, volume and pound is synthetic or illustrative; the trading
figure is "illustrative simulated trading P&L" (decision 0005), never Axle
cash.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from axle_studio.model import events as model_events
from axle_studio.model.action import (
    latest_known_prices,
    slot_cost_effect_gbp,
    visible_day_ahead_prices,
)
from axle_studio.model.market import TradingRun
from axle_studio.model.sampling import MarketPrices, day_ahead_publication_utc_ns

_LONDON = "Europe/London"
END_LABEL = "End of week"


@dataclass(frozen=True)
class ReplayWeek:
    """Replay contract v1 §1: the fleet's week as known at each decision instant.

    ``K`` = study slots / 2 + 1 decision rows (169), ``S`` study slots (336)
    and ``W_r`` sampled worlds.  Arrays are float64 unless stated:

    - ``world_ids``: the sampled worlds, the result's ``sampled_world_ids``
      in the same order; row ``wi`` of every array is ``world_ids[wi]``;
    - ``decisions``: one row per instant (§1.1);
    - ``published`` (K, S) bool: slot t's day-ahead price is published by tau_k;
    - ``day_ahead_visible_gbp_per_mwh`` (W_r, K, S): the day-ahead price
      where published, else the expected price shape (``action``'s B4 rule);
    - ``intraday_known_gbp_per_mwh`` (W_r, K, S): the latest intraday price
      known at tau_k (the close once past gate closure), NaN where the
      day-ahead price is not yet published;
    - ``position_kwh`` (W_r, K, S): the ``full`` strategy's turn-down
      position in force at tau_k, kWh per half-hour, NaN before the night's
      day-ahead decision; None without trading;
    - ``energy_saving_to_date_gbp`` (W_r, S + 1): the drivers' saving over
      slots that have ended, at slot boundaries 0..S (frame k reads 2k);
    - ``trading_cash_to_date_gbp`` (W_r, S + 1): the aggregator's cash over
      delivery slots that have ended; None without trading;
    - ``shocks`` and ``events``: §1.5 frames with ``known_from_utc``.
    """

    world_ids: tuple[int, ...]
    decisions: pd.DataFrame
    published: np.ndarray
    day_ahead_visible_gbp_per_mwh: np.ndarray
    intraday_known_gbp_per_mwh: np.ndarray
    position_kwh: np.ndarray | None
    energy_saving_to_date_gbp: np.ndarray
    trading_cash_to_date_gbp: np.ndarray | None
    shocks: pd.DataFrame
    events: pd.DataFrame


def build_replay_week(
    *,
    study_slots: pd.DataFrame,
    world_ids: Sequence[int],
    market_prices: MarketPrices,
    warmup_slot_count: int,
    fleet_world_intervals: pd.DataFrame,
    forecast_prices: pd.DataFrame,
    public_charge_gbp_per_kwh: float,
    gate_closure_minutes: float,
    surprise_reveal_hours: float,
    trading: TradingRun | None,
    market_shocks: pd.DataFrame,
    events: pd.DataFrame,
) -> ReplayWeek:
    """Build the ``ReplayWeek`` of an action run (replay contract v1 §1).

    Inputs are the run's own outputs: ``study_slots``; the sampled
    ``world_ids``; the generator's ``market_prices`` over the run slots
    (``warmup_slot_count`` warm-up slots first, then the study); the packaged
    ``fleet_world_intervals`` and ``forecast_prices`` (study slots);
    ``public_charge_gbp_per_kwh`` (GBP/kWh, illustrative);
    ``gate_closure_minutes`` (min before delivery) and
    ``surprise_reveal_hours`` (h, a stochastic surprise's reveal lead), the
    run's assumption values; the trading run (None: no trading frames);
    the result's ``market_shocks`` and validated ``events`` table.  Draws
    nothing and changes nothing it is given.
    """

    world_ids = tuple(int(w) for w in world_ids)
    # World ids are the run's world positions 0..W-1, so they index array rows directly.
    rows = np.asarray(world_ids, dtype=np.int64)
    decisions = decision_frame(study_slots)
    decision_ns = decisions["decision_utc"].astype("int64").to_numpy()
    run_count = warmup_slot_count + len(study_slots)
    world_count = market_prices.intraday_path_gbp_per_mwh.shape[0]
    run_starts = pd.DatetimeIndex(market_prices.day_ahead["interval_start_utc"].iloc[:run_count])
    day_ahead_run = (
        market_prices.day_ahead["wholesale_forecast_gbp_per_mwh"]
        .to_numpy(dtype=float)
        .reshape(world_count, run_count)[rows]
    )
    published, visible, known = forward_curves(
        day_ahead_run,
        run_starts,
        market_prices.intraday_path_gbp_per_mwh[rows],
        warmup_slot_count,
        decision_ns,
        gate_closure_minutes,
    )
    positions = cash = None
    if trading is not None:
        positions = positions_in_force(
            trading.position_updates, world_ids, decision_ns, len(study_slots)
        )
        cash = trading_cash_to_date(trading, rows, study_slots, events)
    return ReplayWeek(
        world_ids=world_ids,
        decisions=decisions,
        published=published,
        day_ahead_visible_gbp_per_mwh=visible,
        intraday_known_gbp_per_mwh=known,
        position_kwh=positions,
        energy_saving_to_date_gbp=energy_saving_to_date(
            fleet_world_intervals, forecast_prices, rows, public_charge_gbp_per_kwh
        ),
        trading_cash_to_date_gbp=cash,
        shocks=shock_frame(market_shocks, world_ids, study_slots, events, surprise_reveal_hours),
        events=event_frame(events, study_slots),
    )


def decision_frame(study_slots: pd.DataFrame) -> pd.DataFrame:
    """§1.1 ``decisions``: the study start plus k whole hours, k = 0 ... S / 2.

    Whole UTC hours are whole London hours (the offset is a whole number of
    hours), so these are exactly the trader's and dispatcher's hourly
    instants, plus one closing row at the study end (slot S, nothing ahead).
    """

    slot_count = len(study_slots)
    k = np.arange(slot_count // 2 + 1, dtype=np.int64)
    start = pd.Timestamp(study_slots["interval_start_utc"].iat[0])
    decision = (start + pd.to_timedelta(k, unit="h")).as_unit("ns")
    london = decision.tz_convert(_LONDON)
    slot = 2 * k
    night = study_slots["night_index"].to_numpy(dtype=np.int64)[np.minimum(slot, slot_count - 1)]
    label = [value.strftime("%a %H:%M") for value in london]
    label[-1] = END_LABEL
    return pd.DataFrame(
        {
            "decision_index": k,
            "decision_utc": decision,
            "decision_london": london,
            "slot_index": slot,
            "night_index": night,
            "label": pd.Series(label, dtype=object),
        }
    )


def forward_curves(
    day_ahead_run_gbp_per_mwh: np.ndarray,
    run_interval_start_utc: pd.DatetimeIndex,
    intraday_path_gbp_per_mwh: np.ndarray,
    warmup_slot_count: int,
    decision_ns: np.ndarray,
    gate_closure_minutes: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """§1.2 the prices known at each decision, for the study slots.

    ``day_ahead_run_gbp_per_mwh`` (world, run slot) and
    ``intraday_path_gbp_per_mwh`` (world, run slot, h) cover the run slots
    starting at ``run_interval_start_utc`` (warm-up first); ``decision_ns``
    (K,) are the instants as int64 UTC ns.  Returns ``published`` (K, S)
    bool, ``day_ahead_visible`` and ``intraday_known`` (world, K, S) GBP/MWh.

    Both rules are the run's own, called rather than restated: the smart
    charger's and trader's B4 visibility rule
    (``action.visible_day_ahead_prices``, which needs the warm-up prices for
    the expected shape) and the dispatcher's and trader's latest known price
    (``action.latest_known_prices``, the ceil rule for the latest intraday
    update).  The intraday value is kept only where the day-ahead price is
    published: the path starts from the published price and has no meaning
    before it.
    """

    study_starts = run_interval_start_utc[warmup_slot_count:]
    publication = day_ahead_publication_utc_ns(study_starts)
    published = publication[np.newaxis, :] <= decision_ns[:, np.newaxis]
    run_publication = day_ahead_publication_utc_ns(run_interval_start_utc)
    world_count, slot_count = day_ahead_run_gbp_per_mwh.shape[0], len(study_starts)
    visible = np.empty((world_count, len(decision_ns), slot_count))
    by_publication: dict[int, np.ndarray] = {}
    for k, decision in enumerate(decision_ns):
        # What is visible changes only at a 13:00 publication, so each set
        # of published prices is ranked once (as ``market.run_trading`` does).
        key = int((run_publication <= decision).sum())
        if key not in by_publication:
            by_publication[key] = visible_day_ahead_prices(
                day_ahead_run_gbp_per_mwh, run_interval_start_utc, int(decision)
            )[:, warmup_slot_count:]
        visible[:, k] = by_publication[key]
    gate_ns = study_starts.as_unit("ns").asi8 - round(float(gate_closure_minutes) * 60e9)
    study_path = intraday_path_gbp_per_mwh[:, warmup_slot_count:]
    known = np.empty_like(visible)
    for k, decision in enumerate(decision_ns):
        known[:, k] = latest_known_prices(
            study_path, visible[:, k], publication, gate_ns, int(decision)
        )
    known = np.where(published[np.newaxis], known, np.nan)
    return published, visible, known


def positions_in_force(
    position_updates: pd.DataFrame,
    world_ids: Sequence[int],
    decision_ns: np.ndarray,
    slot_count: int,
) -> np.ndarray:
    """§1.3 the ``full`` position in force at each decision, (world, K, S) kWh per half-hour.

    The ``position_kwh`` of the last ``position_updates`` row with
    ``decision_utc <= tau_k`` for that world and slot, NaN when there is
    none yet.  "At or before" is the trader's own rule: a decision at tau
    uses information at tau and is in force from tau.  Each update lands on
    the first instant at or after its decision (night 0's day-ahead
    decision, in the warm-up, lands on k = 0) and is carried forward to the
    next update of the same slot.
    """

    world_row = {int(w): i for i, w in enumerate(world_ids)}
    updates = position_updates.loc[position_updates["world_id"].isin(world_row)]
    decided = pd.DatetimeIndex(updates["decision_utc"]).as_unit("ns").asi8
    first_k = np.searchsorted(decision_ns, decided, side="left")
    landed = pd.DataFrame(
        {
            "w": updates["world_id"].map(world_row).to_numpy(dtype=np.int64),
            "k": first_k,
            "t": updates["slot_index"].to_numpy(dtype=np.int64),
            "decided": decided,
            "position": updates["position_kwh"].to_numpy(dtype=float),
        }
    )
    # Only the latest update landing on a (world, k, slot) counts.
    landed = landed.loc[landed["k"] < len(decision_ns)].sort_values("decided", kind="stable")
    landed = landed.drop_duplicates(["w", "k", "t"], keep="last")
    cube = np.full((len(world_row), len(decision_ns), slot_count), np.nan)
    cube[landed["w"].to_numpy(), landed["k"].to_numpy(), landed["t"].to_numpy()] = landed[
        "position"
    ].to_numpy()
    # Forward fill along k: the index of the latest row with a value, -1 if none.
    k = np.arange(len(decision_ns))[np.newaxis, :, np.newaxis]
    latest = np.maximum.accumulate(np.where(np.isnan(cube), -1, k), axis=1)
    filled = np.take_along_axis(cube, np.maximum(latest, 0), axis=1)
    return np.where(latest >= 0, filled, np.nan)


def energy_saving_to_date(
    fleet_world_intervals: pd.DataFrame,
    forecast_prices: pd.DataFrame,
    world_rows: np.ndarray,
    public_charge_gbp_per_kwh: float,
) -> np.ndarray:
    """§1.4 the drivers' energy saving over ended slots, (world, S + 1) GBP, illustrative.

    ``action.slot_cost_effect_gbp`` (selected minus normal home import at
    the day-ahead price, plus public import at the public rate) per slot,
    negated so positive means the smart path cost the drivers less, and
    summed over slots ``t < b`` at boundary ``b``.  The customer leg is
    valued at the day-ahead price, as the cost effect values it (decision
    0004 item 53), never at the realised price.  The end-of-week shortfall
    values (items 4 and 13) are not flows and are not accumulated.
    """

    def matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
        ordered = frame.sort_values(["world_id", "slot_index"], kind="stable")
        values = ordered[column].to_numpy(dtype=float)
        return values.reshape(-1, ordered["slot_index"].nunique())[world_rows]

    paths = {
        path: fleet_world_intervals.loc[fleet_world_intervals["path_id"].eq(path)]
        for path in ("normal", "selected")
    }
    slot = slot_cost_effect_gbp(
        matrix(paths["normal"], "home_import_kwh"),
        matrix(paths["selected"], "home_import_kwh"),
        matrix(paths["normal"], "public_import_kwh"),
        matrix(paths["selected"], "public_import_kwh"),
        matrix(forecast_prices, "wholesale_forecast_gbp_per_mwh"),
        public_charge_gbp_per_kwh,
    )
    return _to_date(-slot)


def trading_cash_to_date(
    trading: TradingRun,
    world_rows: np.ndarray,
    study_slots: pd.DataFrame,
    events: pd.DataFrame,
) -> np.ndarray:
    """§1.4 the aggregator's ``full``-strategy cash over ended delivery slots, (world, S + 1) GBP.

    Booked by delivery half-hour, as the ledger books by delivery night, so
    at every night end it equals the cumulative ledger ``net_gbp`` and at the
    last boundary the week's net:

    - each slot's six slot-resolved buckets (``TradingRun.slot_cash_gbp``,
      from ``market.slot_cash_components``, the ledger's own formula) at
      the boundary after the slot;
    - each night's customer revenue share and unmet-charge penalty (night
      buckets, ``trading_ledger_world``) at the night's end;
    - each grid request's payment at the boundary after its window's last
      slot, capped at the end of the night the window starts in (review
      B1).  The payment is computed from metered import over the whole
      window, so booking it at the window start (as the ledger's slot
      bookkeeping does) would put future half-hours into "so far".  The cap
      keeps the night-end identity with the ledger, which books the payment
      in that night; a window crossing 12:00 is therefore booked at the cap
      before its last slot ends (a stated limitation, §7).

    Illustrative simulated trading P&L, never Axle cash.
    """

    slot_count = len(study_slots)
    flow = np.zeros((len(world_rows), slot_count + 1))
    flow[:, 1:] = trading.slot_cash_gbp[world_rows]
    night = study_slots["night_index"].to_numpy()
    night_end = np.append(np.flatnonzero(np.diff(night)) + 1, slot_count)
    ledger = trading.trading_ledger_world
    full = ledger.loc[ledger["strategy"].eq("full")].sort_values(["world_id", "night_index"])
    night_terms = (full["customer_revenue_share_gbp"] + full["unmet_charge_penalty_gbp"]).to_numpy()
    flow[:, night_end] += night_terms.reshape(-1, len(night_end))[world_rows]
    if trading.event_delivery is not None:
        windows = model_events.event_slots(events, study_slots).set_index("event_id")
        delivery = trading.event_delivery
        start = windows["start_slot"].reindex(delivery["event_id"]).to_numpy(dtype=np.int64)
        end = windows["end_slot"].reindex(delivery["event_id"]).to_numpy(dtype=np.int64)
        booked = np.minimum(end, night_end[night[start]])
        row_of = {int(w): i for i, w in enumerate(world_rows)}
        worlds = delivery["world_id"].map(row_of)
        sampled = worlds.notna().to_numpy()
        np.add.at(
            flow,
            (worlds[sampled].to_numpy(dtype=np.int64), booked[sampled]),
            delivery["payment_gbp"].to_numpy(dtype=float)[sampled],
        )
    return np.cumsum(flow, axis=1)


def shock_frame(
    market_shocks: pd.DataFrame,
    world_ids: Sequence[int],
    study_slots: pd.DataFrame,
    events: pd.DataFrame,
    surprise_reveal_hours: float,
) -> pd.DataFrame:
    """§1.5 ``shocks``: the sampled worlds' study shocks plus ``known_from_utc``.

    A known shock entered the day-ahead price of its first slot, so it is
    known at that slot's publication; a stochastic surprise at ``start -
    surprise_reveal_hours`` (the generator's reveal rule, the instant
    intraday trading learns of it); a scripted surprise at its event's
    ``notice_utc`` (lead ruling: the events table is the authority on when a
    scripted event becomes known), joined on ``shock_id == event_id``; a
    scripted shock with no matching event raises ``ValueError``.  A
    non-integer ``surprise_reveal_hours`` can show a stochastic surprise up
    to an hour before the hourly intraday path moves (the defaults are
    whole hours).
    """

    kept = market_shocks.loc[
        market_shocks["world_id"].isin([int(w) for w in world_ids]) & market_shocks["in_study"]
    ].reset_index(drop=True)
    start = pd.DatetimeIndex(kept["start_utc"]).as_unit("ns")
    published = day_ahead_publication_utc_ns(start)
    revealed = (start - pd.Timedelta(hours=float(surprise_reveal_hours))).asi8
    notice = model_events.event_slots(events, study_slots).set_index("event_id")["notice_utc"]
    scripted = kept["source"].eq("scripted").to_numpy()
    noticed = kept["shock_id"].map(notice)
    if noticed[scripted].isna().any():
        raise ValueError("a scripted shock has no enabled event to take its notice from")
    notice_ns = noticed.fillna(0).to_numpy(dtype=np.int64)
    known_ns = np.where(
        kept["known_day_ahead"].to_numpy(dtype=bool),
        published,
        np.where(scripted, notice_ns, revealed),
    )
    kept["known_from_utc"] = pd.to_datetime(known_ns, utc=True).as_unit("ns")
    return kept


def event_frame(events: pd.DataFrame, study_slots: pd.DataFrame) -> pd.DataFrame:
    """§1.5 ``events``: each enabled request or control outage with its window and notice.

    Price shocks are left out (they are in ``shocks``).  ``end_utc`` is the
    end of the window's last slot (``end_slot`` is exclusive) and
    ``known_from_utc`` the event's ``notice_utc``.
    """

    windows = model_events.event_slots(events, study_slots)
    windows = windows.loc[~windows["event_type"].isin(model_events.PRICE_SHOCK_TYPES)]
    windows = windows.reset_index(drop=True)
    starts = pd.DatetimeIndex(study_slots["interval_start_utc"])
    ends = pd.DatetimeIndex(study_slots["interval_end_utc"])
    return pd.DataFrame(
        {
            "event_id": windows["event_id"].astype(object),
            "event_type": windows["event_type"].astype(object),
            "scope": windows["scope"].astype(object),
            "size": windows["size"].to_numpy(dtype=np.float64),
            "payment_gbp_per_mwh": windows["payment_gbp_per_mwh"].to_numpy(dtype=np.float64),
            "start_utc": starts[windows["start_slot"].to_numpy(dtype=np.int64)].as_unit("ns"),
            "end_utc": ends[windows["end_slot"].to_numpy(dtype=np.int64) - 1].as_unit("ns"),
            "known_from_utc": pd.to_datetime(
                windows["notice_utc"].to_numpy(dtype=np.int64), utc=True
            ).as_unit("ns"),
            "evidence_kind": pd.Series(["illustrative_synthetic"] * len(windows), dtype=object),
        }
    )


def _to_date(values: np.ndarray) -> np.ndarray:
    """(world, slot) -> (world, slot + 1): the value at boundary b sums slots t < b."""

    out = np.zeros((values.shape[0], values.shape[1] + 1))
    np.cumsum(values, axis=1, out=out[:, 1:])
    return out


__all__ = [
    "END_LABEL",
    "ReplayWeek",
    "build_replay_week",
    "decision_frame",
    "energy_saving_to_date",
    "event_frame",
    "forward_curves",
    "positions_in_force",
    "shock_frame",
    "trading_cash_to_date",
]
