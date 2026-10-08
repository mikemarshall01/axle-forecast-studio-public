"""Replay contract v1 specs, the SYNTHETIC FIXTURE replay and its validators.

What this owns (``docs/contracts/replay-v1.md`` §4, lane R1a): the
machine-readable form of ``ReplayWeek`` (§1) and ``OneEvTimeline`` (§2), the
reference computations the validators recompute them with, the fixture's own
``replay_week`` and one-EV timeline, and ``validate_replay_week_v2`` and
``validate_one_ev_timeline_v2``.  ``result_contract.validate_result_v2`` calls
``validate_replay_week_v2`` on every action result that carries a replay.

Like the rest of the fixture it never imports the model package, so it
cannot drift with model internals: the validators are duck-typed and check
the fixture and real ``ForecastResult`` objects alike.  Every price, volume
and pound here is SYNTHETIC or illustrative.

Two things the fixture does its own way (both SYNTHETIC, stated in §4):
the expected-shape price for an unpublished slot is the mean published price
of the same London half-hour inside the study (the model's rule needs the
warm-up prices), and the intraday path is a toy that moves linearly from the
day-ahead price 35 hours before gate closure to the close at gate closure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from .result_contract import (
    LONDON,
    MARKET_SHOCK_COLUMNS,
    PATH_ORDER,
    SLOT_COUNT,
    _check_dtypes,
    _term,
    metric_matrix,
)

DECISION_COUNT = SLOT_COUNT // 2 + 1
"""``K``: one decision row per whole hour of the study plus the closing row (169)."""

BOUNDARY_COUNT = SLOT_COUNT + 1
"""Slot boundaries 0..336 at which the running money is read."""

INTRADAY_STEPS = 36
"""``H``: hourly intraday updates h = 0..35 before gate closure (dispatch §3)."""

BOOK_WIDTH = 50
"""Mirrors ``market.BOOK_WIDTH`` (the fixture never imports the model): slots per plan snapshot."""

PLAN_STATUS_CODES = (0, 1, 2, 3, 4)
"""``plan_status`` codes: 0 plan followed, 1-4 the non-response reasons (trading §10.1e)."""

PRICE_SHOCK_TYPES = ("price_shock_known", "price_shock_surprise")
"""Event types already carried by ``shocks`` (mirrors ``events.PRICE_SHOCK_TYPES``)."""

END_LABEL = "End of week"

REPLAY_DECISION_COLUMNS = {
    "decision_index": "int",
    "decision_utc": "utc",
    "decision_london": "london",
    "slot_index": "int",
    "night_index": "int",
    "label": "str",
}
REPLAY_SHOCK_COLUMNS = {**MARKET_SHOCK_COLUMNS, "known_from_utc": "utc"}
REPLAY_EVENT_COLUMNS = {
    "event_id": "str",
    "event_type": "str",
    "scope": "str",
    "size": "float",
    "payment_gbp_per_mwh": "float",
    "start_utc": "utc",
    "end_utc": "utc",
    "known_from_utc": "utc",
    "evidence_kind": "str",
}
TIMELINE_CUMULATIVE_COLUMNS = {
    "path_id": "str",
    "slot_boundary": "int",
    "home_import_to_date_kwh": "float",
    "public_import_to_date_kwh": "float",
    "cost_to_date_gbp": "float",
}

_HOUR_NS = 3600 * 1_000_000_000
_GATE_CLOSURE = pd.Timedelta(minutes=60)  # the fixture's gate, as its position updates use
_DAY_AHEAD_HOUR = 13  # day-ahead prices for London date D are published 13:00 London on D - 1


@dataclass(frozen=True)
class FixtureReplayWeek:
    """SYNTHETIC FIXTURE mirror of ``model.replay.ReplayWeek`` (replay contract §1)."""

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


@dataclass(frozen=True)
class FixtureOneEvTimeline:
    """SYNTHETIC FIXTURE mirror of ``model.individual.OneEvTimeline`` (replay contract §2)."""

    unit_id: str
    world_id: int
    dispatch_locked: bool | None
    decisions: pd.DataFrame
    plan_at_decision_kwh: np.ndarray
    plan_status: np.ndarray
    cumulative: pd.DataFrame
    saving_to_date_gbp: np.ndarray


# --------------------------------------------------------------------------
# Reference computations (used by the fixture builder and the validators)
# --------------------------------------------------------------------------


def decision_frame(horizon_start_utc: pd.Timestamp, study_slots: pd.DataFrame) -> pd.DataFrame:
    """§1.1 ``decisions``: one row per whole hour of the study and a closing row."""

    k = np.arange(DECISION_COUNT, dtype=np.int64)
    decision = (horizon_start_utc + pd.to_timedelta(k, unit="h")).as_unit("ns")
    london = decision.tz_convert(LONDON)
    slot = np.minimum(2 * k, SLOT_COUNT)
    nights = study_slots["night_index"].to_numpy(dtype=np.int64)
    night = nights[np.minimum(slot, SLOT_COUNT - 1)]
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


def publication_utc(interval_start_utc) -> pd.DatetimeIndex:
    """Day-ahead publication instant of each slot: 13:00 London on the day before delivery."""

    london = pd.DatetimeIndex(interval_start_utc).tz_convert(LONDON)
    delivery = london.tz_localize(None).normalize()
    published = (delivery - pd.Timedelta(days=1) + pd.Timedelta(hours=_DAY_AHEAD_HOUR)).tz_localize(
        LONDON
    )
    return published.tz_convert("UTC").as_unit("ns")


def world_slot_matrix(frame: pd.DataFrame, column: str, world_ids) -> np.ndarray:
    """(world in ``world_ids`` order, study slot) values of one per-world, per-slot column."""

    rows = frame.loc[frame["world_id"].isin(world_ids)].sort_values(
        ["world_id", "slot_index"], kind="stable"
    )
    values = rows[column].to_numpy().reshape(-1, SLOT_COUNT)
    order = {w: i for i, w in enumerate(sorted(set(int(w) for w in world_ids)))}
    return values[[order[int(w)] for w in world_ids]]


def positions_in_force(updates: pd.DataFrame, world_ids, decision_utc) -> np.ndarray:
    """§1.3 by ``merge_asof``: the last ``position_kwh`` with ``decision_utc <= tau_k``, else NaN.

    Returns (world, decision, slot) kWh; the rule "at or before" is the
    trader's own (a decision at tau is in force from tau).
    """

    decisions = pd.DatetimeIndex(decision_utc).as_unit("ns")
    grid = pd.DataFrame(
        {
            "world_id": np.repeat(
                np.asarray(world_ids, dtype=np.int64), len(decisions) * SLOT_COUNT
            ),
            "k": np.tile(np.repeat(np.arange(len(decisions)), SLOT_COUNT), len(world_ids)),
            "slot_index": np.tile(
                np.arange(SLOT_COUNT, dtype=np.int64), len(world_ids) * len(decisions)
            ),
            "decision_utc": np.tile(np.repeat(decisions, SLOT_COUNT), len(world_ids)),
        }
    )
    right = updates.loc[
        updates["world_id"].isin(world_ids),
        ["world_id", "slot_index", "decision_utc", "position_kwh"],
    ].sort_values("decision_utc", kind="stable")
    right["decision_utc"] = right["decision_utc"].dt.as_unit("ns")
    merged = pd.merge_asof(
        grid.sort_values("decision_utc", kind="stable"),
        right,
        on="decision_utc",
        by=["world_id", "slot_index"],
        direction="backward",
        allow_exact_matches=True,
    ).sort_values(["world_id", "k", "slot_index"], kind="stable")
    order = {w: i for i, w in enumerate(sorted(set(int(w) for w in world_ids)))}
    cube = (
        merged["position_kwh"]
        .to_numpy(dtype=float)
        .reshape(len(world_ids), len(decisions), SLOT_COUNT)
    )
    return cube[[order[int(w)] for w in world_ids]]


def energy_saving_to_date(result, world_ids, public_charge_gbp_per_kwh: float) -> np.ndarray:
    """§1.4 drivers' saving at slot boundaries, (world, 337) GBP, day-ahead valued.

    ``saving_t = (U_t - M_t) P_DA / 1000 - (Pub_sel - Pub_norm) x rate``,
    accumulated over slots that have ended.  Illustrative.
    """

    world = result.fleet_world_intervals
    rows = [int(w) for w in world_ids]
    home = {p: metric_matrix(world, p, "home_import_kwh")[rows] for p in PATH_ORDER}
    public = {p: metric_matrix(world, p, "public_import_kwh")[rows] for p in PATH_ORDER}
    price = world_slot_matrix(result.forecast_prices, "wholesale_forecast_gbp_per_mwh", rows)
    slot = (home["normal"] - home["selected"]) * price / 1000.0 - (
        public["selected"] - public["normal"]
    ) * float(public_charge_gbp_per_kwh)
    return np.concatenate([np.zeros((len(rows), 1)), np.cumsum(slot, axis=1)], axis=1)


def slot_trading_cash(result, world_ids, *, half_spread: float, supplier_compensation: float):
    """§1.4 ``c_t`` (world, slot) GBP for strategy ``full`` from the result's own frames.

    ``c_t = q P_DA + sum Delta P_ID + (V - f) SIP - s sum|Delta| - c_sup V``
    (all /1000): the six slot-resolved ledger buckets, the grid-event
    payment and the two night buckets excluded.  Illustrative simulated
    trading P&L, never Axle cash.
    """

    deviation = result.deviation_world_slot
    rows = [int(w) for w in world_ids]

    def m(column: str) -> np.ndarray:
        return world_slot_matrix(deviation, column, rows).astype(float)

    q, f, settled = m("position_da_only_kwh"), m("position_full_kwh"), m("settled_kwh")
    intraday = result.position_updates.loc[result.position_updates["stage"].eq("intraday")]
    cash = np.zeros((len(rows), SLOT_COUNT))
    volume = np.zeros((len(rows), SLOT_COUNT))
    for wi, w in enumerate(rows):
        trades = intraday.loc[intraday["world_id"].eq(w)]
        slot = trades["slot_index"].to_numpy()
        np.add.at(cash[wi], slot, (trades["trade_kwh"] * trades["price_gbp_per_mwh"]).to_numpy())
        np.add.at(volume[wi], slot, trades["trade_kwh"].abs().to_numpy())
    return (
        q * m("day_ahead_gbp_per_mwh")
        + cash
        + (settled - f) * m("imbalance_gbp_per_mwh")
        - half_spread * volume
        - supplier_compensation * settled
    ) / 1000.0


def night_ends(study_slots: pd.DataFrame) -> np.ndarray:
    """The slot boundary at which each session night ends (exclusive last slot + 1)."""

    night = study_slots["night_index"].to_numpy()
    return np.append(np.flatnonzero(np.diff(night)) + 1, len(night))


def event_windows(result) -> pd.DataFrame:
    """Enabled events with ``start_utc``, ``end_utc`` and ``notice_utc`` from the events table.

    A window starting before 12:00 London belongs to the next calendar day
    of its session night; a ``day_ahead`` event is known at 13:00 London on
    the day before the night's date, a ``short`` one ``notice_minutes``
    before it starts (trading contract §2.1).
    """

    events = result.events
    enabled = events.loc[events["enabled"].astype(bool)].reset_index(drop=True)
    rows = []
    for row in enabled.itertuples(index=False):
        night = result.study_start_local_date + timedelta(days=int(row.night_index))
        hour, minute = (int(part) for part in row.start_local_time.split(":"))
        day = night if hour >= 12 else night + timedelta(days=1)
        start = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)
        start = start.tz_localize(LONDON).tz_convert("UTC")
        if row.notice == "day_ahead":
            notice = pd.Timestamp(night - timedelta(days=1)) + pd.Timedelta(hours=_DAY_AHEAD_HOUR)
            notice = notice.tz_localize(LONDON).tz_convert("UTC")
        else:
            notice = start - pd.Timedelta(minutes=int(row.notice_minutes))
        rows.append(
            {
                "event_id": row.event_id,
                "event_type": row.event_type,
                "scope": row.scope,
                "night_index": int(row.night_index),
                "size": float(row.size),
                "payment_gbp_per_mwh": float(row.payment_gbp_per_mwh),
                "start_utc": start,
                "end_utc": start + pd.Timedelta(minutes=int(row.duration_minutes)),
                "notice_utc": notice,
            }
        )
    columns = [
        "event_id",
        "event_type",
        "scope",
        "night_index",
        "size",
        "payment_gbp_per_mwh",
        "start_utc",
        "end_utc",
        "notice_utc",
    ]
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------
# The SYNTHETIC FIXTURE replay
# --------------------------------------------------------------------------


def make_replay_week(
    result,
    *,
    half_spread_gbp_per_mwh: float,
    supplier_compensation_gbp_per_mwh: float,
    public_charge_gbp_per_kwh: float,
) -> FixtureReplayWeek:
    """Build the fixture's ``replay_week`` from its own price and trading frames.

    The trading terms and public rate are the fixture's own (passed in, so
    this module does not read another fixture's constants).  Every array is
    SYNTHETIC.
    """

    world_ids = tuple(int(w) for w in result.sampled_world_ids)
    slots = result.study_slots
    decisions = decision_frame(result.horizon_start_utc, slots)
    decision_ns = decisions["decision_utc"].astype("int64").to_numpy()
    starts = pd.DatetimeIndex(slots["interval_start_utc"])
    publication = publication_utc(starts).asi8
    published = publication[np.newaxis, :] <= decision_ns[:, np.newaxis]

    day_ahead = world_slot_matrix(
        result.forecast_prices, "wholesale_forecast_gbp_per_mwh", world_ids
    ).astype(float)
    close = world_slot_matrix(
        result.evaluation_prices, "evaluation_context_price_gbp_per_mwh", world_ids
    ).astype(float)
    half_hour = slots["local_half_hour"].to_numpy()
    visible = np.empty((len(world_ids), DECISION_COUNT, SLOT_COUNT))
    shapes: dict[int, np.ndarray] = {}
    for k in range(DECISION_COUNT):
        row = published[k]
        key = int(row.sum())
        if key not in shapes:
            # SYNTHETIC expected shape: the mean published price of the same
            # London half-hour, or of every published slot when none.
            fallback = day_ahead[:, row].mean(axis=1)
            shape = np.empty_like(day_ahead)
            for h in np.unique(half_hour):
                cells = row & (half_hour == h)
                mean = day_ahead[:, cells].mean(axis=1) if cells.any() else fallback
                shape[:, half_hour == h] = mean[:, np.newaxis]
            shapes[key] = np.where(row, day_ahead, shape)
        visible[:, k] = shapes[key]

    # SYNTHETIC toy intraday path: linear from the day-ahead price at h = 35
    # to the close at h = 0, read at the ceil-rule step h_t(tau_k).
    gate = (starts - _GATE_CLOSURE).as_unit("ns").asi8
    steps = -((decision_ns[:, np.newaxis] - gate[np.newaxis, :]) // _HOUR_NS)
    h = np.clip(steps, 0, INTRADAY_STEPS - 1)
    weight = h / (INTRADAY_STEPS - 1)
    intraday = close[:, np.newaxis, :] + (day_ahead - close)[:, np.newaxis, :] * weight
    intraday = np.where(published[np.newaxis], intraday, np.nan)

    positions = cash = None
    if result.position_updates is not None:
        positions = positions_in_force(
            result.position_updates, world_ids, decisions["decision_utc"]
        )
    if result.trading_ledger_world is not None:
        c = slot_trading_cash(
            result,
            world_ids,
            half_spread=half_spread_gbp_per_mwh,
            supplier_compensation=supplier_compensation_gbp_per_mwh,
        )
        cash = np.concatenate([np.zeros((len(world_ids), 1)), np.cumsum(c, axis=1)], axis=1)
        ledger = result.trading_ledger_world.loc[result.trading_ledger_world["strategy"].eq("full")]
        ends = night_ends(slots)
        for wi, w in enumerate(world_ids):
            nights = ledger.loc[ledger["world_id"].eq(w)].sort_values("night_index")
            terms = (
                nights["customer_revenue_share_gbp"] + nights["unmet_charge_penalty_gbp"]
            ).to_numpy()
            for end, value in zip(ends, terms, strict=True):
                cash[wi, end:] += value
        # The fixture has no grid request, so no payment is booked (§1.4 B1).

    return FixtureReplayWeek(
        world_ids=world_ids,
        decisions=decisions,
        published=published,
        day_ahead_visible_gbp_per_mwh=visible,
        intraday_known_gbp_per_mwh=intraday,
        position_kwh=positions,
        energy_saving_to_date_gbp=energy_saving_to_date(
            result, world_ids, public_charge_gbp_per_kwh
        ),
        trading_cash_to_date_gbp=cash,
        shocks=_fixture_shocks(result, world_ids),
        events=replay_events_from(result),
    )


def _fixture_shocks(result, world_ids) -> pd.DataFrame:
    """§1.5 ``shocks`` for the fixture: its only shocks are known ones (price_shocks)."""

    shocks = result.market_shocks
    kept = shocks.loc[shocks["world_id"].isin(world_ids) & shocks["in_study"]].reset_index(
        drop=True
    )
    if not kept["known_day_ahead"].all():
        raise ValueError("the fixture replay handles known shocks only")
    kept["known_from_utc"] = publication_utc(kept["start_utc"])
    return kept


def replay_events_from(result) -> pd.DataFrame:
    """§1.5 ``events``: the enabled requests and outages of the events table, in table order."""

    windows = event_windows(result)
    windows = windows.loc[~windows["event_type"].isin(PRICE_SHOCK_TYPES)].reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "event_id": pd.Series(windows["event_id"], dtype=object),
            "event_type": pd.Series(windows["event_type"], dtype=object),
            "scope": pd.Series(windows["scope"], dtype=object),
            "size": windows["size"].astype(np.float64),
            "payment_gbp_per_mwh": windows["payment_gbp_per_mwh"].astype(np.float64),
            "start_utc": pd.to_datetime(windows["start_utc"], utc=True).dt.as_unit("ns"),
            "end_utc": pd.to_datetime(windows["end_utc"], utc=True).dt.as_unit("ns"),
            "known_from_utc": pd.to_datetime(windows["notice_utc"], utc=True).dt.as_unit("ns"),
            "evidence_kind": pd.Series(["illustrative_synthetic"] * len(windows), dtype=object),
        }
    )
    return frame.reset_index(drop=True)


def replay_one_ev_timeline(result, unit_id: str, world_id: int) -> FixtureOneEvTimeline:
    """SYNTHETIC FIXTURE ``replay_one_ev_timeline`` (§2) from the fixture's one-EV replay.

    The toy plan in force at ``tau_k`` is the selected path's own home
    import over the rest of the session that was plugged in before
    ``tau_k`` (so a plug-in exactly at ``tau_k`` shows from ``k + 1``), cut
    at ``BOOK_WIDTH`` slots.  ``plan_status`` is 0 except on the last night
    of every second EV (odd position in ``units``), marked 1 so views can
    exercise "Plan on record, not followed".  Both are SYNTHETIC.
    """

    from .result_fixture import replay_one_ev  # the fixture imports this module

    if int(world_id) not in set(result.sampled_world_ids):
        raise KeyError(f"world {world_id} is not a sampled world")
    replay = replay_one_ev(result, unit_id, int(world_id))
    intervals = replay.intervals
    selected = intervals.loc[intervals["path_id"].eq("selected")]
    connected = selected["connected"].to_numpy(dtype=bool)
    home = selected["home_import_kwh"].to_numpy(dtype=float)
    plan = np.zeros((DECISION_COUNT, SLOT_COUNT))
    for k in range(DECISION_COUNT - 1):
        first = 2 * k
        if first == 0 or not connected[first - 1]:
            continue
        end = first
        while end < SLOT_COUNT and end < first + BOOK_WIDTH and connected[end]:
            end += 1
        plan[k, first:end] = home[first:end]
    night = result.study_slots["night_index"].to_numpy()
    status = np.zeros(SLOT_COUNT, dtype=np.int64)
    if list(result.units["unit_id"]).index(unit_id) % 2 == 1:
        status[night == night.max()] = 1

    price = world_slot_matrix(
        result.forecast_prices, "wholesale_forecast_gbp_per_mwh", [int(world_id)]
    )[0].astype(float)
    rate = _term(result, "public_charge_gbp_per_kwh")
    frames = []
    for path in PATH_ORDER:
        rows = intervals.loc[intervals["path_id"].eq(path)]
        path_home = rows["home_import_kwh"].to_numpy(dtype=float)
        path_public = rows["public_import_kwh"].to_numpy(dtype=float)
        cost = path_home * price / 1000.0 + path_public * rate
        frames.append(
            pd.DataFrame(
                {
                    "path_id": pd.Series([path] * BOUNDARY_COUNT, dtype=object),
                    "slot_boundary": np.arange(BOUNDARY_COUNT, dtype=np.int64),
                    "home_import_to_date_kwh": _to_date(path_home),
                    "public_import_to_date_kwh": _to_date(path_public),
                    "cost_to_date_gbp": _to_date(cost),
                }
            )
        )
    cumulative = pd.concat(frames, ignore_index=True)
    cost_by_path = {
        path: cumulative.loc[cumulative["path_id"].eq(path), "cost_to_date_gbp"].to_numpy()
        for path in PATH_ORDER
    }
    return FixtureOneEvTimeline(
        unit_id=unit_id,
        world_id=int(world_id),
        dispatch_locked=None,
        decisions=result.replay_week.decisions,
        plan_at_decision_kwh=plan,
        plan_status=status,
        cumulative=cumulative,
        saving_to_date_gbp=cost_by_path["normal"] - cost_by_path["selected"],
    )


def _to_date(values: np.ndarray) -> np.ndarray:
    """Cumulative sum at slot boundaries 0..n: the value at b sums slots t < b."""

    return np.concatenate([[0.0], np.cumsum(values)])


# --------------------------------------------------------------------------
# Validators
# --------------------------------------------------------------------------


def _close(actual, expected, scale: float | None = None) -> bool:
    """Equal within ``1e-9 x max(1, |value|)`` (the contract's money tolerance)."""

    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    size = np.maximum(1.0, np.abs(expected)) if scale is None else max(1.0, scale)
    return bool((np.abs(actual - expected) <= 1e-9 * size).all())


def validate_replay_week_v2(result) -> None:
    """Assert ``result.replay_week`` satisfies replay contract v1 §1 and §4."""

    replay = result.replay_week
    world_ids = tuple(int(w) for w in result.sampled_world_ids)
    assert tuple(replay.world_ids) == world_ids, "replay_week.world_ids == sampled_world_ids"
    worlds = len(world_ids)
    slots = result.study_slots

    decisions = replay.decisions
    _check_dtypes(decisions, REPLAY_DECISION_COLUMNS, "replay_week.decisions")
    expected = decision_frame(result.horizon_start_utc, slots)
    assert len(decisions) == DECISION_COUNT, "replay_week.decisions has 169 rows"
    pd.testing.assert_frame_equal(decisions, expected, obj="replay_week.decisions")

    cube = (worlds, DECISION_COUNT, SLOT_COUNT)
    assert replay.published.shape == (DECISION_COUNT, SLOT_COUNT), "published is (K, 336)"
    assert replay.published.dtype == np.dtype(bool), "published is bool"
    for name in ("day_ahead_visible_gbp_per_mwh", "intraday_known_gbp_per_mwh"):
        values = getattr(replay, name)
        assert values.shape == cube and values.dtype == np.float64, f"{name} is (W_r, K, 336)"

    # §1.2: what was published and known at tau_k.
    prices = result.forecast_prices
    first = prices.loc[prices["world_id"].eq(world_ids[0])].sort_values("slot_index")
    available = pd.DatetimeIndex(first["forecast_available_at_utc"]).as_unit("ns").asi8
    decision_ns = decisions["decision_utc"].astype("int64").to_numpy()
    published = available[np.newaxis, :] <= decision_ns[:, np.newaxis]
    assert np.array_equal(replay.published, published), "published = available_at <= tau_k"
    day_ahead = world_slot_matrix(prices, "wholesale_forecast_gbp_per_mwh", world_ids)
    close = world_slot_matrix(
        result.evaluation_prices, "evaluation_context_price_gbp_per_mwh", world_ids
    )
    visible = replay.day_ahead_visible_gbp_per_mwh
    known = replay.intraday_known_gbp_per_mwh
    mask = np.broadcast_to(published, cube)
    assert np.isfinite(visible).all(), "day_ahead_visible is finite"
    assert np.array_equal(np.isnan(known), ~mask), "intraday_known is NaN exactly where unpublished"
    da_cube = np.broadcast_to(day_ahead[:, np.newaxis, :], cube)
    assert np.array_equal(visible[mask], da_cube[mask]), "published slots show the day-ahead price"
    before = np.arange(SLOT_COUNT)[np.newaxis, :] < 2 * np.arange(DECISION_COUNT)[:, np.newaxis]
    realised = np.broadcast_to(before, cube)
    close_cube = np.broadcast_to(close[:, np.newaxis, :], cube)
    assert np.array_equal(known[realised], close_cube[realised]), "ended slots show the close"
    assert np.array_equal(visible[:, -1], day_ahead), "end row: day-ahead as published"
    assert np.array_equal(known[:, -1], close), "end row: the close"
    # The expected shape of an unpublished slot moves only at a publication.
    counts = published.sum(axis=1)
    for count in np.unique(counts):
        rows = np.flatnonzero(counts == count)
        hidden = ~published[rows[0]]
        block = visible[:, rows][:, :, hidden]
        assert (block == block[:, :1]).all(), "expected shape changes only at a publication"

    # §1.3: positions in force.
    updates = result.position_updates
    if updates is None:
        assert replay.position_kwh is None, "position_kwh is None exactly without position_updates"
    else:
        positions = replay.position_kwh
        assert positions is not None and positions.shape == cube, "position_kwh is (W_r, K, 336)"
        reference = positions_in_force(updates, world_ids, decisions["decision_utc"])
        assert np.array_equal(positions, reference, equal_nan=True), (
            "position_kwh is the forward fill of position_updates at tau_k"
        )
        final = world_slot_matrix(result.deviation_world_slot, "position_full_kwh", world_ids)
        assert np.array_equal(positions[:, -1], final.astype(float)), "end row: final positions"

    # §1.4: running money at slot boundaries.
    saving = replay.energy_saving_to_date_gbp
    assert saving.shape == (worlds, BOUNDARY_COUNT), "energy_saving_to_date is (W_r, 337)"
    assert np.isfinite(saving).all() and (saving[:, 0] == 0.0).all(), "saving starts at 0"
    cost = result.cost_effect.set_index("world_id").loc[list(world_ids)]
    week = -(
        cost["illustrative_selected_minus_normal_energy_cost_gbp"]
        + cost["illustrative_selected_minus_normal_public_charge_cost_gbp"]
    ).to_numpy()
    assert _close(saving[:, -1], week), "saving at 336 = minus the cost effect's flow components"
    rate = _term(result, "public_charge_gbp_per_kwh")
    assert rate is not None, "the public charge rate is recorded"
    recomputed = energy_saving_to_date(result, world_ids, rate)
    assert _close(saving, recomputed, float(np.abs(recomputed).max())), (
        "saving recomputes from fleet_world_intervals and forecast_prices"
    )
    _validate_trading_cash(result, replay, world_ids)

    _validate_shocks(result, replay, world_ids)
    _validate_events(result, replay)


def _validate_trading_cash(result, replay, world_ids) -> None:
    """§1.4 trading cash: night-end identity with the ledger and B1 payment booking."""

    ledger = result.trading_ledger_world
    cash = replay.trading_cash_to_date_gbp
    if ledger is None:
        assert cash is None, "trading_cash_to_date is None exactly without the ledger"
        return
    assert cash is not None and cash.shape == (len(world_ids), BOUNDARY_COUNT), (
        "trading_cash_to_date is (W_r, 337)"
    )
    assert np.isfinite(cash).all() and (cash[:, 0] == 0.0).all(), "trading cash starts at 0"
    full = ledger.loc[ledger["strategy"].eq("full")]
    ends = night_ends(result.study_slots)
    scale = float(full["net_gbp"].abs().sum())
    night_terms = np.zeros((len(world_ids), BOUNDARY_COUNT))
    payments = np.zeros((len(world_ids), len(ends)))
    for wi, w in enumerate(world_ids):
        nights = full.loc[full["world_id"].eq(w)].sort_values("night_index")
        assert _close(cash[wi, ends], nights["net_gbp"].cumsum().to_numpy(), scale), (
            "trading cash at each night end = cumulative ledger net_gbp (full)"
        )
        week = result.trading_week_world
        net = week.loc[week["world_id"].eq(w) & week["strategy"].eq("full"), "net_gbp"]
        assert _close(cash[wi, -1], net.to_numpy(), scale), "trading cash at 336 = week net"
        night_terms[wi, ends] = (
            nights["customer_revenue_share_gbp"] + nights["unmet_charge_penalty_gbp"]
        ).to_numpy()
        payments[wi] = nights["grid_event_payment_gbp"].to_numpy()

    # Slot by slot: each increment is c_t, plus the night buckets at a night
    # end, plus request payments only at their booking boundary b_e (B1).
    spread = _term(result, "trading.half_spread_gbp_per_mwh")
    supplier = _term(result, "trading.supplier_compensation_gbp_per_mwh")
    if spread is None:
        traded = full["intraday_traded_mwh"].sum()
        spread = -full["trading_cost_gbp"].sum() / traded if traded > 0 else 0.0
    if supplier is None:
        settled = full["settled_mwh"].sum()
        supplier = -full["supplier_compensation_gbp"].sum() / settled if settled > 0 else 0.0
    c = slot_trading_cash(
        result, world_ids, half_spread=float(spread), supplier_compensation=float(supplier)
    )
    residual = np.diff(cash, axis=1) - c - night_terms[:, 1:]
    booking = np.zeros(BOUNDARY_COUNT, dtype=bool)
    events = replay.events
    starts = pd.DatetimeIndex(result.study_slots["interval_start_utc"])
    requests = events.loc[events["event_type"].isin(("turn_down", "turn_up"))]
    for row in requests.itertuples(index=False):
        start = int(starts.searchsorted(row.start_utc))
        end = int(starts.searchsorted(row.end_utc))
        night_end = ends[np.searchsorted(ends, start, side="right")]
        booking[min(end, night_end)] = True
    assert _close(residual[:, ~booking[1:]], 0.0, scale), (
        "trading cash moves by c_t and night buckets only, outside payment boundaries"
    )
    night_of_boundary = np.searchsorted(ends, np.arange(1, BOUNDARY_COUNT), side="left")
    for n in range(len(ends)):
        booked = residual[:, night_of_boundary == n].sum(axis=1)
        assert _close(booked, payments[:, n], scale), (
            "grid-event payments are booked in the night the ledger books them"
        )


def _validate_shocks(result, replay, world_ids) -> None:
    shocks = replay.shocks
    _check_dtypes(shocks, REPLAY_SHOCK_COLUMNS, "replay_week.shocks")
    source = result.market_shocks
    kept = source.loc[source["world_id"].isin(world_ids) & source["in_study"]]
    pd.testing.assert_frame_equal(
        shocks.loc[:, list(MARKET_SHOCK_COLUMNS)],
        kept.reset_index(drop=True),
        obj="replay_week.shocks",
    )
    assert (shocks["known_from_utc"] <= shocks["start_utc"]).all(), "shocks known by their start"
    known = shocks["known_day_ahead"].to_numpy(dtype=bool)
    publication = publication_utc(shocks["start_utc"])
    assert (shocks["known_from_utc"].to_numpy()[known] == publication.to_numpy()[known]).all(), (
        "a known shock is known at the publication of its first slot"
    )
    reveal = _term(result, "surprise_reveal_hours")
    stochastic = ~known & shocks["source"].eq("stochastic").to_numpy()
    if reveal is not None and stochastic.any():
        expected = shocks["start_utc"] - pd.Timedelta(hours=reveal)
        assert (shocks["known_from_utc"][stochastic] == expected[stochastic]).all(), (
            "a stochastic surprise is known surprise_reveal_hours before it starts"
        )
    scripted = ~known & shocks["source"].eq("scripted").to_numpy()
    if scripted.any():
        notice = event_windows(result).set_index("event_id")["notice_utc"]
        expected = pd.to_datetime(shocks["shock_id"].map(notice), utc=True)
        assert (shocks["known_from_utc"][scripted] == expected[scripted]).all(), (
            "a scripted surprise is known at its event's notice_utc"
        )


def _validate_events(result, replay) -> None:
    events = replay.events
    _check_dtypes(events, REPLAY_EVENT_COLUMNS, "replay_week.events")
    windows = event_windows(result)
    windows = windows.loc[~windows["event_type"].isin(PRICE_SHOCK_TYPES)].reset_index(drop=True)
    assert list(events["event_id"]) == list(windows["event_id"]), (
        "one replay event per enabled request or outage, in table order"
    )
    for column in ("event_type", "scope"):
        assert list(events[column]) == list(windows[column]), f"replay_week.events.{column}"
    for column in ("size", "payment_gbp_per_mwh"):
        assert np.array_equal(
            events[column].to_numpy(), windows[column].to_numpy(dtype=float), equal_nan=True
        ), f"replay_week.events.{column}"
    for column, source in (
        ("start_utc", "start_utc"),
        ("end_utc", "end_utc"),
        ("known_from_utc", "notice_utc"),
    ):
        expected = pd.to_datetime(windows[source], utc=True).to_numpy()
        assert (events[column].to_numpy() == expected).all(), f"replay_week.events.{column}"
    assert (events["known_from_utc"] <= events["start_utc"]).all(), "events known by their start"


def validate_one_ev_timeline_v2(timeline, result, replay=None) -> None:
    """Assert a one-EV timeline satisfies replay contract v1 §2 and §4.

    ``replay`` (optional) is the same EV and world's one-EV replay; with it
    the cumulative sums and costs are recomputed from its intervals.
    """

    assert timeline.unit_id in set(result.units["unit_id"]), "timeline unit_id"
    assert timeline.world_id in set(result.sampled_world_ids), "timeline world is sampled"
    assert timeline.dispatch_locked in (True, False, None), "dispatch_locked is bool or None"
    pd.testing.assert_frame_equal(timeline.decisions, result.replay_week.decisions)
    plan = timeline.plan_at_decision_kwh
    assert plan.shape == (DECISION_COUNT, SLOT_COUNT) and plan.dtype == np.float64, "plan shape"
    assert (plan >= 0.0).all(), "plan_at_decision_kwh >= 0"
    k = np.arange(DECISION_COUNT)[:, np.newaxis]
    t = np.arange(SLOT_COUNT)[np.newaxis, :]
    inside = (t >= 2 * k) & (t < 2 * k + BOOK_WIDTH) & (k < DECISION_COUNT - 1)
    assert (plan[~inside] == 0.0).all(), "plan is zero outside [2k, 2k + BOOK_WIDTH) and at the end"
    status = timeline.plan_status
    assert status.shape == (SLOT_COUNT,) and status.dtype == np.int64, "plan_status is (336,)"
    assert np.isin(status, PLAN_STATUS_CODES).all(), "plan_status codes 0-4"

    cumulative = timeline.cumulative
    _check_dtypes(cumulative, TIMELINE_CUMULATIVE_COLUMNS, "timeline.cumulative")
    # ``cumulative``'s path set is ``PATH_ORDER`` plus the optional timed path
    # (decision 0007, model step 2) exactly when this result's fleet frames
    # carry it (``individual._paths``), never assumed fixed at two.
    timeline_paths = (
        (*PATH_ORDER, "timed")
        if "timed" in set(result.fleet_world_intervals["path_id"])
        else PATH_ORDER
    )
    assert list(cumulative["path_id"]) == [p for p in timeline_paths for _ in range(BOUNDARY_COUNT)]
    assert list(cumulative["slot_boundary"]) == list(range(BOUNDARY_COUNT)) * len(timeline_paths)
    at_zero = cumulative.loc[cumulative["slot_boundary"].eq(0)]
    value_columns = ["home_import_to_date_kwh", "public_import_to_date_kwh", "cost_to_date_gbp"]
    assert (at_zero[value_columns] == 0.0).all().all(), "cumulative starts at 0"
    cost = {
        p: cumulative.loc[cumulative["path_id"].eq(p), "cost_to_date_gbp"].to_numpy()
        for p in PATH_ORDER
    }
    saving = timeline.saving_to_date_gbp
    assert saving.shape == (BOUNDARY_COUNT,), "saving_to_date is (337,)"
    assert np.array_equal(saving, cost["normal"] - cost["selected"]), "saving = normal - smart"
    if replay is None:
        return
    price = world_slot_matrix(
        result.forecast_prices, "wholesale_forecast_gbp_per_mwh", [timeline.world_id]
    )[0].astype(float)
    rate = _term(result, "public_charge_gbp_per_kwh")
    for path in timeline_paths:
        rows = replay.intervals.loc[replay.intervals["path_id"].eq(path)]
        got = cumulative.loc[cumulative["path_id"].eq(path)]
        home = rows["home_import_kwh"].to_numpy(dtype=float)
        public = rows["public_import_kwh"].to_numpy(dtype=float)
        for column, values in (
            ("home_import_to_date_kwh", home),
            ("public_import_to_date_kwh", public),
            ("cost_to_date_gbp", home * price / 1000.0 + public * rate),
        ):
            assert _close(got[column].to_numpy(), _to_date(values)), f"cumulative {column}"


__all__ = [
    "BOOK_WIDTH",
    "BOUNDARY_COUNT",
    "DECISION_COUNT",
    "END_LABEL",
    "INTRADAY_STEPS",
    "PLAN_STATUS_CODES",
    "REPLAY_DECISION_COLUMNS",
    "REPLAY_EVENT_COLUMNS",
    "REPLAY_SHOCK_COLUMNS",
    "TIMELINE_CUMULATIVE_COLUMNS",
    "FixtureOneEvTimeline",
    "FixtureReplayWeek",
    "decision_frame",
    "energy_saving_to_date",
    "event_windows",
    "make_replay_week",
    "night_ends",
    "positions_in_force",
    "publication_utc",
    "replay_events_from",
    "replay_one_ev_timeline",
    "slot_trading_cash",
    "validate_one_ev_timeline_v2",
    "validate_replay_week_v2",
    "world_slot_matrix",
]
