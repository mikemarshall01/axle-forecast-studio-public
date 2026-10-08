"""One-EV replay for the Drivers ▸ One EV view (contract v2 section 5) and the
Replay ▸ Customer lens (replay contract v1 §2).

What this owns: ``replay_one_ev`` and ``replay_one_ev_bands``, which show one
EV's week from a stored result, and ``replay_one_ev_timeline``, which adds
that EV's plan-in-force snapshots and running cost for one sampled world.

How it works (decision 0004 items 10 and 20): the vectorised kernel is run on
a one-EV slice of the inputs the result kept in ``replay_state``, with that
EV's smart-charging inputs and the run's public top-up rule.  So the replay
is the *same physics* as the fleet run, including the smart-charging path
(audit B3: the old scalar replay showed no-action physics for action runs)
and clock-change weeks (audit B2), because the kernel already handles fixed
UTC slots across a London offset change.  EVs are independent in this model
(no shared site cap), so a one-EV run reproduces that EV's part of the fleet
sum exactly; the tests check the sum over all EVs equals the fleet frame.

Replays are cached on the result's ``replay_state`` (audit O6), keyed by EV
and world, so revisiting an EV costs nothing.  ``replay_one_ev_timeline``
shares that cached kernel run for the intervals its ``cumulative`` table
sums, and pays two further one-EV kernel passes (replay contract v1 §2) for
the plan book (``market.book_decision_slots``, the same hourly decision grid
the fleet run's trading kernel uses) and the per-EV ``plan_status``
(trading §10.1e): the current physics kernel exposes the two through
different entry points (``simulate_fleet_intervals``'s ``trading_output``
for the book, ``simulate_unit_intervals`` for the per-EV plan outputs), so
one EV pays two cheap passes rather than the fleet kernel gaining a new
combined output for a single caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from axle_studio.model import market
from axle_studio.model.action import ILLUSTRATIVE_PUBLIC_CHARGE_GBP_PER_KWH, slot_cost_effect_gbp
from axle_studio.model.physics import simulate_fleet_intervals, simulate_unit_intervals
from axle_studio.model.summaries import (
    LOCATION_ORDER,
    LONDON,
    PATH_ORDER,
    KernelSlice,
    find_sessions,
    kernel_slice,
    plug_in_events_frame,
)

_CACHE_LIMIT = 256
"""Replays kept per result; one replay is ~700 rows, so this stays a few MB."""
_BELOW_TARGET_TOLERANCE_POINTS = 1e-6
_SLOT_COLUMNS = ["slot_index", "interval_start_utc", "interval_end_utc", "interval_start_london"]
_INTERVAL_COLUMNS = [
    "path_id",
    *_SLOT_COLUMNS,
    "location",
    "connected",
    "battery_soc_percent",
    "closing_battery_kwh",
    "home_import_kwh",
    "public_import_kwh",
    "unserved_travel_kwh",
    "early_departure_shortfall_kwh",
]
_AUDIT_COLUMNS = [
    "path_id",
    "local_date",
    "day_label",
    "drives_today",
    "departure_utc",
    "departure_london",
    "target_soc_percent",
    "departure_soc_percent",
    "trip_energy_need_kwh",
    "shortfall_kwh",
    "departed_below_target",
]


@dataclass(frozen=True)
class OneEvReplay:
    """Contract section 5: one EV in one world, both paths on action results."""

    unit_id: str
    world_id: int
    is_representative_world: bool
    traits: dict[str, object]
    intervals: pd.DataFrame
    plug_events: pd.DataFrame
    daily_audit: pd.DataFrame


@dataclass(frozen=True)
class OneEvTimeline:
    """Replay contract v1 §2: one EV's plan-in-force snapshots and running cost.

    ``world_id`` is a sampled world (the customer lens also shows the
    forward curve, which the fleet replay only builds for those worlds).
    ``decisions`` is the same 169-row frame as ``ForecastResult.replay_week.decisions``.
    ``plan_at_decision_kwh`` (decision, study slot) kWh per half-hour: the
    selected path's plan in force at each hourly decision, read before that
    slot's early departures and new plans (B1), zero outside the plan
    window and on the end row.  ``plan_status`` (study slot,) int64: the
    selected path's plan code (0 followed, 1-4 the non-response reasons,
    trading §10.1e) in force each slot; the selected book holds a
    non-responder's plan on record, so a nonzero code does not mean the bars
    are zero (O11).  ``cumulative`` and ``saving_to_date_gbp``: see
    ``replay_one_ev_timeline``.
    """

    unit_id: str
    world_id: int
    dispatch_locked: bool | None
    decisions: pd.DataFrame
    plan_at_decision_kwh: np.ndarray
    plan_status: np.ndarray
    cumulative: pd.DataFrame
    saving_to_date_gbp: np.ndarray


def replay_one_ev(result, unit_id: str, world_id: int | None = None) -> OneEvReplay:
    """Replay one EV in one world of ``result`` through the kernel.

    ``world_id=None`` means the result's representative world (the One EV
    default week, decision 0004 item 22).  Unknown ``unit_id`` or
    ``world_id`` raises ``KeyError``.
    """

    ev = _ev_index(result, unit_id)
    world = result.representative_world_id if world_id is None else world_id
    if not isinstance(world, (int, np.integer)) or not 0 <= world < result.world_count:
        raise KeyError(f"unknown world_id: {world_id}")
    world = int(world)
    cache = result.replay_state.replay_cache
    key = ("replay", unit_id, world)
    if key not in cache:
        _remember(cache, key, _replay(result, ev, unit_id, world))
    return cache[key]


def replay_one_ev_bands(result, unit_id: str) -> pd.DataFrame:
    """This EV across all worlds: P10/P50/P90 of closing SoC and of plugged-in.

    Forecast spread for one EV (the "Show P10–P90 across weeks" option); the
    kernel runs once on this EV for every world.
    """

    ev = _ev_index(result, unit_id)
    cache = result.replay_state.replay_cache
    key = ("bands", unit_id)
    if key in cache:
        return cache[key]
    state = result.replay_state
    piece = kernel_slice(state, np.arange(result.world_count), [ev], path_id="selected")
    frame = _one_ev_kernel_frame(piece, state.timed_start_allowed)
    capacity = float(state.units["physical_capacity_kwh"].iat[ev])
    frames = []
    for metric, unit in (("battery_soc_percent", "percent"), ("connected_share", "fraction")):
        for path_id in _paths(result):
            rows = frame.loc[frame["path_id"].eq(path_id)].sort_values(
                ["world_id", "interval_start_utc"], kind="stable"
            )
            if metric == "battery_soc_percent":
                values = 100.0 * rows["closing_battery_kwh"].to_numpy(float) / capacity
            else:
                values = rows["connected_count"].to_numpy(float)
            values = values.reshape(result.world_count, -1)
            q = np.quantile(values, (0.1, 0.5, 0.9), axis=0, method="linear")
            band = result.study_slots.loc[:, _SLOT_COLUMNS].copy()
            band.insert(0, "path_id", path_id)
            band.insert(0, "unit", unit)
            band.insert(0, "metric", metric)
            band["world_count"] = np.int64(result.world_count)
            band["p10"], band["p50"], band["p90"] = q[0], q[1], q[2]
            frames.append(band)
    bands = pd.concat(frames, ignore_index=True)
    _remember(cache, key, bands)
    return bands


def replay_one_ev_timeline(result, unit_id: str, world_id: int) -> OneEvTimeline:
    """This EV's plan snapshots and running cost in one sampled world (replay contract v1 §2).

    ``world_id`` must be one of ``result.sampled_world_ids`` (``KeyError``
    otherwise): the customer lens pairs this with the fleet replay's
    forward curves, which exist only for the sampled worlds, from one world
    control. Cached on ``replay_state.replay_cache`` under
    ``("timeline", unit_id, world)``; it shares the one-EV kernel run with
    ``replay_one_ev`` (the same cached ``OneEvReplay`` gives ``cumulative``'s
    per-slot imports), so the timeline's own cost is the plan book and the
    cumulative sums, on top of a run ``replay_one_ev`` may already have paid
    for.
    """

    world = int(world_id)
    if world not in {int(w) for w in result.sampled_world_ids}:
        raise KeyError(f"world {world_id} is not a sampled world")
    cache = result.replay_state.replay_cache
    key = ("timeline", unit_id, world)
    if key not in cache:
        _remember(cache, key, _timeline(result, unit_id, world))
    return cache[key]


def _timeline(result, unit_id: str, world: int) -> OneEvTimeline:
    """Build the ``OneEvTimeline`` of one EV and sampled world (§2)."""

    replay = replay_one_ev(result, unit_id, world)
    plan, status = _plan_snapshot(result, unit_id, world)
    cumulative, saving = _timeline_money(result, replay, world)
    dispatch_locked = (
        bool(replay.traits["dispatch_locked"]) if "dispatch_locked" in replay.traits else None
    )
    return OneEvTimeline(
        unit_id=unit_id,
        world_id=world,
        dispatch_locked=dispatch_locked,
        decisions=result.replay_week.decisions,
        plan_at_decision_kwh=plan,
        plan_status=status,
        cumulative=cumulative,
        saving_to_date_gbp=saving,
    )


def _plan_snapshot(result, unit_id: str, world: int) -> tuple[np.ndarray, np.ndarray]:
    """The selected path's plan book and ``plan_status`` for one EV (§2, trading §1.4, §10.1e).

    Two one-EV kernel passes on the same slice, because the current kernel
    exposes ``book_kwh`` (the trading overlay's preallocated output,
    ``physics._record_trading``) and the per-EV ``plan_status``
    (``physics.simulate_unit_intervals``) through different entry points;
    see the module docstring. Both read the same smart-charging draws the
    fleet run made, so this draws nothing new.
    """

    state = result.replay_state
    slots = result.study_slots
    ev = _ev_index(result, unit_id)
    piece = kernel_slice(state, [world], [ev], path_id="selected")
    slot_count = len(slots)
    decision_count = slot_count // 2 + 1

    book_slots, book_decision = market.book_decision_slots(piece.settings, slots)
    book_kwh = np.zeros((1, len(book_slots), market.BOOK_WIDTH))
    simulate_fleet_intervals(
        piece.settings,
        piece.units,
        piece.inputs,
        evaluation_effective_efficiency_miles_per_battery_kwh=piece.effective_efficiency,
        public_top_up=piece.public_top_up,
        smart_charging=piece.smart_charging,
        intraday_dispatch=piece.intraday_dispatch,
        trading_output={"selected": {"book_kwh": book_kwh, "book_decision": book_decision}},
    )
    # §2: the plan in force at decision k for study slots [book_slots[k],
    # book_slots[k] + BOOK_WIDTH); zero elsewhere and on the end row, which
    # has no book row (nothing is planned looking back from the study end).
    plan = np.zeros((decision_count, slot_count))
    for k, start in enumerate(book_slots):
        end = min(start + market.BOOK_WIDTH, slot_count)
        plan[k, start:end] = book_kwh[0, k, : end - start]

    unit_output = simulate_unit_intervals(
        piece.settings,
        piece.units,
        piece.inputs,
        effective_efficiency_miles_per_battery_kwh=piece.effective_efficiency,
        public_top_up=piece.public_top_up,
        smart_charging=piece.smart_charging,
        intraday_dispatch=piece.intraday_dispatch,
    )
    status = unit_output["plan_status"][0, :, 0].astype(np.int64)
    return plan, status


def _timeline_money(result, replay: OneEvReplay, world: int) -> tuple[pd.DataFrame, np.ndarray]:
    """§2 ``cumulative`` and ``saving_to_date_gbp`` from this EV's own replayed intervals.

    Each path's own running cost comes from ``action.slot_cost_effect_gbp``
    with zero on the "normal" side, so a single path's cost is that shared
    formula's selected-minus-zero: one formula for the weekly cost effect,
    the fleet's running saving (``model/replay.py``) and this EV's, so none
    of the three can drift apart.  ``saving_to_date_gbp`` is then the direct
    difference of the two paths' cost, not a second cumulative sum, so it
    equals ``cost_to_date(normal) - cost_to_date(selected)`` bit for bit
    (the validator checks this exactly).  The optional timed path (decision
    0007), when present, gets its own cumulative row the same way, but never
    changes ``saving_to_date_gbp``, which stays normal versus selected.
    """

    price = _world_price(result.forecast_prices, world)
    rate = _public_charge_rate(result)
    zeros = np.zeros_like(price)
    frames = []
    cost_to_date = {}
    for path_id in _paths(result):
        rows = replay.intervals.loc[replay.intervals["path_id"].eq(path_id)].sort_values(
            "slot_index"
        )
        home = rows["home_import_kwh"].to_numpy(dtype=float)
        public = rows["public_import_kwh"].to_numpy(dtype=float)
        cost = _to_date(slot_cost_effect_gbp(zeros, home, zeros, public, price, rate))
        cost_to_date[path_id] = cost
        frames.append(
            pd.DataFrame(
                {
                    "path_id": pd.Series([path_id] * len(cost), dtype=object),
                    "slot_boundary": np.arange(len(cost), dtype=np.int64),
                    "home_import_to_date_kwh": _to_date(home),
                    "public_import_to_date_kwh": _to_date(public),
                    "cost_to_date_gbp": cost,
                }
            )
        )
    cumulative = pd.concat(frames, ignore_index=True)
    saving = cost_to_date["normal"] - cost_to_date["selected"]
    return cumulative, saving


def _world_price(forecast_prices: pd.DataFrame, world: int) -> np.ndarray:
    """This world's day-ahead price in study-slot order, GBP/MWh."""

    rows = forecast_prices.loc[forecast_prices["world_id"].eq(world)].sort_values("slot_index")
    return rows["wholesale_forecast_gbp_per_mwh"].to_numpy(dtype=float)


def _public_charge_rate(result) -> float:
    """The run's own public charging rate (GBP/kWh), or the model default when not recorded.

    Read from ``result.assumptions`` by name (contract v2 §8.1) rather than
    a hard-coded value, so an edited rate never goes stale here; a result
    without assumption records (a hand-built test result) falls back to the
    same illustrative default ``run_forecast`` itself would have used.
    """

    for record in result.assumptions or ():
        if record.name == "public_charge_gbp_per_kwh":
            return float(record.value)
    return ILLUSTRATIVE_PUBLIC_CHARGE_GBP_PER_KWH


def _to_date(slot_values: np.ndarray) -> np.ndarray:
    """(slot,) -> (slot + 1,): the value at boundary b sums slots t < b."""

    return np.concatenate([[0.0], np.cumsum(slot_values)])


def _ev_index(result, unit_id: str) -> int:
    matches = np.flatnonzero(result.replay_state.units["unit_id"].to_numpy() == unit_id)
    if len(matches) != 1:
        raise KeyError(f"unknown unit_id: {unit_id}")
    return int(matches[0])


def _paths(result) -> list[str]:
    """Paths this result's one-EV frames carry: ``PATH_ORDER`` (or just normal), plus
    "timed" when the result's fleet frames carry it (decision 0007, model step 2)."""

    paths = list(PATH_ORDER) if result.model == "action" else ["normal"]
    if "timed" in set(result.fleet_world_intervals["path_id"]):
        paths.append("timed")
    return paths


def _remember(cache: dict, key: tuple, value: object) -> None:
    if len(cache) >= _CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _one_ev_kernel_frame(
    piece: KernelSlice, timed_start_allowed: np.ndarray | None = None
) -> pd.DataFrame:
    """The kernel's world frame for a slice holding one EV (so fleet sums are that EV).

    ``timed_start_allowed``, when given (decision 0007, model step 2), adds
    the optional timed path beside normal and selected, exactly as the fleet
    run does; it is the run's stored mask (``ReplayState.timed_start_allowed``),
    not recomputed here.
    """

    world_frame, _, _, _ = simulate_fleet_intervals(
        piece.settings,
        piece.units,
        piece.inputs,
        evaluation_effective_efficiency_miles_per_battery_kwh=piece.effective_efficiency,
        public_top_up=piece.public_top_up,
        smart_charging=piece.smart_charging,
        intraday_dispatch=piece.intraday_dispatch,
        timed_start_allowed=timed_start_allowed,
    )
    return world_frame


def _replay(result, ev: int, unit_id: str, world: int) -> OneEvReplay:
    state = result.replay_state
    slots = result.study_slots
    piece = kernel_slice(state, [world], [ev], path_id="selected")
    frame = _one_ev_kernel_frame(piece, state.timed_start_allowed)
    unit = state.units.iloc[ev]
    capacity = float(unit["physical_capacity_kwh"])

    interval_frames, event_frames, audit_frames = [], [], []
    for path_id in _paths(result):
        rows = frame.loc[frame["path_id"].eq(path_id)].sort_values("interval_start_utc")
        rows = rows.reset_index(drop=True)
        intervals = slots.loc[:, _SLOT_COLUMNS].copy()
        intervals.insert(0, "path_id", path_id)
        connected = rows["connected_count"].to_numpy() == 1
        intervals["location"] = _location(rows, connected)
        intervals["connected"] = connected
        closing = rows["closing_battery_kwh"].to_numpy(float)
        intervals["battery_soc_percent"] = 100.0 * closing / capacity
        intervals["closing_battery_kwh"] = closing
        intervals["home_import_kwh"] = rows["home_grid_import_kwh"].to_numpy(float)
        intervals["public_import_kwh"] = rows["public_grid_import_kwh"].to_numpy(float)
        unserved = rows["unserved_outbound_travel_battery_kwh"].to_numpy(float) + rows[
            "unserved_return_travel_battery_kwh"
        ].to_numpy(float)
        intervals["unserved_travel_kwh"] = unserved
        # Battery-side energy the smart plan still owed when the EV left early
        # (decision 0004 item 38), in the slot it left; always 0 on the
        # normal path, which has no plan.
        intervals["early_departure_shortfall_kwh"] = rows["early_departure_shortfall_kwh"].to_numpy(
            float
        )
        interval_frames.append(intervals.loc[:, _INTERVAL_COLUMNS])

        sessions = find_sessions(
            connected[np.newaxis, :, np.newaxis],
            closing[np.newaxis, :, np.newaxis],
            intervals["home_import_kwh"].to_numpy()[np.newaxis, :, np.newaxis],
        )
        event_frames.append(
            plug_in_events_frame(
                sessions,
                study_slots=slots,
                world_ids=np.array([world]),
                unit_ids=np.array([unit_id], dtype=object),
                cohort_ids=np.array([unit["cohort_id"]], dtype=object),
                capacity_kwh=np.array([capacity]),
                home_charge_efficiency=state.settings.home_charge_efficiency,
                path_id=path_id,
            )
        )
        audit_frames.append(
            _daily_audit(
                state,
                piece,
                slots,
                path_id,
                rows,
                unserved,
                capacity,
                float(unit["preferred_target_soc_fraction"]),
            )
        )

    traits = result.units.loc[result.units["unit_id"].eq(unit_id)].iloc[0].to_dict()
    return OneEvReplay(
        unit_id=unit_id,
        world_id=world,
        is_representative_world=world == result.representative_world_id,
        traits=traits,
        intervals=pd.concat(interval_frames, ignore_index=True),
        plug_events=pd.concat(event_frames, ignore_index=True),
        daily_audit=pd.concat(audit_frames, ignore_index=True),
    )


def _location(rows: pd.DataFrame, connected: np.ndarray) -> np.ndarray:
    """One location label per slot for the One EV strip.

    ``home_plugged`` only when plugged in for the whole slot, matching
    ``connected`` (the shading).  Otherwise the state that held most of the
    half-hour, with a partial home connection counted as home but not
    plugged, since the car could not charge in that slot (the kernel charges
    only full connected slots).  "away" is parked away from home.
    """

    shares = np.column_stack(
        [
            rows["home_unplugged_fraction"].to_numpy(float)
            + rows["home_connected_fraction"].to_numpy(float),
            rows["driving_fraction"].to_numpy(float),
            rows["parked_away_fraction"].to_numpy(float),
            rows["public_charging_fraction"].to_numpy(float),
        ]
    )
    others = np.array(LOCATION_ORDER[1:], dtype=object)[np.argmax(shares, axis=1)]
    return np.where(connected, "home_plugged", others).astype(object)


def _daily_audit(
    state,
    piece: KernelSlice,
    slots: pd.DataFrame,
    path_id: str,
    rows: pd.DataFrame,
    unserved: np.ndarray,
    capacity: float,
    target_fraction: float,
) -> pd.DataFrame:
    """Readiness audit per session night (USER_FEEDBACK M12).

    A charging target is not proof the car is ready: this compares the SoC at
    the departure that ends each night with the preferred target and reports
    the trip's energy need and any shortfall.  ``local_date`` is the session
    night (London 12:00 to 12:00, decision 0004 item 52) and the trip is the
    next morning's, the one that night's charging prepares for; a departure
    after the study end (the always-plugged 14:00 trip, a late weekend tail)
    has no departure SoC and shows as missing.  Departure SoC is the
    stock at the start of the departure slot (the kernel stops home charging
    at the first departure, so nothing is added in that slot before the car
    leaves).  The shortfall is the unserved travel in the slots the trip
    overlaps, inside the study.
    """

    inputs = piece.inputs
    warmup = state.settings.warmup_days
    horizon_start = slots["interval_start_utc"].iat[0]
    opening = rows["opening_battery_kwh"].to_numpy(float)
    slot_start = slots["interval_start_utc"]
    slot_end = slots["interval_end_utc"]
    dates = slots.drop_duplicates("local_date")
    records = []
    for offset, (local_date, day_label) in enumerate(zip(dates["local_date"], dates["day_label"])):
        # Sampled date index of the morning after night ``offset``.
        day = warmup + offset + 1
        drives = bool(inputs["drives_today"][0, day, 0])
        departure = pd.NaT
        departure_soc = np.nan
        need = 0.0
        shortfall = 0.0
        if drives:
            # Sampled instants are naive datetime64 in UTC.
            departure = pd.Timestamp(inputs["trip_departure_utc"][0, day, 0], tz="UTC")
            efficiency = float(piece.effective_efficiency[0, day, 0])
            need = float(
                (inputs["outbound_miles"][0, day, 0] + inputs["return_miles"][0, day, 0])
                / efficiency
            )
            slot = int((departure - horizon_start) // pd.Timedelta(minutes=30))
            # An autumn-change week ends at 11:00 London on its last morning, so
            # a late departure can fall after the horizon: SoC is then unavailable.
            if 0 <= slot < len(slots):
                departure_soc = 100.0 * opening[slot] / capacity
            trip_end = departure + _journey_duration(inputs, day)
            overlaps = ((slot_start < trip_end) & (slot_end > departure)).to_numpy()
            shortfall = float(unserved[overlaps].sum())
        target = 100.0 * target_fraction
        records.append(
            {
                "path_id": path_id,
                "local_date": local_date,
                "day_label": day_label,
                "drives_today": drives,
                "departure_utc": departure,
                "target_soc_percent": target,
                "departure_soc_percent": departure_soc,
                "trip_energy_need_kwh": need,
                "shortfall_kwh": shortfall,
                "departed_below_target": bool(
                    drives
                    and not np.isnan(departure_soc)
                    and departure_soc < target - _BELOW_TARGET_TOLERANCE_POINTS
                ),
            }
        )
    audit = pd.DataFrame(records)
    audit["departure_utc"] = pd.to_datetime(audit["departure_utc"], utc=True).dt.as_unit("ns")
    audit["departure_london"] = audit["departure_utc"].dt.tz_convert(LONDON)
    audit["local_date"] = audit["local_date"].astype(object)
    for column in (
        "target_soc_percent",
        "departure_soc_percent",
        "trip_energy_need_kwh",
        "shortfall_kwh",
    ):
        audit[column] = audit[column].astype(np.float64)
    return audit.loc[:, _AUDIT_COLUMNS]


def _journey_duration(inputs: dict[str, np.ndarray], day: int) -> pd.Timedelta:
    speed = float(inputs["drive_speed_mph"][0, day, 0])
    hours = (inputs["outbound_miles"][0, day, 0] + inputs["return_miles"][0, day, 0]) / speed
    dwell = float(inputs["destination_dwell_seconds"][0, day, 0])
    return pd.Timedelta(hours=float(hours)) + pd.Timedelta(seconds=dwell)


__all__ = [
    "OneEvReplay",
    "OneEvTimeline",
    "replay_one_ev",
    "replay_one_ev_bands",
    "replay_one_ev_timeline",
]
