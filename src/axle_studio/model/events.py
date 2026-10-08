"""Scripted grid and market events: the events table, its presets and what each event does.

What this module owns (trading contract v1 §1.1, §2 and §9.6; decision 0004
items 55, 56 and 58 F): the editable events table, its presets and their
validation, and the translation of each enabled event into the arrays the
rest of the model consumes:

- price shocks (``price_shock_known``, ``price_shock_surprise``) become
  signed net-demand profiles the prices generator adds to its own
  stochastic shocks before the supply curve (``scripted_shock_profiles``),
  so a known shock is in the day-ahead price and a surprise only in the
  intraday and imbalance prices;
- grid requests (``turn_down``, ``turn_up``) become a price adjustment the
  smart planner adds inside the window for EVs in scope, visible only to
  plans made at or after the request's notice (``planner_adjustments``), a
  trading mask (``trading_mask``) and a delivery measured against the
  settlement baseline (``event_delivery``);
- a charger control outage (``control_outage``) becomes a non-response
  probability by plug-in slot and zone (``outage_probability``).

Nothing here draws a random number: an event is the same in every world, so
a Compare of "with vs without an event" runs on identical random futures.
Every value is illustrative (none is a forecast, a DFS rate or a DNO tariff).

Slot frames.  Functions that take ``events`` also take ``slots``, a frame of
consecutive UTC half-hours with ``interval_start_utc``, ``night_index`` and
``local_date`` (``summaries.build_study_slots``, optionally preceded by
``summaries.build_warmup_slots``).  Every slot number they return is a
position in that frame, so the physics kernel passes warm-up plus study
slots and the trading overlay passes study slots, and each gets indices in
its own arrays.  ``trading_mask`` takes the output of ``event_slots``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import timedelta

import numpy as np
import pandas as pd

from axle_studio.model.assumptions import EVENT_PRESETS, PRICES, ZONE_IDS
from axle_studio.model.clock import london_wall_time_to_utc
from axle_studio.model.sampling import trapezoid_weight

EVENT_TYPES = (
    "price_shock_known",
    "price_shock_surprise",
    "turn_down",
    "turn_up",
    "control_outage",
)
PRICE_SHOCK_TYPES = ("price_shock_known", "price_shock_surprise")
REQUEST_TYPES = ("turn_down", "turn_up")
EVENT_COLUMNS = (
    "event_id",
    "event_type",
    "enabled",
    "night_index",
    "start_local_time",
    "duration_minutes",
    "size",
    "size_unit",
    "notice",
    "notice_minutes",
    "scope",
    "payment_gbp_per_mwh",
)
"""The events table's columns, in order (trading contract v1 §2.1)."""

EVENT_SLOT_COLUMNS = (
    "event_id",
    "event_type",
    "scope",
    "night_index",
    "size",
    "notice_minutes",
    "payment_gbp_per_mwh",
    "start_utc",
    "start_slot",
    "end_slot",
    "notice_utc",
    "notice_slot",
)
"""``event_slots`` columns: the contract's list plus the night, notice minutes and start."""

EVENT_DELIVERY_COLUMNS = (
    "world_id",
    "event_id",
    "event_type",
    "scope",
    "night_index",
    "delivered_kwh",
    "paid_kwh",
    "payment_gbp",
    "evidence_kind",
)

_SIZE_UNITS = {
    "price_shock_known": "GW",
    "price_shock_surprise": "GW",
    "turn_down": "MW",
    "turn_up": "MW",
    "control_outage": "fraction",
}
_DTYPES = {
    "event_id": object,
    "event_type": object,
    "enabled": bool,
    "night_index": np.int64,
    "start_local_time": object,
    "duration_minutes": np.int64,
    "size": np.float64,
    "size_unit": object,
    "notice": object,
    "notice_minutes": "Int64",
    "scope": object,
    "payment_gbp_per_mwh": np.float64,
}
_TIME_PATTERN = re.compile(r"^([01]\d|2[0-3]):(00|30)$")
_NOON_MINUTES = 12 * 60
_HALF_HOUR_MINUTES = 30
_HALF_HOUR_NS = 30 * 60 * 1_000_000_000
_MINUTE_NS = 60 * 1_000_000_000
_LONDON = "Europe/London"
# A price shock's size is bounded so a typo cannot push net demand far
# outside anything the supply curve was calibrated on (contract §2.3).
_MAX_SHOCK_GW = 10.0
_MAX_DURATION_MINUTES = 720
_MAX_NOTICE_MINUTES = 720


# --------------------------------------------------------------------------
# The table and its presets
# --------------------------------------------------------------------------


def empty_events() -> pd.DataFrame:
    """The default events table: no events, so every existing result is unchanged."""

    return pd.DataFrame({column: pd.Series(dtype=_DTYPES[column]) for column in EVENT_COLUMNS})


def preset_rows(preset_id: str, study_slots: pd.DataFrame) -> pd.DataFrame:
    """The events-table rows one preset adds for this study (contract §2.2, §9.6).

    ``preset_id`` is a key of ``assumptions.EVENT_PRESETS``; ``study_slots``
    the result's study slots.  Most presets are one row with the preset id
    as ``event_id``.  ``cold_still_week`` is one row per session night
    (``cold_still_week_n0`` ...).  ``sunny_negative_weekend`` depends on the
    start weekday: one row per Saturday and Sunday whose 10:00-16:00 window
    lies inside the study, each on the night before (a morning start names
    the day after the night's date), id ``sunny_negative_weekend_n<night>``;
    so a study gives one or two rows (or none if no weekend fits).
    """

    if preset_id not in EVENT_PRESETS:
        raise ValueError(f"unknown event preset {preset_id!r}")
    preset = EVENT_PRESETS[preset_id]
    nights = _night_dates(study_slots)
    if preset_id == "cold_still_week":
        placed = [(f"{preset_id}_n{night}", night) for night in nights]
    elif preset_id == "sunny_negative_weekend":
        study_end_ns = _utc_ns(study_slots["interval_end_utc"]).max()
        placed = []
        for night, night_date in nights.items():
            day = night_date + timedelta(days=1)
            end_ns = _local_instant_ns(day, _minutes(preset["start_local_time"])) + (
                int(preset["duration_minutes"]) * _MINUTE_NS
            )
            if day.weekday() >= 5 and end_ns <= study_end_ns:
                placed.append((f"{preset_id}_n{night}", night))
    else:
        placed = [(preset_id, int(preset["night_index"]))]
    rows = [
        {
            "event_id": event_id,
            "event_type": preset["event_type"],
            "enabled": True,
            "night_index": night,
            "start_local_time": preset["start_local_time"],
            "duration_minutes": preset["duration_minutes"],
            "size": preset["size"],
            "size_unit": _SIZE_UNITS[str(preset["event_type"])],
            "notice": preset["notice"],
            "notice_minutes": preset["notice_minutes"],
            "scope": preset["scope"],
            "payment_gbp_per_mwh": preset["payment_gbp_per_mwh"],
        }
        for event_id, night in placed
    ]
    if not rows:
        return empty_events()
    return _typed(pd.DataFrame(rows, columns=list(EVENT_COLUMNS)))


def event_table(preset_ids: Sequence[str], study_slots: pd.DataFrame) -> pd.DataFrame:
    """The events table holding the rows of each preset in ``preset_ids``, in order.

    This is the run input the Edit assumptions events selection drives; an
    empty sequence gives ``empty_events()``.
    """

    frames = [preset_rows(preset_id, study_slots) for preset_id in preset_ids]
    if not frames:
        return empty_events()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_events(
    events: pd.DataFrame,
    study_slots: pd.DataFrame,
    zone_ids: tuple[str, ...],
    *,
    blackout: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return a typed copy of ``events`` or raise ``ValueError`` naming the row and column.

    ``study_slots`` fixes the session nights and the study end; ``zone_ids``
    the zones a local event may name.  The rules are the contract's (§2.3,
    §9.6) plus one: two enabled scripted surprises may not overlap in time,
    because the prices generator carries one reveal instant per half-hour.
    ``size_unit`` is set from the type (it may be absent from the input).

    ``blackout`` (study slot,) bool marks the blackout half-hours (§10.5b):
    a grid request whose window lies wholly inside them is rejected,
    because nothing can be delivered where the aggregator has promised not
    to move charging.  Partial overlap is allowed; delivery is then
    measured outside the blackout slots only.
    """

    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a DataFrame")
    missing = [c for c in EVENT_COLUMNS if c != "size_unit" and c not in events.columns]
    if missing:
        raise ValueError(f"events table is missing columns {missing}")
    table = events.reset_index(drop=True).copy()
    table["size_unit"] = None
    nights = _night_dates(study_slots)
    study_end_ns = _utc_ns(study_slots["interval_end_utc"]).max()
    slot_starts_ns = _utc_ns(study_slots["interval_start_utc"])
    seen: set[str] = set()
    windows = []
    for row_number, row in table.iterrows():
        label = f"events row {row_number}"

        def fail(column: str, message: str, label: str = label) -> None:
            raise ValueError(f"{label}, {column}: {message}")

        event_id = row["event_id"]
        if not isinstance(event_id, str) or not event_id.strip():
            fail("event_id", "must be a non-empty text id")
        if event_id in seen:
            fail("event_id", f"{event_id!r} is used twice")
        seen.add(event_id)
        label = f"events row {row_number} ({event_id})"
        event_type = row["event_type"]
        if event_type not in EVENT_TYPES:
            fail("event_type", f"must be one of {EVENT_TYPES}")
        if not isinstance(row["enabled"], (bool, np.bool_)):
            fail("enabled", "must be True or False")
        night = _whole_number(row["night_index"])
        if night is None or night not in nights:
            fail("night_index", f"must be a session night {min(nights)}-{max(nights)}")
        start_text = row["start_local_time"]
        if not isinstance(start_text, str) or not _TIME_PATTERN.match(start_text):
            fail("start_local_time", 'must be "HH:MM" on a half-hour')
        duration = _whole_number(row["duration_minutes"])
        if (
            duration is None
            or duration % _HALF_HOUR_MINUTES
            or not _HALF_HOUR_MINUTES <= duration <= _MAX_DURATION_MINUTES
        ):
            fail("duration_minutes", "must be a multiple of 30 between 30 and 720")
        # A start at or after 12:00 is on the night's date, an earlier one on
        # the next morning, so every time names one instant inside the night.
        minutes = _minutes(start_text)
        day = nights[night] + timedelta(days=0 if minutes >= _NOON_MINUTES else 1)
        start_ns = _local_instant_ns(day, minutes)
        wall = pd.Timestamp(start_ns, tz="UTC").tz_convert(_LONDON)
        if wall.date() != day or wall.hour * 60 + wall.minute != minutes:
            fail("start_local_time", f"{start_text} does not exist on {day} (clock change)")
        end_ns = start_ns + duration * _MINUTE_NS
        if end_ns > study_end_ns:
            fail("duration_minutes", "the window must end by the study end")

        size = _float(row["size"])
        payment = _float(row["payment_gbp_per_mwh"])
        notice = row["notice"]
        notice_minutes = row["notice_minutes"]
        if notice not in ("day_ahead", "short"):
            fail("notice", 'must be "day_ahead" or "short"')
        if notice == "day_ahead":
            if not pd.isna(notice_minutes):
                fail("notice_minutes", "must be empty for day-ahead notice")
        else:
            short = _whole_number(notice_minutes)
            if short is None or not 0 <= short <= _MAX_NOTICE_MINUTES:
                fail("notice_minutes", "short notice needs whole minutes between 0 and 720")
        scope = row["scope"]
        if scope != "national" and scope not in zone_ids:
            fail("scope", f'must be "national" or one of {zone_ids}')

        if event_type in PRICE_SHOCK_TYPES:
            # Wholesale prices are national (decision 0004 item 53).
            if scope != "national":
                fail("scope", "price shocks are national")
            if size is None or not np.isfinite(size) or size == 0 or abs(size) > _MAX_SHOCK_GW:
                fail("size", "must be a non-zero GW change between -10 and +10")
            if payment is None or not np.isnan(payment):
                fail("payment_gbp_per_mwh", "must be empty for a price shock")
            if event_type == "price_shock_known" and notice != "day_ahead":
                fail("notice", "a known shock is known day-ahead")
            if event_type == "price_shock_surprise" and notice != "short":
                fail("notice", "a surprise shock has short notice")
        elif event_type in REQUEST_TYPES:
            if payment is None or not np.isfinite(payment) or payment < 0:
                fail("payment_gbp_per_mwh", "must be a payment of at least 0")
            if size is None or not (np.isnan(size) or (np.isfinite(size) and size > 0)):
                fail("size", "must be empty (no cap) or a positive MW cap")
            # Short-notice replanning is task H6 (wave 4), not built yet.
            if notice == "short":
                fail("notice", "short-notice requests are not modelled yet")
            if blackout is not None:
                window = (slot_starts_ns >= start_ns) & (slot_starts_ns < end_ns)
                if np.asarray(blackout, dtype=bool)[window].all():
                    fail("start_local_time", "nothing can be delivered in a blackout window")
        else:  # control_outage (contract §9.6)
            if size is None or not (np.isfinite(size) and 0 < size <= 1):
                fail("size", "must be a share of sessions above 0 and at most 1")
            if payment is None or not np.isnan(payment):
                fail("payment_gbp_per_mwh", "must be empty for a control outage")
            if notice != "short" or _whole_number(notice_minutes) != 0:
                fail("notice", "a control outage is known to nobody in advance: short, 0 minutes")
        table.at[row_number, "size_unit"] = _SIZE_UNITS[event_type]
        if row["enabled"]:
            windows.append((row_number, event_id, event_type, scope, start_ns, end_ns))

    # Two requests whose scopes overlap (national overlaps every zone) may
    # not overlap in time, so delivery is paid once.  Two scripted
    # surprises may not overlap: the generator has one reveal per slot.
    for index, first in enumerate(windows):
        for second in windows[index + 1 :]:
            if not (first[4] < second[5] and second[4] < first[5]):
                continue
            both_requests = first[2] in REQUEST_TYPES and second[2] in REQUEST_TYPES
            scopes_overlap = "national" in (first[3], second[3]) or first[3] == second[3]
            if both_requests and scopes_overlap:
                raise ValueError(
                    f"events row {second[0]} ({second[1]}): overlaps request {first[1]!r} "
                    "in time and scope"
                )
            if first[2] == second[2] == "price_shock_surprise":
                raise ValueError(
                    f"events row {second[0]} ({second[1]}): overlaps surprise {first[1]!r}"
                )
    return _typed(table)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def event_slots(events: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    """One row per enabled event with its window and notice as positions in ``slots``.

    ``events`` is a validated table; ``slots`` a slot frame (module
    docstring).  ``start_slot`` and ``end_slot`` (exclusive) bound the
    window; ``notice_utc`` (int64 UTC ns) is when the event becomes known
    and ``notice_slot`` the first slot starting at or after it (0 if it is
    before the frame).

    Notice (contract §2.1, review B2): a ``day_ahead`` event of night n is
    known at the night's day-ahead decision instant, 13:00 London on the day
    before the night's date (the day-ahead publication hour,
    ``PRICES["day_ahead_publication_local_hour"]``), so the day-ahead
    position and every plan of that night can see it, a morning event
    included.  A ``short`` event is known ``notice_minutes`` before its
    window starts.
    """

    enabled = events.loc[events["enabled"].astype(bool)].reset_index(drop=True)
    starts_ns = _utc_ns(slots["interval_start_utc"])
    nights = _night_dates(slots)
    publication_minutes = 60.0 * PRICES["day_ahead_publication_local_hour"].value
    rows = []
    for row in enabled.itertuples(index=False):
        night = int(row.night_index)
        minutes = _minutes(row.start_local_time)
        day = nights[night] + timedelta(days=0 if minutes >= _NOON_MINUTES else 1)
        start_ns = _local_instant_ns(day, minutes)
        if row.notice == "day_ahead":
            notice_ns = _local_instant_ns(nights[night] - timedelta(days=1), publication_minutes)
        else:
            notice_ns = start_ns - int(row.notice_minutes) * _MINUTE_NS
        start_slot = int(np.searchsorted(starts_ns, start_ns))
        if start_slot >= len(starts_ns) or starts_ns[start_slot] != start_ns:
            raise ValueError(f"event {row.event_id!r} does not start on a slot of this frame")
        rows.append(
            {
                "event_id": row.event_id,
                "event_type": row.event_type,
                "scope": row.scope,
                "night_index": night,
                "size": float(row.size),
                "notice_minutes": row.notice_minutes,
                "payment_gbp_per_mwh": float(row.payment_gbp_per_mwh),
                "start_utc": pd.Timestamp(start_ns, tz="UTC"),
                "start_slot": start_slot,
                "end_slot": start_slot + int(row.duration_minutes) // _HALF_HOUR_MINUTES,
                "notice_utc": notice_ns,
                "notice_slot": int(np.searchsorted(starts_ns, notice_ns, side="left")),
            }
        )
    frame = pd.DataFrame(rows, columns=list(EVENT_SLOT_COLUMNS))
    return frame.astype(
        {
            "night_index": np.int64,
            "size": np.float64,
            "notice_minutes": "Int64",
            "payment_gbp_per_mwh": np.float64,
            "start_utc": "datetime64[ns, UTC]",
            "start_slot": np.int64,
            "end_slot": np.int64,
            "notice_utc": np.int64,
            "notice_slot": np.int64,
        }
    )


# --------------------------------------------------------------------------
# Price shocks
# --------------------------------------------------------------------------


def scripted_shock_profiles(
    events: pd.DataFrame, slots: pd.DataFrame, slot_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """``known_gw`` and ``surprise_gw`` (slot,): the enabled price shocks as net-demand profiles.

    GW added to system net demand in each slot of ``slots`` (``slot_count``
    = its length), signed by ``size`` (positive = tighter).  Each shock is
    the item 56 trapezoid, exactly as the generator shapes its own shocks
    (``sampling.trapezoid_weight``), so a scripted shock and a stochastic one
    of the same size, time and class move prices identically (contract
    §2.5).  Known shocks go in ``known_gw`` (they enter the day-ahead
    price), surprises in ``surprise_gw`` (intraday from their reveal, and
    imbalance).  All zeros when there is no price shock, which reproduces
    the generator bit for bit.
    """

    known = np.zeros(slot_count)
    surprise = np.zeros(slot_count)
    for row in _price_shocks(events, slots).itertuples(index=False):
        slot, weight = _shock_slots(row, slot_count)
        target = known if row.event_type == "price_shock_known" else surprise
        target[slot] += row.size * weight
    return known, surprise


def scripted_surprise_reveal_hours(
    events: pd.DataFrame, slots: pd.DataFrame, slot_count: int, *, gate_closure_minutes: float
) -> np.ndarray:
    """(slot,) whole hours before each slot's gate closure at which its scripted surprise is known.

    The generator's ``scripted_surprise_reveal_hours`` input.  A scripted
    surprise is revealed ``notice_minutes`` before its window starts
    (contract §2.5); half-hour k of the window closes ``gate_closure_minutes``
    before it starts, so the reveal is ``k/2 + (notice − gate)/60`` hours
    before gate closure, floored to the intraday step that already includes
    it (the same rule as ``sampling._reveal_steps`` for stochastic
    surprises).  A negative value means revealed after gate closure: only
    the imbalance price moves.  Zero outside scripted surprise windows,
    where no scripted surprise GW exists to reveal.
    """

    reveal = np.zeros(slot_count)
    for row in _price_shocks(events, slots).itertuples(index=False):
        if row.event_type != "price_shock_surprise":
            continue
        slot, _ = _shock_slots(row, slot_count)
        k = slot - row.start_slot
        lead_hours = (int(row.notice_minutes) - gate_closure_minutes) / 60.0
        reveal[slot] = np.floor(0.5 * k + lead_hours + 1e-9)
    return reveal


def scripted_shock_rows(
    events: pd.DataFrame, slots: pd.DataFrame, world_count: int
) -> pd.DataFrame:
    """The scripted price shocks as shock-table rows, one per world and shock (contract §5.6a).

    Columns: ``world_id``, ``shock_id`` (the ``event_id``), ``source``
    ("scripted"), ``shock_class`` ("scripted"), ``direction``, ``known_day_ahead``,
    ``start_slot_index`` (the ``slot_index`` of ``slots`` at the start),
    ``start_utc``, ``duration_slots`` and ``size_gw`` (> 0; the sign is the
    direction).  The same shock appears in every world because it is a
    scenario, not a draw.
    """

    shocks = _price_shocks(events, slots)
    slot_index = slots["slot_index"].to_numpy(dtype=np.int64)
    rows = [
        {
            "world_id": world,
            "shock_id": row.event_id,
            "source": "scripted",
            "shock_class": "scripted",
            "direction": "up" if row.size > 0 else "down",
            "known_day_ahead": row.event_type == "price_shock_known",
            "start_slot_index": int(slot_index[row.start_slot]),
            "start_utc": row.start_utc,
            "duration_slots": int(row.end_slot - row.start_slot),
            "size_gw": abs(float(row.size)),
        }
        for world in range(world_count)
        for row in shocks.itertuples(index=False)
    ]
    columns = [
        "world_id",
        "shock_id",
        "source",
        "shock_class",
        "direction",
        "known_day_ahead",
        "start_slot_index",
        "start_utc",
        "duration_slots",
        "size_gw",
    ]
    return pd.DataFrame(rows, columns=columns).astype(
        {
            "world_id": np.int64,
            "known_day_ahead": bool,
            "start_slot_index": np.int64,
            "start_utc": "datetime64[ns, UTC]",
            "duration_slots": np.int64,
            "size_gw": np.float64,
        }
    )


# --------------------------------------------------------------------------
# Grid requests and the control outage
# --------------------------------------------------------------------------


def planner_adjustments(
    events: pd.DataFrame, slots: pd.DataFrame, zone_count: int, slot_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """``notice_slot`` (request,) int64 and ``adjustment_gbp_per_mwh`` (request, zone, slot).

    One entry per enabled ``turn_down`` or ``turn_up`` request, in table
    order; zones in ``assumptions.ZONE_IDS`` order (a national request
    covers every zone).  The adjustment is ``+payment`` inside a turn-down
    window and ``−payment`` inside a turn-up window, 0 elsewhere (contract
    §2.4): a plan adds it to its ranking price, so charging in a turn-down
    window is ranked dearer by the payment it would forgo and charging in a
    turn-up window cheaper by the bonus.  It changes plan ranking only; it
    never forces import.  A consumer applies request e only to decisions at
    or after ``notice_slot[e]``, so no plan sees a request before it is
    announced (review B2).
    """

    requests = event_slots(events, slots)
    requests = requests.loc[requests["event_type"].isin(REQUEST_TYPES)].reset_index(drop=True)
    adjustment = np.zeros((len(requests), zone_count, slot_count))
    for index, row in enumerate(requests.itertuples(index=False)):
        sign = 1.0 if row.event_type == "turn_down" else -1.0
        zones = _scope_zones(row.scope, zone_count)
        adjustment[index, zones, row.start_slot : row.end_slot] = sign * row.payment_gbp_per_mwh
    return requests["notice_slot"].to_numpy(dtype=np.int64), adjustment


def outage_probability(
    events: pd.DataFrame, slots: pd.DataFrame, zone_count: int, slot_count: int
) -> np.ndarray:
    """(zone, slot) share of sessions plugging in then that ignore their plan (contract §9.6).

    The largest enabled ``control_outage`` size covering each plug-in slot
    and zone, 0 elsewhere.  The kernel raises a session's non-response
    probability to this value for a session whose plan starts in the slot,
    and uses the session's existing uniform, so no draw moves and base-rate
    non-responders stay non-responders.
    """

    probability = np.zeros((zone_count, slot_count))
    rows = event_slots(events, slots)
    for row in rows.loc[rows["event_type"].eq("control_outage")].itertuples(index=False):
        zones = _scope_zones(row.scope, zone_count)
        window = probability[zones, row.start_slot : row.end_slot]
        probability[zones, row.start_slot : row.end_slot] = np.maximum(window, row.size)
    return probability


def trading_mask(
    slots: pd.DataFrame,
    slot_count: int,
    *,
    known_at_utc: int | None,
    blackout: np.ndarray | None = None,
) -> np.ndarray:
    """(slot,) bool: False inside every request window already announced, True elsewhere.

    ``slots`` is the output of ``event_slots``; ``slot_count`` the length of
    the frame it was built on.  A request pauses trading in its window so
    its delivery is paid once, by the event (contract §4.6).  Only requests
    whose ``notice_utc`` is at or before ``known_at_utc`` (int64 UTC ns)
    count, so a trading decision never uses a request it could not know;
    ``known_at_utc=None`` uses every request (the settlement mask).

    ``blackout`` (slot,) bool is also False in the mask at every
    ``known_at_utc`` (§10.5b): the blackout is set before the run, so no
    position, intraday trade or settlement is ever made there.
    """

    mask = np.ones(slot_count, dtype=bool) if blackout is None else ~np.asarray(blackout, bool)
    requests = slots.loc[slots["event_type"].isin(REQUEST_TYPES)]
    if known_at_utc is not None:
        requests = requests.loc[requests["notice_utc"] <= int(known_at_utc)]
    for row in requests.itertuples(index=False):
        mask[row.start_slot : row.end_slot] = False
    return mask


def event_delivery(
    events: pd.DataFrame,
    slots: pd.DataFrame,
    scope_baseline_kwh: np.ndarray,
    scope_metered_kwh: np.ndarray,
) -> pd.DataFrame:
    """Per (world, request): delivery against the baseline and the illustrative payment.

    ``slots`` are the study slots; ``scope_baseline_kwh`` and
    ``scope_metered_kwh`` are (world, scope, study slot) kWh per half-hour
    of the adjusted settlement baseline ``B`` (trading contract §4.1, owned
    by ``market``) and metered selected-path home import ``M``; scope 0 is
    the fleet and 1-4 the zones in ``ZONE_IDS`` order.  Per window slot the
    delivery is ``B − M`` for a turn-down and ``M − B`` for a turn-up
    (contract §2.4).  Columns:

    - ``delivered_kwh``: the window sum of that signed delivery (it can be
      negative when the scope imported against the request);
    - ``paid_kwh``: the window sum of ``min(max(0, delivery), cap)``, with
      ``cap = size MW × 1000 × 0.5`` kWh per slot (no cap when ``size`` is
      NaN), the volume the request pays for;
    - ``payment_gbp``: ``paid_kwh ÷ 1000 × payment_gbp_per_mwh``,
      illustrative (never Axle cash).

    ``night_index`` is the night the window starts in: the ledger books a
    request's payment on that night (§4.7).  Rows are world-major.
    """

    rows = event_slots(events, slots)
    requests = rows.loc[rows["event_type"].isin(REQUEST_TYPES)].reset_index(drop=True)
    baseline = np.asarray(scope_baseline_kwh, dtype=float)
    metered = np.asarray(scope_metered_kwh, dtype=float)
    world_count = baseline.shape[0]
    frames = []
    for row in requests.itertuples(index=False):
        scope = 0 if row.scope == "national" else 1 + ZONE_IDS.index(row.scope)
        window = slice(row.start_slot, row.end_slot)
        gap = baseline[:, scope, window] - metered[:, scope, window]
        delivery = gap if row.event_type == "turn_down" else -gap
        cap_kwh = np.inf if np.isnan(row.size) else row.size * 1000.0 * 0.5
        paid = np.minimum(np.maximum(delivery, 0.0), cap_kwh).sum(axis=1)
        frames.append(
            pd.DataFrame(
                {
                    "world_id": np.arange(world_count, dtype=np.int64),
                    "event_id": row.event_id,
                    "event_type": row.event_type,
                    "scope": row.scope,
                    "night_index": np.int64(row.night_index),
                    "delivered_kwh": delivery.sum(axis=1),
                    "paid_kwh": paid,
                    "payment_gbp": paid / 1000.0 * row.payment_gbp_per_mwh,
                    "evidence_kind": "illustrative_synthetic",
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            {
                column: pd.Series(dtype=np.int64 if column in ("world_id", "night_index") else None)
                for column in EVENT_DELIVERY_COLUMNS
            }
        ).astype({"delivered_kwh": np.float64, "paid_kwh": np.float64, "payment_gbp": np.float64})
    frame = pd.concat(frames, ignore_index=True)
    order = np.lexsort((frame.index.to_numpy(), frame["world_id"].to_numpy()))
    return frame.iloc[order].reset_index(drop=True).loc[:, list(EVENT_DELIVERY_COLUMNS)]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _price_shocks(events: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    rows = event_slots(events, slots)
    return rows.loc[rows["event_type"].isin(PRICE_SHOCK_TYPES)].reset_index(drop=True)


def _shock_slots(row, slot_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Slots of one shock inside the frame and their trapezoid weights."""

    duration = int(row.end_slot - row.start_slot)
    k = np.arange(duration)
    slot = row.start_slot + k
    inside = slot < slot_count
    return slot[inside], trapezoid_weight(k, np.full(duration, duration))[inside]


def _scope_zones(scope: str, zone_count: int) -> np.ndarray:
    if scope == "national":
        return np.arange(zone_count)
    return np.array([ZONE_IDS.index(scope)])


def _night_dates(slots: pd.DataFrame) -> dict[int, object]:
    """Session night index -> the night's London date (``D_n``) in a slot frame."""

    first = slots.drop_duplicates("night_index")
    return dict(zip(first["night_index"].astype(int), first["local_date"], strict=True))


def _minutes(text: str) -> int:
    hours, minutes = text.split(":")
    return int(hours) * 60 + int(minutes)


def _local_instant_ns(day, minutes: float) -> int:
    """London wall clock ``minutes`` after midnight on ``day`` as int64 UTC ns."""

    return int(london_wall_time_to_utc(np.datetime64(day, "D"), minutes).astype(np.int64))


def _utc_ns(values: pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def _whole_number(value) -> int | None:
    if isinstance(value, (bool, np.bool_)) or value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def _float(value) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if value is None or value is pd.NA:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _typed(table: pd.DataFrame) -> pd.DataFrame:
    typed = table.loc[:, list(EVENT_COLUMNS)].copy()
    typed["notice_minutes"] = pd.array(
        [None if pd.isna(value) else int(value) for value in typed["notice_minutes"]],
        dtype="Int64",
    )
    typed["size"] = [np.nan if _float(v) is None else _float(v) for v in typed["size"]]
    typed["payment_gbp_per_mwh"] = [
        np.nan if _float(v) is None else _float(v) for v in typed["payment_gbp_per_mwh"]
    ]
    return typed.astype({c: t for c, t in _DTYPES.items() if c != "notice_minutes"}).reset_index(
        drop=True
    )


__all__ = [
    "EVENT_COLUMNS",
    "EVENT_DELIVERY_COLUMNS",
    "EVENT_SLOT_COLUMNS",
    "EVENT_TYPES",
    "PRICE_SHOCK_TYPES",
    "REQUEST_TYPES",
    "empty_events",
    "event_delivery",
    "event_slots",
    "event_table",
    "outage_probability",
    "planner_adjustments",
    "preset_rows",
    "scripted_shock_profiles",
    "scripted_shock_rows",
    "scripted_surprise_reveal_hours",
    "trading_mask",
    "validate_events",
]
