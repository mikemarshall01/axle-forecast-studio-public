"""Firm-MW product frames: what the fleet could sell as a product, and what it delivered.

What this owns (trading contract v1 §10.5a and §10.5c-h, lane J5; decision
0004 item 59): the supplier-facing reading of the availability frames that
``availability`` builds (lane J2).

- ``product_sheet`` (§10.5a): per product window (evening, overnight,
  morning), direction and hold duration, the MW the fleet can hold, how firm
  it is, the energy it can move, the recovery after a turn-down call, the
  longest sensible event and the notice.
- ``availability_value_summary`` (§10.5c): deliverable MW weighted by the
  day-ahead price, and the week's deliverable energy valued at that price.
- ``settlement_file`` (§10.5d): a per-EV settlement file for the
  representative week, with the baseline effect as an unallocated fleet row.
- ``charge_completion_summary`` (§10.5e): the share of home sessions that
  reach the preferred target by unplug, on time and early departures apart.
- ``supplier_position_additions`` (§10.5f): the net shape change and the 1-h
  firm figures added to ``supplier_positions``.
- ``firmness_by_manufacturer`` and ``manufacturer_summary`` (§10.5g-h):
  physical firmness per charger maker and the diversification explainer.
- ``new_totals`` / ``accumulate`` / ``build_frames``: the chunk-loop hook
  ``summaries.build_summaries`` calls after the availability hook.
- ``firm_mw_comparison``: the Compare rows for two runs.

Conventions as ``availability``: ``p10`` is ``numpy.quantile(..., 0.1)``,
the value exceeded in 90 % of worlds (the trader's "P90"); ``firm_share`` is
``p10 ÷ p50``.  Statistics are per world first, then across worlds (never a
sum of percentiles), and a world with no value for a row is left out of that
row's statistics, with ``world_count`` counting the worlds kept (NaN rule
O5).  Every MW, MWh and £ is illustrative and synthetic: nothing here is a
bid, a settlement or Axle cash.  Nothing here draws random numbers.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import pandas as pd

from axle_studio.model import availability
from axle_studio.model.action import NOT_RECOVERED_TOLERANCE_KWH

EVIDENCE_KIND = "illustrative_synthetic"
PATH_IDS = ("normal", "selected")
"""Default charge-completion paths; a caller passes ``(*PATH_IDS, "timed")`` to ``new_totals``
to also complete the optional timed path (decision 0007, model step 2), same columns, same
``DEPARTURES`` split, one extra ``path_id`` value."""
DEPARTURES = ("all", "on_time", "early")
_ONE_HOUR = 1.0
_KW_PER_MW = 1000.0

PRODUCT_SHEET_COLUMNS = [
    "window",
    "window_start_local",
    "window_end_local",
    "direction",
    "duration_hours",
    "world_count",
    "window_mean_mw_p05",
    "window_mean_mw_p10",
    "window_mean_mw_p50",
    "window_mean_mw_p90",
    "window_min_mw_p10",
    "window_min_mw_p50",
    "window_min_mw_p90",
    "firm_share",
    "firm_share_p05",
    "intraday_firm_mw_p50",
    "intraday_known_share_p50",
    "movable_energy_mwh_p10",
    "movable_energy_mwh_p50",
    "movable_energy_mwh_p90",
    "recovery_energy_mwh_p50",
    "rebound_capacity_mw_p50",
    "recovery_hours_p50",
    "max_event_length_hours",
    "notice_day_ahead_hours",
    "notice_intraday_hours",
    "ramp_hours",
    "evidence_kind",
]
VALUE_SUMMARY_COLUMNS = [
    "direction",
    "duration_hours",
    "metric",
    "unit",
    "world_count",
    "mean",
    "p10",
    "p50",
    "p90",
    "evidence_kind",
]
VALUE_METRICS = (
    ("price_weighted_mw", "MW"),
    ("simple_mean_mw", "MW"),
    ("value_at_day_ahead_gbp_per_week", "GBP per week"),
)
SETTLEMENT_FILE_COLUMNS = [
    "meter_point_id",
    "cohort_id",
    "manufacturer_id",
    "zone_id",
    "control_group",
    "slot_index",
    "night_index",
    "interval_start_utc",
    "interval_start_london",
    "settlement_date",
    "settlement_period",
    "unmanaged_kwh",
    "metered_kwh",
    "deviation_kwh",
    "settled_kwh",
    "day_ahead_gbp_per_mwh",
    "payment_gbp",
    "evidence_kind",
]
BASELINE_EFFECT_ID = "baseline_effect"
FLEET_ID = "fleet"
COMPLETION_COLUMNS = [
    "group_id",
    "path_id",
    "departure",
    "ev_count",
    "world_count",
    "session_count_mean",
    "completed_share_mean",
    "completed_share_p10",
    "completed_share_p50",
    "completed_share_p90",
    "shortfall_kwh_mean",
    "shortfall_kwh_p50",
    "evidence_kind",
]
POSITION_ADDITION_COLUMNS = [
    "unmanaged_mw_p50",
    "net_change_mw_p10",
    "net_change_mw_p50",
    "net_change_mw_p90",
    "deliverable_turn_down_1h_mw_p10",
    "deliverable_turn_up_1h_mw_p10",
]
FIRMNESS_COLUMNS = [
    "manufacturer_id",
    "manufacturer_label",
    "share",
    "ev_count",
    "response_rate",
    "outage_probability_per_night",
    "direction",
    "duration_hours",
    "world_count",
    "firmness_mean",
    "firmness_p10",
    "firmness_p50",
    "firmness_p90",
    "outage_nights_mean",
    "outage_nights_p50",
    "evidence_kind",
]
MANUFACTURER_SUMMARY_COLUMNS = [
    "manufacturer_id",
    "manufacturer_label",
    "share",
    "ev_count",
    "response_rate",
    "outage_probability_per_night",
    "evening_potential_mw_p50",
    "largest_share",
    "evidence_kind",
]
FIRM_MW_COMPARISON_COLUMNS = ["frame", "key", "metric", "unit", "value_a", "value_b"]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _stats(values: np.ndarray, levels: Sequence[float]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Worlds with a value, and linear quantiles over them, along axis 0 (NaN with none)."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a row with no world
        count = np.isfinite(values).sum(axis=0)
        quantiles = np.nanquantile(values, levels, axis=0, method="linear")
    return count, list(quantiles)


def _nanmean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``numerator ÷ denominator`` where the denominator is positive, else NaN."""

    positive = denominator > 0
    return np.where(positive, numerator / np.where(positive, denominator, 1.0), np.nan)


def _window_mask(labels: np.ndarray, start: str, end: str) -> np.ndarray:
    """Slots whose London "HH:MM" label lies in ``[start, end)``; a window may wrap midnight."""

    if start < end:
        return (labels >= start) & (labels < end)
    return (labels >= start) | (labels < end)


def _noon_to_noon_hours(local_time: str) -> float:
    """Wall-clock hours of "HH:MM" counted from the night's midnight; before 12:00 is next day."""

    hours = int(local_time[:2]) + int(local_time[3:]) / 60.0
    return hours + 24.0 if hours < 12.0 else hours


def _world_slot_cubes(world_slot: pd.DataFrame) -> tuple[int, int, np.ndarray]:
    """World count, slot count and the London "HH:MM" label of each slot of the world-slot frame.

    ``availability_world_slot`` is world-major (world, slot, direction,
    duration), so world 0's first row per slot carries the slot keys.
    """

    cells = len(availability.CELLS)
    world_count = int(world_slot["world_id"].max()) + 1
    slot_count = len(world_slot) // (world_count * cells)
    first = world_slot["interval_start_london"].iloc[: slot_count * cells : cells]
    return world_count, slot_count, first.dt.strftime("%H:%M").to_numpy(dtype=object)


def _cube(world_slot: pd.DataFrame, column: str, world_count: int, slot_count: int) -> np.ndarray:
    """(world, slot, cell) array of one ``availability_world_slot`` column."""

    return (
        world_slot[column]
        .to_numpy(dtype=float)
        .reshape(world_count, slot_count, len(availability.CELLS))
    )


# --------------------------------------------------------------------------
# §10.5a Product sheet
# --------------------------------------------------------------------------


def product_sheet(
    world_slot: pd.DataFrame,
    *,
    day_ahead_decision_local_hour: float,
    intraday_decision_local_time: str,
) -> pd.DataFrame:
    """The §10.5a product sheet from ``availability_world_slot``: 24 rows.

    One row per (window, direction, duration_hours) in ``PRODUCT_WINDOWS``
    and ``availability.CELLS`` order.  A window is every study slot whose
    London half-hour lies in it, on all nights (both copies of a repeated
    autumn half-hour count).  Per world: the window mean and minimum of
    ``deliverable_kw`` in MW (slots whose hold window runs past the study
    end are left out, NaN rule O5), the energy movable in the window (0.5-h
    rows), the recovery after a full-length turn-down call and the 17:00
    firm figure; then quantiles across worlds.

    ``day_ahead_decision_local_hour`` is the day-ahead decision hour on the
    day before the night (the ``day_ahead_publication_local_hour`` record,
    13:00) and ``intraday_decision_local_time`` the intraday decision
    ("17:00" by default record); both give the notice columns in wall-clock
    hours, so a clock-change night's elapsed notice differs by an hour.
    """

    world_count, slot_count, labels = _world_slot_cubes(world_slot)

    def cube(column: str) -> np.ndarray:
        return _cube(world_slot, column, world_count, slot_count)

    deliverable_mw = cube("deliverable_kw") / _KW_PER_MW
    eligible_mw = cube("eligible_power_kw") / _KW_PER_MW
    intraday_p10_mw = cube("intraday_p10_kw") / _KW_PER_MW
    intraday_known = cube("intraday_known_mean_kw")
    intraday_mean = cube("intraday_mean_kw")
    decision_hours = _noon_to_noon_hours(intraday_decision_local_time)
    rows = []
    for window, (start, end) in availability.PRODUCT_WINDOWS.items():
        mask = _window_mask(labels, start, end)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a world with no defined slot
            mean_mw = np.nanmean(deliverable_mw[:, mask], axis=1)  # (world, cell)
            min_mw = np.nanmin(deliverable_mw[:, mask], axis=1)
            rebound_mw = np.nanmean(eligible_mw[:, mask], axis=1)
            # Intraday columns are NaN before the night's decision slot, so
            # the means cover the window's slots at or after s_n.
            intraday_firm = np.nanmean(intraday_p10_mw[:, mask], axis=1)
        known_share = _ratio(
            np.nansum(intraday_known[:, mask], axis=1), np.nansum(intraday_mean[:, mask], axis=1)
        )
        # Each half-hour counted once: the 0.5-h row's slot MW x 0.5 h.
        movable_mwh = np.nansum(deliverable_mw[:, mask] * availability.SLOT_HOURS, axis=1)
        start_hours = _noon_to_noon_hours(start)
        block = []
        for cell, (direction, hours) in enumerate(availability.CELLS):
            count, (p05, p10, p50, p90) = _stats(mean_mw[:, cell], (0.05, 0.1, 0.5, 0.9))
            _, (min_p10, min_p50, min_p90) = _stats(min_mw[:, cell], (0.1, 0.5, 0.9))
            turn_down = direction == "turn_down"
            half_hour = hours == availability.DURATION_HOURS[0]
            # A full-length turn-down call defers hours x window mean of
            # energy past the window; it comes back at most at the power of
            # the EVs that paused (eligible power), so energy ÷ power is the
            # time to recover.  Turn-up pulls charging forward instead; its
            # later reduction is the net change of §10.5f, so NaN here.
            recovery_mwh = hours * mean_mw[:, cell]
            recovery_hours = _ratio(recovery_mwh, rebound_mw[:, cell])
            movable = _stats(movable_mwh[:, cell], (0.1, 0.5, 0.9))[1]
            block.append(
                {
                    "window": window,
                    "window_start_local": start,
                    "window_end_local": end,
                    "direction": direction,
                    "duration_hours": float(hours),
                    "world_count": int(count),
                    "window_mean_mw_p05": p05,
                    "window_mean_mw_p10": p10,
                    "window_mean_mw_p50": p50,
                    "window_mean_mw_p90": p90,
                    "window_min_mw_p10": min_p10,
                    "window_min_mw_p50": min_p50,
                    "window_min_mw_p90": min_p90,
                    "firm_share": float(_ratio(p10, p50)),
                    "firm_share_p05": float(_ratio(p05, p50)),
                    "intraday_firm_mw_p50": _stats(intraday_firm[:, cell], (0.5,))[1][0],
                    "intraday_known_share_p50": _stats(known_share[:, cell], (0.5,))[1][0],
                    "movable_energy_mwh_p10": movable[0] if half_hour else np.nan,
                    "movable_energy_mwh_p50": movable[1] if half_hour else np.nan,
                    "movable_energy_mwh_p90": movable[2] if half_hour else np.nan,
                    "recovery_energy_mwh_p50": (
                        _stats(recovery_mwh, (0.5,))[1][0] if turn_down else np.nan
                    ),
                    "rebound_capacity_mw_p50": (
                        _stats(rebound_mw[:, cell], (0.5,))[1][0] if turn_down else np.nan
                    ),
                    "recovery_hours_p50": (
                        _stats(recovery_hours, (0.5,))[1][0] if turn_down else np.nan
                    ),
                    # Wall-clock hours from τ_DA (the decision hour on the
                    # day before the night) and from τ_n to the window start.
                    "notice_day_ahead_hours": start_hours + 24.0 - day_ahead_decision_local_hour,
                    "notice_intraday_hours": start_hours - decision_hours,
                    # The model dispatches whole half-hours: a call takes
                    # effect at the next slot boundary.  A model property,
                    # not a measurement.
                    "ramp_hours": availability.SLOT_HOURS,
                    "evidence_kind": EVIDENCE_KIND,
                }
            )
        for direction in availability.DIRECTIONS:
            rows_of = [row for row in block if row["direction"] == direction]
            length = max_event_length_hours(
                [row["duration_hours"] for row in rows_of],
                [row["window_mean_mw_p50"] for row in rows_of],
            )
            for row in rows_of:
                row["max_event_length_hours"] = length
        rows.extend(block)
    frame = pd.DataFrame(rows).loc[:, PRODUCT_SHEET_COLUMNS]
    return frame.astype({"world_count": np.int64})


def max_event_length_hours(duration_hours: Sequence[float], mean_mw_p50: Sequence[float]) -> float:
    """Longest hold duration whose median window MW keeps ``MAX_EVENT_SHARE`` of the 0.5-h row's.

    ``duration_hours`` and ``mean_mw_p50`` are one (window, direction)
    block's rows, the 0.5-h row first.  A call longer than this delivers
    less than a quarter of the half-hour figure (the capacity-share
    precedent of results-v2 §3.6e).  NaN when the half-hour figure is NaN or
    not positive (no event to size) or no duration qualifies.
    """

    hours = np.asarray(duration_hours, dtype=float)
    p50 = np.asarray(mean_mw_p50, dtype=float)
    base = p50[0]
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    keep = p50 >= availability.MAX_EVENT_SHARE * base
    return float(hours[keep].max()) if keep.any() else float("nan")


# --------------------------------------------------------------------------
# §10.5c Price-weighted availability
# --------------------------------------------------------------------------


def availability_value_summary(
    world_slot: pd.DataFrame, day_ahead_gbp_per_mwh: np.ndarray
) -> pd.DataFrame:
    """§10.5c: deliverable MW weighted by the day-ahead price, per (direction, duration, metric).

    ``day_ahead_gbp_per_mwh`` (world, study slot) is the day-ahead price after
    every shock and the user curve (``deviation_world_slot``'s column).
    Turn-down is worth the price it avoids, so its weight is ``max(0, P)``;
    turn-up is worth something only when the price is negative, so its
    weight is ``max(0, -P)``.  Per world: ``price_weighted_mw`` (NaN when the
    weights sum to 0, e.g. turn-up in a week with no negative price),
    ``simple_mean_mw`` and ``value_at_day_ahead_gbp_per_week`` (every
    deliverable MWh at its weight price: illustrative, not a P&L, not a bid,
    not comparable with the trading net; 0 when the weights sum to 0).
    Slots whose hold window runs past the study end are left out.
    """

    world_count, slot_count, _ = _world_slot_cubes(world_slot)
    deliverable = _cube(world_slot, "deliverable_kw", world_count, slot_count)
    price = np.asarray(day_ahead_gbp_per_mwh, dtype=float)
    rows = []
    for cell, (direction, hours) in enumerate(availability.CELLS):
        kw = deliverable[..., cell]
        defined = np.isfinite(kw)
        weight = np.maximum(0.0, price if direction == "turn_down" else -price)
        weight = np.where(defined, weight, 0.0)
        kw0 = np.where(defined, kw, 0.0)
        per_world = {
            "price_weighted_mw": _ratio((kw0 * weight).sum(axis=1), weight.sum(axis=1))
            / _KW_PER_MW,
            "simple_mean_mw": _nanmean(kw, axis=1) / _KW_PER_MW,
            "value_at_day_ahead_gbp_per_week": (
                kw0 * availability.SLOT_HOURS / _KW_PER_MW * weight
            ).sum(axis=1),
        }
        for metric, unit in VALUE_METRICS:
            values = per_world[metric]
            count, (p10, p50, p90) = _stats(values, (0.1, 0.5, 0.9))
            rows.append(
                {
                    "direction": direction,
                    "duration_hours": float(hours),
                    "metric": metric,
                    "unit": unit,
                    "world_count": int(count),
                    "mean": _nanmean(values),
                    "p10": p10,
                    "p50": p50,
                    "p90": p90,
                    "evidence_kind": EVIDENCE_KIND,
                }
            )
    return pd.DataFrame(rows, columns=VALUE_SUMMARY_COLUMNS).astype({"world_count": np.int64})


# --------------------------------------------------------------------------
# §10.5d Settlement file
# --------------------------------------------------------------------------


def settlement_file_name(start_local_date: date, seed: int, world_id: int) -> str:
    """``axle_settlement_<study start date>_seed<seed>_world<id>.csv`` (§10.5d)."""

    return f"axle_settlement_{start_local_date.isoformat()}_seed{seed}_world{world_id}.csv"


def settlement_file(
    slot_keys: pd.DataFrame,
    *,
    units: pd.DataFrame,
    unmanaged_kwh: np.ndarray,
    metered_kwh: np.ndarray,
    deviation: pd.DataFrame,
    weights: np.ndarray,
    customer_share_gbp: np.ndarray,
) -> pd.DataFrame:
    """§10.5d: a per-EV settlement file for one world, ``(N + 1) x slots`` rows.

    Inputs for the one (representative) world: ``slot_keys`` one row per
    study slot with ``slot_index``, ``night_index``, ``interval_start_utc``,
    ``interval_start_london``, ``settlement_date`` and ``settlement_period``
    (``supplier_positions``' keys, so both files name a slot alike);
    ``units`` one row per EV with ``unit_id``, ``cohort_id``,
    ``manufacturer_id``, ``zone_id`` and ``control_group``;
    ``unmanaged_kwh`` and ``metered_kwh`` (slot, EV) each EV's normal-path
    and selected-path home grid import; ``deviation`` that world's
    ``deviation_world_slot`` rows in slot order; ``weights`` (EV, night) the
    §9.3c allocation ``a_{i,n}``; ``customer_share_gbp`` (night,) the ``full``
    strategy's ``customer_revenue_share_gbp`` (negative for the aggregator).

    Each EV's settled volume is its own true reduction ``u - m`` in the
    slots where the fleet settles (``V > 0``), so the EV rows share out only
    the fleet's true reduction ``R = U - M`` (lead review B5).  The baseline
    effect ``E = B - U`` is nobody's work: it is one unallocated fleet row
    per slot, so all rows of a slot sum to ``V = R + E``.  The payment is the
    customer's revenue share of the night, allocated with ``a_{i,n}`` and
    spread over the EV's night in proportion to its positive settled kWh
    (evenly over the night's slots when it has none); control EVs get 0 but
    keep their settled kWh (they are in the metered import).  Rows are
    unit-major (every slot of one meter point together) with the fleet
    rows last.  The EV id stands in for the meter point (MPAN).
    """

    slot_count, vehicle_count = unmanaged_kwh.shape
    settles = deviation["settled_kwh"].to_numpy(dtype=float) > 0.0
    true_reduction = unmanaged_kwh - metered_kwh
    settled = np.where(settles[:, np.newaxis], true_reduction, 0.0)
    night = slot_keys["night_index"].to_numpy(dtype=np.int64)
    payment = np.zeros((slot_count, vehicle_count))
    for n in np.unique(night):
        in_night = night == n
        # The customer receives minus the aggregator's customer-share bucket.
        per_ev = weights[:, n] * -customer_share_gbp[n]
        drivers = np.maximum(settled[in_night], 0.0)
        total = drivers.sum(axis=0)
        spread = np.where(
            total > 0.0, drivers / np.where(total > 0.0, total, 1.0), 1.0 / in_night.sum()
        )
        payment[in_night] = spread * per_ev
    baseline_effect = deviation["baseline_effect_kwh"].to_numpy(dtype=float)

    def per_ev(values: np.ndarray) -> np.ndarray:  # (slot, EV) -> unit-major rows
        return values.T.ravel()

    def fleet_row(value) -> np.ndarray:
        return np.full(slot_count, value, dtype=object if isinstance(value, str) else None)

    def text(column: str) -> np.ndarray:
        values = units[column].to_numpy(dtype=object)
        return np.concatenate([np.repeat(values, slot_count), fleet_row(FLEET_ID)])

    def slot_key(column: str) -> pd.Series:
        rows = np.tile(np.arange(slot_count), vehicle_count + 1)
        return slot_keys[column].take(rows).reset_index(drop=True)

    control = units["control_group"].to_numpy(dtype=bool)
    nan = np.full(slot_count, np.nan)
    frame = pd.DataFrame(
        {
            "meter_point_id": np.concatenate(
                [
                    np.repeat(units["unit_id"].to_numpy(dtype=object), slot_count),
                    fleet_row(BASELINE_EFFECT_ID),
                ]
            ),
            "cohort_id": text("cohort_id"),
            "manufacturer_id": text("manufacturer_id"),
            "zone_id": text("zone_id"),
            "control_group": np.concatenate(
                [np.repeat(control, slot_count), np.zeros(slot_count, dtype=bool)]
            ),
            "slot_index": slot_key("slot_index").astype(np.int64),
            "night_index": slot_key("night_index").astype(np.int64),
            "interval_start_utc": slot_key("interval_start_utc"),
            "interval_start_london": slot_key("interval_start_london"),
            "settlement_date": slot_key("settlement_date").astype(object),
            "settlement_period": slot_key("settlement_period").astype(np.int64),
            "unmanaged_kwh": np.concatenate([per_ev(unmanaged_kwh), nan]),
            "metered_kwh": np.concatenate([per_ev(metered_kwh), nan]),
            "deviation_kwh": np.concatenate([per_ev(true_reduction), baseline_effect]),
            "settled_kwh": np.concatenate(
                [per_ev(settled), np.where(settles, baseline_effect, 0.0)]
            ),
            "day_ahead_gbp_per_mwh": np.tile(
                deviation["day_ahead_gbp_per_mwh"].to_numpy(dtype=float), vehicle_count + 1
            ),
            "payment_gbp": np.concatenate([per_ev(payment), np.zeros(slot_count)]),
            "evidence_kind": EVIDENCE_KIND,
        }
    )
    return frame.loc[:, SETTLEMENT_FILE_COLUMNS]


# --------------------------------------------------------------------------
# §10.5e Charge completion
# --------------------------------------------------------------------------


def completion_counts(
    sessions,
    *,
    slot_count: int,
    target_kwh: np.ndarray,
    expected_end_slot: np.ndarray,
    members: np.ndarray,
    world_count: int,
) -> dict[str, np.ndarray]:
    """Per-world session counts for §10.5e, each (world, group, departure).

    ``sessions`` is ``summaries.find_sessions``' result for one path and
    chunk (per-session ``world``, ``ev``, ``first_slot``, ``end_slot``,
    ``plug_out_kwh``); only sessions that end in the study (``end_slot <
    slot_count``) count.  ``target_kwh`` (EV,) is the preferred target
    stock, ``expected_end_slot`` (slot, EV) the planner's expected departure
    as an exclusive slot (``availability.expected_end_slots``), ``members``
    (EV, group) bool.  Departure index 0 ``all``, 1 ``on_time`` (the
    realised unplug at or after the planner's expected departure seen at
    plug-in), 2 ``early``.

    A session is complete when its stock at the last connected slot is at
    least ``target - NOT_RECOVERED_TOLERANCE_KWH``.  Returns ``sessions``,
    ``complete``, ``incomplete`` and ``shortfall_kwh`` (the sum over
    incomplete sessions of ``target - stock at unplug``, battery kWh).
    """

    ended = sessions.end_slot < slot_count
    world = sessions.world[ended]
    ev = sessions.ev[ended]
    stock = sessions.plug_out_kwh[ended]
    target = target_kwh[ev]
    complete = stock >= target - NOT_RECOVERED_TOLERANCE_KWH
    on_time = sessions.end_slot[ended] >= expected_end_slot[sessions.first_slot[ended], ev]
    shortfall = np.where(complete, 0.0, target - stock)
    group_count = members.shape[1]
    out = {
        name: np.zeros((world_count, group_count, len(DEPARTURES)))
        for name in ("sessions", "complete", "incomplete", "shortfall_kwh")
    }
    for group in range(group_count):
        inside = members[ev, group]
        for index, keep in enumerate((inside, inside & on_time, inside & ~on_time)):
            for name, weight in (
                ("sessions", None),
                ("complete", complete),
                ("incomplete", ~complete),
                ("shortfall_kwh", shortfall),
            ):
                out[name][:, group, index] = np.bincount(
                    world[keep],
                    weights=None if weight is None else weight[keep].astype(float),
                    minlength=world_count,
                )
    return out


def charge_completion_summary(
    counts: Mapping[str, np.ndarray],
    group_ids: Sequence[str],
    ev_counts: Sequence[int],
    path_ids: Sequence[str] = PATH_IDS,
) -> pd.DataFrame:
    """§10.5e from per-world counts (world, group, path, departure) of ``completion_counts``.

    One row per (group, path, departure) in ``group_ids``, ``path_ids``
    (default ``PATH_IDS``) and ``DEPARTURES`` order.  Per world: the
    completed share of the group's sessions (NaN with none) and the mean
    shortfall over incomplete sessions (NaN with none); statistics across
    the worlds with a value.  The paired difference across paths is read
    from the two rows, not computed here.  ``path_ids`` is
    ``(*PATH_IDS, "timed")`` when the run completes the optional timed path
    too (decision 0007, model step 2): a third row per (group, departure),
    same columns, same completion rule, no third-path difference.
    """

    share = _ratio(counts["complete"], counts["sessions"])
    shortfall = _ratio(counts["shortfall_kwh"], counts["incomplete"])
    rows = []
    for g, group_id in enumerate(group_ids):
        for p, path_id in enumerate(path_ids):
            for d, departure in enumerate(DEPARTURES):
                values = share[:, g, p, d]
                count, (p10, p50, p90) = _stats(values, (0.1, 0.5, 0.9))
                gap = shortfall[:, g, p, d]
                rows.append(
                    {
                        "group_id": group_id,
                        "path_id": path_id,
                        "departure": departure,
                        "ev_count": int(ev_counts[g]),
                        "world_count": int(count),
                        "session_count_mean": float(counts["sessions"][:, g, p, d].mean()),
                        "completed_share_mean": _nanmean(values),
                        "completed_share_p10": p10,
                        "completed_share_p50": p50,
                        "completed_share_p90": p90,
                        "shortfall_kwh_mean": _nanmean(gap),
                        "shortfall_kwh_p50": _stats(gap, (0.5,))[1][0],
                        "evidence_kind": EVIDENCE_KIND,
                    }
                )
    return pd.DataFrame(rows, columns=COMPLETION_COLUMNS).astype(
        {"ev_count": np.int64, "world_count": np.int64}
    )


# --------------------------------------------------------------------------
# §10.5f Supplier positions: net shape change and firm figures
# --------------------------------------------------------------------------


def supplier_position_additions(
    positions: pd.DataFrame, deviation: pd.DataFrame, availability_bands: pd.DataFrame | None
) -> pd.DataFrame:
    """``supplier_positions`` with the §10.5f columns inserted after ``settled_mw_p50``.

    From ``deviation_world_slot``: ``unmanaged_mw_p50`` (``U`` in MW) and the
    net shape change ``(M - U)`` in MW per world, then linear P10/P50/P90
    across worlds: negative where charging moved out of the slot, positive
    where it landed, including the rebound after a turn-down window.  From
    ``availability_bands`` (``day_ahead``, ``realised``, 1 h): the ``p10``
    firm turn-down and turn-up MW; NaN when no bands are given.
    """

    ordered = deviation.sort_values(["world_id", "slot_index"], kind="stable")
    slot_count = len(positions)

    def mw(column: str) -> np.ndarray:
        values = ordered[column].to_numpy(dtype=float).reshape(-1, slot_count)
        return values / availability.SLOT_HOURS / _KW_PER_MW

    unmanaged = mw("unmanaged_kwh")
    net_change = mw("metered_kwh") - unmanaged
    added = {"unmanaged_mw_p50": np.quantile(unmanaged, 0.5, axis=0, method="linear")}
    for name, row in zip(
        ("p10", "p50", "p90"),
        np.quantile(net_change, (0.1, 0.5, 0.9), axis=0, method="linear"),
        strict=True,
    ):
        added[f"net_change_mw_{name}"] = row
    for direction in availability.DIRECTIONS:
        column = f"deliverable_{direction}_1h_mw_p10"
        if availability_bands is None:
            added[column] = np.full(slot_count, np.nan)
            continue
        rows = availability_bands.loc[
            availability_bands["statistic"].eq("realised")
            & availability_bands["direction"].eq(direction)
            & availability_bands["duration_hours"].eq(_ONE_HOUR)
        ].sort_values("slot_index")
        added[column] = rows["p10"].to_numpy(dtype=float) / _KW_PER_MW
    frame = positions.drop(columns=POSITION_ADDITION_COLUMNS, errors="ignore")
    at = frame.columns.get_loc("settled_mw_p50") + 1
    for offset, column in enumerate(POSITION_ADDITION_COLUMNS):
        frame.insert(at + offset, column, added[column])
    return frame


# --------------------------------------------------------------------------
# §10.5g-h Firmness by manufacturer and the diversification explainer
# --------------------------------------------------------------------------


def manufacturer_table(
    manufacturer_catalogue: Sequence[tuple[str, str, float]],
    manufacturer_index: np.ndarray,
    response_rate: np.ndarray,
    outage_probability: np.ndarray,
) -> pd.DataFrame:
    """One row per maker: id, label, share (records), EV count, response rate and outage chance.

    ``manufacturer_catalogue`` is ``(manufacturer_id, label, share)`` in
    ``m1``... order (the J1a records); ``manufacturer_index`` (EV,) the
    kernel-side maker column; the rates are ``AvailabilityInputs``' records.
    """

    catalogue = list(manufacturer_catalogue)
    return pd.DataFrame(
        {
            "manufacturer_id": pd.Series([row[0] for row in catalogue], dtype=object),
            "manufacturer_label": pd.Series([row[1] for row in catalogue], dtype=object),
            "share": np.array([row[2] for row in catalogue], dtype=float),
            "ev_count": np.bincount(
                np.asarray(manufacturer_index, dtype=np.int64), minlength=len(catalogue)
            ).astype(np.int64),
            "response_rate": np.asarray(response_rate, dtype=float),
            "outage_probability_per_night": np.asarray(outage_probability, dtype=float),
        }
    )


def firmness_by_manufacturer(
    manufacturer_world: pd.DataFrame, manufacturers: pd.DataFrame
) -> pd.DataFrame:
    """§10.5g: physical firmness per maker, direction and duration (32 rows with four makers).

    From ``availability_manufacturer_world``: per world, firmness is the
    week's ``Σ_n realised_mwh ÷ Σ_n potential_mwh`` (delivered ÷ what the
    maker's sessions could have delivered had every one followed its plan;
    NaN when the potential is 0), and the outage count is the nights the
    maker was out.  This is physical firmness; the ledger's ``firmness``
    (§9.1) is against the traded position and fleet-level, and the two are
    labelled apart.  ``manufacturers`` is ``manufacturer_table``'s frame.
    """

    keys = ["manufacturer_id", "direction", "duration_hours", "world_id"]
    per_world = manufacturer_world.groupby(keys).agg(
        realised=("realised_mwh", "sum"),
        potential=("potential_mwh", "sum"),
        outage_nights=("outage", "sum"),
    )
    rows = []
    for maker in manufacturers.itertuples(index=False):
        for direction, hours in availability.CELLS:
            block = per_world.loc[(maker.manufacturer_id, direction, hours)]
            firmness = _ratio(
                block["realised"].to_numpy(dtype=float), block["potential"].to_numpy(dtype=float)
            )
            outages = block["outage_nights"].to_numpy(dtype=float)
            count, (p10, p50, p90) = _stats(firmness, (0.1, 0.5, 0.9))
            rows.append(
                {
                    "manufacturer_id": maker.manufacturer_id,
                    "manufacturer_label": maker.manufacturer_label,
                    "share": maker.share,
                    "ev_count": maker.ev_count,
                    "response_rate": maker.response_rate,
                    "outage_probability_per_night": maker.outage_probability_per_night,
                    "direction": direction,
                    "duration_hours": float(hours),
                    "world_count": int(count),
                    "firmness_mean": _nanmean(firmness),
                    "firmness_p10": p10,
                    "firmness_p50": p50,
                    "firmness_p90": p90,
                    "outage_nights_mean": float(outages.mean()),
                    "outage_nights_p50": float(np.quantile(outages, 0.5, method="linear")),
                    "evidence_kind": EVIDENCE_KIND,
                }
            )
    return pd.DataFrame(rows, columns=FIRMNESS_COLUMNS).astype(
        {"ev_count": np.int64, "world_count": np.int64}
    )


def manufacturer_summary(
    evening_potential_mw: np.ndarray, manufacturers: pd.DataFrame
) -> pd.DataFrame:
    """§10.5h: one row per maker plus a ``fleet`` row, the diversification explainer.

    ``evening_potential_mw`` (world, maker) is each maker's 1-h turn-down
    potential ``ā`` averaged over the evening window's slots, MW.  A night
    with one maker out removes about ``share x evening potential`` of the
    fleet's figure, so the firm figure depends on the largest slice and on
    the chance any maker is out, ``1 - Π(1 - π_m)`` on the fleet row: with
    equal shares one outage costs ``1/M`` of the fleet, so more, smaller
    makers diversify exactly as more independent EVs do.
    """

    potential = np.asarray(evening_potential_mw, dtype=float)
    shares = manufacturers["share"].to_numpy(dtype=float)
    pi = manufacturers["outage_probability_per_night"].to_numpy(dtype=float)
    largest = float(shares.max())
    frame = manufacturers.loc[
        :,
        [
            "manufacturer_id",
            "manufacturer_label",
            "share",
            "ev_count",
            "response_rate",
            "outage_probability_per_night",
        ],
    ].copy()
    fleet = {
        "manufacturer_id": FLEET_ID,
        "manufacturer_label": "Fleet",
        "share": float(shares.sum()),
        "ev_count": int(manufacturers["ev_count"].sum()),
        "response_rate": np.nan,
        # Makers go out independently, so P(at least one out) = 1 - Π(1 - π).
        "outage_probability_per_night": float(1.0 - np.prod(1.0 - pi)),
    }
    frame = pd.concat([frame, pd.DataFrame([fleet])], ignore_index=True)
    frame["evening_potential_mw_p50"] = np.append(
        np.quantile(potential, 0.5, axis=0, method="linear"),
        np.quantile(potential.sum(axis=1), 0.5, method="linear"),
    )
    frame["largest_share"] = largest
    frame["evidence_kind"] = EVIDENCE_KIND
    return frame.loc[:, MANUFACTURER_SUMMARY_COLUMNS].astype({"ev_count": np.int64})


# --------------------------------------------------------------------------
# Chunk-loop hook
# --------------------------------------------------------------------------


def new_totals(
    availability_totals: dict,
    *,
    representative_world_id: int,
    groups: Mapping[str, np.ndarray],
    path_ids: Sequence[str] = PATH_IDS,
) -> dict:
    """Empty accumulators for the product frames, filled chunk by chunk by ``accumulate``.

    ``availability_totals`` is the run's ``availability.new_totals`` (its
    per-EV constants, maker assignment and product windows are reused, so
    both hooks read one set).  ``groups`` {group_id: (EV,) bool} are the
    fleet then each cohort with EVs; the makers ``m1``... are appended.
    ``path_ids`` is ``PATH_IDS`` by default; pass ``(*PATH_IDS, "timed")`` to
    also complete the optional timed path (decision 0007, model step 2).
    ``accumulate`` and ``charge_completion_summary`` then read
    ``totals["path_ids"]`` rather than taking the list again.
    """

    onehot = availability_totals["maker_onehot"]
    group_ids = [*groups, *(f"m{m + 1}" for m in range(onehot.shape[1]))]
    members = np.column_stack([*groups.values(), onehot.astype(bool)])
    world_count = availability_totals["world_count"]
    shape = (world_count, len(group_ids), len(path_ids), len(DEPARTURES))
    return {
        "availability": availability_totals,
        "representative_world_id": representative_world_id,
        "group_ids": group_ids,
        "members": members,
        "path_ids": tuple(path_ids),
        "completion": {
            name: np.zeros(shape)
            for name in ("sessions", "complete", "incomplete", "shortfall_kwh")
        },
        "unmanaged_kwh": None,
        "metered_kwh": None,
        "evening_potential_kw": np.full((world_count, onehot.shape[1]), np.nan),
    }


def accumulate(
    totals: dict,
    worlds: np.ndarray,
    by_path: Mapping[str, Mapping[str, np.ndarray]],
    sessions_by_path: Mapping[str, object],
) -> None:
    """Add one chunk: completion counts, the representative world's imports, evening potential.

    ``by_path`` holds every path named in ``totals["path_ids"]``
    (``new_totals``) of per-EV arrays (world, slot, EV); ``sessions_by_path``
    each such path's ``summaries.find_sessions`` result for the chunk.
    Nothing per EV is kept except the representative world's home imports,
    which the settlement file needs.  Firm-MW availability itself (the
    evening potential below) stays the selected path only (item 6): it is
    not redefined for the optional timed path.
    """

    shared = totals["availability"]
    slot_count = len(shared["night_index"])
    for p, path in enumerate(totals["path_ids"]):
        counts = completion_counts(
            sessions_by_path[path],
            slot_count=slot_count,
            target_kwh=shared["target_kwh"],
            expected_end_slot=shared["expected_end_slot"],
            members=totals["members"],
            world_count=len(worlds),
        )
        for name, values in counts.items():
            totals["completion"][name][worlds, :, p] = values
    here = np.flatnonzero(worlds == totals["representative_world_id"])
    if len(here):
        totals["unmanaged_kwh"] = by_path["normal"]["home_grid_import_kwh"][here[0]].copy()
        totals["metered_kwh"] = by_path["selected"]["home_grid_import_kwh"][here[0]].copy()
    # The maker's 1-h turn-down potential ā per slot (§10.2a), averaged over
    # the evening window.  ``availability_manufacturer_world`` holds only
    # night sums, so the evening slice is reduced here from the per-EV
    # potential while the chunk is in memory.
    potential = availability.ev_contributions(
        by_path["selected"],
        power_kw=shared["power_kw"],
        target_kwh=shared["target_kwh"],
        efficiency=shared["efficiency"],
        expected_end_slot=shared["expected_end_slot"],
        night_index=shared["night_index"],
        night_decision_slots=shared["decision_slots"],
        blackout=shared["inputs"].blackout,
        duration_slots=availability.DURATION_SLOTS[availability.DURATION_HOURS.index(_ONE_HOUR)],
    )["potential"]["turn_down"]
    evening = shared["windows"][list(availability.PRODUCT_WINDOWS).index("evening")]
    by_maker = potential[:, evening] @ shared["maker_onehot"]  # (chunk, evening slot, maker)
    totals["evening_potential_kw"][worlds] = _nanmean(by_maker, axis=1)


def build_frames(
    totals: dict,
    *,
    frames: Mapping[str, object],
    trading,
    units: pd.DataFrame,
    weights: np.ndarray,
    manufacturer_catalogue: Sequence[tuple[str, str, float]],
    seed: int,
    start_local_date: date,
    day_ahead_decision_local_hour: float,
) -> dict[str, object]:
    """The J5 ``ForecastResult`` fields from filled ``totals`` and the run's other frames.

    ``frames`` holds the summaries built so far (the J2 availability frames
    and ``supplier_positions``); ``trading`` is ``market.run_trading``'s
    result; ``units`` the population; ``weights`` (EV, night) the §9.3c
    allocation for the representative world.  Returns ``product_sheet``,
    ``availability_value_summary``, ``settlement_file``,
    ``settlement_file_name``, ``charge_completion_summary``,
    ``firmness_by_manufacturer`` and ``manufacturer_summary``.
    (``supplier_positions`` gets its §10.5f columns in
    ``summaries._trading_summaries``, so every action result has one schema.)
    """

    shared = totals["availability"]
    inputs = shared["inputs"]
    representative = totals["representative_world_id"]
    world_slot = frames["availability_world_slot"]
    deviation = trading.deviation_world_slot.sort_values(["world_id", "slot_index"])
    world_count = shared["world_count"]
    day_ahead = deviation["day_ahead_gbp_per_mwh"].to_numpy(dtype=float).reshape(world_count, -1)
    makers = manufacturer_table(
        manufacturer_catalogue,
        inputs.manufacturer_index,
        inputs.response_rate,
        inputs.outage_probability,
    )
    ledger = trading.trading_ledger_world
    full = ledger.loc[
        ledger["strategy"].eq("full") & ledger["world_id"].eq(representative)
    ].sort_values("night_index")
    maker_ids = np.array([row[0] for row in manufacturer_catalogue], dtype=object)
    population = pd.DataFrame(
        {
            "unit_id": units["unit_id"].to_numpy(dtype=object),
            "cohort_id": units["cohort_id"].to_numpy(dtype=object),
            "manufacturer_id": maker_ids[np.asarray(inputs.manufacturer_index, dtype=np.int64)],
            "zone_id": units["zone_id"].to_numpy(dtype=object),
            "control_group": (
                units["control_group"].to_numpy(dtype=bool)
                if "control_group" in units
                else np.zeros(len(units), dtype=bool)
            ),
        }
    )
    positions = frames["supplier_positions"]
    return {
        "product_sheet": product_sheet(
            world_slot,
            day_ahead_decision_local_hour=day_ahead_decision_local_hour,
            intraday_decision_local_time=inputs.decision_local_time,
        ),
        "availability_value_summary": availability_value_summary(world_slot, day_ahead),
        "settlement_file": settlement_file(
            positions,
            units=population,
            unmanaged_kwh=totals["unmanaged_kwh"],
            metered_kwh=totals["metered_kwh"],
            deviation=deviation.loc[deviation["world_id"].eq(representative)],
            weights=weights,
            customer_share_gbp=full["customer_revenue_share_gbp"].to_numpy(dtype=float),
        ),
        "settlement_file_name": settlement_file_name(start_local_date, seed, representative),
        "charge_completion_summary": charge_completion_summary(
            totals["completion"],
            totals["group_ids"],
            totals["members"].sum(axis=0),
            totals["path_ids"],
        ),
        "firmness_by_manufacturer": firmness_by_manufacturer(
            frames["availability_manufacturer_world"], makers
        ),
        "manufacturer_summary": manufacturer_summary(
            totals["evening_potential_kw"] / _KW_PER_MW, makers
        ),
    }


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------

_COMPARED_SHEET = (
    ("window_mean_mw_p10", "MW"),
    ("window_mean_mw_p50", "MW"),
    ("firm_share", "fraction"),
    ("firm_share_p05", "fraction"),
)


def firm_mw_comparison(
    sheet_a: pd.DataFrame | None,
    firmness_a: pd.DataFrame | None,
    sheet_b: pd.DataFrame | None,
    firmness_b: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Compare rows for the Firm MW frames of two runs, side by side (None unless both have them).

    Product sheet: the evening and overnight turn-down 1-h rows' window MW
    and firm shares; firmness by maker: the 1-h turn-down ``firmness_p50``.
    Columns ``frame``, ``key``, ``metric``, ``unit``, ``value_a``,
    ``value_b``.  No difference column: a difference of two quantiles is
    not a quantile of the paired difference, and the frames keep no
    per-world values to pair.
    """

    if any(frame is None for frame in (sheet_a, firmness_a, sheet_b, firmness_b)):
        return None

    def sheet_rows(sheet: pd.DataFrame) -> pd.DataFrame:
        return sheet.loc[
            sheet["window"].isin(["evening", "overnight"])
            & sheet["direction"].eq("turn_down")
            & sheet["duration_hours"].eq(_ONE_HOUR)
        ].set_index("window")

    def maker_rows(firmness: pd.DataFrame) -> pd.DataFrame:
        return firmness.loc[
            firmness["direction"].eq("turn_down") & firmness["duration_hours"].eq(_ONE_HOUR)
        ].set_index("manufacturer_id")

    rows = []
    a, b = sheet_rows(sheet_a), sheet_rows(sheet_b)
    for window in a.index:
        for metric, unit in _COMPARED_SHEET:
            rows.append(
                ("product_sheet", window, metric, unit, a.at[window, metric], b.at[window, metric])
            )
    a, b = maker_rows(firmness_a), maker_rows(firmness_b)
    for maker in a.index:
        rows.append(
            (
                "firmness_by_manufacturer",
                maker,
                "firmness_p50",
                "fraction",
                a.at[maker, "firmness_p50"],
                b.at[maker, "firmness_p50"] if maker in b.index else np.nan,
            )
        )
    frame = pd.DataFrame(rows, columns=FIRM_MW_COMPARISON_COLUMNS)
    return frame.astype({"value_a": float, "value_b": float})


__all__ = [
    "PRODUCT_SHEET_COLUMNS",
    "accumulate",
    "availability_value_summary",
    "build_frames",
    "charge_completion_summary",
    "completion_counts",
    "firm_mw_comparison",
    "firmness_by_manufacturer",
    "manufacturer_summary",
    "manufacturer_table",
    "max_event_length_hours",
    "new_totals",
    "product_sheet",
    "settlement_file",
    "settlement_file_name",
    "supplier_position_additions",
]
