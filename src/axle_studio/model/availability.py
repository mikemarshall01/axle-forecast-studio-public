"""Firm-MW availability: deliverable MW per world, its bands and the 17:00 forecast.

What this owns (trading contract v1 §10.2, lane J2; decision 0004 items 59,
60 and 64): how many kW the smart-charged fleet could turn down or turn up,
held for 0.5, 1, 2 or 4 hours, in every study half-hour of every simulated
week, and how firm that figure is.

- ``ev_contributions`` is the per-EV rule of §10.2a: realised ``a``,
  potential ``ā`` (as if every session followed its plan) and plan-based
  ``â`` (what the aggregator can compute at the night's 17:00 decision).
- ``new_totals`` / ``accumulate`` / ``build_frames`` are the chunk-loop hook
  ``summaries.build_summaries`` calls: per-EV arrays arrive ten worlds at a
  time and are reduced to fleet sums, per-maker sums and per-EV across-world
  sums before the next chunk, so nothing per EV is kept per world.
- ``intraday_distribution`` is the whole-night forecast made at 17:00
  (§10.2d): the known part (EVs already plugged in, with random response and
  maker outages) plus the late part (sessions not yet plugged in, read off
  the other simulated weeks), exact over the 16 maker-outage states.
- ``build_frames`` returns ``availability_world_slot``, ``availability_bands``,
  ``availability_manufacturer_world``, ``firm_share_by_fleet_size`` and
  ``world_nights`` (§10.1f, §10.2e).
- ``validate_blackout_windows`` checks the blackout-window setting once per
  run (§10.0 step 1, §10.5b; lane J5 by lead ruling §10.11).

Conventions: kW are fleet or per-EV power sustained over the window; kWh are
grid-side energy per half-hour slot.  ``p10`` is ``numpy.quantile(..., 0.1)``,
the value exceeded in 90 % of worlds, i.e. the trader's "P90"; ``firm_share``
is ``p10 ÷ p50`` (the §10 two-convention note).  Every value is illustrative
and synthetic: nothing here is a bid, a settlement or Axle cash.  Nothing
here draws random numbers: every figure is a deterministic function of the
run (§10.0 draw order).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft, special

from axle_studio.model import physics

# ``plan_status`` codes of the kernel (§10.1e), defined once in ``physics``;
# named here too because the lane fixtures read them from this module.
PLAN_FOLLOWS = physics.PLAN_FOLLOWS
BASE_NON_RESPONSE = physics.BASE_NON_RESPONSE
MAKER_OUTAGE = physics.MAKER_OUTAGE
CONTROL_OUTAGE_EVENT = physics.CONTROL_OUTAGE_EVENT
CONTROL_GROUP = physics.CONTROL_GROUP

# --- Reporting constants (§10.8: reporting choices, not model assumptions) ---

DIRECTIONS = ("turn_down", "turn_up")
DURATION_HOURS = (0.5, 1.0, 2.0, 4.0)
"""Hold durations: 0.5 h for the product sheet's energy figure, 1, 2 and 4 h from item 59."""
DURATION_SLOTS = tuple(int(2 * hours) for hours in DURATION_HOURS)
CELLS = tuple((direction, hours) for direction in DIRECTIONS for hours in DURATION_HOURS)
"""(direction, duration) cells in frame order; cell index = direction * 4 + duration index."""
INTRADAY_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
INTRADAY_GRID_BINS = 512
"""Bins per (slot, direction, duration) grid of the intraday CDF (§10.2d)."""
PRODUCT_WINDOWS = {
    "evening": ("17:00", "21:00"),
    "overnight": ("21:00", "06:00"),
    "morning": ("06:00", "12:00"),
}
"""London wall-clock product windows in noon-to-noon order (§10.5a); 12:00-17:00 is in none."""
EVENING = list(PRODUCT_WINDOWS).index("evening")
MAX_EVENT_SHARE = 0.25
"""Product sheet's longest-event rule (§10.5a), read by the J5 product frames."""
FLEET_SIZE_GRID = (100, 300, 1000, 3000)
EVIDENCE_KIND = "illustrative_synthetic"


SLOT_HOURS = 0.5
NEED_TOLERANCE_KWH = 1e-9
"""Float slack (kWh) on need comparisons: need - import is a difference of stocks."""
_ROW_BLOCK = 256
"""Rows per block inside ``intraday_distribution``; bounds its arrays to tens of MB."""


@dataclass(frozen=True)
class AvailabilityInputs:
    """Run-level inputs the availability frames need beyond the per-EV kernel arrays.

    Built by ``forecast`` at integration from J1a's draws and records.
    Shapes: ``W`` worlds, ``N`` EVs, ``M`` makers, study nights, sampled
    London dates (``settings.sampled_day_count``, warm-up dates first).
    """

    manufacturer_index: np.ndarray  # (N,) int, 0 ... M-1, the kernel-side maker column
    response_rate: np.ndarray  # (M,) fraction of a maker's sessions that follow their plan
    outage_probability: np.ndarray  # (M,) chance per night that a maker's control is out
    outage: np.ndarray  # (W, night, M) bool, the sampled outages
    blackout: np.ndarray  # (study slot,) bool, half-hours promised untouched (§10.5b)
    decision_local_time: str  # intraday decision, London "HH:MM" (default record "17:00")
    warmup_days: int  # study night n's evening is sampled date warmup_days + n
    plug_in_week_factor: np.ndarray  # (W,) y_w
    plug_in_day_factor: np.ndarray  # (W, date) z_{w,d}
    plug_in_skip_share: np.ndarray  # (W, date) q_{w,d}
    skip_logit_shift: np.ndarray  # (date,) holiday shift h_d
    holiday: np.ndarray  # (date,) bool
    temperature_c: np.ndarray  # (W, date) sampled evening temperature, degrees C
    solar_clearness: (
        np.ndarray | None
    )  # (W, date) daily clearness, or None if the generator has none


# --------------------------------------------------------------------------
# Slot helpers
# --------------------------------------------------------------------------


def decision_slots(study_slots: pd.DataFrame, decision_local_time: str) -> np.ndarray:
    """Study slot of each night's intraday decision ``s_n`` (§10.2d), shape (night,).

    ``decision_local_time`` is a London "HH:MM" half-hour; the decision slot
    of night ``n`` is the slot that starts at that wall-clock time on the
    night's evening.  Raises ``ValueError`` if a night has no such slot.
    """

    nights = study_slots["night_index"].to_numpy()
    labels = study_slots["local_time_label"].to_numpy()
    slots = []
    for night in range(int(nights.max()) + 1):
        match = np.flatnonzero((nights == night) & (labels == decision_local_time))
        if len(match) == 0:
            raise ValueError(f"night {night} has no {decision_local_time} slot")
        slots.append(match[0])
    return np.asarray(slots, dtype=np.int64)


BLACKOUT_COLUMNS = ("start_local_time", "duration_minutes")
"""Editable columns of the ``availability.blackout_windows`` setting (§10.5b)."""
_BLACKOUT_TIME = re.compile(r"(?:[01]\d|2[0-3]):(?:00|30)")
_BLACKOUT_MAX_MINUTES = 720


def validate_blackout_windows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the typed blackout table or raise ``ValueError`` naming the row and rule (§10.5b).

    ``frame`` has one row per window: ``start_local_time`` (London wall clock
    "HH:MM" on a half-hour) and ``duration_minutes`` (a multiple of 30,
    30-720).  A window may wrap midnight.  The result adds
    ``end_local_time`` ("HH:MM", start plus duration on the wall clock) and
    ``slot_labels`` (tuple of the "HH:MM" half-hours the window covers), the
    labels ``study_slots.blackout`` is marked from; an empty table is valid
    (no blackout).  Meaning: half-hours in which the aggregator promises not
    to move charging in either direction.

    Two windows may not share a half-hour.  Checking the covered wall-clock
    labels is the same as checking in the noon-to-noon order the contract
    names, because a window of at most 12 hours covers each label at most
    once however it is laid out on the night.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("blackout windows must be a DataFrame")
    missing = [column for column in BLACKOUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"blackout windows table is missing columns {missing}")
    table = frame.loc[:, list(BLACKOUT_COLUMNS)].reset_index(drop=True)
    starts, durations, ends, labels = [], [], [], []
    owner: dict[str, int] = {}
    for row, (start, duration) in enumerate(table.itertuples(index=False, name=None)):
        name = f"blackout window row {row}"
        if not isinstance(start, str) or not _BLACKOUT_TIME.fullmatch(start):
            raise ValueError(f'{name}, start_local_time: must be "HH:MM" on a half-hour')
        if isinstance(duration, (bool, np.bool_)) or not isinstance(
            duration, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{name}, duration_minutes: must be a number of minutes")
        if (
            not np.isfinite(duration)
            or duration % 30
            or not 30 <= duration <= _BLACKOUT_MAX_MINUTES
        ):
            raise ValueError(
                f"{name}, duration_minutes: must be a multiple of 30 between 30 and 720"
            )
        minutes = int(start[:2]) * 60 + int(start[3:])
        covered = tuple(
            f"{(m % 1440) // 60:02d}:{m % 60:02d}"
            for m in range(minutes, minutes + int(duration), 30)
        )
        for label in covered:
            if label in owner:
                raise ValueError(f"{name}: overlaps blackout window row {owner[label]} at {label}")
            owner[label] = row
        end = (minutes + int(duration)) % 1440
        starts.append(start)
        durations.append(int(duration))
        ends.append(f"{end // 60:02d}:{end % 60:02d}")
        labels.append(covered)
    return pd.DataFrame(
        {
            "start_local_time": pd.Series(starts, dtype=object),
            "duration_minutes": pd.Series(durations, dtype=np.int64),
            "end_local_time": pd.Series(ends, dtype=object),
            "slot_labels": pd.Series(labels, dtype=object),
        }
    )


def expected_end_slots(hours_to_departure: np.ndarray) -> np.ndarray:
    """Planner's deadline ``Ê`` as an exclusive study-slot index, shape (slot, EV).

    ``hours_to_departure`` (slot, EV) is ``summaries.flexibility_inputs``'s
    time from each slot start to the next expected departure (the planner's
    rule).  Only whole slots before the departure can charge, so the slot
    count is floored, exactly as ``physics`` sizes a plan window; the index is
    clipped to the study end.
    """

    slot_count = hours_to_departure.shape[0]
    start = np.arange(slot_count)[:, np.newaxis]
    return np.minimum(
        start + np.floor(hours_to_departure / SLOT_HOURS).astype(np.int64), slot_count
    )


def window_overlaps(mask: np.ndarray, duration_slots: int) -> np.ndarray:
    """(slot,) True where the window ``t ... t + D - 1`` touches a True slot of ``mask``."""

    padded = np.concatenate([mask, np.zeros(duration_slots - 1, dtype=bool)])
    return sliding_window_view(padded, duration_slots).any(axis=-1)


def _night_end_slots(night_index: np.ndarray) -> np.ndarray:
    """(slot,) exclusive index of the last slot of each slot's night."""

    last = np.flatnonzero(np.diff(night_index, append=night_index[-1] + 1)) + 1
    return last[np.searchsorted(last, np.arange(len(night_index)), side="right")]


def _run_end(connected: np.ndarray) -> np.ndarray:
    """(W, slot, EV) exclusive end slot of the connected run that contains each slot.

    Only meaningful where ``connected`` is True: a slot's run ends at the
    first run end at or after it, found by a running minimum from the right.
    """

    slot_count = connected.shape[1]
    after = np.zeros_like(connected)
    after[:, :-1] = connected[:, 1:]
    slot = np.arange(slot_count)[np.newaxis, :, np.newaxis]
    ends = np.where(connected & ~after, slot + 1, slot_count + 1)
    return np.minimum.accumulate(ends[:, ::-1], axis=1)[:, ::-1]


def _window_stats(values: np.ndarray, duration_slots: int):
    """Window min, max and sum over slots ``t ... t + D - 1`` on axis 1; NaN past the end."""

    shape = values.shape
    low, high, total = (np.full(shape, np.nan) for _ in range(3))
    count = shape[1] - duration_slots + 1
    if count > 0:
        view = sliding_window_view(values, duration_slots, axis=1)
        low[:, :count] = view.min(axis=-1)
        high[:, :count] = view.max(axis=-1)
        total[:, :count] = view.sum(axis=-1)
    return low, high, total


# --------------------------------------------------------------------------
# Per-EV contributions (§10.2a)
# --------------------------------------------------------------------------


def _turn_down_up(import_kwh, need_kwh, deadline, eligible, cap_kwh, duration_slots):
    """The §10.2a formulas for one variant: (turn-down kW, turn-up kW), each (W, slot, EV).

    ``import_kwh`` is the per-slot grid import the variant assumes, ``need_kwh``
    the grid kWh still needed at the opening of each window, ``deadline`` the
    exclusive slot by which the EV must reach target, ``eligible`` whether it
    is plugged through the window and takes instructions, ``cap_kwh`` (EV,)
    the most one slot can import.
    """

    low, high, total = _window_stats(import_kwh, duration_slots)
    window_end = np.arange(import_kwh.shape[1])[np.newaxis, :, np.newaxis] + duration_slots
    after = np.maximum(deadline - window_end, 0)
    # Turn-down pauses the charging the plan does in every slot of the window;
    # the EV's own minimum over the window, so fleet sums stay sums of
    # per-EV terms.  Only if the need still fits in the slots left after the
    # window, else the pause would cost the customer charge.
    fits = need_kwh <= cap_kwh * after + NEED_TOLERANCE_KWH
    turn_down = np.where(eligible & fits, low / SLOT_HOURS, 0.0)
    # Turn-up: per-slot headroom or spare need spread over the window,
    # whichever is smaller.  Spare need counts to the preferred target only,
    # never the physical ceiling (results-v2 §1, O9).
    # Spare below the float slack is rounding in need - import, not energy.
    spare = np.minimum(cap_kwh - high, (need_kwh - total) / duration_slots)
    turn_up = np.where(eligible & (spare > NEED_TOLERANCE_KWH), spare / SLOT_HOURS, 0.0)
    return turn_down, turn_up


def ev_contributions(
    unit: dict[str, np.ndarray],
    *,
    power_kw: np.ndarray,
    target_kwh: np.ndarray,
    efficiency: float,
    expected_end_slot: np.ndarray,
    night_index: np.ndarray,
    night_decision_slots: np.ndarray,
    blackout: np.ndarray,
    duration_slots: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-EV deliverable kW for windows of ``duration_slots`` half-hours (§10.2a).

    ``unit`` holds one chunk's selected-path arrays of shape (W, slot, EV):
    ``home_grid_import_kwh`` (``m``), ``opening_battery_kwh``, ``connected``
    (bool, plugged in for the whole slot), and J1b's ``planned_home_import_kwh``
    (``g``), ``plan_remaining_need_kwh`` (``n``), ``decision_plan_kwh`` and
    ``plan_status`` (§10.1e codes).  ``power_kw`` and ``target_kwh`` are per EV
    (home charger kW, preferred-target battery kWh), ``efficiency`` the home
    charge efficiency, ``expected_end_slot`` (slot, EV) the planner's deadline
    (``expected_end_slots``), ``night_index`` (slot,) each slot's session
    night, ``night_decision_slots`` (night,) the decision slot ``s_n`` and
    ``blackout`` (slot,) the blackout mask.

    Returns ``{"realised"|"potential"|"plan_based": {"turn_down"|"turn_up":
    (W, slot, EV) kW}}``: kW the EV can hold through the window that starts
    at each slot.  NaN where the window runs past the study end; 0 in any
    window that touches a blackout slot.  ``plan_based`` (``â``) is 0 for
    slots before the night's decision slot and for EVs not plugged in at it.
    """

    connected = unit["connected"]
    status = unit["plan_status"]
    slot_count = connected.shape[1]
    slot = np.arange(slot_count)
    cap_kwh = power_kw * SLOT_HOURS
    run_end = _run_end(connected)
    # Plugged through: connected for the whole of every slot in the window.
    through = connected & (run_end >= (slot + duration_slots)[np.newaxis, :, np.newaxis])
    # Realised deadline: the planner's expected departure or the realised
    # unplug, whichever comes first (lead review B2).
    deadline = np.minimum(expected_end_slot[np.newaxis], run_end)
    need = np.maximum(target_kwh - unit["opening_battery_kwh"], 0.0) / efficiency
    realised = _turn_down_up(
        unit["home_grid_import_kwh"],
        need,
        deadline,
        through & (status == PLAN_FOLLOWS),
        cap_kwh,
        duration_slots,
    )
    # Potential (B6): the plan in force and its remaining need, as if every
    # maker were online and every session responded.  A non-responder's
    # realised import is full power, which is not what the aggregator could
    # have dispatched, so the plan stands in for it.  Control EVs are never
    # flexed, so they have no potential either.
    potential = _turn_down_up(
        unit["planned_home_import_kwh"],
        unit["plan_remaining_need_kwh"],
        deadline,
        through & (status != CONTROL_GROUP),
        cap_kwh,
        duration_slots,
    )
    # Plan-based (the known part at 17:00): the plan in force at s_n and its
    # remaining need then, less what that plan charged from s_n to t - 1.  It
    # reads the kernel's snapshot at s_n, so a replan after the decision never
    # changes it (R1: no information published after the decision instant).
    decision = night_decision_slots[night_index]
    plan = unit["decision_plan_kwh"]
    charged = np.zeros(plan.shape[:1] + (slot_count + 1,) + plan.shape[2:])
    np.cumsum(plan, axis=1, out=charged[:, 1:])
    need_hat = np.maximum(
        unit["plan_remaining_need_kwh"][:, decision] - (charged[:, slot] - charged[:, decision]),
        0.0,
    )
    # Deadline under the plan: the expected departure seen at s_n, never past
    # the night's end (the plan at s_n is night n's plan).
    end_hat = np.minimum(expected_end_slot[decision], _night_end_slots(night_index)[:, np.newaxis])
    plugged_at_decision = connected[:, decision] & (status[:, decision] != CONTROL_GROUP)
    eligible_hat = (
        plugged_at_decision
        & (slot >= decision)[np.newaxis, :, np.newaxis]
        & ((slot + duration_slots)[:, np.newaxis] <= end_hat)[np.newaxis]
    )
    plan_based = _turn_down_up(
        plan, need_hat, end_hat[np.newaxis], eligible_hat, cap_kwh, duration_slots
    )

    past_end = slot + duration_slots > slot_count
    blocked = window_overlaps(np.asarray(blackout, dtype=bool), duration_slots)
    out: dict[str, dict[str, np.ndarray]] = {}
    for variant, pair in (
        ("realised", realised),
        ("potential", potential),
        ("plan_based", plan_based),
    ):
        out[variant] = {}
        for direction, values in zip(DIRECTIONS, pair, strict=True):
            values[:, blocked] = 0.0
            values[:, past_end] = np.nan
            out[variant][direction] = values
    return out


# --------------------------------------------------------------------------
# Chunk-loop accumulation
# --------------------------------------------------------------------------


def _window_slot_masks(study_slots: pd.DataFrame) -> np.ndarray:
    """(window, slot) bool: slots whose London wall-clock label lies in each product window."""

    labels = study_slots["local_time_label"].to_numpy(dtype=object)
    masks = []
    for start, end in PRODUCT_WINDOWS.values():
        if start < end:
            masks.append((labels >= start) & (labels < end))
        else:  # wraps midnight
            masks.append((labels >= start) | (labels < end))
    return np.asarray(masks, dtype=bool)


def _fleet_sizes(vehicle_count: int) -> np.ndarray:
    return np.asarray(
        sorted({size for size in FLEET_SIZE_GRID if size <= vehicle_count} | {vehicle_count}),
        dtype=np.int64,
    )


def new_totals(
    inputs: AvailabilityInputs, study_slots: pd.DataFrame, flex, world_count: int
) -> dict:
    """Empty accumulators for one run, filled chunk by chunk by ``accumulate``.

    ``flex`` is the run's ``summaries.FlexibilityInputs`` (per-EV home power
    kW, preferred-target kWh, efficiency and hours to the next expected
    departure).  Sizes at 100 worlds x 1,000 EVs: about 40 MB of per-world
    fleet and per-maker sums and about 65 MB of per-EV across-world sums,
    freed after ``build_frames``.
    """

    slot_count = len(study_slots)
    vehicle_count = len(flex.power_kw)
    maker_count = len(inputs.response_rate)
    cells = len(CELLS)
    night_index = study_slots["night_index"].to_numpy(dtype=np.int64)
    night_count = int(night_index.max()) + 1
    windows = _window_slot_masks(study_slots)
    sizes = _fleet_sizes(vehicle_count)

    def fleet(*shape):
        return np.full((world_count, *shape), np.nan)

    return {
        "inputs": inputs,
        "world_count": world_count,
        "vehicle_count": vehicle_count,
        "power_kw": np.asarray(flex.power_kw, dtype=float),
        "target_kwh": np.asarray(flex.target_kwh, dtype=float),
        "efficiency": float(flex.efficiency),
        "expected_end_slot": expected_end_slots(np.asarray(flex.hours_to_departure, dtype=float)),
        "night_index": night_index,
        "night_count": night_count,
        "decision_slots": decision_slots(study_slots, inputs.decision_local_time),
        "maker_onehot": np.eye(maker_count)[np.asarray(inputs.manufacturer_index, dtype=np.int64)],
        "windows": windows,
        "fleet_sizes": sizes,
        # Per world, slot and cell: A, A^K, A^L, Ā, eligible power (kW).
        "deliverable": fleet(slot_count, cells),
        "known": fleet(slot_count, cells),
        "late": fleet(slot_count, cells),
        "potential": fleet(slot_count, cells),
        "eligible_power": fleet(slot_count, cells),
        # Intraday inputs per maker: S_m, Q_m (plan-based sums) and L_{w,m}.
        "known_sum": fleet(slot_count, cells, maker_count),
        "known_sum_sq": fleet(slot_count, cells, maker_count),
        "late_restored": fleet(slot_count, cells, maker_count),
        "intraday_eligible_count": np.zeros((world_count, slot_count, cells), dtype=np.int64),
        # Per world, night, maker and cell (kW summed over the night's slots).
        "maker_realised": fleet(night_count, maker_count, cells),
        "maker_potential": fleet(night_count, maker_count, cells),
        "plugged_at_decision": np.zeros((world_count, night_count), dtype=np.int64),
        # Per-EV across-world sums of (x - x0), x0 the first world's value, so a
        # value that never varies gives exactly zero variance (§10.2c N_c).
        "ev_shift": np.full((slot_count, vehicle_count, cells), np.nan),
        "ev_shift_set": np.zeros(cells, dtype=bool),
        "ev_sum": np.zeros((slot_count, vehicle_count, cells)),
        "ev_sum_sq": np.zeros((slot_count, vehicle_count, cells)),
        # The same for per-EV product-window means (§10.2e fleet-size frame).
        "window_ev_shift": np.full((len(windows), vehicle_count, cells), np.nan),
        "window_ev_shift_set": np.zeros(cells, dtype=bool),
        "window_ev_sum": np.zeros((len(windows), vehicle_count, cells)),
        "window_ev_sum_sq": np.zeros((len(windows), vehicle_count, cells)),
        "subfleet": fleet(len(windows), len(sizes), cells),
        # Each EV's 1-h evening-window mean per world and direction, kept for
        # the household frame (household contract v1 §4 hunk 3), so the
        # contribution is not computed a second time there.
        "evening_ev_kw": np.full((world_count, vehicle_count, len(DIRECTIONS)), np.nan),
    }


def accumulate(totals: dict, worlds: np.ndarray, unit: dict[str, np.ndarray]) -> None:
    """Add one chunk of worlds to ``totals``.

    ``worlds`` (chunk,) are the run's world ids of the chunk; ``unit`` is the
    selected path's per-EV arrays (see ``ev_contributions``).  Everything per
    EV is reduced here and dropped with the chunk.
    """

    inputs = totals["inputs"]
    onehot = totals["maker_onehot"]
    decision = totals["decision_slots"][totals["night_index"]]
    connected = unit["connected"]
    status = unit["plan_status"]
    # Known/late split of the realised deliverable: P_{w,n} is the set of EVs
    # connected for the whole decision slot of each slot's night.
    in_decision_set = connected[:, decision]
    late_follows = ~in_decision_set & (status == PLAN_FOLLOWS)
    late_out = ~in_decision_set & (status == MAKER_OUTAGE)
    night_matrix = (
        totals["night_index"][np.newaxis, :] == np.arange(totals["night_count"])[:, np.newaxis]
    ).astype(float)
    totals["plugged_at_decision"][worlds] = connected[:, totals["decision_slots"]].sum(axis=2)
    windows = totals["windows"]
    sizes = totals["fleet_sizes"]

    for duration_index, duration_slots in enumerate(DURATION_SLOTS):
        per_ev = ev_contributions(
            unit,
            power_kw=totals["power_kw"],
            target_kwh=totals["target_kwh"],
            efficiency=totals["efficiency"],
            expected_end_slot=totals["expected_end_slot"],
            night_index=totals["night_index"],
            night_decision_slots=totals["decision_slots"],
            blackout=inputs.blackout,
            duration_slots=duration_slots,
        )
        for direction_index, direction in enumerate(DIRECTIONS):
            cell = direction_index * len(DURATION_SLOTS) + duration_index
            # NaN (window past the end) propagates through every sum below, so
            # fleet values are NaN exactly where the window is undefined.
            realised = per_ev["realised"][direction]
            potential = per_ev["potential"][direction]
            plan_based = per_ev["plan_based"][direction]
            totals["deliverable"][worlds, :, cell] = realised.sum(axis=2)
            totals["known"][worlds, :, cell] = (realised * in_decision_set).sum(axis=2)
            totals["late"][worlds, :, cell] = (realised * ~in_decision_set).sum(axis=2)
            totals["potential"][worlds, :, cell] = potential.sum(axis=2)
            totals["eligible_power"][worlds, :, cell] = np.where(
                np.isnan(realised), np.nan, (realised > 0) * totals["power_kw"]
            ).sum(axis=2)
            totals["known_sum"][worlds, :, cell] = plan_based @ onehot
            totals["known_sum_sq"][worlds, :, cell] = plan_based**2 @ onehot
            totals["intraday_eligible_count"][worlds, :, cell] = (plan_based > 0).sum(axis=2)
            # Late part per maker with tonight's outage slice restored (Q2):
            # a late session of an out maker counts at its potential ā, which
            # is exact because code 2 goes only to sessions that would
            # otherwise have followed (§10.1e).
            restored = realised * late_follows + potential * late_out
            totals["late_restored"][worlds, :, cell] = restored @ onehot
            by_maker = np.nan_to_num(realised) @ onehot
            totals["maker_realised"][worlds, ..., cell] = np.einsum(
                "wtm,nt->wnm", by_maker, night_matrix
            )
            by_maker = np.nan_to_num(potential) @ onehot
            totals["maker_potential"][worlds, ..., cell] = np.einsum(
                "wtm,nt->wnm", by_maker, night_matrix
            )
            _add_ev_moments(totals, "ev", realised.transpose(1, 2, 0), cell)
            # Per-EV product-window means per world (NaN slots left out, O5).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # a window with no defined slot
                window_means = np.stack(
                    [np.nanmean(realised[:, mask], axis=1) for mask in windows], axis=1
                )  # (chunk, window, EV)
            _add_ev_moments(totals, "window_ev", window_means.transpose(1, 2, 0), cell)
            if duration_slots == 2:
                totals["evening_ev_kw"][worlds, :, direction_index] = window_means[:, EVENING, :]
            # First-n sub-fleets: cohort, zone, control and maker assignments
            # are permutations, so the first n units are a random sub-fleet
            # with the same shared factors (§10.2e).
            totals["subfleet"][worlds, :, :, cell] = np.cumsum(window_means, axis=2)[..., sizes - 1]


def _add_ev_moments(totals: dict, prefix: str, values: np.ndarray, cell: int) -> None:
    """Add per-EV values (..., EV, chunk world) to the shifted across-world sums of ``prefix``."""

    shift = totals[f"{prefix}_shift"]
    shifted = totals[f"{prefix}_shift_set"]
    if not shifted[cell]:
        # Any fixed reference works; the first world seen is the simplest.
        shift[..., cell] = values[..., 0]
        shifted[cell] = True
    deviation = values - shift[..., cell, np.newaxis]
    totals[f"{prefix}_sum"][..., cell] += deviation.sum(axis=-1)
    totals[f"{prefix}_sum_sq"][..., cell] += (deviation**2).sum(axis=-1)


def _ev_variance(total: np.ndarray, total_sq: np.ndarray, world_count: int) -> np.ndarray:
    """Per-EV across-world variance (ddof 1) from shifted sums; NaN with fewer than two worlds."""

    if world_count < 2:
        return np.full(total.shape, np.nan)
    return np.maximum(total_sq - total**2 / world_count, 0.0) / (world_count - 1)


# --------------------------------------------------------------------------
# Correlation diagnostics (§10.2c)
# --------------------------------------------------------------------------


def correlation_diagnostics(
    fleet_values: np.ndarray,
    ev_variance_sum: np.ndarray,
    ev_sd_sum: np.ndarray,
    contributing: np.ndarray,
) -> dict[str, np.ndarray]:
    """Implied equicorrelation and effective fleet size from fleet and per-EV variances.

    ``fleet_values`` (world, ...) are the per-world fleet values; the other
    arguments (...) are ``Σ_i v_i``, ``Σ_i √v_i`` and ``N_c = #{v_i > 0}`` of
    the per-EV across-world variances ``v_i``.  Returns ``sd`` (ddof 1),
    ``implied_rho`` (unclipped; NaN with fewer than two contributing EVs,
    where there is no pair to correlate) and ``n_eff`` (``N_c / (1 + (N_c -
    1) max(ρ, 0))``, NaN where ρ is).

    ``implied_rho`` is the one correlation that, shared by every pair of EVs,
    reproduces the fleet variance: ``Var(A) = Σv + ρ[(Σ√v)² - Σv]``.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fleet_variance = np.var(fleet_values, axis=0, ddof=1)
        denominator = ev_sd_sum**2 - ev_variance_sum
        # With one varying EV the denominator is float dust (±1e-17), which
        # would store nonsense like ρ = -3; require a pair and a denominator
        # clear of rounding relative to Σv.
        defined = (contributing >= 2) & (denominator > 1e-9 * ev_variance_sum)
        rho = np.where(
            defined,
            (fleet_variance - ev_variance_sum) / np.where(defined, denominator, 1.0),
            np.nan,
        )
        n_eff = contributing / (1.0 + (contributing - 1) * np.maximum(rho, 0.0))
    return {
        "sd": np.sqrt(fleet_variance),
        "implied_rho": rho,
        "n_eff": np.where(np.isnan(rho), np.nan, n_eff),
    }


def realised_diagnostics(totals: dict) -> dict[str, np.ndarray]:
    """§10.2c diagnostics of the realised deliverable per (slot, cell).

    Returns ``sd``, ``implied_rho``, ``n_eff`` (float, NaN where undefined)
    and ``contributing_ev_count`` (int64, EVs whose value varies across
    worlds at that slot), each of shape (slot, cell).
    """

    variance = _ev_variance(totals["ev_sum"], totals["ev_sum_sq"], totals["world_count"])
    contributing = (variance > 0).sum(axis=1)
    out = correlation_diagnostics(
        totals["deliverable"],
        variance.sum(axis=1),
        np.sqrt(variance).sum(axis=1),
        contributing,
    )
    out["contributing_ev_count"] = contributing.astype(np.int64)
    return out


# --------------------------------------------------------------------------
# Intraday whole-night distribution (§10.2d)
# --------------------------------------------------------------------------


def outage_states(outage_probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All maker-outage states ``o ∈ {0,1}^M`` and their probabilities ``P(o)``.

    Returns ``(out, probability)``: ``out`` (state, M) bool, True where the
    maker is out; states are independent makers, so ``P(o)`` is a product.
    """

    pi = np.asarray(outage_probability, dtype=float)
    maker_count = len(pi)
    out = ((np.arange(2**maker_count)[:, np.newaxis] >> np.arange(maker_count)) & 1).astype(bool)
    return out, np.prod(np.where(out, pi, 1.0 - pi), axis=1)


def _lattice_quantiles(masses: np.ndarray, levels, step: np.ndarray) -> np.ndarray:
    """Quantiles of lattice distributions: masses (row, point) at values point x step (row,).

    Values were rounded to the nearest point, so the mass at point j stands
    for ``((j - ½) step, (j + ½) step]`` and is spread evenly over that bin;
    a quantile is where the CDF first reaches the level, linearly
    interpolated within the bin, floored at 0 (point 0's bin reaches below
    0 only by rounding).  Centring the bin on the point avoids the
    half-bin downward bias of spreading it over the bin below.
    """

    masses = np.maximum(masses, 0.0)
    masses = masses / masses.sum(axis=-1, keepdims=True)
    cumulative = np.cumsum(masses, axis=-1)
    rows = np.arange(len(masses))
    out = np.empty((len(masses), len(levels)))
    for k, level in enumerate(levels):
        # The small slack makes a CDF that reaches the level exactly (up to
        # float rounding of the cumulative sum) stop there, as ``numpy``'s
        # inverted-CDF quantile does.
        point = np.minimum((cumulative < level - 1e-12).sum(axis=-1), masses.shape[1] - 1)
        below = np.where(point > 0, cumulative[rows, point - 1], 0.0)
        mass = masses[rows, point]
        fraction = np.where(mass > 0, (level - below) / np.where(mass > 0, mass, 1.0), 1.0)
        out[:, k] = np.maximum((point - 0.5 + np.clip(fraction, 0.0, 1.0)) * step, 0.0)
    return out


def intraday_distribution(
    known_sum: np.ndarray,
    known_sum_sq: np.ndarray,
    response_rate: np.ndarray,
    outage_probability: np.ndarray,
    late_by_maker: np.ndarray,
    levels=INTRADAY_LEVELS,
    x_max: np.ndarray | float | None = None,
    bin_count: int = INTRADAY_GRID_BINS,
) -> dict[str, np.ndarray]:
    """The 17:00 forecast of the whole-fleet deliverable ``A_τ = K + L`` (§10.2d).

    Inputs, vectorised over any leading axes ``...``: ``known_sum`` and
    ``known_sum_sq`` (..., M) are ``S_m = Σ â_i`` and ``Q_m = Σ â_i²`` (kW,
    kW²) over the EVs plugged in at the decision; ``response_rate`` ``r_m``
    and ``outage_probability`` ``π_m`` (M,); ``late_by_maker`` (..., K, M) the
    per-maker late sums ``L_{w',m}`` of the K other worlds (outage slices
    restored); ``levels`` the quantile levels; ``x_max`` (...) the grid top,
    default the largest ``Σ S + Σ L`` in each row.

    Known part: whether each plugged-in session responds to a call is an
    independent Bernoulli(``r_m``) and whether its maker is out tonight a
    Bernoulli(``π_m``) shared by the maker's EVs.  Exactly over the 2^M
    outage states, and as a normal within a state (a weighted sum of
    independent Bernoullis; rough below a few dozen EVs per maker), censored
    to the state's physical range ``[0, Σ_online S_m]``.  Late part: uniform
    over the K other worlds' sums of the online makers' ``L``.  The outage
    state is shared by the two parts (tonight's outage hits the late sessions
    too), so the sum is mixed over states after convolving within each.

    Numerics: each part is put on a lattice of ``bin_count + 1`` points
    ``0, h, ..., x_max`` (``h = x_max / bin_count``, values rounded to the
    nearest point) and the per-state convolutions are done by FFT, summed in
    frequency space with the ``P(o)`` weights (one inverse transform per
    row).  Quantiles are read by ``_lattice_quantiles``; precision is about
    one bin.  Means are exact.

    Returns ``mean``, ``known_mean``, ``late_mean`` (...) and ``quantiles``,
    ``known_quantiles``, ``late_quantiles`` (..., level), in kW; all NaN when
    K is 0 (no other world to read the late part from).
    """

    known_sum = np.asarray(known_sum, dtype=float)
    lead = known_sum.shape[:-1]
    maker_count = known_sum.shape[-1]
    s = known_sum.reshape(-1, maker_count)
    late = np.asarray(late_by_maker, dtype=float)
    late = late.reshape(len(s), late.shape[-2], maker_count)
    q = np.asarray(known_sum_sq, dtype=float).reshape(-1, maker_count)
    r = np.asarray(response_rate, dtype=float)
    pi = np.asarray(outage_probability, dtype=float)
    rows, sample_count = len(s), late.shape[1]
    levels = tuple(levels)

    known_mean = s @ ((1.0 - pi) * r)
    if sample_count == 0:
        nan = np.full(lead, np.nan)
        nan_q = np.full((*lead, len(levels)), np.nan)
        return {
            "mean": nan,
            "known_mean": known_mean.reshape(lead),
            "late_mean": nan,
            "quantiles": nan_q,
            "known_quantiles": nan_q,
            "late_quantiles": nan_q,
        }
    late_mean = late.mean(axis=1) @ (1.0 - pi)
    if x_max is None:
        top = s.sum(axis=1) + late.sum(axis=2).max(axis=1)
    else:
        top = np.broadcast_to(np.asarray(x_max, dtype=float), lead).reshape(-1)
    step = np.where(top > 0, top / bin_count, 1.0)

    out_state, p_state = outage_states(pi)
    # States with probability 0 (π = 0 or 1) contribute nothing and are skipped.
    keep = p_state > 0
    online = ~out_state[keep]
    p_state = p_state[keep]
    n_fft = fft.next_fast_len(2 * bin_count + 1, real=True)
    total_q = np.empty((rows, len(levels)))
    known_q = np.empty((rows, len(levels)))
    late_q = np.empty((rows, len(levels)))
    points = bin_count + 1
    for first in range(0, rows, _ROW_BLOCK):
        block = slice(first, min(first + _ROW_BLOCK, rows))
        h = step[block, np.newaxis, np.newaxis]
        # Known part per state: mean, variance and range of Σ_online R_i â_i.
        mu = (s[block] * r) @ online.T
        sd = np.sqrt(np.maximum((q[block] * r * (1.0 - r)) @ online.T, 0.0))
        cap = s[block] @ online.T
        # CDF at the lattice's interior bin edges (j + ½) h; edges above the
        # state's cap are 1, so the censored mass lands on the cap's point.
        edges = (np.arange(bin_count) + 0.5)[np.newaxis, np.newaxis, :] * h
        with np.errstate(divide="ignore", invalid="ignore"):
            cdf = np.where(
                sd[..., np.newaxis] > 0,
                special.ndtr((edges - mu[..., np.newaxis]) / sd[..., np.newaxis]),
                (edges > mu[..., np.newaxis]).astype(float),
            )
        cdf = np.where(edges > cap[..., np.newaxis], 1.0, cdf)
        known = np.diff(cdf, prepend=0.0, append=1.0, axis=-1)  # (row, state, point)
        # Late part per state: histogram of the other worlds' online sums on
        # the same lattice (each world weight 1/K).
        values = late[block] @ online.T  # (row, K, state)
        index = np.clip(np.floor(values / step[block, np.newaxis, np.newaxis] + 0.5), 0, bin_count)
        row_state = (
            np.arange(index.shape[0])[:, np.newaxis, np.newaxis] * len(p_state)
        ) + np.arange(len(p_state))
        flat = (row_state * points + index).astype(np.int64).ravel()
        late_hist = (
            np.bincount(flat, minlength=index.shape[0] * len(p_state) * points).reshape(
                index.shape[0], len(p_state), points
            )
            / sample_count
        )
        # Sum per state by FFT convolution, mixed in frequency space.
        spectrum = fft.rfft(known, n_fft, axis=-1, workers=-1) * fft.rfft(
            late_hist, n_fft, axis=-1, workers=-1
        )
        total = fft.irfft(np.einsum("s,rsf->rf", p_state, spectrum), n_fft, axis=-1, workers=-1)
        total_q[block] = _lattice_quantiles(total[:, : 2 * bin_count + 1], levels, step[block])
        known_q[block] = _lattice_quantiles(
            np.einsum("s,rsp->rp", p_state, known), levels, step[block]
        )
        late_q[block] = _lattice_quantiles(
            np.einsum("s,rsp->rp", p_state, late_hist), levels, step[block]
        )
    # An all-zero row has no scale: every quantile is exactly 0.
    zero = (top <= 0)[:, np.newaxis]
    return {
        "mean": (known_mean + late_mean).reshape(lead),
        "known_mean": known_mean.reshape(lead),
        "late_mean": late_mean.reshape(lead),
        "quantiles": np.where(zero, 0.0, total_q).reshape(*lead, len(levels)),
        "known_quantiles": np.where(zero, 0.0, known_q).reshape(*lead, len(levels)),
        "late_quantiles": np.where(zero, 0.0, late_q).reshape(*lead, len(levels)),
    }


def _intraday_per_world(totals: dict) -> dict[str, np.ndarray]:
    """Intraday forecast columns (world, slot, cell[, level]) for every world.

    The per-state normals and histograms of ``intraday_distribution`` are
    built one (cell, world) at a time and discarded once its quantiles are
    written.  Slots before the night's decision and windows past the end are
    NaN; with one world there is no other week to read the late part from,
    so every intraday column is NaN.
    """

    inputs = totals["inputs"]
    world_count = totals["world_count"]
    known_sum, late = totals["known_sum"], totals["late_restored"]
    slot_count = known_sum.shape[1]
    shape = (world_count, slot_count, len(CELLS))
    out = {name: np.full(shape, np.nan) for name in ("mean", "known_mean", "late_mean")}
    for name in ("quantiles", "known_quantiles", "late_quantiles"):
        out[name] = np.full((*shape, len(INTRADAY_LEVELS)), np.nan)
    after_decision = np.arange(slot_count) >= totals["decision_slots"][totals["night_index"]]
    # Per-slot grid top: the largest known potential plus the largest late sum
    # over worlds, so a quiet afternoon slot is not binned on the evening's range.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slots past the end
        x_max = np.nanmax(known_sum.sum(axis=3), axis=0) + np.nanmax(late.sum(axis=3), axis=0)
    if world_count < 2:
        return out
    for cell in range(len(CELLS)):
        slots = np.flatnonzero(after_decision & ~np.isnan(x_max[:, cell]))
        cell_late = late[:, slots, cell, :]  # (world, slot, maker)
        for world in range(world_count):
            others = np.delete(cell_late, world, axis=0).transpose(1, 0, 2)
            result = intraday_distribution(
                known_sum[world, slots, cell],
                totals["known_sum_sq"][world, slots, cell],
                inputs.response_rate,
                inputs.outage_probability,
                others,
                INTRADAY_LEVELS,
                x_max[slots, cell],
            )
            for name, values in result.items():
                out[name][world, slots, cell] = values
    return out


# --------------------------------------------------------------------------
# Frames (§10.1f, §10.2e)
# --------------------------------------------------------------------------


def _cell_columns(world_count: int, slot_count: int) -> dict[str, np.ndarray]:
    direction = np.array([d for d, _ in CELLS], dtype=object)
    hours = np.array([h for _, h in CELLS], dtype=float)
    return {
        "direction": np.tile(direction, world_count * slot_count),
        "duration_hours": np.tile(hours, world_count * slot_count),
    }


def _quantile_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """World-first statistics over axis 0: count, mean, sd (ddof 1), P05/P10/P50/P90, shares."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        count = np.isfinite(values).sum(axis=0)
        p05, p10, p50, p90 = np.quantile(values, (0.05, 0.1, 0.5, 0.9), axis=0, method="linear")
        positive = p50 > 0
        safe = np.where(positive, p50, 1.0)
        return {
            "world_count": count.astype(np.int64),
            "mean": values.mean(axis=0),
            "sd": values.std(axis=0, ddof=1),
            "p05": p05,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "firm_share": np.where(positive, p10 / safe, np.nan),
            "firm_share_p05": np.where(positive, p05 / safe, np.nan),
        }


def build_frames(totals: dict, study_slots: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The J2 ``ForecastResult`` frames from filled ``totals`` (§10.1f, §10.2e).

    Returns ``availability_world_slot``, ``availability_bands``,
    ``availability_manufacturer_world``, ``firm_share_by_fleet_size`` and
    ``world_nights``, every one labelled ``illustrative_synthetic``.
    """

    intraday = _intraday_per_world(totals)
    return {
        "availability_world_slot": _world_slot_frame(totals, intraday, study_slots),
        "availability_bands": _bands_frame(totals, intraday, study_slots),
        "availability_manufacturer_world": _manufacturer_world_frame(totals),
        "firm_share_by_fleet_size": _fleet_size_frame(totals),
        "world_nights": world_nights_frame(totals, study_slots),
    }


def _world_slot_frame(totals: dict, intraday: dict, study_slots: pd.DataFrame) -> pd.DataFrame:
    world_count = totals["world_count"]
    slot_count = len(study_slots)
    cells = len(CELLS)
    maker_count = totals["maker_onehot"].shape[1]
    repeat = world_count * slot_count * cells

    slot_rows = np.tile(np.repeat(np.arange(slot_count), cells), world_count)

    def slot_key(column):
        return study_slots[column].take(slot_rows).reset_index(drop=True)

    after_decision = np.arange(slot_count) >= totals["decision_slots"][totals["night_index"]]
    intraday_potential = np.where(
        after_decision[np.newaxis, :, np.newaxis], totals["known_sum"].sum(axis=3), np.nan
    )
    blackout = np.asarray(totals["inputs"].blackout, dtype=bool)
    columns = {
        "world_id": np.repeat(np.arange(world_count, dtype=np.int64), slot_count * cells),
        "slot_index": slot_key("slot_index").astype(np.int64),
        "night_index": slot_key("night_index").astype(np.int64),
        "interval_start_utc": slot_key("interval_start_utc"),
        "interval_start_london": slot_key("interval_start_london"),
        **_cell_columns(world_count, slot_count),
        "deliverable_kw": totals["deliverable"].ravel(),
        "deliverable_known_kw": totals["known"].ravel(),
        "deliverable_late_kw": totals["late"].ravel(),
        "potential_kw": totals["potential"].ravel(),
        "eligible_power_kw": totals["eligible_power"].ravel(),
        "intraday_mean_kw": intraday["mean"].ravel(),
    }
    for k, level in enumerate(INTRADAY_LEVELS):
        columns[f"intraday_p{round(100 * level):02d}_kw"] = intraday["quantiles"][..., k].ravel()
    picks = {
        "p10": INTRADAY_LEVELS.index(0.10),
        "p50": INTRADAY_LEVELS.index(0.50),
        "p90": INTRADAY_LEVELS.index(0.90),
    }
    # The known part keeps all seven levels: the §10.3 backtest scores it at each one.
    columns["intraday_known_mean_kw"] = intraday["known_mean"].ravel()
    for k, level in enumerate(INTRADAY_LEVELS):
        name = f"intraday_known_p{round(100 * level):02d}_kw"
        columns[name] = intraday["known_quantiles"][..., k].ravel()
    columns["intraday_late_mean_kw"] = intraday["late_mean"].ravel()
    for name, k in picks.items():
        columns[f"intraday_late_{name}_kw"] = intraday["late_quantiles"][..., k].ravel()
    for maker in range(maker_count):
        columns[f"late_restored_kw_m{maker + 1}"] = totals["late_restored"][..., maker].ravel()
    columns["intraday_potential_kw"] = intraday_potential.ravel()
    columns["intraday_eligible_count"] = (
        np.where(after_decision[np.newaxis, :, np.newaxis], totals["intraday_eligible_count"], 0)
        .ravel()
        .astype(np.int64)
    )
    columns["blackout"] = np.tile(np.repeat(blackout, cells), world_count)
    columns["evidence_kind"] = np.full(repeat, EVIDENCE_KIND, dtype=object)
    return pd.DataFrame(columns)


def _bands_frame(totals: dict, intraday: dict, study_slots: pd.DataFrame) -> pd.DataFrame:
    slot_count = len(study_slots)
    cells = len(CELLS)
    diagnostics = realised_diagnostics(totals)
    level = {
        name: INTRADAY_LEVELS.index(value)
        for name, value in (("p05", 0.05), ("p10", 0.1), ("p50", 0.5), ("p90", 0.9))
    }
    statistics = [("day_ahead", "realised", totals["deliverable"])]
    for name in ("p05", "p10", "p50", "p90"):
        statistics.append(
            ("intraday", f"conditional_{name}", intraday["quantiles"][..., level[name]])
        )
    statistics.append(
        ("intraday", "conditional_known_p50", intraday["known_quantiles"][..., level["p50"]])
    )

    def by_cell(values):  # (slot, cell) -> cell-major then slot, the frame's row order
        return values.T.ravel()

    blocks = []
    nan = np.full((slot_count, cells), np.nan)
    for horizon, statistic, values in statistics:
        stats = _quantile_stats(values)
        realised = statistic == "realised"
        block = {
            "horizon": np.full(slot_count * cells, horizon, dtype=object),
            "statistic": np.full(slot_count * cells, statistic, dtype=object),
            "direction": np.repeat(np.array([d for d, _ in CELLS], dtype=object), slot_count),
            "duration_hours": np.repeat(np.array([h for _, h in CELLS], dtype=float), slot_count),
        }
        for column in ("slot_index", "night_index", "interval_start_utc", "interval_start_london"):
            block[column] = (
                study_slots[column]
                .take(np.tile(np.arange(slot_count), cells))
                .reset_index(drop=True)
            )
        block["unit"] = np.full(slot_count * cells, "kW", dtype=object)
        for column in (
            "world_count",
            "mean",
            "sd",
            "p05",
            "p10",
            "p50",
            "p90",
            "firm_share",
            "firm_share_p05",
        ):
            block[column] = by_cell(stats[column])
        block["implied_rho"] = by_cell(diagnostics["implied_rho"] if realised else nan)
        block["n_eff"] = by_cell(diagnostics["n_eff"] if realised else nan)
        block["contributing_ev_count"] = by_cell(
            diagnostics["contributing_ev_count"]
            if realised
            else np.zeros((slot_count, cells), dtype=np.int64)
        )
        block["blackout"] = np.tile(np.asarray(totals["inputs"].blackout, dtype=bool), cells)
        block["evidence_kind"] = np.full(slot_count * cells, EVIDENCE_KIND, dtype=object)
        blocks.append(pd.DataFrame(block))
    frame = pd.concat(blocks, ignore_index=True)
    return frame.astype({"slot_index": np.int64, "night_index": np.int64, "world_count": np.int64})


def _manufacturer_world_frame(totals: dict) -> pd.DataFrame:
    world_count = totals["world_count"]
    night_count = totals["night_count"]
    onehot = totals["maker_onehot"]
    maker_count = onehot.shape[1]
    cells = len(CELLS)
    rows = world_count * night_count * maker_count * cells
    outage = np.asarray(totals["inputs"].outage, dtype=bool)
    grid = np.indices((world_count, night_count, maker_count, cells)).reshape(4, -1)
    to_mwh = SLOT_HOURS / 1000.0
    return pd.DataFrame(
        {
            "world_id": grid[0].astype(np.int64),
            "night_index": grid[1].astype(np.int64),
            "manufacturer_id": np.array([f"m{m + 1}" for m in range(maker_count)], dtype=object)[
                grid[2]
            ],
            "direction": np.array([d for d, _ in CELLS], dtype=object)[grid[3]],
            "duration_hours": np.array([h for _, h in CELLS], dtype=float)[grid[3]],
            "ev_count": onehot.sum(axis=0).astype(np.int64)[grid[2]],
            "outage": outage[grid[0], grid[1], grid[2]],
            # kW summed over the night's slots x 0.5 h = kWh; ÷ 1000 = MWh.
            "realised_mwh": totals["maker_realised"].ravel() * to_mwh,
            "potential_mwh": totals["maker_potential"].ravel() * to_mwh,
            "evidence_kind": np.full(rows, EVIDENCE_KIND, dtype=object),
        }
    )


def _fleet_size_frame(totals: dict) -> pd.DataFrame:
    sizes = totals["fleet_sizes"]
    windows = list(PRODUCT_WINDOWS)
    world_count = totals["world_count"]
    variance = _ev_variance(
        totals["window_ev_sum"], totals["window_ev_sum_sq"], world_count
    )  # (window, EV, cell)
    # Cumulative per-EV sums over the first n EVs, in units order.
    variance_sum = np.cumsum(np.nan_to_num(variance), axis=1)[:, sizes - 1]
    sd_sum = np.cumsum(np.sqrt(np.nan_to_num(variance)), axis=1)[:, sizes - 1]
    contributing = np.cumsum(variance > 0, axis=1)[:, sizes - 1]
    diagnostics = correlation_diagnostics(totals["subfleet"], variance_sum, sd_sum, contributing)
    stats = _quantile_stats(totals["subfleet"])  # (window, size, cell)
    grid = np.indices((len(sizes), len(windows), len(CELLS))).reshape(3, -1)

    def pick(values):  # (window, size, cell) -> frame order (size, window, cell)
        return values[grid[1], grid[0], grid[2]]

    rows = grid.shape[1]
    return pd.DataFrame(
        {
            "fleet_size": sizes[grid[0]].astype(np.int64),
            "window": np.array(windows, dtype=object)[grid[1]],
            "direction": np.array([d for d, _ in CELLS], dtype=object)[grid[2]],
            "duration_hours": np.array([h for _, h in CELLS], dtype=float)[grid[2]],
            "ev_count_share": sizes[grid[0]] / totals["vehicle_count"],
            "world_count": pick(stats["world_count"]).astype(np.int64),
            "mean_kw": pick(stats["mean"]),
            "sd_kw": pick(stats["sd"]),
            "p05_kw": pick(stats["p05"]),
            "p10_kw": pick(stats["p10"]),
            "p50_kw": pick(stats["p50"]),
            "p90_kw": pick(stats["p90"]),
            "firm_share": pick(stats["firm_share"]),
            "firm_share_p05": pick(stats["firm_share_p05"]),
            "implied_rho": pick(diagnostics["implied_rho"]),
            "n_eff": pick(diagnostics["n_eff"]),
            "evidence_kind": np.full(rows, EVIDENCE_KIND, dtype=object),
        }
    )


def world_nights_frame(totals: dict, study_slots: pd.DataFrame) -> pd.DataFrame:
    """§10.1f: one row per (world, night) with the night's shared factors and outages.

    ``temperature_c`` and ``solar_clearness`` are the sampled values, so as
    features they are a perfect weather forecast (B4).  The plugged-in count
    is the number of EVs connected for the whole decision slot ``s_n``.
    """

    inputs = totals["inputs"]
    world_count = totals["world_count"]
    night_count = totals["night_count"]
    first = study_slots.drop_duplicates("night_index").set_index("night_index")
    world = np.repeat(np.arange(world_count), night_count)
    night = np.tile(np.arange(night_count), world_count)
    date_index = inputs.warmup_days + night
    outage = np.asarray(inputs.outage, dtype=bool)
    solar = (
        np.full(len(world), np.nan)
        if inputs.solar_clearness is None
        else np.asarray(inputs.solar_clearness, dtype=float)[world, date_index]
    )
    columns = {
        "world_id": world.astype(np.int64),
        "night_index": night.astype(np.int64),
        "night_start_local_date": first.loc[night, "local_date"].to_numpy(dtype=object),
        "day_type": first.loc[night, "day_type"].to_numpy(dtype=object),
        "holiday": np.asarray(inputs.holiday, dtype=bool)[date_index],
        "temperature_c": np.asarray(inputs.temperature_c, dtype=float)[world, date_index],
        "solar_clearness": solar,
        "plug_in_week_factor": np.asarray(inputs.plug_in_week_factor, dtype=float)[world],
        "plug_in_day_factor": np.asarray(inputs.plug_in_day_factor, dtype=float)[world, date_index],
        "skip_logit_shift": np.asarray(inputs.skip_logit_shift, dtype=float)[date_index],
        "plug_in_skip_share": np.asarray(inputs.plug_in_skip_share, dtype=float)[world, date_index],
    }
    for maker in range(outage.shape[2]):
        columns[f"outage_m{maker + 1}"] = outage[world, night, maker]
    count = totals["plugged_at_decision"][world, night].astype(np.int64)
    columns["plugged_in_count_at_decision"] = count
    columns["plugged_in_share_at_decision"] = count / totals["vehicle_count"]
    columns["evidence_kind"] = np.full(len(world), EVIDENCE_KIND, dtype=object)
    return pd.DataFrame(columns)
