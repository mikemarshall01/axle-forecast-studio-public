"""The physics: each EV's battery, half-hour by half-hour, in every world.

What this module owns: the energy balance of the fleet.  Given sampled trips,
home plug-in sessions, weather-adjusted driving efficiency and the run's
public top-up rule, it steps every EV in every Monte Carlo world through the
fixed UTC half-hours (travel, public top-ups, time away, home connection,
home charging, then conservation and 0-100 % SoC checks) and sums the
results, world first, into fleet and cohort frames and across-world bands.

How it fits: ``sampling`` draws the inputs, ``action`` supplies the smart
charger's decision-time inputs and its session planner (decision 0004 item
38), and ``forecast`` calls ``simulate_fleet_intervals``, which runs the
normal path and, with smart charging, the paired selected path (and, with
intraday dispatch, a third reference pass: the day-ahead plan path).  With
``timed_start_allowed`` given, a further, optional "timed" path is run: the
normal rule with home charging barred in some half-hours (decision 0007).
``simulate_unit_intervals`` runs the same kernel but keeps EVs separate, for
the chunked summaries and the one-EV replay; it does not take a timed-start
mask, so that path never appears there.  Nothing here draws a random number.

Reading order: the public top-up rule (decision 0004 item 32), the weather
efficiency equation, then the kernel, whose section comment gives the event
order inside one half-hour.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
import pandas as pd

from axle_studio.model.action import (
    NOT_RECOVERED_TOLERANCE_KWH,
    IntradayDispatch,
    SmartCharging,
    latest_known_prices,
    plan_around_blackout,
    plan_cheapest_slots,
    replan_when_worth,
)
from axle_studio.model.assumptions import WEATHER
from axle_studio.model.clock import utc_half_hour_boundaries
from axle_studio.model.settings import RunSettings

# ============================================================================
# Public top-ups: the three values the kernel reads
# ============================================================================
#
# Decision 0004 item 32 (which supersedes items 3 and 31): EVs never strand.
# Whenever a trip would take an EV's battery below the top-up threshold, it
# tops up at an always-available public charger to the top-up target and
# continues; the time spent charging is not modelled.  Only three values
# therefore affect the physics: the public charging efficiency, the threshold
# and the target.  The earlier destination-availability draw, charger power,
# vehicle acceptance and public SoC ceiling changed no output under item 32 and
# were removed (task M4-layout).


@dataclass(frozen=True, slots=True)
class PublicTopUp:
    """One run's public top-up rule, the same for every EV, world and day.

    ``efficiency_fraction`` converts battery energy added to public grid
    import (grid kWh = battery kWh / efficiency).  The threshold and target
    are SoC fractions of physical capacity with ``0 <= threshold < target <= 1``.
    """

    efficiency_fraction: float
    threshold_soc_fraction: float
    target_soc_fraction: float


def public_top_up(
    *,
    efficiency_fraction: object,
    top_up_threshold_soc_fraction: object,
    top_up_target_soc_fraction: object,
) -> PublicTopUp:
    """Check the caller's public-charge assumptions and return a ``PublicTopUp``.

    These come from outside the kernel (the assumptions module or a test), so
    they are checked once here: each is a real finite number, the efficiency
    is in (0, 1] and ``0 <= threshold < target <= 1``.  A target at or below
    the threshold would add no usable energy, so the kernel's
    repeat-until-served top-up rule could not finish.
    """

    values = {
        "efficiency_fraction": efficiency_fraction,
        "top_up_threshold_soc_fraction": top_up_threshold_soc_fraction,
        "top_up_target_soc_fraction": top_up_target_soc_fraction,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real scalar")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 < efficiency_fraction <= 1.0:
        raise ValueError("efficiency_fraction must be in (0, 1]")
    if not 0.0 <= top_up_threshold_soc_fraction < top_up_target_soc_fraction <= 1.0:
        raise ValueError(
            "top-up fractions must satisfy 0 <= threshold < target <= 1 "
            f"(got threshold {top_up_threshold_soc_fraction}, "
            f"target {top_up_target_soc_fraction})"
        )
    return PublicTopUp(
        float(efficiency_fraction),
        float(top_up_threshold_soc_fraction),
        float(top_up_target_soc_fraction),
    )


# ============================================================================
# Weather: driving efficiency on each world-day
# ============================================================================
#
# One illustrative equation: how far each day's temperature is from a mild
# reference reduces miles per battery kWh.  ``forecast`` calls it once per set
# of worlds with the sampled daily temperatures, and the kernel below divides
# trip miles by the result to get battery energy per trip.


# Illustrative assumptions, not a fit to CNZ or observed weather; the values
# and their reasoning live in the assumptions module.
_REFERENCE_TEMPERATURE_C = WEATHER["reference_temperature_c"].value
_MAXIMUM_LOSS_FRACTION = WEATHER["max_efficiency_loss_fraction"].value


def effective_driving_efficiency(
    base_efficiency_miles_per_battery_kwh: np.ndarray,
    temperature_c: np.ndarray,
    sensitivity_fraction_per_c: float,
) -> np.ndarray:
    """Return driving efficiency (miles per battery kWh) per world, day and EV.

    ``base_efficiency_miles_per_battery_kwh`` has shape (EV,) and
    ``temperature_c`` has shape (world, day); the result has shape
    (world, day, EV).  ``sensitivity_fraction_per_c`` is the fractional
    efficiency loss per degree C away from 20 C, a user-editable assumption.
    """

    base = np.asarray(base_efficiency_miles_per_battery_kwh, dtype=np.float64)
    temperature = np.asarray(temperature_c, dtype=np.float64)
    # Only the user-editable sensitivity is checked: the base efficiencies
    # and temperatures come from the model's own population and sampler.
    sensitivity = float(sensitivity_fraction_per_c)
    if not math.isfinite(sensitivity) or sensitivity < 0.0:
        raise ValueError("sensitivity must be a finite non-negative number")

    # loss = min(s * |T - 20|, 0.5).  The absolute difference makes cold and
    # hot days cost the same (heating and cooling both draw energy); the 50%
    # cap stops extreme sampled temperatures giving near-zero efficiency and
    # absurd trip energy.
    loss = np.clip(
        sensitivity * np.abs(temperature - _REFERENCE_TEMPERATURE_C), 0.0, _MAXIMUM_LOSS_FRACTION
    )
    # efficiency = base * (1 - loss), broadcast to (world, day, EV).
    return base[np.newaxis, np.newaxis, :] * (1.0 - loss[:, :, np.newaxis])


# ============================================================================
# The half-hour kernel and world-first aggregation
# ============================================================================
#
# The half-hour battery stock-flow for every EV in every Monte Carlo world, run
# on NumPy arrays of shape (world, EV), and the aggregation of those per-EV
# results into fleet and cohort totals per world and then P10/P50/P90 bands
# across worlds.  ``forecast`` calls ``simulate_fleet_intervals``; the per-EV
# action screen, the chunked summaries and the one-EV replay call
# ``simulate_unit_intervals``.
#
# Event order inside one half-hour slot ``[start, end)``
# ------------------------------------------------------
# All EVs and worlds are stepped together, slot by slot, in this order:
#
# 1. Travel.  For each journey that overlaps the slot, the outbound leg and
#    then the return leg draw battery energy.  A leg's battery energy is
#    ``miles / effective efficiency`` (kWh, battery side), spread evenly over
#    the leg's driving time, so the slot requests
#    ``leg energy * time driven in slot / leg duration``.
# 2. Public top-ups (decision 0004 item 32).  If a leg's request in the slot
#    would take stock below the top-up threshold, the EV tops up at a public
#    charger to the top-up target when it reaches the threshold, as many
#    times as the request needs, and then continues.  EVs never strand, so
#    every request is served and unserved travel is zero by construction.
# 3. Occupancy.  The slot is split into driving, parked away and home time.
# 4. Home connection.  Connected time is home time covered by an accepted
#    connection session, cut off at the first departure in the slot.
# 5. Home charging.  Only when the EV is connected for the whole slot, up to
#    device power and the preferred target.  On the selected path in the
#    study and the last warm-up night, the smart charger first drops the
#    plan of an EV that has left (an early departure if planned energy was
#    still owed), plans any EV that is plugged in without a plan (leaving
#    blackout half-hours untouched, trading contract v1 §10.5b), and then
#    limits import to the plan's energy for this slot (decision 0004 items 38
#    and 52).  A session that ignores its plan (non-response, a maker outage,
#    a control outage or the control group, §10.1e) keeps the plan on record
#    but charges by the normal rule.
#    With intraday dispatch (intraday dispatch contract v1 §4.1), before
#    anything about the slot is read and at whole hours only, free EVs'
#    plans in force are first re-planned on the latest intraday price when
#    that is worth it; then the trader's plan book is taken; then early
#    departures, new plans (free EVs on the latest price) and the energy.
#    On the optional "timed" path (decision 0007), the normal rule above runs
#    unchanged and then ``charge_allowed`` zeroes the slot's home import when
#    this half-hour is barred: a hard rule, no fallback and no spill into an
#    allowed half-hour, so a session that cannot fill in time leaves short,
#    exactly as a real timer-only charger would.
# 6. Checks.  Per-EV battery conservation and 0-capacity stock bounds.
#
# World-first aggregation: per-EV values are summed to fleet and cohort totals
# inside each world first, and only then are across-world means and
# percentiles taken.  Percentiles are never summed.


_PATH_IDS = ("normal", "selected", "timed")
"""Path ids a fleet/cohort result frame may carry; "timed" is optional (decision 0007)."""
_LONDON = "Europe/London"
_BOOLEAN_INPUTS = ("drives_today", "connection_session_accepted")
_NUMERIC_INPUTS = (
    "outbound_miles",
    "return_miles",
    "destination_dwell_seconds",
    "drive_speed_mph",
    "desired_pre_drive_soc_fraction",
)
_TIMESTAMP_INPUTS = ("trip_departure_utc", "connection_start_utc", "connection_end_utc")
_TRAVEL_FLOWS = tuple(
    f"{kind}_{leg}_travel_battery_kwh"
    for leg in ("outbound", "return")
    for kind in ("requested", "served", "unserved")
)
# Per-EV slot quantities that are summed over EVs, in output column order.
_SUM_COLUMNS = [
    "away_on_trip_fraction",
    "home_unplugged_fraction",
    "home_connected_fraction",
    "driving_fraction",
    "parked_away_fraction",
    "public_charging_fraction",
    "opening_battery_kwh",
    "closing_battery_kwh",
    "physical_capacity_kwh",
    "realised_grid_kw",
    "home_grid_import_kwh",
    "home_battery_added_kwh",
    "public_grid_import_kwh",
    "public_battery_added_kwh",
    *_TRAVEL_FLOWS,
    "v2g_battery_removed_kwh",
    "conservation_residual_kwh",
    "early_departure_count",
    "early_departure_shortfall_kwh",
]
# Kernel output per slot: unit_count, the summed quantities, connected_count.
_KERNEL_COLUMNS = ["unit_count", *_SUM_COLUMNS, "connected_count"]
_KEYS = ["world_id", "path_id", "interval_start_utc", "interval_end_utc"]
_COHORT_KEYS = [*_KEYS, "cohort_id"]
_INTERVAL_COLUMNS = [
    *_KEYS,
    *_KERNEL_COLUMNS,
    "closing_soc_percent",
    "connected_share",
]
_COHORT_INTERVAL_COLUMNS = ["cohort_id", *_INTERVAL_COLUMNS]
_BAND_KEYS = ["path_id", "interval_start_utc", "interval_end_utc"]
_COHORT_BAND_KEYS = ["cohort_id", *_BAND_KEYS]
_BAND_METRICS = [*_KERNEL_COLUMNS, "closing_soc_percent", "connected_share"]
_BAND_STATISTICS = [
    f"{metric}_{stat}" for metric in _BAND_METRICS for stat in ("mean", "p10", "p50", "p90")
]
_BAND_COLUMNS = [*_BAND_KEYS, "world_count", *_BAND_STATISTICS]
_COHORT_BAND_COLUMNS = ["cohort_id", *_BAND_COLUMNS]
_NAT_INT = np.datetime64("NaT", "ns").astype(np.int64)
_NO_EVENT_INT = np.iinfo(np.int64).max
_NS_PER_SECOND = 1_000_000_000.0
_HALF_HOUR_NS = 30 * 60 * 1_000_000_000
_SLOT_HOURS = 0.5
_CHECK_TOLERANCE_KWH = 1e-12

ZONE_SUM_COLUMNS = (
    "unit_count",
    "connected_count",
    "home_grid_import_kwh",
    "early_departure_count",
    "early_departure_shortfall_kwh",
)
"""Kernel columns summed per zone (trading contract v1 §3), in ``ZoneSums`` order."""


@dataclass
class ZoneSums:
    """Per-zone world sums the kernel fills in, like its cohort sums (trading contract v1 §3).

    The caller gives ``zone_index`` (EV,), each EV's zone as a position
    0 ... ``zone_count`` - 1 in ``assumptions.ZONE_IDS`` order.
    ``simulate_fleet_intervals`` fills ``study[path_id]`` with a (world,
    zone, study slot, ``ZONE_SUM_COLUMNS``) array per path, and
    ``warmup_home_import_kwh`` with the normal path's home import (kWh per
    half-hour) per (world, zone, warm-up slot): the unmanaged history a
    zone's settlement baseline is built from.  Zones partition the fleet,
    so the zone sums add up to the fleet sums exactly.  It is an output
    holder, like the ``warmup_home_import_kwh`` array, so the kernel's
    return shape does not change for callers without zones.
    """

    zone_index: np.ndarray
    zone_count: int
    study: dict[str, np.ndarray] = field(default_factory=dict)
    warmup_home_import_kwh: np.ndarray | None = None


@dataclass
class DispatchSums:
    """What the intraday dispatch passes add, filled by the kernel (dispatch contract v1 §5.2).

    An output holder like ``ZoneSums``, so the kernel's return shape does not
    change.  With intraday dispatch on, ``simulate_fleet_intervals`` runs a
    third pass beside ``normal`` and ``selected`` (now the dispatched path):
    the day-ahead plan path, smart charging with every EV planned on
    B4-visible day-ahead prices (what ``selected`` is with dispatch off).
    It is a reference, never a ``path_id``.  All kWh are grid home import per
    half-hour, per (world, study slot) unless stated:

    - ``day_ahead_home_import_kwh``: fleet import on the day-ahead plan path;
      ``day_ahead_zone_home_import_kwh`` (world, zone, study slot) the same
      per zone (one zone when the run has no ``ZoneSums``);
    - ``locked_home_import_kwh``: locked EVs on the dispatched path (equal
      on both smart paths, since a locked EV never ranks an intraday price
      and EVs do not interact);
    - ``free_home_import_kwh`` and ``free_day_ahead_home_import_kwh``: free
      EVs on the dispatched and on the day-ahead plan path;
    - ``replan_count`` (int64): free EVs whose plan changed at that slot's
      decision (0 off whole hours);
    - ``book_kwh``: the trader's plan books (world, decision, slot ahead) by
      path, ``selected`` (the dispatched fleet, after the re-plans made at
      each decision) and ``day_ahead`` (for the frozen-book re-run, §6.2),
      when the caller asked for the selected book.
    """

    day_ahead_home_import_kwh: np.ndarray | None = None
    day_ahead_zone_home_import_kwh: np.ndarray | None = None
    locked_home_import_kwh: np.ndarray | None = None
    free_home_import_kwh: np.ndarray | None = None
    free_day_ahead_home_import_kwh: np.ndarray | None = None
    replan_count: np.ndarray | None = None
    book_kwh: dict[str, np.ndarray] = field(default_factory=dict)


UNIT_INTERVAL_QUANTITIES = (
    "home_grid_import_kwh",
    "public_grid_import_kwh",
    "unserved_travel_battery_kwh",
    "opening_battery_kwh",
    "closing_battery_kwh",
    "planned_home_import_kwh",
    "plan_remaining_need_kwh",
    "decision_plan_kwh",
    "plan_status",
)
"""Per-EV interval quantities returned by ``simulate_unit_intervals``.

The last four describe the smart plan in force on the selected path
(trading contract v1 §10.1e) and are 0 on the normal path; see
``simulate_unit_intervals``."""

PLAN_FOLLOWS = 0
"""``plan_status`` codes (trading contract v1 §10.1e, precedence 4, 1, 2, 3):
the session follows its plan (0), ignores it by base non-response (1),
because its maker's cloud is out that night (2), because the plan was made
inside a scripted control-outage window (3), or is a hold-out control EV (4)."""
BASE_NON_RESPONSE = 1
MAKER_OUTAGE = 2
CONTROL_OUTAGE_EVENT = 3
CONTROL_GROUP = 4


def simulate_fleet_intervals(
    settings: RunSettings,
    units: pd.DataFrame,
    evaluation_inputs: dict[str, np.ndarray],
    *,
    public_top_up: PublicTopUp,
    evaluation_effective_efficiency_miles_per_battery_kwh: np.ndarray | None = None,
    smart_charging: SmartCharging | None = None,
    warmup_home_import_kwh: np.ndarray | None = None,
    zone_sums: ZoneSums | None = None,
    trading_output: dict[str, dict[str, np.ndarray]] | None = None,
    intraday_dispatch: IntradayDispatch | None = None,
    dispatch_sums: DispatchSums | None = None,
    timed_start_allowed: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return fleet and cohort interval results per world and across worlds.

    ``evaluation_inputs`` holds the sampler arrays of shape (world, day, EV):
    trips, home connection sessions (UTC timestamps) and trip parameters.
    ``evaluation_effective_efficiency_miles_per_battery_kwh`` (same shape)
    replaces each EV's base efficiency when given.

    Returns four frames: fleet totals per world and path, cohort totals per
    world and path, and the across-world mean/P10/P50/P90 bands of each.

    With no ``smart_charging``, the physical path is simulated once and
    reported as identical normal and selected paths.  With it, the selected
    path is a second, paired pass on the same sampled inputs (common random
    numbers) in which home charging follows the smart plans, so its battery
    state then evolves independently of the normal path.

    ``public_top_up`` gives the public charging efficiency and the top-up
    threshold and target (decision 0004 item 32); a public charger is always
    available, so every run tops up.

    ``warmup_home_import_kwh``, when given, is a (world, warm-up slot) array
    the kernel fills with the normal path's fleet home import (kWh per
    half-hour) in the warm-up: the unmanaged history the trading baseline is
    built from (decision 0004 item 52, trading contract v1 section 4.1).
    The warm-up is simulated but never reported, so this is its only output.

    ``zone_sums``, when given, is filled with per-zone sums of both paths
    and the normal path's warm-up import per zone (see ``ZoneSums``).

    ``trading_output``, when given, maps ``"normal"`` and ``"selected"`` to
    dicts of preallocated arrays the kernel fills for the trading overlay
    (trading contract v1 §1.4); see ``_record_trading`` for the keys.  It
    reads kernel state only and changes no physics.

    ``intraday_dispatch`` (needs ``smart_charging``) makes the selected path
    the dispatched path (intraday dispatch contract v1 §4) and adds a third
    pass on the same sampled inputs, the day-ahead plan path, whose outputs
    go to ``dispatch_sums`` (see ``DispatchSums``).  Without it no third pass
    runs and the selected path is the day-ahead plan path, exactly as before.

    ``timed_start_allowed`` (run slot,) bool, when given, adds an optional
    "timed" path (decision 0007): the normal rule, except that
    ``_simulate_path`` zeroes home import in the half-hours it marks
    ``False`` (``action.timed_start_allowed``).  It is a further paired pass
    on the same sampled inputs, reported in the returned frames beside
    normal and selected, and, with ``zone_sums``, it also joins
    ``zone_sums.study`` so the zone frames can report it (model step 2).  It
    never enters ``trading_output``, which stays the reported normal and
    selected paths only.  Without it no third pass runs and the returned
    frames are unchanged.
    """

    if settings.vehicle_count == 0:
        return (
            _empty_frame(_INTERVAL_COLUMNS),
            _empty_frame(_COHORT_INTERVAL_COLUMNS),
            _empty_frame(_BAND_COLUMNS),
            _empty_frame(_COHORT_BAND_COLUMNS),
        )
    population, values, boundaries_ns, cohort_codes, cohort_ids = _kernel_inputs(
        settings, units, evaluation_inputs
    )
    smart = _validated_smart_charging(settings, smart_charging)
    dispatch = _validated_intraday_dispatch(settings, smart, intraday_dispatch)
    timed_mask = _validated_timed_start_allowed(settings, timed_start_allowed)
    shape = (settings.evaluation_world_count, settings.study_days * 48)

    def simulate(
        path_smart: SmartCharging | None,
        path_id: str,
        warmup_output: np.ndarray | None = None,
        path_dispatch: IntradayDispatch | None = None,
        split_output: dict[str, np.ndarray] | None = None,
        charge_allowed: np.ndarray | None = None,
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray | None]:
        zone_output = zone_warmup = zone_index = None
        # The timed path now joins per-zone sums when the caller tracks zones
        # (model step 2, decision 0007 extension): zone frames need it, same
        # as normal and selected.  It still never enters the trading overlay
        # (``trading_by_path.get(path_id)`` below is only ever populated for
        # "normal"/"selected"/"day_ahead"), so that hard constraint stands.
        track_zones = zone_sums is not None or path_id == "day_ahead"
        if track_zones:
            # The day-ahead plan path always keeps per-zone import (one zone
            # without ZoneSums) but is the one path that never enters
            # ``zone_sums.study``: that dict reports normal, selected and,
            # since the model step 2 extension (decision 0007), optionally
            # timed too.
            zone_index = (
                zone_sums.zone_index
                if zone_sums is not None
                else np.zeros(settings.vehicle_count, dtype=np.int64)
            )
            zone_count = 1 if zone_sums is None else zone_sums.zone_count
            zone_output = np.empty((*shape[:1], zone_count, shape[1], len(ZONE_SUM_COLUMNS)))
            if path_id != "day_ahead":
                zone_sums.study[path_id] = zone_output
            if warmup_output is not None:
                zone_warmup = np.zeros(
                    (
                        settings.evaluation_world_count,
                        zone_sums.zone_count,
                        settings.warmup_days * 48,
                    )
                )
                zone_sums.warmup_home_import_kwh = zone_warmup
        return _simulate_path(
            settings,
            population,
            values,
            boundaries_ns,
            cohort_codes,
            len(cohort_ids),
            evaluation_effective_efficiency_miles_per_battery_kwh,
            public_top_up,
            path_smart,
            warmup_home_import_kwh=warmup_output,
            zone_index=zone_index,
            zone_output=zone_output,
            warmup_zone_home_import_kwh=zone_warmup,
            trading=trading_by_path.get(path_id),
            dispatch=path_dispatch,
            dispatch_output=split_output,
            charge_allowed=charge_allowed,
        ), zone_output

    trading_by_path = dict(trading_output or {})
    split = {}
    if dispatch is not None:
        # The day-ahead plan path's book, for the frozen-book re-run (§6.2),
        # taken at the same hourly decisions as the selected book.
        selected_trading = trading_by_path.get("selected", {})
        if "book_kwh" in selected_trading:
            trading_by_path["day_ahead"] = {
                "book_decision": selected_trading["book_decision"],
                "book_kwh": np.zeros_like(selected_trading["book_kwh"]),
            }
        for path_id in ("day_ahead", "selected"):
            split[path_id] = {
                "locked_mask": dispatch.locked,
                "locked_home_import_kwh": np.zeros(shape),
                "free_home_import_kwh": np.zeros(shape),
            }
        split["selected"]["replan_count"] = np.zeros(shape, dtype=np.int64)

    # The normal path always fills the warm-up outputs, so a zone run
    # without a caller-supplied warm-up array still gets zone warm-up sums.
    warmup_output = warmup_home_import_kwh
    if warmup_output is None and zone_sums is not None:
        warmup_output = np.zeros((settings.evaluation_world_count, settings.warmup_days * 48))
    normal = simulate(None, "normal", warmup_output)[0]
    if smart is None:
        selected = normal
        if zone_sums is not None:
            zone_sums.study["selected"] = zone_sums.study["normal"]
    else:
        if dispatch is not None:
            # The reference pass first: smart charging without dispatch, on
            # the same sampled inputs (common random numbers).
            day_ahead, day_ahead_zones = simulate(
                smart, "day_ahead", split_output=split["day_ahead"]
            )
        selected = simulate(
            smart, "selected", path_dispatch=dispatch, split_output=split.get("selected")
        )[0]
        if dispatch is not None and dispatch_sums is not None:
            home = _KERNEL_COLUMNS.index("home_grid_import_kwh")
            dispatch_sums.day_ahead_home_import_kwh = day_ahead[0][..., home].copy()
            dispatch_sums.day_ahead_zone_home_import_kwh = day_ahead_zones[
                ..., ZONE_SUM_COLUMNS.index("home_grid_import_kwh")
            ].copy()
            dispatch_sums.locked_home_import_kwh = split["selected"]["locked_home_import_kwh"]
            dispatch_sums.free_home_import_kwh = split["selected"]["free_home_import_kwh"]
            dispatch_sums.free_day_ahead_home_import_kwh = split["day_ahead"][
                "free_home_import_kwh"
            ]
            dispatch_sums.replan_count = split["selected"]["replan_count"]
            dispatch_sums.book_kwh = {
                path_id: trading_by_path[path_id]["book_kwh"]
                for path_id in ("selected", "day_ahead")
                if "book_kwh" in trading_by_path.get(path_id, {})
            }
    timed = None
    if timed_mask is not None:
        # A further paired pass on the same sampled inputs (common random
        # numbers, decision 0007): the normal rule with home charging barred
        # outside the allowed window.  No smart plan, no dispatch and no
        # warm-up output of its own; it does join the per-zone sums when the
        # caller tracks zones (model step 2 extension), the same
        # ``track_zones``/``zone_sums.study`` path normal and selected use.
        timed = simulate(None, "timed", charge_allowed=timed_mask)[0]
    starts = boundaries_ns[settings.warmup_days * 48 : -1]
    world = {"normal": normal[0], "selected": selected[0]}
    cohort = {"normal": normal[1], "selected": selected[1]}
    if timed is not None:
        world["timed"] = timed[0]
        cohort["timed"] = timed[1]
    return (
        _interval_frame(world, starts),
        _interval_frame(cohort, starts, cohort_ids=cohort_ids),
        _band_frame(world, starts),
        _band_frame(cohort, starts, cohort_ids=cohort_ids),
    )


def simulate_unit_intervals(
    settings: RunSettings,
    units: pd.DataFrame,
    inputs: dict[str, np.ndarray],
    *,
    public_top_up: PublicTopUp,
    effective_efficiency_miles_per_battery_kwh: np.ndarray | None = None,
    smart_charging: SmartCharging | None = None,
    intraday_dispatch: IntradayDispatch | None = None,
    charge_allowed: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return per-EV study-interval outcomes for one physical path.

    This runs the same kernel as ``simulate_fleet_intervals`` but keeps each
    EV separate: the normal path without ``smart_charging``, the selected
    path with it, or, with ``charge_allowed`` and neither of those, the
    timed path (model step 2, decision 0007): the same (run slot,) bool
    "may charge" mask ``simulate_fleet_intervals`` takes as
    ``timed_start_allowed``, so the per-EV chunked summaries (charge
    completion, session distribution, household) can learn the timed path
    exactly as they do normal and selected.  Each array in the returned
    mapping (keys
    ``UNIT_INTERVAL_QUANTITIES``) has shape (world, study interval, EV) in
    kWh: home and public grid import in the interval, unserved outbound plus
    return travel (battery-side, always zero since decision 0004 item 32;
    kept as a check) in the interval, and battery stock at the interval start
    and end.  The flexibility metrics read the opening stock: an EV plugged in
    for the whole slot has no travel in it, so its opening stock is the stock
    the charger sees (decision 0004 item 43).

    It exists because SoC spread across EVs and plug-in sessions need each
    EV's outcome, which fleet sums hide.  Memory grows with world x interval
    x EV, so callers pass chunks of a few worlds at a time.

    The smart plan in force (trading contract v1 §10.1e; all 0 on the normal
    path and outside a plan), which the Firm-MW availability reads:

    - ``planned_home_import_kwh``: the plan's grid kWh for the interval.  A
      session that ignores its plan keeps it on record (the aggregator does
      not know who will ignore it), so this is its smart plan, not its
      full-power import.
    - ``plan_remaining_need_kwh``: grid kWh the plan still had to deliver at
      the interval start, the need it was made for minus its own kWh in the
      earlier intervals of the plan, floored at 0; it restarts at a replan.
    - ``decision_plan_kwh``: for an interval of night n at or after that
      night's decision slot ``s_n``, the kWh of the plan that was in force
      at ``s_n`` (EVs plugged in then only; 0 outside that plan's window).
      It is read from a copy of the plans taken at ``s_n``, so a replan
      after the decision never changes it (R1).
    - ``plan_status``: the plan's code (``PLAN_FOLLOWS`` ... ``CONTROL_GROUP``).

    With ``intraday_dispatch`` (and ``smart_charging``) the selected path is
    the dispatched path, so these per-EV frames reconcile with the fleet
    frames of ``simulate_fleet_intervals`` run with the same dispatch.
    """

    shape = (settings.evaluation_world_count, settings.study_days * 48, settings.vehicle_count)
    unit_output = {name: np.zeros(shape, dtype=np.float64) for name in UNIT_INTERVAL_QUANTITIES}
    if settings.vehicle_count == 0:
        return unit_output
    population, values, boundaries_ns, cohort_codes, cohort_ids = _kernel_inputs(
        settings, units, inputs
    )
    smart = _validated_smart_charging(settings, smart_charging)
    _simulate_path(
        settings,
        population,
        values,
        boundaries_ns,
        cohort_codes,
        len(cohort_ids),
        effective_efficiency_miles_per_battery_kwh,
        public_top_up,
        smart,
        unit_output=unit_output,
        dispatch=_validated_intraday_dispatch(settings, smart, intraday_dispatch),
        charge_allowed=_validated_timed_start_allowed(settings, charge_allowed),
    )
    return unit_output


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _kernel_inputs(
    settings: RunSettings,
    units: pd.DataFrame,
    inputs: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Convert sampler output to kernel arrays and derive journey timings.

    The samplers validate their own draws, so this does not re-check dtypes,
    ranges or finiteness.  It converts types (UTC timestamps to int64
    nanoseconds), derives leg end times and checks only the event-order
    invariants the kernel relies on.
    """

    population = {
        column: units[column].to_numpy(dtype=np.float64, copy=True)
        for column in (
            "physical_capacity_kwh",
            "efficiency_miles_per_battery_kwh",
            "home_charger_limit_kw",
            "preferred_target_soc_fraction",
        )
    }
    values = {name: inputs[name].astype(bool, copy=True) for name in _BOOLEAN_INPUTS}
    values.update({name: inputs[name].astype(np.float64, copy=True) for name in _NUMERIC_INPUTS})
    values.update({name: _utc_nanoseconds(inputs[name], name) for name in _TIMESTAMP_INPUTS})
    _add_journey_timings(values)

    # Fixed 48 * days UTC slots from London noon; a London offset change is
    # allowed (decision 0004 items 1 and 52).
    boundaries = utc_half_hour_boundaries(
        settings.start_local_date, settings.warmup_days, settings.study_days
    )
    boundaries_ns = (
        pd.DatetimeIndex(boundaries).to_numpy(dtype="datetime64[ns]").astype(np.int64).copy()
    )
    # Day d is the London calendar date first_local_date + d.  Across an offset
    # change its local midnights differ by one hour from UTC slot 48 * d.
    first_local_date = pd.Timestamp(settings.start_local_date) - pd.Timedelta(
        days=settings.warmup_days
    )
    local_day_starts_ns = (
        pd.date_range(first_local_date, periods=values["drives_today"].shape[1] + 1, freq="D")
        .tz_localize(_LONDON)
        .tz_convert("UTC")
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    _check_event_order(values, local_day_starts_ns)
    cohort_codes, cohort_ids = pd.factorize(units["cohort_id"], sort=False)
    return population, values, boundaries_ns, cohort_codes, cohort_ids.to_numpy()


def _utc_nanoseconds(source: np.ndarray, name: str) -> np.ndarray:
    """Return UTC timestamps as int64 nanoseconds (NaT for missing events).

    The samplers emit ``datetime64[ns]`` arrays holding UTC instants, which
    are int64 nanoseconds already, so they are read without conversion (the
    per-value conversion of object arrays took seconds at 1,000 EVs x 100
    worlds, audit item 2.2).  Object arrays of pandas Timestamps, which
    tests and hand-built cases pass, must be timezone-aware UTC: silently
    reading local time as UTC would shift events by an hour in summer.
    """

    if source.dtype.kind == "M":
        return source.astype("datetime64[ns]").view(np.int64)
    if source.size == 0 or pd.isna(source).all():
        return np.full(source.shape, _NAT_INT, dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        timestamps = pd.to_datetime(source.ravel(), errors="raise", format="mixed")
    if getattr(timestamps, "tz", None) is None or str(timestamps.tz) != "UTC":
        raise ValueError(f"{name} values must be timezone-aware UTC")
    # Pandas 3 may keep microsecond resolution; the kernel works in
    # nanoseconds because the UTC slot boundaries are nanosecond keys.
    return timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64).reshape(source.shape).copy()


def _add_journey_timings(values: dict[str, np.ndarray]) -> None:
    """Add leg end times and leg durations (int64 ns) to ``values``.

    A journey is: depart, drive outbound, dwell at the destination, drive the
    homebound return leg.  Leg time is ``3600 * miles / mph`` seconds.
    """

    drives = values["drives_today"]
    outbound_ns = np.trunc(
        3600.0 * values["outbound_miles"] / values["drive_speed_mph"] * _NS_PER_SECOND
    ).astype(np.int64)
    return_ns = np.trunc(
        3600.0 * values["return_miles"] / values["drive_speed_mph"] * _NS_PER_SECOND
    ).astype(np.int64)
    dwell_ns = np.trunc(
        np.where(drives, values["destination_dwell_seconds"], 0.0) * _NS_PER_SECOND
    ).astype(np.int64)
    values["outbound_end_utc"] = values["trip_departure_utc"] + outbound_ns
    values["return_start_utc"] = values["outbound_end_utc"] + dwell_ns
    values["return_end_utc"] = values["return_start_utc"] + return_ns
    # Durations use microsecond resolution to match the scalar reference
    # (``Timedelta.total_seconds()``), so both kernels spread energy alike.
    values["outbound_duration_ns"] = _total_seconds_ns(outbound_ns)
    values["return_duration_ns"] = _total_seconds_ns(return_ns)


def _check_event_order(values: dict[str, np.ndarray], local_day_starts_ns: np.ndarray) -> None:
    """Check once the event-order invariants the slot loop relies on.

    The loop assumes one EV is in at most one place at a time: journeys do not
    overlap, home sessions do not overlap, and day d's trip departs on London
    date d (weather efficiency is indexed by that day).  A homebound arrival
    at the same instant as a session end is refused because the model
    contract leaves same-boundary arrival/unplug precedence pending rather
    than letting evaluation order decide it (docs/MODEL_CONTRACT.md, "Time
    and boundary rules").
    """

    drives = values["drives_today"]
    accepted = values["connection_session_accepted"]
    starts = values["connection_start_utc"]
    ends = values["connection_end_utc"]
    if (starts[accepted] >= ends[accepted]).any():
        raise ValueError("connection_start_utc must be before connection_end_utc")
    sort_starts = np.where(accepted, starts, np.iinfo(np.int64).max)
    order = np.argsort(sort_starts, axis=1, kind="stable")
    ordered_starts = np.take_along_axis(sort_starts, order, axis=1)
    ordered_ends = np.take_along_axis(
        np.where(accepted, ends, np.iinfo(np.int64).min), order, axis=1
    )
    ordered_accepted = np.take_along_axis(accepted, order, axis=1)
    if (
        ordered_accepted[:, 1:, :]
        & ordered_accepted[:, :-1, :]
        & (ordered_starts[:, 1:, :] <= ordered_ends[:, :-1, :])
    ).any():
        raise ValueError("explicit connection sessions must not overlap or touch")

    departure = values["trip_departure_utc"]
    return_end = values["return_end_utc"]
    day_start = local_day_starts_ns[np.newaxis, :-1, np.newaxis]
    day_end = local_day_starts_ns[np.newaxis, 1:, np.newaxis]
    if (drives & ((departure < day_start) | (departure >= day_end))).any():
        raise ValueError("trip_departure_utc must fall within its indexed local day")
    latest_return = np.maximum.accumulate(
        np.where(drives, return_end, np.iinfo(np.int64).min), axis=1
    )
    if (drives[:, 1:, :] & (departure[:, 1:, :] <= latest_return[:, :-1, :])).any():
        raise ValueError("explicit journeys must not overlap or share a boundary")
    same_instant = (
        drives[:, :, np.newaxis, :]
        & accepted[:, np.newaxis, :, :]
        & (return_end[:, :, np.newaxis, :] == ends[:, np.newaxis, :, :])
    )
    if same_instant.any():
        raise ValueError("physical return and connection end must not share a boundary")


def _validated_smart_charging(
    settings: RunSettings, smart: SmartCharging | None
) -> SmartCharging | None:
    """Check the smart charger's input shapes against this run (or slice).

    ``action.smart_charging_inputs`` builds and checks the values; this only
    guards against a mismatched slice, which would silently plan with another
    EV's departure time.
    """

    if smart is None:
        return None
    slot_count = 48 * (settings.warmup_days + settings.study_days)
    prices_shape = smart.visible_price_gbp_per_mwh.shape
    if prices_shape[1:] != (settings.evaluation_world_count, slot_count) or (
        smart.price_epoch_by_slot.shape != (slot_count,)
    ):
        raise ValueError("smart charging needs visible prices per world and run slot")
    if smart.expected_departure_utc_ns.shape[1] != settings.vehicle_count:
        raise ValueError("smart charging needs one expected departure row per EV")
    return smart


def _validated_intraday_dispatch(
    settings: RunSettings, smart: SmartCharging | None, dispatch: IntradayDispatch | None
) -> IntradayDispatch | None:
    """Check the intraday dispatcher's shapes against this run (or slice).

    ``action.intraday_dispatch_inputs`` checks the values; this guards a
    mismatched slice (another EV's lock, another world's prices) and the
    one combination that has no meaning: dispatch without smart plans.
    """

    if dispatch is None:
        return None
    if smart is None:
        raise ValueError("intraday dispatch re-plans smart plans: it needs smart charging")
    slot_count = 48 * (settings.warmup_days + settings.study_days)
    if (
        dispatch.locked.shape != (settings.vehicle_count,)
        or dispatch.intraday_path_gbp_per_mwh.shape[:2]
        != (settings.evaluation_world_count, slot_count)
        or dispatch.decision_slot.shape != (slot_count,)
    ):
        raise ValueError("intraday dispatch needs a lock per EV and prices per world and run slot")
    return dispatch


def _validated_timed_start_allowed(
    settings: RunSettings, timed_start_allowed: np.ndarray | None
) -> np.ndarray | None:
    """Check the timed-start "may charge" mask's shape against this run (decision 0007).

    ``action.timed_start_allowed`` builds and checks the barred-window rule
    itself; this only guards against a mismatched slice (another run's
    horizon length), the same role ``_validated_smart_charging`` plays for
    the smart charger's inputs.
    """

    if timed_start_allowed is None:
        return None
    slot_count = 48 * (settings.warmup_days + settings.study_days)
    mask = np.asarray(timed_start_allowed, dtype=bool)
    if mask.shape != (slot_count,):
        raise ValueError("timed_start_allowed must have one bool per run slot")
    return mask


# --------------------------------------------------------------------------
# Physics: one physical path, stepped slot by slot
# --------------------------------------------------------------------------


def _simulate_path(
    settings: RunSettings,
    population: dict[str, np.ndarray],
    values: dict[str, np.ndarray],
    boundaries_ns: np.ndarray,
    cohort_codes: np.ndarray,
    cohort_count: int,
    effective_efficiency: np.ndarray | None,
    public: PublicTopUp,
    smart: SmartCharging | None,
    unit_output: dict[str, np.ndarray] | None = None,
    warmup_home_import_kwh: np.ndarray | None = None,
    zone_index: np.ndarray | None = None,
    zone_output: np.ndarray | None = None,
    warmup_zone_home_import_kwh: np.ndarray | None = None,
    trading: dict[str, np.ndarray] | None = None,
    dispatch: IntradayDispatch | None = None,
    dispatch_output: dict[str, np.ndarray] | None = None,
    charge_allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Step one physical path through every slot; return study-slot sums.

    ``smart`` is None for the normal path and the smart charger's inputs for
    the selected path.  Returns fleet sums of shape (world, study slot,
    kernel column) and cohort sums of shape (world, cohort, study slot,
    kernel column), columns in ``_KERNEL_COLUMNS`` order.  Warm-up slots are
    simulated (so the study opens with a settled battery) but not reported.
    When ``unit_output`` is given, per-EV study values are written into it as
    well; when ``warmup_home_import_kwh`` is given, the fleet home import of
    each (world, warm-up slot) is.  With ``zone_index`` (EV,), ``zone_output``
    (world, zone, study slot, ``ZONE_SUM_COLUMNS``) and
    ``warmup_zone_home_import_kwh`` (world, zone, warm-up slot) receive the
    same sums per zone.  When ``trading`` is given, the trading overlay's
    kernel outputs are filled (``_record_trading``).  ``dispatch`` (with
    ``smart``) makes this the dispatched path (intraday dispatch contract v1
    §4.1).  ``dispatch_output``, when given, holds ``locked_mask`` (EV,) and
    receives the locked and free EVs' home import per (world, study slot)
    and, when it has the key, each study slot's ``replan_count``.

    ``charge_allowed`` (run slot,) bool, when given, is the optional "timed"
    path's barred-window rule (decision 0007, ``action.timed_start_allowed``):
    a slot it marks ``False`` gets no home import at all, whatever the normal
    rule above would have charged.  Hard rule, no fallback: a session that
    cannot fill before it must leave simply leaves short, exactly as items 2
    and 32 already leave a session short of its target when it runs out of
    time, rather than borrowing from a barred half-hour.
    """

    world_count = settings.evaluation_world_count
    vehicle_count = settings.vehicle_count
    study_start = settings.warmup_days * 48
    study_slots = settings.study_days * 48
    world_output = np.empty((world_count, study_slots, len(_KERNEL_COLUMNS)), dtype=np.float64)
    cohort_output = np.empty(
        (world_count, cohort_count, study_slots, len(_KERNEL_COLUMNS)), dtype=np.float64
    )

    capacity = population["physical_capacity_kwh"]
    efficiency = population["efficiency_miles_per_battery_kwh"]
    if effective_efficiency is not None:
        efficiency = effective_efficiency
    # Battery-side energy per leg for every (world, day, EV).
    leg_energy = {
        "outbound": values["outbound_miles"] / efficiency,
        "return": values["return_miles"] / efficiency,
    }
    # The preferred target is where ordinary home charging stops; it is not
    # the physical ceiling, which is ``capacity``.
    target_stock = capacity * population["preferred_target_soc_fraction"]
    # Home charging power alone limits home charging: decision 0004 item 34
    # dropped the separate vehicle AC limit (and its min() here).
    device_power_kw = population["home_charger_limit_kw"]

    top_up = _top_up_levels(public, capacity)

    # Path state carried across slots.
    stock = np.broadcast_to(
        capacity * settings.opening_soc_fraction, (world_count, vehicle_count)
    ).copy()
    plans = None if smart is None else _no_plans(smart, world_count, vehicle_count)
    horizon_end_ns = boundaries_ns[-1]
    # No EV is taken as connected before the run, so a session already
    # connected at slot 0 looks new; ``_record_trading`` skips slot 0 for
    # the warm-up need because that session's plug-in stock is unknown.
    previous_connected = np.zeros((world_count, vehicle_count), dtype=bool)

    # Per-day envelopes over all worlds and EVs let each slot skip the days
    # with no journey or session in it.
    journey_span = _day_spans(
        values["drives_today"], "trip_departure_utc", "return_end_utc", values
    )
    session_span = _day_spans(
        values["connection_session_accepted"], "connection_start_utc", "connection_end_utc", values
    )

    zone_masks = (
        [] if zone_index is None else [zone_index == z for z in range(int(zone_output.shape[1]))]
    )
    zone_columns = [_KERNEL_COLUMNS.index(name) for name in ZONE_SUM_COLUMNS]

    for slot, start in enumerate(boundaries_ns[:-1]):
        end = boundaries_ns[slot + 1]
        opening = stock.copy()
        journey_days = np.flatnonzero((journey_span[0] < end) & (journey_span[1] > start))

        # 1-2. Travel and public top-ups (mutates ``stock``).
        flows, public_battery_added = _travel_in_slot(
            start, end, journey_days, values, leg_energy, top_up, stock
        )
        # Item 32: top-up energy is public grid import in the slot where it
        # happens, battery energy divided by the public charging efficiency.
        public_grid_import = public_battery_added / top_up["efficiency"]

        # 3. Occupancy.
        driving_ns, parked_ns, first_departure = _time_away_in_slot(
            start, end, journey_days, values
        )
        away_ns = np.minimum(end - start, driving_ns + parked_ns)
        home_ns = end - start - away_ns

        # 4. Home connection.
        connected_ns = _home_connected_ns(
            start, end, first_departure, home_ns, values, session_span
        )
        unplugged_ns = home_ns - connected_ns
        # Home charging needs connection for the whole slot: the half-hour is
        # the dispatch unit, with one realised grid power per EV-slot
        # (decision 0001), and a return inside the slot only enables charging
        # from the next boundary (docs/MODEL_CONTRACT.md).  Pro-rating partial
        # slots would claim flexibility the EV cannot offer for the full slot.
        connected_full_slot = connected_ns == end - start

        # 5. Home charging.  The normal rule: charge at once, up to device
        # power and the preferred target.
        home_grid_import = _home_charge_kwh(
            stock,
            connected_full_slot,
            target_stock,
            device_power_kw,
            settings.home_charge_efficiency,
        )
        if charge_allowed is not None and not charge_allowed[slot]:
            # Timed-start rule (decision 0007): this half-hour is barred, so
            # home charging is zero here, full stop -- no fallback and no
            # spill into an allowed half-hour.  Public top-ups above already
            # ran, so a session this leaves short simply tops up publicly
            # more often, which the existing columns already report.
            home_grid_import = np.zeros_like(home_grid_import)
        early_count = early_shortfall = np.zeros_like(stock)
        replanned = None
        if dispatch is not None and dispatch.decision_slot[slot]:
            # Step 0 (intraday dispatch contract v1 §4.1, review B1): free
            # EVs re-plan before anything about this slot is read, so the
            # trader's book just below holds the re-plans made now, ranked
            # on the same latest prices the trader sees.
            replanned = _replan_free_evs(smart, dispatch, plans, slot, start, device_power_kw)
        if trading is not None:
            _record_trading(
                trading,
                slot,
                study_start,
                plans,
                previous_connected,
                connected_full_slot,
                opening,
                stock,
                target_stock,
                settings.home_charge_efficiency,
            )
        previous_connected = connected_full_slot
        plan_record = None
        if plans is not None and smart.price_epoch_by_slot[slot] >= 0:
            # Selected path in the study and the last warm-up night: the smart
            # plan decides how much of that the EV takes in this slot
            # (decision 0004 items 38 and 52).  min() keeps every normal
            # limit, so the plan can only move charging.  A session that
            # ignores its plan charges by the normal rule for the whole
            # session (§10.1e): same physics as the old "full power from
            # plug-in" plan, but the smart plan stays on record for the
            # trader's book and the Firm-MW potential.
            planned, ignores, early_count, early_shortfall, plan_record = _smart_charging_slot(
                smart,
                plans,
                slot,
                start,
                horizon_end_ns,
                connected_full_slot,
                stock,
                target_stock,
                device_power_kw,
                settings.home_charge_efficiency,
                record=unit_output is not None and slot >= study_start,
                dispatch=dispatch,
            )
            home_grid_import = np.where(
                ignores, home_grid_import, np.minimum(home_grid_import, planned)
            )
            plans["remaining_kwh"] -= home_grid_import
        home_battery_added = home_grid_import * settings.home_charge_efficiency
        stock += home_battery_added

        # 6. Physical checks.  Closing stock must equal opening plus energy
        # added minus travel served, per EV (battery-energy conservation).
        served_travel = (
            flows["served_outbound_travel_battery_kwh"] + flows["served_return_travel_battery_kwh"]
        )
        residual = stock - (opening + home_battery_added + public_battery_added - served_travel)
        if (np.abs(residual) > _CHECK_TOLERANCE_KWH).any():
            raise ValueError("per-EV conservation residual exceeds 1e-12 kWh")
        if ((stock < -_CHECK_TOLERANCE_KWH) | (stock > capacity + _CHECK_TOLERANCE_KWH)).any():
            raise ValueError("battery stock is outside physical capacity")

        if trading is not None and "control_mask" in trading:
            control_import = home_grid_import[:, trading["control_mask"]].sum(axis=1)
            if slot < study_start:
                trading["control_warmup_import_kwh"][:, slot] = control_import
            else:
                trading["control_import_kwh"][:, slot - study_start] = control_import
        if slot < study_start:
            if warmup_home_import_kwh is not None:
                warmup_home_import_kwh[:, slot] = home_grid_import.sum(axis=1)
            if warmup_zone_home_import_kwh is not None:
                for z, mask in enumerate(zone_masks):
                    warmup_zone_home_import_kwh[:, z, slot] = home_grid_import[:, mask].sum(axis=1)
            continue
        slot_values = {
            "away_on_trip_fraction": away_ns / _HALF_HOUR_NS,
            "home_unplugged_fraction": unplugged_ns / _HALF_HOUR_NS,
            "home_connected_fraction": connected_ns / _HALF_HOUR_NS,
            "driving_fraction": driving_ns / _HALF_HOUR_NS,
            "parked_away_fraction": parked_ns / _HALF_HOUR_NS,
            # Item 32: the time spent topping up is not modelled, so no slot
            # time is public charging; the column stays for the schema.
            "public_charging_fraction": np.zeros_like(stock),
            "opening_battery_kwh": opening,
            "closing_battery_kwh": stock,
            "physical_capacity_kwh": np.broadcast_to(capacity, stock.shape),
            "realised_grid_kw": (home_grid_import + public_grid_import) / _SLOT_HOURS,
            "home_grid_import_kwh": home_grid_import,
            "home_battery_added_kwh": home_battery_added,
            "public_grid_import_kwh": public_grid_import,
            "public_battery_added_kwh": public_battery_added,
            **flows,
            "v2g_battery_removed_kwh": np.zeros_like(stock),
            "conservation_residual_kwh": residual,
            "early_departure_count": early_count,
            "early_departure_shortfall_kwh": early_shortfall,
        }
        per_ev = np.concatenate(
            (
                np.full((*stock.shape, 1), 1.0),
                np.stack([slot_values[column] for column in _SUM_COLUMNS], axis=-1),
                connected_full_slot[..., np.newaxis],
            ),
            axis=-1,
        )
        study_index = slot - study_start
        if dispatch_output is not None:
            locked = dispatch_output["locked_mask"]
            dispatch_output["locked_home_import_kwh"][:, study_index] = home_grid_import[
                :, locked
            ].sum(axis=1)
            dispatch_output["free_home_import_kwh"][:, study_index] = home_grid_import[
                :, ~locked
            ].sum(axis=1)
            if replanned is not None and "replan_count" in dispatch_output:
                dispatch_output["replan_count"][:, study_index] = replanned
        if unit_output is not None:
            unit_output["home_grid_import_kwh"][:, study_index, :] = home_grid_import
            unit_output["public_grid_import_kwh"][:, study_index, :] = public_grid_import
            unit_output["unserved_travel_battery_kwh"][:, study_index, :] = (
                flows["unserved_outbound_travel_battery_kwh"]
                + flows["unserved_return_travel_battery_kwh"]
            )
            unit_output["opening_battery_kwh"][:, study_index, :] = opening
            unit_output["closing_battery_kwh"][:, study_index, :] = stock
            if plan_record is not None:
                for name, recorded in plan_record.items():
                    unit_output[name][:, study_index, :] = recorded
        # World-first: sum EVs within each world before any across-world step.
        world_output[:, study_index, :] = per_ev.sum(axis=1)
        for cohort_index in range(cohort_count):
            cohort_output[:, cohort_index, study_index, :] = per_ev[
                :, cohort_codes == cohort_index, :
            ].sum(axis=1)
        for z, mask in enumerate(zone_masks):
            zone_output[:, z, study_index, :] = per_ev[:, mask][:, :, zone_columns].sum(axis=1)
    return world_output, cohort_output


def _record_trading(
    trading: dict[str, np.ndarray],
    slot: int,
    study_start: int,
    plans: dict[str, np.ndarray] | None,
    previous_connected: np.ndarray,
    connected_full_slot: np.ndarray,
    opening: np.ndarray,
    stock: np.ndarray,
    target_stock: np.ndarray,
    efficiency: float,
) -> None:
    """Fill the trading overlay's kernel outputs for one slot (trading contract v1 §1.4).

    Called before home charging, so ``stock`` is what a plug-in sees.  Only
    the keys present in ``trading`` are filled; each array is preallocated
    by the caller (``forecast``) and holds kWh:

    - ``warmup_need_kwh`` and ``warmup_session_count`` (world, EV): on the
      normal path in the warm-up, the sum of grid kWh each new home session
      needed at plug-in, ``max(0, target - stock) / efficiency``, and the
      number of sessions (the expected need of §4.3 is their mean).  With
      ``first_day_ahead_cutoff_slot`` (0-d int, the first run slot starting
      at or after ``tau_DA(0)``), ``first_day_ahead_need_kwh`` and
      ``first_day_ahead_session_count`` hold the same sums over the
      sessions that began before it: night 0's day-ahead decision falls
      inside the warm-up, so a later session is not yet known (review B3).
      A session already connected at the first run slot began before the
      run; its plug-in stock is unknown (the opening SoC is an assumption,
      not a plug-in state), so it is not counted.
    - ``departure_shortfall_kwh`` (world, study slot): battery kWh below the
      preferred target at unplug, ``max(0, target - stock)``, summed over
      home sessions ending in the slot (the last slot they were plugged in
      for), for the unmet-charge penalty (§4.7, review B3).
    - ``book_kwh`` (world, decision, slot ahead) with ``book_decision``
      (run slot,) giving each hourly decision slot's row or -1: on the
      selected path, the grid kWh the plans in force at the start of the
      slot hold for this and the following slots.  It is read before the
      slot's early departures and new plans, so it uses only what is known
      at the decision instant (the slot start); an EV plugging in exactly
      then is left to the trader's expected sessions (§4.5, leakage rule).
    - ``connected`` (world, study slot + 1, EV) bool: plugged in at home for
      the whole slot, for the slot before the study and every study slot, so
      the trader knows which sessions have started by a decision (§4.5).
    - ``control_mask`` (EV,) with ``control_warmup_import_kwh`` and
      ``control_import_kwh``: filled after charging by ``_simulate_path``.
    """

    if "connected" in trading and slot >= study_start - 1:
        trading["connected"][:, slot - study_start + 1, :] = connected_full_slot
    new_session = connected_full_slot & ~previous_connected
    if "warmup_need_kwh" in trading and 0 < slot < study_start:
        need = np.where(new_session, np.maximum(target_stock - stock, 0.0) / efficiency, 0.0)
        trading["warmup_need_kwh"] += need
        trading["warmup_session_count"] += new_session
        cutoff = trading.get("first_day_ahead_cutoff_slot")
        if cutoff is not None and slot < cutoff:
            trading["first_day_ahead_need_kwh"] += need
            trading["first_day_ahead_session_count"] += new_session
    if "departure_shortfall_kwh" in trading and slot - 1 >= study_start:
        # The session was plugged in for the whole of slot - 1 and is not for
        # this slot, so it ended; ``opening`` is its stock at unplug.
        ended = previous_connected & ~connected_full_slot
        shortfall = np.where(ended, np.maximum(target_stock - opening, 0.0), 0.0)
        trading["departure_shortfall_kwh"][:, slot - 1 - study_start] += shortfall.sum(axis=1)
    if "book_kwh" in trading and plans is not None and trading["book_decision"][slot] >= 0:
        book = trading["book_kwh"][:, trading["book_decision"][slot], :]
        width = plans["kwh"].shape[2]
        ahead = np.arange(book.shape[1])
        in_force = (plans["start"] <= slot) & (slot < plans["end"])
        offset = (slot - plans["start"])[..., np.newaxis] + ahead
        keep = (
            in_force[..., np.newaxis]
            & (offset < width)
            & (slot + ahead < plans["end"][..., np.newaxis])
        )
        planned = np.take_along_axis(plans["kwh"], np.clip(offset, 0, width - 1), axis=2)
        book[:] = np.where(keep, planned, 0.0).sum(axis=1)


def _day_spans(
    active: np.ndarray, start_key: str, end_key: str, values: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return each day's earliest start and latest end over all worlds and EVs."""

    earliest = np.min(np.where(active, values[start_key], np.iinfo(np.int64).max), axis=(0, 2))
    latest = np.max(np.where(active, values[end_key], np.iinfo(np.int64).min), axis=(0, 2))
    return earliest, latest


def _top_up_levels(public: PublicTopUp, capacity: np.ndarray) -> dict[str, np.ndarray | float]:
    """Return the public efficiency and per-EV top-up threshold and target stock (kWh)."""

    return {
        "efficiency": public.efficiency_fraction,
        "threshold": capacity * public.threshold_soc_fraction,
        "target": capacity * public.target_soc_fraction,
    }


def _travel_in_slot(
    start: int,
    end: int,
    journey_days: np.ndarray,
    values: dict[str, np.ndarray],
    leg_energy: dict[str, np.ndarray],
    top_up: dict[str, np.ndarray],
    stock: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Apply travel and public top-ups for one slot, legs in time order.

    Updates ``stock`` in place.  Returns the travel flows (kWh, keys
    ``_TRAVEL_FLOWS``; unserved flows are always zero) and the public battery
    energy added (kWh).
    """

    flows = {name: np.zeros_like(stock) for name in _TRAVEL_FLOWS}
    public_battery_added = np.zeros_like(stock)
    for day in journey_days:
        journey_active = (
            values["drives_today"][:, day, :]
            & (values["trip_departure_utc"][:, day, :] < end)
            & (values["return_end_utc"][:, day, :] > start)
        )
        # Outbound before return: the return leg sees the stock the outbound
        # leg (and any top-up on it) left.
        for leg, leg_start, leg_end in (
            ("outbound", "trip_departure_utc", "outbound_end_utc"),
            ("return", "return_start_utc", "return_end_utc"),
        ):
            overlap = _overlap_ns(
                start, end, values[leg_start][:, day, :], values[leg_end][:, day, :]
            )
            active = journey_active & (overlap > 0)
            requested = np.zeros_like(stock)
            np.divide(
                leg_energy[leg][:, day, :] * overlap,
                values[f"{leg}_duration_ns"][:, day, :],
                out=requested,
                where=active,
            )
            added = _top_up_battery_kwh(stock, requested, top_up["threshold"], top_up["target"])[0]
            stock += added - requested
            public_battery_added += added
            flows[f"requested_{leg}_travel_battery_kwh"] += requested
            flows[f"served_{leg}_travel_battery_kwh"] += requested
    return flows, public_battery_added


def _top_up_battery_kwh(
    stock: np.ndarray,
    requested: np.ndarray,
    threshold: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return public battery energy added (kWh) and top-up count to serve ``requested``.

    Decision 0004 item 32: when drawing ``requested`` kWh would take stock
    below the threshold, the EV tops up to the target at the moment it
    reaches the threshold (or at once, if it is already below it), then
    carries on.  Each later top-up again runs from threshold to target, so a
    request larger than ``target - threshold`` (even larger than the whole
    battery) gets as many top-ups as it needs.  Every top-up goes to the
    target, never just to the energy still needed: the driver stops and
    fills to the target, as the decision describes.  The alternative, buying
    only the shortfall, would leave the EV at the threshold after each trip
    and trigger a top-up on nearly every later trip.

    The count is ``n = ceil(remaining / (target - threshold))`` where
    ``remaining`` is the request still unserved at the first top-up, so the
    stock after the request lies in ``[threshold, target)``.  A request that
    would end below the threshold by at most 1e-12 kWh (the conservation
    tolerance) is floating-point noise, not a real crossing, and gets none.
    """

    first_top_up_level = np.minimum(stock, threshold)
    remaining = requested - (stock - first_top_up_level)
    needs = (requested > 0.0) & (stock - requested < threshold - _CHECK_TOLERANCE_KWH)
    count = np.where(needs, np.ceil(remaining / (target - threshold)), 0.0)
    added = np.where(
        needs, (target - first_top_up_level) + (count - 1.0) * (target - threshold), 0.0
    )
    return added, count


def _time_away_in_slot(
    start: int,
    end: int,
    journey_days: np.ndarray,
    values: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return driving ns, parked-away ns and first departure in the slot.

    EVs never strand (decision 0004 item 32), so every scheduled journey is
    driven as planned; top-up time is not modelled.
    """

    world_count, _, vehicle_count = values["drives_today"].shape
    shape = (world_count, vehicle_count)
    driving_ns = np.zeros(shape)
    parked_ns = np.zeros(shape)
    first_departure = np.full(shape, _NO_EVENT_INT, dtype=np.int64)
    for day in journey_days:
        departure = values["trip_departure_utc"][:, day, :]
        outbound_end = values["outbound_end_utc"][:, day, :]
        return_start = values["return_start_utc"][:, day, :]
        return_end = values["return_end_utc"][:, day, :]
        overlaps = values["drives_today"][:, day, :] & (departure < end) & (return_end > start)
        np.minimum(first_departure, departure, out=first_departure, where=overlaps)
        driving_ns += np.where(
            overlaps,
            _overlap_ns(start, end, departure, outbound_end)
            + _overlap_ns(start, end, return_start, return_end),
            0.0,
        )
        parked_ns += np.where(overlaps, _overlap_ns(start, end, outbound_end, return_start), 0.0)
    return driving_ns, parked_ns, first_departure


def _home_connected_ns(
    start: int,
    end: int,
    first_departure: np.ndarray,
    home_ns: np.ndarray,
    values: dict[str, np.ndarray],
    session_span: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Return home time covered by an accepted connection session.

    A departure unplugs the EV, so connection counts only up to the first
    departure in the slot; connection can never exceed time at home.
    """

    cutoff = np.minimum(end, first_departure)
    connected_ns = np.zeros(home_ns.shape)
    session_days = np.flatnonzero((session_span[0] < cutoff.max()) & (session_span[1] > start))
    for day in session_days:
        connected_ns += np.where(
            values["connection_session_accepted"][:, day, :],
            _overlap_ns(
                start,
                cutoff,
                values["connection_start_utc"][:, day, :],
                values["connection_end_utc"][:, day, :],
            ),
            0.0,
        )
    return np.minimum(home_ns, connected_ns)


def _home_charge_kwh(
    stock: np.ndarray,
    connected_full_slot: np.ndarray,
    target_stock: np.ndarray,
    device_power_kw: np.ndarray,
    efficiency: float,
) -> np.ndarray:
    """Return home grid import (kWh) for one slot under the normal rule.

    Charging runs at device power until the preferred target:
    ``min(power * 0.5 h, (target - stock) / efficiency)``, and only in a slot
    the EV is connected for in full.
    """

    return np.where(
        connected_full_slot & (stock < target_stock),
        np.minimum(device_power_kw * _SLOT_HOURS, (target_stock - stock) / efficiency),
        0.0,
    )


# --------------------------------------------------------------------------
# Smart charging on the selected path (decision 0004 item 38)
# --------------------------------------------------------------------------
#
# Each (world, EV) holds at most one plan: grid kWh for each half-hour from
# its plug-in slot to its expected departure, made with ``action``'s
# cheapest-first planner.  A plan lives while the EV stays plugged in for
# whole slots.  It is stored as (world, EV, slot within the plan) rather than
# (world, EV, study slot), because a window never spans much more than a
# day: at 1,000 EVs x 100 worlds that is about 40 MB instead of 270 MB.


def _no_plans(smart: SmartCharging, world_count: int, vehicle_count: int) -> dict[str, np.ndarray]:
    """Empty plan state for every (world, EV); ``end`` = -1 means no plan.

    Per plan: its grid kWh per slot from ``start``, the kWh still owed to
    the EV (``remaining_kwh``, for early departures), the need still left
    under the plan (``need_left_kwh``, §10.1e) and its ``plan_status``
    code.  ``decision_*`` hold the copy taken at the night's decision slot.
    """

    # The longest window runs from just after one expected departure to the
    # next: the largest gap between an EV's expected departures (a day, or a
    # day plus an hour across the autumn clock change) plus one slot.
    gaps = np.diff(smart.expected_departure_utc_ns, axis=0)
    width = int(-(-(gaps.max() + _HALF_HOUR_NS) // _HALF_HOUR_NS))
    shape = (world_count, vehicle_count)
    return {
        "kwh": np.zeros((*shape, width)),
        "start": np.zeros(shape, dtype=np.int64),
        "end": np.full(shape, -1, dtype=np.int64),
        "remaining_kwh": np.zeros(shape),
        "need_left_kwh": np.zeros(shape),
        "status": np.zeros(shape),
        "decision_kwh": None,
        "decision_start": np.zeros(shape, dtype=np.int64),
        "decision_end": np.full(shape, -1, dtype=np.int64),
    }


def _plan_status(
    smart: SmartCharging, slot: int, world_index: np.ndarray, ev_index: np.ndarray
) -> np.ndarray:
    """``plan_status`` codes for plans made now by (world_index, ev_index) (§10.1e).

    The session's one uniform ``u`` for its night is compared once with
    each reason, in the contract's order: a control EV (4); ``u < 1 - r``,
    base non-response that would ignore any plan (1); ``u < rho``, which
    past the base rate can only be its maker's outage night (2); ``u`` below
    the share of a scripted control outage covering this plug-in slot and
    zone (3); else it follows (0).  Every code but 0 means ``u < max(rho,
    share)``, the ignore rule, so the label never changes behaviour.  One
    uniform per (world, night, EV) means a worse maker setting only turns
    followers into non-followers, never back (Compare pairs futures).
    Maker and base non-response hold for the whole session because every
    plan of the night reads the same uniform; the event's raise applies per
    plan made, so a replan outside its window returns a code-3 session to 0.
    """

    status = np.zeros(len(world_index))
    if smart.non_response_uniform is None:
        return status
    night = smart.night_by_slot[slot]
    uniform = smart.non_response_uniform[world_index, night, ev_index]
    rho = (
        np.zeros(len(world_index))
        if smart.non_response is None
        else smart.non_response[world_index, night, ev_index]
    )
    base = rho if smart.base_non_response is None else smart.base_non_response[ev_index]
    control = (
        np.zeros(len(world_index), dtype=bool)
        if smart.control_group is None
        else smart.control_group[ev_index]
    )
    share = smart.outage_probability[smart.zone_index[ev_index], slot]
    return np.select(
        [control, uniform < base, uniform < rho, uniform < share],
        [CONTROL_GROUP, BASE_NON_RESPONSE, MAKER_OUTAGE, CONTROL_OUTAGE_EVENT],
        PLAN_FOLLOWS,
    ).astype(np.float64)


def _smart_charging_slot(
    smart: SmartCharging,
    plans: dict[str, np.ndarray],
    slot: int,
    slot_start_ns: int,
    horizon_end_ns: int,
    connected_full_slot: np.ndarray,
    stock: np.ndarray,
    target_stock: np.ndarray,
    device_power_kw: np.ndarray,
    efficiency: float,
    *,
    record: bool = False,
    dispatch: IntradayDispatch | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    """Update the plans for one run slot and return what they allow.

    Returns, per (world, EV): the planned grid kWh for this slot; whether
    the plan in force is ignored (the session charges by the normal rule);
    1.0 where an EV left before its plan finished (else 0.0); that early
    departure's battery-side shortfall (kWh the plan still owed times the
    charging efficiency); and, when ``record`` (a study slot of a per-EV
    run), the §10.1e plan outputs keyed as ``UNIT_INTERVAL_QUANTITIES``.
    Steps, in order:

    1. Early departures.  A plan in force whose EV is no longer plugged in
       for the whole slot has seen the EV leave (a trip or an unplug).  The
       energy it still owed is never delivered, so the EV leaves below
       target.  This is the only place the actual session enters: the plan
       itself was made without it.
    2. New plans.  An EV plugged in for the whole slot with no plan in force
       (it has just plugged in, its window ended while it stayed plugged in,
       or planning has just started on the last warm-up night) is planned
       now from its current stock, this slot, its next expected departure
       and the day-ahead prices published by now (plan B4), plus the price
       adjustment of every grid request already announced for the EV's zone
       (trading contract v1 §2.4), leaving blackout half-hours untouched
       (§10.5b).  Every session is planned this way, including one that
       will ignore its plan (§10.1e), and the plan's status code is fixed
       now (``_plan_status``).  With ``dispatch``, a free EV ranks the
       latest intraday price known at this slot's start instead (intraday
       dispatch contract v1 §4.1 step 2); a locked EV plans as above.
    3. The decision snapshot.  At a night's decision slot ``s_n`` the plans
       in force are copied, so later slots of the night can report the plan
       the aggregator held at the decision (R1).
    4. This slot's share of every plan in force.
    """

    in_plan = (plans["start"] <= slot) & (slot < plans["end"])

    # 1. Early departures.
    left = in_plan & ~connected_full_slot
    owed = np.where(left, plans["remaining_kwh"], 0.0)
    early = owed > NOT_RECOVERED_TOLERANCE_KWH
    plans["end"][left] = -1
    plans["remaining_kwh"][left] = 0.0
    in_plan &= ~left

    # 2. New plans.
    new = connected_full_slot & ~in_plan
    if new.any():
        world_index, ev_index = np.nonzero(new)
        departures = smart.expected_departure_utc_ns
        # The first expected departure at least one slot after this slot
        # starts; the window stops there or at the horizon end, whichever
        # comes first.  Only whole slots before it can charge, hence floor.
        # Energy planned after the horizon end would count as not recovered.
        next_departure = departures[
            np.argmax(departures >= slot_start_ns + _HALF_HOUR_NS, axis=0),
            np.arange(departures.shape[1]),
        ]
        window_end = np.minimum(next_departure, horizon_end_ns)
        window_slots = ((window_end - slot_start_ns) // _HALF_HOUR_NS)[ev_index]

        # Each session ranks its own world's day-ahead prices as published by
        # this slot's start (decision 0004 item 48, plan B4): the price if
        # published, else the expected shape.  Slots past the horizon end are
        # padded with +inf.  The expected departures plans target fall before
        # the noon horizon end for clocked cohorts at the default margin (item
        # 52); a window that runs past the end is clipped there and is still
        # long, so there is no cram and the item 49 rule is no longer needed.
        width = plans["kwh"].shape[2]
        visible = smart.visible_price_gbp_per_mwh[smart.price_epoch_by_slot[slot]]
        upcoming = visible[world_index, slot : slot + width]
        if dispatch is not None:
            free = ~dispatch.locked[ev_index]
            if free.any():
                latest = _latest_known_window(dispatch, visible, slot, slot_start_ns, width)
                upcoming = np.where(free[:, np.newaxis], latest[world_index], upcoming)
        prices = np.full((len(world_index), width), np.inf)
        prices[:, : upcoming.shape[1]] = upcoming
        # Grid requests (trading contract v1 §2.4, decision 0004 item 55): a
        # turn-down adds its payment to the ranking price inside its window
        # (charging there forgoes the payment), a turn-up subtracts its bonus.
        # Only requests announced by this slot count, so a plan never sees a
        # request before its notice (review B2); plans made earlier keep their
        # ranking, as the contract states until short-notice replanning (H6).
        # It changes ranking only: the plan still charges just the need.
        known = smart.event_notice_slot <= slot
        if known.any():
            adjustment = smart.event_adjustment_gbp_per_mwh[known].sum(axis=0)
            upcoming_adjustment = adjustment[smart.zone_index[ev_index], slot : slot + width]
            prices[:, : upcoming_adjustment.shape[1]] += upcoming_adjustment
        window_price = np.where(np.arange(width) < window_slots[:, np.newaxis], prices, np.inf)
        # Grid energy to reach the preferred target from the stock at plug-in.
        need_kwh = np.maximum(target_stock[ev_index] - stock[world_index, ev_index], 0.0)
        need_grid_kwh = need_kwh / efficiency
        cap_kwh = device_power_kw[ev_index] * _SLOT_HOURS
        blackout = None
        if smart.blackout is not None:
            upcoming_blackout = np.zeros(width, dtype=bool)
            window_blackout = smart.blackout[slot : slot + width]
            upcoming_blackout[: len(window_blackout)] = window_blackout
            blackout = upcoming_blackout[np.newaxis, :] & np.isfinite(window_price)
        if blackout is not None and blackout.any():
            planned = plan_around_blackout(need_grid_kwh, cap_kwh, window_price, blackout)
        else:
            planned = plan_cheapest_slots(need_grid_kwh, cap_kwh, window_price)
        plans["kwh"][world_index, ev_index] = planned
        plans["start"][world_index, ev_index] = slot
        plans["end"][world_index, ev_index] = slot + window_slots
        plans["remaining_kwh"][world_index, ev_index] = planned.sum(axis=1)
        plans["need_left_kwh"][world_index, ev_index] = need_grid_kwh
        plans["status"][world_index, ev_index] = _plan_status(smart, slot, world_index, ev_index)
        in_plan |= new

    # 3. The decision snapshot (§10.1e R1): the plans in force at s_n, only
    # for per-EV runs, which are the only ones that report it.
    night = smart.night_by_slot[slot]
    decision_slot = (
        None if smart.decision_slot_by_night is None else smart.decision_slot_by_night[night]
    )
    if record and slot == decision_slot:
        plans["decision_kwh"] = plans["kwh"].copy()
        plans["decision_start"] = plans["start"].copy()
        plans["decision_end"] = np.where(in_plan, plans["end"], -1)

    # 4. This slot's planned energy.
    offset = np.clip(slot - plans["start"], 0, plans["kwh"].shape[2] - 1)
    planned_now = np.take_along_axis(plans["kwh"], offset[..., np.newaxis], axis=2)[..., 0]
    planned_now = np.where(in_plan, planned_now, 0.0)
    ignores = in_plan & (plans["status"] != PLAN_FOLLOWS)
    plan_record = None
    if record:
        decision_plan = np.zeros_like(planned_now)
        snapshot = plans["decision_kwh"]
        if decision_slot is not None and slot >= decision_slot and snapshot is not None:
            in_decision = (plans["decision_start"] <= slot) & (slot < plans["decision_end"])
            decision_offset = np.clip(slot - plans["decision_start"], 0, plans["kwh"].shape[2] - 1)
            decision_plan = np.where(
                in_decision,
                np.take_along_axis(snapshot, decision_offset[..., np.newaxis], axis=2)[..., 0],
                0.0,
            )
        plan_record = {
            "planned_home_import_kwh": planned_now,
            "plan_remaining_need_kwh": np.where(
                in_plan, np.maximum(plans["need_left_kwh"], 0.0), 0.0
            ),
            "decision_plan_kwh": decision_plan,
            "plan_status": np.where(in_plan, plans["status"], PLAN_FOLLOWS),
        }
    # The need under the plan falls by exactly the plan's own kWh, slot by
    # slot, whatever the EV actually imports (R1: the aggregator's view).
    plans["need_left_kwh"] -= planned_now
    return (
        planned_now,
        ignores,
        early.astype(np.float64),
        np.where(early, owed * efficiency, 0.0),
        plan_record,
    )


def _latest_known_window(
    dispatch: IntradayDispatch, visible: np.ndarray, slot: int, slot_start_ns: int, width: int
) -> np.ndarray:
    """``action.latest_known_prices`` at this slot's start for run slots [slot, slot + width).

    ``visible`` (world, run slot) is the B4-visible day-ahead price of the
    slot's epoch.  Returns (world, up to ``width``) GBP/MWh; fewer columns
    at the end of the horizon, as ``visible[:, slot : slot + width]``.
    """

    window = slice(slot, slot + width)
    return latest_known_prices(
        dispatch.intraday_path_gbp_per_mwh[:, window],
        visible[:, window],
        dispatch.publication_utc_ns[window],
        dispatch.gate_utc_ns[window],
        slot_start_ns,
    )


def _replan_free_evs(
    smart: SmartCharging,
    dispatch: IntradayDispatch,
    plans: dict[str, np.ndarray],
    slot: int,
    slot_start_ns: int,
    device_power_kw: np.ndarray,
) -> np.ndarray:
    """Step 0 of a decision slot: re-plan free EVs when it is worth it (contract §4.1).

    Changes ``plans["kwh"]`` in place and returns the number of re-plans per
    world (world,) int64.  A candidate has a plan in force at the slot start,
    is free, follows its plan (``plan_status`` 0: a non-responder's maker
    is not listening, so it is never re-planned, B6), still has planned
    energy from this slot on (B5: the plan's own kWh, not the counter that
    actual import decrements) and at least two slots left, so energy can
    move.  Nothing about connection in this slot is read: whether the EV
    stays plugged in through it is not known at the slot start, so a
    candidate leaving now is re-planned in vain and voided at step 1.

    Each candidate's slots rank ``latest_known_prices`` at the slot start
    plus the request adjustments announced by now for its zone (so a spike,
    a DFS payment and a cheap half-hour compete on one scale, §4.2);
    slots before this one, outside the window and in a blackout rank
    ``+inf`` and keep their energy (``replan_when_worth``).  A re-plan keeps
    the plan's window, status and remaining total: it is not a "plan made"
    (B6), so it resolves no non-response and reads no uniform.
    """

    world_count = plans["end"].shape[0]
    width = plans["kwh"].shape[2]
    candidate = (
        (plans["start"] <= slot)
        & (slot < plans["end"])
        & ~dispatch.locked[np.newaxis, :]
        & (plans["status"] == PLAN_FOLLOWS)
        & (plans["end"] - slot >= 2)
    )
    world_index, ev_index = np.nonzero(candidate)
    ahead = np.arange(width)
    # Plan offsets of this slot and of the window end, per candidate.
    first = (slot - plans["start"])[world_index, ev_index]
    last = (plans["end"] - plans["start"])[world_index, ev_index]
    open_slots = (ahead >= first[:, np.newaxis]) & (ahead < last[:, np.newaxis])
    incumbent = plans["kwh"][world_index, ev_index]
    has_energy = np.where(open_slots, incumbent, 0.0).sum(axis=1) > 0.0
    if not has_energy.any():
        return np.zeros(world_count, dtype=np.int64)
    world_index, ev_index = world_index[has_energy], ev_index[has_energy]
    first, open_slots, incumbent = first[has_energy], open_slots[has_energy], incumbent[has_energy]

    # Prices of run slots [slot, slot + width), padded past the horizon:
    # +inf for the price, 0 for the adjustment, False for the blackout.
    visible = smart.visible_price_gbp_per_mwh[smart.price_epoch_by_slot[slot]]
    latest = _latest_known_window(dispatch, visible, slot, slot_start_ns, width)
    price = np.full((world_count, width), np.inf)
    price[:, : latest.shape[1]] = latest
    zone_count = smart.outage_probability.shape[0]
    adjustment = np.zeros((zone_count, width))
    known = smart.event_notice_slot <= slot
    if known.any():
        announced = smart.event_adjustment_gbp_per_mwh[known].sum(axis=0)[:, slot : slot + width]
        adjustment[:, : announced.shape[1]] = announced
    blackout = np.zeros(width, dtype=bool)
    if smart.blackout is not None:
        upcoming = smart.blackout[slot : slot + width]
        blackout[: len(upcoming)] = upcoming

    # Plan offset k is run slot start + k, i.e. slot + (k - first).
    since_now = np.clip(ahead - first[:, np.newaxis], 0, width - 1)
    ranking = (
        price[world_index[:, np.newaxis], since_now]
        + adjustment[smart.zone_index[ev_index][:, np.newaxis], since_now]
    )
    ranking = np.where(open_slots & ~blackout[since_now], ranking, np.inf)
    new, replanned = replan_when_worth(
        incumbent,
        ranking,
        device_power_kw[ev_index] * _SLOT_HOURS,
        replan_threshold_gbp_per_mwh=dispatch.replan_threshold_gbp_per_mwh,
        half_spread_gbp_per_mwh=dispatch.half_spread_gbp_per_mwh,
    )
    plans["kwh"][world_index[replanned], ev_index[replanned]] = new[replanned]
    return np.bincount(world_index[replanned], minlength=world_count).astype(np.int64)


def _overlap_ns(
    interval_start: int,
    interval_end: np.ndarray | int,
    event_start: np.ndarray,
    event_end: np.ndarray,
) -> np.ndarray:
    """Return the length (ns, microsecond resolution) of two intervals' overlap."""

    return _total_seconds_ns(
        np.maximum(
            0.0, np.minimum(interval_end, event_end) - np.maximum(interval_start, event_start)
        )
    )


def _total_seconds_ns(duration_ns: np.ndarray) -> np.ndarray:
    """Match ``Timedelta.total_seconds()`` microsecond resolution in nanoseconds."""

    return np.floor_divide(duration_ns, 1_000) * 1_000.0


# --------------------------------------------------------------------------
# Output frames
# --------------------------------------------------------------------------


def _interval_frame(
    values_by_path: dict[str, np.ndarray],
    starts: np.ndarray,
    *,
    cohort_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return per-world interval totals for both paths as one sorted frame.

    ``values_by_path`` maps path_id to kernel sums of shape
    (world, study slot, column), or (world, cohort, study slot, column) when
    ``cohort_ids`` is given.
    """

    frames = []
    for path_id, values in values_by_path.items():
        if cohort_ids is None:
            values = values[:, np.newaxis]
        world_count, cohort_count, slot_count, _ = values.shape
        frame = pd.DataFrame(
            values.reshape(world_count * cohort_count * slot_count, -1), columns=_KERNEL_COLUMNS
        )
        repeats = world_count * cohort_count
        frame.insert(
            0,
            "interval_end_utc",
            pd.to_datetime(np.tile(starts + _HALF_HOUR_NS, repeats), utc=True),
        )
        frame.insert(0, "interval_start_utc", pd.to_datetime(np.tile(starts, repeats), utc=True))
        if cohort_ids is not None:
            frame.insert(0, "cohort_id", np.tile(np.repeat(cohort_ids, slot_count), world_count))
        frame.insert(0, "world_id", np.repeat(np.arange(world_count), cohort_count * slot_count))
        frame["unit_count"] = frame["unit_count"].astype(np.int64)
        frame["connected_count"] = frame["connected_count"].astype(np.int64)
        frame["closing_soc_percent"] = np.where(
            frame["physical_capacity_kwh"] > 0.0,
            100.0 * frame["closing_battery_kwh"] / frame["physical_capacity_kwh"],
            np.nan,
        )
        frame["connected_share"] = frame["connected_count"] / frame["unit_count"]
        frame.insert(1, "path_id", path_id)
        frames.append(frame)
    columns = _INTERVAL_COLUMNS if cohort_ids is None else _COHORT_INTERVAL_COLUMNS
    keys = _KEYS if cohort_ids is None else _COHORT_KEYS
    return (
        pd.concat(frames, ignore_index=True)
        .loc[:, columns]
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )


def _band_frame(
    values_by_path: dict[str, np.ndarray],
    starts: np.ndarray,
    *,
    cohort_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return across-world mean and P10/P50/P90 per interval for both paths.

    Statistics are taken over the world axis of per-world totals, so each band
    describes the spread of whole-fleet (or whole-cohort) outcomes across
    simulated worlds.  SoC and connected share are derived per world before
    the percentiles, never from summed percentiles.
    """

    capacity_index = _KERNEL_COLUMNS.index("physical_capacity_kwh")
    closing_index = _KERNEL_COLUMNS.index("closing_battery_kwh")
    connected_index = _KERNEL_COLUMNS.index("connected_count")
    frames = []
    for path_id, values in values_by_path.items():
        world_count = values.shape[0]
        closing_soc = np.where(
            values[..., capacity_index] > 0.0,
            100.0 * values[..., closing_index] / values[..., capacity_index],
            np.nan,
        )
        connected_share = values[..., connected_index] / values[..., 0]
        metrics = np.concatenate(
            (values, closing_soc[..., np.newaxis], connected_share[..., np.newaxis]), axis=-1
        )
        quantiles = np.quantile(metrics, (0.1, 0.5, 0.9), axis=0, method="linear")
        statistics = np.stack(
            (metrics.mean(axis=0), quantiles[0], quantiles[1], quantiles[2]), axis=-1
        )
        slot_count = len(starts)
        cohort_count = 1 if cohort_ids is None else len(cohort_ids)
        frame = pd.DataFrame(
            statistics.reshape(cohort_count * slot_count, -1), columns=_BAND_STATISTICS
        )
        frame.insert(0, "world_count", world_count)
        frame.insert(
            0,
            "interval_end_utc",
            pd.to_datetime(np.tile(starts + _HALF_HOUR_NS, cohort_count), utc=True),
        )
        frame.insert(
            0, "interval_start_utc", pd.to_datetime(np.tile(starts, cohort_count), utc=True)
        )
        frame.insert(0, "path_id", path_id)
        if cohort_ids is not None:
            frame.insert(0, "cohort_id", np.repeat(cohort_ids, slot_count))
        frames.append(frame)
    columns = _BAND_COLUMNS if cohort_ids is None else _COHORT_BAND_COLUMNS
    keys = _BAND_KEYS if cohort_ids is None else _COHORT_BAND_KEYS
    return (
        pd.concat(frames, ignore_index=True)
        .loc[:, columns]
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    timestamp_columns = {"interval_start_utc", "interval_end_utc"}
    integer_columns = {"world_id", "unit_count", "connected_count", "world_count"}
    object_columns = {"cohort_id", "path_id"}
    series = {}
    for column in columns:
        if column in timestamp_columns:
            dtype = "datetime64[ns, UTC]"
        elif column in integer_columns:
            dtype = "int64"
        elif column in object_columns:
            dtype = "object"
        else:
            dtype = "float64"
        series[column] = pd.Series(dtype=dtype)
    return pd.DataFrame(series, columns=columns)


__all__ = [
    "PublicTopUp",
    "UNIT_INTERVAL_QUANTITIES",
    "effective_driving_efficiency",
    "public_top_up",
    "simulate_fleet_intervals",
    "simulate_unit_intervals",
]
