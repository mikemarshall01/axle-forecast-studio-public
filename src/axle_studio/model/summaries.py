"""Summary frames for result contract v2 (task M5, `docs/contracts/results-v2.md`).

What this owns: every frame the dashboard reads that is *derived* after a run
rather than produced by the physics kernel itself.

- ``build_summaries`` is the one entry point ``run_forecast`` calls.  It
  returns the average-day bands (contract 3.8), sampled plug-in events and
  their summary (3.9, 3.10), the plug-in heatmap by weekday and hour
  (3.10a), the plug-in and plug-out heatmaps by weekday and half-hour
  (3.10b), the per-archetype session distributions (3.10c), the
  per-world plug-in KPI table, the archetype table (3.3), the across-EV
  fleet week bands (3.6b), the
  flexibility bands (3.6c, decision 0004 item 43), deferrable power by
  slack and the weekly flexibility figures (3.6d, 3.6e, item 54) and, for
  action results, the paired difference frames (4.1, 4.2), the action
  summary (4.3), the cost summary (4.9), the not-recovered magnitude (4.9a),
  the smart-charging outcome frames (4.4-4.6), each week's own peak
  home import and coincidence factor (4.5a), and the relative day-ahead
  price and cheapest half-hour summaries (4.7a, 4.7b).
- ``build_study_slots`` gives the London labels of the 336 study slots (3.1),
  keyed by session night (decision 0004 item 52); ``build_warmup_slots``
  gives the same keys for the warm-up.
- ``compare_runs`` pairs two runs world by world (section 6), from full
  results or the slim ``RunSummary`` records run history keeps (``slim_run``).
- ``ReplayState`` is what a result keeps so this module and
  ``model/individual.py`` can re-run the kernel on a slice of EVs or worlds.

How it fits: the kernel (``physics``) sums EVs inside each
world because the across-world bands only need fleet sums.  Two summaries need
per-EV detail the fleet sums hide: SoC spread *across EVs* and plug-in
sessions.  Rather than keep every EV's half-hours for every world (1,000 EVs x
100 worlds x 336 slots x several quantities is gigabytes), this module re-runs
the kernel's public ``simulate_unit_intervals`` on chunks of at most
``WORLD_CHUNK_SIZE`` worlds, reduces each chunk to small per-world statistics
and drops the per-EV arrays before the next chunk.

Size strategy (lead decision, 28 September 2026):

- Per-world statistics (connected share, SoC mean/P5/P95 across EVs, plug-in
  counts by hour and SoC bin, medians) are kept for **every** world, so every
  quantile is still over all worlds.
- Full plug-in event rows are kept for at most ``PLUG_IN_EVENT_WORLD_LIMIT``
  worlds, the representative world first.  They are a labelled sample for the
  Data expander and scripts; no chart statistic is computed from the sample.
- A compact numeric table of every plug-in (world, cohort, day type, hour,
  SoC; about 20 bytes a row) is built in the chunk loop to compute the
  per-world statistics, then discarded.

Rules kept throughout: worlds first (every "P10/P50/P90 across worlds" is
``numpy.quantile(..., method="linear")`` over values already reduced inside
each world); differences are taken per world before quantiles (decision 0004
item 12); missing is NaN, never zero (contract rule 8).
"""

from __future__ import annotations

import datetime as dt
import math
import warnings
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from axle_studio.model import availability, household, market, product, supplier
from axle_studio.model import events as model_events
from axle_studio.model.action import (
    ACTION_ID,
    ACTION_LABEL,
    NOT_RECOVERED_TOLERANCE_KWH,
    IntradayDispatch,
    SmartCharging,
    expected_departures_utc_ns,
)
from axle_studio.model.assumptions import ZONE_IDS, ZONE_LABELS
from axle_studio.model.clock import STUDY_START_LOCAL_HOUR, session_night_dates
from axle_studio.model.physics import PublicTopUp, simulate_unit_intervals
from axle_studio.model.sampling import (
    SUPPLY_CURVE_INPUTS,
    MarketPrices,
    supply_curve_gbp_per_mwh,
    trapezoid_weight,
)
from axle_studio.model.settings import RunSettings

if TYPE_CHECKING:
    # forecast.py imports this module, so the result type is for hints only.
    from axle_studio.model.forecast import ForecastResult

LONDON = "Europe/London"
HALF_HOUR = pd.Timedelta(minutes=30)
_HALF_HOUR_NS = 30 * 60 * 1_000_000_000
_NS_PER_HOUR = 3600 * 1_000_000_000
_NAT_NS = np.iinfo(np.int64).min

WORLD_CHUNK_SIZE = 10
"""Worlds per per-EV kernel pass; bounds per-EV arrays to ~27 MB each at 1,000 EVs."""
PLUG_IN_EVENT_WORLD_LIMIT = 10
"""Worlds whose full plug-in event rows are kept (representative world first)."""

# Category orders (contract section 1, rule 9).
PATH_ORDER = ("normal", "selected")
"""The strictly pairwise category order every normal-vs-selected consumer in this module
loops over by name (item 6): never extended to the optional timed path, so adding "timed"
to a frame elsewhere can never change one of these consumers' output."""
_OPTIONAL_PATH_ORDER = ("normal", "selected", "timed")
"""Candidate order for a frame that may optionally carry the timed path (decision 0007,
model step 2); ``_present_paths`` filters it to the paths a given frame actually has."""
DAY_TYPE_ORDER = ("weekday", "weekend", "all")
LOCATION_ORDER = ("home_plugged", "home_unplugged", "driving", "away", "public_charging")


def _present_paths(frame: pd.DataFrame) -> list[str]:
    """``_OPTIONAL_PATH_ORDER`` filtered to the paths ``frame["path_id"]`` actually has.

    Used by the handful of frames that gain an optional "timed" row set
    (weekly peak, price-band shift, zone import bands and summary): a run
    without the timed path has only "normal" and "selected" here, exactly as
    before model step 2; a run with it gets the third row too, in display
    order.  Every other, strictly pairwise frame keeps ``PATH_ORDER`` itself.
    """

    present = set(frame["path_id"])
    return [path for path in _OPTIONAL_PATH_ORDER if path in present]


FLEET_METRIC_UNITS = {
    "connected_count": "EVs",
    "connected_share": "fraction",
    "home_import_kwh": "kWh per half-hour",
    "home_import_kw": "kW",
    "public_import_kwh": "kWh per half-hour",
    "total_import_kw": "kW",
    "closing_battery_kwh": "kWh",
    "battery_soc_percent": "percent",
    "driving_share": "fraction",
    "away_share": "fraction",
    "public_charging_share": "fraction",
    "unserved_travel_kwh": "kWh per half-hour",
}
WEEKLY_DIFFERENCE_UNITS = {
    "home_import_kwh": "kWh per week",
    "public_import_kwh": "kWh per week",
    "unserved_travel_kwh": "kWh per week",
    "closing_battery_kwh": "kWh",
}
PLUG_KPI_UNITS = {
    "plug_ins_per_week": "plug-ins",
    "plug_ins_per_ev_per_week": "plug-ins per EV",
    "median_plug_in_soc_percent": "percent",
    "share_below_10_percent_soc": "fraction",
}
COST_COMPONENTS = (
    ("home_import_cost", "illustrative_selected_minus_normal_energy_cost_gbp"),
    ("public_charge_cost", "illustrative_selected_minus_normal_public_charge_cost_gbp"),
    ("unrecovered_energy_value", "illustrative_unrecovered_energy_value_gbp"),
    ("unserved_travel_value", "illustrative_unserved_travel_value_gbp"),
    ("total", "illustrative_selected_minus_normal_total_gbp"),
)
SETTINGS_KEYS = (
    "model",
    "start_local_date",
    "warmup_days",
    "study_days",
    "vehicle_count",
    "seed",
    "evaluation_world_count",
)
PLUG_IN_EVENT_COLUMNS = [
    "world_id",
    "path_id",
    "unit_id",
    "cohort_id",
    "event_index",
    "plug_in_utc",
    "plug_in_london",
    "local_date",
    "day_type",
    "plug_in_local_hour",
    "plug_in_soc_percent",
    "plug_out_utc",
    "plug_out_soc_percent",
    "home_import_kwh",
    "battery_added_kwh",
    "still_plugged_at_horizon_end",
]
_SLOT_KEY_COLUMNS = [
    "slot_index",
    "interval_start_utc",
    "interval_end_utc",
    "interval_start_london",
]
_BAND_STATS = ["world_count", "mean", "p10", "p50", "p90"]
_SOC_BIN_WIDTH = 5
_SOC_BIN_COUNT = 20


# --------------------------------------------------------------------------
# What a result keeps for re-running the kernel
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SampledWorlds:
    """Sampled inputs of one set of evaluation worlds.

    Array axes are (world, day incl. warm-up, EV).
    """

    inputs: dict[str, np.ndarray]
    daily_temperature_c: np.ndarray
    effective_efficiency_miles_per_battery_kwh: np.ndarray


@dataclass(frozen=True, eq=False)
class ReplayState:
    """The kernel inputs a result keeps so summaries and replays can re-run it.

    ``units`` is the kernel's unit frame (``population_id``, ``unit_id``,
    ``cohort_id`` and the physical columns), in the order the run used.
    ``evaluation`` holds the sampled evaluation worlds and ``public_top_up``
    the run's top-up rule (decision 0004 item 32).  ``smart_charging`` holds
    the smart charger's decision-time inputs for the selected path (decision
    0004 item 38) and is ``None`` for a no-action run.  The UI never reads
    this object (contract 2, ``replay_state``).

    ``replay_cache`` holds one-EV replays for this result only (audit O6), so
    clicking back to an EV does not re-run the kernel.  ``eq=False`` keeps
    identity hashing: two results never share a cache.

    ``flexibility`` is the run's ``FlexibilityInputs`` (``None`` without a
    departure margin) and ``availability_inputs`` the firm-MW inputs
    (``None`` until lanes J1a/J1b wire them), kept so the household card
    re-derives one EV's availability without rebuilding them (household
    contract v1 §4 hunk 4).

    ``intraday_dispatch`` is the run's intraday dispatcher inputs
    (intraday-dispatch-v1 §5.1; ``None`` with the switch off or on a
    no-action run): every re-run of the selected path dispatches too, so
    per-EV frames, the One EV replay and the fleet frames reconcile (§5.2).

    ``timed_start_allowed`` is the run's (run slot,) bool "may charge" mask
    for the optional timed path (decision 0007, model step 2,
    ``action.timed_start_allowed``), or ``None`` when the run's
    ``timed_start_local_hour`` is unset: every re-run of the timed path (the
    per-EV chunk pass, the One EV replay) reuses this one mask, so it never
    recomputes the barred window itself.
    """

    settings: RunSettings
    units: pd.DataFrame
    evaluation: SampledWorlds
    public_top_up: PublicTopUp
    smart_charging: SmartCharging | None = None
    intraday_dispatch: IntradayDispatch | None = None
    flexibility: FlexibilityInputs | None = None
    availability_inputs: availability.AvailabilityInputs | None = None
    timed_start_allowed: np.ndarray | None = None
    replay_cache: dict = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class KernelSlice:
    """Kernel arguments for a subset of worlds and EVs of one run.

    EVs are physically independent in this model (there is no shared site
    cap), so running the kernel on a subset of EVs gives exactly those EVs'
    outcomes in the full run.  That is what makes chunked summaries and the
    one-EV replay (decision 0004 item 20) exact rather than approximate.
    """

    settings: RunSettings
    units: pd.DataFrame
    inputs: dict[str, np.ndarray]
    effective_efficiency: np.ndarray
    public_top_up: PublicTopUp
    smart_charging: SmartCharging | None
    intraday_dispatch: IntradayDispatch | None = None
    charge_allowed: np.ndarray | None = None


def kernel_slice(
    state: ReplayState,
    world_index: Sequence[int] | np.ndarray,
    ev_index: Sequence[int] | np.ndarray,
    *,
    path_id: str,
) -> KernelSlice:
    """Slice the stored kernel inputs to the given worlds and EVs, for one path.

    ``path_id`` is ``"normal"``, ``"selected"`` or ``"timed"``.  ``"normal"``
    gives the unmanaged rule only; ``"selected"`` also carries the run's
    smart-charging inputs for those EVs, so the kernel adds the paired
    selected path (``None`` on a no-action run, which has no selected path),
    and the intraday dispatcher's inputs when the run dispatched, so that
    selected path is the dispatched one (intraday-dispatch-v1 §5.2);
    ``"timed"`` carries the run's stored ``timed_start_allowed`` mask
    instead (decision 0007, model step 2; ``None`` when the run's
    ``timed_start_local_hour`` is unset, so a caller must not ask for
    "timed" then).  The mask is per run slot, not per world or EV, so
    slicing never changes it.
    """

    worlds = np.asarray(world_index, dtype=np.int64)
    evs = np.asarray(ev_index, dtype=np.int64)
    evaluation = state.evaluation
    smart = state.smart_charging
    selected = path_id == "selected"
    return KernelSlice(
        settings=replace(
            state.settings, evaluation_world_count=len(worlds), vehicle_count=len(evs)
        ),
        units=state.units.iloc[evs].reset_index(drop=True),
        inputs={name: values[worlds][:, :, evs] for name, values in evaluation.inputs.items()},
        effective_efficiency=evaluation.effective_efficiency_miles_per_battery_kwh[worlds][
            :, :, evs
        ],
        public_top_up=state.public_top_up,
        smart_charging=smart.for_slice(worlds, evs) if selected and smart is not None else None,
        intraday_dispatch=(
            state.intraday_dispatch.for_slice(worlds, evs)
            if selected and state.intraday_dispatch is not None
            else None
        ),
        charge_allowed=state.timed_start_allowed if path_id == "timed" else None,
    )


# --------------------------------------------------------------------------
# Study slots and scalar horizon fields
# --------------------------------------------------------------------------


def build_study_slots(
    start_local_date: date,
    study_days: int = 7,
    *,
    holiday_dates: Collection[date] = (),
    blackout_labels: Collection[str] = (),
) -> pd.DataFrame:
    """Contract 3.1: one row per fixed UTC half-hour of the study, with London labels.

    The study is ``48 * study_days`` fixed UTC slots from London 12:00 on the
    start date (decision 0004 items 1 and 52).  Reporting days are session
    nights: ``local_date`` is the London date of the slot start minus 12 h
    (``clock.session_night_dates``), so Monday 12:00 to Tuesday 11:30 is
    Monday's night, and ``night_index`` counts nights from 0.  ``day_type``
    and ``day_label`` follow the night's date, so a Friday night's small
    hours are a weekday night.  A clock change inside the study gives one
    night 46 or 50 slots; after a spring change the last slots pass London
    noon on the day after the last night and count to the last night, so
    there are always ``study_days`` nights.  ``local_half_hour`` stays the
    London wall-clock half-hour, which average-day views fold on.

    Firm MW (trading contract v1 §10.1c, §10.5b): ``holiday`` is True on the
    slots of a session night whose date is in ``holiday_dates`` (the dates
    ``clock.holiday_flags`` marks; holidays move only the plug-in skip
    share), and ``blackout`` on the slots whose ``local_time_label`` is in
    ``blackout_labels`` (the validated blackout windows' half-hours).
    """

    return _slot_frame(
        start_local_date,
        0,
        48 * study_days,
        (0, study_days - 1),
        holiday_dates=holiday_dates,
        blackout_labels=blackout_labels,
    )


def build_warmup_slots(
    start_local_date: date,
    warmup_days: int,
    *,
    holiday_dates: Collection[date] = (),
    blackout_labels: Collection[str] = (),
) -> pd.DataFrame:
    """The warm-up slots with the same London keys as ``build_study_slots``.

    ``48 * warmup_days`` fixed UTC slots counted back from the study start,
    ``slot_index`` -48 * warmup_days ... -1 and ``night_index`` -warmup_days
    ... -1 (a clock change in the warm-up clips its stray slots to the nearest
    warm-up night).  The trading baseline reads the unmanaged warm-up import
    by these keys (decision 0004 item 52, trading contract v1 section 4.1);
    nothing is displayed from them.  ``holiday`` and ``blackout`` as in
    ``build_study_slots`` (the last warm-up night plans too, so its
    blackout half-hours matter).
    """

    return _slot_frame(
        start_local_date,
        -48 * warmup_days,
        48 * warmup_days,
        (-warmup_days, -1),
        holiday_dates=holiday_dates,
        blackout_labels=blackout_labels,
    )


def _slot_frame(
    start_local_date: date,
    first_slot: int,
    slot_count: int,
    night_range: tuple[int, int],
    *,
    holiday_dates: Collection[date] = (),
    blackout_labels: Collection[str] = (),
) -> pd.DataFrame:
    """Slot keys and session-night labels for ``slot_count`` slots from ``first_slot``.

    Slot 0 starts at London noon on the start date.  Nights outside
    ``night_range`` (only the one or two slots a clock change pushes past a
    boundary) are clipped into it.
    """

    horizon_start = pd.Timestamp(
        dt.datetime.combine(start_local_date, dt.time(STUDY_START_LOCAL_HOUR))
    ).tz_localize(LONDON)
    first = horizon_start.tz_convert("UTC") + first_slot * HALF_HOUR
    starts = pd.date_range(first, periods=slot_count, freq=HALF_HOUR).as_unit("ns")
    london = starts.tz_convert(LONDON)
    nights = session_night_dates(starts) - np.datetime64(start_local_date, "D")
    night_index = np.clip(nights.astype(np.int64), *night_range)
    local_date = [start_local_date + timedelta(days=int(n)) for n in night_index]
    labels = list(london.strftime("%H:%M"))
    holidays, blackouts = set(holiday_dates), set(blackout_labels)
    return pd.DataFrame(
        {
            "slot_index": np.arange(first_slot, first_slot + slot_count, dtype=np.int64),
            "interval_start_utc": starts,
            "interval_end_utc": starts + HALF_HOUR,
            "interval_start_london": london,
            "local_date": pd.Series(local_date, dtype=object),
            "night_index": night_index,
            "day_type": pd.Series([_day_type(value) for value in local_date], dtype=object),
            # Wall-clock half-hour: both copies of the repeated autumn hour map
            # to the same index, and the skipped spring hour has no slot.
            "local_half_hour": np.asarray(london.hour * 2 + london.minute // 30, dtype=np.int64),
            "local_time_label": pd.Series(labels, dtype=object),
            "day_label": pd.Series([_day_label(value) for value in local_date], dtype=object),
            "holiday": np.array([value in holidays for value in local_date], dtype=bool),
            "blackout": np.array([label in blackouts for label in labels], dtype=bool),
        }
    )


def _day_type(value: date) -> str:
    return "weekday" if value.weekday() < 5 else "weekend"


def _day_label(value: date) -> str:
    return f"{value:%a} {value.day}"


# --------------------------------------------------------------------------
# Per-EV kernel passes
# --------------------------------------------------------------------------


def _utc_ns(values: np.ndarray) -> np.ndarray:
    """UTC instants of an input array as int64 nanoseconds (NaT -> int64 min).

    Sampled inputs are ``datetime64[ns]`` UTC already and are read as they
    are; object arrays of Timestamps (hand-built cases) are converted.
    """

    if values.dtype.kind == "M":
        return values.astype("datetime64[ns]").view(np.int64)
    index = pd.DatetimeIndex(pd.to_datetime(values.ravel(), utc=True)).as_unit("ns")
    return index.asi8.reshape(values.shape)


def _home_connected(inputs: dict[str, np.ndarray], slot_start_ns: np.ndarray) -> np.ndarray:
    """(world, slot, EV): plugged in at home for the whole slot.

    ``simulate_unit_intervals`` returns energy flows but not the connection
    state, so this re-derives the kernel's ``connected_full_slot`` from the
    same inputs: an accepted home session covers the whole slot and no trip
    overlaps it (the kernel cuts a session at the first departure and counts
    trip time as away).  EVs never strand (decision 0004 item 32: a trip that
    would go below the top-up threshold tops up in public first), so the
    connection state depends only on these inputs and is the same on both
    paths.  ``_per_ev_chunks`` checks the result against the fleet
    ``connected_count`` for every world and slot, so any drift from the kernel
    fails loudly instead of skewing plug-in statistics.
    """

    slot_end_ns = slot_start_ns + _HALF_HOUR_NS
    start = slot_start_ns[np.newaxis, :, np.newaxis]
    end = slot_end_ns[np.newaxis, :, np.newaxis]
    accepted = inputs["connection_session_accepted"]
    session_start = _utc_ns(inputs["connection_start_utc"])
    session_end = _utc_ns(inputs["connection_end_utc"])
    drives = inputs["drives_today"]
    departure = _utc_ns(inputs["trip_departure_utc"])
    # Journey end exactly as the kernel computes it: each leg and the dwell are
    # truncated to whole nanoseconds before adding, so slot overlaps match.
    speed = inputs["drive_speed_mph"].astype(float)
    outbound_ns = np.trunc(3600.0 * inputs["outbound_miles"] / speed * 1e9)
    return_ns = np.trunc(3600.0 * inputs["return_miles"] / speed * 1e9)
    dwell_ns = np.trunc(np.where(drives, inputs["destination_dwell_seconds"], 0.0) * 1e9)
    journey_ns = np.where(drives, outbound_ns, 0).astype(np.int64)
    journey_ns += dwell_ns.astype(np.int64) + np.where(drives, return_ns, 0).astype(np.int64)
    return_end = np.where(drives, departure + journey_ns, _NAT_NS)

    world_count, day_count, ev_count = accepted.shape
    covered = np.zeros((world_count, len(slot_start_ns), ev_count), dtype=bool)
    away = np.zeros_like(covered)
    for day in range(day_count):
        covered |= (
            accepted[:, day, np.newaxis, :]
            & (session_start[:, day, np.newaxis, :] <= start)
            & (session_end[:, day, np.newaxis, :] >= end)
        )
        away |= (
            drives[:, day, np.newaxis, :]
            & (departure[:, day, np.newaxis, :] < end)
            & (return_end[:, day, np.newaxis, :] > start)
        )
    return covered & ~away


def _per_ev_chunks(
    state: ReplayState,
    study_slots: pd.DataFrame,
    fleet_world_intervals: pd.DataFrame,
    paths: Sequence[str],
) -> Iterator[tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]]:
    """Yield (world ids, {path: per-EV arrays}) for chunks of at most 10 worlds.

    Each path's arrays have shape (chunk world, study slot, EV): the kernel's
    ``UNIT_INTERVAL_QUANTITIES`` plus ``connected`` (bool).  Smart charging
    can move any EV's charging, so the selected pass re-runs every EV; the
    optional timed path (decision 0007, model step 2), when ``"timed"`` is
    in ``paths``, is a further such pass on the same sampled inputs.
    Connection depends only on the sampled inputs (EVs never strand,
    decision 0004 item 32), so every path shares one ``connected`` array.
    """

    settings = state.settings
    all_evs = np.arange(settings.vehicle_count)
    slot_start_ns = pd.DatetimeIndex(study_slots["interval_start_utc"]).as_unit("ns").asi8
    fleet = _fleet_check_arrays(fleet_world_intervals, paths)
    public_efficiency = state.public_top_up.efficiency_fraction
    for first in range(0, settings.evaluation_world_count, WORLD_CHUNK_SIZE):
        worlds = np.arange(first, min(first + WORLD_CHUNK_SIZE, settings.evaluation_world_count))
        normal_slice = kernel_slice(state, worlds, all_evs, path_id="normal")
        connected = _home_connected(normal_slice.inputs, slot_start_ns)
        by_path = {"normal": _simulate(normal_slice)}
        if "selected" in paths:
            selected_slice = kernel_slice(state, worlds, all_evs, path_id="selected")
            by_path["selected"] = _simulate(selected_slice)
        if "timed" in paths:
            by_path["timed"] = _simulate(kernel_slice(state, worlds, all_evs, path_id="timed"))
        for path_index, path in enumerate(paths):
            unit = by_path[path]
            unit["connected"] = connected
            _check_reconciles(
                unit,
                {name: values[path_index, worlds] for name, values in fleet.items()},
                public_efficiency,
                path,
            )
        yield worlds, by_path


def _simulate(piece: KernelSlice) -> dict[str, np.ndarray]:
    return simulate_unit_intervals(
        piece.settings,
        piece.units,
        piece.inputs,
        effective_efficiency_miles_per_battery_kwh=piece.effective_efficiency,
        public_top_up=piece.public_top_up,
        smart_charging=piece.smart_charging,
        intraday_dispatch=piece.intraday_dispatch,
        charge_allowed=piece.charge_allowed,
    )


# Fleet columns the per-EV pass is reconciled against.  ``public_battery_added_kwh``
# is the kernel's audit column of top-up energy (battery side, kWh).
_RECONCILED_FLEET_COLUMNS = (
    "connected_count",
    "home_import_kwh",
    "public_import_kwh",
    "public_battery_added_kwh",
)


def _fleet_check_arrays(
    fleet_world_intervals: pd.DataFrame, paths: Sequence[str]
) -> dict[str, np.ndarray]:
    """(path, world, slot) arrays of each reconciled fleet column."""

    return {
        name: np.stack([_metric_matrix(fleet_world_intervals, p, name) for p in paths])
        for name in _RECONCILED_FLEET_COLUMNS
    }


def _check_reconciles(
    unit: dict[str, np.ndarray],
    fleet: dict[str, np.ndarray],
    public_efficiency: float,
    path: str,
) -> None:
    """Raise ``ValueError`` unless the per-EV pass sums to the stored fleet frame.

    The per-EV pass must be the same physics as the fleet aggregate (audit
    B3's reconciliation).  Public import and top-up energy are checked too:
    an earlier replay silently used the default top-up threshold and target
    while home import still matched.  Top-up energy is public import times
    ``public_efficiency``.
    """

    per_ev = {
        "home_import_kwh": unit["home_grid_import_kwh"],
        "public_import_kwh": unit["public_grid_import_kwh"],
        "public_battery_added_kwh": unit["public_grid_import_kwh"] * public_efficiency,
    }
    if not np.array_equal(unit["connected"].sum(axis=2), fleet["connected_count"]):
        raise ValueError(f"per-EV connection state does not reconcile with the {path} fleet")
    for name, values in per_ev.items():
        if not np.allclose(values.sum(axis=2), fleet[name], rtol=0.0, atol=1e-6):
            raise ValueError(f"per-EV {name} does not reconcile with the {path} fleet")


# --------------------------------------------------------------------------
# Plug-in sessions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Sessions:
    """Plug-in sessions found in per-EV arrays; one entry per session."""

    world: np.ndarray  # index into the arrays' world axis
    ev: np.ndarray  # index into the arrays' EV axis
    first_slot: np.ndarray  # first fully connected slot
    end_slot: np.ndarray  # one past the last connected slot
    plug_in_kwh: np.ndarray  # battery stock at plug-in (closing of the slot before)
    plug_out_kwh: np.ndarray  # battery stock at the end of the last connected slot
    home_import_kwh: np.ndarray  # grid-side import over the session


def find_sessions(
    connected: np.ndarray, closing_kwh: np.ndarray, home_import_kwh: np.ndarray
) -> _Sessions:
    """Sessions = maximal runs of fully connected slots, per world and EV.

    A run starting at slot 0 was already plugged at the horizon start, so it
    is excluded (contract 3.9): its plug-in time and SoC are not observed.
    Plug-in SoC is the closing stock of the slot before the run, a stock-flow
    output, never a sampled value.
    """

    world_count, slot_count, ev_count = connected.shape
    by_ev = np.transpose(connected, (0, 2, 1))  # (world, EV, slot): runs along the last axis
    padded = np.zeros((world_count, ev_count, slot_count + 2), dtype=np.int8)
    padded[:, :, 1:-1] = by_ev
    step = np.diff(padded, axis=2)
    # np.nonzero walks in (world, EV, slot) order, so the k-th start and the
    # k-th end belong to the same run.
    start_w, start_e, first = np.nonzero(step == 1)
    _, _, end = np.nonzero(step == -1)
    keep = first > 0
    start_w, start_e, first, end = start_w[keep], start_e[keep], first[keep], end[keep]
    cumulative = np.zeros((world_count, slot_count + 1, ev_count))
    cumulative[:, 1:, :] = np.cumsum(home_import_kwh, axis=1)
    return _Sessions(
        world=start_w,
        ev=start_e,
        first_slot=first,
        end_slot=end,
        plug_in_kwh=closing_kwh[start_w, first - 1, start_e],
        plug_out_kwh=closing_kwh[start_w, end - 1, start_e],
        home_import_kwh=cumulative[start_w, end, start_e] - cumulative[start_w, first, start_e],
    )


def plug_in_events_frame(
    sessions: _Sessions,
    *,
    study_slots: pd.DataFrame,
    world_ids: np.ndarray,
    unit_ids: np.ndarray,
    cohort_ids: np.ndarray,
    capacity_kwh: np.ndarray,
    home_charge_efficiency: float,
    path_id: str,
) -> pd.DataFrame:
    """Contract 3.9 rows for the given sessions (world and EV ids by array index)."""

    slot_count = len(study_slots)
    first = sessions.first_slot
    open_session = sessions.end_slot == slot_count
    last = sessions.end_slot - 1
    capacity = capacity_kwh[sessions.ev]
    plug_in_utc = study_slots["interval_start_utc"].to_numpy()[first]
    plug_out = study_slots["interval_end_utc"].to_numpy()[last]
    plug_out[open_session] = np.datetime64("NaT")
    frame = pd.DataFrame(
        {
            "world_id": np.asarray(world_ids, dtype=np.int64)[sessions.world],
            "path_id": path_id,
            "unit_id": np.asarray(unit_ids, dtype=object)[sessions.ev],
            "cohort_id": np.asarray(cohort_ids, dtype=object)[sessions.ev],
            "event_index": np.zeros(len(first), dtype=np.int64),
            "plug_in_utc": pd.to_datetime(plug_in_utc, utc=True).as_unit("ns"),
            "plug_in_london": pd.to_datetime(plug_in_utc, utc=True)
            .as_unit("ns")
            .tz_convert(LONDON),
            "local_date": pd.Series(study_slots["local_date"].to_numpy()[first], dtype=object),
            "day_type": pd.Series(study_slots["day_type"].to_numpy()[first], dtype=object),
            "plug_in_local_hour": study_slots["interval_start_london"].dt.hour.to_numpy(
                dtype=np.int64
            )[first],
            "plug_in_soc_percent": 100.0 * sessions.plug_in_kwh / capacity,
            "plug_out_utc": pd.to_datetime(plug_out, utc=True).as_unit("ns"),
            "plug_out_soc_percent": np.where(
                open_session, np.nan, 100.0 * sessions.plug_out_kwh / capacity
            ),
            "home_import_kwh": sessions.home_import_kwh,
            # Battery-side energy: the kernel adds import x home charge efficiency.
            "battery_added_kwh": sessions.home_import_kwh * home_charge_efficiency,
            "still_plugged_at_horizon_end": open_session,
        }
    )
    frame = frame.sort_values(["world_id", "unit_id", "plug_in_utc"], kind="stable")
    frame["event_index"] = frame.groupby(["world_id", "unit_id"]).cumcount().astype(np.int64)
    return frame.reset_index(drop=True).loc[:, PLUG_IN_EVENT_COLUMNS]


def _session_statistics_rows(
    sessions: _Sessions,
    *,
    study_slots: pd.DataFrame,
    world_ids: np.ndarray,
    cohort_codes: np.ndarray,
    capacity_kwh: np.ndarray,
) -> pd.DataFrame:
    """Compact numeric rows (one per plug-in) used for every-world statistics.

    ``plug_in_slot`` is the session's first connected study slot.
    """

    hour = study_slots["interval_start_london"].dt.hour.to_numpy(dtype=np.int64)
    weekend = study_slots["day_type"].eq("weekend").to_numpy()
    weekday = np.array([value.weekday() for value in study_slots["local_date"]], dtype=np.int64)
    return pd.DataFrame(
        {
            "world_id": world_ids[sessions.world],
            "cohort_code": cohort_codes[sessions.ev],
            "weekend": weekend[sessions.first_slot],
            "weekday": weekday[sessions.first_slot],
            "local_hour": hour[sessions.first_slot],
            "soc_percent": 100.0 * sessions.plug_in_kwh / capacity_kwh[sessions.ev],
            "plug_in_slot": sessions.first_slot,
        }
    )


def _day_type_rows(rows: pd.DataFrame, day_type: str) -> pd.DataFrame:
    if day_type == "all":
        return rows
    return rows.loc[rows["weekend"].eq(day_type == "weekend")]


def plug_in_world_kpis(
    rows: pd.DataFrame, world_count: int, groups: dict[str, tuple[int | None, int]]
) -> pd.DataFrame:
    """Contract 3.10: per-world plug-in KPIs for every world, group and day type.

    ``groups`` maps ``group_id`` to ``(cohort_code, ev_count)``; the fleet has
    code ``None``.  One row per ``(world_id, group_id, day_type, metric)``.
    Counts are real zeros when a world has no plug-ins; the median SoC, the
    below-10% share and the modal hour are NaN then (missing, not zero).  The
    modal hour breaks ties towards the earlier hour so it is reproducible.
    Feeds ``plug_in_summary.kpis``, ``cohort_summary`` and Compare.
    """

    metrics = {**PLUG_KPI_UNITS, "modal_plug_in_local_hour": "hour (London)"}
    worlds = pd.Index(np.arange(world_count), name="world_id")
    frames = []
    for group_id, (code, ev_count) in groups.items():
        group_rows = rows if code is None else rows.loc[rows["cohort_code"].eq(code)]
        for day_type in DAY_TYPE_ORDER:
            subset = _day_type_rows(group_rows, day_type)
            grouped = subset.groupby("world_id")["soc_percent"]
            count = grouped.size().reindex(worlds, fill_value=0).to_numpy(dtype=float)
            hours = _count_matrix(
                subset["world_id"].to_numpy(), subset["local_hour"].to_numpy(), 24, world_count
            )
            values = {
                "plug_ins_per_week": count,
                "plug_ins_per_ev_per_week": count / ev_count,
                "median_plug_in_soc_percent": grouped.median().reindex(worlds).to_numpy(float),
                "share_below_10_percent_soc": (subset["soc_percent"] < 10.0)
                .groupby(subset["world_id"])
                .mean()
                .reindex(worlds)
                .to_numpy(float),
                # argmax returns the first (earliest) of tied hours.
                "modal_plug_in_local_hour": np.where(
                    count > 0, np.argmax(hours, axis=1).astype(float), np.nan
                ),
            }
            for metric, unit in metrics.items():
                frames.append(
                    pd.DataFrame(
                        {
                            "world_id": worlds.to_numpy(np.int64),
                            "group_id": group_id,
                            "day_type": day_type,
                            "metric": metric,
                            "unit": unit,
                            "value": np.asarray(values[metric], dtype=np.float64),
                        }
                    )
                )
    frame = pd.concat(frames, ignore_index=True)
    order = {
        "group_id": list(groups).index,
        "day_type": DAY_TYPE_ORDER.index,
        "metric": list(metrics).index,
    }
    frame = frame.sort_values(
        ["world_id", "group_id", "day_type", "metric"],
        key=lambda column: column.map(order[column.name]) if column.name in order else column,
        kind="stable",
    )
    return frame.reset_index(drop=True)


@dataclass(frozen=True)
class PlugInSummary:
    """Contract 3.10: plug-in histograms and KPIs, each a quantile across worlds."""

    hour_bands: pd.DataFrame
    soc_bands: pd.DataFrame
    kpis: pd.DataFrame


def plug_in_summary(
    rows: pd.DataFrame, world_kpis: pd.DataFrame, world_count: int
) -> PlugInSummary:
    """Hour and SoC-bin shares per world, then mean and quantiles across worlds (3.10).

    A world's share is its plug-ins in the bin divided by its plug-ins of that
    day type, so busy and quiet weeks weigh the same.  Worlds with no plug-in
    of a day type have no share and are left out of that day type's
    quantiles (``world_count`` says how many remain).
    """

    hour_rows, soc_rows = [], []
    for day_type in DAY_TYPE_ORDER:
        subset = _day_type_rows(rows, day_type)
        soc_bin = np.minimum(subset["soc_percent"].to_numpy() // _SOC_BIN_WIDTH, _SOC_BIN_COUNT - 1)
        hours = _count_matrix(
            subset["world_id"].to_numpy(), subset["local_hour"].to_numpy(), 24, world_count
        )
        bins = _count_matrix(
            subset["world_id"].to_numpy(), soc_bin.astype(np.int64), _SOC_BIN_COUNT, world_count
        )
        totals = hours.sum(axis=1)
        with_events = totals > 0
        hour_share = hours[with_events] / totals[with_events, np.newaxis]
        bin_share = bins[with_events] / totals[with_events, np.newaxis]
        for hour in range(24):
            hour_rows.append(
                {
                    "day_type": day_type,
                    "local_hour": hour,
                    "world_count": int(with_events.sum()),
                    "count_p50": _quantile(hours[with_events, hour], 0.5),
                    **_share_quantiles(hour_share[:, hour]),
                }
            )
        for index in range(_SOC_BIN_COUNT):
            soc_rows.append(
                {
                    "day_type": day_type,
                    "soc_bin_lower_percent": index * _SOC_BIN_WIDTH,
                    "soc_bin_upper_percent": (index + 1) * _SOC_BIN_WIDTH,
                    "world_count": int(with_events.sum()),
                    **_share_quantiles(bin_share[:, index]),
                }
            )
    kpi_rows = []
    for day_type in DAY_TYPE_ORDER:
        for metric, unit in PLUG_KPI_UNITS.items():
            values = world_kpis.loc[
                world_kpis["group_id"].eq("fleet")
                & world_kpis["day_type"].eq(day_type)
                & world_kpis["metric"].eq(metric),
                "value",
            ].dropna()
            q = _quantiles(values.to_numpy(float))
            kpi_rows.append(
                {
                    "day_type": day_type,
                    "metric": metric,
                    "unit": unit,
                    "p10": q[0],
                    "p50": q[1],
                    "p90": q[2],
                }
            )
    return PlugInSummary(
        hour_bands=_typed(pd.DataFrame(hour_rows), int_columns=("local_hour", "world_count")),
        soc_bands=_typed(
            pd.DataFrame(soc_rows),
            int_columns=("soc_bin_lower_percent", "soc_bin_upper_percent", "world_count"),
        ),
        kpis=_typed(pd.DataFrame(kpi_rows), int_columns=()),
    )


WEEKDAY_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def plug_in_heatmap(
    rows: pd.DataFrame, world_count: int, vehicle_count: int, study_slots: pd.DataFrame
) -> pd.DataFrame:
    """Contract 3.10a: chance a plug-in starts in each weekday x London hour cell.

    ``rows`` are every world's normal-path plug-ins (``weekday`` 0 = Monday,
    the weekday of the plug-in's session night, so a 01:00 Saturday plug-in
    is Friday night's; ``local_hour`` 0-23).  In each world, a cell's value
    is its plug-ins divided by the EV-days it covers (``vehicle_count`` x
    the study nights on that weekday): the chance an EV plugs in during that hour on that day
    (strictly, plug-ins per EV-day, which equals the chance while an EV
    starts at most one session in an hour).  Dividing by EV-days rather than
    by plug-ins keeps a busy and a quiet day comparable, which a share of
    plug-ins would hide.  Then mean and P10/P50/P90 across worlds, so the
    heatmap shows the median week and its spread (decision 0004 item 43).
    A weekday with no study date has NaN.  A session already running at the
    horizon start is not a plug-in (contract 3.9), so the first study hour
    slightly undercounts.
    """

    counts = np.zeros((world_count, 7, 24))
    np.add.at(
        counts,
        (rows["world_id"].to_numpy(), rows["weekday"].to_numpy(), rows["local_hour"].to_numpy()),
        1.0,
    )
    dates = pd.Series(study_slots["local_date"].unique())
    day_count = np.bincount([value.weekday() for value in dates], minlength=7)
    with np.errstate(invalid="ignore", divide="ignore"):
        chance = counts / (vehicle_count * day_count[np.newaxis, :, np.newaxis])
    chance[:, day_count == 0, :] = np.nan
    p10, p50, p90 = np.quantile(chance, (0.1, 0.5, 0.9), axis=0, method="linear")
    frame = pd.DataFrame(
        {
            "weekday": np.repeat(np.arange(7, dtype=np.int64), 24),
            "weekday_label": pd.Series(np.repeat(WEEKDAY_LABELS, 24), dtype=object),
            "local_hour": np.tile(np.arange(24, dtype=np.int64), 7),
            "day_count": np.repeat(day_count.astype(np.int64), 24),
            "world_count": np.int64(world_count),
            "mean": chance.mean(axis=0).ravel(),
            "p10": p10.ravel(),
            "p50": p50.ravel(),
            "p90": p90.ravel(),
        }
    )
    return frame


PLUG_EVENT_HEATMAP_COLUMNS = [
    "weekday",
    "weekday_label",
    "local_half_hour",
    "local_time_label",
    "slot_count",
    "world_count",
    "mean",
    "p10",
    "p50",
    "p90",
]


def plug_event_heatmap(
    world_ids: np.ndarray,
    slot_index: np.ndarray,
    world_count: int,
    vehicle_count: int,
    study_slots: pd.DataFrame,
) -> pd.DataFrame:
    """Contract 3.10b: events per EV-day by London weekday x half-hour (decision 0004 item 54).

    One event per entry: ``world_ids`` (world index) and ``slot_index`` (the
    study slot the event happens at the start of).  Used for plug-ins (the
    session's first connected slot) and plug-outs (the slot after a session's
    last connected slot: the unplug, which is also the departure, decision
    0004 item 51).

    A cell is a London calendar weekday and wall-clock half-hour of the event
    instant.  Per world, its events are divided by ``vehicle_count`` x the
    study slots falling in that cell, i.e. events per EV-day in that
    half-hour; then mean and linear P10/P50/P90 across worlds (worlds
    first).  Counting the study's own slots per cell keeps the rate honest
    when a study does not cover whole days (a clock change repeats or skips
    a half-hour; a horizon that starts at noon covers half of its first and
    last day): a cell with no study slot is NaN, a cell with slots and no
    event a real 0.
    """

    weekday = study_slots["interval_start_london"].dt.weekday.to_numpy(dtype=np.int64)
    half_hour = study_slots["local_half_hour"].to_numpy(dtype=np.int64)
    cell = weekday * 48 + half_hour
    slot_count = np.bincount(cell, minlength=7 * 48)
    counts = np.zeros((world_count, 7 * 48))
    np.add.at(counts, (np.asarray(world_ids), cell[np.asarray(slot_index)]), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = counts / (vehicle_count * slot_count[np.newaxis, :])
    rate[:, slot_count == 0] = np.nan
    p10, p50, p90 = np.quantile(rate, _EV_QUANTILES, axis=0, method="linear")
    cells = np.arange(7 * 48)
    return pd.DataFrame(
        {
            "weekday": cells // 48,
            "weekday_label": pd.Series(np.repeat(WEEKDAY_LABELS, 48), dtype=object),
            "local_half_hour": cells % 48,
            "local_time_label": pd.Series(
                [f"{h // 2:02d}:{30 * (h % 2):02d}" for h in cells % 48], dtype=object
            ),
            "slot_count": slot_count.astype(np.int64),
            "world_count": np.int64(world_count),
            "mean": rate.mean(axis=0),
            "p10": p10,
            "p50": p50,
            "p90": p90,
        },
        columns=PLUG_EVENT_HEATMAP_COLUMNS,
    )


# --------------------------------------------------------------------------
# Session distributions (decision 0004 item 54, plan D-2, contract 3.10c)
# --------------------------------------------------------------------------

SESSION_DISTRIBUTION_METRICS = {
    "plug_in_time": ("hour (London)", 0.0, 0.5, 48),
    "departure_time": ("hour (London)", 0.0, 0.5, 48),
    "plug_in_soc_percent": ("percent", 0.0, 5.0, 20),
    "energy_needed_kwh": ("kWh battery-side", 0.0, 5.0, 16),
    "dwell_hours": ("hours", 0.0, 2.0, 24),
    "flexible_kwh": ("kWh battery-side", 0.0, 5.0, 16),
    "slack_hours": ("hours", -12.0, 2.0, 30),
    "departure_soc_percent": ("percent", 0.0, 5.0, 20),
    "departure_soc_percent_smart": ("percent", 0.0, 5.0, 20),
    # Optional timed-path departure SoC (decision 0007, model step 2): a bin
    # spec only, like the others; ``session_distribution_bands`` adds the row
    # only when a run's rows actually carry this column (the setting is on),
    # so an off run's bands are schema-identical to before this existed.
    "departure_soc_percent_timed": ("percent", 0.0, 5.0, 20),
}
"""metric -> (unit, lower edge of the first bin, bin width, bin count), in display order.

A value below the first bin is counted in the first bin and a value above
the last in the last (``np.clip``), so every observed session lands in
exactly one bin.  Only slack can fall below its range; the bounded metrics
(times, SoC) never leave theirs.
"""
_OPEN_BELOW = frozenset({"slack_hours"})
_OPEN_ABOVE = frozenset({"energy_needed_kwh", "dwell_hours", "flexible_kwh", "slack_hours"})
_NO_BIN = -1
_SESSION_DECIMALS = 9

SESSION_DISTRIBUTION_COLUMNS = [
    "group_id",
    "day_type",
    "metric",
    "unit",
    "bin_index",
    "bin_lower",
    "bin_upper",
    "bin_label",
    "world_count",
    "share_mean",
    "share_p10",
    "share_p50",
    "share_p90",
]


def _session_bin(metric: str, values: np.ndarray) -> np.ndarray:
    """Bin index of each value (int16); ``_NO_BIN`` where the value is NaN (not observed)."""

    _, lower, width, count = SESSION_DISTRIBUTION_METRICS[metric]
    observed = ~np.isnan(values)
    # Rounded to 1e-9 first: charging exactly to target leaves float residue
    # (79.999...% SoC, 1e-15 kWh still needed), which would otherwise put an
    # at-target session on either side of a bin edge at random.
    rounded = np.round(np.where(observed, values, lower), _SESSION_DECIMALS)
    index = np.clip(np.floor((rounded - lower) / width), 0, count - 1)
    return np.where(observed, index, _NO_BIN).astype(np.int16)


def session_distribution_rows(
    sessions: _Sessions,
    *,
    study_slots: pd.DataFrame,
    world_ids: np.ndarray,
    cohort_codes: np.ndarray,
    capacity_kwh: np.ndarray,
    target_kwh: np.ndarray,
    power_kw: np.ndarray,
    efficiency: float,
    smart_closing_kwh: np.ndarray | None = None,
    timed_closing_kwh: np.ndarray | None = None,
) -> pd.DataFrame:
    """One compact row per plug-in session: world, archetype, day type and a bin per metric.

    ``sessions`` are one chunk's normal-path sessions (``find_sessions``: a
    session already running at the horizon start has no observed plug-in and
    is not here, contract 3.9).  ``world_ids`` maps the chunk's world axis to
    world ids; ``cohort_codes``, ``capacity_kwh`` (kWh battery-side),
    ``target_kwh`` (preferred target stock, kWh battery-side) and
    ``power_kw`` (home charging power, grid side) are per EV;
    ``efficiency`` is battery kWh per grid kWh.

    Per session (definitions in contract 3.10c):

    - plug-in time: London wall-clock time of the first fully connected slot;
    - departure time: London time the EV unplugs, the start of the slot
      after the last connected one (unplug = departure, decision 0004 item 51);
    - SoC at plug-in: closing stock of the slot before, a stock-flow output;
    - energy needed: battery kWh from that stock up to the preferred target
      (0 when plugged in at or above it).  Battery-side, like the workbook's
      kWh per plug-in ('Source archetypes'!M6:M11, the crosswalk's battery
      kWh per plug-in) that the view marks against it;
    - dwell: whole connected half-hours x 0.5 h.  The kernel charges only in
      fully connected slots, so this is the time charging could use;
    - flexible kWh: min(energy needed, dwell x power x efficiency), the
      battery energy the session could take at full power;
    - slack: dwell - energy needed / (power x efficiency), the hours charging
      could wait and still reach the target (negative: it cannot, even at
      full power).  ``flexibility_per_world``'s time slack uses the
      expected departure (what the planner knows); this uses the actual
      unplug, because it describes what the session turned out to offer.

    Departure, dwell, flexible kWh and slack need the unplug, so a session
    still plugged in at the horizon end has ``_NO_BIN`` for them.  The day
    type is the session night's (``study_slots.day_type`` of the first slot,
    decision 0004 item 52), so a Friday-night late plug-in is a weekday one.

    Rows keep bin indices, not values (int16, about 20 bytes a session), so
    every world's sessions fit in memory for ``session_distribution_bands``.
    """

    weekend = study_slots["day_type"].eq("weekend").to_numpy()
    values = session_values(
        sessions,
        study_slots=study_slots,
        capacity_kwh=capacity_kwh,
        target_kwh=target_kwh,
        power_kw=power_kw,
        efficiency=efficiency,
        smart_closing_kwh=smart_closing_kwh,
        timed_closing_kwh=timed_closing_kwh,
    )
    return pd.DataFrame(
        {
            "world_id": world_ids[sessions.world],
            "cohort_code": cohort_codes[sessions.ev],
            "weekend": weekend[sessions.first_slot],
            **{metric: _session_bin(metric, values[metric]) for metric in values},
        }
    )


def session_values(
    sessions: _Sessions,
    *,
    study_slots: pd.DataFrame,
    capacity_kwh: np.ndarray,
    target_kwh: np.ndarray,
    power_kw: np.ndarray,
    efficiency: float,
    smart_closing_kwh: np.ndarray | None = None,
    timed_closing_kwh: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """The per-session values of contract 3.10c and supplier §4.6, one entry per session.

    Arguments as ``session_distribution_rows``; returns ``{metric: values}``
    in ``SESSION_DISTRIBUTION_METRICS`` order: London clock hours for the two
    times, percent, battery-side kWh and hours.  A session still plugged in
    at the horizon end has NaN departure, dwell, flexible kWh, slack and
    departure SoC; the smart departure SoC is NaN without ``smart_closing_kwh``.
    ``departure_soc_percent_timed`` (decision 0007, model step 2) is only a
    key of the returned mapping at all when ``timed_closing_kwh`` is given
    (unlike the smart column, which is always present, NaN or not): a run
    without the timed path is still the common case among existing runs and
    fixtures, so this keeps such a run's frame schema-identical to before
    this existed.  Shared by the Sessions distributions and the household
    card (household contract v1 §4 hunk 1), so the two cannot define a
    session differently.
    """

    slot_count = len(study_slots)
    london = study_slots["interval_start_london"]
    clock = (london.dt.hour + london.dt.minute / 60.0).to_numpy(dtype=float)
    first, end, ev = sessions.first_slot, sessions.end_slot, sessions.ev
    closed = end < slot_count
    # Rounded for the same reason as in ``_session_bin``: an at-target
    # session needs exactly 0 kWh, so its slack is exactly its dwell.
    need = np.round(np.maximum(target_kwh[ev] - sessions.plug_in_kwh, 0.0), _SESSION_DECIMALS)
    dwell = np.where(closed, (end - first) * 0.5, np.nan)
    battery_kw = power_kw[ev] * efficiency
    values = {
        "plug_in_time": clock[first],
        "departure_time": np.where(closed, clock[np.minimum(end, slot_count - 1)], np.nan),
        "plug_in_soc_percent": 100.0 * sessions.plug_in_kwh / capacity_kwh[ev],
        "energy_needed_kwh": need,
        "dwell_hours": dwell,
        "flexible_kwh": np.minimum(need, dwell * battery_kw),
        "slack_hours": dwell - need / battery_kw,
        # Supplier contract §4.6: the stock at the last connected slot on each
        # path (plug-in and unplug are exogenous, so the sessions are the same).
        "departure_soc_percent": np.where(
            closed, 100.0 * sessions.plug_out_kwh / capacity_kwh[ev], np.nan
        ),
        "departure_soc_percent_smart": np.full(len(ev), np.nan)
        if smart_closing_kwh is None
        else np.where(
            closed,
            100.0 * smart_closing_kwh[sessions.world, end - 1, ev] / capacity_kwh[ev],
            np.nan,
        ),
    }
    if timed_closing_kwh is not None:
        values["departure_soc_percent_timed"] = np.where(
            closed,
            100.0 * timed_closing_kwh[sessions.world, end - 1, ev] / capacity_kwh[ev],
            np.nan,
        )
    return values


def _session_bin_label(metric: str, index: int) -> str:
    _, lower, width, count = SESSION_DISTRIBUTION_METRICS[metric]
    low, high = lower + index * width, lower + (index + 1) * width
    if metric.endswith("_time"):
        return f"{int(low):02d}:{30 * int(2 * low % 2):02d}"
    if index == 0 and metric in _OPEN_BELOW:
        return f"< {high:g}"
    if index == count - 1 and metric in _OPEN_ABOVE:
        return f"≥ {low:g}"
    return f"{low:g}–{high:g}"


def session_distribution_bands(
    rows: pd.DataFrame, world_count: int, groups: dict[str, int | None]
) -> pd.DataFrame:
    """Contract 3.10c: share of sessions per bin, P10/P50/P90 across worlds.

    ``rows`` are every world's ``session_distribution_rows``; ``groups`` maps
    ``group_id`` to its cohort code (``None`` for the fleet).  Per world,
    group, day type (of the plug-in slot) and metric, a bin's share is its
    sessions over the sessions with that metric observed, so busy and quiet
    weeks weigh the same (the ``plug_in_summary`` rule).  Then mean and
    linear quantiles across the worlds with at least one such session
    (worlds first; ``world_count`` says how many).  ``share_mean`` sums to 1
    over a metric's bins; the quantiles do not and must not be summed.
    ``departure_soc_percent_timed`` (decision 0007, model step 2) only adds
    its rows when ``rows`` actually carries that column (the setting on);
    every other metric is unaffected either way.
    """

    metrics = {
        metric: spec for metric, spec in SESSION_DISTRIBUTION_METRICS.items() if metric in rows
    }
    frames = []
    for group_id, code in groups.items():
        group_rows = rows if code is None else rows.loc[rows["cohort_code"].eq(code)]
        for day_type in DAY_TYPE_ORDER:
            subset = _day_type_rows(group_rows, day_type)
            for metric, (unit, lower, width, count) in metrics.items():
                bins = subset[metric].to_numpy()
                seen = bins != _NO_BIN
                counts = _count_matrix(
                    subset["world_id"].to_numpy()[seen],
                    bins[seen].astype(np.int64),
                    count,
                    world_count,
                )
                totals = counts.sum(axis=1)
                shares = counts[totals > 0] / totals[totals > 0, np.newaxis]
                stats = [_share_quantiles(shares[:, index]) for index in range(count)]
                edges = lower + width * np.arange(count + 1)
                frames.append(
                    pd.DataFrame(
                        {
                            "group_id": group_id,
                            "day_type": day_type,
                            "metric": metric,
                            "unit": unit,
                            "bin_index": np.arange(count, dtype=np.int64),
                            "bin_lower": edges[:-1],
                            "bin_upper": edges[1:],
                            "bin_label": [_session_bin_label(metric, i) for i in range(count)],
                            "world_count": np.int64((totals > 0).sum()),
                            **{
                                name: np.array([s[name] for s in stats], dtype=np.float64)
                                for name in ("share_mean", "share_p10", "share_p50", "share_p90")
                            },
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True).loc[:, SESSION_DISTRIBUTION_COLUMNS]


def _count_matrix(
    world: np.ndarray, category: np.ndarray, size: int, world_count: int
) -> np.ndarray:
    counts = np.zeros((world_count, size), dtype=np.float64)
    np.add.at(counts, (world, category), 1.0)
    return counts


def _share_quantiles(shares: np.ndarray) -> dict[str, float]:
    # The mean share is the one statistic whose bins sum to 1 within a day
    # type (quantiles of shares do not), so histograms can plot it honestly.
    q = _quantiles(shares)
    mean = float(shares.mean()) if len(shares) else np.nan
    return {"share_mean": mean, "share_p10": q[0], "share_p50": q[1], "share_p90": q[2]}


def _quantiles(values: np.ndarray) -> tuple[float, float, float]:
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    q = np.quantile(values, (0.1, 0.5, 0.9), method="linear")
    return (float(q[0]), float(q[1]), float(q[2]))


def _quantile(values: np.ndarray, level: float) -> float:
    return float(np.quantile(values, level, method="linear")) if len(values) else np.nan


def _typed(frame: pd.DataFrame, *, int_columns: Sequence[str]) -> pd.DataFrame:
    for column in frame.columns:
        if column in int_columns:
            frame[column] = frame[column].astype(np.int64)
        elif frame[column].dtype != object:
            frame[column] = frame[column].astype(np.float64)
    return frame


# --------------------------------------------------------------------------
# Average day (contract 3.8)
# --------------------------------------------------------------------------


@dataclass
class _GroupDay:
    """Per-world, per-slot statistics of one group and path, filled chunk by chunk."""

    connected_share: np.ndarray  # (world, slot), fraction of the group's EVs
    soc_mean: np.ndarray  # (world, slot), mean closing SoC across the group's EVs
    soc_p5: np.ndarray
    soc_p95: np.ndarray


def _new_group_day(world_count: int, slot_count: int) -> _GroupDay:
    return _GroupDay(*(np.full((world_count, slot_count), np.nan) for _ in range(4)))


def _fill_group_day(
    target: _GroupDay, worlds: np.ndarray, connected: np.ndarray, soc_percent: np.ndarray
) -> None:
    target.connected_share[worlds] = connected.mean(axis=2)
    target.soc_mean[worlds] = soc_percent.mean(axis=2)
    # Population spread across EVs (decision 0004 item 22), not forecast
    # uncertainty: P5/P95 of the group's EVs in each world and slot.
    p5, p95 = np.percentile(soc_percent, (5.0, 95.0), axis=2)
    target.soc_p5[worlds] = p5
    target.soc_p95[worlds] = p95


# --------------------------------------------------------------------------
# Fleet week across EVs (decision 0004 items 39 and 40)
# --------------------------------------------------------------------------

EV_BAND_METRICS = {"battery_soc_percent": "percent", "home_import_kw": "kW"}
"""Per-EV metrics of ``fleet_interval_ev_bands``, in display order."""
_EV_QUANTILES = (0.1, 0.5, 0.9)


def new_ev_bands(paths: Sequence[str], world_count: int, slot_count: int) -> dict:
    """Empty per-world across-EV quantiles: (path, metric) -> (quantile, world, slot)."""

    return {
        (path, metric): np.full((len(_EV_QUANTILES), world_count, slot_count), np.nan)
        for path in paths
        for metric in EV_BAND_METRICS
    }


def fill_ev_bands(target: dict, path: str, worlds: np.ndarray, unit: dict) -> None:
    """Store P10/P50/P90 across EVs of each EV's SoC and home import, per world and slot.

    ``unit`` holds one chunk's per-EV arrays of shape (world, slot, EV): SoC
    is closing stock over capacity (percent) and home import is the slot's
    kWh over 0.5 h (kW).  Quantiles are across EVs inside each world, so they
    describe the fleet's spread, not forecast uncertainty.
    """

    values = {
        "battery_soc_percent": unit["soc_percent"],
        "home_import_kw": unit["home_grid_import_kwh"] / 0.5,
    }
    for metric, per_ev in values.items():
        target[(path, metric)][:, worlds] = np.quantile(
            per_ev, _EV_QUANTILES, axis=2, method="linear"
        )


def fleet_interval_ev_bands(ev_bands: dict, study_slots: pd.DataFrame) -> pd.DataFrame:
    """Contract 3.6b: across-EV P10/P50/P90 per slot, median across simulated weeks.

    Each world first gives the 10th, 50th and 90th percentile across EVs of
    each EV's value in the slot; then each of those is replaced by its median
    across worlds.  The median keeps order, so ``p10 <= p50 <= p90`` holds.
    Week-to-week spread of the fleet average stays in ``fleet_interval_bands``
    (decision 0004 item 40: both are shown).
    """

    keys = study_slots.loc[:, _SLOT_KEY_COLUMNS]
    frames = []
    for metric, unit in EV_BAND_METRICS.items():
        for path in PATH_ORDER:
            if (path, metric) not in ev_bands:
                continue
            per_world = ev_bands[(path, metric)]
            p10, p50, p90 = np.median(per_world, axis=1)
            frame = keys.copy()
            frame.insert(0, "path_id", path)
            frame.insert(0, "unit", unit)
            frame.insert(0, "metric", metric)
            frame["world_count"] = np.int64(per_world.shape[1])
            frame["p10"], frame["p50"], frame["p90"] = p10, p50, p90
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Flexibility (decision 0004 item 43)
# --------------------------------------------------------------------------
#
# How much home charging could be moved, half-hour by half-hour.  Every
# metric counts only EVs plugged in at home for the whole slot, the same
# condition the kernel needs before it charges (half-hour dispatch unit,
# decision 0001), and reads each EV's stock at the slot start.
#
# - flexible_power_kw: the turn-up ceiling, the most the plugged-in EVs still
#   below their preferred target could draw together in that half-hour: the
#   sum of their home charging power (the decision's "plugged-in EVs x home
#   charging power").  The kernel never charges an EV at or above target, so
#   such EVs add nothing.  An EV 0.1 kWh short counts its full power: this is
#   an instantaneous ceiling, and movable energy says how long it can last.
# - turn_up_headroom_kw: flexible_power_kw minus the home import the path
#   actually drew in the slot (kW), per world: the ceiling still unused.  On
#   the smart path it is larger because charging waits for cheap half-hours;
#   that is headroom still unused while waiting, not extra flexibility.  Only
#   EVs counted in the ceiling can import, so it is never really negative;
#   it is clipped at 0 so floating-point rounding cannot show "-0.0 kW".
# - movable_energy_kwh: grid-side kWh still needed to reach the preferred
#   target, (target - stock) / home charge efficiency, summed over plugged-in
#   EVs still below target (EVs at target need nothing): the energy that must
#   be delivered before the expected departure and so the energy whose
#   timing a smart charger chooses.
# - time_slack_hours: per plugged-in EV below target, hours left until its
#   expected departure minus hours needed to reach target at full power
#   (grid need / power).  Zero means it must charge now to make the target;
#   negative means it cannot make it.  EVs at target are left out: they will
#   not charge, so they have no charging to move (decision 0004 item 46).
#
# The expected departure is the smart charger's (action.expected_departures_
# utc_ns: the typical departure of that date's window minus the departure
# margin), taken as the first one at least one half-hour after the slot
# starts, as the planner does, so both use one definition.
#
# Power, headroom and energy are summed over EVs inside each world, then
# P10/P50/P90 across worlds (worlds first).  Time slack is a per-EV quantity,
# so it has two spreads (item 40: the across-weeks band is always shown):
# across EVs (each world's P10/P50/P90 across its EVs, then the median across
# worlds, as fleet_interval_ev_bands does) and across weeks (P10/P50/P90
# across worlds of each world's median EV slack).

FLEXIBILITY_ROWS = (
    ("flexible_power_kw", "kW", "across_weeks"),
    ("turn_up_headroom_kw", "kW", "across_weeks"),
    ("movable_energy_kwh", "kWh", "across_weeks"),
    ("time_slack_hours", "hours", "across_evs"),
    ("time_slack_hours", "hours", "across_weeks"),
)
"""(metric, unit, spread) of each ``flexibility_bands`` row group, in order (contract 3.6c)."""


@dataclass(frozen=True)
class FlexibilityInputs:
    """Per-EV values the flexibility metrics need that the kernel output does not carry.

    ``hours_to_departure`` is (slot, EV): hours from each study slot's start
    to the EV's next expected departure.  ``target_kwh`` and ``power_kw`` are
    (EV,): preferred target stock (kWh) and home charging power (kW).
    ``efficiency`` is the home charge efficiency (battery kWh per grid kWh).
    """

    hours_to_departure: np.ndarray
    target_kwh: np.ndarray
    power_kw: np.ndarray
    efficiency: float


def flexibility_inputs(
    settings: RunSettings,
    units: pd.DataFrame,
    study_slots: pd.DataFrame,
    departure_margin_hours: float,
) -> FlexibilityInputs:
    """Build ``FlexibilityInputs`` from a run's settings, kernel units and departure margin (h)."""

    departures = expected_departures_utc_ns(settings, units, departure_margin_hours)
    slot_start = pd.DatetimeIndex(study_slots["interval_start_utc"]).as_unit("ns").asi8
    # The planner's rule: the first expected departure at least one slot
    # after the slot starts.  Rows of ``departures`` increase with the date,
    # so counting the earlier ones gives its row.
    row = (
        departures[np.newaxis, :, :] < (slot_start + _HALF_HOUR_NS)[:, np.newaxis, np.newaxis]
    ).sum(axis=1)
    next_departure = np.take_along_axis(departures, row, axis=0)
    capacity = units["physical_capacity_kwh"].to_numpy(dtype=float)
    return FlexibilityInputs(
        hours_to_departure=(next_departure - slot_start[:, np.newaxis]) / _NS_PER_HOUR,
        target_kwh=capacity * units["preferred_target_soc_fraction"].to_numpy(dtype=float),
        power_kw=units["home_charger_limit_kw"].to_numpy(dtype=float),
        efficiency=settings.home_charge_efficiency,
    )


def flexibility_per_world(
    connected: np.ndarray,
    opening_kwh: np.ndarray,
    home_import_kwh: np.ndarray,
    flex: FlexibilityInputs,
) -> dict[str, np.ndarray]:
    """Per-world flexibility for one chunk of worlds on one path.

    ``connected`` (bool), ``opening_kwh`` and ``home_import_kwh`` (the
    path's grid import in the slot) have shape (world, slot, EV).  Returns
    ``flexible_power_kw``, ``turn_up_headroom_kw`` and ``movable_energy_kwh``
    as fleet sums of shape (world, slot), and ``time_slack_hours`` as
    P10/P50/P90 across plugged-in EVs below target, shape (3, world, slot),
    NaN where there is none.
    """

    power = flex.power_kw[np.newaxis, np.newaxis, :]
    below_target, need_kwh = _charge_need(connected, opening_kwh, flex)
    slack = np.where(
        below_target, flex.hours_to_departure[np.newaxis, :, :] - need_kwh / power, np.nan
    )
    with warnings.catch_warnings():
        # A slot with no EV that can charge has no slack; NaN is its honest value.
        warnings.simplefilter("ignore", RuntimeWarning)
        slack_quantiles = np.nanquantile(slack, _EV_QUANTILES, axis=2, method="linear")
    flexible_power = np.where(below_target, power, 0.0).sum(axis=2)
    return {
        "flexible_power_kw": flexible_power,
        "turn_up_headroom_kw": np.maximum(flexible_power - home_import_kwh.sum(axis=2) / 0.5, 0.0),
        "movable_energy_kwh": need_kwh.sum(axis=2),
        "time_slack_hours": slack_quantiles,
    }


def _charge_need(
    connected: np.ndarray, opening_kwh: np.ndarray, flex: FlexibilityInputs
) -> tuple[np.ndarray, np.ndarray]:
    """(world, slot, EV) "can charge" flags and grid-side kWh still needed to reach target."""

    target = flex.target_kwh[np.newaxis, np.newaxis, :]
    # The kernel's own test for "can charge": plugged in for the whole slot
    # and below target (physics._home_charge_kwh).
    below_target = connected & (opening_kwh < target)
    return below_target, np.where(below_target, (target - opening_kwh) / flex.efficiency, 0.0)


def flexibility_bands(per_world: dict, study_slots: pd.DataFrame) -> pd.DataFrame:
    """Contract 3.6c: flexibility per half-hour from every world's values.

    ``per_world`` maps ``(path, metric)`` to the arrays of
    ``flexibility_per_world`` stacked over all worlds.  Power, headroom and
    energy get P10/P50/P90 across worlds.  Time slack gets, across EVs, the
    median across worlds of each across-EV quantile and, across weeks, the
    P10/P50/P90 across worlds of each world's median EV; both over the worlds
    with an EV below target plugged in that slot (``world_count``).
    """

    keys = study_slots.loc[:, _SLOT_KEY_COLUMNS]
    frames = []
    for metric, unit, spread in FLEXIBILITY_ROWS:
        for path in PATH_ORDER:
            if (path, metric) not in per_world:
                continue
            values = per_world[(path, metric)]
            if metric != "time_slack_hours":
                world_count = np.full(values.shape[1], values.shape[0], dtype=np.int64)
                p10, p50, p90 = np.quantile(values, _EV_QUANTILES, axis=0, method="linear")
            else:
                world_count = np.isfinite(values[1]).sum(axis=0).astype(np.int64)
                with warnings.catch_warnings():
                    # A slot where no world has an EV below target has no
                    # slack: NaN is its honest value, so the all-NaN warning
                    # is expected here and only here.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    if spread == "across_evs":
                        p10, p50, p90 = np.nanmedian(values, axis=1)
                    else:
                        p10, p50, p90 = np.nanquantile(
                            values[1], _EV_QUANTILES, axis=0, method="linear"
                        )
            frame = keys.copy()
            frame.insert(0, "path_id", path)
            frame.insert(0, "spread", spread)
            frame.insert(0, "unit", unit)
            frame.insert(0, "metric", metric)
            frame["world_count"] = world_count
            frame["p10"], frame["p50"], frame["p90"] = p10, p50, p90
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Deferrable power by slack (decision 0004 item 54, plan C2)
# --------------------------------------------------------------------------
#
# The time-slack band hid the must-charge-now tail and carried no kW.  This
# splits the unmanaged path's flexible power (the kW of plugged-in EVs still
# below target) by how long each EV's charging could wait: its slack is the
# hours to its expected departure minus the hours it needs at full power.
# The need is rounded up to whole half-hours, ceil(need / (P x 0.5 h)) x 0.5 h,
# because the kernel dispatches in whole slots (decision 0001): an EV needing
# 0.2 h of charging still occupies one half-hour.  The unmanaged path is used
# because it is the counterfactual Axle would be moving charging away from;
# the smart path's EVs are already waiting.

SLACK_BUCKETS = (
    ("under_1h", 1.0),
    ("1_to_2h", 2.0),
    ("2_to_4h", 4.0),
    ("4_to_8h", 8.0),
    ("8h_or_more", np.inf),
)
"""(bucket id, upper edge in hours) in order; each bucket holds slack >= the previous edge.

The first bucket also holds negative slack (the EV cannot reach target in
time), so it is "must run now".  Edges follow plan C2.
"""

AT_LEAST_HOURS = (1.0, 2.0, 4.0, 8.0)
"""Slack floors (hours) of the cumulative ``at_least_<N>h`` rows: "kW that can wait N h or more"."""

DEFERRABLE_BUCKET_ORDER = (
    *(bucket for bucket, _ in SLACK_BUCKETS),
    *(f"at_least_{hours:g}h" for hours in AT_LEAST_HOURS),
    "total",
)


def deferrable_power_per_world(
    connected: np.ndarray, opening_kwh: np.ndarray, flex: FlexibilityInputs
) -> np.ndarray:
    """Per-world kW of EVs that can charge, split by slack bucket, for one chunk.

    ``connected`` (bool) and ``opening_kwh`` have shape (world, slot, EV).
    Returns shape (bucket, world, slot) in ``SLACK_BUCKETS`` order: the sum of
    home charging power (kW) of plugged-in EVs below target whose whole-slot
    slack falls in the bucket.  The buckets add up to ``flexible_power_kw``.
    """

    power = flex.power_kw[np.newaxis, np.newaxis, :]
    below_target, need_kwh = _charge_need(connected, opening_kwh, flex)
    # Whole half-hours needed.  The 1e-9 stops a need of exactly k slots
    # (k x P x 0.5 kWh, give or take float rounding) counting as k + 1.
    need_hours = np.ceil(need_kwh / (power * 0.5) - 1e-9) * 0.5
    slack = flex.hours_to_departure[np.newaxis, :, :] - need_hours
    upper = np.array([edge for _, edge in SLACK_BUCKETS[:-1]])
    # side="right": slack exactly on an edge belongs to the bucket above it.
    bucket = np.searchsorted(upper, slack, side="right")
    return np.stack(
        [
            np.where(below_target & (bucket == index), power, 0.0).sum(axis=2)
            for index in range(len(SLACK_BUCKETS))
        ]
    )


def deferrable_rows_per_world(per_bucket: np.ndarray) -> np.ndarray:
    """Every ``DEFERRABLE_BUCKET_ORDER`` row per world: (row, world, slot) kW.

    ``per_bucket`` is ``deferrable_power_per_world`` (bucket, world, slot).
    The cumulative ``at_least_<N>h`` rows and ``total`` are summed over
    buckets inside each world, so their quantiles are world-first; adding
    bucket quantiles instead would not give the quantile of the sum.
    """

    # Bucket i holds slack >= the upper edge of bucket i - 1, so the buckets
    # from the one starting at N h upwards hold slack of at least N h.
    starts = [edge for _, edge in SLACK_BUCKETS[:-1]]
    cumulative = [per_bucket[starts.index(hours) + 1 :].sum(axis=0) for hours in AT_LEAST_HOURS]
    return np.concatenate([per_bucket, np.stack(cumulative), per_bucket.sum(axis=0)[np.newaxis]])


def deferrable_power_bands(per_bucket: np.ndarray, study_slots: pd.DataFrame) -> pd.DataFrame:
    """Contract 3.6d: unmanaged deferrable kW by slack, P10/P50/P90 across worlds.

    ``per_bucket`` is ``deferrable_power_per_world`` stacked over every world,
    shape (bucket, world, slot).  Rows follow ``DEFERRABLE_BUCKET_ORDER``:
    the five buckets, the cumulative "at least N hours" rows and ``total``
    (equal to the normal path's ``flexible_power_kw``), each summed per
    world before quantiles (``deferrable_rows_per_world``).
    """

    keys = study_slots.loc[:, _SLOT_KEY_COLUMNS]
    world_count = per_bucket.shape[1]
    frames = []
    for bucket, values in zip(
        DEFERRABLE_BUCKET_ORDER, deferrable_rows_per_world(per_bucket), strict=True
    ):
        p10, p50, p90 = np.quantile(values, _EV_QUANTILES, axis=0, method="linear")
        frame = keys.copy()
        frame.insert(0, "path_id", "normal")
        frame.insert(0, "unit", "kW")
        frame.insert(0, "slack_bucket", bucket)
        frame["world_count"] = np.int64(world_count)
        frame["p10"], frame["p50"], frame["p90"] = p10, p50, p90
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# Reporting points of the Key stats table (plan E1, lead decision 29
# September 2026), not model assumptions: they choose what to summarise, not
# how the fleet behaves.  The busy-hours threshold is a share of the fleet's
# own charger capacity rather than a fixed 1 MW, so it means the same thing
# for 200 EVs as for 10,000 (a fixed 1 MW is never reached by 200 x 7 kW
# EVs with some at target).
DEFERRABLE_CAPACITY_SHARE = 0.25
REPORT_HALF_HOURS = {"1800": 36, "2200": 44, "0300": 6}  # London wall-clock half-hours
WAIT_CHECK_HALF_HOUR = 38  # London 19:00-19:30, for "deferrable >= 2 h at 19:00"
MOVABLE_CHECK_HALF_HOUR = 36  # London 18:00-18:30

_QUANTILE_NAMES = ("p10", "p50", "p90")
FLEXIBILITY_WEEKLY_COLUMNS = [
    "path_id",
    "world_count",
    "fleet_charger_capacity_kw",
    "peak_deferrable_kw_p10",
    "peak_deferrable_kw_p50",
    "peak_deferrable_kw_p90",
    "peak_modal_slot_index",
    "peak_modal_interval_start_utc",
    "peak_modal_interval_start_london",
    "peak_modal_week_count",
    *(f"hours_at_least_quarter_capacity_{q}" for q in _QUANTILE_NAMES),
    *(f"movable_energy_1800_kwh_{q}" for q in _QUANTILE_NAMES),
    *(f"deferrable_kw_{time}_{q}" for time in REPORT_HALF_HOURS for q in _QUANTILE_NAMES),
    *(f"w_per_plugged_in_ev_{time}_{q}" for time in REPORT_HALF_HOURS for q in _QUANTILE_NAMES),
    *(f"deferrable_at_least_2h_kw_1900_{q}" for q in _QUANTILE_NAMES),
]


def flexibility_weekly_summary(
    per_bucket: np.ndarray,
    movable_kwh: np.ndarray,
    connected_count: np.ndarray,
    study_slots: pd.DataFrame,
    fleet_charger_capacity_kw: float,
) -> pd.DataFrame:
    """Contract 3.6e: one row of per-week flexibility figures on the unmanaged path.

    Inputs, each (world, slot) unless stated: ``per_bucket`` is
    ``deferrable_power_per_world`` over every world (bucket, world, slot, kW);
    ``movable_kwh`` the normal path's movable energy (kWh);
    ``connected_count`` the EVs plugged in for the whole slot;
    ``fleet_charger_capacity_kw`` the sum of every EV's home charging power.
    Each figure is computed per world (simulated week) first, then linear
    P10/P50/P90 across worlds:

    - peak deferrable kW: the week's highest half-hour of total deferrable
      power; the modal slot is the half-hour most often the week's peak
      (earliest on ties, as in ``weekly_peak_summary``).
    - hours at least a quarter of capacity: half-hours with total deferrable
      power >= ``DEFERRABLE_CAPACITY_SHARE`` x fleet charger capacity, x 0.5 h.
    - movable energy at 18:00, deferrable kW at 18:00, 22:00 and 03:00 and
      deferrable kW with at least 2 h slack at 19:00: the mean over the
      week's half-hours at that London wall-clock time (one per date).
    - W per plugged-in EV at 18:00, 22:00 and 03:00: 1,000 x the week's sum
      of deferrable kW at that time / its sum of plugged-in EVs, so the
      week's evenings are pooled rather than averaging per-evening ratios;
      NaN in a week with no EV plugged in at that time (left out of the
      quantiles).
    """

    rows = dict(zip(DEFERRABLE_BUCKET_ORDER, deferrable_rows_per_world(per_bucket), strict=True))
    total = rows["total"]
    half_hour = study_slots["local_half_hour"].to_numpy()
    peak_counts = np.bincount(total.argmax(axis=1), minlength=total.shape[1])
    modal = int(peak_counts.argmax())
    per_world = {
        "peak_deferrable_kw": total.max(axis=1),
        "hours_at_least_quarter_capacity": 0.5
        * (total >= DEFERRABLE_CAPACITY_SHARE * fleet_charger_capacity_kw).sum(axis=1),
        "movable_energy_1800_kwh": movable_kwh[:, half_hour == MOVABLE_CHECK_HALF_HOUR].mean(
            axis=1
        ),
    }
    for time, index in REPORT_HALF_HOURS.items():
        at_time = half_hour == index
        plugged = connected_count[:, at_time].sum(axis=1)
        deferrable = total[:, at_time]
        per_world[f"deferrable_kw_{time}"] = deferrable.mean(axis=1)
        per_world[f"w_per_plugged_in_ev_{time}"] = np.divide(
            1000.0 * deferrable.sum(axis=1),
            plugged,
            out=np.full(plugged.shape, np.nan),
            where=plugged > 0,
        )
    per_world["deferrable_at_least_2h_kw_1900"] = rows["at_least_2h"][
        :, half_hour == WAIT_CHECK_HALF_HOUR
    ].mean(axis=1)
    row: dict[str, object] = {
        "path_id": "normal",
        "world_count": total.shape[0],
        "fleet_charger_capacity_kw": float(fleet_charger_capacity_kw),
        "peak_modal_slot_index": int(study_slots["slot_index"].iat[modal]),
        "peak_modal_interval_start_utc": study_slots["interval_start_utc"].iat[modal],
        "peak_modal_interval_start_london": study_slots["interval_start_london"].iat[modal],
        "peak_modal_week_count": int(peak_counts[modal]),
    }
    for name, values in per_world.items():
        row.update(
            zip(
                (f"{name}_{q}" for q in _QUANTILE_NAMES),
                _quantiles(values[~np.isnan(values)]),
                strict=True,
            )
        )
    frame = pd.DataFrame([row], columns=FLEXIBILITY_WEEKLY_COLUMNS)
    counts = ["world_count", "peak_modal_slot_index", "peak_modal_week_count"]
    floats = [column for column in FLEXIBILITY_WEEKLY_COLUMNS if column.endswith(_QUANTILE_NAMES)]
    return frame.astype({**dict.fromkeys(counts, np.int64), **dict.fromkeys(floats, np.float64)})


def average_day_bands(
    group_days: dict[tuple[str, str], _GroupDay],
    study_slots: pd.DataFrame,
) -> pd.DataFrame:
    """Contract 3.8 rows from per-world, per-slot group statistics.

    For each group, path, day type and London half-hour: average each world's
    values over the study slots of that day type at that half-hour (two slots
    on the repeated autumn hour, none on the skipped spring half-hour, which
    gives NaN), then

    - ``connected_share``: P10/P50/P90 across worlds (forecast spread);
    - ``battery_soc_percent``: the median across worlds of the across-EV mean,
      P5 and P95 (population spread).  The mean is not a quantile, so only
      ``low <= high`` is guaranteed for this metric.
    """

    half_hour = study_slots["local_half_hour"].to_numpy()
    slot_day_type = study_slots["day_type"].to_numpy()
    dates = study_slots.drop_duplicates("local_date")
    labels = [f"{h // 2:02d}:{30 * (h % 2):02d}" for h in range(48)]
    rows = []
    for (group_id, path_id), stats in group_days.items():
        world_count = stats.connected_share.shape[0]
        for day_type in DAY_TYPE_ORDER:
            in_type = (
                np.ones(len(half_hour), bool) if day_type == "all" else slot_day_type == day_type
            )
            day_count = (
                len(dates) if day_type == "all" else int(dates["day_type"].eq(day_type).sum())
            )
            base = {
                "group_id": group_id,
                "path_id": path_id,
                "day_type": day_type,
                "world_count": world_count,
                "day_count": day_count,
            }
            for h in range(48):
                chosen = in_type & (half_hour == h)
                if chosen.any():
                    share = stats.connected_share[:, chosen].mean(axis=1)
                    low, centre, high = np.quantile(share, (0.1, 0.5, 0.9), method="linear")
                    soc = [
                        float(np.median(values[:, chosen].mean(axis=1)))
                        for values in (stats.soc_mean, stats.soc_p5, stats.soc_p95)
                    ]
                else:
                    low = centre = high = np.nan
                    soc = [np.nan] * 3
                hh = {"local_half_hour": h, "local_time_label": labels[h]}
                rows.append(
                    {
                        **base,
                        **hh,
                        "metric": "connected_share",
                        "unit": "fraction",
                        "spread": "across_worlds",
                        "centre_stat": "p50",
                        "low_stat": "p10",
                        "high_stat": "p90",
                        "centre": centre,
                        "low": low,
                        "high": high,
                    }
                )
                rows.append(
                    {
                        **base,
                        **hh,
                        "metric": "battery_soc_percent",
                        "unit": "percent",
                        "spread": "across_evs",
                        "centre_stat": "mean_across_evs",
                        "low_stat": "p5_across_evs",
                        "high_stat": "p95_across_evs",
                        "centre": soc[0],
                        "low": soc[1],
                        "high": soc[2],
                    }
                )
    frame = pd.DataFrame(rows)
    group_order = list(dict.fromkeys(group_id for group_id, _ in group_days))
    frame["_g"] = frame["group_id"].map(group_order.index)
    frame["_p"] = frame["path_id"].map(PATH_ORDER.index)
    frame["_d"] = frame["day_type"].map(DAY_TYPE_ORDER.index)
    frame["_m"] = frame["metric"].map(list(FLEET_METRIC_UNITS).index)
    frame = frame.sort_values(["_g", "_p", "_d", "_m", "local_half_hour"], kind="stable")
    columns = [
        "group_id",
        "path_id",
        "day_type",
        "metric",
        "unit",
        "spread",
        "local_half_hour",
        "local_time_label",
        "centre_stat",
        "low_stat",
        "high_stat",
        "centre",
        "low",
        "high",
        "world_count",
        "day_count",
    ]
    frame = frame.loc[:, columns].reset_index(drop=True)
    return _typed(frame, int_columns=("local_half_hour", "world_count", "day_count"))


# --------------------------------------------------------------------------
# Archetype table (contract 3.3)
# --------------------------------------------------------------------------


def cohort_summary(
    cohort_catalogue: Sequence[tuple[str, str, float]],
    cohort_codes: np.ndarray,
    world_kpis: pd.DataFrame,
    weekly_unserved_by_ev: np.ndarray,
) -> pd.DataFrame:
    """Contract 3.3: six archetype rows in source order, including empty ones (audit B1).

    ``cohort_catalogue`` is ``(cohort_id, label, source_population_share)`` in
    source order; ``cohort_codes`` maps each EV to its catalogue position.
    Plug-in columns come from ``plug_in_world_kpis`` (day type ``all``): P50
    across worlds, worlds with a missing value left out; the peak hour is the
    most common per-world modal hour, ties to the earlier hour.
    ``weekly_unserved_by_ev`` is (world, EV) normal-path unserved travel, kWh.
    NaN / <NA> where an archetype has no EVs or no plug-ins.
    """

    all_days = world_kpis.loc[world_kpis["day_type"].eq("all")]
    records = []
    for code, (cohort_id, label, share) in enumerate(cohort_catalogue):
        mask = cohort_codes == code
        ev_count = int(mask.sum())
        record = {
            "cohort_id": cohort_id,
            "cohort_label": label,
            "source_population_share": float(share),
            "ev_count": ev_count,
            "plug_ins_per_ev_week_p50": np.nan,
            "median_plug_in_soc_percent_p50": np.nan,
            "peak_plug_in_local_hour": pd.NA,
            "unserved_travel_kwh_p50": np.nan,
        }
        if ev_count:
            group = all_days.loc[all_days["group_id"].eq(cohort_id)]

            def values(metric: str, rows: pd.DataFrame = group) -> np.ndarray:
                return rows.loc[rows["metric"].eq(metric), "value"].dropna().to_numpy(float)

            record["plug_ins_per_ev_week_p50"] = _quantile(values("plug_ins_per_ev_per_week"), 0.5)
            record["median_plug_in_soc_percent_p50"] = _quantile(
                values("median_plug_in_soc_percent"), 0.5
            )
            modal = values("modal_plug_in_local_hour")
            if len(modal):
                hours, counts = np.unique(modal, return_counts=True)
                record["peak_plug_in_local_hour"] = int(hours[counts == counts.max()].min())
            record["unserved_travel_kwh_p50"] = _quantile(
                weekly_unserved_by_ev[:, mask].sum(axis=1), 0.5
            )
        records.append(record)
    frame = pd.DataFrame(records)
    frame["ev_count"] = frame["ev_count"].astype(np.int64)
    for column in (
        "source_population_share",
        "plug_ins_per_ev_week_p50",
        "median_plug_in_soc_percent_p50",
        "unserved_travel_kwh_p50",
    ):
        frame[column] = frame[column].astype(np.float64)
    frame["peak_plug_in_local_hour"] = frame["peak_plug_in_local_hour"].astype("Int64")
    return frame


# --------------------------------------------------------------------------
# Paired differences (contract 4.1, 4.2) and cost summary (4.9)
# --------------------------------------------------------------------------


def _metric_matrix(fleet_world_intervals: pd.DataFrame, path_id: str, metric: str) -> np.ndarray:
    """(world, slot) matrix of one metric for one path of a contract fleet frame."""

    rows = fleet_world_intervals.loc[fleet_world_intervals["path_id"].eq(path_id)]
    rows = rows.sort_values(["world_id", "slot_index"], kind="stable")
    world_count = rows["world_id"].nunique()
    return rows[metric].to_numpy(dtype=float).reshape(world_count, -1)


def _slot_keys(fleet_world_intervals: pd.DataFrame) -> pd.DataFrame:
    first = fleet_world_intervals.loc[
        fleet_world_intervals["world_id"].eq(fleet_world_intervals["world_id"].min())
        & fleet_world_intervals["path_id"].eq("normal")
    ]
    return first.sort_values("slot_index").loc[:, _SLOT_KEY_COLUMNS].reset_index(drop=True)


def _band_statistics(values: np.ndarray) -> dict[str, np.ndarray]:
    """Mean and linear P10/P50/P90 across axis 0 (worlds)."""

    q = np.quantile(values, (0.1, 0.5, 0.9), axis=0, method="linear")
    return {"mean": values.mean(axis=0), "p10": q[0], "p50": q[1], "p90": q[2]}


def paired_difference_bands(
    fleet_a: pd.DataFrame, path_a: str, fleet_b: pd.DataFrame, path_b: str
) -> pd.DataFrame:
    """B minus A for every fleet metric, **per world first**, then quantiles.

    This is not ``band_b - band_a``: the median of a difference is not the
    difference of medians (decision 0004 item 12).  SoC differences are in
    percentage points.
    """

    keys = _slot_keys(fleet_a)
    frames = []
    for metric, unit in FLEET_METRIC_UNITS.items():
        delta = _metric_matrix(fleet_b, path_b, metric) - _metric_matrix(fleet_a, path_a, metric)
        frame = keys.copy()
        frame.insert(0, "unit", "percentage points" if metric == "battery_soc_percent" else unit)
        frame.insert(0, "metric", metric)
        frame["world_count"] = np.int64(delta.shape[0])
        for stat, column in _band_statistics(delta).items():
            frame[stat] = column
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).loc[
        :, ["metric", "unit", *_SLOT_KEY_COLUMNS, *_BAND_STATS]
    ]


def difference_bands(fleet_world_intervals: pd.DataFrame) -> pd.DataFrame:
    """Contract 4.1: selected minus normal per world and slot, then quantiles."""

    return paired_difference_bands(
        fleet_world_intervals, "normal", fleet_world_intervals, "selected"
    )


def difference_weekly(fleet_world_intervals: pd.DataFrame) -> pd.DataFrame:
    """Contract 4.2: weekly selected-minus-normal per world.

    Import and unserved travel are weekly sums; closing battery is a stock, so
    it is the last slot's value, never a sum.
    """

    frames = []
    for metric, unit in WEEKLY_DIFFERENCE_UNITS.items():
        normal = _metric_matrix(fleet_world_intervals, "normal", metric)
        selected = _metric_matrix(fleet_world_intervals, "selected", metric)
        if metric == "closing_battery_kwh":
            normal_value, selected_value = normal[:, -1], selected[:, -1]
        else:
            normal_value, selected_value = normal.sum(axis=1), selected.sum(axis=1)
        frames.append(
            pd.DataFrame(
                {
                    "world_id": np.arange(len(normal_value), dtype=np.int64),
                    "metric": metric,
                    "unit": unit,
                    "normal_value": normal_value,
                    "selected_value": selected_value,
                    "selected_minus_normal": selected_value - normal_value,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def difference_weekly_bands(
    weekly: pd.DataFrame, value: str = "selected_minus_normal"
) -> pd.DataFrame:
    """Contract 4.2: mean and P10/P50/P90 across worlds of each weekly difference."""

    rows = []
    for metric in dict.fromkeys(weekly["metric"]):
        subset = weekly.loc[weekly["metric"].eq(metric)]
        stats = _band_statistics(subset[value].to_numpy(dtype=float))
        rows.append(
            {
                "metric": metric,
                "unit": subset["unit"].iat[0],
                "world_count": len(subset),
                **{stat: float(number) for stat, number in stats.items()},
            }
        )
    return _typed(pd.DataFrame(rows), int_columns=("world_count",))


def cost_effect_summary(cost_effect: pd.DataFrame) -> pd.DataFrame:
    """Contract 4.9: each illustrative cost component across worlds, separately.

    Component quantiles do not add up to the total's quantile (only the
    per-world values add up), so the view must say so.
    """

    rows = []
    for component, column in COST_COMPONENTS:
        stats = _band_statistics(cost_effect[column].to_numpy(dtype=float))
        rows.append(
            {
                "component": component,
                "source_column": column,
                "unit": "GBP",
                **{stat: float(number) for stat, number in stats.items()},
                "evidence_kind": str(cost_effect["evidence_kind"].iat[0]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Not-recovered magnitude (contract 4.9a, decision 0004 item 45)
# --------------------------------------------------------------------------

NOT_RECOVERED_METRIC_UNITS = {
    "unrecovered_kwh": "kWh per week",
    "unrecovered_share": "fraction",
    "sessions_affected_count": "sessions per week",
}


def sessions_affected_per_world(
    connected: np.ndarray, normal_closing_kwh: np.ndarray, selected_closing_kwh: np.ndarray
) -> np.ndarray:
    """Plug-in sessions per world that end with less energy on the smart path.

    Inputs have shape (world, slot, EV): ``connected`` (bool, shared by both
    paths because smart charging never changes when drivers plug in) and each
    path's closing battery stock (kWh).  A session ends in its last fully
    connected slot (the next slot is not connected, or the horizon ends); it
    counts when the selected stock there is below the normal stock by more
    than the reporting tolerance.  Unlike ``find_sessions`` this keeps
    sessions already plugged in at the study start: the smart charger plans
    those too.  Returns an int64 count per world.
    """

    still_connected_next = np.zeros_like(connected)
    still_connected_next[:, :-1, :] = connected[:, 1:, :]
    session_end = connected & ~still_connected_next
    short = normal_closing_kwh - selected_closing_kwh > NOT_RECOVERED_TOLERANCE_KWH
    return (session_end & short).sum(axis=(1, 2)).astype(np.int64)


def not_recovered_summary(cost_effect: pd.DataFrame) -> pd.DataFrame:
    """Contract 4.9a: the not-recovered magnitude across worlds (world-first).

    One row per ``NOT_RECOVERED_METRIC_UNITS`` metric: ``world_count`` and the
    mean and P10/P50/P90 across worlds of that ``cost_effect`` column, worlds
    with NaN (no normal home import) left out, as in ``smart_charging_summary``.
    """

    rows = []
    for metric, unit in NOT_RECOVERED_METRIC_UNITS.items():
        values = cost_effect[metric].to_numpy(dtype=float)
        present = values[~np.isnan(values)]
        stats = (
            _band_statistics(present)
            if present.size
            else dict.fromkeys(("mean", "p10", "p50", "p90"), np.nan)
        )
        rows.append(
            {
                "metric": metric,
                "unit": unit,
                "world_count": present.size,
                **{stat: float(number) for stat, number in stats.items()},
            }
        )
    return _typed(pd.DataFrame(rows), int_columns=("world_count",))


# --------------------------------------------------------------------------
# Action summary and smart-charging outcomes (contract 4.3-4.6)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSummary:
    """Contract 4.3: the facts behind the action summary sentence."""

    action_id: str
    action_label: str
    departure_margin_hours: float
    forecast_available_at_utc: pd.Timestamp
    vehicle_count: int


def action_summary(
    *,
    departure_margin_hours: float,
    forecast_available_at_utc: pd.Timestamp,
    vehicle_count: int,
) -> ActionSummary:
    """Contract 4.3: what the smart charger does and what it knew.

    Smart charging applies to every EV on the selected path (decision 0004
    item 38), so there is no selected-or-not flag or eligible share to report.
    """

    return ActionSummary(
        action_id=ACTION_ID,
        action_label=ACTION_LABEL,
        departure_margin_hours=float(departure_margin_hours),
        forecast_available_at_utc=pd.Timestamp(forecast_available_at_utc),
        vehicle_count=int(vehicle_count),
    )


SMART_CHARGING_WORLD_COLUMNS = [
    "world_id",
    "normal_average_price_gbp_per_mwh",
    "selected_average_price_gbp_per_mwh",
    "moved_home_import_kwh",
    "moved_home_import_share",
    "early_departure_count",
    "early_departure_shortfall_kwh",
    "evidence_kind",
]
SMART_CHARGING_METRIC_UNITS = {
    "normal_average_price_gbp_per_mwh": "GBP/MWh",
    "selected_average_price_gbp_per_mwh": "GBP/MWh",
    "average_price_change_gbp_per_mwh": "GBP/MWh",
    "moved_home_import_share": "fraction",
    "early_departure_count": "sessions per week",
    "early_departure_shortfall_kwh": "kWh per week",
}
PRICE_BANDS = ("low", "middle", "high")
"""Day-ahead price thirds of each world's study half-hours, cheapest first (contract 4.6)."""


def _average_price(import_kwh: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Import-weighted price per world (GBP/MWh); NaN for a world with no import."""

    total = import_kwh.sum(axis=1)
    cost = (import_kwh * price).sum(axis=1)
    return np.divide(cost, total, out=np.full(total.shape, np.nan), where=total > 0.0)


def smart_charging_world(
    fleet_world_intervals: pd.DataFrame, forecast_prices: pd.DataFrame
) -> pd.DataFrame:
    """Contract 4.4: what smart charging did in each simulated week.

    Per world, from the contract fleet frame (normal and selected, plus the
    optional timed path when the run has it) and that world's synthetic
    day-ahead price per slot:

    - ``*_average_price_gbp_per_mwh``: home import weighted by the day-ahead
      price, per path.  Home import is the customer's tariff leg, bought at
      day-ahead prices (plan B3, decision 0004 item 53), so this is the
      price the fleet paid, the same price the cost effect (4.8) uses.  With
      the optional timed path (decision 0007, model step 2) the run has,
      ``timed_average_price_gbp_per_mwh`` joins them the same way, so the
      three price-paid figures are directly comparable; nothing else below
      is redefined for it (moved energy and early departures stay normal
      versus selected, the smart plan's own reading).
    - ``moved_home_import_kwh``: sum over half-hours of
      ``max(0, normal - selected)`` fleet home import, the energy the
      selected path took in other half-hours; ``moved_home_import_share``
      divides it by the week's normal home import.  Fleet sums net out
      opposite moves by different EVs in the same half-hour, so this is a
      lower bound on energy moved EV by EV.
    - ``early_departure_count`` and ``early_departure_shortfall_kwh``: home
      sessions whose EV left before its smart plan finished, and the
      battery-side energy those plans still owed (the kernel's audit
      columns, selected path).
    """

    normal_import = _metric_matrix(fleet_world_intervals, "normal", "home_import_kwh")
    selected_import = _metric_matrix(fleet_world_intervals, "selected", "home_import_kwh")
    prices = forecast_prices.sort_values(["world_id", "slot_index"], kind="stable")
    price = prices["wholesale_forecast_gbp_per_mwh"].to_numpy(dtype=float)
    price = price.reshape(normal_import.shape)
    moved = np.maximum(normal_import - selected_import, 0.0).sum(axis=1)
    normal_total = normal_import.sum(axis=1)
    early_count = _metric_matrix(fleet_world_intervals, "selected", "early_departure_count")
    early_short = _metric_matrix(fleet_world_intervals, "selected", "early_departure_shortfall_kwh")
    columns = {
        "world_id": np.arange(len(moved), dtype=np.int64),
        "normal_average_price_gbp_per_mwh": _average_price(normal_import, price),
        "selected_average_price_gbp_per_mwh": _average_price(selected_import, price),
        "moved_home_import_kwh": moved,
        "moved_home_import_share": np.divide(
            moved, normal_total, out=np.full(moved.shape, np.nan), where=normal_total > 0.0
        ),
        "early_departure_count": np.rint(early_count.sum(axis=1)).astype(np.int64),
        "early_departure_shortfall_kwh": early_short.sum(axis=1),
        "evidence_kind": "illustrative_synthetic",
    }
    column_order = list(SMART_CHARGING_WORLD_COLUMNS)
    if "timed" in set(fleet_world_intervals["path_id"]):
        timed_import = _metric_matrix(fleet_world_intervals, "timed", "home_import_kwh")
        columns["timed_average_price_gbp_per_mwh"] = _average_price(timed_import, price)
        column_order.insert(
            column_order.index("moved_home_import_kwh"), "timed_average_price_gbp_per_mwh"
        )
    return pd.DataFrame(columns, columns=column_order)


def smart_charging_summary(world: pd.DataFrame) -> pd.DataFrame:
    """Contract 4.5: each smart-charging metric across worlds (world-first).

    ``average_price_change_gbp_per_mwh`` is selected minus normal inside each
    world before the quantiles (decision 0004 item 12).  Worlds with a
    missing value (no home import) are left out and ``world_count`` says how
    many were used.  ``timed_average_price_gbp_per_mwh`` (decision 0007,
    model step 2) joins the metrics only when ``world`` (``smart_charging_world``)
    carries it; it never has a "change" reading of its own.
    """

    metric_units = dict(SMART_CHARGING_METRIC_UNITS)
    if "timed_average_price_gbp_per_mwh" in world.columns:
        metric_units["timed_average_price_gbp_per_mwh"] = "GBP/MWh"
    values = {
        name: world[name].to_numpy(dtype=float)
        for name in metric_units
        if name != "average_price_change_gbp_per_mwh"
    }
    values["average_price_change_gbp_per_mwh"] = (
        values["selected_average_price_gbp_per_mwh"] - values["normal_average_price_gbp_per_mwh"]
    )
    rows = []
    for metric, unit in metric_units.items():
        present = values[metric][~np.isnan(values[metric])]
        stats = (
            _band_statistics(present)
            if present.size
            else dict.fromkeys(("mean", "p10", "p50", "p90"), np.nan)
        )
        rows.append(
            {
                "metric": metric,
                "unit": unit,
                "world_count": present.size,
                **{stat: float(number) for stat, number in stats.items()},
            }
        )
    return _typed(pd.DataFrame(rows), int_columns=("world_count",))


WEEKLY_PEAK_COLUMNS = [
    "path_id",
    "unit",
    "world_count",
    "p10",
    "p50",
    "p90",
    "modal_peak_slot_index",
    "modal_peak_interval_start_utc",
    "modal_peak_interval_start_london",
    "modal_peak_week_count",
    "ratio_world_count",
    "ratio_to_normal_p10",
    "ratio_to_normal_p50",
    "ratio_to_normal_p90",
    "coincidence_world_count",
    "coincidence_factor_p10",
    "coincidence_factor_p50",
    "coincidence_factor_p90",
]


def weekly_peak_summary(
    fleet_world_intervals: pd.DataFrame, *, home_charging_power_kw: float
) -> pd.DataFrame:
    """Contract 4.5a: each simulated week's own peak fleet home import, per path.

    Input: the contract fleet frame (3.4), normal and selected, plus the
    optional timed path when the run has it (decision 0007, model step 2).
    Output: one row per path present (``normal``, ``selected``, optionally
    ``timed``):

    - ``p10``/``p50``/``p90`` (kW): per world, the highest half-hour of fleet
      ``home_import_kw`` in the study week; then linear quantiles across
      worlds.
    - ``modal_peak_*``: the study half-hour that is most often the week's
      peak, and in how many weeks.  Ties (within a week, or between equally
      common half-hours) go to the earliest half-hour.
    - ``ratio_to_normal_*``: per world, this path's peak divided by that
      week's normal-path peak, then quantiles across worlds (1 on the normal
      row).  Worlds whose normal peak is 0 have no ratio and are left out;
      ``ratio_world_count`` says how many were used.
    - ``coincidence_factor_*``: per world, the week's peak divided by the
      charger capacity plugged in at that same half-hour (``connected_count``
      x ``home_charging_power_kw``), then quantiles across worlds (decision
      0004 item 54, plan C4).  1 means every plugged-in EV was drawing full
      power at the peak; the unmanaged path sits below 1 because EVs near
      target have stopped.  Worlds with nothing plugged in at their peak slot
      (a week with no import at all) have no factor and are left out;
      ``coincidence_world_count`` says how many were used.

    Why a per-week peak (decision 0004 items 48-49): the herding finding is
    that smart EVs all pile onto the same cheap half-hours.  Which half-hour
    that is moves from week to week with the day-ahead price, so the peak
    of the per-slot median across weeks smears it out and understates it.
    Taking the maximum inside each week first, then the spread, keeps it
    world-first (item 12); the ratio is also per week first, because the
    ratio of two medians is not the median of the ratio.  The coincidence
    factor is per week first for the same reason, and it uses one home
    charging power for every EV, as the model does (decision 0004 item 34).
    """

    keys = _slot_keys(fleet_world_intervals)
    normal_peak = _metric_matrix(fleet_world_intervals, "normal", "home_import_kw").max(axis=1)
    plugged_in = _metric_matrix(fleet_world_intervals, "normal", "connected_count")
    rows = []
    for path in _present_paths(fleet_world_intervals):
        power = _metric_matrix(fleet_world_intervals, path, "home_import_kw")
        peak = power.max(axis=1)
        # argmax and bincount-argmax both return the first index on ties,
        # which is the earliest half-hour.
        peak_counts = np.bincount(power.argmax(axis=1), minlength=power.shape[1])
        modal = int(peak_counts.argmax())
        ratio = np.divide(
            peak, normal_peak, out=np.full(peak.shape, np.nan), where=normal_peak > 0.0
        )
        present = ratio[~np.isnan(ratio)]
        ratio_q = (
            np.quantile(present, _EV_QUANTILES, method="linear")
            if present.size
            else (np.nan, np.nan, np.nan)
        )
        # Connection is the same on every path (charging policy does not move
        # the EV), so the normal frame's count serves them all, normal,
        # selected and the optional timed path alike.  The peak slot is
        # argmax's, the earliest on a tie.
        capacity = plugged_in[np.arange(len(peak)), power.argmax(axis=1)] * home_charging_power_kw
        coincidence = np.divide(
            peak, capacity, out=np.full(peak.shape, np.nan), where=capacity > 0.0
        )
        coincidence = coincidence[~np.isnan(coincidence)]
        coincidence_q = (
            np.quantile(coincidence, _EV_QUANTILES, method="linear")
            if coincidence.size
            else (np.nan, np.nan, np.nan)
        )
        p10, p50, p90 = np.quantile(peak, _EV_QUANTILES, method="linear")
        rows.append(
            {
                "path_id": path,
                "unit": "kW",
                "world_count": len(peak),
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "modal_peak_slot_index": keys["slot_index"].iat[modal],
                "modal_peak_interval_start_utc": keys["interval_start_utc"].iat[modal],
                "modal_peak_interval_start_london": keys["interval_start_london"].iat[modal],
                "modal_peak_week_count": peak_counts[modal],
                "ratio_world_count": present.size,
                "ratio_to_normal_p10": ratio_q[0],
                "ratio_to_normal_p50": ratio_q[1],
                "ratio_to_normal_p90": ratio_q[2],
                "coincidence_world_count": coincidence.size,
                "coincidence_factor_p10": coincidence_q[0],
                "coincidence_factor_p50": coincidence_q[1],
                "coincidence_factor_p90": coincidence_q[2],
            }
        )
    frame = pd.DataFrame(rows, columns=WEEKLY_PEAK_COLUMNS)
    counts = [
        "world_count",
        "modal_peak_slot_index",
        "modal_peak_week_count",
        "ratio_world_count",
        "coincidence_world_count",
    ]
    quantiles = [
        "p10",
        "p50",
        "p90",
        *(
            f"{name}_{q}"
            for name in ("ratio_to_normal", "coincidence_factor")
            for q in ("p10", "p50", "p90")
        ),
    ]
    return frame.astype({**dict.fromkeys(counts, np.int64), **dict.fromkeys(quantiles, np.float64)})


def price_band_shift(
    fleet_world_intervals: pd.DataFrame, forecast_prices: pd.DataFrame, study_slots: pd.DataFrame
) -> pd.DataFrame:
    """Contract 4.6: home import moved out of each day-ahead price band, per day.

    Each world's study half-hours are split into thirds by that world's own
    day-ahead price (``low`` below its 1/3 quantile, ``high`` above its 2/3
    quantile, ``middle`` otherwise).  Those are the prices its smart charger
    ranked (decision 0004 item 48), so the bands show its choices; one
    shared banding would mislabel a world whose cheap hours fell elsewhere.
    Per world and session night (``study_slots.local_date``, London 12:00 to
    12:00, decision 0004 item 52), ``normal - selected`` home import summed
    over the band's half-hours (positive: energy moved out of the band;
    negative: moved into it), then statistics across worlds.  Keying on the
    night keeps a session's evening peak and the overnight half-hours it
    moved to in one row.  The ``normal_kwh_*`` and ``selected_kwh_*``
    columns are each path's own home import in the band that night, per
    world, then P10/P50/P90 across worlds, so the chart can show both bars
    and not only their difference; with the optional timed path (decision
    0007, model step 2) the run has, ``timed_kwh_*`` joins them the same way.
    The "moved" reading itself (``shifted`` below, normal minus selected)
    always stays that pairing, whatever else the frame carries.
    """

    by_path = {
        path: _metric_matrix(fleet_world_intervals, path, "home_import_kwh")
        for path in _present_paths(fleet_world_intervals)
    }
    delta = by_path["normal"] - by_path["selected"]
    prices = forecast_prices.sort_values(["world_id", "slot_index"], kind="stable")
    price = prices["wholesale_forecast_gbp_per_mwh"].to_numpy(dtype=float).reshape(delta.shape)
    band = price_third_bands(price)
    dates = study_slots["local_date"].to_numpy(dtype=object)
    labels = dict(zip(study_slots["local_date"], study_slots["day_label"], strict=False))
    rows = []
    for local_date in dict.fromkeys(dates):
        for band_id in PRICE_BANDS:
            # (world, slot): this world's half-hours of the band on this date.
            mask = (dates == local_date)[np.newaxis, :] & (band == band_id)
            shifted = np.where(mask, delta, 0.0).sum(axis=1)
            stats = _band_statistics(shifted)
            # Worlds first: each path's band total per world, then quantiles.
            absolute = {
                f"{path}_kwh_{name}": float(value)
                for path, values in by_path.items()
                for name, value in zip(
                    ("p10", "p50", "p90"),
                    np.quantile(
                        np.where(mask, values, 0.0).sum(axis=1), _EV_QUANTILES, method="linear"
                    ),
                    strict=True,
                )
            }
            rows.append(
                {
                    "local_date": local_date,
                    "day_label": labels[local_date],
                    "price_band": band_id,
                    "forecast_price_min_gbp_per_mwh": float(price[mask].min())
                    if mask.any()
                    else np.nan,
                    "forecast_price_max_gbp_per_mwh": float(price[mask].max())
                    if mask.any()
                    else np.nan,
                    # A mean over worlds: each world has its own bands, and
                    # the three means still add up to the date's half-hours.
                    "mean_slot_count": float(mask.sum(axis=1).mean()),
                    "unit": "kWh per day",
                    "world_count": delta.shape[0],
                    **{stat: float(number) for stat, number in stats.items()},
                    **absolute,
                }
            )
    frame = _typed(pd.DataFrame(rows), int_columns=("world_count",))
    frame["local_date"] = frame["local_date"].astype(object)
    return frame


def price_third_bands(price: np.ndarray) -> np.ndarray:
    """Each (world, slot) price's third of its own world: ``"low"``, ``"middle"`` or ``"high"``.

    ``price`` is (world, study slot) day-ahead GBP/MWh.  ``low`` is strictly
    below the world's 1/3 linear quantile, ``high`` strictly above its 2/3
    quantile.  One rule for ``price_band_shift`` and the household cheap-hours
    share (household contract v1 §4 hunk 2), so the two cannot drift.
    """

    low_edge, high_edge = np.quantile(price, (1 / 3, 2 / 3), axis=1, keepdims=True)
    return np.where(price < low_edge, "low", np.where(price > high_edge, "high", "middle"))


FORECAST_PRICE_BAND_KEYS = [
    "slot_index",
    "interval_start_utc",
    "interval_end_utc",
    "interval_start_london",
]


def forecast_price_bands(forecast_prices: pd.DataFrame) -> pd.DataFrame:
    """Contract 4.7: each slot's day-ahead price P10/P50/P90 across worlds (GBP/MWh).

    Each world has its own day-ahead path (decision 0004 item 48), so the
    Plan chart shows their spread per half-hour, like every other time
    series (item 40).  One row per study slot with its slot keys,
    ``world_count`` and linear quantiles across worlds; a price is one value
    per world and slot, so there is nothing to aggregate first.
    """

    prices = forecast_prices.sort_values(["world_id", "slot_index"], kind="stable")
    world_count = int(prices["world_id"].nunique())
    matrix = prices["wholesale_forecast_gbp_per_mwh"].to_numpy(dtype=float)
    matrix = matrix.reshape(world_count, -1)
    p10, p50, p90 = np.quantile(matrix, _EV_QUANTILES, axis=0, method="linear")
    frame = prices.loc[prices["world_id"].eq(prices["world_id"].iat[0]), FORECAST_PRICE_BAND_KEYS]
    return frame.reset_index(drop=True).assign(world_count=world_count, p10=p10, p50=p50, p90=p90)


def _price_matrix(forecast_prices: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    """(world, slot) day-ahead prices (GBP/MWh) and the London start of each slot."""

    prices = forecast_prices.sort_values(["world_id", "slot_index"], kind="stable")
    world_count = int(prices["world_id"].nunique())
    matrix = prices["wholesale_forecast_gbp_per_mwh"].to_numpy(dtype=float)
    first = prices.loc[prices["world_id"].eq(prices["world_id"].iat[0]), "interval_start_london"]
    return matrix.reshape(world_count, -1), first.reset_index(drop=True)


def _whole_session_nights(london: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Session night of each slot, and whether the slots cover that whole night.

    A session night runs London 12:00 to 12:00 (decision 0004 item 52;
    ``clock.session_night_dates``), the stretch a smart charger ranks
    half-hours within.  A night is whole when it has as many slots as it has
    half-hours (46, 48 or 50 with a clock change).  A partial night (the
    slots a spring change pushes past the last noon, or a horizon that does
    not start at noon) is left out of the daily price statistics: the mean
    or cheapest half-hour of part of a night is biased towards the hours it
    happens to cover.  Keying on calendar dates would leave the noon-to-noon
    study's first and last dates partial, so Monday would have no whole day.
    """

    # datetime64[D] -> object gives datetime.date values.
    nights = pd.Series(session_night_dates(pd.DatetimeIndex(london)).astype(object), dtype=object)
    # Noon is added on the wall clock before localising, so a night that
    # holds a clock change still runs from noon to noon.
    noon = pd.to_datetime(nights.astype(str)) + pd.Timedelta(hours=STUDY_START_LOCAL_HOUR)
    next_noon = noon + pd.Timedelta(days=1)
    noon, next_noon = noon.dt.tz_localize(LONDON), next_noon.dt.tz_localize(LONDON)
    half_hours = (next_noon - noon) // HALF_HOUR
    slots_in_night = nights.map(nights.value_counts())
    return nights.to_numpy(), (slots_in_night == half_hours).to_numpy()


PRICE_RELATIVE_COLUMNS = [
    "weekday",
    "weekday_label",
    "local_half_hour",
    "local_time_label",
    "unit",
    "sample_count",
    "world_count",
    "mean",
    "p10",
    "p50",
    "p90",
]


def price_relative_bands(forecast_prices: pd.DataFrame) -> pd.DataFrame:
    """Contract 4.7a: day-ahead price relative to its night's mean, by weekday and half-hour.

    Per world and whole session night (``_whole_session_nights``: London
    12:00 to 12:00, decision 0004 item 52), each half-hour's day-ahead price
    minus that night's mean day-ahead price in that world (GBP/MWh): where
    the half-hour sits within its own night, which is what a smart charger
    choosing between half-hours of one night cares about, with the
    night-to-night level moves taken out (plan D-3).  The relative prices of
    every world and night in a cell are then pooled and summarised with mean
    and linear P10/P50/P90 (``sample_count`` world-nights).  Rows: each
    night's weekday (0 = Monday night, its small hours included) x 48
    wall-clock half-hours, then 48 rows over all nights (``weekday`` <NA>,
    label ``"All days"``).  The repeated autumn half-hour gives two samples a
    night; a cell with no sample is NaN.  Uses only ``forecast_prices``
    columns, so it holds for any price generator.
    """

    matrix, london = _price_matrix(forecast_prices)
    days, whole = _whole_session_nights(london)
    relative = np.full(matrix.shape, np.nan)
    for day in dict.fromkeys(days[whole]):
        columns = days == day
        relative[:, columns] = matrix[:, columns] - matrix[:, columns].mean(axis=1, keepdims=True)
    weekday = np.array([value.weekday() for value in days], dtype=np.int64)
    half_hour = (london.dt.hour * 2 + london.dt.minute // 30).to_numpy()
    rows = []
    for day_of_week in [*range(7), None]:
        on_day = whole if day_of_week is None else whole & (weekday == day_of_week)
        for index in range(48):
            values = relative[:, on_day & (half_hour == index)].ravel()
            q = _quantiles(values)
            rows.append(
                {
                    "weekday": day_of_week,
                    "weekday_label": "All days"
                    if day_of_week is None
                    else WEEKDAY_LABELS[day_of_week],
                    "local_half_hour": index,
                    "local_time_label": f"{index // 2:02d}:{30 * (index % 2):02d}",
                    "unit": "GBP/MWh",
                    "sample_count": values.size,
                    "world_count": matrix.shape[0],
                    "mean": float(values.mean()) if values.size else np.nan,
                    "p10": q[0],
                    "p50": q[1],
                    "p90": q[2],
                }
            )
    frame = _typed(
        pd.DataFrame(rows, columns=PRICE_RELATIVE_COLUMNS),
        int_columns=("local_half_hour", "sample_count", "world_count"),
    )
    # Nullable integer: the all-days rows have no weekday (contract rule 7).
    frame["weekday"] = frame["weekday"].astype("Int64")
    return frame


CHEAPEST_HALF_HOUR_COLUMNS = [
    "world_day_count",
    "local_hour_p10",
    "local_hour_p50",
    "local_hour_p90",
    "local_time_label_p10",
    "local_time_label_p50",
    "local_time_label_p90",
]


def cheapest_half_hour_summary(forecast_prices: pd.DataFrame) -> pd.DataFrame:
    """Contract 4.7b: when in the night the cheapest day-ahead half-hour falls (one row).

    Per world and whole session night (``_whole_session_nights``, London
    12:00 to 12:00, decision 0004 item 52), the London clock time of the
    half-hour with the lowest day-ahead price (the earliest on a tie), as
    decimal hours (18.5 = 18:30).  Then P10/P50/P90 over all world-nights
    with ``method="nearest"``, so each answer is a real half-hour start
    rather than a blend of two (moved from ``scripts/headline_numbers.py``,
    plan E1).  The quantiles are taken on hours since the night's noon and
    turned back into clock times, so they follow the night's own order: a
    cheap 13:00 and a cheap 03:00 are 1 and 15 hours into the night, and the
    only wrap is at noon, where a night starts and ends.
    """

    matrix, london = _price_matrix(forecast_prices)
    days, whole = _whole_session_nights(london)
    clock = (london.dt.hour + london.dt.minute / 60.0).to_numpy()
    cheapest = []
    for day in dict.fromkeys(days[whole]):
        columns = np.flatnonzero(days == day)
        # argmin returns the first (earliest) of tied half-hours.
        cheapest.append(clock[columns[matrix[:, columns].argmin(axis=1)]])
    values = np.concatenate(cheapest) if cheapest else np.array([])
    into_night = (values - STUDY_START_LOCAL_HOUR) % 24.0
    row: dict[str, object] = {"world_day_count": values.size}
    for name, level in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        hour = (
            float((np.quantile(into_night, level, method="nearest") + STUDY_START_LOCAL_HOUR) % 24)
            if values.size
            else np.nan
        )
        row[f"local_hour_{name}"] = hour
        minutes = round(hour * 60) if values.size else None
        row[f"local_time_label_{name}"] = (
            f"{minutes // 60:02d}:{minutes % 60:02d}" if minutes is not None else ""
        )
    frame = pd.DataFrame([row], columns=CHEAPEST_HALF_HOUR_COLUMNS)
    return frame.astype({"world_day_count": np.int64})


# --------------------------------------------------------------------------
# The one entry point run_forecast calls
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Trading overlay frames (trading contract v1 §5.5, §5.6a, §9.1, §9.3, §9.5)
# --------------------------------------------------------------------------
#
# ``market.run_trading`` builds the per-world ledger; these turn it, the
# deviation arrays and a few per-EV sums from the chunk pass into bands,
# KPIs, checks and the Supplier page frames.  Every statistic is taken
# inside each world first and then across worlds (contract results-v2 §1
# rule 3); nothing here changes a position or a pound of the ledger.

_TRADING_STATS = ["world_count", "mean", "p10", "p50", "p90"]
LEDGER_SUMMARY_COLUMNS = ["strategy", "night_index", "metric", "unit", *_TRADING_STATS]
TRADING_KPI_COLUMNS = ["strategy", "metric", "unit", *_TRADING_STATS, "cvar5"]
TRADING_CHECK_COLUMNS = ["check_id", "description", "value", "tolerance", "passed"]
SHOCK_ATTRIBUTION_COLUMNS = ["strategy", "shock_kind", "metric", "unit", *_TRADING_STATS]
OPEN_POSITION_COLUMNS = [
    "strategy",
    "metric",
    "unit",
    "local_half_hour",
    "profile_order",
    *_TRADING_STATS,
]
FLEX_COST_CURVE_COLUMNS = [
    "day_type",
    "local_half_hour",
    "profile_order",
    "metric",
    "threshold_gbp_per_mwh",
    "unit",
    *_TRADING_STATS,
    "risk_charge_gbp_per_mwh",
    "evidence_kind",
]
SHAPE_PREMIUM_COLUMNS = ["path_id", "metric", "unit", *_TRADING_STATS, "evidence_kind"]
HOUSEHOLD_DISTRIBUTION_COLUMNS = [
    "group_id",
    "bin_index",
    "bin_lower_gbp",
    "bin_upper_gbp",
    *_TRADING_STATS,
]
HOUSEHOLD_SUMMARY_COLUMNS = [
    "group_id",
    "statistic",
    "unit",
    "ev_count",
    *_TRADING_STATS,
    "evidence_kind",
]
REVENUE_SEGMENT_COLUMNS = [
    "bucket",
    "segment_type",
    "segment_id",
    "metric",
    "unit",
    "ev_count",
    *_TRADING_STATS,
    "evidence_kind",
]
SUPPLIER_POSITION_COLUMNS = [
    "slot_index",
    "interval_start_utc",
    "interval_start_london",
    "settlement_date",
    "settlement_period",
    "night_index",
    *(f"day_ahead_position_mw_{q}" for q in _QUANTILE_NAMES),
    *(f"final_position_mw_{q}" for q in _QUANTILE_NAMES),
    "baseline_mw_p50",
    "metered_mw_p50",
    "settled_mw_p50",
    *(f"day_ahead_price_gbp_per_mwh_{q}" for q in _QUANTILE_NAMES),
    "price_curve_source",
    "evidence_kind",
]
CONTROL_GROUP_COLUMNS = [
    "night_index",
    "metric",
    "unit",
    "control_ev_count",
    "treated_ev_count",
    *_TRADING_STATS,
    "evidence_kind",
]
TRADING_EVIDENCE = "illustrative_synthetic"

# Reporting grids (trading contract §9.10 Q13), not model assumptions.
COST_CURVE_THRESHOLDS = np.arange(-300.0, 300.0 + 1e-9, 10.0)
HOUSEHOLD_BIN_EDGES = np.arange(-20.0, 40.0 + 1e-9, 1.0)
_WEEKS_PER_YEAR = 52.0
_MONTHS_PER_YEAR = 12.0

_BASE_KPIS = (
    ("net_gbp_per_week", "GBP per week"),
    ("net_gbp_per_ev_year", "GBP per EV per year"),
    ("settled_mwh_per_week", "MWh per week"),
    ("imbalance_volume_share", "fraction"),
    ("baseline_effect_share", "fraction"),
    ("event_delivered_mwh_per_week", "MWh per week"),
)
_TRADER_KPIS = (
    ("flex_margin_gbp_per_week", "GBP per week"),
    ("capture_rate", "fraction"),
    ("value_of_intraday_gbp_per_week", "GBP per week"),
    ("cost_of_uncertainty_gbp_per_week", "GBP per week"),
    ("imbalance_gbp_per_mwh_traded", "GBP per MWh"),
    ("imbalance_share_of_traded", "fraction"),
    ("firmness", "fraction"),
    ("firmness_day_ahead", "fraction"),
    ("flex_margin_gbp_per_mw_year", "GBP per MW per year"),
    ("day_ahead_spread_gbp_per_mwh", "GBP per MWh"),
)
# Intraday dispatch rows (intraday-dispatch-v1 §6.3): the full strategy's
# intraday P&L split into rebalancing and re-optimisation (``full`` only),
# then the energy the dispatch moved, a fleet property on every strategy row.
_DISPATCH_KPIS = (
    ("rebalancing_gbp_per_week", "GBP per week"),
    ("reoptimisation_gbp_per_week", "GBP per week"),
    ("dispatch_moved_mwh_per_week", "MWh per week"),
)
# Baseline erosion (decision 0004 item 67): the settled turn-down valued at
# the day-ahead price against the default baseline and against one learnt
# from smart nights, and the share lost.  A fleet property on every strategy
# row, like the day-ahead spread; outside the ledger.
_EROSION_KPIS = (
    ("settled_value_gbp_per_week", "GBP per week"),
    ("settled_mwh_eroded_per_week", "MWh per week"),
    ("settled_value_eroded_gbp_per_week", "GBP per week"),
    ("baseline_erosion_share", "fraction"),
    ("evening_settled_value_gbp_per_week", "GBP per week"),
    ("evening_settled_value_eroded_gbp_per_week", "GBP per week"),
    ("evening_baseline_erosion_share", "fraction"),
)
EROSION_EVENING_WINDOW = ("16:00", "20:00")
"""London wall-clock window of the headline erosion share (lead ruling on item 67):
where smart charging moves load out of, and where the model-questions Q-4
evidence measured the loss.  Outside it the smart-night baseline, learnt from
only 4 (weekday) or 1 (weekend) other nights of a 7-night study, is noisy and
the one-sided rule settles that noise, so the whole-week share understates
the erosion.  A reporting window, not a model assumption; it differs from the
availability product's 17:00-21:00 evening on purpose."""
# The share rows and the (default value, eroded value) keys they divide.
_EROSION_SHARES = {
    "baseline_erosion_share": ("settled_value_gbp", "settled_value_eroded_gbp"),
    "evening_baseline_erosion_share": (
        "evening_settled_value_gbp",
        "evening_settled_value_eroded_gbp",
    ),
}
_FULL_ONLY_KPIS = frozenset(
    {"value_of_intraday_gbp_per_week", "rebalancing_gbp_per_week", "reoptimisation_gbp_per_week"}
)
# ``cvar5`` is filled only for money-per-period rows (§5.5, §9.1).
_CVAR_KPIS = frozenset(
    {
        "net_gbp_per_week",
        "net_gbp_per_ev_year",
        "flex_margin_gbp_per_week",
        "value_of_intraday_gbp_per_week",
        "cost_of_uncertainty_gbp_per_week",
        "rebalancing_gbp_per_week",
        "reoptimisation_gbp_per_week",
    }
)


def _world_stats(values: np.ndarray) -> dict[str, float]:
    """World count, mean and linear P10/P50/P90 of per-world values, NaN left out."""

    kept = np.asarray(values, dtype=float)
    kept = kept[~np.isnan(kept)]
    q = _quantiles(kept)
    return {
        "world_count": int(len(kept)),
        "mean": float(kept.mean()) if len(kept) else np.nan,
        "p10": q[0],
        "p50": q[1],
        "p90": q[2],
    }


def cvar5(values: np.ndarray) -> float:
    """Mean of the per-world values at or below their own 5th percentile (linear); NaN if none."""

    kept = np.asarray(values, dtype=float)
    kept = kept[~np.isnan(kept)]
    if len(kept) == 0:
        return np.nan
    return float(kept[kept <= np.quantile(kept, 0.05, method="linear")].mean())


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator != 0.0
    )


def _week_by_strategy(week: pd.DataFrame, column: str) -> dict[str, np.ndarray]:
    """{strategy: per-world values of one ``trading_week_world`` column}, world order."""

    return {
        strategy: rows.sort_values("world_id")[column].to_numpy(dtype=float)
        for strategy, rows in week.groupby("strategy", sort=False)
    }


def _ledger_unit(column: str, period: str) -> str:
    if column.endswith("_gbp"):
        return f"GBP per {period}"
    if column.endswith("_kwh"):
        return f"kWh per {period}"
    return f"MWh per {period}"


LEDGER_SUMMARY_METRICS = (
    *market.MONEY_COLUMNS,
    *market.DISPATCH_SPLIT_COLUMNS,
    *market.VOLUME_COLUMNS,
)
"""The ``trading_ledger_summary`` metrics: the money columns of §5.3, the four
"of which" dispatch split columns (intraday-dispatch-v1 §6.3; not buckets, so
the bucket means still add to the net without them), then the volume columns."""


def trading_ledger_summary(ledger: pd.DataFrame, week: pd.DataFrame) -> pd.DataFrame:
    """Contract §5.5: bands of every money and volume column, per strategy, week and night.

    One row per (strategy, night_index, metric): ``night_index`` ``<NA>``
    for the week (from ``trading_week_world``), then nights 0-6 (from
    ``trading_ledger_world``).  Means of the buckets add up to the mean net;
    quantiles do not.
    """

    rows = []
    for strategy in market.STRATEGIES:
        blocks = [(pd.NA, "week", week.loc[week["strategy"].eq(strategy)])]
        nights = ledger.loc[ledger["strategy"].eq(strategy)]
        blocks += [
            (int(n), "night", group) for n, group in nights.groupby("night_index", sort=True)
        ]
        for night, period, frame in blocks:
            for metric in LEDGER_SUMMARY_METRICS:
                rows.append(
                    {
                        "strategy": strategy,
                        "night_index": night,
                        "metric": metric,
                        "unit": _ledger_unit(metric, period),
                        **_world_stats(frame.sort_values("world_id")[metric].to_numpy()),
                    }
                )
    frame = pd.DataFrame(rows, columns=LEDGER_SUMMARY_COLUMNS)
    return frame.astype({"night_index": "Int64", "world_count": np.int64})


def flex_margin(week: pd.DataFrame) -> dict[str, np.ndarray]:
    """{strategy: per-world weekly trading cash on flexibility after the spread} (§9.1).

    ``day_ahead_revenue + intraday_pnl + trading_cost + imbalance``: it
    leaves out what is identical across strategies (baseline effect,
    grid-event payments, unmet-charge penalty) and the commercial terms
    (supplier compensation, customer share), so strategy differences are
    exactly what the trader controls.  Not a ledger bucket.
    """

    parts = ("day_ahead_revenue_gbp", "intraday_pnl_gbp", "trading_cost_gbp", "imbalance_gbp")
    by_part = [_week_by_strategy(week, part) for part in parts]
    return {s: sum(part[s] for part in by_part) for s in market.STRATEGIES}


def day_ahead_spread_per_world(day_ahead: np.ndarray, study_slots: pd.DataFrame) -> np.ndarray:
    """Mean over session nights of each night's max - min day-ahead price, per world (GBP/MWh)."""

    night = study_slots["night_index"].to_numpy()
    spreads = [
        day_ahead[:, night == n].max(axis=1) - day_ahead[:, night == n].min(axis=1)
        for n in np.unique(night)
    ]
    return np.mean(spreads, axis=0)


def dispatch_moved_mwh_per_world(
    dispatch_world_slot: pd.DataFrame | None, world_count: int
) -> np.ndarray:
    """Energy the intraday dispatch moved out of its day-ahead half-hours, MWh per week (world,).

    ``sum_t max(0, -moved_kwh_t) / 1000`` over each world's study slots
    (intraday-dispatch-v1 §6.3): only the turned-down side is counted,
    because over a session night the moves net to about zero (the dispatch
    moves energy in time, not in amount), so a signed sum would hide them.
    A net fleet figure per half-hour: two EVs moving opposite ways in one
    half-hour cancel.  0 by definition with the dispatch off
    (``dispatch_world_slot`` is ``None``).
    """

    if dispatch_world_slot is None:
        return np.zeros(world_count)
    # ``dispatch_world_slot`` is world-major (§7.1), so a reshape is the matrix.
    moved = dispatch_world_slot["moved_kwh"].to_numpy(dtype=float).reshape(world_count, -1)
    return np.maximum(0.0, -moved).sum(axis=1) / 1000.0


def baseline_erosion_per_world(
    settled_kwh: np.ndarray,
    settled_eroded_kwh: np.ndarray,
    day_ahead: np.ndarray,
    study_slots: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Per-world weekly settled volume and value, default vs smart-night baseline (item 67).

    ``settled_kwh`` is ``V`` against the default baseline and
    ``settled_eroded_kwh`` ``V`` against ``market.flexed_history_baseline``
    (world, study slot, kWh); ``day_ahead`` the realised day-ahead price
    (world, study slot, GBP/MWh).  Returns {name: (world,)}:
    ``settled_value_gbp`` and ``settled_value_eroded_gbp`` (``sum V P_DA /
    1000``, the turn-down's value at the day-ahead price, GBP per week) and
    ``settled_mwh_eroded`` (MWh per week), then ``evening_settled_value_gbp``
    and ``evening_settled_value_eroded_gbp``: the same two values over the
    slots of ``study_slots`` whose London label lies in
    ``EROSION_EVENING_WINDOW``.  The day-ahead price values both volumes
    alike so the comparison is the baseline alone, not a strategy's trading
    skill.
    """

    labels = study_slots["local_time_label"].to_numpy(dtype=object)
    start, end = EROSION_EVENING_WINDOW
    evening = (labels >= start) & (labels < end)
    value = settled_kwh * day_ahead / 1000.0
    eroded = settled_eroded_kwh * day_ahead / 1000.0
    return {
        "settled_value_gbp": value.sum(axis=1),
        "settled_value_eroded_gbp": eroded.sum(axis=1),
        "settled_mwh_eroded": settled_eroded_kwh.sum(axis=1) / 1000.0,
        "evening_settled_value_gbp": value[:, evening].sum(axis=1),
        "evening_settled_value_eroded_gbp": eroded[:, evening].sum(axis=1),
    }


def trading_kpis(
    week: pd.DataFrame,
    *,
    vehicle_count: int,
    deferrable_2h_kw_1900: np.ndarray,
    day_ahead_spread: np.ndarray,
    erosion: Mapping[str, np.ndarray],
    dispatch_moved_mwh: np.ndarray | None = None,
) -> pd.DataFrame:
    """Contract §5.5 and §9.1: the headline trading KPIs per strategy, world first.

    ``deferrable_2h_kw_1900`` (world,) is each week's mean unmanaged kW at
    19:00 that can wait at least 2 h (results-v2 3.6e), the denominator of
    ``flex_margin_gbp_per_mw_year``; ``day_ahead_spread`` (world,) is
    ``day_ahead_spread_per_world``.  Rows per strategy: the six §5.5 rows,
    then the §9.1 trader rows (``value_of_intraday`` on ``full`` only).
    ``capture_rate``'s ``mean`` is the ratio of mean margins (lead decision
    §9.10 Q12) and its quantiles and ``world_count`` cover the worlds with a
    positive perfect-foresight margin.  ``cvar5`` is filled for money-per-
    period rows only.  Per-EV-year and per-MW-year rows are extrapolated
    from one simulated week.

    Intraday dispatch rows (intraday-dispatch-v1 §6.3), after the trader
    rows: ``rebalancing_gbp_per_week`` and ``reoptimisation_gbp_per_week``
    on ``full`` only, the weekly ``intraday_pnl_rebalancing_gbp`` and
    ``intraday_pnl_reoptimisation_gbp`` (the intraday P&L the forecast
    updates alone, and the dispatch's response, caused; trading cost stays
    in its own ledger columns), ``cvar5`` filled; and
    ``dispatch_moved_mwh_per_week`` from ``dispatch_moved_mwh`` (world,)
    (``dispatch_moved_mwh_per_world``; ``None``: 0, the dispatch off),
    identical on the three strategy rows because it is a property of the
    one dispatched fleet all of them settle on, like the day-ahead spread.

    Baseline erosion rows (decision 0004 item 67), last, identical on every
    strategy row: ``erosion`` is ``baseline_erosion_per_world``.
    ``baseline_erosion_share`` is ``1 - eroded value / default value`` over
    the week and ``evening_baseline_erosion_share`` the same over
    ``EROSION_EVENING_WINDOW`` (the headline: see that constant); each
    ``mean`` is one minus the ratio of mean values (as ``capture_rate``) and
    the quantiles and ``world_count`` cover the worlds with a positive
    default value.
    """

    col = {name: _week_by_strategy(week, name) for name in market.WEEK_COLUMNS[2:-1]}
    world_count = len(day_ahead_spread)
    moved = np.zeros(world_count) if dispatch_moved_mwh is None else dispatch_moved_mwh
    margin = flex_margin(week)
    perfect = margin["perfect_foresight"]
    rows = []
    for strategy in market.STRATEGIES:
        c = {name: values[strategy] for name, values in col.items()}
        per_world = {
            "net_gbp_per_week": c["net_gbp"],
            "net_gbp_per_ev_year": c["net_gbp"] * _WEEKS_PER_YEAR / vehicle_count,
            "settled_mwh_per_week": c["settled_mwh"],
            "imbalance_volume_share": _ratio(c["abs_imbalance_mwh"], c["settled_mwh"]),
            "baseline_effect_share": _ratio(c["baseline_effect_mwh"], c["settled_mwh"]),
            "event_delivered_mwh_per_week": c["event_delivered_mwh"],
            "flex_margin_gbp_per_week": margin[strategy],
            "value_of_intraday_gbp_per_week": margin["full"] - margin["da_only"],
            "cost_of_uncertainty_gbp_per_week": perfect - margin[strategy],
            "imbalance_gbp_per_mwh_traded": _ratio(c["imbalance_gbp"], c["final_position_mwh"]),
            "imbalance_share_of_traded": _ratio(c["abs_imbalance_mwh"], c["final_position_mwh"]),
            "firmness": _ratio(c["delivered_within_final_mwh"], c["final_position_mwh"]),
            "firmness_day_ahead": _ratio(
                c["delivered_within_day_ahead_mwh"], c["sold_day_ahead_mwh"]
            ),
            "flex_margin_gbp_per_mw_year": _ratio(
                margin[strategy] * _WEEKS_PER_YEAR, deferrable_2h_kw_1900 / 1000.0
            ),
            "day_ahead_spread_gbp_per_mwh": day_ahead_spread,
            "rebalancing_gbp_per_week": c["intraday_pnl_rebalancing_gbp"],
            "reoptimisation_gbp_per_week": c["intraday_pnl_reoptimisation_gbp"],
            "dispatch_moved_mwh_per_week": moved,
            "settled_value_gbp_per_week": erosion["settled_value_gbp"],
            "settled_mwh_eroded_per_week": erosion["settled_mwh_eroded"],
            "settled_value_eroded_gbp_per_week": erosion["settled_value_eroded_gbp"],
            "evening_settled_value_gbp_per_week": erosion["evening_settled_value_gbp"],
            "evening_settled_value_eroded_gbp_per_week": erosion[
                "evening_settled_value_eroded_gbp"
            ],
        }
        for metric, unit in (*_BASE_KPIS, *_TRADER_KPIS, *_DISPATCH_KPIS, *_EROSION_KPIS):
            if metric in _FULL_ONLY_KPIS and strategy != "full":
                continue
            if metric == "capture_rate":
                positive = perfect > 0.0
                stats = _world_stats(margin[strategy][positive] / perfect[positive])
                mean_perfect = perfect.mean()
                stats["mean"] = (
                    float(margin[strategy].mean() / mean_perfect) if mean_perfect > 0.0 else np.nan
                )
                rows.append(
                    {"strategy": strategy, "metric": metric, "unit": unit, **stats, "cvar5": np.nan}
                )
                continue
            if metric in _EROSION_SHARES:
                default, eroded = (erosion[key] for key in _EROSION_SHARES[metric])
                positive = default > 0.0
                stats = _world_stats(1.0 - eroded[positive] / default[positive])
                mean_default = default.mean()
                stats["mean"] = (
                    float(1.0 - eroded.mean() / mean_default) if mean_default > 0.0 else np.nan
                )
                rows.append(
                    {"strategy": strategy, "metric": metric, "unit": unit, **stats, "cvar5": np.nan}
                )
                continue
            values = per_world[metric]
            rows.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "unit": unit,
                    **_world_stats(values),
                    "cvar5": cvar5(values) if metric in _CVAR_KPIS else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=TRADING_KPI_COLUMNS).astype({"world_count": np.int64})


def trading_checks(
    ledger: pd.DataFrame,
    week: pd.DataFrame,
    deviation: pd.DataFrame,
    *,
    day_ahead_position: np.ndarray,
    full_position: np.ndarray,
    revised_slots: np.ndarray,
    commitment_share: float,
    fallback_nights: int,
    missing_nights: int = 0,
    day_ahead_forecast: np.ndarray | None = None,
) -> pd.DataFrame:
    """Contract §5.5 runtime checks as rows; informative rows always pass (review B5, §8 Q6).

    ``day_ahead_position`` ``q`` and ``full_position`` ``f`` are (world,
    slot) kWh; ``revised_slots`` (slot,) marks slots with at least one
    intraday decision.  ``mean_intraday_revision`` compares ``f`` with the
    forecast behind ``q``, ``q / c`` (§9.2), over those slots; with the
    newsvendor rule ``day_ahead_forecast`` is that point forecast ``F_hat``
    and the revision is ``f - F_hat`` (§10.4), since ``q`` there is not a
    share of it.
    """

    residual = ledger["net_gbp"] - ledger[list(market.BUCKETS)].sum(axis=1)
    net_scale = max(1.0, float(ledger["net_gbp"].abs().max()))
    perfect = ledger.loc[ledger["strategy"].eq("perfect_foresight"), "imbalance_gbp"]
    identity = (
        deviation["deviation_kwh"]
        - deviation["true_reduction_kwh"]
        - deviation["baseline_effect_kwh"]
    )
    if day_ahead_forecast is not None:
        reference, revision_text = day_ahead_forecast, "f - F_hat"
    elif commitment_share > 0.0:
        reference, revision_text = day_ahead_position / commitment_share, "f - q/c"
    else:
        reference, revision_text = None, "f - q/c"
    if reference is not None and revised_slots.any():
        revision = (full_position - reference)[:, revised_slots]
        per_world = revision.mean(axis=1)
        mean_revision = float(per_world.mean())
        spread = (
            3.0 * float(per_world.std(ddof=1) / np.sqrt(len(per_world)))
            if len(per_world) > 1
            else np.nan
        )
    else:
        mean_revision = spread = np.nan
    net = _week_by_strategy(week, "net_gbp")
    p10 = {s: _quantile(values, 0.1) for s, values in net.items()}
    ordered = p10["perfect_foresight"] >= p10["full"] >= p10["da_only"]
    # Intraday dispatch split (intraday-dispatch-v1 §6.3): on the ``full``
    # rows each "of which" pair sums to its bucket.  One row for both pairs,
    # its tolerance scaled by the largest bucket as the net check is.
    full = ledger.loc[ledger["strategy"].eq("full")]
    split_residual, split_scale = 0.0, 1.0
    for bucket, parts in (
        ("intraday_pnl_gbp", market.DISPATCH_SPLIT_COLUMNS[:2]),
        ("trading_cost_gbp", market.DISPATCH_SPLIT_COLUMNS[2:]),
    ):
        residual_of_pair = (full[list(parts)].sum(axis=1) - full[bucket]).abs()
        split_residual = max(split_residual, float(residual_of_pair.max()))
        split_scale = max(split_scale, float(full[bucket].abs().max()))
    rows = [
        {
            "check_id": "buckets_sum_to_net",
            "description": "Largest |net - sum of the nine buckets| over ledger rows (GBP).",
            "value": float(residual.abs().max()),
            "tolerance": 1e-9 * net_scale,
        },
        {
            "check_id": "perfect_foresight_zero_imbalance",
            "description": "Largest |imbalance cash| of perfect foresight (GBP).",
            "value": float(perfect.abs().max()),
            "tolerance": 0.0,
        },
        {
            "check_id": "deviation_identity",
            "description": "Largest |D - R - E| over worlds and slots (kWh).",
            "value": float(identity.abs().max()),
            "tolerance": 1e-9,
        },
        {
            "check_id": "mean_intraday_revision",
            "description": (
                "Informative: mean over worlds of the mean full-strategy revision "
                f"{revision_text} on intraday-traded slots (kWh), tolerance 3 standard errors "
                "across worlds; failure expected while the day-ahead forecast bias of contract "
                "4.3 stands."
            ),
            "value": mean_revision,
            "tolerance": spread,
            "passed": True,
        },
        {
            "check_id": "p10_ordering",
            "description": (
                "Informative: 1 when P10 weekly net is perfect foresight >= full >= day-ahead "
                "only; expected, not guaranteed."
            ),
            "value": 1.0 if ordered else 0.0,
            "tolerance": 0.0,
            "passed": True,
        },
        {
            "check_id": "baseline_class_fallback",
            "description": (
                "Informative: study nights whose baseline history held no night of their own "
                "class, so the other class was used."
            ),
            "value": float(fallback_nights),
            "tolerance": 0.0,
            "passed": True,
        },
        {
            # Not in contract §5.5: a warm-up under two nights (hand-built
            # runs only) leaves a night no history it could know (§4.1).
            "check_id": "baseline_missing_nights",
            "description": (
                "Informative: study nights with no baseline history night at or before n - 2 "
                "(a warm-up under two nights); their baseline is 0, so nothing trades or settles."
            ),
            "value": float(missing_nights),
            "tolerance": 0.0,
            "passed": True,
        },
        {
            "check_id": "dispatch_split_sums_to_intraday",
            "description": (
                "Largest |rebalancing + re-optimisation - bucket| over full-strategy ledger "
                "rows, for intraday P&L and for trading cost (GBP)."
            ),
            "value": split_residual,
            "tolerance": 1e-9 * split_scale,
        },
    ]
    for row in rows:
        row.setdefault("passed", bool(abs(row["value"]) <= row["tolerance"]))
    return pd.DataFrame(rows, columns=TRADING_CHECK_COLUMNS).astype(
        {"value": np.float64, "tolerance": np.float64, "passed": bool}
    )


def shock_attribution_summary(week: pd.DataFrame) -> pd.DataFrame:
    """Contract §5.6a: weekly "of which" bands per strategy and slot shock kind.

    Per world, the weekly sum of each "of which" column (P&L on half-hours a
    shock hit, not caused by it), then quantiles.  Means add up to the
    bucket mean; quantiles do not.
    """

    rows = []
    metrics = (
        ("day_ahead_revenue_gbp", "GBP per week"),
        ("intraday_pnl_gbp", "GBP per week"),
        ("imbalance_gbp", "GBP per week"),
        ("settled_mwh", "MWh per week"),
    )
    for strategy in market.STRATEGIES:
        rows_of = week.loc[week["strategy"].eq(strategy)].sort_values("world_id")
        for kind in market.SHOCK_KINDS:
            for metric, unit in metrics:
                column = (
                    f"settled_mwh_{kind}"
                    if metric == "settled_mwh"
                    else f"{metric[: -len('_gbp')]}_{kind}_gbp"
                )
                rows.append(
                    {
                        "strategy": strategy,
                        "shock_kind": kind,
                        "metric": metric,
                        "unit": unit,
                        **_world_stats(rows_of[column].to_numpy()),
                    }
                )
    return pd.DataFrame(rows, columns=SHOCK_ATTRIBUTION_COLUMNS).astype({"world_count": np.int64})


def _half_hour_label(index: int) -> str:
    return f"{index // 2:02d}:{30 * (index % 2):02d}"


def _profile_order(index: int) -> int:
    """Noon-to-noon position of a London half-hour: 12:00 is 0, 11:30 is 47."""

    return (index - 24) % 48


def open_position_profile(
    settled: np.ndarray, positions: Mapping[str, np.ndarray], study_slots: pd.DataFrame
) -> pd.DataFrame:
    """Contract §9.1: the open position ``V - f`` by London half-hour, per strategy.

    Per world, the mean over the week's slots with that London label of
    ``(V - f) / 0.5`` kW (signed: positive delivered more than sold, spilled
    at imbalance) and of ``|V - f| / 0.5``; both copies of a repeated autumn
    half-hour enter the mean.  Perfect-foresight rows are exactly 0.
    """

    half_hour = study_slots["local_half_hour"].to_numpy()
    order = sorted(range(48), key=_profile_order)
    rows = []
    for strategy in market.STRATEGIES:
        open_kw = (settled - positions[strategy]) / 0.5
        for metric, values in (
            ("open_position_kw", open_kw),
            ("abs_open_position_kw", np.abs(open_kw)),
        ):
            for h in order:
                cells = half_hour == h
                per_world = (
                    values[:, cells].mean(axis=1) if cells.any() else np.full(len(values), np.nan)
                )
                rows.append(
                    {
                        "strategy": strategy,
                        "metric": metric,
                        "unit": "kW",
                        "local_half_hour": _half_hour_label(h),
                        "profile_order": _profile_order(h),
                        **_world_stats(per_world),
                    }
                )
    return pd.DataFrame(rows, columns=OPEN_POSITION_COLUMNS).astype(
        {"profile_order": np.int64, "world_count": np.int64}
    )


def shape_premium_summary(
    unmanaged_kwh: np.ndarray, metered_kwh: np.ndarray, day_ahead: np.ndarray
) -> pd.DataFrame:
    """Contract §9.3b: the fleet's load shape against a flat block, unmanaged vs smart.

    Per world over the study: load-weighted price ``sum M P / sum M``,
    baseload price ``mean P``, shape premium (their difference, GBP/MWh;
    positive means the load buys dearer than baseload) and shape cost
    ``premium x sum M / 1000`` (GBP per week), on the normal and selected
    paths and their per-world difference.  ``P`` is the day-ahead price
    after all shocks (and any user curve).
    """

    per_path = {}
    for path, load in (("normal", unmanaged_kwh), ("selected", metered_kwh)):
        volume = load.sum(axis=1)
        weighted = _ratio((load * day_ahead).sum(axis=1), volume)
        baseload = day_ahead.mean(axis=1)
        premium = weighted - baseload
        per_path[path] = {
            "load_weighted_price_gbp_per_mwh": weighted,
            "baseload_price_gbp_per_mwh": baseload,
            "shape_premium_gbp_per_mwh": premium,
            "shape_cost_gbp_per_week": premium * volume / 1000.0,
        }
    per_path["difference"] = {
        metric: per_path["selected"][metric] - per_path["normal"][metric]
        for metric in per_path["normal"]
    }
    units = {
        "load_weighted_price_gbp_per_mwh": "GBP/MWh",
        "baseload_price_gbp_per_mwh": "GBP/MWh",
        "shape_premium_gbp_per_mwh": "GBP/MWh",
        "shape_cost_gbp_per_week": "GBP per week",
    }
    rows = [
        {
            "path_id": path,
            "metric": metric,
            "unit": units[metric],
            **_world_stats(values),
            "evidence_kind": TRADING_EVIDENCE,
        }
        for path, metrics in per_path.items()
        for metric, values in metrics.items()
    ]
    return pd.DataFrame(rows, columns=SHAPE_PREMIUM_COLUMNS).astype({"world_count": np.int64})


def _next_index(flags: np.ndarray) -> np.ndarray:
    """(world, slot, EV): the first slot index >= t where ``flags`` is True (slot count if none)."""

    world_count, slot_count, ev_count = flags.shape
    result = np.empty(flags.shape, dtype=np.int64)
    following = np.full((world_count, ev_count), slot_count, dtype=np.int64)
    for t in range(slot_count - 1, -1, -1):
        following = np.where(flags[:, t, :], t, following)
        result[:, t, :] = following
    return result


def flex_cost_curve_chunk(
    normal_import_kwh: np.ndarray,
    connected: np.ndarray,
    visible_day_ahead: np.ndarray,
    hours_to_departure: np.ndarray,
    night_end: np.ndarray,
    risk_charge_gbp_per_mwh: float,
) -> np.ndarray:
    """Turn-down a supplier could buy at each slot, by price, for a chunk of worlds (§9.3a).

    Inputs: the normal path's per-EV home import (world, slot, EV, kWh) and
    connection state (bool); the day-ahead prices visible at each night's
    day-ahead decision (world, slot, GBP/MWh); hours from each slot to the
    EV's next expected departure (slot, EV, the planner's target); each
    slot's night end (slot,, exclusive study slot index); the risk charge
    ``r`` (GBP/MWh).  Returns (world, slot, threshold + 2) kW: available kW
    at each ``COST_CURVE_THRESHOLDS`` price, then movable kW and all
    charging kW.

    An EV charging at ``t`` (normal import > 0, power ``k``) can move to a
    slot after its unmanaged block ends (the next slot with no import) and
    before ``min(realised unplug, expected departure, night end)``: capping
    at the realised unplug stops a slot the EV had already left counting
    (lead review B2); the block's partial last slot is not used
    (conservative).  Its cost is ``min candidate price - price_t + r``;
    it is available at every threshold at or above the cost.
    """

    world_count, slot_count, _ = normal_import_kwh.shape
    charging = normal_import_kwh > 0.0
    power_kw = normal_import_kwh / 0.5
    block_end = _next_index(~charging)
    unplug = _next_index(~connected)
    departure_slots = np.floor(hours_to_departure * 2.0 + 1e-9).astype(np.int64)
    slots = np.arange(slot_count)
    cap_end = np.minimum(
        np.minimum(unplug, (slots[:, np.newaxis] + departure_slots)[np.newaxis]),
        night_end[np.newaxis, :, np.newaxis],
    )
    world_index, slot_index, ev_index = np.nonzero(charging)
    first = block_end[world_index, slot_index, ev_index]
    last = cap_end[world_index, slot_index, ev_index]
    best = np.full(len(world_index), np.inf)
    for offset in range(int(max(0, (last - first).max(initial=0)))):
        candidate = first + offset
        usable = candidate < last
        price = visible_day_ahead[world_index, np.minimum(candidate, slot_count - 1)]
        best = np.where(usable, np.minimum(best, price), best)
    movable = np.isfinite(best)
    cost = best - visible_day_ahead[world_index, slot_index] + risk_charge_gbp_per_mwh
    thresholds = len(COST_CURVE_THRESHOLDS)
    # Index of the first grid price at or above the cost; the tiny tolerance
    # keeps a cost that lands on the grid (up to float rounding) on it.
    step = np.ceil((cost - COST_CURVE_THRESHOLDS[0]) / 10.0 - 1e-9)
    bin_index = np.clip(np.where(movable, step, thresholds), 0, thresholds).astype(np.int64)
    histogram = np.zeros((world_count, slot_count, thresholds + 1))
    kw = power_kw[world_index, slot_index, ev_index]
    np.add.at(histogram, (world_index, slot_index, bin_index), kw)
    result = np.zeros((world_count, slot_count, thresholds + 2))
    result[..., :thresholds] = np.cumsum(histogram[..., :thresholds], axis=2)
    movable_kw = np.zeros((world_count, slot_count))
    np.add.at(movable_kw, (world_index[movable], slot_index[movable]), kw[movable])
    result[..., thresholds] = movable_kw
    result[..., thresholds + 1] = power_kw.sum(axis=2)
    return result


def flex_cost_curve(
    per_slot: np.ndarray, study_slots: pd.DataFrame, risk_charge_gbp_per_mwh: float
) -> pd.DataFrame:
    """Contract §9.3a: ``flex_cost_curve`` from every world's ``flex_cost_curve_chunk`` values.

    Per world and (day type, London half-hour), the mean over the week's
    slots with that label on nights of that type (both copies of a repeated
    autumn half-hour), NaN for a world with none; then quantiles across
    worlds.  Day type goes by the night's date ``D_n``.  Label: turn-down in
    each simulated week's sessions, costed at day-ahead prices visible at
    the day-ahead decision (overnight half-hours on the typical shape).
    """

    half_hour = study_slots["local_half_hour"].to_numpy()
    day_type = study_slots["day_type"].to_numpy()
    metrics = [
        *(("available_kw", float(p)) for p in COST_CURVE_THRESHOLDS),
        ("movable_kw", np.nan),
        ("charging_kw", np.nan),
    ]
    rows = []
    for kind in DAY_TYPE_ORDER:
        in_type = np.ones(len(day_type), dtype=bool) if kind == "all" else day_type == kind
        for h in sorted(range(48), key=_profile_order):
            cells = in_type & (half_hour == h)
            per_world = (
                per_slot[:, cells, :].mean(axis=1)
                if cells.any()
                else np.full((per_slot.shape[0], per_slot.shape[2]), np.nan)
            )
            for index, (metric, threshold) in enumerate(metrics):
                rows.append(
                    {
                        "day_type": kind,
                        "local_half_hour": _half_hour_label(h),
                        "profile_order": _profile_order(h),
                        "metric": metric,
                        "threshold_gbp_per_mwh": threshold,
                        "unit": "kW",
                        **_world_stats(per_world[:, index]),
                        "risk_charge_gbp_per_mwh": float(risk_charge_gbp_per_mwh),
                        "evidence_kind": TRADING_EVIDENCE,
                    }
                )
    return pd.DataFrame(rows, columns=FLEX_COST_CURVE_COLUMNS).astype(
        {"profile_order": np.int64, "world_count": np.int64}
    )


def household_chunk(
    by_path: Mapping[str, Mapping[str, np.ndarray]],
    day_ahead: np.ndarray,
    settled: np.ndarray,
    night_starts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-EV sums for the household and segment frames, for a chunk of worlds (§9.3c).

    ``by_path`` holds the per-EV arrays of both paths (world, slot, EV);
    ``day_ahead`` and ``settled`` ``V`` are (world, slot) for the same
    worlds.  Returns (world, EV) arrays: ``home_cost_gbp`` (selected minus
    normal home import at the day-ahead price), ``public_import_kwh``,
    ``unserved_kwh`` (selected minus normal) and ``closing_kwh`` (selected
    minus normal stock at the study end), and ``reduction_kwh`` (world, EV,
    night): the EV's own positive true reduction ``max(0, u - m)`` in the
    night's settled slots (``V > 0``), the allocation driver.
    """

    normal, selected = by_path["normal"], by_path["selected"]
    difference = selected["home_grid_import_kwh"] - normal["home_grid_import_kwh"]
    settles = (settled > 0.0)[:, :, np.newaxis]
    reduction = np.where(settles, np.maximum(-difference, 0.0), 0.0)
    return {
        "home_cost_gbp": (difference * day_ahead[:, :, np.newaxis]).sum(axis=1) / 1000.0,
        "public_import_kwh": (
            selected["public_grid_import_kwh"] - normal["public_grid_import_kwh"]
        ).sum(axis=1),
        "unserved_kwh": (
            selected["unserved_travel_battery_kwh"] - normal["unserved_travel_battery_kwh"]
        ).sum(axis=1),
        "closing_kwh": selected["closing_battery_kwh"][:, -1, :]
        - normal["closing_battery_kwh"][:, -1, :],
        "reduction_kwh": np.moveaxis(np.add.reduceat(reduction, night_starts, axis=1), 1, 2),
    }


def _pro_rata(world_value: np.ndarray, drivers: np.ndarray) -> np.ndarray:
    """Split each world's value over EVs in proportion to non-negative drivers (world, EV)."""

    total = drivers.sum(axis=1, keepdims=True)
    return np.divide(
        world_value[:, np.newaxis] * drivers,
        total,
        out=np.zeros(drivers.shape),
        where=total > 0.0,
    )


def household_saving_gbp(
    per_ev: Mapping[str, np.ndarray], cost_effect: pd.DataFrame, public_rate_gbp_per_kwh: float
) -> np.ndarray:
    """Each EV's customer saving (world, EV), GBP per week: minus its share of the cost effect.

    The home-import and public-charge parts are linear, so each EV's own
    part sums to the world's ``cost_effect`` column exactly.  The
    unrecovered-energy and unserved-travel values are not linear (the world
    values the fleet's net closing shortfall, so one EV's surplus offsets
    another's shortfall), so each world's value is split over EVs in
    proportion to their own positive shortfall; the parts still sum to the
    world's column.
    """

    ordered = cost_effect.sort_values("world_id")
    unrecovered = ordered["illustrative_unrecovered_energy_value_gbp"].to_numpy(dtype=float)
    unserved = ordered["illustrative_unserved_travel_value_gbp"].to_numpy(dtype=float)
    cost = (
        per_ev["home_cost_gbp"]
        + per_ev["public_import_kwh"] * public_rate_gbp_per_kwh
        + _pro_rata(unrecovered, np.maximum(-per_ev["closing_kwh"], 0.0))
        + _pro_rata(unserved, np.maximum(per_ev["unserved_kwh"], 0.0))
    )
    return -cost


def allocation_weights(reduction_kwh: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """``a[w, i, n]``: each treated EV's share of a night's settled flexibility (§9.3c).

    ``reduction_kwh`` (world, EV, night) is the EV's positive true reduction
    in the night's settled slots.  Equal shares over treated EVs when a
    night has none; control EVs get 0.  Weights sum to 1 per (world, night).
    It is an allocation of fleet settlement, not a per-household settlement.
    """

    drivers = np.where(treated[np.newaxis, :, np.newaxis], reduction_kwh, 0.0)
    total = drivers.sum(axis=1, keepdims=True)
    equal = treated[np.newaxis, :, np.newaxis] / treated.sum()
    return np.where(total > 0.0, drivers / np.where(total > 0.0, total, 1.0), equal)


def household_frames(
    saving_gbp: np.ndarray,
    weights: np.ndarray,
    customer_share_gbp: np.ndarray,
    groups: Mapping[str, np.ndarray],
    reduction_kwh: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Contract §9.3c: value per household per month and its spread across EVs.

    ``saving_gbp`` (world, EV) per week; ``weights`` (world, EV, night)
    from ``allocation_weights``; ``customer_share_gbp`` (world, night) the
    ``full`` strategy's ``customer_revenue_share_gbp`` (negative for the
    aggregator, so the customer receives minus it); ``groups`` {group_id:
    (EV,) bool of treated EVs}; ``reduction_kwh`` (world, EV, night) each
    EV's positive true reduction in the night's settled slots.  ``value =
    (saving + share) x 52 / 12``, extrapolated from one simulated week.
    Customer value only: never add it to the aggregator's net (the same
    shifted energy earns both).

    Returns the distribution and summary frames and ``per_ev`` (supplier
    contract §4.1): ``value_gbp_per_month`` (world, EV) the ``value`` above
    and ``earning`` (world, EV) bool, True when the EV delivered turn-down
    in at least one settled slot (a physical test, not the sign of its cash).
    """

    share = np.einsum("win,wn->wi", weights, -customer_share_gbp)
    monthly = _WEEKS_PER_YEAR / _MONTHS_PER_YEAR
    value = (saving_gbp + share) * monthly
    edges = np.concatenate([[-np.inf], HOUSEHOLD_BIN_EDGES, [np.inf]])
    bin_index = np.searchsorted(HOUSEHOLD_BIN_EDGES, value, side="right")
    distribution, summary = [], []
    unit = "GBP per household per month"
    for group_id, members in groups.items():
        count = int(members.sum())
        in_group = bin_index[:, members]
        for b in range(len(edges) - 1):
            distribution.append(
                {
                    "group_id": group_id,
                    "bin_index": b,
                    "bin_lower_gbp": float(edges[b]),
                    "bin_upper_gbp": float(edges[b + 1]),
                    **_world_stats((in_group == b).mean(axis=1)),
                }
            )
        values = value[:, members]
        per_world = {
            "mean_value": (values.mean(axis=1), unit),
            "p10_across_evs": (np.quantile(values, 0.1, axis=1, method="linear"), unit),
            "p50_across_evs": (np.quantile(values, 0.5, axis=1, method="linear"), unit),
            "p90_across_evs": (np.quantile(values, 0.9, axis=1, method="linear"), unit),
            "share_worse_off": ((values < 0.0).mean(axis=1), "fraction"),
            "mean_customer_saving": ((saving_gbp[:, members] * monthly).mean(axis=1), unit),
            "mean_revenue_share": ((share[:, members] * monthly).mean(axis=1), unit),
        }
        for statistic, (values_per_world, statistic_unit) in per_world.items():
            summary.append(
                {
                    "group_id": group_id,
                    "statistic": statistic,
                    "unit": statistic_unit,
                    "ev_count": count,
                    **_world_stats(values_per_world),
                    "evidence_kind": TRADING_EVIDENCE,
                }
            )
    return (
        pd.DataFrame(distribution, columns=HOUSEHOLD_DISTRIBUTION_COLUMNS).astype(
            {"bin_index": np.int64, "world_count": np.int64}
        ),
        pd.DataFrame(summary, columns=HOUSEHOLD_SUMMARY_COLUMNS).astype(
            {"ev_count": np.int64, "world_count": np.int64}
        ),
        {"value_gbp_per_month": value, "earning": reduction_kwh.sum(axis=2) > 0.0},
    )


def revenue_by_segment(
    ledger: pd.DataFrame,
    weights: np.ndarray,
    segments: Sequence[tuple[str, str, np.ndarray]],
) -> pd.DataFrame:
    """Contract §9.3d: the ``full`` strategy's buckets allocated to EVs and summed by segment.

    ``weights`` (world, EV, night) from ``allocation_weights``;
    ``segments`` ``(segment_type, segment_id, (EV,) bool of treated EVs)``
    in order: ``all``, each cohort, then each zone when the population has
    zones.  Per world, the ``all`` segment equals ``trading_week_world``
    for ``full`` and each segment type sums to it; a penalty or imbalance
    caused by one EV is spread by the flexibility share, not traced to it.
    """

    full = ledger.loc[ledger["strategy"].eq("full")].sort_values(["world_id", "night_index"])
    world_count, _, night_count = weights.shape
    rows = []
    for bucket in market.MONEY_COLUMNS:
        by_night = full[bucket].to_numpy(dtype=float).reshape(world_count, night_count)
        per_ev = np.einsum("win,wn->wi", weights, by_night)
        for segment_type, segment_id, members in segments:
            count = int(members.sum())
            total = per_ev[:, members].sum(axis=1)
            for metric, unit, values in (
                ("gbp_per_week", "GBP per week", total),
                (
                    "gbp_per_ev_per_week",
                    "GBP per EV per week",
                    total / count if count else np.full(world_count, np.nan),
                ),
            ):
                rows.append(
                    {
                        "bucket": bucket,
                        "segment_type": segment_type,
                        "segment_id": segment_id,
                        "metric": metric,
                        "unit": unit,
                        "ev_count": count,
                        **_world_stats(values),
                        "evidence_kind": TRADING_EVIDENCE,
                    }
                )
    return pd.DataFrame(rows, columns=REVENUE_SEGMENT_COLUMNS).astype(
        {"ev_count": np.int64, "world_count": np.int64}
    )


def supplier_positions(
    study_slots: pd.DataFrame,
    *,
    day_ahead_position: np.ndarray,
    full_position: np.ndarray,
    baseline: np.ndarray,
    metered: np.ndarray,
    settled: np.ndarray,
    day_ahead: np.ndarray,
    price_curve_source: str,
) -> pd.DataFrame:
    """Contract §9.3e: the per-slot positions table and CSV export, strategy ``full``.

    Volumes are (world, slot) kWh per half-hour, shown in MW (``/ 0.5 /
    1000``); quantiles are across worlds per slot and do not add across
    slots.  ``settlement_period`` is the 1-based London half-hour of the
    slot's London date, counted in elapsed time from local midnight, so a
    clock-change date runs 1-46 or 1-50.
    """

    def mw(values: np.ndarray) -> np.ndarray:
        return values / 0.5 / 1000.0

    def quantiles(values: np.ndarray) -> np.ndarray:
        return np.quantile(values, (0.1, 0.5, 0.9), axis=0, method="linear")

    london = pd.DatetimeIndex(study_slots["interval_start_london"])
    midnight = london.normalize()
    period = ((london - midnight) // pd.Timedelta(minutes=30)).to_numpy() + 1
    frame = pd.DataFrame(
        {
            "slot_index": study_slots["slot_index"].to_numpy(dtype=np.int64),
            "interval_start_utc": study_slots["interval_start_utc"],
            "interval_start_london": study_slots["interval_start_london"],
            "settlement_date": pd.Series(list(london.strftime("%Y-%m-%d")), dtype=object),
            "settlement_period": period.astype(np.int64),
            "night_index": study_slots["night_index"].to_numpy(dtype=np.int64),
        }
    )
    for name, values in (
        ("day_ahead_position_mw", mw(day_ahead_position)),
        ("final_position_mw", mw(full_position)),
        ("day_ahead_price_gbp_per_mwh", day_ahead),
    ):
        for q, row in zip(_QUANTILE_NAMES, quantiles(values), strict=True):
            frame[f"{name}_{q}"] = row
    for name, values in (("baseline", baseline), ("metered", metered), ("settled", settled)):
        frame[f"{name}_mw_p50"] = np.quantile(mw(values), 0.5, axis=0, method="linear")
    frame["price_curve_source"] = price_curve_source
    frame["evidence_kind"] = TRADING_EVIDENCE
    return frame.loc[:, SUPPLIER_POSITION_COLUMNS]


def control_group_summary(
    trading: market.TradingRun,
    unmanaged_kwh: np.ndarray,
    control: np.ndarray,
    study_slots: pd.DataFrame,
) -> pd.DataFrame | None:
    """Contract §9.5: baseline bias the control group estimates against the true bias.

    Per world and night (and week), over the slots where the fleet settles
    (``V > 0``): estimated bias ``mean (B_c - U_c) / (n_c x 0.5)`` kW per
    EV from the never-flexed control group, true bias ``mean (B_T - U_T) /
    (n_t x 0.5)`` of the treated group (``B_T = B - B_c``, ``U_T = U -
    U_c``; ``U_T`` is known only inside the simulation), and their
    difference.  NaN for a world or night with no settled slot.  ``None``
    when the control group is off.
    """

    if trading.control_baseline is None:
        return None
    n_control, n_treated = int(control.sum()), int((~control).sum())
    settles = trading.settled > 0.0
    estimated = (trading.control_baseline - trading.control_unmanaged) / (n_control * 0.5)
    true = (
        (trading.baseline - trading.control_baseline) - (unmanaged_kwh - trading.control_unmanaged)
    ) / (n_treated * 0.5)
    night = study_slots["night_index"].to_numpy()

    def masked_mean(values: np.ndarray, cells: np.ndarray) -> np.ndarray:
        use = settles & cells[np.newaxis, :]
        return _ratio(np.where(use, values, 0.0).sum(axis=1), use.sum(axis=1))

    rows = []
    periods = [(pd.NA, np.ones(len(night), dtype=bool))]
    periods += [(int(n), night == n) for n in np.unique(night)]
    for night_index, cells in periods:
        est, tru = masked_mean(estimated, cells), masked_mean(true, cells)
        for metric, values in (
            ("estimated_bias_kw_per_ev", est),
            ("true_bias_kw_per_ev", tru),
            ("estimation_error_kw_per_ev", est - tru),
        ):
            rows.append(
                {
                    "night_index": night_index,
                    "metric": metric,
                    "unit": "kW per EV",
                    "control_ev_count": n_control,
                    "treated_ev_count": n_treated,
                    **_world_stats(values),
                    "evidence_kind": TRADING_EVIDENCE,
                }
            )
    return pd.DataFrame(rows, columns=CONTROL_GROUP_COLUMNS).astype(
        {
            "night_index": "Int64",
            "control_ev_count": np.int64,
            "treated_ev_count": np.int64,
            "world_count": np.int64,
        }
    )


# --------------------------------------------------------------------------
# Events, zones and shocks (trading contract v1 §5.6-§5.8; decision 0004
# items 55 and 56)
# --------------------------------------------------------------------------

EVENT_SLOTS_BEFORE = 4
"""Slots (2 h) shown before an event window, where pre-window rebound shows."""
EVENT_SLOTS_AFTER = 8
"""Slots (4 h) shown after an event window, where post-window rebound shows."""

EVENT_RESPONSE_COLUMNS = [
    "event_id",
    "event_type",
    "scope",
    "series",
    "metric",
    "unit",
    "slot_index",
    "interval_start_utc",
    "interval_start_london",
    "relative_slot",
    "in_window",
    *_BAND_STATS,
]
ZONE_IMPORT_BAND_COLUMNS = [
    "zone_id",
    "zone_label",
    "path_id",
    "slot_index",
    "interval_start_utc",
    "interval_start_london",
    "ev_count",
    "headroom_kw",
    *_BAND_STATS,
    "above_headroom_world_share",
]
ZONE_SUMMARY_COLUMNS = [
    "zone_id",
    "zone_label",
    "path_id",
    "share",
    "ev_count",
    "headroom_kw",
    "peak_kw_p10",
    "peak_kw_p50",
    "peak_kw_p90",
    "hours_above_headroom_p10",
    "hours_above_headroom_p50",
    "hours_above_headroom_p90",
]
MARKET_SHOCK_COLUMNS = [
    "world_id",
    "shock_id",
    "source",
    "shock_class",
    "direction",
    "known_day_ahead",
    "start_slot_index",
    "start_utc",
    "start_london",
    "duration_slots",
    "size_gw",
    "in_study",
    "peak_price_increment_gbp_per_mwh",
    "evidence_kind",
]
SHOCK_SUMMARY_COLUMNS = [
    "shock_class",
    "known_day_ahead",
    "direction",
    "world_count",
    "count_per_week_p10",
    "count_per_week_p50",
    "count_per_week_p90",
    "size_gw_p50",
    "peak_price_increment_gbp_per_mwh_p50",
]
SHOCK_CLASS_ORDER = ("mild", "big", "scripted")


def _zone_matrix(zone_world: pd.DataFrame, zone_id: str, path_id: str, metric: str) -> np.ndarray:
    """(world, slot) matrix of one zone's metric on one path (rows are world-major)."""

    rows = zone_world.loc[zone_world["zone_id"].eq(zone_id) & zone_world["path_id"].eq(path_id)]
    return rows[metric].to_numpy(dtype=float).reshape(rows["world_id"].nunique(), -1)


def zone_import_bands(
    zone_world: pd.DataFrame,
    study_slots: pd.DataFrame,
    zone_labels: dict[str, str],
    zone_headroom_kw: Sequence[float],
) -> pd.DataFrame:
    """Trading contract §5.7: each zone's home import (kW) per path and slot across worlds.

    ``zone_world`` is ``zone_world_intervals``; ``zone_labels`` maps zone id
    to label in ``ZONE_IDS`` order and ``zone_headroom_kw`` gives each
    zone's reported headroom (kW, NaN = none).  Per world first, then mean
    and linear P10/P50/P90; ``above_headroom_world_share`` is the share of
    worlds whose zone import is above the headroom in that slot (NaN with
    no headroom).  Headroom is reported, never enforced (contract §3).
    Means add up to the fleet mean; the quantiles do not.  One row per path
    ``zone_world`` actually carries: normal and selected, plus the optional
    timed path (decision 0007, model step 2) when the run has it.
    """

    keys = study_slots.loc[:, ["slot_index", "interval_start_utc", "interval_start_london"]]
    paths = _present_paths(zone_world)
    frames = []
    for (zone_id, label), headroom in zip(zone_labels.items(), zone_headroom_kw, strict=True):
        ev_count = zone_world.loc[zone_world["zone_id"].eq(zone_id), "unit_count"].iat[0]
        for path_id in paths:
            values = _zone_matrix(zone_world, zone_id, path_id, "home_import_kw")
            frame = keys.copy()
            frame.insert(0, "path_id", path_id)
            frame.insert(0, "zone_label", label)
            frame.insert(0, "zone_id", zone_id)
            frame["ev_count"] = np.int64(ev_count)
            frame["headroom_kw"] = float(headroom)
            frame["world_count"] = np.int64(values.shape[0])
            frame = frame.assign(**_band_statistics(values))
            frame["above_headroom_world_share"] = (
                (values > headroom).mean(axis=0) if np.isfinite(headroom) else np.nan
            )
            frames.append(frame)
    return pd.concat(frames, ignore_index=True).loc[:, ZONE_IMPORT_BAND_COLUMNS]


def zone_summary(
    zone_world: pd.DataFrame,
    zone_labels: dict[str, str],
    zone_shares: Sequence[float],
    zone_headroom_kw: Sequence[float],
) -> pd.DataFrame:
    """Trading contract §5.8: per (zone, path), the weekly peak and hours above headroom.

    Per world first: the week's own peak of zone ``home_import_kw`` (as
    ``weekly_peak_summary`` does for the fleet, so a peak that moves between
    weeks is not smeared) and the half-hours above headroom × 0.5 h; then
    linear P10/P50/P90 across worlds.  Hours are NaN with no headroom.
    ``share`` is the zone-share assumption; ``ev_count`` the EVs drawn.  One
    row per path ``zone_world`` actually carries, as ``zone_import_bands``.
    """

    paths = _present_paths(zone_world)
    rows = []
    for (zone_id, label), share, headroom in zip(
        zone_labels.items(), zone_shares, zone_headroom_kw, strict=True
    ):
        ev_count = zone_world.loc[zone_world["zone_id"].eq(zone_id), "unit_count"].iat[0]
        for path_id in paths:
            values = _zone_matrix(zone_world, zone_id, path_id, "home_import_kw")
            peak = np.quantile(values.max(axis=1), _EV_QUANTILES, method="linear")
            hours = (
                np.quantile((values > headroom).sum(axis=1) * 0.5, _EV_QUANTILES, method="linear")
                if np.isfinite(headroom)
                else (np.nan, np.nan, np.nan)
            )
            rows.append(
                {
                    "zone_id": zone_id,
                    "zone_label": label,
                    "path_id": path_id,
                    "share": float(share),
                    "ev_count": int(ev_count),
                    "headroom_kw": float(headroom),
                    "peak_kw_p10": peak[0],
                    "peak_kw_p50": peak[1],
                    "peak_kw_p90": peak[2],
                    "hours_above_headroom_p10": hours[0],
                    "hours_above_headroom_p50": hours[1],
                    "hours_above_headroom_p90": hours[2],
                }
            )
    frame = pd.DataFrame(rows, columns=ZONE_SUMMARY_COLUMNS)
    return frame.astype(
        {"share": np.float64, "ev_count": np.int64}
        | {column: np.float64 for column in ZONE_SUMMARY_COLUMNS[5:]}
    )


# --------------------------------------------------------------------------
# Intraday dispatch (intraday-dispatch-v1 contract §7, lane K3)
# --------------------------------------------------------------------------

DISPATCH_WORLD_SLOT_COLUMNS = [
    "world_id",
    "slot_index",
    "night_index",
    "interval_start_utc",
    "interval_start_london",
    "day_ahead_plan_kwh",
    "dispatched_kwh",
    "locked_kwh",
    "free_kwh",
    "free_day_ahead_plan_kwh",
    "moved_kwh",
    "replan_count",
    "latest_close_gbp_per_mwh",
    "evidence_kind",
]
DISPATCH_BAND_COLUMNS = [
    "series",
    "metric",
    "unit",
    "slot_index",
    "interval_start_utc",
    "interval_start_london",
    *_BAND_STATS,
    "evidence_kind",
]
DISPATCH_SPLIT_COLUMNS = [
    "cohort_id",
    "ev_count",
    "locked_count",
    "free_count",
    "commitment_share",
    "replan_threshold_gbp_per_mwh",
    "evidence_kind",
]


def _evaluation_close_matrix(evaluation_prices: pd.DataFrame, world_count: int) -> np.ndarray:
    """(world, study slot) intraday close, world-major, from ``evaluation_prices`` (results-v2)."""

    rows = evaluation_prices.sort_values(["world_id", "slot_index"], kind="stable")
    column = rows["evaluation_context_price_gbp_per_mwh"].to_numpy(dtype=float)
    return column.reshape(world_count, -1)


def dispatch_world_slot(
    dispatch_sums,
    fleet_world_intervals: pd.DataFrame,
    study_slots: pd.DataFrame,
    evaluation_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Intraday dispatch contract v1 §7.1: fleet dispatch response, one row per (world, study slot).

    ``dispatch_sums`` is the kernel's ``DispatchSums`` holder (contract §5.2,
    the ``ZoneSums`` pattern; K1 owns the real one in ``action.py`` /
    ``physics.py``, duck-typed here on its documented fields):
    ``day_ahead_home_import_kwh``, ``locked_home_import_kwh``,
    ``free_home_import_kwh`` and ``free_day_ahead_home_import_kwh`` (fleet
    home import, kWh per half-hour, on the day-ahead plan reference path and
    its locked/free split) and ``replan_count``, all ``(world, study slot)``.
    ``dispatched_kwh`` is read from ``fleet_world_intervals``'s selected path
    rather than summed from locked/free again, so the two frames agree by
    construction and the validator's identity check (contract §8) is a
    check on the inputs, not a tautology. ``moved_kwh`` is
    ``dispatched_kwh - day_ahead_plan_kwh``: negative is turned down against
    the day-ahead plan in that half-hour, positive is turned up (§7.1
    caption: sums to about zero over a session night, apart from early
    departures and public top-ups). ``latest_close_gbp_per_mwh`` is repeated
    from ``evaluation_prices`` so the Response view needs only this one frame
    for the intraday close.
    """

    world_count = int(dispatch_sums.day_ahead_home_import_kwh.shape[0])
    keys = study_slots.loc[
        :, ["slot_index", "night_index", "interval_start_utc", "interval_start_london"]
    ]
    dispatched = _metric_matrix(fleet_world_intervals, "selected", "home_import_kwh")
    close = _evaluation_close_matrix(evaluation_prices, world_count)
    frames = []
    for world in range(world_count):
        frame = keys.copy()
        frame.insert(0, "world_id", np.int64(world))
        frame["day_ahead_plan_kwh"] = dispatch_sums.day_ahead_home_import_kwh[world]
        frame["dispatched_kwh"] = dispatched[world]
        frame["locked_kwh"] = dispatch_sums.locked_home_import_kwh[world]
        frame["free_kwh"] = dispatch_sums.free_home_import_kwh[world]
        frame["free_day_ahead_plan_kwh"] = dispatch_sums.free_day_ahead_home_import_kwh[world]
        frame["moved_kwh"] = frame["dispatched_kwh"] - frame["day_ahead_plan_kwh"]
        frame["replan_count"] = dispatch_sums.replan_count[world].astype(np.int64)
        frame["latest_close_gbp_per_mwh"] = close[world]
        frame["evidence_kind"] = TRADING_EVIDENCE
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).loc[:, DISPATCH_WORLD_SLOT_COLUMNS]


def dispatch_bands(dispatch_world_slot: pd.DataFrame, study_slots: pd.DataFrame) -> pd.DataFrame:
    """Intraday dispatch contract v1 §7.2: world-first P10/P50/P90 of the fleet's dispatch response.

    Built from ``dispatch_world_slot`` (§7.1), already world-major, so no
    re-sort of the model's own output is needed beyond the stable sort that
    guards against a caller's row order. ``series`` is ``day_ahead_plan``,
    ``dispatched`` or ``difference`` (dispatched minus day-ahead plan, taken
    per world *before* the quantile: the median of a difference is not the
    difference of medians, decision 0004 item 12), each with metric
    ``home_import_kw``; plus one ``dispatched`` / ``replan_count`` row.
    """

    keys = study_slots.loc[:, ["slot_index", "interval_start_utc", "interval_start_london"]]
    world_count = dispatch_world_slot["world_id"].nunique()

    def matrix(column: str) -> np.ndarray:
        rows = dispatch_world_slot.sort_values(["world_id", "slot_index"], kind="stable")
        return rows[column].to_numpy(dtype=float).reshape(world_count, -1)

    day_ahead_kw = matrix("day_ahead_plan_kwh") / 0.5
    dispatched_kw = matrix("dispatched_kwh") / 0.5
    series = [
        ("day_ahead_plan", "home_import_kw", "kW", day_ahead_kw),
        ("dispatched", "home_import_kw", "kW", dispatched_kw),
        ("difference", "home_import_kw", "kW", dispatched_kw - day_ahead_kw),
        ("dispatched", "replan_count", "EVs", matrix("replan_count")),
    ]
    frames = []
    for name, metric, unit, values in series:
        frame = keys.copy()
        frame.insert(0, "unit", unit)
        frame.insert(0, "metric", metric)
        frame.insert(0, "series", name)
        frame["world_count"] = np.int64(values.shape[0])
        frame = frame.assign(**_band_statistics(values))
        frame["evidence_kind"] = TRADING_EVIDENCE
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).loc[:, DISPATCH_BAND_COLUMNS]


def dispatch_split(
    cohort_catalogue: Sequence[tuple[str, str, float]],
    cohort_codes: np.ndarray,
    dispatch_locked: np.ndarray,
    *,
    commitment_share: float,
    replan_threshold_gbp_per_mwh: float,
) -> pd.DataFrame:
    """Intraday dispatch contract v1 §7.3: per-cohort locked/free EV counts (item 62(b)).

    One row per cohort in ``cohort_catalogue`` order (the six archetypes,
    including one with no sampled EVs, as ``cohort_summary`` does).
    ``dispatch_locked`` is ``units.dispatch_locked`` (§2): the counts here
    are read straight off it, so they equal its per-cohort sums by
    construction. The K = round-half-up(c x N), largest-remainder split
    across cohorts is ``action.locked_evs``'s rule (§2, K1's lane); this
    function does not recompute it, only reports what was assigned.
    """

    records = []
    for code, (cohort_id, _label, _share) in enumerate(cohort_catalogue):
        mask = cohort_codes == code
        ev_count = int(mask.sum())
        locked_count = int(dispatch_locked[mask].sum()) if ev_count else 0
        records.append(
            {
                "cohort_id": cohort_id,
                "ev_count": ev_count,
                "locked_count": locked_count,
                "free_count": ev_count - locked_count,
                "commitment_share": float(commitment_share),
                "replan_threshold_gbp_per_mwh": float(replan_threshold_gbp_per_mwh),
                "evidence_kind": TRADING_EVIDENCE,
            }
        )
    frame = pd.DataFrame(records, columns=DISPATCH_SPLIT_COLUMNS)
    return frame.astype(
        {
            "ev_count": np.int64,
            "locked_count": np.int64,
            "free_count": np.int64,
            "commitment_share": np.float64,
            "replan_threshold_gbp_per_mwh": np.float64,
        }
    )


def _scope_day_ahead_import_kw(scope: str, dispatch_sums) -> np.ndarray:
    """(world, study slot) day-ahead-plan import, kW, for one event scope (contract §7.4).

    National reads ``dispatch_sums.day_ahead_home_import_kwh``; a zone reads
    its slice of ``day_ahead_zone_home_import_kwh`` (§5.2), indexed the same
    way ``_scope_import_kw`` indexes ``zone_world`` (``ZONE_IDS`` order).
    """

    if scope == "national":
        kwh = dispatch_sums.day_ahead_home_import_kwh
    else:
        kwh = dispatch_sums.day_ahead_zone_home_import_kwh[:, ZONE_IDS.index(scope), :]
    return kwh / 0.5


def _scope_import_kw(
    scope: str, path_id: str, fleet_world: pd.DataFrame, zone_world: pd.DataFrame
) -> np.ndarray:
    if scope == "national":
        return _metric_matrix(fleet_world, path_id, "home_import_kw")
    return _zone_matrix(zone_world, scope, path_id, "home_import_kw")


def event_response_bands(
    events: pd.DataFrame,
    study_slots: pd.DataFrame,
    fleet_world: pd.DataFrame,
    zone_world: pd.DataFrame,
    *,
    net_demand_gw: np.ndarray,
    day_ahead_gbp_per_mwh: np.ndarray,
    imbalance_gbp_per_mwh: np.ndarray,
    curve_values: dict[str, float],
    scope_baseline_kwh: np.ndarray | None = None,
    dispatch_sums=None,
) -> pd.DataFrame:
    """Trading contract §5.6: the scope's response around each enabled event, P10-P90 across worlds.

    Rows run from ``EVENT_SLOTS_BEFORE`` slots before the window to
    ``EVENT_SLOTS_AFTER`` after it, clipped to the study.  Inputs: the
    validated events table, the study slots, the fleet and zone world
    frames (both paths), and (world, study slot) arrays of system net
    demand (GW, known shocks included), day-ahead and imbalance prices
    (£/MWh, after all shocks); ``curve_values`` are the supply-curve inputs.
    ``scope_baseline_kwh`` (world, scope, study slot; scope 0 fleet, 1-4
    zones) is the settlement baseline owned by ``market``: when given, the
    ``baseline_kw`` and ``delivered_kw`` rows are added; without it (before
    the trading overlay runs) they are left out.  ``dispatch_sums`` is the
    kernel's ``DispatchSums`` holder (intraday-dispatch-v1 §5.2); with it
    given (the switch on), the ``day_ahead_plan`` series is added (§7.4,
    changed by §11.7); without it (the switch off, or a no-action result)
    the series is left out, so this frame carries no extra column either way.

    Series (every per-world value is taken inside the world first):

    - ``normal`` and ``selected`` ``home_import_kw``: the scope's import;
    - ``day_ahead_plan`` ``home_import_kw`` (only with ``dispatch_sums``):
      the scope's import on the day-ahead plan reference path, added after
      ``selected`` so the event view can show day-ahead plan, dispatched
      and unmanaged together;
    - ``difference`` ``home_import_kw``: selected minus normal.  Inside a
      turn-down window it is the reduction; negative before or positive
      after it is charging moved beside the window (rebound);
    - ``market``: ``day_ahead_gbp_per_mwh`` and ``imbalance_gbp_per_mwh``,
      and for a scripted price shock ``price_increment_gbp_per_mwh``, that
      shock's own increment: ``f(ND) − f(ND − shock)`` for a known shock
      (its GW is already inside ND) and ``f(ND + S) − f(ND)`` for a surprise
      (it is not), on the convex supply curve ``f``.
    """

    keys = study_slots.loc[:, ["slot_index", "interval_start_utc", "interval_start_london"]]
    slot_count = len(study_slots)
    enabled = events.loc[events["enabled"].astype(bool)].reset_index(drop=True)
    windows = model_events.event_slots(enabled, study_slots)
    frames = []
    for row in windows.itertuples(index=False):
        first = max(0, row.start_slot - EVENT_SLOTS_BEFORE)
        last = min(slot_count, row.end_slot + EVENT_SLOTS_AFTER)
        shown = slice(first, last)

        series: list[tuple[str, str, str, np.ndarray]] = []
        imports = {
            path: _scope_import_kw(row.scope, path, fleet_world, zone_world) for path in PATH_ORDER
        }
        series.append(("normal", "home_import_kw", "kW", imports["normal"]))
        series.append(("selected", "home_import_kw", "kW", imports["selected"]))
        if dispatch_sums is not None:
            series.append(
                (
                    "day_ahead_plan",
                    "home_import_kw",
                    "kW",
                    _scope_day_ahead_import_kw(row.scope, dispatch_sums),
                )
            )
        if scope_baseline_kwh is not None and row.event_type in model_events.REQUEST_TYPES:
            scope = 0 if row.scope == "national" else 1 + ZONE_IDS.index(row.scope)
            baseline_kw = scope_baseline_kwh[:, scope, :] / 0.5
            gap = baseline_kw - imports["selected"]
            series.append(("selected", "baseline_kw", "kW", baseline_kw))
            series.append(
                ("selected", "delivered_kw", "kW", gap if row.event_type == "turn_down" else -gap)
            )
        series.append(
            ("difference", "home_import_kw", "kW", imports["selected"] - imports["normal"])
        )
        if row.event_type in model_events.PRICE_SHOCK_TYPES:
            one = enabled.loc[enabled["event_id"].eq(row.event_id)]
            known, surprise = model_events.scripted_shock_profiles(one, study_slots, slot_count)
            if row.event_type == "price_shock_known":
                increment = supply_curve_gbp_per_mwh(
                    net_demand_gw, **curve_values
                ) - supply_curve_gbp_per_mwh(net_demand_gw - known[None, :], **curve_values)
            else:
                increment = supply_curve_gbp_per_mwh(
                    net_demand_gw + surprise[None, :], **curve_values
                ) - supply_curve_gbp_per_mwh(net_demand_gw, **curve_values)
            series.append(("market", "price_increment_gbp_per_mwh", "GBP/MWh", increment))
        series.append(("market", "day_ahead_gbp_per_mwh", "GBP/MWh", day_ahead_gbp_per_mwh))
        series.append(("market", "imbalance_gbp_per_mwh", "GBP/MWh", imbalance_gbp_per_mwh))

        for name, metric, unit, values in series:
            frame = keys.iloc[shown].reset_index(drop=True)
            frame.insert(0, "unit", unit)
            frame.insert(0, "metric", metric)
            frame.insert(0, "series", name)
            frame.insert(0, "scope", row.scope)
            frame.insert(0, "event_type", row.event_type)
            frame.insert(0, "event_id", row.event_id)
            frame["relative_slot"] = np.arange(first, last, dtype=np.int64) - row.start_slot
            frame["in_window"] = (frame["relative_slot"] >= 0) & (
                frame["relative_slot"] < row.end_slot - row.start_slot
            )
            frame["world_count"] = np.int64(values.shape[0])
            frames.append(frame.assign(**_band_statistics(values[:, shown])))
    if not frames:
        empty = pd.DataFrame(columns=EVENT_RESPONSE_COLUMNS)
        return empty.astype(
            {
                "slot_index": np.int64,
                "interval_start_utc": "datetime64[ns, UTC]",
                "interval_start_london": f"datetime64[ns, {LONDON}]",
                "relative_slot": np.int64,
                "in_window": bool,
                "world_count": np.int64,
                **dict.fromkeys(("mean", "p10", "p50", "p90"), np.float64),
            }
        )
    return pd.concat(frames, ignore_index=True).loc[:, EVENT_RESPONSE_COLUMNS]


def _shock_peak_increment(
    world: np.ndarray,
    run_start: np.ndarray,
    duration: np.ndarray,
    signed_gw: np.ndarray,
    known: np.ndarray,
    net_demand_gw: np.ndarray,
    curve_values: dict[str, float],
) -> np.ndarray:
    """Largest price increment of each shock alone over its slots (£/MWh, ≥ 0).

    A known shock is already inside net demand ND, so its own increment is
    ``|f(ND) − f(ND − gw)|``; a surprise is not, so it is ``|f(ND + gw) − f(ND)|``
    (trading contract §5.6a).  ``run_start`` counts from the first priced
    (warm-up) slot; a shock running past the priced span is cut there.
    """

    peak = np.zeros(len(world))
    span = net_demand_gw.shape[1]
    for k in range(int(duration.max(initial=0))):
        slot = run_start + k
        on = (k < duration) & (slot < span)
        if not on.any():
            continue
        gw = signed_gw[on] * trapezoid_weight(np.full(on.sum(), k), duration[on])
        nd = net_demand_gw[world[on], slot[on]]
        increment = np.where(
            known[on],
            supply_curve_gbp_per_mwh(nd, **curve_values)
            - supply_curve_gbp_per_mwh(nd - gw, **curve_values),
            supply_curve_gbp_per_mwh(nd + gw, **curve_values)
            - supply_curve_gbp_per_mwh(nd, **curve_values),
        )
        peak[on] = np.maximum(peak[on], np.abs(increment))
    return peak


def market_shocks(
    stochastic: pd.DataFrame,
    events: pd.DataFrame,
    study_slots: pd.DataFrame,
    *,
    world_count: int,
    warmup_slot_count: int,
    net_demand_gw: np.ndarray,
    curve_values: dict[str, float],
) -> pd.DataFrame:
    """Trading contract §5.6a ``market_shocks``: stochastic and scripted shocks, one row each.

    ``stochastic`` is the generator's shock table (``MarketPrices.shocks``,
    start slots counted from the first warm-up slot); ``events`` the
    validated events table; ``net_demand_gw`` (world, warm-up + study slot)
    the day-ahead frame's system net demand.  Start slots are study slot
    indices (warm-up negative); ``in_study`` marks a shock overlapping a
    study slot.  Stochastic ids are ``s<world>-<n>`` (n counts that world's
    shocks from 0 in start order); scripted ones are the ``event_id`` and
    appear in every world.  Ordered by world then start.
    """

    slot_count = len(study_slots)
    shocks = stochastic.reset_index(drop=True)
    rank = shocks.groupby("world_id").cumcount().to_numpy()
    stochastic_rows = pd.DataFrame(
        {
            "world_id": shocks["world_id"].to_numpy(dtype=np.int64),
            "shock_id": [f"s{w}-{n}" for w, n in zip(shocks["world_id"], rank, strict=True)],
            "source": "stochastic",
            "shock_class": shocks["shock_class"].astype(object),
            "direction": shocks["direction"].astype(object),
            "known_day_ahead": shocks["known_day_ahead"].to_numpy(dtype=bool),
            "start_slot_index": shocks["start_slot_index"].to_numpy(dtype=np.int64)
            - warmup_slot_count,
            "start_utc": shocks["start_utc"],
            "duration_slots": shocks["duration_slots"].to_numpy(dtype=np.int64),
            "size_gw": shocks["size_gw"].to_numpy(dtype=float),
        }
    )
    scripted_rows = model_events.scripted_shock_rows(events, study_slots, world_count)
    frame = pd.concat([stochastic_rows, scripted_rows], ignore_index=True)
    signed = np.where(frame["direction"].eq("up"), 1.0, -1.0) * frame["size_gw"].to_numpy()
    start = frame["start_slot_index"].to_numpy(dtype=np.int64)
    duration = frame["duration_slots"].to_numpy(dtype=np.int64)
    frame["peak_price_increment_gbp_per_mwh"] = _shock_peak_increment(
        frame["world_id"].to_numpy(dtype=np.int64),
        start + warmup_slot_count,
        duration,
        signed,
        frame["known_day_ahead"].to_numpy(dtype=bool),
        net_demand_gw,
        curve_values,
    )
    frame["in_study"] = (start < slot_count) & (start + duration > 0)
    frame["start_utc"] = pd.to_datetime(frame["start_utc"], utc=True).dt.as_unit("ns")
    frame["start_london"] = frame["start_utc"].dt.tz_convert(LONDON)
    frame["evidence_kind"] = "synthetic"
    frame = frame.sort_values(["world_id", "start_slot_index"], kind="stable")
    return frame.loc[:, MARKET_SHOCK_COLUMNS].reset_index(drop=True)


def shock_summary(shocks: pd.DataFrame, world_count: int) -> pd.DataFrame:
    """Trading contract §5.6a ``shock_summary``: shocks per week and typical size by group.

    One row per (``shock_class``, ``known_day_ahead``, ``direction``) with
    at least one study shock, in the order mild, big, scripted; known first;
    up first.  Per world, the count of that group's study shocks (0 in a
    world with none), then linear P10/P50/P90 across worlds; the median
    size and median own peak price increment are across all the group's
    study shocks.
    """

    study = shocks.loc[shocks["in_study"]]
    rows = []
    for shock_class in SHOCK_CLASS_ORDER:
        for known in (True, False):
            for direction in ("up", "down"):
                group = study.loc[
                    study["shock_class"].eq(shock_class)
                    & study["known_day_ahead"].eq(known)
                    & study["direction"].eq(direction)
                ]
                if group.empty:
                    continue
                counts = np.bincount(
                    group["world_id"].to_numpy(dtype=np.int64), minlength=world_count
                ).astype(float)
                q = np.quantile(counts, _EV_QUANTILES, method="linear")
                rows.append(
                    {
                        "shock_class": shock_class,
                        "known_day_ahead": known,
                        "direction": direction,
                        "world_count": world_count,
                        "count_per_week_p10": q[0],
                        "count_per_week_p50": q[1],
                        "count_per_week_p90": q[2],
                        "size_gw_p50": float(np.median(group["size_gw"])),
                        "peak_price_increment_gbp_per_mwh_p50": float(
                            np.median(group["peak_price_increment_gbp_per_mwh"])
                        ),
                    }
                )
    frame = pd.DataFrame(rows, columns=SHOCK_SUMMARY_COLUMNS)
    return frame.astype(
        {
            "known_day_ahead": bool,
            "world_count": np.int64,
            **{column: np.float64 for column in SHOCK_SUMMARY_COLUMNS[4:]},
        }
    )


def build_summaries(
    state: ReplayState,
    fleet_world_intervals: pd.DataFrame,
    *,
    model: str,
    study_slots: pd.DataFrame,
    representative_world_id: int,
    cohort_catalogue: Sequence[tuple[str, str, float]],
    cost_effect: pd.DataFrame | None = None,
    timed_cost_effect: pd.DataFrame | None = None,
    forecast_prices: pd.DataFrame | None = None,
    evaluation_prices: pd.DataFrame | None = None,
    departure_margin_hours: float | None = None,
    trading: market.TradingRun | None = None,
    trading_assumptions: Mapping[str, float] | None = None,
    price_curve_source: str | None = None,
    public_charge_gbp_per_kwh: float | None = None,
    zone_world_intervals: pd.DataFrame | None = None,
    zone_shares: Sequence[float] | None = None,
    zone_headroom_kw: Sequence[float] | None = None,
    events: pd.DataFrame | None = None,
    market_prices: MarketPrices | None = None,
    price_assumptions: Mapping[str, object] | None = None,
    warmup_slot_count: int = 0,
    availability_inputs: availability.AvailabilityInputs | None = None,
    supplier_inputs: Mapping[str, float] | None = None,
    manufacturer_catalogue: Sequence[tuple[str, str, float]] | None = None,
    dispatch_sums=None,
) -> dict[str, object]:
    """Build every M5 summary field of ``ForecastResult`` after a run.

    Inputs: the run's ``ReplayState``; ``fleet_world_intervals`` in contract
    3.4 shape (normal only for ``model="no_action"``; normal and selected for
    ``"action"``; either way, plus the optional timed path when the run has
    it, decision 0007 model step 2); the result's ``study_slots`` (3.1) and
    ``representative_world_id``; ``cohort_catalogue`` as ``(cohort_id, label,
    source_population_share)`` for all six archetypes in source order.  Action
    runs also pass ``cost_effect`` (4.8) and the contract price frames
    (4.7).  ``departure_margin_hours`` (h) is the smart charger's margin;
    every run needs it for the flexibility metrics (``flexibility_bands``,
    ``deferrable_power_bands`` and ``flexibility_weekly_summary`` are
    ``None`` without it), and action runs also report it.

    Returns the M5 fields of ``ForecastResult`` as a dict keyed by field name.  Action-only fields
    are ``None`` for a no-action run (contract rule 1).  Action runs also
    return ``cost_effect`` again, with ``sessions_affected_count`` added.
    ``sampled_world_ids`` names the worlds kept in ``plug_in_events``.

    Action runs with the trading overlay also pass ``trading``
    (``market.run_trading``'s result), ``trading_assumptions``,
    ``price_curve_source`` (``"synthetic"`` or ``"user curve"``) and
    ``public_charge_gbp_per_kwh`` (the illustrative rate, for each EV's
    share of the cost effect); the trading and Supplier frames (trading
    contract v1 §5.5, §5.6a, §9.1, §9.3, §9.5) are then filled.

    Zones and events (trading contract v1 §5.6-§5.8): with
    ``zone_world_intervals`` (and the run's ``zone_shares`` and
    ``zone_headroom_kw`` in ``ZONE_IDS`` order), every result gets
    ``zone_import_bands`` and ``zone_summary``.  Action runs also pass the
    validated ``events`` table, the generator's ``market_prices`` (warm-up
    and study), the ``price_assumptions`` (for the supply curve) and
    ``warmup_slot_count``, and get ``event_response_bands``,
    ``market_shocks`` and ``shock_summary``.

    Firm-MW availability (trading contract v1 §10.2, lane J2): action runs
    given ``availability_inputs`` also get the ``availability.build_frames``
    fields, reduced from the selected path's per-EV arrays in the chunk loop.
    With the trading overlay and ``manufacturer_catalogue`` (``(manufacturer_id,
    label, share)`` per maker, lane J5) they also get the ``product.build_frames``
    fields (§10.5).  ``supplier_positions`` always has the §10.5f columns; its
    1-h firm figures are filled when the availability frames exist.
    Trading runs given ``supplier_inputs`` (``assumptions.supplier_inputs``)
    also get the ``supplier.FRAME_FIELDS`` frames (supplier contract v1).

    Intraday dispatch (intraday-dispatch-v1 contract, lane K3): with
    ``dispatch_sums`` (the kernel's ``DispatchSums`` holder, §5.2; ``None``
    with the switch off or on a no-action result, which is what keeps
    ``dispatch_world_slot``, ``dispatch_bands`` and ``dispatch_split``
    ``None`` too, contract §7) and ``trading_assumptions`` carrying the two
    new ``TRADING`` records, action runs also get those three frames and
    ``event_response_bands`` gains the ``day_ahead_plan`` series (§7.4).
    """

    settings = state.settings
    paths = list(PATH_ORDER) if model == "action" else ["normal"]
    # The optional timed path (decision 0007, model step 2): present or not
    # as a whole in the fleet frame already packaged (``forecast._package``),
    # so this is the one place that checks for it rather than threading a
    # separate flag through every caller.  ``chunk_paths`` is ``paths`` plus
    # "timed" for the one per-EV pass that needs it (charge completion,
    # session distribution, household); ``paths`` itself never gains it, so
    # every strictly pairwise consumer below (fleet-week EV bands,
    # flexibility, average-day bands, battery cycles) stays normal/selected
    # only, unaffected by the extra pass.
    timed_active = "timed" in set(fleet_world_intervals["path_id"])
    chunk_paths = [*paths, "timed"] if timed_active else paths
    representative = representative_world_id
    world_count = settings.evaluation_world_count
    slot_count = len(study_slots)

    units = state.units
    catalogue_ids = [cohort_id for cohort_id, _, _ in cohort_catalogue]
    cohort_codes = units["cohort_id"].map(catalogue_ids.index).to_numpy(dtype=np.int64)
    unit_ids = units["unit_id"].to_numpy(dtype=object)
    cohort_ids = units["cohort_id"].to_numpy(dtype=object)
    capacity = units["physical_capacity_kwh"].to_numpy(dtype=float)
    groups = {"fleet": np.ones(len(units), dtype=bool)}
    for code, cohort_id in enumerate(catalogue_ids):
        if (cohort_codes == code).any():
            groups[cohort_id] = cohort_codes == code

    # Representative world first, then the lowest other ids (lead decision).
    others = [w for w in range(world_count) if w != representative]
    sample_worlds = [representative, *others][:PLUG_IN_EVENT_WORLD_LIMIT]

    group_days = {
        (group_id, path): _new_group_day(world_count, slot_count)
        for group_id in groups
        for path in paths
    }
    ev_bands = new_ev_bands(paths, world_count, slot_count)
    # Flexibility needs the expected departure, so a run given no departure
    # margin (a direct no-action call) has none (contract 3.6c).
    # The stored inputs were built from this run's margin (forecast._package);
    # without a margin there are none, whatever the state carries.
    flex = None
    if departure_margin_hours is not None:
        flex = state.flexibility
        if flex is None:
            flex = flexibility_inputs(settings, units, study_slots, departure_margin_hours)
    flex_worlds = {
        (path, metric): np.full(
            (3, world_count, slot_count)
            if metric == "time_slack_hours"
            else (world_count, slot_count),
            np.nan,
        )
        for path in paths
        for metric, _, _ in FLEXIBILITY_ROWS
    }
    deferrable_worlds = np.full((len(SLACK_BUCKETS), world_count, slot_count), np.nan)
    statistics_rows, sample_events, session_rows = [], [], []
    target_kwh = capacity * units["preferred_target_soc_fraction"].to_numpy(dtype=float)
    power_kw = units["home_charger_limit_kw"].to_numpy(dtype=float)
    plug_out_worlds, plug_out_slots = [], []
    weekly_unserved = np.zeros((world_count, len(units)))
    sessions_affected = np.zeros(world_count, dtype=np.int64)
    # The timed path's own version of the same "how often did the alternative
    # policy leave the EV short of the normal path" count (decision 0007,
    # model step 2), read the same way, for timed_cost_effect's
    # sessions_affected_count; stays unused (all 0) when timed is inactive.
    timed_sessions_affected = np.zeros(world_count, dtype=np.int64)
    with_trading = model == "action" and trading is not None and flex is not None
    if with_trading:
        day_ahead_study = (
            forecast_prices.sort_values(["world_id", "slot_index"], kind="stable")[
                "wholesale_forecast_gbp_per_mwh"
            ]
            .to_numpy(dtype=float)
            .reshape(world_count, slot_count)
        )
        night = study_slots["night_index"].to_numpy()
        night_starts = np.flatnonzero(np.diff(night, prepend=-1))
        night_end = np.append(night_starts[1:], slot_count)[night]
        risk_charge = float(trading_assumptions["supplier.early_departure_risk_charge_gbp_per_mwh"])
        cost_curve_worlds = np.zeros((world_count, slot_count, len(COST_CURVE_THRESHOLDS) + 2))
        household_worlds: dict[str, np.ndarray] = {}
        battery_added = {path: np.zeros((world_count, len(units))) for path in paths}
        # Dispatch success (supplier contract §4.2): per (world, EV) sessions
        # and followed sessions from the kernel's plan_status (§10.1e).
        dispatch_sessions = (
            np.zeros((world_count, len(units))),
            np.zeros((world_count, len(units))),
        )
        efficiencies = (settings.home_charge_efficiency, state.public_top_up.efficiency_fraction)
        # Household contract v1 §4: per-(world, EV) weekly outcomes, reduced
        # in this loop; each world's own cheapest price third, formed once.
        household_totals = household.new_totals(
            world_count,
            study_slots,
            capacity_kwh=capacity,
            target_kwh=target_kwh,
            paths=(*household.PATHS, household.TIMED_PATH) if timed_active else household.PATHS,
        )
        low_band = price_third_bands(day_ahead_study) == "low"
        # Per-EV CO2 (household HH1b) uses the supplier's own intensity rule
        # (S§3.8), so it splits carbon_shift_world exactly; none without it.
        intensity = (
            None
            if supplier_inputs is None
            else supplier.carbon_intensity_gco2_per_kwh(
                forecast_prices.sort_values(["world_id", "slot_index"], kind="stable")[
                    "system_net_demand_gw"
                ]
                .to_numpy(dtype=float)
                .reshape(world_count, slot_count),
                supplier_inputs,
                float(price_assumptions["supply_reference_net_demand_gw"]),
            )
        )
    availability_totals = (
        availability.new_totals(availability_inputs, study_slots, flex, world_count)
        if model == "action" and availability_inputs is not None and flex is not None
        else None
    )
    product_totals = (
        product.new_totals(
            availability_totals,
            representative_world_id=representative,
            groups=groups,
            path_ids=chunk_paths,
        )
        if availability_totals is not None and with_trading and manufacturer_catalogue is not None
        else None
    )
    for worlds, by_path in _per_ev_chunks(state, study_slots, fleet_world_intervals, chunk_paths):
        if availability_totals is not None:
            availability.accumulate(availability_totals, worlds, by_path["selected"])
        # The optional timed pass (when ``chunk_paths`` carries it) stays out
        # of every strictly pairwise accumulator below (EV bands, flexibility,
        # average-day bands): they only ever look up ``paths``, not
        # ``by_path`` as a whole, so an extra "timed" entry in ``by_path``
        # changes nothing here.
        for path in paths:
            unit = by_path[path]
            soc = 100.0 * unit["closing_battery_kwh"] / capacity
            fill_ev_bands(ev_bands, path, worlds, {**unit, "soc_percent": soc})
            if flex is not None:
                chunk = flexibility_per_world(
                    unit["connected"],
                    unit["opening_battery_kwh"],
                    unit["home_grid_import_kwh"],
                    flex,
                )
                for metric, values in chunk.items():
                    flex_worlds[(path, metric)][..., worlds, :] = values
                if path == "normal":
                    deferrable_worlds[:, worlds, :] = deferrable_power_per_world(
                        unit["connected"], unit["opening_battery_kwh"], flex
                    )
            for group_id, mask in groups.items():
                _fill_group_day(
                    group_days[(group_id, path)],
                    worlds,
                    unit["connected"][..., mask],
                    soc[..., mask],
                )
        normal = by_path["normal"]
        if with_trading:
            # Supplier frames (§9.3): the cost curve reads the unmanaged
            # path's per-EV charging; the household and segment frames read
            # both paths' per-EV sums.
            cost_curve_worlds[worlds] = flex_cost_curve_chunk(
                normal["home_grid_import_kwh"],
                normal["connected"],
                trading.visible_day_ahead_gbp_per_mwh[worlds],
                flex.hours_to_departure,
                night_end,
                risk_charge,
            )
            for name, values in household_chunk(
                by_path, day_ahead_study[worlds], trading.settled[worlds], night_starts
            ).items():
                if name not in household_worlds:
                    household_worlds[name] = np.zeros((world_count, *values.shape[1:]))
                household_worlds[name][worlds] = values
            for path in paths:  # equivalent full cycles (supplier contract §4.1)
                battery_added[path][worlds] = supplier.battery_added_kwh(
                    by_path[path], *efficiencies
                )
        weekly_unserved[worlds] = normal["unserved_travel_battery_kwh"].sum(axis=1)
        if "selected" in by_path:
            sessions_affected[worlds] = sessions_affected_per_world(
                normal["connected"],
                normal["closing_battery_kwh"],
                by_path["selected"]["closing_battery_kwh"],
            )
        if "timed" in by_path:
            timed_sessions_affected[worlds] = sessions_affected_per_world(
                normal["connected"],
                normal["closing_battery_kwh"],
                by_path["timed"]["closing_battery_kwh"],
            )
        # Plug-outs (contract 3.10b): connected in one slot and not the next,
        # so the EV unplugs at the next slot's start.  Unlike plug-ins this
        # keeps sessions already running at the horizon start: their unplug
        # (the first morning's departures) is observed even though their
        # plug-in is not, and leaving them out would empty that morning.
        world_index, slot_before, _ = np.nonzero(
            normal["connected"][:, :-1, :] & ~normal["connected"][:, 1:, :]
        )
        plug_out_worlds.append(worlds[world_index])
        plug_out_slots.append(slot_before + 1)
        # Plug-in sessions are normal path only: the action does not change
        # when drivers plug in (contract 3.9).
        sessions = find_sessions(
            normal["connected"], normal["closing_battery_kwh"], normal["home_grid_import_kwh"]
        )
        selected_sessions = None
        if "selected" in by_path and (product_totals is not None or with_trading):
            selected = by_path["selected"]
            selected_sessions = find_sessions(
                selected["connected"],
                selected["closing_battery_kwh"],
                selected["home_grid_import_kwh"],
            )
        if product_totals is not None:
            sessions_by_path = {"normal": sessions, "selected": selected_sessions}
            if "timed" in by_path:
                # Charge completion (§10.5e) reads the timed path's own
                # session boundaries too, the same plug-in/unplug sessions
                # (decision 0004 item 32), so completion is just a different
                # stock at the same unplug (model step 2).
                timed_unit = by_path["timed"]
                sessions_by_path["timed"] = find_sessions(
                    timed_unit["connected"],
                    timed_unit["closing_battery_kwh"],
                    timed_unit["home_grid_import_kwh"],
                )
            product.accumulate(product_totals, worlds, by_path, sessions_by_path)
        if with_trading and selected_sessions is not None:
            counts = supplier.dispatch_session_counts(
                by_path["selected"]["plan_status"],
                selected_sessions.world,
                selected_sessions.ev,
                selected_sessions.first_slot,
                selected_sessions.end_slot,
            )
            for total, chunk in zip(dispatch_sessions, counts, strict=True):
                total[worlds] = chunk
        if with_trading:
            household.accumulate(
                household_totals,
                worlds,
                by_path,
                sessions,
                day_ahead=day_ahead_study[worlds],
                low_band=low_band[worlds],
                intensity=None if intensity is None else intensity[worlds],
            )
        statistics_rows.append(
            _session_statistics_rows(
                sessions,
                study_slots=study_slots,
                world_ids=worlds,
                cohort_codes=cohort_codes,
                capacity_kwh=capacity,
            )
        )
        session_rows.append(
            session_distribution_rows(
                sessions,
                study_slots=study_slots,
                world_ids=worlds,
                cohort_codes=cohort_codes,
                capacity_kwh=capacity,
                target_kwh=target_kwh,
                power_kw=power_kw,
                efficiency=settings.home_charge_efficiency,
                smart_closing_kwh=by_path["selected"]["closing_battery_kwh"]
                if "selected" in by_path
                else None,
                timed_closing_kwh=by_path["timed"]["closing_battery_kwh"]
                if "timed" in by_path
                else None,
            )
        )
        # Every chunk adds a (possibly empty) frame, so the result always has
        # the contract columns and dtypes even with no sampled plug-ins.
        in_sample = np.isin(worlds[sessions.world], sample_worlds)
        sample_events.append(
            plug_in_events_frame(
                _select_sessions(sessions, in_sample),
                study_slots=study_slots,
                world_ids=worlds,
                unit_ids=unit_ids,
                cohort_ids=cohort_ids,
                capacity_kwh=capacity,
                home_charge_efficiency=settings.home_charge_efficiency,
                path_id="normal",
            )
        )

    rows = pd.concat(statistics_rows, ignore_index=True)
    plug_in_events = _order_sample(sample_events, sample_worlds)
    kpi_groups: dict[str, tuple[int | None, int]] = {"fleet": (None, len(units))}
    for group_id, mask in groups.items():
        if group_id != "fleet":
            kpi_groups[group_id] = (catalogue_ids.index(group_id), int(mask.sum()))
    world_kpis = plug_in_world_kpis(rows, world_count, kpi_groups)
    summaries: dict[str, object] = {
        "average_day_bands": average_day_bands(group_days, study_slots),
        "fleet_interval_ev_bands": fleet_interval_ev_bands(ev_bands, study_slots),
        "flexibility_bands": None if flex is None else flexibility_bands(flex_worlds, study_slots),
        "plug_in_events": plug_in_events,
        "sampled_world_ids": tuple(sample_worlds),
        "plug_in_world_kpis": world_kpis,
        "plug_in_summary": plug_in_summary(rows, world_kpis, world_count),
        "plug_in_heatmap": plug_in_heatmap(rows, world_count, len(units), study_slots),
        "plug_in_half_hour_heatmap": plug_event_heatmap(
            rows["world_id"].to_numpy(),
            rows["plug_in_slot"].to_numpy(),
            world_count,
            len(units),
            study_slots,
        ),
        "plug_out_heatmap": plug_event_heatmap(
            np.concatenate(plug_out_worlds),
            np.concatenate(plug_out_slots),
            world_count,
            len(units),
            study_slots,
        ),
        "deferrable_power_bands": None
        if flex is None
        else deferrable_power_bands(deferrable_worlds, study_slots),
        "flexibility_weekly_summary": None
        if flex is None
        else flexibility_weekly_summary(
            deferrable_worlds,
            flex_worlds[("normal", "movable_energy_kwh")],
            _metric_matrix(fleet_world_intervals, "normal", "connected_count"),
            study_slots,
            float(units["home_charger_limit_kw"].sum()),
        ),
        "cohort_summary": cohort_summary(
            cohort_catalogue, cohort_codes, world_kpis, weekly_unserved
        ),
        "session_distribution_bands": session_distribution_bands(
            pd.concat(session_rows, ignore_index=True),
            world_count,
            {group_id: code for group_id, (code, _) in kpi_groups.items()},
        ),
        "difference_bands": None,
        "difference_weekly": None,
        "difference_weekly_bands": None,
        "action_summary": None,
        "cost_effect_summary": None,
        "not_recovered_summary": None,
        "timed_cost_effect_summary": None,
        "timed_not_recovered_summary": None,
        "smart_charging_world": None,
        "smart_charging_summary": None,
        "price_band_shift": None,
        "forecast_price_bands": None,
        "weekly_peak_summary": None,
        "price_relative_bands": None,
        "cheapest_half_hour_summary": None,
        **dict.fromkeys(TRADING_SUMMARY_FIELDS),
        "zone_import_bands": None,
        "zone_summary": None,
        "event_response_bands": None,
        "market_shocks": None,
        "shock_summary": None,
    }
    if zone_world_intervals is not None:
        labels = {zone_id: ZONE_LABELS[zone_id] for zone_id in ZONE_IDS}
        summaries.update(
            zone_import_bands=zone_import_bands(
                zone_world_intervals, study_slots, labels, zone_headroom_kw
            ),
            zone_summary=zone_summary(zone_world_intervals, labels, zone_shares, zone_headroom_kw),
        )
    # Intraday dispatch (intraday-dispatch-v1 contract §7): ``dispatch_sums``
    # is only given by an action run with the switch on, so the three frames
    # stay out of the returned dict on a no-action result and with the
    # switch off (§7 header, §8), and ``ForecastResult`` keeps its ``None``
    # defaults, without a separate model check here.
    if dispatch_sums is not None:
        world_slot = dispatch_world_slot(
            dispatch_sums, fleet_world_intervals, study_slots, evaluation_prices
        )
        summaries.update(
            dispatch_world_slot=world_slot,
            dispatch_bands=dispatch_bands(world_slot, study_slots),
            dispatch_split=dispatch_split(
                cohort_catalogue,
                cohort_codes,
                units["dispatch_locked"].to_numpy(dtype=bool),
                commitment_share=float(trading_assumptions["trading.day_ahead_commitment_share"]),
                replan_threshold_gbp_per_mwh=float(
                    trading_assumptions["trading.replan_threshold_gbp_per_mwh"]
                ),
            ),
        )
    # Availability frames come before the trading frames: ``supplier_positions``
    # reads the 1-h firm figures from ``availability_bands`` (§10.5f).
    if availability_totals is not None:
        summaries.update(availability.build_frames(availability_totals, study_slots))
    if with_trading:
        summaries.update(
            _trading_summaries(
                trading,
                study_slots=study_slots,
                units=units,
                catalogue_ids=catalogue_ids,
                fleet_world_intervals=fleet_world_intervals,
                cost_effect=cost_effect,
                day_ahead=day_ahead_study,
                deferrable_worlds=deferrable_worlds,
                cost_curve_worlds=cost_curve_worlds,
                household_worlds=household_worlds,
                household_totals=household_totals,
                evening_ev_kw=None
                if availability_totals is None
                else availability_totals["evening_ev_kw"],
                risk_charge=risk_charge,
                price_curve_source=price_curve_source,
                public_charge_gbp_per_kwh=public_charge_gbp_per_kwh,
                dispatch_world_slot=summaries.get("dispatch_world_slot"),
                supplier_context=None
                if supplier_inputs is None
                else dict(
                    cost_effect=cost_effect,
                    units=units,
                    study_slots=study_slots,
                    cohort_ids=catalogue_ids,
                    battery_added=battery_added,
                    forecast_prices=forecast_prices,
                    price_assumptions=price_assumptions,
                    trading_assumptions=trading_assumptions,
                    supplier_inputs=supplier_inputs,
                    warmup_slot_count=warmup_slot_count,
                    dispatch_sessions=dispatch_sessions,
                ),
                availability_bands=summaries.get("availability_bands"),
            )
        )
    if model == "action":
        # Sessions affected needs per-EV sessions, so it joins the per-world
        # cost frame here rather than in ``action`` (contract 4.8).  Any
        # earlier copy is replaced, so rebuilding from a packaged result works.
        cost_effect = cost_effect.drop(columns="sessions_affected_count", errors="ignore")
        cost_effect.insert(
            cost_effect.columns.get_loc("evidence_kind"),
            "sessions_affected_count",
            sessions_affected[cost_effect["world_id"].to_numpy()],
        )
        if timed_cost_effect is not None:
            # The timed path's own sessions_affected_count (decision 0007,
            # model step 2), read the same way as the selected path's, so
            # timed_not_recovered_summary has the column not_recovered_summary
            # needs; this is the only metric "added after" cost_effect itself
            # (contract 4.8), unchanged from the ordinary cost effect's rule.
            timed_cost_effect = timed_cost_effect.drop(
                columns="sessions_affected_count", errors="ignore"
            )
            timed_cost_effect.insert(
                timed_cost_effect.columns.get_loc("evidence_kind"),
                "sessions_affected_count",
                timed_sessions_affected[timed_cost_effect["world_id"].to_numpy()],
            )
        weekly = difference_weekly(fleet_world_intervals)
        outcomes = smart_charging_world(fleet_world_intervals, forecast_prices)
        summaries.update(
            difference_bands=difference_bands(fleet_world_intervals),
            difference_weekly=weekly,
            difference_weekly_bands=difference_weekly_bands(weekly),
            action_summary=action_summary(
                departure_margin_hours=departure_margin_hours,
                forecast_available_at_utc=forecast_prices["forecast_available_at_utc"].iat[0],
                vehicle_count=settings.vehicle_count,
            ),
            cost_effect=cost_effect,
            cost_effect_summary=cost_effect_summary(cost_effect),
            not_recovered_summary=not_recovered_summary(cost_effect),
            # The optional timed path's own cost effect and summaries
            # (decision 0007, model step 2): ``timed_cost_effect`` replaces
            # the field _package set with the ``sessions_affected_count``
            # added above, exactly as ``cost_effect`` is replaced here; the
            # two summaries are the same two functions, "timed" in the
            # selected slot, so there is nothing new to define for them.
            timed_cost_effect=timed_cost_effect,
            timed_cost_effect_summary=None
            if timed_cost_effect is None
            else cost_effect_summary(timed_cost_effect),
            timed_not_recovered_summary=None
            if timed_cost_effect is None
            else not_recovered_summary(timed_cost_effect),
            smart_charging_world=outcomes,
            smart_charging_summary=smart_charging_summary(outcomes),
            price_band_shift=price_band_shift(fleet_world_intervals, forecast_prices, study_slots),
            forecast_price_bands=forecast_price_bands(forecast_prices),
            weekly_peak_summary=weekly_peak_summary(
                fleet_world_intervals, home_charging_power_kw=_home_charging_power_kw(units)
            ),
            price_relative_bands=price_relative_bands(forecast_prices),
            cheapest_half_hour_summary=cheapest_half_hour_summary(forecast_prices),
        )
    if model == "action" and market_prices is not None:
        curve_values = {name: float(price_assumptions[name]) for name in SUPPLY_CURVE_INPUTS}
        net_demand = (
            market_prices.day_ahead["system_net_demand_gw"]
            .to_numpy(dtype=float)
            .reshape(world_count, -1)
        )
        shocks = market_shocks(
            market_prices.shocks,
            events,
            study_slots,
            world_count=world_count,
            warmup_slot_count=warmup_slot_count,
            net_demand_gw=net_demand,
            curve_values=curve_values,
        )
        summaries.update(
            event_response_bands=event_response_bands(
                events,
                study_slots,
                fleet_world_intervals,
                zone_world_intervals,
                net_demand_gw=net_demand[:, warmup_slot_count:],
                day_ahead_gbp_per_mwh=_price_matrix(forecast_prices)[0],
                imbalance_gbp_per_mwh=evaluation_prices.sort_values(
                    ["world_id", "slot_index"], kind="stable"
                )["imbalance_price_gbp_per_mwh"]
                .to_numpy(dtype=float)
                .reshape(world_count, -1),
                curve_values=curve_values,
                # The settlement baseline per scope (fleet, zones) from the
                # trading overlay, for the baseline and delivered rows.
                scope_baseline_kwh=None if trading is None else trading.scope_baseline,
                # The day-ahead plan series (§7.4): only with the dispatch
                # switch on, so this stays the contract's normal shape
                # without it (no extra series, no extra column).
                dispatch_sums=dispatch_sums,
            ),
            market_shocks=shocks,
            shock_summary=shock_summary(shocks, world_count),
        )
    if product_totals is not None:
        control = (
            units["control_group"].to_numpy(dtype=bool)
            if "control_group" in units
            else np.zeros(len(units), dtype=bool)
        )
        reduction = household_worlds["reduction_kwh"][[representative]]
        summaries.update(
            product.build_frames(
                product_totals,
                frames=summaries,
                trading=trading,
                units=units,
                weights=allocation_weights(reduction, ~control)[0],
                manufacturer_catalogue=manufacturer_catalogue,
                seed=settings.seed,
                start_local_date=settings.start_local_date,
                day_ahead_decision_local_hour=float(
                    trading_assumptions["day_ahead_publication_local_hour"]
                ),
            )
        )
    return summaries


TRADING_SUMMARY_FIELDS = (
    "trading_ledger_summary",
    "trading_kpis",
    "trading_checks",
    "shock_attribution_summary",
    "open_position_profile",
    "flex_cost_curve",
    "shape_premium_summary",
    "household_value_distribution",
    "household_value_summary",
    "household_ev_world",
    "household_outcomes_summary",
    "revenue_by_segment",
    "supplier_positions",
    "control_group_summary",
)
"""``ForecastResult`` fields ``build_summaries`` fills from the trading overlay (action only)."""


def _trading_summaries(
    trading: market.TradingRun,
    *,
    study_slots: pd.DataFrame,
    units: pd.DataFrame,
    catalogue_ids: Sequence[str],
    fleet_world_intervals: pd.DataFrame,
    cost_effect: pd.DataFrame,
    day_ahead: np.ndarray,
    deferrable_worlds: np.ndarray,
    cost_curve_worlds: np.ndarray,
    household_worlds: Mapping[str, np.ndarray],
    household_totals: dict,
    evening_ev_kw: np.ndarray | None,
    risk_charge: float,
    price_curve_source: str,
    public_charge_gbp_per_kwh: float,
    availability_bands: pd.DataFrame | None = None,
    supplier_context: Mapping[str, object] | None = None,
    dispatch_world_slot: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | None]:
    """The trading and Supplier frames of one action run (``TRADING_SUMMARY_FIELDS``).

    ``supplier_positions`` always carries the §10.5f columns (one schema):
    the 1-h firm figures are NaN when the run has no ``availability_bands``.  With
    ``supplier_context`` the supplier contract v1 frames are added.
    """

    ledger, week = trading.trading_ledger_world, trading.trading_week_world
    unmanaged = _metric_matrix(fleet_world_intervals, "normal", "home_import_kwh")
    metered = _metric_matrix(fleet_world_intervals, "selected", "home_import_kwh")
    half_hour = study_slots["local_half_hour"].to_numpy()
    rows = dict(
        zip(DEFERRABLE_BUCKET_ORDER, deferrable_rows_per_world(deferrable_worlds), strict=True)
    )
    # The per-world 19:00 capacity that can wait at least 2 h (results-v2
    # 3.6e), the denominator of the per-MW-year margin (§9.10 Q11).
    deferrable_2h = rows["at_least_2h"][:, half_hour == WAIT_CHECK_HALF_HOUR].mean(axis=1)
    control = (
        units["control_group"].to_numpy(dtype=bool)
        if "control_group" in units
        else np.zeros(len(units), dtype=bool)
    )
    treated = ~control
    weights = allocation_weights(household_worlds["reduction_kwh"], treated)
    full = ledger.loc[ledger["strategy"].eq("full")].sort_values(["world_id", "night_index"])
    customer_share = (
        full["customer_revenue_share_gbp"].to_numpy(dtype=float).reshape(weights.shape[0], -1)
    )
    cohorts = units["cohort_id"].to_numpy(dtype=object)
    groups = {"fleet": treated}
    groups.update(
        {c: treated & (cohorts == c) for c in catalogue_ids if (treated & (cohorts == c)).any()}
    )
    segments = [("all", "all", treated)]
    segments += [("cohort", c, members) for c, members in groups.items() if c != "fleet"]
    zones = units["zone_id"].to_numpy(dtype=object)
    segments += [("zone", z, treated & (zones == z)) for z in ZONE_IDS]
    # Supplier contract §4.4: one segment per maker with treated EVs, once units carry makers.
    segments += [g for g in supplier.partner_groups(units, ()) if g[0] == "manufacturer"]
    # ``per_ev`` feeds the supplier partner frames (supplier contract §4.1)
    # and the household frames (household contract v1 §3.1).
    saving = household_saving_gbp(household_worlds, cost_effect, public_charge_gbp_per_kwh)
    distribution, household_summary, per_ev = household_frames(
        saving, weights, customer_share, groups, household_worlds["reduction_kwh"]
    )
    frames = {
        "trading_ledger_summary": trading_ledger_summary(ledger, week),
        "trading_kpis": trading_kpis(
            week,
            vehicle_count=len(units),
            deferrable_2h_kw_1900=deferrable_2h,
            day_ahead_spread=day_ahead_spread_per_world(day_ahead, study_slots),
            erosion=baseline_erosion_per_world(
                trading.settled, trading.settled_flexed_baseline, day_ahead, study_slots
            ),
            dispatch_moved_mwh=dispatch_moved_mwh_per_world(dispatch_world_slot, len(day_ahead)),
        ),
        "trading_checks": trading_checks(
            ledger,
            week,
            trading.deviation_world_slot,
            day_ahead_position=trading.day_ahead_position,
            full_position=trading.positions["full"],
            revised_slots=trading.revised_slots,
            commitment_share=trading.commitment_share,
            fallback_nights=trading.fallback_nights,
            missing_nights=trading.missing_nights,
            day_ahead_forecast=trading.day_ahead_forecast
            if trading.commitment_rule == "newsvendor"
            else None,
        ),
        "shock_attribution_summary": shock_attribution_summary(week),
        "open_position_profile": open_position_profile(
            trading.settled, trading.positions, study_slots
        ),
        "flex_cost_curve": flex_cost_curve(cost_curve_worlds, study_slots, risk_charge),
        "shape_premium_summary": shape_premium_summary(unmanaged, metered, day_ahead),
        "household_value_distribution": distribution,
        "household_value_summary": household_summary,
        **household.build_frames(
            household_totals,
            units=units,
            saving_gbp=saving,
            per_ev=per_ev,
            evening_ev_kw=evening_ev_kw,
            groups=groups,
            public_charge_gbp_per_kwh=public_charge_gbp_per_kwh,
        ),
        "revenue_by_segment": revenue_by_segment(ledger, weights, segments),
        "supplier_positions": product.supplier_position_additions(
            supplier_positions(
                study_slots,
                day_ahead_position=trading.day_ahead_position,
                full_position=trading.positions["full"],
                baseline=trading.baseline,
                metered=metered,
                settled=trading.settled,
                day_ahead=day_ahead,
                price_curve_source=price_curve_source,
            ),
            trading.deviation_world_slot,
            availability_bands,
        ),
        "control_group_summary": control_group_summary(trading, unmanaged, control, study_slots),
    }
    if supplier_context is not None:  # beside the ledger, never added to it (supplier §3.1)
        frames |= supplier.build_frames(trading, per_ev, weights, **supplier_context)
    return frames


def _home_charging_power_kw(units: pd.DataFrame) -> float:
    """The fleet's one home charging power (kW, decision 0004 item 34)."""

    power = units["home_charger_limit_kw"].unique()
    if len(power) != 1:
        # The coincidence factor's "plugged-in EVs x home charging power"
        # assumes one power; per-EV powers would need the per-EV sum instead.
        raise ValueError("coincidence factor needs one home charging power for every EV")
    return float(power[0])


def _select_sessions(sessions: _Sessions, keep: np.ndarray) -> _Sessions:
    return _Sessions(
        **{name: getattr(sessions, name)[keep] for name in _Sessions.__dataclass_fields__}
    )


def _order_sample(frames: list[pd.DataFrame], sample_worlds: list[int]) -> pd.DataFrame:
    """Sampled events with the representative world first, then by EV and time."""

    events = pd.concat(frames, ignore_index=True)
    events["_order"] = events["world_id"].map(sample_worlds.index)
    events = events.sort_values(["_order", "unit_id", "event_index"], kind="stable")
    return events.drop(columns="_order").reset_index(drop=True)


# --------------------------------------------------------------------------
# Run comparison (contract section 6)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunComparison:
    """Contract section 6: run B minus run A on each run's headline path.

    ``firm_mw`` holds the Firm-MW rows of both runs side by side
    (``product.firm_mw_comparison``), None unless both runs have them.
    """

    matched_futures: bool
    mismatch_reasons: tuple[str, ...]
    path_id_a: str
    path_id_b: str
    changed: pd.DataFrame
    illustrative_total_available: bool
    difference_bands: pd.DataFrame
    weekly_differences: pd.DataFrame
    kpis: pd.DataFrame
    firm_mw: pd.DataFrame | None = None


@dataclass(frozen=True)
class RunSummary:
    """The slim record run history keeps per run for Compare (decision 0004 item 35).

    A full result holds per-world frames for every view plus the replay
    inputs; history keeps only what ``compare_runs`` reads, so three kept runs
    stay small.  ``assumptions``, ``vehicle_count``, ``horizon_start_utc`` and
    ``study_slot_count`` are small scalars/records kept too, because Compare
    lists changed assumptions and checks that worlds can be paired.
    ``weekly_bands`` is carried as produced by the forecast (``None`` if the
    result has none); Compare does not read it.  ``product_sheet`` and
    ``firmness_by_manufacturer`` are the small Firm-MW frames Compare shows
    side by side (``None`` when the result has none).
    """

    model: str
    seed: int
    vehicle_count: int
    world_count: int
    horizon_start_utc: pd.Timestamp
    study_slot_count: int
    settings_snapshot: dict[str, object]
    assumptions: tuple
    fleet_world_intervals: pd.DataFrame
    plug_in_world_kpis: pd.DataFrame
    cost_effect: pd.DataFrame | None
    weekly_bands: pd.DataFrame | None
    product_sheet: pd.DataFrame | None
    firmness_by_manufacturer: pd.DataFrame | None


def slim_run(result: ForecastResult) -> RunSummary:
    """Keep only what Compare needs from a full result (for run history)."""

    return RunSummary(
        model=result.model,
        seed=result.seed,
        vehicle_count=result.vehicle_count,
        world_count=result.world_count,
        horizon_start_utc=result.horizon_start_utc,
        study_slot_count=result.study_slot_count,
        settings_snapshot=dict(result.settings_snapshot),
        assumptions=tuple(result.assumptions),
        fleet_world_intervals=result.fleet_world_intervals,
        plug_in_world_kpis=result.plug_in_world_kpis,
        cost_effect=result.cost_effect,
        weekly_bands=getattr(result, "weekly_bands", None),
        # Firm-MW frames (trading contract v1 §10.10 Compare); None on a
        # no-action result or before the availability lanes are wired.
        product_sheet=getattr(result, "product_sheet", None),
        firmness_by_manufacturer=getattr(result, "firmness_by_manufacturer", None),
    )


def compare_runs(a: RunSummary | ForecastResult, b: RunSummary | ForecastResult) -> RunComparison:
    """Compare two runs world by world, B minus A (decision 0004 item 6).

    ``a`` and ``b`` are ``RunSummary`` records or full results; only the
    ``RunSummary`` attributes are read (decision 0004 item 35).

    Worlds can only be paired when both runs cover the same horizon with the
    same number of worlds, so a difference raises ``ValueError``.  Otherwise
    the comparison is returned; ``matched_futures`` is False, with a reason,
    when the seed or fleet size differ, because then world ``w`` of A and B
    saw different random futures and a per-world difference mixes the change
    with sampling noise.  The two models draw their worlds in the same order
    (the action model's price noise comes last), so a different model alone
    still pairs the same futures.  The headline path is ``selected`` for an
    action run and ``normal`` for a no-action run.
    """

    for name in ("horizon_start_utc", "study_slot_count", "world_count"):
        if getattr(a, name) != getattr(b, name):
            raise ValueError(f"cannot pair worlds: {name} differs")
    reasons = []
    if a.seed != b.seed:
        reasons.append(f"Not matched: A uses seed {a.seed}, B uses seed {b.seed}.")
    if a.vehicle_count != b.vehicle_count:
        reasons.append(f"Not matched: A has {a.vehicle_count} EVs, B has {b.vehicle_count}.")
    path_a = "selected" if a.model == "action" else "normal"
    path_b = "selected" if b.model == "action" else "normal"
    world_a, world_b = a.fleet_world_intervals, b.fleet_world_intervals

    totals_available = a.model == "action" and b.model == "action"
    weekly = {
        "home_import_kwh": (
            "kWh per week",
            _metric_matrix(world_a, path_a, "home_import_kwh").sum(axis=1),
            _metric_matrix(world_b, path_b, "home_import_kwh").sum(axis=1),
        ),
        "unserved_travel_kwh": (
            "kWh per week",
            _metric_matrix(world_a, path_a, "unserved_travel_kwh").sum(axis=1),
            _metric_matrix(world_b, path_b, "unserved_travel_kwh").sum(axis=1),
        ),
        "median_plug_in_soc_percent": (
            "percentage points",
            _world_median_plug_in_soc(a),
            _world_median_plug_in_soc(b),
        ),
    }
    if totals_available:
        column = "illustrative_selected_minus_normal_total_gbp"
        weekly["illustrative_total_gbp"] = (
            "GBP",
            a.cost_effect.sort_values("world_id")[column].to_numpy(dtype=float),
            b.cost_effect.sort_values("world_id")[column].to_numpy(dtype=float),
        )
    weekly_frames, kpi_rows = [], []
    for metric, (unit, values_a, values_b) in weekly.items():
        delta = values_b - values_a
        weekly_frames.append(
            pd.DataFrame(
                {
                    "world_id": np.arange(len(delta), dtype=np.int64),
                    "metric": metric,
                    "unit": unit,
                    "value_a": values_a,
                    "value_b": values_b,
                    "b_minus_a": delta,
                }
            )
        )
        # A world with no plug-ins in either run has no median difference.
        q = _quantiles(delta[~np.isnan(delta)])
        kpi_rows.append({"metric": metric, "unit": unit, "p10": q[0], "p50": q[1], "p90": q[2]})
    return RunComparison(
        matched_futures=not reasons,
        mismatch_reasons=tuple(reasons),
        path_id_a=path_a,
        path_id_b=path_b,
        changed=_changed(a, b),
        illustrative_total_available=totals_available,
        difference_bands=paired_difference_bands(world_a, path_a, world_b, path_b),
        weekly_differences=pd.concat(weekly_frames, ignore_index=True),
        kpis=pd.DataFrame(kpi_rows),
        firm_mw=product.firm_mw_comparison(
            getattr(a, "product_sheet", None),
            getattr(a, "firmness_by_manufacturer", None),
            getattr(b, "product_sheet", None),
            getattr(b, "firmness_by_manufacturer", None),
        ),
    )


def _world_median_plug_in_soc(result) -> np.ndarray:
    # Fleet, all day types, every world (contract section 6); plug-ins are
    # normal path in both runs because the action does not move them.
    kpis = result.plug_in_world_kpis
    chosen = kpis.loc[
        kpis["group_id"].eq("fleet")
        & kpis["day_type"].eq("all")
        & kpis["metric"].eq("median_plug_in_soc_percent")
    ]
    return chosen.sort_values("world_id")["value"].to_numpy(dtype=float)


def _same_value(value_a, value_b) -> bool:
    """Equal, treating two missing floats as the same (a NaN record, such as an
    unset zone headroom, must not read as a change between runs)."""

    if isinstance(value_a, float) and isinstance(value_b, float):
        if math.isnan(value_a) and math.isnan(value_b):
            return True
    return value_a == value_b


def _changed(a, b) -> pd.DataFrame:
    """Assumptions (record order) then settings keys (snapshot order) that differ."""

    rows = []
    records_a = {record.name: record for record in a.assumptions}
    records_b = {record.name: record for record in b.assumptions}
    for name in dict.fromkeys([*records_a, *records_b]):
        record_a, record_b = records_a.get(name), records_b.get(name)
        value_a = record_a.value if record_a else None
        value_b = record_b.value if record_b else None
        if not _same_value(value_a, value_b):
            record = record_b or record_a
            rows.append((name, record.label, record.unit, value_a, value_b))
    for key in SETTINGS_KEYS:
        value_a, value_b = a.settings_snapshot.get(key), b.settings_snapshot.get(key)
        if not _same_value(value_a, value_b):
            rows.append((key, key, "", value_a, value_b))
    return pd.DataFrame(rows, columns=["name", "label", "unit", "value_a", "value_b"], dtype=object)


__all__ = [
    "PLUG_IN_EVENT_WORLD_LIMIT",
    "WORLD_CHUNK_SIZE",
    "ActionSummary",
    "PlugInSummary",
    "ReplayState",
    "SampledWorlds",
    "RunComparison",
    "RunSummary",
    "action_summary",
    "average_day_bands",
    "build_study_slots",
    "build_summaries",
    "cohort_summary",
    "compare_runs",
    "cost_effect_summary",
    "difference_bands",
    "difference_weekly",
    "difference_weekly_bands",
    "fill_ev_bands",
    "find_sessions",
    "fleet_interval_ev_bands",
    "forecast_price_bands",
    "kernel_slice",
    "new_ev_bands",
    "paired_difference_bands",
    "plug_in_events_frame",
    "plug_in_summary",
    "plug_in_world_kpis",
    "price_band_shift",
    "price_third_bands",
    "session_values",
    "slim_run",
    "smart_charging_summary",
    "smart_charging_world",
    "weekly_peak_summary",
]
