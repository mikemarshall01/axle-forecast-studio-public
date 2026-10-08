"""Household outcomes per EV and per group (household contract v1 §3.1-§3.2).

This module owns what one customer gets from smart charging, read per EV
from arrays the run already produces:

- ``new_totals`` / ``accumulate`` / ``build_frames`` are the chunk-loop hook
  in ``summaries.build_summaries`` (the J2 pattern): each chunk of worlds is
  reduced to weekly scalars per (world, EV) and nothing per slot survives
  the chunk (§5: per-EV x slot arrays for every EV are too big to keep).
- ``household_ev_world`` is one row per (world, EV) of weekly totals:
  energy, cost, the pass-through saving and value, sessions, readiness,
  CO2 and evening flexibility.
- ``household_outcomes_summary`` reduces that frame per group (fleet, then
  each cohort with treated EVs) in three readings: set A "in a typical
  week" (per world across EVs, then across worlds), set R the session- or
  energy-weighted group ratio, and set B across customers' average weeks
  (each EV's mean over worlds first).  Per-year and per-month claims read
  set B only, because one week's spread x 52 is not a yearly range (B1).
- ``metric_values`` forms every per-(world, EV) metric value once, so the
  summary, the card and the validator cannot define a ratio differently.
- ``household_card`` is one EV on demand (§3.3): its outcomes across weeks
  with cohort and fleet comparison and percentile rank (reads of the two
  frames), and its timing, reliability and availability profile from one
  cached one-EV kernel run over every world.

Nothing here samples, and no new money formula is introduced: the saving,
value and earning columns are the market lane's per-EV arrays stored bit
for bit (T§9.3c, S§4.1).  Every figure is illustrative and synthetic.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from axle_studio.model import availability, summaries
from axle_studio.model.action import NOT_RECOVERED_TOLERANCE_KWH

EVIDENCE_KIND = "illustrative_synthetic"
PATHS = ("normal", "selected")
TIMED_PATH = "timed"
"""Optional third path (decision 0007, model step 2): a caller passes ``(*PATHS, TIMED_PATH)``
to ``new_totals`` to add its own per-path columns and metrics beside normal and selected,
same naming pattern, no "timed vs normal" difference metric (that reading stays selected
vs normal, §3.2's own contract)."""
DIRECTIONS = ("turn_down", "turn_up")
_WEEKS_PER_YEAR = 52.0
_MONTHS_PER_YEAR = 12.0

_SUM_METRICS = ("home_import_kwh", "cheap_import_kwh", "home_cost_gbp", "public_import_kwh")
"""Per-path sums every path in ``paths`` gets, in ``_SUM_COLUMNS``' name pattern."""


def _sum_columns(paths: Sequence[str] = PATHS) -> tuple[str, ...]:
    return tuple(f"{name}_{path}" for name in _SUM_METRICS for path in paths)


def _count_columns(paths: Sequence[str] = PATHS) -> tuple[str, ...]:
    return (
        "session_count",
        "session_end_count",
        "nights_plugged_count",
        *(f"completed_count_{path}" for path in paths),
        "sessions_affected_count",
    )


_SUM_COLUMNS = _sum_columns(PATHS)
_COUNT_COLUMNS = _count_columns(PATHS)


def ev_world_columns(paths: Sequence[str] = PATHS) -> dict[str, type]:
    """``household_ev_world`` columns and dtypes, in order (§3.1), for the given paths.

    With the optional timed path (``paths = (*PATHS, TIMED_PATH)``), the frame
    gains ``home_import_kwh_timed`` and its siblings in the same pattern as
    ``_normal``/``_selected``, plus ``completed_count_timed`` and
    ``departure_soc_mean_timed``; every other column, and the base columns'
    own order, is unchanged, so an off run's frame is schema-identical to
    before this existed.
    """

    def per_path(name: str) -> dict[str, type]:
        return {f"{name}_{path}": np.float64 for path in paths}

    return {
        "world_id": np.int64,
        "unit_id": object,
        "cohort_id": object,
        "treated": bool,
        "earning": bool,
        **per_path("home_import_kwh"),
        **per_path("cheap_import_kwh"),
        **per_path("public_import_kwh"),
        **per_path("home_cost_gbp"),
        **per_path("public_cost_gbp"),
        "saving_gbp": np.float64,
        "value_gbp_per_month": np.float64,
        "supplier_energy_saving_gbp": np.float64,
        **dict.fromkeys(_count_columns(paths), np.int64),
        **per_path("departure_soc_mean"),
        "co2_shifted_kg": np.float64,
        "evening_turn_down_kw_1h": np.float64,
        "evening_turn_up_kw_1h": np.float64,
        "evidence_kind": object,
    }


EV_WORLD_COLUMNS = ev_world_columns(PATHS)
"""``household_ev_world`` columns and dtypes for the default two paths; see ``ev_world_columns``."""

_BASE_METRICS = (
    ("value_gbp_per_week", "GBP per week", "week_ahead", "higher"),
    ("value_gbp_per_month", "GBP per household per month", "scenario", "higher"),
    ("value_gbp_per_year", "GBP per year", "scenario", "higher"),
    ("saving_gbp_per_week", "GBP per week", "week_ahead", "higher"),
    ("saving_gbp_per_year", "GBP per year", "scenario", "higher"),
    ("supplier_energy_saving_gbp_per_week", "GBP per week", "week_ahead", "higher"),
    (
        "supplier_energy_saving_gbp_per_month",
        "GBP per customer per month",
        "scenario",
        "higher",
    ),
    ("home_cost_gbp_per_kwh_normal", "GBP per kWh", "week_ahead", "lower"),
    ("home_cost_gbp_per_kwh_selected", "GBP per kWh", "week_ahead", "lower"),
    ("home_cost_gbp_per_kwh_difference", "GBP per kWh", "week_ahead", "lower"),
    ("home_import_kwh_per_week_normal", "kWh per week", "week_ahead", ""),
    ("home_import_kwh_per_week_selected", "kWh per week", "week_ahead", ""),
    ("public_import_kwh_per_week_normal", "kWh per week", "week_ahead", "lower"),
    ("public_import_kwh_per_week_selected", "kWh per week", "week_ahead", "lower"),
    ("cheap_share_normal", "fraction", "week_ahead", "higher"),
    ("cheap_share_selected", "fraction", "week_ahead", "higher"),
    ("cheap_share_difference", "fraction", "week_ahead", "higher"),
    ("completed_share_normal", "fraction", "week_ahead", "higher"),
    ("completed_share_selected", "fraction", "week_ahead", "higher"),
    ("completed_share_difference", "fraction", "week_ahead", "higher"),
    ("sessions_per_week", "sessions per week", "week_ahead", ""),
    ("session_ends_per_week", "session ends per week", "week_ahead", ""),
    ("sessions_affected_count", "session ends per week", "week_ahead", "lower"),
    ("sessions_affected_share", "fraction", "week_ahead", "lower"),
    ("nights_plugged_share", "fraction", "week_ahead", ""),
    ("departure_soc_mean_normal", "percent", "week_ahead", "higher"),
    ("departure_soc_mean_selected", "percent", "week_ahead", "higher"),
    ("co2_shifted_kg_per_week", "kg CO2 per week", "week_ahead", "higher"),
    ("co2_shifted_kg_per_month", "kg CO2 per customer per month", "scenario", "higher"),
    ("evening_turn_down_kw_1h", "kW", "week_ahead", "higher"),
    ("evening_turn_up_kw_1h", "kW", "week_ahead", "higher"),
)
"""(metric, unit, horizon, better_is) of every §3.2 base (normal/selected) metric, summary order."""


def household_metrics(paths: Sequence[str] = PATHS) -> tuple[tuple[str, str, str, str], ...]:
    """§3.2 metrics for the given paths: the base set, plus each extra path's own metrics.

    The base normal/selected metrics and their paired differences are fixed
    (the household contract's base reading, always this exact order); a path
    beyond ``PATHS`` (the optional timed path, decision 0007, model step 2)
    adds its own cost-per-kWh, import, cheap-share, completed-share and
    departure-SoC metrics, appended after the base set so the base rows
    never move or renumber for a run without the timed path.  No "timed vs
    normal" difference metric: that reading stays §3.2's own selected vs
    normal.
    """

    metrics = list(_BASE_METRICS)
    for path in paths:
        if path in PATHS:
            continue
        metrics.extend(
            [
                (f"home_cost_gbp_per_kwh_{path}", "GBP per kWh", "week_ahead", "lower"),
                (f"home_import_kwh_per_week_{path}", "kWh per week", "week_ahead", ""),
                (f"public_import_kwh_per_week_{path}", "kWh per week", "week_ahead", "lower"),
                (f"cheap_share_{path}", "fraction", "week_ahead", "higher"),
                (f"completed_share_{path}", "fraction", "week_ahead", "higher"),
                (f"departure_soc_mean_{path}", "percent", "week_ahead", "higher"),
            ]
        )
    return tuple(metrics)


METRICS = household_metrics(PATHS)
"""§3.2 metrics for the default two paths; see ``household_metrics``."""

_BASE_RATIO_METRICS = {
    "home_cost_gbp_per_kwh_normal": ("home_cost_gbp_normal", "home_import_kwh_normal"),
    "home_cost_gbp_per_kwh_selected": ("home_cost_gbp_selected", "home_import_kwh_selected"),
    "cheap_share_normal": ("cheap_import_kwh_normal", "home_import_kwh_normal"),
    "cheap_share_selected": ("cheap_import_kwh_selected", "home_import_kwh_selected"),
    "completed_share_normal": ("completed_count_normal", "session_count"),
    "completed_share_selected": ("completed_count_selected", "session_count"),
    "sessions_affected_share": ("sessions_affected_count", "session_end_count"),
    "nights_plugged_share": ("nights_plugged_count", None),  # denominator: N_n per EV
}


def ratio_metrics(paths: Sequence[str] = PATHS) -> dict[str, tuple[str, str | None]]:
    """Set R (numerator column, denominator column) of each plain ratio metric (§3.2).

    See ``household_metrics``: a path beyond ``PATHS`` adds its own
    ``home_cost_gbp_per_kwh``, ``cheap_share`` and ``completed_share`` ratios.
    """

    ratios = dict(_BASE_RATIO_METRICS)
    for path in paths:
        if path in PATHS:
            continue
        ratios[f"home_cost_gbp_per_kwh_{path}"] = (
            f"home_cost_gbp_{path}",
            f"home_import_kwh_{path}",
        )
        ratios[f"cheap_share_{path}"] = (f"cheap_import_kwh_{path}", f"home_import_kwh_{path}")
        ratios[f"completed_share_{path}"] = (f"completed_count_{path}", "session_count")
    return ratios


RATIO_METRICS = ratio_metrics(PATHS)
"""Set R ratio metrics for the default two paths; see ``ratio_metrics``."""

DIFFERENCE_METRICS = {
    "home_cost_gbp_per_kwh_difference": (
        "home_cost_gbp_per_kwh_selected",
        "home_cost_gbp_per_kwh_normal",
    ),
    "cheap_share_difference": ("cheap_share_selected", "cheap_share_normal"),
    "completed_share_difference": ("completed_share_selected", "completed_share_normal"),
}
"""Paired differences (selected metric, normal metric), taken inside each (world, EV).

Stays selected versus normal only (§3.2, item 6): the optional timed path never
gets a difference metric here."""

SET_A = ("mean_across_evs", "p10_across_evs", "p50_across_evs", "p90_across_evs")
SET_R = ("group_ratio",)
SET_B = (
    "mean_of_ev_means",
    "p10_of_ev_means",
    "p50_of_ev_means",
    "p90_of_ev_means",
    "share_of_ev_means_below_zero",
)

SUMMARY_COLUMNS = {
    "group_id": object,
    "metric": object,
    "statistic": object,
    "unit": object,
    "horizon": object,
    "better_is": object,
    "ev_count": np.int64,
    "ev_value_count": np.int64,
    "world_count": np.int64,
    "mean": np.float64,
    "p10": np.float64,
    "p50": np.float64,
    "p90": np.float64,
    "evidence_kind": object,
}
"""``household_outcomes_summary`` columns and dtypes, in order (§3.2)."""


# --------------------------------------------------------------------------
# Chunk-loop hook (§4)
# --------------------------------------------------------------------------


def new_totals(
    world_count: int,
    study_slots: pd.DataFrame,
    *,
    capacity_kwh: np.ndarray,
    target_kwh: np.ndarray,
    paths: Sequence[str] = PATHS,
) -> dict:
    """Empty (world, EV) accumulators for one run, filled chunk by chunk by ``accumulate``.

    ``capacity_kwh`` and ``target_kwh`` are per EV in ``units`` order, kWh
    battery-side: physical capacity and the preferred target stock
    (capacity x preferred target fraction; the target, not the ceiling).
    At 1,000 EVs x 100 worlds each accumulator is 0.8 MB.  ``paths`` is
    ``PATHS`` by default; pass ``(*PATHS, TIMED_PATH)`` to also accumulate
    the optional timed path (decision 0007, model step 2); ``accumulate``
    and ``build_frames`` then read ``totals["paths"]`` rather than taking
    the list again.
    """

    vehicle_count = len(capacity_kwh)
    night = study_slots["night_index"].to_numpy(dtype=np.int64)
    shape = (world_count, vehicle_count)
    totals: dict = {name: np.zeros(shape) for name in _sum_columns(paths)}
    totals.update({name: np.zeros(shape, dtype=np.int64) for name in _count_columns(paths)})
    totals.update(
        {f"departure_soc_mean_{path}": np.full(shape, np.nan) for path in paths}
        | {"co2_shifted_kg": np.full(shape, np.nan)}
    )
    totals["paths"] = tuple(paths)
    totals["capacity_kwh"] = np.asarray(capacity_kwh, dtype=float)
    totals["target_kwh"] = np.asarray(target_kwh, dtype=float)
    # First slot of each session night.  A clock-change night has 46 or 50
    # slots, so nights are reduced with reduceat on these starts, never by a
    # reshape to fixed-length nights (B4).
    totals["night_starts"] = np.flatnonzero(np.diff(night, prepend=-1))
    totals["night_count"] = int(night.max()) + 1
    return totals


def accumulate(
    totals: dict,
    worlds: np.ndarray,
    by_path: Mapping[str, Mapping[str, np.ndarray]],
    sessions,
    *,
    day_ahead: np.ndarray,
    low_band: np.ndarray,
    intensity: np.ndarray | None,
) -> None:
    """Reduce one chunk of worlds to weekly (world, EV) scalars in ``totals``.

    ``worlds`` (chunk,) are the run's world ids; ``by_path`` holds every path
    named in ``totals["paths"]`` (``new_totals``) of per-EV kernel arrays
    (chunk world, study slot, EV) with the shared ``connected``; ``sessions``
    are the chunk's normal-path ``summaries.find_sessions`` result (reused,
    not found twice); ``day_ahead`` (chunk world, slot) the day-ahead price
    after shocks and the user curve, GBP/MWh; ``low_band`` (chunk world,
    slot) bool, the world's own cheapest price third
    (``summaries.price_third_bands``); ``intensity`` (chunk world, slot)
    gCO2/kWh or ``None`` before S1.

    CO2 shift, nights plugged and sessions affected stay normal versus
    selected only (item 6): plug-in sessions and CO2 shifted are not
    redefined for a third path.
    """

    paths = totals["paths"]
    normal, selected = by_path["normal"], by_path["selected"]
    cheap = low_band[:, :, np.newaxis]
    price = day_ahead[:, :, np.newaxis]
    for path in paths:
        unit = by_path[path]
        home = unit["home_grid_import_kwh"]
        totals[f"home_import_kwh_{path}"][worlds] = home.sum(axis=1)
        totals[f"cheap_import_kwh_{path}"][worlds] = (home * cheap).sum(axis=1)
        # The customer leg at the day-ahead price (pass-through reading, B3).
        # Both absolute costs are kept for cost per kWh per path; their
        # difference is household_chunk's home_cost_gbp (O7, validator).
        totals[f"home_cost_gbp_{path}"][worlds] = (home * price).sum(axis=1) / 1000.0
        totals[f"public_import_kwh_{path}"][worlds] = unit["public_grid_import_kwh"].sum(axis=1)
    if intensity is not None:
        shifted = normal["home_grid_import_kwh"] - selected["home_grid_import_kwh"]
        totals["co2_shifted_kg"][worlds] = (shifted * intensity[:, :, np.newaxis]).sum(
            axis=1
        ) / 1000.0

    connected = normal["connected"]  # the same on every path (decision 0004 item 32)
    totals["nights_plugged_count"][worlds] = np.logical_or.reduceat(
        connected, totals["night_starts"], axis=1
    ).sum(axis=1)
    # Session ends are sessions_affected_per_world's population: the last
    # connected slot of every run, including a run already plugged in at the
    # horizon start and one still plugged in at the end (B9).
    ends = connected.copy()
    ends[:, :-1, :] &= ~connected[:, 1:, :]
    short = (
        normal["closing_battery_kwh"] - selected["closing_battery_kwh"]
        > NOT_RECOVERED_TOLERANCE_KWH
    )
    totals["session_end_count"][worlds] = ends.sum(axis=1)
    totals["sessions_affected_count"][worlds] = (ends & short).sum(axis=1)

    # Closed sessions (observed plug-in and unplug inside the study) are the
    # population of readiness and departure SoC (T§10.5e's rule).
    chunk, slot_count, vehicle_count = connected.shape
    closed = sessions.end_slot < slot_count
    world, ev, last = sessions.world[closed], sessions.ev[closed], sessions.end_slot[closed] - 1
    flat = world * vehicle_count + ev
    size = chunk * vehicle_count

    def per_ev(weights=None):
        return np.bincount(flat, weights=weights, minlength=size).reshape(chunk, vehicle_count)

    count = per_ev()
    totals["session_count"][worlds] = count
    # "normal"'s stock is the sessions' own (the normal-path close at unplug,
    # found once); every other path reads its own per-EV closing stock at the
    # same session boundary (plug-in and unplug are exogenous, decision 0004
    # item 32, so every path shares one set of sessions).
    stocks = {"normal": sessions.plug_out_kwh[closed]}
    stocks.update(
        {
            path: by_path[path]["closing_battery_kwh"][world, last, ev]
            for path in paths
            if path != "normal"
        }
    )
    ready = totals["target_kwh"][ev] - NOT_RECOVERED_TOLERANCE_KWH
    for path, stock in stocks.items():
        totals[f"completed_count_{path}"][worlds] = np.rint(per_ev(stock >= ready)).astype(np.int64)
        soc_sum = per_ev(100.0 * stock / totals["capacity_kwh"][ev])
        totals[f"departure_soc_mean_{path}"][worlds] = np.divide(
            soc_sum, count, out=np.full(count.shape, np.nan), where=count > 0
        )


def build_frames(
    totals: dict,
    *,
    units: pd.DataFrame,
    saving_gbp: np.ndarray,
    per_ev: Mapping[str, np.ndarray],
    evening_ev_kw: np.ndarray | None,
    groups: Mapping[str, np.ndarray],
    public_charge_gbp_per_kwh: float,
) -> dict[str, pd.DataFrame]:
    """``household_ev_world`` and ``household_outcomes_summary`` from filled ``totals``.

    ``saving_gbp`` (world, EV) GBP per week is ``summaries.household_saving_gbp``
    and ``per_ev`` the third return of ``summaries.household_frames``
    (``value_gbp_per_month``, ``earning``), stored bit for bit so this frame
    and T§9.3c cannot disagree.  ``evening_ev_kw`` (world, EV, direction) is
    J2's per-EV 1-h evening mean (``availability`` totals), or ``None``
    before J1b, when the evening columns are NaN.  ``groups`` maps group id
    to the (EV,) bool mask of its treated EVs, as ``household_frames`` uses.
    ``public_charge_gbp_per_kwh`` is the illustrative public rate.
    """

    world_count, vehicle_count = saving_gbp.shape
    paths = totals["paths"]
    treated = (
        ~units["control_group"].to_numpy(dtype=bool)
        if "control_group" in units
        else np.ones(vehicle_count, dtype=bool)
    )
    evening = (
        np.full((world_count, vehicle_count, len(DIRECTIONS)), np.nan)
        if evening_ev_kw is None
        else evening_ev_kw
    )
    columns = {
        "world_id": np.repeat(np.arange(world_count, dtype=np.int64), vehicle_count),
        "unit_id": np.tile(units["unit_id"].to_numpy(dtype=object), world_count),
        "cohort_id": np.tile(units["cohort_id"].to_numpy(dtype=object), world_count),
        "treated": np.tile(treated, world_count),
        "earning": per_ev["earning"],
        **{name: totals[name] for name in _sum_columns(paths)},
        **{
            f"public_cost_gbp_{p}": totals[f"public_import_kwh_{p}"] * public_charge_gbp_per_kwh
            for p in paths
        },
        "saving_gbp": saving_gbp,
        "value_gbp_per_month": per_ev["value_gbp_per_month"],
        # The supplier P&L's energy component for this customer under the
        # flat-tariff reading (S§3.2 component 1); never added to saving_gbp.
        "supplier_energy_saving_gbp": totals["home_cost_gbp_normal"]
        - totals["home_cost_gbp_selected"],
        **{name: totals[name] for name in _count_columns(paths)},
        **{f"departure_soc_mean_{p}": totals[f"departure_soc_mean_{p}"] for p in paths},
        "co2_shifted_kg": totals["co2_shifted_kg"],
        "evening_turn_down_kw_1h": evening[..., 0],
        "evening_turn_up_kw_1h": evening[..., 1],
        "evidence_kind": EVIDENCE_KIND,
    }
    frame = pd.DataFrame(
        {
            name: np.ravel(values) if np.ndim(values) > 1 else values
            for name, values in columns.items()
        }
    )
    schema = ev_world_columns(paths)
    frame = frame.loc[:, list(schema)].astype(schema)
    return {
        "household_ev_world": frame,
        "household_outcomes_summary": outcomes_summary(
            frame, groups=groups, night_count=totals["night_count"]
        ),
    }


# --------------------------------------------------------------------------
# Metric values and the per-group summary (§3.1 metric_values, §3.2)
# --------------------------------------------------------------------------


def _share(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """numerator / denominator, NaN where the denominator is 0."""

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator != 0.0
    )


def _active_paths(ev_world: pd.DataFrame) -> tuple[str, ...]:
    """``PATHS``, plus ``TIMED_PATH`` when ``ev_world`` carries its optional columns.

    ``household_ev_world`` always carries every ``PATHS`` column; the timed
    path (decision 0007, model step 2) is present or absent as a whole, so
    checking one of its columns is enough to know which schema this frame is.
    """

    return (*PATHS, TIMED_PATH) if f"home_import_kwh_{TIMED_PATH}" in ev_world.columns else PATHS


def _ratio_parts(
    ev_world: pd.DataFrame, night_count: int, metrics: Mapping[str, tuple] | None = None
) -> dict[str, tuple]:
    """{ratio metric: (numerator, denominator)} per row of ``ev_world``, for ``metrics``.

    ``metrics`` defaults to ``RATIO_METRICS``, the base two-path set.
    """

    parts = {}
    for metric, (numerator, denominator) in (metrics or RATIO_METRICS).items():
        top = ev_world[numerator].to_numpy(dtype=float)
        bottom = (
            np.full(len(ev_world), float(night_count))
            if denominator is None
            else ev_world[denominator].to_numpy(dtype=float)
        )
        parts[metric] = (top, bottom)
    return parts


def metric_values(ev_world: pd.DataFrame, night_count: int) -> pd.DataFrame:
    """Per-(world, EV) value of every §3.2 metric: the one place a ratio is defined.

    ``ev_world`` is ``household_ev_world`` (or a subset of its rows);
    ``night_count`` is ``N_n = study_slots.night_index.max() + 1``.  Returns
    ``world_id``, ``unit_id``, ``cohort_id``, ``treated`` and one float64
    column per metric in ``household_metrics``' order for this frame's own
    paths (``_active_paths``), in the input's row order.  A ratio is NaN
    where its denominator is 0 (an EV with no closed session has no
    readiness, O4); a difference is selected minus normal inside the row,
    before any statistic, so "Δ vs unmanaged" is a paired difference.
    Weekly to monthly is x 52 / 12 and to yearly x 52 (the S§0 rule).
    """

    paths = _active_paths(ev_world)

    def column(name: str) -> np.ndarray:
        return ev_world[name].to_numpy(dtype=float)

    monthly = _WEEKS_PER_YEAR / _MONTHS_PER_YEAR
    value = column("value_gbp_per_month")
    values = {
        "value_gbp_per_week": value / monthly,
        "value_gbp_per_month": value,
        "value_gbp_per_year": value * _MONTHS_PER_YEAR,
        "saving_gbp_per_week": column("saving_gbp"),
        "saving_gbp_per_year": column("saving_gbp") * _WEEKS_PER_YEAR,
        "supplier_energy_saving_gbp_per_week": column("supplier_energy_saving_gbp"),
        "supplier_energy_saving_gbp_per_month": column("supplier_energy_saving_gbp") * monthly,
    }
    # Ratio metrics: this EV's own share in the week (set R forms the group's).
    values.update(
        {
            m: _share(*parts)
            for m, parts in _ratio_parts(ev_world, night_count, ratio_metrics(paths)).items()
        }
    )
    for path in paths:
        values[f"home_import_kwh_per_week_{path}"] = column(f"home_import_kwh_{path}")
        values[f"public_import_kwh_per_week_{path}"] = column(f"public_import_kwh_{path}")
        values[f"departure_soc_mean_{path}"] = column(f"departure_soc_mean_{path}")
    values.update(
        {
            "sessions_per_week": column("session_count"),
            "session_ends_per_week": column("session_end_count"),
            "sessions_affected_count": column("sessions_affected_count"),
            "co2_shifted_kg_per_week": column("co2_shifted_kg"),
            "co2_shifted_kg_per_month": column("co2_shifted_kg") * monthly,
            "evening_turn_down_kw_1h": column("evening_turn_down_kw_1h"),
            "evening_turn_up_kw_1h": column("evening_turn_up_kw_1h"),
        }
    )
    for metric, (selected_metric, normal_metric) in DIFFERENCE_METRICS.items():
        values[metric] = values[selected_metric] - values[normal_metric]
    keys = ev_world.loc[:, ["world_id", "unit_id", "cohort_id", "treated"]].reset_index(drop=True)
    metrics = pd.DataFrame({metric: values[metric] for metric, *_ in household_metrics(paths)})
    return pd.concat([keys, metrics], axis=1)


def _row(group_id, metric, statistic, unit, horizon, better_is, ev_count, ev_value_count, stats):
    return {
        "group_id": group_id,
        "metric": metric,
        "statistic": statistic,
        "unit": unit,
        "horizon": horizon,
        "better_is": better_is,
        "ev_count": ev_count,
        "ev_value_count": ev_value_count,
        **stats,
        "evidence_kind": EVIDENCE_KIND,
    }


def _across_evs_per_world(values: np.ndarray) -> dict[str, np.ndarray]:
    """Set A inner step: per world, statistics over EVs (NaN EVs left out; NaN when none)."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a world where every EV is NaN
        quantiles = np.nanquantile(values, (0.1, 0.5, 0.9), axis=1, method="linear")
        return {
            "mean_across_evs": np.nanmean(values, axis=1),
            "p10_across_evs": quantiles[0],
            "p50_across_evs": quantiles[1],
            "p90_across_evs": quantiles[2],
        }


def _of_ev_means(ev_means: np.ndarray) -> dict[str, float]:
    """Set B outer step: statistics over the group's defined EV means."""

    kept = ev_means[~np.isnan(ev_means)]
    if len(kept) == 0:
        return dict.fromkeys(SET_B, np.nan)
    quantiles = np.quantile(kept, (0.1, 0.5, 0.9), method="linear")
    return {
        "mean_of_ev_means": float(kept.mean()),
        "p10_of_ev_means": float(quantiles[0]),
        "p50_of_ev_means": float(quantiles[1]),
        "p90_of_ev_means": float(quantiles[2]),
        "share_of_ev_means_below_zero": float((kept < 0.0).mean()),
    }


def outcomes_summary(
    ev_world: pd.DataFrame, *, groups: Mapping[str, np.ndarray], night_count: int
) -> pd.DataFrame:
    """``household_outcomes_summary`` (§3.2): three readings per group and metric.

    ``ev_world`` is ``household_ev_world`` (world-major, EVs in ``units``
    order); ``groups`` maps group id to the (EV,) bool mask of its treated
    EVs, fleet first then cohorts in source order; ``night_count`` is N_n.

    World first, quantiles last (R§1 rule 3).  Set A: per world the
    statistic over the group's EVs, then ``world_count``, mean and linear
    P10/P50/P90 across worlds; ``week_ahead`` metrics only.  Set R: per world
    sum of numerators / sum of denominators (a difference metric: selected
    ratio minus normal ratio), then across worlds; ratio metrics only, and
    the reading any claim about "sessions" or "energy" must use (B2).  Set B:
    each EV's mean over worlds first, then the statistic over EVs, stored in
    ``mean`` and ``p50`` with ``p10``/``p90`` NaN because the weeks are inside
    each average (B1); every metric.
    """

    vehicle_count = len(next(iter(groups.values())))
    world_count = len(ev_world) // vehicle_count
    paths = _active_paths(ev_world)
    group_ratio_metrics = ratio_metrics(paths)
    values = metric_values(ev_world, night_count)
    parts = _ratio_parts(ev_world, night_count, group_ratio_metrics)

    def matrix(array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=float).reshape(world_count, vehicle_count)

    rows = []
    for group_id, members in groups.items():
        ev_count = int(members.sum())
        for metric, unit, horizon, better_is in household_metrics(paths):
            v = matrix(values[metric].to_numpy())[:, members]
            has_value = ~np.isnan(v)
            # The smallest per-world count of EVs with a value (O4).
            per_world_count = int(has_value.sum(axis=1).min()) if world_count else 0
            head = (group_id, metric)
            if horizon == "week_ahead":
                for statistic, per_world in _across_evs_per_world(v).items():
                    rows.append(
                        _row(
                            *head,
                            statistic,
                            unit,
                            horizon,
                            better_is,
                            ev_count,
                            per_world_count,
                            summaries._world_stats(per_world),
                        )
                    )
            if metric in group_ratio_metrics or metric in DIFFERENCE_METRICS:
                ratio = _group_ratio(metric, parts, matrix, members)
                rows.append(
                    _row(
                        *head,
                        "group_ratio",
                        unit,
                        horizon,
                        better_is,
                        ev_count,
                        per_world_count,
                        summaries._world_stats(ratio),
                    )
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # an EV NaN in every world
                ev_means = np.nanmean(v, axis=0)
            defined = ~np.isnan(ev_means)
            worlds_used = has_value.sum(axis=0)[defined]
            for statistic, number in _of_ev_means(ev_means).items():
                unit_b = "fraction" if statistic == "share_of_ev_means_below_zero" else unit
                stats = {
                    "world_count": int(worlds_used.min()) if defined.any() else 0,
                    "mean": number,
                    "p10": np.nan,
                    "p50": number,
                    "p90": np.nan,
                }
                rows.append(
                    _row(
                        *head,
                        statistic,
                        unit_b,
                        horizon,
                        better_is,
                        ev_count,
                        int(defined.sum()),
                        stats,
                    )
                )
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS)).astype(SUMMARY_COLUMNS)


def _group_ratio(metric, parts, matrix, members) -> np.ndarray:
    """Set R per world: Σ numerator / Σ denominator over the group (a difference: sel - normal)."""

    if metric in DIFFERENCE_METRICS:
        selected_metric, normal_metric = DIFFERENCE_METRICS[metric]
        return _group_ratio(selected_metric, parts, matrix, members) - _group_ratio(
            normal_metric, parts, matrix, members
        )
    numerator, denominator = parts[metric]
    return _share(
        matrix(numerator)[:, members].sum(axis=1), matrix(denominator)[:, members].sum(axis=1)
    )


# --------------------------------------------------------------------------
# One EV on demand: the household card (§3.3, lane HH2)
# --------------------------------------------------------------------------

AVAILABILITY_QUANTILES = (("p05", 0.05), ("p10", 0.10), ("p50", 0.50), ("p90", 0.90), ("p95", 0.95))
"""The card's across-weeks quantiles; P5 and P95 only here, for one device's tails (§12 Q12)."""
DAY_TYPES = ("weekday", "weekend", "all")
TIMING_METRICS = ("plug_in_time", "departure_time")
_HALF_HOURS = 24 * 2
_NOON = 12.0


@dataclass(frozen=True)
class HouseholdCard:
    """One EV across every simulated week, beside its archetype and the fleet (§3.3).

    ``treated`` False: a hold-out control EV, not in the product.
    ``dispatch_locked`` is ``units.dispatch_locked`` (D§7.3) or ``None``
    when the run has no such column.  ``availability`` and
    ``availability_day`` are ``None`` until J1b's per-EV plan outputs and
    the run's availability inputs exist.  Every figure is illustrative.
    """

    unit_id: str
    cohort_id: str
    treated: bool
    dispatch_locked: bool | None
    world_count: int
    outcomes: pd.DataFrame
    availability: pd.DataFrame | None
    availability_day: pd.DataFrame | None
    session_timing: pd.DataFrame
    reliability: pd.DataFrame


def household_card(result, unit_id: str) -> HouseholdCard:
    """Build (or return the cached) ``HouseholdCard`` of one EV of an action result.

    Raises ``KeyError`` for an unknown ``unit_id`` and ``ValueError`` when
    the result has no household frames (a no-action result).  Outcomes and
    ranks are reads of ``household_ev_world`` and ``household_outcomes_summary``;
    timing, reliability and availability need this EV's per-slot arrays, so
    the kernel runs once on this EV for every world.  That is exact because
    EVs are physically independent (``summaries.KernelSlice``), the
    ``replay_one_ev_bands`` precedent.  The card is cached on the result's
    ``replay_cache`` under ``("household", unit_id)``.
    """

    if getattr(result, "household_ev_world", None) is None:
        raise ValueError("household frames are not on this result")
    units = result.replay_state.units
    matches = np.flatnonzero(units["unit_id"].to_numpy() == unit_id)
    if len(matches) != 1:
        raise KeyError(f"unknown unit_id: {unit_id}")
    cache = result.replay_state.replay_cache
    key = ("household", unit_id)
    if key not in cache:
        arrays = one_ev_arrays(result.replay_state, int(matches[0]), result.study_slots)
        card = build_card(result, unit_id, **arrays)
        # A private import accepted by the contract (O6): the simpler choice;
        # promote _remember to a public helper when a third caller appears.
        # Imported here because individual imports summaries, which imports
        # this module for the chunk hook.
        from axle_studio.model.individual import _remember

        _remember(cache, key, card)
    return cache[key]


def one_ev_arrays(state, ev: int, study_slots: pd.DataFrame) -> dict:
    """This EV's normal-path per-slot arrays for every world, from one kernel run.

    Returns ``build_card``'s keyword arguments: ``connected`` (bool),
    ``closing_kwh`` and ``home_import_kwh`` (kWh), each (world, slot);
    ``realised_kw``, ``{direction: (world, slot) kW}`` of its realised 1-h
    contribution on the selected path, or ``None`` while the run lacks the
    availability inputs or the kernel lacks J1b's per-EV plan outputs
    (``availability.ev_contributions`` needs them); and the EV's constants.
    """

    worlds = np.arange(state.settings.evaluation_world_count)
    normal_slice = summaries.kernel_slice(state, worlds, [ev], path_id="normal")
    normal = summaries._simulate(normal_slice)
    slot_start_ns = pd.DatetimeIndex(study_slots["interval_start_utc"]).as_unit("ns").asi8
    # The kernel does not return the connection state; this is the same
    # re-derivation the chunk loop uses (and checks against the fleet).
    connected = summaries._home_connected(normal_slice.inputs, slot_start_ns)
    realised_kw = None
    flex, inputs = state.flexibility, state.availability_inputs
    if flex is not None and inputs is not None and state.smart_charging is not None:
        selected = summaries._simulate(
            summaries.kernel_slice(state, worlds, [ev], path_id="selected")
        )
        if "plan_status" in selected:
            selected["connected"] = connected
            out = availability.ev_contributions(
                selected,
                power_kw=flex.power_kw[[ev]],
                target_kwh=flex.target_kwh[[ev]],
                efficiency=flex.efficiency,
                expected_end_slot=availability.expected_end_slots(flex.hours_to_departure[:, [ev]]),
                night_index=study_slots["night_index"].to_numpy(),
                night_decision_slots=availability.decision_slots(
                    study_slots, inputs.decision_local_time
                ),
                blackout=inputs.blackout,
                duration_slots=2,
            )
            realised_kw = {d: out["realised"][d][..., 0] for d in DIRECTIONS}
    unit = state.units.iloc[ev]
    capacity = float(unit["physical_capacity_kwh"])
    # The run's own efficiency, from the stored flexibility inputs when kept.
    efficiency = flex.efficiency if flex is not None else state.settings.home_charge_efficiency
    return {
        "connected": connected[..., 0],
        "closing_kwh": normal["closing_battery_kwh"][..., 0],
        "home_import_kwh": normal["home_grid_import_kwh"][..., 0],
        "realised_kw": realised_kw,
        "capacity_kwh": capacity,
        "target_kwh": capacity * float(unit["preferred_target_soc_fraction"]),
        "power_kw": float(unit["home_charger_limit_kw"]),
        "efficiency": float(efficiency),
    }


def build_card(
    result,
    unit_id: str,
    *,
    connected: np.ndarray,
    closing_kwh: np.ndarray,
    home_import_kwh: np.ndarray,
    realised_kw: Mapping[str, np.ndarray] | None,
    capacity_kwh: float,
    target_kwh: float,
    power_kw: float,
    efficiency: float,
) -> HouseholdCard:
    """Assemble the card from this EV's per-world, per-slot normal-path arrays.

    ``connected``, ``closing_kwh`` and ``home_import_kwh`` are (world, slot);
    ``realised_kw`` is ``{direction: (world, slot) kW}`` or ``None``;
    ``capacity_kwh``, ``target_kwh`` (preferred target stock), ``power_kw``
    (home charger) and ``efficiency`` (battery kWh per grid kWh) are this
    EV's constants.  Split from ``household_card`` so the synthetic fixture
    can build a card from its own toy arrays.
    """

    row = result.units.loc[result.units["unit_id"].eq(unit_id)].iloc[0]
    slots = result.study_slots
    night_count = int(slots["night_index"].max()) + 1
    sessions = summaries.find_sessions(
        connected[..., np.newaxis], closing_kwh[..., np.newaxis], home_import_kwh[..., np.newaxis]
    )
    values = summaries.session_values(
        sessions,
        study_slots=slots,
        capacity_kwh=np.array([capacity_kwh]),
        target_kwh=np.array([target_kwh]),
        power_kw=np.array([power_kw]),
        efficiency=efficiency,
    )
    ev_rows = result.household_ev_world.loc[result.household_ev_world["unit_id"].eq(unit_id)]
    profile, profile_day = (None, None)
    if realised_kw is not None:
        profile, profile_day = availability_profile(realised_kw, slots)
    return HouseholdCard(
        unit_id=unit_id,
        cohort_id=str(row["cohort_id"]),
        treated=bool(ev_rows["treated"].iat[0]),
        dispatch_locked=bool(row["dispatch_locked"]) if "dispatch_locked" in row else None,
        world_count=int(result.world_count),
        outcomes=card_outcomes(
            result.household_ev_world, result.household_outcomes_summary, unit_id, night_count
        ),
        availability=profile,
        availability_day=profile_day,
        session_timing=session_timing(values),
        reliability=reliability(
            values,
            world=sessions.world,
            world_count=int(result.world_count),
            ev_rows=ev_rows,
            night_count=night_count,
        ),
    )


def card_outcomes(
    ev_world: pd.DataFrame, summary: pd.DataFrame, unit_id: str, night_count: int
) -> pd.DataFrame:
    """§3.3a: this EV's outcomes across weeks beside its cohort and the fleet, with ranks.

    Weekly metrics: this EV's per-world values, statistics across worlds;
    group columns are the set-A across-EV quantiles at their median across
    weeks; the rank is per world the share of the group's treated EVs (with
    a value) at or below this EV, then the median across worlds.  Scenario
    metrics: ``mean`` is this EV's mean over weeks (already scaled) and its
    quantiles are NaN, so no one-week spread is ever read as a yearly range
    (B1); group columns are the set-B quantiles of EV means and the rank is
    among the group's EV means.  Ranks are NaN for a control EV.
    """

    values = metric_values(ev_world, night_count)
    world_count = int(values["world_id"].max()) + 1
    vehicle_count = len(values) // world_count
    unit_ids = values["unit_id"].to_numpy(dtype=object)[:vehicle_count]
    ev = int(np.flatnonzero(unit_ids == unit_id)[0])
    treated = values["treated"].to_numpy(dtype=bool)[:vehicle_count]
    cohorts = values["cohort_id"].to_numpy(dtype=object)[:vehicle_count]
    cohort_id = cohorts[ev]
    members = {"cohort": treated & (cohorts == cohort_id), "fleet": treated}
    group_ids = {"cohort": cohort_id, "fleet": "fleet"}
    stored = summary.set_index(["group_id", "metric", "statistic"])

    def group_value(group_id, metric, statistic, column):
        key = (group_id, metric, statistic)
        return float(stored.at[key, column]) if key in stored.index else np.nan

    rows = []
    for metric, unit, horizon, better_is in household_metrics(_active_paths(ev_world)):
        v = values[metric].to_numpy(dtype=float).reshape(world_count, vehicle_count)
        mine = v[:, ev]
        row = {"metric": metric, "unit": unit, "horizon": horizon, "better_is": better_is}
        if horizon == "week_ahead":
            row.update(summaries._world_stats(mine))
            for name, group_id in group_ids.items():
                for level in ("p10", "p50", "p90"):
                    row[f"{name}_{level}"] = group_value(
                        group_id, metric, f"{level}_across_evs", "p50"
                    )
                group = v[:, members[name]]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    at_or_below = np.nansum(group <= mine[:, np.newaxis], axis=1) / np.sum(
                        ~np.isnan(group), axis=1
                    )
                at_or_below[np.isnan(mine)] = np.nan
                row[f"{name}_rank_p50"] = summaries._world_stats(at_or_below)["p50"]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                means = np.nanmean(v, axis=0)
            kept = ~np.isnan(mine)
            row.update(
                world_count=int(kept.sum()), mean=means[ev], p10=np.nan, p50=np.nan, p90=np.nan
            )
            for name, group_id in group_ids.items():
                for level in ("p10", "p50", "p90"):
                    row[f"{name}_{level}"] = group_value(
                        group_id, metric, f"{level}_of_ev_means", "mean"
                    )
                group = means[members[name]]
                group = group[~np.isnan(group)]
                row[f"{name}_rank_p50"] = (
                    float((group <= means[ev]).mean())
                    if len(group) and not np.isnan(means[ev])
                    else np.nan
                )
        if not treated[ev]:
            row["cohort_rank_p50"] = row["fleet_rank_p50"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=list(CARD_OUTCOME_COLUMNS)).astype(CARD_OUTCOME_COLUMNS)


CARD_OUTCOME_COLUMNS = {
    "metric": object,
    "unit": object,
    "horizon": object,
    "better_is": object,
    "world_count": np.int64,
    **dict.fromkeys(("mean", "p10", "p50", "p90"), np.float64),
    **dict.fromkeys(
        (f"{g}_{q}" for g in ("cohort", "fleet") for q in ("p10", "p50", "p90")), np.float64
    ),
    "cohort_rank_p50": np.float64,
    "fleet_rank_p50": np.float64,
}


def _hours_since_noon(clock: np.ndarray) -> np.ndarray:
    """London clock hours as hours since the session night's noon: 18:00 is 6, 02:30 is 14.5."""

    return (np.asarray(clock, dtype=float) - _NOON) % 24.0


def _clock(hours_since_noon: float) -> float:
    return float((hours_since_noon + _NOON) % 24.0)


def clock_label(hours: float) -> str:
    """``"HH:MM"`` of a decimal London clock hour, to the nearest minute; ``""`` for NaN."""

    if np.isnan(hours):
        return ""
    minutes = int(round(hours * 60.0)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def session_timing(values: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """§3.3c: share of this driver's sessions by London half-hour, pooled over weeks.

    ``values`` are ``summaries.session_values`` for this EV's sessions in
    every world.  Plug-in time counts every session (the unplug is not
    needed); departure time only closed ones (NaN otherwise).  One EV has
    five to seven sessions a week, too few for a per-week share, so sessions
    are pooled across weeks (the R§4.7a-b pooling): a habit, not a band.
    ``profile_order`` puts 12:00 first so an overnight habit is one hump.
    """

    rows = []
    for metric in TIMING_METRICS:
        clock = np.asarray(values[metric], dtype=float)
        clock = clock[~np.isnan(clock)]
        # Rounded as _session_bin does, so an exact half-hour lands in its bin.
        index = np.clip(np.floor(np.round(clock, 9) / 0.5), 0, _HALF_HOURS - 1).astype(int)
        counts = np.bincount(index, minlength=_HALF_HOURS)
        total = int(counts.sum())
        for bin_index in range(_HALF_HOURS):
            rows.append(
                {
                    "metric": metric,
                    "unit": "hour (London)",
                    "bin_index": bin_index,
                    "profile_order": summaries._profile_order(bin_index),
                    "bin_lower": bin_index * 0.5,
                    "bin_upper": (bin_index + 1) * 0.5,
                    "bin_label": summaries._half_hour_label(bin_index),
                    "session_count": total,
                    "share": counts[bin_index] / total if total else np.nan,
                    "evidence_kind": EVIDENCE_KIND,
                }
            )
    return pd.DataFrame(rows).astype(
        {"bin_index": np.int64, "profile_order": np.int64, "session_count": np.int64}
    )


RELIABILITY_ROWS = (
    ("nights_plugged_share", "fraction", "weeks"),
    ("sessions_per_week", "sessions per week", "weeks"),
    ("session_ends_per_week", "session ends per week", "weeks"),
    ("plug_in_time", "hour (London)", "sessions pooled across weeks"),
    ("departure_time", "hour (London)", "sessions pooled across weeks"),
    ("dwell_hours", "hours", "sessions pooled across weeks"),
    ("energy_needed_kwh", "kWh battery-side", "sessions pooled across weeks"),
    ("flexible_kwh_per_night", "kWh battery-side", "weeks"),
    ("evening_turn_down_kw_1h", "kW", "weeks"),
    ("evening_turn_up_kw_1h", "kW", "weeks"),
)
"""(metric, unit, sample_kind) of each §3.3d reliability row, in order."""


def reliability(
    values: Mapping[str, np.ndarray],
    *,
    world: np.ndarray,
    world_count: int,
    ev_rows: pd.DataFrame,
    night_count: int,
) -> pd.DataFrame:
    """§3.3d: how reliably this driver is plugged in, when, and what could move.

    ``values`` are this EV's ``session_values``, ``world`` each session's
    world index; ``ev_rows`` its ``household_ev_world`` rows in world order.
    "weeks" rows are statistics across worlds of a per-world value;
    "sessions" rows are over sessions pooled across worlds.  Clock times are
    quantiled as hours since the night's noon and converted back, so plug-ins
    at 23:00, 23:30 and 00:30 have a median of 23:30, not 23:00 (B3; anchored
    at noon, so daytime times either side of 12:00 split, §11).
    """

    per_world = {
        "nights_plugged_share": ev_rows["nights_plugged_count"].to_numpy(dtype=float) / night_count,
        "sessions_per_week": ev_rows["session_count"].to_numpy(dtype=float),
        "session_ends_per_week": ev_rows["session_end_count"].to_numpy(dtype=float),
        "evening_turn_down_kw_1h": ev_rows["evening_turn_down_kw_1h"].to_numpy(dtype=float),
        "evening_turn_up_kw_1h": ev_rows["evening_turn_up_kw_1h"].to_numpy(dtype=float),
    }
    # R§3.10c's flexible kWh summed over the week's closed sessions, per night.
    flexible = np.asarray(values["flexible_kwh"], dtype=float)
    closed = ~np.isnan(flexible)
    per_world["flexible_kwh_per_night"] = (
        np.bincount(world[closed], weights=flexible[closed], minlength=world_count) / night_count
    )
    rows = []
    for metric, unit, sample_kind in RELIABILITY_ROWS:
        labels = ("", "", "")
        if sample_kind == "weeks":
            stats = summaries._world_stats(per_world[metric])
            count, mean, p10, p50, p90 = (
                stats["world_count"],
                stats["mean"],
                stats["p10"],
                stats["p50"],
                stats["p90"],
            )
        else:
            sample = np.asarray(values[metric], dtype=float)
            sample = sample[~np.isnan(sample)]
            count = len(sample)
            is_clock = metric in TIMING_METRICS
            shifted = _hours_since_noon(sample) if is_clock else sample
            if count:
                mean = float(shifted.mean())
                p10, p50, p90 = (float(q) for q in np.quantile(shifted, (0.1, 0.5, 0.9)))
            else:
                mean = p10 = p50 = p90 = np.nan
            if is_clock and count:
                mean, p10, p50, p90 = (_clock(x) for x in (mean, p10, p50, p90))
                labels = (clock_label(p10), clock_label(p50), clock_label(p90))
        rows.append(
            {
                "metric": metric,
                "unit": unit,
                "sample_kind": sample_kind,
                "sample_count": count,
                "mean": mean,
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "p10_label": labels[0],
                "p50_label": labels[1],
                "p90_label": labels[2],
                "ratio_p10_to_p50": p10 / p50 if p50 > 0 else np.nan,
                "evidence_kind": EVIDENCE_KIND,
            }
        )
    return pd.DataFrame(rows).astype({"sample_count": np.int64})


def _band_stats(per_world: np.ndarray) -> dict[str, np.ndarray]:
    """Across-world statistics of (world, n) kW values, NaN worlds left out per column."""

    kept = ~np.isnan(per_world)
    count = kept.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a column with no world
        stats = {
            "world_count": count.astype(np.int64),
            # Share of weeks the EV can move charging at all (O2).
            "available_share": np.where(
                count > 0, np.nansum(per_world > 0, axis=0) / np.maximum(count, 1), np.nan
            ),
            "mean": np.nanmean(per_world, axis=0),
        }
        quantiles = np.nanquantile(
            per_world, [q for _, q in AVAILABILITY_QUANTILES], axis=0, method="linear"
        )
    stats.update({name: quantiles[i] for i, (name, _) in enumerate(AVAILABILITY_QUANTILES)})
    # P90 firm is the across-weeks P10 (the T§10 convention, B10).
    stats["firm_share"] = np.divide(
        stats["p10"],
        stats["p50"],
        out=np.full(stats["p50"].shape, np.nan),
        where=stats["p50"] > 0,
    )
    return stats


def availability_profile(
    realised_kw: Mapping[str, np.ndarray], study_slots: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """§3.3b: this EV's 1-h realised kW across weeks, by slot and as a typical day.

    ``realised_kw`` is ``{direction: (world, slot) kW}``, NaN where the
    window runs past the study end.  The typical day takes per world the
    mean over the day type's slots at each London half-hour (two slots on
    the repeated autumn hour, none on the skipped spring one: the
    ``average_day_bands`` rule), then statistics across worlds.
    """

    keys = study_slots.loc[
        :, ["slot_index", "night_index", "interval_start_utc", "interval_start_london"]
    ]
    half_hour = study_slots["local_half_hour"].to_numpy()
    day_type = study_slots["day_type"].to_numpy(dtype=object)
    slot_rows, day_rows = [], []
    for direction in DIRECTIONS:
        values = np.asarray(realised_kw[direction], dtype=float)
        frame = keys.copy()
        frame.insert(0, "direction", direction)
        frame["unit"] = "kW"
        for name, column in _band_stats(values).items():
            frame[name] = column
        slot_rows.append(frame)
        folded = np.full((values.shape[0], len(DAY_TYPES) * _HALF_HOURS), np.nan)
        for d, kind in enumerate(DAY_TYPES):
            in_type = np.ones(len(day_type), dtype=bool) if kind == "all" else day_type == kind
            for h in range(_HALF_HOURS):
                mask = in_type & (half_hour == h)
                if mask.any():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        folded[:, d * _HALF_HOURS + h] = np.nanmean(values[:, mask], axis=1)
        day = pd.DataFrame(
            {
                "direction": direction,
                "day_type": np.repeat(DAY_TYPES, _HALF_HOURS),
                "local_half_hour": np.tile(np.arange(_HALF_HOURS, dtype=np.int64), len(DAY_TYPES)),
            }
        )
        day["local_time_label"] = [summaries._half_hour_label(h) for h in day["local_half_hour"]]
        day["unit"] = "kW"
        for name, column in _band_stats(folded).items():
            day[name] = column
        day_rows.append(day)
    by_slot = pd.concat(slot_rows, ignore_index=True)
    by_day = pd.concat(day_rows, ignore_index=True)
    by_slot["evidence_kind"] = by_day["evidence_kind"] = EVIDENCE_KIND
    return by_slot, by_day


__all__ = [
    "EV_WORLD_COLUMNS",
    "METRICS",
    "PATHS",
    "RATIO_METRICS",
    "SUMMARY_COLUMNS",
    "TIMED_PATH",
    "HouseholdCard",
    "accumulate",
    "availability_profile",
    "build_card",
    "build_frames",
    "ev_world_columns",
    "household_card",
    "household_metrics",
    "metric_values",
    "new_totals",
    "one_ev_arrays",
    "outcomes_summary",
    "ratio_metrics",
]
