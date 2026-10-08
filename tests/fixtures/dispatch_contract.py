"""Intraday dispatch frames (lane K3) specs and validator (intraday-dispatch-v1 contract §7, §8).

What this owns: a SYNTHETIC ``FixtureDispatchSums`` that mirrors the
kernel's ``DispatchSums`` holder exactly as contract §5.2 specifies its
fields (K1 owns the real one in ``action.py`` / ``physics.py``; this lets
K3 develop and test the frame builders in ``model/summaries.py`` before K1
lands, per the lead's task line), the independent §2 locked-EV counting
rule (``locked_ev_counts`` / ``locked_ev_mask``, so this file can check a
real result's ``units.dispatch_locked`` without importing K1's
``action.locked_evs``), and ``validate_dispatch_frames(result)``.

The validator checks contract §8's rules against ``dispatch_world_slot``,
``dispatch_bands``, ``dispatch_split``, ``units.dispatch_locked`` and the
``event_response_bands`` ``day_ahead_plan`` series addition, everything
recomputed independently from the frames a result already carries (never
by calling ``model.summaries``'s own builders, so a bug shared between the
model and this file would not cancel out).  It is duck-typed (attributes
and columns only) like the other contract fixtures, so it validates the
SYNTHETIC case below and a real ``ForecastResult`` alike.  Tolerance is
``1e-9 x max(1, |value|)`` unless stated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .result_contract import _check_dtypes, _check_ordered_quantiles, _check_unique

EVIDENCE_KIND = "illustrative_synthetic"
"""``TRADING_EVIDENCE`` in ``model/summaries.py``, written out so this check
does not import the model's own constant (independence, module docstring)."""

DISPATCH_WORLD_SLOT_COLUMNS = {
    "world_id": "int",
    "slot_index": "int",
    "night_index": "int",
    "interval_start_utc": "utc",
    "interval_start_london": "london",
    "day_ahead_plan_kwh": "float",
    "dispatched_kwh": "float",
    "locked_kwh": "float",
    "free_kwh": "float",
    "free_day_ahead_plan_kwh": "float",
    "moved_kwh": "float",
    "replan_count": "int",
    "latest_close_gbp_per_mwh": "float",
    "evidence_kind": "str",
}
DISPATCH_BAND_COLUMNS = {
    "series": "str",
    "metric": "str",
    "unit": "str",
    "slot_index": "int",
    "interval_start_utc": "utc",
    "interval_start_london": "london",
    "world_count": "int",
    "mean": "float",
    "p10": "float",
    "p50": "float",
    "p90": "float",
    "evidence_kind": "str",
}
DISPATCH_SPLIT_COLUMNS = {
    "cohort_id": "str",
    "ev_count": "int",
    "locked_count": "int",
    "free_count": "int",
    "commitment_share": "float",
    "replan_threshold_gbp_per_mwh": "float",
    "evidence_kind": "str",
}
DISPATCH_BAND_SERIES = ("day_ahead_plan", "dispatched", "difference")
DISPATCH_FRAME_FIELDS = ("dispatch_world_slot", "dispatch_bands", "dispatch_split")


@dataclass(frozen=True)
class FixtureDispatchSums:
    """SYNTHETIC mirror of the kernel's ``DispatchSums`` holder (contract §5.2).

    Every array is ``(world, study slot)`` except ``day_ahead_zone_home_import_kwh``
    (``(world, zone, study slot)``) and ``book_kwh`` (``path -> (world,
    decision, 48)``, unused by K3's frame builders and defaulted empty
    here).  Built by hand for tests: K1 owns the real ``DispatchSums``.
    """

    day_ahead_home_import_kwh: np.ndarray
    day_ahead_zone_home_import_kwh: np.ndarray
    locked_home_import_kwh: np.ndarray
    free_home_import_kwh: np.ndarray
    free_day_ahead_home_import_kwh: np.ndarray
    replan_count: np.ndarray
    book_kwh: dict[str, np.ndarray] = field(default_factory=dict)


def round_half_up(value: float) -> int:
    """Contract §2: ``K = round-half-up(c x N)``."""

    return math.floor(value + 0.5)


def locked_ev_counts(cohort_sizes: Sequence[int], commitment_share: float) -> list[int]:
    """Contract §2: split ``K`` across cohorts by largest remainder on ``K x n_c / N``.

    Ties go to the earlier cohort (the catalogue's own order): a stable
    sort on descending fraction keeps equal-fraction cohorts in their
    original order, which is exactly "earlier wins".  Independent
    reimplementation of ``action.locked_evs``'s counting rule (K1's lane),
    so a real result's ``units.dispatch_locked`` can be checked without
    importing K1's code.
    """

    total = sum(cohort_sizes)
    if total == 0:
        return [0 for _ in cohort_sizes]
    target = round_half_up(commitment_share * total)
    exact = [target * size / total for size in cohort_sizes]
    counts = [math.floor(value) for value in exact]
    remainder = target - sum(counts)
    fractions = [value - floor for value, floor in zip(exact, counts, strict=True)]
    order = sorted(range(len(cohort_sizes)), key=lambda i: -fractions[i])
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def locked_ev_mask(cohort_codes: np.ndarray, locked_counts: Sequence[int]) -> np.ndarray:
    """Contract §2: the first ``locked_counts[c]`` EVs of each cohort, in ``units`` row order."""

    locked = np.zeros(len(cohort_codes), dtype=bool)
    for code, count in enumerate(locked_counts):
        locked[np.flatnonzero(cohort_codes == code)[:count]] = True
    return locked


def _close(actual, expected, name: str) -> None:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    tolerance = 1e-9 * np.maximum(1.0, np.abs(np.nan_to_num(expected)))
    same_nan = np.isnan(actual) == np.isnan(expected)
    assert same_nan.all(), f"{name}: NaN pattern differs"
    diff = np.abs(np.nan_to_num(actual) - np.nan_to_num(expected))
    assert (diff <= tolerance).all(), f"{name}: max difference {diff.max():.3g}"


def _matrix(frame: pd.DataFrame, column: str, world_count: int) -> np.ndarray:
    ordered = frame.sort_values(["world_id", "slot_index"], kind="stable")
    return ordered[column].to_numpy(dtype=float).reshape(world_count, -1)


def _band_statistics_from(values: np.ndarray) -> dict[str, np.ndarray]:
    """Independent recomputation of ``summaries._band_statistics`` (mean, linear P10/P50/P90)."""

    q = np.quantile(values, (0.1, 0.5, 0.9), axis=0, method="linear")
    return {
        "world_count": values.shape[0],
        "mean": values.mean(axis=0),
        "p10": q[0],
        "p50": q[1],
        "p90": q[2],
    }


def dispatch_bands_from(world_slot: pd.DataFrame) -> dict[str, np.ndarray]:
    """Independent world-first recomputation of ``dispatch_bands`` (§7.2).

    Built from ``dispatch_world_slot``.  Returns ``{(series, metric):
    stats}`` where ``stats`` has ``mean``, ``p10``, ``p50``, ``p90``
    (``world_count`` is a scalar, checked separately).  Written from the
    contract, not by calling ``model.summaries.dispatch_bands``.
    """

    world_count = int(world_slot["world_id"].nunique())
    day_ahead_kw = _matrix(world_slot, "day_ahead_plan_kwh", world_count) / 0.5
    dispatched_kw = _matrix(world_slot, "dispatched_kwh", world_count) / 0.5
    replan = _matrix(world_slot, "replan_count", world_count)
    return {
        ("day_ahead_plan", "home_import_kw"): _band_statistics_from(day_ahead_kw),
        ("dispatched", "home_import_kw"): _band_statistics_from(dispatched_kw),
        ("difference", "home_import_kw"): _band_statistics_from(dispatched_kw - day_ahead_kw),
        ("dispatched", "replan_count"): _band_statistics_from(replan),
    }


def validate_dispatch_frames(result) -> None:
    """Assert intraday-dispatch-v1 §8 rules against ``result``'s dispatch frames.

    ``units.dispatch_locked`` is checked on every result (present, bool);
    when ``dispatch_world_slot`` is ``None`` (the switch off, or a
    no-action result, §7 header) the other two frames must also be
    ``None`` and every EV must be unlocked (§8, "all False ... with the
    switch off").  Otherwise every frame is present and the §8 identities
    are checked against ``fleet_world_intervals`` and ``evaluation_prices``,
    which ``result`` already carries.
    """

    units = result.units
    assert "dispatch_locked" in units.columns, "units.dispatch_locked required on every result"
    locked = units["dispatch_locked"]
    assert locked.dtype == np.dtype(bool), "units.dispatch_locked must be bool"

    frames = {name: getattr(result, name, None) for name in DISPATCH_FRAME_FIELDS}
    # The run's own records, when it carries them (``run_forecast_from_assumptions``;
    # a direct ``run_forecast`` call or the SYNTHETIC fixture may not).
    records = {record.name: record.value for record in (getattr(result, "assumptions", None) or ())}
    switch = records.get("trading.intraday_dispatch")
    if switch is not None:
        dispatched = result.model == "action" and bool(switch)
        assert (frames["dispatch_world_slot"] is not None) == dispatched, (
            "dispatch frames present exactly on an action result with the switch on"
        )
    if frames["dispatch_world_slot"] is None:
        assert all(frame is None for frame in frames.values()), (
            "dispatch frames must be None together (switch off or no-action result)"
        )
        assert not locked.any(), "units.dispatch_locked must be all False without the dispatch"
        _validate_event_response_no_dispatch(result)
        return
    assert all(frame is not None for frame in frames.values()), "every dispatch frame is present"

    world_slot = frames["dispatch_world_slot"]
    _check_dtypes(world_slot, DISPATCH_WORLD_SLOT_COLUMNS, "dispatch_world_slot")
    _check_unique(world_slot, ["world_id", "slot_index"], "dispatch_world_slot")
    world_count = int(result.world_count)

    fleet = result.fleet_world_intervals
    selected = fleet.loc[fleet["path_id"].eq("selected")]
    dispatched = _matrix(selected, "home_import_kwh", world_count)
    _close(
        _matrix(world_slot, "dispatched_kwh", world_count),
        dispatched,
        "dispatch_world_slot.dispatched_kwh vs fleet_world_intervals",
    )
    locked_kwh = _matrix(world_slot, "locked_kwh", world_count)
    free_kwh = _matrix(world_slot, "free_kwh", world_count)
    day_ahead_kwh = _matrix(world_slot, "day_ahead_plan_kwh", world_count)
    free_day_ahead_kwh = _matrix(world_slot, "free_day_ahead_plan_kwh", world_count)
    _close(locked_kwh + free_kwh, dispatched, "dispatch_world_slot: locked + free = dispatched")
    _close(
        locked_kwh + free_day_ahead_kwh,
        day_ahead_kwh,
        "dispatch_world_slot: locked + free_day_ahead_plan = day_ahead_plan",
    )
    _close(
        _matrix(world_slot, "moved_kwh", world_count),
        dispatched - day_ahead_kwh,
        "dispatch_world_slot.moved_kwh recomputation",
    )

    replan = world_slot["replan_count"].to_numpy(dtype=np.int64)
    assert (replan >= 0).all(), "dispatch_world_slot.replan_count >= 0"
    # Decision instants are whole London/UTC hours (contract §1: London and
    # UTC whole hours coincide, since the offset is always a whole number
    # of hours); a non-decision slot must carry no replans.
    decision_slot = world_slot["interval_start_utc"].dt.minute.eq(0).to_numpy()
    assert (replan[~decision_slot] == 0).all(), "replan_count is 0 off decision slots"
    free_total = int(frames["dispatch_split"]["free_count"].sum())
    assert (replan <= free_total).all(), "replan_count <= free EVs in the fleet"

    close = world_slot["latest_close_gbp_per_mwh"].to_numpy(dtype=float)
    evaluation = result.evaluation_prices
    merged = world_slot[["world_id", "slot_index"]].merge(
        evaluation[["world_id", "slot_index", "evaluation_context_price_gbp_per_mwh"]],
        on=["world_id", "slot_index"],
        how="left",
    )
    assert len(merged) == len(world_slot), (
        "dispatch_world_slot keys must all be in evaluation_prices"
    )
    _close(
        close,
        merged["evaluation_context_price_gbp_per_mwh"],
        "latest_close_gbp_per_mwh vs evaluation_prices",
    )

    bands = frames["dispatch_bands"]
    _check_dtypes(bands, DISPATCH_BAND_COLUMNS, "dispatch_bands")
    _check_unique(bands, ["series", "metric", "slot_index"], "dispatch_bands")
    _check_ordered_quantiles(bands, "p10", "p50", "p90", "dispatch_bands")
    assert bands["series"].isin(DISPATCH_BAND_SERIES).all(), "dispatch_bands.series vocabulary"
    expected = dispatch_bands_from(world_slot)
    for (series, metric), stats in expected.items():
        rows = bands.loc[bands["series"].eq(series) & bands["metric"].eq(metric)]
        assert len(rows) == len(world_slot["slot_index"].unique()), (
            f"dispatch_bands {series}/{metric} row count"
        )
        rows = rows.sort_values("slot_index")
        assert (rows["world_count"] == stats["world_count"]).all(), f"{series}/{metric} world_count"
        for stat in ("mean", "p10", "p50", "p90"):
            _close(rows[stat], stats[stat], f"dispatch_bands {series}/{metric} {stat}")

    split = frames["dispatch_split"]
    _check_dtypes(split, DISPATCH_SPLIT_COLUMNS, "dispatch_split")
    _check_unique(split, ["cohort_id"], "dispatch_split")
    assert int(split["ev_count"].sum()) == result.vehicle_count, "dispatch_split ev_count sums"
    assert (split["free_count"] == split["ev_count"] - split["locked_count"]).all(), (
        "dispatch_split free_count = ev_count - locked_count"
    )
    catalogue_ids = list(split["cohort_id"])
    codes = units["cohort_id"].map(catalogue_ids.index).to_numpy(dtype=np.int64)
    code_range = range(len(catalogue_ids))
    counted = pd.Series(codes).value_counts().reindex(code_range, fill_value=0)
    assert (split["ev_count"].to_numpy() == counted.to_numpy()).all(), (
        "dispatch_split ev_count vs units"
    )
    locked_codes = codes[locked.to_numpy(dtype=bool)]
    locked_counted = pd.Series(locked_codes).value_counts().reindex(code_range, fill_value=0)
    assert (split["locked_count"].to_numpy() == locked_counted.to_numpy()).all(), (
        "dispatch_split.locked_count must equal the per-cohort sums of units.dispatch_locked"
    )
    assert split["commitment_share"].nunique() == 1, "dispatch_split.commitment_share is one value"
    assert split["replan_threshold_gbp_per_mwh"].nunique() == 1, (
        "dispatch_split.replan_threshold is one value"
    )
    commitment_share = float(split["commitment_share"].iat[0])
    for column, name in (
        ("commitment_share", "trading.day_ahead_commitment_share"),
        ("replan_threshold_gbp_per_mwh", "trading.replan_threshold_gbp_per_mwh"),
    ):
        if name in records:
            assert (split[column] == float(records[name])).all(), (
                f"dispatch_split.{column} = {name}"
            )
    # §2: K = round-half-up(c x N) by largest remainder, the first k_c EVs of
    # each cohort in units row order, recomputed independently.
    expected_counts = locked_ev_counts(split["ev_count"].tolist(), commitment_share)
    assert split["locked_count"].tolist() == expected_counts, "dispatch_split counts follow §2"
    assert (locked.to_numpy(dtype=bool) == locked_ev_mask(codes, expected_counts)).all(), (
        "units.dispatch_locked is the first k_c EVs of each cohort"
    )

    _validate_event_response_dispatch(result, world_slot)


def _validate_event_response_no_dispatch(result) -> None:
    bands = getattr(result, "event_response_bands", None)
    if bands is None or bands.empty:
        return
    assert not bands["series"].eq("day_ahead_plan").any(), (
        "event_response_bands must carry no day_ahead_plan series without the dispatch"
    )


def _validate_event_response_dispatch(result, world_slot: pd.DataFrame) -> None:
    """Contract §7.4: the ``day_ahead_plan`` series, ordered and checked for the national scope."""

    bands = result.event_response_bands
    block = bands.loc[bands["series"].eq("day_ahead_plan")]
    if block.empty:
        return
    assert (block["metric"] == "home_import_kw").all(), "day_ahead_plan series metric"
    _check_ordered_quantiles(block, "p10", "p50", "p90", "event_response_bands day_ahead_plan")
    national = block.loc[block["scope"].eq("national")]
    if national.empty:
        return
    world_count = int(result.world_count)
    day_ahead_kw = _matrix(world_slot, "day_ahead_plan_kwh", world_count) / 0.5
    for event_id, rows in national.groupby("event_id"):
        rows = rows.sort_values("slot_index")
        values = day_ahead_kw[:, rows["slot_index"].to_numpy()]
        stats = _band_statistics_from(values)
        for stat in ("mean", "p10", "p50", "p90"):
            _close(
                rows[stat],
                stats[stat],
                f"event_response_bands day_ahead_plan (event {event_id}) {stat}",
            )
