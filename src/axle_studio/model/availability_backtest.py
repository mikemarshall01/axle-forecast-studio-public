"""Calibration backtest of the firm-MW forecasts, leave one week out (trading contract v1 §10.3).

What this owns (lane J3): how well the availability forecasts of §10.2 are
calibrated, scored on the run's own simulated weeks.  Each simulated week in
turn is held out and treated as "what happened"; the forecast for it uses
nothing from that week.

Three horizons, each a set of quantile forecasts ``q̂(α)`` at the seven
levels ``α`` of ``availability.INTRADAY_LEVELS`` and a target:

- ``day_ahead``: ``numpy.quantile`` (linear) of ``deliverable_kw`` over the
  other ``W − 1`` weeks at the same slot, against this week's
  ``deliverable_kw``.  The other weeks stand in for history.
- ``intraday``: the 17:00 whole-night quantiles J2 already wrote to
  ``availability_world_slot`` (``intraday_pXX_kw``), against
  ``deliverable_kw``.  Their late part already leaves this week out (§10.2d).
- ``intraday_known``: the known (driveway) part's quantiles
  (``intraday_known_pXX_kw``), against ``deliverable_known_kw``.

Scores per kept world-slot and level: a weak hit when ``target ≤ q̂``, a
strict hit when ``target < q̂``, and the pinball loss
``(target − q̂)(α − 1[target < q̂])`` in kW.  ``coverage`` (the contract's
share of weak hits) should be about ``α``; lower pinball is better.  The
deliverable has point masses (0 kW slots, near-certain known parts), where
ties are common, so a calibrated forecast has ``coverage_strict ≤ α ≤
coverage`` rather than ``coverage ≈ α``.  Both rates are reported instead of
a randomised hit on ties (lead ruling on J3 Q2), so the result stays
deterministic.

- ``availability_backtest`` reads ``availability_world_slot`` and returns the
  per-half-hour frame (horizon, direction, duration, London half-hour, level).
- ``availability_reliability`` pools that frame over half-hours, weighting
  each row by its ``sample_count``.
- ``availability_backtest_summary`` reads the reliability frame and returns
  one headline row per (horizon, direction, duration).

Every figure is illustrative and synthetic: the "history" is the run's own
simulated weeks, not metered data.  Nothing here draws random numbers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from axle_studio.model.availability import CELLS, EVIDENCE_KIND, INTRADAY_LEVELS

HORIZONS = ("day_ahead", "intraday", "intraday_known")
_TARGET = {
    "day_ahead": "deliverable_kw",
    "intraday": "deliverable_kw",
    "intraday_known": "deliverable_known_kw",
}
_HALF_HOURS = 48


def _level_name(level: float) -> str:
    return f"p{round(100 * level):02d}"


def _world_slot_cube(world_slot: pd.DataFrame, column: str, world_count: int) -> np.ndarray:
    """(world, slot, cell) array of one column; J2 writes rows world-major, then slot, then cell."""

    return world_slot[column].to_numpy(dtype=float).reshape(world_count, -1, len(CELLS))


def leave_one_out_quantiles(values: np.ndarray, levels=INTRADAY_LEVELS) -> np.ndarray:
    """Day-ahead forecast of every world from the other worlds, shape (world, ..., level).

    ``values`` is (world, ...) kW.  For world ``w`` and level ``α`` the result
    equals ``numpy.quantile(np.delete(values, w, axis=0), α, axis=0,
    method="linear")``: NaN when any other world is NaN at that position, and
    NaN everywhere with fewer than two worlds.

    Sorting once and skipping world ``w``'s own rank avoids ``W`` separate
    quantile calls: removing one element shifts every sorted position at or
    above its rank down by one, and ties do not matter because removing any
    of equal values leaves the same multiset.

    On exchangeable weeks this forecast covers ``(α (W − 2) + 1) ÷ W`` on
    average, not exactly ``α``: the empirical quantile of ``W − 1`` values
    has a finite-sample offset of about ``(1 − 2α) ÷ W`` (0.009 at α = 0.05
    and 100 worlds).  It is the contract's estimator (§10.3), reported as is.
    """

    world_count = values.shape[0]
    out = np.full((*values.shape, len(levels)), np.nan)
    if world_count < 2:
        return out
    order = np.argsort(values, axis=0, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(world_count).reshape(-1, *[1] * (values.ndim - 1)), 0)
    ordered = np.take_along_axis(values, order, axis=0)
    last = world_count - 2  # highest index into the W − 1 remaining values
    for k, level in enumerate(levels):
        position = level * last  # numpy "linear": virtual index α (n − 1), n = W − 1
        low = int(np.floor(position))
        high = min(low + 1, last)
        fraction = position - low
        # Index j of the remaining values is sorted index j, or j + 1 at or above w's rank.
        below = np.take_along_axis(ordered, low + (low >= ranks), axis=0)
        above = np.take_along_axis(ordered, high + (high >= ranks), axis=0)
        out[..., k] = below + fraction * (above - below)
    # numpy.quantile propagates NaN: world w's forecast is NaN when another world is NaN.
    nan = np.isnan(values)
    others_nan = nan.sum(axis=0) - nan > 0
    out[others_nan] = np.nan
    return out


def _horizon_forecasts(
    world_slot: pd.DataFrame, horizon: str, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Quantile forecasts of one horizon, (world, slot, cell, level), and which levels are stored.

    A level whose ``intraday_known_pXX_kw`` column is not in the frame is all
    NaN and marked not stored, so its rows have ``sample_count`` 0 (J2
    stores all seven known-part levels, so this is only a guard).
    """

    stored = np.ones(len(INTRADAY_LEVELS), dtype=bool)
    if horizon == "day_ahead":
        return leave_one_out_quantiles(target), stored
    world_count = target.shape[0]
    forecast = np.full((*target.shape, len(INTRADAY_LEVELS)), np.nan)
    for k, level in enumerate(INTRADAY_LEVELS):
        column = f"{horizon}_{_level_name(level)}_kw"  # intraday_pXX_kw, intraday_known_pXX_kw
        stored[k] = column in world_slot
        if stored[k]:
            forecast[..., k] = _world_slot_cube(world_slot, column, world_count)
    return forecast, stored


def _kept(target: np.ndarray, forecast: np.ndarray, stored: np.ndarray) -> np.ndarray:
    """(world, slot, cell) world-slots that enter the scores (§10.3).

    Left out when the target or any stored forecast level is NaN (a window
    past the study end, an intraday slot before 17:00, a day-ahead forecast
    with one world), or when the target and every stored forecast quantile
    are 0 (nothing to forecast, e.g. a daytime slot with no home charging).
    """

    levels = forecast[..., stored]
    finite = np.isfinite(target) & np.all(np.isfinite(levels), axis=-1)
    nothing = (target == 0) & np.all(levels == 0, axis=-1)
    return finite & ~nothing & stored.any()


def _half_hour_keys(world_slot: pd.DataFrame, slot_count: int) -> tuple[np.ndarray, list[str]]:
    """Noon-to-noon profile order (0 = 12:00) of each study slot and the 48 "HH:MM" labels.

    Read from London wall-clock time, so both copies of the repeated autumn
    01:00 and 01:30 enter their label (§10.3).
    """

    london = world_slot["interval_start_london"].iloc[:: len(CELLS)].iloc[:slot_count]
    half_hour = london.dt.hour.to_numpy() * 2 + london.dt.minute.to_numpy() // 30
    profile = (half_hour - 24) % _HALF_HOURS
    labels = [f"{(24 + p) % 48 // 2:02d}:{30 * ((24 + p) % 2):02d}" for p in range(_HALF_HOURS)]
    return profile, labels


def availability_backtest(world_slot: pd.DataFrame) -> pd.DataFrame:
    """The §10.3 ``availability_backtest`` frame from ``availability_world_slot``.

    One row per (horizon, direction, duration_hours, local_half_hour, level)
    in the order day_ahead, intraday, intraday_known; turn_down then turn_up;
    0.5, 1, 2, 4 h; half-hours noon to noon; the seven levels.  Per row:
    ``sample_count`` world-nights kept, ``coverage`` (share with target ≤
    forecast), ``coverage_strict`` (share with target < forecast),
    ``pinball_loss_kw``, ``mean_forecast_kw`` and
    ``mean_realised_kw`` (kW means over those world-nights; NaN when none
    are kept).  8,064 rows.
    """

    world_count = int(world_slot["world_id"].max()) + 1
    slot_count = len(world_slot) // (world_count * len(CELLS))
    profile, labels = _half_hour_keys(world_slot, slot_count)
    # (half-hour, slot) indicator: summing slots into their London half-hour.
    to_half_hour = (profile[np.newaxis, :] == np.arange(_HALF_HOURS)[:, np.newaxis]).astype(float)
    levels = np.asarray(INTRADAY_LEVELS)

    blocks = []
    for horizon in HORIZONS:
        target = _world_slot_cube(world_slot, _TARGET[horizon], world_count)
        forecast, stored = _horizon_forecasts(world_slot, horizon, target)
        kept = _kept(target, forecast, stored)
        counted = kept[..., np.newaxis] & stored
        error = target[..., np.newaxis] - forecast
        with np.errstate(invalid="ignore"):
            # Pinball (quantile) loss: under-forecast costs α per kW, over-forecast 1 − α.
            pinball = error * (levels - (error < 0))
            scores = {
                "hits": error <= 0,
                "strict_hits": error < 0,
                "pinball": pinball,
                "forecast": forecast,
                "realised": np.broadcast_to(target[..., np.newaxis], forecast.shape),
            }
        # World-first: sum over worlds per slot, then slots into their half-hour.
        count = np.einsum("hs,scl->chl", to_half_hour, counted.sum(axis=0))
        sums = {
            name: np.einsum("hs,scl->chl", to_half_hour, np.where(counted, values, 0.0).sum(axis=0))
            for name, values in scores.items()
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = {name: np.where(count > 0, sums[name] / count, np.nan) for name in sums}
        rows = count.size  # cell, half-hour, level in C order: the frame's row order
        blocks.append(
            pd.DataFrame(
                {
                    "horizon": np.full(rows, horizon, dtype=object),
                    "direction": np.repeat([d for d, _ in CELLS], _HALF_HOURS * len(levels)).astype(
                        object
                    ),
                    "duration_hours": np.repeat(
                        [h for _, h in CELLS], _HALF_HOURS * len(levels)
                    ).astype(float),
                    "local_half_hour": np.tile(
                        np.repeat(np.array(labels, dtype=object), len(levels)), len(CELLS)
                    ),
                    "profile_order": np.tile(
                        np.repeat(np.arange(_HALF_HOURS, dtype=np.int64), len(levels)), len(CELLS)
                    ),
                    "level": np.tile(levels, len(CELLS) * _HALF_HOURS),
                    "sample_count": count.ravel().astype(np.int64),
                    "coverage": mean["hits"].ravel(),
                    "coverage_strict": mean["strict_hits"].ravel(),
                    "pinball_loss_kw": mean["pinball"].ravel(),
                    "mean_forecast_kw": mean["forecast"].ravel(),
                    "mean_realised_kw": mean["realised"].ravel(),
                    "evidence_kind": np.full(rows, EVIDENCE_KIND, dtype=object),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


_KEYS = ["horizon", "direction", "duration_hours", "level"]
_MEANS = ["coverage", "coverage_strict", "pinball_loss_kw", "mean_forecast_kw", "mean_realised_kw"]


def availability_reliability(backtest: pd.DataFrame) -> pd.DataFrame:
    """The §10.3 ``availability_reliability`` frame: ``backtest`` pooled over half-hours.

    One row per (horizon, direction, duration_hours, level) in the backtest's
    order.  Each mean is the ``sample_count``-weighted mean of the half-hour
    rows, which equals the mean over every kept world-slot; ``sample_count``
    is their sum.  NaN where nothing is kept.
    """

    weight = backtest["sample_count"].to_numpy(dtype=float)
    weighted = backtest[_KEYS].copy()
    weighted["sample_count"] = backtest["sample_count"]
    for column in _MEANS:
        # A row with no samples has NaN means and weight 0; it adds nothing.
        weighted[column] = np.where(weight > 0, backtest[column].to_numpy() * weight, 0.0)
    pooled = weighted.groupby(_KEYS, sort=False, as_index=False).sum()
    count = pooled["sample_count"].to_numpy(dtype=float)
    for column in _MEANS:
        with np.errstate(invalid="ignore", divide="ignore"):
            pooled[column] = np.where(count > 0, pooled[column] / count, np.nan)
    pooled["sample_count"] = pooled["sample_count"].astype(np.int64)
    pooled["evidence_kind"] = EVIDENCE_KIND
    columns = [*_KEYS, "sample_count", *_MEANS, "evidence_kind"]
    return pooled[columns]


def availability_backtest_summary(reliability: pd.DataFrame) -> pd.DataFrame:
    """The §10.3 ``availability_backtest_summary``: one row per (horizon, direction, duration).

    ``coverage_p05`` … ``coverage_p90`` are the pooled coverages at levels
    0.05, 0.1, 0.5 and 0.9, each followed by its ``coverage_strict_pXX``.
    ``pinball_mean_kw`` is the mean of ``pinball_loss_kw`` over the seven
    levels (a discrete CRPS proxy; NaN if any level has no forecast).
    ``sharpness_p10_p90_kw`` is the mean over kept world-slots of
    ``q̂(0.9) − q̂(0.1)``: every level is scored on the same kept
    world-slots, so it is the difference of the two pooled mean forecasts.
    ``sample_count`` is the kept world-slots, the same at every level that
    has a forecast.
    """

    rows = []
    for (horizon, direction, hours), group in reliability.groupby(
        ["horizon", "direction", "duration_hours"], sort=False
    ):
        by_level = group.set_index("level")
        row = {"horizon": horizon, "direction": direction, "duration_hours": hours}
        for level in (0.05, 0.1, 0.5, 0.9):
            row[f"coverage_{_level_name(level)}"] = by_level.at[level, "coverage"]
            row[f"coverage_strict_{_level_name(level)}"] = by_level.at[level, "coverage_strict"]
        row["pinball_mean_kw"] = by_level["pinball_loss_kw"].mean(skipna=False)
        row["sharpness_p10_p90_kw"] = (
            by_level.at[0.9, "mean_forecast_kw"] - by_level.at[0.1, "mean_forecast_kw"]
        )
        row["sample_count"] = int(by_level["sample_count"].max())
        row["evidence_kind"] = EVIDENCE_KIND
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.astype({"duration_hours": float, "sample_count": np.int64})
