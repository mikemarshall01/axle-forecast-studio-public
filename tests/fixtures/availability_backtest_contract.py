"""Calibration backtest (lane J3) frame specs, validator and SYNTHETIC world-slot frames.

What this owns (trading contract v1 §10.3, §10.9): the column specs of
``availability_backtest``, ``availability_reliability`` and
``availability_backtest_summary``; ``validate_availability_backtest``, which
recomputes all three from ``availability_world_slot`` by a plain per-world
loop (``numpy.quantile`` of the other worlds, pandas group-bys) independent of
the model's vectorised code; and ``independent_world_slot``, a SYNTHETIC
stand-in for ``availability_world_slot`` whose deliverable at each slot is an
independent draw from one known gamma distribution in every world.
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd
from fixtures.result_contract import _check_dtypes, _check_unique
from scipy import stats

from axle_studio.model import availability, availability_backtest
from axle_studio.model.summaries import build_study_slots

LEVELS = availability.INTRADAY_LEVELS
TOL = 1e-9

AVAILABILITY_BACKTEST_COLUMNS = {
    "horizon": "str",
    "direction": "str",
    "duration_hours": "float",
    "local_half_hour": "str",
    "profile_order": "int",
    "level": "float",
    "sample_count": "int",
    "coverage": "float",
    "coverage_strict": "float",
    "pinball_loss_kw": "float",
    "mean_forecast_kw": "float",
    "mean_realised_kw": "float",
    "evidence_kind": "str",
}
AVAILABILITY_RELIABILITY_COLUMNS = {
    "horizon": "str",
    "direction": "str",
    "duration_hours": "float",
    "level": "float",
    "sample_count": "int",
    "coverage": "float",
    "coverage_strict": "float",
    "pinball_loss_kw": "float",
    "mean_forecast_kw": "float",
    "mean_realised_kw": "float",
    "evidence_kind": "str",
}
AVAILABILITY_BACKTEST_SUMMARY_COLUMNS = {
    "horizon": "str",
    "direction": "str",
    "duration_hours": "float",
    **{
        f"coverage{kind}_{p}": "float"
        for p in ("p05", "p10", "p50", "p90")
        for kind in ("", "_strict")
    },
    "pinball_mean_kw": "float",
    "sharpness_p10_p90_kw": "float",
    "sample_count": "int",
    "evidence_kind": "str",
}


_MEAN_COLUMNS = (
    "coverage",
    "coverage_strict",
    "pinball_loss_kw",
    "mean_forecast_kw",
    "mean_realised_kw",
)


def _name(level: float) -> str:
    return f"p{round(100 * level):02d}"


# --------------------------------------------------------------------------
# SYNTHETIC independent-draw world-slot frame
# --------------------------------------------------------------------------


def independent_world_slot(
    world_count: int,
    *,
    seed: int = 11,
    shape: float = 4.0,
    scale_kw: float = 250.0,
    zero_share: float = 0.0,
):
    """SYNTHETIC ``availability_world_slot`` subset: independent gamma draws per world.

    Every (world, slot, cell) ``deliverable_kw`` is an independent draw: 0 kW
    with probability ``zero_share`` (a point mass, like a night with nothing
    to flex), else Gamma(``shape``, ``scale_kw``); ``deliverable_known_kw``
    is 60 % of it.
    The intraday columns hold that distribution's exact quantiles (a
    perfectly calibrated forecast) and the known columns 60 % of them, at all
    seven levels.  Only the columns the backtest reads are present.  Returns
    ``(frame, study_slots)``.  All values are illustrative.
    """

    rng = np.random.default_rng(seed)
    slots = build_study_slots(date(2026, 10, 5), 7)
    cells = len(availability.CELLS)
    rows = world_count * len(slots) * cells
    deliverable = np.where(rng.random(rows) < zero_share, 0.0, rng.gamma(shape, scale_kw, rows))
    frame = pd.DataFrame(
        {
            "world_id": np.repeat(np.arange(world_count, dtype=np.int64), len(slots) * cells),
            "slot_index": np.tile(np.repeat(slots["slot_index"].to_numpy(), cells), world_count),
            "interval_start_london": slots["interval_start_london"]
            .take(np.tile(np.repeat(np.arange(len(slots)), cells), world_count))
            .reset_index(drop=True),
            "direction": np.tile([d for d, _ in availability.CELLS], world_count * len(slots)),
            "duration_hours": np.tile(
                [h for _, h in availability.CELLS], world_count * len(slots)
            ).astype(float),
            "deliverable_kw": deliverable,
            "deliverable_known_kw": 0.6 * deliverable,
        }
    )
    for level in LEVELS:
        # Quantile of the mixture: 0 up to the point mass, then the gamma part's.
        above = max(level - zero_share, 0.0) / (1.0 - zero_share)
        exact = stats.gamma.ppf(above, shape, scale=scale_kw)
        frame[f"intraday_{_name(level)}_kw"] = exact
        frame[f"intraday_known_{_name(level)}_kw"] = 0.6 * exact
    return frame, slots


# --------------------------------------------------------------------------
# Reference recomputation (§10.3) and validator (§10.9)
# --------------------------------------------------------------------------


def reference_scores(world_slot: pd.DataFrame) -> pd.DataFrame:
    """Per kept (horizon, world, slot, cell, level) scores, recomputed plainly (§10.3).

    The day-ahead forecast is ``numpy.quantile`` of ``np.delete`` of world
    ``w``, one call per world.  Columns: horizon, direction, duration_hours,
    local_half_hour, level, hit, strict_hit, pinball, forecast, realised, world_id,
    slot_index.
    """

    world_count = int(world_slot["world_id"].max()) + 1
    cells = len(availability.CELLS)
    slot_count = len(world_slot) // (world_count * cells)
    shape = (world_count, slot_count, cells)

    def cube(column):
        return world_slot[column].to_numpy(dtype=float).reshape(shape)

    labels = (
        world_slot["interval_start_london"].iloc[::cells].iloc[:slot_count].dt.strftime("%H:%M")
    ).to_numpy()
    deliverable = cube("deliverable_kw")
    day_ahead = np.full((*shape, len(LEVELS)), np.nan)
    if world_count > 1:
        for w in range(world_count):
            others = np.delete(deliverable, w, axis=0)
            day_ahead[w] = np.moveaxis(np.quantile(others, LEVELS, axis=0, method="linear"), 0, -1)
    horizons = {"day_ahead": (deliverable, day_ahead, list(range(len(LEVELS))))}
    for horizon, target in (
        ("intraday", deliverable),
        ("intraday_known", cube("deliverable_known_kw")),
    ):
        forecast = np.full((*shape, len(LEVELS)), np.nan)
        present = []
        for k, level in enumerate(LEVELS):
            column = f"{horizon}_{_name(level)}_kw"
            if column in world_slot:
                forecast[..., k] = cube(column)
                present.append(k)
        horizons[horizon] = (target, forecast, present)

    world, slot, cell = np.indices(shape)
    directions = np.array([d for d, _ in availability.CELLS], dtype=object)
    hours = np.array([h for _, h in availability.CELLS])
    frames = []
    for horizon, (target, forecast, present) in horizons.items():
        stored = forecast[..., present]
        keep = np.isfinite(target) & np.isfinite(stored).all(axis=-1) & bool(present)
        keep &= ~((target == 0) & (stored == 0).all(axis=-1))
        for k in present:
            y = target[keep]
            q = forecast[..., k][keep]
            level = LEVELS[k]
            frames.append(
                pd.DataFrame(
                    {
                        "horizon": horizon,
                        "direction": directions[cell[keep]],
                        "duration_hours": hours[cell[keep]],
                        "local_half_hour": labels[slot[keep]],
                        "level": level,
                        "hit": (y <= q).astype(float),
                        "strict_hit": (y < q).astype(float),
                        "pinball": (y - q) * (level - (y < q)),
                        "forecast": q,
                        "realised": y,
                        "world_id": world[keep],
                        "slot_index": slot[keep],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _assert_close(actual, expected, what: str) -> None:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert np.array_equal(np.isnan(actual), np.isnan(expected)), f"{what}: NaN pattern differs"
    both = ~np.isnan(actual)
    scale = np.maximum(1.0, np.abs(expected[both]))
    assert np.all(np.abs(actual[both] - expected[both]) <= TOL * scale), f"{what} differs"


def validate_availability_backtest(result) -> None:
    """Assert the §10.9 rules on the J3 frames of ``result``; raise AssertionError naming the rule.

    Duck-typed: reads ``availability_world_slot``, ``availability_backtest``,
    ``availability_reliability`` and ``availability_backtest_summary``.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _validate(result)


def _validate(result) -> None:
    backtest = result.availability_backtest
    reliability = result.availability_reliability
    summary = result.availability_backtest_summary
    _check_dtypes(backtest, AVAILABILITY_BACKTEST_COLUMNS, "availability_backtest")
    _check_dtypes(reliability, AVAILABILITY_RELIABILITY_COLUMNS, "availability_reliability")
    _check_dtypes(summary, AVAILABILITY_BACKTEST_SUMMARY_COLUMNS, "availability_backtest_summary")
    cell_keys = ["horizon", "direction", "duration_hours"]
    _check_unique(backtest, [*cell_keys, "local_half_hour", "level"], "availability_backtest")
    _check_unique(reliability, [*cell_keys, "level"], "availability_reliability")
    _check_unique(summary, cell_keys, "availability_backtest_summary")
    horizons = len(availability_backtest.HORIZONS)
    cells = len(availability.CELLS)
    assert len(backtest) == horizons * cells * 48 * len(LEVELS), "availability_backtest rows"
    assert len(reliability) == horizons * cells * len(LEVELS), "availability_reliability rows"
    assert len(summary) == horizons * cells, "availability_backtest_summary rows"
    assert list(backtest["horizon"].unique()) == list(availability_backtest.HORIZONS)
    profile = backtest["profile_order"].to_numpy().reshape(-1, 48, len(LEVELS))
    assert (profile[:, :, 0] == np.arange(48)).all(), "half-hours must run noon to noon"
    assert backtest["local_half_hour"].iloc[0] == "12:00", "profile_order 0 is 12:00"

    for frame, name in (
        (backtest, "availability_backtest"),
        (reliability, "availability_reliability"),
    ):
        for column in ("coverage", "coverage_strict"):
            share = frame[column].dropna()
            assert ((share >= 0) & (share <= 1)).all(), f"{name}: {column} outside [0, 1]"
        both = frame[["coverage", "coverage_strict"]].dropna()
        assert (both["coverage_strict"] <= both["coverage"]).all(), f"{name}: strict > weak"
        assert (frame["pinball_loss_kw"].dropna() >= -TOL).all(), f"{name}: pinball < 0"
        empty = frame["sample_count"] == 0
        assert frame.loc[empty, ["coverage", "pinball_loss_kw"]].isna().all().all(), (
            f"{name}: rows with no samples must be NaN"
        )
        assert frame.loc[~empty, ["coverage", "pinball_loss_kw"]].notna().all().all(), (
            f"{name}: rows with samples must be finite"
        )

    # Recomputation from availability_world_slot (§10.9).
    scores = reference_scores(result.availability_world_slot)
    grouped = scores.groupby([*cell_keys, "local_half_hour", "level"]).agg(
        sample_count=("hit", "size"),
        coverage=("hit", "mean"),
        coverage_strict=("strict_hit", "mean"),
        pinball_loss_kw=("pinball", "mean"),
        mean_forecast_kw=("forecast", "mean"),
        mean_realised_kw=("realised", "mean"),
    )
    expected = backtest[[*cell_keys, "local_half_hour", "level"]].join(
        grouped, on=[*cell_keys, "local_half_hour", "level"]
    )
    assert (
        backtest["sample_count"].to_numpy() == expected["sample_count"].fillna(0).to_numpy()
    ).all(), "availability_backtest sample_count differs from recomputation"
    for column in _MEAN_COLUMNS:
        _assert_close(backtest[column], expected[column], f"availability_backtest.{column}")

    pooled = scores.groupby([*cell_keys, "level"]).agg(
        sample_count=("hit", "size"),
        coverage=("hit", "mean"),
        coverage_strict=("strict_hit", "mean"),
        pinball_loss_kw=("pinball", "mean"),
        mean_forecast_kw=("forecast", "mean"),
        mean_realised_kw=("realised", "mean"),
    )
    expected = reliability[[*cell_keys, "level"]].join(pooled, on=[*cell_keys, "level"])
    assert (
        reliability["sample_count"].to_numpy() == expected["sample_count"].fillna(0).to_numpy()
    ).all(), "availability_reliability sample_count differs from recomputation"
    by_cell = backtest.groupby([*cell_keys, "level"], sort=False)["sample_count"].sum()
    assert (by_cell.to_numpy() == reliability["sample_count"].to_numpy()).all(), (
        "availability_reliability sample_count must be the sum of the half-hour rows"
    )
    for column in _MEAN_COLUMNS:
        _assert_close(reliability[column], expected[column], f"availability_reliability.{column}")

    # Summary: the matching level rows, the seven-level pinball mean and the P10-P90 width.
    count = scores.drop_duplicates([*cell_keys, "world_id", "slot_index"]).groupby(cell_keys).size()
    width = pd.Series(dtype=float)
    if len(scores):
        width = (
            scores.pivot_table(
                index=[*cell_keys, "world_id", "slot_index"], columns="level", values="forecast"
            )
            .pipe(lambda table: table[0.9] - table[0.1])
            .groupby(level=cell_keys)
            .mean()
        )
    for row in summary.itertuples(index=False):
        key = (row.horizon, row.direction, row.duration_hours)
        rows = reliability.set_index(cell_keys).loc[key].set_index("level")
        for level in (0.05, 0.1, 0.5, 0.9):
            for kind in ("", "_strict"):
                _assert_close(
                    getattr(row, f"coverage{kind}_{_name(level)}"),
                    rows.at[level, f"coverage{kind}"],
                    f"summary coverage{kind}_{_name(level)} {key}",
                )
        _assert_close(
            row.pinball_mean_kw,
            rows["pinball_loss_kw"].mean(skipna=False),
            f"summary pinball_mean_kw {key}",
        )
        _assert_close(
            row.sharpness_p10_p90_kw,
            width.get(key, np.nan),
            f"summary sharpness_p10_p90_kw {key}",
        )
        assert row.sample_count == count.get(key, 0), f"summary sample_count {key}"
