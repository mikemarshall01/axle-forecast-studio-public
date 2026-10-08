"""Calibration backtest (trading contract v1 §10.3, lane J3) on SYNTHETIC world-slot frames.

J1a/J1b are not merged, so the frames come either from J2's synthetic per-EV
fixture run through ``availability.build_frames`` or from
``independent_world_slot`` (independent gamma draws with known quantiles).
Every figure is illustrative.
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest
from fixtures.availability_backtest_contract import (
    LEVELS,
    independent_world_slot,
    validate_availability_backtest,
)
from fixtures.availability_contract import fixture_totals, make_per_ev_fixture

from axle_studio.model import availability
from axle_studio.model.availability_backtest import (
    availability_backtest,
    availability_backtest_summary,
    availability_reliability,
    leave_one_out_quantiles,
)


def _frames(world_slot):
    backtest = availability_backtest(world_slot)
    reliability = availability_reliability(backtest)
    return types.SimpleNamespace(
        availability_world_slot=world_slot,
        availability_backtest=backtest,
        availability_reliability=reliability,
        availability_backtest_summary=availability_backtest_summary(reliability),
    )


def _rows(frame, horizon):
    return frame[frame["horizon"] == horizon]


# --------------------------------------------------------------------------
# Leave-one-week-out quantile
# --------------------------------------------------------------------------


def test_leave_one_out_quantiles_equal_numpy_quantile_of_the_other_worlds():
    rng = np.random.default_rng(3)
    values = rng.integers(0, 6, (9, 5, 3)).astype(float)  # many ties
    values[:, 4, 2] = np.nan  # a window past the study end: NaN in every world
    values[2, 3, 1] = np.nan  # NaN in one world only
    result = leave_one_out_quantiles(values)
    for w in range(values.shape[0]):
        expected = np.quantile(np.delete(values, w, axis=0), LEVELS, axis=0, method="linear")
        np.testing.assert_allclose(result[w], np.moveaxis(expected, 0, -1), rtol=0, atol=1e-12)


def test_leave_one_out_quantiles_need_two_worlds():
    assert np.isnan(leave_one_out_quantiles(np.ones((1, 4)))).all()
    two = leave_one_out_quantiles(np.array([[1.0], [3.0]]))
    np.testing.assert_array_equal(two[0, 0], np.full(len(LEVELS), 3.0))
    np.testing.assert_array_equal(two[1, 0], np.full(len(LEVELS), 1.0))


# --------------------------------------------------------------------------
# §10.3 tests
# --------------------------------------------------------------------------


def test_day_ahead_coverage_on_independent_draws_is_within_three_binomial_standard_errors():
    """Every world's deliverable at a slot is an independent draw from one gamma distribution.

    Leave-one-out with ``W − 1`` others hits in ``⌊h⌋ + 1`` or ``⌊h⌋ + 2`` of
    the ``W`` worlds at each slot (``h = α (W − 2)``), so pooled coverage is
    ``(h + 1) ÷ W`` up to ``1 ÷ W``: a finite-sample offset of about
    ``(1 − 2α) ÷ W`` that the per-half-hour rows' binomial error covers.
    """

    world_count = 100
    result = _frames(independent_world_slot(world_count)[0])
    rows = _rows(result.availability_backtest, "day_ahead")
    assert (rows["sample_count"] == 7 * world_count).all()
    n = rows["sample_count"].to_numpy()
    alpha = rows["level"].to_numpy()
    three_se = 3 * np.sqrt(alpha * (1 - alpha) / n)
    assert (np.abs(rows["coverage"].to_numpy() - alpha) <= three_se).all()

    pooled = _rows(result.availability_reliability, "day_ahead")
    h = pooled["level"].to_numpy() * (world_count - 2)
    low = (np.floor(h) + 1) / world_count
    assert (pooled["coverage"].to_numpy() >= low - 1e-12).all()
    assert (pooled["coverage"].to_numpy() <= low + 1 / world_count + 1e-12).all()


def test_known_only_night_has_full_coverage_and_zero_pinball():
    """The §10.3 degenerate case: every EV on the driveway at 17:00, r = 1, π = 0, none late.

    Built on J2's fixture with every session plugged in at 17:00 and no clock
    spread, so the late part is 0 and the known part is certain.  J2 reads
    the certain value off a 512-bin grid (one bin of reporting precision,
    §10.2d); the test first checks that and then scores quantiles equal to
    the realised value, as the contract states the case.
    """

    fixture = make_per_ev_fixture(
        world_count=6,
        vehicle_count=40,
        response_rate=(1.0, 1.0, 1.0, 1.0),
        outage_probability=(0.0, 0.0, 0.0, 0.0),
        clock_spread_slots=0,
    )
    draws = dict(fixture.draws)
    draws["plug_offset"] = np.full_like(draws["plug_offset"], -2)  # 18:00 − 1 h = 17:00
    fixture = dataclasses.replace(fixture, draws=draws)
    world_slot = availability.build_frames(fixture_totals(fixture), fixture.study_slots)[
        "availability_world_slot"
    ]
    assert (world_slot["deliverable_late_kw"].fillna(0.0) == 0).all()
    realised = world_slot["deliverable_kw"].to_numpy()
    # J2's grid top at a slot is the largest potential over worlds (the late part is 0 here).
    x_max = world_slot.groupby(["slot_index", "direction", "duration_hours"])[
        "intraday_potential_kw"
    ].transform("max")
    for column in [c for c in world_slot if c.startswith("intraday_") and "_p" in c]:
        if column.startswith("intraday_late_"):
            continue
        forecast = world_slot[column].to_numpy()
        finite = np.isfinite(forecast)
        bin_kw = x_max.to_numpy()[finite] / availability.INTRADAY_GRID_BINS
        assert (np.abs(forecast[finite] - realised[finite]) <= bin_kw + 1e-9).all(), column
        world_slot[column] = np.where(finite, realised, np.nan)

    result = _frames(world_slot)
    validate_availability_backtest(result)
    for horizon in ("intraday", "intraday_known"):
        rows = _rows(result.availability_reliability, horizon)
        scored = rows[rows["sample_count"] > 0]
        assert scored["sample_count"].sum() > 0
        assert (scored["coverage"] == 1.0).all()
        assert (scored["pinball_loss_kw"] == 0.0).all()


def test_halved_quantiles_under_cover_and_score_worse():
    world_slot = independent_world_slot(200)[0]
    calibrated = _frames(world_slot).availability_reliability
    biased_slot = world_slot.copy()
    for column in [c for c in biased_slot if c.startswith("intraday_") and c.endswith("_kw")]:
        biased_slot[column] = 0.5 * biased_slot[column]
    biased = _frames(biased_slot)
    for horizon in ("intraday", "intraday_known"):
        good = _rows(calibrated, horizon)
        bad = _rows(biased.availability_reliability, horizon)
        assert (bad["coverage"].to_numpy() < bad["level"].to_numpy()).all()
        assert (bad["pinball_loss_kw"].to_numpy() > good["pinball_loss_kw"].to_numpy()).all()
        # The exact quantiles are calibrated: every pooled row within 4 binomial SEs.
        n = good["sample_count"].to_numpy()
        alpha = good["level"].to_numpy()
        assert (np.abs(good["coverage"] - alpha) <= 4 * np.sqrt(alpha * (1 - alpha) / n)).all()


def test_point_mass_puts_the_level_between_strict_and_weak_coverage():
    """30 % of world-slots deliver exactly 0 kW (lead ruling on J3 Q2).

    The calibrated forecast is 0 at every level up to 0.3, so ``target ≤ q̂``
    hits on every 0 kW slot (weak coverage about 0.3 at α = 0.05) and
    ``target < q̂`` never does (strict 0 where every forecast is 0): the level
    lies between the two, not at either.  Above the point mass both rates are about α.
    """

    world_count = 200
    result = _frames(independent_world_slot(world_count, zero_share=0.3)[0])
    validate_availability_backtest(result)
    reliability = result.availability_reliability
    for horizon in ("day_ahead", "intraday", "intraday_known"):
        rows = _rows(reliability, horizon)
        alpha = rows["level"].to_numpy()
        # Three binomial SEs, plus the leave-one-out offset (1 − 2α) ÷ W for day-ahead.
        slack = 3 * np.sqrt(alpha * (1 - alpha) / rows["sample_count"].to_numpy())
        slack = slack + (2 / world_count if horizon == "day_ahead" else 0.0)
        assert (rows["coverage_strict"].to_numpy() <= alpha + slack).all()
        assert (rows["coverage"].to_numpy() >= alpha - slack).all()
        # Well inside the point mass (α ≤ 0.1) every forecast is 0: no strict hit at all.
        low = rows[rows["level"] <= 0.1]
        assert (low["coverage"] > 0.28).all()
        assert (low["coverage_strict"] == 0).all()


def test_degenerate_and_nan_world_slots_are_left_out():
    world_slot, slots = independent_world_slot(20)
    cells = len(availability.CELLS)
    slot_rows = np.tile(np.repeat(np.arange(len(slots)), cells), 20)
    cell_rows = np.tile(np.arange(cells), 20 * len(slots))
    labels = slots["local_time_label"].to_numpy()[slot_rows]
    forecast_columns = [c for c in world_slot if c.startswith("intraday_")]

    # 12:00-13:30: nothing to forecast (target and every quantile 0) in every world.
    idle = np.isin(labels, ["12:00", "12:30", "13:00", "13:30"])
    world_slot.loc[idle, ["deliverable_kw", "deliverable_known_kw", *forecast_columns]] = 0.0
    # 14:00: target 0 in half the worlds but a positive forecast: a real miss, not degenerate.
    missed = (labels == "14:00") & (world_slot["world_id"].to_numpy() % 2 == 0)
    world_slot.loc[missed, ["deliverable_kw", "deliverable_known_kw"]] = 0.0
    # Before 17:00 on intraday: no forecast (NaN); day-ahead is still scored.
    world_slot.loc[labels == "16:30", forecast_columns] = np.nan
    # The last slot of the study, 4-h cells: the window runs past the end.
    past_end = (slot_rows == len(slots) - 1) & np.isin(cell_rows, [3, 7])
    world_slot.loc[past_end, "deliverable_kw"] = np.nan

    result = _frames(world_slot)
    validate_availability_backtest(result)
    backtest = result.availability_backtest
    for label in ("12:00", "12:30", "13:00", "13:30"):
        rows = backtest[backtest["local_half_hour"] == label]
        assert (rows["sample_count"] == 0).all()
        assert rows[["coverage", "pinball_loss_kw", "mean_forecast_kw"]].isna().all().all()
    missed_rows = backtest[(backtest["local_half_hour"] == "14:00")]
    assert (missed_rows["sample_count"] == 7 * 20).all()
    at_1630 = backtest[backtest["local_half_hour"] == "16:30"]
    assert (_rows(at_1630, "day_ahead")["sample_count"] == 7 * 20).all()
    assert (at_1630[at_1630["horizon"] != "day_ahead"]["sample_count"] == 0).all()
    # The last study slot is 11:30 on the final morning; its 4-h rows lose that night.
    last = backtest[(backtest["local_half_hour"] == "11:30") & (backtest["horizon"] == "day_ahead")]
    four = last["duration_hours"] == 4.0
    assert (last.loc[four, "sample_count"] == 6 * 20).all()
    assert (last.loc[~four, "sample_count"] == 7 * 20).all()


def test_reliability_is_the_sample_weighted_pooling_of_the_backtest():
    world_slot = independent_world_slot(12)[0]
    world_slot.loc[world_slot.index % 5 == 0, "deliverable_kw"] = 0.0
    result = _frames(world_slot)
    backtest, reliability = result.availability_backtest, result.availability_reliability
    keys = ["horizon", "direction", "duration_hours", "level"]
    for (key, group), row in zip(
        backtest.groupby(keys, sort=False), reliability.itertuples(index=False), strict=True
    ):
        assert key == (row.horizon, row.direction, row.duration_hours, row.level)
        weight = group["sample_count"].to_numpy()
        assert row.sample_count == weight.sum()
        kept = weight > 0
        for column in (
            "coverage",
            "coverage_strict",
            "pinball_loss_kw",
            "mean_forecast_kw",
            "mean_realised_kw",
        ):
            expected = np.average(group[column].to_numpy()[kept], weights=weight[kept])
            assert getattr(row, column) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_summary_equals_its_recomputation():
    result = _frames(independent_world_slot(12)[0])
    reliability, summary = result.availability_reliability, result.availability_backtest_summary
    assert len(summary) == 24
    for row in summary.itertuples(index=False):
        rows = reliability[
            (reliability["horizon"] == row.horizon)
            & (reliability["direction"] == row.direction)
            & (reliability["duration_hours"] == row.duration_hours)
        ].set_index("level")
        assert row.coverage_p05 == rows.at[0.05, "coverage"]
        assert row.coverage_p10 == rows.at[0.1, "coverage"]
        assert row.coverage_p50 == rows.at[0.5, "coverage"]
        assert row.coverage_p90 == rows.at[0.9, "coverage"]
        assert row.coverage_strict_p05 == rows.at[0.05, "coverage_strict"]
        assert row.coverage_strict_p10 == rows.at[0.1, "coverage_strict"]
        assert row.coverage_strict_p50 == rows.at[0.5, "coverage_strict"]
        assert row.coverage_strict_p90 == rows.at[0.9, "coverage_strict"]
        assert row.pinball_mean_kw == pytest.approx(rows["pinball_loss_kw"].mean())
        assert row.sharpness_p10_p90_kw == pytest.approx(
            rows.at[0.9, "mean_forecast_kw"] - rows.at[0.1, "mean_forecast_kw"]
        )
        assert row.sample_count == rows["sample_count"].max()
    validate_availability_backtest(result)  # also recomputes sharpness from the world-slots


def test_intraday_known_rows_target_the_known_part():
    result = _frames(independent_world_slot(12)[0])
    reliability = result.availability_reliability
    whole = _rows(reliability, "intraday").reset_index(drop=True)
    known = _rows(reliability, "intraday_known").reset_index(drop=True)
    # The fixture's known part is 60 % of the whole-fleet deliverable, and so are its quantiles.
    np.testing.assert_allclose(known["mean_realised_kw"], 0.6 * whole["mean_realised_kw"])
    np.testing.assert_allclose(known["coverage"], whole["coverage"])
    # Every world-slot of this fixture is kept, so the pooled mean target is the column mean.
    world_slot = result.availability_world_slot
    first_cell = (world_slot["direction"] == "turn_down") & (world_slot["duration_hours"] == 0.5)
    assert known["mean_realised_kw"].iloc[0] == pytest.approx(
        world_slot.loc[first_cell, "deliverable_known_kw"].mean(), rel=1e-12
    )


# --------------------------------------------------------------------------
# Shape, order and the validator on J2's pipeline
# --------------------------------------------------------------------------


def test_frames_on_the_j2_fixture_pass_the_validator():
    fixture = make_per_ev_fixture(world_count=10, vehicle_count=80)
    world_slot = availability.build_frames(fixture_totals(fixture), fixture.study_slots)[
        "availability_world_slot"
    ]
    result = _frames(world_slot)
    validate_availability_backtest(result)
    backtest = result.availability_backtest
    assert len(backtest) == 8064
    assert len(result.availability_reliability) == 168
    assert backtest["local_half_hour"].iloc[:: len(LEVELS)].iloc[:48].tolist()[:3] == [
        "12:00",
        "12:30",
        "13:00",
    ]
    assert (backtest["evidence_kind"] == "illustrative_synthetic").all()
    # J2 stores all seven known-part levels (lead ruling on J3 Q1), so each is scored.
    known = _rows(result.availability_reliability, "intraday_known")
    assert (known.groupby("level")["sample_count"].sum() > 0).all()
    known_summary = _rows(result.availability_backtest_summary, "intraday_known")
    assert known_summary["pinball_mean_kw"].notna().any()


def test_one_world_has_nothing_to_score():
    fixture = make_per_ev_fixture(world_count=1, vehicle_count=20)
    world_slot = availability.build_frames(fixture_totals(fixture), fixture.study_slots)[
        "availability_world_slot"
    ]
    result = _frames(world_slot)
    validate_availability_backtest(result)
    assert (result.availability_backtest["sample_count"] == 0).all()
    assert result.availability_backtest_summary["coverage_p10"].isna().all()
