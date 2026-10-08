"""The newsvendor day-ahead commitment (trading contract v1 §10.4).

Every number here is SYNTHETIC: small hand-made (world, slot) arrays for the
rule itself, and the tiny SYNTHETIC fleet of ``test_market`` for the run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from model.test_market import _tiny_run

from axle_studio.model import market, summaries


def _inputs(worlds: int = 5, slots: int = 4, seed: int = 3) -> dict[str, np.ndarray]:
    """SYNTHETIC one-night inputs: positive forecasts, volumes and prices."""

    rng = np.random.default_rng(seed)
    return {
        "forecast_kwh": rng.uniform(0.0, 10.0, (worlds, slots)),
        "settled_kwh": rng.uniform(0.0, 10.0, (worlds, slots)),
        "day_ahead_gbp_per_mwh": rng.uniform(20.0, 120.0, (worlds, slots)),
        "imbalance_gbp_per_mwh": rng.uniform(40.0, 200.0, (worlds, slots)),
        "system_short": rng.uniform(size=(worlds, slots)) < 0.5,
        "mask": np.ones(slots, dtype=bool),
    }


def test_matches_numpy_quantile_of_the_other_worlds() -> None:
    inputs = _inputs(worlds=7, slots=5)
    q, alpha, error = market.newsvendor_positions(**inputs)
    errors = inputs["settled_kwh"] - inputs["forecast_kwh"]
    for w in range(7):
        for t in range(5):
            others = np.delete(errors[:, t], w)
            assert error[w, t] == np.quantile(others, alpha[w, t])
    expected = np.maximum(inputs["forecast_kwh"] + error, 0.0)
    np.testing.assert_array_equal(q, expected)


def test_identical_errors_give_forecast_plus_error_at_any_level() -> None:
    inputs = _inputs()
    offset = np.array([1.5, -0.5, 2.0, 0.25])
    inputs["settled_kwh"] = inputs["forecast_kwh"] + offset
    for price in (1.0, 50.0, 500.0):
        inputs["day_ahead_gbp_per_mwh"] = np.full((5, 4), price)
        q, _, error = market.newsvendor_positions(**inputs)
        np.testing.assert_allclose(error, np.broadcast_to(offset, (5, 4)), rtol=0, atol=1e-12)
        np.testing.assert_allclose(q, np.maximum(inputs["forecast_kwh"] + offset, 0.0))


def test_level_zero_takes_the_smallest_and_level_one_the_largest_other_error() -> None:
    inputs = _inputs()
    errors = inputs["settled_kwh"] - inputs["forecast_kwh"]
    smallest = np.array([[np.delete(errors[:, t], w).min() for t in range(4)] for w in range(5)])
    largest = np.array([[np.delete(errors[:, t], w).max() for t in range(4)] for w in range(5)])
    # alpha = 0 with a positive price: every shortfall cost k <= 0.
    inputs["imbalance_gbp_per_mwh"] = np.full((5, 4), -10.0)
    q, alpha, _ = market.newsvendor_positions(**inputs)
    assert np.all(alpha == 0.0)
    np.testing.assert_array_equal(q, np.maximum(inputs["forecast_kwh"] + smallest, 0.0))
    # alpha = 1: the price reaches the shortfall cost.
    inputs["imbalance_gbp_per_mwh"] = np.full((5, 4), 30.0)
    inputs["day_ahead_gbp_per_mwh"] = np.full((5, 4), 30.0)
    q, alpha, _ = market.newsvendor_positions(**inputs)
    assert np.all(alpha == 1.0)
    np.testing.assert_array_equal(q, np.maximum(inputs["forecast_kwh"] + largest, 0.0))


def test_position_is_non_decreasing_in_the_day_ahead_price() -> None:
    inputs = _inputs(worlds=9)
    previous = None
    for price in np.linspace(-20.0, 300.0, 65):
        inputs["day_ahead_gbp_per_mwh"] = np.full((9, 4), price)
        q, _, _ = market.newsvendor_positions(**inputs)
        if previous is not None:
            assert np.all(q >= previous)
        previous = q


def test_a_world_own_volume_forecast_and_prices_do_not_move_its_level_or_error() -> None:
    # Leave-one-out (§10.4, B3): world 2's error quantile and k read only
    # the other worlds.  Small prices keep alpha = P / k unclipped, so an
    # unchanged alpha means an unchanged k.
    inputs = _inputs(worlds=6)
    inputs["day_ahead_gbp_per_mwh"] = np.full((6, 4), 5.0)
    _, alpha, error = market.newsvendor_positions(**inputs)
    changed = {k: v.copy() for k, v in inputs.items()}
    changed["settled_kwh"][2] += np.array([50.0, -30.0, 0.1, 7.0])
    changed["forecast_kwh"][2] -= np.array([3.0, 9.0, -4.0, 0.5])
    changed["imbalance_gbp_per_mwh"][2] = np.array([900.0, -400.0, 1.0, 55.0])
    changed["system_short"][2] = ~changed["system_short"][2]
    _, alpha_changed, error_changed = market.newsvendor_positions(**changed)
    assert np.array_equal(alpha[2], alpha_changed[2])
    assert np.array_equal(error[2], error_changed[2])
    # The other worlds do see world 2.
    assert not np.array_equal(error[[0, 1, 3, 4, 5]], error_changed[[0, 1, 3, 4, 5]])


def test_shortfall_cost_is_the_other_worlds_short_state_mean() -> None:
    inputs = _inputs(worlds=3, slots=2)
    inputs["imbalance_gbp_per_mwh"] = np.array([[100.0, 100.0], [200.0, 200.0], [400.0, 400.0]])
    # Slot 0: worlds 1 and 2 short.  Slot 1: nobody short, so every other SIP.
    inputs["system_short"] = np.array([[False, False], [True, False], [True, False]])
    inputs["day_ahead_gbp_per_mwh"] = np.full((3, 2), 10.0)
    _, alpha, _ = market.newsvendor_positions(**inputs)
    k = np.array([[300.0, 300.0], [400.0, 250.0], [200.0, 150.0]])
    np.testing.assert_allclose(alpha, 10.0 / k)


def test_non_positive_day_ahead_price_commits_nothing() -> None:
    inputs = _inputs()
    inputs["day_ahead_gbp_per_mwh"][:, 1] = 0.0
    inputs["day_ahead_gbp_per_mwh"][:, 3] = -25.0
    q, alpha, _ = market.newsvendor_positions(**inputs)
    assert np.all(q[:, [1, 3]] == 0.0)
    assert np.all(alpha[:, [1, 3]] == 0.0)


def test_commit_level_is_a_fraction_and_masked_slots_commit_nothing() -> None:
    inputs = _inputs(worlds=8, slots=6)
    rng = np.random.default_rng(11)
    inputs["day_ahead_gbp_per_mwh"] = rng.uniform(-100.0, 400.0, (8, 6))
    inputs["imbalance_gbp_per_mwh"] = rng.uniform(-100.0, 400.0, (8, 6))
    inputs["mask"] = np.array([True, True, False, True, False, True])
    q, alpha, _ = market.newsvendor_positions(**inputs)
    assert np.all((alpha >= 0.0) & (alpha <= 1.0))
    assert np.all(q[:, ~inputs["mask"]] == 0.0)
    assert np.all(q >= 0.0)


def test_two_worlds_use_the_other_world_error_at_every_level() -> None:
    inputs = _inputs(worlds=2)
    errors = inputs["settled_kwh"] - inputs["forecast_kwh"]
    for price in (1.0, 80.0, 1000.0):
        inputs["day_ahead_gbp_per_mwh"] = np.full((2, 4), price)
        _, _, error = market.newsvendor_positions(**inputs)
        np.testing.assert_array_equal(error, errors[::-1])


def test_one_world_is_rejected() -> None:
    with pytest.raises(ValueError, match="two worlds"):
        market.newsvendor_positions(**_inputs(worlds=1))


# --------------------------------------------------------------------------
# run_trading with the rule switch (tiny SYNTHETIC fleet)
# --------------------------------------------------------------------------


def _short(worlds: int) -> np.ndarray:
    """SYNTHETIC NIV state: short in alternate slots."""

    return np.tile(np.arange(336) % 2 == 0, (worlds, 1))


def test_fixed_share_is_the_default_and_reproduces_the_share_ledger_bit_for_bit() -> None:
    default = _tiny_run(worlds=2)
    explicit = _tiny_run(
        worlds=2, terms={"trading.commitment_rule": "fixed_share"}, system_short=_short(2)
    )
    assert default.commitment_rule == "fixed_share"
    pd.testing.assert_frame_equal(
        default.trading_ledger_world, explicit.trading_ledger_world, check_exact=True
    )
    pd.testing.assert_frame_equal(
        default.deviation_world_slot, explicit.deviation_world_slot, check_exact=True
    )
    # §9.2: q = c x F_hat, exactly.
    assert np.array_equal(
        default.day_ahead_position, default.commitment_share * default.day_ahead_forecast
    )
    assert default.deviation_world_slot["commit_level"].isna().all()
    assert default.deviation_world_slot["forecast_error_quantile_kwh"].isna().all()
    assert default.commit_level is None and default.forecast_error_quantile is None


def test_newsvendor_run_commits_the_settled_volume_when_every_week_errs_alike() -> None:
    # The two SYNTHETIC worlds differ only by a flat 1 GBP/MWh, so plans,
    # forecasts and settled volumes match and every error sample is exact.
    run = _tiny_run(
        worlds=2, terms={"trading.commitment_rule": "newsvendor"}, system_short=_short(2)
    )
    assert run.commitment_rule == "newsvendor"
    frame = run.deviation_world_slot
    level = frame["commit_level"].to_numpy()
    assert np.all((level >= 0.0) & (level <= 1.0))
    priced = run.visible_day_ahead_gbp_per_mwh > 0.0
    np.testing.assert_allclose(run.day_ahead_position[priced], run.settled[priced], atol=1e-12)
    assert np.all(run.day_ahead_position[~priced] == 0.0)
    np.testing.assert_array_equal(
        frame["forecast_error_quantile_kwh"].to_numpy().reshape(2, 336),
        run.forecast_error_quantile,
    )
    assert np.array_equal(run.positions["da_only"], run.day_ahead_position)


def test_newsvendor_needs_the_niv_state_and_a_known_rule() -> None:
    with pytest.raises(ValueError, match="system_short"):
        _tiny_run(worlds=2, terms={"trading.commitment_rule": "newsvendor"})
    with pytest.raises(ValueError, match="commitment_rule"):
        _tiny_run(worlds=2, terms={"trading.commitment_rule": "newsboy"})


def test_mean_intraday_revision_uses_f_minus_the_point_forecast() -> None:
    run = _tiny_run(
        worlds=3, terms={"trading.commitment_rule": "newsvendor"}, system_short=_short(3)
    )

    def checks(**extra: object) -> pd.Series:
        frame = summaries.trading_checks(
            run.trading_ledger_world,
            run.trading_week_world,
            run.deviation_world_slot,
            day_ahead_position=run.day_ahead_position,
            full_position=run.positions["full"],
            revised_slots=run.revised_slots,
            commitment_share=run.commitment_share,
            fallback_nights=run.fallback_nights,
            **extra,
        )
        return frame.set_index("check_id").loc["mean_intraday_revision"]

    row = checks(day_ahead_forecast=run.day_ahead_forecast)
    revision = (run.positions["full"] - run.day_ahead_forecast)[:, run.revised_slots]
    assert row["value"] == pytest.approx(revision.mean(axis=1).mean(), rel=1e-12, abs=1e-12)
    assert "f - F_hat" in row["description"]
    # Without the point forecast the row keeps the §9.2 reading f - q/c.
    share_row = checks()
    share_revision = (run.positions["full"] - run.day_ahead_position / run.commitment_share)[
        :, run.revised_slots
    ]
    assert share_row["value"] == pytest.approx(share_revision.mean(axis=1).mean(), abs=1e-12)
