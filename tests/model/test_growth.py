"""Tests for the partner growth calculator (supplier contract v1 §5).

Every check is a hand-computable fixture: these functions take no model
state and draw no random numbers, so there is nothing to sample and no
fixture result to load. Tolerances follow the contract's
``1e-9 * max(1, |value|)`` rule for float comparisons.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from axle_studio.model import growth


def _close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


# ---------------------------------------------------------------------------
# funnel
# ---------------------------------------------------------------------------


def test_funnel_stage_counts_multiply_through_and_earning_uses_the_models_share() -> None:
    frame = growth.funnel(
        eligible=100_000,
        invited_share=0.5,
        signed_up_share=0.3,
        active_share=0.8,
        share_earning=0.2,
    )
    expected_counts = {
        "eligible": 100_000,
        "invited": 50_000,
        "signed_up": 15_000,
        "active": 12_000,
        "earning": 2_400,
    }
    counts = dict(zip(frame["stage"], frame["count"]))
    for stage, expected in expected_counts.items():
        assert _close(counts[stage], expected), (stage, counts[stage], expected)

    # Stage order matches the funnel, not any other sort.
    assert list(frame["stage"]) == ["eligible", "invited", "signed_up", "active", "earning"]


def test_funnel_is_monotonically_non_increasing_in_count_for_shares_in_0_1() -> None:
    frame = growth.funnel(
        eligible=1_000,
        invited_share=0.6,
        signed_up_share=0.4,
        active_share=0.9,
        share_earning=0.25,
    )
    counts = frame["count"].to_numpy()
    assert np.all(np.diff(counts) <= 1e-9), "each stage must not exceed the stage before it"


def test_funnel_share_of_eligible_is_count_over_eligible() -> None:
    frame = growth.funnel(
        eligible=200,
        invited_share=0.5,
        signed_up_share=0.5,
        active_share=0.5,
        share_earning=0.5,
    )
    for count, share in zip(frame["count"], frame["share_of_eligible"]):
        assert _close(share, count / 200)


def test_funnel_share_of_eligible_is_nan_when_eligible_is_zero() -> None:
    frame = growth.funnel(
        eligible=0,
        invited_share=0.5,
        signed_up_share=0.5,
        active_share=0.5,
        share_earning=0.5,
    )
    assert frame["count"].eq(0.0).all()
    assert frame["share_of_eligible"].apply(math.isnan).all()


# ---------------------------------------------------------------------------
# annual_gbp
# ---------------------------------------------------------------------------


def test_annual_gbp_hand_computed() -> None:
    # 1,000 devices at £2/device/month take 20%, plus a £0.50/device/month fee.
    value = growth.annual_gbp(
        enrolled_devices=1_000,
        gbp_per_enrolled_device_per_month=2.0,
        take_rate=0.2,
        fee_gbp_per_device_per_month=0.5,
    )
    assert _close(value, 1_000 * (2.0 * 0.2 + 0.5) * 12)


def test_annual_gbp_is_linear_in_enrolled_devices_take_rate_and_fee() -> None:
    base = growth.annual_gbp(
        enrolled_devices=500,
        gbp_per_enrolled_device_per_month=3.0,
        take_rate=0.25,
        fee_gbp_per_device_per_month=1.0,
    )
    doubled_devices = growth.annual_gbp(
        enrolled_devices=1_000,
        gbp_per_enrolled_device_per_month=3.0,
        take_rate=0.25,
        fee_gbp_per_device_per_month=1.0,
    )
    doubled_take = growth.annual_gbp(
        enrolled_devices=500,
        gbp_per_enrolled_device_per_month=3.0,
        take_rate=0.5,
        fee_gbp_per_device_per_month=1.0,
    )
    doubled_fee = growth.annual_gbp(
        enrolled_devices=500,
        gbp_per_enrolled_device_per_month=3.0,
        take_rate=0.25,
        fee_gbp_per_device_per_month=2.0,
    )
    # Doubling enrolled_devices doubles the whole figure (linear through the origin).
    assert _close(doubled_devices, 2 * base)
    # Raising take_rate by delta adds enrolled_devices * gbp_per_device * delta * 12.
    assert _close(doubled_take - base, 500 * 3.0 * (0.5 - 0.25) * 12)
    # Raising the fee by delta adds enrolled_devices * delta * 12.
    assert _close(doubled_fee - base, 500 * (2.0 - 1.0) * 12)


def test_annual_gbp_three_call_band_equals_per_world_quantiles() -> None:
    """The lens calls annual_gbp three times at the run's p10/p50/p90 of the per-device
    figure; because annual_gbp is affine in that argument, those three calls equal the
    quantiles of a full per-world recomputation (supplier-v1 §5, "band" paragraph)."""
    rng = np.random.default_rng(0)
    per_world_gbp_per_device = rng.uniform(0.5, 5.0, size=200)
    p10, p50, p90 = np.quantile(per_world_gbp_per_device, (0.1, 0.5, 0.9), method="linear")

    kwargs = dict(enrolled_devices=12_000, take_rate=0.2, fee_gbp_per_device_per_month=0.0)
    band_from_three_calls = (
        growth.annual_gbp(gbp_per_enrolled_device_per_month=p10, **kwargs),
        growth.annual_gbp(gbp_per_enrolled_device_per_month=p50, **kwargs),
        growth.annual_gbp(gbp_per_enrolled_device_per_month=p90, **kwargs),
    )

    per_world_annual = np.array(
        [
            growth.annual_gbp(gbp_per_enrolled_device_per_month=v, **kwargs)
            for v in per_world_gbp_per_device
        ]
    )
    band_from_recomputation = np.quantile(per_world_annual, (0.1, 0.5, 0.9), method="linear")

    for from_calls, from_recomputation in zip(band_from_three_calls, band_from_recomputation):
        assert _close(from_calls, from_recomputation)


# ---------------------------------------------------------------------------
# tornado
# ---------------------------------------------------------------------------


def _base_tornado_kwargs() -> dict:
    return dict(
        enrolled_devices=10_000,
        gbp_per_enrolled_device_per_month=4.0,
        take_rate=0.2,
    )


def test_tornado_omits_fee_row_when_fee_is_zero_and_includes_it_when_set() -> None:
    without_fee = growth.tornado(**_base_tornado_kwargs())
    assert "fee_gbp_per_device_per_month" not in set(without_fee["lever"])
    assert len(without_fee) == 3

    with_fee = growth.tornado(**_base_tornado_kwargs(), fee_gbp_per_device_per_month=0.5)
    assert "fee_gbp_per_device_per_month" in set(with_fee["lever"])
    assert len(with_fee) == 4


def test_tornado_swings_are_symmetric_about_the_base_when_unclipped() -> None:
    frame = growth.tornado(**_base_tornado_kwargs(), fee_gbp_per_device_per_month=1.0, swing=0.3)
    base_annual = growth.annual_gbp(**_base_tornado_kwargs(), fee_gbp_per_device_per_month=1.0)
    assert (~frame["clipped"]).all(), "base case (take_rate 0.2, swing 0.3) must not clip"
    for _, row in frame.iterrows():
        down = base_annual - row["annual_gbp_low"]
        up = row["annual_gbp_high"] - base_annual
        assert _close(down, up), (row["lever"], down, up)


def test_tornado_clips_take_rate_to_0_1_and_flags_it() -> None:
    # take_rate 0.8 swung by +30% would reach 1.04; must clip to 1.0 and flag the row.
    frame = growth.tornado(
        enrolled_devices=10_000,
        gbp_per_enrolled_device_per_month=4.0,
        take_rate=0.8,
        swing=0.3,
    )
    take_rate_row = frame.set_index("lever").loc["take_rate"]
    assert take_rate_row["clipped"]
    assert _close(take_rate_row["high_value"], 1.0)
    assert _close(take_rate_row["low_value"], 0.8 * 0.7)  # unclipped: 0.56, inside [0, 1]

    expected_high_gbp = growth.annual_gbp(
        enrolled_devices=10_000, gbp_per_enrolled_device_per_month=4.0, take_rate=1.0
    )
    assert _close(take_rate_row["annual_gbp_high"], expected_high_gbp)


def test_tornado_is_sorted_by_swing_size_descending() -> None:
    # gbp_per_enrolled_device_per_month and take_rate only move the product gbp * take,
    # a small part of the (gbp * take + fee) bracket, while fee moves its own large
    # value directly: its swing must land ahead of theirs.
    frame = growth.tornado(
        enrolled_devices=10_000,
        gbp_per_enrolled_device_per_month=4.0,
        take_rate=0.2,
        fee_gbp_per_device_per_month=50.0,
        swing=0.3,
    )
    swing_sizes = (frame["annual_gbp_high"] - frame["annual_gbp_low"]).abs().to_numpy()
    assert np.all(np.diff(swing_sizes) <= 1e-9), "swing sizes must be non-increasing down the frame"
    ranked_levers = list(frame["lever"])
    # enrolled_devices scales the whole (gbp * take + fee) bracket, so its swing is the sum
    # of the fee lever's and the gbp/take levers' swings and is always largest or tied.
    assert ranked_levers[0] == "enrolled_devices"
    assert ranked_levers[1] == "fee_gbp_per_device_per_month"
    assert set(ranked_levers[2:]) == {"gbp_per_enrolled_device_per_month", "take_rate"}


# ---------------------------------------------------------------------------
# months_to_fund_discount
# ---------------------------------------------------------------------------


def test_months_to_fund_discount_hand_computed_five_world_fixture() -> None:
    # Fleet gbp/device/month across five worlds; world index 2 has non-positive revenue
    # (a negative per-device figure is a valid simulated outcome, not an error) and must
    # be dropped from the quantile, not turned into an inf or a spurious large number.
    per_world_gbp_per_device = np.array([2.0, 4.0, -1.0, 5.0, 1.0])
    take_rate = 0.5
    discount_gbp = 10.0

    p10, p50, p90, world_count = growth.months_to_fund_discount(
        discount_gbp,
        gbp_per_enrolled_device_per_month_by_world=per_world_gbp_per_device,
        take_rate=take_rate,
    )

    # Hand computation: months_w = discount / (gbp_per_device_w * take_rate) for the
    # four worlds with positive revenue.
    expected_months = np.array(
        [10.0 / (2.0 * 0.5), 10.0 / (4.0 * 0.5), 10.0 / (5.0 * 0.5), 10.0 / (1.0 * 0.5)]
    )
    expected_p10, expected_p50, expected_p90 = np.quantile(
        expected_months, (0.1, 0.5, 0.9), method="linear"
    )

    assert world_count == 4
    assert _close(p10, expected_p10)
    assert _close(p50, expected_p50)
    assert _close(p90, expected_p90)


def test_months_to_fund_discount_all_worlds_non_positive_gives_nan_and_zero_count() -> None:
    p10, p50, p90, world_count = growth.months_to_fund_discount(
        10.0,
        gbp_per_enrolled_device_per_month_by_world=np.array([-1.0, 0.0, -0.5]),
        take_rate=0.5,
    )
    assert world_count == 0
    assert math.isnan(p10)
    assert math.isnan(p50)
    assert math.isnan(p90)


def test_months_to_fund_discount_zero_discount_gives_zero_months_never_inf() -> None:
    p10, p50, p90, world_count = growth.months_to_fund_discount(
        0.0,
        gbp_per_enrolled_device_per_month_by_world=np.array([1.0, 2.0, 3.0]),
        take_rate=0.5,
    )
    assert world_count == 3
    for value in (p10, p50, p90):
        assert value == 0.0
        assert math.isfinite(value)


def test_months_to_fund_discount_zero_revenue_world_is_dropped_not_infinite() -> None:
    # take_rate 0 makes every world's revenue exactly 0: "not positive" per the contract,
    # so every world is dropped rather than producing +inf from a division by zero.
    p10, p50, p90, world_count = growth.months_to_fund_discount(
        10.0,
        gbp_per_enrolled_device_per_month_by_world=np.array([1.0, 2.0, 3.0]),
        take_rate=0.0,
    )
    assert world_count == 0
    assert math.isnan(p10) and math.isnan(p50) and math.isnan(p90)


# ---------------------------------------------------------------------------
# cumulative_payout_fan
# ---------------------------------------------------------------------------


def test_cumulative_payout_fan_is_month_times_monthly_with_twelve_rows_by_default() -> None:
    frame = growth.cumulative_payout_fan(
        monthly_gbp_p10=5.0, monthly_gbp_p50=8.0, monthly_gbp_p90=12.0
    )
    assert len(frame) == 12
    assert list(frame["month"]) == list(range(1, 13))
    for _, row in frame.iterrows():
        assert _close(row["cumulative_p10"], row["month"] * 5.0)
        assert _close(row["cumulative_p50"], row["month"] * 8.0)
        assert _close(row["cumulative_p90"], row["month"] * 12.0)


def test_cumulative_payout_fan_respects_custom_month_count() -> None:
    frame = growth.cumulative_payout_fan(
        monthly_gbp_p10=1.0, monthly_gbp_p50=2.0, monthly_gbp_p90=3.0, months=3
    )
    assert list(frame["month"]) == [1, 2, 3]
    assert _close(frame.iloc[-1]["cumulative_p90"], 9.0)


@pytest.mark.parametrize("months", [1, 6, 24])
def test_cumulative_payout_fan_quantiles_stay_ordered(months: int) -> None:
    frame = growth.cumulative_payout_fan(
        monthly_gbp_p10=4.0, monthly_gbp_p50=9.0, monthly_gbp_p90=15.0, months=months
    )
    assert (frame["cumulative_p10"] <= frame["cumulative_p50"]).all()
    assert (frame["cumulative_p50"] <= frame["cumulative_p90"]).all()
