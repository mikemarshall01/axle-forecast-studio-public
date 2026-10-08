"""Partner growth calculator (supplier contract v1 §5, lane S3).

What this owns: ``funnel``, ``annual_gbp``, ``tornado``, ``months_to_fund_discount``
and ``cumulative_payout_fan`` — the arithmetic behind the Partners lens's
growth expander (supplier-v1 §11). A device maker wants to turn "£X per
enrolled device per month" (``partner_summary``, fleet, §4.2) into "£Y a
year if I sign up my base", a sensitivity read on which lever matters most,
"how many months of revenue clears a subsidised sticker price", and a
cumulative payout picture. None of that is physics, policy or settlement
(decision 0004 item 62a), so it is allowed to take the presenter's own
sales-funnel guesses as inputs; it lives in the model package, not the UI,
so it is unit-tested and the Streamlit lens computes nothing itself
(AGENTS.md: Streamlit consumes validated results, it never recalculates).

Every function here is a pure function of ordinary numbers, NumPy arrays and
pandas DataFrames: no model state, no random draw, no I/O. The one band this
module produces (`annual_gbp` called three times by the lens at the run's
own P10/P50/P90 of ``gbp_per_enrolled_device_per_month``) is world-first by
construction: multiplying a per-world quantity by a positive constant and
adding a constant preserves its quantiles exactly (supplier-v1 §5), so the
three calls need no resampling. `months_to_fund_discount` divides world by
world for the same reason: the lens passes a stored per-world column and the
quantile is taken after the division, never before.

Every £ figure here is illustrative and, once scaled to a month or a year,
labelled "scenario: scaled from one simulated week, not a forecast"
(supplier-v1 §12) — this module does not add that label itself, the lens
does; the module only computes the numbers the label describes. The funnel's
`eligible`, `invited_share`, `signed_up_share`, `active_share`, the tornado's
`take_rate` and fee, and the discount are presenter widget values with no
accepted source in this project (supplier-v1 §16 Q7); they are lens state,
not `Assumption` records, and this module takes them as plain arguments
rather than reading them from `assumptions.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TORNADO_LEVERS = (
    ("enrolled_devices", "Enrolled devices"),
    ("gbp_per_enrolled_device_per_month", "£ per enrolled device per month"),
    ("take_rate", "Take rate"),
    ("fee_gbp_per_device_per_month", "Fee, £ per device per month"),
)
"""The four `annual_gbp` arguments the tornado swings, in the presenter's likely reading order.
Sorted by impact (`swing_size`), not this order, before being returned (§5)."""


def funnel(
    *,
    eligible: float,
    invited_share: float,
    signed_up_share: float,
    active_share: float,
    share_earning: float,
) -> pd.DataFrame:
    """Sales funnel from the addressable fleet down to devices that actually earned.

    Each stage after ``eligible`` is a share of the stage immediately before
    it: ``invited = eligible * invited_share``, ``signed_up = invited *
    signed_up_share``, ``active = signed_up * active_share``. These three
    shares are the presenter's own sales-funnel guesses (no accepted source,
    §16 Q7); the model supplies none of them. The last stage, ``earning =
    active * share_earning``, is the one figure the model contributes: the
    run's own ``share_earning`` (``partner_summary``, fleet, §4.2) says what
    fraction of enrolled devices actually turned down in a settled slot, so
    an "active" device is not assumed to earn just because it signed up.

    Args:
        eligible: size of the addressable device base (count, presenter input).
        invited_share: fraction (0-1) of ``eligible`` that gets invited.
        signed_up_share: fraction (0-1) of ``invited`` that signs up.
        active_share: fraction (0-1) of ``signed_up`` that goes active.
        share_earning: fraction (0-1) of active devices that earn at least
            once in the run (the model's `partner_summary` figure, not a
            presenter lever).

    Returns:
        One row per stage in funnel order, columns ``stage`` (str, ``eligible``,
        ``invited``, ``signed_up``, ``active``, ``earning``), ``count``
        (float, devices) and ``share_of_eligible`` (float, the stage's count
        divided by ``eligible``; ``NaN`` when ``eligible`` is 0, since a
        share of an empty base is undefined rather than zero).
    """
    eligible_count = float(eligible)
    invited_count = eligible_count * invited_share
    signed_up_count = invited_count * signed_up_share
    active_count = signed_up_count * active_share
    earning_count = active_count * share_earning

    stages = ("eligible", "invited", "signed_up", "active", "earning")
    counts = (eligible_count, invited_count, signed_up_count, active_count, earning_count)
    shares = tuple(
        count / eligible_count if eligible_count != 0 else float("nan") for count in counts
    )
    return pd.DataFrame({"stage": stages, "count": counts, "share_of_eligible": shares})


def annual_gbp(
    *,
    enrolled_devices: float,
    gbp_per_enrolled_device_per_month: float,
    take_rate: float,
    fee_gbp_per_device_per_month: float = 0.0,
) -> float:
    """Illustrative annual revenue to a device maker from one fleet's flexibility cash.

    ``enrolled_devices * (gbp_per_enrolled_device_per_month * take_rate +
    fee_gbp_per_device_per_month) * 12``: the maker's cut (``take_rate``) of
    the run's gross flexibility cash per enrolled device per month
    (``partner_summary``, fleet, §4.2), plus an optional flat per-device fee,
    summed over a year. ``enrolled_devices`` is meant to be the funnel's
    ``active`` count (`funnel`, §5), not ``eligible``: only a device that
    actually signed up and went active can earn flexibility cash or be
    charged a fee. Extrapolated from one simulated week and illustrative
    (§0, §12); not a forecast or a contractual entitlement.

    Args:
        enrolled_devices: active devices, count.
        gbp_per_enrolled_device_per_month: gross flexibility cash per
            enrolled device per month, £ (a `partner_summary` quantile).
        take_rate: the maker's share of that gross cash, fraction; ordinarily
            0-1 but not clipped here (see `tornado`, which clips it when
            swinging it for the sensitivity chart).
        fee_gbp_per_device_per_month: optional flat fee per device per
            month, £; 0.0 (no fee) by default.

    Returns:
        £ per year, illustrative.
    """
    return (
        enrolled_devices
        * (gbp_per_enrolled_device_per_month * take_rate + fee_gbp_per_device_per_month)
        * 12
    )


def tornado(
    *,
    enrolled_devices: float,
    gbp_per_enrolled_device_per_month: float,
    take_rate: float,
    fee_gbp_per_device_per_month: float = 0.0,
    swing: float = 0.3,
) -> pd.DataFrame:
    """Sensitivity of `annual_gbp` to each of its four levers, one at a time.

    For each lever, the other three are held at their given (base) value and
    the lever itself is swung to ``(1 - swing) * value`` and ``(1 + swing) *
    value``; `annual_gbp` is recomputed at each end. Because `annual_gbp` is
    linear in any one argument once the other three are fixed, a symmetric
    percentage swing in the lever gives a symmetric £ swing around the base
    annual figure — the classic tornado-chart property — except where
    ``take_rate``'s swing is clipped to stay a fraction (below), which can
    break the symmetry and is exactly what the ``clipped`` column flags.

    The fee lever is only meaningful when a fee is actually charged: swinging
    a £0 fee by any percentage still gives £0, so its row is omitted when
    ``fee_gbp_per_device_per_month`` is 0.0 (the default, "no fee set").

    Args:
        enrolled_devices, gbp_per_enrolled_device_per_month, take_rate,
            fee_gbp_per_device_per_month: the base `annual_gbp` inputs the
            four levers swing around; same units as `annual_gbp`.
        swing: fractional swing applied to each lever, e.g. 0.3 for ±30 %
            (supplier-v1 §16 lists ±30 % as the default).

    Returns:
        One row per lever (three or four, fee included only when non-zero),
        columns ``lever`` (str, the `annual_gbp` argument name), ``label``
        (str, a reader-facing name), ``value`` (float, the lever's base
        value), ``low_value``, ``high_value`` (float, the swung values
        actually used, clipped to [0, 1] for ``take_rate``),
        ``annual_gbp_low``, ``annual_gbp_high`` (float, £ per year at each
        end) and ``clipped`` (bool, true only when a bound was moved by the
        [0, 1] clip). Sorted by swing size (``|annual_gbp_high -
        annual_gbp_low|``), descending, so the chart reads most-to-least
        important top to bottom.
    """
    base_values = {
        "enrolled_devices": enrolled_devices,
        "gbp_per_enrolled_device_per_month": gbp_per_enrolled_device_per_month,
        "take_rate": take_rate,
        "fee_gbp_per_device_per_month": fee_gbp_per_device_per_month,
    }

    rows = []
    for name, label in _TORNADO_LEVERS:
        value = base_values[name]
        if name == "fee_gbp_per_device_per_month" and value == 0.0:
            continue  # a zero fee swings to zero either way and carries no sensitivity information

        low_raw = (1 - swing) * value
        high_raw = (1 + swing) * value
        if name == "take_rate":
            # take_rate is a share of the maker's gross cash; a swing that would push it
            # outside [0, 1] is clipped rather than fed to annual_gbp as a negative or
            # >100% take, and the clip is flagged so the chart caption can say so.
            low_value = min(max(low_raw, 0.0), 1.0)
            high_value = min(max(high_raw, 0.0), 1.0)
            clipped = low_value != low_raw or high_value != high_raw
        else:
            low_value = low_raw
            high_value = high_raw
            clipped = False

        low_gbp = annual_gbp(**{**base_values, name: low_value})
        high_gbp = annual_gbp(**{**base_values, name: high_value})
        rows.append(
            {
                "lever": name,
                "label": label,
                "value": value,
                "low_value": low_value,
                "high_value": high_value,
                "annual_gbp_low": low_gbp,
                "annual_gbp_high": high_gbp,
                "clipped": clipped,
            }
        )

    frame = pd.DataFrame(rows)
    swing_size = (frame["annual_gbp_high"] - frame["annual_gbp_low"]).abs()
    return frame.iloc[swing_size.sort_values(ascending=False, kind="stable").index].reset_index(
        drop=True
    )


def months_to_fund_discount(
    discount_gbp: float,
    *,
    gbp_per_enrolled_device_per_month_by_world: np.ndarray,
    take_rate: float,
) -> tuple[float, float, float, int]:
    """Months of a maker's flexibility revenue needed to fund a sticker-price discount.

    Per world ``w``: ``months_w = discount_gbp / (gbp_per_enrolled_device_per_month_by_world[w]
    * take_rate)``, the number of months of take-rate revenue per device that
    would recoup a one-off discount of ``discount_gbp`` per device. Computed
    world first, so this is a simulated-weeks band, not a quantile-of-a-mean:
    ``months_w`` is ``NaN`` wherever that world's revenue is not strictly
    positive ("not funded" — a discount cannot be recouped from a
    non-positive revenue stream), and the (p10, p50, p90) quantiles are
    `numpy.quantile` over the worlds left after dropping those, with their
    count returned alongside so the lens can say how many worlds fund it at
    all. ``gbp_per_enrolled_device_per_month_by_world`` is meant to be the
    fleet row of `partner_world` (`gross_flex_gbp_per_enrolled_device_per_month`),
    the per-world column the result already carries (§7), so no result frame
    is recomputed here.

    Args:
        discount_gbp: the one-off sticker-price discount to fund, £ per device.
        gbp_per_enrolled_device_per_month_by_world: one gross flexibility cash
            figure per world, £ per enrolled device per month (`partner_world`,
            fleet row).
        take_rate: the maker's share of that gross cash, fraction.

    Returns:
        ``(p10, p50, p90, world_count)``: months, months, months, and the
        count of worlds with strictly positive revenue the quantiles were
        taken over. All three quantiles are ``NaN`` and ``world_count`` is 0
        when no world funds the discount.
    """
    revenue_by_world = (
        np.asarray(gbp_per_enrolled_device_per_month_by_world, dtype=float) * take_rate
    )
    # np.where evaluates both branches, so a zero or negative revenue world would raise a
    # spurious "divide by zero" warning even though its result is discarded; suppress it
    # rather than let it mask a real one elsewhere.
    with np.errstate(divide="ignore", invalid="ignore"):
        months_by_world = np.where(revenue_by_world > 0, discount_gbp / revenue_by_world, np.nan)
    funded = months_by_world[~np.isnan(months_by_world)]
    world_count = int(funded.size)
    if world_count == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    p10, p50, p90 = np.quantile(funded, (0.1, 0.5, 0.9), method="linear")
    return (float(p10), float(p50), float(p90), world_count)


def cumulative_payout_fan(
    *,
    monthly_gbp_p10: float,
    monthly_gbp_p50: float,
    monthly_gbp_p90: float,
    months: int = 12,
) -> pd.DataFrame:
    """Cumulative payout over a number of months, at each monthly quantile.

    Row ``month = k`` holds ``k * monthly_gbp_p10`` and likewise for p50 and
    p90: this assumes the same simulated week's monthly figure repeats every
    month (the months are taken as fully correlated), which is why a plain
    multiple of the monthly quantile gives the cumulative quantile exactly.
    Independent months (the weeks' variation averaging out over a year) would
    give a narrower band; this fan is deliberately the wider, more
    conservative reading rather than a claim the weeks are independent
    (supplier-v1 §5). It is a scenario — "if this week repeated" — not a
    forecast.

    Args:
        monthly_gbp_p10, monthly_gbp_p50, monthly_gbp_p90: the P10/P50/P90 of
            one month's payout, £ (e.g. `supplier_pnl_summary`'s
            ``customer_payment_per_customer_per_month_gbp``, sign-flipped to
            a positive payout, or that times the funnel's active count for
            the fleet fan).
        months: number of months to project, 12 by default (supplier-v1 §8
            Q12).

    Returns:
        One row per month, 1..``months``, columns ``month`` (int),
        ``cumulative_p10``, ``cumulative_p50``, ``cumulative_p90`` (float, £).
    """
    month_number = np.arange(1, months + 1)
    return pd.DataFrame(
        {
            "month": month_number,
            "cumulative_p10": month_number * monthly_gbp_p10,
            "cumulative_p50": month_number * monthly_gbp_p50,
            "cumulative_p90": month_number * monthly_gbp_p90,
        }
    )
