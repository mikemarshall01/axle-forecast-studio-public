"""Supplier P&L and Partners frames (supplier contract v1 §3–§4, lane S1; decision 0006).

What this owns: the illustrative simulated supplier P&L of one action run and
the partner-facing figures built beside it.

- ``supplier_pnl_world`` / ``supplier_pnl_summary`` (§3.2–§3.4): what smart
  charging does to the supplier's procurement cost, its hedge error under
  half-hourly settlement, its third-party grid-event income and its
  customer payments, per simulated week, against a profiled and a flat
  day-ahead hedge.
- ``hedge_block_world`` / ``hedge_block_summary`` (§3.7): the baseload, peak
  and off-peak blocks a supplier buys, unmanaged against smart.
- ``carbon_intensity_gco2_per_kwh``, ``carbon_shift_world`` /
  ``carbon_shift_summary`` (§3.8): CO₂ moved by smart charging under an
  illustrative synthetic intensity tied to the model's own net demand.  The
  intensity function is public so other lanes (household-v1) read per-EV
  CO₂ from the same rule.
- ``partner_world`` / ``partner_summary`` (§4.1–§4.2), the exceedance frame
  (§4.3) and ``revenue_by_market_summary`` (§4.5) for device makers.
- ``build_frames`` builds all ten from the finished run;
  ``summaries.build_summaries`` calls it once, after the trading and
  Supplier frames of trading contract v1 §9.3.

Rules this module keeps:

- **Never added to the trading ledger** (§3.1, decision 0006).  The smart
  supplier's hedge already leaves out the turn-down the aggregator sold, so
  the energy and hedge-error components hold the value the ledger's
  day-ahead and imbalance buckets also hold.  The only ledger bucket used is
  the third-party grid-event payment; the ledger net is shown beside by the
  lens, never summed here.
- **Unset commercial terms are NaN, never £0** (decision 0003): the platform
  fee and the flat customer reward are NaN until set, and every net or
  per-customer figure that needs them is NaN with them.
- **Worlds first**: every figure is computed inside each world, then
  ``numpy.quantile(..., method="linear")`` across worlds (results-v2 §1.3).
- Deterministic: nothing here draws a random number.

Every £, tonne and MWh is illustrative and synthetic; nothing here is Axle
cash, a bid, a tariff or a settlement.  Per-month figures are "extrapolated
from one simulated week" (× 52 ÷ 12).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from axle_studio.model import market

EVIDENCE_KIND = "illustrative_synthetic"
WEEKS_PER_MONTH = 52.0 / 12.0
WEEKS_PER_YEAR = 52.0
HALF_HOUR_HOURS = 0.5
HEDGE_VARIANTS = ("profiled", "flat")
BLOCKS = ("baseload", "peak", "off_peak")
BLOCK_PATHS = ("normal", "selected", "difference")
EXCEEDANCE_THRESHOLDS_GBP_PER_MONTH = np.arange(0, 41, dtype=np.int64)
"""£0 … £40 per household per month in £1 steps (the task's grid; the lens default is £10)."""

# The ledger's seven buckets above the customer share (trading contract §4.7):
# the "gross flexibility cash" a device maker's figure is built from (§4.1).
GROSS_BUCKETS = market.BUCKETS[:7]

# §4.5: each modelled market is a sum of ledger buckets (strategy ``full``).
# ``baseline_effect`` is not a market and ``deductions`` are not income; both
# are listed so the rows add to ``net`` exactly (decision 0005).
MARKET_BUCKETS = {
    "day_ahead": ("day_ahead_revenue_gbp",),
    "intraday": ("intraday_pnl_gbp", "trading_cost_gbp"),
    "imbalance": ("imbalance_gbp",),
    "grid_events": ("grid_event_payment_gbp",),
    "baseline_effect": ("baseline_effect_gbp",),
    "deductions": (
        "supplier_compensation_gbp",
        "customer_revenue_share_gbp",
        "unmet_charge_penalty_gbp",
    ),
    "net": ("net_gbp",),
}
# Listed by name and never shown as zero (decision 0005).
UNMODELLED_MARKETS = (
    "balancing_mechanism",
    "frequency_response",
    "capacity_market",
    "dno_services",
)

SUPPLIER_RECORDS = (
    "supplier.customer_reward_mode",
    "supplier.customer_reward_gbp_per_ev_per_month",
    "supplier.platform_fee_gbp_per_ev_per_month",
)
CARBON_RECORDS = (
    "carbon.intensity_at_reference_gco2_per_kwh",
    "carbon.intensity_slope_gco2_per_kwh_per_gw",
    "carbon.intensity_floor_gco2_per_kwh",
    "carbon.intensity_cap_gco2_per_kwh",
)

FRAME_FIELDS = (
    "supplier_pnl_world",
    "supplier_pnl_summary",
    "hedge_block_world",
    "hedge_block_summary",
    "carbon_shift_world",
    "carbon_shift_summary",
    "partner_world",
    "partner_summary",
    "household_value_exceedance",
    "revenue_by_market_summary",
)
"""The ``ForecastResult`` fields ``build_frames`` returns (action results only)."""

HORIZON_LABELS = {
    "intraday": "intraday: decided at 17:00 on the day",
    "day_ahead": "day-ahead: decided at 13:00 the day before",
    "week_ahead": "week-ahead: the spread across simulated weeks",
    "scenario": "scenario: scaled from one simulated week, not a forecast",
}
"""Item 65's horizon vocabulary (§0, §12) and the label a lens shows for each."""

HORIZON_BY_FRAME = {
    # This module's summaries carry a ``horizon`` column per row (weekly rows
    # ``week_ahead``, per-month and per-year rows ``scenario``); the value
    # here is the frame's headline horizon and the column is authoritative.
    "supplier_pnl_summary": "week_ahead",
    "hedge_block_summary": "week_ahead",
    "carbon_shift_summary": "week_ahead",
    "partner_summary": "week_ahead",
    "household_value_exceedance": "scenario",
    "revenue_by_market_summary": "week_ahead",
    # The trading contract frames the Supplier lenses also show (§12).
    "flex_cost_curve": "day_ahead",
    "supplier_positions": "day_ahead",
    "product_sheet": "day_ahead",
    "shape_premium_summary": "week_ahead",
    "revenue_by_segment": "week_ahead",
    "trading_kpis": "week_ahead",
    "charge_completion_summary": "week_ahead",
    "household_value_distribution": "scenario",
    "household_value_summary": "scenario",
}
"""Frame name -> horizon key of ``HORIZON_LABELS`` (item 65, §12)."""

_STATS = ["world_count", "mean", "p10", "p50", "p90"]
PNL_WORLD_COLUMNS = [
    "world_id",
    "hedge_variant",
    "energy_cost_unmanaged_gbp",
    "energy_cost_smart_gbp",
    "energy_saving_gbp",
    "shape_saving_gbp",
    "volume_value_gbp",
    "forecast_reduction_mwh",
    "realised_reduction_mwh",
    "hedge_error_saving_gbp",
    "worth_of_profiled_hedge_gbp",
    "grid_event_payment_gbp",
    "gross_gain_gbp",
    "customer_payment_gbp",
    "net_gain_before_fee_gbp",
    "platform_fee_gbp",
    "net_gain_gbp",
    "treated_ev_count",
    "net_gain_before_fee_per_customer_per_month_gbp",
    "net_gain_per_customer_per_month_gbp",
    "customer_payment_per_customer_per_month_gbp",
    "platform_fee_per_customer_per_month_gbp",
    "evidence_kind",
]
PNL_SUMMARY_COLUMNS = [
    "hedge_variant",
    "metric",
    "unit",
    "horizon",
    "season",
    *_STATS,
    "evidence_kind",
]
HEDGE_BLOCK_WORLD_COLUMNS = [
    "world_id",
    "block",
    "path_id",
    "block_hours",
    "volume_mwh",
    "mean_mw",
    "evidence_kind",
]
HEDGE_BLOCK_SUMMARY_COLUMNS = [
    "block",
    "path_id",
    "metric",
    "unit",
    "horizon",
    "season",
    "block_hours",
    *_STATS,
    "evidence_kind",
]
CARBON_WORLD_COLUMNS = [
    "world_id",
    "co2_unmanaged_kg",
    "co2_smart_kg",
    "co2_shifted_kg",
    "co2_shifted_kg_per_ev_per_month",
    "evidence_kind",
]
CARBON_SUMMARY_COLUMNS = ["metric", "unit", "horizon", "season", *_STATS, "evidence_kind"]
PARTNER_METRIC_UNITS = {
    "share_earning": "fraction",
    "gross_flex_gbp_per_week": "GBP per week",
    "gross_flex_gbp_per_enrolled_device_per_month": "GBP per device per month",
    "gross_flex_gbp_per_earning_device_per_month": "GBP per device per month",
    "gbp_per_kw_charger_per_year": "GBP per kW per year",
    "customer_value_gbp_per_device_per_month": "GBP per device per month",
    "dispatch_success_rate": "fraction",
    "equivalent_full_cycles_per_ev_per_week_normal": "cycles per EV per week",
    "equivalent_full_cycles_per_ev_per_week_selected": "cycles per EV per week",
    "equivalent_full_cycles_per_ev_per_week_difference": "cycles per EV per week",
}
PARTNER_WORLD_COLUMNS = [
    "world_id",
    "group_type",
    "group_id",
    "enrolled_ev_count",
    "earning_ev_count",
    *PARTNER_METRIC_UNITS,
    "evidence_kind",
]
PARTNER_SUMMARY_COLUMNS = [
    "group_type",
    "group_id",
    "metric",
    "unit",
    "horizon",
    "season",
    "enrolled_ev_count",
    *_STATS,
    "evidence_kind",
]
EXCEEDANCE_COLUMNS = [
    "group_type",
    "group_id",
    "metric",
    "threshold_gbp_per_month",
    "unit",
    "horizon",
    "season",
    "enrolled_ev_count",
    *_STATS,
    "evidence_kind",
]
REVENUE_BY_MARKET_COLUMNS = [
    "market",
    "modelled",
    "metric",
    "unit",
    "horizon",
    "season",
    *_STATS,
    "evidence_kind",
]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _stats(values: np.ndarray) -> dict[str, float]:
    """World count, mean and linear P10/P50/P90 of per-world values, NaN left out."""

    kept = np.asarray(values, dtype=float)
    kept = kept[~np.isnan(kept)]
    if len(kept) == 0:
        return {"world_count": 0, "mean": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan}
    q = np.quantile(kept, (0.1, 0.5, 0.9), method="linear")
    return {
        "world_count": int(len(kept)),
        "mean": float(kept.mean()),
        "p10": float(q[0]),
        "p50": float(q[1]),
        "p90": float(q[2]),
    }


def _per_count(values: np.ndarray, count: float) -> np.ndarray:
    """``values ÷ count``, NaN when the count is 0 (a group or fleet with nobody in it)."""

    values = np.asarray(values, dtype=float)
    return values / count if count > 0 else np.full(values.shape, np.nan)


def _matrix(frame: pd.DataFrame, column: str, world_count: int) -> np.ndarray:
    """(world, slot) array of one column of a world-major (world, slot) frame."""

    ordered = frame.sort_values(["world_id", "slot_index"], kind="stable")
    return ordered[column].to_numpy(dtype=float).reshape(world_count, -1)


def season_of_run(
    study_slots: pd.DataFrame, warmup_slot_count: int, summer_months: Sequence[int]
) -> str:
    """``"summer"`` or ``"winter"`` for the run, by the prices generator's one rule (§0).

    The season is set by the London month of the run's first priced slot,
    the warm-up start ``warmup_slot_count`` half-hours before the study
    (plan D7, decision 0004 item 53), and the ``summer_months`` record, so
    the label agrees with the demand shapes the prices used.
    """

    first = study_slots["interval_start_utc"].iloc[0] - warmup_slot_count * pd.Timedelta(minutes=30)
    month = first.tz_convert("Europe/London").month
    return "summer" if month in tuple(summer_months) else "winter"


def battery_added_kwh(
    unit: Mapping[str, np.ndarray], home_efficiency: float, public_efficiency: float
) -> np.ndarray:
    """Battery-side energy added over the week per (world, EV), kWh, for one path.

    ``unit`` holds one path's per-EV kernel arrays (world, slot, EV): home
    and public grid import in kWh.  The kernel adds grid import × the
    charge efficiency to the battery (the reconciliation rule of
    ``summaries._check_reconciles``), so this is the battery energy the
    equivalent-full-cycle count of §4.1 divides by capacity.
    """

    home = unit["home_grid_import_kwh"].sum(axis=1) * home_efficiency
    public = unit["public_grid_import_kwh"].sum(axis=1) * public_efficiency
    return home + public


# --------------------------------------------------------------------------
# §3.2–§3.4 Supplier P&L
# --------------------------------------------------------------------------


def _shape_cost_gbp(load_kwh: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Per-world shape cost of trading contract §9.3b, £ per week (same arithmetic order).

    ``(load-weighted price − baseload price) × volume ÷ 1000``; NaN for a
    world with no import.  Kept bit-for-bit like ``shape_premium_summary``
    so the "of which shape" figure equals minus its ``difference`` row.
    A 0 for a world with no import would keep ``shape + volume = energy``
    but break that second half of §3.5 identity 2 (the §9.3b row's premium
    is NaN there, so its world count would differ); such a world has no
    fleet home import in a week, which a real run does not produce.
    """

    volume = load_kwh.sum(axis=1)
    weighted = np.divide(
        (load_kwh * price).sum(axis=1),
        volume,
        out=np.full(volume.shape, np.nan),
        where=volume != 0.0,
    )
    return (weighted - price.mean(axis=1)) * volume / 1000.0


def supplier_pnl_world(
    *,
    unmanaged_kwh: np.ndarray,
    metered_kwh: np.ndarray,
    true_reduction_kwh: np.ndarray,
    expected_unmanaged_kwh: np.ndarray,
    expected_metered_kwh: np.ndarray,
    day_ahead_gbp_per_mwh: np.ndarray,
    imbalance_gbp_per_mwh: np.ndarray,
    energy_saving_gbp: np.ndarray,
    grid_event_payment_gbp: np.ndarray,
    treated_ev_count: int,
    records: Mapping[str, float],
) -> pd.DataFrame:
    """Contract §3.3: one row per (world, hedge variant), £ per simulated week.

    Slot arrays are (world, study slot): fleet home import on the unmanaged
    path ``U`` and the dispatched path ``M`` (kWh), the realised reduction
    ``R = U − M`` (kWh), the aggregator's expected import at each night's
    day-ahead decision with every session unmanaged ``x^U`` and as planned
    ``x`` (kWh), and the day-ahead ``P`` and imbalance ``SIP`` prices
    (£/MWh).  ``energy_saving_gbp`` (world,) is minus the ``cost_effect``
    home-import column, reused rather than recomputed (§3.5 identity 1);
    ``grid_event_payment_gbp`` (world,) is the ``full`` strategy's weekly
    ledger bucket.  ``treated_ev_count`` is ``N_T`` (control EVs are not
    customers).  ``records`` holds ``trading.customer_revenue_share`` and
    the ``SUPPLIER`` record values.
    """

    mode = records["supplier.customer_reward_mode"]
    if mode not in (0, 1):
        raise ValueError("supplier.customer_reward_mode must be 0 (revenue share) or 1 (flat)")
    world_count = unmanaged_kwh.shape[0]
    price = day_ahead_gbp_per_mwh
    premium = imbalance_gbp_per_mwh - price

    energy_cost_unmanaged = (unmanaged_kwh * price).sum(axis=1) / 1000.0
    energy_cost_smart = (metered_kwh * price).sum(axis=1) / 1000.0
    # "Of which": the shape saving (the load moved to cheaper half-hours) and
    # the baseload value of any volume change (early departures move energy
    # to public top-ups).  With equal weekly volumes the shape is the whole
    # saving: "it is shape, not less energy".
    shape_saving = _shape_cost_gbp(unmanaged_kwh, price) - _shape_cost_gbp(metered_kwh, price)
    volume_value = (
        price.mean(axis=1) * (unmanaged_kwh.sum(axis=1) - metered_kwh.sum(axis=1)) / 1000.0
    )

    # Hedge error.  A supplier that buys hedge H at day-ahead and settles
    # m − H at SIP pays Σ m P + Σ (m − H)(SIP − P): the energy at day-ahead
    # plus a hedge error that is zero when SIP = P.  Like-for-like hedges
    # (lead choice, overnight log 29 Sep): the unmanaged supplier buys the
    # baseline B0, the smart supplier B0 − F with F = x^U − x the turn-down
    # the aggregator forecast.  B0 cancels from the paired saving, so it is
    # Σ (R − F)(SIP − P) and does not depend on the baseline history mode.
    # The method bias of the expected sessions is in both x^U and x and
    # cancels too; what remains is F's own error against R.
    forecast_reduction = expected_unmanaged_kwh - expected_metered_kwh
    profiled = ((true_reduction_kwh - forecast_reduction) * premium).sum(axis=1) / 1000.0
    # A flat block per night: F sums to 0 over each night (§2), so the two
    # suppliers' flat hedges are equal and the whole realised reduction
    # settles at the imbalance premium.
    flat = (true_reduction_kwh * premium).sum(axis=1) / 1000.0
    hedge_error = {"profiled": profiled, "flat": flat}

    share = records["trading.customer_revenue_share"]
    reward = records["supplier.customer_reward_gbp_per_ev_per_month"]
    fee = records["supplier.platform_fee_gbp_per_ev_per_month"]
    # A monthly £/EV term becomes weekly with × 12 ÷ 52; NaN (unset) stays
    # NaN, so an unset fee or flat reward is unavailable, never £0.
    platform_fee = np.full(world_count, -fee * treated_ev_count / WEEKS_PER_MONTH)

    frames = []
    for variant in HEDGE_VARIANTS:
        gross = energy_saving_gbp + hedge_error[variant] + grid_event_payment_gbp
        if mode == 0:
            # The supplier passes a share of its own positive weekly gain, so
            # the payment never exceeds what funds it (a weekly floor: a
            # supplier settles rewards monthly, not nightly like the ledger).
            customer_payment = -share * np.maximum(gross, 0.0)
        else:
            customer_payment = np.full(world_count, -reward * treated_ev_count / WEEKS_PER_MONTH)
        net_before_fee = gross + customer_payment
        net = net_before_fee + platform_fee
        frames.append(
            pd.DataFrame(
                {
                    "world_id": np.arange(world_count, dtype=np.int64),
                    "hedge_variant": variant,
                    "energy_cost_unmanaged_gbp": energy_cost_unmanaged,
                    "energy_cost_smart_gbp": energy_cost_smart,
                    "energy_saving_gbp": energy_saving_gbp,
                    "shape_saving_gbp": shape_saving,
                    "volume_value_gbp": volume_value,
                    "forecast_reduction_mwh": forecast_reduction.sum(axis=1) / 1000.0,
                    "realised_reduction_mwh": true_reduction_kwh.sum(axis=1) / 1000.0,
                    "hedge_error_saving_gbp": hedge_error[variant],
                    "worth_of_profiled_hedge_gbp": profiled - flat,
                    "grid_event_payment_gbp": grid_event_payment_gbp,
                    "gross_gain_gbp": gross,
                    "customer_payment_gbp": customer_payment,
                    "net_gain_before_fee_gbp": net_before_fee,
                    "platform_fee_gbp": platform_fee,
                    "net_gain_gbp": net,
                    "treated_ev_count": np.int64(treated_ev_count),
                    "net_gain_before_fee_per_customer_per_month_gbp": _per_count(
                        net_before_fee * WEEKS_PER_MONTH, treated_ev_count
                    ),
                    "net_gain_per_customer_per_month_gbp": _per_count(
                        net * WEEKS_PER_MONTH, treated_ev_count
                    ),
                    "customer_payment_per_customer_per_month_gbp": _per_count(
                        customer_payment * WEEKS_PER_MONTH, treated_ev_count
                    ),
                    "platform_fee_per_customer_per_month_gbp": _per_count(
                        platform_fee * WEEKS_PER_MONTH, treated_ev_count
                    ),
                    "evidence_kind": EVIDENCE_KIND,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    # World-major, profiled then flat within each world.
    order = frame["hedge_variant"].map({v: i for i, v in enumerate(HEDGE_VARIANTS)})
    frame = frame.assign(_order=order).sort_values(["world_id", "_order"], kind="stable")
    return frame.drop(columns="_order").reset_index(drop=True).loc[:, PNL_WORLD_COLUMNS]


def _pnl_unit(metric: str) -> tuple[str, str]:
    if metric.endswith("_per_customer_per_month_gbp"):
        return "GBP per customer per month", "scenario"
    if metric.endswith("_mwh"):
        return "MWh per week", "week_ahead"
    return "GBP per week", "week_ahead"


def supplier_pnl_summary(world: pd.DataFrame, season: str) -> pd.DataFrame:
    """Contract §3.4: one row per (hedge variant, metric), quantiles across worlds.

    The means of the components add up to the mean nets; the quantiles do
    not, so the lens's waterfall uses means and says so.
    """

    metrics = [
        c
        for c in PNL_WORLD_COLUMNS[2 : PNL_WORLD_COLUMNS.index("evidence_kind")]
        if c != "treated_ev_count"
    ]
    rows = []
    for variant in HEDGE_VARIANTS:
        rows_of_variant = world.loc[world["hedge_variant"].eq(variant)]
        for metric in metrics:
            unit, horizon = _pnl_unit(metric)
            rows.append(
                {
                    "hedge_variant": variant,
                    "metric": metric,
                    "unit": unit,
                    "horizon": horizon,
                    "season": season,
                    **_stats(rows_of_variant[metric].to_numpy(dtype=float)),
                    "evidence_kind": EVIDENCE_KIND,
                }
            )
    return pd.DataFrame(rows, columns=PNL_SUMMARY_COLUMNS).astype({"world_count": np.int64})


# --------------------------------------------------------------------------
# §3.7 Hedge blocks
# --------------------------------------------------------------------------


def block_masks(study_slots: pd.DataFrame) -> dict[str, np.ndarray]:
    """(slot,) bool per block: ``baseload``, ``peak`` and ``off_peak`` (§3.7).

    Peak is a slot starting 07:00–18:30 London on a Monday–Friday calendar
    date (07:00–19:00 by the clock).  A session night's small hours fall on
    the next calendar date and are classed by that date, as a supplier's
    standard blocks are.  Derived from the study's own slots, never
    hard-coded, so a clock-change week's short or long last day is counted.
    """

    london = pd.DatetimeIndex(study_slots["interval_start_london"])
    peak = np.asarray((london.dayofweek < 5) & (london.hour >= 7) & (london.hour < 19))
    return {"baseload": np.ones(len(london), dtype=bool), "peak": peak, "off_peak": ~peak}


def hedge_block_world(
    unmanaged_kwh: np.ndarray, metered_kwh: np.ndarray, study_slots: pd.DataFrame
) -> pd.DataFrame:
    """Contract §3.7: per (world, block, path) volume (MWh per week) and mean MW.

    ``mean_mw = volume_mwh ÷ block_hours``; ``difference`` is selected minus
    normal inside each world.  The ``peak`` ``difference`` ``mean_mw`` is
    the headline: the change in peak-block requirement, negative when
    smart charging moves energy out of the peak.
    """

    world_count = unmanaged_kwh.shape[0]
    rows = []
    for block, mask in block_masks(study_slots).items():
        hours = HALF_HOUR_HOURS * float(mask.sum())
        normal = unmanaged_kwh[:, mask].sum(axis=1) / 1000.0
        selected = metered_kwh[:, mask].sum(axis=1) / 1000.0
        for path, volume in zip(BLOCK_PATHS, (normal, selected, selected - normal), strict=True):
            rows.append(
                pd.DataFrame(
                    {
                        "world_id": np.arange(world_count, dtype=np.int64),
                        "block": block,
                        "path_id": path,
                        "block_hours": hours,
                        "volume_mwh": volume,
                        "mean_mw": volume / hours if hours > 0 else np.full(world_count, np.nan),
                        "evidence_kind": EVIDENCE_KIND,
                        "_order": BLOCKS.index(block) * 3 + BLOCK_PATHS.index(path),
                    }
                )
            )
    frame = pd.concat(rows, ignore_index=True).sort_values(["world_id", "_order"], kind="stable")
    return frame.reset_index(drop=True).loc[:, HEDGE_BLOCK_WORLD_COLUMNS]


def hedge_block_summary(world: pd.DataFrame, season: str) -> pd.DataFrame:
    """Contract §3.7: per (block, path, metric), quantiles across worlds."""

    metrics = (
        ("volume_mwh", "MWh per week", "week_ahead", 1.0),
        ("mean_mw", "MW", "week_ahead", 1.0),
        ("volume_mwh_per_month", "MWh per month", "scenario", WEEKS_PER_MONTH),
    )
    rows = []
    for block in BLOCKS:
        for path in BLOCK_PATHS:
            cell = world.loc[world["block"].eq(block) & world["path_id"].eq(path)]
            for metric, unit, horizon, scale in metrics:
                column = "mean_mw" if metric == "mean_mw" else "volume_mwh"
                rows.append(
                    {
                        "block": block,
                        "path_id": path,
                        "metric": metric,
                        "unit": unit,
                        "horizon": horizon,
                        "season": season,
                        "block_hours": float(cell["block_hours"].iloc[0]),
                        **_stats(cell[column].to_numpy(dtype=float) * scale),
                        "evidence_kind": EVIDENCE_KIND,
                    }
                )
    return pd.DataFrame(rows, columns=HEDGE_BLOCK_SUMMARY_COLUMNS).astype({"world_count": np.int64})


# --------------------------------------------------------------------------
# §3.8 CO₂ shifted
# --------------------------------------------------------------------------


def carbon_intensity_gco2_per_kwh(
    net_demand_gw: np.ndarray, carbon: Mapping[str, float], reference_net_demand_gw: float
) -> np.ndarray:
    """Illustrative synthetic carbon intensity, gCO₂ per kWh, same shape as ``net_demand_gw``.

    ``I = clip(I_ref + s × (D − D_ref), floor, cap)``: ``D`` is the model's
    own system net demand (GW, demand net of wind and solar, known shocks
    included; surprise shocks are not in it), ``D_ref`` the supply curve's
    reference net demand (GW) and ``carbon`` the four ``CARBON`` record
    values by name (``carbon.intensity_at_reference_gco2_per_kwh`` and so
    on).  Low net demand leaves a clean margin; high net demand is filled by
    gas up to the cap.  A mix intensity, not a marginal one, and not
    observed grid data: every figure built on it is labelled "illustrative
    synthetic carbon intensity tied to the model's own net demand" (§3.8).
    Shared with household-v1, which reads per-EV CO₂ from the same rule.
    """

    raw = carbon["carbon.intensity_at_reference_gco2_per_kwh"] + carbon[
        "carbon.intensity_slope_gco2_per_kwh_per_gw"
    ] * (np.asarray(net_demand_gw, dtype=float) - reference_net_demand_gw)
    return np.clip(
        raw,
        carbon["carbon.intensity_floor_gco2_per_kwh"],
        carbon["carbon.intensity_cap_gco2_per_kwh"],
    )


def carbon_shift_world(
    unmanaged_kwh: np.ndarray,
    metered_kwh: np.ndarray,
    intensity_gco2_per_kwh: np.ndarray,
    treated_ev_count: int,
) -> pd.DataFrame:
    """Contract §3.8: per world kg CO₂ on each path and shifted (positive = less under smart).

    kWh × gCO₂/kWh ÷ 1000 = kg.  The per-EV monthly figure divides by the
    treated count, like every per-customer figure here.
    """

    unmanaged = (unmanaged_kwh * intensity_gco2_per_kwh).sum(axis=1) / 1000.0
    smart = (metered_kwh * intensity_gco2_per_kwh).sum(axis=1) / 1000.0
    shifted = unmanaged - smart
    return pd.DataFrame(
        {
            "world_id": np.arange(len(unmanaged), dtype=np.int64),
            "co2_unmanaged_kg": unmanaged,
            "co2_smart_kg": smart,
            "co2_shifted_kg": shifted,
            "co2_shifted_kg_per_ev_per_month": _per_count(
                shifted * WEEKS_PER_MONTH, treated_ev_count
            ),
            "evidence_kind": EVIDENCE_KIND,
        }
    ).loc[:, CARBON_WORLD_COLUMNS]


def carbon_shift_summary(world: pd.DataFrame, season: str) -> pd.DataFrame:
    """Contract §3.8: one row per metric, quantiles across worlds."""

    rows = [
        {
            "metric": metric,
            "unit": "kg CO2 per EV per month"
            if metric.endswith("_per_month")
            else "kg CO2 per week",
            "horizon": "scenario" if metric.endswith("_per_month") else "week_ahead",
            "season": season,
            **_stats(world[metric].to_numpy(dtype=float)),
            "evidence_kind": EVIDENCE_KIND,
        }
        for metric in CARBON_WORLD_COLUMNS[1:-1]
    ]
    return pd.DataFrame(rows, columns=CARBON_SUMMARY_COLUMNS).astype({"world_count": np.int64})


# --------------------------------------------------------------------------
# §4 Partners
# --------------------------------------------------------------------------


def partner_groups(
    units: pd.DataFrame, cohort_ids: Sequence[str]
) -> list[tuple[str, str, np.ndarray]]:
    """``(group_type, group_id, (EV,) bool)`` for the fleet, cohorts and makers (§4.1).

    Only treated EVs (control EVs are not in the product); cohorts in
    source order and makers in id order, each only when it has a treated EV.
    Maker groups appear once ``units`` carries ``manufacturer_id`` (firm-MW
    lane J1a, trading contract §10.1d); until then there are none.
    """

    treated = ~_control(units)
    cohorts = units["cohort_id"].to_numpy(dtype=object)
    groups = [("fleet", "fleet", treated)]
    groups += [
        ("cohort", c, treated & (cohorts == c))
        for c in cohort_ids
        if (treated & (cohorts == c)).any()
    ]
    if "manufacturer_id" in units:
        makers = units["manufacturer_id"].to_numpy(dtype=object)
        groups += [
            ("manufacturer", m, treated & (makers == m)) for m in sorted(set(makers[treated]))
        ]
    return groups


def _control(units: pd.DataFrame) -> np.ndarray:
    if "control_group" in units:
        return units["control_group"].to_numpy(dtype=bool)
    return np.zeros(len(units), dtype=bool)


def dispatch_session_counts(
    plan_status: np.ndarray,
    world: np.ndarray,
    ev: np.ndarray,
    first_slot: np.ndarray,
    end_slot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per (world, EV): selected-path sessions ending in the study, and those that followed.

    ``plan_status`` (world, slot, EV) is the kernel's code (trading contract
    v1 §10.1e, 0 = the session follows its plan); ``world``, ``ev``,
    ``first_slot`` and ``end_slot`` (session,) are a chunk's
    ``summaries.find_sessions`` table (end exclusive).  Only sessions that
    end inside the study count (the §10.5e charge-completion table's
    sessions, so the two rates share a denominator).  A session follows
    when its code is 0 in every one of its connected slots, so one ignored
    replan fails the whole session (§4.2, lead decision Q11).  Returns two
    (world, EV) float arrays: sessions and followed sessions.
    """

    world_count, slot_count, ev_count = plan_status.shape
    ignored = np.zeros((world_count, slot_count + 1, ev_count))
    np.cumsum(plan_status != 0, axis=1, out=ignored[:, 1:])
    closed = end_slot < slot_count
    w, e = world[closed], ev[closed]
    ignored_slots = ignored[w, end_slot[closed], e] - ignored[w, first_slot[closed], e]
    sessions = np.zeros((world_count, ev_count))
    followed = np.zeros((world_count, ev_count))
    np.add.at(sessions, (w, e), 1.0)
    np.add.at(followed, (w, e), (ignored_slots == 0).astype(float))
    return sessions, followed


def partner_world(
    *,
    groups: Sequence[tuple[str, str, np.ndarray]],
    weights: np.ndarray,
    gross_gbp: np.ndarray,
    value_gbp_per_month: np.ndarray,
    earning: np.ndarray,
    charger_kw: np.ndarray,
    capacity_kwh: np.ndarray,
    battery_added_kwh: Mapping[str, np.ndarray],
    dispatch_sessions: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Contract §4.2: per (world, group) partner figures.

    ``weights`` (world, EV, night) are the ledger allocation weights of
    trading contract §9.3c; ``gross_gbp`` (world, night) the ``full``
    strategy's seven-bucket gross (before the customer share and penalties,
    so it includes the baseline-effect bucket, §16 Q6); ``value_gbp_per_month``
    and ``earning`` (world, EV) come from ``summaries.household_frames``
    (the pass-through customer value and "delivered turn-down in a settled
    slot"); ``charger_kw`` and ``capacity_kwh`` are per EV;
    ``battery_added_kwh`` {path: (world, EV)} is ``battery_added_kwh`` summed
    over the week.  ``dispatch_sessions`` is ``dispatch_session_counts``'s
    (sessions, followed) pair of (world, EV) counts; the group's dispatch
    success rate is its followed sessions over its sessions (session
    weighted, NaN with none).  Without it the rate is NaN, never a guess.
    """

    world_count = weights.shape[0]
    # Each EV's allocated gross: a night's gross spread by the EV's share of
    # the night's settled flexibility.  The weights sum to 1 per (world,
    # night), so the fleet's sum is the ledger's gross exactly (§4.7 id. 1).
    gross_per_ev = np.einsum("win,wn->wi", weights, gross_gbp)
    cycles = {path: battery_added_kwh[path] / capacity_kwh for path in ("normal", "selected")}
    frames = []
    for order, (group_type, group_id, members) in enumerate(groups):
        enrolled = int(members.sum())
        earning_count = earning[:, members].sum(axis=1).astype(np.int64)
        gross_week = gross_per_ev[:, members].sum(axis=1)
        per_enrolled = _per_count(gross_week * WEEKS_PER_MONTH, enrolled)
        per_earning = np.divide(
            gross_week * WEEKS_PER_MONTH,
            earning_count,
            out=np.full(world_count, np.nan),
            where=earning_count > 0,
        )
        normal = cycles["normal"][:, members].mean(axis=1)
        selected = cycles["selected"][:, members].mean(axis=1)
        frames.append(
            pd.DataFrame(
                {
                    "world_id": np.arange(world_count, dtype=np.int64),
                    "group_type": group_type,
                    "group_id": group_id,
                    "enrolled_ev_count": np.int64(enrolled),
                    "earning_ev_count": earning_count,
                    "share_earning": _per_count(earning_count, enrolled),
                    "gross_flex_gbp_per_week": gross_week,
                    "gross_flex_gbp_per_enrolled_device_per_month": per_enrolled,
                    "gross_flex_gbp_per_earning_device_per_month": per_earning,
                    "gbp_per_kw_charger_per_year": _per_count(
                        gross_week * WEEKS_PER_YEAR, float(charger_kw[members].sum())
                    ),
                    "customer_value_gbp_per_device_per_month": value_gbp_per_month[:, members].mean(
                        axis=1
                    ),
                    "dispatch_success_rate": np.full(world_count, np.nan)
                    if dispatch_sessions is None
                    else _per_count_array(
                        dispatch_sessions[1][:, members].sum(axis=1),
                        dispatch_sessions[0][:, members].sum(axis=1),
                    ),
                    "equivalent_full_cycles_per_ev_per_week_normal": normal,
                    "equivalent_full_cycles_per_ev_per_week_selected": selected,
                    "equivalent_full_cycles_per_ev_per_week_difference": selected - normal,
                    "evidence_kind": EVIDENCE_KIND,
                    "_order": order,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True).sort_values(["world_id", "_order"], kind="stable")
    return frame.reset_index(drop=True).loc[:, PARTNER_WORLD_COLUMNS]


def _per_count_array(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``values / counts`` per world, NaN where the count is 0."""

    return np.divide(values, counts, out=np.full(len(values), np.nan), where=counts > 0)


def _partner_horizon(unit: str) -> str:
    return "scenario" if ("month" in unit or "year" in unit) else "week_ahead"


def partner_summary(world: pd.DataFrame, season: str) -> pd.DataFrame:
    """Contract §4.2: per (group, metric), quantiles across worlds, groups in frame order."""

    rows = []
    for (group_type, group_id), cell in world.groupby(["group_type", "group_id"], sort=False):
        for metric, unit in PARTNER_METRIC_UNITS.items():
            rows.append(
                {
                    "group_type": group_type,
                    "group_id": group_id,
                    "metric": metric,
                    "unit": unit,
                    "horizon": _partner_horizon(unit),
                    "season": season,
                    "enrolled_ev_count": int(cell["enrolled_ev_count"].iloc[0]),
                    **_stats(cell[metric].to_numpy(dtype=float)),
                    "evidence_kind": EVIDENCE_KIND,
                }
            )
    return pd.DataFrame(rows, columns=PARTNER_SUMMARY_COLUMNS).astype(
        {"enrolled_ev_count": np.int64, "world_count": np.int64}
    )


def exceedance_per_world(values_gbp_per_month: np.ndarray) -> dict[str, np.ndarray]:
    """Per world (world, threshold) share at or above £X and floor top-up (§4.3).

    ``values_gbp_per_month`` (world, EV) are one group's customer values.
    ``floor_top_up`` is what guaranteeing £X costs per enrolled device per
    month: the mean over the group of ``max(0, X − value)``.
    """

    x = EXCEEDANCE_THRESHOLDS_GBP_PER_MONTH.astype(float)
    values = values_gbp_per_month[:, :, np.newaxis]
    return {
        "share_at_or_above": (values >= x).mean(axis=1),
        "floor_top_up_gbp_per_device_per_month": np.maximum(x - values, 0.0).mean(axis=1),
    }


def household_value_exceedance(
    groups: Sequence[tuple[str, str, np.ndarray]],
    value_gbp_per_month: np.ndarray,
    season: str,
) -> pd.DataFrame:
    """Contract §4.3: probability and cost of a guaranteed £X per month, £0–£40.

    ``value_gbp_per_month`` (world, EV) is always the pass-through customer
    value of trading contract §9.3c (saving plus the allocated customer
    share), whatever the supplier's reward mode (lead decision on Q4): the
    supplier's own reward is on Supplier P&L and never mixed in here.
    """

    units = {
        "share_at_or_above": "fraction",
        "floor_top_up_gbp_per_device_per_month": "GBP per device per month",
    }
    rows = []
    for group_type, group_id, members in groups:
        per_world = exceedance_per_world(value_gbp_per_month[:, members])
        for metric, unit in units.items():
            for index, threshold in enumerate(EXCEEDANCE_THRESHOLDS_GBP_PER_MONTH):
                rows.append(
                    {
                        "group_type": group_type,
                        "group_id": group_id,
                        "metric": metric,
                        "threshold_gbp_per_month": int(threshold),
                        "unit": unit,
                        "horizon": "scenario",
                        "season": season,
                        "enrolled_ev_count": int(members.sum()),
                        **_stats(per_world[metric][:, index]),
                        "evidence_kind": EVIDENCE_KIND,
                    }
                )
    return pd.DataFrame(rows, columns=EXCEEDANCE_COLUMNS).astype(
        {
            "threshold_gbp_per_month": np.int64,
            "enrolled_ev_count": np.int64,
            "world_count": np.int64,
        }
    )


def revenue_by_market_summary(
    full_week: pd.DataFrame, treated_ev_count: int, season: str
) -> pd.DataFrame:
    """Contract §4.5: the ``full`` strategy's weekly trading cash by market, then unmodelled ones.

    ``full_week`` is ``trading_week_world`` for strategy ``full``.  The
    modelled rows' means add up to the ``net`` mean; quantiles do not.
    Unmodelled markets (Balancing Mechanism, frequency response, Capacity
    Market, DNO services) are listed with NaN statistics and
    ``modelled = False``: not modelled is not zero (decision 0005).
    """

    ordered = full_week.sort_values("world_id")
    rows = []
    for market_id, buckets in MARKET_BUCKETS.items():
        per_week = ordered[list(buckets)].to_numpy(dtype=float).sum(axis=1)
        for metric, values, unit, horizon in (
            ("gbp_per_week", per_week, "GBP per week", "week_ahead"),
            (
                "gbp_per_enrolled_device_per_month",
                _per_count(per_week * WEEKS_PER_MONTH, treated_ev_count),
                "GBP per device per month",
                "scenario",
            ),
        ):
            rows.append(
                {
                    "market": market_id,
                    "modelled": True,
                    "metric": metric,
                    "unit": unit,
                    "horizon": horizon,
                    "season": season,
                    **_stats(values),
                    "evidence_kind": EVIDENCE_KIND,
                }
            )
    for market_id in UNMODELLED_MARKETS:
        for metric, unit, horizon in (
            ("gbp_per_week", "GBP per week", "week_ahead"),
            ("gbp_per_enrolled_device_per_month", "GBP per device per month", "scenario"),
        ):
            rows.append(
                {
                    "market": market_id,
                    "modelled": False,
                    "metric": metric,
                    "unit": unit,
                    "horizon": horizon,
                    "season": season,
                    **_stats(np.array([])),
                    "evidence_kind": EVIDENCE_KIND,
                }
            )
    return pd.DataFrame(rows, columns=REVENUE_BY_MARKET_COLUMNS).astype(
        {"modelled": bool, "world_count": np.int64}
    )


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


def build_frames(
    trading,
    per_ev: Mapping[str, np.ndarray],
    weights: np.ndarray,
    *,
    cost_effect: pd.DataFrame,
    units: pd.DataFrame,
    study_slots: pd.DataFrame,
    cohort_ids: Sequence[str],
    battery_added: Mapping[str, np.ndarray],
    forecast_prices: pd.DataFrame,
    price_assumptions: Mapping[str, object],
    trading_assumptions: Mapping[str, float],
    supplier_inputs: Mapping[str, float],
    warmup_slot_count: int,
    dispatch_sessions: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, pd.DataFrame]:
    """Every ``FRAME_FIELDS`` frame of one action run with the trading overlay.

    Inputs: ``trading`` (``market.TradingRun`` or anything with its
    ``deviation_world_slot``, ``trading_ledger_world`` and
    ``trading_week_world`` frames); the run's ``cost_effect``, ``units`` and
    ``study_slots``; ``cohort_ids`` in source order; ``per_ev`` and
    ``weights`` from ``summaries.household_frames`` and
    ``allocation_weights``; ``battery_added`` {path: (world, EV) kWh} from
    ``battery_added_kwh``; ``forecast_prices`` (its ``system_net_demand_gw``
    feeds the carbon intensity); ``price_assumptions`` (``summer_months`` and
    ``supply_reference_net_demand_gw``); ``trading_assumptions``
    (``trading.customer_revenue_share``); ``supplier_inputs`` (the
    ``SUPPLIER`` and ``CARBON`` values, ``assumptions.supplier_inputs``); and
    the warm-up length in slots (for the season); ``dispatch_sessions``
    the (sessions, followed) counts of ``dispatch_session_counts`` (``None``
    without the kernel's ``plan_status``).  Deterministic: no draw.
    """

    deviation = trading.deviation_world_slot
    ledger, week = trading.trading_ledger_world, trading.trading_week_world
    records = {
        **supplier_inputs,
        "trading.customer_revenue_share": trading_assumptions["trading.customer_revenue_share"],
    }
    season = season_of_run(study_slots, warmup_slot_count, price_assumptions["summer_months"])
    world_count = int(deviation["world_id"].nunique())

    def slots(column: str) -> np.ndarray:
        return _matrix(deviation, column, world_count)

    unmanaged, metered = slots("unmanaged_kwh"), slots("metered_kwh")
    treated_count = int((~_control(units)).sum())
    full_week = week.loc[week["strategy"].eq("full")].sort_values("world_id")
    full_nights = ledger.loc[ledger["strategy"].eq("full")].sort_values(["world_id", "night_index"])
    gross = full_nights[list(GROSS_BUCKETS)].to_numpy(dtype=float).sum(axis=1)
    pnl = supplier_pnl_world(
        unmanaged_kwh=unmanaged,
        metered_kwh=metered,
        true_reduction_kwh=slots("true_reduction_kwh"),
        expected_unmanaged_kwh=slots("expected_unmanaged_kwh"),
        expected_metered_kwh=slots("expected_metered_kwh"),
        day_ahead_gbp_per_mwh=slots("day_ahead_gbp_per_mwh"),
        imbalance_gbp_per_mwh=slots("imbalance_gbp_per_mwh"),
        # Exact reuse of the cost-effect column (§3.5 identity 1), so the
        # Value and risk lens and this one cannot disagree.
        energy_saving_gbp=-cost_effect.sort_values("world_id")[
            "illustrative_selected_minus_normal_energy_cost_gbp"
        ].to_numpy(dtype=float),
        grid_event_payment_gbp=full_week["grid_event_payment_gbp"].to_numpy(dtype=float),
        treated_ev_count=treated_count,
        records=records,
    )
    blocks = hedge_block_world(unmanaged, metered, study_slots)
    intensity = carbon_intensity_gco2_per_kwh(
        _matrix(forecast_prices, "system_net_demand_gw", world_count),
        supplier_inputs,
        float(price_assumptions["supply_reference_net_demand_gw"]),
    )
    carbon = carbon_shift_world(unmanaged, metered, intensity, treated_count)
    groups = partner_groups(units, cohort_ids)
    partners = partner_world(
        groups=groups,
        weights=weights,
        gross_gbp=gross.reshape(world_count, -1),
        value_gbp_per_month=per_ev["value_gbp_per_month"],
        earning=per_ev["earning"],
        charger_kw=units["home_charger_limit_kw"].to_numpy(dtype=float),
        capacity_kwh=units["physical_capacity_kwh"].to_numpy(dtype=float),
        battery_added_kwh=battery_added,
        dispatch_sessions=dispatch_sessions,
    )
    return {
        "supplier_pnl_world": pnl,
        "supplier_pnl_summary": supplier_pnl_summary(pnl, season),
        "hedge_block_world": blocks,
        "hedge_block_summary": hedge_block_summary(blocks, season),
        "carbon_shift_world": carbon,
        "carbon_shift_summary": carbon_shift_summary(carbon, season),
        "partner_world": partners,
        "partner_summary": partner_summary(partners, season),
        "household_value_exceedance": household_value_exceedance(
            groups, per_ev["value_gbp_per_month"], season
        ),
        "revenue_by_market_summary": revenue_by_market_summary(full_week, treated_count, season),
    }
