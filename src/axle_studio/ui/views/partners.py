"""Supplier ▸ 5 Partners: what a charger maker's enrolled devices earn, and how sure that is.

Reads (supplier contract v1 §11): ``partner_summary`` (and the fleet row of
``partner_world`` for the months-to-fund figure), ``revenue_by_market_summary``,
``household_value_exceedance``, ``charge_completion_summary`` and
``firmness_by_manufacturer`` (firm-MW frames, shown when present),
``session_distribution_bands`` (the two departure-SoC metrics, §4.6) and,
in the growth expander, ``supplier_pnl_summary`` for the payout fan.

The growth expander calls ``model/growth.py`` (§5): pure arithmetic on the
reader's own lever values, which are widget state and never enter a run,
so editing them does not make the result stale. Its placeholders
(100,000 eligible, 50 % invited, 30 % signed up, 80 % active, 20 % take,
£100 discount) are the lead's accepted §16 Q7 values, labelled illustrative.
The supplier's platform fee (a supplier cost to the platform, decision 0006)
is never fed into this calculator's annual figure: it is a charger maker's
income here, and payer and payee must not be swapped (final critique Q-8).
A partner-side fee lever does not exist yet.

Partner cash is the ledger's gross flexibility cash before the customer
share and penalties, allocated by each EV's share of settled flexibility; it
includes the baseline-effect bucket (§4.1, §16 Q6). Customer value is the
pass-through reading. The supplier's own P&L is on Supplier P&L; the three
are shown side by side, never added.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from axle_studio.model import growth

from ..components import header
from ..components.chart_table import chart_block
from ..components.kpi import badge_line, kpi, kpi_columns
from ..style import (
    CHART_FONT_PX,
    CHART_HEIGHTS,
    MUTED_INK,
    PATH_LABELS,
    PATH_STYLES,
    SERIES_COLOURS,
    UNAVAILABLE,
    assumption_value,
    band_and_line,
    format_quantity,
    money,
    percent,
    present_paths,
)
from .supplier_common import (
    EXTRAPOLATED,
    band_text,
    cohort_labels,
    evidence_caption,
    group_label,
    horizon_label,
    missing,
    p50_text,
    stat_row,
)

GROUP_KEY = "partners-group"
"""The lens's own group selector: it offers makers, which the shared group control does not."""

THRESHOLD_DEFAULT = 10
"""The exceedance view's default £ per month (a reporting choice, supplier contract v1 §4.3)."""

# Supplier contract v1 §16 Q7, placeholders accepted by the lead (overnight
# review log, 29 Sep): no source in this project, so each is labelled
# illustrative on screen and is the reader's to change.
FUNNEL_DEFAULTS = {
    "eligible": 100_000,
    "invited_percent": 50.0,
    "signed_up_percent": 30.0,
    "active_percent": 80.0,
    "take_percent": 20.0,
    "discount_gbp": 100.0,
}
TORNADO_SWING = 0.3
"""±30 % on each lever (supplier contract v1 §5)."""
PAYOUT_MONTHS = 12

_STAGE_LABELS = {
    "eligible": "Eligible",
    "invited": "Invited",
    "signed_up": "Signed up",
    "active": "Active",
    "earning": "Earning",
}
_MARKET_LABELS = {
    "day_ahead": "Day-ahead",
    "intraday": "Intraday",
    "imbalance": "Imbalance",
    "grid_events": "Grid events",
    "baseline_effect": "Baseline effect",
    "deductions": "Deductions",
    "net": "Net",
    "balancing_mechanism": "Balancing Mechanism",
    "frequency_response": "Frequency response",
    "capacity_market": "Capacity market",
    "dno_services": "DNO services",
}

GROSS_NOTE = (
    "Gross flexibility cash before the customer share and penalties. It includes the baseline "
    "effect: settlement value that comes from where the baseline sits, not from any one EV's "
    "flexibility. Shared out by each EV's share of settled flexibility."
)


def _partner(result: Any, group: str, metric: str) -> pd.Series | None:
    return stat_row(result.partner_summary, group_id=group, metric=metric)


def _pounds2(value: float) -> str:
    return money(value, decimals=2, signed=True)


def _share(value: float) -> str:
    return percent(100 * value)


# --- Tiles ------------------------------------------------------------------


def _fleet_net_per_device(result: Any) -> pd.Series | None:
    """The fleet's net £ per enrolled device per month, from ``revenue_by_market_summary``'s
    "net" market row (the trading ledger's net: day-ahead, intraday, imbalance, grid events
    and the baseline effect, less the customer share, supplier compensation and unmet-charge
    penalty). Fleet-wide only: the model does not split the ledger's net by maker, so unlike
    the other tiles here this figure does not change with the Group selector (B-12)."""

    markets = getattr(result, "revenue_by_market_summary", None)
    if markets is None:
        return None
    return stat_row(markets, market="net", metric="gbp_per_enrolled_device_per_month")


def _render_tiles(st: Any, result: Any, group: str) -> None:
    net = _fleet_net_per_device(result)
    enrolled = _partner(result, group, "gross_flex_gbp_per_enrolled_device_per_month")
    earning_share = _partner(result, group, "share_earning")
    per_kw = _partner(result, group, "gbp_per_kw_charger_per_year")
    first = kpi_columns(st, 4)
    kpi(
        first[0],
        "Net cash per device",
        p50_text(net, _pounds2),
        unit="/month",
        context=band_text(net, _pounds2) or "Fleet-wide, after every deduction",
        help="The trading ledger's net per enrolled device per month, fleet-wide: gross cash "
        "less the customer share, supplier compensation and unmet-charge penalty. P50 across "
        f"simulated weeks; {horizon_label('scenario')}. The model does not split this net by "
        "maker, so the tile does not change with the group.",
    )
    kpi(
        first[1],
        "Gross cash, this group",
        p50_text(enrolled, _pounds2),
        unit="/month",
        context=band_text(enrolled, _pounds2) or GROSS_NOTE,
        help=f"{GROSS_NOTE} P50 across simulated weeks; {horizon_label('scenario')}.",
    )
    kpi(
        first[2],
        "Devices earning",
        p50_text(earning_share, _share),
        context=band_text(earning_share, _share),
        help="Share of enrolled devices that turned down in at least one settled half-hour "
        f"of the week; {horizon_label('week_ahead')}.",
    )
    kpi(
        first[3],
        "Per unit of charger power",
        p50_text(per_kw, lambda v: money(v, signed=True)),
        unit="/kW/year",
        context=band_text(per_kw, lambda v: money(v, signed=True)),
        help=f"Gross cash per kW of installed home charger power per year, {EXTRAPOLATED}.",
    )


def _render_secondary_tiles(st: Any, result: Any, group: str) -> None:
    """The four tiles moved off the main surface to make room for the net headline (Q-13)."""

    per_earning = _partner(result, group, "gross_flex_gbp_per_earning_device_per_month")
    value = _partner(result, group, "customer_value_gbp_per_device_per_month")
    dispatch = _partner(result, group, "dispatch_success_rate")
    cycles = _partner(result, group, "equivalent_full_cycles_per_ev_per_week_selected")
    cycles_change = _partner(result, group, "equivalent_full_cycles_per_ev_per_week_difference")
    second = kpi_columns(st, 4)
    kpi(
        second[0],
        "Per earning device",
        p50_text(per_earning, _pounds2),
        unit="/month",
        context=band_text(per_earning, _pounds2),
        help=f"The group's gross cash over its earning devices. {GROSS_NOTE} "
        "Unavailable in a week with no earning device.",
    )
    kpi(
        second[1],
        "Customer value per device",
        p50_text(value, _pounds2),
        unit="/month",
        context="As if day-ahead prices reached the customer",
        help="The pass-through reading: the average customer's saving plus its customer "
        "revenue share, as if the day-ahead price reached the customer. Shown beside the "
        "partner cash, never added to it.",
    )
    kpi(
        second[2],
        "Dispatch success",
        p50_text(dispatch, _share),
        context=band_text(dispatch, _share) or "Plan status not recorded in this run",
        help="Share of smart-path sessions that followed their plan in every connected "
        "half-hour; Unavailable until the run records plan status.",
    )
    change = (
        ""
        if cycles_change is None or missing(cycles_change["p50"])
        else f"change vs unmanaged {format_quantity(cycles_change['p50'], '', decimals=3)}, "
        "paired per week"
    )
    kpi(
        second[3],
        "Battery cycles, smart",
        p50_text(cycles, lambda v: format_quantity(v, "", decimals=2)),
        unit="/week",
        context=change,
        help="Equivalent full cycles per EV per week on the smart path; the paired change "
        "should be small, because smart charging moves energy in time, not amount.",
    )


# --- Revenue by market ----------------------------------------------------------


def _market_figure(rows: pd.DataFrame) -> go.Figure:
    # Horizontal so the market names fit at 390 px (see Supplier P&L's waterfall).
    measures = ["total" if market == "net" else "relative" for market in rows["market"]]
    figure = go.Figure(
        go.Waterfall(
            orientation="h",
            y=[_MARKET_LABELS.get(m, m) for m in rows["market"]],
            x=rows["mean"],
            measure=measures,
            name="Revenue by market",
            increasing={"marker": {"color": SERIES_COLOURS["normal"]}},
            decreasing={"marker": {"color": SERIES_COLOURS["flag"]}},
            totals={"marker": {"color": SERIES_COLOURS["difference"]}},
            connector={"line": {"color": MUTED_INK, "width": 1}},
            hovertemplate="%{y}: £%{x:,.2f} per device per month (mean)<extra></extra>",
        )
    )
    figure.update_yaxes(
        title={"text": "market, £ per device", "standoff": 12}, autorange="reversed"
    )
    figure.update_xaxes(title="£ per enrolled device per month, mean (illustrative)")
    figure.update_layout(showlegend=False, hovermode="closest")
    return figure


# --- Exceedance -------------------------------------------------------------------


def _exceedance_figure(rows: pd.DataFrame, threshold: int) -> go.Figure:
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)
    share = rows.loc[rows["metric"].eq("share_at_or_above")].sort_values("threshold_gbp_per_month")
    floor = rows.loc[rows["metric"].eq("floor_top_up_gbp_per_device_per_month")].sort_values(
        "threshold_gbp_per_month"
    )
    band_and_line(
        figure,
        share["threshold_gbp_per_month"],
        share["p10"] * 100.0,
        share["p50"] * 100.0,
        share["p90"] * 100.0,
        name="Customers at or above",
        colour=SERIES_COLOURS["normal"],
        hovertemplate="£%{x}/month: %{y:.0f}% of customers (P50)<extra></extra>",
        row=1,
        col=1,
    )
    band_and_line(
        figure,
        floor["threshold_gbp_per_month"],
        floor["p10"],
        floor["p50"],
        floor["p90"],
        name="Cost of the floor",
        colour=SERIES_COLOURS["difference"],
        hovertemplate="£%{x}/month: £%{y:,.2f} per device per month (P50)<extra></extra>",
        row=2,
        col=1,
    )
    for row in (1, 2):
        figure.add_vline(
            x=threshold, line_color=SERIES_COLOURS["observed"], line_dash="dash", row=row, col=1
        )
    figure.update_yaxes(title_text="% of customers", rangemode="tozero", row=1, col=1)
    figure.update_yaxes(title_text="£ per device per month", rangemode="tozero", row=2, col=1)
    figure.update_xaxes(title_text="Guaranteed £ per household per month", row=2, col=1)
    figure.update_layout(hovermode="x unified")
    return figure


# --- Departure SoC small multiples (§4.6) ----------------------------------------

_DEPARTURE_METRICS: dict[str, str] = {
    "normal": "departure_soc_percent",
    "timed": "departure_soc_percent_timed",
    "selected": "departure_soc_percent_smart",
}
"""Each policy's own metric name in ``session_distribution_bands`` (decision 0007)."""


def _departure_paths(metrics: pd.Series) -> list[str]:
    """Policies present in a ``session_distribution_bands`` slice, in display order."""

    present = set(metrics)
    return present_paths(path for path, metric in _DEPARTURE_METRICS.items() if metric in present)


def _departure_figure(bands: pd.DataFrame) -> go.Figure:
    """Departure SoC small multiples: one subplot per policy ``bands`` carries (§4.6).

    Gains a Timed tariff panel (decision 0007) whenever ``bands`` carries
    ``departure_soc_percent_timed``; absent, exactly the two panels as before.
    """

    paths = _departure_paths(bands["metric"])
    figure = make_subplots(
        rows=1,
        cols=len(paths),
        shared_yaxes=True,
        subplot_titles=[PATH_LABELS[path] for path in paths],
        horizontal_spacing=0.06,
    )
    y_top = 1.0
    for col, path_id in enumerate(paths, start=1):
        metric = _DEPARTURE_METRICS[path_id]
        rows = bands.loc[bands["metric"].eq(metric)].sort_values("bin_index")
        lower = rows["bin_lower"].to_numpy(dtype=float)
        upper = rows["bin_upper"].to_numpy(dtype=float)
        p10, p50, p90 = (
            rows[stat].to_numpy(dtype=float) * 100.0
            for stat in ("share_p10", "share_p50", "share_p90")
        )
        if len(p90) and np.isfinite(p90).any():
            y_top = max(y_top, float(np.nanmax(p90)))
        figure.add_trace(
            go.Bar(
                x=(lower + upper) / 2.0,
                y=p50,
                width=(upper - lower) * 0.85,
                name=PATH_LABELS[path_id],
                marker_color=PATH_STYLES[path_id]["color"],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": np.maximum(p90 - p50, 0.0),
                    "arrayminus": np.maximum(p50 - p10, 0.0),
                    "color": MUTED_INK,
                    "thickness": 1,
                    "width": 0,
                },
                customdata=np.stack([rows["bin_label"], p10, p90], axis=-1),
                hovertemplate="%{customdata[0]}: %{y:.1f}% of sessions (P50; P10–P90 "
                "%{customdata[1]:.1f}–%{customdata[2]:.1f}%)<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    figure.update_xaxes(title_text="SoC at departure (%)", range=[0, 100])
    figure.update_yaxes(title_text="% of sessions", col=1)
    figure.update_yaxes(range=[0.0, y_top * 1.05])
    figure.update_layout(bargap=0, hovermode="closest", margin={"t": 28})
    figure.update_annotations(font_size=CHART_FONT_PX)
    return figure


# --- Growth expander --------------------------------------------------------------


def _number(st: Any, label: str, key: str, default: float, **kwargs: Any) -> float:
    value = st.number_input(label, value=default, key=key, **kwargs)
    return default if value is None else float(value)


def _render_growth(st: Any, result: Any) -> None:
    fleet_share = _partner(result, "fleet", "share_earning")
    fleet_gross = _partner(result, "fleet", "gross_flex_gbp_per_enrolled_device_per_month")
    if fleet_gross is None or missing(fleet_gross["p50"]):
        st.caption("The growth calculator needs the fleet's gross cash per device from the run.")
        return
    st.caption(
        "Illustrative. These levers are yours to set; the defaults are placeholders with no source."
    )
    columns = st.columns(3)
    eligible = _number(
        columns[0],
        "Eligible devices",
        "partners-eligible",
        FUNNEL_DEFAULTS["eligible"],
        min_value=0,
        step=1_000,
    )
    invited = _number(
        columns[1],
        "Invited (%)",
        "partners-invited",
        FUNNEL_DEFAULTS["invited_percent"],
        min_value=0.0,
        max_value=100.0,
        step=5.0,
    )
    signed_up = _number(
        columns[2],
        "Signed up (%)",
        "partners-signed-up",
        FUNNEL_DEFAULTS["signed_up_percent"],
        min_value=0.0,
        max_value=100.0,
        step=5.0,
    )
    columns = st.columns(3)
    active = _number(
        columns[0],
        "Active (%)",
        "partners-active",
        FUNNEL_DEFAULTS["active_percent"],
        min_value=0.0,
        max_value=100.0,
        step=5.0,
    )
    take = _number(
        columns[1],
        "Partner take (%)",
        "partners-take",
        FUNNEL_DEFAULTS["take_percent"],
        min_value=0.0,
        max_value=100.0,
        step=5.0,
    )
    discount = _number(
        columns[2],
        "Sticker discount (£)",
        "partners-discount",
        FUNNEL_DEFAULTS["discount_gbp"],
        min_value=0.0,
        step=10.0,
    )
    # The supplier's platform fee is a supplier cost paid to the platform
    # (decision 0006), not income to the device maker, so it plays no part
    # in partner revenue here (final critique Q-8: payer and payee did not
    # match). A partner-side per-device fee would need its own lever
    # (supplier contract v1 §5, §16 Q7), which does not exist yet.
    fee_value = 0.0

    share_earning = (
        0.0 if fleet_share is None or missing(fleet_share["p50"]) else float(fleet_share["p50"])
    )
    stages = growth.funnel(
        eligible=eligible,
        invited_share=invited / 100.0,
        signed_up_share=signed_up / 100.0,
        active_share=active / 100.0,
        share_earning=share_earning,
    )
    active_count = float(stages.loc[stages["stage"].eq("active"), "count"].iloc[0])
    take_rate = take / 100.0
    # The band: annual_gbp at the run's own P10/P50/P90 of £ per enrolled
    # device. A positive constant times a per-week value plus a constant keeps
    # its quantiles exactly (supplier contract v1 §5), so no resampling.
    annual = {
        stat: growth.annual_gbp(
            enrolled_devices=active_count,
            gbp_per_enrolled_device_per_month=float(fleet_gross[stat]),
            take_rate=take_rate,
            fee_gbp_per_device_per_month=fee_value,
        )
        for stat in ("p10", "p50", "p90")
    }
    world = result.partner_world
    fleet_world = world.loc[world["group_id"].eq("fleet")].sort_values("world_id")
    months = growth.months_to_fund_discount(
        discount,
        gbp_per_enrolled_device_per_month_by_world=fleet_world[
            "gross_flex_gbp_per_enrolled_device_per_month"
        ].to_numpy(dtype=float),
        take_rate=take_rate,
    )
    tiles = kpi_columns(st, 3)
    kpi(
        tiles[0],
        "Active devices",
        f"{active_count:,.0f}",
        context=f"{stages['count'].iloc[-1]:,.0f} earning at the run's share",
        help="Eligible × invited × signed up × active (your inputs); earning uses the run's own "
        "share of devices earning.",
    )
    kpi(
        tiles[1],
        "Partner revenue per year",
        money(annual["p50"]),
        context=f"P10 {money(annual['p10'])} · P90 {money(annual['p90'])}",
        help="Active devices × gross £ per enrolled device per month × take × 12. No platform "
        "fee comes in: that fee is a cost the supplier pays, not partner income. The band "
        "comes from the simulated weeks' own £ per enrolled device. "
        f"{EXTRAPOLATED.capitalize()}.",
    )
    p10_months, p50_months, p90_months, funded_weeks = months
    kpi(
        tiles[2],
        "Months to fund the discount",
        UNAVAILABLE if missing(p50_months) else f"{p50_months:,.1f}",
        unit=None if missing(p50_months) else "months",
        context=""
        if missing(p50_months)
        else f"P10 {p10_months:,.1f} · P90 {p90_months:,.1f}; {funded_weeks} of "
        f"{len(fleet_world)} weeks fund it",
        help=f"Discount £{discount:,.0f} per device. Per simulated week: the discount ÷ "
        "(gross £ per enrolled device per month × take); weeks where that revenue is not "
        "positive never fund it and are left out.",
    )

    chart_block(
        st,
        _funnel_figure(stages),
        title="From eligible devices to devices that earn",
        caption="Your inputs down to active; earning uses the run's share of devices earning. "
        "Illustrative.",
        frame=stages.assign(stage=stages["stage"].map(_STAGE_LABELS)).rename(
            columns={"stage": "Stage", "count": "Devices", "share_of_eligible": "Share of eligible"}
        ),
        definition="Each stage is a share of the one before it. Sign-up time, opt-outs and "
        "retention are not modelled; they are yours to set.",
        height=CHART_HEIGHTS["diagram"],
        key="partners-funnel",
    )

    levers = growth.tornado(
        enrolled_devices=active_count,
        gbp_per_enrolled_device_per_month=float(fleet_gross["p50"]),
        take_rate=take_rate,
        fee_gbp_per_device_per_month=fee_value,
        swing=TORNADO_SWING,
    )
    clipped = " Take clipped to 0–100%." if levers["clipped"].any() else ""
    chart_block(
        st,
        _tornado_figure(levers, annual["p50"]),
        title="Which lever moves the annual figure most, ±30% each",
        caption=f"Each lever moved alone, the others held at your values.{clipped} Illustrative. "
        # Revenue is proportional to each of these three with no fee term
        # (Q-8: a fee is never fed in here), so the swings are equal by
        # construction; say so rather than let it look odd.
        "No fee lever, so the three swings are equal.",
        frame=levers.drop(columns=["lever"]).rename(
            columns={
                "label": "Lever",
                "value": "Your value",
                "low_value": "Low",
                "high_value": "High",
                "annual_gbp_low": "£ per year, low",
                "annual_gbp_high": "£ per year, high",
                "clipped": "Clipped",
            }
        ),
        definition="Annual £ = active devices × £ per enrolled device per month × take × 12, "
        "worked out again with one lever at 0.7× and 1.3× its value. There is no fee lever: "
        "the platform fee is a cost the supplier pays, not partner income.",
        height=CHART_HEIGHTS["diagram"],
        key="partners-tornado",
    )

    payment = stat_row(
        result.supplier_pnl_summary,
        hedge_variant="profiled",
        metric="customer_payment_per_customer_per_month_gbp",
    )
    if payment is None or missing(payment["p50"]):
        st.caption("Payout fan unavailable: the customer payment is unset (flat reward not set).")
        return
    # A payment is stored negative (the supplier's view); as a payout its sign
    # flips, so the payout's P10 is minus the payment's P90.
    per_customer = growth.cumulative_payout_fan(
        monthly_gbp_p10=-float(payment["p90"]),
        monthly_gbp_p50=-float(payment["p50"]),
        monthly_gbp_p90=-float(payment["p10"]),
        months=PAYOUT_MONTHS,
    )
    fleet = per_customer.assign(
        **{
            column: per_customer[column] * active_count
            for column in ("cumulative_p10", "cumulative_p50", "cumulative_p90")
        }
    )
    chart_block(
        st,
        _payout_figure(per_customer, fleet),
        title="Customer payouts over 12 months, per customer and for your active devices",
        caption="A scenario that repeats the simulated week, not a forecast. Months taken as "
        "fully correlated, so the band is wide.",
        frame=per_customer.merge(fleet, on="month", suffixes=(" per customer", " all active")),
        definition="The supplier's customer payment per customer per month (profiled hedge), "
        "P10/P50/P90 across weeks, times the month number; the fleet fan multiplies by the "
        "funnel's active devices. Independent months would give a narrower band.",
        height=CHART_HEIGHTS["small_multiple_panel"] + 80,
        key="partners-payout-fan",
    )


def _funnel_figure(stages: pd.DataFrame) -> go.Figure:
    labels = [_STAGE_LABELS[s] for s in stages["stage"]]
    figure = go.Figure(
        go.Bar(
            x=stages["count"],
            y=labels,
            orientation="h",
            name="Devices",
            marker_color=SERIES_COLOURS["normal"],
            hovertemplate="%{y}: %{x:,.0f} devices<extra></extra>",
        )
    )
    figure.update_yaxes(title={"text": "EVs at each stage", "standoff": 12}, autorange="reversed")
    figure.update_xaxes(title="EVs (devices)", rangemode="tozero")
    figure.update_layout(showlegend=False, hovermode="closest")
    return figure


def _tornado_figure(levers: pd.DataFrame, base: float) -> go.Figure:
    low = levers[["annual_gbp_low", "annual_gbp_high"]].min(axis=1)
    high = levers[["annual_gbp_low", "annual_gbp_high"]].max(axis=1)
    figure = go.Figure(
        go.Bar(
            x=high - low,
            base=low,
            y=levers["label"],
            orientation="h",
            name="Annual £ at ±30%",
            marker_color=SERIES_COLOURS["difference"],
            customdata=np.stack([low, high], axis=-1),
            hovertemplate="%{y}: £%{customdata[0]:,.0f} to £%{customdata[1]:,.0f} per year"
            "<extra></extra>",
        )
    )
    figure.add_vline(x=base, line_color=SERIES_COLOURS["observed"], line_dash="dash")
    figure.update_yaxes(title={"text": "lever (£ per year)", "standoff": 12}, autorange="reversed")
    figure.update_xaxes(title="£ per year (illustrative)")
    figure.update_layout(showlegend=False, hovermode="closest")
    return figure


def _payout_figure(per_customer: pd.DataFrame, fleet: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Per customer", "All active devices"],
        horizontal_spacing=0.14,
    )
    for col, frame, name in ((1, per_customer, "Per customer"), (2, fleet, "All active")):
        band_and_line(
            figure,
            frame["month"],
            frame["cumulative_p10"],
            frame["cumulative_p50"],
            frame["cumulative_p90"],
            name=name,
            colour=SERIES_COLOURS["difference"],
            showlegend=False,
            hovertemplate="Month %{x}: £%{y:,.0f} (P50)<extra></extra>",
            row=1,
            col=col,
        )
    figure.update_xaxes(title_text="month")
    figure.update_yaxes(title_text="£ paid out, cumulative", col=1)
    figure.update_yaxes(title_text="£ paid out, cumulative", col=2)
    figure.update_layout(margin={"t": 28}, hovermode="closest")
    figure.update_annotations(font_size=CHART_FONT_PX)
    return figure


# --- The lens ---------------------------------------------------------------------


def _groups(result: Any) -> list[str]:
    return list(dict.fromkeys(result.partner_summary["group_id"]))


def render_partners(st: Any, result: Any) -> None:
    """Render Supplier ▸ 5 Partners for one group: the net/gross headline tiles and the
    markets chart as the hero, then three detail tabs (this group's other tiles, the
    guarantee's cost, completion and departure SoC) and the growth expander (Q-13: 2,260 px
    and 11 tiles read as buried, not answered)."""

    labels = cohort_labels(result)
    groups = _groups(result)
    group = header.controls(st).selectbox(
        "Group",
        options=groups,
        format_func=lambda g: group_label(g, labels),
        key=GROUP_KEY,
        label_visibility="collapsed",
    )
    group = group if group in groups else "fleet"
    name = group_label(group, labels)
    weeks = result.world_count
    evidence = evidence_caption(result)

    _render_tiles(st, result, group)
    share = assumption_value(result, "trading.customer_revenue_share")
    take = st.session_state.get("partners-take", FUNNEL_DEFAULTS["take_percent"])
    badge_line(
        st,
        "Beside the cash:",
        f"take {percent(float(take))} · customer share "
        f"{UNAVAILABLE if share is None else percent(100 * float(share))}",
        help="The partner take is your lever in the growth calculator below, not a run input. "
        "The customer revenue share is the trading record the run used.",
    )
    st.caption(
        f"Gross cash before the customer share and penalties; includes the baseline effect. "
        f"{evidence}",
        help=GROSS_NOTE,
    )

    markets = result.revenue_by_market_summary
    per_device = markets.loc[
        markets["metric"].eq("gbp_per_enrolled_device_per_month") & markets["modelled"]
    ]
    unmodelled = markets.loc[~markets["modelled"]].drop_duplicates("market")["market"]
    chart_block(
        st,
        _market_figure(per_device),
        title="Fleet revenue by market, mean £ per enrolled device per month",
        caption="Means add to the net; quantiles do not. Not modelled, not £0: "
        + ", ".join(_MARKET_LABELS.get(m, m) for m in unmodelled)
        + ".",
        frame=per_device.assign(market=per_device["market"].map(_MARKET_LABELS))[
            ["market", "world_count", "mean", "p10", "p50", "p90"]
        ].rename(
            columns={
                "market": "Market",
                "world_count": "Simulated weeks",
                "mean": "Mean (£/device/month)",
                "p10": "P10",
                "p50": "P50",
                "p90": "P90",
            }
        ),
        definition=(
            "Fleet-level ledger buckets by market (full strategy), per enrolled device per month, "
            f"{EXTRAPOLATED}. The baseline effect is not a market and is shown apart. Deductions "
            "are the customer share, supplier compensation and unmet-charge penalty."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="partners-markets",
    )
    grid_events = per_device.loc[per_device["market"].eq("grid_events"), "mean"]
    if len(grid_events) and float(grid_events.iloc[0]) == 0.0:
        st.caption("Grid events £0: none paid out this run (Edit assumptions ▸ Events).")

    more, guarantee, completion_tab = st.tabs(
        ["More on this group", "Guarantee cost", "Completion and departure SoC"]
    )
    # Each helper takes the plain ``st`` module (entered through ``with
    # tab:``), never the tab object itself: chart_block's own nested
    # ``with st.expander(...)`` only redirects later same-object calls when
    # ``st`` is the module Streamlit's context stack tracks, not a tab
    # handle held as a local variable (verified against AppTest).
    with more:
        _render_secondary_tiles(st, result, group)
    with guarantee:
        _render_guarantee(st, result, group, name, weeks)
    with completion_tab:
        _render_completion(st, result, group)
        _render_departure(st, result, group, name)

    with st.expander("Growth calculator (illustrative)"):
        _render_growth(st, result)


def _render_guarantee(st: Any, result: Any, group: str, name: str, weeks: int) -> None:
    """The guaranteed-£-per-household slider, its two tiles and the exceedance chart
    (moved off the main surface, Q-13)."""

    exceedance = result.household_value_exceedance
    rows = exceedance.loc[exceedance["group_id"].eq(group)]
    threshold = st.slider(
        "Guaranteed £ per household per month",
        min_value=0,
        max_value=40,
        value=THRESHOLD_DEFAULT,
        step=1,
        key="partners-threshold",
    )
    threshold = int(threshold) if threshold is not None else THRESHOLD_DEFAULT
    share_row = stat_row(rows, metric="share_at_or_above", threshold_gbp_per_month=threshold)
    floor_row = stat_row(
        rows, metric="floor_top_up_gbp_per_device_per_month", threshold_gbp_per_month=threshold
    )
    left, right = kpi_columns(st, 2)
    kpi(
        left,
        "Customers at the guarantee",
        p50_text(share_row, _share),
        context=band_text(share_row, _share),
        help="Share of the group's customers whose monthly value is at or above the "
        "guarantee, P50 across weeks. Value is the pass-through reading: as if day-ahead "
        "prices reached the customer.",
    )
    kpi(
        right,
        "Cost of the guarantee",
        p50_text(floor_row, _pounds2),
        unit="/device/month",
        context=band_text(floor_row, _pounds2),
        help="What topping every customer up to the guarantee costs, averaged over enrolled "
        "devices, P50 across weeks.",
    )
    chart_block(
        st,
        _exceedance_figure(rows, threshold),
        title=f"Probability and cost of a guaranteed £ per month, {name.lower()}",
        caption=f"P50 with P10–P90 across {weeks} weeks; scaled from one week, not a forecast. "
        "Pass-through value; excludes the supplier's reward.",
        frame=rows.assign(
            metric=rows["metric"].map(
                {
                    "share_at_or_above": "Share of customers at or above",
                    "floor_top_up_gbp_per_device_per_month": "Floor cost, £ per device per month",
                }
            )
        )[["metric", "threshold_gbp_per_month", "p10", "p50", "p90"]].rename(
            columns={
                "metric": "Measure",
                "threshold_gbp_per_month": "Guarantee (£/month)",
                "p10": "P10",
                "p50": "P50",
                "p90": "P90",
            }
        ),
        definition="Customer value under the pass-through reading (saving plus the trading "
        "product's customer share); the supplier's reward is on Supplier P&L and is not "
        "included here.",
        height=CHART_HEIGHTS["time_series"],
        key="partners-exceedance",
    )


def _render_completion(st: Any, result: Any, group: str) -> None:
    completion = getattr(result, "charge_completion_summary", None)
    all_row = stat_row(completion, group_id=group, path_id="selected", departure="all")
    early_row = stat_row(completion, group_id=group, path_id="selected", departure="early")

    def share(row: pd.Series | None) -> str:
        if row is None or missing(row["completed_share_p50"]):
            return UNAVAILABLE
        return percent(100 * float(row["completed_share_p50"]))

    def spread(row: pd.Series | None) -> str:
        if row is None or missing(row["completed_share_p10"]):
            return "Not in this run. Run simulation again to see it" if completion is None else ""
        return (
            f"P10 {percent(100 * float(row['completed_share_p10']))} · "
            f"P90 {percent(100 * float(row['completed_share_p90']))}"
        )

    left, right = kpi_columns(st, 2)
    kpi(
        left,
        "Charged by departure",
        share(all_row),
        context=spread(all_row),
        help="Smart path: share of the group's sessions that reached their target by unplug, "
        "P50 across weeks.",
    )
    kpi(
        right,
        "Charged, early departures",
        share(early_row),
        context=spread(early_row),
        help="The same for sessions that unplugged before the planner's expected departure.",
    )
    firmness = getattr(result, "firmness_by_manufacturer", None)
    if firmness is not None:
        rows = firmness.loc[
            firmness["direction"].eq("turn_down") & firmness["duration_hours"].eq(1.0)
        ]
        st.markdown(
            "**Physical firmness by maker, 1-hour turn-down**",
            help="Firmness is delivered ÷ deliverable, where deliverable is what each maker's "
            "EVs would have given had every session followed its plan. P50 across weeks.",
        )
        # Fable review B3: firmness is a share (delivered ÷ deliverable), so
        # it reads as a percent, matching Trading ▸ 3's own firmness table
        # (trading_pnl._firmness_table) rather than a bare 0-1 ratio.
        firmness_columns = {
            "firmness_p10": "Firmness, P10 (firm)",
            "firmness_p50": "Firmness, P50",
            "firmness_p90": "Firmness, P90",
        }
        table = rows[["manufacturer_label", "ev_count", *firmness_columns]].rename(
            columns={"manufacturer_label": "Maker", "ev_count": "EVs", **firmness_columns}
        )
        for column in firmness_columns.values():
            table[column] = [
                UNAVAILABLE if missing(value) else percent(100 * value) for value in table[column]
            ]
        st.dataframe(table, hide_index=True, width="stretch")


def _render_departure(st: Any, result: Any, group: str, name: str) -> None:
    bands = result.session_distribution_bands
    wanted_metrics = set(_DEPARTURE_METRICS.values()) & set(bands["metric"])
    rows = bands.loc[
        bands["group_id"].eq(group)
        & bands["day_type"].eq("all")
        & bands["metric"].isin(wanted_metrics)
    ]
    if rows.empty:
        st.caption(f"SoC at departure is shown for the fleet and archetypes, not for {name}.")
        return
    paths = _departure_paths(rows["metric"])
    metric_labels = {_DEPARTURE_METRICS[path]: PATH_LABELS[path] for path in paths}
    if "timed" in paths:
        # Named in full (decision 0007): absent, the title and caption below
        # stay the exact two-policy wording this chart always had.
        policy_words = " vs ".join(PATH_LABELS[path] for path in paths)
        title = f"SoC at departure, {policy_words}, {name.lower()}, % of sessions"
        # Timed tariff applies no tariff price; wherever it is named beside
        # money or a cost-adjacent reading, the caption says every policy is
        # costed alike so "tariff" is not read as a retail rate.
        caption = (
            "Early departures cut the smart-path SoC. All paths cost alike at the day-ahead "
            "price; no tariff rate. P50 bars, P10–P90 whiskers."
        )
    else:
        title = f"SoC at departure, both paths, {name.lower()}, % of sessions"
        caption = (
            "Early departures cut the smart-path SoC. Bars P50, whiskers P10–P90 across "
            "weeks; sessions that end in the study."
        )
    chart_block(
        st,
        _departure_figure(rows),
        title=title,
        caption=caption,
        frame=rows.assign(metric=rows["metric"].map(metric_labels))[
            ["metric", "bin_label", "share_p10", "share_p50", "share_p90"]
        ].rename(
            columns={
                "metric": "Path",
                "bin_label": "Bin",
                "share_p10": "Share P10",
                "share_p50": "Share P50",
                "share_p90": "Share P90",
            }
        ),
        definition="State of charge (SoC): battery energy at each session's last connected "
        "half-hour ÷ capacity, on the unmanaged and smart paths of the same sessions (plug-in "
        "and unplug are the same on both).",
        height=CHART_HEIGHTS["small_multiple_panel"] + 60,
        key="partners-departure-soc",
    )
