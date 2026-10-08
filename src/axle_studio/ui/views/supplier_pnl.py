"""Supplier ▸ 3 Supplier P&L: what smart charging does to a supplier's week, per customer.

Reads (supplier contract v1 §11; decision 0006): ``supplier_pnl_summary``
and ``supplier_pnl_world`` for the chosen hedge variant (``profiled`` by
default, ``flat`` beside), ``hedge_block_summary``, ``carbon_shift_summary``,
``product_sheet`` (the de-rated firm figure, §3.9, when the firm-MW frames
exist) and ``trading_kpis`` (the trading desk's net, shown beside and
never added, §3.1). Below them sit the blocks this lens absorbed from the
earlier "Shape and value" lens (trading contract v1 §9.8):
``shape_premium_summary``, ``household_value_distribution`` with
``household_value_summary``, and ``revenue_by_segment``.

Every figure is a stored column. The waterfall uses the components' means,
the only statistic that adds up (§3.4); the caption says quantiles do not.
Two readings of one tariff are kept apart and never added (§3.1, overnight
review ruling): the supplier P&L assumes a flat retail tariff, so the
supplier keeps the procurement saving; the customer value assumes the
day-ahead price reached the customer (pass-through). An unset commercial
term (platform fee, flat reward) reads "Unavailable (unset)", never £0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..components import header
from ..components.chart_table import chart_block
from ..components.kpi import badge_line, kpi, kpi_columns
from ..style import (
    ARCHETYPE_COLOURS,
    CHART_HEIGHTS,
    INK,
    MUTED_INK,
    SERIES_COLOURS,
    UNAVAILABLE,
    assumption_value,
    format_quantity,
    money,
    percent,
)
from .supplier_common import (
    BUCKET_LABELS,
    EXTRAPOLATED,
    UNSET,
    band_text,
    cohort_labels,
    evidence_caption,
    group_label,
    horizon_label,
    missing,
    p50_text,
    season_text,
    stat_row,
)

HEDGE_VARIANTS = {"profiled": "Profiled hedge", "flat": "Flat hedge"}
"""§3.2: the like-for-like profiled hedge is the headline; the flat block is shown beside."""

COMPONENTS = (
    ("energy_saving_gbp", "Energy saving at day-ahead"),
    ("hedge_error_saving_gbp", "Hedge-error saving"),
    ("grid_event_payment_gbp", "Grid-event payments"),
    ("customer_payment_gbp", "Customer payments"),
)
"""Decision 0006's four components, in waterfall order (supplier contract v1 §3.2)."""

_BLOCK_LABELS = {"baseload": "Baseload", "peak": "Peak", "off_peak": "Off-peak"}

DOUBLE_COUNT_HELP = (
    "The supplier hedges the trading desk's full forecast. So its energy and hedge-error "
    "components already hold the day-ahead value of the sold turn-down and its imbalance. "
    "Adding the trading ledger's net would count the same shifted energy twice."
)
TARIFF_HELP = (
    "Flat retail tariff: the customer pays the same p/kWh whenever the EV charges. So the "
    "supplier keeps the saving on the energy it buys, and pays customers only through the "
    "customer payment. The customer value on this page uses the other reading, pass-through: "
    "as if the day-ahead price reached the customer. The two readings are never added."
)


# --- Lookups ------------------------------------------------------------------


def _pnl(result: Any, variant: str, metric: str) -> pd.Series | None:
    return stat_row(result.supplier_pnl_summary, hedge_variant=variant, metric=metric)


def _money2(value: float) -> str:
    return money(value, decimals=2, signed=True)


def _money0(value: float) -> str:
    return money(value, signed=True)


def payment_mode_text(result: Any) -> str:
    """The customer-payment rule in words, from the run's own records (§3.2)."""

    mode = assumption_value(result, "supplier.customer_reward_mode")
    if mode is not None and int(mode) == 1:
        reward = assumption_value(result, "supplier.customer_reward_gbp_per_ev_per_month")
        if reward is None or missing(reward):
            return f"flat reward per customer: {UNSET.lower()}"
        return f"flat {money(float(reward), decimals=2)} per customer per month"
    share = assumption_value(result, "trading.customer_revenue_share")
    share_text = "a share" if share is None else percent(100 * float(share))
    return f"{share_text} of the week's positive gain"


# --- Waterfall and strip ------------------------------------------------------


def waterfall_rows(result: Any, variant: str) -> pd.DataFrame:
    """The bars the waterfall draws: each component's stored mean, then the net.

    A component whose mean is NaN (the customer payment with a flat reward
    unset) is left out, and the net it would feed is then not drawn: an
    unavailable term is never drawn as £0 (decision 0003).
    """

    rows = []
    for metric, label in COMPONENTS:
        row = _pnl(result, variant, metric)
        if row is not None and not missing(row["mean"]):
            rows.append((label, "relative", float(row["mean"])))
    before_fee = _pnl(result, variant, "net_gain_before_fee_gbp")
    fee = _pnl(result, variant, "platform_fee_gbp")
    net = _pnl(result, variant, "net_gain_gbp")
    if before_fee is not None and not missing(before_fee["mean"]):
        rows.append(("Supplier gain before fee", "total", float(before_fee["mean"])))
        if fee is not None and not missing(fee["mean"]):
            rows.append(("Platform fee", "relative", float(fee["mean"])))
            rows.append(("Supplier gain after fee", "total", float(net["mean"])))
    return pd.DataFrame(rows, columns=["Bar", "Measure", "Mean (£ per week)"])


def _waterfall_figure(rows: pd.DataFrame) -> go.Figure:
    # Horizontal, top to bottom: five category labels collided under a
    # vertical waterfall at 390 px; beside the bars they always fit.
    figure = go.Figure(
        go.Waterfall(
            orientation="h",
            y=rows["Bar"],
            x=rows["Mean (£ per week)"],
            measure=rows["Measure"],
            name="Supplier P&L components",
            increasing={"marker": {"color": SERIES_COLOURS["normal"]}},
            decreasing={"marker": {"color": SERIES_COLOURS["flag"]}},
            totals={"marker": {"color": SERIES_COLOURS["difference"]}},
            connector={"line": {"color": MUTED_INK, "width": 1}},
            hovertemplate="%{y}: £%{x:,.2f} per week (mean)<extra></extra>",
        )
    )
    figure.update_yaxes(
        title={"text": "component, £ per week", "standoff": 12}, autorange="reversed"
    )
    figure.update_xaxes(title="£ per week, mean (illustrative)")
    figure.update_layout(showlegend=False, hovermode="closest")
    return figure


def _strip_figure(values: np.ndarray, world_ids: np.ndarray) -> go.Figure:
    # Deterministic jitter to spread overlapping markers; a display position only.
    jitter = (world_ids % 7) / 7.0 - 0.5
    figure = go.Figure(
        go.Scatter(
            x=values,
            y=jitter,
            mode="markers",
            name="One simulated week",
            marker={"color": SERIES_COLOURS["difference"], "size": 7, "opacity": 0.8},
            customdata=world_ids,
            hovertemplate="Week %{customdata}: £%{x:,.2f} per customer per month<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color=MUTED_INK, line_width=1)
    figure.update_xaxes(title="£ per customer per month (illustrative)")
    figure.update_yaxes(title="simulated weeks", showticklabels=False, range=[-1, 1])
    figure.update_layout(hovermode="closest")
    return figure


# --- Hedge blocks -----------------------------------------------------------


def _block_rows(result: Any) -> pd.DataFrame:
    summary = result.hedge_block_summary
    return summary.loc[summary["metric"].eq("mean_mw")]


def _block_figure(rows: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for path_id, name, colour, offset in (
        ("normal", "Unmanaged", SERIES_COLOURS["normal"], -0.08),
        ("selected", "Smart", SERIES_COLOURS["selected"], 0.08),
    ):
        path = rows.loc[rows["path_id"].eq(path_id)].set_index("block").reindex(list(_BLOCK_LABELS))
        x = np.arange(len(_BLOCK_LABELS)) + offset
        p10, p50, p90 = (path[stat].to_numpy(dtype=float) for stat in ("p10", "p50", "p90"))
        # Dots with P10–P90 intervals, not bars with whiskers: a filled bar
        # pulls the eye inside it (uncertainty-viz rule 9).
        figure.add_trace(
            go.Scatter(
                x=x,
                y=p50,
                mode="markers",
                name=name,
                marker={"color": colour, "size": 10},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": np.maximum(p90 - p50, 0.0),
                    "arrayminus": np.maximum(p50 - p10, 0.0),
                    "color": colour,
                    "thickness": 1.5,
                    "width": 4,
                },
                customdata=np.stack([p10, p90], axis=-1),
                hovertemplate="%{y:,.3f} MW (P50; P10–P90 %{customdata[0]:,.3f}–"
                "%{customdata[1]:,.3f})<extra>" + name + "</extra>",
            )
        )
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(_BLOCK_LABELS))),
        ticktext=list(_BLOCK_LABELS.values()),
        range=[-0.5, len(_BLOCK_LABELS) - 0.5],
    )
    figure.update_yaxes(title="MW, mean over the block", rangemode="tozero")
    figure.update_layout(hovermode="closest")
    return figure


def _block_table(result: Any) -> pd.DataFrame:
    summary = result.hedge_block_summary
    labels = {"volume_mwh": "Volume", "mean_mw": "Mean power", "volume_mwh_per_month": "Volume"}
    paths = {"normal": "Unmanaged", "selected": "Smart", "difference": "Smart minus unmanaged"}
    return pd.DataFrame(
        {
            "Block": summary["block"].map(_BLOCK_LABELS),
            "Path": summary["path_id"].map(paths),
            "Measure": summary["metric"].map(labels),
            "Unit": summary["unit"].where(
                summary["metric"].ne("volume_mwh_per_month"), "MWh per month, scaled from weeks"
            ),
            "Hours": summary["block_hours"],
            "P10": summary["p10"],
            "P50": summary["p50"],
            "P90": summary["p90"],
        }
    )


def _firm_tile(result: Any) -> tuple[str, str]:
    """The de-rated evening firm turn-down from ``product_sheet`` (§3.9), or unavailable.

    The firm figure is the table's P10 column (90 % exceedance) and P95
    exceedance is its P05 column (trading contract v1 §10's two conventions).
    """

    sheet = getattr(result, "product_sheet", None)
    row = stat_row(sheet, window="evening", direction="turn_down", duration_hours=1.0)
    if row is None or missing(row["window_mean_mw_p10"]):
        return UNAVAILABLE, "Not in this run. Run simulation again to see it"
    season = str(result.supplier_pnl_summary["season"].iloc[0])
    # B-2: name the two exceedance conventions, not a trader's "P90"/"P95",
    # which called the across-weeks P10 "P90" here while Firm MW's own
    # table called the same column "P10" (one figure, one name; ruling
    # "firm = across-weeks P10"). "Season" is the demand-shape months (the
    # summer_months market record, ``season_text``'s own convention) the run
    # started in, not the BST clock: a run starting in winter must not say
    # "Apr-Sep" (final critique B-2 fixed this the same way for the season
    # badge below; this tile had the same bug, fixed identically here).
    months = "Apr-Sep" if season == "summer" else "Oct-Mar"
    return (
        format_quantity(float(row["window_mean_mw_p10"]), "MW", decimals=2),
        f"{season} evening ({months}), 1 h hold, P10 (90% exceedance); stricter P05 (95%) "
        f"{format_quantity(row['window_mean_mw_p05'], 'MW', decimals=2)}",
    )


# --- Shape and value blocks (trading contract v1 §9.8) -------------------------


def _shape_table(result: Any) -> pd.DataFrame:
    summary = result.shape_premium_summary
    paths = {"normal": "Unmanaged", "selected": "Smart", "difference": "Smart minus unmanaged"}
    rows = []
    for path_id, path_label in paths.items():
        premium = stat_row(summary, path_id=path_id, metric="shape_premium_gbp_per_mwh")
        cost = stat_row(summary, path_id=path_id, metric="shape_cost_gbp_per_week")
        rows.append(
            {
                "Path": path_label,
                "Shape premium P50 (£/MWh)": p50_text(
                    premium, lambda v: format_quantity(v, "", decimals=1)
                ),
                "Shape cost P50 (£ per week)": p50_text(cost, _money0),
                "Shape cost P10–P90": band_text(cost, _money0),
            }
        )
    return pd.DataFrame(rows)


def _household_figure(rows: pd.DataFrame) -> go.Figure:
    inner = rows.loc[np.isfinite(rows["bin_lower_gbp"]) & np.isfinite(rows["bin_upper_gbp"])]
    lower = inner["bin_lower_gbp"].to_numpy(dtype=float)
    upper = inner["bin_upper_gbp"].to_numpy(dtype=float)
    p10, p50, p90 = (inner[stat].to_numpy(dtype=float) * 100.0 for stat in ("p10", "p50", "p90"))
    figure = go.Figure(
        go.Bar(
            x=(lower + upper) / 2.0,
            y=p50,
            width=(upper - lower) * 0.85,
            name="Share of customers",
            marker_color=SERIES_COLOURS["normal"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": np.maximum(p90 - p50, 0.0),
                "arrayminus": np.maximum(p50 - p10, 0.0),
                "color": MUTED_INK,
                "thickness": 1,
                "width": 0,
            },
            customdata=np.stack([lower, upper, p10, p90], axis=-1),
            hovertemplate="£%{customdata[0]:,.0f} to £%{customdata[1]:,.0f}: %{y:.1f}% (P50; "
            "P10–P90 %{customdata[2]:.1f}–%{customdata[3]:.1f}%)<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color=MUTED_INK, line_width=1)
    figure.update_xaxes(title="£ per household per month (illustrative)")
    figure.update_yaxes(title="% of customers", rangemode="tozero")
    figure.update_layout(bargap=0, hovermode="closest")
    return figure


def _segment_figure(rows: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    # Horizontal, bucket names on the category axis: ten rotated labels
    # collided into an unreadable block at 390 px (B-15), the same fix the
    # waterfall above already uses for its own long bucket names.
    figure = go.Figure()
    for index, segment_id in enumerate(dict.fromkeys(rows["segment_id"])):
        segment = rows.loc[rows["segment_id"].eq(segment_id)].set_index("bucket")
        segment = segment.reindex([b for b in BUCKET_LABELS if b in segment.index])
        figure.add_trace(
            go.Bar(
                y=[BUCKET_LABELS[b] for b in segment.index],
                x=segment["mean"],
                orientation="h",
                name=labels.get(segment_id, segment_id),
                marker_color=ARCHETYPE_COLOURS[index % len(ARCHETYPE_COLOURS)],
                hovertemplate="%{y}: £%{x:,.2f} per week (mean)<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(barmode="group", hovermode="closest")
    figure.update_yaxes(title={"text": "bucket, £ per week", "standoff": 12}, autorange="reversed")
    figure.update_xaxes(title="£ per week, mean (illustrative)", zeroline=True, zerolinecolor=INK)
    return figure


def _render_shape_and_value(st: Any, result: Any, variant: str) -> None:
    labels = cohort_labels(result)
    evidence = evidence_caption(result)

    st.markdown(
        "**Shape premium: what the load's timing costs against a flat block**",
        help="Per simulated week: the load-weighted day-ahead price minus the week's mean "
        "price, times the week's volume. Below zero means the load sits in cheaper half-hours "
        "than a flat block would. Smart minus unmanaged is worked out inside each week, then "
        "spread across weeks.",
    )
    st.dataframe(_shape_table(result), hide_index=True, width="stretch")
    st.caption(f"{horizon_label('week_ahead').capitalize()}. {evidence}")

    distribution = result.household_value_distribution
    groups = list(dict.fromkeys(distribution["group_id"]))
    group = st.selectbox(
        "Customer group",
        options=groups,
        format_func=lambda g: group_label(g, labels),
        key="supplier-household-group",
    )
    group = group if group in groups else groups[0]
    summary = result.household_value_summary
    median = stat_row(summary, group_id=group, statistic="p50_across_evs")
    worse_off = stat_row(summary, group_id=group, statistic="share_worse_off")
    mean_value = stat_row(summary, group_id=group, statistic="mean_value")
    supplier_gain = _pnl(result, variant, "net_gain_before_fee_per_customer_per_month_gbp")
    left, middle, right = kpi_columns(st, 3)
    kpi(
        left,
        "Customer value, median EV",
        p50_text(median, _money2),
        context="As if day-ahead prices reached the customer",
        help="The pass-through reading: the median EV's saving plus its share of the trading "
        "desk's customer payment, as if the day-ahead price reached the customer. P50 across "
        f"simulated weeks; {horizon_label('scenario')}.",
    )
    kpi(
        middle,
        "Customers worse off",
        p50_text(worse_off, lambda v: percent(100 * v)),
        context=band_text(worse_off, lambda v: percent(100 * v)),
        help="Share of the group's EVs whose monthly value is below £0, P50 across weeks.",
    )
    kpi(
        right,
        "Supplier gain per customer",
        p50_text(supplier_gain, _money2),
        context="Flat-tariff reading; never added",
        help="Shown beside the customer value, never added to it: the two readings value the "
        "same shifted energy from the two sides of one tariff. " + TARIFF_HELP,
    )
    rows = distribution.loc[distribution["group_id"].eq(group)].sort_values("bin_index")
    chart_block(
        st,
        _household_figure(rows),
        title=f"Value per customer per month, {group_label(group, labels).lower()}",
        caption="Customer value only; never add it to the supplier gain or the trading net. "
        "Bars P50, whiskers P10–P90 across weeks.",
        frame=rows.rename(
            columns={
                "bin_lower_gbp": "From (£/month)",
                "bin_upper_gbp": "To (£/month)",
                "world_count": "Simulated weeks",
                "p10": "Share P10",
                "p50": "Share P50",
                "p90": "Share P90",
            }
        )[
            [
                "From (£/month)",
                "To (£/month)",
                "Simulated weeks",
                "Share P10",
                "Share P50",
                "Share P90",
            ]
        ],
        definition=(
            "Per simulated week, each treated EV's saving plus its customer revenue share "
            f"({horizon_label('scenario')}). A treated EV is one enrolled in smart charging; "
            "the hold-out control EVs are left out. Then the share of EVs in each £1 bin, with "
            "the spread taken across weeks, so the bars do not sum to 100%. The two open outer "
            "bins (below −£20, above £40) are in this table only. Mean value "
            f"{p50_text(mean_value, _money2)} (P50). Shared out by each EV's share of settled "
            "flexibility, not settled per customer."
        ),
        height=CHART_HEIGHTS["histogram"],
        key="supplier-household-value",
    )

    segments = result.revenue_by_segment
    types = list(dict.fromkeys(segments["segment_type"]))
    type_labels = {"all": "Fleet", "cohort": "Archetype", "zone": "Zone", "manufacturer": "Maker"}
    segment_type = st.segmented_control(
        "Split trading money by",
        options=types,
        format_func=lambda t: type_labels.get(t, t),
        default=types[0],
        required=True,
        key="supplier-segment-type",
    )
    segment_type = segment_type if segment_type in types else types[0]
    chosen = segments.loc[
        segments["segment_type"].eq(segment_type) & segments["metric"].eq("gbp_per_week")
    ]
    # Zones read "Zone 1"; archetypes and makers take their own names.
    segment_labels = {
        segment: "Fleet"
        if segment == "all"
        else segment.replace("_", " ").capitalize()
        if segment.startswith("zone_")
        else group_label(segment, labels)
        for segment in chosen["segment_id"]
    }
    chart_block(
        st,
        _segment_figure(chosen, segment_labels),
        title="Trading money by bucket and segment, mean £ per week",
        caption="Allocated by each EV's share of settled flexibility. Means add across segments; "
        f"quantiles do not. {evidence}",
        frame=chosen.assign(
            segment_id=chosen["segment_id"].map(segment_labels),
            bucket=chosen["bucket"].map(BUCKET_LABELS),
        )[["bucket", "segment_id", "ev_count", "mean", "p10", "p50", "p90"]].rename(
            columns={
                "bucket": "Bucket",
                "segment_id": "Segment",
                "ev_count": "EVs",
                "mean": "Mean (£/week)",
                "p10": "P10 (£/week)",
                "p50": "P50 (£/week)",
                "p90": "P90 (£/week)",
            }
        ),
        definition=(
            "The trading ledger's buckets (full strategy: sold day-ahead, re-traded intraday), "
            "shared out to treated EVs night by night by each EV's share of settled "
            "flexibility, then summed by segment. A penalty or imbalance is spread by that "
            "share, not traced to the EV that caused it. This is the trading desk's money, "
            "shown for reference; it is not supplier money."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="supplier-revenue-segments",
    )


# --- The lens ---------------------------------------------------------------


def render_supplier_pnl(st: Any, result: Any) -> None:
    """Render Supplier ▸ 3: headline tiles and the waterfall hero, then three detail tabs
    (per-week spread and components, hedge blocks and CO₂, shape and customer value) that
    hold the rest (Q-13: 3,229 px and 15 tiles read as buried, not answered)."""

    variant = header.controls(st).segmented_control(
        "Hedge",
        options=list(HEDGE_VARIANTS),
        format_func=HEDGE_VARIANTS.__getitem__,
        default="profiled",
        required=True,
        key="supplier-hedge-variant",
        label_visibility="collapsed",
    )
    variant = variant if variant in HEDGE_VARIANTS else "profiled"
    weeks = result.world_count
    evidence = evidence_caption(result)
    season = season_text(result.supplier_pnl_summary)

    before_fee = _pnl(result, variant, "net_gain_before_fee_per_customer_per_month_gbp")
    after_fee = _pnl(result, variant, "net_gain_per_customer_per_month_gbp")
    fee_row = _pnl(result, variant, "platform_fee_per_customer_per_month_gbp")
    kpis = result.trading_kpis
    trading_net = stat_row(kpis, strategy="full", metric="net_gbp_per_week")
    peak = stat_row(
        result.hedge_block_summary, block="peak", path_id="difference", metric="mean_mw"
    )

    first, second, third, fourth = kpi_columns(st, 4)
    kpi(
        first,
        "Supplier gain per customer",
        p50_text(before_fee, _money2),
        unit="/month",
        context=band_text(before_fee, _money2) or "Before platform fee",
        help="Illustrative simulated supplier P&L before the platform fee, per treated customer "
        "(an EV enrolled in smart charging) per month. P50 across "
        f"{weeks} simulated weeks; {horizon_label('scenario')}. " + TARIFF_HELP,
    )
    fee_unset = fee_row is None or missing(fee_row["p50"])
    kpi(
        second,
        "After platform fee",
        p50_text(after_fee, _money2, unset=fee_unset),
        unit=None if fee_unset else "/month",
        context="Platform fee not set" if fee_unset else band_text(after_fee, _money2),
        help="Unavailable until a platform fee is set in Edit assumptions ▸ Supplier. An unset "
        "fee is never treated as £0: that would show a gain nobody has agreed.",
    )
    kpi(
        third,
        "Trading net, not added",
        p50_text(trading_net, _money0),
        unit="/week",
        context="Trading desk's ledger, beside",
        help="The trading ledger's illustrative net (full strategy: sold day-ahead, re-traded "
        "intraday), P50 across weeks. Shown beside, never added. " + DOUBLE_COUNT_HELP,
    )
    kpi(
        fourth,
        "Peak-block change",
        p50_text(peak, lambda v: format_quantity(v, "MW", decimals=3)),
        context=band_text(peak, lambda v: format_quantity(v, "MW", decimals=3)),
        help="Smart minus unmanaged mean power over the peak block (07:00–19:00 Monday to "
        "Friday), P50 across weeks. Negative means energy moved out of the peak. "
        f"{horizon_label('week_ahead').capitalize()}.",
    )
    badge_line(
        st,
        "Customer payments:",
        payment_mode_text(result),
        help="Mode and rate come from Edit assumptions ▸ Supplier. " + TARIFF_HELP,
    )

    rows = waterfall_rows(result, variant)
    chart_block(
        st,
        _waterfall_figure(rows),
        title=f"Supplier P&L components, {HEDGE_VARIANTS[variant].lower()}, mean £ per week",
        caption=f"Means across {weeks} weeks add to the net; quantiles do not. Flat retail "
        "tariff: the supplier keeps the saving on the energy it buys.",
        frame=rows,
        definition=(
            "Four parts. Energy saving: unmanaged minus smart home import, valued at the "
            "day-ahead price. Hedge-error saving: what the fleet delivered against a "
            "like-for-like hedge, priced at imbalance minus day-ahead. Grid-event payments "
            "from third parties. The customer payment "
            f"({payment_mode_text(result)}). {DOUBLE_COUNT_HELP} {TARIFF_HELP}"
        ),
        height=CHART_HEIGHTS["time_series"],
        key="supplier-pnl-waterfall",
    )
    grid_events = rows.loc[rows["Bar"].eq("Grid-event payments"), "Mean (£ per week)"]
    if len(grid_events) and grid_events.iloc[0] == 0.0:
        st.caption(
            "Grid-event payments £0: none paid out this run (events are scripted under Edit "
            "assumptions ▸ Events)."
        )
    st.caption(
        "The trading net is beside, never added: it values the same shifted energy.",
        help=DOUBLE_COUNT_HELP,
    )

    detail, blocks, shape_and_value = st.tabs(
        [
            "Per-week spread and components",
            "Hedge blocks, firm capacity and CO₂",
            "Shape and customer value",
        ]
    )
    # Each helper takes the plain ``st`` module (entered through ``with
    # tab:``), never the tab object itself: chart_block's own nested
    # ``with st.expander(...)`` only redirects later same-object calls when
    # ``st`` is the module Streamlit's context stack tracks, not a tab
    # handle held as a local variable (verified against AppTest).
    with detail:
        _render_per_week_detail(st, result, variant, weeks, season, evidence)
    with blocks:
        _render_hedge_blocks_and_co2(st, result, weeks)
    with shape_and_value:
        _render_shape_and_value(st, result, variant)


def _render_per_week_detail(
    st: Any, result: Any, variant: str, weeks: int, season: str, evidence: str
) -> None:
    """The strip of one dot per simulated week, then the waterfall's own components split
    into shape/volume and the two hedge-error readings (moved off the main surface, Q-13)."""

    world = result.supplier_pnl_world
    chosen = world.loc[world["hedge_variant"].eq(variant)].sort_values("world_id")
    column = "net_gain_before_fee_per_customer_per_month_gbp"
    chart_block(
        st,
        _strip_figure(chosen[column].to_numpy(dtype=float), chosen["world_id"].to_numpy()),
        title="Supplier gain per customer per month, one dot per simulated week",
        caption=f"Before platform fee; {EXTRAPOLATED}. {season.capitalize()}. {evidence}",
        frame=chosen[["world_id", column, "treated_ev_count"]].rename(
            columns={
                "world_id": "Simulated week",
                column: "£ per customer per month",
                "treated_ev_count": "Treated customers",
            }
        ),
        definition=(
            "Each simulated week's before-fee supplier gain × 52 ÷ 12, divided by the treated "
            "customers (EVs enrolled in smart charging; the hold-out control EVs are not "
            f"customers of the product). {horizon_label('scenario').capitalize()}."
        ),
        height=CHART_HEIGHTS["strip"],
        key="supplier-pnl-strip",
    )

    energy = _pnl(result, variant, "energy_saving_gbp")
    shape = _pnl(result, variant, "shape_saving_gbp")
    volume = _pnl(result, variant, "volume_value_gbp")
    for column_, label, row, help_text in zip(
        kpi_columns(st, 3),
        ("Energy saving", "Of which shape", "Of which volume"),
        (energy, shape, volume),
        (
            "Unmanaged minus smart home import valued at the day-ahead price, per week.",
            "The load-weighted premium over the week's mean price: it is shape, not less energy.",
            "The week's mean price times the change in home volume (early departures move "
            "energy to public top-ups).",
        ),
        strict=True,
    ):
        kpi(
            column_,
            label,
            p50_text(row, _money0),
            unit="/week",
            context=band_text(row, _money0),
            help=f"{help_text} P50 across weeks; {horizon_label('week_ahead')}.",
        )
    profiled = _pnl(result, "profiled", "hedge_error_saving_gbp")
    flat = _pnl(result, "flat", "hedge_error_saving_gbp")
    worth = _pnl(result, "profiled", "worth_of_profiled_hedge_gbp")
    for column_, label, row, help_text in zip(
        kpi_columns(st, 3),
        ("Hedge error, profiled", "Hedge error, flat", "Worth of profiled hedge"),
        (profiled, flat, worth),
        (
            "A profiled hedge buys the smart-charged fleet's own forecast shape. This is what "
            "the fleet delivered against that hedge, priced at imbalance minus day-ahead.",
            "The same against a flat block per night: the whole realised shift goes to imbalance.",
            "Profiled minus flat: what hedging the shape is worth under half-hourly settlement. "
            "Beside the P&L, not a component of it.",
        ),
        strict=True,
    ):
        kpi(
            column_,
            label,
            p50_text(row, _money0),
            unit="/week",
            context=band_text(row, _money0),
            help=f"{help_text} P50 across weeks.",
        )
    # Intraday dispatch contract v1 §4.4, lead note at integration: metered
    # import (what the fleet delivered above) includes the dispatch's own
    # moves, so this is where a supplier sees their value, at imbalance
    # minus day-ahead; 0 change with the switch off (the metered path is
    # then the day-ahead plan path).
    if getattr(result, "dispatch_world_slot", None) is not None:
        st.caption(
            "Includes the intraday dispatch's moves, priced at imbalance minus day-ahead.",
            help="Metered import is what the dispatched fleet drew. The smart hedge is the "
            "day-ahead forecast and does not change, so the dispatch's intraday value shows up "
            "here.",
        )


def _render_hedge_blocks_and_co2(st: Any, result: Any, weeks: int) -> None:
    """The hedge-block bands, the firm evening tile and the two CO₂ tiles (moved off the
    main surface, Q-13)."""

    block_rows = _block_rows(result)
    hours = (
        block_rows.drop_duplicates("block").set_index("block")["block_hours"]
        if len(block_rows)
        else pd.Series(dtype=float)
    )
    hours_text = ", ".join(
        f"{_BLOCK_LABELS[b].lower()} {hours[b]:g} h" for b in _BLOCK_LABELS if b in hours.index
    )
    chart_block(
        st,
        _block_figure(block_rows),
        title="Hedge blocks: mean power, unmanaged vs smart, MW",
        caption=f"P50 with P10–P90 across {weeks} weeks. Peak 07:00–19:00 Mon–Fri; this week "
        f"{hours_text}.",
        frame=_block_table(result),
        definition=(
            "The standard blocks a supplier buys, with hours counted from the study's own "
            "half-hours (a clock-change week has 59 or 61 peak hours). Mean power is volume ÷ "
            "block hours. Monthly volumes are scaled up from weeks; they are not forecasts."
        ),
        height=CHART_HEIGHTS["strip"],
        key="supplier-hedge-blocks",
    )

    firm_value, firm_context = _firm_tile(result)
    carbon = result.carbon_shift_summary
    co2_week = stat_row(carbon, metric="co2_shifted_kg")
    co2_ev = stat_row(carbon, metric="co2_shifted_kg_per_ev_per_month")
    firm_column, week_column, ev_column = kpi_columns(st, 3)
    kpi(
        firm_column,
        "Firm evening turn-down",
        firm_value,
        context=firm_context,
        help="Firm capacity: the evening 1-hour turn-down met in 9 weeks out of 10 (the "
        "product sheet's P10 column), decided day-ahead. The season is the run's own, never a "
        "forecast.",
    )
    kpi(
        week_column,
        "CO₂ shifted",
        p50_text(co2_week, lambda v: format_quantity(v, "kg", decimals=0)),
        unit="/week",
        context=band_text(co2_week, lambda v: format_quantity(v, "kg", decimals=0)),
        help="Unmanaged minus smart import times an illustrative synthetic carbon intensity "
        "tied to the model's own net demand; not observed grid data, and a mix intensity, "
        "not a marginal one.",
    )
    kpi(
        ev_column,
        "CO₂ shifted per EV",
        p50_text(co2_ev, lambda v: format_quantity(v, "kg", decimals=1)),
        unit="/month",
        context=band_text(co2_ev, lambda v: format_quantity(v, "kg", decimals=1)),
        help=f"Per treated EV per month, {EXTRAPOLATED}. Illustrative synthetic carbon intensity.",
    )
    st.caption(
        "CO₂: illustrative synthetic intensity tied to the model's own net demand; not "
        "observed grid data."
    )
