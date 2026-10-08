"""Supplier ▸ 1 Availability and cost curve: what the fleet could move, and at what price.

Reads (trading contract v1 §9.8): ``deferrable_power_bands`` ``total`` on
the unmanaged path (the turn-down a supplier could ask for), the smart
path's ``turn_up_headroom_kw`` from ``flexibility_bands`` (decision 0004
item 50: turn-up is shown on the smart path only) and ``flex_cost_curve``
for one London half-hour and day type. The half-hour selector only filters
rows the model built for all 48 half-hours, so choosing one never
recomputes anything. Every chart, tile and table on this page stays in the
stored kW (Q-12 of the final critique): the fleet peaks well under 10 MW,
so kW is the one unit here.

The cost curve describes each simulated week's own sessions costed at the
day-ahead prices visible at the day-ahead decision; it is not a forecast a
supplier held at 13:00 (lead review B2), and its caption says so.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from ..components import header
from ..components.chart_table import chart_block
from ..components.kpi import kpi, kpi_columns
from ..registry import DAY_TYPE_KEY, DAY_TYPE_LABELS
from ..style import (
    CHART_HEIGHTS,
    PATH_STYLES,
    SERIES_COLOURS,
    band_and_line,
    format_quantity,
    hover_time_labels,
    kw,
    london_time_axis,
)
from .supplier_common import evidence_caption, horizon_label, missing

DEFAULT_HALF_HOUR = "18:00"
"""The cost curve's default half-hour (trading contract v1 §9.9, a reporting choice)."""


def _availability_rows(result: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn-down (unmanaged ``total``) and turn-up (smart headroom) rows, slot order."""

    deferrable = result.deferrable_power_bands
    turn_down = deferrable.loc[
        deferrable["slack_bucket"].eq("total") & deferrable["path_id"].eq("normal")
    ].sort_values("slot_index")
    flexibility = result.flexibility_bands
    turn_up = flexibility.loc[
        flexibility["metric"].eq("turn_up_headroom_kw")
        & flexibility["path_id"].eq("selected")
        & flexibility["spread"].eq("across_weeks")
    ].sort_values("slot_index")
    return turn_down, turn_up


def _availability_figure(turn_down: pd.DataFrame, turn_up: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for rows, name, style in (
        (turn_down, "Turn-down, unmanaged path", PATH_STYLES["normal"]),
        (turn_up, "Turn-up headroom, smart path", PATH_STYLES["selected"]),
    ):
        if rows.empty:
            continue
        # kW, matching the tiles, cost curve and this chart's own table below:
        # the fleet peaks well under 10 MW, so kW is the one unit this page
        # uses (Q-12; MW stays reserved for the Firm MW product sheet).
        band_and_line(
            figure,
            rows["interval_start_utc"],
            rows["p10"].to_numpy(dtype=float),
            rows["p50"].to_numpy(dtype=float),
            rows["p90"].to_numpy(dtype=float),
            name=name,
            colour=style["color"],
            width=style["width"],
            customdata=hover_time_labels(rows["interval_start_utc"]),
            hovertemplate="%{customdata}: %{y:,.0f} kW (P50)<extra>" + name + "</extra>",
        )
    starts = pd.concat([turn_down["interval_start_utc"], turn_up["interval_start_utc"]])
    if not starts.empty:
        figure.update_xaxes(tickmode="array", **london_time_axis(starts))
    figure.update_yaxes(title="kW", rangemode="tozero")
    return figure


def _curve_rows(result: Any, *, half_hour: str, day_type: str) -> pd.DataFrame:
    curve = result.flex_cost_curve
    return curve.loc[curve["local_half_hour"].eq(half_hour) & curve["day_type"].eq(day_type)]


def _curve_figure(rows: pd.DataFrame) -> go.Figure:
    available = rows.loc[rows["metric"].eq("available_kw")].sort_values("threshold_gbp_per_mwh")
    figure = go.Figure()
    band_and_line(
        figure,
        available["threshold_gbp_per_mwh"],
        available["p10"],
        available["p50"],
        available["p90"],
        name="Available turn-down",
        colour=SERIES_COLOURS["difference"],
        hovertemplate="At or below £%{x:,.0f}/MWh: %{y:,.0f} kW (P50)<extra></extra>",
    )
    ceiling = rows.loc[rows["metric"].eq("charging_kw")]
    if len(ceiling) and not missing(ceiling["p50"].iloc[0]):
        # The curve's ceiling: every unmanaged kW charging at this half-hour.
        figure.add_trace(
            go.Scatter(
                x=[
                    available["threshold_gbp_per_mwh"].min(),
                    available["threshold_gbp_per_mwh"].max(),
                ],
                y=[float(ceiling["p50"].iloc[0])] * 2,
                mode="lines",
                name="All unmanaged charging (P50)",
                line={"color": SERIES_COLOURS["normal"], "dash": "dash", "width": 1.5},
                hovertemplate="All unmanaged charging: %{y:,.0f} kW (P50)<extra></extra>",
            )
        )
    figure.update_xaxes(title="Cost to move, £/MWh")
    figure.update_yaxes(title="kW available", rangemode="tozero")
    figure.update_layout(hovermode="closest")
    return figure


def _curve_table(rows: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "available_kw": "Available at or below the cost",
        "movable_kw": "Movable at any cost",
        "charging_kw": "All unmanaged charging",
    }
    table = rows.assign(metric=rows["metric"].map(labels))
    return table.rename(
        columns={
            "metric": "Measure",
            "threshold_gbp_per_mwh": "Cost to move (£/MWh)",
            "world_count": "Simulated weeks",
            "p10": "P10 (kW)",
            "p50": "P50 (kW)",
            "p90": "P90 (kW)",
        }
    )[["Measure", "Cost to move (£/MWh)", "Simulated weeks", "P10 (kW)", "P50 (kW)", "P90 (kW)"]]


def _kw_band(row: pd.Series | None) -> tuple[str, str]:
    if row is None or missing(row["p50"]):
        return kw(None), ""
    return kw(float(row["p50"])), f"P10 {kw(float(row['p10']))} · P90 {kw(float(row['p90']))}"


def render_availability(st: Any, result: Any) -> None:
    """Render Supplier ▸ 1: availability over the week, then the cost curve at one half-hour."""

    curve = result.flex_cost_curve
    half_hours = list(
        curve.drop_duplicates("local_half_hour").sort_values("profile_order")["local_half_hour"]
    )
    half_column, day_column = header.controls(st).columns([2, 3])
    with half_column:
        half_hour = st.selectbox(
            "Half-hour (London)",
            options=half_hours,
            index=half_hours.index(DEFAULT_HALF_HOUR) if DEFAULT_HALF_HOUR in half_hours else 0,
            key="supplier-cost-half-hour",
            label_visibility="collapsed",
        )
    with day_column:
        day_type = st.segmented_control(
            "Day",
            options=list(DAY_TYPE_LABELS),
            format_func=DAY_TYPE_LABELS.__getitem__,
            default="weekday",
            required=True,
            key=DAY_TYPE_KEY,
            label_visibility="collapsed",
            persist_state="session",
        )
    half_hour = half_hour if half_hour in half_hours else half_hours[0]
    day_type = day_type if day_type in DAY_TYPE_LABELS else "weekday"
    evidence = evidence_caption(result)
    weeks = result.world_count

    turn_down, turn_up = _availability_rows(result)
    chart_block(
        st,
        _availability_figure(turn_down, turn_up),
        title="What the fleet could turn down or up, kW",
        caption=f"P50 with P10–P90 across {weeks} simulated weeks; 1 week in 5 falls outside. "
        f"{evidence}",
        frame=pd.concat(
            [
                turn_down.assign(measure="Turn-down, unmanaged path")[
                    ["measure", "interval_start_utc", "interval_start_london", "p10", "p50", "p90"]
                ],
                turn_up.assign(measure="Turn-up headroom, smart path")[
                    ["measure", "interval_start_utc", "interval_start_london", "p10", "p50", "p90"]
                ],
            ]
        ).rename(
            columns={
                "measure": "Measure",
                "interval_start_utc": "Start (UTC)",
                "interval_start_london": "Start (London)",
                "p10": "P10 (kW)",
                "p50": "P50 (kW)",
                "p90": "P90 (kW)",
            }
        ),
        definition=(
            "Turn-down: every kW of unmanaged charging that could wait, whatever its slack "
            "(slack is how long the charging could wait before the EV is due to leave). "
            "Turn-up: spare charger power on the smart path, up to each EV's preferred target. "
            "One half-hour's power is not sustained delivery."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="supplier-availability",
    )
    st.caption(
        "Turn-down is read on the unmanaged path: charging that could wait. Turn-up is read on "
        "the smart path: charger power left idle."
    )

    rows = _curve_rows(result, half_hour=half_hour, day_type=day_type)
    at_zero = rows.loc[rows["metric"].eq("available_kw") & rows["threshold_gbp_per_mwh"].eq(0.0)]
    tiles = (
        ("Available at zero cost", at_zero.iloc[0] if len(at_zero) else None),
        ("Movable at any cost", _metric(rows, "movable_kw")),
        ("All unmanaged charging", _metric(rows, "charging_kw")),
    )
    for column, (label, row) in zip(kpi_columns(st, 3), tiles, strict=True):
        value, context = _kw_band(row)
        kpi(column, label, value, context=context)

    risk = rows["risk_charge_gbp_per_mwh"].iloc[0] if len(rows) else None
    risk_text = format_quantity(None if missing(risk) else float(risk), "£/MWh", decimals=0)
    chart_block(
        st,
        _curve_figure(rows),
        title=f"Turn-down available by cost to move, {half_hour}, "
        f"{DAY_TYPE_LABELS[day_type].lower()}",
        caption=f"Includes a {risk_text} early-departure risk charge. P10–P90 across {weeks} "
        f"weeks. {horizon_label('day_ahead').capitalize()}.",
        frame=_curve_table(rows),
        definition=(
            "Turn-down in each simulated week's own sessions, costed at the day-ahead prices "
            "known at the day-ahead decision (overnight half-hours use the typical shape). "
            "Cost to move is the cheapest later half-hour before departure minus this "
            "half-hour's price, plus the risk charge. Negative means moving saves money. This "
            "describes what was movable; it is not a forecast a supplier held at 13:00."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="supplier-cost-curve",
    )
    st.caption(
        "Cost to move = cheapest later half-hour before departure minus this one, plus the "
        "risk charge; negative means moving saves money."
    )


def _metric(rows: pd.DataFrame, metric: str) -> pd.Series | None:
    chosen = rows.loc[rows["metric"].eq(metric)]
    return chosen.iloc[0] if len(chosen) else None
