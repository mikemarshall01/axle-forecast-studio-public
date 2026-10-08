"""Drivers ▸ Sessions: how plug-in sessions are distributed for each archetype.

What this owns: one small-multiple chart per archetype (plan D-2, decision
0004 item 54) for one session metric at a time, read straight from
``result.session_distribution_bands`` (contract v2 3.10c). Bars are the P50
share of sessions in each bin across simulated weeks with P10–P90 whiskers;
the model computes each share per week first, so this view never recomputes
or adds a share (contract v2 rule 3: bin quantiles do not sum to 1).

Dashed markers are the workbook's per-archetype source values (contract v2
section 8.2): plug-in SoC on the SoC metric and battery kWh per plug-in on
Energy needed, read from ``result.assumptions`` by name. They are audit
references, not calibration targets: modelled plug-in SoC stays a stock-flow
output. A result without those records (the synthetic fixture, a direct
model call) simply shows no marker.

The caption states the horizon-end censoring the contract names: sessions
still plugged in when the study ends have no departure, so the dwell, slack,
departure and flexible-energy distributions lean short.

"SoC at departure, timed" (decision 0007, model step 2) reads the optional
third path's own stock at the same unplug, `departure_soc_percent_timed`.
Unlike the smart column, the model leaves this column out of the frame
altogether on a run whose `timed_start_local_hour` is unset (it is not
NaN-filled), so the option is only offered when a run actually has it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..components import header
from ..components.chart_table import chart_block
from ..registry import DAY_TYPE_KEY, DAY_TYPE_LABELS
from ..style import (
    ARCHETYPE_COLOURS,
    CHART_HEIGHTS,
    MUTED_INK,
    SERIES_COLOURS,
    assumption_value,
)

ARCHETYPE_ORDER = (
    "average_uk",
    "intelligent_octopus",
    "infrequent_charging",
    "infrequent_driving",
    "scheduled_charging",
    "always_plugged_in",
)
"""Source order (contract v2 rule 9); one panel each, sampled or not, so a
panel never moves when an archetype has no EVs in a small run."""

# label -> (metric in session_distribution_bands, x-axis title, source record
# suffix or None). Energy is battery-side in both the model and the
# workbook's kWh per plug-in, so they compare directly (contract v2 8.2).
METRICS: dict[str, tuple[str, str, str | None]] = {
    "Plug-in time": ("plug_in_time", "London time", None),
    "Departure time": ("departure_time", "London time", None),
    "SoC at plug-in": ("plug_in_soc_percent", "SoC at plug-in (%)", "source_plug_in_soc_percent"),
    "Energy needed": (
        "energy_needed_kwh",
        "Energy to target (kWh, battery)",
        "source_battery_kwh_per_plug_in",
    ),
    "Dwell": ("dwell_hours", "Hours plugged in", None),
    "Flexible kWh": ("flexible_kwh", "Flexible energy (kWh, battery)", None),
    "Slack": ("slack_hours", "Hours charging could wait", None),
    # Supplier contract v1 §4.6: the battery stock at unplug on each path, for
    # the same sessions (plug-in and unplug are exogenous). Timed tariff sits
    # between Unmanaged and Smart (on-screen path order, decision 0007).
    "SoC at departure": ("departure_soc_percent", "SoC at departure (%)", None),
    "SoC at departure, timed": (
        "departure_soc_percent_timed",
        "SoC at departure, timed (%)",
        None,
    ),
    "SoC at departure, smart": ("departure_soc_percent_smart", "SoC at departure, smart (%)", None),
}

SMART_METRICS = frozenset({"departure_soc_percent_smart"})
"""Metrics measured on the smart path: offered only on a smart charging result."""

TIMED_METRICS = frozenset({"departure_soc_percent_timed"})
"""Metrics measured on the optional timed path (decision 0007, model step 2):
offered only when a run's ``session_distribution_bands`` actually carries the
metric, since -- unlike the smart column -- the model leaves it out of the
frame rather than NaN-filling it on a run without the timed path."""

_COLUMNS = 2
_ROWS = 3
# Two panels a row (three rows) rather than three a row: at 390 px three
# panels would be about 110 px wide each, too narrow for 48 time bins.
_PANEL_GAP = 0.12
_BAR_WIDTH_SHARE = 0.85
"""Bar width as a share of its bin, leaving a hairline between bins."""

CENSORING_CAPTION = (
    "Sessions still plugged in when the week ends have no departure yet, so dwell, slack, "
    "departure and flexible kWh lean short."
)


def _labels(result: Any) -> dict[str, str]:
    summary = result.cohort_summary
    return dict(zip(summary["cohort_id"], summary["cohort_label"], strict=True))


_PANEL_TITLE_SHORT = {"intelligent_octopus": "IO average"}
"""Final critique B-16: at 390 px this chart's two-column subplot titles have
about 170 px each, and the full "Intelligent Octopus average · 300 EVs" ran
into its neighbour. Only this cohort's canonical name (``COHORT_SOURCE_NAMES``,
still used unabridged in the Group column and legend everywhere else) is long
enough to collide; the other five fit unchanged."""


def _panel_titles(result: Any) -> list[str]:
    """Archetype name and its sampled EV count, so a thin panel reads as a small group."""

    summary = result.cohort_summary.set_index("cohort_id")
    titles = []
    for cohort in ARCHETYPE_ORDER:
        if cohort not in summary.index:
            titles.append(cohort)
            continue
        count = int(summary.loc[cohort, "ev_count"])
        label = _PANEL_TITLE_SHORT.get(cohort, summary.loc[cohort, "cohort_label"])
        titles.append(f"{label} · {count:,} EV{'s' if count != 1 else ''}")
    return titles


def _rows(result: Any, *, group_id: str, day_type: str, metric: str) -> pd.DataFrame:
    frame = result.session_distribution_bands
    mask = (
        frame["group_id"].eq(group_id) & frame["day_type"].eq(day_type) & frame["metric"].eq(metric)
    )
    return frame.loc[mask].sort_values("bin_index").reset_index(drop=True)


def _timed_metric_present(result: Any) -> bool:
    """Whether this run's ``session_distribution_bands`` carries the optional
    timed path's departure SoC metric at all (decision 0007, model step 2):
    the model leaves the column, and so every row for it, out of the frame on
    a run whose ``timed_start_local_hour`` is unset, rather than NaN-filling
    it the way the always-present smart column is -- so, unlike
    ``SMART_METRICS``, this cannot be answered from ``result.model`` alone."""

    return bool(result.session_distribution_bands["metric"].isin(TIMED_METRICS).any())


def _source_value(result: Any, cohort_id: str, suffix: str | None) -> float | None:
    """The workbook source value for one archetype, or ``None`` when not carried."""

    if suffix is None:
        return None
    value = assumption_value(result, f"{cohort_id}.{suffix}")
    return None if value is None else float(value)


def _figure(result: Any, *, metric: str, axis_title: str, day_type: str, suffix: str | None):
    """Six panels (one per archetype), shared y, bars P50 with P10–P90 whiskers."""

    labels = _labels(result)
    figure = make_subplots(
        rows=_ROWS,
        cols=_COLUMNS,
        shared_xaxes="all",
        shared_yaxes="all",
        subplot_titles=_panel_titles(result),
        vertical_spacing=_PANEL_GAP,
        horizontal_spacing=0.06,
    )
    is_time = metric in ("plug_in_time", "departure_time")
    y_top = 0.0
    for index, (cohort_id, colour) in enumerate(
        zip(ARCHETYPE_ORDER, ARCHETYPE_COLOURS, strict=True)
    ):
        row, col = index // _COLUMNS + 1, index % _COLUMNS + 1
        rows = _rows(result, group_id=cohort_id, day_type=day_type, metric=metric)
        observed = not rows.empty and int(rows["world_count"].max()) > 0
        if observed:
            y_top = max(y_top, float(np.nanmax(rows["share_p90"].to_numpy(dtype=float))) * 100.0)
        if observed:
            lower = rows["bin_lower"].to_numpy(dtype=float)
            upper = rows["bin_upper"].to_numpy(dtype=float)
            p10, p50, p90 = (
                rows[stat].to_numpy(dtype=float) * 100.0
                for stat in ("share_p10", "share_p50", "share_p90")
            )
            figure.add_trace(
                go.Bar(
                    x=(lower + upper) / 2.0,
                    y=p50,
                    width=(upper - lower) * _BAR_WIDTH_SHARE,
                    marker_color=colour,
                    # Whiskers: P10–P90 across weeks around the P50 bar. Both
                    # come from the same per-week shares, so P50 sits inside.
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": np.maximum(p90 - p50, 0.0),
                        "arrayminus": np.maximum(p50 - p10, 0.0),
                        "color": MUTED_INK,
                        "thickness": 1,
                        "width": 0,
                    },
                    name=labels.get(cohort_id, cohort_id),
                    showlegend=False,
                    customdata=np.stack([rows["bin_label"], p10, p90], axis=-1),
                    hovertemplate=(
                        "%{customdata[0]}: %{y:.1f}% of sessions (P50; P10–P90 "
                        "%{customdata[1]:.1f}–%{customdata[2]:.1f}%)<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        else:
            # An archetype with no sampled EV (or no session of this day
            # type) has no rows or NaN shares: say so, never draw zeros.
            figure.add_annotation(
                text="No sessions",
                showarrow=False,
                font={"color": MUTED_INK},
                xref="x domain" if index == 0 else f"x{index + 1} domain",
                yref="y domain" if index == 0 else f"y{index + 1} domain",
                x=0.5,
                y=0.5,
            )
        source = _source_value(result, cohort_id, suffix)
        if source is not None and observed:
            figure.add_vline(
                x=source,
                line_dash="dash",
                line_width=1.5,
                line_color=SERIES_COLOURS["observed"],
                row=row,
                col=col,
            )
    if is_time:
        figure.update_xaxes(
            tickmode="array",
            # No 24:00 tick: at 390 px it ran into the next panel's 00:00.
            tickvals=[0, 6, 12, 18],
            ticktext=["00:00", "06:00", "12:00", "18:00"],
            range=[0, 24],
        )
    figure.update_xaxes(title_text=axis_title, row=_ROWS)
    figure.update_yaxes(title_text="% of sessions", col=1)
    # One explicit shared range up to the tallest whisker: in the browser the
    # matched axes autoranged from one panel only and clipped the others.
    figure.update_yaxes(range=[0.0, max(y_top, 1.0) * 1.05])
    # The template's 8 px top margin would clip the first row's panel titles.
    figure.update_layout(bargap=0, hovermode="closest", margin={"t": 28})
    # Subplot titles are annotations; the chart font keeps them at 12 px.
    figure.update_annotations(font_size=12)
    return figure


def _table(result: Any, *, metric: str, day_type: str) -> pd.DataFrame:
    frame = result.session_distribution_bands
    rows = frame.loc[frame["metric"].eq(metric) & frame["day_type"].eq(day_type)]
    labels = {"fleet": "Fleet", **_labels(result)}
    return (
        rows.assign(group_id=rows["group_id"].map(labels))
        .rename(
            columns={
                "group_id": "Group",
                "unit": "Unit",
                "bin_label": "Bin",
                "world_count": "Simulated weeks",
                "share_mean": "Share mean",
                "share_p10": "Share P10",
                "share_p50": "Share P50",
                "share_p90": "Share P90",
            }
        )
        .loc[
            :,
            [
                "Group",
                "Bin",
                "Unit",
                "Simulated weeks",
                "Share mean",
                "Share P10",
                "Share P50",
                "Share P90",
            ],
        ]
    )


def render_sessions(st: Any, result: Any) -> None:
    """Render Drivers ▸ Sessions: one session metric, six archetype panels."""

    # In the page header's controls slot (polish plan G8): only the shared
    # day-type control, full width, one column -- as Plug-ins does. Final
    # critique B-16: sharing that slot with the metric selectbox left the day
    # control too narrow for its own labels at 1440 px ("All" clipped off,
    # "Weekend" half visible). The metric selectbox is this page's own
    # control, not a shared one, so it sits on its own line under the
    # header's question caption instead, where it has the full body width.
    (day_column,) = header.controls(st).columns(1)
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
    day_type = day_type if day_type in DAY_TYPE_LABELS else "weekday"

    timed_present = _timed_metric_present(result)
    options = [
        label
        for label, (metric, _, _) in METRICS.items()
        if (result.model == "action" or metric not in SMART_METRICS)
        and (metric not in TIMED_METRICS or timed_present)
    ]
    metric_label = st.selectbox("Session metric", options=options, key="sessions-metric")
    metric_label = metric_label if metric_label in options else options[0]
    metric, axis_title, suffix = METRICS[metric_label]

    fleet_rows = _rows(result, group_id="fleet", day_type=day_type, metric=metric)
    world_count = int(fleet_rows["world_count"].max()) if not fleet_rows.empty else 0
    if metric in SMART_METRICS:
        path = "smart path"
    elif metric in TIMED_METRICS:
        path = "timed tariff path"
    else:
        path = "unmanaged path"
    caption = (
        f"{DAY_TYPE_LABELS[day_type]} sessions, {path}. Bars: P50 share per bin; "
        f"whiskers: P10–P90 across {world_count} weeks."
    )
    if metric in SMART_METRICS:
        caption += " Early departures cut the smart-path SoC."
    elif metric in TIMED_METRICS:
        caption += " Barred until the start time can cut SoC."
    if suffix is not None:
        caption += " Dashed: the value in Axle's sheet."
    chart_block(
        st,
        _figure(result, metric=metric, axis_title=axis_title, day_type=day_type, suffix=suffix),
        title=f"{metric_label} by archetype, % of sessions",
        caption=caption,
        frame=_table(result, metric=metric, day_type=day_type),
        definition=(
            "For each simulated week, the share of each archetype's plug-in sessions in each "
            "bin, then P10, P50 and P90 across weeks. The quantiles do not sum to 100%. Energy "
            "is measured at the battery. Dashed markers are the plug-in SoC (state of charge, "
            "how full the battery is) and kWh per plug-in from Axle's sheet: a check against "
            "the source, not a calibration target."
        ),
        height=CHART_HEIGHTS["small_multiple_panel"] * _ROWS + 90,
        key=f"sessions-chart-{metric}",
    )
    # Clarity critique: "Always plugged-in · 10 EVs" (and any other small
    # archetype) has whiskers taller than its bars, which read as a broken
    # panel -- it is a 10-EV group's own week-to-week spread, wider than a
    # 400-EV group's.
    st.caption(
        "Small groups have taller whiskers: a 10-EV archetype varies more week to week than a "
        "400-EV one."
    )
    # Always shown (contract v2 3.10c): it names the metrics it affects.
    st.caption(CENSORING_CAPTION)
