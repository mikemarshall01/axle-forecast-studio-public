"""Drivers ▸ Plug-ins: when EVs plug in, and at what battery level.

Answers Q1/Q2 (docs/DASHBOARD_DESIGN.md section 1) and closes chart audit B5
(no chart showed "SoC when they plug in"). Brief-first (decision 0004 item
16): these are the two literal asks from the Axle brief, built directly from
``result.plug_in_summary`` (contract v2 section 3.10), which is already a
per-world-then-quantile summary, so this view never recomputes a share or a
percentile itself (contract v2 rule 3) -- it only selects, renames and plots
the columns the model supplies. CNZ reference values come from
``result.assumptions`` (contract v2 section 8.1): they are comparison
context, not a calibration target, and the caption says so every time.
``result.plug_in_half_hour_heatmap`` and ``result.plug_out_heatmap``
(contract v2 3.10b, decision 0004 items 51, 54, plan D-1) drive the two
weekday x half-hour heatmaps, plug-in and departure (unplug = departure,
item 51), on one shared colour scale; the hourly ``plug_in_heatmap`` is no
longer read. ``result.plug_in_events``, ``result.sampled_world_ids`` and
``result.world_count`` back the sampled plug-in events expander at the
foot of the page, whose table doubles as the per-agent CSV export the brief
asks for ("this is the kind of data we want to get out for each agent"):
``_sampled_events_table`` builds the frame once and both the on-screen table
and the download button read it, so the two can never drift apart.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from axle_studio.model.summaries import PLUG_IN_EVENT_WORLD_LIMIT

from .. import usage_log
from ..components import header
from ..components.chart_table import chart_block
from ..components.kpi import kpi, kpi_columns
from ..registry import DAY_TYPE_KEY, DAY_TYPE_LABELS
from ..style import (
    CHART_HEIGHTS,
    MUTED_INK,
    SEQUENTIAL,
    SERIES_COLOURS,
    UNAVAILABLE,
    assumption_value,
    format_quantity,
    percent,
)

# (metric name in plug_in_summary.kpis, tile label, CNZ assumption name or
# None). The first tile reads per-EV (fleet totals do not say whether that
# is many EVs plugging in once or few plugging in often); there is no CNZ
# figure for this metric (report p.13 gives SoC and hour context only, not
# a plug-ins-per-week count), so it carries no comparison. Plug-ins below
# 10% SoC stays the fleet-wide (day_type == "all") figure, matching the
# wireframe in DASHBOARD_DESIGN.md 4.2: it is a structural-zero check
# (_ZERO_BY_CONSTRUCTION_METRIC below), not a rate that means anything split
# by day type. Plug-ins per EV per week and median SoC at plug-in both
# follow this page's own Weekday/Weekend toggle instead (plug_in_world_kpis
# breaks both out by day_type, contract v2 3.10) and label which day type,
# so the SoC tile agrees with Overview's same-named tile, which follows
# Overview's own day-type toggle the same way; either tile falls back to
# the fleet-wide figure, labelled "(All days)", if a result's kpis frame
# has no by-day row for it -- see _render_kpi_row.
_KPI_METRICS: tuple[tuple[str, str, str | None], ...] = (
    ("plug_ins_per_ev_per_week", "Plug-ins per EV per week", None),
    ("median_plug_in_soc_percent", "Median SoC at plug-in", "cnz_median_plug_in_soc_percent"),
    (
        "share_below_10_percent_soc",
        "Low-battery plug-ins",
        "cnz_share_plug_ins_below_10_percent_soc",
    ),
)
_DAY_TYPE_FOLLOWING_METRICS = frozenset({"plug_ins_per_ev_per_week", "median_plug_in_soc_percent"})
# The no-stranding top-up rule (decision 0004 items 32, 37, 41) tops up
# before any trip that would leave less than 10% SoC, so arrival, and so
# plug-in, SoC should never fall below that threshold here -- a structural
# zero, not a modelled finding to compare against CNZ's observed 3% (goal
# review item 8's "or '0% by construction' as appropriate").
_ZERO_BY_CONSTRUCTION_METRIC = "share_below_10_percent_soc"
_ZERO_TOLERANCE = 1e-9


def _cnz_text(value: Any | None) -> str | None:
    """Render a CNZ assumption for the KPI caption; values are percent or fraction."""

    if value is None:
        return None
    # CNZ fractions (e.g. 0.03) and percents (e.g. 52.0) both read as "NN%";
    # the assumption's own unit tells us which scale it is already on.
    points = value * 100.0 if value <= 1.0 else value
    return f"CNZ {percent(points)}"


def _format_kpi(row: pd.Series) -> str:
    value = row["p50"]
    if pd.isna(value):
        return UNAVAILABLE
    if row["unit"] == "plug-ins":
        return f"{value:,.0f} (P50)"
    if row["unit"] == "plug-ins per EV":
        # No "(P50)" suffix here, matching Overview's own per-EV tile (the
        # spread lives in help, not the value text).
        return f"{value:.1f}"
    if row["unit"] == "fraction":
        return percent(value * 100, decimals=1)
    if row["unit"] == "percent":
        return percent(value, decimals=1)
    return format_quantity(value, row["unit"], decimals=1)


_ZERO_BY_CONSTRUCTION_HELP = (
    "The top-up rule: before any trip that would take the battery below the top-up threshold "
    "(10% by default), the EV tops up at a public charger. While that threshold is 10% or "
    "more, no plug-in can start below 10% SoC in this model. The zero is built in, not a "
    "finding to compare with CNZ's observed figure."
)


def _render_kpi_row(st: Any, result: Any, *, day_type: str, day_label: str) -> None:
    kpis = result.plug_in_summary.kpis
    fleet = kpis.loc[kpis["day_type"].eq("all")].set_index("metric")
    by_day = kpis.loc[kpis["day_type"].eq(day_type)].set_index("metric")
    columns = kpi_columns(st, len(_KPI_METRICS))
    # The day type is each tile's context line, not part of its label
    # (polish plan G7: labels at most 28 characters, no units or qualifiers).
    for column, (metric, label, cnz_name) in zip(columns, _KPI_METRICS, strict=True):
        tile_label = label
        if metric in _DAY_TYPE_FOLLOWING_METRICS and metric in by_day.index:
            source, scope = by_day, day_label
        elif metric in _DAY_TYPE_FOLLOWING_METRICS:
            # The data does not allow this metric to follow the toggle
            # (no by-day row for it) -- fall back to the fleet-wide
            # figure and say so, rather than silently showing a
            # day-specific-looking tile next to an all-days number
            # (goal review N12).
            source, scope = fleet, "All days"
        elif metric == _ZERO_BY_CONSTRUCTION_METRIC:
            source, scope = fleet, "Below 10% SoC"
        else:
            source, scope = fleet, None
        if metric not in source.index:
            kpi(column, tile_label, UNAVAILABLE, context=scope)
            continue
        row = source.loc[metric]
        if (
            metric == _ZERO_BY_CONSTRUCTION_METRIC
            and not pd.isna(row["p50"])
            and abs(row["p50"]) <= _ZERO_TOLERANCE
        ):
            # The number stays a number; "by construction" is its context
            # line, not part of the 26 px value (polish plan G5).
            kpi(
                column,
                tile_label,
                "0%",
                context=f"{scope}, by construction",
                help=_ZERO_BY_CONSTRUCTION_HELP,
            )
            continue
        cnz = _cnz_text(assumption_value(result, cnz_name)) if cnz_name else None
        context = " · ".join(part for part in (scope, cnz) if part) or None
        kpi(column, tile_label, _format_kpi(row), context=context)


def _whiskers(mean: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> dict[str, object]:
    """Asymmetric error-bar spec around the mean bar height.

    ``share_mean`` (the bar) and ``share_p10``/``share_p90`` (the whiskers)
    are different statistics of the same per-world shares (contract v2
    section 3.10), so the mean is not guaranteed to sit inside its own
    P10-P90 band; the arrays are clipped at zero so Plotly never draws a
    negative error bar.
    """

    return {
        "type": "data",
        "symmetric": False,
        "array": np.maximum(p90 - mean, 0.0),
        "arrayminus": np.maximum(mean - p10, 0.0),
        # Goal review item 12: SERIES_COLOURS["grid"] (#232A36) is the
        # gridline colour, tuned to be nearly invisible on the dark canvas --
        # it made the whiskers themselves nearly invisible too.
        "color": MUTED_INK,
    }


def _hour_figure(hour_bands: pd.DataFrame, *, day_type: str, cnz_hour: int | None) -> go.Figure:
    rows = hour_bands.loc[hour_bands["day_type"].eq(day_type)].sort_values("local_hour")
    hours = rows["local_hour"].to_numpy(dtype=float)
    mean = rows["share_mean"].to_numpy(dtype=float) * 100.0
    p10 = rows["share_p10"].to_numpy(dtype=float) * 100.0
    p90 = rows["share_p90"].to_numpy(dtype=float) * 100.0
    figure = go.Figure(
        go.Bar(
            x=hours,
            y=mean,
            error_y=_whiskers(mean, p10, p90),
            marker_color=SERIES_COLOURS["normal"],
            name="Share of plug-ins (mean across weeks)",
            hovertemplate="Hour %{x:02.0f}:00 London<br>Share: %{y:.1f}% (mean, whiskers "
            "P10–P90 across weeks)<extra></extra>",
        )
    )
    if cnz_hour is not None:
        # CNZ reports a weekday plug-in mode "about 18:00"; the marker is
        # comparison context (contract v2 section 8.1), never a target line.
        figure.add_vline(x=cnz_hour, line_dash="dash", line_color=SERIES_COLOURS["observed"])
        # B2: add_vline's own annotation is pinned to the axis's "y domain"
        # top edge (yanchor="bottom"), so its text renders above the plot and
        # was clipped by the template's 8 px top margin. A separate
        # paper-anchored annotation sits inside the plot instead.
        # cnz_hour (18) sits close to the axis's right edge (range
        # [-0.5, 23.5]), so a left-anchored label would grow off the plot at
        # both widths. Right-anchoring grows the text into the roomier left
        # side instead, with a small xshift so it clears the dashed line.
        # Goal review item 12: cnz_hour usually sits at or near the chart's
        # own tallest bar (CNZ's reported mode and this model's peak both
        # land near 18:00), so the label sat ON that bar's fill. A background
        # chip lifts the text off the bar visually, readable regardless of
        # what is drawn underneath it.
        figure.add_annotation(
            x=cnz_hour,
            y=0.95,
            xref="x",
            yref="paper",
            yanchor="top",
            xanchor="right",
            xshift=-6,
            showarrow=False,
            text="CNZ weekday peak ~18:00",
            font={"color": SERIES_COLOURS["observed"]},
            bgcolor="rgba(14,17,23,0.85)",
            borderpad=3,
        )
    figure.update_xaxes(title="London hour", dtick=2, range=[-0.5, 23.5])
    figure.update_yaxes(title="Share of plug-ins (%)", rangemode="tozero")
    return figure


def _soc_figure(soc_bands: pd.DataFrame, *, day_type: str, cnz_percent: float | None) -> go.Figure:
    rows = soc_bands.loc[soc_bands["day_type"].eq(day_type)].sort_values("soc_bin_lower_percent")
    midpoints = (
        rows["soc_bin_lower_percent"].to_numpy(dtype=float)
        + rows["soc_bin_upper_percent"].to_numpy(dtype=float)
    ) / 2.0
    mean = rows["share_mean"].to_numpy(dtype=float) * 100.0
    p10 = rows["share_p10"].to_numpy(dtype=float) * 100.0
    p90 = rows["share_p90"].to_numpy(dtype=float) * 100.0
    labels = [
        f"{int(lower)}–{int(upper)}%"
        for lower, upper in zip(
            rows["soc_bin_lower_percent"], rows["soc_bin_upper_percent"], strict=True
        )
    ]
    figure = go.Figure(
        go.Bar(
            x=midpoints,
            y=mean,
            width=4.5,
            error_y=_whiskers(mean, p10, p90),
            marker_color=SERIES_COLOURS["normal"],
            name="Share of plug-ins (mean across weeks)",
            customdata=labels,
            hovertemplate=(
                "SoC %{customdata}<br>Share: %{y:.1f}% (mean, whiskers P10–P90 across weeks)"
                "<extra></extra>"
            ),
        )
    )
    if cnz_percent is not None:
        figure.add_vline(x=cnz_percent, line_dash="dash", line_color=SERIES_COLOURS["observed"])
        # B2: see the matching annotation in _hour_figure -- inside the plot
        # (paper y <= 0.95) rather than clipped above it; same background
        # chip (item 12) so the label reads off whichever bar sits under it.
        figure.add_annotation(
            x=cnz_percent,
            y=0.95,
            xref="x",
            yref="paper",
            yanchor="top",
            xanchor="left",
            showarrow=False,
            text=f"CNZ observed median {cnz_percent:.0f}%",
            font={"color": SERIES_COLOURS["observed"]},
            bgcolor="rgba(14,17,23,0.85)",
            borderpad=3,
        )
    # Goal review item 4: the preferred target is 80% SoC (decision 0004
    # item 42), so the 80-90% bins hold every EV that plugged in already at
    # or near target -- a large bar there is not a realism finding, it is
    # "reached target, nothing left to charge", and needs to say so rather
    # than let it read as an unexplained spike.
    target_row = rows.loc[rows["soc_bin_lower_percent"].eq(80)]
    if not target_row.empty:
        target_x = (
            target_row["soc_bin_lower_percent"].iat[0] + target_row["soc_bin_upper_percent"].iat[0]
        ) / 2.0
        # Browser check at 390 px: the CNZ label (xanchor="left", unbounded
        # text width, y=0.95) can run wide enough at a narrow chart width to
        # reach this bin's x position, so this annotation sits well below
        # that row (not just further along it) to stay clear at every width.
        figure.add_annotation(
            x=target_x,
            y=0.5,
            xref="x",
            yref="paper",
            yanchor="bottom",
            xanchor="center",
            showarrow=False,
            text="At target,<br>nothing to charge",
            font={"color": MUTED_INK},
            bgcolor="rgba(14,17,23,0.85)",
            borderpad=3,
        )
    figure.update_xaxes(title="SoC at plug-in (%)", range=[0, 100])
    figure.update_yaxes(title="Share of plug-ins (%)", rangemode="tozero")
    return figure


_WEEKDAY_ORDER = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_HALF_HOURS = list(range(48))
_TIME_TICKS = list(range(0, 48, 12))
"""Every 6 hours: at 390 px, 3-hourly labels rotated and crowded the axis."""

# Heatmap choice (plan D-1: "a toggle at 390 px"). Streamlit cannot read the
# viewport, so the toggle shows at every width: "Both" puts the two side by
# side on a desktop (they stack on a phone), and one heatmap at a time gives
# a phone its full width.
_HEATMAP_CHOICES = ("Both", "Plug-in", "Departure")
_HEATMAPS = {
    # choice -> (result field, title, hover verb)
    "Plug-in": ("plug_in_half_hour_heatmap", "When drivers plug in", "plugs in"),
    "Departure": ("plug_out_heatmap", "When drivers leave (unplug)", "unplugs"),
}


def _pivot(heatmap: pd.DataFrame, stat: str) -> np.ndarray:
    """One statistic as a 7 x 48 array (Monday first), in % per EV-day."""

    table = heatmap.pivot(index="weekday_label", columns="local_half_hour", values=stat)
    return table.reindex(index=_WEEKDAY_ORDER, columns=_HALF_HOURS).to_numpy(dtype=float) * 100.0


def _shared_colour_top(heatmaps: list[pd.DataFrame]) -> float:
    """Top of the one colour scale both heatmaps use (plan D-1), in %.

    The largest P50 cell of either heatmap, so the same colour means the same
    rate on both and switching the toggle never rescales the colours.
    """

    tops = [float(np.nanmax(frame["p50"].to_numpy(dtype=float), initial=0.0)) for frame in heatmaps]
    return max(max(tops) * 100.0, 1e-9)


def _heatmap_figure(heatmap: pd.DataFrame, *, verb: str, colour_top: float) -> go.Figure:
    """Events per EV-day by weekday x London half-hour (contract v2 3.10b).

    A Plotly heatmap, not six weekday pies (decision 0004 item 43: "no pie
    charts"). ``p50`` is the colour; ``p10``/``p90`` ride along in
    ``customdata`` for hover only, so the chart never adds cells' quantiles
    (contract v2 3.10b's own rule). A cell with no study slot (the skipped
    spring half-hour) is NaN and draws blank, never as a zero.
    """

    labels = [f"{half // 2:02d}:{30 * (half % 2):02d}" for half in _HALF_HOURS]
    customdata = np.stack([_pivot(heatmap, "p10"), _pivot(heatmap, "p90")], axis=-1)
    figure = go.Figure(
        go.Heatmap(
            x=_HALF_HOURS,
            y=list(_WEEKDAY_ORDER),
            z=_pivot(heatmap, "p50"),
            customdata=customdata,
            zmin=0.0,
            zmax=colour_top,
            # The shared sequential scale (polish plan G4): teal is reserved
            # for the smart path, and these charts show unmanaged behaviour.
            colorscale=SEQUENTIAL,
            colorbar={"title": "%", "ticksuffix": "%", "thickness": 10},
            # Clock labels for hover only: a heatmap draws ``text`` in the
            # cells only when given a texttemplate.
            text=np.tile(labels, (len(_WEEKDAY_ORDER), 1)),
            hovertemplate=(
                f"%{{y}} %{{text}} London<br>Chance an EV {verb}: %{{z:.1f}}% (P50; "
                "P10–P90 %{customdata[0]:.1f}–%{customdata[1]:.1f}%)<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(
        title="London time",
        tickmode="array",
        tickvals=_TIME_TICKS,
        ticktext=[labels[tick] for tick in _TIME_TICKS],
        tickangle=0,
    )
    # Calendar convention: Monday at the top, reading down to Sunday.
    figure.update_yaxes(autorange="reversed")
    # The template's unified hover would head each label with the x index
    # ("37"); per-cell hover shows the clock time from ``text`` instead.
    figure.update_layout(hovermode="closest")
    return figure


def _heatmap_table(heatmap: pd.DataFrame) -> pd.DataFrame:
    return heatmap.rename(
        columns={
            "weekday_label": "Weekday",
            "local_time_label": "London time",
            "slot_count": "Study half-hours in this cell",
            "world_count": "Simulated weeks",
            "mean": "Mean",
            "p10": "P10",
            "p50": "P50",
            "p90": "P90",
        }
    ).drop(columns=["weekday", "local_half_hour"])


def _render_heatmaps(st: Any, result: Any) -> None:
    """Plug-in and departure heatmaps on one colour scale (plan D-1)."""

    frames = {choice: getattr(result, field) for choice, (field, _, _) in _HEATMAPS.items()}
    colour_top = _shared_colour_top(list(frames.values()))
    choice = st.segmented_control(
        "Heatmap",
        options=list(_HEATMAP_CHOICES),
        default=_HEATMAP_CHOICES[0],
        required=True,
        key="plug-ins-heatmap-choice",
        label_visibility="collapsed",
    )
    shown = list(_HEATMAPS) if choice not in _HEATMAPS else [choice]
    world_count = int(frames["Plug-in"]["world_count"].iat[0]) if len(frames["Plug-in"]) else 0
    columns = st.columns(len(shown))
    for column, name in zip(columns, shown, strict=True):
        _, title, verb = _HEATMAPS[name]
        with column:
            chart_block(
                st,
                _heatmap_figure(frames[name], verb=verb, colour_top=colour_top),
                title=f"{title}, % of EVs per half-hour",
                caption=(
                    f"Unmanaged path. P50 across {world_count} simulated weeks; hover for "
                    "P10–P90. Both heatmaps share one colour scale."
                ),
                frame=_heatmap_table(frames[name]),
                definition=(
                    f"Events in each weekday and London half-hour divided by EV-days (fleet size "
                    f"× study half-hours in the cell): the chance an EV {verb} then. A departure "
                    "is the moment the EV unplugs. Sessions still plugged in when the week ends "
                    "have no departure."
                ),
                height=CHART_HEIGHTS["histogram"],
                key=f"plug-ins-heatmap-{name}",
            )


def _hour_table(hour_bands: pd.DataFrame, day_type: str) -> pd.DataFrame:
    rows = hour_bands.loc[hour_bands["day_type"].eq(day_type)].sort_values("local_hour")
    return rows.rename(
        columns={
            "local_hour": "London hour",
            "world_count": "Simulated weeks with a plug-in",
            "count_p50": "Plug-ins (P50)",
            "share_mean": "Share mean",
            "share_p10": "Share P10",
            "share_p50": "Share P50",
            "share_p90": "Share P90",
        }
    ).drop(columns="day_type")


def _soc_table(soc_bands: pd.DataFrame, day_type: str) -> pd.DataFrame:
    rows = soc_bands.loc[soc_bands["day_type"].eq(day_type)].sort_values("soc_bin_lower_percent")
    return rows.rename(
        columns={
            "soc_bin_lower_percent": "SoC bin lower (%)",
            "soc_bin_upper_percent": "SoC bin upper (%)",
            "world_count": "Simulated weeks with a plug-in",
            "share_mean": "Share mean",
            "share_p10": "Share P10",
            "share_p50": "Share P50",
            "share_p90": "Share P90",
        }
    ).drop(columns="day_type")


def render_plug_ins(st: Any, result: Any) -> None:
    """Render Drivers ▸ Plug-ins: KPI tiles, the two brief histograms and two heatmaps."""

    # The shell draws the title, lens control and question line; this view's
    # own control (the shared day-type control, plan E2) sits at the right
    # end of that line. One session key across lenses, so Weekday chosen here
    # is still chosen on Overview and Sessions.
    # In the page header's controls slot (polish plan G8); one column so the
    # control keeps the layout calls a test's fake ``st`` records.
    (control_column,) = header.controls(st).columns(1)
    with control_column:
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
    day_label = DAY_TYPE_LABELS[day_type]

    _render_kpi_row(st, result, day_type=day_type, day_label=day_label)

    hour_bands = result.plug_in_summary.hour_bands
    soc_bands = result.plug_in_summary.soc_bands
    cnz_hour = assumption_value(result, "cnz_weekday_plug_in_mode_local_hour")
    cnz_soc = assumption_value(result, "cnz_median_plug_in_soc_percent")
    world_count = int(hour_bands["world_count"].max()) if not hour_bands.empty else 0
    # Goal review item 12: one plain caption for the pair. The bars are
    # shares of a whole (they add to 100% across each chart); the whiskers
    # are a separate week-to-week spread. The CNZ page reference lives in
    # the definition, not on screen (plan G7).
    caption = (
        f"Unmanaged path. Bars: mean share of plug-ins (add to 100%); whiskers: P10–P90 across "
        f"{world_count} weeks. CNZ is context only."
    )

    left, right = st.columns(2)
    with left:
        chart_block(
            st,
            _hour_figure(
                hour_bands, day_type=day_type, cnz_hour=cnz_hour if day_type == "weekday" else None
            ),
            title="Plug-in time (London), share per hour",
            caption=caption,
            frame=_hour_table(hour_bands, day_type),
            definition=(
                "Share of this day type's unmanaged-path plug-ins starting in each London hour, "
                "mean across simulated weeks with a P10–P90 band. CNZ's observed figure is "
                "context, not a calibration target."
            ),
            height=CHART_HEIGHTS["histogram"],
            key="plug-ins-hour-chart",
        )
    with right:
        chart_block(
            st,
            _soc_figure(soc_bands, day_type=day_type, cnz_percent=cnz_soc),
            title="SoC at plug-in, share per 5% bin",
            caption=caption,
            frame=_soc_table(soc_bands, day_type),
            definition=(
                "Share of this day type's unmanaged-path plug-ins landing in each 5-point SoC bin "
                "(SoC is state of charge, how full the battery is), mean across simulated weeks "
                "with a P10–P90 band. CNZ's observed figure is context, not a calibration target."
            ),
            height=CHART_HEIGHTS["histogram"],
            key="plug-ins-soc-chart",
        )

    _render_heatmaps(st, result)
    _render_sampled_events(st, result)


def _sampled_events_table(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.sort_values(["world_id", "unit_id", "plug_in_utc"])
        # Display-only conversion of the stored UTC column, the same way
        # ``plug_in_utc`` is already shown in London time beside it (clarity
        # critique: plug-in and plug-out in different timezones on one row
        # read as a bug). ``plug_out_utc`` is NaT for a still-open session;
        # tz_convert leaves NaT as NaT.
        .assign(plug_out_london=lambda frame: frame["plug_out_utc"].dt.tz_convert("Europe/London"))
        .rename(
            columns={
                "world_id": "Simulated week",
                "unit_id": "EV",
                "cohort_id": "Archetype",
                "plug_in_london": "Plug-in (London)",
                "plug_in_soc_percent": "SoC at plug-in (%)",
                "plug_out_london": "Plug-out (London)",
                "plug_out_soc_percent": "SoC at plug-out (%)",
                "home_import_kwh": "Home import (kWh)",
                "battery_added_kwh": "Battery added (kWh)",
                "still_plugged_at_horizon_end": "Still plugged at end",
            }
        )
        .loc[
            :,
            [
                "Simulated week",
                "EV",
                "Archetype",
                "Plug-in (London)",
                "SoC at plug-in (%)",
                "Plug-out (London)",
                "SoC at plug-out (%)",
                "Home import (kWh)",
                "Battery added (kWh)",
                "Still plugged at end",
            ],
        ]
    )


def _events_title(sampled: int, total: int) -> str:
    # Goal review item 20: "sample of 5 of 5" reads oddly when every week is shown.
    if sampled == total:
        return f"Plug-in events (all {total} simulated weeks)"
    return f"Plug-in events (sample of {sampled} of {total} simulated weeks)"


def _events_download_name(result: Any) -> str:
    """``axle_plug_in_events_<study start date>_seed<seed>.csv``.

    Matches ``supplier_positions.download_name``'s naming so every per-agent
    export in the app is found the same way in a downloads folder.
    """

    return f"axle_plug_in_events_{result.study_start_local_date:%Y-%m-%d}_seed{result.seed}.csv"


def _render_sampled_events(st: Any, result: Any) -> None:
    """Fleet-wide plug-in sessions, a sample of at most 10 worlds (contract v2 section 3.9).

    ``sampled_world_ids`` names exactly which weeks; every KPI and histogram
    above is computed from ``plug_in_summary``/``plug_in_world_kpis`` over
    every world instead, so this table is the only place "sampled" applies.
    The CSV download reads the same built table as the on-screen dataframe
    (built once, below), so the two can never show different rows or names
    -- the brief asks for this per-agent data, and a viewer who exports it
    must get exactly what they saw.
    """

    sampled_ids = result.sampled_world_ids
    with st.expander(_events_title(len(sampled_ids), result.world_count)):
        st.caption(
            "Sampled weeks: " + ", ".join(str(world_id) for world_id in sampled_ids) + ". "
            "Tiles and charts above use every simulated week, not this sample."
        )
        table = _sampled_events_table(result.plug_in_events)
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button(
            "Download plug-in events (CSV)",
            data=table.to_csv(index=False),
            file_name=_events_download_name(result),
            mime="text/csv",
            key="plug-ins-events-download",
            help=(
                "One row per plug-in session for each EV, the sampled weeks only "
                f"({len(sampled_ids)} of {result.world_count} simulated, at most "
                f"{PLUG_IN_EVENT_WORLD_LIMIT}), unmanaged path."
            ),
            on_click=usage_log.log_event,
            args=(st.session_state, "csv_download"),
            kwargs={"which": "plug-in events"},
        )
