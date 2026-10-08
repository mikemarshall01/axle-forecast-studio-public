"""Drivers ▸ Archetypes: do the six archetypes reproduce what CNZ observed?

Answers Q5 (docs/DASHBOARD_DESIGN.md section 1) with a 3x2 grid of small
multiples built from ``result.average_day_bands`` (contract v2 section 3.8,
``group_id`` a cohort id, ``metric == "connected_share"``, the unmanaged
path (internal ``path_id == "normal"``),
``day_type == "all"``) plus the comparison table from ``result.cohort_summary``
(section 3.3). Chart audit B1: a run with fewer than six cohorts sampled (for
example 50 EVs can leave the smallest, 1%, archetype empty) must not crash;
empty cohorts are listed as "no EVs sampled" and draw no panel.

Decision 0004 item 43 adds a second 3x2 small-multiples chart, one panel per
cohort, for Battery SoC across the whole week (not folded to an average day,
unlike the Plugged-in row -- see ``_soc_small_multiples_figure``'s docstring
for why), with both bands: the always-on across-weeks P10-P90
(``result.cohort_interval_bands``, contract v2 section 3.7) and a lighter
across-EVs band broadcast from ``result.average_day_bands``' own across-EVs
SoC row (``spread == "across_evs"``), joined onto each week slot by
``result.study_slots``' ``local_half_hour`` -- a lookup, not a recomputed
percentile (contract v2 rule 3: this view never averages or sums a
percentile the model already returned). The comparison table's CNZ column
reads ``result.assumptions`` for the reported figure to compare against.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..components.chart_table import chart_block
from ..style import (
    CHART_HEIGHTS,
    MUTED_INK,
    PATH_LABELS,
    PATH_STYLES,
    UNAVAILABLE,
    assumption_value,
    band_and_line,
    band_fill,
    hover_time_labels,
    london_time_axis,
)

_GRID_ROWS, _GRID_COLS = 3, 2
_PANEL_GAP_PX = 36  # 3 x 180 + 2 x 36 = 612 total (polish plan G2 panel height)
_SMALL_MULTIPLE_HEIGHT = (
    _GRID_ROWS * CHART_HEIGHTS["small_multiple_panel"] + (_GRID_ROWS - 1) * _PANEL_GAP_PX
)

_TOP_MARGIN_PX = 32
"""B5: make_subplots anchors each row's title at the row's own domain top with
``yanchor="bottom"``, so the top row's title sits right at paper y=1 and grows
upward from there. The shared template's 8 px top margin left no room for that
text, clipping it; this gives it enough headroom."""

_X_TITLE_BOTTOM_MARGIN_PX = 76
"""The plugged-in grid's own bottom margin: a 24 px tick row, a 20 px row for
its "London half-hour of day" annotation, a 4 px gap and one 28 px legend row
(the single "Unmanaged" entry). Set explicitly because ``style_figure`` cannot
see an annotation used as an x-axis title (polish plan G2)."""

_X_TITLE_TICK_ROW_PX = 24
"""Pixel offset from the bottom panels' edge to that annotation: below the tick
labels at every figure height, unlike the earlier paper-fraction y (-0.32 of a
panel), which moved with the panel height."""

_PANEL_TITLE_MAX_CHARS = 20
"""Fits a small-multiple panel (~170 px wide at 390 px, design 3.5) without
clipping into the tick labels of the panel beside it -- a full label such as
"Intelligent Octopus average" would otherwise clip and bleed across the
boundary. The comparison table below keeps every cohort's full label; only
this chart's panel titles shorten."""

_TABLE_COLUMNS = {
    "cohort_label": "Archetype",
    "source_population_share": "Source share",
    "ev_count": "EVs in run",
    "plug_ins_per_ev_week_p50": "Plug-ins/week (P50)",
    "median_plug_in_soc_percent_p50": "Median plug-in SoC (P50)",
    "peak_plug_in_local_hour": "Peak plug-in hour (London)",
    "unserved_travel_kwh_p50": "Unserved travel, kWh/week (P50)",
}

_CNZ_SOC_COLUMN = "CNZ median plug-in SoC"
"""Goal review item 13: the lens question is "do the six archetypes
reproduce what CNZ observed?", but the table compared nothing with CNZ. This
column adds CNZ's own reported figure (``cnz_median_plug_in_soc_percent``,
contract v2 section 8.1) next to this run's own "Median plug-in SoC (P50)"
column, read from ``result.assumptions`` like every other CNZ reference in
the app (never hard-coded here) -- the same value on every row, since it is
a fleet-wide reported figure, not per-archetype. CNZ's report gives no
equally clean "plug-ins per week" figure to add alongside it (Table 2, p.11,
is a distribution of gaps between plug-ins, not a single count); that
finding is a caption note in ``render_archetypes`` instead of a second
invented column."""

# Fleet metric labels for the "one archetype across the week" expander
# (contract v2 section 3.5); presentation labels only, not model values.
_WEEK_METRICS = {
    "connected_share": ("Connected EVs (share)", "fraction"),
    "home_import_kwh": ("Home import energy", "kWh / half-hour"),
    "battery_soc_percent": ("Battery SoC (capacity-weighted)", "percent"),
    "driving_share": ("Driving (share)", "fraction"),
    "away_share": ("Away from home (share)", "fraction"),
    "unserved_travel_kwh": ("Unserved travel energy", "kWh / half-hour"),
}


_TICK_HALF_HOURS = (0, 12, 24, 36)
"""00:00, 06:00, 12:00, 18:00 (contract v2 section 3.2, same four-hour marks
as Overview's average-day x-axis). Milestone verifier, decision 0004 item 26:
the previous fifth tick (47 = 23:30) left five labels to fit across a ~170 px
390 px panel; with no explicit ``tickangle`` Plotly auto-rotated them, and
the rotated labels collided with each other and with the shared title below
the grid. Four evenly spaced ticks fit one line at ``tickangle=0``."""


def _half_hour_label(local_half_hour: int) -> str:
    """Render a 0-47 local half-hour index as "HH:MM" (contract v2 section 3.2)."""

    hour, half = divmod(local_half_hour, 2)
    return f"{hour:02d}:{'30' if half else '00'}"


def _panel_title(label: str, *, max_chars: int = _PANEL_TITLE_MAX_CHARS) -> str:
    """Shorten a cohort label to a panel title that fits without clipping.

    Cuts at the last whole word within ``max_chars`` rather than mid-word, so
    "Intelligent Octopus average" reads "Intelligent Octopus..." instead of a
    browser-clipped "Intelligent Octopus averag...".
    """

    if len(label) <= max_chars:
        return label
    cut = label[:max_chars]
    if label[max_chars] != " " and " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return f"{cut.rstrip()}…"


def _sampled_cohorts(cohort_summary: pd.DataFrame) -> list[tuple[str, str]]:
    rows = cohort_summary.loc[cohort_summary["ev_count"] > 0]
    return list(zip(rows["cohort_id"], rows["cohort_label"], strict=True))


def _small_multiples_figure(
    average_day_bands: pd.DataFrame, cohort_summary: pd.DataFrame
) -> go.Figure:
    labels = dict(zip(cohort_summary["cohort_id"], cohort_summary["cohort_label"], strict=True))
    ev_counts = dict(zip(cohort_summary["cohort_id"], cohort_summary["ev_count"], strict=True))
    cohort_order = list(cohort_summary["cohort_id"])
    figure = make_subplots(
        rows=_GRID_ROWS,
        cols=_GRID_COLS,
        shared_xaxes=True,  # unshared axes let each panel's auto-ticks
        # differ; sharing (by column) hides ticks above the bottom row and
        # matches them to it, so every panel shows the same ticks.
        shared_yaxes=True,
        subplot_titles=[_panel_title(labels[cohort_id]) for cohort_id in cohort_order],
        vertical_spacing=_PANEL_GAP_PX / _SMALL_MULTIPLE_HEIGHT,
    )
    # make_subplots gives its title annotations a 16 px font (bigger than the
    # 13 px body font, style.AXLE_TEMPLATE); at 390 px that crowds the y-tick
    # labels of the panel above, hence the smaller size below. Only these six
    # annotations exist yet -- "No EVs sampled" is added below with its own
    # font.
    figure.update_annotations(font_size=12)
    figure.update_layout(margin={"t": _TOP_MARGIN_PX, "b": _X_TITLE_BOTTOM_MARGIN_PX})
    colour = PATH_STYLES["normal"]["color"]
    band_named = False  # one legend entry (the first sampled panel) speaks for all six
    for index, cohort_id in enumerate(cohort_order):
        row, col = index // _GRID_COLS + 1, index % _GRID_COLS + 1
        if ev_counts[cohort_id] == 0:
            # B1: an archetype with no sampled EVs draws no panel data, only
            # the honest label; it never renders as a silent zero.
            figure.add_annotation(
                text="No EVs sampled",
                showarrow=False,
                row=row,
                col=col,
                font={"color": "#9AA3B2"},
            )
            continue
        rows = average_day_bands.loc[
            average_day_bands["group_id"].eq(cohort_id)
            & average_day_bands["path_id"].eq("normal")
            & average_day_bands["day_type"].eq("all")
            & average_day_bands["metric"].eq("connected_share")
        ].sort_values("local_half_hour")
        x = rows["local_half_hour"]
        low = rows["low"].to_numpy(dtype=float) * 100.0
        high = rows["high"].to_numpy(dtype=float) * 100.0
        centre = rows["centre"].to_numpy(dtype=float) * 100.0
        # Clarity critique: the raw 0-47 index in "Half-hour %{x}" read as a
        # meaningless number in hover, the same bug O4 already fixed on the
        # x-axis ticks below with this same helper.
        times = [_half_hour_label(value) for value in x]
        # One legend entry for band and line (polish plan G3), shown on the
        # first sampled panel only so the grid does not repeat it six times.
        band_and_line(
            figure,
            x,
            low,
            centre,
            high,
            name=PATH_LABELS["normal"],
            colour=colour,
            showlegend=not band_named,
            customdata=times,
            hovertemplate="%{customdata}<br>Plugged in: %{y:.0f}% (P50)<extra></extra>",
            row=row,
            col=col,
        )
        band_named = True
    # O3: every panel keeps the same stated 0-100% range, not an autoscaled one.
    figure.update_yaxes(range=[0, 100], automargin=True)
    figure.update_yaxes(title="% plugged in", col=1)
    # O4: label the shared ticks as a time of day; the bare half-hour index
    # (0-47) read as a meaningless number.
    figure.update_xaxes(
        tickvals=_TICK_HALF_HOURS,
        ticktext=[_half_hour_label(value) for value in _TICK_HALF_HOURS],
        # Milestone verifier, decision 0004 item 26: left to Plotly's own
        # choice, a narrow 390 px panel auto-rotates these labels to fit,
        # and the rotated text collided across panels and with the shared
        # title below. Horizontal is the fix, now that four ticks fit.
        tickangle=0,
        row=_GRID_ROWS,
        automargin=True,
    )
    # One title only: a copy per bottom-row column would collide at 390 px;
    # shared_xaxes above already gives every panel the same ticks. Centred
    # under both columns (paper x=0.5). Anchored to the bottom panels' edge
    # (paper y=0) and shifted down one tick row in pixels, so it sits under
    # the tick labels at any figure height (polish plan G2), not at a
    # fraction of the panel height that crowded the ticks at 390 px.
    figure.add_annotation(
        text="London half-hour of day",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0,
        yanchor="top",
        yshift=-_X_TITLE_TICK_ROW_PX,
        showarrow=False,
        font={"color": MUTED_INK, "size": 12},
    )
    return figure


_EV_BAND_ALPHA = 0.15
"""Fill opacity for the across-EVs SoC band, half of ``style.BAND_ALPHA``
(0.3, the across-weeks band's own opacity, drivers_fleet.py's identical
constant), so the two bands read as distinct layers."""


def _soc_week_rows(result: Any, cohort_id: str) -> pd.DataFrame:
    """This cohort's SoC across the whole week, both bands, one merged frame.

    The across-weeks band (``cohort_interval_bands``, contract v2 3.7) is
    already per-slot; the across-EVs band (``average_day_bands``' SoC row,
    ``spread == "across_evs"``, contract v2 3.8) is per half-hour-of-day
    only, so it is broadcast onto each matching week slot via
    ``study_slots.local_half_hour`` -- every day of the week gets the same
    typical-day population-spread shape (day_type "all", the same
    simplification the Plugged-in row above already makes). This is a plain
    lookup on the model's own output, not a recomputed percentile.
    """

    weeks = result.cohort_interval_bands
    weeks = (
        weeks.loc[
            weeks["cohort_id"].eq(cohort_id)
            & weeks["path_id"].eq("normal")
            & weeks["metric"].eq("battery_soc_percent")
        ]
        .sort_values("slot_index")
        .loc[:, ["slot_index", "interval_start_utc", "p10", "p50", "p90"]]
    )
    half_hours = result.study_slots.loc[:, ["slot_index", "local_half_hour"]]
    ev = result.average_day_bands
    ev = ev.loc[
        ev["group_id"].eq(cohort_id)
        & ev["path_id"].eq("normal")
        & ev["day_type"].eq("all")
        & ev["metric"].eq("battery_soc_percent")
    ].loc[:, ["local_half_hour", "low", "high"]]
    merged = weeks.merge(half_hours, on="slot_index", how="left").merge(
        ev, on="local_half_hour", how="left"
    )
    return merged.sort_values("slot_index")


def _soc_small_multiples_figure(result: Any, cohort_summary: pd.DataFrame) -> go.Figure:
    """SoC across the week, one panel per sampled cohort, both bands (item 43).

    Unlike the Plugged-in row (folded to a single average day), this stays
    on the full 336-slot week x-axis: SoC drifts over the week (depletion,
    top-ups) in a way a single "typical day" would hide, and it is the only
    axis the across-weeks band (``cohort_interval_bands``) is available on
    (contract v2 3.8 gives SoC no across-worlds average-day frame, only
    across-EVs -- see this module's docstring). A separate ``make_subplots``
    call from ``_small_multiples_figure``'s: mixing a 336-slot and a
    48-half-hour x-axis under one ``shared_xaxes`` grid would not line up.
    """

    labels = dict(zip(cohort_summary["cohort_id"], cohort_summary["cohort_label"], strict=True))
    ev_counts = dict(zip(cohort_summary["cohort_id"], cohort_summary["ev_count"], strict=True))
    cohort_order = list(cohort_summary["cohort_id"])
    figure = make_subplots(
        rows=_GRID_ROWS,
        cols=_GRID_COLS,
        shared_xaxes=True,
        subplot_titles=[_panel_title(labels[cohort_id]) for cohort_id in cohort_order],
        vertical_spacing=_PANEL_GAP_PX / _SMALL_MULTIPLE_HEIGHT,
    )
    figure.update_annotations(font_size=12)
    figure.update_layout(margin={"t": _TOP_MARGIN_PX})
    colour = PATH_STYLES["normal"]["color"]
    band_named = False
    x_reference = None
    for index, cohort_id in enumerate(cohort_order):
        row, col = index // _GRID_COLS + 1, index % _GRID_COLS + 1
        if ev_counts[cohort_id] == 0:
            figure.add_annotation(
                text="No EVs sampled", showarrow=False, row=row, col=col, font={"color": "#9AA3B2"}
            )
            continue
        rows = _soc_week_rows(result, cohort_id)
        if x_reference is None:
            x_reference = rows["interval_start_utc"]
        hover = hover_time_labels(rows["interval_start_utc"])
        # Drawn first (bottom layer), lighter: the across-EVs band is
        # typically far wider than the across-weeks band (decision 0004
        # item 39's own evidence), so the thin band stays visible on top of
        # it rather than the other way round.
        figure.add_trace(
            go.Scatter(
                x=rows["interval_start_utc"],
                y=rows["low"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        figure.add_trace(
            go.Scatter(
                x=rows["interval_start_utc"],
                y=rows["high"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=band_fill(colour, _EV_BAND_ALPHA),
                name="SoC P5–P95 across EVs",
                legendgroup="ev-band",
                showlegend=not band_named,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        band_and_line(
            figure,
            rows["interval_start_utc"],
            rows["p10"],
            rows["p50"],
            rows["p90"],
            name=PATH_LABELS["normal"],
            colour=colour,
            showlegend=not band_named,
            customdata=hover,
            hovertemplate="%{customdata}<br>SoC: %{y:.0f}% (P50)<extra></extra>",
            band_hovertemplates=(
                "%{customdata}<br>P10: %{y:.0f}%<extra></extra>",
                "%{customdata}<br>P90: %{y:.0f}%<extra></extra>",
            ),
            row=row,
            col=col,
        )
        band_named = True
    figure.update_yaxes(range=[0, 100], automargin=True)
    figure.update_yaxes(title="Battery SoC (%)", col=1)
    if x_reference is not None:
        # london_time_axis's default is one tick per day (7 for a week),
        # sized for a full-width chart; a ~170 px small-multiple column
        # (half of 390 px, this row's own width) overlaps all seven. Every
        # other day (4 ticks) matches the plugged-in row's own density
        # (_TICK_HALF_HOURS, also 4) and fits.
        axis = london_time_axis(x_reference)
        axis["tickvals"] = axis["tickvals"][::2]
        axis["ticktext"] = axis["ticktext"][::2]
        figure.update_xaxes(tickmode="array", **axis, row=_GRID_ROWS)
    return figure


def _format_or_unavailable(value: object, formatter) -> str:
    if value is None or pd.isna(value):
        return UNAVAILABLE
    return formatter(value)


# One formatter per numeric comparison-table column (chart audit B1: an
# archetype with no sampled EVs shows this whole row's figures, not per-cell
# "Unavailable", as "No EVs sampled" instead).
_ROW_FORMATTERS = {
    "Plug-ins/week (P50)": lambda value: f"{float(value):.1f}",
    "Median plug-in SoC (P50)": lambda value: f"{float(value):.0f}%",
    "Peak plug-in hour (London)": lambda value: f"{int(value):02d}:00",
    "Unserved travel, kWh/week (P50)": lambda value: f"{float(value):,.1f}",
}


def _comparison_table(
    cohort_summary: pd.DataFrame, *, cnz_median_plug_in_soc_percent: float | None = None
) -> pd.DataFrame:
    rows = cohort_summary.rename(columns=_TABLE_COLUMNS).copy()
    rows["Source share"] = (cohort_summary["source_population_share"] * 100.0).map("{:.0f}%".format)
    empty = cohort_summary["ev_count"].eq(0)
    for column, formatter in _ROW_FORMATTERS.items():
        raw = rows[column]
        rows[column] = raw.map(lambda value, fmt=formatter: _format_or_unavailable(value, fmt))
        rows.loc[empty, column] = "No EVs sampled"
    # Item 13: CNZ's own reported figure, the same value on every row (a
    # fleet-wide reported figure, not per-archetype) -- shown even for a row
    # with no EVs sampled, unlike the run's own columns above: the CNZ
    # benchmark does not depend on this run having sampled that cohort.
    rows[_CNZ_SOC_COLUMN] = _format_or_unavailable(
        cnz_median_plug_in_soc_percent, lambda value: f"{float(value):.0f}%"
    )
    return rows.loc[:, [*_TABLE_COLUMNS.values(), _CNZ_SOC_COLUMN]]


def _week_band_figure(rows: pd.DataFrame, *, unit: str) -> go.Figure:
    scale = 100.0 if unit == "fraction" else 1.0
    x = rows["interval_start_utc"]
    colour = PATH_STYLES["normal"]["color"]
    hover = hover_time_labels(x)
    p10 = rows["p10"].to_numpy(dtype=float) * scale
    p50 = rows["p50"].to_numpy(dtype=float) * scale
    p90 = rows["p90"].to_numpy(dtype=float) * scale
    figure = band_and_line(
        go.Figure(),
        x,
        p10,
        p50,
        p90,
        name=PATH_LABELS["normal"],
        colour=colour,
        customdata=hover,
        hovertemplate="%{customdata}<br>Median: %{y:,.1f}<extra></extra>",
        band_hovertemplates=(
            "%{customdata}<br>P10: %{y:,.1f}<extra></extra>",
            "%{customdata}<br>P90: %{y:,.1f}<extra></extra>",
        ),
    )
    # compact=True (goal review V3): the default "Mon 28"-style label left
    # as little as 2 px between adjacent days at 390 px on a plot this
    # width; a single week never repeats a weekday, so the day-of-month
    # adds width, not disambiguation.
    figure.update_xaxes(tickmode="array", **london_time_axis(x, compact=True))
    figure.update_yaxes(title="%" if unit in ("fraction", "percent") else unit)
    if unit == "percent":
        figure.update_yaxes(range=[0, 100])
    return figure


def _render_week_expander(st: Any, result: Any, sampled: list[tuple[str, str]]) -> None:
    with st.expander("One archetype across the week"):
        labels = dict(sampled)
        # Side by side, not one control per row (polish plan G8).
        cohort_column, metric_column = st.columns(2)
        cohort_id = cohort_column.selectbox(
            "Archetype",
            options=list(labels),
            format_func=lambda key: labels.get(key, key),
            key="archetypes-week-cohort",
        )
        metric = metric_column.selectbox(
            "Outcome metric",
            options=list(_WEEK_METRICS),
            format_func=lambda key: _WEEK_METRICS[key][0],
            key="archetypes-week-metric",
        )
        bands = result.cohort_interval_bands
        rows = bands.loc[
            bands["cohort_id"].eq(cohort_id)
            & bands["path_id"].eq("normal")
            & bands["metric"].eq(metric)
        ].sort_values("slot_index")
        metric_title, unit = _WEEK_METRICS[metric]
        world_count = int(rows["world_count"].iloc[0]) if not rows.empty else 0
        chart_block(
            st,
            _week_band_figure(rows, unit=unit),
            title=f"{metric_title} for {labels.get(cohort_id, cohort_id)} (unmanaged path)",
            caption=(
                f"Unmanaged path. Median and P10–P90 across {world_count} simulated weeks. "
                "Illustrative."
            ),
            frame=rows.rename(
                columns={
                    "interval_start_london": "Interval start (London)",
                    "interval_end_utc": "Interval end (UTC)",
                    "world_count": "Simulated weeks",
                    "mean": "Mean",
                    "p10": "P10",
                    "p50": "P50",
                    "p90": "P90",
                }
            ).loc[
                :,
                [
                    "Interval start (London)",
                    "Interval end (UTC)",
                    "Simulated weeks",
                    "Mean",
                    "P10",
                    "P50",
                    "P90",
                ],
            ],
            definition=(
                f"{metric_title} for this archetype's unmanaged path, one archetype at a time."
            ),
            height=CHART_HEIGHTS["time_series"],
            key=f"archetypes-week-chart-{cohort_id}-{metric}",
        )


def render_archetypes(st: Any, result: Any) -> None:
    """Render Drivers ▸ Archetypes: small multiples, an SoC row, and the comparison table."""

    cohort_summary = result.cohort_summary
    sampled = _sampled_cohorts(cohort_summary)
    # Item 13: read the run's own CNZ reference the way every other view
    # does (contract v2 section 8.1) -- never hard-coded here, so an edited
    # or future CNZ_CONTEXT value never goes stale on this page alone.
    cnz_soc = assumption_value(result, "cnz_median_plug_in_soc_percent")
    comparison_table = _comparison_table(cohort_summary, cnz_median_plug_in_soc_percent=cnz_soc)

    # Table first (polish plan G9): the lens question is "do the archetypes
    # reproduce what CNZ observed?", and the table is where that comparison is.
    st.dataframe(comparison_table, width="stretch", hide_index=True)
    # Item 13: CNZ also reports most EVs plugging in daily (CNZ May 2022
    # report p.11, "Most EVs are plugged in daily", corroborating the
    # archetypes' own 1.0/day source probability, decision 0004 item 42) --
    # a caption fact, not a second column, because the report gives a
    # distribution of gaps between plug-ins (Table 2), not a weekly count.
    st.caption(
        "CNZ column: comparison context, not a calibration target. CNZ also finds most EVs "
        "plug in daily.",
        help="From CNZ's May 2022 report: the median plug-in SoC, and the finding that most EVs "
        "are plugged in daily.",
    )

    figure = _small_multiples_figure(result.average_day_bands, cohort_summary)
    chart_block(
        st,
        figure,
        title="Average day, % plugged in, by archetype",
        caption=(
            "Unmanaged path. Median and P10–P90 across simulated weeks; spread across EVs not "
            "shown. Illustrative."
        ),
        frame=comparison_table,
        definition=(
            "Share of each archetype's EVs plugged in at home at each London half-hour, all day "
            "types, P50 across simulated weeks. Archetypes with no sampled EVs show no panel. "
            "An archetype near 100% overnight follows from its plug-in probability of 1.0 in "
            "Axle's sheet: once the battery reaches the preferred target, those EVs stay "
            "plugged in with nothing left to charge for the rest of the night."
        ),
        height=_SMALL_MULTIPLE_HEIGHT,
        key="archetypes-small-multiples",
    )

    if sampled:
        # Item 43: an SoC row, same grid, both bands (see
        # _soc_small_multiples_figure's docstring for why it stays on the
        # full week rather than folding to an average day like the row
        # above).
        soc_figure = _soc_small_multiples_figure(result, cohort_summary)
        chart_block(
            st,
            soc_figure,
            title="Battery SoC across the week, by archetype",
            caption=(
                "Darker band: P10–P90 across simulated weeks; lighter band: P5–P95 across EVs, "
                "same every day. Unmanaged path. Illustrative."
            ),
            frame=comparison_table,
            definition=(
                "Each archetype's battery SoC (state of charge, how full the battery is; "
                "weighted by battery capacity) across the study week: median and P10–P90 "
                "across simulated weeks, plus P5–P95 across the archetype's EVs. The across-EVs "
                "band is taken from the average-day pattern, so it repeats each day. Archetypes "
                "with no sampled EVs show no panel."
            ),
            height=_SMALL_MULTIPLE_HEIGHT,
            key="archetypes-soc-small-multiples",
        )

    if sampled:
        _render_week_expander(st, result, sampled)
    else:
        st.info("No archetype has a sampled EV in this run.")
