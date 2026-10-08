"""Drivers ▸ Fleet week: how the fleet moves through the week, and how much charging can wait.

What this owns: one metric at a time across the simulated week, from a
grouped metric control. The *behaviour* metrics (Plugged in, Home import,
Battery SoC, Travel) read ``result.fleet_interval_bands`` for the fleet or
``result.cohort_interval_bands`` for one archetype (the Group selector,
decision 0004 item 54 and polish plan E2: shown only here, where the frame
has a group key), with ``result.weekly_bands`` for the unserved-travel
caption, ``result.fleet_interval_ev_bands`` for Battery SoC's across-EVs
spread (items 39, 40) and ``result.cohort_summary`` for archetype names.
The *flexibility* metrics, merged in from the retired Flexibility lens
(item 54, plan C1-C3, G10), are fleet-wide: Deferrable power by slack reads
``result.deferrable_power_bands`` (contract v2 3.6d), Turn-up headroom and
Movable energy read ``result.flexibility_bands`` (3.6c). Every frame is
already per-world then quantiles, so this view plots model columns and
never recomputes a percentile (contract v2 rule 3).

Paths: the behaviour metrics, Deferrable power and Movable energy show the
unmanaged path (internal ``path_id == "normal"``; the design's streamlining
pass keeps unmanaged-vs-smart on Smart charging ▸ Response, and item 50
names the path "Unmanaged" on screen). Turn-up headroom is shown for the
smart path only (item 54, plan C3): on the unmanaged path it is the
deferrable power minus the draw, which Deferrable power already shows.

Decision 0004 item 40 amends item 39: the across-weeks P10-P90 band is always
shown, even where it is thin (a thin band is itself the finding). Battery
SoC adds a second, lighter band for the spread across EVs, named separately
in the legend and caption; it has no median line of its own (see
``_ev_band_fill``). Home import does not get this second band: goal review
item 16 found it mostly 0 at P10/P50 and dropped it.

The retired Time slack chart (plan C1: mostly the departure clock counting
down, over a population that changes every half-hour, carrying no kW) keeps
its rows in the Deferrable power data expander.

The *network* metric, Zone import (decision 0004 item 55; trading contract
v1 §3, §5.7, §5.8, §6), reads ``result.zone_import_bands`` and
``result.zone_summary``: one illustrative zone at a time, its EVs' home
import on Unmanaged and Smart, plus the optional Timed tariff path when the
run has it (decision 0007, model step 2), against its optional headroom,
then every zone's weekly peak and hours above headroom in one table. Zones
are fleet-wide by
construction (a zone mixes archetypes), so the Group selector is disabled
as it is for the flexibility metrics. Headroom is reported, never enforced.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from axle_studio.model.assumptions import ACTION

from ..components import header
from ..components.chart_table import chart_block
from ..registry import FLEET_GROUP, GROUP_KEY
from ..style import (
    ARCHETYPE_COLOURS,
    CHART_HEIGHTS,
    INK,
    PATH_DISPLAY_ORDER,
    PATH_LABELS,
    PATH_STYLES,
    SEQUENTIAL,
    UNMANAGED_GLOSSARY,
    band_and_line,
    band_fill,
    hover_time_labels,
    london_time_axis,
    present_paths,
)

_BEHAVIOUR_METRICS = ("Plugged in", "Home import", "Battery SoC", "Travel")
_FLEXIBILITY_METRICS = ("Deferrable power by slack", "Turn-up headroom", "Movable energy")
_NETWORK_METRICS = ("Zone import",)
_METRIC_LABELS = _BEHAVIOUR_METRICS + _FLEXIBILITY_METRICS + _NETWORK_METRICS
"""Every option of the one metric control, behaviour first (plan G10)."""


def _metric_option_label(metric: str) -> str:
    """Group prefix shown in the metric control, so eight options read as three groups."""

    if metric in _BEHAVIOUR_METRICS:
        group = "Behaviour"
    elif metric in _FLEXIBILITY_METRICS:
        group = "Flexibility"
    else:
        group = "Network"
    return f"{group}: {metric}"


# Metrics with one line each (contract v2 section 3.5); "Home import" and
# "Travel" plot two named lines and are built separately below.
_SINGLE_METRIC = {
    "Plugged in": ("connected_share", "Fleet plugged in at home", "fraction"),
    "Battery SoC": ("battery_soc_percent", "Fleet battery SoC (capacity-weighted)", "percent"),
}

# Chart audit O7: "realised_grid_kw" (now total_import_kw) includes public
# import in action results, so it must never be shown unlabelled next to the
# home-only figure. Both lines are drawn here, distinguished by dash rather
# than colour, because teal and the "difference" token are already reserved
# meanings (selected Axle path, selected-minus-normal) that neither line is.
_HOME_IMPORT_SERIES = (
    ("home_import_kw", "Home grid import", "solid"),
    ("total_import_kw", "Home + public grid import", "dash"),
)
_TRAVEL_SERIES = (("driving_share", "Driving", "solid"), ("away_share", "Away from home", "dash"))

# The not-recovered flag tolerance, decision 0004 items 5, 13.
_NOTHING_TOLERANCE_KWH = ACTION["not_recovered_tolerance_kwh"].value

_EV_BAND_ALPHA = 0.15
"""Fill opacity for the across-EVs band (item 40's "separate lighter band"),
half of ``style.BAND_ALPHA`` (0.3, the across-weeks band's own opacity), so
the two read as distinct layers rather than one darker stack."""

_PLUGGED_IN_EXTRA_HEIGHT_PX = 84
"""The fleet total plus up to six cohort lines is seven legend entries, which
wrap to about four rows at 390 px. ``style_figure`` now sizes the bottom margin
to those rows (polish plan G2), so this only adds back the three rows beyond
the first, keeping the plot area at the standard time-series height rather
than letting the legend eat it. Three 28 px rows; down from 140 px, which had
to cover the old fixed-fraction legend placement and a separately listed band."""

_FLEET_LINE_WIDTH = 3.0
"""Fleet-total line width on the Plugged in chart, set apart from the cohort lines."""

_COHORT_LINE_ALPHA = 0.8
"""Goal review item 16: six full-opacity lines crowded the plot and competed
with the fleet total's own bold band and median for attention at 390 px.
ARCHETYPE_COLOURS are already muted, so 0.8 (was 0.6 on the saturated
Okabe-Ito set) keeps them readable while still secondary. Fainter lines
(rather than dropping any cohort, which would make an archetype silently
vanish -- chart audit B1's own concern applied to a line chart) read as
secondary context behind the fleet total, while all six stay comparable and
keep their own legend entry."""


def _band_rows(bands: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = bands.loc[bands["path_id"].eq("normal") & bands["metric"].eq(metric)]
    return rows.sort_values("slot_index", kind="stable")


def _band_traces(
    rows: pd.DataFrame,
    *,
    name: str,
    dash: str,
    scale: float,
    colour: str = PATH_STYLES["normal"]["color"],
    width: float = 2.0,
) -> list[go.Scatter]:
    """P10-P90 band plus median line, the standard centre convention (0004 item 27).

    Built with ``style.band_and_line`` so the band and its line are one
    legend entry named ``name`` (polish plan G3); the caption names the
    spread. The band bounds keep their own hover values, so a unified hover
    still reads P10, median and P90 at each half-hour.
    """

    hover = hover_time_labels(rows["interval_start_utc"])
    figure = band_and_line(
        go.Figure(),
        rows["interval_start_utc"],
        rows["p10"].to_numpy(dtype=float) * scale,
        rows["p50"].to_numpy(dtype=float) * scale,
        rows["p90"].to_numpy(dtype=float) * scale,
        name=name,
        colour=colour,
        width=width,
        dash=dash,
        customdata=hover,
        hovertemplate="%{customdata}<br>Median: %{y:,.1f}<extra></extra>",
        band_hovertemplates=(
            "%{customdata}<br>P10: %{y:,.1f}<extra></extra>",
            "%{customdata}<br>P90: %{y:,.1f}<extra></extra>",
        ),
    )
    return list(figure.data)


def _apply_london_axis(figure: go.Figure, x: pd.Series) -> None:
    figure.update_xaxes(tickmode="array", **london_time_axis(x))


def _ev_band_rows(ev_bands: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = ev_bands.loc[ev_bands["path_id"].eq("normal") & ev_bands["metric"].eq(metric)]
    return rows.sort_values("slot_index", kind="stable")


def _ev_band_fill(rows: pd.DataFrame, *, name: str, scale: float) -> list[go.Scatter]:
    """The across-EVs P10-P90 fill only (decision 0004 items 39, 40).

    No median line of its own: the chart's one centre line stays the fleet
    average from ``_band_traces`` (item 27's standard convention), and this
    band is purely added context -- the typical week's 10th-90th percentile
    EV at that half-hour (contract v2 section 3.6b), not a second forecast.
    Drawn with a lighter fill (``_EV_BAND_ALPHA``) so the two bands read as
    distinct layers.

    Battery SoC is the only caller: a fleet's capacity-weighted SoC and one
    EV's SoC are both already 0-100%, so they share one axis cleanly. Home
    import does not call this any more -- a single EV's kW (0 to one
    charger's worth) and the fleet's summed kW (0 to thousands) are
    different physical quantities that happen to share a unit name, which
    made the band invisible on one axis and needed a secondary axis to fix;
    goal review item 16 dropped the band instead, since it was mostly 0 at
    P10/P50 anyway (only a fraction of the fleet charges at once).
    """

    if rows.empty:
        return []
    x = rows["interval_start_utc"]
    p10 = rows["p10"].to_numpy(dtype=float) * scale
    p90 = rows["p90"].to_numpy(dtype=float) * scale
    hover = hover_time_labels(x)
    colour = PATH_STYLES["normal"]["color"]
    legend_name = "P10–P90 across EVs"
    return [
        go.Scatter(
            x=x,
            y=p10,
            mode="lines",
            line={"color": colour, "width": 0},
            name=legend_name,
            legendgroup=f"{name}-ev",
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=x,
            y=p90,
            mode="lines",
            line={"color": colour, "width": 0},
            fill="tonexty",
            fillcolor=band_fill(colour, _EV_BAND_ALPHA),
            name=legend_name,
            legendgroup=f"{name}-ev",
            showlegend=True,
            customdata=hover,
            hovertemplate="%{customdata}<br>P90 across EVs: %{y:,.1f}<extra></extra>",
        ),
    ]


def _sampled_cohorts(cohort_summary: pd.DataFrame) -> list[tuple[str, str]]:
    rows = cohort_summary.loc[cohort_summary["ev_count"] > 0]
    return list(zip(rows["cohort_id"], rows["cohort_label"], strict=True))


def _cohort_line_traces(
    cohort_bands: pd.DataFrame, cohort_summary: pd.DataFrame, *, metric: str, scale: float
) -> list[go.Scatter]:
    """One median (P50) line per sampled cohort (decision 0004 item 43).

    No band per cohort -- six extra bands on top of the fleet's two would
    crowd the chart well past decision 0004 item 26's "nothing clipped or
    squashed"; the fleet total already carries the across-weeks band, and
    this view's own "One archetype across the week" expander (Archetypes)
    gives any single cohort its band on request.
    """

    traces = []
    # One fixed colour per cohort position (style.ARCHETYPE_COLOURS, polish plan
    # G4): six muted hues at equal lightness, none of them teal (smart path).
    for (cohort_id, cohort_label), colour in zip(
        _sampled_cohorts(cohort_summary), ARCHETYPE_COLOURS, strict=False
    ):
        rows = cohort_bands.loc[
            cohort_bands["cohort_id"].eq(cohort_id)
            & cohort_bands["path_id"].eq("normal")
            & cohort_bands["metric"].eq(metric)
        ].sort_values("slot_index", kind="stable")
        if rows.empty:
            continue
        x = rows["interval_start_utc"]
        traces.append(
            go.Scatter(
                x=x,
                y=rows["p50"].to_numpy(dtype=float) * scale,
                mode="lines",
                # Goal review item 16: faint and slightly thinner than the
                # fleet median's 2 px (_band_traces), so six lines read as
                # context, not a competing spaghetti of full-strength colour.
                line={"color": band_fill(colour, _COHORT_LINE_ALPHA), "width": 1.2},
                name=cohort_label,
                customdata=hover_time_labels(x),
                hovertemplate=f"%{{customdata}}<br>{cohort_label}: %{{y:,.0f}}%<extra></extra>",
            )
        )
    return traces


def _single_metric_figure(
    bands: pd.DataFrame, *, metric: str, title: str, unit: str, ev_bands: pd.DataFrame | None = None
) -> go.Figure:
    rows = _band_rows(bands, metric)
    scale = 100.0 if unit == "fraction" else 1.0
    traces: list[go.Scatter] = []
    if ev_bands is not None:
        # Drawn first (bottom layer): the across-EVs band is typically much
        # wider than the across-weeks band (item 39's own evidence), so the
        # thin band stays visible as a distinct stripe on top of it rather
        # than the other way round.
        traces.extend(_ev_band_fill(_ev_band_rows(ev_bands, metric), name=title, scale=scale))
    traces.extend(_band_traces(rows, name=title, dash="solid", scale=scale))
    figure = go.Figure(traces)
    _apply_london_axis(figure, rows["interval_start_utc"])
    figure.update_yaxes(title="%" if unit in ("fraction", "percent") else unit)
    if metric == "battery_soc_percent":
        # Chart audit O3: a stated 0-100% range, not an autoscaled one that
        # exaggerates small swings.
        figure.update_yaxes(range=[0, 100])
    return figure


def _plugged_in_figure(
    bands: pd.DataFrame, cohort_bands: pd.DataFrame, cohort_summary: pd.DataFrame
) -> go.Figure:
    """Fleet total (with its across-weeks band) plus one line per cohort (item 43)."""

    rows = _band_rows(bands, "connected_share")
    # The fleet total is drawn in the main ink at 3 px, above six 1.2 px
    # archetype lines of equal muted lightness: the unmanaged-path grey sits
    # at the same lightness as ARCHETYPE_COLOURS and read as a seventh cohort.
    traces = _band_traces(
        rows,
        name="Fleet plugged in at home",
        dash="solid",
        scale=100.0,
        colour=INK,
        width=_FLEET_LINE_WIDTH,
    )
    traces.extend(
        _cohort_line_traces(cohort_bands, cohort_summary, metric="connected_share", scale=100.0)
    )
    figure = go.Figure(traces)
    _apply_london_axis(figure, rows["interval_start_utc"])
    figure.update_yaxes(title="%", range=[0, 100])
    return figure


def _dual_series_figure(
    bands: pd.DataFrame,
    series: tuple[tuple[str, str, str], ...],
    *,
    scale: float,
    primary_axis_title: str,
) -> go.Figure:
    """Two named lines (each with its own across-weeks band), one axis.

    Used for Home import (home-only vs home + public) and Travel (driving
    vs away). Goal review item 16 removed Home import's across-EVs band
    (mostly 0 at P10/P50, only a fraction of the fleet charges at once) and
    its secondary axis along with it, so this stays single-axis like Travel.
    """

    traces: list[go.Scatter] = []
    x_reference = None
    for metric, name, dash in series:
        rows = _band_rows(bands, metric)
        if x_reference is None:
            x_reference = rows["interval_start_utc"]
        traces.extend(_band_traces(rows, name=name, dash=dash, scale=scale))
    figure = go.Figure(traces)
    _apply_london_axis(figure, x_reference)
    figure.update_layout(yaxis={"title": primary_axis_title})
    return figure


def _display_frame(bands: pd.DataFrame, metrics: list[str], labels: dict[str, str]) -> pd.DataFrame:
    rows = bands.loc[bands["path_id"].eq("normal") & bands["metric"].isin(metrics)].copy()
    rows = rows.sort_values(["metric", "slot_index"], kind="stable")
    rows["metric"] = rows["metric"].map(labels)
    return rows.loc[
        :,
        [
            "metric",
            "interval_start_london",
            "interval_end_utc",
            "world_count",
            "mean",
            "p10",
            "p50",
            "p90",
        ],
    ].rename(
        columns={
            "metric": "Series",
            "interval_start_london": "Interval start (London)",
            "interval_end_utc": "Interval end (UTC)",
            "world_count": "Simulated weeks",
            "mean": "Mean",
            "p10": "P10",
            "p50": "P50",
            "p90": "P90",
        }
    )


def _unserved_travel_caption(result: Any) -> str:
    """One caption fact naming unserved travel, never a chart of zeros (design 4.2).

    Reads ``result.weekly_bands`` (contract v2 section 3.6a), the only frame
    with a correctly world-first weekly total: each world's weekly sum, then
    statistics across worlds, including ``max``. This view never sums
    ``fleet_interval_bands``' per-slot quantiles into a weekly figure -- a
    sum of quantiles is not the quantile of the sum (contract v2 rule 3).

    Decision 0004 item 32: EVs never strand -- a trip that would take SoC
    below 10% (default threshold, item 37) tops up at an always-available
    public charger to 80% first, so a healthy result reads as a check
    ("top-ups prevent it"), never the word "stranded".
    """

    row = result.weekly_bands.loc[
        result.weekly_bands["path_id"].eq("normal")
        & result.weekly_bands["metric"].eq("unserved_travel_kwh")
    ].iloc[0]
    worst = float(row["max"])
    if worst <= _NOTHING_TOLERANCE_KWH:
        return "Unserved travel: 0 kWh, top-ups prevent it."
    median = float(row["p50"])
    return (
        f"Unserved travel: up to {worst:,.1f} kWh in the worst simulated week "
        f"(P50 {median:,.1f} kWh/week)."
    )


# --- Flexibility metrics (decision 0004 item 54, plan C1-C3) -----------------

# Deferrable power is drawn from the *cumulative* rows ("kW that can wait at
# least N hours"), never by stacking bucket quantiles: contract v2 3.6d sums
# buckets per world before quantiles, and a sum of bucket P50s is not the P50
# of the sum. Each cumulative row is nested inside the one before it in every
# week (>= 2 h is a subset of >= 1 h), and a quantile keeps that order, so
# their P50s drawn as overlapping areas never cross and read like a stack
# while every value shown is a world-first model figure.
MUTED_GREY = "#5B6472"
"""Neutral fill for the deferrable total's residual strip (no hue, so it does
not read as another slack level on the blue scale)."""

_DEFERRABLE_LEVELS = (
    # (slack_bucket row, legend name, hover label, fill colour)
    # The total's exposed strip is the rest of the plugged-in, below-target
    # demand: roughly the under-1 h bucket, but a difference of two medians,
    # so it is named loosely and drawn in a neutral grey, not the blue scale.
    ("total", "Rest of plugged-in demand (under 1 h)", "All deferrable", MUTED_GREY),
    ("at_least_1h", "Can wait 1 h+", "Can wait 1 h+", SEQUENTIAL[2][1]),
    ("at_least_2h", "Can wait 2 h+", "Can wait 2 h+", SEQUENTIAL[3][1]),
    ("at_least_4h", "Can wait 4 h+", "Can wait 4 h+", SEQUENTIAL[4][1]),
    ("at_least_8h", "Can wait 8 h+", "Can wait 8 h+", SEQUENTIAL[5][1]),
)
_SLACK_BUCKET_LABELS = {
    "under_1h": "Under 1 h (must run now)",
    "1_to_2h": "1 to 2 h",
    "2_to_4h": "2 to 4 h",
    "4_to_8h": "4 to 8 h",
    "8h_or_more": "8 h or more",
    "at_least_1h": "At least 1 h",
    "at_least_2h": "At least 2 h",
    "at_least_4h": "At least 4 h",
    "at_least_8h": "At least 8 h",
    "total": "Total deferrable",
}
_TOTAL_BAND_ALPHA = 0.2
"""The total's P10-P90 band sits over the shaded areas, so it is fainter than
``style.BAND_ALPHA`` to keep the areas under it readable."""


def _deferrable_rows(result: Any, slack_bucket: str) -> pd.DataFrame:
    frame = result.deferrable_power_bands
    rows = frame.loc[frame["slack_bucket"].eq(slack_bucket)]
    return rows.sort_values("slot_index", kind="stable").reset_index(drop=True)


def _deferrable_figure(result: Any) -> go.Figure:
    """Deferrable power by slack, unmanaged path (contract v2 3.6d, plan C2)."""

    figure = go.Figure()
    total = _deferrable_rows(result, "total")
    x = total["interval_start_utc"]
    hover = hover_time_labels(x)
    # Largest first, so each smaller (more patient) area paints over it.
    for bucket, name, hover_label, colour in _DEFERRABLE_LEVELS:
        rows = _deferrable_rows(result, bucket)
        figure.add_trace(
            go.Scatter(
                x=rows["interval_start_utc"],
                y=rows["p50"],
                mode="lines",
                line={"color": colour, "width": 0},
                fill="tozeroy",
                fillcolor=colour,
                name=name,
                legendgroup=bucket,
                customdata=hover,
                hovertemplate=f"{hover_label}: %{{y:,.0f}} kW<extra></extra>",
            )
        )
    band_and_line(
        figure,
        x,
        total["p10"],
        total["p50"],
        total["p90"],
        name="Total, P10–P90",
        colour=INK,
        width=1.5,
        band_alpha=_TOTAL_BAND_ALPHA,
        customdata=hover,
        hovertemplate="%{customdata}<br>Total median: %{y:,.0f} kW<extra></extra>",
    )
    figure.update_xaxes(tickmode="array", **london_time_axis(x, compact=True))
    figure.update_yaxes(title="kW", rangemode="tozero")
    return figure


def _deferrable_table(result: Any) -> pd.DataFrame:
    rows = result.deferrable_power_bands.copy()
    rows["slack_bucket"] = rows["slack_bucket"].map(_SLACK_BUCKET_LABELS)
    return rows.rename(
        columns={
            "slack_bucket": "Slack",
            "interval_start_london": "Interval start (London)",
            "world_count": "Simulated weeks",
            "p10": "P10 (kW)",
            "p50": "P50 (kW)",
            "p90": "P90 (kW)",
        }
    ).loc[
        :,
        ["Slack", "Interval start (London)", "Simulated weeks", "P10 (kW)", "P50 (kW)", "P90 (kW)"],
    ]


def _time_slack_table(result: Any) -> pd.DataFrame:
    """The retired Time slack chart's rows (plan C1), unmanaged path, both spreads."""

    frame = result.flexibility_bands
    rows = frame.loc[frame["metric"].eq("time_slack_hours") & frame["path_id"].eq("normal")]
    rows = rows.assign(
        spread=rows["spread"].map({"across_evs": "Across EVs", "across_weeks": "Across weeks"})
    )
    return rows.rename(
        columns={
            "spread": "Spread",
            "interval_start_london": "Interval start (London)",
            "world_count": "Simulated weeks",
            "p10": "P10 (h)",
            "p50": "P50 (h)",
            "p90": "P90 (h)",
        }
    ).loc[
        :, ["Spread", "Interval start (London)", "Simulated weeks", "P10 (h)", "P50 (h)", "P90 (h)"]
    ]


def _flexibility_rows(result: Any, *, metric: str, path_id: str) -> pd.DataFrame:
    """``flexibility_bands`` rows (contract v2 3.6c), across-weeks spread."""

    frame = result.flexibility_bands
    mask = (
        frame["metric"].eq(metric)
        & frame["spread"].eq("across_weeks")
        & frame["path_id"].eq(path_id)
    )
    return frame.loc[mask].sort_values("slot_index").reset_index(drop=True)


def _flexibility_figure(rows: pd.DataFrame, *, path_id: str, unit: str) -> go.Figure:
    """One path's P10-P90 band and median, one legend entry (items 27, G3)."""

    style = PATH_STYLES[path_id]
    label = PATH_LABELS[path_id]
    x = rows["interval_start_utc"]
    figure = band_and_line(
        go.Figure(),
        x,
        rows["p10"],
        rows["p50"],
        rows["p90"],
        name=label,
        colour=style["color"],
        width=style["width"],
        dash=style["dash"],
        customdata=hover_time_labels(x),
        hovertemplate=f"%{{customdata}}<br>{label} median: %{{y:,.0f}} {unit}<extra></extra>",
    )
    figure.update_xaxes(tickmode="array", **london_time_axis(x, compact=True))
    figure.update_yaxes(title=unit, rangemode="tozero")
    return figure


def _flexibility_display(rows: pd.DataFrame, unit: str) -> pd.DataFrame:
    return (
        rows.assign(path_id=rows["path_id"].map(PATH_LABELS))
        .rename(
            columns={
                "path_id": "Path",
                "interval_start_london": "Interval start (London)",
                "world_count": "Simulated weeks",
                "p10": f"P10 ({unit})",
                "p50": f"P50 ({unit})",
                "p90": f"P90 ({unit})",
            }
        )
        .loc[
            :,
            [
                "Path",
                "Interval start (London)",
                "Simulated weeks",
                f"P10 ({unit})",
                f"P50 ({unit})",
                f"P90 ({unit})",
            ],
        ]
    )


def _render_flexibility(st: Any, result: Any, metric_label: str) -> None:
    """Render one of the three fleet-wide flexibility metrics."""

    if result.flexibility_bands is None:
        # Contract v2 3.6c: only a direct model call without a departure
        # margin lacks these; the app always passes one.
        st.info("This run has no flexibility figures: it ran without a departure margin.")
        return
    height = CHART_HEIGHTS["time_series"]
    if metric_label == "Deferrable power by slack":
        world_count = int(result.deferrable_power_bands["world_count"].iat[0])
        chart_block(
            st,
            _deferrable_figure(result),
            title="Fleet deferrable power by slack (kW)",
            caption=(
                "Unmanaged path, fleet. Shades: kW that can wait at least N hours and still reach "
                f"target (P50). Line: total, P10–P90 across {world_count} weeks."
            ),
            frame=_deferrable_table(result),
            definition=(
                "Slack is how long charging could wait: hours until the EV is due to leave, "
                "minus the whole half-hours it still needs at full power. Each shade is the "
                "home charger power of plugged-in, below-target EVs that could wait at least "
                "that long. Each 'at least N hours' level and the total are summed per "
                "simulated week before quantiles, so the levels' quantiles do not add up to "
                "the total's."
            ),
            height=height,
            key="fleet-week-deferrable",
        )
        st.caption(
            "Slack = hours to the expected departure minus hours still needed at full power: "
            "how long charging could wait."
        )
        with st.expander("Time slack data (retired chart)"):
            st.dataframe(_time_slack_table(result), width="stretch", hide_index=True)
            st.caption(
                "Per EV that can charge: hours to its expected departure minus hours needed at "
                "full power. Retired as a chart; deferrable power by slack replaces it."
            )
    elif metric_label == "Turn-up headroom":
        if result.model != "action":
            st.info("Turn-up headroom is shown for the smart path. Run smart charging to see it.")
            return
        rows = _flexibility_rows(result, metric="turn_up_headroom_kw", path_id="selected")
        chart_block(
            st,
            _flexibility_figure(rows, path_id="selected", unit="kW"),
            title="Fleet turn-up headroom, smart path (kW)",
            caption=(
                "Smart path, fleet. Charger power left unused while EVs wait for cheap half-hours, "
                f"not extra flexibility. P10–P90 across {int(rows['world_count'].iat[0])} weeks."
            ),
            frame=_flexibility_display(rows, "kW"),
            definition=(
                "The most the plugged-in, below-target EVs could draw together, minus what the "
                "smart path actually imports at home that half-hour, clipped at 0. It is charger "
                "power the plan is not using yet, not extra flexibility."
            ),
            height=height,
            key="fleet-week-headroom",
        )
    else:
        rows = _flexibility_rows(result, metric="movable_energy_kwh", path_id="normal")
        chart_block(
            st,
            _flexibility_figure(rows, path_id="normal", unit="kWh"),
            title="Fleet movable energy (kWh)",
            caption=(
                "Unmanaged path, fleet. Grid kWh still needed to reach target before the expected "
                f"departure. P10–P90 across {int(rows['world_count'].iat[0])} weeks."
            ),
            frame=_flexibility_display(rows, "kWh"),
            definition=(
                "For each plugged-in EV below target: the energy from its battery level up to "
                "its preferred target, divided by home charging efficiency (so it is grid kWh, "
                "not battery kWh), summed over those EVs."
            ),
            height=height,
            key="fleet-week-movable",
        )
    # The one glossary line for the counterfactual path (polish plan C3).
    st.caption(UNMANAGED_GLOSSARY)


# --- Network: zones (decision 0004 item 55; trading contract v1 §3, §5.7, §5.8) --


def _zone_rows(result: Any, zone_id: str, path_id: str) -> pd.DataFrame:
    frame = result.zone_import_bands
    rows = frame.loc[frame["zone_id"].eq(zone_id) & frame["path_id"].eq(path_id)]
    return rows.sort_values("slot_index", kind="stable")


def _zone_paths(result: Any) -> list[str]:
    """``zone_import_bands``' own paths, in on-screen order (Unmanaged, Timed tariff,
    Smart, decision 0007): two on a run without the optional timed path, three
    on one with it, with no branching here."""

    return present_paths(result.zone_import_bands["path_id"])


def _zone_figure(result: Any, zone_id: str) -> go.Figure:
    """One zone's home import per path, P10-P90 across weeks, with its headroom if set."""

    figure = go.Figure()
    x = None
    headroom = float("nan")
    for path in _zone_paths(result):
        rows = _zone_rows(result, zone_id, path)
        x = rows["interval_start_utc"]
        headroom = float(rows["headroom_kw"].iat[0])
        style = PATH_STYLES[path]
        label = PATH_LABELS[path]
        band_and_line(
            figure,
            x,
            rows["p10"],
            rows["p50"],
            rows["p90"],
            name=label,
            colour=style["color"],
            width=style["width"],
            dash=style["dash"],
            # Unmanaged and Smart stay linear; only the optional "timed" path
            # draws as a step (decision 0007: PATH_STYLES["timed"] holds off
            # or allows charging for a whole half-hour at a time, so the rise
            # at the start time is a vertical jump, not a slope).
            line_shape=style.get("line_shape", "linear"),
            legendgroup=path,
            customdata=hover_time_labels(x),
            hovertemplate=f"%{{customdata}}<br>{label} median: %{{y:,.0f}} kW<extra></extra>",
        )
    if headroom == headroom:
        # A reference rule in the main ink with its own label, so it reads as
        # a limit to compare against, not a third path; the model reports
        # against it and never enforces it (contract §3).
        figure.add_hline(
            y=headroom,
            line_dash="dash",
            line_color=INK,
            line_width=1,
            annotation_text=f"Headroom {headroom:,.0f} kW",
            annotation_position="top left",
            annotation_font_color=INK,
        )
    figure.update_xaxes(tickmode="array", **london_time_axis(x))
    figure.update_yaxes(title="kW", rangemode="tozero")
    return figure


def _zone_band_table(result: Any, zone_id: str) -> pd.DataFrame:
    frame = result.zone_import_bands
    # Path block order on screen (Unmanaged, Timed tariff, Smart, decision
    # 0007), not the alphabetical default a plain sort on "path_id" would give.
    rows = frame.loc[frame["zone_id"].eq(zone_id)].sort_values(
        ["path_id", "slot_index"], key=_path_then_zone
    )
    return pd.DataFrame(
        {
            "Path": rows["path_id"].map(PATH_LABELS),
            "Interval start (London)": rows["interval_start_london"],
            "Simulated weeks": rows["world_count"],
            "P10 (kW)": rows["p10"].round(1),
            "P50 (kW)": rows["p50"].round(1),
            "P90 (kW)": rows["p90"].round(1),
            "Share of weeks above headroom": rows["above_headroom_world_share"],
        }
    )


def _range(low: float, high: float, decimals: int) -> str:
    if low != low:
        return "No headroom set"
    return f"{low:,.{decimals}f} to {high:,.{decimals}f}"


def zone_summary_table(zone_summary: pd.DataFrame) -> pd.DataFrame:
    """Every zone and path: EVs, share, headroom, weekly peak and hours above headroom.

    Straight from ``zone_summary`` (contract §5.8): each simulated week's own
    peak and its hours above headroom are taken inside the week first, then
    P10/P50/P90 across weeks, so nothing here is recomputed.
    """

    rows = zone_summary.sort_values(["zone_id", "path_id"], key=_path_then_zone)
    return pd.DataFrame(
        {
            "Zone": rows["zone_label"],
            "Path": rows["path_id"].map(PATH_LABELS),
            "EVs": rows["ev_count"],
            "Share of fleet": [f"{share:.0%}" for share in rows["share"]],
            "Headroom (kW)": [
                "None set" if value != value else f"{value:,.0f}" for value in rows["headroom_kw"]
            ],
            "Weekly peak P50 (kW)": [f"{value:,.0f}" for value in rows["peak_kw_p50"]],
            "Weekly peak P10–P90 (kW)": [
                _range(low, high, 0)
                for low, high in zip(rows["peak_kw_p10"], rows["peak_kw_p90"], strict=True)
            ],
            "Hours above headroom P50": [
                "No headroom set" if value != value else f"{value:g}"
                for value in rows["hours_above_headroom_p50"]
            ],
            "Hours above headroom P10–P90": [
                _range(low, high, 1)
                for low, high in zip(
                    rows["hours_above_headroom_p10"], rows["hours_above_headroom_p90"], strict=True
                )
            ],
        }
    )


_PATH_DISPLAY_RANK = {path: rank for rank, path in enumerate(PATH_DISPLAY_ORDER)}


def _path_then_zone(column: pd.Series) -> pd.Series:
    """Sort key: zones in id order, paths in on-screen order (Unmanaged, Timed
    tariff, Smart, decision 0007) within each zone -- not the alphabetical order
    a plain sort on ``path_id`` would give once the optional third path is
    present."""

    if column.name == "path_id":
        return column.map(_PATH_DISPLAY_RANK)
    return column


def _render_zones(st: Any, result: Any) -> None:
    """Zone import: one illustrative zone's import against its headroom, then all zones."""

    if result.zone_import_bands is None or result.zone_summary is None:
        st.info("This result has no zone figures.")
        return
    summary = result.zone_summary.drop_duplicates("zone_id")
    labels = dict(zip(summary["zone_id"], summary["zone_label"], strict=True))
    zone_id = st.segmented_control(
        "Zone",
        options=list(labels),
        format_func=labels.__getitem__,
        default=next(iter(labels)),
        required=True,
        key="fleet-week-zone",
    )
    zone_id = zone_id if zone_id in labels else next(iter(labels))
    zone = summary.set_index("zone_id").loc[zone_id]
    rows = _zone_rows(result, zone_id, "normal")
    weeks = int(rows["world_count"].iat[0])
    headroom = float(zone["headroom_kw"])
    headroom_note = (
        f"Dashed line: {headroom:,.0f} kW headroom, reported only."
        if headroom == headroom
        else "No headroom set for this zone."
    )
    chart_block(
        st,
        _zone_figure(result, zone_id),
        title=f"{labels[zone_id]} home import (kW)",
        caption=(
            f"{int(zone['ev_count']):,} EVs, {zone['share']:.0%} of the fleet (illustrative). "
            f"Median, P10–P90 across {weeks} weeks. {headroom_note}"
        ),
        frame=_zone_band_table(result, zone_id),
        definition=(
            "Home import of the EVs placed in this illustrative zone (a reporting split, not a "
            "place), per simulated week, then P10, P50 and P90 across weeks. The four zones' "
            "means add up to the fleet mean; their percentiles do not. Headroom is an optional "
            "reported limit: the model never cuts charging back to it."
        ),
        height=CHART_HEIGHTS["time_series"],
        key=f"fleet-week-zone-{zone_id}",
    )
    st.markdown("**Every zone: weekly peak and hours above headroom**")
    st.dataframe(zone_summary_table(result.zone_summary), width="stretch", hide_index=True)
    st.caption(
        "Each week's own peak, then P10–P90 across weeks. Zones and headroom are illustrative; "
        "headroom is reported, never enforced."
    )
    st.caption(UNMANAGED_GLOSSARY)


# --- Page -------------------------------------------------------------------


def _group_options(cohort_summary: pd.DataFrame) -> dict[str, str]:
    """Group selector values -> labels: the fleet, then each sampled archetype in source order.

    An archetype with no sampled EV has no rows in ``cohort_interval_bands``
    (contract v2 3.7), so it is left out rather than offered as an empty chart.
    """

    return {FLEET_GROUP: "Fleet", **dict(_sampled_cohorts(cohort_summary))}


def render_fleet_week(st: Any, result: Any) -> None:
    """Render Drivers ▸ Fleet week: one metric at a time for the fleet or one archetype."""

    groups = _group_options(result.cohort_summary)
    # In the page header's controls slot (polish plan G8).
    metric_column, group_column = header.controls(st).columns(2)
    with metric_column:
        metric_label = st.selectbox(
            "Metric",
            options=list(_METRIC_LABELS),
            format_func=_metric_option_label,
            key="fleet-week-metric",
            label_visibility="collapsed",
        )
    metric_label = metric_label if metric_label in _METRIC_LABELS else _METRIC_LABELS[0]
    fleet_only = metric_label in _FLEXIBILITY_METRICS + _NETWORK_METRICS
    with group_column:
        if fleet_only:
            # The flexibility frames are fleet-wide, so a disabled, display-
            # only box reads "Fleet" (the layout stays put). It has its own
            # key, so the shared Group choice is not overwritten and returns
            # when a behaviour metric is picked again.
            st.selectbox(
                "Group",
                options=["Fleet"],
                key="fleet-week-group-fleet-only",
                label_visibility="collapsed",
                disabled=True,
                help="Flexibility and zone figures are fleet-wide only.",
            )
            group = FLEET_GROUP
        else:
            # Plan E2: one Group selector, one shared key, only on lenses
            # whose frame has a group key.
            group = st.selectbox(
                "Group",
                options=list(groups),
                format_func=groups.__getitem__,
                key=GROUP_KEY,
                label_visibility="collapsed",
                persist_state="session",
            )
    if group not in groups:
        group = FLEET_GROUP

    if metric_label in _NETWORK_METRICS:
        _render_zones(st, result)
        return
    if fleet_only:
        _render_flexibility(st, result, metric_label)
        return

    if group == FLEET_GROUP:
        bands = result.fleet_interval_bands
    else:
        cohort_bands = result.cohort_interval_bands
        bands = cohort_bands.loc[cohort_bands["cohort_id"].eq(group)]
    ev_bands = result.fleet_interval_ev_bands
    world_count = int(bands["world_count"].iloc[0]) if not bands.empty else 0
    group_label = groups[group]
    caption_base = f"Unmanaged, {group_label}. Median, P10–P90 across {world_count} weeks."
    if group == FLEET_GROUP:
        caption_base += f" {_unserved_travel_caption(result)}"
    height = CHART_HEIGHTS["time_series"]

    if metric_label == "Home import":
        figure = _dual_series_figure(bands, _HOME_IMPORT_SERIES, scale=1.0, primary_axis_title="kW")
        title = f"{group_label} grid import (kW)"
        definition = (
            "Home-only and home + public grid import as average power over each half-hour, "
            "median and P10–P90 across simulated weeks."
        )
        frame = _display_frame(
            bands,
            [metric for metric, _, _ in _HOME_IMPORT_SERIES],
            dict((m, n) for m, n, _ in _HOME_IMPORT_SERIES),
        )
        caption = caption_base
    elif metric_label == "Travel":
        figure = _dual_series_figure(
            bands, _TRAVEL_SERIES, scale=100.0, primary_axis_title="% of EV-time"
        )
        title = f"{group_label} travel shares (%)"
        definition = (
            "Share of EV-time driving and away from home (driving, parked away, or public "
            "charging), median and P10–P90 across simulated weeks."
        )
        frame = _display_frame(
            bands,
            [metric for metric, _, _ in _TRAVEL_SERIES],
            dict((m, n) for m, n, _ in _TRAVEL_SERIES),
        )
        caption = caption_base
    elif metric_label == "Plugged in" and group == FLEET_GROUP:
        cohort_summary = result.cohort_summary
        figure = _plugged_in_figure(bands, result.cohort_interval_bands, cohort_summary)
        title = "Plugged in (%)"
        definition = (
            "Fleet share plugged in at home, median and P10–P90 across simulated weeks, plus "
            "each sampled archetype's own median share plugged in (no band per archetype: "
            "six more bands would hide the fleet's)."
        )
        frame = _display_frame(bands, ["connected_share"], {"connected_share": "Fleet"})
        caption = f"{caption_base} Thin lines: one per archetype."
        height += _PLUGGED_IN_EXTRA_HEIGHT_PX
    else:
        metric, metric_title, unit = _SINGLE_METRIC[metric_label]
        if group != FLEET_GROUP:
            metric_title = f"{group_label} {metric_title.split(' ', 1)[1]}"
        # The across-EVs band exists for the fleet only (contract v2 3.6b).
        metric_ev_bands = (
            ev_bands if metric == "battery_soc_percent" and group == FLEET_GROUP else None
        )
        figure = _single_metric_figure(
            bands, metric=metric, title=metric_title, unit=unit, ev_bands=metric_ev_bands
        )
        if unit == "fraction":
            figure.update_yaxes(range=[0, 100])
        unit_label = "%" if unit in ("fraction", "percent") else unit
        title = f"{metric_title} ({unit_label})"
        frame = _display_frame(bands, [metric], {metric: metric_title})
        if metric_ev_bands is not None:
            definition = (
                f"{metric_title}: median and P10–P90 across simulated weeks (the thin band), "
                "plus P10–P90 across EVs (the lighter band: how much EVs differ from each "
                "other in a typical week). SoC is state of charge, how full the battery is."
            )
            caption = f"{caption_base} Lighter band: P10–P90 across EVs."
        else:
            definition = f"{metric_title}, median and P10–P90 across simulated weeks."
            caption = caption_base

    chart_block(
        st,
        figure,
        title=title,
        caption=caption,
        frame=frame,
        definition=definition,
        height=height,
        key=f"fleet-week-chart-{metric_label}",
    )
