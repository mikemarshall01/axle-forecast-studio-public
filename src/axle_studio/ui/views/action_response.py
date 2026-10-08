"""Smart charging lens 2, Response: what changed physically (Q7).

Reads ``fleet_interval_bands`` (both paths), ``difference_bands``,
``difference_weekly_bands``, ``smart_charging_summary`` (the share-moved
tile) and ``weekly_peak_summary`` (each week's own peak, the herding
measure of decision 0004 items 48-49; docs/contracts/results-v2.md section 9).
Smart charging applies to every EV (decision 0004 item 38), so there is no
"action not selected" state any more; the only guard left is the shared
no-action-model message (design 4.3), handled by ``action_summary is None``
below and by ``pages.py`` before this view is ever called.

The metric selector offers Home import, Battery SoC and Events. Plugged-in
share is dropped (decision 0004 item 43): it is already shown, across
archetypes, on Drivers ▸ Fleet week, and is not a physical response to the
action the way charging power and battery stock are.

Events (decision 0004 items 55, 56, 58; trading contract v1 §5.6, §6) reads
``events`` and ``event_response_bands``: for one chosen scripted event, the
import of the fleet or zone it covers around its window, unmanaged against
smart; the paired smart-minus-unmanaged difference, whose changes just
outside the window are the rebound; and for a scripted price shock, how far
that shock alone moved the price. Every value is a band already summarised
across simulated weeks by the model; the view only looks values up.
Delivery against the settlement baseline and its illustrative payment wait
for the trading baseline, and the view says so rather than estimating them.

The internal path id stays ``"selected"`` (contract v2, ``path_id``); every
user-facing legend, title and caption says "smart" instead (``style.PATH_LABELS``).

The summary sentence and evidence-kind wording come from ``action_decision``,
which owns the one shared text-mapping helper for all three lenses, rather
than three different phrasings of the same reason code.

On the Home import metric, when this run's intraday dispatch switch was on
(``result.dispatch_bands`` not ``None``), a further "Day-ahead plan vs
dispatched" block reads ``dispatch_bands``, the representative world's
``dispatch_world_slot`` and ``dispatch_split`` (intraday-dispatch-v1 §7,
§10): the fleet's day-ahead-plan reference path beside the actual
dispatched path, their difference, how many free EVs re-planned each
half-hour, and one simulated week's own response to the market. The event
view's ``day_ahead_plan`` series (§7.4) rides the same switch. Every figure
here is a world-first statistic the model already computed; the view
formats and plots it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from ..components import header
from ..components.chart_table import chart_block
from ..components.kpi import kpi, kpi_columns
from ..run_controller import event_label
from ..style import (
    ARCHETYPE_COLOURS,
    CHART_HEIGHTS,
    MUTED_INK,
    PATH_DISPLAY_ORDER,
    PATH_LABELS,
    PATH_STYLES,
    SERIES_COLOURS,
    UNAVAILABLE,
    band_and_line,
    format_quantity,
    hover_time_labels,
    london_time_axis,
    percent,
)
from .action_decision import (
    _evidence_text,
    _summary_sentence,
    add_shock_marks,
    scripted_shock_windows,
)

# (metric column, display unit). Design number formatting: fleet power kW and
# percent both round to 0 dp, so one decimals value serves both.
_METRICS: dict[str, tuple[str, str]] = {
    "Home import": ("home_import_kw", "kW"),
    "Battery SoC": ("battery_soc_percent", "%"),
}
EVENTS_METRIC = "Events"
"""The third Metric option: the scripted-event view, not a fleet metric."""
_METRIC_DEFINITIONS = {
    "home_import_kw": (
        "Average power drawn at the home charger, half-hour by half-hour, both paths."
    ),
    "battery_soc_percent": (
        "Fleet battery state of charge (SoC, how full the batteries are) at the end of each "
        "half-hour, weighted by battery size so a large battery counts for more."
    ),
}
_WEEKLY_ROW_LABELS = {
    "home_import_kwh": "Weekly home import",
    "public_import_kwh": "Weekly public import",
    "closing_battery_kwh": "Closing battery energy",
}
"""Rows of the weekly-change table built from ``difference_weekly_bands`` (contract 4.2).

A table, not three more KPI tiles: the lens keeps four tiles (polish plan
G9), and these three are secondary checks on the headline row.

``unserved_travel_kwh`` is not a row here: decision 0004 item 32 (EVs never
strand; a public top-up serves the trip first) makes it zero by construction,
so a row would always read the same illustrative-looking zero. It is
reported as a caption check instead (``_unserved_travel_caption``), which
still shows the real number if a future model version ever makes it nonzero.
"""


def _signed(value: float, unit: str) -> str:
    """A change at 0 dp with an explicit sign: "+12 kW", "−3 kWh" (U+2212, G6).

    ``format_quantity`` rounds before choosing the sign, so a value that
    only rounds to zero (e.g. -0.3 kW) prints "0 kW", never "−0 kW"; this
    adds the "+" a change needs. Missing values stay ``UNAVAILABLE``.
    """

    text = format_quantity(value, unit, decimals=0)
    return f"+{text}" if text != UNAVAILABLE and round(value) > 0 else text


def _metric_control(st: Any) -> str:
    """The Metric control, drawn in the page header's controls slot (polish plan G8)."""

    names = [*_METRICS, EVENTS_METRIC]
    chosen = header.controls(st).segmented_control(
        "Metric",
        options=names,
        default=names[0],
        required=True,
        key="action-response-metric",
        label_visibility="collapsed",
    )
    return chosen if chosen in names else names[0]


def _weekly_change_table(weekly_bands: pd.DataFrame) -> pd.DataFrame:
    """Weekly smart − unmanaged changes as a small table: P50 and P10–P90 (contract 4.2)."""

    rows = []
    for metric, label in _WEEKLY_ROW_LABELS.items():
        row = weekly_bands.loc[weekly_bands["metric"].eq(metric)]
        if row.empty:
            rows.append({"Change, smart − unmanaged": label, "P50": UNAVAILABLE, "P10–P90": ""})
            continue
        unit = row["unit"].iat[0]
        p50, p10, p90 = (float(row[stat].iat[0]) for stat in ("p50", "p10", "p90"))
        rows.append(
            {
                "Change, smart − unmanaged": label,
                "P50": _signed(p50, unit),
                "P10–P90": f"{_signed(p10, unit)} to {_signed(p90, unit)}",
            }
        )
    return pd.DataFrame(rows)


def _unserved_travel_caption(weekly_bands: pd.DataFrame) -> str:
    """A check, not a KPI tile (decision 0004 item 32): stated either way."""

    row = weekly_bands.loc[weekly_bands["metric"].eq("unserved_travel_kwh")]
    if row.empty:
        return "Unserved travel difference: unavailable."
    values = row[["mean", "p10", "p50", "p90"]].to_numpy(dtype=float).ravel()
    if (abs(values) < 1e-6).all():
        return (
            "Unserved travel: 0 kWh difference in every simulated week, because a public "
            "top-up serves the trip first."
        )
    unit = row["unit"].iat[0]
    signed = _signed(row["p50"].iat[0], unit)
    return f"Unserved travel difference, P50 across simulated weeks: {signed}."


def _headline_tile_row(st: Any, result: Any) -> None:
    """Share moved, each week's own smart peak, and the largest median cut/rise.

    Every number is already a world-first statistic in the result, so these
    tiles only look values up and format them (AGENTS.md: never re-derive a
    percentile in the UI).

    - Share moved: ``smart_charging_summary`` (contract 4.5).
    - Each week's own smart peak: ``weekly_peak_summary`` (contract 4.5a),
      the herding measure of decision 0004 items 48-49.  The highest
      half-hour is found inside each simulated week first, because which
      half-hour is cheapest moves from week to week and a peak of the
      median curve would smear it out.
    - Largest median cut/rise in a half-hour: the slot with the smallest and
      largest P50 in ``difference_bands`` (contract 4.1).  That is the
      typical per-half-hour difference, not a weekly peak, so the labels say
      "median" and "in a half-hour" (goal review N3).
    """

    diff = result.difference_bands.loc[result.difference_bands["metric"].eq("home_import_kw")]
    if diff.empty:
        return
    share = result.smart_charging_summary.set_index("metric").loc["moved_home_import_share"]
    cut = diff.loc[diff["p50"].idxmin()]
    rise = diff.loc[diff["p50"].idxmax()]

    columns = kpi_columns(st, 4)
    kpi(
        columns[0],
        "Share of home energy moved",
        percent(100 * share["p50"]),
        context="P50 across simulated weeks",
        help=(
            f"P10–P90 across {int(share['world_count'])} simulated weeks: "
            f"{percent(100 * share['p10'])} to {percent(100 * share['p90'])}."
        ),
    )
    _weekly_peak_tile(columns[1], result.weekly_peak_summary)
    _extreme_tile(columns[2], cut, "Largest half-hour cut")
    _increase_tile(columns[3], rise)


def _peak_rows(
    weekly_peak_summary: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series | None]:
    """Each path's own row of ``weekly_peak_summary`` (contract 4.5a): normal, selected, and
    the optional timed row (decision 0007, model step 2) when the run's frame has it, else
    ``None``."""

    rows = weekly_peak_summary.set_index("path_id")
    timed = rows.loc["timed"] if "timed" in rows.index else None
    return rows.loc["normal"], rows.loc["selected"], timed


def _ratio(value: float) -> str:
    return UNAVAILABLE if value != value else f"×{value:.1f}"


def _weekly_peak_tile(column: Any, weekly_peak_summary: pd.DataFrame) -> None:
    """Each simulated week's own highest smart half-hour, P50 across weeks (contract 4.5a).

    With the optional timed path present (decision 0007, model step 2), its own weekly peak
    joins the context and tooltip beside smart's own figure, never instead of it: the
    hard-barred rule produces its own pile-up the moment charging is allowed again, which is
    worth a figure here as well as in the herding caption below.
    """

    normal, smart, timed = _peak_rows(weekly_peak_summary)
    weeks = int(smart["world_count"])
    when = hover_time_labels(pd.Series([smart["modal_peak_interval_start_utc"]]))[0]
    # A neutral note, not a change: a context line, no arrow or colour.
    context = f"{_ratio(smart['ratio_to_normal_p50'])} of unmanaged peak"
    help_text = (
        "Each simulated week's highest half-hour of smart fleet home import, "
        f"P50 across {weeks} weeks; P10–P90 "
        f"{format_quantity(smart['p10'], 'kW', decimals=0)} to "
        f"{format_quantity(smart['p90'], 'kW', decimals=0)}. The unmanaged path's own weekly "
        f"peak is P50 {format_quantity(normal['p50'], 'kW', decimals=0)}. The ratio is "
        "taken week by week: P10–P90 "
        f"{_ratio(smart['ratio_to_normal_p10'])} to {_ratio(smart['ratio_to_normal_p90'])}. "
        f"Most often at {when} ({int(smart['modal_peak_week_count'])} of {weeks} weeks)."
    )
    if timed is not None:
        context += f" · timed tariff {_ratio(timed['ratio_to_normal_p50'])}"
        timed_when = hover_time_labels(pd.Series([timed["modal_peak_interval_start_utc"]]))[0]
        help_text += (
            f" {PATH_LABELS['timed']}'s own weekly peak is P50 "
            f"{format_quantity(timed['p50'], 'kW', decimals=0)}, "
            f"{_ratio(timed['ratio_to_normal_p50'])} of the unmanaged peak (P10–P90 "
            f"{_ratio(timed['ratio_to_normal_p10'])} to {_ratio(timed['ratio_to_normal_p90'])}), "
            f"most often at {timed_when}."
        )
    kpi(
        column,
        "Weekly smart peak",
        format_quantity(smart["p50"], "", decimals=0),
        unit="kW",
        context=context,
        help=help_text,
    )


def _extreme_tile(column: Any, row: pd.Series, label: str) -> None:
    """One median cut/rise tile: the slot's P50 difference, P10-P90 and London time."""

    when = hover_time_labels(pd.Series([row["interval_start_utc"]]))[0]
    kpi(
        column,
        label,
        _signed(row["p50"], ""),
        unit="kW",
        context="Median, smart − unmanaged",
        help=(
            "The half-hour whose median smart − unmanaged fleet home import across simulated "
            f"weeks is most extreme, at {when}. P10–P90 across {int(row['world_count'])} "
            f"weeks in that half-hour: {_signed(row['p10'], 'kW')} to "
            f"{_signed(row['p90'], 'kW')}. A typical difference, not a weekly peak."
        ),
    )


def _increase_tile(column: Any, row: pd.Series) -> None:
    """The median-rise tile, honest when smart never draws more than normal.

    ``row`` is the slot with the largest smart-minus-normal P50. When every
    slot's P50 is at or below zero, that largest value is really the
    smallest cut, not an increase anywhere; showing it under an "increase"
    label with a negative number would read as an increase that never
    happened, so the tile relabels itself instead of picking a misleading
    reading off a signed number.
    """

    if row["p50"] > 0:
        _extreme_tile(column, row, "Largest half-hour rise")
        return
    kpi(
        column,
        "No half-hour rise",
        "0",
        unit="kW",
        context="Median, smart − unmanaged",
        help=(
            "In every half-hour, the median smart − unmanaged fleet home import across simulated "
            "weeks is zero or below."
        ),
    )


def _herding_caption(weekly_peak_summary: pd.DataFrame) -> tuple[str, str, str | None]:
    """The herding finding for the home-import charts: ``(caption, tooltip, timed_caption)``.

    Decision 0004 items 45, 48-49. Measured as each simulated week's own
    peak (``weekly_peak_summary``), not the peak of the median curve: the
    cheapest half-hour moves between weeks with the day-ahead price, so the
    median curve understates the pile-up. No fixed clock time is named, for
    the same reason (lead direction, 28 September 2026). The mechanism is
    the finding: every smart EV plans from the same forecast and nothing in
    the model limits site or network capacity. The caption keeps to one
    short line (polish plan G7); the figures and ranges go in the tooltip.

    With the optional timed path present (decision 0007, model step 2),
    ``timed_caption`` says plainly how its own peak compares with the
    unmanaged peak: a separate, plain caption rather than a second
    "Finding:" (copy rule: at most one per lens), because the mechanism
    differs from smart's herding story -- every timed EV is released by the
    same clock time, not drawn to the same forecast price, so the midnight
    spike is its own point, not evidence of herding on a forecast. ``None``
    when the run's frame carries no timed row.
    """

    normal, smart, timed = _peak_rows(weekly_peak_summary)
    caption = (
        "Finding: smart charging herds the fleet onto the same cheap half-hours; each week's "
        f"own peak is {_ratio(smart['ratio_to_normal_p50'])} the unmanaged peak (P50)."
    )
    tooltip = (
        "Every smart EV follows the same price forecast and there is no site or network "
        "limit, so they all pile onto the same cheap half-hours. Each simulated week's own "
        f"peak home import is P50 {format_quantity(smart['p50'], 'kW', decimals=0)} on the "
        f"smart path against {format_quantity(normal['p50'], 'kW', decimals=0)} unmanaged. "
        f"The ratio is taken week by week, P10–P90 {_ratio(smart['ratio_to_normal_p10'])} to "
        f"{_ratio(smart['ratio_to_normal_p90'])}. It is each week's own peak, not the peak of "
        "the median curve: the cheapest half-hour moves between weeks, so the median curve "
        "would smear the pile-up out."
    )
    timed_caption = None
    if timed is not None:
        timed_caption = (
            f"{PATH_LABELS['timed']}'s own peak, right when charging is allowed again, is "
            f"{_ratio(timed['ratio_to_normal_p50'])} the unmanaged peak (P50)."
        )
    return caption, tooltip, timed_caption


_STUDY_END_NOTE = (
    "The study runs noon to noon, so every night, the first and last included, is a full "
    "smart-charging night."
)
"""Kept in the difference chart's definition; its caption says it in five words."""


def _no_negative_zero(values: pd.Series, decimals: int) -> pd.Series:
    """Round to a hovertemplate's own precision, clearing a lone "-" off zero.

    d3-format (what Plotly's ``%{y:,.0f}``-style hovertemplate uses) prints
    "-0" for a raw float in (-0.5 * 10**-decimals, 0), even though the same
    value displays as 0 everywhere else. A trace with a hovertemplate must
    carry values already rounded to that precision, with the resulting
    negative zero cleared by adding 0.0 (IEEE 754: -0.0 + 0.0 == 0.0), so
    Plotly never sees the bare negative float.
    """

    return values.round(decimals) + 0.0


def _paths_figure(rows: pd.DataFrame, metric: str, unit: str) -> go.Figure:
    figure = go.Figure()
    suffix = unit if unit == "%" else f" {unit}"  # "82%", "120 kW" (polish plan G6)
    # Every path present in ``rows``, in the on-screen display order
    # (Unmanaged, Timed tariff, Smart, decision 0007): the fleet frame
    # carries "timed" only when the setting is on, so this draws two lines
    # off and three on, with no further branching here.
    present = set(rows["path_id"])
    for path in (path for path in PATH_DISPLAY_ORDER if path in present):
        path_rows = rows.loc[rows["path_id"].eq(path)].sort_values("slot_index")
        style = PATH_STYLES[path]
        label = PATH_LABELS[path]
        x = path_rows["interval_start_utc"]
        p10 = path_rows["p10"]
        p90 = path_rows["p90"]
        # The P50 line has a hovertemplate; the band does not (hoverinfo
        # "skip"), so only P50 needs the negative-zero-safe rounding.
        p50 = _no_negative_zero(path_rows["p50"], decimals=0)
        # Plotly's default hover header shows the raw UTC x-value; the axis
        # itself is UTC too (clock-change safe, see london_time_axis), so the
        # header would read a UTC time with nothing marking it as such
        # (decision 0004 item 27: time axes show London local time). Putting
        # the London-led, UTC-noted string in customdata keeps the axis UTC
        # but makes each point's own hover read London first (goal review
        # action 20).
        band_and_line(
            figure,
            x,
            p10,
            p50,
            p90,
            name=label,
            colour=style["color"],
            width=style["width"],
            dash=style["dash"],
            # Unmanaged and Smart stay the default linear interpolation;
            # only the optional "timed" path draws as a step (decision
            # 0007: see PATH_STYLES["timed"]).
            line_shape=style.get("line_shape", "linear"),
            legendgroup=path,
            customdata=hover_time_labels(x),
            hovertemplate=f"%{{customdata}}<br>{label}: %{{y:,.0f}}{suffix}<extra></extra>",
        )
    figure.update_xaxes(tickmode="array", **london_time_axis(rows["interval_start_utc"]))
    if metric == "home_import_kw":
        figure.update_yaxes(title=unit, rangemode="tozero")
    else:
        # SoC axis fixed 0-100 (chart audit O3): autoscaling to a narrow
        # observed range exaggerates swings that are not there.
        figure.update_yaxes(title=unit, range=[0, 100])
    return figure


_PATHS_TITLE_WORD = {"normal": "unmanaged", "selected": "smart", "timed": "timed tariff"}
"""Lower-case words for the Response chart title, joined "vs" in display order."""


def _paths_title_phrase(path_rows: pd.DataFrame) -> str:
    """The chart title's path phrase: unmanaged vs smart, or with the optional
    "timed" path (decision 0007) present, unmanaged vs timed tariff vs smart --
    the same display order as the chart itself."""

    present = set(path_rows["path_id"])
    return " vs ".join(_PATHS_TITLE_WORD[path] for path in PATH_DISPLAY_ORDER if path in present)


def _difference_figure(
    rows: pd.DataFrame, unit: str, *, name: str = "Smart minus unmanaged"
) -> go.Figure:
    """One P10-P90 difference band; ``name`` is the one legend/hover label.

    Reused for the dispatch block's "dispatched minus day-ahead plan"
    difference (contract §7.2), which is the same shape of frame with a
    different pair of paths behind it, so only the label changes.
    """

    ordered = rows.sort_values("slot_index")
    x = ordered["interval_start_utc"]
    p10 = ordered["p10"]
    p90 = ordered["p90"]
    # The P50 trace below has a hovertemplate; P10/P90 do not (hoverinfo
    # "skip"), so only P50 needs the negative-zero-safe rounding.
    p50 = _no_negative_zero(ordered["p50"], decimals=0)
    hover = hover_time_labels(x)  # London-led hover header (goal review action 20)
    figure = go.Figure()
    band_and_line(
        figure,
        x,
        p10,
        p50,
        p90,
        name=name,
        colour=SERIES_COLOURS["difference"],
        customdata=hover,
        hovertemplate=f"%{{customdata}}<br>%{{y:,.0f}} {unit}<extra></extra>",
    )
    figure.update_xaxes(tickmode="array", **london_time_axis(x))
    # No zero line anywhere else in the shared template (style.py); a
    # difference chart is the one place design 3.5 asks for it, so the reader
    # can see at a glance when the smart path crosses back over unmanaged.
    # MUTED_INK, not the grid token: a zero line in the grid's own colour
    # would be indistinguishable from the gridlines themselves.
    figure.update_yaxes(title=unit, zeroline=True, zerolinecolor=MUTED_INK, zerolinewidth=1)
    return figure


# --- Dispatch response (intraday-dispatch-v1 §10; K5) -----------------------
#
# Day-ahead plan vs dispatched: whether free EVs actually moved energy when
# the intraday price moved, beside the day-ahead reference the addendum
# leaves alone. Local to this module, not style.PATH_STYLES/PATH_LABELS
# (test_style.py pins that shared dict's exact two entries): "day_ahead_plan"
# takes an unused archetype hue, dotted (a reference plan, not a drawn
# path); "dispatched" is exactly the smart path's own teal and width,
# because it *is* that path (contract §7.5: path_id="selected" now means
# "the dispatched fleet"), just under the block's own, more specific label.

_DISPATCH_LABELS = {
    "normal": PATH_LABELS["normal"],
    "day_ahead_plan": "Day-ahead plan",
    "dispatched": "Dispatched (follows the latest intraday price)",
}
_DISPATCH_STYLES = {
    "normal": PATH_STYLES["normal"],
    "day_ahead_plan": {"color": ARCHETYPE_COLOURS[3], "dash": "dot", "width": 2.0},
    "dispatched": PATH_STYLES["selected"],
}
# "selected" for the dispatched series (dashboard_lint's teal-reserved rule:
# teal must sit on a trace whose name or legend group says "smart",
# "selected", "chosen" or "axle"; the block's own label does not, the
# underlying path_id does).
_DISPATCH_LEGENDGROUPS = {
    "normal": "normal",
    "day_ahead_plan": "day_ahead_plan",
    "dispatched": "selected",
}
_SLOT = pd.Timedelta(minutes=30)


def _dispatch_response_figure(bands: pd.DataFrame, normal_rows: pd.DataFrame) -> go.Figure:
    """Unmanaged vs day-ahead plan vs dispatched fleet home import, P50 with P10-P90.

    ``bands`` is ``dispatch_bands`` (contract §7.2, ``metric="home_import_kw"``);
    ``normal_rows`` is the "normal" path's own rows of ``fleet_interval_bands``
    for the same metric, since the day-ahead-plan/dispatched split has no
    unmanaged series of its own (the addendum never touches the unmanaged
    path).
    """

    figure = go.Figure()
    normal_sorted = normal_rows.sort_values("slot_index")
    series_rows = {
        "normal": normal_sorted,
        "day_ahead_plan": bands.loc[
            bands["series"].eq("day_ahead_plan") & bands["metric"].eq("home_import_kw")
        ].sort_values("slot_index"),
        "dispatched": bands.loc[
            bands["series"].eq("dispatched") & bands["metric"].eq("home_import_kw")
        ].sort_values("slot_index"),
    }
    for series, rows in series_rows.items():
        style = _DISPATCH_STYLES[series]
        label = _DISPATCH_LABELS[series]
        x = rows["interval_start_utc"]
        p50 = _no_negative_zero(rows["p50"], decimals=0)
        band_and_line(
            figure,
            x,
            rows["p10"],
            p50,
            rows["p90"],
            name=label,
            colour=style["color"],
            width=style["width"],
            dash=style["dash"],
            legendgroup=_DISPATCH_LEGENDGROUPS[series],
            customdata=hover_time_labels(x),
            hovertemplate=f"%{{customdata}}<br>{label}: %{{y:,.0f}} kW<extra></extra>",
        )
    figure.update_xaxes(tickmode="array", **london_time_axis(normal_sorted["interval_start_utc"]))
    figure.update_yaxes(title="kW", rangemode="tozero")
    return figure


def _replan_count_figure(bands: pd.DataFrame) -> go.Figure:
    """Thin bar row: free EVs whose plan changed that half-hour, mean across weeks (§7.2).

    The mean, not the contract's P50: few free EVs re-plan in any one
    half-hour, so the P50 across weeks is 0 in every half-hour of the default
    run and a P50 bar row drew an empty chart. The mean is the model's own
    world-first column, so the view still computes nothing.
    """

    rows = bands.loc[
        bands["series"].eq("dispatched") & bands["metric"].eq("replan_count")
    ].sort_values("slot_index")
    x = rows["interval_start_utc"]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=x,
            y=rows["mean"],
            name="Re-plans",
            marker_color=MUTED_INK,
            customdata=hover_time_labels(x),
            hovertemplate="%{customdata}<br>%{y:,.1f} EVs re-plan (mean)<extra></extra>",
        )
    )
    figure.update_xaxes(tickmode="array", **london_time_axis(x))
    figure.update_yaxes(title="EVs", rangemode="tozero")
    return figure


def _world_shock_windows(result: Any, world_id: int) -> pd.DataFrame:
    """One simulated week's own price shocks (scripted and stochastic), for ``add_shock_marks``.

    Unlike ``scripted_shock_windows`` (the scripted shocks only, repeated
    identically in every world), this reads ``world_id``'s actual shock
    timeline, since a stochastic shock differs week to week and this block
    shows one representative week's own response.
    """

    shocks = getattr(result, "market_shocks", None)
    columns = ["start_utc", "end_utc", "direction", "known_day_ahead"]
    if shocks is None or shocks.empty:
        return pd.DataFrame(columns=columns)
    rows = shocks.loc[shocks["world_id"].eq(world_id) & shocks["in_study"]].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows["end_utc"] = rows["start_utc"] + rows["duration_slots"] * _SLOT
    return rows.loc[:, columns]


def _dispatch_world_figure(world_slot: pd.DataFrame, shocks: pd.DataFrame) -> go.Figure:
    """One simulated week's own day-ahead plan vs dispatched import (kW) and the intraday close.

    ``world_slot`` is the representative world's rows of ``dispatch_world_slot``
    (contract §7.1); the intraday close sits on a secondary axis (unrelated
    scale, the same reasoning the Plan and Position charts give for pairing
    power with price) and shock spans mark when the price moved for a
    scripted or stochastic reason (``add_shock_marks``).
    """

    ordered = world_slot.sort_values("slot_index")
    x = ordered["interval_start_utc"]
    figure = go.Figure()
    add_shock_marks(figure, shocks)
    columns = (("day_ahead_plan", "day_ahead_plan_kwh"), ("dispatched", "dispatched_kwh"))
    for series, column in columns:
        style = _DISPATCH_STYLES[series]
        label = _DISPATCH_LABELS[series]
        figure.add_trace(
            go.Scatter(
                x=x,
                y=ordered[column] / 0.5,
                mode="lines",
                name=label,
                line={"color": style["color"], "width": style["width"], "dash": style["dash"]},
                legendgroup=_DISPATCH_LEGENDGROUPS[series],
                customdata=hover_time_labels(x),
                hovertemplate=f"%{{customdata}}<br>{label}: %{{y:,.0f}} kW<extra></extra>",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=ordered["latest_close_gbp_per_mwh"],
            mode="lines",
            name="Intraday close",
            line={"color": SERIES_COLOURS["observed"], "width": 1.0, "dash": "dash"},
            yaxis="y2",
            customdata=hover_time_labels(x),
            hovertemplate="%{customdata}<br>%{y:,.0f} £/MWh<extra></extra>",
        )
    )
    figure.update_xaxes(tickmode="array", **london_time_axis(x))
    figure.update_yaxes(title="kW", rangemode="tozero")
    # Second, independent y-axis for the intraday close (unrelated scale to
    # kW): the same pairing action_decision.py's Plan chart and
    # trading_position.py's position chart use for power beside price.
    figure.update_layout(
        yaxis2={
            "title": "Intraday close (£/MWh)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        }
    )
    return figure


def _dispatch_split_caption(split: pd.DataFrame) -> str:
    """ "K of N locked, share c; free EVs re-plan past the threshold" (contract §7.3, §10)."""

    locked = int(split["locked_count"].sum())
    total = int(split["ev_count"].sum())
    share = percent(100.0 * float(split["commitment_share"].iat[0]), decimals=0)
    threshold = float(split["replan_threshold_gbp_per_mwh"].iat[0])
    return (
        f"{locked:,} of {total:,} EVs locked to their day-ahead plan (share {share}); free EVs "
        f"re-plan when the saving is more than £{threshold:g}/MWh plus the spread."
    )


_MOVED_KWH_CAPTION = (
    "Moved energy sums to about zero per session night: charging shifts in time, not amount, "
    "apart from early departures and top-ups."
)
"""Contract §7.1's ``moved_kwh`` caption, in the block's own words."""

_DISPATCH_TABLE_LABELS = {**_DISPATCH_LABELS, "difference": "Dispatched minus day-ahead plan"}
_DISPATCH_RENAME = {
    "series": "Series",
    "interval_start_utc": "Interval start (UTC)",
    "interval_start_london": "Interval start (London)",
    "mean": "Mean",
    "p10": "P10",
    "p50": "P50",
    "p90": "P90",
}


def _dispatch_bands_frame(
    bands: pd.DataFrame, normal_rows: pd.DataFrame, metric: str
) -> pd.DataFrame:
    """Series/Interval/Mean/P10/P50/P90 table for one ``dispatch_bands`` metric (contract §7.2).

    ``normal_rows`` (the unmanaged path's rows of ``fleet_interval_bands``)
    is prepended for ``home_import_kw``, since ``dispatch_bands`` carries no
    unmanaged series of its own.
    """

    rows = bands.loc[bands["metric"].eq(metric)].copy()
    rows["series"] = rows["series"].map(_DISPATCH_TABLE_LABELS)
    parts = [rows.loc[:, list(_DISPATCH_RENAME)]]
    if metric == "home_import_kw":
        normal = normal_rows.loc[
            :, ["interval_start_utc", "interval_start_london", "mean", "p10", "p50", "p90"]
        ].copy()
        normal.insert(0, "series", _DISPATCH_LABELS["normal"])
        parts.insert(0, normal)
    return pd.concat(parts, ignore_index=True).rename(columns=_DISPATCH_RENAME)


def _dispatch_world_frame(world_slot: pd.DataFrame) -> pd.DataFrame:
    """This world's day-ahead plan, dispatched (kW) and intraday close, one row per half-hour."""

    return pd.DataFrame(
        {
            "Interval start (London)": world_slot["interval_start_london"],
            "Day-ahead plan (kW)": (world_slot["day_ahead_plan_kwh"] / 0.5).round(1),
            "Dispatched (kW)": (world_slot["dispatched_kwh"] / 0.5).round(1),
            "Intraday close (£/MWh)": world_slot["latest_close_gbp_per_mwh"].round(1),
        }
    )


def _render_dispatch_block(st: Any, result: Any, path_rows: pd.DataFrame, evidence: str) -> None:
    """The "day-ahead plan vs dispatched" block (intraday-dispatch-v1 §10), home import only.

    Reads ``dispatch_bands``, the representative world's ``dispatch_world_slot``
    and ``dispatch_split``; ``None`` exactly when the run had no action or the
    intraday dispatch switch was off (contract §7), in which case this block
    is left out rather than drawn empty.
    """

    bands = getattr(result, "dispatch_bands", None)
    world_slot = getattr(result, "dispatch_world_slot", None)
    split = getattr(result, "dispatch_split", None)
    if bands is None or world_slot is None or split is None:
        return

    normal_rows = path_rows.loc[path_rows["path_id"].eq("normal")]
    st.markdown("**Day-ahead plan vs dispatched**")
    st.caption(_dispatch_split_caption(split))
    chart_block(
        st,
        _dispatch_response_figure(bands, normal_rows),
        title="Unmanaged vs day-ahead plan vs dispatched (kW)",
        caption=(
            f"Median and P10–P90 across {int(bands['world_count'].iat[0])} simulated weeks. "
            f"{evidence}"
        ),
        frame=_dispatch_bands_frame(bands, normal_rows, "home_import_kw"),
        definition=(
            "The day-ahead plan is the smart path as it would run without intraday dispatch: "
            "every EV plans once, at plug-in, and keeps that plan. Dispatched is the smart "
            "path with intraday dispatch on: locked EVs keep their day-ahead plan, and free "
            "EVs re-plan every hour on the latest intraday price."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="action-response-dispatch-paths",
    )
    diff_rows = bands.loc[bands["series"].eq("difference") & bands["metric"].eq("home_import_kw")]
    chart_block(
        st,
        _difference_figure(diff_rows, "kW", name=_DISPATCH_TABLE_LABELS["difference"]),
        title="Dispatched minus day-ahead plan, paired per simulated week (kW)",
        caption="P10–P90 of each week's own difference, across simulated weeks. Median 0: in "
        "most half-hours no free EV re-plans.",
        frame=_dispatch_bands_frame(bands, normal_rows, "home_import_kw"),
        definition=(
            "Dispatched minus day-ahead plan. The difference is worked out inside each "
            "simulated week first. The band is then the spread across weeks. Percentiles are "
            "never subtracted."
        ),
        height=CHART_HEIGHTS["small_multiple_panel"],
        key="action-response-dispatch-difference",
    )
    st.caption(_MOVED_KWH_CAPTION)
    chart_block(
        st,
        _replan_count_figure(bands),
        title="EVs re-planning each half-hour",
        caption=(
            f"Mean across simulated weeks; the median is 0, as few free EVs re-plan. {evidence}"
        ),
        frame=_dispatch_bands_frame(bands, normal_rows, "replan_count"),
        definition=(
            "Free EVs whose plan changed at that half-hour's hourly decision. Always 0 for a "
            "locked EV, and 0 in the half-hours between decisions."
        ),
        height=CHART_HEIGHTS["small_multiple_panel"],
        key="action-response-dispatch-replan-count",
    )

    representative = result.representative_world_id
    world_rows = world_slot.loc[world_slot["world_id"].eq(representative)]
    shocks = _world_shock_windows(result, representative)
    chart_block(
        st,
        _dispatch_world_figure(world_rows, shocks),
        title="One simulated week: dispatch response to the market (kW, £/MWh)",
        caption=f"The median simulated week's own path, not a statistic across weeks. {evidence}",
        frame=_dispatch_world_frame(world_rows),
        definition=(
            "One simulated week's own day-ahead plan and dispatched fleet home import, with "
            "the intraday close it responded to and the week's own price shocks marked, so "
            "the response around a surprise is visible."
        ),
        height=CHART_HEIGHTS["time_series_dual_axis"],
        key="action-response-dispatch-world",
    )


def _combined_frame(path_rows: pd.DataFrame, diff_rows: pd.DataFrame) -> pd.DataFrame:
    # "One shared data expander" (design 4.3) is approximated here by giving
    # both chart_block calls the same combined frame: chart_block draws one
    # expander per chart (design 3.4), so this is the closest fit without
    # editing that shared component.
    paths = path_rows.loc[
        :, ["path_id", "interval_start_utc", "interval_start_london", "mean", "p10", "p50", "p90"]
    ].copy()
    paths["path_id"] = paths["path_id"].map(PATH_LABELS)
    diff = diff_rows.loc[
        :, ["interval_start_utc", "interval_start_london", "mean", "p10", "p50", "p90"]
    ].copy()
    diff.insert(0, "path_id", "Smart minus unmanaged")
    combined = pd.concat([paths, diff], ignore_index=True)
    return combined.rename(
        columns={
            "path_id": "Series",
            "interval_start_utc": "Interval start (UTC)",
            "interval_start_london": "Interval start (London)",
            "mean": "Mean",
            "p10": "P10",
            "p50": "P50",
            "p90": "P90",
        }
    )


def render_action_response(st: Any, result: Any) -> None:
    """Render the Response lens: unmanaged vs smart, and the paired difference."""

    if result.action_summary is None:
        st.info("This result has no Axle action.")
        return

    st.markdown(_summary_sentence(result.action_summary))
    metric_key = _metric_control(st)
    evidence = f"Evidence: {_evidence_text(result.evidence_kind)}."
    if metric_key == EVENTS_METRIC:
        render_event_response(st, result, evidence)
        return
    metric, unit = _METRICS[metric_key]

    # Four tiles (polish plan G9); the weekly changes are a small table
    # after the charts, as checks on them rather than chrome before them.
    _headline_tile_row(st, result)

    bands = result.fleet_interval_bands
    path_rows = bands.loc[bands["metric"].eq(metric)]
    diff_rows = result.difference_bands.loc[result.difference_bands["metric"].eq(metric)]
    # battery_soc_percent already carries "percentage points" in this frame
    # (contract 4.1); home_import_kw's difference stays in kW.
    diff_unit = diff_rows["unit"].iat[0]
    combined = _combined_frame(path_rows, diff_rows)

    chart_block(
        st,
        _paths_figure(path_rows, metric, unit),
        title=f"{metric_key}: {_paths_title_phrase(path_rows)} ({unit})",
        caption=(
            f"Median and P10–P90 across {int(path_rows['world_count'].iat[0])} simulated "
            f"weeks. {evidence}"
        ),
        frame=combined,
        definition=_METRIC_DEFINITIONS[metric],
        height=CHART_HEIGHTS["time_series"],
        key=f"action-response-paths-{metric}",
    )
    chart_block(
        st,
        _difference_figure(diff_rows, diff_unit),
        title=f"Smart minus unmanaged, paired per simulated week ({diff_unit})",
        caption=(
            "P10–P90 of each week's own difference, across simulated weeks; every night is a "
            f"smart night. {evidence}"
        ),
        frame=combined,
        definition=(
            "Smart minus unmanaged. The difference is worked out inside each simulated week "
            "first. The band is then the spread across weeks. Percentiles are never "
            f"subtracted. {_STUDY_END_NOTE}"
        ),
        height=CHART_HEIGHTS["time_series"],
        key=f"action-response-difference-{metric}",
    )
    if metric == "home_import_kw":
        # The herding finding is about the home-import shape specifically
        # (a new peak where the fleet piles onto cheap forecast half-hours,
        # a cut where normal-path evening charging used to be); it does not
        # describe the Battery SoC view of the same difference.
        finding, tooltip, timed_caption = _herding_caption(result.weekly_peak_summary)
        st.caption(finding, help=tooltip)
        if timed_caption:
            st.caption(timed_caption)
        _render_dispatch_block(st, result, path_rows, evidence)

    st.markdown("**Weekly change, smart minus unmanaged**")
    st.dataframe(
        _weekly_change_table(result.difference_weekly_bands), width="stretch", hide_index=True
    )
    st.caption(_unserved_travel_caption(result.difference_weekly_bands))


# --- Events (decision 0004 items 55, 56, 58; trading contract v1 §5.6) --------

_EVENT_TYPE_TEXT = {
    "price_shock_known": "Price shock, known a day ahead",
    "price_shock_surprise": "Price shock, a surprise",
    "turn_down": "Turn-down request",
    "turn_up": "Turn-up request",
    "control_outage": "Charger control outage",
}
_REQUEST_TYPES = ("turn_down", "turn_up")
_SHOCK_TYPES = ("price_shock_known", "price_shock_surprise")
_SPAN_TEXT = {"before": "Before the window", "in": "In the window", "after": "After the window"}


def _scope_text(scope: str, zone_summary: pd.DataFrame | None) -> str:
    """ "Fleet" for a national event, else the zone's label from the result's ``zone_summary``."""

    if scope == "national":
        return "Fleet"
    if zone_summary is not None:
        labels = dict(zip(zone_summary["zone_id"], zone_summary["zone_label"], strict=False))
        if scope in labels:
            return labels[scope]
    return scope.replace("_", " ").capitalize()


def _event_starts(bands: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Each event's first window half-hour (UTC), from its own response rows."""

    first = bands.loc[bands["relative_slot"].eq(0)].drop_duplicates("event_id")
    return dict(zip(first["event_id"], first["interval_start_utc"], strict=True))


def _event_names(events: pd.DataFrame, bands: pd.DataFrame) -> dict[str, str]:
    """Event id -> on-screen name, in events-table order (enabled events only)."""

    starts = _event_starts(bands)
    names = {}
    for event_id in events.loc[events["enabled"].astype(bool), "event_id"]:
        start = starts.get(event_id)
        london = start.tz_convert("Europe/London") if start is not None else None
        names[event_id] = event_label(event_id, london)
    return names


def _events_table(
    events: pd.DataFrame,
    names: dict[str, str],
    bands: pd.DataFrame,
    zone_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    """Every scripted event of the run, one row each, in plain words."""

    starts = _event_starts(bands)
    rows = []
    for event in events.loc[events["enabled"].astype(bool)].itertuples(index=False):
        start = starts.get(event.event_id)
        size = float(event.size)
        if event.event_type in _SHOCK_TYPES:
            size_text = f"{size:+g} GW"
        elif event.event_type == "control_outage":
            size_text = f"{size:.0%} of sessions"
        else:
            size_text = "No cap" if size != size else f"{size:g} MW cap"
        payment = float(event.payment_gbp_per_mwh)
        rows.append(
            {
                "Event": names[event.event_id],
                "Type": _EVENT_TYPE_TEXT.get(event.event_type, event.event_type),
                "Starts (London)": hover_time_labels(pd.Series([start]))[0] if start else "",
                "Length (h)": int(event.duration_minutes) / 60,
                "Covers": _scope_text(event.scope, zone_summary),
                "Size": size_text,
                "Notice": "Day ahead"
                if event.notice == "day_ahead"
                else f"{int(event.notice_minutes)} min",
                # Blank, not "None", for events that pay nothing (price shocks, outages).
                "Payment, illustrative (£/MWh)": f"{payment:g}" if payment == payment else "",
            }
        )
    return pd.DataFrame(rows)


def _event_rows(bands: pd.DataFrame, event_id: str, series: str, metric: str) -> pd.DataFrame:
    rows = bands.loc[
        bands["event_id"].eq(event_id) & bands["series"].eq(series) & bands["metric"].eq(metric)
    ]
    return rows.sort_values("slot_index")


def _event_axis(figure: go.Figure, x: pd.Series) -> None:
    """London clock ticks every 2 h (every 4 h past a 12 h span), so 390 px stays legible."""

    hours = (x.max() - x.min()) / pd.Timedelta(hours=1)
    every = 2 if hours <= 12 else 4
    figure.update_xaxes(tickmode="array", **london_time_axis(x, every_hours=every))


def _mark_window(figure: go.Figure, event: pd.Series, shocks: pd.DataFrame) -> None:
    """Show the event's window: price shocks by kind, other events by two dotted rules.

    Scripted price shocks (this one and any other overlapping the shown
    hours) use the Plan chart's shock marks, known filled and surprise
    dashed, so the same shape means the same thing on both lenses. A
    request or outage window is two dotted rules instead, so it is never
    mistaken for a price shock.
    """

    if not shocks.empty:
        add_shock_marks(figure, shocks)
    if event["event_type"] in _SHOCK_TYPES:
        return
    for edge, text in ((event["window_start"], "Window"), (event["window_end"], None)):
        label = (
            {
                "annotation_text": text,
                "annotation_position": "top right",
                "annotation_font_color": MUTED_INK,
            }
            if text
            else {}
        )
        figure.add_vline(x=edge, line_dash="dot", line_color=MUTED_INK, **label)


def _event_import_figure(bands: pd.DataFrame, event: pd.Series, shocks: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    _mark_window(figure, event, shocks)
    x = None
    series = ["normal", "selected"]
    # Contract §7.4, changed by §11.7: the day-ahead plan reference path
    # rides beside unmanaged and dispatched only when the intraday dispatch
    # switch was on (the series is absent from ``bands`` otherwise).
    if bands["series"].eq("day_ahead_plan").any():
        series.append("day_ahead_plan")
    for name in series:
        rows = _event_rows(bands, event["event_id"], name, "home_import_kw")
        x = rows["interval_start_utc"]
        style = PATH_STYLES[name] if name in PATH_STYLES else _DISPATCH_STYLES[name]
        label = PATH_LABELS[name] if name in PATH_LABELS else _DISPATCH_LABELS[name]
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
            legendgroup=name,
            customdata=hover_time_labels(x),
            hovertemplate=f"%{{customdata}}<br>{label}: %{{y:,.0f}} kW<extra></extra>",
        )
    baseline = _event_rows(bands, event["event_id"], "selected", "baseline_kw")
    if not baseline.empty:
        figure.add_trace(
            go.Scatter(
                x=baseline["interval_start_utc"],
                y=baseline["p50"],
                mode="lines",
                line={"color": MUTED_INK, "width": 1.5, "dash": "dot"},
                name="Settlement baseline (median)",
                hovertemplate="Baseline: %{y:,.0f} kW<extra></extra>",
            )
        )
    _event_axis(figure, x)
    figure.update_yaxes(title="kW", rangemode="tozero")
    return figure


def _event_difference_figure(
    bands: pd.DataFrame, event: pd.Series, shocks: pd.DataFrame
) -> go.Figure:
    rows = _event_rows(bands, event["event_id"], "difference", "home_import_kw")
    x = rows["interval_start_utc"]
    figure = go.Figure()
    _mark_window(figure, event, shocks)
    band_and_line(
        figure,
        x,
        rows["p10"],
        # Rounded like the hover (see _no_negative_zero), so it never reads "-0".
        _no_negative_zero(rows["p50"], decimals=0),
        rows["p90"],
        name="Smart minus unmanaged",
        colour=SERIES_COLOURS["difference"],
        customdata=hover_time_labels(x),
        hovertemplate="%{customdata}<br>%{y:,.0f} kW<extra></extra>",
    )
    _event_axis(figure, x)
    # The zero line is the point of this chart: above it smart draws more
    # than unmanaged, below it less (see ``_difference_figure``).
    figure.update_yaxes(title="kW", zeroline=True, zerolinecolor=MUTED_INK, zerolinewidth=1)
    return figure


def _price_increment_figure(
    bands: pd.DataFrame, event: pd.Series, shocks: pd.DataFrame
) -> go.Figure:
    rows = _event_rows(bands, event["event_id"], "market", "price_increment_gbp_per_mwh")
    x = rows["interval_start_utc"]
    figure = go.Figure()
    _mark_window(figure, event, shocks)
    band_and_line(
        figure,
        x,
        rows["p10"],
        rows["p50"],
        rows["p90"],
        name="Price moved by this shock",
        colour=SERIES_COLOURS["synthetic"],
        customdata=hover_time_labels(x),
        hovertemplate="%{customdata}<br>%{y:,.0f} £/MWh<extra></extra>",
    )
    _event_axis(figure, x)
    figure.update_yaxes(title="£/MWh", zeroline=True, zerolinecolor=MUTED_INK, zerolinewidth=1)
    return figure


def _span_rows(rows: pd.DataFrame, span: str) -> pd.DataFrame:
    if span == "in":
        return rows.loc[rows["in_window"]]
    before = rows["relative_slot"] < 0
    return rows.loc[before if span == "before" else ~before & ~rows["in_window"]]


def _event_tiles(st: Any, difference: pd.DataFrame) -> None:
    """Before, in and after the window: the half-hour with the largest median change.

    Each tile looks up one half-hour's P50 of the paired smart-minus-
    unmanaged difference (computed inside each week first, then across
    weeks) and names its P10–P90 and time in the tooltip. Nothing is summed
    across half-hours, because summing P50s is not the P50 of the sum
    (AGENTS.md); "largest" is by size, so a cut and a rise are both found.
    """

    columns = kpi_columns(st, 3)
    for column, span in zip(columns, ("before", "in", "after"), strict=True):
        rows = _span_rows(difference, span)
        if rows.empty:
            kpi(column, _SPAN_TEXT[span], UNAVAILABLE, context="Outside the study")
            continue
        row = rows.loc[rows["p50"].abs().idxmax()]
        when = hover_time_labels(pd.Series([row["interval_start_utc"]]))[0]
        kpi(
            column,
            _SPAN_TEXT[span],
            _signed(row["p50"], ""),
            unit="kW",
            context="Largest median change, smart − unmanaged",
            help=(
                f"The half-hour with the largest median change, at {when}. P10–P90 across "
                f"{int(row['world_count'])} simulated weeks: {_signed(row['p10'], 'kW')} to "
                f"{_signed(row['p90'], 'kW')}."
            ),
        )


def _event_frame(bands: pd.DataFrame, event_id: str) -> pd.DataFrame:
    rows = bands.loc[bands["event_id"].eq(event_id)]
    series = rows["series"].map(
        {
            "normal": "Unmanaged",
            "selected": "Smart",
            "difference": "Smart − unmanaged",
            # Contract §7.4, changed by §11.7: present only with the
            # intraday dispatch switch on; every other row falls back to
            # "Market" below (the price/baseline rows, which carry no series).
            "day_ahead_plan": _DISPATCH_LABELS["day_ahead_plan"],
        }
    )
    metric = rows["metric"].map(
        {
            "home_import_kw": "Home import (kW)",
            "baseline_kw": "Settlement baseline (kW)",
            "delivered_kw": "Delivered (kW)",
            "price_increment_gbp_per_mwh": "Price moved by this shock (£/MWh)",
            "day_ahead_gbp_per_mwh": "Day-ahead price (£/MWh)",
            "imbalance_gbp_per_mwh": "Imbalance price (£/MWh)",
        }
    )
    return pd.DataFrame(
        {
            "Series": series.fillna("Market"),
            "Measure": metric,
            "Interval start (London)": hover_time_labels(rows["interval_start_utc"]),
            "In window": rows["in_window"].map({True: "Yes", False: ""}),
            "P10": rows["p10"].round(1),
            "P50": rows["p50"].round(1),
            "P90": rows["p90"].round(1),
        }
    )


def render_event_response(st: Any, result: Any, evidence: str) -> None:
    """The Events metric: the fleet's response around one chosen scripted event."""

    events = result.events
    bands = result.event_response_bands
    if events is None or bands is None or not events["enabled"].astype(bool).any():
        st.info(
            "No scripted events in this run. Tick presets under Edit assumptions, Events, then "
            "Run simulation."
        )
        return
    names = _event_names(events, bands)
    st.dataframe(
        _events_table(events, names, bands, result.zone_summary), width="stretch", hide_index=True
    )
    st.caption("Illustrative scenarios, the same in every simulated week; not forecasts.")

    event_id = st.selectbox(
        "Event",
        options=list(names),
        format_func=names.__getitem__,
        key="action-response-event",
    )
    event_id = event_id if event_id in names else next(iter(names))
    row = events.loc[events["event_id"].eq(event_id)].iloc[0]
    event_rows = bands.loc[bands["event_id"].eq(event_id)]
    window = event_rows.loc[event_rows["in_window"]]
    event = pd.Series(
        {
            "event_id": event_id,
            "event_type": row["event_type"],
            "window_start": window["interval_start_utc"].min(),
            "window_end": window["interval_start_utc"].max() + pd.Timedelta(minutes=30),
        }
    )
    shown = event_rows["interval_start_utc"]
    shocks = scripted_shock_windows(result)
    shocks = shocks.loc[(shocks["end_utc"] > shown.min()) & (shocks["start_utc"] <= shown.max())]
    scope = _scope_text(row["scope"], result.zone_summary)
    weeks = int(event_rows["world_count"].iat[0])
    frame = _event_frame(bands, event_id)

    difference = _event_rows(bands, event_id, "difference", "home_import_kw")
    _event_tiles(st, difference)
    chart_block(
        st,
        _event_import_figure(bands, event, shocks),
        title=f"{scope} home import around the event, unmanaged vs smart (kW)",
        caption=(
            f"Median, P10–P90 across {weeks} simulated weeks, from 2 h before to 4 h after. "
            f"Unmanaged: no smart charging. {evidence}"
        ),
        frame=frame,
        definition=(
            "Home import of the EVs the event covers (the whole fleet, or one zone's EVs), on "
            "each path, per simulated week, then P10, P50 and P90 across weeks. Unmanaged is "
            "the same weeks with no smart charging. To see the same weeks without this event, "
            "run again with the event off: Compare then pairs the two runs week by week."
        ),
        height=CHART_HEIGHTS["time_series"],
        key=f"action-response-event-import-{event_id}",
    )
    chart_block(
        st,
        _event_difference_figure(bands, event, shocks),
        title="Smart minus unmanaged around the event, paired per simulated week (kW)",
        caption=(
            "Below zero smart draws less; changes just outside the window are charging moved "
            f"around it, the rebound. {evidence}"
        ),
        frame=frame,
        definition=(
            "Smart minus unmanaged home import, worked out inside each simulated week first, "
            "then P10, P50 and P90 across weeks. Percentiles are never subtracted. Charging "
            "moved to just before or after the window is the rebound. It is reported, not "
            "penalised, because moving charging around a window is what a fleet does when it "
            "is asked to turn down."
        ),
        height=CHART_HEIGHTS["time_series"],
        key=f"action-response-event-difference-{event_id}",
    )
    if row["event_type"] in _SHOCK_TYPES:
        chart_block(
            st,
            _price_increment_figure(bands, event, shocks),
            title="Price moved by this shock alone (£/MWh)",
            caption=(
                f"Median and P10–P90 across {weeks} simulated weeks. Synthetic prices, "
                "illustrative shock."
            ),
            frame=frame,
            definition=(
                "The shock's own price increment on the synthetic supply curve: the day-ahead "
                "price with the shock minus the price without it, for a known shock; the "
                "intraday and imbalance prices for a surprise, which the day-ahead price "
                "never sees. The curve is convex, so the same GW moves the price more when "
                "the system is already tight."
            ),
            height=CHART_HEIGHTS["time_series"],
            key=f"action-response-event-price-{event_id}",
        )
    if row["event_type"] in _REQUEST_TYPES:
        has_baseline = event_rows["metric"].eq("baseline_kw").any()
        st.caption(
            "This run carries no settlement baseline, so delivery against it and its "
            "illustrative payment are not shown."
            if not has_baseline
            else "Dotted line: the settlement baseline delivery is measured against."
        )
    if row["event_type"] == "control_outage":
        st.caption(
            "Chargers that ignore their plans charge at once, as unmanaged would, so smart "
            "moves towards unmanaged in the window."
        )
