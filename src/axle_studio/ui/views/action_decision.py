"""Smart charging lens 1, Plan: what the smart charger plans, and what it knew.

Reads ``action_summary``, ``forecast_price_bands`` (one day-ahead path per
simulated week, shown as P10/P50/P90 across weeks), ``fleet_interval_bands``
(selected path, ``home_import_kw``, to shade when charging actually
happened), ``smart_charging_summary``, ``price_band_shift`` and
``evidence_kind`` (the chart captions' "illustrative"/"synthetic" wording)
(docs/contracts/results-v2.md section 9, view-to-field map, and decision 0004
item 43). The plans were already made by the run: this view narrates them
and never calls the model (AGENTS.md).

The internal path id stays ``"selected"`` (contract v2, ``path_id``); every
user-facing legend, title and caption says "smart" instead (``style.PATH_LABELS``).

Scripted price shocks (decision 0004 items 55, 56; trading contract v1
§5.6a, §6) are shaded on the price chart from ``market_shocks``: known
day-ahead shocks as a filled span, surprises as a dashed outline, so the two
differ by shape as well as by legend. Only scripted shocks are drawn: they
are the same in every simulated week, so they belong on a chart summarised
across weeks. The random shocks differ from week to week and would mark
half-hours that only one week had. ``scripted_shock_windows``,
and ``add_shock_marks`` are shared with the Response lens's event view;
event names come from ``run_controller.event_label``, which owns the preset
labels, so no Smart charging view imports the model package.

``_summary_sentence`` and ``_evidence_text`` are also imported by
``action_response.py`` and ``action_cost.py``: one text-mapping helper
serves all three lenses, so they never say the same thing in different
words.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..components.chart_table import chart_block
from ..components.kpi import kpi, kpi_columns
from ..run_controller import event_label
from ..style import (
    CHART_HEIGHTS,
    MUTED_INK,
    PATH_DISPLAY_ORDER,
    PATH_LABELS,
    PATH_STYLES,
    SERIES_COLOURS,
    TRANSPARENT,
    band_and_line,
    band_fill,
    format_quantity,
    hover_time_labels,
    london_time_axis,
    percent,
    present_paths,
)

# Plain-English text for the result's evidence_kind values (review: no raw
# slugs anywhere a person reads). A code not listed here still reads as
# comma-separated words rather than an underscored slug.
_EVIDENCE_TEXT = {
    "illustrative": "illustrative",
    "synthetic": "synthetic",
    "illustrative_synthetic": "illustrative, synthetic",
}


def _evidence_text(kind: object) -> str:
    text = str(kind)
    return _EVIDENCE_TEXT.get(text, text.replace("_", ", "))


def _summary_sentence(summary: Any) -> str:
    """The one-sentence summary shown on all three Smart charging lenses."""

    margin = f"{summary.departure_margin_hours:g}"
    unit = "hour" if margin == "1" else "hours"
    return (
        "Smart charging: each EV charges only what it needs to reach its target, in the "
        "cheapest forecast half-hours, and aims to finish by its usual departure time minus "
        f"{margin} {unit}. An EV that leaves earlier than that misses the rest of its plan."
    )


_SLOT = pd.Timedelta(minutes=30)
_TIMED_COSTING_NOTE = (
    "All three policies are costed the same way, at the day-ahead price; no tariff rate is applied."
)
"""Shown wherever this lens displays a price or money figure for the optional timed path
(decision 0007): "timed tariff" names the common real-world pattern of a timed tariff, but no
tariff price is modelled, so the caption says so plainly rather than leaving "tariff" to be
read as a retail rate."""
_KNOWN_SHOCK_ALPHA = 0.10
"""Fill of a known shock's span: faint, so the bands drawn over it stay readable."""

_ARROW_GAP = pd.Timedelta(hours=2)
"""Closest two same-direction shock arrows may start; nearer ones would overlap at 390 px."""

SHOCK_KIND_LABELS = {True: "Known day-ahead shock", False: "Surprise shock"}
"""Legend keys for the two shock kinds (trading contract v1 §6)."""


def scripted_shock_windows(result: Any) -> pd.DataFrame:
    """The run's scripted price shocks in the study, one row each, from ``market_shocks``.

    A scripted shock is the same in every simulated week, so ``market_shocks``
    repeats it once per world; one world's rows are kept. Columns:
    ``shock_id``, ``label``, ``known_day_ahead``, ``direction``, ``size_gw``,
    ``start_utc``, ``end_utc`` (window end, UTC) and ``start_london``. Empty
    when the run has no scripted shocks (or no market, a no-action run).
    """

    columns = [
        "shock_id",
        "label",
        "known_day_ahead",
        "direction",
        "size_gw",
        "start_utc",
        "end_utc",
        "start_london",
    ]
    shocks = getattr(result, "market_shocks", None)
    if shocks is None or shocks.empty:
        return pd.DataFrame(columns=columns)
    scripted = shocks.loc[shocks["source"].eq("scripted") & shocks["in_study"]]
    if scripted.empty:
        return pd.DataFrame(columns=columns)
    rows = scripted.loc[scripted["world_id"].eq(scripted["world_id"].min())].copy()
    rows["end_utc"] = rows["start_utc"] + rows["duration_slots"] * _SLOT
    rows["label"] = [
        event_label(shock_id, start)
        for shock_id, start in zip(rows["shock_id"], rows["start_london"], strict=True)
    ]
    return rows.loc[:, columns].reset_index(drop=True)


def add_shock_marks(figure: go.Figure, windows: pd.DataFrame) -> None:
    """Shade each shock window on ``figure``: known filled, surprise dashed outline.

    Shape as well as legend tells the two apart (a filled span against an
    outline), so the difference survives greyscale and colour-blind
    viewing; both use the muted ink rather than a series colour, because a
    shock is context, not a series. An arrow at the bottom of the span gives
    the direction (up: tighter, dearer; down: surplus, cheaper). One legend
    key per kind present, from an empty trace, because Plotly shapes have no
    legend.
    """

    last_arrow: dict[str, pd.Timestamp] = {}
    for row in windows.sort_values("start_utc").itertuples(index=False):
        # A second same-direction arrow within _ARROW_GAP of the last one
        # would sit on top of it (a surprise 30 min into a cold still
        # evening) and says nothing new, so only its span is drawn.
        previous = last_arrow.get(row.direction)
        show_arrow = previous is None or row.start_utc - previous >= _ARROW_GAP
        common: dict[str, Any] = {"x0": row.start_utc, "x1": row.end_utc, "layer": "below"}
        if show_arrow:
            last_arrow[row.direction] = row.start_utc
            common |= {
                "annotation_text": "▲" if row.direction == "up" else "▼",
                # Bottom, not top: the top edge holds the "Next day's prices
                # published" label, and a shock on night 0 (every cold still
                # week) overlapped it (decision 0004 item 26: no overlapping
                # annotations at 1440 or 390 px).
                "annotation_position": "bottom left",
                "annotation_font_color": MUTED_INK,
            }
        if row.known_day_ahead:
            figure.add_vrect(
                fillcolor=band_fill(MUTED_INK, _KNOWN_SHOCK_ALPHA), line_width=0, **common
            )
        else:
            figure.add_vrect(
                fillcolor=TRANSPARENT,
                line={"color": MUTED_INK, "width": 1, "dash": "dash"},
                **common,
            )
    for known in (True, False):
        if not windows["known_day_ahead"].eq(known).any():
            continue
        # A real timestamp with no y value draws nothing but keeps Plotly's
        # axis type "date": with x=[None], a legend key added before the
        # series made the axis linear and every date shape NaN.
        figure.add_trace(
            go.Scatter(
                x=[windows["start_utc"].iat[0]],
                y=[None],
                mode="markers",
                marker={
                    "symbol": "square" if known else "square-open",
                    "size": 12,
                    "color": band_fill(MUTED_INK, 0.5) if known else MUTED_INK,
                },
                name=SHOCK_KIND_LABELS[known],
                hoverinfo="skip",
            )
        )


def shock_window_table(windows: pd.DataFrame) -> pd.DataFrame:
    """The shaded shocks as a table: name, kind, direction, size and London times."""

    return pd.DataFrame(
        {
            "Scripted shock": windows["label"],
            "Kind": windows["known_day_ahead"].map(SHOCK_KIND_LABELS),
            "Direction": windows["direction"].map({"up": "Up (dearer)", "down": "Down (cheaper)"}),
            "Size (GW)": windows["size_gw"].astype(float),
            "Starts (London)": hover_time_labels(windows["start_utc"]),
            "Ends (London)": hover_time_labels(windows["end_utc"]),
        }
    )


def _selected_home_import(bands: pd.DataFrame) -> pd.DataFrame:
    """The smart (selected-path) fleet home import (kW), one row per slot (contract 3.6)."""

    rows = bands.loc[bands["metric"].eq("home_import_kw") & bands["path_id"].eq("selected")]
    return rows.sort_values("slot_index")


def _price_bands(result: Any) -> pd.DataFrame:
    """Day-ahead price P10/P50/P90 across simulated weeks, one row per slot (contract 4.7)."""

    return result.forecast_price_bands.sort_values("slot_index")


def _price_table(prices: pd.DataFrame, charging: pd.DataFrame) -> pd.DataFrame:
    # Rounded for the table (goal review action 20: tables at 0-1 dp); the
    # chart itself plots the unrounded values.
    price_rounded = prices.loc[
        :, ["interval_start_utc", "interval_start_london", "p10", "p50", "p90"]
    ].round({"p10": 0, "p50": 0, "p90": 0})
    kw_rounded = charging.loc[:, ["interval_start_utc", "p10", "p50", "p90"]].round(
        {"p10": 1, "p50": 1, "p90": 1}
    )
    merged = price_rounded.merge(
        kw_rounded.rename(
            columns={
                "p10": "smart_home_import_p10_kw",
                "p50": "smart_home_import_kw",
                "p90": "smart_home_import_p90_kw",
            }
        ),
        on="interval_start_utc",
        how="left",
    )
    return merged.rename(
        columns={
            "interval_start_utc": "Interval start (UTC)",
            "interval_start_london": "Interval start (London)",
            "p10": "Day-ahead price P10 (£/MWh)",
            "p50": "Day-ahead price, median (£/MWh)",
            "p90": "Day-ahead price P90 (£/MWh)",
            "smart_home_import_p10_kw": "Smart home import P10 (kW)",
            "smart_home_import_kw": "Smart home import, median (kW)",
            "smart_home_import_p90_kw": "Smart home import P90 (kW)",
        }
    )


_PRICE_BAND_ALPHA = 0.15
"""Half the shared band opacity (``style.BAND_ALPHA``, 0.3) for the price band
only (polish plan G9): it overlaps the smart charging band on this dual-axis
chart, and at full opacity the two fills mixed into a colour neither series
has."""


def _publication_times(result: Any, prices: pd.DataFrame) -> list[pd.Timestamp]:
    """Day-ahead publication instants inside the chart's range, from the result.

    Each day's prices are published at 13:00 London the day before (plan B4);
    the result carries that instant per slot, so the view only picks the
    distinct ones it can draw.
    """

    start, end = prices["interval_start_utc"].iat[0], prices["interval_end_utc"].iat[-1]
    published = pd.Series(result.forecast_prices["forecast_available_at_utc"].unique())
    return sorted(published.loc[(published >= start) & (published < end)])


def _price_figure(
    prices: pd.DataFrame,
    charging: pd.DataFrame,
    publications: list[pd.Timestamp],
    shocks: pd.DataFrame | None = None,
) -> go.Figure:
    figure = go.Figure()
    if shocks is not None and not shocks.empty:
        add_shock_marks(figure, shocks)
    # P10-P90 across simulated weeks is shown on every time-series chart
    # (decision 0004 item 40); the charging band doubles as the "shading/area
    # showing when smart charging actually charged" (item 43), because both
    # bounds sit at 0 kW outside a charging window. Charging is added first
    # so the price line draws on top. One legend entry per series (polish
    # plan G3): the caption names the spread.
    band_and_line(
        figure,
        charging["interval_start_utc"],
        charging["p10"],
        charging["p50"],
        charging["p90"],
        name="Smart home charging (right axis)",
        colour=SERIES_COLOURS["selected"],
        legendgroup="charging",
        hovertemplate="%{y:,.0f} kW<extra></extra>",
        yaxis="y2",
    )
    band_and_line(
        figure,
        prices["interval_start_utc"],
        prices["p10"],
        prices["p50"],
        prices["p90"],
        name="Synthetic day-ahead price",
        colour=SERIES_COLOURS["synthetic"],
        band_alpha=_PRICE_BAND_ALPHA,
        legendgroup="price",
        hovertemplate="%{y:,.0f} £/MWh<extra></extra>",
    )
    for index, published in enumerate(publications):
        # One dotted line per day-ahead publication (13:00 London, next
        # day's prices, plan B4); only the first is labelled.  "top right"
        # anchors the label's left edge, so it grows into the plot rather
        # than off it (milestone verifier, decision 0004 item 26).
        label = (
            {
                "annotation_text": "Next day's prices published",
                "annotation_position": "top right",
                "annotation_font_color": MUTED_INK,
            }
            if index == 0
            else {}
        )
        figure.add_vline(x=published, line_dash="dot", line_color=MUTED_INK, **label)
    figure.update_xaxes(
        tickmode="array",
        range=[prices["interval_start_utc"].iat[0], prices["interval_end_utc"].iat[-1]],
        # compact=True (goal review V3): this dual-axis chart's second
        # right-hand axis narrows the plot area enough that the default
        # "Mon 28"-style label ran adjacent days into each other at 390 px
        # ("Tue 29Wed 30"); a single week never repeats a weekday, so
        # dropping the day-of-month loses no information.
        **london_time_axis(prices["interval_start_utc"], compact=True),
    )
    figure.update_yaxes(title="£/MWh", rangemode="tozero")
    # A second, independent y-axis for the charging-power shading: sharing
    # the price axis would either swamp the price line's shape (kW and
    # £/MWh are unrelated scales) or clip the price at the power axis's
    # range, so the two series get their own axes (decision 0004 item 43).
    # Round ticks from zero on the power axis (final critique O-3: Plotly
    # placed 450 / 1,250 / 2,050 kW), from the band's own top.
    top = float(charging["p90"].max()) if len(charging) else 0.0
    figure.update_layout(
        yaxis2={
            "title": "Smart home import (kW)",
            "overlaying": "y",
            "side": "right",
            "rangemode": "tozero",
            "showgrid": False,
            "tick0": 0,
            "dtick": _nice_step(top),
        }
    )
    return figure


def _nice_step(top: float, ticks: int = 4) -> float:
    """A 1, 2 or 5 × 10^n tick step giving about ``ticks`` intervals up to ``top``."""

    if not top > 0:
        return 1.0
    raw = top / ticks
    magnitude = 10.0 ** np.floor(np.log10(raw))
    return float(next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw))


_BAND_LABELS = {"low": "Cheap", "middle": "Mid", "high": "Expensive"}
_BAND_ORDER = ("low", "middle", "high")


def _band_totals(bands: pd.DataFrame) -> pd.DataFrame:
    """One row per price band, the week's net shift smart minus unmanaged (item 43).

    ``price_band_shift``'s ``mean`` column is normal minus selected, summed
    over the band's half-hours on one date (contract 4.6); it is additive
    across the week's seven dates whatever the correlation between days (the
    mean of a sum equals the sum of the means), so summing it here never sums
    a p10/p50/p90 quantile (decision 0004 item 12). Negating that weekly sum
    gives smart minus normal: positive means smart charged more in that band
    than normal that week; negative means normal charged more. Used only for
    the net-shift line in the absolute chart's caption below
    (``_band_absolute_figure``): the contract's ``normal_kwh_*``/
    ``selected_kwh_*`` columns (item 46) are each path's own per-day p10/p50/p90,
    which cannot be summed into a week total the same safe way (only the mean
    is additive across days), so the chart itself stays at day granularity.
    """

    normal_minus_smart = (
        bands.groupby("price_band")["mean"].sum().reindex(_BAND_ORDER, fill_value=0.0)
    )
    return pd.DataFrame(
        {
            "price_band": _BAND_ORDER,
            "label": [_BAND_LABELS[band] for band in _BAND_ORDER],
            "smart_minus_normal_kwh_per_week": (-normal_minus_smart).to_numpy(),
        }
    )


_BAND_PANEL_GAP_PX = 36
_BAND_FIGURE_HEIGHT = (
    len(_BAND_ORDER) * CHART_HEIGHTS["small_multiple_panel"]
    + (len(_BAND_ORDER) - 1) * _BAND_PANEL_GAP_PX
)
"""3 panels (one per band) stacked, each at least the small-multiple minimum
(decision 0004 item 26): 3 x 220 + 2 x 36 = 732 px. Stacked rather than a
3-column grid (the pattern ``archetypes.py`` uses) so all seven day labels on
the x-axis stay full width and legible at 390 px, not squeezed into a third
of the container."""


def _price_band_paths(bands: pd.DataFrame) -> list[str]:
    """Paths whose own energy columns (``{path}_kwh_p50`` etc.) this ``price_band_shift`` frame
    carries, in on-screen display order.

    ``normal`` and ``selected`` always have them (contract 4.6); the optional ``timed`` path's
    ``timed_kwh_*`` columns join them only on a run whose ``timed_start_local_hour`` is set
    (decision 0007, model step 2). Checked by one representative column per path rather than
    assuming a fixed pair, so the bars and table below draw two paths off and three on with no
    further branching at the call site.
    """

    present = {path for path in PATH_DISPLAY_ORDER if f"{path}_kwh_p50" in bands.columns}
    return present_paths(present)


def _band_absolute_table(bands: pd.DataFrame) -> pd.DataFrame:
    """One row per (day, band): each path's own home import, P10/P50/P90 (kWh)."""

    table = pd.DataFrame(
        {
            "Day": bands["day_label"],
            "Forecast price band": bands["price_band"].map(_BAND_LABELS),
        }
    )
    for path in _price_band_paths(bands):
        prefix = PATH_LABELS[path]
        for stat, stat_label in (("p10", "P10"), ("p50", "P50"), ("p90", "P90")):
            table[f"{prefix} {stat_label} (kWh)"] = bands[f"{path}_kwh_{stat}"].round(1)
    return table


def _band_absolute_figure(bands: pd.DataFrame) -> go.Figure:
    """Each path's own home import per band and day (contract 4.6, decision 0004 item 46).

    One panel per band, P50 bars with P10-P90 error bars, grouped normal vs
    smart (goal review action 2, decision 0004 item 20), plus the optional
    timed path's own bar (decision 0007, model step 2) when the run's
    ``price_band_shift`` carries it (``_price_band_paths``). Absolute bars, not a
    signed smart-minus-normal shift: absolute bars show what each path
    actually drew, which a signed shift cannot (a small shift can hide two
    large, nearly cancelling bars). Kept at one bar per study day rather than
    summed into a week total, because only ``price_band_shift``'s ``mean`` column is
    additive across days; its p10/p50/p90 are each a per-day quantile across
    simulated weeks, and summing quantiles across days would sum percentiles
    (AGENTS.md; see ``_band_totals`` for the one column that safely does add).
    """

    paths = _price_band_paths(bands)
    day_order = list(dict.fromkeys(bands["day_label"]))
    figure = make_subplots(
        rows=len(_BAND_ORDER),
        cols=1,
        shared_xaxes=True,
        # One y scale for all three panels (polish plan G9): the point is to
        # compare how much energy lands in Cheap vs Expensive, which
        # independent autoscaled axes hid by stretching each panel to full height.
        shared_yaxes="all",
        subplot_titles=[_BAND_LABELS[band] for band in _BAND_ORDER],
        vertical_spacing=_BAND_PANEL_GAP_PX / _BAND_FIGURE_HEIGHT,
    )
    figure.update_annotations(font_size=12)
    figure.update_layout(margin={"t": 32}, barmode="group")
    for row_index, band_id in enumerate(_BAND_ORDER, start=1):
        rows = bands.loc[bands["price_band"].eq(band_id)]
        for path in paths:
            label = PATH_LABELS[path]
            p50 = rows[f"{path}_kwh_p50"].to_numpy(dtype=float)
            p10 = rows[f"{path}_kwh_p10"].to_numpy(dtype=float)
            p90 = rows[f"{path}_kwh_p90"].to_numpy(dtype=float)
            figure.add_trace(
                go.Bar(
                    x=rows["day_label"],
                    y=p50,
                    name=label,
                    legendgroup=path,
                    showlegend=row_index == 1,
                    marker_color=PATH_STYLES[path]["color"],
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": p90 - p50,
                        "arrayminus": p50 - p10,
                        "color": MUTED_INK,
                    },
                    hovertemplate=f"{label}: %{{y:,.1f}} kWh<extra></extra>",
                ),
                row=row_index,
                col=1,
            )
    figure.update_xaxes(categoryorder="array", categoryarray=day_order)
    figure.update_yaxes(title="kWh per day", rangemode="tozero")
    return figure


def _plan_kpi_row(st: Any, smart_summary: pd.DataFrame) -> None:
    """KPI tiles from ``smart_charging_summary`` (contract 4.5, decision 0004 items 12, 43).

    The headline tile is ``average_price_change_gbp_per_mwh``: computed
    inside each simulated week first, then summarised (world-first, item
    12), never the difference of the two medians shown as context beneath
    it. Share of energy moved gets its own tile; it is not a price.

    With the optional timed path present (decision 0007, model step 2), its own price paid
    (``timed_average_price_gbp_per_mwh``) joins the two medians already in the headline tile's
    context, beside smart's rather than instead of it. It carries no "change" reading of its
    own (that reading stays selected minus normal, contract 4.5), so it only ever adds a third
    figure to the context line and tooltip, never a second headline number.
    """

    rows = smart_summary.set_index("metric")
    change = rows.loc["average_price_change_gbp_per_mwh"]
    normal_price = rows.loc["normal_average_price_gbp_per_mwh", "p50"]
    smart_price = rows.loc["selected_average_price_gbp_per_mwh", "p50"]
    share = rows.loc["moved_home_import_share"]

    headline, moved = kpi_columns(st, 2)
    low = format_quantity(change["p10"], "£/MWh", decimals=0)
    high = format_quantity(change["p90"], "£/MWh", decimals=0)
    # The two paths' own medians are context under the value, not a
    # separate caption (polish plan G5).
    context = (
        f"Unmanaged median {format_quantity(normal_price, '£/MWh', decimals=0)} · "
        f"smart {format_quantity(smart_price, '£/MWh', decimals=0)}"
    )
    help_text = (
        "The average price paid for home charging, smart minus unmanaged. The difference "
        "is worked out inside each simulated week first, then summarised across weeks, so it "
        "is never the gap between the two medians shown underneath. "
        f"P10–P90 across {int(change['world_count'])} simulated weeks: {low} to {high}."
    )
    if "timed_average_price_gbp_per_mwh" in rows.index:
        timed_price = rows.loc["timed_average_price_gbp_per_mwh", "p50"]
        context += f" · timed tariff {format_quantity(timed_price, '£/MWh', decimals=0)}"
        help_text += f" {_TIMED_COSTING_NOTE}"
    kpi(
        headline,
        "Price change vs unmanaged",
        format_quantity(change["p50"], "", decimals=0),
        unit="£/MWh",
        context=context,
        help=help_text,
    )
    moved_low = percent(100 * share["p10"])
    moved_high = percent(100 * share["p90"])
    kpi(
        moved,
        "Share of home charging moved",
        percent(100 * share["p50"]),
        help=f"P10–P90 across {int(share['world_count'])} simulated weeks: "
        f"{moved_low} to {moved_high}.",
    )


def _signed_whole(value: float) -> str:
    """A signed whole number, "+12" or "−8" (U+2212, polish plan G6); the caller adds the unit."""

    text = format_quantity(value, "", decimals=0)
    return f"+{text}" if round(value) > 0 else text


def _finding_text(totals: pd.DataFrame) -> str:
    """The lens's one "Finding:" as a sentence, not a list of means (final critique O-8).

    ``totals`` is ``_band_totals``: the mean weekly shift, smart minus
    unmanaged, per price band. When smart charging takes energy out of the
    dearest third and adds it to the cheapest, the sentence says so; any
    other pattern falls back to the three signed figures.
    """

    shift = totals.set_index("price_band")["smart_minus_normal_kwh_per_week"]
    cheap, dear = float(shift["low"]), float(shift["high"])
    if dear < 0 < cheap:
        moved = format_quantity(-dear, "kWh", decimals=0)
        gained = format_quantity(cheap, "kWh", decimals=0)
        return (
            f"Finding: in a mean week smart charging moves {moved} out of the dearest third "
            f"of half-hours; the cheapest third gains {gained}."
        )
    listed = ", ".join(
        f"{row.label} {_signed_whole(row.smart_minus_normal_kwh_per_week)}"
        for row in totals.itertuples()
    )
    return f"Finding: mean weekly shift, smart minus unmanaged: {listed} kWh."


def render_action_decision(st: Any, result: Any) -> None:
    """Render the Plan lens: summary sentence, the forecast and plan, and the KPI tiles."""

    # pages.py filters no-action-model results before calling this view
    # (design 4.3: "This result has no Axle action." plus a Switch button).
    # This guard only keeps a direct call (a test, a future caller) safe.
    if result.action_summary is None:
        st.info("This result has no Axle action.")
        return

    summary = result.action_summary
    st.markdown(_summary_sentence(summary))
    evidence = f"Evidence: {_evidence_text(result.evidence_kind)}."

    # KPIs first (polish plan G9): the answer before the charts that explain it.
    _plan_kpi_row(st, result.smart_charging_summary)
    if result.smart_charging_summary["metric"].eq("timed_average_price_gbp_per_mwh").any():
        st.caption(_TIMED_COSTING_NOTE)
    # The net weekly shift (mean, additive across days -- see _band_totals) is
    # the lens's one finding, stated once under the tiles rather than buried
    # in a chart caption (polish plan G7).
    totals = _band_totals(result.price_band_shift)
    st.caption(
        _finding_text(totals),
        help="Mean net shift per week, smart minus unmanaged: "
        + ", ".join(
            f"{row.label} {_signed_whole(row.smart_minus_normal_kwh_per_week)}"
            for row in totals.itertuples()
        )
        + " kWh. Means add up across days, so a week total is safe. Percentiles do not add, "
        "so they are never summed.",
    )

    charging = _selected_home_import(result.fleet_interval_bands)
    prices = _price_bands(result)
    shocks = scripted_shock_windows(result)
    chart_block(
        st,
        _price_figure(prices, charging, _publication_times(result, prices), shocks),
        title="Day-ahead price and smart home charging through the week",
        caption=(
            "Median and P10–P90 across simulated weeks: price in £/MWh (left), smart home "
            f"charging in kW (right). {evidence}"
        ),
        frame=_price_table(prices, charging),
        definition=(
            "The prices are illustrative and synthetic, not market data: a daily curve plus "
            "random day-ahead noise. Each simulated week gets its own day-ahead path, so the "
            "cheapest half-hour moves from day to day. Each week's charger plans on that "
            "week's path. The price the fleet actually pays adds a forecast error the plan "
            "never saw. The dotted lines mark 13:00 each day, when the next day's day-ahead "
            "prices come out. A plan only ranks prices already published, because that is all "
            "a real charger could know. The shaded band is the smart path's fleet home import: "
            "the median line with P10–P90 across simulated weeks. Scripted price shocks chosen "
            "under Edit assumptions, Events, are shaded because they are the same in every "
            "simulated week. A known shock is in the day-ahead price and in the plans. A "
            "surprise moves only the intraday and imbalance prices, so the day-ahead band does "
            "not show it. Random background shocks differ between weeks and are not marked."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="action-decision-price",
    )
    if not shocks.empty:
        with st.expander(f"Scripted price shocks ({len(shocks)})"):
            st.dataframe(shock_window_table(shocks), width="stretch", hide_index=True)
            st.caption(
                "Shaded on the chart: filled if known a day ahead, dashed if a surprise. "
                "Illustrative scenarios, the same in every simulated week."
            )

    # Title names every path the frame actually carries, in display order
    # (decision 0007): "unmanaged vs smart" off, "unmanaged vs timed tariff
    # vs smart" on, matching the bars and the table below.
    band_policy_phrase = " vs ".join(
        PATH_LABELS[path].lower() for path in _price_band_paths(result.price_band_shift)
    )
    chart_block(
        st,
        _band_absolute_figure(result.price_band_shift),
        title=f"Home energy by forecast-price band, {band_policy_phrase} (kWh per day)",
        caption=(
            "P50 bars, P10–P90 error bars across simulated weeks; one bar per session night, "
            f"noon to noon. {evidence}"
        ),
        frame=_band_absolute_table(result.price_band_shift),
        definition=(
            "Each simulated week's 336 half-hours are split into thirds by that week's own "
            "day-ahead price: cheapest, middle and most expensive. The bars are absolute, not "
            "a smart-minus-unmanaged shift: each path's own home import in that band and day "
            "shows what was actually drawn, so a small net shift is not mistaken for no "
            "activity. The chart stays per study day rather than summed over the week, "
            "because only the mean adds up across days. P10, P50 and P90 are each a per-day "
            "quantile, and adding them across days would sum percentiles. Each bar is one "
            "session night, noon to noon, so a night's evening peak and the overnight "
            "half-hours it moved to sit in the same bar."
        ),
        height=_BAND_FIGURE_HEIGHT,
        key="action-decision-bands",
    )
    st.caption(
        "Cheap, Mid, Expensive: each week's own day-ahead prices split into thirds. Whiskers "
        "are wide because those thirds move week to week."
    )
