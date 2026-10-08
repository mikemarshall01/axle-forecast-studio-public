"""Supplier ▸ 4 Firm MW: how much turn-down (or turn-up) can be promised firmly.

Reads (trading contract v1 §10.10): ``availability_bands`` (the day-ahead
across-week band and the intraday 17:00 conditional lines),
``availability_world_slot`` of the representative simulated week (this
week's deliverable and the whole-night intraday distribution),
``firm_share_by_fleet_size`` and ``manufacturer_summary`` (diversification),
``product_sheet`` and ``availability_value_summary`` (the product),
``availability_reliability`` and ``availability_backtest_summary``
(calibration), ``firmness_by_manufacturer`` and ``charge_completion_summary``
(firmness and completion), the ``settlement_file`` download, and
``blackout_windows``.

"Firm" here always means the across-weeks P10: ``numpy.quantile(..., 0.1)``,
the value exceeded in 90% of simulated weeks (a trader's "P90"; §10's
two-convention note). A day-ahead promise sized to the worst week in ten,
never the physical ceiling. Every kW, MW, MWh and £ is illustrative and
synthetic: nothing here is a bid, a settlement or Axle cash. This view reads
stored frames only; no quantile, ratio, position or money is computed here.

Units (Q-12 of the final critique): the day-ahead and intraday bands read
raw per-half-hour kW (the fleet peaks well under 10 MW); the product sheet,
its headline capacity tiles and the manufacturer table are the MW-branded
"Firm MW" product itself, a window-level statistic, and stay MW.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from streamlit import column_config as st_column_config

from .. import usage_log
from ..components import header
from ..components.chart_table import chart_block
from ..components.kpi import kpi, kpi_columns
from ..style import (
    ARCHETYPE_COLOURS,
    CHART_HEIGHTS,
    MUTED_INK,
    PATH_LABELS,
    SERIES_COLOURS,
    UNAVAILABLE,
    band_and_line,
    format_quantity,
    hover_time_labels,
    london_time_axis,
    money,
    percent,
    present_paths,
)
from .supplier_common import evidence_caption, group_label, missing, stat_row

_DIRECTION_LABELS = {"turn_down": "Turn-down", "turn_up": "Turn-up"}
_DURATION_LABELS = {0.5: "0.5 h", 1.0: "1 h", 2.0: "2 h", 4.0: "4 h"}
_WINDOW_LABELS = {"evening": "Evening", "overnight": "Overnight", "morning": "Morning"}
_HEADLINE_WINDOW = "evening"
_HEADLINE_DURATION = 1.0
"""The product sheet's headline reference for the day-ahead tiles (§10.10): the
evening window at a 1 h hold, whichever direction the controls have chosen."""


def _controls(st: Any) -> tuple[str, float]:
    """Direction and hold duration, shared by every section on this lens (§10.10)."""

    direction_column, duration_column = header.controls(st).columns([1, 1])
    with direction_column:
        direction = st.segmented_control(
            "Direction",
            options=list(_DIRECTION_LABELS),
            format_func=_DIRECTION_LABELS.__getitem__,
            default="turn_down",
            required=True,
            key="firm-mw-direction",
            label_visibility="collapsed",
        )
    with duration_column:
        duration = st.segmented_control(
            "Hold",
            options=list(_DURATION_LABELS),
            format_func=_DURATION_LABELS.__getitem__,
            default=1.0,
            required=True,
            key="firm-mw-duration",
            label_visibility="collapsed",
        )
    direction = direction if direction in _DIRECTION_LABELS else "turn_down"
    duration = duration if duration in _DURATION_LABELS else 1.0
    return direction, duration


# --------------------------------------------------------------------------
# Day-ahead: the band across simulated weeks (§10.2b, §10.2e)
# --------------------------------------------------------------------------


def _day_ahead_rows(bands: pd.DataFrame, *, direction: str, duration: float) -> pd.DataFrame:
    return bands.loc[
        bands["horizon"].eq("day_ahead")
        & bands["statistic"].eq("realised")
        & bands["direction"].eq(direction)
        & bands["duration_hours"].eq(duration)
    ].sort_values("slot_index")


def _this_week_rows(
    world_slot: pd.DataFrame, *, world_id: int, direction: str, duration: float
) -> pd.DataFrame:
    return world_slot.loc[
        world_slot["world_id"].eq(world_id)
        & world_slot["direction"].eq(direction)
        & world_slot["duration_hours"].eq(duration)
    ].sort_values("slot_index")


def _day_ahead_figure(rows: pd.DataFrame, this_week: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    # kW (Q-12): this chart and its own table below read availability_bands
    # and availability_world_slot, which store kW; only the product sheet
    # and its headline tiles (a different, window-level frame) stay MW.
    # The house band is P10-P90 (decision 0004 item 27): its own lower edge
    # is the firm figure, so no separate "Firm" line is drawn on top of it;
    # the P05 reading stays in the table only.
    times = hover_time_labels(rows["interval_start_utc"])
    # sd in hover (§10.10): the P50 line's own hover carries the across-weeks
    # spread, so the number the band shows visually also reads exactly in text.
    customdata = list(zip(times, rows["sd"].to_numpy(dtype=float), strict=True))
    band_and_line(
        figure,
        rows["interval_start_utc"],
        rows["p10"].to_numpy(dtype=float),
        rows["p50"].to_numpy(dtype=float),
        rows["p90"].to_numpy(dtype=float),
        name="Smart path, P50",
        colour=SERIES_COLOURS["selected"],
        customdata=customdata,
        hovertemplate="%{customdata[0]}: %{y:,.0f} kW (P50), sd %{customdata[1]:,.0f} kW"
        "<extra></extra>",
        band_hovertemplates=(
            "%{customdata[0]}: %{y:,.0f} kW, firm (P10)<extra></extra>",
            "%{customdata[0]}: %{y:,.0f} kW (P90)<extra></extra>",
        ),
    )
    if len(this_week):
        figure.add_trace(
            go.Scatter(
                x=this_week["interval_start_utc"],
                y=this_week["deliverable_kw"].to_numpy(dtype=float),
                mode="lines",
                name="This week",
                line={"color": MUTED_INK, "width": 1.5, "dash": "dot"},
                hovertemplate="This week: %{y:,.0f} kW<extra></extra>",
            )
        )
    if not rows.empty:
        figure.update_xaxes(tickmode="array", **london_time_axis(rows["interval_start_utc"]))
    figure.update_yaxes(title="kW", rangemode="tozero")
    return figure


def _day_ahead_table(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[
        :,
        [
            "interval_start_london",
            "interval_start_utc",
            "mean",
            "sd",
            "p05",
            "p10",
            "p50",
            "p90",
            "firm_share",
            "n_eff",
        ],
    ].rename(
        columns={
            "interval_start_london": "Start (London)",
            "interval_start_utc": "Start (UTC)",
            "mean": "Mean (kW)",
            "sd": "SD across weeks (kW)",
            "p05": "P05 (kW)",
            "p10": "P10, firm (kW)",
            "p50": "P50 (kW)",
            "p90": "P90 (kW)",
            "firm_share": "Firm share (P10÷P50)",
            "n_eff": "Effective independent EVs",
        }
    )


def _render_day_ahead(st: Any, result: Any, *, direction: str, duration: float) -> None:
    bands = result.availability_bands
    world_slot = result.availability_world_slot
    weeks = result.world_count
    evidence = evidence_caption(result)

    sheet_row = stat_row(
        result.product_sheet,
        window=_HEADLINE_WINDOW,
        direction=direction,
        duration_hours=_HEADLINE_DURATION,
    )
    columns = kpi_columns(st, 4)
    kpi(
        columns[0],
        "Firm evening capacity",
        UNAVAILABLE
        if sheet_row is None
        else format_quantity(sheet_row["window_mean_mw_p10"], "MW", decimals=2),
        # "Small because evenings are already moved" is a turn-down-only
        # claim (smart charging empties the evening of turn-down, not
        # turn-up): only show it for that direction (Fable review B1).
        context="1 h hold, P10: the worst week in ten"
        + ("; small because evenings are already moved" if direction == "turn_down" else ""),
    )
    kpi(
        columns[1],
        "Typical evening capacity",
        UNAVAILABLE
        if sheet_row is None
        else format_quantity(sheet_row["window_mean_mw_p50"], "MW", decimals=2),
        context="1 h hold, P50: the typical week",
    )
    kpi(
        columns[2],
        "Firm share, P10÷P50",
        UNAVAILABLE
        if sheet_row is None or missing(sheet_row["firm_share"])
        else percent(100 * sheet_row["firm_share"]),
        context="How much of the typical week survives to the firm figure",
    )
    kpi(
        columns[3],
        "Firm share, P05÷P50",
        UNAVAILABLE
        if sheet_row is None or missing(sheet_row["firm_share_p05"])
        else percent(100 * sheet_row["firm_share_p05"]),
        context="A stricter, 1-in-20 read of the same firm share",
    )

    rows = _day_ahead_rows(bands, direction=direction, duration=duration)
    this_week = _this_week_rows(
        world_slot, world_id=result.representative_world_id, direction=direction, duration=duration
    )
    chart_block(
        st,
        _day_ahead_figure(rows, this_week),
        title=f"{_DIRECTION_LABELS[direction]} deliverable kW, day-ahead, "
        f"{_DURATION_LABELS[duration]} hold",
        # "Firm level", not "firm line": no separate line is drawn for it,
        # only the P10-P90 band's own lower edge (_day_ahead_figure's own
        # comment). A caption naming a line that is not on the chart reads
        # as a bug report waiting to happen.
        caption=f"P10–P90 across {weeks} simulated weeks; 1 week in 10 falls below the firm "
        f"level. {evidence}",
        frame=_day_ahead_table(rows),
        definition=(
            "Deliverable kW is the turn-down (or turn-up) the smart-charged fleet could hold "
            "for this long, starting this half-hour. Firm is the level met in 9 weeks out of "
            "10: the P10 across weeks (90% exceedance), never the physical charger ceiling. "
            "Effective independent EVs: how many uncorrelated EVs the fleet behaves like once "
            "shared weather and prices are allowed for."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="firm-mw-day-ahead",
    )
    if direction == "turn_down":
        # Clarity critique top-15 #3: turn-down beyond what smart charging
        # already moved is small in most half-hours (the evening is already
        # emptied), so the P50 line sits on 0 kW and the band is spikes where
        # a minority of weeks still had something to give: the same
        # zero-median pattern as Drivers ▸ Household §0.1, one level up (the
        # whole fleet). Turn-up has no such pattern (Fable review B1).
        st.caption(
            "Median week near 0 kW: smart charging has already moved the evening, so extra "
            "turn-down exists only where plans are still charging.",
            help=(
                "Turn-down here is beyond the smart schedule, not against unmanaged charging. "
                "In most half-hours fewer than half the weeks have any turn-down, so the P50 "
                "(and the firm P10, the band's lower edge) is 0. The band shows the weeks that "
                "did have some. So the firm product lives overnight (see the Product sheet)."
            ),
        )


# --------------------------------------------------------------------------
# Intraday at 17:00: the whole night from one decision (§10.2d)
# --------------------------------------------------------------------------


def _intraday_rows(
    world_slot: pd.DataFrame, *, world_id: int, night: int, direction: str, duration: float
) -> pd.DataFrame:
    rows = world_slot.loc[
        world_slot["world_id"].eq(world_id)
        & world_slot["night_index"].eq(night)
        & world_slot["direction"].eq(direction)
        & world_slot["duration_hours"].eq(duration)
    ].sort_values("slot_index")
    # Slots before the night's decision instant carry no intraday forecast
    # (§10.2d): the chart starts at s_n without computing it directly.
    return rows.loc[rows["intraday_mean_kw"].notna()]


def _intraday_figure(rows: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    # kW (Q-12): matches this chart's own already-kW table below. The house
    # band is P10-P90, as the day-ahead chart above (decision 0004 item 27);
    # the wider P05/P95 levels this frame also carries stay in the table only.
    band_and_line(
        figure,
        rows["interval_start_utc"],
        rows["intraday_p10_kw"].to_numpy(dtype=float),
        rows["intraday_p50_kw"].to_numpy(dtype=float),
        rows["intraday_p90_kw"].to_numpy(dtype=float),
        name="Smart path, whole night P50",
        colour=SERIES_COLOURS["selected"],
        customdata=hover_time_labels(rows["interval_start_utc"]),
        hovertemplate="%{customdata}: %{y:,.0f} kW (P50)<extra></extra>",
        band_hovertemplates=(
            "%{customdata}: %{y:,.0f} kW (P10)<extra></extra>",
            "%{customdata}: %{y:,.0f} kW (P90)<extra></extra>",
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=rows["interval_start_utc"],
            y=rows["intraday_known_p50_kw"].to_numpy(dtype=float),
            mode="lines",
            name="On the driveway at 17:00",
            line={"color": SERIES_COLOURS["difference"], "width": 2, "dash": "dash"},
            hovertemplate="On the driveway: %{y:,.0f} kW<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=rows["interval_start_utc"],
            y=rows["deliverable_kw"].to_numpy(dtype=float),
            mode="lines",
            name="Realised",
            line={"color": MUTED_INK, "width": 1.5, "dash": "dot"},
            hovertemplate="Realised: %{y:,.0f} kW<extra></extra>",
        )
    )
    if not rows.empty:
        figure.update_xaxes(
            tickmode="array", **london_time_axis(rows["interval_start_utc"], every_hours=1)
        )
    figure.update_yaxes(title="kW", rangemode="tozero")
    return figure


def _intraday_table(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[
        :,
        [
            "interval_start_london",
            "interval_start_utc",
            "intraday_mean_kw",
            "intraday_p05_kw",
            "intraday_p50_kw",
            "intraday_p95_kw",
            "intraday_known_p50_kw",
            "deliverable_kw",
            "deliverable_known_kw",
        ],
    ].rename(
        columns={
            "interval_start_london": "Start (London)",
            "interval_start_utc": "Start (UTC)",
            "intraday_mean_kw": "Forecast mean (kW)",
            "intraday_p05_kw": "Forecast P05 (kW)",
            "intraday_p50_kw": "Forecast P50 (kW)",
            "intraday_p95_kw": "Forecast P95 (kW)",
            "intraday_known_p50_kw": "On the driveway, P50 (kW)",
            "deliverable_kw": "Realised (kW)",
            "deliverable_known_kw": "Realised, known part (kW)",
        }
    )


def _render_intraday(st: Any, result: Any, *, direction: str, duration: float) -> None:
    world_slot = result.availability_world_slot
    world_nights = result.world_nights
    weeks = result.world_count
    night_options = sorted(world_nights["night_index"].unique().tolist())
    night = st.selectbox(
        "Night",
        options=night_options,
        format_func=lambda n: f"Night {n}",
        key="firm-mw-intraday-night",
        help="One study night; the intraday forecast is made at the decision time below.",
    )
    night = night if night in night_options else night_options[0]

    rows = _intraday_rows(
        world_slot,
        world_id=result.representative_world_id,
        night=night,
        direction=direction,
        duration=duration,
    )
    decision_row = stat_row(
        world_nights, world_id=result.representative_world_id, night_index=night
    )
    plugged_share = (
        UNAVAILABLE
        if decision_row is None or missing(decision_row["plugged_in_share_at_decision"])
        else percent(100 * decision_row["plugged_in_share_at_decision"])
    )
    if rows.empty:
        st.info(
            f"Night {night} has no intraday forecast for this simulated week: with one simulated "
            "week there is no other week to read the late part from."
        )
        return

    # ``availability_bands`` has one row per (statistic, slot); pin the
    # decision slot itself (``rows``' first row, already filtered to
    # slots at or after s_n) so "typical" reads the same instant the chart
    # shows, not an arbitrary earlier slot stat_row would otherwise pick.
    decision_slot = int(rows.iloc[0]["slot_index"])
    typical = stat_row(
        result.availability_bands,
        horizon="intraday",
        statistic="conditional_p10",
        direction=direction,
        duration_hours=duration,
        slot_index=decision_slot,
    )
    typical_known = stat_row(
        result.availability_bands,
        horizon="intraday",
        statistic="conditional_known_p50",
        direction=direction,
        duration_hours=duration,
        slot_index=decision_slot,
    )

    chart_block(
        st,
        _intraday_figure(rows),
        title=f"{_DIRECTION_LABELS[direction]} kW, whole night, {_DURATION_LABELS[duration]} hold "
        f"(night {night})",
        # Names the band's own spread (clarity critique: unnamed elsewhere, it
        # is the 17:00 forecast's own distribution for this one night, not
        # the day-ahead chart's across-weeks band above).
        caption=f"Band: the 17:00 forecast's own P10–P90 for this night; {plugged_share} of the "
        f"fleet was on the driveway. {evidence_caption(result)}",
        frame=_intraday_table(rows),
        definition=(
            "Made at the 17:00 decision time. The known part is the EVs already plugged in, "
            "with random response and maker outages applied. The late part is the sessions "
            "not yet plugged in, read from what the other simulated weeks did."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="firm-mw-intraday",
    )
    st.caption(
        f"Dashed: {_DIRECTION_LABELS[direction].lower()} from EVs already plugged in at 17:00 "
        "(the known part); dotted: what actually happened."
    )
    typical_text = (
        UNAVAILABLE
        if typical is None or missing(typical["p50"])
        else format_quantity(typical["p50"], "kW", decimals=0)
    )
    typical_known_text = (
        UNAVAILABLE
        if typical_known is None or missing(typical_known["p50"])
        else format_quantity(typical_known["p50"], "kW", decimals=0)
    )
    st.caption(
        f"Typical firm figure at 17:00, across {weeks} weeks: {typical_text}; "
        f"on the driveway then (known part, P50): {typical_known_text}."
    )


# --------------------------------------------------------------------------
# Diversification: shared factors and manufacturers (§10.2c, §10.5h, §10.6b)
# --------------------------------------------------------------------------


def _fleet_size_figure(rows: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for column, name, colour in (
        ("firm_share", "Firm share (P10÷P50)", ARCHETYPE_COLOURS[0]),
        ("firm_share_p05", "Firm share (P05÷P50)", ARCHETYPE_COLOURS[1]),
    ):
        figure.add_trace(
            go.Scatter(
                x=rows["fleet_size"],
                y=100 * rows[column].to_numpy(dtype=float),
                mode="lines+markers",
                name=name,
                line={"color": colour, "width": 2},
                customdata=rows["n_eff"],
                hovertemplate="%{x:,} EVs: %{y:.0f}%, effective EVs %{customdata:.0f}<extra>"
                + name
                + "</extra>",
            )
        )
    figure.update_xaxes(title="Sub-fleet size (EVs)", type="log")
    figure.update_yaxes(title="%", rangemode="tozero")
    return figure


def _render_diversification(st: Any, result: Any, *, direction: str, duration: float) -> None:
    # B-8: a chart of firm_share per half-hour read as empty. One slot's
    # worst-week-in-ten is mostly sampling noise (a handful of EVs charging
    # that instant, if any), not the shared factors the fleet-size chart
    # below is about, and the ratio collapses towards 0 almost everywhere
    # outside the few charging half-hours; plotting it gave a flat, near-
    # invisible line under a caption that asserted a finding it did not
    # show. Hidden with an honest message rather than fixed (lead ruling):
    # the fleet-size chart already carries the diversification finding.
    st.caption(
        "One half-hour's firm share is mostly sampling noise. Pooled by fleet size below, the "
        "floor set by shared factors is real."
    )

    window = st.segmented_control(
        "Window",
        options=list(_WINDOW_LABELS),
        format_func=_WINDOW_LABELS.__getitem__,
        default=_HEADLINE_WINDOW,
        required=True,
        key="firm-mw-fleet-size-window",
    )
    window = window if window in _WINDOW_LABELS else _HEADLINE_WINDOW
    sizes = result.firm_share_by_fleet_size
    rows = sizes.loc[
        sizes["window"].eq(window)
        & sizes["direction"].eq(direction)
        & sizes["duration_hours"].eq(duration)
    ].sort_values("fleet_size")
    chart_block(
        st,
        _fleet_size_figure(rows),
        title=f"Firm share against fleet size, {_WINDOW_LABELS[window].lower()}",
        caption="Flat beyond a few hundred EVs: shared factors, not sampling noise, set the firm "
        f"share. {evidence_caption(result)}",
        frame=rows.loc[
            :, ["fleet_size", "ev_count_share", "firm_share", "firm_share_p05", "n_eff"]
        ].rename(
            columns={
                "fleet_size": "Sub-fleet size (EVs)",
                "ev_count_share": "Share of this fleet",
                "firm_share": "Firm share (P10÷P50)",
                "firm_share_p05": "Firm share (P05÷P50)",
                "n_eff": "Effective independent EVs",
            }
        ),
        definition="The first n EVs in population order: a random sub-fleet under the same "
        "shared factors, prices and weather, read from this run alone with no extra "
        "simulation. Shared factors are what every EV sees at once (weather, prices, a maker "
        "outage), so adding EVs cannot average them away. Effective independent EVs: how many "
        "uncorrelated EVs the sub-fleet behaves like.",
        height=CHART_HEIGHTS["time_series"],
        key="firm-mw-fleet-size",
    )

    st.markdown(
        "**Manufacturer diversification**",
        help="More, smaller makers spread the risk: one maker's outage then takes out a "
        "smaller share of the fleet at once.",
    )
    manufacturers = result.manufacturer_summary
    st.dataframe(_manufacturer_summary_table(manufacturers), width="stretch", hide_index=True)
    fleet = stat_row(manufacturers, manufacturer_id="fleet")
    fleet_outage = (
        UNAVAILABLE
        if fleet is None or missing(fleet["outage_probability_per_night"])
        else percent(100 * fleet["outage_probability_per_night"])
    )
    st.caption(
        f"Chance any maker is out on a given night: {fleet_outage}. More, smaller makers "
        "diversify this risk, the same way more independent EVs do. Illustrative."
    )


def _manufacturer_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Maker": frame["manufacturer_label"],
            "Share": [percent(100 * v) if not missing(v) else UNAVAILABLE for v in frame["share"]],
            "EVs": frame["ev_count"],
            "Response rate": [
                UNAVAILABLE if missing(v) else percent(100 * v) for v in frame["response_rate"]
            ],
            "Outage chance, per night": [
                percent(100 * v) if not missing(v) else UNAVAILABLE
                for v in frame["outage_probability_per_night"]
            ],
            "Evening turn-down potential, P50 (MW)": [
                format_quantity(v, "MW", decimals=2) for v in frame["evening_potential_mw_p50"]
            ],
        }
    )


# --------------------------------------------------------------------------
# The product sheet and price-weighted value (§10.5a, §10.5c)
# --------------------------------------------------------------------------

_PRODUCT_SHEET_DISPLAY = {
    "window": "Window",
    "direction": "Direction",
    "duration_hours": "Hold (h)",
    "window_mean_mw_p10": "Mean MW, firm (P10)",
    "window_mean_mw_p50": "Mean MW (P50)",
    "window_min_mw_p50": "Min-through-window MW (P50)",
    "firm_share": "Firm share (P10÷P50)",
    "movable_energy_mwh_p50": "Movable energy, P50 (MWh per week)",
    "intraday_known_share_p50": "Known share at 17:00 (P50)",
    "max_event_length_hours": "Longest sensible event (h)",
    "notice_day_ahead_hours": "Day-ahead notice (h)",
    "notice_intraday_hours": "Intraday notice (h)",
    "ramp_hours": "Ramp (h)",
}


# Three decimals on every MW column, as Key stats shows the same firm figure:
# st.dataframe otherwise prints its raw float ("0.0743" beside "0").
_PRODUCT_SHEET_FORMATS = {
    _PRODUCT_SHEET_DISPLAY[name]: st_column_config.NumberColumn(format="%.3f")
    for name in ("window_mean_mw_p10", "window_mean_mw_p50", "window_min_mw_p50")
}


def _product_sheet_table(sheet: pd.DataFrame) -> pd.DataFrame:
    table = sheet.loc[:, list(_PRODUCT_SHEET_DISPLAY)].copy()
    table["window"] = table["window"].map(_WINDOW_LABELS)
    table["direction"] = table["direction"].map(_DIRECTION_LABELS)
    for column in ("firm_share", "intraday_known_share_p50"):
        table[column] = [UNAVAILABLE if missing(v) else percent(100 * v) for v in table[column]]
    # B-7: only the 0.5 h row sums a window's slots into MWh (product.py);
    # every other duration's cell is genuinely unavailable, not a zero
    # (decision 0003), so a raw NaN never reaches the screen as pandas'
    # bare "None".
    table["movable_energy_mwh_p50"] = [
        UNAVAILABLE if missing(v) else format_quantity(v, "MWh", decimals=1)
        for v in table["movable_energy_mwh_p50"]
    ]
    return table.rename(columns=_PRODUCT_SHEET_DISPLAY)


def _render_product_sheet(st: Any, result: Any, *, direction: str, duration: float) -> None:
    st.markdown(
        "**Product sheet**",
        help="The MW the fleet can hold in each product window (evening 17:00–21:00, overnight "
        "21:00–06:00, morning 06:00–12:00; 12:00–17:00 is in no window), how firm it is, the "
        "energy it can move and the notice each window gives.",
    )
    st.dataframe(
        _product_sheet_table(result.product_sheet),
        width="stretch",
        hide_index=True,
        column_config=_PRODUCT_SHEET_FORMATS,
    )
    st.caption(
        f"Quantile columns do not add across windows or durations. {evidence_caption(result)}"
    )
    st.caption(
        "Movable energy is given on each window's 0.5 h row only. Min-through-window is 0 "
        "when any half-hour in the window has 0 kW."
    )

    value = result.availability_value_summary
    weighted = stat_row(
        value, direction=direction, duration_hours=duration, metric="price_weighted_mw"
    )
    simple = stat_row(value, direction=direction, duration_hours=duration, metric="simple_mean_mw")
    weekly_value = stat_row(
        value,
        direction=direction,
        duration_hours=duration,
        metric="value_at_day_ahead_gbp_per_week",
    )
    columns = kpi_columns(st, 2)
    kpi(
        columns[0],
        "Price-weighted MW",
        UNAVAILABLE
        if weighted is None or missing(weighted["mean"])
        else format_quantity(weighted["mean"], "MW", decimals=2),
        context="Weighted toward the half-hours where this direction is worth most",
        help="Turn-down weighted by the day-ahead price it avoids; turn-up weighted only by "
        "negative prices. Simple mean for comparison: "
        f"{UNAVAILABLE if simple is None else format_quantity(simple['mean'], 'MW', decimals=2)}.",
    )
    kpi(
        columns[1],
        "Value at day-ahead prices",
        UNAVAILABLE if weekly_value is None else money(weekly_value["mean"]),
        context="Illustrative: every deliverable MWh at its own price",
        help="Every deliverable MWh valued at the day-ahead price it would avoid or need. Not "
        "a P&L, not a bid, and not comparable with the trading net: nothing here is sold.",
    )


# --------------------------------------------------------------------------
# Calibration: the leave-one-week-out backtest (§10.3)
# --------------------------------------------------------------------------

_HORIZON_LABELS = {
    "day_ahead": "Day-ahead",
    "intraday": "Intraday, whole night",
    "intraday_known": "Intraday, known part",
}


def _reliability_figure(reliability: pd.DataFrame, *, direction: str, duration: float) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfectly calibrated",
            line={"color": MUTED_INK, "width": 1, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    for index, horizon in enumerate(_HORIZON_LABELS):
        rows = reliability.loc[
            reliability["horizon"].eq(horizon)
            & reliability["direction"].eq(direction)
            & reliability["duration_hours"].eq(duration)
        ].sort_values("level")
        if rows.empty:
            continue
        colour = ARCHETYPE_COLOURS[index]
        # B-6: many half-hours forecast and realise exactly 0 kW, so a tie
        # counts as covered only in the weaker rate; showing "at or below"
        # alone reads as a badly over-covered forecast. Both rates, same
        # colour per horizon, solid vs dashed (uncertainty-viz rule 12: show
        # coverage, never alone; redundant coding over a sixth colour).
        for column, dash, qualifier in (
            ("coverage", "solid", "at or below"),
            ("coverage_strict", "dash", "strictly below"),
        ):
            name = f"{_HORIZON_LABELS[horizon]} ({qualifier})"
            figure.add_trace(
                go.Scatter(
                    x=rows["level"],
                    y=rows[column],
                    mode="lines+markers" if dash == "solid" else "lines",
                    name=name,
                    line={"color": colour, "width": 2, "dash": dash},
                    hovertemplate="Nominal %{x:.0%}: observed %{y:.0%}<extra>" + name + "</extra>",
                )
            )
    figure.update_xaxes(title="Nominal level", tickformat=".0%")
    figure.update_yaxes(title="Observed coverage (%)", tickformat=".0%", rangemode="tozero")
    return figure


def _backtest_half_hour_figure(
    backtest: pd.DataFrame, *, direction: str, duration: float
) -> go.Figure:
    rows = backtest.loc[
        backtest["horizon"].eq("day_ahead")
        & backtest["direction"].eq(direction)
        & backtest["duration_hours"].eq(duration)
        & backtest["level"].eq(0.10)
    ].sort_values("profile_order")
    figure = go.Figure()
    figure.add_hline(y=0.10, line={"color": MUTED_INK, "width": 1, "dash": "dot"})
    figure.add_trace(
        go.Scatter(
            x=rows["local_half_hour"],
            y=rows["coverage"],
            mode="lines",
            name="Coverage (at or below)",
            line={"color": ARCHETYPE_COLOURS[0], "width": 2},
            hovertemplate="%{x}: %{y:.0%}<extra></extra>",
        )
    )
    # Clarity critique HIGH item 7: the weak rate alone (a 0 kW forecast tied
    # by a 0 kW realised value counts as "covered") reads as a badly
    # over-covered forecast at 60-90% against a 10% target. The strict rate
    # was already in the table; plotting it too, same colour dashed (the
    # reliability chart's own solid/dash convention above), shows the target
    # is closer to met than the solid line alone implies.
    figure.add_trace(
        go.Scatter(
            x=rows["local_half_hour"],
            y=rows["coverage_strict"],
            mode="lines",
            name="Coverage (strictly below)",
            line={"color": ARCHETYPE_COLOURS[0], "width": 2, "dash": "dash"},
            hovertemplate="%{x}: %{y:.0%}<extra></extra>",
        )
    )
    figure.update_xaxes(title="Half-hour (London, noon to noon)", tickangle=-45, nticks=12)
    figure.update_yaxes(title="Coverage (%)", tickformat=".0%", rangemode="tozero")
    return figure


def _render_calibration(st: Any, result: Any, *, direction: str, duration: float) -> None:
    reliability = result.availability_reliability
    chart_block(
        st,
        _reliability_figure(reliability, direction=direction, duration=duration),
        title=f"Calibration: nominal vs observed coverage, {_DIRECTION_LABELS[direction].lower()}, "
        f"{_DURATION_LABELS[duration]} hold",
        caption="A calibrated forecast sits on the dotted line. Solid: ties at 0 kW count as "
        "covered; dashed: strictly below. Simulated, not telemetry.",
        frame=reliability.loc[
            reliability["direction"].eq(direction) & reliability["duration_hours"].eq(duration),
            ["horizon", "level", "coverage", "coverage_strict", "pinball_loss_kw", "sample_count"],
        ].rename(
            columns={
                "horizon": "Horizon",
                "level": "Level",
                "coverage": "Coverage (at or below)",
                "coverage_strict": "Coverage (strictly below)",
                "pinball_loss_kw": "Pinball loss (kW)",
                "sample_count": "Week-slots kept",
            }
        ),
        definition="Coverage at a level is the share of kept week-slots (one simulated week, "
        "one half-hour) where what happened was at or below the forecast at that level. A "
        "calibrated forecast has coverage equal to its level. Many half-hours are exactly "
        "0 kW, so the strict rate (below, not at or below) is shown beside it rather than "
        "expected to match exactly. Pinball loss scores the quantile forecast: lower is better.",
        height=CHART_HEIGHTS["time_series"],
        key="firm-mw-reliability",
    )

    backtest = result.availability_backtest
    chart_block(
        st,
        _backtest_half_hour_figure(backtest, direction=direction, duration=duration),
        title="Day-ahead coverage by half-hour, P10 level",
        caption="Dotted: the 10% target. Above it because 0 kW forecast and 0 kW realised "
        "half-hours count as covered; the strict (dashed) rate is lower.",
        frame=backtest.loc[
            backtest["horizon"].eq("day_ahead")
            & backtest["direction"].eq(direction)
            & backtest["duration_hours"].eq(duration)
            & backtest["level"].eq(0.10),
            ["local_half_hour", "coverage", "coverage_strict", "pinball_loss_kw", "sample_count"],
        ].rename(
            columns={
                "local_half_hour": "Half-hour (London)",
                "coverage": "Coverage",
                "coverage_strict": "Coverage (strict)",
                "pinball_loss_kw": "Pinball loss (kW)",
                "sample_count": "Week-nights kept",
            }
        ),
        definition="Whether the P10 forecast is calibrated at every half-hour of the week, or "
        "only on average. Leave-one-week-out backtest: each simulated week is forecast from "
        "the others, then checked against itself.",
        height=CHART_HEIGHTS["time_series"],
        key="firm-mw-backtest-half-hour",
    )

    summary = stat_row(
        result.availability_backtest_summary,
        horizon="day_ahead",
        direction=direction,
        duration_hours=duration,
    )
    columns = kpi_columns(st, 4)
    # B-6: show both hit rates, never "at or below" alone. Many half-hours
    # forecast and realise exactly 0 kW; a calibrated forecast sits between
    # the strict and the weak rate, not on either one alone (§10.3).
    coverage_help = (
        "Strictly-below and at-or-below hit rates. Many half-hours forecast and realise "
        "exactly 0 kW, and only the weaker rate counts those as covered. A calibrated "
        "forecast sits between the two."
    )
    kpi(
        columns[0],
        "Coverage at P10",
        UNAVAILABLE
        if summary is None
        else f"{percent(100 * summary['coverage_strict_p10'], decimals=1)}–"
        f"{percent(100 * summary['coverage_p10'], decimals=1)}",
        context="Strict to weak; target 10%",
        help=coverage_help,
    )
    kpi(
        columns[1],
        "Coverage at P90",
        UNAVAILABLE
        if summary is None
        else f"{percent(100 * summary['coverage_strict_p90'], decimals=1)}–"
        f"{percent(100 * summary['coverage_p90'], decimals=1)}",
        context="Strict to weak; target 90%",
        help=coverage_help,
    )
    kpi(
        columns[2],
        "Pinball loss, mean",
        UNAVAILABLE
        if summary is None
        else format_quantity(summary["pinball_mean_kw"], "kW", decimals=1),
        context="Mean over the seven levels; lower is better",
    )
    kpi(
        columns[3],
        "Sharpness, P10-P90",
        UNAVAILABLE
        if summary is None
        else format_quantity(summary["sharpness_p10_p90_kw"], "kW", decimals=1),
        context="Band width; shown beside coverage, never alone",
    )


# --------------------------------------------------------------------------
# Firmness by manufacturer, charge completion and the settlement file
# --------------------------------------------------------------------------


def _firmness_table(firmness: pd.DataFrame, *, direction: str, duration: float) -> pd.DataFrame:
    rows = firmness.loc[
        firmness["direction"].eq(direction) & firmness["duration_hours"].eq(duration)
    ]
    return pd.DataFrame(
        {
            "Maker": rows["manufacturer_label"],
            "Response rate": [percent(100 * v) for v in rows["response_rate"]],
            "Outage chance, per night": [
                percent(100 * v) for v in rows["outage_probability_per_night"]
            ],
            "Firmness, P10": [
                UNAVAILABLE if missing(v) else percent(100 * v) for v in rows["firmness_p10"]
            ],
            "Firmness, P50": [
                UNAVAILABLE if missing(v) else percent(100 * v) for v in rows["firmness_p50"]
            ],
            "Outage nights, P50": rows["outage_nights_p50"],
        }
    )


def _completion_tile(summary: pd.DataFrame, *, path_id: str, departure: str) -> pd.Series | None:
    return stat_row(summary, group_id="fleet", path_id=path_id, departure=departure)


def _completion_maker_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Complete-by-departure, P50, per maker, one column per policy the run has.

    Gains a Timed tariff column (decision 0007) exactly when ``summary``
    carries that path's maker rows; absent, exactly the two columns as before.
    """

    makers = summary.loc[
        summary["group_id"].isin(("m1", "m2", "m3", "m4")) & summary["departure"].eq("all")
    ]
    rows = []
    for maker_id in ("m1", "m2", "m3", "m4"):
        row: dict[str, str] = {"Maker": group_label(maker_id, {})}
        for path_id in present_paths(makers["path_id"]):
            label = PATH_LABELS[path_id]
            match = makers.loc[makers["group_id"].eq(maker_id) & makers["path_id"].eq(path_id)]
            value = match["completed_share_p50"].iloc[0] if len(match) else float("nan")
            row[f"Complete by departure, {label}"] = (
                UNAVAILABLE if missing(value) else percent(100 * value)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _render_firmness_and_completion(
    st: Any, result: Any, *, direction: str, duration: float
) -> None:
    st.markdown(
        "**Firmness by manufacturer**",
        help="Firmness is delivered ÷ deliverable, where deliverable is what the fleet would "
        "have given had every session followed its plan. This is physical firmness. The "
        "firmness on Trading ▸ 3 is measured against the traded position instead.",
    )
    st.dataframe(
        _firmness_table(result.firmness_by_manufacturer, direction=direction, duration=duration),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"P10 and P50 across {result.world_count} simulated weeks. {evidence_caption(result)}"
    )

    st.markdown(
        "**Charge completion by departure**",
        help="The share of home sessions at or above the preferred target when the driver "
        "leaves. The preferred target is the level the driver asks for, not a full battery.",
    )
    completion = result.charge_completion_summary
    columns = kpi_columns(st, 4)
    for column, (label, path_id, departure) in zip(
        columns,
        (
            ("Unmanaged, all sessions", "normal", "all"),
            ("Unmanaged, early departures", "normal", "early"),
            ("Smart, all sessions", "selected", "all"),
            ("Smart, early departures", "selected", "early"),
        ),
        strict=True,
    ):
        row = _completion_tile(completion, path_id=path_id, departure=departure)
        kpi(
            column,
            label,
            UNAVAILABLE
            if row is None or missing(row["completed_share_p50"])
            else percent(100 * row["completed_share_p50"]),
            context="Share complete by unplug, P50",
        )
    if "timed" in set(completion["path_id"]):
        # The timed path's own ready-at-departure share (decision 0007),
        # beside Smart's, never instead of it: a second row rather than a
        # fifth tile past the kpi_columns row limit. Absent, nothing below
        # this point renders and the block above is exactly as today.
        timed_columns = kpi_columns(st, 2)
        for column, (label, departure, help_text) in zip(
            timed_columns,
            (
                ("Timed tariff, all sessions", "all", None),
                (
                    "Timed tariff, early",
                    "early",
                    "Sessions that unplugged before the planner's expected departure, "
                    "as for the row above.",
                ),
            ),
            strict=True,
        ):
            row = _completion_tile(completion, path_id="timed", departure=departure)
            kpi(
                column,
                label,
                UNAVAILABLE
                if row is None or missing(row["completed_share_p50"])
                else percent(100 * row["completed_share_p50"]),
                context="Share complete by unplug, P50",
                help=help_text,
            )
    st.dataframe(_completion_maker_table(completion), width="stretch", hide_index=True)
    st.caption(
        "Complete by departure, P50 across simulated weeks, all sessions ending in the study."
    )

    st.markdown(
        "**Settlement file**",
        help="A per-EV settlement file for the representative simulated week, plus one row per "
        "half-hour for the baseline effect: settlement value that comes from where the "
        "baseline sits, not from any one EV's flexibility, so it is allocated to no EV.",
    )
    settlement = result.settlement_file
    st.download_button(
        "Download settlement file (CSV)",
        data=settlement.to_csv(index=False),
        file_name=result.settlement_file_name,
        mime="text/csv",
        help="The EV id stands in for the meter point (MPAN). One sample week only.",
        on_click=usage_log.log_event,
        args=(st.session_state, "csv_download"),
        kwargs={"which": "firm MW"},
    )
    st.caption(
        f"{len(settlement):,} rows: one per meter point and half-hour, plus the baseline effect."
    )


# --------------------------------------------------------------------------
# Blackout windows note (§10.5b)
# --------------------------------------------------------------------------


def _render_blackout_note(st: Any, result: Any) -> None:
    windows = getattr(result, "blackout_windows", None)
    if windows is None or windows.empty:
        st.caption("No blackout windows this run: charging may be moved in every half-hour.")
        return
    table = windows.loc[:, ["start_local_time", "end_local_time", "duration_minutes"]].rename(
        columns={
            "start_local_time": "Start (London)",
            "end_local_time": "End (London)",
            "duration_minutes": "Duration (minutes)",
        }
    )
    st.markdown(
        "**Blackout windows**",
        help="Half-hours where the trading desk promises not to move charging, in either "
        "direction. Deliverable kW is 0 there.",
    )
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption("Deliverable kW is 0 in and across these half-hours, day-ahead and intraday alike.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

_REQUIRED_FRAMES = (
    "availability_bands",
    "availability_world_slot",
    "firm_share_by_fleet_size",
    "manufacturer_summary",
    "product_sheet",
    "availability_value_summary",
    "availability_reliability",
    "availability_backtest",
    "availability_backtest_summary",
    "firmness_by_manufacturer",
    "charge_completion_summary",
    "settlement_file",
    "world_nights",
)


def render_firm_mw(st: Any, result: Any) -> None:
    """Render Supplier ▸ 4 Firm MW: the day-ahead band and its tiles as the hero, the rest
    (intraday, diversification, product sheet, calibration, firmness, completion, the
    settlement file and blackout note) in detail tabs (Q-13: 4,942 px and 14 tiles read as
    buried, not answered)."""

    missing_frames = [name for name in _REQUIRED_FRAMES if getattr(result, name, None) is None]
    if missing_frames:
        # B-19: no internal contract citation on the main surface (copy rule
        # 9); st.info has no help= in this Streamlit version to hold one.
        st.info(
            "This run predates the Firm MW figures (availability, product sheet and "
            "calibration). Run simulation again to see them."
        )
        return

    direction, duration = _controls(st)
    # Q-6: Firm MW's turn-down is measured against the smart-charged fleet's
    # own schedule; Availability and Key stats measure against the
    # unmanaged baseline instead, a 20-300x gap with no bridge (lead ruling).
    st.caption(
        "Turn-down here is beyond what smart charging already moved, not against unmanaged "
        "charging as Supplier ▸ Availability measures it."
    )

    st.markdown("**Day-ahead: the band across simulated weeks**")
    _render_day_ahead(st, result, direction=direction, duration=duration)

    intraday, diversification, product, calibration, firmness = st.tabs(
        [
            "Intraday",
            "Diversification",
            "Product sheet",
            "Calibration",
            "Firmness, completion and settlement",
        ]
    )
    # Each helper below still takes the plain ``st`` module (entered through
    # the ``with tab:`` context), never the tab object itself: chart_block's
    # own ``with st.expander(...)`` only redirects later same-object calls
    # when ``st`` is the module Streamlit's context stack tracks, not a
    # tab/column handle held as a local variable (verified against AppTest:
    # calling ``.dataframe()`` on a held tab handle after ``.expander()``
    # on that same handle renders as the tab's sibling, not its child).
    with intraday:
        st.markdown("**Intraday: the whole night from a 17:00 decision**")
        _render_intraday(st, result, direction=direction, duration=duration)
    with diversification:
        st.markdown("**Diversification: why the firm share stays below 100%**")
        _render_diversification(st, result, direction=direction, duration=duration)
    with product:
        _render_product_sheet(st, result, direction=direction, duration=duration)
    with calibration:
        st.markdown("**Calibration**")
        _render_calibration(st, result, direction=direction, duration=duration)
    with firmness:
        _render_firmness_and_completion(st, result, direction=direction, duration=duration)
        _render_blackout_note(st, result)
