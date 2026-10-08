"""Drivers ▸ Household: what does this driver get across every simulated week?

One EV seen across all simulated weeks, beside its archetype and the fleet
(household contract v1 §6.1). One EV (sketch 1) is one car in one week, the
physics; this lens is the same car's outcomes: ready mornings, value,
sessions affected and the flexibility it could offer.

Reads only the ``HouseholdCard`` of the chosen EV (contract §3.3), built by
the injected ``household_card(result, unit_id)``. That call is the one-EV
model call pattern One EV already uses: its outcomes are reads of the run's
household frames, and its timing and availability come from one cached
one-EV kernel slice, never a full Monte Carlo. This view formats and plots
the card's columns; it computes no quantile, share, rank, physics or money.
The only arithmetic is exact display scaling: × 100 for a share shown as a
percent and × the week's night count for "6 of 7 nights" (a monotone
transform of a median is the median of the transform).

The EV selectbox is One EV's own (``one_ev.ev_picker``, key "one-ev-unit"),
so the chosen driver survives a switch between the two lenses (O8). Every
£ figure is the illustrative pass-through reading at synthetic day-ahead
prices; "firm" names only an across-weeks P10 (B10).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..components import header
from ..components.chart_table import chart_block
from ..components.kpi import badge_line, kpi, kpi_columns
from ..style import (
    ARCHETYPE_COLOURS,
    BAND_ALPHA,
    CHART_FONT_PX,
    CHART_HEIGHTS,
    MUTED_INK,
    SERIES_COLOURS,
    UNAVAILABLE,
    band_fill,
    format_quantity,
    hover_time_labels,
    london_time_axis,
    money,
    percent,
)
from .one_ev import DISPATCH_LOCK_HELP, _traits_caption, ev_picker
from .sessions import ARCHETYPE_ORDER

HouseholdCardFn = Callable[[Any, str], Any]
"""``household_card(result, unit_id) -> HouseholdCard`` (model/household.py)."""

NO_FRAMES_MESSAGE = "This run has no household outcomes. Run simulation again to add them."
FIRM_MW_PENDING = "Not in this run. Run simulation again to see it."

# Chart A panels: (metric, panel title, value formatter). Four week-ahead
# metrics only (contract §6.1 item 3), so every range is "in a typical week".
_SHARE_TO_PERCENT = 100.0


def _money2(value: float) -> str:
    return money(value, decimals=2)


_PENCE_PER_POUND = 100.0


def _per_kwh(value: float) -> str:
    """A stored £/kWh cost as p/kWh, 1 dp: the customer unit (final critique B-10)."""

    return format_quantity(value * _PENCE_PER_POUND, "p/kWh", decimals=1)


def _share(value: float) -> str:
    return percent(value * _SHARE_TO_PERCENT)


_PANELS = (
    ("value_gbp_per_week", "Value per week (£)", _money2),
    ("home_cost_gbp_per_kwh_selected", "Smart cost (p/kWh)", _per_kwh),
    ("completed_share_selected", "Ready at departure (%)", _share),
    ("cheap_share_selected", "In cheapest third (%)", _share),
)
_PANEL_SCALE = {
    "home_cost_gbp_per_kwh_selected": _PENCE_PER_POUND,
    "completed_share_selected": _SHARE_TO_PERCENT,
    "cheap_share_selected": _SHARE_TO_PERCENT,
}
_SHARE_PANELS = ("completed_share_selected", "cheap_share_selected")
"""Panels plotted as a share: their axis never leaves 0–100% (final critique B-11)."""
_ROWS = ("This EV", "Archetype", "Fleet")
_PANEL_GAP_PX = 64
"""Room between the two rows of small multiples for the lower row's titles."""
_RANGE_WIDTH = 8
_RANGE_PAD_SHARE = 0.06
_TITLE_ROOM_PX = 28
_MARKER_SIZE = 12
_SHARE_AXIS_HEADROOM = 4.0
"""Extra percentage points let onto a share panel's axis range beyond its true
0-100% domain, so a marker sitting exactly on the floor or ceiling (a common
result: "Ready at departure" is often 100%) renders in full instead of being
clipped in half at the plot edge. Fixed ``tick0``/``dtick`` below keep the
labels on round quarters regardless, so this headroom cannot resurface the
"a share axis ran 95-105%" bug the 0-100 clamp was added to fix (final
critique B-11): the drawn axis has room past 100%, the data still cannot."""

_DIRECTIONS = {"Turn-down": "turn_down", "Turn-up": "turn_up"}
_OUTER_BAND_ALPHA = BAND_ALPHA / 2
"""P5–P95 at half the P10–P90 opacity, so the tails read as the fainter level."""


# --- Reading the card ---------------------------------------------------------


def _outcome(card: Any, metric: str) -> pd.Series:
    """This EV's ``card.outcomes`` row for one metric."""

    return card.outcomes.loc[card.outcomes["metric"].eq(metric)].iloc[0]


def _outcome_or_none(card: Any, metric: str) -> pd.Series | None:
    """``card.outcomes`` row for ``metric``, or ``None`` when the model has not added it.

    Decision 0007: the optional Timed tariff path only gets its own
    ``*_timed`` outcome rows once the household per-EV pass adds them
    (``model/household.py``'s ``METRICS``/``DIFFERENCE_METRICS`` carry only
    ``_normal``/``_selected``/``_difference`` today); until then every tile
    must read exactly as it does now (the "absent data, unchanged screen"
    rule), so callers use this instead of ``_outcome`` for a ``_timed`` row.
    """

    rows = card.outcomes.loc[card.outcomes["metric"].eq(metric)]
    return rows.iloc[0] if not rows.empty else None


def _reliability(card: Any, metric: str) -> pd.Series:
    return card.reliability.loc[card.reliability["metric"].eq(metric)].iloc[0]


def _missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def _count(value: Any) -> str:
    """A count's P50: whole when it is whole, else one decimal (a linear quantile)."""

    if _missing(value):
        return UNAVAILABLE
    value = float(value)
    return f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"


def _rank_text(rank: Any, better_is: str, group: str) -> str:
    """Word a percentile rank (share of the group at or below this EV) for the hover.

    ``better_is`` "higher": "ahead of 62% of its archetype"; "lower": the
    share above this EV, "72% of its archetype pay more" (1 − rank, exact
    because it is a monotone transform of the model's median rank).
    """

    if _missing(rank):
        return f"no rank in its {group} (control group or no value)"
    if better_is == "lower":
        return f"{percent((1.0 - float(rank)) * 100.0)} of its {group} pay more"
    return f"ahead of {percent(float(rank) * 100.0)} of its {group}"


# --- Header, status and KPI row -------------------------------------------------


def _status_text(card: Any, *, dispatch_active: bool) -> str:
    """Enrolment, then the intraday dispatch state when this run actually dispatched.

    ``card.dispatch_locked`` is ``units.dispatch_locked``: bool on every
    action result, but False both for a free EV *and* for every EV when the
    intraday dispatch switch was off (intraday-dispatch-v1 §2). Gating on
    ``dispatch_active`` (``result.dispatch_world_slot is not None``,
    K4-flagged bug) keeps a switch-off run from reading its all-False column
    as "every EV re-plans hourly", which it never does with the switch off.
    A hold-out control EV keeps its locked/free status assigned but ignores
    every plan and charges by the normal rule (§2, §10.1e), so it is never
    "locked to its day-ahead plan" or "re-plans hourly" either: the caption
    is shown for enrolled (treated) EVs only.
    """

    status = (
        "Enrolled in smart charging"
        if card.treated
        else "Hold-out control group, not enrolled in smart charging"
    )
    if dispatch_active and card.treated and card.dispatch_locked is not None:
        status += (
            " · Locked to its day-ahead plan"
            if card.dispatch_locked
            else " · Re-plans hourly on the latest intraday price"
        )
    return status


def _kpi_row(st: Any, card: Any) -> None:
    ready_tile, value_tile, affected_tile, flex_tile = kpi_columns(st, 4)

    ready = _outcome(card, "completed_share_selected")
    ready_normal = _outcome(card, "completed_share_normal")
    # Decision 0007: an optional "completed_share_timed" row, read beside
    # unmanaged and smart, never instead of either (absent today -- see
    # _outcome_or_none -- so this tile reads exactly as it did before).
    ready_timed = _outcome_or_none(card, "completed_share_timed")
    if _missing(ready["p50"]):
        # An EV plugged in all week has no closed session, so no readiness (O4).
        kpi(ready_tile, "Ready at departure", UNAVAILABLE, context="(no closed sessions)")
    else:
        context = (
            f"in a typical week P10–P90 {_share(ready['p10'])}–{_share(ready['p90'])} "
            f"· unmanaged {_share(ready_normal['p50'])}"
        )
        if ready_timed is not None and not _missing(ready_timed["p50"]):
            context += f" · timed tariff {_share(ready_timed['p50'])}"
        kpi(
            ready_tile,
            "Ready at departure",
            _share(ready["p50"]),
            context=context,
            help=(
                "Share of this driver's closed sessions that reached the preferred target by "
                "departure on the smart path. P50 across simulated weeks."
            ),
        )

    # A year is this EV's average week × 52; its band is across the archetype's
    # customers' average years, never one week's spread × 52 (B1).
    year = _outcome(card, "value_gbp_per_year")
    kpi(
        value_tile,
        "Value per year",
        money(year["mean"]),
        context=(
            f"archetype's customers {money(year['cohort_p10'])}–{money(year['cohort_p90'])} "
            "(P10–P90 average years) · illustrative"
        ),
        help=(
            "This driver's average simulated week × 52. Value is the charging saving plus this "
            "driver's share of the trading desk's revenue, as if day-ahead prices reached the "
            "customer. Synthetic prices, so illustrative."
        ),
    )

    affected = _outcome(card, "sessions_affected_count")
    ends = _outcome(card, "session_ends_per_week")
    kpi(
        affected_tile,
        "Sessions affected",
        _count(affected["p50"]),
        context=f"of {_count(ends['p50'])} session ends in a typical week (P50)",
        help="Session ends where smart charging left the battery lower than unmanaged charging "
        "would have.",
    )

    turn_down = _outcome(card, "evening_turn_down_kw_1h")
    if _missing(turn_down["p50"]):
        kpi(
            flex_tile,
            "Evening turn-down",
            UNAVAILABLE,
            context="not in this run",
        )
    else:
        kpi(
            flex_tile,
            "Evening turn-down",
            format_quantity(turn_down["p50"], "kW", decimals=1),
            context="1 h, 17:00–21:00, smart path; 0 when the plan is not charging then",
            help=(
                "Mean kW this driver could hold back for an hour on 17:00–21:00 evenings, P50 "
                "across simulated weeks. Smart charging usually moves this driver's evening "
                "charging to cheaper hours, so 0 is common: there is nothing left to turn down. "
                "Firm means the level met in 9 weeks out of 10. Firm (P10 across weeks): "
                f"{format_quantity(turn_down['p10'], 'kW', decimals=1)}."
            ),
        )


# --- Chart A: you vs archetype vs fleet -----------------------------------------


def _archetype_colour(cohort_id: str) -> str:
    if cohort_id in ARCHETYPE_ORDER:
        return ARCHETYPE_COLOURS[ARCHETYPE_ORDER.index(cohort_id)]
    return MUTED_INK


def _range_traces(
    row: pd.Series,
    *,
    label: str,
    prefix: str,
    colour: str,
    name: str,
    scale: float,
    formatter: Callable[[float], str],
    note: str,
) -> list[go.Scatter]:
    """One P10–P90 range (thick segment) and its P50 marker on one category row."""

    low, mid, high = row[f"{prefix}p10"], row[f"{prefix}p50"], row[f"{prefix}p90"]
    hover = f"{label}: P50 {formatter(mid)} · P10–P90 {formatter(low)}–{formatter(high)}"
    if note:
        hover += f"<br>{note}"
    return [
        go.Scatter(
            x=[float(low) * scale, float(high) * scale],
            y=[label, label],
            mode="lines",
            line={"color": colour, "width": _RANGE_WIDTH},
            name=name,
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[float(mid) * scale],
            y=[label],
            mode="markers",
            marker={
                "color": colour,
                "size": _MARKER_SIZE,
                "line": {"color": MUTED_INK, "width": 1},
            },
            name=name,
            showlegend=False,
            customdata=[hover],
            hovertemplate="%{customdata}<extra></extra>",
        ),
    ]


def _padded_range(
    outcome: pd.Series, scale: float, *, bounds: tuple[float, float] | None = None
) -> list[float] | None:
    """An x range around every plotted value with a margin, so no thick range end is clipped.

    Plotly pads autorange for markers but not for line ends, so a P10 or P90
    at the data extent was cut by the plot edge. Display only: the values
    are the card's own columns, scaled like the traces. ``bounds`` clips the
    padded range to the metric's domain: a share axis ran 95–105% because
    the pad crossed 100% (final critique B-11), and a share above 100% is
    impossible.
    """

    columns = [f"{g}{q}" for g in ("", "cohort_", "fleet_") for q in ("p10", "p50", "p90")]
    values = outcome[columns].to_numpy(dtype=float) * scale
    values = values[~np.isnan(values)]
    if values.size == 0:
        return None
    low, high = float(values.min()), float(values.max())
    pad = (high - low) * _RANGE_PAD_SHARE or abs(high) * _RANGE_PAD_SHARE or 1.0
    padded = [low - pad, high + pad]
    if bounds is not None:
        padded = [max(padded[0], bounds[0]), min(padded[1], bounds[1])]
    return padded


def _outcomes_figure(card: Any, cohort_label: str) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        shared_yaxes=True,
        subplot_titles=[title for _, title, _ in _PANELS],
        horizontal_spacing=0.08,
        vertical_spacing=_PANEL_GAP_PX
        / (2 * CHART_HEIGHTS["small_multiple_panel"] + _PANEL_GAP_PX),
    )
    archetype_colour = _archetype_colour(card.cohort_id)
    for index, (metric, _, formatter) in enumerate(_PANELS):
        row, col = index // 2 + 1, index % 2 + 1
        outcome = _outcome(card, metric)
        scale = _PANEL_SCALE.get(metric, 1.0)
        groups = (
            # This EV on the smart path, so teal; the name says so (style.py:
            # teal is reserved for the smart path).
            (
                "This EV",
                "",
                SERIES_COLOURS["selected"],
                "This EV (smart)",
                _rank_text(outcome["cohort_rank_p50"], outcome["better_is"], "archetype"),
            ),
            ("Archetype", "cohort_", archetype_colour, cohort_label, cohort_label),
            ("Fleet", "fleet_", SERIES_COLOURS["normal"], "Fleet", "every enrolled customer"),
        )
        for label, prefix, colour, name, note in groups:
            for trace in _range_traces(
                outcome,
                label=label,
                prefix=prefix,
                colour=colour,
                name=name,
                scale=scale,
                formatter=formatter,
                note=note,
            ):
                figure.add_trace(trace, row=row, col=col)
        bounds = (0.0, 100.0) if metric in _SHARE_PANELS else None
        range_bounds = (
            (bounds[0] - _SHARE_AXIS_HEADROOM, bounds[1] + _SHARE_AXIS_HEADROOM)
            if bounds is not None
            else None
        )
        figure.update_xaxes(
            range=_padded_range(outcome, scale, bounds=range_bounds), row=row, col=col
        )
        if metric in _SHARE_PANELS:
            figure.update_xaxes(tick0=0, dtick=25, row=row, col=col)
    # Category axes list bottom-up, so the array is reversed to put This EV on top.
    figure.update_yaxes(categoryorder="array", categoryarray=list(reversed(_ROWS)))
    # Four ticks at most per panel: at 390 px the two panels' end labels met.
    figure.update_xaxes(nticks=4)
    figure.update_annotations(font_size=CHART_FONT_PX)
    # Room above the top row for its panel titles (the template's top margin is 8 px).
    figure.update_layout(hovermode="closest", margin={"t": _TITLE_ROOM_PX})
    return figure


def _outcomes_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    return outcomes.rename(
        columns={
            "metric": "Metric",
            "unit": "Unit",
            "horizon": "Horizon",
            "better_is": "Better is",
            "world_count": "Weeks",
            "mean": "This EV mean",
            "p10": "This EV P10",
            "p50": "This EV P50",
            "p90": "This EV P90",
            "cohort_p10": "Archetype P10",
            "cohort_p50": "Archetype P50",
            "cohort_p90": "Archetype P90",
            "fleet_p10": "Fleet P10",
            "fleet_p50": "Fleet P50",
            "fleet_p90": "Fleet P90",
            "cohort_rank_p50": "Rank in archetype (P50)",
            "fleet_rank_p50": "Rank in fleet (P50)",
        }
    )


def _outcomes_chart(st: Any, card: Any, cohort_label: str) -> None:
    panels = card.outcomes.loc[card.outcomes["metric"].isin([m for m, _, _ in _PANELS])]
    chart_block(
        st,
        _outcomes_figure(card, cohort_label),
        title="This driver, the archetype and the fleet, in a typical week",
        caption=(
            f"This EV: P10–P90 across {card.world_count} weeks. Archetype and fleet: "
            "P10–P90 across EVs, medians across weeks. Illustrative."
        ),
        frame=_outcomes_table(panels),
        definition=(
            "Two different spreads. This EV: its own value in each simulated week, so P10–P90 "
            "across weeks is week-to-week luck. Archetype and fleet: in each week, P10–P90 "
            "across the group's enrolled customers; each figure is then its median across "
            "weeks. The three rows therefore come from different weeks. Value is the charging "
            "saving plus the customer's share of the trading desk's revenue, as if day-ahead "
            "prices reached the customer. A year is the average week × 52; its band, on the "
            "Value per year tile, is across customers' average years. Money is illustrative, "
            "from synthetic prices."
        ),
        height=2 * CHART_HEIGHTS["small_multiple_panel"] + _PANEL_GAP_PX + _TITLE_ROOM_PX,
        key="household-outcomes",
    )


# --- Chart B: flexibility this driver can offer -----------------------------------


def _band_pair(
    rows: pd.DataFrame, low: str, high: str, *, name: str, group: str, alpha: float, rank: int
) -> list[go.Scatter]:
    colour = SERIES_COLOURS["selected"]
    x = rows["interval_start_utc"]
    return [
        go.Scatter(
            x=x,
            y=rows[low],
            mode="lines",
            line={"color": colour, "width": 0},
            legendgroup=group,
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=x,
            y=rows[high],
            mode="lines",
            line={"color": colour, "width": 0},
            fill="tonexty",
            fillcolor=band_fill(colour, alpha),
            name=name,
            legendgroup=group,
            legendrank=rank,
            hoverinfo="skip",
        ),
    ]


_AVAILABILITY_PANEL_GAP_PX = 40
"""Room between the kW panel and the share-of-weeks panel (the One EV
``make_subplots`` pattern, ``one_ev.py:_BAR_PANEL_GAP_PX``)."""


def _availability_figure(rows: pd.DataFrame, *, word: str) -> go.Figure:
    """The P50/band chart, plus a lower bar panel naming the P50-is-0 pattern.

    Clarity critique section 0.1: one EV can only turn down (or up) in
    half-hours where its smart plan is doing that then, and the plan picks
    each week's own cheapest half-hours, so most half-hours have it in fewer
    than half the weeks -- the P50 line sits on 0 kW and the P10–P90/P5–P95
    bands are isolated spikes where the minority of weeks did move. That is
    not a broken chart; ``available_share`` (household contract §3.3b, the
    same column the hover text already reads) already carries the share of
    weeks with any move at all, and plotting it as its own panel explains the
    spikes above by construction: a spike sits exactly where the bar is low.
    """

    top_height = CHART_HEIGHTS["time_series"]
    bottom_height = CHART_HEIGHTS["small_multiple_panel"]
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[top_height, bottom_height],
        vertical_spacing=_AVAILABILITY_PANEL_GAP_PX / (top_height + bottom_height),
    )
    # Outer P5–P95 first so the inner P10–P90 and the line draw on top. The
    # legend groups carry "smart": the whole chart is the smart path (teal).
    for trace in _band_pair(
        rows, "p05", "p95", name="P5–P95", group="smart-tails", alpha=_OUTER_BAND_ALPHA, rank=3
    ):
        figure.add_trace(trace, row=1, col=1)
    for trace in _band_pair(
        rows, "p10", "p90", name="P10–P90", group="smart", alpha=BAND_ALPHA, rank=2
    ):
        figure.add_trace(trace, row=1, col=1)
    hover = [
        f"{when}<br>P50 {format_quantity(p50, 'kW', decimals=1)} · P10–P90 "
        f"{format_quantity(p10, 'kW', decimals=1)}–{format_quantity(p90, 'kW', decimals=1)}"
        f"<br>on the driveway and able to move charging in {percent(share * 100.0)} of weeks"
        for when, p10, p50, p90, share in zip(
            hover_time_labels(rows["interval_start_utc"]),
            rows["p10"],
            rows["p50"],
            rows["p90"],
            rows["available_share"],
            strict=True,
        )
    ]
    figure.add_trace(
        go.Scatter(
            x=rows["interval_start_utc"],
            y=rows["p50"],
            mode="lines",
            line={"color": SERIES_COLOURS["selected"], "width": 2},
            name="P50",
            legendrank=1,
            legendgroup="smart",
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    share_hover = [
        f"{when}<br>{percent(share * 100.0)} of weeks able to {word}"
        for when, share in zip(
            hover_time_labels(rows["interval_start_utc"]), rows["available_share"], strict=True
        )
    ]
    figure.add_trace(
        go.Bar(
            x=rows["interval_start_utc"],
            y=rows["available_share"] * 100.0,
            marker={"color": SERIES_COLOURS["selected"]},
            name=f"Share of weeks with any {word}",
            # Own legendgroup, not "smart": it is still the smart path's own
            # stat (style.py reserves teal for that), but toggling the P50
            # line should not also hide this panel's bars.
            legendgroup="smart-share",
            legendrank=4,
            customdata=share_hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    # Shared x-axis (make_subplots): ticks set on the bottom row apply to
    # both, the same pattern One EV's bar panel uses.
    figure.update_xaxes(
        tickmode="array", **london_time_axis(rows["interval_start_utc"]), row=2, col=1
    )
    figure.update_yaxes(title="kW", rangemode="tozero", row=1, col=1)
    figure.update_yaxes(title="Share of weeks able (%)", range=[0, 100], row=2, col=1)
    # Legend in reading order P50, P10–P90, P5–P95, then the share bar.
    figure.update_layout(legend={"traceorder": "normal"})
    return figure


def _availability_day_table(day: pd.DataFrame) -> pd.DataFrame:
    return day.rename(
        columns={
            "direction": "Direction",
            "day_type": "Day type",
            "local_half_hour": "Half-hour index",
            "local_time_label": "London time",
            "unit": "Unit",
            "world_count": "Weeks",
            "available_share": "Share of weeks available",
            "mean": "Mean",
            "p05": "P5",
            "p10": "P10",
            "p50": "P50",
            "p90": "P90",
            "p95": "P95",
            "firm_share": "Firm share (P10 ÷ P50)",
            "evidence_kind": "Evidence",
        }
    )


def _availability_chart(st: Any, card: Any) -> None:
    title = "Flexibility this driver can offer"
    if card.availability is None:
        st.markdown(f"**{title}**")
        st.info(FIRM_MW_PENDING)
        return
    choice = st.segmented_control(
        "Direction",
        options=list(_DIRECTIONS),
        default="Turn-down",
        required=True,
        key="household-direction",
        label_visibility="collapsed",
    )
    direction = _DIRECTIONS.get(choice, "turn_down")
    rows = card.availability.loc[card.availability["direction"].eq(direction)].sort_values(
        "slot_index"
    )
    day = card.availability_day.loc[card.availability_day["direction"].eq(direction)]
    word = "turn-down" if direction == "turn_down" else "turn-up"
    # Clarity critique §0.1, Mike's household example: turn-down's P50 sits on
    # 0 kW in most half-hours (this EV can only turn down while its smart
    # plan is charging then, and the plan's cheapest half-hours differ week
    # to week), so the bands read as isolated spikes -- "the P50 line doesn't
    # work". It does; nothing says why, so the caption and a second line
    # below now do. Turn-up has no such reading (headroom exists whenever the
    # EV is below target, not only in a minority of weeks), so it keeps the
    # plain caption.
    zero_median_note = (
        "; median 0 where the plan rarely charges then" if direction == "turn_down" else ""
    )
    chart_block(
        st,
        _availability_figure(rows, word=word),
        title=title,
        caption=(
            f"1 h {word} on the smart path. Bands: P10–P90 and P5–P95 across "
            f"{card.world_count} weeks{zero_median_note}. Illustrative."
        ),
        frame=_availability_day_table(day),
        definition=(
            "The kW this driver could hold back for a full hour when asked. It counts only when "
            "the EV is plugged in through the hour, following its smart plan, and still able "
            "to reach its preferred target. Measured on the smart path; 0 on a night its "
            "charger maker is offline or when it ignores the plan. Firm is the P10 line: in 9 "
            "weeks out of 10 it could hold at least that. The table folds the week to a typical "
            "day by day type, with the share of weeks the driver can move any charging at that "
            "time. Lower panel: that same share through the week. A spike in the upper panel "
            "sits exactly where this bar is low."
        ),
        height=CHART_HEIGHTS["time_series"] + CHART_HEIGHTS["small_multiple_panel"],
        key=f"household-availability-{direction}",
    )
    if direction == "turn_down":
        st.caption(
            "Median 0 kW most of the week: this EV can only turn down while its smart plan "
            "is charging, and that differs week to week.",
            help=(
                "Turn-down counts only half-hours where this EV is plugged in and its smart "
                "plan is charging. The plan picks each week's own cheapest half-hours, so most "
                "half-hours have turn-down in fewer than half the weeks. That puts the P50 at 0; "
                "the spikes are the weeks that did charge then. The lower panel gives that share "
                "directly. Hover either panel, or open Data and definition, for the share by "
                "time of day."
            ),
        )


# --- Chart C: when this driver plugs in and leaves ----------------------------------


def _timing_figure(timing: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    # Behaviour, not the smart path: greys only, no teal. The two greys are
    # close in hue, so departures are also hatched: colour is never the only
    # cue, and the pair still reads in greyscale.
    for metric, name, marker, noun in (
        ("plug_in_time", "Plug-in time", {"color": SERIES_COLOURS["plugged"]}, "plug-ins"),
        (
            "departure_time",
            "Departure time",
            {"color": MUTED_INK, "pattern": {"shape": "/"}},
            "departures",
        ),
    ):
        # profile_order starts at 12:00, so an overnight habit is one hump.
        rows = timing.loc[timing["metric"].eq(metric)].sort_values("profile_order")
        figure.add_trace(
            go.Bar(
                x=rows["bin_label"],
                y=rows["share"] * _SHARE_TO_PERCENT,
                name=name,
                marker=marker,
                hovertemplate=f"%{{x}}: %{{y:.0f}}% of {noun}<extra></extra>",
            )
        )
    labels = timing.loc[timing["metric"].eq("plug_in_time")].sort_values("profile_order")
    # A tick every three hours: 48 half-hour labels overlap at 390 px.
    ticks = [label for label in labels["bin_label"] if label.endswith(":00")][::3]
    figure.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=list(labels["bin_label"]),
        tickmode="array",
        tickvals=ticks,
        title="London time",
    )
    figure.update_yaxes(title="% of sessions", rangemode="tozero")
    figure.update_layout(barmode="group", hovermode="closest")
    return figure


def _timing_table(timing: pd.DataFrame) -> pd.DataFrame:
    return timing.rename(
        columns={
            "metric": "Metric",
            "unit": "Unit",
            "bin_index": "Bin from 00:00",
            "profile_order": "Bin from 12:00",
            "bin_lower": "From hour",
            "bin_upper": "To hour",
            "bin_label": "London time",
            "session_count": "Sessions",
            "share": "Share",
            "evidence_kind": "Evidence",
        }
    )


def _reliability_line(card: Any, night_count: int) -> str:
    nights = _reliability(card, "nights_plugged_share")["p50"]
    plug_in = _reliability(card, "plug_in_time")["p50_label"] or UNAVAILABLE
    departure = _reliability(card, "departure_time")["p50_label"] or UNAVAILABLE
    flexible = _reliability(card, "flexible_kwh_per_night")["p50"]
    nights_text = UNAVAILABLE if _missing(nights) else _count(float(nights) * night_count)
    return (
        f"Plugged in {nights_text} of {night_count} nights (P50); usually in by {plug_in}, "
        f"out by {departure} (P50); {format_quantity(flexible, 'kWh', decimals=1)} a night "
        "could move (P50)."
    )


def _timing_chart(st: Any, card: Any, night_count: int) -> None:
    title = "When this driver plugs in and leaves"
    timing = card.session_timing
    sessions = int(timing.loc[timing["metric"].eq("plug_in_time"), "session_count"].iat[0])
    if sessions == 0:
        st.markdown(f"**{title}**")
        st.info("No plug-in sessions in any simulated week.")
    else:
        chart_block(
            st,
            _timing_figure(timing),
            title=title,
            caption=(
                f"Share of this driver's sessions by London half-hour, pooled over "
                f"{card.world_count} weeks ({sessions:,} sessions). Unmanaged path."
            ),
            frame=_timing_table(timing),
            definition=(
                "Every session counts for plug-in time; only sessions that end inside the "
                "week count for departure. Pooled across simulated weeks because one driver "
                "has too few sessions in a week for a weekly share. This is a habit, not a "
                "forecast band."
            ),
            height=CHART_HEIGHTS["histogram"],
            key="household-timing",
        )
    st.caption(_reliability_line(card, night_count))


def _reliability_table(reliability: pd.DataFrame) -> pd.DataFrame:
    return reliability.rename(
        columns={
            "metric": "Metric",
            "unit": "Unit",
            "sample_kind": "Sample",
            "sample_count": "Count",
            "mean": "Mean",
            "p10": "P10",
            "p50": "P50",
            "p90": "P90",
            "p10_label": "P10 time",
            "p50_label": "P50 time",
            "p90_label": "P90 time",
            "ratio_p10_to_p50": "P10 ÷ P50",
            "evidence_kind": "Evidence",
        }
    )


# --- The lens -----------------------------------------------------------------------


def render_household(st: Any, result: Any, *, household_card: HouseholdCardFn) -> None:
    """Render Drivers ▸ Household: one driver's outcomes across every simulated week.

    ``household_card(result, unit_id)`` returns the chosen EV's
    ``HouseholdCard`` (contract §3.3), cached on the result; the page
    guards no-action results before this view is called.
    """

    units = result.units
    unit_id = ev_picker(header.controls(st), units)
    if getattr(result, "household_ev_world", None) is None:
        st.info(NO_FRAMES_MESSAGE)
        return
    traits = units.loc[units["unit_id"].eq(unit_id)].iloc[0].to_dict()
    st.caption(_traits_caption(traits))
    card = household_card(result, unit_id)
    dispatch_active = getattr(result, "dispatch_world_slot", None) is not None
    # DISPATCH_LOCK_HELP only applies once the status text actually names the
    # locked/re-plan state (mirrors _status_text's own gating).
    dispatch_help = (
        DISPATCH_LOCK_HELP
        if dispatch_active and card.treated and card.dispatch_locked is not None
        else None
    )
    badge_line(
        st, "Status", _status_text(card, dispatch_active=dispatch_active), help=dispatch_help
    )

    _kpi_row(st, card)
    _outcomes_chart(st, card, str(traits["cohort_label"]))
    _availability_chart(st, card)
    night_count = int(result.study_slots["night_index"].max()) + 1
    _timing_chart(st, card, night_count)

    with st.expander("Data and definition"):
        st.markdown("**Outcomes across weeks**")
        st.dataframe(_outcomes_table(card.outcomes), width="stretch", hide_index=True)
        st.markdown("**Reliability**")
        st.dataframe(_reliability_table(card.reliability), width="stretch", hide_index=True)
        st.caption(
            "Sessions run from plug-in to unplug and count only when they end inside the week; "
            "a session still plugged in at the end has no departure, so readiness and departure "
            "times lean short. Session ends also count one already running at the start, so "
            "'x of y session ends' has its own denominator. Weekly figures: P10–P90 across "
            "simulated weeks; a year is the average week × 52. A control-group EV's smart path "
            "equals its unmanaged path. The revenue share inside value is an allocation of "
            "fleet settlement by flexibility delivered, not a meter-level settlement. "
            "Illustrative, synthetic prices; not a bill or a tariff."
        )
