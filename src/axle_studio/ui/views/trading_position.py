"""Trading lens 2, Position: what the trader sold and how the fleet actually moved.

Reads ``deviation_world_slot`` (baseline, unmanaged and metered import per
slot, the three strategies' final position, trading contract v1 section
5.1), ``position_updates`` (the day-ahead and intraday decisions for
sampled worlds, section 5.2), ``trading_ledger_summary`` (settled volume per
night across simulated weeks, section 5.3-5.5) and ``open_position_profile``
(the open position by London half-hour, per strategy, section 9.1 and the
9.8 UI map), plus the baseline-erosion rows of ``trading_kpis`` (decision
0004 item 67). Everything shown is for the representative simulated week: the
day-ahead position, the intraday re-trades and the final position are all
per-world decisions, so there is no week-to-week fan for them the way there
is for the day-ahead price itself (Market lens); the open-position profile
is the one exception, already a world-first mean per half-hour before its
own P10-P90 across weeks.

The night and strategy pickers sit in the page header's controls slot
(polish plan G8), mirroring the EV/week pickers on Drivers ▸ One EV.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from ..components import header
from ..components.chart_table import chart_block
from ..components.kpi import kpi, kpi_columns
from ..style import (
    CHART_HEIGHTS,
    MUTED_INK,
    PATH_LABELS,
    PATH_STYLES,
    SERIES_COLOURS,
    assumption_value,
    band_and_line,
    band_fill,
    format_quantity,
    london_time_axis,
    money,
    percent,
)
from .trading_market import (
    STRATEGY_COLOURS,
    STRATEGY_LABELS,
    STRATEGY_ORDER,
    _evidence_text,
    _guard,
)

_STRATEGY_OPTIONS = ("full", "da_only", "perfect_foresight")
"""Selectbox order: the headline "full" strategy first (decision 0004 item
58), then day-ahead only, then the perfect-foresight upper bound."""


def _night_options(study_slots: pd.DataFrame) -> list[tuple[int, str]]:
    """``[(night_index, day_label), ...]`` in night order, one per session night."""

    nights = study_slots.drop_duplicates("night_index").sort_values("night_index")
    return list(zip(nights["night_index"], nights["day_label"], strict=True))


def _controls(st: Any, study_slots: pd.DataFrame) -> tuple[int, str]:
    """The Night and Strategy controls; returns the chosen ``(night_index, strategy)``.

    Both are ``selectbox``: a segmented control's pills sit side by side and
    can run past the header's narrow controls slot at 1280 px ("Perfect
    foresight" clipped to "Per"); a closed selectbox is one box that elides
    its own text with an ellipsis instead of overflowing its column, so it
    stays inside the layout at every width without touching the shared
    header column ratios.
    """

    night_column, strategy_column = header.controls(st).columns([2, 3])
    options = _night_options(study_slots)
    night = night_column.selectbox(
        "Night",
        options=[index for index, _ in options],
        index=0,
        format_func=lambda index: dict(options)[index],
        key="trading-position-night",
        label_visibility="collapsed",
    )
    strategy = strategy_column.selectbox(
        "Strategy",
        options=list(_STRATEGY_OPTIONS),
        index=0,
        format_func=STRATEGY_LABELS.__getitem__,
        key="trading-position-strategy",
        label_visibility="collapsed",
    )
    return night, strategy if strategy in _STRATEGY_OPTIONS else "full"


def _night_slots(deviation: pd.DataFrame, world_id: int, night_index: int) -> pd.DataFrame:
    rows = deviation.loc[
        deviation["world_id"].eq(world_id) & deviation["night_index"].eq(night_index)
    ]
    return rows.sort_values("slot_index")


def _night_kpis(st: Any, night: pd.DataFrame, strategy: str) -> None:
    """Sold, settled and imbalance turn-down for the chosen night and strategy (kWh)."""

    position_column = f"position_{strategy}_kwh"
    sold = float(night[position_column].sum())
    settled = float(night["settled_kwh"].sum())
    imbalance = settled - sold
    sold_tile, settled_tile, imbalance_tile = kpi_columns(st, 3)
    kpi(
        sold_tile,
        "Sold turn-down",
        format_quantity(sold, "kWh", decimals=0),
        context=f"{STRATEGY_LABELS[strategy]} strategy, this night",
        help="The turn-down this strategy had sold by the time each half-hour arrived, summed "
        "over the night. Representative simulated week (the median week).",
    )
    kpi(
        settled_tile,
        "Settled turn-down",
        format_quantity(settled, "kWh", decimals=0),
        context="Observed deviation, all strategies settle the same volume",
        help="Baseline minus what the fleet actually drew, counted only where positive, summed "
        "over the night. The fleet charges the same way under every strategy; only the money "
        "differs.",
    )
    kpi(
        imbalance_tile,
        "Imbalance, this strategy",
        format_quantity(imbalance, "kWh", decimals=0),
        context="Settled minus sold; positive means more delivered than committed",
        help="Settled minus sold: how far this strategy over- or under-sold against what the "
        "fleet delivered, before the imbalance price is applied.",
    )


def _night_span_axis(figure: go.Figure, x: pd.Series) -> None:
    """London clock ticks every 2 h, or every 4 h past a 12 h span (mirrors
    ``action_response._event_axis``): a full noon-to-noon night is about 24
    h, so this keeps six ticks rather than twelve, which collided into one
    unreadable strip at 390 px ("12:0014:0016:00...")."""

    hours = (x.max() - x.min()) / pd.Timedelta(hours=1)
    every = 2 if hours <= 12 else 4
    figure.update_xaxes(tickmode="array", **london_time_axis(x, every_hours=every))


def _night_figure(night: pd.DataFrame) -> go.Figure:
    """Baseline, unmanaged and metered import for one night, with settled turn-down shaded."""

    open_settlement = night["settled_kwh"] > 0.0
    # Outside the settled half-hours the band collapses onto the smart line
    # (zero height) rather than going blank: Plotly's "tonexty" fill joins
    # the pieces of a trace broken by NaN gaps, which drew diagonal wedges
    # across unsettled half-hours that looked like settled turn-down.
    shaded_low = night["baseline_kw"].where(open_settlement, night["metered_kw"])
    shaded_high = night["metered_kw"]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=night["interval_start_utc"],
            y=shaded_low,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=night["interval_start_utc"],
            y=shaded_high,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=band_fill(SERIES_COLOURS["selected"], 0.25),
            # Named for the smart path, not just "Settled turn-down": this
            # fill is the gap between the smart line and the baseline, so
            # teal is the smart-path colour, not a trading-strategy one
            # (teal is reserved sitewide for the physical dispatch path).
            # It is the settled (delivered, gated on settled_kwh > 0) volume,
            # not the day-ahead sold position the "Sold turn-down" tile
            # above reports -- the two differ by the imbalance (line 110).
            name="Smart settled turn-down",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=night["interval_start_utc"],
            y=night["baseline_kw"],
            mode="lines",
            name="Baseline (BL01-lite)",
            line={"color": SERIES_COLOURS["observed"], "width": 1.5, "dash": "dash"},
            hovertemplate="%{y:,.1f} kW<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=night["interval_start_utc"],
            y=night["unmanaged_kw"],
            mode="lines",
            name=PATH_LABELS["normal"],
            line=PATH_STYLES["normal"],
            hovertemplate="%{y:,.1f} kW<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=night["interval_start_utc"],
            y=night["metered_kw"],
            mode="lines",
            name=PATH_LABELS["selected"],
            line=PATH_STYLES["selected"],
            hovertemplate="%{y:,.1f} kW<extra></extra>",
        )
    )
    figure.update_xaxes(
        range=[night["interval_start_utc"].iat[0], night["interval_start_utc"].iat[-1]]
    )
    _night_span_axis(figure, night["interval_start_utc"])
    figure.update_yaxes(title="kW", rangemode="tozero")
    return figure


def _night_table(night: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Interval start (London)": night["interval_start_london"],
            "Baseline (kW)": night["baseline_kw"].round(1),
            "Unmanaged (kW)": night["unmanaged_kw"].round(1),
            "Metered, smart (kW)": night["metered_kw"].round(1),
            "Settled turn-down (kWh)": night["settled_kwh"].round(2),
        }
    )


def _position_figure(night: pd.DataFrame, strategy: str) -> go.Figure:
    """Day-ahead position against the chosen strategy's final position, with intraday close."""

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=night["interval_start_utc"],
            y=night["position_da_only_kwh"] / 0.5,
            mode="lines",
            name="Day-ahead position",
            line={"color": SERIES_COLOURS["difference"], "width": 1.5, "dash": "dot"},
            hovertemplate="%{y:,.1f} kW<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=night["interval_start_utc"],
            y=night[f"position_{strategy}_kwh"] / 0.5,
            mode="lines",
            name=f"Final position, {STRATEGY_LABELS[strategy].lower()}",
            line={"color": STRATEGY_COLOURS[strategy], "width": 2.0},
            hovertemplate="%{y:,.1f} kW<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=night["interval_start_utc"],
            y=night["intraday_close_gbp_per_mwh"],
            mode="lines",
            name="Intraday close",
            line={"color": SERIES_COLOURS["observed"], "width": 1.0, "dash": "dash"},
            yaxis="y2",
            hovertemplate="%{y:,.0f} £/MWh<extra></extra>",
        )
    )
    figure.update_xaxes(
        range=[night["interval_start_utc"].iat[0], night["interval_start_utc"].iat[-1]]
    )
    _night_span_axis(figure, night["interval_start_utc"])
    figure.update_yaxes(title="Position (kW)", rangemode="tozero")
    # A second, independent y-axis for the intraday close: sharing the
    # position axis would either swamp the position lines' shape or clip the
    # price at the position axis's range, because kW and £/MWh are unrelated
    # scales -- the same reasoning action_decision.py's Plan chart gives for
    # pairing charging power with the day-ahead price (decision 0004 item 43).
    figure.update_layout(
        yaxis2={
            "title": "Intraday close (£/MWh)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        }
    )
    return figure


def _position_table(night: pd.DataFrame, strategy: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Interval start (London)": night["interval_start_london"],
            "Day-ahead position (kW)": (night["position_da_only_kwh"] / 0.5).round(1),
            f"Final position, {STRATEGY_LABELS[strategy]} (kW)": (
                night[f"position_{strategy}_kwh"] / 0.5
            ).round(1),
            "Intraday close (£/MWh)": night["intraday_close_gbp_per_mwh"].round(0),
        }
    )


def _settled_by_night(ledger_summary: pd.DataFrame, study_slots: pd.DataFrame) -> pd.DataFrame:
    """Settled turn-down per night across simulated weeks, ``full`` strategy (identical for all)."""

    rows = ledger_summary.loc[
        ledger_summary["strategy"].eq("full")
        & ledger_summary["night_index"].notna()
        & ledger_summary["metric"].eq("settled_mwh")
    ].copy()
    rows["night_index"] = rows["night_index"].astype(int)
    rows = rows.sort_values("night_index")
    labels = dict(_night_options(study_slots))
    rows["day_label"] = rows["night_index"].map(labels)
    return rows


def _settled_figure(nights: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    band_and_line(
        figure,
        nights["day_label"],
        nights["p10"],
        nights["p50"],
        nights["p90"],
        name="Settled turn-down",
        colour=SERIES_COLOURS["selected"],
        hovertemplate="%{y:,.2f} MWh<extra></extra>",
    )
    figure.update_yaxes(title="MWh per night", rangemode="tozero")
    return figure


def _settled_table(nights: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Night": nights["day_label"],
            "P10 (MWh)": nights["p10"].round(2),
            "Median (MWh)": nights["p50"].round(2),
            "P90 (MWh)": nights["p90"].round(2),
        }
    )


_TICK_EVERY_N_HALF_HOURS = 8
"""Label every 4th hour (8 half-hours) of the 48-slot noon-to-noon profile,
so the axis reads without 48 overlapping labels."""


def _open_position_ticks(rows: pd.DataFrame) -> dict[str, Any]:
    """Tick positions and London-time labels for the noon-to-noon profile axis."""

    ticked = rows.iloc[::_TICK_EVERY_N_HALF_HOURS]
    return {
        "tickmode": "array",
        "tickvals": list(ticked["profile_order"]),
        "ticktext": list(ticked["local_half_hour"]),
        "tickangle": 0,
    }


def _signed_open_position(profile: pd.DataFrame) -> pd.DataFrame:
    """The signed ``open_position_kw`` rows only (``profile`` also carries the absolute metric)."""

    return profile.loc[profile["metric"].eq("open_position_kw")]


def _open_position_figure(profile: pd.DataFrame) -> go.Figure:
    """Open position (``V - f``) by half-hour, one line per strategy (trading contract 9.1)."""

    signed = _signed_open_position(profile)
    figure = go.Figure()
    ticks = None
    for strategy in STRATEGY_ORDER:
        rows = signed.loc[signed["strategy"].eq(strategy)].sort_values("profile_order")
        ticks = ticks or _open_position_ticks(rows)
        figure.add_trace(
            go.Scatter(
                x=rows["profile_order"],
                y=rows["p50"],
                mode="lines",
                name=STRATEGY_LABELS[strategy],
                line={"color": STRATEGY_COLOURS[strategy], "width": 1.5},
                hovertemplate="%{y:,.1f} kW<extra></extra>",
            )
        )
    figure.update_xaxes(title="London half-hour, noon to noon", **(ticks or {}))
    figure.update_yaxes(
        title="Open position (kW)", zeroline=True, zerolinecolor=MUTED_INK, rangemode="normal"
    )
    return figure


def _open_position_table(profile: pd.DataFrame) -> pd.DataFrame:
    """One row per noon-to-noon half-hour, median signed kW per strategy."""

    signed = _signed_open_position(profile)
    pivot = signed.pivot(index="profile_order", columns="strategy", values="p50").sort_index()
    labels = signed.drop_duplicates("profile_order").set_index("profile_order")["local_half_hour"]
    table = pd.DataFrame({"Half-hour": labels.loc[pivot.index]})
    for strategy in STRATEGY_ORDER:
        table[f"{STRATEGY_LABELS[strategy]} (kW)"] = pivot[strategy].round(1).to_numpy()
    return table.reset_index(drop=True)


_EROSION_WEEK_ROWS = (
    ("settled_value_gbp_per_week", "Settled value, default baseline"),
    ("settled_value_eroded_gbp_per_week", "Settled value, smart-night baseline"),
    ("baseline_erosion_share", "Value lost to baseline erosion"),
)
"""The whole-week erosion rows, shown in the expander below the evening headline."""


def _erosion_week_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Whole-week erosion figures: P50 and P10–P90 across simulated weeks; the share's
    headline is its ratio of means, as on the evening tile."""

    table = []
    for metric, label in _EROSION_WEEK_ROWS:
        row = rows.loc[metric]
        if metric == "baseline_erosion_share":
            value = f"{percent(100 * row['mean'])} (mean ratio)"
            spread = f"{percent(100 * row['p10'])} to {percent(100 * row['p90'])}"
        else:
            value = money(row["p50"])
            spread = f"{money(row['p10'])} to {money(row['p90'])}"
        table.append({"Whole week": label, "Value": value, "P10–P90": spread})
    return pd.DataFrame(table)


def _baseline_erosion(st: Any, kpis: pd.DataFrame, evidence: str) -> None:
    """Decision 0004 item 67: the settled value against the default baseline and against
    one learnt from smart nights, from ``trading_kpis`` (``full`` rows; the figures are the
    same on every strategy). The headline is the 16:00–20:00 evening, where smart charging
    moves load (lead ruling on item 67); the whole-week figures sit in an expander because
    outside the evening the smart-night baseline is mostly noise. Absent on a result built
    before item 67."""

    rows = kpis.loc[kpis["strategy"].eq("full")].set_index("metric")
    if "evening_baseline_erosion_share" not in rows.index:
        return
    default = rows.loc["evening_settled_value_gbp_per_week"]
    eroded = rows.loc["evening_settled_value_eroded_gbp_per_week"]
    share = rows.loc["evening_baseline_erosion_share"]

    def spread(row: pd.Series) -> str:
        return f"P10–P90 {money(row['p10'])} to {money(row['p90'])} · illustrative"

    st.markdown(
        "**If the baseline learnt from smart nights**",
        help="The default baseline learns only from unmanaged nights, as BL01 leaves out days "
        "when flex was delivered, so it never sees a flexed night. Baseline erosion is what "
        "happens if the baseline instead rolls over the fleet's own smart nights: the shifted "
        "evening import becomes the new normal and the turn-down stops settling. The "
        "smart-night baseline applies the same rule (same day type, up to five weekday or two "
        "weekend nights, the same in-day adjustment) to the week's other smart nights, "
        "standing in for a history that is all flexed. Settled turn-down is valued at the "
        "day-ahead price, 16:00–20:00 London, per simulated week. This is a what-if beside "
        "the ledger: nothing in the trading P&L uses it.",
    )
    share_tile, default_tile, eroded_tile = kpi_columns(st, 3)
    kpi(
        share_tile,
        "Evening value lost",
        percent(100 * share["mean"], decimals=0),
        context="To baseline erosion, 16:00–20:00, mean across weeks",
        help="One minus the ratio of the mean evening settled values (smart-night ÷ "
        "default). Near 100% means a baseline that learns from flexed nights stops paying for "
        "the evening shift almost entirely.",
    )
    kpi(
        default_tile,
        "Evening value, default",
        money(default["p50"]),
        context=spread(default),
        help="Evening settled turn-down every strategy is paid on, valued at the day-ahead "
        "price, against the default baseline learnt from unmanaged nights. P50 across weeks.",
    )
    kpi(
        eroded_tile,
        "Evening value, smart nights",
        money(eroded["p50"]),
        context=spread(eroded),
        help="The same, if the baseline had learnt from the fleet's own smart nights.",
    )
    st.caption(
        "Default: unmanaged nights only, like BL01. Erosion: value lost if it learnt from smart "
        f"nights. {evidence}"
    )
    with st.expander("Whole week"):
        st.dataframe(_erosion_week_table(rows), width="stretch", hide_index=True)
        st.caption(
            "Outside the evening the smart-night baseline learns from few nights of a 7-night "
            "study, so its noise adds spurious settlement."
        )


def _newsvendor_caption(result: Any) -> str | None:
    """The newsvendor-mode note for the position chart, or ``None`` when it does not apply.

    Intraday-dispatch-v1 §2 (lead decision Q3): under the fixed-share rule one
    commitment share ``c`` sets both the day-ahead position (``c`` x forecast
    turn-down) and the share of EVs locked to their day-ahead plan. Under the
    newsvendor rule the position comes from the error quantile instead, so
    the two come apart and the chart would otherwise invite reading the
    position as ``c`` of the forecast. Read from the run's own records
    (``result.assumptions`` and ``dispatch_split``); ``dispatch_split`` is
    ``None`` when the dispatch did not run, and then nothing is locked.
    """

    if assumption_value(result, "trading.commitment_rule") != "newsvendor":
        return None
    split = getattr(result, "dispatch_split", None)
    if split is None:
        return None
    share = percent(100.0 * float(split["commitment_share"].iat[0]), decimals=0)
    locked = int(split["locked_count"].sum())
    total = int(split["ev_count"].sum())
    return (
        f"Newsvendor: {locked:,} of {total:,} EVs locked (share {share}). The position is sized "
        f"like stock for uncertain demand, not {share} of the forecast."
    )


def render_trading_position(st: Any, result: Any) -> None:
    """Render the Position lens: night KPIs, the night profile, the position fan and the
    per-night settled-volume band across simulated weeks."""

    if _guard(st, result):
        return

    night_index, strategy = _controls(st, result.study_slots)
    representative = result.representative_world_id
    night = _night_slots(result.deviation_world_slot, representative, night_index)
    evidence = f"Evidence: {_evidence_text(result.evidence_kind)}."

    _night_kpis(st, night, strategy)

    chart_block(
        st,
        _night_figure(night),
        title="Baseline, unmanaged and smart import through one night",
        caption=f"One session night, representative week. Shaded: settled turn-down, baseline "
        f"down to the smart line. {evidence}",
        frame=_night_table(night),
        definition=(
            "The shaded band is settled turn-down: the half-hours inside the settlement window "
            "where the baseline sits above what the fleet actually drew. The baseline is the "
            "import the fleet would have had with no flexibility, estimated from unmanaged "
            "nights the way the industry's BL01 method does (a simplified version). It is a "
            "reference for measuring flexibility, not a physical path, so a driver's own cheap "
            "charging never counts as settled flexibility. All three trading strategies see "
            "the same baseline, unmanaged and metered paths; only their day-ahead and intraday "
            "decisions differ."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="trading-position-night-chart",
    )
    st.caption(
        "Baseline: the import the fleet would have had with no flexibility, estimated from "
        "unmanaged nights (a BL01-style method)."
    )

    chart_block(
        st,
        _position_figure(night, strategy),
        title="Day-ahead position and the final position, with the intraday close",
        caption=(
            f"One session night, representative simulated week, {STRATEGY_LABELS[strategy]} "
            f"strategy. {evidence}"
        ),
        frame=_position_table(night, strategy),
        definition=(
            "The day-ahead position is the turn-down sold the day before delivery: a share of "
            "the forecast deviation. The final position is what this strategy holds after any "
            "intraday re-trading. Day-ahead only never re-trades. Perfect foresight already "
            "knows the settled volume, so it sells exactly that. The intraday close is the "
            "last price traded before each half-hour, shown for reference: the day-ahead plan "
            "could not see it."
        ),
        height=CHART_HEIGHTS["time_series_dual_axis"],
        key="trading-position-fan",
    )
    newsvendor = _newsvendor_caption(result)
    if newsvendor is not None:
        st.caption(newsvendor)

    chart_block(
        st,
        _open_position_figure(result.open_position_profile),
        title="Open position by half-hour, across simulated weeks",
        caption=f"Median across simulated weeks, every strategy. {evidence}",
        frame=_open_position_table(result.open_position_profile),
        definition=(
            "Open position is settled turn-down minus the final position, in signed kW. "
            "Positive: the fleet delivered more than this strategy sold, and the extra settles "
            "at the imbalance price. Negative: it sold more than it delivered. Perfect "
            "foresight is exactly 0 at every half-hour, because it sells exactly what settles."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="trading-position-open",
    )

    nights = _settled_by_night(result.trading_ledger_summary, result.study_slots)
    chart_block(
        st,
        _settled_figure(nights),
        title="Settled turn-down per night, across simulated weeks",
        caption=f"P10–P90 across {int(nights['world_count'].iloc[0])} simulated weeks. {evidence}",
        frame=_settled_table(nights),
        definition=(
            "Settled turn-down is the same physical volume for every trading strategy (only "
            "the money each strategy makes from it differs, shown on the P&L and risk lens), "
            "so this band needs no strategy picker."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="trading-position-settled",
    )

    _baseline_erosion(st, result.trading_kpis, evidence)
