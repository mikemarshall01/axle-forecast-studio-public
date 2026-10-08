"""Trading lens 1, Market: the synthetic prices the trading overlay trades on.

Reads ``forecast_price_bands`` (day-ahead P10/P50/P90 across simulated weeks,
trading contract v1 section 5, results-v2 4.7), ``forecast_prices`` and
``evaluation_prices`` (the representative simulated week's intraday close and
imbalance price, trading contract v1 section 1.4), ``market_shocks`` and
``shock_summary`` (the stochastic and scripted net-demand shocks, trading
contract v1 section 5.6a). Every price is synthetic and every trading figure
is illustrative (decision 0005); this lens never settles anything, it only
shows the market the overlay on lenses 2 and 3 reacts to.

``STRATEGY_LABELS`` and the evidence-kind wording are defined here and
imported by ``trading_position`` and ``trading_pnl`` (the "1 Market" lens
owns the shared text-mapping helpers for the whole page, mirroring how
``action_decision`` owns them for the Smart charging lenses).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from streamlit import column_config as st_column_config

from ..components.chart_table import chart_block
from ..components.kpi import kpi, kpi_columns
from ..style import (
    ARCHETYPE_COLOURS,
    CHART_HEIGHTS,
    MUTED_INK,
    SERIES_COLOURS,
    band_and_line,
    format_quantity,
    london_time_axis,
    percent,
)

STRATEGY_LABELS: dict[str, str] = {
    "da_only": "Day-ahead only",
    "full": "Full (day-ahead + intraday)",
    "perfect_foresight": "Perfect foresight",
}
"""Display name for each internal trading strategy id (``market.STRATEGIES``
order). Shared by every Trading lens, so the three strategies are always
named the same way."""

STRATEGY_ORDER: tuple[str, ...] = ("da_only", "full", "perfect_foresight")
"""Table and chart column order (trading contract v1 ``STRATEGIES`` order)."""

STRATEGY_COLOURS: dict[str, str] = {
    "da_only": SERIES_COLOURS["difference"],
    "full": ARCHETYPE_COLOURS[3],
    "perfect_foresight": SERIES_COLOURS["synthetic"],
}
"""One colour per strategy, shared by the Position and P&L lenses: "day-ahead
only" in amber (a simpler, partial strategy), "perfect foresight" in violet
(an idealised counterfactual, not a real strategy, the same role violet
plays for the synthetic day-ahead price) and "full" in one muted archetype
hue (dashboard_lint's teal-reserved rule: teal is reserved sitewide for the
physical smart-charging dispatch path, decision 0004 item 22; a trading
strategy is a different thing even when it is named "full", so it takes a
colour from the same safe, non-reserved palette the six driver archetypes
share, rather than borrowing teal's meaning)."""

STRATEGY_CAPTIONS: dict[str, str] = {
    "da_only": (
        "Sold day-ahead and did not hedge the fleet's intraday moves: the dispatch's "
        "deviation from the day-ahead position is settled at the imbalance price"
    ),
    "full": "The product: day-ahead position, hourly re-positioning on the dispatched fleet",
    "perfect_foresight": (
        "Perfect foresight of volume, not of intraday prices: sells the settled deviation "
        "of the dispatched fleet at the day-ahead price and never trades intraday"
    ),
}
"""What each strategy's row means when intraday dispatch is on (intraday-dispatch-v1 §6.2, Q8).

Mirrors ``model.market.STRATEGY_CAPTIONS`` word for word: no Trading lens
imports the model (``test_view_makes_no_model_calls``), so the wording is
copied here rather than shared by import; keep the two in sync by hand."""

PERFECT_FORESIGHT_CAPTION = "against perfect foresight of volume, not of intraday prices"
"""Mirrors ``model.market.PERFECT_FORESIGHT_CAPTION`` (intraday-dispatch-v1 §6.2, B3): with
intraday dispatch on, ``full`` earns re-optimisation P&L that ``perfect_foresight`` (never
trades intraday) cannot, so capture above 100% and a negative cost of uncertainty can occur."""
PERFECT_FORESIGHT_CAPTION_METRICS = ("capture_rate", "cost_of_uncertainty_gbp_per_week")
"""The ``trading_kpis`` metrics that carry ``PERFECT_FORESIGHT_CAPTION`` (mirrors
``model.market.PERFECT_FORESIGHT_CAPTION_METRICS``)."""

_EVIDENCE_TEXT = {
    "illustrative_synthetic": "illustrative, synthetic",
    "synthetic": "synthetic",
    "illustrative": "illustrative",
}


def _evidence_text(kind: object) -> str:
    """Plain words for a result's ``evidence_kind``, never a raw slug on screen."""

    text = str(kind)
    return _EVIDENCE_TEXT.get(text, text.replace("_", ", "))


def _guard(st: Any, result: Any) -> bool:
    """``True`` and an info message when this result has no Axle action.

    ``pages._action_lens`` already filters a no-action result before this
    view is reached; this only keeps a direct call (a test, a future caller)
    safe, the same pattern ``action_decision`` uses.
    """

    if result.action_summary is None:
        st.info("This result has no Axle action.")
        return True
    return False


def _negative_price_share(forecast_prices: pd.DataFrame) -> float:
    """Share of all simulated day-ahead half-hours priced below £0/MWh.

    A flat share over every world and slot together (not a per-world
    quantile): a single realism check at a glance, not a spread across
    weeks, so it needs no world-first aggregation before it is reported.
    """

    prices = forecast_prices["wholesale_forecast_gbp_per_mwh"].to_numpy(dtype=float)
    return float((prices < 0.0).mean())


def _kpi_row(st: Any, result: Any) -> None:
    """Tiles: negative-price share, day-ahead spread, and this week's shock count."""

    spread = result.trading_kpis.loc[
        result.trading_kpis["strategy"].eq("full")
        & result.trading_kpis["metric"].eq("day_ahead_spread_gbp_per_mwh")
    ].iloc[0]
    representative = result.representative_world_id
    shocks = result.market_shocks.loc[
        result.market_shocks["world_id"].eq(representative) & result.market_shocks["in_study"]
    ]
    known_count = int(shocks["known_day_ahead"].sum())
    surprise_count = len(shocks) - known_count

    negative_column, spread_column, shock_column = kpi_columns(st, 3)
    kpi(
        negative_column,
        "Negative day-ahead prices",
        percent(100 * _negative_price_share(result.forecast_prices), decimals=1),
        context="Share of all simulated day-ahead half-hours below £0/MWh",
        help="A realism check: in Great Britain the day-ahead price goes negative in a few per "
        "cent of half-hours. These prices are illustrative and synthetic, not market data.",
    )
    kpi(
        spread_column,
        "Day-ahead spread",
        format_quantity(spread["p50"], "£/MWh", decimals=0),
        context=(
            f"P10–P90 {format_quantity(spread['p10'], '£/MWh', decimals=0)} to "
            f"{format_quantity(spread['p90'], '£/MWh', decimals=0)}"
        ),
        help="Each session night's highest day-ahead price minus its lowest, then the mean "
        "over the nights. It is the same for every trading strategy: a property of the "
        "market, not a trading choice.",
    )
    kpi(
        shock_column,
        "Shocks this week",
        str(len(shocks)),
        context=f"{known_count} known day-ahead, {surprise_count} surprise",
        help="Random and scripted net-demand shocks that start in the representative "
        "simulated week (the median week). Other simulated weeks draw their own shocks.",
    )


def _publication_times(result: Any, prices: pd.DataFrame) -> list[pd.Timestamp]:
    """Day-ahead publication instants inside the chart's range, from the result.

    Each day's prices are published at one London clock time the day before
    delivery; the result carries that instant per slot, so the view only
    picks the distinct ones inside the plotted range.
    """

    start, end = prices["interval_start_utc"].iat[0], prices["interval_end_utc"].iat[-1]
    published = pd.Series(result.forecast_prices["forecast_available_at_utc"].unique())
    return sorted(published.loc[(published >= start) & (published < end)])


def _shock_marks(figure: go.Figure, shocks: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Shade each in-study shock's window and mark its start (trading contract v1 5.6a).

    Known day-ahead shocks are shaded in the day-ahead price's own violet, at
    low opacity, because they already move that line; surprise shocks are
    outlined in dashed red instead of shaded, because they move only the
    intraday and imbalance lines the day-ahead fan never sees. A true hatch
    fill was tried (Plotly ``fillpattern``) but is not a valid shape property
    in the installed Plotly version, so "outlined, not shaded" carries the
    known/surprise distinction instead. No per-shock text label: a week can
    hold a dozen or more mild shocks (checked on a 20-simulated-week run),
    and Plotly does not stagger overlapping shape annotations, so labelling
    every shock made the strip unreadable; class, direction and size stay in
    the hover text and the shock table below instead.
    """

    if shocks.empty:
        return
    by_slot = prices.set_index("slot_index")
    half_hour = pd.Timedelta(minutes=30)
    for known, dash in ((True, None), (False, "dot")):
        group = shocks.loc[shocks["known_day_ahead"].eq(known)]
        if group.empty:
            continue
        for _, shock in group.iterrows():
            x0 = shock["start_utc"]
            x1 = x0 + shock["duration_slots"] * half_hour
            if known:
                figure.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=SERIES_COLOURS["synthetic"],
                    opacity=0.14,
                    line_width=0,
                    layer="below",
                )
            else:
                figure.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=SERIES_COLOURS["flag"],
                    opacity=0.10,
                    line_width=1,
                    line_dash=dash,
                    line_color=SERIES_COLOURS["flag"],
                    layer="below",
                )
        # One marker trace per group, at the price line each kind of shock
        # actually moves (known shocks move day-ahead; surprises move only
        # intraday and imbalance, so a marker on the day-ahead line would
        # misstate what it did): this also carries the legend entry and the
        # hover text, since a shape alone has neither.
        marks = group.loc[group["start_slot_index"].isin(by_slot.index)]
        if marks.empty:
            continue
        y = by_slot.loc[marks["start_slot_index"], "p50"].to_numpy()
        symbols = ["triangle-up" if d == "up" else "triangle-down" for d in marks["direction"]]
        figure.add_trace(
            go.Scatter(
                x=marks["start_utc"],
                y=y,
                mode="markers",
                marker={
                    "symbol": symbols,
                    "size": 10,
                    "color": SERIES_COLOURS["synthetic"] if known else SERIES_COLOURS["flag"],
                    "line": {"color": MUTED_INK, "width": 1},
                },
                name="Known day-ahead shock" if known else "Surprise shock",
                customdata=marks.loc[:, ["shock_class", "direction", "size_gw"]].to_numpy(),
                hovertemplate=(
                    "%{customdata[0]} %{customdata[1]} shock, %{customdata[2]:.1f} GW"
                    "<extra></extra>"
                ),
            )
        )


def _price_figure(
    prices: pd.DataFrame,
    evaluation: pd.DataFrame,
    shocks: pd.DataFrame,
    publications: list[pd.Timestamp],
) -> go.Figure:
    """Day-ahead fan, the representative week's intraday close and imbalance price."""

    figure = go.Figure()
    band_and_line(
        figure,
        prices["interval_start_utc"],
        prices["p10"],
        prices["p50"],
        prices["p90"],
        name="Day-ahead (synthetic)",
        colour=SERIES_COLOURS["synthetic"],
        hovertemplate="%{y:,.0f} £/MWh<extra></extra>",
    )
    figure.add_trace(
        go.Scatter(
            x=evaluation["interval_start_utc"],
            y=evaluation["evaluation_context_price_gbp_per_mwh"],
            mode="lines",
            name="Intraday close (representative week)",
            line={"color": SERIES_COLOURS["observed"], "width": 1.5, "dash": "dash"},
            hovertemplate="%{y:,.0f} £/MWh<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=evaluation["interval_start_utc"],
            y=evaluation["imbalance_price_gbp_per_mwh"],
            mode="lines",
            name="Imbalance price (representative week)",
            line={"color": SERIES_COLOURS["flag"], "width": 1.5},
            hovertemplate="%{y:,.0f} £/MWh<extra></extra>",
        )
    )
    _shock_marks(figure, shocks, prices)
    for index, published in enumerate(publications):
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
        **london_time_axis(prices["interval_start_utc"], compact=True),
    )
    figure.update_yaxes(title="£/MWh")
    return figure


def _price_table(prices: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    price_rounded = prices.loc[
        :, ["interval_start_utc", "interval_start_london", "p10", "p50", "p90"]
    ].round({"p10": 0, "p50": 0, "p90": 0})
    other = evaluation.loc[
        :,
        [
            "interval_start_utc",
            "evaluation_context_price_gbp_per_mwh",
            "imbalance_price_gbp_per_mwh",
        ],
    ].round({"evaluation_context_price_gbp_per_mwh": 0, "imbalance_price_gbp_per_mwh": 0})
    merged = price_rounded.merge(other, on="interval_start_utc", how="left")
    return merged.rename(
        columns={
            "interval_start_utc": "Interval start (UTC)",
            "interval_start_london": "Interval start (London)",
            "p10": "Day-ahead price P10 (£/MWh)",
            "p50": "Day-ahead price, median (£/MWh)",
            "p90": "Day-ahead price P90 (£/MWh)",
            "evaluation_context_price_gbp_per_mwh": "Intraday close, representative week (£/MWh)",
            "imbalance_price_gbp_per_mwh": "Imbalance price, representative week (£/MWh)",
        }
    )


_SHOCK_CLASS_LABELS = {"mild": "Mild", "big": "Big", "scripted": "Scripted"}


# One decimal place per column, so "4" and "3.8" in one column do not read
# as different precisions (st.dataframe drops trailing zeros by default).
_SHOCK_TABLE_FORMATS = {
    name: st_column_config.NumberColumn(format="%.1f")
    for name in (
        "Shocks per week, P10",
        "Shocks per week, median",
        "Shocks per week, P90",
        "Typical size (GW)",
    )
} | {"Typical peak price move (£/MWh)": st_column_config.NumberColumn(format="%.0f")}


def _shock_table(shock_summary: pd.DataFrame) -> pd.DataFrame:
    """``shock_summary`` as a table, not summed tiles.

    Each row's ``count_per_week_*`` is already a per-group quantile across
    simulated weeks (trading contract v1 5.6a); summing those across shock
    classes would sum percentiles, so classes stay in separate rows rather
    than being combined into one figure.
    """

    return pd.DataFrame(
        {
            "Class": shock_summary["shock_class"].map(_SHOCK_CLASS_LABELS),
            "Timing": shock_summary["known_day_ahead"].map(
                {True: "Known day-ahead", False: "Surprise"}
            ),
            "Direction": shock_summary["direction"].str.capitalize(),
            "Shocks per week, P10": shock_summary["count_per_week_p10"].round(1),
            "Shocks per week, median": shock_summary["count_per_week_p50"].round(1),
            "Shocks per week, P90": shock_summary["count_per_week_p90"].round(1),
            "Typical size (GW)": shock_summary["size_gw_p50"].round(1),
            "Typical peak price move (£/MWh)": shock_summary[
                "peak_price_increment_gbp_per_mwh_p50"
            ].round(0),
        }
    )


def render_trading_market(st: Any, result: Any) -> None:
    """Render the Market lens: price KPIs, the price chart with shocks, the shock table."""

    if _guard(st, result):
        return

    _kpi_row(st, result)

    representative = result.representative_world_id
    prices = result.forecast_price_bands.sort_values("slot_index")
    evaluation = result.evaluation_prices.loc[
        result.evaluation_prices["world_id"].eq(representative)
    ].sort_values("slot_index")
    shocks = result.market_shocks.loc[
        result.market_shocks["world_id"].eq(representative) & result.market_shocks["in_study"]
    ]
    evidence = f"Evidence: {_evidence_text(result.evidence_kind)}."

    chart_block(
        st,
        _price_figure(prices, evaluation, shocks, _publication_times(result, prices)),
        title="Day-ahead, intraday and imbalance prices through the week",
        caption=(
            "Day-ahead median and P10–P90 across weeks; intraday and imbalance for the "
            f"representative week only. {evidence}"
        ),
        frame=_price_table(prices, evaluation),
        definition=(
            "Illustrative synthetic prices, not market data: a daily curve plus random "
            "day-ahead noise, so each simulated week has its own day-ahead path. Three prices "
            "matter to the trading desk, which reads what the fleet did and sells its "
            "turn-down without changing how the EVs charge. The day-ahead price is set the "
            "afternoon before "
            "delivery. The intraday close is the last price traded before a half-hour starts. "
            "The imbalance price is what any gap between what was sold and what was delivered "
            "settles at afterwards. The intraday close and imbalance price are shown for the "
            "representative week only (the median simulated week), because each week has its "
            "own realised path, not a forecast fan across weeks. The dotted vertical lines "
            "mark 13:00 each day, when the next day's day-ahead prices come out; a smart "
            "charging plan only ranks prices already published. Shaded and outlined spans "
            "mark net-demand shocks that start in the representative week. Known day-ahead "
            "shocks are shaded and already inside the day-ahead price. Surprise shocks are "
            "outlined and move only the intraday and imbalance prices. A marker shows each "
            "shock's direction where it lands."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="trading-market-price",
    )
    st.caption(
        "Shaded spans: shocks known a day ahead (in the day-ahead price); dotted red outline: "
        "surprises (in intraday and imbalance only)."
    )

    st.markdown(
        "**Shocks per simulated week, by class and timing**",
        help=(
            "Each row is that group's shocks per week: P10, P50 and P90 across every simulated "
            "week, with a week that had none counting as 0. Adding the rows would sum "
            "percentiles, so no combined total is shown."
        ),
    )
    st.dataframe(
        _shock_table(result.shock_summary),
        width="stretch",
        hide_index=True,
        column_config=_SHOCK_TABLE_FORMATS,
    )
    st.caption(
        "P50 with P10–P90 across simulated weeks. A week with no shocks of a kind counts as 0."
    )
