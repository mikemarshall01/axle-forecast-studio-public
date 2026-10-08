"""Trading lens 3, P&L and risk: the illustrative ledger and trader metrics.

Reads ``trading_ledger_summary`` (the nine ledger buckets and their net,
mean across simulated weeks, trading contract v1 section 5.3-5.5),
``trading_kpis`` (the headline and trader metrics per strategy, section
5.5 and 9.1) and ``trading_checks`` (the runtime sanity checks, section
5.5). The three strategies (day-ahead only, full, perfect foresight) come
from one physics run, so they are directly comparable on the same random
futures (decision 0004 item 57).

Every £ here is "illustrative simulated trading P&L" (decision 0005), never
"Axle cash": no view sums it with the customer energy-cost effect shown on
Smart charging, because both come from the same shifted energy and would
double-count it.

Intraday dispatch (intraday-dispatch-v1 §6.3, §10): the ledger's rebalancing
vs re-optimisation split of intraday P&L and trading cost, the re-optimisation
P&L and energy-moved KPI tiles, and the strategy captions all read the same
``trading_kpis`` / ``trading_ledger_summary`` frames every other row here
reads; they are 0 with the intraday dispatch switch off, not absent, since
the frozen-book re-run then equals the actual book (§6.2). ``STRATEGY_CAPTIONS``
and ``PERFECT_FORESIGHT_CAPTION`` are copied here from ``model.market``
word for word rather than imported: no Trading lens imports the model
(``test_view_makes_no_model_calls``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from ..components.chart_table import chart_block
from ..components.kpi import kpi, kpi_columns
from ..style import (
    MUTED_INK,
    SERIES_COLOURS,
    UNAVAILABLE,
    assumption_value,
    format_quantity,
    money,
    percent,
)
from .trading_market import (
    PERFECT_FORESIGHT_CAPTION,
    PERFECT_FORESIGHT_CAPTION_METRICS,
    STRATEGY_CAPTIONS,
    STRATEGY_COLOURS,
    STRATEGY_LABELS,
    STRATEGY_ORDER,
    _evidence_text,
    _guard,
)

_BUCKET_LABELS = {
    "day_ahead_revenue_gbp": "Day-ahead revenue on flexibility",
    "intraday_pnl_gbp": "Intraday P&L",
    "trading_cost_gbp": "Trading cost",
    "imbalance_gbp": "Imbalance settlement",
    "baseline_effect_gbp": "Baseline effect",
    "grid_event_payment_gbp": "Grid-event payments",
    "supplier_compensation_gbp": "Supplier compensation",
    "customer_revenue_share_gbp": "Customer revenue share",
    "unmet_charge_penalty_gbp": "Unmet-charge penalty",
    "net_gbp": "Net (illustrative)",
}
_BUCKETS_WITH_NET = tuple(_BUCKET_LABELS)
"""The nine decision-0005 buckets, then their net, in ledger order."""
_BUCKET_SHORT_LABELS = {
    "day_ahead_revenue_gbp": "Day-ahead revenue",
    "intraday_pnl_gbp": "Intraday P&L",
    "trading_cost_gbp": "Trading cost",
    "imbalance_gbp": "Imbalance",
    "baseline_effect_gbp": "Baseline effect",
    "grid_event_payment_gbp": "Grid events",
    "supplier_compensation_gbp": "Supplier comp.",
    "customer_revenue_share_gbp": "Customer share",
    "unmet_charge_penalty_gbp": "Unmet-charge penalty",
    "net_gbp": "Net (illustrative)",
}
"""Axis labels short enough to sit beside the bars at 390 px (final critique B-15);
the full ``_BUCKET_LABELS`` name is in the hover and the table."""
_BAR_PX = 12
"""Height per bar: ten buckets × three strategies, so the chart grows with the rows."""
_LEDGER_HEIGHT = len(_BUCKETS_WITH_NET) * 3 * _BAR_PX + 120


def _kpi_rows(st: Any, full: pd.DataFrame, *, dispatch_active: bool) -> None:
    """Two KPI rows (at most four tiles each, polish plan G5): the ``full`` strategy's
    headline trader metrics against perfect foresight and day-ahead only."""

    net = full.loc["net_gbp_per_week"]
    capture = full.loc["capture_rate"]
    value_of_intraday = full.loc["value_of_intraday_gbp_per_week"]
    firmness = full.loc["firmness"]

    net_column, capture_column, value_column, firmness_column = kpi_columns(st, 4)
    kpi(
        net_column,
        "Net P&L, full strategy",
        money(net["p50"], signed=True),
        context=f"P10 {money(net['p10'], signed=True)} · illustrative",
        # The tile is the P50; the help said "mean" (final critique B-3).
        help="Illustrative simulated trading P&L, never Axle cash: the ledger's nine buckets "
        "summed inside each week, then the median (P50) and 10th percentile (P10) across "
        "simulated weeks.",
    )
    kpi(
        capture_column,
        "Capture rate",
        # One decimal and no clip (model questions Q-3): above 100% is real
        # imbalance P&L, since the benchmark knows the volume but not the
        # imbalance price; the margin difference is Σ(V − q)(SIP − P).
        percent(100 * capture["p50"], decimals=1),
        context="Trading margin vs perfect volume foresight",
        help="Capture rate is how much of the best possible margin the full strategy actually "
        "took: its trading margin on flexibility (after the spread cost) divided by the margin "
        "of perfect volume foresight sold day-ahead, over weeks where that margin is positive. "
        "It can pass 100%: perfect foresight knows the volume but not the imbalance price, so "
        "leaving some volume to imbalance earns more in weeks when the imbalance price runs "
        "above day-ahead."
        + (
            # Intraday dispatch contract v1 §6.2 (B3): a second, dispatch-
            # specific route past 100% opens once the fleet itself trades
            # intraday on price moves, on top of the imbalance-price route
            # above; this route is 0 with the switch off.
            f" Measured {PERFECT_FORESIGHT_CAPTION}: perfect foresight never trades intraday, "
            "so the full strategy can also earn more from the dispatch's own re-optimisation."
            if dispatch_active
            else ""
        ),
    )
    kpi(
        value_column,
        "Value of intraday",
        money(value_of_intraday["p50"], signed=True),
        # "illustrative" on every money tile (final critique B-20, money rule).
        context="Full minus day-ahead only · illustrative",
        help="How much re-trading intraday adds to the trading margin: full minus day-ahead "
        "only, paired inside each simulated week, then the P50. Negative is expected while "
        "the day-ahead to intraday premium is 0, because re-trading then pays the spread "
        "without a price edge. Illustrative.",
    )
    kpi(
        firmness_column,
        "Firmness",
        percent(100 * firmness["p50"], decimals=0),
        context="Delivered within the final position",
        help="Share of the final committed turn-down actually delivered by the fleet.",
    )

    cvar_column, mw_column, reopt_column, moved_column = kpi_columns(st, 4)
    kpi(
        cvar_column,
        "Net P&L, CVaR5",
        money(net["cvar5"], signed=True),
        context="Mean of the worst 5% of weeks · illustrative",
        help="CVaR5 is the mean net P&L over the worst 5% of simulated weeks (those at or "
        "below their own 5th percentile): what a bad week looks like, not a typical one.",
    )
    per_mw_year = full.loc["flex_margin_gbp_per_mw_year"]
    kpi(
        mw_column,
        "Margin per MW per year",
        # money() for the £ sign and the true minus, as Key stats F (B-20).
        money(per_mw_year["p50"], signed=True),
        unit="/MW/yr",
        context="Per MW that can wait 2 h at 19:00; from one simulated week · illustrative",
        help="Trading margin on flexibility scaled to a year, divided by the fleet's own "
        "deferrable power at 19:00 (charging that could wait at least 2 hours). That capacity "
        "is a property of the fleet, not of the strategy.",
    )
    # Intraday dispatch contract v1 §6.3, §10: the two dispatch KPI rows, on
    # the full strategy only; both are exactly 0 with the switch off, since
    # the frozen-book re-run then equals the actual book and the dispatch
    # moves nothing (§6.2).
    reoptimisation = full.loc["reoptimisation_gbp_per_week"]
    kpi(
        reopt_column,
        "Re-optimisation P&L",
        money(reoptimisation["p50"], signed=True),
        context=(
            f"Before trading cost and imbalance · P10 {money(reoptimisation['p10'], signed=True)} "
            f"· CVaR5 {money(reoptimisation['cvar5'], signed=True)} · illustrative"
        ),
        help="With intraday dispatch on, intraday trading splits in two. Rebalancing is the "
        "trades the forecast updates alone would have caused. Re-optimisation is the extra "
        "the dispatch's own response caused, at the traded intraday prices; 0 with intraday "
        "dispatch off. Shown before trading cost and imbalance, and not the dispatch's whole "
        "value: it leaves out the changed settled volume, its imbalance settlement and the "
        "drivers' cost at the day-ahead price. A Compare run with the switch off shows the "
        "whole value.",
    )
    moved = full.loc["dispatch_moved_mwh_per_week"]
    kpi(
        moved_column,
        "Energy moved by dispatch",
        format_quantity(moved["p50"], "MWh", decimals=2),
        context="Per week, P50 · illustrative",
        help="Energy the intraday dispatch moved out of its day-ahead half-hour, summed per "
        "week (net per half-hour); 0 with intraday dispatch off.",
    )


def _week_rows(ledger_summary: pd.DataFrame) -> pd.DataFrame:
    """The week rows of ``trading_ledger_summary`` (``night_index`` is ``<NA>``)."""

    return ledger_summary.loc[ledger_summary["night_index"].isna()]


def _bucket_figure(ledger_summary: pd.DataFrame) -> go.Figure:
    """Grouped horizontal bars: each bucket's mean £ per week, one bar per strategy.

    Horizontal, with short bucket names on the y axis, because ten rotated
    names on an x axis overlapped into one unreadable block at 390 px
    (final critique B-15; chart-design: long categories go on the y axis).
    """

    week = _week_rows(ledger_summary)
    labels = [_BUCKET_SHORT_LABELS[bucket] for bucket in _BUCKETS_WITH_NET]
    full_names = [_BUCKET_LABELS[bucket] for bucket in _BUCKETS_WITH_NET]
    figure = go.Figure()
    for strategy in STRATEGY_ORDER:
        rows = week.loc[week["strategy"].eq(strategy)].set_index("metric")
        y = [
            float(rows.loc[bucket, "mean"]) if bucket in rows.index else 0.0
            for bucket in _BUCKETS_WITH_NET
        ]
        figure.add_trace(
            go.Bar(
                x=y,
                y=labels,
                orientation="h",
                name=STRATEGY_LABELS[strategy],
                marker_color=STRATEGY_COLOURS[strategy],
                customdata=full_names,
                hovertemplate=(
                    f"{STRATEGY_LABELS[strategy]}, %{{customdata}}: £%{{x:,.0f}} per week"
                    "<extra></extra>"
                ),
            )
        )
    figure.update_layout(barmode="group", hovermode="closest")
    # Ledger order top to bottom, net last.
    figure.update_yaxes(
        title={"text": "bucket (£ per week)", "standoff": 12},
        autorange="reversed",
    )
    figure.update_xaxes(
        title="£ per week (mean, illustrative)", zeroline=True, zerolinecolor=MUTED_INK
    )
    return figure


def _bucket_table(ledger_summary: pd.DataFrame) -> pd.DataFrame:
    week = _week_rows(ledger_summary)
    rows = []
    for bucket in _BUCKETS_WITH_NET:
        row: dict[str, str] = {"Bucket": _BUCKET_LABELS[bucket]}
        for strategy in STRATEGY_ORDER:
            values = week.loc[week["strategy"].eq(strategy) & week["metric"].eq(bucket)]
            row[STRATEGY_LABELS[strategy]] = (
                money(values.iloc[0]["mean"], signed=True) if len(values) else UNAVAILABLE
            )
        rows.append(row)
    return pd.DataFrame(rows)


# --- Intraday dispatch split: rebalancing vs re-optimisation (contract §6.3, §10) ---
#
# The "of which" split of the full strategy's intraday P&L and trading cost:
# rebalancing is what the frozen-book re-run (the trader on the day-ahead
# plan path's book) would have traded at the same prices; re-optimisation is
# the extra the dispatch's own response caused. Both are exactly 0 on the
# other two strategies and with the intraday dispatch switch off, so this
# split is shown for the full strategy only.

_DISPATCH_SPLIT_ROWS = (
    ("Intraday P&L", "intraday_pnl_rebalancing_gbp", "intraday_pnl_reoptimisation_gbp"),
    ("Trading cost", "trading_cost_rebalancing_gbp", "trading_cost_reoptimisation_gbp"),
)
_DISPATCH_SPLIT_HEIGHT = len(_DISPATCH_SPLIT_ROWS) * 2 * _BAR_PX + 120
"""Same per-bar formula as ``_LEDGER_HEIGHT``: two rows, two bars each."""


def _dispatch_split_figure(ledger_summary: pd.DataFrame) -> go.Figure:
    """Intraday P&L and trading cost, full strategy, split rebalancing vs re-optimisation."""

    week = _week_rows(ledger_summary)
    full = week.loc[week["strategy"].eq("full")].set_index("metric")
    labels = [label for label, _, _ in _DISPATCH_SPLIT_ROWS]

    def means(column: int) -> list[float]:
        return [
            float(full.loc[row[column], "mean"]) if row[column] in full.index else 0.0
            for row in _DISPATCH_SPLIT_ROWS
        ]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=means(1),
            y=labels,
            orientation="h",
            name="Rebalancing",
            marker_color=SERIES_COLOURS["normal"],
            hovertemplate="Rebalancing: £%{x:,.0f} per week<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=means(2),
            y=labels,
            orientation="h",
            name="Re-optimisation",
            marker_color=SERIES_COLOURS["difference"],
            hovertemplate="Re-optimisation: £%{x:,.0f} per week<extra></extra>",
        )
    )
    figure.update_layout(barmode="group")
    figure.update_yaxes(title={"text": "bucket (£ per week)", "standoff": 12}, autorange="reversed")
    figure.update_xaxes(
        title="£ per week (mean, illustrative)", zeroline=True, zerolinecolor=MUTED_INK
    )
    return figure


def _dispatch_split_table(ledger_summary: pd.DataFrame) -> pd.DataFrame:
    week = _week_rows(ledger_summary)
    full = week.loc[week["strategy"].eq("full")].set_index("metric")

    def value(metric: str) -> str:
        if metric not in full.index:
            return UNAVAILABLE
        return money(full.loc[metric, "mean"], signed=True)

    return pd.DataFrame(
        [
            {"Bucket": label, "Rebalancing": value(reb), "Re-optimisation": value(reo)}
            for label, reb, reo in _DISPATCH_SPLIT_ROWS
        ]
    )


_METRIC_ROWS = (
    ("net_gbp_per_week", "Net P&L"),
    ("net_gbp_per_ev_year", "Net P&L per EV per year"),
    ("settled_mwh_per_week", "Settled turn-down"),
    ("imbalance_volume_share", "Imbalance share of settled volume"),
    ("baseline_effect_share", "Baseline-effect share of settled volume"),
    ("flex_margin_gbp_per_week", "Trading margin on flexibility"),
    ("capture_rate", "Capture rate vs perfect foresight"),
    ("value_of_intraday_gbp_per_week", "Value of intraday (full strategy only)"),
    ("cost_of_uncertainty_gbp_per_week", "Cost of uncertainty vs perfect foresight"),
    # Signed as the ledger, positive = income (final critique B-4).
    ("imbalance_gbp_per_mwh_traded", "Imbalance cash per MWh (+ income)"),
    ("imbalance_share_of_traded", "Imbalance share of the final position"),
    ("firmness", "Firmness (delivered ÷ committed)"),
    ("firmness_day_ahead", "Firmness vs the day-ahead commitment"),
    ("flex_margin_gbp_per_mw_year", "Trading margin per MW per year"),
    ("day_ahead_spread_gbp_per_mwh", "Day-ahead spread"),
)


def _format_metric(value: float, unit: str) -> str:
    """One display string per ``trading_kpis`` unit, missing values as "Unavailable"."""

    if pd.isna(value):
        return UNAVAILABLE
    if unit == "GBP per week":
        return money(value, signed=True)
    if unit == "GBP per EV per year":
        return money(value, decimals=2, signed=True)
    if unit == "MWh per week":
        return format_quantity(value, "MWh", decimals=2)
    if unit == "fraction":
        return percent(100 * value, decimals=1, signed=True)
    if unit == "GBP per MWh":
        return format_quantity(value, "£/MWh", decimals=1)
    if unit == "GBP per MW per year":
        return f"{money(value, signed=True)}/MW/yr"
    return format_quantity(value, "", decimals=2)


def _strategy_table(kpis: pd.DataFrame) -> pd.DataFrame:
    """One row per trader metric, one column per strategy, median across simulated weeks."""

    rows = []
    for metric, label in _METRIC_ROWS:
        group = kpis.loc[kpis["metric"].eq(metric)].set_index("strategy")
        row: dict[str, str] = {"Metric": label}
        for strategy in STRATEGY_ORDER:
            if strategy in group.index:
                value, unit = group.loc[strategy, ["p50", "unit"]]
                row[STRATEGY_LABELS[strategy]] = _format_metric(float(value), str(unit))
            else:
                row[STRATEGY_LABELS[strategy]] = UNAVAILABLE
        rows.append(row)
    return pd.DataFrame(rows)


_CHECK_LABELS = {
    "buckets_sum_to_net": "Ledger buckets sum to net",
    "perfect_foresight_zero_imbalance": "Perfect foresight has zero imbalance",
    "deviation_identity": "Deviation splits exactly into flexibility and baseline effect",
    "mean_intraday_revision": "Mean intraday revision (informative)",
    "p10_ordering": "P10 net ordering: perfect foresight, full, day-ahead only (informative)",
    "baseline_class_fallback": "Nights that used the other day type's baseline history",
    "baseline_missing_nights": "Nights with no baseline history to trade against",
    "dispatch_split_sums_to_intraday": "Rebalancing plus re-optimisation equals intraday P&L",
}


def _check_table(checks: pd.DataFrame) -> pd.DataFrame:
    """The sanity-check table (plan section F): human labels, never the raw check id."""

    rows = []
    for row in checks.itertuples():
        rows.append(
            {
                "Check": _CHECK_LABELS.get(row.check_id, row.check_id.replace("_", " ")),
                "Value": round(float(row.value), 4),
                "Tolerance": "" if pd.isna(row.tolerance) else round(float(row.tolerance), 4),
                "Passed": bool(row.passed),
            }
        )
    return pd.DataFrame(rows)


_CONTROL_METRIC_LABELS = {
    "estimated_bias_kw_per_ev": "Estimated bias (kW/EV)",
    "true_bias_kw_per_ev": "True bias (kW/EV)",
    "estimation_error_kw_per_ev": "Estimation error (kW/EV)",
}
"""Trading contract v1 section 9.5: the hold-out control group's baseline-bias
check. Estimated is what the control group alone suggests; true is the
treated group's actual bias, known only inside the simulation; the
difference is what a real supplier could never measure directly."""


def _control_group_table(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per night (then the week), the three bias metrics, kW per EV."""

    def add(rows: list[dict[str, str]], label: str, frame: pd.DataFrame) -> None:
        by_metric = frame.set_index("metric")
        row: dict[str, str] = {"Night": label}
        for metric, label_text in _CONTROL_METRIC_LABELS.items():
            value = by_metric.loc[metric, "p50"] if metric in by_metric.index else float("nan")
            row[label_text] = format_quantity(value, "", decimals=3)
        rows.append(row)

    nights = summary.loc[summary["night_index"].notna()].copy()
    nights["night_index"] = nights["night_index"].astype(int)
    rows: list[dict[str, str]] = []
    for night_index in sorted(nights["night_index"].unique()):
        add(rows, f"Night {night_index}", nights.loc[nights["night_index"].eq(night_index)])
    add(rows, "Week", summary.loc[summary["night_index"].isna()])
    return pd.DataFrame(rows)


def _control_group_caption(summary: pd.DataFrame) -> str:
    control = int(summary["control_ev_count"].iloc[0])
    treated = int(summary["treated_ev_count"].iloc[0])
    total = control + treated
    share = percent(100 * control / total, decimals=0) if total else UNAVAILABLE
    return f"{control} control EVs of {total} ({share}), held out from every plan."


def _firmness_table(firmness: pd.DataFrame) -> pd.DataFrame:
    """1 h turn-down firmness by maker (trading contract v1 §10.5g), one row per maker.

    P10 first: "firm" is the across-weeks P10 by ruling, and the makers differ
    there while their P50s are alike (final critique Q-17).
    """

    rows = firmness.loc[firmness["direction"].eq("turn_down") & firmness["duration_hours"].eq(1.0)]

    def shares(column: str) -> list[str]:
        return [UNAVAILABLE if pd.isna(value) else percent(100 * value) for value in rows[column]]

    return pd.DataFrame(
        {
            "Maker": rows["manufacturer_label"],
            "Firmness, P10 (firm)": shares("firmness_p10"),
            "Firmness, P50": shares("firmness_p50"),
        }
    )


_COMMITMENT_RULE_CAPTIONS = {
    "fixed_share": "Day-ahead position: a fixed share of the forecast turn-down.",
    "newsvendor": (
        "Day-ahead position: the newsvendor rule, a risk choice. It assumes unpaid spill and "
        "imbalance-priced shortfall, not the ledger's best cash."
    ),
}


def render_trading_pnl(st: Any, result: Any) -> None:
    """Render the P&L and risk lens: trader KPIs, the bucket bars, the strategy table
    and the sanity checks."""

    if _guard(st, result):
        return

    kpis = result.trading_kpis
    full = kpis.loc[kpis["strategy"].eq("full")].set_index("metric")
    evidence = f"Evidence: {_evidence_text(result.evidence_kind)}."
    # Intraday dispatch contract v1 §7: dispatch_world_slot is None exactly
    # when the switch was off or this is a no-action result (_guard above
    # has already ruled out no-action).
    dispatch_active = getattr(result, "dispatch_world_slot", None) is not None

    _kpi_rows(st, full, dispatch_active=dispatch_active)
    if dispatch_active:
        # Intraday dispatch contract v1 §6.2 (B3): with the dispatch on,
        # full earns re-optimisation P&L that perfect_foresight (volume
        # only, never trades intraday) cannot, so the old zero-premium
        # "capture stays near 1" reading no longer holds on its own.
        st.caption(
            "Capture can pass 100% and cost of uncertainty turn negative: the full strategy "
            "earns re-optimisation P&L that perfect foresight cannot.",
            help=f"Measured {PERFECT_FORESIGHT_CAPTION}: perfect foresight settles its volume "
            "at the day-ahead price alone and never trades intraday. The full strategy also "
            "earns the dispatch's own re-optimisation P&L, on top of the rebalancing trades "
            "that a zero premium makes roughly break-even.",
        )
    else:
        st.caption(
            "Capture near 100% and a negative value of intraday are expected: the premium is "
            "0, so intraday mostly rebalances rather than bets on price.",
            help="By default the day-ahead price is the expected price given what is known the "
            "day before, with a premium of 0. So on average the intraday price minus the "
            "day-ahead price is about 0 by construction, and intraday trading has no "
            "built-in edge.",
        )

    chart_block(
        st,
        _bucket_figure(result.trading_ledger_summary),
        title="Illustrative ledger buckets, mean £ per week",
        caption=f"Mean across simulated weeks; the nine buckets sum exactly to net. {evidence}",
        frame=_bucket_table(result.trading_ledger_summary),
        definition=(
            "The nine ledger buckets and their net, from the trading desk's point of view "
            "(positive is income). The chart shows means because only the mean adds up across "
            "buckets and across simulated weeks; medians and other percentiles are not summed. "
            "This "
            "net is never added to the drivers' saving on Smart charging: the saving and this "
            "revenue come from the same shifted energy, so adding them would count it twice. "
            "Under P415, the market rule that lets an independent flexibility provider sell a "
            "supplier's customers' turn-down, the supplier's lost energy is paid for by "
            "compensation shared across all suppliers, not by new value."
        ),
        height=_LEDGER_HEIGHT,
        key="trading-pnl-buckets",
    )
    st.caption(
        "Baseline effect: settled value from baseline error, not flexibility. Customer share: "
        "paid to drivers. Imbalance: delivered minus sold.",
    )
    full_week = _week_rows(result.trading_ledger_summary)
    full_week = full_week.loc[full_week["strategy"].eq("full")].set_index("metric")
    grid_and_compensation_zero = all(
        bucket in full_week.index and float(full_week.loc[bucket, "mean"]) == 0.0
        for bucket in ("grid_event_payment_gbp", "supplier_compensation_gbp")
    )
    if grid_and_compensation_zero:
        st.caption(
            "Grid events and supplier compensation are £0 here: no grid event paid out, "
            "and suppliers are paid from a shared fund instead."
        )
    chart_block(
        st,
        _dispatch_split_figure(result.trading_ledger_summary),
        title="Intraday P&L and trading cost, of which rebalancing vs re-optimisation",
        caption=(
            "Full strategy, mean across simulated weeks; means add up, quantiles do not. "
            f"{evidence}"
        ),
        frame=_dispatch_split_table(result.trading_ledger_summary),
        definition=(
            "Rebalancing is the trading the forecast updates alone would have caused, at the "
            "prices actually traded. Re-optimisation is the extra the dispatch's own response "
            "caused, at the same prices; it can be positive or negative for either bucket. "
            "The re-optimisation trading cost can be positive when the dispatch's trades net "
            "against the forecast's. Both are 0 on the other two strategies and with intraday "
            "dispatch off. The split stops before imbalance settlement: the dispatch also "
            "changes the settled volume and its imbalance price, which this split does not "
            "carry."
        ),
        height=_DISPATCH_SPLIT_HEIGHT,
        key="trading-pnl-dispatch-split",
    )

    metric_labels = dict(_METRIC_ROWS)
    perfect_foresight_rows = " and ".join(
        metric_labels[metric].lower() for metric in PERFECT_FORESIGHT_CAPTION_METRICS
    )
    st.markdown(
        "**Trader metrics, by strategy**",
        help=(
            "Median across simulated weeks. A strategy with no value for a metric shows "
            "Unavailable: day-ahead only and perfect foresight have no value of intraday. "
            "What each strategy means. "
            + " ".join(f"{STRATEGY_LABELS[s]}: {STRATEGY_CAPTIONS[s]}." for s in STRATEGY_ORDER)
            + (
                f" The {perfect_foresight_rows} rows are measured {PERFECT_FORESIGHT_CAPTION}."
                if dispatch_active
                else ""
            )
        ),
    )
    # height="content" shows every row (final critique O-9: the fixed height hid
    # the last three).
    st.dataframe(_strategy_table(kpis), width="stretch", hide_index=True, height="content")
    st.caption(f"Median across simulated weeks. {evidence}")

    st.markdown(
        "**Sanity checks**",
        help="Checks the model runs on its own ledger. Informative checks always show as "
        "passed; their value is there for context, not as a pass or fail bar.",
    )
    st.dataframe(_check_table(result.trading_checks), width="stretch", hide_index=True)
    st.caption("Computed on this run. Informative checks report a value for context only.")

    # Lead ruling Q5 (trading contract v1 section 9.5): shown only when the
    # hold-out control group is switched on; most runs carry no control
    # group, so this section is absent rather than an empty table.
    if result.control_group_summary is not None:
        st.markdown(
            "**Control group: estimated vs true baseline bias**",
            help="A control group is a share of EVs held out from every smart plan, charging as "
            "unmanaged. Because they never follow a plan, their own baseline error estimates "
            "the bias in the treated group's baseline. Comparing that estimate with the true "
            "bias, which is known only inside the simulation, checks how well a real supplier "
            "could do.",
        )
        st.dataframe(
            _control_group_table(result.control_group_summary), width="stretch", hide_index=True
        )
        st.caption(
            f"Median across simulated weeks. {_control_group_caption(result.control_group_summary)}"
        )

    # Trading contract v1 §10.5g, §10.10: physical firmness by charger maker
    # next to the ledger's own firmness metric above, and the commitment
    # rule the day-ahead position used (§10.4). Absent on a result built
    # before the availability lanes landed.
    firmness = getattr(result, "firmness_by_manufacturer", None)
    if firmness is not None:
        st.markdown(
            "**Firmness by manufacturer, 1 h turn-down**",
            help="Delivered turn-down divided by what would have been delivered had every "
            "session followed its plan, for a 1 hour turn-down. This is physical firmness by "
            "charger maker; the Firmness tile above is measured against the sold position "
            "instead.",
        )
        st.dataframe(_firmness_table(firmness), width="stretch", hide_index=True)
        st.caption(
            f"P10 and P50 across {result.world_count} simulated weeks. Firm is the P10: the "
            f"level met in 9 weeks out of 10. {evidence}"
        )

    rule = assumption_value(result, "trading.commitment_rule")
    if rule is not None:
        st.caption(_COMMITMENT_RULE_CAPTIONS.get(str(rule), f"Day-ahead position rule: {rule}."))
