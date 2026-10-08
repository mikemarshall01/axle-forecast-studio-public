"""Supplier ▸ 6 Customers: the supplier's marketing view of its own customers.

Household contract v1 §6.2. Partners (supplier contract v1 §11) speaks to
device makers; this lens speaks about the supplier's own customers: what can
be claimed about a typical customer, and which customers gain least. Every
figure is a read of ``household_outcomes_summary`` (§3.2) or
``household_value_summary`` (T§9.3c); this view computes no quantile, share,
ratio, physics or money, only exact display scaling (a fraction to a
percent, a GBP/kWh rate to pence).

Money here is the pass-through reading (saving plus the customer's share of
trading revenue, as if day-ahead prices reached the customer, T§9.3c): the
marketing claims block is headed "If customers paid day-ahead prices" (the
lead's ruling, household-v1 §6.2 Q1), because the supplier's own P&L (Supplier
▸ 3) keeps the flat-tariff procurement saving instead. The two readings are
never added or drawn in one bar. A yearly or monthly claim reads the set-B
statistics (each customer's own average week first, then across customers)
so the band is across customers' average years, never one week's spread
scaled by 52 (household-v1 §3.2 rule B1); a weekly claim reads the group's
session- or energy-weighted ratio, never the mean of per-customer ratios
(rule B2, O9). "Firm" is never used here for a P10 across customers; that
reading is worded "P10 customer" (rule B10).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from ..components import header
from ..components.chart_table import chart_block
from ..components.kpi import kpi, kpi_columns
from ..registry import FLEET_GROUP, GROUP_KEY
from ..style import (
    ARCHETYPE_COLOURS,
    CHART_HEIGHTS,
    MUTED_INK,
    SERIES_COLOURS,
    UNAVAILABLE,
    format_quantity,
    money,
    percent,
)
from .sessions import ARCHETYPE_ORDER
from .supplier_common import cohort_labels, evidence_caption, group_label, missing, stat_row

NO_FRAMES_MESSAGE = "This run has no customer outcomes. Run simulation again to add them."
FIRM_MW_PENDING = "Not in this run. Run simulation again to see it."

_SHARE_TO_PERCENT = 100.0
_PENCE_PER_POUND = 100.0
_RANGE_WIDTH = 8
_MARKER_SIZE = 12


def _money2(value: float) -> str:
    return money(value, decimals=2)


def _share(value: float) -> str:
    return percent(value * _SHARE_TO_PERCENT)


def _pence_per_kwh(value: float) -> str:
    return format_quantity(value * _PENCE_PER_POUND, "p/kWh", decimals=1)


def _kw(value: float) -> str:
    return format_quantity(value, "kW", decimals=1)


# --- Reading the summary frame -----------------------------------------------


def _row(summary: Any, group: str, metric: str, statistic: str) -> Any:
    return stat_row(summary, group_id=group, metric=metric, statistic=statistic)


def _p50(row: Any) -> float:
    """This row's own ``p50`` column, or NaN.

    Set A and set R rows carry a real across-weeks P50; a set B row stores
    the same number in ``mean`` and ``p50`` (household-v1 §3.2), so one
    accessor reads every statistic's value.
    """

    if row is None or missing(row["p50"]):
        return float("nan")
    return float(row["p50"])


def _band_range(row: Any, fmt: Any) -> str:
    """ "{low}-{high}" from one row's own ``p10``/``p90`` columns (set A or R), or ""."""

    if row is None or missing(row["p10"]) or missing(row["p90"]):
        return ""
    return f"{fmt(float(row['p10']))}–{fmt(float(row['p90']))}"


def _ev_means_range(low_row: Any, high_row: Any, fmt: Any) -> str:
    """ "{low}-{high}" from two set-B rows' own ``p50`` (the EV-means P10 and P90), or ""."""

    low, high = _p50(low_row), _p50(high_row)
    if np.isnan(low) or np.isnan(high):
        return ""
    return f"{fmt(low)}–{fmt(high)}"


def _archetype_colour(cohort_id: str) -> str:
    if cohort_id in ARCHETYPE_ORDER:
        return ARCHETYPE_COLOURS[ARCHETYPE_ORDER.index(cohort_id)]
    return MUTED_INK


def _cohort_groups(summary: Any) -> list[str]:
    """Cohorts with treated EVs, in source order (the summary's own group ids, minus fleet)."""

    present = set(summary["group_id"]) - {FLEET_GROUP}
    return [cohort_id for cohort_id in ARCHETYPE_ORDER if cohort_id in present]


# --- Claims block (item 1) ---------------------------------------------------


def _claims_row(st: Any, result: Any, group: str) -> None:
    """The four marketing-claim tiles (item 1): smart headline, unmanaged beside it in
    context, plus the optional timed-tariff reading (decision 0007, household-v1 model
    step 2) in the same context line, for the three tiles that already compare policies.
    "Typical value per year" has no unmanaged or timed counterpart (value is the smart
    saving itself), so it is left exactly as before.
    """

    summary = result.household_outcomes_summary
    tiles = kpi_columns(st, 4)

    year_p10 = _row(summary, group, "value_gbp_per_year", "p10_of_ev_means")
    year_p50 = _row(summary, group, "value_gbp_per_year", "p50_of_ev_means")
    year_p90 = _row(summary, group, "value_gbp_per_year", "p90_of_ev_means")
    weeks = int(year_p50["world_count"]) if year_p50 is not None else 0
    kpi(
        tiles[0],
        "Typical value per year",
        money(_p50(year_p50)),
        # Names its statistic (final critique Q-10): the median customer's
        # own average year, not a mean across customers or one EV.
        context=(
            f"median customer · {_ev_means_range(year_p10, year_p90, money)} for 8 in 10 "
            f"customers, each averaged over {weeks} weeks · illustrative"
        ),
        help="The median customer's average simulated week × 52: the saving plus its share "
        "of trading revenue, as if day-ahead prices reached the customer (the pass-through "
        "reading).",
    )

    ready = _row(summary, group, "completed_share_selected", "group_ratio")
    ready_normal = _row(summary, group, "completed_share_normal", "group_ratio")
    ready_timed = _row(summary, group, "completed_share_timed", "group_ratio")
    ready_context = (
        f"in a typical week P10–P90 {_band_range(ready, _share)} · unmanaged "
        f"{_share(_p50(ready_normal))}"
    )
    if ready_timed is not None:
        ready_context += f" · Timed tariff {_share(_p50(ready_timed))}"
    kpi(
        tiles[1],
        "Sessions ready at departure",
        _share(_p50(ready)),
        context=ready_context,
        help="Session-weighted share of the group's closed sessions that reached the "
        "preferred target by departure on the smart path; P50 across simulated weeks.",
    )

    cost = _row(summary, group, "home_cost_gbp_per_kwh_selected", "group_ratio")
    cost_normal = _row(summary, group, "home_cost_gbp_per_kwh_normal", "group_ratio")
    cost_timed = _row(summary, group, "home_cost_gbp_per_kwh_timed", "group_ratio")
    cost_context = (
        f"in a typical week · unmanaged {_pence_per_kwh(_p50(cost_normal))} · "
        "day-ahead wholesale, synthetic"
    )
    cost_help = (
        "Energy-weighted: the group's overall cost per kWh, not the mean of each "
        "customer's own; P50 across simulated weeks. 1 p/kWh = £10/MWh, the unit of the "
        "fleet and trading screens."
    )
    if cost_timed is not None:
        cost_context = (
            f"in a typical week · unmanaged {_pence_per_kwh(_p50(cost_normal))} · "
            f"Timed tariff {_pence_per_kwh(_p50(cost_timed))} · day-ahead wholesale, synthetic"
        )
        # Money is shown for the timed path here, so the costing rule it
        # shares with unmanaged and smart is said once, in words.
        cost_help += (
            " All three policies are costed the same way at the day-ahead price; no tariff "
            "rate is applied to the timed path."
        )
    kpi(
        tiles[2],
        "Smart charging cost",
        _pence_per_kwh(_p50(cost)),
        context=cost_context,
        # p/kWh is the customer unit; fleet and wholesale screens use £/MWh
        # (final critique B-10, axle-conventions units rule).
        help=cost_help,
    )

    cheap = _row(summary, group, "cheap_share_selected", "group_ratio")
    cheap_normal = _row(summary, group, "cheap_share_normal", "group_ratio")
    cheap_timed = _row(summary, group, "cheap_share_timed", "group_ratio")
    cheap_context = (
        f"in a typical week · unmanaged {_share(_p50(cheap_normal))} · each "
        "week's own cheapest third of half-hours"
    )
    if cheap_timed is not None:
        cheap_context = (
            f"in a typical week · unmanaged {_share(_p50(cheap_normal))} · Timed tariff "
            f"{_share(_p50(cheap_timed))} · each week's own cheapest third of half-hours"
        )
    kpi(
        tiles[3],
        "Charging in cheapest third",
        _share(_p50(cheap)),
        context=cheap_context,
        help="Energy-weighted share of home import that fell in each simulated week's own "
        "cheapest third of day-ahead half-hours; P50 across simulated weeks.",
    )


# --- Chart: value per year by archetype (item 2) -----------------------------


def _year_range(summary: Any, group: str) -> tuple[float, float, float]:
    low = _p50(_row(summary, group, "value_gbp_per_year", "p10_of_ev_means"))
    mid = _p50(_row(summary, group, "value_gbp_per_year", "p50_of_ev_means"))
    high = _p50(_row(summary, group, "value_gbp_per_year", "p90_of_ev_means"))
    return low, mid, high


def _range_traces(label: str, low: float, mid: float, high: float, colour: str) -> list[go.Scatter]:
    hover = f"{label}: P50 {money(mid)} · P10–P90 {money(low)}–{money(high)}"
    marker_line = {"color": MUTED_INK, "width": 1}
    return [
        go.Scatter(
            x=[low, high],
            y=[label, label],
            mode="lines",
            line={"color": colour, "width": _RANGE_WIDTH},
            name=label,
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[mid],
            y=[label],
            mode="markers",
            marker={"color": colour, "size": _MARKER_SIZE, "line": marker_line},
            name=label,
            showlegend=False,
            customdata=[hover],
            hovertemplate="%{customdata}<extra></extra>",
        ),
    ]


def _value_year_figure(summary: Any, groups: list[str], labels: dict[str, str]) -> go.Figure:
    figure = go.Figure()
    categories = []
    for group in groups:
        label = group_label(group, labels)
        categories.append(label)
        colour = MUTED_INK if group == FLEET_GROUP else _archetype_colour(group)
        low, mid, high = _year_range(summary, group)
        for trace in _range_traces(label, low, mid, high, colour):
            figure.add_trace(trace)
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(categories)),
        title={"text": "archetype, £ per year", "standoff": 12},
    )
    figure.update_xaxes(title="£ per year (illustrative)")
    figure.update_layout(hovermode="closest")
    return figure


def _least_gaining(summary: Any, cohorts: list[str], labels: dict[str, str]) -> str:
    values = {group: _year_range(summary, group)[0] for group in cohorts}
    valid = {group: value for group, value in values.items() if not np.isnan(value)}
    if not valid:
        return ""
    worst = min(valid, key=valid.get)
    return (
        f"Least: {group_label(worst, labels)}, P10 customer {money(valid[worst])} a year "
        "(average over weeks)."
    )


def _value_year_table(summary: Any, groups: list[str]) -> Any:
    rows = summary.loc[
        summary["group_id"].isin(groups)
        & summary["metric"].eq("value_gbp_per_year")
        & summary["statistic"].isin(("p10_of_ev_means", "p50_of_ev_means", "p90_of_ev_means"))
    ]
    return rows.rename(
        columns={
            "group_id": "Group",
            "statistic": "Statistic",
            "ev_count": "Customers",
            "world_count": "Weeks (min)",
            "mean": "Mean",
            "p10": "P10",
            "p50": "P50",
            "p90": "P90",
        }
    )[["Group", "Statistic", "Customers", "Weeks (min)", "Mean", "P10", "P50", "P90"]]


def _value_year_chart(st: Any, result: Any, labels: dict[str, str]) -> None:
    summary = result.household_outcomes_summary
    cohorts = _cohort_groups(summary)
    groups = [FLEET_GROUP, *cohorts]
    weeks = int(_row(summary, FLEET_GROUP, "value_gbp_per_year", "p50_of_ev_means")["world_count"])
    chart_block(
        st,
        _value_year_figure(summary, groups, labels),
        title="Value per year across customers, by archetype",
        caption=(
            f"P10–P90 of customers' average years ({weeks} weeks each) per archetype; "
            f"marker: P50 customer. {evidence_caption(result)}"
        ),
        frame=_value_year_table(summary, groups),
        definition="Pass-through reading: the saving plus the trading product's customer "
        "revenue share, as if day-ahead prices reached the customer. Each customer's own "
        "mean over its simulated weeks first, then P10/P50/P90 across the archetype's "
        "customers: an average year's spread, not one week's spread scaled by 52.",
        height=CHART_HEIGHTS["histogram"],
        key="customers-value-year",
    )
    verdict = _least_gaining(summary, cohorts, labels)
    if verdict:
        st.caption(verdict)


# --- Chart: flexible kW by archetype (item 3) --------------------------------


def _kw_range(summary: Any, group: str, metric: str) -> tuple[float, float, float]:
    low = _p50(_row(summary, group, metric, "p10_across_evs"))
    mid = _p50(_row(summary, group, metric, "p50_across_evs"))
    high = _p50(_row(summary, group, metric, "p90_across_evs"))
    return low, mid, high


def _kw_figure(summary: Any, cohorts: list[str], labels: dict[str, str]) -> go.Figure:
    category_labels = [group_label(group, labels) for group in cohorts]
    figure = go.Figure()
    for metric, name, colour in (
        ("evening_turn_down_kw_1h", "Turn-down (smart)", SERIES_COLOURS["selected"]),
        ("evening_turn_up_kw_1h", "Turn-up (smart)", SERIES_COLOURS["difference"]),
    ):
        lows, mids, highs = [], [], []
        for group in cohorts:
            low, mid, high = _kw_range(summary, group, metric)
            lows.append(low)
            mids.append(mid)
            highs.append(high)
        mid_arr, low_arr, high_arr = (np.array(a, dtype=float) for a in (mids, lows, highs))
        hovers = [
            f"{label}: P50 customer {_kw(mid)} · P10 customer {_kw(low)} · P90 customer {_kw(high)}"
            for label, low, mid, high in zip(category_labels, lows, mids, highs, strict=True)
        ]
        figure.add_trace(
            go.Bar(
                x=category_labels,
                y=mid_arr,
                name=name,
                marker_color=colour,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": np.maximum(high_arr - mid_arr, 0.0),
                    "arrayminus": np.maximum(mid_arr - low_arr, 0.0),
                    "color": MUTED_INK,
                    "thickness": 1,
                    "width": 4,
                },
                customdata=hovers,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )
    figure.update_yaxes(title="kW, smart path", rangemode="tozero")
    figure.update_xaxes(title="archetype")
    figure.update_layout(barmode="group", hovermode="closest")
    return figure


def _kw_table(summary: Any, cohorts: list[str]) -> Any:
    rows = summary.loc[
        summary["group_id"].isin(cohorts)
        & summary["metric"].isin(("evening_turn_down_kw_1h", "evening_turn_up_kw_1h"))
        & summary["statistic"].isin(("p10_across_evs", "p50_across_evs", "p90_across_evs"))
    ]
    return rows.rename(
        columns={
            "group_id": "Group",
            "metric": "Direction",
            "statistic": "Statistic",
            "ev_count": "Customers",
            "world_count": "Weeks",
            "mean": "Mean",
            "p10": "P10",
            "p50": "P50",
            "p90": "P90",
        }
    )[["Group", "Direction", "Statistic", "Customers", "Weeks", "Mean", "P10", "P50", "P90"]]


def _flexible_kw_chart(st: Any, result: Any, labels: dict[str, str]) -> None:
    summary = result.household_outcomes_summary
    cohorts = _cohort_groups(summary)
    title = "Flexible kW per customer, by archetype"
    fleet_mid = _p50(_row(summary, FLEET_GROUP, "evening_turn_down_kw_1h", "p50_across_evs"))
    if np.isnan(fleet_mid):
        st.markdown(f"**{title}**")
        st.info(FIRM_MW_PENDING)
        return
    chart_block(
        st,
        _kw_figure(summary, cohorts, labels),
        title=title,
        caption=(
            f"Realised 1 h evening flexibility per customer, smart path; P10–P90 across "
            f"customers, medians across weeks. {evidence_caption(result)}"
        ),
        frame=_kw_table(summary, cohorts),
        definition="Each customer's kW this hour if called, realised on the smart path; 0 on "
        "a night its maker is out or when it ignores the plan. Whiskers: the P10 and P90 "
        "customer, each at its own median across simulated weeks (never 'firm', which names "
        "only an across-weeks P10).",
        height=CHART_HEIGHTS["histogram"],
        key="customers-flexible-kw",
    )
    # Clarity critique: turn-down bars sit near 0 next to large turn-up bars,
    # which a first-time reader reads as broken. It is not: smart charging
    # has already moved the evening charge, so little extra turn-down is
    # left to offer (the same pattern Supplier ▸ 4 Firm MW explains).
    st.caption(
        "Turn-down reads near 0 here: smart charging already moved this customer's evening "
        "charge to cheaper half-hours."
    )


# --- Value row (item 4) ------------------------------------------------------


def _value_row(st: Any, result: Any, group: str) -> None:
    summary = result.household_outcomes_summary
    tiles = kpi_columns(st, 4)

    energy = _row(summary, group, "supplier_energy_saving_gbp_per_month", "mean_of_ev_means")
    kpi(
        tiles[0],
        "Supplier energy saving",
        _money2(_p50(energy)),
        unit="/month",
        context="energy component only, flat-tariff reading; hedge and events are fleet cash",
        help="The energy part of the supplier's P&L under the flat-tariff reading (the "
        "supplier keeps the saving on the energy it buys), per customer, averaged over each "
        "customer's own simulated weeks. The hedge-error and grid-event parts are fleet-level "
        "cash and are not split per customer.",
    )

    saving = _row(summary, group, "saving_gbp_per_week", "mean_across_evs")
    kpi(
        tiles[1],
        "Weekly saving per customer",
        _money2(_p50(saving)),
        context="typical week, before the trading share; value = saving + share",
        help="The average customer's median week, pass-through reading, before its share of "
        "trading revenue is added.",
    )

    worse_year = _row(summary, group, "value_gbp_per_year", "share_of_ev_means_below_zero")
    worse_week = stat_row(
        result.household_value_summary, group_id=group, statistic="share_worse_off"
    )
    worse_week_text = (
        "" if worse_week is None or missing(worse_week["p50"]) else _share(float(worse_week["p50"]))
    )
    kpi(
        tiles[2],
        "Customers worse off",
        _share(_p50(worse_year)),
        context=(f"average year below £0 · in a typical week {worse_week_text or UNAVAILABLE}"),
        help="Share of the group's customers whose average simulated year (pass-through "
        "reading) is below zero, and the same for one typical week.",
    )

    co2 = _row(summary, group, "co2_shifted_kg_per_month", "mean_of_ev_means")
    co2_value = _p50(co2)
    kpi(
        tiles[3],
        "CO₂ shifted per customer",
        format_quantity(co2_value, "kg", decimals=1),
        unit="/month",
        # Names its statistic (final critique Q-11): the mean customer's average
        # month, which is why it differs slightly from Supplier P&L's P50 week.
        context=(
            "average customer, average month · illustrative synthetic intensity"
            if not np.isnan(co2_value)
            else "Not in this run. Run simulation again to see it"
        ),
        help="Unmanaged minus smart home import times an illustrative synthetic carbon "
        "intensity tied to the model's own net demand; not observed grid data. Mean across "
        "customers of each customer's average week × 52 ÷ 12, so it differs slightly from "
        "Supplier P&L's figure, which is the P50 week.",
    )
    st.caption(f"{evidence_caption(result)}")


# --- Data and definition (item 5) --------------------------------------------


def _summary_table(summary: Any, group: str) -> Any:
    rows = summary.loc[summary["group_id"].eq(group)]
    return rows.rename(
        columns={
            "metric": "Metric",
            "statistic": "Statistic",
            "unit": "Unit",
            "horizon": "Horizon",
            "better_is": "Better is",
            "ev_count": "Customers",
            "ev_value_count": "Customers with a value",
            "world_count": "Weeks",
            "mean": "Mean",
            "p10": "P10",
            "p50": "P50",
            "p90": "P90",
            "evidence_kind": "Evidence",
        }
    )


# --- The lens -----------------------------------------------------------------


def render_customers(st: Any, result: Any) -> None:
    """Render Supplier ▸ 6 Customers: what can be told to customers, and who gains least.

    Reads ``result.household_outcomes_summary`` (household contract v1 §3.2)
    and, for one tile, ``result.household_value_summary`` (T§9.3c). No
    kernel run, no quantile, no ratio: every number is a stored column.
    """

    summary = getattr(result, "household_outcomes_summary", None)
    if summary is None:
        st.info(NO_FRAMES_MESSAGE)
        return

    labels = cohort_labels(result)
    groups = list(dict.fromkeys(summary["group_id"]))
    group = header.controls(st).selectbox(
        "Group",
        options=groups,
        format_func=lambda g: group_label(g, labels),
        key=GROUP_KEY,
        label_visibility="collapsed",
        persist_state="session",
    )
    if group not in groups:
        group = FLEET_GROUP

    st.subheader("If customers paid day-ahead prices")
    _claims_row(st, result, group)
    # Three of the four claim tiles gain an optional Timed tariff reading
    # beside unmanaged (decision 0007, household-v1 model step 2); one of
    # those is money (the cost tile), so the caption says every policy is
    # costed the same way whenever that reading is on screen.
    if "completed_share_timed" in set(summary["metric"]):
        st.caption(
            "Simulated fleet, synthetic prices, pass-through: what customers would pay at "
            "day-ahead prices. All three costed alike; not a tariff."
        )
    else:
        st.caption(
            "Simulated fleet, synthetic prices, pass-through reading: what customers would pay "
            "if day-ahead prices reached them. Not a tariff."
        )

    _value_year_chart(st, result, labels)
    _flexible_kw_chart(st, result, labels)
    _value_row(st, result, group)

    with st.expander("Data and definition"):
        st.dataframe(_summary_table(summary, group), width="stretch", hide_index=True)
        st.caption(
            "Two readings, never added. The pass-through value above is the saving plus the "
            "trading desk's customer revenue share, as if day-ahead prices reached the "
            "customer. The supplier's own flat-tariff reading is on Supplier P&L. Three "
            "spreads: in a typical week (across weeks); across customers in a typical week "
            "(medians across weeks); and across customers' average years (each customer's own "
            "mean over weeks first). A customer is a treated EV of the simulated fleet, one "
            "enrolled in smart charging; a hold-out control EV is left out of every group "
            "figure. Illustrative, synthetic prices; not a bill, a tariff or a guarantee."
        )
