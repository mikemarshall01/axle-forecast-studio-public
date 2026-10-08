"""Partners: a pitch card per outside audience, linking the model to "how we get partners".

What this owns: five persona lenses (Driver, Energy supplier, Charger maker,
Carmaker, Fleet and leasing), each one line naming what that partner buys and
its route to market, four KPI tiles, one hero chart and up to four supporting
charts, a "What they would ask" table and a row of drill-through buttons onto
the deep lens that already answers each chart's question. It draws nothing
new: every chart is an existing builder from ``overview``, ``plug_ins``,
``sessions``, ``customers``, ``action_cost``, ``action_response``,
``household``, ``supplier_availability``, ``supplier_firm_mw``,
``supplier_pnl`` and ``partners`` (Supplier ▸ 5 Charger makers), called with
the same row selection those pages use by default (fleet group, the page's
own default day type, the profiled hedge, a 1 h turn-down), so the numbers on
this card always match the deep page a reader is sent to. No quantile, share,
ratio or money is computed here beyond exact display scaling, the same rule
every other view in this package follows.

The "What they would ask" table is the honest core (design brainstorm
section 6): each row's reading is pulled from ``key_stats.key_stats_table``
by its exact ``Metric`` text, or the table says "Not modelled", never a
number invented for the page. "Where" always names a screen (page ▸ lens),
never a frame or column name.

Every persona reads action-only frames (partner cash, supplier P&L, firm MW,
smart-charging outcomes do not exist on a no-action result), the same guard
``pages._action_lens`` gives every Supplier lens.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from axle_studio.model import growth

from ..components import header
from ..components.chart_table import chart_block
from ..components.kpi import kpi, kpi_columns
from ..registry import FLEET_GROUP
from ..style import (
    CHART_HEIGHTS,
    UNAVAILABLE,
    assumption_value,
    format_quantity,
    money,
    percent,
)
from .action_cost import _cost_points_table, _strip_figure
from .action_response import _paths_figure
from .customers import _value_year_figure
from .household import HouseholdCardFn, _outcomes_figure
from .key_stats import key_stats_table
from .one_ev import ev_picker
from .overview import _average_day_figure, _average_day_row, _switch_to
from .partners import (
    _MARKET_LABELS,
    FUNNEL_DEFAULTS,
    PAYOUT_MONTHS,
    THRESHOLD_DEFAULT,
    _departure_figure,
    _exceedance_figure,
    _funnel_figure,
    _market_figure,
    _payout_figure,
)
from .plug_ins import _heatmap_figure, _hour_figure, _shared_colour_top, _soc_figure
from .sessions import _figure as _sessions_figure
from .supplier_availability import _curve_figure, _curve_rows
from .supplier_common import band_text, cohort_labels, evidence_caption, missing, p50_text, stat_row
from .supplier_firm_mw import (
    _day_ahead_figure,
    _day_ahead_rows,
    _fleet_size_figure,
    _product_sheet_table,
    _reliability_figure,
    _this_week_rows,
)
from .supplier_pnl import _block_figure, _block_rows, _waterfall_figure, waterfall_rows

NOT_MODELLED = "Not modelled"

EV_ONLY_CAPTION = "EV-only model: home batteries, heat pumps and vehicle-to-grid are not modelled."
"""Design brainstorm section 5: one line under the persona selector, every lens."""

LIMITS_SCREEN = "How it works ▸ Limits"
KEY_STATS_SCREEN = "Overview ▸ Key stats"

FIRM_MW_PENDING = "Not in this run. Run simulation again to see it."

# The firm-MW frames this card reads directly (not through key_stats_table,
# which already guards its own reads); a result built before that lane
# landed has none of them (supplier_firm_mw.py's own ``_REQUIRED_FRAMES``
# gates the whole Firm MW page the same way).
_FIRM_MW_FRAMES = (
    "availability_bands",
    "availability_world_slot",
    "firm_share_by_fleet_size",
    "product_sheet",
    "availability_reliability",
)


def _firm_mw_available(result: Any) -> bool:
    return all(getattr(result, name, None) is not None for name in _FIRM_MW_FRAMES)


# Row-selection defaults this page replicates from the deep lenses it reuses
# (design brainstorm section 6): fleet group, each page's own default day
# type, the profiled hedge, a 1 h turn-down.
_DAY_TYPE = "weekday"
_HEDGE_VARIANT = "profiled"
_DIRECTION = "turn_down"
_DURATION_HOURS = 1.0


def _reading(table: pd.DataFrame, metric: str) -> str:
    """The P50 with its unit from ``key_stats_table``'s row named ``metric``.

    "Not modelled" when no such row exists (the digest's NO items: plug-in
    uplift, override rate, home/public share, telematics load and the
    rest); "Unavailable" when the row exists but this run's P50 is missing
    (an unset commercial term, decision 0003). The two must never be
    confused, so this table never invents a number to fill either gap.
    """

    rows = table.loc[table["Metric"].eq(metric)]
    if rows.empty:
        return NOT_MODELLED
    row = rows.iloc[0]
    value, unit = str(row["P50"]), str(row["Unit"])
    if value in ("", UNAVAILABLE):
        return UNAVAILABLE
    if "£" in value:
        # A money row already carries its own £ sign (key_stats.py's
        # is_money path); the unit text repeats "£", so only its "per ..."
        # remainder is worth adding.
        remainder = unit.replace("£", "").strip()
        return f"{value} {remainder}".strip()
    if unit.startswith("%"):
        return f"{value}%{unit[1:]}"
    if unit in ("", "ratio"):
        return value
    return f"{value} {unit}"


def _reference(table: pd.DataFrame, metric: str) -> str:
    """The Reference cell of ``key_stats_table``'s row named ``metric``, or "Not modelled"."""

    rows = table.loc[table["Metric"].eq(metric)]
    if rows.empty:
        return NOT_MODELLED
    reference = str(rows.iloc[0]["Reference"])
    return reference if reference else UNAVAILABLE


def _firm_reading(table: pd.DataFrame, metric: str) -> str:
    """The P10 ("firm", 90% exceedance) cell of a Firm MW row, with its P50 beside it.

    A plain ``_reading`` would show the P50, which the digest's own weak-numbers
    guidance flags as misleading here: "firm" names only the across-weeks P10
    (trading contract v1 §10's two-convention note; key_stats.py's own
    Reference cell for this row says so). Both figures come straight from the
    stored table, never recomputed.
    """

    rows = table.loc[table["Metric"].eq(metric)]
    if rows.empty:
        return NOT_MODELLED
    row = rows.iloc[0]
    firm, typical, unit = str(row["P10"]), str(row["P50"]), str(row["Unit"])
    if firm in ("", UNAVAILABLE):
        return UNAVAILABLE
    return f"{firm} {unit} firm (P10); typical (P50) {typical} {unit}"


def _wants_table(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """(Want, Reading, Where) rows as a plain table for ``st.dataframe``."""

    return pd.DataFrame(rows, columns=["Want", "Reading", "Where"])


def _wants_heading(st: Any, audience: str) -> None:
    """The unlabelled table's missing lead-in (clarity critique): who is asking, and how to read it.

    Every persona's table otherwise has no heading at all, so a first-time
    reader had no way to know what it was for.
    """

    st.markdown(
        f"**What {audience} would ask, and where the app answers it**",
        help=(
            "Reading: the run's own P50, read the same way a tile above already reads it. "
            "Where: the page and lens with the full chart and definition. Not modelled: the "
            "question is outside this EV-only model; How it works ▸ Limits says why."
        ),
    )


def _drill_through(st: Any, targets: tuple[tuple[str, str], ...]) -> None:
    """A row of buttons through ``overview._switch_to`` to each named deep lens."""

    columns = st.columns(len(targets))
    for column, (page_name, lens_name) in zip(columns, targets, strict=True):
        column.button(
            f"{page_name} ▸ {lens_name} →",
            key=f"partners-persona-drill::{page_name}::{lens_name}",
            on_click=_switch_to,
            args=(page_name, lens_name),
            width="stretch",
        )


def _intro(st: Any, text: str) -> None:
    """The persona's one line: what they buy, and the route (design brainstorm section 5)."""

    st.caption(EV_ONLY_CAPTION)
    st.markdown(f"**{text}**")


# --- Driver -------------------------------------------------------------------


def render_driver(st: Any, result: Any) -> None:
    """Driver: is my car ready, what do I save, does my battery get abused."""

    _intro(st, "Buys nothing directly: the software runs on their own home charger and tariff.")
    table = key_stats_table(result)

    year_row = stat_row(
        result.household_outcomes_summary,
        group_id="fleet",
        metric="value_gbp_per_year",
        statistic="p50_of_ev_means",
    )
    ready_row = stat_row(
        getattr(result, "charge_completion_summary", None),
        group_id="fleet",
        path_id="selected",
        departure="all",
    )
    smart_price = stat_row(
        result.smart_charging_summary, metric="selected_average_price_gbp_per_mwh"
    )
    unmanaged_price = stat_row(
        result.smart_charging_summary, metric="normal_average_price_gbp_per_mwh"
    )
    early = stat_row(result.smart_charging_summary, metric="early_departure_count")

    columns = kpi_columns(st, 4)
    kpi(
        columns[0],
        "Value per year, median",
        p50_text(year_row, lambda v: money(v)),
        context="Median customer's own average year, illustrative",
        help="Each customer's own average over its simulated weeks first, then the median "
        "across customers. The pass-through reading: the saving plus the customer's share of "
        "trading revenue, as if day-ahead prices reached the customer.",
    )
    kpi(
        columns[1],
        # Fable review O5: household.py's own tile for the same statistic
        # says "Ready at departure"; Driver said "Charged by departure":
        # one name for one thing.
        "Ready at departure",
        UNAVAILABLE
        if ready_row is None or missing(ready_row["completed_share_p50"])
        else percent(100 * float(ready_row["completed_share_p50"])),
        context=None
        if ready_row is None
        else f"P10 {percent(100 * float(ready_row['completed_share_p10']))} · "
        f"P90 {percent(100 * float(ready_row['completed_share_p90']))}",
        help="Smart path: share of sessions that reached their target by unplug, P50 across "
        "simulated weeks.",
    )
    # Fable review Q1: this persona is driver-facing, so the price reads in
    # p/kWh (the customer unit, 1 dp; household.py's own "Smart cost"
    # panel), not the wholesale £/MWh a supplier or trader reads elsewhere
    # on this page. £/MWh to p/kWh is ÷10 (£1/MWh = 0.1p/kWh).
    kpi(
        columns[2],
        "Price paid, smart",
        p50_text(smart_price, lambda v: format_quantity(v / 10, "p/kWh", decimals=1)),
        context="Fleet energy-weighted average"
        + (
            ""
            if unmanaged_price is None
            else f"; unmanaged {format_quantity(unmanaged_price['p50'] / 10, 'p/kWh', decimals=1)}"
        ),
        help="Day-ahead synthetic price paid, energy-weighted, P50 across simulated weeks.",
    )
    kpi(
        columns[3],
        "Early departures",
        p50_text(early, lambda v: format_quantity(v, "", decimals=1)),
        unit="/week",
        context=band_text(early, lambda v: format_quantity(v, "", decimals=1)),
        help="Sessions per week that left before the smart plan finished.",
    )

    chart_block(
        st,
        # Reuses customers._value_year_figure with groups=["fleet"] only
        # (design brainstorm section 6: "group fleet"), a single-row range
        # chart rather than the six-archetype comparison Customers draws.
        _value_year_figure(result.household_outcomes_summary, [FLEET_GROUP], {}),
        title="Value per year, illustrative (fleet)",
        caption=f"P10–P90 of customers' average years. {evidence_caption(result)}",
        frame=_value_year_frame(result),
        definition="Pass-through reading: the saving plus the trading product's customer "
        "revenue share, as if day-ahead prices reached the customer.",
        height=CHART_HEIGHTS["histogram"],
        key="partners-persona-driver-value-year",
    )

    left, right = st.columns(2)
    with left:
        chart_block(
            st,
            _strip_figure(
                result.cost_effect,
                material_share_percent=assumption_value(
                    result, "not_recovered_material_share_percent"
                ),
            ),
            title="Illustrative weekly saving (£)",
            caption=f"One point per simulated week ({len(result.cost_effect)}). Illustrative.",
            frame=_cost_points_table(result.cost_effect),
            definition="Unmanaged minus smart total cost, one point per simulated week.",
            height=CHART_HEIGHTS["strip"],
            key="partners-persona-driver-cost-strip",
        )
    with right:
        bands = result.session_distribution_bands
        departure_rows = bands.loc[
            bands["group_id"].eq("fleet")
            & bands["day_type"].eq("all")
            & bands["metric"].isin(["departure_soc_percent", "departure_soc_percent_smart"])
        ]
        chart_block(
            st,
            _departure_figure(departure_rows),
            title="SoC at departure, both paths, fleet, % of sessions",
            caption="Bars P50, whiskers P10–P90 across weeks; sessions ending in the study.",
            frame=departure_rows[["metric", "bin_label", "share_p10", "share_p50", "share_p90"]],
            definition="State of charge (SoC): battery energy at each session's last connected "
            "half-hour ÷ capacity, unmanaged and smart paths of the same sessions.",
            height=CHART_HEIGHTS["small_multiple_panel"] + 60,
            key="partners-persona-driver-departure-soc",
        )

    left, right = st.columns(2)
    with left:
        soc_bands = result.plug_in_summary.soc_bands
        cnz_soc = assumption_value(result, "cnz_median_plug_in_soc_percent")
        chart_block(
            st,
            _soc_figure(soc_bands, day_type=_DAY_TYPE, cnz_percent=cnz_soc),
            title="SoC at plug-in, share per 5% bin",
            caption=(
                "Unmanaged path, weekday. Bars: mean share of plug-ins; whiskers: P10–P90 "
                f"across {int(soc_bands['world_count'].max())} weeks. The CNZ marker is "
                "observed context only."
            ),
            frame=soc_bands.loc[soc_bands["day_type"].eq(_DAY_TYPE)],
            definition="Share of weekday unmanaged-path plug-ins landing in each 5-point SoC "
            "bin, mean across simulated weeks.",
            height=CHART_HEIGHTS["histogram"],
            key="partners-persona-driver-soc",
        )
    with right:
        connected = _average_day_row(result, metric="connected_share", day_type=_DAY_TYPE)
        soc = _average_day_row(result, metric="battery_soc_percent", day_type=_DAY_TYPE)
        chart_block(
            st,
            _average_day_figure(connected, soc),
            title="Average day: % plugged in and battery SoC",
            caption="Weekday. Bars: P50 share plugged in. Line and band: mean SoC, P5–P95 "
            "across EVs.",
            frame=pd.concat([connected, soc], ignore_index=True),
            definition="One row per London half-hour: the bar is the fleet's share plugged in "
            "at home; the line and band are the mean battery SoC and its 5th-95th percentile "
            "spread across EVs.",
            height=CHART_HEIGHTS["time_series_dual_axis"],
            key="partners-persona-driver-average-day",
        )

    _wants_heading(st, "a driver")
    st.dataframe(
        _wants_table(
            [
                (
                    "What's a typical year worth to me?",
                    p50_text(year_row, lambda v: f"{money(v)} a year, median customer"),
                    "Supplier ▸ 6 Customers",
                ),
                (
                    "Is my car ready when I need it?",
                    _reading(table, "Charge completed by departure, smart"),
                    "Drivers ▸ Household",
                ),
                (
                    "Do I pay more for public charging?",
                    _reading(table, "Public top-up change"),
                    "Smart charging ▸ 2 Response",
                ),
                (
                    "Is my battery cycled harder?",
                    _reading(table, "Equivalent full cycles, smart"),
                    "Supplier ▸ 5 Charger makers",
                ),
                (
                    "Will plugging in more often earn me more?",
                    NOT_MODELLED,
                    LIMITS_SCREEN,
                ),
                (
                    "Can I get an override or boost button?",
                    NOT_MODELLED,
                    LIMITS_SCREEN,
                ),
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    _drill_through(st, (("Drivers", "Household"), ("Supplier", "6 Customers")))


def _value_year_frame(result: Any) -> pd.DataFrame:
    summary = result.household_outcomes_summary
    return summary.loc[
        summary["group_id"].eq("fleet")
        & summary["metric"].eq("value_gbp_per_year")
        & summary["statistic"].isin(("p10_of_ev_means", "p50_of_ev_means", "p90_of_ev_means"))
    ][["group_id", "statistic", "ev_count", "world_count", "mean", "p10", "p50", "p90"]]


# --- Energy supplier ------------------------------------------------------------


def render_energy_supplier(st: Any, result: Any) -> None:
    """Energy supplier: procurement cost shape, hedge error, peak block, margin, firm MW."""

    _intro(
        st,
        "Between routes: the supplier trades; Axle forecasts, positions and dispatches for it.",
    )
    table = key_stats_table(result)

    net = stat_row(
        result.supplier_pnl_summary,
        hedge_variant=_HEDGE_VARIANT,
        metric="net_gain_before_fee_per_customer_per_month_gbp",
    )
    moved = stat_row(result.smart_charging_summary, metric="moved_home_import_share")
    erosion = stat_row(
        result.trading_kpis, strategy="full", metric="evening_baseline_erosion_share"
    )
    shape = stat_row(
        result.shape_premium_summary, path_id="selected", metric="shape_premium_gbp_per_mwh"
    )

    columns = kpi_columns(st, 4)
    kpi(
        columns[0],
        "Supplier gain per customer",
        p50_text(net, lambda v: money(v, decimals=2, signed=True)),
        unit="/month",
        context=(band_text(net, lambda v: money(v, decimals=2, signed=True)) or "Before fee"),
        help="Illustrative simulated supplier gain before the platform fee, per treated "
        "customer (an EV enrolled in smart charging) per month, with a profiled hedge (the "
        "supplier buys the fleet's own forecast shape). P50 across simulated weeks.",
    )
    kpi(
        columns[1],
        "Energy moved",
        p50_text(moved, lambda v: percent(100 * v)),
        context="Share of home import shifted vs unmanaged",
        help="Energy-weighted share of unmanaged home import moved to a different half-hour.",
    )
    kpi(
        columns[2],
        "Evening value lost",
        UNAVAILABLE
        if erosion is None or missing(erosion["mean"])
        else percent(100 * float(erosion["mean"]), decimals=0),
        context="of the evening's settled value, if the baseline learnt from smart nights",
        help="One minus the ratio of mean evening settled values: a settlement baseline learnt "
        "from smart nights ÷ the default baseline, 16:00–20:00 London. A mean ratio, not a "
        "P50. Full detail on Trading ▸ 2 Position.",
    )
    kpi(
        columns[3],
        "Shape premium, smart",
        p50_text(shape, lambda v: format_quantity(v, "£/MWh", decimals=1)),
        context="below 0: load sits in cheaper half-hours than a flat block",
        help="The load-weighted day-ahead price minus the week's mean price: what the load's "
        "timing costs against a flat block. Smart path, P50 across simulated weeks.",
    )

    rows = waterfall_rows(result, _HEDGE_VARIANT)
    chart_block(
        st,
        _waterfall_figure(rows),
        title="Supplier P&L by component, mean £ per week",
        caption="Means add to the net; quantiles do not. Illustrative, profiled hedge.",
        frame=rows,
        definition="Each component's stored mean, then the net; the only statistic that adds.",
        height=CHART_HEIGHTS["time_series"],
        key="partners-persona-supplier-waterfall",
    )

    left, right = st.columns(2)
    with left:
        bands = result.fleet_interval_bands
        path_rows = bands.loc[bands["metric"].eq("home_import_kw")]
        chart_block(
            st,
            _paths_figure(path_rows, "home_import_kw", "kW"),
            title="Home import: unmanaged vs smart (kW)",
            caption="P50 and P10–P90 across weeks. Smart peaks after midnight as EVs pick the "
            "same cheap half-hours: the shape moves, it does not shrink.",
            frame=path_rows,
            definition="Average power drawn at the home charger, half-hour by half-hour, both "
            "paths.",
            height=CHART_HEIGHTS["time_series"],
            key="partners-persona-supplier-paths",
        )
    with right:
        block_rows = _block_rows(result)
        chart_block(
            st,
            _block_figure(block_rows),
            title="Hedge blocks: mean MW, unmanaged vs smart",
            caption="P10–P90 across simulated weeks. Baseload, peak and off-peak blocks.",
            frame=block_rows,
            definition="Mean power over each hedge block, both paths.",
            height=CHART_HEIGHTS["strip"],
            key="partners-persona-supplier-blocks",
        )

    left, right = st.columns(2)
    with left:
        curve_rows = _curve_rows(result, half_hour="18:00", day_type=_DAY_TYPE)
        chart_block(
            st,
            _curve_figure(curve_rows),
            title="Turn-down available by cost to move, 18:00, weekday",
            caption="P10–P90 across weeks. Day-ahead. Cost to move: the cheapest later "
            "half-hour minus this one; negative means moving saves money.",
            frame=curve_rows,
            definition="Turn-down in each simulated week's sessions, costed at the day-ahead "
            "prices visible at the day-ahead decision.",
            height=CHART_HEIGHTS["time_series"],
            key="partners-persona-supplier-cost-curve",
        )
    with right:
        if _firm_mw_available(result):
            day_ahead_rows = _day_ahead_rows(
                result.availability_bands, direction=_DIRECTION, duration=_DURATION_HOURS
            )
            this_week = _this_week_rows(
                result.availability_world_slot,
                world_id=result.representative_world_id,
                direction=_DIRECTION,
                duration=_DURATION_HOURS,
            )
            chart_block(
                st,
                _day_ahead_figure(day_ahead_rows, this_week),
                title="Turn-down deliverable kW, day-ahead, 1 h hold",
                caption="Band: P10–P90 across weeks, firm at the lower edge. Small in the "
                "evening: already emptied, so the firm product lives overnight.",
                frame=day_ahead_rows,
                definition="Deliverable kW is what the smart-charged fleet could hold for this "
                "long, starting this half-hour. Firm is the level met in 9 weeks out of 10: "
                "the P10 across weeks (90% exceedance).",
                height=CHART_HEIGHTS["time_series"],
                key="partners-persona-supplier-day-ahead",
            )
        else:
            st.markdown("**Turn-down deliverable kW, day-ahead, 1 h hold**")
            st.caption(FIRM_MW_PENDING)

    with st.expander("Grid route: firm MW"):
        st.caption("Illustrative; not a registered product.")
        if not _firm_mw_available(result):
            st.caption(FIRM_MW_PENDING)
        else:
            reliability = result.availability_reliability
            chart_block(
                st,
                _reliability_figure(reliability, direction=_DIRECTION, duration=_DURATION_HOURS),
                title="Calibration: nominal vs observed coverage",
                caption="A calibrated forecast sits on the dotted line. Illustrative, "
                "simulated weeks, not real telemetry.",
                frame=reliability.loc[
                    reliability["direction"].eq(_DIRECTION)
                    & reliability["duration_hours"].eq(_DURATION_HOURS)
                ],
                definition="Coverage at a level is the share of kept week-slots (one simulated "
                "week, one half-hour) where what happened was at or below the forecast at "
                "that level.",
                height=CHART_HEIGHTS["time_series"],
                key="partners-persona-supplier-reliability",
            )
            sizes = result.firm_share_by_fleet_size
            fleet_size_rows = sizes.loc[
                sizes["window"].eq("evening")
                & sizes["direction"].eq(_DIRECTION)
                & sizes["duration_hours"].eq(_DURATION_HOURS)
            ].sort_values("fleet_size")
            chart_block(
                st,
                _fleet_size_figure(fleet_size_rows),
                title="Firm share against fleet size, evening",
                caption="Flat beyond a few hundred EVs: shared factors, not sampling noise, "
                "set the firm share.",
                frame=fleet_size_rows,
                definition="The first n EVs in population order, a random sub-fleet under the "
                "same shared factors, prices and weather.",
                height=CHART_HEIGHTS["time_series"],
                key="partners-persona-supplier-fleet-size",
            )
            st.markdown("**Product sheet**")
            st.dataframe(
                _product_sheet_table(result.product_sheet), hide_index=True, width="stretch"
            )

    trading_net = stat_row(result.trading_kpis, strategy="full", metric="net_gbp_per_week")
    st.caption(
        "Trading desk (beside, not added): "
        + p50_text(trading_net, lambda v: money(v, signed=True))
        + "/week.",
        help="The trading desk's own illustrative net for the week (a desk that reads what "
        "the fleet did without changing it). Never added to the supplier's gain above. Full "
        "detail on Trading ▸ 3 P&L and risk.",
    )

    _wants_heading(st, "an energy supplier")
    st.dataframe(
        _wants_table(
            [
                (
                    "What's the shape-cost saving?",
                    _reading(table, "Shape premium, smart"),
                    "Supplier ▸ 3 Supplier P&L",
                ),
                (
                    "How much does hedge error cost fall?",
                    _reading(table, "Hedge-error saving, profiled hedge"),
                    "Supplier ▸ 3 Supplier P&L",
                ),
                (
                    "What's the import-forecast MAPE per settlement period?",
                    NOT_MODELLED,
                    LIMITS_SCREEN,
                ),
                (
                    "Does turn-down cut the peak?",
                    _reading(table, "Smart ÷ unmanaged peak"),
                    "Smart charging ▸ 2 Response",
                ),
                (
                    "What's the smart-tariff margin per customer?",
                    _reading(table, "Supplier gain per customer, before fee"),
                    "Supplier ▸ 3 Supplier P&L",
                ),
                (
                    "What share of flexible load moved?",
                    _reading(table, "Home import moved"),
                    "Smart charging ▸ 2 Response",
                ),
                (
                    "How firm is the capacity?",
                    _firm_reading(table, "Evening turn-down, 1 h"),
                    "Supplier ▸ 4 Firm MW",
                ),
                (
                    "What's CAC, churn or cost to serve?",
                    NOT_MODELLED,
                    LIMITS_SCREEN,
                ),
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    _drill_through(st, (("Supplier", "3 Supplier P&L"), ("Supplier", "4 Firm MW")))


# --- Charger maker ----------------------------------------------------------------


def render_charger_maker(st: Any, result: Any) -> None:
    """Charger maker: everything on Supplier ▸ 5 Charger makers, summarised."""

    _intro(st, "Buys the smart-charging software as a feature on its own chargers.")
    table = key_stats_table(result)
    # The wants table's own row (below), not a tile here: this persona's KPIs
    # are all per-device-per-month, so there is no per-year figure to reuse
    # the way Driver's "Value per year" tile feeds its own row.
    driver_saving_row = stat_row(
        result.household_outcomes_summary,
        group_id="fleet",
        metric="saving_gbp_per_year",
        statistic="p50_of_ev_means",
    )

    gross = stat_row(
        result.partner_summary,
        group_id="fleet",
        metric="gross_flex_gbp_per_enrolled_device_per_month",
    )
    earning = stat_row(result.partner_summary, group_id="fleet", metric="share_earning")
    per_kw = stat_row(
        result.partner_summary, group_id="fleet", metric="gbp_per_kw_charger_per_year"
    )
    fleet_share = 0.0 if earning is None or missing(earning["p50"]) else float(earning["p50"])
    months = growth.months_to_fund_discount(
        FUNNEL_DEFAULTS["discount_gbp"],
        gbp_per_enrolled_device_per_month_by_world=result.partner_world.loc[
            result.partner_world["group_id"].eq("fleet")
        ]["gross_flex_gbp_per_enrolled_device_per_month"].to_numpy(dtype=float),
        take_rate=FUNNEL_DEFAULTS["take_percent"] / 100.0,
    )
    p10_months, p50_months, p90_months, funded_weeks = months

    columns = kpi_columns(st, 4)
    kpi(
        columns[0],
        "Gross cash per device",
        p50_text(gross, lambda v: money(v, decimals=2)),
        unit="/month",
        context=band_text(gross, lambda v: money(v, decimals=2)),
        help="Gross flexibility cash per enrolled device per month, fleet-wide, before the "
        "customer share and penalties.",
    )
    kpi(
        columns[1],
        "Devices earning",
        p50_text(earning, lambda v: percent(100 * v)),
        context=band_text(earning, lambda v: percent(100 * v)),
        help="Share of enrolled devices that turned down in at least one settled half-hour.",
    )
    kpi(
        columns[2],
        "Per unit of charger power",
        p50_text(per_kw, lambda v: money(v, signed=True)),
        unit="/kW/year",
        context=band_text(per_kw, lambda v: money(v, signed=True)),
        help="Gross cash per kW of installed home charger power per year, extrapolated from "
        "one simulated week.",
    )
    kpi(
        columns[3],
        "Months to fund discount",
        UNAVAILABLE if missing(p50_months) else f"{p50_months:,.1f}",
        unit=None if missing(p50_months) else "months",
        context=""
        if missing(p50_months)
        else (
            f"{money(FUNNEL_DEFAULTS['discount_gbp'])} sticker discount paid back from "
            "flexibility cash; illustrative levers"
        ),
        help="Illustrative growth-calculator levers (Supplier ▸ 5 Charger makers' defaults): "
        f"{FUNNEL_DEFAULTS['take_percent']:g}% take. "
        f"{funded_weeks} of the run's simulated weeks fund it.",
    )

    markets = result.revenue_by_market_summary
    per_device = markets.loc[
        markets["metric"].eq("gbp_per_enrolled_device_per_month") & markets["modelled"]
    ]
    unmodelled = markets.loc[~markets["modelled"]].drop_duplicates("market")["market"]
    chart_block(
        st,
        _market_figure(per_device),
        title="Fleet revenue by market, mean £ per enrolled device per month",
        caption="Means add to the net; quantiles do not. Not modelled, not £0: "
        + ", ".join(_MARKET_LABELS.get(m, m) for m in unmodelled)
        + ".",
        frame=per_device,
        definition="Fleet-level ledger buckets by market, per enrolled device per month.",
        height=CHART_HEIGHTS["time_series"],
        key="partners-persona-charger-maker-market",
    )
    grid_events = per_device.loc[per_device["market"].eq("grid_events"), "mean"]
    if len(grid_events) and float(grid_events.iloc[0]) == 0.0:
        st.caption("Grid events £0: none paid out this run (Edit assumptions ▸ Events).")

    left, right = st.columns(2)
    with left:
        exceedance = result.household_value_exceedance
        exceedance_rows = exceedance.loc[exceedance["group_id"].eq("fleet")]
        chart_block(
            st,
            _exceedance_figure(exceedance_rows, THRESHOLD_DEFAULT),
            title="Probability and cost of a guaranteed £ per month, fleet",
            caption="P50 with P10–P90 across simulated weeks; scaled from one week, not a "
            "forecast.",
            frame=exceedance_rows,
            definition="Customer value under the pass-through reading; the supplier's own "
            "reward is on Supplier P&L and is not included here.",
            height=CHART_HEIGHTS["time_series"],
            key="partners-persona-charger-maker-exceedance",
        )
    with right:
        stages = growth.funnel(
            eligible=FUNNEL_DEFAULTS["eligible"],
            invited_share=FUNNEL_DEFAULTS["invited_percent"] / 100.0,
            signed_up_share=FUNNEL_DEFAULTS["signed_up_percent"] / 100.0,
            active_share=FUNNEL_DEFAULTS["active_percent"] / 100.0,
            share_earning=fleet_share,
        )
        chart_block(
            st,
            _funnel_figure(stages),
            title="From eligible devices to devices that earn",
            caption="Illustrative levers: the defaults from Supplier ▸ 5 Charger makers, not "
            "run inputs.",
            frame=stages,
            definition="Each stage is a share of the one before it; earning uses the run's "
            "own share of devices earning.",
            height=CHART_HEIGHTS["diagram"],
            key="partners-persona-charger-maker-funnel",
        )

    active_count = float(stages.loc[stages["stage"].eq("active"), "count"].iloc[0])
    payment = stat_row(
        result.supplier_pnl_summary,
        hedge_variant=_HEDGE_VARIANT,
        metric="customer_payment_per_customer_per_month_gbp",
    )
    if payment is not None and not missing(payment["p50"]):
        per_customer = growth.cumulative_payout_fan(
            monthly_gbp_p10=-float(payment["p90"]),
            monthly_gbp_p50=-float(payment["p50"]),
            monthly_gbp_p90=-float(payment["p10"]),
            months=PAYOUT_MONTHS,
        )
        fleet = per_customer.assign(
            **{
                column: per_customer[column] * active_count
                for column in ("cumulative_p10", "cumulative_p50", "cumulative_p90")
            }
        )
        chart_block(
            st,
            _payout_figure(per_customer, fleet),
            title="Customer payouts over 12 months",
            caption="A scenario that repeats the simulated week, not a forecast. Illustrative "
            "levers.",
            frame=per_customer.merge(fleet, on="month", suffixes=(" per customer", " active")),
            definition="The supplier's customer payment per customer per month, P10/P50/P90 "
            "across weeks, times the month number.",
            height=CHART_HEIGHTS["small_multiple_panel"] + 80,
            key="partners-persona-charger-maker-payout",
        )
    else:
        st.caption("Payout fan unavailable: the customer payment is unset.")

    firmness = getattr(result, "firmness_by_manufacturer", None)
    if firmness is not None:
        maker_rows = firmness.loc[
            firmness["direction"].eq(_DIRECTION) & firmness["duration_hours"].eq(_DURATION_HOURS)
        ]
        st.markdown("**Physical firmness by maker, 1 h turn-down**")
        # Fable review B3: firmness is a share (delivered ÷ deliverable), so
        # it reads as a percent, matching Trading ▸ 3's own firmness table
        # (trading_pnl._firmness_table) rather than a bare 0-1 ratio.
        firmness_columns = {
            "firmness_p10": "Firmness, P10 (firm)",
            "firmness_p50": "Firmness, P50",
            "firmness_p90": "Firmness, P90",
        }
        firmness_table = maker_rows[["manufacturer_label", "ev_count", *firmness_columns]].rename(
            columns={"manufacturer_label": "Maker", "ev_count": "EVs", **firmness_columns}
        )
        for column in firmness_columns.values():
            firmness_table[column] = [
                UNAVAILABLE if missing(value) else percent(100 * value)
                for value in firmness_table[column]
            ]
        st.dataframe(firmness_table, hide_index=True, width="stretch")

    _wants_heading(st, "a charger maker")
    st.dataframe(
        _wants_table(
            [
                (
                    "£ per driver per year saved vs unmanaged",
                    p50_text(driver_saving_row, lambda v: f"{money(v)} a year, median customer"),
                    "Supplier ▸ 6 Customers",
                ),
                (
                    "Share of energy in the cheapest window",
                    _reading(table, "Charging in the cheapest third"),
                    KEY_STATS_SCREEN,
                ),
                (
                    "Ready-by-time hit rate",
                    _reading(table, "Charge completed by departure, smart"),
                    "Supplier ▸ 5 Charger makers",
                ),
                (
                    "Plug-in rate uplift from the product",
                    NOT_MODELLED,
                    LIMITS_SCREEN,
                ),
                (
                    "Override or boost rate",
                    NOT_MODELLED,
                    LIMITS_SCREEN,
                ),
                (
                    "Subscription conversion, support tickets, integration cost",
                    NOT_MODELLED,
                    LIMITS_SCREEN,
                ),
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    _drill_through(st, (("Supplier", "5 Charger makers"),))


# --- Carmaker ---------------------------------------------------------------------


def render_carmaker(st: Any, result: Any) -> None:
    """Carmaker / telematics: driver £, plug-in behaviour, departure SoC, battery, carbon."""

    _intro(
        st,
        "Buys plug-in and state-of-charge (SoC) telematics insight as a feature in its own app.",
    )
    table = key_stats_table(result)

    soc_kpi = stat_row(
        result.plug_in_summary.kpis, day_type="all", metric="median_plug_in_soc_percent"
    )
    cnz_soc = assumption_value(result, "cnz_median_plug_in_soc_percent")
    nights = stat_row(
        result.household_outcomes_summary,
        group_id="fleet",
        metric="nights_plugged_share",
        statistic="group_ratio",
    )
    cycles = stat_row(
        result.partner_summary,
        group_id="fleet",
        metric="equivalent_full_cycles_per_ev_per_week_selected",
    )
    cycles_change = stat_row(
        result.partner_summary,
        group_id="fleet",
        metric="equivalent_full_cycles_per_ev_per_week_difference",
    )
    co2 = stat_row(result.carbon_shift_summary, metric="co2_shifted_kg_per_ev_per_month")

    columns = kpi_columns(st, 4)
    kpi(
        columns[0],
        "Median SoC at plug-in",
        p50_text(soc_kpi, lambda v: percent(v, decimals=1)),
        context=None if cnz_soc is None else f"CNZ observed {percent(float(cnz_soc))}",
        help="Median battery SoC when EVs plug in, unmanaged path, across simulated weeks.",
    )
    kpi(
        columns[1],
        "Nights plugged in",
        p50_text(nights, lambda v: percent(100 * v)),
        context="of all EV-nights, fleet pooled",
        help=(
            "Share of the fleet's EV-nights with the EV plugged in at home (all EVs pooled), "
            "P50 across simulated weeks."
        ),
    )
    change_text = (
        ""
        if cycles_change is None or missing(cycles_change["p50"])
        else f"change vs unmanaged {format_quantity(cycles_change['p50'], '', decimals=3)}, "
        "paired per week"
    )
    kpi(
        columns[2],
        "Battery cycles, smart",
        p50_text(cycles, lambda v: format_quantity(v, "", decimals=2)),
        unit="/week",
        context=change_text,
        help="Equivalent full cycles per EV per week, smart path; the paired change should be "
        "small, because smart charging moves energy in time, not amount.",
    )
    kpi(
        columns[3],
        "CO₂ shifted per EV",
        p50_text(co2, lambda v: format_quantity(v, "", decimals=1)),
        unit="kg/month",
        help="Illustrative synthetic carbon intensity tied to the model's own net demand; not "
        "observed grid data. Scaled from one simulated week, not a forecast.",
    )

    chart_block(
        st,
        _sessions_figure(
            result,
            metric="dwell_hours",
            axis_title="Hours plugged in",
            day_type=_DAY_TYPE,
            suffix=None,
        ),
        title="Dwell (hours plugged in) by archetype, % of sessions",
        caption="Weekday sessions, unmanaged path. Bars: P50 share per bin; whiskers: "
        "P10–P90 across weeks.",
        frame=result.session_distribution_bands.loc[
            result.session_distribution_bands["day_type"].eq(_DAY_TYPE)
            & result.session_distribution_bands["metric"].eq("dwell_hours")
        ],
        definition="Per simulated week, the share of each archetype's sessions in each dwell "
        "bin, then P10/P50/P90 across weeks.",
        height=CHART_HEIGHTS["small_multiple_panel"] * 3 + 90,
        key="partners-persona-carmaker-dwell",
    )

    left, right = st.columns(2)
    with left:
        heatmap = result.plug_in_half_hour_heatmap
        chart_block(
            st,
            _heatmap_figure(heatmap, verb="plugs in", colour_top=_shared_colour_top([heatmap])),
            title="When drivers plug in, % of EVs per half-hour",
            caption="Unmanaged path. P50 across simulated weeks; hover for P10–P90.",
            frame=heatmap,
            definition="Events in each weekday and London half-hour divided by EV-days: the "
            "chance an EV plugs in then.",
            height=CHART_HEIGHTS["histogram"],
            key="partners-persona-carmaker-heatmap",
        )
    with right:
        bands = result.session_distribution_bands
        departure_rows = bands.loc[
            bands["group_id"].eq("fleet")
            & bands["day_type"].eq("all")
            & bands["metric"].isin(["departure_soc_percent", "departure_soc_percent_smart"])
        ]
        chart_block(
            st,
            _departure_figure(departure_rows),
            title="SoC at departure, both paths, fleet, % of sessions",
            caption="Bars P50, whiskers P10–P90 across weeks; sessions ending in the study.",
            frame=departure_rows[["metric", "bin_label", "share_p10", "share_p50", "share_p90"]],
            definition="State of charge (SoC): battery energy at each session's last connected "
            "half-hour ÷ capacity, unmanaged and smart paths of the same sessions.",
            height=CHART_HEIGHTS["small_multiple_panel"] + 60,
            key="partners-persona-carmaker-departure-soc",
        )

    left, right = st.columns(2)
    with left:
        hour_bands = result.plug_in_summary.hour_bands
        cnz_hour = assumption_value(result, "cnz_weekday_plug_in_mode_local_hour")
        chart_block(
            st,
            _hour_figure(hour_bands, day_type=_DAY_TYPE, cnz_hour=cnz_hour),
            title="Plug-in time (London), share per hour",
            caption="Weekday, unmanaged path. Bars: mean share; whiskers: P10–P90 across weeks.",
            frame=hour_bands.loc[hour_bands["day_type"].eq(_DAY_TYPE)],
            definition="Share of weekday unmanaged-path plug-ins starting in each London hour.",
            height=CHART_HEIGHTS["histogram"],
            key="partners-persona-carmaker-hour",
        )
    with right:
        summary = result.household_outcomes_summary
        cohorts = [cohort for cohort in dict.fromkeys(summary["group_id"]) if cohort != "fleet"]
        fleet_mid = stat_row(
            summary, group_id="fleet", metric="evening_turn_down_kw_1h", statistic="p50_across_evs"
        )
        if fleet_mid is None or missing(fleet_mid["p50"]):
            st.markdown("**Flexible kW per customer, by archetype**")
            st.caption(FIRM_MW_PENDING)
        else:
            from .customers import _kw_figure, _kw_table

            # Fable review B2: an empty labels dict fell through to the raw
            # cohort_id ("average_uk", "infrequent_charging") on screen.
            labels = cohort_labels(result)
            chart_block(
                st,
                _kw_figure(summary, cohorts, labels),
                title="Flexible kW per customer, by archetype",
                caption="Realised 1 h evening flexibility per customer, smart path; P10–P90 "
                "across customers, medians across weeks.",
                frame=_kw_table(summary, cohorts),
                definition="Each customer's kW this hour if called, realised on the smart "
                "path; 0 on a night its maker is out or when it ignores the plan.",
                height=CHART_HEIGHTS["histogram"],
                key="partners-persona-carmaker-kw",
            )
            # Clarity critique: as Supplier ▸ 6 Customers, turn-down near 0
            # is smart charging having already moved the evening charge, not
            # a broken chart.
            st.caption(
                "Turn-down reads near 0 here: smart charging already moved this customer's "
                "evening charge to cheaper half-hours."
            )

    _wants_heading(st, "a carmaker")
    st.dataframe(
        _wants_table(
            [
                (
                    "Median plug-in SoC",
                    _reading(table, "Median SoC at plug-in"),
                    "Drivers ▸ Plug-ins",
                ),
                (
                    "Ready-by-time and SoC at departure",
                    _reading(table, "Charge completed by departure, smart"),
                    "Supplier ▸ 5 Charger makers",
                ),
                (
                    "Time at high SoC (battery health)",
                    NOT_MODELLED,
                    LIMITS_SCREEN,
                ),
                (
                    "Home vs public energy share",
                    "Not shown as a share; home and public kWh are separate readings",
                    "Drivers ▸ One EV",
                ),
                (
                    "CO₂ shifted per EV",
                    _reading(table, "CO₂ shifted per EV, P50 week"),
                    "Supplier ▸ 3 Supplier P&L",
                ),
                (
                    "Telematics API load, app funnel, warranty impact",
                    NOT_MODELLED,
                    LIMITS_SCREEN,
                ),
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    _drill_through(st, (("Drivers", "Sessions"), ("Drivers", "Plug-ins")))


# --- Fleet and leasing --------------------------------------------------------------


def render_fleet_and_leasing(st: Any, result: Any, *, household_card: HouseholdCardFn) -> None:
    """Fleet/leasing operator: per-vehicle saving, readiness, top-ups and cycles.

    Thin on purpose (design brainstorm section 1): the six archetypes are
    household drivers from Axle's sheet, and there is no depot, site limit
    or shift pattern in this model, so this lens reuses the driver-level,
    per-vehicle numbers rather than inventing a fleet-operations view.
    """

    _intro(st, "Buys the smart-charging software across its leased fleet, same as one driver.")
    st.caption(
        "Household archetypes stand in for the fleet; no depot, site limit or shift pattern "
        "is modelled."
    )
    table = key_stats_table(result)

    saving_row = stat_row(
        result.household_outcomes_summary,
        group_id="fleet",
        metric="saving_gbp_per_year",
        statistic="p50_of_ev_means",
    )
    ready_row = stat_row(
        getattr(result, "charge_completion_summary", None),
        group_id="fleet",
        path_id="selected",
        departure="all",
    )
    public = stat_row(result.difference_weekly_bands, metric="public_import_kwh")
    cycles = stat_row(
        result.partner_summary,
        group_id="fleet",
        metric="equivalent_full_cycles_per_ev_per_week_selected",
    )

    columns = kpi_columns(st, 4)
    kpi(
        columns[0],
        "Saving per vehicle",
        p50_text(saving_row, lambda v: money(v)),
        unit="/year",
        context="Median vehicle's own average year, illustrative",
        help="Home-charging cost saving only, smart minus unmanaged. It leaves out the trading "
        "revenue share that Value per year adds. Each vehicle's own average over its "
        "simulated weeks first, then the median across vehicles.",
    )
    kpi(
        columns[1],
        "Ready at departure",
        UNAVAILABLE
        if ready_row is None or missing(ready_row["completed_share_p50"])
        else percent(100 * float(ready_row["completed_share_p50"])),
        help="Smart path: share of sessions that reached their target by unplug, P50 across "
        "simulated weeks.",
    )
    kpi(
        columns[2],
        "Public top-up change",
        p50_text(public, lambda v: format_quantity(v, "", decimals=1)),
        unit="kWh/week",
        context="Paired change, smart minus unmanaged",
        help="Fleet weekly public charging import, smart minus unmanaged, paired per "
        "simulated week.",
    )
    kpi(
        columns[3],
        "Battery cycles, smart",
        p50_text(cycles, lambda v: format_quantity(v, "", decimals=2)),
        unit="/week",
        help="Equivalent full cycles per EV per week, smart path, fleet.",
    )

    units = result.units
    unit_id = ev_picker(header.controls(st), units)
    if getattr(result, "household_ev_world", None) is None:
        st.info("This run has no customer outcomes for one vehicle's own week.")
    else:
        traits = units.loc[units["unit_id"].eq(unit_id)].iloc[0].to_dict()
        card = household_card(result, unit_id)
        chart_block(
            st,
            _outcomes_figure(card, str(traits["cohort_label"])),
            title="This vehicle, its archetype and the fleet, in a typical week",
            caption=(
                f"This vehicle: P10–P90 across {card.world_count} weeks. Archetype and fleet: "
                "P10–P90 across vehicles, medians across weeks. Illustrative."
            ),
            frame=card.outcomes,
            definition="Pass-through reading: the saving plus the customer share of trading "
            "revenue, as if day-ahead prices reached the customer.",
            height=2 * CHART_HEIGHTS["small_multiple_panel"] + 64 + 28,
            key="partners-persona-fleet-outcomes",
        )

    left, right = st.columns(2)
    with left:
        chart_block(
            st,
            _sessions_figure(
                result,
                metric="flexible_kwh",
                axis_title="Flexible energy (kWh, battery)",
                day_type=_DAY_TYPE,
                suffix=None,
            ),
            title="Flexible energy by archetype, % of sessions",
            caption="Weekday sessions, unmanaged path. Bars: P50 share per bin; whiskers: "
            "P10–P90 across weeks.",
            frame=result.session_distribution_bands.loc[
                result.session_distribution_bands["day_type"].eq(_DAY_TYPE)
                & result.session_distribution_bands["metric"].eq("flexible_kwh")
            ],
            definition="Per simulated week, the share of each archetype's sessions with this "
            "much flexible (battery-side) energy, then P10/P50/P90 across weeks.",
            height=CHART_HEIGHTS["small_multiple_panel"] * 3 + 90,
            key="partners-persona-fleet-flexible-kwh",
        )
    with right:
        chart_block(
            st,
            _strip_figure(
                result.cost_effect,
                material_share_percent=assumption_value(
                    result, "not_recovered_material_share_percent"
                ),
            ),
            title="Illustrative weekly saving (£)",
            caption=f"One point per simulated week ({len(result.cost_effect)}). Illustrative.",
            frame=_cost_points_table(result.cost_effect),
            definition="Unmanaged minus smart total cost, one point per simulated week.",
            height=CHART_HEIGHTS["strip"],
            key="partners-persona-fleet-cost-strip",
        )

    _wants_heading(st, "a fleet or leasing operator")
    st.dataframe(
        _wants_table(
            [
                (
                    "What's a vehicle worth per year?",
                    # Same metric the "Saving per vehicle" tile above already
                    # reads (saving_gbp_per_year, home-charging cost only):
                    # this operator's own worth question is a saving, not the
                    # pass-through "Value per year" a driver-facing persona uses.
                    p50_text(saving_row, lambda v: f"{money(v)} a year, median vehicle"),
                    "Supplier ▸ 6 Customers",
                ),
                (
                    "Ready-for-shift (charged by departure)",
                    _reading(table, "Charge completed by departure, smart"),
                    "Drivers ▸ Household",
                ),
                (
                    "Per-site reporting and depot limits",
                    NOT_MODELLED,
                    LIMITS_SCREEN,
                ),
                (
                    "Battery cycles, smart",
                    _reading(table, "Equivalent full cycles, smart"),
                    "Supplier ▸ 5 Charger makers",
                ),
                (
                    "Public top-up change",
                    _reading(table, "Public top-up change"),
                    "Smart charging ▸ 2 Response",
                ),
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    _drill_through(st, (("Drivers", "One EV"), ("Drivers", "Household")))
