"""Smart charging lens 3, Value and risk: what the action cost or saved,
and the driver-side risk it carries (Q7, decision 0004 item 43).

Reads ``action_summary`` (the guard and shared summary sentence),
``cost_effect``, ``cost_effect_summary``, ``smart_charging_summary``,
``not_recovered_summary`` (contract 4.9a: unrecovered kWh, share and sessions
affected, every week, not only flagged ones), ``not_recovered_world_count``
(materially not-recovered weeks), the ``public_charge_gbp_per_kwh`` and
``not_recovered_material_share_percent`` assumptions
(docs/contracts/results-v2.md section 9). Decision 0003 rules out an
aggregate labelled that way; every money label on this page says
"illustrative", and the ``_CAVEAT`` text below is the only place that other
phrase appears, always as a negation.

The summary sentence comes from ``action_decision``, which owns the one
shared text-mapping helper for all three lenses.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..components.chart_table import chart_block
from ..components.kpi import kpi, kpi_columns
from ..style import (
    CHART_FONT_PX,
    CHART_HEIGHTS,
    FLAG_LABEL,
    MUTED_INK,
    PATH_LABELS,
    RISK,
    SERIES_COLOURS,
    assumption_value,
    format_quantity,
    money,
    percent,
)
from .action_decision import _summary_sentence
from .key_stats import _kpi as _key_stats_kpi

_CAVEAT = "Positive is a saving. Illustrative; not a bid, settlement or Axle cash."
"""The strip's sign and status note (decision 0003: never Axle cash)."""
_COMPONENT_CAVEAT = "Negative is cheaper."
"""Sign note for the cost-change table, which stays smart minus unmanaged."""
_COMPONENT_LABELS = {
    "home_import_cost": "Home import cost",
    "public_charge_cost": "Public charge cost",
    "unrecovered_energy_value": "Unrecovered energy value",
    "unserved_travel_value": "Unserved travel value",
    "total": "Illustrative total cost change",
}
_TOTAL_DEFINITION = (
    "Illustrative total = home import cost + public charge cost + unrecovered energy value + "
    "unserved travel value, per week."
)
_TIMED_COSTING_NOTE = (
    "Timed tariff is costed the same way as Smart: at the day-ahead price, with no tariff "
    "rate applied."
)
"""Shown once, visibly, whenever this lens shows a Timed tariff money figure (decision 0007):
"timed tariff" names the common real-world pattern of a timed tariff, but no tariff price is
modelled, so the page says so plainly rather than leaving "tariff" to be read as a retail rate."""
_STRIP_PANEL_GAP_PX = 36
_TWO_POLICY_STRIP_HEIGHT = (
    CHART_HEIGHTS["strip"] + CHART_HEIGHTS["small_multiple_panel"] + _STRIP_PANEL_GAP_PX
)
"""The strip figure stacked two rows high (Timed tariff above Smart, decision 0007 display
order) when the run has the optional timed path: the single-row height plus one more
small-multiple-sized row and the gap between them, the same reasoning
``action_decision._BAND_FIGURE_HEIGHT`` uses for its three stacked panels."""


def _total_row(summary: pd.DataFrame) -> pd.Series:
    return summary.loc[summary["component"].eq("total")].iloc[0]


def _saving_stats(total: pd.Series) -> tuple[float, float, float]:
    """The weekly saving's (P10, P50, P90) from the cost total's own quantiles.

    One quantity, one sign and one word across Overview, this lens and Key
    stats C (final critique B-9): the saving is unmanaged minus smart cost,
    positive when smart is cheaper. The model's total is smart minus
    unmanaged, so the saving is its negative, and negation swaps the tails
    (saving P10 = −cost P90). Exact for quantiles; nothing is re-quantiled.
    """

    return -float(total["p90"]), -float(total["p50"]), -float(total["p10"])


def _kpi_row(st: Any, result: Any) -> None:
    """The lens's four tiles (polish plan G9): value, weeks at risk, energy, early departures.

    The other magnitudes (unrecovered share, sessions affected, shortfall
    energy) are in the risk table below (``_risk_table``), so no row holds
    more than four tiles and each tile answers one question.

    With the optional timed path present (decision 0007, model step 2), ``timed_cost_effect``'s
    own saving (``timed_cost_effect_summary``, same schema and sign convention as the ordinary
    cost effect, "timed" in the selected slot) joins the value tile's context and tooltip
    beside smart's own figure, never instead of it.
    """

    saving_p10, saving_p50, saving_p90 = _saving_stats(_total_row(result.cost_effect_summary))
    not_recovered = result.not_recovered_summary.set_index("metric")
    early = result.smart_charging_summary.set_index("metric").loc["early_departure_count"]
    kwh = not_recovered.loc["unrecovered_kwh"]
    value_column, weeks_column, kwh_column, early_column = kpi_columns(st, 4)
    # Fleet-scale money in whole pounds (polish plan G6); the P10–P90 range
    # is the median tile's context line rather than a tile of its own.
    context = f"P10–P90 {money(saving_p10)} to {money(saving_p90)}"
    help_text = f"Unmanaged minus smart cost, per week, median across simulated weeks. {_CAVEAT}"
    # getattr, not a direct attribute (AGENTS.md convention this view already
    # follows for dispatch_bands etc.): the shared synthetic test fixture
    # (tests/fixtures/result_fixture.py) is deliberately two-policy and
    # carries no timed_* fields at all, only a real run's ForecastResult does.
    timed_summary = getattr(result, "timed_cost_effect_summary", None)
    if timed_summary is not None:
        timed_p10, timed_p50, timed_p90 = _saving_stats(_total_row(timed_summary))
        context += f" · timed tariff {money(timed_p50)}"
        help_text += (
            f" {PATH_LABELS['timed']}'s own median saving is {money(timed_p50)} (P10–P90 "
            f"{money(timed_p10)} to {money(timed_p90)}), unmanaged minus timed cost, the same "
            f"way. {_TIMED_COSTING_NOTE}"
        )
    kpi(
        value_column,
        "Illustrative saving, median",
        money(saving_p50),
        context=context,
        help=help_text,
    )
    material_share = assumption_value(result, "not_recovered_material_share_percent")
    kpi(
        weeks_column,
        "Weeks materially short",
        f"{result.not_recovered_world_count} of {result.world_count}",
        context=(
            "At or above the threshold"
            if material_share is None
            else f"shortfall of {material_share:g}% or more of the week's home import"
        ),
        help=_material_help(result),
    )
    kpi(
        kwh_column,
        "Unrecovered energy",
        format_quantity(kwh["p50"], "", decimals=1),
        unit=kwh["unit"],
        context="P50 across simulated weeks",
        help=_range_help(kwh, decimals=1),
    )
    kpi(
        early_column,
        "Early departures",
        format_quantity(early["p50"], "", decimals=1),
        unit=early["unit"],
        context=_early_departure_context(early, _key_stats_kpi(result, "plug_ins_per_week", "all")),
        help="Sessions that left before the smart plan finished. " + _range_help(early, 1),
    )


def _early_departure_context(early: pd.Series, sessions: pd.Series | None) -> str:
    """ "P50 across simulated weeks", plus the ratio to sessions a week when it is known.

    45.5 sessions a week sounds high in isolation; the denominator (a ratio
    of two medians, key_stats.py's own section E wording, since the result
    carries no per-week share to median directly) says it is about 1% of
    all sessions.
    """

    base = "P50 across simulated weeks"
    if sessions is None or pd.isna(sessions["p50"]) or sessions["p50"] <= 0:
        return base
    share = float(early["p50"]) / float(sessions["p50"])
    return f"{base}; of about {sessions['p50']:,.0f} sessions a week (about {100 * share:.1f}%)"


def _range_help(row: pd.Series, decimals: int) -> str:
    unit = row["unit"]
    low = format_quantity(row["p10"], unit, decimals=decimals)
    high = format_quantity(row["p90"], unit, decimals=decimals)
    return f"P10–P90 across {int(row['world_count'])} simulated weeks: {low} to {high}."


def _material_help(result: Any) -> str:
    """Help text for the material-weeks count, with the run's threshold (decision 0004 item 45)."""

    share = assumption_value(result, "not_recovered_material_share_percent")
    threshold = "the material threshold" if share is None else f"{share:g}%"
    return (
        "A week counts as materially short when the energy smart charging did not recover (a "
        f"lower closing battery, extra public top-ups or unserved travel) is at least {threshold} "
        "of that week's unmanaged home import. Smaller shortfalls are shown as a size, not "
        "flagged."
    )


def _risk_rows(not_recovered: pd.DataFrame, smart: pd.DataFrame | None) -> list[dict[str, str]]:
    """One policy's own risk rows: the three ``not_recovered_summary`` magnitudes, plus the
    two early-departure rows when ``smart`` (``smart_charging_summary``) is given.

    Early departure has no reading for the optional timed path (decision 0007 does not define
    one for the hard-barred rule: there is no smart plan for a session to leave early from), so
    ``smart=None`` stops after the three shared rows.
    """

    indexed = not_recovered.set_index("metric")
    rows: list[dict[str, str]] = []

    def add(label: str, row: pd.Series, decimals: int) -> None:
        unit = row["unit"]
        rows.append(
            {
                "Measure": label,
                "P50": format_quantity(row["p50"], unit, decimals=decimals),
                "P10–P90": (
                    f"{format_quantity(row['p10'], unit, decimals=decimals)} to "
                    f"{format_quantity(row['p90'], unit, decimals=decimals)}"
                ),
            }
        )

    add("Unrecovered energy", indexed.loc["unrecovered_kwh"], 1)
    share = indexed.loc["unrecovered_share"]
    rows.append(
        {
            "Measure": "Share of weekly home import unrecovered",
            "P50": percent(100 * share["p50"], decimals=2),
            "P10–P90": (
                f"{percent(100 * share['p10'], decimals=2)} to "
                f"{percent(100 * share['p90'], decimals=2)}"
            ),
        }
    )
    add("Sessions affected", indexed.loc["sessions_affected_count"], 1)
    if smart is not None:
        add("Early departures", smart.loc["early_departure_count"], 1)
        add("Early-departure shortfall", smart.loc["early_departure_shortfall_kwh"], 1)
    return rows


def _risk_table(
    not_recovered_summary: pd.DataFrame,
    smart_summary: pd.DataFrame,
    timed_not_recovered_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Driver-side risk magnitudes as one small table: P50 and P10–P90 across weeks.

    From ``not_recovered_summary`` (contract 4.9a, item 45: every week, not
    only flagged ones) and ``smart_charging_summary`` (contract 4.5, item 43);
    both are already world-first statistics, so this only formats them.
    ``unrecovered_share`` is shown as a percentage at 2 dp, not the
    contract's raw fraction (goal review action 20, N4): a typical shortfall
    is well under 1% of weekly home import, and 0 dp would print "0%".

    With the optional timed path present (decision 0007, model step 2), its own
    ``timed_not_recovered_summary`` (contract 4.9b, same three metrics, same reading) adds a
    second "Policy" block ahead of Smart's own rows (display order: Timed tariff before
    Smart); ``None`` (the default) leaves the table exactly as before the optional path
    existed, with no "Policy" column at all.
    """

    smart = smart_summary.set_index("metric")
    rows = pd.DataFrame(_risk_rows(not_recovered_summary, smart))
    if timed_not_recovered_summary is None:
        return rows
    rows.insert(0, "Policy", PATH_LABELS["selected"])
    timed_rows = pd.DataFrame(_risk_rows(timed_not_recovered_summary, None))
    timed_rows.insert(0, "Policy", PATH_LABELS["timed"])
    return pd.concat([timed_rows, rows], ignore_index=True)


def _no_negative_zero(values: pd.Series, decimals: int) -> pd.Series:
    """Round to a hovertemplate's own precision, clearing a lone "-" off zero.

    Mirrors ``action_response._no_negative_zero``: d3-format prints "-0" for
    a raw float in (-0.5 * 10**-decimals, 0) even though it displays as 0
    everywhere else, so a trace with a hovertemplate must carry values
    already rounded, with the resulting negative zero cleared by adding 0.0
    (IEEE 754: -0.0 + 0.0 == 0.0).
    """

    return values.round(decimals) + 0.0


def _strip_traces(
    cost: pd.DataFrame,
    *,
    colour_range: tuple[float, float],
    show_scale: bool,
    colourbar_shown: bool,
    legend_seen: set[str] | None,
) -> tuple[list[go.Scatter], bool]:
    """The Recovered/Flagged marker traces for one policy's own cost_effect frame (contract
    4.8/4.8a): jitter, colour and symbol, shared by the Smart-only strip and the two-row
    Smart/Timed tariff strip (decision 0007, model step 2).

    ``colour_range`` and ``colourbar_shown`` are threaded across both panels so the two frames
    share one colour scale with exactly one colourbar between them; the updated
    ``colourbar_shown`` is returned for the next panel's call. ``legend_seen`` does the same
    for the legend, except in the single-panel case (``None``), where every trace keeps its
    own legend entry exactly as before the optional path existed.
    """

    ordered = cost.sort_values("world_id")
    # Deterministic jitter purely to spread overlapping markers apart; it is a
    # display position only, never data, so it must not depend on anything
    # random (tests need the same figure every run).
    jitter = (ordered["world_id"].to_numpy() % 7) / 7.0 - 0.5
    # Colour by shortfall size, not only the material/not-material split
    # (goal review actions 2, 47): a flat two-colour marker could not show
    # that one flagged week fell short by far more than another. NaN only
    # when that week's unmanaged-path home import is 0 kWh (contract 4.8), a
    # near-impossible edge case at fleet scale; drawn as 0% so it never
    # crashes the colour scale.
    share_percent = ordered["unrecovered_share"].fillna(0.0).to_numpy(dtype=float) * 100.0
    traces: list[go.Scatter] = []
    for flagged, name, symbol in (
        # Symbol keeps the material/not-material split visible (decision
        # 0004 item 45) alongside the continuous colour; at fleet scale
        # almost every week has some tiny shortfall, so only material weeks
        # get the diamond.
        (False, "Recovered", "circle"),
        (True, FLAG_LABEL, "diamond"),
    ):
        mask = ordered["not_recovered_material"].eq(flagged).to_numpy()
        if not mask.any():
            continue
        subset = ordered.loc[mask]
        # Whole pounds at fleet scale (polish plan G6): the hover shows the
        # axis's own precision. x carries the hovertemplate's %{x:,.0f};
        # round it first so a value in (-0.5, 0) does not print "-0".
        # Plotted as the saving (unmanaged minus this policy), the same sign
        # as the tile above and Overview (B-9): the model's column is the
        # policy minus unmanaged, so negate it point by point.
        x = _no_negative_zero(-subset["illustrative_selected_minus_normal_total_gbp"], decimals=0)
        # world_id (int) and unrecovered share (float) share one customdata
        # array, so world_id is formatted %.0f in the hovertemplate below
        # rather than relying on its now-upcast float dtype (mirrors
        # one_ev.py's note on mixed-dtype customdata).
        hover_data = pd.DataFrame(
            {"world_id": subset["world_id"], "share_percent": share_percent[mask]}
        ).to_numpy()
        show_legend = True if legend_seen is None else name not in legend_seen
        if legend_seen is not None:
            legend_seen.add(name)
        traces.append(
            go.Scatter(
                x=x,
                y=jitter[mask],
                mode="markers",
                marker={
                    **(
                        {
                            "color": share_percent[mask],
                            "cmin": colour_range[0],
                            "cmax": colour_range[1],
                            "colorscale": RISK,
                            "showscale": not colourbar_shown,
                            "colorbar": {"title": "Unrecovered<br>share", "ticksuffix": "%"},
                        }
                        if show_scale
                        else {"color": SERIES_COLOURS["normal"]}
                    ),
                    "symbol": symbol,
                    "size": 9,
                    "line": {"color": MUTED_INK, "width": 1},
                },
                name=name,
                # Plotly's default hides a lone trace's legend entry; when
                # every simulated week recovers, "Recovered" is the whole
                # figure's only trace, so this must be explicit rather than
                # left to that >1-trace default.
                showlegend=show_legend,
                customdata=hover_data,
                hovertemplate=(
                    "Simulated week %{customdata[0]:.0f}: £%{x:,.0f} saving (illustrative), "
                    "%{customdata[1]:.1f}% of home import unrecovered<extra></extra>"
                ),
            )
        )
        colourbar_shown = True
    return traces, colourbar_shown


def _strip_figure(
    cost: pd.DataFrame,
    *,
    material_share_percent: float | None = None,
    timed_cost: pd.DataFrame | None = None,
) -> go.Figure:
    """The weekly-saving strip: one marker per simulated week.

    With the optional timed path present (decision 0007, model step 2) and ``timed_cost`` (its
    own ``timed_cost_effect``) given, a second panel for it is stacked above the smart panel
    (display order: Timed tariff ahead of Smart), sharing one £ scale so the two savings are
    directly comparable, and one colour scale/colourbar for unrecovered share built across both
    frames. ``None`` (the default) draws exactly the single-panel chart this view drew before
    the optional path existed.
    """

    frames = [cost] if timed_cost is None else [timed_cost, cost]
    # The top of the risk scale is at least the material threshold (polish
    # plan G4), so red means "material", not merely "the largest tiny
    # shortfall this run happened to draw"; built across every panel so one
    # colour scale serves them all.
    threshold = material_share_percent or 0.0
    max_share = max(
        (frame["unrecovered_share"].fillna(0.0).max() * 100.0 for frame in frames), default=0.0
    )
    colour_range = (0.0, float(max(max_share, threshold, 1e-9)))
    # No material week in any panel: a graded colour would only rank
    # sub-threshold noise, and without its colourbar it would have no key,
    # so every marker is one neutral colour instead (the chart caption says
    # so). With a material week the graded colour and its colourbar are
    # shown together.
    show_scale = any(bool(frame["not_recovered_material"].any()) for frame in frames)

    if timed_cost is None:
        figure = go.Figure()
        traces, _ = _strip_traces(
            cost,
            colour_range=colour_range,
            show_scale=show_scale,
            colourbar_shown=not show_scale,
            legend_seen=None,
        )
        for trace in traces:
            figure.add_trace(trace)
        figure.update_xaxes(
            title="£ per week (illustrative), unmanaged minus smart",
            zeroline=True,
            # MUTED_INK, not the grid token: a zero line in the grid's own
            # colour would be indistinguishable from the gridlines themselves.
            zerolinecolor=MUTED_INK,
            zerolinewidth=1,
        )
        figure.update_yaxes(visible=False, range=[-1, 1])
        return figure

    # Two rows, Timed tariff above Smart (display order, decision 0007): the
    # same stacked-small-multiples pattern action_decision._band_absolute_figure
    # uses to add a row per category.
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[PATH_LABELS["timed"], PATH_LABELS["selected"]],
        vertical_spacing=_STRIP_PANEL_GAP_PX / _TWO_POLICY_STRIP_HEIGHT,
    )
    figure.update_annotations(font_size=CHART_FONT_PX)
    legend_seen: set[str] = set()
    colourbar_shown = not show_scale
    for row, frame in enumerate(frames, start=1):
        traces, colourbar_shown = _strip_traces(
            frame,
            colour_range=colour_range,
            show_scale=show_scale,
            colourbar_shown=colourbar_shown,
            legend_seen=legend_seen,
        )
        for trace in traces:
            figure.add_trace(trace, row=row, col=1)
    figure.update_xaxes(zeroline=True, zerolinecolor=MUTED_INK, zerolinewidth=1)
    figure.update_xaxes(title="£ per week (illustrative), unmanaged minus policy", row=2, col=1)
    figure.update_yaxes(visible=False, range=[-1, 1])
    return figure


def _cost_points_row_frame(cost: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Simulated week": cost["world_id"],
            "Illustrative saving (£)": (-cost["illustrative_selected_minus_normal_total_gbp"]).map(
                lambda value: money(value, signed=True)
            ),
            # The two columns the strip now colours by (goal review action 2):
            # shown alongside the point they belong to, not only as a colour.
            "Unrecovered (kWh)": cost["unrecovered_kwh"].round(1),
            "Unrecovered share": (100 * cost["unrecovered_share"]).round(0),
            "Materially not recovered": cost["not_recovered_material"],
        }
    )


def _cost_points_table(cost: pd.DataFrame, timed_cost: pd.DataFrame | None = None) -> pd.DataFrame:
    """The strip chart's "Data and definition" frame: one row per simulated week.

    With the optional timed path present (decision 0007, model step 2) and ``timed_cost``
    given, its own rows (same columns, same reading) are added in a leading "Policy" column
    ahead of smart's own rows (display order), matching the strip figure's two panels; ``None``
    (the default) leaves the frame exactly as before the optional path existed.
    """

    table = _cost_points_row_frame(cost)
    if timed_cost is None:
        return table
    table.insert(0, "Policy", PATH_LABELS["selected"])
    timed_table = _cost_points_row_frame(timed_cost)
    timed_table.insert(0, "Policy", PATH_LABELS["timed"])
    return pd.concat([timed_table, table], ignore_index=True)


def _components_table(
    summary: pd.DataFrame, timed_summary: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per cost component, smart minus unmanaged (contract 4.9).

    With the optional timed path present (decision 0007, model step 2), its own
    ``timed_cost_effect_summary`` (contract 4.9b, same five components, same sign convention:
    timed minus normal) gets an identical block in a leading "Policy" column, ahead of smart's
    own rows (display order: Timed tariff before Smart). ``None`` (the default) leaves the
    table exactly as before the optional path existed, with no "Policy" column at all.
    """

    table = pd.DataFrame(
        {
            "Component": summary["component"].map(_COMPONENT_LABELS),
            "P50": summary["p50"].map(lambda value: money(value, signed=True)),
            "P10": summary["p10"].map(lambda value: money(value, signed=True)),
            "P90": summary["p90"].map(lambda value: money(value, signed=True)),
        }
    )
    if timed_summary is None:
        return table
    table.insert(0, "Policy", PATH_LABELS["selected"])
    timed_table = pd.DataFrame(
        {
            "Policy": PATH_LABELS["timed"],
            "Component": timed_summary["component"].map(_COMPONENT_LABELS),
            "P50": timed_summary["p50"].map(lambda value: money(value, signed=True)),
            "P10": timed_summary["p10"].map(lambda value: money(value, signed=True)),
            "P90": timed_summary["p90"].map(lambda value: money(value, signed=True)),
        }
    )
    return pd.concat([timed_table, table], ignore_index=True)


def _public_rate(result: Any) -> float | None:
    return assumption_value(result, "public_charge_gbp_per_kwh")


def render_action_cost(st: Any, result: Any) -> None:
    """Render Value and risk: KPIs, the weekly saving strip, cost components and driver risk."""

    if result.action_summary is None:
        st.info("This result has no Axle action.")
        return

    st.markdown(_summary_sentence(result.action_summary))
    _kpi_row(st, result)

    # The optional timed path's own cost effect (decision 0007, model step
    # 2): None exactly when the run's timed_start_local_hour is unset, so
    # every block below reads the same absence to stay two-policy. getattr,
    # not a direct attribute: the shared synthetic test fixture
    # (tests/fixtures/result_fixture.py) is deliberately two-policy and
    # carries no timed_* fields at all, only a real run's ForecastResult does
    # (the same reason this view's dispatch-block sibling in action_response.py
    # reads dispatch_bands the same defensive way).
    timed_cost_effect = getattr(result, "timed_cost_effect", None)
    timed_cost_effect_summary = getattr(result, "timed_cost_effect_summary", None)
    timed_not_recovered_summary = getattr(result, "timed_not_recovered_summary", None)
    if timed_cost_effect is not None:
        st.caption(_TIMED_COSTING_NOTE)

    any_material = bool(result.cost_effect["not_recovered_material"].any()) or (
        timed_cost_effect is not None and bool(timed_cost_effect["not_recovered_material"].any())
    )
    colour_note = (
        "Coloured by unrecovered share: redder means more of that week's home import went "
        "unrecovered."
        if any_material
        else "No week reached the material threshold, so every week is drawn in one colour."
    )
    strip_height = CHART_HEIGHTS["strip"] if timed_cost_effect is None else _TWO_POLICY_STRIP_HEIGHT
    chart_block(
        st,
        _strip_figure(
            result.cost_effect,
            material_share_percent=assumption_value(result, "not_recovered_material_share_percent"),
            timed_cost=timed_cost_effect,
        ),
        title="Illustrative weekly saving (£)",
        caption=f"One point per simulated week ({len(result.cost_effect)}). {_CAVEAT}",
        frame=_cost_points_table(result.cost_effect, timed_cost_effect),
        definition=(
            f"{colour_note} One point per simulated week: the illustrative saving from smart "
            "charging (unmanaged minus smart total cost, the components table's total with "
            "the sign flipped), whether that week's deferred energy was fully recovered by "
            "the end of the week, and by how much it fell short when not. The components "
            "table below keeps home import cost apart from unrecovered energy value, so a "
            "shortfall's cost and its energy are never read as one number."
            + (
                ""
                if timed_cost_effect is None
                else " Timed tariff gets its own panel, above Smart's, reading the same way."
            )
        ),
        height=strip_height,
        key="action-cost-strip",
    )

    # A visible table, not a bar chart (polish plan G9): four components with
    # signed P10–P90 read more exactly as numbers, and the old chart was
    # mostly one bar.
    component_title = (
        "Illustrative cost change by component, smart minus unmanaged (£ per week)"
        if timed_cost_effect_summary is None
        else "Illustrative cost change by component, smart and timed tariff minus unmanaged "
        "(£ per week)"
    )
    st.markdown(f"**{component_title}**")
    st.dataframe(
        _components_table(result.cost_effect_summary, timed_cost_effect_summary),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"{_COMPONENT_CAVEAT} P50 and P10–P90 across simulated weeks. Components do not add "
        "to the total's quantiles.",
        help="Each component is summarised across simulated weeks on its own, so the four "
        f"P50s do not add up to the total's P50. {_TOTAL_DEFINITION}",
    )

    # Early departures are usually why a session appears in Sessions
    # affected (goal review N10): leaving before the plan finishes is the
    # main way a session's closing charge ends below the unmanaged path's.
    st.markdown(
        "**Driver-side risk**",
        help="An early departure is usually why a session appears in Sessions affected: leaving "
        "before the plan finishes is the main way its closing charge ends up below the "
        "unmanaged path's.",
    )
    st.dataframe(
        _risk_table(
            result.not_recovered_summary, result.smart_charging_summary, timed_not_recovered_summary
        ),
        width="stretch",
        hide_index=True,
    )

    rate = _public_rate(result)
    if rate is not None:
        st.caption(f"Public rate £{rate:.2f}/kWh is an editable illustrative assumption.")
