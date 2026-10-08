"""Overview ▸ At a glance: the landing lens (docs/DASHBOARD_DESIGN.md section 4.1).

What this owns: the KPI tiles, the "average day" chart (Q1: when EVs are
plugged in; Q4: how much drivers vary), a four-tile flexibility strip with a
link to Drivers ▸ Fleet week (decision 0004 item 54, plan C3, E1: the strip
replaces the flexible-power chart), and the "Next" links onto the rest of
the app. Reads ``average_day_bands``, ``plug_in_summary``, ``weekly_bands``
and ``flexibility_weekly_summary`` for what both models share;
``action_summary``, ``cost_effect_summary``, ``not_recovered_world_count``
and ``weekly_peak_summary`` (coincidence factor) on an action result only;
``model`` and ``world_count`` to pick which tiles to draw and how to caption
them; and ``assumptions`` for the CNZ context figure
(docs/contracts/results-v2.md). On a run whose ``timed_start_local_hour`` is
set, ``timed_cost_effect_summary`` adds a fourth "Timed tariff saving" tile
beside the Smart one (decision 0007), and ``weekly_peak_summary``'s optional
``timed`` row adds a figure to the coincidence tile's context; absent, both
stay exactly as before this path existed. It never calls the model. Overview
▸ Key stats lives in ``key_stats.py``.

Tested against ``tests/fixtures/result_fixture.py``'s ``make_result``, the
same SYNTHETIC stand-in ``model/summaries.py`` is itself checked against
(``tests/fixtures/result_contract.py``), so this view is written to the
documented field names only and needs no change when a real run replaces
the fixture in a test.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as _streamlit

from ..components import header
from ..components.chart_table import chart_block
from ..components.kpi import badge_line, kpi, kpi_columns
from ..registry import DAY_TYPE_KEY, DAY_TYPE_LABELS, PAGES
from ..style import (
    CHART_HEIGHTS,
    PATH_STYLES,
    PLUGGED_ALPHA,
    SERIES_COLOURS,
    UNAVAILABLE,
    UNMANAGED_GLOSSARY,
    assumption_value,
    band_and_line,
    format_quantity,
    kw,
    money,
    percent,
)

# local_half_hour is 0-47 counting London half-hours from local midnight
# (contract v2 section 3.2); 18:00 is therefore hour 18 * 2 half-hours.
EIGHTEEN_HUNDRED_LOCAL_HALF_HOUR = 36

_HALF_HOUR_AXIS_ORIGIN = pd.Timestamp("2000-01-01")
"""Presentation-only anchor date for the average-day x-axis (chart audit O4:
Plotly's unified hover header showed the raw half-hour index, e.g. "36",
instead of a time of day). ``local_half_hour`` carries no calendar date;
mapping it onto an arbitrary day lets Plotly's date-axis ``hoverformat`` show
"18:00" in the hover header, the same way the rest of the app formats real
UTC timestamps (style.hover_time_labels). Only the time-of-day component is
ever displayed -- the anchor date itself never appears on screen."""


MIDDAY_UNPLUG_NOTE = (
    "An EV unplugs only on a morning it drives. On a day with no trip it stays plugged in "
    "until its next trip, so low-mileage drivers are often still plugged in at midday. CNZ "
    "reports that about 30% of plug-ins come two or more days after the last one."
)
"""Model questions Q-20, the doc's wording: why the average day's weekday midday bars are low."""


def _half_hour_axis_x(local_half_hour: pd.Series) -> pd.Series:
    """Map a 0-47 local half-hour index to the presentation timestamp above."""

    return _HALF_HOUR_AXIS_ORIGIN + pd.to_timedelta(
        local_half_hour.to_numpy(dtype="int64") * 30, unit="min"
    )


# "Next" row (design 4.1): each link both switches page and preselects the
# named lens, using the session-state convention pages.py documents
# ("write the lens name to st.session_state[f'lens::{slug}'] before
# st.switch_page"). Slugs are read from the registry rather than hard-coded,
# so a page rename cannot silently break the link.
_NEXT_LINKS = (
    ("Plug-in times and SoC →", "Drivers", "Plug-ins"),
    ("One driver's week →", "Drivers", "One EV"),
    ("Why this action →", "Smart charging", "1 Plan"),
)


def _percent_of_fraction(value: float | None) -> str:
    """Format a 0-1 fraction as a whole-number percentage ("82%"), or "Unavailable"."""

    return percent(None if value is None else 100.0 * value)


def _average_day_row(
    result: Any, *, metric: str, day_type: str, local_half_hour: int | None = None
) -> pd.DataFrame:
    """Fleet, unmanaged-path (``normal``) rows of ``average_day_bands`` (contract 3.8).

    Unmanaged path only: the average-day chart shows population behaviour, not
    smart charging's effect (that is Smart charging, Response lens).
    """

    frame = result.average_day_bands
    mask = (
        frame["group_id"].eq("fleet")
        & frame["path_id"].eq("normal")
        & frame["metric"].eq(metric)
        & frame["day_type"].eq(day_type)
    )
    if local_half_hour is not None:
        mask &= frame["local_half_hour"].eq(local_half_hour)
    return frame.loc[mask].sort_values("local_half_hour").reset_index(drop=True)


def _plug_in_kpi(result: Any, *, metric: str, day_type: str) -> pd.Series | None:
    kpis = result.plug_in_summary.kpis
    row = kpis.loc[kpis["day_type"].eq(day_type) & kpis["metric"].eq(metric)]
    return row.iloc[0] if len(row) else None


def _weekly_unserved_travel_kwh_p50(result: Any) -> float:
    """Fleet weekly unserved travel, P50 across worlds, normal path.

    Reads the model-supplied ``weekly_bands`` (contract v2 section 3.6a,
    decision 0004 item 35's slim run records carry it too): one row per
    ``(path_id, metric)`` with cross-world statistics of the per-world weekly
    sum, so the view never sums or requantiles anything itself. No-action
    results only have the ``normal`` path.
    """

    bands = result.weekly_bands
    row = bands.loc[bands["path_id"].eq("normal") & bands["metric"].eq("unserved_travel_kwh")]
    return float(row["p50"].iat[0])


def _action_tile(summary: Any) -> tuple[str, str | None]:
    """("Smart charging", caption) for the Overview action tile (design 4.1)."""

    # Smart charging applies to every EV (decision 0004 item 38), so the tile
    # names the policy and its safety margin; there is no eligible share.
    value = "Smart charging"
    caption = f"Cheapest half-hours, ready {summary.departure_margin_hours:g} h before departure"
    return value, caption


def _average_day_figure(connected: pd.DataFrame, soc: pd.DataFrame) -> go.Figure:
    """Bars = share plugged in; line + band = SoC mean and spread across EVs.

    This is the brief's sketch 2 (design section 1, Q4): the SoC band is
    spread across EVs (population variation), never across simulated worlds,
    so it uses ``centre``/``low``/``high`` from ``average_day_bands`` exactly
    as the model computed them (never recalculated here).
    """

    soc_style = PATH_STYLES["normal"]
    connected_x = _half_hour_axis_x(connected["local_half_hour"])
    soc_x = _half_hour_axis_x(soc["local_half_hour"])
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=connected_x,
            y=connected["centre"] * 100.0,
            name="Plugged in at home",
            marker_color=SERIES_COLOURS["plugged"],
            opacity=PLUGGED_ALPHA,
            yaxis="y",
            hovertemplate="%{y:.0f}% plugged in<extra></extra>",
        )
    )
    # One legend entry for the SoC band and its mean (polish plan G3).
    band_and_line(
        figure,
        soc_x,
        soc["low"],
        soc["centre"],
        soc["high"],
        name="SoC mean, P5–P95 across EVs",
        colour=soc_style["color"],
        width=soc_style["width"],
        dash=soc_style["dash"],
        hovertemplate="%{y:.0f}% SoC<extra></extra>",
        yaxis="y2",
    )
    figure.update_layout(
        xaxis={
            "tickmode": "array",
            "tickvals": [
                _HALF_HOUR_AXIS_ORIGIN + pd.Timedelta(hours=hour) for hour in (0, 6, 12, 18)
            ],
            "ticktext": ["00:00", "06:00", "12:00", "18:00"],
            # O4: the hover header otherwise falls back to a full date/time
            # stamp for this presentation-only date axis.
            "hoverformat": "%H:%M",
            "title": "London time",
        },
        yaxis={"title": "Plugged in at home (%)", "range": [0, 100]},
        # Dual-axis right margin (design section 3.5, decision 0004 item 26);
        # the template's single-axis default (16 px) would clip this label.
        yaxis2={"title": "Battery SoC (%)", "overlaying": "y", "side": "right", "range": [0, 100]},
        margin={"r": 56},
    )
    return figure


def _hours(value: float | None) -> str:
    return format_quantity(value, "h", decimals=1)


def _ratio(value: float | None) -> str:
    return format_quantity(value, "", decimals=2)


def _flexibility_strip(st: Any, result: Any) -> None:
    """Four flexibility tiles for the unmanaged week (decision 0004 item 54, plan C3, E1).

    Every value is one column of ``flexibility_weekly_summary`` (contract v2
    3.6e) or ``weekly_peak_summary`` (4.5a), already per week first and then
    P50 across weeks; P10–P90 goes in help. The coincidence tile's context
    line names Smart's own figure beside the unmanaged one shown, and, on a
    run whose ``timed_start_local_hour`` is set, Timed tariff's too (decision
    0007); the four-tile row itself never grows a fifth tile.

    The peak tile reads ``peak_deferrable_kw``: the full charger power of
    plugged-in EVs still below target, summed (contract v2 3.6d ``total``).
    That is a one-half-hour capacity, not charging that is happening or
    that could wait: it includes EVs that must charge now, and it runs
    above the fleet's actual unmanaged import because an EV that plugs in
    or reaches target part-way through a half-hour draws for only part of
    it (final critique B-1). So the tile says "charger power below target",
    not "deferrable"; only the "2 h+" tile is power that can wait.
    """

    summary = result.flexibility_weekly_summary
    st.markdown("**Flexibility in the unmanaged week**")
    if summary is None:
        st.caption("This run has no flexibility figures: it ran without a departure margin.")
        return
    row = summary.iloc[0]
    weeks = int(row["world_count"])
    capacity = float(row["fleet_charger_capacity_kw"])

    def spread(prefix: str, formatter) -> str:
        low, high = formatter(row[f"{prefix}_p10"]), formatter(row[f"{prefix}_p90"])
        return f"P50 across {weeks} simulated weeks; P10 to P90: {low} to {high}."

    columns = kpi_columns(st, 4)
    peak_when = row["peak_modal_interval_start_london"]
    kpi(
        columns[0],
        "Peak power below target",
        kw(row["peak_deferrable_kw_p50"]),
        context=(
            f"full charger power of plugged-in EVs not yet at target; most often "
            f"{peak_when:%a %H:%M}"
            if not pd.isna(peak_when)
            else "full charger power of plugged-in EVs not yet at target"
        ),
        help="The busiest half-hour of each week, measured as the full charger power of every "
        "plugged-in EV still below its target. It is the most the fleet could draw, or hold "
        "back, in that one half-hour. It is not charging under way, and not all of it can "
        "wait. It runs above the actual unmanaged import, because an EV that plugs in or "
        "finishes part-way through a half-hour draws for only part of it. "
        + spread("peak_deferrable_kw", kw),
    )
    kpi(
        columns[1],
        "Can wait 2 h+ at 19:00",
        kw(row["deferrable_at_least_2h_kw_1900_p50"]),
        help="Charging at 19:00 that could wait at least 2 hours and still reach target before "
        "the EV leaves. This is the part of the fleet's charging that can move. "
        + spread("deferrable_at_least_2h_kw_1900", kw),
    )
    kpi(
        columns[2],
        "Hours at ¼ capacity or more",
        _hours(row["hours_at_least_quarter_capacity_p50"]),
        context=f"hours a week with ≥{capacity / 4:,.0f} kW of charger power still below target",
        help="Hours a week when the charger power below target is at least 25% of the fleet's "
        "home charger capacity. The 25% is a reporting threshold, not a model assumption. "
        + spread("hours_at_least_quarter_capacity", _hours),
    )
    peaks = result.weekly_peak_summary
    if peaks is None:
        kpi(
            columns[3],
            "Coincidence factor",
            UNAVAILABLE,
            context="after a smart-charging run",
            help="Each week's peak home import divided by the full power of the chargers "
            "plugged in at that half-hour. A run without smart charging does not carry it.",
        )
    else:
        normal = peaks.loc[peaks["path_id"].eq("normal")].iloc[0]
        smart = peaks.loc[peaks["path_id"].eq("selected")].iloc[0]
        context = (
            "peak import ÷ plugged-in charger power; smart "
            f"{_ratio(smart['coincidence_factor_p50'])}"
        )
        # Decision 0007: the Timed tariff figure joins Smart's in the same
        # context line, beside it rather than instead of it; absent (the
        # setting off, or a no-action result above), the line is unchanged.
        timed_rows = peaks.loc[peaks["path_id"].eq("timed")]
        if len(timed_rows):
            context += f"; timed tariff {_ratio(timed_rows.iloc[0]['coincidence_factor_p50'])}"
        kpi(
            columns[3],
            "Coincidence factor",
            _ratio(normal["coincidence_factor_p50"]),
            context=context,
            help="Each week's peak home import divided by the full power of the chargers "
            "plugged in at that half-hour. A value of 1 means every plugged-in charger is "
            f"drawing full power. P50 across {int(normal['coincidence_world_count'])} "
            "simulated weeks.",
        )
    st.button(
        "Charger power by slack through the week →",
        key="overview-to-fleet-week",
        on_click=_switch_to,
        args=("Drivers", "Fleet week", {"fleet-week-metric": "Deferrable power by slack"}),
    )


def _registry_page(page_name: str) -> Any:
    """The named ``Page`` from the registry (shared by every cross-page link below)."""

    return next(candidate for candidate in PAGES if candidate.name == page_name)


def _switch_to(page_name: str, lens: str, preset: dict[str, str] | None = None) -> None:
    """ "Next" button callback: preselect a lens, then land on that page.

    ``st.switch_page``/``st.page_link`` require a ``Page`` object for a page
    defined by a callable (this app has no file-based pages); Streamlit
    matches it to the running app's navigation by ``url_path`` alone, so a
    placeholder callable is enough here -- verified directly against the
    installed Streamlit (1.64.0) with a small AppTest, since the docstring's
    wording ("the source ... must match") is ambiguous about object
    identity. This avoids importing ``pages.page_renderer`` here, which would
    create a circular import once ``pages.py`` imports this module's
    ``render_overview`` (the same reasoning as run_controller's dialog import
    of ``views.parameters``).
    """

    page = _registry_page(page_name)
    _streamlit.session_state[f"lens::{page.slug}"] = lens
    # ``preset`` preselects the target view's own widgets (for example Fleet
    # week's metric), by the same write-before-render convention.
    for key, value in (preset or {}).items():
        _streamlit.session_state[key] = value
    _streamlit.switch_page(_streamlit.Page(lambda: None, url_path=page.slug))


# --- "Start here" guide (polish lane, goal review "a 5-step demo path") -----
# Five stops for a reader seeing the app for the first time: here, then the
# smart charger's plan, the trading overlay's P&L, what a supplier and one
# customer get, and one replayed week. Grouped as five *rows* rather than six
# links because the brief pairs Supplier's two lenses and Replay with
# Drivers/Household as one stop each; within a row, each link still states
# its own lens's question.
_START_HERE_STOPS: tuple[tuple[tuple[str, str], ...], ...] = (
    (("Smart charging", "1 Plan"),),
    (("Trading", "3 P&L and risk"),),
    (("Supplier", "3 Supplier P&L"), ("Supplier", "4 Firm MW")),
    (("Replay", "Fleet"), ("Drivers", "Household")),
)


def _start_here_link(st: Any, page_name: str, lens_name: str) -> None:
    """One demo-path stop: a button that preselects the lens, then switches page.

    Always a button through ``_switch_to`` (the "Next" row's own callback),
    never ``st.page_link``, even when the wanted lens is the target's default:
    a page link keeps whichever lens the reader last chose on that page, so
    "Smart charging ▸ 1 Plan" could land on 2 Response, and the two widget
    styles made the list look uneven. Each link's own text is the target
    lens's ``question`` (``registry.Lens``), never new prose, so this guide
    cannot claim a question its target does not itself ask.
    """

    page = _registry_page(page_name)
    lens = next(candidate for candidate in page.lenses if candidate.name == lens_name)
    st.button(
        f"**{page.name} ▸ {lens.name}** · {lens.question}",
        key=f"start-here::{page.slug}::{lens.name}",
        on_click=_switch_to,
        args=(page_name, lens_name),
        width="stretch",
    )


def _start_here_guide(st: Any) -> None:
    """The collapsible "Start here" block at the top of Overview ▸ At a glance.

    Open by default: How it works is the app's landing page (Fable landing
    pass), but its own "The forecast" lens points a reader on to this tour,
    so the suggested path should still be the first thing they see here, not
    a title they must know to click. ``expanded`` only sets the initial state
    -- Streamlit tracks a user's own open/closed choice itself from then on,
    so closing it once does not reopen on every later rerun.
    """

    with st.expander("Start here: a five-stop tour for a first look", expanded=True):
        st.caption("You are here, on At a glance. Then, in order:")
        # One full-width button per stop, not columns: inside a column a long
        # label is cut with an ellipsis at 390 px, a full-width one wraps.
        for stops in _START_HERE_STOPS:
            for page_name, lens_name in stops:
                _start_here_link(st, page_name, lens_name)


def _render_next_links(st: Any) -> None:
    columns = st.columns(len(_NEXT_LINKS))
    for column, (label, page_name, lens) in zip(columns, _NEXT_LINKS, strict=True):
        column.button(
            label,
            key=f"overview-next::{page_name}::{lens}",
            on_click=_switch_to,
            args=(page_name, lens),
            width="stretch",
        )


def render_overview(st: Any, result: Any) -> None:
    """Render Overview ▸ At a glance for either model (design 4.1)."""

    # First thing on the landing lens (polish lane "Start here"), above the
    # day-type control: a first-time reader's very first choice is which
    # screen to open next, not the day type.
    _start_here_guide(st)

    # In the page header's controls slot (polish plan G8); one column so the
    # control keeps the layout calls a test's fake ``st`` records.
    (day_column,) = header.controls(st).columns(1)
    with day_column:
        # The shared day-type control (plan E2): one session key across lenses.
        day_type = st.segmented_control(
            "Day",
            options=list(DAY_TYPE_LABELS),
            default="weekday",
            format_func=DAY_TYPE_LABELS.__getitem__,
            required=True,
            key=DAY_TYPE_KEY,
            label_visibility="collapsed",
            persist_state="session",
        )
    day_type = day_type if day_type in DAY_TYPE_LABELS else "weekday"

    connected = _average_day_row(result, metric="connected_share", day_type=day_type)
    soc = _average_day_row(result, metric="battery_soc_percent", day_type=day_type)
    plugged_18_00 = connected.loc[
        connected["local_half_hour"].eq(EIGHTEEN_HUNDRED_LOCAL_HALF_HOUR), "centre"
    ]
    soc_kpi = _plug_in_kpi(result, metric="median_plug_in_soc_percent", day_type=day_type)

    if result.model == "action":
        # Text-valued, so a badge line rather than a 26 px tile (polish plan G5).
        action_value, action_caption = _action_tile(result.action_summary)
        badge_line(
            st,
            "Action chosen",
            f"{action_value} · {action_caption}",
            help="The Axle action this run tests: smart home charging that fills the cheapest "
            "forecast half-hours before each EV's expected departure.",
        )

    # Decision 0007: a fourth tile joins the headline row only when this run
    # carries the Timed tariff path's own saving (``timed_cost_effect_summary``
    # not None); absent, the row stays the three tiles it has always had.
    # ``getattr``: the SYNTHETIC fixture (two-policy, no timed path) has no
    # such field at all, unlike the real model's ``ForecastResult``.
    timed_cost_summary = (
        getattr(result, "timed_cost_effect_summary", None) if result.model == "action" else None
    )
    if result.model == "action":
        headline_tile_count = 4 if timed_cost_summary is not None else 3
    else:
        headline_tile_count = 4
    columns = kpi_columns(st, headline_tile_count)
    kpi(
        columns[0],
        "Plugged in at 18:00",
        _percent_of_fraction(plugged_18_00.iat[0] if len(plugged_18_00) else None),
        # The value follows the day-type toggle, so the context line names it,
        # as the SoC tile beside it does (final critique Q-26).
        context=DAY_TYPE_LABELS[day_type],
        help="Share of the fleet plugged in at home at 18:00 London time, on the unmanaged "
        "path. Median (P50) across simulated weeks.",
    )
    cnz_soc = assumption_value(result, "cnz_median_plug_in_soc_percent")
    # This tile reads this page's own day-type toggle, the same as Plug-ins'
    # same-named tile, and the label states which day type: two tiles with
    # the same name must show the same number for a given day type (goal
    # review items 4, 7). CNZ is the tile's context line, never a delta
    # (no direction arrow on a plain reference figure); why the model's own
    # figure typically reads above CNZ's is explained in help (item 4).
    kpi(
        columns[1],
        # The day type is the context line, not part of the label (polish
        # plan G7: labels at most 28 characters, no qualifiers).
        "Median SoC at plug-in",
        # 1 dp, the same as Plug-ins' same-named tile: two tiles with one name
        # must show the same number (goal review items 4, 7).
        percent(soc_kpi["p50"] if soc_kpi is not None else None, decimals=1),
        context=f"{DAY_TYPE_LABELS[day_type]}"
        + (f" · CNZ observed {percent(cnz_soc)}" if cnz_soc is not None else ""),
        help=(
            "Median state of charge (SoC, how full the battery is) when EVs plug in, on the "
            "unmanaged path, across simulated weeks. CNZ's figure is context, not a "
            "calibration target. The model usually reads above CNZ because most archetypes "
            "plug in every day and charge to an 80% preferred target, so the battery rarely "
            "gets low. That is a modelling choice, not a calibration gap to close."
        ),
    )

    if result.model == "action":
        total_row = result.cost_effect_summary.loc[
            result.cost_effect_summary["component"].eq("total")
        ].iloc[0]
        not_recovered = result.not_recovered_world_count
        flag = " ⚠" if not_recovered else ""
        # One quantity, one sign, one word on every screen (final critique
        # B-9): the weekly saving, positive when smart is cheaper. The model's
        # total is smart minus unmanaged cost, so the saving is its negative;
        # negation swaps the tails (saving P10 = −cost P90), exact for
        # quantiles, the same flip Key stats C and Value and risk use.
        saving_p10 = money(-total_row["p90"])
        saving_p90 = money(-total_row["p10"])
        # Fleet-scale money at 0 dp (G6). Material weeks only (decision 0004
        # item 45); the context line says what the count counts.
        kpi(
            columns[2],
            "Illustrative saving, median",
            money(-total_row["p50"]),
            context=f"{not_recovered} of {result.world_count} weeks materially not recovered{flag}",
            help=(
                "Illustrative weekly saving from smart charging: unmanaged cost minus smart "
                "cost, so a negative number would mean smart cost more. Median across "
                f"simulated weeks; P10 to P90: {saving_p10} to {saving_p90}. Not a bid, a "
                "settlement or an Axle-cash amount."
            ),
        )
        if timed_cost_summary is not None:
            timed_total_row = timed_cost_summary.loc[
                timed_cost_summary["component"].eq("total")
            ].iloc[0]
            timed_saving_p10 = money(-timed_total_row["p90"])
            timed_saving_p90 = money(-timed_total_row["p10"])
            # Beside the Smart tile, never instead of it (decision 0007):
            # Timed tariff has no smart plan of its own, only the same held-off
            # unmanaged rule, so its saving is read from its own cost-effect
            # sibling rather than computed here.
            kpi(
                columns[3],
                "Timed tariff saving, median",
                money(-timed_total_row["p50"]),
                help=(
                    "Illustrative weekly saving from the timed tariff rule: unmanaged cost "
                    "minus timed cost, costed the same way as smart charging (the day-ahead "
                    "price; no tariff rate is applied). Median across simulated weeks; P10 to "
                    f"P90: {timed_saving_p10} to {timed_saving_p90}. Not a bid, a settlement or "
                    "an Axle-cash amount."
                ),
            )
    else:
        plug_ins_kpi = _plug_in_kpi(result, metric="plug_ins_per_ev_per_week", day_type=day_type)
        # The label states the day type: this tile's value follows this
        # page's own toggle (weekday counts differ from weekend counts), so
        # an unchanged-looking label would otherwise hide a changed number.
        kpi(
            columns[2],
            "Plug-ins per EV per week",
            format_quantity(
                plug_ins_kpi["p50"] if plug_ins_kpi is not None else None, "", decimals=1
            ),
            context=DAY_TYPE_LABELS[day_type],
            help="Median (P50) plug-ins per EV in a simulated week, across simulated weeks.",
        )
        kpi(
            columns[3],
            "Unserved travel",
            format_quantity(_weekly_unserved_travel_kwh_p50(result), "kWh", decimals=0),
            help="Trip energy the fleet could not supply in a simulated week, median (P50) "
            "across simulated weeks. It stays at or near zero because an EV that would run "
            "low tops up at a public charger before the trip. It stays in the results as a "
            "check. This run has no smart charging, so no Axle action or commercial outcome "
            "is modelled.",
        )
    # The one glossary line for the counterfactual path (polish plan C3).
    st.caption(UNMANAGED_GLOSSARY)

    figure = _average_day_figure(connected, soc)
    subset = pd.concat([connected, soc], ignore_index=True)
    chart_block(
        st,
        figure,
        title="Average day: % plugged in at home and battery SoC",
        caption=(
            f"{DAY_TYPE_LABELS[day_type]} days. Bars: share plugged in, P50 across weeks. Line "
            "and band: mean SoC, P5–P95 across EVs (how much drivers differ)."
        ),
        frame=subset,
        definition=(
            "One row per London half-hour. The bar is the fleet's share plugged in at home, "
            "median across simulated weeks. The line is the mean battery SoC (state of charge, "
            "how full the battery is) and the band is its 5th to 95th percentile across EVs, "
            "for the chosen day type. That band shows how much drivers differ from each "
            "other, not how uncertain the forecast is."
        ),
        height=CHART_HEIGHTS["time_series_dual_axis"],
        key="overview-average-day",
    )
    # Why some EVs are plugged in at weekday midday (model questions Q-20;
    # decision 0004 item 68): an EV unplugs only on mornings it drives.
    st.caption(
        "Midday connection comes from drivers without a trip that day: they stay plugged in.",
        help=MIDDAY_UNPLUG_NOTE,
    )

    _flexibility_strip(st, result)

    st.caption("Next:")
    _render_next_links(st)
