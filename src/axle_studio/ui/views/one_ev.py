"""Drivers ▸ One EV: what does one driver's week look like?

The brief's sketch 1 (docs/DASHBOARD_DESIGN.md section 1, Q3): plugged-in
shading behind a battery SoC line, with plug-in and departure markers, for
one EV in one simulated week (default the representative week, decision
0004 item 22). Decision 0004 items 10 and 20 resolve the earlier "unavailable
for action results" limitation: the replay now runs the vectorised kernel on
a one-EV slice with its own smart-charging inputs and public charging, so
this view shows both paths for action results (chart audit B2, B3).

The only model call this view makes is the replay entry point, injected as
``replay_one_ev``/``replay_one_ev_bands`` (contract v2 section 5) so a later
integration step can point it at the real ``model/individual.py`` functions
without this file importing model or test-fixture code directly. Reads
``result.units`` to build the EV picker, ``result.world_count`` and
``result.representative_world_id`` for the week picker, and
``result.assumptions`` for the public top-up threshold/target caption; the
picked EV and world are then handed to the replay functions above, not read
off ``result`` directly.

The internal path id stays ``"selected"`` (contract v2, ``path_id``); every
user-facing legend, label and table cell says "Smart" instead (goal review
item 6, the same convention ``action_response.py``/``action_decision.py``
already use).

Decision 0007 adds an optional third path, ``"timed"`` ("Timed tariff"):
home charging barred daily from 12:00 London until an editable start hour.
This view never hardcodes a path count: every trace and table loop reads
``replay.intervals["path_id"]``'s actual values (``_present_paths``), so the
third path, its band and its own home-charging bar series appear the moment
the one-EV replay reports it and nothing changes while it does not.
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
from ..style import (
    CHART_HEIGHTS,
    PATH_DISPLAY_ORDER,
    PATH_LABELS,
    PATH_STYLES,
    PLUGGED_ALPHA,
    SERIES_COLOURS,
    UNAVAILABLE,
    assumption_value,
    band_fill,
    hover_time_labels,
    london_time_axis,
)

_CHARGING_BAR_ALPHA = 0.55
"""Bar opacity for the home-charging kWh traces (secondary axis): solid
enough to read against the grid, translucent enough that overlaid normal and
selected bars in the same half-hour do not hide each other."""

_NORMAL_UNDER_SELECTED_WIDTH = 4.0
"""O7: at style.PATH_STYLES' shared widths (normal 1.5 px, selected 2.5 px),
normal sat entirely under selected wherever the two paths coincide -- most of
the week outside the action window -- so it read as a single teal line with
no normal path at all. This one_ev-only override widens normal, just when
both paths are drawn, so a sliver of it always shows beneath selected. It
does not touch style.PATH_STYLES itself (shared by every other view) or
either path's colour or dash (decision 0004 item 27)."""

ReplayOneEv = Callable[[Any, str, int], Any]
ReplayOneEvBands = Callable[[Any, str], pd.DataFrame]

_BAR_PANEL_GAP_PX = 40
"""Goal review item 17: a secondary axis overlaid on the SoC panel would add
two legend entries (one bar series per path) on top of the panel's own five
(shading, two SoC lines, two bands) -- seven on a two-path result, over the
"at most five" limit. The home-charging bars instead live in their own panel
below (shared x-axis, ``make_subplots``), out of the legend entirely
(``showlegend=False``, like the markers); the panel's own title/axis names
them instead. This is the gap between the two panels; each panel's own
height still comes from ``style.CHART_HEIGHTS`` (time_series_dual_axis for
the SoC panel, small_multiple_panel for the bars, chosen only because it is
an existing "small secondary panel" height, not because this is a small
multiple)."""


def _traits_caption(traits: dict[str, object]) -> str:
    miles_per_year = float(traits["daily_miles_mean"]) * 365.0
    return (
        f"{traits['cohort_label']} · {miles_per_year:,.0f} miles/yr · "
        f"{float(traits['physical_capacity_kwh']):.0f} kWh battery · "
        f"{float(traits['home_charger_limit_kw']):.0f} kW charger · "
        f"target SoC {float(traits['preferred_target_soc_percent']):.0f}%"
    )


DISPATCH_LOCK_HELP = (
    "With intraday dispatch on, a fixed share of EVs keep the plan they made at plug-in. The "
    "rest re-plan each hour on the latest intraday price. Set under Edit assumptions ▸ Trading."
)
"""Shared tooltip for "Locked to its day-ahead plan" / "Re-plans hourly...", wherever it
appears (One EV, Household's status badge, Replay ▸ Customer): clarity critique cross-cutting
note -- none of the three explained "locked" on its own."""


def _dispatch_caption(traits: dict[str, object]) -> str | None:
    """This EV's intraday dispatch state, worded as Household and Replay > Customer say it.

    ``traits["dispatch_locked"]`` is ``units.dispatch_locked`` (bool on every
    result, intraday-dispatch-v1 §2); the caller only shows this when the
    run's dispatch switch was on (``result.dispatch_world_slot is not None``)
    and this EV is not a hold-out control EV, since the column is False both
    for a free EV and for every EV on a switch-off run, and a control EV
    keeps a locked/free status but ignores every plan and charges by the
    normal rule (§2, §10.1e), so it is never "locked" or "re-planned" either.
    """

    if bool(traits.get("control_group", False)):
        return None
    locked = traits.get("dispatch_locked")
    if locked is None:
        return None
    if locked:
        return "Locked to its day-ahead plan"
    return "Re-plans hourly on the latest intraday price"


def _present_paths(intervals: pd.DataFrame) -> list[str]:
    """Paths this replay actually carries, in on-screen order.

    Decision 0007: frames keep the kernel's own normal/selected/timed order,
    but the screen reads Unmanaged, Timed tariff, Smart everywhere
    (``style.PATH_DISPLAY_ORDER`` is the one place that reorders them for
    display); every trace loop below draws in that order too. A no-action
    replay carries only "normal"; an action replay without the timed-tariff
    setting carries "normal" and "selected" exactly as before -- this reads
    off the replay's own ``path_id`` values rather than assuming a fixed
    count, so a third path shows up the moment the model reports one and
    nothing changes while it does not.
    """

    present = set(intervals["path_id"])
    return [path for path in PATH_DISPLAY_ORDER if path in present]


def _shading_trace(normal_intervals: pd.DataFrame) -> go.Scatter:
    connected = normal_intervals["connected"].to_numpy()
    return go.Scatter(
        x=normal_intervals["interval_start_utc"],
        y=np.where(connected, 100.0, 0.0),
        mode="lines",
        line={"width": 0},
        line_shape="hv",  # each point is a half-hour value; steps avoid a diagonal blend
        fill="tozeroy",
        fillcolor=band_fill(SERIES_COLOURS["plugged"], PLUGGED_ALPHA),
        name="Plugged in at home",
        hoverinfo="skip",
    )


def _soc_trace(
    rows: pd.DataFrame, *, path: str, label: str, width: float | None = None
) -> go.Scatter:
    style = PATH_STYLES[path]
    hover = hover_time_labels(rows["interval_start_utc"])
    return go.Scatter(
        x=rows["interval_start_utc"],
        y=rows["battery_soc_percent"],
        mode="lines",
        line={
            "color": style["color"],
            "dash": style["dash"],
            "width": style["width"] if width is None else width,
            # Decision 0007: style.PATH_STYLES["timed"]'s "hv" step shape --
            # charging is barred or allowed for a whole half-hour at a time,
            # so the timed line should jump at each boundary, not slope
            # through it; .get() leaves normal/selected at Plotly's default
            # "linear", the only two PATH_STYLES entries without this key.
            "shape": style.get("line_shape"),
        },
        name=label,
        legendgroup=path,
        customdata=hover,
        hovertemplate="%{customdata}<br>SoC: %{y:.0f}%<extra></extra>",
    )


def _topup_marker_trace(rows: pd.DataFrame, *, path: str) -> go.Scatter | None:
    # Decision 0004 item 32: EVs never strand -- a trip that would take SoC
    # below the top-up threshold tops up at an always-available public
    # charger to the top-up target first. Both are illustrative, editable
    # assumptions (public_top_up_threshold_soc_percent,
    # public_top_up_target_soc_percent) whose defaults can change independently
    # of this view, so the caption in render_one_ev reads the run's actual
    # values from result.assumptions rather than hard-coding a number (chart
    # audit O5). There is no separate "top-up event" field; a top-up is any
    # slot with public import in the replay. Not keyed on ``location``: a
    # top-up takes no modelled time, so the replay labels its slot by where
    # the EV spent most of the half-hour (usually "driving"), never
    # "public_charging".
    topped_up = rows.loc[rows["public_import_kwh"] > 0]
    if topped_up.empty:
        return None
    style = PATH_STYLES[path]
    return go.Scatter(
        x=topped_up["interval_start_utc"],
        y=topped_up["battery_soc_percent"],
        mode="markers",
        marker={"symbol": "diamond", "size": 9, "color": style["color"]},
        name="Public top-up",
        # B4: markers stay out of the legend entirely -- the caption's own
        # "▲ plug-in, ▼ departure, ◆ public top-up" key already names them,
        # so a legend entry (doubled per path on an action result) only
        # crowded the legend at 390 px.
        showlegend=False,
        customdata=hover_time_labels(topped_up["interval_start_utc"]),
        hovertemplate="%{customdata}<br>Public top-up<extra></extra>",
    )


def _shortfall_marker_trace(rows: pd.DataFrame) -> go.Scatter | None:
    """Mark half-hours where this EV left before its smart plan finished.

    Decision 0004 item 43: "One EV marks... 'left before plan finished'
    events." ``early_departure_shortfall_kwh`` (contract v2 section 5) is
    always 0 on the normal path -- only the smart plan can be interrupted --
    so this only ever draws on the selected path, but the check stays
    generic rather than hard-coding that (the model owns which paths carry
    the column, not this view).
    """

    left_early = rows.loc[rows["early_departure_shortfall_kwh"] > 1e-9]
    if left_early.empty:
        return None
    # A plain list of (label, kwh) pairs, not np.column_stack: stacking a
    # string column with a float column into one array upcasts the floats
    # to strings, which breaks the hovertemplate's "%{customdata[1]:.1f}"
    # numeric formatting.
    customdata = list(
        zip(
            hover_time_labels(left_early["interval_start_utc"]),
            left_early["early_departure_shortfall_kwh"].tolist(),
            strict=True,
        )
    )
    return go.Scatter(
        x=left_early["interval_start_utc"],
        y=left_early["battery_soc_percent"],
        mode="markers",
        marker={"symbol": "x", "size": 10, "color": SERIES_COLOURS["flag"]},
        name="Left before plan finished",
        # B4: see _topup_marker_trace -- the caption key names this marker.
        showlegend=False,
        customdata=customdata,
        hovertemplate=(
            "%{customdata[0]}<br>Left before plan finished, short by "
            "%{customdata[1]:.1f} kWh<extra></extra>"
        ),
    )


def _charging_bar_trace(rows: pd.DataFrame, *, path: str) -> go.Bar:
    """Home charging kWh each half-hour, in the panel below the SoC chart (item 43).

    Drawn only when both paths are present (action results): the point is
    to show which half-hours smart charging used relative to the normal
    (charge-as-soon-as-plugged-in) path, so a no-action result -- one path,
    no smart plan -- gets no bars at all (see ``_figure``). Goal review item
    17: kept out of the legend (``showlegend=False``), like the markers --
    the panel's own y-axis title and the caption's "unmanaged vs smart" wording
    name it instead, so it adds no legend entry on top of the SoC panel's
    five.
    """

    style = PATH_STYLES[path]
    hover = hover_time_labels(rows["interval_start_utc"])
    return go.Bar(
        x=rows["interval_start_utc"],
        y=rows["home_import_kwh"],
        name=f"Home charging ({PATH_LABELS[path].lower()})",
        marker={"color": band_fill(style["color"], _CHARGING_BAR_ALPHA)},
        showlegend=False,
        customdata=hover,
        hovertemplate="%{customdata}<br>Home charging: %{y:.1f} kWh<extra></extra>",
    )


def _path_name_list(paths: list[str]) -> str:
    """Join present paths' display names for a caption, e.g. "unmanaged vs smart".

    Two paths join with "vs" (today's exact wording, decision 0004 item 43);
    three or more read as a plain list ("a, b and c"), since "vs" implies a
    pairwise contrast the timed-tariff addition (decision 0007) no longer is.
    """

    names = [PATH_LABELS[path].lower() for path in paths]
    if len(names) == 2:
        return " vs ".join(names)
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _charging_note(paths: list[str]) -> str:
    """Caption clause for the lower home-charging bar panel, naming only the paths drawn there.

    Decision 0007: the timed-tariff bars show a hard clock-time rule, not a
    plan, so they earn their own sentence rather than being folded into the
    "smart bars ... plan" wording, which is specific to the smart path. With
    exactly normal and selected present this reproduces item 43's original
    wording unchanged (contract rule: absent data, unchanged screen).
    """

    note = f" Lower panel: this EV's home charging each half-hour, {_path_name_list(paths)}."
    if "selected" in paths:
        note += (
            " The smart bars show where the plan put its charging: the cheapest "
            "forecast half-hours."
        )
    if "timed" in paths:
        note += (
            " The timed tariff bars show the clock-time rule: no home charging "
            "from 12:00 until the set start hour."
        )
    return note


def _plug_in_marker_trace(plug_events: pd.DataFrame) -> go.Scatter | None:
    if plug_events.empty:
        return None
    return go.Scatter(
        x=plug_events["plug_in_utc"],
        y=plug_events["plug_in_soc_percent"],
        mode="markers",
        marker={"symbol": "triangle-up", "size": 10, "color": SERIES_COLOURS["observed"]},
        name="Plug-in",
        # B4: see _topup_marker_trace -- the caption key names this marker.
        showlegend=False,
        customdata=hover_time_labels(plug_events["plug_in_utc"]),
        hovertemplate="%{customdata}<br>Plug-in SoC: %{y:.0f}%<extra></extra>",
    )


def _departure_marker_trace(daily_audit: pd.DataFrame, *, path: str) -> go.Scatter | None:
    rows = daily_audit.loc[daily_audit["drives_today"]].dropna(subset=["departure_utc"])
    if rows.empty:
        return None
    style = PATH_STYLES[path]
    return go.Scatter(
        x=rows["departure_utc"],
        y=rows["departure_soc_percent"],
        mode="markers",
        marker={"symbol": "triangle-down", "size": 10, "color": style["color"]},
        name="Departure",
        # B4: see _topup_marker_trace -- the caption key names this marker.
        showlegend=False,
        customdata=hover_time_labels(rows["departure_utc"]),
        hovertemplate="%{customdata}<br>Departure SoC: %{y:.0f}%<extra></extra>",
    )


def _band_label(rows: pd.DataFrame, *, path: str, multi_path: bool) -> str:
    """Legend name for the P10-P90 band (B3 names the spread with N).

    A two-path (action) result draws the band twice: the full "P10-P90
    across N simulated weeks (normal/selected)" name would double the legend
    to 5 entries and squeeze the plot below 260 px at 390 px wide, so the
    legend gets a short per-path name instead; render_one_ev's caption still
    carries the full "across N simulated weeks" wording for both paths, so
    the spread stays named per decision 0004 item 22 even though the legend
    itself does not repeat it.
    """

    if multi_path:
        return f"{PATH_LABELS[path]} P10–P90"
    world_count = int(rows["world_count"].iat[0]) if len(rows) else 0
    return f"P10–P90 across {world_count} simulated weeks"


def _band_traces(bands: pd.DataFrame, *, path: str, multi_path: bool) -> list[go.Scatter]:
    """The across-weeks SoC band for one path, grouped with that path's SoC line.

    Not ``style.band_and_line``: the line here is this EV's SoC in one chosen
    simulated week, not the band's median, so they come from different
    frames. The band still joins the line's ``legendgroup`` with
    ``showlegend=False`` (polish plan G3): one legend entry per path, and
    clicking it hides line and band together; the caption names the spread.
    """

    rows = bands.loc[
        bands["path_id"].eq(path) & bands["metric"].eq("battery_soc_percent")
    ].sort_values("slot_index")
    style = PATH_STYLES[path]
    label = _band_label(rows, path=path, multi_path=multi_path)
    # Same step shape as the path's own SoC line (_soc_trace): a linear band
    # edge would not align with a stepped centre line at each half-hour
    # boundary (decision 0007; style.PATH_STYLES["timed"]'s "hv").
    shape = style.get("line_shape")
    return [
        go.Scatter(
            x=rows["interval_start_utc"],
            y=rows["p10"],
            mode="lines",
            line={"width": 0, "color": style["color"], "shape": shape},
            legendgroup=path,
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=rows["interval_start_utc"],
            y=rows["p90"],
            mode="lines",
            line={"width": 0, "color": style["color"], "shape": shape},
            fill="tonexty",
            fillcolor=band_fill(style["color"]),
            name=label,
            legendgroup=path,
            showlegend=False,
            hoverinfo="skip",
        ),
    ]


def _figure(replay: Any, *, bands: pd.DataFrame) -> go.Figure:
    paths = _present_paths(replay.intervals)
    multi_path = len(paths) > 1
    normal_rows = replay.intervals.loc[replay.intervals["path_id"].eq("normal")].sort_values(
        "slot_index"
    )

    if multi_path:
        # Goal review item 17: the home-charging bars get their own panel
        # below the SoC panel (shared x-axis), not a secondary axis overlaid
        # on it, so they add no legend entry of their own -- see
        # _charging_bar_trace and _BAR_PANEL_GAP_PX.
        bar_panel_height = CHART_HEIGHTS["small_multiple_panel"]
        soc_panel_height = CHART_HEIGHTS["time_series_dual_axis"]
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[soc_panel_height, bar_panel_height],
            vertical_spacing=_BAR_PANEL_GAP_PX / (soc_panel_height + bar_panel_height),
        )
    else:
        figure = go.Figure()

    def add(trace: go.Scatter | go.Bar, *, row: int) -> None:
        if multi_path:
            figure.add_trace(trace, row=row, col=1)
        else:
            figure.add_trace(trace)

    add(_shading_trace(normal_rows), row=1)
    # B4: an action result could add up to ten legend entries -- shading,
    # two SoC lines, two bands, two top-up markers, two departure markers,
    # plus plug-in -- which wrapped over roughly half the chart's height at
    # 390 px. Markers no longer take a legend entry at all (see
    # _topup_marker_trace) and the bars now live in their own panel
    # (_charging_bar_trace), so the SoC panel's legend never carries more
    # than five entries (shading, two SoC lines, two bands).
    for path in paths:
        rows = replay.intervals.loc[replay.intervals["path_id"].eq(path)].sort_values("slot_index")
        soc_label = f"SoC ({PATH_LABELS[path].lower()})" if multi_path else "SoC"
        # Item 40: the across-weeks band is always shown now (no checkbox);
        # One EV never gets an across-EVs band (it is one car).
        for band_trace in _band_traces(bands, path=path, multi_path=multi_path):
            add(band_trace, row=1)
        # O7: normal (1.5 px) sat entirely under selected (2.5 px) wherever
        # the two paths coincide -- most of the week outside the action
        # window -- reading as a single teal line with no normal path at all.
        # Widening normal only here, and only when both paths are drawn,
        # leaves a visible sliver beneath selected (colours/dash unchanged).
        soc_width = _NORMAL_UNDER_SELECTED_WIDTH if multi_path and path == "normal" else None
        add(_soc_trace(rows, path=path, label=soc_label, width=soc_width), row=1)
        topup = _topup_marker_trace(rows, path=path)
        if topup is not None:
            add(topup, row=1)
        departure = _departure_marker_trace(
            replay.daily_audit.loc[replay.daily_audit["path_id"].eq(path)], path=path
        )
        if departure is not None:
            add(departure, row=1)
        shortfall = _shortfall_marker_trace(rows)
        if shortfall is not None:
            add(shortfall, row=1)
        # Item 43: mark the half-hours smart charging used for this EV,
        # against the normal (charge-as-soon-as-plugged-in) path. Only an
        # action result has a "smart" path to compare against, so a
        # no-action (one-path) result draws no bars at all.
        if multi_path:
            add(_charging_bar_trace(rows, path=path), row=2)

    # Plug-in timing is a physical event unaffected by the Axle action, so it
    # is drawn once (from the normal path) rather than doubled on action
    # results.
    plug_in_events = replay.plug_events.loc[replay.plug_events["path_id"].eq("normal")]
    plug_in_marker = _plug_in_marker_trace(plug_in_events)
    if plug_in_marker is not None:
        add(plug_in_marker, row=1)

    x_axis_kwargs: dict[str, Any] = {
        "tickmode": "array",
        # B6: marker-only traces (plug-in, departure, top-up) pad Plotly's
        # autorange to leave room for the marker symbols at the data extent,
        # which pushed the day tick labels together at 390 px. An explicit
        # range pinned to the study span (the normal path's own slots) stops
        # markers from widening it.
        "range": [normal_rows["interval_start_utc"].min(), normal_rows["interval_end_utc"].max()],
        **london_time_axis(normal_rows["interval_start_utc"]),
    }
    if multi_path:
        # Shared x-axis (make_subplots): setting range/ticks on the bottom
        # row's axis is enough -- it "matches" the top row's, the same
        # pattern archetypes.py's small multiples use.
        figure.update_xaxes(**x_axis_kwargs, row=2, col=1)
        figure.update_yaxes(title="%", range=[0, 100], row=1, col=1)
        figure.update_yaxes(title="Home charging (kWh)", rangemode="tozero", row=2, col=1)
        figure.update_layout(barmode="overlay")
    else:
        figure.update_xaxes(**x_axis_kwargs)
        figure.update_yaxes(title="%", range=[0, 100])
    return figure


def _intervals_table(intervals: pd.DataFrame) -> pd.DataFrame:
    rows = intervals.sort_values(["path_id", "slot_index"]).copy()
    # Goal review item 6: the table's "Path" column is user-facing (the
    # "Data and definition" expander), so it reads "Smart" too, not the
    # model's own internal "selected" (``style.PATH_LABELS``).
    rows["path_id"] = rows["path_id"].map(PATH_LABELS)
    return rows.rename(
        columns={
            "path_id": "Path",
            "interval_start_london": "Interval start (London)",
            "interval_end_utc": "Interval end (UTC)",
            "location": "Location",
            "connected": "Plugged in at home",
            "battery_soc_percent": "Battery SoC (%)",
            "home_import_kwh": "Home import (kWh)",
            "public_import_kwh": "Public import (kWh)",
            "unserved_travel_kwh": "Unserved travel (kWh)",
            "early_departure_shortfall_kwh": "Left before plan finished, short by (kWh)",
        }
    ).loc[
        :,
        [
            "Path",
            "Interval start (London)",
            "Interval end (UTC)",
            "Location",
            "Plugged in at home",
            "Battery SoC (%)",
            "Home import (kWh)",
            "Public import (kWh)",
            "Unserved travel (kWh)",
            "Left before plan finished, short by (kWh)",
        ],
    ]


def _plug_events_table(events: pd.DataFrame) -> pd.DataFrame:
    # Display-only conversion of the stored UTC column, the same way
    # ``plug_in_utc`` is already shown in London time beside it (clarity
    # critique: plug-in and plug-out in different timezones on one row read
    # as a bug). ``plug_out_utc`` is NaT for a still-open session; tz_convert
    # leaves NaT as NaT.
    events = events.assign(
        plug_out_london=lambda frame: frame["plug_out_utc"].dt.tz_convert("Europe/London")
    )
    return events.rename(
        columns={
            "plug_in_london": "Plug-in (London)",
            "plug_in_soc_percent": "SoC at plug-in (%)",
            "plug_out_london": "Plug-out (London)",
            "plug_out_soc_percent": "SoC at plug-out (%)",
            "home_import_kwh": "Home import (kWh)",
            "battery_added_kwh": "Battery added (kWh)",
            "still_plugged_at_horizon_end": "Still plugged at end",
        }
    ).loc[
        :,
        [
            "Plug-in (London)",
            "SoC at plug-in (%)",
            "Plug-out (London)",
            "SoC at plug-out (%)",
            "Home import (kWh)",
            "Battery added (kWh)",
            "Still plugged at end",
        ],
    ]


def _topup_row(path: str, ordered: pd.DataFrame, start: int, end: int) -> dict[str, object]:
    run = ordered.iloc[start : end + 1]
    return {
        "Path": PATH_LABELS[path],  # goal review item 6: "Smart", not "selected"
        "Start (London)": run["interval_start_london"].iat[0],
        "End (UTC)": run["interval_end_utc"].iat[-1],
        "Energy added (kWh)": float(run["public_import_kwh"].sum()),
    }


def _public_topup_sessions(intervals: pd.DataFrame) -> pd.DataFrame:
    """Group the replay's own per-slot public-charging rows into sessions.

    This sums values the replay already computed per half-hour; it does not
    calculate energy, physics or policy (decision 0004 item 32).
    """

    sessions: list[dict[str, object]] = []
    for path, group in intervals.groupby("path_id", sort=False):
        ordered = group.sort_values("slot_index").reset_index(drop=True)
        # Public import alone marks a top-up slot (see _topup_marker_trace).
        is_topup = (ordered["public_import_kwh"] > 0).tolist()
        start: int | None = None
        for index, flag in enumerate(is_topup):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                sessions.append(_topup_row(path, ordered, start, index - 1))
                start = None
        if start is not None:
            sessions.append(_topup_row(path, ordered, start, len(ordered) - 1))
    return pd.DataFrame(sessions)


def _daily_audit_table(daily_audit: pd.DataFrame) -> pd.DataFrame:
    rows = daily_audit.sort_values(["path_id", "local_date"]).copy()
    # Departing below the preferred *target* SoC does not by itself raise the
    # warning icon: the target is a preference, not the physical ceiling
    # (AGENTS.md), and item 32's public top-ups mean a real shortfall is
    # rare. The icon is reserved for shortfall_kwh > 0 (this date's trip
    # energy actually not served); "below target" with no shortfall is a
    # plain, unflagged note, never a false alarm (goal review item 11).
    has_shortfall = rows["shortfall_kwh"] > 1e-9
    below_target = rows["departed_below_target"]
    status = np.select(
        [has_shortfall, below_target], ["⚠ shortfall", "Below target"], default="On target"
    )
    rows["Status"] = np.where(rows["drives_today"], status, "No trip")
    # Goal review item 6: the table's "Path" column is user-facing (the
    # "Daily readiness audit" expander), so it reads "Smart" too, not the
    # model's own internal "selected" (``style.PATH_LABELS``).
    rows["path_id"] = rows["path_id"].map(PATH_LABELS)
    return rows.rename(
        columns={
            "path_id": "Path",
            "day_label": "Day",
            "drives_today": "Drives today",
            "departure_london": "Departure (London)",
            "target_soc_percent": "Target SoC (%)",
            "departure_soc_percent": "Departure SoC (%)",
            "trip_energy_need_kwh": "Trip energy need (kWh)",
            "shortfall_kwh": "Shortfall (kWh)",
        }
    ).loc[
        :,
        [
            "Path",
            "Day",
            "Drives today",
            "Departure (London)",
            "Target SoC (%)",
            "Departure SoC (%)",
            "Trip energy need (kWh)",
            "Shortfall (kWh)",
            "Status",
        ],
    ]


def ev_picker(st: Any, units: pd.DataFrame) -> str:
    """Draw the shared EV selectbox into ``st`` (a page, column or slot) and return a ``unit_id``.

    One helper for Drivers ▸ One EV, Drivers ▸ Household (household contract
    v1 §6.1, O8) and Partners ▸ Fleet and leasing: widgets that share the key
    "one-ev-unit" must have an identical label, options and default, or
    Streamlit treats them as different widgets and resets the choice on a
    lens switch. Building all three from here keeps them identical, so the
    chosen driver or vehicle survives the switch between any of them.
    """

    unit_ids = list(units["unit_id"])
    labels = {row.unit_id: f"{row.unit_id} · {row.cohort_label}" for row in units.itertuples()}
    # Goal review item 11: default to a typical commuter, not whichever EV
    # happens to sort first. Of the two options offered ("first Average (UK)
    # EV, whose cohort would plug in most evenings" or "the EV nearest the
    # median plug-ins per week"), this takes the first: `units` (contract v2
    # 3.2) has no per-EV plug-in-frequency column, only cohort-level figures
    # (`cohort_summary`), so "nearest the median" would need a replay call
    # per candidate EV just to pick a default -- against this view's own
    # "only model call is the replay entry point" design. "Average (UK)"
    # needs no such call: decision 0004 item 42 gives its cohort the top
    # plug probability, 1.0, so the first Average (UK) `unit_id` is the
    # closest cohort-level match to "plugs in most evenings" without
    # inspecting its actual plug events. That 1.0 is a would-plug-in rate,
    # not a nightly guarantee: every cohort except Always plugged-in still
    # goes through the shared skip share that night (decision 0004 item 60;
    # sampling.plug_in_skip_share, sampling.sample_connection_opportunities'
    # `clocked` EVs), so an Average (UK) EV still misses some nights -- the
    # median share skipped on an ordinary night is
    # assumptions.SHARED_FACTORS["behaviour.plug_in_skip_median"]'s value.
    # Only Always plugged-in is exempt (sampling._ALWAYS_PLUGGED_COHORT):
    # its one session covers the whole run, with no plug-in clock or skip
    # draw. Falls back to the first EV if no Average (UK) EV was sampled
    # (chart audit B1: a small run can miss a cohort).
    average_uk_ids = list(units.loc[units["cohort_id"].eq("average_uk"), "unit_id"])
    default_unit = average_uk_ids[0] if average_uk_ids else unit_ids[0]
    return st.selectbox(
        "EV",
        options=unit_ids,
        index=unit_ids.index(default_unit),
        format_func=lambda uid: labels.get(uid, uid),
        key="one-ev-unit",
        label_visibility="collapsed",
    )


def render_one_ev(
    st: Any,
    result: Any,
    *,
    replay_one_ev: ReplayOneEv,
    replay_one_ev_bands: ReplayOneEvBands,
) -> None:
    """Render Drivers ▸ One EV: one driver's simulated week (sketch 1, Q3)."""

    units = result.units
    # EV and week pickers sit in the page header's controls slot (polish
    # plan G8), side by side, labels collapsed: each option names itself.
    ev_column, week_column = header.controls(st).columns([3, 2])
    selected_unit = ev_picker(ev_column, units)
    representative = result.representative_world_id
    world_id = week_column.selectbox(
        "Simulated week",
        options=list(range(result.world_count)),
        index=representative,
        format_func=lambda world: (
            f"Week {world} (median)" if world == representative else f"Week {world}"
        ),
        key="one-ev-world",
        label_visibility="collapsed",
    )

    traits = units.loc[units["unit_id"].eq(selected_unit)].iloc[0].to_dict()
    st.caption(_traits_caption(traits))
    if getattr(result, "dispatch_world_slot", None) is not None:
        dispatch_text = _dispatch_caption(traits)
        if dispatch_text is not None:
            st.caption(dispatch_text, help=DISPATCH_LOCK_HELP)

    # Contract v2 section 1 (decision 0004 item 38): smart charging applies
    # to every EV on an action result, so `action_status` is always
    # "selected" there; this view has no "no action selected" state to
    # special-case.

    replay = replay_one_ev(result, selected_unit, world_id)
    # Item 40: the across-weeks P10-P90 band is always shown now (one car has
    # no across-EVs spread to add, so there is no second band to gate behind
    # a toggle either) -- no checkbox.
    bands = replay_one_ev_bands(result, selected_unit)

    week_note = "Median week" if replay.is_representative_world else f"Week {replay.world_id}"
    # B3: name the band's spread in the caption too, not only its legend
    # entry (_band_label).
    band_world_count = int(bands["world_count"].iat[0]) if not bands.empty else 0
    paths = _present_paths(replay.intervals)
    multi_path = len(paths) > 1
    chart_height = CHART_HEIGHTS["time_series_dual_axis"]
    charging_note = ""
    if multi_path:
        # Goal review item 17: the bars get their own panel below (see
        # _figure), so the total height is just the sum of the two panels'
        # own CHART_HEIGHTS entries.
        chart_height += CHART_HEIGHTS["small_multiple_panel"]
        charging_note = _charging_note(paths)
    # Decision 0007: this sentence is informational background (what the two
    # possible extra paths are), not a readout of which paths are on screen
    # today, so it stays exactly as item 38 wrote it when timed is absent
    # and only gains a clause once the run actually carries it.
    timed_definition = (
        " Setting a timed tariff start time adds a third path, held off daily from 12:00 "
        "London until that time."
        if "timed" in paths
        else ""
    )
    chart_block(
        st,
        _figure(replay, bands=bands),
        title="Plugged in, battery SoC and home charging",
        caption=(
            f"{week_note}, not an average. Band: SoC P10–P90 across {band_world_count} "
            "simulated weeks. ▲ plug-in ▼ departure ◆ top-up ✕ left early. Illustrative."
        ),
        frame=_intervals_table(replay.intervals),
        definition=(
            "This EV's battery SoC (state of charge, how full the battery is), plugged-in "
            "status, home charging and location for each half-hour of the chosen simulated "
            "week. The unmanaged path is always shown; a smart charging run adds the smart "
            "path. Shaded: plugged in at home. ✕ marks a session that left before its smart "
            f"plan finished.{charging_note}{timed_definition}"
        ),
        height=chart_height,
        key="one-ev-chart",
    )

    normal_events = replay.plug_events.loc[replay.plug_events["path_id"].eq("normal")].sort_values(
        "plug_in_utc"
    )
    topups = _public_topup_sessions(replay.intervals)
    # One expander with tabs (polish plan G9) instead of three stacked
    # expanders: the events, top-ups and daily audit are one "detail" read.
    with st.expander("Plug-ins, top-ups and daily readiness"):
        events_tab, topups_tab, audit_tab = st.tabs(
            [
                f"Plug-in events ({len(normal_events)})",
                f"Public top-ups ({len(topups)})",
                "Daily readiness",
            ]
        )
        events_tab.dataframe(_plug_events_table(normal_events), width="stretch", hide_index=True)
        if topups.empty:
            topups_tab.caption("No public top-ups in this simulated week.")
        else:
            # O5: read the run's own top-up threshold/target rather than
            # repeating decision 0004 item 32's illustrative defaults, which
            # an editable-assumptions run may have changed.
            threshold = assumption_value(result, "public_top_up_threshold_soc_percent")
            target = assumption_value(result, "public_top_up_target_soc_percent")
            threshold_text = f"{threshold:.0f}%" if threshold is not None else UNAVAILABLE
            target_text = f"{target:.0f}%" if target is not None else UNAVAILABLE
            topups_tab.caption(
                f"Public top-ups: a trip that would take SoC below {threshold_text} tops up at "
                f"an always-available public charger to {target_text} first."
            )
            topups_tab.dataframe(topups, width="stretch", hide_index=True)
        audit_tab.dataframe(
            _daily_audit_table(replay.daily_audit), width="stretch", hide_index=True
        )
