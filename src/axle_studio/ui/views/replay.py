"""Replay ▸ Fleet and Replay ▸ Customer: play one simulated week back, hour by hour.

What this module owns (replay contract v1 §3, decision 0004 item 66): the
two lenses of the Replay page. Each draws one Plotly figure with 169
animation frames, one per hourly decision instant ``tau_k`` of the week plus
a closing frame. A play button and a slider step through the frames in the
browser; at each frame the chart shows what had happened by ``tau_k`` (solid,
realised) and what was known about the rest of the week at ``tau_k``
(dashed, the forward curves), with the running money above.

Everything drawn comes from the result: ``result.replay_week`` (what was known
at each instant, the running money; ``model/replay.py``), the realised frames
the contract names, and for the Customer lens the injected one-EV replay and
timeline functions. This view only masks arrays by ``k`` and converts kWh
per half-hour to kW (divide by 0.5 h). It applies no cut-off rule, computes
no money and never runs the model; no control starts a run.

Why frames and not a Streamlit slider: a Streamlit widget reruns the script
on every change, so dragging would stall; Plotly frames are stepped in the
browser. Every animated element is therefore a trace (the now line, the
counters, the markers), so frames use ``redraw: False`` and dragging stays
smooth (contract §3.2). The future is hidden (``None``), not dimmed: a dimmed
line would still show its future values in the unified hover (§3.2, lead
ruling 6).

Every price, volume and pound is synthetic or illustrative; the trading
figure is "illustrative simulated trading P&L", never Axle cash.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..components import header
from ..components.chart_table import chart_block
from ..components.kpi import kpi, kpi_columns
from ..style import (
    _LEGEND_CHAR_PX,
    _LEGEND_GAP_PX,
    _LEGEND_ROW_PX,
    _LEGEND_SYMBOL_PX,
    _NARROW_FIGURE_WIDTH_PX,
    _TICK_ROW_PX,
    BORDER,
    CHART_HEIGHTS,
    INK,
    MUTED_INK,
    PATH_LABELS,
    PATH_STYLES,
    PLUGGED_ALPHA,
    SERIES_COLOURS,
    TRANSPARENT,
    _legend_names,
    band_fill,
    hover_time_labels,
    london_time_axis,
    money,
    present_paths,
)
from .one_ev import DISPATCH_LOCK_HELP

PLAY_MS = 150
"""Milliseconds per frame when playing (lead ruling 3): the 169 frames play in about 25 s."""

UNAVAILABLE_MESSAGE = "Replay is not available for this run."

WEEK_KEY = "replay-week"
"""One session key for the Simulated week control, shared by both lenses (§3.1)."""
JUMP_KEY = "replay-jump"
EV_KEY = "replay-ev"

ReplayOneEv = Callable[[Any, str, int], Any]
ReplayOneEvTimeline = Callable[[Any, str, int], Any]

_SLOT_HOURS = 0.5
"""Study slots are half-hours: kW = kWh per slot / 0.5 h."""

# Figure geometry in px. Panels come from style.CHART_HEIGHTS; these are the
# pieces around them, measured in the browser at 1440 and 390 px.
_TOP_PX = 40  # the Play/Pause buttons sit in the top margin
_GAP_PX = 28  # between panels
_SLIDER_PX = 64  # the slider's "Now:" line, rail and grip, below the tick labels
_GRIP_INSET_PX = 8  # how far Plotly keeps the grip centre from each end of the rail

_FIGURE_CACHE_LIMIT = 8
"""Replay figures kept per result: about 3 MB of frames each."""

_PLAN_STATUS_TEXT = {
    1: "this session charges as unmanaged; it does not respond to plans",
    2: "this session charges as unmanaged; maker offline",
    3: "this session charges as unmanaged; control outage",
    4: "this session charges as unmanaged; control group",
}
"""Why a plan is on record but not followed (``plan_status`` codes 1–4, trading §10.1e)."""

_SHOCK_CLASS_TEXT = {"mild": "Mild", "big": "Big", "scripted": "Scripted"}
_EVENT_TEXT = {
    "turn_down": "Turn-down request",
    "turn_up": "Turn-up request",
    "control_outage": "Control outage",
}


# --- Frame parts --------------------------------------------------------------


@dataclass(frozen=True)
class _Track:
    """One trace of the replay figure.

    ``base`` holds the attributes that never change (x for full-length
    traces, style, hover template). ``frames`` holds, per decision instant,
    only the attributes that do change (``y``, or ``x`` for the now line and
    sparse traces, ``text`` for the counters); ``None`` for a static trace.
    One kind and one style per trace for the whole week, so no frame changes
    a trace's type or look (§3.2).
    """

    kind: type
    base: dict[str, Any]
    row: int
    frames: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class _Parts:
    """Everything the figure needs, built once per (world, EV) and cached (§3.6)."""

    tracks: list[_Track]
    row_heights: list[int]
    y_axes: list[dict[str, Any]]
    x_axis: dict[str, Any]
    labels: list[str]
    table: pd.DataFrame


def _values(values: Any) -> np.ndarray:
    """One frame's values as float32, NaN where hidden or missing (a gap, never zero).

    Plotly sends a NumPy array as a typed binary array, 4 bytes a value in
    float32, about the size of one-decimal text (§3.5), and validates it
    without walking every element, which keeps reassembling 169 frames
    under a second. NaN is Plotly's "no value", the typed-array form of
    ``None``: no point, no bar and no hover. float32 keeps 7 significant
    figures, more than any label shows. A line frame with nothing to show
    sends one NaN (Plotly draws a trace up to the shorter of x and y).
    """

    array = np.asarray(values, dtype=np.float32)
    # One NaN, not an empty array: with redraw off Plotly skips drawing a
    # trace with no points and would leave the previous frame's line on screen.
    return array if np.isfinite(array).any() else array[:1] * np.nan


def _bar_values(values: Any) -> np.ndarray:
    """``_values`` for a bar trace, always full length.

    With redraw off, bars that come back after a one-point frame are drawn
    without their style (black, seen in the browser), so bar frames keep
    every slot and hide with NaN.
    """

    return np.asarray(values, dtype=np.float32)


def _past(values: np.ndarray, k: int) -> np.ndarray:
    """Slots that have ended by ``tau_k`` (t < 2k); later slots are hidden."""

    out = np.array(values, dtype=float)
    out[2 * k :] = np.nan
    return out


def _ahead(values: np.ndarray, k: int) -> np.ndarray:
    """Slots from ``tau_k`` on (t >= 2k); earlier slots are hidden."""

    out = np.array(values, dtype=float)
    out[: 2 * k] = np.nan
    return out


def _cached(result: Any, key: tuple, build: Callable[[], _Parts]) -> _Parts:
    """Per-result cache on ``result.replay_state.replay_cache`` (§3.6, review B2).

    Keyed on the result's own object, never a process-wide cache on
    ``id(result)``: an id is reused after garbage collection, so a new run
    could be served the previous run's frames. A result without that cache
    (a test fixture) builds each time.
    """

    cache = getattr(getattr(result, "replay_state", None), "replay_cache", None)
    if cache is None:
        return build()
    if key not in cache:
        # Each entry holds a few MB of frames, so only the most recent few
        # (world, EV) choices are kept; the oldest goes first, like the
        # one-EV replays that share this dict.
        figures = [name for name in cache if name[0] == key[0]]
        if len(figures) >= _FIGURE_CACHE_LIMIT:
            cache.pop(figures[0])
        cache[key] = build()
    return cache[key]


# --- Shared tracks ------------------------------------------------------------


def _counter_track(texts: list[str], x0: Any) -> _Track:
    """The running counters: one text trace in the slim top row (§3.2).

    Text, not KPI tiles, because a tile only changes on a rerun and the
    slider must not rerun the page.
    """

    return _Track(
        go.Scatter,
        {
            "x": [x0],
            "y": [0],
            "mode": "text",
            "textposition": "middle right",
            "textfont": {"color": INK},
            # Text is not clipped to the counter row's hidden axes.
            "cliponaxis": False,
            "showlegend": False,
            "hoverinfo": "skip",
        },
        1,
        [{"text": [text]} for text in texts],
    )


def _now_track(decision_utc: pd.Series, row: int, y_range: list[float]) -> _Track:
    """The "now" line of one panel: a two-point scatter whose x moves per frame."""

    return _Track(
        go.Scatter,
        {
            "y": y_range,
            "mode": "lines",
            "line": {"color": INK, "width": 1.5},
            "showlegend": False,
            "hoverinfo": "skip",
        },
        row,
        [{"x": [moment, moment]} for moment in decision_utc],
    )


def _price_tracks(replay: Any, wi: int, starts: pd.Series, realised_extra: pd.DataFrame | None):
    """The price panel's per-frame traces and its fixed y range (§3.3 row 2).

    Realised: the intraday close to now (the end row of the known-intraday
    cube, which the validator checks equals the realised close). From now:
    the day-ahead price where published, the expected shape where not (out of
    the legend), and the latest intraday value known at now.
    """

    visible = replay.day_ahead_visible_gbp_per_mwh[wi]
    known = replay.intraday_known_gbp_per_mwh[wi]
    published = replay.published
    close = known[-1]
    count = len(replay.decisions)
    hover = hover_time_labels(starts)
    if realised_extra is None:
        realised_custom = [[label] for label in hover]
        realised_hover = "%{customdata[0]}<br>Realised: £%{y:.1f}/MWh<extra></extra>"
    else:
        realised_custom = [
            [label, day_ahead, imbalance]
            for label, day_ahead, imbalance in zip(
                hover,
                realised_extra["day_ahead_gbp_per_mwh"].round(1),
                realised_extra["imbalance_gbp_per_mwh"].round(1),
                strict=True,
            )
        ]
        realised_hover = (
            "%{customdata[0]}<br>Realised (intraday close): £%{y:.1f}/MWh"
            "<br>Day-ahead £%{customdata[1]}, imbalance £%{customdata[2]}/MWh<extra></extra>"
        )
    forward_style = {"color": SERIES_COLOURS["normal"], "width": 1.5}
    tracks = [
        _Track(
            go.Scatter,
            {
                "x": starts,
                "name": "Realised price",
                "mode": "lines",
                "line": {"color": INK, "width": 1.5},
                "customdata": realised_custom,
                "hovertemplate": realised_hover,
            },
            2,
            [{"y": _values(_past(close, k))} for k in range(count)],
        ),
        _Track(
            go.Scatter,
            {
                "x": starts,
                "name": "Day-ahead (published)",
                "mode": "lines",
                "line": {**forward_style, "dash": "dash"},
                "hovertemplate": "Day-ahead (published): £%{y:.1f}/MWh<extra></extra>",
            },
            2,
            [
                {"y": _values(_ahead(np.where(published[k], visible[k], np.nan), k))}
                for k in range(count)
            ],
        ),
        _Track(
            go.Scatter,
            {
                "x": starts,
                "name": "Expected shape",
                "mode": "lines",
                "line": {**forward_style, "dash": "dot"},
                "showlegend": False,
                "hovertemplate": (
                    "Expected shape, not yet published: £%{y:.1f}/MWh<extra></extra>"
                ),
            },
            2,
            [
                {"y": _values(_ahead(np.where(published[k], np.nan, visible[k]), k))}
                for k in range(count)
            ],
        ),
        _Track(
            go.Scatter,
            {
                "x": starts,
                "name": "Latest intraday",
                "mode": "lines",
                "line": {"color": SERIES_COLOURS["difference"], "width": 1.5, "dash": "dash"},
                "hovertemplate": "Latest intraday: £%{y:.1f}/MWh<extra></extra>",
            },
            2,
            [{"y": _values(_ahead(known[k], k))} for k in range(count)],
        ),
    ]
    finite = np.concatenate([visible[np.isfinite(visible)], known[np.isfinite(known)]])
    low, high = float(finite.min()), float(finite.max())
    pad = 0.06 * max(high - low, 1.0)
    # The marker row sits just above the highest price, inside a fixed range,
    # so no hidden point ever rescales the panel (§3.2).
    marker_y = high + pad
    return tracks, [low - pad, marker_y + pad], marker_y


def _known_markers(replay: Any, wi: int, marker_y: float) -> _Track | None:
    """Shocks and events, each shown from the frame at which it became known (§1.5, §3.2).

    One marker trace: every shock and event of this week is in the base
    trace with its fixed position, symbol and hover, and each frame only sets
    ``y`` to the marker row or ``None`` (not yet known), so no frame changes
    a marker's style.
    """

    world = replay.world_ids[wi]
    shocks = replay.shocks.loc[replay.shocks["world_id"].eq(world)]
    events = replay.events
    if shocks.empty and events.empty:
        return None
    x, symbols, texts, known = [], [], [], []
    for row in shocks.itertuples(index=False):
        up = row.direction == "up"
        lead_hours = (row.start_utc - row.known_from_utc) / pd.Timedelta(hours=1)
        seen = (
            "in the day-ahead price"
            if row.known_day_ahead
            else f"surprise, seen {lead_hours:.0f} h ahead"
        )
        x.append(row.start_utc)
        symbols.append("triangle-up" if up else "triangle-down")
        texts.append(
            f"{_SHOCK_CLASS_TEXT.get(row.shock_class, row.shock_class)} price shock "
            f"{'up' if up else 'down'}, {row.size_gw:.1f} GW, "
            f"{row.duration_slots * _SLOT_HOURS:g} h; {seen}"
        )
        known.append(row.known_from_utc)
    for row in events.itertuples(index=False):
        x.append(row.start_utc)
        symbols.append("square")
        label = _EVENT_TEXT.get(row.event_type, row.event_type)
        ends = row.end_utc.tz_convert("Europe/London")
        window = f"{hover_time_labels([row.start_utc])[0]} to {ends:%H:%M}"
        pays = (
            f"; pays £{row.payment_gbp_per_mwh:.0f}/MWh (illustrative)"
            if row.event_type in ("turn_down", "turn_up")
            else ""
        )
        texts.append(f"{label}, {window}{pays}")
        known.append(row.known_from_utc)
    known = pd.DatetimeIndex(known)
    return _Track(
        go.Scatter,
        {
            "x": x,
            "mode": "markers",
            "marker": {"symbol": symbols, "size": 10, "color": SERIES_COLOURS["observed"]},
            "name": "Shocks and events",
            "showlegend": False,
            "text": texts,
            "hovertemplate": "%{text}<extra></extra>",
        },
        2,
        [
            {"y": [marker_y if when <= moment else None for when in known]}
            for moment in replay.decisions["decision_utc"]
        ],
    )


def _x_axis(starts: pd.Series, ends: pd.Series) -> dict[str, Any]:
    return {
        "tickmode": "array",
        # Pinned to the study so markers never widen the axis.
        "range": [starts.iat[0], ends.iat[-1]],
        **london_time_axis(starts),
    }


def _figure(parts: _Parts, active: int) -> go.Figure:
    """Assemble the animated figure with frame ``active`` as the base data (§3.2, §3.6).

    Cheap (no model call): the per-frame arrays are cached, so the Jump
    control and lens switches only reassemble this.
    """

    heights = parts.row_heights
    plot_px = sum(heights) + _GAP_PX * (len(heights) - 1)
    figure = make_subplots(
        rows=len(heights),
        cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=_GAP_PX / plot_px,
    )
    for track in parts.tracks:
        attributes = {**track.base, **(track.frames[active] if track.frames else {})}
        if len(track.base.get("x", ())) > 1 and len(attributes.get("y", ())) == 1:
            # Plotly leaves a trace with no points out of the legend, so the
            # base keeps every full-length trace at full length (all gaps).
            attributes["y"] = np.full(len(track.base["x"]), np.nan, dtype=np.float32)
        figure.add_trace(track.kind(**attributes), row=track.row, col=1)
    animated = [index for index, track in enumerate(parts.tracks) if track.frames]
    count = len(parts.labels)
    figure.frames = [
        go.Frame(
            name=str(k),
            traces=animated,
            data=[parts.tracks[index].kind(**parts.tracks[index].frames[k]) for index in animated],
        )
        for k in range(count)
    ]
    for row, y_axis in enumerate(parts.y_axes, start=1):
        figure.update_yaxes(**y_axis, row=row, col=1)
    # The counter row has no axes of its own to read.
    figure.update_xaxes(showgrid=False, row=1, col=1)
    figure.update_xaxes(**parts.x_axis, row=len(heights), col=1)
    domain = list(figure.layout.xaxis.domain or (0, 1))
    frame_args = {"frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}
    figure.update_layout(
        barmode="overlay",
        sliders=[
            {
                "active": active,
                # On the x-axis domain so the grip sits under the now line (O1).
                "x": domain[0],
                "len": domain[1] - domain[0],
                "xanchor": "left",
                "y": -_TICK_ROW_PX / plot_px,
                "yanchor": "top",
                # Plotly insets the grip's travel by half a grip at each end
                # of the rail; negative side padding cancels that, so the grip
                # centre, not only the rail, spans the x-axis (measured in the
                # browser at 1440 and 390 px).
                "pad": {"t": 0, "b": 0, "l": -_GRIP_INSET_PX, "r": -_GRIP_INSET_PX},
                "ticklen": 0,
                "minorticklen": 0,
                "tickcolor": TRANSPARENT,
                "font": {"size": 1, "color": TRANSPARENT},  # step labels hidden
                "currentvalue": {"prefix": "Now: ", "font": {"color": INK, "size": 12}},
                "bgcolor": MUTED_INK,
                "activebgcolor": INK,
                "bordercolor": MUTED_INK,
                "steps": [
                    {
                        "method": "animate",
                        "label": label,
                        "args": [[str(k)], {"mode": "immediate", **frame_args}],
                    }
                    for k, label in enumerate(parts.labels)
                ],
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": domain[0],
                "xanchor": "left",
                "y": 1,
                "yanchor": "bottom",
                "pad": {"t": 0, "b": 8, "l": 0, "r": 0},
                "bgcolor": BORDER,
                "bordercolor": MUTED_INK,
                "font": {"color": INK, "size": 12},
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            # Plotly's "from current" needs a frame it has
                            # animated to; after a Jump nothing has been
                            # animated yet, so Play lists the frames from the
                            # jumped-to instant (all frames when that is 0).
                            # Dragging back before that instant and pressing
                            # Play resumes from the jumped-to instant.
                            None if active == 0 else [str(k) for k in range(active, count)],
                            {
                                "frame": {"duration": PLAY_MS, "redraw": False},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", **frame_args}],
                    },
                ],
            }
        ],
    )
    # A view's own bottom margin wins in style_figure. Below the plot: tick
    # labels, the slider, then the legend. Plotly lays a wrapping horizontal
    # legend out in columns as wide as its widest entry, so at 390 px
    # "Plan on record, not followed" or "Day-ahead (published)" leaves one
    # entry per row. Those rows are reserved and the legend hangs from just
    # under the slider, so at 1440 px (one row) the spare rows sit empty at
    # the very bottom rather than between slider and legend.
    names = _legend_names(figure)
    widest = max((_LEGEND_SYMBOL_PX + _LEGEND_CHAR_PX * len(name) for name in names), default=0)
    columns = max(1, int(_NARROW_FIGURE_WIDTH_PX // widest)) if names else 1
    legend_px = _LEGEND_GAP_PX + _LEGEND_ROW_PX * -(-len(names) // columns)
    bottom = _TICK_ROW_PX + _SLIDER_PX + legend_px
    height = _TOP_PX + plot_px + bottom
    figure.update_layout(
        margin={"t": _TOP_PX, "b": bottom},
        height=height,
        legend={"yanchor": "top", "y": (legend_px - _LEGEND_GAP_PX) / height},
    )
    return figure


# --- Controls -----------------------------------------------------------------


def _week_label(world: int, representative: int) -> str:
    return f"Week {world} (median)" if world == representative else f"Week {world}"


def _week_control(column: Any, result: Any) -> int:
    """The Simulated week selectbox: the replay's sampled worlds, the median week first."""

    worlds = list(result.replay_week.world_ids)
    representative = result.representative_world_id
    return column.selectbox(
        "Simulated week",
        options=worlds,
        index=worlds.index(representative) if representative in worlds else 0,
        format_func=lambda world: _week_label(world, representative),
        key=WEEK_KEY,
        label_visibility="collapsed",
    )


def _jump_control(st: Any, labels: list[str]) -> int:
    """The keyboard path to any instant (§3.6).

    Plotly's slider and buttons are not keyboard-operable, so this selectbox
    reruns the page with the chosen frame as the chart's starting point. It
    never starts a run: the figure is reassembled from cached arrays.
    """

    with st.expander("Keyboard access"):
        return st.selectbox(
            "Jump to",
            options=list(range(len(labels))),
            format_func=lambda k: labels[k],
            key=JUMP_KEY,
            help="The play button and slider need a mouse or touch; this moves the chart too.",
        )


# --- Fleet lens ---------------------------------------------------------------


def _world_rows(frame: pd.DataFrame | None, world: int) -> pd.DataFrame | None:
    if frame is None:
        return None
    return frame.loc[frame["world_id"].eq(world)].sort_values("slot_index")


def _fleet_counter_texts(result: Any, wi: int, world: int) -> list[str]:
    """The Fleet counters per instant: now, the drivers' saving and the trading P&L so far.

    Four short lines, each under about 50 characters so they fit a 390 px
    screen; every pound says "illustrative".
    """

    replay = result.replay_week
    saving = replay.energy_saving_to_date_gbp[wi]
    cash = replay.trading_cash_to_date_gbp
    shortfall = _shortfall_value(result, world)
    last = len(replay.decisions) - 1
    texts = []
    for k, label in enumerate(replay.decisions["label"]):
        lines = [
            f"<b>Now: {label}</b>",
            f"Drivers’ saving so far (illustrative) {money(saving[2 * k])}",
        ]
        if cash is not None:
            lines.append(f"Trading P&L so far (illustrative simulated) {money(cash[wi, 2 * k])}")
        if k == last and shortfall is not None:
            # The two end-of-week shortfall values (energy not recovered,
            # travel not served) are not flows, so they are not accumulated;
            # the end frame shows them beside the running figure (§1.4).
            lines.append(f"Less shortfall value at week end (illustrative) {money(shortfall)}")
        texts.append("<br>".join(lines))
    return texts


def _shortfall_value(result: Any, world: int) -> float | None:
    cost = result.cost_effect
    if cost is None:
        return None
    row = cost.loc[cost["world_id"].eq(world)]
    if row.empty:
        return None
    return float(
        row["illustrative_unrecovered_energy_value_gbp"].iat[0]
        + row["illustrative_unserved_travel_value_gbp"].iat[0]
    )


def _import_kw(result: Any, world: int) -> dict[str, np.ndarray]:
    """Fleet home import (kW) per policy present, from the trading or the fleet frame (§1.6).

    With trading frames the fleet lens reads the ledger's own unmanaged/metered
    path for normal and selected: the money side of the fleet lens (deviation,
    baseline, book, position) stays normal versus selected only (trading
    contract v1 §1.6, decision 0007 model step 2 item 6) and is never
    redefined for the timed path. The optional timed import line itself is a
    plain, uncosted series (decision 0007): it still reads from
    ``fleet_world_intervals`` for this same world, exactly as the no-trading
    branch below reads every path, because that frame's own "normal" row is
    bit-identical to the ledger's ``unmanaged_kw`` (checked directly against
    a real run before this was wired in), so the three lines share one kW
    basis. Without trading frames every path, timed included, comes from
    ``fleet_world_intervals`` directly.
    """

    deviation = _world_rows(result.deviation_world_slot, world)
    if deviation is not None:
        import_kw = {
            "normal": deviation["unmanaged_kw"].to_numpy(float),
            "selected": deviation["metered_kw"].to_numpy(float),
        }
        intervals = result.fleet_world_intervals
        timed_rows = _world_rows(intervals.loc[intervals["path_id"].eq("timed")], world)
        if not timed_rows.empty:
            import_kw["timed"] = timed_rows["home_import_kw"].to_numpy(float)
        return import_kw
    intervals = result.fleet_world_intervals
    return {
        path: _world_rows(intervals.loc[intervals["path_id"].eq(path)], world)[
            "home_import_kw"
        ].to_numpy(float)
        for path in present_paths(intervals["path_id"])
    }


def _trade_tracks(result: Any, wi: int, world: int, starts: pd.Series) -> list[_Track]:
    """Trades made at now, on the sold turn-down line (§3.3 row 3).

    The rows of ``position_updates`` whose ``decision_utc`` equals the
    instant exactly (no cut-off logic); ``▲`` sold more turn-down, ``▼``
    bought some back. Sparse traces: each frame carries its own x and y.
    """

    updates = result.position_updates
    replay = result.replay_week
    rows = updates.loc[updates["world_id"].eq(world) & (updates["trade_kwh"].abs() > 1e-9)]
    by_instant = dict(tuple(rows.groupby("decision_utc")))
    empty = rows.iloc[0:0]
    tracks = []
    kinds = (("Sold", "triangle-up", True), ("Bought back", "triangle-down", False))
    for name, symbol, sold in kinds:
        frames = []
        for moment in replay.decisions["decision_utc"]:
            trades = by_instant.get(moment, empty)
            trades = trades.loc[(trades["trade_kwh"] > 0) == sold]
            if trades.empty:
                # One hidden point, not none: see ``_values``.
                frames.append({"x": [starts.iat[0]], "y": [None], "customdata": [["", 0, 0, ""]]})
                continue
            frames.append(
                {
                    "x": list(trades["interval_start_utc"]),
                    "y": _values(trades["position_kwh"] / _SLOT_HOURS),
                    "customdata": [
                        [label, round(trade, 1), round(price, 1), stage.replace("_", " ")]
                        for label, trade, price, stage in zip(
                            hover_time_labels(trades["interval_start_utc"]),
                            trades["trade_kwh"],
                            trades["price_gbp_per_mwh"],
                            trades["stage"],
                            strict=True,
                        )
                    ],
                }
            )
        tracks.append(
            _Track(
                go.Scatter,
                {
                    "mode": "markers",
                    "marker": {"symbol": symbol, "size": 9, "color": SERIES_COLOURS["difference"]},
                    "name": f"{name} at now",
                    "showlegend": False,
                    "hovertemplate": (
                        f"{name} at now, for %{{customdata[0]}}<br>Δ %{{customdata[1]}} kWh at "
                        "£%{customdata[2]}/MWh (%{customdata[3]})<extra></extra>"
                    ),
                },
                3,
                frames,
            )
        )
    return tracks


def _fleet_parts(result: Any, world: int) -> _Parts:
    replay = result.replay_week
    wi = list(replay.world_ids).index(world)
    slots = result.study_slots
    starts, ends = slots["interval_start_utc"], slots["interval_end_utc"]
    decisions = replay.decisions
    count = len(decisions)
    deviation = _world_rows(result.deviation_world_slot, world)

    counter = _counter_track(_fleet_counter_texts(result, wi, world), starts.iat[0])
    price_tracks, price_range, marker_y = _price_tracks(replay, wi, starts, deviation)
    markers = _known_markers(replay, wi, marker_y)

    import_kw = _import_kw(result, world)
    hover = hover_time_labels(starts)
    action_tracks = []
    for path in present_paths(import_kw):
        values = import_kw[path]
        style = PATH_STYLES[path]
        action_tracks.append(
            _Track(
                go.Scatter,
                {
                    "x": starts,
                    "name": PATH_LABELS[path],
                    "mode": "lines",
                    "line": {
                        "color": style["color"],
                        "width": style["width"],
                        "dash": style.get("dash", "solid"),
                    },
                    # Linear for unmanaged/smart; the optional timed path
                    # draws as a step (decision 0007: the barred/allowed
                    # state holds for a whole half-hour, see PATH_STYLES).
                    "line_shape": style.get("line_shape", "linear"),
                    "customdata": hover,
                    "hovertemplate": (
                        f"%{{customdata}}<br>{PATH_LABELS[path]}: %{{y:,.0f}} kW<extra></extra>"
                    ),
                },
                3,
                [{"y": _values(_past(values, k))} for k in range(count)],
            )
        )
    peak = max(max(float(np.nanmax(values)) for values in import_kw.values()), 1.0)
    low = 0.0
    if replay.position_kwh is not None:
        # The position as it stood at now: past slots final, open slots the
        # current target (§1.3), in kW of turn-down.
        turn_down_kw = replay.position_kwh[wi] / _SLOT_HOURS
        action_tracks.append(
            _Track(
                go.Scatter,
                {
                    "x": starts,
                    "name": "Sold turn-down",
                    "mode": "lines",
                    "line": {"color": SERIES_COLOURS["difference"], "width": 1.5, "dash": "dash"},
                    "line_shape": "hv",
                    "hovertemplate": "Sold turn-down: %{y:,.0f} kW<extra></extra>",
                },
                3,
                [{"y": _values(turn_down_kw[k])} for k in range(count)],
            )
        )
        if np.isfinite(turn_down_kw).any():
            peak = max(peak, float(np.nanmax(turn_down_kw)))
            low = min(low, float(np.nanmin(turn_down_kw)))
    if result.position_updates is not None:
        action_tracks.extend(_trade_tracks(result, wi, world, starts))
    action_range = [low - 0.04 * peak, 1.06 * peak]

    tracks = [counter, *price_tracks]
    if markers is not None:
        tracks.append(markers)
    tracks.append(_now_track(decisions["decision_utc"], 2, price_range))
    tracks.extend(action_tracks)
    tracks.append(_now_track(decisions["decision_utc"], 3, action_range))

    table = decisions.loc[:, ["label", "decision_utc"]].rename(
        columns={"label": "Instant (London)", "decision_utc": "Instant (UTC)"}
    )
    table["Drivers’ saving so far (£, illustrative)"] = np.round(
        replay.energy_saving_to_date_gbp[wi, 0::2], 2
    )
    if replay.trading_cash_to_date_gbp is not None:
        table["Trading P&L so far (£, illustrative simulated)"] = np.round(
            replay.trading_cash_to_date_gbp[wi, 0::2], 2
        )
    return _Parts(
        tracks=tracks,
        row_heights=[
            CHART_HEIGHTS["replay_counter"],
            CHART_HEIGHTS["replay_panel"],
            CHART_HEIGHTS["replay_panel"],
        ],
        y_axes=[
            {"visible": False, "range": [-1, 1], "fixedrange": True},
            {"title": "£/MWh", "range": price_range, "fixedrange": True},
            {"title": "kW", "range": action_range, "fixedrange": True},
        ],
        x_axis=_x_axis(starts, ends),
        labels=list(decisions["label"]),
        table=table,
    )


def _fleet_kpis(st: Any, result: Any, world: int) -> None:
    """Static week totals for this world above the chart (§3.3), illustrative."""

    cost = result.cost_effect.loc[result.cost_effect["world_id"].eq(world)]
    saving = -float(cost["illustrative_selected_minus_normal_total_gbp"].iat[0])
    # The timed path's own saving (decision 0007, model step 2): shown beside
    # the smart saving, never instead of it, and only when the model actually
    # built ``timed_cost_effect`` for this run (``timed_start_local_hour``
    # set); otherwise this tile is left out and the row is exactly as today.
    timed_cost_effect = getattr(result, "timed_cost_effect", None)
    timed_saving = None
    if timed_cost_effect is not None:
        timed_rows = timed_cost_effect.loc[timed_cost_effect["world_id"].eq(world)]
        if not timed_rows.empty:
            timed_saving = -float(timed_rows["illustrative_selected_minus_normal_total_gbp"].iat[0])
    week = result.trading_week_world
    net = None
    if week is not None:
        rows = week.loc[week["world_id"].eq(world) & week["strategy"].eq("full")]
        net = float(rows["net_gbp"].iat[0]) if not rows.empty else None
    columns = kpi_columns(st, 1 + (timed_saving is not None) + (net is not None))
    kpi(
        columns[0],
        # A typographic apostrophe: html.escape turns ' into six characters
        # and the copy rules count the escaped label.
        "Drivers’ saving this week",
        money(saving, signed=True),
        context="Illustrative, this simulated week",
        help=(
            "Unmanaged minus smart cost at the day-ahead price, plus public top-ups, less the "
            "value of energy not recovered and travel not served by the end of the week."
        ),
    )
    index = 1
    if timed_saving is not None:
        kpi(
            columns[index],
            "Timed saving this week",
            money(timed_saving, signed=True),
            context="Illustrative, this simulated week",
            help=(
                "Unmanaged minus Timed tariff cost, read the same way as the smart saving "
                "beside it: the day-ahead price for home charging, plus public top-ups. All "
                "three policies are costed alike; no tariff rate is applied to any of them."
            ),
        )
        index += 1
    if net is not None:
        kpi(
            columns[index],
            "Trading P&L this week",
            money(net, signed=True),
            context="Illustrative simulated, this week",
            help="The trading desk's net from selling turn-down, after trading costs and payments.",
        )


def render_replay_fleet(st: Any, result: Any) -> None:
    """Render Replay ▸ Fleet: prices known, actions taken and money earned, hour by hour."""

    replay = getattr(result, "replay_week", None)
    if replay is None:
        st.info(UNAVAILABLE_MESSAGE)
        return
    world = _week_control(header.controls(st), result)
    _fleet_kpis(st, result, world)
    parts = _cached(result, ("replay_figure", world, None), lambda: _fleet_parts(result, world))
    active = _jump_control(st, parts.labels)
    figure = _figure(parts, active)
    # Clarity critique HIGH item 5: frame 0 is before anything has happened,
    # so the SoC/kWh lower panels are empty and only a few triangles show in
    # the kW panel -- a first-time viewer reads that as the chart having
    # failed, not as "press Play".
    st.caption(
        f"Press Play or drag the slider. The chart starts at {parts.labels[0]}, before "
        "anything has happened, so the lower panels start empty."
    )
    chart_block(
        st,
        figure,
        title="Prices, fleet charging and money as the week unfolds",
        caption=(
            f"{_week_label(world, result.representative_world_id)}, one simulated week. "
            "Dashed: prices known at the cursor; solid: what had happened. Illustrative."
        ),
        frame=parts.table,
        definition=(
            "One row per hourly decision instant. At each instant the chart shows only what "
            "had happened by then. The rest of the week is hidden, and the dashed lines are "
            "the prices known at that moment (dotted: the expected shape before the day-ahead "
            "price is published). The drivers' saving values home charging at the day-ahead "
            "price, as the weekly cost does. Trading P&L is booked by delivery half-hour; the "
            "positions are the trading desk's full strategy, and the ledger pays for "
            "turn-down only. A grid request whose window crosses noon is booked at the end of "
            "the night it starts in, so the P&L shows its payment before the window's last "
            "half-hours have ended."
        ),
        height=figure.layout.height,
        key="replay-fleet-chart",
    )


# --- Customer lens ------------------------------------------------------------


def _ev_control(column: Any, result: Any) -> str:
    """The EV selectbox with One EV's labels and default (first Average (UK) EV)."""

    units = result.units
    unit_ids = list(units["unit_id"])
    labels = {row.unit_id: f"{row.unit_id} · {row.cohort_label}" for row in units.itertuples()}
    average_uk = list(units.loc[units["cohort_id"].eq("average_uk"), "unit_id"])
    default = average_uk[0] if average_uk else unit_ids[0]
    return column.selectbox(
        "EV",
        options=unit_ids,
        index=unit_ids.index(default),
        format_func=lambda uid: labels.get(uid, uid),
        key=EV_KEY,
        label_visibility="collapsed",
    )


def _customer_counter_texts(labels: list[str], timeline: Any | None) -> list[str]:
    """The Customer counters per instant: now, each path's cost so far and the saving.

    One figure per line rather than the one-line "Cost so far
    (illustrative): unmanaged ..., smart ..." so each line fits 390 px.
    """

    texts = []
    for k, label in enumerate(labels):
        lines = [f"<b>Now: {label}</b>"]
        if timeline is None:
            lines.append("Running cost not available for this run")
        else:
            cost = timeline.cumulative
            at = cost.loc[cost["slot_boundary"].eq(2 * k)].set_index("path_id")["cost_to_date_gbp"]
            saved = timeline.saving_to_date_gbp[2 * k]
            lines += [
                f"Unmanaged cost so far (illustrative) {money(at['normal'], decimals=2)}",
                f"Smart cost so far (illustrative) {money(at['selected'], decimals=2)}",
                f"Saved so far (illustrative) {money(saved, decimals=2)}",
            ]
        texts.append("<br>".join(lines))
    return texts


def _soc_tracks(ev: Any, decisions: pd.DataFrame) -> tuple[list[_Track], _Track | None]:
    """SoC panel (§3.4 row 3): plugged-in shading and each present policy's SoC line, and markers.

    Draws the optional timed path's own SoC line beside unmanaged and smart
    (decision 0007) whenever this EV's replay carries it; off, exactly the
    two lines as before.
    """

    count = len(decisions)
    intervals = ev.intervals
    paths = present_paths(intervals["path_id"])
    rows = {
        path: intervals.loc[intervals["path_id"].eq(path)].sort_values("slot_index")
        for path in paths
    }
    starts = rows["normal"]["interval_start_utc"]
    hover = hover_time_labels(starts)
    # A future plug-in is not yet known at now, so the shading stops at now
    # too. Shading is 100 or 0 (a zero-height fill draws nothing), so the
    # hidden future is 0 rather than NaN and the frames go as one byte a
    # slot, which keeps this lens inside the payload bound (§3.5).
    connected = rows["normal"]["connected"].to_numpy(bool)
    shading = [
        np.where(connected & (np.arange(len(connected)) < 2 * k), 100, 0).astype(np.uint8)
        for k in range(count)
    ]
    tracks = [
        _Track(
            go.Scatter,
            {
                "x": starts,
                "name": "Plugged in at home",
                "mode": "lines",
                "line": {"width": 0},
                "line_shape": "hv",
                "fill": "tozeroy",
                "fillcolor": band_fill(SERIES_COLOURS["plugged"], PLUGGED_ALPHA),
                "hoverinfo": "skip",
            },
            3,
            [{"y": values} for values in shading],
        )
    ]
    for path in paths:
        style = PATH_STYLES[path]
        tracks.append(
            _Track(
                go.Scatter,
                {
                    "x": starts,
                    # One legend entry per path: the charging bars below
                    # join this line's group, so each policy is named once
                    # for both panels (§3.2 legend names).
                    "name": PATH_LABELS[path],
                    "legendgroup": path,
                    "mode": "lines",
                    "line": {
                        "color": style["color"],
                        "width": style["width"],
                        "dash": style.get("dash", "solid"),
                    },
                    # The optional timed path draws as a step, same reasoning
                    # as the import line above (decision 0007).
                    "line_shape": style.get("line_shape", "linear"),
                    "customdata": hover,
                    "hovertemplate": (
                        f"%{{customdata}}<br>SoC ({PATH_LABELS[path].lower()}): "
                        "%{y:.0f}%<extra></extra>"
                    ),
                },
                3,
                [
                    {"y": _values(_past(rows[path]["battery_soc_percent"].to_numpy(), k))}
                    for k in range(count)
                ],
            )
        )
    return tracks, _ev_markers(ev, rows["selected"], decisions)


def _ev_markers(ev: Any, selected: pd.DataFrame, decisions: pd.DataFrame) -> _Track | None:
    """One EV's markers, each shown only once what it marks has happened (§3.4).

    ``▲`` plug-in, ``▼`` departure, ``◆`` public top-up, ``✕`` left before
    the plan finished, on the smart path. As with the shock markers, the base
    trace holds every marker and each frame only sets ``y`` or ``None``.

    Plug-in, top-up and left-early markers are stamped with their slot's
    start and plotted at that slot's closing SoC, which is known only once
    the slot has ended; they appear from ``tau >= start + 30 min`` (slot
    index < 2k, the same rule as the realised lines). Showing them at the
    slot's start would leak the slot's outcome (review of 9f2a022, B1). A
    departure is an instant with its own SoC, so it shows from ``tau >=
    departure``.
    """

    plugs = ev.plug_events.loc[ev.plug_events["path_id"].eq("normal")]
    audit = ev.daily_audit
    departures = audit.loc[audit["path_id"].eq("selected") & audit["drives_today"]].dropna(
        subset=["departure_utc"]
    )
    topups = selected.loc[selected["public_import_kwh"] > 0]
    early = selected.loc[selected["early_departure_shortfall_kwh"] > 1e-9]
    slot_end = pd.Timedelta(hours=_SLOT_HOURS)
    instant = pd.Timedelta(0)
    groups = [
        (
            plugs["plug_in_utc"],
            plugs["plug_in_soc_percent"],
            "triangle-up",
            "Plug-in",
            INK,
            slot_end,
        ),
        (
            departures["departure_utc"],
            departures["departure_soc_percent"],
            "triangle-down",
            "Departure",
            SERIES_COLOURS["selected"],
            instant,
        ),
        (
            topups["interval_start_utc"],
            topups["battery_soc_percent"],
            "diamond",
            "Public top-up",
            SERIES_COLOURS["selected"],
            slot_end,
        ),
        (
            early["interval_start_utc"],
            early["battery_soc_percent"],
            "x",
            "Left before plan finished",
            SERIES_COLOURS["flag"],
            slot_end,
        ),
    ]
    x, y, symbols, colours, texts, known = [], [], [], [], [], []
    for times, socs, symbol, name, colour, lag in groups:
        for moment, soc, label in zip(times, socs, hover_time_labels(times), strict=True):
            x.append(moment)
            known.append(pd.Timestamp(moment) + lag)
            y.append(round(float(soc), 1))
            symbols.append(symbol)
            colours.append(colour)
            texts.append(f"{label}<br>{name}, SoC {soc:.0f}%")
    if not x:
        return None
    pairs = list(zip(y, pd.DatetimeIndex(known), strict=True))
    return _Track(
        go.Scatter,
        {
            "x": x,
            "mode": "markers",
            "marker": {"symbol": symbols, "size": 9, "color": colours},
            "name": "Plug-ins and departures",
            "showlegend": False,
            "text": texts,
            "hovertemplate": "%{text}<extra></extra>",
        },
        3,
        [
            {"y": [value if when <= moment else None for value, when in pairs]}
            for moment in decisions["decision_utc"]
        ],
    )


def _charging_tracks(ev: Any, timeline: Any | None, decisions: pd.DataFrame) -> list[_Track]:
    """Home charging panel (§3.4 row 4): each present policy to now, then the plan from now.

    Draws the optional timed path's own charging bars beside unmanaged and
    smart (decision 0007) whenever this EV's replay carries it; off, exactly
    the two bars as before. ``Plan in force`` and ``Plan on record, not
    followed`` are two traces, not one whose look changes per frame: a plan
    on record is the smart plan a non-responding session keeps while it
    charges as unmanaged (trading §10.1e), so it is drawn grey, and only one
    of the two carries values at any instant, by the plan status of the slot
    at now. The plan stays smart-only: the timed path has no plan.
    """

    count = len(decisions)
    intervals = ev.intervals
    tracks = []
    for path in present_paths(intervals["path_id"]):
        rows = intervals.loc[intervals["path_id"].eq(path)].sort_values("slot_index")
        style = PATH_STYLES[path]
        tracks.append(
            _Track(
                go.Bar,
                {
                    "x": rows["interval_start_utc"],
                    "name": PATH_LABELS[path],
                    "legendgroup": path,
                    "showlegend": False,
                    "marker": {"color": band_fill(style["color"], 0.55)},
                    "hovertemplate": f"{PATH_LABELS[path]}: %{{y:.2f}} kWh<extra></extra>",
                },
                4,
                [
                    {"y": _bar_values(_past(rows["home_import_kwh"].to_numpy(), k))}
                    for k in range(count)
                ],
            )
        )
    if timeline is None:
        return tracks
    starts = intervals.loc[intervals["path_id"].eq("normal")].sort_values("slot_index")[
        "interval_start_utc"
    ]
    slot_count = len(starts)
    plan = np.where(timeline.plan_at_decision_kwh > 0, timeline.plan_at_decision_kwh, np.nan)
    status = timeline.plan_status
    empty = np.full(slot_count, np.nan)
    in_force, on_record = [], []
    for k in range(count):
        # The status of the slot at now decides which trace carries the plan.
        code = int(status[2 * k]) if 2 * k < slot_count else 0
        in_force.append({"y": _bar_values(plan[k] if code == 0 else empty)})
        reason = _PLAN_STATUS_TEXT.get(code, "")
        on_record.append(
            {
                "y": _bar_values(empty if code == 0 else plan[k]),
                # Hover names why the plan is not followed; text only, no style change.
                "hovertemplate": (
                    f"Plan on record, not followed: %{{y:.2f}} kWh<br>{reason}<extra></extra>"
                ),
            }
        )
    selected_colour = SERIES_COLOURS["selected"]
    grey = SERIES_COLOURS["normal"]
    tracks.append(
        _Track(
            go.Bar,
            {
                "x": starts,
                "name": "Plan in force",
                "marker": {
                    "color": band_fill(selected_colour, 0.12),
                    "line": {"color": selected_colour, "width": 1},
                },
                "hovertemplate": "Plan in force: %{y:.2f} kWh<extra></extra>",
            },
            4,
            in_force,
        )
    )
    tracks.append(
        _Track(
            go.Bar,
            {
                "x": starts,
                "name": "Plan on record, not followed",
                "marker": {
                    "color": band_fill(grey, 0.08),
                    "line": {"color": grey, "width": 1},
                    "pattern": {"shape": "/", "fgcolor": grey, "size": 6},
                },
            },
            4,
            on_record,
        )
    )
    return tracks


def _customer_parts(result: Any, world: int, unit_id: str, replay_one_ev, timeline_fn) -> _Parts:
    replay = result.replay_week
    wi = list(replay.world_ids).index(world)
    slots = result.study_slots
    starts, ends = slots["interval_start_utc"], slots["interval_end_utc"]
    decisions = replay.decisions
    ev = replay_one_ev(result, unit_id, world)
    timeline = timeline_fn(result, unit_id, world) if timeline_fn is not None else None

    labels = list(decisions["label"])
    counter = _counter_track(_customer_counter_texts(labels, timeline), starts.iat[0])
    deviation = _world_rows(getattr(result, "deviation_world_slot", None), world)
    price_tracks, price_range, marker_y = _price_tracks(replay, wi, starts, deviation)
    markers = _known_markers(replay, wi, marker_y)
    soc_tracks, ev_markers = _soc_tracks(ev, decisions)
    charging = _charging_tracks(ev, timeline, decisions)
    peak = max(
        float(ev.intervals["home_import_kwh"].max()),
        float(np.max(timeline.plan_at_decision_kwh)) if timeline is not None else 0.0,
        0.1,
    )
    charging_range = [0.0, 1.08 * peak]

    tracks = [counter, *price_tracks]
    if markers is not None:
        tracks.append(markers)
    tracks.append(_now_track(decisions["decision_utc"], 2, price_range))
    tracks.extend(soc_tracks)
    if ev_markers is not None:
        tracks.append(ev_markers)
    tracks.append(_now_track(decisions["decision_utc"], 3, [0, 100]))
    tracks.extend(charging)
    tracks.append(_now_track(decisions["decision_utc"], 4, charging_range))

    table = decisions.loc[:, ["label", "decision_utc"]].rename(
        columns={"label": "Instant (London)", "decision_utc": "Instant (UTC)"}
    )
    if timeline is not None:
        cost = timeline.cumulative
        # The optional timed path's own running cost (decision 0007), read
        # from the same one-EV kernel run as normal and selected, gains a
        # column here exactly when ``cumulative`` carries it; "Saved so
        # far" always stays the normal-versus-selected reading (replay
        # contract v1 §2), whatever else this table carries.
        for path in present_paths(cost["path_id"]):
            values = cost.loc[cost["path_id"].eq(path)].sort_values("slot_boundary")
            table[f"{PATH_LABELS[path]} cost so far (£, illustrative)"] = np.round(
                values["cost_to_date_gbp"].to_numpy()[0::2], 2
            )
        table["Saved so far (£, illustrative)"] = np.round(timeline.saving_to_date_gbp[0::2], 2)
    return _Parts(
        tracks=tracks,
        row_heights=[
            CHART_HEIGHTS["replay_counter"],
            CHART_HEIGHTS["replay_panel"],
            CHART_HEIGHTS["replay_panel"],
            CHART_HEIGHTS["small_multiple_panel"],
        ],
        y_axes=[
            {"visible": False, "range": [-1, 1], "fixedrange": True},
            {"title": "£/MWh", "range": price_range, "fixedrange": True},
            {"title": "SoC %", "range": [0, 100], "fixedrange": True},
            {"title": "kWh", "range": charging_range, "fixedrange": True},
        ],
        x_axis=_x_axis(starts, ends),
        labels=labels,
        table=table,
    )


def _traits_caption(traits: dict[str, Any]) -> str:
    """One EV's traits line, the same wording as Drivers ▸ One EV."""

    miles_per_year = float(traits["daily_miles_mean"]) * 365.0
    return (
        f"{traits['cohort_label']} · {miles_per_year:,.0f} miles/yr · "
        f"{float(traits['physical_capacity_kwh']):.0f} kWh battery · "
        f"{float(traits['home_charger_limit_kw']):.0f} kW charger · "
        f"target SoC {float(traits['preferred_target_soc_percent']):.0f}%"
    )


def _customer_kpis(st: Any, timeline: Any) -> None:
    """This driver's week totals (static tiles; the counters above the chart run with it).

    One tile per policy ``cumulative`` carries (decision 0007: the optional
    timed path's own cost, beside unmanaged and smart, never instead of
    either), plus the saving tile, which always stays the normal-versus-
    selected reading.
    """

    end = timeline.cumulative.loc[
        timeline.cumulative["slot_boundary"].eq(timeline.cumulative["slot_boundary"].max())
    ].set_index("path_id")["cost_to_date_gbp"]
    paths = present_paths(end.index)
    context = "Illustrative, day-ahead price"
    columns = kpi_columns(st, len(paths) + 1)
    for column, path in zip(columns, paths, strict=False):
        kpi(
            column,
            f"{PATH_LABELS[path]} cost this week",
            money(end[path], decimals=2),
            context=context,
        )
    kpi(
        columns[-1],
        "Saved this week",
        money(float(timeline.saving_to_date_gbp[-1]), decimals=2, signed=True),
        context=context,
        help="Home charging at the day-ahead price plus public top-ups, unmanaged minus smart.",
    )


def _snapshot_tables(st: Any, timeline: Any, starts: pd.Series, k: int, label: str) -> None:
    """The plan in force and the running cost at the jumped-to instant (§3.4, O4).

    Streamlit tables cannot follow the client-side slider, so these follow
    the Jump control only, and say so.
    """

    with st.expander(f"Plan and cost at the jumped-to instant ({label})"):
        plan = timeline.plan_at_decision_kwh[k]
        slots = np.flatnonzero(plan > 0)
        st.dataframe(
            pd.DataFrame(
                {
                    "Half-hour (London)": hover_time_labels(starts.iloc[slots]),
                    "Plan (kWh)": np.round(plan[slots], 2),
                    "Followed": [int(timeline.plan_status[s]) == 0 for s in slots],
                }
            ),
            width="stretch",
            hide_index=True,
        )
        cost = timeline.cumulative.loc[timeline.cumulative["slot_boundary"].eq(2 * k)]
        st.dataframe(
            pd.DataFrame(
                {
                    "Path": cost["path_id"].map(PATH_LABELS),
                    "Home import so far (kWh)": cost["home_import_to_date_kwh"].round(2),
                    "Public import so far (kWh)": cost["public_import_to_date_kwh"].round(2),
                    "Cost so far (£, illustrative)": cost["cost_to_date_gbp"].round(2),
                }
            ),
            width="stretch",
            hide_index=True,
        )


def _dispatch_caption(timeline: Any, *, dispatch_active: bool, treated: bool) -> str | None:
    """This EV's intraday dispatch state, or ``None`` when it should not be shown.

    ``timeline.dispatch_locked`` is ``units.dispatch_locked`` (bool on every
    action result, intraday-dispatch-v1 §2), but False both for a free EV
    *and* for every EV when the intraday dispatch switch was off, so
    ``dispatch_active`` (the caller's ``result.dispatch_world_slot is not
    None``, contract §7) tells the two apart: a switch-off run must never
    read its all-False column as "every EV re-plans hourly" (K4-flagged
    bug). A hold-out control EV (``treated`` False) keeps a locked/free
    status but ignores every plan and charges by the normal rule (§2,
    §10.1e), so it is never "locked" or "re-planned" either.
    """

    if not dispatch_active or not treated or timeline.dispatch_locked is None:
        return None
    if timeline.dispatch_locked:
        return "Locked to its day-ahead plan"
    return "Re-plans hourly on the latest intraday price"


def render_replay_customer(
    st: Any,
    result: Any,
    *,
    replay_one_ev: ReplayOneEv,
    replay_one_ev_timeline: ReplayOneEvTimeline | None,
) -> None:
    """Render Replay ▸ Customer: one driver's prices, plan, charging and cost, hour by hour.

    ``replay_one_ev`` and ``replay_one_ev_timeline`` are the one-EV model
    calls (injected, as One EV does), cached on the result. The timeline is
    ``None`` until the model provides it (replay lane R1c): the lens then
    shows prices, SoC and charging but no plan snapshots or running cost,
    and says so.
    """

    replay = getattr(result, "replay_week", None)
    if replay is None:
        st.info(UNAVAILABLE_MESSAGE)
        return
    ev_column, week_column = header.controls(st).columns([3, 2])
    unit_id = _ev_control(ev_column, result)
    world = _week_control(week_column, result)
    traits = result.units.loc[result.units["unit_id"].eq(unit_id)].iloc[0].to_dict()
    st.caption(_traits_caption(traits))

    timeline = (
        replay_one_ev_timeline(result, unit_id, world)
        if replay_one_ev_timeline is not None
        else None
    )
    if timeline is None:
        st.caption("Plan snapshots and running cost for one driver are not available for this run.")
    else:
        _customer_kpis(st, timeline)
        dispatch_active = getattr(result, "dispatch_world_slot", None) is not None
        treated = not bool(traits.get("control_group", False))
        caption = _dispatch_caption(timeline, dispatch_active=dispatch_active, treated=treated)
        if caption is not None:
            st.caption(caption, help=DISPATCH_LOCK_HELP)

    parts = _cached(
        result,
        ("replay_figure", world, unit_id),
        lambda: _customer_parts(result, world, unit_id, replay_one_ev, replay_one_ev_timeline),
    )
    active = _jump_control(st, parts.labels)
    figure = _figure(parts, active)
    # Clarity critique HIGH item 5 (as Replay ▸ Fleet above).
    st.caption(
        f"Press Play or drag the slider. The chart starts at {parts.labels[0]}, before "
        "anything has happened, so the lower panels start empty."
    )
    chart_block(
        st,
        figure,
        title="Prices, battery, charging and cost for one driver as the week unfolds",
        caption=(
            f"{_week_label(world, result.representative_world_id)}, one driver. Saving = "
            "unmanaged minus smart cost so far at the day-ahead price, plus public top-ups. "
            "Illustrative."
        ),
        frame=parts.table,
        definition=(
            "One row per hourly decision instant. The chart shows what had happened by each "
            "instant and, from then on, the prices known and the plan this charger held. "
            "▲ plug-in ▼ departure ◆ public top-up ✕ left before plan finished. Cost values "
            "home charging at the day-ahead price and public top-ups at the public rate. A "
            "plan on record, not followed, is the smart plan a session kept while charging "
            "as unmanaged."
        ),
        height=figure.layout.height,
        key="replay-customer-chart",
    )
    if timeline is not None:
        _snapshot_tables(
            st, timeline, result.study_slots["interval_start_utc"], active, parts.labels[active]
        )
