"""Chart tokens, the shared Plotly template, app CSS and display formatting.

Streamlit's own colours and widget styling live in .streamlit/config.toml.
This module holds what every chart and KPI tile shares (docs/DASHBOARD_DESIGN.md
section 3.5, decision 0004 items 26, 27 and 54; polish plan G1-G6). Model
assumptions and calculations stay out of it.

Why these colours: the earlier unmanaged (#3E7298 dotted) and smart (#0F766E)
lines differed by only 1.06:1 in luminance and 0.2-alpha bands were faint on
the #0E1117 canvas (chart audit O2). Unmanaged is now a lighter solid line,
the smart path is teal, and bands use 0.3 opacity. Teal marks the smart
(selected) Axle action path everywhere, the same meaning as the teal UI
accent ("what Axle chose", decision 0004 item 22), so no other scale or
category colour uses teal (polish plan G4).
"""

from math import isfinite
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

BORDER = "#2A3140"
"""Muted border and gridline colour (polish plan G1, D10); matches config.toml's borderColor."""

SERIES_COLOURS = {
    "normal": "#9AA9BF",  # unmanaged charging, no Axle action (solid, decision 0004 item 27)
    "selected": "#2DD4BF",  # smart (selected) Axle action path (teal)
    "difference": "#F2B45A",  # smart minus unmanaged, paired per simulated week
    # Optional third "timed" path (decision 0007): reuses the difference
    # token rather than a new hex -- BAND_ALPHA above already proves this
    # exact value legible as a third line beside normal and selected -- and
    # it is never teal, which stays reserved for the smart path.
    "timed": "#F2B45A",
    "observed": "#E9EDF3",  # CNZ observed context value (dashed rule plus text label)
    "flag": "#F87171",  # energy-not-recovered week or EV (with FLAG_LABEL), or one_ev.py's
    # "left before plan finished" ✕ marker (with its own caption key, not FLAG_LABEL)
    "plugged": "#9AA9BF",  # plugged in at home, drawn at PLUGGED_ALPHA
    "synthetic": "#A78BFA",  # the Synthetic evidence badge
    "grid": BORDER,  # gridlines only; no plot border
}
TRANSPARENT = "rgba(0,0,0,0)"
"""Fully transparent: an outline-only fill, hidden slider ticks and labels."""
INK = "#E9EDF3"
MUTED_INK = "#9AA3B2"
SURFACE = "#171B23"
"""Raised surface: config.toml's secondaryBackgroundColor. Hover boxes sit on it so their
light text reads on the dark canvas (Plotly's default hover box follows the trace colour
and can come out white)."""
BAND_ALPHA = 0.55
"""Fill opacity for P10–P90 (across worlds) or P5–P95 (across EVs) bands.

Final critique B-21 (computed, not eyeballed): at the old 0.3, compositing
the three path colours ("normal" #9AA9BF, "selected" #2DD4BF, "difference"
#F2B45A) over the dark canvas #0E1117 gave 1.78-1.99:1 -- WCAG 1.4.11's 3:1
floor for a graphical object needed to understand the content, which a
P10-P90 band is (it carries the forecast's uncertainty, not decoration).
0.55 is the smallest of the usual half-hundredths above every path colour's
own worst case (#9AA9BF needs >= 0.523 for exactly 3:1; the accessibility
skill's ``contrast_check.py`` script bears this out). Raising the alpha,
not adding an outline: an outline on every P10/P90 edge would need a second
pass through ``band_and_line`` and every already-fainter secondary band
derived from this constant (an EV band, an outer P5-P95 tail) for no
extra clarity here, where the median line already carries the path's
identity at full colour (about 8-10:1) -- comfortably above the band's new
3.2-3.8:1, so band and line still read as two different things, just both
now legible on their own.
"""
PLUGGED_ALPHA = 0.6
"""Opacity for plugged-in bars and shaded connected spans.

Same fix as ``BAND_ALPHA`` (final critique B-21): 0.35 composited "plugged"
(#9AA9BF) over #0E1117 gave 1.99:1. Kept slightly above ``BAND_ALPHA``, as
before the fix, because a bar or a whole connected span is a bigger, bolder
fill than a band's edge.
"""
FLAG_LABEL = "⚠ not recovered"
"""Text that accompanies the flag colour, so a flagged mark reads without colour."""

# Six archetype (cohort) line colours (polish plan G4). Generated at equal
# OKLCH lightness (0.74) and low chroma (0.085), hues 20/70/120/245/290/335:
# equal lightness means no archetype looks more important than another, and
# the hue set skips teal (~180), which is reserved for the smart path. Plotly's
# default colourway, used before, mixed saturated and dull hues and included
# a teal-green.
ARCHETYPE_COLOURS = ("#DB9595", "#CEA26F", "#A4B376", "#7DB1DD", "#AAA2DD", "#CD97C0")

SEQUENTIAL = [
    [0.0, "#1B1E2E"],
    [0.2, "#2F3557"],
    [0.4, "#475087"],
    [0.6, "#6571B8"],
    [0.8, "#8C99DD"],
    [1.0, "#EAEEFC"],
]
"""Heatmap scale: one blue-violet hue from the canvas to near-white (polish plan
G4). It rises in OKLCH lightness only, so "more" always reads as "lighter", and
it never passes through teal, which would read as "smart charging did this"."""

RISK = [[0.0, SERIES_COLOURS["normal"]], [0.5, SERIES_COLOURS["difference"]], [1.0, "#F87171"]]
"""Risk scale for shortfall-style values: neutral grey, amber, then the flag red
only at the top (polish plan G4). A view sets the scale's top to the larger of
the largest value and the material threshold, so full red means at or above
material, never merely "the largest tiny shortfall this run drew"."""

# Line style per path. Width and dash differ as well as colour, so the paths
# stay distinguishable in greyscale. The standard centre line is the median
# (P50) with a P10–P90 band (decision 0004 item 27).
PATH_STYLES = {
    "normal": {"color": SERIES_COLOURS["normal"], "dash": "solid", "width": 1.5},
    "selected": {"color": SERIES_COLOURS["selected"], "dash": "solid", "width": 2.5},
    "difference": {"color": SERIES_COLOURS["difference"], "dash": "solid", "width": 2},
    # Dashed, not solid: stays distinguishable from normal and selected in
    # greyscale even though it shares the difference token's colour.  Step
    # ("hv"), not linear: home charging is barred or allowed for a whole
    # half-hour at a time (decision 0007), so the value genuinely holds
    # across the interval and the rise at the start time is a vertical
    # jump, not a slope drawn through the last barred half-hour.
    "timed": {"color": SERIES_COLOURS["timed"], "dash": "dash", "width": 2, "line_shape": "hv"},
}

PATH_LABELS = {"normal": "Unmanaged", "selected": "Smart", "timed": "Timed tariff"}
"""User-facing name for each internal ``path_id``. The model's own path ids stay
``"normal"``/``"selected"``/``"timed"`` (contract v2); every legend, title and
caption reads "Unmanaged"/"Smart"/"Timed tariff" (decision 0004 item 50:
"Baseline" is reserved for the trading settlement baseline). "timed" is the
optional third path (decision 0007), present when the "Timed tariff policy"
switch is on (the default, start 00:00), absent when off: most frames and
every pairwise view use only the first two names here. "Timed tariff" names the
rule after the common real-world pattern of a timed tariff (charging held off
until the start time); no tariff price is modelled and every path is costed
the same way, so the wording never claims otherwise."""

PATH_DISPLAY_ORDER = ("normal", "timed", "selected")
"""Legend/line order on screen: Unmanaged, Timed tariff, Smart (lead decision, 8 October
2026). Frames and ``PATH_ORDER`` (``model/forecast.py``) keep normal/selected/timed, the
order the kernel reports them in; this is the one place that reorders them for display."""


def present_paths(path_ids) -> list[str]:
    """``PATH_DISPLAY_ORDER`` filtered to the paths ``path_ids`` actually has.

    ``path_ids`` is anything iterable of path-id strings: a frame's
    ``path_id`` column, or a plain list.  A view's "normal versus selected"
    loop becomes ``for path in present_paths(frame["path_id"]):``, so a
    frame that also carries the optional timed path (decision 0007) draws
    it between the two it is compared against, in the lead-decided order,
    without each view re-deriving that order; a frame without it (the
    setting unset, or a strictly pairwise frame, model step 2 item 6) still
    draws exactly the two (or one) it has.
    """

    present = set(path_ids)
    return [path for path in PATH_DISPLAY_ORDER if path in present]


UNMANAGED_GLOSSARY = (
    "Unmanaged = charge at full power from plug-in to target; every moved and saving "
    "figure is measured against it."
)
"""The one glossary line that defines the counterfactual path (polish plan C3).
Headroom is not named: each path's headroom is measured against that path's
own draw, not against the unmanaged path."""

# Explicit heights (decision 0004 item 26; polish plan G2). Streamlit cannot
# read the viewport width, so one height serves 1440 and 390 px; without it
# Plotly's default 450 px squashes or clips charts. Lower than before because
# the bottom margin now only reserves what the chart actually has below it.
CHART_HEIGHTS = {
    "time_series": 340,
    "time_series_dual_axis": 360,  # Overview Average day, One EV
    "histogram": 280,
    "small_multiple_panel": 180,
    "strip": 240,  # Value and risk: one marker per simulated week
    "diagram": 260,
    # Replay page (replay contract v1 section 3): the running counters' text
    # row (four 12 px lines), and each animated panel, lower than
    # time_series so the prices, the now line and the slider fit one laptop
    # screen while playing.
    "replay_counter": 72,
    "replay_panel": 240,
}

DISPLAY_TIMEZONE = "Europe/London"
"""Axis labels use London time: the study is defined on London dates (0004 item 1)."""

CHART_FONT_PX = 12
"""Ticks, axis titles, legend and hover all use one size (polish plan G1): the
earlier 16 px axis titles were larger than the data they labelled."""

_COLORWAY = [SERIES_COLOURS[name] for name in ("selected", "normal", "difference")]

# The legend sits in the bottom margin, anchored to the figure (container)
# bottom rather than to a fraction of the plot height. A paper-fraction y
# moves with the plot height, which is why the old layout needed a fixed
# 104 px margin and still collided at 390 px; a container anchor stays put
# and ``_bottom_margin`` reserves exactly the rows the chart has (G2).
_LEGEND_BELOW = {
    "orientation": "h",
    "xref": "container",
    "yref": "container",
    "x": 0,
    "y": 0,
    "xanchor": "left",
    "yanchor": "bottom",
    "font": {"size": CHART_FONT_PX},
}
_CHART_MARGIN = {"l": 56, "r": 16, "t": 8, "b": 32}

# Row heights (px) for the content-based bottom margin (G2), measured in the
# browser on the 12 px chart font: a tick-label row, an x-axis title row and
# one legend row (a wrapped horizontal legend grows by about 29 px per row,
# padding included).
_TICK_ROW_PX = 24
_AXIS_TITLE_ROW_PX = 20
_LEGEND_ROW_PX = 28
_LEGEND_GAP_PX = 4
# The legend wraps at the narrowest width the app supports (a 390 px phone,
# less 16 px page gutters), so the reserved rows hold at 390 px. At 1440 px a
# legend that wraps only on a phone leaves its extra rows empty (one or more;
# Fleet week's seven entries leave three): the lesser evil next to a legend
# overlapping the tick labels at 390 px.
_NARROW_FIGURE_WIDTH_PX = 358
_LEGEND_SYMBOL_PX = 44  # symbol, its gap and the gap to the next entry
_LEGEND_CHAR_PX = 6.6  # average glyph width at 12 px sans-serif


def _axis_layout(title_colour: str, grid: str) -> dict[str, Any]:
    return {
        "gridcolor": grid,
        "zeroline": False,
        "automargin": True,
        "tickfont": {"size": CHART_FONT_PX},
        "title": {"font": {"color": title_colour, "size": CHART_FONT_PX}},
    }


AXLE_TEMPLATE = go.layout.Template(
    layout={
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": INK, "size": CHART_FONT_PX},
        "colorway": _COLORWAY,
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": SURFACE,
            "bordercolor": BORDER,
            "font": {"color": INK, "size": CHART_FONT_PX},
        },
        # Legend below the plot at every width, so a 390 px plot keeps its full
        # width and the layout never changes between widths (design 3.5).
        "legend": _LEGEND_BELOW,
        # Titles are Streamlit text above the chart, so the top margin is small;
        # style_figure sets the bottom margin from what sits below the plot.
        "margin": _CHART_MARGIN,
        "xaxis": _axis_layout(MUTED_INK, SERIES_COLOURS["grid"]),
        "yaxis": _axis_layout(MUTED_INK, SERIES_COLOURS["grid"]),
    }
)
pio.templates["axle"] = AXLE_TEMPLATE

LIGHT_BACKGROUND = "#FFFFFF"
"""Explicit white canvas. Unlike ``AXLE_TEMPLATE``'s transparent background
(which relies on the dark Streamlit shell showing through), a notebook has no
app shell behind it and can be viewed on GitHub or in either Jupyter theme,
so the background must be set explicitly, not inherited (chart audit B8)."""
LIGHT_INK = "#0E1117"
"""Reuses the app's own dark canvas colour (``.streamlit/config.toml``) as the
notebook's ink: the same value that anchors the dark theme gives strong
(18.9:1) contrast on white, so the light variant needs no new ink colour."""
LIGHT_MUTED_INK = "#5B6472"
"""De-emphasised axis-title ink for the light template, the same role as
``MUTED_INK`` on dark, at >=4.5:1 contrast on ``LIGHT_BACKGROUND``."""
LIGHT_GRID = "#D8DCE3"
"""Subtle gridline colour for a white canvas: the light-theme equivalent of
``SERIES_COLOURS["grid"]``, which is tuned for the near-black dark canvas and
would look like a border rather than a gridline on white."""

AXLE_LIGHT_TEMPLATE = go.layout.Template(
    layout={
        "paper_bgcolor": LIGHT_BACKGROUND,
        "plot_bgcolor": LIGHT_BACKGROUND,
        "font": {"color": LIGHT_INK, "size": CHART_FONT_PX},
        # Same series colourway as the app: the paths carry the same meaning
        # in a notebook, so the light variant only changes what makes them
        # readable on white (background, ink, gridlines).
        "colorway": _COLORWAY,
        "hovermode": "x unified",
        "hoverlabel": {"font": {"size": CHART_FONT_PX}},
        "legend": _LEGEND_BELOW,
        "margin": _CHART_MARGIN,
        "xaxis": _axis_layout(LIGHT_MUTED_INK, LIGHT_GRID),
        "yaxis": _axis_layout(LIGHT_MUTED_INK, LIGHT_GRID),
    }
)
pio.templates["axle_light"] = AXLE_LIGHT_TEMPLATE

PLOTLY_CONFIG = {"displayModeBar": False}
"""Passed to ``st.plotly_chart``: a hidden modebar cannot cover a title or legend."""

UNAVAILABLE = "Unavailable"

MINUS = "−"
"""U+2212, the typographic minus every displayed negative number uses (G6), so
"−£1,391" and "−12 kW" never mix with an ASCII hyphen."""

APP_CSS = f"""
<style>
/* Polish plan G1: one block for page padding. Streamlit's default is 6rem
   above the first element. The top navigation bar is fixed over the page, so
   the padding must clear it: 4.75rem (deliberate) is the bar plus roughly
   the plan's 1.25rem gap. Checked in the browser at 1440 and 390 px: the run
   bar starts just under the navigation with no overlap.
   Clarity critique layout item: on a ~2000 px monitor the uncapped
   container stretched KPI tiles roughly 450 px apart and widened tables and
   the strip chart past a readable line length. 1600px/centred caps that
   without moving anything at <=1440 px or 390 px, where content is already
   narrower than the cap and this rule has no effect. The run bar and every
   view render inside this same container, so they stay one aligned column;
   Streamlit's own top navigation is a separate fixed bar above it (the
   padding-top rule above), unaffected by this cap. */
[data-testid="stMainBlockContainer"] {{
  padding-top: 4.75rem; padding-bottom: 2rem;
  max-width: 1600px; margin-left: auto; margin-right: auto;
}}
/* Layout ruling 4 (30 Sep 2026): on a wide monitor the top navigation bar's
   own content (the wordmark logo, then the page links) rendered flush
   against the raw browser edge while the content column above capped
   itself to 1600 px and centred, leaving a growing, uneven gap between the
   two on anything wider than 1600 px -- most visibly the logo, which then
   read as clipped against the edge rather than simply left-aligned.
   Streamlit gives the toolbar no ``data-testid`` hook narrower than the
   whole bar, so this centres that whole row the same way as the content
   column, by the same rule, rather than reaching for its private,
   version-coupled inner padding (a number this project has no stable way
   to read and would drift on a Streamlit front-end update, the same
   DOM-hacking this file already declines for the drawer note below).
   That aligns the OUTER edges (checked in the browser: about 19 px apart
   at 2000 px, versus roughly 250 px before), not the logo's own inner
   margin against the column's own inner padding, which remains a known,
   smaller residual gap at desktop widths; at <=1600 px this rule has no
   effect (both are already unconstrained to the viewport). */
[data-testid="stToolbar"] {{
  max-width: 1600px; margin-left: auto; margin-right: auto;
}}
/* Polish plan G5: KPI tiles (components/kpi.py). */
.axle-kpi {{ line-height: 1.25; margin-bottom: 0.75rem; }}
.axle-kpi-label {{ font-size: 12px; color: {MUTED_INK}; }}
.axle-kpi-value {{
  font-size: 26px; font-weight: 600; color: {INK};
  font-variant-numeric: tabular-nums; white-space: nowrap;
}}
.axle-kpi-unit {{ font-size: 14px; font-weight: 400; color: {MUTED_INK}; margin-left: 0.25rem; }}
.axle-kpi-context {{ font-size: 12px; color: {MUTED_INK}; margin-top: 0.125rem; }}
</style>
"""
"""Injected once by ``streamlit_app.py``. Colours and fonts otherwise come from
config.toml (the Streamlit skill's rule); this covers only what the theme has
no setting for: the main padding and the KPI tile's type.

Known limitation, documented rather than patched (final critique B-17): at
390 px, ``st.navigation(pages, position="top")`` collapses to a drawer that
stays open (``aria-expanded`` true) after a page is chosen from it. Checked
against the developing-with-streamlit skill's ``multipage-apps.md`` and
``api-reference.md`` (2026-09-29): ``st.navigation``'s ``position`` takes
only ``"sidebar"``, ``"top"`` or ``"hidden"``, with no parameter or callback
for the drawer's open state, because that state lives in the client's own
JS, not in anything a Python rerun touches. Closing it would mean scripting
against that undocumented DOM from this CSS block, which breaks on any
Streamlit front-end update and is the DOM-hacking this project avoids for a
cosmetic fix (AGENTS.md: no custom machinery without a supported hook). A
real fix would replace the top nav with an in-page page selectbox at narrow
widths, which touches ``streamlit_app.py``, ``pages.py`` and ``registry.py``
(none owned by this task) and is tracked as open, not solved here."""


def _legend_names(figure: go.Figure) -> list[str]:
    """Names of the traces Plotly will show in the legend, in order.

    Mirrors Plotly's own rule: with ``layout.showlegend`` unset, a legend is
    drawn when two or more traces would be listed or one trace sets
    ``showlegend=True`` explicitly.
    """

    if figure.layout.showlegend is False:
        return []
    names = [
        str(trace.name)
        for trace in figure.data
        if trace.showlegend is not False and trace.visible is not False and trace.name
    ]
    explicit = any(trace.showlegend is True for trace in figure.data)
    if figure.layout.showlegend is None and len(names) < 2 and not explicit:
        return []
    return names


def _legend_rows(names: list[str]) -> int:
    """Rows a horizontal legend wraps to at the narrowest supported width."""

    rows, used = 0, _NARROW_FIGURE_WIDTH_PX
    for name in names:
        width = _LEGEND_SYMBOL_PX + _LEGEND_CHAR_PX * len(name)
        if used + width > _NARROW_FIGURE_WIDTH_PX:
            rows, used = rows + 1, 0
        used += width
    return rows


def _bottom_margin(figure: go.Figure) -> int:
    """Bottom margin (px) that holds exactly what sits below the plot area (G2).

    Tick labels, an x-axis title if any axis has one, and the legend rows,
    replacing the old fixed 72/104 px floor that left plots at 60-65% of
    their height even with no legend.
    """

    layout = figure.layout.to_plotly_json()
    axes = [value for key, value in layout.items() if key.startswith("xaxis")] or [{}]
    has_ticks = any(axis.get("visible", True) and axis.get("showticklabels", True) for axis in axes)
    has_title = any((axis.get("title") or {}).get("text") for axis in axes)
    rows = _legend_rows(_legend_names(figure))
    margin = (_TICK_ROW_PX if has_ticks else 4) + (_AXIS_TITLE_ROW_PX if has_title else 0)
    if rows:
        margin += _LEGEND_GAP_PX + rows * _LEGEND_ROW_PX
    return margin


def _apply(figure: go.Figure, *, template: str, background: str, height: int) -> go.Figure:
    # A view's own explicit bottom margin wins (a small-multiple grid with an
    # annotation title below it, for example); otherwise size it to content.
    bottom = figure.layout.margin.b
    figure.update_layout(
        template=template,
        height=height,
        # Explicit, not only in the template: Streamlit's plotly_chart writes
        # its own paper/plot colours unless the figure sets them (polish audit,
        # D10 flat charts).
        paper_bgcolor=background,
        plot_bgcolor=background,
        margin={"b": bottom if bottom is not None else _bottom_margin(figure)},
    )
    return figure


def style_figure(figure: go.Figure, *, height: int) -> go.Figure:
    """Apply the shared dark app template, an explicit height and a fitted bottom margin.

    ``height`` is the whole figure in px (``CHART_HEIGHTS``). Values the view
    set explicitly (axis titles, a dual-axis right margin, its own bottom
    margin) take precedence over the template defaults.
    """

    return _apply(figure, template="axle", background="rgba(0,0,0,0)", height=height)


def apply_notebook_style(figure: go.Figure, *, height: int) -> go.Figure:
    """Apply the light notebook template, an explicit height and a fitted bottom margin.

    Notebooks are read on GitHub or in a light-mode Jupyter/VS Code canvas,
    not the dark Streamlit shell ``style_figure`` targets, so this applies
    ``axle_light`` (white background, dark ink, same series colours) instead
    of ``axle`` (chart audit B8). Values the caller set explicitly take
    precedence over the template defaults, exactly as ``style_figure``.
    """

    return _apply(figure, template="axle_light", background=LIGHT_BACKGROUND, height=height)


def band_fill(colour: str, alpha: float = BAND_ALPHA) -> str:
    """Return a ``#RRGGBB`` token as the ``rgba(...)`` string Plotly fills use."""

    red, green, blue = (int(colour[index : index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def band_and_line(
    figure: go.Figure,
    x: Any,
    low: Any,
    mid: Any,
    high: Any,
    *,
    name: str,
    colour: str,
    width: float = 2.0,
    dash: str = "solid",
    line_shape: str = "linear",
    band_alpha: float = BAND_ALPHA,
    legendgroup: str | None = None,
    showlegend: bool = True,
    hovertemplate: str | None = None,
    customdata: Any = None,
    band_hovertemplates: tuple[str, str] | None = None,
    yaxis: str | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Add a P10–P90 band and its median line as one legend entry (polish plan G3).

    ``low``/``high`` are the band bounds and ``mid`` the centre line, all in
    the chart's y unit; ``name`` is the one legend label ("Unmanaged",
    "Smart"). The band traces share the line's ``legendgroup`` and set
    ``showlegend=False``, so clicking the entry hides band and line together
    and the legend lists paths, not statistics; the caption names the spread.
    Bands have no hover unless ``band_hovertemplates`` gives (low, high)
    templates. Pass ``row``/``col`` for a subplot and ``yaxis`` ("y2") for a
    secondary axis. ``line_shape`` ("linear" by default) is Plotly's
    interpolation between points; a caller whose value genuinely holds across
    each interval until a sharp change (the "timed" path's barred-window
    rule, decision 0007) passes ``"hv"`` so the rise reads as vertical at the
    change, not a slope drawn through the last barred half-hour -- applied to
    the band as well as the line, so the two still read as one shape. Returns
    ``figure`` for chaining.
    """

    group = legendgroup or name
    shared: dict[str, Any] = {"x": x, "mode": "lines", "legendgroup": group}
    if yaxis is not None:
        shared["yaxis"] = yaxis
    place = {"row": row, "col": col} if row is not None else {}
    for index, (bound, fill) in enumerate(((low, None), (high, "tonexty"))):
        hover: dict[str, Any] = {"hoverinfo": "skip"}
        if band_hovertemplates is not None:
            hover = {"customdata": customdata, "hovertemplate": band_hovertemplates[index]}
        figure.add_trace(
            go.Scatter(
                y=bound,
                line={"color": colour, "width": 0, "shape": line_shape},
                fill=fill,
                fillcolor=band_fill(colour, band_alpha) if fill else None,
                name=name,
                showlegend=False,
                **shared,
                **hover,
            ),
            **place,
        )
    figure.add_trace(
        go.Scatter(
            y=mid,
            line={"color": colour, "width": width, "dash": dash, "shape": line_shape},
            name=name,
            showlegend=showlegend,
            customdata=customdata,
            hovertemplate=hovertemplate,
            **shared,
        ),
        **place,
    )
    return figure


def london_time_axis(
    utc_values: pd.Series, *, every_hours: int | None = None, compact: bool = False
) -> dict[str, list]:
    """Return ``xaxis`` tick positions and London-time labels for a UTC series.

    The x values stay in UTC, so a week with a clock change still plots in
    order (the repeated autumn hour never folds back). Plotly cannot read the
    viewport, so one tick density must read at both 1440 and 390 px. Default:
    one tick per London calendar day ("Mon 28"), which is exactly the seven
    ticks a week-long study needs at either width -- a denser interval (one
    tick every few hours) would give a full week up to 28 ticks that overlap
    at 390 px. Intraday time stays out of the axis and only appears in hover
    text (``hover_time_labels``). Pass ``every_hours`` for a caller that
    wants the denser, compact intraday variant instead (no current caller
    needs it).

    ``compact=True`` drops the day-of-month ("Mon" instead of "Mon 28"):
    a one-week study never repeats a weekday, so the number adds no
    disambiguation, only width. A dual-axis chart's narrower plot area left
    as little as 2 px between adjacent day labels at 390 px (goal review
    V3); the default keeps the fuller label for callers happy with the
    extra margin, so this stays opt-in rather than a global change.
    Use as ``figure.update_xaxes(tickmode="array", **london_time_axis(starts))``.
    """

    local = pd.DatetimeIndex(utc_values).tz_convert(DISPLAY_TIMEZONE)
    if every_hours is None:
        ticks = pd.date_range(local.min().floor("D"), local.max().floor("D"), freq="D")
        labels = (
            [f"{tick:%a}" for tick in ticks]
            if compact
            else [f"{tick:%a} {tick.day}" for tick in ticks]
        )
    else:
        hours = pd.date_range(local.min().floor("h"), local.max(), freq="h")
        ticks = hours[hours.hour % every_hours == 0]
        labels = [f"{tick:%a} {tick.day}" if tick.hour == 0 else f"{tick:%H:%M}" for tick in ticks]
    return {
        "tickvals": list(ticks.tz_convert("UTC")),
        "ticktext": labels,
        "tickangle": 0,
        "automargin": True,
    }


def hover_time_labels(utc_values: pd.Series) -> list[str]:
    """Return hover text such as ``Mon 28 18:30 (17:30 UTC)`` for each timestamp.

    London time leads because axes use it; UTC stays visible because the
    model keys every interval in UTC (decision 0004 item 27).
    """

    utc = pd.DatetimeIndex(utc_values).tz_convert("UTC")
    local = utc.tz_convert(DISPLAY_TIMEZONE)
    return [
        f"{london:%a} {london.day} {london:%H:%M} ({moment:%H:%M} UTC)"
        for london, moment in zip(local, utc, strict=True)
    ]


def _missing(value: float | None) -> bool:
    return value is None or not isfinite(value)


def _number(value: float, decimals: int, *, signed: bool = False) -> str:
    """``1,234.5`` with a U+2212 minus; ``signed`` adds "+" to positives.

    Rounds first so a value that displays as zero never carries a sign
    ("−0" or "+0").
    """

    rounded = round(float(value), decimals)
    if rounded == 0:
        rounded = 0.0
    sign = MINUS if rounded < 0 else ("+" if signed and rounded > 0 else "")
    return f"{sign}{abs(rounded):,.{decimals}f}"


def format_quantity(value: float | None, unit: str, *, decimals: int = 1) -> str:
    """Format a displayed quantity with its unit; missing values never look like zero.

    Negative values use the U+2212 minus and "%" sits against the number
    ("82%", G6); other units are separated by a space.
    """

    if _missing(value):
        return UNAVAILABLE
    number = _number(value, decimals)
    if unit == "%":
        return f"{number}%"
    return f"{number} {unit}".strip()


def money(value: float | None, *, decimals: int = 0, signed: bool = False) -> str:
    """Format pounds as ``£1,391`` or ``−£1,250`` (U+2212 before the £, G6).

    ``decimals`` is 0 for fleet-scale totals and 2 for per-EV amounts, where
    pence matter. ``signed=True`` prefixes "+" to positive values (a change).
    Missing values return ``UNAVAILABLE``.
    """

    if _missing(value):
        return UNAVAILABLE
    number = _number(value, decimals, signed=signed)
    sign = number[0] if number[0] in (MINUS, "+") else ""
    return f"{sign}£{number.lstrip(MINUS + '+')}"


def percent(value: float | None, *, decimals: int = 0, signed: bool = False) -> str:
    """Format a value already in percentage points as ``82%`` (no space, G6)."""

    if _missing(value):
        return UNAVAILABLE
    return f"{_number(value, decimals, signed=signed)}%"


def kw(value: float | None, *, signed: bool = False) -> str:
    """Format fleet power as ``1,234 kW`` (whole kW, G6)."""

    if _missing(value):
        return UNAVAILABLE
    return f"{_number(value, 0, signed=signed)} kW"


def assumption_value(result: Any, name: str) -> Any | None:
    """Return the value of one named record in ``result.assumptions``, or ``None``.

    A view reads the run's own assumption this way rather than a hard-coded
    constant, so an edited or future assumption value never goes stale on
    one page alone (contract v2 section 8.1). A result without assumption
    records, or without a record of that name, returns ``None`` rather than
    raising -- not every result carries every assumption.
    """

    for record in result.assumptions or ():
        if record.name == name:
            return record.value
    return None
