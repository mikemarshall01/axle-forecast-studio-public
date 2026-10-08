"""Lint the Plotly figures the views build against the chart conventions.

The rules come from docs/DASHBOARD_DESIGN.md section 3.5 and "Chart
conventions", decision 0004 items 26, 27 and 50, ``ui/style.py`` and the
Chartability heuristics (Elavsky, Bennett and Moritz 2022) where a rule can
be read off the figure object. Each finding names a rule id, so a test can
allowlist an accepted exception by id and chart rather than switching a rule
off. What cannot be read off the figure (annotation overlap, rendered text
size, contrast of an actual screenshot) belongs to the browser pass in
``docs/VERIFY_APP.md`` and the dashboard-review skill, not here.

Usage from the repository root (renders every view on the SYNTHETIC
FIXTURE result and prints findings, exit 1 when any):

    uv run --frozen --group dev env PYTHONPATH=src:tests python scripts/dashboard_lint.py

``tests/ui/test_dashboard_lint.py`` runs the same walk in pytest with the
accepted exceptions recorded.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go

from axle_studio.ui import style

# --- Tokens ------------------------------------------------------------------

THEME_COLOURS = {
    # .streamlit/config.toml, the Streamlit theme the charts sit on. A view may
    # use the canvas colour (at any opacity) to knock text out of a busy plot.
    "canvas": "#0E1117",
    "surface": "#171B23",
    "sidebar": "#131720",
    "primary": "#0F766E",
    "link": "#6EE7D0",
}

TOKEN_COLOURS: frozenset[str] = frozenset(
    colour.upper()
    for colour in (
        *style.SERIES_COLOURS.values(),
        *style.ARCHETYPE_COLOURS,
        *(stop[1] for stop in style.SEQUENTIAL),
        *(stop[1] for stop in style.RISK),
        style.INK,
        style.MUTED_INK,
        style.BORDER,
        style.LIGHT_BACKGROUND,
        style.LIGHT_INK,
        style.LIGHT_MUTED_INK,
        style.LIGHT_GRID,
        *THEME_COLOURS.values(),
    )
)
"""Every colour a chart may use, as upper-case ``#RRGGBB``. ``rgba`` forms of
these (any alpha) are accepted, so band fills and knock-out boxes pass."""

_HEX = re.compile(r"^#([0-9a-fA-F]{6})$")
_HEX_SHORT = re.compile(r"^#([0-9a-fA-F]{3})$")
_RGB = re.compile(r"^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([0-9.]+)\s*)?\)$")

UNIT_PATTERN = re.compile(
    r"kWh|kW|MWh|\bMW\b|£/MWh|£|%|\bp/kWh\b|\bhours?\b|\bminutes?\b|\bEVs?\b|\bweeks?\b|\bdays?\b"
    r"|\bcount\b|\bplug-ins\b|\bsessions\b|\bshare\b|\bslots?\b|\bmiles\b|\bevents\b"
    r"|\bpercentage points\b|\bpts\b",
    re.IGNORECASE,
)
"""A y-axis title must carry a unit (design: "Every chart has a y-axis title
with unit"). Dimensionless shares are allowed to say "%" or "share"."""

PIE_TYPES = frozenset({"pie", "funnelarea", "sunburst"})
LEGEND_TYPES = frozenset({"scatter", "scattergl", "bar", "box", "violin", "histogram"})
"""Trace types Plotly lists in the legend, so an unnamed one shows as "trace N"."""

SELECTED_WORDS = ("smart", "selected", "chosen", "axle")
"""A teal trace must be about the smart (selected) Axle action path (style.py:
teal is reserved for "what Axle chose")."""


@dataclass(frozen=True, slots=True)
class Finding:
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


# --- Colour helpers ---------------------------------------------------------


def normalise_colour(value: str) -> str | None:
    """Return ``#RRGGBB`` upper-case for a hex or rgb(a) string, or ``None``.

    Alpha is dropped: opacity is a style choice (``BAND_ALPHA``), the hue is
    what the token governs. Fully transparent ``rgba(0,0,0,0)`` returns
    ``None`` because it draws nothing.
    """

    text = value.strip()
    if match := _HEX.match(text):
        return f"#{match.group(1).upper()}"
    if match := _HEX_SHORT.match(text):
        return "#" + "".join(ch * 2 for ch in match.group(1)).upper()
    if match := _RGB.match(text):
        red, green, blue, alpha = match.groups()
        if alpha is not None and float(alpha) == 0.0:
            return None
        return f"#{int(red):02X}{int(green):02X}{int(blue):02X}"
    return f"named:{text}"


def _colour_values(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Yield ``(path, colour)`` for every colour-carrying key in a figure JSON tree."""

    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in ("colorscale", "colorway") and isinstance(value, list):
                for entry in value:
                    colour = entry[1] if isinstance(entry, list) else entry
                    if isinstance(colour, str):
                        yield here, colour
            elif key.endswith("color") and isinstance(value, str):
                yield here, value
            elif key.endswith("color") and isinstance(value, list):
                for colour in value:
                    if isinstance(colour, str):
                        yield here, colour
            else:
                yield from _colour_values(value, here)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _colour_values(value, f"{path}[{index}]")


def off_token_colours(figure: go.Figure) -> list[tuple[str, str]]:
    """Colours in ``figure`` (traces, layout, shapes, annotations) that are not tokens."""

    spec = figure.to_plotly_json()
    layout = dict(spec.get("layout", {}))
    # The template is style.py's own; scanning it would only repeat the tokens.
    layout.pop("template", None)
    found = []
    for path, raw in _colour_values({"data": spec.get("data", []), "layout": layout}):
        colour = normalise_colour(raw)
        if colour is not None and colour not in TOKEN_COLOURS:
            found.append((path, raw))
    return found


# --- Figure rules -----------------------------------------------------------


def _axis_id(trace: Any, letter: str) -> str:
    value = getattr(trace, f"{letter}axis", None) or letter
    return letter + "axis" + value[1:]  # "y" -> "yaxis", "y2" -> "yaxis2"


def _has_hover(trace: Any) -> bool:
    # Plotly shows x/y hover by default; only an explicit skip/none removes it.
    return getattr(trace, "hoverinfo", None) not in ("skip", "none")


def _is_band(trace: Any) -> bool:
    return isinstance(trace, go.Scatter) and trace.fill not in (None, "none")


def _is_line(trace: Any) -> bool:
    if not isinstance(trace, go.Scatter) or _is_band(trace):
        return False
    mode = trace.mode or "lines"
    return "lines" in mode and (trace.line.width is None or trace.line.width > 0)


def _looks_like_time(values: Any) -> bool:
    if values is None:
        return False
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        return True
    if array.size == 0:
        return False
    first = array.flat[0]
    return hasattr(first, "year") or (isinstance(first, str) and "T" in first and "-" in first)


def _trace_colour(trace: Any) -> str | None:
    for candidate in (
        getattr(getattr(trace, "line", None), "color", None),
        getattr(getattr(trace, "marker", None), "color", None),
        getattr(trace, "fillcolor", None),
    ):
        if isinstance(candidate, str):
            return normalise_colour(candidate)
    return None


def lint_figure(figure: go.Figure) -> list[Finding]:
    """Return every convention the styled ``figure`` breaks, empty when none."""

    findings: list[Finding] = []
    traces = list(figure.data)
    layout = figure.layout

    for trace in traces:
        if trace.type in PIE_TYPES:
            findings.append(Finding("pie", f"{trace.type} trace {trace.name!r}: no part-to-whole"))

    for name in layout:
        if name.startswith("yaxis") and getattr(layout[name], "overlaying", None):
            findings.append(Finding("secondary-axis", f"{name} overlays another y-axis"))

    if layout.height is None:
        findings.append(Finding("height", "no explicit height (decision 0004 item 26)"))

    used_y = {_axis_id(trace, "y") for trace in traces if hasattr(trace, "yaxis")}
    if len(used_y) >= 3:
        # A small-multiple grid titles one axis per row or column, not each panel.
        titled = [
            name
            for name in used_y
            if name in layout and layout[name].title is not None and layout[name].title.text
        ]
        used_y = set() if titled else {sorted(used_y)[0]}
    for name in sorted(used_y):
        axis = layout[name] if name in layout else None
        if axis is None:
            findings.append(Finding("y-axis-title", f"{name} has no title"))
            continue
        if axis.visible is False or axis.showticklabels is False or axis.matches:
            continue  # a shared inner axis of a small-multiple grid
        title = (axis.title.text or "") if axis.title is not None else ""
        if not title.strip():
            findings.append(Finding("y-axis-title", f"{name} has no title"))
        elif not UNIT_PATTERN.search(title):
            findings.append(Finding("y-axis-unit", f"{name} title {title!r} names no unit"))

    for path, raw in off_token_colours(figure):
        findings.append(Finding("colour-token", f"{path} uses {raw!r}, not a style.py token"))

    legend_names = [
        str(trace.name)
        for trace in traces
        if trace.showlegend is not False and trace.visible is not False and trace.name
    ]
    if len(set(legend_names)) >= 2 and layout.showlegend is False:
        findings.append(Finding("legend", f"{len(set(legend_names))} named series, legend off"))
    for index, trace in enumerate(traces):
        if trace.type in LEGEND_TYPES and trace.showlegend is not False and not trace.name:
            findings.append(Finding("trace-name", f"trace {index} ({trace.type}) has no name"))

    bands_by_group = {
        trace.legendgroup or trace.name for trace in traces if _is_band(trace) and trace.name
    }
    for trace in traces:
        if not _is_line(trace) or not _looks_like_time(trace.x):
            continue
        group = trace.legendgroup or trace.name
        if group not in bands_by_group:
            findings.append(
                Finding("line-without-band", f"time series {trace.name!r} has no P10–P90 band")
            )

    teal = style.SERIES_COLOURS["selected"].upper()
    for trace in traces:
        if _trace_colour(trace) != teal:
            continue
        label = f"{trace.name or ''} {trace.legendgroup or ''}".lower()
        if not any(word in label for word in SELECTED_WORDS):
            findings.append(
                Finding("teal-reserved", f"teal on {trace.name!r}, not a smart-path series")
            )

    if traces and not any(_has_hover(trace) for trace in traces if not _is_band(trace)):
        findings.append(Finding("hover", "no trace offers hover text (Chartability: details)"))

    return findings


# --- Walking the views on the fixture --------------------------------------


class RecordingStreamlit:
    """A fake ``st`` that records every call and hands back defaults.

    Widgets return their default (``choices`` overrides by widget key), so a
    view renders the way it first opens. Columns, tabs, containers and
    expanders share this one log, so a chart drawn into a column is recorded
    like any other. ``plotly_chart`` calls are what the lint reads.
    """

    def __init__(self, *, choices: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.session_state: dict[str, Any] = {}
        self.controls: dict[str, list[Any]] = {}
        self._choices = choices or {}

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def record(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            key = kwargs.get("key")
            options = kwargs.get("options")
            if options is not None and key is not None and name != "dataframe":
                self.controls.setdefault(key, list(options))
            if key in self._choices:
                return self._choices[key]
            if (
                key is not None
                and key in self.session_state
                and name not in ("button", "dataframe")
            ):
                # Real Streamlit: a widget's value is whatever the view wrote
                # under its key before drawing it (Compare's A/B default).
                return self.session_state[key]
            if name == "columns":
                spec = args[0] if args else kwargs.get("spec", 1)
                count = spec if isinstance(spec, int) else len(spec)
                return [self for _ in range(count)]
            if name == "tabs":
                return [self for _ in args[0]]
            if name in ("selectbox", "radio"):
                index = kwargs.get("index") or 0
                return list(options)[index] if options else None
            if name in ("segmented_control", "pills"):
                default = kwargs.get("default")
                if default is not None:
                    return default
                return (
                    list(options)[0]
                    if options and kwargs.get("selection_mode") != "multi"
                    else None
                )
            if name in ("checkbox", "toggle"):
                return kwargs.get("value", False)
            if name in ("button", "form_submit_button"):
                return False
            if name in ("number_input", "slider"):
                return kwargs.get("value", 0)
            if name in ("expander", "container", "form", "popover", "dialog"):
                return self
            return None

        return record

    def __enter__(self) -> RecordingStreamlit:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def figures(self) -> list[tuple[str | None, go.Figure]]:
        """``(chart key, figure)`` for every ``plotly_chart`` call, in render order."""

        return [
            (kwargs.get("key"), args[0])
            for name, args, kwargs in self.calls
            if name == "plotly_chart" and args and isinstance(args[0], go.Figure)
        ]


@dataclass(frozen=True, slots=True)
class ChartFindings:
    view: str
    choice: str
    chart: str | None
    findings: tuple[Finding, ...]


def _render(view: Any, result: Any, choices: dict[str, Any] | None = None) -> RecordingStreamlit:
    st = RecordingStreamlit(choices=choices)
    view(st, result)
    return st


def lint_views(
    views: dict[tuple[str, str | None], Any], result: Any, *, expand_controls: bool = True
) -> list[ChartFindings]:
    """Render each view on ``result`` and lint every figure it draws.

    With ``expand_controls`` each single-choice control the default render
    recorded (metric, group, day type, EV) is re-rendered once per other
    option, so a lens with a Metric selector is linted for every metric.
    """

    out: list[ChartFindings] = []
    for (page, lens), view in views.items():
        name = f"{page} ▸ {lens}" if lens else page
        try:
            st = _render(view, result)
        except Exception as error:  # a view that needs more than ``result`` is skipped, loudly
            out.append(
                ChartFindings(
                    name, "default", None, (Finding("render", f"{type(error).__name__}: {error}"),)
                )
            )
            continue
        renders: list[tuple[str, RecordingStreamlit]] = [("default", st)]
        if expand_controls:
            for key, options in st.controls.items():
                for option in list(options)[1:12]:
                    try:
                        renders.append((f"{key}={option}", _render(view, result, {key: option})))
                    except Exception as error:
                        out.append(
                            ChartFindings(
                                name,
                                f"{key}={option}",
                                None,
                                (Finding("render", f"{type(error).__name__}: {error}"),),
                            )
                        )
        for choice, rendered in renders:
            for chart_key, figure in rendered.figures():
                findings = tuple(lint_figure(figure))
                if findings:
                    out.append(ChartFindings(name, choice, chart_key, findings))
    return out


def format_report(results: Iterable[ChartFindings]) -> str:
    lines = []
    for item in results:
        for finding in item.findings:
            lines.append(f"{item.view} | {item.choice} | {item.chart or '-'} | {finding}")
    return "\n".join(lines)


def fixture_views() -> dict[tuple[str, str | None], Any]:
    """``pages.VIEWS`` with the one-EV views and Household on the fixture, Compare left out.

    One EV and Replay ▸ Customer call the model's one-EV replay, which needs
    the real result's kernel state, so both are bound to the fixture's own
    replay (and Replay ▸ Customer to the fixture's timeline). Compare reads
    the run history rather than ``result``; the pytest wrapper supplies that
    history and lints it there.
    """

    from functools import partial

    from fixtures.household_contract import make_household_card
    from fixtures.replay_contract import replay_one_ev_timeline
    from fixtures.result_fixture import replay_one_ev, replay_one_ev_bands

    from axle_studio.ui import pages
    from axle_studio.ui.views import household, one_ev, partner_personas, replay

    views = {key: view for key, view in pages.VIEWS.items() if key[0] != "Compare"}
    views[("Drivers", "One EV")] = partial(
        one_ev.render_one_ev, replay_one_ev=replay_one_ev, replay_one_ev_bands=replay_one_ev_bands
    )
    # The fixture's card is built from its toy arrays (no kernel run); the
    # page's no-action guard stays in place around it.
    views[("Drivers", "Household")] = pages._action_lens(
        "household", partial(household.render_household, household_card=make_household_card)
    )
    # Partners ▸ Fleet and leasing takes the same injected household_card as
    # Drivers ▸ Household above, for the same reason (its hero chart is the
    # same one-EV card, on this page for a chosen vehicle rather than driver).
    views[("Partners", "Fleet and leasing")] = pages._action_lens(
        "partners-fleet-and-leasing",
        partial(partner_personas.render_fleet_and_leasing, household_card=make_household_card),
    )
    views[("Replay", "Customer")] = pages._action_lens(
        "replay-customer",
        partial(
            replay.render_replay_customer,
            replay_one_ev=replay_one_ev,
            replay_one_ev_timeline=replay_one_ev_timeline,
        ),
    )
    return views


def main() -> int:
    from fixtures.result_fixture import make_result

    views = fixture_views()
    results = []
    for model in ("action", "no_action"):
        results.extend(lint_views(views, make_result(model, evs=12, worlds=6, seed=42)))
    report = format_report(results)
    print(report or "dashboard_lint: no findings")
    return 1 if report else 0


if __name__ == "__main__":
    sys.exit(main())
