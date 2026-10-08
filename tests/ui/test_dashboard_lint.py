"""Every figure the views draw follows the chart conventions, bar the recorded exceptions.

``scripts/dashboard_lint.py`` renders each view on the SYNTHETIC FIXTURE
result (both models, every control option) and lints the figures handed to
``st.plotly_chart``. ``ACCEPTED`` lists the findings the design accepts on
purpose, each with its reason and authority; the test fails on any finding
outside that list and on any listed exception that no longer occurs, so the
list cannot go stale. Fixing a listed item means deleting its entry here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest
from fixtures.result_fixture import compare_runs, make_result

from axle_studio.ui.views import compare as compare_module

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from dashboard_lint import (  # noqa: E402
    ChartFindings,
    Finding,
    RecordingStreamlit,
    fixture_views,
    lint_figure,
    lint_views,
    normalise_colour,
)

# (view, chart key, rule) -> reason. The chart key is the ``key=`` the view
# gives ``chart_block``; the finding text is not matched, so a fix that
# changes the wording without removing the exception still fails here.
ACCEPTED: dict[tuple[str, str, str], str] = {
    ("Overview ▸ At a glance", "overview-average-day", "secondary-axis"): (
        "Design 3.5 and decision 0004 item 26 accept this dual-axis chart (plugged-in "
        "share plus SoC); revisit under the built-in dataviz one-axis rule."
    ),
    ("Smart charging ▸ 1 Plan", "action-decision-price", "secondary-axis"): (
        "Design 4.3: forecast price and the chosen slots share one time axis on two "
        "scales; same accepted dual-axis exception as the Overview chart."
    ),
    ("Drivers ▸ Plug-ins", "plug-ins-heatmap-Plug-in", "y-axis-title"): (
        "Half-hour by day heatmap: the y axis is the London day (categorical tick "
        "labels) and the colourbar carries the unit (%)."
    ),
    ("Drivers ▸ Plug-ins", "plug-ins-heatmap-Departure", "y-axis-title"): (
        "Same heatmap for departures; the day axis is categorical."
    ),
    ("Drivers ▸ Fleet week", "fleet-week-chart-Plugged in", "line-without-band"): (
        "Decision 0004 item 54: the six archetype lines are context (world medians) "
        "drawn thin around the fleet's own P10–P90 band; a band each would be unreadable."
    ),
    ("Drivers ▸ Household", "household-outcomes", "y-axis-title"): (
        "Horizontal P10–P90 range rows (This EV, Archetype, Fleet): the y axis is the three "
        "category rows and each panel title carries its unit (£, £/kWh, %)."
    ),
    ("Drivers ▸ One EV", "one-ev-chart", "teal-reserved"): (
        "Departure and public top-up markers take the colour of the path they belong "
        "to, so the smart path's markers are teal on purpose; the trace name is the "
        "event, not the path."
    ),
    ("Trading ▸ 1 Market", "trading-market-price", "line-without-band"): (
        "The intraday close and imbalance price are the representative simulated "
        "week's own realised path, not a forecast fan across weeks (the day-ahead "
        "line on the same chart is the P10-P90 band); trading contract v1 section 1.4."
    ),
    ("Trading ▸ 2 Position", "trading-position-night-chart", "line-without-band"): (
        "Baseline, unmanaged and metered are the representative simulated week's own "
        "physical trajectory for one chosen night, not a statistic across weeks; the "
        "across-week spread of the same settled volume is the separate "
        "trading-position-settled band chart on this lens."
    ),
    ("Trading ▸ 2 Position", "trading-position-fan", "secondary-axis"): (
        "Position (kW) and the intraday close (£/MWh) are unrelated scales on one time "
        "axis, the same accepted dual-axis exception as the Overview and Plan charts "
        "(decision 0004 item 43)."
    ),
    ("Trading ▸ 2 Position", "trading-position-fan", "line-without-band"): (
        "The day-ahead position, final position and intraday close are all the "
        "representative simulated week's own per-slot decisions, not a statistic "
        "across weeks; same reasoning as the night-profile chart above."
    ),
    ("Trading ▸ 2 Position", "trading-position-settled", "teal-reserved"): (
        "Settled turn-down is the smart-charging outcome itself (the delivered "
        "flexibility, identical for every trading strategy), so teal is the smart "
        "path's own colour here; the trace name is the physical quantity, not a "
        "strategy label, the same pattern as the One EV exception above."
    ),
    ("Replay ▸ Fleet", "replay-fleet-chart", "line-without-band"): (
        "Replay contract v1 §7 and decision 0004 item 66: the replay plays back one "
        "simulated week by design (what was known and done at each instant), so its lines "
        "are that week's values; the other lenses carry the across-weeks spread."
    ),
    ("Replay ▸ Customer", "replay-customer-chart", "line-without-band"): (
        "Same as Replay ▸ Fleet: one driver in one simulated week (replay contract v1 §7)."
    ),
    ("Replay ▸ Customer", "replay-customer-chart", "teal-reserved"): (
        "'Plan in force' is the smart charger's own plan for this EV (the smart path's "
        "plan from now on), so it takes the smart path's teal, as an outline, on purpose "
        "(replay contract v1 §3.4)."
    ),
    ("Partners ▸ Driver", "partners-persona-driver-average-day", "secondary-axis"): (
        "Reuses overview._average_day_figure unchanged; same accepted dual-axis exception "
        "as Overview ▸ At a glance (plugged-in share plus SoC)."
    ),
    ("Partners ▸ Carmaker", "partners-persona-carmaker-heatmap", "y-axis-title"): (
        "Reuses plug_ins._heatmap_figure unchanged; same accepted exception as Drivers ▸ "
        "Plug-ins: the y axis is the London day (categorical tick labels) and the "
        "colourbar carries the unit (%)."
    ),
    ("Partners ▸ Fleet and leasing", "partners-persona-fleet-outcomes", "y-axis-title"): (
        "Reuses household._outcomes_figure unchanged; same accepted exception as Drivers ▸ "
        "Household: horizontal P10-P90 range rows, the y axis is the three category rows "
        "and each panel title carries its unit."
    ),
}


def _key(item: ChartFindings, finding: Finding) -> tuple[str, str, str]:
    return (item.view, item.chart or "-", finding.rule)


@pytest.fixture(scope="module")
def results() -> list[ChartFindings]:
    found: list[ChartFindings] = []
    for model in ("action", "no_action"):
        found.extend(lint_views(fixture_views(), make_result(model, evs=12, worlds=6, seed=42)))
    return found


def test_every_view_figure_follows_the_conventions_or_is_accepted(results) -> None:
    unexpected = [
        f"{item.view} | {item.choice} | {item.chart} | {finding}"
        for item in results
        for finding in item.findings
        if _key(item, finding) not in ACCEPTED
    ]
    assert unexpected == []


def test_no_accepted_exception_is_stale(results) -> None:
    seen = {_key(item, finding) for item in results for finding in item.findings}
    stale = sorted(set(ACCEPTED) - seen)
    assert stale == [], "delete these entries from ACCEPTED: the finding no longer occurs"


def test_compare_figures_follow_the_conventions(monkeypatch) -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=7.0)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)

    class _Slim:
        def __init__(self, label, backing):
            self.label, self.number, self._backing = (
                label,
                int(re.search(r"\d+", label)[0]),
                backing,
            )

        def __getattr__(self, name):
            if name == "result":
                raise AttributeError("slim run records do not carry a full result")
            return getattr(self._backing, name)

    records = [_Slim("Run 2 · base", a), _Slim("Run 3 · home charger 11 kW", b)]
    monkeypatch.setattr(compare_module, "run_history", lambda state: records)
    st = RecordingStreamlit()

    compare_module.render_compare(st, None, compare_runs=compare_runs)

    figures = st.figures()
    assert figures, "Compare drew no chart"
    assert [str(f) for _, figure in figures for f in lint_figure(figure)] == []


# --- The linter itself ----------------------------------------------------


def test_the_rules_catch_what_they_describe() -> None:
    x = ["2026-09-28T18:00:00", "2026-09-28T18:30:00"]
    bad = go.Figure(
        [
            go.Pie(values=[1, 2], name="mix"),
            go.Scatter(x=x, y=[1, 2], mode="lines", name="Unmanaged"),
            go.Scatter(x=x, y=[2, 3], mode="lines", line={"color": "#FF00FF"}),
            go.Scatter(x=x, y=[3, 4], mode="lines", name="Other", line={"color": "#2DD4BF"}),
        ]
    )
    bad.update_layout(showlegend=False, yaxis2={"overlaying": "y", "side": "right"})
    for trace in bad.data:
        trace.hoverinfo = "skip"
    rules = {finding.rule for finding in lint_figure(bad)}
    assert rules == {
        "pie",
        "secondary-axis",
        "height",
        "y-axis-title",
        "colour-token",
        "legend",
        "trace-name",
        "line-without-band",
        "teal-reserved",
        "hover",
    }

    good = go.Figure(
        [
            go.Scatter(
                x=x,
                y=[0, 1],
                mode="lines",
                name="Smart",
                legendgroup="smart",
                line={"color": "#2DD4BF", "width": 0},
                showlegend=False,
            ),
            go.Scatter(
                x=x,
                y=[2, 3],
                mode="lines",
                name="Smart",
                legendgroup="smart",
                fill="tonexty",
                fillcolor="rgba(45,212,191,0.3)",
                line={"color": "#2DD4BF", "width": 0},
                showlegend=False,
            ),
            go.Scatter(
                x=x,
                y=[1, 2],
                mode="lines",
                name="Smart",
                legendgroup="smart",
                line={"color": "#2DD4BF"},
            ),
        ]
    )
    good.update_layout(height=340, yaxis={"title": "kW"})
    assert lint_figure(good) == []


def test_axis_unit_rule_accepts_percentage_points_and_rejects_bare_words() -> None:
    x = ["2026-09-28T18:00:00", "2026-09-28T18:30:00"]

    def figure(title: str) -> go.Figure:
        fig = go.Figure(go.Bar(x=x, y=[1, 2], name="a"))
        fig.update_layout(height=200, yaxis={"title": title})
        return fig

    assert {f.rule for f in lint_figure(figure("percentage points"))} == set()
    assert {f.rule for f in lint_figure(figure("Battery SoC (%)"))} == set()
    assert {f.rule for f in lint_figure(figure("Value"))} == {"y-axis-unit"}


def test_colour_normalisation_maps_rgba_to_the_token_hue() -> None:
    assert normalise_colour("rgba(45,212,191,0.3)") == "#2DD4BF"
    assert normalise_colour("#2dd4bf") == "#2DD4BF"
    assert normalise_colour("#fff") == "#FFFFFF"
    assert normalise_colour("rgba(0,0,0,0)") is None
    assert normalise_colour("white") == "named:white"
