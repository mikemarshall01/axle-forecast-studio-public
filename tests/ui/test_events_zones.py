"""Scripted events and zones in the dashboard (decision 0004 items 55, 56, 58).

Covers the Events group of Edit assumptions and its draft (views/parameters.py,
run_controller.py), the scripted-shock marks on the Plan price chart
(views/action_decision.py), the Events metric on Smart charging ▸ Response
(views/action_response.py) and the Zone import metric on Drivers ▸ Fleet week
(views/drivers_fleet.py), against docs/contracts/trading-events-v1.md §2, §5.6
to §5.8 and §6.

View tests read a real small run with every preset selected (30 EVs x 4
simulated weeks, about 2 s), because the contract v2 fixture has no scripted
events; the zone headroom line uses the fixture, whose Zone A has one.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fixtures.result_fixture import make_result
from streamlit.testing.v1 import AppTest
from test_copy_rules import _texts, _violations

from axle_studio.model import assumptions
from axle_studio.model.summaries import compare_runs
from axle_studio.ui import run_controller as rc
from axle_studio.ui.registry import PAGES
from axle_studio.ui.style import CHART_HEIGHTS, INK
from axle_studio.ui.views import parameters
from axle_studio.ui.views.action_decision import (
    SHOCK_KIND_LABELS,
    _price_bands,
    _price_figure,
    _publication_times,
    _selected_home_import,
    scripted_shock_windows,
)
from axle_studio.ui.views.action_response import (
    EVENTS_METRIC,
    render_action_response,
    render_event_response,
)
from axle_studio.ui.views.drivers_fleet import (
    _zone_band_table,
    render_fleet_week,
    zone_summary_table,
)

TODAY = date(2026, 9, 29)
APP_PATH = Path(__file__).parents[2] / "streamlit_app.py"
SMALL = {"vehicle_count": 30, "evaluation_world_count": 4}


@pytest.fixture(scope="module")
def events_result():
    """A real small smart charging run with every event preset (illustrative)."""

    values = assumptions.editable_defaults() | SMALL
    return rc.run_model(rc.MODEL_ACTION, values, TODAY, rc.EVENT_PRESET_IDS)


@pytest.fixture(scope="module")
def timed_zone_result():
    """A real small run with the optional timed path on (decision 0007, model step
    2): the shared contract v2 fixture stays two-policy (``fixtures.result_fixture``),
    so the zone chart's third path needs an actual model run to exercise."""

    values = assumptions.editable_defaults() | SMALL | {"timed_start_local_hour": 0.0}
    return rc.run_model(rc.MODEL_ACTION, values, TODAY)


class RecordingStreamlit:
    """Fake ``st``: records calls; selectbox/segmented_control return a chosen option."""

    def __init__(self, calls: list | None = None, choices: dict | None = None) -> None:
        self.calls: list[tuple[str, tuple, dict]] = calls if calls is not None else []
        self.choices = choices or {}

    def __getattr__(self, name: str):
        def record(*args: object, **kwargs: object) -> object:
            self.calls.append((name, args, kwargs))
            if name in ("selectbox", "segmented_control"):
                options = list(kwargs.get("options", []))
                chosen = self.choices.get(kwargs.get("key"))
                return chosen if chosen in options else (options[0] if options else None)
            return self

        return record

    def __enter__(self) -> "RecordingStreamlit":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def columns(self, spec: int | list, **kwargs: object) -> tuple["RecordingStreamlit", ...]:
        count = spec if isinstance(spec, int) else len(spec)
        self.calls.append(("columns", (spec,), kwargs))
        return tuple(RecordingStreamlit(self.calls, self.choices) for _ in range(count))


def calls_for(st: RecordingStreamlit, name: str) -> list[tuple[tuple, dict]]:
    return [(args, kwargs) for call_name, args, kwargs in st.calls if call_name == name]


def figures_from(st: RecordingStreamlit) -> list:
    return [args[0] for args, _ in calls_for(st, "plotly_chart")]


def _fresh_state() -> dict:
    state: dict = {}
    rc.initialise_session(state, today=TODAY)
    return state


def _tick(state: dict, preset_id: str, ticked: bool = True) -> None:
    """What an Events checkbox does: write its key, then call its callback."""

    state[rc.EVENT_WIDGET_PREFIX + preset_id] = ticked
    rc.commit_event_preset(state, preset_id)


def _recording_runner(calls: list):
    def runner(model, values, start, event_presets=()):
        calls.append(event_presets)
        values = assumptions.editable_defaults() | dict(values) | SMALL
        return rc.run_model(model, values, start, event_presets)

    return runner


# --- Draft, run label, identity and Compare (run_controller.py) ---------------


def test_presets_are_part_of_the_draft_in_preset_order_and_ticking_never_runs() -> None:
    state = _fresh_state()
    assert state["draft_event_presets"] == ()
    _tick(state, "local_turn_up")
    _tick(state, "cold_still_evening")

    # Stored in EVENT_PRESETS order whatever the click order, so the same
    # choice always compares equal.
    assert state["draft_event_presets"] == ("cold_still_evening", "local_turn_up")
    assert state["draft_error"] is None
    assert state["run_log"].history == []  # a tick never runs the model
    _tick(state, "local_turn_up", ticked=False)
    assert state["draft_event_presets"] == ("cold_still_evening",)


def test_run_passes_presets_and_the_draft_goes_stale_when_they_change() -> None:
    state = _fresh_state()
    _tick(state, "dfs_turn_down")
    calls: list = []
    rc.run_full(state, _recording_runner(calls))

    assert calls == [("dfs_turn_down",)]
    record = rc.latest_run(state)
    assert record.event_presets == ("dfs_turn_down",)
    assert list(record.result.events["event_id"]) == ["dfs_turn_down"]
    assert rc.run_status(state).state == "current"
    _tick(state, "cold_still_week")
    assert "events" in rc.unrun_changes(state)
    assert rc.run_status(state).state == "stale"

    rc.reset_to_active_run(state)
    assert state["draft_event_presets"] == ("dfs_turn_down",)
    assert state[rc.EVENT_WIDGET_PREFIX + "cold_still_week"] is False
    rc.reset_to_defaults(state)
    assert state["draft_event_presets"] == ()


def test_label_identity_and_compare_change_line_name_the_events() -> None:
    state = _fresh_state()
    rc.run_full(state, _recording_runner([]))
    _tick(state, "cold_still_evening")
    _tick(state, "surprise_evening_spike")
    rc.run_full(state, _recording_runner([]))

    first, second = rc.run_history(state)
    assert first.label == "Run 1 · base"
    assert second.label == "Run 2 · events none → 2 presets"
    identity = rc.result_identity(second)
    assert identity.endswith("· 2 event presets") and len(identity) <= 80
    assert "event" not in rc.result_identity(first)

    changed = compare_runs(first, second).changed.set_index("name")
    row = changed.loc["events.presets"]
    assert row["label"] == "Scripted events"
    assert row["value_a"] == "none"
    assert row["value_b"] == "Cold still evening (known day-ahead); Surprise evening spike"
    # Only the kept record carries it: the result's own assumptions are unchanged.
    assert "events.presets" not in {record.name for record in second.result.assumptions}


def test_a_no_action_run_never_claims_events_it_did_not_simulate() -> None:
    state = _fresh_state()
    state["model_mode"] = rc.MODEL_NO_ACTION
    _tick(state, "dfs_turn_down")
    rc.run_full(state, _recording_runner([]))

    record = rc.latest_run(state)
    assert record.result.events is None
    assert "events" not in record.label
    assert "event" not in rc.result_identity(record)
    assert record.assumptions[-1].value == "none"
    # The choice stays in the draft for the next smart charging run.
    assert state["draft_event_presets"] == ("dfs_turn_down",)


def test_an_invalid_event_selection_blocks_run(monkeypatch) -> None:
    # The fixed presets validate together; the model's own rules still own
    # the check (contract §2.3), so a rejection must block Run like any
    # invalid value.
    assert rc.event_presets_error(rc.EVENT_PRESET_IDS, TODAY) is None
    assert "unknown event preset" in rc.event_presets_error(("no_such_event",), TODAY)
    monkeypatch.setattr(rc, "event_presets_error", lambda presets, start: "requests overlap")
    state = _fresh_state()
    _tick(state, "dfs_turn_down")

    assert state["draft_error"] == "Events: requests overlap"
    calls: list = []
    rc.run_full(state, _recording_runner(calls))
    assert calls == []


# --- Events group of Edit assumptions (parameters.py) --------------------------


@pytest.mark.parametrize(
    ("preset_id", "expected"),
    [
        ("cold_still_evening", "Price shock known a day ahead, 4 GW of extra demand; Thu 1 Oct"),
        ("surprise_evening_spike", "known 90 min ahead; Sat 3 Oct at 17:00 for 2 h"),
        ("dfs_turn_down", "paid £500/MWh (illustrative); Wed 30 Sep at 17:30 for 1 h"),
        ("local_turn_up", "in Zone B only"),
        ("cold_still_week", "every evening at 16:30 for 3 h"),
        ("sunny_negative_weekend", "6 GW of surplus supply; Saturday and Sunday at 10:00"),
        ("charger_control_outage", "for 50% of sessions plugging in, with no warning"),
    ],
)
def test_each_preset_has_a_one_line_description_from_its_own_values(preset_id, expected) -> None:
    text = parameters.preset_description(preset_id, TODAY)
    assert expected in text
    assert "_" not in text and len(text) <= 140


def test_events_pill_ticks_a_preset_into_the_draft_without_running() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.get_by_key("edit-assumptions").click().run()
    app.get_by_key(parameters.GROUP_KEY).set_value(parameters.EVENTS_GROUP).run()
    app.get_by_key("edit-assumptions").click().run()
    assert not app.exception

    boxes = {box.key for box in app.checkbox}
    assert boxes == {rc.EVENT_WIDGET_PREFIX + preset for preset in rc.EVENT_PRESET_IDS}
    app.get_by_key(rc.EVENT_WIDGET_PREFIX + "local_turn_up").check().run()
    assert not app.exception
    assert app.session_state["draft_event_presets"] == ("local_turn_up",)
    assert rc.run_history(app.session_state) == []


# --- Plan price chart: scripted shock marks (action_decision.py) ---------------


def test_plan_chart_shades_each_scripted_shock_known_filled_surprise_outlined(
    events_result,
) -> None:
    windows = scripted_shock_windows(events_result)
    scripted = events_result.market_shocks.query("source == 'scripted' and world_id == 0")
    assert len(windows) == len(scripted) == 11  # one per scripted shock, not per week
    assert windows["known_day_ahead"].sum() == 10  # only the surprise spike is not known
    assert "Cold still week, Thu 1 Oct" in set(windows["label"])

    prices = _price_bands(events_result)
    figure = _price_figure(
        prices,
        _selected_home_import(events_result.fleet_interval_bands),
        _publication_times(events_result, prices),
        windows,
    )
    spans = [shape for shape in figure.layout.shapes if shape.type == "rect"]
    assert len(spans) == 11
    outlined = [shape for shape in spans if shape.line.dash == "dash"]
    assert len(outlined) == 1  # the surprise: an outline, not a fill
    legend = [trace.name for trace in figure.data if trace.showlegend is not False]
    assert SHOCK_KIND_LABELS[True] in legend and SHOCK_KIND_LABELS[False] in legend
    # Regression (browser check): the legend keys come before the series, so
    # an x of None made Plotly type the axis linear and every shape NaN.
    keys = [t for t in figure.data if t.name in SHOCK_KIND_LABELS.values()]
    assert all(isinstance(t.x[0], pd.Timestamp) for t in keys)


def test_shock_arrows_sit_at_the_bottom_clear_of_the_publication_label(events_result) -> None:
    # Review of ebec3bb: a night-0 shock (every cold still week) put its
    # arrow on top of "Next day's prices published", which is anchored at
    # the plot's top edge. Arrows go to the bottom edge instead (decision
    # 0004 item 26: annotations must not overlap).
    prices = _price_bands(events_result)
    figure = _price_figure(
        prices,
        _selected_home_import(events_result.fleet_interval_bands),
        _publication_times(events_result, prices),
        scripted_shock_windows(events_result),
    )
    arrows = [a for a in figure.layout.annotations if a.text in ("▲", "▼")]
    (label,) = [a for a in figure.layout.annotations if a.text == "Next day's prices published"]
    assert (label.y, label.yanchor) == (1, "top")
    assert all((a.y, a.yanchor) == (0, "bottom") for a in arrows)
    # No two same-direction arrows closer than 2 h: the Saturday surprise
    # starts 30 min into that night's cold still evening, so only one of
    # the two gets an arrow (both spans are still drawn: 11 rectangles).
    for glyph in ("▲", "▼"):
        starts = sorted(pd.Timestamp(a.x) for a in arrows if a.text == glyph)
        assert all(b - a >= pd.Timedelta(hours=2) for a, b in zip(starts, starts[1:]))
    # 7 up (one per cold still evening; the cold still evening preset and the
    # surprise share those evenings) and 2 down (the sunny Saturday and Sunday).
    assert len(arrows) == 9
    assert len([s for s in figure.layout.shapes if s.type == "rect"]) == 11


def test_plan_chart_marks_no_random_shock() -> None:
    # The fixture's shocks are all stochastic: they differ between weeks, so
    # a chart summarised across weeks marks none of them.
    result = make_result("action")
    assert scripted_shock_windows(result).empty
    st = RecordingStreamlit()
    from axle_studio.ui.views.action_decision import render_action_decision

    render_action_decision(st, result)
    figure = figures_from(st)[0]
    assert not [shape for shape in figure.layout.shapes if shape.type == "rect"]


# --- Response ▸ Events (action_response.py) -----------------------------------


def test_events_metric_without_events_says_how_to_add_them() -> None:
    st = RecordingStreamlit(choices={"action-response-metric": EVENTS_METRIC})
    render_action_response(st, make_result("action"))
    assert "Edit assumptions, Events" in calls_for(st, "info")[0][0][0]
    assert figures_from(st) == []


def test_events_view_lists_every_event_and_plots_its_bands(events_result) -> None:
    st = RecordingStreamlit(choices={"action-response-event": "dfs_turn_down"})
    render_event_response(st, events_result, "Evidence: illustrative.")

    table = calls_for(st, "dataframe")[0][0][0]
    assert len(table) == len(events_result.events)
    assert "_" not in "".join(table["Event"])
    selector = next(k for _, k in calls_for(st, "selectbox") if k["key"] == "action-response-event")
    assert selector["options"] == list(events_result.events["event_id"])

    bands = events_result.event_response_bands
    rows = bands.loc[bands["event_id"].eq("dfs_turn_down")]
    import_figure, difference_figure = figures_from(st)  # a request has no price chart
    # The trading overlay adds baseline_kw and delivered_kw rows to the selected
    # series; the Smart line is the home import metric only.
    smart = rows.loc[
        rows["series"].eq("selected") & rows["metric"].eq("home_import_kw")
    ].sort_values("slot_index")
    smart_line = next(t for t in import_figure.data if t.name == "Smart" and t.showlegend)
    assert list(smart_line.y) == pytest.approx(list(smart["p50"]))
    diff = rows.loc[rows["series"].eq("difference")].sort_values("slot_index")
    band_top = [t for t in difference_figure.data if t.fill == "tonexty"][0]
    assert list(band_top.y) == pytest.approx(list(diff["p90"]))
    assert difference_figure.layout.yaxis.zeroline is True
    # A request window is two dotted rules, never a shock span.
    rules = [s for s in import_figure.layout.shapes if s.type == "line"]
    assert len(rules) == 2
    captions = " ".join(args[0] for args, _ in calls_for(st, "caption"))
    # The trading overlay now supplies the settlement baseline for a request, so
    # the caption names the baseline line rather than promising it later.
    assert rows["metric"].eq("baseline_kw").any()
    assert "Dotted line: the settlement baseline" in captions


def test_event_tiles_look_up_the_largest_median_change_either_side(events_result) -> None:
    st = RecordingStreamlit(choices={"action-response-event": "dfs_turn_down"})
    render_event_response(st, events_result, "")

    bands = events_result.event_response_bands
    diff = bands.loc[
        bands["event_id"].eq("dfs_turn_down") & bands["series"].eq("difference")
    ].reset_index(drop=True)
    inside = diff.loc[diff["in_window"]]
    expected = inside.loc[inside["p50"].abs().idxmax(), "p50"]
    tiles = [args[0] for args, _ in calls_for(st, "markdown") if "axle-kpi" in str(args[0])]
    assert len(tiles) == 3
    assert "In the window" in tiles[1]
    assert f"{abs(round(expected)):,}" in tiles[1]


def test_price_shock_event_adds_its_own_price_increment(events_result) -> None:
    st = RecordingStreamlit(choices={"action-response-event": "surprise_evening_spike"})
    render_event_response(st, events_result, "")

    figures = figures_from(st)
    assert len(figures) == 3
    bands = events_result.event_response_bands
    increment = bands.loc[
        bands["event_id"].eq("surprise_evening_spike")
        & bands["metric"].eq("price_increment_gbp_per_mwh")
    ].sort_values("slot_index")
    name = "Price moved by this shock"
    line = next(t for t in figures[2].data if t.name == name and t.showlegend)
    assert list(line.y) == pytest.approx(list(increment["p50"]))
    # The shock's own window is a dashed outline (a surprise), as on Plan.
    assert any(s.type == "rect" and s.line.dash == "dash" for s in figures[0].layout.shapes)


def test_a_zone_event_is_titled_by_its_zone(events_result) -> None:
    st = RecordingStreamlit(choices={"action-response-event": "local_turn_up"})
    render_event_response(st, events_result, "")
    titles = [args[0] for args, _ in calls_for(st, "markdown")]
    assert any(title.startswith("**Zone B home import around the event") for title in titles)


def test_event_charts_use_the_time_series_height(events_result) -> None:
    st = RecordingStreamlit(choices={"action-response-event": "cold_still_evening"})
    render_event_response(st, events_result, "")
    for figure in figures_from(st):
        assert figure.layout.height == CHART_HEIGHTS["time_series"]


# --- Fleet week ▸ Zone import (drivers_fleet.py) --------------------------------


def _zone_st(zone_id: str = "zone_1") -> RecordingStreamlit:
    return RecordingStreamlit(
        choices={"fleet-week-metric": "Zone import", "fleet-week-zone": zone_id}
    )


def test_zone_import_plots_both_paths_and_the_headroom_line() -> None:
    result = make_result("action", evs=6, worlds=6)
    st = _zone_st("zone_1")
    render_fleet_week(st, result)

    (figure,) = figures_from(st)
    medians = [t for t in figure.data if t.showlegend is not False]
    assert [t.name for t in medians] == ["Unmanaged", "Smart"]
    rows = result.zone_import_bands.query("zone_id == 'zone_1' and path_id == 'selected'")
    assert list(medians[1].y) == pytest.approx(list(rows.sort_values("slot_index")["p50"]))
    (rule,) = [s for s in figure.layout.shapes if s.type == "line"]
    assert rule.y0 == 5.0 and rule.line.color == INK  # the fixture's Zone A headroom
    group = next(k for _, k in calls_for(st, "selectbox") if k["key"] != "fleet-week-metric")
    assert group["disabled"] is True  # zones are fleet-wide


def test_zone_without_headroom_draws_no_line_and_says_so() -> None:
    st = _zone_st("zone_2")
    render_fleet_week(st, make_result("no_action", evs=6, worlds=6))
    (figure,) = figures_from(st)
    assert not [s for s in figure.layout.shapes if s.type == "line"]
    assert [t.name for t in figure.data if t.showlegend is not False] == ["Unmanaged"]
    captions = " ".join(args[0] for args, _ in calls_for(st, "caption"))
    assert "No headroom set for this zone." in captions


def test_zone_summary_table_reads_zone_summary_unchanged() -> None:
    summary = make_result("action", evs=6, worlds=6).zone_summary
    table = zone_summary_table(summary)
    assert len(table) == len(summary)
    first = table.iloc[0]
    source = summary.query("zone_id == 'zone_1' and path_id == 'normal'").iloc[0]
    assert (first["Zone"], first["Path"]) == ("Zone A", "Unmanaged")
    assert first["Weekly peak P50 (kW)"] == f"{source['peak_kw_p50']:,.0f}"
    assert table.iloc[2]["Headroom (kW)"] == "None set"
    assert table.iloc[2]["Hours above headroom P50"] == "No headroom set"


# --- Zone import with the optional timed path on (decision 0007, model step 2) --


def test_zone_import_plots_the_timed_path_between_unmanaged_and_smart(timed_zone_result) -> None:
    st = _zone_st("zone_1")
    render_fleet_week(st, timed_zone_result)

    (figure,) = figures_from(st)
    medians = [t for t in figure.data if t.showlegend is not False]
    assert [t.name for t in medians] == ["Unmanaged", "Timed tariff", "Smart"]
    timed = medians[1]
    # Decision 0007: amber and dashed (never the teal smart colour), drawn as a
    # step, because charging is held off or allowed for a whole half-hour at a
    # time, so the rise at the start time is a vertical jump, not a slope.
    assert timed.line.dash == "dash"
    assert timed.line.shape == "hv"
    rows = timed_zone_result.zone_import_bands.query("zone_id == 'zone_1' and path_id == 'timed'")
    assert list(timed.y) == pytest.approx(list(rows.sort_values("slot_index")["p50"]))


def test_zone_band_table_orders_rows_unmanaged_timed_smart(timed_zone_result) -> None:
    table = _zone_band_table(timed_zone_result, "zone_1")
    per_path = table["Path"].eq("Unmanaged").sum()
    assert per_path > 0
    assert list(table["Path"]) == (
        ["Unmanaged"] * per_path + ["Timed tariff"] * per_path + ["Smart"] * per_path
    )


def test_zone_summary_table_includes_the_timed_row_in_display_order(timed_zone_result) -> None:
    table = zone_summary_table(timed_zone_result.zone_summary)
    zone_a = table.loc[table["Zone"].eq("Zone A")]
    assert list(zone_a["Path"]) == ["Unmanaged", "Timed tariff", "Smart"]


# --- Copy rules on the rendered app, every preset selected ---------------------


def test_events_and_zones_views_follow_the_copy_rules() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
    state = app.session_state
    values = dict(state["draft_values"]) | SMALL
    state["draft_values"] = values
    for preset in rc.EVENT_PRESET_IDS:
        state[rc.EVENT_WIDGET_PREFIX + preset] = True
    state["draft_event_presets"] = rc.EVENT_PRESET_IDS
    app.run()
    app.get_by_key("run-simulation").click().run()
    assert not app.exception
    assert rc.latest_run(state).event_presets == rc.EVENT_PRESET_IDS

    found = []
    for page_name, lens, key, option in (
        ("Smart charging", "2 Response", "action-response-metric", EVENTS_METRIC),
        ("Smart charging", "1 Plan", None, None),
        ("Drivers", "Fleet week", "fleet-week-metric", "Zone import"),
    ):
        app._page_hash = next(
            page_hash
            for page_hash, page in app._registered_pages.items()
            if page["page_name"] == page_name
        )
        app.run()
        slug = next(page.slug for page in PAGES if page.name == page_name)
        app.get_by_key(f"lens::{slug}").set_value(lens).run()
        if key is not None:
            app.get_by_key(key).set_value(option).run()
        assert not app.exception
        found += [f"{page_name} {lens}: {v}" for v in _violations(list(_texts(app.main)))]
    assert found == [], "\n".join(found)
