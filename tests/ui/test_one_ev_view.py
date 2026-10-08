"""Drivers ▸ One EV renders sketch 1 (plugged-in shading + SoC line) from the fixture."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fixtures.result_contract import Assumption
from fixtures.result_fixture import make_result, replay_one_ev, replay_one_ev_bands

from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.model.individual import replay_one_ev as model_replay_one_ev
from axle_studio.model.individual import replay_one_ev_bands as model_replay_one_ev_bands
from axle_studio.ui.style import CHART_HEIGHTS, PATH_STYLES
from axle_studio.ui.views.one_ev import (
    _daily_audit_table,
    _dispatch_caption,
    _plug_events_table,
    _present_paths,
    render_one_ev,
)


class _Container:
    def __enter__(self) -> "_Container":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class RecordingStreamlit:
    """Fake ``st`` module; widgets return their default/index unless overridden by key."""

    def __init__(self, *, choices: dict[str, object] | None = None) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self._choices = choices or {}

    def __getattr__(self, name: str):
        def record(*args: object, **kwargs: object) -> object:
            self.calls.append((name, args, kwargs))
            key = kwargs.get("key")
            if key in self._choices:
                return self._choices[key]
            if name == "selectbox":
                options = kwargs["options"]
                index = kwargs.get("index") or 0
                return options[index]
            if name == "checkbox":
                return kwargs.get("value", False)
            # Columns, tabs and expanders share this fake's call log, so a
            # control drawn into a column is recorded like any other call.
            if name == "columns":
                count = args[0] if args else kwargs.get("spec")
                n = count if isinstance(count, int) else len(count)
                return [self for _ in range(n)]
            if name == "tabs":
                return [self for _ in args[0]]
            if name == "expander":
                return _Container()
            return None

        return record

    def __enter__(self) -> "RecordingStreamlit":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def calls_for(st: RecordingStreamlit, name: str) -> list[tuple[tuple, dict]]:
    return [(args, kwargs) for call_name, args, kwargs in st.calls if call_name == name]


def _render(st: RecordingStreamlit, result) -> None:
    render_one_ev(st, result, replay_one_ev=replay_one_ev, replay_one_ev_bands=replay_one_ev_bands)


@pytest.mark.parametrize("model", ["action", "no_action"])
def test_renders_both_models_without_model_calls(model: str) -> None:
    result = make_result(model, evs=6, worlds=12)
    st = RecordingStreamlit()

    _render(st, result)

    assert len(calls_for(st, "plotly_chart")) == 1
    assert not calls_for(st, "button")


def test_five_ev_case_with_fewer_than_six_cohorts_renders() -> None:
    result = make_result("no_action", evs=5, worlds=8)
    st = RecordingStreamlit()

    _render(st, result)  # must not raise

    assert len(calls_for(st, "plotly_chart")) == 1


@pytest.mark.parametrize("clock_change", ["autumn", "spring"])
def test_clock_change_week_renders(clock_change: str) -> None:
    result = make_result("action", evs=6, worlds=6, clock_change=clock_change)
    st = RecordingStreamlit()

    _render(st, result)

    assert len(calls_for(st, "plotly_chart")) == 1


def test_chart_height_and_legend_below() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert figure.layout.height == CHART_HEIGHTS["time_series_dual_axis"]
    # Polish plan G2: legend anchored to the figure's own bottom edge.
    assert figure.layout.template.layout.legend.yref == "container"


def test_london_axis_labels() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert figure.layout.xaxis.ticktext


def test_defaults_to_representative_world() -> None:
    result = make_result("no_action", evs=6, worlds=12)
    st = RecordingStreamlit()

    _render(st, result)

    world_selectbox = next(
        kwargs
        for name, _, kwargs in st.calls
        if name == "selectbox" and kwargs.get("key") == "one-ev-world"
    )
    assert world_selectbox["index"] == result.representative_world_id
    chart_caption = calls_for(st, "caption")[1][0][0]
    assert "median week" in chart_caption.lower()


def test_default_ev_is_the_first_average_uk_ev() -> None:
    # Goal review item 11: default to a typical commuter, not whichever EV
    # sorts first. The fixture's own cohort cycling (evs % 6) happens to put
    # "average_uk" at unit_id "ev-0000" anyway, so this alone would not
    # distinguish the new rule from "just pick index 0" -- see the next
    # test for that.
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    unit_selectbox = next(
        kwargs
        for name, _, kwargs in st.calls
        if name == "selectbox" and kwargs.get("key") == "one-ev-unit"
    )
    expected = result.units.loc[result.units["cohort_id"].eq("average_uk"), "unit_id"].iat[0]
    assert unit_selectbox["options"][unit_selectbox["index"]] == expected


def test_default_ev_follows_cohort_not_row_position() -> None:
    # Proves the rule reads cohort_id, not "row 0": relabel which two EVs
    # carry the "average_uk" cohort (unit_id order and every underlying
    # replay array stay keyed by the original row position, so this only
    # changes which EV *counts* as Average (UK) for the default, not the
    # data any replay of either EV would show).
    result = make_result("no_action", evs=6, worlds=6)
    units = result.units.copy()
    units.loc[units["unit_id"].eq("ev-0000"), ["cohort_id", "cohort_label"]] = [
        "intelligent_octopus",
        "Intelligent Octopus average",
    ]
    units.loc[units["unit_id"].eq("ev-0001"), ["cohort_id", "cohort_label"]] = [
        "average_uk",
        "Average (UK)",
    ]
    result = replace(result, units=units)
    st = RecordingStreamlit()

    _render(st, result)

    unit_selectbox = next(
        kwargs
        for name, _, kwargs in st.calls
        if name == "selectbox" and kwargs.get("key") == "one-ev-unit"
    )
    assert unit_selectbox["options"][unit_selectbox["index"]] == "ev-0001"


def test_default_ev_falls_back_to_the_first_ev_without_an_average_uk_sample() -> None:
    # Chart audit B1: a small run can miss a cohort entirely.
    result = make_result("no_action", evs=6, worlds=6)
    units = result.units.copy()
    units.loc[units["cohort_id"].eq("average_uk"), ["cohort_id", "cohort_label"]] = [
        "intelligent_octopus",
        "Intelligent Octopus average",
    ]
    result = replace(result, units=units)
    st = RecordingStreamlit()

    _render(st, result)

    unit_selectbox = next(
        kwargs
        for name, _, kwargs in st.calls
        if name == "selectbox" and kwargs.get("key") == "one-ev-unit"
    )
    assert unit_selectbox["options"][unit_selectbox["index"]] == "ev-0000"


def test_action_result_shows_normal_and_smart_soc_lines() -> None:
    # Goal review item 6: "smart", not the model's internal "selected".
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    names = {trace.name for trace in figure.data}
    assert "SoC (unmanaged)" in names
    assert "SoC (smart)" in names


def test_no_action_result_shows_one_unlabelled_path_soc_line() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    names = {trace.name for trace in figure.data}
    assert "SoC" in names
    assert "SoC (unmanaged)" not in names


def test_plugged_in_shading_and_plug_departure_markers_present() -> None:
    result = make_result("no_action", evs=6, worlds=12)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    names = [trace.name for trace in figure.data]
    assert "Plugged in at home" in names
    assert "Departure" in names


def test_plug_in_events_table_count_matches_normal_path_events() -> None:
    result = make_result("no_action", evs=6, worlds=6)
    st = RecordingStreamlit()

    _render(st, result)

    replay = replay_one_ev(result, "ev-0000", result.representative_world_id)
    expected = len(replay.plug_events.loc[replay.plug_events["path_id"].eq("normal")])
    # One expander with tabs (polish plan G9): the count is on the tab.
    tab_labels = [label for args, _ in calls_for(st, "tabs") for label in args[0]]
    assert f"Plug-in events ({expected})" in tab_labels


def test_daily_readiness_audit_never_says_stranded() -> None:
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    tables = [args[0] for args, _ in calls_for(st, "dataframe")]
    audit_table = tables[-1]
    assert "Status" in audit_table.columns
    assert not audit_table["Status"].str.contains("stranded", case=False).any()
    assert not any("stranded" in str(call).lower() for call in calls_for(st, "caption"))


def test_daily_audit_status_reserves_the_warning_icon_for_a_real_shortfall() -> None:
    # Goal review item 11: departing below the preferred *target* SoC is not
    # itself evidence of a problem -- the target is a preference, not the
    # physical ceiling (AGENTS.md), and item 32's public top-ups mean a real
    # shortfall is rare. Only shortfall_kwh > 0 (this date's trip energy
    # actually not served) earns the warning icon now; "below target" alone
    # is a plain, unflagged note.
    daily_audit = pd.DataFrame(
        [
            {  # No trip: always "No trip", regardless of the SoC fields.
                "path_id": "normal",
                "local_date": 1,
                "day_label": "Mon 12",
                "drives_today": False,
                "departure_london": pd.NaT,
                "target_soc_percent": 80.0,
                "departure_soc_percent": float("nan"),
                "trip_energy_need_kwh": 0.0,
                "shortfall_kwh": 0.0,
                "departed_below_target": False,
            },
            {  # Departed at target, trip fully served: "On target".
                "path_id": "normal",
                "local_date": 2,
                "day_label": "Tue 13",
                "drives_today": True,
                "departure_london": pd.Timestamp("2026-01-13T07:00"),
                "target_soc_percent": 80.0,
                "departure_soc_percent": 80.0,
                "trip_energy_need_kwh": 10.0,
                "shortfall_kwh": 0.0,
                "departed_below_target": False,
            },
            {  # Below the preferred target, but with plenty of margin for
                # the trip -- no real shortfall, so no warning icon.
                "path_id": "selected",
                "local_date": 2,
                "day_label": "Tue 13",
                "drives_today": True,
                "departure_london": pd.Timestamp("2026-01-13T07:00"),
                "target_soc_percent": 80.0,
                "departure_soc_percent": 60.0,
                "trip_energy_need_kwh": 10.0,
                "shortfall_kwh": 0.0,
                "departed_below_target": True,
            },
            {  # Below target AND the trip was not fully served: the real
                # shortfall the icon is now reserved for.
                "path_id": "selected",
                "local_date": 3,
                "day_label": "Wed 14",
                "drives_today": True,
                "departure_london": pd.Timestamp("2026-01-14T07:00"),
                "target_soc_percent": 80.0,
                "departure_soc_percent": 15.0,
                "trip_energy_need_kwh": 10.0,
                "shortfall_kwh": 2.5,
                "departed_below_target": True,
            },
        ]
    )

    table = _daily_audit_table(daily_audit)

    statuses = dict(zip(table["Day"] + " " + table["Path"], table["Status"], strict=True))
    assert statuses["Mon 12 Unmanaged"] == "No trip"
    assert statuses["Tue 13 Unmanaged"] == "On target"
    assert statuses["Tue 13 Smart"] == "Below target"
    assert statuses["Wed 14 Smart"] == "⚠ shortfall"


def test_no_public_topup_content_for_no_action_model() -> None:
    # Public charging is action-only (contract v2: public_charging_status is
    # "not_modelled" for no-action results), so this exercises the "if
    # present" branch's absence path without crashing.
    result = make_result("no_action", evs=6, worlds=6)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert not any("top-up" in (trace.name or "").lower() for trace in figure.data)
    assert not any("Public top-ups" in str(args) for args, _ in calls_for(st, "caption"))


def test_public_topup_renders_real_content_in_flagged_world() -> None:
    # The amended fixture puts a synthetic public top-up on ev-0000, world 1,
    # selected path (decision 0004 item 32): a trip that would take SoC
    # below 20% tops up at a public charger to 80% first.
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit(choices={"one-ev-world": 1})

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    topup_traces = [trace for trace in figure.data if "top-up" in (trace.name or "").lower()]
    assert topup_traces
    assert all(len(trace.x) > 0 for trace in topup_traces)

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert any(caption.startswith("Public top-ups") for caption in captions)

    tables = [args[0] for args, _ in calls_for(st, "dataframe")]
    topup_table = tables[2]  # intervals table, plug-events table, then top-ups
    assert not topup_table.empty
    assert "Energy added (kWh)" in topup_table.columns
    assert float(topup_table["Energy added (kWh)"].iloc[0]) > 0
    assert topup_table["Path"].iloc[0] == "Smart"  # goal review item 6: not "selected"


def test_soc_line_values_equal_fixture_replay_values() -> None:
    result = make_result("no_action", evs=6, worlds=6)
    st = RecordingStreamlit()

    _render(st, result)

    replay = replay_one_ev(result, "ev-0000", result.representative_world_id)
    normal_rows = replay.intervals.loc[replay.intervals["path_id"].eq("normal")].sort_values(
        "slot_index"
    )
    figure = calls_for(st, "plotly_chart")[0][0][0]
    soc_trace = next(trace for trace in figure.data if trace.name == "SoC")
    np.testing.assert_allclose(
        np.asarray(soc_trace.y, dtype=float), normal_rows["battery_soc_percent"].to_numpy()
    )


def test_departure_and_topup_markers_never_take_a_legend_entry() -> None:
    # B4: the legend wrapped over roughly half the chart's height at 390 px
    # on action results, in part because each marker kind claimed a legend
    # entry (doubled per path). The caption's own "▲ plug-in, ▼ departure,
    # ◆ public top-up" key now names them, so no marker trace shows in the
    # legend at all. World 1 is the fixture's flagged world (a synthetic
    # public top-up on ev-0000, selected path only).
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit(choices={"one-ev-world": 1})

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    departures = [trace for trace in figure.data if trace.name == "Departure"]
    assert len(departures) == 2  # normal and selected both drove and departed
    assert not any(trace.showlegend for trace in departures)

    topups = [trace for trace in figure.data if trace.name == "Public top-up"]
    assert len(topups) == 1  # fixture's synthetic rescue is selected-path only
    assert not topups[0].showlegend

    plug_ins = [trace for trace in figure.data if trace.name == "Plug-in"]
    assert plug_ins and not plug_ins[0].showlegend


def test_band_always_shown_no_action_result() -> None:
    # Decision 0004 item 40: the across-weeks band is always on now -- no
    # checkbox, no opt-in. B3: named with the actual simulated-week count,
    # not a bare "weeks".
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert any(trace.name == "P10–P90 across 10 simulated weeks" for trace in figure.data)
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert any("Band: SoC P10–P90 across 10 simulated weeks." in caption for caption in captions)
    assert figure.layout.height == CHART_HEIGHTS["time_series_dual_axis"]


def test_no_band_checkbox_in_the_week_expander() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    assert not calls_for(st, "checkbox")


def test_two_path_band_uses_short_legend_names_and_extra_height() -> None:
    # Blocking review finding: on an action (two-path) result with the band
    # on, the full "P10-P90 across N simulated weeks (normal/selected)"
    # legend names doubled to 5 entries and wrapped at 390 px, squeezing the
    # plot below 260 px. The legend now gets short per-path names, the
    # caption still carries the full "across N simulated weeks" wording, and
    # the figure gets extra height so the plot area survives any wrap. The
    # band is always on now (item 40), so this needs no checkbox choice.
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    names = {trace.name for trace in figure.data}
    assert "Unmanaged P10–P90" in names
    assert "Smart P10–P90" in names  # goal review item 6: not "Selected"
    assert not any(name and "across" in name for name in names)  # kept out of the legend
    # Polish plan G3: each band joins its path's SoC line in the legend, so
    # the legend lists one entry per path, not one per statistic.
    listed = [trace.name for trace in figure.data if trace.showlegend is not False]
    assert "Unmanaged P10–P90" not in listed and "Smart P10–P90" not in listed
    bands = [trace for trace in figure.data if trace.name in ("Unmanaged P10–P90", "Smart P10–P90")]
    lines = {trace.legendgroup for trace in figure.data if (trace.name or "").startswith("SoC (")}
    assert {trace.legendgroup for trace in bands} == lines == {"normal", "selected"}

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert any("Band: SoC P10–P90 across 10 simulated weeks." in caption for caption in captions)

    # Goal review item 17: the bars moved to their own panel below (no more
    # legend-crowding compensation), so the total height is just the sum of
    # the two panels' own CHART_HEIGHTS entries.
    assert (
        figure.layout.height
        == CHART_HEIGHTS["time_series_dual_axis"] + CHART_HEIGHTS["small_multiple_panel"]
    )


def test_no_action_result_keeps_the_base_chart_height() -> None:
    # One path has no bars and only one band, so it never needs the
    # two-path extra height.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert figure.layout.height == CHART_HEIGHTS["time_series_dual_axis"]


def test_action_result_shows_home_charging_bars_normal_vs_smart() -> None:
    # Decision 0004 item 43: mark the half-hours smart charging used for
    # this EV, against the normal path, as home-charging kWh bars. Goal
    # review item 17: in their own panel below (shared x-axis), not on a
    # secondary axis, and out of the legend (at most five entries there).
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    bar_traces = {trace.name: trace for trace in figure.data if trace.type == "bar"}
    assert "Home charging (unmanaged)" in bar_traces
    assert "Home charging (smart)" in bar_traces  # goal review item 6: not "selected"
    assert all(not trace.showlegend for trace in bar_traces.values())
    assert figure.layout.yaxis2.title.text == "Home charging (kWh)"
    legend_traces = [trace for trace in figure.data if trace.showlegend is not False]
    assert len(legend_traces) <= 5

    replay = replay_one_ev(result, "ev-0000", result.representative_world_id)
    normal_rows = replay.intervals.loc[replay.intervals["path_id"].eq("normal")].sort_values(
        "slot_index"
    )
    np.testing.assert_allclose(
        np.asarray(bar_traces["Home charging (unmanaged)"].y, dtype=float),
        normal_rows["home_import_kwh"].to_numpy(),
    )

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert any("Lower panel" in caption and "unmanaged vs smart" in caption for caption in captions)


def test_no_action_result_has_no_home_charging_bars() -> None:
    # One path has nothing to compare against, so item 43's bars only make
    # sense on an action result.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert not any(trace.type == "bar" for trace in figure.data)
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert not any("Lower panel" in caption for caption in captions)


def test_action_result_marks_left_before_plan_finished_events() -> None:
    # Decision 0004 item 43: show "left before plan finished" events -- the
    # fixture's toy smart plan (charge 02:00-06:00 London) commonly leaves
    # an EV short at a departure outside that window (result_fixture.py
    # ``_simulate_path``); ev-0001 has a real case in every world, unlike
    # ev-0000 (this view's default EV).
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit(choices={"one-ev-unit": "ev-0001", "one-ev-world": 0})
    replay = replay_one_ev(result, "ev-0001", 0)
    selected_rows = replay.intervals.loc[replay.intervals["path_id"].eq("selected")]
    assert (selected_rows["early_departure_shortfall_kwh"] > 0).any()  # fixture has a real case

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    shortfall_traces = [trace for trace in figure.data if trace.name == "Left before plan finished"]
    assert shortfall_traces
    assert all(len(trace.x) > 0 for trace in shortfall_traces)
    assert not any(trace.showlegend for trace in shortfall_traces)

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    # The caption's marker key says "✕ left early"; the definition spells it out.
    assert any("✕ left early" in caption for caption in captions)
    assert any("left before its smart plan finished" in caption for caption in captions)


def test_no_action_result_never_marks_left_before_plan_finished() -> None:
    # The normal path has no plan to leave early on (contract v2 section 5:
    # early_departure_shortfall_kwh is always 0 there).
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert not any(trace.name == "Left before plan finished" for trace in figure.data)


def test_x_range_is_pinned_to_the_study_span() -> None:
    # B6: marker-only traces (plug-in, departure, top-up) used to pad
    # Plotly's autorange beyond the study span, so day labels crowded at
    # 390 px. The range must exactly match the normal path's own slots.
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit(choices={"one-ev-world": 1})  # the flagged world: has topups

    _render(st, result)

    replay = replay_one_ev(result, "ev-0000", 1)
    normal_rows = replay.intervals.loc[replay.intervals["path_id"].eq("normal")].sort_values(
        "slot_index"
    )
    figure = calls_for(st, "plotly_chart")[0][0][0]
    # Shared x-axis (make_subplots, goal review item 17): the range is set
    # on the bottom (bars) row's own axis, "xaxis2" in Plotly's subplot
    # numbering; the top row's axis matches it.
    range_start, range_end = figure.layout.xaxis2.range
    assert pd.Timestamp(range_start) == normal_rows["interval_start_utc"].min()
    assert pd.Timestamp(range_end) == normal_rows["interval_end_utc"].max()


def test_normal_soc_line_is_wider_than_selected_so_it_shows_underneath() -> None:
    # O7: at equal-ish widths, normal sat entirely under selected wherever
    # the two paths coincide (most of the week outside the action window),
    # reading as a single teal line. Normal must render wider so a sliver
    # always shows, without changing either path's colour (item 27).
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    normal_trace = next(trace for trace in figure.data if trace.name == "SoC (unmanaged)")
    selected_trace = next(trace for trace in figure.data if trace.name == "SoC (smart)")
    assert normal_trace.line.width > selected_trace.line.width
    assert normal_trace.line.color == PATH_STYLES["normal"]["color"]
    assert selected_trace.line.color == PATH_STYLES["selected"]["color"]


def test_no_action_result_keeps_the_shared_soc_line_width() -> None:
    # A single-path result has nothing to peek out from underneath, so it
    # keeps style.PATH_STYLES' shared width rather than the two-path override.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    soc_trace = next(trace for trace in figure.data if trace.name == "SoC")
    assert soc_trace.line.width == PATH_STYLES["normal"]["width"]


def test_topup_caption_reads_threshold_and_target_from_assumptions() -> None:
    # O5: the caption must report the run's own top-up threshold/target,
    # not decision 0004 item 32's hard-coded illustrative defaults, so an
    # edited-assumptions run is described correctly.
    result = make_result("action", evs=6, worlds=10)
    extra = (
        Assumption(
            name="public_top_up_threshold_soc_percent",
            label="Public top-up threshold SoC",
            value=15.0,
            unit="percent",
            evidence="illustrative",
            source="test",
            meaning="test",
            editable=True,
            bounds=(0.0, 100.0),
            group="Public & prices",
            affects="test",
        ),
        Assumption(
            name="public_top_up_target_soc_percent",
            label="Public top-up target SoC",
            value=90.0,
            unit="percent",
            evidence="illustrative",
            source="test",
            meaning="test",
            editable=True,
            bounds=(0.0, 100.0),
            group="Public & prices",
            affects="test",
        ),
    )
    result = replace(result, assumptions=result.assumptions + extra)
    st = RecordingStreamlit(choices={"one-ev-world": 1})

    _render(st, result)

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    topup_caption = next(caption for caption in captions if caption.startswith("Public top-ups"))
    assert "below 15%" in topup_caption
    assert "to 90% first" in topup_caption


def test_topup_caption_falls_back_to_unavailable_without_the_assumption() -> None:
    # The fixture does not (yet) carry these two assumption records; the
    # caption must degrade to "Unavailable" rather than inventing a number.
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit(choices={"one-ev-world": 1})

    _render(st, result)

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    topup_caption = next(caption for caption in captions if caption.startswith("Public top-ups"))
    assert "below Unavailable" in topup_caption
    assert "to Unavailable first" in topup_caption


# --- Intraday dispatch caption (contract v1 §10; K4-flagged bug applies here too) ---


def test_dispatch_caption_names_locked_free_and_excludes_control_evs() -> None:
    assert _dispatch_caption({"dispatch_locked": True}) == "Locked to its day-ahead plan"
    assert (
        _dispatch_caption({"dispatch_locked": False})
        == "Re-plans hourly on the latest intraday price"
    )
    # A control EV keeps a locked/free status but ignores every plan and
    # charges by the normal rule (intraday-dispatch-v1 §2, §10.1e).
    assert _dispatch_caption({"dispatch_locked": True, "control_group": True}) is None
    assert _dispatch_caption({"dispatch_locked": False, "control_group": True}) is None
    assert _dispatch_caption({"dispatch_locked": None}) is None
    assert _dispatch_caption({}) is None


def test_dispatch_caption_absent_on_the_synthetic_fixture() -> None:
    # No run ever dispatched in the SYNTHETIC fixture: render_one_ev's own
    # ``result.dispatch_world_slot is not None`` gate keeps the caption off,
    # whatever units.dispatch_locked (all-False, see result_fixture.py) says.
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit()

    _render(st, result)

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert not any("day-ahead plan" in c or "re-plan" in c.lower() for c in captions)


def test_plug_events_table_shows_plug_out_in_london_time_not_utc() -> None:
    # Clarity critique: plug-in was already London time, plug-out stayed
    # UTC on the same row -- fixed the same way as Drivers > Plug-ins'
    # sampled events table.
    plug_in_utc = pd.to_datetime(["2026-07-06 17:00"], utc=True)
    plug_out_utc = pd.to_datetime(["2026-07-06 22:00"], utc=True)  # BST: 23:00 London
    events = pd.DataFrame(
        {
            "plug_in_london": plug_in_utc.tz_convert("Europe/London"),
            "plug_in_soc_percent": [40.0],
            "plug_out_utc": plug_out_utc,
            "plug_out_soc_percent": [80.0],
            "home_import_kwh": [10.0],
            "battery_added_kwh": [9.5],
            "still_plugged_at_horizon_end": [False],
        }
    )

    table = _plug_events_table(events)

    assert "Plug-out (UTC)" not in table.columns
    assert "Plug-out (London)" in table.columns
    assert table["Plug-out (London)"].iloc[0] == pd.Timestamp(
        "2026-07-06 23:00", tz="Europe/London"
    )


# --- Timed tariff, the optional third path (decision 0007) ------------------
#
# Real small forecasts, not an injected fake path: an earlier version of this
# file found no "timed" rows in replay_one_ev's output on this worktree and
# wrongly reported the model as not ready, because the ad hoc check that
# produced that finding ran under the main checkout's editable install
# (PYTHONPATH=src was set for pytest but not for that one-off script, and
# ``python -I`` ignores PYTHONPATH outright). Re-run correctly
# (``axle_studio.__file__`` printed from this worktree's own ``src/``),
# ``model.individual.replay_one_ev``/``replay_one_ev_bands`` on this worktree
# DO carry "timed" once ``timed_start_local_hour`` is set, so these tests use
# the real model functions on a real small run for both states (decision
# 0007's own 0.0 default-start case and NaN, off) rather than a fake.
_TIMED_RUN_DATE = date(2026, 10, 12)
_TIMED_RUN_VALUES = {"vehicle_count": 20, "evaluation_world_count": 3, "seed": 42}


@pytest.fixture(scope="module")
def timed_result():
    """A real small action run with the timed-tariff setting on (start hour 0.0, midnight)."""

    return run_forecast_from_assumptions(
        _TIMED_RUN_DATE,
        model="action",
        values={**_TIMED_RUN_VALUES, "timed_start_local_hour": 0.0},
    )


@pytest.fixture(scope="module")
def untimed_result():
    """The same shape of run with the Timed tariff policy switched off."""

    return run_forecast_from_assumptions(
        _TIMED_RUN_DATE,
        model="action",
        values={**_TIMED_RUN_VALUES, "timed_tariff_enabled": 0},
    )


def _render_real(st: RecordingStreamlit, result) -> None:
    render_one_ev(
        st, result, replay_one_ev=model_replay_one_ev, replay_one_ev_bands=model_replay_one_ev_bands
    )


def test_present_paths_reads_display_order_not_a_fixed_count() -> None:
    # Decision 0007: Unmanaged, Timed tariff, Smart -- the screen order,
    # which differs from the kernel/contract order (normal, selected, timed).
    # Pure unit test on a crafted frame: no model call can isolate "every
    # order the input might arrive in" the way a literal DataFrame can.
    intervals = pd.DataFrame({"path_id": ["selected", "normal", "timed", "normal"]})
    assert _present_paths(intervals) == ["normal", "timed", "selected"]

    # A two-path replay (today's only real case) is unaffected.
    two_path = pd.DataFrame({"path_id": ["selected", "normal"]})
    assert _present_paths(two_path) == ["normal", "selected"]


def test_timed_path_adds_a_third_soc_line_in_display_order(timed_result) -> None:
    st = RecordingStreamlit()

    _render_real(st, timed_result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    names = [trace.name for trace in figure.data if trace.name and trace.name.startswith("SoC (")]
    assert names == ["SoC (unmanaged)", "SoC (timed tariff)", "SoC (smart)"]


def test_timed_soc_line_uses_the_step_shape_from_path_styles(timed_result) -> None:
    # style.PATH_STYLES["timed"]'s "hv" (decision 0007: charging is barred or
    # allowed for a whole half-hour at a time, so the line should jump at
    # each boundary, not slope through it); normal/selected have no
    # "line_shape" entry and keep Plotly's default (linear, reported as None).
    st = RecordingStreamlit()

    _render_real(st, timed_result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    by_name = {trace.name: trace for trace in figure.data if trace.name}
    assert by_name["SoC (timed tariff)"].line.shape == PATH_STYLES["timed"]["line_shape"]
    assert by_name["SoC (unmanaged)"].line.shape is None
    assert by_name["SoC (smart)"].line.shape is None


def test_timed_path_stays_within_the_five_entry_legend_limit(timed_result) -> None:
    # B4: shading + one SoC line per path; bands and markers are already
    # kept out of the legend (showlegend=False), so a third path adds one
    # legend entry, not a second batch of five.
    st = RecordingStreamlit()

    _render_real(st, timed_result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    legend_traces = [trace for trace in figure.data if trace.showlegend is not False]
    assert len(legend_traces) <= 5
    assert {trace.name for trace in legend_traces} == {
        "Plugged in at home",
        "SoC (unmanaged)",
        "SoC (timed tariff)",
        "SoC (smart)",
    }


def test_timed_path_adds_a_third_home_charging_bar_series(timed_result) -> None:
    st = RecordingStreamlit()

    _render_real(st, timed_result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    bar_names = {trace.name for trace in figure.data if trace.type == "bar"}
    assert bar_names == {
        "Home charging (unmanaged)",
        "Home charging (timed tariff)",
        "Home charging (smart)",
    }
    assert all(not trace.showlegend for trace in figure.data if trace.type == "bar")
    # Chart height is still just the two panels' own heights, regardless of
    # how many path series share the bar panel.
    assert (
        figure.layout.height
        == CHART_HEIGHTS["time_series_dual_axis"] + CHART_HEIGHTS["small_multiple_panel"]
    )


def test_timed_path_captions_name_all_three_policies(timed_result) -> None:
    # chart_block renders ``definition`` with one st.caption(...) call inside
    # the "Data and definition" expander (components/chart_table.py), so the
    # lower-panel wording and the timed-path sentence are both in this one
    # recorded caption, the same text the expander shows.
    st = RecordingStreamlit()

    _render_real(st, timed_result)

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    definition = next(caption for caption in captions if "Lower panel" in caption)
    assert "unmanaged, timed tariff and smart" in definition
    assert "timed tariff bars show the clock-time rule" in definition
    assert "Setting a timed tariff start time adds a third path" in definition


def test_timed_path_table_rows_use_the_timed_tariff_label(timed_result) -> None:
    st = RecordingStreamlit()

    _render_real(st, timed_result)

    tables = [args[0] for args, _ in calls_for(st, "dataframe")]
    intervals_table = tables[0]
    audit_table = tables[-1]
    assert "Timed tariff" in set(intervals_table["Path"])
    assert "Timed tariff" in set(audit_table["Path"])
    # Table row order is contract (kernel) order, not display order (decision
    # 0007 rule 2): normal, selected, timed sorts the same as alphabetical.
    assert list(intervals_table["Path"].unique()) == ["Unmanaged", "Smart", "Timed tariff"]


def test_without_timed_setting_a_real_run_matches_the_two_path_screen(untimed_result) -> None:
    # Contract rule: absent data, unchanged screen. A real run with the
    # setting off (NaN) must read exactly as item 43's original two-path
    # wording, not just the synthetic fixture's toy two-path case.
    st = RecordingStreamlit()

    _render_real(st, untimed_result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    names = [trace.name for trace in figure.data if trace.name and trace.name.startswith("SoC (")]
    assert names == ["SoC (unmanaged)", "SoC (smart)"]
    assert not any(trace.name == "SoC (timed tariff)" for trace in figure.data)

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    definition = next(caption for caption in captions if "Lower panel" in caption)
    assert definition.endswith(
        " Lower panel: this EV's home charging each half-hour, unmanaged vs smart. The "
        "smart bars show where the plan put its charging: the cheapest forecast half-hours."
    )
    assert "timed tariff" not in definition.lower()


def test_replay_cache_does_not_leak_between_runs_with_and_without_timed(
    timed_result, untimed_result
) -> None:
    # ReplayState.replay_cache is a fresh dict per result (a default_factory
    # field; ReplayState's own eq=False keeps identity hashing), so two
    # results never share a cache; this is a regression guard on that
    # invariant from the view's own call pattern, not a model-code change.
    # Both runs share the same seed/cohort assignment, so the same unit_id
    # names the same EV in each.
    unit_id = untimed_result.units["unit_id"].iat[0]
    assert unit_id == timed_result.units["unit_id"].iat[0]

    off = model_replay_one_ev(untimed_result, unit_id, untimed_result.representative_world_id)
    on = model_replay_one_ev(timed_result, unit_id, timed_result.representative_world_id)
    assert sorted(off.intervals["path_id"].unique()) == ["normal", "selected"]
    assert sorted(on.intervals["path_id"].unique()) == ["normal", "selected", "timed"]

    # Re-fetch (the cache-hit branch) on each result and confirm no drift
    # from re-reading the other result's cache.
    off_again = model_replay_one_ev(untimed_result, unit_id, untimed_result.representative_world_id)
    on_again = model_replay_one_ev(timed_result, unit_id, timed_result.representative_world_id)
    assert sorted(off_again.intervals["path_id"].unique()) == ["normal", "selected"]
    assert sorted(on_again.intervals["path_id"].unique()) == ["normal", "selected", "timed"]
