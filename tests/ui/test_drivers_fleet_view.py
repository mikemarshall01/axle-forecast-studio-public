"""Drivers ▸ Fleet week renders normal-path behaviour from the contract v2 fixture."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fixtures.result_fixture import make_result

from axle_studio.ui.registry import FLEET_GROUP, GROUP_KEY
from axle_studio.ui.style import (
    ARCHETYPE_COLOURS,
    CHART_HEIGHTS,
    INK,
    UNMANAGED_GLOSSARY,
    band_fill,
)
from axle_studio.ui.views.drivers_fleet import (
    _COHORT_LINE_ALPHA,
    _METRIC_LABELS,
    _PLUGGED_IN_EXTRA_HEIGHT_PX,
    _unserved_travel_caption,
    render_fleet_week,
)


class _Container:
    def __enter__(self) -> "_Container":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class RecordingStreamlit:
    def __init__(self, *, metric: str = "Plugged in", group: str = FLEET_GROUP) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self._choices = {"fleet-week-metric": metric, GROUP_KEY: group}

    def __getattr__(self, name: str):
        def record(*args: object, **kwargs: object) -> object:
            self.calls.append((name, args, kwargs))
            if name == "columns":
                count = args[0] if args else kwargs.get("spec")
                n = count if isinstance(count, int) else len(count)
                return [_Container() for _ in range(n)]
            if name == "expander":
                return _Container()
            if name == "selectbox":
                return self._choices.get(kwargs["key"], kwargs["options"][0])
            return None

        return record


def calls_for(st: RecordingStreamlit, name: str) -> list[tuple[tuple, dict]]:
    return [(args, kwargs) for call_name, args, kwargs in st.calls if call_name == name]


def legend_names(figure) -> list[str]:
    """Names of the traces that take a legend entry (band traces never do)."""

    return [trace.name for trace in figure.data if trace.showlegend is not False]


def line_trace(figure, name: str):
    """The median line of a ``band_and_line`` series: its one legend-listed trace."""

    return next(
        trace for trace in figure.data if trace.name == name and trace.showlegend is not False
    )


@pytest.mark.parametrize("model", ["action", "no_action"])
def test_renders_both_models_without_model_calls(model: str) -> None:
    result = make_result(model, evs=6, worlds=12)
    st = RecordingStreamlit()

    render_fleet_week(st, result)

    assert len(calls_for(st, "plotly_chart")) == 1
    assert not calls_for(st, "button")


def test_five_ev_case_with_fewer_than_six_cohorts_renders() -> None:
    result = make_result("no_action", evs=5, worlds=8)
    st = RecordingStreamlit()

    render_fleet_week(st, result)  # must not raise

    assert len(calls_for(st, "plotly_chart")) == 1


@pytest.mark.parametrize("clock_change", ["autumn", "spring"])
def test_clock_change_week_renders(clock_change: str) -> None:
    result = make_result("action", evs=6, worlds=6, clock_change=clock_change)
    st = RecordingStreamlit()

    render_fleet_week(st, result)

    assert len(calls_for(st, "plotly_chart")) == 1


def test_chart_height_and_legend_below() -> None:
    # Travel has no across-EVs band and no per-cohort lines, so it keeps the
    # base height (only Plugged in adds height, for its cohort legend rows).
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit(metric="Travel")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert figure.layout.height == CHART_HEIGHTS["time_series"]
    # Polish plan G2: legend anchored to the figure's own bottom edge.
    assert figure.layout.template.layout.legend.yref == "container"


def test_plugged_in_gets_extra_height_for_cohort_lines() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit(metric="Plugged in")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert figure.layout.height == CHART_HEIGHTS["time_series"] + _PLUGGED_IN_EXTRA_HEIGHT_PX


def test_battery_soc_keeps_the_base_height() -> None:
    # Polish plan G2: style_figure's content-based bottom margin now makes
    # room for the across-EVs band's one legend entry, so no extra height.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit(metric="Battery SoC")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert figure.layout.height == CHART_HEIGHTS["time_series"]


def test_home_import_keeps_the_base_height() -> None:
    # Goal review item 16: Home import's across-EVs band (and the extra
    # height that came with it) is gone.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit(metric="Home import")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert figure.layout.height == CHART_HEIGHTS["time_series"]


def test_london_axis_labels() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert any(label.endswith(":00") or " " in label for label in figure.layout.xaxis.ticktext)


def test_battery_soc_axis_is_fixed_0_to_100() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit(metric="Battery SoC")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert list(figure.layout.yaxis.range) == [0, 100]


def test_home_import_shows_home_and_total_lines_labelled_per_o7() -> None:
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit(metric="Home import")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert legend_names(figure) == ["Home grid import", "Home + public grid import"]


def test_travel_shows_driving_and_away_named_lines() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit(metric="Travel")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert legend_names(figure) == ["Driving", "Away from home"]

    table = calls_for(st, "dataframe")[0][0][0]
    assert set(table["Series"]) == {"Driving", "Away from home"}


def test_unserved_travel_caption_reads_weekly_bands_max() -> None:
    # Decision 0004 item 32: EVs never strand, so the fixture's weekly
    # unserved-travel max is 0 for every seed/world combination -- this
    # asserts the caption is actually driven by ``weekly_bands.max``, not
    # invented or read from a different frame.
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_fleet_week(st, result)

    row = result.weekly_bands.loc[
        result.weekly_bands["path_id"].eq("normal")
        & result.weekly_bands["metric"].eq("unserved_travel_kwh")
    ].iloc[0]
    assert float(row["max"]) == 0.0
    caption = calls_for(st, "caption")[0][0][0]
    assert "stranded" not in caption.lower()
    assert "top-ups prevent it" in caption


def test_unserved_travel_caption_nonzero_branch_names_worst_week() -> None:
    # The fixture never produces a nonzero weekly max (item 32 prevents
    # stranding), so this exercises the caption's other branch directly
    # against a minimal fake ``weekly_bands`` frame.
    weekly_bands = pd.DataFrame(
        [
            {
                "path_id": "normal",
                "metric": "unserved_travel_kwh",
                "unit": "kWh per week",
                "world_count": 10,
                "mean": 1.5,
                "p10": 0.0,
                "p50": 0.8,
                "p90": 3.0,
                "max": 4.2,
            }
        ]
    )
    fake_result = SimpleNamespace(weekly_bands=weekly_bands)

    caption = _unserved_travel_caption(fake_result)

    assert "stranded" not in caption.lower()
    assert "4.2 kWh" in caption
    assert "0.8 kWh/week" in caption


def test_band_and_median_are_one_legend_entry_per_series() -> None:
    # Polish plan G3: the across-weeks band shares its median line's legend
    # group and never lists itself, so the legend names series, not
    # statistics; the caption names the spread instead.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit(metric="Travel")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    for name in ("Driving", "Away from home"):
        series = [trace for trace in figure.data if trace.name == name]
        assert len(series) == 3  # P10 bound, filled P90 bound, median line
        assert {trace.legendgroup for trace in series} == {name}
        assert [trace.showlegend for trace in series] == [False, False, True]
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert "Median, P10–P90 across 10 weeks." in captions[0]


def test_key_numbers_equal_fixture_band_values() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_fleet_week(st, result)

    bands = result.fleet_interval_bands
    rows = bands.loc[
        bands["path_id"].eq("normal") & bands["metric"].eq("connected_share")
    ].sort_values("slot_index")
    figure = calls_for(st, "plotly_chart")[0][0][0]
    median_trace = line_trace(figure, "Fleet plugged in at home")
    np.testing.assert_allclose(
        np.asarray(median_trace.y, dtype=float), rows["p50"].to_numpy() * 100.0
    )


def test_battery_soc_shows_both_bands_named_separately() -> None:
    # Decision 0004 items 39, 40: the always-on across-weeks band plus a
    # separate, lighter across-EVs band, each named in the legend.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit(metric="Battery SoC")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert legend_names(figure) == ["P10–P90 across EVs", "Fleet battery SoC (capacity-weighted)"]

    ev_bands = result.fleet_interval_ev_bands
    rows = ev_bands.loc[
        ev_bands["path_id"].eq("normal") & ev_bands["metric"].eq("battery_soc_percent")
    ].sort_values("slot_index")
    # The band is two traces (hidden P10 boundary, filled P90); only the
    # visible (legend-shown) one carries the P90 values.
    ev_trace = next(
        trace for trace in figure.data if trace.name == "P10–P90 across EVs" and trace.showlegend
    )
    np.testing.assert_allclose(np.asarray(ev_trace.y, dtype=float), rows["p90"].to_numpy())

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert any("P10–P90 across EVs" in caption for caption in captions)


def test_home_import_shows_only_the_across_weeks_band_no_ev_band() -> None:
    # Goal review item 16: Home import's across-EVs band was mostly 0 at
    # P10/P50 (only a fraction of the fleet charges at once), so it is
    # dropped; the across-weeks band and the fleet line stay, single axis.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit(metric="Home import")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    names = {trace.name for trace in figure.data}
    assert any(
        trace.name == "Home grid import" and trace.fill == "tonexty" for trace in figure.data
    )
    assert not any("across EVs" in name for name in names)
    assert figure.layout.yaxis.title.text == "kW"
    assert "yaxis2" not in figure.layout

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    caption = next(iter(captions))
    assert "across EVs" not in caption


def test_travel_metric_keeps_a_single_axis_with_no_ev_band() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit(metric="Travel")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert figure.layout.yaxis.title.text == "% of EV-time"
    assert "yaxis2" not in figure.layout


def test_plugged_in_shows_fleet_total_and_one_line_per_cohort() -> None:
    # Decision 0004 item 43: fleet total (with its band) plus one line per
    # sampled archetype, from cohort_interval_bands.
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit(metric="Plugged in")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    names = {trace.name for trace in figure.data}
    sampled = result.cohort_summary.loc[result.cohort_summary["ev_count"] > 0, "cohort_label"]
    cohort_labels = set(sampled)
    assert cohort_labels
    assert cohort_labels.issubset(names)
    assert any("Fleet plugged in at home" in name for name in names)

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert any("one per archetype" in caption for caption in captions)


def test_plugged_in_cohort_line_values_equal_fixture_cohort_bands() -> None:
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit(metric="Plugged in")

    render_fleet_week(st, result)

    cohort_id, cohort_label = next(
        (row.cohort_id, row.cohort_label)
        for row in result.cohort_summary.itertuples()
        if row.ev_count > 0
    )
    cohort_bands = result.cohort_interval_bands
    rows = cohort_bands.loc[
        cohort_bands["cohort_id"].eq(cohort_id)
        & cohort_bands["path_id"].eq("normal")
        & cohort_bands["metric"].eq("connected_share")
    ].sort_values("slot_index")
    figure = calls_for(st, "plotly_chart")[0][0][0]
    cohort_trace = next(trace for trace in figure.data if trace.name == cohort_label)
    np.testing.assert_allclose(
        np.asarray(cohort_trace.y, dtype=float), rows["p50"].to_numpy() * 100.0
    )


def test_plugged_in_cohort_lines_are_faint_and_thinner_than_the_fleet_median() -> None:
    # Goal review item 16: fainter (not full-strength Okabe-Ito), so six
    # cohort lines read as context behind the fleet total's own bold band
    # and median, keeping the legend readable at 390 px without dropping a
    # cohort.
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit(metric="Plugged in")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    fleet_median = line_trace(figure, "Fleet plugged in at home")
    cohort_labels = set(result.cohort_summary["cohort_label"])
    cohort_traces = [trace for trace in figure.data if trace.name in cohort_labels]
    assert cohort_traces
    for trace in cohort_traces:
        assert trace.line.width < fleet_median.line.width


def test_plugged_in_cohort_lines_use_the_archetype_palette_in_order() -> None:
    # Polish plan G4: cohort lines take style.ARCHETYPE_COLOURS (six muted
    # hues, no teal) by cohort position, not Plotly's or a local palette.
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit(metric="Plugged in")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    sampled = result.cohort_summary.loc[result.cohort_summary["ev_count"] > 0, "cohort_label"]
    by_name = {trace.name: trace for trace in figure.data}
    expected = [band_fill(colour, _COHORT_LINE_ALPHA) for colour in ARCHETYPE_COLOURS]
    assert [by_name[label].line.color for label in sampled] == expected[: len(sampled)]
    # One legend entry for the fleet series plus one per cohort line.
    assert legend_names(figure) == ["Fleet plugged in at home", *sampled]


def test_plugged_in_fleet_total_stands_apart_from_the_cohort_lines() -> None:
    # Review O5: the unmanaged grey sat at the archetype colours' lightness,
    # so the fleet total is drawn in the main ink and thicker than any cohort.
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit(metric="Plugged in")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    sampled = set(result.cohort_summary["cohort_label"])
    fleet = [t for t in figure.data if t.name == "Fleet plugged in at home"][-1]
    cohorts = [t for t in figure.data if t.name in sampled]
    assert fleet.line.color == INK
    assert fleet.line.color not in ARCHETYPE_COLOURS
    assert all(fleet.line.width > cohort.line.width for cohort in cohorts)


def test_five_ev_case_plugged_in_renders_with_fewer_cohorts() -> None:
    # Chart audit B1: an archetype with no sampled EVs must not crash the
    # per-cohort line loop.
    result = make_result("no_action", evs=5, worlds=8)
    st = RecordingStreamlit(metric="Plugged in")

    render_fleet_week(st, result)  # must not raise

    assert len(calls_for(st, "plotly_chart")) == 1


# --- Merged flexibility metrics and the group selector (decision 0004 item 54) ---


def test_metric_control_groups_behaviour_then_flexibility() -> None:
    result = make_result("action", evs=6, worlds=6)
    st = RecordingStreamlit()

    render_fleet_week(st, result)

    metric_control = next(
        kwargs for _, kwargs in calls_for(st, "selectbox") if kwargs["key"] == "fleet-week-metric"
    )
    labels = [metric_control["format_func"](option) for option in metric_control["options"]]
    assert labels == [
        "Behaviour: Plugged in",
        "Behaviour: Home import",
        "Behaviour: Battery SoC",
        "Behaviour: Travel",
        "Flexibility: Deferrable power by slack",
        "Flexibility: Turn-up headroom",
        "Flexibility: Movable energy",
        "Network: Zone import",
    ]
    assert "Time slack" not in metric_control["options"]  # retired (plan C1)


@pytest.mark.parametrize("model", ["action", "no_action"])
@pytest.mark.parametrize("metric", _METRIC_LABELS)
def test_every_metric_renders_for_both_models(model: str, metric: str) -> None:
    result = make_result(model, evs=6, worlds=6)
    st = RecordingStreamlit(metric=metric)

    render_fleet_week(st, result)

    headroom_without_smart = metric == "Turn-up headroom" and model == "no_action"
    assert len(calls_for(st, "plotly_chart")) == (0 if headroom_without_smart else 1)
    if headroom_without_smart:
        assert "smart path" in calls_for(st, "info")[0][0][0]


def test_deferrable_power_plots_cumulative_rows_never_stacked_buckets() -> None:
    # Contract v2 3.6d: bucket quantiles do not add up, so the chart draws
    # the per-week-summed "at least N hours" rows as overlapping areas.
    result = make_result("action", evs=6, worlds=8)
    st = RecordingStreamlit(metric="Deferrable power by slack")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    frame = result.deferrable_power_bands
    areas = [trace for trace in figure.data if trace.fill == "tozeroy"]
    assert [trace.legendgroup for trace in areas] == [
        "total",
        "at_least_1h",
        "at_least_2h",
        "at_least_4h",
        "at_least_8h",
    ]
    for trace in areas:
        expected = frame.loc[frame["slack_bucket"].eq(trace.legendgroup)].sort_values("slot_index")
        np.testing.assert_allclose(np.asarray(trace.y, dtype=float), expected["p50"].to_numpy())
    assert all(trace.stackgroup is None for trace in figure.data)
    total_line = line_trace(figure, "Total, P10–P90")
    total = frame.loc[frame["slack_bucket"].eq("total")].sort_values("slot_index")
    np.testing.assert_allclose(np.asarray(total_line.y, dtype=float), total["p50"].to_numpy())

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert "kW that can wait at least N hours and still reach target" in captions[0]
    assert UNMANAGED_GLOSSARY in captions


def test_deferrable_power_keeps_the_time_slack_rows_in_an_expander() -> None:
    result = make_result("action", evs=6, worlds=8)
    st = RecordingStreamlit(metric="Deferrable power by slack")

    render_fleet_week(st, result)

    assert ("Time slack data (retired chart)",) in [args for args, _ in calls_for(st, "expander")]
    slack = calls_for(st, "dataframe")[1][0][0]
    expected = result.flexibility_bands
    expected = expected.loc[
        expected["metric"].eq("time_slack_hours") & expected["path_id"].eq("normal")
    ]
    assert len(slack) == len(expected)
    assert set(slack["Spread"]) == {"Across EVs", "Across weeks"}


def test_turn_up_headroom_shows_the_smart_path_only() -> None:
    result = make_result("action", evs=6, worlds=8)
    st = RecordingStreamlit(metric="Turn-up headroom")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert legend_names(figure) == ["Smart"]
    bands = result.flexibility_bands
    expected = bands.loc[
        bands["metric"].eq("turn_up_headroom_kw")
        & bands["spread"].eq("across_weeks")
        & bands["path_id"].eq("selected")
    ].sort_values("slot_index")
    np.testing.assert_allclose(
        np.asarray(line_trace(figure, "Smart").y, dtype=float), expected["p50"].to_numpy()
    )


def test_movable_energy_shows_the_unmanaged_path() -> None:
    result = make_result("action", evs=6, worlds=8)
    st = RecordingStreamlit(metric="Movable energy")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert legend_names(figure) == ["Unmanaged"]
    assert figure.layout.yaxis.title.text == "kWh"


def test_group_selector_lists_fleet_then_sampled_archetypes() -> None:
    result = make_result("no_action", evs=5, worlds=6)  # one archetype has no EV
    st = RecordingStreamlit()

    render_fleet_week(st, result)

    group_control = next(
        kwargs for _, kwargs in calls_for(st, "selectbox") if kwargs["key"] == GROUP_KEY
    )
    sampled = result.cohort_summary.loc[result.cohort_summary["ev_count"] > 0, "cohort_id"]
    assert group_control["options"] == [FLEET_GROUP, *sampled]
    assert group_control["persist_state"] == "session"
    assert not group_control.get("disabled")


def test_group_selector_is_disabled_for_fleet_only_flexibility_metrics() -> None:
    result = make_result("action", evs=6, worlds=6)
    st = RecordingStreamlit(metric="Movable energy", group="average_uk")

    render_fleet_week(st, result)

    # A separate display-only box reads "Fleet"; the shared Group widget is
    # not rendered, so the stored archetype choice is not overwritten.
    group_controls = [
        kwargs for _, kwargs in calls_for(st, "selectbox") if kwargs["key"] != "fleet-week-metric"
    ]
    assert len(group_controls) == 1
    assert group_controls[0]["key"] != GROUP_KEY
    assert group_controls[0]["options"] == ["Fleet"]
    assert group_controls[0]["disabled"] is True
    # The flexibility chart stays fleet-wide whatever the stored group.
    assert len(calls_for(st, "plotly_chart")) == 1
    title = next(args[0] for args, _ in calls_for(st, "markdown") if "movable" in str(args[0]))
    assert "Fleet" in title
    assert calls_for(st, "caption")[0][0][0].startswith("Unmanaged path, fleet.")


@pytest.mark.parametrize("metric", ["Plugged in", "Home import", "Battery SoC", "Travel"])
def test_archetype_group_plots_that_archetypes_bands(metric: str) -> None:
    result = make_result("no_action", evs=6, worlds=8)
    st = RecordingStreamlit(metric=metric, group="intelligent_octopus")

    render_fleet_week(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    cohort = result.cohort_interval_bands
    metric_id = {
        "Plugged in": "connected_share",
        "Home import": "home_import_kw",
        "Battery SoC": "battery_soc_percent",
        "Travel": "driving_share",
    }[metric]
    expected = cohort.loc[
        cohort["cohort_id"].eq("intelligent_octopus")
        & cohort["path_id"].eq("normal")
        & cohort["metric"].eq(metric_id)
    ].sort_values("slot_index")
    scale = 100.0 if metric_id.endswith("share") else 1.0
    first_line = next(trace for trace in figure.data if trace.showlegend is not False)
    np.testing.assert_allclose(
        np.asarray(first_line.y, dtype=float), expected["p50"].to_numpy() * scale
    )
    # No fleet-only extras on an archetype: no across-EVs band, no cohort lines.
    assert "P10–P90 across EVs" not in legend_names(figure)
    assert len(legend_names(figure)) <= 2
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert captions[0].startswith("Unmanaged, Intelligent Octopus average.")
    assert all(len(caption) <= 140 for caption in captions)
