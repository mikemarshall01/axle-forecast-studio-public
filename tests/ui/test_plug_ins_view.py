"""Drivers ▸ Plug-ins renders the brief's two histograms from the contract v2 fixture."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fixtures.result_fixture import make_result
from kpi_calls import kpi_calls

from axle_studio.ui.registry import DAY_TYPE_KEY
from axle_studio.ui.style import CHART_HEIGHTS, MUTED_INK, SEQUENTIAL, SERIES_COLOURS
from axle_studio.ui.views.plug_ins import _render_kpi_row, render_plug_ins


class _Container:
    def __enter__(self) -> "_Container":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class RecordingStreamlit:
    """Fake ``st`` module: records every call; widgets return a fixed choice."""

    def __init__(self, *, day: str = "Weekday", choices: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self._day = day
        self._choices = choices or {}

    def __getattr__(self, name: str):
        def record(*args: object, **kwargs: object) -> object:
            self.calls.append((name, args, kwargs))
            if name == "columns":
                count = args[0] if args else kwargs.get("spec")
                n = count if isinstance(count, int) else len(count)
                # Columns share this log, so a KPI tile drawn into a column
                # (``kpi(column, ...)``) is recorded in render order.
                return [self for _ in range(n)]
            if name == "expander":
                return _Container()
            if name == "segmented_control":
                if kwargs.get("key") == DAY_TYPE_KEY:
                    return self._day.lower()
                return self._choices.get(kwargs.get("key"), kwargs.get("default"))
            return None

        return record

    def __enter__(self) -> "RecordingStreamlit":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def calls_for(st: RecordingStreamlit, name: str) -> list[tuple[tuple, dict]]:
    return [(args, kwargs) for call_name, args, kwargs in st.calls if call_name == name]


@pytest.mark.parametrize("model", ["action", "no_action"])
def test_renders_both_models_without_model_calls(model: str) -> None:
    result = make_result(model, evs=6, worlds=12)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    figures = [args[0] for args, _ in calls_for(st, "plotly_chart")]
    # Hour and SoC histograms, then the plug-in and departure heatmaps.
    assert len(figures) == 4
    assert not calls_for(st, "button")  # no run/button-triggered model call


def test_five_ev_case_with_fewer_than_six_cohorts_renders() -> None:
    result = make_result("no_action", evs=5, worlds=8)
    st = RecordingStreamlit()

    render_plug_ins(st, result)  # must not raise

    assert len(calls_for(st, "plotly_chart")) == 4


@pytest.mark.parametrize("clock_change", ["autumn", "spring"])
def test_clock_change_week_renders(clock_change: str) -> None:
    result = make_result("action", evs=6, worlds=6, clock_change=clock_change)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    assert len(calls_for(st, "plotly_chart")) == 4


def test_chart_heights_and_legend_placement() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    charts = calls_for(st, "plotly_chart")
    for args, _ in charts[:2]:  # the two histograms
        figure = args[0]
        assert figure.layout.height == CHART_HEIGHTS["histogram"]
        # Legend anchored to the figure's own bottom edge (polish plan G2).
        legend = figure.layout.template.layout.legend
        assert (legend.yref, legend.y, legend.yanchor) == ("container", 0, "bottom")

    for args, _ in charts[2:]:  # the two weekday x half-hour heatmaps
        assert args[0].layout.height == CHART_HEIGHTS["histogram"]


def test_hour_axis_is_labelled_london() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    hour_figure = calls_for(st, "plotly_chart")[0][0][0]
    assert hour_figure.layout.xaxis.title.text == "London hour"
    assert "London" in hour_figure.data[0].hovertemplate


def test_kpi_values_equal_fixture_values() -> None:
    result = make_result("no_action", evs=6, worlds=20)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    kpis = result.plug_in_summary.kpis
    # Plug-ins per EV per week and median SoC at plug-in both follow the day
    # toggle (goal review N12/review finding), default Weekday here; only
    # "Plug-ins below 10% SoC" stays the fleet-wide "all" figure.
    weekday = kpis.loc[kpis["day_type"].eq("weekday")].set_index("metric")
    expected_plug_ins = f"{weekday.loc['plug_ins_per_ev_per_week', 'p50']:.1f}"
    expected_soc = f"{weekday.loc['median_plug_in_soc_percent', 'p50']:.1f}%"

    tiles = kpi_calls(st)
    values = {tile[0]: tile[1] for tile in tiles}
    # Goal review item 8: per-EV, not the raw fleet total; no CNZ figure
    # exists for this metric (report p.13 gives SoC and hour context only).
    # The day type is each tile's context line, not its label (polish plan G7).
    assert values["Plug-ins per EV per week"] == expected_plug_ins
    assert values["Median SoC at plug-in"] == expected_soc
    contexts = {tile[0]: tile[3] for tile in tiles}
    assert contexts["Median SoC at plug-in"] == "Weekday · CNZ 52%"
    assert contexts["Plug-ins per EV per week"] == "Weekday"


def test_median_soc_tile_follows_the_day_toggle_like_overview_does() -> None:
    # Review finding: this tile was pinned to "all" while Overview's
    # same-named tile followed its own day-type toggle, so the two pages
    # silently disagreed. It must now track this page's own toggle and say
    # which day type, the same fix as Overview's tile.
    result = make_result("no_action", evs=6, worlds=20)
    st = RecordingStreamlit(day="Weekend")

    render_plug_ins(st, result)

    weekend = result.plug_in_summary.kpis.loc[
        result.plug_in_summary.kpis["day_type"].eq("weekend")
    ].set_index("metric")
    expected_soc = f"{weekend.loc['median_plug_in_soc_percent', 'p50']:.1f}%"
    expected_plug_ins = f"{weekend.loc['plug_ins_per_ev_per_week', 'p50']:.1f}"

    tiles = kpi_calls(st)
    values = {tile[0]: tile[1] for tile in tiles}
    contexts = {tile[0]: tile[3] for tile in tiles}
    assert values["Median SoC at plug-in"].startswith(expected_soc)
    assert contexts["Median SoC at plug-in"].startswith("Weekend")
    # Plug-ins per EV per week also follows the toggle (goal review N12);
    # only the low-battery tile stays the fleet-wide figure.
    assert values["Plug-ins per EV per week"] == expected_plug_ins
    assert contexts["Plug-ins per EV per week"] == "Weekend"
    assert contexts["Low-battery plug-ins"].startswith("Below 10% SoC")


class _FakeResult:
    """Bare stand-in with just the attributes ``_render_kpi_row`` reads."""

    def __init__(self, kpis: pd.DataFrame) -> None:
        self.plug_in_summary = type("_Summary", (), {"kpis": kpis})()
        self.assumptions = ()


def test_kpi_falls_back_to_all_days_when_the_data_has_no_by_day_row() -> None:
    # Goal review N12: plug-ins per EV per week (and median SoC at plug-in)
    # follow the Weekday/Weekend toggle when the kpis frame has a by-day
    # row for them; when it does not, the tile must fall back to the
    # fleet-wide "all" figure and say so in its label, rather than silently
    # showing a day-specific-looking label next to an all-days number.
    kpis = pd.DataFrame(
        [
            {
                "day_type": "all",
                "metric": "plug_ins_per_ev_per_week",
                "p50": 3.2,
                "unit": "plug-ins per EV",
            },
            {
                "day_type": "all",
                "metric": "median_plug_in_soc_percent",
                "p50": 55.0,
                "unit": "percent",
            },
            {
                "day_type": "all",
                "metric": "share_below_10_percent_soc",
                "p50": 0.02,
                "unit": "fraction",
            },
        ]
    )
    result = _FakeResult(kpis)
    st = RecordingStreamlit(day="Weekday")

    _render_kpi_row(st, result, day_type="weekday", day_label="Weekday")

    tiles = kpi_calls(st)
    values = {tile[0]: tile[1] for tile in tiles}
    contexts = {tile[0]: tile[3] for tile in tiles}
    assert values["Plug-ins per EV per week"] == "3.2"
    assert values["Median SoC at plug-in"] == "55.0%"
    assert contexts["Plug-ins per EV per week"] == "All days"
    assert contexts["Median SoC at plug-in"].startswith("All days")
    # Not a toggle-following metric: its context names its threshold instead.
    assert contexts["Low-battery plug-ins"].startswith("Below 10% SoC")


def test_weekday_marker_only_shown_on_weekday() -> None:
    result = make_result("no_action", evs=6, worlds=10)

    weekday_st = RecordingStreamlit(day="Weekday")
    render_plug_ins(weekday_st, result)
    weekday_hour_figure = calls_for(weekday_st, "plotly_chart")[0][0][0]
    assert any(
        "CNZ weekday peak" in (shape.text or "") for shape in weekday_hour_figure.layout.annotations
    )

    weekend_st = RecordingStreamlit(day="Weekend")
    render_plug_ins(weekend_st, result)
    weekend_hour_figure = calls_for(weekend_st, "plotly_chart")[0][0][0]
    assert not weekend_hour_figure.layout.annotations


def test_soc_chart_always_shows_cnz_median_marker() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    soc_figure = calls_for(st, "plotly_chart")[1][0][0]
    assert any(
        "CNZ observed median" in (shape.text or "") for shape in soc_figure.layout.annotations
    )


def test_cnz_annotations_sit_inside_the_plot_not_clipped_above_it() -> None:
    # B2: add_vline's default annotation anchors to the axis's "y domain" top
    # edge with yanchor="bottom", rendering above the plot where the
    # template's 8 px top margin clipped it. Both CNZ labels must instead sit
    # inside the plot, paper-anchored at y <= 0.95.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit(day="Weekday")

    render_plug_ins(st, result)

    charts = calls_for(st, "plotly_chart")
    hour_figure, soc_figure = charts[0][0][0], charts[1][0][0]
    for figure, needle in ((hour_figure, "CNZ weekday peak"), (soc_figure, "CNZ observed median")):
        annotation = next(a for a in figure.layout.annotations if needle in (a.text or ""))
        assert annotation.yref == "paper"
        assert annotation.y <= 0.95


def test_cnz_weekday_annotation_anchors_away_from_the_near_edge() -> None:
    # Blocking review finding: left-anchored, the label at cnz_hour=18 (axis
    # range [-0.5, 23.5]) grew off the plot's right edge and clipped at both
    # 1440 and 390 px. cnz_hour sits closer to the axis's right edge than its
    # left, so the annotation must anchor right -- growing into the wider
    # left-hand room -- and its anchored (text-side) edge must stay inside
    # the axis range rather than spilling past either bound.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit(day="Weekday")

    render_plug_ins(st, result)

    hour_figure = calls_for(st, "plotly_chart")[0][0][0]
    annotation = next(
        a for a in hour_figure.layout.annotations if "CNZ weekday peak" in (a.text or "")
    )
    x_min, x_max = hour_figure.layout.xaxis.range
    assert (annotation.x - x_min) > (x_max - annotation.x)  # closer to the right edge
    assert annotation.xanchor == "right"
    assert x_min <= annotation.x <= x_max


def test_data_and_definition_tables_carry_renamed_columns() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    tables = [args[0] for args, _ in calls_for(st, "dataframe")]
    # Hour histogram, SoC histogram, two heatmaps, sampled plug-in events.
    assert len(tables) == 5
    assert "London hour" in tables[0].columns
    assert "Share mean" in tables[0].columns
    assert "SoC bin lower (%)" in tables[1].columns
    assert "Share mean" in tables[1].columns
    for table in tables[:2]:
        assert isinstance(table, pd.DataFrame)
        assert "day_type" not in table.columns
    for table in tables[2:4]:
        assert {"Weekday", "London time", "P50"} <= set(table.columns)
        assert "local_half_hour" not in table.columns
    assert "weekday" not in tables[2].columns


def test_histogram_bars_use_share_mean_not_median() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    hour_bands = result.plug_in_summary.hour_bands
    weekday = hour_bands.loc[hour_bands["day_type"].eq("weekday")].sort_values("local_hour")
    hour_figure = calls_for(st, "plotly_chart")[0][0][0]
    np.testing.assert_allclose(
        np.asarray(hour_figure.data[0].y, dtype=float), weekday["share_mean"].to_numpy() * 100.0
    )


def test_sampled_plug_in_events_expander_labels_the_sample() -> None:
    # worlds > 10 so plug_in_events is a strict subset (contract v2 section
    # 3.9: at most 10 worlds), never the source of all-world statistics.
    result = make_result("no_action", evs=6, worlds=15)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    sampled_ids = result.sampled_world_ids
    assert len(sampled_ids) == 10
    assert len(sampled_ids) < result.world_count
    expander_titles = [args[0] for args, _ in calls_for(st, "expander")]
    assert (
        f"Plug-in events (sample of {len(sampled_ids)} of {result.world_count} simulated weeks)"
        in (expander_titles)
    )
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    sample_caption = next(c for c in captions if c.startswith("Sampled weeks"))
    assert all(str(world_id) in sample_caption for world_id in sampled_ids)
    assert set(result.plug_in_events["world_id"]) == set(sampled_ids)

    tables = [args[0] for args, _ in calls_for(st, "dataframe")]
    events_table = tables[-1]
    assert len(events_table) == len(result.plug_in_events)
    assert "Simulated week" in events_table.columns
    assert "EV" in events_table.columns


def test_sampled_events_download_matches_the_table() -> None:
    # Per-agent export (the brief: "this is the kind of data we want to get
    # out for each agent"). The button must read the same built table as the
    # dataframe above it, so a download can never show different rows or
    # column names than what the viewer already saw on screen.
    result = make_result("no_action", evs=6, worlds=15)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    events_table = [args[0] for args, _ in calls_for(st, "dataframe")][-1]
    downloads = calls_for(st, "download_button")
    assert len(downloads) == 1
    args, kwargs = downloads[0]
    assert args[0] == "Download plug-in events (CSV)"
    assert kwargs["key"] == "plug-ins-events-download"
    assert kwargs["mime"] == "text/csv"
    assert (
        kwargs["file_name"]
        == f"axle_plug_in_events_{result.study_start_local_date:%Y-%m-%d}_seed{result.seed}.csv"
    )

    written = pd.read_csv(io.StringIO(kwargs["data"]))
    assert list(written.columns) == list(events_table.columns)
    assert len(written) == len(events_table) == len(result.plug_in_events)

    # The help tooltip names the sampled-week count and total, not a snake_case
    # or internal citation (copy rules exempt help text, but this app's own
    # style still keeps it plain prose).
    sampled_ids = result.sampled_world_ids
    assert str(len(sampled_ids)) in kwargs["help"]
    assert str(result.world_count) in kwargs["help"]


def test_whiskers_use_muted_ink_not_the_gridline_colour() -> None:
    # Goal review item 12: SERIES_COLOURS["grid"] is tuned to be nearly
    # invisible on the dark canvas, which made the whiskers themselves
    # nearly invisible; they must use the visible MUTED_INK token instead.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    charts = calls_for(st, "plotly_chart")
    hour_figure, soc_figure = charts[0][0][0], charts[1][0][0]
    assert hour_figure.data[0].error_y.color == MUTED_INK
    assert soc_figure.data[0].error_y.color == MUTED_INK


def test_zero_below_10_percent_reads_by_construction() -> None:
    # The no-stranding top-up rule (decision 0004 items 32, 37, 41) keeps
    # arrival, and so plug-in, SoC at or above 10% by construction; the
    # fixture's toy behaviour never drops below ~15%, so this reads 0.0 in
    # every case -- the tile must say so plainly, not compare it to CNZ's 3%
    # as if it were a modelled finding.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    kpis = result.plug_in_summary.kpis
    fleet = kpis.loc[kpis["day_type"].eq("all")].set_index("metric")
    assert abs(fleet.loc["share_below_10_percent_soc", "p50"]) < 1e-9

    tiles = kpi_calls(st)
    tile = next(tile for tile in tiles if tile[0] == "Low-battery plug-ins")
    # A number in the value, the prose in its context line (polish plan G5).
    assert tile[1] == "0%"
    assert tile[3] == "Below 10% SoC, by construction"
    assert "top-up rule" in tile[4]
    assert "CNZ" not in tile[1]


def test_soc_chart_annotates_the_at_target_bin() -> None:
    # Goal review item 4: the 80-85% bin holds EVs that plugged in already
    # at or near the 80% preferred target, so a large bar there needs its
    # own explanation, not to read as an unflagged realism finding.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    soc_figure = calls_for(st, "plotly_chart")[1][0][0]
    assert any(
        "At target" in (annotation.text or "") for annotation in soc_figure.layout.annotations
    )


def test_cnz_annotations_carry_a_background_chip_off_the_bar() -> None:
    # Goal review item 12: "move the CNZ hour label off the bar" -- a
    # background chip keeps the label legible regardless of the bar (or
    # whisker) drawn underneath it, since it sits at the chart's own peak
    # hour by construction (CNZ's reported mode and this model's peak both
    # land near 18:00).
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit(day="Weekday")

    render_plug_ins(st, result)

    hour_figure = calls_for(st, "plotly_chart")[0][0][0]
    annotation = next(
        a for a in hour_figure.layout.annotations if "CNZ weekday peak" in (a.text or "")
    )
    assert annotation.bgcolor is not None


def test_heatmaps_are_weekday_by_half_hour_monday_first() -> None:
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    for args, _ in calls_for(st, "plotly_chart")[2:]:
        trace = args[0].data[0]
        assert trace.type == "heatmap"
        assert list(trace.y) == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        assert list(trace.x) == list(range(48))
        assert np.asarray(trace.z).shape == (7, 48)
        assert args[0].layout.yaxis.autorange == "reversed"
        assert "P10–P90" in trace.hovertemplate
        assert args[0].layout.hovermode == "closest"  # clock time, not the index


def test_heatmaps_share_one_colour_scale_without_teal() -> None:
    # Plan D-1: one colour scale, so a colour means the same rate on both;
    # polish plan G4: teal marks the smart path only.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    traces = [args[0].data[0] for args, _ in calls_for(st, "plotly_chart")[2:]]
    top = 100.0 * max(
        result.plug_in_half_hour_heatmap["p50"].max(), result.plug_out_heatmap["p50"].max()
    )
    for trace in traces:
        assert (trace.zmin, trace.zmax) == (0.0, pytest.approx(top))
        colours = [colour.upper() for _, colour in trace.colorscale]
        assert colours == [colour.upper() for _, colour in SEQUENTIAL]
        assert SERIES_COLOURS["selected"].upper() not in colours


def test_heatmap_values_equal_the_fixture() -> None:
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    plug_in_trace, plug_out_trace = (
        args[0].data[0] for args, _ in calls_for(st, "plotly_chart")[2:]
    )
    for trace, frame in (
        (plug_in_trace, result.plug_in_half_hour_heatmap),
        (plug_out_trace, result.plug_out_heatmap),
    ):
        cell = frame.loc[frame["weekday"].eq(0) & frame["local_half_hour"].eq(37)]
        np.testing.assert_allclose(np.asarray(trace.z)[0, 37], cell["p50"].iat[0] * 100.0)
        np.testing.assert_allclose(
            np.asarray(trace.customdata)[0, 37], cell[["p10", "p90"]].iloc[0] * 100.0
        )


def test_heatmap_toggle_shows_one_heatmap_at_a_time() -> None:
    # Plan D-1: a phone can show one heatmap at full width.
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit(choices={"plug-ins-heatmap-choice": "Departure"})

    render_plug_ins(st, result)

    charts = calls_for(st, "plotly_chart")
    assert len(charts) == 3
    assert "unplugs" in charts[2][0][0].data[0].hovertemplate


def test_day_type_control_is_shared_and_offers_all_days() -> None:
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit(day="All")

    render_plug_ins(st, result)

    day_control = next(
        kwargs for _, kwargs in calls_for(st, "segmented_control") if kwargs["key"] == DAY_TYPE_KEY
    )
    assert day_control["options"] == ["weekday", "weekend", "all"]
    assert day_control["persist_state"] == "session"
    soc = next(tile for tile in kpi_calls(st) if tile[0] == "Median SoC at plug-in")
    assert soc[3].startswith("All")


def test_no_pie_chart_types_anywhere_on_the_page() -> None:
    result = make_result("action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_plug_ins(st, result)

    for args, _ in calls_for(st, "plotly_chart"):
        figure = args[0]
        assert all(trace.type != "pie" for trace in figure.data)


def test_events_title_says_all_weeks_when_every_week_is_shown() -> None:
    # Goal review item 20: no "sample of 5 of 5".
    from axle_studio.ui.views.plug_ins import _events_title

    assert _events_title(5, 5) == "Plug-in events (all 5 simulated weeks)"
    assert _events_title(10, 100) == "Plug-in events (sample of 10 of 100 simulated weeks)"


def test_sampled_events_table_shows_plug_out_in_london_time_not_utc() -> None:
    # Clarity critique: plug-in was already London time, plug-out stayed
    # UTC on the same row, a summer-time-sized (1 h) mismatch that read as
    # a bug. Both must read in the same clock.
    from axle_studio.ui.views.plug_ins import _sampled_events_table

    plug_in_utc = pd.to_datetime(["2026-07-06 17:00"], utc=True)
    plug_out_utc = pd.to_datetime(["2026-07-06 22:00"], utc=True)  # BST: 23:00 London
    events = pd.DataFrame(
        {
            "world_id": [1],
            "unit_id": ["ev-0000"],
            "cohort_id": ["commuter"],
            "plug_in_utc": plug_in_utc,
            "plug_in_london": plug_in_utc.tz_convert("Europe/London"),
            "plug_in_soc_percent": [40.0],
            "plug_out_utc": plug_out_utc,
            "plug_out_soc_percent": [80.0],
            "home_import_kwh": [10.0],
            "battery_added_kwh": [9.5],
            "still_plugged_at_horizon_end": [False],
        }
    )

    table = _sampled_events_table(events)

    assert "Plug-out (UTC)" not in table.columns
    assert "Plug-out (London)" in table.columns
    assert table["Plug-out (London)"].iloc[0] == pd.Timestamp(
        "2026-07-06 23:00", tz="Europe/London"
    )


def test_sampled_events_table_leaves_a_still_open_session_plug_out_blank() -> None:
    from axle_studio.ui.views.plug_ins import _sampled_events_table

    plug_in_utc = pd.to_datetime(["2026-01-06 17:00"], utc=True)
    events = pd.DataFrame(
        {
            "world_id": [1],
            "unit_id": ["ev-0000"],
            "cohort_id": ["commuter"],
            "plug_in_utc": plug_in_utc,
            "plug_in_london": plug_in_utc.tz_convert("Europe/London"),
            "plug_in_soc_percent": [40.0],
            "plug_out_utc": pd.to_datetime([pd.NaT], utc=True),
            "plug_out_soc_percent": [np.nan],
            "home_import_kwh": [10.0],
            "battery_added_kwh": [9.5],
            "still_plugged_at_horizon_end": [True],
        }
    )

    table = _sampled_events_table(events)

    assert pd.isna(table["Plug-out (London)"].iloc[0])
