"""Drivers ▸ Archetypes renders the six small multiples and comparison table."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from fixtures.result_fixture import make_result

from axle_studio.ui.style import CHART_HEIGHTS, UNAVAILABLE
from axle_studio.ui.views.archetypes import (
    _PANEL_TITLE_MAX_CHARS,
    _SMALL_MULTIPLE_HEIGHT,
    _panel_title,
    _soc_week_rows,
    render_archetypes,
)


class _Container:
    def __enter__(self) -> "_Container":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class RecordingStreamlit:
    def __init__(self, *, selectbox_index: int = 0) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self._selectbox_index = selectbox_index

    def __getattr__(self, name: str):
        def record(*args: object, **kwargs: object) -> object:
            self.calls.append((name, args, kwargs))
            if name == "columns":
                count = args[0] if args else kwargs.get("spec")
                n = count if isinstance(count, int) else len(count)
                # Columns share this fake's call log, so a selectbox drawn
                # into a column is recorded like any other call.
                return [self for _ in range(n)]
            if name == "expander":
                return _Container()
            if name == "selectbox":
                options = kwargs["options"]
                return options[min(self._selectbox_index, len(options) - 1)]
            return None

        return record

    def __enter__(self) -> "RecordingStreamlit":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def calls_for(st: RecordingStreamlit, name: str) -> list[tuple[tuple, dict]]:
    return [(args, kwargs) for call_name, args, kwargs in st.calls if call_name == name]


@pytest.mark.parametrize("model", ["action", "no_action"])
def test_renders_both_models_without_model_calls(model: str) -> None:
    result = make_result(model, evs=6, worlds=12)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    # Plugged-in small multiples + SoC small multiples (item 43) + the
    # one-archetype expander.
    assert len(calls_for(st, "plotly_chart")) == 3
    assert not calls_for(st, "button")


def test_five_ev_case_lists_the_empty_archetype_as_no_evs_sampled() -> None:
    result = make_result("no_action", evs=5, worlds=8)
    st = RecordingStreamlit()

    render_archetypes(st, result)  # must not raise (chart audit B1)

    tables = [args[0] for args, _ in calls_for(st, "dataframe")]
    comparison = tables[0]
    assert len(comparison) == 6
    empty_rows = comparison.loc[comparison["EVs in run"].eq(0)]
    assert len(empty_rows) == 1
    # Item 13's CNZ column is excluded: it is a fleet-wide reported figure
    # repeated on every row, not a per-cohort figure this run measured, so an
    # empty cohort still shows it rather than "No EVs sampled".
    per_cohort_columns = ["Archetype", "Source share", "EVs in run", "CNZ median plug-in SoC"]
    assert (empty_rows.drop(columns=per_cohort_columns) == "No EVs sampled").all(axis=None)
    assert empty_rows["CNZ median plug-in SoC"].eq("52%").all()


@pytest.mark.parametrize("clock_change", ["autumn", "spring"])
def test_clock_change_week_renders(clock_change: str) -> None:
    result = make_result("action", evs=6, worlds=6, clock_change=clock_change)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    assert len(calls_for(st, "plotly_chart")) == 3


def test_chart_heights_and_legend_placement() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    figures = [args[0] for args, _ in calls_for(st, "plotly_chart")]
    assert figures[0].layout.height == _SMALL_MULTIPLE_HEIGHT
    assert figures[1].layout.height == _SMALL_MULTIPLE_HEIGHT  # the new SoC row (item 43)
    assert figures[2].layout.height == CHART_HEIGHTS["time_series"]
    for figure in figures:
        # Polish plan G2: legend anchored to the figure's own bottom edge.
        assert figure.layout.template.layout.legend.yref == "container"
        assert figure.layout.template.layout.legend.y == 0


def test_small_multiples_panel_height_matches_design_table() -> None:
    # Polish plan G2: 3 x 180 + 2 x 36 = 612
    assert _SMALL_MULTIPLE_HEIGHT == 3 * CHART_HEIGHTS["small_multiple_panel"] + 2 * 36 == 612


def test_week_expander_axis_is_london() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    week_figure = calls_for(st, "plotly_chart")[2][0][0]
    assert week_figure.layout.xaxis.ticktext  # london_time_axis populated tickvals/ticktext


def test_comparison_table_key_numbers_equal_fixture_values() -> None:
    result = make_result("no_action", evs=12, worlds=20)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    comparison = [args[0] for args, _ in calls_for(st, "dataframe")][0]
    cohort_summary = result.cohort_summary
    assert list(comparison["EVs in run"]) == list(cohort_summary["ev_count"])
    assert list(comparison["Archetype"]) == list(cohort_summary["cohort_label"])


def test_comparison_table_has_cnz_median_soc_column_from_assumptions() -> None:
    # Goal review item 13: the lens question is "do the six archetypes
    # reproduce what CNZ observed", so the table needs CNZ's own figure
    # beside the run's own "Median plug-in SoC (P50)" column, read from
    # result.assumptions (contract v2 8.1) like every other CNZ reference.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    comparison = [args[0] for args, _ in calls_for(st, "dataframe")][0]
    assert "CNZ median plug-in SoC" in comparison.columns
    assert comparison["CNZ median plug-in SoC"].eq("52%").all()  # the fixture's own CNZ record

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert any("CNZ column" in caption and "daily" in caption for caption in captions)


def test_comparison_table_cnz_column_falls_back_to_unavailable_without_the_assumption() -> None:
    result = make_result("no_action", evs=6, worlds=10)
    result = replace(
        result,
        assumptions=tuple(
            record
            for record in result.assumptions
            if record.name != "cnz_median_plug_in_soc_percent"
        ),
    )
    st = RecordingStreamlit()

    render_archetypes(st, result)

    comparison = [args[0] for args, _ in calls_for(st, "dataframe")][0]
    assert comparison["CNZ median plug-in SoC"].eq(UNAVAILABLE).all()


def test_small_multiples_band_and_median_are_named_once_in_legend() -> None:
    # Review finding: six unnamed panels left the P10-P90 band with no legend
    # entry anywhere (decision 0004 items 22, 27 require the band and centre
    # line to be named). One legend entry speaks for all six panels.
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    # Polish plan G3: band and median are one legend entry ("Unmanaged"),
    # shown once for all six panels; the band traces never list themselves.
    figure = calls_for(st, "plotly_chart")[0][0][0]
    listed = [trace.name for trace in figure.data if trace.showlegend is not False]
    assert listed == ["Unmanaged"]
    band_traces = [trace for trace in figure.data if trace.fill == "tonexty"]
    assert band_traces
    assert all(trace.legendgroup == "Unmanaged" for trace in figure.data)

    # Caption 0 is the CNZ note under the table, which now comes first
    # (polish plan G9); caption 1 is this chart's.
    caption = calls_for(st, "caption")[1][0][0]
    assert "Median and P10–P90 across simulated weeks" in caption


def test_panel_title_shortens_only_labels_longer_than_the_panel() -> None:
    # Review finding: "Intelligent Octopus average" clipped mid-word at
    # 390 px ("Intelligent Octopus averag..."); cut at the last whole word.
    assert _panel_title("Infrequent charging") == "Infrequent charging"
    assert _panel_title("Intelligent Octopus average") == "Intelligent Octopus…"


def test_small_multiples_hide_ticks_above_the_bottom_row_and_share_them() -> None:
    # Review finding: unshared x-axes let each panel auto-pick its own ticks.
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    bottom_row = [figure.layout.xaxis5, figure.layout.xaxis6]
    upper_rows = [
        figure.layout.xaxis,
        figure.layout.xaxis2,
        figure.layout.xaxis3,
        figure.layout.xaxis4,
    ]
    assert all(axis.showticklabels is False for axis in upper_rows)
    assert all(axis.tickvals == (0, 12, 24, 36) for axis in bottom_row)


def test_small_multiples_top_margin_clears_the_top_row_titles() -> None:
    # B5: make_subplots anchors each row's title at that row's domain top with
    # yanchor="bottom", so the top row's title sits at paper y=1 and grows
    # upward; the template's 8 px default top margin clipped it.
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert figure.layout.margin.t >= 24


def test_small_multiples_x_axis_ticks_show_a_time_not_a_half_hour_index() -> None:
    # O4: the bare half-hour index (e.g. "36") read as a meaningless number.
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    bottom_row = [figure.layout.xaxis5, figure.layout.xaxis6]
    # Milestone verifier, decision 0004 item 26: four ticks, not five, and
    # tickangle=0 -- see test_small_multiples_x_axis_ticks_are_not_rotated.
    assert all(axis.ticktext == ("00:00", "06:00", "12:00", "18:00") for axis in bottom_row)


def test_small_multiples_hover_shows_a_time_not_a_half_hour_index() -> None:
    # Clarity critique: hover said "Half-hour 36" even though the x-axis
    # ticks beside it already read as a clock time (O4 above).
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    line = next(trace for trace in figure.data if trace.hovertemplate is not None)
    assert line.hovertemplate.startswith("%{customdata}")
    assert "Half-hour" not in line.hovertemplate
    assert all(":" in label for label in line.customdata)


def test_small_multiples_x_axis_ticks_are_not_rotated() -> None:
    # Milestone verifier, decision 0004 item 26: left to Plotly's default,
    # a narrow 390 px panel auto-rotated these five labels, and the rotated
    # text collided across panels and with the shared title below the grid.
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    bottom_row = [figure.layout.xaxis5, figure.layout.xaxis6]
    assert all(axis.tickangle == 0 for axis in bottom_row)


def test_small_multiples_show_one_centred_x_axis_title() -> None:
    # Review finding: a title copy on each bottom-row column collided at
    # 390 px (170 px panels, "London half-hour of day" repeated twice); a
    # follow-up review found the single-column fix left it off-centre under
    # the left panel only. It is now one annotation, centred under both
    # columns (paper x=0.5), not an axis title.
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert not any(axis.title.text for axis in figure.select_xaxes())
    titles = [a for a in figure.layout.annotations if a.text == "London half-hour of day"]
    assert len(titles) == 1
    assert titles[0].xref == "paper"
    assert titles[0].x == 0.5
    # Polish plan G2: anchored to the bottom panels' edge and shifted down a
    # fixed pixel row, so it clears the tick labels at any figure height; the
    # figure reserves its own bottom margin for ticks, this title and legend.
    assert (titles[0].yref, titles[0].y, titles[0].yanchor) == ("paper", 0, "top")
    assert titles[0].yshift == -24
    assert figure.layout.margin.b == 76


def test_panel_titles_use_the_body_font_size() -> None:
    # Review finding: make_subplots' default 16 px title font crowded the
    # y-tick labels of the panel above at 390 px.
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    # Excludes "No EVs sampled" and the shared x-axis caption below the grid,
    # neither of which is one of the six per-cohort subplot titles this test
    # covers.
    excluded = ("No EVs", "London half-hour of day")
    panel_titles = [
        a
        for a in figure.layout.annotations
        if a.text and not any(text in a.text for text in excluded)
    ]
    assert panel_titles
    assert all(a.font.size == 12 for a in panel_titles)


def test_full_six_cohort_run_shows_no_empty_panels() -> None:
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    figure = calls_for(st, "plotly_chart")[0][0][0]
    assert not figure.layout.annotations or all(
        "No EVs sampled" not in (a.text or "") for a in figure.layout.annotations if a.text
    )


def test_soc_row_shows_both_bands_named_in_legend_and_caption() -> None:
    # Decision 0004 item 43: an SoC row with both bands, each named.
    result = make_result("no_action", evs=6, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    soc_figure = calls_for(st, "plotly_chart")[1][0][0]
    listed = {trace.name for trace in soc_figure.data if trace.showlegend is not False}
    # Polish plan G3: the across-weeks band joins its median line's entry;
    # the across-EVs band has no line, so it keeps its own single entry.
    assert listed == {"Unmanaged", "SoC P5–P95 across EVs"}

    # The CNZ note under the table is caption 0; chart_block then emits two
    # captions per chart (the caption itself, then the "Data and definition"
    # expander's definition text), so the SoC chart's own caption is caption 3.
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    soc_caption = captions[3]
    assert "P10–P90 across simulated weeks" in soc_caption
    assert "P5–P95 across EVs" in soc_caption


def test_soc_row_band_values_equal_fixture_cohort_bands() -> None:
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit()

    cohort_id = next(
        row.cohort_id for row in result.cohort_summary.itertuples() if row.ev_count > 0
    )
    expected = _soc_week_rows(result, cohort_id)

    render_archetypes(st, result)

    soc_figure = calls_for(st, "plotly_chart")[1][0][0]
    week_band_trace = next(
        trace
        for trace in soc_figure.data
        if trace.name == "Unmanaged" and trace.fill == "tonexty" and trace.xaxis == "x"
    )
    np.testing.assert_allclose(
        np.asarray(week_band_trace.y, dtype=float), expected["p90"].to_numpy()
    )


def test_soc_week_rows_broadcasts_ev_band_from_average_day() -> None:
    # A plain lookup, not a recomputed percentile: every one of the 336 week
    # slots for a given local_half_hour must carry that half-hour's own
    # average_day_bands across-EVs low/high, unchanged.
    result = make_result("no_action", evs=6, worlds=10)
    cohort_id = next(
        row.cohort_id for row in result.cohort_summary.itertuples() if row.ev_count > 0
    )

    rows = _soc_week_rows(result, cohort_id)

    half_hours = result.study_slots.set_index("slot_index")["local_half_hour"]
    average_day = result.average_day_bands
    for _, row in rows.iterrows():
        half_hour = half_hours.loc[row["slot_index"]]
        expected = average_day.loc[
            average_day["group_id"].eq(cohort_id)
            & average_day["path_id"].eq("normal")
            & average_day["day_type"].eq("all")
            & average_day["metric"].eq("battery_soc_percent")
            & average_day["local_half_hour"].eq(half_hour)
        ].iloc[0]
        assert row["low"] == pytest.approx(expected["low"])
        assert row["high"] == pytest.approx(expected["high"])


def test_soc_row_five_ev_case_lists_no_evs_sampled_and_does_not_crash() -> None:
    # Chart audit B1, extended to the new row.
    result = make_result("no_action", evs=5, worlds=8)
    st = RecordingStreamlit()

    render_archetypes(st, result)  # must not raise

    soc_figure = calls_for(st, "plotly_chart")[1][0][0]
    assert any("No EVs sampled" in (a.text or "") for a in soc_figure.layout.annotations)


def test_soc_row_panel_titles_not_truncated() -> None:
    # Reuses _panel_title's own word-boundary shortening (B5/O4 precedent):
    # every panel title must equal that function's output for its cohort's
    # full label, never a raw mid-word clip.
    result = make_result("no_action", evs=12, worlds=10)
    st = RecordingStreamlit()

    render_archetypes(st, result)

    soc_figure = calls_for(st, "plotly_chart")[1][0][0]
    expected_titles = {_panel_title(label) for label in result.cohort_summary["cohort_label"]}
    panel_titles = {
        a.text for a in soc_figure.layout.annotations if a.text and a.text != "No EVs sampled"
    }
    assert panel_titles == expected_titles
    for title in panel_titles:
        assert not title.endswith("...")  # never a raw, mid-word clip
        assert len(title.rstrip("…")) <= _PANEL_TITLE_MAX_CHARS
