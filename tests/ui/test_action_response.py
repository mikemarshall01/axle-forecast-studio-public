"""Smart charging lens 2, Response (docs/DASHBOARD_DESIGN.md section 4.3)."""

from __future__ import annotations

import math
from datetime import date
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fixtures.result_fixture import make_result
from kpi_calls import kpi_calls
from streamlit.testing.v1 import AppTest
from test_copy_rules import APP_PATH, _edit, _go, _run, _small_draft

from axle_studio.model.forecast import ForecastResult, run_forecast_from_assumptions
from axle_studio.ui.style import (
    CHART_HEIGHTS,
    MUTED_INK,
    PATH_DISPLAY_ORDER,
    PATH_LABELS,
    PATH_STYLES,
    SERIES_COLOURS,
    UNAVAILABLE,
    hover_time_labels,
)
from axle_studio.ui.views import action_response
from axle_studio.ui.views.action_response import (
    _WEEKLY_ROW_LABELS,
    _increase_tile,
    _no_negative_zero,
    _signed,
    render_action_response,
)


class RecordingStreamlit:
    """Fake ``st``; ``segmented_control`` returns a configured metric choice."""

    def __init__(self, calls: list | None = None, *, metric: str | None = None) -> None:
        self.calls: list[tuple[str, tuple, dict]] = calls if calls is not None else []
        self._metric = metric

    def __getattr__(self, name: str):
        def record(*args: object, **kwargs: object) -> object:
            self.calls.append((name, args, kwargs))
            return self

        return record

    def __enter__(self) -> "RecordingStreamlit":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def columns(self, spec: int | list) -> tuple["RecordingStreamlit", ...]:
        count = spec if isinstance(spec, int) else len(spec)
        self.calls.append(("columns", (spec,), {}))
        return tuple(RecordingStreamlit(self.calls, metric=self._metric) for _ in range(count))

    def segmented_control(self, *args: object, **kwargs: object) -> object:
        self.calls.append(("segmented_control", args, kwargs))
        options = kwargs.get("options", [])
        return self._metric if self._metric in options else (options[0] if options else None)


def calls_for(st: RecordingStreamlit, name: str) -> list[tuple[tuple, dict]]:
    return [(args, kwargs) for call_name, args, kwargs in st.calls if call_name == name]


def figures_from(st: RecordingStreamlit) -> list:
    return [args[0] for args, _ in calls_for(st, "plotly_chart")]


def _parse_number(text: str) -> float:
    if text == UNAVAILABLE:
        return float("nan")
    return float(text.replace("−", "-").split(" ")[0].replace(",", ""))


def test_weekly_change_table_equals_fixture_difference_weekly_bands() -> None:
    # Polish plan G9: four tiles per lens; the weekly changes are a visible
    # table under them, not three more tiles.
    result = make_result("action")
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    assert len(kpi_calls(st)) == 4
    weekly = result.difference_weekly_bands.set_index("metric")
    table = next(
        args[0]
        for args, _ in calls_for(st, "dataframe")
        if "Change, smart − unmanaged" in args[0].columns
    )
    assert list(table["Change, smart − unmanaged"]) == list(_WEEKLY_ROW_LABELS.values())
    for (_, row), metric_name in zip(table.iterrows(), _WEEKLY_ROW_LABELS, strict=True):
        assert round(_parse_number(row["P50"])) == round(weekly.loc[metric_name, "p50"])
        assert row["P50"].endswith(weekly.loc[metric_name, "unit"])


def test_headline_tile_row_shows_share_weekly_peak_and_median_cut_and_rise() -> None:
    # Goal review action 8 and N3 (decision 0004 items 4, 20, 43, 48-49):
    # share moved, each week's own smart peak, and the largest median
    # cut/rise in a half-hour, all looked up from world-first summaries.
    result = make_result("action")
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    share = result.smart_charging_summary.set_index("metric").loc["moved_home_import_share"]
    peak = result.weekly_peak_summary.set_index("path_id").loc["selected"]
    diff = result.difference_bands.loc[result.difference_bands["metric"].eq("home_import_kw")]
    cut = diff.loc[diff["p50"].idxmin()]
    rise = diff.loc[diff["p50"].idxmax()]

    tiles = kpi_calls(st)[:4]
    share_label, share_value, _, _, _ = tiles[0]
    assert share_label == "Share of home energy moved"
    assert share_value.endswith("%") and " " not in share_value  # "82%", polish plan G6
    assert _parse_number(share_value.rstrip("%")) == pytest.approx(round(100 * share["p50"]))

    peak_label, peak_value, peak_unit, peak_context, peak_help = tiles[1]
    assert peak_label == "Weekly smart peak"
    assert _parse_number(peak_value) == pytest.approx(round(peak["p50"]))
    assert peak_unit == "kW"
    # The ratio is a neutral context line, not an st.metric delta arrow.
    assert peak_context == f"×{peak['ratio_to_normal_p50']:.1f} of unmanaged peak"
    assert f"×{peak['ratio_to_normal_p10']:.1f}" in peak_help
    assert f"{round(peak['p90']):,} kW" in peak_help

    cut_label, cut_value, _, _, cut_help = tiles[2]
    assert cut_label == "Largest half-hour cut"
    assert _parse_number(cut_value) == pytest.approx(round(cut["p50"]))
    assert "not a weekly peak" in cut_help

    rise_label, rise_value, _, rise_context, _ = tiles[3]
    assert rise_label == "Largest half-hour rise"
    assert rise_context == "Median, smart − unmanaged"
    assert _parse_number(rise_value) == pytest.approx(round(rise["p50"]))


def test_increase_tile_relabels_honestly_when_no_slot_increases() -> None:
    # Goal review item 19: the largest smart-minus-unmanaged P50 can still be
    # <= 0 (every slot cut home import, some by more than others), and that
    # is really the smallest cut, not an increase. Showing it under
    # an "increase" label with a negative number would read as an increase
    # that never happened.
    column = RecordingStreamlit()
    row = pd.Series(
        {
            "p10": -3.0,
            "p50": -1.0,
            "p90": -0.2,
            "world_count": 500,
            "interval_start_utc": pd.Timestamp("2026-09-28T18:00:00Z"),
        }
    )
    _increase_tile(column, row)

    label, value, _, _, _ = kpi_calls(column)[0]
    assert label == "No half-hour rise"
    assert "−" not in value and "-" not in value


def test_increase_tile_keeps_the_rise_label_when_a_slot_increases() -> None:
    column = RecordingStreamlit()
    row = pd.Series(
        {
            "p10": 0.5,
            "p50": 2.0,
            "p90": 3.5,
            "world_count": 500,
            "interval_start_utc": pd.Timestamp("2026-09-28T18:00:00Z"),
        }
    )
    _increase_tile(column, row)

    label, value, _, _, _ = kpi_calls(column)[0]
    assert label == "Largest half-hour rise"
    assert _parse_number(value) == pytest.approx(2.0)


def test_home_import_chart_matches_fleet_interval_bands_both_paths() -> None:
    result = make_result("action")
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    bands = result.fleet_interval_bands
    figure = figures_from(st)[0]
    for path, label in (("normal", "Unmanaged"), ("selected", "Smart")):
        rows = bands.loc[
            bands["metric"].eq("home_import_kw") & bands["path_id"].eq(path)
        ].sort_values("slot_index")
        trace = next(t for t in figure.data if t.name == label and t.showlegend)
        # P50 is rounded to the hovertemplate's 0 dp before plotting (chart
        # audit 2 follow-up, O1: a hover trace must never show "-0").
        assert np.allclose(trace.y, np.round(rows["p50"].to_numpy(), 0))
        assert trace.line.color == PATH_STYLES[path]["color"]


def test_difference_chart_matches_difference_bands_with_zero_line() -> None:
    result = make_result("action")
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    diff_rows = result.difference_bands.loc[
        result.difference_bands["metric"].eq("home_import_kw")
    ].sort_values("slot_index")
    figure = figures_from(st)[1]
    trace = next(t for t in figure.data if t.name == "Smart minus unmanaged" and t.showlegend)
    assert np.allclose(trace.y, np.round(diff_rows["p50"].to_numpy(), 0))
    assert figure.layout.yaxis.zeroline is True
    # Chart audit 2, O6: the zero line must be visible against the grid, not
    # the same colour as it.
    assert figure.layout.yaxis.zerolinecolor == MUTED_INK
    assert figure.layout.yaxis.zerolinecolor != SERIES_COLOURS["grid"]


def test_signed_never_prints_negative_zero() -> None:
    # Chart audit 2, O1: a value that only rounds to zero at 0 dp (e.g. a
    # tiny negative kW difference) must not print "-0 kW".
    assert _signed(-0.3, "kW") == "0 kW"
    assert _signed(0.3, "kW") == "0 kW"
    assert _signed(-1.6, "kW") == "\u22122 kW"  # U+2212, polish plan G6
    assert _signed(1.6, "kW") == "+2 kW"
    assert _signed(float("nan"), "kW") == UNAVAILABLE


def test_metric_selector_offers_home_import_battery_soc_and_events() -> None:
    # Decision 0004 item 43: "Response no longer offers 'Plugged in'"; items
    # 55-58 add the scripted-event view as a third option.
    result = make_result("action")
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    segmented_calls = calls_for(st, "segmented_control")
    options = segmented_calls[0][1]["options"]
    assert options == ["Home import", "Battery SoC", "Events"]
    assert "Plugged in" not in options


def test_battery_soc_metric_axis_is_fixed_0_to_100() -> None:
    result = make_result("action")
    st = RecordingStreamlit(metric="Battery SoC")
    render_action_response(st, result)

    figure = figures_from(st)[0]
    assert figure.layout.yaxis.range == (0, 100)


def test_no_negative_zero_clears_the_sign_a_hovertemplate_would_show() -> None:
    # Chart audit 2 follow-up, O1 (BLOCKING): a raw float in (-0.5, 0) prints
    # as "-0" in a Plotly d3-format hovertemplate (%{y:,.0f}), even though
    # the same value displays as plain "0" everywhere else in the app. A P50
    # trace carrying a value like -0.224 must come out as a clean +0.0, not
    # a still-negative -0.0, before it ever reaches Plotly.
    rounded = _no_negative_zero(pd.Series([-0.224, -1.6, 1.6, 0.0]), decimals=0)
    assert list(rounded) == [0.0, -2.0, 2.0, 0.0]
    assert math.copysign(1.0, rounded.iloc[0]) == 1.0, "must be +0.0, not -0.0"
    # The exact failure mode: formatting a lingering -0.0 the way d3-format
    # would (":,.0f") prints the bare minus sign.
    assert f"{rounded.iloc[0]:,.0f}" == "0"


def test_heights_and_london_axis() -> None:
    result = make_result("action")
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    figures = figures_from(st)
    assert [figure.layout.height for figure in figures] == [
        CHART_HEIGHTS["time_series"],
        CHART_HEIGHTS["time_series"],
    ]
    ticktext = list(figures[0].layout.xaxis.ticktext)
    assert any(label.startswith("Mon 28") for label in ticktext)


def test_each_path_has_one_legend_entry_that_also_toggles_its_band() -> None:
    # Polish plan G3: band and line share a legend group; only the line is
    # listed, so the legend names the two paths, not four statistics.
    result = make_result("action")
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    paths_figure, difference_figure = figures_from(st)
    listed = [trace.name for trace in paths_figure.data if trace.showlegend is not False]
    assert listed == ["Unmanaged", "Smart"]
    for group in ("normal", "selected"):
        members = [trace for trace in paths_figure.data if trace.legendgroup == group]
        assert len(members) == 3  # lower bound, filled upper bound, median line
        assert [trace.fill for trace in members] == [None, "tonexty", None]
    listed = [trace.name for trace in difference_figure.data if trace.showlegend is not False]
    assert listed == ["Smart minus unmanaged"]


def test_unserved_travel_is_a_caption_check_not_a_kpi_tile() -> None:
    # Decision 0004 item 32: EVs never strand, so the weekly unserved-travel
    # difference would always be a zero KPI tile; it is reported as a caption
    # check instead (review, 28 September 2026).
    result = make_result("action")
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    assert "unserved_travel_kwh" not in _WEEKLY_ROW_LABELS
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert any("Unserved travel" in text for text in captions)


def test_difference_chart_unit_is_percentage_points_for_battery_soc() -> None:
    # Review: a difference of a percent metric is measured in percentage
    # points, never "%" (a level, not a point-difference). battery_soc_percent
    # already carries "percentage points" in ``difference_bands`` (contract 4.1).
    st = RecordingStreamlit(metric="Battery SoC")
    render_action_response(st, make_result("action"))

    figure = figures_from(st)[1]
    assert figure.layout.yaxis.title.text == "percentage points"
    trace = next(t for t in figure.data if t.name == "Smart minus unmanaged" and t.showlegend)
    assert "percentage points" in trace.hovertemplate
    titles = [args[0] for args, _ in calls_for(st, "markdown")]
    chart_block_title = next(
        title for title in titles if title.startswith("**Smart minus unmanaged")
    )
    assert "percentage points" in chart_block_title
    assert not chart_block_title.rstrip("*").endswith("%)")


def test_summary_sentence_is_the_shared_smart_charging_sentence() -> None:
    # Decision 0004 item 38: no "No action selected" state any more.
    result = make_result("action")
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    sentence = calls_for(st, "markdown")[0][0][0]
    assert sentence.startswith("Smart charging:")


def test_render_both_models_from_fixture_without_crashing() -> None:
    action_st = RecordingStreamlit(metric="Home import")
    render_action_response(action_st, make_result("action"))
    assert figures_from(action_st)

    no_action_st = RecordingStreamlit(metric="Home import")
    render_action_response(no_action_st, make_result("no_action"))
    assert figures_from(no_action_st) == []
    assert calls_for(no_action_st, "info")[0][0] == ("This result has no Axle action.",)


def test_view_makes_no_model_calls() -> None:
    source = Path(action_response.__file__).read_text(encoding="utf-8")
    assert "axle_studio.model" not in source
    assert "run_forecast" not in source


def test_hover_header_shows_london_time_not_raw_utc() -> None:
    # Goal review action 20: hover header in London format on Response. The
    # x-axis stays UTC (clock-change safe, style.london_time_axis); each
    # point's own hover leads with the London-formatted time instead, via
    # customdata (style.hover_time_labels), rather than Plotly's default
    # header showing the raw UTC x-value.
    result = make_result("action")
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    bands = result.fleet_interval_bands
    normal_rows = bands.loc[
        bands["metric"].eq("home_import_kw") & bands["path_id"].eq("normal")
    ].sort_values("slot_index")
    expected_hover = hover_time_labels(normal_rows["interval_start_utc"])

    paths_figure, difference_figure = figures_from(st)
    normal_trace = next(t for t in paths_figure.data if t.name == "Unmanaged" and t.showlegend)
    assert "%{customdata}" in normal_trace.hovertemplate
    assert list(normal_trace.customdata) == expected_hover

    diff_trace = next(
        t for t in difference_figure.data if t.name == "Smart minus unmanaged" and t.showlegend
    )
    assert "%{customdata}" in diff_trace.hovertemplate


def test_herding_caption_appears_for_home_import_not_soc() -> None:
    # Decision 0004 item 45, worded generically per lead direction, 28
    # September 2026: no fixed clock time, because prices are now a
    # day-ahead forecast that varies day to day, so naming one half-hour
    # would be wrong on other days or runs.
    # The mechanism moved to the caption's tooltip (polish plan G7: one
    # short "Finding:" line on screen).
    home_st = RecordingStreamlit(metric="Home import")
    render_action_response(home_st, make_result("action"))
    home_findings = [
        (args[0], kwargs.get("help") or "")
        for args, kwargs in calls_for(home_st, "caption")
        if args[0].startswith("Finding:")
    ]
    assert len(home_findings) == 1
    caption, tooltip = home_findings[0]
    assert "herds" in caption
    assert "same price forecast" in tooltip and "no site or network limit" in tooltip
    assert "05:30" not in caption + tooltip

    soc_st = RecordingStreamlit(metric="Battery SoC")
    render_action_response(soc_st, make_result("action"))
    soc_captions = [args[0] for args, _ in calls_for(soc_st, "caption")]
    assert not any(c.startswith("Finding:") for c in soc_captions)


def test_herding_caption_quotes_each_weeks_own_peak_not_the_median_curve() -> None:
    # Goal review N3, decision 0004 items 48-49: the herding measure is each
    # week's own peak and the per-week ratio, from ``weekly_peak_summary``.
    result = make_result("action")
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    peaks = result.weekly_peak_summary.set_index("path_id")
    caption, tooltip = next(
        (args[0], kwargs["help"]) for args, kwargs in calls_for(st, "caption") if "herd" in args[0]
    )
    assert f"×{peaks.loc['selected', 'ratio_to_normal_p50']:.1f} the unmanaged peak" in caption
    assert f"{round(peaks.loc['selected', 'p50']):,} kW on the smart path" in tooltip
    assert f"{round(peaks.loc['normal', 'p50']):,} kW unmanaged" in tooltip
    assert "week by week" in tooltip


@pytest.mark.parametrize("metric", ["Home import", "Battery SoC"])
def test_difference_caption_says_every_night_is_a_smart_night(metric) -> None:
    # Decision 0004 item 52 (noon-to-noon study) retired item 49's note that
    # the last night looks like unmanaged charging.
    st = RecordingStreamlit(metric=metric)
    render_action_response(st, make_result("action"))
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    # Short on screen; the rule is spelt out in the definition (polish plan
    # G7) in words, never as a decision citation (voice pass).
    assert any("every night is a smart night" in c for c in captions)
    assert any("full smart-charging night" in c and "noon to noon" in c for c in captions)
    assert not any("item 49" in c or "item 52" in c for c in captions)
    # The study starts on the Run's London date, so the last night is not
    # always a Sunday; the note must not name a weekday.
    assert not any("Sunday" in c for c in captions)


# --- Day-ahead plan vs dispatched block (intraday-dispatch-v1 §10, K5) ------
#
# The SYNTHETIC fixture never carries dispatch frames (make_result's toy
# physics has no intraday dispatch), so the block is tested against a real
# small run instead, the same route test_dispatch_wiring.py uses: it is
# cached, so the tests below share one run per (dispatch, event_presets) pair.

_DISPATCH_START = date(2026, 10, 12)
_DISPATCH_VALUES = {"vehicle_count": 60, "evaluation_world_count": 6}


@cache
def _dispatch_result(dispatch: int = 1, event_presets: tuple[str, ...] = ()) -> ForecastResult:
    return run_forecast_from_assumptions(
        _DISPATCH_START,
        model="action",
        values=_DISPATCH_VALUES | {"trading.intraday_dispatch": float(dispatch)},
        event_presets=event_presets,
    )


_DISPATCH_CHART_KEYS = (
    "action-response-dispatch-paths",
    "action-response-dispatch-difference",
    "action-response-dispatch-replan-count",
    "action-response-dispatch-world",
)


def test_dispatch_block_renders_its_charts_and_captions_when_the_run_dispatched() -> None:
    result = _dispatch_result()
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    keys = {kwargs.get("key") for _, kwargs in calls_for(st, "plotly_chart")}
    assert set(_DISPATCH_CHART_KEYS) <= keys
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert any("EVs locked to their day-ahead plan" in c for c in captions)
    assert any("Moved energy sums to about zero" in c for c in captions)


def test_dispatch_block_absent_with_the_switch_off() -> None:
    off = _dispatch_result(dispatch=0)
    assert off.dispatch_world_slot is None
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, off)

    keys = {kwargs.get("key") for _, kwargs in calls_for(st, "plotly_chart")}
    assert not keys & set(_DISPATCH_CHART_KEYS)


def test_dispatch_block_absent_on_battery_soc_metric() -> None:
    result = _dispatch_result()
    st = RecordingStreamlit(metric="Battery SoC")
    render_action_response(st, result)

    keys = {kwargs.get("key") for _, kwargs in calls_for(st, "plotly_chart")}
    assert not keys & set(_DISPATCH_CHART_KEYS)


def test_dispatch_split_caption_reports_locked_count_share_and_threshold() -> None:
    result = _dispatch_result()
    split = result.dispatch_split
    locked = int(split["locked_count"].sum())
    total = int(split["ev_count"].sum())
    caption = action_response._dispatch_split_caption(split)

    assert caption.startswith(f"{locked} of {total} EVs locked to their day-ahead plan")
    assert "£10/MWh" in caption
    assert len(caption) <= 140


def test_dispatched_teal_trace_names_a_smart_path_word() -> None:
    # dashboard_lint's teal-reserved rule: a teal trace must say "smart",
    # "selected", "chosen" or "axle" in its name or legend group. The block's
    # own label does not, so the underlying path_id carries it instead.
    result = _dispatch_result()
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    paths_figure = next(
        args[0]
        for args, kwargs in calls_for(st, "plotly_chart")
        if kwargs.get("key") == "action-response-dispatch-paths"
    )
    teal = SERIES_COLOURS["selected"]
    dispatched = next(trace for trace in paths_figure.data if trace.line.color == teal)
    assert dispatched.legendgroup == "selected"


# --- Timed tariff path (decision 0007), present by default ------------------
#
# The SYNTHETIC fixture never carries a "timed" path (make_result's toy
# physics has no timed-start rule), so this is tested against a real small
# run instead, the same route the dispatch block above uses.

_TIMED_START = date(2026, 2, 9)
_TIMED_VALUES = {"vehicle_count": 12, "evaluation_world_count": 2}


@cache
def _timed_result(*, on: bool) -> ForecastResult:
    # Decision 0007 follow-up (8 October 2026): present by default (the
    # "Timed tariff policy" switch), at midnight; "off" means the switch,
    # not the hour field (a bounded number_input has no reachable "unset"
    # value in Streamlit).
    values = dict(_TIMED_VALUES)
    values["timed_tariff_enabled"] = 1 if on else 0
    values["timed_start_local_hour"] = 0.0
    return run_forecast_from_assumptions(_TIMED_START, values=values)


def _paths_figure_for(*, on: bool):
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, _timed_result(on=on))
    return next(
        args[0]
        for args, kwargs in calls_for(st, "plotly_chart")
        if kwargs.get("key") == "action-response-paths-home_import_kw"
    )


def test_response_chart_has_three_paths_on_and_two_off_in_display_order() -> None:
    # Lead decision, 8 October 2026: Unmanaged, Timed tariff, Smart on
    # screen (style.PATH_DISPLAY_ORDER), not the frames' own normal,
    # selected, timed order.
    assert PATH_DISPLAY_ORDER == ("normal", "timed", "selected")
    on_figure = _paths_figure_for(on=True)
    off_figure = _paths_figure_for(on=False)

    on_names = [trace.name for trace in on_figure.data if trace.showlegend]
    off_names = [trace.name for trace in off_figure.data if trace.showlegend]
    assert on_names == [PATH_LABELS[path] for path in PATH_DISPLAY_ORDER]
    assert off_names == ["Unmanaged", "Smart"]
    assert len(on_figure.data) == 3 * 3  # band low, band high, median line per path
    assert len(off_figure.data) == 2 * 3


def test_timed_tariff_trace_is_not_teal_and_has_its_own_dash() -> None:
    figure = _paths_figure_for(on=True)
    teal = SERIES_COLOURS["selected"]
    timed_trace = next(
        trace for trace in figure.data if trace.name == "Timed tariff" and trace.showlegend
    )
    assert timed_trace.line.color != teal
    assert timed_trace.line.dash != "solid"


def test_timed_tariff_trace_steps_up_exactly_at_midnight_with_start_time_0() -> None:
    # Mike, reviewing the running app: the rise must read as vertical at the
    # start time, not a slope drawn through the last barred half-hour
    # (decision 0007). "hv" holds the value across each half-hour and jumps
    # exactly at the next point, so the barred-to-allowed jump lands on the
    # 00:00 point itself, not before it.
    figure = _paths_figure_for(on=True)
    timed_trace = next(
        trace for trace in figure.data if trace.name == "Timed tariff" and trace.showlegend
    )
    assert timed_trace.line.shape == "hv"
    # Plotly stores the x values as naive datetime64 (the UTC instants the
    # figure was built from, see hover_time_labels/london_time_axis).
    london = pd.DatetimeIndex(timed_trace.x).tz_localize("UTC").tz_convert("Europe/London")
    y = np.asarray(timed_trace.y, dtype=float)
    barred = london.hour >= 12
    assert barred.any()
    assert (y[barred] == 0.0).all()
    first_nonzero = london[y > 0.0][0]
    assert (first_nonzero.hour, first_nonzero.minute) == (0, 0)


def test_response_chart_title_names_all_three_paths_in_display_order() -> None:
    # Mike, reviewing the running app: the title must name every policy
    # actually drawn, in the same order as the legend (decision 0007).
    # chart_block renders the title through st.markdown, before the chart.
    on_st = RecordingStreamlit(metric="Home import")
    render_action_response(on_st, _timed_result(on=True))
    off_st = RecordingStreamlit(metric="Home import")
    render_action_response(off_st, _timed_result(on=False))

    def paths_title(st: RecordingStreamlit) -> str:
        return next(
            args[0] for args, _ in calls_for(st, "markdown") if args[0].startswith("**Home import:")
        )

    assert paths_title(on_st) == "**Home import: unmanaged vs timed tariff vs smart (kW)**"
    assert paths_title(off_st) == "**Home import: unmanaged vs smart (kW)**"


def test_weekly_peak_tile_shows_timed_tariffs_own_peak_beside_smarts() -> None:
    # Decision 0007, model step 2: the tile stays "Weekly smart peak" (the
    # headline figure never changes to the timed path's own number), with
    # the timed path's own ratio joining the context and tooltip beside it.
    result = _timed_result(on=True)
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    peaks = result.weekly_peak_summary.set_index("path_id")
    timed = peaks.loc["timed"]
    peak_label, _, _, peak_context, peak_help = kpi_calls(st)[1]
    assert peak_label == "Weekly smart peak"
    assert f"timed tariff ×{timed['ratio_to_normal_p50']:.1f}" in peak_context
    assert "Timed tariff" in peak_help and f"{round(timed['p50']):,} kW" in peak_help


def test_weekly_peak_tile_has_no_timed_figure_when_the_path_is_absent() -> None:
    result = _timed_result(on=False)
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    _, _, _, peak_context, peak_help = kpi_calls(st)[1]
    assert "timed tariff" not in peak_context
    assert "Timed tariff" not in peak_help


def test_herding_caption_gains_a_plain_timed_tariff_line_when_present() -> None:
    # The herding "Finding:" stays Smart's own, unchanged (at most one
    # "Finding:" caption per lens); a second, separate plain caption says
    # how the timed path's own peak compares with the unmanaged peak.
    result = _timed_result(on=True)
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    peaks = result.weekly_peak_summary.set_index("path_id")
    timed = peaks.loc["timed"]
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    findings = [c for c in captions if c.startswith("Finding:")]
    assert len(findings) == 1
    timed_caption = next(c for c in captions if c.startswith("Timed tariff"))
    assert f"×{timed['ratio_to_normal_p50']:.1f} the unmanaged peak" in timed_caption
    assert len(timed_caption) <= 140


def test_herding_caption_has_no_timed_line_when_the_path_is_absent() -> None:
    result = _timed_result(on=False)
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert not any(c.startswith("Timed tariff") for c in captions)


def test_data_table_already_carries_the_timed_paths_rows_when_present() -> None:
    # decision 0007: _combined_frame is built from the metric's unfiltered
    # path_rows (every path_id row the fleet frame carries), so the timed
    # path's rows reach the "Data and definition" table with no extra code;
    # this locks that behaviour in rather than leaving it unverified.
    result = _timed_result(on=True)
    st = RecordingStreamlit(metric="Home import")
    render_action_response(st, result)

    frame = next(args[0] for args, _ in calls_for(st, "dataframe"))
    assert set(frame["Series"]) == {"Unmanaged", "Timed tariff", "Smart", "Smart minus unmanaged"}


def test_event_view_adds_the_day_ahead_plan_series_beside_normal_and_selected() -> None:
    result = _dispatch_result(event_presets=("cold_still_evening",))
    st = RecordingStreamlit(metric="Events")
    render_action_response(st, result)

    figure = figures_from(st)[0]
    names = {trace.name for trace in figure.data}
    assert "Day-ahead plan" in names
    assert {"Unmanaged", "Smart"} <= names


def test_event_view_has_no_day_ahead_plan_series_with_the_switch_off() -> None:
    off = _dispatch_result(dispatch=0, event_presets=("cold_still_evening",))
    st = RecordingStreamlit(metric="Events")
    render_action_response(st, off)

    figure = figures_from(st)[0]
    names = {trace.name for trace in figure.data}
    assert "Day-ahead plan" not in names


def test_apptest_smart_response_renders_with_and_without_the_dispatch_switch() -> None:
    # §12 K5: AppTest renders with and without the optional dispatch frames,
    # through the real dialog toggle this task made editable, not a fixture.
    app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
    _small_draft(app)
    _run(app)
    _go(app, "Smart charging")
    app.get_by_key("lens::smart-charging").set_value("2 Response").run()
    assert not app.exception
    with_dispatch = "\n".join(str(m.value) for m in app.markdown)
    assert "Day-ahead plan vs dispatched" in with_dispatch

    _edit(app, "assumption::trading.intraday_dispatch", False)
    _run(app)
    _go(app, "Smart charging")
    app.get_by_key("lens::smart-charging").set_value("2 Response").run()
    assert not app.exception
    without_dispatch = "\n".join(str(m.value) for m in app.markdown)
    assert "Day-ahead plan vs dispatched" not in without_dispatch


def test_replan_count_row_plots_the_mean_not_the_all_zero_median() -> None:
    # Few free EVs re-plan in any one half-hour, so the P50 across weeks is 0
    # everywhere in the default run and a P50 bar row was an empty chart.
    result = _dispatch_result()
    bands = result.dispatch_bands
    rows = bands.loc[
        bands["series"].eq("dispatched") & bands["metric"].eq("replan_count")
    ].sort_values("slot_index")
    figure = action_response._replan_count_figure(bands)

    assert list(figure.data[0].y) == list(rows["mean"])


def test_dispatch_split_caption_uses_a_thousands_separator() -> None:
    split = pd.DataFrame(
        {
            "ev_count": [600, 400],
            "locked_count": [480, 320],
            "commitment_share": [0.8, 0.8],
            "replan_threshold_gbp_per_mwh": [10.0, 10.0],
        }
    )
    assert action_response._dispatch_split_caption(split).startswith("800 of 1,000 EVs")
