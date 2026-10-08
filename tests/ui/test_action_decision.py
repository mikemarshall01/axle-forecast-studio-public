"""Smart charging lens 1, Plan (docs/decisions/0004-explicit-action-and-horizon-choices.md
item 43; docs/contracts/results-v2.md section 9).
"""

from __future__ import annotations

from datetime import date
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fixtures.result_fixture import make_result
from kpi_calls import kpi_calls

from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.ui.style import (
    CHART_HEIGHTS,
    PATH_LABELS,
    SERIES_COLOURS,
    format_quantity,
    percent,
)
from axle_studio.ui.views import action_decision
from axle_studio.ui.views.action_decision import (
    _band_absolute_table,
    _band_totals,
    _price_band_paths,
    _price_bands,
    _selected_home_import,
    render_action_decision,
)


class RecordingStreamlit:
    """Fake ``st`` that records every call; columns share the parent's log."""

    def __init__(self, calls: list | None = None) -> None:
        self.calls: list[tuple[str, tuple, dict]] = calls if calls is not None else []

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
        return tuple(RecordingStreamlit(self.calls) for _ in range(count))


def calls_for(st: RecordingStreamlit, name: str) -> list[tuple[tuple, dict]]:
    return [(args, kwargs) for call_name, args, kwargs in st.calls if call_name == name]


def figures_from(st: RecordingStreamlit) -> list:
    return [args[0] for args, _ in calls_for(st, "plotly_chart")]


def test_summary_sentence_describes_smart_charging_and_its_margin() -> None:
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)

    sentence = calls_for(st, "markdown")[0][0][0]
    assert sentence.startswith("Smart charging:")
    assert "cheapest forecast half-hours" in sentence
    assert "minus 1 hour" in sentence
    # The removed 18:00 cap and eligibility screen are not mentioned.
    assert "18:00" not in sentence and "eligible" not in sentence


def test_price_chart_height_and_london_axis() -> None:
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)

    price_figure = figures_from(st)[0]
    assert price_figure.layout.height == CHART_HEIGHTS["time_series"]
    ticktext = list(price_figure.layout.xaxis.ticktext)
    # compact=True on this dual-axis chart (goal review V3): "Mon", not
    # "Mon 28" -- the day-of-month is dropped to stop adjacent days
    # overlapping at 390 px, since a single week never repeats a weekday.
    assert "Mon" in ticktext
    assert not any(label.startswith("Mon 28") for label in ticktext)
    line_trace = next(
        t for t in price_figure.data if t.name == "Synthetic day-ahead price" and t.showlegend
    )
    assert np.allclose(line_trace.y, _price_bands(result)["p50"].to_numpy())


def test_price_chart_shows_day_ahead_p10_p90_across_weeks() -> None:
    # Decision 0004 items 40 and 48: each simulated week has its own
    # day-ahead path, so the chart shows their spread per half-hour.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)

    # The view reads the model's bands (contract 4.7); it computes nothing.
    bands = _price_bands(result)
    pd.testing.assert_frame_equal(bands, result.forecast_price_bands)
    assert (bands["p90"] > bands["p10"]).any()

    price_figure = figures_from(st)[0]
    fill = next(
        t
        for t in price_figure.data
        if t.name == "Synthetic day-ahead price" and t.fill == "tonexty"
    )
    assert np.allclose(fill.y, bands["p90"].to_numpy())


def test_price_chart_legend_has_one_entry_per_series() -> None:
    # Polish plan G3: band and line share a legend group and only the line
    # is listed, so the legend names series, not statistics.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)

    price_figure = figures_from(st)[0]
    listed = [t.name for t in price_figure.data if t.showlegend is not False]
    assert listed == ["Smart home charging (right axis)", "Synthetic day-ahead price"]
    groups = {t.legendgroup for t in price_figure.data}
    assert groups == {"charging", "price"}


def test_price_chart_x_range_is_pinned_to_the_study_span() -> None:
    # Chart audit 2, B6: an explicit range keeps the day ticks clear at 390 px.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)

    price_figure = figures_from(st)[0]
    prices = _price_bands(result)
    start, end = price_figure.layout.xaxis.range
    assert pd.Timestamp(start) == prices["interval_start_utc"].iat[0]
    assert pd.Timestamp(end) == prices["interval_end_utc"].iat[-1]


def test_publication_lines_mark_each_day_ahead_publication_in_the_week() -> None:
    # Plan B4 (decision 0004 item 53): each day's prices are published at
    # 13:00 London the day before, so the week shows one dotted line per
    # publication inside the noon-to-noon study (seven: Monday 13:00 to
    # Sunday 13:00), taken from the result.  The one label anchors its left
    # edge so it grows into the plot (milestone verifier, item 26).
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)

    price_figure = figures_from(st)[0]
    lines = [shape for shape in price_figure.layout.shapes if shape.line.dash == "dot"]
    published = sorted(result.forecast_prices["forecast_available_at_utc"].unique())
    in_range = [t for t in published if t >= result.horizon_start_utc]
    assert [pd.Timestamp(line.x0) for line in lines] == in_range
    assert len(in_range) == 7
    assert all(
        pd.Timestamp(t).tz_convert("Europe/London").strftime("%H:%M") == "13:00" for t in in_range
    )
    labels = [a for a in price_figure.layout.annotations if a.text == "Next day's prices published"]
    assert len(labels) == 1 and labels[0].xanchor == "left"


def test_price_chart_shows_smart_charging_band_and_median_on_a_second_axis() -> None:
    # Decision 0004 item 43 ("shading/area showing when smart charging
    # actually charged") and item 40 (P10-P90 across simulated weeks on
    # every time-series chart): the band collapses to zero outside a
    # charging window and only puffs up while smart charging runs, so it
    # doubles as that shading.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)

    price_figure = figures_from(st)[0]
    charging = _selected_home_import(result.fleet_interval_bands)
    y2_traces = [t for t in price_figure.data if t.yaxis == "y2"]
    assert len(y2_traces) == 3  # P10 (hidden), P90 (band fill), P50 (median line)

    band_trace = next(t for t in y2_traces if t.fill == "tonexty")
    assert np.allclose(band_trace.y, charging["p90"].to_numpy())
    assert band_trace.line.color == SERIES_COLOURS["selected"]
    lower_trace = y2_traces[0]
    assert lower_trace.showlegend is False
    assert np.allclose(lower_trace.y, charging["p10"].to_numpy())

    median_trace = next(t for t in y2_traces if t.showlegend)
    assert median_trace.name == "Smart home charging (right axis)"
    assert np.allclose(median_trace.y, charging["p50"].to_numpy())
    assert median_trace.fill is None  # a line, not a second filled area

    yaxis2 = price_figure.layout.yaxis2
    assert yaxis2.overlaying == "y"
    assert yaxis2.side == "right"
    assert "kW" in yaxis2.title.text


def test_price_chart_data_table_carries_band_and_median() -> None:
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)

    price_table = [args[0] for args, _ in calls_for(st, "dataframe")][0]
    assert list(price_table.columns) == [
        "Interval start (UTC)",
        "Interval start (London)",
        "Day-ahead price P10 (£/MWh)",
        "Day-ahead price, median (£/MWh)",
        "Day-ahead price P90 (£/MWh)",
        "Smart home import P10 (kW)",
        "Smart home import, median (kW)",
        "Smart home import P90 (kW)",
    ]
    # Rounded for the table, 0-1 dp (goal review action 20); the chart itself
    # (tested above) plots the unrounded values.
    prices = _price_bands(result)
    for column, bound in (
        ("Day-ahead price P10 (£/MWh)", "p10"),
        ("Day-ahead price, median (£/MWh)", "p50"),
        ("Day-ahead price P90 (£/MWh)", "p90"),
    ):
        assert np.allclose(price_table[column], prices[bound].round(0).to_numpy())
    charging = _selected_home_import(result.fleet_interval_bands)
    assert np.allclose(
        price_table["Smart home import, median (kW)"], charging["p50"].round(1).to_numpy()
    )
    assert np.allclose(
        price_table["Smart home import P90 (kW)"], charging["p90"].round(1).to_numpy()
    )


def test_band_totals_computes_smart_minus_normal_summed_across_the_week() -> None:
    # Pure-function check with a tiny hand-checkable frame, independent of
    # the fixture (decision 0004 item 12: only ``mean`` is summed).
    # price_band_shift's mean is normal minus selected per date (contract
    # 4.6); _band_totals sums it across dates then negates for smart minus
    # normal.
    bands = pd.DataFrame(
        {
            "price_band": ["low", "middle", "high", "low", "middle", "high"],
            "mean": [-3.0, 1.0, 2.0, -1.0, 0.5, 1.0],
        }
    )
    totals = _band_totals(bands).set_index("price_band")
    assert totals.loc["low", "smart_minus_normal_kwh_per_week"] == pytest.approx(4.0)
    assert totals.loc["middle", "smart_minus_normal_kwh_per_week"] == pytest.approx(-1.5)
    assert totals.loc["high", "smart_minus_normal_kwh_per_week"] == pytest.approx(-3.0)


def test_band_totals_fills_a_band_with_no_rows_as_zero_not_missing() -> None:
    # "no grouped/empty slots" (review, 28 September 2026): every band
    # appears even if price_band_shift happens not to carry a row for it.
    bands = pd.DataFrame({"price_band": ["low"], "mean": [2.0]})
    totals = _band_totals(bands).set_index("price_band")
    assert list(totals.index) == ["low", "middle", "high"]
    assert totals.loc["middle", "smart_minus_normal_kwh_per_week"] == 0.0
    assert totals.loc["high", "smart_minus_normal_kwh_per_week"] == 0.0


def test_band_chart_shows_grouped_absolute_bars_per_band_and_day() -> None:
    # Decision 0004 items 20, 46: absolute bars for each path, not a single
    # signed shift bar per band -- a small net shift can hide two large,
    # nearly cancelling bars, which the earlier diverging chart could not show.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)

    band_figure = figures_from(st)[1]
    bands = result.price_band_shift
    low_rows = bands.loc[bands["price_band"].eq("low")]
    # 3 panels (Cheap/Mid/Expensive), 2 traces each (Unmanaged, Smart).
    assert len(band_figure.data) == 6
    normal_trace = next(t for t in band_figure.data if t.name == "Unmanaged" and t.showlegend)
    assert list(normal_trace.x) == list(low_rows["day_label"])
    assert np.allclose(normal_trace.y, low_rows["normal_kwh_p50"].to_numpy())
    assert normal_trace.error_y.array is not None
    smart_trace = next(t for t in band_figure.data if t.name == "Smart" and t.showlegend)
    assert np.allclose(smart_trace.y, low_rows["selected_kwh_p50"].to_numpy())
    # Only the first panel's traces carry a legend entry (one "Unmanaged"/"Smart"
    # key for all three panels, not three repeats).
    assert sum(t.showlegend is True for t in band_figure.data) == 2
    subplot_titles = [a.text for a in band_figure.layout.annotations]
    assert subplot_titles == ["Cheap", "Mid", "Expensive"]
    assert band_figure.layout.height == action_decision._BAND_FIGURE_HEIGHT


def test_band_chart_title_and_caption_state_absolute_energy_and_net_shift() -> None:
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)

    titles = [args[0] for args, _ in calls_for(st, "markdown")]
    assert any(
        title == "**Home energy by forecast-price band, unmanaged vs smart (kWh per day)**"
        for title in titles
    )
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    # The net shift the old chart plotted is the lens's one "Finding:"
    # caption (polish plan G7), computed from the one column that is safely
    # additive across the week's days.
    totals = _band_totals(result.price_band_shift)
    findings = [c for c in captions if c.startswith("Finding:")]
    assert len(findings) == 1
    finding = findings[0]
    # Final critique O-8: a sentence, not a list of means; the three signed
    # figures move to the caption's help.
    shifts = totals.set_index("price_band")["smart_minus_normal_kwh_per_week"]
    assert shifts["high"] < 0 < shifts["low"], "fixture moves energy dear -> cheap"
    assert finding == (
        f"Finding: in a mean week smart charging moves {abs(shifts['high']):,.0f} kWh out of "
        f"the dearest third of half-hours; the cheapest third gains {shifts['low']:,.0f} kWh."
    )
    assert len(finding) <= 140
    help_text = next(
        kwargs["help"] for args, kwargs in calls_for(st, "caption") if args[0] == finding
    )
    low_shift = shifts["low"]
    sign = "+" if round(low_shift) > 0 else ("\u2212" if round(low_shift) < 0 else "")
    assert f"Cheap {sign}{abs(low_shift):,.0f}" in help_text
    assert "-" not in help_text.split(" kWh.")[0]  # U+2212 minus, never a hyphen


def test_band_chart_caption_says_bars_are_session_nights_without_naming_a_weekday() -> None:
    # Decision 0004 item 52 retired item 49's study-end rule: every night is a
    # smart night and each bar is one noon-to-noon session night.  The caption
    # must read whatever weekday the run starts on, so it names no weekday.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    # Short on screen; the rule is spelt out in words in the definition (Data
    # and definition expander, polish plan G7), never as a decision citation
    # (voice pass: name the rule, not the artefact).
    band_caption = next(c for c in captions if c.startswith("P50 bars, P10"))
    assert "session night" in band_caption and "noon to noon" in band_caption
    assert "item" not in band_caption
    definition = next(c for c in captions if "split into thirds" in c)
    assert "session night, noon to noon" in definition and "evening peak" in definition
    assert "decision" not in definition and "item 49" not in definition
    for weekday in ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"):
        assert weekday not in band_caption + definition


def test_band_chart_data_table_has_absolute_normal_and_smart_columns() -> None:
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)

    band_table = [args[0] for args, _ in calls_for(st, "dataframe")][1]
    assert list(band_table.columns) == [
        "Day",
        "Forecast price band",
        "Unmanaged P10 (kWh)",
        "Unmanaged P50 (kWh)",
        "Unmanaged P90 (kWh)",
        "Smart P10 (kWh)",
        "Smart P50 (kWh)",
        "Smart P90 (kWh)",
    ]
    expected = _band_absolute_table(result.price_band_shift)
    pd.testing.assert_frame_equal(
        band_table.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_plan_kpi_tiles_show_world_first_price_change_as_headline() -> None:
    # BLOCKING finding, 28 September 2026: the headline tile is the
    # world-first average_price_change_gbp_per_mwh (decision 0004 item 12),
    # never the difference of the two medians, which appear as smaller
    # context underneath instead.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)

    summary = result.smart_charging_summary.set_index("metric")
    tiles = kpi_calls(st)
    assert len(tiles) == 2

    change_label, change_value, change_unit, context, _ = tiles[0]
    assert change_label == "Price change vs unmanaged"
    assert change_value == format_quantity(
        summary.loc["average_price_change_gbp_per_mwh", "p50"], "", decimals=0
    )
    assert change_unit == "£/MWh"

    normal_text = format_quantity(
        summary.loc["normal_average_price_gbp_per_mwh", "p50"], "£/MWh", decimals=0
    )
    smart_text = format_quantity(
        summary.loc["selected_average_price_gbp_per_mwh", "p50"], "£/MWh", decimals=0
    )
    assert context.startswith("Unmanaged median")
    assert normal_text in context
    assert smart_text in context

    share_label, share_value, _, _, _ = tiles[1]
    assert share_label == "Share of home charging moved"
    assert share_value == percent(100 * summary.loc["moved_home_import_share", "p50"])


def test_render_both_models_from_fixture_without_crashing() -> None:
    action_result = make_result("action")
    no_action_result = make_result("no_action")

    action_st = RecordingStreamlit()
    render_action_decision(action_st, action_result)
    assert len(figures_from(action_st)) == 2, "the price chart and the price-band chart"

    no_action_st = RecordingStreamlit()
    render_action_decision(no_action_st, no_action_result)
    assert figures_from(no_action_st) == []
    assert calls_for(no_action_st, "info")[0][0] == ("This result has no Axle action.",)


def test_view_makes_no_model_calls() -> None:
    # Static check: the view only reads the finished result (AGENTS.md: a full
    # Monte Carlo starts only from an explicit Run click); it must not import
    # anything from the model package.
    source = Path(action_decision.__file__).read_text(encoding="utf-8")
    assert "axle_studio.model" not in source
    assert "run_forecast" not in source


def test_power_axis_ticks_are_round_from_zero() -> None:
    # Final critique O-3: the right axis read 450 / 1,250 / 2,050 kW; ticks now
    # start at 0 with a 1, 2 or 5 × 10^n step.
    from axle_studio.ui.views.action_decision import _nice_step

    assert _nice_step(4450.0) == 2000.0
    assert _nice_step(3650.0) == 1000.0
    assert _nice_step(0.0) == 1.0
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_decision(st, result)
    figure = next(
        args[0] for args, _ in calls_for(st, "plotly_chart") if args[0].layout.yaxis2.title.text
    )
    assert figure.layout.yaxis2.tick0 == 0
    assert figure.layout.yaxis2.dtick in {m * 10.0**e for m in (1, 2, 5, 10) for e in range(-2, 7)}


# --- Timed tariff path (decision 0007), off by default ----------------------
#
# The SYNTHETIC fixture never carries a "timed" path (make_result's toy
# physics has no timed-start rule), so the "present" cases below are tested
# against a real small run instead, the same route test_action_response.py
# uses for the Response chart.

_TIMED_START = date(2026, 2, 9)
_TIMED_VALUES = {"vehicle_count": 12, "evaluation_world_count": 2}


@cache
def _timed_result(*, on: bool):
    # The policy is on by default; "absent" switches it off explicitly.
    values = {**_TIMED_VALUES, "timed_tariff_enabled": 1 if on else 0}
    if on:
        values["timed_start_local_hour"] = 0.0
    return run_forecast_from_assumptions(_TIMED_START, values=values)


def test_price_band_paths_returns_two_without_timed_columns_and_three_with() -> None:
    # Pure-function check with tiny hand-made frames: normal and selected
    # always have their own kwh columns; timed joins only when its column is
    # there, in on-screen display order (Unmanaged, Timed tariff, Smart).
    two = pd.DataFrame({"normal_kwh_p50": [1.0], "selected_kwh_p50": [1.0]})
    assert _price_band_paths(two) == ["normal", "selected"]
    three = pd.DataFrame(
        {"normal_kwh_p50": [1.0], "selected_kwh_p50": [1.0], "timed_kwh_p50": [1.0]}
    )
    assert _price_band_paths(three) == ["normal", "timed", "selected"]


def test_plan_kpi_tile_shows_timed_tariff_price_beside_smart_when_present() -> None:
    result = _timed_result(on=True)
    st = RecordingStreamlit()
    render_action_decision(st, result)

    summary = result.smart_charging_summary.set_index("metric")
    timed_text = format_quantity(
        summary.loc["timed_average_price_gbp_per_mwh", "p50"], "£/MWh", decimals=0
    )
    _, _, _, context, help_text = kpi_calls(st)[0]
    assert "timed tariff" in context and timed_text in context
    assert "costed the same way" in help_text


def test_plan_kpi_tile_has_no_timed_figure_when_the_path_is_absent() -> None:
    result = _timed_result(on=False)
    st = RecordingStreamlit()
    render_action_decision(st, result)

    _, _, _, context, help_text = kpi_calls(st)[0]
    assert "timed tariff" not in context
    assert "costed the same way" not in help_text


def test_timed_costing_caption_shown_only_when_the_path_is_present() -> None:
    on_st = RecordingStreamlit()
    render_action_decision(on_st, _timed_result(on=True))
    off_st = RecordingStreamlit()
    render_action_decision(off_st, _timed_result(on=False))

    on_captions = [args[0] for args, _ in calls_for(on_st, "caption")]
    off_captions = [args[0] for args, _ in calls_for(off_st, "caption")]
    assert any("costed the same way" in c and "day-ahead price" in c for c in on_captions)
    assert not any("costed the same way" in c for c in off_captions)


def test_band_chart_names_all_three_paths_and_draws_three_bars_when_present() -> None:
    result = _timed_result(on=True)
    st = RecordingStreamlit()
    render_action_decision(st, result)

    titles = [args[0] for args, _ in calls_for(st, "markdown")]
    assert (
        "**Home energy by forecast-price band, unmanaged vs timed tariff vs smart "
        "(kWh per day)**" in titles
    )
    band_figure = figures_from(st)[1]
    legend_names = [t.name for t in band_figure.data if t.showlegend]
    assert legend_names == ["Unmanaged", "Timed tariff", "Smart"]
    # 3 panels (Cheap/Mid/Expensive), 3 traces each now (Unmanaged, Timed
    # tariff, Smart).
    assert len(band_figure.data) == 9

    table = _band_absolute_table(result.price_band_shift)
    assert list(table.columns) == [
        "Day",
        "Forecast price band",
        "Unmanaged P10 (kWh)",
        "Unmanaged P50 (kWh)",
        "Unmanaged P90 (kWh)",
        f"{PATH_LABELS['timed']} P10 (kWh)",
        f"{PATH_LABELS['timed']} P50 (kWh)",
        f"{PATH_LABELS['timed']} P90 (kWh)",
        "Smart P10 (kWh)",
        "Smart P50 (kWh)",
        "Smart P90 (kWh)",
    ]


def test_band_chart_stays_two_policy_when_the_timed_path_is_absent() -> None:
    result = _timed_result(on=False)
    st = RecordingStreamlit()
    render_action_decision(st, result)

    band_figure = figures_from(st)[1]
    legend_names = [t.name for t in band_figure.data if t.showlegend]
    assert legend_names == ["Unmanaged", "Smart"]
    assert len(band_figure.data) == 6
