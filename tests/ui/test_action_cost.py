"""Smart charging lens 3, Value and risk (docs/decisions/0004... item 43)."""

from __future__ import annotations

import math
from datetime import date
from functools import cache
from pathlib import Path

import pandas as pd
import pytest
from fixtures.result_fixture import make_result
from kpi_calls import kpi_calls

from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.ui.style import (
    CHART_HEIGHTS,
    FLAG_LABEL,
    MUTED_INK,
    PATH_LABELS,
    RISK,
    SERIES_COLOURS,
    UNAVAILABLE,
    format_quantity,
    money,
    percent,
)
from axle_studio.ui.views import action_cost
from axle_studio.ui.views.action_cost import (
    _no_negative_zero,
    _risk_rows,
    _strip_figure,
    render_action_cost,
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


def _parse_money(text: str) -> float:
    if text == UNAVAILABLE:
        return float("nan")
    return float(text.replace("−", "-").replace("£", "").replace(",", ""))


def test_money_formatting_uses_explicit_sign_u2212_and_unavailable_for_nan() -> None:
    # Polish plan G6: fleet-scale money in whole pounds, U+2212 for negatives.
    assert money(3.05, signed=True) == "+£3"
    assert money(-12.4, signed=True) == "−£12"
    assert money(-12.4, signed=True).startswith("\u2212")
    assert money(0.0, signed=True) == "£0"
    assert money(float("nan"), signed=True) == UNAVAILABLE


def test_kpi_row_equals_fixture_cost_effect_summary() -> None:
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    total = result.cost_effect_summary.set_index("component").loc["total"]
    tiles = kpi_calls(st)
    label, value, _, context, _ = tiles[0]
    # Final critique B-9: the tile is the saving (the cost total negated), and
    # the negation swaps the tails: saving P10 = −cost P90.
    assert (label, value) == ("Illustrative saving, median", money(-total["p50"]))
    # The P10-P90 range is the median tile's context line (polish plan G5).
    assert context == f"P10–P90 {money(-total['p90'])} to {money(-total['p10'])}"
    assert tiles[1][1] == f"{result.not_recovered_world_count} of {result.world_count}"


def test_flagged_worlds_count_matches_fixture_not_recovered_count() -> None:
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    figure = figures_from(st)[0]
    by_name = {trace.name: trace for trace in figure.data}
    flagged_count = len(by_name[FLAG_LABEL].x) if FLAG_LABEL in by_name else 0
    recovered_count = len(by_name["Recovered"].x) if "Recovered" in by_name else 0
    assert flagged_count == result.not_recovered_world_count
    assert flagged_count + recovered_count == len(result.cost_effect)
    # Flagged worlds are distinguishable without colour: a different marker
    # symbol as well as the FLAG_LABEL text in the legend.
    if flagged_count:
        assert by_name[FLAG_LABEL].marker.symbol != by_name["Recovered"].marker.symbol


def test_material_kpi_label_and_help_state_the_threshold() -> None:
    # Decision 0004 item 45: the count is of material weeks, and the help
    # names the run's threshold rather than "any shortfall".
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    label, value, _, context, help_text = kpi_calls(st)[1]
    assert label == "Weeks materially short"
    assert value == f"{result.not_recovered_world_count} of {result.world_count}"
    # Clarity critique: the context said "At or above the threshold" without
    # naming it, on a tile whose help already does.
    assert context == "shortfall of 1% or more of the week's home import"
    assert "at least 1% of that week's unmanaged home import" in help_text
    assert "normal" not in help_text


def test_early_departures_context_gives_the_sessions_denominator() -> None:
    # Clarity critique HIGH item 11: 45.5 sessions/week sounds high alone;
    # the context now gives the ratio to all sessions (key_stats.py's own
    # section E wording, a ratio of two medians).
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    _, _, _, context, _ = kpi_calls(st)[3]
    sessions = result.plug_in_summary.kpis
    row = sessions.loc[sessions["metric"].eq("plug_ins_per_week") & sessions["day_type"].eq("all")]
    expected_sessions = row.iloc[0]["p50"]
    assert f"of about {expected_sessions:,.0f} sessions a week" in context
    assert "%)" in context


def test_strip_flags_material_weeks_not_any_shortfall() -> None:
    # A week with a tiny shortfall (energy_not_recovered) that is not material
    # must plot as recovered.
    cost = pd.DataFrame(
        {
            "world_id": [0, 1, 2],
            "illustrative_selected_minus_normal_total_gbp": [-1.0, 2.0, 0.5],
            "energy_not_recovered": [True, True, False],
            "not_recovered_material": [False, True, False],
            "unrecovered_share": [0.001, 0.08, 0.0],
        }
    )
    by_name = {trace.name: trace for trace in _strip_figure(cost).data}
    assert list(by_name[FLAG_LABEL].customdata[:, 0]) == [1]
    assert list(by_name["Recovered"].customdata[:, 0]) == [0, 2]


def test_strip_chart_height() -> None:
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)
    assert figures_from(st)[0].layout.height == CHART_HEIGHTS["strip"]


def test_strip_chart_zero_line_is_visible_against_the_grid() -> None:
    # Chart audit 2, O6: the zero line must be a distinct colour from the
    # gridlines, not the same one.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    figure = figures_from(st)[0]
    assert figure.layout.xaxis.zerolinecolor == MUTED_INK
    assert figure.layout.xaxis.zerolinecolor != SERIES_COLOURS["grid"]


def test_recovered_trace_has_a_legend_entry_even_when_it_is_the_only_trace() -> None:
    # Chart audit 2, O6: when every simulated week recovers, "Recovered" is
    # the figure's only trace. Plotly hides a lone trace's legend by
    # default, so this must be explicit.
    cost = pd.DataFrame(
        {
            "world_id": [0, 1, 2],
            "illustrative_selected_minus_normal_total_gbp": [-1.0, 2.0, 0.5],
            "not_recovered_material": [False, False, False],
            "unrecovered_share": [0.0, 0.002, 0.001],
        }
    )
    figure = _strip_figure(cost)

    assert [trace.name for trace in figure.data] == ["Recovered"]
    assert figure.data[0].showlegend is True


def test_strip_plots_the_saving_with_the_same_sign_as_the_tiles() -> None:
    # Final critique B-9: the chart is titled "saving", so a week where smart
    # cost £120 less than unmanaged plots at +120, not −120.
    cost = pd.DataFrame(
        {
            "world_id": [0, 1],
            "illustrative_selected_minus_normal_total_gbp": [-120.0, 30.0],
            "not_recovered_material": [False, False],
            "unrecovered_share": [0.0, 0.0],
        }
    )
    figure = _strip_figure(cost)

    assert list(figure.data[0].x) == [120.0, -30.0]
    assert figure.layout.xaxis.title.text == "£ per week (illustrative), unmanaged minus smart"
    assert "saving" in figure.data[0].hovertemplate
    assert "World" not in figure.data[0].hovertemplate


def test_strip_chart_hover_never_shows_negative_zero() -> None:
    # Chart audit 2 follow-up, O1 (BLOCKING): a raw float in (-0.5, 0)
    # prints as "-0" in the strip chart's %{x:,.0f} hovertemplate (whole
    # pounds at fleet scale, polish plan G6), even though the same value
    # displays as plain "0" everywhere else.
    cost = pd.DataFrame(
        {
            "world_id": [0],
            "illustrative_selected_minus_normal_total_gbp": [-0.224],
            "not_recovered_material": [False],
            "unrecovered_share": [0.0],
        }
    )
    figure = _strip_figure(cost)

    x_value = figure.data[0].x[0]
    assert math.copysign(1.0, x_value) == 1.0, "must be +0.0, not -0.0"
    assert f"{x_value:,.0f}" == "0"
    assert "%{x:,.0f}" in figure.data[0].hovertemplate


def test_no_negative_zero_clears_the_sign_a_hovertemplate_would_show() -> None:
    rounded = _no_negative_zero(pd.Series([-0.00224, -1.6, 1.6, 0.0]), decimals=2)
    assert list(rounded) == [0.0, -1.6, 1.6, 0.0]
    assert math.copysign(1.0, rounded.iloc[0]) == 1.0, "must be +0.0, not -0.0"
    assert f"{rounded.iloc[0]:,.2f}" == "0.00"


def test_strip_colours_markers_by_unrecovered_share_size() -> None:
    # Goal review actions 2, 47: colour by shortfall size, not only the flat
    # material/not-material split. A bigger shortfall must map to a marker
    # colour further along the scale than a smaller one, even within the
    # same (material) group.
    cost = pd.DataFrame(
        {
            "world_id": [0, 1, 2],
            "illustrative_selected_minus_normal_total_gbp": [1.0, 2.0, 3.0],
            "not_recovered_material": [True, True, False],
            "unrecovered_share": [0.02, 0.10, 0.0],
        }
    )
    by_name = {trace.name: trace for trace in _strip_figure(cost).data}
    flagged = by_name[FLAG_LABEL]
    # world 1 (10% unrecovered) must colour further up the scale than
    # world 0 (2% unrecovered); both share one colour range (cmax) with the
    # "Recovered" trace so the two groups are visually comparable.
    assert list(flagged.marker.color) == [2.0, 10.0]
    assert flagged.marker.cmax == by_name["Recovered"].marker.cmax
    assert flagged.marker.cmax == pytest.approx(10.0)
    assert flagged.marker.showscale or by_name["Recovered"].marker.showscale
    assert list(flagged.marker.colorscale) == [tuple(stop) for stop in RISK]


def test_strip_scale_tops_out_at_the_material_threshold_at_least() -> None:
    # Polish plan G4: the scale's top is max(largest share, threshold), so full
    # red is at or above material. With a material week at 1.2% and a 1%
    # threshold the scale runs to 1.2% and shows its colourbar as the key.
    cost = pd.DataFrame(
        {
            "world_id": [0, 1],
            "illustrative_selected_minus_normal_total_gbp": [1.0, 2.0],
            "not_recovered_material": [True, False],
            "unrecovered_share": [0.012, 0.004],
        }
    )
    traces = _strip_figure(cost, material_share_percent=1.0).data
    assert all(trace.marker.cmax == pytest.approx(1.2) for trace in traces)
    assert any(trace.marker.showscale for trace in traces)


def test_strip_draws_one_neutral_colour_when_no_week_is_material() -> None:
    # Polish plan G4: every week recovered (below the material threshold),
    # so there is nothing for the risk scale to grade. Graded colours without
    # their colourbar would have no key, so every marker is one neutral
    # colour and the caption says so.
    cost = pd.DataFrame(
        {
            "world_id": [0, 1],
            "illustrative_selected_minus_normal_total_gbp": [1.0, 2.0],
            "not_recovered_material": [False, False],
            "unrecovered_share": [0.002, 0.0],
        }
    )
    traces = _strip_figure(cost).data
    assert not any(trace.marker.showscale for trace in traces)
    assert all(trace.marker.color == SERIES_COLOURS["normal"] for trace in traces)
    assert all(trace.marker.colorscale is None for trace in traces)

    result = make_result("action")
    result.cost_effect["not_recovered_material"] = False
    st = RecordingStreamlit()
    render_action_cost(st, result)
    assert not any(trace.marker.showscale for trace in figures_from(st)[0].data)
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert any("every week is drawn in one colour" in caption for caption in captions)


def _risk_table(st) -> pd.DataFrame:
    return next(args[0] for args, _ in calls_for(st, "dataframe") if "Measure" in args[0].columns)


def test_lens_has_four_tiles_and_a_risk_table_from_the_summaries() -> None:
    # Polish plan G9: four tiles; the remaining magnitudes (contract 4.9a,
    # decision 0004 items 43, 45) are one visible table, each value already
    # world-first in ``not_recovered_summary``/``smart_charging_summary``.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    tiles = kpi_calls(st)
    assert [tile[0] for tile in tiles] == [
        "Illustrative saving, median",
        "Weeks materially short",
        "Unrecovered energy",
        "Early departures",
    ]
    rows = result.not_recovered_summary.set_index("metric")
    smart = result.smart_charging_summary.set_index("metric")
    _, kwh_value, kwh_unit, _, _ = tiles[2]
    assert kwh_value == format_quantity(rows.loc["unrecovered_kwh", "p50"], "", decimals=1)
    assert kwh_unit == "kWh per week"
    _, early_value, early_unit, _, _ = tiles[3]
    assert early_value == format_quantity(smart.loc["early_departure_count", "p50"], "", decimals=1)
    assert early_unit == "sessions per week"

    table = _risk_table(st).set_index("Measure")
    # goal review N4: 2 dp, not 0 -- a typical shortfall share rounds to a
    # bare "0%" at 0 dp, hiding the magnitude this row exists to show.
    assert table.loc["Share of weekly home import unrecovered", "P50"] == percent(
        100 * rows.loc["unrecovered_share", "p50"], decimals=2
    )
    assert table.loc["Sessions affected", "P50"] == format_quantity(
        rows.loc["sessions_affected_count", "p50"], "sessions per week", decimals=1
    )
    assert table.loc["Early-departure shortfall", "P50"] == format_quantity(
        smart.loc["early_departure_shortfall_kwh", "p50"], "kWh per week", decimals=1
    )


def test_components_table_has_five_labelled_rows_in_order() -> None:
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    tables = [args[0] for args, _ in calls_for(st, "dataframe")]
    # dataframe order: the strip chart's data expander, then the components
    # bar chart's data expander (both via ``chart_block``).
    table = tables[1]
    assert list(table["Component"]) == [
        "Home import cost",
        "Public charge cost",
        "Unrecovered energy value",
        "Unserved travel value",
        "Illustrative total cost change",
    ]
    summary = result.cost_effect_summary
    for row, (_, source_row) in zip(table.itertuples(), summary.iterrows(), strict=True):
        assert _parse_money(row.P50) == pytest.approx(source_row["p50"], abs=0.5)


def test_components_are_a_visible_table_not_a_bar_chart() -> None:
    # Polish plan G9: the components table is on screen (not only in a data
    # expander), and the old mostly-one-bar chart is gone: the strip is the
    # lens's only chart.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    assert len(figures_from(st)) == 1
    titles = [args[0] for args, _ in calls_for(st, "markdown")]
    assert "**Illustrative cost change by component, smart minus unmanaged (£ per week)**" in titles


def test_driver_risk_section_relates_early_departures_to_sessions_affected() -> None:
    # Goal review N10: "Sessions affected" (not-recovered magnitude, above)
    # and "Early departures" (driver-side risk, below) are two different
    # tiles a reader could otherwise assume are unrelated or identical; one
    # line must say how they relate.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    # The relation is the "Driver-side risk" heading's tooltip (polish plan G7).
    helps = [kwargs.get("help") or "" for _, kwargs in calls_for(st, "markdown")]
    assert any("Sessions affected" in text and "early departure" in text for text in helps)


def test_public_charge_component_is_shown_and_can_be_nonzero() -> None:
    # Coordinator amendment (28 September 2026): one flagged world gets a
    # small synthetic public top-up, so this component must not be hidden or
    # hard-coded to zero even though most worlds still show none.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    table = [args[0] for args, _ in calls_for(st, "dataframe")][1]
    public_row = table.loc[table["Component"].eq("Public charge cost")].iloc[0]
    summary_row = result.cost_effect_summary.set_index("component").loc["public_charge_cost"]
    assert _parse_money(public_row["P50"]) == pytest.approx(summary_row["p50"], abs=0.5)
    per_world_column = "illustrative_selected_minus_normal_public_charge_cost_gbp"
    assert result.cost_effect[per_world_column].abs().sum() > 0


def test_unserved_travel_row_stays_visible_even_when_zero() -> None:
    # Decision 0004 item 32: EVs never strand (a public top-up serves the
    # trip first), so unserved travel is usually zero. The row is a check,
    # not hidden data, so it must still appear.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    table = [args[0] for args, _ in calls_for(st, "dataframe")][1]
    assert "Unserved travel value" in list(table["Component"])


def test_strip_chart_data_frame_uses_human_readable_column_names() -> None:
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    strip_table = [args[0] for args, _ in calls_for(st, "dataframe")][0]
    assert list(strip_table.columns) == [
        "Simulated week",
        "Illustrative saving (£)",
        "Unrecovered (kWh)",
        "Unrecovered share",
        "Materially not recovered",
    ]
    assert not any("_" in column for column in strip_table.columns)


def test_total_definition_is_stated() -> None:
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    # In the components caption's tooltip (polish plan G7).
    helps = [kwargs.get("help") or "" for _, kwargs in calls_for(st, "caption")]
    assert any(
        "home import cost + public charge cost + unrecovered energy value + "
        "unserved travel value" in text
        for text in helps
    )


def test_summary_sentence_is_the_shared_smart_charging_sentence() -> None:
    # Decision 0004 item 38: smart charging applies to every EV, so there is
    # no "No action selected" state; Value and risk reads the same sentence
    # as Plan.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)

    sentence = calls_for(st, "markdown")[0][0][0]
    assert sentence.startswith("Smart charging:")


def test_render_both_models_from_fixture_without_crashing() -> None:
    action_st = RecordingStreamlit()
    render_action_cost(action_st, make_result("action"))
    assert figures_from(action_st)

    no_action_st = RecordingStreamlit()
    render_action_cost(no_action_st, make_result("no_action"))
    assert figures_from(no_action_st) == []
    assert calls_for(no_action_st, "info")[0][0] == ("This result has no Axle action.",)


def test_no_text_says_axle_cash_except_the_negating_caveat() -> None:
    # Checks the actual rendered text (design and decision 0003 are about what
    # a viewer reads, not about how a long sentence happens to wrap across
    # source lines), across every string the view passes to st and every
    # dataframe cell it displays.
    result = make_result("action")
    st = RecordingStreamlit()
    render_action_cost(st, result)
    texts: list[str] = []
    for _, args, kwargs in st.calls:
        for value in (*args, *kwargs.values()):
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, pd.DataFrame):
                texts.extend(str(cell) for cell in value.to_numpy().ravel())
    matches = [text for text in texts if "Axle cash" in text]
    assert matches, "expected the caveat sentence to mention Axle cash"
    assert all("not a bid, settlement or Axle cash" in text for text in matches)


def test_view_makes_no_model_calls() -> None:
    source = Path(action_cost.__file__).read_text(encoding="utf-8")
    assert "axle_studio.model" not in source
    assert "run_forecast" not in source


# --- Timed tariff path (decision 0007), off by default ----------------------
#
# The SYNTHETIC fixture never carries a "timed" path (make_result's toy
# physics has no timed-start rule, and its FixtureForecastResult has no
# timed_* attributes at all), so the "present" cases below are tested
# against a real small run instead, the same route test_action_response.py
# and test_action_decision.py use.

_TIMED_START = date(2026, 2, 9)
_TIMED_VALUES = {"vehicle_count": 12, "evaluation_world_count": 2}


@cache
def _timed_result(*, on: bool):
    # The policy is on by default; "absent" switches it off explicitly.
    values = {**_TIMED_VALUES, "timed_tariff_enabled": 1 if on else 0}
    if on:
        values["timed_start_local_hour"] = 0.0
    return run_forecast_from_assumptions(_TIMED_START, values=values)


def test_risk_rows_omits_early_departure_rows_without_a_smart_summary() -> None:
    # Pure-function check: decision 0007 defines no early-departure reading
    # for the hard-barred timed rule (there is no smart plan to leave early
    # from), so those two rows only ever appear with a real smart summary.
    not_recovered = pd.DataFrame(
        {
            "metric": ["unrecovered_kwh", "unrecovered_share", "sessions_affected_count"],
            "unit": ["kWh per week", "fraction", "sessions per week"],
            "p10": [0.1, 0.001, 0.0],
            "p50": [0.2, 0.002, 1.0],
            "p90": [0.3, 0.003, 2.0],
        }
    )
    rows = _risk_rows(not_recovered, None)
    assert [row["Measure"] for row in rows] == [
        "Unrecovered energy",
        "Share of weekly home import unrecovered",
        "Sessions affected",
    ]


def test_kpi_row_shows_timed_tariff_saving_beside_smart_when_present() -> None:
    result = _timed_result(on=True)
    st = RecordingStreamlit()
    render_action_cost(st, result)

    timed_total = result.timed_cost_effect_summary.set_index("component").loc["total"]
    timed_saving = money(-float(timed_total["p50"]))
    _, _, _, context, help_text = kpi_calls(st)[0]
    assert "timed tariff" in context and timed_saving in context
    assert "costed the same way" in help_text


def test_kpi_row_has_no_timed_figure_when_the_path_is_absent() -> None:
    result = _timed_result(on=False)
    st = RecordingStreamlit()
    render_action_cost(st, result)

    _, _, _, context, help_text = kpi_calls(st)[0]
    assert "timed tariff" not in context
    assert "costed the same way" not in help_text


def test_timed_costing_caption_shown_only_when_the_path_is_present() -> None:
    on_st = RecordingStreamlit()
    render_action_cost(on_st, _timed_result(on=True))
    off_st = RecordingStreamlit()
    render_action_cost(off_st, _timed_result(on=False))

    on_captions = [args[0] for args, _ in calls_for(on_st, "caption")]
    off_captions = [args[0] for args, _ in calls_for(off_st, "caption")]
    assert any("costed the same way" in c and "day-ahead price" in c for c in on_captions)
    assert not any("costed the same way" in c for c in off_captions)


def test_strip_chart_grows_a_second_panel_and_shares_one_legend_when_present() -> None:
    result = _timed_result(on=True)
    st = RecordingStreamlit()
    render_action_cost(st, result)

    figure = figures_from(st)[0]
    assert figure.layout.height == action_cost._TWO_POLICY_STRIP_HEIGHT
    subplot_titles = [a.text for a in figure.layout.annotations]
    assert subplot_titles == [PATH_LABELS["timed"], PATH_LABELS["selected"]]
    legend_names = [t.name for t in figure.data if t.showlegend]
    # Each of Recovered/Flagged appears in the legend at most once however
    # many panels draw it (decision 0007: one shared legend, not a repeat
    # per panel).
    assert len(legend_names) == len(set(legend_names))


def test_strip_chart_stays_one_panel_when_the_timed_path_is_absent() -> None:
    result = _timed_result(on=False)
    st = RecordingStreamlit()
    render_action_cost(st, result)

    figure = figures_from(st)[0]
    assert figure.layout.height == CHART_HEIGHTS["strip"]
    assert figure.layout.annotations == ()


def test_strip_chart_data_frame_adds_a_policy_column_when_timed_is_present() -> None:
    result = _timed_result(on=True)
    st = RecordingStreamlit()
    render_action_cost(st, result)

    strip_table = [args[0] for args, _ in calls_for(st, "dataframe")][0]
    assert list(strip_table.columns)[0] == "Policy"
    assert set(strip_table["Policy"]) == {PATH_LABELS["timed"], PATH_LABELS["selected"]}


def test_components_table_adds_timed_tariff_rows_ahead_of_smart_when_present() -> None:
    result = _timed_result(on=True)
    st = RecordingStreamlit()
    render_action_cost(st, result)

    table = [args[0] for args, _ in calls_for(st, "dataframe")][1]
    assert list(table.columns)[0] == "Policy"
    assert list(table["Policy"]) == [PATH_LABELS["timed"]] * 5 + [PATH_LABELS["selected"]] * 5
    component_order = [
        "Home import cost",
        "Public charge cost",
        "Unrecovered energy value",
        "Unserved travel value",
        "Illustrative total cost change",
    ]
    assert list(table["Component"]) == component_order * 2

    titles = [args[0] for args, _ in calls_for(st, "markdown")]
    assert any("smart and timed tariff minus unmanaged" in title for title in titles)


def test_components_table_stays_two_policy_when_the_timed_path_is_absent() -> None:
    result = _timed_result(on=False)
    st = RecordingStreamlit()
    render_action_cost(st, result)

    table = [args[0] for args, _ in calls_for(st, "dataframe")][1]
    assert "Policy" not in table.columns

    titles = [args[0] for args, _ in calls_for(st, "markdown")]
    assert any(
        title == "**Illustrative cost change by component, smart minus unmanaged (£ per week)**"
        for title in titles
    )


def test_risk_table_adds_timed_tariff_rows_without_early_departure_rows() -> None:
    result = _timed_result(on=True)
    st = RecordingStreamlit()
    render_action_cost(st, result)

    table = _risk_table(st)
    assert "Policy" in table.columns
    timed_rows = table.loc[table["Policy"].eq(PATH_LABELS["timed"])]
    assert list(timed_rows["Measure"]) == [
        "Unrecovered energy",
        "Share of weekly home import unrecovered",
        "Sessions affected",
    ]
    smart_rows = table.loc[table["Policy"].eq(PATH_LABELS["selected"])]
    assert list(smart_rows["Measure"]) == [
        "Unrecovered energy",
        "Share of weekly home import unrecovered",
        "Sessions affected",
        "Early departures",
        "Early-departure shortfall",
    ]


def test_risk_table_stays_two_policy_when_the_timed_path_is_absent() -> None:
    result = _timed_result(on=False)
    st = RecordingStreamlit()
    render_action_cost(st, result)

    table = _risk_table(st)
    assert "Policy" not in table.columns
