"""Overview view tests: KPIs and chart data must equal fixture v2 numbers.

Uses the SYNTHETIC FIXTURE result (``fixtures.result_fixture.make_result``),
per T5's brief: build and test against the contract-v2 fixture before the
real model (T1/M5) lands. A lightweight recording ``st`` double stands in for
Streamlit (mirrors ``tests/ui/test_fleet_view.py``); it never calls the model.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from functools import cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fixtures.result_fixture import make_result
from kpi_calls import kpi_calls

from axle_studio.model.forecast import ForecastResult, run_forecast_from_assumptions
from axle_studio.ui.registry import DAY_TYPE_KEY
from axle_studio.ui.style import UNMANAGED_GLOSSARY, assumption_value, money
from axle_studio.ui.views import overview as overview_module
from axle_studio.ui.views.overview import render_overview


class RecordingStreamlit:
    """Fake Streamlit: records every call; ``columns``/``expander`` nest."""

    def __init__(self, calls=None, *, segmented_control_returns=None):
        self.calls = calls if calls is not None else []
        self.session_state = {}
        self._segmented_control_returns = segmented_control_returns or {}

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "columns":
                spec = args[0] if args else kwargs.get("spec", 1)
                count = spec if isinstance(spec, int) else len(spec)
                return [
                    RecordingStreamlit(
                        self.calls, segmented_control_returns=self._segmented_control_returns
                    )
                    for _ in range(count)
                ]
            if name == "segmented_control":
                key = kwargs.get("key")
                if key in self._segmented_control_returns:
                    return self._segmented_control_returns[key]
                return kwargs.get("default")
            if name == "expander":
                return self
            return None

        return record

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def calls_for(st: RecordingStreamlit, name: str) -> list[tuple[tuple, dict]]:
    return [(args, kwargs) for call_name, args, kwargs in st.calls if call_name == name]


def tile_by_label(st: RecordingStreamlit, label: str) -> tuple:
    """``(label, value, unit, context, help)`` of the KPI tile with this label."""

    for tile in kpi_calls(st):
        if tile[0] == label:
            return tile
    raise AssertionError(f"no KPI tile for {label!r}")


def markdown_texts(st: RecordingStreamlit) -> list[str]:
    return [str(args[0]) for args, _ in calls_for(st, "markdown")]


def caption_texts(st: RecordingStreamlit) -> list[str]:
    return [args[0] for args, _ in calls_for(st, "caption")]


@pytest.fixture
def action_result():
    return make_result(model="action", evs=6, worlds=5, seed=42)


@pytest.fixture
def no_action_result():
    return make_result(model="no_action", evs=6, worlds=5, seed=42)


def test_action_result_shows_an_action_badge_and_three_tiles(action_result) -> None:
    st = RecordingStreamlit()
    render_overview(st, action_result)

    _, plugged_value, _, _, _ = tile_by_label(st, "Plugged in at 18:00")
    connected = overview_module._average_day_row(
        action_result, metric="connected_share", day_type="weekday"
    )
    expected = connected.loc[connected["local_half_hour"].eq(36), "centre"].iat[0]
    assert plugged_value == f"{expected:.0%}"
    assert not calls_for(st, "metric")  # every tile is the shared KPI component (G5)

    # The day type is the context line, not the label (polish plan G7).
    _, soc_value, _, soc_context, soc_help = tile_by_label(st, "Median SoC at plug-in")
    soc_kpi = overview_module._plug_in_kpi(
        action_result, metric="median_plug_in_soc_percent", day_type="weekday"
    )
    assert soc_value == f"{soc_kpi['p50']:.1f}%"
    cnz = assumption_value(action_result, "cnz_median_plug_in_soc_percent")
    # Item 4: the model-vs-CNZ gap is explained, not left silent.
    assert "archetypes plug in every day" in soc_help
    assert soc_context == f"Weekday · CNZ observed {cnz:.0f}%"

    # Text-valued "Action chosen" is a badge line, not a tile (G5).
    summary = action_result.action_summary
    badge = next(text for text in markdown_texts(st) if text.startswith("Action chosen"))
    assert ":gray-badge[Smart charging" in badge
    assert f"{summary.departure_margin_hours:g} h before departure" in badge

    _, cost_value, _, cost_context, cost_help = tile_by_label(st, "Illustrative saving, median")
    total = action_result.cost_effect_summary.loc[
        action_result.cost_effect_summary["component"].eq("total")
    ].iloc[0]
    # Fleet-scale money: whole pounds and a U+2212 minus (G6).
    assert f"£{abs(total['p50']):,.0f}" in cost_value
    assert "-" not in cost_value
    assert "unmanaged cost minus smart cost" in cost_help
    not_recovered = action_result.not_recovered_world_count
    assert cost_context.startswith(f"{not_recovered} of {action_result.world_count} weeks")

    # No no-action tiles leak into an action result.
    assert all(not tile[0].startswith("Plug-ins per EV") for tile in kpi_calls(st))


def test_kpi_rows_hold_at_most_four_tiles_and_the_glossary_follows(
    action_result, no_action_result
) -> None:
    # Headline tiles (3 action, 4 no-action) plus the four-tile flexibility strip.
    for result, expected_tiles in ((action_result, 7), (no_action_result, 8)):
        st = RecordingStreamlit()
        render_overview(st, result)

        column_counts = [
            args[0] for args, _ in calls_for(st, "columns") if isinstance(args[0], int)
        ]
        assert max(column_counts) <= 4
        assert len(kpi_calls(st)) == expected_tiles
        assert UNMANAGED_GLOSSARY in caption_texts(st)


def test_saving_tile_is_the_negated_cost_with_swapped_tails(action_result) -> None:
    # Final critique B-9: one saving, one sign across Overview, Value and risk
    # and Key stats C. A cost change of −£1,391 is a saving of £1,391; a cost
    # rise shows as a negative saving with the U+2212 minus.
    summary = action_result.cost_effect_summary.copy()
    total = summary["component"].eq("total")
    summary.loc[total, ["p10", "p50", "p90"]] = [-2000.0, -1391.4, 250.0]
    st = RecordingStreamlit()
    render_overview(st, dataclasses.replace(action_result, cost_effect_summary=summary))

    _, value, _, _, help_text = tile_by_label(st, "Illustrative saving, median")
    assert value == "£1,391"
    assert "P10 to P90: −£250 to £2,000" in help_text

    summary.loc[total, "p50"] = 80.0
    st = RecordingStreamlit()
    render_overview(st, dataclasses.replace(action_result, cost_effect_summary=summary))
    assert tile_by_label(st, "Illustrative saving, median")[1] == "−£80"


def test_median_soc_tile_follows_the_day_toggle_and_labels_it(no_action_result) -> None:
    # Review finding: this tile silently disagreed with Plug-ins' same-named
    # tile because the two used different day types. Changing this page's
    # own toggle must change both the value and the label.
    st = RecordingStreamlit(segmented_control_returns={DAY_TYPE_KEY: "all"})
    render_overview(st, no_action_result)

    _, soc_value, _, soc_context, _ = tile_by_label(st, "Median SoC at plug-in")
    assert soc_context.startswith("All")
    soc_kpi = overview_module._plug_in_kpi(
        no_action_result, metric="median_plug_in_soc_percent", day_type="all"
    )
    assert soc_value == f"{soc_kpi['p50']:.1f}%"


def test_plug_ins_per_ev_tile_follows_the_day_toggle_and_labels_it(no_action_result) -> None:
    # B1: this tile's number already followed the day toggle; only the label
    # was silent about it.
    st = RecordingStreamlit(segmented_control_returns={DAY_TYPE_KEY: "weekend"})
    render_overview(st, no_action_result)

    _, value, _, context, _ = tile_by_label(st, "Plug-ins per EV per week")
    assert context == "Weekend"
    kpi = overview_module._plug_in_kpi(
        no_action_result, metric="plug_ins_per_ev_per_week", day_type="weekend"
    )
    assert f"{kpi['p50']:.1f}" in value


def test_cost_tile_context_says_not_recovered(action_result) -> None:
    # O3: the bare "N of M weeks" count did not say what it was counting.
    # Item 7: this lives in the tile's context line, not a delta (no arrow).
    st = RecordingStreamlit()
    render_overview(st, action_result)

    not_recovered = action_result.not_recovered_world_count
    expected = f"{not_recovered} of {action_result.world_count} weeks materially not recovered"
    # ⚠ only when some week is material (decision 0004 item 45).
    context = tile_by_label(st, "Illustrative saving, median")[3]
    assert context == (f"{expected} ⚠" if not_recovered else expected)


def test_average_day_chart_hovers_show_a_time_not_a_half_hour_index(action_result) -> None:
    # O4: the hover header for this half-hour-indexed axis used to fall back
    # to the raw index (e.g. "36"); it must read as a time of day instead.
    st = RecordingStreamlit()
    render_overview(st, action_result)

    (figure,), _ = calls_for(st, "plotly_chart")[0]
    assert figure.layout.xaxis.hoverformat == "%H:%M"
    assert list(figure.layout.xaxis.ticktext) == ["00:00", "06:00", "12:00", "18:00"]
    bar_trace = figure.data[0]
    assert pd.Timestamp(bar_trace.x[36]).strftime("%H:%M") == "18:00"


def test_no_action_result_shows_four_real_tiles_never_not_in_this_model(no_action_result) -> None:
    st = RecordingStreamlit()
    render_overview(st, no_action_result)

    labels = [tile[0] for tile in kpi_calls(st)]
    assert labels[:4] == [
        "Plugged in at 18:00",
        "Median SoC at plug-in",
        "Plug-ins per EV per week",
        "Unserved travel",
    ]
    for tile in kpi_calls(st):
        assert "not in this model" not in tile[1].lower()
    assert not any(text.startswith("Action chosen") for text in markdown_texts(st))

    plug_ins_value = tile_by_label(st, "Plug-ins per EV per week")[1]
    kpi = overview_module._plug_in_kpi(
        no_action_result, metric="plug_ins_per_ev_per_week", day_type="weekday"
    )
    assert f"{kpi['p50']:.1f}" in plug_ins_value

    unserved_value = tile_by_label(st, "Unserved travel")[1]
    expected = overview_module._weekly_unserved_travel_kwh_p50(no_action_result)
    assert f"{expected:,.0f}" in unserved_value
    assert "kWh" in unserved_value


def test_day_toggle_changes_the_kpi_and_the_chart_data(action_result) -> None:
    weekday_st = RecordingStreamlit(segmented_control_returns={DAY_TYPE_KEY: "weekday"})
    render_overview(weekday_st, action_result)
    weekday_value = tile_by_label(weekday_st, "Plugged in at 18:00")[1]

    weekend_st = RecordingStreamlit(segmented_control_returns={DAY_TYPE_KEY: "weekend"})
    render_overview(weekend_st, action_result)
    weekend_value = tile_by_label(weekend_st, "Plugged in at 18:00")[1]

    weekday_frame = overview_module._average_day_row(
        action_result, metric="connected_share", day_type="weekday"
    )
    weekend_frame = overview_module._average_day_row(
        action_result, metric="connected_share", day_type="weekend"
    )
    assert not weekday_frame["centre"].equals(weekend_frame["centre"])
    assert weekday_value != weekend_value or not np.isclose(
        weekday_frame.loc[weekday_frame["local_half_hour"].eq(36), "centre"].iat[0],
        weekend_frame.loc[weekend_frame["local_half_hour"].eq(36), "centre"].iat[0],
    )

    weekday_figure = calls_for(weekday_st, "plotly_chart")[0][0][0]
    weekend_figure = calls_for(weekend_st, "plotly_chart")[0][0][0]
    assert list(weekday_figure.data[0].y) != list(weekend_figure.data[0].y)


def test_chart_uses_dual_axis_and_style_heights(action_result) -> None:
    st = RecordingStreamlit()
    render_overview(st, action_result)

    (figure,), kwargs = calls_for(st, "plotly_chart")[0]
    assert figure.layout.height == overview_module.CHART_HEIGHTS["time_series_dual_axis"]
    assert figure.layout.yaxis2.overlaying == "y"
    assert figure.layout.margin.r == 56  # dual-axis right margin (design 3.5, item 26)
    assert kwargs["config"] == {"displayModeBar": False}
    # Legend-below-plot and single-axis defaults live in the shared "axle"
    # template (style.py), already covered by its own tests; this view only
    # needs to prove it applied that template via chart_block/style_figure.
    assert figure.layout.template.layout.legend.orientation == "h"


def test_next_links_render_three_buttons_with_correct_targets(action_result) -> None:
    st = RecordingStreamlit()
    render_overview(st, action_result)

    button_calls = [
        call for call in calls_for(st, "button") if call[1]["key"].startswith("overview-next")
    ]
    assert [args[0] for args, _ in button_calls] == [
        "Plug-in times and SoC →",
        "One driver's week →",
        "Why this action →",
    ]
    assert [kwargs["args"] for _, kwargs in button_calls] == [
        ("Drivers", "Plug-ins"),
        ("Drivers", "One EV"),
        ("Smart charging", "1 Plan"),
    ]
    assert all(kwargs["on_click"] is overview_module._switch_to for _, kwargs in button_calls)


def test_switch_to_sets_the_lens_before_switching_page(monkeypatch) -> None:
    """The acceptance criterion: page links land on the named lens.

    ``st.switch_page``/``st.page_link`` require a real ``Page`` object for a
    callable-defined page (verified against the installed Streamlit, see
    ``overview._switch_to``'s docstring); this test fakes the ``streamlit``
    module ``_switch_to`` calls, and checks the ordering and values Streamlit
    itself cannot check for us: the lens key is written before navigation.
    """

    switched = []
    fake_streamlit = SimpleNamespace(
        session_state={},
        Page=lambda fn, url_path: SimpleNamespace(url_path=url_path),
        switch_page=lambda page: switched.append(page),
    )
    monkeypatch.setattr(overview_module, "_streamlit", fake_streamlit)

    overview_module._switch_to("Drivers", "Plug-ins")

    assert fake_streamlit.session_state["lens::drivers"] == "Plug-ins"
    assert len(switched) == 1
    assert switched[0].url_path == "drivers"


def test_switch_to_writes_a_preset_before_switching(monkeypatch) -> None:
    switched = []
    fake_streamlit = SimpleNamespace(
        session_state={},
        Page=lambda fn, url_path: SimpleNamespace(url_path=url_path),
        switch_page=lambda page: switched.append(dict(fake_streamlit.session_state)),
    )
    monkeypatch.setattr(overview_module, "_streamlit", fake_streamlit)

    overview_module._switch_to("Drivers", "Fleet week", {"fleet-week-metric": "Movable energy"})

    assert switched == [{"lens::drivers": "Fleet week", "fleet-week-metric": "Movable energy"}]


def test_numbers_equal_the_fixture(action_result) -> None:
    """Cross-check: every KPI traces back to the exact fixture value, no invention."""

    st = RecordingStreamlit()
    render_overview(st, action_result)

    connected = overview_module._average_day_row(
        action_result, metric="connected_share", day_type="weekday"
    )
    expected_18 = connected.loc[connected["local_half_hour"].eq(36), "centre"].iat[0]
    assert tile_by_label(st, "Plugged in at 18:00")[1] == f"{expected_18:.0%}"


def test_flexible_power_chart_is_replaced_by_a_four_tile_strip(action_result) -> None:
    """Decision 0004 item 54 (plan C3, E1): the average day stays the one chart."""

    st = RecordingStreamlit()
    render_overview(st, action_result)

    assert len(calls_for(st, "plotly_chart")) == 1
    labels = [tile[0] for tile in kpi_calls(st)][-4:]
    assert labels == [
        "Peak power below target",
        "Can wait 2 h+ at 19:00",
        "Hours at ¼ capacity or more",
        "Coincidence factor",
    ]
    assert all(len(label) <= 28 for label in labels)  # plan G7


def test_flexibility_tiles_equal_the_fixture(action_result) -> None:
    st = RecordingStreamlit()
    render_overview(st, action_result)

    row = action_result.flexibility_weekly_summary.iloc[0]
    peak = tile_by_label(st, "Peak power below target")
    assert peak[1] == f"{row['peak_deferrable_kw_p50']:,.0f} kW"
    # Clarity critique: the context named neither "power" nor "target" plainly.
    assert peak[3] == (
        "full charger power of plugged-in EVs not yet at target; most often "
        f"{row['peak_modal_interval_start_london']:%a %H:%M}"
    )
    # Final critique B-1: the tile is charger power of EVs below target, not
    # charging that could wait, and its help says it can exceed actual import.
    assert "could wait" not in peak[4]
    assert "above the actual unmanaged import" in peak[4]
    assert tile_by_label(st, "Can wait 2 h+ at 19:00")[1] == (
        f"{row['deferrable_at_least_2h_kw_1900_p50']:,.0f} kW"
    )
    capacity = float(row["fleet_charger_capacity_kw"])
    hours_tile = tile_by_label(st, "Hours at ¼ capacity or more")
    assert hours_tile[1] == f"{row['hours_at_least_quarter_capacity_p50']:.1f} h"
    # Clarity critique: "¼ of what?" was unanswered; give the kW threshold.
    assert hours_tile[3] == (
        f"hours a week with ≥{capacity / 4:,.0f} kW of charger power still below target"
    )
    peaks = action_result.weekly_peak_summary.set_index("path_id")
    coincidence = tile_by_label(st, "Coincidence factor")
    assert coincidence[1] == f"{peaks.loc['normal', 'coincidence_factor_p50']:.2f}"
    # Clarity critique: "coincidence factor" is jargon; the context now says
    # what it is (a ratio), not only the smart-path comparison value.
    assert coincidence[3] == (
        "peak import ÷ plugged-in charger power; smart "
        f"{peaks.loc['selected', 'coincidence_factor_p50']:.2f}"
    )


def test_plugged_in_tile_names_its_day_type(action_result) -> None:
    # Final critique Q-26: the value follows the day toggle, so the context says which.
    st = RecordingStreamlit(segmented_control_returns={DAY_TYPE_KEY: "weekend"})
    render_overview(st, action_result)

    assert tile_by_label(st, "Plugged in at 18:00")[3] == "Weekend"


def test_average_day_explains_the_low_weekday_midday_share(action_result) -> None:
    # Decision 0004 item 68: an EV without a trip that day stays plugged in.
    st = RecordingStreamlit()
    render_overview(st, action_result)

    note = next(
        kwargs.get("help")
        for args, kwargs in calls_for(st, "caption")
        if args[0].startswith("Midday connection comes from drivers without a trip")
    )
    assert note == overview_module.MIDDAY_UNPLUG_NOTE
    assert "stays plugged in" in note


def test_coincidence_tile_reads_unavailable_without_a_smart_run(no_action_result) -> None:
    # weekly_peak_summary is None on a no-action result: say so, never 0.
    st = RecordingStreamlit()
    render_overview(st, no_action_result)

    assert tile_by_label(st, "Coincidence factor")[1] == "Unavailable"


def test_flexibility_link_opens_fleet_week_on_deferrable_power(action_result) -> None:
    st = RecordingStreamlit()
    render_overview(st, action_result)

    _, kwargs = next(
        call for call in calls_for(st, "button") if call[1]["key"] == "overview-to-fleet-week"
    )
    assert kwargs["on_click"] is overview_module._switch_to
    page, lens, preset = kwargs["args"]
    assert (page, lens) == ("Drivers", "Fleet week")
    assert preset == {"fleet-week-metric": "Deferrable power by slack"}


def test_day_type_control_uses_the_shared_session_key(action_result) -> None:
    # Plan E2: one day-type key across lenses, persisted across pages.
    st = RecordingStreamlit()
    render_overview(st, action_result)

    ((_, kwargs),) = calls_for(st, "segmented_control")
    assert kwargs["key"] == DAY_TYPE_KEY
    assert kwargs["persist_state"] == "session"


def test_start_here_stops_are_all_lens_setting_buttons(action_result) -> None:
    # A page link would keep the reader's last lens on the target page (so
    # "1 Plan" could land on "2 Response"); every stop sets its lens first.
    st = RecordingStreamlit()
    render_overview(st, action_result)

    stops = [
        kwargs for _, kwargs in calls_for(st, "button") if kwargs["key"].startswith("start-here")
    ]
    assert [kwargs["args"] for kwargs in stops] == [
        ("Smart charging", "1 Plan"),
        ("Trading", "3 P&L and risk"),
        ("Supplier", "3 Supplier P&L"),
        ("Supplier", "4 Firm MW"),
        ("Replay", "Fleet"),
        ("Drivers", "Household"),
    ]
    assert all(kwargs["on_click"] is overview_module._switch_to for kwargs in stops)
    assert not calls_for(st, "page_link")


# --- Timed tariff path (decision 0007), off by default ----------------------
#
# The SYNTHETIC fixture never carries a "timed" path or its cost-effect
# sibling (make_result's toy physics has no timed-start rule), so this is
# tested against a real small run instead, the same route
# test_action_response.py's own timed-tariff tests use.

_TIMED_START = date(2026, 2, 9)
_TIMED_VALUES = {"vehicle_count": 30, "evaluation_world_count": 3}


@cache
def _timed_result(*, on: bool) -> ForecastResult:
    # The policy is on by default; "off" switches it off explicitly.
    values = {**_TIMED_VALUES, "timed_tariff_enabled": 1 if on else 0}
    if on:
        values["timed_start_local_hour"] = 0.0
    return run_forecast_from_assumptions(_TIMED_START, values=values)


def test_timed_tariff_saving_tile_appears_beside_smart_when_present() -> None:
    result = _timed_result(on=True)
    st = RecordingStreamlit()
    render_overview(st, result)

    labels = [tile[0] for tile in kpi_calls(st)]
    # Beside Smart's own tile, never instead of it: both are on screen.
    assert "Illustrative saving, median" in labels
    assert "Timed tariff saving, median" in labels

    timed_total = result.timed_cost_effect_summary.loc[
        result.timed_cost_effect_summary["component"].eq("total")
    ].iloc[0]
    _, value, _, _, help_text = tile_by_label(st, "Timed tariff saving, median")
    assert value == money(-timed_total["p50"])
    assert "costed the same way as smart charging" in help_text
    assert "no tariff rate is applied" in help_text


def test_timed_tariff_saving_tile_absent_and_row_unchanged_with_the_setting_off() -> None:
    result = _timed_result(on=False)
    assert result.timed_cost_effect_summary is None
    st = RecordingStreamlit()
    render_overview(st, result)

    labels = [tile[0] for tile in kpi_calls(st)]
    assert "Timed tariff saving, median" not in labels
    # Three headline tiles plus the four-tile flexibility strip, exactly as
    # with the setting unset today (test_kpi_rows_hold_at_most_four_tiles...).
    assert len(kpi_calls(st)) == 7
    column_counts = [args[0] for args, _ in calls_for(st, "columns") if isinstance(args[0], int)]
    assert max(column_counts) <= 4
    assert 3 in column_counts


def test_coincidence_tile_names_the_timed_tariff_figure_beside_smart() -> None:
    on = _timed_result(on=True)
    off = _timed_result(on=False)

    st_on = RecordingStreamlit()
    render_overview(st_on, on)
    _, _, _, context_on, _ = tile_by_label(st_on, "Coincidence factor")
    peaks_on = on.weekly_peak_summary.set_index("path_id")
    assert context_on == (
        "peak import ÷ plugged-in charger power; smart "
        f"{peaks_on.loc['selected', 'coincidence_factor_p50']:.2f}; timed tariff "
        f"{peaks_on.loc['timed', 'coincidence_factor_p50']:.2f}"
    )

    st_off = RecordingStreamlit()
    render_overview(st_off, off)
    _, _, _, context_off, _ = tile_by_label(st_off, "Coincidence factor")
    assert "timed" not in context_off
