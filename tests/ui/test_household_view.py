"""Drivers ▸ Household (household contract v1 §6.1, §9 UI rows) on the SYNTHETIC fixture.

The view is rendered into the recording fake ``st`` with the fixture's
``make_household_card`` injected, so every number on screen can be traced to
a card column. One AppTest on a small real run checks that the EV choice
survives a switch between One EV and Household (O8).
"""

from __future__ import annotations

import re
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fixtures.household_contract import make_household_card
from fixtures.result_fixture import make_result
from kpi_calls import kpi_calls
from streamlit.testing.v1 import AppTest
from test_copy_rules import APP_PATH, _run, _show, _small_draft

from axle_studio.model import household as household_model
from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.ui import pages
from axle_studio.ui.views import household, one_ev

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from dashboard_lint import RecordingStreamlit, lint_figure  # noqa: E402

_TIMED_RUN_DATE = date(2026, 10, 12)
_TIMED_RUN_VALUES = {"vehicle_count": 20, "evaluation_world_count": 3, "seed": 42}


@pytest.fixture(scope="module")
def result():
    return make_result("action", evs=12, worlds=6, seed=42)


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


def _card_fn(card=None, calls=None):
    """A ``household_card`` stand-in: the fixture card, or ``card`` for every EV."""

    def build(result, unit_id):
        if calls is not None:
            calls.append(unit_id)
        return card if card is not None else make_household_card(result, unit_id)

    return build


def _render(result, card=None, choices=None, calls=None) -> RecordingStreamlit:
    st = RecordingStreamlit(choices=choices)
    household.render_household(st, result, household_card=_card_fn(card, calls))
    return st


def _texts(st, name: str) -> list[str]:
    return [str(args[0]) for call, args, _ in st.calls if call == name and args]


def _with_outcome(card, metric: str, **values):
    outcomes = card.outcomes.copy()
    row = outcomes["metric"].eq(metric)
    for column, value in values.items():
        outcomes.loc[row, column] = value
    return replace(card, outcomes=outcomes)


def _with_availability(card, result):
    """Attach a SYNTHETIC availability profile built by the model's own fold."""

    slots = result.study_slots
    rng = np.random.default_rng(0)
    shape = (result.world_count, len(slots))
    realised = {
        "turn_down": rng.uniform(0.0, 7.0, shape),
        "turn_up": rng.uniform(0.0, 3.0, shape),
    }
    by_slot, by_day = household_model.availability_profile(realised, slots)
    return replace(card, availability=by_slot, availability_day=by_day)


def test_renders_every_block_and_calls_the_card_once_for_the_picked_ev(result) -> None:
    calls: list[str] = []
    st = _render(result, calls=calls)

    assert calls == ["ev-0000"]  # the first Average (UK) EV, One EV's default
    assert [tile[0] for tile in kpi_calls(st)] == [
        "Ready at departure",
        "Value per year",
        "Sessions affected",
        "Evening turn-down",
    ]
    keys = [key for key, _ in st.figures()]
    assert keys == ["household-outcomes", "household-timing"]
    assert household.FIRM_MW_PENDING in _texts(st, "info")
    assert any(
        text.startswith("Status :gray-badge[Enrolled in smart charging")
        for text in _texts(st, "markdown")
    )


def test_every_figure_follows_the_chart_conventions_bar_the_accepted_axis(result) -> None:
    card = _with_availability(make_household_card(result, "ev-0000"), result)
    for direction in ("Turn-down", "Turn-up"):
        st = _render(result, card=card, choices={"household-direction": direction})
        for key, figure in st.figures():
            rules = [finding.rule for finding in lint_figure(figure)]
            # Horizontal range rows: the unit is in each panel title, the y axis
            # is the three category rows (ACCEPTED in test_dashboard_lint).
            expected = ["y-axis-title"] if key == "household-outcomes" else []
            assert rules == expected, (key, rules)


def test_value_tile_reads_the_ev_mean_and_the_archetypes_average_years(result) -> None:
    card = _with_outcome(
        make_household_card(result, "ev-0000"),
        "value_gbp_per_year",
        mean=52.0,
        cohort_p10=38.0,
        cohort_p90=71.0,
    )
    tile = next(t for t in kpi_calls(_render(result, card=card)) if t[0] == "Value per year")

    assert tile[1] == "£52"
    assert "£38–£71 (P10–P90 average years)" in tile[3]
    assert "illustrative" in tile[3]


def test_ready_tile_reads_p50_and_shows_unavailable_without_closed_sessions(result) -> None:
    card = make_household_card(result, "ev-0000")
    card = _with_outcome(card, "completed_share_selected", p10=0.8, p50=0.9, p90=1.0)
    card = _with_outcome(card, "completed_share_normal", p50=0.91)
    tile = kpi_calls(_render(result, card=card))[0]
    assert tile[1] == "90%"
    assert tile[3] == "in a typical week P10–P90 80%–100% · unmanaged 91%"

    empty = _with_outcome(card, "completed_share_selected", p10=np.nan, p50=np.nan, p90=np.nan)
    tile = kpi_calls(_render(result, card=empty))[0]
    assert (tile[1], tile[3]) == ("Unavailable", "(no closed sessions)")


def test_ready_tile_shows_timed_tariff_beside_unmanaged_and_smart(timed_result) -> None:
    # Decision 0007: Timed tariff reads beside Unmanaged and Smart, never
    # instead of either, and only once the card actually carries the row.
    # A real run (not an injected row): an earlier check of this worktree
    # under the wrong import path (the main checkout's editable install, not
    # PYTHONPATH=src) wrongly found no "timed" columns on household_ev_world
    # /household_outcomes_summary; re-run correctly, model.household.household_card
    # does carry a "completed_share_timed" outcome row for every EV once the
    # setting is on, so this reads the real card, not a crafted one.
    unit_id = timed_result.units["unit_id"].iat[0]
    card = household_model.household_card(timed_result, unit_id)
    assert (card.outcomes["metric"] == "completed_share_timed").any(), (
        "completed_share_timed missing from card.outcomes -- dropped between "
        "household_outcomes_summary and card_outcomes"
    )
    tile = kpi_calls(_render(timed_result, card=card))[0]

    assert re.fullmatch(
        r"in a typical week P10–P90 \d+%–\d+% · unmanaged \d+% · timed tariff \d+%", tile[3]
    )


def test_ready_tile_timed_clause_is_omitted_when_the_timed_value_is_missing(timed_result) -> None:
    # The row can exist with a NaN spread (an EV with no closed timed-path
    # sessions, O4's "of N customers with closed sessions" case); _missing
    # must gate on the value, not just the row's presence. Edits a real
    # card's real row rather than inventing one (the same pattern
    # test_ready_tile_reads_p50_and_shows_unavailable_without_closed_sessions
    # already uses for the selected-path NaN case).
    unit_id = timed_result.units["unit_id"].iat[0]
    card = household_model.household_card(timed_result, unit_id)
    card = _with_outcome(card, "completed_share_timed", p10=np.nan, p50=np.nan, p90=np.nan)
    tile = kpi_calls(_render(timed_result, card=card))[0]

    assert "timed tariff" not in tile[3]


def test_ready_tile_is_unchanged_on_a_real_run_without_the_timed_setting(untimed_result) -> None:
    # Contract rule: absent data, unchanged screen. A real run with the
    # setting off (NaN) carries no "completed_share_timed" row at all.
    unit_id = untimed_result.units["unit_id"].iat[0]
    card = household_model.household_card(untimed_result, unit_id)
    assert not (card.outcomes["metric"] == "completed_share_timed").any()
    tile = kpi_calls(_render(untimed_result, card=card))[0]

    assert re.fullmatch(r"in a typical week P10–P90 \d+%–\d+% · unmanaged \d+%", tile[3])
    assert "timed tariff" not in tile[3]


def test_sessions_affected_tile_says_x_of_y_session_ends(result) -> None:
    card = _with_outcome(make_household_card(result, "ev-0000"), "sessions_affected_count", p50=1.0)
    card = _with_outcome(card, "session_ends_per_week", p50=7.0)
    tile = kpi_calls(_render(result, card=card))[2]

    assert (tile[1], tile[3]) == ("1", "of 7 session ends in a typical week (P50)")


def test_evening_tile_names_the_hold_and_explains_a_zero_reading(result) -> None:
    # Clarity critique §0.1: 0.0 kW reads as broken without saying why smart
    # charging usually leaves nothing to turn down at 17:00-21:00.
    card = _with_outcome(
        make_household_card(result, "ev-0000"), "evening_turn_down_kw_1h", p10=3.4, p50=4.1
    )
    tile = kpi_calls(_render(result, card=card))[3]

    assert tile[1] == "4.1 kW"
    assert tile[3] == "1 h, 17:00–21:00, smart path; 0 when the plan is not charging then"
    assert "Firm (P10 across weeks): 3.4 kW." in tile[4]


def test_outcomes_chart_plots_the_card_columns(result) -> None:
    card = make_household_card(result, "ev-0000")
    st = _render(result, card=card)
    figure = dict(st.figures())["household-outcomes"]
    ready = card.outcomes.set_index("metric").loc["completed_share_selected"]
    # Panel 3 (ready at departure, %): traces are (range, marker) × (EV, archetype, fleet).
    panel = [trace for trace in figure.data if trace.xaxis == "x3"]
    ev_range, ev_marker, cohort_range = panel[0], panel[1], panel[2]

    assert ev_marker.x[0] == pytest.approx(ready["p50"] * 100.0)
    assert list(ev_range.x) == pytest.approx([ready["p10"] * 100.0, ready["p90"] * 100.0])
    assert list(cohort_range.x) == pytest.approx(
        [ready["cohort_p10"] * 100.0, ready["cohort_p90"] * 100.0]
    )
    assert ev_marker.name == "This EV (smart)"


def test_rank_hover_words_higher_and_lower_metrics(result) -> None:
    card = make_household_card(result, "ev-0000")
    card = _with_outcome(card, "value_gbp_per_week", cohort_rank_p50=0.62)
    card = _with_outcome(card, "home_cost_gbp_per_kwh_selected", cohort_rank_p50=0.28)
    figure = dict(_render(result, card=card).figures())["household-outcomes"]
    hovers = [trace.customdata[0] for trace in figure.data if trace.customdata is not None]

    assert any("ahead of 62% of its archetype" in text for text in hovers)
    assert any("72% of its archetype pay more" in text for text in hovers)


def test_control_ev_is_labelled_outside_the_product_with_no_rank(result) -> None:
    card = make_household_card(result, "ev-0000")
    outcomes = card.outcomes.assign(cohort_rank_p50=np.nan, fleet_rank_p50=np.nan)
    card = replace(card, treated=False, dispatch_locked=True, outcomes=outcomes)
    st = _render(result, card=card)

    # The SYNTHETIC fixture carries no ``dispatch_world_slot`` (no run ever
    # dispatched), so the status badge never adds a locked/re-plan suffix
    # here regardless of ``dispatch_locked``; see the ``_status_text`` unit
    # tests below for the dispatch-active cases.
    assert "Status :gray-badge[Hold-out control group, not enrolled in smart charging]" in _texts(
        st, "markdown"
    )
    figure = dict(st.figures())["household-outcomes"]
    assert any(
        "no rank in its archetype" in trace.customdata[0]
        for trace in figure.data
        if trace.customdata is not None
    )


# --- Intraday dispatch status caption (K4-flagged bug, fixed by K5) ---------
#
# ``_status_text`` is tested directly (not through ``render_household``)
# because the SYNTHETIC fixture result carries no ``dispatch_world_slot``
# field at all (only a real dispatched run does), so ``dispatch_active`` is
# exercised as the caller (``render_household``) would compute it, not
# re-derived from a fixture result.


def test_status_text_hides_the_dispatch_caption_for_a_control_ev(result) -> None:
    # Intraday-dispatch-v1 §2: a control EV keeps a locked/free status but
    # ignores every plan and charges by the normal rule, so it is never
    # "locked" or "re-planned" even when the run actually dispatched.
    card = replace(make_household_card(result, "ev-0000"), treated=False, dispatch_locked=True)
    assert (
        household._status_text(card, dispatch_active=True)
        == "Hold-out control group, not enrolled in smart charging"
    )


def test_status_text_shows_the_dispatch_caption_for_a_treated_ev(result) -> None:
    locked_card = replace(
        make_household_card(result, "ev-0000"), treated=True, dispatch_locked=True
    )
    assert (
        household._status_text(locked_card, dispatch_active=True)
        == "Enrolled in smart charging · Locked to its day-ahead plan"
    )
    free_card = replace(make_household_card(result, "ev-0000"), treated=True, dispatch_locked=False)
    assert (
        household._status_text(free_card, dispatch_active=True)
        == "Enrolled in smart charging · Re-plans hourly on the latest intraday price"
    )


def test_status_text_hides_the_dispatch_caption_when_the_run_did_not_dispatch(result) -> None:
    # The K4-flagged bug: ``dispatch_locked`` is False both for a free EV
    # and for every EV on a switch-off run, so a switch-off run must never
    # read its all-False column as "every EV re-plans hourly".
    card = replace(make_household_card(result, "ev-0000"), treated=True, dispatch_locked=False)
    assert household._status_text(card, dispatch_active=False) == "Enrolled in smart charging"


def test_availability_chart_has_four_legend_entries_and_the_driveway_hover(result) -> None:
    card = _with_availability(make_household_card(result, "ev-0000"), result)
    st = _render(result, card=card, choices={"household-direction": "Turn-up"})
    figure = dict(st.figures())["household-availability-turn_up"]

    legend = [t.name for t in figure.data if t.showlegend is not False and t.name]
    assert sorted(legend, key=lambda n: [t.legendrank for t in figure.data if t.name == n][0]) == [
        "P50",
        "P10–P90",
        "P5–P95",
        "Share of weeks with any turn-up",
    ]
    line = next(trace for trace in figure.data if trace.name == "P50")
    rows = card.availability.loc[card.availability["direction"].eq("turn_up")]
    assert list(line.y) == pytest.approx(list(rows.sort_values("slot_index")["p50"]))
    assert "able to move charging in" in line.customdata[0]
    assert any("P10–P90 and P5–P95" in text for text in _texts(st, "caption"))
    assert household.FIRM_MW_PENDING not in _texts(st, "info")


def test_no_action_result_gets_the_page_message_and_never_builds_a_card() -> None:
    st = RecordingStreamlit()
    pages.VIEWS[("Drivers", "Household")](st, make_result("no_action", evs=6, worlds=4))

    assert _texts(st, "info") == [pages.NO_ACTION_MESSAGE]
    assert st.figures() == []


def test_action_result_without_household_frames_says_so(result) -> None:
    calls: list[str] = []
    st = _render(replace(result, household_ev_world=None), calls=calls)

    assert _texts(st, "info") == [household.NO_FRAMES_MESSAGE]
    assert calls == []


def test_ev_picker_is_the_one_ev_widget(result) -> None:
    """Both lenses draw the EV selectbox with the same key, label, options and default (O8)."""

    def picker_call(st):
        return next(kw for name, _, kw in st.calls if kw.get("key") == "one-ev-unit")

    household_st = _render(result)
    from fixtures.result_fixture import replay_one_ev, replay_one_ev_bands

    one_ev_st = RecordingStreamlit()
    one_ev.render_one_ev(
        one_ev_st, result, replay_one_ev=replay_one_ev, replay_one_ev_bands=replay_one_ev_bands
    )
    a, b = picker_call(household_st), picker_call(one_ev_st)
    same = ("options", "index", "key", "label_visibility")
    assert {k: a[k] for k in same} == {k: b[k] for k in same}
    assert [a["format_func"](u) for u in a["options"]] == [
        b["format_func"](u) for u in b["options"]
    ]


def test_ev_choice_survives_a_switch_between_one_ev_and_household() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    _small_draft(app)
    _run(app)
    _show(app, "Drivers", "One EV")
    app.get_by_key("one-ev-unit").select_index(3).run()
    chosen = app.get_by_key("one-ev-unit").value
    assert chosen != app.get_by_key("one-ev-unit").options[0]

    app.get_by_key("lens::drivers").set_value("Household").run()
    assert not app.exception
    assert app.get_by_key("one-ev-unit").value == chosen
    assert any('class="axle-kpi"' in element.value for element in app.markdown)

    app.get_by_key("lens::drivers").set_value("One EV").run()
    assert app.get_by_key("one-ev-unit").value == chosen


def test_share_panel_range_never_passes_100_percent() -> None:
    # Final critique B-11: the "Ready at departure" axis ran 95–105% because the
    # display pad crossed 100%; a share is clipped to its 0–100% domain.
    columns = [f"{g}{q}" for g in ("", "cohort_", "fleet_") for q in ("p10", "p50", "p90")]
    outcome = pd.Series(dict.fromkeys(columns, 1.0) | {"p10": 0.97})
    low, high = household._padded_range(outcome, 100.0, bounds=(0.0, 100.0))
    assert high == 100.0
    assert 90.0 < low < 97.0
    # Unbounded metrics keep their symmetric pad.
    assert household._padded_range(outcome, 100.0)[1] > 100.0


def test_cost_panel_is_in_pence_per_kwh() -> None:
    # Final critique B-10: the customer charging cost reads p/kWh at 1 dp.
    titles = [title for _, title, _ in household._PANELS]
    assert "Smart cost (p/kWh)" in titles
    assert household._per_kwh(0.0931) == "9.3 p/kWh"
    assert household._PANEL_SCALE["home_cost_gbp_per_kwh_selected"] == 100.0
