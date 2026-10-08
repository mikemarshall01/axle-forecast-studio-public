"""AppTest checks of the dashboard shell and its run lifecycle.

The rule under test (AGENTS.md): a full Monte Carlo starts only from an
explicit valid Run click. Opening, navigating, changing lens, editing in the
dialog, Reset, switching model and reruns must never call a model runner.

Runs here are real ``run_forecast`` runs at a small size (30 EVs x 4
simulated weeks), so every page and lens is checked against real model
output, not a fixture.
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import axle_studio.ui.run_controller as run_controller
from axle_studio.model import assumptions
from axle_studio.model.forecast import ForecastResult, run_forecast_from_assumptions
from axle_studio.model.individual import replay_one_ev
from axle_studio.model.summaries import RunSummary, compare_runs
from axle_studio.ui.pages import NO_ACTION_MESSAGE
from axle_studio.ui.registry import PAGES
from axle_studio.ui.views import one_ev

APP_PATH = Path(__file__).parents[2] / "streamlit_app.py"
SMALL_FLEET = 30
SMALL_WORLDS = 4


class RunnerSpy:
    """Stands in for ``run_forecast_from_assumptions``, recording every call.

    It delegates to the real function so pages render real results;
    ``fail_with`` makes later calls raise instead.  Each call is recorded as
    ``(model, start_local_date, values)``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, dict]] = []
        self.fail_with: Exception | None = None

    def __call__(self, start_local_date, *, model, values, event_presets=()):
        self.calls.append((model, start_local_date, dict(values)))
        if self.fail_with is not None:
            raise self.fail_with
        return run_forecast_from_assumptions(
            start_local_date, model=model, values=values, event_presets=event_presets
        )


@pytest.fixture
def spy(monkeypatch) -> RunnerSpy:
    runner_spy = RunnerSpy()
    monkeypatch.setattr(run_controller, "run_forecast_from_assumptions", runner_spy)
    return runner_spy


def _open() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    return app


def _go(app: AppTest, page_name: str) -> None:
    """Switch page the way the top navigation does, keeping the session."""

    app._page_hash = next(
        page_hash
        for page_hash, page in app._registered_pages.items()
        if page["page_name"] == page_name
    )
    app.run()
    assert not app.exception


def _show_group_of(app: AppTest, key: str) -> None:
    """Pick the dialog's group pill that holds field ``key`` (only that group is drawn)."""

    # AppTest reruns the whole script on a pill click, which closes the
    # dialog (a browser reruns only the dialog), so reopen it: the dialog
    # remembers the group last chosen.
    name = key.removeprefix("assumption::")
    group = next(record.group for record in assumptions.editable_records() if record.name == name)
    if app.get_by_key("assumptions-group").value != group:
        app.get_by_key("assumptions-group").set_value(group).run()
        app.get_by_key("edit-assumptions").click().run()


def _edit(app: AppTest, key: str, value: object) -> None:
    """Open Edit assumptions, pick the field's group and change it, as a user would."""

    app.get_by_key("edit-assumptions").click().run()
    _show_group_of(app, key)
    app.get_by_key(key).set_value(value).run()
    assert not app.exception


def _small_draft(app: AppTest) -> None:
    """Shrink the draft so a real Run takes well under a second."""

    _edit(app, "assumption::vehicle_count", SMALL_FLEET)
    _edit(app, "assumption::evaluation_world_count", SMALL_WORLDS)


def _click_run(app: AppTest) -> None:
    app.get_by_key("run-simulation").click().run()
    assert not app.exception


def _chip(app: AppTest) -> str | None:
    """The run bar chip's word, or None before the first run.

    ``st.badge`` renders as markdown such as ``:green-badge[Current]``.
    """

    for element in app.markdown:
        match = re.fullmatch(r":\w+-badge\[(.*)\]", element.value)
        if match:
            return match.group(1)
    return None


def _captions(app: AppTest) -> list[str]:
    return [element.value for element in app.caption]


def _set_lens(app: AppTest, page_slug: str, lens_name: str) -> None:
    app.get_by_key(f"lens::{page_slug}").set_value(lens_name).run()
    assert not app.exception, (page_slug, lens_name)


def _visit_every_page_and_lens(app: AppTest) -> None:
    for page in PAGES:
        _go(app, page.name)
        for lens in page.lenses:
            _set_lens(app, page.slug, lens.name)


def _chart_count(app: AppTest) -> int:
    return len(app.get("plotly_chart"))


def test_opening_navigating_editing_resetting_and_switching_never_run(spy) -> None:
    app = _open()

    # How it works is the landing page (Fable landing pass): a first-time
    # reader sees what is calculated and assumed before any result.
    assert app.subheader[0].value == "How it works"
    assert app.session_state["model_mode"] == run_controller.MODEL_ACTION
    assert "No result yet" in _captions(app)
    assert _chip(app) is None
    _visit_every_page_and_lens(app)
    _edit(app, "assumption::home_charging_power_kw", 3.6)
    _edit(app, "assumption::vehicle_count", 200)
    app.get_by_key("edit-assumptions").click().run()
    app.get_by_key("reset-to-defaults").click().run()
    app.get_by_key("model_mode").set_value(run_controller.MODEL_NO_ACTION).run()
    app.run()

    assert not app.exception
    assert spy.calls == []
    assert run_controller.run_history(app.session_state) == []


def test_run_uses_the_dialog_values_and_carries_the_records_it_ran_with(spy) -> None:
    app = _open()
    _small_draft(app)
    # Run from another page: the edited value must survive the dialog closing.
    _go(app, "Drivers")
    _click_run(app)

    assert len(spy.calls) == 1
    model, _, values = spy.calls[0]
    assert model == "action"
    result = run_controller.active_result(app.session_state)
    assert isinstance(result, ForecastResult)
    # Regression: the fleet-size slider once showed 200 while the run used 1,000.
    assert values["vehicle_count"] == SMALL_FLEET
    assert values["evaluation_world_count"] == SMALL_WORLDS
    assert (result.vehicle_count, result.world_count) == (SMALL_FLEET, SMALL_WORLDS)
    # Every other value is the assumptions-module default, and the result
    # carries the records it ran with (decision 0004 item 21).
    assert values == assumptions.editable_defaults() | {
        "vehicle_count": SMALL_FLEET,
        "evaluation_world_count": SMALL_WORLDS,
    }
    assert result.assumptions == assumptions.result_assumptions(values)
    assert _chip(app) == "Current"
    assert any(
        caption.startswith(f"Run 1 · {SMALL_FLEET} EVs · {SMALL_WORLDS} weeks")
        for caption in _captions(app)
    )


def test_empty_state_run_button_runs_once(spy) -> None:
    app = _open()
    # The empty-state body (with its own Run button) is not on the landing
    # page (How it works needs no result, design section 3.2); Overview is
    # the nearest page that shows it.
    _go(app, "Overview")
    _small_draft(app)
    app.get_by_key("run-simulation-empty").click().run()

    assert len(spy.calls) == 1
    assert _chip(app) == "Current"


def test_empty_state_shows_the_intro_archetypes_and_measured_run_time() -> None:
    # Goal review action 9: the design's three-sentence intro and six
    # archetypes with source shares, and an honest run-time estimate so
    # there is something to read while the default 1,000 x 100 run works.
    # Final critique B-14: the promised "70-90 s" went stale once the
    # plan-status wiring lane made the default run heavier (measured
    # 3 min 0 s-3 min 50 s); the words now come off a named, re-timeable
    # constant rather than a fixed string baked into the sentence.
    app = _open()
    # The intro and archetype summary are Overview's empty state, not the
    # landing page's (How it works needs no result, design section 3.2).
    _go(app, "Overview")

    markdown_text = "\n".join(element.value for element in app.markdown)

    assert "This app forecasts how much" in markdown_text
    assert "Run simulation to forecast 1,000 simulated EV drivers over 100 possible weeks" in (
        markdown_text
    )
    assert (
        f"about {run_controller.DEFAULT_RUN_MINUTES_ESTIMATE} minutes at 1,000 EVs × 100 weeks, "
        "longer if the machine is busy"
    ) in markdown_text
    for name, share in (
        ("Average (UK)", "40%"),
        ("Intelligent Octopus average", "30%"),
        ("Infrequent charging", "10%"),
        ("Infrequent driving", "10%"),
        ("Scheduled charging", "9%"),
        ("Always plugged-in", "1%"),
    ):
        assert name in markdown_text and share in markdown_text
    assert "(Source)" in markdown_text


def test_run_from_the_dialog_footer_runs_once(spy) -> None:
    app = _open()
    _small_draft(app)
    app.get_by_key("edit-assumptions").click().run()
    app.get_by_key("run-from-dialog").click().run()

    assert len(spy.calls) == 1
    assert _chip(app) == "Current"


def test_edit_after_run_keeps_result_and_marks_it_stale(spy) -> None:
    app = _open()
    _small_draft(app)
    _click_run(app)
    previous = run_controller.active_result(app.session_state)

    _edit(app, "assumption::home_charging_power_kw", 3.6)

    assert len(spy.calls) == 1
    assert run_controller.active_result(app.session_state) is previous
    assert _chip(app) == "Stale · 1 change"
    assert "Showing Run 1; draft differs" in _captions(app)

    app.get_by_key("edit-assumptions").click().run()
    app.get_by_key("reset-to-active").click().run()
    assert _chip(app) == "Current"
    assert len(spy.calls) == 1


def test_model_switch_marks_stale_and_next_run_uses_no_action_model(spy) -> None:
    app = _open()
    _small_draft(app)
    _click_run(app)

    app.get_by_key("model_mode").set_value(run_controller.MODEL_NO_ACTION).run()
    assert len(spy.calls) == 1
    assert _chip(app) == "Stale · 1 change"

    _click_run(app)
    assert [call[0] for call in spy.calls] == ["action", "no_action"]
    # Both models take the same values; public top-ups happen in both.
    assert spy.calls[0][2] == spy.calls[1][2]
    labels = [record.label for record in run_controller.run_history(app.session_state)]
    action_label = run_controller.MODEL_LABELS[run_controller.MODEL_ACTION].lower()
    no_action_label = run_controller.MODEL_LABELS[run_controller.MODEL_NO_ACTION].lower()
    assert labels == [
        f"Run 1 · Fleet size 1000 → {SMALL_FLEET} EVs (+1 more)",
        f"Run 2 · {action_label} → {no_action_label}",
    ]


def test_failed_run_keeps_previous_result_and_shows_one_banner(spy) -> None:
    app = _open()
    _small_draft(app)
    _click_run(app)
    previous = run_controller.active_result(app.session_state)

    spy.fail_with = RuntimeError("synthetic failure")
    _edit(app, "assumption::seed", 7)
    _click_run(app)

    assert run_controller.active_result(app.session_state) is previous
    assert _chip(app) == "Run failed"
    assert "Still showing Run 1" in _captions(app)
    assert [element.value for element in app.error] == ["Run failed: synthetic failure"]


def test_invalid_draft_disables_run(spy) -> None:
    # The widgets keep each value inside its record's bounds; the one rule the
    # widgets cannot enforce alone is target above threshold (decision 0004
    # item 32), so a target at the threshold reaches the invalid state.
    app = _open()
    # The empty-state body button is not on the landing page (How it works
    # needs no result, design section 3.2); Overview is the nearest page
    # that shows it.
    _go(app, "Overview")
    _edit(app, "assumption::public_top_up_threshold_soc_percent", 50.0)
    _edit(app, "assumption::public_top_up_target_soc_percent", 50.0)

    assert app.get_by_key("run-simulation").disabled
    assert app.get_by_key("run-simulation-empty").disabled
    assert _chip(app) == "Draft invalid"
    assert any(
        "Public top-up target SoC must be above the top-up threshold" in caption
        for caption in _captions(app)
    )
    app.run()
    assert spy.calls == []


# Pages that draw at least one chart from a result (How it works and the
# Compare empty state draw none).
_CHART_VIEWS = {
    ("Overview", "At a glance"),
    ("Drivers", "Plug-ins"),
    ("Drivers", "Sessions"),
    ("Drivers", "One EV"),
    ("Drivers", "Archetypes"),
    ("Drivers", "Fleet week"),
}
_ACTION_CHART_VIEWS = {
    ("Smart charging", "1 Plan"),
    ("Smart charging", "3 Value and risk"),
    ("Trading", "1 Market"),
    ("Trading", "2 Position"),
    ("Trading", "3 P&L and risk"),
    ("Supplier", "1 Availability and cost curve"),
    ("Supplier", "2 Positions"),
    ("Supplier", "3 Supplier P&L"),
    ("Supplier", "5 Charger makers"),
    ("Partners", "Driver"),
    ("Partners", "Energy supplier"),
    ("Partners", "Charger maker"),
    ("Partners", "Carmaker"),
    ("Partners", "Fleet and leasing"),
}


@pytest.mark.parametrize("model", run_controller.MODELS)
def test_every_page_and_lens_renders_a_real_result(spy, model) -> None:
    app = _open()
    _small_draft(app)
    app.get_by_key("model_mode").set_value(model).run()
    _click_run(app)

    expected_charts = _CHART_VIEWS | (_ACTION_CHART_VIEWS if model == "action" else set())
    for page in PAGES:
        _go(app, page.name)
        for lens in page.lenses or (None,):
            if lens is not None:
                _set_lens(app, page.slug, lens.name)
            key = (page.name, lens.name if lens else None)
            if key in expected_charts:
                assert _chart_count(app) >= 1, key
            # Once a result exists no view falls back to the empty state.
            assert "run-simulation-empty" not in [button.key for button in app.button], key
    assert [call[0] for call in spy.calls] == [model]


def test_no_action_result_on_wholesale_action_offers_a_switch_without_running(spy) -> None:
    app = _open()
    _small_draft(app)
    app.get_by_key("model_mode").set_value(run_controller.MODEL_NO_ACTION).run()
    _click_run(app)
    _go(app, "Smart charging")
    for lens in ("1 Plan", "2 Response", "3 Value and risk"):
        _set_lens(app, "smart-charging", lens)
        assert NO_ACTION_MESSAGE in [element.value for element in app.info]
    app.get_by_key("switch-to-action::value").click().run()

    assert app.session_state["model_mode"] == run_controller.MODEL_ACTION
    assert len(spy.calls) == 1


def test_one_ev_replays_the_chosen_ev_from_the_stored_result(spy) -> None:
    app = _open()
    _small_draft(app)
    _click_run(app)
    _go(app, "Drivers")
    _set_lens(app, "drivers", "One EV")
    result = run_controller.active_result(app.session_state)

    # The view replayed one EV through model/individual.py (cached on the
    # result), in the representative week, without a second full run.
    # Goal review item 11: the default EV is the first Average (UK) unit
    # (decision 0004 item 42's plug_probability 1.0 is that cohort's top
    # would-plug-in rate, not a nightly guarantee -- like every cohort
    # except Always plugged-in it still goes through the shared skip share,
    # see one_ev.ev_picker), falling back to row 0 if none was sampled --
    # the same rule one_ev.render_one_ev applies, not row 0 unconditionally.
    unit_ids = result.units["unit_id"]
    average_uk_units = unit_ids.loc[result.units["cohort_id"].eq("average_uk")]
    default_ev = average_uk_units.iat[0] if len(average_uk_units) else unit_ids.iat[0]
    replays = [key for key in result.replay_state.replay_cache if key[0] == "replay"]
    assert replays == [("replay", default_ev, result.representative_world_id)]
    assert _chart_count(app) >= 1

    second_ev = next(uid for uid in unit_ids if uid != default_ev)
    app.selectbox[0].set_value(second_ev).run()
    assert not app.exception
    assert ("replay", second_ev, result.representative_world_id) in result.replay_state.replay_cache
    assert len(spy.calls) == 1


def test_compare_matched_runs_from_slim_records(spy) -> None:
    app = _open()
    _small_draft(app)
    _click_run(app)
    _edit(app, "assumption::home_charging_power_kw", 3.6)
    _click_run(app)
    _go(app, "Compare")

    history = run_controller.run_history(app.session_state)
    assert [record.result is None for record in history] == [True, False]
    assert all(isinstance(record, RunSummary) for record in history)
    comparison = compare_runs(*history)
    assert comparison.matched_futures
    assert comparison.illustrative_total_available
    captions = _captions(app)
    assert f"Matched futures: yes (seed 42, {SMALL_WORLDS} weeks)" in captions
    # Compare names the change from the records the runs carry (M3b).
    assert "Changed: Home charging power 7.0 → 3.6 kW" in captions
    assert _chart_count(app) == 1
    assert len(spy.calls) == 2


def test_compare_after_a_model_change_pairs_the_same_futures(spy) -> None:
    app = _open()
    _small_draft(app)
    _click_run(app)
    app.get_by_key("model_mode").set_value(run_controller.MODEL_NO_ACTION).run()
    _click_run(app)
    _go(app, "Compare")

    comparison = compare_runs(*run_controller.run_history(app.session_state))
    # Decision 0004 item 38 removed the planning world, so both models draw
    # the same worlds for one seed and Compare pairs them.
    assert comparison.matched_futures
    assert not comparison.illustrative_total_available
    # The model change is named, so "no changes" never shows (review, 28 Sep).
    action_label = run_controller.MODEL_LABELS[run_controller.MODEL_ACTION]
    no_action_label = run_controller.MODEL_LABELS[run_controller.MODEL_NO_ACTION]
    assert f"Changed: Model: {action_label} → {no_action_label}" in _captions(app)
    assert "No assumption or setting changes between these runs." not in _captions(app)
    assert _chart_count(app) == 1


def test_assumptions_lens_shows_the_records_before_and_after_a_run(spy) -> None:
    app = _open()
    _go(app, "How it works")
    _set_lens(app, "how-it-works", "Assumptions")
    # Before a run: every record at its default (model/assumptions.py), split
    # across the always-visible shared table and the collapsed cohort table
    # (goal review action 14).
    before_shared, before_cohort, _references = (frame.value for frame in app.dataframe)
    assert len(before_shared) + len(before_cohort) == len(assumptions.assumption_rows())

    _small_draft(app)
    _edit(app, "assumption::home_charging_power_kw", 3.6)
    _click_run(app)
    # After a run: the records the result ran with, edited value included.
    after_shared, after_cohort, _references = (frame.value for frame in app.dataframe)
    assert len(after_shared) + len(after_cohort) == len(assumptions.result_assumptions())
    power = after_shared.loc[after_shared["Label"].eq("Home charging power"), "Value"]
    assert power.tolist() == ["3.6"]


def test_each_page_has_one_title_and_no_heading_repeating_the_lens(spy) -> None:
    app = _open()
    _small_draft(app)
    _click_run(app)
    for page in PAGES:
        _go(app, page.name)
        for lens in page.lenses or (None,):
            if lens is not None:
                app.get_by_key(f"lens::{page.slug}").set_value(lens.name).run()
            headings = [element.value for element in (*app.header, *app.subheader)]
            assert headings.count(page.name) == 1
            assert lens is None or lens.name not in headings
            assert (lens.question if lens else page.question) in _captions(app)


def test_lens_choice_survives_navigation(spy) -> None:
    app = _open()
    _go(app, "Drivers")
    app.get_by_key("lens::drivers").set_value("Archetypes").run()
    _go(app, "Overview")
    _go(app, "Drivers")

    assert app.get_by_key("lens::drivers").value == "Archetypes"


def test_run_history_keeps_three_and_only_the_latest_full_result(spy) -> None:
    app = _open()
    _small_draft(app)
    for seed in (1, 2, 3, 4):
        _edit(app, "assumption::seed", seed)
        _click_run(app)

    history = run_controller.run_history(app.session_state)
    assert [record.number for record in history] == [2, 3, 4]
    assert [record.result is None for record in history] == [True, True, False]
    assert all(not record.fleet_world_intervals.empty for record in history)
    assert all(not record.plug_in_world_kpis.empty for record in history)
    # Seeds differ, so the kept slim records compare as unmatched futures.
    assert not compare_runs(history[0], history[1]).matched_futures


def test_run_is_the_only_primary_button_in_the_bar() -> None:
    app = _open()

    assert app.get_by_key("run-simulation").proto.type == "primary"
    assert app.get_by_key("edit-assumptions").proto.type == "secondary"


def test_click_survives_a_script_stopped_before_the_run(spy, monkeypatch) -> None:
    # Stand-in for a nav or lens click that stops the script while the page
    # draws: the end-of-script run never happens in that script run. AppTest
    # executes streamlit_app.py afresh, so it picks up this patched function.
    app = _open()
    _small_draft(app)
    real_execute = run_controller.execute_requested_run
    monkeypatch.setattr(run_controller, "execute_requested_run", lambda st, slot: None)
    _click_run(app)
    assert spy.calls == []
    assert app.session_state["run_requested"] is True

    monkeypatch.setattr(run_controller, "execute_requested_run", real_execute)
    app.run()

    assert len(spy.calls) == 1
    assert app.session_state["run_requested"] is False
    assert _chip(app) == "Current"


def test_one_ev_finds_public_top_ups_in_a_real_replay() -> None:
    # Regression: the view once required location == "public_charging", which
    # the real replay never reports for a top-up (it takes no modelled time),
    # so top-up markers and the session table never appeared on real results.
    values = assumptions.editable_defaults() | {
        "vehicle_count": SMALL_FLEET,
        "evaluation_world_count": SMALL_WORLDS,
    }
    result = run_controller.run_model(
        run_controller.MODEL_ACTION, values, run_controller.london_today()
    )
    public = result.fleet_world_intervals.groupby("world_id")["public_import_kwh"].sum()
    world = int(public.idxmax())
    replays = [replay_one_ev(result, unit, world) for unit in result.units["unit_id"]]
    topped_up = [replay for replay in replays if replay.intervals["public_import_kwh"].gt(0).any()]

    assert public.max() > 0 and topped_up
    sessions = one_ev._public_topup_sessions(topped_up[0].intervals)
    assert len(sessions) >= 1


def test_compare_explains_runs_with_different_week_counts(spy) -> None:
    app = _open()
    _small_draft(app)
    _click_run(app)
    _edit(app, "assumption::evaluation_world_count", SMALL_WORLDS + 1)
    _click_run(app)
    _go(app, "Compare")

    assert [element.value for element in app.info] == [
        f"Cannot compare: Run 1 has {SMALL_WORLDS} simulated weeks, Run 2 has "
        f"{SMALL_WORLDS + 1}. Keep weeks and start date the same to compare."
    ]
    assert _chart_count(app) == 0


def test_compare_explains_runs_either_side_of_midnight(spy, monkeypatch) -> None:
    app = _open()
    _small_draft(app)
    _click_run(app)
    first = run_controller.latest_run(app.session_state).start_local_date
    next_day = first + timedelta(days=1)
    # A London midnight rollover moves the draft's start date (refresh_start_date).
    monkeypatch.setattr(run_controller, "london_today", lambda: next_day)
    _click_run(app)
    _go(app, "Compare")

    assert [element.value for element in app.info] == [
        f"Cannot compare: Run 1 starts {first:%a} {first.day} {first:%b}, Run 2 starts "
        f"{next_day:%a} {next_day.day} {next_day:%b}. Keep weeks and start date the same "
        "to compare."
    ]
    assert len(spy.calls) == 2


def test_edit_dialog_says_there_is_no_active_run_before_the_first_run(spy) -> None:
    app = _open()
    app.get_by_key("edit-assumptions").click().run()

    captions = _captions(app)
    assert any("No run yet" in caption for caption in captions)
    assert not any("Active run: none" in caption for caption in captions)
