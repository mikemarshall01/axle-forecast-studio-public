"""AppTest checks of the Compare page's A/B run selectors (views/compare.py).

Real small runs (30 EVs x 4 simulated weeks, ``RunnerSpy`` delegating to the
real model), not the synthetic fixture used in ``test_compare_view.py``: the
milestone finding this fixes is about run *history* plumbing
(``run_controller.run_history`` -> ``compare.render_compare``), which a
fixture-backed unit test bypasses entirely.

Milestone verifier finding (docs/DASHBOARD_DESIGN.md 4.4; decision 0004
items 6, 28, 35): Compare hard-coded ``records[-2], records[-1]``, so three
runs where only the first two could be paired showed nothing but "Cannot
compare", with no way to pick Run 1 vs Run 2. Repro below: run small, run
small with an edit (pairable with Run 1), run with a different world count
(unpairable with Run 2, the new default) -- the fix must default to Run 1 vs
Run 2 instead and say why.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import axle_studio.ui.run_controller as run_controller
from axle_studio.model import assumptions
from axle_studio.model.forecast import run_forecast_from_assumptions

APP_PATH = Path(__file__).parents[2] / "streamlit_app.py"
SMALL_FLEET = 30
SMALL_WORLDS = 4


class RunnerSpy:
    """Delegates to the real model, recording every call (mirrors test_app_shell.py)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, dict]] = []

    def __call__(self, start_local_date, *, model, values, event_presets=()):
        self.calls.append((model, start_local_date, dict(values)))
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
    app._page_hash = next(
        page_hash
        for page_hash, page in app._registered_pages.items()
        if page["page_name"] == page_name
    )
    app.run()
    assert not app.exception


def _edit(app: AppTest, key: str, value: object) -> None:
    app.get_by_key("edit-assumptions").click().run()
    # Only the chosen group's fields are drawn (group pills, polish plan G9).
    name = key.removeprefix("assumption::")
    group = next(record.group for record in assumptions.editable_records() if record.name == name)
    if app.get_by_key("assumptions-group").value != group:
        # AppTest's full rerun closes the dialog; it reopens on that group.
        app.get_by_key("assumptions-group").set_value(group).run()
        app.get_by_key("edit-assumptions").click().run()
    app.get_by_key(key).set_value(value).run()
    assert not app.exception


def _small_draft(app: AppTest) -> None:
    _edit(app, "assumption::vehicle_count", SMALL_FLEET)
    _edit(app, "assumption::evaluation_world_count", SMALL_WORLDS)


def _click_run(app: AppTest) -> None:
    app.get_by_key("run-simulation").click().run()
    assert not app.exception


def _captions(app: AppTest) -> list[str]:
    return [element.value for element in app.caption]


def _chart_count(app: AppTest) -> int:
    return len(app.get("plotly_chart"))


def _ab_selection(app: AppTest) -> tuple[int, int]:
    return app.get_by_key("compare-run-a").value, app.get_by_key("compare-run-b").value


def test_default_falls_back_to_the_most_recent_pairable_pair_when_latest_is_not(spy) -> None:
    # The repro: Run 1 and Run 2 share a world count (pairable); Run 3 does
    # not (unpairable with Run 2, the plain previous-vs-latest default).
    app = _open()
    _small_draft(app)
    _click_run(app)  # Run 1: SMALL_FLEET x SMALL_WORLDS
    _edit(app, "assumption::home_charging_power_kw", 3.6)
    _click_run(app)  # Run 2: same shape, one assumption changed
    _edit(app, "assumption::evaluation_world_count", SMALL_WORLDS + 1)
    _click_run(app)  # Run 3: a different world count
    _go(app, "Compare")

    assert _ab_selection(app) == (1, 2)
    notes = " ".join(_captions(app))
    assert "Cannot compare: Run 2 has" in notes and "Run 3 has" in notes
    assert "Showing Run 1 vs Run 2 instead." in notes
    assert _chart_count(app) == 1

    # The presenter can still explicitly pick the unpairable pair; the plain
    # message shows with no "instead" note, exactly as before this fix, and
    # the fallback stays reachable by picking Run 1 vs Run 2 again.
    app.get_by_key("compare-run-a").set_value(2).run()
    app.get_by_key("compare-run-b").set_value(3).run()

    assert [element.value for element in app.info] == [
        f"Cannot compare: Run 2 has {SMALL_WORLDS} simulated weeks, Run 3 has "
        f"{SMALL_WORLDS + 1}. Keep weeks and start date the same to compare."
    ]
    assert not any("instead" in caption for caption in _captions(app))
    assert _chart_count(app) == 0
    assert len(spy.calls) == 3


def test_a_and_b_the_same_run_shows_a_short_message_not_a_chart(spy) -> None:
    app = _open()
    _small_draft(app)
    _click_run(app)
    _edit(app, "assumption::home_charging_power_kw", 3.6)
    _click_run(app)
    _go(app, "Compare")
    assert _chart_count(app) == 1

    app.get_by_key("compare-run-b").set_value(1).run()

    assert _ab_selection(app) == (1, 1)
    assert [element.value for element in app.info] == ["Choose two different runs to compare."]
    assert _chart_count(app) == 0


def test_unmatched_futures_chart_title_says_so_and_never_calls_it_paired(spy) -> None:
    app = _open()
    _small_draft(app)
    _click_run(app)
    _edit(app, "assumption::seed", 7)
    _click_run(app)
    _go(app, "Compare")

    comparison_matched = any(
        caption.startswith("Matched futures: yes") for caption in _captions(app)
    )
    assert not comparison_matched
    titles = [element.value for element in app.markdown if element.value.startswith("**B")]
    assert titles == ["**B − A (not matched futures)**"]
    assert "paired" not in titles[0]
    assert _chart_count(app) == 1


def test_changing_the_ab_selection_never_runs_the_model(spy) -> None:
    app = _open()
    _small_draft(app)
    _click_run(app)
    _edit(app, "assumption::home_charging_power_kw", 3.6)
    _click_run(app)
    _edit(app, "assumption::seed", 7)
    _click_run(app)
    _go(app, "Compare")
    assert len(spy.calls) == 3

    app.get_by_key("compare-run-a").set_value(1).run()
    assert not app.exception
    app.get_by_key("compare-run-b").set_value(2).run()
    assert not app.exception
    app.get_by_key("compare-run-a").set_value(2).run()
    app.get_by_key("compare-run-b").set_value(3).run()
    assert not app.exception

    # A/B selection lives in session state keys no run-triggering code
    # reads (compare.py's ``_resolve_pair``); only the three explicit Run
    # clicks above may have called the model.
    assert len(spy.calls) == 3
    assert _ab_selection(app) == (2, 3)
