"""Replay ▸ Fleet and Replay ▸ Customer on the SYNTHETIC fixture (replay contract v1 §3, §5).

The views are rendered into a recording fake ``st``; the Plotly figure handed
to ``st.plotly_chart`` is inspected: 169 frames, what each frame carries, the
hidden future, the payload bound, the controls and the per-result cache.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fixtures.replay_contract import replay_one_ev_timeline
from fixtures.result_fixture import make_result, replay_one_ev
from kpi_calls import kpi_calls

from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.model.individual import replay_one_ev as model_replay_one_ev
from axle_studio.model.individual import replay_one_ev_timeline as model_replay_one_ev_timeline
from axle_studio.ui import pages
from axle_studio.ui.views import replay as view

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from dashboard_lint import lint_figure  # noqa: E402

K = 169
SLOTS = 336
PAYLOAD_BOUND = 4_500_000  # bytes of figure JSON per lens (contract §3.5, lead ruling 7)


class _Container:
    def __enter__(self) -> _Container:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class RecordingStreamlit:
    """Fake ``st``; widgets return their default unless ``choices`` names the key."""

    def __init__(self, *, choices: dict[str, object] | None = None) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._choices = choices or {}

    def __getattr__(self, name: str):
        def record(*args: object, **kwargs: object) -> object:
            self.calls.append((name, args, kwargs))
            key = kwargs.get("key")
            if key in self._choices:
                return self._choices[key]
            if name == "selectbox":
                return kwargs["options"][kwargs.get("index") or 0]
            if name == "columns":
                spec = args[0]
                return [self for _ in range(spec if isinstance(spec, int) else len(spec))]
            if name == "expander":
                return _Container()
            return None

        return record


def calls_for(st: RecordingStreamlit, name: str) -> list[tuple[tuple, dict]]:
    return [(args, kwargs) for call, args, kwargs in st.calls if call == name]


def _figure(st: RecordingStreamlit):
    ((args, _),) = calls_for(st, "plotly_chart")
    return args[0]


def _fleet(result, **choices):
    st = RecordingStreamlit(choices=choices)
    view.render_replay_fleet(st, result)
    return st


def _customer(result, timeline=replay_one_ev_timeline, **choices):
    st = RecordingStreamlit(choices=choices)
    view.render_replay_customer(
        st, result, replay_one_ev=replay_one_ev, replay_one_ev_timeline=timeline
    )
    return st


@pytest.fixture(scope="module")
def result():
    return make_result("action", evs=6, worlds=12)


def _trace(figure, name: str) -> int:
    return next(i for i, trace in enumerate(figure.data) if trace.name == name)


def _frame_y(figure, k: int, index: int) -> np.ndarray:
    frame = figure.frames[k]
    position = list(frame.traces).index(index)
    return np.asarray(frame.data[position].y, dtype=float)


def _now_lines(figure) -> list[int]:
    return [
        i
        for i, trace in enumerate(figure.data)
        if trace.type == "scatter"
        and trace.mode == "lines"
        and trace.name is None
        and trace.hoverinfo == "skip"
    ]


# --- Rendering and states ------------------------------------------------------


def test_fleet_renders_with_trading_frames(result) -> None:
    st = _fleet(result)

    figure = _figure(st)
    assert len(figure.frames) == K
    names = {trace.name for trace in figure.data}
    assert {"Realised price", "Day-ahead (published)", "Latest intraday"} <= names
    assert {"Unmanaged", "Smart", "Sold turn-down"} <= names
    assert not calls_for(st, "info")


def test_fleet_without_position_updates_drops_the_turn_down_line(result) -> None:
    trimmed = replace(
        result,
        position_updates=None,
        replay_week=replace(result.replay_week, position_kwh=None),
    )

    figure = _figure(_fleet(trimmed))

    names = {trace.name for trace in figure.data}
    assert "Sold turn-down" not in names
    assert "Sold at now" not in names


def test_fleet_without_the_ledger_drops_the_trading_figure(result) -> None:
    trimmed = replace(
        result,
        deviation_world_slot=None,
        position_updates=None,
        trading_week_world=None,
        replay_week=replace(result.replay_week, position_kwh=None, trading_cash_to_date_gbp=None),
    )

    st = _fleet(trimmed)

    figure = _figure(st)
    counter = figure.frames[40].data[0].text[0]
    assert "Trading P&L" not in counter
    assert "Drivers’ saving so far (illustrative)" in counter
    # Without trading frames the import lines come from the fleet frame.
    assert {"Unmanaged", "Smart"} <= {trace.name for trace in figure.data}


@pytest.mark.parametrize("lens", ["fleet", "customer"])
def test_replay_unavailable_shows_one_message(result, lens: str) -> None:
    old = replace(result, replay_week=None)

    st = _fleet(old) if lens == "fleet" else _customer(old)

    assert [args[0] for args, _ in calls_for(st, "info")] == [view.UNAVAILABLE_MESSAGE]
    assert not calls_for(st, "plotly_chart")


@pytest.mark.parametrize("lens", ["Fleet", "Customer"])
def test_no_action_result_gets_the_action_message(lens: str) -> None:
    st = RecordingStreamlit()

    pages.VIEWS[("Replay", lens)](st, make_result("no_action", evs=6, worlds=6))

    assert [args[0] for args, _ in calls_for(st, "info")] == [pages.NO_ACTION_MESSAGE]
    assert len(calls_for(st, "button")) == 1
    assert not calls_for(st, "plotly_chart")


def test_customer_renders_with_the_timeline(result) -> None:
    st = _customer(result)

    figure = _figure(st)
    assert len(figure.frames) == K
    names = {trace.name for trace in figure.data}
    assert {"Plan in force", "Plan on record, not followed", "Plugged in at home"} <= names
    assert "Smart cost so far (illustrative)" in figure.frames[60].data[0].text[0]


def test_customer_without_the_timeline_says_so_and_draws_no_plan(result) -> None:
    st = _customer(result, timeline=None)

    figure = _figure(st)
    names = {trace.name for trace in figure.data}
    assert "Plan in force" not in names
    assert "not available" in figure.frames[10].data[0].text[0]
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert any("not available for this run" in text for text in captions)


def test_customer_shows_no_dispatch_caption_when_the_fixture_never_dispatched(result) -> None:
    # The SYNTHETIC fixture's timeline carries ``dispatch_locked=None`` (no
    # run ever dispatched); the wiring must not show a locked/re-plan
    # caption regardless.
    st = _customer(result)
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert not any("Locked to its day-ahead plan" in text for text in captions)
    assert not any("Re-plans hourly" in text for text in captions)


# --- _dispatch_caption (K4-flagged bug, fixed by K5) ------------------------


class _Timeline:
    def __init__(self, dispatch_locked: bool | None) -> None:
        self.dispatch_locked = dispatch_locked


def test_dispatch_caption_names_locked_and_free_when_the_run_dispatched() -> None:
    locked = view._dispatch_caption(_Timeline(True), dispatch_active=True, treated=True)
    free = view._dispatch_caption(_Timeline(False), dispatch_active=True, treated=True)
    assert locked == "Locked to its day-ahead plan"
    assert free == "Re-plans hourly on the latest intraday price"


def test_dispatch_caption_is_none_when_the_switch_was_off() -> None:
    # The K4-flagged bug: dispatch_locked is False both for a free EV and
    # for every EV on a switch-off run, so the switch-off case must not
    # read as "every EV re-plans hourly".
    assert view._dispatch_caption(_Timeline(False), dispatch_active=False, treated=True) is None
    assert view._dispatch_caption(_Timeline(True), dispatch_active=False, treated=True) is None


def test_dispatch_caption_is_none_for_a_control_ev_even_when_dispatched() -> None:
    # Intraday-dispatch-v1 §2: a control EV keeps a locked/free status but
    # ignores every plan and charges by the normal rule.
    assert view._dispatch_caption(_Timeline(True), dispatch_active=True, treated=False) is None
    assert view._dispatch_caption(_Timeline(False), dispatch_active=True, treated=False) is None


def test_dispatch_caption_is_none_without_a_dispatch_locked_value() -> None:
    assert view._dispatch_caption(_Timeline(None), dispatch_active=True, treated=True) is None


# --- Frames --------------------------------------------------------------------


@pytest.mark.parametrize("lens", ["fleet", "customer"])
def test_frames_name_existing_traces_and_full_length_traces_carry_y_only(result, lens) -> None:
    figure = _figure(_fleet(result) if lens == "fleet" else _customer(result))

    assert [frame.name for frame in figure.frames] == [str(k) for k in range(K)]
    for frame in figure.frames:
        assert set(frame.traces) <= set(range(len(figure.data)))
        for index, data in zip(frame.traces, frame.data, strict=True):
            base = figure.data[index]
            assert data.type == base.type  # no frame changes a trace's type
            if base.x is not None and len(base.x) == SLOTS:
                assert data.x is None  # x sent once, in the base trace
                assert data.marker.to_plotly_json() == {}  # no per-frame style


@pytest.mark.parametrize("lens", ["fleet", "customer"])
def test_base_data_equals_the_active_frame_and_now_lines_follow_decisions(result, lens) -> None:
    k = 57
    st = (
        _fleet(result, **{view.JUMP_KEY: k})
        if lens == "fleet"
        else _customer(result, **{view.JUMP_KEY: k})
    )
    figure = _figure(st)
    decisions = result.replay_week.decisions["decision_utc"]

    assert figure.layout.sliders[0].active == k
    for index, data in zip(figure.frames[k].traces, figure.frames[k].data, strict=True):
        base = figure.data[index]
        if data.text is not None:
            assert list(base.text) == list(data.text)
        if data.y is not None and len(data.y) == 1 and len(base.y) > 1:
            # An all-hidden frame sends one NaN; the base keeps full length.
            assert np.isnan(np.asarray(base.y, dtype=float)).all()
        elif data.y is not None:
            # NaN (hidden) compares equal here, as it should.
            np.testing.assert_array_equal(
                np.asarray(base.y, dtype=float), np.asarray(data.y, dtype=float)
            )
    for index in _now_lines(figure):
        assert list(figure.data[index].x) == [decisions.iat[k]] * 2
        for j in (0, 100, K - 1):
            position = list(figure.frames[j].traces).index(index)
            assert list(figure.frames[j].data[position].x) == [decisions.iat[j]] * 2


def test_play_and_slider_step_without_redraw(result) -> None:
    figure = _figure(_fleet(result))

    slider = figure.layout.sliders[0]
    assert len(slider.steps) == K
    assert [step.label for step in slider.steps] == list(result.replay_week.decisions["label"])
    assert all(step.args[1]["frame"]["redraw"] is False for step in slider.steps)
    play, pause = figure.layout.updatemenus[0].buttons
    assert play.args[1]["frame"] == {"duration": view.PLAY_MS, "redraw": False}
    assert pause.args[1]["frame"]["redraw"] is False
    # The slider spans the x-axis domain so the grip sits under the now line.
    assert (slider.x, slider.len) == (figure.layout.xaxis.domain[0], 1.0)


# --- Hidden future ---------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 40, 111, 168])
def test_fleet_hides_the_future_and_the_forward_curves_hide_the_past(result, k: int) -> None:
    figure = _figure(_fleet(result))

    for name in ("Realised price", "Unmanaged", "Smart"):
        y = _frame_y(figure, k, _trace(figure, name))
        assert np.isnan(y[2 * k :]).all(), name
        if k > 0:
            assert np.isfinite(y[: 2 * k]).any(), name
    published = result.replay_week.published[k]
    for name in ("Day-ahead (published)", "Latest intraday"):
        y = _frame_y(figure, k, _trace(figure, name))
        if len(y) == SLOTS:
            assert np.isnan(y[: 2 * k]).all(), name
            assert np.isnan(y[~published]).all(), name
    expected = _frame_y(figure, k, _trace(figure, "Expected shape"))
    if len(expected) == SLOTS:
        assert np.isnan(expected[published]).all()


def test_customer_hides_the_future_and_plans_start_at_now(result) -> None:
    figure = _figure(_customer(result))

    for k in (0, 30, 90, 150, 168):
        for name in ("Unmanaged", "Smart", "Plugged in at home"):
            for index, trace in enumerate(figure.data):
                if trace.name == name:
                    y = _frame_y(figure, k, index)
                    # Shading hides the future as a zero-height fill, not NaN.
                    hidden = (
                        y[2 * k :] == 0 if name == "Plugged in at home" else np.isnan(y[2 * k :])
                    )
                    assert hidden.all(), (name, k)
        for name in ("Plan in force", "Plan on record, not followed"):
            y = _frame_y(figure, k, _trace(figure, name))
            if len(y) == SLOTS:
                assert np.isnan(y[: 2 * k]).all(), (name, k)


def test_plan_on_record_carries_the_plan_when_the_session_does_not_follow_it(result) -> None:
    # The SYNTHETIC timeline marks the last night of every second EV status 1.
    unit_id = result.units["unit_id"].iat[1]
    world = result.replay_week.world_ids[0]
    timeline = replay_one_ev_timeline(result, unit_id, world)
    st = _customer(result, **{view.EV_KEY: unit_id})
    figure = _figure(st)
    in_force = _trace(figure, "Plan in force")
    on_record = _trace(figure, "Plan on record, not followed")

    not_followed = [
        k
        for k in range(K - 1)
        if timeline.plan_status[2 * k] != 0 and timeline.plan_at_decision_kwh[k].sum() > 0
    ]
    followed = [
        k
        for k in range(K - 1)
        if timeline.plan_status[2 * k] == 0 and timeline.plan_at_decision_kwh[k].sum() > 0
    ]
    assert not_followed and followed
    for k in not_followed[:3]:
        assert np.isnan(_frame_y(figure, k, in_force)).all()
        y = _frame_y(figure, k, on_record)
        np.testing.assert_allclose(
            np.nan_to_num(y), timeline.plan_at_decision_kwh[k], rtol=1e-6, atol=1e-6
        )
    for k in followed[:3]:
        assert np.isnan(_frame_y(figure, k, on_record)).all()
        assert np.nansum(_frame_y(figure, k, in_force)) > 0


# --- Payload -----------------------------------------------------------------------


@pytest.mark.parametrize("lens", ["fleet", "customer"])
def test_payload_within_the_bound(result, lens: str) -> None:
    figure = _figure(_fleet(result) if lens == "fleet" else _customer(result))

    size = len(figure.to_json())
    print(f"Replay {lens} lens figure JSON: {size / 1e6:.2f} MB")
    assert size <= PAYLOAD_BOUND


# --- Controls, cache and copy --------------------------------------------------------


def test_week_control_offers_the_sampled_worlds_median_first(result) -> None:
    st = _fleet(result)

    ((_, kwargs),) = [
        (a, k) for a, k in calls_for(st, "selectbox") if k.get("key") == view.WEEK_KEY
    ]
    assert kwargs["options"] == list(result.replay_week.world_ids)
    first = kwargs["options"][kwargs["index"]]
    assert first == result.representative_world_id
    assert kwargs["format_func"](first) == f"Week {first} (median)"


def test_ev_control_matches_one_ev_labels_and_default(result) -> None:
    st = _customer(result)

    ((_, kwargs),) = [(a, k) for a, k in calls_for(st, "selectbox") if k.get("key") == view.EV_KEY]
    row = result.units.iloc[2]
    assert kwargs["format_func"](row["unit_id"]) == f"{row['unit_id']} · {row['cohort_label']}"
    default = result.units.loc[result.units["cohort_id"].eq("average_uk"), "unit_id"].iat[0]
    assert kwargs["options"][kwargs["index"]] == default


def test_jump_moves_play_to_start_at_the_jumped_to_instant(result) -> None:
    figure = _figure(_fleet(result, **{view.JUMP_KEY: 100}))

    play = figure.layout.updatemenus[0].buttons[0]
    assert list(play.args[0]) == [str(k) for k in range(100, K)]
    assert _figure(_fleet(result)).layout.updatemenus[0].buttons[0].args[0] is None


def test_nothing_runs_the_model(result, monkeypatch) -> None:
    from axle_studio.model import forecast

    def refuse(*args, **kwargs):
        raise AssertionError("a view started a run")

    monkeypatch.setattr(forecast, "run_forecast", refuse)
    monkeypatch.setattr(forecast, "run_forecast_from_assumptions", refuse)

    _fleet(result, **{view.JUMP_KEY: 20})
    _customer(result, **{view.JUMP_KEY: 20})


def test_frames_are_cached_on_the_result_and_reused_by_jump(result, monkeypatch) -> None:
    cached = replace(result, replay_state=SimpleNamespace(replay_cache={}))
    builds = []
    real = view._fleet_parts
    monkeypatch.setattr(view, "_fleet_parts", lambda *a: builds.append(a) or real(*a))

    _fleet(cached)
    figure = _figure(_fleet(cached, **{view.JUMP_KEY: 12}))

    assert len(builds) == 1
    world = result.replay_week.world_ids[0]
    assert ("replay_figure", world, None) in cached.replay_state.replay_cache
    assert figure.layout.sliders[0].active == 12


@pytest.mark.parametrize("lens", ["fleet", "customer"])
def test_copy_is_short_and_every_pound_is_illustrative(result, lens: str) -> None:
    st = _fleet(result) if lens == "fleet" else _customer(result)
    figure = _figure(st)

    # The "Data and definition" text (the caption after the block's table)
    # is reference text, exempt from the length rule as in test_copy_rules.
    captions = [
        args[0]
        for (name, args, _), (previous, _, _) in zip(st.calls[1:], st.calls, strict=False)
        if name == "caption" and previous != "dataframe"
    ]
    assert captions and all(len(text) <= 140 for text in captions)
    for frame in (figure.frames[0], figure.frames[80], figure.frames[K - 1]):
        for line in frame.data[0].text[0].split("<br>"):
            if "£" in line:
                assert "illustrative" in line, line
    text = " ".join(captions) + " ".join(str(trace.name) for trace in figure.data)
    assert "Axle cash" not in text
    assert "Baseline" not in text
    assert "selected" not in text.lower().replace("selected worlds", "")


def test_shock_markers_appear_only_once_known(result) -> None:
    replay = result.replay_week
    world = replay.world_ids[0]
    shocks = replay.shocks.loc[replay.shocks["world_id"].eq(world)]
    assert not shocks.empty  # the SYNTHETIC fixture draws shocks in its sampled worlds

    track = view._known_markers(replay, 0, 99.0)

    known = list(shocks["known_from_utc"]) + list(replay.events["known_from_utc"])
    for k, moment in enumerate(replay.decisions["decision_utc"]):
        expected = [99.0 if when <= moment else None for when in known]
        assert track.frames[k]["y"] == expected


def test_the_figure_cache_keeps_only_the_latest_few(result) -> None:
    cached = replace(result, replay_state=SimpleNamespace(replay_cache={("replay", "ev", 0): 1}))
    worlds = list(result.replay_week.world_ids)

    for world in worlds * 2:
        _fleet(cached, **{view.WEEK_KEY: world})

    cache = cached.replay_state.replay_cache
    figures = [key for key in cache if key[0] == "replay_figure"]
    assert len(figures) == min(len(worlds), view._FIGURE_CACHE_LIMIT)
    assert ("replay", "ev", 0) in cache  # the one-EV replays sharing the dict are untouched


@pytest.mark.parametrize("lens", ["fleet", "customer"])
def test_no_frame_sends_an_empty_trace(result, lens: str) -> None:
    # Regression: with redraw off, Plotly skips a trace with no points and
    # leaves the previous frame's line on screen (seen in the browser at the
    # end of the week), so an all-hidden frame sends one NaN instead.
    figure = _figure(_fleet(result) if lens == "fleet" else _customer(result))

    for frame in figure.frames:
        for data in frame.data:
            if data.y is not None:
                assert len(data.y) >= 1


def test_ev_markers_appear_only_once_their_slot_has_ended(result) -> None:
    # Review of 9f2a022, B1: plug-in, top-up and left-early markers are
    # stamped with a slot start and drawn at that slot's closing SoC, so they
    # show only once the slot has ended (slot index < 2k); a departure is an
    # instant and shows from tau >= departure.
    unit_id = result.units["unit_id"].iat[0]
    world = result.replay_week.world_ids[0]
    ev = replay_one_ev(result, unit_id, world)
    decisions = result.replay_week.decisions
    selected = ev.intervals.loc[ev.intervals["path_id"].eq("selected")].sort_values("slot_index")

    track = view._ev_markers(ev, selected, decisions)

    half_hour = pd.Timedelta(minutes=30)
    known = [
        when + (half_hour if symbol != "triangle-down" else pd.Timedelta(0))
        for when, symbol in zip(
            pd.DatetimeIndex(track.base["x"]), track.base["marker"]["symbol"], strict=True
        )
    ]
    slot_stamped = [s != "triangle-down" for s in track.base["marker"]["symbol"]]
    assert any(slot_stamped)
    for k, moment in enumerate(decisions["decision_utc"]):
        shown = [y is not None for y in track.frames[k]["y"]]
        assert shown == [when <= moment for when in known]
        # A slot-stamped marker never shows while its slot (starting at tau_k) is open.
        for x, is_slot, visible in zip(track.base["x"], slot_stamped, shown, strict=True):
            if is_slot and pd.Timestamp(x) == moment:
                assert not visible


# --- The optional timed path (decision 0007) --------------------------------
#
# The shared SYNTHETIC fixture stays two-policy by design, so the timed path
# is exercised here with a small real run instead (as the task brief directs),
# ``timed_start_local_hour=0.0``. The "absent" state (every test above) is the
# existing fixture's own two-policy frames, unchanged by this file's edits.

_TIMED_START = date(2026, 1, 12)  # a Monday, no clock change (test_timed_policy.py's own date)
_TIMED_VALUES = {"vehicle_count": 30, "evaluation_world_count": 3, "timed_start_local_hour": 0.0}

_ACCEPTED_REPLAY_RULES = {"line-without-band", "teal-reserved"}
"""Rules ``test_dashboard_lint.ACCEPTED`` already accepts for both Replay charts
(one simulated week's own realised lines, not an across-weeks statistic); a new
finding of any other rule here would be this change's own regression."""


def _no_new_findings(figure) -> None:
    findings = [f for f in lint_figure(figure) if f.rule not in _ACCEPTED_REPLAY_RULES]
    assert findings == [], findings


@pytest.fixture(scope="module")
def timed_result():
    """A small real run with the optional timed path on, for the three-policy checks."""

    return run_forecast_from_assumptions(_TIMED_START, values=_TIMED_VALUES)


def _real_customer(result, **choices):
    """Render Replay ▸ Customer with the real one-EV model functions (not the fixture's)."""

    st = RecordingStreamlit(choices=choices)
    view.render_replay_customer(
        st,
        result,
        replay_one_ev=model_replay_one_ev,
        replay_one_ev_timeline=model_replay_one_ev_timeline,
    )
    return st


def test_fleet_import_line_gains_timed_tariff_without_trading_frames(timed_result) -> None:
    # Replay contract v1 §1.6: the fleet lens reads the timed path from
    # ``fleet_world_intervals`` only when there are no trading frames; this
    # isolates that branch the same way
    # ``test_fleet_without_the_ledger_drops_the_trading_figure`` does for the
    # existing two-policy case. A real run's ``replay_state`` carries a
    # genuine figure cache (unlike the SYNTHETIC fixture's), so this gets its
    # own empty cache rather than sharing ``timed_result``'s: otherwise this
    # test and the one beside it, same world, would read back each other's
    # cached figure instead of each building its own.
    trimmed = replace(
        timed_result,
        replay_state=replace(timed_result.replay_state, replay_cache={}),
        deviation_world_slot=None,
        position_updates=None,
        trading_week_world=None,
        replay_week=replace(
            timed_result.replay_week, position_kwh=None, trading_cash_to_date_gbp=None
        ),
    )
    figure = _figure(_fleet(trimmed))
    names = {trace.name for trace in figure.data}
    assert {"Unmanaged", "Timed tariff", "Smart"} <= names
    _no_new_findings(figure)


def test_fleet_ledger_unmanaged_kw_matches_the_fleet_frames_normal_path(timed_result) -> None:
    # The premise the Timed tariff import line below depends on: the
    # ledger's own ``unmanaged_kw`` and ``fleet_world_intervals``' "normal"
    # ``home_import_kw`` are the same quantity on the same kW basis, for
    # every sampled world, so the three import lines are directly comparable.
    deviation = timed_result.deviation_world_slot
    intervals = timed_result.fleet_world_intervals
    for world in timed_result.replay_week.world_ids:
        dev_rows = deviation.loc[deviation["world_id"].eq(world)].sort_values("slot_index")
        normal_rows = intervals.loc[
            intervals["path_id"].eq("normal") & intervals["world_id"].eq(world)
        ].sort_values("slot_index")
        np.testing.assert_allclose(
            dev_rows["unmanaged_kw"].to_numpy(float),
            normal_rows["home_import_kw"].to_numpy(float),
            rtol=0,
            atol=1e-9,
        )


def test_fleet_import_line_gains_timed_tariff_with_trading_frames(timed_result) -> None:
    # Follow-up: the money side of the fleet lens (deviation, baseline, book,
    # position) never redefines for the timed path, but the plain import
    # line itself now draws from ``fleet_world_intervals`` even when trading
    # frames are present, so the policy is visible under the live app's
    # actual wiring (every real action run sets ``deviation_world_slot``).
    world = timed_result.replay_week.world_ids[0]
    intervals = timed_result.fleet_world_intervals
    expected_timed = (
        intervals.loc[intervals["path_id"].eq("timed") & intervals["world_id"].eq(world)]
        .sort_values("slot_index")["home_import_kw"]
        .to_numpy(float)
    )
    figure = _figure(_fleet(timed_result))
    names = {trace.name for trace in figure.data}
    assert {"Unmanaged", "Timed tariff", "Smart"} <= names
    # Display order: Timed tariff sits between Unmanaged and Smart.
    action_names = [t.name for t in figure.data if t.name in ("Unmanaged", "Timed tariff", "Smart")]
    assert action_names == ["Unmanaged", "Timed tariff", "Smart"]
    drawn = _frame_y(figure, K - 1, _trace(figure, "Timed tariff"))
    # The Plotly figure stores trace arrays as float32 (a JSON-payload
    # saving, not a view calculation), so this allows that rounding rather
    # than the exact match the two source frames get above.
    np.testing.assert_allclose(drawn, expected_timed, rtol=1e-6, atol=1e-6)
    # A plain import series only: every trading trace is unaffected.
    assert {"Sold turn-down", "Sold at now", "Bought back at now"} <= names
    _no_new_findings(figure)


def test_fleet_timed_import_line_hides_the_future_with_trading_frames(timed_result) -> None:
    figure = _figure(_fleet(timed_result))
    index = _trace(figure, "Timed tariff")
    for k in (0, 40, 100, K - 1):
        y = _frame_y(figure, k, index)
        assert np.isnan(y[2 * k :]).all()
        if k > 0:
            assert np.isfinite(y[: 2 * k]).any()


def test_fleet_import_line_stays_two_policy_without_the_timed_path(result) -> None:
    # Absent: the fleet lens (trading frames present, the ``result`` fixture's
    # own default) draws exactly the two lines it always has.
    figure = _figure(_fleet(result))
    names = {trace.name for trace in figure.data}
    assert names & {"Unmanaged", "Smart", "Timed tariff"} == {"Unmanaged", "Smart"}


def test_fleet_kpis_add_a_timed_saving_tile_beside_smart_when_the_model_provides_it(
    timed_result,
) -> None:
    st = _fleet(timed_result)
    labels = [tile[0] for tile in kpi_calls(st)]
    assert "Drivers’ saving this week" in labels
    assert "Timed saving this week" in labels
    assert labels.index("Timed saving this week") == labels.index("Drivers’ saving this week") + 1


def test_fleet_kpis_drop_the_timed_tile_without_timed_cost_effect(timed_result) -> None:
    without = replace(timed_result, timed_cost_effect=None)
    st = _fleet(without)
    labels = [tile[0] for tile in kpi_calls(st)]
    assert "Timed saving this week" not in labels


def test_customer_soc_and_charging_tracks_gain_the_timed_path(timed_result) -> None:
    figure = _figure(_real_customer(timed_result))
    names = {trace.name for trace in figure.data}
    assert {"Unmanaged", "Timed tariff", "Smart"} <= names
    _no_new_findings(figure)


def test_customer_kpis_and_cost_table_show_all_three_policies_then_saved(timed_result) -> None:
    st = _real_customer(timed_result)
    labels = [tile[0] for tile in kpi_calls(st)]
    assert labels[-4:] == [
        "Unmanaged cost this week",
        "Timed tariff cost this week",
        "Smart cost this week",
        "Saved this week",
    ]
    parts = _customer_parts_table(timed_result)
    assert list(parts.columns[-4:]) == [
        "Unmanaged cost so far (£, illustrative)",
        "Timed tariff cost so far (£, illustrative)",
        "Smart cost so far (£, illustrative)",
        "Saved so far (£, illustrative)",
    ]


def _customer_parts_table(result) -> pd.DataFrame:
    world = result.replay_week.world_ids[0]
    unit_id = result.units["unit_id"].iat[0]
    parts = view._customer_parts(
        result, world, unit_id, model_replay_one_ev, model_replay_one_ev_timeline
    )
    return parts.table
