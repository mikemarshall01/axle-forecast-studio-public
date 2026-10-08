"""Compare view tests: run history plumbing, matched/unmatched, no-action total.

Uses the SYNTHETIC FIXTURE's ``make_result`` and ``compare_runs``
(``fixtures.result_fixture``), injected exactly as the task brief requires
("no model calls except compare_runs via an injectable function"). A fake
run history stands in for ``run_controller.run_history`` (monkeypatched),
since building a full ``run_controller`` session just to get two
``RunRecord``s is unrelated machinery this view does not own.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from fixtures.result_fixture import compare_runs, make_result
from kpi_calls import kpi_calls

from axle_studio.ui.style import MUTED_INK, SERIES_COLOURS
from axle_studio.ui.views import compare as compare_module
from axle_studio.ui.views.compare import (
    _default_unpairable_reason,
    _evening_peak_kw,
    _format_money,
    _format_signed,
    _no_negative_zero,
    _paired_quantiles,
    _public_top_up_kwh,
    _saving_change_sentence,
    _weekly_metric_by_world,
    render_compare,
)


class _SlimRecord:
    """Stands in for a decision 0004 item 35 "slim" run record.

    A real slim record carries ``fleet_world_intervals``, ``plug_in_world_kpis``,
    ``cost_effect``, ``weekly_bands``, the settings snapshot, seed, model and
    world count for *every* kept run -- never a full stored result except (on
    the real ``RunRecord``) for the latest run, which is irrelevant to
    Compare. This proxy is backed by a full fixture result so tests can reuse
    ``make_result`` for those slim fields, while ``.result`` deliberately
    raises: if ``render_compare`` ever reached for it again, these tests
    would fail loudly instead of silently reintroducing the old gap.

    ``number`` is parsed off the label ("Run 2 · base" -> 2): every label in
    this file already follows that "Run N ..." shape (``run_label``'s real
    format), and the A/B selectors (compare.py's ``_resolve_pair``) key on
    ``.number``, not on object identity, so it must be present and correct
    here too.
    """

    def __init__(self, label, backing):
        self.label = label
        self.number = int(re.match(r"Run (\d+)", label).group(1))
        self._backing = backing

    def __getattr__(self, name):
        if name == "result":
            raise AttributeError("slim run records do not carry a full result")
        return getattr(self._backing, name)


class RecordingStreamlit:
    def __init__(
        self,
        calls=None,
        *,
        session_state=None,
        segmented_control_returns=None,
        selectbox_returns=None,
    ):
        self.calls = calls if calls is not None else []
        self.session_state = session_state if session_state is not None else {}
        self._segmented_control_returns = segmented_control_returns or {}
        self._selectbox_returns = selectbox_returns or {}

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "columns":
                spec = args[0] if args else kwargs.get("spec", 1)
                count = spec if isinstance(spec, int) else len(spec)
                return [
                    RecordingStreamlit(
                        self.calls,
                        # Real st.columns() panes share one session_state
                        # with their parent; a fresh dict per pane would
                        # silently drop the A/B selectors' pre-seeded
                        # default (_resolve_pair sets it on the outer `st`
                        # before rendering the selectbox inside a column).
                        session_state=self.session_state,
                        segmented_control_returns=self._segmented_control_returns,
                        selectbox_returns=self._selectbox_returns,
                    )
                    for _ in range(count)
                ]
            if name == "segmented_control":
                key = kwargs.get("key")
                if key in self._segmented_control_returns:
                    return self._segmented_control_returns[key]
                return kwargs.get("default")
            if name == "selectbox":
                key = kwargs.get("key")
                if key in self._selectbox_returns:
                    return self._selectbox_returns[key]
                # Mirrors real Streamlit: with no explicit override, the
                # widget's current value is whatever _resolve_pair already
                # wrote to session_state (the computed default, or a
                # previous rerun's choice) under this key.
                return self.session_state.get(key)
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


def _record(label: str, result) -> _SlimRecord:
    return _SlimRecord(label, result)


def test_fewer_than_two_runs_shows_the_empty_state(monkeypatch) -> None:
    monkeypatch.setattr(compare_module, "run_history", lambda state: [])
    st = RecordingStreamlit()

    render_compare(st, None)

    assert calls_for(st, "markdown")
    text = calls_for(st, "markdown")[0][0][0]
    assert "kept here automatically" in text
    assert text.startswith("Run once")
    assert calls_for(st, "button")


def test_one_kept_run_says_one_more_is_needed_not_run_once(monkeypatch) -> None:
    # Regression: a fresh session already holds one run (run_controller's
    # precomputed "Run 0" default), so this state must not tell a reader who
    # has already run once to "Run once".
    result = make_result(model="action", evs=6, worlds=5)
    monkeypatch.setattr(compare_module, "run_history", lambda state: [_record("Run 0", result)])
    st = RecordingStreamlit()

    render_compare(st, None)

    text = calls_for(st, "markdown")[0][0][0]
    assert "Run once" not in text
    assert "one more run" in text.lower()
    assert "kept here automatically" in text
    assert calls_for(st, "button")


def test_matched_futures_runs_show_kpis_and_metric_chart(monkeypatch) -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=7.0)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · base", a), _record("Run 3 · home charger 11 kW", b)],
    )
    st = RecordingStreamlit()

    render_compare(st, None, compare_runs=compare_runs)

    comparison = compare_runs(a, b)
    assert comparison.matched_futures
    matched_caption = [
        args[0] for args, _ in calls_for(st, "caption") if "Matched futures" in args[0]
    ]
    assert matched_caption == [f"Matched futures: yes (seed {b.seed}, {b.world_count} weeks)"]

    changed_caption = [
        args[0] for args, _ in calls_for(st, "caption") if args[0].startswith("Changed")
    ]
    assert changed_caption
    assert "home_charger_kw" in changed_caption[0] or "Home charger power" in changed_caption[0]

    kpi_labels = [tile[0] for tile in kpi_calls(st)]
    assert kpi_labels == [
        "Weekly home import",
        "Largest evening change",
        "Public top-ups",
        "Illustrative cost change",
    ]
    total = tile_by_label(st, "Illustrative cost change")
    assert total[1] != "Unavailable"
    # No st.metric (and so no delta arrow) anywhere (goal review action 7):
    # an arrow reads as up-good/down-bad on top of a value that already is
    # the change. The P10-P90 range is each tile's visible context line.
    assert not calls_for(st, "metric")
    assert all(tile[3].startswith("P10–P90:") for tile in kpi_calls(st))
    # One legend entry for the band and its median (polish plan G3).
    legend = [t.name for t in calls_for(st, "plotly_chart")[0][0][0].data if t.showlegend]
    assert legend == ["B − A"]

    (figure,), _ = calls_for(st, "plotly_chart")[0]
    expected_frame = comparison.difference_bands.loc[
        comparison.difference_bands["metric"].eq("home_import_kw")
    ].sort_values("slot_index")
    # P50 is rounded to the hovertemplate's 2 dp before plotting (chart audit
    # 2 follow-up, O1: a hover trace must never show "-0").
    # Hover precision = axis precision (polish plan G6): whole kW, 1 dp for points.
    assert np.allclose(list(figure.data[-1].y), np.round(expected_frame["p50"], 0))


def test_unmatched_seeds_are_labelled(monkeypatch) -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=7)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · base", a), _record("Run 3 · seed 7", b)],
    )
    st = RecordingStreamlit()

    render_compare(st, None, compare_runs=compare_runs)

    reasons = [args[0] for args, _ in calls_for(st, "caption")]
    assert any("Not matched" in reason for reason in reasons)
    assert not any(reason.startswith("Matched futures: yes") for reason in reasons)


def test_no_action_run_makes_the_total_unavailable(monkeypatch) -> None:
    a = make_result(model="no_action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=42)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · no action", a), _record("Run 3 · action", b)],
    )
    st = RecordingStreamlit()

    render_compare(st, None, compare_runs=compare_runs)

    total = tile_by_label(st, "Illustrative cost change")
    assert total[1] == "Unavailable"
    assert "no Axle action" in total[4]


def test_compare_never_reads_a_full_result_off_the_run_record(monkeypatch) -> None:
    """Decision 0004 item 35: Compare works from every kept run's slim record,
    not only the latest run's full stored result."""

    a = make_result(model="action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=42)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · base", a), _record("Run 3 · action", b)],
    )
    st = RecordingStreamlit()

    # _SlimRecord raises AttributeError on `.result`, so this would fail if
    # render_compare still reached into a full stored result.
    render_compare(st, None, compare_runs=compare_runs)

    assert not calls_for(st, "info")
    assert calls_for(st, "plotly_chart")


def test_metric_control_switches_which_band_is_plotted(monkeypatch) -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=7.0)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · base", a), _record("Run 3 · charger", b)],
    )
    comparison = compare_runs(a, b)

    soc_st = RecordingStreamlit(segmented_control_returns={"compare-metric": "Battery SoC"})
    render_compare(soc_st, None, compare_runs=compare_runs)
    (soc_figure,), _ = calls_for(soc_st, "plotly_chart")[0]
    expected_soc = comparison.difference_bands.loc[
        comparison.difference_bands["metric"].eq("battery_soc_percent")
    ].sort_values("slot_index")
    assert np.allclose(list(soc_figure.data[-1].y), np.round(expected_soc["p50"], 1))

    home_st = RecordingStreamlit(segmented_control_returns={"compare-metric": "Home import"})
    render_compare(home_st, None, compare_runs=compare_runs)
    (home_figure,), _ = calls_for(home_st, "plotly_chart")[0]
    assert list(home_figure.data[-1].y) != list(soc_figure.data[-1].y)


def test_chart_height_and_no_zero_line_suppression(monkeypatch) -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=7.0)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · base", a), _record("Run 3 · charger", b)],
    )
    st = RecordingStreamlit()

    render_compare(st, None, compare_runs=compare_runs)

    (figure,), kwargs = calls_for(st, "plotly_chart")[0]
    assert figure.layout.height == compare_module.CHART_HEIGHTS["time_series"]
    assert figure.layout.yaxis.zeroline is True
    assert kwargs["config"] == {"displayModeBar": False}
    # Chart audit 2, O6: the zero line must be visible against the grid.
    assert figure.layout.yaxis.zerolinecolor == MUTED_INK
    assert figure.layout.yaxis.zerolinecolor != SERIES_COLOURS["grid"]


def test_plugged_in_difference_chart_is_percentage_points_not_fraction(monkeypatch) -> None:
    # Chart audit 2, B7: "Plugged in" used to show the raw 0-1 fraction; it
    # must read in percentage points, ×100, the same convention Response
    # uses for this metric, consistently on the axis, hover and title.
    a = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=7.0)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · base", a), _record("Run 3 · charger", b)],
    )
    st = RecordingStreamlit(segmented_control_returns={"compare-metric": "Plugged in"})

    render_compare(st, None, compare_runs=compare_runs)

    comparison = compare_runs(a, b)
    expected = comparison.difference_bands.loc[
        comparison.difference_bands["metric"].eq("connected_share")
    ].sort_values("slot_index")

    (figure,), _ = calls_for(st, "plotly_chart")[0]
    assert np.allclose(list(figure.data[-1].y), np.round(100 * expected["p50"], 1))
    assert figure.layout.yaxis.title.text == "Plugged in difference (percentage points)"
    median_trace = figure.data[-1]
    assert "percentage points" in median_trace.hovertemplate
    assert "fraction" not in median_trace.hovertemplate
    titles = [args[0] for args, _ in calls_for(st, "markdown")]
    chart_title = next(
        title for title in titles if title.startswith("**B − A, paired per week: Plugged in")
    )
    assert "percentage points" in chart_title
    assert "fraction" not in chart_title


def test_format_money_and_signed_never_print_negative_zero() -> None:
    # Chart audit 2, O1: a value that only rounds to zero at the display
    # precision must not print "−£0" or "-0 pts". Fleet-scale money is
    # whole pounds with a U+2212 minus (polish plan G6).
    assert _format_money(-0.001) == "£0"
    assert _format_money(0.4) == "£0"
    assert _format_money(-1250.51) == "−£1,251"
    assert _format_money(380.0) == "+£380"
    assert _format_signed(-0.3, suffix=" pts") == "+0 pts"
    assert _format_signed(0.3, suffix=" pts") == "+0 pts"
    assert _format_signed(-1.6, suffix=" pts") == "−2 pts"


def test_saving_change_sentence_reads_the_sign_correctly() -> None:
    # Hand example (goal review N2): the model's own cost column is
    # selected-minus-normal, so a *positive* B-minus-A figure means B's cost
    # rose relative to A -- B saved £380 LESS than A, the opposite of what a
    # bare "+£380" next to a tile once called "Weekly saving change" would
    # suggest. The sentence must say "less", not "more", for a positive
    # delta, and vice versa for a negative one.
    assert _saving_change_sentence(380.0) == "B saves £380 less than A."
    assert _saving_change_sentence(-380.0) == "B saves £380 more than A."
    assert _saving_change_sentence(0.0) == "B and A save about the same amount."
    assert _saving_change_sentence(None) == "B's saving relative to A is unavailable."


def test_illustrative_cost_change_tile_states_the_saving_direction(monkeypatch) -> None:
    # Integration check for goal review N2: the caption under the renamed
    # "Illustrative cost change" tile must state the saving
    # direction matching the sign of the underlying B-minus-A cost figure.
    a = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=7.0)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · base", a), _record("Run 3 · home charger 11 kW", b)],
    )
    st = RecordingStreamlit()

    render_compare(st, None, compare_runs=compare_runs)

    comparison = compare_runs(a, b)
    p50 = comparison.kpis.set_index("metric").loc["illustrative_total_gbp", "p50"]
    expected_sentence = _saving_change_sentence(p50)
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert any(caption.startswith(expected_sentence) for caption in captions)


def test_no_negative_zero_clears_the_sign_a_hovertemplate_would_show() -> None:
    # Chart audit 2 follow-up, O1 (BLOCKING): a raw float in (-0.005, 0)
    # prints as "-0.00" in a Plotly d3-format hovertemplate (%{y:,.2f}), even
    # though the same value displays as plain "0.00" everywhere else. A B-A
    # median trace carrying a value like -0.00224 must come out as a clean
    # +0.0, not a still-negative -0.0, before it ever reaches Plotly.
    rounded = _no_negative_zero(pd.Series([-0.00224, -1.6, 1.6, 0.0]), decimals=2)
    assert list(rounded) == [0.0, -1.6, 1.6, 0.0]
    assert math.copysign(1.0, rounded.iloc[0]) == 1.0, "must be +0.0, not -0.0"
    # The exact failure mode: formatting a lingering -0.0 the way d3-format
    # would (":,.2f") prints the bare minus sign.
    assert f"{rounded.iloc[0]:,.2f}" == "0.00"


def test_default_compare_runs_is_not_called_when_injected(monkeypatch) -> None:
    """No model calls except compare_runs, and only via the injected callable."""

    a = make_result(model="action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=42)
    monkeypatch.setattr(
        compare_module, "run_history", lambda state: [_record("Run 2", a), _record("Run 3", b)]
    )

    def boom(a, b):
        raise AssertionError("the default compare_runs must not be used when one is injected")

    monkeypatch.setattr(compare_module, "_default_compare_runs", boom)
    st = RecordingStreamlit()

    render_compare(st, None, compare_runs=compare_runs)

    assert calls_for(st, "plotly_chart")


def test_default_compare_runs_defers_to_the_model_function(monkeypatch) -> None:
    # The view imports model.summaries lazily, so the real compare_runs is used at call time.
    import axle_studio.model.summaries as summaries

    calls = []
    monkeypatch.setattr(summaries, "compare_runs", lambda a, b: calls.append((a, b)) or "ok")
    assert compare_module._default_compare_runs("a", "b") == "ok"
    assert calls == [("a", "b")]


# --- A/B selectors (milestone verifier finding: hard-coded records[-2:]) ----


def test_ab_selectors_render_with_every_kept_run_labelled(monkeypatch) -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · base", a), _record("Run 3 · home charger 11 kW", b)],
    )
    st = RecordingStreamlit()

    render_compare(st, None, compare_runs=compare_runs)

    selectbox_calls = calls_for(st, "selectbox")
    a_args, a_kwargs = next(call for call in selectbox_calls if call[1]["key"] == "compare-run-a")
    b_args, b_kwargs = next(call for call in selectbox_calls if call[1]["key"] == "compare-run-b")
    assert (a_args[0], b_args[0]) == ("A", "B")
    assert a_kwargs["options"] == [2, 3] and b_kwargs["options"] == [2, 3]
    # Labels are collapsed in the header row (polish plan G8), so each
    # option names its own side.
    assert [a_kwargs["format_func"](n) for n in a_kwargs["options"]] == [
        "A: Run 2 · base",
        "A: Run 3 · home charger 11 kW",
    ]
    assert b_kwargs["format_func"](3) == "B: Run 3 · home charger 11 kW"
    # Default is previous vs latest (design 4.4), pre-seeded before the
    # widget call so a fresh session never has to fall back to `index=`.
    assert (st.session_state["compare-run-a"], st.session_state["compare-run-b"]) == (2, 3)


def test_stale_selection_resets_to_the_default_pair(monkeypatch) -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 3 · base", a), _record("Run 4 · charger", b)],
    )
    # Run 2 has since dropped out of the kept history (HISTORY_SIZE = 3); a
    # session left pointing at it must reset, not crash the page (a stale
    # selectbox value that is not in `options` raises in real Streamlit).
    st = RecordingStreamlit(session_state={"compare-run-a": 2, "compare-run-b": 4})

    render_compare(st, None, compare_runs=compare_runs)

    assert (st.session_state["compare-run-a"], st.session_state["compare-run-b"]) == (3, 4)
    assert calls_for(st, "plotly_chart")


def test_new_run_moves_the_default_forward_even_when_the_old_pair_still_exists(
    monkeypatch,
) -> None:
    # Final critique B-18: Run 1 and Run 2 both survive Run 3 landing
    # (HISTORY_SIZE = 3 keeps all three), so the old "not in by_number" reset
    # never fires; a session left on Run 1 vs Run 2 must still move forward
    # to Run 2 vs Run 3, the pair the "run again, open Compare" demo step
    # needs, because a new run has landed since Compare last resolved a pair.
    a = make_result(model="action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    c = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=22.0)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [
            _record("Run 1 · base", a),
            _record("Run 2 · charger", b),
            _record("Run 3 · charger", c),
        ],
    )
    # Compare was already open on Run 1 vs Run 2 before Run 3 landed.
    st = RecordingStreamlit(
        session_state={"compare-run-a": 1, "compare-run-b": 2, "compare-last-seen-run": 2}
    )

    render_compare(st, None, compare_runs=compare_runs)

    assert (st.session_state["compare-run-a"], st.session_state["compare-run-b"]) == (2, 3)
    assert st.session_state["compare-last-seen-run"] == 3


def test_explicit_choice_survives_a_rerun_with_no_new_run(monkeypatch) -> None:
    # Once the last-seen marker matches the latest run, a later rerun (a
    # lens click, editing a draft) must not keep re-snapping to the default:
    # a pair the presenter chose "in this run" sticks until the next new run.
    a = make_result(model="action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    c = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=22.0)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [
            _record("Run 1 · base", a),
            _record("Run 2 · charger", b),
            _record("Run 3 · charger", c),
        ],
    )
    # The presenter picked Run 1 vs Run 3 after Run 3 landed (marker already 3).
    st = RecordingStreamlit(
        session_state={"compare-run-a": 1, "compare-run-b": 3, "compare-last-seen-run": 3}
    )

    render_compare(st, None, compare_runs=compare_runs)

    assert (st.session_state["compare-run-a"], st.session_state["compare-run-b"]) == (1, 3)


def test_a_and_b_the_same_run_shows_a_short_message(monkeypatch) -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · base", a), _record("Run 3 · charger", b)],
    )
    st = RecordingStreamlit(selectbox_returns={"compare-run-a": 3, "compare-run-b": 3})

    render_compare(st, None, compare_runs=compare_runs)

    assert [args[0] for args, _ in calls_for(st, "info")] == [
        "Choose two different runs to compare."
    ]
    assert not calls_for(st, "plotly_chart")


def test_default_pair_prefers_previous_vs_latest_when_pairable() -> None:
    a = make_result(worlds=5)
    b = make_result(worlds=5)
    records = [_record("Run 2", a), _record("Run 3", b)]

    chosen_a, chosen_b, note = compare_module._default_pair(records, _default_unpairable_reason)

    assert (chosen_a, chosen_b, note) == (records[0], records[1], None)


def test_default_pair_falls_back_to_a_pairable_pair_and_names_it() -> None:
    # Milestone repro: Run 2 and Run 3 share a world count (pairable); Run 4
    # does not (unpairable with Run 3, the plain previous-vs-latest default).
    a = make_result(worlds=5)
    b = make_result(worlds=5)
    c = make_result(worlds=6)
    records = [_record("Run 2", a), _record("Run 3", b), _record("Run 4", c)]

    chosen_a, chosen_b, note = compare_module._default_pair(records, _default_unpairable_reason)

    assert (chosen_a, chosen_b) == (records[0], records[1])
    assert note == (
        "Cannot compare: Run 3 has 5 simulated weeks, Run 4 has 6. Keep weeks and start date "
        "the same to compare. Showing Run 2 vs Run 3 instead."
    )


def test_default_pair_keeps_the_plain_default_when_nothing_pairs() -> None:
    a = make_result(worlds=4)
    b = make_result(worlds=5)
    records = [_record("Run 2", a), _record("Run 3", b)]

    chosen_a, chosen_b, note = compare_module._default_pair(records, _default_unpairable_reason)

    assert (chosen_a, chosen_b, note) == (records[0], records[1], None)


def test_render_compare_falls_back_to_a_pairable_default_and_says_so(monkeypatch) -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    c = make_result(model="action", evs=6, worlds=6, seed=42)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2", a), _record("Run 3", b), _record("Run 4", c)],
    )
    st = RecordingStreamlit()

    render_compare(st, None, compare_runs=compare_runs)

    notes = [args[0] for args, _ in calls_for(st, "caption")]
    assert any("Showing Run 2 vs Run 3 instead." in note for note in notes)
    assert (st.session_state["compare-run-a"], st.session_state["compare-run-b"]) == (2, 3)
    assert calls_for(st, "plotly_chart")


def test_explicit_unpairable_choice_keeps_the_plain_message(monkeypatch) -> None:
    # The default falls back to Run 2 vs Run 3; explicitly picking the
    # unpairable Run 3 vs Run 4 still shows the plain message, not a
    # fallback note (that note only explains an unrequested default move).
    a = make_result(model="action", evs=6, worlds=4, seed=42)
    b = make_result(model="action", evs=6, worlds=4, seed=42)
    c = make_result(model="action", evs=6, worlds=5, seed=42)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2", a), _record("Run 3", b), _record("Run 4", c)],
    )
    st = RecordingStreamlit(selectbox_returns={"compare-run-a": 3, "compare-run-b": 4})

    render_compare(st, None, compare_runs=compare_runs)

    assert [args[0] for args, _ in calls_for(st, "info")] == [
        "Cannot compare: Run 3 has 4 simulated weeks, Run 4 has 5. Keep weeks and start date "
        "the same to compare."
    ]
    assert not any("instead" in args[0] for args, _ in calls_for(st, "caption"))
    assert not calls_for(st, "plotly_chart")


# --- New action-8 tiles: evening peak kW, public top-ups -------------------


def test_evening_peak_kw_picks_the_largest_evening_shift(monkeypatch) -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=7.0)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    comparison = compare_runs(a, b)

    result = _evening_peak_kw(comparison.difference_bands)

    assert result is not None
    p10, p50, p90, when = result
    assert p10 <= p50 <= p90
    hour = int(when.split(":")[0])
    assert 16 <= hour < 22
    # The picked row really is the largest-magnitude evening row, not just any one.
    frame = comparison.difference_bands.loc[
        comparison.difference_bands["metric"].eq("home_import_kw")
    ]
    evening = frame.loc[frame["interval_start_london"].dt.hour.between(16, 22, inclusive="left")]
    assert p50 == evening.loc[evening["p50"].abs().idxmax(), "p50"]


def test_evening_peak_kw_is_none_when_the_window_is_empty() -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    comparison = compare_runs(a, b)
    empty = comparison.difference_bands.loc[comparison.difference_bands["metric"].eq("nope")]

    assert _evening_peak_kw(empty) is None


def test_weekly_metric_by_world_sums_one_paths_slots_per_world() -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42)

    values = _weekly_metric_by_world(a.fleet_world_intervals, "selected", "public_import_kwh")

    rows = a.fleet_world_intervals.loc[a.fleet_world_intervals["path_id"].eq("selected")]
    expected = rows.groupby("world_id")["public_import_kwh"].sum().sort_index().to_numpy()
    assert np.allclose(values, expected)
    assert len(values) == a.world_count


def test_paired_quantiles_never_subtracts_percentiles() -> None:
    values_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    values_b = np.array([2.0, 2.0, 2.0, 10.0, 10.0])

    p10, p50, p90 = _paired_quantiles(values_a, values_b)

    delta = values_b - values_a
    assert (p10, p50, p90) == tuple(np.quantile(delta, (0.1, 0.5, 0.9), method="linear"))
    # Not the difference of each side's own median (that would be 2 - 3 = -1).
    assert p50 != np.quantile(values_b, 0.5) - np.quantile(values_a, 0.5)


def test_public_top_up_kwh_matches_a_hand_computed_paired_difference() -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=7.0)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=3.6)

    p10, p50, p90 = _public_top_up_kwh(a, b, "selected", "selected")

    values_a = _weekly_metric_by_world(a.fleet_world_intervals, "selected", "public_import_kwh")
    values_b = _weekly_metric_by_world(b.fleet_world_intervals, "selected", "public_import_kwh")
    expected = np.quantile(values_b - values_a, (0.1, 0.5, 0.9), method="linear")
    assert np.allclose((p10, p50, p90), expected)


def test_evening_peak_and_public_top_up_tiles_show_no_delta_and_a_range(monkeypatch) -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=7.0)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=3.6)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · base", a), _record("Run 3 · charger", b)],
    )
    st = RecordingStreamlit()

    render_compare(st, None, compare_runs=compare_runs)

    assert not calls_for(st, "metric")  # no delta arrow anywhere
    for label in ("Largest evening change", "Public top-ups"):
        _, value, _, context, _ = tile_by_label(st, label)
        assert value != "Unavailable"
        assert context.startswith("P10–P90:")
    # kW at whole numbers with a U+2212 minus, never an ASCII hyphen (G6).
    assert "-" not in tile_by_label(st, "Largest evening change")[1]


def test_firm_mw_absent_on_the_fixture_shows_no_block(monkeypatch) -> None:
    # The fixture's injectable compare_runs predates the firm_mw field
    # (trading contract v1 §10.10); render_compare must not crash and must
    # not draw the block when it is missing or None.
    a = make_result(model="action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · base", a), _record("Run 3 · home charger 11 kW", b)],
    )
    st = RecordingStreamlit()

    render_compare(st, None, compare_runs=compare_runs)

    titles = [args[0] for args, _ in calls_for(st, "markdown") if args[0].startswith("**")]
    assert not any("Firm MW" in title for title in titles)


def test_firm_mw_table_formats_mw_and_fraction_cells_and_names_the_maker() -> None:
    frame = pd.DataFrame(
        [
            {
                "frame": "product_sheet",
                "key": "evening",
                "metric": "window_mean_mw_p10",
                "unit": "MW",
                "value_a": 1.2345,
                "value_b": float("nan"),
            },
            {
                "frame": "firmness_by_manufacturer",
                "key": "m1",
                "metric": "firmness_p50",
                "unit": "fraction",
                "value_a": 0.987,
                "value_b": 0.95,
            },
        ]
    )
    table = compare_module._firm_mw_table(frame)
    assert list(table["Frame"]) == ["Product sheet", "Firmness"]
    assert list(table["Row"]) == ["Evening", "Maker A"]
    assert table["Run A"].iloc[0] == "1.23 MW"
    assert table["Run B"].iloc[0] == "Unavailable"
    assert table["Run A"].iloc[1] == "99%"


def test_dataframe_height_shows_every_row_of_a_short_table() -> None:
    # Goal review O-9: with no explicit height, st.dataframe's own default
    # auto-height scrolled the Firm MW table and cut it off at Maker B.
    # A height sized to the row count leaves nothing to scroll past.
    assert compare_module._dataframe_height(0) == compare_module._TABLE_HEADER_PX
    for row_count in (1, 4, 10):
        height = compare_module._dataframe_height(row_count)
        assert height == compare_module._TABLE_HEADER_PX + compare_module._TABLE_ROW_PX * row_count
        assert height > compare_module._TABLE_HEADER_PX


def test_firm_mw_table_height_grows_with_its_row_count(monkeypatch) -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=42, home_charger_kw=11.0)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · base", a), _record("Run 3 · charger", b)],
    )
    firm_mw = pd.DataFrame(
        [
            {
                "frame": "firmness_by_manufacturer",
                "key": key,
                "metric": "firmness_p50",
                "unit": "fraction",
                "value_a": 0.9,
                "value_b": 0.9,
            }
            for key in ("m1", "m2", "m3", "m4")
        ]
    )

    class _WithFirmMW:
        """Wraps the fixture's frozen comparison, adding the field it predates."""

        def __init__(self, backing, firm_mw):
            self._backing = backing
            self.firm_mw = firm_mw

        def __getattr__(self, name):
            return getattr(self._backing, name)

    def compare_with_firm_mw(record_a, record_b):
        return _WithFirmMW(compare_runs(record_a, record_b), firm_mw)

    st = RecordingStreamlit()

    render_compare(st, None, compare_runs=compare_with_firm_mw)

    dataframe_calls = calls_for(st, "dataframe")
    firm_mw_call = next(
        (args, kwargs) for args, kwargs in dataframe_calls if len(args[0]) == len(firm_mw)
    )
    assert firm_mw_call[1]["height"] == compare_module._dataframe_height(len(firm_mw))


def test_unmatched_futures_chart_title_is_never_called_paired(monkeypatch) -> None:
    a = make_result(model="action", evs=6, worlds=5, seed=42)
    b = make_result(model="action", evs=6, worlds=5, seed=7)
    monkeypatch.setattr(
        compare_module,
        "run_history",
        lambda state: [_record("Run 2 · base", a), _record("Run 3 · seed 7", b)],
    )
    st = RecordingStreamlit()

    render_compare(st, None, compare_runs=compare_runs)

    titles = [args[0] for args, _ in calls_for(st, "markdown") if args[0].startswith("**B")]
    assert titles == ["**B − A (not matched futures)**"]
