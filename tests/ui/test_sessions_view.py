"""Drivers ▸ Sessions renders session_distribution_bands (decision 0004 item 54, plan D-2)."""

from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np
import pytest
from fixtures.result_fixture import make_result

from axle_studio.model.assumptions import SESSION_SOURCE_REFERENCES
from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.ui.registry import DAY_TYPE_KEY
from axle_studio.ui.style import ARCHETYPE_COLOURS, SERIES_COLOURS
from axle_studio.ui.views.sessions import (
    ARCHETYPE_ORDER,
    CENSORING_CAPTION,
    METRICS,
    render_sessions,
)

TODAY = date(2026, 9, 29)


class RecordingStreamlit:
    """Fake ``st``: records every call; widgets return the chosen value or their default."""

    def __init__(self, *, metric: str = "Plug-in time", day: str = "weekday") -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self._choices = {"sessions-metric": metric, DAY_TYPE_KEY: day}

    def __getattr__(self, name: str):
        def record(*args: object, **kwargs: object) -> object:
            self.calls.append((name, args, kwargs))
            if name == "columns":
                count = args[0] if args else kwargs.get("spec")
                return [self] * (count if isinstance(count, int) else len(count))
            if name == "expander":
                return self
            if name in ("selectbox", "segmented_control"):
                return self._choices[kwargs["key"]]
            return None

        return record

    def __enter__(self) -> "RecordingStreamlit":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def calls_for(st: RecordingStreamlit, name: str) -> list[tuple[tuple, dict]]:
    return [(args, kwargs) for call_name, args, kwargs in st.calls if call_name == name]


def with_source_records(result):
    """The fixture omits the 8.2 source records; add the real ones from the model."""

    records = tuple(
        record for pair in SESSION_SOURCE_REFERENCES.values() for record in pair.values()
    )
    return dataclasses.replace(result, assumptions=result.assumptions + records)


def figure_of(st: RecordingStreamlit):
    return calls_for(st, "plotly_chart")[0][0][0]


@pytest.fixture(scope="module")
def timed_result():
    """A real small run with the optional timed path on (decision 0007, model step
    2): the shared contract v2 fixture (``fixtures.result_fixture``) stays
    two-policy, so ``departure_soc_percent_timed`` needs an actual model run."""

    return run_forecast_from_assumptions(
        TODAY,
        model="action",
        values={"vehicle_count": 30, "evaluation_world_count": 3, "timed_start_local_hour": 0.0},
    )


@pytest.mark.parametrize("model", ["action", "no_action"])
@pytest.mark.parametrize("metric", list(METRICS))
def test_every_metric_renders_six_panels(model: str, metric: str) -> None:
    result = make_result(model, evs=6, worlds=6)
    st = RecordingStreamlit(metric=metric)

    render_sessions(st, result)

    figure = figure_of(st)
    titles = [annotation.text for annotation in figure.layout.annotations][:6]
    summary = result.cohort_summary.set_index("cohort_id")
    # "IO average" for Intelligent Octopus only (final critique B-16: the
    # canonical label collided with its neighbour at 390 px); the other five
    # archetypes keep their canonical cohort_label unabridged.
    short_labels = {"intelligent_octopus": "IO average"}
    for title, cohort in zip(titles, ARCHETYPE_ORDER, strict=True):
        count = int(summary.loc[cohort, "ev_count"])
        label = short_labels.get(cohort, summary.loc[cohort, "cohort_label"])
        assert title == f"{label} · {count} EV{'s' if count != 1 else ''}"
    assert not calls_for(st, "button")


def test_header_slot_holds_only_the_day_control() -> None:
    # Final critique B-16: the metric selectbox used to share the header
    # controls slot with the day-type control ([3, 2] columns), leaving the
    # day control too narrow at 1440 px ("All" clipped off, "Weekend" half
    # visible). It now sits alone in the slot, one full-width column, as
    # Plug-ins' day control does; the metric selectbox moves to its own line
    # in the page body, where width is not scarce.
    result = make_result("action", evs=6, worlds=6)
    st = RecordingStreamlit()

    render_sessions(st, result)

    columns_calls = calls_for(st, "columns")
    assert columns_calls[0][0] == (1,)
    ((_, day_kwargs),) = calls_for(st, "segmented_control")
    assert day_kwargs["key"] == DAY_TYPE_KEY
    metric_calls = [
        kwargs for _, kwargs in calls_for(st, "selectbox") if kwargs["key"] == "sessions-metric"
    ]
    assert len(metric_calls) == 1


def test_bars_are_p50_shares_with_p10_p90_whiskers_in_archetype_colours() -> None:
    result = make_result("action", evs=6, worlds=8)
    st = RecordingStreamlit(metric="Energy needed", day="all")

    render_sessions(st, result)

    figure = figure_of(st)
    bands = result.session_distribution_bands
    for trace in figure.data:
        cohort = next(
            c
            for c in ARCHETYPE_ORDER
            if trace.name
            == dict(zip(result.cohort_summary["cohort_id"], result.cohort_summary["cohort_label"]))[
                c
            ]
        )
        rows = bands.loc[
            bands["group_id"].eq(cohort)
            & bands["day_type"].eq("all")
            & bands["metric"].eq("energy_needed_kwh")
        ].sort_values("bin_index")
        p50 = rows["share_p50"].to_numpy() * 100.0
        np.testing.assert_allclose(np.asarray(trace.y, dtype=float), p50)
        np.testing.assert_allclose(
            np.asarray(trace.error_y.array, dtype=float), rows["share_p90"] * 100.0 - p50
        )
        np.testing.assert_allclose(
            np.asarray(trace.error_y.arrayminus, dtype=float), p50 - rows["share_p10"] * 100.0
        )
        assert trace.marker.color == ARCHETYPE_COLOURS[ARCHETYPE_ORDER.index(cohort)]


def test_panels_share_one_y_axis() -> None:
    result = make_result("action", evs=6, worlds=6)
    st = RecordingStreamlit()

    render_sessions(st, result)

    layout = figure_of(st).layout
    names = ["yaxis", *(f"yaxis{n}" for n in range(2, 7))]
    # Plotly ties every panel to one anchor axis, which itself has no match.
    anchors = {layout[name].matches for name in names} - {None}
    assert len(anchors) == 1
    assert sum(layout[name].matches is None for name in names) == 1


def test_source_markers_on_soc_and_energy_only_when_records_exist() -> None:
    plain = make_result("action", evs=6, worlds=6)
    with_records = with_source_records(plain)

    def dashed(result, metric):
        st = RecordingStreamlit(metric=metric)
        render_sessions(st, result)
        return [
            shape
            for shape in figure_of(st).layout.shapes
            if shape.line.dash == "dash" and shape.line.color == SERIES_COLOURS["observed"]
        ]

    # The fixture carries no 8.2 records: no marker, never a guessed one.
    assert dashed(plain, "SoC at plug-in") == []
    soc = dashed(with_records, "SoC at plug-in")
    assert sorted(shape.x0 for shape in soc) == sorted(
        SESSION_SOURCE_REFERENCES[cohort]["source_plug_in_soc_percent"].value
        for cohort in ARCHETYPE_ORDER
    )
    energy = dashed(with_records, "Energy needed")
    assert sorted(shape.x0 for shape in energy) == sorted(
        SESSION_SOURCE_REFERENCES[cohort]["source_battery_kwh_per_plug_in"].value
        for cohort in ARCHETYPE_ORDER
    )
    assert dashed(with_records, "Dwell") == []


def test_archetype_without_evs_says_no_sessions_and_gets_no_marker() -> None:
    result = with_source_records(make_result("no_action", evs=5, worlds=6))
    st = RecordingStreamlit(metric="SoC at plug-in")

    render_sessions(st, result)

    figure = figure_of(st)
    assert "No sessions" in [annotation.text for annotation in figure.layout.annotations]
    assert len(figure.data) == 5
    assert len(figure.layout.shapes) == 5


def test_censoring_caption_and_short_captions() -> None:
    result = make_result("action", evs=6, worlds=6)
    st = RecordingStreamlit(metric="Dwell")

    render_sessions(st, result)

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert CENSORING_CAPTION in captions
    # The on-screen captions (the definition sits inside the expander).
    assert all(len(caption) <= 140 for caption in captions[:1] + [CENSORING_CAPTION])


def test_day_type_control_is_the_shared_one() -> None:
    result = make_result("action", evs=6, worlds=6)
    st = RecordingStreamlit(day="weekend")

    render_sessions(st, result)

    ((_, kwargs),) = calls_for(st, "segmented_control")
    assert kwargs["key"] == DAY_TYPE_KEY
    assert kwargs["persist_state"] == "session"
    table = calls_for(st, "dataframe")[0][0][0]
    bands = result.session_distribution_bands
    assert len(table) == len(
        bands.loc[bands["day_type"].eq("weekend") & bands["metric"].eq("plug_in_time")]
    )


def test_shared_y_range_reaches_the_tallest_whisker() -> None:
    # Browser check: the matched axes autoranged from one panel and clipped
    # the others, so the view sets the range explicitly.
    result = make_result("action", evs=6, worlds=6)
    st = RecordingStreamlit(metric="SoC at plug-in")

    render_sessions(st, result)

    bands = result.session_distribution_bands
    tallest = (
        100.0
        * bands.loc[
            bands["group_id"].isin(ARCHETYPE_ORDER)
            & bands["day_type"].eq("weekday")
            & bands["metric"].eq("plug_in_soc_percent"),
            "share_p90",
        ].max()
    )
    low, high = figure_of(st).layout.yaxis.range
    assert low == 0.0
    assert high >= tallest


# --- "SoC at departure, timed" (decision 0007, model step 2) -------------------


def test_timed_departure_soc_offered_only_when_the_run_has_it(timed_result) -> None:
    off = make_result("action", evs=6, worlds=6)
    st_off = RecordingStreamlit()
    render_sessions(st_off, off)
    ((_, off_kwargs),) = calls_for(st_off, "selectbox")
    assert "SoC at departure, timed" not in off_kwargs["options"]

    st_on = RecordingStreamlit()
    render_sessions(st_on, timed_result)
    ((_, on_kwargs),) = calls_for(st_on, "selectbox")
    options = on_kwargs["options"]
    assert "SoC at departure, timed" in options
    # On-screen path order (decision 0007): Unmanaged, Timed tariff, Smart.
    assert (
        options.index("SoC at departure")
        < options.index("SoC at departure, timed")
        < options.index("SoC at departure, smart")
    )


def test_timed_departure_soc_renders_from_its_own_metric_column(timed_result) -> None:
    st = RecordingStreamlit(metric="SoC at departure, timed", day="all")

    render_sessions(st, timed_result)

    figure = figure_of(st)
    bands = timed_result.session_distribution_bands
    labels = dict(
        zip(timed_result.cohort_summary["cohort_id"], timed_result.cohort_summary["cohort_label"])
    )
    traced = 0
    for trace in figure.data:
        cohort = next(c for c in ARCHETYPE_ORDER if trace.name == labels[c])
        rows = bands.loc[
            bands["group_id"].eq(cohort)
            & bands["day_type"].eq("all")
            & bands["metric"].eq("departure_soc_percent_timed")
        ].sort_values("bin_index")
        p50 = rows["share_p50"].to_numpy() * 100.0
        np.testing.assert_allclose(np.asarray(trace.y, dtype=float), p50)
        traced += 1
    assert traced > 0  # at least one archetype actually plotted, not all "No sessions"
    # The chart's own caption (first; the rest, including the "Data and
    # definition" expander's longer text, are not copy-rule length checked).
    chart_caption = calls_for(st, "caption")[0][0][0]
    assert "timed tariff path" in chart_caption
    assert len(chart_caption) <= 140
