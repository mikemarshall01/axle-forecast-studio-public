"""Supplier ▸ 4 Firm MW (trading contract v1 §10.10).

The SYNTHETIC FIXTURE (``fixtures.result_fixture.make_result``) predates the
availability, product and calibration frames (§10), so it only exercises the
"missing frames" path. Chart content is checked against a small REAL run
(``run_forecast_from_assumptions``, 30 EVs x 6 simulated weeks): the only
model call this file makes, mirroring ``tests/model/test_assumption_effects.py``'s
own small-real-run pattern, because no shared fixture yet carries these frames.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pytest
from fixtures.result_fixture import make_result
from kpi_calls import kpi_calls

from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.ui.views import supplier_firm_mw

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from dashboard_lint import RecordingStreamlit, lint_figure  # noqa: E402

_START = date(2026, 1, 12)  # a Monday; matches test_assumption_effects.py's _START


@pytest.fixture(scope="module")
def toy_result():
    return make_result("action", evs=12, worlds=6, seed=42)


@pytest.fixture(scope="module")
def result():
    return run_forecast_from_assumptions(
        _START, model="action", values={"vehicle_count": 30, "evaluation_world_count": 6}
    )


@pytest.fixture(scope="module")
def timed_result():
    """Same small real run, with the optional timed path on (decision 0007)."""

    return run_forecast_from_assumptions(
        _START,
        model="action",
        values={
            "vehicle_count": 30,
            "evaluation_world_count": 6,
            "timed_start_local_hour": 0.0,
        },
    )


def _figures(st: RecordingStreamlit) -> dict:
    return dict(st.figures())


def _texts(st: RecordingStreamlit, *names: str) -> list[str]:
    return [str(args[0]) for name, args, _ in st.calls if name in names and args]


def test_missing_frames_shows_one_info_message_and_draws_nothing(toy_result) -> None:
    st = RecordingStreamlit()

    supplier_firm_mw.render_firm_mw(st, toy_result)

    assert st.figures() == []
    info = _texts(st, "info")
    assert len(info) == 1
    assert "Firm MW" in info[0]
    # B-19: no internal contract citation on the main surface.
    assert "§" not in info[0] and "contract" not in info[0].lower()


def test_renders_every_chart_for_the_default_controls(result) -> None:
    st = RecordingStreamlit()

    supplier_firm_mw.render_firm_mw(st, result)

    keys = set(_figures(st))
    # B-8: the empty per-half-hour "firm share across the week" chart was
    # dropped for an honest caption; the fleet-size chart below it already
    # carries the diversification finding.
    assert keys == {
        "firm-mw-day-ahead",
        "firm-mw-intraday",
        "firm-mw-fleet-size",
        "firm-mw-reliability",
        "firm-mw-backtest-half-hour",
    }


@pytest.mark.parametrize(
    ("direction", "duration", "night", "window"),
    [
        ("turn_down", 1.0, 0, "evening"),
        ("turn_up", 4.0, 6, "morning"),
        ("turn_down", 0.5, 3, "overnight"),
    ],
)
def test_renders_for_every_control_combination(result, direction, duration, night, window) -> None:
    st = RecordingStreamlit(
        choices={
            "firm-mw-direction": direction,
            "firm-mw-duration": duration,
            "firm-mw-intraday-night": night,
            "firm-mw-fleet-size-window": window,
        }
    )

    supplier_firm_mw.render_firm_mw(st, result)

    assert len(_figures(st)) == 5


def test_day_ahead_band_reads_the_stored_quantiles_unchanged(result) -> None:
    bands = result.availability_bands
    rows = bands.loc[
        bands["horizon"].eq("day_ahead")
        & bands["statistic"].eq("realised")
        & bands["direction"].eq("turn_down")
        & bands["duration_hours"].eq(1.0)
    ].sort_values("slot_index")

    st = RecordingStreamlit()
    supplier_firm_mw.render_firm_mw(st, result)
    figure = _figures(st)["firm-mw-day-ahead"]

    # band_and_line adds the low and high band edges before the mid line,
    # all three sharing one legend name (style.py); the mid line is last and
    # is the only one with a visible line width.
    named = [trace for trace in figure.data if trace.name == "Smart path, P50"]
    p50_trace = named[-1]
    assert p50_trace.line.width != 0
    # Q-12: kW, unscaled (the stored column, matching this chart's own
    # already-kW table).
    expected = rows["p50"].to_numpy(dtype=float).tolist()
    assert list(p50_trace.y) == pytest.approx(expected, nan_ok=True)
    # §10.10: the P50 line's hover carries the across-weeks sd too.
    assert "sd" in p50_trace.hovertemplate
    expected_sd = rows["sd"].to_numpy(dtype=float).tolist()
    assert [pair[1] for pair in p50_trace.customdata] == pytest.approx(expected_sd, nan_ok=True)


def test_intraday_caption_names_the_typical_firm_and_known_part_figures(result) -> None:
    # §10.10: "typical firm figure at 17:00" (conditional_p10) beside the
    # known-part typical figure (conditional_known_p50), both read at the
    # decision slot the chart itself starts from, one row of many in
    # availability_bands (a naive lookup by horizon/statistic alone would
    # pick an arbitrary, often pre-decision, slot).
    st = RecordingStreamlit()
    supplier_firm_mw.render_firm_mw(st, result)
    typical = next(c for c in _texts(st, "caption") if c.startswith("Typical firm figure"))
    assert len(typical) <= 140

    world_slot = result.availability_world_slot
    night_rows = world_slot.loc[
        world_slot["world_id"].eq(result.representative_world_id)
        & world_slot["night_index"].eq(0)
        & world_slot["direction"].eq("turn_down")
        & world_slot["duration_hours"].eq(1.0)
        & world_slot["intraday_mean_kw"].notna()
    ].sort_values("slot_index")
    decision_slot = int(night_rows.iloc[0]["slot_index"])

    for statistic in ("conditional_p10", "conditional_known_p50"):
        row = supplier_firm_mw.stat_row(
            result.availability_bands,
            horizon="intraday",
            statistic=statistic,
            direction="turn_down",
            duration_hours=1.0,
            slot_index=decision_slot,
        )
        expected = (
            "Unavailable"
            if row is None or supplier_firm_mw.missing(row["p50"])
            else supplier_firm_mw.format_quantity(row["p50"], "kW", decimals=0)
        )
        assert expected in typical, f"{statistic}: {expected!r} not in {typical!r}"


def test_blackout_note_says_none_by_default(result) -> None:
    st = RecordingStreamlit()

    supplier_firm_mw.render_firm_mw(st, result)

    captions = _texts(st, "caption")
    assert any("No blackout windows" in caption for caption in captions)


def test_settlement_file_download_writes_the_stored_frame_and_name(result) -> None:
    st = RecordingStreamlit()

    supplier_firm_mw.render_firm_mw(st, result)

    downloads = [(args, kwargs) for name, args, kwargs in st.calls if name == "download_button"]
    assert len(downloads) == 1
    _, kwargs = downloads[0]
    assert kwargs["file_name"] == result.settlement_file_name
    assert kwargs["data"] == result.settlement_file.to_csv(index=False)


def test_kpi_tiles_never_exceed_four_per_row(result) -> None:
    # kpi_columns raises past four (components/kpi.py); reaching the
    # assertion below is itself the check.
    st = RecordingStreamlit()

    supplier_firm_mw.render_firm_mw(st, result)

    assert st.figures()


def test_calibration_tiles_and_chart_show_both_hit_rates(result) -> None:
    # B-6: many half-hours forecast and realise exactly 0 kW; showing only
    # the "at or below" rate reads as a badly over-covered forecast.
    st = RecordingStreamlit()
    supplier_firm_mw.render_firm_mw(st, result)
    tiles = "\n".join(_texts(st, "markdown"))
    assert "Strict to weak" in tiles

    figure = _figures(st)["firm-mw-reliability"]
    names = {trace.name for trace in figure.data}
    assert any(name and "at or below" in name for name in names)
    assert any(name and "strictly below" in name for name in names)


def test_product_sheet_movable_energy_states_the_period_and_hides_none(result) -> None:
    # B-7: the column sums a window's slots over all seven nights (MWh per
    # week, not per slot or event), and a NaN duration row must read
    # "Unavailable", never pandas' bare "None".
    st = RecordingStreamlit()
    supplier_firm_mw.render_firm_mw(st, result)
    table = next(
        args[0]
        for name, args, _ in st.calls
        if name == "dataframe" and "Movable energy, P50 (MWh per week)" in list(args[0].columns)
    )
    column = table["Movable energy, P50 (MWh per week)"].astype(str).tolist()
    assert "None" not in column
    assert supplier_firm_mw.UNAVAILABLE in column


def test_header_names_the_base_against_availability(result) -> None:
    # Q-6: Firm MW measures turn-down beyond the smart schedule; Availability
    # and Key stats measure against the unmanaged baseline, a 20-300x gap.
    st = RecordingStreamlit()
    supplier_firm_mw.render_firm_mw(st, result)
    captions = _texts(st, "caption")
    assert any("Availability" in c and "unmanaged" in c for c in captions)


def test_diversification_tab_explains_the_dropped_chart(result) -> None:
    # B-8: the empty per-half-hour firm-share chart is hidden with an honest
    # message rather than shown under a caption asserting a finding.
    st = RecordingStreamlit()
    supplier_firm_mw.render_firm_mw(st, result)
    captions = _texts(st, "caption")
    new_caption = next(c for c in captions if "sampling noise" in c and "fleet size" in c)
    assert len(new_caption) <= 140


def test_charts_follow_the_dashboard_conventions_or_a_known_accepted_pattern(result) -> None:
    # Mirrors scripts/dashboard_lint.py's rules directly: the toy fixture used
    # by tests/ui/test_dashboard_lint.py never populates these frames (§10),
    # so that sweep cannot exercise this view; this test is the substitute.
    # Every finding here is the same accepted "one realised/reference line on
    # a banded time series" pattern already recorded for Trading Market and
    # Trading Position in tests/ui/test_dashboard_lint.py's ACCEPTED list.
    accepted = {
        ("firm-mw-day-ahead", "line-without-band"),
        ("firm-mw-intraday", "line-without-band"),
    }
    st = RecordingStreamlit()
    supplier_firm_mw.render_firm_mw(st, result)

    for key, figure in _figures(st).items():
        for finding in lint_figure(figure):
            assert (key, finding.rule) in accepted, f"{key}: {finding}"


_CITATION = re.compile(r"contract v\d|§|\bQ-\d|decision \d{4}|\baudit O\d|\bplan C\d")


def test_tooltips_and_definitions_carry_no_internal_citations(result) -> None:
    # Voice pass: a tooltip or definition says the rule in words; "trading
    # contract v1 §10.5g" tells an Axle reader nothing.
    st = RecordingStreamlit()
    supplier_firm_mw.render_firm_mw(st, result)
    texts = [
        text
        for _, args, kwargs in st.calls
        for text in (*args, kwargs.get("help", ""))
        if isinstance(text, str)
    ]
    assert [text for text in texts if _CITATION.search(text)] == []


# --- The optional timed path (decision 0007, model step 2) ------------------


@pytest.fixture(scope="module")
def untimed_result():
    return run_forecast_from_assumptions(
        _START,
        model="action",
        values={"vehicle_count": 30, "evaluation_world_count": 6, "timed_tariff_enabled": 0},
    )


def test_completion_stays_two_policy_without_the_timed_path(untimed_result) -> None:
    # Absent (the Timed tariff policy switched off): the completion row and
    # the maker table are exactly as before.
    result = untimed_result
    st = RecordingStreamlit()
    supplier_firm_mw.render_firm_mw(st, result)
    labels = [tile[0] for tile in kpi_calls(st)]
    start = labels.index("Unmanaged, all sessions")
    assert labels[start : start + 4] == [
        "Unmanaged, all sessions",
        "Unmanaged, early departures",
        "Smart, all sessions",
        "Smart, early departures",
    ]
    assert not any(label.startswith("Timed") for label in labels)
    table = next(
        args[0]
        for name, args, _ in st.calls
        if name == "dataframe" and "Complete by departure, Smart" in list(args[0].columns)
    )
    assert list(table.columns) == [
        "Maker",
        "Complete by departure, Unmanaged",
        "Complete by departure, Smart",
    ]


def test_completion_gains_a_timed_tariff_row_beside_smart(timed_result) -> None:
    st = RecordingStreamlit()
    supplier_firm_mw.render_firm_mw(st, timed_result)
    labels = [tile[0] for tile in kpi_calls(st)]
    start = labels.index("Unmanaged, all sessions")
    # The first four tiles (Unmanaged, Smart) are unchanged; the timed pair
    # is a second row below them, never replacing either.
    assert labels[start : start + 4] == [
        "Unmanaged, all sessions",
        "Unmanaged, early departures",
        "Smart, all sessions",
        "Smart, early departures",
    ]
    assert labels[start + 4 : start + 6] == ["Timed tariff, all sessions", "Timed tariff, early"]


def test_completion_maker_table_gains_a_timed_tariff_column(timed_result) -> None:
    st = RecordingStreamlit()
    supplier_firm_mw.render_firm_mw(st, timed_result)
    table = next(
        args[0]
        for name, args, _ in st.calls
        if name == "dataframe" and "Complete by departure, Timed tariff" in list(args[0].columns)
    )
    assert list(table.columns) == [
        "Maker",
        "Complete by departure, Unmanaged",
        "Complete by departure, Timed tariff",
        "Complete by departure, Smart",
    ]


def test_completion_block_charts_still_follow_the_dashboard_conventions(timed_result) -> None:
    accepted = {
        ("firm-mw-day-ahead", "line-without-band"),
        ("firm-mw-intraday", "line-without-band"),
    }
    st = RecordingStreamlit()
    supplier_firm_mw.render_firm_mw(st, timed_result)
    for key, figure in _figures(st).items():
        for finding in lint_figure(figure):
            assert (key, finding.rule) in accepted, f"{key}: {finding}"


def test_product_sheet_mw_columns_carry_a_fixed_three_decimal_format() -> None:
    # st.dataframe prints raw floats ("0.0743" beside "0") without a format.
    formats = supplier_firm_mw._PRODUCT_SHEET_FORMATS
    assert set(formats) == {"Mean MW, firm (P10)", "Mean MW (P50)", "Min-through-window MW (P50)"}
    assert set(formats) <= set(supplier_firm_mw._PRODUCT_SHEET_DISPLAY.values())
    assert all(config["type_config"]["format"] == "%.3f" for config in formats.values())
