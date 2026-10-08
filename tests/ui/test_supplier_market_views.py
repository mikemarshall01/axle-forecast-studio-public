"""Supplier ▸ 1 Availability and cost curve and ▸ 2 Positions on the SYNTHETIC FIXTURE."""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
from fixtures.result_fixture import make_result

from axle_studio.ui.views import supplier_availability, supplier_positions

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from dashboard_lint import RecordingStreamlit  # noqa: E402


@pytest.fixture(scope="module")
def result():
    return make_result("action", evs=12, worlds=6, seed=42)


def test_cost_curve_defaults_to_18_00_and_plots_the_stored_rows(result) -> None:
    st = RecordingStreamlit()
    supplier_availability.render_availability(st, result)
    figures = dict(st.figures())
    curve = figures["supplier-cost-curve"]
    stored = result.flex_cost_curve
    rows = stored.loc[
        stored["local_half_hour"].eq("18:00")
        & stored["day_type"].eq("weekday")
        & stored["metric"].eq("available_kw")
    ].sort_values("threshold_gbp_per_mwh")
    assert list(curve.data[2].y) == list(rows["p50"])


def test_availability_is_shown_in_kw_unscaled(result) -> None:
    # Q-12: kW throughout Supplier ▸ 1 (fleet power stays below 10 MW), matching
    # this chart's own tiles, cost curve and table, which were already kW.
    st = RecordingStreamlit()
    supplier_availability.render_availability(st, result)
    figure = dict(st.figures())["supplier-availability"]
    bands = result.deferrable_power_bands
    total = bands.loc[bands["slack_bucket"].eq("total")].sort_values("slot_index")
    assert list(figure.data[2].y) == pytest.approx(list(total["p50"]))


def test_positions_chart_and_table_are_kw_the_csv_stays_mw(result) -> None:
    # Q-12: the day-ahead/final position chart and its on-screen table show
    # kW, matching the open-position chart below; only the CSV download
    # (a separate test) keeps the model's stored MW columns unscaled.
    st = RecordingStreamlit()
    supplier_positions.render_positions(st, result)
    figure = dict(st.figures())["supplier-positions"]
    positions = result.supplier_positions.sort_values("slot_index")
    # band_and_line adds the band's low and high edges before its P50 line
    # (style.py); day-ahead position is the first of the two series plotted.
    day_ahead_p50 = figure.data[2]
    assert day_ahead_p50.name == "Day-ahead position"
    assert list(day_ahead_p50.y) == pytest.approx(
        list(positions["day_ahead_position_mw_p50"] * 1000.0)
    )

    table = [args[0] for name, args, _ in st.calls if name == "dataframe"][0]
    assert "Day-ahead position P50 (kW)" in table.columns
    assert list(table["Day-ahead position P50 (kW)"]) == pytest.approx(
        list(positions["day_ahead_position_mw_p50"] * 1000.0)
    )


def test_positions_download_is_the_stored_frame(result) -> None:
    st = RecordingStreamlit()
    supplier_positions.render_positions(st, result)
    call = next(kwargs for name, _, kwargs in st.calls if name == "download_button")
    assert call["file_name"] == "axle_positions_2026-09-28_seed42.csv"
    written = pd.read_csv(io.StringIO(call["data"]))
    assert list(written.columns) == list(result.supplier_positions.columns)
    assert len(written) == 336


class _Upload:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_upload_callback_sets_the_draft_or_keeps_the_error() -> None:
    stored: list = []
    curve = pd.DataFrame({"half_hour_start": ["00:00"], "price_gbp_per_mwh": [1.0]})

    def validate(state, data: bytes) -> pd.DataFrame:
        if data == b"bad":
            raise ValueError("price curve must have 48 rows, not 1")
        return curve

    def set_curve(state, value) -> None:
        stored.append(value)

    state = {supplier_positions.UPLOAD_KEY: _Upload(b"good")}
    supplier_positions._on_upload(state, validate, set_curve)
    assert stored == [curve] and state[supplier_positions.UPLOAD_ERROR_KEY] is None

    state[supplier_positions.UPLOAD_KEY] = _Upload(b"bad")
    supplier_positions._on_upload(state, validate, set_curve)
    assert len(stored) == 1  # the draft keeps the earlier curve
    assert "48 rows" in state[supplier_positions.UPLOAD_ERROR_KEY]

    state[supplier_positions.UPLOAD_KEY] = None
    supplier_positions._on_upload(state, validate, set_curve)
    assert stored[-1] is None


def test_upload_note_says_run_to_apply(result) -> None:
    st = RecordingStreamlit()
    st.session_state[supplier_positions.DRAFT_CURVE_KEY] = pd.DataFrame({"a": [1]})
    supplier_positions.render_positions(
        st, result, validate_price_curve=lambda s, d: None, set_price_curve=lambda s, c: None
    )
    captions = [str(args[0]) for name, args, _ in st.calls if name == "caption"]
    assert "Curve loaded into the draft. Run simulation to apply it." in captions


_CITATION = re.compile(r"contract v\d|§|\bQ-\d|decision \d{4}|\baudit O\d|\bplan C\d")


def test_tooltips_and_definitions_carry_no_internal_citations(result) -> None:
    # Voice pass: a tooltip or definition says the rule in words; "trading
    # contract v1 §9.4" tells an Axle reader nothing.
    for render in (supplier_availability.render_availability, supplier_positions.render_positions):
        st = RecordingStreamlit()
        render(st, result)
        texts = [
            text
            for _, args, kwargs in st.calls
            for text in (*args, kwargs.get("help", ""))
            if isinstance(text, str)
        ]
        assert [text for text in texts if _CITATION.search(text)] == []
