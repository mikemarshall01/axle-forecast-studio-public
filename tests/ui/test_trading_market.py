"""Trading lens 1, Market (trading contract v1 sections 5.1, 5.6a, 6; decision 0005)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fixtures.result_fixture import make_result
from kpi_calls import kpi_calls

from axle_studio.ui.style import SERIES_COLOURS
from axle_studio.ui.views import trading_market
from axle_studio.ui.views.trading_market import (
    _evidence_text,
    _negative_price_share,
    _shock_table,
    render_trading_market,
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


def test_negative_price_share_is_a_flat_share_over_every_world_and_slot() -> None:
    prices = pd.DataFrame({"wholesale_forecast_gbp_per_mwh": [-1.0, 2.0, -3.0, 4.0]})
    assert _negative_price_share(prices) == pytest.approx(0.5)


def test_evidence_text_never_shows_a_raw_slug() -> None:
    assert _evidence_text("illustrative_synthetic") == "illustrative, synthetic"
    assert _evidence_text("synthetic") == "synthetic"
    assert _evidence_text("something_new") == "something, new"


def test_shock_table_has_no_raw_slugs_and_orders_known_before_surprise() -> None:
    summary = pd.DataFrame(
        {
            "shock_class": ["big", "mild"],
            "known_day_ahead": [True, False],
            "direction": ["up", "down"],
            "world_count": [4, 4],
            "count_per_week_p10": [1.0, 0.0],
            "count_per_week_p50": [1.0, 1.0],
            "count_per_week_p90": [1.0, 2.0],
            "size_gw_p50": [4.0, 1.5],
            "peak_price_increment_gbp_per_mwh_p50": [12.0, 3.0],
        }
    )
    table = _shock_table(summary)
    assert list(table["Class"]) == ["Big", "Mild"]
    assert list(table["Timing"]) == ["Known day-ahead", "Surprise"]
    assert list(table["Direction"]) == ["Up", "Down"]


def test_render_both_models_without_crashing() -> None:
    action_result = make_result("action")
    no_action_result = make_result("no_action")

    action_st = RecordingStreamlit()
    render_trading_market(action_st, action_result)
    assert len(figures_from(action_st)) == 1

    no_action_st = RecordingStreamlit()
    render_trading_market(no_action_st, no_action_result)
    assert figures_from(no_action_st) == []
    assert calls_for(no_action_st, "info")[0][0] == ("This result has no Axle action.",)


def test_kpi_row_reports_negative_share_spread_and_shock_count() -> None:
    result = make_result("action")
    st = RecordingStreamlit()

    render_trading_market(st, result)

    tiles = kpi_calls(st)
    assert len(tiles) == 3
    labels = [tile[0] for tile in tiles]
    assert labels == ["Negative day-ahead prices", "Day-ahead spread", "Shocks this week"]
    # The fixture's toy price generator draws one big known shock per world
    # at slot 36 (item 56's shock table); the representative world carries
    # exactly that one shock, no surprises.
    shock_value, shock_context = tiles[2][1], tiles[2][3]
    assert shock_value == "1"
    assert shock_context == "1 known day-ahead, 0 surprise"


def test_price_chart_marks_the_known_shock_and_never_marks_a_surprise() -> None:
    result = make_result("action")
    st = RecordingStreamlit()

    render_trading_market(st, result)

    figure = figures_from(st)[0]
    # One shaded rectangle for the known shock's window; no surprise shapes.
    assert len(figure.layout.shapes) >= 1
    marker_traces = [t for t in figure.data if t.mode == "markers"]
    names = {t.name for t in marker_traces}
    assert "Known day-ahead shock" in names
    assert "Surprise shock" not in names


def test_price_chart_shows_day_ahead_fan_and_representative_week_lines() -> None:
    result = make_result("action")
    st = RecordingStreamlit()

    render_trading_market(st, result)

    figure = figures_from(st)[0]
    names = {t.name for t in figure.data if t.showlegend is not False and t.name}
    assert "Day-ahead (synthetic)" in names
    assert "Intraday close (representative week)" in names
    assert "Imbalance price (representative week)" in names
    day_ahead_line = next(
        t for t in figure.data if t.name == "Day-ahead (synthetic)" and t.showlegend
    )
    assert day_ahead_line.line.color == SERIES_COLOURS["synthetic"]


def test_view_makes_no_model_calls() -> None:
    source = Path(trading_market.__file__).read_text(encoding="utf-8")
    assert "axle_studio.model" not in source
    assert "run_forecast" not in source


def test_shock_table_formats_every_numeric_column_to_fixed_decimals() -> None:
    # st.dataframe drops trailing zeros, so "4" and "3.8" in one column read
    # as two precisions unless each column carries an explicit format.
    table = trading_market._shock_table(make_result("action").shock_summary)
    numeric = [name for name in table.columns if pd.api.types.is_numeric_dtype(table[name])]
    assert numeric
    assert set(numeric) == set(trading_market._SHOCK_TABLE_FORMATS)
