"""Trading lens 3, P&L and risk (trading contract v1 sections 4.7, 5.5, 9.1; decision 0005)."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pandas as pd
from fixtures.result_fixture import make_result
from kpi_calls import kpi_calls

from axle_studio.model import summaries
from axle_studio.ui.style import UNAVAILABLE
from axle_studio.ui.views import trading_pnl
from axle_studio.ui.views.trading_pnl import (
    _BUCKETS_WITH_NET,
    _bucket_table,
    _check_table,
    _control_group_caption,
    _control_group_table,
    _format_metric,
    _strategy_table,
    render_trading_pnl,
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


def test_format_metric_covers_every_trading_kpi_unit() -> None:
    assert _format_metric(12.3, "GBP per week") == "+£12"
    assert _format_metric(-1.5, "GBP per EV per year") == "−£1.50"
    assert _format_metric(2.5, "MWh per week") == "2.50 MWh"
    assert _format_metric(0.92, "fraction") == "+92.0%"
    assert _format_metric(-3.4, "GBP per MWh") == "−3.4 £/MWh"
    # Final critique B-20: money() for the £ sign and the true minus.
    assert _format_metric(120.0, "GBP per MW per year") == "+£120/MW/yr"
    assert _format_metric(-120.0, "GBP per MW per year") == "−£120/MW/yr"
    assert _format_metric(float("nan"), "GBP per week") == UNAVAILABLE


def test_bucket_table_has_one_row_per_bucket_and_net() -> None:
    result = make_result("action")
    table = _bucket_table(result.trading_ledger_summary)
    assert len(table) == len(_BUCKETS_WITH_NET) == 10
    assert table["Bucket"].iloc[-1] == "Net (illustrative)"
    assert set(table.columns) >= {"Bucket", "Day-ahead only", "Full (day-ahead + intraday)"}


def test_strategy_table_shows_unavailable_for_full_only_metrics() -> None:
    result = make_result("action")
    table = _strategy_table(result.trading_kpis).set_index("Metric")
    row = table.loc["Value of intraday (full strategy only)"]
    assert row["Day-ahead only"] == UNAVAILABLE
    assert row["Perfect foresight"] == UNAVAILABLE
    assert row["Full (day-ahead + intraday)"] != UNAVAILABLE


def test_check_table_never_shows_a_raw_check_id() -> None:
    result = make_result("action")
    table = _check_table(result.trading_checks)
    assert "buckets_sum_to_net" not in table["Check"].to_list()
    assert "Ledger buckets sum to net" in table["Check"].to_list()
    assert table["Passed"].dtype == bool


def test_render_both_models_without_crashing() -> None:
    action_result = make_result("action")
    no_action_result = make_result("no_action")

    action_st = RecordingStreamlit()
    render_trading_pnl(action_st, action_result)
    # The ledger buckets, then the intraday-dispatch rebalancing vs
    # re-optimisation split (§6.3, §10): 0 on the SYNTHETIC fixture (no
    # dispatch ever ran), still drawn so the chart is always present.
    assert len(figures_from(action_st)) == 2

    no_action_st = RecordingStreamlit()
    render_trading_pnl(no_action_st, no_action_result)
    assert figures_from(no_action_st) == []
    assert calls_for(no_action_st, "info")[0][0] == ("This result has no Axle action.",)


def test_kpi_rows_show_the_eight_trader_metrics() -> None:
    result = make_result("action")
    st = RecordingStreamlit()

    render_trading_pnl(st, result)

    tiles = kpi_calls(st)
    labels = [tile[0] for tile in tiles]
    assert labels == [
        "Net P&L, full strategy",
        "Capture rate",
        "Value of intraday",
        "Firmness",
        "Net P&L, CVaR5",
        "Margin per MW per year",
        # Intraday dispatch contract v1 §6.3, §10: 0 on the SYNTHETIC fixture.
        "Re-optimisation P&L",
        "Energy moved by dispatch",
    ]


def test_bucket_chart_has_one_bar_group_per_strategy() -> None:
    result = make_result("action")
    st = RecordingStreamlit()

    render_trading_pnl(st, result)

    figure = figures_from(st)[0]
    assert len(figure.data) == 3, "one Bar trace per strategy"
    assert all(trace.type == "bar" for trace in figure.data)
    # Final critique B-15: horizontal bars with short names on the y axis (no
    # rotated x labels to overlap at 390 px); full names stay in the hover.
    assert all(trace.orientation == "h" for trace in figure.data)
    assert figure.layout.xaxis.tickangle is None
    assert list(figure.data[0].y) == [
        "Day-ahead revenue",
        "Intraday P&L",
        "Trading cost",
        "Imbalance",
        "Baseline effect",
        "Grid events",
        "Supplier comp.",
        "Customer share",
        "Unmet-charge penalty",
        "Net (illustrative)",
    ]
    assert all(len(label) <= 20 for label in figure.data[0].y)
    assert list(figure.data[0].customdata)[0] == "Day-ahead revenue on flexibility"
    assert figure.layout.yaxis.autorange == "reversed"


def test_money_tiles_say_illustrative_and_help_names_the_statistic() -> None:
    # Final critique B-3 (the net tile is P50; help said "mean") and B-20
    # ("illustrative" on every money tile; £ on the per-MW margin).
    result = make_result("action")
    st = RecordingStreamlit()
    render_trading_pnl(st, result)

    tiles = {tile[0]: tile for tile in kpi_calls(st)}
    for label in ("Net P&L, full strategy", "Value of intraday", "Net P&L, CVaR5"):
        assert "illustrative" in tiles[label][3], label
    margin = tiles["Margin per MW per year"]
    assert "illustrative" in margin[3]
    assert margin[1].lstrip("+−").startswith("£")
    assert margin[2] == "/MW/yr"
    net_help = tiles["Net P&L, full strategy"][4]
    assert "median (P50)" in net_help and "mean" not in net_help
    # Model questions Q-3: capture at 1 dp, and the help explains passing 100%.
    capture = tiles["Capture rate"]
    assert capture[1].count(".") == 1
    assert "can pass 100" in capture[4]


def test_imbalance_per_mwh_is_named_as_signed_cash() -> None:
    # Final critique B-4: positive is income, so the row is not called a cost.
    table = _strategy_table(make_result("action").trading_kpis)
    assert "Imbalance cash per MWh (+ income)" in list(table["Metric"])
    assert not table["Metric"].str.contains("Imbalance cost").any()


def _control_group_frame() -> pd.DataFrame:
    """A tiny hand-built ``control_group_summary`` (trading contract v1 9.5)."""

    metrics = ("estimated_bias_kw_per_ev", "true_bias_kw_per_ev", "estimation_error_kw_per_ev")
    rows = []
    for night_index in (0, 1, pd.NA):
        for index, metric in enumerate(metrics):
            rows.append(
                {
                    "night_index": night_index,
                    "metric": metric,
                    "unit": "kW/EV",
                    "control_ev_count": 10,
                    "treated_ev_count": 90,
                    "world_count": 4,
                    "mean": 0.1 * (index + 1),
                    "p10": 0.05 * (index + 1),
                    "p50": 0.1 * (index + 1),
                    "p90": 0.15 * (index + 1),
                }
            )
    return pd.DataFrame(rows)


def test_control_group_section_hidden_when_absent() -> None:
    # The fixture's default action result carries no control group (the
    # switch defaults off); the section must not appear at all.
    result = make_result("action")
    assert result.control_group_summary is None
    st = RecordingStreamlit()

    render_trading_pnl(st, result)

    headings = [args[0] for args, _ in calls_for(st, "markdown")]
    assert not any("Control group" in heading for heading in headings)


def test_control_group_section_shown_when_present() -> None:
    result = dataclasses.replace(
        make_result("action"), control_group_summary=_control_group_frame()
    )
    st = RecordingStreamlit()

    render_trading_pnl(st, result)

    headings = [args[0] for args, _ in calls_for(st, "markdown")]
    assert any("Control group" in heading for heading in headings)
    tables = [args[0] for args, _ in calls_for(st, "dataframe")]
    control_table = tables[-1]
    assert list(control_table["Night"]) == ["Night 0", "Night 1", "Week"]
    captions = [args[0] for args, _ in calls_for(st, "caption")]
    assert any("10 control EVs of 100" in caption for caption in captions)


def test_control_group_table_reads_p50_per_night_and_week() -> None:
    table = _control_group_table(_control_group_frame())

    assert list(table["Night"]) == ["Night 0", "Night 1", "Week"]
    assert table.loc[0, "Estimated bias (kW/EV)"] == "0.100"
    assert table.loc[0, "True bias (kW/EV)"] == "0.200"
    assert table.loc[0, "Estimation error (kW/EV)"] == "0.300"


def test_control_group_caption_reports_the_split() -> None:
    caption = _control_group_caption(_control_group_frame())
    assert caption == "10 control EVs of 100 (10%), held out from every plan."


def test_view_makes_no_model_calls() -> None:
    source = Path(trading_pnl.__file__).read_text(encoding="utf-8")
    assert "axle_studio.model" not in source
    assert "run_forecast" not in source


def test_every_model_check_id_has_a_human_label() -> None:
    # The dispatch check was missing and showed as "dispatch split sums to
    # intraday"; read the ids from the model's own source so a new check fails here.
    source = Path(summaries.__file__).read_text()
    check_ids = set(re.findall(r'"check_id": "([a-z0-9_]+)"', source))
    assert "dispatch_split_sums_to_intraday" in check_ids
    assert check_ids <= set(trading_pnl._CHECK_LABELS)
