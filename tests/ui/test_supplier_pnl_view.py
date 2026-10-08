"""Supplier ▸ 3 Supplier P&L on the SYNTHETIC FIXTURE (supplier contract v1 §11, decision 0006)."""

from __future__ import annotations

import dataclasses
import math
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fixtures.result_fixture import make_result

from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.ui.views import supplier_pnl
from axle_studio.ui.views.supplier_common import season_text

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from dashboard_lint import RecordingStreamlit  # noqa: E402


@pytest.fixture(scope="module")
def result():
    return make_result("action", evs=12, worlds=6, seed=42)


@pytest.fixture(scope="module")
def real_result():
    """A small REAL run (mirrors test_supplier_firm_mw_view.py): unlike ``result``,
    it carries the firm-MW product-sheet frame ``_firm_tile`` reads, so the tile's
    text can be checked (B-2). The synthetic fixture predates that frame."""

    return run_forecast_from_assumptions(
        date(2026, 1, 12), model="action", values={"vehicle_count": 30, "evaluation_world_count": 6}
    )


def _with_pnl(result, **means: float):
    """The result with some ``profiled`` summary rows' statistics replaced (one scenario)."""

    summary = result.supplier_pnl_summary.copy()
    for metric, value in means.items():
        mask = summary["metric"].eq(metric)
        summary.loc[mask, ["mean", "p10", "p50", "p90"]] = value
    return dataclasses.replace(result, supplier_pnl_summary=summary)


def _texts(st: RecordingStreamlit) -> str:
    return "\n".join(
        str(arg) for name, args, _ in st.calls if name in ("markdown", "caption") for arg in args
    )


def test_waterfall_uses_the_stored_means_and_ends_at_the_stored_net(result) -> None:
    rows = supplier_pnl.waterfall_rows(result, "profiled")
    summary = result.supplier_pnl_summary.set_index(["hedge_variant", "metric"])

    assert list(rows["Bar"]) == [
        "Energy saving at day-ahead",
        "Hedge-error saving",
        "Grid-event payments",
        "Customer payments",
        "Supplier gain before fee",
    ]
    total = rows.iloc[-1]
    assert total["Measure"] == "total"
    assert (
        total["Mean (£ per week)"] == summary.loc[("profiled", "net_gain_before_fee_gbp"), "mean"]
    )
    # Decision 0006: the four component means add to the before-fee net mean.
    assert math.isclose(rows["Mean (£ per week)"].iloc[:-1].sum(), total["Mean (£ per week)"])


def test_unset_fee_draws_no_fee_bar_and_shows_unavailable_unset(result) -> None:
    rows = supplier_pnl.waterfall_rows(result, "profiled")
    assert "Platform fee" not in set(rows["Bar"])

    st = RecordingStreamlit()
    supplier_pnl.render_supplier_pnl(st, result)
    text = _texts(st)
    assert "Unavailable (unset)" in text
    assert "Platform fee not set" in text


def test_set_fee_adds_the_fee_and_the_after_fee_total(result) -> None:
    changed = _with_pnl(
        result,
        platform_fee_gbp=-2.0,
        net_gain_gbp=24.0,
        platform_fee_per_customer_per_month_gbp=-1.0,
        net_gain_per_customer_per_month_gbp=8.0,
    )
    rows = supplier_pnl.waterfall_rows(changed, "profiled")
    assert list(rows["Bar"])[-2:] == ["Platform fee", "Supplier gain after fee"]
    assert rows.iloc[-1]["Mean (£ per week)"] == 24.0


def test_unset_flat_reward_leaves_no_net_rather_than_zero(result) -> None:
    changed = _with_pnl(
        result,
        customer_payment_gbp=math.nan,
        net_gain_before_fee_gbp=math.nan,
        net_gain_gbp=math.nan,
    )
    rows = supplier_pnl.waterfall_rows(changed, "profiled")
    assert "Customer payments" not in set(rows["Bar"])
    assert "Supplier gain before fee" not in set(rows["Bar"])


def test_trading_net_is_beside_and_never_added(result) -> None:
    st = RecordingStreamlit()
    supplier_pnl.render_supplier_pnl(st, result)
    text = _texts(st)
    assert "Trading net, not added" in text
    assert "beside, never added" in text


def test_hedge_error_caption_absent_when_the_fixture_never_dispatched(result) -> None:
    # Intraday dispatch contract v1 §4.4, lead note at integration: the
    # SYNTHETIC fixture carries no ``dispatch_world_slot`` (no run ever
    # dispatched), so the hedge-error caption must not appear.
    st = RecordingStreamlit()
    supplier_pnl.render_supplier_pnl(st, result)
    text = _texts(st)
    assert "intraday dispatch" not in text


def test_hedge_error_caption_present_on_a_real_dispatched_run() -> None:
    dispatched = run_forecast_from_assumptions(
        date(2026, 10, 12),
        model="action",
        values={"vehicle_count": 60, "evaluation_world_count": 6},
    )
    assert dispatched.dispatch_world_slot is not None
    st = RecordingStreamlit()
    supplier_pnl.render_supplier_pnl(st, dispatched)
    captions = _main_captions(st)
    assert any(
        "Includes the intraday dispatch's moves, priced at imbalance minus day-ahead." == c
        for c in captions
    )


def test_flat_variant_reads_its_own_rows(result) -> None:
    st = RecordingStreamlit(choices={"supplier-hedge-variant": "flat"})
    supplier_pnl.render_supplier_pnl(st, result)
    titles = [args[0] for name, args, _ in st.calls if name == "markdown"]
    assert any("flat hedge" in str(title) for title in titles)


def test_firm_tile_names_p10_and_p05_never_a_traders_p90(real_result) -> None:
    # B-2: the across-weeks P10 was called "P90" here while Firm MW's own
    # table called the same figure "P10" (ruling "firm = across-weeks P10").
    value, context = supplier_pnl._firm_tile(real_result)
    assert value != supplier_pnl.UNAVAILABLE
    assert "P10 (90% exceedance)" in context
    assert "P05 (95%)" in context
    assert "P90" not in context
    assert "P95" not in context


def test_firm_tile_names_the_runs_own_season_months(real_result) -> None:
    # Clarity critique: the tile hard-coded "(Apr-Sep)" regardless of season,
    # wrong on this fixture's January (winter) start. Fixed to derive the
    # month range from the run's own season, as season_text does below.
    season = str(real_result.supplier_pnl_summary["season"].iloc[0])
    assert season == "winter"
    _, context = supplier_pnl._firm_tile(real_result)
    assert "(Oct-Mar)" in context
    assert "(Apr-Sep)" not in context


def test_revenue_by_segment_chart_uses_horizontal_bars_for_readability(result) -> None:
    # B-15: ten bucket names collided when rotated on the x axis at 390 px;
    # horizontal bars put them on the y axis instead, unrotated.
    st = RecordingStreamlit()
    supplier_pnl.render_supplier_pnl(st, result)
    figure = dict(st.figures())["supplier-revenue-segments"]
    assert figure.data
    for trace in figure.data:
        assert trace.orientation == "h"
        assert set(trace.y).issubset(set(supplier_pnl.BUCKET_LABELS.values()))


def test_payment_mode_text_follows_the_run_records(result) -> None:
    # The fixture carries no assumption records: the mode reads as revenue share.
    assert supplier_pnl.payment_mode_text(result).endswith("of the week's positive gain")


def test_every_money_caption_is_short_and_says_illustrative_somewhere(result) -> None:
    st = RecordingStreamlit()
    supplier_pnl.render_supplier_pnl(st, result)
    captions = _main_captions(st)
    assert captions and all(len(caption) <= 140 for caption in captions)
    assert any("Illustrative" in caption for caption in captions)


def _main_captions(st: RecordingStreamlit) -> list[str]:
    """Captions on the main surface: a chart's definition (the caption right after its
    data table, inside the "Data and definition" expander) is reference text, exempt from
    the 140-character rule like the copy-rules test."""

    captions, previous = [], None
    for name, args, _ in st.calls:
        if name == "caption" and previous != "dataframe":
            captions.append(str(args[0]))
        previous = name
    return captions


_CITATION = re.compile(r"contract v\d|§|\bQ-\d|decision \d{4}|\baudit O\d|\bplan C\d")


def test_tooltips_and_definitions_carry_no_internal_citations(result) -> None:
    # Voice pass: a tooltip or definition says the rule in words; "supplier
    # contract v1 §3.1, decision 0006" tells an Axle reader nothing.
    st = RecordingStreamlit()
    supplier_pnl.render_supplier_pnl(st, result)
    texts = [
        text
        for _, args, kwargs in st.calls
        for text in (*args, kwargs.get("help", ""))
        if isinstance(text, str)
    ]
    assert [text for text in texts if _CITATION.search(text)] == []


def test_season_text_names_the_months_it_means() -> None:
    # A bare "summer run" read as wrong on a run starting 29 September.
    assert season_text(pd.DataFrame({"season": ["summer"]})) == (
        "summer demand shapes (Apr–Sep start)"
    )
    assert season_text(pd.DataFrame({"season": ["winter"]})) == (
        "winter demand shapes (Oct–Mar start)"
    )
    assert season_text(None) == ""
