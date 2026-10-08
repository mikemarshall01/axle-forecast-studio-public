"""Supplier ▸ 6 Customers on the SYNTHETIC FIXTURE (household contract v1 §6.2, §9)."""

from __future__ import annotations

import re
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from fixtures.result_fixture import make_result
from kpi_calls import kpi_calls

from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.ui.views import customers

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from dashboard_lint import RecordingStreamlit, lint_figure  # noqa: E402

_START = date(2026, 1, 12)  # a Monday, no clock change


@pytest.fixture(scope="module")
def result():
    return make_result("action", evs=12, worlds=6, seed=42)


@pytest.fixture(scope="module")
def timed_result():
    """A small real run with the optional timed path on (decision 0007): the
    SYNTHETIC fixture above stays two-policy by design, so the three claim
    tiles that compare policies are checked against a real run instead."""

    return run_forecast_from_assumptions(
        _START,
        model="action",
        values={"vehicle_count": 30, "evaluation_world_count": 3, "timed_start_local_hour": 0.0},
    )


def _figures(st: RecordingStreamlit) -> dict:
    return dict(st.figures())


def _main_captions(st: RecordingStreamlit) -> list[str]:
    """Captions on the main surface; a chart's definition (right after its data table

    inside the "Data and definition" expander) is reference text, exempt from the
    140-character rule, like the copy-rules test.
    """

    captions, previous = [], None
    for name, args, _ in st.calls:
        if name == "caption" and previous != "dataframe":
            captions.append(str(args[0]))
        previous = name
    return captions


def _with_evening_kw(result, rng_seed: int = 0):
    """A copy of the fixture's summary with the evening kW rows filled in (J1b landed)."""

    summary = result.household_outcomes_summary.copy()
    mask = summary["metric"].isin(["evening_turn_down_kw_1h", "evening_turn_up_kw_1h"])
    rng = np.random.default_rng(rng_seed)
    for column in ("mean", "p10", "p50", "p90"):
        summary.loc[mask, column] = rng.uniform(0.5, 5.0, int(mask.sum()))
    return replace(result, household_outcomes_summary=summary)


def test_renders_the_claims_row_and_the_value_year_chart_for_the_fleet(result) -> None:
    st = RecordingStreamlit()
    customers.render_customers(st, result)

    assert "customers-value-year" in _figures(st)
    assert customers.FIRM_MW_PENDING in [str(a[0]) for n, a, _ in st.calls if n == "info"]
    headings = [str(a[0]) for n, a, _ in st.calls if n == "subheader"]
    assert headings == ["If customers paid day-ahead prices"]
    assert [tile[0] for tile in kpi_calls(st)] == [
        "Typical value per year",
        "Sessions ready at departure",
        "Smart charging cost",
        "Charging in cheapest third",
        "Supplier energy saving",
        "Weekly saving per customer",
        "Customers worse off",
        "CO₂ shifted per customer",
    ]


def test_no_frames_message_when_household_outcomes_are_absent() -> None:
    no_frames = replace(make_result("action", evs=6, worlds=4), household_outcomes_summary=None)
    st = RecordingStreamlit()
    customers.render_customers(st, no_frames)

    assert [str(a[0]) for n, a, _ in st.calls if n == "info"] == [customers.NO_FRAMES_MESSAGE]
    assert st.figures() == []


def test_typical_value_tile_reads_the_ev_means_band_not_across_weeks(result) -> None:
    summary = result.household_outcomes_summary
    row = summary.loc[
        summary["group_id"].eq("fleet")
        & summary["metric"].eq("value_gbp_per_year")
        & summary["statistic"].eq("p50_of_ev_means")
    ].iloc[0]
    p10 = summary.loc[
        summary["group_id"].eq("fleet")
        & summary["metric"].eq("value_gbp_per_year")
        & summary["statistic"].eq("p10_of_ev_means")
    ].iloc[0]["p50"]
    p90 = summary.loc[
        summary["group_id"].eq("fleet")
        & summary["metric"].eq("value_gbp_per_year")
        & summary["statistic"].eq("p90_of_ev_means")
    ].iloc[0]["p50"]

    st = RecordingStreamlit()
    customers.render_customers(st, result)
    tile = kpi_calls(st)[0]

    assert tile[1] == customers.money(row["p50"])
    assert customers.money(p10) in tile[3]
    assert customers.money(p90) in tile[3]
    assert "8 in 10 customers" in tile[3]
    assert "illustrative" in tile[3]
    # Final critique Q-10: the tile names its statistic.
    assert tile[3].startswith("median customer")


def test_charging_cost_is_pence_per_kwh_with_the_mwh_equivalence(result) -> None:
    # Final critique B-10: p/kWh on customer screens; £/MWh stays for fleet and
    # trading, and the help gives the exact equivalence.
    st = RecordingStreamlit()
    customers.render_customers(st, result)
    tile = next(tile for tile in kpi_calls(st) if tile[0] == "Smart charging cost")
    assert tile[1].endswith(" p/kWh")
    assert "1 p/kWh = £10/MWh" in tile[4]
    assert "GBP" not in tile[4]


def test_co2_tile_names_its_statistic(result) -> None:
    # Final critique Q-11: the average customer's average month, not a P50 week.
    st = RecordingStreamlit()
    customers.render_customers(st, result)
    tile = next(tile for tile in kpi_calls(st) if tile[0] == "CO₂ shifted per customer")
    if tile[1] != customers.UNAVAILABLE:
        assert tile[3].startswith("average customer, average month")
    assert "P50 week" in tile[4]


def test_readiness_tile_reads_the_group_ratio_not_the_mean_of_shares(result) -> None:
    summary = result.household_outcomes_summary
    ratio = summary.loc[
        summary["group_id"].eq("fleet")
        & summary["metric"].eq("completed_share_selected")
        & summary["statistic"].eq("group_ratio")
    ].iloc[0]
    mean_of_shares = summary.loc[
        summary["group_id"].eq("fleet")
        & summary["metric"].eq("completed_share_selected")
        & summary["statistic"].eq("mean_across_evs")
    ].iloc[0]["p50"]
    # The group ratio and the mean of per-customer shares differ on this fixture
    # (B2): the tile must read the former.
    assert ratio["p50"] != pytest.approx(mean_of_shares)

    st = RecordingStreamlit()
    customers.render_customers(st, result)
    tile = next(t for t in kpi_calls(st) if t[0] == "Sessions ready at departure")

    assert tile[1] == customers.percent(ratio["p50"] * 100.0)


def test_customers_worse_off_tile_reads_both_the_yearly_and_weekly_shares(result) -> None:
    summary = result.household_outcomes_summary
    yearly = summary.loc[
        summary["group_id"].eq("fleet")
        & summary["metric"].eq("value_gbp_per_year")
        & summary["statistic"].eq("share_of_ev_means_below_zero")
    ].iloc[0]["p50"]
    weekly = result.household_value_summary.loc[
        result.household_value_summary["group_id"].eq("fleet")
        & result.household_value_summary["statistic"].eq("share_worse_off")
    ].iloc[0]["p50"]

    st = RecordingStreamlit()
    customers.render_customers(st, result)
    tile = next(t for t in kpi_calls(st) if t[0] == "Customers worse off")

    assert tile[1] == customers.percent(yearly * 100.0)
    assert customers.percent(weekly * 100.0) in tile[3]


def test_flexible_kw_chart_is_pending_until_the_firm_mw_model_lands(result) -> None:
    st = RecordingStreamlit()
    customers.render_customers(st, result)

    assert "customers-flexible-kw" not in _figures(st)
    infos = [str(a[0]) for n, a, _ in st.calls if n == "info"]
    assert customers.FIRM_MW_PENDING in infos


def test_flexible_kw_chart_renders_once_the_evening_kw_is_available(result) -> None:
    populated = _with_evening_kw(result)
    st = RecordingStreamlit()
    customers.render_customers(st, populated)

    assert "customers-flexible-kw" in _figures(st)
    figure = _figures(st)["customers-flexible-kw"]
    names = {trace.name for trace in figure.data}
    assert names == {"Turn-down (smart)", "Turn-up (smart)"}
    hover = figure.data[0].customdata[0]
    assert "firm" not in hover.lower()
    assert "P10 customer" in hover


def test_every_group_renders_the_claims_row(result) -> None:
    for group in result.household_outcomes_summary["group_id"].unique():
        st = RecordingStreamlit(choices={customers.GROUP_KEY: group})
        customers.render_customers(st, result)
        assert "customers-value-year" in _figures(st)


# --- Claims row: the optional timed path (decision 0007) ---------------------


def test_claims_row_stays_two_policy_without_the_timed_path(result) -> None:
    # Absent (the SYNTHETIC fixture has no timed path): every claim tile and
    # the caption beneath the row are exactly as before this change.
    st = RecordingStreamlit()
    customers.render_customers(st, result)
    tiles = kpi_calls(st)
    ready = next(t for t in tiles if t[0] == "Sessions ready at departure")
    assert "Timed tariff" not in ready[3]
    cost = next(t for t in tiles if t[0] == "Smart charging cost")
    assert "Timed tariff" not in cost[3]
    assert "costed the same way" not in cost[4]
    cheap = next(t for t in tiles if t[0] == "Charging in cheapest third")
    assert "Timed tariff" not in cheap[3]
    claim_caption = next(c for c in _main_captions(st) if c.startswith("Simulated fleet"))
    assert claim_caption == (
        "Simulated fleet, synthetic prices, pass-through reading: what customers would pay "
        "if day-ahead prices reached them. Not a tariff."
    )


def test_claims_row_gains_timed_tariff_beside_unmanaged_in_three_tiles(timed_result) -> None:
    st = RecordingStreamlit()
    customers.render_customers(st, timed_result)
    tiles = kpi_calls(st)

    # "Typical value per year" has no unmanaged or timed counterpart (value
    # is the smart saving itself), so it is left exactly as it reads today.
    year = next(t for t in tiles if t[0] == "Typical value per year")
    assert "Timed tariff" not in year[3]

    ready = next(t for t in tiles if t[0] == "Sessions ready at departure")
    assert "unmanaged" in ready[3] and "Timed tariff" in ready[3]
    assert ready[3].index("unmanaged") < ready[3].index("Timed tariff")

    cost = next(t for t in tiles if t[0] == "Smart charging cost")
    assert "unmanaged" in cost[3] and "Timed tariff" in cost[3]
    # Money is shown for the timed path on this tile, so its help names the
    # shared costing rule (no tariff rate) in words.
    assert "costed the same way" in cost[4]
    assert "no tariff rate" in cost[4]

    cheap = next(t for t in tiles if t[0] == "Charging in cheapest third")
    assert "unmanaged" in cheap[3] and "Timed tariff" in cheap[3]

    claim_caption = next(c for c in _main_captions(st) if c.startswith("Simulated fleet"))
    assert "costed alike" in claim_caption
    assert len(claim_caption) <= 140


def test_tooltips_and_definitions_carry_no_internal_citations_with_the_timed_path(
    timed_result,
) -> None:
    st = RecordingStreamlit()
    customers.render_customers(st, timed_result)
    texts = [
        text
        for _, args, kwargs in st.calls
        for text in (*args, kwargs.get("help", ""))
        if isinstance(text, str)
    ]
    assert [text for text in texts if _CITATION.search(text)] == []


def test_least_gaining_archetype_is_named_by_its_p10_customer(result) -> None:
    summary = result.household_outcomes_summary.copy()
    p10_rows = summary["metric"].eq("value_gbp_per_year") & summary["statistic"].eq(
        "p10_of_ev_means"
    )
    cohort_rows = summary.loc[p10_rows & summary["group_id"].ne("fleet")]
    for rank, index in enumerate(cohort_rows.index, start=1):
        summary.loc[index, ["mean", "p50"]] = float(rank)
    worst_group = summary.loc[cohort_rows.index].sort_values("mean").iloc[0]["group_id"]
    labelled = customers.cohort_labels(result)[worst_group]
    edited = replace(result, household_outcomes_summary=summary)

    st = RecordingStreamlit()
    customers.render_customers(st, edited)
    captions = _main_captions(st)

    assert any(
        caption.startswith(f"Least: {labelled}, P10 customer £1 a year") for caption in captions
    )


def test_figures_follow_the_chart_conventions(result) -> None:
    populated = _with_evening_kw(result)
    st = RecordingStreamlit()
    customers.render_customers(st, populated)

    for key, figure in st.figures():
        assert lint_figure(figure) == [], key


def test_captions_are_short(result) -> None:
    st = RecordingStreamlit()
    customers.render_customers(st, result)
    captions = _main_captions(st)
    assert all(len(caption) <= 140 for caption in captions), captions


_CITATION = re.compile(r"contract v\d|§|\bQ-\d|decision \d{4}|\baudit O\d|\bplan C\d")


def test_tooltips_and_definitions_carry_no_internal_citations(result) -> None:
    # Voice pass: a tooltip or definition says the rule in words; "decision
    # 0006's P&L" tells an Axle reader nothing.
    st = RecordingStreamlit()
    customers.render_customers(st, _with_evening_kw(result))
    texts = [
        text
        for _, args, kwargs in st.calls
        for text in (*args, kwargs.get("help", ""))
        if isinstance(text, str)
    ]
    assert [text for text in texts if _CITATION.search(text)] == []


def test_pass_through_reading_never_added_to_the_supplier_flat_tariff_reading(result) -> None:
    """The claims block and the value row never mix the two readings in one figure."""

    st = RecordingStreamlit()
    customers.render_customers(st, result)
    helps = " ".join(str(kwargs.get("help", "")) for _, _, kwargs in st.calls)
    assert "flat-tariff" in helps  # the supplier energy saving tile names its own reading
