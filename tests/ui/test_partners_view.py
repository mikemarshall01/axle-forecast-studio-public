"""Supplier ▸ 5 Partners on the SYNTHETIC FIXTURE (supplier contract v1 §11, §5)."""

from __future__ import annotations

import dataclasses
import re
import sys
from datetime import date
from pathlib import Path

import pytest
from fixtures.result_fixture import make_result

from axle_studio.model.assumptions import Assumption
from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.ui.views import partners

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from dashboard_lint import RecordingStreamlit, lint_figure  # noqa: E402

_START = date(2026, 1, 12)  # a Monday, no clock change


@pytest.fixture(scope="module")
def result():
    return make_result("action", evs=12, worlds=6, seed=42)


@pytest.fixture(scope="module")
def timed_result():
    """A small real run with the optional timed path on (decision 0007): the
    SYNTHETIC fixture above stays two-policy by design, so the departure-SoC
    chart's Timed tariff panel is checked against a real run instead."""

    return run_forecast_from_assumptions(
        _START,
        model="action",
        values={"vehicle_count": 30, "evaluation_world_count": 3, "timed_start_local_hour": 0.0},
    )


def _figures(st: RecordingStreamlit) -> dict:
    return dict(st.figures())


def _kpi_html(st: RecordingStreamlit) -> str:
    return "\n".join(
        str(args[0]) for name, args, _ in st.calls if name == "markdown" and "axle-kpi" in str(args)
    )


def test_renders_every_block_for_the_fleet(result) -> None:
    st = RecordingStreamlit()
    partners.render_partners(st, result)
    keys = set(_figures(st))
    assert {
        "partners-markets",
        "partners-exceedance",
        "partners-departure-soc",
        "partners-funnel",
        "partners-tornado",
        "partners-payout-fan",
    } <= keys


def test_headline_tile_is_the_fleet_net_not_the_group_gross(result) -> None:
    # B-12: the headline showed gross cash where a reader expects what a
    # device earns (~£15 net vs ~£33 gross in the critique's run); net
    # (after every deduction) headlines now, gross beside it, labelled.
    st = RecordingStreamlit()
    partners.render_partners(st, result)
    html = _kpi_html(st)
    assert "Net cash per device" in html
    assert "Gross cash, this group" in html

    net_row = partners.stat_row(
        result.revenue_by_market_summary, market="net", metric="gbp_per_enrolled_device_per_month"
    )
    assert partners.p50_text(net_row, partners._pounds2) in html


def test_unmodelled_markets_are_named_never_zero(result) -> None:
    st = RecordingStreamlit()
    partners.render_partners(st, result)
    figure = _figures(st)["partners-markets"]
    drawn = set(figure.data[0].y)
    assert "Balancing Mechanism" not in drawn
    captions = " ".join(str(args[0]) for name, args, _ in st.calls if name == "caption")
    assert "Not modelled, not £0: Balancing Mechanism" in captions


def test_threshold_moves_only_the_marker_and_the_two_tiles(result) -> None:
    exceedance = result.household_value_exceedance
    stored = exceedance.loc[
        exceedance["group_id"].eq("fleet")
        & exceedance["metric"].eq("floor_top_up_gbp_per_device_per_month")
        & exceedance["threshold_gbp_per_month"].eq(40)
    ].iloc[0]
    st = RecordingStreamlit(choices={"partners-threshold": 40})
    partners.render_partners(st, result)
    figure = _figures(st)["partners-exceedance"]
    assert figure.layout.shapes[0].x0 == 40
    assert partners.money(stored["p50"], decimals=2, signed=True) in _kpi_html(st)


def test_platform_fee_never_enters_partner_revenue(result) -> None:
    # Final critique Q-8: the supplier's platform fee is a supplier cost to
    # the platform (decision 0006), income to nobody here. Setting it must
    # not move the growth calculator's figures or add a fee lever, or it
    # would read as the device maker's income (payer and payee swapped).
    st_unset = RecordingStreamlit()
    partners.render_partners(st_unset, result)
    tornado_unset = _figures(st_unset)["partners-tornado"]
    assert len(tornado_unset.data[0].y) == 3  # the fee row is always left out

    fee_record = Assumption(
        name="supplier.platform_fee_gbp_per_ev_per_month",
        label="Platform fee",
        value=5.0,
        unit="GBP/EV/month",
        evidence="illustrative",
        source="test",
        meaning="test",
        editable=True,
        bounds=(0.0, 100.0),
        group="Supplier",
        affects="test",
    )
    with_fee = dataclasses.replace(result, assumptions=(*result.assumptions, fee_record))
    st_set = RecordingStreamlit()
    partners.render_partners(st_set, with_fee)
    tornado_set = _figures(st_set)["partners-tornado"]
    assert len(tornado_set.data[0].y) == 3
    assert list(tornado_set.data[0].x) == pytest.approx(list(tornado_unset.data[0].x))

    badges = [str(args[1]) for name, args, _ in st_set.calls if name == "badge_line"]
    assert not any("fee" in badge.lower() for badge in badges)


def test_payout_fan_flips_the_payment_sign_and_scales_by_month(result) -> None:
    summary = result.supplier_pnl_summary
    payment = summary.loc[
        summary["hedge_variant"].eq("profiled")
        & summary["metric"].eq("customer_payment_per_customer_per_month_gbp")
    ].iloc[0]
    st = RecordingStreamlit()
    partners.render_partners(st, result)
    fan = _figures(st)["partners-payout-fan"]
    per_customer_mid = fan.data[2]  # band low, band high, then the P50 line
    assert list(per_customer_mid.x) == list(range(1, 13))
    assert per_customer_mid.y[-1] == pytest.approx(-12 * payment["p50"])
    low, high = fan.data[0].y, fan.data[1].y
    assert low[0] == pytest.approx(-payment["p90"]) and high[0] == pytest.approx(-payment["p10"])


def test_every_archetype_group_renders(result) -> None:
    for group in result.partner_summary["group_id"].unique():
        st = RecordingStreamlit(choices={partners.GROUP_KEY: group})
        partners.render_partners(st, result)
        assert "partners-exceedance" in _figures(st)


# --- Departure SoC: the optional timed path (decision 0007) ------------------


def test_departure_chart_stays_two_policy_without_the_timed_path(result) -> None:
    # Absent (the SYNTHETIC fixture has no timed path): the title, caption
    # and panel count are exactly as before this change.
    st = RecordingStreamlit()
    partners.render_partners(st, result)
    figure = _figures(st)["partners-departure-soc"]
    assert [ann.text for ann in figure.layout.annotations] == ["Unmanaged", "Smart"]
    title = next(
        str(args[0])
        for name, args, _ in st.calls
        if name == "markdown" and "SoC at departure" in str(args[0])
    )
    assert "both paths" in title
    caption = next(c for c in _main_captions(st) if c.startswith("Early departures"))
    assert "tariff rate" not in caption


def test_departure_chart_gains_a_timed_tariff_panel(timed_result) -> None:
    st = RecordingStreamlit()
    partners.render_partners(st, timed_result)
    figure = _figures(st)["partners-departure-soc"]
    assert [ann.text for ann in figure.layout.annotations] == ["Unmanaged", "Timed tariff", "Smart"]
    bar_names = [trace.name for trace in figure.data]
    assert bar_names == ["Unmanaged", "Timed tariff", "Smart"]
    title = next(
        str(args[0])
        for name, args, _ in st.calls
        if name == "markdown" and "SoC at departure" in str(args[0])
    )
    assert "Unmanaged vs Timed tariff vs Smart" in title
    caption = next(c for c in _main_captions(st) if c.startswith("Early departures"))
    assert "no tariff rate" in caption
    assert len(caption) <= 140
    table = next(
        args[0]
        for name, args, _ in st.calls
        if name == "dataframe"
        and "Path" in list(args[0].columns)
        and "Bin" in list(args[0].columns)
    )
    assert set(table["Path"].unique()) == {"Unmanaged", "Timed tariff", "Smart"}
    assert lint_figure(figure) == []


def test_captions_are_short(result) -> None:
    st = RecordingStreamlit()
    partners.render_partners(st, result)
    captions = _main_captions(st)
    assert all(len(caption) <= 140 for caption in captions), captions


def test_tornado_caption_stays_short_when_take_clips_too(result) -> None:
    # The "take clipped" text and the always-on "no fee lever" text (Q-8:
    # the fee is never fed in here) can both apply to one caption at once.
    st = RecordingStreamlit(choices={"partners-take": 90.0})
    partners.render_partners(st, result)
    tornado_caption = next(c for c in _main_captions(st) if "lever moved alone" in c)
    assert "Take clipped" in tornado_caption
    assert "No fee lever" in tornado_caption
    assert len(tornado_caption) <= 140


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
    # Voice pass: a tooltip or definition says the rule in words; "Q-8" or
    # "decision 0006's before-fee headline" tells an Axle reader nothing.
    st = RecordingStreamlit()
    partners.render_partners(st, result)
    texts = [
        text
        for _, args, kwargs in st.calls
        for text in (*args, kwargs.get("help", ""))
        if isinstance(text, str)
    ]
    assert [text for text in texts if _CITATION.search(text)] == []
