"""Partners: the five persona lenses on the SYNTHETIC FIXTURE.

Covers: each persona renders its hero and supporting charts and a
"What they would ask" table; every drill-through button targets a real
(page, lens) pair in the registry; the Supplier ▸ 5 rename to "Charger
makers" is reflected in the registry and the dispatch table.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from fixtures.household_contract import make_household_card
from fixtures.result_fixture import make_result

from axle_studio.ui import pages
from axle_studio.ui.registry import PAGES
from axle_studio.ui.views import partner_personas

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from dashboard_lint import RecordingStreamlit  # noqa: E402

PERSONA_RENDERERS = {
    "Driver": partner_personas.render_driver,
    "Energy supplier": partner_personas.render_energy_supplier,
    "Charger maker": partner_personas.render_charger_maker,
    "Carmaker": partner_personas.render_carmaker,
}

EXPECTED_CHART_KEYS = {
    "Driver": {
        "partners-persona-driver-value-year",
        "partners-persona-driver-cost-strip",
        "partners-persona-driver-departure-soc",
        "partners-persona-driver-soc",
        "partners-persona-driver-average-day",
    },
    "Energy supplier": {
        "partners-persona-supplier-waterfall",
        "partners-persona-supplier-paths",
        "partners-persona-supplier-blocks",
        "partners-persona-supplier-cost-curve",
        # The day-ahead, reliability and fleet-size charts need the firm-MW
        # frames, which this SYNTHETIC FIXTURE predates (like Supplier ▸ 4
        # Firm MW's own fixture-based tests); their pending fallback is
        # checked separately below, not their chart keys.
    },
    "Charger maker": {
        "partners-persona-charger-maker-market",
        "partners-persona-charger-maker-exceedance",
        "partners-persona-charger-maker-funnel",
        "partners-persona-charger-maker-payout",
    },
    "Carmaker": {
        "partners-persona-carmaker-dwell",
        "partners-persona-carmaker-heatmap",
        "partners-persona-carmaker-departure-soc",
        "partners-persona-carmaker-hour",
        # The flexible-kW-by-archetype panel needs firm-MW-derived household
        # columns the fixture does not carry (customers.py's own
        # _flexible_kw_chart shows the same "Available after..." fallback
        # on this fixture); checked separately below.
    },
}


@pytest.fixture(scope="module")
def result():
    return make_result("action", evs=12, worlds=6, seed=42)


def _figure_keys(st: RecordingStreamlit) -> set[str]:
    return {key for key, _ in st.figures() if key is not None}


def _dataframes(st: RecordingStreamlit) -> list:
    return [args[0] for name, args, _ in st.calls if name == "dataframe" and args]


_WANTS_COLUMNS = ["Want", "Reading", "Where"]


def _wants_tables(st: RecordingStreamlit) -> list:
    return [frame for frame in _dataframes(st) if list(frame.columns) == _WANTS_COLUMNS]


def _button_targets(st: RecordingStreamlit) -> list[tuple[str, str]]:
    return [
        kwargs["args"]
        for name, _args, kwargs in st.calls
        if name == "button" and kwargs.get("on_click") is partner_personas._switch_to
    ]


@pytest.mark.parametrize("persona", sorted(PERSONA_RENDERERS))
def test_persona_renders_its_charts_and_wants_table(result, persona: str) -> None:
    st = RecordingStreamlit()
    PERSONA_RENDERERS[persona](st, result)

    assert EXPECTED_CHART_KEYS[persona] <= _figure_keys(st)

    wants_tables = _wants_tables(st)
    assert len(wants_tables) == 1, f"{persona}: expected exactly one wants table"
    table = wants_tables[0]
    assert len(table) >= 4
    # Every reading is a real value, "Unavailable" or "Not modelled" -- never blank.
    assert table["Reading"].apply(lambda value: bool(str(value).strip())).all()
    # "Where" always looks like a screen name ("Page" or "Page ▸ Lens"), never a
    # snake_case frame or column name.
    assert table["Where"].apply(lambda where: "_" not in where).all()


def test_fleet_and_leasing_renders_with_the_injected_household_card(result) -> None:
    st = RecordingStreamlit()
    partner_personas.render_fleet_and_leasing(st, result, household_card=make_household_card)

    keys = _figure_keys(st)
    assert {
        "partners-persona-fleet-outcomes",
        "partners-persona-fleet-flexible-kwh",
        "partners-persona-fleet-cost-strip",
    } <= keys
    assert len(_wants_tables(st)) == 1


def test_ev_only_caption_appears_once_per_persona(result) -> None:
    for render in [*PERSONA_RENDERERS.values()]:
        st = RecordingStreamlit()
        render(st, result)
        captions = [str(args[0]) for name, args, _ in st.calls if name == "caption"]
        assert captions.count(partner_personas.EV_ONLY_CAPTION) == 1


def test_energy_supplier_names_the_firm_mw_gap_on_the_fixture(result) -> None:
    # The SYNTHETIC FIXTURE predates the firm-MW frames (no
    # availability_bands, product_sheet, ...), the same gap Supplier ▸ 4
    # Firm MW itself is tested against; the day-ahead chart and the grid
    # expander must say so plainly rather than crash or draw nothing.
    assert not partner_personas._firm_mw_available(result)
    st = RecordingStreamlit()
    partner_personas.render_energy_supplier(st, result)
    captions = [str(args[0]) for name, args, _ in st.calls if name == "caption"]
    assert captions.count(partner_personas.FIRM_MW_PENDING) == 2  # day-ahead slot + expander


def test_carmaker_names_the_firm_mw_gap_on_the_fixture(result) -> None:
    st = RecordingStreamlit()
    partner_personas.render_carmaker(st, result)
    captions = [str(args[0]) for name, args, _ in st.calls if name == "caption"]
    assert partner_personas.FIRM_MW_PENDING in captions


def test_grid_route_expander_names_the_caveat(result) -> None:
    st = RecordingStreamlit()
    partner_personas.render_energy_supplier(st, result)

    expander_labels = [args[0] for name, args, _ in st.calls if name == "expander"]
    assert "Grid route: firm MW" in expander_labels
    captions = [str(args[0]) for name, args, _ in st.calls if name == "caption"]
    assert "Illustrative; not a registered product." in captions


def test_fleet_and_leasing_states_the_household_caveat(result) -> None:
    st = RecordingStreamlit()
    partner_personas.render_fleet_and_leasing(st, result, household_card=make_household_card)

    captions = [str(args[0]) for name, args, _ in st.calls if name == "caption"]
    assert any("no depot" in caption for caption in captions)


_CITATION = re.compile(r"contract v\d|§|\bQ-\d|decision \d{4}|\baudit O\d|\bplan C\d")


def test_tooltips_and_definitions_carry_no_internal_citations(result) -> None:
    # Voice pass: a tooltip or definition says the rule in words, never a
    # contract section, decision number or review item.
    renders = [*PERSONA_RENDERERS.values()]
    for render in renders:
        st = RecordingStreamlit()
        render(st, result)
        _assert_no_citation(st)
    st = RecordingStreamlit()
    partner_personas.render_fleet_and_leasing(st, result, household_card=make_household_card)
    _assert_no_citation(st)


def _assert_no_citation(st: RecordingStreamlit) -> None:
    texts = [
        text
        for _, args, kwargs in st.calls
        for text in (*args, kwargs.get("help", ""))
        if isinstance(text, str)
    ]
    assert [text for text in texts if _CITATION.search(text)] == []


# --- Drill-through targets and the rename --------------------------------------


_ALL_LENS_PAIRS = {(page.name, lens.name) for page in PAGES for lens in page.lenses}


@pytest.mark.parametrize("persona", sorted(PERSONA_RENDERERS))
def test_drill_through_targets_are_real_lenses(result, persona: str) -> None:
    st = RecordingStreamlit()
    PERSONA_RENDERERS[persona](st, result)

    targets = _button_targets(st)
    assert targets, f"{persona}: no drill-through buttons recorded"
    for page_name, lens_name in targets:
        assert (page_name, lens_name) in _ALL_LENS_PAIRS, (page_name, lens_name)


def test_fleet_and_leasing_drill_through_targets_are_real_lenses(result) -> None:
    st = RecordingStreamlit()
    partner_personas.render_fleet_and_leasing(st, result, household_card=make_household_card)

    targets = _button_targets(st)
    assert targets
    for page_name, lens_name in targets:
        assert (page_name, lens_name) in _ALL_LENS_PAIRS, (page_name, lens_name)


def test_supplier_five_is_renamed_charger_makers() -> None:
    supplier = next(page for page in PAGES if page.name == "Supplier")
    lens_names = [lens.name for lens in supplier.lenses]
    assert "5 Charger makers" in lens_names
    assert "5 Partners" not in lens_names
    assert ("Supplier", "5 Charger makers") in pages.VIEWS
    assert ("Supplier", "5 Partners") not in pages.VIEWS


def test_partners_page_sits_after_supplier_and_before_replay() -> None:
    names = [page.name for page in PAGES]
    assert names.index("Supplier") < names.index("Partners") < names.index("Replay")


def test_partners_page_has_the_five_personas_in_order() -> None:
    partners_page = next(page for page in PAGES if page.name == "Partners")
    assert [lens.name for lens in partners_page.lenses] == [
        "Driver",
        "Energy supplier",
        "Charger maker",
        "Carmaker",
        "Fleet and leasing",
    ]
