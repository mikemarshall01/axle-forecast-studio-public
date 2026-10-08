"""Copy rules for on-screen text (polish plan G7), checked on the rendered app.

The rules: no internal citations ("decision 0004 item N", "contract v2 ...",
report page numbers) and no snake_case field names on screen; they belong
in ``help=`` tooltips or the "Data and definition" expander. Captions are at
most 140 characters, KPI labels at most 28 characters with no unit, at most
one "Finding:" caption per lens, the run identity at most 80 characters, and
the empty state quotes the draft's own size.

The app is run for real through AppTest (a small 30 EV x 4 week run, twice,
so Compare has a pair), and the rendered element tree is walked, so the test
reads what a viewer reads rather than the source. Tooltips are not scanned
(that is where citations go), nor are the "Data and definition" and "Read
the full explainer" expanders (reference text a viewer opens on purpose).
Plotly figure text (axis titles, legends) is outside this walk.

Scope: ``CHECKED_VIEWS`` are the views swept so far (all of them since the
Drivers/Overview branch merged); ``PENDING_VIEWS`` holds any new view until
it follows the rules. ``test_every_view_is_checked_or_pending`` keeps the two
lists covering ``pages.VIEWS`` exactly, so a new view cannot slip past both.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from axle_studio.model import assumptions
from axle_studio.ui import pages, run_controller
from axle_studio.ui.registry import PAGES

APP_PATH = Path(__file__).parents[2] / "streamlit_app.py"
SMALL_FLEET = 30
SMALL_WORLDS = 4

CHECKED_VIEWS: tuple[tuple[str, str | None], ...] = (
    ("Overview", "At a glance"),
    ("Overview", "Key stats"),
    ("Drivers", "Plug-ins"),
    ("Drivers", "Sessions"),
    ("Drivers", "One EV"),
    ("Drivers", "Household"),
    ("Drivers", "Archetypes"),
    ("Drivers", "Fleet week"),
    ("Smart charging", "1 Plan"),
    ("Smart charging", "2 Response"),
    ("Smart charging", "3 Value and risk"),
    ("Trading", "1 Market"),
    ("Trading", "2 Position"),
    ("Trading", "3 P&L and risk"),
    ("Supplier", "1 Availability and cost curve"),
    ("Supplier", "2 Positions"),
    ("Supplier", "3 Supplier P&L"),
    ("Supplier", "4 Firm MW"),
    ("Supplier", "5 Charger makers"),
    ("Supplier", "6 Customers"),
    ("Partners", "Driver"),
    ("Partners", "Energy supplier"),
    ("Partners", "Charger maker"),
    ("Partners", "Carmaker"),
    ("Partners", "Fleet and leasing"),
    ("Replay", "Fleet"),
    ("Replay", "Customer"),
    ("Compare", None),
    ("How it works", "Why this model"),
    ("How it works", "The forecast"),
    ("How it works", "Assumptions"),
    ("How it works", "Limits"),
    ("How it works", "How it was built"),
)
PENDING_VIEWS: tuple[tuple[str, str | None], ...] = ()
"""Views not yet swept; a new view goes here until it follows the rules."""

CAPTION_MAX = 140
KPI_LABEL_MAX = 28
IDENTITY_MAX = 80
EXEMPT_EXPANDERS = frozenset({"Data and definition", "Read the full explainer"})

FORBIDDEN = {
    "decision number": re.compile(r"\bdecision \d{4}\b", re.IGNORECASE),
    "decision item": re.compile(r"\bitems? \d+\b"),
    "contract reference": re.compile(r"\bcontract v\d", re.IGNORECASE),
    "report page": re.compile(r"\bp\.\s?\d+"),
    "snake_case name": re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"),
}
_KPI_LABEL = re.compile(r'<div class="axle-kpi-label">(.*?)</div>', re.S)
_UNIT_IN_LABEL = re.compile(r"£|%|\((?:[^)]*)\)|\b(?:kW|kWh|MWh|pts|GBP)\b")
_SNAKE_COLUMN = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")


# --- Walking the rendered tree ---------------------------------------------


def _texts(node) -> Iterator[tuple[str, str]]:
    """Yield ``(kind, text)`` for every piece of on-screen text under ``node``."""

    for child in getattr(node, "children", {}).values():
        kind = child.type
        label = getattr(child, "label", None)
        if kind == "expander":
            yield "label", label
            if label not in EXEMPT_EXPANDERS:
                yield from _texts(child)
            continue
        if kind == "tab":
            yield "label", label
        if kind == "dataframe":
            yield from (("column", str(name)) for name in child.value.columns)
            continue
        if kind in ("caption", "markdown", "info", "warning", "error", "success"):
            yield kind, str(child.value)
        elif kind in ("header", "subheader", "title"):
            yield "heading", str(child.value)
        elif isinstance(label, str) and kind != "tab":
            yield "label", label
        # Widget options are on screen too (review of ecacff9: a run label
        # carried the seed's internal unit into Compare's A/B selectors).
        for option in getattr(child, "options", None) or ():
            yield "option", str(getattr(option, "content", option))
        yield from _texts(child)


def _violations(texts: list[tuple[str, str]]) -> list[str]:
    found = []
    for kind, text in texts:
        if kind == "column":
            if _SNAKE_COLUMN.fullmatch(text):
                found.append(f"snake_case column {text!r}")
            continue
        # Inline SVG (the pipeline diagram) is markup, not prose.
        prose = re.sub(r"<svg.*?</svg>|<style>.*?</style>", "", text, flags=re.S)
        for rule, pattern in FORBIDDEN.items():
            match = pattern.search(prose)
            if match:
                found.append(f"{rule} {match.group(0)!r} in {kind}: {text[:90]!r}")
        if kind == "caption" and len(text) > CAPTION_MAX:
            found.append(f"caption of {len(text)} chars: {text[:90]!r}")
        for label in _KPI_LABEL.findall(text):
            if len(label) > KPI_LABEL_MAX:
                found.append(f"KPI label of {len(label)} chars: {label!r}")
            if _UNIT_IN_LABEL.search(label):
                found.append(f"unit in KPI label: {label!r}")
    findings = [text for kind, text in texts if kind == "caption" and text.startswith("Finding:")]
    if len(findings) > 1:
        found.append(f"{len(findings)} 'Finding:' captions: {findings}")
    return found


# --- Driving the app --------------------------------------------------------


def _go(app: AppTest, page_name: str) -> None:
    app._page_hash = next(
        page_hash
        for page_hash, page in app._registered_pages.items()
        if page["page_name"] == page_name
    )
    app.run()
    assert not app.exception


def _edit(app: AppTest, key: str, value: object) -> None:
    app.get_by_key("edit-assumptions").click().run()
    name = key.removeprefix("assumption::")
    group = next(record.group for record in assumptions.editable_records() if record.name == name)
    if app.get_by_key("assumptions-group").value != group:
        # AppTest's full rerun closes the dialog; it reopens on that group.
        app.get_by_key("assumptions-group").set_value(group).run()
        app.get_by_key("edit-assumptions").click().run()
    app.get_by_key(key).set_value(value).run()
    assert not app.exception


def _small_draft(app: AppTest) -> None:
    _edit(app, "assumption::vehicle_count", SMALL_FLEET)
    _edit(app, "assumption::evaluation_world_count", SMALL_WORLDS)


def _run(app: AppTest) -> None:
    app.get_by_key("run-simulation").click().run()
    assert not app.exception


@pytest.fixture(scope="module")
def app() -> AppTest:
    """Two small real runs (the second with a changed charger power and seed), for every view."""

    app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
    _small_draft(app)
    _run(app)
    _edit(app, "assumption::home_charging_power_kw", 3.6)
    # The seed change puts "Random seed" into run labels and Compare's change
    # line, where its record unit once leaked a snake_case name.
    _edit(app, "assumption::seed", 7)
    _run(app)
    return app


def _show(app: AppTest, page_name: str, lens_name: str | None) -> None:
    _go(app, page_name)
    if lens_name is not None:
        slug = next(page.slug for page in PAGES if page.name == page_name)
        app.get_by_key(f"lens::{slug}").set_value(lens_name).run()
        assert not app.exception


# --- Tests ------------------------------------------------------------------


def test_every_view_is_checked_or_pending() -> None:
    assert set(CHECKED_VIEWS) | set(PENDING_VIEWS) == set(pages.VIEWS)
    assert not set(CHECKED_VIEWS) & set(PENDING_VIEWS)


@pytest.mark.parametrize(("page_name", "lens_name"), CHECKED_VIEWS)
def test_view_copy_follows_the_rules(app: AppTest, page_name: str, lens_name: str | None) -> None:
    _show(app, page_name, lens_name)

    texts = list(_texts(app.main))
    assert texts, "nothing rendered"
    assert _violations(texts) == []


def test_run_bar_identity_is_short_and_evidence_is_a_badge(app: AppTest) -> None:
    _show(app, "How it works", "The forecast")

    identity = next(
        element.value for element in app.caption if element.value.startswith("Run 2 · ")
    )
    assert len(identity) <= IDENTITY_MAX
    badges = [element.value for element in app.markdown if "-badge[" in element.value]
    assert any("Illustrative" in badge for badge in badges)


def test_edit_assumptions_dialog_copy_follows_the_rules(app: AppTest) -> None:
    app.get_by_key("edit-assumptions").click().run()

    assert _violations(list(_texts(app.main))) == []


def test_empty_state_quotes_the_draft_size() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    # The empty-state body is not on the landing page (How it works needs no
    # result, design section 3.2); Overview is the nearest page that shows it.
    _go(app, "Overview")
    _edit(app, "assumption::vehicle_count", 200)

    text = "\n".join(element.value for element in app.markdown)
    assert "forecast 200 simulated EV drivers over 100 possible weeks" in text
    assert _violations(list(_texts(app.main))) == []
    assert run_controller.run_history(app.session_state) == []


def test_the_rules_catch_what_they_describe() -> None:
    # The checker itself: each rule fires on a sample it exists to stop.
    samples = [
        ("caption", "Shown per decision 0004 item 12."),
        ("markdown", "See contract v2 section 4.9."),
        ("caption", "CNZ report p.13 figure."),
        ("markdown", "The home_import_kw column."),
        ("caption", "x" * (CAPTION_MAX + 1)),
        ("markdown", '<div class="axle-kpi-label">Evening peak kW</div>'),
        ("markdown", '<div class="axle-kpi-label">A label that is far too long to fit</div>'),
        ("column", "interval_start_utc"),
        ("option", "Run 3 · Random seed 42 → 7 np.random.default_rng seed"),
    ]
    for sample in samples:
        assert _violations([sample]), sample
    two_findings = [("caption", "Finding: one."), ("caption", "Finding: two.")]
    assert _violations(two_findings)
    assert _violations([("caption", "P10–P90 across 100 simulated weeks. Illustrative.")]) == []
