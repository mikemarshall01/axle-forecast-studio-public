"""The page vocabulary and its dispatch table (DASHBOARD_DESIGN.md section 2)."""

from axle_studio.ui.pages import RESULT_OPTIONAL, VIEWS
from axle_studio.ui.registry import PAGES


def test_exactly_the_nine_design_pages_in_order_with_their_lenses() -> None:
    # DASHBOARD_DESIGN.md section 2 after the streamlining pass (0004 item 28),
    # reshaped by decision 0004 item 54 (polish plan G10), and by the Partners
    # page (overnight review log). The Fable landing pass moved How it works
    # to first, ahead of Overview, so the app opens on what is calculated and
    # assumed before any result rather than an empty landing page; its first
    # lens is Mike's opening note (overnight review log, "why-note" ruling).
    assert [(page.name, tuple(lens.name for lens in page.lenses)) for page in PAGES] == [
        (
            "How it works",
            ("Why this model", "The forecast", "Assumptions", "Limits", "How it was built"),
        ),
        ("Overview", ("At a glance", "Key stats")),
        (
            "Drivers",
            ("Plug-ins", "Sessions", "One EV", "Household", "Archetypes", "Fleet week"),
        ),
        ("Smart charging", ("1 Plan", "2 Response", "3 Value and risk")),
        ("Trading", ("1 Market", "2 Position", "3 P&L and risk")),
        (
            "Supplier",
            (
                "1 Availability and cost curve",
                "2 Positions",
                "3 Supplier P&L",
                "4 Firm MW",
                "5 Charger makers",
                "6 Customers",
            ),
        ),
        (
            "Partners",
            ("Driver", "Energy supplier", "Charger maker", "Carmaker", "Fleet and leasing"),
        ),
        ("Replay", ("Fleet", "Customer")),
        ("Compare", ()),
    ]
    assert all(page.question.endswith("?") for page in PAGES)
    assert all(lens.question.endswith("?") for page in PAGES for lens in page.lenses)


def test_url_paths_are_unique_and_url_safe() -> None:
    slugs = [page.slug for page in PAGES]
    assert len(set(slugs)) == len(slugs)
    assert all(slug.replace("-", "").isalnum() and slug.islower() for slug in slugs)


def test_dispatch_table_has_exactly_one_view_per_page_and_lens() -> None:
    expected = {
        (page.name, lens.name if lens else None)
        for page in PAGES
        for lens in page.lenses or (None,)
    }
    assert set(VIEWS) == expected
    assert RESULT_OPTIONAL <= expected


def test_retired_lenses_are_gone() -> None:
    retired = {
        "Command Centre",
        "Markets",
        "Portfolio & revenue",
        "Regions",
        "Route ledgers",
        "Axle cash",
        "Stress/world",
        "Customer outcomes",
        "Parameters",
        "Scenarios",
        "Flexibility",  # merged into Fleet week (decision 0004 item 54)
    }
    names = {page.name for page in PAGES} | {lens.name for page in PAGES for lens in page.lenses}
    assert not names & retired
