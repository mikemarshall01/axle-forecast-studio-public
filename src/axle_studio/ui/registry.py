"""The dashboard's page and lens vocabulary.

Five pages in a fixed order, each with its lenses and the one question each
screen answers (docs/DASHBOARD_DESIGN.md sections 0 and 2; decision 0004
item 28, which amends item 22). Lenses that can never apply to the two
explicit models (regions, portfolio, markets, route ledgers and similar) are
retired rather than shown as stubs (0004 item 19).

This module imports neither Streamlit nor the model, so the navigation,
dispatch table and tests share one list. Reshaping the navigation means
editing ``PAGES`` and the matching entries in ``pages.VIEWS``. It also names
the session keys of the two controls several lenses share (day type and
group), so those lenses read one value rather than each keeping its own.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Lens:
    """One segmented-control choice on a page and the question it answers."""

    name: str
    question: str


@dataclass(frozen=True, slots=True)
class Page:
    """A top-navigation page.

    ``lenses`` is empty for a page with a single view; ``question`` is then
    the line shown under the title. ``slug`` is the URL path.
    """

    name: str
    slug: str
    question: str
    lenses: tuple[Lens, ...] = ()


PAGES: tuple[Page, ...] = (
    # How it works opens the app (Fable landing pass): a first-time reader
    # sees what is being calculated, what is assumed and what is left out
    # before any result, rather than an empty Overview waiting on a Run
    # click. Its first lens is Mike's opening note on why the model exists
    # and why it is built this way (overnight review log, 30 Sep "why-note"
    # ruling), so the very first screen is the purpose, not the mechanics.
    # The "Start here" tour still opens from Overview ▸ At a glance
    # (registry order does not change where a tour starts), so this page
    # points a reader on to it (views/methods.py).
    Page(
        "How it works",
        "how-it-works",
        "How is it calculated, what is assumed, and what is left out?",
        (
            Lens("Why this model", "What is this model for, and why is it built this way?"),
            Lens("The forecast", "How is the forecast calculated?"),
            Lens("Assumptions", "What does the model assume, and where does each value come from?"),
            Lens("Limits", "What is not modelled, and what are the known limitations?"),
            Lens("How it was built", "How was this built, and what was left out?"),
        ),
    ),
    Page(
        "Overview",
        "overview",
        "What does a simulated week of these drivers look like?",
        # Decision 0004 item 54, polish plan G10: the landing lens first, then
        # the dense P10/P50/P90 table.
        (
            Lens("At a glance", "What does a simulated week of these drivers look like?"),
            Lens("Key stats", "What are the headline numbers, how wide is their spread?"),
        ),
    ),
    Page(
        "Drivers",
        "drivers",
        "How do individual drivers and archetypes behave?",
        # Brief first (0004 item 16): the two literal asks (Plug-ins, then
        # Sessions per archetype), then sketch 1, then the population checks.
        # Decision 0004 item 54 (plan G10) merges the old Flexibility lens
        # into Fleet week, so "how much can move" now sits with the fleet's
        # own week, and adds Sessions after Plug-ins.
        (
            Lens("Plug-ins", "When do drivers plug in and leave, and at what battery level?"),
            Lens("Sessions", "How are plug-in sessions distributed for each archetype?"),
            Lens("One EV", "What does one driver's week look like?"),
            # Household contract v1 §6.1: the same car across every simulated week.
            Lens(
                "Household", "What does this driver get: ready mornings, savings and flexibility?"
            ),
            Lens("Archetypes", "Do the six archetypes reproduce what CNZ observed?"),
            Lens("Fleet week", "How does the fleet move through the week, and what can wait?"),
        ),
    ),
    Page(
        "Smart charging",
        "smart-charging",
        "What did the smart charger do, what changed, and what did it cost?",
        # The slug now matches the page name (goal review N13; supersedes
        # the earlier "wholesale-action" slug that goal review section 8
        # item 6 had left alone as URL-only): the page's own content is the
        # smart-charging plan, response and cost -- decision 0004 item 38
        # superseded the wholesale-ceiling action the old name described.
        # Numbered so the presenter walks them in order (design 4.3,
        # decision 0004 item 43: "1 Plan, 2 Response, 3 Value and risk").
        (
            Lens("1 Plan", "What does the smart charger plan, and what did it know?"),
            Lens("2 Response", "What changed physically when the plan applied?"),
            Lens("3 Value and risk", "What did it cost or save, and what risk did drivers carry?"),
        ),
    ),
    Page(
        "Trading",
        "trading",
        "What did the illustrative simulated trading overlay buy, sell and settle?",
        # Numbered so the presenter walks them in order, mirroring Smart
        # charging's "1 Plan, 2 Response, 3 Value and risk" (decision 0004
        # item 57 planner, item 58 trader views; plan section F).
        (
            Lens("1 Market", "What prices did the fleet trade against this week?"),
            Lens("2 Position", "What did the trader sell, and how did the fleet actually move?"),
            Lens(
                "3 P&L and risk",
                "What did the illustrative trading overlay make or lose, and how firm was it?",
            ),
        ),
    ),
    # Supplier contract v1 §11 and trading contract v1 §9.8, §10.10: after the
    # Trading page (plan §8 IA). "4 Firm MW" keeps its number
    # (§16 Q8) and is filled by the firm-MW UI lane.
    Page(
        "Supplier",
        "supplier",
        "What can a supplier buy from this fleet, and what is it worth?",
        (
            Lens("1 Availability and cost curve", "What could the fleet move, and at what price?"),
            Lens("2 Positions", "What was sold day-ahead, and what was held at the end?"),
            Lens("3 Supplier P&L", "What does smart charging do to a supplier's week?"),
            Lens("4 Firm MW", "How much turn-down can be promised firmly?"),
            # Renamed from "5 Partners" once the Partners page existed as a
            # top-level destination of its own (below): two lenses named
            # "Partners" would be ambiguous in a demo ("go to Partners" could
            # mean either). This lens keeps its number and its frames; only
            # the on-screen name changes.
            Lens("5 Charger makers", "What does a charger maker's enrolled device earn?"),
            # Household contract v1 §6.2: the supplier's own customers, after Partners.
            Lens("6 Customers", "What can we tell customers, and which customers gain least?"),
        ),
    ),
    # "Who is this for" page: a pitch card per outside audience (driver,
    # energy supplier, charger maker, carmaker, fleet/leasing operator),
    # reusing the charts and numbers already built for Drivers, Smart
    # charging, Supplier and Trading rather than drawing anything new. After
    # Supplier (its frames are action-only, the same guard as Supplier's own
    # lenses) and before Replay.
    Page(
        "Partners",
        "partners",
        "Who is this for, and what would each partner want to see?",
        (
            Lens("Driver", "What would a driver want to see?"),
            Lens("Energy supplier", "What would an energy supplier want to see?"),
            Lens("Charger maker", "What would a charger maker want to see?"),
            Lens("Carmaker", "What would a carmaker want to see?"),
            Lens("Fleet and leasing", "What would a fleet or leasing operator want to see?"),
        ),
    ),
    Page(
        "Replay",
        "replay",
        "What did the fleet know, do and earn at each moment of one simulated week?",
        # Replay contract v1 section 3.1 (decision 0004 item 66): play one
        # simulated week back with a moving "now"; a page of its own after
        # Smart charging, Trading and Supplier.
        (
            Lens("Fleet", "What did the fleet know, do and earn as the week unfolded?"),
            Lens(
                "Customer",
                "What did one driver's charger know, plan and save as the week unfolded?",
            ),
        ),
    ),
    Page(
        "Compare",
        "compare",
        "What changed between two runs?",
    ),
)
"""Pages in navigation order; the first is the landing page."""


# --- Shared controls (decision 0004 item 54, polish plan E2) -----------------
# One day-type control and one group selector, each with one session key, so a
# choice made on one lens holds on every lens that shows the same control.
# Views pass ``persist_state="session"`` so the value survives page changes.

DAY_TYPE_KEY = "day-type"
DAY_TYPE_LABELS = {"weekday": "Weekday", "weekend": "Weekend", "all": "All"}
"""Day-type values (the model's ``day_type`` column) and their on-screen labels."""

GROUP_KEY = "group"
FLEET_GROUP = "fleet"
"""The group selector's fleet value; archetypes use their ``cohort_id``."""
