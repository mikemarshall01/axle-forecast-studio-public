"""Page layout and the one dispatch table from (page, lens) to a view.

The page contract every view follows (docs/DASHBOARD_DESIGN.md section 6):

    render_<name>(st, result) -> None

``result`` is the active ``ForecastResult`` (docs/contracts/results-v2.md)
from ``model/forecast.run_forecast``; it is ``None`` only for views listed in
``RESULT_OPTIONAL``, which render before any run. A view only reads the
result: it never runs the full model, because a full Monte Carlo starts only
from an explicit Run click (AGENTS.md). Stale results are shown once, by the
run bar's chip, so views take no ``stale`` argument (design section 0).

This module owns the header (drawn by ``components/header.py``: a title row
with the view's own controls, then a full-width lens row), the empty state,
and the wiring a view needs from outside itself: the One EV replay
functions, the ``compare_runs`` function and the Edit assumptions dialog
opener. Views stay free of shell and model imports so they can be tested
against a fixture.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any

import pandas as pd

from axle_studio.model import market
from axle_studio.model.assumptions import PRICES, SIMULATION, assumption_rows
from axle_studio.model.forecast import ForecastResult
from axle_studio.model.household import household_card
from axle_studio.model.individual import (
    replay_one_ev,
    replay_one_ev_bands,
    replay_one_ev_timeline,
)
from axle_studio.model.summaries import RunComparison, compare_runs

from . import usage_log
from .components import header
from .registry import Lens, Page
from .run_controller import (
    MODEL_LABELS,
    RunRecord,
    active_result,
    display_label,
    display_unit,
    render_empty_state,
    set_user_price_curve,
    switch_to_action_model,
)
from .views import (
    action_cost,
    action_decision,
    action_response,
    archetypes,
    compare,
    customers,
    drivers_fleet,
    household,
    key_stats,
    methods,
    one_ev,
    overview,
    partner_personas,
    partners,
    plug_ins,
    replay,
    sessions,
    supplier_availability,
    supplier_firm_mw,
    supplier_pnl,
    supplier_positions,
    trading_market,
    trading_pnl,
    trading_position,
)

View = Callable[[Any, "ForecastResult | None"], None]
"""``view(st, result) -> None``; see the module docstring."""

NO_ACTION_MESSAGE = "This result has no Axle action."


# --- Smart charging ----------------------------------------------------------


def _action_lens(lens: str, view: View) -> View:
    """Guard an action lens so a no-action result gets one message and a way forward."""

    def render(st: Any, result: ForecastResult) -> None:
        if result.model != "action":
            # Design 4.3: one message and a way forward. The button changes
            # only the draft model; Run stays an explicit click.
            st.info(NO_ACTION_MESSAGE)
            st.button(
                "Switch to smart charging",
                key=f"switch-to-action::{lens}",
                on_click=switch_to_action_model,
                args=(st.session_state,),
            )
            return
        view(st, result)

    return render


# --- Supplier ---------------------------------------------------------------


def _validate_price_curve(state: Any, csv_bytes: bytes) -> pd.DataFrame:
    """Check an uploaded supplier curve against the draft's own price floor and cap (§9.4)."""

    values = state.get("draft_values", {})
    bound = {
        name: values.get(name, PRICES[name].value)
        for name in ("price_floor_gbp_per_mwh", "price_cap_gbp_per_mwh")
    }
    return market.validate_user_price_curve(
        csv_bytes,
        floor_gbp_per_mwh=float(bound["price_floor_gbp_per_mwh"]),
        cap_gbp_per_mwh=float(bound["price_cap_gbp_per_mwh"]),
    )


# --- Compare ----------------------------------------------------------------


def _compare_records(a: RunRecord, b: RunRecord) -> RunComparison:
    """``compare_runs`` on two kept runs, with every changed input named.

    ``compare_runs`` lists changed assumption records (each result carries
    the values it ran with) and run-shape settings.  A model change is added
    first, so Compare never says "no changes" between an action and a
    no-action run.
    """

    comparison = compare_runs(a, b)
    changed = comparison.changed
    # Run-shape keys come back under their raw snapshot names; show the
    # record labels instead ("Fleet size", "Random seed"), with the same
    # display overrides the dialog uses, so no internal unit such as the
    # seed's "np.random.default_rng seed" reaches the screen.
    labels = {name: record.label for name, record in SIMULATION.items()}
    labels["start_local_date"] = "Start date"
    changed = changed.assign(
        label=[
            display_label(n, labels.get(n, lab))
            for n, lab in zip(changed["name"], changed["label"], strict=True)
        ],
        unit=[display_unit(n, u) for n, u in zip(changed["name"], changed["unit"], strict=True)],
    )
    # A model change is named once, below; the planning-world count follows
    # from the model (one for action, none for no-action), so it is not a
    # separate change.
    changed = changed.loc[~changed["name"].isin(["model", "planning_world_count"])]
    if a.model != b.model:
        # The view prints "<label> <A> → <B>"; the colon makes it read
        # "Model: Smart charging → No action" once run_controller's
        # MODEL_LABELS carries the same rename (goal review item 6).
        model_row = pd.DataFrame(
            [("model", "Model:", "", MODEL_LABELS[a.model], MODEL_LABELS[b.model])],
            columns=list(changed.columns),
            dtype=object,
        )
        changed = pd.concat([model_row, changed], ignore_index=True)
    return replace(comparison, changed=changed)


def unpairable_reason(a: RunRecord, b: RunRecord) -> str | None:
    """Why two runs cannot be paired world by world, or ``None`` if they can.

    ``compare_runs`` raises ``ValueError`` when the world count, horizon
    start or slot count differ, because world w of A and B then cover
    different weeks or do not exist in both. Passed into
    ``compare.render_compare`` (below), which checks these same three fields
    on the presenter's chosen A/B pair before calling ``compare_runs``, so
    Compare shows a plain sentence instead of a crash -- and, for the
    *default* pair only, can fall back to the most recent pair of kept runs
    that passes this check: a hard-coded previous-vs-latest default would
    otherwise leave an actually-comparable earlier pair unreachable.
    """

    if a.world_count != b.world_count:
        difference = (
            f"Run {a.number} has {a.world_count} simulated weeks, "
            f"Run {b.number} has {b.world_count}"
        )
    elif a.horizon_start_utc != b.horizon_start_utc or a.study_slot_count != b.study_slot_count:
        start_a, start_b = a.start_local_date, b.start_local_date
        difference = (
            f"Run {a.number} starts {start_a:%a} {start_a.day} {start_a:%b}, "
            f"Run {b.number} starts {start_b:%a} {start_b.day} {start_b:%b}"
        )
    else:
        return None
    return f"Cannot compare: {difference}. Keep weeks and start date the same to compare."


def render_compare(st: Any, result: ForecastResult | None) -> None:
    """Compare two picked runs (A/B selectors, defaulting to previous vs latest, design 4.4)."""

    compare.render_compare(
        st, result, compare_runs=_compare_records, unpairable_reason=unpairable_reason
    )


# --- How it works -----------------------------------------------------------


def render_assumptions(st: Any, result: ForecastResult | None) -> None:
    """How it works ▸ Assumptions: the read-only table, from the active result.

    After a run the table shows the records the active result carries (the
    values it ran with); before a run it shows every record in
    ``model/assumptions.py`` at its default (decision 0004 item 21). The run
    bar's own "Edit assumptions" button, always visible, opens the shared
    dialog (layout ruling 3: this lens no longer draws a second one).
    """

    methods.render_assumptions(st, result, assumption_rows_provider=assumption_rows)


# --- Dispatch ---------------------------------------------------------------

VIEWS: dict[tuple[str, str | None], View] = {
    ("Overview", "At a glance"): overview.render_overview,
    ("Overview", "Key stats"): key_stats.render_key_stats,
    ("Drivers", "Plug-ins"): plug_ins.render_plug_ins,
    ("Drivers", "Sessions"): sessions.render_sessions,
    # The replay is the only model call a view makes (contract v2 section 5):
    # a one-EV kernel slice, cached on the result, never a full Monte Carlo.
    ("Drivers", "One EV"): partial(
        one_ev.render_one_ev,
        replay_one_ev=replay_one_ev,
        replay_one_ev_bands=replay_one_ev_bands,
    ),
    # The same one-EV model call pattern as One EV: a card cached on the result.
    ("Drivers", "Household"): _action_lens(
        "household", partial(household.render_household, household_card=household_card)
    ),
    ("Drivers", "Archetypes"): archetypes.render_archetypes,
    ("Drivers", "Fleet week"): drivers_fleet.render_fleet_week,
    ("Smart charging", "1 Plan"): _action_lens("plan", action_decision.render_action_decision),
    ("Smart charging", "2 Response"): _action_lens(
        "response", action_response.render_action_response
    ),
    ("Smart charging", "3 Value and risk"): _action_lens("value", action_cost.render_action_cost),
    ("Trading", "1 Market"): _action_lens("trading-market", trading_market.render_trading_market),
    ("Trading", "2 Position"): _action_lens(
        "trading-position", trading_position.render_trading_position
    ),
    ("Trading", "3 P&L and risk"): _action_lens("trading-pnl", trading_pnl.render_trading_pnl),
    # Every Supplier lens reads action-only frames (supplier contract v1 §1).
    ("Supplier", "1 Availability and cost curve"): _action_lens(
        "supplier-availability", supplier_availability.render_availability
    ),
    ("Supplier", "2 Positions"): _action_lens(
        "supplier-positions",
        partial(
            supplier_positions.render_positions,
            validate_price_curve=_validate_price_curve,
            set_price_curve=set_user_price_curve,
        ),
    ),
    ("Supplier", "3 Supplier P&L"): _action_lens("supplier-pnl", supplier_pnl.render_supplier_pnl),
    ("Supplier", "4 Firm MW"): _action_lens("supplier-firm-mw", supplier_firm_mw.render_firm_mw),
    ("Supplier", "5 Charger makers"): _action_lens("supplier-partners", partners.render_partners),
    # Household contract v1 §6.2: the supplier's own customers, after Partners.
    ("Supplier", "6 Customers"): _action_lens("supplier-customers", customers.render_customers),
    # Partners: a pitch card per outside audience, reusing the frames and
    # chart builders Drivers/Smart charging/Trading/Supplier already draw
    # (docs/DASHBOARD_DESIGN.md section 2). Every persona reads action-only
    # frames, the same guard as every Supplier lens.
    ("Partners", "Driver"): _action_lens("partners-driver", partner_personas.render_driver),
    ("Partners", "Energy supplier"): _action_lens(
        "partners-energy-supplier", partner_personas.render_energy_supplier
    ),
    ("Partners", "Charger maker"): _action_lens(
        "partners-charger-maker", partner_personas.render_charger_maker
    ),
    ("Partners", "Carmaker"): _action_lens("partners-carmaker", partner_personas.render_carmaker),
    # The same one-EV model call pattern as Drivers ▸ Household: a card
    # cached on the result, injected so this module stays free of a model
    # import (module docstring).
    ("Partners", "Fleet and leasing"): _action_lens(
        "partners-fleet-and-leasing",
        partial(partner_personas.render_fleet_and_leasing, household_card=household_card),
    ),
    # Replay contract v1 section 3.1. The customer lens's plan snapshots and
    # running cost come from the one-EV kernel run (``individual``, lane
    # R1c), the same model call pattern as One EV above.
    ("Replay", "Fleet"): _action_lens("replay-fleet", replay.render_replay_fleet),
    ("Replay", "Customer"): _action_lens(
        "replay-customer",
        partial(
            replay.render_replay_customer,
            replay_one_ev=replay_one_ev,
            replay_one_ev_timeline=replay_one_ev_timeline,
        ),
    ),
    # Compare reads the run history's slim records, not ``result``.
    ("Compare", None): render_compare,
    ("How it works", "Why this model"): methods.render_why_this_model,
    ("How it works", "The forecast"): methods.render_forecast_method,
    ("How it works", "Assumptions"): render_assumptions,
    ("How it works", "Limits"): methods.render_limits,
    ("How it works", "How it was built"): methods.render_build_notes,
}
"""Every (page, lens) in the registry maps to exactly one view.

One table, rather than if/elif chains per page, so a reader sees the whole
app on one screen and a registry test proves nothing is missing.
"""

READING_COLUMN_PAGES = frozenset({"How it works"})
"""Pages whose header is capped to the prose reading column's width (layout
ruling 2, 30 Sep 2026): title, lens bar, the pipeline diagram and the prose
all share one left edge and one maximum width (``views/methods.PROSE_WIDTH_PX``),
so the lens bar never floats wider than the text underneath it. The
Assumptions lens's table is the one thing on this page that still uses the
full content width (ruling 2's own exception; the table needs the room)."""

RESULT_OPTIONAL = frozenset(
    {
        ("Compare", None),
        ("How it works", "Why this model"),
        ("How it works", "The forecast"),
        ("How it works", "Assumptions"),
        ("How it works", "Limits"),
        ("How it works", "How it was built"),
    }
)
"""Views that render before any run (design 3.2: How it works needs no result;
Compare shows its own "run twice" state). Every other view receives a result
or is not called; the shell shows the empty state with its own Run button."""


def _lens_control(st: Any, page: Page) -> Lens | None:
    if not page.lenses:
        return None
    names = [lens.name for lens in page.lenses]
    # persist_state="session" keeps the lens when the user visits another
    # page and returns (design section 2). Another page can preselect a lens
    # by writing its name to st.session_state[f"lens::{slug}"] before
    # st.switch_page.
    chosen = st.segmented_control(
        "Lens",
        options=names,
        default=names[0],
        required=True,
        key=f"lens::{page.slug}",
        label_visibility="collapsed",
        width="stretch",
        wrap=True,  # four labels wrap onto two rows at 390 px instead of clipping
        persist_state="session",
    )
    return page.lenses[names.index(chosen)] if chosen in names else page.lenses[0]


def render_page(st: Any, page: Page) -> None:
    """Render one page: the two-row header, then its view or the empty state.

    Row 1 (``components/header.py``): title and question │ the view's own
    controls. Row 2, full width: the lens control alone, on the pages that
    have one. The view runs inside ``header.controls_slot`` so its
    Day/Metric/EV/A-B controls land in row 1 rather than on a row of their
    own above the first chart. How it works caps both rows to the prose
    reading column instead of the header's ordinary full width
    (``READING_COLUMN_PAGES`` above).
    """

    max_width_px = methods.PROSE_WIDTH_PX if page.name in READING_COLUMN_PAGES else None
    title_column, controls_column, lens_row = header.header_row(
        st, has_lens=bool(page.lenses), max_width_px=max_width_px
    )
    lens = None
    if lens_row is not None:
        # Placed straight in the container, not a right-aligned horizontal
        # one: that stopped the labels wrapping and clipped them at 390 px.
        with lens_row:
            lens = _lens_control(st, page)
    # No second heading repeating the lens name (design section 3).
    header.render_title(title_column, page.name, lens.question if lens else page.question)

    key = (page.name, lens.name if lens else None)
    # Anonymous usage logging (AXLE_USAGE_LOG=1; ui/usage_log.py): only when
    # the page or lens actually changes, not on every rerun a widget causes
    # while the visitor stays put.
    if st.session_state.get("usage_last_view") != key:
        st.session_state["usage_last_view"] = key
        usage_log.log_event(st.session_state, "view", page=key[0], lens=key[1])
    result = active_result(st.session_state)
    if result is None and key not in RESULT_OPTIONAL:
        render_empty_state(st)
        return
    with header.controls_slot(controls_column):
        VIEWS[key](st, result)


def page_renderer(st: Any, page: Page) -> Callable[[], None]:
    """Bind a page to the zero-argument callable ``st.Page`` expects."""

    def render() -> None:
        render_page(st, page)

    return render
