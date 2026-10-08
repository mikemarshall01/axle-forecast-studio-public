"""Run lifecycle for the dashboard: the draft, the explicit Run and the run bar.

This module owns the session state that decides which forecast the pages
show. The rules come from AGENTS.md and decision 0004 (items 9, 18, 21, 22,
28):

* Two models can be selected: the explicit smart charging model (default)
  and the explicit no-action model.
* The draft is every editable assumption in ``model/assumptions.py``, by
  record name, edited freely in the Edit assumptions dialog. Editing,
  navigating, Reset, switching model and ordinary Streamlit reruns never run
  the model.
* A full Monte Carlo starts only from a click on "Run simulation" while the
  draft is valid. A failed run keeps the previous result.
* The active result is "stale" when the draft (model, start date or any
  value) differs from what produced it. Pages keep showing it; the run bar
  says so.
* The last three completed runs are kept for Compare (design section 4.4),
  as slim records; only the latest keeps its full result (decision 0004
  item 35).
* A new session opens on the precomputed default run as "Run 0" when a
  compatible saved file exists (``ui/demo_run.py``; loaded once per server
  process and shared read-only across sessions). Loading it is not a
  run; the next Run click is Run 1.

Design choices, and the alternatives rejected:

* Staleness compares the draft's values with the active run's values name
  by name, rather than counting edit revisions. An edit that is undone
  leaves the result current, and the dialog can name each unrun change.
* The draft values dict is the single owner of every editable value. Widget
  keys are rewritten from it on each full rerun (``sync_widgets_from_draft``),
  so a widget can never show a value the next Run will not use.
* Validation comes from the records' bounds (``assumptions.validation_errors``),
  so the dialog, the run bar and the model reject the same values.
* A Run click only sets a flag (a button callback, which runs before the
  script). The script then draws the whole page from the current result and
  executes the run at the end, into a status placeholder in the bar, and
  reruns (design section 3.1). The page stays readable while the model works,
  and the controls are drawn disabled with the word "Running".
* The run bar sits in the main area of every page, not a sidebar, so Run is
  reachable at 390 px without opening a menu.

Session-state keys owned here:

``model_mode``          selected model id (also the model control's widget key)
``assumption::<name>``  widget values for the editable assumptions
``draft_values``        ``{name: value}`` for every editable assumption
``draft_event_presets`` scripted event preset ids chosen for the next run
``draft_user_price_curve`` the supplier's validated 48-value price curve, or ``None``
``draft_blackout_windows`` the ``availability.blackout_windows`` draft table (§10.5b;
                        a run input beside the edited values, like the price curve)
``event_preset::<id>``  widget values for the Events checkboxes
``draft_start_date``    the London date the next run's study starts (today)
``draft_errors``        ``{name: message}`` for invalid draft values
``draft_event_error``   why the chosen event presets are invalid, or ``None``
``draft_blackout_error`` why the draft blackout table is invalid, or ``None``
``draft_error``         one-line summary of every error kind, or ``None``
``run_log``             ``RunLog``: kept runs, last failure and a pending notice
``run_requested``       set by a Run click, consumed by the next script run
``run_in_progress``     True only during the script run that executes the model
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, fields, replace
from datetime import date, datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from axle_studio.model import assumptions, availability
from axle_studio.model import events as model_events
from axle_studio.model.forecast import ForecastResult, run_forecast_from_assumptions
from axle_studio.model.summaries import RunSummary, build_study_slots, slim_run

from . import demo_run, run_guard, usage_log

# The ids ``run_forecast`` takes, so the draft model, a run record and a
# ``ForecastResult`` all name a model the same way.
MODEL_ACTION = "action"
MODEL_NO_ACTION = "no_action"
MODELS = (MODEL_ACTION, MODEL_NO_ACTION)
MODEL_LABELS = {MODEL_ACTION: "Smart charging", MODEL_NO_ACTION: "No action"}

HISTORY_SIZE = 3
"""Completed runs kept for Compare (design section 4.4)."""

DEFAULT_RUN_MINUTES_ESTIMATE = 5
"""Rough current wall-clock time for the default 1,000 EV x 100 week run.

Final critique B-14: the empty state promised "70-90 s", a figure from
before the plan-status wiring lane (30 Sep) made the default run heavier
(overnight log: ~116 s -> ~183 s measured then); the final critique measured
3 min 0 s-3 min 50 s on 29 Sep. A named constant, not a string baked into the
sentence, so re-timing the model updates the on-screen claim in one place;
re-time this whenever the default run's cost changes materially, rather than
letting the words drift from what a presenter actually sees again.
"""

WIDGET_PREFIX = "assumption::"
"""Session-state key prefix of the dialog's widgets, one per editable record."""

EVENT_WIDGET_PREFIX = "event_preset::"
"""Session-state key prefix of the Events checkboxes, one per preset id."""

EVENT_PRESET_IDS = tuple(assumptions.EVENT_PRESETS)
"""Every scripted event preset, in ``assumptions.EVENT_PRESETS`` order.

The draft stores its chosen presets in this order whatever the click order,
so two drafts with the same choice compare equal and a run label never
reports a reordering as a change."""

_LONDON = ZoneInfo("Europe/London")
# Every editable record, by name: label and unit for chips, run labels and
# the dialog.  Built once; the records are module constants.
_EDITABLE = {record.name: record for record in assumptions.editable_records()}
_SHELL_VERSION = 9
"""Bumped when session keys change, so a code reload starts a clean session."""

EMPTY_BLACKOUT_WINDOWS = pd.DataFrame(
    {
        "start_local_time": pd.Series(dtype=object),
        "duration_minutes": pd.Series(dtype="int64"),
    }
)
"""The draft's blackout-windows table with no rows: the record's default, no blackout (§10.5b)."""


def _same_blackout_windows(a: pd.DataFrame | None, b: pd.DataFrame | None) -> bool:
    """Whether two draft blackout tables are the same run input, ``None`` as "no rows".

    A kept run built before this table existed (or a bare ``RunRecord`` in a
    test) carries ``blackout_windows=None`` by the dataclass default, the
    same meaning as the draft's own no-rows default, so the two must not
    read as "changed" against each other.
    """

    return same_curve(
        EMPTY_BLACKOUT_WINDOWS if a is None else a, EMPTY_BLACKOUT_WINDOWS if b is None else b
    )


# Display-only overrides for the Simulation fields (goal review action 10):
# the world count is shown as "Simulated weeks" with no unit, and the seed
# with no unit, so a run label reads "Simulated weeks 100 → 4", not
# "Simulated weeks 100 → 4 weeks". Cosmetic only: record.name, .value,
# .bounds and .meaning are untouched, so the model, the draft and the dialog
# agree on what each field means. Kept here, not in the dialog, because run
# labels and Compare's change line name the same fields (review of ecacff9,
# blocking 2).
_DISPLAY_LABELS: dict[str, str] = {"evaluation_world_count": "Simulated weeks"}
_DISPLAY_UNITS: dict[str, str] = {
    "evaluation_world_count": "",
    "seed": "",
    # The record's unit text ("choice (fixed share or newsvendor)") is a
    # description, not a unit; run labels read "Commitment rule fixed share
    # → newsvendor" without it.
    "trading.commitment_rule": "",
}


def display_label(name: str, label: str) -> str:
    """The on-screen label for assumption ``name`` (its record label unless overridden)."""

    return _DISPLAY_LABELS.get(name, label)


def display_unit(name: str, unit: str) -> str:
    """The on-screen unit for assumption ``name`` ("" for a unit that is not one)."""

    return _DISPLAY_UNITS.get(name, unit)


Runner = Callable[..., ForecastResult]
"""Runs one model: ``(model, values, start_local_date, event_presets) -> ForecastResult``,
with ``user_price_curve=`` and ``blackout_windows=`` as keywords, each passed only when the
draft carries one (a supplier curve, or a non-empty blackout table, §10.5b)."""


def applied_presets(model: str, event_presets: tuple[str, ...]) -> tuple[str, ...]:
    """The presets a run of ``model`` actually applies: none for the no-action model.

    The no-action model has no events (trading contract v1 §2, the forecast
    ignores them), so labels, the run identity and Compare must not claim
    events it never simulated. The draft keeps the choice either way, so
    switching back to smart charging does not lose it.
    """

    return event_presets if model == MODEL_ACTION else ()


def event_presets_text(event_presets: tuple[str, ...]) -> str:
    """Plain-English list of presets by label, or "none"."""

    labels = [str(assumptions.EVENT_PRESETS[preset]["label"]) for preset in event_presets]
    return "; ".join(labels) or "none"


def event_label(event_id: str, start_london: datetime | None = None) -> str:
    """The on-screen name of a scripted event: its preset's label, plus the day for a series.

    Scripted event ids are the preset id, or ``<preset>_n<night>`` for the
    two presets that add one event per night (``events.preset_rows``), so
    "cold_still_week_n3" reads "Cold still week, Thu 1 Oct" and never shows
    the id itself.
    """

    for preset_id, preset in assumptions.EVENT_PRESETS.items():
        if event_id == preset_id:
            return str(preset["label"])
        if event_id.startswith(f"{preset_id}_n"):
            if start_london is None:
                return str(preset["label"])
            return f"{preset['label']}, {start_london:%a} {start_london.day} {start_london:%b}"
    return event_id.replace("_", " ").capitalize()


def _event_presets_short(event_presets: tuple[str, ...]) -> str:
    """Short form for run labels: "none", the one preset's label, or "N presets"."""

    if len(event_presets) <= 1:
        return event_presets_text(event_presets)
    return f"{len(event_presets)} presets"


def event_presets_error(event_presets: tuple[str, ...], start_local_date: date) -> str | None:
    """Why this preset choice cannot run from ``start_local_date``, or ``None``.

    Built and checked by the model's own functions (``events.event_table``
    and ``events.validate_events``, contract §2.3), the same ones the run
    uses, so the dialog and the model reject the same selection (for
    example two grid requests overlapping in time). Checked on the study
    slots of the draft's start date, because a preset's rows depend on it
    (the sunny weekend lands on whichever Saturday and Sunday fit).
    """

    if not event_presets:
        return None
    slots = build_study_slots(start_local_date, int(assumptions.SIMULATION["study_days"].value))
    try:
        model_events.validate_events(
            model_events.event_table(event_presets, slots), slots, assumptions.ZONE_IDS
        )
    except ValueError as error:
        return str(error)
    return None


@dataclass(frozen=True)
class RunRecord(RunSummary):
    """One completed Run: the slim summary Compare reads, plus the shell's bookkeeping.

    A ``RunRecord`` *is* a ``summaries.RunSummary`` (model, seed, world count,
    per-world frames, settings snapshot, assumptions), so Compare passes two
    records straight to ``compare_runs`` (decision 0004 item 35). The shell
    adds the run ``number``, its ``label`` (e.g. "Run 3 · Home charging power
    7 → 3.6 kW": the first way the run differs from the previous one, or
    "base"), the editable ``values``, ``event_presets`` and
    ``start_local_date`` the draft is compared with, and ``result``.

    Only the latest record keeps the full ``result``; older records keep
    ``result=None``. Why: a full 1,000 EV × 100 week result holds per-EV
    replay inputs and per-world summaries for every view, and three of them
    would take a large share of a gigabyte per browser session.
    """

    number: int
    label: str
    values: dict[str, Any]
    start_local_date: date
    result: ForecastResult | None
    event_presets: tuple[str, ...] = ()
    user_price_curve: pd.DataFrame | None = None
    blackout_windows: pd.DataFrame | None = None
    """The draft ``availability.blackout_windows`` table this run used (§10.5b; ``None``: none),
    a run input beside ``values`` like ``user_price_curve``, restored by "Reset to active run"."""


def _events_record(model: str, event_presets: tuple[str, ...]) -> assumptions.Assumption:
    """The run's scripted events as one read-only record, for Compare's change line.

    Why a record: Compare lists what changed from each run's assumption
    records (``summaries.compare_runs``), and the events choice is a run
    input like any assumption. Adding it to the kept record's assumptions
    makes "Scripted events none → Cold still evening" appear there with no
    second change list. It is added to the slim ``RunRecord`` only, never to
    the result, so the How it works assumption tables are unchanged.
    """

    return assumptions.Assumption(
        name="events.presets",
        label="Scripted events",
        value=event_presets_text(applied_presets(model, event_presets)),
        unit="",
        evidence="illustrative",
        source="Edit assumptions, Events group (illustrative presets)",
        meaning="The scripted events this run applied. The no-action model applies none.",
        editable=False,
        bounds=None,
        group="Simulation",
        affects="Prices, smart charging plans or charger control inside each event's window.",
    )


def _record(
    number: int,
    label: str,
    values: dict[str, Any],
    start: date,
    result: ForecastResult,
    event_presets: tuple[str, ...] = (),
    user_price_curve: pd.DataFrame | None = None,
    blackout_windows: pd.DataFrame | None = None,
) -> RunRecord:
    summary = slim_run(result)
    kept = {name.name: getattr(summary, name.name) for name in fields(RunSummary)}
    kept["assumptions"] = (*kept["assumptions"], _events_record(summary.model, event_presets))
    return RunRecord(
        **kept,
        number=number,
        label=label,
        values=dict(values),
        start_local_date=start,
        result=result,
        event_presets=tuple(event_presets),
        user_price_curve=user_price_curve,
        blackout_windows=blackout_windows,
    )


@dataclass
class RunLog:
    """Mutable record of runs, fetched from session state before a run starts.

    Why mutable and fetched first: every ``st.session_state`` read or write is
    a Streamlit yield point, where a click made during the ~60-70 s run stops
    the script. The run's outcome is therefore stored by changing this object
    in place, with no session-state access between the model finishing and
    the outcome being kept (completion-plan gap check 1).
    """

    history: list[RunRecord] = field(default_factory=list)
    """Up to ``HISTORY_SIZE`` completed runs, oldest first."""
    error: str | None = None
    """Message from the last failed Run, cleared by an edit or a good Run."""
    notice: str | None = None
    """One-off message shown as a toast on the next rerun."""
    size_capped: bool = False
    """True only when ``error`` is a hosted-demo size refusal (``run_guard.size_refusal``),
    so the run bar can offer "Run at the hosted size" instead of a bare failure message."""


def _log(state: Any) -> RunLog:
    return state["run_log"]


# --- Draft values -----------------------------------------------------------


def london_today() -> date:
    """Return today's civil date in Europe/London; every study starts on it."""

    return datetime.now(_LONDON).date()


def default_values() -> dict[str, Any]:
    """Every editable assumption at its default, by record name."""

    return dict(assumptions.editable_defaults())


def commitment_rule_error(
    commitment_rule: str, world_count: int, event_presets: tuple[str, ...]
) -> str | None:
    """Why the newsvendor commitment rule cannot run with this draft, or ``None``.

    Trading contract v1 §10.4; overnight review log, 30 Sep 2026: with one
    simulated week the rule has no other week to read a forecast-error
    quantile from, and a short-notice scripted event is the same scenario in
    every simulated week, so the "other weeks" it reads from also contain
    this week's own scripted surprise. ``event_presets`` are the presets an
    action run would actually apply (``applied_presets``); the no-action
    model has none, so the rule is never blocked by them there.
    """

    if commitment_rule != "newsvendor":
        return None
    if world_count <= 1:
        return "needs more than one simulated week to learn a forecast error from"
    short_notice = {
        preset_id
        for preset_id, preset in assumptions.EVENT_PRESETS.items()
        if preset["notice"] == "short"
    }
    if short_notice.intersection(event_presets):
        return "cannot run with a short-notice scripted event selected"
    return None


def _recompute_draft_error(state: Any) -> None:
    """Rebuild the one-line ``draft_error`` banner from every error source.

    Called after any draft change that can affect it: an edited value or
    event choice (from ``_set_draft``), an edited blackout table
    (``set_blackout_windows``) or a model switch (``model_changed``, since
    the commitment-rule/event conflict below reads the model-specific
    ``applied_presets``).
    """

    messages = [
        f"{display_label(name, _EDITABLE[name].label)} {message}"
        for name, message in state.get("draft_errors", {}).items()
    ]
    if state.get("draft_event_error"):
        messages.append(f"Events: {state['draft_event_error']}")
    if state.get("draft_blackout_error"):
        messages.append(f"Blackout windows: {state['draft_blackout_error']}")
    values = state.get("draft_values", {})
    conflict = commitment_rule_error(
        values.get("trading.commitment_rule", "fixed_share"),
        int(values.get("evaluation_world_count", 0)),
        applied_presets(
            state.get("model_mode", MODEL_ACTION), state.get("draft_event_presets", ())
        ),
    )
    if conflict:
        messages.append(f"Commitment rule: {conflict}")
    state["draft_error"] = "; ".join(messages) or None


def _set_draft(
    state: Any, values: dict[str, Any], event_presets: tuple[str, ...] | None = None
) -> None:
    """Store the draft values and presets and their validation outcome. Never runs the model.

    ``event_presets`` ``None`` keeps the draft's current presets.
    """

    if event_presets is not None:
        chosen = set(event_presets)
        state["draft_event_presets"] = tuple(p for p in EVENT_PRESET_IDS if p in chosen)
        unknown = chosen - set(EVENT_PRESET_IDS)
        state["draft_event_error"] = (
            f"unknown event preset {sorted(unknown)[0]!r}"
            if unknown
            else event_presets_error(state["draft_event_presets"], state["draft_start_date"])
        )
    state["draft_values"] = values
    state["draft_errors"] = assumptions.validation_errors(values)
    _recompute_draft_error(state)


def same_value(a: Any, b: Any) -> bool:
    """Equal, with two NaN values (an unset commercial term) counted as the same.

    Plain ``!=`` would call an unset platform fee "changed" on every rerun
    (NaN never equals NaN), leaving the run bar stale for ever.
    """

    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def same_curve(a: pd.DataFrame | None, b: pd.DataFrame | None) -> bool:
    """Whether two user price curves (or two ``None``) are the same run input."""

    if a is None or b is None:
        return a is None and b is None
    return a.equals(b)


def is_switch(name: str) -> bool:
    """A 0/1 switch record, drawn as a toggle rather than a number field (lead ruling)."""

    return _EDITABLE[name].unit.startswith("switch")


def widget_value(name: str, value: Any) -> Any:
    """The draft value as its widget holds it.

    An unset term (NaN, decision 0003: unavailable, never 0) is an empty
    number field, ``None``; a 0/1 switch is a toggle's ``bool``.
    """

    if is_switch(name):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def draft_value(name: str, value: Any) -> Any:
    """The widget value as the draft stores it: the inverse of ``widget_value``."""

    if is_switch(name):
        return int(bool(value))
    return float("nan") if value is None else value


def sync_widgets_from_draft(state: Any) -> None:
    """Write the draft's values into every assumption widget key.

    Called on every full rerun. Why: Streamlit drops a widget's value when the
    widget is not on screen (another page, or the dialog closed), and a value
    written earlier through ``session_state`` can then resurface. That is how
    the fleet-size slider once showed one number while the draft and the run
    used another (design section 3.3). The draft is the single owner; widget
    callbacks update it before the script reruns, so copying it back here
    never loses an edit.
    """

    for name, value in state["draft_values"].items():
        state[WIDGET_PREFIX + name] = widget_value(name, value)
    chosen = state.get("draft_event_presets", ())
    for preset in EVENT_PRESET_IDS:
        state[EVENT_WIDGET_PREFIX + preset] = preset in chosen


def initialise_session(state: Any, today: date | None = None) -> None:
    """Create the draft on a session's first script run; later runs keep it."""

    if state.get("shell_version") == _SHELL_VERSION:
        # A code reload can leave a retired model id behind; fall back to the default.
        if state.get("model_mode") not in MODELS:
            state["model_mode"] = MODEL_ACTION
        return
    state.update(
        shell_version=_SHELL_VERSION,
        model_mode=MODEL_ACTION,
        run_log=RunLog(),
        run_requested=False,
        draft_start_date=today or london_today(),
        draft_user_price_curve=None,
        draft_blackout_windows=EMPTY_BLACKOUT_WINDOWS,
        draft_blackout_error=None,
    )
    _set_draft(state, default_values(), ())
    sync_widgets_from_draft(state)
    state["run_log"].history[:] = _precomputed_history()
    usage_log.log_event(state, "session_start")


def _precomputed_history() -> list[RunRecord]:
    """The saved default run as "Run 0", or no runs when there is no compatible file.

    Only reads a file; never runs the model (AGENTS.md). The run keeps the
    start date and values it was built with, so after midnight, or for a
    file built at a smaller size, the run bar shows it as stale like any
    other kept run. Numbered 0 so the first Run click in the session is
    still Run 1.
    """

    loaded = demo_run.shared_demo_run()
    if loaded is None:
        return []
    result, stamp = loaded
    values = dict(stamp["values"])
    if result.model == MODEL_ACTION and not _value_changes(default_values(), values):
        label = "Run 0 · precomputed default"
    else:
        label = run_label(0, result.model, values).replace("Run 0 ·", "Run 0 · precomputed ·", 1)
    return [_record(0, label, values, result.study_start_local_date, result)]


def _run_name(record: RunRecord) -> str:
    """'Run 3', or 'Run 0 (precomputed)' for the saved default loaded at start-up."""

    return f"Run {record.number} (precomputed)" if record.number == 0 else f"Run {record.number}"


def refresh_start_date(state: Any, today: date | None = None) -> None:
    """Move the draft's start date to today's London date after midnight.

    A result from yesterday then shows as stale; nothing is rerun.
    """

    state["draft_start_date"] = today or london_today()


def commit_assumption_edit(state: Any, name: str) -> None:
    """Widget callback: copy the one edited field ``name`` into the draft. Never runs the model.

    Only the edited field is copied. The dialog draws one group at a time,
    and the keys of fields in hidden groups can hold stale values (Streamlit
    keeps or drops an off-screen widget's key); copying every key would let
    those overwrite earlier edits in other groups (review of ecacff9,
    blocking 1: a seed edit reverted when home charging power was edited).
    """

    values = dict(state["draft_values"])
    values[name] = draft_value(name, state[WIDGET_PREFIX + name])
    _set_draft(state, values)
    _log(state).error = None


def commit_event_preset(state: Any, preset_id: str) -> None:
    """Checkbox callback: add or remove one event preset from the draft. Never runs the model."""

    chosen = set(state.get("draft_event_presets", ()))
    if state[EVENT_WIDGET_PREFIX + preset_id]:
        chosen.add(preset_id)
    else:
        chosen.discard(preset_id)
    _set_draft(state, dict(state["draft_values"]), tuple(chosen))
    _log(state).error = None


def set_user_price_curve(state: Any, curve: pd.DataFrame | None) -> None:
    """Put a validated supplier price curve (or ``None``) in the draft. Never runs the model.

    The curve is checked by ``market.validate_user_price_curve`` before it
    gets here (trading contract §9.4); like any edit it applies at the next
    Run, and until then the run bar shows the result as stale.
    """

    state["draft_user_price_curve"] = curve
    _log(state).error = None


def set_blackout_windows(state: Any, frame: pd.DataFrame) -> None:
    """Put an edited blackout-windows table in the draft. Never runs the model.

    Unlike a number field's bounds check, this table's validity depends on
    every row together (no two windows may share a half-hour, §10.5b), so
    it is checked here with ``availability.validate_blackout_windows``
    rather than through ``assumptions.validation_errors``; an invalid table
    sets ``draft_blackout_error`` and blocks Run, exactly as an invalid
    event selection does. Like the price curve, it applies at the next Run.
    """

    state["draft_blackout_windows"] = frame
    try:
        availability.validate_blackout_windows(frame)
    except (TypeError, ValueError) as error:
        state["draft_blackout_error"] = str(error)
    else:
        state["draft_blackout_error"] = None
    _recompute_draft_error(state)
    _log(state).error = None


def _restore_draft(
    state: Any,
    values: dict[str, Any],
    event_presets: tuple[str, ...],
    user_price_curve: pd.DataFrame | None = None,
    blackout_windows: pd.DataFrame | None = None,
) -> None:
    state["draft_user_price_curve"] = user_price_curve
    state["draft_blackout_windows"] = (
        EMPTY_BLACKOUT_WINDOWS if blackout_windows is None else blackout_windows
    )
    state["draft_blackout_error"] = None
    _set_draft(state, dict(values), event_presets)
    sync_widgets_from_draft(state)
    _log(state).error = None


def reset_to_defaults(state: Any) -> None:
    """Dialog button: restore the default values and no events. Keeps the model and result.

    The model is a separate run-bar choice, so resetting assumptions does not
    undo it, and Reset never clears or reruns a result (AGENTS.md). The
    default events table is empty (trading contract v1 §2.1), and the
    default blackout table has no rows (§10.5b).
    """

    _restore_draft(state, default_values(), ())


def reset_to_active_run(state: Any) -> None:
    """Dialog button: make the draft match the active run again (model too).

    Undoing every edit this way returns the run bar to "Current" (on the day
    of the run; after midnight the start date still differs).
    """

    record = latest_run(state)
    if record is None:
        return
    state["model_mode"] = record.model
    _restore_draft(
        state, record.values, record.event_presets, record.user_price_curve, record.blackout_windows
    )


def model_changed(state: Any) -> None:
    """Model control callback: the draft now differs; nothing is run.

    Values are left as they are: both explicit models share one run shape,
    so switching model changes only which model the next Run calls. The
    newsvendor/short-notice-event conflict (§10.4) reads the model through
    ``applied_presets``, so it is re-checked here too.
    """

    _recompute_draft_error(state)
    _log(state).error = None


def switch_to_action_model(state: Any) -> None:
    """ "Switch to smart charging" button: change the draft model only.

    Run stays an explicit click (design 4.3), so this never runs anything.
    """

    state["model_mode"] = MODEL_ACTION
    model_changed(state)


# --- Running a model --------------------------------------------------------


def run_model(
    model: str,
    values: dict[str, Any],
    start_local_date: date,
    event_presets: tuple[str, ...] = (),
    user_price_curve: pd.DataFrame | None = None,
    blackout_windows: pd.DataFrame | None = None,
) -> ForecastResult:
    """Run one full Monte Carlo forecast for the chosen model (``"action"`` or ``"no_action"``).

    Every input comes from ``model/assumptions.py`` with the draft's values
    applied, and the result carries the records it ran with.
    ``event_presets`` are ids of ``assumptions.EVENT_PRESETS``; the model
    builds, validates and applies their events table (the no-action model
    has none). ``user_price_curve`` is the supplier's validated curve or
    ``None``. ``blackout_windows`` is the draft's ``availability.
    blackout_windows`` table or ``None`` (no blackout, §10.5b); the model
    re-validates it, so a draft the dialog already accepted is checked once
    more, not skipped.
    """

    if model not in MODELS:
        raise ValueError(f"unknown model: {model}")
    # Passed only when set, like run_full's runner call, so a stand-in for
    # the model written before the upload or the blackout table existed
    # keeps working.
    extra = {} if user_price_curve is None else {"user_price_curve": user_price_curve}
    if blackout_windows is not None and len(blackout_windows):
        extra["blackout_windows"] = blackout_windows
    return run_forecast_from_assumptions(
        start_local_date, model=model, values=values, event_presets=tuple(event_presets), **extra
    )


def request_run(state: Any) -> None:
    """Run button callback: record the click. The script runs the model later."""

    state["run_requested"] = True


def run_at_hosted_size(state: Any) -> None:
    """ "Run at the hosted size" button callback: shrink the draft to fit the cap, then run it.

    Only offered by the run bar after a hosted-demo size refusal
    (``RunLog.size_capped``; ``AXLE_MAX_EV_WEEKS``, ``ui/run_guard.py``).
    Reuses ``run_guard.hosted_run_size`` on the draft's own fleet size and
    weeks, so the button does exactly what its own label promises, then
    requests a run the same way the ordinary Run button does: a flag the
    script honours at the end of this rerun (AGENTS.md), never run
    synchronously inside a callback.
    """

    limit = run_guard.env_limit(run_guard.MAX_EV_WEEKS_ENV)
    if limit is None:
        return
    values = dict(state["draft_values"])
    values["vehicle_count"], values["evaluation_world_count"] = run_guard.hosted_run_size(
        values, limit
    )
    _set_draft(state, values)
    sync_widgets_from_draft(state)
    _log(state).error = None
    request_run(state)


def _format_value(value: object) -> str:
    # A string choice is stored as an id ("fixed_share"); no snake_case on
    # screen (copy rule), so show it as words.
    return f"{value:g}" if isinstance(value, float) else str(value).replace("_", " ")


def _value_changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Plain-English "label old → new unit" for each editable value that differs."""

    changes = []
    for name, record in _EDITABLE.items():
        old, new = before.get(name), after.get(name)
        if not same_value(old, new):
            unit_text = display_unit(name, record.unit)
            unit = f" {unit_text}" if unit_text else ""
            label = display_label(name, record.label)
            changes.append(f"{label} {_format_value(old)} → {_format_value(new)}{unit}")
    return changes


def run_label(
    number: int,
    model: str,
    values: dict[str, Any],
    previous: RunRecord | None = None,
    event_presets: tuple[str, ...] = (),
) -> str:
    """Label a run by how it differs from the previous kept run.

    Examples: "Run 3 · Home charging power 7 → 3.6 kW", "Run 4 · smart
    charging → no action", "Run 5 · Home charging power 7.0 → 3.6 kW (+1 more)",
    "Run 6 · events none → 2 presets". Events are compared as applied, so a
    no-action run never claims events it did not simulate.
    Why the previous run and not the defaults: Compare's default is previous
    against latest, so the label names exactly the change being compared
    (design 4.4). The first run has no previous run and is compared with the
    defaults and the default model instead ("base" when it matches them).
    """

    if previous is None:
        before_model, before, before_presets = MODEL_ACTION, default_values(), ()
    else:
        before_model, before = previous.model, previous.values
        before_presets = applied_presets(previous.model, previous.event_presets)
    changes = []
    if model != before_model:
        changes.append(f"{MODEL_LABELS[before_model].lower()} → {MODEL_LABELS[model].lower()}")
    after_presets = applied_presets(model, tuple(event_presets))
    if after_presets != before_presets:
        changes.append(
            f"events {_event_presets_short(before_presets)} → {_event_presets_short(after_presets)}"
        )
    changes.extend(_value_changes(before, values))
    if not changes:
        return f"Run {number} · {'base' if previous is None else 'same inputs'}"
    more = f" (+{len(changes) - 1} more)" if len(changes) > 1 else ""
    return f"Run {number} · {changes[0]}{more}"


def run_full(state: Any, runner: Runner = run_model) -> None:
    """Run the valid draft once and keep it; on failure keep the previous result."""

    # Read everything from session state before the model starts; after it
    # finishes, only the fetched RunLog is changed (see RunLog for why).
    if state.get("draft_error"):
        state["run_requested"] = False
        return
    values = dict(state["draft_values"])
    event_presets = tuple(state.get("draft_event_presets", ()))
    curve = state.get("draft_user_price_curve")
    blackout = state.get("draft_blackout_windows")
    start = state["draft_start_date"]
    model = state["model_mode"]
    log = _log(state)
    # The click is consumed here, just before the model, not when the bar is
    # drawn: a nav or lens click while the page draws stops the script before
    # this point, and the Run must then still happen on the next rerun rather
    # than be dropped silently. After this line no rerun can repeat it.
    state["run_requested"] = False
    # A hosted demo's limits (ui/run_guard.py; off unless the host sets them).
    # A refusal is shown like a failed run: the reason in the banner, the
    # previous result kept. Reset here so every path below (size refusal,
    # busy, failure, success) leaves size_capped correct for the run just
    # attempted, not a stale True from an earlier refused click.
    log.size_capped = False
    refusal = run_guard.size_refusal(values)
    # Fleet size and weeks only, plus the hosted-cap outcome, never the full
    # draft (anonymous usage logging, AXLE_USAGE_LOG=1; ui/usage_log.py).
    usage_log.log_event(
        state,
        "run_requested",
        fleet_size=values["vehicle_count"],
        weeks=values["evaluation_world_count"],
        refused=refusal is not None,
    )
    if refusal is not None:
        log.error = refusal
        log.size_capped = True
        return
    with run_guard.run_slot() as may_run:
        if not may_run:
            log.error = run_guard.BUSY_MESSAGE
            return
        started = time.monotonic()
        try:
            # Each extra is passed only when set, so a runner written before the
            # upload or the blackout table existed (the tests' fakes) keeps its
            # four-argument call.
            extra = {} if curve is None else {"user_price_curve": curve}
            if blackout is not None and len(blackout):
                extra["blackout_windows"] = blackout
            result = runner(model, values, start, event_presets, **extra)
        # Broad on purpose: whatever the model raises, the viewer sees the reason
        # and keeps the last good result (design section 3.1, "Failed" state).
        except Exception as error:
            log.error = str(error) or type(error).__name__
            return
    usage_log.log_event(state, "run_finished", seconds=round(time.monotonic() - started, 1))
    history = [replace(record, result=None) for record in log.history]
    number = history[-1].number + 1 if history else 1
    label = run_label(number, model, values, history[-1] if history else None, event_presets)
    if history and not same_curve(history[-1].user_price_curve, curve):
        # The curve is not an editable value, so run_label cannot see it;
        # name it rather than calling the run "same inputs".
        label = label.replace("same inputs", "supplier price curve changed")
    if history and not _same_blackout_windows(history[-1].blackout_windows, blackout):
        label = label.replace("same inputs", "blackout windows changed")
    history.append(_record(number, label, values, start, result, event_presets, curve, blackout))
    if len(history) > HISTORY_SIZE:
        log.notice = f"{history[0].label} dropped: Compare keeps the last three runs."
    log.history[:] = history[-HISTORY_SIZE:]
    log.error = None


# --- Result state ----------------------------------------------------------


def run_history(state: Any) -> list[RunRecord]:
    """Completed runs in this session, oldest first, at most three (for Compare)."""

    return list(_log(state).history)


def latest_run(state: Any) -> RunRecord | None:
    """The most recent completed run, whose result every page shows."""

    history = _log(state).history
    return history[-1] if history else None


def active_result(state: Any) -> ForecastResult | None:
    """The result every page shows: the latest completed run's, or ``None``."""

    record = latest_run(state)
    return record.result if record is not None else None


def unrun_changes(state: Any) -> list[str]:
    """Name each difference between the draft and the active run's inputs."""

    record = latest_run(state)
    if record is None:
        return []
    changes = ["model"] if state.get("model_mode") != record.model else []
    if state.get("draft_start_date") != record.start_local_date:
        changes.append("start date")
    if tuple(state.get("draft_event_presets", ())) != record.event_presets:
        changes.append("events")
    if not same_curve(state.get("draft_user_price_curve"), record.user_price_curve):
        changes.append("supplier price curve")
    if not _same_blackout_windows(state.get("draft_blackout_windows"), record.blackout_windows):
        changes.append("blackout windows")
    draft = state.get("draft_values", {})
    changes.extend(
        display_label(name, _EDITABLE[name].label)
        for name in _EDITABLE
        if not same_value(draft.get(name), record.values.get(name))
    )
    return changes


_EVIDENCE_BADGES = {
    "illustrative": (
        "Illustrative",
        "Every value is an illustrative assumption. This model draws no synthetic prices.",
    ),
    # The no-action model has no price channel at all, so it never draws the
    # synthetic day-ahead and forecast-error paths the smart charging model
    # uses (decision 0004 items 14, 48). The badge names what is synthetic
    # rather than leaving a bare mismatch between the two models' words
    # (goal review V6); the tooltip carries the longer sentence.
    "illustrative_synthetic": (
        "Illustrative · synthetic prices",
        "Illustrative assumptions and synthetic day-ahead prices, not real market data.",
    ),
}


def result_identity(record: RunRecord) -> str:
    """One short line naming a run.

    For example ``Run 2 · 1,000 EVs · 100 weeks · seed 42 · from Tue 29 Sep``,
    with `` · 3 event presets`` added when the run applied scripted events.

    At most about 75 characters, inside the polish plan's 80 (G7): the
    evidence class is a separate badge (``evidence_badge``), not more words
    on this line, and events are a count; their names are in the run label
    and Compare.
    """

    weeks = record.world_count
    start = record.start_local_date
    presets = applied_presets(record.model, record.event_presets)
    events = f" · {_count(len(presets), 'event preset')}" if presets else ""
    precomputed = " · precomputed" if record.number == 0 else ""
    return (
        f"Run {record.number}{precomputed} · {record.vehicle_count:,} EVs · "
        f"{weeks} {'week' if weeks == 1 else 'weeks'} · seed {record.seed} · from "
        f"{start:%a} {start.day} {start:%b}{events}"
    )


def evidence_badge(record: RunRecord) -> tuple[str, str]:
    """``(badge text, tooltip)`` for the run's evidence class (``result.evidence_kind``)."""

    raw = str(record.result.evidence_kind) if record.result is not None else "illustrative"
    fallback = raw.replace("_", ", ").capitalize()
    text, tooltip = _EVIDENCE_BADGES.get(raw, (fallback, fallback))
    if record.number == 0:
        tooltip += (
            " Precomputed: saved earlier from this same model code and loaded when the app opened."
        )
    return text, tooltip


def _count(number: int, word: str) -> str:
    return f"{number:,} {word}" if number == 1 else f"{number:,} {word}s"


@dataclass(frozen=True)
class RunStatus:
    """What the run bar shows: a state name, a worded chip and a caption.

    ``chip`` is ``None`` only before the first run. Colour is a second cue;
    the chip's word always carries the state (design section 3.1).
    """

    state: str
    chip: str | None
    colour: str
    caption: str


def run_status(state: Any, *, running: bool = False) -> RunStatus:
    """Return the run bar's status for the six states in design section 3.1.

    ``running`` is True only in the script run that executes the model.
    """

    record = latest_run(state)
    if running:
        values = state["draft_values"]
        return RunStatus(
            "running",
            "Running",
            "gray",
            f"{values['vehicle_count']:,} EVs × "
            f"{_count(values['evaluation_world_count'], 'simulated week')}…",
        )
    still_showing = f"Still showing {_run_name(record)}" if record else "No earlier result"
    if state.get("draft_error"):
        return RunStatus(
            "invalid",
            "Draft invalid",
            "red",
            f"Fix the draft in Edit assumptions: {state['draft_error']}.",
        )
    if _log(state).error:
        return RunStatus("failed", "Run failed", "red", still_showing)
    if record is None:
        return RunStatus("no_result", None, "gray", "No result yet")
    changes = unrun_changes(state)
    if changes:
        return RunStatus(
            "stale",
            f"Stale · {_count(len(changes), 'change')}",
            "orange",
            f"Showing {_run_name(record)}; draft differs",
        )
    return RunStatus("current", "Current", "green", result_identity(record))


# --- Run bar, dialog and empty state -----------------------------------------


def render_run_button(
    st_module: Any, *, key: str, disabled: bool = False, width: str = "stretch"
) -> None:
    """The one primary action. Used by the bar (``width="content"``) and the empty state."""

    state = st_module.session_state
    st_module.button(
        "Run simulation",
        key=key,
        type="primary",
        width=width,
        # Disabled during a run too, so a second click cannot queue another run.
        disabled=disabled or bool(state.get("draft_error") or state.get("run_in_progress")),
        on_click=request_run,
        args=(state,),
    )


@st.dialog("Edit assumptions", width="large", on_dismiss="rerun")
def edit_assumptions_dialog() -> None:
    """The Edit assumptions dialog (design section 3.3).

    Dialogs rerun as fragments, so editing here never reruns the page and
    never starts a run. Closing reruns the app so the run bar updates.
    """

    state = st.session_state
    record = latest_run(state)
    changes = unrun_changes(state)
    summary = (
        f"{_count(len(changes), 'change')} from Run {record.number}"
        if record
        else "No run yet. The first Run uses these values."
    )
    summary_column, active_column, defaults_column = st.columns(
        [2, 1.2, 1.2], vertical_alignment="center"
    )
    summary_column.caption(summary)
    active_column.button(
        "Reset to active run",
        key="reset-to-active",
        disabled=record is None,
        on_click=reset_to_active_run,
        args=(state,),
        width="stretch",
    )
    defaults_column.button(
        "Reset to defaults",
        key="reset-to-defaults",
        on_click=reset_to_defaults,
        args=(state,),
        width="stretch",
    )
    if state.get("draft_error"):
        st.error(f"Draft invalid: {state['draft_error']}.")

    # Imported here, not at module top: the editor imports this module's
    # callbacks, so a top-level import would be circular.
    from .views.parameters import render_assumptions_editor

    render_assumptions_editor(st)

    st.divider()
    _, close_column, run_column = st.columns([2, 1, 1.4])
    if close_column.button("Close", key="close-assumptions", width="stretch"):
        st.rerun()
    # A full-app rerun closes the dialog; the Run click's flag is then honoured
    # at the end of that script run like any other Run click.
    run_column.button(
        "Run simulation",
        key="run-from-dialog",
        type="primary",
        width="stretch",
        disabled=bool(state.get("draft_error")),
        on_click=request_run,
        args=(state,),
    )
    if state.get("run_requested"):
        st.rerun()


def render_edit_button(
    st_module: Any, *, key: str, disabled: bool = False, width: str = "stretch"
) -> None:
    """A secondary "Edit assumptions" button that opens the dialog."""

    if st_module.button("Edit assumptions", key=key, width=width, disabled=disabled):
        usage_log.log_event(st_module.session_state, "edit_assumptions_opened")
        edit_assumptions_dialog()


def render_run_bar(st_module: Any = st) -> Any | None:
    """Draw the run bar at the top of every page.

    Returns a status placeholder when a Run was clicked, so the entry script
    can call ``execute_requested_run`` after the page has been drawn;
    otherwise returns ``None``.
    """

    state = st_module.session_state
    initialise_session(state)
    refresh_start_date(state)
    sync_widgets_from_draft(state)
    # A Run click is the only route to ``run_full`` (AGENTS.md). The flag is
    # cleared by ``run_full`` itself; an invalid draft drops the click here so
    # it cannot fire later once the draft becomes valid.
    running = bool(state.get("run_requested")) and not state.get("draft_error")
    if not running:
        state["run_requested"] = False
    # Read by every Run button drawn later in this script run (bar, empty state).
    state["run_in_progress"] = running
    log = _log(state)
    if log.notice:
        st_module.toast(log.notice)
        log.notice = None

    # One row, about 48 px at desktop width (polish plan G9): a small "Model"
    # eyebrow beside the collapsed-label model control, the state chip, the
    # short run identity, the evidence badge, the inline running status, then
    # Edit assumptions and Run pushed to the right. A horizontal container
    # rather than fixed columns, so each item takes its own width and the row
    # wraps (model first, Run last) at 390 px instead of stacking four
    # full-width blocks. Run stays in the main area, never behind a menu.
    record = latest_run(state)
    status = run_status(state, running=running)
    status_slot = None
    with st_module.container(horizontal=True, vertical_alignment="center", gap="small"):
        st_module.caption("Model", width="content")
        st_module.segmented_control(
            "Model",
            options=MODELS,
            format_func=MODEL_LABELS.__getitem__,
            key="model_mode",
            required=True,
            disabled=running,
            on_change=model_changed,
            args=(state,),
            label_visibility="collapsed",
        )
        if status.chip:
            st_module.badge(status.chip, color=status.colour)
        st_module.caption(status.caption, width="content")
        if record is not None and status.state in ("current", "stale"):
            text, tooltip = evidence_badge(record)
            st_module.badge(text, color="gray", help=tooltip)
        # The running spinner draws here, in the row, not as a block below it.
        status_slot = st_module.empty()
        st_module.space("stretch")
        render_edit_button(st_module, key="edit-assumptions", disabled=running, width="content")
        render_run_button(st_module, key="run-simulation", disabled=running, width="content")

    # Only a failed run earns a banner: its reason must be read (design 0).
    if status.state == "failed":
        st_module.error(f"Run failed: {log.error}")
        if log.size_capped:
            _render_hosted_size_button(st_module, state)
    # Shown under the run bar's Run button whenever the host set a cap, not
    # only after a refusal: a visitor should know the limit before hitting it,
    # and see where the full-size result already lives (Run 0, the
    # precomputed default) rather than only being told after a failed click.
    cap = run_guard.env_limit(run_guard.MAX_EV_WEEKS_ENV)
    if cap is not None:
        st_module.caption(
            f"Hosted demo: a live run can use up to {cap:,} EV-weeks (fleet size × weeks). "
            "Run 0 already holds the full-size result."
        )
    return status_slot if running else None


def _render_hosted_size_button(st_module: Any, state: Any) -> None:
    """The "Run at the hosted size" recovery button shown after a size refusal.

    Quotes the exact fleet size and weeks ``run_at_hosted_size`` will set, so
    the button's label is a promise it then keeps in one click.
    """

    limit = run_guard.env_limit(run_guard.MAX_EV_WEEKS_ENV)
    if limit is None:
        return
    vehicles, weeks = run_guard.hosted_run_size(state["draft_values"], limit)
    st_module.button(
        f"Run at the hosted size ({vehicles:,} EVs × {weeks:,} weeks)",
        key="run-hosted-size",
        on_click=run_at_hosted_size,
        args=(state,),
    )


def execute_requested_run(st_module: Any, status_slot: Any) -> None:
    """Run the model after the page is drawn, then rerun to show the result.

    Called at the end of the entry script only when ``render_run_bar``
    returned a placeholder, which happens only after a Run click.
    """

    state = st_module.session_state
    # A compact spinner in the run bar's own row; the chip and caption beside
    # it already say what is running.
    with status_slot, st_module.spinner("Running…", show_time=True):
        run_full(state)
    state["run_in_progress"] = False
    st_module.rerun()


_APP_INTRO = (
    "This app forecasts how much of a fleet's EV charging can move to cheaper, off-peak "
    "half-hours, and what that shift is worth once the risk to the driver is counted. "
    "It simulates every EV half-hour by half-hour, across many possible weeks. "
    "The EVs are drawn from the six driver archetypes in Axle's own workbook. "
    "Click Run simulation to forecast the default fleet. The six archetypes and their "
    "source shares are listed below while it runs."
)
"""The "what this app shows" intro for the empty state (goal review action 9,
design section 3.2)."""


def _archetype_summary() -> str:
    """The six archetypes with their source population shares, e.g. "Average (UK) 40%, ...".

    Read from ``model.assumptions`` rather than hard-coded, so a change to the
    cohort sheet's shares shows up here without a shell edit (decision 0004
    item 21: the dialog and this line must not drift from the model).
    """

    # COHORT_SHARES[cohort_id].value is the fixed source default (a percent),
    # not the editable draft: decision 0004 item 54 moved the field out of
    # COHORTS so it could become an editable illustrative override, but this
    # caption is explicitly labelled "(Source)", so it always shows the
    # default, whatever the dialog's current draft says.
    shares = ", ".join(
        f"{assumptions.COHORT_SOURCE_NAMES[cohort_id]} "
        f"{assumptions.COHORT_SHARES[cohort_id].value / 100.0:.0%}"
        for cohort_id in assumptions.COHORT_SOURCE_NAMES
    )
    return f"{shares} (Source)."


def render_empty_state(st_module: Any) -> None:
    """Shown in the page body before the first run (design section 3.2).

    Goal review action 9: the default 1,000 EV x 100 week run takes a while,
    so there is something to read besides a spinner: what the app shows, and
    the six archetypes it is about to simulate. The quoted time (goal review
    N15's "60-70 s", widened to "70-90 s") went stale when the plan-status
    wiring lane made the default run heavier (final critique B-14: measured
    3 min 0 s-3 min 50 s on 29 Sep, against a promised 70-90 s); it now reads
    off ``DEFAULT_RUN_MINUTES_ESTIMATE`` so the words move when that constant
    is re-timed, rather than needing a second, easily missed edit here. It
    still says a busy machine runs slower still, rather than a fixed figure
    that reads as a promise on hardware this run has never seen.
    """

    state = st_module.session_state
    values = state.get("draft_values")
    # Quote the draft's own size (polish plan G7), so a presenter who shrank
    # the fleet sees what the next Run will actually simulate; the timing is
    # then stated for the default size it was measured at.
    size = (
        f"{values['vehicle_count']:,} simulated EV drivers over "
        f"{_count(values['evaluation_world_count'], 'possible week')}"
        if values
        else "the draft"
    )
    st_module.markdown(
        f"Nothing to show yet. Run simulation to forecast {size}. "
        f"The default takes about {DEFAULT_RUN_MINUTES_ESTIMATE} minutes at 1,000 EVs × 100 "
        "weeks, longer if the machine is busy. Reloading the browser clears every result."
    )
    _, button_column, _ = st_module.columns([2, 1.2, 2])
    with button_column:
        render_run_button(st_module, key="run-simulation-empty")
    st_module.divider()
    st_module.markdown(_APP_INTRO)
    # Body text, not a caption: six names and shares run past the 140
    # character caption limit (polish plan G7).
    st_module.markdown(_archetype_summary())


__all__ = [
    "DEFAULT_RUN_MINUTES_ESTIMATE",
    "EVENT_PRESET_IDS",
    "EVENT_WIDGET_PREFIX",
    "HISTORY_SIZE",
    "MODELS",
    "MODEL_ACTION",
    "MODEL_LABELS",
    "MODEL_NO_ACTION",
    "RunLog",
    "RunRecord",
    "RunStatus",
    "active_result",
    "applied_presets",
    "commit_assumption_edit",
    "commit_event_preset",
    "default_values",
    "display_label",
    "display_unit",
    "edit_assumptions_dialog",
    "event_label",
    "event_presets_error",
    "event_presets_text",
    "evidence_badge",
    "execute_requested_run",
    "initialise_session",
    "latest_run",
    "render_edit_button",
    "render_empty_state",
    "render_run_bar",
    "reset_to_active_run",
    "reset_to_defaults",
    "run_at_hosted_size",
    "run_full",
    "run_history",
    "run_model",
    "run_status",
    "switch_to_action_model",
    "sync_widgets_from_draft",
    "unrun_changes",
]
