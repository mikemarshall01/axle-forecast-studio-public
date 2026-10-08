"""Body of the Edit assumptions dialog, generated from ``model/assumptions.py``.

What this owns: one pill per assumption group (design section 3.3, polish
plan G9), one number input per editable record of the chosen group, and a
read-only list of the fixed records in that group. Nothing here states a
value, a bound or an explanation: the label, unit, bounds, meaning, source
and "affects" text all come from the records, so the dialog cannot drift
from the model (decision 0004 item 21). The Fleet group's six cohort-mix
fields (decision 0004 item 54) are one exception to "one number input per
editable record, rendered generically": "must sum to 100%" is a rule across
all six, not any one of them, so they are rendered together with one shared
running total instead of through the per-record loop the other groups use.

The Events group (decision 0004 items 55, 56 and 58; trading contract v1
§2.2 and §9.6) is the other exception: it is not an assumption group but
one checkbox per scripted event preset in ``assumptions.EVENT_PRESETS``,
each with a one-line description written from the preset's own values, so
the text cannot drift from what the model runs. The chosen presets are
part of the draft (``draft_event_presets``) and are validated by the
model's own event rules before Run is enabled.

How it fits: the widgets write to ``assumption::<name>`` keys and their
callback copies them into the draft owned by ``run_controller``; the
controller writes the draft back into the keys on every full rerun, and
this module does so again before each field is drawn, so a widget never
shows a value the next Run will not use.  Editing never runs
the model (dialogs rerun as fragments; only the Run button starts a run).
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import pandas as pd
from streamlit import column_config

from axle_studio.model import assumptions

from ..run_controller import (
    EVENT_PRESET_IDS,
    EVENT_WIDGET_PREFIX,
    MODEL_ACTION,
    WIDGET_PREFIX,
    applied_presets,
    commit_assumption_edit,
    commit_event_preset,
    commitment_rule_error,
    display_label,
    display_unit,
    event_presets_text,
    is_switch,
    latest_run,
    same_value,
    set_blackout_windows,
    widget_value,
)

_EVIDENCE_LABELS = {"source": "Source", "illustrative": "Illustrative", "synthetic": "Synthetic"}
_EVIDENCE_COLOURS = {"source": "blue", "illustrative": "gray", "synthetic": "violet"}
"""Badge colour per evidence class; the word always carries the meaning."""


def _step(record: assumptions.Assumption) -> int | float:
    """A widget step about one hundredth of the record's range, as a round number.

    Whole-number records (fleet size, worlds, seed) step by 1.  For the rest,
    a power of ten near range/100 gives 0.001 for the weather sensitivity,
    0.01 for efficiencies and the public rate and 1 for price levels, so every
    field can reach its default and its bounds exactly.
    """

    if isinstance(record.value, int):
        return 1
    low, high = record.bounds
    return 10.0 ** math.floor(math.log10((high - low) / 100.0))


def _format_number(value: Any) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "unset"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _may_be_unset(record: assumptions.Assumption) -> bool:
    """A record whose default is NaN: an optional commercial term (decision 0003)."""

    return isinstance(record.value, float) and math.isnan(record.value)


def _help(record: assumptions.Assumption) -> str:
    """Tooltip text: meaning, source and effect (design section 3.3)."""

    low, high = record.bounds
    return (
        f"{record.meaning}\n\nSource: {record.source} "
        f"({_EVIDENCE_LABELS[record.evidence].lower()}).\n\n"
        f"What it changes: {record.affects}\n\nAllowed range: {low:g} to {high:g}"
        + ("; leave it empty to keep the term unset." if _may_be_unset(record) else ".")
    )


def _render_field(st: Any, record: assumptions.Assumption, active: dict[str, Any] | None) -> None:
    """One editable record: number input, then its error or its unrun change."""

    state = st.session_state
    low, high = record.bounds
    whole = isinstance(record.value, int)
    step = _step(record)
    decimals = 0 if whole else max(0, -math.floor(math.log10(step)))
    # Plain display names for the Simulation fields (run_controller owns them,
    # so run labels and Compare say the same thing).
    label = display_label(record.name, record.label)
    unit_text = display_unit(record.name, record.unit)
    unit = f" ({unit_text})" if unit_text else ""
    # Refill the key from the draft: with group pills, this widget may not
    # have been on screen last run, and Streamlit dropped its value then.
    # ``widget_value`` turns an unset NaN into an empty field and a 0/1
    # switch into a toggle's True/False.
    state[WIDGET_PREFIX + record.name] = widget_value(
        record.name, state["draft_values"][record.name]
    )
    # Input and a small evidence badge on one line (lead decision on the
    # ecacff9 review): the evidence class is visible without the tooltip.
    # A horizontal container keeps the badge beside the input at 390 px,
    # where columns would stack it underneath.
    row = st.container(horizontal=True, vertical_alignment="bottom", gap="small")
    if is_switch(record.name):
        # Lead ruling (supplier contract v1 §8): a 0/1 switch is a toggle,
        # so no value other than 0 or 1 can be entered.
        row.toggle(
            label,
            key=WIDGET_PREFIX + record.name,
            help=f"{_help(record)}\n\nOff is 0, on is 1: {record.unit}.",
            on_change=commit_assumption_edit,
            args=(state, record.name),
        )
    else:
        row.number_input(
            f"{label}{unit}",
            min_value=int(low) if whole else float(low),
            max_value=int(high) if whole else float(high),
            step=step,
            format="%d" if whole else f"%.{decimals}f",
            key=WIDGET_PREFIX + record.name,
            help=_help(record),
            on_change=commit_assumption_edit,
            args=(state, record.name),
            # An optional term may be left empty: "unset", never £0.
            placeholder="unset" if _may_be_unset(record) else None,
        )
    row.badge(_EVIDENCE_LABELS[record.evidence], color=_EVIDENCE_COLOURS[record.evidence])
    error = state.get("draft_errors", {}).get(record.name)
    draft_value = state["draft_values"][record.name]
    if error:
        st.caption(f":red[{label} {error}.]")
        return
    if is_switch(record.name):
        # Say what off and on mean, from the record's own unit text.
        # A plain "1 on, 0 off" says nothing a toggle does not already show,
        # so only the rest of it (if any) is captioned.
        meaning = record.unit.removeprefix("switch").strip(" ()").replace("GBP", "£")
        meaning = meaning.removeprefix("1 on, 0 off").removeprefix(";").strip()
        meaning = meaning.replace("0 =", "Off (0):").replace(", 1 =", "; on (1):")
        if meaning:
            st.caption(meaning[0].upper() + meaning[1:] + ".")
    if _may_be_unset(record) and math.isnan(draft_value):
        st.caption("Unset: figures that need it show Unavailable, never £0.")
    if active is not None and not same_value(active.get(record.name), draft_value):
        # Shown only on changed fields, so an unchanged dialog stays quiet.
        run = latest_run(state)
        st.caption(
            f"Draft {_format_number(draft_value)} · Run {run.number} used "
            f"{_format_number(active.get(record.name))}"
        )


def _render_cohort_shares(
    st: Any, records: tuple[assumptions.Assumption, ...], active: dict[str, Any] | None
) -> None:
    """The six cohort-mix fields together, with one shared "sum to 100%" total.

    Drawn as a group rather than through ``_render_field`` (decision 0004
    item 54): no single field owns "the six must sum to 100%", so
    ``assumptions.validation_errors`` attaches that message to only the first
    of the six names, and the group shows one running total and one message
    here instead of the same red caption repeated six times.
    """

    state = st.session_state
    st.caption(
        "The fleet is drawn from these six archetype shares. Each starts at its source share. "
        "They must sum to 100% before Run is enabled.",
        help="An editable, illustrative override of the shares in Axle's workbook.",
    )
    for record in records:
        low, high = record.bounds
        step = _step(record)
        decimals = max(0, -math.floor(math.log10(step)))
        cohort_label = record.label.split(": ", 1)[-1]  # strip the repeated "Population share:"
        # Refill from the draft (see _render_field) and commit only this
        # field, so hidden-group keys never overwrite the draft.
        state[WIDGET_PREFIX + record.name] = state["draft_values"][record.name]
        row = st.container(horizontal=True, vertical_alignment="bottom", gap="small")
        row.number_input(
            f"{cohort_label} (%)",
            min_value=float(low),
            max_value=float(high),
            step=step,
            format=f"%.{decimals}f",
            key=WIDGET_PREFIX + record.name,
            help=_help(record),
            on_change=commit_assumption_edit,
            args=(state, record.name),
        )
        row.badge(_EVIDENCE_LABELS[record.evidence], color=_EVIDENCE_COLOURS[record.evidence])
        st.caption(f"Source default: {_format_number(record.value)}%")
        draft_value = state["draft_values"][record.name]
        if active is not None and active.get(record.name) != draft_value:
            run = latest_run(state)
            st.caption(
                f"Draft {_format_number(draft_value)} · Run {run.number} used "
                f"{_format_number(active.get(record.name))}"
            )
    total = math.fsum(state["draft_values"][record.name] for record in records)
    message = state.get("draft_errors", {}).get(assumptions.COHORT_SHARE_NAMES[0])
    if message:
        st.caption(f":red[Total {_format_number(total)}%: {message}.]")
    else:
        st.caption(f"Total {_format_number(total)}%")


def _render_manufacturer_shares(
    st: Any, records: tuple[assumptions.Assumption, ...], active: dict[str, Any] | None
) -> None:
    """The four manufacturer-share fields together, with one shared "sum to 1" total.

    Mirrors ``_render_cohort_shares`` (trading contract v1 §10.1d): no
    single share owns "the four must sum to 1", so
    ``assumptions.validation_errors`` attaches that message to only the
    first of the four names.
    """

    state = st.session_state
    st.caption(
        "The fleet is split across these four illustrative charger makers. They must sum "
        "to 1 before Run is enabled.",
        help=(
            "Four made-up makers with no real brand behind them, like the zones. Each has its "
            "own response rate and cloud outage chance, so a maker's outage moves all of its "
            "EVs together."
        ),
    )
    for record in records:
        low, high = record.bounds
        step = _step(record)
        decimals = max(0, -math.floor(math.log10(step)))
        maker_label = record.label.removeprefix("Share of EVs with ")
        state[WIDGET_PREFIX + record.name] = state["draft_values"][record.name]
        row = st.container(horizontal=True, vertical_alignment="bottom", gap="small")
        row.number_input(
            f"{maker_label} (fraction)",
            min_value=float(low),
            max_value=float(high),
            step=step,
            format=f"%.{decimals}f",
            key=WIDGET_PREFIX + record.name,
            help=_help(record),
            on_change=commit_assumption_edit,
            args=(state, record.name),
        )
        row.badge(_EVIDENCE_LABELS[record.evidence], color=_EVIDENCE_COLOURS[record.evidence])
        st.caption(f"Source default: {_format_number(record.value)}")
        draft_value = state["draft_values"][record.name]
        if active is not None and active.get(record.name) != draft_value:
            run = latest_run(state)
            st.caption(
                f"Draft {_format_number(draft_value)} · Run {run.number} used "
                f"{_format_number(active.get(record.name))}"
            )
    total = math.fsum(state["draft_values"][record.name] for record in records)
    message = state.get("draft_errors", {}).get(assumptions.MANUFACTURER_SHARE_NAMES[0])
    if message:
        st.caption(f":red[Total {_format_number(total)}: {message}.]")
    else:
        st.caption(f"Total {_format_number(total)}")


_COMMITMENT_RULE_LABELS = {"fixed_share": "Fixed share", "newsvendor": "Newsvendor"}
"""On-screen names for ``assumptions.COMMITMENT_RULES`` (trading contract v1 §10.4)."""


def _render_commitment_rule(
    st: Any, record: assumptions.Assumption, active: dict[str, Any] | None
) -> None:
    """The commitment-rule switch: a two-option control, not a generic number field.

    Trading contract v1 §10.4, §10.8. ``record.bounds`` is ``None`` (it is
    checked against ``assumptions.COMMITMENT_RULES``, not a numeric range),
    so this does not go through ``_render_field``. Newsvendor is blocked
    with a named reason when the draft has only one simulated week or a
    short-notice scripted event selected (overnight review log, 30 Sep
    2026); the reason also joins the dialog's own error banner.
    """

    state = st.session_state
    state[WIDGET_PREFIX + record.name] = state["draft_values"][record.name]
    row = st.container(horizontal=True, vertical_alignment="bottom", gap="small")
    row.segmented_control(
        display_label(record.name, record.label),
        options=list(_COMMITMENT_RULE_LABELS),
        format_func=_COMMITMENT_RULE_LABELS.__getitem__,
        required=True,
        key=WIDGET_PREFIX + record.name,
        help=f"{record.meaning}\n\nSource: {record.source}.",
        on_change=commit_assumption_edit,
        args=(state, record.name),
    )
    row.badge(_EVIDENCE_LABELS[record.evidence], color=_EVIDENCE_COLOURS[record.evidence])
    draft_value = state["draft_values"][record.name]
    conflict = commitment_rule_error(
        draft_value,
        int(state["draft_values"].get("evaluation_world_count", 0)),
        applied_presets(
            state.get("model_mode", MODEL_ACTION), state.get("draft_event_presets", ())
        ),
    )
    if conflict:
        st.caption(f":red[Newsvendor {conflict}.]")
    if active is not None and active.get(record.name) != draft_value:
        run = latest_run(state)
        active_label = _COMMITMENT_RULE_LABELS.get(active.get(record.name), active.get(record.name))
        st.caption(
            f"Draft {_COMMITMENT_RULE_LABELS[draft_value]} · Run {run.number} used {active_label}"
        )


def _render_blackout_windows(st: Any) -> None:
    """The blackout-windows table: an editable list of daily London windows.

    Trading contract v1 §10.5b. Validated on every edit by
    ``run_controller.set_blackout_windows`` (``availability.
    validate_blackout_windows``): a window overlapping another, an
    off-half-hour start or a duration that is not a multiple of 30 minutes
    between 30 and 720 names the row and rule, joins the dialog's error
    banner and blocks Run, exactly as an invalid event selection does.
    """

    state = st.session_state
    st.caption(
        "No charging moves into or out of these half-hours. "
        'Start on a half-hour ("HH:MM"), duration a multiple of 30 minutes; no rows means none.',
        help=(
            "Smart plans leave charging where it would have been in these windows, and the "
            "Firm MW lens offers no flexibility in them."
        ),
    )
    edited = st.data_editor(
        state["draft_blackout_windows"],
        column_config={
            "start_local_time": column_config.TextColumn(
                "Start (London)", help='"HH:MM" on a half-hour, for example "23:00"'
            ),
            "duration_minutes": column_config.NumberColumn(
                "Duration (minutes)", min_value=30, max_value=720, step=30
            ),
        },
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key="blackout-windows-editor",
    )
    if edited is not None and not edited.equals(state["draft_blackout_windows"]):
        set_blackout_windows(state, edited)
    error = state.get("draft_blackout_error")
    if error:
        st.caption(f":red[{error}.]")


def _archetype_label(name: str) -> str:
    """Which archetype a record belongs to, or "All" for a shared value (as ``methods``)."""

    for segment in name.split("."):
        label = assumptions.COHORT_SOURCE_NAMES.get(segment)
        if label is not None:
            return label
    return "All"


def _fixed_records_table(records: tuple[assumptions.Assumption, ...]) -> pd.DataFrame:
    table = pd.DataFrame(
        [
            {
                "Archetype": _archetype_label(record.name),
                "Label": record.label,
                "Value": _format_number(record.value),
                "Unit": record.unit,
                "Evidence": _EVIDENCE_LABELS[record.evidence],
            }
            for record in records
        ]
    )
    # Per-archetype records share one label ("Battery capacity" six times), so
    # the archetype is what tells the rows apart; a group with only shared
    # values keeps the four columns it always had.
    if table["Archetype"].eq("All").all():
        return table.drop(columns="Archetype")
    return table


EVENTS_GROUP = "Events"
"""The dialog's last pill: scripted event presets rather than assumption records."""

# What each event type does, in plain words; the numbers are added from the
# preset itself (``preset_description``).
_EVENT_TYPE_TEXT = {
    "price_shock_known": "Price shock known a day ahead",
    "price_shock_surprise": "Surprise price shock",
    "turn_down": "Grid asks EVs to use less",
    "turn_up": "Grid asks EVs to use more",
    "control_outage": "Chargers ignore their plans",
}


def _clock(night_date: date, start_local_time: str) -> tuple[date, str]:
    """The London day and "HH:MM" a preset's window starts on.

    Contract §2.1: a start at or after 12:00 falls on the night's date
    ``D_n``, an earlier one on the next day, so a night runs noon to noon.
    """

    hour = int(start_local_time[:2])
    return (night_date if hour >= 12 else night_date + timedelta(days=1)), start_local_time


def _duration(minutes: int) -> str:
    hours = minutes / 60
    return f"{hours:g} h"


def preset_description(preset_id: str, start_local_date: date) -> str:
    """One plain line saying what preset ``preset_id`` does, from its own values.

    ``start_local_date`` is the draft's study start, so a night reads as a
    weekday ("Thu 1 Oct 16:30"). Every figure is the preset's illustrative
    value (contract §2.2, §9.6): none is a forecast, a DFS rate or a DNO
    tariff, and the line says "illustrative" wherever it quotes money.
    """

    preset = assumptions.EVENT_PRESETS[preset_id]
    kind = str(preset["event_type"])
    parts = [_EVENT_TYPE_TEXT[kind]]
    size = float(preset["size"])
    if kind in ("price_shock_known", "price_shock_surprise"):
        extra = "extra demand" if size > 0 else "surplus supply"
        parts.append(f"{format(abs(size), 'g')} GW of {extra}")
    elif kind == "control_outage":
        parts.append(f"for {size:.0%} of sessions plugging in")
    if preset["scope"] != "national":
        parts.append(f"in {assumptions.ZONE_LABELS[str(preset['scope'])]} only")
    if preset["notice"] == "short":
        notice = int(preset["notice_minutes"])
        parts.append("with no warning" if notice == 0 else f"known {notice} min ahead")
    if kind in ("turn_down", "turn_up"):
        parts.append(f"paid £{float(preset['payment_gbp_per_mwh']):g}/MWh (illustrative)")
    length = _duration(int(preset["duration_minutes"]))
    if preset_id == "cold_still_week":
        when = f"every evening at {preset['start_local_time']} for {length}"
    elif preset_id == "sunny_negative_weekend":
        when = f"Saturday and Sunday at {preset['start_local_time']} for {length}"
    else:
        night_date = start_local_date + timedelta(days=int(preset["night_index"]))
        day, clock = _clock(night_date, str(preset["start_local_time"]))
        when = f"{day:%a} {day.day} {day:%b} at {clock} for {length}"
    return f"{', '.join(parts)}; {when}."


def _render_events(st: Any) -> None:
    """The Events group: one checkbox per preset, its description, and any error.

    Checkboxes rather than a multiselect: each preset needs its own line of
    explanation, and the list is short and fixed. The checkbox keys are
    refilled from the draft before drawing, as the number inputs are.
    """

    state = st.session_state
    start = state["draft_start_date"]
    st.caption(
        "Scripted events laid on top of the random market: illustrative scenarios, the same "
        "in every simulated week. Smart charging model only."
    )
    chosen = state.get("draft_event_presets", ())
    for preset_id in EVENT_PRESET_IDS:
        key = EVENT_WIDGET_PREFIX + preset_id
        state[key] = preset_id in chosen
        st.checkbox(
            str(assumptions.EVENT_PRESETS[preset_id]["label"]),
            key=key,
            on_change=commit_event_preset,
            args=(state, preset_id),
        )
        st.caption(preset_description(preset_id, start))
    error = state.get("draft_event_error")
    if error:
        # The model's own message names rows and columns, so it goes in the
        # tooltip; the caption says what to do.
        st.caption(
            ":red[This event selection cannot run: untick one. Hover for the reason.]", help=error
        )
    if state.get("model_mode") != MODEL_ACTION and chosen:
        st.caption("The no-action model ignores events. Switch to smart charging to apply these.")
    run = latest_run(state)
    if run is not None and tuple(chosen) != run.event_presets:
        used = event_presets_text(applied_presets(run.model, run.event_presets))
        st.caption(f"Run {run.number} used: {used}")


GROUP_KEY = "assumptions-group"
"""Session-state key of the dialog's group pills."""

LAST_GROUP_KEY = "assumptions-group-last"
"""The group last chosen, kept after the dialog closes (see ``render_assumptions_editor``)."""


def _remember_group(state: Any) -> None:
    """Pill callback: keep the chosen group for the next time the dialog opens."""

    state[LAST_GROUP_KEY] = state[GROUP_KEY]


_FIRST_FIELDS = {"Simulation": ("vehicle_count", "evaluation_world_count", "seed")}
"""Fields shown first in a group, in this order (polish plan G9: fleet size,
then weeks, then seed, the three a first-time user changes); the rest of the
group follows in module order."""

_ADVANCED_PREFIXES = (
    "clock_t_df",
    "departure_scale_minutes.",
    "plug_in_scale_minutes.",
    "weekend_plug_in_scale_minutes.",
)
"""The plug-in and departure clock-spread records (Student-t degrees of
freedom and per-archetype scales). Fourteen fine-grained fields that few
viewers change, so they sit in an "Advanced" expander below the group's
everyday fields (polish plan G9) instead of ahead of home charging power."""


def _is_advanced(record: assumptions.Assumption) -> bool:
    return record.name.startswith(_ADVANCED_PREFIXES)


def _ordered(group: str, records: list[assumptions.Assumption]) -> list[assumptions.Assumption]:
    first = _FIRST_FIELDS.get(group, ())
    rank = {name: index for index, name in enumerate(first)}
    # Stable sort: named fields first in the stated order, the rest unchanged.
    return sorted(records, key=lambda record: rank.get(record.name, len(first)))


def render_assumptions_editor(st: Any) -> None:
    """Draw the dialog body: group pills, that group's editable fields, then fixed values.

    Pills rather than tabs (polish plan G9): seven tab labels overflowed the
    dialog at 390 px, and pills wrap. Only the chosen group's widgets exist,
    so each field's widget key is refilled from the draft before it is drawn
    (Streamlit drops the key of a widget that was not on screen); the draft
    stays the single owner of every value, as ``run_controller`` requires.
    """

    state = st.session_state
    st.caption(
        "Edits change the draft only. Nothing runs until you click Run simulation. "
        "Hover ⓘ for what a value means, its source and its effect."
    )
    run = latest_run(state)
    active = run.values if run is not None else None
    groups = [*assumptions.ASSUMPTION_GROUPS, EVENTS_GROUP]
    # The chosen group is remembered in LAST_GROUP_KEY (not only the widget
    # key, which Streamlit drops once the dialog closes), so reopening Edit
    # assumptions returns to the group last edited. The first opening shows
    # "Simulation": fleet size, simulated weeks and seed are what a
    # first-time user actually changes (goal review action 10).
    if state.get(GROUP_KEY) not in groups:
        state[GROUP_KEY] = state.get(LAST_GROUP_KEY, "Simulation")
    group = st.pills(
        "Group",
        options=groups,
        required=True,
        key=GROUP_KEY,
        label_visibility="collapsed",
        on_change=_remember_group,
        args=(state,),
    )
    group = group if group in groups else "Simulation"
    if group == EVENTS_GROUP:
        _render_events(st)
        return
    if group == "Fleet":
        # Supplier contract v1 §8, "your fleet" framing (item 65 (3)): the
        # fleet is the pre-sales estimator's input. Fleet size sits under
        # Simulation with weeks and seed, so the caption points there.
        st.caption(
            "Set your fleet: the archetype mix here, the size under Simulation. Every supplier "
            "and partner figure then answers for that fleet."
        )
    records = assumptions.records_in_group(group)
    editable = _ordered(group, [record for record in records if record.editable])
    # The six cohort-mix fields (decision 0004 item 54) and, in "Firm MW",
    # the four manufacturer-share fields (trading contract v1 §10.1d)
    # render as one set with a shared total, not through the per-record
    # loop below: see _render_cohort_shares and _render_manufacturer_shares.
    # The commitment rule (§10.4) is a two-option control, not a number
    # field, so it is also pulled out of that loop.
    shares = tuple(r for r in editable if r.name in assumptions.COHORT_SHARE_NAMES)
    maker_shares = tuple(r for r in editable if r.name in assumptions.MANUFACTURER_SHARE_NAMES)
    commitment = tuple(r for r in editable if r.name == "trading.commitment_rule")
    if shares:
        _render_cohort_shares(st, shares, active)
    if maker_shares:
        _render_manufacturer_shares(st, maker_shares, active)
    others = [record for record in editable if record not in shares + maker_shares + commitment]
    everyday = [record for record in others if not _is_advanced(record)]
    advanced = [record for record in others if _is_advanced(record)]
    for record in everyday:
        _render_field(st, record, active)
    if advanced:
        with st.expander(f"Advanced: plug-in and departure clock spread ({len(advanced)})"):
            for record in advanced:
                _render_field(st, record, active)
    if commitment:
        _render_commitment_rule(st, commitment[0], active)
    if group == "Firm MW":
        _render_blackout_windows(st)
    if not editable:
        st.caption("Nothing in this group is editable. The values below are fixed.")
    fixed = tuple(record for record in records if not record.editable)
    if fixed:
        with st.expander(f"Fixed values ({len(fixed)})"):
            st.dataframe(_fixed_records_table(fixed), hide_index=True, width="stretch")


__all__ = ["EVENTS_GROUP", "preset_description", "render_assumptions_editor"]
