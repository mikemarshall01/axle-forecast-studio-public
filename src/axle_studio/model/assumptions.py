"""The single source of truth for every value the explicit model uses.

What this owns: every active assumption as a labelled ``Assumption`` record
(value, unit, evidence kind, source, meaning, editable bounds), and the small
builders that turn those records, with any values edited in the dashboard,
into the inputs a run needs (decision 0004 item 21).  The explicit JSON
configs and their loaders were removed in task M3b; nothing else in the
package states a model value.  Modules that need a fixed value (the weather
reference temperature, the AR(1) price coefficient and the reporting
tolerance) read it from the records here, so changing a value
here changes the model.

How it fits: ``editable_defaults`` and ``resolve_values`` give the editable
values (the Edit assumptions dialog's draft); ``run_settings``,
``cohort_fixture`` and ``forecast_inputs`` turn them into the arguments of
``forecast.run_forecast``; ``result_assumptions`` returns the records a
result carries, with the run's actual values, so Compare and How it works
read exactly what the run used.

``Assumption`` matches ``docs/contracts/results-v2.md`` section 8 exactly.

Decision 0004 item 23 asks for code that explains its reasoning, so every
section below cites the workbook cell, decision item or code location a
value comes from, and ``MODEL_RULES`` cites the file:function that
implements each rule so a reader can check this module against the code.

Scope: the explicit model only (the default per decision 0004 item 9).
Legacy cohort/compact/synthetic-six-route values (decision 0004 item 18,
superseded) are not carried forward here.

Evidence-vocabulary note (a WHY, not a restatement): the contract fixes
``evidence`` to three tokens -- ``"source"``, ``"illustrative"``,
``"synthetic"`` -- narrower than the crosswalk's own five-way classification
(``supplied``, ``derived``, ``assumed``, ``technical``, ``synthetic``). This
module maps crosswalk ``supplied`` and ``derived`` onto ``"source"`` (a
directly-cited workbook cell, or plain arithmetic on one), and crosswalk
``assumed``/``technical`` onto ``"illustrative"`` (a modelling choice or
engineering constant with no source cell).  ``"synthetic"`` is unchanged,
for the invented price and weather scenarios. Wherever collapsing to three
tokens would lose the finer crosswalk kind, that kind is kept as a short
prefix in the ``source`` text instead (for example ``"derived: ..."`` or
``"technical model constant; ..."``).  A source-traced default that the
dashboard lets you edit (home charging power, and the cohort mix per
decision 0004 item 54) is labelled illustrative, because the contract keeps
``"source"`` records read-only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from axle_studio.model.settings import CohortFixture, CohortSpec, RunSettings


@dataclass(frozen=True)
class Assumption:
    """One labelled, source-cited model input (contract results-v2.md section 8)."""

    name: str  # unique key, e.g. "public_charge_gbp_per_kwh" or "average_uk.daily_miles_mean"
    label: str  # dialog/table label, e.g. "Public charge rate"
    value: float | int | str | tuple[float, ...]
    unit: str  # "GBP/kWh", "kW", "percent", "fraction", "" for unitless
    evidence: Literal["source", "illustrative", "synthetic"]
    source: str  # workbook cell, decision number, or "interview-demo choice"
    meaning: str  # one line
    editable: bool
    bounds: tuple[float, float] | None  # inclusive; required when editable and numeric
    group: str  # dialog tab, see _ASSUMPTION_GROUPS below
    affects: str  # what changes when it changes, for the help tooltip


# The contract's seven "Edit assumptions" dialog tabs (results-v2.md section 8),
# in tab order.  Not itself enforced by the shared contract validator (which
# only checks `group` is a string), but every entry below is checked against
# this tuple in tests/model/test_assumptions.py and the dialog draws its tabs
# from it.
ASSUMPTION_GROUPS = (
    "Fleet",
    "Driving",
    "Plugging & home charging",
    "Battery",
    "Public & prices",
    "Weather",
    # Supplier contract v1 §8 (lane S2): the supplier P&L terms, the
    # illustrative carbon intensity and the cost curve's risk charge.
    "Supplier",
    # Trading contract v1 §10.8 (lane J6): the manufacturer, availability
    # and commitment records. The skip-share records and the two holiday
    # switches stay in "Plugging & home charging" next to the plug
    # probabilities they act on (§10.8's placement decision).
    "Firm MW",
    "Simulation",
)


# ---------------------------------------------------------------------------
# Cohorts: the six supplied archetype rows, crosswalk-reviewed against
# 'Source archetypes'!A6:P11 in the preserved Source_Evidence_Workbook.xlsx
# (docs/sources/cohort-crosswalk.md). Locators below are Excel A1 references.
# ``cohort_fixture`` turns these records into the ``CohortFixture`` the
# population is drawn from.
#
# Built from a small per-field metadata table plus per-cohort raw values,
# rather than 48 hand-written Assumption(...) calls: the metadata (unit,
# evidence, label, group, meaning, affects) is identical across cohorts for a
# given field, and writing it once cuts the repetition that made the
# hand-written version error-prone to review. Per-cohort/per-field
# exceptions (the Intelligent Octopus mileage caveat, the Always
# Plugged-in adaptation) are layered on afterwards, explicitly.
#
# Home charging power ('Source archetypes'!H6:H11, 7 kW in every row) is not
# repeated per cohort: decision 0004 item 34 makes it one fleet-wide value,
# HOME_CHARGING['home_charging_power_kw'], which the dashboard can edit.
#
# Population share ('Source archetypes'!C6:C11) is not built by the generic
# per-field loop below: decision 0004 item 54 (plan 2026-09-29-analyst-
# trading-polish-plan.md D9) makes the cohort mix an editable illustrative
# override, source share as default, so it lives in its own ``COHORT_SHARES``
# section after ``COHORTS`` -- the same "source-traced default the dashboard
# lets you edit is labelled illustrative" pattern as home charging power
# (module docstring above), because the contract keeps "source" records
# read-only.
# ---------------------------------------------------------------------------

# 'Source archetypes'!B6:B11, shown as each cohort's label in the dashboard.
COHORT_SOURCE_NAMES: dict[str, str] = {
    "average_uk": "Average (UK)",
    "intelligent_octopus": "Intelligent Octopus average",
    "infrequent_charging": "Infrequent charging",
    "infrequent_driving": "Infrequent driving",
    "scheduled_charging": "Scheduled charging",
    "always_plugged_in": "Always plugged-in",
}

_COHORT_ROWS: dict[str, int] = {
    "average_uk": 6,
    "intelligent_octopus": 7,
    "infrequent_charging": 8,
    "infrequent_driving": 9,
    "scheduled_charging": 10,
    "always_plugged_in": 11,
}

_COHORT_RAW_VALUES: dict[str, dict[str, float | int]] = {
    "average_uk": dict(
        population_share_fraction=0.4,
        daily_miles_mean=9435 / 365,
        battery_capacity_kwh=60.0,
        efficiency_miles_per_battery_kwh=3.5,
        plug_probability=1.0,
        preferred_target_soc_fraction=0.8,
        arrival_local_hour=18,
        departure_local_hour=7,
    ),
    "intelligent_octopus": dict(
        population_share_fraction=0.3,
        daily_miles_mean=28105 / 365,
        battery_capacity_kwh=72.5,
        efficiency_miles_per_battery_kwh=3.5,
        plug_probability=1.0,
        preferred_target_soc_fraction=0.8,
        arrival_local_hour=18,
        departure_local_hour=7,
    ),
    "infrequent_charging": dict(
        population_share_fraction=0.1,
        daily_miles_mean=9435 / 365,
        battery_capacity_kwh=60.0,
        efficiency_miles_per_battery_kwh=3.5,
        plug_probability=0.2,
        preferred_target_soc_fraction=0.8,
        arrival_local_hour=18,
        departure_local_hour=7,
    ),
    "infrequent_driving": dict(
        population_share_fraction=0.1,
        daily_miles_mean=5700 / 365,
        battery_capacity_kwh=60.0,
        efficiency_miles_per_battery_kwh=3.5,
        plug_probability=1.0,
        preferred_target_soc_fraction=0.8,
        arrival_local_hour=18,
        departure_local_hour=7,
    ),
    "scheduled_charging": dict(
        population_share_fraction=0.09,
        daily_miles_mean=9435 / 365,
        battery_capacity_kwh=60.0,
        efficiency_miles_per_battery_kwh=3.5,
        plug_probability=1.0,
        preferred_target_soc_fraction=0.8,
        arrival_local_hour=22,
        departure_local_hour=9,
    ),
    "always_plugged_in": dict(
        population_share_fraction=0.01,
        daily_miles_mean=9435 / 365,
        battery_capacity_kwh=60.0,
        efficiency_miles_per_battery_kwh=3.5,
        plug_probability=1.0,
        preferred_target_soc_fraction=0.8,
        arrival_local_hour=15,
        departure_local_hour=14,
    ),
}

# column, unit, label, group, default meaning, affects -- one row per field, shared
# across all six cohorts (row/value/evidence/source vary per cohort, handled below).
# Population share is not here: it is the one cohort field the dashboard can
# edit (decision 0004 item 54), so it is built separately as ``COHORT_SHARES``.
_COHORT_FIELD_META: dict[str, dict[str, str]] = {
    "daily_miles_mean": dict(
        column="D",
        unit="mi/day",
        label="Average daily miles",
        group="Driving",
        meaning=(
            "Average miles driven per day. Axle's sheet gives miles per year; dividing by 365 "
            "is a modelling step, not a figure from the sheet."
        ),
        affects=(
            "Changes how far this archetype drives each day, and so how much energy its trips use."
        ),
    ),
    "battery_capacity_kwh": dict(
        column="E",
        unit="kWh in the battery",
        label="Battery capacity",
        group="Battery",
        meaning="Battery size of this archetype's typical EV.",
        affects=(
            "Changes how much energy this archetype can store, and how many miles each "
            "percentage point of state of charge (SoC) is worth."
        ),
    ),
    "efficiency_miles_per_battery_kwh": dict(
        column="F",
        unit="miles per kWh from the battery",
        label="Driving efficiency",
        group="Battery",
        meaning="Miles driven per kWh taken from the battery, before the weather adjustment.",
        affects="Changes how far this archetype goes on each kWh, before the weather adjustment.",
    ),
    "plug_probability": dict(
        column="G",
        unit="chance per day",
        label="Daily plug-in probability",
        group="Plugging & home charging",
        meaning=(
            "The chance this archetype plugs in at home on any given day. Axle's sheet gives it "
            "as plug-ins per day; the simulation reads that number as a daily probability."
        ),
        affects="Changes how often this archetype's EVs are plugged in at home overnight.",
    ),
    "preferred_target_soc_fraction": dict(
        column="K",
        unit="fraction of a full battery",
        label="Preferred target SoC",
        group="Battery",
        meaning=(
            "The state of charge (SoC, how full the battery is) that home charging aims for. "
            "A preference, not the battery's 100% ceiling."
        ),
        affects="Changes how full this archetype's home charging tries to get the battery.",
    ),
    "arrival_local_hour": dict(
        column="I",
        unit="London clock hour",
        label="Home arrival time",
        group="Plugging & home charging",
        meaning=(
            "The hour this archetype usually gets home and plugs in. The day's trips have to "
            "be over by then."
        ),
        affects="Changes when this archetype gets home and can start charging.",
    ),
    "departure_local_hour": dict(
        column="J",
        unit="London clock hour",
        label="Home departure time",
        group="Plugging & home charging",
        meaning=(
            "The hour this archetype usually unplugs and leaves home. The day's trips start "
            "from here."
        ),
        affects="Changes when this archetype's charging window ends each morning.",
    ),
}

# Per-(cohort, field) meaning overrides where the shared default in
# _COHORT_FIELD_META does not capture a cohort-specific caveat.
_COHORT_MEANING_OVERRIDES: dict[tuple[str, str], str] = {
    ("intelligent_octopus", "battery_capacity_kwh"): (
        "Battery size of the early Intelligent Octopus group, which was heavy on Teslas. Not "
        "typical of UK drivers as a whole."
    ),
    ("intelligent_octopus", "daily_miles_mean"): (
        "About 77 miles a day, copied straight from the sheet. High because the early "
        "Intelligent Octopus group drove a lot; checked, and not a transcription error."
    ),
    ("always_plugged_in", "daily_miles_mean"): (
        "This archetype still drives a little, even though it is plugged in nearly all the time."
    ),
}

# Always Plugged-in's arrival/departure are a documented crosswalk adaptation
# (23h connection except a daily 14:00-15:00 trip), not a literal source cell.
_ALWAYS_PLUGGED_ADAPTED_MEANINGS: dict[str, str] = {
    "arrival_local_hour": (
        "Adapted from the sheet, not copied: plugged in about 23 hours a day, with one short "
        "trip from 14:00 to 15:00 (the sheet's cell I11 reads 0d)."
    ),
    "departure_local_hour": (
        "Adapted from the sheet, not copied: one short trip a day leaving about 14:00 (the "
        "sheet's cell J11 reads about 1d). The departure clock spreads that by up to 3 hours."
    ),
}


def _build_cohorts() -> dict[str, dict[str, Assumption]]:
    cohorts: dict[str, dict[str, Assumption]] = {}
    for cohort_id, row in _COHORT_ROWS.items():
        raw = _COHORT_RAW_VALUES[cohort_id]
        fields: dict[str, Assumption] = {}
        for field, meta in _COHORT_FIELD_META.items():
            column = meta["column"]
            if cohort_id == "always_plugged_in" and field in _ALWAYS_PLUGGED_ADAPTED_MEANINGS:
                evidence: Literal["source", "illustrative", "synthetic"] = "illustrative"
                source = "'Assumptions'!A23:H23"
                meaning = _ALWAYS_PLUGGED_ADAPTED_MEANINGS[field]
            elif field == "plug_probability":
                evidence = "illustrative"
                source = f"'Source archetypes'!{column}{row}"
                meaning = meta["meaning"]
            elif field == "daily_miles_mean":
                evidence = "source"
                source = f"derived: 'Source archetypes'!{column}{row} (annual mi/year) / 365"
                meaning = _COHORT_MEANING_OVERRIDES.get((cohort_id, field), meta["meaning"])
            else:
                evidence = "source"
                source = f"'Source archetypes'!{column}{row}"
                meaning = _COHORT_MEANING_OVERRIDES.get((cohort_id, field), meta["meaning"])
            fields[field] = Assumption(
                name=f"{cohort_id}.{field}",
                label=meta["label"],
                value=raw[field],
                unit=meta["unit"],
                evidence=evidence,
                source=source,
                meaning=meaning,
                editable=False,
                bounds=None,
                group=meta["group"],
                affects=meta["affects"],
            )
        cohorts[cohort_id] = fields
    return cohorts


COHORTS: dict[str, dict[str, Assumption]] = _build_cohorts()


# ---------------------------------------------------------------------------
# Cohort mix (decision 0004 items 21 and 54; plan 2026-09-29-analyst-trading-
# polish-plan.md D9): the population share becomes an editable illustrative
# override, defaulting to the source share ('Source archetypes'!C6:C11).  It
# is kept out of ``COHORTS`` rather than just flipping its ``editable`` flag
# in place, because ``test_source_evidence_records_are_never_editable`` holds
# every other cohort column to the item 21 default (source cells stay
# read-only); this one field's evidence changes to "illustrative" instead,
# the same choice already made for HOME_CHARGING['home_charging_power_kw'].
#
# ``cohort_fixture`` reads these through ``resolve_values`` (not the record's
# own ``.value``), so an edited draft -- or a value passed straight into
# ``cohort_fixture`` for a test -- reaches ``sampling.build_population``.
# ``validation_errors`` checks the six together (no single field owns "must
# sum to 100%"), the same shape as the public top-up threshold/target pair
# below, and attaches the message to the first name so the draft-error banner
# names one field, not six.
# ---------------------------------------------------------------------------

COHORT_SHARE_NAMES: tuple[str, ...] = tuple(
    f"{cohort_id}.population_share_percent" for cohort_id in _COHORT_ROWS
)
"""Names of the six editable cohort-mix fields, in source-row order."""

COHORT_SHARES: dict[str, Assumption] = {
    cohort_id: Assumption(
        name=f"{cohort_id}.population_share_percent",
        label=f"Population share: {COHORT_SOURCE_NAMES[cohort_id]}",
        value=_COHORT_RAW_VALUES[cohort_id]["population_share_fraction"] * 100.0,
        unit="percent",
        evidence="illustrative",
        source=(
            f"default from 'Source archetypes'!C{row} "
            f"({_COHORT_RAW_VALUES[cohort_id]['population_share_fraction']:.0%} source share, "
            "docs/sources/cohort-crosswalk.md); editable illustrative override, "
            "decision 0004 items 21 and 54"
        ),
        meaning=(
            "Share of the fleet drawn from this archetype. Defaults to the source share in "
            "Axle's sheet; the six shares must sum to 100% before Run is enabled."
        ),
        editable=True,
        bounds=(0.0, 100.0),
        group="Fleet",
        affects="Changes how many of the fleet's EVs are drawn from this archetype.",
    )
    for cohort_id, row in _COHORT_ROWS.items()
}

COHORT_SHARE_SUM_TOLERANCE_PERCENT: float = 0.01
"""Draft shares must sum to 100% within this many percentage points (item 54)."""


# ---------------------------------------------------------------------------
# Weekend home connection windows (decision 0004 item 42), consumed by
# cohort_fixture -> sampling.build_population ->
# sampling.sample_connection_opportunities and action.expected_departures_utc_ns.
#
# CNZ's "Learning from Intelligent Octopus" (May 2022) p.11 Fig.4 and p.12
# Fig.5 show plug-in and plug-out times by weekday and weekend: weekday
# arrival about 18:00 and departure 07:00-08:00; weekend arrival about 17:00
# and departure 09:00-11:00 with a wider spread.  The weekday window stays the
# supplied 'Source archetypes' cells (18:00 -> 07:00, COHORTS above), which
# already sit inside CNZ's weekday range, so no supplied value is overridden.
# The weekend window below is an illustrative adaptation of the figures: the
# centre of each reported range, read by eye, not a fit.  Only Average UK and
# Intelligent Octopus get it, because CNZ observed Intelligent Octopus drivers;
# the other four cohorts keep flat windows (no source), so their weekend
# hours equal their weekday hours.  The wider weekend spread is the
# ``weekend_plug_in_scale_minutes.*`` records in CONNECTION_CLOCKS below.
# ---------------------------------------------------------------------------

_CNZ_WINDOW_SOURCE = (
    "illustrative adaptation of CNZ figures: 'Learning from Intelligent Octopus' (May 2022) "
    "p.11 Fig.4 and p.12 Fig.5 (weekend arrival about 17:00, departure 09:00-11:00 with a "
    "wider spread than weekdays); decision 0004 item 42"
)
WEEKEND_WINDOW_COHORTS = ("average_uk", "intelligent_octopus")
"""Cohorts with a separate weekend connection window; the rest are flat."""

CONNECTION_WINDOWS: dict[str, dict[str, Assumption]] = {
    cohort_id: {
        "weekend_connection_arrival_local_hour": Assumption(
            name=f"{cohort_id}.weekend_connection_arrival_local_hour",
            label="Weekend home arrival time",
            value=17,
            unit="local clock hour",
            evidence="illustrative",
            source=_CNZ_WINDOW_SOURCE,
            meaning=(
                "The hour this archetype usually plugs in on a Saturday or Sunday: an hour "
                "earlier than the weekday 18:00."
            ),
            editable=False,
            bounds=None,
            group="Plugging & home charging",
            affects=(
                "Moves this archetype's weekend plug-ins, and its weekend flexibility, earlier."
            ),
        ),
        "weekend_connection_departure_local_hour": Assumption(
            name=f"{cohort_id}.weekend_connection_departure_local_hour",
            label="Weekend home departure time",
            value=10,
            unit="local clock hour",
            evidence="illustrative",
            source=_CNZ_WINDOW_SOURCE,
            meaning=(
                "The hour this archetype usually unplugs on a Saturday or Sunday morning: the "
                "middle of the 09:00 to 11:00 range CNZ reports. The smart charger plans to "
                "finish before this, less its departure margin."
            ),
            editable=False,
            bounds=None,
            group="Plugging & home charging",
            affects="A later weekend departure gives a longer weekend charging window.",
        ),
    }
    for cohort_id in WEEKEND_WINDOW_COHORTS
}


# ---------------------------------------------------------------------------
# Mileage CVs: per-cohort coefficients of variation for daily trip-distance
# sampling and for each EV's persistent mileage multiplier, consumed by
# sampling.build_population and
# forecast._trip_parameters. Interview-demo choices ("assumed" in the
# crosswalk vocabulary, "illustrative" here): CV controls the positive
# lognormal draw and mean x CV gives the displayed SD.
# ---------------------------------------------------------------------------

_MILEAGE_VARIATION_CONFIG = "interview-demo choice (illustrative variation, not fitted to data)"

MILEAGE_CV: dict[str, dict[str, Assumption]] = {
    cohort_id: {
        "daily_miles_cv": Assumption(
            name=f"{cohort_id}.daily_miles_cv",
            label="Day-to-day mileage spread",
            value=daily_cv,
            unit="fraction of the average",
            evidence="illustrative",
            source=_MILEAGE_VARIATION_CONFIG,
            meaning=(
                "How much this archetype's daily miles vary from one day to the next, as a "
                "share of its average. Each day's miles are drawn from a skewed distribution "
                "that never goes below zero."
            ),
            editable=False,
            bounds=None,
            group="Driving",
            affects="Higher values give more day-to-day mileage variation for this archetype.",
        ),
        "personal_mileage_cv": Assumption(
            name=f"{cohort_id}.personal_mileage_cv",
            label="Driver-to-driver mileage spread",
            value=personal_cv,
            unit="fraction of the average",
            evidence="illustrative",
            source=_MILEAGE_VARIATION_CONFIG,
            meaning=(
                "How much drivers within this archetype differ from each other in mileage, as a "
                "share of the average. Each EV gets one fixed multiplier for the whole run, so a "
                "heavy driver stays a heavy driver."
            ),
            editable=False,
            bounds=None,
            group="Driving",
            affects="Higher values give more EV-to-EV mileage variation within the archetype.",
        ),
    }
    for cohort_id, daily_cv, personal_cv in (
        ("average_uk", 0.30, 0.15),
        ("intelligent_octopus", 0.20, 0.10),
        ("infrequent_charging", 0.40, 0.20),
        ("infrequent_driving", 0.45, 0.225),
        ("scheduled_charging", 0.20, 0.10),
        ("always_plugged_in", 0.30, 0.15),
    )
}


# ---------------------------------------------------------------------------
# Trip behaviour per cohort: weekday/weekend drive probability, dwell and
# drive speed, consumed in forecast._trip_parameters. All interview-demo
# choices, not fitted CNZ observations or calibrated behaviour. There is no
# separate trip departure time: a trip starts at the home departure, which is
# also the unplug (decision 0004 item 51), so its clock hour is the cohort's
# home departure (COHORTS / CONNECTION_WINDOWS) and its spread is in
# CONNECTION_CLOCKS.
# ---------------------------------------------------------------------------

_TRIP_BEHAVIOUR_CONFIG = (
    "interview-demo choice (illustrative trip behaviour, not fitted to observed data)"
)
_TRIP_BEHAVIOUR_FIELD_META: dict[str, dict[str, str]] = {
    "weekday_drive_probability": dict(
        unit="chance per day",
        label="Weekday drive probability",
        meaning="The chance this archetype drives at all on a Monday to Friday.",
        affects="Changes how many weekdays this archetype drives at all.",
    ),
    "weekend_drive_probability": dict(
        unit="chance per day",
        label="Weekend drive probability",
        meaning="The chance this archetype drives at all on a Saturday or Sunday.",
        affects="Changes how many weekend days this archetype drives at all.",
    ),
    "destination_dwell_minutes": dict(
        unit="minutes",
        label="Away dwell time",
        meaning="Time parked away from home between the outbound and return legs of a trip.",
        affects="A longer stay keeps the EV away from home, and off its charger, for longer.",
    ),
    "drive_speed_mph": dict(
        unit="mph",
        label="Drive speed",
        meaning="Average road speed. It turns each leg's miles into a driving time.",
        affects="A faster speed makes the same miles take less time, so the EV is home sooner.",
    ),
}

_TRIP_BEHAVIOUR_RAW_VALUES: dict[str, dict[str, float | int]] = {
    cohort_id: dict(
        weekday_drive_probability=weekday_p,
        weekend_drive_probability=weekend_p,
        destination_dwell_minutes=dwell,
        drive_speed_mph=30.0,
    )
    for cohort_id, weekday_p, weekend_p, dwell in (
        ("average_uk", 0.85, 0.75, 480),
        ("intelligent_octopus", 0.85, 0.75, 480),
        ("infrequent_charging", 0.85, 0.75, 480),
        ("infrequent_driving", 0.35, 0.45, 180),
        ("scheduled_charging", 0.85, 0.75, 480),
        ("always_plugged_in", 0.2, 0.35, 120),
    )
}


def _build_trip_behaviour() -> dict[str, dict[str, Assumption]]:
    behaviour: dict[str, dict[str, Assumption]] = {}
    for cohort_id, raw in _TRIP_BEHAVIOUR_RAW_VALUES.items():
        fields: dict[str, Assumption] = {}
        for field, meta in _TRIP_BEHAVIOUR_FIELD_META.items():
            fields[field] = Assumption(
                name=f"{cohort_id}.{field}",
                label=meta["label"],
                value=raw[field],
                unit=meta["unit"],
                evidence="illustrative",
                source=_TRIP_BEHAVIOUR_CONFIG,
                meaning=meta["meaning"],
                editable=False,
                bounds=None,
                group="Driving",
                affects=meta["affects"],
            )
        behaviour[cohort_id] = fields
    return behaviour


TRIP_BEHAVIOUR: dict[str, dict[str, Assumption]] = _build_trip_behaviour()

# Weekend dwell for the two weekend-window cohorts (lead decision 28
# September 2026, follow-up to decision 0004 item 42).  With the 10:00
# weekend trip and the 8 h weekday dwell, cars got home about 18:50, after
# the 17:00 weekend window opened, so weekend plug-ins stayed near 19:00.  A
# 6 h weekend dwell brings them home about 16:50, so they plug in when the
# window opens, toward CNZ's weekend plug-in timing.  Dwell is an
# illustrative trip-behaviour value, not a supplied one.
for _cohort_id in ("average_uk", "intelligent_octopus"):
    TRIP_BEHAVIOUR[_cohort_id]["weekend_destination_dwell_minutes"] = Assumption(
        name=f"{_cohort_id}.weekend_destination_dwell_minutes",
        label="Weekend away dwell time",
        value=360,
        unit="minutes",
        evidence="illustrative",
        source=(
            "illustrative adaptation of CNZ weekend plug-in timing: 'Learning from Intelligent "
            "Octopus' (May 2022) p.11 Fig.4 (weekend plug-in about 17:00)"
        ),
        meaning="Time parked away between the legs of a Saturday or Sunday trip.",
        editable=False,
        bounds=None,
        group="Driving",
        affects="A shorter weekend stay brings this archetype home, and plugged in, earlier.",
    )
del _cohort_id


# ---------------------------------------------------------------------------
# Home clocks (decision 0004 item 51), consumed by
# sampling.sample_departure_times (departure) and
# sampling.sample_connection_opportunities (plug-in) via forecast_inputs.
#
# Each EV-night has two independent clocks around the cohort's clock hours
# (COHORTS, CONNECTION_WINDOWS): the plug-in, and the next morning's
# departure, which is both the unplug and the trip start.  Each shift is a
# truncated Student-t, t(df).ppf(lo + U(hi - lo)) x scale with
# lo, hi = t.cdf(-/+clip/scale), rounded to 30 minutes.  The scale is the t
# scale, not the standard deviation (for df 4, before truncation, the SD is
# about 1.4 x scale).
#
# All values are illustrative choices, not fitted: no source gives
# household-level day-to-day spread of plug-in or plug-out times (CNZ's
# figures are fleet histograms).  They replaced a normal with SD 45 minutes
# (60 at weekends) clipped at +/-90 minutes, with one shared draw for both
# ends of a session, which made plug-in and plug-out move together and
# rarely more than an hour from the cohort time.  The defaults give P5-P95
# about +/-2 h, 92-95% of shifts within +/-2 h and 1-2% at the +/-3 h bound
# (plan 2026-09-29 section 2A, A1).  Scales and df are editable; the clip is
# fixed because the session-gap and same-day checks in the samplers rely on
# it.
# ---------------------------------------------------------------------------

_CLOCK_SOURCE = "illustrative choice (decision 0004 item 51; plan 2026-09-29 A1)"

# Departure scale, minutes.  75 gives the +/-2 h P5-P95 spread; infrequent
# drivers have no commute routine, so 90; scheduled chargers keep a fixed
# routine, so 30.
_DEPARTURE_SCALE_MINUTES: dict[str, float] = {
    "average_uk": 75.0,
    "intelligent_octopus": 75.0,
    "infrequent_charging": 75.0,
    "infrequent_driving": 90.0,
    "scheduled_charging": 30.0,
    "always_plugged_in": 75.0,
}

# Weekday plug-in scale, minutes.  60 for the evening-arrival cohorts;
# scheduled charging plugs in for a set start time, so 20.  Always Plugged-in
# has no plug-in clock (one session covers the whole run, see
# sampling._ALWAYS_PLUGGED_COHORT), so its 0 is fixed rather than editable.
_PLUG_IN_SCALE_MINUTES: dict[str, float] = {
    "average_uk": 60.0,
    "intelligent_octopus": 60.0,
    "infrequent_charging": 60.0,
    "infrequent_driving": 60.0,
    "scheduled_charging": 20.0,
    "always_plugged_in": 0.0,
}


def _clock_record(
    name: str, cohort_id: str, label: str, value: float, meaning: str, *, editable: bool = True
) -> Assumption:
    return Assumption(
        name=f"{name}.{cohort_id}",
        label=f"{label}: {COHORT_SOURCE_NAMES[cohort_id]}",
        value=value,
        unit="minutes",
        evidence="illustrative",
        source=_CLOCK_SOURCE,
        meaning=meaning,
        editable=editable,
        bounds=(0.0, 180.0) if editable else None,
        group="Plugging & home charging",
        affects="A larger spread lets this clock land further from the archetype's usual time.",
    )


CONNECTION_CLOCKS: dict[str, Assumption] = {
    "clock_clip_minutes": Assumption(
        name="clock_clip_minutes",
        label="Clock shift bound",
        value=180.0,
        unit="minutes",
        evidence="illustrative",
        source=_CLOCK_SOURCE,
        meaning=(
            "No plug-in or departure moves more than 3 hours from the archetype's usual time. "
            "The shift is drawn only from inside this range, rather than drawn freely and then "
            "capped, so there is no pile-up at exactly 3 hours."
        ),
        editable=False,
        bounds=None,
        group="Plugging & home charging",
        affects="A wider bound allows rarer, larger swings in plug-in and departure times.",
    ),
    "clock_t_df": Assumption(
        name="clock_t_df",
        label="Clock tail weight",
        # 4 gives visibly fat tails (the odd very early or late day) while
        # most days stay near the usual time; 30 or more is practically a
        # normal distribution.
        value=4.0,
        unit="degrees of freedom",
        evidence="illustrative",
        source=_CLOCK_SOURCE,
        meaning=(
            "How often a plug-in or departure lands unusually early or late. A small value "
            "gives fat tails (the odd very early or late day); 30 or more is close to a normal "
            "bell curve."
        ),
        editable=True,
        bounds=(1.0, 100.0),
        group="Plugging & home charging",
        affects="Fewer degrees of freedom make very early or late days more common.",
    ),
    **{
        f"departure_scale_minutes.{cohort_id}": _clock_record(
            "departure_scale_minutes",
            cohort_id,
            "Departure time spread",
            scale,
            "How far the morning departure (the unplug, and the trip start) wanders from this "
            "archetype's usual time. A scale for the shift, not a standard deviation.",
        )
        for cohort_id, scale in _DEPARTURE_SCALE_MINUTES.items()
    },
    **{
        f"plug_in_scale_minutes.{cohort_id}": _clock_record(
            "plug_in_scale_minutes",
            cohort_id,
            "Plug-in time spread",
            scale,
            "No plug-in clock: one home session covers the whole run."
            if cohort_id == "always_plugged_in"
            else "How far this archetype's evening plug-in wanders from its usual arrival hour "
            "on weekdays (and at weekends, for an archetype with no separate weekend window).",
            editable=cohort_id != "always_plugged_in",
        )
        for cohort_id, scale in _PLUG_IN_SCALE_MINUTES.items()
    },
    # Decision 0004 item 42: CNZ's weekend plug-in and plug-out times spread
    # wider than weekday ones, so the two weekend-window cohorts get a wider
    # weekend plug-in scale (75 minutes, item 51).  Flat-window cohorts use
    # their weekday scale at weekends.
    **{
        f"weekend_plug_in_scale_minutes.{cohort_id}": replace(
            _clock_record(
                "weekend_plug_in_scale_minutes",
                cohort_id,
                "Weekend plug-in time spread",
                75.0,
                "How far this archetype's Saturday and Sunday plug-in wanders from its weekend "
                "arrival hour. Wider than on weekdays, as CNZ's figures show.",
            ),
            source=f"{_CLOCK_SOURCE}; wider weekend spread from {_CNZ_WINDOW_SOURCE}",
        )
        for cohort_id in WEEKEND_WINDOW_COHORTS
    },
}


# ---------------------------------------------------------------------------
# Weather: synthetic seven-day base temperatures (a scenario pattern, not
# observed or API-sourced weather) and the illustrative efficiency
# sensitivity model. The base temperatures feed
# sampling.sample_daily_temperature; the sensitivity, reference
# temperature and loss cap feed physics.effective_driving_efficiency.
# ---------------------------------------------------------------------------

_WEATHER_CONFIG = (
    "synthetic scenario (an invented seven-day temperature pattern, not observed weather)"
)

WEATHER: dict[str, Assumption] = {
    "warmup_base_temperature_c": Assumption(
        name="warmup_base_temperature_c",
        label="Warm-up base temperature",
        value=8.0,
        unit="degrees C",
        evidence="synthetic",
        source=_WEATHER_CONFIG,
        meaning="Base temperature applied to every warm-up day before the study starts.",
        editable=False,
        bounds=None,
        group="Weather",
        affects="Shifts the warm-up days' temperature, and so how far a kWh goes on those days.",
    ),
    **{
        f"study_base_temperature_c.day_{day}": Assumption(
            name=f"study_base_temperature_c.day_{day}",
            label=f"Day {day} base temperature",
            value=value,
            unit="degrees C",
            evidence="synthetic",
            source=_WEATHER_CONFIG,
            meaning=(
                "One day of an invented seven-day temperature pattern. Not a real forecast or "
                "a historical record."
            ),
            editable=False,
            bounds=None,
            group="Weather",
            affects="Shifts that day's temperature, and so how far a kWh goes that day.",
        )
        for day, value in enumerate((8.0, 9.0, 11.0, 12.0, 10.0, 7.0, 9.0), start=1)
    },
    "daily_sd_c": Assumption(
        name="daily_sd_c",
        label="Daily temperature spread",
        value=2.0,
        unit="degrees C (standard deviation)",
        evidence="synthetic",
        source=_WEATHER_CONFIG,
        meaning=(
            "Random day-to-day wobble around each day's base temperature, as a standard deviation."
        ),
        editable=False,
        bounds=None,
        group="Weather",
        affects="Higher values give more day-to-day temperature variation.",
    ),
    "sensitivity_fraction_per_c": Assumption(
        name="sensitivity_fraction_per_c",
        label="Weather efficiency sensitivity",
        value=0.01,
        unit="fraction lost per degree C",
        evidence="illustrative",
        source="interview-demo choice",
        meaning=(
            "Share of driving efficiency lost for each degree the day is away from the 20 °C "
            "reference, colder or warmer. At 0.01, a 10 °C day costs 10% of the miles per kWh."
        ),
        editable=True,
        bounds=(0.0, 0.10),
        group="Weather",
        affects="Higher values make driving efficiency more sensitive to cold and heat.",
    ),
    "reference_temperature_c": Assumption(
        name="reference_temperature_c",
        label="Weather reference temperature",
        value=20.0,
        unit="degrees C",
        evidence="illustrative",
        source="illustrative choice",
        meaning="Temperature at which driving efficiency is not adjusted.",
        editable=False,
        bounds=None,
        group="Weather",
        affects="Shifts the temperature at which no weather efficiency adjustment applies.",
    ),
    "max_efficiency_loss_fraction": Assumption(
        name="max_efficiency_loss_fraction",
        label="Maximum weather efficiency loss",
        value=0.50,
        unit="fraction",
        evidence="illustrative",
        source="illustrative choice",
        meaning=(
            "The most that cold or heat can cut driving efficiency: at 0.5, an EV never loses "
            "more than half its miles per kWh."
        ),
        editable=False,
        bounds=None,
        group="Weather",
        affects="Caps how much efficiency can drop in extreme cold or heat.",
    ),
}


# ---------------------------------------------------------------------------
# Home charging: home charging power, home charge efficiency, opening SoC and
# the mobility reserve. Decision 0004 item 34 drops the separate vehicle-side
# AC limit: an EV at home charges at the home charging power alone, so the
# kernel no longer takes min(home charger, vehicle AC limit) (see
# SOURCE_CHECKS finding (b)).
#
# Wording note: Mike found "AC" confusing, so no label, meaning or group here
# uses "AC" -- "home charging power" is used throughout instead.
# ---------------------------------------------------------------------------

HOME_CHARGING: dict[str, Assumption] = {
    "home_charging_power_kw": Assumption(
        name="home_charging_power_kw",
        label="Home charging power",
        value=7.0,
        unit="kW",
        evidence="illustrative",
        source=(
            "default from 'Source archetypes'!H6:H11 (7 kW in all six rows, "
            "docs/sources/cohort-crosswalk.md); one fleet-wide value, decision 0004 item 34"
        ),
        meaning=(
            "The most power an EV draws while plugged in at home; 7 kW is a typical UK "
            "wallbox. In one half-hour the battery can gain at most half this figure in kWh."
        ),
        editable=True,
        # 1 kW is below any real home charger; 22 kW is the largest three-phase
        # home wallbox, so the dialog cannot ask for a power no home supplies.
        bounds=(1.0, 22.0),
        group="Plugging & home charging",
        affects=(
            "Lower power spreads home charging over more half-hours and lowers the "
            "evening peak; very low power moves energy to public top-ups."
        ),
    ),
    "home_charge_efficiency": Assumption(
        name="home_charge_efficiency",
        label="Home charge efficiency",
        value=0.92,
        unit="efficiency fraction",
        evidence="illustrative",
        source="interview-demo choice",
        meaning=(
            "Share of the electricity drawn from the grid that ends up in the battery when "
            "charging at home. The rest is lost as heat in the charger and battery."
        ),
        editable=True,
        bounds=(0.01, 1.0),
        group="Plugging & home charging",
        affects="Lower efficiency means more grid import is needed for the same battery gain.",
    ),
    "opening_soc_fraction": Assumption(
        name="opening_soc_fraction",
        label="Opening battery SoC",
        # Settled start (decision 0004 item 36): equal to the cohorts' 80 %
        # preferred target, so warm-up is not spent refilling from an
        # arbitrary level.  The earlier 55 % start left refill spilling into
        # the study week, so cutting home charging power from 7.0 to 3.6 kW
        # *raised* study-week home import by about 190 kWh.
        # Read-only since M3b: after three warm-up days the opening SoC changes
        # the reported week only slightly (under 0.5 % in a 200 EV x 10 week
        # check), so a dial would suggest a larger effect than it has.
        value=0.8,
        unit="fraction of a full battery",
        evidence="illustrative",
        source="decision 0004 item 36 (settled start: the preferred target SoC)",
        meaning=(
            "How full every battery is when the warm-up starts, before any simulated charging. "
            "Equal to the preferred target, so the run starts settled rather than refilling "
            "from an arbitrary level."
        ),
        editable=False,
        bounds=None,
        group="Battery",
        affects=(
            "Changes the reported week only slightly (under 0.5% in a 200 EV x 10 week check "
            "with a three-day warm-up), because the warm-up settles the fleet first. That is "
            "why it is fixed."
        ),
    ),
    "reserve_soc_fraction": Assumption(
        name="reserve_soc_fraction",
        label="Mobility reserve SoC",
        value=0.2,
        unit="fraction of a full battery",
        evidence="illustrative",
        source="interview-demo choice (the crosswalk's per-group reserve field)",
        meaning=(
            "An assumed floor the driver would like to keep for unplanned trips. Recorded for "
            "each EV but not used by the physics yet: charging aims only at the preferred "
            "target."
        ),
        editable=False,
        bounds=None,
        group="Battery",
        affects="Nothing yet. If wired in, it would set a floor on how low the battery may go.",
    ),
}


# ---------------------------------------------------------------------------
# Public charging: the decision 0004 item 32 threshold top-ups (threshold,
# target and charging efficiency) and the illustrative GBP/kWh rate
# (decision 0004 items 4 and 11, editable, not a tariff claim).  Under item 32
# a public charger is always available and charging time is not modelled, so
# the earlier destination availability, charger power, vehicle acceptance and
# public SoC ceiling had no effect and were removed (task M4-layout).
# ---------------------------------------------------------------------------

_PUBLIC_CONFIG = "interview-demo choice, not calibrated to any observed data"

PUBLIC_CHARGING: dict[str, Assumption] = {
    "efficiency_fraction": Assumption(
        name="efficiency_fraction",
        label="Public charge efficiency",
        value=0.9,
        unit="fraction",
        evidence="illustrative",
        source=_PUBLIC_CONFIG,
        meaning=(
            "Share of the electricity bought at a public charger that ends up in the battery. "
            "The grid import for a top-up is the battery energy added divided by this."
        ),
        editable=True,
        bounds=(0.5, 1.0),
        group="Public & prices",
        affects="Lower efficiency means more grid import is needed for the same battery gain.",
    ),
    "public_charge_gbp_per_kwh": Assumption(
        name="public_charge_gbp_per_kwh",
        label="Public charge rate",
        value=0.79,
        unit="GBP/kWh",
        evidence="illustrative",
        source="illustrative choice, decision 0004 items 4 and 11",
        meaning=(
            "Illustrative price of public charging. It prices public top-ups and, in the smart "
            "charging cost check, any energy the driver would have to buy back. Not a real "
            "tariff, bid or settlement figure."
        ),
        editable=True,
        bounds=(0.0, 5.0),
        group="Public & prices",
        affects="Changes the illustrative cost of public charging, unrecovered energy and travel.",
    ),
    "public_top_up_threshold_soc_percent": Assumption(
        name="public_top_up_threshold_soc_percent",
        label="Public top-up threshold SoC",
        value=10.0,
        unit="percent",
        evidence="illustrative",
        source="decision 0004 item 37 (3971f40, 28 September 2026); interview-demo choice",
        meaning=(
            "If a trip would take the battery below this SoC, the EV tops up first at a public "
            "charger, which the model treats as always available. The default was lowered from "
            "20% to 10%, closer to CNZ's finding that under 10% of plug-ins start below 20% SoC."
        ),
        editable=True,
        bounds=(5.0, 50.0),
        group="Public & prices",
        affects="Higher values top up more often and move energy from home to public charging.",
    ),
    "public_top_up_target_soc_percent": Assumption(
        name="public_top_up_target_soc_percent",
        label="Public top-up target SoC",
        value=80.0,
        unit="percent",
        evidence="illustrative",
        source="decision 0004 item 32 (aa5f65e, 28 September 2026); interview-demo choice",
        meaning=(
            "The SoC a public top-up charges the battery to before the trip carries on. Must "
            "stay above the top-up threshold."
        ),
        editable=True,
        bounds=(50.0, 100.0),
        group="Public & prices",
        affects="Higher values mean bigger, more expensive top-ups.",
    ),
}


# ---------------------------------------------------------------------------
# Prices, system and trading: every input of
# sampling.generate_market_price_paths (decision 0004 items 53 and 55; plan
# §2B B1, B5 and §7 H1).  The day-ahead price is a convex supply curve of a
# synthetic system net demand (SYSTEM), plus a daily level and half-hour
# noise (PRICES); intraday and imbalance prices build on it (TRADING).
#
# The simulated prices are synthetic, not market data.  Where a value is
# calibrated, it is to the Elexon fit of GB day-ahead (APX market index) and
# system prices from 28 September 2025 to 28 September 2026
# (docs/research/elexon-price-calibration.md, plan B2): the demand shapes,
# the half-hour noise persistence, the daily-level persistence and the
# imbalance shares and premiums.  The rest are illustrative choices, tuned
# so a default (October, winter-shape) run matches that fit's other
# statistics: 2.1% negative half-hours, daily means with SD about £30, and a
# half-hour residual SD about £22.  Where the model gives one of these only
# jointly (daily-mean spread comes from wind, heating and the level
# together), the individual values are illustrative.
# ---------------------------------------------------------------------------

_MARKET_SOURCE = "illustrative choice (decision 0004 items 53 and 55)"
_ELEXON_SOURCE = (
    "calibrated to the Elexon fit, docs/research/elexon-price-calibration.md (decision 0004 "
    "item 53, plan B2). Contains BMRS data © Elexon Limited copyright and database right 2026"
)


def _market_record(
    name: str,
    label: str,
    value: float | tuple[float, ...],
    unit: str,
    meaning: str,
    affects: str,
    *,
    bounds: tuple[float, float] | None = None,
    source: str = _MARKET_SOURCE,
) -> Assumption:
    """One illustrative price-model record; editable in the dialog when it has bounds."""

    return Assumption(
        name=name,
        label=label,
        value=value,
        unit=unit,
        evidence="illustrative",
        source=source,
        meaning=meaning,
        editable=bounds is not None,
        bounds=bounds,
        group="Public & prices",
        affects=affects,
    )


# Demand shapes: GW of national demand before wind and solar, by London
# clock hour, as 3-harmonic Fourier coefficients (a0, a1, b1, a2, b2, a3,
# b3); see sampling.demand_shape_gw.  Derived from the Elexon price shapes
# (the 3-harmonic fits of the mean half-hourly price per weekday/weekend x
# winter/summer group): each fitted price was mapped back through the
# supply curve below (net demand = f⁻¹(price)); the ordinary day's heating
# (9.4 °C, the scenario's mean) was removed and its median wind and median
# solar added back; then the shape was refitted to three harmonics and
# corrected over five simulation rounds until the simulated mean day-ahead
# price of each half-hour was within £2 (winter) and £6 (summer) of the
# Elexon shape.  Since decision 0004 item 58 the summer midday trough comes
# mainly from solar, so these summer shapes are fairly flat by day.  A
# default run reproduces the observed shape: winter weekday trough 02:00-
# 03:00 (£64) and peak 17:30 (£109); summer troughs at 13:00 and peaks at
# 20:00-21:00.  Decision 0004 item 53 sets the season by the start month.
# If the supply curve, wind or solar records are edited, these
# coefficients must be re-derived.
_DEMAND_SHAPES = {
    "winter_weekday": (30.43, -3.07, -1.72, -0.65, -1.43, 0.16, 0.68),
    "winter_weekend": (27.29, -1.35, -2.18, -0.34, -0.43, -0.06, 1.12),
    "summer_weekday": (34.38, -2.0, -1.12, -1.6, -1.85, 0.36, 0.54),
    "summer_weekend": (28.42, 1.45, -3.02, -1.77, -1.61, -0.05, 1.0),
}

SYSTEM: dict[str, Assumption] = {
    **{
        f"demand_shape_gw.{shape}": _market_record(
            f"demand_shape_gw.{shape}",
            f"Demand shape, {shape.replace('_', ' ')}",
            coefficients,
            "GW (seven curve coefficients)",
            "The daily shape of national demand, after rooftop and other embedded solar, by "
            "London clock hour. Stored as seven smooth-curve coefficients, not 48 half-hour "
            "values.",
            "Moves when in the day net demand, and so the day-ahead price, is high or low.",
            source=_ELEXON_SOURCE,
        )
        for shape, coefficients in _DEMAND_SHAPES.items()
    },
    # Plan D7: one season per run, set by the start month, so a week never
    # switches shape part-way through.
    "summer_months": _market_record(
        "summer_months",
        "Summer-shape months",
        (4, 5, 6, 7, 8, 9),
        "month numbers",
        "Start months that use the summer demand shapes; every other month uses winter.",
        "Decides which pair of demand shapes a run uses.",
    ),
    # Electric heating is a small share of GB heating (most homes burn gas),
    # so demand rises by well under 1 GW per degree of cold.  0.6 GW/°C
    # below 15.5 °C (the usual degree-day base) adds about 4 GW on the
    # scenario's autumn days.  Limitation: the weather scenario is not
    # seasonal, so a summer run also sees this autumn heating.
    "heating_threshold_c": _market_record(
        "heating_threshold_c",
        "Heating threshold temperature",
        15.5,
        "degrees C",
        "Daily temperature below which heating adds to national demand.",
        "A higher threshold adds heating demand on more days.",
    ),
    "heating_gw_per_c": _market_record(
        "heating_gw_per_c",
        "Heating demand per degree",
        0.6,
        "GW per degree C below the threshold",
        "Extra national demand for each degree the day's sampled temperature is below the "
        "heating threshold; a cold week is a dearer week.",
        "Higher values make cold days raise day-ahead prices more.",
        bounds=(0.0, 1.5),
    ),
    # About 30 GW of GB wind (onshore plus offshore, 2025).
    "wind_installed_gw": _market_record(
        "wind_installed_gw",
        "Installed wind",
        30.0,
        "GW",
        "Wind capacity; each day's output is this times that day's capacity factor.",
        "More wind lowers net demand and prices, and makes windy days cheaper and negative "
        "prices more common.",
        bounds=(10.0, 60.0),
    ),
    # Median daily capacity factors: winter windier than summer (GB fleet
    # averages about 40% and 25-30%).
    "wind_median_capacity_factor.winter": _market_record(
        "wind_median_capacity_factor.winter",
        "Wind capacity factor, winter median",
        0.40,
        "fraction",
        "Median daily wind capacity factor in a winter-shape run.",
        "Higher values lower winter net demand and prices.",
    ),
    "wind_median_capacity_factor.summer": _market_record(
        "wind_median_capacity_factor.summer",
        "Wind capacity factor, summer median",
        0.28,
        "fraction",
        "Median daily wind capacity factor in a summer-shape run.",
        "Higher values lower summer net demand and prices.",
    ),
    # The capacity factor is expit(logit(median) + x_d) with x_d a daily
    # AR(1): the logit keeps it inside (0, 1), SD 0.75 gives about 20-63% on
    # the P10-P90 range in winter, and φ = 0.75 makes a windy or calm spell
    # last several days, as weather systems do.  Wind is what moves whole
    # days: these two were tuned with the daily level so a default run's
    # daily means have SD about £30 (Elexon fit £30, lag-1 φ 0.64; simulated
    # about 0.58) and 2% of half-hours are negative (Elexon 2.1%).
    "wind_logit_sd": _market_record(
        "wind_logit_sd",
        "Wind day-to-day spread",
        0.75,
        "standard deviation on the logit scale",
        "How much the daily wind capacity factor varies from day to day. Measured on the logit "
        "scale, which keeps the factor between 0 and 1.",
        "Higher values give more windy and calm days, so daily prices vary more and negative "
        "prices are more common.",
    ),
    "wind_ar1_phi": _market_record(
        "wind_ar1_phi",
        "Wind persistence",
        0.75,
        "correlation between consecutive days",
        "How much today's wind resembles yesterday's: 0 is no memory, 1 is a spell that never "
        "ends.",
        "Higher values make windy and calm spells last longer.",
    ),
    # Solar (decision 0004 item 58): about 18 GW of GB solar, mostly
    # embedded, so it lowers the demand the transmission market sees.  Output
    # = installed × a clear-sky half-sine between sunrise and sunset (London
    # clock) × a daily clearness.  Illustrative: GB solar peaks near 13-14 GW
    # on the clearest summer days (0.75 × a clear 0.95 × 18 GW ≈ 13 GW) and
    # a few GW on a bright winter noon.
    "solar_installed_gw": _market_record(
        "solar_installed_gw",
        "Installed solar",
        18.0,
        "GW",
        "Solar capacity; each half-hour's output is this times the daylight shape and that "
        "day's clearness.",
        "More solar lowers daytime net demand and prices, most on sunny summer weekends, and "
        "makes negative prices more common there.",
        bounds=(0.0, 40.0),
    ),
    **{
        f"solar_clear_sky_peak_cf.{season}": _market_record(
            f"solar_clear_sky_peak_cf.{season}",
            f"Solar clear-sky peak, {season}",
            peak,
            "fraction of installed",
            f"Output at solar noon on a perfectly clear {season} day, as a share of installed "
            "solar.",
            "Higher values give more midday solar in that season.",
        )
        for season, peak in (("winter", 0.35), ("summer", 0.75))
    },
    # Sunrise and sunset by London clock (BST in summer): about 08:00-16:00
    # in midwinter and 05:00-21:00 in midsummer; the records use those.
    **{
        f"solar_{edge}_local_hour.{season}": _market_record(
            f"solar_{edge}_local_hour.{season}",
            f"Solar {edge}, {season}",
            hour,
            "London clock hour",
            f"London clock time of {edge} for the daylight shape in a {season}-shape run.",
            "Wider daylight spreads solar output over more half-hours.",
        )
        for season, edge, hour in (
            ("winter", "sunrise", 8.0),
            ("winter", "sunset", 16.0),
            ("summer", "sunrise", 5.0),
            ("summer", "sunset", 21.0),
        )
    },
    # Clearness: expit(logit(0.6) + x_d), x_d a daily AR(1) (SD 1.0 on the
    # logit scale, φ 0.5), so a day runs from dull (about 25%) to clear
    # (about 85%) on its P10-P90 range and sunny spells last a day or two.
    "solar_clearness_median": _market_record(
        "solar_clearness_median",
        "Solar clearness, median day",
        0.6,
        "fraction of clear-sky output",
        "Median daily share of clear-sky solar output after cloud.",
        "Higher values give more solar on a typical day.",
    ),
    "solar_clearness_logit_sd": _market_record(
        "solar_clearness_logit_sd",
        "Solar day-to-day spread",
        1.0,
        "standard deviation on the logit scale",
        "How much the daily clearness varies from day to day, on the logit scale.",
        "Higher values give more sunny and dull days.",
    ),
    "solar_clearness_ar1_phi": _market_record(
        "solar_clearness_ar1_phi",
        "Solar persistence",
        0.5,
        "correlation between consecutive days",
        "How much today's cloud cover resembles yesterday's: 0 is no memory, 1 is a spell "
        "that never ends.",
        "Higher values make sunny and dull spells last longer.",
    ),
}

# Net-demand shocks (decision 0004 item 56; sampling.sample_net_demand_shocks).
# Illustrative, not fitted: GB has a few material scarcity or surplus
# surprises a week (plant trips, wind forecast misses, cold snaps), and many
# small ones.  Shocks start only in the evening-to-morning window because
# those are the hours home charging and its trading position depend on.
_SHOCK_SOURCE = "illustrative choice (decision 0004 item 56)"

SHOCKS: dict[str, Assumption] = {
    "shock_window_start_local_hour": _market_record(
        "shock_window_start_local_hour",
        "Shock window start",
        15.0,
        "London clock hour",
        "Earliest London time a net-demand shock may start each day.",
        "Moves the window in which shocks can start.",
        source=_SHOCK_SOURCE,
    ),
    "shock_window_end_local_hour": _market_record(
        "shock_window_end_local_hour",
        "Shock window end",
        9.0,
        "London clock hour (next morning)",
        "Latest London time a shock may start, the next morning.",
        "Moves the window in which shocks can start.",
        source=_SHOCK_SOURCE,
    ),
    # Mild: about two a night, typically 1 GW (exponential scale), a few
    # pounds on price; big: about five a week, median 4 GW with a lognormal
    # tail (P99 about 13 GW), tens of pounds on a tight evening.
    "mild_shock_rate_per_night": _market_record(
        "mild_shock_rate_per_night",
        "Mild shocks per night",
        2.0,
        "shocks per night",
        "Expected number of mild net-demand shocks starting in each night's window.",
        "More mild shocks make half-hour prices noisier around the evening and night.",
        bounds=(0.0, 6.0),
        source=_SHOCK_SOURCE,
    ),
    "big_shock_rate_per_week": _market_record(
        "big_shock_rate_per_week",
        "Big shocks per week",
        5.0,
        "shocks per week",
        "Expected number of big net-demand shocks per simulated week.",
        "More big shocks give more price spikes and dips, known and surprise.",
        bounds=(0.0, 14.0),
        source=_SHOCK_SOURCE,
    ),
    "mild_shock_scale_gw": _market_record(
        "mild_shock_scale_gw",
        "Mild shock size",
        1.0,
        "GW (mean size)",
        "Average size of a mild shock. Most are smaller than this and a few are much larger.",
        "Larger values make mild shocks move price more.",
        source=_SHOCK_SOURCE,
    ),
    "big_shock_median_gw": _market_record(
        "big_shock_median_gw",
        "Big shock median size",
        4.0,
        "GW",
        "Typical (median) size of a big shock. Half are smaller; the larger half has a long tail.",
        "Larger values make big shocks move price more.",
        bounds=(0.5, 8.0),
        source=_SHOCK_SOURCE,
    ),
    "big_shock_log_sd": _market_record(
        "big_shock_log_sd",
        "Big shock size spread",
        0.5,
        "standard deviation on the log scale",
        "How varied big shocks are in size, on the log scale. A higher value gives a fatter "
        "tail of very large shocks.",
        "Higher values make the largest big shocks larger.",
        source=_SHOCK_SOURCE,
    ),
    "shock_up_probability": _market_record(
        "shock_up_probability",
        "Share of shocks that raise demand",
        0.5,
        "fraction",
        "Probability a shock raises net demand (scarcity) rather than lowering it (surplus).",
        "Higher values give more spikes and fewer dips.",
        bounds=(0.0, 1.0),
        source=_SHOCK_SOURCE,
    ),
    "mild_shock_min_hours": _market_record(
        "mild_shock_min_hours",
        "Mild shock shortest duration",
        0.5,
        "hours",
        "Shortest mild shock. Durations are whole half-hours, equally likely between the "
        "shortest and longest.",
        "Longer shocks affect more half-hours.",
        source=_SHOCK_SOURCE,
    ),
    "mild_shock_max_hours": _market_record(
        "mild_shock_max_hours",
        "Mild shock longest duration",
        2.0,
        "hours",
        "Longest mild shock.",
        "Longer shocks affect more half-hours.",
        source=_SHOCK_SOURCE,
    ),
    "big_shock_min_hours": _market_record(
        "big_shock_min_hours",
        "Big shock shortest duration",
        1.0,
        "hours",
        "Shortest big shock. Durations are whole half-hours, equally likely between the "
        "shortest and longest.",
        "Longer shocks affect more half-hours.",
        source=_SHOCK_SOURCE,
    ),
    "big_shock_max_hours": _market_record(
        "big_shock_max_hours",
        "Big shock longest duration",
        4.0,
        "hours",
        "Longest big shock.",
        "Longer shocks affect more half-hours.",
        source=_SHOCK_SOURCE,
    ),
    # Half the mild and 40% of the big shocks are forecast before the
    # day-ahead auction (a cold still evening is seen coming); the rest
    # surprise the market and land in intraday and imbalance.
    "mild_shock_known_share": _market_record(
        "mild_shock_known_share",
        "Mild shocks known day-ahead",
        0.5,
        "fraction",
        "Share of mild shocks known before the day-ahead auction (in the day-ahead price).",
        "Higher values move more of the mild shocks' effect from intraday into day-ahead.",
        source=_SHOCK_SOURCE,
    ),
    "big_shock_known_share": _market_record(
        "big_shock_known_share",
        "Big shocks known day-ahead",
        0.4,
        "fraction",
        "Share of big shocks known before the day-ahead auction; the rest are surprises.",
        "Higher values let smart charging plan around more big shocks; lower values leave "
        "more of them to intraday and imbalance.",
        bounds=(0.0, 1.0),
        source=_SHOCK_SOURCE,
    ),
    # A cap on any one shock: a large wind-forecast miss plus a large plant
    # or interconnector trip is several GW; 15 GW is a generous ceiling, and
    # keeps the lognormal tail from inventing shocks larger than the fleet.
    "shock_max_gw": _market_record(
        "shock_max_gw",
        "Largest shock",
        15.0,
        "GW",
        "Ceiling on the size of any one mild or big shock.",
        "Lower values trim the rarest, largest price spikes and dips.",
        source=_SHOCK_SOURCE,
    ),
    # Trading contract §2.5 and Q10: a stochastic surprise becomes known a
    # fixed lead before it starts (a forecast update on the day), so
    # intraday trading can react to it for about an hour before gate
    # closure; a lead shorter than gate closure leaves its first half-hours
    # to imbalance only.  The bound stops at 11 h so a surprise at 00:00 on D
    # is never revealed before the 13:00 D-1 day-ahead auction it is kept out of.
    "surprise_reveal_hours": _market_record(
        "surprise_reveal_hours",
        "Surprise shock notice",
        2.0,
        "hours before the shock starts",
        "How long before it starts a surprise shock becomes known to intraday trading.",
        "Longer notice lets more of a surprise into intraday prices before gate closure; "
        "shorter notice leaves more of it to imbalance.",
        bounds=(0.0, 11.0),
        source=_SHOCK_SOURCE,
    ),
}

PRICES: dict[str, Assumption] = {
    # The supply curve f(N) = p_ref + s·(N − N_ref) + A·exp((N − N_scar)/w)
    # (sampling.supply_curve_gbp_per_mwh).  At the reference 20 GW, gas sets
    # a price near £78/MWh; each GW adds £5, so an overnight 12 GW is
    # about £38 and prices fall below zero under about 4.4 GW (windy nights
    # and sunny weekend middays).  Scarcity adds £25 at 33 GW and rises
    # e-fold every 3 GW, giving evening spikes on cold still days.
    "supply_reference_net_demand_gw": _market_record(
        "supply_reference_net_demand_gw",
        "Supply curve reference net demand",
        20.0,
        "GW",
        "Net demand at which the supply curve's gas-set reference price applies.",
        "Shifts the whole supply curve along the net-demand axis.",
    ),
    "supply_reference_price_gbp_per_mwh": _market_record(
        "supply_reference_price_gbp_per_mwh",
        "Gas-set reference price",
        78.0,
        "GBP/MWh",
        "Day-ahead price at the reference net demand, set by gas plant (fuel plus carbon).",
        "Shifts every day-ahead price up or down by the same amount.",
        bounds=(20.0, 200.0),
    ),
    "supply_slope_gbp_per_mwh_per_gw": _market_record(
        "supply_slope_gbp_per_mwh_per_gw",
        "Supply curve slope",
        5.0,
        "GBP/MWh per GW",
        "How much dearer the marginal plant is for each extra GW of net demand.",
        "Higher values widen the gap between cheap and dear half-hours and give more negative "
        "prices.",
    ),
    "scarcity_threshold_gw": _market_record(
        "scarcity_threshold_gw",
        "Scarcity threshold",
        33.0,
        "GW",
        "Net demand at which the scarcity term adds its scarcity price.",
        "Lower values make evening spikes more frequent.",
    ),
    "scarcity_width_gw": _market_record(
        "scarcity_width_gw",
        "Scarcity steepness width",
        3.0,
        "GW",
        "Extra net demand over which the scarcity price grows by about 2.7 times (a factor e).",
        "Smaller values make the curve steepen more sharply near the threshold.",
    ),
    "scarcity_price_gbp_per_mwh": _market_record(
        "scarcity_price_gbp_per_mwh",
        "Scarcity price at threshold",
        25.0,
        "GBP/MWh",
        "Price the scarcity term adds at the scarcity threshold.",
        "Higher values make tight evenings dearer.",
    ),
    # The daily level stands for gas, carbon, interconnector and outage news
    # that the net-demand model leaves out.  A stationary AR(1) across days
    # is the mean-reverting walk of plan B1, with the Elexon daily-mean
    # persistence φ = 0.64.  Its SD is only £5 because wind and heating
    # already supply most of the observed £30 day-to-day SD.
    "daily_level_sd_gbp_per_mwh": _market_record(
        "daily_level_sd_gbp_per_mwh",
        "Daily price level spread",
        5.0,
        "GBP/MWh (standard deviation)",
        "How far a whole day's price level sits above or below normal, as a standard "
        "deviation. The level drifts back toward normal over the following days.",
        "Higher values move whole days up and down more, without changing which half-hour of "
        "a day is cheapest.",
        bounds=(0.0, 20.0),
    ),
    "daily_level_ar1_phi": _market_record(
        "daily_level_ar1_phi",
        "Daily price level persistence",
        0.64,
        "correlation between consecutive days",
        "How much of today's price level carries into tomorrow: 0 is no memory, 1 never fades.",
        "Higher values let a dear or cheap spell last longer.",
        source=_ELEXON_SOURCE,
    ),
    # Decision 0004 item 48 made each world's day-ahead path its own; the
    # half-hour noise is kept so the cheapest half-hour of a night still
    # moves.  The Elexon fit's half-hour residual (after removing each day's
    # mean and its group's shape) has SD £21.6 and lag-1 φ 0.92: with φ set
    # to 0.92 (``ar1_phi``), an SD of £24 gives a simulated residual of about
    # £20 in a default run (the day's own mean absorbs part of a slow AR(1)).
    "day_ahead_sd_gbp_per_mwh": _market_record(
        "day_ahead_sd_gbp_per_mwh",
        "Day-ahead price noise SD",
        24.0,
        "GBP/MWh (standard deviation)",
        "Size of the half-hour-to-half-hour noise in each simulated week's own day-ahead price "
        "path, the prices the smart charger plans on.",
        "Higher values move each night's cheapest half-hour around more, so smart charging "
        "spreads across more half-hours.",
        bounds=(0.0, 40.0),
        source=(
            "illustrative choice tuned to the Elexon residual SD (decision 0004 items 48 and "
            "53, docs/research/elexon-price-calibration.md). Contains BMRS "
            "data © Elexon Limited copyright and database right 2026"
        ),
    ),
    # sampling reads this fixed value from the record itself, not through the
    # edited values (``forecast_inputs`` leaves it out; decision 0004 item 14).
    "ar1_phi": Assumption(
        name="ar1_phi",
        label="Price noise persistence",
        value=0.92,
        unit="correlation between adjacent half-hours",
        evidence="illustrative",
        source=_ELEXON_SOURCE + "; decision 0004 item 14",
        meaning=(
            "How much one half-hour's price noise carries into the next: the correlation "
            "between adjacent half-hours in the Elexon data, once each day's mean and shape "
            "are removed. Below 1, so the noise settles rather than drifting away over the "
            "week."
        ),
        editable=False,
        bounds=None,
        group="Public & prices",
        affects="Changes how correlated price noise is between adjacent half-hours.",
    ),
    # GB day-ahead auction limits (Nord Pool N2EX GB day-ahead and GB
    # half-hourly auctions): maximum clearing price £4,000/MWh from the 29
    # June 2022 delivery day, minimum −£500/MWh from 14 September 2021
    # (Nord Pool operational messages of 8 June 2022 and 13 September 2021).
    # Not re-verified for 2026.  Every simulated price (day-ahead, intraday
    # path and close, imbalance) is clipped to them, which bounds the
    # scarcity tail of the exponential supply curve.  Real imbalance prices
    # can exceed the auction cap; that is left out.
    "price_cap_gbp_per_mwh": _market_record(
        "price_cap_gbp_per_mwh",
        "Price cap",
        4000.0,
        "GBP/MWh",
        "Highest price any simulated market clears at: the GB day-ahead auction maximum.",
        "Lower values trim the rarest scarcity spikes.",
        source=(
            "Nord Pool operational message, 8 June 2022: N2EX and GB half-hourly auction "
            "maximum clearing price raised from £3,000 to £4,000/MWh"
        ),
    ),
    "price_floor_gbp_per_mwh": _market_record(
        "price_floor_gbp_per_mwh",
        "Price floor",
        -500.0,
        "GBP/MWh",
        "Lowest price any simulated market clears at: the GB day-ahead auction minimum.",
        "Higher values trim the deepest negative prices.",
        source=(
            "Nord Pool operational message, 13 September 2021: N2EX day-ahead minimum "
            "price threshold lowered from −£150 to −£500/MWh"
        ),
    ),
    # No free money (Mike, W1-prices review): day-ahead is set to the
    # expected intraday close less this premium, so on average the intraday
    # close exceeds day-ahead by exactly this amount.  0 means neither market
    # is cheaper on average; a positive value is a deliberate, visible risk
    # premium for buying ahead, not an artefact of the convex supply curve.
    "da_id_premium_gbp_per_mwh": _market_record(
        "da_id_premium_gbp_per_mwh",
        "Intraday over day-ahead premium",
        0.0,
        "GBP/MWh",
        "Expected intraday close minus day-ahead price, set deliberately.",
        "Positive values make intraday dearer than day-ahead on average, so buying ahead "
        "pays; 0 leaves no systematic gain from either market.",
        bounds=(-20.0, 20.0),
    ),
    # GB day-ahead auctions publish around 13:00 London the day before
    # delivery (plan B4 uses the same time for what a plan may see).
    "day_ahead_publication_local_hour": _market_record(
        "day_ahead_publication_local_hour",
        "Day-ahead publication time",
        13.0,
        "London clock hour on the day before delivery",
        "When a delivery day's day-ahead prices become known; intraday trading of a period "
        "starts here.",
        "An earlier time gives each period more intraday updates, so a wider intraday spread.",
    ),
}

TRADING: dict[str, Assumption] = {
    # GB intraday continuous trading closes one hour before delivery.
    # One gate-closure record, shared with the trading overlay (trading
    # contract §7).
    "gate_closure_minutes": _market_record(
        "gate_closure_minutes",
        "Gate closure lead time",
        60.0,
        "minutes before delivery",
        "Intraday trading of a period stops this long before it starts; the intraday price is "
        "frozen there.",
        "A longer lead time leaves fewer intraday updates, so the close stays nearer the "
        "day-ahead price.",
    ),
    # Hourly news moves the intraday price by a mean-zero shock: £2.50 per
    # hour over the 10-34 hours a period trades gives an intraday close
    # about £12 from the day-ahead price (1 SD).  Illustrative: the Elexon
    # fit has no intraday data to calibrate it against.
    "intraday_hourly_sd_gbp_per_mwh": _market_record(
        "intraday_hourly_sd_gbp_per_mwh",
        "Intraday hourly update SD",
        2.5,
        "GBP/MWh per hourly update (standard deviation)",
        "Standard deviation of each hourly intraday price update between day-ahead "
        "publication and gate closure.",
        "Higher values move the intraday close, the realised price, further from the "
        "day-ahead price; no plan sees it.",
        bounds=(0.0, 6.0),
    ),
    "intraday_shock_ar1_phi": _market_record(
        "intraday_shock_ar1_phi",
        "Intraday shock correlation",
        0.8,
        "correlation between adjacent half-hours",
        "How much one piece of intraday news moves neighbouring half-hours together.",
        "Higher values make news move a whole evening together rather than single periods.",
    ),
    # Two-state net imbalance volume (NIV), from the Elexon fit over 12
    # months: the system was long (NIV < 0) in 52.8% of half-hours, with the
    # system price £16.80 below the market index on average, and short in
    # 47.2%, £20.18 above it.  (September 2025 alone was wider, about −£35
    # and +£32.)  The long/short labels follow GB market commentary; the
    # Elexon glossary does not state the sign (research note, Limitations).
    "imbalance_long_share": _market_record(
        "imbalance_long_share",
        "Share of periods the system is long",
        0.53,
        "fraction",
        "Long-run share of half-hours with the system long (more generation than demand).",
        "Higher values make the imbalance price lower on average.",
        source=_ELEXON_SOURCE,
    ),
    "imbalance_state_persistence": _market_record(
        "imbalance_state_persistence",
        "Imbalance state persistence",
        0.8,
        "correlation between adjacent half-hours",
        "How strongly one half-hour's long or short state carries into the next.",
        "Higher values give longer runs of long or short periods.",
    ),
    "imbalance_long_premium_gbp_per_mwh": _market_record(
        "imbalance_long_premium_gbp_per_mwh",
        "Imbalance premium when long",
        -16.8,
        "GBP/MWh",
        "Imbalance price minus the intraday close in a long half-hour.",
        "More negative values pay less for spilled energy.",
        source=_ELEXON_SOURCE,
    ),
    "imbalance_short_premium_gbp_per_mwh": _market_record(
        "imbalance_short_premium_gbp_per_mwh",
        "Imbalance premium when short",
        20.2,
        "GBP/MWh",
        "Imbalance price minus the intraday close in a short half-hour.",
        "Higher values charge more for energy a position fails to deliver.",
        source=_ELEXON_SOURCE,
    ),
    # Trading contract Q (NIV): an unforeseen demand rise (an up surprise)
    # leaves the system short more often, a fall long.  In a half-hour a
    # surprise covers, the chance of long moves by this much against the
    # surprise's direction.  Otherwise NIV is independent of prices, a stated
    # limitation.
    "niv_surprise_shift": _market_record(
        "niv_surprise_shift",
        "Surprise effect on imbalance direction",
        0.25,
        "probability",
        "How much a surprise shock, while it lasts, shifts the chance that the system is "
        "short, in the surprise's direction.",
        "Higher values tie the imbalance price more closely to surprise shocks.",
    ),
    # Student-t with 3 degrees of freedom: finite variance, but occasional
    # £50+ moves.  Illustrative shape and scale; the Elexon fit gives only
    # the mean premium in each state.
    "imbalance_tail_scale_gbp_per_mwh": _market_record(
        "imbalance_tail_scale_gbp_per_mwh",
        "Imbalance tail scale",
        8.0,
        "GBP/MWh",
        "Typical size of the random noise added to the imbalance price. The noise has fat "
        "tails, so occasional moves of £50 or more happen.",
        "Higher values make imbalance spikes larger.",
    ),
    "imbalance_tail_df": _market_record(
        "imbalance_tail_df",
        "Imbalance tail degrees of freedom",
        3.0,
        "degrees of freedom",
        "How fat the tails of the imbalance noise are. Lower gives more rare, large moves.",
        "Lower values make rare large imbalance prices more common.",
    ),
}


# ---------------------------------------------------------------------------
# Action: price-optimised smart home charging (decision 0004 item 38, which
# replaced the 18:00 import cap) and its reporting tolerance and material
# threshold (item 45), read by action.smart_charging_inputs (the departure
# margin) and action.calculate_wholesale_world_cost_effect (the tolerance
# and threshold).
# ---------------------------------------------------------------------------

ACTION: dict[str, Assumption] = {
    "departure_margin_hours": Assumption(
        name="departure_margin_hours",
        label="Smart charging departure margin",
        # Decision 0004 item 51: with the wider, fat-tailed departure clock a
        # 1 h margin lets about 8% of home sessions leave before the plan
        # finishes (about 3% with the old clock); 2 h brings that to about 2%
        # for about 6% less energy-cost saving (300 EVs x 10 weeks, seed 42).
        value=2.0,
        unit="hours",
        evidence="illustrative",
        source="illustrative choice (decision 0004 item 38, option (a); 2 h from item 51)",
        meaning=(
            "The smart charger plans to finish this long before the archetype's usual home "
            "departure, so a driver who leaves a little early still leaves charged."
        ),
        editable=True,
        bounds=(0.0, 6.0),
        group="Plugging & home charging",
        affects=(
            "A larger margin shortens the smart-charging window: fewer early departures "
            "leave short, but less charging moves to the cheapest half-hours."
        ),
    ),
    "timed_tariff_enabled": Assumption(
        name="timed_tariff_enabled",
        label="Timed tariff policy",
        # On by default (lead decision, 8 October 2026): a bounded
        # number_input has no reachable "unset" value in Streamlit (an empty
        # commit resolves to min_value, not NaN, decision 0007 follow-up), so
        # the optional third "Timed tariff" path is switched by this record,
        # not by clearing ``timed_start_local_hour`` below, which is now
        # always a plain number.
        value=1,
        unit="switch (1 on, 0 off)",
        evidence="illustrative",
        source="illustrative choice (decision 0007)",
        meaning=(
            "Whether the optional third reported path runs: unmanaged home charging, except "
            "charging is held off from London 12:00 until the start time below each day, "
            "named for the common real-world pattern of a timed tariff. No tariff price is "
            "modelled and every path is costed the same way."
        ),
        editable=True,
        bounds=(0.0, 1.0),
        group="Plugging & home charging",
        affects=(
            "On (the default): adds a 'Timed tariff' path to the fleet and cohort interval "
            "frames and its P10-P90 band to the Smart charging Response chart. Off: the "
            "path is absent and nothing else changes. Every other chart, cost and "
            "comparison stays Unmanaged versus Smart."
        ),
    ),
    "timed_start_local_hour": Assumption(
        name="timed_start_local_hour",
        label="Timed tariff charging start time",
        value=0.0,
        unit="hours",
        evidence="illustrative",
        source="illustrative choice (decision 0007)",
        meaning=(
            "Clock hour the timed tariff policy above switches charging back on, London "
            "time (0 = midnight), each day from the barred noon start."
        ),
        editable=True,
        bounds=(0.0, 23.5),
        group="Plugging & home charging",
        affects=(
            "Moves the barred window's end when the policy above is on; unused when it is off."
        ),
    ),
    "expected_price_shape_days": Assumption(
        name="expected_price_shape_days",
        label="Expected price shape look-back",
        # Plan B4: the simplest honest stand-in for an unpublished price is
        # what the same half-hour cost on recent published days.  Seven days
        # covers a whole weekly cycle, so no weekday is over-weighted.
        value=7,
        unit="published days",
        evidence="illustrative",
        source="illustrative choice (plan B4)",
        meaning=(
            "When a half-hour's day-ahead price is not out yet, the smart charger stands in "
            "for it with the average price of the same London half-hour over this many recent "
            "days."
        ),
        editable=False,
        bounds=None,
        group="Plugging & home charging",
        affects="Changes the expected price shape used beyond the published day-ahead prices.",
    ),
    "not_recovered_material_share_percent": Assumption(
        name="not_recovered_material_share_percent",
        label="Material not-recovered share",
        value=1.0,
        unit="percent",
        evidence="illustrative",
        source="illustrative choice (decision 0004 item 45)",
        meaning=(
            "A simulated week counts as materially not recovered when the energy the smart "
            "path failed to recover is at least this share of the week's unmanaged home import."
        ),
        editable=True,
        bounds=(0.0, 100.0),
        group="Simulation",
        affects=(
            "Reporting only: a lower share flags more weeks as materially not recovered; "
            "no energy or cost changes."
        ),
    ),
    "not_recovered_tolerance_kwh": Assumption(
        name="not_recovered_tolerance_kwh",
        label="Not-recovered reporting tolerance",
        value=1e-6,
        unit="kWh",
        evidence="illustrative",
        source="technical model constant (decision 0004 items 5 and 13)",
        meaning=(
            "How close to zero counts as zero when checking whether the smart path recovered "
            "all its energy, valuing any shortfall and counting early departures."
        ),
        editable=False,
        bounds=None,
        group="Simulation",
        affects="Changes how small an energy difference is ignored before the flag trips.",
    ),
}


# ---------------------------------------------------------------------------
# Trading overlay: the illustrative trading ledger, baseline, commitment,
# non-response, control group and supplier records (trading contract v1 §7
# and §9.9; decision 0004 items 57-58; decision 0005), read by
# ``trading_inputs`` into ``market.run_trading``.
#
# Kept apart from ``TRADING`` above because every ``TRADING`` record is a
# price-generator input (``sampling.MARKET_PRICE_INPUTS`` must match it
# exactly); these records change the ledger only, never a price or a kernel
# path, except non-response and the control group, which change the smart
# path (§4.5, §9.5), and the intraday dispatch records, which change the
# selected (dispatched) path: the switch ``trading.intraday_dispatch``, the
# re-plan threshold theta, the commitment share (also the share of EVs
# locked to their day-ahead plan) and the half spread (the re-plan bar is
# theta + 2 x half spread) (intraday-dispatch-v1 §2, §4.1, §9).  None of
# them moves a draw or a price.  The gate closure (``gate_closure_minutes``) and the
# day-ahead decision instant (``day_ahead_publication_local_hour``, 13:00 on
# D-1, the same instant the day-ahead prices are published) are reused from
# the price records rather than duplicated (§7 lists them as trading
# records; one record each keeps the market and the trader in step).
#
# Switches are whole numbers 0/1; the contract's ``trading.baseline_history``
# string switch is ``trading.baseline_trap`` here (0 = "warmup_unmanaged",
# 1 = "flexed_study_nights").  Every value is illustrative.  Most records here
# are not dialog-editable: a run takes other values through
# ``forecast.run_forecast(trading_assumptions=...)``.  The three intraday
# dispatch records (``trading.intraday_dispatch``, its threshold and the
# commitment share) are editable, on the "Public & prices" dialog tab
# (intraday-dispatch-v1 §10, K5); their causal tests live in
# ``tests/model/test_assumption_effects.py`` beside every other editable
# record's.  ``bounds`` are the valid ranges ``forecast`` checks values
# against (plausible illustrative ranges, not sourced limits).
# ---------------------------------------------------------------------------

_TRADING_SOURCE = "illustrative choice (trading contract v1 §7; decision 0005)"


def _trading_record(
    name: str,
    label: str,
    value: float | int | str,
    unit: str,
    meaning: str,
    affects: str,
    *,
    bounds: tuple[float, float] | None,
    source: str = _TRADING_SOURCE,
    group: str = "Public & prices",
    editable: bool = False,
) -> Assumption:
    """One illustrative trading-overlay record (not dialog-editable unless ``editable``)."""

    return Assumption(
        name=name,
        label=label,
        value=value,
        unit=unit,
        evidence="illustrative",
        source=source,
        meaning=meaning,
        editable=editable,
        bounds=bounds,
        group=group,
        affects=affects,
    )


_LEDGER_ONLY = "The trading ledger only; no charging or price changes."

TRADING_OVERLAY: dict[str, Assumption] = {
    "trading.day_ahead_commitment_share": _trading_record(
        "trading.day_ahead_commitment_share",
        "Day-ahead commitment share",
        0.8,
        "fraction",
        "Share of the forecast turn-down sold at day-ahead; the rest is traded intraday "
        "(full strategy) or settled at imbalance (day-ahead only).",
        # Intraday dispatch contract v1 §2 and §9: one commitment, two views.
        "The trading ledger, and the share of EVs locked to their day-ahead plan when "
        "intraday dispatch is on. No random draw or price changes.",
        bounds=(0.0, 1.0),
        source="illustrative choice (decision 0004 item 58 (C); trading contract §9.2)",
        # Intraday dispatch contract v1 §10 (K5): editable on the Public &
        # prices tab alongside the two new dispatch records, since it is now
        # also the share of EVs locked to their day-ahead plan (§2).
        editable=True,
    ),
    "trading.half_spread_gbp_per_mwh": _trading_record(
        "trading.half_spread_gbp_per_mwh",
        "Intraday half spread",
        1.0,
        "GBP/MWh",
        "Half the intraday bid-ask spread, paid on every MWh traded intraday.",
        # Intraday dispatch contract v1 §4.1: a re-plan must beat theta + 2 s.
        "The trading ledger, and how often free EVs re-plan when intraday dispatch is on (a "
        "re-plan must also beat the round-trip spread). No random draw or price changes.",
        bounds=(0.0, 20.0),
    ),
    # Intraday dispatch contract v1 §9 (decision 0004 item 62(b)).  On by
    # default (lead decision Q1), since the forecast wiring lane (K4).  No
    # numeric seeded pin moved; tests and notebook checks whose claims hold
    # only for the day-ahead plan path now set this switch off explicitly.
    "trading.intraday_dispatch": _trading_record(
        "trading.intraday_dispatch",
        "Intraday dispatch follows the latest price",
        1,
        "switch (1 on, 0 off)",
        "When on, EVs not locked by the commitment share re-plan hourly on the latest "
        "intraday price when the saving beats the re-plan threshold plus the spread.",
        "The smart path and everything settled on it. No random draw or price changes.",
        bounds=(0.0, 1.0),
        source="illustrative product choice (decision 0004 item 62(b); dispatch contract §9)",
        # Intraday dispatch contract v1 §10 (K5): a dialog toggle (unit
        # starts "switch", so run_controller.is_switch draws it as one).
        editable=True,
    ),
    "trading.replan_threshold_gbp_per_mwh": _trading_record(
        "trading.replan_threshold_gbp_per_mwh",
        "Re-plan threshold",
        10.0,
        "GBP/MWh",
        "The saving per MWh moved, over and above the round-trip spread, that a free EV's new "
        "plan must beat before it replaces the old one. Set well above the £2.5/MWh hourly "
        "price wobble and the £2/MWh spread, so the fleet does not chase noise.",
        "How often free EVs re-plan when intraday dispatch is on. No random draw or price changes.",
        # Illustrative range: 0 re-plans on any saving above the spread;
        # GBP 500/MWh is well above the everyday intraday moves, so near it
        # only large spikes and grid requests move a free EV.
        bounds=(0.0, 500.0),
        source="illustrative choice (decision 0004 item 62(b); dispatch contract §9, lead Q2)",
        # Intraday dispatch contract v1 §10 (K5): editable, bounds as above.
        editable=True,
    ),
    "trading.customer_revenue_share": _trading_record(
        "trading.customer_revenue_share",
        "Customer revenue share",
        0.5,
        "fraction",
        "Share of each night's gross trading profit, when positive, paid to customers.",
        _LEDGER_ONLY,
        bounds=(0.0, 1.0),
    ),
    "trading.supplier_compensation_gbp_per_mwh": _trading_record(
        "trading.supplier_compensation_gbp_per_mwh",
        "Supplier compensation",
        0.0,
        "GBP/MWh",
        "Paid to the customer's energy supplier for each MWh of settled turn-down. 0 by "
        "default, because the P415 market rule spreads that cost across all suppliers instead.",
        _LEDGER_ONLY,
        bounds=(0.0, 200.0),
    ),
    "trading.unmet_charge_penalty_gbp_per_kwh": _trading_record(
        "trading.unmet_charge_penalty_gbp_per_kwh",
        "Unmet-charge penalty",
        0.79,
        "GBP/kWh",
        "Charged for each kWh an EV is shorter at departure under smart charging than it "
        "would have been unmanaged. Set equal to the illustrative public charge rate.",
        _LEDGER_ONLY,
        bounds=(0.0, 5.0),
        source="illustrative choice (decision 0005; trading contract §8 Q1)",
    ),
    "trading.baseline_in_day_adjustment": _trading_record(
        "trading.baseline_in_day_adjustment",
        "Baseline in-day adjustment",
        1,
        "switch (1 on, 0 off)",
        "When on, each night's baseline is shifted up or down by the average gap between "
        "metered and baseline import in its first few half-hours, as the BL01 settlement "
        "method does.",
        _LEDGER_ONLY,
        bounds=(0.0, 1.0),
    ),
    "trading.baseline_working_nights": _trading_record(
        "trading.baseline_working_nights",
        "Baseline working nights",
        5,
        "nights",
        "How many recent Monday to Friday nights are averaged to make a working night's baseline.",
        _LEDGER_ONLY,
        bounds=None,
    ),
    "trading.baseline_non_working_nights": _trading_record(
        "trading.baseline_non_working_nights",
        "Baseline non-working nights",
        2,
        "nights",
        "How many recent Saturday and Sunday nights are averaged to make a weekend night's "
        "baseline.",
        _LEDGER_ONLY,
        bounds=None,
    ),
    "trading.baseline_adjustment_window_slots": _trading_record(
        "trading.baseline_adjustment_window_slots",
        "Baseline adjustment window",
        3,
        "half-hours",
        "How many half-hours at the start of each night (from 12:00 London; 3 is 12:00 to "
        "13:30) the in-day adjustment averages over.",
        _LEDGER_ONLY,
        bounds=None,
        source="illustrative choice (trading contract §8 Q3)",
    ),
    "trading.baseline_trap": _trading_record(
        "trading.baseline_trap",
        "Baseline from flexed nights (trap)",
        0,
        "switch (1 on, 0 off)",
        "When on, each night's baseline also uses the smart-charged study nights just before "
        "it, as a rolling BL01 baseline would. That is the trap: the baseline then already "
        "contains the flexing it is meant to measure.",
        _LEDGER_ONLY,
        bounds=(0.0, 1.0),
        source="illustrative choice (decision 0004 item 58 (E); trading contract §9.5)",
    ),
    "trading.control_group": _trading_record(
        "trading.control_group",
        "Hold-out control group",
        0,
        "switch (1 on, 0 off)",
        "When on, a share of EVs in every archetype never follows a smart plan. Comparing "
        "them with the rest shows how biased the baseline is.",
        "Control EVs charge as unmanaged on the smart path; the random draws are unchanged.",
        bounds=(0.0, 1.0),
        source="illustrative choice (decision 0004 item 58 (E); trading contract §9.5)",
    ),
    "trading.control_group_share": _trading_record(
        "trading.control_group_share",
        "Control group share",
        0.1,
        "fraction",
        "Share of EVs held out as the control group when the switch is on.",
        "Only which EVs are held out; used only when the control group is on.",
        bounds=(0.01, 0.99),
        source="illustrative choice (trading contract §9.10 Q4)",
    ),
    # The six per-archetype non-response records of §9.5 were retired by
    # §10.1e: non-response now comes from the manufacturers' response rates
    # and outages (``SHARED_FACTORS``), so one observed thing has one knob.
    "supplier.early_departure_risk_charge_gbp_per_mwh": _trading_record(
        "supplier.early_departure_risk_charge_gbp_per_mwh",
        "Early-departure risk charge",
        24.0,
        "GBP/MWh",
        "Added to the cost of moving charging in the supplier cost curve. A rough figure: "
        "about 3% of sessions leave early, times the £790/MWh public charge rate.",
        "Supplier cost curve only.",
        bounds=(0.0, 300.0),
        source="illustrative, derived and rough (decision 0004 item 58 (D); contract §9.10 Q1)",
        group="Supplier",
        editable=True,
    ),
    "supplier.user_price_curve": _trading_record(
        "supplier.user_price_curve",
        "Supplier price curve",
        "none",
        "48 x GBP/MWh",
        "Optional 48-value London half-hour price curve that replaces the typical daily "
        "day-ahead shape; supplied with the run, not edited here.",
        "Shifts day-ahead, intraday and imbalance prices by the same offset per half-hour.",
        bounds=None,
        source="user-supplied (decision 0004 item 58 (D); trading contract §9.4)",
    ),
}


def trading_inputs(values: Mapping[str, object] | None = None) -> dict[str, float]:
    """The trading overlay's numeric inputs by record name, edited where the dialog allows.

    Returns every ``TRADING_OVERLAY`` value except the user price curve (a
    run input of its own), plus the two reused price records the trader needs:
    ``gate_closure_minutes`` and ``day_ahead_publication_local_hour``.
    Units as in the records; switches are 0/1. ``trading.commitment_rule``
    (§10.4, §10.8) is not one of these: ``forecast_inputs`` adds it to the
    ``trading_assumptions`` mapping separately, because
    ``forecast._trading_boundary`` checks this function's own keys against
    the mapping it is given and must see the same set either way.
    """

    resolved = resolve_values(values)
    inputs = {
        name: float(resolved.get(name, record.value))
        for name, record in TRADING_OVERLAY.items()
        if name != "supplier.user_price_curve"
    }
    for name, record in (
        ("gate_closure_minutes", TRADING["gate_closure_minutes"]),
        ("day_ahead_publication_local_hour", PRICES["day_ahead_publication_local_hour"]),
    ):
        inputs[name] = float(resolved.get(name, record.value))
    return inputs


# ---------------------------------------------------------------------------
# Supplier P&L terms and the illustrative carbon intensity (supplier contract
# v1 §8, decision 0006).  They feed only ``supplier.build_frames``: editing
# any of them changes no kernel frame, price, position or ledger row.  Unset
# commercial terms are NaN and read "unset", never 0 (decision 0003); the
# carbon values are placeholders (contract §16 Q5), not observed grid data.
# Editable in the dialog's "Supplier" tab (§8, lane S2); a NaN default means
# the term may stay unset, which ``validation_errors`` accepts.
# ---------------------------------------------------------------------------

_SUPPLIER_SOURCE = "illustrative choice (supplier contract v1 §8; decision 0006)"
_SUPPLIER_ONLY = "The supplier P&L only; no charging, price or ledger changes."
_CARBON_SOURCE = "illustrative placeholder (supplier contract v1 §3.8, §16 Q5; item 65 (4))"
_CARBON_ONLY = "The carbon-shifted figures only; no charging, price or ledger changes."

SUPPLIER: dict[str, Assumption] = {
    "supplier.customer_reward_mode": _trading_record(
        "supplier.customer_reward_mode",
        "Customer reward mode",
        0,
        "switch (0 = revenue share of the supplier's positive weekly gain, "
        "1 = flat GBP per EV per month)",
        "How the supplier pays customers: the customer revenue share of its own positive "
        "weekly gain, or a flat amount per EV per month.",
        _SUPPLIER_ONLY,
        bounds=(0.0, 1.0),
        source=_SUPPLIER_SOURCE,
        group="Supplier",
        editable=True,
    ),
    "supplier.customer_reward_gbp_per_ev_per_month": _trading_record(
        "supplier.customer_reward_gbp_per_ev_per_month",
        "Flat customer reward",
        float("nan"),
        "GBP per EV per month",
        "Flat reward per enrolled customer per month, used only when the reward mode is on. "
        "Unset until given, so the flat-mode payment and its nets show Unavailable, never £0.",
        _SUPPLIER_ONLY,
        bounds=(0.0, 50.0),
        source=_SUPPLIER_SOURCE,
        group="Supplier",
        editable=True,
    ),
    "supplier.platform_fee_gbp_per_ev_per_month": _trading_record(
        "supplier.platform_fee_gbp_per_ev_per_month",
        "Platform fee",
        float("nan"),
        "GBP per EV per month",
        "Optional platform fee per enrolled customer per month. Unset until given, so the "
        "after-fee net shows Unavailable, never £0.",
        _SUPPLIER_ONLY,
        bounds=(0.0, 50.0),
        source=_SUPPLIER_SOURCE,
        group="Supplier",
        editable=True,
    ),
}

CARBON: dict[str, Assumption] = {
    "carbon.intensity_at_reference_gco2_per_kwh": _trading_record(
        "carbon.intensity_at_reference_gco2_per_kwh",
        "Carbon intensity at reference net demand",
        180.0,
        "gCO2/kWh",
        "Illustrative grid carbon intensity when net demand sits at the supply curve's 20 GW "
        "reference point.",
        _CARBON_ONLY,
        bounds=(0.0, 600.0),
        source=_CARBON_SOURCE,
        group="Supplier",
        editable=True,
    ),
    "carbon.intensity_slope_gco2_per_kwh_per_gw": _trading_record(
        "carbon.intensity_slope_gco2_per_kwh_per_gw",
        "Carbon intensity slope",
        12.0,
        "gCO2/kWh per GW",
        "Change in the illustrative intensity per GW of net demand above or below the reference.",
        _CARBON_ONLY,
        bounds=(0.0, 50.0),
        source=_CARBON_SOURCE,
        group="Supplier",
        editable=True,
    ),
    "carbon.intensity_floor_gco2_per_kwh": _trading_record(
        "carbon.intensity_floor_gco2_per_kwh",
        "Carbon intensity floor",
        20.0,
        "gCO2/kWh",
        "Lowest illustrative intensity, when low net demand leaves a clean margin.",
        _CARBON_ONLY,
        bounds=(0.0, 600.0),
        source=_CARBON_SOURCE,
        group="Supplier",
        editable=True,
    ),
    "carbon.intensity_cap_gco2_per_kwh": _trading_record(
        "carbon.intensity_cap_gco2_per_kwh",
        "Carbon intensity cap",
        450.0,
        "gCO2/kWh",
        "Highest illustrative intensity, a gas-set margin; at least the floor.",
        _CARBON_ONLY,
        bounds=(0.0, 600.0),
        source=_CARBON_SOURCE,
        group="Supplier",
        editable=True,
    ),
}


def supplier_inputs(values: Mapping[str, object] | None = None) -> dict[str, float]:
    """The ``SUPPLIER`` and ``CARBON`` values by record name, edited where given.

    Units as in the records; unset terms stay NaN (decision 0003).  The
    supplier P&L also reads ``trading.customer_revenue_share`` from
    ``trading_inputs`` (not duplicated here).
    """

    resolved = resolve_values(values)
    inputs = {
        name: float(resolved.get(name, record.value))
        for name, record in (*SUPPLIER.items(), *CARBON.items())
    }
    # A switch, not a share: the dialog offers a 0/1 toggle (lead ruling), and
    # anything else is refused here, before a run starts.
    if inputs["supplier.customer_reward_mode"] not in (0.0, 1.0):
        raise ValueError(
            "supplier.customer_reward_mode must be 0 (revenue share) or 1 (flat reward)"
        )
    return inputs


# ---------------------------------------------------------------------------
# Zones (decision 0004 item 55; trading contract v1 §3 and §7, lead decision
# Q4): four illustrative network zones with no geography.  Each EV is placed
# in one zone by ``sampling.build_population`` (exact counts by largest
# remainder on these shares, then one permutation).  A zone's headroom is
# reported against, never enforced by the physics; NaN means "no headroom
# set".  Headroom is not editable yet because the Edit assumptions dialog
# only takes finite numbers; a run can still pass headrooms to
# ``forecast.run_forecast`` directly.
# ---------------------------------------------------------------------------

ZONE_IDS: tuple[str, ...] = ("zone_1", "zone_2", "zone_3", "zone_4")
ZONE_LABELS: dict[str, str] = {
    "zone_1": "Zone A",
    "zone_2": "Zone B",
    "zone_3": "Zone C",
    "zone_4": "Zone D",
}
ZONE_SHARE_NAMES: tuple[str, ...] = tuple(f"zones.share.{zone_id}" for zone_id in ZONE_IDS)
ZONE_SHARE_SUM_TOLERANCE: float = 1e-4
"""Draft zone shares must sum to 1 within this (a fraction, so 0.01 percentage points)."""

ZONES: dict[str, Assumption] = {
    **{
        f"share.{zone_id}": Assumption(
            name=f"zones.share.{zone_id}",
            label=f"Share of EVs in {ZONE_LABELS[zone_id]}",
            value=0.25,
            unit="fraction",
            evidence="illustrative",
            source="illustrative choice (decision 0004 item 55; trading contract v1 §3, Q4)",
            meaning=(
                "Share of the fleet placed in this illustrative network zone. The four "
                "shares must sum to 1."
            ),
            editable=True,
            bounds=(0.0, 1.0),
            group="Fleet",
            affects=(
                "Changes how many EVs sit in this zone, so its import and any local event's "
                "reach; no other random draw moves."
            ),
        )
        for zone_id in ZONE_IDS
    },
    **{
        f"headroom_kw.{zone_id}": Assumption(
            name=f"zones.headroom_kw.{zone_id}",
            label=f"Network headroom: {ZONE_LABELS[zone_id]}",
            value=float("nan"),
            unit="kW",
            evidence="illustrative",
            source="illustrative choice (decision 0004 item 55; trading contract v1 §3, §7)",
            meaning=(
                "Optional import limit this zone's EVs are compared with (unset means none). "
                "Reported only, never enforced."
            ),
            editable=False,
            bounds=None,
            group="Fleet",
            affects=(
                "Reporting only: the hours and simulated weeks above headroom in the zone "
                "summaries."
            ),
        )
        for zone_id in ZONE_IDS
    },
}


# ---------------------------------------------------------------------------
# Shared plug-in factors, holidays and charger manufacturers (trading
# contract v1 §10.1 and §10.8; decision 0004 items 59, 60, 63 and 64).
#
# Why shared factors: a firm-MW promise is only as honest as the tails of
# the fleet's deliverable MW.  If every EV plugged in independently the
# fleet's relative spread would fall as 1/sqrt(N); a day when fewer people
# plug in, a plug-in rate that is itself an estimate, and a charger maker
# whose cloud goes down make EVs move together, so part of the spread never
# diversifies away (§10.0).
#
# The skip share (§10.1b, item 60 option A): the source plug probabilities
# are "would plug in" rates, and a shared share q of those sessions is
# skipped each night, logit q = logit(median) + week_sd·y + day_sd·z + h.
# The arithmetic behind the SDs is in §10.1b: at day_sd 0.5 every
# skip-driven share varies about 6.5 % (relative) from day to day, about
# 13 % of would-be sessions are skipped on average, and evening
# availability has a pairwise correlation of about 0.0006.  day_sd 1.0 is
# the value that matches the source example exactly (item 63); Mike chose
# the milder 0.5 (item 64).
#
# Manufacturers (§10.1d): four illustrative charger makers with no real
# brand, like the zones.  Non-response comes from the maker's response rate
# and its per-night cloud outage (§10.1e).
#
# Every value here is illustrative.  Like the trading records, these are
# not dialog-editable yet (the Edit assumptions fields belong to the UI
# lane, §10.7 J6), so a run takes other values through
# ``forecast.run_forecast(shared_factor_assumptions=...)``; ``bounds`` are
# the ranges ``forecast`` checks them against.
# ---------------------------------------------------------------------------

MANUFACTURER_IDS: tuple[str, ...] = ("m1", "m2", "m3", "m4")
MANUFACTURER_LABELS: dict[str, str] = {
    "m1": "Maker A",
    "m2": "Maker B",
    "m3": "Maker C",
    "m4": "Maker D",
}
MANUFACTURER_SHARE_SUM_TOLERANCE: float = 1e-4
"""Manufacturer shares must sum to 1 within this (a fraction, as for the zones)."""

HOLIDAY_SKIP_LOGIT_SHIFT: float = 0.30
"""Skip-logit shift ``h_d`` on a holiday evening (trading contract v1 §10.1c).

+0.30 multiplies the odds of skipping by about 1.35: the median skip share
goes from 0.12 to about 0.155, so about 84.5 % of would-be sessions plug in
on an ordinary holiday night (lead review of the half-term preset).  The
bank-holiday switch reuses it rather than inventing a second value (§10.11
open item 1: the lead may confirm or set it).  Illustrative.
"""

_SHARED_FACTOR_SOURCE = "illustrative choice (trading contract v1 §10.1, §10.8)"
_PLUG_IN_AFFECTS = (
    "Changes how many would-be home sessions plug in each night, and how much that moves "
    "together across the fleet; no random draw moves."
)
_MAKER_AFFECTS = "Changes which smart-path sessions ignore their plan; no random draw moves."


def _shared_factor_record(
    name: str,
    label: str,
    value: float | int | str,
    unit: str,
    meaning: str,
    affects: str,
    *,
    bounds: tuple[float, float] | None,
    source: str = _SHARED_FACTOR_SOURCE,
    group: str = "Plugging & home charging",
    editable: bool = True,
) -> Assumption:
    """One illustrative §10 record, editable in the dialog unless ``editable=False``.

    ``availability.intraday_decision_local_time`` and
    ``availability.blackout_windows`` stay ``editable=False``: the first is
    a reporting-time choice with no numeric bounds to check, the second is
    a table edited through its own ``st.data_editor`` (§10.5b), not the
    generic per-record number field this factory's callers otherwise get.
    """

    return Assumption(
        name=name,
        label=label,
        value=value,
        unit=unit,
        evidence="illustrative",
        source=source,
        meaning=meaning,
        editable=editable,
        bounds=bounds,
        group=group,
        affects=affects,
    )


SHARED_FACTORS: dict[str, Assumption] = {
    "behaviour.plug_in_skip_median": _shared_factor_record(
        "behaviour.plug_in_skip_median",
        "Plug-in skip share (median night)",
        0.12,
        "fraction",
        "On a typical night, the share of would-be home plug-ins that do not happen. The same "
        "share applies to every EV that night, so a low plug-in night hits the whole fleet at "
        "once.",
        _PLUG_IN_AFFECTS,
        bounds=(0.0, 0.5),
        source="illustrative choice (decision 0004 item 60; trading contract v1 §10.1b)",
    ),
    "behaviour.plug_in_day_factor_sd": _shared_factor_record(
        "behaviour.plug_in_day_factor_sd",
        "Plug-in night-to-night swing",
        0.5,
        "standard deviation on the logit scale",
        "How much the shared skip share swings from night to night, on the logit scale. At "
        "0.5, a one-standard-deviation bad night has a skip share of about 18%, and every "
        "share that depends on skipping varies about 6.5% (relative) from day to day. 1.0 "
        "would match the source example exactly.",
        _PLUG_IN_AFFECTS,
        bounds=(0.0, 3.0),
        source=(
            "illustrative: Mike's choice (decision 0004 item 64; items 61 and 63 for the "
            "arithmetic; trading contract v1 §10.1b)"
        ),
    ),
    "behaviour.plug_in_week_factor_sd": _shared_factor_record(
        "behaviour.plug_in_week_factor_sd",
        "Plug-in week-to-week swing",
        0.17,
        "standard deviation on the logit scale",
        "One shift of the skip share that lasts a whole simulated week, because the plug-in "
        "rate in Axle's sheet is itself an estimate. About a third of the night-to-night "
        "swing.",
        _PLUG_IN_AFFECTS,
        bounds=(0.0, 3.0),
        source="illustrative: about a third of the night-to-night swing (decision 0004 item 64)",
    ),
    "behaviour.holiday_bank_holiday_monday": _shared_factor_record(
        "behaviour.holiday_bank_holiday_monday",
        "Bank holiday Monday",
        0,
        "switch (1 on, 0 off; the odds of skipping a plug-in rise by about a third on that "
        "evening)",
        "When on, the first Monday evening of the study is a bank holiday: its skip share "
        "rises (median 0.12 to about 0.155). No effect if the study has no Monday evening.",
        "Changes that evening's skip share only; trips, clocks, plans and draws are unchanged.",
        bounds=(0.0, 1.0),
        source=(
            "illustrative (decision 0004 item 59 optional presets; trading contract v1 "
            "§10.1c; lead answer: skip share only)"
        ),
    ),
    "behaviour.holiday_half_term_week": _shared_factor_record(
        "behaviour.holiday_half_term_week",
        "Half-term week",
        0,
        "switch (1 on, 0 off; the odds of skipping a plug-in rise by about a third every evening)",
        "When on, every study evening is a half-term evening: the skip share rises (median "
        "0.12 to about 0.155).",
        "Changes the study evenings' skip share only; trips, clocks, plans and draws are "
        "unchanged.",
        bounds=(0.0, 1.0),
        source="illustrative (decision 0004 item 59; trading contract v1 §10.1c; lead review)",
    ),
    **{
        f"manufacturers.share.{maker_id}": _shared_factor_record(
            f"manufacturers.share.{maker_id}",
            f"Share of EVs with {MANUFACTURER_LABELS[maker_id]}",
            0.25,
            "fraction",
            "Share of the fleet whose home charger is from this illustrative manufacturer. "
            "The four shares must sum to 1.",
            "Changes which EVs share this maker's response rate and outages; no other random "
            "draw moves.",
            bounds=(0.0, 1.0),
            source=(
                "illustrative (decision 0004 item 59; lead decision: equal shares, the zone "
                "precedent; trading contract v1 §10.1d)"
            ),
            group="Firm MW",
        )
        for maker_id in MANUFACTURER_IDS
    },
    **{
        f"manufacturers.response_rate.{maker_id}": _shared_factor_record(
            f"manufacturers.response_rate.{maker_id}",
            f"Response rate: {MANUFACTURER_LABELS[maker_id]}",
            0.95,
            "fraction",
            "Share of this maker's sessions that follow their smart plan on an ordinary night.",
            _MAKER_AFFECTS,
            bounds=(0.0, 1.0),
            source=(
                "illustrative: 1 - 0.05, the approved non-response value (trading contract "
                "v1 §8 Q2, §10.1e; decision 0004 item 59)"
            ),
            group="Firm MW",
        )
        for maker_id in MANUFACTURER_IDS
    },
    **{
        f"manufacturers.outage_probability_per_night.{maker_id}": _shared_factor_record(
            f"manufacturers.outage_probability_per_night.{maker_id}",
            f"Cloud outage chance: {MANUFACTURER_LABELS[maker_id]}",
            0.02,
            "probability per night",
            "Chance this maker's cloud control is out for a whole study night, so every one of "
            "its sessions that night ignores its plan.",
            _MAKER_AFFECTS,
            bounds=(0.0, 1.0),
            source=(
                "illustrative (decision 0004 item 59; lead decision; trading contract v1 §10.1d)"
            ),
            group="Firm MW",
        )
        for maker_id in MANUFACTURER_IDS
    },
}
MANUFACTURER_SHARE_NAMES: tuple[str, ...] = tuple(
    f"manufacturers.share.{maker_id}" for maker_id in MANUFACTURER_IDS
)
MANUFACTURER_RESPONSE_RATE_NAMES: tuple[str, ...] = tuple(
    f"manufacturers.response_rate.{maker_id}" for maker_id in MANUFACTURER_IDS
)
MANUFACTURER_OUTAGE_NAMES: tuple[str, ...] = tuple(
    f"manufacturers.outage_probability_per_night.{maker_id}" for maker_id in MANUFACTURER_IDS
)

# The availability and commitment records of §10.8, for the Firm MW
# summaries (§10.2, §10.5b) and the newsvendor commitment (§10.4).  They
# sit here rather than in ``TRADING_OVERLAY`` so ``trading_inputs`` keeps
# its numeric fields; the lanes that implement them read these records.
AVAILABILITY: dict[str, Assumption] = {
    "availability.intraday_decision_local_time": _shared_factor_record(
        "availability.intraday_decision_local_time",
        "Intraday availability decision time",
        "17:00",
        "London time (half-hour, 12:00-23:30)",
        "The instant each night's intraday Firm MW forecast is made, given who is plugged in.",
        "The Firm MW intraday forecast only.",
        bounds=None,
        source="illustrative (decision 0004 item 59: 'at 17:00'; trading contract v1 §10.8)",
        group="Firm MW",
        # A London time-of-day string, not a bounded number: shown as a
        # fixed value in the dialog rather than a new time-of-day widget,
        # which the task did not ask for.
        editable=False,
    ),
    "availability.blackout_windows": _shared_factor_record(
        "availability.blackout_windows",
        "Blackout windows",
        "none",
        "rows of (start London time, duration in minutes)",
        "Daily London windows in which no flexibility is offered and smart plans avoid "
        "moving charging (none by default).",
        "The smart charging plans and the Firm MW availability.",
        bounds=None,
        source="user choice (decision 0004 item 59; lead decision; trading contract v1 §10.8)",
        group="Firm MW",
        # A table, not a number: edited through its own st.data_editor
        # (run_controller.draft_blackout_windows), validated by
        # availability.validate_blackout_windows, not the generic per-record
        # number field the ``editable_records()`` pipeline renders.
        editable=False,
    ),
    "trading.commitment_rule": _shared_factor_record(
        "trading.commitment_rule",
        "Commitment rule",
        "fixed_share",
        # "choice", not "switch": run_controller.is_switch matches a
        # "switch" prefix to draw a 0/1 toggle, which would silently turn
        # this two-option string into 0 or 1 (int(bool(value))).
        "choice (fixed share or newsvendor)",
        "How much of the forecast turn-down is sold day-ahead: a fixed share, or the "
        "newsvendor rule, which sells more when the price is high relative to the cost of "
        "falling short.",
        "The day-ahead position and the trading ledger only.",
        bounds=None,
        source="illustrative (decision 0004 item 59; trading contract v1 §10.4, §10.8)",
        group="Firm MW",
    ),
}
COMMITMENT_RULES: tuple[str, ...] = ("fixed_share", "newsvendor")
"""Mirrors ``market.COMMITMENT_RULES``, duplicated by value to avoid a cycle
(``market`` imports ``events`` and ``sampling``, which import this module)."""


def shared_factor_inputs(values: Mapping[str, object] | None = None) -> dict[str, float]:
    """The §10.1 inputs of a run by record name: skip share, holidays and manufacturers.

    Units as in ``SHARED_FACTORS``: the skip median and maker shares and
    response rates are fractions, the SDs logit units, outage chances
    probabilities per night and the holiday switches 0/1.  ``values`` can
    override any of them (a run's edited values; names not given keep
    their record's value).
    """

    given = values or {}
    return {name: float(given.get(name, record.value)) for name, record in SHARED_FACTORS.items()}


# ---------------------------------------------------------------------------
# Event presets (decision 0004 items 55 and 58 F; trading contract v1 §2.2 and
# §9.6, lead decisions §9.10 Q5-Q7).  Each preset is one events-table row
# (``events.EVENT_COLUMNS``); ``events.preset_rows`` turns an id into the rows
# for one study (the two multi-row presets expand there).  Every value is an
# illustrative scenario choice: none is a forecast, a DFS rate or a DNO
# tariff.  ``size`` is GW of added net demand for price shocks, a MW cap on
# paid delivery for requests (NaN: no cap) and the share of covered sessions
# that ignore their plan for the control outage.  A run takes a sequence of
# preset ids (``forecast.run_forecast_from_assumptions(event_presets=...)``),
# which is what the events editor will drive.
#
# - cold_still_week: seven cold_still_evening rows, one per night 0-6 (Q7).
# - sunny_negative_weekend: one row per Saturday and Sunday whose 10:00-16:00
#   window lies inside the study; -6 GW is a placeholder the prices lane may
#   retune (Q5).  A known surplus: solar forecasts are known day-ahead.
# - charger_control_outage: chargers ignore plans for half the sessions that
#   plug in on night 3 between 15:00 and 03:00; known to nobody in advance,
#   hence "short" notice with 0 minutes (Q6, renamed from telemetry outage).
# ---------------------------------------------------------------------------

EVENT_PRESETS: dict[str, dict[str, object]] = {
    "cold_still_evening": {
        "label": "Cold still evening (known day-ahead)",
        "event_type": "price_shock_known",
        "night_index": 2,
        "start_local_time": "16:30",
        "duration_minutes": 180,
        "size": 4.0,
        "notice": "day_ahead",
        "notice_minutes": None,
        "scope": "national",
        "payment_gbp_per_mwh": float("nan"),
    },
    "surprise_evening_spike": {
        "label": "Surprise evening spike",
        "event_type": "price_shock_surprise",
        "night_index": 4,
        "start_local_time": "17:00",
        "duration_minutes": 120,
        "size": 3.0,
        "notice": "short",
        "notice_minutes": 90,
        "scope": "national",
        "payment_gbp_per_mwh": float("nan"),
    },
    "dfs_turn_down": {
        "label": "Demand-flexibility turn-down called day-ahead",
        "event_type": "turn_down",
        "night_index": 1,
        "start_local_time": "17:30",
        "duration_minutes": 60,
        "size": float("nan"),
        "notice": "day_ahead",
        "notice_minutes": None,
        "scope": "national",
        "payment_gbp_per_mwh": 500.0,
    },
    "local_turn_up": {
        "label": "Local turn-up in one zone",
        "event_type": "turn_up",
        "night_index": 5,
        "start_local_time": "23:00",
        "duration_minutes": 120,
        "size": float("nan"),
        "notice": "day_ahead",
        "notice_minutes": None,
        "scope": "zone_2",
        "payment_gbp_per_mwh": 60.0,
    },
    "cold_still_week": {
        "label": "Cold still week",
        "event_type": "price_shock_known",
        "night_index": None,  # every night 0-6
        "start_local_time": "16:30",
        "duration_minutes": 180,
        "size": 4.0,
        "notice": "day_ahead",
        "notice_minutes": None,
        "scope": "national",
        "payment_gbp_per_mwh": float("nan"),
    },
    "sunny_negative_weekend": {
        "label": "Sunny negative-price weekend",
        "event_type": "price_shock_known",
        "night_index": None,  # the nights before each Saturday and Sunday in the study
        "start_local_time": "10:00",
        "duration_minutes": 360,
        "size": -6.0,
        "notice": "day_ahead",
        "notice_minutes": None,
        "scope": "national",
        "payment_gbp_per_mwh": float("nan"),
    },
    "charger_control_outage": {
        "label": "Charger control outage: plans not followed",
        "event_type": "control_outage",
        "night_index": 3,
        "start_local_time": "15:00",
        "duration_minutes": 720,
        "size": 0.5,
        "notice": "short",
        "notice_minutes": 0,
        "scope": "national",
        "payment_gbp_per_mwh": float("nan"),
    },
}


# ---------------------------------------------------------------------------
# Simulation: the explicit model's default run shape, turned into
# ``RunSettings`` by ``run_settings``. Excluded from `result_assumptions()`:
# these values live in a result's settings_snapshot instead (contract
# SETTINGS_KEYS), not its assumptions tuple -- lead decision on the M3 review
# question (keeps "how big was this run" separate from "what
# behavioural/economic assumption did it use").
# ---------------------------------------------------------------------------

SIMULATION: dict[str, Assumption] = {
    "vehicle_count": Assumption(
        name="vehicle_count",
        label="Fleet size",
        value=1_000,
        unit="EVs",
        evidence="illustrative",
        source="technical model constant; the accepted 1,000 EV x 100 week run shape",
        meaning="How many EVs the run simulates.",
        editable=True,
        bounds=(1.0, 1_000.0),
        group="Simulation",
        affects="Changes how many EVs are simulated; larger fleets take longer to run.",
    ),
    "evaluation_world_count": Assumption(
        name="evaluation_world_count",
        label="Simulated weeks",
        value=100,
        unit="weeks",
        evidence="illustrative",
        source="technical model constant; the accepted 1,000 EV x 100 week run shape",
        meaning=(
            "How many separate weeks the model simulates. Every P10–P90 band across weeks is "
            "the spread across these."
        ),
        editable=True,
        bounds=(1.0, 100.0),
        group="Simulation",
        affects=(
            "More weeks give a steadier estimate of the P10–P90 spread, but take longer to run."
        ),
    ),
    "study_days": Assumption(
        name="study_days",
        label="Study horizon",
        value=7,
        unit="days",
        evidence="illustrative",
        source=(
            "technical model constant; decision 0004 item 1 fixes the study at 336 UTC "
            "half-hour slots"
        ),
        meaning="The reported week: seven days, fixed.",
        editable=False,
        bounds=None,
        group="Simulation",
        affects="Nothing: the study is always seven days.",
    ),
    "warmup_days": Assumption(
        name="warmup_days",
        label="Warm-up days",
        # Decision 0004 item 36: a warm-up lets every cohort (including the
        # plug-in-one-day-in-five "infrequent charging" EVs) reach its usual
        # pattern before the reported week starts.  Item 52 raised it from 3
        # to 7 days so the trading baseline has a full week of history (five
        # working and two non-working nights) and made it editable.  Three is
        # the floor item 36 found settled; 14 bounds the run time.
        value=7,
        unit="days",
        evidence="illustrative",
        source="decision 0004 items 36 (settled start) and 52 (seven days, editable 3-14)",
        meaning=(
            "Days simulated before the reported week, with prices, but not reported. They let "
            "every EV settle into its routine, so the study week is not distorted by start-up "
            "catch-up, and they give the trading baseline a week of history."
        ),
        editable=True,
        bounds=(3.0, 14.0),
        group="Simulation",
        affects=(
            "Changes how many days are simulated before the reported week; more days take "
            "longer to run. Changing it moves every random draw, so every simulated week "
            "changes too."
        ),
    ),
    # One np.random.default_rng(seed) per run, passed to every sampler
    # (decision 0004 item 14); the record's own text stays in plain words.
    "seed": Assumption(
        name="seed",
        label="Random seed",
        value=42,
        unit="whole number",
        evidence="illustrative",
        source="technical model constant",
        meaning=(
            "The starting point for every random draw in the run. Change it to see a different "
            "set of possible weeks; keep it to reproduce the same run exactly."
        ),
        editable=True,
        bounds=(0.0, float(2**32 - 1)),
        group="Simulation",
        affects="Changes which random future is drawn; the same seed reproduces the same run.",
    ),
}


# ---------------------------------------------------------------------------
# CNZ context: comparison figures from the cited report, not calibration
# targets or observed cohort data. Names, units and the median's
# percent-scaling match docs/contracts/results-v2.md section 8.1 exactly.
# They moved here from the removed cnz_plug_comparison module (task M4-layout),
# whose source note is kept in the two source strings below.
# ---------------------------------------------------------------------------

_CNZ_SOURCE_P13 = (
    "CNZ May 2022 report p.13: reported median 52%; 3% below 10%; fewer than 10% below 20%"
)
_CNZ_SOURCE_P11 = "CNZ May 2022 report p.11: weekday modal plug around 18:00"
_CNZ_AFFECTS = "Comparison context only; does not feed the simulation."

CNZ_CONTEXT: dict[str, Assumption] = {
    "cnz_median_plug_in_soc_percent": Assumption(
        name="cnz_median_plug_in_soc_percent",
        label="CNZ median plug-in SoC",
        value=52.0,
        unit="percent",
        evidence="source",
        source=_CNZ_SOURCE_P13,
        meaning=(
            "The median battery level (SoC) at plug-in that CNZ reported. Shown for comparison "
            "only; the model is never tuned to it."
        ),
        editable=False,
        bounds=None,
        group="Plugging & home charging",
        affects=_CNZ_AFFECTS,
    ),
    "cnz_share_plug_ins_below_10_percent_soc": Assumption(
        name="cnz_share_plug_ins_below_10_percent_soc",
        label="CNZ share below 10% SoC",
        value=0.03,
        unit="fraction",
        evidence="source",
        source=_CNZ_SOURCE_P13,
        meaning="The share of plug-ins CNZ found below 10% SoC. Shown for comparison only.",
        editable=False,
        bounds=None,
        group="Plugging & home charging",
        affects=_CNZ_AFFECTS,
    ),
    "cnz_share_plug_ins_below_20_percent_soc_upper_bound": Assumption(
        name="cnz_share_plug_ins_below_20_percent_soc_upper_bound",
        label="CNZ share below 20% SoC (upper bound)",
        value=0.10,
        unit="fraction",
        evidence="source",
        source=_CNZ_SOURCE_P13,
        meaning="Reported as 'fewer than 10%' below 20% SoC, so an upper bound, not a point value.",
        editable=False,
        bounds=None,
        group="Plugging & home charging",
        affects=_CNZ_AFFECTS,
    ),
    "cnz_weekday_plug_in_mode_local_hour": Assumption(
        name="cnz_weekday_plug_in_mode_local_hour",
        label="CNZ weekday plug-in mode hour",
        value=18,
        unit="hour (London)",
        evidence="source",
        source=_CNZ_SOURCE_P11,
        meaning="The most common weekday plug-in time CNZ reported, about 18:00; not exact.",
        editable=False,
        bounds=None,
        group="Plugging & home charging",
        affects=_CNZ_AFFECTS,
    ),
}


# ---------------------------------------------------------------------------
# Per-archetype session references (decision 0004 item 54, plan D-2 and D-5):
# the workbook's own plug-in SoC ('Source archetypes'!N6:N11) and kWh per
# plug-in (M6:M11), literal cells checked against docs/sources/
# cohort-crosswalk.md.  The Drivers ▸ Sessions lens draws them as dashed
# markers over the modelled distributions.  Audit references only (crosswalk
# "Physical rules"): plug-in SoC stays a stock-flow output and neither value
# feeds the simulation or calibrates it.  SoC is stored in percent (the
# cell's fraction x 100) so it sits on the same axis as the modelled SoC.
# M is battery-side energy (the crosswalk's "battery kWh/plug-in"), so it
# is compared with the battery-side "energy needed" metric.
# ---------------------------------------------------------------------------

_SESSION_SOURCE_VALUES: dict[str, tuple[float, float]] = {
    # cohort: (N plug-in SoC as a fraction, M kWh per plug-in)
    "average_uk": (0.68, 7.0),
    "intelligent_octopus": (0.52, 22.0),
    "infrequent_charging": (0.18, 37.0),
    "infrequent_driving": (0.73, 4.0),
    "scheduled_charging": (0.68, 7.0),
    "always_plugged_in": (0.68, 7.0),
}
_SESSION_SOURCE_AFFECTS = "Comparison marker only; does not feed the simulation."

SESSION_SOURCE_REFERENCES: dict[str, dict[str, Assumption]] = {
    cohort_id: {
        "source_plug_in_soc_percent": Assumption(
            name=f"{cohort_id}.source_plug_in_soc_percent",
            label="Source plug-in SoC",
            value=round(100.0 * soc_fraction, 6),
            unit="percent",
            evidence="source",
            source=f"'Source archetypes'!N{row} ({soc_fraction:g} as a fraction)",
            meaning=(
                "The plug-in SoC in Axle's sheet for this archetype. Shown for comparison only: "
                "the modelled SoC comes out of the simulation and is never tuned to this."
            ),
            editable=False,
            bounds=None,
            group="Plugging & home charging",
            affects=_SESSION_SOURCE_AFFECTS,
        ),
        "source_battery_kwh_per_plug_in": Assumption(
            name=f"{cohort_id}.source_battery_kwh_per_plug_in",
            label="Source kWh per plug-in",
            value=kwh_per_plug_in,
            unit="kWh battery-side per plug-in",
            evidence="source",
            source=f"'Source archetypes'!M{row}",
            meaning=(
                "The battery energy per plug-in in Axle's sheet for this archetype. Shown for "
                "comparison only, never a tuning target."
            ),
            editable=False,
            bounds=None,
            group="Plugging & home charging",
            affects=_SESSION_SOURCE_AFFECTS,
        ),
    }
    for cohort_id, row in _COHORT_ROWS.items()
    for soc_fraction, kwh_per_plug_in in [_SESSION_SOURCE_VALUES[cohort_id]]
}


# ---------------------------------------------------------------------------
# Model rules: plain-English statements of behaviour that is otherwise only
# visible by reading the vectorised kernel. Each is checked against the cited
# file:function before being written here (decision 0004 item 23).
# ---------------------------------------------------------------------------

MODEL_RULES: tuple[str, ...] = (
    # physics.py:_simulate_path -- home import only
    # where connected_full_slot, i.e. connected for the whole half-hour.
    "Home charging only happens in half-hours where the EV is connected to its "
    "home charger for the entire slot; a partially-connected slot draws no home charge.",
    # physics.py:_top_up_battery_kwh (decision 0004 item 32; default lowered
    # to 10% by item 37).
    "EVs never strand. Whenever a trip would take an EV's battery below the public "
    "top-up threshold (illustrative default 10% SoC), it first tops up at an "
    "always-available public charger to the top-up target (illustrative default 80% "
    "SoC), then continues, so unserved travel is zero by construction. Top-up grid "
    "import is the battery energy added divided by the public charge efficiency; the "
    "time spent charging is not modelled.",
    # physics.py:_travel_in_slot -- requested =
    # energy * overlap / duration, drawn every slot the leg overlaps.
    "A trip leg's travel energy is drawn from the battery at a constant rate over "
    "the leg's duration (energy x slot-overlap / total leg duration), not all at once.",
    # sampling.py:sample_daily_trip_inputs -- outbound_miles =
    # total_miles / 2.0; return_miles = total_miles / 2.0.
    "Each day's total miles are split into two equal legs: the outbound leg is "
    "exactly half the day's total miles, and the return leg is the other half.",
    # forecast.py:_trip_parameters -- weekday = (first_date +
    # timedelta(days=d)).weekday() < 5, using Python's plain calendar weekday.
    "A day counts as a weekday (for drive probability and departure timing) "
    "whenever its plain calendar weekday is Monday-Friday; UK bank holidays are "
    "not treated as weekends or given separate behaviour.",
    # sampling.py:sample_departure_times, sample_connection_opportunities and
    # action.py:expected_departures_utc_ns (decision 0004 items 42 and 51).
    "A home connection session runs from the arrival time on one London date to the "
    "departure time on the next. The departure is both the unplug and, on a driving day, "
    "the trip start. Plug-in and departure are independent fat-tailed shifts of at most "
    "3 hours around the cohort's times. The arrival uses the arrival date's weekday or weekend "
    "window and the departure uses the departure date's, so a Friday evening plug-in "
    "follows the weekday arrival and ends at the weekend departure on Saturday. Average UK "
    "and Intelligent Octopus have separate weekend windows (an illustrative adaptation of "
    "CNZ figures); the other cohorts use the same window every day. The smart charger "
    "expects the departure of the date it plans for.",
    # physics.py:_home_charge_kwh: the only per-EV import limit is
    # units['home_charger_limit_kw'] (decision 0004 item 34).
    "The explicit model has no shared fleet-wide or site-wide import cap: each EV "
    "is limited only by its own home charging power.",
    # action.py:plan_cheapest_slots and physics.py:_smart_charging_slot
    # (decision 0004 item 38).
    "On the selected (smart charging) path, each home session is planned when it starts: "
    "the charger puts just enough energy to reach the preferred target into the cheapest "
    "forecast half-hours before the expected departure (the cohort's typical departure time "
    "minus the departure margin). If the EV leaves earlier, the rest of the plan is missed "
    "and the EV leaves below target. The normal path charges as soon as it is plugged in.",
)


# ---------------------------------------------------------------------------
# Source checks required by docs/plans/2026-09-28-completion-plan.md (M3) and
# decision 0004 items 21 and 34.
# ---------------------------------------------------------------------------

SOURCE_CHECKS: tuple[str, ...] = (
    "(a) intelligent_octopus annual mileage: 'Source archetypes'!D7 = 28105 mi/year is a "
    "literal, directly-copied workbook cell (crosswalk row 2: 'Yes; no mismatch'); "
    "28105 / 365 = 76.99 mi/day, matching the ~77 mi/day check. This is SOURCED, not an "
    "arithmetic or transcription error. It is separately flagged in the crosswalk's "
    "'Provenance and limitations' section as a historical, Tesla-heavy Intelligent Octopus "
    "early-adopter cohort that is 'not representative of all current UK EV drivers': a "
    "representativeness caveat about the cohort, not a defect in how the figure was sourced.",
    "(b) home charging power (decision 0004 item 34, one home charging power; supersedes the "
    "earlier '7.0 vs 7.2 kW' note under decision 0004 item 21 alone): the crosswalk traces "
    "'Source archetypes'!H6:H11 = 7 kW to each cohort's home charger power, so "
    "HOME_CHARGING['home_charging_power_kw'] defaults to 7.0 kW. Crosswalk row 51 notes the "
    "legacy config's distinct vehiclePower field was a 'coincident technical value, not "
    "sourced by H6'. Task M3b removed it: the kernel's min(home charger, vehicle AC limit) is "
    "now the home charging power alone, and the stale 7.2 kW settings-level fields of the "
    "removed configs/illustrative-demo.json are gone. At the 7.0 kW default this changed no "
    "output, because both inputs of the old min() were already 7.0 kW for explicit runs.",
)


_SECTIONS: tuple[tuple[str, dict[str, object]], ...] = (
    ("cohorts", COHORTS),
    ("cohort_shares", COHORT_SHARES),
    ("connection_windows", CONNECTION_WINDOWS),
    ("mileage_cv", MILEAGE_CV),
    ("trip_behaviour", TRIP_BEHAVIOUR),
    ("connection_clocks", CONNECTION_CLOCKS),
    ("weather", WEATHER),
    ("home_charging", HOME_CHARGING),
    ("public_charging", PUBLIC_CHARGING),
    ("system", SYSTEM),
    ("shocks", SHOCKS),
    ("prices", PRICES),
    ("trading", TRADING),
    ("trading_overlay", TRADING_OVERLAY),
    ("supplier", SUPPLIER),
    ("carbon", CARBON),
    ("action", ACTION),
    ("zones", ZONES),
    ("shared_factors", SHARED_FACTORS),
    ("availability", AVAILABILITY),
    ("simulation", SIMULATION),
    ("cnz_context", CNZ_CONTEXT),
    ("session_source_references", SESSION_SOURCE_REFERENCES),
)


def _flatten_section(container: dict[str, object]) -> list[Assumption]:
    """Return every ``Assumption`` in a flat or one-level-nested section, in order."""

    records: list[Assumption] = []
    for value in container.values():
        if isinstance(value, Assumption):
            records.append(value)
        elif isinstance(value, dict):
            for subvalue in value.values():
                if not isinstance(subvalue, Assumption):
                    raise TypeError(f"expected Assumption, got {type(subvalue)!r}")
                records.append(subvalue)
        else:
            raise TypeError(f"expected Assumption or dict, got {type(value)!r}")
    return records


def _all_records() -> list[tuple[str, Assumption]]:
    """Return ``(section_name, Assumption)`` for every record in ``_SECTIONS``, in order."""

    return [
        (section_name, assumption)
        for section_name, container in _SECTIONS
        for assumption in _flatten_section(container)
    ]


def assumption_rows() -> pd.DataFrame:
    """Flatten every section into one table for Parameters/How-it-works views.

    Columns mirror the contract ``Assumption`` fields (results-v2.md section
    8) plus ``section``, this module's own grouping (distinct from the
    contract's dialog-tab ``group``).
    """

    records = [
        {
            "name": assumption.name,
            "section": section_name,
            "label": assumption.label,
            "value": assumption.value,
            "unit": assumption.unit,
            "evidence": assumption.evidence,
            "source": assumption.source,
            "meaning": assumption.meaning,
            "editable": assumption.editable,
            "bounds": assumption.bounds,
            "group": assumption.group,
            "affects": assumption.affects,
        }
        for section_name, assumption in _all_records()
    ]
    return pd.DataFrame(
        records,
        columns=[
            "name",
            "section",
            "label",
            "value",
            "unit",
            "evidence",
            "source",
            "meaning",
            "editable",
            "bounds",
            "group",
            "affects",
        ],
    )


def result_assumptions(values: Mapping[str, object] | None = None) -> tuple[Assumption, ...]:
    """Return every assumption for a result's ``assumptions`` field (contract section 8).

    ``values`` are the run's editable values (see ``resolve_values``); each
    editable record carries the value the run actually used, so Compare and
    How it works show edited values, not the defaults.  ``None`` means the
    defaults.

    Excludes the "simulation" section: those values (fleet size, world
    counts, seed, horizon) live in the result's ``settings_snapshot`` instead
    (contract ``SETTINGS_KEYS``), so a run's behavioural/economic assumptions
    stay distinct from how big the run was -- lead decision on the M3 review
    question. The "action" section is kept: it is technical action-mechanics,
    not run-scale, even though it shares the "Simulation" dialog tab.
    """

    resolved = resolve_values(values)
    return tuple(
        replace(assumption, value=resolved[assumption.name])
        if assumption.name in resolved
        else assumption
        for section_name, assumption in _all_records()
        if section_name != "simulation"
    )


def editable_defaults() -> dict[str, float | int | str]:
    """Return ``{name: value}`` for every assumption marked editable.

    Built directly from the records, not from ``assumption_rows()``'s
    DataFrame: a DataFrame "value" column upcasts a mix of Python int and
    float to float64, which would silently turn integer defaults such as
    ``seed`` or ``vehicle_count`` into floats.
    """

    return {
        assumption.name: assumption.value for _, assumption in _all_records() if assumption.editable
    }


def editable_records() -> tuple[Assumption, ...]:
    """Every editable record, in module order (the Edit assumptions dialog's fields)."""

    return tuple(assumption for _, assumption in _all_records() if assumption.editable)


def records_in_group(group: str) -> tuple[Assumption, ...]:
    """Every record in one dialog tab (``ASSUMPTION_GROUPS``), in module order."""

    return tuple(assumption for _, assumption in _all_records() if assumption.group == group)


# --------------------------------------------------------------------------
# Edited values: checking them and turning them into run inputs
# --------------------------------------------------------------------------


def validation_errors(values: Mapping[str, object]) -> dict[str, str]:
    """Return ``{name: message}`` for every editable value that is not usable.

    ``values`` maps editable names to candidate values (for example the
    dialog's draft).  Each value must be a finite real number inside its
    record's inclusive bounds, whole where the default is an integer, or
    (``trading.commitment_rule``, the one text switch, §10.4, §10.8) one of
    ``COMMITMENT_RULES``.  Four rules span more than one field: the top-up
    target must lie above the top-up threshold, otherwise a top-up would not
    add energy (decision 0004 item 32); the six editable cohort shares must
    sum to 100% (decision 0004 item 54), otherwise
    ``sampling.build_population`` would not draw an exact 1,000-EV (or
    whatever fleet size) population from them; the four zone shares must sum
    to 1 (trading contract v1 §3); and the four manufacturer shares must sum
    to 1 (§10.1d). An empty dict means the values are valid.
    """

    errors: dict[str, str] = {}
    for record in editable_records():
        value = values.get(record.name)
        if isinstance(record.value, str):
            # The one text-switch record: checked against its allowed set,
            # not a numeric range (bounds is None; §10.4, §10.8).
            if value not in COMMITMENT_RULES:
                errors[record.name] = f"must be one of {COMMITMENT_RULES}"
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors[record.name] = "must be a number"
            continue
        if math.isnan(value) and isinstance(record.value, float) and math.isnan(record.value):
            # A record whose default is NaN may stay unset (supplier contract
            # v1 §8, decision 0003): unavailable downstream, never 0.
            continue
        if not math.isfinite(value):
            errors[record.name] = "must be a finite number"
            continue
        if isinstance(record.value, int) and not float(value).is_integer():
            errors[record.name] = "must be a whole number"
            continue
        low, high = record.bounds
        if not low <= value <= high:
            errors[record.name] = f"must be between {low:g} and {high:g}"
    # The four manufacturer shares split the fleet exactly (§10.1d), so they
    # must sum to 1; the message goes on the first share, as the cohort and
    # zone shares do above.
    if not (set(MANUFACTURER_SHARE_NAMES) & errors.keys()):
        total_share = math.fsum(values[name] for name in MANUFACTURER_SHARE_NAMES)
        if not math.isclose(
            total_share, 1.0, rel_tol=0.0, abs_tol=MANUFACTURER_SHARE_SUM_TOLERANCE
        ):
            errors[MANUFACTURER_SHARE_NAMES[0]] = (
                f"the four manufacturer shares must sum to 1 (currently {total_share:g})"
            )
    threshold, target = "public_top_up_threshold_soc_percent", "public_top_up_target_soc_percent"
    if not ({threshold, target} & errors.keys()) and values[target] <= values[threshold]:
        errors[target] = "must be above the top-up threshold"
    # Attached to the first share only (not all six): draft_error joins every
    # {name: message} pair into one banner line, so six identical messages
    # would repeat the same sentence six times over.
    if not (set(COHORT_SHARE_NAMES) & errors.keys()):
        total_percent = math.fsum(values[name] for name in COHORT_SHARE_NAMES)
        if not math.isclose(
            total_percent, 100.0, rel_tol=0.0, abs_tol=COHORT_SHARE_SUM_TOLERANCE_PERCENT
        ):
            errors[COHORT_SHARE_NAMES[0]] = (
                f"the six cohort shares must sum to 100% (currently {total_percent:g}%)"
            )
    # The illustrative carbon intensity is clipped to [floor, cap]
    # (supplier contract v1 §3.8), so the floor may not exceed the cap.
    floor, cap = "carbon.intensity_floor_gco2_per_kwh", "carbon.intensity_cap_gco2_per_kwh"
    if not ({floor, cap} & errors.keys()) and values[floor] > values[cap]:
        errors[floor] = "must not be above the carbon intensity cap"
    # The zone shares split the fleet exactly (trading contract v1 §3), so
    # they must sum to 1; the message goes on the first share, as above.
    if not (set(ZONE_SHARE_NAMES) & errors.keys()):
        total_share = math.fsum(values[name] for name in ZONE_SHARE_NAMES)
        if not math.isclose(total_share, 1.0, rel_tol=0.0, abs_tol=ZONE_SHARE_SUM_TOLERANCE):
            errors[ZONE_SHARE_NAMES[0]] = (
                f"the four zone shares must sum to 1 (currently {total_share:g})"
            )
    # The timed-start barred window begins at London noon and ends at this
    # clock hour (decision 0007), so only a half-hour clock time is a usable
    # start.  Checked whatever ``timed_tariff_enabled`` holds, the same way
    # supplier.customer_reward_gbp_per_ev_per_month's bounds are checked
    # whatever supplier.customer_reward_mode holds: a switch here never gates
    # another field's own validity.
    timed_hour = "timed_start_local_hour"
    if timed_hour not in errors and not (float(values[timed_hour]) * 2.0).is_integer():
        errors[timed_hour] = "must be a half-hour multiple"
    return errors


def resolve_values(overrides: Mapping[str, object] | None = None) -> dict[str, float | int | str]:
    """Return every editable value: the defaults with ``overrides`` applied, checked.

    Raises ``ValueError`` naming the first unknown name or invalid value, so a
    run never starts from a value the dialog would reject.  Integer records
    (fleet size, worlds, seed) come back as ``int``; ``trading.commitment_rule``
    (the one text switch, §10.4, §10.8) comes back as ``str``.
    """

    values = editable_defaults()
    for name, value in (overrides or {}).items():
        if name not in values:
            raise ValueError(f"unknown editable assumption: {name}")
        values[name] = value
    errors = validation_errors(values)
    if errors:
        name, message = next(iter(errors.items()))
        raise ValueError(f"{name} {message}")

    def resolved(record: Assumption) -> float | int | str:
        if isinstance(record.value, str):
            return str(values[record.name])
        if isinstance(record.value, int):
            return int(values[record.name])
        return float(values[record.name])

    return {record.name: resolved(record) for record in editable_records()}


def run_settings(values: Mapping[str, object] | None, start_local_date: date) -> RunSettings:
    """Build one run's ``RunSettings`` from the edited values and the study start date."""

    resolved = resolve_values(values)
    return RunSettings(
        start_local_date=start_local_date,
        warmup_days=resolved["warmup_days"],
        study_days=SIMULATION["study_days"].value,
        vehicle_count=resolved["vehicle_count"],
        seed=resolved["seed"],
        evaluation_world_count=resolved["evaluation_world_count"],
        opening_soc_fraction=HOME_CHARGING["opening_soc_fraction"].value,
        reserve_soc_fraction=HOME_CHARGING["reserve_soc_fraction"].value,
        home_charge_efficiency=resolved["home_charge_efficiency"],
        weather_efficiency_sensitivity_fraction_per_c=resolved["sensitivity_fraction_per_c"],
    )


def cohort_fixture(values: Mapping[str, object] | None = None) -> CohortFixture:
    """Build the six-cohort ``CohortFixture`` the population is drawn from.

    Every cohort gets the one home charging power (decision 0004 item 34).
    Population shares come from the resolved draft, not ``COHORT_SHARES``'
    own ``.value`` (its fixed default): decision 0004 item 54 makes the mix
    editable, and ``resolve_values`` is what applies an edited draft (or a
    value passed straight in, for a test) on top of the source defaults, the
    same route ``home_charging_power_kw`` already takes below. ``resolve_values``
    also re-checks the "sum to 100%" rule, so a fixture is never built from
    shares the dialog would have rejected. ``daily_miles_sd`` is 0 because the
    run derives each EV's spread from the mileage CVs (``MILEAGE_CV``), not
    from a fixed SD.
    """

    resolved = resolve_values(values)
    power_kw = resolved["home_charging_power_kw"]
    cohorts = []
    for cohort_id, row in _COHORT_ROWS.items():
        record = COHORTS[cohort_id]
        weekend = CONNECTION_WINDOWS.get(cohort_id, {})
        cohorts.append(
            CohortSpec(
                cohort_id=cohort_id,
                source_name=COHORT_SOURCE_NAMES[cohort_id],
                source_row=row,
                population_share_fraction=resolved[f"{cohort_id}.population_share_percent"] / 100.0,
                daily_miles_mean=record["daily_miles_mean"].value,
                daily_miles_sd=0.0,
                battery_capacity_kwh=record["battery_capacity_kwh"].value,
                efficiency_miles_per_battery_kwh=record["efficiency_miles_per_battery_kwh"].value,
                plug_probability=record["plug_probability"].value,
                home_charger_limit_kw=power_kw,
                preferred_target_soc_fraction=record["preferred_target_soc_fraction"].value,
                arrival_local_hour=record["arrival_local_hour"].value,
                departure_local_hour=record["departure_local_hour"].value,
                # Flat window unless the cohort has a weekend record (item 42).
                weekend_arrival_local_hour=weekend.get(
                    "weekend_connection_arrival_local_hour", record["arrival_local_hour"]
                ).value,
                weekend_departure_local_hour=weekend.get(
                    "weekend_connection_departure_local_hour", record["departure_local_hour"]
                ).value,
            )
        )
    # required_vehicle_count is the 100-EV size the six source shares were
    # reviewed at; the population builder only checks it is a positive count.
    return CohortFixture(
        label="six-source-archetypes",
        evidence_kind="illustrative",
        required_vehicle_count=100,
        cohorts=tuple(cohorts),
    )


def forecast_inputs(
    values: Mapping[str, object] | None, *, warmup_days: int, study_days: int
) -> dict[str, object]:
    """Return the keyword inputs of ``forecast.run_forecast`` from the records.

    Keys match ``run_forecast``: per-cohort daily and personal mileage CVs,
    the trip-behaviour mapping, the base temperature per sampled London date
    (deg C, warm-up days first, ``RunSettings.sampled_day_count`` values),
    its world SD, the public-charge assumptions (SoC as fractions, as the
    kernel takes them), the synthetic price assumptions and the
    illustrative public rate (GBP/kWh), and the smart charger's
    departure margin (h) and the material not-recovered share (percent).
    Both models take the same public inputs (decision 0004 item 32); only
    the action model reads the prices, the margin and the material share.
    """

    if not 1 <= study_days <= 7:
        raise ValueError("study_days must be between 1 and 7: the weather scenario has 7 days")
    resolved = resolve_values(values)

    def clock(name: str) -> float:
        # Edited value when the clock record is editable, else its fixed value.
        return resolved.get(name, CONNECTION_CLOCKS[name].value)

    study_base = [
        WEATHER[f"study_base_temperature_c.day_{day}"].value for day in range(1, study_days + 1)
    ]
    # The noon-to-noon horizon touches one more London date than it has days:
    # the morning after the last session night (RunSettings.sampled_day_count,
    # decision 0004 item 52).  Its trips run before the study ends at noon;
    # they keep the last study day's base temperature (an illustrative
    # carry-forward, not a new weather value).
    study_base.append(study_base[-1])
    return {
        "daily_miles_cv_by_cohort": {
            cohort_id: cvs["daily_miles_cv"].value for cohort_id, cvs in MILEAGE_CV.items()
        },
        "personal_mileage_cv_by_cohort": {
            cohort_id: cvs["personal_mileage_cv"].value for cohort_id, cvs in MILEAGE_CV.items()
        },
        "trip_behaviour": {
            "clock_clip_minutes": CONNECTION_CLOCKS["clock_clip_minutes"].value,
            "clock_t_df": clock("clock_t_df"),
            "by_cohort": {
                cohort_id: {
                    **{field: record.value for field, record in fields.items()},
                    "departure_scale_minutes": clock(f"departure_scale_minutes.{cohort_id}"),
                    "plug_in_scale_minutes": clock(f"plug_in_scale_minutes.{cohort_id}"),
                    # Flat-window cohorts keep their weekday scale (item 42).
                    "weekend_plug_in_scale_minutes": clock(
                        f"weekend_plug_in_scale_minutes.{cohort_id}"
                        if f"weekend_plug_in_scale_minutes.{cohort_id}" in CONNECTION_CLOCKS
                        else f"plug_in_scale_minutes.{cohort_id}"
                    ),
                }
                for cohort_id, fields in TRIP_BEHAVIOUR.items()
            },
        },
        "base_temperature_c_by_day": np.array(
            [WEATHER["warmup_base_temperature_c"].value] * warmup_days + study_base,
            dtype=np.float64,
        ),
        "weather_sd_c": WEATHER["daily_sd_c"].value,
        "public_charge_assumptions": {
            "efficiency_fraction": resolved["efficiency_fraction"],
            # The records state percent (as the dialog shows it); the
            # kernel takes SoC fractions.
            "top_up_threshold_soc_fraction": resolved["public_top_up_threshold_soc_percent"]
            / 100.0,
            "top_up_target_soc_fraction": resolved["public_top_up_target_soc_percent"] / 100.0,
        },
        # Every SYSTEM, SHOCKS, PRICES and TRADING value by record name, edited
        # where the dialog allows it (sampling.MARKET_PRICE_INPUTS).  ``ar1_phi``
        # is left out: sampling reads that fixed value from its record.
        "price_assumptions": {
            name: resolved.get(name, record.value)
            for section in (SYSTEM, SHOCKS, PRICES, TRADING)
            for name, record in section.items()
            if name != "ar1_phi"
        },
        "public_charge_gbp_per_kwh": resolved["public_charge_gbp_per_kwh"],
        "departure_margin_hours": resolved["departure_margin_hours"],
        # Decision 0007 follow-up: a bounded number_input has no reachable
        # "unset" value in Streamlit, so "off" is the separate
        # ``timed_tariff_enabled`` switch, not a cleared hour.  This is the
        # one place that turns switch + hour back into the NaN-means-off
        # value ``run_forecast``/the kernel already expect, so neither needs
        # to know the switch exists.
        "timed_start_local_hour": (
            resolved["timed_start_local_hour"] if resolved["timed_tariff_enabled"] else float("nan")
        ),
        "not_recovered_material_share_percent": resolved["not_recovered_material_share_percent"],
        # The trading overlay (trading contract v1 §7, §9.9): ledger terms,
        # baseline, commitment share and control group.  Non-response comes
        # from the manufacturer records below (§10.1e).  The commitment
        # rule (§10.4, §10.8) is added on top, edited where the "Firm MW"
        # tab allows: it is not one of ``trading_inputs``'s own keys, so
        # ``forecast._trading_boundary``'s own ``trading_inputs(None)`` call
        # sees the same key set whether or not a caller passes this mapping.
        "trading_assumptions": {
            **trading_inputs(resolved),
            "trading.commitment_rule": str(
                resolved.get(
                    "trading.commitment_rule", AVAILABILITY["trading.commitment_rule"].value
                )
            ),
        },
        # Zones (trading contract v1 §3): shares in ZONE_IDS order, and the
        # optional headrooms (kW, NaN = none), which are not editable yet.
        "zone_shares": tuple(resolved[name] for name in ZONE_SHARE_NAMES),
        "zone_headroom_kw": tuple(ZONES[f"headroom_kw.{zone_id}"].value for zone_id in ZONE_IDS),
        # Shared plug-in factors, holidays and manufacturers (trading
        # contract v1 §10.1), edited where the "Plugging & home charging"
        # and "Firm MW" tabs allow (§10.8; the record's own value otherwise).
        "shared_factor_assumptions": shared_factor_inputs(resolved),
    }
