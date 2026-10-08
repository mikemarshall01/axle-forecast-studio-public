"""Run-shape and cohort types the model is called with.

What this owns: three small frozen types.  ``RunSettings`` is one run's shape
(dates, fleet size, worlds, seed) plus the few scalar assumptions the kernel
reads directly; ``CohortSpec`` and ``CohortFixture`` describe the six source
archetypes the population is drawn from.

How it fits: nothing here holds a value.  ``model/assumptions.py`` is the
single source of every value (decision 0004 item 21) and builds these types
with ``assumptions.run_settings`` and ``assumptions.cohort_fixture``.  This
module was ``axle_studio/scenario.py`` until the module-layout step
(M4-layout, decision 0004 item 15).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class CohortSpec:
    """One source archetype (a row of 'Source archetypes'!A6:P11).

    ``home_charger_limit_kw`` is the home charging power in kW.  Decision
    0004 item 34 dropped the separate vehicle AC limit, so this is the only
    limit on home charging power.

    ``arrival_local_hour`` and ``departure_local_hour`` are the weekday home
    connection window (London clock hours); the ``weekend_*`` pair is the
    weekend window (decision 0004 item 42).  A cohort with no weekend source
    keeps a flat window: its weekend hours equal its weekday hours.
    """

    cohort_id: str
    source_name: str
    source_row: int
    population_share_fraction: float
    daily_miles_mean: float
    daily_miles_sd: float
    battery_capacity_kwh: float
    efficiency_miles_per_battery_kwh: float
    plug_probability: float
    home_charger_limit_kw: float
    preferred_target_soc_fraction: float
    arrival_local_hour: int
    departure_local_hour: int
    weekend_arrival_local_hour: int
    weekend_departure_local_hour: int


@dataclass(frozen=True, slots=True)
class CohortFixture:
    """The six cohorts a population is drawn from, in source row order."""

    label: str
    evidence_kind: str
    required_vehicle_count: int
    cohorts: tuple[CohortSpec, ...]


@dataclass(frozen=True)
class RunSettings:
    """One explicit run's shape and the scalar assumptions the kernel reads.

    Dates are London civil dates; the study is ``study_days`` session nights
    of 48 UTC half-hours from London 12:00 on ``start_local_date`` (decision
    0004 item 52) and warm-up days count back from that instant (item 1).
    SoC and efficiency values are fractions in [0, 1]; the weather
    sensitivity is the fractional driving-efficiency loss per degree C.
    """

    start_local_date: date
    warmup_days: int
    study_days: int
    vehicle_count: int
    seed: int
    evaluation_world_count: int
    opening_soc_fraction: float
    reserve_soc_fraction: float
    home_charge_efficiency: float
    weather_efficiency_sensitivity_fraction_per_c: float

    @property
    def sampled_day_count(self) -> int:
        """London dates the samplers draw trips, sessions and weather for.

        The span runs from London noon ``warmup_days`` before the start date
        to noon ``study_days`` after it, so it touches one more calendar date
        than it has days: the last night's morning departure and trip fall on
        the date after the last session night (decision 0004 item 52).  Date
        index d is ``start_local_date - warmup_days + d``.
        """

        return self.warmup_days + self.study_days + 1


__all__ = ["CohortFixture", "CohortSpec", "RunSettings"]
