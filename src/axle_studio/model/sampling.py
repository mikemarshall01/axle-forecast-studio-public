"""Every random draw of a run, in one place.

What this module owns: the persistent EV population (including each EV's
charger manufacturer), each world's daily trips, the shared plug-in skip
factors, home plug-in sessions, daily weather, manufacturer outages and
non-response draws, and the synthetic market prices (each world's system
net demand, day-ahead, intraday and imbalance paths).  All of them take the
run's one ``np.random.default_rng(seed)`` (decision 0004 item 14)
and use named ``scipy.stats`` distributions, or a statsmodels AR(1) process
for price noise.

How it fits: ``forecast`` calls these in a fixed order (trading contract v1
§10.0): the population (ending with the charger manufacturers); then for
each set of worlds home departures, trips, the shared plug-in factors,
connections and weather; then, in the action model only, the market price
channels, the non-response uniforms and, last, the manufacturer outages.
So a seed reproduces a run exactly.  The outputs are plain arrays of shape
(world, day, EV), with event times as UTC ``datetime64[ns]``; ``physics`` turns
them into energy.

Common random numbers (plan M6): every sampler takes a fixed number of draws
whatever its parameter values, so editing one assumption changes only its
own channel and Compare keeps matched futures.  The comments at each draw say
which "obvious" scipy call was rejected because it would skip draws.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from statsmodels.tsa.arima_process import ArmaProcess

from axle_studio.model.assumptions import (
    MANUFACTURER_IDS,
    MANUFACTURER_SHARE_SUM_TOLERANCE,
    PRICES,
    ZONE_IDS,
    ZONES,
)
from axle_studio.model.clock import london_wall_time_to_utc, utc_half_hour_boundaries
from axle_studio.model.settings import CohortFixture, RunSettings

# ============================================================================
# Population: the persistent EVs, drawn once per run
# ============================================================================
#
# The EV population a run is built on: cohort assignment and, optionally, a
# persistent personal mileage multiplier per EV.  These traits are drawn once
# from the run's single ``np.random.Generator`` (decision 0004 item 14) and are
# shared by every future world; the trip and connection samplers draw the
# day-to-day variation.


# Population columns, in order.  ``home_charger_limit_kw`` is each EV's home
# charging power; decision 0004 item 34 dropped the separate vehicle AC limit
# column.  ``reserve_soc_fraction`` is carried for display; the kernel does
# not read it.  The ``runtime_*_local_hour`` columns are the home connection
# window: weekday hours, then weekend hours (decision 0004 item 42; equal to
# the weekday hours for a flat-window cohort).
UNIT_COLUMNS = (
    "population_id",
    "unit_id",
    "cohort_id",
    "cohort_source_name",
    "daily_miles_mean",
    "daily_miles_sd",
    "plug_probability",
    "physical_capacity_kwh",
    "efficiency_miles_per_battery_kwh",
    "home_charger_limit_kw",
    "preferred_target_soc_fraction",
    "reserve_soc_fraction",
    "runtime_departure_local_hour",
    "runtime_arrival_local_hour",
    "runtime_weekend_departure_local_hour",
    "runtime_weekend_arrival_local_hour",
    "travel_window_half_hours",
    "zone_id",
)
# Upper bound on the persistent personal-mileage CV.  A CV of 10 already means a
# lognormal log-SD of about 2.15, so most EVs drive a small fraction of the cohort
# mean and a few drive many times it; larger values are not a meaningful mileage
# spread.  Rejecting them also keeps CV² and the lognormal draw far from float
# overflow (CV above ~1.3e154 overflowed CV² and produced zero mileage).
_MAX_PERSONAL_MILEAGE_CV = 10.0


def _cv_by_cohort(
    fixture: CohortFixture,
    values: Mapping[str, float],
    *,
    field_name: str,
    label: str,
    max_cv: float = math.inf,
) -> dict[str, float]:
    """Validate one non-negative CV (at most ``max_cv``) per fixture cohort."""

    expected_ids = {cohort.cohort_id for cohort in fixture.cohorts}
    if set(values) != expected_ids:
        raise ValueError(f"{field_name} must contain exactly the fixture cohort IDs")

    validated: dict[str, float] = {}
    for cohort_id, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} CVs must be finite and non-negative")
        try:
            numeric_value = float(value)
        except OverflowError as error:
            raise ValueError(f"{label} CVs must be finite and non-negative") from error
        if not math.isfinite(numeric_value) or numeric_value < 0.0:
            raise ValueError(f"{label} CVs must be finite and non-negative")
        if numeric_value > max_cv:
            raise ValueError(f"{label} CVs must be at most {max_cv:g}")
        validated[cohort_id] = numeric_value
    return validated


def _daily_miles_cv_by_cohort(
    fixture: CohortFixture,
    values: Mapping[str, float],
) -> dict[str, float]:
    return _cv_by_cohort(
        fixture, values, field_name="daily_miles_cv_by_cohort", label="daily mileage"
    )


def _daily_miles_sd(mean: float, coefficient_of_variation: float) -> float:
    derived = mean * coefficient_of_variation
    if not math.isfinite(derived):
        raise ValueError("derived daily mileage standard deviations must be finite")
    return derived


def _personal_mileage_cv_by_cohort(
    fixture: CohortFixture,
    values: Mapping[str, float],
) -> dict[str, float]:
    return _cv_by_cohort(
        fixture,
        values,
        field_name="personal_mileage_cv_by_cohort",
        label="personal mileage",
        max_cv=_MAX_PERSONAL_MILEAGE_CV,
    )


def _cohort_counts(vehicle_count: int, fixture: CohortFixture) -> tuple[int, ...]:
    if isinstance(vehicle_count, bool) or not isinstance(vehicle_count, int) or vehicle_count < 1:
        raise ValueError("vehicle_count must be a positive integer")
    if len(fixture.cohorts) != 6:
        raise ValueError("cohort fixture must contain exactly six cohorts")
    if (
        isinstance(fixture.required_vehicle_count, bool)
        or not isinstance(fixture.required_vehicle_count, int)
        or fixture.required_vehicle_count < 1
    ):
        raise ValueError("fixture required_vehicle_count must be a positive integer")

    shares = tuple(cohort.population_share_fraction for cohort in fixture.cohorts)
    if any(
        isinstance(share, bool)
        or not isinstance(share, (int, float))
        or not math.isfinite(share)
        or share < 0.0
        or share > 1.0
        for share in shares
    ):
        raise ValueError("cohort population shares must be finite and between zero and one")
    # 1e-3 (0.1 percentage points), not the bit-exact tolerance a fixed source
    # fixture could use: decision 0004 item 54 makes the six shares an
    # editable draft (percent fields, dialog-checked to only +/-0.01 points
    # in assumptions.validation_errors), and dividing a typed percent by 100
    # does not reproduce the literal source fractions' floating-point sum.
    # This still rejects a genuinely wrong fixture (for example one cohort's
    # share edited to 0.5 with the other five untouched, off by 10 points).
    if not math.isclose(math.fsum(shares), 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError("cohort population shares must sum to one")

    return _largest_remainder_counts(vehicle_count, shares)


def _largest_remainder_counts(vehicle_count: int, shares: tuple[float, ...]) -> tuple[int, ...]:
    """Exact whole counts summing to ``vehicle_count``, closest to ``share × vehicle_count``.

    Largest remainder: floor every exact count, then give the EVs left over
    to the largest fractional parts, ties to the lower index.  Used for both
    the cohort mix and the zone split, so both are exact, not sampled.
    """

    exact = tuple(share * vehicle_count for share in shares)
    counts = [math.floor(value) for value in exact]
    remaining = vehicle_count - sum(counts)
    order = sorted(
        range(len(exact)),
        key=lambda index: (-(exact[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    if sum(counts) != vehicle_count:
        raise ValueError("counts must sum to vehicle_count")
    return tuple(counts)


def _zone_counts(vehicle_count: int, zone_shares: Sequence[float] | None) -> tuple[int, ...]:
    """EVs per zone in ``ZONE_IDS`` order (trading contract v1 §3)."""

    if zone_shares is None:
        zone_shares = [ZONES[f"share.{zone_id}"].value for zone_id in ZONE_IDS]
    shares = tuple(float(share) for share in zone_shares)
    if len(shares) != len(ZONE_IDS):
        raise ValueError(f"zone_shares must hold {len(ZONE_IDS)} shares")
    if any(not math.isfinite(share) or not 0.0 <= share <= 1.0 for share in shares):
        raise ValueError("zone shares must be finite and between zero and one")
    if not math.isclose(math.fsum(shares), 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError("zone shares must sum to one")
    return _largest_remainder_counts(vehicle_count, shares)


def build_population(
    settings: RunSettings,
    fixture: CohortFixture,
    rng: np.random.Generator,
    *,
    daily_miles_cv_by_cohort: Mapping[str, float] | None = None,
    personal_mileage_cv_by_cohort: Mapping[str, float] | None = None,
    zone_shares: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Build one population whose completed traits persist across all future worlds.

    The source annual-mile centre is a source-backed cohort value.  When requested,
    the personal mileage overlay draws one persistent mean-one multiplier per EV and
    stores its daily expectation as ``daily_miles_mean``.  The daily CV controls
    positive trip-distance sampling in ``forecast``; ``daily_miles_sd`` is
    its displayed ``daily_miles_mean × CV`` value.

    ``zone_shares`` are the fractions of the fleet in each illustrative
    network zone, in ``assumptions.ZONE_IDS`` order (``None``: the ``ZONES``
    defaults, 0.25 each); each EV's zone is the ``zone_id`` column.
    """

    if daily_miles_cv_by_cohort is not None and not isinstance(daily_miles_cv_by_cohort, Mapping):
        raise TypeError("daily_miles_cv_by_cohort must be a mapping or None")
    if personal_mileage_cv_by_cohort is not None and not isinstance(
        personal_mileage_cv_by_cohort, Mapping
    ):
        raise TypeError("personal_mileage_cv_by_cohort must be a mapping or None")

    counts = _cohort_counts(settings.vehicle_count, fixture)
    zone_counts = _zone_counts(settings.vehicle_count, zone_shares)
    validated_cvs = (
        None
        if daily_miles_cv_by_cohort is None
        else _daily_miles_cv_by_cohort(fixture, daily_miles_cv_by_cohort)
    )
    validated_personal_cvs = (
        None
        if personal_mileage_cv_by_cohort is None
        else _personal_mileage_cv_by_cohort(fixture, personal_mileage_cv_by_cohort)
    )
    cohort_indices = np.concatenate(
        [np.full(count, index, dtype=int) for index, count in enumerate(counts)]
    )
    assignments = rng.permutation(cohort_indices)
    unit_ids = tuple(f"ev-{index:03d}" for index in range(1, settings.vehicle_count + 1))

    cohorts = [fixture.cohorts[int(cohort_index)] for cohort_index in assignments]
    daily_miles_means = np.array([cohort.daily_miles_mean for cohort in cohorts], dtype=float)
    if validated_personal_cvs is not None:
        # Each EV keeps one persistent mileage multiplier, mean one so the
        # source-backed cohort centre stays the population expectation:
        # lognormal with log-SD s = sqrt(log(1 + CV²)) and scale exp(-s²/2),
        # since E[lognorm(s, scale)] = scale × exp(s²/2).  A lognormal keeps
        # mileage positive.  CV is validated to at most 10, so log1p(CV²) is
        # accurate and finite and no separate large-CV formula is needed.
        personal_cvs = np.array(
            [validated_personal_cvs[cohort.cohort_id] for cohort in cohorts], dtype=float
        )
        sigma = np.sqrt(np.log1p(personal_cvs**2))
        # scipy rejects s = 0, so zero-CV EVs draw with a placeholder s = 1 and
        # keep multiplier 1.  Every EV still takes one draw, so a CV edit never
        # shifts the draws seen by other EVs or later channels (plan M6).
        positive_cv = personal_cvs > 0.0
        draw_sigma = np.where(positive_cv, sigma, 1.0)
        multipliers = stats.lognorm(s=draw_sigma, scale=np.exp(-0.5 * draw_sigma**2)).rvs(
            random_state=rng
        )
        daily_miles_means = daily_miles_means * np.where(positive_cv, multipliers, 1.0)

    # Zones (trading contract v1 §3, §4.8): exact counts per zone, then one
    # permutation, the same method as the cohorts, so zones are independent
    # of cohorts.  It is the last population draw and its size depends only
    # on vehicle_count, so editing a zone share moves no other draw.
    zone_indices = np.concatenate(
        [np.full(count, index, dtype=int) for index, count in enumerate(zone_counts)]
    )
    zone_ids = [ZONE_IDS[int(index)] for index in rng.permutation(zone_indices)]

    trait_rows: list[dict[str, object]] = []
    for unit_id, cohort, daily_miles_mean, zone_id in zip(
        unit_ids, cohorts, daily_miles_means.tolist(), zone_ids, strict=True
    ):
        travel_slots = (cohort.arrival_local_hour - cohort.departure_local_hour) % 24 * 2
        trait_rows.append(
            {
                "unit_id": unit_id,
                "cohort_id": cohort.cohort_id,
                "cohort_source_name": cohort.source_name,
                "daily_miles_mean": daily_miles_mean,
                "daily_miles_sd": (
                    _daily_miles_sd(
                        daily_miles_mean,
                        validated_cvs[cohort.cohort_id],
                    )
                    if validated_cvs is not None
                    else cohort.daily_miles_sd
                ),
                "plug_probability": cohort.plug_probability,
                "physical_capacity_kwh": cohort.battery_capacity_kwh,
                "efficiency_miles_per_battery_kwh": (cohort.efficiency_miles_per_battery_kwh),
                "home_charger_limit_kw": cohort.home_charger_limit_kw,
                "preferred_target_soc_fraction": cohort.preferred_target_soc_fraction,
                "reserve_soc_fraction": settings.reserve_soc_fraction,
                "runtime_departure_local_hour": cohort.departure_local_hour,
                "runtime_arrival_local_hour": cohort.arrival_local_hour,
                "runtime_weekend_departure_local_hour": cohort.weekend_departure_local_hour,
                "runtime_weekend_arrival_local_hour": cohort.weekend_arrival_local_hour,
                "travel_window_half_hours": travel_slots,
                "zone_id": zone_id,
            }
        )

    population_traits = tuple(
        tuple(row[column] for column in UNIT_COLUMNS[1:]) for row in trait_rows
    )
    population_id = "population:" + sha256(repr(population_traits).encode()).hexdigest()
    return pd.DataFrame.from_records(
        [{"population_id": population_id, **row} for row in trait_rows],
        columns=UNIT_COLUMNS,
    )


def assign_manufacturers(
    rng: np.random.Generator, vehicle_count: int, shares: Sequence[float]
) -> np.ndarray:
    """Each EV's charger manufacturer, as an index into ``assumptions.MANUFACTURER_IDS``.

    ``shares`` are the fractions of the fleet per maker (they must sum to 1).
    Returns an int64 (EV,) array.  Trading contract v1 §10.1d: exact counts
    by largest remainder (ties to the lower index), then one permutation of
    the EVs, the zone method, so makers are independent of cohort, zone and
    control group.  This is the last population draw (§10.0) and takes
    ``vehicle_count`` draws whatever the shares, so editing a share moves no
    other draw and a Compare of two maker mixes runs on identical futures.
    """

    values = tuple(float(share) for share in shares)
    if len(values) != len(MANUFACTURER_IDS):
        raise ValueError(f"manufacturer shares must hold {len(MANUFACTURER_IDS)} values")
    if any(not math.isfinite(share) or not 0.0 <= share <= 1.0 for share in values):
        raise ValueError("manufacturer shares must be finite and between zero and one")
    if not math.isclose(
        math.fsum(values), 1.0, rel_tol=0.0, abs_tol=MANUFACTURER_SHARE_SUM_TOLERANCE
    ):
        raise ValueError("manufacturer shares must sum to one")
    counts = _largest_remainder_counts(vehicle_count, values)
    indices = np.concatenate(
        [np.full(count, index, dtype=np.int64) for index, count in enumerate(counts)]
    )
    return rng.permutation(indices)


# ============================================================================
# Trips: whether each EV drives, how far and when it leaves
# ============================================================================
#
# The per-world daily travel draws: whether each EV drives, how far it goes
# and when it leaves.  Every draw comes from the run's single
# ``np.random.Generator`` through named ``scipy.stats`` distributions (decision
# 0004 item 14); the physics kernel turns these inputs into energy.


def _as_float_vector(value: object, name: str, size: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric array with shape ({size},)") from error
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _as_float_matrix(value: object, name: str, days: int, size: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric array with shape ({days}, {size})") from error
    if result.shape != (days, size):
        raise ValueError(f"{name} must have shape ({days}, {size})")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _validate_scalar(value: object, name: str, *, multiple_of: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    if multiple_of is not None and not np.isclose(
        result / multiple_of, round(result / multiple_of), rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"{name} must be a multiple of {multiple_of:g}")
    return result


def _check_journeys_fit_the_kernel(
    drives: np.ndarray,
    outbound_miles: np.ndarray,
    return_miles: np.ndarray,
    trip_departure: np.ndarray,
    dwell_seconds: np.ndarray,
    speeds: np.ndarray,
) -> None:
    """Check every driving day is a journey the kernel can time in int64 nanoseconds.

    A driving day needs two positive legs.  The kernel adds leg and dwell
    times to the departure as int64 nanoseconds, so an extreme caller value
    (a speed of 1e308 mph, a distance of 1e100 miles) would otherwise
    overflow or round a leg to zero time without any error.  Checked on
    whole arrays; the order of checks decides which message a case gets.
    """

    if not drives.any():
        return
    legs = np.stack((outbound_miles[drives], return_miles[drives]))
    if (legs <= 0.0).any():
        raise ValueError("sampled driving events must have two positive mileage legs")
    int64_max = float(np.iinfo(np.int64).max)
    with np.errstate(all="ignore"):
        leg_ns = 3600.0 * legs / speeds[drives] * 1e9
        dwell_ns = dwell_seconds[drives] * 1e9
    if not np.isfinite(leg_ns).all() or (leg_ns >= int64_max).any():
        raise ValueError("sampled journey leg durations must be finite, positive and representable")
    if (leg_ns < 1.0).any():
        raise ValueError("sampled journey leg durations must be at least one nanosecond")
    if not np.isfinite(dwell_ns).all() or (dwell_ns >= int64_max).any():
        raise ValueError("sampled destination dwell must be representable")
    departure_ns = trip_departure[drives].astype("datetime64[ns]").view(np.int64)
    journey_ns = leg_ns.sum(axis=0) + dwell_ns
    if (journey_ns > float(pd.Timestamp.max.value) - departure_ns).any():
        raise ValueError("sampled journey return timestamp is outside the pandas range")


def sample_daily_trip_inputs(
    settings: RunSettings,
    units: pd.DataFrame,
    rng: np.random.Generator,
    *,
    departure_utc: np.ndarray,
    drive_probability: np.ndarray,
    expected_daily_miles: np.ndarray,
    distance_cv: np.ndarray,
    destination_dwell_minutes: np.ndarray,
    drive_speed_mph: np.ndarray,
    desired_pre_drive_soc_fraction: np.ndarray,
) -> dict[str, np.ndarray]:
    """Sample illustrative daily trip inputs for every evaluation world.

    The caller supplies all behavioural inputs, including day-type variation.  Annual
    mileage and EV properties remain fields of ``units`` and are not sampled here.

    ``departure_utc`` is ``sample_departure_times``'s (world, days + 1, EV)
    array: a driving day's trip starts at that day's home departure, the same
    instant the EV unplugs (decision 0004 item 51), so this function draws no
    departure time of its own.  Per-(day, EV) inputs: ``drive_probability``
    (0–1) and ``expected_daily_miles`` (unconditional daily mean, miles).
    Per-EV inputs: ``distance_cv`` (driving-day distance CV), speed (mph) and
    desired pre-drive SoC (0–1).  ``destination_dwell_minutes`` is
    per EV or per (day, EV), so it can differ at weekends (decision 0004 item
    42 follow-up); it draws nothing, so its shape never changes the draws.  Returns (world, day, EV)
    arrays keyed as the physics kernel expects; each leg is half the day's miles.
    """

    vehicle_count = settings.vehicle_count
    days = settings.sampled_day_count
    worlds = settings.evaluation_world_count
    shape = (worlds, days, vehicle_count)
    probabilities = _as_float_matrix(drive_probability, "drive_probability", days, vehicle_count)
    expected_miles = _as_float_matrix(
        expected_daily_miles, "expected_daily_miles", days, vehicle_count
    )
    distance_cvs = _as_float_vector(distance_cv, "distance_cv", vehicle_count)
    departures = np.asarray(departure_utc)
    if departures.dtype.kind != "M" or departures.shape != (worlds, days + 1, vehicle_count):
        raise ValueError(
            f"departure_utc must be datetime64 with shape ({worlds}, {days + 1}, {vehicle_count})"
        )
    dwell_minutes = (
        _as_float_matrix(
            destination_dwell_minutes, "destination_dwell_minutes", days, vehicle_count
        )
        if np.ndim(destination_dwell_minutes) == 2
        else _as_float_vector(destination_dwell_minutes, "destination_dwell_minutes", vehicle_count)
    )
    speeds = _as_float_vector(drive_speed_mph, "drive_speed_mph", vehicle_count)
    desired_soc = _as_float_vector(
        desired_pre_drive_soc_fraction, "desired_pre_drive_soc_fraction", vehicle_count
    )

    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("drive_probability must be between 0 and 1")
    if (distance_cvs < 0.0).any():
        raise ValueError("distance_cv must be non-negative")
    if (dwell_minutes < 0.0).any():
        raise ValueError("destination_dwell_minutes must be non-negative")
    if (speeds <= 0.0).any():
        raise ValueError("drive_speed_mph must be positive")
    if (desired_soc < 0.0).any() or (desired_soc > 1.0).any():
        raise ValueError("desired_pre_drive_soc_fraction must be between 0 and 1")
    if ((probabilities == 0.0) & (expected_miles != 0.0)).any():
        raise ValueError("expected_daily_miles must be zero when drive_probability is zero")
    if ((probabilities > 0.0) & (expected_miles <= 0.0)).any():
        raise ValueError("expected_daily_miles must be positive when drive_probability is positive")

    # Common random numbers (plan M6): the two channels below (drive, distance)
    # each consume exactly one full (world, day, EV) block in a fixed order,
    # whatever the probabilities or CVs are.  A parameter edit
    # therefore changes only its own channel, and Compare keeps matched futures.
    #
    # Drives are Bernoulli(p) by inverse transform (drive when U < p).
    # stats.bernoulli.rvs was rejected: it calls Generator.binomial, which takes
    # no draw when p is exactly 0, so the draw count would depend on p.
    drives = stats.uniform.rvs(size=shape, random_state=rng) < probabilities
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            conditional_means = np.divide(
                expected_miles,
                probabilities,
                out=np.zeros_like(expected_miles),
                where=probabilities > 0.0,
            )
            # Trip distance on a driving day = conditional mean × a mean-one
            # lognormal multiplier with the cohort CV: log-SD s = sqrt(log(1 + CV²))
            # and scale exp(-s²/2) so E[multiplier] = 1 and the configured daily
            # expectation is preserved.  A lognormal keeps distances positive.
            sigma = np.sqrt(np.log1p(distance_cvs**2))
            # scipy rejects s = 0, so zero-CV EVs draw with a placeholder s = 1
            # and their multiplier is then fixed at 1 (a deterministic distance).
            # Drawing for them anyway keeps the draw count independent of CV.
            positive_cv = distance_cvs > 0.0
            draw_sigma = np.where(positive_cv, sigma, 1.0)
            multiplier = stats.lognorm(s=draw_sigma, scale=np.exp(-0.5 * draw_sigma**2)).rvs(
                size=shape, random_state=rng
            )
            multiplier[:, :, ~positive_cv] = 1.0
            total_miles = conditional_means * multiplier
    except FloatingPointError as error:
        raise ValueError("daily mileage parameters or outputs must remain finite") from error
    if not np.isfinite(total_miles).all():
        raise ValueError("daily mileage parameters or outputs must remain finite")
    total_miles[~drives] = 0.0

    # Plug-out is departure (decision 0004 item 51): the trip starts at the
    # day's sampled home departure, and a non-driving day has no trip.
    trip_departure = departures[:, :days, :].astype("datetime64[ns]", copy=True)
    trip_departure[~drives] = np.datetime64("NaT")
    try:
        with np.errstate(over="raise", invalid="raise"):
            dwell_seconds = dwell_minutes * 60.0
    except FloatingPointError as error:
        raise ValueError(
            "destination_dwell_minutes is too large to represent in seconds"
        ) from error
    if not np.isfinite(dwell_seconds).all():
        raise ValueError("destination_dwell_minutes is too large to represent in seconds")

    outbound_miles = total_miles / 2.0
    return_miles = total_miles / 2.0
    dwell_by_event = np.broadcast_to(dwell_seconds, shape).copy()
    speeds_by_event = np.broadcast_to(speeds, shape).copy()
    _check_journeys_fit_the_kernel(
        drives,
        outbound_miles,
        return_miles,
        trip_departure,
        dwell_by_event,
        speeds_by_event,
    )

    return {
        "drives_today": drives,
        "outbound_miles": outbound_miles,
        "return_miles": return_miles,
        "trip_departure_utc": trip_departure,
        "destination_dwell_seconds": dwell_by_event,
        "drive_speed_mph": speeds_by_event,
        "desired_pre_drive_soc_fraction": np.broadcast_to(desired_soc, shape).copy(),
    }


# ============================================================================
# Home clocks and connections: departures and plug-in sessions
# ============================================================================
#
# The home clock draws (decision 0004 item 51): one departure per EV and
# London date, which is both the unplug and the trip start, and, for each
# night, whether the EV plugs in and an independent plug-in time.  Both clocks
# are truncated Student-t shifts around the cohort's clock hours.  They
# consume the run's single ``np.random.Generator`` (decision 0004 item 14) and
# return plain arrays that the physics kernel reads; they make no charging or
# travel decisions.


_ALWAYS_PLUGGED_COHORT = "always_plugged_in"


def connection_window_hours(units: pd.DataFrame, end: str) -> tuple[np.ndarray, np.ndarray]:
    """Weekday and weekend connection clock hours per EV for ``end`` ("arrival"/"departure").

    Decision 0004 item 42: the weekend hours are the population's
    ``runtime_weekend_*`` columns.  A hand-built unit frame without them has
    flat windows (weekend = weekday), the same rule the four cohorts with no
    weekend source follow.
    """

    weekday = units[f"runtime_{end}_local_hour"].to_numpy(dtype=float, copy=True)
    weekend_column = f"runtime_weekend_{end}_local_hour"
    if weekend_column not in units:
        return weekday, weekday.copy()
    return weekday, units[weekend_column].to_numpy(dtype=float, copy=True)


def _check_connection_clocks(units: pd.DataFrame) -> None:
    """Check the cohort values the sessions are built from (from the cohort fixture).

    The population frame itself is built by the model, so only these
    caller-supplied values are checked: a probability in [0, 1] and whole
    London clock hours, with arrival and departure hours different on the
    same day type.
    """

    # Written as "not inside" so NaN fails too.
    probabilities = units["plug_probability"].to_numpy(dtype=float)
    if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
        raise ValueError("plug_probability must be between 0 and 1")
    arrivals = connection_window_hours(units, "arrival")
    departures = connection_window_hours(units, "departure")
    for end, hours in (("arrival", arrivals), ("departure", departures)):
        if not np.isin(np.concatenate(hours), np.arange(24)).all():
            raise ValueError(f"runtime_{end}_local_hour must be an integer between 0 and 23")
    if any((arrival == departure).any() for arrival, departure in zip(arrivals, departures)):
        raise ValueError("connection reference hours must differ")


def _local_midnights(settings: RunSettings, count: int) -> tuple[np.ndarray, np.ndarray]:
    """The first ``count`` London dates of the run and whether each is a weekend.

    Date index d is ``first_local_date + d`` (warm-up days first), as
    ``datetime64[D]``, with a (date, 1) weekend flag ready to broadcast over EVs.
    """

    first_local_date = settings.start_local_date - timedelta(days=settings.warmup_days)
    local_midnight = np.datetime64(first_local_date, "D") + np.arange(count)
    # numpy's weekday: 1970-01-01 (day 0) was a Thursday, so Monday = 0.
    weekend = ((local_midnight.astype(np.int64) + 3) % 7 >= 5)[:, np.newaxis]
    return local_midnight, weekend


def _validate_t_df(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("clock t degrees of freedom must be a finite positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("clock t degrees of freedom must be a finite positive number")
    return result


def _scale_by_unit(
    units: pd.DataFrame, scale_by_cohort: Mapping[str, float], name: str
) -> np.ndarray:
    """Per-EV clock scale (minutes) from a per-cohort mapping, each value checked."""

    return np.array(
        [_validate_scalar(scale_by_cohort[c], f"{name} for {c}") for c in units["cohort_id"]],
        dtype=float,
    )


def truncated_t_shift_minutes(
    uniforms: np.ndarray, scale_minutes: np.ndarray, t_df: float, clip_minutes: float
) -> np.ndarray:
    """Clock shifts (minutes) from U(0, 1) draws: a truncated Student-t, in half hours.

    ``uniforms`` has any shape; ``scale_minutes`` (the t scale, minutes, not
    the standard deviation) broadcasts against it, and 0 means no shift.
    Returns shifts rounded to whole half hours and never beyond
    ±``clip_minutes`` (a multiple of 30).

    Decision 0004 item 51: ``shift = t(df).ppf(lo + U·(hi − lo))·scale`` with
    ``lo, hi = t.cdf(∓clip/scale)``.  This is inverse-transform sampling from
    the t restricted to ±clip, so the tail mass is spread inside the bound
    rather than piled on exactly ±clip as clipping would do.  A Student-t,
    not a normal, because real plug-in and plug-out times have fat tails (the
    odd very early or late day); df ≥ 30 is practically a normal.  One
    uniform per cell whatever the scale or df keeps the draw count fixed
    (common random numbers, plan M6).
    """

    scale = np.broadcast_to(np.asarray(scale_minutes, dtype=float), uniforms.shape)
    moving = scale > 0.0
    distribution = stats.t(t_df)
    # A zero scale gives bound 0, so lo = hi = 0.5 and the shift is exactly 0.
    bound = np.divide(clip_minutes, scale, out=np.zeros_like(scale), where=moving)
    low = distribution.cdf(-bound)
    high = distribution.cdf(bound)
    raw = distribution.ppf(low + uniforms * (high - low)) * scale
    # Round to the half hour the kernel works in; the final clip only guards
    # float error at the bound (the truncation already keeps |raw| <= clip).
    shifts = np.where(moving, np.rint(raw / 30.0) * 30.0, 0.0)
    return np.clip(shifts, -clip_minutes, clip_minutes)


def sample_departure_times(
    settings: RunSettings,
    units: pd.DataFrame,
    rng: np.random.Generator,
    *,
    scale_minutes_by_cohort: Mapping[str, float],
    t_df: float,
    clip_minutes: float,
) -> np.ndarray:
    """Home departure instants for every world, London date and EV.

    Decision 0004 item 51: plug-out is departure.  One draw per (world, date,
    EV) is both the moment the EV unplugs at home and, on a driving day, the
    moment its trip starts, so ``sample_daily_trip_inputs`` and
    ``sample_connection_opportunities`` both take this array.  It is centred
    on the cohort's home departure clock hour for the date's day type
    (``connection_window_hours(units, "departure")``), the same typical
    departure the smart charger plans to, less its margin; an early departure
    is simply leaving before that plan finishes.

    ``scale_minutes_by_cohort`` is the t scale in minutes per cohort, ``t_df``
    its degrees of freedom and ``clip_minutes`` the truncation bound (see
    ``truncated_t_shift_minutes``).  Returns a (world, days + 1, EV)
    ``datetime64[ns]`` array of naive UTC instants for London dates
    ``first_local_date + d``; the extra last date ends the last night's
    session, so every session end is a sampled departure.  Draws exactly one
    (world, days + 1, EV) block of uniforms whatever the parameters.
    """

    clip = _validate_scalar(clip_minutes, "clock_clip_minutes", multiple_of=30.0)
    df = _validate_t_df(t_df)
    scale = _scale_by_unit(units, scale_minutes_by_cohort, "departure scale")
    _check_connection_clocks(units)
    dates = settings.sampled_day_count + 1
    local_midnight, weekend = _local_midnights(settings, dates)
    weekday_hours, weekend_hours = connection_window_hours(units, "departure")
    departure_minutes = 60.0 * np.where(weekend, weekend_hours, weekday_hours)
    # Every departure must stay on its own London date, because the kernel
    # indexes a day's trip (and its weather) by that date.
    if ((departure_minutes - clip < 0.0) | (departure_minutes + clip >= 24 * 60)).any():
        raise ValueError("home departure hours and clock_clip_minutes must stay within a local day")

    shape = (settings.evaluation_world_count, dates, settings.vehicle_count)
    uniforms = stats.uniform.rvs(size=shape, random_state=rng)
    shifts = truncated_t_shift_minutes(uniforms, scale, df, clip)
    return london_wall_time_to_utc(
        local_midnight[np.newaxis, :, np.newaxis], departure_minutes[np.newaxis] + shifts
    )


def sample_plug_in_factors(
    rng: np.random.Generator, world_count: int, day_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """The shared plug-in factors: one week factor per world, one day factor per (world, date).

    Returns ``(y, z)``: ``y`` a float64 (world,) array and ``z`` a float64
    (world, day) array of standard normals, in that draw order (trading
    contract v1 §10.1b, decision 0004 items 59 and 60).  ``plug_in_skip_share``
    turns them into the night's skip share.  They are drawn straight after
    the trips and before the plug uniforms they act on (§10.0), always, one
    block each whatever the SDs, median or holidays are: a zero SD still
    draws, so editing any skip-share record moves no other draw.
    """

    week = stats.norm.rvs(size=world_count, random_state=rng)
    day = stats.norm.rvs(size=(world_count, day_count), random_state=rng)
    return np.asarray(week, dtype=np.float64), np.asarray(day, dtype=np.float64)


def plug_in_skip_share(
    week_factor: np.ndarray,
    day_factor: np.ndarray,
    skip_logit_shift: np.ndarray,
    *,
    median: float,
    day_sd: float,
    week_sd: float,
) -> np.ndarray:
    """The shared skip share ``q`` per (world, date): the share of would-be sessions not plugged in.

    ``week_factor`` (world,) and ``day_factor`` (world, day) are
    ``sample_plug_in_factors``'s normals; ``skip_logit_shift`` (day,) is the
    holiday shift ``h_d`` (``clock.holiday_flags``); ``median`` is the
    median skip share on an ordinary night (fraction) and the SDs are on the
    logit scale.  Returns a float64 (world, day) array in [0, 1).

    Trading contract v1 §10.1b (decision 0004 item 60, option A):
    ``logit q = logit(median) + week_sd·y + day_sd·z + h``.  The shock sits
    on the logit of the *skip* share, not of the plug probability, because
    five of the six source cohorts plug in with probability 1 and
    ``logit(1)`` is infinite, so a shock there could not move them.  The
    logit keeps ``q`` inside (0, 1) whatever the shock; ``median`` is a
    median, not a mean, because the logit-normal is symmetric only on the
    logit scale.  A median of 0 gives ``q = 0`` everywhere.
    """

    logit_q = (
        logit(median)
        + week_sd * week_factor[:, np.newaxis]
        + day_sd * day_factor
        + skip_logit_shift[np.newaxis, :]
    )
    return expit(logit_q)


def sample_connection_opportunities(
    settings: RunSettings,
    units: pd.DataFrame,
    rng: np.random.Generator,
    *,
    departure_utc: np.ndarray,
    drives_today: np.ndarray,
    plug_in_scale_minutes_by_cohort: Mapping[str, float],
    t_df: float,
    clip_minutes: float,
    weekend_plug_in_scale_minutes_by_cohort: Mapping[str, float] | None = None,
    skip_share: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample accepted home-connection sessions for each evaluation world.

    Cohort clocks and probabilities are illustrative caller inputs.  A
    session runs from the plug-in on London date d to the home departure on
    the first later date the EV drives, taken from ``departure_utc``
    (``sample_departure_times``'s (world, days + 1, EV) array, drawn with the
    same ``t_df`` and ``clip_minutes``): plug-out is departure (decision 0004
    item 51), and an EV that does not drive stays plugged in (item 68), so a
    session can span several nights.  ``drives_today`` is
    ``sample_daily_trip_inputs``'s (world, day, EV) boolean drive flag.  While
    a session is open, later evenings start no new session (their plug
    uniforms go unused).  A session still open after the last sampled date
    ends at the extra last departure, past the horizon end.  The
    plug-in is an independent truncated-t shift (``truncated_t_shift_minutes``)
    around the cohort's arrival clock hour for the date's day type, with t
    scale ``plug_in_scale_minutes_by_cohort`` on weekdays and
    ``weekend_plug_in_scale_minutes_by_cohort`` at weekends (none means the
    weekday scales apply every day).

    ``skip_share`` (world, day), from ``plug_in_skip_share``, is the shared
    share of would-be sessions skipped that night (trading contract v1
    §10.1b): a clocked EV plugs in when its uniform is below
    ``p × (1 − q)``.  ``None`` means no skip, the behaviour before §10.  An
    EV still plugged in from a non-driving day is not skipped: it stays in.

    Returns three (world, day, EV) arrays, indexed by the session's plug-in
    date: a boolean accepted flag and the UTC session start and end instants
    as ``datetime64[ns]`` (naive UTC, ``NaT`` where not accepted).  The
    generator advances by exactly two full (world, day, EV) blocks,
    independent of the parameter values and the drive flags.
    """

    _check_connection_clocks(units)
    clip = _validate_scalar(clip_minutes, "clock_clip_minutes", multiple_of=30.0)
    df = _validate_t_df(t_df)
    weekday_scale = _scale_by_unit(units, plug_in_scale_minutes_by_cohort, "plug-in scale")
    weekend_scale = _scale_by_unit(
        units,
        weekend_plug_in_scale_minutes_by_cohort or plug_in_scale_minutes_by_cohort,
        "weekend plug-in scale",
    )

    # The explicit horizon is a fixed 48 * (warm-up + study) UTC slots from
    # London noon (decision 0004 items 1 and 52); a London offset change inside
    # it is allowed.  Day d is the London calendar date first_local_date + d,
    # and its local clock times use that date's offset.  The last sampled date
    # is the morning after the last night (``sampled_day_count``); its evening
    # session normally starts after the horizon end, where the kernel ignores it.
    boundaries = utc_half_hour_boundaries(
        settings.start_local_date, settings.warmup_days, settings.study_days
    )
    days = settings.sampled_day_count
    shape = (settings.evaluation_world_count, days, settings.vehicle_count)
    departures = np.asarray(departure_utc)
    if departures.dtype.kind != "M" or departures.shape != (shape[0], days + 1, shape[2]):
        raise ValueError(
            f"departure_utc must be datetime64 with shape ({shape[0]}, {days + 1}, {shape[2]})"
        )
    drives = np.asarray(drives_today)
    if drives.dtype != bool or drives.shape != shape:
        raise ValueError(f"drives_today must be boolean with shape {shape}")
    if skip_share is None:
        skip = np.zeros(shape[:2])
    else:
        skip = np.asarray(skip_share, dtype=np.float64)
        # Written as "not inside" so NaN fails too.
        if skip.shape != shape[:2] or not ((skip >= 0.0) & (skip <= 1.0)).all():
            raise ValueError(f"skip_share must be fractions in 0-1 with shape {shape[:2]}")
    accepted = np.zeros(shape, dtype=bool)

    cohort_values = units["cohort_id"].tolist()
    always_mask = np.array([cohort == _ALWAYS_PLUGGED_COHORT for cohort in cohort_values])
    clocked = ~always_mask
    local_midnight, weekend = _local_midnights(settings, days + 1)
    # (date, EV) values; each date takes its own day type's window (decision
    # 0004 item 42), so Friday 18:00 -> Saturday 10:00 for a weekend-window cohort.
    weekday_arrival, weekend_arrival = connection_window_hours(units, "arrival")
    weekday_departure, weekend_departure = connection_window_hours(units, "departure")
    arrival_minutes = 60.0 * np.where(weekend, weekend_arrival, weekday_arrival)
    departure_minutes = 60.0 * np.where(weekend, weekend_departure, weekday_departure)
    scale = np.where(weekend, weekend_scale, weekday_scale)

    # Sessions must never overlap or touch, whatever the draws.  A departure
    # can move by up to ±clip (its scale is not known here, so the check
    # assumes it moves); a plug-in moves by up to ±clip only when its scale is
    # positive.  So on each date the latest departure must come before the
    # earliest plug-in, and each plug-in before the next date's earliest
    # departure.  A deterministic check on the clock hours, rather than on the
    # draws, makes a bad setting fail every time, not in the odd world.
    #
    # Caveat: the check works in London wall-clock minutes.  On the spring
    # clock-change night the wall clock skips 01:00-02:00, so the real (UTC)
    # gap is an hour shorter than the wall-clock gap for a session that spans
    # it.  The defaults never come near (the earliest departure is 04:00 and
    # the latest plug-in 01:00, a scheduled 22:00 plus 3 h, still hours
    # apart), but a hand-built caller with a very tight clock could pass here
    # and then be refused by the kernel's overlap check
    # (physics._check_event_order), which works on UTC instants.
    plug_in_reach = np.where(scale > 0.0, clip, 0.0)
    after_departure = (arrival_minutes - plug_in_reach) - (departure_minutes + clip)
    before_next = (24 * 60 + departure_minutes[1:] - clip) - (
        arrival_minutes[:-1] + plug_in_reach[:-1]
    )
    if clocked.any() and (
        (after_departure[:, clocked] <= 0.0).any() or (before_next[:, clocked] <= 0.0).any()
    ):
        raise ValueError(
            "clock_clip_minutes must leave a strictly positive gap between adjacent sessions"
        )

    # Common random numbers (plan M6): both channels draw a full (world, day, EV)
    # block whatever the cohorts, probabilities or scales are, so a parameter
    # edit never moves the generator position seen by later channels
    # (Compare's "matched futures").  Always-plugged units ignore their draws.
    #
    # Plug acceptance is Bernoulli(p) by inverse transform, accepted when U < p.
    # stats.bernoulli.rvs was rejected here: it calls Generator.binomial, which
    # consumes no draw when p is exactly 0, so a probability edit would shift
    # every later draw.  One uniform per cell keeps the count fixed.
    #
    # The shared skip share scales every clocked EV's probability that night
    # (§10.1b): p_eff = p × (1 − q), the same relative cut for a p = 1 cohort
    # and a p = 0.2 one, compared with the same uniform, so a skip edit only
    # turns sessions off or on and never re-draws them.  The always-plugged
    # cohort is excluded: its one session is the whole horizon.
    probabilities = units["plug_probability"].to_numpy(dtype=float, copy=True)
    effective = probabilities[np.newaxis, np.newaxis, :] * (1.0 - skip[:, :, np.newaxis])
    plug_uniforms = stats.uniform.rvs(size=shape, random_state=rng)
    accepted[:, :, clocked] = (plug_uniforms < effective)[:, :, clocked]
    # The plug-in time is its own draw, independent of the departure (item
    # 51): the old model shared one normal draw between a session's two ends,
    # which made a late plug-in always mean a late plug-out.
    plug_in_uniforms = stats.uniform.rvs(size=shape, random_state=rng)
    plug_in_shifts = truncated_t_shift_minutes(plug_in_uniforms, scale[:-1], df, clip)

    starts = london_wall_time_to_utc(
        local_midnight[np.newaxis, :-1, np.newaxis],
        arrival_minutes[np.newaxis, :-1] + plug_in_shifts,
    )
    ends = np.full(shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    # Multi-day sessions (decision 0004 item 68, amending item 51): an EV
    # unplugs only on a morning it drives, at that date's departure draw; on
    # a non-driving day it stays plugged in, so its session runs on until
    # the next driving day's departure.  The rejected alternative (unplug at
    # every departure draw, the item 51 rule) left low-mileage drivers at
    # home unplugged all day and plugging in again every evening, far from
    # CNZ's ~30 % of plug-ins two or more days after the last.  While such a
    # session is open the evening plug-in is not a new event: its plug
    # uniform is left unused, and the skip share has nothing to act on
    # (a driver does not unplug to "skip" a night).  Everything here is a
    # deterministic function of draws already made (drives, plug uniforms,
    # both clocks), so no draw is added or moved (common random numbers).
    # The walk is over dates because each morning's state depends on the
    # evening before; it is vectorised over worlds and EVs.
    world_index, ev_index = np.indices((shape[0], shape[2]))
    open_night = np.full((shape[0], shape[2]), -1, dtype=np.int64)
    for day in range(days):
        leaves = (open_night >= 0) & drives[:, day, :]
        ends[world_index[leaves], open_night[leaves], ev_index[leaves]] = departures[:, day, :][
            leaves
        ]
        open_night[leaves] = -1
        accepted[:, day, :] &= open_night < 0
        open_night[accepted[:, day, :]] = day
    # A session still open after the last sampled date ends at the extra
    # last departure, the morning after it, which is past the horizon end
    # (it is only a closing instant for the kernel, never a trip).
    still_open = open_night >= 0
    ends[world_index[still_open], open_night[still_open], ev_index[still_open]] = departures[
        :, days, :
    ][still_open]
    # Always-plugged EVs have one session covering the whole horizon, stored
    # on day 0 (no clock, no draw used); their departures only start trips.
    accepted[:, 0, always_mask] = True
    starts[:, 0, always_mask] = pd.Timestamp(boundaries[0]).tz_convert(None).to_datetime64()
    ends[:, 0, always_mask] = pd.Timestamp(boundaries[-1]).tz_convert(None).to_datetime64()
    starts[~accepted] = np.datetime64("NaT")
    ends[~accepted] = np.datetime64("NaT")
    return accepted, starts, ends


# ============================================================================
# Weather: daily temperature per world
# ============================================================================
#
# One draw: one daily Celsius temperature per world and day around the
# synthetic scenario's base temperatures, from the run's single
# ``np.random.Generator`` (decision 0004 item 14).  The base temperatures and
# SD are records in ``model/assumptions.py`` (``WEATHER``);
# ``physics.effective_driving_efficiency`` turns the temperatures into
# driving efficiency.


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be a finite number")
    return numeric_value


def sample_daily_temperature(
    rng: np.random.Generator,
    *,
    world_count: int,
    base_temperature_c_by_day: np.ndarray,
    sd_c: float,
) -> np.ndarray:
    """Sample world-by-day Celsius temperatures from one supplied generator.

    Each day's temperature is Normal(base for that day, ``sd_c``) in °C, drawn
    independently per world and day.  Returns a float64 (world, day) array.
    The generator always advances by exactly ``world_count × days`` draws.
    """
    # The base temperatures and SD are caller inputs, so their values are
    # checked; the world count comes from the run settings.
    base = np.asarray(base_temperature_c_by_day, dtype=np.float64)
    if base.size == 0 or not np.all(np.isfinite(base)):
        raise ValueError("base_temperature_c_by_day must be a non-empty finite array")

    sd = _finite_number(sd_c, name="sd_c")
    if sd < 0.0:
        raise ValueError("sd_c must be non-negative")
    # Temperature = base + SD × a standard-normal draw.  Passing SD as `scale`
    # (or returning early for SD 0) was rejected: both skip the draws when the
    # SD is zero, so a weather-SD edit would shift every later channel and break
    # common random numbers across a Compare (plan M6).
    standard_offsets = stats.norm.rvs(size=(world_count, base.size), random_state=rng)
    return np.asarray(base + sd * standard_offsets, dtype=np.float64)


# ============================================================================
# Prices: a synthetic system, its day-ahead price, intraday and imbalance
# ============================================================================
#
# Decision 0004 items 53, 55 and 56 (plan §2B B1, B5 and §7 H1).  Per world
# and UTC half-hour:
#
#   net demand (GW) = demand shape + heating − wind − solar + known shocks
#   day-ahead price = expected intraday close given day-ahead information
#                     (f(net demand) + daily level + half-hour noise + the
#                     expected impact of surprises) − a deliberate premium
#   intraday close  = day-ahead price + hourly martingale updates to gate closure
#                     + price impact of surprise shocks
#   imbalance (SIP) = intraday close + NIV premium (long or short) + fat tail
#
# Every price is bounded by the GB day-ahead auction limits (PRICES
# price_cap / price_floor).
#
# Why a net-demand model rather than a price-only Fourier shape: the same
# cold still evening then raises price more when the system is already
# tight (the supply curve is convex), which is what the events work (item
# 55) needs, and the temperatures that already change driving efficiency
# also move price, so the two stay paired inside a world.  The demand shape
# is a 3-harmonic Fourier series in London clock hour, one set per
# (winter/summer × weekday/weekend); wind is one capacity factor per day,
# persistent across days (weather systems last several days), so whole
# days move together by £20–30 as they do in GB.  Nothing here is market
# data: every coefficient is a record in ``assumptions`` (``SYSTEM``,
# ``SHOCKS``, ``PRICES``, ``TRADING``), some calibrated to the Elexon fit of
# GB prices (plan B2), the rest illustrative; the records say which.
#
# Shocks (decision 0004 item 56) are temporary GW changes to net demand in
# the evening-to-morning window, mild and frequent or big and rare, up or
# down.  They act on net demand before the supply curve, so the same shock
# moves price more when the system is tight.  A known shock is in the
# day-ahead price; a surprise moves only intraday and imbalance.
#
# Roles of the price channels (decision 0004 item 53):
# - day-ahead: what the smart charger plans on, and the price home import
#   is valued at (the tariff leg, plan B3);
# - intraday close: the day-ahead price plus the news that arrives before
#   gate closure.  It is the realised price (``evaluation_prices``) and
#   replaces the earlier independent AR(1) forecast error: the difference
#   between it and the day-ahead price is now that forecast error;
# - imbalance (SIP): what an unhedged deviation settles at, for the trading
#   ledger (plan F1).  The intraday and imbalance prices are not yet used by
#   any view.
#
# Common random numbers (plan M6): every channel below takes a fixed number
# of draws whatever its parameter values, drawn in this order after the
# worlds: day-ahead half-hour noise, daily level, daily wind, intraday
# shocks, NIV state uniforms, imbalance-tail uniforms, the mild and big
# net-demand shock candidates (decision 0004 item 56), then the daily solar
# clearness (item 58).  So editing one price
# assumption changes only its own channel, and no world draw moves.


_LONDON = "Europe/London"
_HALF_HOUR = pd.Timedelta(minutes=30)
# AR(1) coefficient of the half-hourly day-ahead noise, calibrated in the
# assumptions module (PRICES['ar1_phi']).  Real prices persist across
# neighbouring half-hours, so independent noise would make a cheap slot
# beside an expensive one far too common.
ILLUSTRATIVE_PRICE_NOISE_AR1_COEFFICIENT = PRICES["ar1_phi"].value
# Warm-up samples discarded so every AR(1) path starts from its stationary
# distribution: with φ ≤ 0.92 the zero start's influence decays below
# 0.92**100 ≈ 2e-4 of one SD.  A technical constant, not a model assumption.
_AR1_BURNIN = 100
# Intraday updates drawn per half-hour, whatever the lead time: day-ahead
# publication at 13:00 London the day before to the default 60-minute gate
# closure is at most 34 whole hours (35 on a 25-hour day), so 36 covers
# every period; longer trading windows (an edited timing) are capped at 36
# updates, so the draw count stays fixed.  Updates
# beyond a period's own count are drawn and not used.
_INTRADAY_UPDATE_DRAWS = 36
# Candidate shocks drawn per world and night (decision 0004 item 56): each
# happens with probability rate ÷ candidates, so the draw count stays fixed
# while a rate is edited.  Six mild candidates allow up to six mild shocks
# a night and three big ones allow up to 21 a week, well above the default
# rates; a rate is capped at its candidate count.
_MILD_SHOCK_CANDIDATES = 6
_BIG_SHOCK_CANDIDATES = 3

# The names the ``price_assumptions`` mapping of ``generate_market_price_paths``
# must hold; the runner checks a mapping against them so no value falls back
# to an invented default.
MARKET_PRICE_INPUTS = (
    "demand_shape_gw.winter_weekday",
    "demand_shape_gw.winter_weekend",
    "demand_shape_gw.summer_weekday",
    "demand_shape_gw.summer_weekend",
    "summer_months",
    "heating_threshold_c",
    "heating_gw_per_c",
    "wind_installed_gw",
    "wind_median_capacity_factor.winter",
    "wind_median_capacity_factor.summer",
    "wind_logit_sd",
    "wind_ar1_phi",
    "supply_reference_net_demand_gw",
    "supply_reference_price_gbp_per_mwh",
    "supply_slope_gbp_per_mwh_per_gw",
    "scarcity_threshold_gw",
    "scarcity_width_gw",
    "scarcity_price_gbp_per_mwh",
    "daily_level_sd_gbp_per_mwh",
    "daily_level_ar1_phi",
    "day_ahead_sd_gbp_per_mwh",
    "day_ahead_publication_local_hour",
    "gate_closure_minutes",
    "intraday_hourly_sd_gbp_per_mwh",
    "intraday_shock_ar1_phi",
    "imbalance_long_share",
    "imbalance_state_persistence",
    "imbalance_long_premium_gbp_per_mwh",
    "imbalance_short_premium_gbp_per_mwh",
    "imbalance_tail_scale_gbp_per_mwh",
    "imbalance_tail_df",
    "shock_window_start_local_hour",
    "shock_window_end_local_hour",
    "mild_shock_rate_per_night",
    "big_shock_rate_per_week",
    "mild_shock_scale_gw",
    "big_shock_median_gw",
    "big_shock_log_sd",
    "shock_up_probability",
    "mild_shock_min_hours",
    "mild_shock_max_hours",
    "big_shock_min_hours",
    "big_shock_max_hours",
    "mild_shock_known_share",
    "big_shock_known_share",
    "shock_max_gw",
    "price_cap_gbp_per_mwh",
    "price_floor_gbp_per_mwh",
    "da_id_premium_gbp_per_mwh",
    "surprise_reveal_hours",
    "niv_surprise_shift",
    "solar_installed_gw",
    "solar_clear_sky_peak_cf.winter",
    "solar_clear_sky_peak_cf.summer",
    "solar_sunrise_local_hour.winter",
    "solar_sunrise_local_hour.summer",
    "solar_sunset_local_hour.winter",
    "solar_sunset_local_hour.summer",
    "solar_clearness_median",
    "solar_clearness_logit_sd",
    "solar_clearness_ar1_phi",
)


def demand_shape_gw(coefficients: tuple[float, ...], local_hour: np.ndarray) -> np.ndarray:
    """Evaluate a 3-harmonic daily demand shape (GW) at London clock hours (0–24).

    ``coefficients`` is ``(a0, a1, b1, a2, b2, a3, b3)``:
    ``a0 + Σ_k a_k cos(2πk·h/24) + b_k sin(2πk·h/24)`` for k = 1..3.  Three
    harmonics are enough for an overnight trough, a morning shoulder, a
    midday (solar) dip and an evening peak; this is the form a
    least-squares fit to half-hourly data returns (plan B2), so fitted
    coefficients paste straight into the records.
    """

    angle = 2 * np.pi * np.asarray(local_hour, dtype=float) / 24.0
    shape = np.full(angle.shape, float(coefficients[0]))
    for k in (1, 2, 3):
        shape += coefficients[2 * k - 1] * np.cos(k * angle) + coefficients[2 * k] * np.sin(
            k * angle
        )
    return shape


def supply_curve_gbp_per_mwh(
    net_demand_gw: np.ndarray,
    *,
    supply_reference_net_demand_gw: float,
    supply_reference_price_gbp_per_mwh: float,
    supply_slope_gbp_per_mwh_per_gw: float,
    scarcity_threshold_gw: float,
    scarcity_width_gw: float,
    scarcity_price_gbp_per_mwh: float,
) -> np.ndarray:
    """Day-ahead price (£/MWh) the illustrative merit order clears at a net demand (GW).

    ``f(N) = p_ref + s·(N − N_ref) + A·exp((N − N_scarcity) / w)``.

    The linear part is the broad gas-set middle of the merit order: each GW
    of net demand calls a slightly dearer plant.  It keeps falling below
    zero when net demand is very low, which stands in for inflexible and
    subsidised generation paying to keep running (the source of GB's
    negative half-hours).  The exponential part is scarcity: it is almost
    nothing below the threshold and steepens sharply above it.  With
    ``s ≥ 0`` and ``A ≥ 0`` the curve is monotone and convex, so a shock of
    the same size moves price more when the system is tight (decision 0004
    item 55).  A piecewise merit order with a steep negative "cliff" was
    rejected: it is not convex and adds breakpoints nobody can calibrate yet.
    """

    net = np.asarray(net_demand_gw, dtype=float)
    return (
        supply_reference_price_gbp_per_mwh
        + supply_slope_gbp_per_mwh_per_gw * (net - supply_reference_net_demand_gw)
        + scarcity_price_gbp_per_mwh * np.exp((net - scarcity_threshold_gw) / scarcity_width_gw)
    )


@dataclass(frozen=True)
class MarketPrices:
    """Everything ``generate_market_price_paths`` returns for one run.

    - ``day_ahead``: long frame, one row per world and slot (world-major):
      ``wholesale_forecast_gbp_per_mwh`` (£/MWh, the price the charger ranks
      and home import is valued at) and ``system_net_demand_gw`` (GW, the
      synthetic net demand behind it, known shocks included);
    - ``realised``: same rows: ``evaluation_context_price_gbp_per_mwh``
      (£/MWh, the intraday close at gate closure),
      ``imbalance_price_gbp_per_mwh`` (£/MWh, the system imbalance price) and
      ``system_long`` (bool, the NIV state that set its premium);
    - ``shocks``: one row per stochastic net-demand shock that starts inside
      the priced span (see ``sample_net_demand_shocks``), by world and start;
    - ``known_shock_gw`` and ``surprise_shock_gw``: (world, slot) GW of all
      known and all surprise shocks, stochastic plus any scripted ones;
    - ``intraday_path_gbp_per_mwh``: (world, slot, h) intraday price h whole
      hours before gate closure, h = 0 the close, for every world (about
      20 MB at 100 worlds over a 7-day warm-up and the study; the trading
      overlay stores only a sample).
    """

    day_ahead: pd.DataFrame
    realised: pd.DataFrame
    shocks: pd.DataFrame
    known_shock_gw: np.ndarray
    surprise_shock_gw: np.ndarray
    intraday_path_gbp_per_mwh: np.ndarray


def day_ahead_publication_utc_ns(interval_start_utc: pd.DatetimeIndex) -> np.ndarray:
    """When each slot's day-ahead price is published, as int64 UTC nanoseconds.

    ``interval_start_utc`` are timezone-aware slot starts.  A slot's delivery
    day is the London calendar date of its start; every price of delivery
    day D is published at ``PRICES["day_ahead_publication_local_hour"]``
    (13:00) London on D - 1, the same record the intraday updates start
    from (plan B4, decision 0004 item 53), converted at that date's own
    London offset.  It lives beside the price generator,
    which stamps it on every day-ahead row; ``action.visible_day_ahead_prices``
    applies it to what a plan may rank.
    """

    delivery_date = (
        pd.DatetimeIndex(interval_start_utc)
        .tz_convert(_LONDON)
        .tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype("datetime64[D]")
    )
    published = london_wall_time_to_utc(
        delivery_date - np.timedelta64(1, "D"),
        60.0 * PRICES["day_ahead_publication_local_hour"].value,
    )
    return published.astype(np.int64)


def generate_market_price_paths(
    rng: np.random.Generator,
    *,
    interval_start_utc: pd.DatetimeIndex,
    evaluation_world_count: int,
    daily_temperature_c: np.ndarray,
    price_assumptions: Mapping[str, object],
    scripted_known_shock_gw: np.ndarray | None = None,
    scripted_surprise_shock_gw: np.ndarray | None = None,
    scripted_surprise_reveal_hours: np.ndarray | float = 0.0,
) -> MarketPrices:
    """Generate each world's synthetic day-ahead, intraday and imbalance prices.

    ``interval_start_utc`` are the priced UTC half-hours.
    ``daily_temperature_c`` is the (world, day) array of the run's sampled
    temperatures (°C), column 0 being the London date of the first slot; it
    drives the heating term, so a cold world is also a dear one.
    ``price_assumptions`` maps every name in ``MARKET_PRICE_INPUTS`` to its
    value (the ``SYSTEM``, ``SHOCKS``, ``PRICES`` and ``TRADING`` records).

    ``scripted_known_shock_gw`` and ``scripted_surprise_shock_gw`` are
    optional scripted event profiles (GW, broadcastable to (world, slot);
    trading contract §2.5): added to the stochastic known and surprise
    shocks before the supply curve, and drawing nothing, so the random
    futures are the same with and without them.  A scripted surprise is
    revealed ``scripted_surprise_reveal_hours`` (whole hours, broadcastable
    to (world, slot)) before each period's gate closure; a negative value
    means after gate closure, so it moves only the imbalance price.  Unlike
    the stochastic surprises, a scripted one is not priced into day-ahead:
    it is the event under study.

    Returns a ``MarketPrices``.
    """

    values = _checked_market_inputs(price_assumptions)
    starts = pd.DatetimeIndex(interval_start_utc)
    slot_count = len(starts)
    worlds = int(evaluation_world_count)
    scripted = [
        np.zeros((worlds, slot_count))
        if profile is None
        else np.broadcast_to(np.asarray(profile, dtype=float), (worlds, slot_count))
        for profile in (scripted_known_shock_gw, scripted_surprise_shock_gw)
    ]
    if not all(np.isfinite(profile).all() for profile in scripted):
        raise ValueError("scripted shock profiles must be finite")
    scripted_reveal = np.broadcast_to(
        np.floor(np.asarray(scripted_surprise_reveal_hours, dtype=float)), (worlds, slot_count)
    )
    temperature = np.asarray(daily_temperature_c, dtype=float)
    local = starts.tz_convert(_LONDON)
    local_hour = np.asarray(local.hour + local.minute / 60.0, dtype=float)
    local_midnight = local.tz_localize(None).normalize()
    day_index = np.asarray((local_midnight - local_midnight[0]).days, dtype=np.int64)
    # A study that ends after a spring-forward change runs one hour into the
    # next London date; that hour reuses the last sampled day's weather.
    day_index = np.minimum(day_index, temperature.shape[1] - 1)
    day_count = temperature.shape[1]

    # Demand shape: one season for the whole run, set by the first slot's
    # London month (plan D7), so a week straddling 1 April does not jump.
    season = "summer" if local[0].month in values["summer_months"] else "winter"
    weekend = np.asarray(local.dayofweek >= 5)
    demand = np.where(
        weekend,
        demand_shape_gw(values[f"demand_shape_gw.{season}_weekend"], local_hour),
        demand_shape_gw(values[f"demand_shape_gw.{season}_weekday"], local_hour),
    )

    # Draws, in a fixed order and number (see the section comment).
    noise = _ar1_noise(
        rng,
        (worlds, slot_count),
        values["day_ahead_sd_gbp_per_mwh"],
        ILLUSTRATIVE_PRICE_NOISE_AR1_COEFFICIENT,
    )
    level = _ar1_noise(
        rng,
        (worlds, day_count),
        values["daily_level_sd_gbp_per_mwh"],
        values["daily_level_ar1_phi"],
    )
    wind_latent = _ar1_noise(
        rng, (worlds, day_count), values["wind_logit_sd"], values["wind_ar1_phi"]
    )
    intraday_shocks = _ar1_noise(
        rng,
        (worlds, _INTRADAY_UPDATE_DRAWS, slot_count),
        1.0,
        values["intraday_shock_ar1_phi"],
    )
    state_uniforms = rng.random((worlds, slot_count))
    tail_uniforms = rng.random((worlds, slot_count))
    shocks = sample_net_demand_shocks(
        rng, starts=starts, local_midnight=local_midnight, world_count=worlds, values=values
    )
    # Decision 0004 item 58: solar, appended after every earlier channel so
    # adding it moved no existing draw.
    clearness_latent = _ar1_noise(
        rng,
        (worlds, day_count),
        values["solar_clearness_logit_sd"],
        values["solar_clearness_ar1_phi"],
    )

    # Heating: each °C a day is below the threshold adds demand for the
    # whole day.  Most GB homes heat with gas, so the electric effect is a
    # few GW, not the whole heating load.
    heating = values["heating_gw_per_c"] * np.maximum(
        values["heating_threshold_c"] - temperature, 0.0
    )
    # Wind: a daily capacity factor on the logit scale, so it stays inside
    # (0, 1) however windy; the latent AR(1) across days makes calm and
    # windy spells last several days.
    wind = values["wind_installed_gw"] * expit(
        logit(values[f"wind_median_capacity_factor.{season}"]) + wind_latent
    )
    # Solar (item 58): installed GW × the clear-sky daylight shape of the
    # season × a daily clearness (logit-normal AR(1) across days, like wind,
    # so sunny and dull spells last).  Zero at night by construction.
    clearness = expit(logit(values["solar_clearness_median"]) + clearness_latent)
    solar = (
        values["solar_installed_gw"]
        * solar_daylight_shape(local_hour, values, season)[None, :]
        * clearness[:, day_index]
    )
    known_shock_gw, surprise_path_gw, surprise_late_gw = _shock_profiles(
        shocks, worlds, slot_count, values
    )
    known_shock_gw = known_shock_gw + scripted[0]
    # A scripted surprise joins every intraday step at or after its reveal
    # (h ≤ reveal hours before gate closure), or only imbalance if negative.
    steps = np.arange(_INTRADAY_UPDATE_DRAWS)
    surprise_path_gw = surprise_path_gw + scripted[1][:, :, None] * (
        steps[None, None, :] <= scripted_reveal[:, :, None]
    )
    surprise_late_gw = surprise_late_gw + np.where(scripted_reveal < 0, scripted[1], 0.0)
    surprise_shock_gw = surprise_path_gw[:, :, 0] + surprise_late_gw
    # Known shocks (a cold still evening forecast the day before) are in the
    # net demand the day-ahead auction clears on; surprises arrive later.
    net_demand = (
        demand[None, :] + heating[:, day_index] - wind[:, day_index] - solar + known_shock_gw
    )

    curve_values = {name: values[name] for name in SUPPLY_CURVE_INPUTS}
    curve = supply_curve_gbp_per_mwh(net_demand, **curve_values)
    floor, cap = values["price_floor_gbp_per_mwh"], values["price_cap_gbp_per_mwh"]
    # The daily level stands for what the net-demand model leaves out (gas
    # and carbon prices, interconnector flows, outages): it moves every
    # half-hour of a day together.  Negative prices come from the model
    # itself; only the auction limits bound it.
    base = curve + level[:, day_index] + noise

    # No free money (Mike, review of W1-prices): the day-ahead price is the
    # expected intraday close given day-ahead information, less a deliberate
    # premium.  The convex curve makes zero-mean surprises raise the mean
    # price (E[f(N + S)] > f(N)), so without this the intraday close would
    # drift above day-ahead and a trader would earn by buying day-ahead.
    # The expectation is exact over the surprise distribution (enumerated
    # and convolved on a 0.1 GW grid, no random draw), capped as the close
    # is; it ignores the cap acting on the zero-mean intraday updates.
    expected_close, revealed_share = _expected_surprise_close(
        base, net_demand, curve_values, starts, local_midnight, values
    )
    day_ahead = np.clip(expected_close - values["da_id_premium_gbp_per_mwh"], floor, cap)
    _validate_generated_prices(day_ahead, "day-ahead")

    # Intraday path (trading contract §1.4, §2.5): at step h (h whole hours
    # before gate closure) the price has every hourly update made by then,
    # the supply-curve impact of the surprises revealed by then, and the
    # expected impact of those still to come (the part of the day-ahead
    # expectation not yet revealed).  At h = 0 every surprise revealed by
    # gate closure is in, so the path ends at the close.
    updates = _intraday_updates(starts, local_midnight, intraday_shocks, values)
    updates_by_step = np.moveaxis(np.cumsum(updates[:, ::-1, :], axis=1)[:, ::-1, :], 1, 2)
    revealed_impact = (
        supply_curve_gbp_per_mwh(net_demand[:, :, None] + surprise_path_gw, **curve_values)
        - curve[:, :, None]
    )
    still_expected = (expected_close - base)[:, :, None] * (1.0 - revealed_share[None, :, :])
    intraday_path = np.clip(
        base[:, :, None] + updates_by_step + revealed_impact + still_expected, floor, cap
    )
    intraday_close = intraday_path[:, :, 0]

    # Imbalance: surprises revealed only after gate closure land here, then
    # the NIV premium and a fat tail.  A surprise shifts the chance the
    # system is short in its own direction (an unforeseen demand rise leaves
    # the system short).
    late_impact = supply_curve_gbp_per_mwh(
        net_demand + surprise_shock_gw, **curve_values
    ) - supply_curve_gbp_per_mwh(net_demand + surprise_path_gw[:, :, 0], **curve_values)
    system_long = _niv_states(
        state_uniforms,
        values["imbalance_long_share"],
        values["imbalance_state_persistence"],
        long_shift=-values["niv_surprise_shift"] * np.sign(surprise_shock_gw),
    )
    # Fat tails: Student-t through its quantile function on uniforms, not
    # ``stats.t.rvs``, so an edit to the degrees of freedom keeps the draws.
    tail = values["imbalance_tail_scale_gbp_per_mwh"] * stats.t.ppf(
        tail_uniforms, values["imbalance_tail_df"]
    )
    premium = np.where(
        system_long,
        values["imbalance_long_premium_gbp_per_mwh"],
        values["imbalance_short_premium_gbp_per_mwh"],
    )
    imbalance = np.clip(intraday_close + late_impact + premium + tail, floor, cap)
    _validate_generated_prices(intraday_path, "intraday")
    _validate_generated_prices(imbalance, "imbalance")

    world_id = np.repeat(np.arange(worlds), slot_count)
    world_starts = np.tile(starts, worlds)
    world_ends = np.tile(starts + _HALF_HOUR, worlds)
    forecast = pd.DataFrame(
        {
            "world_id": world_id,
            "interval_start_utc": world_starts,
            "interval_end_utc": world_ends,
            # Plan B4: each day's prices are published at 13:00 London the
            # day before delivery; a plan ranks only what is published by
            # then (action.visible_day_ahead_prices).
            "forecast_available_at_utc": pd.to_datetime(
                np.tile(day_ahead_publication_utc_ns(starts), worlds), utc=True
            ),
            "wholesale_forecast_gbp_per_mwh": day_ahead.reshape(-1),
            "system_net_demand_gw": net_demand.reshape(-1),
            "evidence_kind": "synthetic",
        }
    )
    evaluation = pd.DataFrame(
        {
            "world_id": world_id,
            "interval_start_utc": world_starts,
            "interval_end_utc": world_ends,
            "evaluation_context_price_gbp_per_mwh": intraday_close.reshape(-1),
            "imbalance_price_gbp_per_mwh": imbalance.reshape(-1),
            "system_long": system_long.reshape(-1),
            "outcome_available_at_utc": world_ends,
            "evidence_kind": "synthetic",
        }
    )
    shock_table = pd.DataFrame(
        {
            "world_id": shocks["world_id"],
            "shock_class": shocks["shock_class"],
            "start_slot_index": shocks["start_slot"],
            "start_utc": starts[shocks["start_slot"]],
            "duration_slots": shocks["duration_slots"],
            "size_gw": shocks["size_gw"],
            "direction": np.where(shocks["sign"] > 0, "up", "down"),
            "known_day_ahead": shocks["known"],
            "evidence_kind": "synthetic",
        }
    )
    return MarketPrices(
        day_ahead=forecast,
        realised=evaluation,
        shocks=shock_table,
        known_shock_gw=known_shock_gw,
        surprise_shock_gw=surprise_shock_gw,
        intraday_path_gbp_per_mwh=intraday_path,
    )


# Keyword names of ``supply_curve_gbp_per_mwh``; summaries read them from the
# price assumptions to recompute a shock's own price increment.
SUPPLY_CURVE_INPUTS = (
    "supply_reference_net_demand_gw",
    "supply_reference_price_gbp_per_mwh",
    "supply_slope_gbp_per_mwh_per_gw",
    "scarcity_threshold_gw",
    "scarcity_width_gw",
    "scarcity_price_gbp_per_mwh",
)


def solar_daylight_shape(
    local_hour: np.ndarray, values: Mapping[str, Any], season: str
) -> np.ndarray:
    """Clear-sky solar capacity factor (0–1) at London clock hours, for a season.

    A half sine between sunrise and sunset (by London clock, so BST is in
    the summer hours), peaking at the season's clear-sky peak capacity
    factor at the midpoint and zero outside daylight.  The slot's midpoint
    (+15 min) is used, so a half-hour straddling sunrise gets a little
    output.  Decision 0004 item 58: longer and higher days in summer are
    the records' sunrise, sunset and peak.
    """

    rise = values[f"solar_sunrise_local_hour.{season}"]
    set_ = values[f"solar_sunset_local_hour.{season}"]
    middle = np.asarray(local_hour, dtype=float) + 0.25
    phase = (middle - rise) / (set_ - rise)
    daylight = (phase > 0.0) & (phase < 1.0)
    return np.where(
        daylight, values[f"solar_clear_sky_peak_cf.{season}"] * np.sin(np.pi * phase), 0.0
    )


def _shock_nights(
    starts: pd.DatetimeIndex, local_midnight: pd.DatetimeIndex, values: Mapping[str, Any]
) -> tuple[np.ndarray, int]:
    """Slot index at which each night's shock window opens, and its length in slots.

    One night per London date in the span plus the night before the first
    date, so the first morning can have shocks too.  Each window opens at
    its London start hour; later starts are whole half-hours after it
    (absolute time, so on a clock-change night a start lands up to an hour
    off the London clock).
    """

    day_count = int((local_midnight[-1] - local_midnight[0]).days) + 2
    window_start = values["shock_window_start_local_hour"]
    window_slots = int(round(((values["shock_window_end_local_hour"] - window_start) % 24) * 2))
    night_open = (
        (local_midnight[0] + pd.to_timedelta(np.arange(-1, day_count - 1), unit="D"))
        + pd.Timedelta(hours=window_start)
    ).tz_localize(_LONDON)
    night_open_slot = np.asarray((night_open - starts[0]) / _HALF_HOUR, dtype=float).round()
    return night_open_slot.astype(np.int64), window_slots


def _shock_classes(values: Mapping[str, Any]) -> tuple[tuple[str, int, float], ...]:
    """(class, candidates per night, expected shocks per night) for mild and big."""

    return (
        ("mild", _MILD_SHOCK_CANDIDATES, values["mild_shock_rate_per_night"]),
        ("big", _BIG_SHOCK_CANDIDATES, values["big_shock_rate_per_week"] / 7.0),
    )


def _shock_size_cdf(shock_class: str, size_gw: np.ndarray, values: Mapping[str, Any]) -> np.ndarray:
    """CDF of a shock's size (GW), capped at ``shock_max_gw`` (the mass above lumps there)."""

    size = np.asarray(size_gw, dtype=float)
    if shock_class == "mild":
        cdf = stats.expon.cdf(size, scale=values["mild_shock_scale_gw"])
    else:
        cdf = stats.lognorm.cdf(
            size, s=values["big_shock_log_sd"], scale=values["big_shock_median_gw"]
        )
    return np.where(size >= values["shock_max_gw"], 1.0, cdf)


def sample_net_demand_shocks(
    rng: np.random.Generator,
    *,
    starts: pd.DatetimeIndex,
    local_midnight: pd.DatetimeIndex,
    world_count: int,
    values: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Sample each world's mild and big net-demand shocks (decision 0004 item 56).

    A shock is a temporary change in national net demand (GW): up is a
    scarcity event (a cold still evening, a plant trip), down a surplus
    (a windy, mild night).  Each may start only inside the nightly window
    (London 15:00 to 09:00 by default), the hours that matter for home
    charging.  Per world and night, each class has a fixed number of
    candidate shocks; a candidate happens when its uniform is below
    rate ÷ candidates, so the expected count is the rate and the draw count
    never depends on it (common random numbers).  Six uniforms per
    candidate: happens, start, size, sign, duration, notice.

    Sizes (GW): mild shocks are exponential with the mild scale; big ones
    lognormal around the big median (a Student-t was rejected: its tail
    reaches shocks bigger than the whole wind fleet).  Both are capped at
    ``shock_max_gw``, a credible largest single event.  Durations are whole
    half-hours, uniform between the class's minimum and maximum.  Each
    shock is either known day-ahead or a surprise, with the class's known
    share.  ``_expected_surprise_close`` enumerates exactly this
    distribution, so the two must change together.

    Returns flat arrays, one entry per shock that starts inside the priced
    span, ordered by world then start slot: ``world_id``, ``shock_class``
    ("mild"/"big"), ``start_slot``, ``duration_slots``, ``size_gw`` (> 0),
    ``sign`` (+1 up, −1 down) and ``known`` (bool).
    """

    night_open_slot, window_slots = _shock_nights(starts, local_midnight, values)
    night_count = len(night_open_slot)
    per_class = []
    for shock_class, candidates, rate in _shock_classes(values):
        happens, start, size, sign, duration, notice = rng.random(
            (6, world_count, night_count, candidates)
        )
        if shock_class == "mild":
            size_gw = values["mild_shock_scale_gw"] * -np.log1p(-size)
        else:
            size_gw = values["big_shock_median_gw"] * np.exp(
                values["big_shock_log_sd"] * stats.norm.ppf(size)
            )
        low = int(round(values[f"{shock_class}_shock_min_hours"] * 2))
        high = int(round(values[f"{shock_class}_shock_max_hours"] * 2))
        per_class.append(
            {
                "active": happens < rate / candidates,
                "world_id": np.broadcast_to(np.arange(world_count)[:, None, None], happens.shape),
                "shock_class": np.full(happens.shape, shock_class),
                "start_slot": (
                    night_open_slot[None, :, None] + np.floor(start * window_slots)
                ).astype(np.int64),
                "duration_slots": low + np.floor(duration * (high - low + 1)).astype(np.int64),
                "size_gw": np.minimum(size_gw, values["shock_max_gw"]),
                "sign": np.where(sign < values["shock_up_probability"], 1, -1),
                "known": notice < values[f"{shock_class}_shock_known_share"],
            }
        )
    fields = per_class[0].keys()
    kept = {
        name: np.concatenate([part[name][part["active"]] for part in per_class]) for name in fields
    }
    # A shock whose size rounds to nothing is not an event worth a row.
    in_span = (
        (kept["start_slot"] >= 0) & (kept["start_slot"] < len(starts)) & (kept["size_gw"] > 0.0)
    )
    kept = {name: array[in_span] for name, array in kept.items() if name != "active"}
    order = np.lexsort((kept["start_slot"], kept["world_id"]))
    return {name: array[order] for name, array in kept.items()}


def trapezoid_weight(k: np.ndarray, duration: np.ndarray) -> np.ndarray:
    """Weight of half-hour ``k`` of a ``duration``-slot shock (item 56 trapezoid).

    A shock ramps up over its first quarter, holds, and decays over its last
    quarter: min(1, (k + 1)/(r + 1), (D − k)/(r + 1)) with r = D // 4, so a
    2-hour shock is ½, 1, …, 1, ½ and a half-hour one is a single full step.
    """

    ramp = duration // 4 + 1
    return np.minimum(1.0, np.minimum((k + 1) / ramp, (duration - k) / ramp))


def _reveal_steps(k: np.ndarray, values: Mapping[str, Any]) -> np.ndarray:
    """Last intraday step h at which half-hour ``k`` of a surprise is already known.

    A stochastic surprise is revealed ``surprise_reveal_hours`` before it
    starts (trading contract §2.5, Q10).  Its half-hour k has gate closure
    ``gate_closure_minutes`` before it, so it is known from step
    h = floor(k/2 + reveal − gate) onwards (h whole hours before gate
    closure).  A negative value means revealed only after gate closure: it
    moves imbalance, not the intraday close.
    """

    lead = values["surprise_reveal_hours"] - values["gate_closure_minutes"] / 60.0
    return np.floor(0.5 * np.asarray(k, dtype=float) + lead + 1e-9).astype(np.int64)


def _shock_profiles(
    shocks: Mapping[str, np.ndarray],
    world_count: int,
    slot_count: int,
    values: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Known GW per (world, slot); surprise GW revealed by each intraday step; late surprise GW.

    Returns ``known`` (world, slot), ``surprise_path`` (world, slot, h): the
    surprise GW already known h whole hours before gate closure, and
    ``surprise_late`` (world, slot): surprise GW revealed only after gate
    closure.  Shocks overlap additively; one running past the span end is
    cut there.
    """

    known = np.zeros((world_count, slot_count))
    surprise_path = np.zeros((world_count, slot_count, _INTRADAY_UPDATE_DRAWS))
    surprise_late = np.zeros((world_count, slot_count))
    duration = shocks["duration_slots"]
    signed = shocks["sign"] * shocks["size_gw"]
    steps = np.arange(_INTRADAY_UPDATE_DRAWS)
    for k in range(int(duration.max(initial=0))):
        gw = signed * trapezoid_weight(np.full(duration.shape, k), duration)
        slot = shocks["start_slot"] + k
        on = (k < duration) & (slot < slot_count)
        chosen = on & shocks["known"]
        np.add.at(known, (shocks["world_id"][chosen], slot[chosen]), gw[chosen])
        chosen = on & ~shocks["known"]
        last_step = int(_reveal_steps(np.array(k), values))
        if last_step >= 0:
            np.add.at(
                surprise_path,
                (shocks["world_id"][chosen], slot[chosen]),
                gw[chosen, None] * (steps <= last_step)[None, :],
            )
        else:
            np.add.at(surprise_late, (shocks["world_id"][chosen], slot[chosen]), gw[chosen])
    return known, surprise_path, surprise_late


# Grid step (GW) on which the surprise distribution is convolved.  Fine
# enough that the expected price is within a few pence of the continuous
# value on the default curve (its scarcity term changes e-fold every 3 GW).
_SHOCK_GRID_GW = 0.1


def _expected_surprise_close(
    base: np.ndarray,
    net_demand: np.ndarray,
    curve_values: Mapping[str, float],
    starts: pd.DatetimeIndex,
    local_midnight: pd.DatetimeIndex,
    values: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Expected intraday close from day-ahead information, and the share revealed by each step.

    For each slot, the surprise shocks that could cover it are enumerated
    exactly as ``sample_net_demand_shocks`` draws them: every night, class,
    start offset and duration, with the class's occurrence, surprise and
    up-probabilities and its (capped) size distribution.  Candidates are
    independent, so the surprise total's distribution is the convolution of
    each candidate's, on a ``_SHOCK_GRID_GW`` grid.  Only surprises
    revealed by gate closure are counted: later ones move imbalance, not
    the close.  Then, per world,

        E[close] = Σ_s P(S = s) · clip(base + f(N + s) − f(N))

    with the auction floor and cap, where ``base`` is the no-surprise price.
    No random number is drawn.

    Returns the expected close (world, slot) and the share (slot, h) of
    each slot's surprise occurrence probability already revealed h whole
    hours before gate closure; the intraday path keeps the unrevealed part
    of the expected impact until its reveal.
    """

    slot_count = len(starts)
    night_open_slot, window_slots = _shock_nights(starts, local_midnight, values)
    floor, cap = values["price_floor_gbp_per_mwh"], values["price_cap_gbp_per_mwh"]
    up = values["shock_up_probability"]
    step = _SHOCK_GRID_GW
    half = int(math.ceil(values["shock_max_gw"] / step))
    edges = (np.arange(-half, half + 2) - 0.5) * step  # bin edges on the signed grid
    size_pmfs: dict[tuple[str, float], np.ndarray] = {}

    def signed_pmf(shock_class: str, weight: float) -> np.ndarray:
        # PMF of sign × weight × size on the grid −half..half.
        key = (shock_class, weight)
        if key not in size_pmfs:
            positive = np.diff(
                _shock_size_cdf(shock_class, np.maximum(edges, 0.0) / weight, values)
            )
            size_pmfs[key] = up * positive + (1.0 - up) * positive[::-1]
        return size_pmfs[key]

    expected = base.copy()
    revealed_share = np.ones((slot_count, _INTRADAY_UPDATE_DRAWS))
    for slot in range(slot_count):
        pmf = np.ones(1)
        revealed = np.zeros(_INTRADAY_UPDATE_DRAWS)
        total = 0.0
        for shock_class, candidates, rate in _shock_classes(values):
            low = int(round(values[f"{shock_class}_shock_min_hours"] * 2))
            high = int(round(values[f"{shock_class}_shock_max_hours"] * 2))
            durations = np.arange(low, high + 1)
            surprise = (rate / candidates) * (1.0 - values[f"{shock_class}_shock_known_share"])
            combo_probability = surprise / (window_slots * len(durations))
            for night_open in night_open_slot:
                offsets = np.arange(window_slots)
                begin = night_open + offsets
                k = slot - begin[:, None]
                duration = durations[None, :]
                # Candidates the sampler drops (start outside the span) or
                # that do not cover this slot contribute nothing.
                covers = (k >= 0) & (k < duration) & (begin[:, None] >= 0)
                covers &= begin[:, None] < slot_count
                last_step = np.broadcast_to(_reveal_steps(k, values), covers.shape)
                counted = covers & (last_step >= 0)
                if not counted.any():
                    continue
                weights = trapezoid_weight(k, duration)[counted]
                reveal = last_step[counted]
                candidate = np.zeros(2 * half + 1)
                candidate[half] = 1.0 - combo_probability * counted.sum()
                for weight in np.unique(weights):
                    share = combo_probability * np.count_nonzero(weights == weight)
                    candidate += share * signed_pmf(shock_class, float(weight))
                for _ in range(candidates):
                    pmf = np.convolve(pmf, candidate)
                    revealed += combo_probability * (
                        reveal[:, None] >= np.arange(_INTRADAY_UPDATE_DRAWS)[None, :]
                    ).sum(axis=0)
                    total += combo_probability * counted.sum()
        if total == 0.0:
            continue
        revealed_share[slot] = revealed / total
        centre = (len(pmf) - 1) // 2
        support = np.flatnonzero(pmf > 1e-12)
        shift = (support - centre) * step
        net = net_demand[:, slot, None]
        impact = supply_curve_gbp_per_mwh(net + shift[None, :], **curve_values)
        impact -= supply_curve_gbp_per_mwh(net, **curve_values)
        prices = np.clip(base[:, slot, None] + impact, floor, cap)
        expected[:, slot] = prices @ pmf[support] / pmf[support].sum()
    # Slots no surprise can reach keep the no-surprise price, capped.
    return np.clip(expected, floor, cap), revealed_share


def _intraday_updates(
    starts: pd.DatetimeIndex,
    local_midnight: pd.DatetimeIndex,
    shocks: np.ndarray,
    values: Mapping[str, Any],
) -> np.ndarray:
    """Each period's hourly intraday price updates, (world, lead h, slot) in £/MWh.

    A period's price is first known when the day-ahead auction publishes
    (13:00 London the day before delivery) and trades until gate closure
    (``gate_closure_minutes`` before delivery).  At each whole hour in
    between, news moves it by ``σ_hour × shock``.  The shocks have mean
    zero, so without surprises the intraday price is a martingale around
    the day-ahead price, and a period traded for longer ends further from
    it on average.  Update j of neighbouring periods happens at nearly the
    same clock time, so ``shocks`` (world, update, slot) is AR(1) along the
    slot axis: news about an evening moves all of its half-hours together.
    Update index j is the lead: it happens j whole hours before the
    period's gate closure, and is zero for leads before day-ahead
    publication.
    """

    publication_hour = values["day_ahead_publication_local_hour"]
    publication = (
        (local_midnight - pd.Timedelta(days=1) + pd.Timedelta(hours=publication_hour))
        .tz_localize(_LONDON)
        .tz_convert("UTC")
    )
    gate = starts - pd.Timedelta(minutes=values["gate_closure_minutes"])
    hours_open = np.asarray((gate - publication) / pd.Timedelta(hours=1), dtype=float)
    update_count = np.clip(np.floor(hours_open), 0, _INTRADAY_UPDATE_DRAWS).astype(np.int64)
    used = np.arange(_INTRADAY_UPDATE_DRAWS)[:, None] < update_count[None, :]
    return values["intraday_hourly_sd_gbp_per_mwh"] * shocks * used[None, :, :]


def _niv_states(
    uniforms: np.ndarray, long_share: float, persistence: float, long_shift: np.ndarray
) -> np.ndarray:
    """Two-state Markov chain of the system's net imbalance (True = long), (world, slot).

    A long system (more generation than demand) settles below the market
    price, a short one above it.  Without surprises the chain is stationary
    with P(long) = ``long_share``, and ``persistence`` is the lag-one
    autocorrelation of the state: P(long → long) = 1 − (1 − ρ)(1 − π) and
    P(short → long) = (1 − ρ)π.  ``long_shift`` (world, slot) moves both
    probabilities in a half-hour a surprise covers: an unforeseen demand
    rise makes the system short more often (lead's answer to the trading
    contract's NIV question, review of W1-prices).  Otherwise NIV is
    independent of prices, a stated limitation.
    The first slot is drawn from the stationary share.  One uniform per
    slot, so the draw count is fixed.
    """

    stay_long = 1.0 - (1.0 - persistence) * (1.0 - long_share)
    turn_long = (1.0 - persistence) * long_share
    states = np.empty(uniforms.shape, dtype=bool)
    states[:, 0] = uniforms[:, 0] < np.clip(long_share + long_shift[:, 0], 0.0, 1.0)
    for slot in range(1, uniforms.shape[1]):
        threshold = np.where(states[:, slot - 1], stay_long, turn_long) + long_shift[:, slot]
        states[:, slot] = uniforms[:, slot] < np.clip(threshold, 0.0, 1.0)
    return states


def _ar1_noise(
    rng: np.random.Generator, shape: tuple[int, ...], sd: float, phi: float
) -> np.ndarray:
    """Stationary AR(1) noise with marginal SD ``sd`` along the last axis of ``shape``.

    e_t = φ e_(t-1) + u_t.  For a stationary AR(1), Var(e) = Var(u) / (1 - φ²),
    so the innovation SD is sd × sqrt(1 - φ²).  ``distrvs`` is always the
    run's generator: statsmodels otherwise falls back to NumPy's global RNG
    and the run would stop replaying from its seed.  A zero SD still takes
    its draws, so an SD edit moves no other channel.
    """

    return ArmaProcess([1.0, -phi], [1.0]).generate_sample(
        nsample=shape,
        scale=sd * math.sqrt(1.0 - phi**2),
        distrvs=rng.standard_normal,
        burnin=_AR1_BURNIN,
        axis=len(shape) - 1,
    )


def _checked_market_inputs(price_assumptions: Mapping[str, object]) -> dict[str, Any]:
    """Check the price assumptions (caller inputs) and return them by name.

    Every name in ``MARKET_PRICE_INPUTS`` is required; the bounds below are
    the ones the equations need (a stationary AR(1), a probability, a
    positive width, a monotone convex curve), not calibration judgements.
    """

    missing = [name for name in MARKET_PRICE_INPUTS if name not in price_assumptions]
    if missing:
        raise ValueError(f"price_assumptions missing required field: {missing[0]}")
    values = {name: price_assumptions[name] for name in MARKET_PRICE_INPUTS}
    for name, value in values.items():
        if name.startswith("demand_shape_gw."):
            if len(value) != 7 or not all(_is_finite_number(c) for c in value):
                raise ValueError(f"{name} must be seven finite Fourier coefficients")
        elif name == "summer_months":
            if not all(isinstance(m, int) and 1 <= m <= 12 for m in value):
                raise ValueError("summer_months must be month numbers 1-12")
        elif not _is_finite_number(value):
            raise ValueError(f"{name} must be finite")
    non_negative = (
        "heating_gw_per_c",
        "wind_installed_gw",
        "wind_logit_sd",
        "supply_slope_gbp_per_mwh_per_gw",
        "scarcity_price_gbp_per_mwh",
        "daily_level_sd_gbp_per_mwh",
        "day_ahead_sd_gbp_per_mwh",
        "gate_closure_minutes",
        "intraday_hourly_sd_gbp_per_mwh",
        "imbalance_tail_scale_gbp_per_mwh",
        "solar_installed_gw",
        "solar_clearness_logit_sd",
    )
    for name in non_negative:
        if values[name] < 0:
            raise ValueError(f"{name} must be non-negative")
    for name in (
        "wind_ar1_phi",
        "daily_level_ar1_phi",
        "intraday_shock_ar1_phi",
        "solar_clearness_ar1_phi",
    ):
        if not -1.0 < values[name] < 1.0:
            raise ValueError(f"{name} must be strictly between -1 and 1")
    for name in (
        "wind_median_capacity_factor.winter",
        "wind_median_capacity_factor.summer",
        "solar_clearness_median",
    ):
        if not 0.0 < values[name] < 1.0:
            raise ValueError(f"{name} must be strictly between 0 and 1")
    if not 0.0 <= values["imbalance_long_share"] <= 1.0:
        raise ValueError("imbalance_long_share must be between 0 and 1")
    if not 0.0 <= values["imbalance_state_persistence"] < 1.0:
        raise ValueError("imbalance_state_persistence must be in [0, 1)")
    if values["scarcity_width_gw"] <= 0 or values["imbalance_tail_df"] <= 0:
        raise ValueError("scarcity_width_gw and imbalance_tail_df must be positive")
    for name in (
        "day_ahead_publication_local_hour",
        "shock_window_start_local_hour",
        "shock_window_end_local_hour",
    ):
        if not 0 <= values[name] < 24:
            raise ValueError(f"{name} must be in [0, 24)")
    if values["shock_window_start_local_hour"] == values["shock_window_end_local_hour"]:
        raise ValueError("the shock window must not be empty")
    for name in ("mild_shock_rate_per_night", "big_shock_rate_per_week", "big_shock_log_sd"):
        if values[name] < 0:
            raise ValueError(f"{name} must be non-negative")
    # A shock has a size; zero rates, not zero sizes, turn shocks off.
    for name in ("mild_shock_scale_gw", "big_shock_median_gw"):
        if values[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if values["mild_shock_rate_per_night"] > _MILD_SHOCK_CANDIDATES:
        raise ValueError(f"mild_shock_rate_per_night must be at most {_MILD_SHOCK_CANDIDATES}")
    if values["big_shock_rate_per_week"] > 7 * _BIG_SHOCK_CANDIDATES:
        raise ValueError(f"big_shock_rate_per_week must be at most {7 * _BIG_SHOCK_CANDIDATES}")
    for name in ("shock_up_probability", "mild_shock_known_share", "big_shock_known_share"):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if values["shock_max_gw"] <= 0.0:
        raise ValueError("shock_max_gw must be positive")
    if not values["price_floor_gbp_per_mwh"] < values["price_cap_gbp_per_mwh"]:
        raise ValueError("price_floor_gbp_per_mwh must be below price_cap_gbp_per_mwh")
    if not 0.0 <= values["surprise_reveal_hours"] <= 24.0:
        raise ValueError("surprise_reveal_hours must be between 0 and 24")
    if not 0.0 <= values["niv_surprise_shift"] <= 1.0:
        raise ValueError("niv_surprise_shift must be between 0 and 1")
    for season in ("winter", "summer"):
        peak = values[f"solar_clear_sky_peak_cf.{season}"]
        rise = values[f"solar_sunrise_local_hour.{season}"]
        set_ = values[f"solar_sunset_local_hour.{season}"]
        if not (0.0 <= peak <= 1.0 and 0.0 <= rise < set_ <= 24.0):
            raise ValueError(f"solar {season} needs 0 <= peak <= 1 and sunrise before sunset")
    for shock_class in ("mild", "big"):
        low = values[f"{shock_class}_shock_min_hours"]
        high = values[f"{shock_class}_shock_max_hours"]
        whole = float(2 * low).is_integer() and float(2 * high).is_integer()
        if not (0.5 <= low <= high <= 24 and whole):
            raise ValueError(
                f"{shock_class} shock durations must be whole half-hours, 0.5 h to 24 h, min <= max"
            )
    return values


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _validate_generated_prices(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"generated {name} prices must be finite")


# ============================================================================
# Non-response: whether a session follows its smart plan
# ============================================================================


def sample_non_response(
    rng: np.random.Generator, world_count: int, study_night_count: int, vehicle_count: int
) -> np.ndarray:
    """One uniform per (world, study night, EV) deciding whether a session ignores its plan.

    Returns a (world, night, EV) float64 array in [0, 1).  A home session
    belongs to the night of its plug-in and ignores its smart plan when its
    uniform is below its non-response probability (trading contract v1
    §4.5, §9.5, §9.6: a charger control outage raises that probability for
    the sessions it covers).  The uniforms are drawn after every price
    channel and always, whatever the probabilities, so editing a
    probability or adding an outage moves no other draw (§4.8).
    """

    return rng.random((world_count, study_night_count, vehicle_count))


def sample_manufacturer_outages(
    rng: np.random.Generator, world_count: int, night_count: int, manufacturer_count: int
) -> np.ndarray:
    """One uniform per (world, study night, manufacturer) deciding whether its cloud is out.

    Returns a float64 (world, night, maker) array in [0, 1); the maker is
    out for the whole night when its uniform is below its per-night outage
    probability (trading contract v1 §10.1d).  Nights are study nights, the
    non-response uniforms' indexing.  Drawn last in the run and always, one
    block whatever the probabilities, so editing an outage probability moves
    no other draw and only switches outages on or off (§10.0).
    """

    return stats.uniform.rvs(size=(world_count, night_count, manufacturer_count), random_state=rng)


__all__ = [
    "ILLUSTRATIVE_PRICE_NOISE_AR1_COEFFICIENT",
    "UNIT_COLUMNS",
    "assign_manufacturers",
    "build_population",
    "connection_window_hours",
    "MARKET_PRICE_INPUTS",
    "MarketPrices",
    "demand_shape_gw",
    "day_ahead_publication_utc_ns",
    "generate_market_price_paths",
    "sample_connection_opportunities",
    "sample_daily_temperature",
    "sample_daily_trip_inputs",
    "sample_net_demand_shocks",
    "sample_manufacturer_outages",
    "sample_non_response",
    "sample_plug_in_factors",
    "plug_in_skip_share",
    "SUPPLY_CURVE_INPUTS",
    "solar_daylight_shape",
    "supply_curve_gbp_per_mwh",
    "trapezoid_weight",
]
