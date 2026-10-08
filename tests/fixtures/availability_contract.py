"""Availability (lane J2) frame specs, validator and SYNTHETIC per-EV fixture.

What this owns (trading contract v1 §10.2, §10.9): the column specs of the J2
frames, ``validate_availability`` (the §10.9 rules for
``availability_world_slot``, ``availability_bands``,
``availability_manufacturer_world``, ``firm_share_by_fleet_size`` and
``world_nights``) and a small synthetic stand-in for the per-EV kernel
outputs J1b will produce (``planned_home_import_kwh``,
``plan_remaining_need_kwh``, ``decision_plan_kwh``, ``plan_status``), so J2 is
built and tested before J1a/J1b land.

The fixture is SYNTHETIC: a toy night per EV (plug in around 18:00, leave
around 07:00, a need of a few hours' charging, a cheapest-slot plan made at
plug-in), a skip share per (world, night) with the §10.1b logit-normal day
and week factors, makers with response rates and nightly outages.  It uses
the model's own ``action.plan_cheapest_slots`` for the plan and nothing
else from the kernel, so it cannot drift with kernel internals.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from fixtures.result_contract import _check_dtypes, _check_unique
from scipy import special

from axle_studio.model import availability
from axle_studio.model.action import plan_cheapest_slots
from axle_studio.model.summaries import FlexibilityInputs, build_study_slots

EVIDENCE_KIND = "illustrative_synthetic"
MAKER_IDS = ("m1", "m2", "m3", "m4")

AVAILABILITY_WORLD_SLOT_COLUMNS = {
    "world_id": "int",
    "slot_index": "int",
    "night_index": "int",
    "interval_start_utc": "utc",
    "interval_start_london": "london",
    "direction": "str",
    "duration_hours": "float",
    "deliverable_kw": "float",
    "deliverable_known_kw": "float",
    "deliverable_late_kw": "float",
    "potential_kw": "float",
    "eligible_power_kw": "float",
    "intraday_mean_kw": "float",
    **{f"intraday_{p}_kw": "float" for p in ("p05", "p10", "p25", "p50", "p75", "p90", "p95")},
    "intraday_known_mean_kw": "float",
    **{
        f"intraday_known_{p}_kw": "float" for p in ("p05", "p10", "p25", "p50", "p75", "p90", "p95")
    },
    "intraday_late_mean_kw": "float",
    **{f"intraday_late_{p}_kw": "float" for p in ("p10", "p50", "p90")},
    **{f"late_restored_kw_{m}": "float" for m in MAKER_IDS},
    "intraday_potential_kw": "float",
    "intraday_eligible_count": "int",
    "blackout": "bool",
    "evidence_kind": "str",
}
AVAILABILITY_MANUFACTURER_WORLD_COLUMNS = {
    "world_id": "int",
    "night_index": "int",
    "manufacturer_id": "str",
    "direction": "str",
    "duration_hours": "float",
    "ev_count": "int",
    "outage": "bool",
    "realised_mwh": "float",
    "potential_mwh": "float",
    "evidence_kind": "str",
}
AVAILABILITY_BAND_COLUMNS = {
    "horizon": "str",
    "statistic": "str",
    "direction": "str",
    "duration_hours": "float",
    "slot_index": "int",
    "night_index": "int",
    "interval_start_utc": "utc",
    "interval_start_london": "london",
    "unit": "str",
    "world_count": "int",
    "mean": "float",
    "sd": "float",
    "p05": "float",
    "p10": "float",
    "p50": "float",
    "p90": "float",
    "firm_share": "float",
    "firm_share_p05": "float",
    "implied_rho": "float",
    "n_eff": "float",
    "contributing_ev_count": "int",
    "blackout": "bool",
    "evidence_kind": "str",
}
FIRM_SHARE_BY_FLEET_SIZE_COLUMNS = {
    "fleet_size": "int",
    "window": "str",
    "direction": "str",
    "duration_hours": "float",
    "ev_count_share": "float",
    "world_count": "int",
    "mean_kw": "float",
    "sd_kw": "float",
    "p05_kw": "float",
    "p10_kw": "float",
    "p50_kw": "float",
    "p90_kw": "float",
    "firm_share": "float",
    "firm_share_p05": "float",
    "implied_rho": "float",
    "n_eff": "float",
    "evidence_kind": "str",
}
WORLD_NIGHTS_COLUMNS = {
    "world_id": "int",
    "night_index": "int",
    "night_start_local_date": "date",
    "day_type": "str",
    "holiday": "bool",
    "temperature_c": "float",
    "solar_clearness": "float",
    "plug_in_week_factor": "float",
    "plug_in_day_factor": "float",
    "skip_logit_shift": "float",
    "plug_in_skip_share": "float",
    **{f"outage_{m}": "bool" for m in MAKER_IDS},
    "plugged_in_count_at_decision": "int",
    "plugged_in_share_at_decision": "float",
    "evidence_kind": "str",
}
BAND_STATISTICS = (
    ("day_ahead", "realised"),
    ("intraday", "conditional_p05"),
    ("intraday", "conditional_p10"),
    ("intraday", "conditional_p50"),
    ("intraday", "conditional_p90"),
    ("intraday", "conditional_known_p50"),
)
_BAND_SOURCE = {
    "realised": "deliverable_kw",
    "conditional_p05": "intraday_p05_kw",
    "conditional_p10": "intraday_p10_kw",
    "conditional_p50": "intraday_p50_kw",
    "conditional_p90": "intraday_p90_kw",
    "conditional_known_p50": "intraday_known_p50_kw",
}
TOL = 1e-9


# --------------------------------------------------------------------------
# SYNTHETIC per-EV fixture
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PerEvFixture:
    """A SYNTHETIC run: slots, per-EV constants, run-level inputs and the world draws.

    ``unit(worlds)`` builds the selected-path per-EV arrays of any worlds.
    """

    study_slots: pd.DataFrame
    flex: FlexibilityInputs
    inputs: availability.AvailabilityInputs
    draws: dict
    price: np.ndarray  # (W, slot) planning price, GBP/MWh (illustrative)
    cap_kwh: float
    efficiency: float

    def unit(self, worlds) -> dict[str, np.ndarray]:
        return _fixture_unit(self, np.asarray(worlds, dtype=np.int64))


def make_per_ev_fixture(
    *,
    world_count: int,
    vehicle_count: int,
    seed: int = 7,
    night_count: int = 7,
    skip_median: float = 0.12,
    sigma_day: float = 0.5,
    sigma_week: float = 0.17,
    response_rate=(0.95, 0.95, 0.95, 0.95),
    outage_probability=(0.02, 0.02, 0.02, 0.02),
    clock_spread_slots: int = 4,
    flat_prices: bool = False,
    control_share: float = 0.0,
    blackout: np.ndarray | None = None,
    power_kw: float = 7.4,
    need_range_kwh: tuple[float, float] = (4.0, 20.0),
) -> PerEvFixture:
    """SYNTHETIC fixture of one p = 1 cohort (every EV would plug in every night).

    Clocks: plug-in at 18:00 plus a uniform offset of up to
    ``clock_spread_slots`` half-hours either way; unplug at 07:00 plus up to
    three half-hours earlier or two later (none when the spread is 0); the
    planner expects departure at 06:00 (07:00 less a 1-h margin).  The skip
    share of night ``n`` in world ``w`` is ``expit(logit(skip_median) +
    sigma_week y_w + sigma_day z_{w,n})`` (§10.1b); a session is accepted when
    its uniform is below ``1 - q``.  Makers are equal shares, assigned by a
    permutation.  All values are illustrative.
    """

    rng = np.random.default_rng(seed)
    slots = build_study_slots(date(2026, 10, 5), night_count)
    slot_count = len(slots)
    maker_count = len(response_rate)
    shape = (world_count, night_count, vehicle_count)
    week = rng.standard_normal(world_count)
    day = rng.standard_normal((world_count, night_count + 1))
    logit = special.logit(skip_median) + sigma_week * week[:, np.newaxis] + sigma_day * day
    skip = special.expit(logit)
    spread = clock_spread_slots
    late_spread = min(spread, 2)
    draws = {
        "accept": rng.random(shape) < (1.0 - skip[:, :night_count, np.newaxis]),
        "plug_offset": rng.integers(-spread, spread + 1, shape),
        "unplug_offset": rng.integers(-min(spread, 3), late_spread + 1, shape),
        "need_kwh": rng.uniform(*need_range_kwh, shape),
        "response_uniform": rng.random(shape),
        "control": rng.random(vehicle_count) < control_share,
    }
    outage = rng.random((world_count, night_count, maker_count)) < np.asarray(outage_probability)
    makers = rng.permutation(np.arange(vehicle_count) % maker_count)
    base = 100.0 + 40.0 * np.cos(2 * np.pi * (np.arange(slot_count) % 48 - 30) / 48)
    price = (
        np.zeros((world_count, slot_count))
        if flat_prices
        else base + rng.normal(0.0, 10.0, (world_count, slot_count))
    )

    # Planner's expected departure: 06:00 each morning (slot 36 of a night).
    expected = 48 * np.arange(night_count + 1) + 36
    slot = np.arange(slot_count)
    next_departure = expected[np.searchsorted(expected, slot + 1)]
    hours = np.repeat(((next_departure - slot) * 0.5)[:, np.newaxis], vehicle_count, axis=1)
    target = np.full(vehicle_count, 0.8 * 60.0)
    flex = FlexibilityInputs(
        hours_to_departure=hours,
        target_kwh=target,
        power_kw=np.full(vehicle_count, power_kw),
        efficiency=0.9,
    )
    days = night_count + 1
    inputs = availability.AvailabilityInputs(
        manufacturer_index=makers.astype(np.int64),
        response_rate=np.asarray(response_rate, dtype=float),
        outage_probability=np.asarray(outage_probability, dtype=float),
        outage=outage,
        blackout=np.zeros(slot_count, dtype=bool) if blackout is None else np.asarray(blackout),
        decision_local_time="17:00",
        warmup_days=0,
        plug_in_week_factor=week,
        plug_in_day_factor=day,
        plug_in_skip_share=skip,
        skip_logit_shift=np.zeros(days),
        holiday=np.zeros(days, dtype=bool),
        temperature_c=rng.normal(8.0, 3.0, (world_count, days)),
        solar_clearness=None,
    )
    return PerEvFixture(slots, flex, inputs, draws, price, power_kw * 0.5, 0.9)


def _fixture_unit(fixture: PerEvFixture, worlds: np.ndarray) -> dict[str, np.ndarray]:
    """Selected-path per-EV arrays (world, slot, EV) for ``worlds`` of the fixture."""

    draws = fixture.draws
    inputs = fixture.inputs
    slot_count = len(fixture.study_slots)
    accept = draws["accept"][worlds]
    world_count, night_count, vehicle_count = accept.shape
    cap = fixture.cap_kwh
    target = fixture.flex.target_kwh[0]
    eta = fixture.efficiency
    w, n, i = np.nonzero(accept)
    night_start = 48 * n
    plug = night_start + 12 + draws["plug_offset"][worlds][w, n, i]
    unplug = night_start + 38 + draws["unplug_offset"][worlds][w, n, i]
    need = draws["need_kwh"][worlds][w, n, i]
    decision = night_start + 10
    maker = inputs.manufacturer_index[i]
    # plan_status when the plan is made at plug-in (§10.1e order).
    uniform = draws["response_uniform"][worlds][w, n, i]
    status = np.where(
        draws["control"][i],
        availability.CONTROL_GROUP,
        np.where(
            uniform < 1.0 - inputs.response_rate[maker],
            availability.BASE_NON_RESPONSE,
            np.where(
                inputs.outage[worlds][w, n, maker],
                availability.MAKER_OUTAGE,
                availability.PLAN_FOLLOWS,
            ),
        ),
    ).astype(float)

    width = 48
    offset = np.arange(width)
    t = plug[:, np.newaxis] + offset
    window_end = np.minimum(night_start + 36, slot_count)  # planner's expected departure
    in_window = t < window_end[:, np.newaxis]
    safe_t = np.minimum(t, slot_count - 1)
    window_price = np.where(in_window, fixture.price[worlds][w[:, np.newaxis], safe_t], np.inf)
    plan = plan_cheapest_slots(need, np.full(len(need), cap), window_price)
    # Normal rule: full power from plug-in until the need is met.
    normal = np.clip(need[:, np.newaxis] - offset * cap, 0.0, cap)
    connected = t < unplug[:, np.newaxis]
    follows = (status == availability.PLAN_FOLLOWS)[:, np.newaxis]
    metered = np.where(connected, np.where(follows, plan, normal), 0.0)
    before = np.cumsum(metered, axis=1) - metered
    plan_before = np.cumsum(plan, axis=1) - plan
    # The plan in force at 17:00 (snapshot): kept after an early unplug,
    # zero before the decision slot and after the night's end.
    at_decision = (plug <= decision) & (unplug > decision)
    snapshot = (
        at_decision[:, np.newaxis]
        & (t >= decision[:, np.newaxis])
        & (t < night_start[:, np.newaxis] + 48)
    )

    shape = (world_count, slot_count, vehicle_count)
    out = {
        "home_grid_import_kwh": np.zeros(shape),
        "opening_battery_kwh": np.full(shape, target),
        "connected": np.zeros(shape, dtype=bool),
        "planned_home_import_kwh": np.zeros(shape),
        "plan_remaining_need_kwh": np.zeros(shape),
        "decision_plan_kwh": np.zeros(shape),
        "plan_status": np.zeros(shape),
    }
    ww = np.broadcast_to(w[:, np.newaxis], t.shape)
    ii = np.broadcast_to(i[:, np.newaxis], t.shape)
    live = connected & (t < slot_count)
    key = (ww[live], t[live], ii[live])
    out["connected"][key] = True
    out["home_grid_import_kwh"][key] = metered[live]
    out["opening_battery_kwh"][key] = (target - need[:, np.newaxis] * eta + eta * before)[live]
    out["planned_home_import_kwh"][key] = plan[live]
    out["plan_remaining_need_kwh"][key] = np.maximum(need[:, np.newaxis] - plan_before, 0.0)[live]
    out["plan_status"][key] = np.broadcast_to(status[:, np.newaxis], t.shape)[live]
    kept = snapshot & (t < slot_count)
    out["decision_plan_kwh"][ww[kept], t[kept], ii[kept]] = plan[kept]
    return out


def fixture_totals(fixture: PerEvFixture, chunk_size: int = 10) -> dict:
    """Run the chunk-loop hook over every world of the fixture; returns filled totals."""

    world_count = fixture.draws["accept"].shape[0]
    totals = availability.new_totals(fixture.inputs, fixture.study_slots, fixture.flex, world_count)
    for first in range(0, world_count, chunk_size):
        worlds = np.arange(first, min(first + chunk_size, world_count))
        availability.accumulate(totals, worlds, fixture.unit(worlds))
    return totals


def per_ev_realised(fixture: PerEvFixture) -> np.ndarray:
    """(world, slot, cell, EV) realised kW of every EV, for the validator's ρ recomputation."""

    world_count = fixture.draws["accept"].shape[0]
    unit = fixture.unit(np.arange(world_count))
    cells = []
    for direction, hours in availability.CELLS:
        out = availability.ev_contributions(
            unit,
            power_kw=fixture.flex.power_kw,
            target_kwh=fixture.flex.target_kwh,
            efficiency=fixture.flex.efficiency,
            expected_end_slot=availability.expected_end_slots(fixture.flex.hours_to_departure),
            night_index=fixture.study_slots["night_index"].to_numpy(),
            night_decision_slots=availability.decision_slots(fixture.study_slots, "17:00"),
            blackout=fixture.inputs.blackout,
            duration_slots=int(2 * hours),
        )
        cells.append(out["realised"][direction])
    return np.stack(cells, axis=2)


def fleet_home_import_kw(fixture: PerEvFixture) -> np.ndarray:
    """(world, slot) selected-path fleet home import in kW, for the validator."""

    world_count = fixture.draws["accept"].shape[0]
    return fixture.unit(np.arange(world_count))["home_grid_import_kwh"].sum(axis=2) / 0.5


# --------------------------------------------------------------------------
# Validator (§10.9)
# --------------------------------------------------------------------------


def _cube(frame: pd.DataFrame, column: str, world_count: int, slot_count: int) -> np.ndarray:
    """(world, slot, cell) array of a world-slot column (rows are world-major)."""

    return (
        frame[column]
        .to_numpy(dtype=float)
        .reshape(world_count, slot_count, len(availability.CELLS))
    )


def validate_availability(
    result,
    *,
    response_rate,
    outage_probability,
    decision_local_time: str = "17:00",
    home_import_kw: np.ndarray | None = None,
    per_ev_realised: np.ndarray | None = None,
) -> None:
    """Assert the §10.9 rules on the J2 frames of ``result``; raise AssertionError naming the rule.

    Duck-typed: reads the attributes ``availability_world_slot``,
    ``availability_bands``, ``availability_manufacturer_world``,
    ``firm_share_by_fleet_size`` and ``world_nights``.  ``response_rate`` and
    ``outage_probability`` are the makers' records; ``home_import_kw``
    (world, slot), when given, is the selected-path fleet home import;
    ``per_ev_realised`` (world, slot, cell, EV), when given, the per-EV
    realised kW from which ``implied_rho``, ``n_eff`` and
    ``contributing_ev_count`` of the realised band rows are recomputed.
    """

    with warnings.catch_warnings():
        # All-NaN slots (windows past the end) are expected; NaN is their value.
        warnings.simplefilter("ignore", RuntimeWarning)
        _validate(
            result,
            response_rate,
            outage_probability,
            decision_local_time,
            home_import_kw,
            per_ev_realised,
        )


def _validate(
    result, response_rate, outage_probability, decision_local_time, home_import_kw, per_ev_realised
) -> None:
    world_slot = result.availability_world_slot
    bands = result.availability_bands
    makers = result.availability_manufacturer_world
    sizes = result.firm_share_by_fleet_size
    nights = result.world_nights
    _check_dtypes(world_slot, AVAILABILITY_WORLD_SLOT_COLUMNS, "availability_world_slot")
    _check_dtypes(bands, AVAILABILITY_BAND_COLUMNS, "availability_bands")
    _check_dtypes(
        makers, AVAILABILITY_MANUFACTURER_WORLD_COLUMNS, "availability_manufacturer_world"
    )
    _check_dtypes(sizes, FIRM_SHARE_BY_FLEET_SIZE_COLUMNS, "firm_share_by_fleet_size")
    _check_dtypes(nights, WORLD_NIGHTS_COLUMNS, "world_nights")
    for frame, name in ((world_slot, "availability_world_slot"), (bands, "availability_bands")):
        assert (frame["evidence_kind"] == EVIDENCE_KIND).all(), f"{name} evidence_kind"
    _check_unique(
        world_slot,
        ["world_id", "slot_index", "direction", "duration_hours"],
        "availability_world_slot",
    )
    _check_unique(
        bands,
        ["horizon", "statistic", "direction", "duration_hours", "slot_index"],
        "availability_bands",
    )
    _check_unique(
        makers,
        ["world_id", "night_index", "manufacturer_id", "direction", "duration_hours"],
        "availability_manufacturer_world",
    )
    _check_unique(
        sizes, ["fleet_size", "window", "direction", "duration_hours"], "firm_share_by_fleet_size"
    )
    _check_unique(nights, ["world_id", "night_index"], "world_nights")

    world_count = int(world_slot["world_id"].max()) + 1
    slot_count = int(world_slot["slot_index"].max()) + 1
    cells = len(availability.CELLS)
    assert len(world_slot) == world_count * slot_count * cells, "world-slot row count"
    # World-major order: world, slot, direction, duration.
    assert (
        world_slot["world_id"].to_numpy() == np.repeat(np.arange(world_count), slot_count * cells)
    ).all()
    assert (
        world_slot["slot_index"].to_numpy()
        == np.tile(np.repeat(np.arange(slot_count), cells), world_count)
    ).all()
    cell_directions = np.array([d for d, _ in availability.CELLS], dtype=object)
    cell_hours = np.array([h for _, h in availability.CELLS])
    assert (
        world_slot["direction"].to_numpy() == np.tile(cell_directions, world_count * slot_count)
    ).all()
    assert (
        world_slot["duration_hours"].to_numpy() == np.tile(cell_hours, world_count * slot_count)
    ).all()

    def cube(column):
        return _cube(world_slot, column, world_count, slot_count)

    deliverable = cube("deliverable_kw")
    known = cube("deliverable_known_kw")
    late = cube("deliverable_late_kw")
    potential = cube("potential_kw")
    slot = np.arange(slot_count)
    duration_slots = np.asarray(availability.DURATION_SLOTS * len(availability.DIRECTIONS))
    past_end = slot[:, np.newaxis] + duration_slots[np.newaxis, :] > slot_count
    assert (np.isnan(deliverable) == past_end[np.newaxis]).all(), (
        "deliverable NaN exactly past the end"
    )
    defined = ~np.isnan(deliverable)
    assert (deliverable[defined] >= -TOL).all(), "deliverable_kw >= 0"
    assert (deliverable[defined] <= potential[defined] + 1e-6).all(), (
        "deliverable_kw <= potential_kw"
    )
    assert np.allclose(deliverable, known + late, atol=1e-6, equal_nan=True), (
        "deliverable = known + late"
    )
    assert (cube("eligible_power_kw")[defined] >= deliverable[defined] - 1e-6).all(), (
        "eligible >= deliverable"
    )
    by_duration = deliverable.reshape(world_count, slot_count, len(availability.DIRECTIONS), -1)
    steps = np.diff(by_duration, axis=3)
    assert (steps[~np.isnan(steps)] <= 1e-6).all(), "deliverable_kw non-increasing in duration"
    if home_import_kw is not None:
        half_hour_down = deliverable[..., 0]
        assert (
            half_hour_down[~np.isnan(half_hour_down)]
            <= home_import_kw[~np.isnan(half_hour_down)] + 1e-6
        ).all(), "0.5-h turn-down exceeds the fleet home import"

    # Intraday NaN pattern: before the night's decision slot and past the end.
    first = world_slot.iloc[:: cells * 1][
        ["slot_index", "night_index", "interval_start_london"]
    ].iloc[:slot_count]
    labels = first["interval_start_london"].dt.strftime("%H:%M").to_numpy()
    night_index = first["night_index"].to_numpy()
    decision = np.array(
        [
            np.flatnonzero((night_index == n) & (labels == decision_local_time))[0]
            for n in range(night_index.max() + 1)
        ]
    )
    before = slot < decision[night_index]
    intraday_nan = before[:, np.newaxis] | past_end
    intraday_mean = cube("intraday_mean_kw")
    if world_count > 1:
        assert (np.isnan(intraday_mean) == intraday_nan[np.newaxis]).all(), (
            "intraday NaN exactly before s_n / past end"
        )
    levels = ("p05", "p10", "p25", "p50", "p75", "p90", "p95")
    quantile_names = [f"intraday_{p}_kw" for p in levels]
    for part in ("intraday", "intraday_known"):
        quantiles = np.stack([cube(f"{part}_{p}_kw") for p in levels], axis=-1)
        steps = np.diff(quantiles, axis=-1)
        assert (steps[~np.isnan(steps)] >= -1e-6).all(), (
            f"{part} quantiles ordered p05 <= ... <= p95"
        )
    known_mean = cube("intraday_known_mean_kw")
    late_mean = cube("intraday_late_mean_kw")
    assert np.allclose(intraday_mean, known_mean + late_mean, atol=1e-6, equal_nan=True), (
        "intraday mean = known + late"
    )
    restored = np.stack([cube(f"late_restored_kw_{m}") for m in MAKER_IDS], axis=-1)
    assert (restored.sum(axis=-1)[defined] >= late[defined] - 1e-6).all(), "restoration only adds"
    r = np.asarray(response_rate, dtype=float)
    pi = np.asarray(outage_probability, dtype=float)
    intraday_potential = cube("intraday_potential_kw")
    weight = (1.0 - pi) * r
    ok = ~np.isnan(known_mean)
    assert (known_mean[ok] >= weight.min() * intraday_potential[ok] - 1e-6).all(), (
        "known mean below its range"
    )
    assert (known_mean[ok] <= weight.max() * intraday_potential[ok] + 1e-6).all(), (
        "known mean above its range"
    )
    if np.ptp(weight) == 0:
        assert np.allclose(known_mean[ok], weight[0] * intraday_potential[ok], atol=1e-6), (
            "known mean recomputed"
        )
    if world_count > 1:
        others = (restored.sum(axis=0, keepdims=True) - restored) / (
            world_count - 1
        )  # leave-one-out means
        expected_late = others @ (1.0 - pi)
        assert np.allclose(
            late_mean,
            np.where(intraday_nan[np.newaxis], np.nan, expected_late),
            atol=1e-6,
            equal_nan=True,
        ), "late mean recomputed"
        # Reporting precision is one bin of the per-slot grid (§10.2d).  The
        # upper bound allows 1.5 bins: each part rounds to its nearest point
        # (up to ½ bin each) and the centred bin reaches ½ bin above it.
        x_max = np.nanmax(intraday_potential, axis=0) + np.nanmax(restored.sum(axis=-1), axis=0)
        bin_kw = x_max / availability.INTRADAY_GRID_BINS
        high = np.nanmax(restored.sum(axis=-1), axis=0)[np.newaxis] + intraday_potential
        top = quantiles[..., -1]
        ok = ~np.isnan(top)
        assert (quantiles[..., 0][ok] >= -TOL).all(), "intraday quantiles >= 0"
        assert (top[ok] <= (high + 1.5 * np.broadcast_to(bin_kw, high.shape))[ok] + 1e-6).all(), (
            "intraday quantiles in range"
        )
        if (pi == 0).all():
            _check_late_quantiles(world_slot, restored, bin_kw, world_count, slot_count)
    # Blackout: zero on and across blackout slots.
    blackout = world_slot["blackout"].to_numpy()[: slot_count * cells : cells]
    for cell, duration in enumerate(duration_slots):
        blocked = availability.window_overlaps(blackout, int(duration)) & ~past_end[:, cell]
        for column in (
            "deliverable_kw",
            "deliverable_known_kw",
            "deliverable_late_kw",
            "potential_kw",
        ):
            assert (cube(column)[:, blocked, cell] == 0).all(), f"{column} not 0 across a blackout"
        blocked_intraday = blocked & ~before
        if world_count > 1:
            for name in ["intraday_mean_kw", *quantile_names]:
                assert (cube(name)[:, blocked_intraday, cell] == 0).all(), (
                    f"{name} not 0 across a blackout"
                )

    _check_bands(bands, world_slot, world_count, slot_count, per_ev_realised)
    _check_manufacturer_world(makers, nights, deliverable, potential, night_index, world_count)
    _check_fleet_sizes(sizes, deliverable, first, world_count)
    vehicle_count = int(sizes["fleet_size"].max())
    assert len(nights) == world_count * (night_index.max() + 1), (
        "world_nights: one row per (world, night)"
    )
    count = nights["plugged_in_count_at_decision"].to_numpy()
    assert ((count >= 0) & (count <= vehicle_count)).all(), "plugged-in count in [0, N]"
    assert np.allclose(nights["plugged_in_share_at_decision"], count / vehicle_count), (
        "plugged-in share"
    )


def _check_late_quantiles(world_slot, restored, bin_kw, world_count, slot_count) -> None:
    """With every π = 0, the late quantiles are the other worlds' inverted-CDF quantiles.

    The lattice rounds each value to the nearest bin point and interpolates
    within that point's centred bin, so the two agree within one bin.
    """

    total = restored.sum(axis=-1)  # (world, slot, cell)
    for name, level in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        reported = _cube(world_slot, f"intraday_late_{name}_kw", world_count, slot_count)
        for world in range(world_count):
            others = np.delete(total, world, axis=0)
            expected = np.quantile(others, level, axis=0, method="inverted_cdf")
            ok = ~np.isnan(reported[world])
            gap = np.abs(reported[world] - expected)[ok]
            assert (gap <= np.broadcast_to(bin_kw, expected.shape)[ok] + 1e-6).all(), (
                f"intraday_late_{name}_kw is not the other worlds' quantile"
            )


def _check_bands(bands, world_slot, world_count, slot_count, per_ev_realised) -> None:
    cells = len(availability.CELLS)
    assert len(bands) == len(BAND_STATISTICS) * cells * slot_count, "availability_bands row count"
    order = [(h, s) for h, s in BAND_STATISTICS for _ in range(cells * slot_count)]
    assert list(zip(bands["horizon"], bands["statistic"], strict=True)) == order, "band row order"
    for horizon, statistic in BAND_STATISTICS:
        rows = bands[(bands["horizon"] == horizon) & (bands["statistic"] == statistic)]
        values = _cube(world_slot, _BAND_SOURCE[statistic], world_count, slot_count)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = {
                "mean": values.mean(axis=0),
                "p05": np.quantile(values, 0.05, axis=0),
                "p10": np.quantile(values, 0.1, axis=0),
                "p50": np.quantile(values, 0.5, axis=0),
                "p90": np.quantile(values, 0.9, axis=0),
            }
        for column, array in expected.items():
            got = rows[column].to_numpy(dtype=float)
            assert np.allclose(got, array.T.ravel(), atol=1e-9, equal_nan=True), (
                f"availability_bands {statistic}.{column} is not the world-first recomputation"
            )
        _ordered = rows[["p05", "p10", "p50", "p90"]].dropna().to_numpy()
        assert (np.diff(_ordered, axis=1) >= -TOL).all(), "band quantiles ordered"
        p50 = rows["p50"].to_numpy()
        positive = p50 > 0
        assert np.allclose(
            rows["firm_share"].to_numpy()[positive], (rows["p10"].to_numpy() / p50)[positive]
        ), "firm_share"
        assert rows["firm_share"][~positive].isna().all(), "firm_share NaN where p50 = 0"
        if horizon == "intraday":
            assert rows["implied_rho"].isna().all() and rows["n_eff"].isna().all(), (
                "rho/n_eff NaN on intraday rows"
            )
            assert (rows["contributing_ev_count"] == 0).all(), "contributing count 0 on intraday"
        else:
            if per_ev_realised is not None:
                _check_implied_rho(rows, per_ev_realised)
            rho = rows["implied_rho"].to_numpy()
            ok = rho >= 0
            assert (
                rows["n_eff"].to_numpy()[ok] <= rows["contributing_ev_count"].to_numpy()[ok] + 1e-9
            ).all(), "n_eff <= contributing_ev_count"


def _check_implied_rho(rows: pd.DataFrame, per_ev: np.ndarray) -> None:
    """§10.2c recomputed from per-EV realised kW (world, slot, cell, EV); rows are cell-major."""

    fleet = per_ev.sum(axis=3)
    ev_var = per_ev.var(axis=0, ddof=1)  # (slot, cell, EV)
    ev_var = np.where(ev_var < 1e-24, 0.0, ev_var)  # an EV constant across worlds
    variance_sum = ev_var.sum(axis=2)
    sd_sum = np.sqrt(ev_var).sum(axis=2)
    count = (ev_var > 0).sum(axis=2)
    denominator = sd_sum**2 - variance_sum
    defined = (count >= 2) & (denominator > 1e-9 * variance_sum)
    rho = np.where(
        defined,
        (fleet.var(axis=0, ddof=1) - variance_sum) / np.where(defined, denominator, 1.0),
        np.nan,
    )
    n_eff = np.where(defined, count / (1 + (count - 1) * np.maximum(rho, 0.0)), np.nan)
    assert (rows["contributing_ev_count"].to_numpy() == count.T.ravel()).all(), (
        "contributing_ev_count is not the count of EVs that vary across worlds"
    )
    for column, expected in (("implied_rho", rho), ("n_eff", n_eff)):
        assert np.allclose(
            rows[column].to_numpy(), expected.T.ravel(), rtol=1e-6, atol=1e-9, equal_nan=True
        ), f"availability_bands {column} is not the §10.2c recomputation"


def _check_manufacturer_world(
    makers, nights, deliverable, potential, night_index, world_count
) -> None:
    cells = len(availability.CELLS)
    night_count = night_index.max() + 1
    maker_count = len(MAKER_IDS)
    realised = (
        makers["realised_mwh"].to_numpy().reshape(world_count, night_count, maker_count, cells)
    )
    possible = (
        makers["potential_mwh"].to_numpy().reshape(world_count, night_count, maker_count, cells)
    )
    indicator = (night_index[np.newaxis, :] == np.arange(night_count)[:, np.newaxis]).astype(float)
    fleet = np.einsum("wtc,nt->wnc", np.nan_to_num(deliverable), indicator) * 0.5 / 1000
    fleet_potential = np.einsum("wtc,nt->wnc", np.nan_to_num(potential), indicator) * 0.5 / 1000
    assert np.allclose(realised.sum(axis=2), fleet, atol=1e-9), (
        "maker realised_mwh sums to the fleet"
    )
    assert np.allclose(possible.sum(axis=2), fleet_potential, atol=1e-9), (
        "maker potential_mwh sums to the fleet"
    )
    outage = (
        makers["outage"].to_numpy().reshape(world_count, night_count, maker_count, cells)[..., 0]
    )
    stored = (
        nights[[f"outage_{m}" for m in MAKER_IDS]]
        .to_numpy()
        .reshape(world_count, night_count, maker_count)
    )
    assert (outage == stored).all(), "outage equals world_nights.outage_m<k>"


def _check_fleet_sizes(sizes, deliverable, first_rows, world_count) -> None:
    fleet_sizes = sizes["fleet_size"].drop_duplicates().to_numpy()
    assert (np.diff(fleet_sizes) > 0).all(), "fleet sizes ascending"
    vehicle_count = fleet_sizes[-1]
    labels = first_rows["interval_start_london"].dt.strftime("%H:%M").to_numpy(dtype=object)
    full = sizes[sizes["fleet_size"] == vehicle_count]
    for k, (window, (start, end)) in enumerate(availability.PRODUCT_WINDOWS.items()):
        mask = (
            (labels >= start) & (labels < end)
            if start < end
            else (labels >= start) | (labels < end)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a window with no defined slot
            per_world = np.nanmean(deliverable[:, mask], axis=1)  # (world, cell)
        rows = full[full["window"] == window]
        assert np.allclose(rows["mean_kw"], per_world.mean(axis=0), atol=1e-9, equal_nan=True), (
            f"fleet_size = N mean differs from the fleet ({window})"
        )
        assert np.allclose(
            rows["p50_kw"], np.quantile(per_world, 0.5, axis=0), atol=1e-9, equal_nan=True
        ), f"fleet_size = N p50 differs from the fleet ({window})"
    for (window, direction, hours), group in sizes.groupby(
        ["window", "direction", "duration_hours"], sort=False
    ):
        p50 = group["p50_kw"].dropna().to_numpy()
        assert (np.diff(p50) >= -1e-9).all(), (
            f"p50_kw non-decreasing in fleet size ({window} {direction} {hours})"
        )
