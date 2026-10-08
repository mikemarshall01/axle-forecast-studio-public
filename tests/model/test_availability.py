"""Firm-MW availability (trading contract v1 §10.2, lane J2) on SYNTHETIC per-EV fixtures.

J1a/J1b (shared factors, makers, the kernel's plan outputs) are not built
yet, so these tests feed ``availability`` hand-built and toy per-EV arrays
with the §10.1e shapes and codes.  Every figure is illustrative.
"""

from __future__ import annotations

import dataclasses
import types
import warnings

import numpy as np
import pytest
from fixtures.availability_contract import (
    fixture_totals,
    fleet_home_import_kw,
    make_per_ev_fixture,
    per_ev_realised,
    validate_availability,
)
from scipy import special, stats

from axle_studio.model import availability, physics

NAN = np.nan


# --------------------------------------------------------------------------
# Hand fixture: one EV, six slots (§10.2a)
# --------------------------------------------------------------------------


def _one_ev(status: float, *, unplug: int = 6, follows_plan: bool = True) -> dict[str, np.ndarray]:
    """One EV, six slots, 4 kW (2 kWh a slot), efficiency 1, target 10 kWh, 6 kWh at slot 0.

    The plan charges its 4 kWh need in slots 1 and 2 (the cheap ones); a
    session that ignores it charges at full power from slot 0.
    """

    plan = np.array([0.0, 2.0, 2.0, 0.0, 0.0, 0.0])
    normal = np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    connected = np.arange(6) < unplug
    metered = np.where(connected, plan if follows_plan else normal, 0.0)
    opening = 6.0 + np.concatenate([[0.0], np.cumsum(metered)[:-1]])
    remaining = np.where(connected, 4.0 - np.concatenate([[0.0], np.cumsum(plan)[:-1]]), 0.0)

    def cube(values):
        return np.asarray(values, dtype=float).reshape(1, 6, 1)

    return {
        "home_grid_import_kwh": cube(metered),
        "opening_battery_kwh": cube(opening),
        "connected": connected.reshape(1, 6, 1),
        "planned_home_import_kwh": cube(np.where(connected, plan, 0.0)),
        "plan_remaining_need_kwh": cube(remaining),
        # The plan in force at the decision slot 1, kept after an early unplug.
        "decision_plan_kwh": cube([0.0, 2.0, 2.0, 0.0, 0.0, 0.0]),
        "plan_status": cube(np.where(connected, status, 0.0)),
    }


def _hand(unit, duration_slots, blackout=None):
    return availability.ev_contributions(
        unit,
        power_kw=np.array([4.0]),
        target_kwh=np.array([10.0]),
        efficiency=1.0,
        expected_end_slot=np.full((6, 1), 6),  # planner expects departure at the end
        night_index=np.zeros(6, dtype=np.int64),
        night_decision_slots=np.array([1]),
        blackout=np.zeros(6, dtype=bool) if blackout is None else blackout,
        duration_slots=duration_slots,
    )


# Worked by hand from the §10.2a formulas (kW; NaN = window past the end).
# Follower, plugged in all six slots: m = g = [0, 2, 2, 0, 0, 0], need [4, 4, 2, 0, 0, 0].
FOLLOWER = {
    1: {"turn_down": [0, 4, 4, 0, 0, 0], "turn_up": [4, 0, 0, 0, 0, 0]},
    2: {"turn_down": [0, 4, 0, 0, 0, NAN], "turn_up": [0, 0, 0, 0, 0, NAN]},
    4: {"turn_down": [0, 0, 0, NAN, NAN, NAN], "turn_up": [0, 0, 0, NAN, NAN, NAN]},
    8: {"turn_down": [NAN] * 6, "turn_up": [NAN] * 6},
}
# Plan-based from the decision at slot 1: nothing before it; the plan's
# remaining need 4 kWh at slot 1, deadline 6.
PLAN_BASED = {
    1: {"turn_down": [0, 4, 4, 0, 0, 0], "turn_up": [0, 0, 0, 0, 0, 0]},
    2: {"turn_down": [0, 4, 0, 0, 0, NAN], "turn_up": [0, 0, 0, 0, 0, NAN]},
    4: {"turn_down": [0, 0, 0, NAN, NAN, NAN], "turn_up": [0, 0, 0, NAN, NAN, NAN]},
    8: {"turn_down": [NAN] * 6, "turn_up": [NAN] * 6},
}


def _assert_values(actual, expected):
    np.testing.assert_allclose(actual[0, :, 0], np.asarray(expected, dtype=float), atol=1e-12)


@pytest.mark.parametrize("duration_slots", [1, 2, 4, 8])
def test_hand_fixture_follower(duration_slots):
    out = _hand(_one_ev(availability.PLAN_FOLLOWS), duration_slots)
    for direction in availability.DIRECTIONS:
        _assert_values(out["realised"][direction], FOLLOWER[duration_slots][direction])
        # A follower's potential is its realised value (the plan is what it does).
        _assert_values(out["potential"][direction], FOLLOWER[duration_slots][direction])
        _assert_values(out["plan_based"][direction], PLAN_BASED[duration_slots][direction])


@pytest.mark.parametrize("duration_slots", [1, 2, 4, 8])
def test_hand_fixture_non_responder_counts_at_its_plan(duration_slots):
    # It charges at full power from plug-in and takes no instruction: realised
    # 0, but the aggregator's potential is its plan (B6), and the 17:00 known
    # part does not yet know it will ignore calls.
    out = _hand(_one_ev(physics.BASE_NON_RESPONSE, follows_plan=False), duration_slots)
    past_end = np.isnan(np.asarray(FOLLOWER[duration_slots]["turn_down"], dtype=float))
    for direction in availability.DIRECTIONS:
        _assert_values(out["realised"][direction], np.where(past_end, NAN, 0.0))
        _assert_values(out["potential"][direction], FOLLOWER[duration_slots][direction])
        _assert_values(out["plan_based"][direction], PLAN_BASED[duration_slots][direction])


def test_hand_fixture_early_unplug_cuts_the_realised_deadline():
    # Unplugged at slot 4, before the planner's expected departure (6): the
    # realised deadline is the unplug (B2), so a 1-h pause at slot 1 no longer
    # fits (need 4 kWh, one slot of 2 kWh left); the plan-based figure uses
    # the planner's expectation only and still counts it.
    out = _hand(_one_ev(availability.PLAN_FOLLOWS, unplug=4), 2)
    _assert_values(out["realised"]["turn_down"], [0, 0, 0, 0, 0, NAN])
    _assert_values(out["plan_based"]["turn_down"], [0, 4, 0, 0, 0, NAN])
    out = _hand(_one_ev(availability.PLAN_FOLLOWS, unplug=4), 1)
    _assert_values(out["realised"]["turn_down"], [0, 4, 4, 0, 0, 0])
    _assert_values(out["realised"]["turn_up"], [4, 0, 0, 0, 0, 0])


def test_control_ev_and_blackout_give_zero():
    out = _hand(_one_ev(availability.CONTROL_GROUP), 1)
    for variant in ("realised", "potential", "plan_based"):
        for direction in availability.DIRECTIONS:
            _assert_values(out[variant][direction], [0] * 6)
    blackout = np.array([False, False, True, False, False, False])
    out = _hand(_one_ev(availability.PLAN_FOLLOWS), 2, blackout)
    # Windows starting at slots 1 and 2 touch the blackout slot 2.
    _assert_values(out["realised"]["turn_down"], [0, 0, 0, 0, 0, NAN])
    _assert_values(out["plan_based"]["turn_down"], [0, 0, 0, 0, 0, NAN])


def test_plan_based_deadline_stops_at_the_night_end():
    # The plan snapshot at s_n is night n's plan: a window running into the
    # next night is not plan-based deliverable even if the planner's expected
    # departure is later.  Slots 0-2 are night 0 (decision slot 1), 3-5 night 1.
    unit = _one_ev(availability.PLAN_FOLLOWS)
    unit["decision_plan_kwh"] = np.array([0.0, 2.0, 2.0, 2.0, 0.0, 0.0]).reshape(1, 6, 1)
    unit["plan_remaining_need_kwh"][0, 1, 0] = 6.0

    def plan_based(night_index, decision):
        return availability.ev_contributions(
            unit,
            power_kw=np.array([4.0]),
            target_kwh=np.array([10.0]),
            efficiency=1.0,
            expected_end_slot=np.full((6, 1), 6),
            night_index=np.asarray(night_index),
            night_decision_slots=np.asarray(decision),
            blackout=np.zeros(6, dtype=bool),
            duration_slots=2,
        )["plan_based"]["turn_down"]

    # One night (deadline 6): pausing {1, 2} leaves 6 kWh for slots 3-5, and
    # pausing {2, 3} leaves 4 kWh for slots 4-5, so both count 4 kW.
    _assert_values(plan_based([0] * 6, [1]), [0, 4, 4, 0, 0, NAN])
    # Two nights: the deadline is the night end (slot 3), so {1, 2} leaves no
    # slot for the need and {2, 3} crosses into night 1: both 0.
    _assert_values(plan_based([0, 0, 0, 1, 1, 1], [1, 4]), [0, 0, 0, 0, 0, NAN])


def test_implied_rho_is_nan_with_a_single_varying_ev():
    # Regression (review B1): with one EV that varies across worlds, (Σ√v)² - Σv
    # is float dust and used to give ρ values like -3.
    one = availability.correlation_diagnostics(
        np.array([[0.0], [0.7], [1.3]]), np.array([0.1]), np.array([np.sqrt(0.1)]), np.array([1])
    )
    assert np.isnan(one["implied_rho"]).all() and np.isnan(one["n_eff"]).all()
    fixture = make_per_ev_fixture(world_count=5, vehicle_count=12, seed=8)
    worlds = np.arange(5)
    unit = fixture.unit(worlds)
    for values in unit.values():
        values[:, :, 1:] = values[:1, :, 1:]  # every EV but the first is the same in every world
    totals = availability.new_totals(fixture.inputs, fixture.study_slots, fixture.flex, 5)
    availability.accumulate(totals, worlds, unit)
    diagnostics = availability.realised_diagnostics(totals)
    single = diagnostics["contributing_ev_count"] == 1
    assert single.any()
    assert np.isnan(diagnostics["implied_rho"][single]).all()
    assert np.isnan(diagnostics["n_eff"][single]).all()
    assert (diagnostics["contributing_ev_count"] <= 1).all()


def test_contributions_non_increasing_in_duration():
    fixture = make_per_ev_fixture(world_count=3, vehicle_count=40, seed=11)
    unit = fixture.unit(np.arange(3))
    previous = None
    for duration_slots in availability.DURATION_SLOTS:
        out = availability.ev_contributions(
            unit,
            power_kw=fixture.flex.power_kw,
            target_kwh=fixture.flex.target_kwh,
            efficiency=fixture.flex.efficiency,
            expected_end_slot=availability.expected_end_slots(fixture.flex.hours_to_departure),
            night_index=fixture.study_slots["night_index"].to_numpy(),
            night_decision_slots=availability.decision_slots(fixture.study_slots, "17:00"),
            blackout=fixture.inputs.blackout,
            duration_slots=duration_slots,
        )
        if previous is not None:
            for variant in out:
                for direction in availability.DIRECTIONS:
                    now, before = out[variant][direction], previous[variant][direction]
                    both = ~np.isnan(now)
                    assert (now[both] <= before[both] + 1e-9).all(), (variant, direction)
        previous = out


# --------------------------------------------------------------------------
# Frames and the §10.9 validator
# --------------------------------------------------------------------------


def _frames(fixture):
    totals = fixture_totals(fixture)
    return totals, types.SimpleNamespace(**availability.build_frames(totals, fixture.study_slots))


def test_frames_pass_the_validator_on_the_synthetic_fixture():
    fixture = make_per_ev_fixture(world_count=8, vehicle_count=60, seed=5, control_share=0.1)
    _, frames = _frames(fixture)
    validate_availability(
        frames,
        response_rate=fixture.inputs.response_rate,
        outage_probability=fixture.inputs.outage_probability,
        home_import_kw=fleet_home_import_kw(fixture),
        per_ev_realised=per_ev_realised(fixture),
    )
    assert len(frames.availability_world_slot) == 8 * 336 * 8
    assert len(frames.availability_bands) == 16_128
    assert len(frames.availability_manufacturer_world) == 8 * 7 * 4 * 8
    assert list(frames.firm_share_by_fleet_size["fleet_size"].unique()) == [60]
    assert (
        frames.world_nights["outage_m1"].to_numpy() == fixture.inputs.outage[..., 0].ravel()
    ).all()


def test_fixture_maker_realises_nothing_on_its_outage_nights():
    # Lead ruling: a fixture-only check.  Fixture sessions never span two
    # nights; real sessions can, so the general validator does not assert it.
    fixture = make_per_ev_fixture(
        world_count=6, vehicle_count=40, seed=5, outage_probability=(0.3, 0.3, 0.0, 0.0)
    )
    frame = availability._manufacturer_world_frame(fixture_totals(fixture))
    assert frame["outage"].any()
    assert (frame.loc[frame["outage"], "realised_mwh"].abs() <= 1e-12).all()
    assert (frame.loc[frame["outage"], "potential_mwh"] > 0).any()


def test_blackout_zeroes_every_window_that_touches_it():
    slots = 336
    blackout = np.zeros(slots, dtype=bool)
    blackout[48 * 2 + 14 : 48 * 2 + 18] = True  # 19:00-21:00 on night 2
    fixture = make_per_ev_fixture(world_count=4, vehicle_count=40, seed=9, blackout=blackout)
    _, frames = _frames(fixture)
    validate_availability(
        frames,
        response_rate=fixture.inputs.response_rate,
        outage_probability=fixture.inputs.outage_probability,
    )
    rows = frames.availability_world_slot
    assert rows.loc[rows["blackout"], "deliverable_kw"].eq(0).all()
    assert rows["blackout"].sum() == 4 * 4 * 8


def test_frames_pass_with_all_outages_off():
    # With every π = 0 the validator also checks the late quantiles against
    # the other worlds' inverted-CDF quantiles (§10.9).
    fixture = make_per_ev_fixture(
        world_count=6, vehicle_count=40, seed=2, outage_probability=(0, 0, 0, 0)
    )
    _, frames = _frames(fixture)
    validate_availability(
        frames, response_rate=fixture.inputs.response_rate, outage_probability=(0, 0, 0, 0)
    )


def test_fleet_size_rows_are_exact_sub_fleets():
    fixture = make_per_ev_fixture(world_count=6, vehicle_count=320, seed=4, night_count=2)
    totals = fixture_totals(fixture)
    frame = availability._fleet_size_frame(totals)
    assert list(frame["fleet_size"].unique()) == [100, 300, 320]
    # The n = 100 row is the first 100 EVs in units order: recompute it from
    # a run of those EVs alone (EVs are physically independent).
    first = slice(0, 100)
    sub = {key: value[..., first] for key, value in fixture.unit(np.arange(6)).items()}
    flex = dataclasses.replace(
        fixture.flex,
        hours_to_departure=fixture.flex.hours_to_departure[:, first],
        target_kwh=fixture.flex.target_kwh[first],
        power_kw=fixture.flex.power_kw[first],
    )
    inputs = dataclasses.replace(
        fixture.inputs, manufacturer_index=fixture.inputs.manufacturer_index[first]
    )
    sub_totals = availability.new_totals(inputs, fixture.study_slots, flex, 6)
    availability.accumulate(sub_totals, np.arange(6), sub)
    sub_frame = availability._fleet_size_frame(sub_totals)
    left = frame[frame["fleet_size"] == 100].reset_index(drop=True)
    right = sub_frame[sub_frame["fleet_size"] == 100].reset_index(drop=True)
    for column in ("mean_kw", "p10_kw", "p50_kw", "sd_kw", "implied_rho", "n_eff"):
        np.testing.assert_allclose(left[column], right[column], atol=1e-9, equal_nan=True)


# --------------------------------------------------------------------------
# Intraday distribution (§10.2d)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outage_probability",
    [(0.5, 0.02, 0.1, 0.0), (0.02, 0.02, 0.02, 0.02)],
    ids=["one_maker_half_out", "defaults"],
)
def test_intraday_distribution_matches_monte_carlo(outage_probability):
    rng = np.random.default_rng(20260929)
    ev_count, late_worlds, draws = 300, 30, 20_000
    maker = np.arange(ev_count) % 4
    known = np.where(rng.random(ev_count) < 0.3, 0.0, rng.uniform(0.5, 7.4, ev_count))
    onehot = np.eye(4)[maker]
    s, q = known @ onehot, known**2 @ onehot
    late = rng.gamma(4.0, 20.0, (late_worlds, 4))
    r = np.array([0.9, 0.95, 0.8, 1.0])
    pi = np.asarray(outage_probability)
    out = availability.intraday_distribution(s, q, r, pi, late, availability.INTRADAY_LEVELS)

    out_draw = rng.random((draws, 4)) < pi
    respond = rng.random((draws, ev_count)) < r[maker]
    k = ((~out_draw[:, maker]) & respond) @ known
    l_draw = ((~out_draw) * late[rng.integers(0, late_worlds, draws)]).sum(axis=1)
    x_max = s.sum() + late.sum(axis=1).max()
    tolerance = x_max / availability.INTRADAY_GRID_BINS + 0.02 * x_max
    levels = availability.INTRADAY_LEVELS
    np.testing.assert_allclose(out["quantiles"], np.quantile(k + l_draw, levels), atol=tolerance)
    np.testing.assert_allclose(out["known_quantiles"], np.quantile(k, levels), atol=tolerance)
    np.testing.assert_allclose(out["late_quantiles"], np.quantile(l_draw, levels), atol=tolerance)
    # Means are exact: E[K] = Σ (1 - π_m) r_m S_m, E[L] = Σ (1 - π_m) mean L_m.
    assert out["known_mean"] == pytest.approx(((1 - pi) * r) @ s)
    assert out["late_mean"] == pytest.approx(late.mean(axis=0) @ (1 - pi))
    assert out["mean"] == pytest.approx(k.mean() + l_draw.mean(), rel=0.01)


def test_intraday_distribution_degenerate_cases():
    # Nothing plugged in and every late week empty: all quantiles exactly 0.
    zero = availability.intraday_distribution(
        np.zeros((2, 4)), np.zeros((2, 4)), np.full(4, 0.9), np.full(4, 0.1), np.zeros((2, 5, 4))
    )
    assert (zero["quantiles"] == 0).all() and (zero["mean"] == 0).all()
    # No other week to read the late part from: NaN, not zero.
    empty = availability.intraday_distribution(
        np.ones(4), np.ones(4), np.full(4, 0.9), np.full(4, 0.1), np.zeros((0, 4))
    )
    assert np.isnan(empty["quantiles"]).all() and np.isnan(empty["mean"])
    # Everyone responds and no maker can fail: K is a point mass at Σ S.
    point = availability.intraday_distribution(
        np.array([10.0, 20.0, 0.0, 5.0]),
        np.array([10.0, 50.0, 0.0, 5.0]),
        np.ones(4),
        np.zeros(4),
        np.zeros((3, 4)),
    )
    np.testing.assert_allclose(point["known_quantiles"], 35.0, atol=35.0 / 512)


def test_late_outage_slice_is_restored_at_its_potential():
    # Maker 1 (index 0) is out every night: its late sessions that would have
    # followed carry code 2 and count at ā in late_restored_kw_m1, never 0 (Q2).
    fixture = make_per_ev_fixture(
        world_count=6, vehicle_count=80, seed=13, outage_probability=(1.0, 0.0, 0.0, 0.0)
    )
    totals = fixture_totals(fixture)
    unit = fixture.unit(np.arange(6))
    decision = availability.decision_slots(fixture.study_slots, "17:00")[
        fixture.study_slots["night_index"].to_numpy()
    ]
    late_ev = ~unit["connected"][:, decision]
    maker0 = fixture.inputs.manufacturer_index == 0
    per_ev = availability.ev_contributions(
        unit,
        power_kw=fixture.flex.power_kw,
        target_kwh=fixture.flex.target_kwh,
        efficiency=fixture.flex.efficiency,
        expected_end_slot=availability.expected_end_slots(fixture.flex.hours_to_departure),
        night_index=fixture.study_slots["night_index"].to_numpy(),
        night_decision_slots=availability.decision_slots(fixture.study_slots, "17:00"),
        blackout=fixture.inputs.blackout,
        duration_slots=2,
    )
    potential = per_ev["potential"]["turn_down"]
    outage_code = unit["plan_status"] == availability.MAKER_OUTAGE
    expected = (potential * (late_ev & outage_code & maker0)).sum(axis=2)
    cell = availability.CELLS.index(("turn_down", 1.0))
    np.testing.assert_allclose(totals["late_restored"][..., cell, 0], expected, atol=1e-9)
    assert np.nansum(expected) > 0, (
        "the fixture must have late out-of-control sessions with potential"
    )
    realised_maker0 = (per_ev["realised"]["turn_down"] * maker0).sum(axis=2)
    assert np.nansum(realised_maker0) == 0

    # π_1 = 1: tonight's outage is certain, so the late quantiles and mean
    # leave maker 1's restored slice out in every world.
    frames = availability.build_frames(totals, fixture.study_slots)
    rows = frames["availability_world_slot"]
    pick = rows[(rows["direction"] == "turn_down") & (rows["duration_hours"] == 1.0)]
    others = pick[["late_restored_kw_m2", "late_restored_kw_m3", "late_restored_kw_m4"]].sum(axis=1)
    others = others.to_numpy().reshape(6, -1)
    mean = pick["intraday_late_mean_kw"].to_numpy().reshape(6, -1)
    for world in range(6):
        expected_mean = np.delete(others, world, axis=0).mean(axis=0)
        ok = ~np.isnan(mean[world])
        np.testing.assert_allclose(mean[world][ok], expected_mean[ok], atol=1e-9)
    # ... and the late quantiles are the other worlds' quantiles of the online
    # makers' sum alone, within one bin of the per-slot grid.
    restored = pick[[f"late_restored_kw_m{k}" for k in range(1, 5)]].sum(axis=1)
    restored = restored.to_numpy().reshape(6, -1)
    potential = pick["intraday_potential_kw"].to_numpy().reshape(6, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slots past the end
        bin_kw = (np.nanmax(potential, axis=0) + np.nanmax(restored, axis=0)) / 512
    for name, level in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        reported = pick[f"intraday_late_{name}_kw"].to_numpy().reshape(6, -1)
        for world in range(6):
            expected = np.quantile(
                np.delete(others, world, axis=0), level, axis=0, method="inverted_cdf"
            )
            ok = ~np.isnan(reported[world])
            assert (np.abs(reported[world] - expected)[ok] <= bin_kw[ok] + 1e-9).all(), name
    validate_availability(
        types.SimpleNamespace(**frames),
        response_rate=fixture.inputs.response_rate,
        outage_probability=fixture.inputs.outage_probability,
        per_ev_realised=per_ev_realised(fixture),
    )


def test_known_part_unchanged_by_post_decision_replans():
    # R1 leakage: a replan after 17:00 (a short-notice event) changes the plan
    # in force, the remaining need, the import and the status of plugged-in
    # EVs, but not the snapshot at s_n, so every K column stays the same.
    fixture = make_per_ev_fixture(world_count=4, vehicle_count=40, seed=21)
    worlds = np.arange(4)
    unit = fixture.unit(worlds)
    changed = {key: value.copy() for key, value in unit.items()}
    night = fixture.study_slots["night_index"].to_numpy()
    decision = availability.decision_slots(fixture.study_slots, "17:00")
    after = np.arange(336) > decision[night]
    replanned = (
        unit["connected"][:, decision[night]] & after[np.newaxis, :, np.newaxis] & unit["connected"]
    )
    changed["planned_home_import_kwh"][replanned] = np.roll(
        unit["planned_home_import_kwh"], 2, axis=1
    )[replanned]
    changed["home_grid_import_kwh"][replanned] = changed["planned_home_import_kwh"][replanned]
    changed["plan_remaining_need_kwh"][replanned] *= 0.5
    changed["plan_status"][replanned] = physics.CONTROL_OUTAGE_EVENT

    def run(per_ev):
        totals = availability.new_totals(fixture.inputs, fixture.study_slots, fixture.flex, 4)
        availability.accumulate(totals, worlds, per_ev)
        return totals, availability.build_frames(totals, fixture.study_slots)[
            "availability_world_slot"
        ]

    before_totals, before = run(unit)
    after_totals, after_frame = run(changed)
    assert replanned.any()
    for name in ("known_sum", "known_sum_sq", "intraday_eligible_count"):
        np.testing.assert_array_equal(before_totals[name], after_totals[name])
    known_columns = [
        c for c in before.columns if c.startswith("intraday_known") or c == "intraday_potential_kw"
    ]
    for column in known_columns:
        np.testing.assert_array_equal(before[column].to_numpy(), after_frame[column].to_numpy())
    assert not np.array_equal(
        before["deliverable_kw"].to_numpy(), after_frame["deliverable_kw"].to_numpy()
    )


# --------------------------------------------------------------------------
# Shared factors: variance, implied correlation and firm share (§10.2c)
# --------------------------------------------------------------------------


def _skip_moments(median: float, sigma: float) -> tuple[float, float]:
    """E[q] and Var(q) of q = expit(logit(median) + σ z) by a 200,001-point quadrature (§10.1b)."""

    z = np.linspace(-12.0, 12.0, 200_001)
    weight = stats.norm.pdf(z) * (z[1] - z[0])
    q = special.expit(special.logit(median) + sigma * z)
    mean = float((q * weight).sum())
    return mean, float(((q - mean) ** 2 * weight).sum())


def _controlled(world_count, vehicle_count, *, sigma_day, response=1.0, seed=17):
    """§10.2c controlled fixture: one p = 1 cohort, fixed clocks, flat prices, π = 0, no week."""

    fixture = make_per_ev_fixture(
        world_count=world_count,
        vehicle_count=vehicle_count,
        seed=seed,
        night_count=1,
        sigma_day=sigma_day,
        sigma_week=0.0,
        response_rate=(response,) * 4,
        outage_probability=(0.0,) * 4,
        clock_spread_slots=0,
        flat_prices=True,
    )
    return fixture_totals(fixture, chunk_size=50)


PLUG_IN_SLOT = 12  # 18:00 on night 0: every accepted session plugs in here
CELL = availability.CELLS.index(("turn_down", 0.5))


@pytest.mark.parametrize("response", [1.0, 0.8])
def test_deliverable_sd_matches_the_shared_factor_formula(response):
    world_count, vehicle_count, kw = 400, 200, 7.4
    totals = _controlled(world_count, vehicle_count, sigma_day=1.0, response=response)
    diagnostics = availability.realised_diagnostics(totals)
    mean_q, var_q = _skip_moments(0.12, 1.0)
    p_bar = (1.0 - mean_q) * response
    var_p = var_q * response**2
    # sd = kW · √(N p̄(1 - p̄) + (N² - N) Var(p_eff)), "kW x response" exact
    # because response is a second independent Bernoulli per EV.
    expected = kw * np.sqrt(
        vehicle_count * p_bar * (1 - p_bar) + (vehicle_count**2 - vehicle_count) * var_p
    )
    sd = diagnostics["sd"][PLUG_IN_SLOT, CELL]
    standard_error = sd / np.sqrt(2 * (world_count - 1))
    assert abs(sd - expected) <= 3 * standard_error, (sd, expected, standard_error)
    # ρ = Var(p_eff) / (p̄(1 - p̄)) with p̄ and Var(p_eff) already carrying r,
    # i.e. r² Var(q) / (p̄r(1 - p̄r)) in the review's notation.
    rho = diagnostics["implied_rho"][PLUG_IN_SLOT, CELL]
    expected_rho = var_p / (p_bar * (1 - p_bar))
    # ρ̂ ≈ (Var A - Σv) / (N² v): its error is the fleet variance's (relative
    # SE √(2/(W-1))) scaled by Var A / ((Σ√v)² - Σv).
    ev_var = kw**2 * p_bar * (1 - p_bar)
    rho_error = (
        np.sqrt(2 / (world_count - 1)) * sd**2 / ((vehicle_count**2 - vehicle_count) * ev_var)
    )
    assert abs(rho - expected_rho) <= 3 * rho_error, (rho, expected_rho, rho_error)
    assert diagnostics["contributing_ev_count"][PLUG_IN_SLOT, CELL] == vehicle_count


def _firm_share(totals) -> float:
    values = totals["deliverable"][:, PLUG_IN_SLOT, CELL]
    p10, p50 = np.quantile(values, (0.1, 0.5))
    return p10 / p50


def test_firm_share_approaches_one_without_shared_factors_and_levels_off_with_them():
    independent = {n: _controlled(400, n, sigma_day=0.0) for n in (50, 200, 800)}
    gaps = {n: (1 - _firm_share(totals)) * np.sqrt(n) for n, totals in independent.items()}
    for n in (200, 800):
        assert gaps[50] / 2 <= gaps[n] <= gaps[50] * 2, gaps
    # Independent EVs: ρ is 0 within the noise of the fleet variance, whose
    # relative SE √(2/(W - 1)) becomes ~√(2/(W - 1)) / (N - 1) on ρ.
    rho = availability.realised_diagnostics(independent[800])["implied_rho"][PLUG_IN_SLOT, CELL]
    assert abs(rho) <= 3 * np.sqrt(2 / 399) / 799, rho
    shared = {n: 1 - _firm_share(_controlled(400, n, sigma_day=1.0)) for n in (200, 800)}
    assert shared[800] >= shared[200] / 2, shared
