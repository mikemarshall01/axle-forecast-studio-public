"""The shared plug-in skip share and the holiday switches (trading contract v1 §10.1b, §10.1c).

Every EV in a world shares one skip share per night, so plug-ins are
positively correlated and the fleet's plug-in count has an N² variance term
that does not diversify away.  These tests check the sampler against that
algebra, the fixed draw counts (common random numbers) and the draw order
of §10.0.  All fixtures are synthetic.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from axle_studio.model import assumptions, forecast
from axle_studio.model.clock import holiday_flags
from axle_studio.model.forecast import simulate_forecast
from axle_studio.model.sampling import (
    plug_in_skip_share,
    sample_connection_opportunities,
    sample_departure_times,
    sample_plug_in_factors,
)
from axle_studio.model.settings import RunSettings

_CLIP = 180.0
_T_DF = 5.0


def _settings(**changes) -> RunSettings:
    values = dict(
        start_local_date=date(2026, 1, 12),  # a Monday
        warmup_days=1,
        study_days=3,
        vehicle_count=40,
        seed=17,
        evaluation_world_count=4,
        opening_soc_fraction=0.55,
        reserve_soc_fraction=0.2,
        home_charge_efficiency=0.92,
        weather_efficiency_sensitivity_fraction_per_c=0.01,
    )
    values.update(changes)
    return RunSettings(**values)


def _units(vehicle_count: int, plug_probability: float, *, always: int = 0) -> pd.DataFrame:
    """A hand-built synthetic fleet: one clocked cohort, plus ``always`` always-plugged EVs."""

    cohorts = ["clocked"] * (vehicle_count - always) + ["always_plugged_in"] * always
    return pd.DataFrame(
        {
            "cohort_id": cohorts,
            "plug_probability": plug_probability,
            "runtime_departure_local_hour": 8,
            "runtime_arrival_local_hour": 18,
        }
    )


def _sessions(settings: RunSettings, units: pd.DataFrame, skip_share, seed: int = 3):
    rng = np.random.default_rng(seed)
    scales = {"clocked": 60.0, "always_plugged_in": 0.0}
    departures = sample_departure_times(
        settings, units, rng, scale_minutes_by_cohort=scales, t_df=_T_DF, clip_minutes=_CLIP
    )
    accepted, start, end = sample_connection_opportunities(
        settings,
        units,
        rng,
        departure_utc=departures,
        # Every EV drives every day, so each night is its own session and the
        # skip share alone decides acceptance (item 68's multi-day sessions
        # are tested in test_sampling_connections.py).
        drives_today=np.ones(
            (settings.evaluation_world_count, settings.sampled_day_count, len(units)), dtype=bool
        ),
        plug_in_scale_minutes_by_cohort=scales,
        t_df=_T_DF,
        clip_minutes=_CLIP,
        skip_share=skip_share,
    )
    return accepted, start, end, rng.random()


def _skip_moments(median: float, sd: float) -> tuple[float, float]:
    """E[q] and Var(q) of q = expit(logit(median) + sd·z), z ~ N(0, 1), by quadrature.

    A 200,001-point trapezoid rule over z in [-10, 10] (§10.1b's test
    recipe), independent of the sampler's code path.
    """

    z = np.linspace(-10.0, 10.0, 200_001)
    weight = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    q = expit(logit(median) + sd * z)
    mean = np.trapezoid(q * weight, z)
    second = np.trapezoid(q**2 * weight, z)
    return float(mean), float(second - mean**2)


def _fourth_central_moment(median: float, sd: float) -> float:
    """E[(q − E[q])⁴] for the same logit-normal, by the same quadrature."""

    z = np.linspace(-10.0, 10.0, 200_001)
    weight = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    q = expit(logit(median) + sd * z)
    mean = np.trapezoid(q * weight, z)
    return float(np.trapezoid((q - mean) ** 4 * weight, z))


def _variance_standard_error(
    shared_scale: float, binomial_variance: float, median: float, sd: float, worlds: int
) -> float:
    """Standard error of a sample variance of ``shared_scale × q + B`` over ``worlds``.

    ``B`` is the (near-normal) binomial noise with variance
    ``binomial_variance``.  The logit-normal skip share has a heavy right
    tail, so the SE uses its true fourth moment rather than the sample's
    own, which a few hundred worlds understate:
    Var(s²) ≈ (μ₄ − σ⁴) / W with μ₄ = a⁴μ₄(q) + 6 a² Var(q) v_B + 3 v_B².
    """

    _, var_q = _skip_moments(median, sd)
    shared = shared_scale**2 * var_q
    fourth = (
        shared_scale**4 * _fourth_central_moment(median, sd)
        + 6.0 * shared * binomial_variance
        + 3.0 * binomial_variance**2
    )
    total = shared + binomial_variance
    return float(np.sqrt((fourth - total**2) / worlds))


# --- The skip share in the connections sampler ---------------------------------


def test_no_skip_share_reproduces_zero_skip_bit_for_bit() -> None:
    settings = _settings(evaluation_world_count=5)
    units = _units(40, 0.7, always=4)
    days = settings.sampled_day_count
    without = _sessions(settings, units, None)
    zero = _sessions(settings, units, np.zeros((5, days)))

    for left, right in zip(without[:3], zero[:3], strict=True):
        assert pd.Series(left.ravel()).equals(pd.Series(right.ravel()))
    # Same generator position afterwards: the skip share draws nothing.
    assert without[3] == zero[3]


def test_skip_share_only_turns_sessions_off_on_the_same_uniforms() -> None:
    settings = _settings(evaluation_world_count=5)
    units = _units(40, 1.0)
    days = settings.sampled_day_count
    base, _, _, after_base = _sessions(settings, units, np.zeros((5, days)))
    skipped, _, _, after_skip = _sessions(settings, units, np.full((5, days), 0.4))

    assert not (skipped & ~base).any()
    assert skipped.sum() < base.sum()
    assert after_base == after_skip


def test_always_plugged_cohort_is_never_skipped() -> None:
    settings = _settings(evaluation_world_count=3, vehicle_count=10)
    units = _units(10, 1.0, always=4)
    accepted, start, _, _ = _sessions(settings, units, np.ones((3, settings.sampled_day_count)))

    always = units["cohort_id"].eq("always_plugged_in").to_numpy()
    assert accepted[:, 0, always].all()
    assert not pd.isna(start[:, 0, always]).any()
    assert not accepted[:, :, ~always].any()


def test_skip_share_is_checked() -> None:
    settings = _settings(vehicle_count=10)
    units = _units(10, 1.0)
    with pytest.raises(ValueError, match="skip_share"):
        _sessions(settings, units, np.full((4, settings.sampled_day_count), 1.5))
    with pytest.raises(ValueError, match="skip_share"):
        _sessions(settings, units, np.zeros((4, 1)))


# --- The factors: draw count and the skip-share formula --------------------------


@pytest.mark.parametrize(("worlds", "days"), [(1, 1), (4, 9), (30, 3)])
def test_factors_take_w_plus_w_times_days_normals_whatever_the_values(
    worlds: int, days: int
) -> None:
    rng = np.random.default_rng(5)
    week, day = sample_plug_in_factors(rng, worlds, days)
    reference = np.random.default_rng(5)
    expected_week = reference.standard_normal(worlds)
    expected_day = reference.standard_normal((worlds, days))

    np.testing.assert_array_equal(week, expected_week)
    np.testing.assert_array_equal(day, expected_day)
    assert rng.random() == reference.random()


def test_skip_share_formula_and_zero_median() -> None:
    week = np.array([0.0, 1.0, -2.0])
    day = np.array([[0.0, 1.0], [0.5, -1.0], [2.0, 0.0]])
    shift = np.array([0.0, 0.3])
    q = plug_in_skip_share(week, day, shift, median=0.12, day_sd=0.5, week_sd=0.17)

    expected = expit(logit(0.12) + 0.17 * week[:, None] + 0.5 * day + shift[None, :])
    np.testing.assert_allclose(q, expected, rtol=0.0, atol=1e-15)
    assert q[0, 0] == pytest.approx(0.12)
    assert (plug_in_skip_share(week, day, shift, median=0.0, day_sd=0.5, week_sd=0.17) == 0.0).all()


def test_default_arithmetic_matches_the_contract() -> None:
    # §10.1b and decision 0004 item 64: at σ_day = 0.5 the mean skip is about
    # 0.130 and the relative day-to-day SD of any skip-driven share about 6.5 %.
    mean, variance = _skip_moments(0.12, 0.5)

    assert mean == pytest.approx(0.130, abs=0.001)
    assert np.sqrt(variance) / (1.0 - mean) == pytest.approx(0.065, abs=0.001)
    assert expit(logit(0.12) + 0.5) == pytest.approx(0.184, abs=0.001)


# --- The variance of the fleet plug-in count (the N² term) ------------------------


@pytest.mark.parametrize("plug_probability", [1.0, 0.2])
def test_fleet_plug_in_count_variance_has_the_n_squared_term(plug_probability: float) -> None:
    # §10.1b: one p cohort, σ_day = 1.0 (a fixture value, not the default),
    # σ_week = 0, no holidays, 400 worlds.  The across-world variance of one
    # night's plug-in count is N p̄(1 − p̄) + (N² − N) Var(p_eff), with
    # p_eff = p (1 − q), p̄ = p (1 − E[q]) and Var(p_eff) = p² Var(q).
    worlds, vehicles, median, sd = 400, 200, 0.12, 1.0
    settings = _settings(
        warmup_days=0, study_days=1, vehicle_count=vehicles, evaluation_world_count=worlds
    )
    week, day = sample_plug_in_factors(
        np.random.default_rng(11), worlds, settings.sampled_day_count
    )
    q = plug_in_skip_share(
        week, day, np.zeros(settings.sampled_day_count), median=median, day_sd=sd, week_sd=0.0
    )
    accepted, _, _, _ = _sessions(settings, _units(vehicles, plug_probability), q, seed=12)
    count = accepted[:, 0, :].sum(axis=1).astype(float)

    mean_q, var_q = _skip_moments(median, sd)
    p_bar = plug_probability * (1.0 - mean_q)
    var_p_eff = plug_probability**2 * var_q
    expected = vehicles * p_bar * (1.0 - p_bar) + (vehicles**2 - vehicles) * var_p_eff
    # E[N p_eff (1 − p_eff)], the binomial part of the variance.
    binomial = vehicles * (p_bar - plug_probability**2 * (var_q + (1.0 - mean_q) ** 2))
    standard_error = _variance_standard_error(
        vehicles * plug_probability, binomial, median, sd, worlds
    )
    assert abs(count.var(ddof=1) - expected) < 3.0 * standard_error
    # The shared term shows: independent plug-ins would give only N p̄(1 − p̄).
    assert count.var(ddof=1) > 1.5 * vehicles * p_bar * (1.0 - p_bar)
    assert count.mean() == pytest.approx(vehicles * p_bar, rel=0.05)


def test_week_factor_moves_every_night_of_a_world_together() -> None:
    # σ_day = 0, σ_week = 1.0 (fixture values): every night of a world shares
    # one skip share, so the across-night mean count carries the N² term and
    # the within-world night-to-night differences do not.
    worlds, vehicles, nights, median, sd = 400, 200, 7, 0.12, 1.0
    settings = _settings(
        warmup_days=0, study_days=nights, vehicle_count=vehicles, evaluation_world_count=worlds
    )
    week, day = sample_plug_in_factors(
        np.random.default_rng(21), worlds, settings.sampled_day_count
    )
    q = plug_in_skip_share(
        week, day, np.zeros(settings.sampled_day_count), median=median, day_sd=0.0, week_sd=sd
    )
    assert (q == q[:, :1]).all()
    accepted, _, _, _ = _sessions(settings, _units(vehicles, 1.0), q, seed=22)
    counts = accepted[:, :nights, :].sum(axis=2).astype(float)

    mean_q, var_q = _skip_moments(median, sd)
    # E[N p_eff (1 − p_eff)] with p = 1: N E[(1 − q) q] = N (E[q] − E[q²]).
    binomial = vehicles * (mean_q - (var_q + mean_q**2))
    week_mean = counts.mean(axis=1)
    expected_mean_var = vehicles**2 * var_q + binomial / nights
    standard_error = _variance_standard_error(vehicles, binomial / nights, median, sd, worlds)
    assert abs(week_mean.var(ddof=1) - expected_mean_var) < 3.0 * standard_error
    differences = np.diff(counts, axis=1).ravel()
    assert differences.mean() == pytest.approx(0.0, abs=0.5)
    squared = (differences - differences.mean()) ** 2
    assert abs(differences.var(ddof=1) - 2.0 * binomial) < 3.0 * squared.std(ddof=1) / np.sqrt(
        differences.size / (nights - 1)
    )
    assert differences.var(ddof=1) < 0.15 * week_mean.var(ddof=1)


# --- Holidays (§10.1c) --------------------------------------------------------------


def test_holiday_flags_off_are_all_zero() -> None:
    flags = holiday_flags(_settings(), bank_holiday_monday=False, half_term_week=False)
    assert not flags["holiday"].any()
    assert (flags["skip_logit_shift"] == 0.0).all()
    assert flags["skip_logit_shift"].shape == (_settings().sampled_day_count,)


def test_bank_holiday_marks_the_first_monday_study_evening_only() -> None:
    # Warm-up 2 days from Monday 12 January 2026: dates Sat, Sun, Mon (the
    # first study evening), Tue, Wed and the Thursday morning after.
    settings = _settings(warmup_days=2)
    flags = holiday_flags(settings, bank_holiday_monday=True, half_term_week=False)

    expected = np.zeros(settings.sampled_day_count)
    expected[2] = assumptions.HOLIDAY_SKIP_LOGIT_SHIFT
    np.testing.assert_array_equal(flags["skip_logit_shift"], expected)
    np.testing.assert_array_equal(flags["holiday"], expected > 0.0)
    assert assumptions.HOLIDAY_SKIP_LOGIT_SHIFT == 0.30


def test_bank_holiday_has_no_effect_without_a_monday_evening() -> None:
    # Tuesday to Thursday evenings, with a Monday warm-up date that is not a study evening.
    settings = _settings(start_local_date=date(2026, 1, 13), warmup_days=1, study_days=3)
    flags = holiday_flags(settings, bank_holiday_monday=True, half_term_week=False)
    assert not flags["holiday"].any()


def test_half_term_marks_every_study_evening_and_no_warm_up_date() -> None:
    settings = _settings(warmup_days=2, study_days=3)
    flags = holiday_flags(settings, bank_holiday_monday=False, half_term_week=True)

    expected = np.array([0.0, 0.0, 0.3, 0.3, 0.3, 0.0])
    np.testing.assert_array_equal(flags["skip_logit_shift"], expected)
    # Both switches on: the Monday is one holiday, shifted once.
    both = holiday_flags(settings, bank_holiday_monday=True, half_term_week=True)
    np.testing.assert_array_equal(both["skip_logit_shift"], expected)


def _run(**factor_edits):
    inputs = assumptions.forecast_inputs(None, warmup_days=1, study_days=3)
    inputs["shared_factor_assumptions"] = {**inputs["shared_factor_assumptions"], **factor_edits}
    return simulate_forecast(_settings(), assumptions.cohort_fixture(), model="no_action", **inputs)


def test_holiday_switch_changes_only_that_evenings_skip_share() -> None:
    base = _run()
    edited = _run(**{"behaviour.holiday_bank_holiday_monday": 1})
    monday = 1  # warm-up date 0 is Sunday 11 January; date 1 is Monday's study evening

    shift = edited.plug_in_factors["skip_logit_shift"] - base.plug_in_factors["skip_logit_shift"]
    np.testing.assert_array_equal(shift, np.eye(1, shift.size, monday).ravel() * 0.30)
    for key in ("week", "day"):
        np.testing.assert_array_equal(edited.plug_in_factors[key], base.plug_in_factors[key])
    changed = edited.plug_in_factors["skip_share"] != base.plug_in_factors["skip_share"]
    assert changed[:, monday].all() and not np.delete(changed, monday, axis=1).any()

    base_inputs, edited_inputs = base.evaluation.inputs, edited.evaluation.inputs
    for key in ("drives_today", "outbound_miles", "trip_departure_utc"):
        assert pd.Series(base_inputs[key].ravel()).equals(pd.Series(edited_inputs[key].ravel()))
    np.testing.assert_array_equal(
        base.evaluation.daily_temperature_c, edited.evaluation.daily_temperature_c
    )
    base_accepted = base_inputs["connection_session_accepted"]
    edited_accepted = edited_inputs["connection_session_accepted"]
    # A higher skip share only turns that evening's sessions off, and the
    # evenings before it are untouched.  With multi-day sessions (decision
    # 0004 item 68) an EV that skipped Monday and does not drive on Tuesday
    # was not plugged in from Monday, so its Tuesday plug draw now starts a
    # session: later evenings gain sessions only for EVs Monday turned off.
    # Every session in both runs keeps its clocks (its end depends only on
    # the unchanged drive flags after it).
    turned_off = base_accepted[:, monday] & ~edited_accepted[:, monday]
    assert not (edited_accepted & ~base_accepted)[:, : monday + 1].any()
    assert (edited_accepted == base_accepted)[:, :monday].all()
    gained = (edited_accepted & ~base_accepted)[:, monday + 1 :].any(axis=1)
    assert not (gained & ~turned_off).any()
    kept = edited_accepted & base_accepted
    for key in ("connection_start_utc", "connection_end_utc"):
        np.testing.assert_array_equal(edited_inputs[key][kept], base_inputs[key][kept])


def test_switches_off_reproduce_the_default_run() -> None:
    base = _run()
    explicit = _run(
        **{"behaviour.holiday_bank_holiday_monday": 0, "behaviour.holiday_half_term_week": 0}
    )
    pd.testing.assert_frame_equal(base.fleet_world_intervals, explicit.fleet_world_intervals)


# --- Draw order (§10.0) ---------------------------------------------------------------


_TRACKED = (
    "build_population",
    "assign_manufacturers",
    "sample_departure_times",
    "sample_daily_trip_inputs",
    "sample_plug_in_factors",
    "sample_connection_opportunities",
    "sample_daily_temperature",
    "generate_market_price_paths",
    "sample_non_response",
    "sample_manufacturer_outages",
)


def _generator(args, kwargs) -> np.random.Generator:
    return next(
        value for value in (*args, *kwargs.values()) if isinstance(value, np.random.Generator)
    )


def test_draw_order_follows_the_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each sampler is wrapped to record the generator state before and after
    # it, so the test sees the exact order and that nothing draws in between
    # (apart from the control-group permutation, the one inline draw).
    calls: list[tuple[str, dict, dict, np.random.Generator]] = []

    def tracked(name, function):
        def wrapper(*args, **kwargs):
            rng = _generator(args, kwargs)
            before = rng.bit_generator.state
            result = function(*args, **kwargs)
            calls.append((name, before, rng.bit_generator.state, rng))
            return result

        return wrapper

    for name in _TRACKED:
        monkeypatch.setattr(forecast, name, tracked(name, getattr(forecast, name)))
    inputs = assumptions.forecast_inputs(None, warmup_days=1, study_days=3)
    simulate_forecast(_settings(), assumptions.cohort_fixture(), model="action", **inputs)

    assert [call[0] for call in calls] == list(_TRACKED)
    for (name, _, after, _), (next_name, before, _, _) in zip(calls, calls[1:], strict=False):
        if next_name == "assign_manufacturers":
            # The control-group permutation (§9.5) sits between them.
            replay = np.random.default_rng()
            replay.bit_generator.state = after
            replay.permutation(_settings().vehicle_count)
            assert replay.bit_generator.state == before
        else:
            assert after == before, f"a draw between {name} and {next_name}"
    # The outage uniforms close the run: the generator does not move afterwards.
    _, _, after_last, rng = calls[-1]
    assert rng.bit_generator.state == after_last
