from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from axle_studio.model.clock import utc_half_hour_boundaries
from axle_studio.model.physics import PublicTopUp, simulate_fleet_intervals
from axle_studio.model.sampling import sample_connection_opportunities, sample_departure_times
from axle_studio.model.settings import RunSettings

# The decision 0004 item 32/37 defaults: 90 % public efficiency, top up below
# 10 % SoC to 80 % SoC (re-baselined from 20 % by item 37).
_TOP_UP = PublicTopUp(efficiency_fraction=0.9, threshold_soc_fraction=0.1, target_soc_fraction=0.8)


def _utc(value: object) -> pd.Timestamp:
    """A sampled instant (naive ``datetime64[ns]`` in UTC) as an aware Timestamp."""

    return pd.Timestamp(value, tz="UTC")


def _settings(**changes: object) -> RunSettings:
    base = RunSettings(
        start_local_date=date(2025, 1, 6),
        warmup_days=1,
        study_days=2,
        vehicle_count=2,
        seed=1,
        evaluation_world_count=4,
        opening_soc_fraction=0.5,
        reserve_soc_fraction=0.1,
        home_charge_efficiency=1.0,
        weather_efficiency_sensitivity_fraction_per_c=0.01,
    )
    return replace(base, **changes)


def _units(
    *, cohort_ids: tuple[str, ...] = ("ordinary", "scheduled"), probability: float = 1.0
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population_id": ["synthetic-pop"] * 2,
            "unit_id": ["ev-1", "ev-2"],
            "cohort_id": list(cohort_ids),
            "plug_probability": [probability] * 2,
            "runtime_arrival_local_hour": [18, 22],
            "runtime_departure_local_hour": [7, 9],
            "physical_capacity_kwh": [10.0, 10.0],
            "efficiency_miles_per_battery_kwh": [4.0, 4.0],
            "home_charger_limit_kw": [2.0, 2.0],
            "vehicle_ac_limit_kw": [2.0, 2.0],
            "preferred_target_soc_fraction": [0.8, 0.8],
        }
    )


def _departures(
    settings: RunSettings, units: pd.DataFrame, scale: float = 0.0, seed: int = 99
) -> np.ndarray:
    """Home departures for the sessions to end at (a fixed clock when ``scale`` is 0)."""

    return sample_departure_times(
        settings,
        units,
        np.random.default_rng(seed),
        scale_minutes_by_cohort={cohort: scale for cohort in units.cohort_id.unique()},
        t_df=4.0,
        clip_minutes=60.0,
    )


def _connect(
    settings: RunSettings,
    units: pd.DataFrame,
    rng: np.random.Generator,
    scale_by_cohort: dict[str, float],
    *,
    clip: float = 60.0,
    t_df: float = 4.0,
    departures: np.ndarray | None = None,
    weekend_scale_by_cohort: dict[str, float] | None = None,
    drives: np.ndarray | None = None,
):
    # Default: every EV drives every day, so every session ends at the next
    # morning's departure (the item 51 rule these clock tests check); the
    # multi-day tests (item 68) pass their own drive flags.
    if drives is None:
        drives = np.ones(
            (settings.evaluation_world_count, settings.sampled_day_count, len(units)), dtype=bool
        )
    return sample_connection_opportunities(
        settings,
        units,
        rng,
        departure_utc=_departures(settings, units) if departures is None else departures,
        drives_today=drives,
        plug_in_scale_minutes_by_cohort=scale_by_cohort,
        weekend_plug_in_scale_minutes_by_cohort=weekend_scale_by_cohort,
        t_df=t_df,
        clip_minutes=clip,
    )


def _sample(units: pd.DataFrame, settings: RunSettings | None = None, seed: int = 7):
    settings = settings or _settings()
    scales = {cohort: 30.0 for cohort in units.cohort_id.unique()}
    return _connect(settings, units, np.random.default_rng(seed), scales)


def test_shapes_dtypes_repeatability_and_no_input_mutation() -> None:
    units = _units()
    before = units.copy(deep=True)
    result = _sample(units)
    repeated = _sample(units)
    # warm-up 1 + study 2 + the morning after the last noon-to-noon night
    # (RunSettings.sampled_day_count, decision 0004 item 52) = 4 dates.
    assert [value.shape for value in result] == [(4, 4, 2)] * 3
    assert result[0].dtype == bool
    assert result[1].dtype == np.dtype("datetime64[ns]")
    assert result[2].dtype == np.dtype("datetime64[ns]")
    assert np.array_equal(result[0], repeated[0])
    assert np.array_equal(result[1], repeated[1], equal_nan=True)
    assert np.array_equal(result[2], repeated[2], equal_nan=True)
    pd.testing.assert_frame_equal(units, before)


@pytest.mark.parametrize("probability, expected", [(0.0, False), (1.0, True)])
def test_plug_probability_is_respected(probability: float, expected: bool) -> None:
    accepted, starts, ends = _sample(_units(probability=probability))
    assert bool(accepted[:, :, 0].all()) is expected
    assert all(pd.isna(value) for value in starts[~accepted])
    assert all(pd.isna(value) for value in ends[~accepted])


def test_session_ends_at_the_next_dates_sampled_departure() -> None:
    # Decision 0004 item 51: plug-out is departure.  For an EV that drives
    # every day (the helper's default), session d (plug-in on date d) ends
    # exactly at the departure drawn for date d + 1, the same
    # instant a trip that day starts; the plug-in is its own half-hour draw.
    settings = _settings(evaluation_world_count=30)
    units = _units(cohort_ids=("ordinary", "ordinary"))
    departures = _departures(settings, units, scale=60.0)
    accepted, starts, ends = _connect(
        settings, units, np.random.default_rng(9), {"ordinary": 60.0}, departures=departures
    )
    assert accepted.all()
    np.testing.assert_array_equal(ends, departures[:, 1:])
    start_minutes = starts.astype("datetime64[m]").astype(np.int64)
    assert (start_minutes % 30 == 0).all()
    # Independent ends: session lengths vary, unlike the old shared shift.
    assert len(np.unique(ends - starts)) > 3


def test_supplied_scheduled_plug_in_scale_has_smaller_spread() -> None:
    settings = _settings(evaluation_world_count=4000, warmup_days=0, study_days=1)
    accepted, starts, _ = _connect(
        settings,
        _units(),
        np.random.default_rng(19),
        {"ordinary": 60.0, "scheduled": 20.0},
        clip=180.0,
    )
    ordinary = np.array(
        [
            pd.Timestamp(value).value / 60_000_000_000
            for value in starts[:, 0, 0]
            if value is not pd.NaT
        ]
    )
    scheduled = np.array(
        [
            pd.Timestamp(value).value / 60_000_000_000
            for value in starts[:, 0, 1]
            if value is not pd.NaT
        ]
    )
    assert accepted[:, 0, :].all()
    assert np.std(ordinary) > 2.0 * np.std(scheduled)


@pytest.mark.parametrize(
    ("start_date", "expected_start", "expected_end"),
    [
        (date(2025, 1, 6), "2025-01-06 18:00:00+00:00", "2025-01-07 07:00:00+00:00"),
        (date(2025, 7, 7), "2025-07-07 17:00:00+00:00", "2025-07-08 06:00:00+00:00"),
    ],
)
def test_london_connection_references_convert_to_utc(
    start_date: date, expected_start: str, expected_end: str
) -> None:
    settings = _settings(
        start_local_date=start_date, warmup_days=0, vehicle_count=1, evaluation_world_count=1
    )
    accepted, starts, ends = _connect(
        settings, _units().iloc[:1], np.random.default_rng(1), {"ordinary": 0.0}
    )
    assert accepted[0, 0, 0]
    assert _utc(starts[0, 0, 0]) == pd.Timestamp(expected_start)
    assert _utc(ends[0, 0, 0]) == pd.Timestamp(expected_end)


def test_no_trip_sampler_output_passes_current_explicit_kernel() -> None:
    settings = _settings(vehicle_count=1, evaluation_world_count=1, warmup_days=0, study_days=1)
    units = _units().iloc[:1].copy()
    accepted, starts, ends = _sample(units, settings)
    shape = accepted.shape
    fleet, *_ = simulate_fleet_intervals(
        settings,
        units,
        {
            "drives_today": np.zeros(shape, dtype=bool),
            "outbound_miles": np.zeros(shape),
            "return_miles": np.zeros(shape),
            "trip_departure_utc": np.full(shape, pd.NaT, dtype=object),
            "destination_dwell_seconds": np.zeros(shape),
            "drive_speed_mph": np.ones(shape),
            "connection_session_accepted": accepted,
            "connection_start_utc": starts,
            "connection_end_utc": ends,
            "desired_pre_drive_soc_fraction": np.zeros(shape),
        },
        public_top_up=_TOP_UP,
    )
    assert not fleet.empty


def test_minimal_connection_columns_are_sufficient() -> None:
    units = _units().loc[
        :,
        [
            "population_id",
            "unit_id",
            "cohort_id",
            "plug_probability",
            "runtime_arrival_local_hour",
            "runtime_departure_local_hour",
        ],
    ]
    accepted, starts, ends = _sample(units)
    assert accepted.shape == starts.shape == ends.shape == (4, 4, 2)


@pytest.mark.parametrize("clip", [330.0, 390.0])
def test_clock_clip_must_leave_adjacent_session_gap(clip: float) -> None:
    # 07:00 departure and 18:00 plug-in: 11 hours apart, so two 330-minute
    # shifts (11 hours together) would let the sessions touch.  The check is
    # on the clock hours, so it fails every time, not only in an unlucky world.
    settings = _settings(vehicle_count=1)
    units = _units().iloc[:1]
    with pytest.raises(ValueError, match="strictly positive gap"):
        _connect(
            settings,
            units,
            np.random.default_rng(1),
            {"ordinary": 60.0},
            clip=clip,
            departures=_departures(settings, units),
        )


def test_a_fixed_plug_in_needs_a_gap_only_beyond_the_departure_bound() -> None:
    # A plug-in with scale 0 never moves, so only the departure's own reach
    # (up to the clip) must stay before it: a 09:00 departure and a fixed
    # 13:00 plug-in fit a 180-minute clip (scripts/run_archetypes.py's
    # "home most of the day"), but not once the plug-in may move too.
    settings = _settings(vehicle_count=1)
    units = (
        _units().iloc[:1].assign(runtime_departure_local_hour=[9], runtime_arrival_local_hour=[13])
    )
    departures = sample_departure_times(
        settings,
        units,
        np.random.default_rng(2),
        scale_minutes_by_cohort={"ordinary": 60.0},
        t_df=4.0,
        clip_minutes=180.0,
    )
    rng = np.random.default_rng(1)
    accepted, starts, _ = _connect(
        settings, units, rng, {"ordinary": 0.0}, clip=180.0, departures=departures
    )
    assert accepted.all() and {_utc(value).hour for value in starts.ravel()} == {13}
    with pytest.raises(ValueError, match="strictly positive gap"):
        _connect(settings, units, rng, {"ordinary": 30.0}, clip=180.0, departures=departures)


def test_always_plugged_is_one_horizon_session() -> None:
    settings = _settings(evaluation_world_count=1, warmup_days=1, study_days=2, vehicle_count=1)
    units = _units(cohort_ids=("always_plugged_in", "always_plugged_in")).iloc[:1].copy()
    accepted, starts, ends = _sample(units, settings)
    assert accepted[0, :, 0].tolist() == [True, False, False, False]
    boundaries = utc_half_hour_boundaries(settings.start_local_date, 1, 2)
    # One session over the whole noon-to-noon horizon (decision 0004 item 52).
    assert _utc(starts[0, 0, 0]) == pd.Timestamp(boundaries[0])
    assert _utc(ends[0, 0, 0]) == pd.Timestamp(boundaries[-1])
    assert _utc(ends[0, 0, 0]) - _utc(starts[0, 0, 0]) == pd.Timedelta(days=3)


@pytest.mark.parametrize(
    "mapping, clip, t_df",
    [
        ({"ordinary": np.nan}, 60.0, 4.0),
        ({"ordinary": -1.0}, 60.0, 4.0),
        ({"ordinary": 1.0}, 15.0, 4.0),
        ({"ordinary": 1.0}, 60.0, 0.0),
        ({"ordinary": 1.0}, 60.0, np.inf),
    ],
)
def test_invalid_clock_configuration_rejected(
    mapping: dict[str, float], clip: float, t_df: float
) -> None:
    settings = _settings(vehicle_count=1)
    units = _units().iloc[:1]
    with pytest.raises((ValueError, TypeError)):
        _connect(
            settings,
            units,
            np.random.default_rng(1),
            mapping,
            clip=clip,
            t_df=t_df,
            departures=_departures(settings, units),
        )


def test_departures_must_cover_every_date_and_the_next_morning() -> None:
    settings = _settings(vehicle_count=1)
    units = _units().iloc[:1]
    with pytest.raises(ValueError, match="departure_utc"):
        _connect(
            settings,
            units,
            np.random.default_rng(1),
            {"ordinary": 0.0},
            departures=_departures(settings, units)[:, :-1],
        )


@pytest.mark.parametrize(
    ("start_local_date", "change_date"),
    [(date(2026, 3, 27), date(2026, 3, 29)), (date(2026, 10, 25), date(2026, 10, 25))],
)
def test_offset_change_uses_local_date_clock_and_fixed_utc_horizon(
    start_local_date: date, change_date: date
) -> None:
    settings = _settings(
        start_local_date=start_local_date,
        warmup_days=1,
        study_days=7,
        evaluation_world_count=1,
    )
    units = _units(cohort_ids=("ordinary", "always_plugged_in"))
    accepted, starts, ends = _connect(
        settings, units, np.random.default_rng(1), {"ordinary": 0.0, "always_plugged_in": 0.0}
    )
    boundaries = utc_half_hour_boundaries(start_local_date, 1, 7)
    first_local_date = start_local_date - timedelta(days=1)

    assert accepted[0, :, 0].all()
    # Nine sampled dates: warm-up, seven nights and the morning after (item 52).
    for day_index in range(settings.sampled_day_count):
        local_day = first_local_date + timedelta(days=day_index)
        start_local = _utc(starts[0, day_index, 0]).tz_convert("Europe/London")
        end_local = _utc(ends[0, day_index, 0]).tz_convert("Europe/London")
        assert (start_local.date(), start_local.hour, start_local.minute) == (local_day, 18, 0)
        assert (end_local.date(), end_local.hour, end_local.minute) == (
            local_day + timedelta(days=1),
            7,
            0,
        )
    change_index = (change_date - first_local_date).days
    before = _utc(starts[0, change_index - 1, 0])
    after = _utc(starts[0, change_index, 0])
    assert after - before != pd.Timedelta(days=1)

    assert accepted[0, :, 1].tolist() == [True] + [False] * 8
    assert _utc(starts[0, 0, 1]) == pd.Timestamp(boundaries[0])
    assert _utc(ends[0, 0, 1]) == pd.Timestamp(boundaries[-1])


def test_plug_acceptance_rate_matches_probability() -> None:
    # Bernoulli(p) by inverse transform (decision 0004 item 14): over many
    # world-days the accepted share should be close to p.
    settings = _settings(evaluation_world_count=5000, warmup_days=0, study_days=1)
    accepted, _, _ = _sample(_units(probability=0.7), settings)
    assert accepted.mean() == pytest.approx(0.7, abs=0.015)


@pytest.mark.parametrize(
    ("probability", "scale_by_cohort", "cohort_ids", "t_df"),
    [
        (0.0, {"ordinary": 30.0, "scheduled": 30.0}, ("ordinary", "scheduled"), 4.0),
        (1.0, {"ordinary": 0.0, "scheduled": 0.0}, ("ordinary", "scheduled"), 4.0),
        (0.5, {"ordinary": 30.0, "scheduled": 30.0}, ("ordinary", "scheduled"), 30.0),
        (0.5, {"always_plugged_in": 30.0}, ("always_plugged_in", "always_plugged_in"), 4.0),
    ],
)
def test_draw_count_does_not_depend_on_parameters(
    probability: float, scale_by_cohort: dict[str, float], cohort_ids: tuple[str, str], t_df: float
) -> None:
    # Common random numbers (plan M6): boundary probabilities, zero scales,
    # another df and the always-plugged cohort all consume the same draws as
    # an ordinary run, so the next channel drawn sees identical numbers.
    settings = _settings()
    base_rng = np.random.default_rng(3)
    _connect(settings, _units(probability=0.5), base_rng, {"ordinary": 30.0, "scheduled": 30.0})
    edited_rng = np.random.default_rng(3)
    _connect(
        settings,
        _units(cohort_ids=cohort_ids, probability=probability),
        edited_rng,
        scale_by_cohort,
        t_df=t_df,
    )
    assert edited_rng.random() == base_rng.random()


def test_zero_scale_cohort_has_no_shift_and_other_cohort_keeps_its_draws() -> None:
    units = _units()
    base = _connect(
        _settings(), units, np.random.default_rng(4), {"ordinary": 60.0, "scheduled": 60.0}
    )
    edited = _connect(
        _settings(), units, np.random.default_rng(4), {"ordinary": 60.0, "scheduled": 0.0}
    )
    np.testing.assert_array_equal(edited[0], base[0])
    np.testing.assert_array_equal(edited[1][:, :, 0], base[1][:, :, 0])
    assert {_utc(value).hour for value in edited[1][:, :, 1].ravel()} == {22}


# --------------------------------------------------------------------------
# Weekday and weekend windows (decision 0004 item 42)
# --------------------------------------------------------------------------


def _weekend_window_units() -> pd.DataFrame:
    # EV 1 has the CNZ-adapted weekend window (17:00 -> 10:00); EV 2 is flat.
    units = _units(cohort_ids=("ordinary", "flat"))
    units["runtime_arrival_local_hour"] = [18, 18]
    units["runtime_departure_local_hour"] = [7, 7]
    units["runtime_weekend_arrival_local_hour"] = [17, 18]
    units["runtime_weekend_departure_local_hour"] = [10, 7]
    return units


def test_each_session_end_uses_its_own_dates_window() -> None:
    # Winter week (GMT), Thursday 9 to Monday 13 January 2025, no jitter.
    settings = _settings(
        start_local_date=date(2025, 1, 9), warmup_days=0, study_days=5, evaluation_world_count=1
    )
    units = _weekend_window_units()
    accepted, starts, ends = _connect(
        settings, units, np.random.default_rng(3), {"ordinary": 0.0, "flat": 0.0}
    )
    assert accepted.all()
    sessions = [(_utc(starts[0, d, 0]), _utc(ends[0, d, 0])) for d in range(5)]
    # Thu weekday -> Fri weekday; Fri weekday arrival -> Sat weekend departure;
    # Sat and Sun weekend arrivals; Sun -> Mon weekday departure.
    assert sessions == [
        (pd.Timestamp("2025-01-09 18:00Z"), pd.Timestamp("2025-01-10 07:00Z")),
        (pd.Timestamp("2025-01-10 18:00Z"), pd.Timestamp("2025-01-11 10:00Z")),
        (pd.Timestamp("2025-01-11 17:00Z"), pd.Timestamp("2025-01-12 10:00Z")),
        (pd.Timestamp("2025-01-12 17:00Z"), pd.Timestamp("2025-01-13 07:00Z")),
        (pd.Timestamp("2025-01-13 18:00Z"), pd.Timestamp("2025-01-14 07:00Z")),
    ]
    # The flat EV keeps 18:00 -> 07:00 every day.
    assert {_utc(starts[0, d, 1]).hour for d in range(5)} == {18}
    assert {_utc(ends[0, d, 1]).hour for d in range(5)} == {7}


def test_weekend_windows_move_no_draw_and_leave_flat_evs_unchanged() -> None:
    # Common random numbers: the weekend window and scale only change how the
    # same uniforms are mapped, so a flat EV's sessions are identical with or
    # without another EV's weekend window, and so are all later draws.
    settings = _settings(start_local_date=date(2025, 1, 9), study_days=5, evaluation_world_count=50)
    flat = _weekend_window_units()
    flat["runtime_weekend_arrival_local_hour"] = flat["runtime_arrival_local_hour"]
    flat["runtime_weekend_departure_local_hour"] = flat["runtime_departure_local_hour"]
    results = []
    for units, weekend_scale in ((flat, 60.0), (_weekend_window_units(), 75.0)):
        rng = np.random.default_rng(5)
        departures = _departures(settings, units, scale=75.0)
        results.append(
            _connect(
                settings,
                units,
                rng,
                {"ordinary": 60.0, "flat": 60.0},
                weekend_scale_by_cohort={"ordinary": weekend_scale, "flat": 60.0},
                clip=180.0,
                departures=departures,
            )
        )
        results[-1] = (*results[-1], rng.random())
    flat_accepted, flat_starts, flat_ends, flat_next = results[0]
    accepted, starts, ends, next_draw = results[1]
    assert flat_next == next_draw
    assert np.array_equal(flat_accepted, accepted)
    assert np.array_equal(flat_starts[..., 1], starts[..., 1])
    assert np.array_equal(flat_ends[..., 1], ends[..., 1])
    # A hand-built frame with no weekend columns is flat too.
    no_weekend = flat.drop(
        columns=["runtime_weekend_arrival_local_hour", "runtime_weekend_departure_local_hour"]
    )
    bare = _connect(
        settings,
        no_weekend,
        np.random.default_rng(5),
        {"ordinary": 60.0, "flat": 60.0},
        clip=180.0,
        departures=_departures(settings, no_weekend, scale=75.0),
    )
    assert np.array_equal(bare[1], flat_starts) and np.array_equal(bare[2], flat_ends)


def test_weekend_plug_ins_spread_wider_than_weekday_plug_ins() -> None:
    # Thursday 9 January 2025: date 0 is a weekday, date 2 a Saturday.
    settings = _settings(
        start_local_date=date(2025, 1, 9), warmup_days=0, study_days=3, evaluation_world_count=3000
    )
    units = _weekend_window_units()
    _, starts, _ = _connect(
        settings,
        units,
        np.random.default_rng(11),
        {"ordinary": 60.0, "flat": 60.0},
        weekend_scale_by_cohort={"ordinary": 75.0, "flat": 60.0},
        clip=180.0,
    )
    minutes = starts[..., 0].astype("datetime64[m]").astype(np.int64) % (24 * 60)
    weekday_start, weekend_start = minutes[:, 0], minutes[:, 2]  # Thu 18:00, Sat 17:00
    assert abs(weekday_start.mean() - 18 * 60) < 5 and abs(weekend_start.mean() - 17 * 60) < 5
    assert weekend_start.std() > 1.1 * weekday_start.std()


# --------------------------------------------------------------------------
# Multi-day sessions (decision 0004 item 68, amending item 51)
# --------------------------------------------------------------------------


def _drive_pattern(settings: RunSettings, units: pd.DataFrame, pattern: list[bool]) -> np.ndarray:
    """Drive flags with ``pattern`` (one per sampled date) for EV 0; EV 1 drives every day."""

    drives = np.ones(
        (settings.evaluation_world_count, settings.sampled_day_count, len(units)), dtype=bool
    )
    drives[:, :, 0] = pattern
    return drives


def test_non_driving_day_keeps_the_session_plugged_in_to_the_next_driving_day() -> None:
    # Five sampled dates (warm-up 1, study 3, the morning after).  EV 0 drives
    # on dates 0 and 3 only: it plugs in on the evening of date 0, stays in
    # through dates 1 and 2 (no trip, so no unplug and no new plug-in), and
    # unplugs at date 3's departure, the instant its trip starts.  It plugs
    # in again that evening and, not driving on date 4, stays in past the
    # horizon: that session ends at the extra last departure (date 5).
    settings = _settings(study_days=3, evaluation_world_count=3)
    units = _units(cohort_ids=("ordinary", "ordinary"))
    departures = _departures(settings, units, scale=60.0)
    drives = _drive_pattern(settings, units, [True, False, False, True, False])
    accepted, starts, ends = _connect(
        settings,
        units,
        np.random.default_rng(9),
        {"ordinary": 60.0},
        departures=departures,
        drives=drives,
    )
    np.testing.assert_array_equal(accepted[:, :, 0], [[True, False, False, True, False]] * 3)
    np.testing.assert_array_equal(ends[:, 0, 0], departures[:, 3, 0])
    np.testing.assert_array_equal(ends[:, 3, 0], departures[:, 5, 0])
    # EV 1 drives every day: one session a night, each ending next morning.
    assert accepted[:, :, 1].all()
    np.testing.assert_array_equal(ends[:, :, 1], departures[:, 1:, 1])


def test_multi_day_sessions_add_no_draw_and_keep_every_plug_in_time() -> None:
    # Common random numbers: whether an EV drives only decides when a session
    # ends and which evenings start one.  The generator ends in the same
    # place, and every session that still starts keeps its plug-in instant.
    settings = _settings(study_days=4, evaluation_world_count=200)
    units = _units(cohort_ids=("ordinary", "ordinary"), probability=0.8)
    departures = _departures(settings, units, scale=60.0)
    every_day_rng, sparse_rng = np.random.default_rng(12), np.random.default_rng(12)
    every_day = _connect(settings, units, every_day_rng, {"ordinary": 60.0}, departures=departures)
    sparse_drives = np.random.default_rng(13).random(every_day[0].shape) < 0.4
    sparse = _connect(
        settings,
        units,
        sparse_rng,
        {"ordinary": 60.0},
        departures=departures,
        drives=sparse_drives,
    )
    assert every_day_rng.random() == sparse_rng.random()
    assert not (sparse[0] & ~every_day[0]).any()  # never a session the plug draw refused
    np.testing.assert_array_equal(sparse[1][sparse[0]], every_day[1][sparse[0]])
    assert sparse[0].sum() < every_day[0].sum()


def test_every_session_ends_at_the_first_driving_departure_after_it() -> None:
    # The item 68 rule as an invariant over random drive flags and skips:
    # session d ends at the departure of the first later date k the EV
    # drives (or the extra last date), no session starts in between, and
    # sessions never overlap (so an EV plugged in from a non-driving day is
    # never "skipped": it simply stays in).
    settings = _settings(study_days=5, evaluation_world_count=300)
    units = _units(cohort_ids=("ordinary", "ordinary"), probability=0.7)
    departures = _departures(settings, units, scale=60.0)
    drives = (
        np.random.default_rng(31).random(
            (settings.evaluation_world_count, settings.sampled_day_count, 2)
        )
        < 0.5
    )
    accepted, starts, ends = _connect(
        settings,
        units,
        np.random.default_rng(32),
        {"ordinary": 60.0},
        departures=departures,
        drives=drives,
    )
    days = settings.sampled_day_count
    multi_day = 0
    for world, day, ev in zip(*np.nonzero(accepted)):
        later = [k for k in range(day + 1, days) if drives[world, k, ev]]
        unplug = later[0] if later else days
        assert ends[world, day, ev] == departures[world, unplug, ev]
        assert not accepted[world, day + 1 : unplug, ev].any()
        multi_day += unplug > day + 1
    assert multi_day > 0


def test_always_plugged_session_ignores_drive_flags() -> None:
    settings = _settings(evaluation_world_count=2)
    units = _units(cohort_ids=("always_plugged_in", "ordinary"))
    scales = {"always_plugged_in": 0.0, "ordinary": 30.0}
    no_drives = np.zeros(
        (settings.evaluation_world_count, settings.sampled_day_count, 2), dtype=bool
    )
    base = _connect(settings, units, np.random.default_rng(4), scales)
    idle = _connect(settings, units, np.random.default_rng(4), scales, drives=no_drives)
    for base_value, idle_value in zip(base, idle):
        np.testing.assert_array_equal(base_value[:, :, 0], idle_value[:, :, 0])


@pytest.mark.parametrize(
    "drives",
    [np.ones((4, 4, 2)), np.ones((4, 3, 2), dtype=bool)],
    ids=["not-boolean", "wrong-shape"],
)
def test_drive_flags_are_checked(drives: np.ndarray) -> None:
    with pytest.raises(ValueError, match="drives_today"):
        _connect(
            _settings(),
            _units(),
            np.random.default_rng(1),
            {"ordinary": 30.0, "scheduled": 30.0},
            drives=drives,
        )
