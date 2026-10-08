import warnings
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from axle_studio.model.clock import london_wall_time_to_utc
from axle_studio.model.physics import PublicTopUp, simulate_fleet_intervals
from axle_studio.model.sampling import sample_connection_opportunities, sample_daily_trip_inputs
from axle_studio.model.settings import RunSettings

# The decision 0004 item 32/37 defaults: 90 % public efficiency, top up below
# 10 % SoC to 80 % SoC (re-baselined from 20 % by item 37).
_TOP_UP = PublicTopUp(efficiency_fraction=0.9, threshold_soc_fraction=0.1, target_soc_fraction=0.8)


def _utc(value: object) -> pd.Timestamp:
    """A sampled instant (naive ``datetime64[ns]`` in UTC) as an aware Timestamp."""

    return pd.Timestamp(value, tz="UTC")


def _settings(**changes: object) -> RunSettings:
    return replace(
        RunSettings(
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
        ),
        **changes,
    )


def _units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population_id": ["persistent-pop"] * 2,
            "unit_id": ["ev-1", "ev-2"],
            "cohort_id": ["weekday", "weekend"],
            "plug_probability": [1.0, 1.0],
            "runtime_arrival_local_hour": [18, 22],
            "runtime_departure_local_hour": [7, 9],
            "physical_capacity_kwh": [10.0, 10.0],
            "efficiency_miles_per_battery_kwh": [4.0, 4.0],
            "home_charger_limit_kw": [2.0, 2.0],
            "vehicle_ac_limit_kw": [2.0, 2.0],
            "preferred_target_soc_fraction": [0.8, 0.7],
        }
    )


def _departures(settings: RunSettings, minutes: tuple[float, float] = (420.0, 480.0)) -> np.ndarray:
    """Fixed home departures (London clock minutes per EV) on every date and world.

    ``sample_daily_trip_inputs`` takes ``sample_departure_times``'s
    (world, sampled days + 1, EV) array; a fixed one keeps these trip tests
    about trips.  Sampled days are ``RunSettings.sampled_day_count`` (warm-up +
    study + the morning after the last noon-to-noon night, decision 0004 item 52).
    """

    days = settings.sampled_day_count
    first_local_date = settings.start_local_date - timedelta(days=settings.warmup_days)
    midnights = np.datetime64(first_local_date, "D") + np.arange(days + 1)
    one_world = london_wall_time_to_utc(midnights[:, np.newaxis], np.asarray(minutes)[np.newaxis])
    shape = (settings.evaluation_world_count, days + 1, settings.vehicle_count)
    return np.broadcast_to(one_world, shape).copy()


def _sample(seed: int = 7, **changes: object) -> dict[str, np.ndarray]:
    settings = _settings(**changes)
    days = settings.sampled_day_count
    return sample_daily_trip_inputs(
        settings,
        _units(),
        np.random.default_rng(seed),
        departure_utc=_departures(settings),
        drive_probability=np.tile([[1.0, 0.5]], (days, 1)),
        expected_daily_miles=np.tile([[20.0, 10.0]], (days, 1)),
        distance_cv=np.array([0.0, 0.5]),
        destination_dwell_minutes=np.array([30.0, 45.0]),
        drive_speed_mph=np.array([30.0, 25.0]),
        desired_pre_drive_soc_fraction=np.array([0.8, 0.7]),
    )


def _direct_sample(
    *,
    rng: np.random.Generator | None = None,
    drive_probability: float = 1.0,
    expected_daily_miles: float = 1.0,
    distance_cv: float = 0.0,
    destination_dwell_minutes: float = 0.0,
    drive_speed_mph: float = 30.0,
) -> dict[str, np.ndarray]:
    settings = _settings(evaluation_world_count=1, warmup_days=0, study_days=1)
    return sample_daily_trip_inputs(
        settings,
        _units(),
        np.random.default_rng(1) if rng is None else rng,
        departure_utc=_departures(settings),
        drive_probability=np.full((2, 2), drive_probability),
        expected_daily_miles=np.full((2, 2), expected_daily_miles),
        distance_cv=np.full(2, distance_cv),
        destination_dwell_minutes=np.full(2, destination_dwell_minutes),
        drive_speed_mph=np.full(2, drive_speed_mph),
        desired_pre_drive_soc_fraction=np.full(2, 0.8),
    )


def test_shape_keys_repeatability_and_persistent_unit_inputs() -> None:
    result = _sample()
    repeated = _sample()
    assert tuple(result) == (
        "drives_today",
        "outbound_miles",
        "return_miles",
        "trip_departure_utc",
        "destination_dwell_seconds",
        "drive_speed_mph",
        "desired_pre_drive_soc_fraction",
    )
    # warm-up 1 + study 2 + the morning after the last night = 4 sampled dates.
    assert all(value.shape == (4, 4, 2) for value in result.values())
    assert result["drives_today"].dtype == bool
    assert result["trip_departure_utc"].dtype == np.dtype("datetime64[ns]")
    for key in result:
        if key == "trip_departure_utc":
            assert all(
                (pd.isna(left) and pd.isna(right)) or left == right
                for left, right in zip(result[key].flat, repeated[key].flat, strict=True)
            )
        else:
            np.testing.assert_array_equal(result[key], repeated[key])
    assert np.all(result["drive_speed_mph"][:, :, 0] == 30.0)


def test_non_drive_days_are_zero_and_have_nat_departure() -> None:
    result = _sample()
    non_drive = ~result["drives_today"]
    assert np.all(result["outbound_miles"][non_drive] == 0.0)
    assert np.all(result["return_miles"][non_drive] == 0.0)
    assert all(pd.isna(value) for value in result["trip_departure_utc"][non_drive])


def test_miles_are_equal_legs_and_unconditional_mean_is_preserved() -> None:
    settings = _settings(evaluation_world_count=20_000, warmup_days=0, study_days=1)
    days = settings.sampled_day_count
    result = sample_daily_trip_inputs(
        settings,
        _units(),
        np.random.default_rng(3),
        departure_utc=_departures(settings),
        drive_probability=np.full((days, 2), 0.5),
        expected_daily_miles=np.full((days, 2), 12.0),
        distance_cv=np.zeros(2),
        destination_dwell_minutes=np.zeros(2),
        drive_speed_mph=np.full(2, 30.0),
        desired_pre_drive_soc_fraction=np.full(2, 0.8),
    )
    miles = result["outbound_miles"] + result["return_miles"]
    np.testing.assert_allclose(miles.mean(axis=0), 12.0, atol=0.15)
    np.testing.assert_array_equal(result["outbound_miles"], result["return_miles"])


def test_weekday_weekend_inputs_are_caller_controlled() -> None:
    result = _sample()
    assert np.all(result["drive_speed_mph"][:, :, 0] == 30.0)
    assert np.all(result["drive_speed_mph"][:, :, 1] == 25.0)
    assert np.all(result["destination_dwell_seconds"][:, :, 1] == 2700.0)


def test_a_driving_day_starts_its_trip_at_that_days_home_departure() -> None:
    # Decision 0004 item 51: plug-out is departure.  The trip sampler draws
    # no clock of its own; day d's trip starts at the supplied departure for
    # London date d, and the extra last date (the next morning) is not a trip.
    settings = _settings()
    departures = _departures(settings)
    departures[:, :, 0] += np.timedelta64(90, "m")
    result = sample_daily_trip_inputs(
        settings,
        _units(),
        np.random.default_rng(4),
        departure_utc=departures,
        drive_probability=np.tile([[1.0, 0.5]], (4, 1)),
        expected_daily_miles=np.tile([[20.0, 10.0]], (4, 1)),
        distance_cv=np.array([0.0, 0.5]),
        destination_dwell_minutes=np.array([30.0, 45.0]),
        drive_speed_mph=np.array([30.0, 25.0]),
        desired_pre_drive_soc_fraction=np.array([0.8, 0.7]),
    )
    drives = result["drives_today"]
    assert drives[:, :, 0].all()
    np.testing.assert_array_equal(result["trip_departure_utc"][drives], departures[:, :4][drives])
    assert _utc(result["trip_departure_utc"][0, 1, 0]) == pd.Timestamp("2025-01-06 08:30:00+00:00")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"drive_probability": np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])},
        {"distance_cv": np.array([-0.1, 0.0])},
        {"drive_speed_mph": np.array([0.0, 25.0])},
        # One date short: the trips need every date's departure plus the next morning's.
        {"departure_utc": _departures(_settings())[:, :-1]},
        {"departure_utc": np.zeros((4, 4, 2))},
    ],
)
def test_invalid_inputs_are_rejected(kwargs: dict[str, object]) -> None:
    settings = _settings()
    defaults = {
        "departure_utc": _departures(settings),
        "drive_probability": np.ones((4, 2)),
        "expected_daily_miles": np.ones((4, 2)),
        "distance_cv": np.zeros(2),
        "destination_dwell_minutes": np.zeros(2),
        "drive_speed_mph": np.full(2, 30.0),
        "desired_pre_drive_soc_fraction": np.full(2, 0.8),
    }
    defaults.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        sample_daily_trip_inputs(settings, _units(), np.random.default_rng(1), **defaults)


def test_zero_probability_requires_zero_expected_miles() -> None:
    days = 4
    with pytest.raises(ValueError, match="zero when drive_probability"):
        sample_daily_trip_inputs(
            _settings(),
            _units(),
            np.random.default_rng(1),
            departure_utc=_departures(_settings()),
            drive_probability=np.zeros((days, 2)),
            expected_daily_miles=np.ones((days, 2)),
            distance_cv=np.zeros(2),
            destination_dwell_minutes=np.zeros(2),
            drive_speed_mph=np.full(2, 30.0),
            desired_pre_drive_soc_fraction=np.full(2, 0.8),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"destination_dwell_minutes": np.array([1e308, 0.0])},
            "destination_dwell_minutes",
        ),
        (
            {
                "drive_probability": np.full((4, 2), 0.5),
                "expected_daily_miles": np.full((4, 2), 1e308),
            },
            "daily mileage",
        ),
        ({"distance_cv": np.array([1e308, 0.0])}, "daily mileage"),
    ],
)
def test_overflowing_derived_values_are_rejected_without_warnings(
    kwargs: dict[str, object], message: str
) -> None:
    defaults = {
        "departure_utc": _departures(_settings()),
        "drive_probability": np.ones((4, 2)),
        "expected_daily_miles": np.ones((4, 2)),
        "distance_cv": np.zeros(2),
        "destination_dwell_minutes": np.zeros(2),
        "drive_speed_mph": np.full(2, 30.0),
        "desired_pre_drive_soc_fraction": np.full(2, 0.8),
    }
    defaults.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match=message):
            sample_daily_trip_inputs(_settings(), _units(), np.random.default_rng(2), **defaults)


@pytest.mark.parametrize("distance_cv", [0.0, 1.0])
def test_smallest_positive_expected_mileage_cannot_create_zero_legs(
    distance_cv: float,
) -> None:
    with pytest.raises(ValueError, match="two positive mileage legs"):
        _direct_sample(
            expected_daily_miles=np.nextafter(0.0, 1.0),
            distance_cv=distance_cv,
        )


def test_sampled_zero_mileage_is_rejected_after_sampling() -> None:
    # Mock re-targeted: distances now come from scipy.stats.lognorm, which draws
    # exp(s × z) from Generator.standard_normal (decision 0004 item 14).  The
    # first standard-normal block is the distance channel; a huge negative z
    # underflows the lognormal multiplier to exactly zero miles.
    class ZeroLognormalGenerator(np.random.Generator):
        calls = 0

        def standard_normal(self, size=None, dtype=np.float64, out=None):
            ZeroLognormalGenerator.calls += 1
            if ZeroLognormalGenerator.calls == 1:
                return np.full(size, -1e4, dtype=float)
            return super().standard_normal(size=size, dtype=dtype, out=out)

    rng = ZeroLognormalGenerator(np.random.PCG64(1))
    with pytest.raises(ValueError, match="two positive mileage legs"):
        _direct_sample(rng=rng, distance_cv=1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"destination_dwell_minutes": 1e100}, "destination dwell"),
        ({"expected_daily_miles": 1e100}, "leg durations"),
        ({"drive_speed_mph": 1e308}, "at least one nanosecond"),
        ({"expected_daily_miles": 3e6, "drive_speed_mph": 1.0}, "return timestamp"),
    ],
)
def test_active_events_must_fit_the_scalar_kernel_domain(
    kwargs: dict[str, float], message: str
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match=message):
            _direct_sample(**kwargs)


def test_no_drive_entries_ignore_representable_external_extremes() -> None:
    result = _direct_sample(
        drive_probability=0.0,
        expected_daily_miles=0.0,
        destination_dwell_minutes=1e100,
        drive_speed_mph=1e308,
    )
    assert not result["drives_today"].any()
    assert np.all(result["outbound_miles"] == 0.0)
    assert np.all(result["return_miles"] == 0.0)
    assert all(pd.isna(value) for value in result["trip_departure_utc"].flat)


def test_output_passes_explicit_kernel_with_connection_inputs() -> None:
    settings = _settings(vehicle_count=2, evaluation_world_count=1, warmup_days=0, study_days=1)
    units = _units()
    trips = _sample(seed=5, evaluation_world_count=1, warmup_days=0, study_days=1)
    accepted, starts, ends = sample_connection_opportunities(
        settings,
        units,
        np.random.default_rng(6),
        departure_utc=_departures(settings),
        drives_today=trips["drives_today"],
        plug_in_scale_minutes_by_cohort={"weekday": 0.0, "weekend": 0.0},
        t_df=4.0,
        clip_minutes=0.0,
    )
    fleet, *_ = simulate_fleet_intervals(
        settings,
        units,
        {
            **trips,
            "connection_session_accepted": accepted,
            "connection_start_utc": starts,
            "connection_end_utc": ends,
        },
        public_top_up=_TOP_UP,
    )
    assert not fleet.empty


def _many_world_sample(
    rng: np.random.Generator,
    *,
    drive_probability: tuple[float, float] = (0.6, 1.0),
    distance_cv: tuple[float, float] = (0.4, 0.0),
    worlds: int = 20_000,
) -> dict[str, np.ndarray]:
    settings = _settings(evaluation_world_count=worlds, warmup_days=0, study_days=1)
    return sample_daily_trip_inputs(
        settings,
        _units(),
        rng,
        departure_utc=_departures(settings),
        drive_probability=np.tile([drive_probability], (2, 1)),
        expected_daily_miles=np.tile(np.array([20.0, 10.0]) * np.array(drive_probability), (2, 1)),
        distance_cv=np.array(distance_cv),
        destination_dwell_minutes=np.zeros(2),
        drive_speed_mph=np.full(2, 30.0),
        desired_pre_drive_soc_fraction=np.full(2, 0.8),
    )


def test_drive_and_distance_draws_match_their_distributions() -> None:
    # Shape check for the scipy.stats draws (decision 0004 item 14): Bernoulli
    # drive rate and mean-one lognormal distance (mean 20 mi, CV 0.4 on a
    # driving day).  The departure clock is tested in test_sampling_clocks.py.
    result = _many_world_sample(np.random.default_rng(11))
    drives = result["drives_today"][:, 0, 0]
    miles = 2.0 * result["outbound_miles"][:, 0, 0][drives]

    assert drives.mean() == pytest.approx(0.6, abs=0.015)
    assert miles.mean() == pytest.approx(20.0, rel=0.02)
    assert miles.std() / miles.mean() == pytest.approx(0.4, rel=0.05)
    assert (miles > 0.0).all()


@pytest.mark.parametrize(
    "changes",
    [
        {"drive_probability": (1.0, 1.0)},
        {"distance_cv": (0.0, 0.0)},
    ],
)
def test_parameter_edits_change_only_their_own_trip_channel(
    changes: dict[str, tuple[float, float]],
) -> None:
    # Common random numbers (plan M6): each channel takes a fixed block of draws,
    # so editing EV 1's probability or CV (including to the boundary 1 or 0)
    # leaves EV 2 and the other channels unchanged and the generator ends in the
    # same position for whatever is sampled next.
    base_rng = np.random.default_rng(5)
    base = _many_world_sample(base_rng, worlds=200)
    edited_rng = np.random.default_rng(5)
    edited = _many_world_sample(edited_rng, worlds=200, **changes)

    assert edited_rng.random() == base_rng.random()
    np.testing.assert_array_equal(edited["drives_today"][:, :, 1], base["drives_today"][:, :, 1])
    np.testing.assert_array_equal(
        edited["outbound_miles"][:, :, 1], base["outbound_miles"][:, :, 1]
    )
    assert _same_times(edited["trip_departure_utc"][:, :, 1], base["trip_departure_utc"][:, :, 1])
    if "drive_probability" not in changes:
        np.testing.assert_array_equal(edited["drives_today"], base["drives_today"])
    if "drive_probability" not in changes:
        assert _same_times(edited["trip_departure_utc"], base["trip_departure_utc"])


def _same_times(left: np.ndarray, right: np.ndarray) -> bool:
    # NaT never equals NaT elementwise, so compare as DatetimeIndex values.
    return pd.DatetimeIndex(left.ravel()).equals(pd.DatetimeIndex(right.ravel()))
