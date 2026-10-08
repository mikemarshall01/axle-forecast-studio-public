"""Home departure and plug-in clocks (decision 0004 item 51).

Both clocks are truncated Student-t shifts drawn from one uniform per cell;
the departure is both the unplug and the trip start, and the plug-in is an
independent draw.  These tests pin the shape of the distribution at the
default illustrative scales, the df -> normal limit, independence and the
fixed draw counts that common random numbers rely on.
"""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from axle_studio.model import assumptions
from axle_studio.model.sampling import (
    sample_connection_opportunities,
    sample_departure_times,
    truncated_t_shift_minutes,
)
from axle_studio.model.settings import RunSettings

_CLIP = assumptions.CONNECTION_CLOCKS["clock_clip_minutes"].value
_DF = assumptions.CONNECTION_CLOCKS["clock_t_df"].value


def _settings(**changes: object) -> RunSettings:
    base = RunSettings(
        start_local_date=date(2025, 1, 6),
        warmup_days=0,
        study_days=1,
        vehicle_count=1,
        seed=1,
        evaluation_world_count=4,
        opening_soc_fraction=0.5,
        reserve_soc_fraction=0.1,
        home_charge_efficiency=1.0,
        weather_efficiency_sensitivity_fraction_per_c=0.01,
    )
    return replace(base, **changes)


def _units(cohort_id: str = "ordinary") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["ev-1"],
            "cohort_id": [cohort_id],
            "plug_probability": [1.0],
            "runtime_arrival_local_hour": [18],
            "runtime_departure_local_hour": [7],
        }
    )


def _shifts(scale: float, t_df: float = _DF, count: int = 200_000, seed: int = 3) -> np.ndarray:
    uniforms = stats.uniform.rvs(size=count, random_state=np.random.default_rng(seed))
    return truncated_t_shift_minutes(uniforms, np.full(count, scale), t_df, _CLIP)


def _local_minutes(instants: np.ndarray) -> np.ndarray:
    local = pd.DatetimeIndex(instants.ravel()).tz_localize("UTC").tz_convert("Europe/London")
    return (local.hour * 60 + local.minute).to_numpy().reshape(instants.shape)


# --------------------------------------------------------------------------
# The truncated-t shift
# --------------------------------------------------------------------------


def test_the_default_clock_is_zero_centred_whole_half_hours_inside_the_clip() -> None:
    assert _CLIP == 180.0 and _DF == 4.0
    shifts = _shifts(75.0)
    assert (shifts % 30.0 == 0.0).all()
    assert np.abs(shifts).max() == _CLIP
    assert abs(np.median(shifts)) == 0.0
    assert abs(shifts.mean()) < 1.0


@pytest.mark.parametrize("scale", [60.0, 75.0, 90.0])
def test_default_scales_give_about_two_hours_p5_to_p95_and_never_beyond_three(
    scale: float,
) -> None:
    # Decision 0004 item 51: P5-P95 about +/-2 h (within +/-2.5 h), none
    # beyond the +/-3 h bound, and only a small share at the bound itself.
    shifts = _shifts(scale)
    p5, p95 = np.percentile(shifts, [5, 95])
    assert -150.0 <= p5 <= -90.0 and 90.0 <= p95 <= 150.0
    assert (np.abs(shifts) <= 180.0).all()
    assert np.mean(np.abs(shifts) == 180.0) < 0.04
    if scale == 75.0:
        # Plan A1: 92-95 % within +/-2 h and 1-2 % at +/-3 h for the default.
        assert 0.90 <= np.mean(np.abs(shifts) <= 120.0) <= 0.95
        assert 0.01 <= np.mean(np.abs(shifts) == 180.0) <= 0.025


def test_many_degrees_of_freedom_reproduce_a_rounded_normal() -> None:
    # df >= 30 is practically a normal: each half-hour bin's share matches a
    # normal with the same scale, rounded to the half hour, within 1 point.
    scale = 60.0
    shifts = _shifts(scale, t_df=30.0)
    bins = np.arange(-180.0, 181.0, 30.0)
    edges = np.concatenate(([-np.inf], bins[1:] - 15.0, [np.inf]))
    normal = np.diff(stats.norm.cdf(edges / scale))
    sampled = np.array([np.mean(shifts == value) for value in bins])
    np.testing.assert_allclose(sampled, normal, atol=0.01)
    # Fat tails at the default df: clearly more mass at 2.5 h or beyond than the normal.
    fat = _shifts(scale, t_df=4.0)
    assert np.mean(np.abs(fat) >= 150.0) > 1.5 * np.mean(np.abs(shifts) >= 150.0)


def test_zero_scale_means_no_shift() -> None:
    uniforms = np.array([0.0, 0.2, 0.5, 0.999])
    np.testing.assert_array_equal(
        truncated_t_shift_minutes(uniforms, np.zeros(4), 4.0, 180.0), np.zeros(4)
    )


# --------------------------------------------------------------------------
# Departure times: the unplug and the trip start
# --------------------------------------------------------------------------


def test_departures_cover_every_date_and_the_next_morning_on_local_clock() -> None:
    # Two noon-to-noon study nights (6 and 7 January): sampled dates 6, 7 and
    # 8 January (RunSettings.sampled_day_count, decision 0004 item 52), plus
    # the 9 January departure that ends the last sampled date's session.
    settings = _settings(study_days=2)
    departures = sample_departure_times(
        settings,
        _units(),
        np.random.default_rng(1),
        scale_minutes_by_cohort={"ordinary": 0.0},
        t_df=4.0,
        clip_minutes=180.0,
    )
    assert departures.shape == (4, 4, 1) and departures.dtype == np.dtype("datetime64[ns]")
    assert (_local_minutes(departures) == 7 * 60).all()
    local_dates = pd.DatetimeIndex(departures[0, :, 0]).date
    assert list(local_dates) == [date(2025, 1, 6 + d) for d in range(4)]


@pytest.mark.parametrize("start_local_date", [date(2026, 3, 27), date(2026, 10, 25)])
def test_offset_change_keeps_the_local_departure_clock_on_each_local_date(
    start_local_date: date,
) -> None:
    settings = _settings(start_local_date=start_local_date, warmup_days=1, study_days=7)
    departures = sample_departure_times(
        settings,
        _units(),
        np.random.default_rng(1),
        scale_minutes_by_cohort={"ordinary": 0.0},
        t_df=4.0,
        clip_minutes=180.0,
    )
    first_local_date = start_local_date - timedelta(days=1)
    local = pd.DatetimeIndex(departures[0, :, 0]).tz_localize("UTC").tz_convert("Europe/London")
    assert list(local.date) == [first_local_date + timedelta(days=d) for d in range(10)]
    assert set(local.hour) == {7} and set(local.minute) == {0}


def test_departures_follow_the_weekend_window() -> None:
    # Friday 10 to Monday 13 January 2025 (GMT): 07:00 on weekdays, 10:00 at weekends.
    settings = _settings(start_local_date=date(2025, 1, 10), study_days=2)
    units = _units().assign(runtime_weekend_departure_local_hour=[10])
    departures = sample_departure_times(
        settings,
        units,
        np.random.default_rng(1),
        scale_minutes_by_cohort={"ordinary": 0.0},
        t_df=4.0,
        clip_minutes=180.0,
    )
    assert _local_minutes(departures[0, :, 0]).tolist() == [420, 600, 600, 420]


def test_a_departure_that_could_leave_its_local_date_is_rejected() -> None:
    units = _units().assign(runtime_departure_local_hour=[2])
    with pytest.raises(ValueError, match="within a local day"):
        sample_departure_times(
            _settings(),
            units,
            np.random.default_rng(1),
            scale_minutes_by_cohort={"ordinary": 60.0},
            t_df=4.0,
            clip_minutes=180.0,
        )


@pytest.mark.parametrize(("scale", "t_df"), [(0.0, 4.0), (90.0, 4.0), (75.0, 30.0)])
def test_departure_draw_count_does_not_depend_on_scale_or_df(scale: float, t_df: float) -> None:
    settings = _settings(study_days=3)
    base_rng = np.random.default_rng(8)
    sample_departure_times(
        settings,
        _units(),
        base_rng,
        scale_minutes_by_cohort={"ordinary": 75.0},
        t_df=4.0,
        clip_minutes=180.0,
    )
    edited_rng = np.random.default_rng(8)
    sample_departure_times(
        settings,
        _units(),
        edited_rng,
        scale_minutes_by_cohort={"ordinary": scale},
        t_df=t_df,
        clip_minutes=180.0,
    )
    assert edited_rng.random() == base_rng.random()


# --------------------------------------------------------------------------
# Plug-in and departure are independent (item 51)
# --------------------------------------------------------------------------


def test_plug_in_and_departure_shifts_are_independent() -> None:
    # The old model shared one draw between a session's two ends (correlation
    # 1).  Now a session's plug-in shift and its ending departure shift, and
    # a date's departure and that evening's plug-in, are uncorrelated.
    settings = _settings(evaluation_world_count=20_000, study_days=2)
    units = _units()
    rng = np.random.default_rng(21)
    departures = sample_departure_times(
        settings,
        units,
        rng,
        scale_minutes_by_cohort={"ordinary": 75.0},
        t_df=4.0,
        clip_minutes=180.0,
    )
    accepted, starts, _ = sample_connection_opportunities(
        settings,
        units,
        rng,
        departure_utc=departures,
        drives_today=np.ones(
            (settings.evaluation_world_count, settings.sampled_day_count, len(units)), dtype=bool
        ),
        plug_in_scale_minutes_by_cohort={"ordinary": 60.0},
        t_df=4.0,
        clip_minutes=180.0,
    )
    assert accepted.all()
    departure_shift = _local_minutes(departures[:, :, 0]) - 7 * 60
    plug_in_shift = _local_minutes(starts[:, :, 0]) - 18 * 60
    for plug_in, departure in (
        (plug_in_shift[:, 0], departure_shift[:, 1]),  # one session's two ends
        (plug_in_shift[:, 1], departure_shift[:, 1]),  # one date's morning and evening
    ):
        assert abs(np.corrcoef(plug_in, departure)[0, 1]) < 0.03
    assert departure_shift.std() > plug_in_shift.std()  # scale 75 against 60
