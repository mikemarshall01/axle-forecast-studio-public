from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from axle_studio.model.clock import (
    STUDY_START_LOCAL_HOUR,
    london_wall_time_to_utc,
    session_night_dates,
    utc_half_hour_boundaries,
)

_LONDON = ZoneInfo("Europe/London")


def test_safe_study_day_counts_produce_utc_half_hour_boundaries() -> None:
    for study_days in (1, 3, 7):
        boundaries = utc_half_hour_boundaries(date(2025, 1, 6), 0, study_days)

        assert len(boundaries) == 48 * study_days + 1
        assert all(boundary.tzinfo == timezone.utc for boundary in boundaries)
        assert all(
            later - earlier == timedelta(minutes=30)
            for earlier, later in zip(boundaries, boundaries[1:])
        )


def test_safe_nonzero_warmup_has_exact_count_and_utc_steps() -> None:
    boundaries = utc_half_hour_boundaries(date(2025, 1, 6), 2, 3)

    assert len(boundaries) == 48 * (2 + 3) + 1
    assert all(
        later - earlier == timedelta(minutes=30)
        for earlier, later in zip(boundaries, boundaries[1:])
    )


@pytest.mark.parametrize(
    ("start_local_date", "warmup_days", "study_days", "first_local", "last_local"),
    [
        # Spring-forward on Sunday 29 March 2026 inside the study: 336 fixed
        # UTC slots end at 13:00 London (decision 0004 items 1 and 52).
        (date(2026, 3, 23), 1, 7, "2026-03-22 12:00", "2026-03-30 13:00"),
        # Spring-forward inside the warm-up: warm-up counts back 24 h in UTC.
        (date(2026, 3, 29), 1, 1, "2026-03-28 11:00", "2026-03-30 12:00"),
        # Fall-back on Sunday 25 October 2026 inside the study: ends 11:00.
        (date(2026, 10, 19), 1, 7, "2026-10-18 12:00", "2026-10-26 11:00"),
        # Fall-back inside the warm-up.
        (date(2026, 10, 25), 1, 1, "2026-10-24 13:00", "2026-10-26 12:00"),
    ],
)
def test_study_starts_at_london_noon_across_offset_changes(
    start_local_date: date,
    warmup_days: int,
    study_days: int,
    first_local: str,
    last_local: str,
) -> None:
    boundaries = utc_half_hour_boundaries(start_local_date, warmup_days, study_days)

    assert len(boundaries) == 48 * (warmup_days + study_days) + 1
    assert all(boundary.tzinfo == timezone.utc for boundary in boundaries)
    assert all(
        later - earlier == timedelta(minutes=30)
        for earlier, later in zip(boundaries, boundaries[1:])
    )
    study_start = boundaries[48 * warmup_days].astimezone(_LONDON)
    assert (study_start.date(), study_start.time()) == (start_local_date, time(12))
    assert boundaries[0].astimezone(_LONDON).strftime("%Y-%m-%d %H:%M") == first_local
    assert boundaries[-1].astimezone(_LONDON).strftime("%Y-%m-%d %H:%M") == last_local


def test_study_start_is_the_documented_noon_constant() -> None:
    boundaries = utc_half_hour_boundaries(date(2025, 1, 6), 2, 3)

    assert STUDY_START_LOCAL_HOUR == 12
    assert boundaries[96] == datetime(2025, 1, 6, 12, tzinfo=timezone.utc)
    assert boundaries[0] == datetime(2025, 1, 4, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("london", "night"),
    [
        ("2026-01-12 12:00", "2026-01-12"),
        ("2026-01-12 23:30", "2026-01-12"),
        ("2026-01-13 03:00", "2026-01-12"),
        ("2026-01-13 11:30", "2026-01-12"),
        ("2026-01-13 12:00", "2026-01-13"),
        # Clock-change nights: the boundary stays at London noon.
        ("2026-03-29 11:30", "2026-03-28"),
        ("2026-03-29 12:00", "2026-03-29"),
        # The second (GMT) 01:30 of the repeated autumn hour.
        ("2026-10-25 01:30+00:00", "2026-10-24"),
        ("2026-10-25 12:00", "2026-10-25"),
    ],
)
def test_session_night_is_london_time_minus_twelve_hours(london: str, night: str) -> None:
    instant = pd.Timestamp(london)
    instant = instant.tz_localize(_LONDON) if instant.tz is None else instant
    start = pd.DatetimeIndex([instant.tz_convert("UTC")])

    assert session_night_dates(start)[0] == np.datetime64(night, "D")


@pytest.mark.parametrize(
    "start_local_date",
    [datetime(2025, 1, 6), "2025-01-06", None],
)
def test_rejects_invalid_start_local_date(start_local_date: object) -> None:
    with pytest.raises(TypeError, match="start_local_date must be a date"):
        utc_half_hour_boundaries(start_local_date, 0, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("warmup_days", "study_days"),
    [
        (-1, 1),
        (0, 0),
        (0, -1),
        (False, 1),
        (0, True),
        (0.0, 1),
        (0, 1.0),
        (0, "1"),
    ],
)
def test_rejects_invalid_day_counts(warmup_days: object, study_days: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        utc_half_hour_boundaries(
            date(2025, 1, 6),
            warmup_days,
            study_days,  # type: ignore[arg-type]
        )


def test_boundaries_are_datetime_values() -> None:
    boundaries = utc_half_hour_boundaries(date(2025, 1, 6), 0, 1)

    assert all(isinstance(boundary, datetime) for boundary in boundaries)


@pytest.mark.parametrize(
    "local_date",
    [date(2026, 3, 29), date(2026, 10, 25), date(2026, 1, 12), date(2026, 7, 1)],
)
def test_london_wall_time_to_utc_matches_python_datetime_arithmetic(local_date: date) -> None:
    # The vectorised helper must give exactly what the per-value loop gave:
    # combine(date, 00:00, London) + timedelta(minutes), then UTC.  The
    # minutes cover the spring gap and the autumn repeated hour (01:00-02:00),
    # whole half-hours and a fractional minute.
    minutes = np.array([0.0, 59.5, 60.0, 75.0, 90.0, 119.0, 120.0, 150.0, 450.0, 1410.0, 7.25])
    actual = london_wall_time_to_utc(np.datetime64(local_date, "D"), minutes)

    midnight = datetime.combine(local_date, datetime.min.time(), tzinfo=_LONDON)
    expected = [
        pd.Timestamp(midnight + timedelta(minutes=float(m))).tz_convert("UTC").tz_convert(None)
        for m in minutes
    ]
    assert actual.dtype == np.dtype("datetime64[ns]")
    assert list(pd.DatetimeIndex(actual)) == expected
