"""UTC simulation-clock helpers.

What this owns: the run's fixed UTC half-hour slot boundaries (a noon-to-noon
study, decision 0004 item 52), the session-night date that reporting days key
on, turning London wall-clock times into UTC instants, and which sampled
dates are holiday evenings (trading contract v1 §10.1c).  Every interval
key in the model is UTC; London time is only how behaviour (departure and
plug-in clocks) is stated and how results are labelled (decision 0004 item
1).
"""

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from axle_studio.model.assumptions import HOLIDAY_SKIP_LOGIT_SHIFT
from axle_studio.model.settings import RunSettings

_LONDON = ZoneInfo("Europe/London")
_HALF_HOUR = timedelta(minutes=30)


STUDY_START_LOCAL_HOUR = 12
"""London clock hour the study (and each session night) starts at (decision 0004 item 52).

A night's charging runs from the evening plug-in to the next morning's
departure, so a noon-to-noon block holds a whole overnight session.  The
earlier midnight start cut the first night in half: sessions already plugged
in at 00:00 were planned from the first study slot with no view of their
evening, which made the first study night an unmanaged night.
"""


def utc_half_hour_boundaries(
    start_local_date: date, warmup_days: int, study_days: int
) -> tuple[datetime, ...]:
    """Return ``48 * (warmup_days + study_days) + 1`` UTC half-hour boundaries.

    Slot ``48 * warmup_days`` starts at London 12:00 on ``start_local_date``
    (decision 0004 item 52).  The study is the fixed ``48 * study_days`` UTC
    slots from there and the warm-up is ``48 * warmup_days`` UTC slots counted
    back from it, so a London offset change anywhere in the span is allowed
    (decision 0004 item 1).  A change in the warm-up moves ``boundaries[0]``
    one hour off London noon; a change in the study moves the horizon end one
    hour off it (13:00 after spring-forward, 11:00 after fall-back).  Callers
    treat sampled "day d" as the London calendar date ``first date + d`` and
    convert its local clock times at that date's actual offset; they must not
    assume slot ``48 * d`` is any particular London clock time.

    The earlier midnight-anchored variant that rejected offset changes had no
    caller left and was removed with the noon start.
    """
    if isinstance(start_local_date, datetime) or not isinstance(start_local_date, date):
        raise TypeError("start_local_date must be a date")
    if not _is_integer(warmup_days) or not _is_integer(study_days):
        raise TypeError("warmup_days and study_days must be integers")
    if warmup_days < 0:
        raise ValueError("warmup_days must be non-negative")
    if study_days <= 0:
        raise ValueError("study_days must be positive")

    study_start_local = datetime.combine(start_local_date, time(STUDY_START_LOCAL_HOUR), _LONDON)
    first_utc = study_start_local.astimezone(timezone.utc) - 48 * warmup_days * _HALF_HOUR
    interval_count = 48 * (warmup_days + study_days)
    return tuple(first_utc + index * _HALF_HOUR for index in range(interval_count + 1))


def session_night_dates(interval_start_utc: pd.DatetimeIndex) -> np.ndarray:
    """London date of the session night each slot belongs to, as ``datetime64[D]``.

    A session night runs from London 12:00 on its date to 12:00 the next day
    (decision 0004 item 52), so the night's date is the London date of the
    slot start minus 12 hours: Monday 23:30 and Tuesday 03:00 both belong to
    Monday night.  Reporting days, day types and the trading ledger's nights
    all key on this date, so a night's evening peak and its overnight
    charging are never split across two reporting days.  ``interval_start_utc``
    must be timezone-aware.  Wall-clock arithmetic (London time, then minus
    12 h) keeps the boundary at London noon on both sides of a clock change.
    """

    london = pd.DatetimeIndex(interval_start_utc).tz_convert(_LONDON).tz_localize(None)
    shifted = london - pd.Timedelta(hours=STUDY_START_LOCAL_HOUR)
    return shifted.to_numpy(dtype="datetime64[ns]").astype("datetime64[D]")


def london_wall_time_to_utc(
    local_midnight: np.ndarray, minutes_after_midnight: np.ndarray
) -> np.ndarray:
    """Return UTC instants (``datetime64[ns]``, naive UTC) for London clock times.

    ``local_midnight`` holds London calendar dates as ``datetime64`` and
    ``minutes_after_midnight`` the wall-clock minutes after that midnight;
    the two broadcast together.

    The samplers state behaviour as "leave at 08:00 London plus a shift", so
    this adds the minutes to the wall clock and only then applies that
    instant's London offset.  It reproduces, for whole arrays at once, what
    ``datetime.combine(day, time(0), tzinfo=London) + timedelta(minutes=m)``
    gives one value at a time (the per-value loop was most of the sampling
    time, audit item 2.2):

    - minutes are rounded to whole microseconds, as ``timedelta`` does;
    - a clock time in the autumn repeated hour (01:00-02:00) takes the
      first, BST occurrence, as Python's ``fold=0`` does;
    - a clock time in the spring gap (01:00-02:00, which never happens on
      the wall) takes the offset from before the change (GMT), again as
      ``fold=0`` does, which pandas expresses as "shift forward one hour
      and read it as BST".
    """

    microseconds = np.rint(np.asarray(minutes_after_midnight, dtype=np.float64) * 60_000_000.0)
    local = np.asarray(local_midnight).astype("datetime64[ns]") + (
        microseconds.astype(np.int64) * 1_000
    ).astype("timedelta64[ns]")
    flat = pd.DatetimeIndex(local.ravel())
    utc = (
        flat.tz_localize(
            _LONDON,
            ambiguous=np.ones(len(flat), dtype=bool),
            nonexistent=pd.Timedelta(hours=1),
        )
        .tz_convert("UTC")
        .tz_localize(None)
    )
    return utc.to_numpy(dtype="datetime64[ns]").reshape(local.shape)


def holiday_flags(
    settings: RunSettings, *, bank_holiday_monday: bool, half_term_week: bool
) -> dict[str, np.ndarray]:
    """Which sampled London dates are holiday evenings, and their skip-logit shift.

    Returns ``holiday`` (bool) and ``skip_logit_shift`` (float64, logit
    units), one value per sampled date ``d = 0 … settings.sampled_day_count
    − 1`` (warm-up dates first; date ``d`` is ``start_local_date −
    warmup_days + d``).  Trading contract v1 §10.1c: ``bank_holiday_monday``
    marks the first Monday that is a study evening (none if the study has no
    Monday evening); ``half_term_week`` marks every study evening.  A
    holiday evening's shift is ``assumptions.HOLIDAY_SKIP_LOGIT_SHIFT``.

    Holidays act on the plug-in skip share only (lead decision, 29 September
    2026): no trip, clock or planner deadline reads these flags, and they
    change no draw, only the share a plug uniform is compared with.  A date
    both switches mark is one holiday and takes the shift once, because a
    bank-holiday Monday inside half-term is not two holidays.
    """

    first_date = settings.start_local_date - timedelta(days=settings.warmup_days)
    dates = [first_date + timedelta(days=d) for d in range(settings.sampled_day_count)]
    study_evenings = range(settings.warmup_days, settings.warmup_days + settings.study_days)
    holiday = np.zeros(settings.sampled_day_count, dtype=bool)
    if half_term_week:
        holiday[list(study_evenings)] = True
    if bank_holiday_monday:
        mondays = [d for d in study_evenings if dates[d].weekday() == 0]
        if mondays:
            holiday[mondays[0]] = True
    return {
        "holiday": holiday,
        "skip_logit_shift": np.where(holiday, HOLIDAY_SKIP_LOGIT_SHIFT, 0.0),
    }


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
