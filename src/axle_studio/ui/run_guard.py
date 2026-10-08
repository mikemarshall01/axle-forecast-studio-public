"""Limits on full runs for a hosted demo, set by environment variables.

A full default run holds about 2 GB of memory and several CPU cores for a
few minutes (measured in the Docker image: two concurrent default runs
peaked at 4.4 GB beside the 0.7 GB idle app). On a laptop that is the
user's own choice, so both limits are off unless an environment variable
sets them. On a shared hosted demo, several visitors clicking Run at once
could exhaust the machine's memory and stop the app for everyone, so the
host sets:

``AXLE_MAX_CONCURRENT_RUNS``  full runs allowed at once across all sessions of
                              this server process (a positive whole number).
                              A Run click beyond it is refused at once with a
                              plain message; it is not queued, because a
                              queued click would hold a script thread and the
                              viewer's page for minutes with no feedback.
``AXLE_MAX_EV_WEEKS``         the largest fleet size x simulated weeks one run
                              may use (a positive whole number). A larger
                              draft is refused with a message pointing to the
                              downloadable repository.

Unset or empty means no limit. Anything else that is not a positive whole
number raises ``ValueError`` at the Run click, so a misconfigured host fails
loudly rather than running unguarded.

Why a process-wide semaphore and not a queue or a worker pool: Streamlit runs
each session's script in a thread of one process, so one
``threading.BoundedSemaphore`` shared through ``st.cache_resource`` is enough
to count runs across sessions; nothing here runs the model.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import streamlit as st

MAX_CONCURRENT_RUNS_ENV = "AXLE_MAX_CONCURRENT_RUNS"
MAX_EV_WEEKS_ENV = "AXLE_MAX_EV_WEEKS"

BUSY_MESSAGE = (
    "Someone else's run is in progress on this hosted demo. Try again in a few minutes. "
    "The current result stays on screen."
)


def env_limit(name: str) -> int | None:
    """The positive whole-number limit in environment variable ``name``, or ``None`` if unset."""

    text = os.environ.get(name, "").strip()
    if not text:
        return None
    try:
        limit = int(text)
    except ValueError:
        limit = 0
    if limit < 1:
        raise ValueError(f"{name} must be a positive whole number or unset, not {text!r}")
    return limit


@st.cache_resource(show_spinner=False)
def _run_slots(limit: int) -> threading.BoundedSemaphore:
    """One semaphore per limit value, shared by every session of this server process."""

    return threading.BoundedSemaphore(limit)


@contextmanager
def run_slot() -> Iterator[bool]:
    """Hold one of the process's run slots for the ``with`` block.

    Yields ``True`` when the run may go ahead (a slot was free, or no limit
    is set) and ``False`` when every slot is taken. It never waits: a busy
    server refuses the click (see the module docstring).
    """

    limit = env_limit(MAX_CONCURRENT_RUNS_ENV)
    if limit is None:
        yield True
        return
    slots = _run_slots(limit)
    if not slots.acquire(blocking=False):
        yield False
        return
    try:
        yield True
    finally:
        slots.release()


def size_refusal(values: dict[str, Any]) -> str | None:
    """Why this draft is too large for the hosted demo, or ``None`` when it may run.

    ``values`` are the draft's editable assumption values by record name; the
    run size is fleet size (EVs) x simulated weeks, in EV-weeks.
    """

    limit = env_limit(MAX_EV_WEEKS_ENV)
    if limit is None:
        return None
    vehicles = int(values["vehicle_count"])
    weeks = int(values["evaluation_world_count"])
    if vehicles * weeks <= limit:
        return None
    return (
        f"This hosted demo limits runs to {limit:,} EV-weeks (fleet size x simulated weeks). "
        f"This draft is {vehicles:,} EVs x {weeks:,} weeks = {vehicles * weeks:,}. "
        "Reduce the fleet size or the weeks, or download the repository to run full-size "
        "forecasts."
    )


def hosted_run_size(values: dict[str, Any], limit: int) -> tuple[int, int]:
    """The largest (fleet size, weeks) pair within the hosted EV-weeks cap ``limit``.

    Used by the run bar's "Run at the hosted size" button, offered after a
    ``size_refusal``, so the refused visitor has a one-click way to see a
    real result instead of downloading the repository. Mike's decision for
    the current cap (15,000 EV-weeks; overnight review log, 30 Sep) is 300
    EVs x 50 weeks: it fills the cap exactly and is large enough to show
    cross-fleet variety and most of a year of weeks. Rather than naming
    300/50 as a literal pair (which would silently break under a different
    ``AXLE_MAX_EV_WEEKS``), this derives the same pair from that decision:
    cap weeks at 50 (never above the draft's own weeks or the limit itself),
    then cap fleet size at 300 (never above the draft's own fleet size or
    what the remaining budget allows at that many weeks). A smaller host
    cap therefore still returns the largest pair that fits, rather than one
    that exceeds it.
    """

    weeks = max(1, min(int(values["evaluation_world_count"]), 50, limit))
    vehicles = max(1, min(int(values["vehicle_count"]), 300, limit // weeks))
    return vehicles, weeks


__all__ = [
    "BUSY_MESSAGE",
    "MAX_CONCURRENT_RUNS_ENV",
    "MAX_EV_WEEKS_ENV",
    "env_limit",
    "hosted_run_size",
    "run_slot",
    "size_refusal",
]
