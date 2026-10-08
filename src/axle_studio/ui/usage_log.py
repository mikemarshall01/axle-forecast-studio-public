"""Anonymous usage logging for the hosted demo.

This module owns turning a handful of shell and view events (a page view, a
run, a download) into one JSON line on stdout, so Mike can see how the
hosted demo gets used -- which pages and lenses people open, whether they
run a simulation, open Edit assumptions, download a CSV, and roughly how
long a session lasts -- without collecting anything that could identify a
visitor. Decision (Mike, hosted-demo usage logging, 30 Sep 2026): anonymous
and lawful means no IP address, no user agent, no request headers, no
cookies, no fingerprinting and no free-text user input. Only a random
per-session id, a UTC timestamp and the event name are logged by this
module itself; call sites add a few short numeric or categorical fields
(fleet size, a page name, which download), never anything a viewer typed.

Off unless the host sets ``AXLE_USAGE_LOG=1``, so an ordinary local run or a
test session stays silent by default -- this is a hosted-demo feature, not
something a developer needs switched on to work on the app.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

ENABLED_ENV = "AXLE_USAGE_LOG"
"""Set to ``"1"`` to turn logging on. Any other value, or unset, stays silent."""

_LOGGER_NAME = "axle.usage"
_SESSION_ID_KEY = "usage_session_id"

_logger = logging.getLogger(_LOGGER_NAME)
_logger.propagate = False
if not _logger.handlers:
    # Added once per process: a module-level guard, not a per-call check, so
    # a Streamlit script rerun (this module is imported once, then reused)
    # never ends up with a second handler doubling every line.
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))  # prefix-free: the message is the JSON
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO)


def _session_id(state: Any) -> str:
    """This session's anonymous id: the first 8 hex characters of a fresh uuid4.

    Stored in ``st.session_state`` (``state`` here) so it is stable for the
    life of the browser tab, and created on this session's first logged
    event. Never derived from anything identifying -- no IP, no cookie, no
    browser fingerprint -- so two sessions from the same visitor are
    indistinguishable by design, which is the point (purpose: anonymous
    usage counts, not visitor tracking).
    """

    sid = state.get(_SESSION_ID_KEY)
    if not sid:
        sid = uuid.uuid4().hex[:8]
        state[_SESSION_ID_KEY] = sid
    return sid


def log_event(state: Any, event: str, **fields: Any) -> None:
    """Write one JSON usage line if logging is on; otherwise do nothing.

    ``state`` is ``st.session_state`` (holds the per-session id).  ``event``
    is a short name such as ``"view"`` or ``"run_finished"``; ``fields`` are
    the event's own small set of numeric or short categorical values (see
    the call sites in ``pages.py`` and ``run_controller.py`` for what each
    event carries). The line always has exactly ``ts`` (UTC, ISO 8601),
    ``sid`` (the session id above) and ``event``, plus ``fields`` -- never
    an IP, a header, a cookie or free text.

    Never raises: this is a side channel for Mike's own analytics, and a
    logging failure (a bad field type, a full pipe) must not break the page
    render or run it was called from.
    """

    if os.environ.get(ENABLED_ENV) != "1":
        return
    try:
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "sid": _session_id(state),
            "event": event,
            **fields,
        }
        _logger.info(json.dumps(record, default=str))
    except Exception:
        pass


__all__ = ["ENABLED_ENV", "log_event"]
