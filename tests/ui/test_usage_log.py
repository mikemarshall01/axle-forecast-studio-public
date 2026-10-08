"""Anonymous usage logging (ui/usage_log.py): off by default, JSON shape, never raises.

Purpose (Mike, hosted-demo usage logging, 30 Sep 2026): see how the hosted
demo gets used -- pages, lenses, runs, downloads -- without collecting
anything that could identify a visitor. These tests check the module's own
contract (off unless ``AXLE_USAGE_LOG=1``, the exact JSON shape, no way for
an IP/header/cookie/fingerprint field to appear, never raises) and, through
one real AppTest run, that a page view and a lens change each emit exactly
one ``view`` event while an unrelated rerun emits none.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from axle_studio.ui import usage_log

APP_PATH = Path(__file__).parents[2] / "streamlit_app.py"

# Anything from this family must never appear in a logged line, whatever a
# caller passes: the module adds only ts/sid/event, and no hook in the app
# passes anything identifying (AGENTS.md; the brief for this task).
_FORBIDDEN_KEYS = {
    "ip",
    "ip_address",
    "remote_addr",
    "user_agent",
    "headers",
    "cookie",
    "cookies",
    "fingerprint",
}


class _ListHandler(logging.Handler):
    """Records formatted log lines instead of writing them anywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> _ListHandler:
    """Replace ``usage_log``'s real stdout handler with one that records lines."""

    handler = _ListHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    monkeypatch.setattr(usage_log._logger, "handlers", [handler])
    return handler


def test_off_by_default(captured: _ListHandler, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(usage_log.ENABLED_ENV, raising=False)

    usage_log.log_event({}, "view", page="Overview", lens="At a glance")

    assert captured.lines == []


@pytest.mark.parametrize("off_value", ["0", "true", "yes", ""])
def test_off_for_any_value_other_than_one(
    captured: _ListHandler, monkeypatch: pytest.MonkeyPatch, off_value: str
) -> None:
    monkeypatch.setenv(usage_log.ENABLED_ENV, off_value)

    usage_log.log_event({}, "view", page="Overview", lens="At a glance")

    assert captured.lines == []


def test_json_shape_and_fields(captured: _ListHandler, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(usage_log.ENABLED_ENV, "1")

    usage_log.log_event({}, "view", page="Overview", lens="At a glance")

    assert len(captured.lines) == 1
    record = json.loads(captured.lines[0])
    assert set(record) == {"ts", "sid", "event", "page", "lens"}
    assert record["event"] == "view"
    assert record["page"] == "Overview"
    assert record["lens"] == "At a glance"
    assert re.fullmatch(r"[0-9a-f]{8}", record["sid"])
    # ts round-trips as a UTC-aware ISO 8601 timestamp.
    from datetime import datetime

    parsed = datetime.fromisoformat(record["ts"])
    assert parsed.utcoffset().total_seconds() == 0


def test_session_id_is_stable_within_a_session_and_random_across_sessions(
    captured: _ListHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(usage_log.ENABLED_ENV, "1")
    state_a: dict = {}
    state_b: dict = {}

    usage_log.log_event(state_a, "session_start")
    usage_log.log_event(state_a, "run_requested", fleet_size=1000, weeks=100, refused=False)
    usage_log.log_event(state_b, "session_start")

    sid_a1, sid_a2, sid_b = (json.loads(line)["sid"] for line in captured.lines)
    assert sid_a1 == sid_a2
    assert sid_a1 != sid_b


def test_no_forbidden_identifying_field_in_any_event_this_app_logs(
    captured: _ListHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One line per event kind actually used in the app (pages.py,
    # run_controller.py, views/compare.py, the three CSV download sites):
    # never more than ts/sid/event plus the named fields below.
    monkeypatch.setenv(usage_log.ENABLED_ENV, "1")
    state: dict = {}

    usage_log.log_event(state, "session_start")
    usage_log.log_event(state, "view", page="Drivers", lens="Plug-ins")
    usage_log.log_event(state, "run_requested", fleet_size=1000, weeks=100, refused=False)
    usage_log.log_event(state, "run_finished", seconds=12.3)
    usage_log.log_event(state, "edit_assumptions_opened")
    usage_log.log_event(state, "csv_download", which="plug-in events")
    usage_log.log_event(state, "compare_viewed")

    for line in captured.lines:
        record = json.loads(line)
        assert not _FORBIDDEN_KEYS & set(record)
        assert set(record) <= {
            "ts",
            "sid",
            "event",
            "page",
            "lens",
            "fleet_size",
            "weeks",
            "refused",
            "seconds",
            "which",
        }


def test_never_raises_with_a_state_that_has_no_get(
    captured: _ListHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(usage_log.ENABLED_ENV, "1")

    usage_log.log_event(object(), "view", page="Overview", lens="At a glance")  # must not raise

    assert captured.lines == []


def test_never_raises_when_the_logging_backend_fails(
    captured: _ListHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(usage_log.ENABLED_ENV, "1")

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic backend failure")

    monkeypatch.setattr(usage_log._logger, "info", _boom)

    usage_log.log_event({}, "view", page="Overview", lens="At a glance")  # must not raise


# --- Real app: view/lens-change dedup (AppTest) ------------------------------


def _go(app: AppTest, page_name: str) -> None:
    app._page_hash = next(
        page_hash
        for page_hash, page in app._registered_pages.items()
        if page["page_name"] == page_name
    )
    app.run()
    assert not app.exception


def _view_events(handler: _ListHandler) -> list[dict]:
    return [json.loads(line) for line in handler.lines if json.loads(line)["event"] == "view"]


def test_app_test_a_page_view_and_a_lens_change_each_log_one_view_event(
    captured: _ListHandler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(usage_log.ENABLED_ENV, "1")

    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    # The landing page's first lens, on open: exactly one view event so far.
    assert len(_view_events(captured)) == 1

    # An unrelated rerun of the same page/lens logs no further view event.
    app.run()
    assert len(_view_events(captured)) == 1

    # Switching page logs exactly one more view event.
    _go(app, "Overview")
    assert len(_view_events(captured)) == 2
    assert _view_events(captured)[-1] == {
        **_view_events(captured)[-1],
        "page": "Overview",
        "lens": "At a glance",
    }

    # Changing lens on the same page logs exactly one more.
    app.get_by_key("lens::overview").set_value("Key stats").run()
    assert not app.exception
    events = _view_events(captured)
    assert len(events) == 3
    assert events[-1]["page"] == "Overview"
    assert events[-1]["lens"] == "Key stats"
