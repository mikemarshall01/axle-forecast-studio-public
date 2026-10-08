"""The page header: its controls slot and its two-row layout (components/header.py,
layout ruling 1, 30 Sep 2026).
"""

from __future__ import annotations

import threading

import pytest

from axle_studio.ui.components import header


def test_controls_are_drawn_in_place_outside_a_page_render() -> None:
    page = object()

    assert header.controls(page) is page


def test_controls_go_to_the_slot_while_a_view_renders_and_the_slot_clears_after() -> None:
    page, column = object(), object()

    with header.controls_slot(column):
        assert header.controls(page) is column
    assert header.controls(page) is page


def test_slot_clears_even_when_the_view_raises() -> None:
    page, column = object(), object()

    with pytest.raises(RuntimeError), header.controls_slot(column):
        raise RuntimeError("view stopped")
    assert header.controls(page) is page


def test_another_thread_never_sees_this_threads_slot() -> None:
    # Streamlit runs each session's script on its own thread.
    page, column = object(), object()
    seen: list[object] = []

    with header.controls_slot(column):
        worker = threading.Thread(target=lambda: seen.append(header.controls(page)))
        worker.start()
        worker.join()

    assert seen == [page]


# --- Two-row layout: title row always, lens row only when the page has one -


class _FakeColumn:
    """A column or container placeholder: itself a no-op context manager.

    Real Streamlit's ``st.columns``/``st.container`` return objects usable as
    ``with column:`` to draw inside them later; this stands in for that
    without a real Streamlit runtime.
    """

    def __enter__(self) -> "_FakeColumn":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


class _FakeStreamlit:
    """Records ``header_row``'s ``columns``/``container`` calls, in order.

    Order is what proves the layout: a ``columns`` call opens the title row,
    and a later, separate ``container`` call (not sharing that row's column
    widths) opens the lens row below it.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def columns(self, widths: object, *, vertical_alignment: str) -> tuple[_FakeColumn, ...]:
        self.calls.append(("columns", tuple(widths), {"vertical_alignment": vertical_alignment}))
        return tuple(_FakeColumn() for _ in widths)

    def container(self, *, width: int | None = None) -> _FakeColumn:
        self.calls.append(("container", (), {"width": width}))
        return _FakeColumn()


def test_row_one_is_the_title_and_controls_split_whether_or_not_there_is_a_lens() -> None:
    st = _FakeStreamlit()

    title, controls, lens_row = header.header_row(st, has_lens=False)

    assert st.calls == [("columns", header._TITLE_CONTROLS, {"vertical_alignment": "center"})]
    assert title is not None
    assert controls is not None
    assert lens_row is None


def test_a_lens_gets_its_own_full_width_row_after_the_title_row() -> None:
    st = _FakeStreamlit()

    _title, _controls, lens_row = header.header_row(st, has_lens=True)

    # The lens row is a second, separate call -- not part of the title row's
    # columns -- so it lays out below the title row, at the header's own
    # full width (no ``width=`` passed).
    assert st.calls == [
        ("columns", header._TITLE_CONTROLS, {"vertical_alignment": "center"}),
        ("container", (), {"width": None}),
    ]
    assert lens_row is not None


def test_max_width_px_wraps_the_title_row_and_the_lens_row_in_one_reading_column() -> None:
    st = _FakeStreamlit()

    header.header_row(st, has_lens=True, max_width_px=720)

    # The outer, capped container opens first, then both rows draw inside
    # it: title/lens bar/pipeline/prose then share one left edge and one
    # maximum width (layout ruling 2).
    assert st.calls == [
        ("container", (), {"width": 720}),
        ("columns", header._TITLE_CONTROLS, {"vertical_alignment": "center"}),
        ("container", (), {"width": None}),
    ]


def test_a_page_without_a_lens_ignores_max_width_px_the_same_way() -> None:
    st = _FakeStreamlit()

    _title, _controls, lens_row = header.header_row(st, has_lens=False, max_width_px=720)

    assert st.calls == [
        ("container", (), {"width": 720}),
        ("columns", header._TITLE_CONTROLS, {"vertical_alignment": "center"}),
    ]
    assert lens_row is None
