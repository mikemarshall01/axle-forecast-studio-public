"""The two-row page header every page starts with (layout ruling 1, 30 Sep 2026).

Row 1: title with the screen's question under it (left) and the view's own
controls slot (right) -- Day, time, Metric, EV, filters, whatever that view
draws. Row 2, only on a page with lenses, is the lens control alone across
the header's full width. Splitting the lens bar onto its own row replaces
the earlier three-column layout (title | lens | controls): a lens with
several or long tabs (Supplier's six, one 30-character label) wrapped onto
two lines even at 1440 px when the lens column held only a third of the
row, and a lens whose view left the controls slot empty needed a second
rule just to fold that width into its own column instead of stopping
mid-screen. One row of full width fixes both without a per-lens exception
list.

How a view reaches the controls slot without changing its
``render_<name>(st, result)`` signature: ``pages.render_page`` opens the
header, then renders the view inside ``controls_slot(column)``. While that
block runs, ``controls(st)`` returns the header's controls column; outside it
(a view rendered on its own in a test) it returns ``st`` itself, so the
control simply draws in place. The slot is held per thread because Streamlit
runs each session's script on its own thread, so two sessions never see each
other's slot. A keyword argument on every view was the alternative; it was
rejected because only some views have controls and ``pages.VIEWS`` would then
need a different call per view.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

_SLOT = threading.local()

_TITLE_CONTROLS = (2.4, 5.6)
"""Row 1's column widths: title/question (narrower -- one line of text) and
the view's controls slot (wider -- room for a Day/Metric/EV selector or two).
One ratio for every page, lensed or not: the lens bar now has its own row
(module docstring), so row 1 never shares width with a lens column."""


def header_row(
    st: Any, *, has_lens: bool, max_width_px: int | None = None
) -> tuple[Any, Any, Any | None]:
    """Open the two-row header and return its ``(title, controls, lens_row)`` slots.

    Row 1 is ``st.columns(_TITLE_CONTROLS)``: title+question on the left, the
    view's own controls on the right. Row 2, ``lens_row``, is a plain
    container holding the page's lens control on its own full-width line;
    it is ``None`` for a page with one view (Overview, Compare), which has
    no lens to show.

    ``max_width_px`` caps both rows to a narrower reading column instead of
    the header's ordinary full content width: How it works (layout ruling
    2) puts its title, lens bar, pipeline diagram and prose in one such
    column, sharing one left edge and one maximum width, because a lens bar
    stretched wider than the text below it reads as unrelated to that text.
    Every other page passes ``None``.
    """

    reading_column = nullcontext() if max_width_px is None else st.container(width=max_width_px)
    with reading_column:
        title, controls_column = st.columns(_TITLE_CONTROLS, vertical_alignment="center")
        lens_row = st.container() if has_lens else None
    return title, controls_column, lens_row


def render_title(column: Any, title: str, question: str) -> None:
    """The page title, then the one question this screen answers as a caption."""

    column.subheader(title, anchor=False)
    column.caption(question)


@contextmanager
def controls_slot(column: Any) -> Iterator[None]:
    """Make ``column`` the controls slot while a view renders (see module docstring)."""

    _SLOT.column = column
    try:
        yield
    finally:
        # Cleared even when a view stops the script (a rerun), so a later
        # direct render never draws into a column from an old script run.
        _SLOT.column = None


def controls(st: Any) -> Any:
    """Where a view draws its header controls: the slot, or ``st`` outside a page render."""

    column = getattr(_SLOT, "column", None)
    return st if column is None else column
