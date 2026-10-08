"""The one KPI tile every dashboard number uses (polish plan G5, decision 0004 item 54).

A tile is a 12 px muted label, a ~26 px value in tabular numerals with an
optional unit beside it, and an optional one-line context under it (a CNZ
reference, a P10–P90 range). It replaces ``st.metric``, whose 36 px values
were sized for one short number and wrapped or truncated when a value carried
prose, and whose delta always drew a direction arrow.

Rules the helpers enforce: at most four tiles per row (``kpi_columns``) and
text-valued facts ("Smart charging", "0% by construction") go in a compact
badge line (``badge_line``), not a tile. Styles live in ``style.APP_CSS``.
This module only formats and lays out values a view already looked up.
"""

from __future__ import annotations

from html import escape
from typing import Any

MAX_KPIS_PER_ROW = 4


def kpi_html(label: str, value: str, unit: str | None = None, context: str | None = None) -> str:
    """Return the tile's HTML: label, value (+ unit) and an optional context line."""

    unit_html = f'<span class="axle-kpi-unit">{escape(unit)}</span>' if unit else ""
    context_html = f'<div class="axle-kpi-context">{escape(context)}</div>' if context else ""
    return (
        '<div class="axle-kpi">'
        f'<div class="axle-kpi-label">{escape(label)}</div>'
        f'<div class="axle-kpi-value">{escape(value)}{unit_html}</div>'
        f"{context_html}</div>"
    )


def kpi(
    st: Any,
    label: str,
    value: str,
    unit: str | None = None,
    context: str | None = None,
    help: str | None = None,  # noqa: A002 - mirrors Streamlit's own parameter name
) -> None:
    """Render one KPI tile into ``st`` (the page or one column).

    ``label`` names the quantity without its unit; ``value`` is the already
    formatted number (``style.money``, ``percent``, ``kw``, ``format_quantity``);
    ``unit`` sits beside it in smaller type when the value carries none;
    ``context`` is one short line under it; ``help`` becomes the tooltip.
    """

    # width="content" keeps the help icon beside the tile's text instead of
    # at the far edge of a wide column.
    st.markdown(
        kpi_html(label, value, unit, context), unsafe_allow_html=True, help=help, width="content"
    )


def kpi_columns(st: Any, count: int) -> tuple[Any, ...]:
    """Return ``count`` columns for a KPI row, refusing more than four.

    Four is the most that stay legible side by side at 1440 px and stack
    cleanly at 390 px (polish plan G5); a fifth figure belongs in a table.
    """

    if not 1 <= count <= MAX_KPIS_PER_ROW:
        raise ValueError(f"A KPI row holds 1 to {MAX_KPIS_PER_ROW} tiles, not {count}.")
    return tuple(st.columns(count))


def badge_line(
    st: Any,
    label: str,
    text: str,
    help: str | None = None,  # noqa: A002 - mirrors Streamlit's own parameter name
) -> None:
    """Render a text-valued fact as ``label`` plus a grey badge on one compact line.

    Used where a tile would put prose in a 26 px value ("Action chosen: Smart
    charging"). ``text`` is shown as-is inside the badge; square brackets are
    escaped so they cannot close Streamlit's badge directive early.
    """

    safe = text.replace("[", "\\[").replace("]", "\\]")
    st.markdown(f"{label} :gray-badge[{safe}]", help=help)
