"""Test helper: read KPI tiles back out of a recording fake ``st``.

``components.kpi.kpi`` renders each tile as one ``st.markdown`` call holding
``kpi_html``; this parses those calls so view tests can assert on the label,
value, unit, context and help of each tile in render order.
"""

from __future__ import annotations

import re
from html import unescape

_PART = {
    name: re.compile(rf'<(?:div|span) class="axle-kpi-{name}">(.*?)</(?:div|span)>', re.S)
    for name in ("label", "value", "unit", "context")
}
_UNIT = re.compile(r'<span class="axle-kpi-unit">.*?</span>', re.S)


def _part(name: str, body: str) -> str | None:
    match = _PART[name].search(body)
    return unescape(match.group(1)) if match else None


def kpi_calls(st) -> list[tuple[str, str, str | None, str | None, str | None]]:
    """Return ``(label, value, unit, context, help)`` for every KPI tile ``st`` recorded."""

    tiles = []
    for name, args, kwargs in st.calls:
        if name != "markdown" or not args or 'class="axle-kpi"' not in str(args[0]):
            continue
        body = str(args[0])
        value_match = re.search(r'<div class="axle-kpi-value">(.*?)</div>', body, re.S)
        value = unescape(_UNIT.sub("", value_match.group(1))) if value_match else ""
        tiles.append(
            (
                _part("label", body),
                value,
                _part("unit", body),
                _part("context", body),
                kwargs.get("help"),
            )
        )
    return tiles
