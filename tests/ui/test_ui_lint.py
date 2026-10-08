"""No colour, ``px`` length or chart font size is spelt inline in the UI, bar recorded exceptions.

``scripts/ui_lint.py`` scans ``src/axle_studio/ui`` (except ``style.py``,
the token module) for hex and rgb(a) colours, ``px`` lengths and hard-coded
Plotly font sizes. ``ACCEPTED`` records the current inline values with a
reason; each is a candidate for the next polish pass (docs/DESIGN_TOKENS.md).
The test fails on any new literal and on any listed literal that has gone,
so the list cannot go stale.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from ui_lint import lint_source, lint_ui  # noqa: E402

# (path under ui/, rule, literal) -> (count, reason). Line numbers are not
# matched because other tasks move code around.
ACCEPTED: dict[tuple[str, str, str], tuple[int, str]] = {
    ("views/archetypes.py", "hex-colour", "#9AA3B2"): (
        2,
        "Subplot annotation ink equal to MUTED_INK, spelt inline; should reference the token.",
    ),
    ("views/drivers_fleet.py", "hex-colour", "#5B6472"): (
        1,
        "MUTED_GREY duplicates LIGHT_MUTED_INK's value for de-emphasised cohort lines; "
        "should become a named token in style.py.",
    ),
    ("views/plug_ins.py", "rgba-colour", "rgba(14,17,23,0.85)"): (
        3,
        "Knock-out box behind annotations uses the theme canvas (#0E1117) at 0.85; "
        "style.py has no canvas token yet.",
    ),
    ("views/methods.py", "px-length", "520px"): (
        1,
        "Media query breakpoint for the inline SVG pipeline diagram; the only CSS outside APP_CSS.",
    ),
    ("views/action_decision.py", "font-size", "12 (use CHART_FONT_PX)"): (
        1,
        "Subplot title size equals CHART_FONT_PX but is spelt as 12.",
    ),
    ("views/archetypes.py", "font-size", "12 (use CHART_FONT_PX)"): (
        3,
        "Subplot title and 'No EVs sampled' annotation sizes equal CHART_FONT_PX, spelt as 12.",
    ),
    ("views/sessions.py", "font-size", "12 (use CHART_FONT_PX)"): (
        1,
        "Subplot title size equals CHART_FONT_PX but is spelt as 12.",
    ),
}


def test_ui_source_uses_tokens_or_a_recorded_exception() -> None:
    found = Counter((f.path, f.rule, f.literal) for f in lint_ui())
    expected = Counter({key: count for key, (count, _) in ACCEPTED.items()})
    new = found - expected
    gone = expected - found
    assert not new, f"new inline values, use style.py tokens: {dict(new)}"
    assert not gone, f"delete these from ACCEPTED, they no longer occur: {dict(gone)}"


def test_the_rules_catch_what_they_describe() -> None:
    source = (
        'colour = "#ABCDEF"  # a hex in a comment: #123456 is ignored\n'
        'fill = "rgba(1,2,3,0.5)"\n'
        'css = "<style>.x{margin:4px}</style>"\n'
        "figure.update_annotations(font_size=16)\n"
        'trace(font={"size": 11, "color": INK})\n'
        'marker={"size": 9}\n'
    )
    rules = [(f.rule, f.literal) for f in lint_source("views/example.py", source)]
    assert rules == [
        ("hex-colour", "#ABCDEF"),
        ("rgba-colour", "rgba(1,2,3,0.5)"),
        ("px-length", "4px"),
        ("font-size", "16 (use CHART_FONT_PX)"),
        ("font-size", "11 (use CHART_FONT_PX)"),
    ]
