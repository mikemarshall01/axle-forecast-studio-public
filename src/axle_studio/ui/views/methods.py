"""How it works: five read-only lenses on why and how the forecast is built and run.

Design authority: docs/DASHBOARD_DESIGN.md section 4.5 ("How it works"),
decision 0004 items 18, 21, 23, 24, 28, 30 and 31, and the assumption record
shape in docs/contracts/results-v2.md section 8. Every lens follows the page
contract ``render_<name>(st, result) -> None`` (design section 6); ``result``
may be ``None`` because this page renders fully before any run (design
section 3.2): the shell lists all five lenses in ``pages.RESULT_OPTIONAL``.

This module never runs the model and never reads Streamlit session state
directly: the run bar, the draft/active-run distinction and the shared Edit
assumptions dialog belong to ``run_controller`` and ``pages`` (the shell).
Where this module needs outside behaviour (the table of default assumption
values before the first run) it takes an injectable callable, so a test can
observe the call without a real Streamlit runtime. The default table is
``model.assumptions.assumption_rows()``.

Lenses, in nav order:

* ``render_why_this_model``: "Why this model", the landing lens. Mike's
  opening note on what the model is for and why it is built this way,
  shown before any mechanics.
* ``render_forecast_method``: "The forecast", the five-step pipeline
  diagram with the two core equations, then the full explainer in an
  expander.
* ``render_assumptions``: the read-only assumptions table, filterable by
  evidence class. Editing goes through the run bar's own "Edit assumptions"
  button (layout ruling 3, 30 Sep 2026), which is on screen on every page.
* ``render_limits``: the "Not modelled" list, "What this model cannot
  answer", known limitations, then the CNZ/Axle-sheet crosswalk reference
  in an expander.
* ``render_build_notes``: Mike's build notes, then the full "how it was
  built" explainer in an expander.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from streamlit import column_config

from axle_studio.model.assumptions import ACTION, COHORT_SOURCE_NAMES, assumption_rows

from ..style import INK, MUTED_INK, SERIES_COLOURS
from .overview import _switch_to

# --------------------------------------------------------------------------
# Explainer and build-notes files
# --------------------------------------------------------------------------
#
# The explainers (decision 0004 item 24) live in docs/explainers/, and the
# Dockerfile copies them into the image. Reading them by a module-level path,
# rather than importing them, means a missing file (a build that left docs
# out) degrades to a short note instead of a crash, and a test can point
# these constants at a temp file.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXPLAINER_DIR = _REPO_ROOT / "docs" / "explainers"
FORECAST_EXPLAINER_PATH = _EXPLAINER_DIR / "how-the-forecast-works.md"
BUILD_EXPLAINER_PATH = _EXPLAINER_DIR / "how-it-was-built.md"
BUILD_NOTES_PATH = _EXPLAINER_DIR / "build-notes.md"
# Mike's opening note, the first thing a reader sees (overnight review log,
# 30 Sep "why-note" ruling): read from its file like the build notes, so his
# own words are shown verbatim and can be edited without touching code.
WHY_THIS_MODEL_PATH = _EXPLAINER_DIR / "why-this-model.md"
# Reference only (section is a short pointer, not a reproduction: the
# crosswalk is long-form migration evidence pending separate approval).
COHORT_CROSSWALK_PATH = _REPO_ROOT / "docs" / "sources" / "cohort-crosswalk.md"
# Reference only, same as the crosswalk above: the full evidence catalogue
# (which values are source, illustrative or synthetic, and why) lives in the
# repository, not reproduced here (polish lane "Start here").
DATA_AND_SOURCES_PATH = _REPO_ROOT / "docs" / "explainers" / "data-and-sources.md"

_EXPLAINER_MISSING = (
    "This explainer is missing from this build: it lives in docs/explainers/ in the repository."
)


def _read_text(path: Path) -> str | None:
    """Return a doc's text, or ``None`` when it has not been written yet."""

    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _render_explainer_expander(st: Any, *, path: Path, label: str) -> None:
    """One expander holding a full explainer, or the "missing" note."""

    text = _read_text(path)
    with st.expander(label):
        st.markdown(text if text is not None else _EXPLAINER_MISSING)


PROSE_WIDTH_PX = 720
"""Running text measure (polish plan G9): about 75-90 characters a line at
14 px, the readable range, instead of three quarters of a 1440 px screen."""


def _prose_column(st: Any) -> Any:
    """A container at most ``PROSE_WIDTH_PX`` wide for running text.

    A fixed-width container rather than ``st.columns([3, 1])``: the column
    grew with the window (about 1,000 px at 1440), and at 390 px a pixel
    width is capped at the screen, so no side column is needed to stack.
    """

    return st.container(width=PROSE_WIDTH_PX)


_PARENTHETICAL = re.compile(r"(\s*)\(([^()]*)\)")
_CITATION_PART = re.compile(r"^\s*(decision \d{4}|contract v\d|docs/|plan \d{4})", re.IGNORECASE)


def split_citations(text: str) -> tuple[str, str | None]:
    """Move internal citations out of on-screen text: ``(clean text, tooltip or None)``.

    Polish plan G7: "decision 0004 item N" and "contract v2 ..." belong in a
    tooltip, not in the sentence a reader reads. The constants below keep
    their citations (the source of truth, and what a code reader checks);
    only the rendered copy is split. Inside each parenthesis, the parts that
    are citations move to the tooltip and any plain-English part stays, so
    "(7 kW by default, editable; decision 0004 item 34)" reads
    "(7 kW by default, editable)".
    """

    citations: list[str] = []

    def keep(match: re.Match[str]) -> str:
        space, inside = match.groups()
        parts = re.split(
            r"[;,](?=\s*(?:decision|contract|docs/|plan \d))", inside, flags=re.IGNORECASE
        )
        plain = [part.strip() for part in parts if not _CITATION_PART.match(part)]
        citations.extend(part.strip() for part in parts if _CITATION_PART.match(part))
        # A parenthesis that was only citations goes, with the space before it.
        return f"{space}({'; '.join(plain)})" if plain else ""

    clean = _PARENTHETICAL.sub(keep, text)
    return clean, ("Source: " + "; ".join(citations) + ".") if citations else None


def _cited_markdown(st: Any, text: str) -> None:
    """``st.markdown`` of ``text`` with its citations in the help tooltip."""

    clean, citations = split_citations(text)
    st.markdown(clean, help=citations)


def _overview_tour_button(st: Any) -> None:
    """The "Next" button on to Overview ▸ At a glance and its five-stop "Start here" tour.

    How it works is the landing page (Fable landing pass), so a first-time
    reader arrives here before Overview's own tour. The opening note and
    The forecast both end with this button rather than leaving the reader
    to find the nav tab themselves.
    """

    st.button(
        "Next: Overview ▸ At a glance has a five-stop tour",
        key="how-it-works-to-overview-tour",
        on_click=_switch_to,
        args=("Overview", "At a glance"),
    )


# --------------------------------------------------------------------------
# Why this model
# --------------------------------------------------------------------------


def render_why_this_model(st: Any, result: Any | None) -> None:
    """ "Why this model" lens: Mike's opening note, then the way on to Overview's tour.

    The note (``WHY_THIS_MODEL_PATH``) says what the model is for and why it
    is built this way, before any mechanics, because it is the first screen
    a reader sees. It is read from its file like the build notes, so Mike's
    own words show verbatim and a missing file degrades to a short note
    rather than a crash.
    """

    del result  # unused: the note describes the model's purpose, not one run's output
    with _prose_column(st):
        note = _read_text(WHY_THIS_MODEL_PATH)
        st.markdown(note if note is not None else _EXPLAINER_MISSING)
    _overview_tour_button(st)


# --------------------------------------------------------------------------
# The forecast
# --------------------------------------------------------------------------

# (title, description, LaTeX equation or None, plain-word reading of the
# equation or None). Five steps: population -> simulated weeks (each draws
# its own trips, plug-ins, weather and, for smart charging, its own
# day-ahead price path) -> smart charging plan (per plug-in, against that
# week's own path; decision 0004 item 38 removed the earlier separate
# planning world) -> physics -> summaries; only the two steps with a stated
# equation carry one. Fable clarity pass: every equation is read out in
# plain words naming each symbol right under it, because LaTeX alone excludes
# a reader who does not read maths notation fluently. The citations stay in
# parentheses so ``split_citations`` can move them to the tooltip.
_FORECAST_STEPS: tuple[tuple[str, str, str | None, str | None], ...] = (
    (
        "1 Population",
        "The fleet is drawn once. Each EV is given one of six archetypes (driver types) in the "
        "shares from Axle's cohort sheet, and each archetype brings its own battery, mileage and "
        "plug-in habit. Every EV shares one home charging power (decision 0004 item 34). The "
        "same fleet is used in every simulated week, so weeks differ in what happened, not in "
        "who the drivers are.",
        None,
        None,
    ),
    (
        "2 Evaluation weeks",
        "A simulated week (an evaluation world) is one possible week for the whole fleet. Each "
        "one draws its own trips, home plug-ins and weather, and, for the smart-charging model, "
        "its own day-ahead price path (decision 0004 items 14, 48). There is no separate "
        "planning world: each week's smart charger plans against that same week's own prices, "
        "not a forecast shared by every week (decision 0004 item 38).",
        None,
        None,
    ),
    (
        "3 Smart charging plan",
        "At each plug-in the charger knows four things: the EV's state of charge (SoC, how full "
        "the battery is), the plug-in time, its expected departure (the archetype's usual "
        "departure time minus a {margin_hours:g} h safety margin, editable in Edit assumptions) "
        "and that week's own day-ahead price path. It plans to reach the preferred target SoC "
        "by that expected departure, filling the cheapest forecast half-hours in the window "
        "first, up to the home charging power. If the window cannot deliver enough energy, it "
        "charges as soon as possible instead (decision 0004 item 38). An EV that leaves earlier "
        "than expected goes short; I price that shortfall as a public top-up or as energy not "
        "recovered (decision 0004 items 4, 5, 13). With intraday dispatch on (the default), a "
        "fixed share of EVs keeps this day-ahead plan; the rest re-plan each hour on the latest "
        "intraday price when the saving beats a threshold (decision 0004 item 62).",
        None,
        None,
    ),
    (
        "4 Physics",
        "Each simulated week then runs the fleet's half-hour battery physics twice: once on the "
        "unmanaged path (charge at full power from plug-in until the target) and once on the "
        "smart path (the plan applied). Both share the same drawn trips, plug-ins, weather and "
        "prices, so any difference between them comes from the smart charging alone, not from "
        "different random futures. No EV runs flat: whenever a trip would take the battery "
        "below the top-up threshold (10% SoC by default), the EV tops up at a public charger "
        "first, to the top-up target (80% SoC), then carries on (decision 0004 items 32 and 37).",
        # Single-letter subscripts (h = home, p = public), not the full words
        # "home"/"pub": KaTeX's rendered width of a word subscript overflowed
        # a 390 px screen (goal review action 21); h/p keep the equation on
        # one line without changing what it says.
        r"E_{t+1} = E_t + \eta_h I_t^{h} + \eta_p I_t^{p} - d_t",
        "In plain words: battery energy at the start of the next half-hour equals energy now, "
        "plus home grid energy times home charging efficiency, plus public top-up energy times "
        "public charging efficiency, minus energy used driving in this half-hour.",
    ),
    (
        "5 Summaries",
        "Every chart adds up inside each simulated week first (a fleet total, or smart minus "
        "unmanaged), and only then takes the P10, P50 and P90 across weeks (decision 0004 item "
        "12). Percentiles are never subtracted, averaged or summed after that.",
        # Q_w(...) rather than \operatorname{quantile}_w(\text{world}_w): the
        # longer form also overflowed a 390 px screen (goal review action 21).
        r"Q_w\big(f(w)\big) \ne f\big(Q_w(w)\big)",
        "In plain words: take the percentile across simulated weeks of a total worked out "
        "inside each week. Never build a total by combining percentiles. For example, the P90 "
        "of each week's fleet total is not the same as adding up each half-hour's own P90.",
    ),
)


# --------------------------------------------------------------------------
# Pipeline diagram (goal review action 21): the one picture that shows how a
# run is calculated, at the module level (docs/CODE_GUIDE.md's reading
# order), a different altitude to the five run-mechanics steps below.
# --------------------------------------------------------------------------

_PIPELINE_STAGES: tuple[tuple[str, str, str], ...] = (
    ("Assumptions", "Every value,", "sourced or labelled"),
    ("Sampling", "Trips, plug-ins,", "weather, prices"),
    ("Physics", "Half-hour battery", "balance per EV"),
    ("Smart charging", "Cheapest forecast", "half-hours"),
    ("Summaries", "Add up each week,", "then quantile"),
    ("Dashboard", "Streamlit shows", "the result"),
)
_SMART_CHARGING_STAGE = "Smart charging"
"""Outlined in teal, the same meaning as the selected path elsewhere (decision 0004 item 27)."""


def _pipeline_box(
    x: float, y: float, w: float, h: float, title: str, line1: str, line2: str
) -> str:
    stroke = SERIES_COLOURS["selected"] if title == _SMART_CHARGING_STAGE else MUTED_INK
    mid_x = x + w / 2
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="8" '
        f'fill="rgba(255,255,255,0.04)" stroke="{stroke}" stroke-width="1.2"/>'
        f'<text x="{mid_x:g}" y="{y + h * 0.32:g}" text-anchor="middle" fill="{INK}" '
        f'font-size="12" font-weight="600">{title}</text>'
        f'<text x="{mid_x:g}" y="{y + h * 0.60:g}" text-anchor="middle" fill="{MUTED_INK}" '
        f'font-size="10">{line1}</text>'
        f'<text x="{mid_x:g}" y="{y + h * 0.82:g}" text-anchor="middle" fill="{MUTED_INK}" '
        f'font-size="10">{line2}</text>'
    )


def _pipeline_arrow(x: float, y: float) -> str:
    return (
        f'<path d="M {x:g} {y:g} l 18 0 l -6 -5 m 6 5 l -6 5" '
        f'stroke="{MUTED_INK}" stroke-width="1.4" fill="none"/>'
    )


def _pipeline_diagram_svg() -> str:
    """Six-stage pipeline as inline SVG.

    Decision 0004 item 29 is a Plotly rule for charts; a static diagram with
    no data would be an odd fit for a chart component, so this is plain SVG
    instead (goal review action 21 names it as the alternative).

    Two fixed layouts, swapped by a CSS breakpoint, rather than one shape
    scaled to the viewport: scaling a single row down to 390 px either clips
    six boxes or shrinks the text below reading size (decision 0004 item 26,
    "nothing clipped or squashed"). Desktop keeps arrows a reader's eye can
    follow left to right; mobile drops them for a 2x3 grid read in the same
    order, numbered so the sequence still reads without an arrow.
    """

    box_w, box_h, gap = 150.0, 90.0, 24.0
    row: list[str] = []
    for index, (title, line1, line2) in enumerate(_PIPELINE_STAGES):
        x = 10 + index * (box_w + gap)
        row.append(_pipeline_box(x, 15, box_w, box_h, title, line1, line2))
        if index < len(_PIPELINE_STAGES) - 1:
            row.append(_pipeline_arrow(x + box_w + 3, 15 + box_h / 2))
    desktop_width = 10 + len(_PIPELINE_STAGES) * (box_w + gap)
    desktop = (
        f'<svg class="pipeline-desktop" viewBox="0 0 {desktop_width:g} 120" '
        'preserveAspectRatio="xMinYMid meet" '
        'width="100%" height="120" role="img" '
        'aria-label="Pipeline: assumptions, sampling, physics, smart charging, '
        'summaries, dashboard">' + "".join(row) + "</svg>"
    )

    cols, mob_w, mob_h, mob_gap = 2, 150.0, 78.0, 16.0
    grid: list[str] = []
    for index, (title, line1, line2) in enumerate(_PIPELINE_STAGES):
        col, rownum = index % cols, index // cols
        x = 10 + col * (mob_w + mob_gap)
        y = 10 + rownum * (mob_h + mob_gap)
        grid.append(_pipeline_box(x, y, mob_w, mob_h, title, line1, line2))
        grid.append(
            f'<text x="{x + 10:g}" y="{y + 14:g}" fill="{MUTED_INK}" font-size="9">'
            f"{index + 1}</text>"
        )
    mobile_width = 10 + cols * (mob_w + mob_gap)
    mobile_height = 10 + 3 * (mob_h + mob_gap)
    mobile = (
        f'<svg class="pipeline-mobile" viewBox="0 0 {mobile_width:g} {mobile_height:g}" '
        'preserveAspectRatio="xMinYMin meet" '
        'width="100%" height="300" role="img" '
        'aria-label="Pipeline: assumptions, sampling, physics, smart charging, '
        'summaries, dashboard">' + "".join(grid) + "</svg>"
    )

    return (
        "<style>.pipeline-mobile{display:none}"
        "@media (max-width: 520px){.pipeline-desktop{display:none}"
        ".pipeline-mobile{display:block}}</style>" + desktop + mobile
    )


def render_forecast_method(st: Any, result: Any | None) -> None:
    """ "The forecast" lens: the pipeline diagram, the five-step walkthrough, then the explainer.

    The pipeline description is the same for every run, so this lens renders
    identically before and after the first Run (design section 3.2), with one
    exception: step 3's departure margin is editable (``ACTION[
    "departure_margin_hours"]``, bounds 0-6 h), so a fixed "one-hour" figure
    in the text drifted from the 2 h default and from whatever margin an
    active run actually used (final critique B-13). Reading the run's own
    value when one exists, and the dialog default otherwise (before the
    first run, or on a no-action result, which has no ``action_summary``),
    keeps this sentence honest without hard-coding either.
    """

    action_summary = result.action_summary if result is not None else None
    margin_hours = (
        action_summary.departure_margin_hours
        if action_summary is not None
        else ACTION["departure_margin_hours"].value
    )
    # In its own prose column, not full width (layout ruling 2): the diagram
    # used to render wider than the reading column below it (1030 px of
    # boxes against a 720 px prose measure), so the two no longer shared a
    # right edge even though they shared a left one.
    with _prose_column(st):
        st.markdown(_pipeline_diagram_svg(), unsafe_allow_html=True)
    with _prose_column(st):
        for title, description, equation, equation_plain in _FORECAST_STEPS:
            text = description.format(margin_hours=margin_hours)
            _cited_markdown(st, f"**{title}.** {text}")
            if equation:
                st.latex(equation)
            if equation_plain:
                st.markdown(equation_plain)

    _render_explainer_expander(st, path=FORECAST_EXPLAINER_PATH, label="Read the full explainer")
    _overview_tour_button(st)


# --------------------------------------------------------------------------
# Assumptions
# --------------------------------------------------------------------------

_ASSUMPTION_TABLE_COLUMNS = (
    "Archetype",
    "Label",
    "Value",
    "Unit",
    "Evidence",
    "Source",
    "Meaning",
    "Affects",
)
_REFERENCE_COLUMNS = ("Archetype", "Label", "Source detail")
"""The full source text, with its sheet cells, files and decision numbers,
shown in the "Full source references" expander rather than the visible table
(lead decision on the ecacff9 review: the visible table stays clean)."""
_SHARED_ASSUMPTION_TABLE_COLUMNS = tuple(
    column for column in _ASSUMPTION_TABLE_COLUMNS if column != "Archetype"
)
"""Columns for the always-visible shared table, which drops the "All" Archetype
column (goal review action 14): every row in it is the same value, so it adds
a column without adding information."""
_EVIDENCE_LABELS = {"source": "Source", "illustrative": "Illustrative", "synthetic": "Synthetic"}
_EVIDENCE_FILTER_OPTIONS = ("All", "Source", "Illustrative", "Synthetic")
# Contract section 8's raw field names, the columns of
# ``model.assumptions.assumption_rows()`` (docs/contracts/results-v2.md section 8).
CONTRACT_ASSUMPTION_COLUMNS = (
    "name",
    "label",
    "value",
    "unit",
    "evidence",
    "source",
    "meaning",
    "editable",
    "bounds",
    "group",
    "affects",
)

_ASSUMPTION_COLUMN_CONFIG = {
    # Layout ruling 3: Streamlit's default dataframe column width fits one
    # line and ellipsis-truncates the rest, which cut "Unit" (some records
    # carry a descriptive unit, not just "kW") and every "Meaning"/"Affects"
    # sentence. Widening these four, paired with the taller row below, lets
    # the grid wrap instead of truncating; the short columns (Archetype,
    # Label, Value, Evidence) keep their auto-fit width. Explicit pixel
    # widths, not "large": checked in the browser, when the four columns'
    # requested widths summed wider than the table (two at "large" plus the
    # rest at their auto width), the grid silently shrank only the last
    # column to make the row fit -- and that shrunk column stopped
    # wrapping, showing one clipped line with no "...". Kept comfortably
    # under a 1440 px screen's table width so every configured column keeps
    # its own width instead of one absorbing the difference.
    "Unit": column_config.TextColumn(width=180),
    "Source": column_config.TextColumn(width=160),
    "Meaning": column_config.TextColumn(width=320),
    "Affects": column_config.TextColumn(width=320),
}
_ASSUMPTION_ROW_HEIGHT_PX = 88
"""About three wrapped lines (Streamlit's default row height fits one), so a
widened cell wraps its text instead of showing "..." (layout ruling 3)."""


def _archetype_label(name: str) -> str:
    """Which archetype a record belongs to, or "All" for a shared, fleet-wide value.

    Cohort records carry a cohort id as one "."-delimited segment of the
    name, usually the prefix (``"average_uk.daily_miles_mean"``), but a few
    put it last (``"sd_minutes.average_uk"``), so every segment is checked
    rather than assuming a position (goal review action 14).
    """

    for segment in name.split("."):
        label = COHORT_SOURCE_NAMES.get(segment)
        if label is not None:
            return label
    return "All"


def short_source(source: str) -> str:
    """A short, plain source name for the assumptions table's Source column.

    Polish plan G9 ("a clean Source column"): the records' ``source`` text
    carries cell references, file paths and decision numbers that a reader
    does not need at a glance. The full text stays in the "Full source
    references" expander under the table, so nothing is hidden; the Elexon
    short name keeps its copyright mark on screen.
    """

    text = source.strip()
    lowered = text.lower()
    if "'source archetypes'" in lowered or lowered.startswith("'assumptions'"):
        return "Axle cohort sheet"
    if "elexon" in lowered:
        return "Elexon price fit, © Elexon"
    if "cnz" in lowered and not lowered.startswith("illustrative choice"):
        return "CNZ report (May 2022)"
    if lowered.startswith(("illustrative", "interview-demo choice")):
        return "Illustrative choice"
    if lowered.startswith("technical model constant"):
        return "Model constant"
    if lowered.startswith("nord pool"):
        return "Nord Pool market notice"
    if lowered.startswith("decision "):
        # A decision number is a citation, not a name (polish plan G7).
        return "Project decision"
    # Otherwise the text before its first detail: "Nord Pool operational
    # message: ..." reads "Nord Pool operational message".
    first = re.split(r"[;(:]", text, maxsplit=1)[0].strip()
    return first[:1].upper() + first[1:]


def _format_assumption_value(value: Any) -> str:
    """Render one assumption value for the table, including tuple-valued ones."""

    if isinstance(value, tuple):
        return ", ".join(_format_assumption_value(item) for item in value)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _assumption_rows(assumptions: Any) -> pd.DataFrame:
    """Build the display table from a result's assumption tuple or a raw DataFrame.

    Two shapes come in here, both carrying the section 8 fields (name, label,
    value, unit, evidence, source, meaning, affects, ...): a result's
    ``assumptions`` tuple of ``Assumption`` records (attribute access), or a
    DataFrame from ``model.assumptions.assumption_rows()`` (every record at
    its default, shown before a first run).
    Normalising both here means the rest of this module reads one table.
    """

    if isinstance(assumptions, pd.DataFrame):
        records: list[Any] = assumptions.to_dict("records")

        def field(record: Any, name: str) -> Any:
            return record[name]

    else:
        records = list(assumptions)

        def field(record: Any, name: str) -> Any:
            return getattr(record, name)

    rows = [
        {
            "Archetype": _archetype_label(field(record, "name")),
            "Label": field(record, "label"),
            "Value": _format_assumption_value(field(record, "value")),
            "Unit": field(record, "unit") or "—",
            "Evidence": _EVIDENCE_LABELS.get(field(record, "evidence"), field(record, "evidence")),
            "Source": short_source(field(record, "source")),
            "Meaning": field(record, "meaning"),
            "Affects": field(record, "affects"),
            "Source detail": field(record, "source"),
        }
        for record in records
    ]
    return pd.DataFrame(rows, columns=[*_ASSUMPTION_TABLE_COLUMNS, "Source detail"])


def render_assumptions(
    st: Any,
    result: Any | None,
    *,
    assumption_rows_provider: Callable[[], pd.DataFrame] = assumption_rows,
) -> None:
    """ "Assumptions" lens: the read-only table, filterable by evidence class.

    From the active result's ``assumptions`` tuple once a run exists; before
    that, from ``assumption_rows_provider`` (default:
    ``model.assumptions.assumption_rows``, every record at its default).

    The evidence filter is its own row directly above the table (layout
    ruling 3), not the page header's controls slot: How it works caps its
    header to the narrower prose reading column (ruling 2), and the filter
    reads more naturally beside the table it narrows. This lens no longer
    draws its own "Edit assumptions" button either -- the run bar's copy is
    on screen on every page already and opens the same dialog, so a second
    one here was a duplicate, not a different action.
    """

    chosen = st.segmented_control(
        "Evidence class",
        options=_EVIDENCE_FILTER_OPTIONS,
        default="All",
        key="how-it-works-evidence-filter",
        label_visibility="collapsed",
        wrap=True,
    )

    if result is not None:
        table = _assumption_rows(result.assumptions)
    else:
        table = _assumption_rows(assumption_rows_provider())
        st.caption(
            "No run yet: these are the default values. Open Edit assumptions to change what "
            "the first run will use."
        )
    if chosen and chosen != "All" and not table.empty:
        table = table.loc[table["Evidence"] == chosen]

    # Full width, not the narrower prose column (goal review action 14): a
    # wide data table benefits from the room a running-text measure does
    # not need. Shared, fleet-wide rows show first; the six archetypes'
    # per-cohort rows (the same field name repeated six times -- population
    # share, average daily miles, and so on) collapse into an expander,
    # sortable by the new Archetype column, instead of 108 rows always on
    # screen ahead of the 38 shared ones.
    references = table.loc[:, list(_REFERENCE_COLUMNS)]
    visible = table.drop(columns="Source detail")
    shared = visible.loc[visible["Archetype"].eq("All")].drop(columns="Archetype")
    cohort = visible.loc[visible["Archetype"].ne("All")]
    st.dataframe(
        shared,
        width="stretch",
        hide_index=True,
        column_config=_ASSUMPTION_COLUMN_CONFIG,
        row_height=_ASSUMPTION_ROW_HEIGHT_PX,
    )
    if not cohort.empty:
        # "Archetype", not "cohort" (goal review Q-14: one term on screen);
        # ``cohort`` stays as the internal variable and column name (the
        # model's own name for the concept, per ``COHORT_SOURCE_NAMES``).
        with st.expander(f"Archetype-specific values ({len(cohort)} rows across six archetypes)"):
            st.dataframe(
                cohort,
                width="stretch",
                hide_index=True,
                column_config=_ASSUMPTION_COLUMN_CONFIG,
                row_height=_ASSUMPTION_ROW_HEIGHT_PX,
            )
    with st.expander("Full source references"):
        st.dataframe(references, width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------

# One combined "Not modelled" list (design section 0: "No page, tab or
# disabled control exists for them" beyond this one listing), each with a
# one-line reason. "Charging taper" is a model characteristic stated without
# a decision citation: decision 0004 item 31 once cited it but item 32
# supersedes item 31 outright (no stranding; low-SoC trips top up at a
# public charger instead), so that citation no longer applies here. The
# other seven items are the mechanisms design section 0 cuts outright.
NOT_MODELLED: tuple[tuple[str, str], ...] = (
    (
        "DFS",
        "The fleet is not bid into the Demand Flexibility Service. A grid turn-down request "
        "appears only as an illustrative scripted event with an illustrative payment.",
    ),
    (
        "Frequency response",
        "The model steps in half-hours. Frequency response works in under a second, which "
        "this model cannot represent.",
    ),
    (
        "Capacity market",
        "A long-term mechanism for paying capacity to be there, separate from the day-ahead "
        "smart charging modelled here.",
    ),
    (
        "Local flexibility",
        "Local network (DNO) flexibility markets use a different mechanism and price signal "
        "from the smart charging modelled here.",
    ),
    (
        "V2G",
        "EVs only take power in from the grid in this model; there is no discharge or export path.",
    ),
    (
        "Geography",
        # Final critique B-13: this used to say flatly "no locational network
        # constraints; every EV sees the same price and the same action",
        # which the four illustrative zones (decision 0004 item 55, contract
        # v1 §3 and §5.7) and their zone-scoped grid events (events.py's
        # per-zone scope) contradict: a scripted event can already turn down
        # one zone's EVs on their own. What genuinely is not modelled is a
        # real place: zones are a reporting split (no geography), headroom is
        # reported not enforced, and every EV still plans against the same
        # one national price path.
        "A zone is a reporting split, not a place. Zones group EVs and can scope a scripted "
        "grid event to one group. Every EV still sees the same one national synthetic price, "
        "and no local network limit applies.",
    ),
    (
        "Settlement cash",
        "The £ figures are illustrative estimates of cost and value, not Axle's real "
        "settlement or bid cash flow (decision 0003).",
    ),
    (
        "Charging taper",
        "A real charger slows as the battery nears full. This model charges at a constant rate "
        "up to the target, so there is no taper.",
    ),
    # Supplier contract v1 §15, in its order (lane S2).
    (
        "Retail tariff",
        "No retail tariff is modelled. The supplier P&L assumes a flat tariff, and the customer "
        "value assumes the wholesale saving is passed straight through to the customer. Neither "
        "models the supplier's retail margin on a change in volume.",
    ),
    (
        "Supplier intraday re-hedging",
        "The supplier buys its hedge day-ahead, and the error settles at the imbalance price. "
        "Only the trading desk trades intraday.",
    ),
    (
        "Sign-up, opt-outs and retention",
        "Sign-up time, opt-outs, complaints, satisfaction and retention curves are inputs you "
        "set in the growth calculator on Supplier ▸ 5 Charger makers; nothing in the run moves "
        "them.",
    ),
    (
        "Command latency",
        "A plan is followed or ignored at the half-hour; no delay between a command and the "
        "charger acting on it is modelled.",
    ),
    (
        "V2G wear, batteries and heat pumps",
        "No export path, and no home battery or heat pump in the fleet.",
    ),
    (
        "Attach-rate uplift",
        "A charger maker's extra sales from the product are not modelled; the growth "
        "calculator counts illustrative flexibility income only.",
    ),
    (
        "Balancing Mechanism",
        "The fleet is not bid into the Balancing Mechanism; the partner revenue chart lists it "
        "as not modelled, never as £0.",
    ),
)

# Nine questions this build cannot answer, distinct from NOT_MODELLED (a
# mechanism Axle does not use here) and KNOWN_LIMITATIONS (a technical
# footnote on a mechanism that is here). Interview-facing: each line names a
# question a reader would reasonably ask next, in plain English, so it is
# said before anyone has to ask.
CANNOT_ANSWER: tuple[str, ...] = (
    "How real drivers would respond: there is no field data on how drivers react to smart "
    "charging, so the model cannot say whether they would put up with it, override it or opt "
    "out.",
    # Final critique B-13, model-questions review Q-2: this used to say EVs
    # are "sampled mostly independently within a world", which the shared
    # plug-in factors, maker outages and the herding finding all contradict.
    # Q-2 measured it: at a night's turn-down half-hours a 1,000-EV fleet
    # behaves like about 3-4 independent EVs (n_eff), from the one price
    # path every smart EV shares that week; unchanged with the shared
    # plug-in/maker factors switched off, so the price path is the main
    # driver, not those factors.
    "Whether the uncertainty bands are wide enough: EVs already move together within a "
    "simulated week. Every smart EV plans against the same price path that week (herding), so "
    "at its turn-down half-hours a 1,000-EV fleet behaves like about 3-4 independent EVs, not "
    "1,000. A shared plug-in skip factor and each maker's own outage rate add more on top. A "
    "specific real-world trigger beyond these, such as a cold snap or a local power cut, is "
    "still not modelled, and the true spread could be wider than shown.",
    "What a real flexibility product would allow: the price path is synthetic, not observed "
    "market history, and nothing checks a real product's market-access or eligibility rules.",
    "What a large fleet would do to prices: prices are generated without regard to the fleet's "
    "own charging, so the model cannot show how shifting thousands of EVs at once would move "
    "the wholesale price itself.",
    "What a different week would show: one seven-day study window with an invented daily "
    "temperature stands in for a real year's weather.",
    "How sensitive the results are to the assumptions themselves: the model draws random "
    "events (trips, plug-ins, weather, prices) around fixed assumption values; it does not "
    "draw uncertainty in those values.",
    "How real charging hardware would behave: one flat charging power per EV, with no taper "
    "near full and no limit per site, stands in for real chargers and site constraints.",
    "How a real public charging network would behave: top-ups are instant, never queue and "
    "are always there at the moment needed, standing in for a network with cost, "
    "availability and journey-time effects.",
    "Whether it holds at Axle's full fleet size: the model is proven at 1,000 EVs; a 10,000-EV "
    "run has not been benchmarked for run time or memory.",
)

KNOWN_LIMITATIONS: tuple[str, ...] = (
    "No calibration: outputs have not been fitted to or checked against real fleet telemetry.",
    # Final critique B-13: this used to say prices come from "a daily cosine
    # plus AR(1) noise", which was the September 28 model; prices now come
    # from a synthetic system net demand (season, weekday/weekend shape,
    # wind, solar, a daily level and half-hour noise) pushed through a
    # convex supply curve (assumptions.py's price-model records).
    "Synthetic, random prices: each simulated week draws its own day-ahead price. It starts "
    "from a synthetic national net demand (a season and weekday/weekend shape, plus wind, "
    "solar, a daily level and half-hour noise), runs it through a convex supply curve, then "
    "adds its own forecast error (AR(1) noise, where each half-hour's error carries part of "
    "the last one's) to give the realised price. None of this is observed market data "
    "(decision 0004 items 48, 53, 55).",
    "One home charging power: every EV shares it (7 kW by default, editable; decision 0004 "
    "item 34).",
    "Clock changes: a study week that crosses a London clock change has one night of 46 or 50 "
    "half-hours; read charts on that night with care (decision 0004 item 1).",
    "Time spent at a public top-up is not modelled (decision 0004 item 32).",
    "Top-ups at the threshold (10% SoC by default, decision 0004 item 37) mean no plug-in "
    "below 10% SoC can happen in the model, while plug-ins between 10% and 20% SoC can. CNZ "
    "observed about 3% of plug-ins below 10% SoC.",
    "No site or network limit: one shared price signal per week means every EV in a simulated "
    "week plans against the same day-ahead path, so smart charging can bunch into that week's "
    "cheapest half-hours. The peak's timing and size vary week to week rather than sitting at "
    "one clock time. This stays labelled as a finding and is not corrected with a cap "
    "(decision 0004 item 48).",
    "Price look-ahead: a plan ranks a day's day-ahead prices only once they are published, at "
    "13:00 London the day before. Half-hours not yet published are ranked on the expected "
    "shape: the mean price of the same half-hour over the last seven published days, a simple "
    "stand-in for a real price forecast (decision 0004 items 52–53).",
    "Study window: the reported week runs noon to noon over seven session nights, so every "
    "overnight session sits inside it. A seven-day warm-up with prices runs first and is not "
    "shown (decision 0004 item 52).",
    # Supplier contract v1 §15 (lane S2).
    "Carbon: the CO₂ figures use an illustrative synthetic carbon intensity tied to the "
    "model's own net demand, not observed grid carbon data, and the average mix rather than "
    "the marginal plant.",
    "Hedge blocks: peak is 07:00–19:00 Monday to Friday by calendar date, 60 hours in an "
    "ordinary week and 59 or 61 in a clock-change week that ends on a weekday. Every block's "
    "hours are counted from the study's own half-hours.",
    "Horizons: month, season and year figures are scaled up from one simulated week "
    "(scenarios), not forecasts. Only the day-ahead and intraday figures are decided at an "
    "instant.",
    "Shared timing variation (people plugging in earlier or later together) is not modelled "
    "(decision 0004 item 64), so partner and supplier bands may be narrow on timing.",
    # Trading contract v1 §4.3's own two promised Limits lines (model-questions
    # review Q-4, "other observations": omitted until now).
    "Forecast upward bias: each archetype's expected arrival uses its fixed typical clock "
    "time, so the expected unmanaged import, and with it the day-ahead position, is too peaky "
    "and too high in each archetype's typical arrival slots. Imbalance and intraday re-trading "
    "show this rather than hide it (contract v1 §4.3).",
    "Always-plugged tail: the always-plugged archetype's session runs two hours past the "
    "night end. The expected-availability forecast clips it there, so the 12:00-14:00 tail "
    "of the previous session is not forecast into the next night (contract v1 §4.3).",
    # Model-questions review Q-4 (material) and decision 0004 item 67: the
    # headline trading net rests on this baseline choice.
    "Settlement baseline: the default baseline learns only from unmanaged warm-up nights (the "
    "last five weekday or two weekend nights), never from a night with smart charging, the "
    "way BL01 leaves out days with delivered flex. A baseline that rolled over the fleet's own "
    "smart nights would absorb the evening shift and stop settling most of it. Trading ▸ 2 "
    "Position shows this baseline erosion for 16:00–20:00 beside the default, using the "
    "week's other smart nights as a steady-state stand-in (decision 0004 item 67).",
    # Model-questions review Q-5: settlement is one-sided by the contract's own
    # design (T§4.2), not a real product's blind spot.
    "Turn-down-only settlement: every open half-hour settles only the turn-down side of the "
    "deviation from baseline, never the rebound. Energy drawn back above baseline later the "
    "same night is never charged to the trading product. That is by design, not because a "
    "real product could not see it.",
    # Model-questions review Q-5: decision 0005's "P415 mutualises it" is true
    # but loose enough to read as "nobody pays" (critique Q-5 interviewer risk).
    "Supplier compensation: the trading product pays none. Under P415 as approved (Ofgem, 6 "
    "October 2023), the affected supplier is compensated at a price-cap sourcing price from a "
    "fund shared by all suppliers. That fund, and the supplier's evening premium it does not "
    "cover, are not shown; the supplier P&L instead assumes the supplier captures the shift "
    "itself.",
)


def render_limits(st: Any, result: Any | None) -> None:
    """ "Limits" lens: Not modelled, then what this model cannot answer, then known limitations."""

    del result  # unused: this lens states fixed, build-wide limits (design section 3.2)
    with _prose_column(st):
        st.markdown("**Not modelled**")
        for item, reason in NOT_MODELLED:
            _cited_markdown(st, f"- **{item}**: {reason}")

        st.markdown("**What this model cannot answer**")
        for line in CANNOT_ANSWER:
            _cited_markdown(st, f"- {line}")

        st.markdown("**Known limitations**")
        for line in KNOWN_LIMITATIONS:
            _cited_markdown(st, f"- {line}")

        with st.expander("CNZ report and Axle sheet crosswalk"):
            st.markdown(
                "The six archetypes' values (population share, mileage, battery, charger "
                "power, plug-in timing) come from Axle's cohort sheet. The exact mapping from "
                "each sheet cell to the model's value, and any known mismatches, is recorded "
                f"in `{COHORT_CROSSWALK_PATH.relative_to(_REPO_ROOT)}`. The CNZ report's "
                "reference points (median plug-in SoC, share below 10% SoC, weekday plug-in "
                "mode) are shown on the plug-in charts as context, not as calibration targets. "
                "Every value's own class, source, illustrative or synthetic, is catalogued in "
                f"`{DATA_AND_SOURCES_PATH.relative_to(_REPO_ROOT)}`."
            )


# --------------------------------------------------------------------------
# How it was built
# --------------------------------------------------------------------------


def render_build_notes(st: Any, result: Any | None) -> None:
    """ "How it was built" lens: Mike's build notes, then the full explainer.

    Build notes come from ``BUILD_NOTES_PATH`` only when Mike has published
    them there; otherwise this shows a plain status line rather than draft
    narrative text standing in for his approved copy.
    """

    del result  # unused: build notes describe the build, not one run's output
    with _prose_column(st):
        build_notes = _read_text(BUILD_NOTES_PATH)
        if build_notes is not None:
            st.markdown(build_notes)
        else:
            st.caption("Build notes are not published yet.")

        _render_explainer_expander(st, path=BUILD_EXPLAINER_PATH, label="Read the full explainer")
