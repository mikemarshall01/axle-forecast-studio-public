"""Compare: A vs B over the kept run history, B minus A (docs/DASHBOARD_DESIGN.md 4.4).

What this owns: reading the shell's run history, the A/B run selectors,
calling ``compare_runs`` on the chosen pair, and rendering the
matched-futures label, the changed-assumptions diff, the four B-A KPIs and
the per-slot paired B-A band (decision 0004 item 6; contract v2 section 6).
This module never runs the model itself; the one exception decision 0004
item 6 allows is calling ``compare_runs`` (an injectable function, so a test
can pass the fixture's version and this module stays importable without the
model package -- see ``_default_compare_runs``).

Two of the four KPI tiles (evening peak kW, public top-ups) are computed in
this view rather than read off ``comparison.kpis``: ``model.summaries.
compare_runs`` only totals the three metrics decision 0004 item 6 names, and
this task owns the view only, not that model function (goal review action
8). ``_evening_peak_kw`` reads a slot out of the already-quantiled
``difference_bands`` the chart already uses; ``_public_top_up_kwh`` sums
``fleet_world_intervals`` per world first, then quantiles the paired
difference, the same order every other aggregation in this app uses.

Decision 0004 item 35 ("Compare works from slim run records"): run history
keeps enough of every kept run -- not only the latest -- for ``compare_runs``
(``fleet_world_intervals``, ``plug_in_world_kpis``, ``cost_effect`` on action
runs, ``weekly_bands``, the settings snapshot, seed, model, world count).
This view therefore passes the two run records straight to ``compare_runs``
and never reaches into a ``.result`` attribute; whether a record additionally
carries a full stored result (true only for the latest run, kept for the
other pages) is ``compare_runs``'s business, not this view's.

A and B selectors (design 4.4 mockup): two selectors over the kept run
history (at most three runs), defaulting to previous vs latest. A
hard-coded ``records[-2], records[-1]`` would leave no way to pick an
earlier pair when the default pair cannot be paired (different world count,
horizon start or slot count) -- three runs where only the first two are
comparable would show only the "Cannot compare" message, with the pair
that *is* comparable unreachable. ``_resolve_pair``/``_default_pair`` avoid
that: the default still prefers previous vs latest, but falls back to the
most recent pair among the kept runs that can be paired, naming the
fallback; a pair the presenter picks
explicitly still shows the plain unpairable message (never a silent
fallback for a deliberate choice). Selection lives in the
``compare-run-a``/``compare-run-b`` session-state keys, holding a run
number, not a widget that can call the model, so changing A or B never
starts a run.

A new run also moves the default forward (final critique B-18): a stale A/B
choice from before the run resets to the fresh previous-vs-latest pair (or
its fallback), tracked by ``_LAST_SEEN_RUN_KEY`` rather than by whether the
old choice happens to still be a valid option -- Run 1 and Run 2 both
outlive Run 3 landing (``HISTORY_SIZE`` = 3), so "still valid" alone would
never have caught this. A pair the presenter picks after that reset stays
put across an unrelated rerun; only the next new run moves it again.
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import usage_log
from ..components import header
from ..components.chart_table import chart_block
from ..components.kpi import kpi, kpi_columns
from ..run_controller import render_edit_button, run_history
from ..style import (
    CHART_HEIGHTS,
    MUTED_INK,
    PATH_STYLES,
    UNAVAILABLE,
    band_and_line,
    format_quantity,
    kw,
    london_time_axis,
    money,
    percent,
)
from .supplier_common import group_label

# design 4.3-style "Metric [...]" segmented control; kW/percentage-points read
# more directly on a chart than the underlying kWh-per-half-hour/fraction
# columns, so the label maps to the friendlier column contract v2 also
# defines (section 3.5).
METRIC_OPTIONS = {
    "Home import": ("home_import_kw", "kW"),
    "Battery SoC": ("battery_soc_percent", "percentage points"),
    # connected_share is a 0-1 fraction in the contract's own unit column;
    # _scale below turns a B-A difference of it into percentage points for
    # display (never the raw fraction, e.g. "-0.03"), the same convention
    # Response uses for this metric.
    "Plugged in": ("connected_share", "percentage points"),
}

_FRACTION_METRICS = frozenset({"connected_share"})


def _scale(values: Any, metric_key: str) -> Any:
    """Display-only ×100 for a 0-1 fraction metric (mirrors action_response._scale)."""

    return 100 * values if metric_key in _FRACTION_METRICS else values


def _no_negative_zero(values: Any, decimals: int) -> Any:
    """Round to a hovertemplate's own precision, clearing a lone "-" off zero.

    Mirrors ``action_response._no_negative_zero``: d3-format prints "-0" for
    a raw float in (-0.5 * 10**-decimals, 0) even though it displays as 0
    everywhere else, so a trace with a hovertemplate must carry values
    already rounded, with the resulting negative zero cleared by adding 0.0
    (IEEE 754: -0.0 + 0.0 == 0.0).
    """

    return values.round(decimals) + 0.0


# Order and labels for the B-A KPI row (goal review action 8). "Median
# plug-in SoC" and "Unserved travel" are left out: both are fleet-wide
# medians/totals that barely move between two runs (goal review section 3),
# almost always +0, so the evening import peak and public top-up change are
# shown instead. "Illustrative total" keeps the model's own cost-sign
# convention (selected minus normal; negative is cheaper, matching
# action_cost.py's "Illustrative total"), so a positive B-A value here means
# B's cost went *up* -- B saved less than A, the opposite of what a tile
# once called "Weekly saving change" would suggest next to a bare "+£380"
# (goal review N2). Relabelled instead of inverting the figure, so this
# tile stays in the same sign convention as the rest of the smart-charging
# cost views; ``_saving_change_sentence`` states the saving direction in
# words.
_KPI_ORDER = (
    ("home_import_kwh", "Weekly home import"),
    # "Evening peak" (goal review O-20) read as the change in the peak
    # itself; the value is B - A at the half-hour with the largest median
    # shift in the evening window (``_evening_peak_kw``), which need not be
    # either run's own peak half-hour.
    ("evening_peak_kw", "Largest evening change"),
    ("public_top_up_kwh", "Public top-ups"),
    ("illustrative_total_gbp", "Illustrative cost change"),
)

_EVENING_WINDOW_LOCAL_HOURS = (16, 22)
"""Half-hours counted as "evening" when picking the peak slot for the tile.

Wide enough to catch a normal-path evening peak wherever CNZ or cohort
windows put it, without reaching into a smart-charging morning trough
(goal review section 6: smart charging can herd into an early-hours slot).
"""


def _default_compare_runs(a: Any, b: Any) -> Any:
    """Real ``compare_runs``, imported lazily.

    Imported inside the function, not at module top: ``pages.py`` imports
    this view (``from .views import compare``), so a top-level import of
    ``axle_studio.model.summaries`` back into this module would be circular.
    ``render_compare`` takes ``compare_runs`` as an injectable parameter
    (decision 0004 item 6) so tests can pass the fixture's version instead,
    and production is wired through ``pages.py`` to this same default.
    """

    from axle_studio.model.summaries import compare_runs as real_compare_runs

    return real_compare_runs(a, b)


def _evening_peak_kw(difference_bands: Any) -> tuple[float, float, float, str] | None:
    """B - A home import at the evening half-hour with the largest median shift.

    Reads a row out of the model's own already-quantiled, world-first
    ``home_import_kw`` band (``paired_difference_bands``, via
    ``comparison.difference_bands``) rather than recomputing anything: this
    only picks which half-hour to show as the tile (goal review action 8).
    Returns ``None`` when the metric has no row in the evening window.
    """

    frame = difference_bands.loc[difference_bands["metric"].eq("home_import_kw")]
    start, end = _EVENING_WINDOW_LOCAL_HOURS
    local_hour = frame["interval_start_london"].dt.hour
    evening = frame.loc[local_hour.between(start, end, inclusive="left")]
    if evening.empty:
        return None
    row = evening.loc[evening["p50"].abs().idxmax()]
    when = f"{row['interval_start_london']:%H:%M}"
    return float(row["p10"]), float(row["p50"]), float(row["p90"]), when


def _weekly_metric_by_world(fleet_world_intervals: Any, path_id: str, metric: str) -> np.ndarray:
    """Per-world weekly total of one fleet metric on one path.

    Mirrors ``summaries._metric_matrix(...).sum(axis=1)``: ``compare_runs``
    only totals three metrics for its KPI row (decision 0004 item 6), so
    public top-up energy needs the same per-world-first sum computed here
    instead (goal review action 8; this task owns the view, not
    ``model/summaries.py``).
    """

    rows = fleet_world_intervals.loc[fleet_world_intervals["path_id"].eq(path_id)]
    return rows.groupby("world_id")[metric].sum().sort_index().to_numpy(dtype=float)


def _paired_quantiles(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float, float, float]:
    """B minus A per world, then P10/P50/P90 (never subtract percentiles)."""

    delta = values_b - values_a
    if len(delta) == 0:
        return (float("nan"), float("nan"), float("nan"))
    p10, p50, p90 = np.quantile(delta, (0.1, 0.5, 0.9), method="linear")
    return float(p10), float(p50), float(p90)


def _public_top_up_kwh(
    record_a: Any, record_b: Any, path_a: str, path_b: str
) -> tuple[float, float, float]:
    """B - A weekly public top-up (grid import) energy, paired per world."""

    metric = "public_import_kwh"
    values_a = _weekly_metric_by_world(record_a.fleet_world_intervals, path_a, metric)
    values_b = _weekly_metric_by_world(record_b.fleet_world_intervals, path_b, metric)
    return _paired_quantiles(values_a, values_b)


def _format_money(value: float | None) -> str:
    """Fleet-scale money change: whole pounds, explicit sign, U+2212 (polish plan G6).

    A zero change (after rounding) reads "£0", never "−£0" or "+£0".
    """

    return money(value, signed=True)


def _format_signed(value: float | None, *, suffix: str, decimals: int = 0) -> str:
    if value is None or not isfinite(value):
        return UNAVAILABLE
    # Round before choosing the sign: see _format_money.
    rounded = round(value, decimals)
    sign = "−" if rounded < 0 else "+"
    return f"{sign}{abs(rounded):,.{decimals}f}{suffix}"


def _format_kpi(metric: str, unit: str, value: float | None) -> str:
    if metric == "illustrative_total_gbp":
        return _format_money(value)
    if unit == "percentage points":
        return _format_signed(value, suffix=" pts")
    if "kWh" in unit:
        return _format_signed(value, suffix=" kWh")
    return format_quantity(value, unit)


def _saving_change_sentence(cost_delta: float | None) -> str:
    """Plain-English direction for the B-minus-A illustrative cost change (goal review N2).

    ``illustrative_total_gbp`` is B minus A of the model's own "selected
    minus normal" cost column, so a positive value means B's cost rose --
    B saved *less* than A, not more. Stating the direction in words here,
    rather than negating the figure, keeps the tile's number in the same
    sign convention every other smart-charging cost view uses (negative is
    cheaper).
    """

    if cost_delta is None or not isfinite(cost_delta):
        return "B's saving relative to A is unavailable."
    rounded = round(cost_delta)
    amount = money(abs(rounded))
    if rounded == 0:
        return "B and A save about the same amount."
    return f"B saves {amount} less than A." if rounded > 0 else f"B saves {amount} more than A."


_CAPTION_LIMIT = 140
"""Longest caption line (polish plan G7); longer lists move to the tooltip."""


def _changed_line(changed: Any) -> tuple[str, str | None]:
    """``(line, tooltip)`` naming every changed assumption or setting (design 4.4).

    The line lists changes until it would pass the caption limit, then says
    "+N more"; the tooltip then lists every change, so none is hidden.
    """

    if changed.empty:
        return "No assumption or setting changes between these runs.", None
    parts = []
    for row in changed.itertuples():
        unit = f" {row.unit}" if row.unit else ""
        parts.append(f"{row.label} {row.value_a} → {row.value_b}{unit}")
    full = "Changed: " + "; ".join(parts)
    if len(full) <= _CAPTION_LIMIT:
        return full, None
    shown = []
    for part in parts:
        more = f" (+{len(parts) - len(shown) - 1} more)"
        if len("Changed: " + "; ".join([*shown, part]) + more) > _CAPTION_LIMIT and shown:
            break
        shown.append(part)
    remaining = len(parts) - len(shown)
    return "Changed: " + "; ".join(shown) + f" (+{remaining} more)", "; ".join(parts)


def _difference_figure(frame: Any, *, metric_key: str, metric_label: str, unit: str) -> go.Figure:
    """Selected-minus-normal-style band, but B minus A per world (contract 4.1 shape).

    Uses the ``difference`` token (style.py): the same "selected minus
    normal, paired per world" meaning as elsewhere, here paired as B minus A.
    """

    style = PATH_STYLES["difference"]
    x = frame["interval_start_utc"]
    # Hover precision = axis precision (polish plan G6): fleet kW reads in
    # whole kW, percentage points to one decimal. Only the median carries a
    # hovertemplate, so only it needs the negative-zero-safe rounding.
    decimals = 0 if unit == "kW" else 1
    p50 = _no_negative_zero(_scale(frame["p50"], metric_key), decimals=decimals)
    figure = go.Figure()
    # One legend entry for the band and its median (polish plan G3).
    band_and_line(
        figure,
        x,
        _scale(frame["p10"], metric_key),
        p50,
        _scale(frame["p90"], metric_key),
        name="B − A",
        colour=style["color"],
        width=style["width"],
        dash=style["dash"],
        hovertemplate=f"%{{y:,.{decimals}f}} {unit}<extra></extra>",
    )
    # compact=True (goal review V3): the default "Mon 28"-style label left
    # as little as 2 px between adjacent days at 390 px on a plot this
    # width; a single week never repeats a weekday, so the day-of-month
    # adds width, not disambiguation.
    figure.update_xaxes(tickmode="array", **london_time_axis(x, compact=True))
    figure.update_yaxes(
        title=f"{metric_label} difference ({unit})",
        # Difference charts are the one documented exception to "no zero
        # line" (design 3.5 "Chart conventions"): a visible zero baseline is
        # what makes "B above/below A" readable at a glance. MUTED_INK, not
        # a literal grid hex: the grid colour matches the gridlines exactly,
        # so a zero line in that colour would be invisible against them.
        zeroline=True,
        zerolinecolor=MUTED_INK,
        zerolinewidth=1,
    )
    return figure


def _default_unpairable_reason(a: Any, b: Any) -> str | None:
    """Why two runs cannot be paired world by world, or ``None`` if they can.

    Mirrors ``ui.pages.unpairable_reason`` (Compare wiring): a per-world B-A
    difference needs the same world count, horizon start and study slot
    count in both runs -- otherwise world w of A and B cover different weeks
    or do not both exist. Duplicated here rather than imported: ``pages.py``
    already imports this view (``from .views import compare``), so the
    reverse import would be circular. Production always passes
    ``pages.unpairable_reason`` in instead; this default only keeps the view
    importable and testable on its own, the same reason
    ``_default_compare_runs`` is deferred.
    """

    if a.world_count != b.world_count:
        difference = (
            f"Run {a.number} has {a.world_count} simulated weeks, "
            f"Run {b.number} has {b.world_count}"
        )
    elif a.horizon_start_utc != b.horizon_start_utc or a.study_slot_count != b.study_slot_count:
        start_a, start_b = a.start_local_date, b.start_local_date
        difference = (
            f"Run {a.number} starts {start_a:%a} {start_a.day} {start_a:%b}, "
            f"Run {b.number} starts {start_b:%a} {start_b.day} {start_b:%b}"
        )
    else:
        return None
    return f"Cannot compare: {difference}. Keep weeks and start date the same to compare."


def _default_pair(
    records: list[Any], unpairable_reason: Callable[[Any, Any], str | None]
) -> tuple[Any, Any, str | None]:
    """Previous vs latest (design 4.4), or the most recent pair that can be paired.

    For example: run 200x10, run 200x10 edited, run 200x100 -- previous
    (10 weeks) vs latest (100 weeks) cannot be paired, but the first two runs
    can. Rather than dead-end on an unpairable default with the working
    comparison unreachable, search the kept runs (at most three, so at most
    three pairs) from the most recent pair backwards and default to the
    first one ``unpairable_reason`` accepts, naming the fallback in the
    returned note. A pair the presenter picks explicitly keeps the plain
    unpairable message instead (see ``render_compare``); this note only
    explains why the *default* moved.
    """

    previous, latest = records[-2], records[-1]
    reason = unpairable_reason(previous, latest)
    if reason is None:
        return previous, latest, None
    for b_index in range(len(records) - 1, -1, -1):
        for a_index in range(b_index - 1, -1, -1):
            a, b = records[a_index], records[b_index]
            if unpairable_reason(a, b) is None:
                note = f"{reason} Showing Run {a.number} vs Run {b.number} instead."
                return a, b, note
    return previous, latest, None


_LAST_SEEN_RUN_KEY = "compare-last-seen-run"
"""Session-state key: the latest kept run's ``.number`` the last time Compare
resolved a pair. Final critique B-18: Run numbers only increase (``run_full``
never reuses one, even after a run drops out of the kept-3 history), so a
mismatch between this and the current latest run means a new run has landed
since Compare last rendered."""


def _resolve_pair(
    st: Any, records: list[Any], unpairable_reason: Callable[[Any, Any], str | None]
) -> tuple[Any, Any, str | None]:
    """Render the A/B run selectors and return the chosen pair.

    Selection lives in the ``compare-run-a``/``compare-run-b`` session-state
    keys, holding a run *number* rather than the record itself: ``run_full``
    rebuilds every kept ``RunRecord`` (via ``dataclasses.replace``) on each
    new run, so record objects are not stable identity across reruns, but
    ``.number`` is. Neither key is read by any run-triggering code, so
    changing A or B never starts a model run.

    Final critique B-18: a new run landing while Compare is open (or was left
    open in an earlier tab) must move the pair forward to the fresh
    previous-vs-latest default (or its fallback) -- that is the whole point
    of the demo step "run again, open Compare". The old reset rule alone
    ("not in by_number") missed this: with ``HISTORY_SIZE`` kept at 3, Run 1
    and Run 2 both survive Run 3 landing, so a session still pointing at
    Run 1 vs Run 2 stayed there, unpairable with Run 3, even though Run 2 vs
    Run 3 was sitting right there. ``_LAST_SEEN_RUN_KEY`` names the latest
    run Compare last resolved a pair for; when the current latest run's
    number has moved on since, the default is forced, exactly as on a first
    visit. Once that number matches again (the very next rerun), the forcing
    stops, so a pair the presenter then picks explicitly -- "chose otherwise
    in this run" -- survives an unrelated rerun (a lens click, a draft edit)
    until the *next* new run moves the marker again.
    """

    by_number = {record.number: record for record in records}
    label_by_number = {number: record.label for number, record in by_number.items()}
    default_a, default_b, note = _default_pair(records, unpairable_reason)

    latest_number = records[-1].number
    is_new_run = st.session_state.get(_LAST_SEEN_RUN_KEY) != latest_number
    st.session_state[_LAST_SEEN_RUN_KEY] = latest_number

    # A kept run can also drop out of history (HISTORY_SIZE = 3) between
    # reruns; a stored selection that is no longer a valid option would make
    # Streamlit raise, so a stale or first-ever choice resets to the freshly
    # computed default instead of crashing the page.
    for key, fallback in (
        ("compare-run-a", default_a.number),
        ("compare-run-b", default_b.number),
    ):
        if is_new_run or st.session_state.get(key) not in by_number:
            st.session_state[key] = fallback

    # In the page header's controls slot (polish plan G8), labels collapsed:
    # each option names its side ("A: Run 1 · base") so the row stays one line.
    column_a, column_b = header.controls(st).columns(2)
    a_number = column_a.selectbox(
        "A",
        options=list(by_number),
        format_func=lambda number: f"A: {label_by_number[number]}",
        key="compare-run-a",
        label_visibility="collapsed",
    )
    b_number = column_b.selectbox(
        "B",
        options=list(by_number),
        format_func=lambda number: f"B: {label_by_number[number]}",
        key="compare-run-b",
        label_visibility="collapsed",
    )

    is_default = (a_number, b_number) == (default_a.number, default_b.number)
    return by_number[a_number], by_number[b_number], note if is_default else None


_FIRM_MW_FRAME_LABELS = {"product_sheet": "Product sheet", "firmness_by_manufacturer": "Firmness"}
_FIRM_MW_KEY_LABELS = {"evening": "Evening", "overnight": "Overnight"}
_FIRM_MW_METRIC_LABELS = {
    "window_mean_mw_p10": "Window mean MW, firm (P10)",
    "window_mean_mw_p50": "Window mean MW (P50)",
    "firm_share": "Firm share (P10÷P50)",
    "firm_share_p05": "Firm share (P05÷P50)",
    "firmness_p50": "Firmness, P50",
}
_FIRM_MW_FRACTIONS = frozenset({"firm_share", "firm_share_p05", "firmness_p50"})


def _firm_mw_table(firm_mw: pd.DataFrame) -> pd.DataFrame:
    """``product.firm_mw_comparison``'s rows, formatted (trading contract v1 §10.10).

    No difference column: the frames keep no per-world values to pair, and
    a difference of two quantiles is not a quantile of the paired
    difference (lead ruling on the J5 build).
    """

    def cell(value: float, unit: str) -> str:
        if pd.isna(value):
            return UNAVAILABLE
        return (
            percent(100 * value) if unit == "fraction" else format_quantity(value, "MW", decimals=2)
        )

    return pd.DataFrame(
        {
            "Frame": firm_mw["frame"].map(_FIRM_MW_FRAME_LABELS),
            "Row": [_FIRM_MW_KEY_LABELS.get(key, group_label(key, {})) for key in firm_mw["key"]],
            "Metric": firm_mw["metric"].map(lambda m: _FIRM_MW_METRIC_LABELS.get(m, m)),
            "Run A": [cell(v, u) for v, u in zip(firm_mw["value_a"], firm_mw["unit"], strict=True)],
            "Run B": [cell(v, u) for v, u in zip(firm_mw["value_b"], firm_mw["unit"], strict=True)],
        }
    )


_TABLE_HEADER_PX = 38
_TABLE_ROW_PX = 35
"""``st.dataframe``'s own approximate row and header heights: with no explicit
``height=``, its default auto-height showed only the first few rows of the
Firm MW table and scrolled the rest (goal review O-9, "cut at Maker B"). Not
measured in a browser here (none available to this task); if the real row
height differs, retune these two numbers, not the formula that uses them."""


def _dataframe_height(row_count: int) -> int:
    """Pixel ``height`` that shows every row of a short table, no scrollbar."""

    return _TABLE_HEADER_PX + _TABLE_ROW_PX * row_count


def render_compare(
    st: Any,
    result: Any,
    *,
    compare_runs: Callable[[Any, Any], Any] = _default_compare_runs,
    unpairable_reason: Callable[[Any, Any], str | None] = _default_unpairable_reason,
) -> None:
    """Render Compare: pick two kept runs and show B minus A (design 4.4)."""

    records = run_history(st.session_state)
    if len(records) < 2:
        # A session can already hold one run when this renders (the
        # precomputed "Run 0" default, run_controller._precomputed_history):
        # "Run once" would then wrongly tell someone who has already run
        # once to do the very thing they have done.
        if records:
            st.markdown(
                "One more run is needed to compare. Change an assumption in Edit assumptions "
                "and run again. Each run is kept here automatically (last three)."
            )
        else:
            st.markdown(
                "Run once, change an assumption in Edit assumptions, then run again. Each run "
                "is kept here automatically (last three)."
            )
        render_edit_button(st, key="edit-assumptions-compare")
        return

    record_a, record_b, note = _resolve_pair(st, records, unpairable_reason)
    # The chart's Metric control joins A and B in the header (polish plan
    # G8), drawn now so the header row is complete even when the pair below
    # turns out not to be comparable.
    chosen_metric = header.controls(st).segmented_control(
        "Metric",
        options=list(METRIC_OPTIONS),
        default="Home import",
        required=True,
        key="compare-metric",
        label_visibility="collapsed",
    )

    if record_a.number == record_b.number:
        st.info("Choose two different runs to compare.")
        return

    if note:
        st.caption(note)

    reason = unpairable_reason(record_a, record_b)
    if reason:
        st.info(reason)
        return

    # Anonymous usage logging (AXLE_USAGE_LOG=1), once per distinct A/B pair
    # actually shown, not on every rerun while the same pair stays chosen.
    pair = (record_a.number, record_b.number)
    if st.session_state.get("usage_last_compare_pair") != pair:
        st.session_state["usage_last_compare_pair"] = pair
        usage_log.log_event(st.session_state, "compare_viewed")

    # Slim run records (decision 0004 item 35) carry everything compare_runs
    # needs directly, for every kept run -- not just a full result on the
    # latest one -- so the records go straight in.
    comparison = compare_runs(record_a, record_b)

    if comparison.matched_futures:
        st.caption(f"Matched futures: yes (seed {record_b.seed}, {record_b.world_count} weeks)")
    else:
        for reason_text in comparison.mismatch_reasons:
            st.caption(reason_text)
    changed_line, changed_help = _changed_line(comparison.changed)
    st.caption(changed_line, help=changed_help)

    kpis = comparison.kpis.set_index("metric")
    evening_peak = _evening_peak_kw(comparison.difference_bands)
    public_top_up = _public_top_up_kwh(
        record_a, record_b, comparison.path_id_a, comparison.path_id_b
    )

    # No delta arrow here (goal review action 7): an arrow reads as
    # up-good/down-bad, but every value below already is the change (B minus
    # A). The P10-P90 range is each tile's visible context line instead.
    columns = kpi_columns(st, len(_KPI_ORDER))
    for column, (metric, label) in zip(columns, _KPI_ORDER, strict=True):
        if metric == "illustrative_total_gbp" and not comparison.illustrative_total_available:
            kpi(
                column,
                label,
                UNAVAILABLE,
                help="Illustrative cost change unavailable: run A or B has no Axle action.",
            )
            continue
        if metric == "evening_peak_kw":
            if evening_peak is None:
                kpi(column, label, UNAVAILABLE)
                continue
            p10, p50, p90, when = evening_peak
            kpi(
                column,
                label,
                kw(p50, signed=True),
                context=f"P10–P90: {kw(p10, signed=True)} to {kw(p90, signed=True)}",
                help=f"B minus A at {when} London, across simulated weeks.",
            )
            continue
        if metric == "public_top_up_kwh":
            p10, p50, p90 = public_top_up
            low = _format_signed(p10, suffix=" kWh")
            high = _format_signed(p90, suffix=" kWh")
            kpi(
                column,
                label,
                _format_signed(p50, suffix=" kWh"),
                context=f"P10–P90: {low} to {high}",
                help="B minus A weekly public top-up energy, paired per simulated week, then "
                "P10, P50 and P90 across weeks.",
            )
            continue
        if metric not in kpis.index:
            kpi(column, label, UNAVAILABLE)
            continue
        row = kpis.loc[metric]
        kpi(
            column,
            label,
            _format_kpi(metric, row["unit"], row["p50"]),
            context=(
                f"P10–P90: {_format_kpi(metric, row['unit'], row['p10'])} to "
                f"{_format_kpi(metric, row['unit'], row['p90'])}"
            ),
            help=f"B minus A, {row['unit']}, across simulated weeks.",
        )
        if metric == "illustrative_total_gbp":
            # The sign convention above is "cost", not "saving" (see
            # _KPI_ORDER's comment); this sentence states which way the
            # saving actually moved (goal review N2), and names it
            # illustrative so it is never mistaken for real settlement or
            # Axle cash (decision 0003).
            column.caption(f"{_saving_change_sentence(row['p50'])} Illustrative.")

    metric_label = chosen_metric or "Home import"
    metric_key, unit = METRIC_OPTIONS[metric_label]
    frame = (
        comparison.difference_bands.loc[comparison.difference_bands["metric"].eq(metric_key)]
        .sort_values("slot_index")
        .reset_index(drop=True)
    )
    table = frame.loc[
        :, ["interval_start_london", "interval_start_utc", "mean", "p10", "p50", "p90"]
    ].rename(
        columns={
            "interval_start_london": "Interval start (London)",
            "interval_start_utc": "Interval start (UTC)",
            "mean": f"Mean, B − A ({unit})",
            "p10": "P10",
            "p50": "P50",
            "p90": "P90",
        }
    )
    world_count = int(frame["world_count"].iat[0]) if len(frame) else 0
    matched_note = "" if comparison.matched_futures else " Not matched: see the note above."
    # A differing seed or fleet size means each world's B-A is not a true
    # paired difference (same-world, matched futures), just B and A's own
    # independent draws subtracted -- the title must not call that "paired"
    # even though the chart still renders it, so the caption's fuller "Not
    # matched" explanation is not the only place this shows.
    chart_title = (
        f"B − A, paired per week: {metric_label} ({unit})"
        if comparison.matched_futures
        else "B − A (not matched futures)"
    )

    chart_block(
        st,
        _difference_figure(frame, metric_key=metric_key, metric_label=metric_label, unit=unit),
        title=chart_title,
        caption=(
            f"P10–P90 across {world_count} simulated weeks"
            # "Paired" only when the futures match: otherwise each week's
            # B − A subtracts two independent draws.
            f"{', paired per week.' if comparison.matched_futures else '.'}"
            f"{matched_note} Evidence: illustrative."
        ),
        frame=table,
        definition=(
            "One row per half-hour of the study week: B minus A for this metric, worked out "
            "inside each paired simulated week, then P10, P50 and P90 across weeks. The "
            "futures are matched when both runs used the same seed and fleet size: week 3 of "
            "A and week 3 of B then saw the same prices and the same driving, so the "
            "difference is the change in assumptions alone. Quantiles of components and "
            "totals do not add; each is taken separately."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="compare-difference",
    )

    # Trading contract v1 §10.10: the small Firm-MW frames side by side,
    # None unless both runs carry them (product.firm_mw_comparison).
    # getattr, not comparison.firm_mw: the injectable compare_runs in the
    # fixture-based UI tests predates this field and carries no default.
    firm_mw = getattr(comparison, "firm_mw", None)
    if firm_mw is not None:
        st.markdown(
            "**Firm MW, A vs B**",
            help="The evening and overnight 1 hour turn-down windows and each charger maker's "
            "physical firmness, read separately for A and B. There is no difference column: "
            "these tables keep no per-week values to pair, and a difference of two quantiles "
            "is not a quantile of the difference.",
        )
        firm_mw_table = _firm_mw_table(firm_mw)
        st.dataframe(
            firm_mw_table,
            width="stretch",
            hide_index=True,
            # goal review O-9: with no explicit height this short table's
            # default auto-height scrolled and cut the last rows off ("cut
            # at Maker B"). Sized to the row count instead, so every row is
            # visible without a scrollbar, matching the data-tables rule
            # that a short table shows all of itself.
            height=_dataframe_height(len(firm_mw_table)),
        )
        st.caption("Illustrative; P50 across simulated weeks, read separately for A and B.")
