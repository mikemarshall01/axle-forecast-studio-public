"""Supplier ▸ 2 Positions: the day-ahead and final positions, their CSV, and the price-curve upload.

Reads (trading contract v1 §9.3e, §9.8, §10.5f): ``supplier_positions``
(one row per study slot, strategy ``full``, quantiles across simulated
weeks), shown as a chart, a table and a CSV download written exactly as
stored (``to_csv(index=False)``); and ``open_position_profile`` for ``full``
as a small chart by London half-hour. The stored ``_mw_`` columns are shown
on screen in kW (Q-12 of the final critique: the fleet peaks well under
10 MW, matching the open-position chart, which is already kW); the CSV
keeps the model's own MW columns and names untouched.

The upload control (§9.4) sets only the draft's supplier price curve,
validated by the model's own ``market.validate_user_price_curve``; it never
starts a run. The functions that validate and store the curve are passed in
by ``pages.py``, so this view stays free of the run controller and can be
rendered against a fixture.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .. import usage_log
from ..components.chart_table import chart_block
from ..style import (
    CHART_HEIGHTS,
    PATH_STYLES,
    SERIES_COLOURS,
    band_and_line,
    hover_time_labels,
    london_time_axis,
)
from .supplier_common import evidence_caption, horizon_label

UPLOAD_KEY = "supplier-price-curve-upload"
UPLOAD_ERROR_KEY = "supplier-price-curve-error"
DRAFT_CURVE_KEY = "draft_user_price_curve"
"""The run controller's draft key for the curve (read here, written only through the callback)."""

_COLUMN_LABELS = {
    "slot_index": "Slot",
    "interval_start_utc": "Start (UTC)",
    "interval_start_london": "Start (London)",
    "settlement_date": "Settlement date",
    "settlement_period": "Settlement period",
    "night_index": "Night",
    "day_ahead_position_mw_p50": "Day-ahead position P50 (kW)",
    "final_position_mw_p50": "Final position P50 (kW)",
    "baseline_mw_p50": "Baseline P50 (kW)",
    "metered_mw_p50": "Metered P50 (kW)",
    "settled_mw_p50": "Settled P50 (kW)",
    "net_change_mw_p50": "Net shape change P50 (kW)",
    "deliverable_turn_down_1h_mw_p10": "Firm turn-down 1 h P10 (kW)",
    "day_ahead_price_gbp_per_mwh_p50": "Day-ahead price P50 (£/MWh)",
}
"""The on-screen table's columns and names. The stored ``_mw_`` columns are shown
in kW (Q-12: fleet power stays below 10 MW, matching the open-position chart
below, which is already kW); the CSV download (``render_positions``'s
``download_button``) keeps every column exactly as stored, in MW."""

_KW_PER_MW = 1000.0
_MW_COLUMNS = tuple(column for column in _COLUMN_LABELS if column.endswith(("_mw_p50", "_mw_p10")))
"""Stored MW columns the on-screen table scales to kW; a constant scale keeps
every quantile exact (supplier_common.py's module docstring)."""


def _display_table(positions: pd.DataFrame) -> pd.DataFrame:
    """The on-screen positions table: the stored MW columns rescaled to kW."""

    table = positions.loc[:, list(_COLUMN_LABELS)].copy()
    table[list(_MW_COLUMNS)] = table[list(_MW_COLUMNS)] * _KW_PER_MW
    return table.rename(columns=_COLUMN_LABELS)


def download_name(result: Any) -> str:
    """``axle_positions_<study start date>_seed<seed>.csv`` (trading contract v1 §9.3e)."""

    return f"axle_positions_{result.study_start_local_date:%Y-%m-%d}_seed{result.seed}.csv"


def _positions_figure(positions: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    times = positions["interval_start_utc"]
    hover = hover_time_labels(times)
    for prefix, name, style in (
        ("day_ahead_position_mw", "Day-ahead position", PATH_STYLES["normal"]),
        ("final_position_mw", "Final position", PATH_STYLES["difference"]),
    ):
        # kW: matches the open-position chart below and the fleet-power
        # convention (Q-12); a constant scale of the stored MW columns.
        band_and_line(
            figure,
            times,
            positions[f"{prefix}_p10"] * _KW_PER_MW,
            positions[f"{prefix}_p50"] * _KW_PER_MW,
            positions[f"{prefix}_p90"] * _KW_PER_MW,
            name=name,
            colour=style["color"],
            width=style["width"],
            customdata=hover,
            hovertemplate="%{customdata}: %{y:,.0f} kW (P50)<extra>" + name + "</extra>",
        )
    figure.update_xaxes(tickmode="array", **london_time_axis(times))
    figure.update_yaxes(title="kW turn-down sold")
    return figure


def _net_change_figure(positions: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    times = positions["interval_start_utc"]
    band_and_line(
        figure,
        times,
        positions["net_change_mw_p10"] * _KW_PER_MW,
        positions["net_change_mw_p50"] * _KW_PER_MW,
        positions["net_change_mw_p90"] * _KW_PER_MW,
        name="Smart minus unmanaged",
        colour=SERIES_COLOURS["difference"],
        customdata=hover_time_labels(times),
        hovertemplate="%{customdata}: %{y:,.0f} kW (P50)<extra></extra>",
    )
    figure.add_hline(y=0, line_color=SERIES_COLOURS["normal"], line_width=1)
    figure.update_xaxes(tickmode="array", **london_time_axis(times))
    figure.update_yaxes(title="kW, smart minus unmanaged")
    return figure


def _open_position_figure(profile: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    band_and_line(
        figure,
        profile["local_half_hour"],
        profile["p10"],
        profile["p50"],
        profile["p90"],
        name="Open position, full strategy",
        colour=SERIES_COLOURS["difference"],
        hovertemplate="%{x}: %{y:,.1f} kW (P50)<extra></extra>",
    )
    figure.add_hline(y=0, line_color=SERIES_COLOURS["normal"], line_width=1)
    figure.update_xaxes(title="London half-hour", nticks=8)
    figure.update_yaxes(title="kW left open")
    return figure


def _on_upload(
    state: Any,
    validate_price_curve: Callable[[Any, bytes], pd.DataFrame],
    set_price_curve: Callable[[Any, pd.DataFrame | None], None],
) -> None:
    """Uploader callback: validate the file and put it in the draft, or keep the error.

    Removing the file clears the draft's curve. Never runs the model: the
    curve applies at the next Run, like any other edit.
    """

    uploaded = state.get(UPLOAD_KEY)
    if uploaded is None:
        state[UPLOAD_ERROR_KEY] = None
        set_price_curve(state, None)
        return
    try:
        curve = validate_price_curve(state, uploaded.getvalue())
    except ValueError as error:
        # Keep the draft as it was; the model's message names the row and rule.
        state[UPLOAD_ERROR_KEY] = str(error)
        return
    state[UPLOAD_ERROR_KEY] = None
    set_price_curve(state, curve)


def _render_upload(
    st: Any,
    result: Any,
    validate_price_curve: Callable[[Any, bytes], pd.DataFrame] | None,
    set_price_curve: Callable[[Any, pd.DataFrame | None], None] | None,
) -> None:
    state = st.session_state
    st.markdown(
        "**Your own day-ahead price shape**",
        help="A CSV of at most 10 kB with the header half_hour_start,price_gbp_per_mwh and 48 "
        "rows, one per London half-hour 00:00 to 23:30. Each price must sit within the price "
        "floor and cap (−£500 to £4,000/MWh). The curve replaces the typical daily shape only. "
        "The random walk, the shocks and the intraday updates still sit on top of it, so the "
        "simulated weeks still differ from each other.",
    )
    if validate_price_curve is None or set_price_curve is None:
        st.caption("Upload is available in the app.")
        return
    st.file_uploader(
        "Price curve (CSV, 48 half-hours)",
        type=["csv"],
        key=UPLOAD_KEY,
        on_change=_on_upload,
        args=(state, validate_price_curve, set_price_curve),
    )
    error = state.get(UPLOAD_ERROR_KEY)
    if error:
        # The model's message names the row and rule; shown inline unless it
        # quotes a column name, which the copy rules keep to the tooltip.
        reason = "Hover for why" if "_" in error else error.capitalize()
        st.caption(
            f":red[This file cannot be used: {reason}. Fix it and upload again.]", help=error
        )
    draft = state.get(DRAFT_CURVE_KEY)
    applied = result.user_price_curve
    if draft is not None and (applied is None or not draft.equals(applied)):
        st.caption("Curve loaded into the draft. Run simulation to apply it.")
    elif draft is None and applied is not None:
        st.caption(
            "Curve removed from the draft. Run simulation to go back to the synthetic shape."
        )
    if result.price_curve_source == "user curve":
        st.caption("This run used the uploaded curve as its day-ahead shape.")


def render_positions(
    st: Any,
    result: Any,
    *,
    validate_price_curve: Callable[[Any, bytes], pd.DataFrame] | None = None,
    set_price_curve: Callable[[Any, pd.DataFrame | None], None] | None = None,
) -> None:
    """Render Supplier ▸ 2: positions chart, net shape change, open position, table, CSV, upload.

    ``validate_price_curve(state, file_bytes) -> DataFrame`` raises ``ValueError``
    with the reason; ``set_price_curve(state, curve)`` stores it in the
    draft. Both come from ``pages.py``; without them the upload is not drawn.
    """

    positions = result.supplier_positions.sort_values("slot_index")
    weeks = result.world_count
    evidence = evidence_caption(result)
    chart_block(
        st,
        _positions_figure(positions),
        title="Turn-down sold day-ahead and held at the end, kW",
        caption=f"P50 with P10–P90 per slot across {weeks} weeks; slots do not add. {evidence}",
        frame=_display_table(positions),
        definition=(
            "Full strategy: the trading desk sells turn-down day-ahead, then re-trades it "
            "intraday. Day-ahead position: what was sold at 13:00 the day before. Final "
            "position: what was held after the intraday re-trading. Each half-hour's band is "
            "its own spread across simulated weeks, so the bands do not add across half-hours. "
            f"{horizon_label('day_ahead').capitalize()}."
        ),
        height=CHART_HEIGHTS["time_series"],
        key="supplier-positions",
    )
    chart_block(
        st,
        _net_change_figure(positions),
        title="Net shape change with rebound, kW",
        caption="Metered minus unmanaged: negative where charging left, positive where it "
        "landed. P10–P90 across weeks.",
        frame=positions[
            ["interval_start_utc", "net_change_mw_p10", "net_change_mw_p50", "net_change_mw_p90"]
        ]
        .assign(
            net_change_mw_p10=lambda d: d["net_change_mw_p10"] * _KW_PER_MW,
            net_change_mw_p50=lambda d: d["net_change_mw_p50"] * _KW_PER_MW,
            net_change_mw_p90=lambda d: d["net_change_mw_p90"] * _KW_PER_MW,
        )
        .rename(
            columns={
                "interval_start_utc": "Start (UTC)",
                "net_change_mw_p10": "P10 (kW)",
                "net_change_mw_p50": "P50 (kW)",
                "net_change_mw_p90": "P90 (kW)",
            }
        ),
        definition="Smart (metered) import minus unmanaged import in each half-hour, both "
        "paths of the same simulated week. The band is then the spread across weeks.",
        height=CHART_HEIGHTS["strip"],
        key="supplier-net-change",
    )
    profile = result.open_position_profile
    if profile is not None:
        full = profile.loc[
            profile["strategy"].eq("full") & profile["metric"].eq("open_position_kw")
        ].sort_values("profile_order")
        chart_block(
            st,
            _open_position_figure(full),
            title="Open position by half-hour, full strategy, kW",
            caption=f"Delivered minus final position; P10–P90 across {weeks} weeks. {evidence}",
            frame=full[["local_half_hour", "world_count", "p10", "p50", "p90"]].rename(
                columns={
                    "local_half_hour": "Half-hour (London)",
                    "world_count": "Simulated weeks",
                    "p10": "P10 (kW)",
                    "p50": "P50 (kW)",
                    "p90": "P90 (kW)",
                }
            ),
            definition="For each London half-hour: what the fleet delivered minus its final "
            "position, averaged over that half-hour's seven days, then the spread across weeks.",
            height=CHART_HEIGHTS["strip"],
            key="supplier-open-position",
        )

    st.download_button(
        "Download positions (CSV)",
        data=positions.to_csv(index=False),
        file_name=download_name(result),
        mime="text/csv",
        key="supplier-positions-download",
        help="Every stored column, one row per half-hour of the study week, in MW as stored "
        "(the table above shows kW). "
        "Illustrative, synthetic prices. Each half-hour's quantiles are its own spread across "
        "weeks, so they do not add across half-hours.",
        on_click=usage_log.log_event,
        args=(st.session_state, "csv_download"),
        kwargs={"which": "supplier positions"},
    )
    _render_upload(st, result, validate_price_curve, set_price_curve)
