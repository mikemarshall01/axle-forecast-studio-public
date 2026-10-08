"""Small lookups and wording the five Supplier lenses share.

What this owns: reading one stored row of a summary frame, formatting a
stored P10/P50/P90 triple for a KPI tile, the horizon labels of supplier
contract v1 §12 and the group names the Partners and Supplier P&L lenses
show. Nothing here computes a price, a cost, an allocation or a quantile:
every number is a column the model already supplies (supplier contract v1
§11), and the only arithmetic is the sign and constant scaling the Key stats
view also allows (a positive constant preserves quantiles; a sign flip swaps
P10 and P90).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from axle_studio.model.assumptions import MANUFACTURER_LABELS
from axle_studio.model.supplier import HORIZON_LABELS

from ..style import UNAVAILABLE

UNSET = f"{UNAVAILABLE} (unset)"
"""An optional commercial term nobody has set (platform fee, flat reward).

Overnight review ruling and decision 0003: an unset term is unavailable,
never £0, and the screen says why."""

EXTRAPOLATED = "extrapolated from one simulated week"

BUCKET_LABELS = {
    "day_ahead_revenue_gbp": "Day-ahead",
    "intraday_pnl_gbp": "Intraday",
    "trading_cost_gbp": "Trading cost",
    "imbalance_gbp": "Imbalance",
    "baseline_effect_gbp": "Baseline effect",
    "grid_event_payment_gbp": "Grid events",
    "supplier_compensation_gbp": "Supplier compensation",
    "customer_revenue_share_gbp": "Customer share",
    "unmet_charge_penalty_gbp": "Unmet-charge penalty",
    "net_gbp": "Net",
}
"""The trading ledger's nine buckets and net (decision 0005), short enough for bar labels."""


def horizon_label(horizon: str) -> str:
    """The on-screen words for a ``horizon`` value (supplier contract v1 §12)."""

    return HORIZON_LABELS.get(horizon, horizon)


def stat_row(frame: pd.DataFrame | None, **keys: Any) -> pd.Series | None:
    """The one row of ``frame`` whose columns equal ``keys``, or ``None``.

    ``None`` when the frame is absent (a result built before the frame
    existed) or no row matches, so a caller renders "Unavailable" rather
    than failing.
    """

    if frame is None:
        return None
    mask = pd.Series(True, index=frame.index)
    for column, value in keys.items():
        mask &= frame[column].eq(value)
    rows = frame.loc[mask]
    return rows.iloc[0] if len(rows) else None


def missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def p50_text(row: pd.Series | None, fmt: Callable[[float], str], *, unset: bool = False) -> str:
    """The P50 cell of a stored row; ``UNSET`` for an unset term, "Unavailable" when absent."""

    if row is None or missing(row["p50"]):
        return UNSET if unset else UNAVAILABLE
    return fmt(float(row["p50"]))


def band_text(row: pd.Series | None, fmt: Callable[[float], str], *, flip: bool = False) -> str:
    """ "P10 a · P90 b" from a stored row, or "" when absent.

    ``flip`` reads the row with its sign flipped (a payment as a payout):
    the flipped P10 is minus the stored P90, so the two swap places.
    """

    if row is None or missing(row["p10"]) or missing(row["p90"]):
        return ""
    low, high = float(row["p10"]), float(row["p90"])
    if flip:
        low, high = -high, -low
    return f"P10 {fmt(low)} · P90 {fmt(high)}"


def group_label(group_id: str, cohort_labels: dict[str, str]) -> str:
    """ "Fleet", an archetype's name or a charger maker's name for a ``group_id``."""

    if group_id == "fleet":
        return "Fleet"
    if group_id in MANUFACTURER_LABELS:
        return MANUFACTURER_LABELS[group_id]
    return cohort_labels.get(group_id, group_id)


def cohort_labels(result: Any) -> dict[str, str]:
    summary = result.cohort_summary
    return dict(zip(summary["cohort_id"], summary["cohort_label"], strict=True))


def season_text(frame: pd.DataFrame | None) -> str:
    """ "summer demand shapes (Apr–Sep start)" or the winter equivalent, from a
    frame's stored ``season`` column (§0).

    Says what the season means: a bare "summer run" read as wrong on a run
    starting 29 September (final critique B-2). The months follow the
    ``summer_months`` market record (4-9), which is not editable.
    """

    if frame is None or "season" not in frame.columns or frame.empty:
        return ""
    season = str(frame["season"].iloc[0])
    months = "Apr–Sep" if season == "summer" else "Oct–Mar"
    return f"{season} demand shapes ({months} start)"


def evidence_caption(result: Any) -> str:
    """The evidence sentence every supplier caption ends with, naming a user price curve."""

    if getattr(result, "price_curve_source", None) == "user curve":
        return "Illustrative; the day-ahead shape is the uploaded curve."
    return "Illustrative, synthetic prices."
