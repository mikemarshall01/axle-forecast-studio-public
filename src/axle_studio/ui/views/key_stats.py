"""Overview ▸ Key stats: the run's headline numbers in one dense table.

What this owns: one ``st.dataframe`` (plan E1, decision 0004 item 54) with
the columns Metric · Unit · P50 · P10 · P90 · Reference · Δ vs unmanaged ·
Verdict · Section, in sections A-E plus F (trading, action results only; a
one-row placeholder when a result carries no trading KPIs, such as a
no-action run). Every number is a column the model already supplies,
each computed per simulated week first and then quantiled across weeks
(contract v2 rule 3), so the rows never add up and this view never sums,
averages or re-quantiles them. The only arithmetic here is exact for
quantiles: a sign flip (a cost's P90 is a saving's P10), division by a
constant (fleet size, capacity) and × 100 for display.

Verdicts are short reading aids computed from those same columns with fixed
rules (``_spread_verdict`` and the per-row rules below), not model output.
The one ratio of two medians ("about 3% of sessions") is labelled "about",
because a ratio of medians is not the median of the per-week ratio.

Reads: ``plug_in_summary.kpis``, ``average_day_bands``,
``session_distribution_bands``, ``flexibility_weekly_summary``, and on an
action result ``smart_charging_summary``, ``cost_effect_summary``,
``weekly_peak_summary``, ``cheapest_half_hour_summary``,
``not_recovered_summary``, ``not_recovered_world_count``,
``difference_weekly_bands`` (docs/contracts/results-v2.md sections 3-4) and
``trading_kpis`` for the ``full`` strategy (trading contract v1 sections 5.5
and 9.1; decision 0005); for section H, Firm MW, ``product_sheet``,
``availability_backtest_summary``, ``availability_bands`` and
``manufacturer_summary`` (trading contract v1 §10.10); and for section I,
Households, ``household_outcomes_summary`` (household contract v1 §6.3:
group ratios for shares, customers' average years for the year).

On a run whose ``timed_start_local_hour`` is set (decision 0007), sections C
and D also read ``timed_cost_effect_summary`` and the optional ``timed`` row
of ``smart_charging_summary``/``weekly_peak_summary``; section I and the
completion row also read the optional ``_timed`` metrics of
``household_outcomes_summary`` and the optional ``timed`` ``path_id`` block
of ``charge_completion_summary``. Each adds rows beside the existing Smart
ones; none replaces them, and all are absent, not blank, when the run has no
Timed tariff path.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from streamlit import column_config

from ..style import (
    PATH_LABELS,
    UNAVAILABLE,
    UNMANAGED_GLOSSARY,
    assumption_value,
    format_quantity,
    money,
    present_paths,
)

# Section last: Metric is pinned, so at 390 px the numbers sit next to it
# instead of scrolling off behind a wide section label.
COLUMNS = (
    "Metric",
    "Unit",
    "P50",
    "P10",
    "P90",
    "Reference",
    "Δ vs unmanaged",
    "Verdict",
    "Section",
)

TIGHT_SPREAD = 0.10
"""(P90 − P10) ÷ |P50| below this reads "tight" (plan E1's example rule)."""
WIDE_SPREAD = 0.50
"""(P90 − P10) ÷ |P50| above this reads "wide": the week-to-week range is
more than half the typical value."""
NEAR_REFERENCE_POINTS = 2.0
"""Within this many percentage points of CNZ reads "near CNZ"."""
LOW_RISK_SHARE = 0.05
"""Early departures under 5% of sessions read "risk low" (plan E1)."""
HERDING_RATIO = 1.5
"""A smart peak above 1.5 × the unmanaged peak reads "herding high" (plan E1)."""
_ZERO = 1e-9

SECTION_A = "Fleet and behaviour"
SECTION_B = "Flexibility, unmanaged"
SECTION_C = "Smart charging outcome"
SECTION_D = "Coordination"
SECTION_E = "Driver risk"
SECTION_F = "Trading"
SECTION_G = "Supplier"
SECTION_H = "Firm MW"
SECTION_I = "Households"
TRADING_PLACEHOLDER = "Not in this run. Run again with smart charging to see it."
HOUSEHOLDS_UNAVAILABLE = "Unavailable: this run carries no household outcomes"
READINESS_TOLERANCE = 0.01
"""Smart readiness within 1 point of unmanaged (difference P10 ≥ −0.01) reads
"no worse than unmanaged" (household contract v1 §6.3)."""

# London half-hour index (0-47) for the plugged-in rows of section A.
_PLUGGED_IN_TIMES = (("18:00", 36), ("22:00", 44), ("03:00", 6), ("07:00", 14))
_FLEX_TIMES = (("18:00", "1800"), ("22:00", "2200"), ("03:00", "0300"))


# --- Formatting and verdict rules ---------------------------------------------


def _missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def _num(value: Any, decimals: int) -> str:
    """A table number with the U+2212 minus; "Unavailable" when missing (never 0)."""

    return format_quantity(None if _missing(value) else float(value), "", decimals=decimals)


def _spread_verdict(p10: Any, p50: Any, p90: Any) -> str:
    """ "tight" or "wide" from the relative P10–P90 range, else no verdict."""

    if any(_missing(v) for v in (p10, p50, p90)) or abs(float(p50)) < _ZERO:
        return ""
    relative = (float(p90) - float(p10)) / abs(float(p50))
    if relative < TIGHT_SPREAD:
        return "tight"
    if relative > WIDE_SPREAD:
        return "wide"
    return ""


def _row(
    section: str,
    metric: str,
    unit: str,
    p10: Any,
    p50: Any,
    p90: Any,
    *,
    decimals: int = 0,
    reference: str = "",
    delta: str = "",
    verdict: str | None = None,
) -> dict[str, str]:
    """One table row; ``verdict=None`` applies the spread rule."""

    return {
        "Section": section,
        "Metric": metric,
        "Unit": unit,
        "P10": _num(p10, decimals),
        "P50": _num(p50, decimals),
        "P90": _num(p90, decimals),
        "Reference": reference,
        "Δ vs unmanaged": delta,
        "Verdict": _spread_verdict(p10, p50, p90) if verdict is None else verdict,
    }


def _text_row(section: str, metric: str, unit: str, p50: str, **extra: str) -> dict[str, str]:
    """A row whose P50 is text (a clock time, a count "k of N"); P10/P90 blank."""

    row = dict.fromkeys(COLUMNS, "")
    row.update({"Section": section, "Metric": metric, "Unit": unit, "P50": p50})
    row.update({key.replace("_", " "): value for key, value in extra.items()})
    return row


def _when(timestamp: Any) -> str:
    """A London instant as "Mon 18:30"."""

    return UNAVAILABLE if _missing(timestamp) else f"{timestamp:%a %H:%M}"


def _versus_cnz(value: Any, cnz: float | None) -> str:
    """ "above CNZ by 11 pts", "below …" or "near CNZ" (percentage points)."""

    if cnz is None or _missing(value):
        return ""
    gap = float(value) - cnz
    if abs(gap) < NEAR_REFERENCE_POINTS:
        return "near CNZ"
    return f"{'above' if gap > 0 else 'below'} CNZ by {abs(gap):.0f} pts"


def _cnz_points(result: Any, name: str) -> float | None:
    """A CNZ context record on the percent scale, or ``None`` when not carried.

    Reads the record's own unit (contract v2 8.1): ``fraction`` records are
    × 100, ``percent`` records are used as they are.
    """

    record = next((r for r in result.assumptions or () if r.name == name), None)
    if record is None:
        return None
    return float(record.value) * (100.0 if record.unit == "fraction" else 1.0)


# --- Sections -----------------------------------------------------------------


def _kpi(result: Any, metric: str, day_type: str) -> pd.Series | None:
    kpis = result.plug_in_summary.kpis
    rows = kpis.loc[kpis["metric"].eq(metric) & kpis["day_type"].eq(day_type)]
    return rows.iloc[0] if len(rows) else None


def _modal_time(result: Any, metric: str) -> str:
    """Weekday bin with the highest mean share of fleet sessions, as "18:30".

    ``share_mean`` (the mean across weeks of each week's share) is compared
    bin against bin; nothing is added. Weekday, to match CNZ's weekday mode.
    """

    frame = result.session_distribution_bands
    rows = frame.loc[
        frame["group_id"].eq("fleet") & frame["day_type"].eq("weekday") & frame["metric"].eq(metric)
    ]
    shares = rows["share_mean"]
    if shares.isna().all():
        return UNAVAILABLE
    return str(rows.loc[shares.idxmax(), "bin_label"])


def _clock_gap_verdict(label: str, reference_hour: float | None) -> str:
    if reference_hour is None or label == UNAVAILABLE:
        return ""
    hours, minutes = (int(part) for part in label.split(":"))
    gap = hours + minutes / 60.0 - reference_hour
    if abs(gap) <= 1.0:
        return "near CNZ"
    return f"{abs(gap):.1f} h {'later' if gap > 0 else 'earlier'} than CNZ"


def _section_a(result: Any) -> list[dict[str, str]]:
    rows = []
    for day_type, day_label in (
        ("all", "all days"),
        ("weekday", "weekdays"),
        ("weekend", "weekend"),
    ):
        kpi = _kpi(result, "plug_ins_per_ev_per_week", day_type)
        if kpi is not None:
            rows.append(
                _row(
                    SECTION_A,
                    f"Plug-ins per EV per week, {day_label}",
                    "plug-ins",
                    kpi["p10"],
                    kpi["p50"],
                    kpi["p90"],
                    decimals=1,
                )
            )
    cnz_soc = _cnz_points(result, "cnz_median_plug_in_soc_percent")
    soc = _kpi(result, "median_plug_in_soc_percent", "all")
    if soc is not None:
        rows.append(
            _row(
                SECTION_A,
                "Median SoC at plug-in",
                "%",
                soc["p10"],
                soc["p50"],
                soc["p90"],
                decimals=1,
                reference="" if cnz_soc is None else f"CNZ {cnz_soc:.0f}%",
                verdict=_versus_cnz(soc["p50"], cnz_soc),
            )
        )
    low = _kpi(result, "share_below_10_percent_soc", "all")
    if low is not None:
        cnz_low = _cnz_points(result, "cnz_share_plug_ins_below_10_percent_soc")
        # The no-stranding top-up rule keeps plug-in SoC at or above 10%
        # (decision 0004 items 32, 37): a zero here is structural.
        zero = not _missing(low["p50"]) and abs(float(low["p50"])) < _ZERO
        rows.append(
            _row(
                SECTION_A,
                "Plug-ins below 10% SoC",
                "%",
                low["p10"] * 100.0,
                low["p50"] * 100.0,
                low["p90"] * 100.0,
                decimals=1,
                reference="" if cnz_low is None else f"CNZ {cnz_low:.0f}%",
                verdict="0 by construction" if zero else _versus_cnz(low["p50"] * 100.0, cnz_low),
            )
        )
    bands = result.average_day_bands
    connected = bands.loc[
        bands["group_id"].eq("fleet")
        & bands["path_id"].eq("normal")
        & bands["metric"].eq("connected_share")
        & bands["day_type"].eq("all")
    ].set_index("local_half_hour")
    for label, half_hour in _PLUGGED_IN_TIMES:
        if half_hour in connected.index:
            band = connected.loc[half_hour]
            rows.append(
                _row(
                    SECTION_A,
                    f"Plugged in at {label}, all days",
                    "% of fleet",
                    band["low"] * 100.0,
                    band["centre"] * 100.0,
                    band["high"] * 100.0,
                )
            )
    cnz_hour = assumption_value(result, "cnz_weekday_plug_in_mode_local_hour")
    plug_in = _modal_time(result, "plug_in_time")
    rows.append(
        _text_row(
            SECTION_A,
            "Most common weekday plug-in",
            "London time",
            plug_in,
            Reference="" if cnz_hour is None else f"CNZ about {int(cnz_hour):02d}:00",
            Verdict=_clock_gap_verdict(plug_in, None if cnz_hour is None else float(cnz_hour)),
        )
    )
    rows.append(
        _text_row(
            SECTION_A,
            "Most common weekday departure",
            "London time",
            _modal_time(result, "departure_time"),
        )
    )
    return rows


def _section_b(result: Any) -> list[dict[str, str]]:
    summary = result.flexibility_weekly_summary
    if summary is None:
        return []
    row = summary.iloc[0]
    capacity = float(row["fleet_charger_capacity_kw"])

    def stat(prefix: str) -> tuple[Any, Any, Any]:
        return row[f"{prefix}_p10"], row[f"{prefix}_p50"], row[f"{prefix}_p90"]

    # "Charger power below target", not "deferrable" (final critique B-1):
    # the model's ``deferrable`` total is the full charger power of plugged-in
    # EVs still below target, a one-half-hour capacity that includes EVs that
    # must charge now and exceeds the actual unmanaged import. Only the
    # "Can wait 2 h+" row is power that can wait.
    peak = stat("peak_deferrable_kw")
    rows = [
        _row(
            SECTION_B,
            "Peak charger power below target",
            "kW",
            *peak,
            reference=f"{100.0 * float(peak[1]) / capacity:.0f}% of {capacity:,.0f} kW chargers"
            if capacity > 0 and not _missing(peak[1])
            else "",
            verdict=f"most often {_when(row['peak_modal_interval_start_london'])}",
        )
    ]
    for label, suffix in _FLEX_TIMES:
        rows.append(
            _row(
                SECTION_B,
                f"Charger power below target at {label}",
                "kW",
                *stat(f"deferrable_kw_{suffix}"),
                reference="full power of plugged-in EVs below target; not all can wait",
            )
        )
    for label, suffix in _FLEX_TIMES:
        rows.append(
            _row(
                SECTION_B,
                f"Below-target power per plugged-in EV at {label}",
                "W",
                *stat(f"w_per_plugged_in_ev_{suffix}"),
                verdict="",
            )
        )
    rows.append(_row(SECTION_B, "Movable energy at 18:00", "kWh", *stat("movable_energy_1800_kwh")))
    rows.append(
        _row(SECTION_B, "Can wait 2 h+ at 19:00", "kW", *stat("deferrable_at_least_2h_kw_1900"))
    )
    rows.append(
        _row(
            SECTION_B,
            "Hours at ¼ capacity or more",
            "h per week",
            *stat("hours_at_least_quarter_capacity"),
            decimals=1,
            reference=f"25% of {capacity:,.0f} kW",
        )
    )
    return rows


def _metric_row(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    return frame.loc[frame[column].eq(value)].iloc[0]


def _metric_row_if_present(frame: pd.DataFrame, column: str, value: str) -> pd.Series | None:
    """Like ``_metric_row``, but ``None`` when no row matches.

    Used for the optional Timed tariff rows (decision 0007): a run without
    the path simply lacks the row, rather than carrying it NaN-filled.
    """

    rows = frame.loc[frame[column].eq(value)]
    return rows.iloc[0] if len(rows) else None


def _section_c(result: Any) -> list[dict[str, str]]:
    summary = result.smart_charging_summary
    normal = _metric_row(summary, "metric", "normal_average_price_gbp_per_mwh")
    smart = _metric_row(summary, "metric", "selected_average_price_gbp_per_mwh")
    change = _metric_row(summary, "metric", "average_price_change_gbp_per_mwh")
    moved = _metric_row(summary, "metric", "moved_home_import_share")
    total = _metric_row(result.cost_effect_summary, "component", "total")
    # The cost total is smart minus unmanaged (negative = saving). A saving is
    # its negative, and negation swaps the tails: saving P10 = −cost P90.
    saving = (-total["p90"], -total["p50"], -total["p10"])
    fleet = result.vehicle_count
    rows = [
        _row(
            SECTION_C,
            f"Price paid, {PATH_LABELS['normal'].lower()}",
            "£/MWh",
            normal["p10"],
            normal["p50"],
            normal["p90"],
            reference="synthetic prices",
        ),
        _row(
            SECTION_C,
            f"Price paid, {PATH_LABELS['selected'].lower()}",
            "£/MWh",
            smart["p10"],
            smart["p50"],
            smart["p90"],
            reference="synthetic prices",
            delta=f"{_num(change['p50'], 0)} £/MWh",
            verdict="cheaper"
            if not _missing(change["p50"]) and change["p50"] < 0
            else "not cheaper",
        ),
    ]
    # Decision 0007, beside Smart's own price-paid row, never instead of it:
    # the timed path has no "change" reading of its own (that stays selected
    # minus normal, model step 2), so no delta/verdict here either.
    timed_price = _metric_row_if_present(summary, "metric", "timed_average_price_gbp_per_mwh")
    if timed_price is not None:
        rows.append(
            _row(
                SECTION_C,
                f"Price paid, {PATH_LABELS['timed'].lower()}",
                "£/MWh",
                timed_price["p10"],
                timed_price["p50"],
                timed_price["p90"],
                reference="synthetic prices; no tariff rate applied, costed like the other paths",
            )
        )
    rows += [
        {
            **_row(SECTION_C, "Weekly saving, illustrative", "£ per week", *saving),
            "P10": money(saving[0]),
            "P50": money(saving[1]),
            "P90": money(saving[2]),
            "Reference": "not Axle cash",
            "Verdict": "saves" if not _missing(saving[1]) and saving[1] > 0 else "no saving",
        },
        {
            **_row(SECTION_C, "Saving per EV, illustrative", "£ per EV per week", *saving),
            "P10": money(saving[0] / fleet, decimals=2),
            "P50": money(saving[1] / fleet, decimals=2),
            "P90": money(saving[2] / fleet, decimals=2),
            "Reference": f"{fleet:,} EVs",
            "Verdict": "",
        },
    ]
    # Decision 0007: Timed tariff's own saving against Unmanaged, from its
    # cost-effect sibling (built by the same function as Smart's, "timed" in
    # the selected slot), beside Smart's two saving rows above. ``getattr``:
    # the SYNTHETIC fixture (two-policy) has no such field at all.
    timed_cost_summary = getattr(result, "timed_cost_effect_summary", None)
    if timed_cost_summary is not None:
        timed_total = _metric_row(timed_cost_summary, "component", "total")
        timed_saving = (-timed_total["p90"], -timed_total["p50"], -timed_total["p10"])
        rows += [
            {
                **_row(
                    SECTION_C,
                    "Weekly saving, timed tariff, illustrative",
                    "£ per week",
                    *timed_saving,
                ),
                "P10": money(timed_saving[0]),
                "P50": money(timed_saving[1]),
                "P90": money(timed_saving[2]),
                "Reference": "not Axle cash; same day-ahead pricing as smart",
                "Verdict": "saves"
                if not _missing(timed_saving[1]) and timed_saving[1] > 0
                else "no saving",
            },
            {
                **_row(
                    SECTION_C,
                    "Saving per EV, timed tariff, illustrative",
                    "£ per EV per week",
                    *timed_saving,
                ),
                "P10": money(timed_saving[0] / fleet, decimals=2),
                "P50": money(timed_saving[1] / fleet, decimals=2),
                "P90": money(timed_saving[2] / fleet, decimals=2),
                "Reference": f"{fleet:,} EVs",
                "Verdict": "",
            },
        ]
    rows.append(
        _row(
            SECTION_C,
            "Home import moved",
            "% of unmanaged",
            moved["p10"] * 100.0,
            moved["p50"] * 100.0,
            moved["p90"] * 100.0,
        )
    )
    return rows


def _section_d(result: Any) -> list[dict[str, str]]:
    peaks = result.weekly_peak_summary
    normal = _metric_row(peaks, "path_id", "normal")
    smart = _metric_row(peaks, "path_id", "selected")
    herding = (
        not _missing(smart["ratio_to_normal_p50"]) and smart["ratio_to_normal_p50"] > HERDING_RATIO
    )
    rows = [
        _row(
            SECTION_D,
            "Weekly peak, unmanaged",
            "kW",
            normal["p10"],
            normal["p50"],
            normal["p90"],
            verdict=f"most often {_when(normal['modal_peak_interval_start_london'])}",
        ),
        _row(
            SECTION_D,
            "Weekly peak, smart",
            "kW",
            smart["p10"],
            smart["p50"],
            smart["p90"],
            delta=f"×{_num(smart['ratio_to_normal_p50'], 2)}",
            verdict=f"most often {_when(smart['modal_peak_interval_start_london'])}",
        ),
        _row(
            SECTION_D,
            "Smart ÷ unmanaged peak",
            "ratio",
            smart["ratio_to_normal_p10"],
            smart["ratio_to_normal_p50"],
            smart["ratio_to_normal_p90"],
            decimals=2,
            verdict="herding high" if herding else "herding low",
        ),
    ]
    # Decision 0007: Timed tariff's own peak and herding ratio, beside Smart's
    # two rows above (never instead); absent, this run's ``weekly_peak_summary``
    # carries no "timed" path_id and these rows are simply not added.
    timed = _metric_row_if_present(peaks, "path_id", "timed")
    if timed is not None:
        rows.insert(
            2,
            _row(
                SECTION_D,
                f"Weekly peak, {PATH_LABELS['timed'].lower()}",
                "kW",
                timed["p10"],
                timed["p50"],
                timed["p90"],
                delta=f"×{_num(timed['ratio_to_normal_p50'], 2)}",
                verdict=f"most often {_when(timed['modal_peak_interval_start_london'])}",
            ),
        )
        herding_timed = (
            not _missing(timed["ratio_to_normal_p50"])
            and timed["ratio_to_normal_p50"] > HERDING_RATIO
        )
        rows.append(
            _row(
                SECTION_D,
                "Timed tariff ÷ unmanaged peak",
                "ratio",
                timed["ratio_to_normal_p10"],
                timed["ratio_to_normal_p50"],
                timed["ratio_to_normal_p90"],
                decimals=2,
                verdict="herding high" if herding_timed else "herding low",
            )
        )
    # Display order (normal, timed, selected): the one place this section
    # loops over whatever paths the frame actually carries, since every path
    # reads the same way here (style.present_paths, decision 0007).
    for path_id in present_paths(peaks["path_id"]):
        path_row = _metric_row(peaks, "path_id", path_id)
        rows.append(
            _row(
                SECTION_D,
                f"Coincidence factor, {PATH_LABELS[path_id].lower()}",
                "ratio",
                path_row["coincidence_factor_p10"],
                path_row["coincidence_factor_p50"],
                path_row["coincidence_factor_p90"],
                decimals=2,
                reference="1 = every plugged-in charger at full power",
            )
        )
    cheapest = result.cheapest_half_hour_summary
    if cheapest is not None and len(cheapest):
        first = cheapest.iloc[0]
        rows.append(
            {
                **_text_row(
                    SECTION_D,
                    "Cheapest day-ahead half-hour",
                    "London time",
                    str(first["local_time_label_p50"]),
                ),
                "P10": str(first["local_time_label_p10"]),
                "P90": str(first["local_time_label_p90"]),
                "Reference": "synthetic prices; the spread is across days, each noon to noon",
            }
        )
    return rows


def _section_e(result: Any) -> list[dict[str, str]]:
    summary = result.smart_charging_summary
    early = _metric_row(summary, "metric", "early_departure_count")
    shortfall = _metric_row(summary, "metric", "early_departure_shortfall_kwh")
    sessions = _kpi(result, "plug_ins_per_week", "all")
    reference, verdict = "", ""
    if sessions is not None and not _missing(sessions["p50"]) and sessions["p50"] > 0:
        # A ratio of two medians, so "about": not the median of the per-week
        # share, which the result does not carry.
        share = float(early["p50"]) / float(sessions["p50"])
        reference = f"of about {sessions['p50']:,.0f} sessions"
        verdict = (
            f"risk {'low' if share < LOW_RISK_SHARE else 'material'}, about {100 * share:.1f}%"
        )
    not_recovered = result.not_recovered_summary
    unrecovered = _metric_row(not_recovered, "metric", "unrecovered_kwh")
    unrecovered_share = _metric_row(not_recovered, "metric", "unrecovered_share")
    material = int(result.not_recovered_world_count or 0)
    public = _metric_row(result.difference_weekly_bands, "metric", "public_import_kwh")
    material_share = assumption_value(result, "not_recovered_material_share_percent")
    return [
        _row(
            SECTION_E,
            "Early departures",
            "sessions per week",
            early["p10"],
            early["p50"],
            early["p90"],
            decimals=1,
            reference=reference,
            verdict=verdict,
        ),
        _row(
            SECTION_E,
            "Shortfall at early departure",
            "kWh per week",
            shortfall["p10"],
            shortfall["p50"],
            shortfall["p90"],
            decimals=1,
        ),
        _row(
            SECTION_E,
            "Energy not recovered",
            "kWh per week",
            unrecovered["p10"],
            unrecovered["p50"],
            unrecovered["p90"],
            decimals=1,
        ),
        _row(
            SECTION_E,
            "Not recovered share",
            "% of home import",
            unrecovered_share["p10"] * 100.0,
            unrecovered_share["p50"] * 100.0,
            unrecovered_share["p90"] * 100.0,
            decimals=2,
            reference="" if material_share is None else f"material at {float(material_share):g}%",
        ),
        _text_row(
            SECTION_E,
            "Weeks materially not recovered",
            "weeks",
            f"{material} of {result.world_count}",
            Verdict="all recovered" if material == 0 else "check Value and risk",
        ),
        _row(
            SECTION_E,
            "Public top-up change",
            "kWh per week",
            public["p10"],
            public["p50"],
            public["p90"],
            decimals=1,
            delta="paired change",
            verdict="no extra top-ups"
            if not _missing(public["p90"]) and public["p90"] <= _ZERO
            else "",
        ),
    ]


def _household(summary: pd.DataFrame, metric: str, statistic: str) -> pd.Series:
    """One ``household_outcomes_summary`` row of the fleet group."""

    rows = summary.loc[
        summary["group_id"].eq("fleet")
        & summary["metric"].eq(metric)
        & summary["statistic"].eq(statistic)
    ]
    return rows.iloc[0]


def _household_if_present(summary: pd.DataFrame, metric: str, statistic: str) -> pd.Series | None:
    """Like ``_household``, but ``None`` when the run has no such metric.

    The optional Timed tariff metrics (decision 0007, household contract v1
    §3.2 model step 2) are absent, not NaN-filled, on a run without the path.
    """

    rows = summary.loc[
        summary["group_id"].eq("fleet")
        & summary["metric"].eq(metric)
        & summary["statistic"].eq(statistic)
    ]
    return rows.iloc[0] if len(rows) else None


def _signed(value: Any, unit: str, decimals: int) -> str:
    """A paired change with its sign ("+1.2 pts", "−3.6 p/kWh"); "" when missing."""

    if _missing(value):
        return ""
    text = format_quantity(float(value), unit, decimals=decimals)
    return f"+{text}" if round(float(value), decimals) > 0 else text


def _section_i(result: Any) -> list[dict[str, str]]:
    """Section I, Households (household contract v1 §6.3), fleet group only.

    Weekly rows read set A (across customers per week, then across weeks) or
    set R (the session- or energy-weighted group ratio, never a mean of
    per-customer shares, O9/B2); the yearly figure in the Reference cell reads
    set B, across customers' average years (B1). Shares are × 100 for display.
    """

    summary = getattr(result, "household_outcomes_summary", None)
    if summary is None:
        return [_text_row(SECTION_I, "Household outcomes", "", "", Verdict=HOUSEHOLDS_UNAVAILABLE)]

    def get(metric: str, statistic: str) -> pd.Series:
        return _household(summary, metric, statistic)

    def spread(row: pd.Series, scale: float = 1.0) -> tuple[Any, Any, Any]:
        return row["p10"] * scale, row["p50"] * scale, row["p90"] * scale

    # Customer charging cost in p/kWh, the unit a UK household reads on its
    # bill; fleet and wholesale prices stay £/MWh (sections C and F). One
    # unit per audience (final critique B-10; axle-conventions units rule):
    # 1 p/kWh = £10/MWh, an exact × 100 of the stored £/kWh.
    pence = 100.0

    value = get("value_gbp_per_week", "mean_across_evs")
    year = {q: get("value_gbp_per_year", f"{q}_of_ev_means")["mean"] for q in ("p10", "p50", "p90")}
    ready = get("completed_share_selected", "group_ratio")
    ready_normal = get("completed_share_normal", "group_ratio")
    ready_change = get("completed_share_difference", "group_ratio")
    cost = get("home_cost_gbp_per_kwh_selected", "group_ratio")
    cost_change = get("home_cost_gbp_per_kwh_difference", "group_ratio")
    cheap = get("cheap_share_selected", "group_ratio")
    cheap_change = get("cheap_share_difference", "group_ratio")
    affected = get("sessions_affected_share", "group_ratio")
    ends = get("session_ends_per_week", "mean_across_evs")
    nights = get("nights_plugged_share", "group_ratio")
    turn_down = get("evening_turn_down_kw_1h", "mean_across_evs")
    turn_down_p10 = get("evening_turn_down_kw_1h", "p10_across_evs")
    co2 = get("co2_shifted_kg_per_week", "mean_across_evs")

    ready_verdict = ""
    if not _missing(ready_change["p10"]):
        ready_verdict = (
            "no worse than unmanaged"
            if ready_change["p10"] >= -READINESS_TOLERANCE
            else "check early departures"
        )
    rows = [
        {
            # Two statistics, named apart (final critique Q-10): the row is
            # the mean across customers each week, then P10/P50/P90 across
            # weeks; the Reference quotes the median customer's own average
            # year, which is a different statistic and not this row × 52.
            **_row(
                SECTION_I,
                "Value per customer, mean",
                "£ per week (illustrative)",
                *spread(value),
            ),
            "P10": money(value["p10"], decimals=2),
            "P50": money(value["p50"], decimals=2),
            "P90": money(value["p90"], decimals=2),
            "Reference": (
                "mean across customers each week, as if day-ahead prices reached the "
                f"customer. Median customer's average year {money(year['p50'])}; P10–P90 "
                f"across customers {money(year['p10'])}–{money(year['p90'])}"
            ),
        },
        _row(
            SECTION_I,
            "Sessions ready at departure, smart",
            "%",
            *spread(ready, 100.0),
            reference=(
                f"counted session by session over {int(ready['ev_value_count']):,} customers "
                f"with closed sessions; unmanaged P50 {_num(ready_normal['p50'] * 100.0, 0)}%"
            ),
            delta=_signed(ready_change["p50"] * 100.0, "pts", 1),
            verdict=ready_verdict,
        ),
    ]
    # Decision 0007, beside Smart's own readiness row, never instead of it: no
    # "change" metric exists for the timed path (that reading stays selected
    # versus normal, household contract v1 §3.2 model step 2), so no delta.
    timed_ready = _household_if_present(summary, "completed_share_timed", "group_ratio")
    if timed_ready is not None:
        rows.append(
            _row(
                SECTION_I,
                "Sessions ready at departure, timed tariff",
                "%",
                *spread(timed_ready, 100.0),
                reference=f"unmanaged P50 {_num(ready_normal['p50'] * 100.0, 0)}%",
            )
        )
    rows.append(
        _row(
            SECTION_I,
            "Smart charging cost",
            "p/kWh",
            *spread(cost, pence),
            decimals=1,
            reference=(
                "cost of all smart home charging divided by its kWh, at synthetic day-ahead "
                "prices; 1 p/kWh = £10/MWh"
            ),
            delta=_signed(cost_change["p50"] * pence, "p/kWh", 1),
            verdict="cheaper"
            if not _missing(cost_change["p90"]) and cost_change["p90"] < 0
            else "",
        )
    )
    timed_cost = _household_if_present(summary, "home_cost_gbp_per_kwh_timed", "group_ratio")
    if timed_cost is not None:
        rows.append(
            _row(
                SECTION_I,
                "Timed tariff charging cost",
                "p/kWh",
                *spread(timed_cost, pence),
                decimals=1,
                reference=(
                    "cost of all timed-tariff home charging divided by its kWh, at synthetic "
                    "day-ahead prices, costed the same way as smart; 1 p/kWh = £10/MWh"
                ),
            )
        )
    rows.append(
        _row(
            SECTION_I,
            "Charging in the cheapest third",
            "%",
            *spread(cheap, 100.0),
            reference="share of home charging in each week's own cheapest third of half-hours",
            delta=_signed(cheap_change["p50"] * 100.0, "pts", 1),
            verdict="",
        )
    )
    timed_cheap = _household_if_present(summary, "cheap_share_timed", "group_ratio")
    if timed_cheap is not None:
        rows.append(
            _row(
                SECTION_I,
                "Charging in the cheapest third, timed tariff",
                "%",
                *spread(timed_cheap, 100.0),
                reference=(
                    "share of timed-tariff home charging in each week's own cheapest third of "
                    "half-hours"
                ),
            )
        )
    rows += [
        _row(
            SECTION_I,
            "Session ends affected",
            "%",
            *spread(affected, 100.0),
            decimals=1,
            reference=f"of {_num(ends['p50'], 1)} session ends per customer (P50)",
            verdict="risk low"
            if not _missing(affected["p50"]) and affected["p50"] < LOW_RISK_SHARE
            else "",
        ),
        _row(SECTION_I, "Nights plugged in", "%", *spread(nights, 100.0), verdict=""),
        _row(
            SECTION_I,
            "Evening turn-down per customer",
            "kW",
            *spread(turn_down),
            decimals=1,
            reference="1 h on the smart path"
            if _missing(turn_down_p10["p50"])
            else f"1 h on the smart path; the P10 customer holds {_num(turn_down_p10['p50'], 1)} "
            "kW (medians across weeks)",
            verdict="Unavailable: this run has no evening turn-down figures"
            if _missing(turn_down["p50"])
            else None,
        ),
        _row(
            SECTION_I,
            "CO₂ shifted per customer",
            "kg per week",
            *spread(co2),
            decimals=1,
            # Mean across customers per week (Q-11); Supplier ▸ 6 shows the
            # same family as the average customer's average month, and
            # section G's row is a P50 week per month, so the three differ.
            reference=(
                "mean across customers; a month is the average week × 52 ÷ 12; illustrative, "
                "from a synthetic carbon intensity"
            ),
            verdict="Unavailable: this run has no CO₂ figures" if _missing(co2["p50"]) else None,
        ),
    ]
    return rows


# --- Section G: Supplier (trading contract v1 §9.8, supplier contract v1 §6) ----

_MONEY = "£ (illustrative)"
"""Every £ row's unit cell starts with this (supplier contract v1 §6)."""
_PER_MONTH = "per month, scaled from one simulated week"
"""The supplier contract's "scenario" horizon in plain words: a month or year
built from one simulated week, not a forecast of one."""


def _stat(frame: pd.DataFrame | None, **keys: Any) -> pd.Series | None:
    """The stored row matching ``keys``, or ``None`` when the frame or row is absent."""

    if frame is None:
        return None
    mask = pd.Series(True, index=frame.index)
    for column, value in keys.items():
        mask &= frame[column].eq(value)
    rows = frame.loc[mask]
    return rows.iloc[0] if len(rows) else None


def _money_cells(row: pd.Series | None, decimals: int) -> dict[str, str]:
    stats = ("p10", "p50", "p90")
    return {
        stat.upper(): money(None if row is None else row[stat], decimals=decimals, signed=True)
        for stat in stats
    }


def _g_row(
    metric: str,
    unit: str,
    row: pd.Series | None,
    *,
    section: str = SECTION_G,
    decimals: int = 0,
    scale: float = 1.0,
    is_money: bool = False,
    reference: str = "",
    delta: str = "",
    verdict: str | None = None,
    unavailable: str = "",
) -> dict[str, str]:
    """One row from a stored summary row (``p10``/``p50``/``p90`` columns), section G by default.

    ``section`` names the row's section; section H passes its own (final
    critique B-5: a hard-coded section G put an H row under G). ``scale``
    is a positive constant (× 100 for a share), exact for quantiles. A
    missing row, or a NaN P50, shows "Unavailable" in every cell and
    ``unavailable`` as the verdict, never 0 (decision 0003).
    """

    stats = [None if row is None else row[stat] for stat in ("p10", "p50", "p90")]
    scaled = [None if _missing(v) else float(v) * scale for v in stats]
    if row is None or _missing(stats[1]):
        verdict = unavailable
    built = _row(
        section,
        metric,
        unit,
        *scaled,
        decimals=decimals,
        reference=reference,
        delta=delta,
        verdict=verdict,
    )
    if is_money:
        built.update(_money_cells(row, decimals))
    return built


def _p10_verdict(row: pd.Series | None) -> str:
    """Supplier contract v1 §6: the before-fee net's week-to-week reading."""

    if row is None or _missing(row["p10"]):
        return ""
    if row["p10"] > 0:
        return "P10 positive: a gain in 9 weeks out of 10"
    if row["p10"] < 0:
        return "P10 negative: a loss in some weeks"
    return ""


def _zero_verdict(row: pd.Series | None, text: str) -> str | None:
    """``text`` when every week is exactly 0 (P10 = P90 = 0), else the spread rule."""

    if row is not None and not _missing(row["p10"]) and row["p10"] == 0 and row["p90"] == 0:
        return text
    return None


def _section_g(result: Any) -> list[dict[str, str]]:
    """Supplier rows: the §9.8 cost-curve, shape and household rows, then supplier-v1 §6.

    Every supplier P&L row reads the ``profiled`` hedge; every row names its
    horizon in the Reference cell. The trading net is quoted beside the
    supplier net, never added (supplier contract v1 §3.1).
    """

    # Imported here: supplier_pnl imports the style module this one does,
    # and keeping the mode wording in one place keeps the two screens alike.
    from .supplier_pnl import payment_mode_text

    curve = result.flex_cost_curve
    shape = result.shape_premium_summary
    household = result.household_value_summary
    pnl = result.supplier_pnl_summary
    blocks = result.hedge_block_summary
    partner = result.partner_summary
    exceedance = result.household_value_exceedance

    def pnl_row(metric: str) -> pd.Series | None:
        return _stat(pnl, hedge_variant="profiled", metric=metric)

    def partner_row(metric: str) -> pd.Series | None:
        return _stat(partner, group_type="fleet", group_id="fleet", metric=metric)

    at_zero = _stat(
        curve,
        day_type="all",
        local_half_hour="18:00",
        metric="available_kw",
        threshold_gbp_per_mwh=0.0,
    )
    charging = _stat(curve, day_type="all", local_half_hour="18:00", metric="charging_kw")
    rows = [
        _g_row(
            "Turn-down at 18:00, cost ≤ £0/MWh",
            "kW",
            at_zero,
            reference="day-ahead prices; all days; includes the risk charge",
        ),
        _g_row(
            "All unmanaged charging at 18:00",
            "kW",
            charging,
            reference="the most the cost curve can reach",
        ),
    ]
    for path_id, label in (
        ("normal", "Shape premium, unmanaged"),
        ("selected", "Shape premium, smart"),
        ("difference", "Shape premium change"),
    ):
        rows.append(
            _g_row(
                label,
                "£/MWh",
                _stat(shape, path_id=path_id, metric="shape_premium_gbp_per_mwh"),
                decimals=1,
                reference=(
                    "per week; the price paid per MWh of charging minus the flat baseload price"
                ),
                delta="paired change" if path_id == "difference" else "",
            )
        )
    rows += [
        _g_row(
            "Household value, mean",
            f"{_MONEY} per household per month",
            _stat(household, group_id="fleet", statistic="mean_value"),
            decimals=2,
            is_money=True,
            reference=f"{_PER_MONTH}; as if day-ahead prices reached the household",
        ),
        _g_row(
            "Households worse off",
            "% of households",
            _stat(household, group_id="fleet", statistic="share_worse_off"),
            decimals=1,
            scale=100.0,
            reference=f"{_PER_MONTH}; value below £0 a month",
        ),
    ]
    if pnl is None:
        rows.append(_text_row(SECTION_G, "Supplier P&L", "£", "", Verdict=TRADING_PLACEHOLDER))
        return rows

    trading_net = _stat(result.trading_kpis, strategy="full", metric="net_gbp_per_week")
    trading_text = (
        ""
        if trading_net is None
        else f"; the trading desk's net {money(trading_net['p50'], signed=True)} a week sits "
        "beside it, not added"
    )
    net = pnl_row("net_gain_before_fee_per_customer_per_month_gbp")
    shape_saving = pnl_row("shape_saving_gbp")
    volume = pnl_row("volume_value_gbp")
    hedge = pnl_row("hedge_error_saving_gbp")
    flat = _stat(pnl, hedge_variant="flat", metric="hedge_error_saving_gbp")
    worth = pnl_row("worth_of_profiled_hedge_gbp")
    grid = pnl_row("grid_event_payment_gbp")
    payment = pnl_row("customer_payment_gbp")
    fee = pnl_row("platform_fee_gbp")
    peak = _stat(blocks, block="peak", path_id="difference", metric="mean_mw")
    hours = "" if peak is None else f", {float(peak['block_hours']):g} hours this week"

    def p50(row: pd.Series | None) -> str:
        return money(None if row is None else row["p50"], decimals=2, signed=True)

    peak_verdict = ""
    if peak is not None and not _missing(peak["p90"]) and peak["p90"] < 0:
        peak_verdict = "moved out of the peak"
    elif peak is not None and not _missing(peak["p10"]) and peak["p10"] > 0:
        peak_verdict = "moved into the peak"
    rows += [
        _g_row(
            "Supplier gain per customer, before fee",
            f"{_MONEY} per customer per month",
            net,
            decimals=2,
            is_money=True,
            reference=f"{_PER_MONTH}{trading_text}",
            verdict=_p10_verdict(net),
        ),
        _g_row(
            "Energy saving at day-ahead",
            f"{_MONEY} per week",
            pnl_row("energy_saving_gbp"),
            is_money=True,
            reference=(
                f"per week; of which shape {p50(shape_saving)} and volume {p50(volume)} (P50)"
            ),
        ),
        _g_row(
            "Hedge-error saving, profiled hedge",
            f"{_MONEY} per week",
            hedge,
            decimals=2,
            is_money=True,
            reference=(
                f"per week; a flat hedge would save {p50(flat)}; the profiled hedge is worth "
                f"{p50(worth)} more (P50)"
            ),
            verdict=_zero_verdict(hedge, "0 in every week"),
        ),
        _g_row(
            "Grid-event payments",
            f"{_MONEY} per week",
            grid,
            is_money=True,
            reference="per week",
            verdict=_zero_verdict(grid, "no events") or "",
        ),
        _g_row(
            "Customer payments",
            f"{_MONEY} per week",
            payment,
            is_money=True,
            reference=payment_mode_text(result),
            unavailable="Unavailable: flat reward not set",
        ),
        _g_row(
            "Platform fee",
            f"{_MONEY} per week",
            fee,
            decimals=2,
            is_money=True,
            reference="per week, from the set £ per EV per month",
            verdict="",  # a set fee is the same every week; no spread to read
            unavailable="Unavailable: not set",
        ),
        _g_row(
            "Peak-block requirement change",
            "MW",
            peak,
            decimals=3,
            reference=f"per week; the peak block is 07:00–19:00 Mon–Fri{hours}",
            delta="paired change",
            verdict=peak_verdict,
        ),
        _g_row(
            "CO₂ shifted per EV, P50 week",
            "kg CO₂ per EV per month",
            _stat(result.carbon_shift_summary, metric="co2_shifted_kg_per_ev_per_month"),
            decimals=1,
            reference=(
                "the P50 week scaled to a month; illustrative, from a synthetic carbon "
                "intensity that follows net demand"
            ),
        ),
        _g_row(
            "Gross flexibility cash per enrolled device",
            f"{_MONEY} per device per month",
            partner_row("gross_flex_gbp_per_enrolled_device_per_month"),
            decimals=2,
            is_money=True,
            reference=(
                f"{_PER_MONTH}; includes the baseline effect (settled volume the baseline "
                "moved, not charging)"
            ),
        ),
        _g_row(
            "Share of devices earning",
            "% of enrolled devices",
            partner_row("share_earning"),
            decimals=1,
            scale=100.0,
            reference="per week",
            verdict="",
        ),
        _g_row(
            "Gross flexibility cash per earning device",
            f"{_MONEY} per device per month",
            partner_row("gross_flex_gbp_per_earning_device_per_month"),
            decimals=2,
            is_money=True,
            reference=_PER_MONTH,
            verdict="",
        ),
        _g_row(
            "£ per kW of charger",
            f"{_MONEY} per kW per year",
            partner_row("gbp_per_kw_charger_per_year"),
            is_money=True,
            reference="per year, scaled from one simulated week",
            verdict="",
        ),
        _g_row(
            "Dispatch success rate",
            "% of sessions",
            partner_row("dispatch_success_rate"),
            decimals=1,
            scale=100.0,
            reference="per week; share of sessions that followed their plan in every half-hour",
            verdict="",
            unavailable="Unavailable: this run does not record plan status",
        ),
        _cycles_row(partner_row),
        _g_row(
            "Households at or above £10/month",
            "% of households",
            _stat(
                exceedance,
                group_id="fleet",
                metric="share_at_or_above",
                threshold_gbp_per_month=10,
            ),
            decimals=1,
            scale=100.0,
            reference=f"{_PER_MONTH}; as if day-ahead prices reached the household",
            verdict="",
        ),
        _g_row(
            "Floor cost of guaranteeing £10/month",
            f"{_MONEY} per device per month",
            _stat(
                exceedance,
                group_id="fleet",
                metric="floor_top_up_gbp_per_device_per_month",
                threshold_gbp_per_month=10,
            ),
            decimals=2,
            is_money=True,
            reference=_PER_MONTH,
            verdict="",
        ),
        *_completion_row(result),
    ]
    return rows


def _cycles_row(partner_row: Any) -> dict[str, str]:
    smart = partner_row("equivalent_full_cycles_per_ev_per_week_selected")
    normal = partner_row("equivalent_full_cycles_per_ev_per_week_normal")
    change = partner_row("equivalent_full_cycles_per_ev_per_week_difference")
    verdict = ""
    if change is not None and not _missing(change["p10"]):
        if -0.05 <= change["p10"] and change["p90"] <= 0.05:
            verdict = "energy moved in time, not amount"
    return _g_row(
        "Equivalent full cycles, smart",
        "cycles per EV per week",
        smart,
        decimals=2,
        reference="" if normal is None else f"unmanaged {_num(normal['p50'], 2)} (P50)",
        delta="" if change is None else f"{_num(change['p50'], 3)} paired",
        verdict=verdict,
    )


def _completion_share(row: pd.Series | None) -> pd.Series | None:
    """``(p10, p50, p90)`` of ``charge_completion_summary``'s ``completed_share_*`` columns."""

    if row is None:
        return None
    return pd.Series(
        {
            "p10": row["completed_share_p10"],
            "p50": row["completed_share_p50"],
            "p90": row["completed_share_p90"],
        }
    )


def _completion_row(result: Any) -> list[dict[str, str]]:
    summary = getattr(result, "charge_completion_summary", None)
    smart = _stat(summary, group_id="fleet", path_id="selected", departure="all")
    normal = _stat(summary, group_id="fleet", path_id="normal", departure="all")
    reference = (
        ""
        if normal is None or _missing(normal["completed_share_p50"])
        else f"unmanaged {_num(100 * normal['completed_share_p50'], 1)}% (P50)"
    )
    rows = [
        _g_row(
            "Charge completed by departure, smart",
            "% of sessions",
            _completion_share(smart),
            decimals=1,
            scale=100.0,
            reference=reference,
            verdict="",
            unavailable=FIRM_MW_UNAVAILABLE,
        )
    ]
    # Decision 0007, beside Smart's own completion row, never instead of it:
    # the timed path's own path_id block (trading contract v1 §10.5e model
    # step 2), read the same way, no third-path difference computed here.
    timed = _stat(summary, group_id="fleet", path_id="timed", departure="all")
    if timed is not None:
        rows.append(
            _g_row(
                "Charge completed by departure, timed tariff",
                "% of sessions",
                _completion_share(timed),
                decimals=1,
                scale=100.0,
                reference=reference,
                verdict="",
                unavailable=FIRM_MW_UNAVAILABLE,
            )
        )
    return rows


def _money_row(
    section: str,
    metric: str,
    unit: str,
    row: pd.Series,
    *,
    decimals: int = 0,
    reference: str = "",
) -> dict[str, str]:
    """A £ row: ``_row``'s numeric skeleton with its P10/P50/P90 overridden by ``money``.

    ``_row``'s own ``_num`` formatter has no £ sign or thousands-aware minus,
    so every money row in this table is built this way (matches the net P&L
    rows above).
    """

    stats = (row["p10"], row["p50"], row["p90"])
    return {
        **_row(section, metric, unit, *stats, decimals=decimals, reference=reference),
        "P10": money(row["p10"], decimals=decimals, signed=True),
        "P50": money(row["p50"], decimals=decimals, signed=True),
        "P90": money(row["p90"], decimals=decimals, signed=True),
    }


FIRM_MW_UNAVAILABLE = "Unavailable: this run has no firm-MW figures"


CALIBRATION_LEVEL = 0.10
"""The P10 forecast level whose hit rates the calibration row shows."""


def _calibration_row(coverage: pd.Series | None) -> dict[str, str]:
    """Day-ahead calibration at the P10 level: both hit rates beside the 10% target.

    The backtest reports two rates over held-out half-hours (overnight
    review log, 30 Sep; trading contract v1 §10.3): the share at or below
    the P10 forecast and the share strictly below it. Many half-hours
    realise exactly 0 kW against a 0 kW forecast, and those ties count as
    hits only in the first, so a calibrated forecast has strict ≤ 10% ≤ at
    or below. Showing the "at or below" rate alone next to "target 10%" read
    as a failed forecast (final critique B-6); no random tie-break is used.
    """

    label, unit = "Day-ahead calibration, P10 level", "% of held-out half-hours"
    reference = (
        "share of held-out half-hours strictly below the P10 forecast, to the share at or "
        "below it; target 10%; half-hours that tie at 0 kW make the two differ"
    )
    if coverage is None or _missing(coverage["coverage_p10"]):
        return _text_row(SECTION_H, label, unit, UNAVAILABLE, Reference=reference)
    weak = float(coverage["coverage_p10"])
    strict = coverage.get("coverage_strict_p10")
    if _missing(strict):
        return _text_row(
            SECTION_H,
            label,
            unit,
            f"{_num(100 * weak, 1)}% at or below",
            Reference=reference,
        )
    strict = float(strict)
    if strict <= CALIBRATION_LEVEL <= weak:
        verdict = "target within the range"
    elif weak < CALIBRATION_LEVEL:
        verdict = "under-covered: P10 too low"
    else:
        verdict = "over-covered: P10 too high"
    return _text_row(
        SECTION_H,
        label,
        unit,
        f"{_num(100 * strict, 1)}–{_num(100 * weak, 1)}%",
        Reference=reference,
        Verdict=verdict,
    )


def _section_h(result: Any) -> list[dict[str, str]]:
    """Section H, Firm MW (trading contract v1 §10.10): the headline turn-down figure.

    Reads ``product_sheet`` (the evening 1 h row), ``availability_backtest_summary``
    (day-ahead coverage at the P10 level), ``availability_bands`` (the effective EV
    count ``n_eff``, median over half-hours with turn-down)
    and ``manufacturer_summary`` (the fleet's chance any maker is out one
    night). Absent on a result built before the availability lanes landed.
    """

    sheet = getattr(result, "product_sheet", None)
    backtest_summary = getattr(result, "availability_backtest_summary", None)
    bands = getattr(result, "availability_bands", None)
    manufacturers = getattr(result, "manufacturer_summary", None)
    if sheet is None or backtest_summary is None or bands is None or manufacturers is None:
        return [_text_row(SECTION_H, "Firm MW", "", "", Verdict=FIRM_MW_UNAVAILABLE)]

    # The one evening firm-MW row (final critique B-5: section G carried a
    # second copy of these values under another name). "Firm" is the P10
    # column, the across-weeks P10 (90% exceedance), and the P05 is 95%
    # exceedance (B-2); the base is turn-down beyond the smart schedule, not
    # against unmanaged charging as in section G (Q-6).
    evening = _stat(sheet, window="evening", direction="turn_down", duration_hours=1.0)
    reference = ""
    if evening is not None:
        parts = [
            "day-ahead; turn-down beyond the smart schedule, not against unmanaged charging",
            "firm = P10 column (90% exceedance): the level met in 9 weeks out of 10",
            f"P05 (95% exceedance) {_num(evening['window_mean_mw_p05'], 3)} MW",
        ]
        if not _missing(evening["firm_share"]):
            parts.append(f"firm share (P10÷P50) {_num(100 * evening['firm_share'], 0)}%")
        if not _missing(evening["intraday_firm_mw_p50"]):
            parts.append(f"typical 17:00 forecast {_num(evening['intraday_firm_mw_p50'], 2)} MW")
        reference = "; ".join(parts)
    evening_row = _g_row(
        "Evening turn-down, 1 h",
        "MW",
        None
        if evening is None
        else pd.Series(
            {
                "p10": evening["window_mean_mw_p10"],
                "p50": evening["window_mean_mw_p50"],
                "p90": evening["window_mean_mw_p90"],
            }
        ),
        section=SECTION_H,
        decimals=3,
        reference=reference,
        unavailable=FIRM_MW_UNAVAILABLE,
    )

    coverage = _stat(
        backtest_summary, horizon="day_ahead", direction="turn_down", duration_hours=1.0
    )
    coverage_row = _calibration_row(coverage)

    realised = bands.loc[
        bands["horizon"].eq("day_ahead")
        & bands["statistic"].eq("realised")
        & bands["direction"].eq("turn_down")
        & bands["duration_hours"].eq(1.0)
    ]
    # n_eff is per half-hour. Read it where the fleet has turn-down to correlate
    # (median 1-h turn-down above 0); at 19:00 the smart fleet's median is 0 kW
    # and only a few EVs vary, so one night's value there says little.
    active = realised.loc[realised["n_eff"].notna() & realised["p50"].gt(0.0), "n_eff"]
    n_eff = float(active.median()) if len(active) else float("nan")
    n_eff_row = _text_row(
        SECTION_H,
        "Effective EV count, turn-down half-hours",
        "EVs",
        UNAVAILABLE if _missing(n_eff) else f"{n_eff:,.1f}",
        Reference=(
            "how many independent EVs the fleet behaves like, since one price path a week "
            "moves them together; median over half-hours with turn-down"
        ),
    )

    fleet = _stat(manufacturers, manufacturer_id="fleet")
    outage_row = _text_row(
        SECTION_H,
        "Chance any maker is out, one night",
        "%",
        UNAVAILABLE
        if fleet is None or _missing(fleet["outage_probability_per_night"])
        else f"{_num(100 * fleet['outage_probability_per_night'], 0)}%",
        Reference="1 minus the chance every maker is online",
    )

    return [evening_row, coverage_row, n_eff_row, outage_row]


def _section_f(result: Any) -> list[dict[str, str]]:
    """Illustrative trading headline rows, ``full`` strategy (trading contract v1 9.1, 9.8).

    A placeholder row stands in when the result carries no trading KPIs (a
    no-action run: trading needs a smart charging plan to trade against).
    """

    kpis = result.trading_kpis
    if kpis is None:
        return [_text_row(SECTION_F, "Trading P&L", "£", "", Verdict=TRADING_PLACEHOLDER)]
    full = kpis.loc[kpis["strategy"].eq("full")].set_index("metric")
    net = full.loc["net_gbp_per_week"]
    per_ev = full.loc["net_gbp_per_ev_year"]
    settled = full.loc["settled_mwh_per_week"]
    imbalance_share = full.loc["imbalance_volume_share"]
    baseline_share = full.loc["baseline_effect_share"]
    capture = full.loc["capture_rate"]
    firmness = full.loc["firmness"]
    cvar_text = (
        ""
        if _missing(net["cvar5"])
        else f"CVaR5 {money(net['cvar5'], signed=True)}: the mean of the worst 5% of weeks"
    )
    net_stats = (net["p10"], net["p50"], net["p90"])
    per_ev_stats = (per_ev["p10"], per_ev["p50"], per_ev["p90"])
    return [
        {
            **_row(SECTION_F, "Illustrative trading net P&L", "£ per week", *net_stats),
            "P10": money(net["p10"], signed=True),
            "P50": money(net["p50"], signed=True),
            "P90": money(net["p90"], signed=True),
            "Reference": cvar_text,
            "Verdict": "not Axle cash",
        },
        {
            **_row(SECTION_F, "Illustrative trading net P&L", "£ per EV per year", *per_ev_stats),
            "P10": money(per_ev["p10"], decimals=2, signed=True),
            "P50": money(per_ev["p50"], decimals=2, signed=True),
            "P90": money(per_ev["p90"], decimals=2, signed=True),
            "Reference": "extrapolated from one simulated week",
            "Verdict": "",
        },
        _row(
            SECTION_F,
            "Settled turn-down volume",
            "MWh per week",
            settled["p10"],
            settled["p50"],
            settled["p90"],
            decimals=2,
        ),
        _row(
            SECTION_F,
            "Imbalance share of settled volume",
            "% of settled",
            imbalance_share["p10"] * 100.0,
            imbalance_share["p50"] * 100.0,
            imbalance_share["p90"] * 100.0,
            decimals=1,
        ),
        _row(
            SECTION_F,
            "Baseline-effect share of settled volume",
            "% of settled",
            baseline_share["p10"] * 100.0,
            baseline_share["p50"] * 100.0,
            baseline_share["p90"] * 100.0,
            decimals=1,
            reference="can be negative",
        ),
        # Trading contract v1 section 9.1 and 9.8 (decision 0004 item 58):
        # the trader-metric rows added to ``trading_kpis``, ``full`` strategy.
        _row(
            SECTION_F,
            "Capture rate",
            "% of perfect foresight",
            capture["p10"] * 100.0,
            capture["p50"] * 100.0,
            capture["p90"] * 100.0,
            decimals=1,
            reference=(
                f"{int(capture['world_count'])} weeks with a positive perfect-foresight margin"
            ),
        ),
        _money_row(
            SECTION_F,
            "Value of intraday",
            "£ per week",
            full.loc["value_of_intraday_gbp_per_week"],
            reference="full strategy only",
        ),
        _money_row(
            SECTION_F,
            "Cost of uncertainty",
            "£ per week",
            full.loc["cost_of_uncertainty_gbp_per_week"],
            # Model questions Q-3: the benchmark knows the volume, not the
            # imbalance price, so a negative value is real imbalance P&L.
            reference=(
                "against knowing the volume in advance and selling it all day-ahead; negative "
                "when leaving volume to imbalance earned more"
            ),
        ),
        # Signed as the ledger, positive = income (trading contract v1 §5.5;
        # final critique B-4): "cost" read an income as a cost.
        _money_row(
            SECTION_F,
            "Imbalance cash per MWh traded",
            "£ per MWh",
            full.loc["imbalance_gbp_per_mwh_traded"],
            decimals=1,
            reference="positive is income, negative a cost",
        ),
        _row(
            SECTION_F,
            "Firmness",
            "% delivered within the final position",
            firmness["p10"] * 100.0,
            firmness["p50"] * 100.0,
            firmness["p90"] * 100.0,
            decimals=1,
        ),
        _money_row(
            SECTION_F,
            "Trading margin per MW per year",
            "£ per MW per year",
            full.loc["flex_margin_gbp_per_mw_year"],
            reference="extrapolated from one simulated week",
        ),
        _money_row(
            SECTION_F,
            "Day-ahead spread",
            "£ per MWh",
            full.loc["day_ahead_spread_gbp_per_mwh"],
            reference="same for every strategy",
        ),
        # Intraday dispatch contract v1 §6.3, §10: 0 by definition with the
        # switch off (the frozen-book re-run then equals the actual book).
        _money_row(
            SECTION_F,
            "Illustrative re-optimisation P&L",
            "£ per week",
            full.loc["reoptimisation_gbp_per_week"],
            reference="on top of rebalancing; 0 with intraday dispatch off",
        ),
        _row(
            SECTION_F,
            "Energy moved by intraday dispatch",
            "MWh per week, illustrative",
            full.loc["dispatch_moved_mwh_per_week", "p10"],
            full.loc["dispatch_moved_mwh_per_week", "p50"],
            full.loc["dispatch_moved_mwh_per_week", "p90"],
            decimals=2,
            reference="net per half-hour; 0 with intraday dispatch off",
        ),
    ]


def key_stats_table(result: Any) -> pd.DataFrame:
    """Every Key stats row, sections A-E (C-E on an action result only), F, G, H, then I.

    Returns strings throughout: each row carries its own unit and precision,
    which one numeric column per statistic could not show.
    """

    rows = _section_a(result) + _section_b(result)
    if result.model == "action":
        rows += _section_c(result) + _section_d(result) + _section_e(result)
    rows += _section_f(result)
    if result.model == "action":
        rows += _section_g(result)
        rows += _section_h(result)
        rows += _section_i(result)
    return pd.DataFrame(rows, columns=list(COLUMNS))


_COLUMN_CONFIG = {
    "Section": column_config.TextColumn(width="medium"),
    # Pixel widths sized to the longest strings in a default run, so neither
    # column cuts text: "medium" (200 px) cut the metric names where they
    # differ ("Below-target power per plugged-in EV at 18:00" / "at 22:00"
    # read as three identical rows); the longest unit ("£ (illustrative) per
    # household per month") fits 280 px, so "large" (400 px) wasted width.
    # Not pinned: at 1440 px every column now fits without scrolling, and at
    # 390 px a pinned 290 px column would leave the numbers a 70 px strip.
    "Metric": column_config.TextColumn(width=290),
    "Unit": column_config.TextColumn(width=280),
    "P10": column_config.TextColumn(
        width="small", alignment="right", help="10th percentile across simulated weeks"
    ),
    "P50": column_config.TextColumn(
        width="small", alignment="right", help="Median across simulated weeks"
    ),
    "P90": column_config.TextColumn(
        width="small", alignment="right", help="90th percentile across simulated weeks"
    ),
    "Reference": column_config.TextColumn(
        width="large", help="CNZ or workbook context, not a target"
    ),
    "Δ vs unmanaged": column_config.TextColumn(
        width="small", help="Smart minus unmanaged, paired week by week; × is a ratio"
    ),
    "Verdict": column_config.TextColumn(width="medium", help="A short rule-based reading aid"),
}


def render_key_stats(st: Any, result: Any) -> None:
    """Render Overview ▸ Key stats: one dense P10/P50/P90 table."""

    st.dataframe(
        key_stats_table(result),
        hide_index=True,
        width="stretch",
        height="content",
        column_config=_COLUMN_CONFIG,
    )
    st.caption(
        f"Each figure is worked out per simulated week first, then P10–P90 across "
        f"{result.world_count} weeks, so rows do not add up. Money is illustrative."
    )
    if result.model != "action":
        st.caption("Smart charging, coordination and driver-risk rows appear after a smart run.")
    st.caption(UNMANAGED_GLOSSARY)
