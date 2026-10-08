"""Print the headline numbers quoted in the interview notes.

This script owns nothing about the model. It runs the app's own
``run_forecast_from_assumptions`` (every input from ``model/assumptions.py``)
at the app defaults, with a fixed London start date so the figures are
reproducible, and prints the numbers ``docs/explainers/interview-notes.md``,
``build-notes.md`` and ``how-the-forecast-works.md`` quote: smart charging,
trading (including the intraday dispatch and baseline-erosion rows),
the firm-MW product and its calibration backtest, the supplier P&L and
household outcomes. Every band is a statistic across simulated weeks,
already taken after each week's own totals by the model layer (world-first,
decision 0004 item 12); this script only selects and formats rows from the
result, it does not itself take a quantile across worlds, except for the
plug-in gap share, which is pooled across a small event sample instead (see
``plug_in_gap_share``). Every GBP figure is an illustrative valuation of an
energy difference on synthetic prices, never a bill, settlement or Axle
cash (decisions 0003 and 0004 item 4).

With ``--sensitivities`` it also reruns the same seed with one assumption
changed at a time (day-ahead price noise SD 5 and 25 GBP/MWh, departure
margin 1 and 3 h, intraday dispatch off; goal review item 23) and prints
the same headline numbers for each, so the deltas against the default can
be read off. The margin pair moved from 0.5/2 h to 1/3 h because the
departure margin's own default moved to 2 h (decision 0004 item 51), so a
"margin 2 h" row would no longer be a change from default.

Usage (from the repo root):

    PYTHONPATH=src uv run --frozen --no-sync python scripts/headline_numbers.py
    PYTHONPATH=src uv run --frozen --no-sync python scripts/headline_numbers.py --sensitivities
"""

from __future__ import annotations

import argparse
import time
from datetime import date

import numpy as np
import pandas as pd

from axle_studio.model import market
from axle_studio.model.forecast import ForecastResult, run_forecast_from_assumptions

SENSITIVITIES = (
    ("day-ahead noise SD 5 GBP/MWh", {"day_ahead_sd_gbp_per_mwh": 5.0}),
    ("day-ahead noise SD 25 GBP/MWh", {"day_ahead_sd_gbp_per_mwh": 25.0}),
    ("departure margin 1 h", {"departure_margin_hours": 1.0}),
    ("departure margin 3 h", {"departure_margin_hours": 3.0}),
    ("intraday dispatch off", {"trading.intraday_dispatch": 0}),
)


def _p10_p50_p90(values: np.ndarray) -> tuple[float, float, float]:
    """P10, P50 and P90 across simulated weeks (linear interpolation, as the model)."""

    low, mid, high = np.quantile(np.asarray(values, dtype=float), (0.1, 0.5, 0.9))
    return float(low), float(mid), float(high)


def _one_row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    """The one row of ``frame`` whose named columns equal ``filters``.

    A stand-in for the UI's own ``supplier_common.stat_row``: this script
    is a plain Python report, not a Streamlit view, and unlike the UI it
    should fail loudly (not print "Unavailable") if a frame's shape ever
    changes under it, so a missing or duplicated row is a ``ValueError``.
    """

    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        mask &= frame[column].eq(value)
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row for {filters}, found {len(rows)}")
    return rows.iloc[0]


def _stat_band(row: pd.Series, fmt: str) -> str:
    """ "P50 x (P10 y, P90 z)" from a row the model already aggregated world-first."""

    low, mid, high = (row[q] for q in ("p10", "p50", "p90"))
    return f"P50 {mid:{fmt}} (P10 {low:{fmt}}, P90 {high:{fmt}})"


def _percent_band(row: pd.Series, decimals: int = 1) -> str:
    """ "P50 x% (P10 y%, P90 z%)" from a row whose p10/p50/p90 are a fraction, not a percent."""

    low, mid, high = (100.0 * row[q] for q in ("p10", "p50", "p90"))
    return f"P50 {mid:.{decimals}f}% (P10 {low:.{decimals}f}%, P90 {high:.{decimals}f}%)"


def plug_in_gap_share(events: pd.DataFrame) -> str:
    """Share of plug-ins starting 2 or more days after the same EV's previous plug-in.

    Compared against CNZ's own ~30% finding (report Table 2), cited as the
    target this model's multi-day sessions were built to reach
    (``sampling.py:sample_connection_opportunities``, decision 0004 item
    68): an EV now unplugs only on a morning it drives, so a low-mileage
    EV can stay plugged in across several non-driving days, which is what
    lets a long gap happen at all under the archetype sheet's near-daily
    plug probabilities. ``events`` is ``result.plug_in_events``, which
    keeps only ``PLUG_IN_EVENT_WORLD_LIMIT`` (10) simulated weeks (a memory
    limit on the detail rows, not a modelling choice), so this is one
    share pooled across that sample's plug-ins, not a world-first
    P10-P50-P90 band: ten worlds is too few for a meaningful spread, and
    the CNZ figure it is read against is itself a single reported number.
    A gap is measured plug-in to plug-in (not plug-out to plug-in), because
    that is what CNZ's own "days since last plug-in" reading measures.
    """

    ordered = events.sort_values(["world_id", "unit_id", "plug_in_utc"])
    gap_days = (
        ordered.groupby(["world_id", "unit_id"])["plug_in_utc"].diff().dt.total_seconds() / 86400.0
    ).dropna()
    share = float((gap_days >= 2.0).mean())
    weeks = int(events["world_id"].nunique())
    return (
        f"{100 * share:.0f}% ({len(gap_days)} plug-ins with a previous one, {weeks} sampled weeks)"
    )


def headline(result: ForecastResult, seconds: float) -> dict[str, str]:
    """The quoted headline numbers for one action run, as printable strings."""

    world = result.smart_charging_world
    cost = result.cost_effect
    # Saving is normal minus selected, per week first: positive means the smart
    # path cost less. Illustrative GBP only.
    saving = -cost["illustrative_selected_minus_normal_total_gbp"].to_numpy()
    # Each week's own peak, taken inside the week first (contract 4.5a,
    # decision 0004 items 48-49).
    peaks = result.weekly_peak_summary.set_index("path_id")
    # Built by the model (contract 4.7b, summaries.cheapest_half_hour_summary).
    cheapest = result.cheapest_half_hour_summary.iloc[0]
    soc = result.plug_in_summary.kpis.query(
        "day_type == 'all' and metric == 'median_plug_in_soc_percent'"
    )
    weeks = len(cost)
    any_short = int(cost["energy_not_recovered"].sum())
    material = int(cost["not_recovered_material"].sum())

    def band(values: np.ndarray, fmt: str) -> str:
        low, mid, high = _p10_p50_p90(values)
        return f"P50 {mid:{fmt}} (P10 {low:{fmt}}, P90 {high:{fmt}})"

    def peak(path: str, prefix: str = "") -> str:
        row = peaks.loc[path]
        fmt = ".2f" if prefix else ",.0f"
        low, mid, high = (row[f"{prefix}{q}"] for q in ("p10", "p50", "p90"))
        return f"P50 {mid:{fmt}} (P10 {low:{fmt}}, P90 {high:{fmt}})"

    def peak_when(path: str) -> str:
        row = peaks.loc[path]
        when = row["modal_peak_interval_start_london"].strftime("%a %H:%M")
        return f"{when} ({row['modal_peak_week_count']} of {row['world_count']} weeks)"

    # How far the fleet average moves between simulated weeks: P90 - P10 per
    # half-hour on the normal path, then the median over the week's half-hours.
    bands = result.fleet_interval_bands.query("path_id == 'normal'").set_index("metric")
    width = (bands["p90"] - bands["p10"]).groupby(level="metric").median()

    # Trading (trading contract v1 section 9.1; decision 0004 items 57-58,
    # 0005): the three strategies share one physics run, so their net P&L
    # is directly comparable on the same random futures. "full" is the
    # product row for capture rate, value of intraday and the intraday
    # dispatch split (intraday-dispatch-v1 section 6.3); those three KPIs
    # are 0 (or 1 for capture rate's cvar-free ratio) with the dispatch
    # switch off, not absent, so this reads the same whether or not that
    # sensitivity is on.
    kpis = result.trading_kpis.set_index(["strategy", "metric"])
    full = kpis.loc["full"]
    reoptimisation = full.loc["reoptimisation_gbp_per_week"]
    dispatch_moved = full.loc["dispatch_moved_mwh_per_week"]
    erosion = full.loc["evening_baseline_erosion_share"]

    # Firm MW (trading contract v1 section 10.5a; decision 0004 item 59):
    # the product sheet's own headline reference cell, matching Supplier
    # Firm MW's default controls (evening window, 1 h hold, turn-down).
    # "Firm" means the across-weeks P10 throughout that lens.
    firm = _one_row(
        result.product_sheet, window="evening", direction="turn_down", duration_hours=1.0
    )

    # Calibration backtest (trading contract v1 section 10.3): the same
    # (day-ahead, evening, 1 h, turn-down) cell's P10 and P90 hit rates,
    # strict (target <) to weak (target <=); a calibrated forecast sits
    # between the two, not on either alone.
    calibration = _one_row(
        result.availability_backtest_summary,
        horizon="day_ahead",
        direction="turn_down",
        duration_hours=1.0,
    )

    # Supplier P&L (supplier contract v1 section 3; decision 0006): the
    # profiled hedge is the like-for-like headline; "before fee" because
    # the platform fee is unset (NaN) at the app defaults (decision 0003:
    # an unset commercial term is unavailable, never GBP 0), so the
    # after-fee net cannot be quoted from a default run.
    supplier_net = _one_row(
        result.supplier_pnl_summary,
        hedge_variant="profiled",
        metric="net_gain_before_fee_per_customer_per_month_gbp",
    )

    # Household outcomes (household contract v1 section 3.2, set B): each
    # customer's own mean saving across the simulated weeks, then P10-P50-P90
    # across customers. "8 in 10 customers" is everyone between the P10 and
    # P90 customer, the same reading Drivers/Supplier Customers use for a
    # per-year or per-month claim (set B is the only reading such a claim
    # may use, B1).
    household = result.household_outcomes_summary
    saving_year = {
        statistic: _one_row(
            household, group_id="fleet", metric="saving_gbp_per_year", statistic=statistic
        )["p50"]
        for statistic in ("p10_of_ev_means", "p50_of_ev_means", "p90_of_ev_means")
    }

    return {
        "normal average price paid, GBP/MWh (illustrative)": band(
            world["normal_average_price_gbp_per_mwh"], ".0f"
        ),
        "smart average price paid, GBP/MWh (illustrative)": band(
            world["selected_average_price_gbp_per_mwh"], ".0f"
        ),
        "weekly saving, GBP per fleet week (illustrative)": band(saving, ",.0f"),
        "share of home import moved, %": band(100 * world["moved_home_import_share"], ".0f"),
        "early departures, sessions per week": band(world["early_departure_count"], ".0f"),
        "unrecovered share of normal home import, %": band(100 * cost["unrecovered_share"], ".3f"),
        "unrecovered energy, kWh per week": band(cost["unrecovered_kwh"], ".1f"),
        "weeks with any unrecovered energy": f"{any_short} of {weeks}",
        "weeks materially not recovered": f"{material} of {weeks}",
        "normal weekly peak home import, kW": peak("normal"),
        "normal peak most often at (London)": peak_when("normal"),
        "smart weekly peak home import, kW": peak("selected"),
        "smart peak most often at (London)": peak_when("selected"),
        "smart peak / normal peak (per-week ratio)": peak("selected", "ratio_to_normal_"),
        "cheapest day-ahead half-hour each day, London": ", ".join(
            f"P{level} {cheapest[f'local_time_label_p{level}']}" for level in (10, 50, 90)
        ),
        "median across half-hours of P90-P10, plugged-in share, % points": (
            f"{100 * width['connected_share']:.1f}"
        ),
        "median across half-hours of P90-P10, fleet SoC, % points": (
            f"{width['battery_soc_percent']:.1f}"
        ),
        "median plug-in SoC, % (P50 across weeks)": f"{soc['p50'].iat[0]:.0f}",
        "share of plug-ins 2+ days after the last (CNZ context ~30%)": plug_in_gap_share(
            result.plug_in_events
        ),
        **{
            f"trading net, GBP per week (illustrative simulated), {strategy}": _stat_band(
                kpis.loc[(strategy, "net_gbp_per_week")], ",.0f"
            )
            for strategy in market.STRATEGIES
        },
        "trading capture rate, full vs perfect foresight, %": _percent_band(
            full.loc["capture_rate"]
        ),
        "value of intraday, GBP per week (illustrative simulated, full - day-ahead only)": (
            _stat_band(full.loc["value_of_intraday_gbp_per_week"], ",.0f")
        ),
        "dispatch re-optimisation P&L, GBP per week (illustrative simulated)": _stat_band(
            reoptimisation, ",.0f"
        ),
        "dispatch energy moved, MWh per week (P50)": f"{dispatch_moved['p50']:.2f}",
        "evening baseline erosion share, 16:00-20:00 (mean ratio across weeks)": (
            f"{100 * erosion['mean']:.0f}%"
        ),
        "firm MW, evening 1 h turn-down, P10 across weeks (product sheet)": (
            f"{firm['window_mean_mw_p10']:.2f} MW (P50 {firm['window_mean_mw_p50']:.2f} MW)"
        ),
        "calibration hit rate at P10 (day-ahead, evening 1 h turn-down), strict-weak": (
            f"{100 * calibration['coverage_strict_p10']:.1f}%-"
            f"{100 * calibration['coverage_p10']:.1f}% (target 10%)"
        ),
        "calibration hit rate at P90 (day-ahead, evening 1 h turn-down), strict-weak": (
            f"{100 * calibration['coverage_strict_p90']:.1f}%-"
            f"{100 * calibration['coverage_p90']:.1f}% (target 90%)"
        ),
        "supplier net per customer per month before fee, GBP (illustrative, profiled hedge)": (
            _stat_band(supplier_net, ",.2f")
        ),
        "household saving per year, GBP per customer (illustrative, set B, 8 in 10 customers)": (
            f"P50 {saving_year['p50_of_ev_means']:,.0f} "
            f"(P10 {saving_year['p10_of_ev_means']:,.0f}, "
            f"P90 {saving_year['p90_of_ev_means']:,.0f})"
        ),
        "run time, s (model run only, this machine)": f"{seconds:.1f}",
    }


def run(start: date, values: dict[str, object]) -> dict[str, str]:
    """Run one action forecast with the given edited assumptions and time it."""

    began = time.perf_counter()
    result = run_forecast_from_assumptions(start, model="action", values=values)
    return headline(result, time.perf_counter() - began)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2026-01-12", help="London study start date (Monday)")
    parser.add_argument("--vehicles", type=int, default=None, help="default: app default")
    parser.add_argument("--weeks", type=int, default=None, help="default: app default")
    parser.add_argument("--seed", type=int, default=None, help="default: app default")
    parser.add_argument("--sensitivities", action="store_true")
    args = parser.parse_args()

    base: dict[str, object] = {}
    for name, value in (
        ("vehicle_count", args.vehicles),
        ("evaluation_world_count", args.weeks),
        ("seed", args.seed),
    ):
        if value is not None:
            base[name] = value
    start = date.fromisoformat(args.start)
    runs = [("default", base)]
    if args.sensitivities:
        runs += [(label, {**base, **change}) for label, change in SENSITIVITIES]
    table = {label: run(start, values) for label, values in runs}
    print(f"start {start}, edited values {base or 'none (app defaults)'}")
    with pd.option_context("display.max_colwidth", None, "display.width", 250):
        print(pd.DataFrame(table).to_string())


if __name__ == "__main__":
    main()
