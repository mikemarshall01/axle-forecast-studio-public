"""Fit the daily price shape and noise statistics that calibrate the synthetic
day-ahead price generator, from real Elexon market data fetched by
`scripts/fetch_elexon_prices.py` (decision 0004 item 53; plan
`docs/plans/2026-09-29-analyst-trading-polish-plan.md` section 2B item B2).

This script owns nothing about the model: it prints coefficients and stats
for a human (or Lane 2's B1 writer) to read and paste into
`model/assumptions.py`. It never runs inside the Streamlit app.

Model structure being calibrated (plan item B1): a simulated day-ahead price
is `level_d + shape(half_hour, weekday/weekend, winter/summer) + noise`,
where `level_d` is a slow day-to-day AR(1) random walk and `shape` is a fixed
intraday pattern that differs by weekday/weekend and by season. This script
estimates each piece separately from history:

1. Daily-mean level: the SD and lag-1 AR(1) coefficient of each day's own
   mean price, over the whole fetched range (one global process, not split
   by weekday/season -- the weekday/season pattern is the shape, below).
2. Daily-mean spread: P10/P50/P90 of the daily-mean series, and the
   negative-price half-hour share.
3. Intraday shape: for each of the four (weekday/weekend x winter/summer)
   groups, the mean price by half-hour-of-day, fitted with a 3-harmonic
   Fourier series by ordinary least squares (`np.linalg.lstsq`).
4. Half-hour residual noise: what is left after removing a day's own mean
   and its group's shape deviation -- its SD and lag-1 AR(1) coefficient.
5. System (imbalance) price vs market index price: split by Net Imbalance
   Volume (NIV) sign, comparing the mean system-price premium in each half
   (plan item B5's two-state NIV premium).

Half-hour-of-day and weekday/weekend/season groupings use the API's own
`settlement_date`/`settlement_period` fields directly, not a UTC-to-London
timezone conversion of `start_time_utc`. Alternative rejected: GB settlement
periods are *defined* against the London clock (that is what "settlement
date" means), so converting `start_time_utc` via zoneinfo would just
reconstruct the same settlement_date/settlement_period pair Elexon already
publishes -- redoing that conversion locally would be extra machinery for no
more correctness.

Usage (from the repo root):

    PYTHONPATH=src uv run --frozen --no-sync python scripts/fit_price_shape.py \\
        --market-index-csv data/elexon/market_index_2025-09-29_2026-09-28.csv \\
        --system-prices-csv data/elexon/system_prices_2025-09-29_2026-09-28.csv \\
        --chart /tmp/elexon_price_shape_check.html

With no `--market-index-csv`/`--system-prices-csv`, the most recently
modified matching file in `data/elexon/` is used.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from axle_studio.ui import style

HALF_HOURS_PER_DAY = 48
N_HARMONICS = 3
WINTER_MONTHS = {10, 11, 12, 1, 2, 3}
GROUPS = ("weekday_winter", "weekday_summer", "weekend_winter", "weekend_summer")
COEFFICIENT_NAMES = ("a0", "a1", "b1", "a2", "b2", "a3", "b3")


def half_hour_bucket(settlement_period: pd.Series) -> pd.Series:
    """Map a 1-based settlement period to a 0-47 half-hour-of-day bucket.

    Normal days have 48 periods (period 1 = 00:00-00:30 London local); the
    two clock-change days a year have 46 (spring forward) or 50 (autumn
    back, a repeated hour). Periods 49/50 fold onto buckets 0/1 by modulo,
    slightly double-weighting those two buckets on one day a year -- an
    immaterial simplification for a mean profile fitted over a full year,
    recorded as a limitation rather than handled with extra clock-change
    machinery.
    """
    return (settlement_period.astype(int) - 1) % HALF_HOURS_PER_DAY


def season_of(month: pd.Series) -> pd.Series:
    """ "winter" (Oct-Mar) or "summer" (Apr-Sep) for a calendar month, per plan item B1."""
    return np.where(month.isin(WINTER_MONTHS), "winter", "summer")


def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add `bucket`, `day_type`, `season` and `group` columns to a market-index frame.

    `settlement_date` is parsed as a date (not tz-aware): it is already the
    London settlement calendar date, so no timezone conversion is applied.
    """
    out = df.copy()
    dates = pd.to_datetime(out["settlement_date"])
    out["bucket"] = half_hour_bucket(out["settlement_period"])
    out["day_type"] = np.where(dates.dt.dayofweek < 5, "weekday", "weekend")
    out["season"] = season_of(dates.dt.month)
    out["group"] = out["day_type"] + "_" + out["season"]
    return out


def fourier_design_matrix(
    t: np.ndarray, n_harmonics: int = N_HARMONICS, period: int = HALF_HOURS_PER_DAY
) -> np.ndarray:
    """Design matrix [1, cos(2*pi*k*t/period), sin(2*pi*k*t/period) for k=1..n_harmonics]."""
    columns = [np.ones_like(t, dtype=float)]
    for k in range(1, n_harmonics + 1):
        angle = 2 * np.pi * k * t / period
        columns.append(np.cos(angle))
        columns.append(np.sin(angle))
    return np.column_stack(columns)


def bucket_to_clock(bucket: float) -> str:
    """Half-hour bucket (0-47) to an HH:MM London clock string."""
    minutes = round(bucket) * 30
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def fit_fourier_shape(profile: pd.Series) -> tuple[dict[str, float], np.ndarray]:
    """Least-squares 3-harmonic Fourier fit to a 48-point mean half-hourly price profile.

    `profile` is indexed 0-47 (half-hour bucket) with mean GBP/MWh values.
    Returns the named coefficients (`a0`..`b3`) plus derived read-off stats
    (R-squared, trough/peak local time and level, peak-to-trough swing), and
    the fitted 48-point curve itself (used later to isolate the half-hour
    noise residual).
    """
    t = profile.index.to_numpy(dtype=float)
    y = profile.to_numpy(dtype=float)
    design = fourier_design_matrix(t)
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coeffs
    residual_ss = float(np.sum((y - fitted) ** 2))
    total_ss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else float("nan")
    trough_bucket = int(np.argmin(fitted))
    peak_bucket = int(np.argmax(fitted))
    result = dict(zip(COEFFICIENT_NAMES, (float(c) for c in coeffs), strict=True))
    result.update(
        {
            "r_squared": r_squared,
            "trough_local_time": bucket_to_clock(trough_bucket),
            "trough_gbp_per_mwh": float(fitted[trough_bucket]),
            "peak_local_time": bucket_to_clock(peak_bucket),
            "peak_gbp_per_mwh": float(fitted[peak_bucket]),
            "swing_gbp_per_mwh": float(fitted[peak_bucket] - fitted[trough_bucket]),
        }
    )
    return result, fitted


def build_group_profile(df: pd.DataFrame, group: str) -> pd.Series:
    """Mean price by half-hour bucket (0-47) for one weekday/weekend x season group."""
    subset = df.loc[df["group"] == group]
    return subset.groupby("bucket")["price_gbp_per_mwh"].mean().reindex(range(HALF_HOURS_PER_DAY))


def ar1_phi(series: np.ndarray) -> float:
    """Lag-1 AR(1) coefficient of a series, by OLS on the demeaned series (`np.linalg.lstsq`)."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    design = x[:-1].reshape(-1, 1)
    target = x[1:]
    phi, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(phi[0])


def daily_mean_series(df: pd.DataFrame) -> pd.Series:
    """Mean half-hourly price per calendar day (London settlement date), sorted chronologically."""
    return df.groupby("settlement_date")["price_gbp_per_mwh"].mean().sort_index()


def compute_residuals(
    df: pd.DataFrame, group_fits: dict[str, np.ndarray], daily_means: pd.Series
) -> np.ndarray:
    """Half-hour noise residual: price minus the day's own mean minus the group's zero-mean shape.

    This isolates the fast half-hour AR(1) noise from the slow daily-mean
    level (a separate random-walk component) and from the fixed intraday
    shape, matching the additive model `level_d + shape + noise` (plan item
    B1): both other terms are subtracted out first, so what remains is what
    the noise term alone is meant to explain.
    """
    residual = np.empty(len(df), dtype=float)
    day_level = df["settlement_date"].map(daily_means).to_numpy(dtype=float)
    prices = df["price_gbp_per_mwh"].to_numpy(dtype=float)
    for group, fitted in group_fits.items():
        zero_mean_shape = fitted - fitted.mean()
        mask = (df["group"] == group).to_numpy()
        buckets = df.loc[mask, "bucket"].to_numpy()
        residual[mask] = prices[mask] - day_level[mask] - zero_mean_shape[buckets]
    return residual


def summarise_market_index(df: pd.DataFrame) -> dict:
    """Fit the shape and noise statistics from an assigned (grouped) market-index frame."""
    group_fits: dict[str, dict[str, float]] = {}
    group_curves: dict[str, np.ndarray] = {}
    for group in GROUPS:
        profile = build_group_profile(df, group)
        coeffs, fitted = fit_fourier_shape(profile)
        group_fits[group] = coeffs
        group_curves[group] = fitted

    daily_means = daily_mean_series(df)
    residuals = compute_residuals(df, group_curves, daily_means)

    daily_values = daily_means.to_numpy(dtype=float)
    p10, p50, p90 = np.quantile(daily_values, (0.1, 0.5, 0.9))

    return {
        "groups": group_fits,
        "level": {
            "sd_gbp_per_mwh": float(np.std(daily_values, ddof=1)),
            "ar1_phi": ar1_phi(daily_values),
            "negative_price_share": float((df["price_gbp_per_mwh"] < 0).mean()),
            "daily_mean_p10_gbp_per_mwh": float(p10),
            "daily_mean_p50_gbp_per_mwh": float(p50),
            "daily_mean_p90_gbp_per_mwh": float(p90),
            "daily_mean_p90_minus_p10_gbp_per_mwh": float(p90 - p10),
        },
        "residual": {
            "ar1_phi": ar1_phi(residuals),
            "sd_gbp_per_mwh": float(np.std(residuals, ddof=1)),
        },
        "n_rows": int(len(df)),
        "n_days": int(daily_means.shape[0]),
    }


def summarise_system_prices(market_df: pd.DataFrame, system_df: pd.DataFrame) -> dict:
    """Compare the system (imbalance) price to the market index price by settlement period.

    Sign convention: Net Imbalance Volume (NIV) > 0 is treated as a "short"
    period (National Grid ESO net accepted more balancing offers than bids)
    and NIV < 0 as "long". This is the convention used in GB power-market
    commentary (e.g. Modo Energy, LCP Delta write-ups) but Elexon's own
    published NIV glossary entry does not spell out the sign, so treat the
    long/short *labels* as an assumption -- the NIV-sign split itself and the
    system-minus-market premium numbers are read directly from the data
    regardless of which label is attached to which sign (see limitations in
    docs/research/elexon-price-calibration.md).
    """
    mip = market_df.groupby(["settlement_date", "settlement_period"], as_index=False)[
        "price_gbp_per_mwh"
    ].mean()
    mip = mip.rename(columns={"price_gbp_per_mwh": "mip_gbp_per_mwh"})
    merged = mip.merge(
        system_df[
            [
                "settlement_date",
                "settlement_period",
                "system_sell_price_gbp_per_mwh",
                "net_imbalance_volume_mwh",
            ]
        ],
        on=["settlement_date", "settlement_period"],
        how="inner",
    )
    merged["premium_gbp_per_mwh"] = (
        merged["system_sell_price_gbp_per_mwh"] - merged["mip_gbp_per_mwh"]
    )
    short = merged["net_imbalance_volume_mwh"] > 0
    long_ = merged["net_imbalance_volume_mwh"] < 0
    return {
        "n_periods_joined": int(len(merged)),
        "share_niv_positive_short": float(short.mean()),
        "share_niv_negative_long": float(long_.mean()),
        "mean_sip_premium_niv_positive_gbp_per_mwh": float(
            merged.loc[short, "premium_gbp_per_mwh"].mean()
        ),
        "mean_sip_premium_niv_negative_gbp_per_mwh": float(
            merged.loc[long_, "premium_gbp_per_mwh"].mean()
        ),
    }


def _find_latest(pattern: str) -> Path | None:
    """Most recently modified file matching a glob pattern, or None if there is no match."""
    matches = glob.glob(pattern)
    if not matches:
        return None
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))


def _shape_table(groups: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Copy-pasteable table of Fourier coefficients and read-off stats, one row per group."""
    return pd.DataFrame(groups).T[
        list(COEFFICIENT_NAMES)
        + [
            "r_squared",
            "trough_local_time",
            "trough_gbp_per_mwh",
            "peak_local_time",
            "peak_gbp_per_mwh",
            "swing_gbp_per_mwh",
        ]
    ]


def build_check_figure(df: pd.DataFrame, groups: dict[str, dict[str, float]]) -> go.Figure:
    """Observed mean profile (markers) plus fitted 3-harmonic curve (line) per group.

    A visual check that the harmonic fit tracks the real overnight trough and
    evening peak shape, not a shipped app chart -- reuses the app's own
    Plotly template (`ui/style.py`) for consistent, readable styling
    (decision 0004 item 29), but this script and its chart run outside the
    Streamlit app.
    """
    figure = go.Figure()
    clock = [bucket_to_clock(b) for b in range(HALF_HOURS_PER_DAY)]
    for group, colour in zip(GROUPS, ("#9AA9BF", "#2DD4BF", "#F2B45A", "#A78BFA"), strict=False):
        profile = build_group_profile(df, group)
        _, fitted = fit_fourier_shape(profile)
        figure.add_trace(
            go.Scatter(
                x=clock,
                y=profile.to_numpy(),
                mode="markers",
                name=f"{group} observed",
                marker={"color": colour, "size": 5},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=clock,
                y=fitted,
                mode="lines",
                name=f"{group} 3-harmonic fit",
                line={"color": colour, "width": 2},
            )
        )
    figure.update_layout(
        title="Elexon APXMIDP mean half-hourly price: observed vs 3-harmonic fit, by group",
        xaxis_title="London local half-hour",
        yaxis_title="Price (GBP/MWh)",
    )
    return style.apply_notebook_style(figure, height=style.CHART_HEIGHTS.get("time_series", 420))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--market-index-csv", type=Path, default=None)
    parser.add_argument("--system-prices-csv", type=Path, default=None)
    parser.add_argument(
        "--chart", type=Path, default=None, help="optional path to write a Plotly HTML check chart"
    )
    args = parser.parse_args()

    market_path = args.market_index_csv or _find_latest("data/elexon/market_index_*.csv")
    if market_path is None:
        parser.error("no --market-index-csv given and none found under data/elexon/")
    print(f"Market index: {market_path}")
    market_df = assign_groups(pd.read_csv(market_path))

    result = summarise_market_index(market_df)
    print(f"\nRows: {result['n_rows']} half-hours over {result['n_days']} days\n")

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("Intraday shape (3-harmonic Fourier fit, GBP/MWh):")
        print(_shape_table(result["groups"]).to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nDaily-mean level:")
    for key, value in result["level"].items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\nHalf-hour residual noise:")
    for key, value in result["residual"].items():
        print(f"  {key}: {value:.4f}")

    system_path = args.system_prices_csv or _find_latest("data/elexon/system_prices_*.csv")
    if system_path is not None:
        print(f"\nSystem prices: {system_path}")
        system_df = pd.read_csv(system_path)
        system_result = summarise_system_prices(market_df, system_df)
        print("\nSystem (imbalance) price vs market index price:")
        for key, value in system_result.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    else:
        print("\nNo system-prices CSV given or found under data/elexon/ -- skipping.")

    if args.chart is not None:
        build_check_figure(market_df, result["groups"]).write_html(args.chart)
        print(f"\nCheck chart written to {args.chart}")


if __name__ == "__main__":
    main()
