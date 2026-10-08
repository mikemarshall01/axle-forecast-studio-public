"""Unit tests for scripts/fit_price_shape.py on synthetic data with known coefficients.

The script lives outside the `axle_studio` package, so it is loaded directly
from its file path (same pattern as tests/test_run_archetypes.py), rather
than imported by module name.
"""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fit_price_shape.py"


def _load_fit_price_shape():
    spec = importlib.util.spec_from_file_location("fit_price_shape", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fps = _load_fit_price_shape()

# One distinct, hand-picked 3-harmonic coefficient set per weekday/weekend x
# season group, used to build a noiseless synthetic dataset: if the fitting
# pipeline recovers these exactly, the grouping, profile-averaging and
# np.linalg.lstsq fit are all wired correctly.
KNOWN_COEFFS = {
    "weekday_winter": {
        "a0": 95.0,
        "a1": -18.0,
        "b1": 6.0,
        "a2": 7.0,
        "b2": -2.5,
        "a3": 1.5,
        "b3": -1.0,
    },
    "weekday_summer": {
        "a0": 70.0,
        "a1": -10.0,
        "b1": 3.0,
        "a2": 4.0,
        "b2": 1.0,
        "a3": 0.5,
        "b3": 0.5,
    },
    "weekend_winter": {
        "a0": 88.0,
        "a1": -14.0,
        "b1": 4.0,
        "a2": 5.0,
        "b2": -1.0,
        "a3": 1.0,
        "b3": 0.2,
    },
    "weekend_summer": {
        "a0": 60.0,
        "a1": -8.0,
        "b1": 2.0,
        "a2": 3.0,
        "b2": 0.5,
        "a3": 0.3,
        "b3": -0.4,
    },
}


def _dates_for(day_type: str, season: str, count: int) -> list[date]:
    """`count` real calendar dates in 2026 matching a weekday/weekend x season combination.

    Computed by walking forward from the season's first month, rather than
    hand-picked literal dates, so the test cannot be wrong about which
    weekday a given date falls on.
    """
    month = 1 if season == "winter" else 7
    found: list[date] = []
    day = date(2026, month, 1)
    while len(found) < count:
        is_weekday = day.weekday() < 5
        if (day_type == "weekday") == is_weekday:
            found.append(day)
        day += timedelta(days=1)
    return found


def _build_synthetic_market_df() -> pd.DataFrame:
    """A noiseless market-index frame: every day in a group repeats that group's exact profile."""
    rows = []
    t = np.arange(fps.HALF_HOURS_PER_DAY, dtype=float)
    design = fps.fourier_design_matrix(t)
    for group, coeffs in KNOWN_COEFFS.items():
        day_type, season = group.split("_")
        coeff_vector = np.array([coeffs[name] for name in fps.COEFFICIENT_NAMES])
        profile = design @ coeff_vector
        for day in _dates_for(day_type, season, count=4):
            for period in range(1, fps.HALF_HOURS_PER_DAY + 1):
                rows.append(
                    {
                        "settlement_date": day.isoformat(),
                        "settlement_period": period,
                        "price_gbp_per_mwh": profile[period - 1],
                        "volume_mwh": 1000.0,
                        "start_time_utc": f"{day.isoformat()}T00:00:00Z",
                    }
                )
    return pd.DataFrame(rows)


def test_half_hour_bucket_wraps_extra_clock_change_periods():
    buckets = fps.half_hour_bucket(pd.Series([1, 48, 49, 50]))
    assert buckets.tolist() == [0, 47, 0, 1]


def test_assign_groups_labels_weekday_weekend_and_season():
    winter_weekday = _dates_for("weekday", "winter", 1)[0]
    winter_weekend = _dates_for("weekend", "winter", 1)[0]
    summer_weekday = _dates_for("weekday", "summer", 1)[0]
    summer_weekend = _dates_for("weekend", "summer", 1)[0]
    df = pd.DataFrame(
        {
            "settlement_date": [
                d.isoformat()
                for d in (winter_weekday, winter_weekend, summer_weekday, summer_weekend)
            ],
            "settlement_period": [1, 1, 1, 1],
            "price_gbp_per_mwh": [0.0, 0.0, 0.0, 0.0],
        }
    )
    grouped = fps.assign_groups(df)
    assert grouped["group"].tolist() == [
        "weekday_winter",
        "weekend_winter",
        "weekday_summer",
        "weekend_summer",
    ]


def test_fit_fourier_shape_recovers_known_coefficients():
    df = fps.assign_groups(_build_synthetic_market_df())
    result = fps.summarise_market_index(df)
    for group, coeffs in KNOWN_COEFFS.items():
        fitted = result["groups"][group]
        for name, value in coeffs.items():
            assert fitted[name] == pytest.approx(value, abs=1e-6)
        assert fitted["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_residuals_are_near_zero_for_noiseless_synthetic_data():
    df = fps.assign_groups(_build_synthetic_market_df())
    result = fps.summarise_market_index(df)
    assert result["residual"]["sd_gbp_per_mwh"] == pytest.approx(0.0, abs=1e-6)


def test_ar1_phi_recovers_known_coefficient():
    rng = np.random.default_rng(42)
    phi_true = 0.6
    n = 2000
    noise = rng.normal(scale=1.0, size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi_true * x[i - 1] + noise[i]
    assert fps.ar1_phi(x) == pytest.approx(phi_true, abs=0.05)


def test_summarise_system_prices_splits_by_niv_sign():
    market_df = pd.DataFrame(
        {
            "settlement_date": ["2026-01-05", "2026-01-05"],
            "settlement_period": [1, 2],
            "price_gbp_per_mwh": [100.0, 100.0],
        }
    )
    system_df = pd.DataFrame(
        {
            "settlement_date": ["2026-01-05", "2026-01-05"],
            "settlement_period": [1, 2],
            "system_sell_price_gbp_per_mwh": [130.0, 80.0],
            "net_imbalance_volume_mwh": [10.0, -5.0],
        }
    )
    result = fps.summarise_system_prices(market_df, system_df)
    assert result["n_periods_joined"] == 2
    assert result["share_niv_positive_short"] == pytest.approx(0.5)
    assert result["share_niv_negative_long"] == pytest.approx(0.5)
    assert result["mean_sip_premium_niv_positive_gbp_per_mwh"] == pytest.approx(30.0)
    assert result["mean_sip_premium_niv_negative_gbp_per_mwh"] == pytest.approx(-20.0)
