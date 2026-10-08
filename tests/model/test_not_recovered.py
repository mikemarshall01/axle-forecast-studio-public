"""Not-recovered magnitude per week and across weeks (decision 0004 item 45)."""

from __future__ import annotations

from datetime import date
from functools import cache

import numpy as np

from axle_studio.model.forecast import ForecastResult, run_forecast_from_assumptions
from axle_studio.model.summaries import sessions_affected_per_world

_START = date(2026, 1, 12)


@cache
def _result() -> ForecastResult:
    return run_forecast_from_assumptions(
        _START, model="action", values={"vehicle_count": 60, "evaluation_world_count": 4}
    )


def test_sessions_affected_counts_session_ends_short_on_the_smart_path() -> None:
    # One world, five slots, two EVs.  EV 0 has sessions ending in slots 1
    # and 4 (the horizon end); EV 1 one ending in slot 2.  The smart path is
    # short at EV 0's first end and EV 1's end; EV 0 is short in slot 3 too,
    # but that is mid-session, and within 1e-6 kWh at its last end.
    connected = np.array([[[1, 0], [1, 1], [0, 1], [1, 0], [1, 0]]], dtype=bool)
    normal = np.full((1, 5, 2), 50.0)
    selected = normal.copy()
    selected[0, 1, 0] = 49.0
    selected[0, 2, 1] = 48.0
    selected[0, 3, 0] = 40.0
    selected[0, 4, 0] = 50.0 - 1e-7
    np.testing.assert_array_equal(sessions_affected_per_world(connected, normal, selected), [2])


def test_real_run_reports_magnitude_per_week_and_across_weeks() -> None:
    result = _result()
    cost = result.cost_effect
    assert cost["sessions_affected_count"].dtype == np.int64
    # Material weeks only; an immaterial shortfall still shows as magnitude.
    assert result.not_recovered_world_count == int(cost["not_recovered_material"].sum())
    assert (cost["not_recovered_material"] <= cost["energy_not_recovered"]).all()
    summary = result.not_recovered_summary.set_index("metric")
    assert list(summary.index) == [
        "unrecovered_kwh",
        "unrecovered_share",
        "sessions_affected_count",
    ]
    # World-first: quantiles over the per-week values.
    for metric in summary.index:
        values = cost[metric].to_numpy(dtype=float)
        np.testing.assert_allclose(
            summary.loc[metric, ["p10", "p50", "p90"]].to_numpy(dtype=float),
            np.quantile(values, (0.1, 0.5, 0.9), method="linear"),
        )


def test_no_action_result_has_no_magnitude() -> None:
    result = run_forecast_from_assumptions(
        _START, model="no_action", values={"vehicle_count": 20, "evaluation_world_count": 2}
    )
    assert result.not_recovered_summary is None
    assert result.not_recovered_world_count is None
