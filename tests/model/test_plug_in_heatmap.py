"""Plug-in heatmap by weekday and London hour (decision 0004 item 43, contract 3.10a)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.model.summaries import build_study_slots, plug_in_heatmap


def test_cells_are_plug_ins_per_ev_day_then_quantiles_across_worlds() -> None:
    # Two EVs, a 7-day study from Monday 12 January 2026 (one date per
    # weekday), two worlds.  World 0: both EVs plug in Monday 18:00-18:59;
    # world 1: one EV does.  Monday 18:00 is then 2 / (2 EVs x 1 day) = 1.0
    # and 0.5, so mean 0.75, P10 0.55, P50 0.75, P90 0.95.
    rows = pd.DataFrame(
        {
            "world_id": [0, 0, 1, 1],
            "weekday": [0, 0, 0, 5],
            "local_hour": [18, 18, 18, 17],
        }
    )
    slots = build_study_slots(date(2026, 1, 12))
    heatmap = plug_in_heatmap(rows, world_count=2, vehicle_count=2, study_slots=slots)

    assert len(heatmap) == 168
    assert heatmap["day_count"].eq(1).all() and heatmap["world_count"].eq(2).all()
    monday_six = heatmap.loc[heatmap["weekday"].eq(0) & heatmap["local_hour"].eq(18)].iloc[0]
    assert monday_six["weekday_label"] == "Mon"
    assert monday_six[["mean", "p10", "p50", "p90"]].tolist() == pytest.approx(
        [0.75, 0.55, 0.75, 0.95]
    )
    saturday_five = heatmap.loc[heatmap["weekday"].eq(5) & heatmap["local_hour"].eq(17)].iloc[0]
    assert saturday_five[["mean", "p50"]].tolist() == pytest.approx([0.25, 0.25])
    # A cell with no plug-in in any world is a real zero.
    assert heatmap.loc[heatmap["local_hour"].eq(3), "p90"].eq(0.0).all()


def test_a_weekday_with_no_study_date_is_missing() -> None:
    rows = pd.DataFrame({"world_id": [0], "weekday": [0], "local_hour": [18]})
    slots = build_study_slots(date(2026, 1, 12), study_days=2)  # Monday and Tuesday
    heatmap = plug_in_heatmap(rows, world_count=1, vehicle_count=1, study_slots=slots)
    assert heatmap.loc[heatmap["weekday"] >= 2, "p50"].isna().all()
    assert heatmap.loc[heatmap["weekday"] < 2, "p50"].notna().all()


def test_real_run_heatmap_peaks_in_the_evening_and_splits_by_day_type() -> None:
    result = run_forecast_from_assumptions(
        date(2026, 1, 12),
        model="no_action",
        values={"vehicle_count": 40, "evaluation_world_count": 3},
    )
    heatmap = result.plug_in_heatmap
    # Weekday evenings: the busiest cell of each weekday is between 17:00
    # and 19:59 (weekday arrival 18:00, decision 0004 item 42).
    weekdays = heatmap.loc[heatmap["weekday"] < 5]
    peaks = weekdays.loc[weekdays.groupby("weekday")["mean"].idxmax(), "local_hour"]
    assert peaks.between(17, 19).all()
    # Nothing plugs in at 03:00 in this model.
    assert heatmap.loc[heatmap["local_hour"].eq(3), "p90"].eq(0.0).all()
    # Plug-ins per EV per week are available split by day type (contract 3.10).
    kpis = result.plug_in_summary.kpis.set_index(["day_type", "metric"])
    per_ev = kpis.xs("plug_ins_per_ev_per_week", level="metric")["p50"]
    assert set(per_ev.index) == {"weekday", "weekend", "all"}
    assert np.isfinite(per_ev.to_numpy()).all() and (per_ev > 0).all()
