"""The firm-MW chain end to end on real (small) runs (trading contract v1 §10.0, §10.9).

A SYNTHETIC illustrative Monte Carlo at 60 EVs x 6 worlds with the default
assumptions: the kernel's plan outputs (J1b) feed the availability frames
(J2), their calibration backtest (J3) and the product frames (J5), and the
result passes ``validate_result_v2`` with every lane validator hooked in.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from functools import cache

import numpy as np
import pandas as pd
import pytest
from fixtures.result_contract import FIRM_MW_FIELDS, validate_result_v2

from axle_studio.model import assumptions
from axle_studio.model.events import EVENT_COLUMNS
from axle_studio.model.forecast import (
    ForecastResult,
    run_forecast,
    run_forecast_from_assumptions,
)
from axle_studio.model.household import household_card

_START = date(2026, 10, 12)  # a Monday, no clock change
_SMALL = {"vehicle_count": 60, "evaluation_world_count": 6}


@cache
def _default() -> ForecastResult:
    return run_forecast_from_assumptions(_START, values=_SMALL)


def _direct(**kwargs) -> ForecastResult:
    inputs = assumptions.forecast_inputs(
        assumptions.resolve_values(_SMALL), warmup_days=7, study_days=7
    )
    trading = kwargs.pop("trading", None)
    if trading is not None:
        inputs["trading_assumptions"] = inputs["trading_assumptions"] | trading
    result = run_forecast(
        assumptions.run_settings(_SMALL, _START),
        assumptions.cohort_fixture(_SMALL),
        **inputs,
        **kwargs,
    )
    return replace(result, assumptions=assumptions.result_assumptions(_SMALL))


def test_default_action_run_fills_every_firm_mw_field_and_passes_the_validator() -> None:
    result = _default()
    for name in (*FIRM_MW_FIELDS, "blackout_windows"):
        assert getattr(result, name) is not None, name
    validate_result_v2(result)
    # The chain is live, not inert: plans exist, so there is deliverable
    # turn-down in the evening, and the backtest has kept world-slots.
    bands = result.availability_bands
    realised = bands.loc[bands["statistic"].eq("realised") & bands["direction"].eq("turn_down")]
    assert realised["p50"].max() > 0.0
    assert result.availability_backtest_summary["sample_count"].sum() > 0
    assert result.world_nights["plugged_in_count_at_decision"].sum() > 0
    # The plan outputs reach the other lanes: household evening kW, the
    # household card's realised-kW profile and the partner dispatch rate.
    evening = result.household_ev_world[["evening_turn_down_kw_1h", "evening_turn_up_kw_1h"]]
    assert evening.notna().all().all()
    assert result.replay_state.availability_inputs is not None
    card = household_card(result, result.units["unit_id"].iat[0])
    assert card.availability is not None
    rate = result.partner_summary.query("group_id == 'fleet' and metric == 'dispatch_success_rate'")
    assert rate["p50"].between(0.0, 1.0).all()


def test_no_action_run_has_only_the_blackout_table() -> None:
    result = run_forecast_from_assumptions(_START, model="no_action", values=_SMALL)
    validate_result_v2(result)
    assert result.blackout_windows is not None and result.blackout_windows.empty
    for name in FIRM_MW_FIELDS:
        assert getattr(result, name) is None, name


def test_newsvendor_commitment_reaches_the_ledger_and_passes_the_validator() -> None:
    result = _direct(trading={"trading.commitment_rule": "newsvendor"})
    # The record in the result says fixed share (the default), so name the
    # rule the run used for the validator.
    records = tuple(
        replace(record, value="newsvendor") if record.name == "trading.commitment_rule" else record
        for record in result.assumptions
    )
    validate_result_v2(replace(result, assumptions=records))
    level = result.deviation_world_slot["commit_level"]
    assert level.notna().any() and level.dropna().between(0.0, 1.0).all()
    fixed = _direct()
    assert fixed.deviation_world_slot["commit_level"].isna().all()
    # The rule changes positions, never the kernel.
    pd.testing.assert_frame_equal(fixed.fleet_world_intervals, result.fleet_world_intervals)
    with pytest.raises(ValueError, match="commitment_rule"):
        _direct(trading={"trading.commitment_rule": "best_guess"})


def test_a_blackout_run_passes_the_validator_with_zero_availability_in_the_blackout() -> None:
    windows = pd.DataFrame({"start_local_time": ["18:00"], "duration_minutes": [60]})
    result = run_forecast_from_assumptions(_START, values=_SMALL, blackout_windows=windows)
    validate_result_v2(result)
    world_slot = result.availability_world_slot
    blocked = world_slot.loc[world_slot["blackout"], "deliverable_kw"]
    assert len(blocked) > 0 and (blocked.fillna(0.0) == 0.0).all()


def _late_outage() -> pd.DataFrame:
    row = {
        "event_id": "late_outage",
        "event_type": "control_outage",
        "enabled": True,
        "night_index": 2,
        "start_local_time": "18:00",
        "duration_minutes": 180,
        "size": 1.0,
        "size_unit": "fraction",
        "notice": "short",
        "notice_minutes": 0,
        "scope": "national",
        "payment_gbp_per_mwh": np.nan,
    }
    return pd.DataFrame([row], columns=list(EVENT_COLUMNS))


_KNOWN_EXACT = ["intraday_known_mean_kw", "intraday_potential_kw", "intraday_eligible_count"]
_KNOWN_QUANTILES = [f"intraday_known_p{level}_kw" for level in ("05", "10", "50", "90", "95")]
_LATE = [f"late_restored_kw_m{maker}" for maker in range(1, 5)]


def test_a_control_outage_announced_after_17_00_leaves_the_known_part_unchanged() -> None:
    # R1 (§10.1e, §10.2d): the event changes plans made in its window, so the
    # realised deliverable moves, but the known part reads the plans in
    # force at 17:00 and does not.  Later nights start from batteries the
    # event touched, so the check is on the event's own night.  The known
    # part's quantiles are read on a per-slot grid whose range also spans
    # the late part (x_max, §10.2d), so they may move by up to two bins.
    base = _direct()
    event = _direct(events=_late_outage())
    a, b = (
        r.availability_world_slot.loc[lambda f: f["night_index"].eq(2)].reset_index(drop=True)
        for r in (base, event)
    )
    assert not np.allclose(a["deliverable_kw"].fillna(0), b["deliverable_kw"].fillna(0))
    pd.testing.assert_frame_equal(a[_KNOWN_EXACT], b[_KNOWN_EXACT])
    keys = ["slot_index", "direction", "duration_hours"]
    span = pd.concat(
        [f.assign(x=f["intraday_potential_kw"] + f[_LATE].sum(axis=1)) for f in (a, b)]
    )
    bin_kw = np.nan_to_num(span.groupby(keys)["x"].transform("max").to_numpy()[: len(a)] / 512.0)
    for column in _KNOWN_QUANTILES:
        gap = (a[column] - b[column]).abs().fillna(0.0).to_numpy()
        # Each run's quantile is within one bin of the exact one, so the two
        # runs' within two bins of each other.
        worst = np.max(gap - 2.0 * bin_kw)
        assert worst <= 1e-9, (column, float(np.max(gap / np.maximum(bin_kw, 1e-12))))
