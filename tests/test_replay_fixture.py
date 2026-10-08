"""Replay contract v1 §4 on the SYNTHETIC FIXTURE (docs/contracts/replay-v1.md).

The fixture's ``replay_week`` and one-EV timeline must pass the validators,
and the validators must reject a replay that leaks the future, books money
early or breaks a shape, rather than passing anything.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from fixtures.replay_contract import (
    BOOK_WIDTH,
    DECISION_COUNT,
    replay_one_ev_timeline,
    validate_one_ev_timeline_v2,
    validate_replay_week_v2,
)
from fixtures.result_fixture import make_result, replay_one_ev, validate_result_v2


@pytest.fixture(scope="module")
def action():
    return make_result()


def _with(result, **changes):
    return dataclasses.replace(
        result, replay_week=dataclasses.replace(result.replay_week, **changes)
    )


def test_action_fixture_carries_a_valid_replay_and_no_action_none(action) -> None:
    validate_result_v2(action)
    replay = action.replay_week
    assert replay.world_ids == action.sampled_world_ids
    assert replay.decisions["label"].iat[0] == "Mon 12:00"
    assert replay.decisions["label"].iat[-1] == "End of week"
    no_action = make_result("no_action")
    assert no_action.replay_week is None
    with pytest.raises(AssertionError, match="replay_week must be None"):
        validate_result_v2(dataclasses.replace(no_action, replay_week=replay))


def test_night_one_positions_are_hidden_until_its_day_ahead_decision(action) -> None:
    positions = action.replay_week.position_kwh
    night = action.study_slots["night_index"].to_numpy()
    # Night 0's day-ahead decision is before the study: in force at k = 0.
    assert np.isfinite(positions[:, 0, night == 0]).all()
    # Night 1's is 13:00 London on the first day, one hour after the start.
    assert np.isnan(positions[:, 0, night == 1]).all()
    assert np.isfinite(positions[:, 1, night == 1]).all()


def test_rejects_a_future_price_in_the_expected_shape(action) -> None:
    replay = action.replay_week
    visible = replay.day_ahead_visible_gbp_per_mwh.copy()
    k = 5
    hidden = np.flatnonzero(~replay.published[k])
    visible[:, k, hidden[0]] += 10.0
    with pytest.raises(AssertionError, match="expected shape"):
        validate_replay_week_v2(_with(action, day_ahead_visible_gbp_per_mwh=visible))


def test_rejects_an_intraday_value_before_publication(action) -> None:
    replay = action.replay_week
    known = replay.intraday_known_gbp_per_mwh.copy()
    known[:, 0, -1] = 50.0
    with pytest.raises(AssertionError, match="NaN exactly where unpublished"):
        validate_replay_week_v2(_with(action, intraday_known_gbp_per_mwh=known))


def test_rejects_a_position_before_its_decision(action) -> None:
    positions = action.replay_week.position_kwh.copy()
    night = action.study_slots["night_index"].to_numpy()
    positions[:, 0, night == 1] = 1.0
    with pytest.raises(AssertionError, match="forward fill"):
        validate_replay_week_v2(_with(action, position_kwh=positions))


def test_rejects_money_booked_before_its_slot_has_ended(action) -> None:
    cash = action.replay_week.trading_cash_to_date_gbp.copy()
    # Book night 0's customer share one boundary early: the night-end
    # value still reconciles, the slot-by-slot rule does not.
    ledger = action.trading_ledger_world
    first = ledger.loc[
        ledger["strategy"].eq("full")
        & ledger["world_id"].eq(action.sampled_world_ids[0])
        & ledger["night_index"].eq(0)
    ]
    end = int((action.study_slots["night_index"] == 0).sum())
    cash[0, end - 1] += float(first["customer_revenue_share_gbp"].iat[0])
    with pytest.raises(AssertionError, match="outside payment boundaries"):
        validate_replay_week_v2(_with(action, trading_cash_to_date_gbp=cash))


def test_rejects_a_saving_that_does_not_reconcile(action) -> None:
    saving = action.replay_week.energy_saving_to_date_gbp.copy()
    saving[:, -1] += 1.0
    with pytest.raises(AssertionError, match="cost effect"):
        validate_replay_week_v2(_with(action, energy_saving_to_date_gbp=saving))


def test_rejects_a_wrong_shape_and_unsampled_worlds(action) -> None:
    published = action.replay_week.published[:-1]
    with pytest.raises(AssertionError, match="published is"):
        validate_replay_week_v2(_with(action, published=published))
    with pytest.raises(AssertionError, match="sampled_world_ids"):
        validate_replay_week_v2(_with(action, world_ids=(99,)))


def test_rejects_positions_without_updates(action) -> None:
    result = dataclasses.replace(action, position_updates=None)
    with pytest.raises(AssertionError, match="position_kwh is None exactly"):
        validate_replay_week_v2(result)


def test_timeline_passes_and_rejects_a_plan_outside_its_window(action) -> None:
    unit_id = action.units["unit_id"].iat[1]
    world = action.sampled_world_ids[0]
    timeline = replay_one_ev_timeline(action, unit_id, world)
    validate_one_ev_timeline_v2(timeline, action, replay_one_ev(action, unit_id, world))
    assert timeline.plan_at_decision_kwh.sum() > 0.0
    assert (timeline.plan_status != 0).any(), "the fixture exercises a plan on record"
    assert timeline.plan_at_decision_kwh[-1].sum() == 0.0
    plan = timeline.plan_at_decision_kwh.copy()
    plan[10, 20 + BOOK_WIDTH] = 1.0
    with pytest.raises(AssertionError, match="zero outside"):
        validate_one_ev_timeline_v2(
            dataclasses.replace(timeline, plan_at_decision_kwh=plan), action
        )
    assert timeline.plan_at_decision_kwh.shape == (DECISION_COUNT, 336)


def test_timeline_needs_a_sampled_world() -> None:
    result = make_result(worlds=12)
    unsampled = next(w for w in range(result.world_count) if w not in result.sampled_world_ids)
    with pytest.raises(KeyError):
        replay_one_ev_timeline(result, result.units["unit_id"].iat[0], unsampled)


def test_replay_decisions_are_the_contract_frame(action) -> None:
    decisions = action.replay_week.decisions
    assert list(decisions["slot_index"]) == [min(2 * k, 336) for k in range(DECISION_COUNT)]
    steps = decisions["decision_utc"].diff().dropna()
    assert steps.eq(pd.Timedelta(hours=1)).all()
