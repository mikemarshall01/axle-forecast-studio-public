"""Tests for the one-EV kernel replay (``model/individual.py``, contract v2 section 5).

The key invariant (audit B3): the one-EV replays of every EV in a world add
up to that world's fleet frame, for BOTH models, so the One EV view shows the
same physics as the fleet, including smart charging and public top-ups.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fixtures.replay_contract import validate_one_ev_timeline_v2
from fixtures.result_fixture import validate_one_ev_replay_v2
from test_summaries import real_result

from axle_studio.model import assumptions, individual, market
from axle_studio.model.forecast import run_forecast
from axle_studio.model.individual import replay_one_ev, replay_one_ev_bands, replay_one_ev_timeline
from axle_studio.model.physics import CONTROL_GROUP, simulate_fleet_intervals
from axle_studio.model.summaries import ReplayState, kernel_slice

_START = date(2026, 10, 12)


def _control_result(share: float = 0.3):
    """A tiny real action run with a stratified hold-out control group (trading §10.1e).

    ``run_forecast`` directly (not ``run_forecast_from_assumptions``), like
    the household card's own control-group fixture, so this stays a small,
    fast, deterministic run rather than a full assumptions-driven one.
    """

    small = {"vehicle_count": 20, "evaluation_world_count": 2}
    inputs = assumptions.forecast_inputs(
        assumptions.resolve_values(small), warmup_days=7, study_days=7
    )
    inputs["trading_assumptions"] = inputs["trading_assumptions"] | {
        "trading.control_group": 1.0,
        "trading.control_group_share": share,
    }
    return run_forecast(
        assumptions.run_settings(small, _START), assumptions.cohort_fixture(small), **inputs
    )


_SUMMED = {
    "home_import_kwh": "home_import_kwh",
    "public_import_kwh": "public_import_kwh",
    "unserved_travel_kwh": "unserved_travel_kwh",
    "closing_battery_kwh": "closing_battery_kwh",
}


@pytest.fixture(scope="module")
def action_result():
    return real_result("action", vehicles=10, worlds=3)


@pytest.fixture(scope="module")
def no_action_result():
    return real_result("no_action", vehicles=10, worlds=3)


@pytest.mark.parametrize("name", ["action_result", "no_action_result"])
def test_replays_of_every_ev_sum_to_the_fleet_frame(name: str, request) -> None:
    result = request.getfixturevalue(name)
    paths = ["normal", "selected"] if result.model == "action" else ["normal"]
    world_id = 1
    replays = [replay_one_ev(result, unit_id, world_id) for unit_id in result.units["unit_id"]]
    fleet = result.fleet_world_intervals
    for path_id in paths:
        expected = fleet.loc[fleet["world_id"].eq(world_id) & fleet["path_id"].eq(path_id)]
        expected = expected.sort_values("slot_index")
        for replay_column, fleet_column in _SUMMED.items():
            total = sum(
                replay.intervals.loc[
                    replay.intervals["path_id"].eq(path_id), replay_column
                ].to_numpy()
                for replay in replays
            )
            assert np.allclose(total, expected[fleet_column].to_numpy(), rtol=0.0, atol=1e-9)
        connected = sum(
            replay.intervals.loc[replay.intervals["path_id"].eq(path_id), "connected"].to_numpy(int)
            for replay in replays
        )
        assert (connected == expected["connected_count"].to_numpy()).all()


@pytest.mark.parametrize("name", ["action_result", "no_action_result"])
def test_replay_passes_contract_checks_and_defaults_to_representative_world(
    name: str, request
) -> None:
    result = request.getfixturevalue(name)
    replay = replay_one_ev(result, result.units["unit_id"].iat[0])
    assert replay.world_id == result.representative_world_id
    assert replay.is_representative_world
    validate_one_ev_replay_v2(replay, result)
    assert replay.traits["unit_id"] == replay.unit_id


def test_unknown_ev_or_world_raises_key_error(no_action_result) -> None:
    with pytest.raises(KeyError):
        replay_one_ev(no_action_result, "ev-999")
    with pytest.raises(KeyError):
        replay_one_ev(no_action_result, "ev-001", world_id=3)
    with pytest.raises(KeyError):
        replay_one_ev_bands(no_action_result, "ev-999")


def test_replays_are_cached_per_result(no_action_result, action_result) -> None:
    first = replay_one_ev(no_action_result, "ev-002", 0)
    assert replay_one_ev(no_action_result, "ev-002", 0) is first
    assert ("replay", "ev-002", 0) in no_action_result.replay_state.replay_cache
    # Each result keeps its own cache, so runs never share replays.
    assert first not in action_result.replay_state.replay_cache.values()


def test_selected_path_shows_early_departures_that_sum_to_the_fleet(action_result) -> None:
    # The replay's early-departure shortfall is the same kernel output as the
    # fleet's audit column, EV by EV (decision 0004 item 38).
    world_id = 1
    fleet = action_result.fleet_world_intervals
    expected = fleet.loc[fleet["world_id"].eq(world_id) & fleet["path_id"].eq("selected")]
    total = sum(
        replay_one_ev(action_result, unit_id, world_id)
        .intervals.query("path_id == 'selected'")["early_departure_shortfall_kwh"]
        .to_numpy()
        for unit_id in action_result.units["unit_id"]
    )
    np.testing.assert_allclose(
        total, expected.sort_values("slot_index")["early_departure_shortfall_kwh"], atol=1e-9
    )


def test_daily_audit_reads_the_replay(action_result) -> None:
    replay = replay_one_ev(action_result, "ev-003", 2)
    intervals = replay.intervals
    for row in replay.daily_audit.itertuples():
        if not row.drives_today:
            assert pd.isna(row.departure_utc) and row.trip_energy_need_kwh == 0.0
            continue
        assert row.trip_energy_need_kwh > 0.0
        assert 0.0 <= row.shortfall_kwh <= row.trip_energy_need_kwh + 1e-9
        # A row is a session night (decision 0004 item 52); its departure is
        # the next morning's, which the noon-to-noon study always contains.
        assert row.departure_london.date() == row.local_date + timedelta(days=1)
        assert row.departure_london.hour < 12
        path = intervals.loc[intervals["path_id"].eq(row.path_id)].reset_index(drop=True)
        slot = int(
            (row.departure_utc - path["interval_start_utc"].iat[0]) // pd.Timedelta(minutes=30)
        )
        if slot == 0:
            continue  # opening stock of the horizon is not in the frame
        # SoC at departure is the stock at the start of the departure slot.
        assert row.departure_soc_percent == pytest.approx(path["battery_soc_percent"].iat[slot - 1])
        assert row.departed_below_target == (
            row.departure_soc_percent < row.target_soc_percent - 1e-6
        )


def test_replay_plug_events_match_the_fleet_sample(no_action_result) -> None:
    result = no_action_result
    unit_id, world_id = "ev-004", 1
    replay = replay_one_ev(result, unit_id, world_id)
    fleet_events = result.plug_in_events
    expected = fleet_events.loc[
        fleet_events["unit_id"].eq(unit_id) & fleet_events["world_id"].eq(world_id)
    ].reset_index(drop=True)
    got = replay.plug_events.reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected)


@pytest.mark.parametrize("start", [date(2026, 10, 21), date(2027, 3, 22)], ids=["autumn", "spring"])
def test_clock_change_week_replays_both_paths(start: date) -> None:
    result = real_result("action", start=start, vehicles=6, worlds=2)
    for unit_id in result.units["unit_id"]:
        replay = replay_one_ev(result, unit_id, 1)
        validate_one_ev_replay_v2(replay, result)
        assert len(replay.intervals) == 2 * 336
        assert len(replay.daily_audit) == 2 * 7


def test_replay_bands_are_quantiles_of_this_ev_across_worlds(action_result) -> None:
    unit_id = "ev-005"
    bands = replay_one_ev_bands(action_result, unit_id)
    assert list(dict.fromkeys(bands["metric"])) == ["battery_soc_percent", "connected_share"]
    assert (bands["p10"] <= bands["p50"] + 1e-12).all()
    assert (bands["p50"] <= bands["p90"] + 1e-12).all()
    replays = [replay_one_ev(action_result, unit_id, w) for w in range(action_result.world_count)]
    soc = np.stack(
        [
            r.intervals.loc[r.intervals["path_id"].eq("selected"), "battery_soc_percent"].to_numpy()
            for r in replays
        ]
    )
    selected = bands.loc[
        bands["metric"].eq("battery_soc_percent") & bands["path_id"].eq("selected")
    ]
    assert np.allclose(selected["p50"].to_numpy(), np.median(soc, axis=0), atol=1e-12)
    assert replay_one_ev_bands(action_result, unit_id) is bands


def test_replay_uses_the_run_top_up_threshold_and_target(action_result) -> None:
    # Regression: kernel_slice once rebuilt the public inputs field by field
    # and dropped the top-up threshold and target (decision 0004 item 32), so
    # replays silently used the 20 % / 80 % defaults.  A run stored with other
    # fractions must replay with them, and still sum to that run's fleet.
    state = action_result.replay_state
    public = replace(state.public_top_up, threshold_soc_fraction=0.6, target_soc_fraction=0.95)
    changed_state = ReplayState(
        settings=state.settings,
        units=state.units,
        evaluation=state.evaluation,
        public_top_up=public,
        smart_charging=state.smart_charging,
    )
    changed = replace(action_result, replay_state=changed_state)
    world_id = 1

    piece = kernel_slice(changed_state, [world_id], [0], path_id="selected")
    assert piece.public_top_up.threshold_soc_fraction == 0.6
    assert piece.public_top_up.target_soc_fraction == 0.95

    def public_import(result) -> np.ndarray:
        return np.stack(
            [
                replay_one_ev(result, unit_id, world_id).intervals["public_import_kwh"].to_numpy()
                for unit_id in result.units["unit_id"]
            ]
        )

    assert not np.allclose(public_import(changed), public_import(action_result))
    fleet, _, _, _ = simulate_fleet_intervals(
        replace(state.settings, evaluation_world_count=1),
        state.units,
        {name: values[[world_id]] for name, values in state.evaluation.inputs.items()},
        evaluation_effective_efficiency_miles_per_battery_kwh=(
            state.evaluation.effective_efficiency_miles_per_battery_kwh[[world_id]]
        ),
        public_top_up=public,
        smart_charging=state.smart_charging.for_slice(
            [world_id], np.arange(state.settings.vehicle_count)
        ),
    )
    for path_index, path_id in enumerate(("normal", "selected")):
        expected = fleet.loc[fleet["path_id"].eq(path_id)].sort_values("interval_start_utc")
        rows = public_import(changed)[:, path_index * 336 : (path_index + 1) * 336]
        assert np.allclose(
            rows.sum(axis=0), expected["public_grid_import_kwh"].to_numpy(), rtol=0.0, atol=1e-9
        )


# --------------------------------------------------------------------------
# replay_one_ev_timeline (replay contract v1 §2, lane R1c)
# --------------------------------------------------------------------------


def test_book_decision_slots_matches_the_forecast_runs_hourly_grid(action_result) -> None:
    # Regression: ``market.book_decision_slots`` must reproduce the inline
    # rule ``forecast._trading_kernel_outputs`` used to build the fleet
    # run's own book, or the fleet and one-EV plan books drift apart.
    settings = action_result.replay_state.settings
    slots = action_result.study_slots
    book_slots, book_decision = market.book_decision_slots(settings, slots)

    minute = pd.DatetimeIndex(slots["interval_start_london"]).minute
    expected_slots = np.flatnonzero(np.asarray(minute) == 0)
    expected_decision = np.full(settings.warmup_days * 48 + len(slots), -1, dtype=np.int64)
    expected_decision[settings.warmup_days * 48 + expected_slots] = np.arange(len(expected_slots))

    np.testing.assert_array_equal(book_slots, expected_slots)
    np.testing.assert_array_equal(book_decision, expected_decision)
    # A whole-hour UTC offset (contract §0) puts a decision at every other
    # study slot: 168 hourly rows for a 336-slot study without a DST anomaly.
    assert len(book_slots) == len(slots) // 2


def test_timeline_passes_the_contract_validator(action_result) -> None:
    result = action_result
    world = result.sampled_world_ids[0]
    for unit_id in result.units["unit_id"].iloc[:3]:
        timeline = replay_one_ev_timeline(result, unit_id, world)
        replay = replay_one_ev(result, unit_id, world)
        validate_one_ev_timeline_v2(timeline, result, replay)
        assert timeline.world_id == world
        # ``units.dispatch_locked`` is on every result (dispatch §2, §7.3).
        assert timeline.dispatch_locked is bool(
            result.units.set_index("unit_id").at[unit_id, "dispatch_locked"]
        )


def test_timeline_needs_a_sampled_world() -> None:
    # More than 10 worlds so at least one is not in ``sampled_world_ids``
    # (results-v2 §2: at most 10 sampled worlds).
    result = real_result("action", vehicles=6, worlds=12, seed=5)
    unsampled = next(w for w in range(result.world_count) if w not in result.sampled_world_ids)
    with pytest.raises(KeyError):
        replay_one_ev_timeline(result, result.units["unit_id"].iat[0], unsampled)


def test_timeline_shares_the_one_ev_kernel_run_with_replay_one_ev(monkeypatch) -> None:
    # §2: "one run per (EV, world) fills both objects".  ``replay_one_ev``'s
    # own kernel pass (``individual._replay``) must run once whichever of
    # the two callers asks first; the timeline's extra cost is its own
    # book and unit-output passes, not a second copy of that one.
    result = real_result("action", vehicles=8, worlds=2, seed=99)
    world, unit_id = result.sampled_world_ids[0], result.units["unit_id"].iat[0]
    calls = []
    original = individual._replay

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(individual, "_replay", counted)
    replay_one_ev_timeline(result, unit_id, world)
    replay_one_ev(result, unit_id, world)
    assert len(calls) == 1
    assert ("timeline", unit_id, world) in result.replay_state.replay_cache
    assert ("replay", unit_id, world) in result.replay_state.replay_cache


def test_saving_to_date_starts_at_zero_and_reconciles_with_the_fleet_running_saving(
    action_result,
) -> None:
    # §5 "Customer money": summed over every EV of a sampled world, this
    # EV's own running saving must equal the fleet replay's running saving
    # (both value the customer leg at the day-ahead price, §1.4).
    result = action_result
    world = result.sampled_world_ids[0]
    wi = list(result.sampled_world_ids).index(world)
    total = np.zeros(337)
    for unit_id in result.units["unit_id"]:
        timeline = replay_one_ev_timeline(result, unit_id, world)
        assert timeline.saving_to_date_gbp[0] == 0.0
        total += timeline.saving_to_date_gbp
    fleet_saving = result.replay_week.energy_saving_to_date_gbp[wi]
    np.testing.assert_allclose(total, fleet_saving, rtol=0.0, atol=1e-6)


def test_plug_in_exactly_at_a_decision_shows_no_plan_until_the_next_frame() -> None:
    # B1 (review of contract-replay 3c923b1): a plan made at a plug-in
    # exactly at tau_k must not show before frame k + 1, because the book
    # is read before that slot's new plans.  Fixed seed/size chosen because
    # it naturally produces a plug-in slot that lands on an hourly decision.
    result = real_result("action", vehicles=20, worlds=2, seed=11)
    world = result.sampled_world_ids[0]
    found = False
    for unit_id in result.units["unit_id"]:
        replay = replay_one_ev(result, unit_id, world)
        selected = replay.intervals.loc[replay.intervals["path_id"].eq("selected")].sort_values(
            "slot_index"
        )
        connected = selected["connected"].to_numpy()
        home = selected["home_import_kwh"].to_numpy()
        plug_ins = np.flatnonzero(connected[1:] & ~connected[:-1]) + 1
        even_plug_ins = [t for t in plug_ins if t % 2 == 0 and home[t] > 0]
        if not even_plug_ins:
            continue
        t = even_plug_ins[0]
        k = t // 2
        timeline = replay_one_ev_timeline(result, unit_id, world)
        width = min(market.BOOK_WIDTH, len(home) - t)
        # Frame k (at the plug-in instant) shows nothing for this EV yet.
        assert (timeline.plan_at_decision_kwh[k, t : t + width] == 0.0).all()
        # Frame k + 1 (the next hourly decision, slot t + 2) shows the plan
        # from its own book slot forward; it is not read at slot t, which is
        # already in the past from frame k + 1's own instant.
        next_width = min(market.BOOK_WIDTH, len(home) - (t + 2))
        np.testing.assert_allclose(
            timeline.plan_at_decision_kwh[k + 1, t + 2 : t + 2 + next_width],
            home[t + 2 : t + 2 + next_width],
            atol=1e-9,
        )
        assert timeline.plan_at_decision_kwh[k + 1, t + 2] > 0.0
        found = True
        break
    assert found, "fixture no longer has a plug-in landing on an hourly decision slot"


def test_control_group_ev_keeps_its_smart_plan_on_record_when_it_ignores_it() -> None:
    # O11: the selected book holds a non-responder's plan, not its actual
    # (full-power) import, and its plan_status names why (trading §10.1e).
    result = _control_result()
    world = result.sampled_world_ids[0]
    control_ids = result.units.loc[result.units["control_group"], "unit_id"]
    assert len(control_ids) > 0
    divergence_found = False
    for unit_id in control_ids:
        timeline = replay_one_ev_timeline(result, unit_id, world)
        replay = replay_one_ev(result, unit_id, world)
        control_slots = np.flatnonzero(timeline.plan_status == CONTROL_GROUP)
        if len(control_slots) == 0:
            continue
        selected = replay.intervals.loc[replay.intervals["path_id"].eq("selected")].sort_values(
            "slot_index"
        )
        home = selected["home_import_kwh"].to_numpy()
        for t in control_slots:
            k = t // 2
            if t != 2 * k:
                continue  # only the column each book row actually carries
            plan_value = timeline.plan_at_decision_kwh[k, t]
            if plan_value > 0.0 and not np.isclose(plan_value, home[t]):
                divergence_found = True
                break
        if divergence_found:
            break
    assert divergence_found, "expected a control EV whose plan on record differs from its import"
