"""The household card (household contract v1 §3.3, §9 HH2 rows) on SYNTHETIC fixtures.

Pure pieces (ranks, clock quantiles, timing, the availability fold) are
checked on hand arrays; the assembled card on the synthetic ``make_result``
fixture and on small seeded real runs.  Every figure is illustrative.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from functools import cache

import numpy as np
import pandas as pd
import pytest
from fixtures.household_contract import make_household_card, validate_household_card_v2
from fixtures.result_fixture import make_result

from axle_studio.model import assumptions, household, summaries
from axle_studio.model.forecast import run_forecast, run_forecast_from_assumptions

_START = date(2026, 10, 12)


def _ev_world(values: dict[str, list[list[float]]], cohorts=("c", "c", "c"), treated=None):
    """A ``household_ev_world``-shaped frame: ``values[column][world][ev]``; others 0."""

    world_count = len(next(iter(values.values())))
    vehicle_count = len(cohorts)
    treated = [True] * vehicle_count if treated is None else treated
    rows = []
    for w in range(world_count):
        for e in range(vehicle_count):
            row = {
                name: (0 if dtype in (np.int64, np.float64) else False)
                for name, dtype in household.EV_WORLD_COLUMNS.items()
            }
            row.update(
                world_id=w,
                unit_id=f"ev-{e}",
                cohort_id=cohorts[e],
                treated=treated[e],
                evidence_kind="illustrative_synthetic",
            )
            row.update({name: grid[w][e] for name, grid in values.items()})
            rows.append(row)
    return pd.DataFrame(rows).astype(household.EV_WORLD_COLUMNS)


def _outcomes(frame, unit_id, groups=None):
    vehicle_count = frame["unit_id"].nunique()
    treated = frame["treated"].to_numpy()[:vehicle_count]
    summary = household.outcomes_summary(
        frame, groups=groups or {"fleet": treated, "c": treated}, night_count=7
    )
    return household.card_outcomes(frame, summary, unit_id, 7).set_index("metric")


# --------------------------------------------------------------------------
# Outcomes and ranks (§3.3a)
# --------------------------------------------------------------------------


def test_card_quantiles_and_ranks_on_three_evs():
    # Savings per world: week 0 [1, 2, 3], week 1 [3, 2, 0] GBP.
    frame = _ev_world({"saving_gbp": [[1.0, 2.0, 3.0], [3.0, 2.0, 0.0]]})
    ev0 = _outcomes(frame, "ev-0")
    weekly = ev0.loc["saving_gbp_per_week"]
    assert weekly[["p10", "p50", "p90"]].tolist() == pytest.approx(
        np.quantile([1.0, 3.0], (0.1, 0.5, 0.9)).tolist()
    )
    # Per week: 1 of 3 at or below 1, then 3 of 3 at or below 3; median 2/3.
    assert weekly["fleet_rank_p50"] == pytest.approx(2 / 3)
    ev2 = _outcomes(frame, "ev-2").loc["saving_gbp_per_week"]
    assert ev2["cohort_rank_p50"] == pytest.approx(np.median([1.0, 1 / 3]))
    # Scenario rows: the EV mean x 52, no weekly quantiles; EV means are
    # (2, 2, 1.5), so EV 0 is at the top and EV 2 at the bottom.
    year = ev0.loc["saving_gbp_per_year"]
    assert year["mean"] == pytest.approx(104.0)
    assert np.isnan(year[["p10", "p50", "p90"]].astype(float)).all()
    assert year["fleet_rank_p50"] == 1.0
    assert _outcomes(frame, "ev-2").loc["saving_gbp_per_year", "fleet_rank_p50"] == pytest.approx(
        1 / 3
    )
    # Group columns: set A at its median across weeks, set B of EV means.
    assert weekly["fleet_p50"] == pytest.approx(2.0)  # medians of weeks: 2 and 2
    assert year["fleet_p50"] == pytest.approx(104.0)


def test_control_ev_card_has_no_rank_and_is_outside_the_groups():
    frame = _ev_world(
        {"saving_gbp": [[1.0, 2.0, 30.0], [3.0, 2.0, 30.0]]}, treated=[True, True, False]
    )
    treated = np.array([True, True, False])
    control = _outcomes(frame, "ev-2", {"fleet": treated, "c": treated})
    assert control["fleet_rank_p50"].isna().all() and control["cohort_rank_p50"].isna().all()
    # The control EV's 30 GBP is in no group statistic.
    assert control.loc["saving_gbp_per_week", "fleet_p90"] < 30.0


# --------------------------------------------------------------------------
# Timing and reliability (§3.3c-d)
# --------------------------------------------------------------------------


def _session_values(plug_in, departure, **others):
    n = len(plug_in)
    return {
        "plug_in_time": np.asarray(plug_in, dtype=float),
        "departure_time": np.asarray(departure, dtype=float),
        "plug_in_soc_percent": np.full(n, 50.0),
        "energy_needed_kwh": others.get("need", np.full(n, 10.0)),
        "dwell_hours": others.get("dwell", np.full(n, 8.0)),
        "flexible_kwh": others.get("flexible", np.full(n, 10.0)),
        "slack_hours": np.full(n, 2.0),
    }


def _ev_rows(world_count: int) -> pd.DataFrame:
    return _ev_world({"session_count": [[3]] * world_count}, cohorts=("c",))


def test_clock_quantiles_are_taken_on_hours_since_noon():
    values = _session_values([23.0, 23.5, 0.5], [6.5, 7.0, 7.5])
    rows = household.reliability(
        values, world=np.zeros(3, dtype=int), world_count=1, ev_rows=_ev_rows(1), night_count=7
    ).set_index("metric")
    assert rows.loc["plug_in_time", "p50"] == pytest.approx(23.5)  # a plain quantile: 23.0
    assert rows.loc["plug_in_time", "p50_label"] == "23:30"
    assert rows.loc["departure_time", "p50"] == pytest.approx(7.0)
    assert rows.loc["departure_time", "p50_label"] == "07:00"
    assert rows.loc["plug_in_time", "sample_count"] == 3
    assert rows.loc["flexible_kwh_per_night", "mean"] == pytest.approx(30.0 / 7)
    timing = household.session_timing(values)
    order = timing.drop_duplicates("bin_index").set_index("bin_index")["profile_order"]
    # Overnight bins are contiguous in the noon-to-noon order (one hump).
    assert order[47] + 1 == order[0] and order[0] + 1 == order[1]
    assert order[24] == 0


def test_pooled_timing_counts_open_sessions_for_plug_in_only():
    values = _session_values([18.0, 18.0, 18.0], [7.0, 7.0, np.nan])
    timing = household.session_timing(values).set_index(["metric", "bin_index"])
    assert timing.loc[("plug_in_time", 36), "share"] == 1.0
    assert timing.loc[("plug_in_time", 36), "session_count"] == 3
    assert timing.loc[("departure_time", 14), "session_count"] == 2
    empty = household.session_timing(_session_values([], []))
    assert empty["share"].isna().all() and (empty["session_count"] == 0).all()


# --------------------------------------------------------------------------
# Availability profile (§3.3b), on a clock-change week
# --------------------------------------------------------------------------


def test_availability_profile_and_its_typical_day_on_the_autumn_change():
    result = make_result(evs=6, worlds=5, clock_change="autumn")
    slots = result.study_slots
    rng = np.random.default_rng(8)
    realised = {
        d: np.where(rng.random((5, len(slots))) < 0.3, 0.0, rng.uniform(0, 7, (5, len(slots))))
        for d in household.DIRECTIONS
    }
    for values in realised.values():
        values[:, -1] = np.nan  # the last 1-h window runs past the study end
    by_slot, by_day = household.availability_profile(realised, slots)
    down = realised["turn_down"]
    first = by_slot.iloc[0]
    assert first["available_share"] == pytest.approx((down[:, 0] > 0).mean())
    assert first["p05"] == pytest.approx(np.quantile(down[:, 0], 0.05))
    last = by_slot.loc[by_slot["direction"].eq("turn_down")].iloc[-1]
    assert last["world_count"] == 0 and np.isnan(last["p50"]) and np.isnan(last["available_share"])
    # The repeated 01:00-02:00 hour gives two slots at 01:00 on the change night.
    half_hour = slots["local_half_hour"].to_numpy()
    all_days = slots["day_type"].to_numpy()
    for kind in ("weekday", "weekend", "all"):
        mask = (half_hour == 2) & ((all_days == kind) if kind != "all" else True)
        per_world = np.nanmean(down[:, mask], axis=1)
        row = by_day.loc[
            by_day["direction"].eq("turn_down")
            & by_day["day_type"].eq(kind)
            & by_day["local_half_hour"].eq(2)
        ].iloc[0]
        assert row["mean"] == pytest.approx(per_world.mean())
        assert row["available_share"] == pytest.approx((per_world > 0).mean())
    assert ((half_hour == 2).sum()) == 8  # seven nights plus the repeated hour
    # A card built with this profile passes the card validator (dtypes too).
    card = make_household_card(result, "ev-0000")
    card = replace(card, availability=by_slot, availability_day=by_day)
    validate_household_card_v2(card, result)


# --------------------------------------------------------------------------
# Synthetic fixture and real runs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("clock_change", [None, "spring"])
def test_fixture_cards_pass_the_validator(clock_change):
    result = make_result(evs=6, worlds=4, clock_change=clock_change)
    for unit_id in result.units["unit_id"]:
        card = make_household_card(result, unit_id)
        # The fixture has no dispatch, so every EV is unlocked (dispatch §2).
        assert card.availability is None and card.dispatch_locked is False
        validate_household_card_v2(card, result)


@cache
def _run(vehicle_count: int = 30, world_count: int = 4):
    values = {"vehicle_count": vehicle_count, "evaluation_world_count": world_count}
    return run_forecast_from_assumptions(_START, values=values)


def test_real_card_is_cached_and_runs_the_kernel_once(monkeypatch):
    result = run_forecast_from_assumptions(
        _START, values={"vehicle_count": 12, "evaluation_world_count": 4}
    )
    calls = []
    simulate = summaries._simulate

    def counted(piece):
        calls.append(piece.settings.vehicle_count)
        return simulate(piece)

    monkeypatch.setattr(summaries, "_simulate", counted)
    unit_id = result.units["unit_id"].iat[5]
    card = household.household_card(result, unit_id)
    assert household.household_card(result, unit_id) is card
    # One one-EV normal-path run and, with J1b's plan outputs, one selected
    # run for the realised-kW profile; the cached card runs nothing more.
    assert calls == [1, 1]
    validate_household_card_v2(card, result)
    assert card.availability is not None and card.availability_day is not None
    with pytest.raises(KeyError):
        household.household_card(result, "no-such-ev")
    no_action = run_forecast_from_assumptions(
        _START, model="no_action", values={"vehicle_count": 6, "evaluation_world_count": 2}
    )
    with pytest.raises(ValueError, match="household frames are not on this result"):
        household.household_card(no_action, no_action.units["unit_id"].iat[0])


def test_one_ev_slice_reproduces_the_chunk_arrays():
    # EVs are physically independent, so the one-EV kernel run gives exactly
    # this EV's part of the fleet run (the load-bearing slice identity).
    result = _run(6, 4)
    state = result.replay_state
    worlds, by_path = next(
        summaries._per_ev_chunks(
            state, result.study_slots, result.fleet_world_intervals, ["normal"]
        )
    )
    assert list(worlds) == [0, 1, 2, 3]
    normal = by_path["normal"]
    events = result.plug_in_events
    for ev in range(6):
        arrays = household.one_ev_arrays(state, ev, result.study_slots)
        assert np.array_equal(arrays["connected"], normal["connected"][..., ev])
        assert np.array_equal(arrays["closing_kwh"], normal["closing_battery_kwh"][..., ev])
        assert np.array_equal(arrays["home_import_kwh"], normal["home_grid_import_kwh"][..., ev])
        sessions = summaries.find_sessions(
            arrays["connected"][..., None],
            arrays["closing_kwh"][..., None],
            arrays["home_import_kwh"][..., None],
        )
        unit_id = state.units["unit_id"].iat[ev]
        for world in result.sampled_world_ids:
            logged = events.loc[events["world_id"].eq(world) & events["unit_id"].eq(unit_id)]
            assert (sessions.world == world).sum() == len(logged)


def _direct(trading: dict[str, float]):
    small = {"vehicle_count": 30, "evaluation_world_count": 4}
    inputs = assumptions.forecast_inputs(
        assumptions.resolve_values(small), warmup_days=7, study_days=7
    )
    inputs["trading_assumptions"] = inputs["trading_assumptions"] | trading
    return run_forecast(
        assumptions.run_settings(small, _START), assumptions.cohort_fixture(small), **inputs
    )


def test_real_control_ev_card_is_flagged_and_unranked():
    result = _direct({"trading.control_group": 1.0, "trading.control_group_share": 0.2})
    control = result.units.loc[result.units["control_group"], "unit_id"].iat[0]
    card = household.household_card(result, control)
    assert card.treated is False
    assert card.outcomes[["cohort_rank_p50", "fleet_rank_p50"]].isna().all().all()


def test_two_runs_with_one_seed_give_identical_cards():
    first, second = (
        _run(),
        run_forecast_from_assumptions(
            _START, values={"vehicle_count": 30, "evaluation_world_count": 4}
        ),
    )
    unit_id = first.units["unit_id"].iat[7]
    a, b = household.household_card(first, unit_id), household.household_card(second, unit_id)
    for name in ("outcomes", "session_timing", "reliability"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))


def test_card_validator_rejects_a_wrong_rank():
    result = make_result(evs=6, worlds=4)
    card = make_household_card(result, "ev-0001")
    outcomes = card.outcomes.copy()
    outcomes.loc[0, "fleet_rank_p50"] = 0.01
    with pytest.raises(AssertionError, match="fleet rank"):
        validate_household_card_v2(replace(card, outcomes=outcomes), result)
