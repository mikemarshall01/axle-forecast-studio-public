"""Tests for the intraday dispatch frames in ``model/summaries.py`` (contract §7, lane K3).

K1's kernel (``action.py``'s ``DispatchSums``) is being built in parallel on
another branch, so every test here develops against ``FixtureDispatchSums``,
a SYNTHETIC hand fixture mirroring §5.2's fields exactly
(``tests/fixtures/dispatch_contract.py``), never K1's production code.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fixtures.dispatch_contract import (
    DISPATCH_WORLD_SLOT_COLUMNS,
    FixtureDispatchSums,
    dispatch_bands_from,
    locked_ev_counts,
    locked_ev_mask,
    validate_dispatch_frames,
)

from axle_studio.model import events as model_events
from axle_studio.model.summaries import (
    build_study_slots,
    dispatch_bands,
    dispatch_split,
    dispatch_world_slot,
    event_response_bands,
)

# --------------------------------------------------------------------------
# dispatch_world_slot (contract §7.1)
# --------------------------------------------------------------------------


def _toy_evaluation_prices(close: list[list[float]]) -> pd.DataFrame:
    rows = []
    for world_id, values in enumerate(close):
        for slot_index, value in enumerate(values):
            rows.append(
                {
                    "world_id": np.int64(world_id),
                    "slot_index": np.int64(slot_index),
                    "evaluation_context_price_gbp_per_mwh": float(value),
                }
            )
    return pd.DataFrame(rows)


def _toy_selected_fleet(dispatched: list[list[float]]) -> pd.DataFrame:
    """A minimal ``fleet_world_intervals``-shaped frame: selected path only."""

    rows = []
    for world_id, values in enumerate(dispatched):
        for slot_index, value in enumerate(values):
            rows.append(
                {
                    "world_id": np.int64(world_id),
                    "path_id": "selected",
                    "slot_index": np.int64(slot_index),
                    "home_import_kwh": float(value),
                }
            )
    return pd.DataFrame(rows)


def test_dispatch_world_slot_identities_by_hand() -> None:
    # Two worlds, four slots.  World 0: day-ahead plan [5, 3, 0, 0], the
    # dispatch turns down slots 0-1 by 1 kWh each (moved out to later in the
    # window, not shown here) so dispatched is [4, 2, 0, 0].  World 1: a
    # flat day-ahead plan [2, 2, 2, 2] turned down by 1 kWh throughout.
    study_slots = build_study_slots(date(2026, 1, 12)).iloc[:4].reset_index(drop=True)
    dispatch_sums = FixtureDispatchSums(
        day_ahead_home_import_kwh=np.array([[5.0, 3.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]),
        day_ahead_zone_home_import_kwh=np.zeros((2, 4, 4)),
        locked_home_import_kwh=np.array([[3.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
        free_home_import_kwh=np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        free_day_ahead_home_import_kwh=np.array([[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
        replan_count=np.array([[0, 2, 0, 0], [1, 0, 0, 0]], dtype=np.int64),
    )
    fleet = _toy_selected_fleet([[4.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    evaluation = _toy_evaluation_prices([[50.0, 60.0, 70.0, 80.0], [10.0, 20.0, 30.0, 40.0]])

    frame = dispatch_world_slot(dispatch_sums, fleet, study_slots, evaluation)

    assert list(frame.columns) == list(DISPATCH_WORLD_SLOT_COLUMNS)
    assert list(frame["world_id"]) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert list(frame["slot_index"]) == [0, 1, 2, 3, 0, 1, 2, 3]
    assert list(frame["night_index"]) == [0, 0, 0, 0, 0, 0, 0, 0]
    assert frame["day_ahead_plan_kwh"].tolist() == pytest.approx(
        [5.0, 3.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
    )
    assert frame["dispatched_kwh"].tolist() == pytest.approx(
        [4.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    )
    # locked + free = dispatched; locked + free_day_ahead_plan = day_ahead_plan.
    assert frame["locked_kwh"].tolist() == pytest.approx([3.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert frame["free_kwh"].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert frame["free_day_ahead_plan_kwh"].tolist() == pytest.approx(
        [2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    )
    # moved_kwh: negative is turned down against the day-ahead plan (§7.1 caption).
    assert frame["moved_kwh"].tolist() == pytest.approx(
        [-1.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0]
    )
    assert frame["replan_count"].tolist() == [0, 2, 0, 0, 1, 0, 0, 0]
    assert frame["replan_count"].dtype == np.int64
    assert frame["latest_close_gbp_per_mwh"].tolist() == pytest.approx(
        [50.0, 60.0, 70.0, 80.0, 10.0, 20.0, 30.0, 40.0]
    )
    assert (frame["evidence_kind"] == "illustrative_synthetic").all()


# --------------------------------------------------------------------------
# dispatch_bands (contract §7.2)
# --------------------------------------------------------------------------


def test_dispatch_bands_are_world_first_not_band_subtraction() -> None:
    # Three worlds, one slot: day-ahead plan 1, 2, 3 kWh; dispatched 3, 1, 2
    # kWh (the same numbers as the fleet's own
    # test_difference_bands_are_per_world_differences_not_band_subtraction,
    # so the "world-first" point is directly comparable).  kW = kWh / 0.5.
    study_slots = build_study_slots(date(2026, 1, 12)).iloc[:1].reset_index(drop=True)
    dispatch_sums = FixtureDispatchSums(
        day_ahead_home_import_kwh=np.array([[1.0], [2.0], [3.0]]),
        day_ahead_zone_home_import_kwh=np.zeros((3, 4, 1)),
        locked_home_import_kwh=np.zeros((3, 1)),
        free_home_import_kwh=np.array([[3.0], [1.0], [2.0]]),
        free_day_ahead_home_import_kwh=np.array([[1.0], [2.0], [3.0]]),
        replan_count=np.array([[1], [0], [2]], dtype=np.int64),
    )
    fleet = _toy_selected_fleet([[3.0], [1.0], [2.0]])
    evaluation = _toy_evaluation_prices([[0.0], [0.0], [0.0]])
    world_slot = dispatch_world_slot(dispatch_sums, fleet, study_slots, evaluation)

    bands = dispatch_bands(world_slot, study_slots)

    difference = bands.loc[bands["series"].eq("difference")].iloc[0]
    # Per-world differences (kW): +4, -2, -2; median -2, mean 0.
    assert difference["p50"] == pytest.approx(-2.0)
    assert difference["mean"] == pytest.approx(0.0)
    # Subtracting the bands (median dispatched - median day-ahead, in kW)
    # would give 2*2 - 2*2 = 0: a different, wrong, answer for the median.
    band_subtraction = np.median([3.0, 1.0, 2.0]) * 2 - np.median([1.0, 2.0, 3.0]) * 2
    assert difference["p50"] != pytest.approx(band_subtraction)

    is_replan = bands["series"].eq("dispatched") & bands["metric"].eq("replan_count")
    replan = bands.loc[is_replan].iloc[0]
    assert replan["unit"] == "EVs"
    assert replan["mean"] == pytest.approx(1.0)
    assert replan["p50"] == pytest.approx(1.0)

    # Independent recomputation (dispatch_contract.py) agrees exactly.
    expected = dispatch_bands_from(world_slot)
    for (series, metric), stats in expected.items():
        row = bands.loc[bands["series"].eq(series) & bands["metric"].eq(metric)].iloc[0]
        for stat in ("mean", "p10", "p50", "p90"):
            assert row[stat] == pytest.approx(stats[stat][0])


def test_dispatch_bands_row_order_and_columns() -> None:
    study_slots = build_study_slots(date(2026, 1, 12)).iloc[:2].reset_index(drop=True)
    dispatch_sums = FixtureDispatchSums(
        day_ahead_home_import_kwh=np.zeros((1, 2)),
        day_ahead_zone_home_import_kwh=np.zeros((1, 4, 2)),
        locked_home_import_kwh=np.zeros((1, 2)),
        free_home_import_kwh=np.zeros((1, 2)),
        free_day_ahead_home_import_kwh=np.zeros((1, 2)),
        replan_count=np.zeros((1, 2), dtype=np.int64),
    )
    fleet = _toy_selected_fleet([[0.0, 0.0]])
    evaluation = _toy_evaluation_prices([[0.0, 0.0]])
    world_slot = dispatch_world_slot(dispatch_sums, fleet, study_slots, evaluation)

    bands = dispatch_bands(world_slot, study_slots)

    from fixtures.dispatch_contract import DISPATCH_BAND_COLUMNS

    assert list(bands.columns) == list(DISPATCH_BAND_COLUMNS)
    assert set(zip(bands["series"], bands["metric"], strict=True)) == {
        ("day_ahead_plan", "home_import_kw"),
        ("dispatched", "home_import_kw"),
        ("difference", "home_import_kw"),
        ("dispatched", "replan_count"),
    }
    assert len(bands) == 4 * len(study_slots)


# --------------------------------------------------------------------------
# dispatch_split (contract §7.3, item 62(b))
# --------------------------------------------------------------------------


def test_dispatch_split_half_up_case_c_half_n_five() -> None:
    # §8's example: three cohorts of sizes 2, 2, 1 (N = 5), c = 0.5, so
    # K = round-half-up(2.5) = 3, split 1/1/1 by largest remainder.
    catalogue = [("a", "A", 0.4), ("b", "B", 0.4), ("c", "C", 0.2)]
    cohort_codes = np.array([0, 0, 1, 1, 2])
    counts = locked_ev_counts([2, 2, 1], commitment_share=0.5)
    assert counts == [1, 1, 1]
    locked = locked_ev_mask(cohort_codes, counts)

    split = dispatch_split(
        catalogue, cohort_codes, locked, commitment_share=0.5, replan_threshold_gbp_per_mwh=10.0
    )

    assert split["cohort_id"].tolist() == ["a", "b", "c"]
    assert split["ev_count"].tolist() == [2, 2, 1]
    assert split["locked_count"].tolist() == [1, 1, 1]
    assert split["free_count"].tolist() == [1, 1, 0]
    assert (split["commitment_share"] == 0.5).all()
    assert (split["replan_threshold_gbp_per_mwh"] == 10.0).all()
    assert (split["evidence_kind"] == "illustrative_synthetic").all()
    assert split["ev_count"].dtype == np.int64


def test_dispatch_split_counts_are_read_straight_off_dispatch_locked() -> None:
    # dispatch_split must report whatever units.dispatch_locked says, not
    # recompute the §2 rule itself: an arbitrary (non-§2) locked pattern.
    catalogue = [("a", "A", 0.5), ("b", "B", 0.5)]
    cohort_codes = np.array([0, 0, 0, 1])
    locked = np.array([True, True, True, False])  # every "a" EV locked, "b" free

    split = dispatch_split(
        catalogue, cohort_codes, locked, commitment_share=0.75, replan_threshold_gbp_per_mwh=5.0
    )

    assert split.set_index("cohort_id")["locked_count"].to_dict() == {"a": 3, "b": 0}
    assert split.set_index("cohort_id")["free_count"].to_dict() == {"a": 0, "b": 1}


def test_dispatch_split_empty_cohort_is_zero_not_missing() -> None:
    catalogue = [("a", "A", 1.0), ("b", "B", 0.0)]
    cohort_codes = np.array([0, 0])
    locked = np.array([True, False])

    split = dispatch_split(
        catalogue, cohort_codes, locked, commitment_share=0.5, replan_threshold_gbp_per_mwh=10.0
    )

    empty = split.loc[split["cohort_id"].eq("b")].iloc[0]
    assert empty["ev_count"] == 0
    assert empty["locked_count"] == 0
    assert empty["free_count"] == 0


# --------------------------------------------------------------------------
# event_response_bands: the day_ahead_plan series (contract §7.4)
# --------------------------------------------------------------------------


def _dfs_turn_down_fixture(study_days: int = 2):
    study_slots = build_study_slots(date(2026, 1, 12), study_days=study_days)
    slot_count = len(study_slots)
    events = model_events.event_table(["dfs_turn_down"], study_slots)  # national, night 1
    fleet_world = pd.concat(
        [
            pd.DataFrame(
                {
                    "world_id": np.int64(0),
                    "path_id": path_id,
                    "slot_index": study_slots["slot_index"],
                    "home_import_kw": value,
                }
            )
            for path_id, value in (("normal", 1.0), ("selected", 2.0))
        ],
        ignore_index=True,
    )
    dispatch_sums = FixtureDispatchSums(
        day_ahead_home_import_kwh=np.full((1, slot_count), 3.0),
        day_ahead_zone_home_import_kwh=np.zeros((1, 4, slot_count)),
        locked_home_import_kwh=np.zeros((1, slot_count)),
        free_home_import_kwh=np.zeros((1, slot_count)),
        free_day_ahead_home_import_kwh=np.zeros((1, slot_count)),
        replan_count=np.zeros((1, slot_count), dtype=np.int64),
    )
    call_kwargs = dict(
        events=events,
        study_slots=study_slots,
        fleet_world=fleet_world,
        zone_world=None,
        net_demand_gw=np.zeros((1, slot_count)),
        day_ahead_gbp_per_mwh=np.zeros((1, slot_count)),
        imbalance_gbp_per_mwh=np.zeros((1, slot_count)),
        curve_values={},
    )
    return call_kwargs, dispatch_sums


def test_event_response_bands_adds_day_ahead_plan_series_after_selected() -> None:
    call_kwargs, dispatch_sums = _dfs_turn_down_fixture()

    bands = event_response_bands(**call_kwargs, dispatch_sums=dispatch_sums)

    # Added after "selected" (contract §7.4): normal, selected, day_ahead_plan
    # in that order, before difference and market.
    assert list(dict.fromkeys(bands["series"])) == [
        "normal",
        "selected",
        "day_ahead_plan",
        "difference",
        "market",
    ]
    day_ahead = bands.loc[bands["series"].eq("day_ahead_plan")]
    assert (day_ahead["metric"] == "home_import_kw").all()
    # day_ahead_home_import_kwh = 3.0 kWh -> 6.0 kW, one world, so every
    # statistic collapses to that one value.
    assert day_ahead[["mean", "p10", "p50", "p90"]].to_numpy() == pytest.approx(6.0)


def test_event_response_bands_omits_day_ahead_plan_without_dispatch_sums() -> None:
    call_kwargs, _ = _dfs_turn_down_fixture()

    bands = event_response_bands(**call_kwargs)  # dispatch_sums defaults to None

    assert "day_ahead_plan" not in set(bands["series"])
    assert list(dict.fromkeys(bands["series"])) == ["normal", "selected", "difference", "market"]


# --------------------------------------------------------------------------
# validate_dispatch_frames: identities, None rules (contract §8)
# --------------------------------------------------------------------------


def _consistent_result():
    """A duck-typed result with mutually consistent dispatch frames, for the validator."""

    study_slots = build_study_slots(date(2026, 1, 12)).iloc[:2].reset_index(drop=True)
    dispatch_sums = FixtureDispatchSums(
        day_ahead_home_import_kwh=np.array([[5.0, 3.0], [2.0, 4.0]]),
        day_ahead_zone_home_import_kwh=np.zeros((2, 4, 2)),
        locked_home_import_kwh=np.array([[2.0, 3.0], [1.0, 4.0]]),
        free_home_import_kwh=np.array([[2.0, 0.0], [2.0, 0.0]]),
        free_day_ahead_home_import_kwh=np.array([[3.0, 0.0], [1.0, 0.0]]),
        # Slot 0 (12:00, a decision instant) may carry replans; slot 1
        # (12:30) must not (§8: replan_count is 0 off decision slots).
        replan_count=np.array([[1, 0], [2, 0]], dtype=np.int64),
    )
    fleet = _toy_selected_fleet([[4.0, 3.0], [3.0, 4.0]])
    evaluation = _toy_evaluation_prices([[50.0, 55.0], [20.0, 25.0]])
    world_slot = dispatch_world_slot(dispatch_sums, fleet, study_slots, evaluation)
    bands = dispatch_bands(world_slot, study_slots)

    catalogue = [("a", "A", 0.5), ("b", "B", 0.5)]
    units = pd.DataFrame(
        {
            "unit_id": ["ev0", "ev1", "ev2"],
            "cohort_id": ["a", "a", "b"],
            "dispatch_locked": [True, False, False],
        }
    )
    cohort_codes = np.array([0, 0, 1])
    split = dispatch_split(
        catalogue,
        cohort_codes,
        units["dispatch_locked"].to_numpy(),
        commitment_share=1.0 / 3.0,
        replan_threshold_gbp_per_mwh=10.0,
    )
    fleet_world_intervals = fleet.assign(
        interval_start_utc=pd.NaT, interval_end_utc=pd.NaT, interval_start_london=pd.NaT
    )
    event_columns = ["series", "metric", "scope", "slot_index", "p10", "p50", "p90"]
    events_frame = pd.DataFrame(columns=event_columns)
    return SimpleNamespace(
        units=units,
        world_count=2,
        vehicle_count=3,
        fleet_world_intervals=fleet_world_intervals,
        evaluation_prices=evaluation,
        dispatch_world_slot=world_slot,
        dispatch_bands=bands,
        dispatch_split=split,
        event_response_bands=events_frame,
    )


def test_validate_dispatch_frames_passes_a_consistent_fixture() -> None:
    validate_dispatch_frames(_consistent_result())


def test_validate_dispatch_frames_catches_a_broken_locked_free_identity() -> None:
    result = _consistent_result()
    broken = result.dispatch_world_slot.copy()
    broken.loc[0, "free_kwh"] = 999.0
    result.dispatch_world_slot = broken
    with pytest.raises(AssertionError, match="locked \\+ free"):
        validate_dispatch_frames(result)


def test_validate_dispatch_frames_catches_a_replan_off_a_decision_slot() -> None:
    result = _consistent_result()
    broken = result.dispatch_world_slot.copy()
    # Slot index 1 (12:30) is not a decision instant.
    broken.loc[broken["slot_index"].eq(1), "replan_count"] = 1
    result.dispatch_world_slot = broken
    with pytest.raises(AssertionError, match="decision slots"):
        validate_dispatch_frames(result)


def test_validate_dispatch_frames_none_rules_without_the_switch() -> None:
    units = pd.DataFrame({"unit_id": ["ev0"], "cohort_id": ["a"], "dispatch_locked": [False]})
    result = SimpleNamespace(
        units=units,
        dispatch_world_slot=None,
        dispatch_bands=None,
        dispatch_split=None,
        event_response_bands=None,
    )
    validate_dispatch_frames(result)  # must not raise


def test_validate_dispatch_frames_rejects_a_locked_ev_without_the_switch() -> None:
    units = pd.DataFrame({"unit_id": ["ev0"], "cohort_id": ["a"], "dispatch_locked": [True]})
    result = SimpleNamespace(
        units=units,
        dispatch_world_slot=None,
        dispatch_bands=None,
        dispatch_split=None,
        event_response_bands=None,
    )
    with pytest.raises(AssertionError, match="all False"):
        validate_dispatch_frames(result)


# --------------------------------------------------------------------------
# Clock-change week: slot counts (item 52) must not leak an assumption of
# 48 slots per night into the dispatch frames.
# --------------------------------------------------------------------------


def test_dispatch_frames_across_an_autumn_clock_change_week() -> None:
    study_slots = build_study_slots(date(2026, 10, 21))  # clocks go back Sun 25 Oct
    slot_count = len(study_slots)
    assert slot_count == 336  # always 48 x 7 fixed UTC slots (results-v2 §1 rule 4)
    per_night = study_slots.groupby("local_date").size()
    assert per_night.iloc[3] == 50  # the night the clock goes back

    rng = np.random.default_rng(0)
    day_ahead = rng.uniform(0, 5, size=(2, slot_count))
    locked = day_ahead * 0.4
    free_day_ahead = day_ahead - locked
    dispatched = day_ahead - 0.1  # a uniform small turn-down, so it stays hand-checkable
    free = dispatched - locked
    dispatch_sums = FixtureDispatchSums(
        day_ahead_home_import_kwh=day_ahead,
        day_ahead_zone_home_import_kwh=np.zeros((2, 4, slot_count)),
        locked_home_import_kwh=locked,
        free_home_import_kwh=free,
        free_day_ahead_home_import_kwh=free_day_ahead,
        replan_count=np.zeros((2, slot_count), dtype=np.int64),
    )
    fleet = _toy_selected_fleet(dispatched.tolist())
    evaluation = _toy_evaluation_prices((day_ahead * 10).tolist())

    world_slot = dispatch_world_slot(dispatch_sums, fleet, study_slots, evaluation)
    assert len(world_slot) == 2 * slot_count
    assert world_slot.groupby("world_id").size().tolist() == [slot_count, slot_count]
    # night_index must match study_slots exactly, clock-change night included.
    assert world_slot.loc[world_slot["world_id"].eq(0), "night_index"].tolist() == (
        study_slots["night_index"].tolist()
    )

    bands = dispatch_bands(world_slot, study_slots)
    assert len(bands) == 4 * slot_count
    for series, metric in (
        ("day_ahead_plan", "home_import_kw"),
        ("dispatched", "home_import_kw"),
        ("difference", "home_import_kw"),
        ("dispatched", "replan_count"),
    ):
        rows = bands.loc[bands["series"].eq(series) & bands["metric"].eq(metric)]
        assert rows["slot_index"].tolist() == study_slots["slot_index"].tolist()
