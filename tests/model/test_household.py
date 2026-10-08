"""Household frames (household contract v1 §3.1-§3.2, §9 HH1 rows) on SYNTHETIC fixtures.

Hand-checkable arrays first, then the synthetic ``make_result`` fixture
(including clock-change weeks) and small seeded real runs.  Every figure
is illustrative and synthetic.
"""

from __future__ import annotations

import inspect
from datetime import date
from functools import cache

import numpy as np
import pandas as pd
import pytest
from fixtures.availability_contract import fixture_totals, make_per_ev_fixture, per_ev_realised
from fixtures.household_contract import (
    EV_WORLD_COLUMNS,
    carbon_intensity,
    make_household_frames,
    price_matrix,
    validate_household_v2,
)
from fixtures.result_fixture import make_result, validate_result_v2

from axle_studio.model import assumptions, availability, household, summaries
from axle_studio.model.forecast import run_forecast, run_forecast_from_assumptions

_START = date(2026, 10, 12)  # a Monday, no clock change
_SMALL = {"vehicle_count": 30, "evaluation_world_count": 4}


# --------------------------------------------------------------------------
# Hand-built per-EV arrays
# --------------------------------------------------------------------------


def _slots(night_index) -> pd.DataFrame:
    return pd.DataFrame({"night_index": np.asarray(night_index, dtype=np.int64)})


def _by_path(normal: dict, selected: dict, connected: np.ndarray) -> dict:
    """Both paths' per-EV arrays (world, slot, EV); missing quantities are zero."""

    out = {}
    for path, given in (("normal", normal), ("selected", selected)):
        unit = {
            name: np.zeros(connected.shape)
            for name in (
                "home_grid_import_kwh",
                "public_grid_import_kwh",
                "closing_battery_kwh",
                "unserved_travel_battery_kwh",
            )
        }
        unit.update({name: np.asarray(v, dtype=float) for name, v in given.items()})
        unit["connected"] = connected
        out[path] = unit
    return out


def _frames(
    by_path,
    *,
    night_index,
    price,
    capacity,
    target,
    saving=None,
    value=None,
    intensity=None,
    groups=None,
    rate=0.79,
):
    """Run the chunk hook on one chunk and build both frames."""

    connected = by_path["normal"]["connected"]
    world_count, _, vehicle_count = connected.shape
    slots = _slots(night_index)
    price = np.asarray(price, dtype=float)
    totals = household.new_totals(
        world_count, slots, capacity_kwh=np.asarray(capacity), target_kwh=np.asarray(target)
    )
    normal = by_path["normal"]
    sessions = summaries.find_sessions(
        connected, normal["closing_battery_kwh"], normal["home_grid_import_kwh"]
    )
    household.accumulate(
        totals,
        np.arange(world_count),
        by_path,
        sessions,
        day_ahead=price,
        low_band=summaries.price_third_bands(price) == "low",
        intensity=intensity,
    )
    zeros = np.zeros((world_count, vehicle_count))
    units = pd.DataFrame(
        {
            "unit_id": [f"ev-{i}" for i in range(vehicle_count)],
            "cohort_id": ["c"] * vehicle_count,
        }
    )
    return household.build_frames(
        totals,
        units=units,
        saving_gbp=zeros if saving is None else saving,
        per_ev={
            "value_gbp_per_month": zeros if value is None else value,
            "earning": zeros > 0,
        },
        evening_ev_kw=None,
        groups=groups or {"fleet": np.ones(vehicle_count, dtype=bool)},
        public_charge_gbp_per_kwh=rate,
    )


def _summary_value(summary, metric, statistic, column="p50", group_id="fleet"):
    row = summary.loc[
        summary["group_id"].eq(group_id)
        & summary["metric"].eq(metric)
        & summary["statistic"].eq(statistic)
    ]
    assert len(row) == 1
    return float(row[column].iat[0])


def test_hand_fixture_one_ev_two_slots_costs_saving_and_cheap_share():
    # Prices [100, 50] GBP/MWh; unmanaged u = [10, 0], smart m = [2, 8] kWh;
    # one 1 kWh public top-up on the smart path at 0.79 GBP/kWh.
    connected = np.zeros((1, 2, 1), dtype=bool)
    by_path = _by_path(
        {"home_grid_import_kwh": [[[10.0], [0.0]]]},
        {"home_grid_import_kwh": [[[2.0], [8.0]]], "public_grid_import_kwh": [[[0.0], [1.0]]]},
        connected,
    )
    price = [[100.0, 50.0]]
    # The saving is the market lane's function on the same arrays, stored as is.
    chunk = summaries.household_chunk(by_path, np.asarray(price), np.zeros((1, 2)), np.array([0]))
    cost = pd.DataFrame(
        {
            "world_id": [0],
            "illustrative_unrecovered_energy_value_gbp": [0.0],
            "illustrative_unserved_travel_value_gbp": [0.0],
        }
    )
    saving = summaries.household_saving_gbp(chunk, cost, 0.79)
    frames = _frames(
        by_path, night_index=[0, 0], price=price, capacity=[60.0], target=[48.0], saving=saving
    )
    row = frames["household_ev_world"].iloc[0]
    assert row["home_cost_gbp_normal"] == pytest.approx(1.00)
    assert row["home_cost_gbp_selected"] == pytest.approx(0.60)
    assert row["public_cost_gbp_selected"] == pytest.approx(0.79)
    assert row["public_cost_gbp_normal"] == 0.0
    assert row["supplier_energy_saving_gbp"] == pytest.approx(0.40)
    assert row["saving_gbp"] == pytest.approx(-0.39)
    # Slot 1 (50 GBP/MWh) is the world's low third.
    assert row["cheap_import_kwh_normal"] == 0.0
    assert row["cheap_import_kwh_selected"] == 8.0
    values = household.metric_values(frames["household_ev_world"], 1).iloc[0]
    assert values["cheap_share_normal"] == 0.0
    assert values["cheap_share_selected"] == pytest.approx(0.8)
    assert values["cheap_share_difference"] == pytest.approx(0.8)
    assert values["home_cost_gbp_per_kwh_difference"] == pytest.approx(0.06 - 0.1)


def _stock(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def test_completion_readiness_and_the_two_session_populations():
    # One world, 14 slots, target 48 kWh of 60.  EV 0: a run from slot 0
    # (a session end, not a session), a session complete on both paths, one
    # the smart path leaves 3 kWh short, and one still plugged in at the end.
    # EV 1: plugged in all week.  EV 2: a run from slot 0 plus one closed
    # session, so session ends = sessions + 1.
    slots = 14
    connected = np.zeros((1, slots, 3), dtype=bool)
    connected[0, 0:3, 0] = connected[0, 4:7, 0] = connected[0, 8:10, 0] = True
    connected[0, 11:, 0] = True
    connected[0, :, 1] = True
    connected[0, 0:2, 2] = connected[0, 5:8, 2] = True
    normal = np.full((1, slots, 3), 48.0)
    selected = normal.copy()
    normal[0, 6, 0] = selected[0, 6, 0] = 50.0  # session B ends at 50 kWh on both paths
    selected[0, 9, 0] = 45.0  # session C ends 3 kWh short on the smart path
    by_path = _by_path(
        {"closing_battery_kwh": normal}, {"closing_battery_kwh": selected}, connected
    )
    frames = _frames(
        by_path,
        night_index=[0] * 7 + [1] * 7,
        price=[np.linspace(100.0, 30.0, slots)],
        capacity=[60.0] * 3,
        target=[48.0] * 3,
    )
    ev = frames["household_ev_world"].set_index("unit_id")
    assert ev.loc["ev-0", "session_count"] == 2
    assert ev.loc["ev-0", "session_end_count"] == 4
    assert ev.loc["ev-0", "completed_count_normal"] == 2
    assert ev.loc["ev-0", "completed_count_selected"] == 1
    assert ev.loc["ev-0", "sessions_affected_count"] == 1
    assert ev.loc["ev-0", "departure_soc_mean_normal"] == pytest.approx((50 / 60 + 0.8) * 50)
    assert ev.loc["ev-0", "departure_soc_mean_selected"] == pytest.approx((50 / 60 + 0.75) * 50)
    assert ev.loc["ev-1", "session_count"] == 0
    assert ev.loc["ev-1", "session_end_count"] == 1
    assert np.isnan(ev.loc["ev-1", "departure_soc_mean_normal"])
    assert ev.loc["ev-2", "session_end_count"] == ev.loc["ev-2", "session_count"] + 1 == 2
    summary = frames["household_outcomes_summary"]
    rows = summary.loc[summary["metric"].eq("completed_share_selected")]
    assert (rows["ev_count"] == 3).all()
    assert (rows["ev_value_count"] == 2).all()  # EV 1 has no closed session (O4)
    energy = summary.loc[summary["metric"].eq("sessions_per_week")]
    assert (energy["ev_value_count"] == 3).all()


def test_group_ratio_is_session_weighted_not_a_mean_of_shares():
    # Customer A: one session, complete.  Customer B: six sessions, three
    # complete.  Mean of shares 0.75; group ratio 4 / 7 (B2).
    frame = _ev_world_rows(
        [
            {"session_count": 1, "session_end_count": 1, "completed_count_selected": 1},
            {"session_count": 6, "session_end_count": 6, "completed_count_selected": 3},
        ]
    )
    summary = household.outcomes_summary(
        frame, groups={"fleet": np.ones(2, dtype=bool)}, night_count=7
    )
    assert _summary_value(summary, "completed_share_selected", "mean_across_evs") == 0.75
    assert _summary_value(summary, "completed_share_selected", "group_ratio") == pytest.approx(
        4 / 7
    )


def test_yearly_rows_read_ev_means_not_one_weeks_spread():
    # EV 0 saves 10 then -8 GBP; EV 1 saves 1 in both weeks (B1).
    frame = _ev_world_rows(
        [{"saving_gbp": 10.0}, {"saving_gbp": 1.0}, {"saving_gbp": -8.0}, {"saving_gbp": 1.0}],
        world_count=2,
    )
    groups = {"fleet": np.ones(2, dtype=bool)}
    summary = household.outcomes_summary(frame, groups=groups, night_count=7)
    for statistic in ("mean_of_ev_means", "p10_of_ev_means", "p90_of_ev_means"):
        assert _summary_value(summary, "saving_gbp_per_year", statistic) == pytest.approx(52.0)
    assert _summary_value(summary, "saving_gbp_per_year", "share_of_ev_means_below_zero") == 0.0
    year = summary.loc[summary["metric"].eq("saving_gbp_per_year")]
    assert set(year["statistic"]) == set(household.SET_B)  # no set A for a scenario metric
    assert year[["p10", "p90"]].isna().all().all()
    assert (year["mean"] == year["p50"]).all()
    # Set A: per world P10 over EVs (1.9, -7.1), then P10 across worlds.
    assert _summary_value(summary, "saving_gbp_per_week", "p10_across_evs", "p10") == pytest.approx(
        -6.2
    )
    worse = _ev_world_rows(
        [{"saving_gbp": 10.0}, {"saving_gbp": 1.0}, {"saving_gbp": -12.0}, {"saving_gbp": 1.0}],
        world_count=2,
    )
    summary = household.outcomes_summary(worse, groups=groups, night_count=7)
    assert _summary_value(summary, "saving_gbp_per_year", "share_of_ev_means_below_zero") == 0.5


def _ev_world_rows(rows: list[dict], world_count: int = 1) -> pd.DataFrame:
    """A ``household_ev_world``-shaped frame of hand rows (world-major); other columns 0."""

    vehicle_count = len(rows) // world_count
    base = {
        name: (0 if dtype in (np.int64, np.float64) else False)
        for name, dtype in household.EV_WORLD_COLUMNS.items()
    }
    frame = pd.DataFrame(
        [
            base
            | {
                "world_id": index // vehicle_count,
                "unit_id": f"ev-{index % vehicle_count}",
                "cohort_id": "c",
                "treated": True,
                "evidence_kind": "illustrative_synthetic",
            }
            | row
            for index, row in enumerate(rows)
        ]
    )
    return frame.astype(household.EV_WORLD_COLUMNS)


def test_co2_needs_an_intensity_and_splits_the_fleet_figure_exactly():
    connected = np.zeros((2, 3, 2), dtype=bool)
    rng = np.random.default_rng(3)
    u, m = rng.uniform(0, 5, (2, 3, 2)), rng.uniform(0, 5, (2, 3, 2))
    by_path = _by_path({"home_grid_import_kwh": u}, {"home_grid_import_kwh": m}, connected)
    common = {
        "night_index": [0, 0, 0],
        "price": [[50.0, 60.0, 70.0]] * 2,
        "capacity": [60.0] * 2,
        "target": [48.0] * 2,
    }
    without = _frames(by_path, **common)["household_ev_world"]
    assert without["co2_shifted_kg"].isna().all()
    intensity = rng.uniform(100, 300, (2, 3))  # gCO2/kWh, SYNTHETIC
    frame = _frames(by_path, intensity=intensity, **common)["household_ev_world"]
    fleet = ((u - m).sum(axis=2) * intensity).sum(axis=1) / 1000.0
    per_world = frame.groupby("world_id")["co2_shifted_kg"].sum().to_numpy()
    assert np.allclose(per_world, fleet, rtol=0, atol=1e-12)


def test_evening_ev_kw_is_the_one_hour_evening_mean_of_realised_contributions():
    # Two makers, a 10 % control group and a blackout inside night 2's
    # evening, SYNTHETIC (availability contract fixture).
    blackout = np.zeros(336, dtype=bool)
    blackout[48 * 2 + 12 : 48 * 2 + 14] = True  # 18:00-19:00 on night 2
    fixture = make_per_ev_fixture(
        world_count=4,
        vehicle_count=30,
        seed=11,
        control_share=0.1,
        blackout=blackout,
        response_rate=(0.9, 0.9),
        outage_probability=(0.2, 0.0),
    )
    totals = fixture_totals(fixture)
    realised = per_ev_realised(fixture)  # (world, slot, cell, EV)
    labels = fixture.study_slots["local_time_label"].to_numpy(dtype=object)
    evening = (labels >= "17:00") & (labels < "21:00")
    for direction_index, direction in enumerate(availability.DIRECTIONS):
        cell = availability.CELLS.index((direction, 1.0))
        expected = np.nanmean(realised[:, evening, cell, :], axis=1)
        assert np.array_equal(totals["evening_ev_kw"][..., direction_index], expected)
        # Summed over EVs it is the fleet's evening mean of deliverable kW.
        fleet = totals["deliverable"][:, evening, cell].mean(axis=1)
        assert np.allclose(expected.sum(axis=1), fleet, rtol=0, atol=1e-6)
    control = fixture.draws["control"]
    assert control.any() and (totals["evening_ev_kw"][:, control, :] == 0.0).all()


# --------------------------------------------------------------------------
# The synthetic make_result fixture, including clock-change weeks
# --------------------------------------------------------------------------


def _model_frames_on_fixture(result) -> pd.DataFrame:
    """The model's chunk hook run on the fixture's per-EV toy arrays."""

    sim = result.replay_state
    units = result.units
    capacity = units["physical_capacity_kwh"].to_numpy(dtype=float)
    target = capacity * units["preferred_target_soc_percent"].to_numpy(dtype=float) / 100.0
    by_path = {
        path: {
            "home_grid_import_kwh": sim.home_import_kwh[path],
            "public_grid_import_kwh": sim.public_import_kwh[path],
            "closing_battery_kwh": sim.closing_kwh[path],
            "connected": sim.connected["normal"],
        }
        for path in ("normal", "selected")
    }
    totals = household.new_totals(
        result.world_count, result.study_slots, capacity_kwh=capacity, target_kwh=target
    )
    worlds = np.arange(result.world_count)
    intensity = carbon_intensity(result)
    for chunk in (worlds[:2], worlds[2:]):  # two chunks, as the real loop runs them
        piece = {p: {k: v[chunk] for k, v in unit.items()} for p, unit in by_path.items()}
        normal = piece["normal"]
        price = price_matrix(result)
        household.accumulate(
            totals,
            chunk,
            piece,
            summaries.find_sessions(
                normal["connected"], normal["closing_battery_kwh"], normal["home_grid_import_kwh"]
            ),
            day_ahead=price[chunk],
            low_band=(summaries.price_third_bands(price) == "low")[chunk],
            intensity=intensity[chunk],
        )
    expected = result.household_ev_world
    shape = (result.world_count, result.vehicle_count)
    frames = household.build_frames(
        totals,
        units=units,
        saving_gbp=expected["saving_gbp"].to_numpy().reshape(shape),
        per_ev={
            "value_gbp_per_month": expected["value_gbp_per_month"].to_numpy().reshape(shape),
            "earning": expected["earning"].to_numpy().reshape(shape),
        },
        evening_ev_kw=None,
        groups={"fleet": np.ones(result.vehicle_count, dtype=bool)}
        | {c: units["cohort_id"].eq(c).to_numpy() for c in dict.fromkeys(units["cohort_id"])},
        public_charge_gbp_per_kwh=0.79,
    )
    return frames["household_ev_world"]


@pytest.mark.parametrize("clock_change", [None, "autumn", "spring"])
def test_model_hook_matches_the_independent_fixture_loops(clock_change):
    # The fixture's frame is built by plain per-EV loops (household_contract);
    # the model's reduceat-on-night-starts hook must agree on every column,
    # including a 50-slot (autumn) and a 46-slot (spring) night (B4).
    result = make_result(evs=12, worlds=4, clock_change=clock_change)
    validate_result_v2(result)
    model = _model_frames_on_fixture(result)
    expected = result.household_ev_world
    for column in EV_WORLD_COLUMNS:
        if expected[column].dtype == np.float64:
            assert np.allclose(
                model[column], expected[column], rtol=1e-12, atol=1e-9, equal_nan=True
            ), column
        else:
            assert (model[column].to_numpy() == expected[column].to_numpy()).all(), column


@pytest.mark.parametrize("clock_change", ["autumn", "spring"])
def test_nights_plugged_on_clock_change_weeks(clock_change):
    result = make_result(evs=3, worlds=1, clock_change=clock_change)
    slots = result.study_slots
    night = slots["night_index"].to_numpy()
    lengths = np.bincount(night)
    assert set(lengths) >= {50 if clock_change == "autumn" else 46}
    night_count = int(night.max()) + 1
    starts = np.flatnonzero(np.diff(night, prepend=-1))
    connected = np.zeros((1, len(slots), 3), dtype=bool)
    connected[0, starts + 10, 0] = True  # one slot on every night
    connected[0, starts[[0, 2, 3]] + 10, 1] = True  # nights 0, 2 and 3
    # EV 2 never plugs in.
    by_path = _by_path(
        {"closing_battery_kwh": np.full(connected.shape, 48.0)},
        {"closing_battery_kwh": np.full(connected.shape, 48.0)},
        connected,
    )
    frames = _frames(
        by_path,
        night_index=night,
        price=[np.arange(len(slots), dtype=float)],
        capacity=[60.0] * 3,
        target=[48.0] * 3,
    )
    ev = frames["household_ev_world"]
    assert list(ev["nights_plugged_count"]) == [night_count, 3, 0]
    assert np.isnan(ev["departure_soc_mean_normal"].iat[2])
    # The hook never assumes a 48-slot night or a 336-slot week (B4).
    source = inspect.getsource(household.new_totals) + inspect.getsource(household.accumulate)
    assert "336" not in source and "48" not in source


def test_validator_rejects_broken_frames():
    result = make_result(evs=6, worlds=4)
    frame = result.household_ev_world.copy()
    frame.loc[0, "home_import_kwh_normal"] += 1.0
    with pytest.raises(AssertionError, match="home_import_kwh_normal"):
        validate_household_v2(_with(result, household_ev_world=frame))
    summary = result.household_outcomes_summary.copy()
    summary.loc[summary["statistic"].eq("group_ratio"), "p50"] += 0.01
    with pytest.raises(AssertionError, match="summary p50"):
        validate_household_v2(_with(result, household_outcomes_summary=summary))
    frame = result.household_ev_world.copy()
    frame["completed_count_selected"] = frame["session_count"] + 1
    with pytest.raises(AssertionError, match="completed"):
        validate_household_v2(_with(result, household_ev_world=frame))
    with pytest.raises(AssertionError, match="None without trading"):
        validate_household_v2(_with(make_result("no_action"), household_ev_world=frame))


def _with(result, **changes):
    from dataclasses import replace

    return replace(result, **changes)


def test_fixture_builder_is_none_on_no_action():
    result = make_result("no_action")
    assert make_household_frames(result) == {
        "household_ev_world": None,
        "household_outcomes_summary": None,
    }


# --------------------------------------------------------------------------
# Real runs (SYNTHETIC, seeded)
# --------------------------------------------------------------------------


@cache
def _run():
    return run_forecast_from_assumptions(_START, values=_SMALL)


def _captured_run(monkeypatch, trading: dict[str, float]):
    """A fresh run recording household_frames' per_ev and household_chunk's arrays."""

    seen: dict[str, list] = {"per_ev": [], "chunks": []}
    frames, chunk = summaries.household_frames, summaries.household_chunk

    def record_frames(*args, **kwargs):
        out = frames(*args, **kwargs)
        seen["per_ev"].append(out[2])
        return out

    def record_chunk(*args, **kwargs):
        out = chunk(*args, **kwargs)
        seen["chunks"].append(out)
        return out

    monkeypatch.setattr(summaries, "household_frames", record_frames)
    monkeypatch.setattr(summaries, "household_chunk", record_chunk)
    return _direct(trading), seen


def _direct(trading: dict[str, float], supplier_edits: dict[str, float] | None = None):
    """A run with edited trading or supplier inputs, as test_trading_run builds it."""

    inputs = assumptions.forecast_inputs(
        assumptions.resolve_values(_SMALL), warmup_days=7, study_days=7
    )
    inputs["trading_assumptions"] = inputs["trading_assumptions"] | trading
    return run_forecast(
        assumptions.run_settings(_SMALL, _START),
        assumptions.cohort_fixture(_SMALL),
        supplier_inputs=assumptions.supplier_inputs() | (supplier_edits or {}),
        **inputs,
    )


def test_real_run_reconciles_and_stores_the_market_arrays_bit_for_bit(monkeypatch):
    result, seen = _captured_run(
        monkeypatch, {"trading.control_group": 1.0, "trading.control_group_share": 0.2}
    )
    validate_household_v2(result)  # a direct run has no records for validate_result_v2
    frame = result.household_ev_world
    shape = (result.world_count, result.vehicle_count)
    (per_ev,) = seen["per_ev"]
    assert np.array_equal(
        frame["value_gbp_per_month"].to_numpy().reshape(shape), per_ev["value_gbp_per_month"]
    )
    assert np.array_equal(frame["earning"].to_numpy().reshape(shape), per_ev["earning"])
    # The two absolute home costs differ by household_chunk's home_cost_gbp (O7).
    home_cost = np.concatenate([c["home_cost_gbp"] for c in seen["chunks"]])
    difference = (frame["home_cost_gbp_selected"] - frame["home_cost_gbp_normal"]).to_numpy()
    assert np.allclose(difference.reshape(shape), home_cost, rtol=0, atol=1e-9)
    # Control EVs: in the frame, share 0, and outside every group statistic.
    control = result.units["control_group"].to_numpy()
    assert control.any()
    rows = frame.loc[~frame["treated"]]
    assert len(rows) == control.sum() * result.world_count
    assert np.allclose(rows["value_gbp_per_month"] * 12 / 52, rows["saving_gbp"], atol=1e-12)
    summary = result.household_outcomes_summary
    fleet = summary.loc[summary["group_id"].eq("fleet")]
    assert (fleet["ev_count"] == (~control).sum()).all()
    # The fleet's nights and the fleet frame's connection agree.
    assert list(dict.fromkeys(summary["group_id"])) == list(
        dict.fromkeys(result.household_value_summary["group_id"])
    )


def test_default_run_passes_the_validator():
    validate_result_v2(_run())


def test_two_runs_with_one_seed_give_identical_frames():
    first = _run()
    second = run_forecast_from_assumptions(_START, values=_SMALL)
    for name in ("household_ev_world", "household_outcomes_summary"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))


def test_no_action_run_has_no_household_frames():
    result = run_forecast_from_assumptions(_START, model="no_action", values=_SMALL)
    assert result.household_ev_world is None and result.household_outcomes_summary is None
    validate_result_v2(result)


def test_editing_the_revenue_share_changes_only_value_rows():
    base = _direct({})
    edited = _direct({"trading.customer_revenue_share": 0.3})
    for name in ("fleet_world_intervals", "forecast_prices", "deviation_world_slot"):
        pd.testing.assert_frame_equal(getattr(base, name), getattr(edited, name))
    a, b = base.household_ev_world, edited.household_ev_world
    changed = [c for c in a.columns if not a[c].equals(b[c])]
    assert changed == ["value_gbp_per_month"]
    sa, sb = base.household_outcomes_summary, edited.household_outcomes_summary
    differs = ~np.isclose(sa["mean"], sb["mean"], rtol=0, atol=0, equal_nan=True)
    assert set(sa.loc[differs, "metric"]) == {
        "value_gbp_per_week",
        "value_gbp_per_month",
        "value_gbp_per_year",
    }


# --------------------------------------------------------------------------
# The lead's two summaries helpers reproduce the code they replaced (B7)
# --------------------------------------------------------------------------


def _old_session_values(sessions, *, study_slots, capacity_kwh, target_kwh, power_kw, efficiency):
    """Frozen copy of the per-session values session_distribution_rows formed inline."""

    slot_count = len(study_slots)
    london = study_slots["interval_start_london"]
    clock = (london.dt.hour + london.dt.minute / 60.0).to_numpy(dtype=float)
    first, end, ev = sessions.first_slot, sessions.end_slot, sessions.ev
    closed = end < slot_count
    need = np.round(np.maximum(target_kwh[ev] - sessions.plug_in_kwh, 0.0), 9)
    dwell = np.where(closed, (end - first) * 0.5, np.nan)
    battery_kw = power_kw[ev] * efficiency
    return {
        "plug_in_time": clock[first],
        "departure_time": np.where(closed, clock[np.minimum(end, slot_count - 1)], np.nan),
        "plug_in_soc_percent": 100.0 * sessions.plug_in_kwh / capacity_kwh[ev],
        "energy_needed_kwh": need,
        "dwell_hours": dwell,
        "flexible_kwh": np.minimum(need, dwell * battery_kw),
        "slack_hours": dwell - need / battery_kw,
    }


def test_session_values_and_price_third_bands_reproduce_the_replaced_code():
    result = _run()
    state = result.replay_state
    units = state.units
    capacity = units["physical_capacity_kwh"].to_numpy(dtype=float)
    arguments = {
        "study_slots": result.study_slots,
        "capacity_kwh": capacity,
        "target_kwh": capacity * units["preferred_target_soc_fraction"].to_numpy(dtype=float),
        "power_kw": units["home_charger_limit_kw"].to_numpy(dtype=float),
        "efficiency": state.settings.home_charge_efficiency,
    }
    worlds, by_path = next(
        summaries._per_ev_chunks(
            state, result.study_slots, result.fleet_world_intervals, ["normal"]
        )
    )
    normal = by_path["normal"]
    sessions = summaries.find_sessions(
        normal["connected"], normal["closing_battery_kwh"], normal["home_grid_import_kwh"]
    )
    new = summaries.session_values(sessions, **arguments)
    old = _old_session_values(sessions, **arguments)
    # "departure_soc_percent_timed" (decision 0007, model step 2) is a key
    # only when ``timed_closing_kwh`` is given, unlike this test's call.
    assert list(new) == [
        m for m in summaries.SESSION_DISTRIBUTION_METRICS if m != "departure_soc_percent_timed"
    ]
    for metric in old:
        assert np.array_equal(new[metric], old[metric], equal_nan=True), metric
    price = price_matrix(result)
    low_edge, high_edge = np.quantile(price, (1 / 3, 2 / 3), axis=1, keepdims=True)
    old_band = np.where(price < low_edge, "low", np.where(price > high_edge, "high", "middle"))
    assert np.array_equal(summaries.price_third_bands(price), old_band)
    pd.testing.assert_frame_equal(
        summaries.price_band_shift(
            result.fleet_world_intervals, result.forecast_prices, result.study_slots
        ),
        result.price_band_shift,
    )
    del worlds


def test_j5_charge_completion_cross_check_on_a_real_run():
    # J5's frame needs J1b's inputs on a full run, so it is built here from
    # the same run's per-EV chunks with J5's own functions and attached; the
    # control group is off by default, so the two populations coincide.
    from dataclasses import replace

    from axle_studio.model import product

    result = _run()
    assert not result.units["control_group"].any()
    state = result.replay_state
    units = state.units
    cohorts = units["cohort_id"].to_numpy(dtype=object)
    group_ids = list(dict.fromkeys(result.household_value_summary["group_id"]))
    members = np.stack(
        [np.ones(len(units), bool) if g == "fleet" else cohorts == g for g in group_ids], axis=1
    )
    capacity = units["physical_capacity_kwh"].to_numpy(dtype=float)
    target = capacity * units["preferred_target_soc_fraction"].to_numpy(dtype=float)
    expected_end = availability.expected_end_slots(state.flexibility.hours_to_departure)
    counts = {}
    for worlds, by_path in summaries._per_ev_chunks(
        state, result.study_slots, result.fleet_world_intervals, ["normal", "selected"]
    ):
        for p, path in enumerate(product.PATH_IDS):
            unit = by_path[path]
            sessions = summaries.find_sessions(
                unit["connected"], unit["closing_battery_kwh"], unit["home_grid_import_kwh"]
            )
            chunk = product.completion_counts(
                sessions,
                slot_count=len(result.study_slots),
                target_kwh=target,
                expected_end_slot=expected_end,
                members=members,
                world_count=len(worlds),
            )
            for name, values in chunk.items():
                shape = (result.world_count, len(group_ids), 2, values.shape[-1])
                counts.setdefault(name, np.zeros(shape))[worlds, :, p] = values
    completion = product.charge_completion_summary(counts, group_ids, members.sum(axis=0))
    validate_household_v2(replace(result, charge_completion_summary=completion))
    broken = completion.copy()
    broken["completed_share_p50"] += 0.01
    with pytest.raises(AssertionError, match="J5 completed share"):
        validate_household_v2(replace(result, charge_completion_summary=broken))


def test_carbon_edits_change_only_co2_and_supplier_edits_change_nothing_here():
    base = _direct({})
    assert not base.household_ev_world["co2_shifted_kg"].isna().any()
    carbon = _direct({}, {"carbon.intensity_slope_gco2_per_kwh_per_gw": 20.0})
    rows = _direct({}, {"supplier.platform_fee_gbp_per_ev_per_month": 1.0})
    for name in ("fleet_world_intervals", "deviation_world_slot"):
        pd.testing.assert_frame_equal(getattr(base, name), getattr(carbon, name))
    a, b = base.household_ev_world, carbon.household_ev_world
    assert [c for c in a.columns if not a[c].equals(b[c])] == ["co2_shifted_kg"]
    sa, sb = base.household_outcomes_summary, carbon.household_outcomes_summary
    differs = ~np.isclose(sa["mean"], sb["mean"], rtol=0, atol=0, equal_nan=True)
    assert set(sa.loc[differs, "metric"]) == {"co2_shifted_kg_per_week", "co2_shifted_kg_per_month"}
    for name in ("household_ev_world", "household_outcomes_summary"):
        pd.testing.assert_frame_equal(getattr(base, name), getattr(rows, name))
