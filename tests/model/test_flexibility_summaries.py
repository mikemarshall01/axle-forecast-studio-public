"""Deferrable power, weekly flexibility, half-hour plug heatmaps and relative prices.

Decision 0004 item 54; plan C2, C4, D-1, D-3 and E1; contract 3.6d, 3.6e,
3.10b, 4.5a, 4.7a and 4.7b.  Hand-checkable inputs first (round numbers,
one or two EVs), then the validator on real runs and on tampered frames.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fixtures.result_fixture import make_result, validate_result_v2

from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.model.individual import replay_one_ev
from axle_studio.model.summaries import (
    DEFERRABLE_BUCKET_ORDER,
    SLACK_BUCKETS,
    FlexibilityInputs,
    build_study_slots,
    cheapest_half_hour_summary,
    deferrable_power_bands,
    deferrable_power_per_world,
    flexibility_inputs,
    flexibility_per_world,
    flexibility_weekly_summary,
    plug_event_heatmap,
    price_relative_bands,
)

# --------------------------------------------------------------------------
# Deferrable power by slack (contract 3.6d)
# --------------------------------------------------------------------------


def _five_evs(hours_to_departure: list[float]) -> FlexibilityInputs:
    """Five EVs, one slot: 4 kW chargers (2 kWh a half-hour), 10 kWh target."""

    return FlexibilityInputs(
        hours_to_departure=np.array([hours_to_departure]),  # (slot, EV)
        target_kwh=np.full(5, 10.0),
        power_kw=np.full(5, 4.0),
        efficiency=1.0,
    )


def test_deferrable_power_buckets_by_whole_slot_slack_by_hand() -> None:
    # Needs (kWh): 1, 2, 3, 4 and 0 (at target).  At 2 kWh a half-hour the
    # whole-slot needs are 0.5, 0.5, 1.0 and 1.0 h: 1 kWh still takes a whole
    # slot, and exactly 2 kWh is one slot, not two.  Hours to departure
    # 1.4, 2.5, 3.0, 12.0 and 5.0 give slack 0.9 (under 1 h), 2.0 (on the
    # edge: 2-4 h), 2.0 (2-4 h) and 11.0 (8 h or more).  The EV at target
    # has nothing to move and is in no bucket.
    connected = np.ones((1, 1, 5), dtype=bool)
    opening = np.array([[[9.0, 8.0, 7.0, 6.0, 10.0]]])
    flex = _five_evs([1.4, 2.5, 3.0, 12.0, 5.0])

    buckets = deferrable_power_per_world(connected, opening, flex)

    assert buckets.shape == (5, 1, 1)
    np.testing.assert_allclose(buckets[:, 0, 0], [4.0, 0.0, 8.0, 0.0, 4.0])
    # The buckets split flexible power exactly.
    flexible = flexibility_per_world(connected, opening, np.zeros((1, 1, 5)), flex)
    assert buckets.sum(axis=0)[0, 0] == flexible["flexible_power_kw"][0, 0] == 16.0


def test_negative_slack_and_unplugged_evs() -> None:
    # EV 0 cannot make target in time (slack -0.5 h): must run, first bucket.
    # EV 1 is below target but not plugged in for the whole slot: no power.
    connected = np.array([[[True, False, False, False, False]]])
    opening = np.full((1, 1, 5), 4.0)  # 6 kWh short: 1.5 h at 4 kW
    buckets = deferrable_power_per_world(connected, opening, _five_evs([1.0] * 5))
    np.testing.assert_allclose(buckets[:, 0, 0], [4.0, 0.0, 0.0, 0.0, 0.0])


def test_cumulative_and_total_rows_are_per_world_sums_not_sums_of_quantiles() -> None:
    # Three worlds, slot 0 only.  Buckets (kW): under_1h 0, 10, 20;
    # 2_to_4h 0, 0, 30; 8h_or_more 20, 10, 0.  Per world "at least 1 h" is
    # 20, 10, 30 (P10/P50/P90 12, 20, 28), not the sum of the bucket P90s
    # (24 + 18 = 42).  "At least 4 h" and "at least 8 h" are the 8-hour
    # bucket (2, 10, 18); the total is 20, 20, 50 (20, 20, 44).
    per_bucket = np.zeros((5, 3, 336))
    per_bucket[0, :, 0] = [0.0, 10.0, 20.0]
    per_bucket[2, :, 0] = [0.0, 0.0, 30.0]
    per_bucket[4, :, 0] = [20.0, 10.0, 0.0]
    bands = deferrable_power_bands(per_bucket, build_study_slots(date(2026, 1, 12)))

    assert list(DEFERRABLE_BUCKET_ORDER) == [
        "under_1h",
        "1_to_2h",
        "2_to_4h",
        "4_to_8h",
        "8h_or_more",
        "at_least_1h",
        "at_least_2h",
        "at_least_4h",
        "at_least_8h",
        "total",
    ]
    assert list(dict.fromkeys(bands["slack_bucket"])) == list(DEFERRABLE_BUCKET_ORDER)
    assert len(bands) == 10 * 336 and bands["path_id"].eq("normal").all()
    first = bands.loc[bands["slot_index"].eq(0)].set_index("slack_bucket")[["p10", "p50", "p90"]]
    assert first.loc["under_1h"].tolist() == pytest.approx([2, 10, 18])
    assert first.loc["at_least_1h"].tolist() == pytest.approx([12, 20, 28])
    assert first.loc["at_least_2h"].tolist() == pytest.approx([12, 20, 28])
    assert first.loc["at_least_4h"].tolist() == pytest.approx([2, 10, 18])
    assert first.loc["at_least_8h"].tolist() == pytest.approx([2, 10, 18])
    assert first.loc["total"].tolist() == pytest.approx([20, 20, 44])


# --------------------------------------------------------------------------
# Weekly flexibility figures (contract 3.6e)
# --------------------------------------------------------------------------


def test_flexibility_weekly_summary_by_hand() -> None:
    # Three weeks, fleet charger capacity 4,000 kW, so a quarter is 1,000 kW.
    # Bucket order: 0 under_1h, 1 1_to_2h, 2 2_to_4h, 3 4_to_8h, 4 8h_or_more.
    slots = build_study_slots(date(2026, 1, 12))
    half_hour = slots["local_half_hour"].to_numpy()
    per_bucket = np.zeros((5, 3, 336))
    connected = np.zeros((3, 336))
    # Peaks (8-hour bucket): week 0 1,200 kW at slot 1 and 900 at slot 2;
    # week 1 1,000 kW at slots 1 and 3 (a tie: the earlier, slot 1); week 2
    # 800 kW at slot 4.  Peaks 1,200, 1,000, 800; modal slot 1 (12:30, the
    # study starts at London noon, decision 0004 item 52) in
    # two weeks; half-hours at or above 1,000 kW: 1, 2, 0 -> 0.5, 1.0, 0 h.
    per_bucket[4, 0, [1, 2]] = [1200.0, 900.0]
    per_bucket[4, 1, [1, 3]] = 1000.0
    per_bucket[4, 2, 4] = 800.0
    # 18:00: 70 kW (1-2 h slack) with 20 EVs plugged in, every evening and
    # week: 70 kW and 1,000 x 70 x 7 / (20 x 7) = 3,500 W per plugged-in EV.
    per_bucket[1][:, half_hour == 36] = 70.0
    connected[:, half_hour == 36] = 20.0
    # 22:00: 14, 28, 42 kW (2-4 h slack); week 0 has nobody plugged in (no
    # W per EV, left out), weeks 1 and 2 have 7 EVs: 4,000 and 6,000 W.
    per_bucket[2][:, half_hour == 44] = np.array([[14.0], [28.0], [42.0]])
    connected[1:, half_hour == 44] = 7.0
    # 19:00: 50 kW with 1-2 h slack every evening (not "at least 2 h"), plus
    # 30 kW with 4-8 h slack on Monday only in week 0 (the first 19:00 slot):
    # the week means of "at least 2 h" are 30 / 7, 0, 0.
    per_bucket[1][:, half_hour == 38] = 50.0
    per_bucket[3, 0, np.flatnonzero(half_hour == 38)[0]] = 30.0
    # Movable energy is 10 x (week + 1) kWh at every 18:00 half-hour.
    movable = np.zeros((3, 336))
    movable[:, half_hour == 36] = np.array([[10.0], [20.0], [30.0]])

    row = flexibility_weekly_summary(per_bucket, movable, connected, slots, 4000.0).iloc[0]

    def band(name: str) -> list[float]:
        return row[[f"{name}_{q}" for q in ("p10", "p50", "p90")]].tolist()

    assert band("peak_deferrable_kw") == pytest.approx([840.0, 1000.0, 1160.0])
    assert row["peak_modal_slot_index"] == 1 and row["peak_modal_week_count"] == 2
    assert row["peak_modal_interval_start_london"] == pd.Timestamp(
        "2026-01-12 12:30", tz="Europe/London"
    )
    assert row["fleet_charger_capacity_kw"] == 4000.0
    assert band("hours_at_least_quarter_capacity") == pytest.approx([0.1, 0.5, 0.9])
    assert band("movable_energy_1800_kwh") == pytest.approx([12.0, 20.0, 28.0])
    assert band("deferrable_kw_1800") == pytest.approx([70.0, 70.0, 70.0])
    assert band("w_per_plugged_in_ev_1800") == pytest.approx([3500.0, 3500.0, 3500.0])
    assert band("deferrable_kw_2200") == pytest.approx([16.8, 28.0, 39.2])
    assert band("w_per_plugged_in_ev_2200") == pytest.approx([4200.0, 5000.0, 5800.0])
    assert band("deferrable_kw_0300") == pytest.approx([0.0, 0.0, 0.0])
    # Nobody plugged in at 03:00 in any week: missing, not zero.
    assert np.isnan(band("w_per_plugged_in_ev_0300")).all()
    assert band("deferrable_at_least_2h_kw_1900") == pytest.approx([0.0, 0.0, 0.8 * 30 / 7])


# --------------------------------------------------------------------------
# Plug-in and plug-out heatmaps by half-hour (contract 3.10b)
# --------------------------------------------------------------------------


def test_plug_event_heatmap_is_events_per_ev_day_by_hand() -> None:
    # Two EVs, two worlds, a 7-day study from Monday 12 January 2026 at
    # London noon.  Slot 13 is Monday 18:30.  World 0: both EVs; world 1: one
    # EV.  So the cell is 1.0 and 0.5 per EV-day: mean 0.75, P10 0.55, P90 0.95.
    slots = build_study_slots(date(2026, 1, 12))
    heatmap = plug_event_heatmap(
        np.array([0, 0, 1]), np.array([13, 13, 13]), 2, 2, slots
    ).set_index(["weekday", "local_half_hour"])

    assert len(heatmap) == 7 * 48 and heatmap["slot_count"].eq(1).all()
    cell = heatmap.loc[(0, 37)]
    assert cell["local_time_label"] == "18:30" and cell["weekday_label"] == "Mon"
    assert cell[["mean", "p10", "p50", "p90"]].tolist() == pytest.approx([0.75, 0.55, 0.75, 0.95])
    # A cell with study slots and no event is a real zero.
    assert heatmap.loc[(0, 36), "p90"] == 0.0


def test_plug_event_heatmap_counts_study_slots_per_cell() -> None:
    # Autumn change (Sun 25 Oct 2026): 01:00 and 01:30 London happen twice,
    # so Sunday's 01:00 cell has two study slots and an EV-day rate of
    # events / (EVs x 2).  A 2-day noon-to-noon study from Monday covers
    # Monday afternoon to Wednesday morning: every other cell is NaN.
    autumn = build_study_slots(date(2026, 10, 19))
    london = autumn["interval_start_london"]
    sunday_one = np.flatnonzero((london.dt.weekday == 6) & (autumn["local_half_hour"] == 2))
    assert len(sunday_one) == 2
    heatmap = plug_event_heatmap(np.array([0]), sunday_one[:1], 1, 1, autumn)
    cell = heatmap.loc[heatmap["weekday"].eq(6) & heatmap["local_half_hour"].eq(2)].iloc[0]
    assert cell["slot_count"] == 2 and cell["p50"] == pytest.approx(0.5)

    short = build_study_slots(date(2026, 1, 12), study_days=2)
    heatmap = plug_event_heatmap(np.array([0]), np.array([0]), 1, 1, short)
    covered = heatmap["slot_count"] > 0
    assert heatmap.loc[~covered, "p50"].isna().all()
    assert heatmap.loc[covered, "p50"].notna().all()
    monday = heatmap["weekday"].eq(0)
    assert covered[monday].tolist() == [False] * 24 + [True] * 24
    assert covered[heatmap["weekday"].eq(1)].all()
    wednesday = heatmap["weekday"].eq(2)
    assert covered[wednesday].tolist() == [True] * 24 + [False] * 24
    assert not covered[heatmap["weekday"] >= 3].any()


def test_real_run_plug_out_heatmap_peaks_in_the_weekday_morning() -> None:
    # Regression: plug-outs keep sessions already plugged in at the horizon
    # start, so the first morning (Monday) is not empty.
    result = run_forecast_from_assumptions(
        date(2026, 1, 12),
        model="no_action",
        values={"vehicle_count": 40, "evaluation_world_count": 3},
    )
    validate_result_v2(result)
    out = result.plug_out_heatmap
    weekdays = out.loc[out["weekday"] < 5]
    busiest = weekdays.loc[weekdays.groupby("weekday")["mean"].idxmax(), "local_half_hour"]
    # Departures are centred on the morning (unplug = departure, item 51).
    assert busiest.between(10, 22).all()  # 05:00-11:00 London
    monday_morning = out.loc[out["weekday"].eq(0) & out["local_half_hour"].between(10, 22), "mean"]
    assert monday_morning.sum() > 0.2


# --------------------------------------------------------------------------
# Relative day-ahead price and the cheapest half-hour (contract 4.7a, 4.7b)
# --------------------------------------------------------------------------


def _prices(slots: pd.DataFrame, by_world: list[np.ndarray]) -> pd.DataFrame:
    keys = slots.loc[:, ["slot_index", "interval_start_london"]]
    return pd.concat(
        [keys.assign(world_id=w, wholesale_forecast_gbp_per_mwh=p) for w, p in enumerate(by_world)],
        ignore_index=True,
    )


def test_price_relative_bands_take_out_each_days_level_by_hand() -> None:
    # Two worlds, 7 days.  Each day's price is its level (world 0: 50 + 10 x
    # day; world 1: 100) plus 20 at 18:00 and 0 elsewhere.  Relative to the
    # day's mean (level + 20/48), 18:00 is +20 - 20/48 and other half-hours
    # -20/48 in every world-day, whatever the level.
    slots = build_study_slots(date(2026, 1, 12))
    day = np.arange(336) // 48
    bump = np.where(slots["local_half_hour"].to_numpy() == 36, 20.0, 0.0)
    bands = price_relative_bands(_prices(slots, [50.0 + 10.0 * day + bump, 100.0 + bump]))

    assert len(bands) == 8 * 48 and bands["weekday"].isna().sum() == 48
    all_days = bands.loc[bands["weekday"].isna()].set_index("local_half_hour")
    assert all_days.loc[36, "sample_count"] == 14  # 2 worlds x 7 dates
    assert all_days.loc[36, ["p10", "p50", "p90"]].tolist() == pytest.approx([20 - 20 / 48] * 3)
    assert all_days.loc[3, "p50"] == pytest.approx(-20 / 48)
    tuesday = bands.loc[bands["weekday"].eq(1) & bands["local_half_hour"].eq(36)].iloc[0]
    assert tuesday["sample_count"] == 2 and tuesday["weekday_label"] == "Tue"


def test_partial_london_days_are_left_out() -> None:
    # Slots from Monday 12:00 London: Monday and the last Monday are half
    # days, so only the six whole dates in between count.
    slots = build_study_slots(date(2026, 1, 12), study_days=8).iloc[24 : 24 + 336]
    slots = slots.reset_index(drop=True)
    prices = _prices(slots, [np.arange(336, dtype=float) % 48])
    bands = price_relative_bands(prices)
    all_days = bands.loc[bands["weekday"].isna()]
    assert all_days["sample_count"].eq(6).all()
    summary = cheapest_half_hour_summary(prices).iloc[0]
    assert summary["world_day_count"] == 6


def test_cheapest_half_hour_summary_by_hand() -> None:
    # Three worlds x 7 dates.  World 0's cheapest half-hour is 03:00 every
    # day, world 2's 04:00, world 1's 05:30 except a tie on Monday between
    # 04:00 and 06:00, which goes to the earlier.  21 world-dates, sorted:
    # seven at 3.0, eight at 4.0, six at 5.5.  Nearest quantiles sit at
    # positions 2, 10 and 18: P10 3.0, P50 4.0, P90 5.5.
    slots = build_study_slots(date(2026, 1, 12))
    half_hour = slots["local_half_hour"].to_numpy()
    world_0 = np.where(half_hour == 6, 10.0, 50.0)
    world_1 = np.where(half_hour == 11, 10.0, 50.0)
    world_1[:48] = np.where(np.isin(half_hour[:48], (8, 12)), 5.0, 50.0)
    world_2 = np.where(half_hour == 8, 10.0, 50.0)
    row = cheapest_half_hour_summary(_prices(slots, [world_0, world_1, world_2])).iloc[0]

    assert row["world_day_count"] == 21
    assert [row[f"local_hour_{q}"] for q in ("p10", "p50", "p90")] == [3.0, 4.0, 5.5]
    assert [row[f"local_time_label_{q}"] for q in ("p10", "p50", "p90")] == [
        "03:00",
        "04:00",
        "05:30",
    ]


# --------------------------------------------------------------------------
# The validator on real runs and on tampered frames
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def action_run():
    return run_forecast_from_assumptions(
        date(2026, 1, 12), model="action", values={"vehicle_count": 30, "evaluation_world_count": 4}
    )


def test_real_action_run_passes_validator_with_new_frames(action_run) -> None:
    validate_result_v2(action_run)
    peaks = action_run.weekly_peak_summary.set_index("path_id")
    # Smart charging bunches plugged-in EVs onto the same cheap half-hours,
    # so a larger share of the plugged-in charger capacity draws at once.
    assert (
        peaks.loc["selected", "coincidence_factor_p50"]
        > peaks.loc["normal", "coincidence_factor_p50"]
    )


def test_validator_rejects_a_deferrable_total_built_from_bucket_quantiles() -> None:
    result = make_result("action")
    bands = result.deferrable_power_bands.copy()
    buckets = bands.loc[bands["slack_bucket"].ne("total")].groupby("slot_index")["p90"].sum()
    total = bands["slack_bucket"].eq("total")
    bands.loc[total, "p90"] = bands.loc[total, "slot_index"].map(buckets).to_numpy() + 1.0
    with pytest.raises(AssertionError, match="deferrable"):
        validate_result_v2(replace(result, deferrable_power_bands=bands))


def test_validator_rejects_a_coincidence_factor_from_median_bands() -> None:
    result = make_result("action")
    peaks = result.weekly_peak_summary.copy()
    # The toy's factor is 1 in every week; a factor of the median bands
    # (median peak / median plugged-in capacity) would differ.
    peaks["coincidence_factor_p10"] = 0.5
    peaks["coincidence_factor_p50"] = 0.5
    with pytest.raises(AssertionError, match="weekly_peak_summary"):
        validate_result_v2(replace(result, weekly_peak_summary=peaks))


def test_validator_rejects_hourly_values_in_the_half_hour_heatmap() -> None:
    result = make_result("no_action")
    heatmap = result.plug_in_half_hour_heatmap.copy()
    # Doubling every rate (an hourly count on half-hour cells) breaks the
    # exact recomputation from the sampled events.
    heatmap[["mean", "p10", "p50", "p90"]] *= 2.0
    with pytest.raises(AssertionError, match="plug_in_half_hour_heatmap"):
        validate_result_v2(replace(result, plug_in_half_hour_heatmap=heatmap))


def test_validator_rejects_plug_outs_that_miss_closed_sessions() -> None:
    result = make_result("no_action")
    heatmap = result.plug_out_heatmap.copy()
    heatmap[["mean", "p10", "p50", "p90"]] *= 0.5
    with pytest.raises(AssertionError, match="plug_out_heatmap"):
        validate_result_v2(replace(result, plug_out_heatmap=heatmap))


def test_validator_rejects_absolute_prices_as_relative() -> None:
    result = make_result("action")
    bands = result.price_relative_bands.copy()
    bands["p50"] += 90.0
    bands["p90"] += 90.0
    with pytest.raises(AssertionError, match="price_relative_bands"):
        validate_result_v2(replace(result, price_relative_bands=bands))


def test_no_action_result_has_no_price_summaries() -> None:
    result = make_result("no_action")
    assert result.price_relative_bands is None and result.cheapest_half_hour_summary is None
    assert result.deferrable_power_bands is not None
    validate_result_v2(result)


def test_deferrable_buckets_rebuild_from_one_ev_replays() -> None:
    # Independent of the chunked pass: rebuild every EV's unmanaged week with
    # the One EV replay, bucket its whole-slot slack by hand and add the EVs
    # up.  With one world, P10 = P50 = P90 = that world's value.  Slot 0 has
    # no opening stock in the replay, so it is skipped.
    result = run_forecast_from_assumptions(
        date(2026, 1, 12), model="action", values={"vehicle_count": 12, "evaluation_world_count": 1}
    )
    state = result.replay_state
    flex = flexibility_inputs(
        state.settings,
        state.units,
        result.study_slots,
        result.action_summary.departure_margin_hours,
    )
    expected = {bucket: np.zeros(336) for bucket in DEFERRABLE_BUCKET_ORDER}
    for ev, unit_id in enumerate(state.units["unit_id"]):
        rows = replay_one_ev(result, unit_id, 0).intervals
        rows = rows.loc[rows["path_id"].eq("normal")].sort_values("slot_index")
        closing = rows["closing_battery_kwh"].to_numpy()
        connected = rows["connected"].to_numpy(dtype=bool)
        power = flex.power_kw[ev]
        for slot in range(1, 336):
            short = flex.target_kwh[ev] - closing[slot - 1]
            if not connected[slot] or short <= 0.0:
                continue
            half_hours = np.ceil(round(short / flex.efficiency / (power * 0.5), 9))
            slack = flex.hours_to_departure[slot, ev] - 0.5 * half_hours
            names = [bucket for bucket, upper in SLACK_BUCKETS if slack < upper][:1]
            names += [f"at_least_{h}h" for h in (1, 2, 4, 8) if slack >= h]
            for name in [*names, "total"]:
                expected[name][slot] += power

    assert expected["total"].max() > 0.0 and expected["at_least_2h"].max() > 0.0
    bands = result.deferrable_power_bands
    for bucket, values in expected.items():
        got = bands.loc[bands["slack_bucket"].eq(bucket)].sort_values("slot_index")
        np.testing.assert_allclose(got["p50"].to_numpy()[1:], values[1:], atol=1e-9, err_msg=bucket)
