"""Tests for ``model/summaries.py`` (contract v2, task M5).

Hand-checkable tiny inputs for each builder, then a tiny *real* run of both
explicit models assembled into a contract v2 result and checked with the
shared ``validate_result_v2``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fixtures.result_fixture import (
    make_result,
    validate_result_v2,
    validate_run_comparison_v2,
)

from axle_studio.model import assumptions
from axle_studio.model.forecast import run_forecast, run_forecast_from_assumptions
from axle_studio.model.settings import RunSettings
from axle_studio.model.summaries import (
    _GroupDay,
    action_summary,
    average_day_bands,
    build_study_slots,
    build_summaries,
    build_warmup_slots,
    compare_runs,
    cost_effect_summary,
    difference_bands,
    difference_weekly,
    difference_weekly_bands,
    fill_ev_bands,
    find_sessions,
    fleet_interval_ev_bands,
    forecast_price_bands,
    new_ev_bands,
    plug_in_events_frame,
    plug_in_summary,
    plug_in_world_kpis,
    price_band_shift,
    slim_run,
    smart_charging_summary,
    smart_charging_world,
    weekly_peak_summary,
)


def _default_inputs() -> dict:
    # Fresh run inputs from model/assumptions.py (the removed JSON configs'
    # values) on every call, so a test that edits them cannot leak into another.
    return assumptions.forecast_inputs(None, warmup_days=1, study_days=7)


def _weather(*, warmup_days: int, study_days: int):
    inputs = assumptions.forecast_inputs(None, warmup_days=warmup_days, study_days=study_days)
    return inputs["base_temperature_c_by_day"], inputs["weather_sd_c"]


_ROOT = Path(__file__).parents[2]

# --------------------------------------------------------------------------
# Tiny hand-made inputs
# --------------------------------------------------------------------------


def _toy_fleet(normal: list[list[float]], selected: list[list[float]]) -> pd.DataFrame:
    """Contract-shaped fleet frame where every metric equals the given values."""

    slots = build_study_slots(date(2026, 1, 12)).iloc[: len(normal[0])]
    frames = []
    for path_id, values in (("normal", normal), ("selected", selected)):
        for world_id, row in enumerate(values):
            frame = slots.loc[
                :, ["slot_index", "interval_start_utc", "interval_end_utc", "interval_start_london"]
            ].copy()
            frame.insert(0, "path_id", path_id)
            frame.insert(0, "world_id", np.int64(world_id))
            for metric in (
                "connected_count",
                "connected_share",
                "home_import_kwh",
                "home_import_kw",
                "public_import_kwh",
                "total_import_kw",
                "closing_battery_kwh",
                "battery_soc_percent",
                "driving_share",
                "away_share",
                "public_charging_share",
                "unserved_travel_kwh",
            ):
                frame[metric] = np.asarray(row, dtype=float)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_difference_bands_are_per_world_differences_not_band_subtraction() -> None:
    # Three worlds, one slot: normal 1, 2, 3; selected 3, 1, 2.
    fleet = _toy_fleet([[1.0], [2.0], [3.0]], [[3.0], [1.0], [2.0]])
    bands = difference_bands(fleet)
    row = bands.loc[bands["metric"].eq("home_import_kwh")].iloc[0]
    # Per-world differences are +2, -1, -1: median -1, mean 0.
    assert row["p50"] == pytest.approx(-1.0)
    assert row["mean"] == pytest.approx(0.0)
    # Subtracting the bands would give median 2 - 2 = 0: a different answer.
    assert row["p50"] != pytest.approx(np.median([3, 1, 2]) - np.median([1, 2, 3]))
    soc = bands.loc[bands["metric"].eq("battery_soc_percent")]
    assert (soc["unit"] == "percentage points").all()


def test_difference_weekly_sums_flows_and_takes_last_closing_stock() -> None:
    fleet = _toy_fleet([[1.0, 2.0], [0.0, 1.0]], [[2.0, 4.0], [0.0, 0.5]])
    weekly = difference_weekly(fleet)
    imports = weekly.loc[weekly["metric"].eq("home_import_kwh")]
    assert imports["selected_minus_normal"].tolist() == pytest.approx([3.0, -0.5])
    closing = weekly.loc[weekly["metric"].eq("closing_battery_kwh")]
    # A stock: the last slot's difference (4 - 2, 0.5 - 1), never a sum.
    assert closing["selected_minus_normal"].tolist() == pytest.approx([2.0, -0.5])
    bands = difference_weekly_bands(weekly)
    row = bands.loc[bands["metric"].eq("home_import_kwh")].iloc[0]
    assert row["world_count"] == 2
    assert row["mean"] == pytest.approx(1.25)
    assert row["p50"] == pytest.approx(1.25)


def test_cost_effect_summary_takes_each_component_separately() -> None:
    cost = pd.DataFrame(
        {
            "world_id": [0, 1, 2],
            "illustrative_selected_minus_normal_energy_cost_gbp": [-3.0, -1.0, -2.0],
            "illustrative_selected_minus_normal_public_charge_cost_gbp": [0.0, 4.0, 0.0],
            "illustrative_unrecovered_energy_value_gbp": [0.0, 0.0, 1.0],
            "illustrative_unserved_travel_value_gbp": [0.0, 0.0, 0.0],
            "illustrative_selected_minus_normal_total_gbp": [-3.0, 3.0, -1.0],
            "evidence_kind": "illustrative_synthetic",
        }
    )
    summary = cost_effect_summary(cost).set_index("component")
    assert list(summary.index) == [
        "home_import_cost",
        "public_charge_cost",
        "unrecovered_energy_value",
        "unserved_travel_value",
        "total",
    ]
    assert summary.loc["home_import_cost", "p50"] == pytest.approx(-2.0)
    assert summary.loc["total", "p50"] == pytest.approx(-1.0)
    # Component medians (-2, 0, 0, 0) do not add up to the total's median.
    parts = summary.loc[summary.index != "total", "p50"].sum()
    assert parts != pytest.approx(summary.loc["total", "p50"])


def test_action_summary_describes_smart_charging() -> None:
    summary = action_summary(
        departure_margin_hours=1.5,
        forecast_available_at_utc=pd.Timestamp("2026-01-12", tz="UTC"),
        vehicle_count=100,
    )
    assert summary.action_id == "smart_charging_v1"
    assert "cheapest forecast half-hours" in summary.action_label
    assert (summary.departure_margin_hours, summary.vehicle_count) == (1.5, 100)


def _smart_toy(normal: list[list[float]], selected: list[list[float]]) -> pd.DataFrame:
    fleet = _toy_fleet(normal, selected)
    fleet["early_departure_count"] = np.where(fleet["path_id"].eq("selected"), 1, 0)
    fleet["early_departure_shortfall_kwh"] = np.where(fleet["path_id"].eq("selected"), 0.5, 0.0)
    return fleet


def _toy_prices(fleet: pd.DataFrame, values: list[list[float]]) -> pd.DataFrame:
    rows = fleet.loc[fleet["path_id"].eq("normal"), ["world_id", "slot_index"]]
    return rows.assign(wholesale_forecast_gbp_per_mwh=np.ravel(values)).reset_index(drop=True)


def test_smart_charging_world_prices_import_at_day_ahead_prices_by_hand() -> None:
    # Home import is valued at the day-ahead price (plan B3, decision 0004 item 53).
    # World 0: normal buys 2 kWh at 100 and 0 at 20; selected buys 0 and 2.
    # World 1: normal 1 and 1; selected 0 and 2, with prices 50 and 10.
    fleet = _smart_toy([[2.0, 0.0], [1.0, 1.0]], [[0.0, 2.0], [0.0, 2.0]])
    prices = _toy_prices(fleet, [[100.0, 20.0], [50.0, 10.0]])
    world = smart_charging_world(fleet, prices)

    assert world["normal_average_price_gbp_per_mwh"].tolist() == [100.0, 30.0]
    assert world["selected_average_price_gbp_per_mwh"].tolist() == [20.0, 10.0]
    # Moved = sum of max(0, normal - selected): 2 kWh and 1 kWh.
    assert world["moved_home_import_kwh"].tolist() == [2.0, 1.0]
    assert world["moved_home_import_share"].tolist() == [1.0, 0.5]
    assert world["early_departure_count"].tolist() == [2, 2]
    assert world["early_departure_shortfall_kwh"].tolist() == [1.0, 1.0]

    summary = smart_charging_summary(world).set_index("metric")
    # Per-world change first (-80, -20), then the quantiles (item 12).
    assert summary.loc["average_price_change_gbp_per_mwh", "mean"] == pytest.approx(-50.0)
    assert summary.loc["average_price_change_gbp_per_mwh", "p10"] == pytest.approx(-74.0)


def test_smart_charging_world_leaves_a_week_without_import_missing() -> None:
    fleet = _smart_toy([[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 2.0]])
    prices = _toy_prices(fleet, [[100.0, 20.0], [50.0, 10.0]])
    world = smart_charging_world(fleet, prices)
    assert np.isnan(world["normal_average_price_gbp_per_mwh"].iat[0])
    assert np.isnan(world["moved_home_import_share"].iat[0])
    summary = smart_charging_summary(world).set_index("metric")
    assert summary.loc["normal_average_price_gbp_per_mwh", "world_count"] == 1


def _world_forecast(slots: pd.DataFrame, prices_by_world: list[np.ndarray]) -> pd.DataFrame:
    """A day-ahead price frame with one row per (world, slot), world-major."""

    return pd.concat(
        [
            slots.loc[:, ["slot_index"]].assign(
                world_id=world, wholesale_forecast_gbp_per_mwh=price
            )
            for world, price in enumerate(prices_by_world)
        ],
        ignore_index=True,
    )


def test_price_band_shift_sums_moved_energy_per_date_and_forecast_third() -> None:
    slots = build_study_slots(date(2026, 1, 12))
    # A 48-slot daily day-ahead price, the same in both worlds: the first 16
    # half-hours of each day are cheapest, the next 16 middle, the last 16
    # dearest.
    daily = (slots["slot_index"] % 48 // 16).astype(float).to_numpy()
    forecast = _world_forecast(slots, [daily, daily])
    normal = np.zeros((2, 336))
    selected = np.zeros((2, 336))
    normal[:, 40] = 3.0  # Monday 20:00, dearest third
    selected[:, 2] = 3.0  # Monday 01:00, cheapest third
    normal[1, 60] = 1.0  # Tuesday 06:00 in world 1 only, cheapest third
    fleet = _toy_fleet(normal.tolist(), selected.tolist())
    bands = price_band_shift(fleet, forecast, slots)

    assert len(bands) == 7 * 3
    monday = bands.loc[bands["day_label"].eq("Mon 12")].set_index("price_band")
    assert monday["mean_slot_count"].tolist() == [16.0, 16.0, 16.0]
    assert monday.loc["high", "p50"] == 3.0  # moved out of the dearest third
    assert monday.loc["low", "p50"] == -3.0  # moved into the cheapest third
    tuesday = bands.loc[bands["day_label"].eq("Tue 13")].set_index("price_band")
    assert tuesday.loc["low", "mean"] == pytest.approx(0.5)  # worlds 0 and 1: 0 and 1


def test_price_band_shift_bands_each_world_by_its_own_day_ahead_prices() -> None:
    # Decision 0004 item 48: every world ranks its own day-ahead path, so the
    # same half-hour can be cheap in one world and dear in another.  World 1
    # has world 0's daily prices reversed; the same move (20:00 to 01:00)
    # goes from dear to cheap in world 0 and from cheap to dear in world 1.
    slots = build_study_slots(date(2026, 1, 12))
    daily = (slots["slot_index"] % 48 // 16).astype(float).to_numpy()
    forecast = _world_forecast(slots, [daily, 2.0 - daily])
    normal = np.zeros((2, 336))
    selected = np.zeros((2, 336))
    normal[:, 40] = 3.0
    selected[:, 2] = 3.0
    bands = price_band_shift(_toy_fleet(normal.tolist(), selected.tolist()), forecast, slots)

    monday = bands.loc[bands["day_label"].eq("Mon 12")].set_index("price_band")
    assert monday.loc["low", "mean"] == pytest.approx(0.0)  # -3 in world 0, +3 in world 1
    assert monday.loc["low", "p10"] == pytest.approx(-3.0 + 0.1 * 6.0)
    assert monday.loc["high", "mean"] == pytest.approx(0.0)
    assert monday["mean_slot_count"].sum() == pytest.approx(48.0)


def test_forecast_price_bands_are_linear_quantiles_across_worlds_per_slot() -> None:
    # Decision 0004 item 48: one day-ahead path per world; the Plan chart
    # needs each slot's spread across worlds.  Five worlds at 0..4 plus the
    # slot index: P10 = 0.4, P50 = 2, P90 = 3.6 above the slot index.
    slots = build_study_slots(date(2026, 1, 12))
    keys = slots.loc[
        :, ["slot_index", "interval_start_utc", "interval_end_utc", "interval_start_london"]
    ]
    base = slots["slot_index"].to_numpy(dtype=float)
    forecast = pd.concat(
        [keys.assign(world_id=w, wholesale_forecast_gbp_per_mwh=base + w) for w in (3, 0, 4, 1, 2)],
        ignore_index=True,
    )
    bands = forecast_price_bands(forecast)
    assert list(bands["slot_index"]) == list(range(336))
    assert bands["world_count"].eq(5).all()
    np.testing.assert_allclose(bands["p10"], base + 0.4)
    np.testing.assert_allclose(bands["p50"], base + 2.0)
    np.testing.assert_allclose(bands["p90"], base + 3.6)
    pd.testing.assert_frame_equal(bands[list(keys)], keys.reset_index(drop=True))


def test_weekly_peak_summary_takes_each_weeks_own_peak_by_hand() -> None:
    # Decision 0004 items 48-49.  Three weeks, four half-hours, fleet kW.
    # Normal peaks: 10, 20, 40 (slots 0, 0, 1).  Smart peaks: 30, 50, 60
    # (slots 2, 3, 2), so the modal smart peak is slot 2 (2 of 3 weeks).
    # Per-week ratios 3, 2.5 and 1.5, so the ratio's P10 is 1.7, not the
    # ratio of the P10 peaks (34 / 12 = 2.83).
    normal = [[10, 5, 5, 5], [20, 10, 0, 0], [0, 40, 40, 0]]
    selected = [[0, 0, 30, 0], [0, 0, 0, 50], [0, 0, 60, 60]]
    peaks = weekly_peak_summary(_toy_fleet(normal, selected), home_charging_power_kw=2.0).set_index(
        "path_id"
    )

    assert list(peaks.index) == ["normal", "selected"]
    # Linear P10/P50/P90 of (10, 20, 40): 12, 20, 36; of (30, 50, 60): 34, 50, 58.
    assert peaks.loc["normal", ["p10", "p50", "p90"]].tolist() == pytest.approx([12, 20, 36])
    assert peaks.loc["selected", ["p10", "p50", "p90"]].tolist() == pytest.approx([34, 50, 58])
    # Normal: slot 0 in two weeks; week 3's tie (slots 1 and 2) goes to slot 1.
    assert peaks.loc["normal", "modal_peak_slot_index"] == 0
    assert peaks.loc["normal", "modal_peak_week_count"] == 2
    # Smart: week 3 ties slots 2 and 3 and takes the earlier, so slot 2 twice.
    assert peaks.loc["selected", "modal_peak_slot_index"] == 2
    assert peaks.loc["selected", "modal_peak_week_count"] == 2
    # The study starts at London noon (decision 0004 item 52): slot 2 is 13:00.
    assert peaks.loc["selected", "modal_peak_interval_start_london"] == pd.Timestamp(
        "2026-01-12 13:00", tz="Europe/London"
    )
    ratio = ["ratio_to_normal_p10", "ratio_to_normal_p50", "ratio_to_normal_p90"]
    assert peaks.loc["selected", ratio].tolist() == pytest.approx([1.7, 2.5, 2.9])
    assert peaks.loc["normal", ratio].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert peaks["ratio_world_count"].tolist() == [3, 3]
    # Coincidence factor (decision 0004 item 54): the toy's connected_count
    # equals the normal values, at 2 kW per EV.  Normal: every peak is
    # 10 / (10 x 2) = 0.5.  Smart: week 1 peaks at slot 2 with 5 plugged in,
    # 30 / 10 = 3; week 2 at slot 3 with none plugged in, left out; week 3 at
    # slot 2 with 40 plugged in, 60 / 80 = 0.75.  (A toy: the kernel keeps
    # the factor at or below 1.)
    factor = ["coincidence_factor_p10", "coincidence_factor_p50", "coincidence_factor_p90"]
    assert peaks.loc["normal", factor].tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert peaks.loc["selected", factor].tolist() == pytest.approx([0.975, 1.875, 2.775])
    assert peaks["coincidence_world_count"].tolist() == [3, 2]


def test_weekly_peak_summary_leaves_out_a_week_with_no_normal_import() -> None:
    peaks = weekly_peak_summary(
        _toy_fleet([[0, 0], [4, 2]], [[3, 0], [8, 0]]), home_charging_power_kw=7.0
    )
    selected = peaks.set_index("path_id").loc["selected"]
    assert selected["ratio_world_count"] == 1
    assert selected["ratio_to_normal_p50"] == pytest.approx(2.0)
    assert selected["world_count"] == 2


def test_validator_rejects_a_weekly_peak_from_the_median_band() -> None:
    # Goal review N3: the peak of the per-slot median difference is not the
    # per-week peak; the validator's recomputation must catch a swap.
    result = make_result("action")
    peaks = result.weekly_peak_summary.copy()
    bands = result.fleet_interval_bands
    median = bands.loc[bands["metric"].eq("home_import_kw")].groupby("path_id")["p50"].max()
    peaks["p50"] = peaks["path_id"].map(median) - 1.0
    with pytest.raises(AssertionError, match="weekly_peak_summary"):
        validate_result_v2(replace(result, weekly_peak_summary=peaks))


def test_study_slots_across_the_autumn_clock_change() -> None:
    slots = build_study_slots(date(2026, 10, 21))  # clocks go back Sun 25 Oct
    per_night = slots.groupby("local_date").size()
    # Sunday 01:00 repeats inside Saturday's session night (decision 0004 item 52).
    assert per_night[date(2026, 10, 24)] == 50
    # 336 fixed UTC slots end at 11:00 on the last morning: the last night has 46.
    assert per_night[date(2026, 10, 27)] == 46
    repeated = slots.loc[slots["local_date"].eq(date(2026, 10, 24))]
    assert (repeated["local_half_hour"] == 2).sum() == 2  # 01:00 happens twice
    assert slots["night_index"].tolist() == [
        (value - date(2026, 10, 21)).days for value in slots["local_date"]
    ]


def test_study_slots_after_spring_change_count_to_the_last_date() -> None:
    slots = build_study_slots(date(2027, 3, 22))  # clocks go forward Sun 28 Mar
    assert slots["local_date"].nunique() == 7
    per_night = slots.groupby("local_date").size()
    assert per_night[date(2027, 3, 27)] == 46  # Saturday night skips 01:00
    # The last two UTC slots pass Monday noon and count to the last night.
    assert per_night[date(2027, 3, 28)] == 50
    assert slots["local_date"].iat[-1] == date(2027, 3, 28)
    assert slots["night_index"].iat[-1] == 6


def test_study_slots_are_noon_to_noon_session_nights() -> None:
    # Decision 0004 item 52: reporting days are session nights (London time
    # minus 12 h); day type follows the night, so Friday night's small hours
    # are a weekday and Sunday night's are a weekend.
    slots = build_study_slots(date(2026, 1, 12))  # a Monday
    london = slots["interval_start_london"]
    assert london.iat[0] == pd.Timestamp("2026-01-12 12:00", tz="Europe/London")
    assert slots["night_index"].tolist() == [n for n in range(7) for _ in range(48)]
    friday_small_hours = slots.loc[london.dt.strftime("%a %H:%M").eq("Sat 03:00")]
    assert friday_small_hours["day_label"].tolist() == ["Fri 16"]
    assert friday_small_hours["day_type"].tolist() == ["weekday"]
    sunday_night = slots.loc[london.dt.strftime("%a %H:%M").eq("Mon 03:00")]
    assert sunday_night["day_type"].tolist() == ["weekend"]


def test_warmup_slots_share_the_study_keys_and_count_back() -> None:
    warmup = build_warmup_slots(date(2026, 1, 12), 7)
    study = build_study_slots(date(2026, 1, 12))
    assert list(warmup.columns) == list(study.columns)
    assert warmup["slot_index"].tolist() == list(range(-336, 0))
    assert warmup["night_index"].tolist() == [n for n in range(-7, 0) for _ in range(48)]
    assert warmup["interval_end_utc"].iat[-1] == study["interval_start_utc"].iat[0]
    # Seven warm-up nights hold five working and two non-working nights.
    nights = warmup.drop_duplicates("night_index")
    assert nights["day_type"].value_counts().to_dict() == {"weekday": 5, "weekend": 2}


def test_find_sessions_by_hand() -> None:
    # One world, one EV, eight slots: plugged in slots 0-1 (already at the
    # start, so excluded), 3-4 (closed) and 7 (open at the horizon end).
    connected = np.array([1, 1, 0, 1, 1, 0, 0, 1], dtype=bool)[None, :, None]
    closing = np.array([30, 31, 20, 25, 30, 30, 18, 22], dtype=float)[None, :, None]
    imports = np.array([1, 1, 0, 5, 5, 0, 0, 4], dtype=float)[None, :, None]
    sessions = find_sessions(connected, closing, imports)
    assert sessions.first_slot.tolist() == [3, 7]
    assert sessions.end_slot.tolist() == [5, 8]
    # Plug-in stock is the closing stock of the slot before the session.
    assert sessions.plug_in_kwh.tolist() == [20.0, 18.0]
    assert sessions.home_import_kwh.tolist() == [10.0, 4.0]

    slots = build_study_slots(date(2026, 1, 12)).iloc[:8].reset_index(drop=True)
    events = plug_in_events_frame(
        sessions,
        study_slots=slots,
        world_ids=np.array([4]),
        unit_ids=np.array(["ev-001"], dtype=object),
        cohort_ids=np.array(["average_uk"], dtype=object),
        capacity_kwh=np.array([40.0]),
        home_charge_efficiency=0.9,
        path_id="normal",
    )
    assert events["event_index"].tolist() == [0, 1]
    assert events["plug_in_soc_percent"].tolist() == [50.0, 45.0]
    assert events["battery_added_kwh"].tolist() == pytest.approx([9.0, 3.6])
    closed, still_open = events.iloc[0], events.iloc[1]
    assert closed["plug_out_utc"] == slots["interval_end_utc"].iat[4]
    assert closed["plug_out_soc_percent"] == 75.0
    assert still_open["still_plugged_at_horizon_end"]
    assert pd.isna(still_open["plug_out_utc"]) and np.isnan(still_open["plug_out_soc_percent"])


def _statistics_rows() -> pd.DataFrame:
    # World 0: three weekday plug-ins; world 1: one weekend plug-in.
    return pd.DataFrame(
        {
            "world_id": [0, 0, 0, 1],
            "cohort_code": [0, 0, 1, 1],
            "weekend": [False, False, False, True],
            "local_hour": [18, 18, 7, 12],
            "soc_percent": [5.0, 40.0, 62.0, 30.0],
        }
    )


# Fleet of two EVs: EV 0 in cohort "a" (code 0), EV 1 in cohort "b" (code 1).
_GROUPS = {"fleet": (None, 2), "a": (0, 1), "b": (1, 1)}


def test_plug_in_world_kpis_cover_every_world_with_nan_for_missing() -> None:
    kpis = plug_in_world_kpis(_statistics_rows(), 3, _GROUPS)
    value = kpis.set_index(["world_id", "group_id", "day_type", "metric"])["value"]
    assert value[(0, "fleet", "weekday", "plug_ins_per_week")] == 3.0
    assert value[(0, "fleet", "weekday", "plug_ins_per_ev_per_week")] == 1.5
    assert value[(0, "fleet", "weekday", "median_plug_in_soc_percent")] == 40.0
    assert value[(0, "fleet", "weekday", "share_below_10_percent_soc")] == pytest.approx(1 / 3)
    assert value[(0, "fleet", "weekday", "modal_plug_in_local_hour")] == 18.0
    # Cohort b in world 0: one plug-in at 07:00 by its single EV.
    assert value[(0, "b", "all", "plug_ins_per_ev_per_week")] == 1.0
    assert value[(0, "b", "all", "modal_plug_in_local_hour")] == 7.0
    # World 2 has no plug-ins: a real zero count, but no median or hour (NaN).
    assert value[(2, "fleet", "all", "plug_ins_per_week")] == 0.0
    assert np.isnan(value[(2, "fleet", "all", "median_plug_in_soc_percent")])
    assert np.isnan(value[(2, "fleet", "all", "modal_plug_in_local_hour")])
    assert len(kpis) == 3 * 3 * 3 * 5


def test_plug_in_summary_shares_are_per_world_then_across_worlds() -> None:
    rows = _statistics_rows()
    summary = plug_in_summary(rows, plug_in_world_kpis(rows, 3, _GROUPS), world_count=3)
    hours = summary.hour_bands.set_index(["day_type", "local_hour"])
    # Only world 0 has weekday plug-ins; 2 of its 3 start at 18:00.
    assert hours.loc[("weekday", 18), "world_count"] == 1
    assert hours.loc[("weekday", 18), "share_p50"] == pytest.approx(2 / 3)
    # "all": world 0 share 2/3 at 18:00, world 1 share 0; world 2 has none.
    assert hours.loc[("all", 18), "world_count"] == 2
    assert hours.loc[("all", 18), "share_mean"] == pytest.approx(1 / 3)
    for day_type in ("weekday", "weekend", "all"):
        assert hours.loc[day_type, "share_mean"].sum() == pytest.approx(1.0)
    bins = summary.soc_bands.set_index(["day_type", "soc_bin_lower_percent"])
    assert bins.loc[("weekend", 30), "share_p50"] == 1.0
    kpis = summary.kpis.set_index(["day_type", "metric"])
    # Plug-ins per week counts every world, including the empty one: 0, 0, 3.
    assert kpis.loc[("weekday", "plug_ins_per_week"), "p50"] == 0.0
    assert kpis.loc[("all", "plug_ins_per_week"), "p90"] == pytest.approx(2.6)


def test_average_day_averages_slots_per_half_hour_then_quantiles() -> None:
    slots = build_study_slots(date(2026, 1, 12))
    worlds = 3
    # Connected share = world index / 2 everywhere; SoC mean 50, P5 10, P95 90.
    share = np.repeat(np.array([0.0, 0.5, 1.0])[:, None], len(slots), axis=1)
    stats = _GroupDay(
        connected_share=share,
        soc_mean=np.full((worlds, len(slots)), 50.0),
        soc_p5=np.full((worlds, len(slots)), 10.0),
        soc_p95=np.full((worlds, len(slots)), 90.0),
    )
    frame = average_day_bands({("fleet", "normal"): stats}, slots)
    assert len(frame) == 3 * 2 * 48
    connected = frame.loc[frame["metric"].eq("connected_share") & frame["day_type"].eq("all")]
    assert connected["centre"].tolist() == pytest.approx([0.5] * 48)
    assert connected["low"].tolist() == pytest.approx([0.1] * 48)
    soc = frame.loc[frame["metric"].eq("battery_soc_percent")]
    assert set(soc["spread"]) == {"across_evs"}
    assert np.allclose(soc[["centre", "low", "high"]].to_numpy(), [50.0, 10.0, 90.0])
    assert frame.loc[frame["day_type"].eq("weekend"), "day_count"].iat[0] == 2


def test_average_day_spring_half_hour_missing_on_its_only_date_is_nan() -> None:
    slots = build_study_slots(date(2027, 3, 22))
    worlds = 2
    stats = _GroupDay(*(np.ones((worlds, len(slots))) for _ in range(4)))
    frame = average_day_bands({("fleet", "normal"): stats}, slots)
    # 01:00-01:30 London does not exist on Sun 28 Mar; Sat 27 still has it.
    weekend = frame.loc[frame["day_type"].eq("weekend") & frame["local_half_hour"].eq(2)]
    assert weekend["centre"].notna().all()
    assert frame.loc[frame["day_type"].eq("all"), "centre"].notna().all()


# --------------------------------------------------------------------------
# Tiny real runs through run_forecast
# --------------------------------------------------------------------------


def _settings(start: date, vehicles: int, worlds: int, seed: int) -> RunSettings:
    return RunSettings(
        start_local_date=start,
        warmup_days=1,
        study_days=7,
        vehicle_count=vehicles,
        seed=seed,
        evaluation_world_count=worlds,
        opening_soc_fraction=0.55,
        reserve_soc_fraction=0.2,
        home_charge_efficiency=0.92,
        weather_efficiency_sensitivity_fraction_per_c=0.01,
    )


def real_result(
    model: str,
    *,
    start: date = date(2026, 1, 12),
    vehicles: int = 12,
    worlds: int = 3,
    seed: int = 73,
):
    """A tiny real ``run_forecast`` result with every contract v2 field.

    These runs take hand-made public and price inputs, which no assumption
    record describes, so ``run_forecast`` leaves ``assumptions`` as None and
    the synthetic fixture's records stand in.  A run built wholly from
    ``model/assumptions.py`` carries its own records (tested below).
    """

    settings = _settings(start, vehicles, worlds, seed)
    fixture = assumptions.cohort_fixture()
    temperature, weather_sd = _weather(warmup_days=1, study_days=7)
    # Both models top up in public (decision 0004 item 32); default threshold
    # re-baselined from 20 % to 10 % by item 37.
    model_inputs: dict[str, object] = {
        "public_charge_assumptions": {
            "efficiency_fraction": 0.9,
            "top_up_threshold_soc_fraction": 0.1,
            "top_up_target_soc_fraction": 0.8,
        },
    }
    if model == "action":
        model_inputs |= {
            "price_assumptions": _default_inputs()["price_assumptions"],
            "departure_margin_hours": 1.0,
        }
    result = run_forecast(
        settings,
        fixture,
        daily_miles_cv_by_cohort=_default_inputs()["daily_miles_cv_by_cohort"],
        personal_mileage_cv_by_cohort=_default_inputs()["personal_mileage_cv_by_cohort"],
        trip_behaviour=_default_inputs()["trip_behaviour"],
        base_temperature_c_by_day=temperature,
        weather_sd_c=weather_sd,
        model=model,
        **model_inputs,
    )
    return replace(result, assumptions=make_result(model).assumptions)


@pytest.fixture(scope="module")
def action_result():
    return real_result("action")


@pytest.fixture(scope="module")
def no_action_result():
    return real_result("no_action")


@pytest.mark.parametrize("name", ["action_result", "no_action_result"])
def test_real_run_passes_contract_validator(name: str, request) -> None:
    result = request.getfixturevalue(name)
    validate_result_v2(result)


@pytest.mark.parametrize("model", ["action", "no_action"])
def test_run_from_assumptions_carries_its_own_records_and_passes_validator(model: str) -> None:
    values = {"vehicle_count": 10, "evaluation_world_count": 2, "public_charge_gbp_per_kwh": 0.5}
    result = run_forecast_from_assumptions(date(2026, 1, 12), model=model, values=values)

    validate_result_v2(result)
    assert result.assumptions == assumptions.result_assumptions(values)
    records = {record.name: record.value for record in result.assumptions}
    assert records["public_charge_gbp_per_kwh"] == 0.5
    assert records["cnz_median_plug_in_soc_percent"] == 52.0


@pytest.mark.parametrize(
    "start",
    [date(2026, 10, 21), date(2027, 3, 22)],
    ids=["autumn", "spring"],
)
def test_clock_change_week_real_action_run_passes_validator(start: date) -> None:
    result = real_result("action", start=start, vehicles=8, worlds=2)
    assert result.has_clock_change
    validate_result_v2(result)


def test_plug_in_events_are_a_labelled_sample_led_by_the_representative_world() -> None:
    result = real_result("no_action", vehicles=6, worlds=13, seed=5)
    worlds = result.sampled_world_ids
    assert len(worlds) == 10 and worlds[0] == result.representative_world_id
    assert list(dict.fromkeys(result.plug_in_events["world_id"])) == [
        w for w in worlds if w in set(result.plug_in_events["world_id"])
    ]
    # Statistics still cover all 13 worlds.
    assert result.plug_in_world_kpis["world_id"].nunique() == 13


def test_plug_in_kpis_match_the_sampled_events(no_action_result) -> None:
    # With 3 worlds every world is in the sample, so the per-world KPIs
    # can be recomputed from the event rows.
    events = no_action_result.plug_in_events
    kpis = no_action_result.plug_in_world_kpis
    for world_id, group in events.groupby("world_id"):
        chosen = kpis.loc[
            kpis["world_id"].eq(world_id)
            & kpis["group_id"].eq("fleet")
            & kpis["day_type"].eq("all")
            & kpis["metric"].eq("median_plug_in_soc_percent"),
            "value",
        ].iat[0]
        assert chosen == pytest.approx(group["plug_in_soc_percent"].median())


def test_average_day_fleet_connected_share_matches_fleet_frame(no_action_result) -> None:
    result = no_action_result
    fleet = result.fleet_world_intervals.merge(
        result.study_slots[["slot_index", "local_half_hour"]], on="slot_index"
    )
    per_world = fleet.groupby(["world_id", "local_half_hour"])["connected_share"].mean()
    expected = per_world.groupby("local_half_hour").median()
    bands = result.average_day_bands
    got = bands.loc[
        bands["group_id"].eq("fleet")
        & bands["day_type"].eq("all")
        & bands["metric"].eq("connected_share")
    ].set_index("local_half_hour")["centre"]
    assert np.allclose(got.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_compare_runs_pairs_and_flags_mismatches(action_result, no_action_result) -> None:
    same = compare_runs(action_result, action_result)
    validate_run_comparison_v2(same)
    assert same.matched_futures and same.illustrative_total_available
    assert np.allclose(same.difference_bands[["mean", "p10", "p50", "p90"]], 0.0)
    mixed = compare_runs(no_action_result, action_result)
    validate_run_comparison_v2(mixed)
    # Both models draw the same worlds for one seed (decision 0004 item 38).
    assert mixed.matched_futures
    other_seed = compare_runs(no_action_result, real_result("action", seed=8))
    assert not other_seed.matched_futures
    assert (mixed.path_id_a, mixed.path_id_b) == ("normal", "selected")
    assert not mixed.illustrative_total_available
    other_horizon = SimpleNamespace(**{**vars(action_result), "world_count": 4})
    with pytest.raises(ValueError, match="world_count"):
        compare_runs(action_result, other_horizon)


def _rebuild(result, state, fleet_world_intervals) -> dict[str, object]:
    return build_summaries(
        state,
        fleet_world_intervals,
        model=result.model,
        study_slots=result.study_slots,
        representative_world_id=result.representative_world_id,
        cohort_catalogue=[
            (row.cohort_id, row.cohort_label, row.source_population_share)
            for row in result.cohort_summary.itertuples()
        ],
        cost_effect=result.cost_effect,
        forecast_prices=result.forecast_prices,
        evaluation_prices=result.evaluation_prices,
        departure_margin_hours=(
            result.action_summary.departure_margin_hours if result.action_summary else None
        ),
    )


@pytest.mark.parametrize(
    "column",
    ["connected_count", "home_import_kwh", "public_import_kwh", "public_battery_added_kwh"],
)
def test_reconciliation_guard_rejects_a_fleet_frame_from_another_run(
    column: str, no_action_result
) -> None:
    result = no_action_result
    tampered = result.fleet_world_intervals.copy()
    tampered.loc[0, column] += 1
    with pytest.raises(ValueError, match="reconcile"):
        _rebuild(result, replace(result.replay_state, replay_cache={}), tampered)


@pytest.mark.parametrize("name", ["action_result", "no_action_result"])
def test_reconciliation_guard_rejects_replay_inputs_with_other_top_up_levels(
    name: str, request
) -> None:
    # Home import and connection can still match when only the top-up
    # threshold and target differ, so public import must be checked too.
    result = request.getfixturevalue(name)
    assert result.fleet_world_intervals["public_import_kwh"].sum() > 0.0
    state = result.replay_state
    _rebuild(result, replace(state, replay_cache={}), result.fleet_world_intervals)
    public = replace(state.public_top_up, threshold_soc_fraction=0.6, target_soc_fraction=0.95)
    other = replace(state, public_top_up=public, replay_cache={})
    with pytest.raises(ValueError, match="reconcile"):
        _rebuild(result, other, result.fleet_world_intervals)


def test_compare_runs_works_from_slim_history_records(action_result, no_action_result) -> None:
    # Run history keeps slim records (decision 0004 item 35); Compare must
    # give the same answer from them as from the full results.
    full = compare_runs(no_action_result, action_result)
    slim = compare_runs(slim_run(no_action_result), slim_run(action_result))
    validate_run_comparison_v2(slim)
    assert slim.mismatch_reasons == full.mismatch_reasons
    for name in ("changed", "difference_bands", "weekly_differences", "kpis"):
        pd.testing.assert_frame_equal(getattr(slim, name), getattr(full, name))
    record = slim_run(action_result)
    assert not hasattr(record, "replay_state") and not hasattr(record, "average_day_bands")


def test_fleet_interval_ev_bands_by_hand() -> None:
    # Two weeks, three EVs, two slots.  Per week and slot: P10/P50/P90
    # across the EVs; then the median of each across the two weeks.
    slots = build_study_slots(date(2026, 1, 12)).iloc[:2]
    bands = new_ev_bands(["normal"], world_count=2, slot_count=2)
    soc = np.array(
        [[[10.0, 50.0, 90.0], [20.0, 20.0, 20.0]], [[30.0, 70.0, 110.0], [0.0, 40.0, 80.0]]]
    )
    home = np.zeros_like(soc)
    home[:, 0, 0] = 3.5  # EV 0 imports 3.5 kWh (7 kW) in slot 0 of both weeks
    fill_ev_bands(
        bands, "normal", np.array([0, 1]), {"soc_percent": soc, "home_grid_import_kwh": home}
    )
    frame = fleet_interval_ev_bands(bands, slots).set_index(["metric", "slot_index"])

    # Week 0 slot 0: P10 = 10 + 0.2 * 40 = 18, P50 = 50, P90 = 82.
    # Week 1 slot 0: 38, 70, 102.  Medians across the weeks: 28, 60, 92.
    assert frame.loc[("battery_soc_percent", 0), ["p10", "p50", "p90"]].tolist() == pytest.approx(
        [28.0, 60.0, 92.0]
    )
    # Slot 1: week 0 all 20; week 1 8, 40, 72 -> medians 14, 30, 46.
    assert frame.loc[("battery_soc_percent", 1), ["p10", "p50", "p90"]].tolist() == pytest.approx(
        [14.0, 30.0, 46.0]
    )
    # Power: one EV of three at 7 kW; sorted 0, 0, 7, so P90 sits 0.8 of the
    # way from 0 to 7 (linear method): 5.6 kW, and P10 = P50 = 0.
    assert frame.loc[("home_import_kw", 0), ["p10", "p50", "p90"]].tolist() == pytest.approx(
        [0.0, 0.0, 5.6]
    )
    assert frame["unit"].tolist() == ["percent", "percent", "kW", "kW"]


def test_compare_changes_treat_two_missing_values_as_the_same() -> None:
    # An unset (NaN) record such as a zone's headroom must not read as a change.
    from axle_studio.model.summaries import _same_value

    assert _same_value(float("nan"), float("nan"))
    assert not _same_value(float("nan"), 1.0)
    assert _same_value(7.0, 7.0)
    assert not _same_value(7.0, 3.6)
