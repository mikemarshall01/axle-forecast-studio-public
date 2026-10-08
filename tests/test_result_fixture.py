"""Contract v2 checks on the SYNTHETIC FIXTURE result (docs/contracts/results-v2.md).

The validators live in ``tests/fixtures/result_fixture.py`` so M5's real-run
test calls the same ``validate_result_v2``.  These tests prove the fixture
passes, that it has the shapes the views need (B1 empty archetypes, clock
changes, an 18:00 dip, flagged worlds), and that the validators reject
broken results rather than passing anything.
"""

from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fixtures.replay_contract import replay_events_from
from fixtures.result_fixture import (
    compare_runs,
    make_result,
    replay_one_ev,
    replay_one_ev_bands,
    validate_one_ev_replay_v2,
    validate_result_v2,
    validate_run_comparison_v2,
)


@pytest.fixture(scope="module")
def action():
    return make_result()


@pytest.fixture(scope="module")
def no_action():
    return make_result("no_action")


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"model": "no_action"},
        {"evs": 5},
        {"clock_change": "autumn"},
        {"clock_change": "spring"},
        {"model": "no_action", "clock_change": "autumn"},
        {"evs": 12, "worlds": 10, "seed": 7},
    ],
)
def test_fixture_satisfies_contract(kwargs):
    validate_result_v2(make_result(**kwargs))


def test_fixture_is_deterministic():
    first, second = make_result(), make_result()
    pd.testing.assert_frame_equal(first.fleet_interval_bands, second.fleet_interval_bands)
    pd.testing.assert_frame_equal(first.plug_in_events, second.plug_in_events)


def test_horizon_is_336_utc_slots_from_london_noon(action):
    # Decision 0004 item 52: noon to noon, reported by session night.
    slots = action.study_slots
    assert action.study_slot_count == len(slots) == 336
    assert slots["interval_start_london"].iat[0] == pd.Timestamp(
        "2026-09-28 12:00", tz="Europe/London"
    )
    assert str(slots["interval_start_utc"].dtype) == "datetime64[ns, UTC]"
    assert slots["local_time_label"].iat[12] == "18:00"
    assert slots["day_label"].iat[0] == "Mon 28"
    # Tuesday 06:00 is still Monday's night.
    assert slots["local_time_label"].iat[36] == "06:00"
    assert slots["day_label"].iat[36] == "Mon 28"
    assert slots["night_index"].tolist() == [n for n in range(7) for _ in range(48)]


def test_autumn_week_repeats_the_one_oclock_hour():
    result = make_result(clock_change="autumn")
    slots = result.study_slots
    # Sunday 01:00 repeats, inside Saturday's session night (item 52).
    night = slots.loc[slots["day_label"].eq("Sat 24")]
    assert result.has_clock_change
    assert len(night) == 50
    assert night["local_time_label"].tolist().count("01:00") == 2
    # The price bands cover every study half-hour of each night, including
    # both copies of the repeated hour on the change night.
    bands = result.price_band_shift
    assert bands.loc[bands["day_label"].eq("Sat 24"), "mean_slot_count"].sum() == pytest.approx(
        len(night)
    )


def test_spring_week_skips_one_oclock_and_keeps_average_day_complete():
    result = make_result(clock_change="spring")
    slots = result.study_slots
    on_sunday = slots.loc[slots["interval_start_london"].dt.date.eq(date(2027, 3, 28))]
    assert len(on_sunday) == 46
    assert "01:00" not in set(on_sunday["local_time_label"])
    # The last UTC slots pass Monday noon but count to the last night.
    assert slots["interval_start_london"].iat[-1].strftime("%a %H:%M") == "Mon 12:30"
    assert slots["local_date"].iat[-1] == date(2027, 3, 28)
    fleet = result.average_day_bands.loc[result.average_day_bands["group_id"].eq("fleet")]
    assert fleet.loc[fleet["day_type"].eq("all"), "centre"].notna().all()


def test_fewer_than_six_archetypes_is_explicit_not_zero():
    result = make_result(evs=5)
    summary = result.cohort_summary.set_index("cohort_id")
    assert summary.loc["always_plugged_in", "ev_count"] == 0
    assert np.isnan(summary.loc["always_plugged_in", "plug_ins_per_ev_week_p50"])
    assert pd.isna(summary.loc["always_plugged_in", "peak_plug_in_local_hour"])
    assert "always_plugged_in" not in set(result.cohort_interval_bands["cohort_id"])
    assert "always_plugged_in" not in set(result.average_day_bands["group_id"])


def test_no_action_has_one_path_and_no_action_fields(no_action):
    assert set(no_action.fleet_interval_bands["path_id"]) == {"normal"}
    assert no_action.difference_bands is None
    assert no_action.cost_effect is None
    assert no_action.action_status == "not_applicable"


def test_selected_path_moves_evening_charging_into_the_night(action):
    diff = action.difference_bands.loc[action.difference_bands["metric"].eq("home_import_kw")]
    labels = action.study_slots.set_index("slot_index")["local_time_label"]
    hour = diff["slot_index"].map(labels).str.slice(0, 2).astype(int)
    assert diff.loc[hour.between(18, 21), "mean"].sum() < 0
    assert diff.loc[hour.between(2, 5), "mean"].sum() > 0


def test_difference_bands_are_world_first_not_band_subtraction(action):
    bands = action.fleet_interval_bands
    metric = "closing_battery_kwh"
    kw = bands.loc[bands["metric"].eq(metric)].pivot(
        index="slot_index", columns="path_id", values="p50"
    )
    diff = action.difference_bands.loc[action.difference_bands["metric"].eq(metric)]
    subtracted = (kw["selected"] - kw["normal"]).to_numpy()
    assert np.abs(subtracted - diff["p50"].to_numpy()).max() > 1e-6


def test_some_worlds_are_flagged_and_priced(action):
    cost = action.cost_effect
    assert 0 < action.not_recovered_world_count < action.world_count
    flagged = cost.loc[cost["energy_not_recovered"]]
    priced = (
        flagged["illustrative_unrecovered_energy_value_gbp"]
        + flagged["illustrative_selected_minus_normal_public_charge_cost_gbp"]
    )
    assert (priced > 0).all()


def test_plug_ins_are_evening_with_realistic_soc(action):
    events = action.plug_in_events
    assert events["plug_in_local_hour"].between(17, 22).mean() > 0.8
    assert events["plug_in_soc_percent"].between(10, 90).all()
    assert events["still_plugged_at_horizon_end"].any()
    assert events.loc[events["still_plugged_at_horizon_end"], "plug_out_utc"].isna().all()


def test_smart_charging_outcomes_show_cheaper_charging_and_early_departures(action):
    summary = action.smart_charging_summary.set_index("metric")
    assert summary.loc["average_price_change_gbp_per_mwh", "p90"] < 0.0
    assert summary.loc["early_departure_count", "p50"] > 0
    bands = action.price_band_shift
    assert len(bands) == 7 * 3
    # Energy leaves the dearest third and arrives in the cheapest.
    assert bands.loc[bands["price_band"].eq("high"), "mean"].sum() > 0.0
    assert bands.loc[bands["price_band"].eq("low"), "mean"].sum() < 0.0


def test_cnz_context_records_are_source_and_read_only(action):
    records = {record.name: record for record in action.assumptions}
    assert records["cnz_median_plug_in_soc_percent"].value == 52.0
    assert records["cnz_median_plug_in_soc_percent"].evidence == "source"
    assert not records["cnz_median_plug_in_soc_percent"].editable


@pytest.mark.parametrize("model", ["action", "no_action"])
def test_one_ev_replay_satisfies_contract(model):
    result = make_result(model)
    for unit_id in result.units["unit_id"]:
        replay = replay_one_ev(result, unit_id)
        validate_one_ev_replay_v2(replay, result)
        assert replay.world_id == result.representative_world_id
    bands = replay_one_ev_bands(result, "ev-0000")
    assert set(bands["metric"]) == {"battery_soc_percent", "connected_share"}


def test_action_replay_carries_early_departures_and_audit(action):
    early = replay_one_ev(action, "ev-0001", world_id=0).intervals
    assert early.loc[early["path_id"].eq("selected"), "early_departure_shortfall_kwh"].gt(0).any()
    replay = replay_one_ev(action, "ev-0000", world_id=1)
    audit = replay.daily_audit
    assert set(audit["path_id"]) == {"normal", "selected"}
    assert audit["departure_london"].dropna().dt.strftime("%H:%M").eq("07:30").all()
    with pytest.raises(KeyError):
        replay_one_ev(action, "ev-9999")
    with pytest.raises(KeyError):
        replay_one_ev(action, "ev-0000", world_id=99)


def test_compare_matched_runs_reports_changed_assumption(action):
    faster = make_result(home_charger_kw=11.0)
    comparison = compare_runs(action, faster)
    validate_run_comparison_v2(comparison)
    assert comparison.matched_futures
    assert comparison.changed["name"].tolist() == ["home_charger_kw"]
    assert comparison.illustrative_total_available


def test_compare_pairs_the_models_but_hides_totals_for_no_action(action, no_action):
    # Both models draw the same futures (decision 0004 item 38 removed the
    # planning world), so a model change alone still pairs worlds.
    comparison = compare_runs(no_action, action)
    validate_run_comparison_v2(comparison)
    assert comparison.matched_futures
    assert not comparison.illustrative_total_available
    other_seed = compare_runs(action, make_result(seed=7))
    assert any("seed" in reason for reason in other_seed.mismatch_reasons)


def test_compare_refuses_unpairable_worlds(action):
    with pytest.raises(ValueError, match="world_count"):
        compare_runs(action, make_result(worlds=5))


def _broken(result, **changes):
    return dataclasses.replace(result, **changes)


def test_validator_rejects_swapped_quantiles(action):
    bands = action.fleet_interval_bands.copy()
    bands[["p10", "p90"]] = bands[["p90", "p10"]].to_numpy()
    with pytest.raises(AssertionError):
        validate_result_v2(_broken(action, fleet_interval_bands=bands))


def test_validator_rejects_band_subtraction_as_difference(action):
    bands = action.fleet_interval_bands
    diff = action.difference_bands.copy()
    for metric in diff["metric"].unique():
        rows = diff["metric"].eq(metric)
        by_path = bands.loc[bands["metric"].eq(metric)].pivot(
            index="slot_index", columns="path_id", values="p50"
        )
        diff.loc[rows, "p50"] = (by_path["selected"] - by_path["normal"]).to_numpy()
    with pytest.raises(AssertionError, match="world-first"):
        validate_result_v2(_broken(action, difference_bands=diff))


def test_validator_rejects_wrong_dtype_and_columns(action):
    slots = action.study_slots.assign(slot_index=action.study_slots["slot_index"].astype(float))
    with pytest.raises(AssertionError, match="int64"):
        validate_result_v2(_broken(action, study_slots=slots))
    cost = action.cost_effect.drop(columns="energy_not_recovered")
    with pytest.raises(AssertionError, match="columns"):
        validate_result_v2(_broken(action, cost_effect=cost))


def test_validator_rejects_fake_selected_path_on_no_action(no_action, action):
    with pytest.raises(AssertionError):
        validate_result_v2(_broken(no_action, fleet_world_intervals=action.fleet_world_intervals))
    with pytest.raises(AssertionError, match="None"):
        validate_result_v2(_broken(no_action, difference_bands=action.difference_bands))


def test_large_fleet_across_ev_band_keeps_mean_centre():
    # Regression (review B1): with many EVs the across-EV mean can sit outside
    # the world-median P5-P95 band; the contract keeps the mean and checks
    # only low <= high for that spread.
    result = make_result(evs=200, worlds=20)
    validate_result_v2(result)
    evs = result.average_day_bands.loc[result.average_day_bands["spread"].eq("across_evs")]
    assert set(evs["centre_stat"]) == {"mean_across_evs"}


def test_weekly_bands_on_every_result(no_action, action):
    for result in (no_action, action):
        weekly = result.weekly_bands
        assert set(weekly["metric"]) == {
            "home_import_kwh",
            "public_import_kwh",
            "unserved_travel_kwh",
        }
        assert (weekly["max"] >= weekly["p90"]).all()
    assert set(no_action.weekly_bands["path_id"]) == {"normal"}


def test_public_charging_is_exercised_on_the_selected_path(action):
    world = action.fleet_world_intervals
    selected = world.loc[world["path_id"].eq("selected")]
    assert (selected["total_import_kw"] > selected["home_import_kw"]).any()
    assert (
        action.cost_effect["illustrative_selected_minus_normal_public_charge_cost_gbp"] > 0
    ).any()
    replay = replay_one_ev(action, "ev-0000", world_id=1)
    assert replay.intervals["location"].eq("public_charging").any()


def test_plug_in_events_are_a_sample_but_kpis_cover_every_world():
    result = make_result(worlds=12)
    validate_result_v2(result)
    assert len(result.sampled_world_ids) == 10
    assert result.sampled_world_ids[0] == result.representative_world_id
    assert set(result.plug_in_events["world_id"]) == set(result.sampled_world_ids)
    assert result.plug_in_world_kpis["world_id"].nunique() == 12


def test_plug_in_histogram_mean_shares_sum_to_one(action):
    for frame in (action.plug_in_summary.hour_bands, action.plug_in_summary.soc_bands):
        totals = frame.groupby("day_type")["share_mean"].sum()
        assert np.allclose(totals, 1.0)


def test_validator_rejects_removed_action_fields(action):
    # Decision 0004 item 38 removed the eligibility screen and its frames.
    @dataclasses.dataclass(frozen=True)
    class WithOldField(type(action)):
        action_eligibility: object = None

    old = WithOldField(**{f.name: getattr(action, f.name) for f in dataclasses.fields(action)})
    with pytest.raises(AssertionError, match="action_eligibility"):
        validate_result_v2(old)


def test_validator_rejects_a_smart_charging_frame_not_derived_from_the_worlds(action):
    outcomes = action.smart_charging_world.assign(
        moved_home_import_kwh=action.smart_charging_world["moved_home_import_kwh"] + 1.0
    )
    with pytest.raises(AssertionError, match="smart_charging_world"):
        validate_result_v2(_broken(action, smart_charging_world=outcomes))


def test_validator_rejects_kwh_stored_as_kw(action):
    world = action.fleet_world_intervals.assign(
        home_import_kwh=action.fleet_world_intervals["home_import_kw"]
    )
    with pytest.raises(AssertionError, match="home_import_kw"):
        validate_result_v2(_broken(action, fleet_world_intervals=world))


def test_validator_rejects_counts_stored_as_shares(action):
    world = action.fleet_world_intervals.assign(
        connected_share=action.fleet_world_intervals["connected_count"].astype(float)
    )
    with pytest.raises(AssertionError, match="connected_share"):
        validate_result_v2(_broken(action, fleet_world_intervals=world))


def test_validator_rejects_histogram_shares_that_do_not_sum_to_one(action):
    summary = action.plug_in_summary
    hours = summary.hour_bands.assign(share_mean=summary.hour_bands["share_mean"] * 0.5)
    broken = dataclasses.replace(summary, hour_bands=hours)
    with pytest.raises(AssertionError, match="sum to 1"):
        validate_result_v2(_broken(action, plug_in_summary=broken))


def test_validator_rejects_kpis_not_derived_from_world_kpis(action):
    summary = action.plug_in_summary
    kpis = summary.kpis.assign(**{q: summary.kpis[q] + 1.0 for q in ("p10", "p50", "p90")})
    with pytest.raises(AssertionError, match="plug-in kpis"):
        validate_result_v2(_broken(action, plug_in_summary=dataclasses.replace(summary, kpis=kpis)))


def test_existing_public_names_still_import_from_the_fixture_module():
    import fixtures.result_fixture as fixture

    for name in (
        "Assumption",
        "FixtureForecastResult",
        "FLEET_METRICS",
        "fleet_bands",
        "difference_bands_from_world",
        "cost_summary_from_cost_effect",
        "difference_weekly_from_world",
        "validate_result_v2",
        "validate_one_ev_replay_v2",
        "validate_run_comparison_v2",
    ):
        assert hasattr(fixture, name), name


def test_validator_rejects_zone_sums_that_do_not_add_to_the_fleet(action):
    zones = action.zone_world_intervals.copy()
    zones.loc[zones["zone_id"].eq("zone_1"), "home_import_kwh"] += 1.0
    zones["home_import_kw"] = zones["home_import_kwh"] / 0.5
    with pytest.raises(AssertionError, match="add up to fleet"):
        validate_result_v2(_broken(action, zone_world_intervals=zones))


def test_validator_rejects_zone_band_mean_not_world_first(action):
    bands = action.zone_import_bands.copy()
    bands["mean"] = bands["p50"]
    with pytest.raises(AssertionError, match="zone_import_bands"):
        validate_result_v2(_broken(action, zone_import_bands=bands))


def test_validator_rejects_zone_frames_missing_on_no_action(no_action):
    with pytest.raises(AssertionError):
        validate_result_v2(_broken(no_action, zone_summary=None))


def test_validator_rejects_event_response_in_window_flag(action):
    # A hand-made one-event block: a 2-slot window shown from 1 slot before.
    events = pd.DataFrame(
        {
            "event_id": ["dfs"],
            "event_type": ["turn_down"],
            "enabled": [True],
            "night_index": np.array([1], dtype=np.int64),
            "start_local_time": ["17:30"],
            "duration_minutes": np.array([60], dtype=np.int64),
            "size": [np.nan],
            "size_unit": ["MW"],
            "notice": ["day_ahead"],
            "notice_minutes": pd.array([None], dtype="Int64"),
            "scope": ["national"],
            "payment_gbp_per_mwh": [500.0],
        }
    )
    slots = action.study_slots.iloc[82:86].reset_index(drop=True)
    bands = pd.DataFrame(
        {
            "event_id": "dfs",
            "event_type": "turn_down",
            "scope": "national",
            "series": "normal",
            "metric": "home_import_kw",
            "unit": "kW",
            "slot_index": slots["slot_index"],
            "interval_start_utc": slots["interval_start_utc"],
            "interval_start_london": slots["interval_start_london"],
            "relative_slot": np.arange(-1, 3, dtype=np.int64),
            "in_window": [False, True, True, False],
            "world_count": np.int64(action.world_count),
            "mean": 1.0,
            "p10": 1.0,
            "p50": 1.0,
            "p90": 1.0,
        }
    )
    changed = _broken(action, events=events, event_response_bands=bands)
    # The replay's events frame follows the events table (replay contract §1.5).
    replay = dataclasses.replace(action.replay_week, events=replay_events_from(changed))
    validate_result_v2(_broken(changed, replay_week=replay))
    wrong = bands.assign(in_window=[True, True, True, False])
    with pytest.raises(AssertionError, match="in_window"):
        validate_result_v2(_broken(action, events=events, event_response_bands=wrong))
