"""Firm-MW product frames (trading contract v1 §10.5, lane J5) on SYNTHETIC fixtures.

J1a/J1b are not merged, so the product frames are built from J2's
synthetic per-EV fixture (``fixtures.availability_contract``) and a toy
trading overlay (``fixtures.product_contract``), plus small hand-built
arrays whose answers are worked by hand.  Every figure is illustrative.
"""

from __future__ import annotations

import types
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fixtures.product_contract import (
    MAKER_CATALOGUE,
    make_product_case,
    validate_case,
    validate_settlement_file,
)

from axle_studio.model import availability, product
from axle_studio.model.summaries import RunSummary, build_study_slots, find_sessions, slim_run

NAN = np.nan


@pytest.fixture(scope="module")
def case():
    return make_product_case()


# --------------------------------------------------------------------------
# §10.5b Blackout-window validator (lead ruling §10.11: lane J5)
# --------------------------------------------------------------------------


def test_blackout_validator_types_the_table_and_wraps_midnight() -> None:
    table = availability.validate_blackout_windows(
        pd.DataFrame({"start_local_time": ["17:00", "23:00"], "duration_minutes": [60, 90.0]})
    )
    assert list(table.columns) == [
        "start_local_time",
        "duration_minutes",
        "end_local_time",
        "slot_labels",
    ]
    assert table["duration_minutes"].dtype == np.int64
    assert list(table["end_local_time"]) == ["18:00", "00:30"]
    assert table["slot_labels"].iat[0] == ("17:00", "17:30")
    assert table["slot_labels"].iat[1] == ("23:00", "23:30", "00:00")


def test_blackout_validator_accepts_an_empty_table() -> None:
    empty = pd.DataFrame({"start_local_time": [], "duration_minutes": []})
    table = availability.validate_blackout_windows(empty)
    assert table.empty and "slot_labels" in table.columns


@pytest.mark.parametrize(
    ("starts", "durations", "message"),
    [
        (["18:10"], [60], "row 0, start_local_time"),  # a 10-minute start
        (["18:00"], [45], "row 0, duration_minutes"),  # a 45-minute duration
        (["18:00"], [0], "row 0, duration_minutes"),  # zero duration
        (["18:00"], [750], "row 0, duration_minutes"),  # longer than 12 hours
        (["7:00"], [30], "row 0, start_local_time"),  # not "HH:MM"
        (["18:00", "19:00"], [120, 60], "row 1: overlaps blackout window row 0 at 19:00"),
        (["23:30", "00:00"], [60, 30], "row 1: overlaps blackout window row 0 at 00:00"),
    ],
)
def test_blackout_validator_rejects(starts, durations, message) -> None:
    frame = pd.DataFrame({"start_local_time": starts, "duration_minutes": durations})
    with pytest.raises(ValueError, match=message):
        availability.validate_blackout_windows(frame)


def test_blackout_validator_needs_both_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        availability.validate_blackout_windows(pd.DataFrame({"start_local_time": ["18:00"]}))


# --------------------------------------------------------------------------
# A hand-built availability_world_slot over one night
# --------------------------------------------------------------------------


def _hand_world_slot(deliverable_kw: np.ndarray) -> pd.DataFrame:
    """World-slot frame from (world, slot, cell) kW over one study night; other columns derived.

    Eligible power is twice the deliverable, the intraday P10 is half of it
    and the known part of the intraday mean is a quarter of the mean (which
    equals the deliverable), so the hand answers are simple.
    """

    slots = build_study_slots(date(2026, 10, 5), 1)
    world_count, slot_count, cells = deliverable_kw.shape
    slot_rows = np.tile(np.repeat(np.arange(slot_count), cells), world_count)
    kw = deliverable_kw.ravel()
    return pd.DataFrame(
        {
            "world_id": np.repeat(np.arange(world_count), slot_count * cells),
            "slot_index": slot_rows,
            "interval_start_london": slots["interval_start_london"]
            .take(slot_rows)
            .reset_index(drop=True),
            "direction": np.tile([d for d, _ in availability.CELLS], world_count * slot_count),
            "duration_hours": np.tile(
                [h for _, h in availability.CELLS], world_count * slot_count
            ).astype(float),
            "deliverable_kw": kw,
            "eligible_power_kw": 2 * kw,
            "intraday_p10_kw": kw / 2,
            "intraday_mean_kw": kw,
            "intraday_known_mean_kw": kw / 4,
        }
    )


def _labels() -> np.ndarray:
    return build_study_slots(date(2026, 10, 5), 1)["local_time_label"].to_numpy(dtype=object)


# --------------------------------------------------------------------------
# §10.5a Product sheet
# --------------------------------------------------------------------------


def test_product_sheet_recomputes_from_availability_world_slot(case) -> None:
    validate_case(case)  # every §10.9 rule for the J5 frames on the SYNTHETIC run
    sheet = case.built["product_sheet"]
    assert list(sheet["window"].unique()) == ["evening", "overnight", "morning"]
    by_window = sheet.drop_duplicates("window").set_index("window")
    assert list(by_window["notice_day_ahead_hours"]) == [28.0, 32.0, 41.0]
    assert list(by_window["notice_intraday_hours"]) == [0.0, 4.0, 13.0]
    assert (sheet["ramp_hours"] == 0.5).all()


def test_product_sheet_hand_values_and_nan_rule() -> None:
    labels = _labels()
    kw = np.full((2, 48, 8), 1000.0)
    # World 1's morning is undefined (as if its windows ran past the study
    # end): left out of that world's window, so the morning rows count one world.
    morning = (labels >= "06:00") & (labels < "12:00")
    kw[1, morning] = NAN
    # World 0 holds 2 MW in the evening, world 1 holds 1 MW.
    evening = (labels >= "17:00") & (labels < "21:00")
    kw[0, evening] = 2000.0
    sheet = product.product_sheet(
        _hand_world_slot(kw),
        day_ahead_decision_local_hour=13.0,
        intraday_decision_local_time="17:00",
    ).set_index(["window", "direction", "duration_hours"])
    row = sheet.loc[("evening", "turn_down", 0.5)]
    assert row["world_count"] == 2
    assert row["window_mean_mw_p50"] == pytest.approx(1.5)
    assert row["window_mean_mw_p10"] == pytest.approx(1.1)  # linear: 1 + 0.1 x (2 - 1)
    assert row["firm_share"] == pytest.approx(1.1 / 1.5)
    # 8 evening half-hours x MW x 0.5 h: 8 MWh and 4 MWh.
    assert row["movable_energy_mwh_p50"] == pytest.approx(6.0)
    assert row["intraday_firm_mw_p50"] == pytest.approx(0.75)
    assert row["intraday_known_share_p50"] == pytest.approx(0.25)
    one_hour = sheet.loc[("evening", "turn_down", 1.0)]
    assert np.isnan(one_hour["movable_energy_mwh_p50"])
    assert one_hour["recovery_energy_mwh_p50"] == pytest.approx(1.5)  # 1 h x window mean
    assert one_hour["rebound_capacity_mw_p50"] == pytest.approx(3.0)  # eligible = 2 x
    assert one_hour["recovery_hours_p50"] == pytest.approx(0.5)
    turn_up = sheet.loc[("evening", "turn_up", 1.0)]
    assert np.isnan(turn_up["recovery_energy_mwh_p50"]) and np.isnan(turn_up["recovery_hours_p50"])
    morning_row = sheet.loc[("morning", "turn_down", 4.0)]
    assert morning_row["world_count"] == 1
    assert morning_row["window_mean_mw_p50"] == pytest.approx(1.0)
    # Every duration holds the same MW here, so the longest qualifies.
    assert (sheet["max_event_length_hours"] == 4.0).all()


@pytest.mark.parametrize(
    ("p50", "expected"),
    [
        ((10.0, 5.0, 2.5, 2.4), 2.0),  # 2.4 < 25 % of 10
        ((10.0, 9.0, 8.0, 7.0), 4.0),
        ((10.0, 1.0, 5.0, 0.0), 2.0),  # the largest qualifying duration, not the first gap
        ((0.0, 0.0, 0.0, 0.0), NAN),  # nothing to size
        ((NAN, NAN, NAN, NAN), NAN),
    ],
)
def test_max_event_length_follows_the_25_percent_rule(p50, expected) -> None:
    got = product.max_event_length_hours(availability.DURATION_HOURS, p50)
    assert got == expected or (np.isnan(got) and np.isnan(expected))


# --------------------------------------------------------------------------
# §10.5c Price-weighted availability
# --------------------------------------------------------------------------


def test_value_summary_hand_values_and_nan_rule() -> None:
    kw = np.full((2, 48, 8), 1000.0)
    price = np.full((2, 48), 100.0)
    price[1, 0] = -50.0  # only world 1 has a negative half-hour
    summary = product.availability_value_summary(_hand_world_slot(kw), price).set_index(
        ["direction", "duration_hours", "metric"]
    )
    down = summary.loc[("turn_down", 0.5, "price_weighted_mw")]
    assert down["world_count"] == 2 and down["p50"] == pytest.approx(1.0)
    up = summary.loc[("turn_up", 0.5, "price_weighted_mw")]
    # World 0 has no negative price: no weight, NaN, left out of the statistics.
    assert up["world_count"] == 1
    assert up["p50"] == pytest.approx(1.0)
    value_up = summary.loc[("turn_up", 0.5, "value_at_day_ahead_gbp_per_week")]
    # 0 (not NaN) in world 0; world 1: 1 MW x 0.5 h x £50.
    assert value_up["world_count"] == 2
    assert value_up["mean"] == pytest.approx(12.5)
    value_down = summary.loc[("turn_down", 0.5, "value_at_day_ahead_gbp_per_week")]
    # World 0: 48 slots x 0.5 MWh x £100 = £2,400; world 1 loses the
    # negative slot (weight 0): £2,350.  Linear quantiles between them.
    assert value_down["p10"] == pytest.approx(2350.0 + 0.1 * 50.0)
    assert value_down["p90"] == pytest.approx(2350.0 + 0.9 * 50.0)


def test_value_summary_with_no_negative_price_anywhere_is_nan_for_turn_up() -> None:
    kw = np.full((3, 48, 8), 500.0)
    summary = product.availability_value_summary(_hand_world_slot(kw), np.full((3, 48), 60.0))
    up = summary.loc[summary["direction"].eq("turn_up")].set_index(["duration_hours", "metric"])
    weighted = up.xs("price_weighted_mw", level="metric")
    assert (weighted["world_count"] == 0).all() and weighted["p50"].isna().all()
    assert weighted["mean"].isna().all()
    assert (up.xs("value_at_day_ahead_gbp_per_week", level="metric")["p50"] == 0.0).all()


# --------------------------------------------------------------------------
# §10.5d Settlement file
# --------------------------------------------------------------------------


def test_settlement_file_hand_allocation_and_baseline_effect_row() -> None:
    slots = build_study_slots(date(2026, 10, 5), 1)
    keys = slots.assign(settlement_date="2026-10-05", settlement_period=1)
    slot_count = len(slots)
    # EV 0 turns down 2 kWh in slot 0 and rebounds 2 kWh in slot 1; EV 1
    # does nothing; EV 2 (control) turns down 1 kWh in slot 0.
    unmanaged = np.zeros((slot_count, 3))
    metered = np.zeros((slot_count, 3))
    unmanaged[0] = [2.0, 0.0, 1.0]
    metered[1] = [2.0, 0.0, 0.0]
    baseline_effect = np.full(slot_count, 0.5)
    true_reduction = (unmanaged - metered).sum(axis=1)
    settled = np.where(np.arange(slot_count) == 0, true_reduction + baseline_effect, 0.0)
    deviation = pd.DataFrame(
        {
            "slot_index": np.arange(slot_count),
            "unmanaged_kwh": unmanaged.sum(axis=1),
            "metered_kwh": metered.sum(axis=1),
            "true_reduction_kwh": true_reduction,
            "baseline_effect_kwh": baseline_effect,
            "settled_kwh": settled,
            "day_ahead_gbp_per_mwh": np.full(slot_count, 90.0),
        }
    )
    units = pd.DataFrame(
        {
            "unit_id": ["ev_0", "ev_1", "ev_2"],
            "cohort_id": ["a", "a", "b"],
            "manufacturer_id": ["m1", "m2", "m1"],
            "zone_id": ["zone_1", "zone_2", "zone_3"],
            "control_group": [False, False, True],
        }
    )
    weights = np.array([[0.75], [0.25], [0.0]])  # a_{i,n}; the control EV gets 0
    file = product.settlement_file(
        keys,
        units=units,
        unmanaged_kwh=unmanaged,
        metered_kwh=metered,
        deviation=deviation,
        weights=weights,
        customer_share_gbp=np.array([-8.0]),
    )
    validate_settlement_file(
        file.assign(settlement_date=file["settlement_date"].astype(object)),
        deviation_world=deviation,
        customer_share=np.array([-8.0]),
        vehicle_count=3,
    )
    # Unit-major: every slot of ev_0, then ev_1, ev_2, then the fleet rows.
    assert list(file["meter_point_id"].iloc[::slot_count]) == [
        "ev_0",
        "ev_1",
        "ev_2",
        "baseline_effect",
    ]
    rows = file.set_index(["meter_point_id", "slot_index"])
    assert rows.loc[("ev_0", 0), "settled_kwh"] == 2.0
    assert rows.loc[("ev_0", 1), "settled_kwh"] == 0.0  # the rebound slot does not settle
    assert rows.loc[("ev_0", 1), "deviation_kwh"] == -2.0
    assert rows.loc[("ev_2", 0), "settled_kwh"] == 1.0  # control keeps its settled kWh
    assert rows.loc[("baseline_effect", 0), "settled_kwh"] == 0.5
    assert rows.loc[("baseline_effect", 1), "settled_kwh"] == 0.0
    # ev_0's £6 all lands on its one positively settled slot; ev_1 has none,
    # so its £2 is spread evenly over the night; the control EV is paid 0.
    assert rows.loc[("ev_0", 0), "payment_gbp"] == pytest.approx(6.0)
    assert rows.loc[("ev_1", 5), "payment_gbp"] == pytest.approx(2.0 / slot_count)
    assert (rows.loc["ev_2", "payment_gbp"] == 0).all()


def test_settlement_file_on_the_synthetic_run(case) -> None:
    file = case.built["settlement_file"]
    assert file["meter_point_id"].nunique() == len(case.units) + 1
    assert case.built["settlement_file_name"] == (
        f"axle_settlement_2026-10-05_seed{case.seed}_world{case.representative_world_id}.csv"
    )
    # Treated EVs share the week's customer payment; the file sums to it.
    assert file["payment_gbp"].sum() == pytest.approx(
        -case.customer_share[case.representative_world_id].sum()
    )


def test_settlement_file_pays_control_evs_nothing() -> None:
    case = make_product_case(world_count=4, vehicle_count=16, control_share=0.3)
    assert case.units["control_group"].any()
    validate_case(case)
    file = case.built["settlement_file"]
    control = file.loc[file["control_group"]]
    assert (control["payment_gbp"] == 0).all()
    # Control EVs are never flexed, so their true reduction is 0 here; the
    # hand test shows a non-zero one kept.  They stay in the file.
    assert len(control) == case.units["control_group"].sum() * len(case.study_slots)


# --------------------------------------------------------------------------
# §10.5e Charge completion
# --------------------------------------------------------------------------


def test_completion_counts_hand_case() -> None:
    slot_count = 8
    connected = np.zeros((1, slot_count, 3), dtype=bool)
    closing = np.zeros((1, slot_count, 3))
    connected[0, 1:6, 0] = True  # EV 0: slots 1-5, unplugs at 6 = the expected departure
    closing[0, 5, 0] = 10.0  # at target
    connected[0, 1:3, 1] = True  # EV 1: unplugs at 3, early
    closing[0, 2, 1] = 7.0  # 3 kWh short
    connected[0, 0:4, 2] = True  # EV 2: plugged at the horizon start (not observed) ...
    connected[0, 5:, 2] = True  # ... and still plugged at the end (not ended)
    sessions = find_sessions(connected, closing, np.zeros_like(closing))
    members = np.array([[True, False], [True, True], [True, False]])  # fleet, group of EV 1
    counts = product.completion_counts(
        sessions,
        slot_count=slot_count,
        target_kwh=np.full(3, 10.0),
        expected_end_slot=np.full((slot_count, 3), 6),
        members=members,
        world_count=1,
    )
    assert counts["sessions"][0, 0].tolist() == [2, 1, 1]  # all, on time, early
    assert counts["complete"][0, 0].tolist() == [1, 1, 0]
    assert counts["shortfall_kwh"][0, 0].tolist() == [3, 0, 3]
    assert counts["sessions"][0, 1].tolist() == [1, 0, 1]
    both = {name: np.stack([v, v], axis=2) for name, v in counts.items()}
    summary = product.charge_completion_summary(both, ["fleet", "g"], [3, 1]).set_index(
        ["group_id", "path_id", "departure"]
    )
    fleet_all = summary.loc[("fleet", "selected", "all")]
    assert fleet_all["completed_share_p50"] == 0.5
    assert fleet_all["shortfall_kwh_mean"] == 3.0
    on_time = summary.loc[("fleet", "normal", "on_time")]
    assert on_time["completed_share_mean"] == 1.0 and np.isnan(on_time["shortfall_kwh_p50"])
    none = summary.loc[("g", "normal", "on_time")]
    assert none["world_count"] == 0 and np.isnan(none["completed_share_p50"])


def test_completion_groups_are_fleet_cohorts_then_makers(case) -> None:
    summary = case.built["charge_completion_summary"]
    assert list(summary["group_id"].unique()) == [
        "fleet",
        "cohort_a",
        "cohort_b",
        "m1",
        "m2",
        "m3",
        "m4",
    ]
    fleet = summary.loc[summary["group_id"].eq("fleet")].set_index(["path_id", "departure"])
    makers = summary.loc[summary["group_id"].str.startswith("m")]
    for path in product.PATH_IDS:
        assert makers.loc[
            makers["path_id"].eq(path) & makers["departure"].eq("all"), "session_count_mean"
        ].sum() == pytest.approx(fleet.loc[(path, "all"), "session_count_mean"])


# --------------------------------------------------------------------------
# §10.5f Net shape change with rebound
# --------------------------------------------------------------------------


def test_net_change_quantiles_show_the_rebound() -> None:
    # Three worlds, four slots: turn-down in slot 0, the charging lands in slot 2.
    unmanaged = np.array([[4.0, 0, 0, 0], [3.0, 0, 0, 0], [2.0, 0, 0, 0]])
    metered = np.array([[0.0, 0, 4, 0], [0.0, 0, 3, 0], [0.0, 0, 2, 0]])
    deviation = pd.DataFrame(
        {
            "world_id": np.repeat(np.arange(3), 4),
            "slot_index": np.tile(np.arange(4), 3),
            "unmanaged_kwh": unmanaged.ravel(),
            "metered_kwh": metered.ravel(),
        }
    )
    positions = pd.DataFrame(
        {"slot_index": np.arange(4), "settled_mw_p50": 0.0, "price_curve_source": "synthetic"}
    )
    frame = product.supplier_position_additions(positions, deviation, None)
    assert list(frame.columns) == [
        "slot_index",
        "settled_mw_p50",
        *product.POSITION_ADDITION_COLUMNS,
        "price_curve_source",
    ]
    # kWh per half-hour / 0.5 / 1000 = MW.
    assert frame["net_change_mw_p50"].tolist() == pytest.approx([-0.006, 0, 0.006, 0])
    # Linear quantiles of (-8, -6, -4) and (4, 6, 8) kW.
    assert frame["net_change_mw_p10"].iat[0] == pytest.approx(-0.0076)
    assert frame["net_change_mw_p90"].iat[2] == pytest.approx(0.0076)
    assert frame["unmanaged_mw_p50"].iat[0] == pytest.approx(0.006)
    assert frame["deliverable_turn_down_1h_mw_p10"].isna().all()


# --------------------------------------------------------------------------
# §10.5g-h Firmness by manufacturer and the diversification explainer
# --------------------------------------------------------------------------


def test_firmness_with_one_maker_always_out() -> None:
    case = make_product_case(world_count=4, vehicle_count=16, outage_probability=(1.0, 0, 0, 0))
    validate_case(case)
    firmness = case.built["firmness_by_manufacturer"]
    m1 = firmness.loc[firmness["manufacturer_id"].eq("m1")]
    assert (m1["outage_nights_mean"] == 7).all()
    # Out every night: it delivers nothing of a positive potential.
    defined = m1.dropna(subset=["firmness_p50"])
    assert not defined.empty and (defined["firmness_p50"] == 0).all()
    others = firmness.loc[~firmness["manufacturer_id"].eq("m1")]
    assert (others["outage_nights_mean"] == 0).all()
    assert (others["firmness_p50"].dropna() > 0).any()


def test_manufacturer_summary_any_outage_and_largest_share() -> None:
    makers = product.manufacturer_table(
        (
            ("m1", "Maker A", 0.4),
            ("m2", "Maker B", 0.3),
            ("m3", "Maker C", 0.2),
            ("m4", "Maker D", 0.1),
        ),
        np.array([0, 0, 1, 2, 3, 0]),
        np.array([0.95, 0.9, 0.95, 1.0]),
        np.array([0.02, 0.1, 0.2, 0.5]),
    )
    potential = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 2.0, 1.0, 0.0], [2.0, 2.0, 2.0, 2.0]])
    summary = product.manufacturer_summary(potential, makers).set_index("manufacturer_id")
    assert summary.loc["fleet", "outage_probability_per_night"] == pytest.approx(
        1 - 0.98 * 0.9 * 0.8 * 0.5
    )
    assert summary.loc["fleet", "ev_count"] == 6 and summary.loc["m1", "ev_count"] == 3
    assert (summary["largest_share"] == 0.4).all()
    assert summary.loc["m1", "evening_potential_mw_p50"] == 2.0
    # Fleet row: the median of the per-world totals (10, 6, 8), not a sum of medians.
    assert summary.loc["fleet", "evening_potential_mw_p50"] == 8.0
    assert np.isnan(summary.loc["fleet", "response_rate"])


def test_evening_potential_sums_to_the_fleet_potential(case) -> None:
    # The per-maker evening potential of the chunk loop adds up to the
    # fleet's 1-h turn-down potential_kw of availability_world_slot.
    world_slot = case.frames["availability_world_slot"]
    rows = world_slot.loc[
        world_slot["direction"].eq("turn_down")
        & world_slot["duration_hours"].eq(1.0)
        & world_slot["interval_start_london"].dt.strftime("%H:%M").between("17:00", "20:30")
    ]
    fleet = rows.groupby("world_id")["potential_kw"].mean().to_numpy()
    np.testing.assert_allclose(case.totals["evening_potential_kw"].sum(axis=1), fleet, atol=1e-9)
    summary = case.built["manufacturer_summary"].set_index("manufacturer_id")
    assert summary.loc["fleet", "evening_potential_mw_p50"] == pytest.approx(
        np.median(fleet) / 1000
    )


# --------------------------------------------------------------------------
# Determinism and Compare
# --------------------------------------------------------------------------


def test_product_frames_are_deterministic(case) -> None:
    again = make_product_case()
    for name in (
        "product_sheet",
        "availability_value_summary",
        "settlement_file",
        "charge_completion_summary",
        "firmness_by_manufacturer",
        "manufacturer_summary",
        "supplier_positions",
    ):
        pd.testing.assert_frame_equal(case.built[name], again.built[name])


def test_firm_mw_comparison_is_side_by_side(case) -> None:
    sheet, firmness = case.built["product_sheet"], case.built["firmness_by_manufacturer"]
    rows = product.firm_mw_comparison(sheet, firmness, sheet, firmness)
    assert list(rows.columns) == product.FIRM_MW_COMPARISON_COLUMNS
    assert list(rows["key"]) == ["evening"] * 4 + ["overnight"] * 4 + ["m1", "m2", "m3", "m4"]
    np.testing.assert_array_equal(rows["value_a"], rows["value_b"])
    assert product.firm_mw_comparison(None, firmness, sheet, firmness) is None


def test_slim_run_keeps_the_firm_mw_frames(case) -> None:
    fields = {name: None for name in RunSummary.__dataclass_fields__}
    fields.update(settings_snapshot={}, assumptions=())
    result = types.SimpleNamespace(
        **{
            **fields,
            "product_sheet": case.built["product_sheet"],
            "firmness_by_manufacturer": case.built["firmness_by_manufacturer"],
        }
    )
    summary = slim_run(result)
    assert summary.product_sheet is case.built["product_sheet"]
    assert summary.firmness_by_manufacturer is case.built["firmness_by_manufacturer"]
    older = types.SimpleNamespace(**fields)
    del older.product_sheet, older.firmness_by_manufacturer
    assert slim_run(older).product_sheet is None


def test_maker_catalogue_matches_the_fixture_makers(case) -> None:
    summary = case.built["manufacturer_summary"]
    assert list(summary["manufacturer_label"][:4]) == [label for _, label, _ in MAKER_CATALOGUE]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no stray RuntimeWarning from the NaN rules
        product.firmness_by_manufacturer(
            case.frames["availability_manufacturer_world"],
            product.manufacturer_table(
                MAKER_CATALOGUE,
                case.fixture.inputs.manufacturer_index,
                case.fixture.inputs.response_rate,
                case.fixture.inputs.outage_probability,
            ),
        )
