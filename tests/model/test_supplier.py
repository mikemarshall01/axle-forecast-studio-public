"""Supplier P&L and Partners frames (supplier contract v1 §3–§4, §10; lane S1).

Hand-checkable fixtures come first (the §3.6 worked example and the §10
checklist cases), then small SYNTHETIC real runs that pass the §9
validator.  Every figure is illustrative; nothing here is Axle cash.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from functools import cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fixtures.supplier_contract import validate_supplier_frames

from axle_studio.model import assumptions, forecast, market, summaries, supplier
from axle_studio.model.summaries import build_study_slots

_DEFAULTS = assumptions.supplier_inputs()
_SHARE = {"trading.customer_revenue_share": 0.5}
_PRICES = {"summer_months": (4, 5, 6, 7, 8, 9), "supply_reference_net_demand_gw": 20.0}


# --------------------------------------------------------------------------
# §3.6 worked example: one world, one night of two slots, N_T = 2
# --------------------------------------------------------------------------


def _example(
    *,
    x_u=(10.0, 0.0),
    x=(4.0, 6.0),
    sip=(120.0, 40.0),
    control: bool = False,
    ledger_edit: dict[str, float] | None = None,
    **records: float,
) -> dict[str, pd.DataFrame]:
    """``build_frames`` on the §3.6 hand fixture (SYNTHETIC)."""

    slots = build_study_slots(date(2026, 10, 12)).iloc[:2].reset_index(drop=True)
    price = np.array([100.0, 50.0])
    u, m = np.array([10.0, 0.0]), np.array([2.0, 8.0])
    deviation = pd.DataFrame(
        {
            "world_id": 0,
            "slot_index": [0, 1],
            "night_index": 0,
            "unmanaged_kwh": u,
            "metered_kwh": m,
            "true_reduction_kwh": u - m,
            "expected_unmanaged_kwh": np.array(x_u),
            "expected_metered_kwh": np.array(x),
            "day_ahead_gbp_per_mwh": price,
            "imbalance_gbp_per_mwh": np.array(sip),
        }
    )
    buckets = dict.fromkeys(market.MONEY_COLUMNS, 0.0) | (ledger_edit or {})
    ledger = pd.DataFrame([{"world_id": 0, "strategy": "full", "night_index": 0, **buckets}])
    week = ledger.drop(columns="night_index")
    # ``cost_effect`` holds selected minus normal energy cost: 0.60 - 1.00.
    cost_effect = pd.DataFrame(
        {"world_id": [0], "illustrative_selected_minus_normal_energy_cost_gbp": [-0.4]}
    )
    units = pd.DataFrame(
        {
            "cohort_id": ["average_uk"] * (3 if control else 2),
            "control_group": [False, False, True] if control else [False, False],
            "home_charger_limit_kw": 7.0,
            "physical_capacity_kwh": 50.0,
        }
    )
    evs = len(units)
    per_ev = {
        "value_gbp_per_month": np.zeros((1, evs)),
        "earning": np.array([[True] * 2 + [False] * (evs - 2)]),
    }
    weights = np.array([[[0.5], [0.5]] + [[0.0]] * (evs - 2)])
    return supplier.build_frames(
        SimpleNamespace(
            deviation_world_slot=deviation, trading_ledger_world=ledger, trading_week_world=week
        ),
        per_ev,
        weights,
        cost_effect=cost_effect,
        units=units,
        study_slots=slots,
        cohort_ids=["average_uk"],
        battery_added={"normal": np.full((1, evs), 10.0), "selected": np.full((1, evs), 10.0)},
        forecast_prices=pd.DataFrame(
            {"world_id": 0, "slot_index": [0, 1], "system_net_demand_gw": [20.0, 20.0]}
        ),
        price_assumptions=_PRICES,
        trading_assumptions=_SHARE,
        supplier_inputs=_DEFAULTS | records,
        warmup_slot_count=48 * 7,
    )


def _row(frames, variant: str) -> pd.Series:
    world = frames["supplier_pnl_world"]
    return world.loc[world["hedge_variant"].eq(variant)].iloc[0]


def test_worked_example_profiled_and_flat_hedge() -> None:
    frames = _example()
    profiled, flat = _row(frames, "profiled"), _row(frames, "flat")
    assert profiled["energy_saving_gbp"] == pytest.approx(0.40)
    assert profiled["shape_saving_gbp"] == pytest.approx(0.40)
    assert profiled["volume_value_gbp"] == pytest.approx(0.0)
    assert profiled["hedge_error_saving_gbp"] == pytest.approx(0.06)
    assert flat["hedge_error_saving_gbp"] == pytest.approx(0.24)
    assert profiled["worth_of_profiled_hedge_gbp"] == pytest.approx(-0.18)
    assert profiled["gross_gain_gbp"] == pytest.approx(0.46)
    assert profiled["customer_payment_gbp"] == pytest.approx(-0.23)
    assert profiled["net_gain_before_fee_gbp"] == pytest.approx(0.23)
    assert profiled["net_gain_before_fee_per_customer_per_month_gbp"] == pytest.approx(
        0.23 * 52 / 12 / 2
    )
    assert flat["net_gain_before_fee_gbp"] == pytest.approx(0.32)
    assert flat["net_gain_before_fee_per_customer_per_month_gbp"] == pytest.approx(0.693, abs=5e-4)
    # The fee is unset, so the after-fee figures are unavailable, not zero.
    for column in ("platform_fee_gbp", "net_gain_gbp", "net_gain_per_customer_per_month_gbp"):
        assert np.isnan(profiled[column]) and np.isnan(flat[column])


def test_worked_example_with_a_platform_fee() -> None:
    profiled = _row(_example(**{"supplier.platform_fee_gbp_per_ev_per_month": 1.0}), "profiled")
    assert profiled["platform_fee_gbp"] == pytest.approx(-1 * 2 * 12 / 52)
    assert profiled["net_gain_gbp"] == pytest.approx(0.23 - 2 * 12 / 52)
    assert profiled["platform_fee_per_customer_per_month_gbp"] == pytest.approx(-1.0)
    assert profiled["net_gain_per_customer_per_month_gbp"] == pytest.approx(
        0.23 * 52 / 12 / 2 - 1.0
    )


def test_method_bias_common_to_both_forecasts_cancels() -> None:
    base = _example()["supplier_pnl_world"]
    biased = _example(x_u=(9.0, 1.0), x=(3.0, 7.0))["supplier_pnl_world"]
    pd.testing.assert_frame_equal(base, biased)


def test_a_control_ev_changes_only_the_treated_count_figures() -> None:
    # A control EV is in both x^U and x (at rho = 1), so F and every weekly
    # component are unchanged; only N_T and the per-customer columns move.
    base = _example()["supplier_pnl_world"]
    with_control = _example(control=True)["supplier_pnl_world"]
    assert (with_control["treated_ev_count"] == 2).all()
    pd.testing.assert_frame_equal(base, with_control)


def test_equal_imbalance_and_day_ahead_prices_give_no_hedge_error() -> None:
    world = _example(sip=(100.0, 50.0))["supplier_pnl_world"]
    assert (world["hedge_error_saving_gbp"] == 0.0).all()
    assert (world["worth_of_profiled_hedge_gbp"] == 0.0).all()


def test_a_perfect_forecast_leaves_only_the_flat_hedge_error() -> None:
    # F = R = [8, -8]: x^U - x equals U - M.
    frames = _example(x_u=(10.0, 0.0), x=(2.0, 8.0))
    assert _row(frames, "profiled")["hedge_error_saving_gbp"] == pytest.approx(0.0)
    assert _row(frames, "flat")["hedge_error_saving_gbp"] == pytest.approx((160 + 80) / 1000)


def test_negative_gross_week_pays_the_customer_nothing() -> None:
    # SIP far below P in slot 0 turns the week's gross negative.
    profiled = _row(_example(sip=(-2000.0, 40.0)), "profiled")
    assert profiled["gross_gain_gbp"] < 0.0
    assert profiled["customer_payment_gbp"] == 0.0


def test_flat_reward_mode_is_nan_until_set_and_changes_only_the_payment() -> None:
    base = _example()["supplier_pnl_world"]
    unset = _example(**{"supplier.customer_reward_mode": 1})["supplier_pnl_world"]
    set_ = _example(
        **{
            "supplier.customer_reward_mode": 1,
            "supplier.customer_reward_gbp_per_ev_per_month": 2.0,
        }
    )["supplier_pnl_world"]
    moved = [
        "customer_payment_gbp",
        "net_gain_before_fee_gbp",
        "net_gain_gbp",
        "net_gain_before_fee_per_customer_per_month_gbp",
        "net_gain_per_customer_per_month_gbp",
        "customer_payment_per_customer_per_month_gbp",
    ]
    pd.testing.assert_frame_equal(base.drop(columns=moved), set_.drop(columns=moved))
    assert unset["customer_payment_gbp"].isna().all()
    assert unset["net_gain_before_fee_gbp"].isna().all()
    assert set_["customer_payment_gbp"].to_numpy() == pytest.approx([-2 * 2 * 12 / 52] * 2)
    assert set_["customer_payment_per_customer_per_month_gbp"].to_numpy() == pytest.approx(
        [-2.0, -2.0]
    )


def test_only_the_grid_event_bucket_of_the_ledger_enters_the_supplier_pnl() -> None:
    # Decision 0006: the ledger is shown beside and never added.  Moving any
    # other bucket (and the ledger net) leaves every supplier figure alone.
    base = _example()["supplier_pnl_world"]
    other_buckets = {c: 5.0 for c in market.MONEY_COLUMNS if c != "grid_event_payment_gbp"}
    pd.testing.assert_frame_equal(base, _example(ledger_edit=other_buckets)["supplier_pnl_world"])
    paid = _row(_example(ledger_edit={"grid_event_payment_gbp": 1.0}), "profiled")
    assert paid["grid_event_payment_gbp"] == 1.0
    assert paid["gross_gain_gbp"] == pytest.approx(1.46)


def test_summary_means_add_up_and_carry_horizon_and_season() -> None:
    summary = _example()["supplier_pnl_summary"]
    assert set(summary["season"]) == {"winter"}
    per_month = summary["metric"].str.endswith("_per_customer_per_month_gbp")
    assert (summary.loc[per_month, "horizon"] == "scenario").all()
    assert (summary.loc[~per_month, "horizon"] == "week_ahead").all()
    fee = summary.loc[summary["metric"].eq("platform_fee_gbp")]
    assert (fee["world_count"] == 0).all() and fee["mean"].isna().all()


def test_revenue_share_mode_must_be_a_switch() -> None:
    with pytest.raises(ValueError, match="customer_reward_mode"):
        _example(**{"supplier.customer_reward_mode": 2})


# --------------------------------------------------------------------------
# §3.7 hedge blocks, §3.8 carbon, season
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "clock_change", "peak_hours"),
    [
        (date(2026, 10, 12), None, 60.0),  # Monday
        (date(2026, 10, 17), None, 60.0),  # Saturday
        (date(2026, 10, 19), "autumn", 59.0),  # Monday; clocks go back 25 Oct
        (date(2027, 3, 22), "spring", 61.0),  # Monday; clocks go forward 28 Mar
    ],
)
def test_block_hours_come_from_the_study_slots(start, clock_change, peak_hours) -> None:
    slots = build_study_slots(start)
    masks = supplier.block_masks(slots)
    assert masks["baseload"].sum() * 0.5 == 168.0
    assert masks["peak"].sum() * 0.5 == peak_hours
    assert (masks["off_peak"] == ~masks["peak"]).all()


def test_monday_and_saturday_starts_put_the_peak_on_different_slots() -> None:
    monday = supplier.block_masks(build_study_slots(date(2026, 10, 12)))["peak"]
    saturday = supplier.block_masks(build_study_slots(date(2026, 10, 17)))["peak"]
    assert monday.sum() == saturday.sum() and not np.array_equal(monday, saturday)


def test_all_import_in_one_peak_slot_is_all_peak() -> None:
    slots = build_study_slots(date(2026, 10, 12))
    peak_slot = int(np.flatnonzero(supplier.block_masks(slots)["peak"])[0])
    unmanaged = np.zeros((2, 336))
    unmanaged[:, peak_slot] = 1000.0
    metered = np.roll(unmanaged, 30, axis=1)
    world = supplier.hedge_block_world(unmanaged, metered, slots)
    normal = world.loc[world["path_id"].eq("normal")].set_index(["world_id", "block"])
    assert normal.loc[(0, "peak"), "volume_mwh"] == normal.loc[(0, "baseload"), "volume_mwh"]
    assert normal.loc[(0, "off_peak"), "volume_mwh"] == 0.0
    difference = world.loc[world["path_id"].eq("difference")].reset_index(drop=True)
    selected = world.loc[world["path_id"].eq("selected")].reset_index(drop=True)
    normal = world.loc[world["path_id"].eq("normal")].reset_index(drop=True)
    assert np.array_equal(difference["volume_mwh"], selected["volume_mwh"] - normal["volume_mwh"])


def test_carbon_with_no_slope_is_the_reference_intensity_times_the_reduction() -> None:
    carbon = _DEFAULTS | {"carbon.intensity_slope_gco2_per_kwh_per_gw": 0.0}
    demand = np.array([[5.0, 30.0, 18.0]])
    intensity = supplier.carbon_intensity_gco2_per_kwh(demand, carbon, 20.0)
    assert (intensity == 180.0).all()
    u, m = np.array([[4.0, 0.0, 3.0]]), np.array([[0.0, 5.0, 1.0]])
    world = supplier.carbon_shift_world(u, m, intensity, treated_ev_count=2)
    assert world["co2_shifted_kg"].iat[0] == pytest.approx(180.0 * (u - m).sum() / 1000.0)


def test_carbon_intensity_is_clipped_at_the_floor_and_the_cap() -> None:
    intensity = supplier.carbon_intensity_gco2_per_kwh(
        np.array([-50.0, 20.0, 80.0]), _DEFAULTS, 20.0
    )
    assert intensity.tolist() == [20.0, 180.0, 450.0]


def test_season_follows_the_warm_up_start_month() -> None:
    # A 1 April study with a seven-day warm-up starts pricing in March.
    april = build_study_slots(date(2027, 4, 1))
    assert supplier.season_of_run(april, 48 * 7, _PRICES["summer_months"]) == "winter"
    assert supplier.season_of_run(april, 0, _PRICES["summer_months"]) == "summer"


# --------------------------------------------------------------------------
# §4 Partners and exceedance
# --------------------------------------------------------------------------


def _partners(**overrides) -> pd.DataFrame:
    """Four treated EVs, two makers, two nights (SYNTHETIC)."""

    units = pd.DataFrame(
        {
            "cohort_id": ["average_uk", "average_uk", "infrequent_driving", "average_uk"],
            "manufacturer_id": ["m1", "m2", "m1", "m2"],
            "home_charger_limit_kw": 7.0,
            "physical_capacity_kwh": 50.0,
        }
    )
    reduction = np.array([[[2.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]])
    treated = np.ones(4, dtype=bool)
    inputs = {
        "groups": supplier.partner_groups(units, ["average_uk", "infrequent_driving"]),
        "weights": summaries.allocation_weights(reduction, treated),
        "gross_gbp": np.array([[8.0, 4.0]]),
        "value_gbp_per_month": np.array([[-5.0, 4.0, 12.0, 30.0]]),
        "earning": reduction.sum(axis=2) > 0.0,
        "charger_kw": units["home_charger_limit_kw"].to_numpy(dtype=float),
        "capacity_kwh": units["physical_capacity_kwh"].to_numpy(dtype=float),
        "battery_added_kwh": {"normal": np.full((1, 4), 10.0), "selected": np.full((1, 4), 10.0)},
    }
    return supplier.partner_world(**(inputs | overrides))


def test_partner_groups_sum_to_the_fleet_and_spread_an_earnerless_night_equally() -> None:
    world = _partners().set_index(["group_type", "group_id"])
    fleet = world.loc[("fleet", "fleet")]
    # Night 0 by reduction (2:1:1:0), night 1 has no earner so 4/4 each.
    assert fleet["gross_flex_gbp_per_week"] == pytest.approx(12.0)
    makers = world.loc["manufacturer", "gross_flex_gbp_per_week"]
    assert makers["m1"] == pytest.approx(8 * 3 / 4 + 2.0)  # EVs 0 and 2
    assert makers["m2"] == pytest.approx(8 * 1 / 4 + 2.0)
    assert makers.sum() == pytest.approx(12.0)
    assert world.loc["cohort", "gross_flex_gbp_per_week"].sum() == pytest.approx(12.0)
    assert fleet["share_earning"] == 0.75
    assert fleet["gross_flex_gbp_per_enrolled_device_per_month"] == pytest.approx(
        fleet["share_earning"] * fleet["gross_flex_gbp_per_earning_device_per_month"]
    )
    assert fleet["gbp_per_kw_charger_per_year"] == pytest.approx(12.0 * 52 / 28.0)
    assert fleet["customer_value_gbp_per_device_per_month"] == pytest.approx(41.0 / 4)


def test_a_group_with_no_earner_has_a_nan_per_earning_figure() -> None:
    world = _partners(gross_gbp=np.array([[0.0, 4.0]]))
    # With no reduction at all, every night spreads equally.
    no_earner = _partners(
        weights=np.full((1, 4, 2), 0.25), earning=np.zeros((1, 4), dtype=bool)
    ).set_index("group_id")
    assert np.isnan(no_earner.loc["fleet", "gross_flex_gbp_per_earning_device_per_month"])
    assert no_earner.loc["fleet", "gross_flex_gbp_per_enrolled_device_per_month"] > 0.0
    assert world["dispatch_success_rate"].isna().all()  # firm-MW J1b seam


def test_dispatch_success_is_session_weighted_from_plan_status() -> None:
    # One world, 8 slots, 4 EVs (SYNTHETIC).  Five sessions end in the
    # study; two have an ignored slot, so the fleet rate is 3 / 5 = 0.6.
    # A sixth session still plugged in at the study end does not count.
    status = np.zeros((1, 8, 4))
    status[0, 2, 0] = 1.0  # EV 0's first session (slots 1-3) ignores one slot
    status[0, 6, 2] = 2.0  # EV 2's session (slots 5-6) is on a maker outage night
    sessions = {
        "world": np.zeros(6, dtype=np.int64),
        "ev": np.array([0, 0, 1, 2, 3, 3]),
        "first_slot": np.array([1, 5, 2, 5, 1, 6]),
        "end_slot": np.array([4, 7, 4, 7, 3, 8]),
    }
    counts = supplier.dispatch_session_counts(status, **sessions)
    np.testing.assert_array_equal(counts[0], [[2.0, 1.0, 1.0, 1.0]])
    np.testing.assert_array_equal(counts[1], [[1.0, 1.0, 0.0, 1.0]])
    world = _partners(dispatch_sessions=counts).set_index("group_id")
    assert world.loc["fleet", "dispatch_success_rate"] == pytest.approx(0.6)
    # m1 is EVs 0 and 2: 1 of 3 sessions followed; m2 is EVs 1 and 3: 2 of 2.
    assert world.loc["m1", "dispatch_success_rate"] == pytest.approx(1.0 / 3.0)
    assert world.loc["m2", "dispatch_success_rate"] == pytest.approx(1.0)
    # A group with no session ending in the study has no rate.
    none = (np.zeros((1, 4)), np.zeros((1, 4)))
    assert _partners(dispatch_sessions=none)["dispatch_success_rate"].isna().all()


def test_cycles_are_battery_energy_over_capacity() -> None:
    fleet = _partners().iloc[0]
    assert fleet["equivalent_full_cycles_per_ev_per_week_normal"] == pytest.approx(0.2)
    assert fleet["equivalent_full_cycles_per_ev_per_week_selected"] == pytest.approx(0.2)
    assert fleet["equivalent_full_cycles_per_ev_per_week_difference"] == 0.0


def test_makers_appear_only_with_manufacturer_id_and_treated_evs() -> None:
    units = pd.DataFrame({"cohort_id": ["average_uk"] * 3, "control_group": [False, False, True]})
    assert [g[0] for g in supplier.partner_groups(units, ["average_uk"])] == ["fleet", "cohort"]
    units["manufacturer_id"] = ["m1", "m1", "m3"]
    ids = [g[1] for g in supplier.partner_groups(units, ["average_uk"])]
    assert ids == ["fleet", "average_uk", "m1"]  # m3's only EV is a control EV


def test_exceedance_hand_fixture() -> None:
    per_world = supplier.exceedance_per_world(np.array([[-5.0, 4.0, 12.0, 30.0]]))
    assert per_world["share_at_or_above"][0, 10] == 0.5
    assert per_world["floor_top_up_gbp_per_device_per_month"][0, 10] == pytest.approx(5.25)
    assert 1.0 - per_world["share_at_or_above"][0, 0] == 0.25  # the share worse off
    share = per_world["share_at_or_above"][0]
    floor = per_world["floor_top_up_gbp_per_device_per_month"][0]
    assert (np.diff(share) <= 0).all() and (np.diff(floor) >= 0).all()
    step = np.diff(floor)
    assert (step >= 1 - share[:-1] - 1e-12).all() and (step <= 1 - share[1:] + 1e-12).all()


def test_revenue_by_market_adds_to_net_and_lists_unmodelled_markets() -> None:
    rng = np.random.default_rng(3)
    week = pd.DataFrame(rng.normal(size=(5, len(market.BUCKETS))), columns=list(market.BUCKETS))
    week["net_gbp"] = week[list(market.BUCKETS)].sum(axis=1)
    week["world_id"] = np.arange(5)
    frame = supplier.revenue_by_market_summary(week, treated_ev_count=10, season="winter")
    weekly = frame.loc[frame["modelled"] & frame["metric"].eq("gbp_per_week")].set_index("market")
    assert weekly["mean"].drop("net").sum() == pytest.approx(weekly.loc["net", "mean"])
    unmodelled = frame.loc[~frame["modelled"]]
    assert set(unmodelled["market"]) == set(supplier.UNMODELLED_MARKETS)
    assert unmodelled["mean"].isna().all() and (unmodelled["world_count"] == 0).all()


# --------------------------------------------------------------------------
# Real (small) runs: the §9 validator, records and runtime
# --------------------------------------------------------------------------

_START = date(2026, 10, 12)  # a Monday, winter prices, no clock change
_SMALL = {"vehicle_count": 40, "evaluation_world_count": 6}


def run_with_supplier(
    trading_edits: dict | None = None, supplier_edits: dict | None = None
) -> SimpleNamespace:
    """A real run with edited trading and supplier inputs, plus the records it used.

    ``records`` on the returned object are the record values the run used,
    for the validator (a direct ``run_forecast`` result lists no records).
    """

    resolved = assumptions.resolve_values(_SMALL)
    inputs = assumptions.forecast_inputs(resolved, warmup_days=7, study_days=7)
    inputs["trading_assumptions"] = inputs["trading_assumptions"] | (trading_edits or {})
    supplier_inputs = assumptions.supplier_inputs(resolved) | (supplier_edits or {})
    result = forecast.run_forecast(
        assumptions.run_settings(_SMALL, _START),
        assumptions.cohort_fixture(_SMALL),
        **inputs,
        supplier_inputs=supplier_inputs,
    )
    records = {
        **supplier_inputs,
        **inputs["trading_assumptions"],
        "summer_months": inputs["price_assumptions"]["summer_months"],
        "supply_reference_net_demand_gw": inputs["price_assumptions"][
            "supply_reference_net_demand_gw"
        ],
    }
    return SimpleNamespace(**vars(result), records=records)


@cache
def _real() -> SimpleNamespace:
    return run_with_supplier()


def test_real_run_passes_the_supplier_validator() -> None:
    result = _real()
    validate_supplier_frames(result, result.records)
    assert set(result.supplier_pnl_summary["season"]) == {"winter"}
    # Default records: revenue-share mode, fee unset.
    pnl = result.supplier_pnl_world
    assert pnl["platform_fee_gbp"].isna().all() and pnl["net_gain_gbp"].isna().all()
    assert pnl["net_gain_before_fee_gbp"].notna().all()


def test_real_run_with_a_control_group_and_flat_reward_passes() -> None:
    edits = {
        "supplier.customer_reward_mode": 1.0,
        "supplier.customer_reward_gbp_per_ev_per_month": 3.0,
        "supplier.platform_fee_gbp_per_ev_per_month": 1.0,
    }
    result = run_with_supplier({"trading.control_group": 1.0}, edits)
    validate_supplier_frames(result, result.records)
    n_t = int((~result.units["control_group"]).sum())
    assert n_t < 40 and (result.supplier_pnl_world["treated_ev_count"] == n_t).all()
    fleet = result.partner_world.loc[result.partner_world["group_type"].eq("fleet")]
    assert (fleet["enrolled_ev_count"] == n_t).all()


def test_supplier_records_change_no_kernel_frame_price_or_ledger_row() -> None:
    base = _real()
    edits = {
        "supplier.platform_fee_gbp_per_ev_per_month": 2.0,
        "carbon.intensity_slope_gco2_per_kwh_per_gw": 0.0,
    }
    edited = run_with_supplier(supplier_edits=edits)
    for name in (
        "fleet_world_intervals",
        "forecast_prices",
        "evaluation_prices",
        "deviation_world_slot",
        "trading_ledger_world",
        "household_value_summary",
    ):
        pd.testing.assert_frame_equal(getattr(base, name), getattr(edited, name))
    for name in ("hedge_block_world", "partner_world", "household_value_exceedance"):
        pd.testing.assert_frame_equal(getattr(base, name), getattr(edited, name))
    assert not base.carbon_shift_world.equals(edited.carbon_shift_world)
    validate_supplier_frames(edited, edited.records)


def test_supplier_frames_are_deterministic() -> None:
    again = run_with_supplier()
    for name in supplier.FRAME_FIELDS:
        pd.testing.assert_frame_equal(getattr(_real(), name), getattr(again, name))


def test_the_validator_wants_no_supplier_frame_without_the_trading_overlay() -> None:
    validate_supplier_frames(SimpleNamespace(deviation_world_slot=None))
    with pytest.raises(AssertionError, match="None without the trading overlay"):
        validate_supplier_frames(
            SimpleNamespace(deviation_world_slot=None, supplier_pnl_world=pd.DataFrame())
        )


def test_the_supplier_records_are_unset_nan_and_listed_with_the_run() -> None:
    names = {record.name: record for record in assumptions.result_assumptions()}
    for name in supplier.SUPPLIER_RECORDS + supplier.CARBON_RECORDS:
        assert name in names and names[name].evidence == "illustrative"
    assert np.isnan(names["supplier.platform_fee_gbp_per_ev_per_month"].value)
    assert np.isnan(names["supplier.customer_reward_gbp_per_ev_per_month"].value)
    assert names["supplier.customer_reward_mode"].value == 0


@pytest.mark.parametrize(
    ("frame", "column", "message"),
    [
        ("supplier_pnl_world", "net_gain_before_fee_gbp", "before-fee net"),
        ("hedge_block_world", "volume_mwh", "hedge_block_world"),
        ("carbon_shift_world", "co2_smart_kg", "carbon_shift_world"),
        ("partner_world", "gross_flex_gbp_per_week", "partner_world"),
    ],
)
def test_the_validator_catches_a_tampered_frame(frame: str, column: str, message: str) -> None:
    result = _real()
    tampered = getattr(result, frame).copy()
    # A +1 edit to one cell must break an identity the validator recomputes.
    tampered.loc[0, column] += 1.0
    with pytest.raises(AssertionError, match=message):
        validate_supplier_frames(
            SimpleNamespace(**(vars(result) | {frame: tampered})), result.records
        )


def test_departure_soc_is_read_on_both_paths_for_the_same_session() -> None:
    # §4.6: a session that reaches its 48 kWh target on the normal path and
    # leaves early at 30 kWh on the smart path lands in the top bin on
    # departure_soc_percent (48/48 = 100 %) and a lower bin on the smart one.
    slots = build_study_slots(date(2026, 1, 12))
    connected = np.zeros((1, len(slots), 1), dtype=bool)
    connected[0, 12:40, 0] = True
    normal = np.full((1, len(slots), 1), 48.0)
    smart = normal.copy()
    smart[0, 39, 0] = 30.0
    normal[0, 11, 0] = smart[0, 11, 0] = 20.0
    sessions = summaries.find_sessions(connected, normal, np.zeros_like(normal))
    rows = summaries.session_distribution_rows(
        sessions,
        study_slots=slots,
        world_ids=np.array([0]),
        cohort_codes=np.array([0]),
        capacity_kwh=np.array([48.0]),
        target_kwh=np.array([48.0]),
        power_kw=np.array([7.0]),
        efficiency=0.9,
        smart_closing_kwh=smart,
    )
    assert rows["departure_soc_percent"].iat[0] == 19  # 100 % in the last bin
    assert rows["departure_soc_percent_smart"].iat[0] == 12  # 62.5 % in [60, 65)
    bands = summaries.session_distribution_bands(rows, 1, {"fleet": None})
    counts = bands.loc[bands["metric"].str.startswith("departure_soc")].groupby("metric")
    assert counts["world_count"].max().tolist() == [1, 1]


def test_run_forecast_from_assumptions_carries_the_frames_and_passes_validation() -> None:
    result = forecast.run_forecast_from_assumptions(_START, values=_SMALL)
    assert all(getattr(result, name) is not None for name in supplier.FRAME_FIELDS)
    validate_supplier_frames(result)  # records read from result.assumptions
    no_action = forecast.run_forecast_from_assumptions(_START, values=_SMALL, model="no_action")
    assert all(getattr(no_action, name) is None for name in supplier.FRAME_FIELDS)


def test_supplier_inputs_refuse_a_reward_mode_that_is_not_a_switch() -> None:
    # The records are not dialog-editable yet, so the test swaps the record's default.
    record = assumptions.SUPPLIER["supplier.customer_reward_mode"]
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(assumptions.SUPPLIER, record.name, replace(record, value=0.5))
        with pytest.raises(ValueError, match="customer_reward_mode must be 0"):
            assumptions.supplier_inputs()
