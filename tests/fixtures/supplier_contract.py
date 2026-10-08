"""Supplier P&L and Partners (lane S1) frame specs and validator (supplier contract v1 §9).

``validate_supplier_frames(result)`` checks the ten ``supplier.FRAME_FIELDS``
frames of one result against the contract: exact columns and dtypes, row
order and keys, and the §3.5, §3.7, §3.8, §4.3, §4.5 and §4.7 identities
recomputed independently from the frames the result already carries
(``deviation_world_slot``, ``cost_effect``, ``trading_week_world``,
``trading_ledger_world``, ``units``, ``study_slots``, ``forecast_prices``,
``shape_premium_summary``, ``household_value_summary``) and the run's
records.  It is duck-typed (attributes and columns only), so the lead's one
hook line in ``validate_result_v2`` can call it on real results and on any
fixture that carries the frames.  It raises ``AssertionError`` naming the
frame and the rule.

Tolerance is ``1e-9 × max(1, |value|)`` unless stated (contract §10).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from fixtures.result_contract import (
    TRADING_MONEY,
    _check_dtypes,
    _check_ordered_quantiles,
    _check_unique,
)

from axle_studio.model import supplier

# The ledger's seven buckets above the customer share (trading contract §4.7),
# written out so the check does not reuse the model's own constant.
GROSS_BUCKETS = (
    "day_ahead_revenue_gbp",
    "intraday_pnl_gbp",
    "trading_cost_gbp",
    "imbalance_gbp",
    "baseline_effect_gbp",
    "grid_event_payment_gbp",
    "supplier_compensation_gbp",
)
STATS = {"world_count": "int", "mean": "float", "p10": "float", "p50": "float", "p90": "float"}
HORIZONS = ("intraday", "day_ahead", "week_ahead", "scenario")
SEASONS = ("winter", "summer")

SUPPLIER_PNL_WORLD_COLUMNS = {
    "world_id": "int",
    "hedge_variant": "str",
    "energy_cost_unmanaged_gbp": "float",
    "energy_cost_smart_gbp": "float",
    "energy_saving_gbp": "float",
    "shape_saving_gbp": "float",
    "volume_value_gbp": "float",
    "forecast_reduction_mwh": "float",
    "realised_reduction_mwh": "float",
    "hedge_error_saving_gbp": "float",
    "worth_of_profiled_hedge_gbp": "float",
    "grid_event_payment_gbp": "float",
    "gross_gain_gbp": "float",
    "customer_payment_gbp": "float",
    "net_gain_before_fee_gbp": "float",
    "platform_fee_gbp": "float",
    "net_gain_gbp": "float",
    "treated_ev_count": "int",
    "net_gain_before_fee_per_customer_per_month_gbp": "float",
    "net_gain_per_customer_per_month_gbp": "float",
    "customer_payment_per_customer_per_month_gbp": "float",
    "platform_fee_per_customer_per_month_gbp": "float",
    "evidence_kind": "str",
}
SUPPLIER_PNL_SUMMARY_COLUMNS = {
    "hedge_variant": "str",
    "metric": "str",
    "unit": "str",
    "horizon": "str",
    "season": "str",
    **STATS,
    "evidence_kind": "str",
}
HEDGE_BLOCK_WORLD_COLUMNS = {
    "world_id": "int",
    "block": "str",
    "path_id": "str",
    "block_hours": "float",
    "volume_mwh": "float",
    "mean_mw": "float",
    "evidence_kind": "str",
}
HEDGE_BLOCK_SUMMARY_COLUMNS = {
    "block": "str",
    "path_id": "str",
    "metric": "str",
    "unit": "str",
    "horizon": "str",
    "season": "str",
    "block_hours": "float",
    **STATS,
    "evidence_kind": "str",
}
CARBON_WORLD_COLUMNS = {
    "world_id": "int",
    "co2_unmanaged_kg": "float",
    "co2_smart_kg": "float",
    "co2_shifted_kg": "float",
    "co2_shifted_kg_per_ev_per_month": "float",
    "evidence_kind": "str",
}
CARBON_SUMMARY_COLUMNS = {
    "metric": "str",
    "unit": "str",
    "horizon": "str",
    "season": "str",
    **STATS,
    "evidence_kind": "str",
}
PARTNER_WORLD_COLUMNS = {
    "world_id": "int",
    "group_type": "str",
    "group_id": "str",
    "enrolled_ev_count": "int",
    "earning_ev_count": "int",
    **dict.fromkeys(supplier.PARTNER_METRIC_UNITS, "float"),
    "evidence_kind": "str",
}
PARTNER_SUMMARY_COLUMNS = {
    "group_type": "str",
    "group_id": "str",
    "metric": "str",
    "unit": "str",
    "horizon": "str",
    "season": "str",
    "enrolled_ev_count": "int",
    **STATS,
    "evidence_kind": "str",
}
EXCEEDANCE_COLUMNS = {
    "group_type": "str",
    "group_id": "str",
    "metric": "str",
    "threshold_gbp_per_month": "int",
    "unit": "str",
    "horizon": "str",
    "season": "str",
    "enrolled_ev_count": "int",
    **STATS,
    "evidence_kind": "str",
}
REVENUE_BY_MARKET_COLUMNS = {
    "market": "str",
    "modelled": "bool",
    "metric": "str",
    "unit": "str",
    "horizon": "str",
    "season": "str",
    **STATS,
    "evidence_kind": "str",
}


def _close(actual, expected, name: str) -> None:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    tolerance = 1e-9 * np.maximum(1.0, np.abs(np.nan_to_num(expected)))
    same_nan = np.isnan(actual) == np.isnan(expected)
    assert same_nan.all(), f"{name}: NaN pattern differs"
    diff = np.abs(np.nan_to_num(actual) - np.nan_to_num(expected))
    assert (diff <= tolerance).all(), f"{name}: max difference {diff.max():.3g}"


def _stats(values) -> np.ndarray:
    kept = np.asarray(values, dtype=float)
    kept = kept[~np.isnan(kept)]
    if len(kept) == 0:
        return np.array([0.0, np.nan, np.nan, np.nan, np.nan])
    q = np.quantile(kept, (0.1, 0.5, 0.9), method="linear")
    return np.array([len(kept), kept.mean(), *q])


def _check_row_stats(row: pd.Series, values, name: str) -> None:
    _close(row[list(STATS)].to_numpy(dtype=float), _stats(values), name)


def _matrix(frame: pd.DataFrame, column: str, world_count: int) -> np.ndarray:
    ordered = frame.sort_values(["world_id", "slot_index"], kind="stable")
    return ordered[column].to_numpy(dtype=float).reshape(world_count, -1)


RECORD_NAMES = (
    *supplier.SUPPLIER_RECORDS,
    *supplier.CARBON_RECORDS,
    "trading.customer_revenue_share",
    "supply_reference_net_demand_gw",
)


def run_records(result) -> dict[str, object] | None:
    """The record values a result used, by name, from ``result.assumptions``.

    ``None`` when the result does not list them all (a direct ``run_forecast``
    call or the synthetic fixture's short record list): the record-dependent
    identities (payment rule, fee, carbon intensity) are then skipped and the
    rest still checked.
    """

    values = {record.name: record.value for record in getattr(result, "assumptions", None) or ()}
    return values if all(name in values for name in RECORD_NAMES) else None


def validate_supplier_frames(result, records: Mapping[str, object] | None = None) -> None:
    """Assert the supplier contract v1 §9 rules on ``result``'s supplier frames.

    ``records`` are the run's record values by name (``SUPPLIER``,
    ``CARBON``, ``trading.customer_revenue_share``, ``summer_months`` and
    ``supply_reference_net_demand_gw``); ``None`` reads them from
    ``result.assumptions``.
    """

    frames = {name: getattr(result, name, None) for name in supplier.FRAME_FIELDS}
    if getattr(result, "deviation_world_slot", None) is None:
        assert all(frame is None for frame in frames.values()), (
            "supplier frames must be None without the trading overlay (no-action result)"
        )
        return
    assert all(frame is not None for frame in frames.values()), "every supplier frame is present"
    records = run_records(result) if records is None else records
    world_count = int(result.world_count)
    _validate_pnl(result, frames, records, world_count)
    _validate_hedge_blocks(result, frames, world_count)
    _validate_carbon(result, frames, records, world_count)
    _validate_partners(result, frames, world_count)
    _validate_exceedance(result, frames)
    _validate_revenue_by_market(result, frames)
    _validate_manufacturer_segments(result)
    seasons = {
        season
        for name, frame in frames.items()
        if name.endswith(("_summary", "_exceedance"))
        for season in frame["season"]
    }
    assert len(seasons) == 1 and seasons <= set(SEASONS), f"one run season, got {seasons}"
    for name, frame in frames.items():
        assert (frame["evidence_kind"] == supplier.EVIDENCE_KIND).all(), f"{name} evidence_kind"
        if "horizon" in frame:
            assert frame["horizon"].isin(HORIZONS).all(), f"{name} horizon vocabulary"


def _treated_count(units: pd.DataFrame) -> int:
    if "control_group" in units:
        return int((~units["control_group"].to_numpy(dtype=bool)).sum())
    return len(units)


def _validate_pnl(result, frames, records, world_count: int) -> None:
    name = "supplier_pnl_world"
    world = frames[name]
    _check_dtypes(world, SUPPLIER_PNL_WORLD_COLUMNS, name)
    keys = [(w, v) for w in range(world_count) for v in supplier.HEDGE_VARIANTS]
    assert list(zip(world["world_id"], world["hedge_variant"], strict=True)) == keys, (
        f"{name} rows are world-major, profiled then flat"
    )
    deviation = result.deviation_world_slot

    def slots(column):
        return _matrix(deviation, column, world_count)

    u, m, r = slots("unmanaged_kwh"), slots("metered_kwh"), slots("true_reduction_kwh")
    x_u, x = slots("expected_unmanaged_kwh"), slots("expected_metered_kwh")
    p, sip = slots("day_ahead_gbp_per_mwh"), slots("imbalance_gbp_per_mwh")
    f = x_u - x
    n_t = _treated_count(result.units)
    cost = result.cost_effect.sort_values("world_id")
    week = result.trading_week_world
    events = week.loc[week["strategy"].eq("full")].sort_values("world_id")
    if records is not None:
        share = float(records["trading.customer_revenue_share"])
        mode = int(records["supplier.customer_reward_mode"])
        reward = float(records["supplier.customer_reward_gbp_per_ev_per_month"])
        fee = float(records["supplier.platform_fee_gbp_per_ev_per_month"])

    hedge_free = [
        "energy_cost_unmanaged_gbp",
        "energy_cost_smart_gbp",
        "energy_saving_gbp",
        "shape_saving_gbp",
        "volume_value_gbp",
        "forecast_reduction_mwh",
        "realised_reduction_mwh",
        "worth_of_profiled_hedge_gbp",
        "grid_event_payment_gbp",
        "platform_fee_gbp",
        "treated_ev_count",
    ]
    profiled = world.loc[world["hedge_variant"].eq("profiled")].reset_index(drop=True)
    flat = world.loc[world["hedge_variant"].eq("flat")].reset_index(drop=True)
    for column in hedge_free:
        assert profiled[column].equals(flat[column]), f"{name}.{column} same in both variants"

    # §3.5 identity 1: exact reuse of the cost-effect column, and the recomputation.
    saving = profiled["energy_saving_gbp"].to_numpy(dtype=float)
    reused = -cost["illustrative_selected_minus_normal_energy_cost_gbp"].to_numpy(dtype=float)
    assert np.array_equal(saving, reused), f"{name}.energy_saving_gbp must reuse cost_effect"
    _close(saving, ((u - m) * p).sum(axis=1) / 1000.0, f"{name} energy recomputation")
    _close(profiled["energy_cost_unmanaged_gbp"], (u * p).sum(axis=1) / 1000.0, f"{name} U cost")
    _close(profiled["energy_cost_smart_gbp"], (m * p).sum(axis=1) / 1000.0, f"{name} M cost")
    # Identity 2: shape + volume, and the shape saving is minus §9.3b's difference row.
    _close(
        profiled["shape_saving_gbp"] + profiled["volume_value_gbp"],
        saving,
        f"{name} shape + volume",
    )
    _close(
        profiled["volume_value_gbp"],
        p.mean(axis=1) * (u.sum(axis=1) - m.sum(axis=1)) / 1000.0,
        f"{name} volume value",
    )
    premium = result.shape_premium_summary
    difference = premium.loc[
        premium["path_id"].eq("difference") & premium["metric"].eq("shape_cost_gbp_per_week")
    ].iloc[0]
    _check_row_stats(difference, -profiled["shape_saving_gbp"], f"{name} shape vs §9.3b")
    # Identities 3-4: the hedge-error components recomputed.
    _close(profiled["forecast_reduction_mwh"], f.sum(axis=1) / 1000.0, f"{name} forecast volume")
    _close(profiled["forecast_reduction_mwh"], 0.0 * f.sum(axis=1), f"{name} sum F = 0")
    _close(profiled["realised_reduction_mwh"], r.sum(axis=1) / 1000.0, f"{name} realised volume")
    expected_hedge = {
        "profiled": ((r - f) * (sip - p)).sum(axis=1) / 1000.0,
        "flat": (r * (sip - p)).sum(axis=1) / 1000.0,
    }
    for variant, rows in (("profiled", profiled), ("flat", flat)):
        _close(rows["hedge_error_saving_gbp"], expected_hedge[variant], f"{name} {variant} hedge")
    _close(
        profiled["worth_of_profiled_hedge_gbp"],
        expected_hedge["profiled"] - expected_hedge["flat"],
        f"{name} worth of the profiled hedge",
    )
    _close(
        profiled["grid_event_payment_gbp"],
        events["grid_event_payment_gbp"],
        f"{name} grid events from the ledger",
    )
    assert (profiled["grid_event_payment_gbp"] >= 0.0).all(), f"{name} grid events >= 0"
    # Identity 5: components, signs and NaN exactly where a term is unset.
    for variant, rows in (("profiled", profiled), ("flat", flat)):
        where = f"{name} {variant}"
        gross = saving + expected_hedge[variant] + rows["grid_event_payment_gbp"]
        _close(rows["gross_gain_gbp"], gross, f"{where} gross")
        payment = rows["customer_payment_gbp"].to_numpy(dtype=float)
        if records is not None and mode == 0:
            _close(payment, -share * np.maximum(gross, 0.0), f"{where} revenue-share payment")
        elif records is not None:
            _close(payment, np.full(world_count, -reward * n_t * 12.0 / 52.0), f"{where} flat")
        if records is not None:
            assert np.isnan(payment).all() == (mode == 1 and np.isnan(reward)), f"{where} NaN"
        assert (payment[~np.isnan(payment)] <= 0.0).all(), f"{where} customer payment <= 0"
        _close(rows["net_gain_before_fee_gbp"], gross + payment, f"{where} before-fee net")
        platform = rows["platform_fee_gbp"].to_numpy(dtype=float)
        if records is not None:
            assert np.isnan(platform).all() == np.isnan(fee), f"{where} fee NaN exactly when unset"
        assert (platform[~np.isnan(platform)] <= 0.0).all(), f"{where} platform fee <= 0"
        _close(rows["net_gain_gbp"], gross + payment + platform, f"{where} net")
        # Identity 6: per-customer columns use the treated count.
        assert (rows["treated_ev_count"] == n_t).all(), f"{where} treated count"
        for per_customer, weekly in (
            ("net_gain_before_fee_per_customer_per_month_gbp", "net_gain_before_fee_gbp"),
            ("net_gain_per_customer_per_month_gbp", "net_gain_gbp"),
            ("customer_payment_per_customer_per_month_gbp", "customer_payment_gbp"),
            ("platform_fee_per_customer_per_month_gbp", "platform_fee_gbp"),
        ):
            _close(rows[per_customer], rows[weekly] * 52.0 / 12.0 / n_t, f"{where} {per_customer}")
        if records is not None and not np.isnan(fee):
            _close(rows["platform_fee_per_customer_per_month_gbp"], -fee, f"{where} fee per EV")

    name = "supplier_pnl_summary"
    summary = frames[name]
    _check_dtypes(summary, SUPPLIER_PNL_SUMMARY_COLUMNS, name)
    metrics = [c for c in list(SUPPLIER_PNL_WORLD_COLUMNS)[2:-1] if c != "treated_ev_count"]
    keys = [(v, metric) for v in supplier.HEDGE_VARIANTS for metric in metrics]
    assert list(zip(summary["hedge_variant"], summary["metric"], strict=True)) == keys, (
        f"{name} rows"
    )
    _check_ordered_quantiles(summary, "p10", "p50", "p90", name)
    for _, row in summary.iterrows():
        per_customer = row["metric"].endswith("_per_customer_per_month_gbp")
        assert row["horizon"] == ("scenario" if per_customer else "week_ahead"), f"{name} horizon"
        values = world.loc[world["hedge_variant"].eq(row["hedge_variant"]), row["metric"]]
        _check_row_stats(row, values, f"{name} {row['hedge_variant']} {row['metric']}")
    for variant in supplier.HEDGE_VARIANTS:
        mean = summary.loc[summary["hedge_variant"].eq(variant)].set_index("metric")["mean"]
        parts = mean[
            [
                "energy_saving_gbp",
                "hedge_error_saving_gbp",
                "grid_event_payment_gbp",
                "customer_payment_gbp",
            ]
        ]
        if not np.isnan(mean["net_gain_before_fee_gbp"]):
            _close(parts.sum(), mean["net_gain_before_fee_gbp"], f"{name} means add (before fee)")
        if not np.isnan(mean["net_gain_gbp"]):
            _close(
                parts.sum() + mean["platform_fee_gbp"], mean["net_gain_gbp"], f"{name} means add"
            )


def _validate_hedge_blocks(result, frames, world_count: int) -> None:
    name = "hedge_block_world"
    world = frames[name]
    _check_dtypes(world, HEDGE_BLOCK_WORLD_COLUMNS, name)
    keys = [
        (w, b, p) for w in range(world_count) for b in supplier.BLOCKS for p in supplier.BLOCK_PATHS
    ]
    assert list(zip(world["world_id"], world["block"], world["path_id"], strict=True)) == keys, (
        f"{name} rows"
    )
    # Block hours recomputed from study_slots, independently of the model.
    london = pd.DatetimeIndex(result.study_slots["interval_start_london"])
    minutes = london.hour * 60 + london.minute
    peak = np.asarray((london.dayofweek <= 4) & (minutes >= 7 * 60) & (minutes <= 18 * 60 + 30))
    hours = {
        "baseload": 0.5 * len(london),
        "peak": 0.5 * peak.sum(),
        "off_peak": 0.5 * (~peak).sum(),
    }
    deviation = result.deviation_world_slot
    u = _matrix(deviation, "unmanaged_kwh", world_count)
    m = _matrix(deviation, "metered_kwh", world_count)
    masks = {"baseload": np.ones(len(london), bool), "peak": peak, "off_peak": ~peak}
    for block in supplier.BLOCKS:
        rows = world.loc[world["block"].eq(block)]
        assert (rows["block_hours"] == hours[block]).all(), f"{name} {block} hours"
        by_path = {
            p: rows.loc[rows["path_id"].eq(p)].reset_index(drop=True)
            for p in ("normal", "selected", "difference")
        }
        _close(
            by_path["normal"]["volume_mwh"],
            u[:, masks[block]].sum(axis=1) / 1000.0,
            f"{name} {block} normal",
        )
        _close(
            by_path["selected"]["volume_mwh"],
            m[:, masks[block]].sum(axis=1) / 1000.0,
            f"{name} {block} selected",
        )
        _close(
            by_path["difference"]["volume_mwh"],
            by_path["selected"]["volume_mwh"] - by_path["normal"]["volume_mwh"],
            f"{name} {block} difference",
        )
        _close(rows["mean_mw"] * rows["block_hours"], rows["volume_mwh"], f"{name} {block} mean MW")
    volume = world.set_index(["world_id", "block", "path_id"])["volume_mwh"].unstack("block")
    _close(
        volume["baseload"],
        volume["peak"] + volume["off_peak"],
        f"{name} baseload = peak + off-peak",
    )

    name = "hedge_block_summary"
    summary = frames[name]
    _check_dtypes(summary, HEDGE_BLOCK_SUMMARY_COLUMNS, name)
    metrics = ("volume_mwh", "mean_mw", "volume_mwh_per_month")
    keys = [
        (b, p, metric) for b in supplier.BLOCKS for p in supplier.BLOCK_PATHS for metric in metrics
    ]
    assert (
        list(zip(summary["block"], summary["path_id"], summary["metric"], strict=True)) == keys
    ), f"{name} rows"
    for _, row in summary.iterrows():
        cell = world.loc[world["block"].eq(row["block"]) & world["path_id"].eq(row["path_id"])]
        column = "mean_mw" if row["metric"] == "mean_mw" else "volume_mwh"
        scale = 52.0 / 12.0 if row["metric"] == "volume_mwh_per_month" else 1.0
        _check_row_stats(row, cell[column] * scale, f"{name} {row['block']} {row['metric']}")
        assert row["block_hours"] == hours[row["block"]], f"{name} block hours"
        expected = "scenario" if row["metric"] == "volume_mwh_per_month" else "week_ahead"
        assert row["horizon"] == expected, f"{name} horizon"


def _validate_carbon(result, frames, records, world_count: int) -> None:
    name = "carbon_shift_world"
    world = frames[name]
    _check_dtypes(world, CARBON_WORLD_COLUMNS, name)
    assert world["world_id"].tolist() == list(range(world_count)), f"{name} rows"
    deviation = result.deviation_world_slot
    u = _matrix(deviation, "unmanaged_kwh", world_count)
    m = _matrix(deviation, "metered_kwh", world_count)
    r = _matrix(deviation, "true_reduction_kwh", world_count)
    _close(
        world["co2_shifted_kg"],
        world["co2_unmanaged_kg"] - world["co2_smart_kg"],
        f"{name} shifted",
    )
    demand = _matrix(result.forecast_prices, "system_net_demand_gw", world_count)
    intensity = (
        None
        if records is None
        else np.clip(
            float(records["carbon.intensity_at_reference_gco2_per_kwh"])
            + float(records["carbon.intensity_slope_gco2_per_kwh_per_gw"])
            * (demand - float(records["supply_reference_net_demand_gw"])),
            float(records["carbon.intensity_floor_gco2_per_kwh"]),
            float(records["carbon.intensity_cap_gco2_per_kwh"]),
        )
    )
    if intensity is not None:
        _close(world["co2_unmanaged_kg"], (u * intensity).sum(axis=1) / 1000.0, f"{name} U")
        _close(world["co2_smart_kg"], (m * intensity).sum(axis=1) / 1000.0, f"{name} smart")
        _close(world["co2_shifted_kg"], (r * intensity).sum(axis=1) / 1000.0, f"{name} sum R I")
    n_t = _treated_count(result.units)
    _close(
        world["co2_shifted_kg_per_ev_per_month"],
        world["co2_shifted_kg"] * 52.0 / 12.0 / n_t,
        f"{name} per EV per month",
    )
    name = "carbon_shift_summary"
    summary = frames[name]
    _check_dtypes(summary, CARBON_SUMMARY_COLUMNS, name)
    assert summary["metric"].tolist() == list(CARBON_WORLD_COLUMNS)[1:-1], f"{name} rows"
    for _, row in summary.iterrows():
        _check_row_stats(row, world[row["metric"]], f"{name} {row['metric']}")


def _validate_partners(result, frames, world_count: int) -> None:
    name = "partner_world"
    world = frames[name]
    _check_dtypes(world, PARTNER_WORLD_COLUMNS, name)
    units = result.units
    treated = (
        ~units["control_group"].to_numpy(dtype=bool)
        if "control_group" in units
        else np.ones(len(units), bool)
    )
    cohorts = units["cohort_id"].to_numpy(dtype=object)
    order = list(dict.fromkeys(result.cohort_summary["cohort_id"]))
    groups = [("fleet", "fleet", treated)]
    groups += [
        ("cohort", c, treated & (cohorts == c)) for c in order if (treated & (cohorts == c)).any()
    ]
    if "manufacturer_id" in units:
        makers = units["manufacturer_id"].to_numpy(dtype=object)
        groups += [
            ("manufacturer", mk, treated & (makers == mk)) for mk in sorted(set(makers[treated]))
        ]
    else:
        assert not world["group_type"].eq("manufacturer").any(), (
            f"{name} makers need manufacturer_id"
        )
    keys = [(w, t, g) for w in range(world_count) for t, g, _ in groups]
    got = list(zip(world["world_id"], world["group_type"], world["group_id"], strict=True))
    assert got == keys, f"{name} rows are world-major, fleet, cohorts, makers"
    _check_unique(world, ["world_id", "group_type", "group_id"], name)
    charger = units["home_charger_limit_kw"].to_numpy(dtype=float)
    ledger = result.trading_ledger_world
    full = ledger.loc[ledger["strategy"].eq("full")].sort_values(["world_id", "night_index"])
    gross = full[list(GROSS_BUCKETS)].to_numpy(dtype=float).sum(axis=1)
    gross_week = gross.reshape(world_count, -1).sum(axis=1)
    for group_type, group_id, members in groups:
        rows = world.loc[world["group_type"].eq(group_type) & world["group_id"].eq(group_id)]
        where = f"{name} {group_id}"
        assert (rows["enrolled_ev_count"] == members.sum()).all(), f"{where} enrolled count"
        assert (rows["earning_ev_count"] <= rows["enrolled_ev_count"]).all(), f"{where} earning"
        assert rows["share_earning"].between(0.0, 1.0).all(), f"{where} share 0-1"
        earning = rows["earning_ev_count"] > 0
        _close(
            rows.loc[earning, "gross_flex_gbp_per_enrolled_device_per_month"],
            rows.loc[earning, "share_earning"]
            * rows.loc[earning, "gross_flex_gbp_per_earning_device_per_month"],
            f"{where} per enrolled = share x per earning",
        )
        assert rows.loc[~earning, "gross_flex_gbp_per_earning_device_per_month"].isna().all(), (
            f"{where} per earning NaN with no earner"
        )
        _close(
            rows["gbp_per_kw_charger_per_year"] * charger[members].sum() / 52.0,
            rows["gross_flex_gbp_per_week"],
            f"{where} per kW",
        )
        rate = rows["dispatch_success_rate"]
        assert (rate.isna() | rate.between(0.0, 1.0)).all(), f"{where} dispatch rate"
        # The kernel's plan_status (firm-MW J1b) feeds the rate on a real
        # run, which also carries the firm-MW frames; the contract fixture
        # has neither, so its rate is NaN everywhere.  A real run's group
        # may still be NaN when none of its sessions ends in the study.
        if getattr(result, "availability_world_slot", None) is None:
            assert rate.isna().all(), f"{where} dispatch success rate NaN without plan_status"
        elif group_id == "fleet":
            assert rate.notna().any(), f"{where} dispatch success rate filled from plan_status"
        cycles = rows[[c for c in PARTNER_WORLD_COLUMNS if c.startswith("equivalent_full_cycles")]]
        assert (cycles.iloc[:, :2] >= 0.0).all().all(), f"{where} cycles >= 0"
        _close(
            cycles.iloc[:, 2], cycles.iloc[:, 1] - cycles.iloc[:, 0], f"{where} cycle difference"
        )
    # §4.7 identity 1: cohort and maker groups each sum to the fleet, which is the ledger gross.
    fleet = world.loc[world["group_type"].eq("fleet"), "gross_flex_gbp_per_week"].to_numpy()
    _close(fleet, gross_week, f"{name} fleet gross = ledger seven-bucket gross")
    for group_type in ("cohort", "manufacturer"):
        rows = world.loc[world["group_type"].eq(group_type)]
        if len(rows):
            _close(
                rows.groupby("world_id")["gross_flex_gbp_per_week"].sum(),
                fleet,
                f"{name} {group_type} sum",
            )
    # Identity 5: the fleet's customer value is household_value_summary's mean value.
    household = result.household_value_summary
    mean_value = household.loc[
        household["group_id"].eq("fleet") & household["statistic"].eq("mean_value")
    ].iloc[0]
    fleet_value = world.loc[
        world["group_type"].eq("fleet"), "customer_value_gbp_per_device_per_month"
    ]
    _check_row_stats(mean_value, fleet_value, f"{name} customer value vs household_value_summary")

    name = "partner_summary"
    summary = frames[name]
    _check_dtypes(summary, PARTNER_SUMMARY_COLUMNS, name)
    keys = [(t, g, metric) for t, g, _ in groups for metric in supplier.PARTNER_METRIC_UNITS]
    assert (
        list(zip(summary["group_type"], summary["group_id"], summary["metric"], strict=True))
        == keys
    ), f"{name} rows"
    for _, row in summary.iterrows():
        cell = world.loc[
            world["group_type"].eq(row["group_type"]) & world["group_id"].eq(row["group_id"])
        ]
        _check_row_stats(row, cell[row["metric"]], f"{name} {row['group_id']} {row['metric']}")
        assert row["unit"] == supplier.PARTNER_METRIC_UNITS[row["metric"]], f"{name} unit"


def _validate_exceedance(result, frames) -> None:
    name = "household_value_exceedance"
    frame = frames[name]
    _check_dtypes(frame, EXCEEDANCE_COLUMNS, name)
    thresholds = list(range(41))
    groups = list(
        dict.fromkeys(
            zip(
                frames["partner_world"]["group_type"],
                frames["partner_world"]["group_id"],
                strict=True,
            )
        )
    )
    metrics = ("share_at_or_above", "floor_top_up_gbp_per_device_per_month")
    keys = [(t, g, metric, x) for t, g in groups for metric in metrics for x in thresholds]
    got = list(
        zip(
            frame["group_type"],
            frame["group_id"],
            frame["metric"],
            frame["threshold_gbp_per_month"],
            strict=True,
        )
    )
    assert got == keys, f"{name} rows: groups, metrics, thresholds 0-40 ascending"
    assert (frame["horizon"] == "scenario").all(), f"{name} horizon"
    household = result.household_value_summary.set_index(["group_id", "statistic"])
    for (group_type, group_id), cell in frame.groupby(["group_type", "group_id"], sort=False):
        share = cell.loc[cell["metric"].eq("share_at_or_above")]
        floor = cell.loc[cell["metric"].ne("share_at_or_above")]
        where = f"{name} {group_id}"
        for column in ("mean", "p10", "p50", "p90"):
            s = share[column].to_numpy(dtype=float)
            fl = floor[column].to_numpy(dtype=float)
            assert (np.diff(s) <= 1e-12).all(), f"{where} share {column} non-increasing"
            assert (np.diff(fl) >= -1e-12).all(), f"{where} floor {column} non-decreasing"
            assert ((s >= -1e-12) & (s <= 1 + 1e-12)).all(), f"{where} share {column} 0-1"
        # Step bounds hold per world and therefore for the mean (linear).
        s, fl = share["mean"].to_numpy(dtype=float), floor["mean"].to_numpy(dtype=float)
        step = np.diff(fl)
        assert (step >= 1.0 - s[:-1] - 1e-9).all() and (step <= 1.0 - s[1:] + 1e-9).all(), (
            f"{where} floor steps between 1 - share(X) and 1 - share(X + 1)"
        )
        # 1 - share(0) is the share worse off (complements), so the means agree.
        if group_type != "manufacturer" and (group_id, "share_worse_off") in household.index:
            worse = household.loc[(group_id, "share_worse_off"), "mean"]
            _close(1.0 - s[0], worse, f"{where} share_worse_off identity")


def _validate_revenue_by_market(result, frames) -> None:
    name = "revenue_by_market_summary"
    frame = frames[name]
    _check_dtypes(frame, REVENUE_BY_MARKET_COLUMNS, name)
    markets = [*supplier.MARKET_BUCKETS, *supplier.UNMODELLED_MARKETS]
    metrics = ("gbp_per_week", "gbp_per_enrolled_device_per_month")
    keys = [(mk, metric) for mk in markets for metric in metrics]
    assert list(zip(frame["market"], frame["metric"], strict=True)) == keys, f"{name} rows"
    unmodelled = frame.loc[~frame["modelled"]]
    assert set(unmodelled["market"]) == set(supplier.UNMODELLED_MARKETS), f"{name} unmodelled"
    assert unmodelled[["mean", "p10", "p50", "p90"]].isna().all().all(), f"{name} unmodelled NaN"
    assert (unmodelled["world_count"] == 0).all(), f"{name} unmodelled world_count 0"
    week = result.trading_week_world
    full = week.loc[week["strategy"].eq("full")].sort_values("world_id")
    n_t = _treated_count(result.units)
    for _, row in frame.loc[frame["modelled"]].iterrows():
        values = (
            full[list(supplier.MARKET_BUCKETS[row["market"]])].to_numpy(dtype=float).sum(axis=1)
        )
        if row["metric"] == "gbp_per_enrolled_device_per_month":
            values = values / n_t * 52.0 / 12.0
        _check_row_stats(row, values, f"{name} {row['market']} {row['metric']}")
    weekly = frame.loc[frame["modelled"] & frame["metric"].eq("gbp_per_week")].set_index("market")[
        "mean"
    ]
    _close(weekly.drop("net").sum(), weekly["net"], f"{name} market means add to net")


def _validate_manufacturer_segments(result) -> None:
    """§4.4: maker segments of ``revenue_by_segment`` exist exactly for makers with treated EVs.

    The frame stores statistics across worlds only, so the "sum to ``all``
    per world" rule is checked on the means (linear in the per-world values).
    """

    name = "revenue_by_segment"
    segments = result.revenue_by_segment
    units = result.units
    treated = (
        ~units["control_group"].to_numpy(dtype=bool)
        if "control_group" in units
        else np.ones(len(units), bool)
    )
    expected = (
        sorted(set(units["manufacturer_id"].to_numpy(dtype=object)[treated]))
        if "manufacturer_id" in units
        else []
    )
    weekly = segments.loc[segments["metric"].eq("gbp_per_week")]
    for bucket in TRADING_MONEY:
        rows = weekly.loc[weekly["bucket"].eq(bucket)]
        makers = rows.loc[rows["segment_type"].eq("manufacturer")]
        assert makers["segment_id"].tolist() == expected, f"{name} maker segments for {bucket}"
        if expected:
            all_mean = rows.loc[rows["segment_type"].eq("all"), "mean"].iat[0]
            _close(makers["mean"].sum(), all_mean, f"{name} {bucket} makers sum to all")
