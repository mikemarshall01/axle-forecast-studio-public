"""Product frames (lane J5) specs, validators and a SYNTHETIC trading stand-in.

What this owns (trading contract v1 §10.5, §10.9): the column specs of the
J5 frames, validators that recompute each frame from its inputs by a
separate (pandas) route, and ``make_product_case``: J2's SYNTHETIC per-EV
fixture with both paths, run through the availability and product hooks,
plus a toy trading overlay (baseline, settled volume, customer share) so the
settlement file and the §10.5f columns can be checked before J1a/J1b land.

Everything here is SYNTHETIC and illustrative: the baseline is the
unmanaged import plus a seeded offset, the customer share a seeded negative
amount per night; nothing is a settlement or Axle cash.
"""

from __future__ import annotations

import types
import warnings

import numpy as np
import pandas as pd
from fixtures.availability_contract import fixture_totals, make_per_ev_fixture
from fixtures.result_contract import _check_dtypes, _check_unique

from axle_studio.model import availability, product
from axle_studio.model.summaries import allocation_weights, find_sessions

EVIDENCE_KIND = "illustrative_synthetic"
MAKER_CATALOGUE = (
    ("m1", "Maker A", 0.25),
    ("m2", "Maker B", 0.25),
    ("m3", "Maker C", 0.25),
    ("m4", "Maker D", 0.25),
)
"""The §10.1d illustrative makers (labels as the contract; J1a owns the records)."""
TOL = 1e-9

PRODUCT_SHEET_COLUMNS = {
    "window": "str",
    "window_start_local": "str",
    "window_end_local": "str",
    "direction": "str",
    "duration_hours": "float",
    "world_count": "int",
    **{f"window_mean_mw_{q}": "float" for q in ("p05", "p10", "p50", "p90")},
    **{f"window_min_mw_{q}": "float" for q in ("p10", "p50", "p90")},
    "firm_share": "float",
    "firm_share_p05": "float",
    "intraday_firm_mw_p50": "float",
    "intraday_known_share_p50": "float",
    **{f"movable_energy_mwh_{q}": "float" for q in ("p10", "p50", "p90")},
    "recovery_energy_mwh_p50": "float",
    "rebound_capacity_mw_p50": "float",
    "recovery_hours_p50": "float",
    "max_event_length_hours": "float",
    "notice_day_ahead_hours": "float",
    "notice_intraday_hours": "float",
    "ramp_hours": "float",
    "evidence_kind": "str",
}
VALUE_SUMMARY_COLUMNS = {
    "direction": "str",
    "duration_hours": "float",
    "metric": "str",
    "unit": "str",
    "world_count": "int",
    "mean": "float",
    "p10": "float",
    "p50": "float",
    "p90": "float",
    "evidence_kind": "str",
}
SETTLEMENT_FILE_COLUMNS = {
    "meter_point_id": "str",
    "cohort_id": "str",
    "manufacturer_id": "str",
    "zone_id": "str",
    "control_group": "bool",
    "slot_index": "int",
    "night_index": "int",
    "interval_start_utc": "utc",
    "interval_start_london": "london",
    "settlement_date": "str",
    "settlement_period": "int",
    "unmanaged_kwh": "float",
    "metered_kwh": "float",
    "deviation_kwh": "float",
    "settled_kwh": "float",
    "day_ahead_gbp_per_mwh": "float",
    "payment_gbp": "float",
    "evidence_kind": "str",
}
COMPLETION_COLUMNS = {
    "group_id": "str",
    "path_id": "str",
    "departure": "str",
    "ev_count": "int",
    "world_count": "int",
    "session_count_mean": "float",
    **{f"completed_share_{s}": "float" for s in ("mean", "p10", "p50", "p90")},
    "shortfall_kwh_mean": "float",
    "shortfall_kwh_p50": "float",
    "evidence_kind": "str",
}
FIRMNESS_COLUMNS = {
    "manufacturer_id": "str",
    "manufacturer_label": "str",
    "share": "float",
    "ev_count": "int",
    "response_rate": "float",
    "outage_probability_per_night": "float",
    "direction": "str",
    "duration_hours": "float",
    "world_count": "int",
    **{f"firmness_{s}": "float" for s in ("mean", "p10", "p50", "p90")},
    "outage_nights_mean": "float",
    "outage_nights_p50": "float",
    "evidence_kind": "str",
}
MANUFACTURER_SUMMARY_COLUMNS = {
    "manufacturer_id": "str",
    "manufacturer_label": "str",
    "share": "float",
    "ev_count": "int",
    "response_rate": "float",
    "outage_probability_per_night": "float",
    "evening_potential_mw_p50": "float",
    "largest_share": "float",
    "evidence_kind": "str",
}


# --------------------------------------------------------------------------
# SYNTHETIC case: J2's per-EV fixture with both paths and a toy trading overlay
# --------------------------------------------------------------------------


def _with_closing(unit: dict[str, np.ndarray], efficiency: float) -> dict[str, np.ndarray]:
    """Add ``closing_battery_kwh``: the fixture has no driving inside a session."""

    return {
        **unit,
        "closing_battery_kwh": unit["opening_battery_kwh"]
        + efficiency * unit["home_grid_import_kwh"],
    }


def make_product_case(
    *,
    world_count: int = 8,
    vehicle_count: int = 24,
    seed: int = 7,
    control_share: float = 0.0,
    outage_probability=(0.02, 0.02, 0.02, 0.02),
    representative_world_id: int = 1,
) -> types.SimpleNamespace:
    """Run J2's SYNTHETIC fixture through both hooks and a toy trading overlay.

    The normal path is the same fixture with every response rate 0: the
    draws do not depend on the rates, so every session charges by the
    normal rule on identical futures.  The toy overlay (SYNTHETIC): baseline
    ``B = U + offset`` with a seeded offset of up to ±2 kWh per slot,
    ``V = max(0, B - M)`` everywhere, the ``full`` strategy's customer share a
    seeded negative amount per night, and a seeded day-ahead price with some
    negative half-hours.  Returns a namespace with the frames, the product
    totals and the inputs the validators need.
    """

    kwargs = dict(
        world_count=world_count,
        vehicle_count=vehicle_count,
        seed=seed,
        control_share=control_share,
        outage_probability=outage_probability,
    )
    selected_fixture = make_per_ev_fixture(**kwargs)
    normal_fixture = make_per_ev_fixture(**kwargs, response_rate=(0.0, 0.0, 0.0, 0.0))
    study_slots = selected_fixture.study_slots
    slot_count = len(study_slots)
    availability_totals = fixture_totals(selected_fixture)
    frames = availability.build_frames(availability_totals, study_slots)

    control = selected_fixture.draws["control"]
    cohorts = np.where(np.arange(vehicle_count) % 2 == 0, "cohort_a", "cohort_b").astype(object)
    groups = {"fleet": np.ones(vehicle_count, dtype=bool)}
    groups.update({c: cohorts == c for c in ("cohort_a", "cohort_b")})
    totals = product.new_totals(
        availability_totals, representative_world_id=representative_world_id, groups=groups
    )
    unmanaged = np.zeros((world_count, slot_count))
    metered = np.zeros((world_count, slot_count))
    per_ev_normal = np.zeros((world_count, slot_count, vehicle_count))
    per_ev_selected = np.zeros_like(per_ev_normal)
    for first in range(0, world_count, 5):
        worlds = np.arange(first, min(first + 5, world_count))
        by_path = {
            "normal": _with_closing(normal_fixture.unit(worlds), selected_fixture.efficiency),
            "selected": _with_closing(selected_fixture.unit(worlds), selected_fixture.efficiency),
        }
        sessions = {
            path: find_sessions(
                unit["connected"], unit["closing_battery_kwh"], unit["home_grid_import_kwh"]
            )
            for path, unit in by_path.items()
        }
        product.accumulate(totals, worlds, by_path, sessions)
        per_ev_normal[worlds] = by_path["normal"]["home_grid_import_kwh"]
        per_ev_selected[worlds] = by_path["selected"]["home_grid_import_kwh"]
    unmanaged = per_ev_normal.sum(axis=2)
    metered = per_ev_selected.sum(axis=2)

    rng = np.random.default_rng(seed + 1)
    baseline = unmanaged + rng.uniform(-2.0, 2.0, unmanaged.shape)
    settled = np.maximum(0.0, baseline - metered)
    day_ahead = rng.normal(80.0, 60.0, unmanaged.shape)
    night_count = int(study_slots["night_index"].max()) + 1
    customer_share = -rng.uniform(1.0, 5.0, (world_count, night_count))
    deviation = pd.DataFrame(
        {
            "world_id": np.repeat(np.arange(world_count, dtype=np.int64), slot_count),
            "slot_index": np.tile(np.arange(slot_count, dtype=np.int64), world_count),
            "baseline_kwh": baseline.ravel(),
            "unmanaged_kwh": unmanaged.ravel(),
            "metered_kwh": metered.ravel(),
            "true_reduction_kwh": (unmanaged - metered).ravel(),
            "baseline_effect_kwh": (baseline - unmanaged).ravel(),
            "settled_kwh": settled.ravel(),
            "day_ahead_gbp_per_mwh": day_ahead.ravel(),
        }
    )
    ledger = pd.DataFrame(
        {
            "world_id": np.repeat(np.arange(world_count), night_count * 2),
            "strategy": np.tile(np.repeat(["da_only", "full"], night_count), world_count),
            "night_index": np.tile(np.arange(night_count), world_count * 2),
            "customer_revenue_share_gbp": np.stack(
                [customer_share * 0.5, customer_share], axis=1
            ).ravel(),
        }
    )
    trading = types.SimpleNamespace(deviation_world_slot=deviation, trading_ledger_world=ledger)
    london = pd.DatetimeIndex(study_slots["interval_start_london"])
    positions = pd.DataFrame(
        {
            "slot_index": study_slots["slot_index"].to_numpy(dtype=np.int64),
            "interval_start_utc": study_slots["interval_start_utc"],
            "interval_start_london": study_slots["interval_start_london"],
            "settlement_date": pd.Series(list(london.strftime("%Y-%m-%d")), dtype=object),
            "settlement_period": (
                (london - london.normalize()) // pd.Timedelta(minutes=30)
            ).to_numpy()
            + 1,
            "night_index": study_slots["night_index"].to_numpy(dtype=np.int64),
            "settled_mw_p50": np.quantile(settled / 0.5 / 1000, 0.5, axis=0),
            "price_curve_source": "synthetic",
            "evidence_kind": EVIDENCE_KIND,
        }
    )
    units = pd.DataFrame(
        {
            "unit_id": pd.Series([f"ev_{i:03d}" for i in range(vehicle_count)], dtype=object),
            "cohort_id": pd.Series(cohorts, dtype=object),
            "zone_id": pd.Series([f"zone_{1 + i % 4}" for i in range(vehicle_count)], dtype=object),
            "control_group": control,
        }
    )
    # §9.3c allocation driver: each EV's positive true reduction in the
    # night's settled slots.
    night = study_slots["night_index"].to_numpy()
    night_starts = np.flatnonzero(np.diff(night, prepend=-1))
    reduction = np.where(
        (settled > 0)[:, :, np.newaxis], np.maximum(per_ev_normal - per_ev_selected, 0.0), 0.0
    )
    reduction = np.moveaxis(np.add.reduceat(reduction, night_starts, axis=1), 1, 2)
    weights = allocation_weights(reduction, ~control)
    built = product.build_frames(
        totals,
        frames={**frames, "supplier_positions": positions},
        trading=trading,
        units=units,
        weights=weights[representative_world_id],
        manufacturer_catalogue=MAKER_CATALOGUE,
        seed=seed,
        start_local_date=study_slots["local_date"].iat[0],
        day_ahead_decision_local_hour=13.0,
    )
    # summaries._trading_summaries adds the §10.5f columns on every action run.
    built["supplier_positions"] = product.supplier_position_additions(
        positions, deviation, frames["availability_bands"]
    )
    return types.SimpleNamespace(
        fixture=selected_fixture,
        study_slots=study_slots,
        availability_totals=availability_totals,
        totals=totals,
        frames=frames,
        built=built,
        trading=trading,
        positions=positions,
        units=units,
        weights=weights,
        customer_share=customer_share,
        per_ev_normal=per_ev_normal,
        per_ev_selected=per_ev_selected,
        day_ahead=day_ahead,
        representative_world_id=representative_world_id,
        seed=seed,
    )


# --------------------------------------------------------------------------
# Validators (§10.9)
# --------------------------------------------------------------------------


def _slot_labels(world_slot: pd.DataFrame) -> pd.Series:
    return world_slot["interval_start_london"].dt.strftime("%H:%M")


def _in_window(labels: pd.Series, start: str, end: str) -> pd.Series:
    if start < end:
        return (labels >= start) & (labels < end)
    return (labels >= start) | (labels < end)


def _quantiles(values: pd.Series, levels) -> list[float]:
    kept = values.dropna().to_numpy(dtype=float)
    if len(kept) == 0:
        return [np.nan] * len(levels)
    return list(np.quantile(kept, levels, method="linear"))


def _close(got, expected, name: str) -> None:
    assert np.allclose(
        np.asarray(got, dtype=float), np.asarray(expected, dtype=float), atol=1e-9, equal_nan=True
    ), f"{name}: {got} != {expected}"


def validate_product_sheet(
    sheet: pd.DataFrame,
    world_slot: pd.DataFrame,
    *,
    day_ahead_decision_local_hour: float = 13.0,
    decision_local_time: str = "17:00",
) -> None:
    """§10.5a recomputed with pandas group-bys from ``availability_world_slot``."""

    _check_dtypes(sheet, PRODUCT_SHEET_COLUMNS, "product_sheet")
    _check_unique(sheet, ["window", "direction", "duration_hours"], "product_sheet")
    assert len(sheet) == 24, "product_sheet has 24 rows"
    assert (sheet["evidence_kind"] == EVIDENCE_KIND).all()
    frame = world_slot.assign(label=_slot_labels(world_slot))
    decision = int(decision_local_time[:2]) + int(decision_local_time[3:]) / 60
    for row in sheet.itertuples(index=False):
        rows = frame.loc[
            _in_window(frame["label"], row.window_start_local, row.window_end_local)
            & frame["direction"].eq(row.direction)
            & frame["duration_hours"].eq(row.duration_hours)
        ]
        by_world = rows.groupby("world_id")
        # pandas mean/min skip NaN: the O5 rule (slots past the end left out).
        mean_mw = by_world["deliverable_kw"].mean() / 1000
        min_mw = by_world["deliverable_kw"].min() / 1000
        name = f"product_sheet {row.window} {row.direction} {row.duration_hours}"
        assert row.world_count == mean_mw.notna().sum(), f"{name} world_count"
        _close(
            [row.window_mean_mw_p05, row.window_mean_mw_p10, row.window_mean_mw_p50],
            _quantiles(mean_mw, (0.05, 0.1, 0.5)),
            f"{name} window mean",
        )
        _close(row.window_mean_mw_p90, _quantiles(mean_mw, (0.9,))[0], f"{name} p90")
        _close(
            [row.window_min_mw_p10, row.window_min_mw_p50, row.window_min_mw_p90],
            _quantiles(min_mw, (0.1, 0.5, 0.9)),
            f"{name} window min",
        )
        assert (min_mw <= mean_mw + 1e-12).all(), f"{name}: window_min <= window_mean per world"
        p50 = row.window_mean_mw_p50
        _close(row.firm_share, row.window_mean_mw_p10 / p50 if p50 > 0 else np.nan, name)
        _close(row.firm_share_p05, row.window_mean_mw_p05 / p50 if p50 > 0 else np.nan, name)
        firm = by_world["intraday_p10_kw"].mean() / 1000
        _close(row.intraday_firm_mw_p50, _quantiles(firm, (0.5,))[0], f"{name} intraday firm")
        known = by_world["intraday_known_mean_kw"].sum()
        total = by_world["intraday_mean_kw"].sum()
        share = (known / total.where(total > 0)).where(total > 0)
        _close(row.intraday_known_share_p50, _quantiles(share, (0.5,))[0], f"{name} known share")
        if not np.isnan(row.intraday_known_share_p50):
            assert 0 <= row.intraday_known_share_p50 <= 1 + 1e-12, f"{name} known share in [0, 1]"
        if row.duration_hours == 0.5:
            movable = by_world["deliverable_kw"].sum() * 0.5 / 1000
            _close(
                [
                    row.movable_energy_mwh_p10,
                    row.movable_energy_mwh_p50,
                    row.movable_energy_mwh_p90,
                ],
                _quantiles(movable, (0.1, 0.5, 0.9)),
                f"{name} movable",
            )
        else:
            assert np.isnan(row.movable_energy_mwh_p50), f"{name} movable NaN beyond 0.5 h"
        if row.direction == "turn_down":
            energy = row.duration_hours * mean_mw
            rebound = by_world["eligible_power_kw"].mean() / 1000
            _close(row.recovery_energy_mwh_p50, _quantiles(energy, (0.5,))[0], f"{name} energy")
            _close(row.rebound_capacity_mw_p50, _quantiles(rebound, (0.5,))[0], f"{name} rebound")
            hours = (energy / rebound.where(rebound > 0)).where(rebound > 0)
            _close(row.recovery_hours_p50, _quantiles(hours, (0.5,))[0], f"{name} recovery")
        else:
            assert np.isnan(row.recovery_energy_mwh_p50) and np.isnan(row.recovery_hours_p50)
            assert np.isnan(row.rebound_capacity_mw_p50)
        start = int(row.window_start_local[:2]) + int(row.window_start_local[3:]) / 60
        start = start + 24 if start < 12 else start
        _close(row.notice_day_ahead_hours, start + 24 - day_ahead_decision_local_hour, name)
        _close(row.notice_intraday_hours, start - decision, name)
        assert row.ramp_hours == 0.5
    for (_, _), block in sheet.groupby(["window", "direction"], sort=False):
        base = block["window_mean_mw_p50"].iat[0]
        qualifies = block["window_mean_mw_p50"] >= 0.25 * base
        expected = block["duration_hours"][qualifies].max() if base > 0 else np.nan
        assert (block["max_event_length_hours"].nunique(dropna=False)) == 1
        _close(block["max_event_length_hours"].iat[0], expected, "max_event_length 25 % rule")


def validate_value_summary(summary: pd.DataFrame, world_slot: pd.DataFrame, day_ahead) -> None:
    """§10.5c recomputed per world with pandas; NaN rule of O5."""

    _check_dtypes(summary, VALUE_SUMMARY_COLUMNS, "availability_value_summary")
    _check_unique(summary, ["direction", "duration_hours", "metric"], "availability_value_summary")
    assert len(summary) == 24
    world_count = int(world_slot["world_id"].max()) + 1
    price = pd.DataFrame(
        {
            "world_id": np.repeat(np.arange(world_count), day_ahead.shape[1]),
            "slot_index": np.tile(np.arange(day_ahead.shape[1]), world_count),
            "price": np.asarray(day_ahead).ravel(),
        }
    )
    frame = world_slot.merge(price, on=["world_id", "slot_index"]).dropna(subset=["deliverable_kw"])
    for row in summary.itertuples(index=False):
        rows = frame.loc[
            frame["direction"].eq(row.direction) & frame["duration_hours"].eq(row.duration_hours)
        ]
        sign = 1.0 if row.direction == "turn_down" else -1.0
        weight = (sign * rows["price"]).clip(lower=0.0)
        by_world = rows.assign(w=weight, aw=weight * rows["deliverable_kw"]).groupby("world_id")
        if row.metric == "price_weighted_mw":
            sums = by_world[["w", "aw"]].sum()
            values = (sums["aw"] / sums["w"].where(sums["w"] > 0)) / 1000
        elif row.metric == "simple_mean_mw":
            values = by_world["deliverable_kw"].mean() / 1000
        else:
            values = by_world["aw"].sum() * 0.5 / 1000
            assert values.notna().all(), "value at day-ahead is 0, never NaN"
        name = f"value summary {row.direction} {row.duration_hours} {row.metric}"
        assert row.world_count == values.notna().sum(), f"{name} world_count"
        _close(row.mean, values.mean(), f"{name} mean")
        _close([row.p10, row.p50, row.p90], _quantiles(values, (0.1, 0.5, 0.9)), name)


def validate_settlement_file(
    file: pd.DataFrame,
    *,
    deviation_world: pd.DataFrame,
    customer_share: np.ndarray,
    vehicle_count: int,
) -> None:
    """§10.5d sums: EV rows to ``1[V > 0] R``, all rows to ``V``, the fleet row to ``E``."""

    _check_dtypes(file, SETTLEMENT_FILE_COLUMNS, "settlement_file")
    slot_count = len(deviation_world)
    assert len(file) == (vehicle_count + 1) * slot_count, "(N + 1) x slots rows"
    _check_unique(file, ["meter_point_id", "slot_index"], "settlement_file")
    fleet = file["meter_point_id"].eq(product.BASELINE_EFFECT_ID)
    assert fleet.sum() == slot_count and fleet.iloc[-slot_count:].all(), "fleet rows last"
    for column in ("cohort_id", "manufacturer_id", "zone_id"):
        assert (file.loc[fleet, column] == "fleet").all(), f"{column} is fleet on the fleet row"
    assert not file.loc[fleet, "control_group"].any()
    assert file.loc[fleet, ["unmanaged_kwh", "metered_kwh"]].isna().all().all()
    assert (file.loc[fleet, "payment_gbp"] == 0).all()
    deviation = deviation_world.sort_values("slot_index").reset_index(drop=True)
    settles = deviation["settled_kwh"].to_numpy() > 0
    ev = file.loc[~fleet]
    by_slot = ev.groupby("slot_index", sort=True)
    _close(
        by_slot["settled_kwh"].sum().to_numpy(),
        np.where(settles, deviation["true_reduction_kwh"], 0.0),
        "EV settled sums to 1[V > 0] R",
    )
    _close(
        file.groupby("slot_index", sort=True)["settled_kwh"].sum().to_numpy(),
        deviation["settled_kwh"].to_numpy(),
        "all rows sum to V",
    )
    _close(
        file.loc[fleet, "deviation_kwh"].to_numpy(),
        deviation["baseline_effect_kwh"].to_numpy(),
        "fleet row deviation is E",
    )
    _close(by_slot["unmanaged_kwh"].sum().to_numpy(), deviation["unmanaged_kwh"], "U sums")
    _close(by_slot["metered_kwh"].sum().to_numpy(), deviation["metered_kwh"], "M sums")
    _close(
        file.groupby("night_index", sort=True)["payment_gbp"].sum().to_numpy(),
        -np.asarray(customer_share),
        "payment per night is the customer share",
    )
    assert (file.loc[file["control_group"], "payment_gbp"] == 0).all(), "control EVs are not paid"


def validate_completion(summary: pd.DataFrame, group_ids) -> None:
    """§10.5e: row order and keys, ``all = on_time + early`` counts, shares in [0, 1].

    ``path_ids`` is read off the summary itself (``product.PATH_IDS``, plus
    the optional "timed" path, decision 0007, model step 2, when the run
    completes it), not assumed fixed at two, so the row-order check is exact
    either way.
    """

    _check_dtypes(summary, COMPLETION_COLUMNS, "charge_completion_summary")
    path_ids = list(dict.fromkeys(summary["path_id"]))
    expected = [(g, p, d) for g in group_ids for p in path_ids for d in product.DEPARTURES]
    assert list(zip(summary["group_id"], summary["path_id"], summary["departure"])) == expected
    counts = summary.set_index(["group_id", "path_id", "departure"])["session_count_mean"]
    for g in group_ids:
        for p in path_ids:
            _close(
                counts[(g, p, "all")],
                counts[(g, p, "on_time")] + counts[(g, p, "early")],
                "all = on_time + early",
            )
    shares = summary[[f"completed_share_{s}" for s in ("mean", "p10", "p50", "p90")]]
    kept = shares.to_numpy()[~np.isnan(shares.to_numpy())]
    assert ((kept >= 0) & (kept <= 1)).all(), "completed shares in [0, 1]"


def validate_firmness(
    firmness: pd.DataFrame, manufacturer_world: pd.DataFrame, makers: pd.DataFrame
) -> None:
    """§10.5g recomputed with pandas from ``availability_manufacturer_world``."""

    _check_dtypes(firmness, FIRMNESS_COLUMNS, "firmness_by_manufacturer")
    assert len(firmness) == 8 * len(makers)
    sums = manufacturer_world.groupby(
        ["manufacturer_id", "direction", "duration_hours", "world_id"]
    )[["realised_mwh", "potential_mwh", "outage"]].sum()
    for row in firmness.itertuples(index=False):
        block = sums.loc[(row.manufacturer_id, row.direction, row.duration_hours)]
        potential = block["potential_mwh"]
        values = (block["realised_mwh"] / potential.where(potential > 0)).where(potential > 0)
        name = f"firmness {row.manufacturer_id} {row.direction} {row.duration_hours}"
        assert row.world_count == values.notna().sum(), f"{name} world_count"
        _close(row.firmness_mean, values.mean(), f"{name} mean")
        _close(
            [row.firmness_p10, row.firmness_p50, row.firmness_p90],
            _quantiles(values, (0.1, 0.5, 0.9)),
            name,
        )
        _close(row.outage_nights_mean, block["outage"].mean(), f"{name} outage nights")
        record = makers.set_index("manufacturer_id").loc[row.manufacturer_id]
        assert row.ev_count == record["ev_count"] and row.share == record["share"]


def validate_manufacturer_summary(summary: pd.DataFrame, makers: pd.DataFrame) -> None:
    """§10.5h: maker rows then ``fleet``; ``P(any outage) = 1 - Π(1 - π_m)``."""

    _check_dtypes(summary, MANUFACTURER_SUMMARY_COLUMNS, "manufacturer_summary")
    assert list(summary["manufacturer_id"]) == [*makers["manufacturer_id"], "fleet"]
    fleet = summary.iloc[-1]
    pi = makers["outage_probability_per_night"].to_numpy()
    _close(fleet["outage_probability_per_night"], 1 - np.prod(1 - pi), "P(any outage)")
    assert np.isnan(fleet["response_rate"])
    assert fleet["ev_count"] == makers["ev_count"].sum()
    assert (summary["largest_share"] == makers["share"].max()).all()


def validate_position_additions(
    positions: pd.DataFrame, deviation: pd.DataFrame, bands: pd.DataFrame
) -> None:
    """§10.5f: the new columns follow ``settled_mw_p50`` and equal their recomputation."""

    columns = list(positions.columns)
    at = columns.index("settled_mw_p50")
    assert columns[at + 1 : at + 7] == product.POSITION_ADDITION_COLUMNS
    wide = deviation.pivot(index="world_id", columns="slot_index")
    unmanaged = wide["unmanaged_kwh"].to_numpy() / 0.5 / 1000
    change = wide["metered_kwh"].to_numpy() / 0.5 / 1000 - unmanaged
    _close(positions["unmanaged_mw_p50"], np.median(unmanaged, axis=0), "unmanaged p50")
    for name, level in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        _close(
            positions[f"net_change_mw_{name}"],
            np.quantile(change, level, axis=0),
            f"net change {name}",
        )
    for direction in ("turn_down", "turn_up"):
        rows = bands.query(
            "statistic == 'realised' and direction == @direction and duration_hours == 1.0"
        )
        _close(
            positions[f"deliverable_{direction}_1h_mw_p10"],
            rows.sort_values("slot_index")["p10"] / 1000,
            f"firm {direction}",
        )


def validate_case(case: types.SimpleNamespace) -> None:
    """Every §10.9 rule for the J5 frames of a ``make_product_case`` run."""

    built = case.built
    world_slot = case.frames["availability_world_slot"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        validate_product_sheet(built["product_sheet"], world_slot)
    validate_value_summary(built["availability_value_summary"], world_slot, case.day_ahead)
    rep = case.representative_world_id
    deviation = case.trading.deviation_world_slot
    validate_settlement_file(
        built["settlement_file"],
        deviation_world=deviation.loc[deviation["world_id"].eq(rep)],
        customer_share=case.customer_share[rep],
        vehicle_count=len(case.units),
    )
    validate_completion(built["charge_completion_summary"], case.totals["group_ids"])
    inputs = case.fixture.inputs
    makers = product.manufacturer_table(
        MAKER_CATALOGUE, inputs.manufacturer_index, inputs.response_rate, inputs.outage_probability
    )
    validate_firmness(
        built["firmness_by_manufacturer"], case.frames["availability_manufacturer_world"], makers
    )
    validate_manufacturer_summary(built["manufacturer_summary"], makers)
    validate_position_additions(
        built["supplier_positions"], deviation, case.frames["availability_bands"]
    )


__all__ = [
    "MAKER_CATALOGUE",
    "make_product_case",
    "validate_case",
    "validate_completion",
    "validate_firmness",
    "validate_manufacturer_summary",
    "validate_position_additions",
    "validate_product_sheet",
    "validate_settlement_file",
    "validate_value_summary",
]
