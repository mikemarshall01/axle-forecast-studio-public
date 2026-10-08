"""SYNTHETIC household frames and the household-v1 validator (household contract v1 §3.1-§3.2, §8).

``make_household_frames(result)`` builds ``household_ev_world`` and
``household_outcomes_summary`` for the synthetic ``make_result`` fixture
from its per-EV toy arrays (``result.replay_state``), written independently
of ``model/household.py``'s chunk hook: sessions are found by a plain loop
over each EV's connected runs.  The fixture's saving and value per EV are
the fixture's own equal split of each world's cost effect and customer
share (``result_fixture._household_frames``), so its frames agree with its
``household_value_summary``; ``earning`` is a SYNTHETIC stand-in (any
half-hour where the smart path imports less).  The summary itself comes
from ``household.outcomes_summary``; the validator recomputes it here.

``validate_household_v2(result)`` checks §8 on the fixture and on real
``ForecastResult`` objects alike (duck-typed).  Every figure is
illustrative and synthetic.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from axle_studio.model import household
from axle_studio.model.summaries import price_third_bands

TOL_KWH = 1e-6
TOL_GBP = 1e-6
EVIDENCE = "illustrative_synthetic"
_MONTHLY = 52.0 / 12.0
_READY_TOLERANCE_KWH = 1e-6  # NOT_RECOVERED_TOLERANCE_KWH (contract §3.1)

EV_WORLD_COLUMNS = {
    "world_id": "int64",
    "unit_id": "object",
    "cohort_id": "object",
    "treated": "bool",
    "earning": "bool",
    "home_import_kwh_normal": "float64",
    "home_import_kwh_selected": "float64",
    "cheap_import_kwh_normal": "float64",
    "cheap_import_kwh_selected": "float64",
    "public_import_kwh_normal": "float64",
    "public_import_kwh_selected": "float64",
    "home_cost_gbp_normal": "float64",
    "home_cost_gbp_selected": "float64",
    "public_cost_gbp_normal": "float64",
    "public_cost_gbp_selected": "float64",
    "saving_gbp": "float64",
    "value_gbp_per_month": "float64",
    "supplier_energy_saving_gbp": "float64",
    "session_count": "int64",
    "session_end_count": "int64",
    "nights_plugged_count": "int64",
    "completed_count_normal": "int64",
    "completed_count_selected": "int64",
    "sessions_affected_count": "int64",
    "departure_soc_mean_normal": "float64",
    "departure_soc_mean_selected": "float64",
    "co2_shifted_kg": "float64",
    "evening_turn_down_kw_1h": "float64",
    "evening_turn_up_kw_1h": "float64",
    "evidence_kind": "object",
}
SUMMARY_COLUMNS = {
    "group_id": "object",
    "metric": "object",
    "statistic": "object",
    "unit": "object",
    "horizon": "object",
    "better_is": "object",
    "ev_count": "int64",
    "ev_value_count": "int64",
    "world_count": "int64",
    "mean": "float64",
    "p10": "float64",
    "p50": "float64",
    "p90": "float64",
    "evidence_kind": "object",
}

# The §3.2 table, restated here so the validator does not trust the model's
# own copy: metric -> (unit, horizon, better_is, set R rule).  The rule is
# (numerator column, denominator column or None for N_n per EV), or
# ("difference", selected metric, normal metric), or None (not a ratio).
METRIC_TABLE = {
    "value_gbp_per_week": ("GBP per week", "week_ahead", "higher", None),
    "value_gbp_per_month": ("GBP per household per month", "scenario", "higher", None),
    "value_gbp_per_year": ("GBP per year", "scenario", "higher", None),
    "saving_gbp_per_week": ("GBP per week", "week_ahead", "higher", None),
    "saving_gbp_per_year": ("GBP per year", "scenario", "higher", None),
    "supplier_energy_saving_gbp_per_week": ("GBP per week", "week_ahead", "higher", None),
    "supplier_energy_saving_gbp_per_month": (
        "GBP per customer per month",
        "scenario",
        "higher",
        None,
    ),
    "home_cost_gbp_per_kwh_normal": (
        "GBP per kWh",
        "week_ahead",
        "lower",
        ("home_cost_gbp_normal", "home_import_kwh_normal"),
    ),
    "home_cost_gbp_per_kwh_selected": (
        "GBP per kWh",
        "week_ahead",
        "lower",
        ("home_cost_gbp_selected", "home_import_kwh_selected"),
    ),
    "home_cost_gbp_per_kwh_difference": (
        "GBP per kWh",
        "week_ahead",
        "lower",
        ("difference", "home_cost_gbp_per_kwh_selected", "home_cost_gbp_per_kwh_normal"),
    ),
    "home_import_kwh_per_week_normal": ("kWh per week", "week_ahead", "", None),
    "home_import_kwh_per_week_selected": ("kWh per week", "week_ahead", "", None),
    "public_import_kwh_per_week_normal": ("kWh per week", "week_ahead", "lower", None),
    "public_import_kwh_per_week_selected": ("kWh per week", "week_ahead", "lower", None),
    "cheap_share_normal": (
        "fraction",
        "week_ahead",
        "higher",
        ("cheap_import_kwh_normal", "home_import_kwh_normal"),
    ),
    "cheap_share_selected": (
        "fraction",
        "week_ahead",
        "higher",
        ("cheap_import_kwh_selected", "home_import_kwh_selected"),
    ),
    "cheap_share_difference": (
        "fraction",
        "week_ahead",
        "higher",
        ("difference", "cheap_share_selected", "cheap_share_normal"),
    ),
    "completed_share_normal": (
        "fraction",
        "week_ahead",
        "higher",
        ("completed_count_normal", "session_count"),
    ),
    "completed_share_selected": (
        "fraction",
        "week_ahead",
        "higher",
        ("completed_count_selected", "session_count"),
    ),
    "completed_share_difference": (
        "fraction",
        "week_ahead",
        "higher",
        ("difference", "completed_share_selected", "completed_share_normal"),
    ),
    "sessions_per_week": ("sessions per week", "week_ahead", "", None),
    "session_ends_per_week": ("session ends per week", "week_ahead", "", None),
    "sessions_affected_count": ("session ends per week", "week_ahead", "lower", None),
    "sessions_affected_share": (
        "fraction",
        "week_ahead",
        "lower",
        ("sessions_affected_count", "session_end_count"),
    ),
    "nights_plugged_share": ("fraction", "week_ahead", "", ("nights_plugged_count", None)),
    "departure_soc_mean_normal": ("percent", "week_ahead", "higher", None),
    "departure_soc_mean_selected": ("percent", "week_ahead", "higher", None),
    "co2_shifted_kg_per_week": ("kg CO2 per week", "week_ahead", "higher", None),
    "co2_shifted_kg_per_month": ("kg CO2 per customer per month", "scenario", "higher", None),
    "evening_turn_down_kw_1h": ("kW", "week_ahead", "higher", None),
    "evening_turn_up_kw_1h": ("kW", "week_ahead", "higher", None),
}


def _metric_table(paths) -> dict:
    """``METRIC_TABLE``, plus the timed path's own rows when ``paths`` names it.

    Decision 0007, model step 2: mirrors ``household.household_metrics``, so
    the fixture's independent recomputation and the model agree on which
    metrics a run with the timed path carries, with no "timed vs normal"
    difference row (that reading stays selected vs normal).
    """

    table = dict(METRIC_TABLE)
    if "timed" in paths:
        table.update(
            {
                "home_cost_gbp_per_kwh_timed": (
                    "GBP per kWh",
                    "week_ahead",
                    "lower",
                    ("home_cost_gbp_timed", "home_import_kwh_timed"),
                ),
                "home_import_kwh_per_week_timed": ("kWh per week", "week_ahead", "", None),
                "public_import_kwh_per_week_timed": ("kWh per week", "week_ahead", "lower", None),
                "cheap_share_timed": (
                    "fraction",
                    "week_ahead",
                    "higher",
                    ("cheap_import_kwh_timed", "home_import_kwh_timed"),
                ),
                "completed_share_timed": (
                    "fraction",
                    "week_ahead",
                    "higher",
                    ("completed_count_timed", "session_count"),
                ),
                "departure_soc_mean_timed": ("percent", "week_ahead", "higher", None),
            }
        )
    return table


SET_A = ("mean_across_evs", "p10_across_evs", "p50_across_evs", "p90_across_evs")
SET_B = (
    "mean_of_ev_means",
    "p10_of_ev_means",
    "p50_of_ev_means",
    "p90_of_ev_means",
    "share_of_ev_means_below_zero",
)


# --------------------------------------------------------------------------
# Fixture builder
# --------------------------------------------------------------------------


def runs(connected: np.ndarray) -> list[tuple[int, int]]:
    """(first slot, one past the last slot) of each run of True in a 1-D array."""

    out, start = [], None
    for t, value in enumerate(connected):
        if value and start is None:
            start = t
        elif not value and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, len(connected)))
    return out


def price_matrix(result) -> np.ndarray:
    """(world, slot) day-ahead GBP/MWh of a result's ``forecast_prices``."""

    prices = result.forecast_prices.sort_values(["world_id", "slot_index"], kind="stable")
    return (
        prices["wholesale_forecast_gbp_per_mwh"]
        .to_numpy(dtype=float)
        .reshape(result.world_count, -1)
    )


def low_third(price: np.ndarray) -> np.ndarray:
    """Independent restatement of the R§4.6 low band: strictly below the world's 1/3 quantile."""

    return price < np.quantile(price, 1 / 3, axis=1, keepdims=True)


def night_count(result) -> int:
    return int(result.study_slots["night_index"].max()) + 1


def _groups(result) -> dict[str, np.ndarray]:
    """Treated-EV masks: fleet, then the cohorts of ``household_value_summary`` in its order."""

    units = result.units
    treated = _treated(units)
    cohorts = units["cohort_id"].to_numpy(dtype=object)
    groups = {"fleet": treated}
    for group_id in dict.fromkeys(result.household_value_summary["group_id"]):
        if group_id != "fleet":
            groups[group_id] = treated & (cohorts == group_id)
    return groups


def _treated(units: pd.DataFrame) -> np.ndarray:
    if "control_group" in units:
        return ~units["control_group"].to_numpy(dtype=bool)
    return np.ones(len(units), dtype=bool)


def _public_rate(result) -> float | None:
    """The run's public charge rate (GBP/kWh); ``None`` for a direct run without records."""

    for record in result.assumptions or ():
        if record.name == "public_charge_gbp_per_kwh":
            return float(record.value)
    return None


def carbon_intensity(result) -> np.ndarray | None:
    """(world, slot) gCO2/kWh by S§3.8's rule, restated here; ``None`` without the records.

    ``clip(I_ref + s (D - D_ref), floor, cap)`` on the day-ahead frame's
    system net demand ``D`` (GW).  Illustrative and synthetic.
    """

    records = {record.name: record.value for record in result.assumptions or ()}
    names = (
        "carbon.intensity_at_reference_gco2_per_kwh",
        "carbon.intensity_slope_gco2_per_kwh_per_gw",
        "carbon.intensity_floor_gco2_per_kwh",
        "carbon.intensity_cap_gco2_per_kwh",
        "supply_reference_net_demand_gw",
    )
    if getattr(result, "carbon_shift_world", None) is None or not all(n in records for n in names):
        return None
    prices = result.forecast_prices.sort_values(["world_id", "slot_index"], kind="stable")
    demand = prices["system_net_demand_gw"].to_numpy(dtype=float).reshape(result.world_count, -1)
    reference, slope, floor, cap, demand_ref = (float(records[n]) for n in names)
    return np.clip(reference + slope * (demand - demand_ref), floor, cap)


def make_household_frames(result) -> dict[str, pd.DataFrame | None]:
    """SYNTHETIC ``household_ev_world`` and ``household_outcomes_summary`` for a fixture result.

    ``None`` for both on a no-action result (contract §3.1).
    """

    if result.model != "action" or result.household_value_summary is None:
        return {"household_ev_world": None, "household_outcomes_summary": None}
    sim = result.replay_state
    units = result.units
    slots = result.study_slots
    world_count, vehicle_count = result.world_count, result.vehicle_count
    slot_count = len(slots)
    night = slots["night_index"].to_numpy()
    price = price_matrix(result)
    low = low_third(price)
    rate = _public_rate(result)
    intensity = carbon_intensity(result)
    capacity = units["physical_capacity_kwh"].to_numpy(dtype=float)
    target = capacity * units["preferred_target_soc_percent"].to_numpy(dtype=float) / 100.0
    treated = _treated(units)

    # The fixture's equal split (result_fixture._household_frames), per world.
    cost = result.cost_effect.sort_values("world_id")
    saving_world = -cost["illustrative_selected_minus_normal_total_gbp"].to_numpy()
    full = result.trading_ledger_world.loc[result.trading_ledger_world["strategy"].eq("full")]
    share_world = -full.groupby("world_id")["customer_revenue_share_gbp"].sum().to_numpy()

    rows = []
    for w in range(world_count):
        for e in range(vehicle_count):
            connected = sim.connected["normal"][w, :, e]
            row = {
                "world_id": w,
                "unit_id": units["unit_id"].iat[e],
                "cohort_id": units["cohort_id"].iat[e],
                "treated": bool(treated[e]),
            }
            home = {p: sim.home_import_kwh[p][w, :, e] for p in ("normal", "selected")}
            row["earning"] = bool((home["normal"] - home["selected"] > 0.0).any())
            for p in ("normal", "selected"):
                row[f"home_import_kwh_{p}"] = home[p].sum()
                row[f"cheap_import_kwh_{p}"] = home[p][low[w]].sum()
                row[f"public_import_kwh_{p}"] = sim.public_import_kwh[p][w, :, e].sum()
                row[f"home_cost_gbp_{p}"] = (home[p] * price[w]).sum() / 1000.0
                row[f"public_cost_gbp_{p}"] = row[f"public_import_kwh_{p}"] * rate
            per_ev_saving = saving_world[w] / vehicle_count * _MONTHLY
            per_ev_share = share_world[w] / vehicle_count * _MONTHLY
            row["saving_gbp"] = per_ev_saving / _MONTHLY
            row["value_gbp_per_month"] = per_ev_saving + per_ev_share
            row["supplier_energy_saving_gbp"] = (
                row["home_cost_gbp_normal"] - row["home_cost_gbp_selected"]
            )
            all_runs = runs(connected)
            closed = [(a, b) for a, b in all_runs if a > 0 and b < slot_count]
            stock = {p: sim.closing_kwh[p][w, :, e] for p in ("normal", "selected")}
            row["session_count"] = len(closed)
            row["session_end_count"] = len(all_runs)
            row["nights_plugged_count"] = len(set(night[connected]))
            for p in ("normal", "selected"):
                ends = [stock[p][b - 1] for _, b in closed]
                row[f"completed_count_{p}"] = sum(
                    s >= target[e] - _READY_TOLERANCE_KWH for s in ends
                )
                row[f"departure_soc_mean_{p}"] = (
                    float(np.mean([100.0 * s / capacity[e] for s in ends])) if ends else np.nan
                )
            row["sessions_affected_count"] = sum(
                stock["normal"][b - 1] - stock["selected"][b - 1] > _READY_TOLERANCE_KWH
                for _, b in all_runs
            )
            row["co2_shifted_kg"] = (
                np.nan
                if intensity is None
                else ((home["normal"] - home["selected"]) * intensity[w]).sum() / 1000.0
            )
            row["evening_turn_down_kw_1h"] = np.nan
            row["evening_turn_up_kw_1h"] = np.nan
            row["evidence_kind"] = EVIDENCE
            rows.append(row)
    ev_world = pd.DataFrame(rows).loc[:, list(EV_WORLD_COLUMNS)].astype(EV_WORLD_COLUMNS)
    return {
        "household_ev_world": ev_world,
        "household_outcomes_summary": household.outcomes_summary(
            ev_world, groups=_groups(result), night_count=night_count(result)
        ),
    }


# --------------------------------------------------------------------------
# Validator (§8)
# --------------------------------------------------------------------------


def _close(actual, expected, tol: float, message: str) -> None:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    same_nan = np.isnan(actual) == np.isnan(expected)
    assert same_nan.all(), f"{message}: NaN pattern"
    kept = ~np.isnan(actual)
    bound = tol * np.maximum(1.0, np.abs(expected[kept]))
    assert (np.abs(actual[kept] - expected[kept]) <= bound).all(), message


def _check_dtypes(frame, spec: dict[str, str], name: str) -> None:
    assert isinstance(frame, pd.DataFrame), f"{name} must be a DataFrame"
    assert list(frame.columns) == list(spec), f"{name} columns and order"
    for column, dtype in spec.items():
        assert str(frame[column].dtype) == dtype, f"{name}.{column} must be {dtype}"


def _stats(values: np.ndarray) -> tuple[int, float, float, float, float]:
    """world_count, mean, P10, P50, P90 of per-world values, NaN left out (linear)."""

    kept = np.asarray(values, dtype=float)
    kept = kept[~np.isnan(kept)]
    if len(kept) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    q = np.quantile(kept, (0.1, 0.5, 0.9), method="linear")
    return len(kept), float(kept.mean()), float(q[0]), float(q[1]), float(q[2])


def validate_household_v2(result) -> None:
    """Assert household contract v1 §8 on ``result``; raise AssertionError naming the rule."""

    ev_world = getattr(result, "household_ev_world", None)
    summary = getattr(result, "household_outcomes_summary", None)
    if result.model != "action" or getattr(result, "household_value_summary", None) is None:
        assert ev_world is None and summary is None, "household frames None without trading"
        return
    assert ev_world is not None and summary is not None, "household frames on action results"
    # Model step 2 (decision 0007): on a run whose fleet frames carry the
    # timed path, household_ev_world also carries its own *_timed columns
    # (household.ev_world_columns), so the expected schema is read off the
    # frame itself rather than assumed fixed at two paths.
    timed_active = "home_import_kwh_timed" in ev_world.columns
    household_paths = (*household.PATHS, household.TIMED_PATH) if timed_active else household.PATHS
    ev_world_schema = {
        name: dtype.__name__ for name, dtype in household.ev_world_columns(household_paths).items()
    }
    _check_dtypes(ev_world, ev_world_schema, "household_ev_world")
    _check_dtypes(summary, SUMMARY_COLUMNS, "household_outcomes_summary")
    units = result.units
    world_count, vehicle_count = result.world_count, len(units)
    assert len(ev_world) == world_count * vehicle_count, "world x EV rows"
    expected_world = np.repeat(np.arange(world_count), vehicle_count)
    assert (ev_world["world_id"].to_numpy() == expected_world).all(), "world-major rows"
    unit_ids = np.tile(units["unit_id"].to_numpy(dtype=object), world_count)
    assert (ev_world["unit_id"].to_numpy(dtype=object) == unit_ids).all(), "EVs in units order"
    cohorts = np.tile(units["cohort_id"].to_numpy(dtype=object), world_count)
    assert (ev_world["cohort_id"].to_numpy(dtype=object) == cohorts).all(), "cohort_id from units"
    treated = _treated(units)
    assert (ev_world["treated"].to_numpy() == np.tile(treated, world_count)).all(), "treated"
    assert ev_world["evidence_kind"].eq(EVIDENCE).all(), "evidence_kind"
    assert summary["evidence_kind"].eq(EVIDENCE).all(), "summary evidence_kind"

    def matrix(column: str) -> np.ndarray:
        return ev_world[column].to_numpy(dtype=float).reshape(world_count, vehicle_count)

    _check_fleet_reconciliation(result, matrix, treated)
    _check_value_summary(result, matrix, treated)
    _check_counts_and_gates(result, ev_world, matrix, treated)
    _check_summary(result, ev_world, summary)
    _check_charge_completion(result, summary, matrix, treated)


def _check_fleet_reconciliation(result, matrix, treated) -> None:
    world = result.fleet_world_intervals
    for path in ("normal", "selected"):
        rows = world.loc[world["path_id"].eq(path)].sort_values(["world_id", "slot_index"])
        for column in ("home_import_kwh", "public_import_kwh"):
            weekly = rows.groupby("world_id", sort=True)[column].sum().to_numpy()
            assert np.allclose(
                matrix(f"{column}_{path}").sum(axis=1), weekly, rtol=0.0, atol=TOL_KWH
            ), f"sum over EVs of {column}_{path} equals the fleet week"
    cost = result.cost_effect.sort_values("world_id")
    energy = matrix("home_cost_gbp_selected") - matrix("home_cost_gbp_normal")
    assert np.allclose(
        energy.sum(axis=1),
        cost["illustrative_selected_minus_normal_energy_cost_gbp"].to_numpy(),
        rtol=0.0,
        atol=TOL_GBP,
    ), "home cost difference sums to the energy cost effect"
    assert np.allclose(matrix("supplier_energy_saving_gbp"), -energy, rtol=0.0, atol=1e-9), (
        "supplier energy saving = normal - selected home cost"
    )
    rate = _public_rate(result)
    for path in ("normal", "selected") if rate is not None else ():
        assert np.allclose(
            matrix(f"public_cost_gbp_{path}"),
            matrix(f"public_import_kwh_{path}") * rate,
            rtol=1e-12,
            atol=0.0,
        ), "public cost = public import x rate"
    saving = matrix("saving_gbp")
    assert np.allclose(
        saving.sum(axis=1),
        -cost["illustrative_selected_minus_normal_total_gbp"].to_numpy(),
        rtol=0.0,
        atol=TOL_GBP,
    ), "sum of savings = minus the total cost effect (T§9.3c)"
    assert (
        matrix("sessions_affected_count").sum(axis=1) == cost["sessions_affected_count"].to_numpy()
    ).all(), "sessions affected sum to cost_effect"
    share = matrix("value_gbp_per_month") / _MONTHLY - saving
    ledger = result.trading_ledger_world
    full = ledger.loc[ledger["strategy"].eq("full")]
    customer = full.groupby("world_id", sort=True)["customer_revenue_share_gbp"].sum().to_numpy()
    assert np.allclose(share.sum(axis=1), -customer, rtol=0.0, atol=TOL_GBP), (
        "allocated shares sum to the customer revenue share"
    )
    if (~treated).any():
        assert np.allclose(share[:, ~treated], 0.0, rtol=0.0, atol=1e-9), "control EVs share 0"


def _check_value_summary(result, matrix, treated) -> None:
    """T§9.3c identity: household_value_summary recomputed from the frame (treated EVs)."""

    value = matrix("value_gbp_per_month")
    saving = matrix("saving_gbp") * _MONTHLY
    share = value - saving
    summary = result.household_value_summary
    for group_id, members in _groups(result).items():
        v = value[:, members]
        per_world = {
            "mean_value": v.mean(axis=1),
            "p10_across_evs": np.quantile(v, 0.1, axis=1, method="linear"),
            "p50_across_evs": np.quantile(v, 0.5, axis=1, method="linear"),
            "p90_across_evs": np.quantile(v, 0.9, axis=1, method="linear"),
            "share_worse_off": (v < 0.0).mean(axis=1),
            "mean_customer_saving": saving[:, members].mean(axis=1),
            "mean_revenue_share": share[:, members].mean(axis=1),
        }
        rows = summary.loc[summary["group_id"].eq(group_id)].set_index("statistic")
        for statistic, values in per_world.items():
            count, mean, p10, p50, p90 = _stats(values)
            stored = rows.loc[statistic, ["mean", "p10", "p50", "p90"]].to_numpy(dtype=float)
            _close(stored, [mean, p10, p50, p90], 1e-9, f"household_value_summary {statistic}")


def _check_counts_and_gates(result, ev_world, matrix, treated) -> None:
    home = {p: matrix(f"home_import_kwh_{p}") for p in ("normal", "selected")}
    for path in ("normal", "selected"):
        assert (matrix(f"cheap_import_kwh_{path}") <= home[path] + 1e-9).all(), "cheap <= home"
    # The shared helper reproduces the independent low-third mask, and the
    # normal-path cheap import sums to the fleet's low-band energy per world.
    price = price_matrix(result)
    low = low_third(price)
    assert ((price_third_bands(price) == "low") == low).all(), "price_third_bands low mask"
    world = result.fleet_world_intervals
    normal = world.loc[world["path_id"].eq("normal")].sort_values(["world_id", "slot_index"])
    fleet = normal["home_import_kwh"].to_numpy(dtype=float).reshape(result.world_count, -1)
    assert np.allclose(
        matrix("cheap_import_kwh_normal").sum(axis=1),
        (fleet * low).sum(axis=1),
        rtol=0.0,
        atol=TOL_KWH,
    ), "cheap import sums to the fleet's low-band energy"

    sessions = matrix("session_count")
    ends = matrix("session_end_count")
    for path in ("normal", "selected"):
        completed = matrix(f"completed_count_{path}")
        assert ((completed >= 0) & (completed <= sessions)).all(), "0 <= completed <= sessions"
        soc = matrix(f"departure_soc_mean_{path}")
        assert (np.isnan(soc) == (sessions == 0)).all(), "departure SoC NaN iff no session"
        kept = soc[~np.isnan(soc)]
        assert ((kept >= -1e-9) & (kept <= 100.0 + 1e-9)).all(), "departure SoC in 0-100"
    assert (sessions <= ends).all(), "sessions <= session ends"
    assert (matrix("sessions_affected_count") <= ends).all(), "affected <= session ends"
    nights = matrix("nights_plugged_count")
    assert ((nights >= 0) & (nights <= night_count(result))).all(), "0 <= nights <= N_n"

    co2 = matrix("co2_shifted_kg")
    carbon = getattr(result, "carbon_shift_world", None)
    if carbon is None:
        assert np.isnan(co2).all(), "co2 NaN without carbon_shift_world"
    else:
        per_world = carbon.sort_values("world_id")["co2_shifted_kg"].to_numpy(dtype=float)
        assert np.allclose(co2.sum(axis=1), per_world, rtol=1e-9, atol=1e-6), (
            "sum over EVs of co2_shifted_kg equals carbon_shift_world"
        )
        assert not np.isnan(co2).any(), "co2 defined for every EV with the intensity"

    world_slot = getattr(result, "availability_world_slot", None)
    one_hour = None if world_slot is None else world_slot.loc[world_slot["duration_hours"].eq(1.0)]
    for direction in ("turn_down", "turn_up"):
        kw = matrix(f"evening_{direction}_kw_1h")
        if one_hour is None or one_hour["deliverable_kw"].isna().all():
            assert np.isnan(kw).all(), "evening kW NaN until J1b"
            continue
        assert (kw >= 0.0).all(), "evening kW >= 0"
        if (~treated).any():
            assert (kw[:, ~treated] == 0.0).all(), "control EVs offer 0 kW"
        rows = one_hour.loc[one_hour["direction"].eq(direction)].sort_values(
            ["world_id", "slot_index"]
        )
        deliverable = rows["deliverable_kw"].to_numpy(dtype=float).reshape(result.world_count, -1)
        labels = result.study_slots["local_time_label"].to_numpy(dtype=object)
        evening = (labels >= "17:00") & (labels < "21:00")
        assert np.allclose(
            kw.sum(axis=1), deliverable[:, evening].mean(axis=1), rtol=0.0, atol=1e-6
        ), f"evening {direction} kW sums to the fleet evening mean"


def _check_charge_completion(result, summary, matrix, treated) -> None:
    """J5 cross-check (§8): with no control group the two session populations coincide.

    ``completed_share_p`` set R quantiles per group equal J5's
    ``completed_share_p10/p50/p90`` (``departure = all``), and the mean over
    worlds of the group's summed ``session_count`` equals ``session_count_mean``.
    """

    completion = getattr(result, "charge_completion_summary", None)
    if completion is None or (~treated).any():
        return
    sessions = matrix("session_count")
    j5 = completion.loc[completion["departure"].eq("all")].set_index(["group_id", "path_id"])
    for group_id, members in _groups(result).items():
        for path in ("normal", "selected"):
            row = j5.loc[(group_id, path)]
            ours = summary.loc[
                summary["group_id"].eq(group_id)
                & summary["metric"].eq(f"completed_share_{path}")
                & summary["statistic"].eq("group_ratio")
            ].iloc[0]
            _close(
                ours[["p10", "p50", "p90"]],
                row[["completed_share_p10", "completed_share_p50", "completed_share_p90"]],
                1e-9,
                f"J5 completed share {group_id} {path}",
            )
            _close(
                [sessions[:, members].sum(axis=1).mean()],
                [row["session_count_mean"]],
                1e-9,
                f"J5 session count {group_id}",
            )


def _check_summary(result, ev_world, summary) -> None:
    """The §3.2 summary equals its recomputation from ``metric_values``, set by set."""

    n_nights = night_count(result)
    values = household.metric_values(ev_world, n_nights)
    paths = (
        (*household.PATHS, household.TIMED_PATH)
        if "home_import_kwh_timed" in ev_world.columns
        else household.PATHS
    )
    metric_table = _metric_table(paths)
    assert list(values.columns[4:]) == list(metric_table), "metric_values columns"
    # One ratio checked by hand, independently of metric_values: completed
    # sessions over closed sessions, NaN for an EV with none (O4).
    sessions = ev_world["session_count"].to_numpy(dtype=float)
    completed = ev_world["completed_count_selected"].to_numpy(dtype=float)
    by_hand = np.array([c / s if s else np.nan for c, s in zip(completed, sessions, strict=True)])
    _close(values["completed_share_selected"], by_hand, 1e-12, "completed_share_selected by hand")
    world_count, vehicle_count = result.world_count, len(result.units)

    def matrix(array) -> np.ndarray:
        return np.asarray(array, dtype=float).reshape(world_count, vehicle_count)

    def group_ratio(metric: str, members: np.ndarray) -> np.ndarray:
        rule = metric_table[metric][3]
        if rule[0] == "difference":
            return group_ratio(rule[1], members) - group_ratio(rule[2], members)
        top = matrix(ev_world[rule[0]])[:, members].sum(axis=1)
        if rule[1] is None:
            bottom = np.full(world_count, float(n_nights * members.sum()))
        else:
            bottom = matrix(ev_world[rule[1]])[:, members].sum(axis=1)
        out = np.full(world_count, np.nan)
        np.divide(top, bottom, out=out, where=bottom != 0)
        return out

    expected = []
    for group_id, members in _groups(result).items():
        ev_count = int(members.sum())
        for metric, (unit, horizon, better_is, rule) in metric_table.items():
            v = matrix(values[metric])[:, members]
            has = ~np.isnan(v)
            per_world_count = int(has.sum(axis=1).min())
            head = [group_id, metric]
            if horizon == "week_ahead":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    inner = {
                        "mean_across_evs": np.nanmean(v, axis=1),
                        "p10_across_evs": np.nanquantile(v, 0.1, axis=1, method="linear"),
                        "p50_across_evs": np.nanquantile(v, 0.5, axis=1, method="linear"),
                        "p90_across_evs": np.nanquantile(v, 0.9, axis=1, method="linear"),
                    }
                for statistic in SET_A:
                    expected.append(
                        [
                            *head,
                            statistic,
                            unit,
                            horizon,
                            better_is,
                            ev_count,
                            per_world_count,
                            *_stats(inner[statistic]),
                        ]
                    )
            if rule is not None:
                expected.append(
                    [
                        *head,
                        "group_ratio",
                        unit,
                        horizon,
                        better_is,
                        ev_count,
                        per_world_count,
                        *_stats(group_ratio(metric, members)),
                    ]
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                means = np.nanmean(v, axis=0)
            defined = ~np.isnan(means)
            kept = means[defined]
            worlds_used = int(has.sum(axis=0)[defined].min()) if defined.any() else 0
            if len(kept):
                q = np.quantile(kept, (0.1, 0.5, 0.9), method="linear")
                numbers = [kept.mean(), q[0], q[1], q[2], (kept < 0).mean()]
            else:
                numbers = [np.nan] * 5
            for statistic, number in zip(SET_B, numbers, strict=True):
                unit_b = "fraction" if statistic == "share_of_ev_means_below_zero" else unit
                expected.append(
                    [
                        *head,
                        statistic,
                        unit_b,
                        horizon,
                        better_is,
                        ev_count,
                        int(defined.sum()),
                        worlds_used,
                        number,
                        np.nan,
                        number,
                        np.nan,
                    ]
                )
    columns = list(SUMMARY_COLUMNS)[:-1]
    frame = pd.DataFrame(expected, columns=columns)
    assert len(summary) == len(frame), "summary rows"
    for column in columns[:9]:
        assert (summary[column].to_numpy() == frame[column].to_numpy()).all(), (
            f"summary {column} (keys, order, counts)"
        )
    for column in ("mean", "p10", "p50", "p90"):
        _close(summary[column], frame[column], 1e-9, f"summary {column} recomputation")
    assert set(summary["better_is"]) <= {"higher", "lower", ""}, "better_is values"
    set_b = summary["statistic"].isin(SET_B)
    assert summary.loc[set_b, ["p10", "p90"]].isna().all().all(), "set B has no week spread"
    ordered = summary.loc[~set_b].dropna(subset=["p10", "p50", "p90"])
    assert (
        (ordered["p10"] <= ordered["p50"] + 1e-12) & (ordered["p50"] <= ordered["p90"] + 1e-12)
    ).all(), "set A and R quantiles ordered"
    share_rows = summary.loc[
        summary["unit"].eq("fraction") & ~summary["metric"].str.endswith("_difference")
    ]
    kept = share_rows[["mean", "p10", "p50", "p90"]].to_numpy(dtype=float)
    kept = kept[~np.isnan(kept)]
    assert ((kept >= -1e-12) & (kept <= 1.0 + 1e-12)).all(), "shares within [0, 1]"
    assert (summary["ev_value_count"] <= summary["ev_count"]).all(), "ev_value_count <= ev_count"


# --------------------------------------------------------------------------
# The household card (§3.3): fixture builder and validator (lane HH2)
# --------------------------------------------------------------------------

CARD_OUTCOME_COLUMNS = {
    "metric": "object",
    "unit": "object",
    "horizon": "object",
    "better_is": "object",
    "world_count": "int64",
    **dict.fromkeys(("mean", "p10", "p50", "p90"), "float64"),
    **dict.fromkeys(
        (f"{g}_{q}" for g in ("cohort", "fleet") for q in ("p10", "p50", "p90")), "float64"
    ),
    "cohort_rank_p50": "float64",
    "fleet_rank_p50": "float64",
}
AVAILABILITY_COLUMNS = {
    "direction": "object",
    "slot_index": "int64",
    "night_index": "int64",
    "interval_start_utc": "datetime64[ns, UTC]",
    "interval_start_london": "datetime64[ns, Europe/London]",
    "unit": "object",
    "world_count": "int64",
    "available_share": "float64",
    **dict.fromkeys(("mean", "p05", "p10", "p50", "p90", "p95", "firm_share"), "float64"),
    "evidence_kind": "object",
}
AVAILABILITY_DAY_COLUMNS = {
    "direction": "object",
    "day_type": "object",
    "local_half_hour": "int64",
    "local_time_label": "object",
    "unit": "object",
    "world_count": "int64",
    "available_share": "float64",
    **dict.fromkeys(("mean", "p05", "p10", "p50", "p90", "p95", "firm_share"), "float64"),
    "evidence_kind": "object",
}
SESSION_TIMING_COLUMNS = {
    "metric": "object",
    "unit": "object",
    "bin_index": "int64",
    "profile_order": "int64",
    "bin_lower": "float64",
    "bin_upper": "float64",
    "bin_label": "object",
    "session_count": "int64",
    "share": "float64",
    "evidence_kind": "object",
}
RELIABILITY_COLUMNS = {
    "metric": "object",
    "unit": "object",
    "sample_kind": "object",
    "sample_count": "int64",
    **dict.fromkeys(("mean", "p10", "p50", "p90"), "float64"),
    **dict.fromkeys(("p10_label", "p50_label", "p90_label"), "object"),
    "ratio_p10_to_p50": "float64",
    "evidence_kind": "object",
}
RELIABILITY_METRICS = (
    "nights_plugged_share",
    "sessions_per_week",
    "session_ends_per_week",
    "plug_in_time",
    "departure_time",
    "dwell_hours",
    "energy_needed_kwh",
    "flexible_kwh_per_night",
    "evening_turn_down_kw_1h",
    "evening_turn_up_kw_1h",
)


def make_household_card(result, unit_id: str):
    """SYNTHETIC ``HouseholdCard`` from the fixture's per-EV toy arrays (no kernel run).

    The fixture has no firm-MW inputs, so ``availability`` is ``None``, as on
    a real run before J1b.
    """

    from .result_fixture import _CHARGE_EFFICIENCY

    sim = result.replay_state
    units = result.units
    e = list(units["unit_id"]).index(unit_id)
    capacity = float(units["physical_capacity_kwh"].iat[e])
    return household.build_card(
        result,
        unit_id,
        connected=sim.connected["normal"][:, :, e],
        closing_kwh=sim.closing_kwh["normal"][:, :, e],
        home_import_kwh=sim.home_import_kwh["normal"][:, :, e],
        realised_kw=None,
        capacity_kwh=capacity,
        target_kwh=capacity * float(units["preferred_target_soc_percent"].iat[e]) / 100.0,
        power_kw=float(units["home_charger_limit_kw"].iat[e]),
        efficiency=_CHARGE_EFFICIENCY,
    )


def _label(hours: float) -> str:
    minutes = int(round(hours * 60.0)) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def validate_household_card_v2(card, result) -> None:
    """Assert household contract v1 §8 "Card" on one card of ``result``."""

    ev_world = result.household_ev_world
    units = result.units
    unit_ids = units["unit_id"].to_numpy(dtype=object)
    e = int(np.flatnonzero(unit_ids == card.unit_id)[0])
    world_count, vehicle_count = result.world_count, len(units)
    assert card.cohort_id == units["cohort_id"].iat[e], "card cohort"
    treated = _treated(units)
    assert card.treated == bool(treated[e]), "card treated flag"
    assert card.world_count == world_count, "card world_count"

    outcomes = card.outcomes
    _check_dtypes(outcomes, CARD_OUTCOME_COLUMNS, "card.outcomes")
    # Decision 0007, model step 2: the card's own metric table follows
    # ``household_ev_world`` exactly as ``_check_summary`` does below, so a
    # run with the timed path reports its extra rows here too.
    paths = (
        (*household.PATHS, household.TIMED_PATH)
        if "home_import_kwh_timed" in ev_world.columns
        else household.PATHS
    )
    metric_table = _metric_table(paths)
    assert list(outcomes["metric"]) == list(metric_table), "card metrics in §3.2 order"
    values = household.metric_values(ev_world, night_count(result))
    summary = result.household_outcomes_summary.set_index(["group_id", "metric", "statistic"])
    cohorts = units["cohort_id"].to_numpy(dtype=object)
    members = {"cohort": treated & (cohorts == cohorts[e]), "fleet": treated}
    group_ids = {"cohort": cohorts[e], "fleet": "fleet"}
    for _, row in outcomes.iterrows():
        unit, horizon, better_is, _ = metric_table[row["metric"]]
        assert (row["unit"], row["horizon"], row["better_is"]) == (unit, horizon, better_is)
        v = values[row["metric"]].to_numpy(dtype=float).reshape(world_count, vehicle_count)
        mine = v[:, e]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(v, axis=0)
        if horizon == "week_ahead":
            expected = _stats(mine)
            statistics = ("p10_across_evs", "p50_across_evs", "p90_across_evs")
            column = "p50"
        else:
            expected = (int((~np.isnan(mine)).sum()), means[e], np.nan, np.nan, np.nan)
            statistics = ("p10_of_ev_means", "p50_of_ev_means", "p90_of_ev_means")
            column = "mean"
        assert row["world_count"] == expected[0], f"card {row['metric']} world_count"
        _close(row[["mean", "p10", "p50", "p90"]], expected[1:], 1e-9, f"card {row['metric']}")
        for name, group_id in group_ids.items():
            copied = [
                summary.at[(group_id, row["metric"], s), column]
                if (group_id, row["metric"], s) in summary.index
                else np.nan
                for s in statistics
            ]
            stored = row[[f"{name}_p10", f"{name}_p50", f"{name}_p90"]]
            _close(stored, copied, 1e-12, f"card {name} columns of {row['metric']}")
            rank = row[f"{name}_rank_p50"]
            if not treated[e]:
                assert np.isnan(rank), "control EV has no rank"
                continue
            group = v[:, members[name]]
            if horizon == "week_ahead":
                shares = np.array(
                    [
                        (g[~np.isnan(g)] <= m).mean() if not np.isnan(m) else np.nan
                        for g, m in zip(group, mine, strict=True)
                    ]
                )
                expected_rank = _stats(shares)[3]
            else:
                pool = means[members[name]]
                pool = pool[~np.isnan(pool)]
                expected_rank = (pool <= means[e]).mean() if not np.isnan(means[e]) else np.nan
            _close([rank], [expected_rank], 1e-12, f"card {name} rank of {row['metric']}")
            assert np.isnan(rank) or 0.0 <= rank <= 1.0, "rank in [0, 1]"

    if card.availability is not None:
        _check_availability(card, result)
    else:
        assert card.availability_day is None, "availability_day None with availability"

    timing = card.session_timing
    _check_dtypes(timing, SESSION_TIMING_COLUMNS, "card.session_timing")
    assert list(dict.fromkeys(timing["metric"])) == ["plug_in_time", "departure_time"]
    for _, block in timing.groupby("metric", sort=False):
        assert list(block["bin_index"]) == list(range(48)), "48 half-hour bins"
        assert sorted(block["profile_order"]) == list(range(48)), "profile_order permutation"
        assert block.loc[block["bin_index"].eq(24), "profile_order"].iat[0] == 0, "12:00 first"
        if block["session_count"].iat[0] == 0:
            assert block["share"].isna().all(), "no sessions: NaN shares"
        else:
            assert np.isclose(block["share"].sum(), 1.0, atol=1e-12), "shares sum to 1"

    rows = card.reliability
    _check_dtypes(rows, RELIABILITY_COLUMNS, "card.reliability")
    assert tuple(rows["metric"]) == RELIABILITY_METRICS, "reliability rows in order"
    mine = ev_world.loc[ev_world["unit_id"].eq(card.unit_id)]
    weekly = {
        "nights_plugged_share": mine["nights_plugged_count"] / night_count(result),
        "sessions_per_week": mine["session_count"],
        "session_ends_per_week": mine["session_end_count"],
        "evening_turn_down_kw_1h": mine["evening_turn_down_kw_1h"],
        "evening_turn_up_kw_1h": mine["evening_turn_up_kw_1h"],
    }
    by_metric = rows.set_index("metric")
    for metric, per_world in weekly.items():
        stored = by_metric.loc[metric, ["sample_count", "mean", "p10", "p50", "p90"]]
        _close(stored, _stats(per_world.to_numpy(dtype=float)), 1e-9, f"reliability {metric}")
    for metric in ("plug_in_time", "departure_time"):
        row = by_metric.loc[metric]
        for level in ("p10", "p50", "p90"):
            value = row[level]
            if np.isnan(value):
                assert row[f"{level}_label"] == "", "no label without a value"
            else:
                assert 0.0 <= value < 24.0, "clock hours in [0, 24)"
                assert row[f"{level}_label"] == _label(value), f"{metric} {level} label"
    for _, row in rows.iterrows():
        expected = row["p10"] / row["p50"] if row["p50"] > 0 else np.nan
        _close([row["ratio_p10_to_p50"]], [expected], 1e-12, "ratio_p10_to_p50")
        if row["metric"] not in ("plug_in_time", "departure_time"):
            assert row["p10_label"] == row["p50_label"] == row["p90_label"] == ""


def _check_availability(card, result) -> None:
    slot_count = len(result.study_slots)
    for frame, spec, rows in (
        (card.availability, AVAILABILITY_COLUMNS, 2 * slot_count),
        (card.availability_day, AVAILABILITY_DAY_COLUMNS, 2 * 3 * 48),
    ):
        _check_dtypes(frame, spec, "card availability")
        assert len(frame) == rows, "card availability rows"
        levels = frame[["p05", "p10", "p50", "p90", "p95"]].to_numpy(dtype=float)
        kept = ~np.isnan(levels).any(axis=1)
        assert (np.diff(levels[kept], axis=1) >= -1e-12).all(), "p05 <= ... <= p95"
        share = frame["available_share"].to_numpy(dtype=float)
        assert (np.isnan(share) == frame["world_count"].eq(0).to_numpy()).all(), "share NaN"
        assert ((share[~np.isnan(share)] >= 0) & (share[~np.isnan(share)] <= 1)).all()
        p10, p50 = frame["p10"].to_numpy(dtype=float), frame["p50"].to_numpy(dtype=float)
        firm = np.where(p50 > 0, p10 / np.where(p50 > 0, p50, 1.0), np.nan)
        _close(frame["firm_share"], firm, 1e-12, "firm_share = p10 / p50")
