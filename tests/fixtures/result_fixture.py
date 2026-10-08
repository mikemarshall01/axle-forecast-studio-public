"""SYNTHETIC FIXTURE result for contract v2 (`docs/contracts/results-v2.md`).

What this owns: a small, deterministic, hand-readable stand-in for the final
``ForecastResult`` so dashboard views can be built and tested before model
tasks M4/M5 land.  The contract specs and validators live in
``result_contract.py`` and are re-exported here, so existing imports from
this module keep working.

Nothing here is model output.  The "physics" below is a toy day plan per EV
(leave in the morning, come home in the evening, plug in, charge to a target)
chosen so the frames have realistic shapes: evening plug-ins, plug-in SoC
spread across roughly 15-65%, a selected (smart charging) path that moves
evening charging into the night, some early departures that leave short, and
some worlds where the selected path ends short (flagged).  It never calls the
model package, so it cannot drift with model internals.

Why a toy simulation rather than hand-typed tables: every frame is then
derived from the same per-EV arrays, so the fixture is internally consistent
(bands really are quantiles of world values, difference bands really are
per-world differences) and the validators can check those rules on it and on
real results alike.

Public API:
- ``make_result`` builds a ``FixtureForecastResult``.
- ``replay_one_ev``, ``replay_one_ev_bands`` and ``compare_runs`` mirror the
  M5 functions in ``model/individual.py`` and ``model/summaries.py``.
- ``validate_result_v2``, ``validate_one_ev_replay_v2`` and
  ``validate_run_comparison_v2`` (re-exported from ``result_contract``) check
  any object against the contract, so M5 runs them on real results.

Authority: decision 0004 items 1-38 and 54 (deferrable power by slack,
coincidence factor, half-hour plug heatmaps, relative day-ahead prices).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from axle_studio.model import assumptions, supplier

from .household_contract import make_household_frames
from .replay_contract import make_replay_week
from .result_contract import (
    ACTION_ID,
    AT_LEAST_HOURS,
    AVERAGE_DAY_COLUMNS,
    BAND_STAT_COLUMNS,
    CHANGED_COLUMNS,
    CHEAPEST_HALF_HOUR_COLUMNS,
    CNZ_RECORD_NAMES,
    COHORT_BAND_COLUMNS,
    COHORT_ORDER,
    COHORT_SUMMARY_COLUMNS,
    COHORTS,
    COMPARE_KPI_COLUMNS,
    COMPARE_WEEKLY_COLUMNS,
    COST_COMPONENTS,
    COST_CURVE_THRESHOLDS,
    COST_EFFECT_COLUMNS,
    COST_SUMMARY_COLUMNS,
    DAILY_AUDIT_COLUMNS,
    DAY_TYPE_ORDER,
    DEFERRABLE_BUCKET_ORDER,
    DEFERRABLE_CAPACITY_SHARE,
    DEFERRABLE_POWER_BAND_COLUMNS,
    DEVIATION_WORLD_SLOT_COLUMNS,
    DIFFERENCE_BAND_COLUMNS,
    DIFFERENCE_WEEKLY_BAND_COLUMNS,
    DIFFERENCE_WEEKLY_COLUMNS,
    EV_BAND_METRICS,
    EVALUATION_PRICE_COLUMNS,
    EVENT_RESPONSE_COLUMNS,
    EVENT_TABLE_COLUMNS,
    EVIDENCE_LABEL,
    FLEET_BAND_COLUMNS,
    FLEET_EV_BAND_COLUMNS,
    FLEET_METRICS,
    FLEET_WORLD_COLUMNS,
    FLEX_COST_CURVE_COLUMNS,
    FLEXIBILITY_BAND_COLUMNS,
    FLEXIBILITY_ROWS,
    FLEXIBILITY_WEEKLY_COLUMNS,
    FORECAST_PRICE_COLUMNS,
    HALF_HOUR,
    HOUR_BAND_COLUMNS,
    HOUSEHOLD_BIN_EDGES,
    HOUSEHOLD_DISTRIBUTION_COLUMNS,
    HOUSEHOLD_SUMMARY_COLUMNS,
    LOCATION_ORDER,
    LONDON,
    MARKET_SHOCK_COLUMNS,
    MAX_PLUG_IN_EVENT_WORLDS,
    MOVABLE_CHECK_HALF_HOUR,
    NOT_RECOVERED_MATERIAL_SHARE_PERCENT,
    NOT_RECOVERED_TOLERANCE_KWH,
    PATH_ORDER,
    PLUG_EVENT_HEATMAP_COLUMNS,
    PLUG_IN_EVENT_COLUMNS,
    PLUG_IN_HEATMAP_COLUMNS,
    PLUG_IN_WORLD_KPI_COLUMNS,
    PLUG_KPI_COLUMNS,
    PLUG_KPI_METRICS,
    POSITION_UPDATE_COLUMNS,
    PRICE_BAND_SHIFT_COLUMNS,
    PRICE_BANDS,
    PRICE_RELATIVE_COLUMNS,
    PRICE_SHOCK_COLUMNS,
    REPLAY_BAND_COLUMNS,
    REPLAY_INTERVAL_COLUMNS,
    REPORT_HALF_HOURS,
    REVENUE_SEGMENT_COLUMNS,
    SESSION_DISTRIBUTION_COLUMNS,
    SESSION_DISTRIBUTION_METRICS,
    SESSION_SOURCE_RECORD_NAMES,
    SETTINGS_KEYS,
    SHARE_METRICS,
    SLACK_BUCKETS,
    SLOT_COLUMNS,
    SLOT_COUNT,
    SMART_CHARGING_METRICS,
    SMART_CHARGING_SUMMARY_COLUMNS,
    SMART_CHARGING_WORLD_COLUMNS,
    SOC_BAND_COLUMNS,
    STUDY_DAYS,
    STUDY_SLOT_COLUMNS,
    TRADING_BUCKETS,
    TRADING_CHECK_COLUMNS,
    TRADING_CHECK_IDS,
    TRADING_MONEY,
    TRADING_STRATEGIES,
    UNIT_COLUMNS,
    WAIT_CHECK_HALF_HOUR,
    WEEKDAY_LABELS,
    WEEKLY_BAND_COLUMNS,
    WEEKLY_BAND_METRICS,
    WEEKLY_DIFFERENCE_METRICS,
    WEEKLY_PEAK_COLUMNS,
    ZONE_IDS,
    ZONE_WORLD_COLUMNS,
    Assumption,
    baseline_erosion_from,
    cheapest_half_hour_from,
    cohort_plug_in_stats,
    cost_summary_from_cost_effect,
    day_ahead_spread_from,
    difference_bands_from_world,
    difference_weekly_from_world,
    empty_frame,
    fleet_bands,
    forecast_price_bands_from,
    metric_matrix,
    not_recovered_summary_from,
    open_position_from,
    plug_event_heatmap_from,
    plug_in_kpis_from_world_kpis,
    plug_in_world_kpis_from_events,
    price_band_shift_from,
    price_relative_bands_from,
    quantile_stats,
    session_distribution_bands_from,
    shape_premium_from,
    shock_attribution_from,
    shock_summary_from,
    slot_keys,
    smart_charging_summary_from,
    smart_charging_world_from,
    supplier_positions_from,
    trading_kpis_from,
    trading_ledger_from,
    trading_ledger_summary_from,
    trading_week_from,
    typed_frame,
    validate_one_ev_replay_v2,
    validate_result_v2,
    validate_run_comparison_v2,
    weekly_bands_from_world,
    weekly_difference_bands,
    weekly_peak_summary_from,
    world_stats,
    zone_import_bands_from,
    zone_summary_from,
)

# Toy behaviour per archetype (SYNTHETIC, chosen for readable shapes, not
# sourced).  Times are London minutes after midnight.  ``drive_weekdays`` None
# means every day; ``plug_every`` 3 means the EV plugs in every third date.
_BEHAVIOUR = {
    "average_uk": dict(depart=450, arrive=1080, trip_kwh=10.0, capacity=60.0),
    "intelligent_octopus": dict(depart=420, arrive=1110, trip_kwh=24.0, capacity=72.5),
    "infrequent_charging": dict(depart=480, arrive=1050, trip_kwh=9.0, capacity=60.0, plug_every=3),
    "infrequent_driving": dict(
        depart=540, arrive=1080, trip_kwh=8.0, capacity=60.0, drive_weekdays=(0, 2, 5)
    ),
    "scheduled_charging": dict(depart=510, arrive=1320, trip_kwh=10.0, capacity=60.0),
    "always_plugged_in": dict(
        depart=600, arrive=900, trip_kwh=3.0, capacity=60.0, drive_weekdays=(1,)
    ),
}
_TARGET_SOC_FRACTION = 0.8
_OPENING_SOC_FRACTION = 0.7
_CHARGE_EFFICIENCY = 0.9
# SYNTHETIC stand-in for smart charging (decision 0004 item 38): the toy's
# cheapest forecast half-hours are 02:00-06:00 London, so the selected path
# charges only then.  Eight half-hours hold 25.2 kWh at 7 kW, so EVs that
# plug in every third day leave short some mornings (early departures).
_SMART_NIGHT_MINUTES = (2 * 60, 6 * 60)
_DEPARTURE_MARGIN_HOURS = 1.0
_PUBLIC_RATE = 0.79  # GBP/kWh, the fixture's illustrative public charging rate
_CLOCK_CHANGE_STARTS = {
    None: date(2026, 9, 28),
    "autumn": date(2026, 10, 19),  # clocks go back Sun 25 Oct 2026: 50 half-hours
    "spring": date(2027, 3, 22),  # clocks go forward Sun 28 Mar 2027: 46 half-hours
}


# --------------------------------------------------------------------------
# Result types (mirrors of the contract; names carry a Fixture prefix)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FixturePlugInSummary:
    hour_bands: pd.DataFrame
    soc_bands: pd.DataFrame
    kpis: pd.DataFrame


@dataclass(frozen=True)
class FixtureActionSummary:
    action_id: str
    action_label: str
    departure_margin_hours: float
    forecast_available_at_utc: pd.Timestamp
    vehicle_count: int


@dataclass(frozen=True)
class FixtureForecastResult:
    """SYNTHETIC FIXTURE mirror of ``model.forecast.ForecastResult`` (contract v2)."""

    model: str
    model_label: str
    policy_id: str
    evidence_kind: str
    run_completed_at_utc: pd.Timestamp
    seed: int
    vehicle_count: int
    world_count: int
    study_start_local_date: date
    horizon_start_utc: pd.Timestamp
    horizon_end_utc: pd.Timestamp
    study_slot_count: int
    has_clock_change: bool
    settings_snapshot: dict[str, object]
    assumptions: tuple[Assumption, ...]
    action_status: str
    public_charging_status: str
    not_recovered_world_count: int | None
    representative_world_id: int
    study_slots: pd.DataFrame
    units: pd.DataFrame
    cohort_summary: pd.DataFrame
    fleet_world_intervals: pd.DataFrame
    fleet_interval_bands: pd.DataFrame
    cohort_interval_bands: pd.DataFrame
    average_day_bands: pd.DataFrame
    fleet_interval_ev_bands: pd.DataFrame
    flexibility_bands: pd.DataFrame
    weekly_bands: pd.DataFrame
    plug_in_events: pd.DataFrame
    sampled_world_ids: tuple[int, ...]
    plug_in_world_kpis: pd.DataFrame
    plug_in_summary: FixturePlugInSummary
    plug_in_heatmap: pd.DataFrame
    deferrable_power_bands: pd.DataFrame
    flexibility_weekly_summary: pd.DataFrame
    plug_in_half_hour_heatmap: pd.DataFrame
    plug_out_heatmap: pd.DataFrame
    session_distribution_bands: pd.DataFrame
    difference_bands: pd.DataFrame | None
    difference_weekly: pd.DataFrame | None
    difference_weekly_bands: pd.DataFrame | None
    action_summary: FixtureActionSummary | None
    forecast_prices: pd.DataFrame | None
    evaluation_prices: pd.DataFrame | None
    price_shocks: pd.DataFrame | None
    cost_effect: pd.DataFrame | None
    cost_effect_summary: pd.DataFrame | None
    not_recovered_summary: pd.DataFrame | None
    smart_charging_world: pd.DataFrame | None
    smart_charging_summary: pd.DataFrame | None
    price_band_shift: pd.DataFrame | None
    forecast_price_bands: pd.DataFrame | None
    weekly_peak_summary: pd.DataFrame | None
    price_relative_bands: pd.DataFrame | None
    cheapest_half_hour_summary: pd.DataFrame | None
    zone_ids: tuple[str, ...]
    zone_world_intervals: pd.DataFrame
    zone_import_bands: pd.DataFrame
    zone_summary: pd.DataFrame
    events: pd.DataFrame | None
    event_response_bands: pd.DataFrame | None
    market_shocks: pd.DataFrame | None
    shock_summary: pd.DataFrame | None
    replay_state: object = field(repr=False)
    fixture_label: str = EVIDENCE_LABEL
    # Trading overlay (trading contract v1 §5, §9); None on no-action results.
    deviation_world_slot: pd.DataFrame | None = None
    position_updates: pd.DataFrame | None = None
    trading_ledger_world: pd.DataFrame | None = None
    trading_week_world: pd.DataFrame | None = None
    trading_ledger_summary: pd.DataFrame | None = None
    trading_kpis: pd.DataFrame | None = None
    trading_checks: pd.DataFrame | None = None
    shock_attribution_summary: pd.DataFrame | None = None
    open_position_profile: pd.DataFrame | None = None
    flex_cost_curve: pd.DataFrame | None = None
    shape_premium_summary: pd.DataFrame | None = None
    household_value_distribution: pd.DataFrame | None = None
    household_value_summary: pd.DataFrame | None = None
    household_ev_world: pd.DataFrame | None = None
    household_outcomes_summary: pd.DataFrame | None = None
    revenue_by_segment: pd.DataFrame | None = None
    supplier_positions: pd.DataFrame | None = None
    control_group_summary: pd.DataFrame | None = None
    price_curve_source: str | None = None
    user_price_curve: pd.DataFrame | None = None
    # Replay contract v1 §1 (``replay_contract.FixtureReplayWeek``); None on no-action results.
    replay_week: object | None = None
    # Supplier P&L and Partners (supplier contract v1); None on no-action results.
    supplier_pnl_world: pd.DataFrame | None = None
    supplier_pnl_summary: pd.DataFrame | None = None
    hedge_block_world: pd.DataFrame | None = None
    hedge_block_summary: pd.DataFrame | None = None
    carbon_shift_world: pd.DataFrame | None = None
    carbon_shift_summary: pd.DataFrame | None = None
    partner_world: pd.DataFrame | None = None
    partner_summary: pd.DataFrame | None = None
    household_value_exceedance: pd.DataFrame | None = None
    revenue_by_market_summary: pd.DataFrame | None = None


@dataclass(frozen=True)
class FixtureOneEvReplay:
    unit_id: str
    world_id: int
    is_representative_world: bool
    traits: dict[str, object]
    intervals: pd.DataFrame
    plug_events: pd.DataFrame
    daily_audit: pd.DataFrame


@dataclass(frozen=True)
class FixtureRunComparison:
    matched_futures: bool
    mismatch_reasons: tuple[str, ...]
    path_id_a: str
    path_id_b: str
    changed: pd.DataFrame
    illustrative_total_available: bool
    difference_bands: pd.DataFrame
    weekly_differences: pd.DataFrame
    kpis: pd.DataFrame


@dataclass(frozen=True)
class _Simulated:
    """Per-EV toy outcomes by path, each array shaped (world, slot, EV)."""

    connected: dict[str, np.ndarray]
    driving: dict[str, np.ndarray]
    away: dict[str, np.ndarray]
    public_charging: dict[str, np.ndarray]
    home_import_kwh: dict[str, np.ndarray]
    public_import_kwh: dict[str, np.ndarray]
    closing_kwh: dict[str, np.ndarray]
    unserved_kwh: dict[str, np.ndarray]
    early_count: dict[str, np.ndarray]
    early_shortfall_kwh: dict[str, np.ndarray]
    events: pd.DataFrame  # both paths
    audit: pd.DataFrame  # both paths, one row per (path, world, EV, date)


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


def make_result(
    model: str = "action",
    *,
    evs: int = 6,
    worlds: int = 4,
    seed: int = 42,
    clock_change: str | None = None,
    home_charger_kw: float = 7.0,
) -> FixtureForecastResult:
    """Build a SYNTHETIC FIXTURE result that satisfies contract v2.

    ``model`` is ``"action"`` or ``"no_action"``.  EVs cycle through the six
    archetypes in source order, so ``evs < 6`` leaves archetypes with no EVs
    (audit B1).  ``clock_change`` picks a study week containing the autumn
    (50 half-hour day) or spring (46 half-hour day) London change.  ``seed``
    only moves the toy arrival and trip jitter; ``home_charger_kw`` changes
    charging speed (kW), so two results differing only in it compare on
    matched futures.  The two models share the toy's futures, as the real
    runner's do.
    """

    if model not in ("action", "no_action"):
        raise ValueError("model must be 'action' or 'no_action'")
    if evs < 1 or worlds < 1:
        raise ValueError("evs and worlds must be at least 1")
    if clock_change not in _CLOCK_CHANGE_STARTS:
        raise ValueError("clock_change must be None, 'autumn' or 'spring'")
    is_action = model == "action"
    start = _CLOCK_CHANGE_STARTS[clock_change]
    slots = _study_slots(start)
    units = _units(evs, home_charger_kw)
    sim = _simulate(slots, units, worlds, seed, with_action=is_action)
    paths = PATH_ORDER if is_action else ("normal",)

    world = _world_frame(sim, slots, units, paths, np.ones(evs, dtype=bool))
    deferrable = _deferrable_rows(sim, slots, units)
    cohort_worlds = {
        cohort_id: _world_frame(sim, slots, units, paths, mask)
        for cohort_id in COHORT_ORDER
        if (mask := units["cohort_id"].eq(cohort_id).to_numpy()).any()
    }
    all_events = sim.events.loc[sim.events["path_id"].eq("normal")].reset_index(drop=True)
    representative = _representative_world(world)
    # M5 builds per-EV summaries in chunks of at most 10 worlds, so the event
    # table is a sample: the representative world first, then the lowest
    # other ids.  World-level KPIs below still cover every world (lead Q1).
    others = [w for w in range(worlds) if w != representative]
    sampled = (representative, *others[: MAX_PLUG_IN_EVENT_WORLDS - 1])
    events = _sample_events(all_events, sampled)
    world_kpis = plug_in_world_kpis_from_events(all_events, units, worlds)
    horizon_start = slots["interval_start_utc"].iat[0]
    settings_snapshot = {
        "model": model,
        "start_local_date": start,
        "warmup_days": 0,
        "study_days": STUDY_DAYS,
        "vehicle_count": evs,
        "seed": seed,
        "evaluation_world_count": worlds,
    }

    action_fields: dict[str, object] = dict.fromkeys(
        (
            "difference_bands",
            "difference_weekly",
            "difference_weekly_bands",
            "action_summary",
            "forecast_prices",
            "evaluation_prices",
            "price_shocks",
            "cost_effect",
            "cost_effect_summary",
            "not_recovered_summary",
            "smart_charging_world",
            "smart_charging_summary",
            "price_band_shift",
            "forecast_price_bands",
            "weekly_peak_summary",
            "price_relative_bands",
            "cheapest_half_hour_summary",
        )
    )
    # SYNTHETIC FIXTURE zones: 0.25 each; zone_1 gets a small headroom so
    # the above-headroom columns are exercised, the others have none.
    zone_world = _zone_world_frame(sim, slots, units, paths)
    zone_headroom = {zone_id: np.nan for zone_id in ZONE_IDS} | {"zone_1": 5.0}
    zone_fields = {
        "zone_ids": ZONE_IDS,
        "zone_world_intervals": zone_world,
        "zone_import_bands": zone_import_bands_from(zone_world, zone_headroom),
        "zone_summary": zone_summary_from(zone_world, dict.fromkeys(ZONE_IDS, 0.25), zone_headroom),
    }
    action_fields.update(
        dict.fromkeys(("events", "event_response_bands", "market_shocks", "shock_summary"))
    )
    not_recovered = None
    action_status = "not_applicable"
    if is_action:
        action_fields = _action_fields(sim, slots, units, world, worlds)
        action_fields.update(
            _trading_fields(
                world,
                slots,
                action_fields["forecast_prices"],
                action_fields["evaluation_prices"],
                units,
                sampled,
                action_fields["cost_effect"],
            )
        )
        not_recovered = int(action_fields["cost_effect"]["not_recovered_material"].sum())
        action_status = "selected"

    result = FixtureForecastResult(
        model=model,
        model_label="Explicit wholesale action" if is_action else "Explicit no-action",
        policy_id="explicit_synthetic_wholesale_v1" if is_action else "explicit_no_action_v1",
        evidence_kind="illustrative_synthetic" if is_action else "illustrative",
        run_completed_at_utc=pd.Timestamp("2026-09-28T09:00:00", tz="UTC"),
        seed=seed,
        vehicle_count=evs,
        world_count=worlds,
        study_start_local_date=start,
        horizon_start_utc=horizon_start,
        horizon_end_utc=slots["interval_end_utc"].iat[-1],
        study_slot_count=SLOT_COUNT,
        has_clock_change=clock_change is not None,
        settings_snapshot=settings_snapshot,
        assumptions=_assumptions(home_charger_kw),
        action_status=action_status,
        public_charging_status="modelled",
        not_recovered_world_count=not_recovered,
        representative_world_id=representative,
        study_slots=slots,
        units=units,
        cohort_summary=_cohort_summary(sim, units, all_events, world_kpis),
        fleet_world_intervals=world,
        fleet_interval_bands=fleet_bands(world),
        cohort_interval_bands=_cohort_bands(cohort_worlds),
        average_day_bands=_average_day_bands(sim, slots, units, paths, worlds),
        fleet_interval_ev_bands=_fleet_ev_bands(sim, slots, units, paths),
        flexibility_bands=_flexibility_bands(sim, slots, units, paths),
        weekly_bands=weekly_bands_from_world(world),
        plug_in_events=events,
        sampled_world_ids=sampled,
        plug_in_world_kpis=world_kpis,
        plug_in_summary=_plug_in_summary(all_events, world_kpis, worlds),
        plug_in_heatmap=_plug_in_heatmap(all_events, slots, evs, worlds),
        deferrable_power_bands=_deferrable_power_bands(deferrable, slots),
        flexibility_weekly_summary=_flexibility_weekly(deferrable, sim, slots, units),
        plug_in_half_hour_heatmap=plug_event_heatmap_from(
            all_events["world_id"].to_numpy(), all_events["plug_in_london"], slots, evs, worlds
        ),
        plug_out_heatmap=_plug_out_heatmap(sim, slots, evs, worlds),
        session_distribution_bands=_session_distribution_bands(
            all_events, units, worlds, sim.events.loc[sim.events["path_id"].eq("selected")]
        ),
        replay_state=sim,
        **zone_fields,
        **action_fields,
    )
    # Household contract v1 §8: frames built from the fixture's per-EV arrays.
    result = replace(result, **make_household_frames(result))
    if not is_action:
        return result
    # Replay contract v1 §4: the fixture's own replay, from its own frames.
    replay_week = make_replay_week(
        result,
        half_spread_gbp_per_mwh=_TRADING_TERMS["trading.half_spread_gbp_per_mwh"],
        supplier_compensation_gbp_per_mwh=_TRADING_TERMS[
            "trading.supplier_compensation_gbp_per_mwh"
        ],
        public_charge_gbp_per_kwh=_PUBLIC_RATE,
    )
    return replace(result, replay_week=replay_week)


def _study_slots(start: date) -> pd.DataFrame:
    # 336 fixed UTC half-hours from London noon of the start date (decision
    # 0004 items 1 and 52); London labels are derived, never stored instead.
    horizon_start = (pd.Timestamp(start) + pd.Timedelta(hours=12)).tz_localize(LONDON)
    starts = pd.date_range(horizon_start.tz_convert("UTC"), periods=SLOT_COUNT, freq=HALF_HOUR)
    london = starts.tz_convert(LONDON)
    last = start + timedelta(days=STUDY_DAYS - 1)
    # Reporting days are session nights (London time minus 12 h).  After a
    # spring change the last slots pass the last noon; they count to the last
    # night, as the model's build_study_slots does.
    shifted = (london.tz_localize(None) - pd.Timedelta(hours=12)).date
    local_date = [min(value, last) for value in shifted]
    return pd.DataFrame(
        {
            "slot_index": np.arange(SLOT_COUNT, dtype=np.int64),
            "interval_start_utc": starts,
            "interval_end_utc": starts + HALF_HOUR,
            "interval_start_london": london,
            "local_date": pd.Series(local_date, dtype=object),
            "night_index": np.array([(value - start).days for value in local_date], np.int64),
            "day_type": [_day_type(value) for value in local_date],
            "local_half_hour": np.asarray(london.hour * 2 + london.minute // 30, dtype=np.int64),
            "local_time_label": london.strftime("%H:%M"),
            "day_label": [_day_label(value) for value in local_date],
            # No holiday switch and no blackout window in the fixture.
            "holiday": np.zeros(SLOT_COUNT, dtype=bool),
            "blackout": np.zeros(SLOT_COUNT, dtype=bool),
        }
    )


def _published(slots: pd.DataFrame) -> np.ndarray:
    """Each slot's day-ahead publication instant: 13:00 London the day before delivery."""

    delivery = slots["interval_start_london"].dt.tz_localize(None).dt.normalize()
    published = (delivery - pd.Timedelta(days=1) + pd.Timedelta(hours=13)).dt.tz_localize(LONDON)
    return published.dt.tz_convert("UTC").to_numpy()


def _day_type(value: date) -> str:
    return "weekday" if value.weekday() < 5 else "weekend"


def _day_label(value: date) -> str:
    return f"{value:%a} {value.day}"


def _units(evs: int, home_charger_kw: float) -> pd.DataFrame:
    cohort_index = np.arange(evs) % len(COHORTS)
    cohort_ids = [COHORTS[i][0] for i in cohort_index]
    return pd.DataFrame(
        {
            "unit_id": [f"ev-{i:04d}" for i in range(evs)],
            "cohort_id": cohort_ids,
            "cohort_label": [COHORTS[i][1] for i in cohort_index],
            "daily_miles_mean": [_BEHAVIOUR[c]["trip_kwh"] * 3.5 for c in cohort_ids],
            "physical_capacity_kwh": [_BEHAVIOUR[c]["capacity"] for c in cohort_ids],
            "home_charger_limit_kw": np.full(evs, float(home_charger_kw)),
            "preferred_target_soc_percent": np.full(evs, 100.0 * _TARGET_SOC_FRACTION),
            "zone_id": _zone_ids(evs),
            # Intraday dispatch v1 §2: on every result; the fixture has no
            # dispatch, so no EV is locked.
            "dispatch_locked": np.zeros(evs, dtype=bool),
        }
    )


def _zone_ids(evs: int) -> list[str]:
    """SYNTHETIC FIXTURE zones: exact counts at 0.25 each, then a fixed shuffle.

    The shuffle uses its own generator (seed 0), so zones never move the
    toy's trips; the real model draws them last in the population instead.
    """

    counts = [evs // len(ZONE_IDS)] * len(ZONE_IDS)
    for index in range(evs - sum(counts)):
        counts[index] += 1
    zone_index = np.repeat(np.arange(len(ZONE_IDS)), counts)
    return [ZONE_IDS[i] for i in np.random.default_rng(0).permutation(zone_index)]


def _zone_world_frame(sim, slots, units, paths) -> pd.DataFrame:
    """Per-zone sums of the toy's per-EV arrays (trading contract §3), long by zone."""

    zones = units["zone_id"].to_numpy()
    frames = []
    for path in paths:
        for zone_id in ZONE_IDS:
            mask = zones == zone_id
            home = sim.home_import_kwh[path][..., mask].sum(axis=-1)
            worlds = home.shape[0]
            frames.append(
                pd.DataFrame(
                    {
                        "world_id": np.repeat(np.arange(worlds, dtype=np.int64), SLOT_COUNT),
                        "path_id": path,
                        "zone_id": zone_id,
                        "slot_index": np.tile(slots["slot_index"].to_numpy(), worlds),
                        "interval_start_utc": np.tile(
                            slots["interval_start_utc"].to_numpy(), worlds
                        ),
                        "unit_count": np.int64(mask.sum()),
                        "connected_count": sim.connected[path][..., mask]
                        .sum(axis=-1)
                        .astype(np.int64)
                        .reshape(-1),
                        "home_import_kwh": home.reshape(-1),
                        "home_import_kw": home.reshape(-1) / 0.5,
                        "early_departure_count": sim.early_count[path][..., mask]
                        .sum(axis=-1)
                        .astype(np.int64)
                        .reshape(-1),
                        "early_departure_shortfall_kwh": sim.early_shortfall_kwh[path][..., mask]
                        .sum(axis=-1)
                        .reshape(-1),
                    }
                )
            )
    frame = pd.concat(frames, ignore_index=True)
    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True)
    frame["interval_start_london"] = frame["interval_start_utc"].dt.tz_convert(LONDON)
    frame["_path_rank"] = frame["path_id"].map(PATH_ORDER.index)
    frame["_zone_rank"] = frame["zone_id"].map(ZONE_IDS.index)
    frame = frame.sort_values(["world_id", "_path_rank", "_zone_rank", "slot_index"], kind="stable")
    return typed_frame(frame.reset_index(drop=True), ZONE_WORLD_COLUMNS)


def _day_plan(cohort_id: str, e: int, w: int, d: int, day: date, seed: int) -> dict:
    """Toy plan for one EV on one London date (all values SYNTHETIC)."""

    behaviour = _BEHAVIOUR[cohort_id]
    jitter = (w + e + d + seed) % 3 - 1  # -1, 0 or +1 half-hour
    arrive = behaviour["arrive"] + 30 * jitter
    trip = behaviour["trip_kwh"] * (1.0 + 0.1 * ((w + 2 * e + d) % 3 - 1))
    drive_weekdays = behaviour.get("drive_weekdays")
    drives = drive_weekdays is None or day.weekday() in drive_weekdays
    plug_every = behaviour.get("plug_every", 1)
    if e == 0 and day.weekday() == 6:
        # EV 0 always arrives at 18:00 on Sunday, the last study date; in odd
        # worlds its long trip leaves it charging past the horizon end.
        arrive = 1080
        if w % 2 == 1:
            trip = 40.0
    return {
        "drives": drives,
        "depart": behaviour["depart"],
        "arrive": arrive,
        "trip_kwh": trip if drives else 0.0,
        "plugs": d % plug_every == plug_every - 1,
    }


def _simulate(
    slots: pd.DataFrame,
    units: pd.DataFrame,
    worlds: int,
    seed: int,
    *,
    with_action: bool,
) -> _Simulated:
    evs = len(units)
    dates = list(dict.fromkeys(slots["local_date"]))
    day_index = np.array([dates.index(value) for value in slots["local_date"]])
    minute = (slots["local_half_hour"] * 30).to_numpy()
    plans = [
        [
            [
                _day_plan(units["cohort_id"].iat[e], e, w, d, day, seed)
                for d, day in enumerate(dates)
            ]
            for e in range(evs)
        ]
        for w in range(worlds)
    ]
    runs = {
        "normal": _simulate_path(
            slots, units, plans, day_index, minute, "normal", smart=False, public_rescue=False
        )
    }
    if with_action:
        runs["selected"] = _simulate_path(
            slots, units, plans, day_index, minute, "selected", smart=True, public_rescue=True
        )
    return _Simulated(
        **{
            name: {path: run[name] for path, run in runs.items()}
            for name in (
                "connected",
                "driving",
                "away",
                "public_charging",
                "home_import_kwh",
                "public_import_kwh",
                "closing_kwh",
                "unserved_kwh",
                "early_count",
                "early_shortfall_kwh",
            )
        },
        events=pd.concat([run["events"] for run in runs.values()], ignore_index=True),
        audit=pd.concat([run["audit"] for run in runs.values()], ignore_index=True),
    )


def _simulate_path(slots, units, plans, day_index, minute, path_id, *, smart, public_rescue):
    """Step every EV through the 336 slots of every world for one path.

    Order within a slot: departure or arrival happens at the slot start, then
    charging for the rest of the slot.  Charging stops at the preferred
    target, which is not the physical ceiling.  ``smart`` limits charging to
    the toy's cheap night (02:00-06:00 London), a SYNTHETIC stand-in for
    smart charging; an EV that leaves below target with charging still owed
    counts as an early departure.  ``public_rescue`` adds one SYNTHETIC
    public top-up (EV 0, world 1, Sunday 17:00 London) so "home + public"
    import and the public cost component carry non-zero values on the
    selected path.
    """

    worlds, evs = len(plans), len(units)
    shape = (worlds, SLOT_COUNT, evs)
    out = {
        name: np.zeros(shape)
        for name in (
            "home_import_kwh",
            "public_import_kwh",
            "closing_kwh",
            "unserved_kwh",
            "early_count",
            "early_shortfall_kwh",
        )
    }
    for name in ("connected", "driving", "away", "public_charging"):
        out[name] = np.zeros(shape, dtype=bool)
    starts = slots["interval_start_utc"]
    dates = list(dict.fromkeys(slots["local_date"]))
    events, audit = [], []
    for w in range(worlds):
        for e in range(evs):
            unit = units.iloc[e]
            capacity = unit["physical_capacity_kwh"]
            charger_kw = unit["home_charger_limit_kw"]
            target = capacity * _TARGET_SOC_FRACTION
            stock, plugged, location = capacity * _OPENING_SOC_FRACTION, True, "home"
            session, leg_kwh = None, 0.0
            day_rows = {
                d: {"departure": pd.NaT, "departure_soc": np.nan, "shortfall": 0.0}
                for d in range(len(dates))
            }
            for t in range(SLOT_COUNT):
                d, m = day_index[t], minute[t]
                plan = plans[w][e][d]
                unserved = 0.0
                if plan["drives"] and m == plan["depart"]:
                    if smart and plugged and stock < target - 1e-9:
                        # Left before the smart plan finished: short by the rest.
                        out["early_count"][w, t, e] = 1.0
                        out["early_shortfall_kwh"][w, t, e] = target - stock
                    if session is not None:
                        session.update(plug_out_utc=starts.iat[t], plug_out_soc=stock / capacity)
                        events.append(session)
                    # A session already open at the horizon start has no
                    # plug-in inside the horizon, so it is not an event (3.9).
                    session, plugged = None, False
                    day_rows[d].update(departure=starts.iat[t], departure_soc=stock / capacity)
                    # The return leg uses the same trip, not the next night's plan.
                    leg_kwh = plan["trip_kwh"] / 2
                    location, stock, unserved = "driving", *_use(stock, leg_kwh)
                elif m == plan["arrive"] - 30 and location != "home":
                    # The return leg of the trip that left this morning.  A
                    # morning departure and its evening return fall in
                    # different session nights (decision 0004 item 52), so
                    # the toy follows the car rather than the evening's plan
                    # (the first evening starts at home).
                    location, stock, unserved = "driving", *_use(stock, leg_kwh)
                elif m == plan["arrive"]:
                    location = "home"
                    if plan["plugs"] and not plugged:
                        plugged = True
                        session = _open_session(w, path_id, unit, e, t, starts, slots, stock)
                elif location in ("driving", "public_charging"):
                    location = "away"
                public = 0.0
                if (
                    public_rescue
                    and (w, e) == (1, 0)
                    and dates[d].weekday() == 6
                    and location == "away"
                    and m == plan["arrive"] - 60
                ):
                    # 6 kWh is about 1.2 % of that week's normal home
                    # import, so the week is materially not recovered
                    # (decision 0004 item 45) and flagged views have one.
                    location, public = "public_charging", 6.0
                    stock += public * _CHARGE_EFFICIENCY
                day_rows[d]["shortfall"] += unserved
                connected = location == "home" and plugged
                grid = 0.0
                # Every night's cheap window sits inside the noon-to-noon study
                # (decision 0004 item 52), so the toy needs no last-day rule.
                night = _SMART_NIGHT_MINUTES[0] <= m < _SMART_NIGHT_MINUTES[1]
                if connected and stock < target and (not smart or night):
                    grid = min(charger_kw * 0.5, (target - stock) / _CHARGE_EFFICIENCY)
                    stock += grid * _CHARGE_EFFICIENCY
                    if session is not None:
                        session["grid"] += grid
                        session["battery"] += grid * _CHARGE_EFFICIENCY
                out["connected"][w, t, e] = connected
                out["driving"][w, t, e] = location == "driving"
                out["away"][w, t, e] = location in ("driving", "away", "public_charging")
                out["public_charging"][w, t, e] = location == "public_charging"
                out["home_import_kwh"][w, t, e] = grid
                out["public_import_kwh"][w, t, e] = public
                out["closing_kwh"][w, t, e] = stock
                out["unserved_kwh"][w, t, e] = unserved
            if session is not None:
                events.append(session)  # still plugged at the horizon end
            for d, day in enumerate(dates):
                plan, row = plans[w][e][d], day_rows[d]
                audit.append(
                    {
                        "path_id": path_id,
                        "world_id": w,
                        "unit_id": unit["unit_id"],
                        "local_date": day,
                        "drives_today": plan["drives"],
                        "departure_utc": row["departure"],
                        "target_soc_percent": 100.0 * _TARGET_SOC_FRACTION,
                        "departure_soc_percent": 100.0 * row["departure_soc"],
                        "trip_energy_need_kwh": plan["trip_kwh"],
                        "shortfall_kwh": row["shortfall"],
                    }
                )
    out["events"] = _events_frame(events, slots)
    out["audit"] = pd.DataFrame(audit)
    return out


def _use(stock: float, need: float) -> tuple[float, float]:
    # Travel draws the battery; what the battery cannot supply is unserved,
    # so stock never goes below zero (physical 0-100% SoC).
    served = min(stock, need)
    return stock - served, need - served


def _open_session(w, path_id, unit, e, t, starts, slots, stock) -> dict:
    return {
        "world_id": w,
        "path_id": path_id,
        "unit_id": unit["unit_id"],
        "cohort_id": unit["cohort_id"],
        "plug_in_utc": starts.iat[t],
        "local_date": slots["local_date"].iat[t],
        "plug_in_soc": stock / unit["physical_capacity_kwh"],
        "plug_out_utc": pd.NaT,
        "plug_out_soc": np.nan,
        "grid": 0.0,
        "battery": 0.0,
    }


def _sample_events(events: pd.DataFrame, world_ids: tuple[int, ...]) -> pd.DataFrame:
    """Keep the sampled worlds' events, in ``world_ids`` order (contract 3.9)."""

    rank = {world_id: position for position, world_id in enumerate(world_ids)}
    kept = events.loc[events["world_id"].isin(world_ids)].copy()
    kept["_rank"] = kept["world_id"].map(rank)
    kept = kept.sort_values(["_rank", "unit_id", "event_index"], kind="stable")
    return kept.drop(columns="_rank").reset_index(drop=True)


def _events_frame(events: list[dict], slots: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(events)
    if frame.empty:
        return empty_frame(PLUG_IN_EVENT_COLUMNS)
    frame = frame.sort_values(["world_id", "unit_id", "plug_in_utc"], kind="stable")
    frame = frame.reset_index(drop=True)
    plug_in = pd.to_datetime(frame["plug_in_utc"], utc=True)
    london = plug_in.dt.tz_convert(LONDON)
    result = pd.DataFrame(
        {
            "world_id": frame["world_id"].astype(np.int64),
            "path_id": frame["path_id"],
            "unit_id": frame["unit_id"],
            "cohort_id": frame["cohort_id"],
            "event_index": frame.groupby(["world_id", "unit_id"]).cumcount().astype(np.int64),
            "plug_in_utc": plug_in,
            "plug_in_london": london,
            "local_date": frame["local_date"],
            "day_type": [_day_type(value) for value in frame["local_date"]],
            "plug_in_local_hour": london.dt.hour.astype(np.int64),
            "plug_in_soc_percent": 100.0 * frame["plug_in_soc"].astype(float),
            "plug_out_utc": pd.to_datetime(frame["plug_out_utc"], utc=True),
            "plug_out_soc_percent": 100.0 * frame["plug_out_soc"].astype(float),
            "home_import_kwh": frame["grid"].astype(float),
            "battery_added_kwh": frame["battery"].astype(float),
            "still_plugged_at_horizon_end": frame["plug_out_utc"].isna().to_numpy(),
        }
    )
    return result.reset_index(drop=True)


# --------------------------------------------------------------------------
# Fleet and cohort frames
# --------------------------------------------------------------------------


def _world_frame(sim, slots, units, paths, mask) -> pd.DataFrame:
    """Per-world sums over the EVs in ``mask`` (contract 3.4), wide."""

    capacity = units["physical_capacity_kwh"].to_numpy()[mask].sum()
    count = int(mask.sum())
    frames = []
    for path in paths:
        home = sim.home_import_kwh[path][..., mask].sum(axis=-1)
        closing = sim.closing_kwh[path][..., mask].sum(axis=-1)
        public = sim.public_import_kwh[path][..., mask].sum(axis=-1)
        connected = sim.connected[path][..., mask].sum(axis=-1)
        worlds = home.shape[0]
        metrics = {
            "connected_count": connected.astype(np.int64),
            "connected_share": connected / count,
            "home_import_kwh": home,
            "home_import_kw": home / 0.5,
            "public_import_kwh": public,
            "total_import_kw": (home + public) / 0.5,
            "closing_battery_kwh": closing,
            "battery_soc_percent": 100.0 * closing / capacity,
            "driving_share": sim.driving[path][..., mask].mean(axis=-1),
            "away_share": sim.away[path][..., mask].mean(axis=-1),
            "public_charging_share": sim.public_charging[path][..., mask].mean(axis=-1),
            "unserved_travel_kwh": sim.unserved_kwh[path][..., mask].sum(axis=-1),
            "early_departure_count": sim.early_count[path][..., mask].sum(axis=-1).astype(np.int64),
            "early_departure_shortfall_kwh": sim.early_shortfall_kwh[path][..., mask].sum(axis=-1),
        }
        frame = pd.DataFrame(
            {
                "world_id": np.repeat(np.arange(worlds, dtype=np.int64), SLOT_COUNT),
                "path_id": path,
                **{key: np.tile(slots[key].to_numpy(), worlds) for key in SLOT_COLUMNS},
                "unit_count": np.int64(count),
                "physical_capacity_kwh": float(capacity),
                **{name: values.reshape(-1) for name, values in metrics.items()},
            }
        )
        for key in ("interval_start_utc", "interval_end_utc"):
            frame[key] = pd.to_datetime(frame[key], utc=True)
        frame["interval_start_london"] = frame["interval_start_utc"].dt.tz_convert(LONDON)
        frames.append(frame)
    world = pd.concat(frames, ignore_index=True)
    world["_path_rank"] = world["path_id"].map(PATH_ORDER.index)
    world = world.sort_values(["world_id", "_path_rank", "slot_index"], kind="stable")
    return world.drop(columns="_path_rank").reset_index(drop=True)


def _cohort_bands(cohort_worlds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for cohort_id, world in cohort_worlds.items():
        frame = fleet_bands(world)
        frame.insert(0, "cohort_id", cohort_id)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _representative_world(world: pd.DataFrame) -> int:
    # Lower median of weekly normal-path home import; ties go to the lower id.
    weekly = (
        world.loc[world["path_id"].eq("normal")]
        .groupby("world_id")["home_import_kwh"]
        .sum()
        .reset_index()
        .sort_values(["home_import_kwh", "world_id"], kind="stable")
    )
    return int(weekly["world_id"].iat[(len(weekly) - 1) // 2])


def _fleet_ev_bands(sim, slots, units, paths) -> pd.DataFrame:
    """Contract 3.6b: P10/P50/P90 across EVs per world and slot, then median across worlds."""

    capacity = units["physical_capacity_kwh"].to_numpy()
    frames = []
    for metric, unit in EV_BAND_METRICS.items():
        for path in paths:
            if metric == "battery_soc_percent":
                per_ev = 100.0 * sim.closing_kwh[path] / capacity
            else:
                per_ev = sim.home_import_kwh[path] / 0.5
            per_world = np.quantile(per_ev, (0.1, 0.5, 0.9), axis=2)
            p10, p50, p90 = np.median(per_world, axis=1)
            frame = slots.loc[:, list(SLOT_COLUMNS)].copy()
            frame.insert(0, "path_id", path)
            frame.insert(0, "unit", unit)
            frame.insert(0, "metric", metric)
            frame["world_count"] = np.int64(per_ev.shape[0])
            frame["p10"], frame["p50"], frame["p90"] = p10, p50, p90
            frames.append(frame)
    return pd.concat(frames, ignore_index=True).loc[:, list(FLEET_EV_BAND_COLUMNS)]


def _toy_charge_state(sim, slots, units, path):
    """(below target, grid kWh needed, hours to expected departure, power) on the toy.

    Opening stock is the previous slot's closing stock (the toy's opening
    SoC for slot 0).  The expected departure is the toy's departure minute
    minus the one-hour margin, on the first London date at least half an
    hour after the slot starts; London minutes are used directly, so a
    clock-change night is an hour out, which the toy accepts.
    """

    capacity = units["physical_capacity_kwh"].to_numpy()
    target = capacity * _TARGET_SOC_FRACTION
    power = units["home_charger_limit_kw"].to_numpy()
    expected = np.array(
        [_BEHAVIOUR[c]["depart"] - 60 * _DEPARTURE_MARGIN_HOURS for c in units["cohort_id"]]
    )
    minute = (slots["local_half_hour"] * 30).to_numpy()[:, np.newaxis]
    minutes_left = np.where(expected >= minute + 30, expected - minute, expected + 1440 - minute)
    closing = sim.closing_kwh[path]
    start = np.broadcast_to(capacity * _OPENING_SOC_FRACTION, closing[:, :1].shape)
    opening = np.concatenate([start, closing[:, :-1]], axis=1)
    below = sim.connected[path] & (opening < target)
    need = np.where(below, (target - opening) / _CHARGE_EFFICIENCY, 0.0)
    return below, need, minutes_left / 60.0, power


def _flexibility_bands(sim, slots, units, paths) -> pd.DataFrame:
    """Contract 3.6c on the toy (SYNTHETIC): same definitions as the model."""

    frames = []
    for metric, unit, spread in FLEXIBILITY_ROWS:
        for path in paths:
            below, need, hours_left, power = _toy_charge_state(sim, slots, units, path)
            flexible = np.where(below, power, 0.0).sum(axis=2)
            per_world = {
                "flexible_power_kw": flexible,
                "turn_up_headroom_kw": flexible - sim.home_import_kwh[path].sum(axis=2) / 0.5,
                "movable_energy_kwh": need.sum(axis=2),
            }.get(metric)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if per_world is not None:
                    p10, p50, p90 = np.quantile(per_world, (0.1, 0.5, 0.9), axis=0)
                    world_count = np.full(SLOT_COUNT, per_world.shape[0])
                else:
                    slack = np.where(below, hours_left[np.newaxis] - need / power, np.nan)
                    per_ev = np.nanquantile(slack, (0.1, 0.5, 0.9), axis=2)
                    world_count = np.isfinite(per_ev[1]).sum(axis=0)
                    if spread == "across_evs":
                        p10, p50, p90 = np.nanmedian(per_ev, axis=1)
                    else:
                        p10, p50, p90 = np.nanquantile(per_ev[1], (0.1, 0.5, 0.9), axis=0)
            frame = slots.loc[:, list(SLOT_COLUMNS)].copy()
            frame.insert(0, "path_id", path)
            frame.insert(0, "spread", spread)
            frame.insert(0, "unit", unit)
            frame.insert(0, "metric", metric)
            frame["world_count"] = world_count.astype(np.int64)
            frame["p10"], frame["p50"], frame["p90"] = p10, p50, p90
            frames.append(frame)
    return pd.concat(frames, ignore_index=True).loc[:, list(FLEXIBILITY_BAND_COLUMNS)]


def _deferrable_rows(sim, slots, units) -> dict[str, np.ndarray]:
    """Contract 3.6d on the toy (SYNTHETIC): unmanaged kW per world by whole-slot slack.

    Returns {row name: (world, slot)} for every ``DEFERRABLE_BUCKET_ORDER``
    row.  Whole-slot need is the number of half-hours at full power needed to
    reach target, times 0.5 h.  The cumulative rows test the slack directly.
    """

    below, need, hours_left, power = _toy_charge_state(sim, slots, units, "normal")
    slots_needed = np.ceil(np.round(need / (power * 0.5), 9))
    slack = hours_left[np.newaxis] - slots_needed * 0.5

    def kw(mask):
        return np.where(below & mask, power, 0.0).sum(axis=2)

    rows = {}
    lower = -np.inf
    for bucket, upper in SLACK_BUCKETS:
        rows[bucket] = kw((slack >= lower) & (slack < upper))
        lower = upper
    for hours in AT_LEAST_HOURS:
        rows[f"at_least_{hours}h"] = kw(slack >= hours)
    rows["total"] = kw(np.ones_like(below))
    return rows


def _deferrable_power_bands(deferrable: dict[str, np.ndarray], slots: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for bucket in DEFERRABLE_BUCKET_ORDER:
        per_world = deferrable[bucket]
        frame = slots.loc[:, list(SLOT_COLUMNS)].copy()
        frame.insert(0, "path_id", "normal")
        frame.insert(0, "unit", "kW")
        frame.insert(0, "slack_bucket", bucket)
        frame["world_count"] = per_world.shape[0]
        frame["p10"], frame["p50"], frame["p90"] = np.quantile(per_world, (0.1, 0.5, 0.9), axis=0)
        frames.append(frame)
    return typed_frame(pd.concat(frames, ignore_index=True), DEFERRABLE_POWER_BAND_COLUMNS)


def _flexibility_weekly(deferrable: dict[str, np.ndarray], sim, slots, units) -> pd.DataFrame:
    """Contract 3.6e on the toy (SYNTHETIC): per-week figures, then quantiles."""

    total = deferrable["total"]
    capacity = float(units["home_charger_limit_kw"].sum())
    _, need, _, _ = _toy_charge_state(sim, slots, units, "normal")
    plugged = sim.connected["normal"].sum(axis=2)
    half_hour = slots["local_half_hour"].to_numpy()
    per_world = {
        "peak_deferrable_kw": total.max(axis=1),
        "hours_at_least_quarter_capacity": 0.5
        * (total >= DEFERRABLE_CAPACITY_SHARE * capacity).sum(axis=1),
        "movable_energy_1800_kwh": need.sum(axis=2)[:, half_hour == MOVABLE_CHECK_HALF_HOUR].mean(
            axis=1
        ),
    }
    for time, index in REPORT_HALF_HOURS.items():
        at = half_hour == index
        per_world[f"deferrable_kw_{time}"] = total[:, at].mean(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_world[f"w_per_plugged_in_ev_{time}"] = (
                1000.0 * total[:, at].sum(axis=1) / plugged[:, at].sum(axis=1)
            )
    per_world["deferrable_at_least_2h_kw_1900"] = deferrable["at_least_2h"][
        :, half_hour == WAIT_CHECK_HALF_HOUR
    ].mean(axis=1)
    peak_slots = pd.Series(total.argmax(axis=1)).value_counts()
    modal = int(peak_slots[peak_slots == peak_slots.max()].index.min())
    row = {
        "path_id": "normal",
        "world_count": total.shape[0],
        "fleet_charger_capacity_kw": capacity,
        "peak_modal_slot_index": modal,
        "peak_modal_interval_start_utc": slots["interval_start_utc"].iat[modal],
        "peak_modal_interval_start_london": slots["interval_start_london"].iat[modal],
        "peak_modal_week_count": int(peak_slots[modal]),
    }
    for name, values in per_world.items():
        values = values[np.isfinite(values)]
        quantiles = np.quantile(values, (0.1, 0.5, 0.9)) if values.size else [np.nan] * 3
        for label, q in zip(("p10", "p50", "p90"), quantiles, strict=True):
            row[f"{name}_{label}"] = q
    return typed_frame(pd.DataFrame([row]), FLEXIBILITY_WEEKLY_COLUMNS)


def _average_day_bands(sim, slots, units, paths, worlds) -> pd.DataFrame:
    """Brief sketch 2 (contract 3.8): connected share across worlds, SoC across EVs."""

    groups = {"fleet": np.ones(len(units), dtype=bool)}
    for cohort_id in COHORT_ORDER:
        mask = units["cohort_id"].eq(cohort_id).to_numpy()
        if mask.any():
            groups[cohort_id] = mask
    half_hour = slots["local_half_hour"].to_numpy()
    day_type = slots["day_type"].to_numpy()
    dates = slots.drop_duplicates("local_date")
    capacity = units["physical_capacity_kwh"].to_numpy()
    labels = [f"{h // 2:02d}:{30 * (h % 2):02d}" for h in range(48)]
    rows = []
    for group_id, mask in groups.items():
        for path in paths:
            connected = sim.connected[path][..., mask].mean(axis=-1)  # (world, slot)
            soc = 100.0 * sim.closing_kwh[path][..., mask] / capacity[mask]
            soc_stats = {
                "mean": soc.mean(axis=-1),
                "p5": np.percentile(soc, 5, axis=-1),
                "p95": np.percentile(soc, 95, axis=-1),
            }
            for kind in DAY_TYPE_ORDER:
                in_type = np.ones(SLOT_COUNT, bool) if kind == "all" else day_type == kind
                day_count = len(dates) if kind == "all" else int(dates["day_type"].eq(kind).sum())
                for h in range(48):
                    chosen = in_type & (half_hour == h)
                    base = {
                        "group_id": group_id,
                        "path_id": path,
                        "day_type": kind,
                        "local_half_hour": h,
                        "local_time_label": labels[h],
                        "world_count": worlds,
                        "day_count": day_count,
                    }
                    # A half-hour absent on every day of this type (spring
                    # change) is missing, never zero.
                    share = connected[:, chosen].mean(axis=1) if chosen.any() else None
                    q = (
                        np.quantile(share, (0.1, 0.5, 0.9), method="linear")
                        if share is not None
                        else [np.nan] * 3
                    )
                    rows.append(
                        {
                            **base,
                            "metric": "connected_share",
                            "unit": "fraction",
                            "spread": "across_worlds",
                            "centre_stat": "p50",
                            "low_stat": "p10",
                            "high_stat": "p90",
                            "centre": q[1],
                            "low": q[0],
                            "high": q[2],
                        }
                    )
                    world_medians = {
                        stat: np.median(values[:, chosen].mean(axis=1)) if chosen.any() else np.nan
                        for stat, values in soc_stats.items()
                    }
                    rows.append(
                        {
                            **base,
                            "metric": "battery_soc_percent",
                            "unit": "percent",
                            "spread": "across_evs",
                            "centre_stat": "mean_across_evs",
                            "low_stat": "p5_across_evs",
                            "high_stat": "p95_across_evs",
                            "centre": world_medians["mean"],
                            "low": world_medians["p5"],
                            "high": world_medians["p95"],
                        }
                    )
    frame = pd.DataFrame(rows)
    frame["_metric_rank"] = frame["metric"].map(list(FLEET_METRICS).index)
    frame = frame.sort_values(
        ["group_id", "path_id", "day_type", "_metric_rank", "local_half_hour"],
        key=lambda column: column.map(_order_key) if column.name != "local_half_hour" else column,
        kind="stable",
    )
    frame = frame.drop(columns="_metric_rank").loc[:, list(AVERAGE_DAY_COLUMNS)]
    for name in ("local_half_hour", "world_count", "day_count"):
        frame[name] = frame[name].astype(np.int64)
    return frame.reset_index(drop=True)


def _order_key(value):
    for order in (("fleet", *COHORT_ORDER), PATH_ORDER, DAY_TYPE_ORDER):
        if value in order:
            return order.index(value)
    return value


# --------------------------------------------------------------------------
# Plug-in summaries
# --------------------------------------------------------------------------


def _plug_in_summary(events: pd.DataFrame, world_kpis: pd.DataFrame, worlds: int):
    """Histograms from every world's normal-path events; KPIs from world KPIs (3.10).

    Per world first: each world's share of its own plug-ins in each bin, then
    quantiles across worlds.  ``share_mean`` sums to 1 over the bins of a day
    type; the P10/P50/P90 columns do not, and must not be summed.
    """

    hour_rows, soc_rows = [], []
    soc_bin = np.minimum(events["plug_in_soc_percent"] // 5, 19).astype(int) * 5
    for kind in DAY_TYPE_ORDER:
        in_kind = np.ones(len(events), bool) if kind == "all" else events["day_type"].eq(kind)
        per_world = [
            events.index[in_kind & events["world_id"].eq(w)]
            for w in range(worlds)
            if (in_kind & events["world_id"].eq(w)).any()
        ]
        for hour in range(24):
            counts = [events.loc[rows, "plug_in_local_hour"].eq(hour).sum() for rows in per_world]
            hour_rows.append(
                {
                    "day_type": kind,
                    "local_hour": hour,
                    "world_count": len(per_world),
                    "count_p50": float(np.quantile(counts, 0.5)) if per_world else np.nan,
                    **_share_stats([c / len(rows) for c, rows in zip(counts, per_world)]),
                }
            )
        for lower in range(0, 100, 5):
            shares = [soc_bin.loc[rows].eq(lower).mean() for rows in per_world]
            soc_rows.append(
                {
                    "day_type": kind,
                    "soc_bin_lower_percent": lower,
                    "soc_bin_upper_percent": lower + 5,
                    "world_count": len(per_world),
                    **_share_stats(shares),
                }
            )
    return FixturePlugInSummary(
        hour_bands=typed_frame(pd.DataFrame(hour_rows), HOUR_BAND_COLUMNS),
        soc_bands=typed_frame(pd.DataFrame(soc_rows), SOC_BAND_COLUMNS),
        kpis=plug_in_kpis_from_world_kpis(world_kpis),
    )


def _plug_in_heatmap(events: pd.DataFrame, slots: pd.DataFrame, evs: int, worlds: int):
    """Contract 3.10a on the toy (SYNTHETIC): plug-ins per EV-day by weekday and hour."""

    counts = np.zeros((worlds, 7, 24))
    weekday = events["local_date"].map(lambda value: value.weekday()).to_numpy(dtype=int)
    hour = events["plug_in_london"].dt.hour.to_numpy(dtype=int)
    np.add.at(counts, (events["world_id"].to_numpy(dtype=int), weekday, hour), 1.0)
    day_count = np.bincount([d.weekday() for d in dict.fromkeys(slots["local_date"])], minlength=7)
    chance = counts / (evs * day_count[np.newaxis, :, np.newaxis])
    p10, p50, p90 = np.quantile(chance, (0.1, 0.5, 0.9), axis=0)
    return pd.DataFrame(
        {
            "weekday": np.repeat(np.arange(7, dtype=np.int64), 24),
            "weekday_label": pd.Series(np.repeat(WEEKDAY_LABELS, 24), dtype=object),
            "local_hour": np.tile(np.arange(24, dtype=np.int64), 7),
            "day_count": np.repeat(day_count.astype(np.int64), 24),
            "world_count": np.int64(worlds),
            "mean": chance.mean(axis=0).ravel(),
            "p10": p10.ravel(),
            "p50": p50.ravel(),
            "p90": p90.ravel(),
        }
    ).loc[:, list(PLUG_IN_HEATMAP_COLUMNS)]


def _plug_out_heatmap(sim, slots: pd.DataFrame, evs: int, worlds: int) -> pd.DataFrame:
    """Contract 3.10b plug-outs on the toy (SYNTHETIC), from the connection state.

    An unplug is a connected slot followed by an unconnected one; it happens
    at the later slot's start.  Sessions already plugged in at the horizon
    start count too (their departure is observed).
    """

    connected = sim.connected["normal"]
    world_ids, slot_before, _ = np.nonzero(connected[:, :-1, :] & ~connected[:, 1:, :])
    when = slots["interval_start_london"].iloc[slot_before + 1].reset_index(drop=True)
    return plug_event_heatmap_from(world_ids, when, slots, evs, worlds)


def _session_distribution_bands(
    events: pd.DataFrame, units: pd.DataFrame, worlds: int, smart_events: pd.DataFrame
):
    """Contract 3.10c from every world's toy sessions, with the contract's bin labels.

    ``smart_events`` are the toy selected path's sessions (empty on a
    no-action fixture), for the supplier §4.6 smart departure SoC.
    """

    bands = session_distribution_bands_from(
        events, units, worlds, _CHARGE_EFFICIENCY, smart_events if len(smart_events) else None
    )
    bands["bin_label"] = pd.Series(
        [
            _session_bin_label(metric, index)
            for metric, index in zip(bands["metric"], bands["bin_index"], strict=True)
        ],
        dtype=object,
    )
    return bands


def _session_bin_label(metric: str, index: int) -> str:
    _, lower, width, count = SESSION_DISTRIBUTION_METRICS[metric]
    low, high = lower + index * width, lower + (index + 1) * width
    if metric.endswith("_time"):
        return f"{int(low):02d}:{30 * int(2 * low % 2):02d}"
    if metric == "slack_hours" and index == 0:
        return f"< {high:g}"
    if "soc_percent" not in metric and index == count - 1:  # SoC bins are closed at 100 %
        return f"≥ {low:g}"
    return f"{low:g}–{high:g}"


def _share_stats(shares) -> dict[str, float]:
    if len(shares) == 0:
        return dict.fromkeys(("share_mean", "share_p10", "share_p50", "share_p90"), np.nan)
    values = np.asarray(shares, dtype=float)
    q = np.quantile(values, (0.1, 0.5, 0.9))
    return {"share_mean": values.mean(), "share_p10": q[0], "share_p50": q[1], "share_p90": q[2]}


def _cohort_summary(sim, units, events, world_kpis) -> pd.DataFrame:
    rows = []
    for cohort_id, label, share in COHORTS:
        mask = units["cohort_id"].eq(cohort_id).to_numpy()
        count = int(mask.sum())
        row = {
            "cohort_id": cohort_id,
            "cohort_label": label,
            "source_population_share": share,
            "ev_count": count,
            "plug_ins_per_ev_week_p50": np.nan,
            "median_plug_in_soc_percent_p50": np.nan,
            "peak_plug_in_local_hour": pd.NA,
            "unserved_travel_kwh_p50": np.nan,
        }
        if count:
            row.update(cohort_plug_in_stats(world_kpis, cohort_id))
            weekly = sim.unserved_kwh["normal"][..., mask].sum(axis=(1, 2))
            row["unserved_travel_kwh_p50"] = np.quantile(weekly, 0.5)
        rows.append(row)
    return typed_frame(pd.DataFrame(rows), COHORT_SUMMARY_COLUMNS)


# --------------------------------------------------------------------------
# Action fields
# --------------------------------------------------------------------------


def _action_fields(sim, slots, units, world, worlds) -> dict[str, object]:
    minute = (slots["local_half_hour"] * 30).to_numpy()
    starts = slots["interval_start_utc"]
    # SYNTHETIC price shape peaking at 18:00 London (cheapest at 06:00), so
    # the toy's night charging is cheaper.  Each world's day-ahead path adds
    # its own small repeating offset (decision 0004 item 48); its intraday
    # close adds a second one, and imbalance a long/short premium (item 53).
    # The net demand is a hand-made linear inverse of the price: 5 £/MWh per
    # GW around 20 GW at £80.
    shape = 90.0 + 45.0 * np.cos(2 * np.pi * (minute / 60.0 - 18.0) / 24.0)
    world_slot = np.arange(worlds)[:, None] + np.arange(SLOT_COUNT)[None, :]
    day_ahead = shape[None, :] + 4.0 * (world_slot % 3 - 1)
    realised = day_ahead + 6.0 * (world_slot % 5 - 2)
    system_long = world_slot % 2 == 0
    imbalance = realised + np.where(system_long, -17.0, 20.0)
    per_world = {
        "world_id": np.repeat(np.arange(worlds, dtype=np.int64), SLOT_COUNT),
        "slot_index": np.tile(slots["slot_index"].to_numpy(), worlds),
        "interval_start_utc": np.tile(starts.to_numpy(), worlds),
        "interval_end_utc": np.tile(slots["interval_end_utc"].to_numpy(), worlds),
    }
    forecast_prices = pd.DataFrame(
        {
            **per_world,
            "interval_start_london": np.tile(slots["interval_start_london"].to_numpy(), worlds),
            # Plan B4: published 13:00 London the day before delivery.
            "forecast_available_at_utc": np.tile(_published(slots), worlds),
            "wholesale_forecast_gbp_per_mwh": day_ahead.reshape(-1),
            "system_net_demand_gw": 20.0 + (day_ahead.reshape(-1) - 80.0) / 5.0,
            "evidence_kind": "synthetic",
        }
    )
    evaluation_prices = pd.DataFrame(
        {
            **per_world,
            "evaluation_context_price_gbp_per_mwh": realised.reshape(-1),
            "imbalance_price_gbp_per_mwh": imbalance.reshape(-1),
            "system_long": system_long.reshape(-1),
            "outcome_available_at_utc": np.tile(slots["interval_end_utc"].to_numpy(), worlds),
            "evidence_kind": "synthetic",
        }
    )
    for frame in (forecast_prices, evaluation_prices):
        for key in (
            "interval_start_utc",
            "interval_end_utc",
            "outcome_available_at_utc",
            "forecast_available_at_utc",
        ):
            if key in frame:
                frame[key] = pd.to_datetime(frame[key], utc=True)
    forecast_prices["interval_start_london"] = pd.to_datetime(
        forecast_prices["interval_start_london"], utc=True
    ).dt.tz_convert("Europe/London")

    # One SYNTHETIC big known up shock per world at 18:00 on day one (item 56).
    price_shocks = typed_frame(
        pd.DataFrame(
            {
                "world_id": np.arange(worlds, dtype=np.int64),
                "shock_class": "big",
                "start_slot_index": 36,
                "start_utc": starts.iat[36],
                "duration_slots": 4,
                "size_gw": 4.0,
                "direction": "up",
                "known_day_ahead": True,
                "evidence_kind": "synthetic",
            }
        ),
        PRICE_SHOCK_COLUMNS,
    )
    market_shocks = typed_frame(
        price_shocks.assign(
            shock_id=[
                f"s{w}-{n}"
                for w, n in zip(
                    price_shocks["world_id"],
                    price_shocks.groupby("world_id").cumcount(),
                    strict=True,
                )
            ],
            source="stochastic",
            start_london=price_shocks["start_utc"].dt.tz_convert(LONDON),
            in_study=True,
            peak_price_increment_gbp_per_mwh=0.0,
        ),
        MARKET_SHOCK_COLUMNS,
    )
    # Home import is valued at the day-ahead price (plan B3, item 53).
    cost = _cost_effect(world, forecast_prices, _sessions_affected(sim), public_rate=_PUBLIC_RATE)
    summary = FixtureActionSummary(
        action_id=ACTION_ID,
        action_label="Smart home charging in the cheapest forecast half-hours before departure",
        departure_margin_hours=_DEPARTURE_MARGIN_HOURS,
        forecast_available_at_utc=forecast_prices["forecast_available_at_utc"].iat[0],
        vehicle_count=len(units),
    )
    weekly = difference_weekly_from_world(world)
    outcomes = smart_charging_world_from(world, forecast_prices)
    return {
        "difference_bands": difference_bands_from_world(world),
        "difference_weekly": weekly,
        "difference_weekly_bands": weekly_difference_bands(weekly),
        "action_summary": summary,
        "forecast_prices": forecast_prices,
        "evaluation_prices": evaluation_prices,
        "price_shocks": price_shocks,
        "cost_effect": cost,
        "cost_effect_summary": cost_summary_from_cost_effect(cost),
        "not_recovered_summary": not_recovered_summary_from(cost),
        "smart_charging_world": outcomes,
        "smart_charging_summary": smart_charging_summary_from(outcomes),
        "price_band_shift": price_band_shift_from(world, forecast_prices, slots),
        "forecast_price_bands": forecast_price_bands_from(forecast_prices),
        "weekly_peak_summary": weekly_peak_summary_from(
            world, float(units["home_charger_limit_kw"].iat[0])
        ),
        "price_relative_bands": price_relative_bands_from(forecast_prices),
        "cheapest_half_hour_summary": cheapest_half_hour_from(forecast_prices),
        # No scripted events in the fixture: an empty events table and no
        # response blocks.  Its market shocks are the toy price_shocks as
        # stochastic rows with a 0 price increment (SYNTHETIC FIXTURE).
        "events": empty_frame(EVENT_TABLE_COLUMNS),
        "event_response_bands": empty_frame(EVENT_RESPONSE_COLUMNS),
        "market_shocks": market_shocks,
        "shock_summary": shock_summary_from(market_shocks, worlds),
    }


def _sessions_affected(sim: _Simulated) -> np.ndarray:
    """Sessions per world ending with less stock on the selected path (decision 0004 item 45).

    Written independently of the model: a session ends in a connected slot
    whose next slot is not connected (or the horizon ends).
    """

    connected = sim.connected["normal"]
    ends = connected.copy()
    ends[:, :-1, :] &= ~connected[:, 1:, :]
    short = sim.closing_kwh["normal"] - sim.closing_kwh["selected"] > NOT_RECOVERED_TOLERANCE_KWH
    return (ends & short).sum(axis=(1, 2)).astype(np.int64)


def _cost_effect(
    world: pd.DataFrame,
    prices: pd.DataFrame,
    sessions_affected: np.ndarray,
    *,
    public_rate: float,
) -> pd.DataFrame:
    """Same formulas as ``calculate_wholesale_world_cost_effect`` (items 4, 13, 45)."""

    normal = world.loc[world["path_id"].eq("normal")].set_index(["world_id", "slot_index"])
    selected = world.loc[world["path_id"].eq("selected")].set_index(["world_id", "slot_index"])
    price = prices.set_index(["world_id", "slot_index"])["wholesale_forecast_gbp_per_mwh"]
    energy_cost = (
        ((selected["home_import_kwh"] - normal["home_import_kwh"]) * price / 1000.0)
        .groupby("world_id")
        .sum()
    )
    by_world = lambda frame, column: frame[column].groupby("world_id").sum()  # noqa: E731
    closing = (
        (selected["closing_battery_kwh"] - normal["closing_battery_kwh"]).groupby("world_id").last()
    )
    public = by_world(selected, "public_import_kwh") - by_world(normal, "public_import_kwh")
    unserved = by_world(selected, "unserved_travel_kwh") - by_world(normal, "unserved_travel_kwh")
    shortfall = (-closing).where(-closing > NOT_RECOVERED_TOLERANCE_KWH, 0.0)
    added_unserved = unserved.where(unserved > NOT_RECOVERED_TOLERANCE_KWH, 0.0)
    frame = pd.DataFrame(
        {
            "world_id": energy_cost.index.to_numpy(dtype=np.int64),
            "normal_home_import_kwh": by_world(normal, "home_import_kwh").to_numpy(),
            "selected_home_import_kwh": by_world(selected, "home_import_kwh").to_numpy(),
            "illustrative_selected_minus_normal_energy_cost_gbp": energy_cost.to_numpy(),
            "selected_minus_normal_closing_battery_kwh": closing.to_numpy(),
            "selected_minus_normal_public_import_kwh": public.to_numpy(),
            "selected_minus_normal_unserved_travel_kwh": unserved.to_numpy(),
            "illustrative_selected_minus_normal_public_charge_cost_gbp": (
                public * public_rate
            ).to_numpy(),
            "illustrative_unrecovered_energy_value_gbp": (shortfall * public_rate).to_numpy(),
            "illustrative_unserved_travel_value_gbp": (added_unserved * public_rate).to_numpy(),
        }
    )
    frame["illustrative_selected_minus_normal_total_gbp"] = frame[
        [column for _, column in COST_COMPONENTS[:4]]
    ].sum(axis=1)
    frame["energy_not_recovered"] = (
        (-closing > NOT_RECOVERED_TOLERANCE_KWH)
        | (public > NOT_RECOVERED_TOLERANCE_KWH)
        | (unserved > NOT_RECOVERED_TOLERANCE_KWH)
    ).to_numpy()
    extra_public = public.where(public > NOT_RECOVERED_TOLERANCE_KWH, 0.0)
    unrecovered = (shortfall + extra_public + added_unserved).to_numpy()
    normal_import = frame["normal_home_import_kwh"].to_numpy()
    share = np.where(
        normal_import > 0.0, unrecovered / np.where(normal_import > 0.0, normal_import, 1.0), np.nan
    )
    frame["unrecovered_kwh"] = unrecovered
    frame["unrecovered_share"] = share
    frame["not_recovered_material"] = (unrecovered > 0.0) & ~(
        100.0 * share < NOT_RECOVERED_MATERIAL_SHARE_PERCENT
    )
    frame["sessions_affected_count"] = sessions_affected
    frame["evidence_kind"] = "illustrative_synthetic"
    return frame.loc[:, list(COST_EFFECT_COLUMNS)]


# --------------------------------------------------------------------------
# Assumptions
# --------------------------------------------------------------------------


def _assumptions(home_charger_kw: float) -> tuple[Assumption, ...]:
    cnz = "CNZ report (see model/assumptions.py CNZ_CONTEXT); context, not a calibration target"
    return (
        Assumption(
            name="home_charger_kw",
            label="Home charger power",
            value=float(home_charger_kw),
            unit="kW",
            evidence="source",
            source="Axle cohort sheet: 7 kW home chargers",
            meaning="Maximum home import power per EV.",
            editable=True,
            bounds=(2.3, 22.0),
            group="Plugging & home charging",
            affects="How fast EVs recover energy after plugging in.",
        ),
        Assumption(
            name="not_recovered_material_share_percent",
            label="Material not-recovered share",
            value=NOT_RECOVERED_MATERIAL_SHARE_PERCENT,
            unit="percent",
            evidence="illustrative",
            source="illustrative choice (decision 0004 item 45)",
            meaning="A week is materially not recovered from this share of weekly home import.",
            editable=True,
            bounds=(0.0, 100.0),
            group="Simulation",
            affects="Reporting only: which weeks count as materially not recovered.",
        ),
        Assumption(
            name="public_charge_gbp_per_kwh",
            label="Public charge rate",
            value=_PUBLIC_RATE,
            unit="GBP/kWh",
            evidence="illustrative",
            source="illustrative choice (decision 0004 item 4)",
            meaning="Values public charging, unrecovered energy and unserved travel.",
            editable=True,
            bounds=(0.0, 5.0),
            group="Public & prices",
            affects="Illustrative cost components only; not sampling or the action choice.",
        ),
        *(
            Assumption(
                name=name,
                label=name,
                value=value,
                unit="",
                evidence="illustrative",
                source="SYNTHETIC FIXTURE: the model's illustrative default (contract 7, 9.9)",
                meaning="Trading ledger term.",
                editable=False,
                bounds=None,
                group="Public & prices",
                affects="Trading ledger only.",
            )
            for name, value in _TRADING_TERMS.items()
        ),
        # The model's own supplier and carbon records and the supply-curve
        # reference, so the supplier validator checks the payment, fee and
        # carbon rules on the fixture too.
        *assumptions.SUPPLIER.values(),
        *assumptions.CARBON.values(),
        assumptions.PRICES["supply_reference_net_demand_gw"],
        Assumption(
            name="cohort_population_shares",
            label="Archetype shares",
            value=tuple(share for _, _, share in COHORTS),
            unit="fraction",
            evidence="source",
            source="Axle cohort sheet rows 6-11",
            meaning="Share of the fleet in each of the six archetypes.",
            editable=False,
            bounds=None,
            group="Fleet",
            affects="How many EVs of each archetype are sampled.",
        ),
        Assumption(
            name="synthetic_price_base_gbp_per_mwh",
            label="Synthetic price level",
            value=90.0,
            unit="GBP/MWh",
            evidence="synthetic",
            source="synthetic choice; not a market forecast",
            meaning="Mean level of the synthetic wholesale price path.",
            editable=True,
            bounds=(0.0, 500.0),
            group="Public & prices",
            affects="Expected value of the action and illustrative energy cost.",
        ),
        *(
            Assumption(
                name=name,
                label=label,
                value=value,
                unit=unit,
                evidence="source",
                source=cnz,
                meaning=meaning,
                editable=False,
                bounds=None,
                group="Plugging & home charging",
                affects="Reference marker on the plug-in charts only.",
            )
            for name, label, value, unit, meaning in (
                (
                    CNZ_RECORD_NAMES[0],
                    "CNZ median SoC at plug-in",
                    52.0,
                    "percent",
                    "Reported median SoC at plug-in.",
                ),
                (
                    CNZ_RECORD_NAMES[1],
                    "CNZ plug-ins below 10% SoC",
                    0.03,
                    "fraction",
                    "Reported share of plug-ins below 10% SoC.",
                ),
                (
                    CNZ_RECORD_NAMES[2],
                    "CNZ plug-ins below 20% SoC (upper bound)",
                    0.10,
                    "fraction",
                    "Reported as fewer than 10% below 20% SoC: an upper bound.",
                ),
                (
                    CNZ_RECORD_NAMES[3],
                    "CNZ weekday plug-in mode",
                    18,
                    "hour (London)",
                    "Reported weekday plug-in mode, about 18:00.",
                ),
            )
        ),
    )


# --------------------------------------------------------------------------
# Trading overlay (trading contract v1 §5 and §9): SYNTHETIC FIXTURE frames
# --------------------------------------------------------------------------

# SYNTHETIC ledger terms, the model's illustrative defaults (contract §7, §9.9).
_TRADING_TERMS = {
    "trading.half_spread_gbp_per_mwh": 1.0,
    "trading.customer_revenue_share": 0.5,
    "trading.supplier_compensation_gbp_per_mwh": 0.0,
    "trading.unmet_charge_penalty_gbp_per_kwh": 0.79,
    "trading.day_ahead_commitment_share": 0.8,
}
# Trading frames label every row as model-shaped illustrative output; the
# fixture's own status is in ``fixture_label``.
_EVIDENCE = "illustrative_synthetic"


def _trading_fields(
    world: pd.DataFrame,
    slots: pd.DataFrame,
    forecast_prices: pd.DataFrame,
    evaluation_prices: pd.DataFrame,
    units: pd.DataFrame,
    sampled: tuple[int, ...],
    cost: pd.DataFrame,
) -> dict[str, object]:
    """SYNTHETIC FIXTURE trading frames, not the model's rules.

    The toy baseline is each world's mean unmanaged import at the same London
    half-hour over the week (not BL01-lite); the day-ahead position is 80% of
    the baseline less the across-world mean metered import; the full strategy
    moves half-way from its forecast to the settled volume in one intraday
    trade, two hours before delivery, from the third slot of each night on.
    Every frame is then derived with the contract's reference arithmetic, so
    the fixture is internally consistent for the views.
    """

    worlds = int(world["world_id"].nunique())
    unmanaged = metric_matrix(world, "normal", "home_import_kwh")
    metered = metric_matrix(world, "selected", "home_import_kwh")
    half_hour = slots["local_half_hour"].to_numpy()
    night = slots["night_index"].to_numpy()
    baseline = np.zeros_like(unmanaged)
    for h in np.unique(half_hour):
        baseline[:, half_hour == h] = unmanaged[:, half_hour == h].mean(axis=1, keepdims=True)
    settled = np.maximum(baseline - metered, 0.0)
    commitment = _TRADING_TERMS["trading.day_ahead_commitment_share"]
    forecast_target = np.maximum(baseline - metered.mean(axis=0, keepdims=True), 0.0)
    q = commitment * forecast_target
    position_in_night = np.arange(SLOT_COUNT) - np.searchsorted(night, night)
    revised = position_in_night >= 2
    full = np.where(revised, 0.5 * forecast_target + 0.5 * settled, q)

    def price(frame: pd.DataFrame, column: str) -> np.ndarray:
        ordered = frame.sort_values(["world_id", "slot_index"])
        return ordered[column].to_numpy(dtype=float).reshape(worlds, SLOT_COUNT)

    day_ahead = price(forecast_prices, "wholesale_forecast_gbp_per_mwh")
    close = price(evaluation_prices, "evaluation_context_price_gbp_per_mwh")
    imbalance = price(evaluation_prices, "imbalance_price_gbp_per_mwh")
    world_slot = np.arange(worlds)[:, None] + np.arange(SLOT_COUNT)[None, :]
    intraday_price = day_ahead + (world_slot % 3 - 1.0)
    trade = full - q
    known = np.zeros_like(unmanaged)
    known[:, 36:40] = 4.0  # the fixture's big known shock (price_shocks)
    surprise = np.zeros_like(unmanaged)
    kind = np.where(surprise != 0.0, "surprise", np.where(known != 0.0, "known", "none"))
    starts = slots["interval_start_utc"]
    per_slot = {
        "world_id": np.repeat(np.arange(worlds, dtype=np.int64), SLOT_COUNT),
        "slot_index": np.tile(slots["slot_index"].to_numpy(), worlds),
        "interval_start_utc": np.tile(starts.to_numpy(), worlds),
        "interval_start_london": np.tile(slots["interval_start_london"].to_numpy(), worlds),
        "night_index": np.tile(night, worlds),
    }
    values = {
        "baseline_kwh": baseline,
        "unmanaged_kwh": unmanaged,
        "metered_kwh": metered,
        "baseline_kw": baseline / 0.5,
        "unmanaged_kw": unmanaged / 0.5,
        "metered_kw": metered / 0.5,
        "expected_metered_kwh": np.broadcast_to(metered.mean(axis=0), metered.shape),
        # Toy x^U = x: no expected turn-down, so each night's sum is 0.
        "expected_unmanaged_kwh": np.broadcast_to(metered.mean(axis=0), metered.shape),
        "deviation_kwh": baseline - metered,
        "true_reduction_kwh": unmanaged - metered,
        "baseline_effect_kwh": baseline - unmanaged,
        "settled_kwh": settled,
        "settlement_open": np.ones_like(settled, dtype=bool),
        "position_da_only_kwh": q,
        "position_full_kwh": full,
        "position_perfect_foresight_kwh": settled,
        # Fixed-share mode (the default): no newsvendor level (§10.4).
        "commit_level": np.full_like(settled, np.nan),
        "forecast_error_quantile_kwh": np.full_like(settled, np.nan),
        "day_ahead_gbp_per_mwh": day_ahead,
        "intraday_close_gbp_per_mwh": close,
        "imbalance_gbp_per_mwh": imbalance,
        "known_shock_gw": known,
        "surprise_shock_gw": surprise,
        "shock_kind": kind.astype(object),
    }
    deviation = pd.DataFrame(
        {**per_slot, **{k: v.reshape(-1) for k, v in values.items()}, "evidence_kind": _EVIDENCE}
    )
    deviation["interval_start_utc"] = pd.to_datetime(deviation["interval_start_utc"], utc=True)
    deviation["interval_start_london"] = deviation["interval_start_utc"].dt.tz_convert(LONDON)
    deviation = typed_frame(deviation, DEVIATION_WORLD_SLOT_COLUMNS)

    # Position updates for the sampled worlds: the day-ahead decision at
    # 13:00 London the day before the night, then one intraday trade.
    decisions = []
    london = slots["interval_start_london"]
    night_date = slots["local_date"]
    for w in sampled:
        for t in range(SLOT_COUNT):
            gate = starts.iat[t] - pd.Timedelta(minutes=60)
            da_decision = (
                pd.Timestamp(night_date.iat[t]) - pd.Timedelta(days=1) + pd.Timedelta(hours=13)
            ).tz_localize(LONDON)
            rows = [("day_ahead", da_decision.tz_convert("UTC"), q[w, t], q[w, t], day_ahead[w, t])]
            if revised[t]:
                # London whole hours are UTC whole hours (whole-hour offsets).
                decision = (starts.iat[t] - pd.Timedelta(hours=2)).floor("h")
                rows.append(("intraday", decision, full[w, t], trade[w, t], intraday_price[w, t]))
            for stage, when, position, traded, paid in rows:
                decisions.append(
                    {
                        "world_id": w,
                        "slot_index": t,
                        "night_index": int(night[t]),
                        "interval_start_utc": starts.iat[t],
                        "interval_start_london": london.iat[t],
                        "decision_utc": when,
                        "decision_london": when.tz_convert(LONDON),
                        "stage": stage,
                        "hours_to_gate_closure": int(
                            np.ceil((gate - when) / pd.Timedelta(hours=1))
                        ),
                        "baseline_known_kwh": baseline[w, t],
                        "forecast_metered_kwh": metered.mean(axis=0)[t],
                        "position_kwh": position,
                        "trade_kwh": traded,
                        "price_gbp_per_mwh": paid,
                        # Dispatch off: the frozen-book re-run repeats the
                        # actual trades (dispatch contract §6.2).
                        "frozen_position_kwh": position,
                        "frozen_trade_kwh": traded,
                        "evidence_kind": _EVIDENCE,
                    }
                )
    updates = typed_frame(pd.DataFrame(decisions), POSITION_UPDATE_COLUMNS)
    for column in ("decision_utc", "interval_start_utc"):
        updates[column] = pd.to_datetime(updates[column], utc=True).dt.as_unit("ns")
    for column in ("decision_london", "interval_start_london"):
        updates[column] = updates[column].dt.tz_convert(LONDON).dt.as_unit("ns")

    shortfall = metric_matrix(world, "selected", "early_departure_shortfall_kwh")
    unmet = np.stack([shortfall[:, night == n].sum(axis=1) for n in np.unique(night)], axis=1)
    ledger = trading_ledger_from(
        deviation,
        slots,
        terms=_TRADING_TERMS,
        unmet_kwh=unmet,
        intraday_cash=trade * intraday_price / 1000.0,
        intraday_volume=np.abs(trade),
    )
    week = trading_week_from(ledger)
    # SYNTHETIC per-world capacity for the per-MW-year margin: the mean
    # unmanaged kW at 19:00 London.
    capacity = (unmanaged / 0.5)[:, half_hour == 38].mean(axis=1)
    kpis = trading_kpis_from(
        week,
        vehicle_count=len(units),
        deferrable_2h_kw=capacity,
        spread=day_ahead_spread_from(forecast_prices, slots),
        erosion=baseline_erosion_from(deviation, slots),
    )
    residual = (ledger["net_gbp"] - ledger[list(TRADING_BUCKETS)].sum(axis=1)).abs().max()
    net = {s: week.loc[week["strategy"].eq(s), "net_gbp"] for s in TRADING_STRATEGIES}
    p10 = {s: float(np.quantile(v, 0.1, method="linear")) for s, v in net.items()}
    revision = (full - q / commitment)[:, revised].mean(axis=1)
    checks = typed_frame(
        pd.DataFrame(
            {
                "check_id": list(TRADING_CHECK_IDS),
                "description": ["SYNTHETIC FIXTURE check"] * len(TRADING_CHECK_IDS),
                "value": [
                    float(residual),
                    float(
                        ledger.loc[ledger["strategy"].eq("perfect_foresight"), "imbalance_gbp"]
                        .abs()
                        .max()
                    ),
                    0.0,
                    float(revision.mean()),
                    float(p10["perfect_foresight"] >= p10["full"] >= p10["da_only"]),
                    0.0,
                    0.0,
                    # No dispatch: rebalancing is the whole bucket (dispatch §6.2).
                    0.0,
                ],
                "tolerance": [
                    1e-9 * max(1.0, ledger["net_gbp"].abs().max()),
                    0.0,
                    1e-9,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1e-9,
                ],
                "passed": True,
            }
        ),
        TRADING_CHECK_COLUMNS,
    )
    return {
        "deviation_world_slot": deviation,
        "position_updates": updates,
        "trading_ledger_world": ledger,
        "trading_week_world": week,
        "trading_ledger_summary": trading_ledger_summary_from(ledger, week),
        "trading_kpis": kpis,
        "trading_checks": checks,
        "shock_attribution_summary": shock_attribution_from(week),
        "open_position_profile": open_position_from(deviation, slots),
        "flex_cost_curve": _flex_cost_curve(unmanaged, slots),
        "shape_premium_summary": shape_premium_from(world, forecast_prices),
        **_household_frames(ledger, cost, units, worlds),
        "revenue_by_segment": _revenue_by_segment(ledger, units, worlds),
        "supplier_positions": supplier_positions_from(
            deviation, forecast_prices, slots, "synthetic"
        ),
        "control_group_summary": None,
        "price_curve_source": "synthetic",
        "user_price_curve": None,
        **_supplier_frames(deviation, ledger, week, cost, units, slots, forecast_prices),
    }


def _supplier_frames(deviation, ledger, week, cost, units, slots, forecast_prices) -> dict:
    """Supplier contract v1 frames on the SYNTHETIC toy trading frames, via the model's builder.

    The per-EV inputs are toys that match ``_household_frames``: every EV
    takes an equal share of each night's gross and of the week's customer
    value, every EV counts as earning, and each EV's battery energy is an
    equal share of the fleet's home import × the toy efficiency.
    """

    worlds, evs = int(deviation["world_id"].nunique()), len(units)
    nights = int(deviation["night_index"].nunique())
    saving = -cost.sort_values("world_id")["illustrative_selected_minus_normal_total_gbp"]
    full = ledger.loc[ledger["strategy"].eq("full")]
    share = -full.groupby("world_id")["customer_revenue_share_gbp"].sum().to_numpy()
    value = np.repeat(((saving.to_numpy() + share) / evs * 52.0 / 12.0)[:, None], evs, axis=1)
    added = {
        path: np.repeat(
            (deviation.groupby("world_id")[column].sum().to_numpy() * _CHARGE_EFFICIENCY / evs)[
                :, None
            ],
            evs,
            axis=1,
        )
        for path, column in (("normal", "unmanaged_kwh"), ("selected", "metered_kwh"))
    }
    return supplier.build_frames(
        SimpleNamespace(
            deviation_world_slot=deviation, trading_ledger_world=ledger, trading_week_world=week
        ),
        {"value_gbp_per_month": value, "earning": np.ones((worlds, evs), dtype=bool)},
        np.full((worlds, evs, nights), 1.0 / evs),
        cost_effect=cost,
        units=units,
        study_slots=slots,
        cohort_ids=COHORT_ORDER,
        battery_added=added,
        forecast_prices=forecast_prices,
        price_assumptions={
            "summer_months": (4, 5, 6, 7, 8, 9),
            "supply_reference_net_demand_gw": assumptions.PRICES[
                "supply_reference_net_demand_gw"
            ].value,
        },
        trading_assumptions=_TRADING_TERMS,
        supplier_inputs=assumptions.supplier_inputs(),
        warmup_slot_count=48 * 7,
    )


def _flex_cost_curve(unmanaged: np.ndarray, slots: pd.DataFrame) -> pd.DataFrame:
    """SYNTHETIC cost curve: 80% of unmanaged charging movable, available linearly in price."""

    charging = unmanaged / 0.5
    movable = 0.8 * charging
    share = np.clip((COST_CURVE_THRESHOLDS + 300.0) / 600.0, 0.0, 1.0)
    half_hour = slots["local_half_hour"].to_numpy()
    day_type = slots["day_type"].to_numpy()
    rows = []
    for kind in DAY_TYPE_ORDER:
        in_type = np.ones(len(day_type), bool) if kind == "all" else day_type == kind
        for h in sorted(range(48), key=lambda i: (i - 24) % 48):
            cells = in_type & (half_hour == h)
            mean_charging = charging[:, cells].mean(axis=1)
            mean_movable = movable[:, cells].mean(axis=1)
            metrics = [
                *(
                    ("available_kw", p, mean_movable * s)
                    for p, s in zip(COST_CURVE_THRESHOLDS, share, strict=True)
                ),
                ("movable_kw", np.nan, mean_movable),
                ("charging_kw", np.nan, mean_charging),
            ]
            for metric, threshold, per_world in metrics:
                rows.append(
                    {
                        "day_type": kind,
                        "local_half_hour": f"{h // 2:02d}:{30 * (h % 2):02d}",
                        "profile_order": (h - 24) % 48,
                        "metric": metric,
                        "threshold_gbp_per_mwh": threshold,
                        "unit": "kW",
                        **world_stats(per_world),
                        "risk_charge_gbp_per_mwh": 24.0,
                        "evidence_kind": _EVIDENCE,
                    }
                )
    return typed_frame(pd.DataFrame(rows), FLEX_COST_CURVE_COLUMNS)


def _household_frames(ledger, cost, units, worlds) -> dict[str, pd.DataFrame]:
    """SYNTHETIC household value: each world's saving and customer share split equally."""

    evs = len(units)
    saving = -cost.sort_values("world_id")[
        "illustrative_selected_minus_normal_total_gbp"
    ].to_numpy()
    full = ledger.loc[ledger["strategy"].eq("full")]
    share = -full.groupby("world_id")["customer_revenue_share_gbp"].sum().to_numpy()
    monthly = 52.0 / 12.0
    per_ev_saving = np.repeat((saving / evs)[:, None], evs, axis=1) * monthly
    per_ev_share = np.repeat((share / evs)[:, None], evs, axis=1) * monthly
    value = per_ev_saving + per_ev_share
    edges = np.concatenate([[-np.inf], HOUSEHOLD_BIN_EDGES, [np.inf]])
    groups = {"fleet": np.ones(evs, bool)}
    for cohort_id in COHORT_ORDER:
        members = units["cohort_id"].eq(cohort_id).to_numpy()
        if members.any():
            groups[cohort_id] = members
    distribution, summary = [], []
    unit = "GBP per household per month"
    bins = np.searchsorted(HOUSEHOLD_BIN_EDGES, value, side="right")
    for group_id, members in groups.items():
        for b in range(len(edges) - 1):
            distribution.append(
                {
                    "group_id": group_id,
                    "bin_index": b,
                    "bin_lower_gbp": edges[b],
                    "bin_upper_gbp": edges[b + 1],
                    **world_stats((bins[:, members] == b).mean(axis=1)),
                }
            )
        v = value[:, members]
        stats = {
            "mean_value": (v.mean(axis=1), unit),
            "p10_across_evs": (np.quantile(v, 0.1, axis=1), unit),
            "p50_across_evs": (np.quantile(v, 0.5, axis=1), unit),
            "p90_across_evs": (np.quantile(v, 0.9, axis=1), unit),
            "share_worse_off": ((v < 0.0).mean(axis=1), "fraction"),
            "mean_customer_saving": (per_ev_saving[:, members].mean(axis=1), unit),
            "mean_revenue_share": (per_ev_share[:, members].mean(axis=1), unit),
        }
        for statistic, (per_world, statistic_unit) in stats.items():
            summary.append(
                {
                    "group_id": group_id,
                    "statistic": statistic,
                    "unit": statistic_unit,
                    "ev_count": int(members.sum()),
                    **world_stats(per_world),
                    "evidence_kind": _EVIDENCE,
                }
            )
    return {
        "household_value_distribution": typed_frame(
            pd.DataFrame(distribution), HOUSEHOLD_DISTRIBUTION_COLUMNS
        ),
        "household_value_summary": typed_frame(pd.DataFrame(summary), HOUSEHOLD_SUMMARY_COLUMNS),
    }


def _revenue_by_segment(ledger, units, worlds) -> pd.DataFrame:
    """SYNTHETIC allocation: every EV takes an equal share of each full-strategy bucket."""

    full = ledger.loc[ledger["strategy"].eq("full")]
    evs = len(units)
    segments = [("all", "all", np.ones(evs, bool))]
    for cohort_id in COHORT_ORDER:
        members = units["cohort_id"].eq(cohort_id).to_numpy()
        if members.any():
            segments.append(("cohort", cohort_id, members))
    rows = []
    for bucket in TRADING_MONEY:
        weekly = full.groupby("world_id")[bucket].sum().to_numpy()
        for segment_type, segment_id, members in segments:
            count = int(members.sum())
            total = weekly * count / evs
            for metric, unit, values in (
                ("gbp_per_week", "GBP per week", total),
                ("gbp_per_ev_per_week", "GBP per EV per week", total / count),
            ):
                rows.append(
                    {
                        "bucket": bucket,
                        "segment_type": segment_type,
                        "segment_id": segment_id,
                        "metric": metric,
                        "unit": unit,
                        "ev_count": count,
                        **world_stats(values),
                        "evidence_kind": _EVIDENCE,
                    }
                )
    return typed_frame(pd.DataFrame(rows), REVENUE_SEGMENT_COLUMNS)


# --------------------------------------------------------------------------
# One-EV replay and run comparison (mirrors of the M5 functions)
# --------------------------------------------------------------------------


def replay_one_ev(
    result: FixtureForecastResult, unit_id: str, world_id: int | None = None
) -> FixtureOneEvReplay:
    """Replay one EV in one world (contract section 5), from ``replay_state``."""

    sim: _Simulated = result.replay_state
    unit_ids = list(result.units["unit_id"])
    if unit_id not in unit_ids:
        raise KeyError(f"unknown unit_id: {unit_id}")
    world = result.representative_world_id if world_id is None else world_id
    if not 0 <= world < result.world_count:
        raise KeyError(f"unknown world_id: {world_id}")
    e = unit_ids.index(unit_id)
    unit = result.units.iloc[e]
    slots = result.study_slots
    frames = []
    for path in sim.connected:
        connected = sim.connected[path][world, :, e]
        driving = sim.driving[path][world, :, e]
        away = sim.away[path][world, :, e]
        public = sim.public_charging[path][world, :, e]
        location = np.select(
            [connected, driving, public, away],
            ["home_plugged", "driving", "public_charging", "away"],
            "home_unplugged",
        )
        closing = sim.closing_kwh[path][world, :, e]
        frame = slots.loc[:, list(SLOT_COLUMNS)].copy()
        frame.insert(0, "path_id", path)
        frame["location"] = location.astype(object)
        frame["connected"] = connected
        frame["battery_soc_percent"] = 100.0 * closing / unit["physical_capacity_kwh"]
        frame["closing_battery_kwh"] = closing
        frame["home_import_kwh"] = sim.home_import_kwh[path][world, :, e]
        frame["public_import_kwh"] = sim.public_import_kwh[path][world, :, e]
        frame["unserved_travel_kwh"] = sim.unserved_kwh[path][world, :, e]
        frame["early_departure_shortfall_kwh"] = sim.early_shortfall_kwh[path][world, :, e]
        frames.append(frame)
    events = sim.events
    plug_events = events.loc[events["world_id"].eq(world) & events["unit_id"].eq(unit_id)]
    audit = sim.audit.loc[sim.audit["world_id"].eq(world) & sim.audit["unit_id"].eq(unit_id)]
    return FixtureOneEvReplay(
        unit_id=unit_id,
        world_id=int(world),
        is_representative_world=world == result.representative_world_id,
        traits=unit.to_dict(),
        intervals=pd.concat(frames, ignore_index=True).loc[:, list(REPLAY_INTERVAL_COLUMNS)],
        plug_events=plug_events.reset_index(drop=True),
        daily_audit=_daily_audit(audit),
    )


def _daily_audit(audit: pd.DataFrame) -> pd.DataFrame:
    frame = audit.reset_index(drop=True).copy()
    frame["day_label"] = [_day_label(value) for value in frame["local_date"]]
    frame["departure_utc"] = pd.to_datetime(frame["departure_utc"], utc=True)
    frame["departure_london"] = frame["departure_utc"].dt.tz_convert(LONDON)
    frame["departed_below_target"] = (
        frame["departure_soc_percent"] < frame["target_soc_percent"] - 1e-6
    ).fillna(False)
    return typed_frame(frame, DAILY_AUDIT_COLUMNS)


def replay_one_ev_bands(result: FixtureForecastResult, unit_id: str) -> pd.DataFrame:
    """This EV across worlds: P10/P50/P90 of SoC and plugged-in (contract section 5)."""

    sim: _Simulated = result.replay_state
    unit_ids = list(result.units["unit_id"])
    if unit_id not in unit_ids:
        raise KeyError(f"unknown unit_id: {unit_id}")
    e = unit_ids.index(unit_id)
    capacity = result.units["physical_capacity_kwh"].iat[e]
    frames = []
    for metric, unit in (("battery_soc_percent", "percent"), ("connected_share", "fraction")):
        for path in sim.connected:
            values = (
                100.0 * sim.closing_kwh[path][:, :, e] / capacity
                if metric == "battery_soc_percent"
                else sim.connected[path][:, :, e].astype(float)
            )
            q = np.quantile(values, (0.1, 0.5, 0.9), axis=0, method="linear")
            frame = result.study_slots.loc[:, list(SLOT_COLUMNS)].copy()
            frame.insert(0, "path_id", path)
            frame.insert(0, "unit", unit)
            frame.insert(0, "metric", metric)
            frame["world_count"] = np.int64(values.shape[0])
            frame["p10"], frame["p50"], frame["p90"] = q[0], q[1], q[2]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True).loc[:, list(REPLAY_BAND_COLUMNS)]


def compare_runs(a, b) -> FixtureRunComparison:
    """B minus A on the headline path of each run, per world (contract section 6)."""

    for name in ("horizon_start_utc", "study_slot_count", "world_count"):
        if getattr(a, name) != getattr(b, name):
            raise ValueError(f"cannot pair worlds: {name} differs")
    reasons = []
    if a.seed != b.seed:
        reasons.append(f"Not matched: A uses seed {a.seed}, B uses seed {b.seed}.")
    if a.vehicle_count != b.vehicle_count:
        reasons.append(f"Not matched: A has {a.vehicle_count} EVs, B has {b.vehicle_count}.")
    path_a = "selected" if a.model == "action" else "normal"
    path_b = "selected" if b.model == "action" else "normal"
    world_a, world_b = a.fleet_world_intervals, b.fleet_world_intervals
    keys = slot_keys(world_a)
    band_frames = []
    for metric, unit in FLEET_METRICS.items():
        delta = metric_matrix(world_b, path_b, metric) - metric_matrix(world_a, path_a, metric)
        frame = keys.copy()
        frame.insert(0, "unit", "percentage points" if metric == "battery_soc_percent" else unit)
        frame.insert(0, "metric", metric)
        frame["world_count"] = np.int64(delta.shape[0])
        for stat, column in quantile_stats(delta).items():
            frame[stat] = column
        band_frames.append(frame)

    totals_available = a.model == "action" and b.model == "action"
    weekly = {
        "home_import_kwh": ("kWh per week", _weekly(world_a, path_a), _weekly(world_b, path_b)),
        "unserved_travel_kwh": (
            "kWh per week",
            _weekly(world_a, path_a, "unserved_travel_kwh"),
            _weekly(world_b, path_b, "unserved_travel_kwh"),
        ),
        "median_plug_in_soc_percent": (
            "percentage points",
            _world_median_soc(a),
            _world_median_soc(b),
        ),
    }
    if totals_available:
        column = "illustrative_selected_minus_normal_total_gbp"
        weekly["illustrative_total_gbp"] = (
            "GBP",
            a.cost_effect[column].to_numpy(),
            b.cost_effect[column].to_numpy(),
        )
    weekly_rows, kpi_rows = [], []
    for metric, (unit, values_a, values_b) in weekly.items():
        delta = values_b - values_a
        for w in range(len(delta)):
            weekly_rows.append(
                {
                    "world_id": w,
                    "metric": metric,
                    "unit": unit,
                    "value_a": values_a[w],
                    "value_b": values_b[w],
                    "b_minus_a": delta[w],
                }
            )
        q = np.quantile(delta, (0.1, 0.5, 0.9))
        kpi_rows.append({"metric": metric, "unit": unit, "p10": q[0], "p50": q[1], "p90": q[2]})
    return FixtureRunComparison(
        matched_futures=not reasons,
        mismatch_reasons=tuple(reasons),
        path_id_a=path_a,
        path_id_b=path_b,
        changed=_changed(a, b),
        illustrative_total_available=totals_available,
        difference_bands=pd.concat(band_frames, ignore_index=True).loc[
            :, list(DIFFERENCE_BAND_COLUMNS)
        ],
        weekly_differences=typed_frame(pd.DataFrame(weekly_rows), COMPARE_WEEKLY_COLUMNS),
        kpis=typed_frame(pd.DataFrame(kpi_rows), COMPARE_KPI_COLUMNS),
    )


def _weekly(world, path, metric="home_import_kwh") -> np.ndarray:
    return metric_matrix(world, path, metric).sum(axis=1)


def _world_median_soc(result) -> np.ndarray:
    # Normal-path plug-ins, every world, from the world KPI table (contract 3.10).
    kpis = result.plug_in_world_kpis
    rows = kpis.loc[
        kpis["group_id"].eq("fleet")
        & kpis["day_type"].eq("all")
        & kpis["metric"].eq("median_plug_in_soc_percent")
    ].sort_values("world_id")
    return rows["value"].to_numpy(dtype=float)


def _changed(a, b) -> pd.DataFrame:
    rows = []
    records_a = {record.name: record for record in a.assumptions}
    records_b = {record.name: record for record in b.assumptions}
    for name in dict.fromkeys([*records_a, *records_b]):
        record_a, record_b = records_a.get(name), records_b.get(name)
        value_a = record_a.value if record_a else None
        value_b = record_b.value if record_b else None
        both_unset = all(isinstance(v, float) and np.isnan(v) for v in (value_a, value_b))
        if value_a != value_b and not both_unset:  # unset (NaN) records match, as the model
            record = record_b or record_a
            rows.append((name, record.label, record.unit, value_a, value_b))
    for key in SETTINGS_KEYS:
        value_a, value_b = a.settings_snapshot.get(key), b.settings_snapshot.get(key)
        if value_a != value_b:
            rows.append((key, key, "", value_a, value_b))
    return pd.DataFrame(rows, columns=list(CHANGED_COLUMNS), dtype=object)


# --------------------------------------------------------------------------
# Validators (run unchanged on real results by M5)
# --------------------------------------------------------------------------


# Re-exported so existing imports from this module keep working.
__all__ = [
    "ACTION_ID",
    "AVERAGE_DAY_COLUMNS",
    "Assumption",
    "BAND_STAT_COLUMNS",
    "CHANGED_COLUMNS",
    "CHEAPEST_HALF_HOUR_COLUMNS",
    "CNZ_RECORD_NAMES",
    "COHORTS",
    "COHORT_BAND_COLUMNS",
    "COHORT_ORDER",
    "COHORT_SUMMARY_COLUMNS",
    "COMPARE_KPI_COLUMNS",
    "COMPARE_WEEKLY_COLUMNS",
    "COST_COMPONENTS",
    "COST_EFFECT_COLUMNS",
    "COST_SUMMARY_COLUMNS",
    "DAILY_AUDIT_COLUMNS",
    "DAY_TYPE_ORDER",
    "DEFERRABLE_BUCKET_ORDER",
    "DEFERRABLE_POWER_BAND_COLUMNS",
    "DIFFERENCE_BAND_COLUMNS",
    "DIFFERENCE_WEEKLY_BAND_COLUMNS",
    "DIFFERENCE_WEEKLY_COLUMNS",
    "EVALUATION_PRICE_COLUMNS",
    "EVIDENCE_LABEL",
    "FLEET_BAND_COLUMNS",
    "FLEET_METRICS",
    "FLEET_WORLD_COLUMNS",
    "FLEXIBILITY_BAND_COLUMNS",
    "FLEXIBILITY_ROWS",
    "FLEXIBILITY_WEEKLY_COLUMNS",
    "FORECAST_PRICE_COLUMNS",
    "FixtureActionSummary",
    "FixtureForecastResult",
    "FixtureOneEvReplay",
    "FixturePlugInSummary",
    "FixtureRunComparison",
    "HALF_HOUR",
    "HOUR_BAND_COLUMNS",
    "LOCATION_ORDER",
    "LONDON",
    "MAX_PLUG_IN_EVENT_WORLDS",
    "NOT_RECOVERED_TOLERANCE_KWH",
    "PATH_ORDER",
    "PLUG_EVENT_HEATMAP_COLUMNS",
    "SESSION_DISTRIBUTION_COLUMNS",
    "SESSION_DISTRIBUTION_METRICS",
    "SESSION_SOURCE_RECORD_NAMES",
    "session_distribution_bands_from",
    "PLUG_IN_EVENT_COLUMNS",
    "PLUG_IN_HEATMAP_COLUMNS",
    "PLUG_IN_WORLD_KPI_COLUMNS",
    "PLUG_KPI_COLUMNS",
    "PLUG_KPI_METRICS",
    "PRICE_RELATIVE_COLUMNS",
    "REPLAY_BAND_COLUMNS",
    "REPLAY_INTERVAL_COLUMNS",
    "PRICE_BAND_SHIFT_COLUMNS",
    "PRICE_BANDS",
    "SETTINGS_KEYS",
    "SLACK_BUCKETS",
    "SMART_CHARGING_METRICS",
    "SMART_CHARGING_SUMMARY_COLUMNS",
    "SMART_CHARGING_WORLD_COLUMNS",
    "SHARE_METRICS",
    "SLOT_COLUMNS",
    "SLOT_COUNT",
    "SOC_BAND_COLUMNS",
    "STUDY_DAYS",
    "STUDY_SLOT_COLUMNS",
    "UNIT_COLUMNS",
    "WEEKDAY_LABELS",
    "WEEKLY_BAND_COLUMNS",
    "WEEKLY_BAND_METRICS",
    "WEEKLY_DIFFERENCE_METRICS",
    "WEEKLY_PEAK_COLUMNS",
    "cheapest_half_hour_from",
    "cohort_plug_in_stats",
    "compare_runs",
    "cost_summary_from_cost_effect",
    "difference_bands_from_world",
    "difference_weekly_from_world",
    "empty_frame",
    "fleet_bands",
    "make_result",
    "metric_matrix",
    "plug_event_heatmap_from",
    "plug_in_kpis_from_world_kpis",
    "plug_in_world_kpis_from_events",
    "price_band_shift_from",
    "price_relative_bands_from",
    "quantile_stats",
    "replay_one_ev",
    "replay_one_ev_bands",
    "slot_keys",
    "smart_charging_summary_from",
    "smart_charging_world_from",
    "typed_frame",
    "validate_one_ev_replay_v2",
    "validate_result_v2",
    "validate_run_comparison_v2",
    "weekly_bands_from_world",
    "weekly_difference_bands",
    "weekly_peak_summary_from",
]
