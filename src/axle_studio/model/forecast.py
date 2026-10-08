"""The one forecast runner and its result type (result contract v2).

What this owns: ``run_forecast`` runs either model (``"action"``, the
default, or ``"no_action"``) from one ``np.random.default_rng(seed)`` and
returns one ``ForecastResult`` (``docs/contracts/results-v2.md``).  Both
models share every setup step here: input checks, the persistent EV
population, the per-cohort trip parameters, the sampled worlds and the
fleet aggregation, including public top-ups (decision 0004 item 32), which
both models take from the same public-charge assumptions.  The action model
adds synthetic market prices per world (day-ahead, intraday and imbalance,
from a net-demand model with shocks, decision 0004 items 48, 53 and 56), a
smart-charging selected path (item 38; with intraday dispatch on, item
62(b), the dispatched path, beside a day-ahead plan reference path) and the
illustrative cost effect.

How it fits: ``sampling`` draws the inputs, ``physics`` runs the half-hour
kernel and sums per world, ``action`` supplies the smart charger's planner
and values the outcome, and this module orders the calls and names the
outputs for the dashboard.
Summaries that need more than a rename (average-day bands, plug-in events,
difference bands and the action and cost summaries) belong to
``summaries``; ``run_forecast`` calls its ``build_summaries`` once, after
packaging, and this module re-exports its result types.
``run_forecast_from_assumptions`` is the app's entry point: it takes every
input from ``assumptions``.

Random order (why it matters): draws happen in a fixed order from the one
generator, so a seed reproduces a run exactly (trading contract v1 §10.0).
The population draws its cohorts, mileage multipliers, zones, the
control-group permutation and, last, the charger manufacturers.  The worlds
draw departures, trips, the shared plug-in week and day factors, the plug
and plug-in-time uniforms, then weather.  Both models draw the population
and then the evaluation worlds; the action model draws the market prices
next, then the non-response uniforms and, last, the manufacturer outage
uniforms, so a price assumption never moves a world.  Scripted events draw
nothing, so a run with and without an event shares every random future.
So the two models see identical random futures for the same seed, and
Compare pairs them world by world.  Smart charging, intraday dispatch and
public top-ups draw nothing (decision 0004 items 32, 38 and 62(b)), so the
dispatch switch, the commitment share and the re-plan threshold move no
draw.  The planning world of the earlier 18:00
action was removed with item 38; it was drawn before the evaluation worlds,
so removing it re-baselined the action model's worlds for a given seed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from axle_studio.model import assumptions as model_assumptions
from axle_studio.model import availability, availability_backtest, market, replay
from axle_studio.model import events as model_events
from axle_studio.model.action import (
    ILLUSTRATIVE_PUBLIC_CHARGE_GBP_PER_KWH,
    NOT_RECOVERED_MATERIAL_SHARE_PERCENT,
    IntradayDispatch,
    SmartCharging,
    calculate_wholesale_world_cost_effect,
    intraday_dispatch_inputs,
    locked_evs,
    smart_charging_inputs,
    timed_start_allowed,
)
from axle_studio.model.assumptions import (
    MANUFACTURER_IDS,
    MANUFACTURER_LABELS,
    ZONE_IDS,
    Assumption,
)
from axle_studio.model.clock import holiday_flags, utc_half_hour_boundaries
from axle_studio.model.physics import (
    ZONE_SUM_COLUMNS,
    DispatchSums,
    PublicTopUp,
    ZoneSums,
    effective_driving_efficiency,
    public_top_up,
    simulate_fleet_intervals,
)
from axle_studio.model.sampling import (
    MARKET_PRICE_INPUTS,
    MarketPrices,
    assign_manufacturers,
    build_population,
    generate_market_price_paths,
    plug_in_skip_share,
    sample_connection_opportunities,
    sample_daily_temperature,
    sample_daily_trip_inputs,
    sample_departure_times,
    sample_manufacturer_outages,
    sample_non_response,
    sample_plug_in_factors,
)
from axle_studio.model.settings import CohortFixture, RunSettings
from axle_studio.model.summaries import (
    PLUG_IN_EVENT_WORLD_LIMIT,
    ActionSummary,
    PlugInSummary,
    ReplayState,
    RunComparison,
    RunSummary,
    SampledWorlds,
    build_study_slots,
    build_summaries,
    build_warmup_slots,
    compare_runs,
    flexibility_inputs,
    slim_run,
)

Model = Literal["action", "no_action"]

# --- Names and category orders (contract section 1, rule 9) -----------------

MODEL_LABELS = {"action": "Explicit wholesale action", "no_action": "Explicit no-action"}
POLICY_IDS = {"action": "explicit_synthetic_wholesale_v1", "no_action": "explicit_no_action_v1"}
EVIDENCE_KINDS = {"action": "illustrative_synthetic", "no_action": "illustrative"}
PATH_ORDER = ("normal", "selected", "timed")
"""Category order (contract rule 9). "timed" (decision 0007) is optional: it is present
only in ``fleet_world_intervals``, ``cohort_world_intervals``, ``fleet_interval_bands``
and ``cohort_interval_bands``, and only when ``timed_start_local_hour`` is set. Every
other, pairwise consumer (savings, trading, supplier, firm MW, zones, household, replay,
run comparison) stays the explicit two-path ``("normal", "selected")``, never this tuple."""
DAY_TYPE_ORDER = ("weekday", "weekend", "all")
LOCATION_ORDER = ("home_plugged", "home_unplugged", "driving", "away", "public_charging")
COHORT_ORDER = (
    "average_uk",
    "intelligent_octopus",
    "infrequent_charging",
    "infrequent_driving",
    "scheduled_charging",
    "always_plugged_in",
)
FLEET_METRICS = {
    "connected_count": "EVs",
    "connected_share": "fraction",
    "home_import_kwh": "kWh per half-hour",
    "home_import_kw": "kW",
    "public_import_kwh": "kWh per half-hour",
    "total_import_kw": "kW",
    "closing_battery_kwh": "kWh",
    "battery_soc_percent": "percent",
    "driving_share": "fraction",
    "away_share": "fraction",
    "public_charging_share": "fraction",
    "unserved_travel_kwh": "kWh per half-hour",
}
"""Fleet metric name -> unit, in display order (contract 3.5)."""
WEEKLY_METRICS = ("home_import_kwh", "public_import_kwh", "unserved_travel_kwh")
"""Metrics of ``weekly_bands``: weekly totals in kWh per week."""

_SLOT_HOURS = 0.5
_SLOT_KEYS = ["slot_index", "interval_start_utc", "interval_end_utc", "interval_start_london"]
_BAND_STATS = ["world_count", "mean", "p10", "p50", "p90"]
# Physical column names the kernel uses, renamed to the contract's names.
_KERNEL_RENAMES = {
    "home_grid_import_kwh": "home_import_kwh",
    "public_grid_import_kwh": "public_import_kwh",
    "realised_grid_kw": "total_import_kw",
    "closing_soc_percent": "battery_soc_percent",
}
# The kernel's ``*_fraction`` location columns are sums of per-EV time
# fractions (EV-equivalents), not shares.  The contract divides each by
# ``unit_count`` and names it ``*_share`` (contract 1 rule 5).
_FRACTION_TO_SHARE = {
    "driving_fraction": "driving_share",
    "away_on_trip_fraction": "away_share",
    "public_charging_fraction": "public_charging_share",
    "home_unplugged_fraction": "home_unplugged_share",
    "home_connected_fraction": "home_connected_share",
    "parked_away_fraction": "parked_away_share",
}
_PUBLIC_ASSUMPTION_NAMES = frozenset(
    {"efficiency_fraction", "top_up_threshold_soc_fraction", "top_up_target_soc_fraction"}
)
_PRICE_ASSUMPTION_NAMES = frozenset(MARKET_PRICE_INPUTS)
_COHORT_BEHAVIOUR_FIELDS = (
    "weekday_drive_probability",
    "weekend_drive_probability",
    "destination_dwell_minutes",
    "drive_speed_mph",
    "departure_scale_minutes",
    "plug_in_scale_minutes",
)


# --- Result types --------------------------------------------------------


@dataclass(frozen=True)
class ForecastResult:
    """One explicit forecast, in the shape of result contract v2.

    The fields after ``replay_state`` come from the summaries builder in
    ``model/summaries.py``; ``run_forecast`` always fills them (the ``None``
    defaults exist only for the packaged result before the builder runs).
    ``assumptions`` holds the ``model/assumptions.py`` records with the
    values the run used; ``run_forecast_from_assumptions`` fills it, and a
    direct ``run_forecast`` call with hand-made inputs leaves it ``None``
    because no record describes those inputs.  Action-only fields are
    ``None`` on a no-action result.
    """

    model: Model
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
    assumptions: tuple[Assumption, ...] | None
    action_status: str
    public_charging_status: str
    not_recovered_world_count: int | None
    representative_world_id: int
    study_slots: pd.DataFrame
    units: pd.DataFrame
    fleet_world_intervals: pd.DataFrame
    cohort_world_intervals: pd.DataFrame
    fleet_interval_bands: pd.DataFrame
    cohort_interval_bands: pd.DataFrame
    weekly_bands: pd.DataFrame
    forecast_prices: pd.DataFrame | None
    evaluation_prices: pd.DataFrame | None
    cost_effect: pd.DataFrame | None
    price_shocks: pd.DataFrame | None
    # The optional timed path's own cost effect (decision 0007, model step 2),
    # same schema as ``cost_effect``, "timed" in the selected slot. ``None``
    # when ``timed_start_local_hour`` is unset; never added to ``cost_effect``
    # itself.
    timed_cost_effect: pd.DataFrame | None
    # Weeks materially not recovered on the timed path (decision 0004 item 45,
    # extended by decision 0007 model step 2), built exactly like
    # ``not_recovered_world_count`` but summed from ``timed_cost_effect``.
    # ``None`` whenever that frame is (no action model, or the setting unset).
    timed_not_recovered_world_count: int | None
    replay_state: ReplayState = field(repr=False)
    # Filled by model/summaries.build_summaries in run_forecast (task M5).
    sampled_world_ids: tuple[int, ...] | None = None
    cohort_summary: pd.DataFrame | None = None
    average_day_bands: pd.DataFrame | None = None
    fleet_interval_ev_bands: pd.DataFrame | None = None
    flexibility_bands: pd.DataFrame | None = None
    plug_in_events: pd.DataFrame | None = None
    plug_in_world_kpis: pd.DataFrame | None = None
    plug_in_summary: PlugInSummary | None = None
    plug_in_heatmap: pd.DataFrame | None = None
    difference_bands: pd.DataFrame | None = None
    difference_weekly: pd.DataFrame | None = None
    difference_weekly_bands: pd.DataFrame | None = None
    action_summary: ActionSummary | None = None
    cost_effect_summary: pd.DataFrame | None = None
    not_recovered_summary: pd.DataFrame | None = None
    # The optional timed path's own cost summaries (decision 0007, model step
    # 2): same schema as ``cost_effect_summary``/``not_recovered_summary``,
    # built from ``timed_cost_effect``.  None when ``timed_start_local_hour``
    # is unset.
    timed_cost_effect_summary: pd.DataFrame | None = None
    timed_not_recovered_summary: pd.DataFrame | None = None
    smart_charging_world: pd.DataFrame | None = None
    smart_charging_summary: pd.DataFrame | None = None
    price_band_shift: pd.DataFrame | None = None
    forecast_price_bands: pd.DataFrame | None = None
    weekly_peak_summary: pd.DataFrame | None = None
    # Decision 0004 item 54 (plan C2, C4, D-1, D-3, E1): contract 3.6d, 3.6e,
    # 3.10b, 4.7a and 4.7b.
    deferrable_power_bands: pd.DataFrame | None = None
    flexibility_weekly_summary: pd.DataFrame | None = None
    plug_in_half_hour_heatmap: pd.DataFrame | None = None
    plug_out_heatmap: pd.DataFrame | None = None
    price_relative_bands: pd.DataFrame | None = None
    cheapest_half_hour_summary: pd.DataFrame | None = None
    # Decision 0004 item 54 (plan D-2): contract 3.10c.
    session_distribution_bands: pd.DataFrame | None = None
    # The illustrative trading overlay (trading contract v1 §5 and §9;
    # decisions 0004 items 57-58 and 0005), action results only.
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
    # Household contract v1 §3.1-§3.2: per-(world, EV) weekly outcomes and
    # their per-group summary (action results with the trading overlay).
    household_ev_world: pd.DataFrame | None = None
    household_outcomes_summary: pd.DataFrame | None = None
    revenue_by_segment: pd.DataFrame | None = None
    supplier_positions: pd.DataFrame | None = None
    control_group_summary: pd.DataFrame | None = None
    price_curve_source: str | None = None
    user_price_curve: pd.DataFrame | None = None
    # Supplier P&L and Partners (supplier contract v1, decision 0006), action
    # results with the trading overlay; never added to the trading ledger.
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
    # Events and zones (decision 0004 items 55, 56 and 58; trading contract v1
    # §3, §5).  ``events`` is the validated events table (empty when none;
    # ``None`` on a no-action result, which has no events); ``zone_ids`` and
    # ``zone_world_intervals`` exist on every result, because zones are part
    # of the population.
    events: pd.DataFrame | None = None
    zone_ids: tuple[str, ...] | None = None
    zone_world_intervals: pd.DataFrame | None = None
    event_response_bands: pd.DataFrame | None = None
    zone_import_bands: pd.DataFrame | None = None
    zone_summary: pd.DataFrame | None = None
    market_shocks: pd.DataFrame | None = None
    shock_summary: pd.DataFrame | None = None
    # Firm-MW availability (trading contract v1 §10.0; decision 0004 item
    # 59): ``run_forecast`` fills them on an action result from the
    # availability (J2), backtest (J3) and product (J5) builders; ``None`` on
    # a no-action result except ``blackout_windows``, which every result
    # carries.
    blackout_windows: pd.DataFrame | None = None
    world_nights: pd.DataFrame | None = None
    availability_world_slot: pd.DataFrame | None = None
    availability_bands: pd.DataFrame | None = None
    availability_manufacturer_world: pd.DataFrame | None = None
    firm_share_by_fleet_size: pd.DataFrame | None = None
    availability_backtest: pd.DataFrame | None = None
    availability_reliability: pd.DataFrame | None = None
    availability_backtest_summary: pd.DataFrame | None = None
    product_sheet: pd.DataFrame | None = None
    availability_value_summary: pd.DataFrame | None = None
    settlement_file: pd.DataFrame | None = None
    settlement_file_name: str | None = None
    charge_completion_summary: pd.DataFrame | None = None
    firmness_by_manufacturer: pd.DataFrame | None = None
    manufacturer_summary: pd.DataFrame | None = None
    # The fleet replay of the sampled worlds (replay contract v1 §1,
    # decision 0004 item 66); None on a no-action result.
    replay_week: replay.ReplayWeek | None = None
    # Intraday dispatch (intraday-dispatch-v1 §7, decision 0004 item 62(b)):
    # the day-ahead plan against the dispatched fleet per (world, slot), its
    # world-first bands and the per-cohort locked/free split.  None unless
    # an action run has ``trading.intraday_dispatch`` on; ``units`` carries
    # ``dispatch_locked`` on every result.
    dispatch_world_slot: pd.DataFrame | None = None
    dispatch_bands: pd.DataFrame | None = None
    dispatch_split: pd.DataFrame | None = None


@dataclass(frozen=True)
class SimulatedForecast:
    """Raw outputs of one run, with the kernel's column names.

    ``run_forecast`` packages this into a ``ForecastResult``; tests and the
    walkthrough notebook read it directly when they need the sampled worlds
    or the kernel's own column names.  Action-only fields are None for the
    no-action model.
    """

    model: Model
    settings: RunSettings
    units: pd.DataFrame
    interval_start_utc: pd.DatetimeIndex
    evaluation: SampledWorlds
    public_top_up: PublicTopUp
    fleet_world_intervals: pd.DataFrame
    cohort_world_intervals: pd.DataFrame
    fleet_interval_bands: pd.DataFrame
    cohort_interval_distribution: pd.DataFrame
    forecast_prices: pd.DataFrame | None = None
    evaluation_prices: pd.DataFrame | None = None
    # Everything the price generator returned, including the shock table,
    # the known and surprise shock profiles and every world's hour-by-hour
    # intraday path, for the trading overlay (trading contract §1.4).
    market_prices: MarketPrices | None = None
    smart_charging: SmartCharging | None = None
    departure_margin_hours: float | None = None
    # NaN (the default) means off: no "timed" path ran, and the raw interval
    # frames above carry only normal and selected (decision 0007).
    timed_start_local_hour: float = float("nan")
    # Per world, without ``sessions_affected_count``: build_summaries adds it
    # from the per-EV pass, so the packaged result's ``cost_effect`` has it.
    cost_effect: pd.DataFrame | None = None
    # The optional timed path's own cost effect (decision 0007, model step 2):
    # the same calculation, "timed" in the selected slot.  None with the
    # setting off or on a no-action run (cost effect is an action-only
    # concept, same as ``cost_effect`` itself).
    timed_cost_effect: pd.DataFrame | None = None
    # Warm-up history for the trading baseline (decision 0004 item 52,
    # trading contract v1 sections 1.4 and 4.1): the warm-up slots with their
    # London keys (``summaries.build_warmup_slots``) and the normal path's
    # fleet home import per (world, warm-up slot) in kWh.  Simulated, never
    # shown.
    warmup_slots: pd.DataFrame | None = None
    warmup_home_import_kwh: np.ndarray | None = None
    # The trading overlay's kernel outputs by path (``physics._record_trading``;
    # trading contract v1 §1.4), the hourly decision slots of the plan books
    # (study slot indices), the trading assumptions, each EV's expected
    # non-response rate before its session starts (the trader's rho,
    # §10.1e) and the user price curve (None when not supplied).
    trading_kernel: dict[str, dict[str, np.ndarray]] | None = None
    book_slots: np.ndarray | None = None
    trading_assumptions: dict[str, float | str] | None = None
    non_response_probability: np.ndarray | None = None
    user_price_curve: pd.DataFrame | None = None
    # Zones and events (trading contract v1 §3, §2).  ``zone_sums`` holds the
    # kernel's per-zone sums (``physics.ZoneSums``), including the normal
    # path's warm-up import per (world, zone, warm-up slot) for the zone
    # baselines; ``zone_shares`` and ``zone_headroom_kw`` (kW, NaN = none) are
    # the run's zone inputs in ``ZONE_IDS`` order.  ``events`` is the
    # validated events table (action model only).
    zone_sums: ZoneSums | None = None
    zone_shares: tuple[float, ...] | None = None
    zone_headroom_kw: tuple[float, ...] | None = None
    events: pd.DataFrame | None = None
    # Shared factors and manufacturers (trading contract v1 §10.1), the
    # inputs of ``world_nights`` (§10.1f) and the availability lanes.
    # ``plug_in_factors`` (both models): ``week`` y (world,), ``day`` z
    # (world, sampled date), ``skip_share`` q (world, sampled date), and the
    # per-date ``holiday`` flags and ``skip_logit_shift`` h (sampled date,).
    # ``shared_factor_assumptions`` are the §10.1 values the run used.
    # Action model only: ``manufacturer_outage`` (world, study night, maker)
    # bool, and ``session_non_response_probability`` (world, study night,
    # EV) float64, each session's chance of ignoring its plan from the
    # control group, its maker's outage and its maker's response rate
    # (§10.1e), which the kernel reads as ``SmartCharging.non_response``.
    # ``blackout_windows`` (both models) is the validated §10.5b table.
    plug_in_factors: dict[str, np.ndarray] | None = None
    shared_factor_assumptions: dict[str, float] | None = None
    manufacturer_outage: np.ndarray | None = None
    session_non_response_probability: np.ndarray | None = None
    blackout_windows: pd.DataFrame | None = None
    # Intraday dispatch (intraday-dispatch-v1 §5): the dispatcher's inputs
    # and the kernel's ``DispatchSums`` (the day-ahead plan path, the
    # locked/free split, re-plan counts and both plan books).  None unless
    # an action run has ``trading.intraday_dispatch`` on.
    intraday_dispatch: IntradayDispatch | None = None
    dispatch_sums: DispatchSums | None = None


# --- Running ---------------------------------------------------------------


def run_forecast_from_assumptions(
    start_local_date: date,
    *,
    model: Model = "action",
    values: Mapping[str, object] | None = None,
    user_price_curve: pd.DataFrame | None = None,
    event_presets: Sequence[str] = (),
    blackout_windows: pd.DataFrame | None = None,
) -> ForecastResult:
    """Run one full forecast with every input taken from ``model/assumptions.py``.

    ``user_price_curve`` is the supplier's optional 48-value curve, already
    checked by ``market.validate_user_price_curve`` (trading contract §9.4);
    it is a run input beside the edited values, not an editable number.
    ``blackout_windows`` is the ``availability.blackout_windows`` table
    (§10.5b; ``None``: the record's default, no blackout), a run input for
    the same reason.

    ``values`` are edited assumption values by record name (the Edit
    assumptions dialog's draft); names left out keep their defaults, and an
    unknown name or out-of-bounds value raises ``ValueError`` before any
    sampling.  ``start_local_date`` is the London date the seven-day study
    starts.  ``event_presets`` are ids of ``assumptions.EVENT_PRESETS`` whose
    rows make up the run's events table (the events selection the dashboard
    drives; empty: no events).  The result carries
    ``result_assumptions(values)``, so every record it shows is a value this
    run actually used.
    """

    resolved = model_assumptions.resolve_values(values)
    settings = model_assumptions.run_settings(resolved, start_local_date)
    events = model_events.event_table(
        event_presets, build_study_slots(start_local_date, settings.study_days)
    )
    result = run_forecast(
        settings,
        model_assumptions.cohort_fixture(resolved),
        **model_assumptions.forecast_inputs(
            resolved, warmup_days=settings.warmup_days, study_days=settings.study_days
        ),
        events=events,
        model=model,
        user_price_curve=user_price_curve,
        blackout_windows=blackout_windows,
        supplier_inputs=model_assumptions.supplier_inputs(resolved),
    )
    return replace(result, assumptions=model_assumptions.result_assumptions(resolved))


def run_forecast(
    settings: RunSettings,
    fixture: CohortFixture,
    *,
    daily_miles_cv_by_cohort: Mapping[str, float],
    personal_mileage_cv_by_cohort: Mapping[str, float],
    trip_behaviour: Mapping[str, object],
    base_temperature_c_by_day: np.ndarray,
    weather_sd_c: float,
    public_charge_assumptions: Mapping[str, float],
    price_assumptions: Mapping[str, float] | None = None,
    public_charge_gbp_per_kwh: float = ILLUSTRATIVE_PUBLIC_CHARGE_GBP_PER_KWH,
    departure_margin_hours: float | None = None,
    timed_start_local_hour: float = float("nan"),
    not_recovered_material_share_percent: float = NOT_RECOVERED_MATERIAL_SHARE_PERCENT,
    zone_shares: Sequence[float] | None = None,
    zone_headroom_kw: Sequence[float] | None = None,
    events: pd.DataFrame | None = None,
    model: Model = "action",
    trading_assumptions: Mapping[str, float | str] | None = None,
    user_price_curve: pd.DataFrame | None = None,
    shared_factor_assumptions: Mapping[str, float] | None = None,
    blackout_windows: pd.DataFrame | None = None,
    supplier_inputs: Mapping[str, float] | None = None,
) -> ForecastResult:
    """Run one full explicit Monte Carlo forecast and return a ``ForecastResult``.

    Inputs are the run settings, the six-cohort fixture and the illustrative
    inputs ``assumptions.forecast_inputs`` returns: per-cohort daily and
    personal mileage coefficients of variation, the trip-behaviour mapping,
    the base temperature per day (deg C, warm-up days first), its world
    standard deviation and the public-charge assumptions, including the
    top-up threshold and target as SoC fractions (both models top up in
    public, decision 0004 item 32).  The action model also needs the
    synthetic price assumptions and ``departure_margin_hours`` (h, the smart
    charger's safety margin before the typical departure, decision 0004 item
    38); ``public_charge_gbp_per_kwh`` (GBP/kWh, illustrative, decision 0004
    item 4) only values the outcome, and
    ``not_recovered_material_share_percent`` (percent of weekly normal home
    import, item 45) only sets which weeks count as materially not
    recovered.  Both models use the margin for the
    flexibility metrics' expected departure (item 43); a no-action run given
    none has ``flexibility_bands=None``.

    The action model also runs the illustrative trading overlay (trading
    contract v1): ``trading_assumptions`` are ``assumptions.trading_inputs``
    values (default: the records' values, so hand-built callers need not
    pass them), plus the text switch ``trading.commitment_rule``
    (``fixed_share`` or ``newsvendor``, §10.4; absent: ``fixed_share``);
    ``user_price_curve`` optionally replaces the day-ahead shape (§9.4).
    With ``trading.intraday_dispatch`` on (the default, decision 0004 item
    62(b)) the EVs the commitment share does not lock re-plan hourly on the
    latest intraday price, so the selected path is the dispatched fleet;
    the result then carries ``dispatch_world_slot``, ``dispatch_bands`` and
    ``dispatch_split`` (intraday-dispatch-v1 §7), and ``units`` carries
    ``dispatch_locked`` on every result.

    Zones and events (decision 0004 items 55 and 58; trading contract v1 §2,
    §3): ``zone_shares`` are the fractions of EVs in each zone of
    ``assumptions.ZONE_IDS`` (``None``: the defaults) and
    ``zone_headroom_kw`` each zone's optional reported headroom (kW, NaN =
    none; ``None``: none anywhere).  ``events`` is an events table
    (``events.EVENT_COLUMNS``; ``None``: no events); the action model
    applies it, the no-action model has no events.

    Shared factors and manufacturers (trading contract v1 §10.1):
    ``shared_factor_assumptions`` are ``assumptions.shared_factor_inputs``
    values by record name (``None``: the records' values): the plug-in skip
    share's median (fraction) and its day and week logit SDs, the two
    holiday switches (0/1) and each manufacturer's share, response rate and
    per-night outage probability.  Non-response comes from those makers
    (§10.1e): the per-archetype rates of §9.5 are retired.

    ``blackout_windows`` (§10.5b) is a table of daily London windows
    (``start_local_time``, ``duration_minutes``; ``None``: none), checked by
    ``availability.validate_blackout_windows``.  Plans leave those
    half-hours untouched and trading is closed there; every result carries
    the validated table and ``study_slots.blackout``.

    ``timed_start_local_hour`` (decision 0007): NaN (the default) is off and
    changes nothing.  Set to a London clock hour (0-23.5, half-hour steps)
    and a third "timed" path runs -- the normal rule, except home charging is
    barred from London 12:00 until that hour each day (hard rule, no
    fallback) -- on the same sampled inputs as normal and selected (no new
    random draw).  It is reported only in ``fleet_world_intervals``,
    ``cohort_world_intervals``, ``fleet_interval_bands`` and
    ``cohort_interval_bands``; every pairwise frame (savings, trading,
    zones, the per-EV summaries, replay and run comparison) stays normal
    versus selected.  Both models accept it.
    """

    simulated = simulate_forecast(
        settings,
        fixture,
        daily_miles_cv_by_cohort=daily_miles_cv_by_cohort,
        personal_mileage_cv_by_cohort=personal_mileage_cv_by_cohort,
        trip_behaviour=trip_behaviour,
        base_temperature_c_by_day=base_temperature_c_by_day,
        weather_sd_c=weather_sd_c,
        public_charge_assumptions=public_charge_assumptions,
        price_assumptions=price_assumptions,
        public_charge_gbp_per_kwh=public_charge_gbp_per_kwh,
        departure_margin_hours=departure_margin_hours,
        timed_start_local_hour=timed_start_local_hour,
        not_recovered_material_share_percent=not_recovered_material_share_percent,
        zone_shares=zone_shares,
        zone_headroom_kw=zone_headroom_kw,
        events=events,
        model=model,
        trading_assumptions=trading_assumptions,
        user_price_curve=user_price_curve,
        shared_factor_assumptions=shared_factor_assumptions,
        blackout_windows=blackout_windows,
    )
    result = _package(simulated, fixture)
    # The trading overlay reads the finished kernel outputs (trading contract
    # §1.5 step 10).  Its sampled worlds are the ones the per-EV summaries
    # keep: the representative world first, then the lowest other ids.
    trading = None
    if model == "action":
        others = [w for w in range(result.world_count) if w != result.representative_world_id]
        sampled = [result.representative_world_id, *others][:PLUG_IN_EVENT_WORLD_LIMIT]
        trading = _run_trading(simulated, result, sampled)
    # Every price, cost and trading view says when the supplier's own curve
    # set the day-ahead shape (§9.4).
    price_curve_source = (
        None
        if model != "action"
        else ("user curve" if simulated.user_price_curve is not None else "synthetic")
    )
    # Summaries that need per-EV detail or pairing (task M5) are built from
    # the packaged frames plus a chunked kernel rerun from ``replay_state``.
    summaries = build_summaries(
        result.replay_state,
        result.fleet_world_intervals,
        model=model,
        study_slots=result.study_slots,
        representative_world_id=result.representative_world_id,
        cohort_catalogue=[
            (spec.cohort_id, spec.source_name, spec.population_share_fraction)
            for spec in fixture.cohorts
        ],
        cost_effect=result.cost_effect,
        timed_cost_effect=result.timed_cost_effect,
        forecast_prices=result.forecast_prices,
        evaluation_prices=result.evaluation_prices,
        departure_margin_hours=simulated.departure_margin_hours,
        trading=trading,
        trading_assumptions=simulated.trading_assumptions,
        price_curve_source=price_curve_source,
        public_charge_gbp_per_kwh=public_charge_gbp_per_kwh,
        zone_world_intervals=result.zone_world_intervals,
        zone_shares=simulated.zone_shares,
        zone_headroom_kw=simulated.zone_headroom_kw,
        events=result.events,
        market_prices=simulated.market_prices,
        price_assumptions=price_assumptions,
        warmup_slot_count=48 * settings.warmup_days,
        # Firm MW (trading contract v1 §10.0 step 10): the availability hook
        # in the per-EV chunk loop (J2) and the product frames (J5).
        availability_inputs=result.replay_state.availability_inputs,
        manufacturer_catalogue=[
            (
                maker_id,
                MANUFACTURER_LABELS[maker_id],
                simulated.shared_factor_assumptions[f"manufacturers.share.{maker_id}"],
            )
            for maker_id in MANUFACTURER_IDS
        ],
        # SUPPLIER and CARBON values, unset terms NaN; None takes the records.
        supplier_inputs=supplier_inputs or model_assumptions.supplier_inputs(),
        # Intraday dispatch frames (intraday-dispatch-v1 §7); None with the
        # switch off or on a no-action run leaves them None.
        dispatch_sums=simulated.dispatch_sums,
    )
    result = replace(result, **summaries)
    world_slot = summaries.get("availability_world_slot")
    if world_slot is not None:
        # The leave-one-week-out calibration backtest (§10.3, J3) reads only
        # the world-slot frame, so it is built after the summaries.
        backtest = availability_backtest.availability_backtest(world_slot)
        reliability = availability_backtest.availability_reliability(backtest)
        result = replace(
            result,
            availability_backtest=backtest,
            availability_reliability=reliability,
            availability_backtest_summary=availability_backtest.availability_backtest_summary(
                reliability
            ),
        )
    if model == "action":
        result = replace(
            result,
            deviation_world_slot=trading.deviation_world_slot,
            position_updates=trading.position_updates,
            trading_ledger_world=trading.trading_ledger_world,
            trading_week_world=trading.trading_week_world,
            price_curve_source=price_curve_source,
            user_price_curve=simulated.user_price_curve,
        )
        result = replace(
            result,
            replay_week=replay.build_replay_week(
                study_slots=result.study_slots,
                world_ids=result.sampled_world_ids,
                market_prices=simulated.market_prices,
                warmup_slot_count=len(simulated.warmup_slots),
                fleet_world_intervals=result.fleet_world_intervals,
                forecast_prices=result.forecast_prices,
                public_charge_gbp_per_kwh=public_charge_gbp_per_kwh,
                gate_closure_minutes=simulated.trading_assumptions["gate_closure_minutes"],
                surprise_reveal_hours=_assumption_snapshot(
                    price_assumptions, _PRICE_ASSUMPTION_NAMES, "price_assumptions"
                )["surprise_reveal_hours"],
                trading=trading,
                market_shocks=result.market_shocks,
                events=result.events,
            ),
        )
    return result


def simulate_forecast(
    settings: RunSettings,
    fixture: CohortFixture,
    *,
    daily_miles_cv_by_cohort: Mapping[str, float],
    personal_mileage_cv_by_cohort: Mapping[str, float],
    trip_behaviour: Mapping[str, object],
    base_temperature_c_by_day: np.ndarray,
    weather_sd_c: float,
    public_charge_assumptions: Mapping[str, float],
    price_assumptions: Mapping[str, float] | None = None,
    public_charge_gbp_per_kwh: float = ILLUSTRATIVE_PUBLIC_CHARGE_GBP_PER_KWH,
    departure_margin_hours: float | None = None,
    timed_start_local_hour: float = float("nan"),
    not_recovered_material_share_percent: float = NOT_RECOVERED_MATERIAL_SHARE_PERCENT,
    zone_shares: Sequence[float] | None = None,
    zone_headroom_kw: Sequence[float] | None = None,
    events: pd.DataFrame | None = None,
    model: Model = "action",
    trading_assumptions: Mapping[str, float | str] | None = None,
    user_price_curve: pd.DataFrame | None = None,
    shared_factor_assumptions: Mapping[str, float] | None = None,
    blackout_windows: pd.DataFrame | None = None,
) -> SimulatedForecast:
    """Sample and aggregate one run (both paths for the action model); see ``run_forecast``.

    Returns the raw ``SimulatedForecast`` with the kernel's column names.
    Its price frames cover the warm-up and the study (``_package`` keeps the
    study slots); its warm-up fields carry the unmanaged warm-up import.
    ``timed_start_local_hour`` (NaN: off) adds the optional "timed" path to
    both models (decision 0007); see ``run_forecast``.
    """

    # External inputs are checked once here; everything below is trusted.
    if model not in MODEL_LABELS:
        raise ValueError(f"model must be 'action' or 'no_action', not {model!r}")
    if not isinstance(settings, RunSettings):
        raise TypeError("settings must be a RunSettings instance")
    if not isinstance(fixture, CohortFixture):
        raise TypeError("fixture must be a CohortFixture instance")
    days = settings.sampled_day_count
    if not isinstance(base_temperature_c_by_day, np.ndarray):
        raise TypeError("base_temperature_c_by_day must be a NumPy array")
    if base_temperature_c_by_day.shape != (days,):
        raise ValueError(f"base_temperature_c_by_day must have shape ({days},)")
    behaviour_by_cohort, clip_minutes, t_df = _behaviour_boundary(trip_behaviour)
    factor_values = _shared_factor_boundary(shared_factor_assumptions)
    # Both models top up in public with the same assumptions (lead
    # decision, 28 September 2026): a no-action run with a fallback
    # efficiency would value the same physical top-up differently.
    top_up = public_top_up(
        **_assumption_snapshot(
            public_charge_assumptions, _PUBLIC_ASSUMPTION_NAMES, "public_charge_assumptions"
        )
    )
    if departure_margin_hours is not None:
        margin = float(departure_margin_hours)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("departure_margin_hours must be a finite non-negative number")
    # The optional "timed" path (decision 0007): NaN is off, in both models,
    # and the mask is built once here rather than inside the kernel because
    # it is a policy/time rule (``action`` owns it), not kernel mechanics.
    timed_hour = float(timed_start_local_hour)
    timed_mask = None if math.isnan(timed_hour) else timed_start_allowed(settings, timed_hour)
    if model == "action":
        if departure_margin_hours is None:
            raise ValueError("the action model needs departure_margin_hours")
        price_snapshot = _assumption_snapshot(
            price_assumptions, _PRICE_ASSUMPTION_NAMES, "price_assumptions"
        )
        public_rate = validate_public_charge_rate(public_charge_gbp_per_kwh)
        material_share = float(not_recovered_material_share_percent)
        if not math.isfinite(material_share) or not 0.0 <= material_share <= 100.0:
            raise ValueError("not_recovered_material_share_percent must be between 0 and 100")
        trading_values = _trading_boundary(trading_assumptions)
    # Zones and events are external inputs too (trading contract v1 §1.5
    # step 1).  Shares are checked by the population builder.
    headroom = (
        (float("nan"),) * len(ZONE_IDS)
        if zone_headroom_kw is None
        else tuple(float(value) for value in zone_headroom_kw)
    )
    if len(headroom) != len(ZONE_IDS) or any(
        not (math.isnan(value) or (math.isfinite(value) and value >= 0.0)) for value in headroom
    ):
        raise ValueError("zone_headroom_kw must hold one non-negative kW value or NaN per zone")
    if zone_shares is None:
        zone_shares = [model_assumptions.ZONES[f"share.{zone_id}"].value for zone_id in ZONE_IDS]
    zone_shares = tuple(float(share) for share in zone_shares)
    # Blackout windows (§10.0 step 1, §10.5b) and the holiday dates (§10.1c,
    # no draw) mark the slot frames; a request wholly inside a blackout is
    # rejected with the events.
    blackout_table = availability.validate_blackout_windows(
        pd.DataFrame(columns=list(availability.BLACKOUT_COLUMNS))
        if blackout_windows is None
        else blackout_windows
    )
    flags = holiday_flags(
        settings,
        bank_holiday_monday=bool(factor_values["behaviour.holiday_bank_holiday_monday"]),
        half_term_week=bool(factor_values["behaviour.holiday_half_term_week"]),
    )
    slot_flags = _slot_flags(settings, flags["holiday"], blackout_table)
    study_slots = build_study_slots(settings.start_local_date, settings.study_days, **slot_flags)
    valid_events = model_events.validate_events(
        model_events.empty_events() if events is None else events,
        study_slots,
        ZONE_IDS,
        blackout=study_slots["blackout"].to_numpy(),
    )

    rng = np.random.default_rng(settings.seed)
    units = build_population(
        settings,
        fixture,
        rng,
        daily_miles_cv_by_cohort=daily_miles_cv_by_cohort,
        personal_mileage_cv_by_cohort=personal_mileage_cv_by_cohort,
        zone_shares=zone_shares,
    )
    zone_sums = ZoneSums(
        zone_index=units["zone_id"].map(ZONE_IDS.index).to_numpy(dtype=np.int64),
        zone_count=len(ZONE_IDS),
    )
    # Hold-out control group (trading contract §9.5, §4.8): the last
    # population draw, one permutation of the EVs, always taken whether the
    # group is on or off, so switching it moves no other draw.
    control_order = rng.permutation(settings.vehicle_count)
    # Charger manufacturers (§10.1d): the population's last draw, in both
    # models, so the maker mix never moves a world draw.  ``manufacturer_id``
    # is the reported label; ``manufacturer_index`` (0-3) indexes the
    # per-maker response rates and outages.
    maker_index = assign_manufacturers(
        rng,
        settings.vehicle_count,
        [factor_values[f"manufacturers.share.{maker_id}"] for maker_id in MANUFACTURER_IDS],
    )
    units = units.assign(
        manufacturer_id=np.asarray(MANUFACTURER_IDS, dtype=object)[maker_index],
        manufacturer_index=maker_index,
    )
    # Intraday dispatch split (intraday-dispatch-v1 §2, §5.3 step 2): the
    # locked EVs follow from the commitment share c, the switch and the
    # model, with no draw, so editing any of them moves no random channel.
    # Every EV is unlocked on a no-action run and with the switch off, so
    # ``units.dispatch_locked`` exists on every result (§2, lead Q9).
    dispatch_on = model == "action" and bool(trading_values["trading.intraday_dispatch"])
    units = units.assign(
        dispatch_locked=locked_evs(units, trading_values["trading.day_ahead_commitment_share"])
        if dispatch_on
        else np.zeros(settings.vehicle_count, dtype=bool)
    )
    # Holidays (flags above) change the skip share only (§10.1c), never a
    # clock or a draw.
    trip_parameters, clock_scales = _trip_parameters(
        settings, units, behaviour_by_cohort, daily_miles_cv_by_cohort
    )

    def sample(world_settings: RunSettings) -> tuple[SampledWorlds, dict[str, np.ndarray]]:
        # One set of worlds: home departures, trips, the shared plug-in
        # factors, home connections and weather, in that fixed draw order.
        # The one departure draw is passed to both the trips and the
        # sessions, because plug-out is departure (decision 0004 item 51).
        departures = sample_departure_times(
            world_settings,
            units,
            rng,
            scale_minutes_by_cohort=clock_scales["departure"],
            t_df=t_df,
            clip_minutes=clip_minutes,
        )
        trips = sample_daily_trip_inputs(
            world_settings, units, rng, departure_utc=departures, **trip_parameters
        )
        # The shared plug-in factors (§10.1b) sit just before the plug
        # uniforms they act on, so acceptance is decided in one place, the
        # connections sampler; the alternative (draw them last and re-decide
        # acceptance here) would split one decision across two modules.
        week_factor, day_factor = sample_plug_in_factors(
            rng, world_settings.evaluation_world_count, world_settings.sampled_day_count
        )
        skip_share = plug_in_skip_share(
            week_factor,
            day_factor,
            flags["skip_logit_shift"],
            median=factor_values["behaviour.plug_in_skip_median"],
            day_sd=factor_values["behaviour.plug_in_day_factor_sd"],
            week_sd=factor_values["behaviour.plug_in_week_factor_sd"],
        )
        accepted, start, end = sample_connection_opportunities(
            world_settings,
            units,
            rng,
            departure_utc=departures,
            drives_today=trips["drives_today"],
            plug_in_scale_minutes_by_cohort=clock_scales["plug_in"],
            weekend_plug_in_scale_minutes_by_cohort=clock_scales["weekend_plug_in"],
            t_df=t_df,
            clip_minutes=clip_minutes,
            skip_share=skip_share,
        )
        inputs = {
            **trips,
            "connection_session_accepted": accepted,
            "connection_start_utc": start,
            "connection_end_utc": end,
        }
        temperature = sample_daily_temperature(
            rng,
            world_count=world_settings.evaluation_world_count,
            base_temperature_c_by_day=base_temperature_c_by_day,
            sd_c=weather_sd_c,
        )
        efficiency = effective_driving_efficiency(
            units["efficiency_miles_per_battery_kwh"].to_numpy(dtype=float, copy=True),
            temperature,
            settings.weather_efficiency_sensitivity_fraction_per_c,
        )
        factors = {
            "week": week_factor,
            "day": day_factor,
            "skip_share": skip_share,
            "holiday": flags["holiday"],
            "skip_logit_shift": flags["skip_logit_shift"],
        }
        return SampledWorlds(inputs, temperature, efficiency), factors

    # --- Horizon slicing (decision 0004 items 1 and 52) ---
    # Fixed UTC slots with the study starting at London noon of the start
    # date, so all seven session nights sit inside it; a London offset change
    # is allowed.  ``run_start_utc`` is every simulated slot (warm-up, then
    # study): prices are generated over all of them so the charger has
    # published history and the trading baseline has warm-up prices.
    boundaries = pd.DatetimeIndex(
        utc_half_hour_boundaries(
            settings.start_local_date, settings.warmup_days, settings.study_days
        )
    )
    run_start_utc = boundaries[:-1]
    interval_start_utc = run_start_utc[settings.warmup_days * 48 :]
    warmup_slots = build_warmup_slots(settings.start_local_date, settings.warmup_days, **slot_flags)
    warmup_home_import = np.zeros((settings.evaluation_world_count, settings.warmup_days * 48))

    # Both models draw the evaluation worlds straight after the population,
    # so the same seed gives both models the same random futures.
    evaluation, plug_in_factors = sample(settings)
    if model == "no_action":
        # No shared fleet-site cap: each EV is limited only by its own home
        # charging power (decision 0004 item 34).
        fleet_world, cohort_world, fleet_bands, cohort_bands = simulate_fleet_intervals(
            settings,
            units,
            evaluation.inputs,
            evaluation_effective_efficiency_miles_per_battery_kwh=(
                evaluation.effective_efficiency_miles_per_battery_kwh
            ),
            public_top_up=top_up,
            warmup_home_import_kwh=warmup_home_import,
            zone_sums=zone_sums,
            timed_start_allowed=timed_mask,
        )
        return SimulatedForecast(
            model=model,
            settings=settings,
            units=units,
            interval_start_utc=interval_start_utc,
            evaluation=evaluation,
            public_top_up=top_up,
            fleet_world_intervals=fleet_world,
            cohort_world_intervals=cohort_world,
            fleet_interval_bands=fleet_bands,
            cohort_interval_distribution=cohort_bands,
            departure_margin_hours=None if departure_margin_hours is None else margin,
            timed_start_local_hour=timed_hour,
            warmup_slots=warmup_slots,
            warmup_home_import_kwh=warmup_home_import,
            zone_sums=zone_sums,
            zone_shares=zone_shares,
            zone_headroom_kw=headroom,
            plug_in_factors=plug_in_factors,
            shared_factor_assumptions=factor_values,
            blackout_windows=blackout_table,
        )

    # Action model.  The market prices are drawn after the worlds, so no
    # world draw moves when a price assumption changes.  Their heating term
    # reads each priced date's sampled temperature, so a cold world is also a
    # dear one (decision 0004 item 53).  They cover the warm-up as well as
    # the study (item 52), so the charger has published history and the
    # trading baseline has warm-up prices; the temperature array starts at
    # the first warm-up date, as the generator's column 0 must.  The smart
    # charger sees only its own world's day-ahead prices as published when it
    # plans (13:00 London the day before delivery, plan B4) and each EV's
    # typical departure; home import is valued at the same day-ahead prices
    # (plan B3).  The intraday and imbalance prices and the shock table are
    # carried for the trading ledger and value nothing yet.
    #
    # Scripted events (trading contract v1 §1.5 steps 4-7): the price-shock
    # presets are the same in every world and draw nothing, so the price
    # draws are identical with and without them; known shocks join the
    # net demand the day-ahead auction clears on, surprises only intraday
    # (from their reveal) and imbalance.  Positions below are run slots
    # (warm-up then study), the kernel's own indexing.
    run_slots = pd.concat([warmup_slots, study_slots], ignore_index=True)
    run_slot_count = len(run_start_utc)
    known_gw, surprise_gw = model_events.scripted_shock_profiles(
        valid_events, run_slots, run_slot_count
    )
    market_prices = generate_market_price_paths(
        rng,
        interval_start_utc=run_start_utc,
        evaluation_world_count=settings.evaluation_world_count,
        daily_temperature_c=evaluation.daily_temperature_c,
        price_assumptions=price_snapshot,
        scripted_known_shock_gw=known_gw,
        scripted_surprise_shock_gw=surprise_gw,
        scripted_surprise_reveal_hours=model_events.scripted_surprise_reveal_hours(
            valid_events,
            run_slots,
            run_slot_count,
            gate_closure_minutes=price_snapshot["gate_closure_minutes"],
        ),
    )
    # Non-response uniforms (trading contract v1 §4.8 step 4): drawn after
    # every price channel and always, so an outage or a non-response edit
    # moves no other draw.
    non_response_uniform = sample_non_response(
        rng, settings.evaluation_world_count, settings.study_days, settings.vehicle_count
    )
    # Manufacturer outages (§10.1d, §10.0 step 6): the last draw of the run,
    # always one (world, study night, maker) block, so an outage-probability
    # edit only switches outages on or off and moves nothing else.
    outage_probability = np.array(
        [
            factor_values[f"manufacturers.outage_probability_per_night.{maker_id}"]
            for maker_id in MANUFACTURER_IDS
        ]
    )
    manufacturer_outage = (
        sample_manufacturer_outages(
            rng, settings.evaluation_world_count, settings.study_days, len(MANUFACTURER_IDS)
        )
        < outage_probability
    )
    notice_slot, adjustment = model_events.planner_adjustments(
        valid_events, run_slots, len(ZONE_IDS), run_slot_count
    )
    if user_price_curve is not None:
        # The supplier's curve replaces the typical daily shape after
        # generation, so it draws nothing and every channel stays paired
        # (trading contract §9.4).
        market_prices = market.apply_user_price_curve(
            market_prices,
            user_price_curve,
            run_start_utc,
            floor_gbp_per_mwh=float(price_snapshot["price_floor_gbp_per_mwh"]),
            cap_gbp_per_mwh=float(price_snapshot["price_cap_gbp_per_mwh"]),
        )
    forecast_prices, evaluation_prices = market_prices.day_ahead, market_prices.realised
    # Hold-out control group (§9.5): the permutation was drawn with the
    # population; the switch only picks EVs from it.  A control EV never
    # follows a plan (probability 1).
    control = (
        market.control_group_mask(
            units,
            control_order,
            trading_values["trading.control_group_share"],
            [spec.cohort_id for spec in fixture.cohorts],
        )
        if trading_values["trading.control_group"]
        else np.zeros(settings.vehicle_count, dtype=bool)
    )
    units = units.assign(control_group=control)
    # Non-response comes from the makers (§10.1e, decision 0004 item 59):
    # each session's chance rho of ignoring its plan, compared in the kernel
    # with the session's one uniform when each plan is made.
    response_rate = np.array(
        [factor_values[f"manufacturers.response_rate.{maker_id}"] for maker_id in MANUFACTURER_IDS]
    )
    session_non_response = _session_non_response_probability(
        control, maker_index, manufacturer_outage, response_rate
    )
    # The trader's expectation for a session not yet started (§4.4 as
    # changed by §10.1e): it ignores its plan by either route, base
    # non-response or its maker's outage, 1 - r (1 - pi); a control EV
    # always.  Once plugged in, the book shows what it does.
    expected_non_response = np.where(
        control,
        1.0,
        1.0 - response_rate[maker_index] * (1.0 - outage_probability[maker_index]),
    )
    warmup_count = 48 * settings.warmup_days
    smart = smart_charging_inputs(
        settings,
        units,
        # The frame is world-major, so this is (world, run slot).
        forecast_prices["wholesale_forecast_gbp_per_mwh"]
        .to_numpy(dtype=float)
        .reshape(settings.evaluation_world_count, run_slot_count),
        departure_margin_hours=departure_margin_hours,
        zone_index=zone_sums.zone_index,
        event_notice_slot=notice_slot,
        event_adjustment_gbp_per_mwh=adjustment,
        outage_probability=model_events.outage_probability(
            valid_events, run_slots, len(ZONE_IDS), run_slot_count
        ),
        non_response_uniform=non_response_uniform,
        non_response=session_non_response,
        base_non_response=1.0 - response_rate[maker_index],
        control_group=control,
        blackout=run_slots["blackout"].to_numpy(dtype=bool),
        # Each night's intraday availability decision s_n (§10.2d), where
        # the kernel copies the plans in force (R1), as run-slot positions.
        decision_slot_by_night=warmup_count
        + availability.decision_slots(study_slots, _decision_local_time()),
    )
    trading_kernel, book_slots = _trading_kernel_outputs(
        settings,
        study_slots,
        control,
        trading_values["day_ahead_publication_local_hour"],
    )
    # Intraday dispatch (intraday-dispatch-v1 §5.3 steps 7-8): free EVs
    # re-plan hourly on the latest intraday price, so the selected path
    # becomes the dispatched path and the kernel adds the day-ahead plan
    # path as a third pass into ``dispatch_sums``.  It reads the price paths
    # already drawn (after any user curve) and draws nothing.  Off: no third
    # pass, and the selected path is the day-ahead plan path exactly as
    # before.
    dispatch = dispatch_sums = None
    if dispatch_on:
        dispatch = intraday_dispatch_inputs(
            settings,
            units,
            market_prices.intraday_path_gbp_per_mwh,
            replan_threshold_gbp_per_mwh=trading_values["trading.replan_threshold_gbp_per_mwh"],
            half_spread_gbp_per_mwh=trading_values["trading.half_spread_gbp_per_mwh"],
            gate_closure_minutes=trading_values["gate_closure_minutes"],
        )
        dispatch_sums = DispatchSums()
    fleet_world, cohort_world, fleet_bands, cohort_bands = simulate_fleet_intervals(
        settings,
        units,
        evaluation.inputs,
        evaluation_effective_efficiency_miles_per_battery_kwh=(
            evaluation.effective_efficiency_miles_per_battery_kwh
        ),
        public_top_up=top_up,
        smart_charging=smart,
        warmup_home_import_kwh=warmup_home_import,
        trading_output=trading_kernel,
        zone_sums=zone_sums,
        intraday_dispatch=dispatch,
        dispatch_sums=dispatch_sums,
        timed_start_allowed=timed_mask,
    )
    cost_effect = calculate_wholesale_world_cost_effect(
        fleet_world,
        forecast_prices,
        public_charge_gbp_per_kwh=public_rate,
        material_share_percent=material_share,
    )
    # The optional timed path's own cost effect (decision 0007, model step 2):
    # the same function, "timed" in the selected slot, so the two savings are
    # valued identically (same prices, same public rate, same shortfall rule).
    timed_cost_effect = (
        None
        if timed_mask is None
        else calculate_wholesale_world_cost_effect(
            fleet_world,
            forecast_prices,
            public_charge_gbp_per_kwh=public_rate,
            material_share_percent=material_share,
            selected_path_id="timed",
        )
    )
    return SimulatedForecast(
        model=model,
        settings=settings,
        units=units,
        interval_start_utc=interval_start_utc,
        evaluation=evaluation,
        public_top_up=top_up,
        fleet_world_intervals=fleet_world,
        cohort_world_intervals=cohort_world,
        fleet_interval_bands=fleet_bands,
        cohort_interval_distribution=cohort_bands,
        forecast_prices=forecast_prices,
        evaluation_prices=evaluation_prices,
        market_prices=market_prices,
        smart_charging=smart,
        departure_margin_hours=float(departure_margin_hours),
        timed_start_local_hour=timed_hour,
        cost_effect=cost_effect,
        timed_cost_effect=timed_cost_effect,
        warmup_slots=warmup_slots,
        warmup_home_import_kwh=warmup_home_import,
        trading_kernel=trading_kernel,
        book_slots=book_slots,
        trading_assumptions=trading_values,
        non_response_probability=expected_non_response,
        user_price_curve=user_price_curve,
        zone_sums=zone_sums,
        zone_shares=zone_shares,
        zone_headroom_kw=headroom,
        events=valid_events,
        plug_in_factors=plug_in_factors,
        shared_factor_assumptions=factor_values,
        manufacturer_outage=manufacturer_outage,
        session_non_response_probability=session_non_response,
        blackout_windows=blackout_table,
        intraday_dispatch=dispatch,
        dispatch_sums=dispatch_sums,
    )


def validate_public_charge_rate(value: float) -> float:
    """Return the illustrative public rate (GBP/kWh) after checking it is usable."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("public_charge_gbp_per_kwh must be a real number")
    if not math.isfinite(value) or value < 0:
        raise ValueError("public_charge_gbp_per_kwh must be finite and non-negative")
    return float(value)


# --- Input checks and trip parameters --------------------------------------


def _behaviour_boundary(
    trip_behaviour: Mapping[str, object],
) -> tuple[Mapping[str, object], object, object]:
    if not isinstance(trip_behaviour, Mapping):
        raise TypeError("trip_behaviour must be a mapping")
    required = {"by_cohort", "clock_clip_minutes", "clock_t_df"}
    missing = sorted(required - set(trip_behaviour))
    if missing:
        raise ValueError(f"trip_behaviour missing required field: {missing[0]}")
    return (
        trip_behaviour["by_cohort"],
        trip_behaviour["clock_clip_minutes"],
        trip_behaviour["clock_t_df"],
    )


def _assumption_snapshot(
    assumptions: Mapping[str, float] | None, expected_names: frozenset[str], name: str
) -> dict[str, float]:
    # An exact name set: a missing value must not fall back to an invented
    # default, and an unknown name is probably a typo.
    if not isinstance(assumptions, Mapping):
        raise TypeError(f"{name} must be a mapping")
    missing = sorted(expected_names - set(assumptions))
    if missing:
        raise ValueError(f"{name} missing required field: {missing[0]}")
    extra = sorted(set(assumptions) - expected_names)
    if extra:
        raise ValueError(f"{name} contains unknown field: {extra[0]}")
    return {key: assumptions[key] for key in sorted(expected_names)}


def _trip_parameters(
    settings: RunSettings,
    units: pd.DataFrame,
    behaviour_by_cohort: Mapping[str, object],
    daily_miles_cv_by_cohort: Mapping[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    """Per-EV trip parameters and per-cohort clock scales from each cohort's behaviour.

    Returns the keyword arrays for ``sample_daily_trip_inputs`` (drive
    probability is (day, EV) over ``settings.sampled_day_count`` London
    dates, picking the weekday or weekend value by each date's weekday,
    warm-up days included) and the truncated-t clock
    scales in minutes per cohort: ``"departure"``, ``"plug_in"`` (weekday)
    and ``"weekend_plug_in"`` (decision 0004 items 42 and 51).  The departure
    clock hour itself is the cohort's home departure hour, read from the
    population by ``sample_departure_times``.
    """

    cohort_ids = units["cohort_id"].to_numpy(dtype=object, copy=True)
    used_cohort_ids = set(cohort_ids)
    missing_cohorts = sorted(used_cohort_ids - set(behaviour_by_cohort))
    if missing_cohorts:
        raise ValueError(f"trip behaviour missing used cohort: {missing_cohorts[0]}")
    for cohort_id in used_cohort_ids:
        missing_fields = sorted(set(_COHORT_BEHAVIOUR_FIELDS) - set(behaviour_by_cohort[cohort_id]))
        if missing_fields:
            raise ValueError(
                f"trip behaviour for {cohort_id} missing required field: {missing_fields[0]}"
            )

    days = settings.sampled_day_count
    first_date = settings.start_local_date - timedelta(days=settings.warmup_days)
    weekday = np.array([(first_date + timedelta(days=d)).weekday() < 5 for d in range(days)])

    def per_unit(name: str) -> np.ndarray:
        return np.array([behaviour_by_cohort[c][name] for c in cohort_ids], dtype=float)

    def by_day_type(weekday_name: str, weekend_name: str) -> np.ndarray:
        return np.where(
            weekday[:, np.newaxis],
            per_unit(weekday_name)[np.newaxis, :],
            per_unit(weekend_name)[np.newaxis, :],
        )

    trip_parameters = {
        "drive_probability": by_day_type("weekday_drive_probability", "weekend_drive_probability"),
        "expected_daily_miles": np.broadcast_to(
            units["daily_miles_mean"].to_numpy(dtype=float, copy=True),
            (days, settings.vehicle_count),
        ).copy(),
        "distance_cv": np.array([daily_miles_cv_by_cohort[c] for c in cohort_ids], dtype=float),
        # Weekend dwell (decision 0004 item 42 follow-up): a cohort given no
        # weekend value dwells the same every day.
        "destination_dwell_minutes": np.where(
            weekday[:, np.newaxis],
            per_unit("destination_dwell_minutes")[np.newaxis, :],
            np.array(
                [
                    behaviour_by_cohort[c].get(
                        "weekend_destination_dwell_minutes",
                        behaviour_by_cohort[c]["destination_dwell_minutes"],
                    )
                    for c in cohort_ids
                ],
                dtype=float,
            )[np.newaxis, :],
        ),
        "drive_speed_mph": per_unit("drive_speed_mph"),
        "desired_pre_drive_soc_fraction": units["preferred_target_soc_fraction"].to_numpy(
            dtype=float, copy=True
        ),
    }

    def by_cohort(name: str) -> dict[str, object]:
        return {cohort_id: behaviour_by_cohort[cohort_id][name] for cohort_id in used_cohort_ids}

    plug_in = by_cohort("plug_in_scale_minutes")
    # Decision 0004 item 42: a cohort given no weekend plug-in scale has a flat
    # window, so its weekend scale is its weekday scale (hand-built callers
    # may leave it out).
    weekend_plug_in = {
        cohort_id: behaviour_by_cohort[cohort_id].get(
            "weekend_plug_in_scale_minutes", plug_in[cohort_id]
        )
        for cohort_id in used_cohort_ids
    }
    return trip_parameters, {
        "departure": by_cohort("departure_scale_minutes"),
        "plug_in": plug_in,
        "weekend_plug_in": weekend_plug_in,
    }


# --- Trading overlay inputs and call ---------------------------------------


def _trading_boundary(
    trading_assumptions: Mapping[str, float | str] | None,
) -> dict[str, float | str]:
    """Check the trading inputs once; ``None`` takes the records' values (§1.5 step 1).

    Every value must be a finite number inside its record's bounds; a switch
    must be 0 or 1, and a count of nights or half-hours a whole number of at
    least 1 (the baseline needs at least one history night of a class and
    one adjustment slot).  ``trading.commitment_rule`` (absent: the record's
    ``fixed_share``) must be one of ``market.COMMITMENT_RULES``.  Raises
    ``ValueError`` naming the input.
    """

    defaults = model_assumptions.trading_inputs(None)
    values = dict(defaults if trading_assumptions is None else trading_assumptions)
    # The commitment rule (§10.4, §10.8) is the one text switch: it is
    # checked on its own and passed to ``market.run_trading`` as a string.
    record = model_assumptions.AVAILABILITY["trading.commitment_rule"]
    rule = values.pop("trading.commitment_rule", record.value)
    if rule not in market.COMMITMENT_RULES:
        raise ValueError(f"trading.commitment_rule must be one of {market.COMMITMENT_RULES}")
    missing = sorted(set(defaults) - set(values))
    if missing:
        raise ValueError(f"trading_assumptions missing required field: {missing[0]}")
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"trading assumption {name} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"trading assumption {name} must be finite")
    records = {
        **model_assumptions.TRADING_OVERLAY,
        "gate_closure_minutes": model_assumptions.TRADING["gate_closure_minutes"],
        "day_ahead_publication_local_hour": model_assumptions.PRICES[
            "day_ahead_publication_local_hour"
        ],
    }
    for name in defaults:
        value, record = values[name], records[name]
        if record.bounds is not None and not record.bounds[0] <= value <= record.bounds[1]:
            low, high = record.bounds
            raise ValueError(f"trading assumption {name} must be between {low:g} and {high:g}")
        if record.unit.startswith("switch") and value not in (0.0, 1.0):
            raise ValueError(f"trading assumption {name} is a switch and must be 0 or 1")
        if record.unit in ("nights", "half-hours") and (value < 1 or value != int(value)):
            raise ValueError(f"trading assumption {name} must be a whole number of at least 1")
    return {**values, "trading.commitment_rule": rule}


def _shared_factor_boundary(values: Mapping[str, float] | None) -> dict[str, float]:
    """Check the §10.1 inputs once; ``None`` takes the records' values (§10.0 step 1).

    Every value must be a finite number inside its record's bounds, the
    holiday switches 0 or 1; the manufacturer shares' sum is checked where
    they are drawn (``sampling.assign_manufacturers``).
    """

    defaults = model_assumptions.shared_factor_inputs()
    given = dict(defaults if values is None else values)
    missing = sorted(set(defaults) - set(given))
    if missing:
        raise ValueError(f"shared_factor_assumptions missing required field: {missing[0]}")
    checked: dict[str, float] = {}
    for name in defaults:
        value = given[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"shared-factor assumption {name} must be a number")
        low, high = model_assumptions.SHARED_FACTORS[name].bounds
        # Written as "not inside" so NaN fails too.
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low:g} and {high:g}")
        if name.startswith("behaviour.holiday_") and value not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1")
        checked[name] = float(value)
    return checked


def _slot_flags(
    settings: RunSettings, holiday: np.ndarray, blackout_windows: pd.DataFrame
) -> dict[str, object]:
    """``build_study_slots``/``build_warmup_slots`` keywords for the holiday and blackout flags.

    ``holiday`` (sampled date,) is ``clock.holiday_flags``'s flag, date
    ``d`` being ``start_local_date - warmup_days + d``; ``blackout_windows``
    the validated §10.5b table, whose ``slot_labels`` are London "HH:MM"
    half-hours.
    """

    first_date = settings.start_local_date - timedelta(days=settings.warmup_days)
    return {
        "holiday_dates": [first_date + timedelta(days=int(d)) for d in np.flatnonzero(holiday)],
        "blackout_labels": [
            label for labels in blackout_windows["slot_labels"] for label in labels
        ],
    }


def _availability_inputs(
    simulated: SimulatedForecast, study_slots: pd.DataFrame
) -> availability.AvailabilityInputs:
    """The run-level inputs of the Firm-MW frames (§10.2, §10.1f) from an action run.

    ``study_slots`` is the packaged result's frame (its ``blackout`` column
    is the blackout mask).

    Maker index, response rates and outage chances are the §10.1d records
    and draws; the factors, holidays and temperatures are the sampled
    worlds'.  The prices generator does not return its daily solar
    clearness, so ``world_nights.solar_clearness`` is NaN (§10.1f allows it).
    """

    values = simulated.shared_factor_assumptions
    factors = simulated.plug_in_factors
    return availability.AvailabilityInputs(
        manufacturer_index=simulated.units["manufacturer_index"].to_numpy(dtype=np.int64),
        response_rate=np.array(
            [values[f"manufacturers.response_rate.{maker_id}"] for maker_id in MANUFACTURER_IDS]
        ),
        outage_probability=np.array(
            [
                values[f"manufacturers.outage_probability_per_night.{maker_id}"]
                for maker_id in MANUFACTURER_IDS
            ]
        ),
        outage=simulated.manufacturer_outage,
        blackout=study_slots["blackout"].to_numpy(dtype=bool),
        decision_local_time=_decision_local_time(),
        warmup_days=simulated.settings.warmup_days,
        plug_in_week_factor=factors["week"],
        plug_in_day_factor=factors["day"],
        plug_in_skip_share=factors["skip_share"],
        skip_logit_shift=factors["skip_logit_shift"],
        holiday=factors["holiday"],
        temperature_c=simulated.evaluation.daily_temperature_c,
        solar_clearness=None,
    )


def _decision_local_time() -> str:
    """The intraday availability decision time, London "HH:MM" (§10.2d; a fixed record)."""

    return str(model_assumptions.AVAILABILITY["availability.intraday_decision_local_time"].value)


def _session_non_response_probability(
    control: np.ndarray,
    manufacturer_index: np.ndarray,
    outage: np.ndarray,
    response_rate: np.ndarray,
) -> np.ndarray:
    """Each session's chance of ignoring its plan, per (world, study night, EV) (§10.1e).

    ``control`` (EV,) marks the hold-out control group, ``manufacturer_index``
    (EV,) each EV's maker, ``outage`` (world, night, maker) the nights a
    maker's cloud is out and ``response_rate`` (maker,) the share of a
    maker's sessions that follow their plan.  Returns float64 fractions:
    1 for a control EV (never flexed, §9.5), 1 on its maker's outage night
    (the whole slice loses control), otherwise ``1 − r``.  Item 59 makes
    non-response come from the maker rather than the archetype, because a
    supplier can diversify makers; one uniform per session is compared with
    this, so a worse maker setting only turns responders into
    non-responders (a Compare runs on identical futures).  A scripted
    control outage still raises it per plan made, in the kernel (§9.6).
    """

    maker_out = outage[:, :, manufacturer_index]
    probability = np.where(maker_out, 1.0, 1.0 - response_rate[manufacturer_index])
    return np.where(control[np.newaxis, np.newaxis, :], 1.0, probability)


def _trading_kernel_outputs(
    settings: RunSettings,
    study_slots: pd.DataFrame,
    control: np.ndarray,
    decision_local_hour: float,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Preallocate what ``physics._record_trading`` fills, and the book decision slots.

    Plan books are taken at every whole London hour of the study (the
    intraday decision instants, §4.5).  ``decision_local_hour`` is the
    day-ahead decision hour: night 0's decision ``tau_DA(0)`` falls in the
    warm-up, and the run slots starting before it are the ones whose new
    sessions night 0's day-ahead expected need may use (§4.3, review B3).
    """

    worlds, evs = settings.evaluation_world_count, settings.vehicle_count
    slot_count = len(study_slots)
    # The same hourly decision grid the one-EV replay's plan book uses
    # (``individual.replay_one_ev_timeline``): one shared rule, so the fleet
    # run and the replay can never drift onto different book slots
    # (regression: tests/model/test_individual.py).
    book_slots, book_decision = market.book_decision_slots(settings, study_slots)
    run_ns = (
        pd.DatetimeIndex(
            utc_half_hour_boundaries(
                settings.start_local_date, settings.warmup_days, settings.study_days
            )[:-1]
        )
        .as_unit("ns")
        .asi8
    )
    first_decision_ns = market.day_ahead_decision_utc_ns(
        study_slots["local_date"].iat[0], decision_local_hour
    )
    normal: dict[str, np.ndarray] = {
        "warmup_need_kwh": np.zeros((worlds, evs)),
        "warmup_session_count": np.zeros((worlds, evs), dtype=np.int64),
        "first_day_ahead_cutoff_slot": np.int64(np.searchsorted(run_ns, first_decision_ns)),
        "first_day_ahead_need_kwh": np.zeros((worlds, evs)),
        "first_day_ahead_session_count": np.zeros((worlds, evs), dtype=np.int64),
        "departure_shortfall_kwh": np.zeros((worlds, slot_count)),
    }
    if control.any():
        normal["control_mask"] = control
        normal["control_warmup_import_kwh"] = np.zeros((worlds, settings.warmup_days * 48))
        normal["control_import_kwh"] = np.zeros((worlds, slot_count))
    selected = {
        "departure_shortfall_kwh": np.zeros((worlds, slot_count)),
        "book_decision": book_decision,
        "book_kwh": np.zeros((worlds, len(book_slots), market.BOOK_WIDTH)),
        "connected": np.zeros((worlds, slot_count + 1, evs), dtype=bool),
    }
    return {"normal": normal, "selected": selected}, book_slots


def _run_trading(
    simulated: SimulatedForecast, result: ForecastResult, sampled_world_ids: list[int]
) -> market.TradingRun:
    """Call ``market.run_trading`` with the run's kernel outputs and prices (§1.5 step 10)."""

    settings = simulated.settings
    worlds = settings.evaluation_world_count
    kernel = simulated.trading_kernel
    normal, selected = kernel["normal"], kernel["selected"]
    fleet = result.fleet_world_intervals
    run_slots = len(simulated.warmup_slots) + result.study_slot_count
    warmup = len(simulated.warmup_slots)

    def study(frame: pd.DataFrame, column: str) -> np.ndarray:
        return frame[column].to_numpy(dtype=float).reshape(worlds, run_slots)[:, warmup:]

    prices = simulated.market_prices
    smart, zones = simulated.smart_charging, simulated.zone_sums
    control = simulated.units["control_group"].to_numpy(dtype=bool)
    dispatch_sums = simulated.dispatch_sums
    dispatch_locked = simulated.units["dispatch_locked"].to_numpy(dtype=bool)
    return market.run_trading(
        study_slots=result.study_slots,
        warmup_slots=simulated.warmup_slots,
        units=simulated.units,
        unmanaged_kwh=_metric_matrix(fleet, "normal", "home_import_kwh", result.study_slot_count),
        metered_kwh=_metric_matrix(fleet, "selected", "home_import_kwh", result.study_slot_count),
        warmup_unmanaged_kwh=simulated.warmup_home_import_kwh,
        expected_need=market.expected_need_kwh(
            normal["warmup_need_kwh"],
            normal["warmup_session_count"],
            simulated.units["cohort_id"].to_numpy(dtype=object),
        ),
        first_night_expected_need=market.expected_need_kwh(
            normal["first_day_ahead_need_kwh"],
            normal["first_day_ahead_session_count"],
            simulated.units["cohort_id"].to_numpy(dtype=object),
        ),
        departure_shortfall_kwh={
            "normal": normal["departure_shortfall_kwh"],
            "selected": selected["departure_shortfall_kwh"],
        },
        book_kwh=selected["book_kwh"],
        book_slots=simulated.book_slots,
        connected=selected["connected"],
        expected_departure_utc_ns=simulated.smart_charging.expected_departure_utc_ns,
        day_ahead_gbp_per_mwh=prices.day_ahead["wholesale_forecast_gbp_per_mwh"]
        .to_numpy(dtype=float)
        .reshape(worlds, run_slots),
        run_interval_start_utc=pd.DatetimeIndex(
            prices.day_ahead["interval_start_utc"].iloc[:run_slots]
        ),
        intraday_path_gbp_per_mwh=prices.intraday_path_gbp_per_mwh,
        intraday_close_gbp_per_mwh=study(prices.realised, "evaluation_context_price_gbp_per_mwh"),
        imbalance_gbp_per_mwh=study(prices.realised, "imbalance_price_gbp_per_mwh"),
        # The NIV state behind each SIP, for the newsvendor's k (§10.4).
        system_short=~prices.realised["system_long"]
        .to_numpy(dtype=bool)
        .reshape(worlds, run_slots)[:, warmup:],
        known_shock_gw=prices.known_shock_gw[:, warmup:],
        surprise_shock_gw=prices.surprise_shock_gw[:, warmup:],
        non_response_probability=simulated.non_response_probability,
        assumptions=simulated.trading_assumptions,
        sampled_world_ids=sampled_world_ids,
        control_mask=control if control.any() else None,
        control_warmup_kwh=normal.get("control_warmup_import_kwh"),
        control_unmanaged_kwh=normal.get("control_import_kwh"),
        # Events and zones (trading contract §2.4, §4.6): request windows
        # pause trading once known and are paid against the scope's own
        # baseline; the expected plan ranks the requests' planner signals.
        events=simulated.events,
        zone_index=smart.zone_index,
        zone_warmup_kwh=zones.warmup_home_import_kwh,
        zone_metered_kwh=zones.study["selected"][
            ..., ZONE_SUM_COLUMNS.index("home_grid_import_kwh")
        ],
        event_notice_slot=smart.event_notice_slot,
        event_adjustment_gbp_per_mwh=smart.event_adjustment_gbp_per_mwh,
        # Blackout half-hours are closed to trading at every decision (§10.5b).
        blackout=result.study_slots["blackout"].to_numpy(dtype=bool),
        # Intraday dispatch (intraday-dispatch-v1 §6): metered import, the
        # zone import, the book and the connection states above are all the
        # selected path's, which is the dispatched fleet; free sessions not
        # yet started are expected to plan on the latest price, and the
        # day-ahead plan path's book feeds the frozen-book re-run that splits
        # intraday P&L into rebalancing and re-optimisation (§6.2).
        dispatch_locked=None if dispatch_sums is None else dispatch_locked,
        day_ahead_book_kwh=None if dispatch_sums is None else dispatch_sums.book_kwh["day_ahead"],
    )


# --- Packaging to contract v2 ----------------------------------------------


def _package(simulated: SimulatedForecast, fixture: CohortFixture) -> ForecastResult:
    """Rename and reshape the raw run into a ``ForecastResult`` (contract v2)."""

    settings = simulated.settings
    model = simulated.model
    is_action = model == "action"
    study_slots = build_study_slots(
        settings.start_local_date,
        settings.study_days,
        **_slot_flags(settings, simulated.plug_in_factors["holiday"], simulated.blackout_windows),
    )
    # Contract 1 rule 2: a no-action run has only the normal path.  The
    # kernel copies it as "selected"; that copy is dropped so no chart can
    # show a fake selected path.
    two_path = ("normal", "selected") if is_action else ("normal",)
    # The optional "timed" path (decision 0007), when set, joins the
    # fleet/cohort interval frames and, from model step 2, the zone frames
    # below (``zone_world_intervals``, since the kernel now tracks its zone
    # sums too); every other pairwise consumer further down (weekly totals,
    # and everything ``build_summaries`` adds beyond zones) keeps the
    # explicit two (or one) paths above, never this tuple.
    timed_active = not math.isnan(simulated.timed_start_local_hour)
    interval_paths = (*two_path, "timed") if timed_active else two_path
    # The run's own barred-window mask (decision 0007), recomputed from the
    # stored hour rather than kept as a second field: a pure function of
    # ``settings`` and the hour, so there is nothing to drift.  Kept on
    # ``ReplayState`` so the per-EV chunk pass and the one-EV replay reuse
    # it rather than each re-deriving the rule.
    timed_start_allowed_mask = (
        timed_start_allowed(settings, simulated.timed_start_local_hour) if timed_active else None
    )
    fleet_world = _world_frame(simulated.fleet_world_intervals, study_slots, interval_paths)
    cohort_order = [spec.cohort_id for spec in fixture.cohorts]
    cohort_world = _world_frame(
        simulated.cohort_world_intervals, study_slots, interval_paths, cohort_order=cohort_order
    )
    cost_effect = simulated.cost_effect
    horizon_start = study_slots["interval_start_utc"].iat[0]
    london_offsets = study_slots["interval_start_london"].map(lambda t: t.utcoffset())
    return ForecastResult(
        model=model,
        model_label=MODEL_LABELS[model],
        policy_id=POLICY_IDS[model],
        evidence_kind=EVIDENCE_KINDS[model],
        run_completed_at_utc=pd.Timestamp.now(tz="UTC"),
        seed=settings.seed,
        vehicle_count=settings.vehicle_count,
        world_count=settings.evaluation_world_count,
        study_start_local_date=settings.start_local_date,
        horizon_start_utc=horizon_start,
        horizon_end_utc=study_slots["interval_end_utc"].iat[-1],
        study_slot_count=len(study_slots),
        has_clock_change=london_offsets.nunique() > 1,
        settings_snapshot={
            "model": model,
            "start_local_date": settings.start_local_date,
            "warmup_days": settings.warmup_days,
            "study_days": settings.study_days,
            "vehicle_count": settings.vehicle_count,
            "seed": settings.seed,
            "evaluation_world_count": settings.evaluation_world_count,
        },
        assumptions=None,
        # Smart charging applies to every EV on the selected path; there is no
        # select-or-not decision any more (decision 0004 item 38).
        action_status="selected" if is_action else "not_applicable",
        # Both models top up in public (decision 0004 item 32, lead decision).
        public_charging_status="modelled",
        # Weeks materially not recovered (decision 0004 item 45), not every
        # week with any shortfall beyond the tolerance.
        not_recovered_world_count=(
            int(cost_effect["not_recovered_material"].sum()) if is_action else None
        ),
        representative_world_id=_representative_world(fleet_world),
        study_slots=study_slots,
        units=_units_frame(simulated.units),
        fleet_world_intervals=fleet_world,
        cohort_world_intervals=cohort_world,
        fleet_interval_bands=_band_frame(fleet_world, study_slots, interval_paths),
        cohort_interval_bands=pd.concat(
            [
                _band_frame(group, study_slots, interval_paths).assign(cohort_id=cohort_id)
                for cohort_id, group in cohort_world.groupby("cohort_id", sort=False)
            ],
            ignore_index=True,
        ).loc[:, ["cohort_id", "metric", "unit", "path_id", *_SLOT_KEYS, *_BAND_STATS]],
        weekly_bands=_weekly_bands(fleet_world, two_path),
        forecast_prices=_with_slot_keys(simulated.forecast_prices, study_slots, london=True),
        evaluation_prices=_with_slot_keys(simulated.evaluation_prices, study_slots, london=False),
        cost_effect=cost_effect,
        timed_cost_effect=simulated.timed_cost_effect,
        # Same rule as ``not_recovered_world_count`` above, from the timed
        # path's own cost effect; ``None`` whenever that frame is (no-action
        # model, or ``timed_start_local_hour`` unset) rather than keyed off
        # ``is_action`` directly, since a no-action run never fills it in.
        timed_not_recovered_world_count=(
            None
            if simulated.timed_cost_effect is None
            else int(simulated.timed_cost_effect["not_recovered_material"].sum())
        ),
        price_shocks=_study_shocks(simulated),
        events=simulated.events,
        blackout_windows=simulated.blackout_windows,
        zone_ids=ZONE_IDS,
        zone_world_intervals=_zone_world_frame(simulated, study_slots, interval_paths),
        replay_state=ReplayState(
            settings=settings,
            units=simulated.units,
            evaluation=simulated.evaluation,
            public_top_up=simulated.public_top_up,
            smart_charging=simulated.smart_charging,
            # The per-EV chunk pass and the One EV replay dispatch too, so
            # their selected path reconciles with the fleet's (§5.2).
            intraday_dispatch=simulated.intraday_dispatch,
            timed_start_allowed=timed_start_allowed_mask,
            # Built once and kept: build_summaries and the household card
            # read the same inputs (household contract v1 §4 hunk 4).
            flexibility=None
            if simulated.departure_margin_hours is None
            else flexibility_inputs(
                settings, simulated.units, study_slots, simulated.departure_margin_hours
            ),
            # The firm-MW inputs (§10.2), kept for build_summaries and the
            # household card's realised-kW path; action runs only.
            availability_inputs=(
                _availability_inputs(simulated, study_slots) if is_action else None
            ),
        ),
    )


def _zone_world_frame(
    simulated: SimulatedForecast, study_slots: pd.DataFrame, paths: tuple[str, ...]
) -> pd.DataFrame:
    """Per-zone world sums (trading contract v1 §3), one row per (world, path, zone, slot).

    Built from the kernel's ``ZoneSums`` arrays with the contract's names:
    ``unit_count``, ``connected_count`` and ``early_departure_count`` as
    counts, ``home_import_kwh`` (kWh per half-hour) and ``home_import_kw``.
    Zones partition the fleet, so each (world, path, slot) adds up to
    ``fleet_world_intervals``.  World-major, then path, zone and slot.
    """

    sums = simulated.zone_sums
    world_count = simulated.settings.evaluation_world_count
    zone_count, slot_count = len(ZONE_IDS), len(study_slots)
    column = {name: index for index, name in enumerate(ZONE_SUM_COLUMNS)}
    frames = []
    for path_id in paths:
        values = sums.study[path_id]
        frames.append(
            pd.DataFrame(
                {
                    "world_id": np.repeat(np.arange(world_count), zone_count * slot_count),
                    "path_id": path_id,
                    "zone_id": np.tile(
                        np.repeat(np.array(ZONE_IDS, dtype=object), slot_count), world_count
                    ),
                    "slot_index": np.tile(
                        study_slots["slot_index"].to_numpy(), world_count * zone_count
                    ),
                    **{name: values[..., column[name]].reshape(-1) for name in ZONE_SUM_COLUMNS},
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True).rename(
        columns={"home_grid_import_kwh": "home_import_kwh"}
    )
    for name in ("unit_count", "connected_count", "early_departure_count"):
        frame[name] = np.rint(frame[name]).astype(np.int64)
    frame["home_import_kw"] = frame["home_import_kwh"] / _SLOT_HOURS
    frame = frame.merge(
        study_slots.loc[:, ["slot_index", "interval_start_utc", "interval_start_london"]],
        on="slot_index",
        how="left",
    )
    frame["_path_rank"] = frame["path_id"].map(PATH_ORDER.index)
    frame["_zone_rank"] = frame["zone_id"].map(ZONE_IDS.index)
    return (
        frame.sort_values(["world_id", "_path_rank", "_zone_rank", "slot_index"], kind="stable")
        .loc[
            :,
            [
                "world_id",
                "path_id",
                "zone_id",
                "slot_index",
                "interval_start_utc",
                "interval_start_london",
                "unit_count",
                "connected_count",
                "home_import_kwh",
                "home_import_kw",
                "early_departure_count",
                "early_departure_shortfall_kwh",
            ],
        ]
        .reset_index(drop=True)
    )


def _world_frame(
    frame: pd.DataFrame,
    study_slots: pd.DataFrame,
    paths: tuple[str, ...],
    *,
    cohort_order: list[str] | None = None,
) -> pd.DataFrame:
    """Rename kernel world sums to contract 3.4/3.5 names, one row per key.

    Columns without a contract name (for example ``conservation_residual_kwh``
    and the served/unserved travel split) stay as audit columns (contract 1
    rule 10).  Renamed kernel columns are dropped, so each value has one name.
    """

    world = frame.loc[frame["path_id"].isin(paths)].rename(columns=_KERNEL_RENAMES)
    for old, new in _FRACTION_TO_SHARE.items():
        world[new] = world.pop(old) / world["unit_count"]
    world["home_import_kw"] = world["home_import_kwh"] / _SLOT_HOURS
    world["unserved_travel_kwh"] = (
        world["unserved_outbound_travel_battery_kwh"] + world["unserved_return_travel_battery_kwh"]
    )
    # The kernel sums per-EV 0/1 flags as floats; a count is an integer
    # (contract 1 rule 5).
    world["early_departure_count"] = np.rint(world["early_departure_count"]).astype(np.int64)
    world = world.merge(
        study_slots.loc[:, _SLOT_KEYS], on=["interval_start_utc", "interval_end_utc"], how="left"
    )
    keys = ["world_id", "path_id", *_SLOT_KEYS, "unit_count", "physical_capacity_kwh"]
    sort_keys = ["world_id", "_path_rank", "slot_index"]
    if cohort_order is not None:
        keys.insert(0, "cohort_id")
        world["_cohort_rank"] = world["cohort_id"].map(cohort_order.index)
        sort_keys.insert(0, "_cohort_rank")
    world["_path_rank"] = world["path_id"].map(PATH_ORDER.index)
    extras = [c for c in world.columns if c not in {*keys, *FLEET_METRICS} and c[0] != "_"]
    return (
        world.sort_values(sort_keys, kind="stable")
        .loc[:, [*keys, *FLEET_METRICS, *extras]]
        .reset_index(drop=True)
    )


def _metric_matrix(world: pd.DataFrame, path_id: str, metric: str, slot_count: int) -> np.ndarray:
    """(world, slot) matrix of one metric on one path; rows are sorted by world, slot."""

    values = world.loc[world["path_id"].eq(path_id), metric].to_numpy(dtype=float)
    return values.reshape(-1, slot_count)


def _quantile_columns(values: np.ndarray) -> dict[str, np.ndarray]:
    # Worlds first (contract 1 rule 3): each row of ``values`` is already a
    # per-world total, so quantiles are taken across worlds only here.
    p10, p50, p90 = np.quantile(values, (0.1, 0.5, 0.9), axis=0, method="linear")
    return {
        "world_count": np.int64(values.shape[0]),
        "mean": values.mean(axis=0),
        "p10": p10,
        "p50": p50,
        "p90": p90,
    }


def _band_frame(world: pd.DataFrame, study_slots: pd.DataFrame, paths) -> pd.DataFrame:
    """Long across-world bands, one row per (metric, path, slot) (contract 3.6)."""

    slot_keys = study_slots.loc[:, _SLOT_KEYS]
    frames = []
    for metric, unit in FLEET_METRICS.items():
        for path_id in paths:
            values = _metric_matrix(world, path_id, metric, len(slot_keys))
            frame = slot_keys.assign(**_quantile_columns(values))
            frame.insert(0, "path_id", path_id)
            frame.insert(0, "unit", unit)
            frame.insert(0, "metric", metric)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _weekly_bands(world: pd.DataFrame, paths) -> pd.DataFrame:
    """Weekly totals per world, then P10/P50/P90 and max across worlds.

    One row per (path, metric) for ``WEEKLY_METRICS`` in kWh per week.
    """

    rows = []
    for path_id in paths:
        weekly = world.loc[world["path_id"].eq(path_id)].groupby("world_id")[list(WEEKLY_METRICS)]
        totals = weekly.sum()
        for metric in WEEKLY_METRICS:
            values = totals[metric].to_numpy(dtype=float)
            stats = _quantile_columns(values[:, np.newaxis])
            rows.append(
                {
                    "path_id": path_id,
                    "metric": metric,
                    "unit": "kWh per week",
                    **{key: np.asarray(value).item() for key, value in stats.items()},
                    "max": float(values.max()),
                }
            )
    frame = pd.DataFrame(rows)
    return frame.astype({"world_count": np.int64})


def _representative_world(fleet_world: pd.DataFrame) -> int:
    """World with the lower-median weekly normal-path home import (contract 2).

    Rank ``(n - 1) // 2`` so the choice is an actual world, not an average of
    two; ties go to the lower ``world_id``.
    """

    weekly = (
        fleet_world.loc[fleet_world["path_id"].eq("normal")]
        .groupby("world_id")["home_import_kwh"]
        .sum()
        .reset_index()
        .sort_values(["home_import_kwh", "world_id"], kind="stable")
    )
    return int(weekly["world_id"].iat[(len(weekly) - 1) // 2])


def _units_frame(units: pd.DataFrame) -> pd.DataFrame:
    """EV traits with contract 3.2 names; kernel-only columns stay as extras."""

    frame = units.rename(columns={"cohort_source_name": "cohort_label"})
    frame["preferred_target_soc_percent"] = 100.0 * frame.pop("preferred_target_soc_fraction")
    first = [
        "unit_id",
        "cohort_id",
        "cohort_label",
        "daily_miles_mean",
        "physical_capacity_kwh",
        "home_charger_limit_kw",
        "preferred_target_soc_percent",
    ]
    return frame.loc[:, [*first, *(c for c in frame.columns if c not in first)]]


def _study_shocks(simulated: SimulatedForecast) -> pd.DataFrame | None:
    """The shocks that start in the study, with study slot indices (results-v2 ``price_shocks``).

    Prices, and so shocks, are generated over the warm-up too (decision 0004
    item 52); the generator counts ``start_slot_index`` from the first warm-up
    slot.  The result keeps the study's shocks and counts from slot 0.
    """

    if simulated.market_prices is None:
        return None
    shocks = simulated.market_prices.shocks
    warmup_slots = 48 * simulated.settings.warmup_days
    study = shocks.loc[shocks["start_slot_index"] >= warmup_slots].copy()
    study["start_slot_index"] -= warmup_slots
    return study.reset_index(drop=True)


def _with_slot_keys(
    prices: pd.DataFrame | None, study_slots: pd.DataFrame, *, london: bool
) -> pd.DataFrame | None:
    """Keep a price frame's study slots and add ``slot_index`` (and London time), contract 4.7.

    The generated prices also cover the warm-up (decision 0004 item 52); the
    result shows the study only, so the inner merge drops warm-up rows.
    """

    if prices is None:
        return None
    keys = ["slot_index", "interval_start_utc", *(["interval_start_london"] if london else [])]
    frame = prices.merge(study_slots.loc[:, keys], on="interval_start_utc", how="inner")
    lead = ["world_id"] if "world_id" in frame else []
    first = [*lead, "slot_index", "interval_start_utc", "interval_end_utc"]
    if london:
        first.append("interval_start_london")
    return frame.loc[:, [*first, *(c for c in frame.columns if c not in first)]]


__all__ = [
    "COHORT_ORDER",
    "DAY_TYPE_ORDER",
    "FLEET_METRICS",
    "LOCATION_ORDER",
    "MODEL_LABELS",
    "PATH_ORDER",
    "WEEKLY_METRICS",
    "ActionSummary",
    "ForecastResult",
    "PlugInSummary",
    "ReplayState",
    "RunComparison",
    "RunSummary",
    "SampledWorlds",
    "SimulatedForecast",
    "compare_runs",
    "run_forecast",
    "run_forecast_from_assumptions",
    "simulate_forecast",
    "slim_run",
    "validate_public_charge_rate",
]
