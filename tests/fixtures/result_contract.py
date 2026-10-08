"""Contract v2 specs, reference computations and validators (`docs/contracts/results-v2.md`).

What this owns: the machine-readable form of the result contract (column
names and dtypes, units, category orders), the reference world-first
computations the contract defines, and ``validate_result_v2`` and friends.
The validators are duck-typed (attributes and columns only), so the
SYNTHETIC FIXTURE in ``result_fixture.py`` and M5's real results pass the
same checks.

Authority: decision 0004 items 1-38, 52 and 54 (notably 1 and 52: 336 fixed
UTC slots from London noon, reported by session night; 12: world-first
difference bands; 13: unserved travel priced; 32/37: public top-ups, never
stranded; 38: smart charging replaced the 18:00 import cap, its eligibility
screen and the planning world).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .household_contract import validate_household_v2

LONDON = "Europe/London"
SLOT_COUNT = 336
STUDY_DAYS = 7
HALF_HOUR = pd.Timedelta(minutes=30)
EVIDENCE_LABEL = "SYNTHETIC FIXTURE: not model output"

# Category orders (contract section 1, rule 9).
PATH_ORDER = ("normal", "selected")
INTERVAL_PATH_ORDER = ("normal", "selected", "timed")
"""``PATH_ORDER`` plus the optional "timed" path (decision 0007). Used, filtered to
the paths a given frame actually has, for every frame that may carry it:
fleet_world_intervals, cohort_world_intervals, fleet_interval_bands,
cohort_interval_bands and, from model step 2, zone_world_intervals,
zone_import_bands, zone_summary and weekly_peak_summary. The SYNTHETIC FIXTURE in
``result_fixture.py``, replay, and every strictly pairwise frame below (difference
bands, trading, availability, dispatch, fleet_interval_ev_bands, weekly_bands,
average_day_bands, run comparison) keep ``PATH_ORDER`` or an explicit two-path
tuple, so they never pick up "timed"."""
DAY_TYPE_ORDER = ("weekday", "weekend", "all")
LOCATION_ORDER = ("home_plugged", "home_unplugged", "driving", "away", "public_charging")
COHORTS = (
    ("average_uk", "Average (UK)", 0.40),
    ("intelligent_octopus", "Intelligent Octopus average", 0.30),
    ("infrequent_charging", "Infrequent charging", 0.10),
    ("infrequent_driving", "Infrequent driving", 0.10),
    ("scheduled_charging", "Scheduled charging", 0.09),
    ("always_plugged_in", "Always plugged-in", 0.01),
)
COHORT_ORDER = tuple(cohort_id for cohort_id, _, _ in COHORTS)

# Fleet metrics and units (contract 3.5), in display order.
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
SHARE_METRICS = ("connected_share", "driving_share", "away_share", "public_charging_share")
COST_COMPONENTS = (
    ("home_import_cost", "illustrative_selected_minus_normal_energy_cost_gbp"),
    ("public_charge_cost", "illustrative_selected_minus_normal_public_charge_cost_gbp"),
    ("unrecovered_energy_value", "illustrative_unrecovered_energy_value_gbp"),
    ("unserved_travel_value", "illustrative_unserved_travel_value_gbp"),
    ("total", "illustrative_selected_minus_normal_total_gbp"),
)
WEEKLY_DIFFERENCE_METRICS = {
    "home_import_kwh": "kWh per week",
    "public_import_kwh": "kWh per week",
    "unserved_travel_kwh": "kWh per week",
    "closing_battery_kwh": "kWh",
}
CNZ_RECORD_NAMES = (
    "cnz_median_plug_in_soc_percent",
    "cnz_share_plug_ins_below_10_percent_soc",
    "cnz_share_plug_ins_below_20_percent_soc_upper_bound",
    "cnz_weekday_plug_in_mode_local_hour",
)
SETTINGS_KEYS = (
    "model",
    "start_local_date",
    "warmup_days",
    "study_days",
    "vehicle_count",
    "seed",
    "evaluation_world_count",
)
NOT_RECOVERED_TOLERANCE_KWH = 1e-6  # decision 0004 items 5 and 13
# Decision 0004 item 45: a week is materially not recovered from this share
# (percent of weekly normal home import); the default the fixture uses.
NOT_RECOVERED_MATERIAL_SHARE_PERCENT = 1.0
# M5 builds per-EV summaries in chunks of at most 10 worlds, so the fleet
# plug-in event table holds at most 10 worlds (lead decision Q1).
MAX_PLUG_IN_EVENT_WORLDS = 10
WEEKLY_BAND_METRICS = {
    "home_import_kwh": "kWh per week",
    "public_import_kwh": "kWh per week",
    "unserved_travel_kwh": "kWh per week",
}
# Fleet metrics that add across EVs, so cohort band means sum to fleet means.
ADDITIVE_METRICS = (
    "connected_count",
    "home_import_kwh",
    "home_import_kw",
    "public_import_kwh",
    "total_import_kw",
    "closing_battery_kwh",
    "unserved_travel_kwh",
)

# Column specs: name -> kind.  Kinds are checked by ``_check_dtypes``.
SLOT_COLUMNS = {
    "slot_index": "int",
    "interval_start_utc": "utc",
    "interval_end_utc": "utc",
    "interval_start_london": "london",
}
BAND_STAT_COLUMNS = {
    "world_count": "int",
    "mean": "float",
    "p10": "float",
    "p50": "float",
    "p90": "float",
}
STUDY_SLOT_COLUMNS = {
    **SLOT_COLUMNS,
    "local_date": "date",
    "night_index": "int",
    "day_type": "str",
    "local_half_hour": "int",
    "local_time_label": "str",
    "day_label": "str",
    # Firm MW (trading contract v1 §10.1c, §10.5b): holiday nights and
    # blackout half-hours, on every result.
    "holiday": "bool",
    "blackout": "bool",
}
UNIT_COLUMNS = {
    "unit_id": "str",
    "cohort_id": "str",
    "cohort_label": "str",
    "daily_miles_mean": "float",
    "physical_capacity_kwh": "float",
    "home_charger_limit_kw": "float",
    "preferred_target_soc_percent": "float",
    "zone_id": "str",
}
COHORT_SUMMARY_COLUMNS = {
    "cohort_id": "str",
    "cohort_label": "str",
    "source_population_share": "float",
    "ev_count": "int",
    "plug_ins_per_ev_week_p50": "float",
    "median_plug_in_soc_percent_p50": "float",
    "peak_plug_in_local_hour": "Int64",
    "unserved_travel_kwh_p50": "float",
}
FLEET_WORLD_COLUMNS = {
    "world_id": "int",
    "path_id": "str",
    **SLOT_COLUMNS,
    "unit_count": "int",
    "physical_capacity_kwh": "float",
    **{name: "int" if name == "connected_count" else "float" for name in FLEET_METRICS},
    # Smart-charging audit columns (decision 0004 item 38); always 0 on the
    # normal path, which has no plans.
    "early_departure_count": "int",
    "early_departure_shortfall_kwh": "float",
}
FLEET_BAND_COLUMNS = {
    "metric": "str",
    "unit": "str",
    "path_id": "str",
    **SLOT_COLUMNS,
    **BAND_STAT_COLUMNS,
}
COHORT_BAND_COLUMNS = {"cohort_id": "str", **FLEET_BAND_COLUMNS}
EV_BAND_METRICS = {"battery_soc_percent": "percent", "home_import_kw": "kW"}
FLEET_EV_BAND_COLUMNS = {
    "metric": "str",
    "unit": "str",
    "path_id": "str",
    **SLOT_COLUMNS,
    "world_count": "int",
    "p10": "float",
    "p50": "float",
    "p90": "float",
}
# Contract 3.6c (decision 0004 item 43): (metric, unit, spread) row groups.
FLEXIBILITY_ROWS = (
    ("flexible_power_kw", "kW", "across_weeks"),
    ("turn_up_headroom_kw", "kW", "across_weeks"),
    ("movable_energy_kwh", "kWh", "across_weeks"),
    ("time_slack_hours", "hours", "across_evs"),
    ("time_slack_hours", "hours", "across_weeks"),
)
FLEXIBILITY_BAND_COLUMNS = {
    "metric": "str",
    "unit": "str",
    "spread": "str",
    "path_id": "str",
    **SLOT_COLUMNS,
    "world_count": "int",
    "p10": "float",
    "p50": "float",
    "p90": "float",
}
# Contract 3.10a (decision 0004 item 43).
WEEKDAY_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
PLUG_IN_HEATMAP_COLUMNS = {
    "weekday": "int",
    "weekday_label": "str",
    "local_hour": "int",
    "day_count": "int",
    "world_count": "int",
    "mean": "float",
    "p10": "float",
    "p50": "float",
    "p90": "float",
}
# Contract 3.10b (decision 0004 item 54): plug-in and plug-out heatmaps by
# London calendar weekday x wall-clock half-hour.
PLUG_EVENT_HEATMAP_COLUMNS = {
    "weekday": "int",
    "weekday_label": "str",
    "local_half_hour": "int",
    "local_time_label": "str",
    "slot_count": "int",
    "world_count": "int",
    "mean": "float",
    "p10": "float",
    "p50": "float",
    "p90": "float",
}
# Contract 3.10c (decision 0004 item 54, plan D-2): metric -> (unit, lower edge
# of the first bin, bin width, bin count).  Values outside the range count in
# the first or last bin.
SESSION_DISTRIBUTION_METRICS = {
    "plug_in_time": ("hour (London)", 0.0, 0.5, 48),
    "departure_time": ("hour (London)", 0.0, 0.5, 48),
    "plug_in_soc_percent": ("percent", 0.0, 5.0, 20),
    "energy_needed_kwh": ("kWh battery-side", 0.0, 5.0, 16),
    "dwell_hours": ("hours", 0.0, 2.0, 24),
    "flexible_kwh": ("kWh battery-side", 0.0, 5.0, 16),
    "slack_hours": ("hours", -12.0, 2.0, 30),
    # Supplier contract v1 §4.6: SoC at unplug on the normal and smart paths.
    "departure_soc_percent": ("percent", 0.0, 5.0, 20),
    "departure_soc_percent_smart": ("percent", 0.0, 5.0, 20),
    # Decision 0007, model step 2: the timed path's own departure SoC, a row
    # (not a column) that only exists when a run's `session_distribution_bands`
    # actually carries it (a run with `timed_start_local_hour` set); unlike
    # `departure_soc_percent_smart` this is never NaN-filled in on an
    # unrelated run, so this spec entry is looked up, not iterated, for a run
    # without it (see `_validate_session_distributions`).
    "departure_soc_percent_timed": ("percent", 0.0, 5.0, 20),
}
SESSION_DISTRIBUTION_COLUMNS = {
    "group_id": "str",
    "day_type": "str",
    "metric": "str",
    "unit": "str",
    "bin_index": "int",
    "bin_lower": "float",
    "bin_upper": "float",
    "bin_label": "str",
    "world_count": "int",
    "share_mean": "float",
    "share_p10": "float",
    "share_p50": "float",
    "share_p90": "float",
}
# Contract 8.2: the workbook's per-archetype plug-in SoC and kWh per plug-in.
SESSION_SOURCE_RECORD_NAMES = tuple(
    f"{cohort_id}.{field}"
    for cohort_id in COHORT_ORDER
    for field in ("source_plug_in_soc_percent", "source_battery_kwh_per_plug_in")
)
# Contract 3.6d (decision 0004 item 54, plan C2): (bucket, upper edge in hours).
SLACK_BUCKETS = (
    ("under_1h", 1.0),
    ("1_to_2h", 2.0),
    ("2_to_4h", 4.0),
    ("4_to_8h", 8.0),
    ("8h_or_more", np.inf),
)
AT_LEAST_HOURS = (1, 2, 4, 8)
DEFERRABLE_BUCKET_ORDER = (
    *(bucket for bucket, _ in SLACK_BUCKETS),
    *(f"at_least_{hours}h" for hours in AT_LEAST_HOURS),
    "total",
)
DEFERRABLE_POWER_BAND_COLUMNS = {
    "slack_bucket": "str",
    "unit": "str",
    "path_id": "str",
    **SLOT_COLUMNS,
    "world_count": "int",
    "p10": "float",
    "p50": "float",
    "p90": "float",
}
# Contract 3.6e (plan E1, lead decision): reporting points, not model assumptions.
DEFERRABLE_CAPACITY_SHARE = 0.25
REPORT_HALF_HOURS = {"1800": 36, "2200": 44, "0300": 6}
WAIT_CHECK_HALF_HOUR = 38
MOVABLE_CHECK_HALF_HOUR = 36
_Q = ("p10", "p50", "p90")
FLEXIBILITY_WEEKLY_COLUMNS = {
    "path_id": "str",
    "world_count": "int",
    "fleet_charger_capacity_kw": "float",
    **{f"peak_deferrable_kw_{q}": "float" for q in _Q},
    "peak_modal_slot_index": "int",
    "peak_modal_interval_start_utc": "utc",
    "peak_modal_interval_start_london": "london",
    "peak_modal_week_count": "int",
    **{f"hours_at_least_quarter_capacity_{q}": "float" for q in _Q},
    **{f"movable_energy_1800_kwh_{q}": "float" for q in _Q},
    **{f"deferrable_kw_{t}_{q}": "float" for t in REPORT_HALF_HOURS for q in _Q},
    **{f"w_per_plugged_in_ev_{t}_{q}": "float" for t in REPORT_HALF_HOURS for q in _Q},
    **{f"deferrable_at_least_2h_kw_1900_{q}": "float" for q in _Q},
}
# Contract 4.7a and 4.7b (plan D-3, E1).
PRICE_RELATIVE_COLUMNS = {
    "weekday": "Int64",
    "weekday_label": "str",
    "local_half_hour": "int",
    "local_time_label": "str",
    "unit": "str",
    "sample_count": "int",
    "world_count": "int",
    "mean": "float",
    "p10": "float",
    "p50": "float",
    "p90": "float",
}
CHEAPEST_HALF_HOUR_COLUMNS = {
    "world_day_count": "int",
    "local_hour_p10": "float",
    "local_hour_p50": "float",
    "local_hour_p90": "float",
    "local_time_label_p10": "str",
    "local_time_label_p50": "str",
    "local_time_label_p90": "str",
}
DIFFERENCE_BAND_COLUMNS = {"metric": "str", "unit": "str", **SLOT_COLUMNS, **BAND_STAT_COLUMNS}
AVERAGE_DAY_COLUMNS = {
    "group_id": "str",
    "path_id": "str",
    "day_type": "str",
    "metric": "str",
    "unit": "str",
    "spread": "str",
    "local_half_hour": "int",
    "local_time_label": "str",
    "centre_stat": "str",
    "low_stat": "str",
    "high_stat": "str",
    "centre": "float",
    "low": "float",
    "high": "float",
    "world_count": "int",
    "day_count": "int",
}
PLUG_IN_EVENT_COLUMNS = {
    "world_id": "int",
    "path_id": "str",
    "unit_id": "str",
    "cohort_id": "str",
    "event_index": "int",
    "plug_in_utc": "utc",
    "plug_in_london": "london",
    "local_date": "date",
    "day_type": "str",
    "plug_in_local_hour": "int",
    "plug_in_soc_percent": "float",
    "plug_out_utc": "utc",
    "plug_out_soc_percent": "float",
    "home_import_kwh": "float",
    "battery_added_kwh": "float",
    "still_plugged_at_horizon_end": "bool",
}
HOUR_BAND_COLUMNS = {
    "day_type": "str",
    "local_hour": "int",
    "world_count": "int",
    "count_p50": "float",
    "share_mean": "float",
    "share_p10": "float",
    "share_p50": "float",
    "share_p90": "float",
}
SOC_BAND_COLUMNS = {
    "day_type": "str",
    "soc_bin_lower_percent": "int",
    "soc_bin_upper_percent": "int",
    "world_count": "int",
    "share_mean": "float",
    "share_p10": "float",
    "share_p50": "float",
    "share_p90": "float",
}
PLUG_KPI_COLUMNS = {
    "day_type": "str",
    "metric": "str",
    "unit": "str",
    "p10": "float",
    "p50": "float",
    "p90": "float",
}
WORLD_KPI_EXTRA_METRICS = {"modal_plug_in_local_hour": "hour (London)"}
PLUG_KPI_METRICS = {
    "plug_ins_per_week": "plug-ins",
    "plug_ins_per_ev_per_week": "plug-ins per EV",
    "median_plug_in_soc_percent": "percent",
    "share_below_10_percent_soc": "fraction",
}
PLUG_IN_WORLD_KPI_COLUMNS = {
    "world_id": "int",
    "group_id": "str",
    "day_type": "str",
    "metric": "str",
    "unit": "str",
    "value": "float",
}
WEEKLY_BAND_COLUMNS = {
    "path_id": "str",
    "metric": "str",
    "unit": "str",
    **BAND_STAT_COLUMNS,
    "max": "float",
}
DIFFERENCE_WEEKLY_COLUMNS = {
    "world_id": "int",
    "metric": "str",
    "unit": "str",
    "normal_value": "float",
    "selected_value": "float",
    "selected_minus_normal": "float",
}
DIFFERENCE_WEEKLY_BAND_COLUMNS = {"metric": "str", "unit": "str", **BAND_STAT_COLUMNS}
ACTION_ID = "smart_charging_v1"
SMART_CHARGING_WORLD_COLUMNS = {
    "world_id": "int",
    "normal_average_price_gbp_per_mwh": "float",
    "selected_average_price_gbp_per_mwh": "float",
    "moved_home_import_kwh": "float",
    "moved_home_import_share": "float",
    "early_departure_count": "int",
    "early_departure_shortfall_kwh": "float",
    "evidence_kind": "str",
}
SMART_CHARGING_METRICS = {
    "normal_average_price_gbp_per_mwh": "GBP/MWh",
    "selected_average_price_gbp_per_mwh": "GBP/MWh",
    "average_price_change_gbp_per_mwh": "GBP/MWh",
    "moved_home_import_share": "fraction",
    "early_departure_count": "sessions per week",
    "early_departure_shortfall_kwh": "kWh per week",
    # Decision 0007, model step 2: the timed path's own price paid, present
    # only on a run that has it (see smart_charging_world_from); no "change"
    # reading of its own, unlike the normal/selected pair above.
    "timed_average_price_gbp_per_mwh": "GBP/MWh",
}
SMART_CHARGING_SUMMARY_COLUMNS = {"metric": "str", "unit": "str", **BAND_STAT_COLUMNS}
PRICE_BANDS = ("low", "middle", "high")
PRICE_BAND_SHIFT_COLUMNS = {
    "local_date": "date",
    "day_label": "str",
    "price_band": "str",
    "forecast_price_min_gbp_per_mwh": "float",
    "forecast_price_max_gbp_per_mwh": "float",
    "mean_slot_count": "float",
    "unit": "str",
    **BAND_STAT_COLUMNS,
    # Each path's own home import in the band (decision 0004 item 46).  Price
    # band shift stays normal versus selected only, never the optional
    # "timed" path (decision 0007).
    **{
        f"{path}_kwh_{stat}": "float"
        for path in ("normal", "selected")
        for stat in ("p10", "p50", "p90")
    },
}
FORECAST_PRICE_COLUMNS = {
    "world_id": "int",
    **SLOT_COLUMNS,
    "forecast_available_at_utc": "utc",
    "wholesale_forecast_gbp_per_mwh": "float",
    # Synthetic system net demand behind the price, known shocks included
    # (decision 0004 items 53 and 56).
    "system_net_demand_gw": "float",
    "evidence_kind": "str",
}
FORECAST_PRICE_BAND_COLUMNS = {
    **SLOT_COLUMNS,
    "world_count": "int",
    "p10": "float",
    "p50": "float",
    "p90": "float",
}
WEEKLY_PEAK_COLUMNS = {
    "path_id": "str",
    "unit": "str",
    "world_count": "int",
    "p10": "float",
    "p50": "float",
    "p90": "float",
    "modal_peak_slot_index": "int",
    "modal_peak_interval_start_utc": "utc",
    "modal_peak_interval_start_london": "london",
    "modal_peak_week_count": "int",
    "ratio_world_count": "int",
    "ratio_to_normal_p10": "float",
    "ratio_to_normal_p50": "float",
    "ratio_to_normal_p90": "float",
    # Decision 0004 item 54 (plan C4): peak / plugged-in charger capacity.
    "coincidence_world_count": "int",
    "coincidence_factor_p10": "float",
    "coincidence_factor_p50": "float",
    "coincidence_factor_p90": "float",
}
EVALUATION_PRICE_COLUMNS = {
    "world_id": "int",
    "slot_index": "int",
    "interval_start_utc": "utc",
    "interval_end_utc": "utc",
    # The intraday close, then the imbalance price and the NIV state that set
    # its premium (decision 0004 item 53, plan B5).
    "evaluation_context_price_gbp_per_mwh": "float",
    "imbalance_price_gbp_per_mwh": "float",
    "system_long": "bool",
    "outcome_available_at_utc": "utc",
    "evidence_kind": "str",
}
# One row per stochastic net-demand shock starting in the study (item 56).
PRICE_SHOCK_COLUMNS = {
    "world_id": "int",
    "shock_class": "str",
    "start_slot_index": "int",
    "start_utc": "utc",
    "duration_slots": "int",
    "size_gw": "float",
    "direction": "str",
    "known_day_ahead": "bool",
    "evidence_kind": "str",
}
# Zones, events and shocks (trading contract v1 §2.1, §3, §5.6-§5.8;
# decision 0004 items 55, 56 and 58).
ZONE_IDS = ("zone_1", "zone_2", "zone_3", "zone_4")
ZONE_LABELS = {"zone_1": "Zone A", "zone_2": "Zone B", "zone_3": "Zone C", "zone_4": "Zone D"}
ZONE_WORLD_COLUMNS = {
    "world_id": "int",
    "path_id": "str",
    "zone_id": "str",
    "slot_index": "int",
    "interval_start_utc": "utc",
    "interval_start_london": "london",
    "unit_count": "int",
    "connected_count": "int",
    "home_import_kwh": "float",
    "home_import_kw": "float",
    "early_departure_count": "int",
    "early_departure_shortfall_kwh": "float",
}
# Columns that add up over zones to the fleet frame's column of the same name.
ZONE_ADDITIVE_COLUMNS = (
    "unit_count",
    "connected_count",
    "home_import_kwh",
    "early_departure_count",
    "early_departure_shortfall_kwh",
)
ZONE_IMPORT_BAND_COLUMNS = {
    "zone_id": "str",
    "zone_label": "str",
    "path_id": "str",
    "slot_index": "int",
    "interval_start_utc": "utc",
    "interval_start_london": "london",
    "ev_count": "int",
    "headroom_kw": "float",
    **BAND_STAT_COLUMNS,
    "above_headroom_world_share": "float",
}
ZONE_SUMMARY_COLUMNS = {
    "zone_id": "str",
    "zone_label": "str",
    "path_id": "str",
    "share": "float",
    "ev_count": "int",
    "headroom_kw": "float",
    "peak_kw_p10": "float",
    "peak_kw_p50": "float",
    "peak_kw_p90": "float",
    "hours_above_headroom_p10": "float",
    "hours_above_headroom_p50": "float",
    "hours_above_headroom_p90": "float",
}
EVENT_TYPES = (
    "price_shock_known",
    "price_shock_surprise",
    "turn_down",
    "turn_up",
    "control_outage",
)
EVENT_TABLE_COLUMNS = {
    "event_id": "str",
    "event_type": "str",
    "enabled": "bool",
    "night_index": "int",
    "start_local_time": "str",
    "duration_minutes": "int",
    "size": "float",
    "size_unit": "str",
    "notice": "str",
    "notice_minutes": "Int64",
    "scope": "str",
    "payment_gbp_per_mwh": "float",
}
EVENT_RESPONSE_COLUMNS = {
    "event_id": "str",
    "event_type": "str",
    "scope": "str",
    "series": "str",
    "metric": "str",
    "unit": "str",
    "slot_index": "int",
    "interval_start_utc": "utc",
    "interval_start_london": "london",
    "relative_slot": "int",
    "in_window": "bool",
    **BAND_STAT_COLUMNS,
}
# ``day_ahead_plan`` only with intraday dispatch on (intraday-dispatch-v1 §7.4).
EVENT_RESPONSE_SERIES = ("normal", "selected", "day_ahead_plan", "difference", "market")
MARKET_SHOCK_COLUMNS = {
    "world_id": "int",
    "shock_id": "str",
    "source": "str",
    "shock_class": "str",
    "direction": "str",
    "known_day_ahead": "bool",
    "start_slot_index": "int",
    "start_utc": "utc",
    "start_london": "london",
    "duration_slots": "int",
    "size_gw": "float",
    "in_study": "bool",
    "peak_price_increment_gbp_per_mwh": "float",
    "evidence_kind": "str",
}
SHOCK_CLASS_ORDER = ("mild", "big", "scripted")
SHOCK_SUMMARY_COLUMNS = {
    "shock_class": "str",
    "known_day_ahead": "bool",
    "direction": "str",
    "world_count": "int",
    "count_per_week_p10": "float",
    "count_per_week_p50": "float",
    "count_per_week_p90": "float",
    "size_gw_p50": "float",
    "peak_price_increment_gbp_per_mwh_p50": "float",
}
COST_EFFECT_COLUMNS = {
    "world_id": "int",
    "normal_home_import_kwh": "float",
    "selected_home_import_kwh": "float",
    "illustrative_selected_minus_normal_energy_cost_gbp": "float",
    "selected_minus_normal_closing_battery_kwh": "float",
    "selected_minus_normal_public_import_kwh": "float",
    "selected_minus_normal_unserved_travel_kwh": "float",
    "illustrative_selected_minus_normal_public_charge_cost_gbp": "float",
    "illustrative_unrecovered_energy_value_gbp": "float",
    "illustrative_unserved_travel_value_gbp": "float",
    "illustrative_selected_minus_normal_total_gbp": "float",
    "energy_not_recovered": "bool",
    "unrecovered_kwh": "float",
    "unrecovered_share": "float",
    "not_recovered_material": "bool",
    "sessions_affected_count": "int",
    "evidence_kind": "str",
}
NOT_RECOVERED_METRICS = {
    "unrecovered_kwh": "kWh per week",
    "unrecovered_share": "fraction",
    "sessions_affected_count": "sessions per week",
}
NOT_RECOVERED_SUMMARY_COLUMNS = {"metric": "str", "unit": "str", **BAND_STAT_COLUMNS}
COST_SUMMARY_COLUMNS = {
    "component": "str",
    "source_column": "str",
    "unit": "str",
    "mean": "float",
    "p10": "float",
    "p50": "float",
    "p90": "float",
    "evidence_kind": "str",
}
# Decision 0007, model step 2: the timed path's own cost reading, present
# only when the run's fleet frames actually carry "timed" (never merely
# because this is an action result); ``None`` on every other result.
TIMED_COST_FIELDS = (
    "timed_cost_effect",
    "timed_cost_effect_summary",
    "timed_not_recovered_summary",
    "timed_not_recovered_world_count",
)
REPLAY_INTERVAL_COLUMNS = {
    "path_id": "str",
    **SLOT_COLUMNS,
    "location": "str",
    "connected": "bool",
    "battery_soc_percent": "float",
    "closing_battery_kwh": "float",
    "home_import_kwh": "float",
    "public_import_kwh": "float",
    "unserved_travel_kwh": "float",
    "early_departure_shortfall_kwh": "float",
}
DAILY_AUDIT_COLUMNS = {
    "path_id": "str",
    "local_date": "date",
    "day_label": "str",
    "drives_today": "bool",
    "departure_utc": "utc",
    "departure_london": "london",
    "target_soc_percent": "float",
    "departure_soc_percent": "float",
    "trip_energy_need_kwh": "float",
    "shortfall_kwh": "float",
    "departed_below_target": "bool",
}
REPLAY_BAND_COLUMNS = {
    "metric": "str",
    "unit": "str",
    "path_id": "str",
    **SLOT_COLUMNS,
    "world_count": "int",
    "p10": "float",
    "p50": "float",
    "p90": "float",
}
CHANGED_COLUMNS = {
    "name": "str",
    "label": "str",
    "unit": "str",
    "value_a": "object",
    "value_b": "object",
}
COMPARE_WEEKLY_COLUMNS = {
    "world_id": "int",
    "metric": "str",
    "unit": "str",
    "value_a": "float",
    "value_b": "float",
    "b_minus_a": "float",
}
COMPARE_KPI_COLUMNS = {
    "metric": "str",
    "unit": "str",
    "p10": "float",
    "p50": "float",
    "p90": "float",
}


@dataclass(frozen=True)
class Assumption:
    """One assumption record (contract section 8, decision 0004 item 21)."""

    name: str
    label: str
    value: float | int | str | tuple
    unit: str
    evidence: str
    source: str
    meaning: str
    editable: bool
    bounds: tuple[float, float] | None
    group: str
    affects: str


def quantile_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """Mean and P10/P50/P90 across axis 0 (worlds), linear method."""

    q = np.quantile(values, (0.1, 0.5, 0.9), axis=0, method="linear")
    return {"mean": values.mean(axis=0), "p10": q[0], "p50": q[1], "p90": q[2]}


def metric_matrix(world: pd.DataFrame, path: str, metric: str) -> np.ndarray:
    """(world, slot) matrix of one metric for one path of a wide world frame."""

    rows = world.loc[world["path_id"].eq(path)].sort_values(["world_id", "slot_index"])
    return rows[metric].to_numpy(dtype=float).reshape(-1, SLOT_COUNT)


def slot_keys(world: pd.DataFrame) -> pd.DataFrame:
    first = world.loc[world["world_id"].eq(world["world_id"].min()) & world["path_id"].eq("normal")]
    return first.sort_values("slot_index").loc[:, list(SLOT_COLUMNS)].reset_index(drop=True)


def fleet_bands(world: pd.DataFrame) -> pd.DataFrame:
    """Long across-world bands (contract 3.6) from a wide world frame.

    Worlds first: each world's fleet value per slot is already a sum, and only
    then are quantiles taken across worlds.  Also the reference computation
    the validator uses on real results.
    """

    keys = slot_keys(world)
    # fleet_interval_bands is one of the four frames the optional "timed"
    # path (decision 0007) may appear in, so this reads off whatever paths
    # are actually present rather than assuming exactly two.
    paths = [path for path in INTERVAL_PATH_ORDER if path in set(world["path_id"])]
    frames = []
    for metric, unit in FLEET_METRICS.items():
        for path in paths:
            values = metric_matrix(world, path, metric)
            frame = keys.copy()
            frame.insert(0, "path_id", path)
            frame.insert(0, "unit", unit)
            frame.insert(0, "metric", metric)
            frame["world_count"] = np.int64(values.shape[0])
            for stat, column in quantile_stats(values).items():
                frame[stat] = column
            frames.append(frame)
    return pd.concat(frames, ignore_index=True).loc[:, list(FLEET_BAND_COLUMNS)]


def difference_bands_from_world(world: pd.DataFrame) -> pd.DataFrame:
    """Selected minus normal per world, then quantiles (contract 4.1, item 12).

    This is deliberately not ``selected band - normal band``: the quantile of a
    difference is not the difference of quantiles.
    """

    keys = slot_keys(world)
    frames = []
    for metric, unit in FLEET_METRICS.items():
        delta = metric_matrix(world, "selected", metric) - metric_matrix(world, "normal", metric)
        frame = keys.copy()
        frame.insert(0, "unit", "percentage points" if metric == "battery_soc_percent" else unit)
        frame.insert(0, "metric", metric)
        frame["world_count"] = np.int64(delta.shape[0])
        for stat, column in quantile_stats(delta).items():
            frame[stat] = column
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).loc[:, list(DIFFERENCE_BAND_COLUMNS)]


def typed_frame(frame: pd.DataFrame, spec: dict[str, str]) -> pd.DataFrame:
    frame = frame.loc[:, list(spec)].copy()
    for name, kind in spec.items():
        if kind == "int":
            frame[name] = frame[name].astype(np.int64)
        elif kind == "float":
            frame[name] = frame[name].astype(np.float64)
        elif kind == "bool":
            frame[name] = frame[name].astype(bool)
        elif kind == "Int64":
            frame[name] = frame[name].astype("Int64")
    return frame


def cost_summary_from_cost_effect(cost: pd.DataFrame) -> pd.DataFrame:
    """Each component's quantiles across worlds, separately (contract 4.9).

    Component quantiles do not add up to the total's quantile; the view says so.
    """

    rows = []
    for component, column in COST_COMPONENTS:
        stats = quantile_stats(cost[column].to_numpy(dtype=float)[:, None])
        rows.append(
            {
                "component": component,
                "source_column": column,
                "unit": "GBP",
                **{stat: float(value[0]) for stat, value in stats.items()},
                "evidence_kind": "illustrative_synthetic",
            }
        )
    return pd.DataFrame(rows).loc[:, list(COST_SUMMARY_COLUMNS)]


def difference_weekly_from_world(world: pd.DataFrame) -> pd.DataFrame:
    """Weekly selected-minus-normal totals per world (contract 4.2)."""

    rows = []
    for metric, unit in WEEKLY_DIFFERENCE_METRICS.items():
        normal = metric_matrix(world, "normal", metric)
        selected = metric_matrix(world, "selected", metric)
        # Closing stock is a level: take the last slot, never a weekly sum.
        reduce = (lambda m: m[:, -1]) if metric == "closing_battery_kwh" else (lambda m: m.sum(1))
        for w, (n, s) in enumerate(zip(reduce(normal), reduce(selected), strict=True)):
            rows.append(
                {
                    "world_id": w,
                    "metric": metric,
                    "unit": unit,
                    "normal_value": n,
                    "selected_value": s,
                    "selected_minus_normal": s - n,
                }
            )
    return typed_frame(pd.DataFrame(rows), DIFFERENCE_WEEKLY_COLUMNS)


def weekly_difference_bands(
    weekly: pd.DataFrame, value: str = "selected_minus_normal"
) -> pd.DataFrame:
    rows = []
    for metric in dict.fromkeys(weekly["metric"]):
        subset = weekly.loc[weekly["metric"].eq(metric)]
        stats = quantile_stats(subset[value].to_numpy(dtype=float)[:, None])
        rows.append(
            {
                "metric": metric,
                "unit": subset["unit"].iat[0],
                "world_count": len(subset),
                **{stat: float(v[0]) for stat, v in stats.items()},
            }
        )
    return typed_frame(pd.DataFrame(rows), DIFFERENCE_WEEKLY_BAND_COLUMNS)


def smart_charging_world_from(world: pd.DataFrame, forecast_prices: pd.DataFrame) -> pd.DataFrame:
    """Per-world smart-charging outcomes (contract 4.4), recomputed from the fleet frame.

    Average prices weight home import by the day-ahead price (plan B3, item 53).
    """

    normal = metric_matrix(world, "normal", "home_import_kwh")
    selected = metric_matrix(world, "selected", "home_import_kwh")
    prices = forecast_prices.sort_values(["world_id", "slot_index"])
    price = prices["wholesale_forecast_gbp_per_mwh"].to_numpy(dtype=float)
    price = price.reshape(normal.shape)

    def average(imports: np.ndarray) -> np.ndarray:
        total = imports.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 0.0, (imports * price).sum(axis=1) / total, np.nan)

    moved = np.maximum(normal - selected, 0.0).sum(axis=1)
    normal_total = normal.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(normal_total > 0.0, moved / normal_total, np.nan)
    columns = {
        "world_id": np.arange(len(moved)),
        "normal_average_price_gbp_per_mwh": average(normal),
        "selected_average_price_gbp_per_mwh": average(selected),
        "moved_home_import_kwh": moved,
        "moved_home_import_share": share,
        "early_departure_count": metric_matrix(world, "selected", "early_departure_count")
        .sum(axis=1)
        .round(),
        "early_departure_shortfall_kwh": metric_matrix(
            world, "selected", "early_departure_shortfall_kwh"
        ).sum(axis=1),
        "evidence_kind": "illustrative_synthetic",
    }
    schema = dict(SMART_CHARGING_WORLD_COLUMNS)
    if "timed" in set(world["path_id"]):
        # Decision 0007, model step 2: inserted after the selected price, the
        # same position the model's own column occupies.
        columns["timed_average_price_gbp_per_mwh"] = average(
            metric_matrix(world, "timed", "home_import_kwh")
        )
        schema = {}
        for key, kind in SMART_CHARGING_WORLD_COLUMNS.items():
            schema[key] = kind
            if key == "selected_average_price_gbp_per_mwh":
                schema["timed_average_price_gbp_per_mwh"] = "float"
    return typed_frame(pd.DataFrame(columns), schema)


def not_recovered_summary_from(cost: pd.DataFrame) -> pd.DataFrame:
    """Each not-recovered magnitude across worlds, NaN worlds left out (contract 4.9a)."""

    rows = []
    for metric, unit in NOT_RECOVERED_METRICS.items():
        values = cost[metric].to_numpy(dtype=float)
        present = values[~np.isnan(values)]
        stats = (
            {k: float(v[0]) for k, v in quantile_stats(present[:, None]).items()}
            if present.size
            else dict.fromkeys(("mean", "p10", "p50", "p90"), np.nan)
        )
        rows.append({"metric": metric, "unit": unit, "world_count": present.size, **stats})
    return typed_frame(pd.DataFrame(rows), NOT_RECOVERED_SUMMARY_COLUMNS)


def smart_charging_summary_from(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Each smart-charging metric across worlds, NaN worlds left out (contract 4.5)."""

    values = {
        name: outcomes[name].to_numpy(dtype=float)
        for name in SMART_CHARGING_METRICS
        if name in outcomes
    }
    values["average_price_change_gbp_per_mwh"] = (
        values["selected_average_price_gbp_per_mwh"] - values["normal_average_price_gbp_per_mwh"]
    )
    rows = []
    for metric, unit in SMART_CHARGING_METRICS.items():
        if metric not in values:
            continue
        present = values[metric][~np.isnan(values[metric])]
        stats = (
            {k: float(v[0]) for k, v in quantile_stats(present[:, None]).items()}
            if present.size
            else dict.fromkeys(("mean", "p10", "p50", "p90"), np.nan)
        )
        rows.append({"metric": metric, "unit": unit, "world_count": present.size, **stats})
    return typed_frame(pd.DataFrame(rows), SMART_CHARGING_SUMMARY_COLUMNS)


def price_band_shift_from(
    world: pd.DataFrame, forecast_prices: pd.DataFrame, slots: pd.DataFrame
) -> pd.DataFrame:
    """Home import moved out of each day-ahead price third, per London date (contract 4.6).

    Each world is banded by its own day-ahead prices (decision 0004 item 48).
    """

    # Price band shift pairs normal and selected only (decision 0007 keeps
    # the optional "timed" path out of it), whatever else ``world`` carries.
    by_path = {
        path: metric_matrix(world, path, "home_import_kwh") for path in ("normal", "selected")
    }
    delta = by_path["normal"] - by_path["selected"]
    price = (
        forecast_prices.sort_values(["world_id", "slot_index"])["wholesale_forecast_gbp_per_mwh"]
        .to_numpy(dtype=float)
        .reshape(delta.shape)
    )
    low, high = np.quantile(price, (1 / 3, 2 / 3), axis=1, keepdims=True)
    band = np.where(price < low, "low", np.where(price > high, "high", "middle"))
    dates = slots["local_date"].to_numpy(dtype=object)
    rows = []
    for local_date in dict.fromkeys(dates):
        for band_id in PRICE_BANDS:
            mask = (dates == local_date)[None, :] & (band == band_id)
            stats = quantile_stats((delta * mask).sum(axis=1)[:, None])
            rows.append(
                {
                    "local_date": local_date,
                    "day_label": f"{local_date:%a} {local_date.day}",
                    "price_band": band_id,
                    "forecast_price_min_gbp_per_mwh": price[mask].min() if mask.any() else np.nan,
                    "forecast_price_max_gbp_per_mwh": price[mask].max() if mask.any() else np.nan,
                    "mean_slot_count": mask.sum(axis=1).mean(),
                    "unit": "kWh per day",
                    "world_count": delta.shape[0],
                    **{k: float(v[0]) for k, v in stats.items()},
                    **{
                        f"{path}_kwh_{k}": float(v[0])
                        for path, values in by_path.items()
                        for k, v in quantile_stats((values * mask).sum(axis=1)[:, None]).items()
                        if k in ("p10", "p50", "p90")
                    },
                }
            )
    return typed_frame(pd.DataFrame(rows), PRICE_BAND_SHIFT_COLUMNS)


def forecast_price_bands_from(forecast_prices: pd.DataFrame) -> pd.DataFrame:
    """Each slot's day-ahead price P10/P50/P90 across worlds (contract 4.7)."""

    rows = []
    for slot_index, group in forecast_prices.groupby("slot_index", sort=True):
        prices = group["wholesale_forecast_gbp_per_mwh"].to_numpy(dtype=float)
        p10, p50, p90 = np.quantile(prices, (0.1, 0.5, 0.9), method="linear")
        first = group.iloc[0]
        rows.append(
            {
                "slot_index": slot_index,
                "interval_start_utc": first["interval_start_utc"],
                "interval_end_utc": first["interval_end_utc"],
                "interval_start_london": first["interval_start_london"],
                "world_count": len(prices),
                "p10": p10,
                "p50": p50,
                "p90": p90,
            }
        )
    return typed_frame(pd.DataFrame(rows), FORECAST_PRICE_BAND_COLUMNS)


def weekly_peak_summary_from(world: pd.DataFrame, home_charging_power_kw: float) -> pd.DataFrame:
    """Each week's own peak fleet home import per path, then across weeks (contract 4.5a).

    Written with pandas group-bys, independently of the model's matrix code.
    The coincidence factor divides each week's peak by the normal path's
    ``connected_count`` at that peak slot times ``home_charging_power_kw``
    (decision 0004 item 54).
    """

    rows = world.sort_values(["path_id", "world_id", "slot_index"])
    # First row of each (path, world) at its maximum: the earliest peak slot.
    peaks = rows.loc[rows.groupby(["path_id", "world_id"], sort=False)["home_import_kw"].idxmax()]
    peaks = peaks.set_index(["path_id", "world_id"])
    normal = peaks.loc["normal", "home_import_kw"]
    plugged = rows.loc[rows["path_id"].eq("normal")].set_index(["world_id", "slot_index"])[
        "connected_count"
    ]
    records = []
    # One row per path ``world`` actually has: normal, selected, plus the
    # optional timed path (decision 0007, model step 2) when the run has it.
    for path in dict.fromkeys(world["path_id"]):
        path_peaks = peaks.loc[path]
        counts = path_peaks["slot_index"].value_counts()
        modal_slot = int(counts[counts == counts.max()].index.min())
        modal = rows.loc[rows["slot_index"].eq(modal_slot)].iloc[0]
        ratio = (path_peaks["home_import_kw"] / normal.loc[path_peaks.index]).where(normal > 0.0)
        ratio = ratio.dropna().to_numpy(dtype=float)
        ratio_q = (
            np.quantile(ratio, (0.1, 0.5, 0.9), method="linear") if ratio.size else [np.nan] * 3
        )
        capacity = (
            plugged.loc[
                list(zip(path_peaks.index, path_peaks["slot_index"], strict=True))
            ].to_numpy(dtype=float)
            * home_charging_power_kw
        )
        factor = pd.Series(path_peaks["home_import_kw"].to_numpy(dtype=float) / capacity)
        factor = factor.where(capacity > 0.0).dropna().to_numpy(dtype=float)
        factor_q = (
            np.quantile(factor, (0.1, 0.5, 0.9), method="linear") if factor.size else [np.nan] * 3
        )
        p10, p50, p90 = np.quantile(
            path_peaks["home_import_kw"].to_numpy(dtype=float), (0.1, 0.5, 0.9), method="linear"
        )
        records.append(
            {
                "path_id": path,
                "unit": "kW",
                "world_count": len(path_peaks),
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "modal_peak_slot_index": modal_slot,
                "modal_peak_interval_start_utc": modal["interval_start_utc"],
                "modal_peak_interval_start_london": modal["interval_start_london"],
                "modal_peak_week_count": int(counts[modal_slot]),
                "ratio_world_count": ratio.size,
                "ratio_to_normal_p10": ratio_q[0],
                "ratio_to_normal_p50": ratio_q[1],
                "ratio_to_normal_p90": ratio_q[2],
                "coincidence_world_count": factor.size,
                "coincidence_factor_p10": factor_q[0],
                "coincidence_factor_p50": factor_q[1],
                "coincidence_factor_p90": factor_q[2],
            }
        )
    return typed_frame(pd.DataFrame(records), WEEKLY_PEAK_COLUMNS)


def _half_hour_label(index: int) -> str:
    return f"{index // 2:02d}:{30 * (index % 2):02d}"


def plug_event_heatmap_from(
    world_ids: np.ndarray, when: pd.Series, slots: pd.DataFrame, evs: int, worlds: int
) -> pd.DataFrame:
    """Events per EV-day by London weekday x half-hour (contract 3.10b).

    Independent of the model: counts events (``world_ids`` and their instants
    ``when``, London time) by London weekday and wall-clock half-hour, and
    divides by EVs x the study slots in that cell.
    """

    event_cells = pd.DataFrame(
        {
            "world_id": np.asarray(world_ids, dtype=int),
            "weekday": when.dt.weekday.to_numpy(),
            "half_hour": (when.dt.hour * 2 + when.dt.minute // 30).to_numpy(),
        }
    ).value_counts()
    london = slots["interval_start_london"]
    slot_cells = pd.Series(
        list(zip(london.dt.weekday, london.dt.hour * 2 + london.dt.minute // 30, strict=True))
    ).value_counts()
    rows = []
    for weekday in range(7):
        for half_hour in range(48):
            slot_count = int(slot_cells.get((weekday, half_hour), 0))
            counts = np.array(
                [event_cells.get((w, weekday, half_hour), 0) for w in range(worlds)], dtype=float
            )
            rate = counts / (evs * slot_count) if slot_count else np.full(worlds, np.nan)
            q = np.quantile(rate, (0.1, 0.5, 0.9), method="linear")
            rows.append(
                {
                    "weekday": weekday,
                    "weekday_label": WEEKDAY_LABELS[weekday],
                    "local_half_hour": half_hour,
                    "local_time_label": _half_hour_label(half_hour),
                    "slot_count": slot_count,
                    "world_count": worlds,
                    "mean": rate.mean(),
                    "p10": q[0],
                    "p50": q[1],
                    "p90": q[2],
                }
            )
    return typed_frame(pd.DataFrame(rows), PLUG_EVENT_HEATMAP_COLUMNS)


def session_distribution_bands_from(
    events: pd.DataFrame,
    units: pd.DataFrame,
    world_count: int,
    efficiency: float,
    smart_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Share of sessions per bin, quantiles across worlds (contract 3.10c).

    Independent of the model: every metric is recomputed from the plug-in
    event rows (3.9) of worlds ``0 .. world_count - 1`` and the units frame
    (3.2); ``efficiency`` is battery kWh per grid kWh; ``smart_events`` are
    the same sessions on the selected path (their unplug SoC gives the
    smart departure metric, NaN without them).  Values are rounded
    to 1e-9 before binning, as the contract states.
    """

    unit = units.set_index("unit_id").loc[events["unit_id"]]
    capacity = unit["physical_capacity_kwh"].to_numpy(float)
    target = unit["preferred_target_soc_percent"].to_numpy(float) / 100.0 * capacity
    battery_kw = unit["home_charger_limit_kw"].to_numpy(float) * efficiency
    plug_in = events["plug_in_utc"].dt.tz_convert(LONDON)
    plug_out = events["plug_out_utc"].dt.tz_convert(LONDON)
    stock = events["plug_in_soc_percent"].to_numpy(float) / 100.0 * capacity
    need = np.round(np.maximum(target - stock, 0.0), 9)
    dwell = ((events["plug_out_utc"] - events["plug_in_utc"]).dt.total_seconds() / 3600.0).to_numpy(
        float
    )
    values = {
        "plug_in_time": (plug_in.dt.hour + plug_in.dt.minute / 60.0).to_numpy(float),
        "departure_time": (plug_out.dt.hour + plug_out.dt.minute / 60.0).to_numpy(float),
        "plug_in_soc_percent": events["plug_in_soc_percent"].to_numpy(float),
        "energy_needed_kwh": need,
        "dwell_hours": dwell,
        "flexible_kwh": np.minimum(need, dwell * battery_kw),
        "slack_hours": dwell - need / battery_kw,
        "departure_soc_percent": events["plug_out_soc_percent"].to_numpy(float),
        "departure_soc_percent_smart": np.full(len(events), np.nan)
        if smart_events is None
        else events[["world_id", "unit_id", "plug_in_utc"]]
        .merge(smart_events, on=["world_id", "unit_id", "plug_in_utc"], how="left")[
            "plug_out_soc_percent"
        ]
        .to_numpy(float),
    }
    counts = units["cohort_id"].value_counts()
    groups = ["fleet", *(c for c in COHORT_ORDER if counts.get(c, 0) > 0)]
    rows = []
    for group_id in groups:
        in_group = (
            np.ones(len(events), bool)
            if group_id == "fleet"
            else (events["cohort_id"].eq(group_id).to_numpy())
        )
        for day_type in DAY_TYPE_ORDER:
            in_day = in_group & (
                True if day_type == "all" else events["day_type"].eq(day_type).to_numpy()
            )
            for metric, (unit_name, lower, width, count) in SESSION_DISTRIBUTION_METRICS.items():
                if metric not in values:
                    # "departure_soc_percent_timed" (decision 0007, model
                    # step 2) needs the timed path's own sessions, which
                    # plug_in_events does not hold; the validator excludes it
                    # from this recomputation and checks it by count instead.
                    continue
                value = values[metric]
                keep = in_day & ~np.isnan(value)
                bins = np.clip(
                    np.floor((np.round(value[keep], 9) - lower) / width), 0, count - 1
                ).astype(int)
                matrix = np.zeros((world_count, count))
                np.add.at(matrix, (events["world_id"].to_numpy()[keep], bins), 1.0)
                totals = matrix.sum(axis=1)
                shares = matrix[totals > 0] / totals[totals > 0, None]
                for index in range(count):
                    column = shares[:, index]
                    q = (
                        np.quantile(column, (0.1, 0.5, 0.9), method="linear")
                        if len(column)
                        else np.full(3, np.nan)
                    )
                    rows.append(
                        {
                            "group_id": group_id,
                            "day_type": day_type,
                            "metric": metric,
                            "unit": unit_name,
                            "bin_index": index,
                            "bin_lower": lower + index * width,
                            "bin_upper": lower + (index + 1) * width,
                            "bin_label": "",
                            "world_count": int((totals > 0).sum()),
                            "share_mean": column.mean() if len(column) else np.nan,
                            "share_p10": q[0],
                            "share_p50": q[1],
                            "share_p90": q[2],
                        }
                    )
    return typed_frame(pd.DataFrame(rows), SESSION_DISTRIBUTION_COLUMNS)


def _whole_day_prices(forecast_prices: pd.DataFrame) -> pd.DataFrame:
    """Day-ahead prices on whole session nights, with night date, weekday and half-hour.

    A session night is London 12:00 to 12:00 (decision 0004 item 52); its
    date and weekday are the evening's.
    """

    prices = forecast_prices.copy()
    london = prices["interval_start_london"]
    prices["date"] = (london.dt.tz_localize(None) - pd.Timedelta(hours=12)).dt.date
    prices["weekday"] = [value.weekday() for value in prices["date"]]
    prices["half_hour"] = london.dt.hour * 2 + london.dt.minute // 30
    first = prices.loc[prices["world_id"].eq(prices["world_id"].min())]
    whole = []
    for day, count in first["date"].value_counts().items():
        start = (pd.Timestamp(day) + pd.Timedelta(hours=12)).tz_localize(LONDON)
        end = (pd.Timestamp(day) + pd.Timedelta(days=1, hours=12)).tz_localize(LONDON)
        if count == (end - start) / HALF_HOUR:
            whole.append(day)
    return prices.loc[prices["date"].isin(whole)]


def price_relative_bands_from(forecast_prices: pd.DataFrame) -> pd.DataFrame:
    """Day-ahead price minus its world's session-night mean, pooled by cell (contract 4.7a)."""

    prices = _whole_day_prices(forecast_prices)
    price = prices["wholesale_forecast_gbp_per_mwh"]
    daily_mean = price.groupby([prices["world_id"], prices["date"]]).transform("mean")
    prices = prices.assign(relative=price - daily_mean)
    worlds = forecast_prices["world_id"].nunique()
    rows = []
    for weekday in [*range(7), None]:
        on_day = prices if weekday is None else prices.loc[prices["weekday"].eq(weekday)]
        for half_hour in range(48):
            values = on_day.loc[on_day["half_hour"].eq(half_hour), "relative"].to_numpy(float)
            q = (
                np.quantile(values, (0.1, 0.5, 0.9), method="linear")
                if values.size
                else [np.nan] * 3
            )
            rows.append(
                {
                    "weekday": weekday,
                    "weekday_label": "All days" if weekday is None else WEEKDAY_LABELS[weekday],
                    "local_half_hour": half_hour,
                    "local_time_label": _half_hour_label(half_hour),
                    "unit": "GBP/MWh",
                    "sample_count": values.size,
                    "world_count": worlds,
                    "mean": values.mean() if values.size else np.nan,
                    "p10": q[0],
                    "p50": q[1],
                    "p90": q[2],
                }
            )
    frame = pd.DataFrame(rows)
    frame["weekday"] = frame["weekday"].astype("Int64")
    return typed_frame(frame, PRICE_RELATIVE_COLUMNS)


def cheapest_half_hour_from(forecast_prices: pd.DataFrame) -> pd.DataFrame:
    """London clock time of each world-night's cheapest day-ahead half-hour (contract 4.7b).

    Quantiles are taken on hours since the night's noon, then turned back
    into clock times, so they follow the night's order (item 52).
    """

    prices = _whole_day_prices(forecast_prices).sort_values(["world_id", "slot_index"])
    # idxmin returns the first (earliest) row of a tie.
    cheapest = prices.loc[
        prices.groupby(["world_id", "date"])["wholesale_forecast_gbp_per_mwh"].idxmin()
    ]
    hours = (cheapest["half_hour"] / 2.0).to_numpy(dtype=float)
    into_night = (hours - 12.0) % 24.0
    row: dict[str, object] = {"world_day_count": len(hours)}
    for name, level in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        hour = float((np.quantile(into_night, level, method="nearest") + 12.0) % 24.0)
        row[f"local_hour_{name}"] = hour
        row[f"local_time_label_{name}"] = _half_hour_label(int(round(hour * 2)))
    return typed_frame(pd.DataFrame([row]), CHEAPEST_HALF_HOUR_COLUMNS)


def weekly_bands_from_world(world: pd.DataFrame) -> pd.DataFrame:
    """Weekly totals per world, then stats across worlds, per path (contract 3.6a).

    Feeds the no-action Overview "Unserved travel" tile and Fleet week caption;
    ``max`` lets a caption say "0 kWh in all weeks" honestly.
    """

    rows = []
    # Weekly bands stay normal versus selected only (decision 0007 keeps the
    # optional "timed" path out of it), whatever else ``world`` carries.
    for path in [path for path in ("normal", "selected") if path in set(world["path_id"])]:
        for metric, unit in WEEKLY_BAND_METRICS.items():
            weekly = metric_matrix(world, path, metric).sum(axis=1)
            stats = quantile_stats(weekly[:, None])
            rows.append(
                {
                    "path_id": path,
                    "metric": metric,
                    "unit": unit,
                    "world_count": len(weekly),
                    **{stat: float(value[0]) for stat, value in stats.items()},
                    "max": float(weekly.max()),
                }
            )
    return typed_frame(pd.DataFrame(rows), WEEKLY_BAND_COLUMNS)


def _zone_matrix(zone_world: pd.DataFrame, zone_id: str, path: str) -> np.ndarray:
    """(world, slot) zone ``home_import_kw`` on one path."""

    rows = zone_world.loc[zone_world["zone_id"].eq(zone_id) & zone_world["path_id"].eq(path)]
    rows = rows.sort_values(["world_id", "slot_index"])
    return rows["home_import_kw"].to_numpy(dtype=float).reshape(-1, SLOT_COUNT)


def zone_import_bands_from(zone_world: pd.DataFrame, headroom_kw: dict[str, float]) -> pd.DataFrame:
    """Each zone's home import across worlds per path and slot (trading contract §5.7).

    Worlds first; ``above_headroom_world_share`` is the share of worlds above
    the zone's headroom in the slot (NaN with no headroom).
    """

    keys = (
        zone_world.loc[
            zone_world["world_id"].eq(zone_world["world_id"].min())
            & zone_world["path_id"].eq("normal")
            & zone_world["zone_id"].eq(ZONE_IDS[0])
        ]
        .sort_values("slot_index")
        .loc[:, ["slot_index", "interval_start_utc", "interval_start_london"]]
        .reset_index(drop=True)
    )
    paths = [path for path in INTERVAL_PATH_ORDER if path in set(zone_world["path_id"])]
    frames = []
    for zone_id in ZONE_IDS:
        ev_count = zone_world.loc[zone_world["zone_id"].eq(zone_id), "unit_count"].iat[0]
        headroom = float(headroom_kw[zone_id])
        for path in paths:
            values = _zone_matrix(zone_world, zone_id, path)
            frame = keys.copy()
            frame.insert(0, "path_id", path)
            frame.insert(0, "zone_label", ZONE_LABELS[zone_id])
            frame.insert(0, "zone_id", zone_id)
            frame["ev_count"] = ev_count
            frame["headroom_kw"] = headroom
            frame["world_count"] = values.shape[0]
            for stat, column in quantile_stats(values).items():
                frame[stat] = column
            frame["above_headroom_world_share"] = (
                (values > headroom).mean(axis=0) if np.isfinite(headroom) else np.nan
            )
            frames.append(frame)
    return typed_frame(pd.concat(frames, ignore_index=True), ZONE_IMPORT_BAND_COLUMNS)


def zone_summary_from(
    zone_world: pd.DataFrame, shares: dict[str, float], headroom_kw: dict[str, float]
) -> pd.DataFrame:
    """Per (zone, path): each week's own peak and hours above headroom, then quantiles (§5.8)."""

    rows = []
    for zone_id in ZONE_IDS:
        ev_count = zone_world.loc[zone_world["zone_id"].eq(zone_id), "unit_count"].iat[0]
        headroom = float(headroom_kw[zone_id])
        for path in [path for path in INTERVAL_PATH_ORDER if path in set(zone_world["path_id"])]:
            values = _zone_matrix(zone_world, zone_id, path)
            peak = np.quantile(values.max(axis=1), (0.1, 0.5, 0.9), method="linear")
            hours = (
                np.quantile((values > headroom).sum(axis=1) * 0.5, (0.1, 0.5, 0.9))
                if np.isfinite(headroom)
                else (np.nan,) * 3
            )
            rows.append(
                {
                    "zone_id": zone_id,
                    "zone_label": ZONE_LABELS[zone_id],
                    "path_id": path,
                    "share": float(shares[zone_id]),
                    "ev_count": ev_count,
                    "headroom_kw": headroom,
                    **{f"peak_kw_{q}": v for q, v in zip(("p10", "p50", "p90"), peak, strict=True)},
                    **{
                        f"hours_above_headroom_{q}": v
                        for q, v in zip(("p10", "p50", "p90"), hours, strict=True)
                    },
                }
            )
    return typed_frame(pd.DataFrame(rows), ZONE_SUMMARY_COLUMNS)


def shock_summary_from(shocks: pd.DataFrame, world_count: int) -> pd.DataFrame:
    """Study shocks per week and typical size per (class, known, direction) (§5.6a)."""

    study = shocks.loc[shocks["in_study"]]
    rows = []
    for shock_class in SHOCK_CLASS_ORDER:
        for known in (True, False):
            for direction in ("up", "down"):
                group = study.loc[
                    study["shock_class"].eq(shock_class)
                    & study["known_day_ahead"].eq(known)
                    & study["direction"].eq(direction)
                ]
                if group.empty:
                    continue
                counts = np.bincount(
                    group["world_id"].to_numpy(dtype=np.int64), minlength=world_count
                ).astype(float)
                q = np.quantile(counts, (0.1, 0.5, 0.9), method="linear")
                rows.append(
                    {
                        "shock_class": shock_class,
                        "known_day_ahead": known,
                        "direction": direction,
                        "world_count": world_count,
                        "count_per_week_p10": q[0],
                        "count_per_week_p50": q[1],
                        "count_per_week_p90": q[2],
                        "size_gw_p50": float(np.median(group["size_gw"])),
                        "peak_price_increment_gbp_per_mwh_p50": float(
                            np.median(group["peak_price_increment_gbp_per_mwh"])
                        ),
                    }
                )
    if not rows:
        return empty_frame(SHOCK_SUMMARY_COLUMNS)
    return typed_frame(pd.DataFrame(rows), SHOCK_SUMMARY_COLUMNS)


def plug_in_world_kpis_from_events(
    events: pd.DataFrame, units: pd.DataFrame, world_ids
) -> pd.DataFrame:
    """Per-world plug-in KPIs for the fleet and each sampled archetype (contract 3.10).

    ``events`` are normal-path plug-in events covering ``world_ids`` (an int
    means ``range(n)``).  Counts are 0 for a world with no plug-ins; SoC and
    hour statistics are NaN there (missing, not zero).
    """

    world_ids = list(range(world_ids)) if isinstance(world_ids, int) else list(world_ids)
    groups = {"fleet": len(units)}
    counts = units["cohort_id"].value_counts()
    groups.update({c: int(counts[c]) for c in COHORT_ORDER if counts.get(c, 0) > 0})
    metrics = {**PLUG_KPI_METRICS, **WORLD_KPI_EXTRA_METRICS}
    rows = []
    for world_id in world_ids:
        in_world = events.loc[events["world_id"].eq(world_id)]
        for group_id, ev_count in groups.items():
            group = (
                in_world
                if group_id == "fleet"
                else in_world.loc[in_world["cohort_id"].eq(group_id)]
            )
            for kind in DAY_TYPE_ORDER:
                subset = group if kind == "all" else group.loc[group["day_type"].eq(kind)]
                soc = subset["plug_in_soc_percent"]
                hours = subset["plug_in_local_hour"].value_counts()
                values = {
                    "plug_ins_per_week": float(len(subset)),
                    "plug_ins_per_ev_per_week": len(subset) / ev_count,
                    "median_plug_in_soc_percent": soc.median() if len(subset) else np.nan,
                    "share_below_10_percent_soc": soc.lt(10.0).mean() if len(subset) else np.nan,
                    # Ties go to the earliest hour so the value is reproducible.
                    "modal_plug_in_local_hour": (
                        float(hours[hours.eq(hours.max())].index.min()) if len(subset) else np.nan
                    ),
                }
                for metric, unit in metrics.items():
                    rows.append(
                        {
                            "world_id": world_id,
                            "group_id": group_id,
                            "day_type": kind,
                            "metric": metric,
                            "unit": unit,
                            "value": values[metric],
                        }
                    )
    return typed_frame(pd.DataFrame(rows), PLUG_IN_WORLD_KPI_COLUMNS)


def plug_in_kpis_from_world_kpis(world_kpis: pd.DataFrame) -> pd.DataFrame:
    """Fleet KPI quantiles across worlds; worlds with a missing value are left out."""

    rows = []
    fleet = world_kpis.loc[world_kpis["group_id"].eq("fleet")]
    for kind in DAY_TYPE_ORDER:
        for metric, unit in PLUG_KPI_METRICS.items():
            values = fleet.loc[fleet["day_type"].eq(kind) & fleet["metric"].eq(metric), "value"]
            values = values.dropna().to_numpy(dtype=float)
            q = np.quantile(values, (0.1, 0.5, 0.9)) if len(values) else [np.nan] * 3
            rows.append(
                {
                    "day_type": kind,
                    "metric": metric,
                    "unit": unit,
                    "p10": q[0],
                    "p50": q[1],
                    "p90": q[2],
                }
            )
    return typed_frame(pd.DataFrame(rows), PLUG_KPI_COLUMNS)


def cohort_plug_in_stats(world_kpis: pd.DataFrame, cohort_id: str) -> dict[str, object]:
    """The plug-in columns of one ``cohort_summary`` row, from every world's KPIs."""

    rows = world_kpis.loc[world_kpis["group_id"].eq(cohort_id) & world_kpis["day_type"].eq("all")]

    def values(metric: str) -> np.ndarray:
        return rows.loc[rows["metric"].eq(metric), "value"].dropna().to_numpy(dtype=float)

    per_ev, soc, modal = (
        values("plug_ins_per_ev_per_week"),
        values("median_plug_in_soc_percent"),
        values("modal_plug_in_local_hour"),
    )
    peak = pd.NA
    if len(modal):
        hours, counts = np.unique(modal, return_counts=True)
        peak = int(hours[counts == counts.max()].min())
    return {
        "plug_ins_per_ev_week_p50": float(np.quantile(per_ev, 0.5)) if len(per_ev) else np.nan,
        "median_plug_in_soc_percent_p50": float(np.quantile(soc, 0.5)) if len(soc) else np.nan,
        "peak_plug_in_local_hour": peak,
    }


def empty_frame(spec: dict[str, str]) -> pd.DataFrame:
    dtypes = {
        "int": np.int64,
        "float": np.float64,
        "bool": bool,
        "utc": "datetime64[ns, UTC]",
        "london": f"datetime64[ns, {LONDON}]",
        "Int64": "Int64",
    }
    return pd.DataFrame(
        {name: pd.Series(dtype=dtypes.get(kind, object)) for name, kind in spec.items()}
    )


def _check_dtypes(frame, spec: dict[str, str], name: str, *, exact: bool = True) -> None:
    assert isinstance(frame, pd.DataFrame), f"{name} must be a DataFrame"
    if exact:
        assert list(frame.columns) == list(spec), (
            f"{name} columns {list(frame.columns)} != contract {list(spec)}"
        )
    else:
        missing = [column for column in spec if column not in frame.columns]
        assert not missing, f"{name} missing columns {missing}"
    assert isinstance(frame.index, pd.RangeIndex), f"{name} must have a RangeIndex"
    for column, kind in spec.items():
        series = frame[column]
        dtype = series.dtype
        where = f"{name}.{column} ({dtype})"
        if kind == "int":
            assert dtype == np.int64, f"{where} must be int64"
        elif kind == "float":
            assert dtype == np.float64, f"{where} must be float64"
        elif kind == "bool":
            assert dtype == np.dtype(bool), f"{where} must be bool"
        elif kind == "Int64":
            assert str(dtype) == "Int64", f"{where} must be Int64"
        elif kind in ("utc", "london"):
            tz = "UTC" if kind == "utc" else LONDON
            assert isinstance(dtype, pd.DatetimeTZDtype), f"{where} must be tz-aware"
            assert str(dtype.tz) == tz and dtype.unit == "ns", f"{where} must be ns {tz}"
        elif kind == "str":
            assert dtype == np.dtype(object), f"{where} must be object"
            assert series.map(lambda v: isinstance(v, str)).all(), f"{where} must hold str"
        elif kind == "date":
            assert dtype == np.dtype(object), f"{where} must be object dates"
            assert series.map(lambda v: type(v) is date).all(), f"{where} must hold dates"
        elif kind == "object":
            assert dtype == np.dtype(object), f"{where} must be object"


def _check_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    assert not frame.duplicated(keys).any(), f"{name} keys {keys} must be unique"


def _check_ordered_quantiles(frame, low, mid, high, name) -> None:
    both = frame[[low, mid, high]].dropna()
    tol = 1e-9
    assert (both[low] <= both[mid] + tol).all(), f"{name}: {low} > {mid}"
    assert (both[mid] <= both[high] + tol).all(), f"{name}: {mid} > {high}"


def _check_slot_keys(frame, slots, name) -> None:
    merged = frame[list(SLOT_COLUMNS)].merge(
        slots[list(SLOT_COLUMNS)], on="slot_index", suffixes=("", "_s")
    )
    assert len(merged) == len(frame), f"{name} has slot_index outside 0-335"
    for key in ("interval_start_utc", "interval_end_utc", "interval_start_london"):
        assert (merged[key] == merged[f"{key}_s"]).all(), f"{name}.{key} disagrees with study_slots"


def _assert_close(actual: pd.DataFrame, expected: pd.DataFrame, columns, name) -> None:
    for column in columns:
        assert np.allclose(
            actual[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-9,
            equal_nan=True,
        ), f"{name}.{column} does not match the world-first recomputation"


def validate_result_v2(result) -> None:
    """Assert that ``result`` satisfies contract v2; raise AssertionError naming the rule.

    Duck-typed: it reads attributes and columns only, so it validates the
    fixture and real ``ForecastResult`` objects alike.
    """

    model = result.model
    assert model in ("action", "no_action"), "model must be 'action' or 'no_action'"
    is_action = model == "action"
    # Every pairwise/per-EV frame below (EV bands, flexibility, average day,
    # zones, replay, run comparison) stays this explicit two (or one) paths,
    # never picking up the optional "timed" path (decision 0007): see
    # ``interval_paths`` just below for the four frames that may show it.
    paths = ["normal", "selected"] if is_action else ["normal"]
    expected_scalars = {
        "policy_id": "explicit_synthetic_wholesale_v1" if is_action else "explicit_no_action_v1",
        "evidence_kind": "illustrative_synthetic" if is_action else "illustrative",
        "public_charging_status": "modelled",
        "study_slot_count": SLOT_COUNT,
    }
    for name, expected in expected_scalars.items():
        assert getattr(result, name) == expected, f"{name} must be {expected!r}"
    assert isinstance(result.model_label, str) and result.model_label, "model_label required"
    for name in ("seed", "vehicle_count", "world_count", "representative_world_id"):
        assert isinstance(getattr(result, name), int), f"{name} must be int"
    assert result.vehicle_count >= 1 and result.world_count >= 1, "need EVs and worlds"
    assert 0 <= result.representative_world_id < result.world_count, "representative world"
    assert isinstance(result.has_clock_change, bool), "has_clock_change must be bool"
    assert type(result.study_start_local_date) is date, "study_start_local_date must be a date"
    for name in ("run_completed_at_utc", "horizon_start_utc", "horizon_end_utc"):
        value = getattr(result, name)
        assert isinstance(value, pd.Timestamp) and str(value.tz) == "UTC", f"{name} must be UTC"
    # Decision 0004 item 52: the study runs noon to noon.
    horizon_start = (
        (pd.Timestamp(result.study_start_local_date) + pd.Timedelta(hours=12))
        .tz_localize(LONDON)
        .tz_convert("UTC")
    )
    assert result.horizon_start_utc == horizon_start, "horizon starts at London noon"
    assert result.horizon_end_utc == horizon_start + SLOT_COUNT * HALF_HOUR, "horizon is 336 slots"
    assert list(result.settings_snapshot) == list(SETTINGS_KEYS), "settings_snapshot keys"
    assert result.settings_snapshot["model"] == model, "settings_snapshot.model"
    # Smart charging applies to every EV on the selected path: there is no
    # select-or-not decision (decision 0004 item 38).
    assert result.action_status == ("selected" if is_action else "not_applicable"), "action_status"
    for removed in (
        "planning_world_count",
        "action_decision",
        "action_eligibility",
        "selected_schedule",
    ):
        assert not hasattr(result, removed), f"{removed} was removed by decision 0004 item 38"

    slots = result.study_slots
    _check_dtypes(slots, STUDY_SLOT_COLUMNS, "study_slots")
    assert list(slots["slot_index"]) == list(range(SLOT_COUNT)), "study_slots has 336 slots"
    starts = pd.date_range(horizon_start, periods=SLOT_COUNT, freq=HALF_HOUR)
    assert (slots["interval_start_utc"].to_numpy() == starts.to_numpy()).all(), "UTC slot starts"
    assert (slots["interval_end_utc"] - slots["interval_start_utc"]).eq(HALF_HOUR).all()
    assert (slots["interval_start_london"] == slots["interval_start_utc"]).all()
    offsets = {ts.utcoffset() for ts in slots["interval_start_london"]}
    assert result.has_clock_change == (len(offsets) > 1), "has_clock_change matches the slots"
    assert set(slots["day_type"]) <= {"weekday", "weekend"}, "day_type values"
    assert slots["local_half_hour"].between(0, 47).all(), "local_half_hour 0-47"
    # Session nights (item 52): London time minus 12 h, clipped to the seven
    # nights (a spring change pushes the last slots past the last noon).
    shifted = slots["interval_start_london"].dt.tz_localize(None) - pd.Timedelta(hours=12)
    first = result.study_start_local_date
    expected_night = np.clip([(value - first).days for value in shifted.dt.date], 0, STUDY_DAYS - 1)
    assert (slots["night_index"].to_numpy() == expected_night).all(), "night_index is the night"
    nights = [(value - first).days for value in slots["local_date"]]
    assert (np.asarray(nights) == expected_night).all(), "local_date is the session night"
    weekend = [value.weekday() >= 5 for value in slots["local_date"]]
    assert (slots["day_type"].eq("weekend").to_numpy() == np.asarray(weekend)).all(), "day_type"

    _validate_units_and_cohorts(result)
    _validate_assumptions(result.assumptions)

    world = result.fleet_world_intervals
    _check_dtypes(world, FLEET_WORLD_COLUMNS, "fleet_world_intervals", exact=False)
    # "timed" (decision 0007) is the one optional extra path this frame (and
    # cohort_world_intervals, fleet_interval_bands, cohort_interval_bands)
    # may carry, always last; nothing else may sneak in.
    present_paths = list(dict.fromkeys(world["path_id"]))
    extra_paths = [path for path in present_paths if path not in paths]
    assert extra_paths in ([], ["timed"]), (
        f"fleet_world_intervals paths must be {paths}, optionally plus 'timed'"
    )
    interval_paths = [*paths, *extra_paths]
    assert present_paths == interval_paths, f"paths must be {interval_paths} in order"
    assert len(world) == result.world_count * len(interval_paths) * SLOT_COUNT, (
        "world x path x slot rows"
    )
    _check_unique(world, ["world_id", "path_id", "slot_index"], "fleet_world_intervals")
    _check_slot_keys(world, slots, "fleet_world_intervals")
    for metric in SHARE_METRICS:
        assert world[metric].between(0.0, 1.0).all(), f"{metric} must be a fraction"
    assert world["battery_soc_percent"].between(0.0, 100.0).all(), "SoC must be 0-100"
    assert (world["unit_count"] == result.vehicle_count).all(), "unit_count"
    _validate_world_units(world)
    assert world["early_departure_count"].ge(0).all(), "early departures are counts"
    assert world["early_departure_shortfall_kwh"].ge(0.0).all(), "shortfall never negative"
    normal_rows = world["path_id"].eq("normal")
    assert world.loc[normal_rows, "early_departure_count"].eq(0).all(), "no plans on normal"

    weekly = result.weekly_bands
    _check_dtypes(weekly, WEEKLY_BAND_COLUMNS, "weekly_bands")
    _assert_close(
        weekly, weekly_bands_from_world(world), ["mean", "p10", "p50", "p90", "max"], "weekly_bands"
    )

    bands = result.fleet_interval_bands
    _check_dtypes(bands, FLEET_BAND_COLUMNS, "fleet_interval_bands")
    _check_ordered_quantiles(bands, "p10", "p50", "p90", "fleet_interval_bands")
    _assert_close(bands, fleet_bands(world), ["mean", "p10", "p50", "p90"], "fleet_interval_bands")
    assert (bands["world_count"] == result.world_count).all(), "band world_count"

    cohort_bands = result.cohort_interval_bands
    _check_dtypes(cohort_bands, COHORT_BAND_COLUMNS, "cohort_interval_bands")
    sampled = result.cohort_summary.loc[result.cohort_summary["ev_count"] > 0, "cohort_id"]
    assert list(dict.fromkeys(cohort_bands["cohort_id"])) == list(sampled), "cohort bands"
    _check_ordered_quantiles(cohort_bands, "p10", "p50", "p90", "cohort_interval_bands")
    # Means are linear, so for additive metrics the archetype means add up to
    # the fleet mean (percentiles do not, and are never summed).
    for metric in ADDITIVE_METRICS:
        cohort_sum = (
            cohort_bands.loc[cohort_bands["metric"].eq(metric)]
            .groupby(["path_id", "slot_index"])["mean"]
            .sum()
        )
        fleet = bands.loc[bands["metric"].eq(metric)].set_index(["path_id", "slot_index"])["mean"]
        assert np.allclose(cohort_sum.loc[fleet.index], fleet, atol=1e-6), (
            f"cohort_interval_bands.{metric} means must add up to the fleet mean"
        )

    _validate_average_day(result, paths, list(sampled))
    _validate_ev_bands(result, paths, slots)
    _validate_flexibility(result, paths, slots, world)
    _validate_deferrable(result, slots, world)
    _validate_plug_ins(result)
    _validate_zones(result, world)
    _validate_firm_mw(result, world)

    action_names = (
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
        "events",
        "event_response_bands",
        "market_shocks",
        "shock_summary",
    )
    validate_household_v2(result)  # household contract v1 §8
    from fixtures.supplier_contract import validate_supplier_frames  # imports this module

    validate_supplier_frames(result)  # supplier contract v1 §9 (lane S1)
    # Intraday dispatch v1 §8 (imported here: dispatch_contract imports this module).
    from .dispatch_contract import validate_dispatch_frames

    validate_dispatch_frames(result)
    if not is_action:
        for name in (
            *action_names,
            *TRADING_FIELDS,
            *TIMED_COST_FIELDS,
            "control_group_summary",
            "replay_week",
        ):
            assert getattr(result, name, None) is None, f"{name} must be None on a no-action result"
        assert result.not_recovered_world_count is None, "not_recovered_world_count is None"
        return
    for name in action_names:
        assert getattr(result, name) is not None, f"{name} required on an action result"
    _validate_action(result, world, slots)
    _validate_events_and_shocks(result, slots)
    # Replay contract v1 §4 (imported here: replay_contract imports this module).
    assert getattr(result, "replay_week", None) is not None, "replay_week required on an action"
    from .replay_contract import validate_replay_week_v2

    validate_replay_week_v2(result)


def _validate_zones(result, world) -> None:
    """Zone frames exist on both models (trading contract §3, §5.7-§5.9)."""

    assert tuple(result.zone_ids) == ZONE_IDS, f"zone_ids must be {ZONE_IDS}"
    units = result.units
    assert units["zone_id"].isin(ZONE_IDS).all(), "units.zone_id must be a zone id"
    zone_world = result.zone_world_intervals
    _check_dtypes(zone_world, ZONE_WORLD_COLUMNS, "zone_world_intervals")
    _check_unique(
        zone_world, ["world_id", "path_id", "zone_id", "slot_index"], "zone_world_intervals"
    )
    # Zones match the fleet's own path set, including the optional "timed"
    # path when the run has it (decision 0007, model step 2: the kernel now
    # tracks the timed path's zone sums too).
    paths = list(dict.fromkeys(world["path_id"]))
    assert list(dict.fromkeys(zone_world["path_id"])) == paths, "zone paths match the fleet"
    fleet_zoned = world.loc[world["path_id"].isin(paths)]
    assert len(zone_world) == len(fleet_zoned) * len(ZONE_IDS), "world x path x zone x slot rows"
    # Zones partition the fleet, so every additive column sums to the fleet.
    keys = ["world_id", "path_id", "slot_index"]
    summed = zone_world.groupby(keys)[list(ZONE_ADDITIVE_COLUMNS)].sum().reset_index()
    merged = summed.merge(
        fleet_zoned[[*keys, *ZONE_ADDITIVE_COLUMNS]], on=keys, suffixes=("", "_f")
    )
    assert len(merged) == len(fleet_zoned), "zone sums cover every fleet row"
    for column in ZONE_ADDITIVE_COLUMNS:
        assert np.allclose(merged[column], merged[f"{column}_f"], rtol=0.0, atol=1e-6), (
            f"zone_world_intervals.{column} must add up to fleet_world_intervals"
        )
    assert np.allclose(zone_world["home_import_kw"], zone_world["home_import_kwh"] / 0.5), (
        "zone home_import_kw = kWh / 0.5 h"
    )
    counts = units["zone_id"].value_counts()
    for zone_id in ZONE_IDS:
        in_zone = zone_world.loc[zone_world["zone_id"].eq(zone_id), "unit_count"]
        assert (in_zone == counts.get(zone_id, 0)).all(), f"{zone_id} unit_count matches units"

    bands = result.zone_import_bands
    _check_dtypes(bands, ZONE_IMPORT_BAND_COLUMNS, "zone_import_bands")
    _check_unique(bands, ["zone_id", "path_id", "slot_index"], "zone_import_bands")
    _check_ordered_quantiles(bands, "p10", "p50", "p90", "zone_import_bands")
    headroom = bands.groupby("zone_id", sort=False)["headroom_kw"].first().to_dict()
    expected = zone_import_bands_from(zone_world, headroom)
    _assert_close(
        bands,
        expected,
        ["ev_count", "mean", "p10", "p50", "p90", "above_headroom_world_share"],
        "zone_import_bands",
    )
    first = bands.drop_duplicates("zone_id")
    assert first["ev_count"].sum() == result.vehicle_count, "zone ev_count sums to vehicle_count"
    # Means are linear, so zone means add up to the fleet mean (quantiles do not).
    zone_sum = bands.groupby(["path_id", "slot_index"])["mean"].sum()
    fleet = result.fleet_interval_bands
    fleet_mean = fleet.loc[
        fleet["metric"].eq("home_import_kw") & fleet["path_id"].isin(paths)
    ].set_index(["path_id", "slot_index"])["mean"]
    assert np.allclose(zone_sum.loc[fleet_mean.index], fleet_mean, atol=1e-6), (
        "zone_import_bands means must add up to the fleet home_import_kw mean"
    )

    summary = result.zone_summary
    _check_dtypes(summary, ZONE_SUMMARY_COLUMNS, "zone_summary")
    _check_unique(summary, ["zone_id", "path_id"], "zone_summary")
    shares = summary.groupby("zone_id", sort=False)["share"].first().to_dict()
    assert np.isclose(sum(shares.values()), 1.0, atol=1e-4), "zone shares sum to 1"
    _assert_close(
        summary,
        zone_summary_from(zone_world, shares, headroom),
        [
            "ev_count",
            "headroom_kw",
            *(f"peak_kw_{q}" for q in ("p10", "p50", "p90")),
            *(f"hours_above_headroom_{q}" for q in ("p10", "p50", "p90")),
        ],
        "zone_summary",
    )
    _check_ordered_quantiles(summary, "peak_kw_p10", "peak_kw_p50", "peak_kw_p90", "zone_summary")


def _validate_events_and_shocks(result, slots) -> None:
    """Action-only event and shock frames (trading contract §2.1, §5.6, §5.6a)."""

    events = result.events
    _check_dtypes(events, EVENT_TABLE_COLUMNS, "events")
    _check_unique(events, ["event_id"], "events")
    assert events["event_type"].isin(EVENT_TYPES).all(), "events.event_type"
    enabled = events.loc[events["enabled"]]

    bands = result.event_response_bands
    _check_dtypes(bands, EVENT_RESPONSE_COLUMNS, "event_response_bands")
    _check_unique(bands, ["event_id", "series", "metric", "slot_index"], "event_response_bands")
    _check_ordered_quantiles(bands, "p10", "p50", "p90", "event_response_bands")
    assert list(dict.fromkeys(bands["event_id"])) == list(enabled["event_id"]), (
        "event_response_bands has one block per enabled event, in table order"
    )
    assert bands["series"].isin(EVENT_RESPONSE_SERIES).all(), "event_response_bands.series"
    if not bands.empty:
        _check_slot_keys(
            bands.assign(interval_end_utc=bands["interval_start_utc"] + HALF_HOUR),
            slots,
            "event_response_bands",
        )
    window = (enabled.set_index("event_id")["duration_minutes"] // 30).to_dict()
    length = bands["event_id"].map(window)
    expected_in = (bands["relative_slot"] >= 0) & (bands["relative_slot"] < length)
    assert (bands["in_window"] == expected_in).all(), (
        "event_response_bands.in_window must be 0 <= relative_slot < window length"
    )
    for _, block in bands.groupby(["event_id", "series", "metric"], sort=False):
        steps = np.diff(block["slot_index"].to_numpy())
        rel = np.diff(block["relative_slot"].to_numpy())
        assert (steps == 1).all() and (rel == 1).all(), "event blocks are consecutive slots"

    shocks = result.market_shocks
    _check_dtypes(shocks, MARKET_SHOCK_COLUMNS, "market_shocks")
    _check_unique(shocks, ["world_id", "shock_id"], "market_shocks")
    assert (shocks["size_gw"] > 0).all(), "market_shocks.size_gw > 0"
    assert shocks["direction"].isin(("up", "down")).all(), "market_shocks.direction"
    assert (shocks["peak_price_increment_gbp_per_mwh"] >= 0).all(), "peak increment >= 0"
    assert shocks["evidence_kind"].eq("synthetic").all(), "market_shocks are synthetic"
    scripted = shocks["source"].eq("scripted")
    assert shocks["source"].isin(("stochastic", "scripted")).all(), "market_shocks.source"
    assert (shocks["shock_class"].eq("scripted") == scripted).all(), "scripted class iff source"
    assert shocks.loc[~scripted, "shock_class"].isin(("mild", "big")).all(), "stochastic class"
    order = shocks[["world_id", "start_slot_index"]].to_numpy()
    assert (np.diff(order[:, 0]) >= 0).all(), "market_shocks ordered by world"
    same_world = np.diff(order[:, 0]) == 0
    assert (np.diff(order[:, 1])[same_world] >= 0).all(), "then by start slot"
    start, duration = shocks["start_slot_index"], shocks["duration_slots"]
    assert (shocks["in_study"] == ((start < SLOT_COUNT) & (start + duration > 0))).all(), (
        "market_shocks.in_study means it overlaps a study slot"
    )
    assert (shocks["start_london"] == shocks["start_utc"]).all(), "start_london is start_utc"
    # Stochastic shocks starting in the study are exactly the price_shocks rows.
    columns = ["world_id", "start_slot_index", "duration_slots", "size_gw", "direction"]
    stochastic = shocks.loc[
        ~scripted & start.between(0, SLOT_COUNT - 1), [*columns, "known_day_ahead"]
    ]
    reference = result.price_shocks.loc[:, [*columns, "known_day_ahead"]]
    pd.testing.assert_frame_equal(
        stochastic.reset_index(drop=True), reference.reset_index(drop=True), check_dtype=False
    )
    # One scripted row per enabled scripted price shock per world.
    price_events = enabled.loc[enabled["event_type"].str.startswith("price_shock")]
    per_world = shocks.loc[scripted].groupby("world_id")["shock_id"].apply(sorted)
    for world_id in range(result.world_count):
        got = per_world.get(world_id, [])
        assert got == sorted(price_events["event_id"]), (
            f"market_shocks: world {world_id} needs one row per scripted price shock"
        )

    summary = result.shock_summary
    _check_dtypes(summary, SHOCK_SUMMARY_COLUMNS, "shock_summary")
    assert len(summary) == len(shock_summary_from(shocks, result.world_count)), (
        "shock_summary has one row per group present"
    )
    _assert_close(
        summary,
        shock_summary_from(shocks, result.world_count),
        [c for c, kind in SHOCK_SUMMARY_COLUMNS.items() if kind in ("float", "int")],
        "shock_summary",
    )
    expected_keys = shock_summary_from(shocks, result.world_count)[
        ["shock_class", "known_day_ahead", "direction"]
    ]
    pd.testing.assert_frame_equal(
        summary[["shock_class", "known_day_ahead", "direction"]].reset_index(drop=True),
        expected_keys.reset_index(drop=True),
    )


def _validate_ev_bands(result, paths, slots) -> None:
    """Contract 3.6b: across-EV quantiles per slot, median across weeks."""

    bands = result.fleet_interval_ev_bands
    _check_dtypes(bands, FLEET_EV_BAND_COLUMNS, "fleet_interval_ev_bands")
    expected = [(m, p) for m in EV_BAND_METRICS for p in paths]
    got = list(dict.fromkeys(zip(bands["metric"], bands["path_id"], strict=True)))
    assert got == expected, "fleet_interval_ev_bands metric/path order"
    assert len(bands) == len(expected) * SLOT_COUNT, "one row per metric, path and slot"
    _check_slot_keys(bands, slots, "fleet_interval_ev_bands")
    assert (bands["unit"] == bands["metric"].map(EV_BAND_METRICS)).all(), "ev band units"
    assert (bands["world_count"] == result.world_count).all(), "ev band world_count"
    # The median across weeks keeps order, so the quantiles stay ordered.
    _check_ordered_quantiles(bands, "p10", "p50", "p90", "fleet_interval_ev_bands")
    soc = bands.loc[bands["metric"].eq("battery_soc_percent")]
    assert soc[["p10", "p90"]].stack().between(0.0, 100.0).all(), "per-EV SoC 0-100"
    power = bands.loc[bands["metric"].eq("home_import_kw")]
    limit = result.units["home_charger_limit_kw"].max()
    assert power[["p10", "p90"]].stack().between(0.0, limit + 1e-9).all(), "per-EV kW"


def _validate_flexibility(result, paths, slots, world) -> None:
    """Contract 3.6c: flexibility per slot (decision 0004 item 43)."""

    bands = result.flexibility_bands
    if bands is None:
        # Only a direct no-action run given no departure margin has none.
        assert result.model == "no_action", "flexibility_bands required on an action result"
        return
    _check_dtypes(bands, FLEXIBILITY_BAND_COLUMNS, "flexibility_bands")
    expected = [(m, u, sp, p) for m, u, sp in FLEXIBILITY_ROWS for p in paths]
    columns = ["metric", "unit", "spread", "path_id"]
    got = list(dict.fromkeys(bands[columns].itertuples(index=False, name=None)))
    assert got == expected, "flexibility_bands metric/unit/spread/path order"
    assert len(bands) == len(expected) * SLOT_COUNT, "one row per group, path and slot"
    _check_slot_keys(bands, slots, "flexibility_bands")
    _check_ordered_quantiles(bands, "p10", "p50", "p90", "flexibility_bands")
    assert bands["world_count"].between(0, result.world_count).all(), "world_count range"
    # Connection is the same on both paths, so the normal path's most-plugged
    # world bounds every path's flexible power and marks slots with no EV.
    connected = (
        world.loc[world["path_id"].eq("normal")].groupby("slot_index")["connected_count"].max()
    )
    limit = result.units["home_charger_limit_kw"].max()
    for metric in ("flexible_power_kw", "turn_up_headroom_kw", "movable_energy_kwh"):
        rows = bands.loc[bands["metric"].eq(metric)]
        assert (rows["world_count"] == result.world_count).all(), f"{metric} world_count"
        plugged = connected.loc[rows["slot_index"]].to_numpy()
        values = rows[["p10", "p50", "p90"]].to_numpy()
        # Headroom = ceiling minus import, and import never exceeds the ceiling.
        assert np.isfinite(values).all() and (values >= -1e-9).all(), f"{metric} >= 0"
        assert (np.abs(values[plugged == 0]) <= 1e-9).all(), f"{metric} is 0 with none plugged in"
        if metric != "movable_energy_kwh":
            ceiling = plugged[:, np.newaxis] * limit + 1e-9
            assert (values <= ceiling).all(), f"{metric} <= plugged-in EVs x charger power"
    slack = bands.loc[bands["metric"].eq("time_slack_hours")]
    missing = slack[["p10", "p50", "p90"]].isna()
    assert (missing.all(axis=1) == slack["world_count"].eq(0)).all(), "slack NaN iff no worlds"
    assert missing.all(axis=1).eq(missing.any(axis=1)).all(), "slack quantiles NaN together"


def _one_charging_power(result) -> float:
    power = result.units["home_charger_limit_kw"].unique()
    assert len(power) == 1, "one home charging power for every EV (decision 0004 item 34)"
    return float(power[0])


def _validate_deferrable(result, slots, world) -> None:
    """Contract 3.6d and 3.6e: deferrable power by slack and weekly flexibility figures."""

    bands = result.deferrable_power_bands
    weekly = result.flexibility_weekly_summary
    if result.flexibility_bands is None:
        assert bands is None and weekly is None, "no flexibility, no deferrable power"
        return
    _check_dtypes(bands, DEFERRABLE_POWER_BAND_COLUMNS, "deferrable_power_bands")
    got = list(dict.fromkeys(bands["slack_bucket"]))
    assert got == list(DEFERRABLE_BUCKET_ORDER), "deferrable_power_bands bucket order"
    assert len(bands) == len(DEFERRABLE_BUCKET_ORDER) * SLOT_COUNT, "one row per bucket and slot"
    assert bands["unit"].eq("kW").all() and bands["path_id"].eq("normal").all(), "kW, unmanaged"
    assert bands["world_count"].eq(result.world_count).all(), "deferrable world_count"
    _check_slot_keys(bands, slots, "deferrable_power_bands")
    _check_ordered_quantiles(bands, "p10", "p50", "p90", "deferrable_power_bands")
    stats = ["p10", "p50", "p90"]
    total = bands.loc[bands["slack_bucket"].eq("total"), stats].to_numpy()
    flexible = result.flexibility_bands
    flexible = flexible.loc[
        flexible["metric"].eq("flexible_power_kw") & flexible["path_id"].eq("normal"), stats
    ].to_numpy()
    # Each world's buckets add up to its flexible power, so the total rows
    # are the same per-world values and have the same quantiles.
    assert np.allclose(total, flexible, rtol=0.0, atol=1e-9), "deferrable total = flexible power"
    assert (bands[stats].to_numpy() >= 0.0).all(), "deferrable kW >= 0"
    rows = {
        bucket: bands.loc[bands["slack_bucket"].eq(bucket), stats].to_numpy()
        for bucket in DEFERRABLE_BUCKET_ORDER
    }
    for bucket in DEFERRABLE_BUCKET_ORDER[:-1]:
        # Per world a bucket never exceeds the total, so neither does any quantile.
        assert (rows[bucket] <= total + 1e-9).all(), f"deferrable {bucket} <= total"
    # Per world the cumulative rows shrink as the floor rises, and "at least
    # 8 h" is the 8-hour bucket; quantiles keep both relations.
    for low, high in zip(AT_LEAST_HOURS, AT_LEAST_HOURS[1:], strict=False):
        assert (rows[f"at_least_{high}h"] <= rows[f"at_least_{low}h"] + 1e-9).all(), (
            f"deferrable at_least_{high}h <= at_least_{low}h"
        )
    assert np.allclose(rows["at_least_8h"], rows["8h_or_more"], atol=1e-9), "deferrable 8 h rows"
    assert (rows["under_1h"] <= total + 1e-9).all()

    _check_dtypes(weekly, FLEXIBILITY_WEEKLY_COLUMNS, "flexibility_weekly_summary")
    assert len(weekly) == 1 and weekly["path_id"].iat[0] == "normal", "one unmanaged row"
    assert weekly["world_count"].iat[0] == result.world_count, "weekly flexibility world_count"
    capacity = result.units["home_charger_limit_kw"].sum()
    assert np.isclose(weekly["fleet_charger_capacity_kw"].iat[0], capacity), "fleet capacity"
    prefixes = {
        column.rsplit("_", 1)[0] for column in FLEXIBILITY_WEEKLY_COLUMNS if column[-3:] in _Q
    }
    for name in prefixes:
        _check_ordered_quantiles(weekly, f"{name}_p10", f"{name}_p50", f"{name}_p90", name)
        values = weekly[[f"{name}_{q}" for q in _Q]].to_numpy()
        assert (values[~np.isnan(values)] >= 0.0).all(), f"{name} >= 0"
    peak = weekly[[f"peak_deferrable_kw_{q}" for q in stats]].to_numpy()[0]
    # Per world the peak is at least every slot's value, so each quantile of
    # the peak is at least that quantile of every slot.
    assert (peak >= total.max(axis=0) - 1e-9).all(), "peak >= every slot's deferrable power"
    assert (peak <= capacity + 1e-9).all(), "peak <= fleet charger capacity"
    hours = weekly[[f"hours_at_least_quarter_capacity_{q}" for q in stats]].to_numpy()
    assert (hours <= SLOT_COUNT * 0.5).all(), "hours within the week"
    limit = 1000.0 * result.units["home_charger_limit_kw"].max() + 1e-6
    for time in REPORT_HALF_HOURS:
        watts = weekly[[f"w_per_plugged_in_ev_{time}_{q}" for q in stats]].to_numpy()
        assert (watts[~np.isnan(watts)] <= limit).all(), "W per plugged-in EV <= charger power"
        kw = weekly[[f"deferrable_kw_{time}_{q}" for q in stats]].to_numpy()
        assert (kw <= capacity + 1e-9).all(), f"deferrable kW at {time} <= fleet capacity"
    assert weekly["peak_modal_week_count"].between(1, result.world_count).all(), "modal count"
    modal = weekly.rename(
        columns={
            "peak_modal_slot_index": "slot_index",
            "peak_modal_interval_start_utc": "interval_start_utc",
            "peak_modal_interval_start_london": "interval_start_london",
        }
    )
    modal["interval_end_utc"] = modal["interval_start_utc"] + HALF_HOUR
    _check_slot_keys(modal, slots, "flexibility_weekly_summary modal slot")


def _validate_plug_event_heatmaps(result) -> None:
    """Contract 3.10b: plug-in and plug-out per EV-day by weekday x half-hour (item 54)."""

    london = result.study_slots["interval_start_london"]
    cells = pd.Series(
        list(zip(london.dt.weekday, london.dt.hour * 2 + london.dt.minute // 30, strict=True))
    ).value_counts()
    totals = {}
    for event in ("plug_in", "plug_out"):
        name = "plug_in_half_hour_heatmap" if event == "plug_in" else "plug_out_heatmap"
        heatmap = getattr(result, name)
        _check_dtypes(heatmap, PLUG_EVENT_HEATMAP_COLUMNS, name)
        assert heatmap["weekday"].tolist() == [d for d in range(7) for _ in range(48)], name
        assert heatmap["local_half_hour"].tolist() == list(range(48)) * 7, name
        labels = heatmap["weekday"].map(dict(enumerate(WEEKDAY_LABELS)))
        assert (heatmap["weekday_label"] == labels).all(), f"{name} weekday labels"
        expected_slots = [cells.get((d, h), 0) for d in range(7) for h in range(48)]
        assert heatmap["slot_count"].tolist() == expected_slots, f"{name} slot_count"
        assert heatmap["world_count"].eq(result.world_count).all(), f"{name} world_count"
        _check_ordered_quantiles(heatmap, "p10", "p50", "p90", name)
        present = heatmap["slot_count"] > 0
        stats = heatmap[["mean", "p10", "p50", "p90"]]
        assert (stats.loc[present].to_numpy() >= 0.0).all(), f"{name} rate >= 0"
        assert stats.loc[~present].isna().all().all(), f"{name} NaN with no study slot"
        totals[event] = (heatmap.loc[present, "mean"] * heatmap.loc[present, "slot_count"]).sum()
    events = result.plug_in_events
    all_sampled = set(result.sampled_world_ids) == set(range(result.world_count))
    if all_sampled:
        # Every world's plug-ins are in plug_in_events: recompute exactly.
        _assert_close(
            result.plug_in_half_hour_heatmap,
            plug_event_heatmap_from(
                events["world_id"].to_numpy(),
                events["plug_in_london"],
                result.study_slots,
                result.vehicle_count,
                result.world_count,
            ),
            ["mean", "p10", "p50", "p90"],
            "plug_in_half_hour_heatmap",
        )
    kpis = result.plug_in_world_kpis
    per_ev = kpis.loc[
        kpis["group_id"].eq("fleet")
        & kpis["day_type"].eq("all")
        & kpis["metric"].eq("plug_ins_per_ev_per_week"),
        "value",
    ]
    assert np.isclose(totals["plug_in"], per_ev.mean(), rtol=0.0, atol=1e-9), (
        "plug_in_half_hour_heatmap adds up to plug-ins per EV"
    )
    # Plug-outs close a session that plugged in inside the study, or the one
    # session per EV already running at the horizon start.
    assert totals["plug_out"] <= totals["plug_in"] + 1.0 + 1e-9, "plug_out_heatmap too many"
    if all_sampled:
        closed = events["plug_out_utc"].notna().sum() / (result.world_count * result.vehicle_count)
        assert totals["plug_out"] >= closed - 1e-9, "plug_out_heatmap misses closed sessions"


def _validate_session_distributions(result) -> None:
    """Contract 3.10c: session distributions per archetype (decision 0004 item 54, plan D-2)."""

    name = "session_distribution_bands"
    bands = result.session_distribution_bands
    _check_dtypes(bands, SESSION_DISTRIBUTION_COLUMNS, name)
    sampled = result.cohort_summary.loc[result.cohort_summary["ev_count"] > 0, "cohort_id"]
    groups = ["fleet", *sampled]
    # "departure_soc_percent_timed" (decision 0007, model step 2) is a row,
    # not a column: present only on a run whose fleet frames carry the timed
    # path, so the expected metric set is read off what the frame actually
    # has, in the contract's own dict order, rather than assumed fixed.
    present_metrics = [m for m in SESSION_DISTRIBUTION_METRICS if m in set(bands["metric"])]
    keys = [
        (g, d, m, i)
        for g in groups
        for d in DAY_TYPE_ORDER
        for m in present_metrics
        for i in range(SESSION_DISTRIBUTION_METRICS[m][3])
    ]
    got = list(zip(bands["group_id"], bands["day_type"], bands["metric"], bands["bin_index"]))
    assert got == keys, f"{name} rows must be group x day type x metric x bin in contract order"
    spec = bands["metric"].map(SESSION_DISTRIBUTION_METRICS)
    assert (bands["unit"] == spec.str[0]).all(), f"{name} units"
    lower = spec.str[1] + spec.str[2] * bands["bin_index"]
    assert np.allclose(bands["bin_lower"], lower) and np.allclose(
        bands["bin_upper"], lower + spec.str[2]
    ), f"{name} bin edges"
    assert bands["bin_label"].str.len().gt(0).all(), f"{name} bin labels"
    assert bands["world_count"].between(0, result.world_count).all(), f"{name} world_count"
    _check_ordered_quantiles(bands, "share_p10", "share_p50", "share_p90", name)
    shares = bands[["share_mean", "share_p10", "share_p50", "share_p90"]]
    present = bands["world_count"] > 0
    assert shares.loc[present].stack().between(0.0, 1.0).all(), f"{name} shares must be 0-1"
    assert shares.loc[~present].isna().all().all(), f"{name} NaN where no world has a session"
    totals = bands.loc[present].groupby(["group_id", "day_type", "metric"])["share_mean"].sum()
    assert np.allclose(totals, 1.0), f"{name} share_mean must sum to 1 over a metric's bins"
    # Every observed plug-in has a plug-in time, SoC and need, so those
    # metrics are present in exactly the worlds with plug-ins.
    fleet_all = bands.loc[bands["group_id"].eq("fleet") & bands["day_type"].eq("all")]
    kpis = result.plug_in_world_kpis
    with_plug_ins = kpis.loc[
        kpis["group_id"].eq("fleet")
        & kpis["day_type"].eq("all")
        & kpis["metric"].eq("plug_ins_per_week")
        & kpis["value"].gt(0)
    ]
    for metric in ("plug_in_time", "plug_in_soc_percent", "energy_needed_kwh"):
        counts = fleet_all.loc[fleet_all["metric"].eq(metric), "world_count"]
        assert counts.eq(len(with_plug_ins)).all(), f"{name}.{metric} world_count"
    events = result.plug_in_events
    if set(result.sampled_world_ids) == set(range(result.world_count)):
        # Every world's sessions are in plug_in_events: recompute exactly.
        # battery_added_kwh = home import x efficiency (3.9), so the ratio
        # gives the efficiency; rounded, since summed float ratios drift.
        charged = events["home_import_kwh"] > 1e-9
        efficiency = (
            round(
                float((events["battery_added_kwh"] / events["home_import_kwh"])[charged].median()),
                6,
            )
            if charged.any()
            else 1.0
        )
        # The smart and timed departure SoC need the selected/timed path's
        # sessions, which plug_in_events does not hold; both are checked by
        # count below instead.  The reference never computes a "timed" row
        # at all (it has no timed sessions to read), so each side excludes
        # what its own `metric` column actually carries; what remains lines
        # up row for row because both iterate the same contract dict order.
        reference = session_distribution_bands_from(
            events, result.units, result.world_count, efficiency
        )
        excluded_metrics = ("departure_soc_percent_smart", "departure_soc_percent_timed")
        normal_only = (~bands["metric"].isin(excluded_metrics)).to_numpy()
        reference_only = (~reference["metric"].isin(excluded_metrics)).to_numpy()
        _assert_close(
            bands.loc[normal_only],
            reference.loc[reference_only],
            ["world_count", "share_mean", "share_p10", "share_p50", "share_p90"],
            name,
        )
    # §4.6: the smart metric counts the same closed sessions as the normal one
    # (plug-in and unplug are exogenous); a no-action result has no smart path.
    smart = bands.loc[bands["metric"].eq("departure_soc_percent_smart"), "world_count"]
    normal = bands.loc[bands["metric"].eq("departure_soc_percent"), "world_count"]
    if result.model == "action":
        assert np.array_equal(smart.to_numpy(), normal.to_numpy()), f"{name} smart SoC sessions"
    else:
        assert (smart == 0).all(), f"{name} no smart departure SoC without a smart path"
    # Decision 0007, model step 2: the timed metric is present as a whole row
    # set exactly when the result's fleet frames carry the timed path, and,
    # when present, counts the same closed sessions as the normal metric
    # (plug-in and unplug are exogenous to every path).
    timed_active = "timed" in set(result.fleet_world_intervals["path_id"])
    timed = bands.loc[bands["metric"].eq("departure_soc_percent_timed"), "world_count"]
    if timed_active:
        assert len(timed), f"{name} missing departure_soc_percent_timed with the timed path"
        assert np.array_equal(timed.to_numpy(), normal.to_numpy()), f"{name} timed SoC sessions"
    else:
        assert not len(timed), f"{name} departure_soc_percent_timed without the timed path"


def _validate_plug_in_heatmap(result) -> None:
    """Contract 3.10a: plug-in chance per weekday x London hour (decision 0004 item 43)."""

    heatmap = result.plug_in_heatmap
    _check_dtypes(heatmap, PLUG_IN_HEATMAP_COLUMNS, "plug_in_heatmap")
    assert heatmap["weekday"].tolist() == [d for d in range(7) for _ in range(24)], "weekdays"
    assert heatmap["local_hour"].tolist() == list(range(24)) * 7, "hours"
    labels = heatmap["weekday"].map(dict(enumerate(WEEKDAY_LABELS)))
    assert (heatmap["weekday_label"] == labels).all(), "weekday labels"
    assert (heatmap["world_count"] == result.world_count).all(), "heatmap world_count"
    days = heatmap.drop_duplicates("weekday")["day_count"]
    assert days.sum() == len(set(result.study_slots["local_date"])), "day_count covers the study"
    _check_ordered_quantiles(heatmap, "p10", "p50", "p90", "plug_in_heatmap")
    present = heatmap["day_count"] > 0
    values = heatmap.loc[present, ["mean", "p10", "p50", "p90"]]
    assert np.isfinite(values.to_numpy()).all() and (values >= 0.0).all().all(), "chance >= 0"
    assert heatmap.loc[~present, ["mean", "p10", "p50", "p90"]].isna().all().all(), "no date: NaN"
    # Each world's cells times their EV-days add up to its plug-ins, so the
    # means add up to the mean plug-ins per EV per week.
    kpis = result.plug_in_world_kpis
    per_ev = kpis.loc[
        kpis["group_id"].eq("fleet")
        & kpis["day_type"].eq("all")
        & kpis["metric"].eq("plug_ins_per_ev_per_week"),
        "value",
    ]
    total = (heatmap.loc[present, "mean"] * heatmap.loc[present, "day_count"]).sum()
    assert np.isclose(total, per_ev.mean(), rtol=0.0, atol=1e-9), "heatmap adds up to plug-ins"


def _validate_world_units(world: pd.DataFrame) -> None:
    """Unit relations that catch kWh stored as kW and counts stored as shares."""

    def same(left, right, rule):
        assert np.allclose(left, right, rtol=1e-9, atol=1e-9), f"fleet_world_intervals: {rule}"

    same(world["home_import_kw"], world["home_import_kwh"] / 0.5, "home_import_kw = kWh / 0.5 h")
    same(
        world["total_import_kw"],
        (world["home_import_kwh"] + world["public_import_kwh"]) / 0.5,
        "total_import_kw = (home + public kWh) / 0.5 h",
    )
    same(
        world["connected_share"],
        world["connected_count"] / world["unit_count"],
        "connected_share = connected_count / unit_count",
    )
    same(
        world["battery_soc_percent"],
        100.0 * world["closing_battery_kwh"] / world["physical_capacity_kwh"],
        "battery_soc_percent = 100 * closing / capacity",
    )
    assert (world["connected_count"] <= world["unit_count"]).all(), "connected_count <= EVs"
    assert (world["driving_share"] <= world["away_share"] + 1e-12).all(), "driving is away"
    assert (world["public_charging_share"] <= world["away_share"] + 1e-12).all(), "public is away"


def _validate_units_and_cohorts(result) -> None:
    units = result.units
    _check_dtypes(units, UNIT_COLUMNS, "units", exact=False)
    assert len(units) == result.vehicle_count, "units has one row per EV"
    _check_unique(units, ["unit_id"], "units")
    assert units["cohort_id"].isin(COHORT_ORDER).all(), "units.cohort_id"
    assert units["preferred_target_soc_percent"].between(0.0, 100.0).all(), "target SoC 0-100"
    summary = result.cohort_summary
    _check_dtypes(summary, COHORT_SUMMARY_COLUMNS, "cohort_summary")
    assert tuple(summary["cohort_id"]) == COHORT_ORDER, "cohort_summary has six rows in order"
    assert summary["ev_count"].sum() == result.vehicle_count, "ev_count sums to vehicle_count"
    counts = units["cohort_id"].value_counts()
    for row in summary.itertuples():
        assert row.ev_count == counts.get(row.cohort_id, 0), f"ev_count for {row.cohort_id}"
        if row.ev_count == 0:
            # Missing, not zero: the view shows "no EVs sampled" (audit B1).
            assert np.isnan(row.plug_ins_per_ev_week_p50), "empty cohort stats must be NaN"
            assert pd.isna(row.peak_plug_in_local_hour), "empty cohort peak hour must be <NA>"


def _validate_assumptions(assumptions) -> None:
    assert isinstance(assumptions, tuple) and assumptions, "assumptions must be a non-empty tuple"
    names = [record.name for record in assumptions]
    assert len(names) == len(set(names)), "assumption names must be unique"
    for record in assumptions:
        for attribute in ("label", "unit", "source", "meaning", "group", "affects"):
            assert isinstance(getattr(record, attribute), str), f"{record.name}.{attribute}"
        assert record.evidence in ("source", "illustrative", "synthetic"), f"{record.name} evidence"
        assert isinstance(record.editable, bool), f"{record.name}.editable must be bool"
        numeric = isinstance(record.value, (int, float)) and not isinstance(record.value, bool)
        if record.editable and numeric:
            assert record.bounds is not None, f"{record.name} editable numeric needs bounds"
            low, high = record.bounds
            # NaN is an unset optional term (supplier contract v1 §8, decision
            # 0003): unavailable downstream, never a value to bound-check.
            unset = isinstance(record.value, float) and math.isnan(record.value)
            assert unset or low <= record.value <= high, f"{record.name} value outside bounds"
        if record.evidence == "source" and record.name.startswith("cnz_"):
            assert not record.editable, f"{record.name} source context is not editable"
    for name in CNZ_RECORD_NAMES:
        assert name in names, f"assumption {name} required"
    # Contract 8.2: the model's results carry all twelve per-archetype source
    # records; the fixture keeps a short record list without cohort names, so
    # the records are checked whenever any is present.
    by_name = {record.name: record for record in assumptions}
    if any(name in by_name for name in SESSION_SOURCE_RECORD_NAMES):
        for name in SESSION_SOURCE_RECORD_NAMES:
            assert name in by_name, f"assumption {name} required (contract 8.2)"
            record = by_name[name]
            assert record.evidence == "source" and not record.editable, f"{name} read-only"
    assert "public_charge_gbp_per_kwh" in names, "assumption public_charge_gbp_per_kwh required"


def _validate_average_day(result, paths, sampled) -> None:
    frame = result.average_day_bands
    _check_dtypes(frame, AVERAGE_DAY_COLUMNS, "average_day_bands")
    assert list(dict.fromkeys(frame["group_id"])) == ["fleet", *sampled], "average-day groups"
    per_group = len(paths) * len(DAY_TYPE_ORDER) * 2 * 48
    assert len(frame) == per_group * (1 + len(sampled)), "one row per group/path/day/metric/hh"
    _check_unique(
        frame,
        ["group_id", "path_id", "day_type", "metric", "local_half_hour"],
        "average_day_bands",
    )
    assert set(frame["path_id"]) == set(paths), "average-day paths"
    spread = frame.drop_duplicates("metric").set_index("metric")["spread"].to_dict()
    assert spread == {
        "connected_share": "across_worlds",
        "battery_soc_percent": "across_evs",
    }, "average-day spreads"
    worlds = frame.loc[frame["spread"].eq("across_worlds")]
    _check_ordered_quantiles(worlds, "low", "centre", "high", "average_day_bands")
    # Across EVs the centre is the mean (the brief's sketch), which can sit
    # outside a skewed P5-P95 band after the world median, so only low <= high.
    evs = frame.loc[frame["spread"].eq("across_evs")].dropna(subset=["low", "high"])
    assert (evs["low"] <= evs["high"] + 1e-9).all(), "average_day_bands: across-EV low > high"
    all_days = frame.loc[frame["day_type"].eq("all")]
    assert all_days["centre"].notna().all(), "'all' day type covers every half-hour"


def _validate_plug_ins(result) -> None:
    events = result.plug_in_events
    _check_dtypes(events, PLUG_IN_EVENT_COLUMNS, "plug_in_events", exact=False)
    assert events["path_id"].eq("normal").all(), "fleet plug_in_events are normal path only"
    _check_unique(events, ["world_id", "unit_id", "event_index"], "plug_in_events")
    assert events["plug_in_utc"].ge(result.horizon_start_utc).all(), "plug-in inside horizon"
    assert events["plug_in_utc"].lt(result.horizon_end_utc).all(), "plug-in inside horizon"
    assert (events["plug_in_london"] == events["plug_in_utc"]).all(), "London plug-in instant"
    assert events["plug_in_soc_percent"].between(0.0, 100.0).all(), "plug-in SoC 0-100"
    open_sessions = events["plug_out_utc"].isna()
    assert (open_sessions == events["still_plugged_at_horizon_end"]).all(), "NaT = still plugged"
    assert events.loc[open_sessions, "plug_out_soc_percent"].isna().all(), "open session SoC NaN"
    closed = events.loc[~open_sessions]
    assert (closed["plug_out_utc"] > closed["plug_in_utc"]).all(), "plug-out after plug-in"
    assert (events["battery_added_kwh"] <= events["home_import_kwh"] + 1e-9).all(), "losses"
    expected_index = events.groupby(["world_id", "unit_id"]).cumcount()
    ordered = events.sort_values(["world_id", "unit_id", "plug_in_utc"], kind="stable")
    assert (ordered["event_index"] == expected_index.loc[ordered.index]).all(), "event_index order"

    _validate_plug_in_heatmap(result)
    _validate_plug_event_heatmaps(result)
    _validate_session_distributions(result)
    summary = result.plug_in_summary
    _check_dtypes(summary.hour_bands, HOUR_BAND_COLUMNS, "plug_in_summary.hour_bands")
    _check_dtypes(summary.soc_bands, SOC_BAND_COLUMNS, "plug_in_summary.soc_bands")
    _check_dtypes(summary.kpis, PLUG_KPI_COLUMNS, "plug_in_summary.kpis")
    assert len(summary.hour_bands) == 3 * 24 and len(summary.soc_bands) == 3 * 20, "bin counts"
    assert tuple(dict.fromkeys(summary.hour_bands["day_type"])) == DAY_TYPE_ORDER, "day order"
    assert list(summary.kpis["metric"].unique()) == list(PLUG_KPI_METRICS), "plug KPI metrics"
    for name in ("hour_bands", "soc_bands"):
        _check_ordered_quantiles(
            getattr(summary, name), "share_p10", "share_p50", "share_p90", name
        )
    _check_ordered_quantiles(summary.kpis, "p10", "p50", "p90", "plug_in_summary.kpis")
    for name in ("hour_bands", "soc_bands"):
        frame = getattr(summary, name)
        shares = frame[["share_mean", "share_p10", "share_p50", "share_p90"]].stack()
        assert shares.between(0.0, 1.0).all(), f"plug_in_summary.{name} shares must be 0-1"
        totals = frame.loc[frame["world_count"] > 0].groupby("day_type")["share_mean"].sum()
        assert np.allclose(totals, 1.0), f"plug_in_summary.{name} share_mean must sum to 1"

    sampled = result.sampled_world_ids
    assert isinstance(sampled, tuple) and sampled, "sampled_world_ids must be a non-empty tuple"
    assert len(sampled) == min(result.world_count, MAX_PLUG_IN_EVENT_WORLDS), "sample size"
    assert sampled[0] == result.representative_world_id, "representative world first"
    assert len(set(sampled)) == len(sampled), "sampled_world_ids unique"
    assert all(0 <= w < result.world_count for w in sampled), "sampled_world_ids in range"
    assert set(events["world_id"]) <= set(sampled), "plug_in_events only from sampled worlds"
    order = events["world_id"].map({w: i for i, w in enumerate(sampled)})
    assert order.is_monotonic_increasing, "plug_in_events follow sampled_world_ids order"

    kpis = result.plug_in_world_kpis
    _check_dtypes(kpis, PLUG_IN_WORLD_KPI_COLUMNS, "plug_in_world_kpis")
    sampled_cohorts = result.cohort_summary.loc[result.cohort_summary["ev_count"] > 0]
    metric_count = len(PLUG_KPI_METRICS) + len(WORLD_KPI_EXTRA_METRICS)
    expected_rows = result.world_count * (1 + len(sampled_cohorts)) * 3 * metric_count
    assert len(kpis) == expected_rows, "plug_in_world_kpis covers every world, group, day, metric"
    _check_unique(kpis, ["world_id", "group_id", "day_type", "metric"], "plug_in_world_kpis")
    keys = ["world_id", "group_id", "day_type", "metric"]
    recomputed = plug_in_world_kpis_from_events(events, result.units, sampled).set_index(keys)
    stored = kpis.loc[kpis["world_id"].isin(sampled)].set_index(keys).loc[recomputed.index]
    _assert_close(
        stored.reset_index(), recomputed.reset_index(), ["value"], "plug_in_world_kpis (sampled)"
    )
    _assert_close(
        summary.kpis, plug_in_kpis_from_world_kpis(kpis), ["p10", "p50", "p90"], "plug-in kpis"
    )
    for row in sampled_cohorts.itertuples():
        expected = cohort_plug_in_stats(kpis, row.cohort_id)
        for column in ("plug_ins_per_ev_week_p50", "median_plug_in_soc_percent_p50"):
            assert np.isclose(getattr(row, column), expected[column], equal_nan=True), (
                f"cohort_summary.{column} for {row.cohort_id} disagrees with plug_in_world_kpis"
            )
        assert pd.isna(row.peak_plug_in_local_hour) == pd.isna(expected["peak_plug_in_local_hour"])
        if not pd.isna(row.peak_plug_in_local_hour):
            assert row.peak_plug_in_local_hour == expected["peak_plug_in_local_hour"], "peak hour"


def _validate_action(result, world, slots) -> None:
    diff = result.difference_bands
    _check_dtypes(diff, DIFFERENCE_BAND_COLUMNS, "difference_bands")
    _assert_close(
        diff, difference_bands_from_world(world), ["mean", "p10", "p50", "p90"], "difference_bands"
    )
    _check_ordered_quantiles(diff, "p10", "p50", "p90", "difference_bands")
    weekly = result.difference_weekly
    _check_dtypes(weekly, DIFFERENCE_WEEKLY_COLUMNS, "difference_weekly")
    _assert_close(
        weekly,
        difference_weekly_from_world(world),
        ["normal_value", "selected_value", "selected_minus_normal"],
        "difference_weekly",
    )
    weekly_bands = result.difference_weekly_bands
    _check_dtypes(weekly_bands, DIFFERENCE_WEEKLY_BAND_COLUMNS, "difference_weekly_bands")
    _assert_close(
        weekly_bands, weekly_difference_bands(weekly), ["mean", "p10", "p50", "p90"], "weekly"
    )

    forecast = result.forecast_prices
    _check_dtypes(forecast, FORECAST_PRICE_COLUMNS, "forecast_prices")
    # One day-ahead price per (world, slot), world-major (decision 0004 item 48).
    assert list(forecast["world_id"]) == [
        w for w in range(result.world_count) for _ in range(SLOT_COUNT)
    ], "forecast price rows are world-major"
    assert list(forecast["slot_index"]) == list(range(SLOT_COUNT)) * result.world_count, (
        "forecast price per world and slot"
    )
    _check_slot_keys(forecast, slots, "forecast_prices")
    price_bands = result.forecast_price_bands
    _check_dtypes(price_bands, FORECAST_PRICE_BAND_COLUMNS, "forecast_price_bands")
    assert list(price_bands["slot_index"]) == list(range(SLOT_COUNT)), "price band per slot"
    _check_slot_keys(price_bands, slots, "forecast_price_bands")
    assert price_bands["world_count"].eq(result.world_count).all(), "price bands use every world"
    _check_ordered_quantiles(price_bands, "p10", "p50", "p90", "forecast_price_bands")
    _assert_close(
        price_bands,
        forecast_price_bands_from(forecast),
        ["p10", "p50", "p90"],
        "forecast_price_bands",
    )
    evaluation = result.evaluation_prices
    _check_dtypes(evaluation, EVALUATION_PRICE_COLUMNS, "evaluation_prices")
    assert len(evaluation) == result.world_count * SLOT_COUNT, "evaluation price per world/slot"
    shocks = result.price_shocks
    _check_dtypes(shocks, PRICE_SHOCK_COLUMNS, "price_shocks")
    assert shocks["start_slot_index"].between(0, SLOT_COUNT - 1).all(), "shocks start in study"
    assert shocks["world_id"].between(0, result.world_count - 1).all(), "shock world ids"
    assert shocks["shock_class"].isin(["mild", "big"]).all(), "shock classes"
    assert shocks["direction"].isin(["up", "down"]).all(), "shock directions"
    assert (shocks["size_gw"] >= 0.0).all() and (shocks["duration_slots"] >= 1).all()
    assert list(shocks.index) == list(range(len(shocks))), "shock rows reset"
    order = shocks[["world_id", "start_slot_index"]].to_numpy()
    assert (np.diff(order[:, 0] * SLOT_COUNT + order[:, 1]) >= 0).all(), "shocks by world, start"
    # Decision-time cut-off (plan B4): each day-ahead price is published at
    # 13:00 London on the day before its London delivery date, so before its
    # slot starts; the smart charger only ranks prices published by then.
    available = forecast["forecast_available_at_utc"]
    delivery = forecast["interval_start_london"].dt.tz_localize(None).dt.normalize()
    published = (delivery - pd.Timedelta(days=1) + pd.Timedelta(hours=13)).dt.tz_localize(LONDON)
    assert (available == published).all(), "published 13:00 London the day before delivery"
    assert (available < forecast["interval_start_utc"]).all(), "published before delivery"

    summary = result.action_summary
    assert summary.action_id == ACTION_ID, "action_summary.action_id"
    assert isinstance(summary.action_label, str) and summary.action_label, "action_label"
    assert summary.departure_margin_hours >= 0.0, "departure margin is not negative"
    assert summary.vehicle_count == result.vehicle_count, "action_summary.vehicle_count"
    assert summary.forecast_available_at_utc == available.iat[0], "forecast publication time"

    outcomes = result.smart_charging_world
    # Decision 0007, model step 2: with the timed path, both frames gain one
    # more column/row (timed_average_price_gbp_per_mwh); read off what the
    # real frame has rather than assumed fixed at two paths.
    timed_active = "timed" in set(world["path_id"])
    world_columns = {}
    for key, kind in SMART_CHARGING_WORLD_COLUMNS.items():
        world_columns[key] = kind
        if key == "selected_average_price_gbp_per_mwh" and timed_active:
            world_columns["timed_average_price_gbp_per_mwh"] = "float"
    _check_dtypes(outcomes, world_columns, "smart_charging_world")
    assert list(outcomes["world_id"]) == list(range(result.world_count)), "one row per world"
    _assert_close(
        outcomes,
        smart_charging_world_from(world, forecast),
        [c for c, kind in world_columns.items() if kind in ("float", "int")],
        "smart_charging_world",
    )
    share = outcomes["moved_home_import_share"].dropna()
    assert share.between(0.0, 1.0 + 1e-12).all(), "moved share is a fraction"
    assert outcomes["early_departure_count"].ge(0).all(), "early departures are counts"
    smart_summary = result.smart_charging_summary
    _check_dtypes(smart_summary, SMART_CHARGING_SUMMARY_COLUMNS, "smart_charging_summary")
    expected_metrics = [
        m for m in SMART_CHARGING_METRICS if timed_active or m != "timed_average_price_gbp_per_mwh"
    ]
    assert list(smart_summary["metric"]) == expected_metrics, "metric order"
    _assert_close(
        smart_summary,
        smart_charging_summary_from(outcomes),
        ["world_count", "mean", "p10", "p50", "p90"],
        "smart_charging_summary",
    )
    bands = result.price_band_shift
    # Decision 0007, model step 2: "moved" (the `mean`/`p10`/`p50`/`p90`
    # columns) always stays normal minus selected, whatever paths the run
    # has; the per-path `*_kwh_*` columns gain a `timed_kwh_*` triple when
    # the run has the timed path, in the same position its own `_kwh_`
    # columns would sit (after `selected_kwh_*`).
    price_band_columns = {}
    for key, kind in PRICE_BAND_SHIFT_COLUMNS.items():
        price_band_columns[key] = kind
        if key == "selected_kwh_p90" and "timed_kwh_p10" in bands.columns:
            for stat in ("p10", "p50", "p90"):
                price_band_columns[f"timed_kwh_{stat}"] = "float"
    _check_dtypes(bands, price_band_columns, "price_band_shift")
    dates = list(dict.fromkeys(slots["local_date"]))
    assert list(bands["local_date"]) == [d for d in dates for _ in PRICE_BANDS], "date rows"
    assert list(bands["price_band"]) == list(PRICE_BANDS) * len(dates), "band order"
    assert np.allclose(
        bands.groupby("local_date")["mean_slot_count"].sum().to_numpy(dtype=float),
        [slots["local_date"].eq(d).sum() for d in dates],
    ), "bands cover every slot of each date"
    _check_ordered_quantiles(bands, "p10", "p50", "p90", "price_band_shift")
    _assert_close(
        bands,
        price_band_shift_from(world, forecast, slots),
        ["mean", "p10", "p50", "p90", *(c for c in PRICE_BAND_SHIFT_COLUMNS if "_kwh_" in c)],
        "price_band_shift",
    )
    # The "moved" reading itself stays normal versus selected only (decision
    # 0007); the optional timed path only ever adds its own `*_kwh_*` columns.
    for path in ("normal", "selected", "timed"):
        if f"{path}_kwh_p10" not in bands.columns:
            continue
        _check_ordered_quantiles(
            bands, f"{path}_kwh_p10", f"{path}_kwh_p50", f"{path}_kwh_p90", "price_band_shift"
        )
        assert (bands[f"{path}_kwh_p10"] >= 0.0).all(), f"{path} band energy >= 0"

    peaks = result.weekly_peak_summary
    # Decision 0007, model step 2: one row per path the fleet frame actually
    # has (normal, selected, plus timed when the run has it), in that order.
    peak_paths = list(dict.fromkeys(world["path_id"]))
    _check_dtypes(peaks, WEEKLY_PEAK_COLUMNS, "weekly_peak_summary")
    assert list(peaks["path_id"]) == peak_paths, "weekly_peak_summary path order"
    _check_ordered_quantiles(peaks, "p10", "p50", "p90", "weekly_peak_summary")
    _check_ordered_quantiles(
        peaks, "ratio_to_normal_p10", "ratio_to_normal_p50", "ratio_to_normal_p90", "peak ratio"
    )
    assert peaks["world_count"].eq(result.world_count).all(), "weekly peaks use every world"
    assert peaks["modal_peak_week_count"].between(1, result.world_count).all(), "modal count"
    modal = peaks.rename(
        columns={
            "modal_peak_slot_index": "slot_index",
            "modal_peak_interval_start_utc": "interval_start_utc",
            "modal_peak_interval_start_london": "interval_start_london",
        }
    )
    modal["interval_end_utc"] = modal["interval_start_utc"] + HALF_HOUR
    _check_slot_keys(modal, slots, "weekly_peak_summary modal slot")
    normal_ratio = peaks.loc[peaks["path_id"].eq("normal"), ["ratio_to_normal_p50"]].dropna()
    assert np.allclose(normal_ratio, 1.0), "normal peak is 1x itself"
    _check_ordered_quantiles(
        peaks,
        "coincidence_factor_p10",
        "coincidence_factor_p50",
        "coincidence_factor_p90",
        "coincidence factor",
    )
    factors = peaks[["coincidence_factor_p10", "coincidence_factor_p90"]].stack()
    # Home import only happens in fully plugged-in slots at most at full power.
    assert factors.between(0.0, 1.0 + 1e-9).all(), "coincidence factor within 0-1"
    assert peaks["coincidence_world_count"].between(0, result.world_count).all()
    _assert_close(
        peaks,
        weekly_peak_summary_from(world, _one_charging_power(result)),
        [c for c, kind in WEEKLY_PEAK_COLUMNS.items() if kind in ("float", "int")],
        "weekly_peak_summary",
    )

    relative = result.price_relative_bands
    _check_dtypes(relative, PRICE_RELATIVE_COLUMNS, "price_relative_bands")
    assert len(relative) == 8 * 48, "7 weekdays and all days x 48 half-hours"
    assert relative["weekday"].isna().sum() == 48, "48 all-days rows"
    _check_ordered_quantiles(relative, "p10", "p50", "p90", "price_relative_bands")
    _assert_close(
        relative,
        price_relative_bands_from(result.forecast_prices),
        ["sample_count", "world_count", "mean", "p10", "p50", "p90"],
        "price_relative_bands",
    )
    cheapest = result.cheapest_half_hour_summary
    _check_dtypes(cheapest, CHEAPEST_HALF_HOUR_COLUMNS, "cheapest_half_hour_summary")
    assert len(cheapest) == 1, "cheapest_half_hour_summary has one row"
    expected = cheapest_half_hour_from(result.forecast_prices)
    _assert_close(
        cheapest,
        expected,
        [c for c, kind in CHEAPEST_HALF_HOUR_COLUMNS.items() if kind != "str"],
        "cheapest_half_hour_summary",
    )
    assert (
        cheapest.filter(like="label").to_numpy() == expected.filter(like="label").to_numpy()
    ).all(), "cheapest half-hour labels"

    cost = result.cost_effect
    _check_dtypes(cost, COST_EFFECT_COLUMNS, "cost_effect")
    assert list(cost["world_id"]) == list(range(result.world_count)), "one cost row per world"
    parts = cost[[column for _, column in COST_COMPONENTS[:4]]].sum(axis=1)
    assert np.allclose(parts, cost["illustrative_selected_minus_normal_total_gbp"]), "total = sum"
    assert cost["illustrative_unrecovered_energy_value_gbp"].ge(0.0).all(), "never negative"
    assert cost["illustrative_unserved_travel_value_gbp"].ge(0.0).all(), "never negative"
    # Decision 0004 item 45: magnitude per world, and the count is of
    # material weeks only.
    shortfall = (-cost["selected_minus_normal_closing_battery_kwh"]).clip(lower=0.0)
    public = cost["selected_minus_normal_public_import_kwh"].clip(lower=0.0)
    unserved = cost["selected_minus_normal_unserved_travel_kwh"].clip(lower=0.0)
    parts = [
        part.where(part > NOT_RECOVERED_TOLERANCE_KWH, 0.0)
        for part in (shortfall, public, unserved)
    ]
    assert np.allclose(cost["unrecovered_kwh"], sum(parts)), "unrecovered = the three shortfalls"
    assert cost["energy_not_recovered"].eq(cost["unrecovered_kwh"].gt(0.0)).all(), "flag = any"
    normal_import = cost["normal_home_import_kwh"]
    share = (cost["unrecovered_kwh"] / normal_import).where(normal_import > 0.0)
    assert np.allclose(cost["unrecovered_share"], share, equal_nan=True), "share of normal import"
    assert (cost["not_recovered_material"] <= cost["energy_not_recovered"]).all(), "material => any"
    assert cost["sessions_affected_count"].ge(0).all(), "sessions affected >= 0"
    assert result.not_recovered_world_count == int(cost["not_recovered_material"].sum())
    recovered_summary = result.not_recovered_summary
    _check_dtypes(recovered_summary, NOT_RECOVERED_SUMMARY_COLUMNS, "not_recovered_summary")
    assert list(recovered_summary["metric"]) == list(NOT_RECOVERED_METRICS), "metric order"
    _assert_close(
        recovered_summary,
        not_recovered_summary_from(cost),
        ["world_count", "mean", "p10", "p50", "p90"],
        "not_recovered_summary",
    )
    cost_summary = result.cost_effect_summary
    _check_dtypes(cost_summary, COST_SUMMARY_COLUMNS, "cost_effect_summary")
    assert list(cost_summary["component"]) == [c for c, _ in COST_COMPONENTS], "component order"
    _assert_close(
        cost_summary,
        cost_summary_from_cost_effect(cost),
        ["mean", "p10", "p50", "p90"],
        "cost_effect_summary",
    )
    # Decision 0007, model step 2: the timed path's own cost reading (§4.8a,
    # §4.9b), same schema and reference functions as the ordinary cost effect
    # above, "timed" in the selected slot.  Present only when this result's
    # fleet frames actually carry "timed" (``timed_active``, set above beside
    # the smart-charging-outcomes check), never merely because this is an
    # action result; ``None`` otherwise, even on an action result.
    if timed_active:
        timed_cost = result.timed_cost_effect
        assert timed_cost is not None, "timed_cost_effect required with the timed path"
        _check_dtypes(timed_cost, COST_EFFECT_COLUMNS, "timed_cost_effect")
        assert list(timed_cost["world_id"]) == list(range(result.world_count)), (
            "one timed_cost_effect row per world"
        )
        timed_parts = timed_cost[[column for _, column in COST_COMPONENTS[:4]]].sum(axis=1)
        assert np.allclose(
            timed_parts, timed_cost["illustrative_selected_minus_normal_total_gbp"]
        ), "timed_cost_effect total = sum"
        assert timed_cost["illustrative_unrecovered_energy_value_gbp"].ge(0.0).all()
        assert timed_cost["illustrative_unserved_travel_value_gbp"].ge(0.0).all()
        timed_shortfall = (-timed_cost["selected_minus_normal_closing_battery_kwh"]).clip(lower=0.0)
        timed_public = timed_cost["selected_minus_normal_public_import_kwh"].clip(lower=0.0)
        timed_unserved = timed_cost["selected_minus_normal_unserved_travel_kwh"].clip(lower=0.0)
        timed_shortfall_parts = [
            part.where(part > NOT_RECOVERED_TOLERANCE_KWH, 0.0)
            for part in (timed_shortfall, timed_public, timed_unserved)
        ]
        assert np.allclose(timed_cost["unrecovered_kwh"], sum(timed_shortfall_parts)), (
            "timed_cost_effect unrecovered = the three shortfalls"
        )
        assert timed_cost["energy_not_recovered"].eq(timed_cost["unrecovered_kwh"].gt(0.0)).all()
        timed_normal_import = timed_cost["normal_home_import_kwh"]
        timed_share = (timed_cost["unrecovered_kwh"] / timed_normal_import).where(
            timed_normal_import > 0.0
        )
        assert np.allclose(timed_cost["unrecovered_share"], timed_share, equal_nan=True)
        assert (timed_cost["not_recovered_material"] <= timed_cost["energy_not_recovered"]).all()
        # Unlike the ordinary cost effect, this count is paired against
        # "timed" rather than "selected" (same function, see
        # ``sessions_affected_per_world``); real counts, not a placeholder.
        assert timed_cost["sessions_affected_count"].ge(0).all(), "timed sessions affected >= 0"
        assert result.timed_not_recovered_world_count == int(
            timed_cost["not_recovered_material"].sum()
        )
        timed_recovered_summary = result.timed_not_recovered_summary
        _check_dtypes(
            timed_recovered_summary, NOT_RECOVERED_SUMMARY_COLUMNS, "timed_not_recovered_summary"
        )
        assert list(timed_recovered_summary["metric"]) == list(NOT_RECOVERED_METRICS)
        _assert_close(
            timed_recovered_summary,
            not_recovered_summary_from(timed_cost),
            ["world_count", "mean", "p10", "p50", "p90"],
            "timed_not_recovered_summary",
        )
        timed_cost_summary = result.timed_cost_effect_summary
        _check_dtypes(timed_cost_summary, COST_SUMMARY_COLUMNS, "timed_cost_effect_summary")
        assert list(timed_cost_summary["component"]) == [c for c, _ in COST_COMPONENTS]
        _assert_close(
            timed_cost_summary,
            cost_summary_from_cost_effect(timed_cost),
            ["mean", "p10", "p50", "p90"],
            "timed_cost_effect_summary",
        )
    else:
        # ``getattr(..., None)`` default (as the no-action branch above
        # uses): a narrower duck-typed stand-in (``FixtureForecastResult``)
        # may not declare these fields at all rather than setting them None,
        # and that is just as much "not applicable" here.
        for name in TIMED_COST_FIELDS:
            assert getattr(result, name, None) is None, (
                f"{name} must be None without the timed path"
            )
    _validate_trading(result, world, slots)


def validate_one_ev_replay_v2(replay, result) -> None:
    """Assert a one-EV replay satisfies contract section 5."""

    # One-EV replay's path set mirrors ``individual._paths``: normal (plus
    # selected on an action result), plus the optional timed path (decision
    # 0007, model step 2) exactly when this result's fleet frames carry it.
    paths = ["normal", "selected"] if result.model == "action" else ["normal"]
    if "timed" in set(result.fleet_world_intervals["path_id"]):
        paths.append("timed")
    assert replay.unit_id in set(result.units["unit_id"]), "replay unit_id"
    assert 0 <= replay.world_id < result.world_count, "replay world_id"
    assert replay.is_representative_world == (replay.world_id == result.representative_world_id)
    assert isinstance(replay.traits, dict), "traits must be a dict"
    intervals = replay.intervals
    _check_dtypes(intervals, REPLAY_INTERVAL_COLUMNS, "replay.intervals")
    assert list(dict.fromkeys(intervals["path_id"])) == paths, "replay paths"
    assert len(intervals) == len(paths) * SLOT_COUNT, "one row per path and slot"
    _check_slot_keys(intervals, result.study_slots, "replay.intervals")
    assert intervals["location"].isin(LOCATION_ORDER).all(), "location values"
    assert (intervals["connected"] == intervals["location"].eq("home_plugged")).all()
    assert intervals["battery_soc_percent"].between(0.0, 100.0).all(), "SoC 0-100"
    shortfall = intervals["early_departure_shortfall_kwh"]
    assert shortfall.ge(0.0).all(), "early-departure shortfall never negative"
    assert shortfall[intervals["path_id"].eq("normal")].eq(0.0).all(), "no plans on normal"
    _check_dtypes(replay.plug_events, PLUG_IN_EVENT_COLUMNS, "replay.plug_events", exact=False)
    assert replay.plug_events["unit_id"].eq(replay.unit_id).all(), "plug events for this EV"
    audit = replay.daily_audit
    _check_dtypes(audit, DAILY_AUDIT_COLUMNS, "replay.daily_audit")
    assert len(audit) == len(paths) * STUDY_DAYS, "one audit row per path and study date"
    no_trip = ~audit["drives_today"]
    assert audit.loc[no_trip, "departure_utc"].isna().all(), "no trip, no departure"
    assert audit.loc[no_trip, "departure_soc_percent"].isna().all(), "no trip, SoC NaN"
    assert audit["shortfall_kwh"].ge(0.0).all(), "shortfall never negative"
    assert (audit["shortfall_kwh"] <= audit["trip_energy_need_kwh"] + 1e-9).all()

    # The audit's reconciliation: this EV's normal path is part of the fleet sum.
    world = result.fleet_world_intervals
    fleet = world.loc[world["world_id"].eq(replay.world_id) & world["path_id"].eq("normal")]
    normal = intervals.loc[intervals["path_id"].eq("normal")]
    assert (
        normal["home_import_kwh"].to_numpy() <= fleet["home_import_kwh"].to_numpy() + 1e-9
    ).all(), "one EV's import cannot exceed the fleet's"


def validate_run_comparison_v2(comparison) -> None:
    """Assert a run comparison satisfies contract section 6."""

    assert isinstance(comparison.matched_futures, bool), "matched_futures must be bool"
    assert comparison.matched_futures == (not comparison.mismatch_reasons), "reasons iff unmatched"
    # The headline path is always normal or selected (decision 0007 keeps
    # the optional "timed" path out of run comparison).
    assert comparison.path_id_a in ("normal", "selected")
    assert comparison.path_id_b in ("normal", "selected")
    _check_dtypes(comparison.changed, CHANGED_COLUMNS, "comparison.changed")
    _check_dtypes(comparison.difference_bands, DIFFERENCE_BAND_COLUMNS, "comparison bands")
    _check_ordered_quantiles(comparison.difference_bands, "p10", "p50", "p90", "comparison bands")
    _check_dtypes(comparison.weekly_differences, COMPARE_WEEKLY_COLUMNS, "weekly_differences")
    _check_dtypes(comparison.kpis, COMPARE_KPI_COLUMNS, "comparison.kpis")
    metrics = set(comparison.weekly_differences["metric"])
    assert ("illustrative_total_gbp" in metrics) == comparison.illustrative_total_available
    weekly = comparison.weekly_differences
    assert np.allclose(weekly["b_minus_a"], weekly["value_b"] - weekly["value_a"], equal_nan=True)


# ==========================================================================
# Trading overlay (trading contract v1 §5 and §9; decisions 0004 items
# 57-58 and 0005).  Specs, reference computations and the validator, written
# from the contract text without importing the model, so the fixture and
# real results are checked against the same independent arithmetic.
# ==========================================================================

TRADING_STRATEGIES = ("da_only", "full", "perfect_foresight")
TRADING_BUCKETS = (
    "day_ahead_revenue_gbp",
    "intraday_pnl_gbp",
    "trading_cost_gbp",
    "imbalance_gbp",
    "baseline_effect_gbp",
    "grid_event_payment_gbp",
    "supplier_compensation_gbp",
    "customer_revenue_share_gbp",
    "unmet_charge_penalty_gbp",
)
TRADING_MONEY = (*TRADING_BUCKETS, "net_gbp")
SHOCK_KINDS = ("known", "surprise", "none")
TRADING_ATTRIBUTION = tuple(
    f"{bucket}_{kind}_gbp"
    for bucket in ("day_ahead_revenue", "intraday_pnl", "imbalance")
    for kind in SHOCK_KINDS
)
# Intraday dispatch contract §6.2-§6.3: the "of which" split of the full
# strategy's intraday P&L and trading cost; not buckets.
TRADING_DISPATCH_SPLIT = (
    "intraday_pnl_rebalancing_gbp",
    "intraday_pnl_reoptimisation_gbp",
    "trading_cost_rebalancing_gbp",
    "trading_cost_reoptimisation_gbp",
)
TRADING_VOLUMES = (
    "sold_day_ahead_mwh",
    "final_position_mwh",
    "settled_mwh",
    "flexibility_mwh",
    "baseline_effect_mwh",
    "imbalance_mwh",
    "abs_imbalance_mwh",
    "intraday_traded_mwh",
    "delivered_within_final_mwh",
    "delivered_within_day_ahead_mwh",
    "event_delivered_mwh",
    "unmet_charge_kwh",
)
# The ledger terms a result's assumption records carry (trading contract §7, §9.9).
TRADING_TERM_NAMES = (
    "trading.half_spread_gbp_per_mwh",
    "trading.customer_revenue_share",
    "trading.supplier_compensation_gbp_per_mwh",
    "trading.unmet_charge_penalty_gbp_per_kwh",
    "trading.day_ahead_commitment_share",
)
_T_STATS = {"world_count": "int", "mean": "float", "p10": "float", "p50": "float", "p90": "float"}

DEVIATION_WORLD_SLOT_COLUMNS = {
    "world_id": "int",
    "slot_index": "int",
    "interval_start_utc": "utc",
    "interval_start_london": "london",
    "night_index": "int",
    "baseline_kwh": "float",
    "unmanaged_kwh": "float",
    "metered_kwh": "float",
    "baseline_kw": "float",
    "unmanaged_kw": "float",
    "metered_kw": "float",
    # The aggregator's expected metered import at the day-ahead decision
    # (x_t of trading contract §4.4), an addition for the supplier P&L view.
    "expected_metered_kwh": "float",
    # x^U_t of supplier contract §2: the same day-ahead expected sessions
    # with every session unmanaged (rho = 1).
    "expected_unmanaged_kwh": "float",
    "deviation_kwh": "float",
    "true_reduction_kwh": "float",
    "baseline_effect_kwh": "float",
    "settled_kwh": "float",
    "settlement_open": "bool",
    **{f"position_{s}_kwh": "float" for s in TRADING_STRATEGIES},
    # Trading contract §10.4: the newsvendor level alpha and e(alpha); NaN
    # in fixed-share mode.
    "commit_level": "float",
    "forecast_error_quantile_kwh": "float",
    "day_ahead_gbp_per_mwh": "float",
    "intraday_close_gbp_per_mwh": "float",
    "imbalance_gbp_per_mwh": "float",
    "known_shock_gw": "float",
    "surprise_shock_gw": "float",
    "shock_kind": "str",
    "evidence_kind": "str",
}
POSITION_UPDATE_COLUMNS = {
    "world_id": "int",
    "slot_index": "int",
    "night_index": "int",
    "interval_start_utc": "utc",
    "interval_start_london": "london",
    "decision_utc": "utc",
    "decision_london": "london",
    "stage": "str",
    "hours_to_gate_closure": "int",
    "baseline_known_kwh": "float",
    "forecast_metered_kwh": "float",
    "position_kwh": "float",
    "trade_kwh": "float",
    "price_gbp_per_mwh": "float",
    "frozen_position_kwh": "float",
    "frozen_trade_kwh": "float",
    "evidence_kind": "str",
}
TRADING_LEDGER_COLUMNS = {
    "world_id": "int",
    "strategy": "str",
    "night_index": "int",
    "night_start_local_date": "date",
    "day_label": "str",
    **dict.fromkeys(TRADING_MONEY, "float"),
    **dict.fromkeys(TRADING_ATTRIBUTION, "float"),
    **dict.fromkeys(TRADING_DISPATCH_SPLIT, "float"),
    **{f"settled_mwh_{kind}": "float" for kind in SHOCK_KINDS},
    **dict.fromkeys(TRADING_VOLUMES, "float"),
    "evidence_kind": "str",
}
TRADING_WEEK_COLUMNS = {
    name: kind
    for name, kind in TRADING_LEDGER_COLUMNS.items()
    if name not in ("night_index", "night_start_local_date", "day_label")
}
TRADING_LEDGER_SUMMARY_COLUMNS = {
    "strategy": "str",
    "night_index": "Int64",
    "metric": "str",
    "unit": "str",
    **_T_STATS,
}
TRADING_KPI_COLUMNS = {
    "strategy": "str",
    "metric": "str",
    "unit": "str",
    **_T_STATS,
    "cvar5": "float",
}
TRADING_KPI_METRICS = (
    ("net_gbp_per_week", "GBP per week"),
    ("net_gbp_per_ev_year", "GBP per EV per year"),
    ("settled_mwh_per_week", "MWh per week"),
    ("imbalance_volume_share", "fraction"),
    ("baseline_effect_share", "fraction"),
    ("event_delivered_mwh_per_week", "MWh per week"),
    ("flex_margin_gbp_per_week", "GBP per week"),
    ("capture_rate", "fraction"),
    ("value_of_intraday_gbp_per_week", "GBP per week"),
    ("cost_of_uncertainty_gbp_per_week", "GBP per week"),
    ("imbalance_gbp_per_mwh_traded", "GBP per MWh"),
    ("imbalance_share_of_traded", "fraction"),
    ("firmness", "fraction"),
    ("firmness_day_ahead", "fraction"),
    ("flex_margin_gbp_per_mw_year", "GBP per MW per year"),
    ("day_ahead_spread_gbp_per_mwh", "GBP per MWh"),
    # Intraday dispatch contract §6.3: the first two on ``full`` only.
    ("rebalancing_gbp_per_week", "GBP per week"),
    ("reoptimisation_gbp_per_week", "GBP per week"),
    ("dispatch_moved_mwh_per_week", "MWh per week"),
    # Decision 0004 item 67: baseline erosion, a fleet property on every strategy.
    ("settled_value_gbp_per_week", "GBP per week"),
    ("settled_mwh_eroded_per_week", "MWh per week"),
    ("settled_value_eroded_gbp_per_week", "GBP per week"),
    ("baseline_erosion_share", "fraction"),
    ("evening_settled_value_gbp_per_week", "GBP per week"),
    ("evening_settled_value_eroded_gbp_per_week", "GBP per week"),
    ("evening_baseline_erosion_share", "fraction"),
)
TRADING_EROSION_METRICS = (
    "settled_value_gbp_per_week",
    "settled_mwh_eroded_per_week",
    "settled_value_eroded_gbp_per_week",
    "baseline_erosion_share",
    "evening_settled_value_gbp_per_week",
    "evening_settled_value_eroded_gbp_per_week",
    "evening_baseline_erosion_share",
)
# The headline erosion window, 16:00-20:00 London (lead ruling on item 67).
TRADING_EROSION_EVENING = ("16:00", "20:00")
TRADING_EROSION_SHARES = {
    "baseline_erosion_share": (
        "settled_value_gbp_per_week",
        "settled_value_eroded_gbp_per_week",
    ),
    "evening_baseline_erosion_share": (
        "evening_settled_value_gbp_per_week",
        "evening_settled_value_eroded_gbp_per_week",
    ),
}
TRADING_FULL_ONLY_METRICS = (
    "value_of_intraday_gbp_per_week",
    "rebalancing_gbp_per_week",
    "reoptimisation_gbp_per_week",
)
TRADING_CVAR_METRICS = (
    "net_gbp_per_week",
    "net_gbp_per_ev_year",
    "flex_margin_gbp_per_week",
    "value_of_intraday_gbp_per_week",
    "cost_of_uncertainty_gbp_per_week",
    "rebalancing_gbp_per_week",
    "reoptimisation_gbp_per_week",
)
TRADING_CHECK_COLUMNS = {
    "check_id": "str",
    "description": "str",
    "value": "float",
    "tolerance": "float",
    "passed": "bool",
}
TRADING_CHECK_IDS = (
    "buckets_sum_to_net",
    "perfect_foresight_zero_imbalance",
    "deviation_identity",
    "mean_intraday_revision",
    "p10_ordering",
    "baseline_class_fallback",
    "baseline_missing_nights",
    "dispatch_split_sums_to_intraday",
)
SHOCK_ATTRIBUTION_COLUMNS = {
    "strategy": "str",
    "shock_kind": "str",
    "metric": "str",
    "unit": "str",
    **_T_STATS,
}
OPEN_POSITION_COLUMNS = {
    "strategy": "str",
    "metric": "str",
    "unit": "str",
    "local_half_hour": "str",
    "profile_order": "int",
    **_T_STATS,
}
FLEX_COST_CURVE_COLUMNS = {
    "day_type": "str",
    "local_half_hour": "str",
    "profile_order": "int",
    "metric": "str",
    "threshold_gbp_per_mwh": "float",
    "unit": "str",
    **_T_STATS,
    "risk_charge_gbp_per_mwh": "float",
    "evidence_kind": "str",
}
COST_CURVE_THRESHOLDS = np.arange(-300.0, 300.0 + 1e-9, 10.0)
SHAPE_PREMIUM_COLUMNS = {
    "path_id": "str",
    "metric": "str",
    "unit": "str",
    **_T_STATS,
    "evidence_kind": "str",
}
SHAPE_PREMIUM_METRICS = (
    ("load_weighted_price_gbp_per_mwh", "GBP/MWh"),
    ("baseload_price_gbp_per_mwh", "GBP/MWh"),
    ("shape_premium_gbp_per_mwh", "GBP/MWh"),
    ("shape_cost_gbp_per_week", "GBP per week"),
)
HOUSEHOLD_DISTRIBUTION_COLUMNS = {
    "group_id": "str",
    "bin_index": "int",
    "bin_lower_gbp": "float",
    "bin_upper_gbp": "float",
    **_T_STATS,
}
HOUSEHOLD_BIN_EDGES = np.arange(-20.0, 40.0 + 1e-9, 1.0)
HOUSEHOLD_SUMMARY_COLUMNS = {
    "group_id": "str",
    "statistic": "str",
    "unit": "str",
    "ev_count": "int",
    **_T_STATS,
    "evidence_kind": "str",
}
HOUSEHOLD_STATISTICS = (
    "mean_value",
    "p10_across_evs",
    "p50_across_evs",
    "p90_across_evs",
    "share_worse_off",
    "mean_customer_saving",
    "mean_revenue_share",
)
REVENUE_SEGMENT_COLUMNS = {
    "bucket": "str",
    "segment_type": "str",
    "segment_id": "str",
    "metric": "str",
    "unit": "str",
    "ev_count": "int",
    **_T_STATS,
    "evidence_kind": "str",
}
SUPPLIER_POSITION_COLUMNS = {
    "slot_index": "int",
    "interval_start_utc": "utc",
    "interval_start_london": "london",
    "settlement_date": "str",
    "settlement_period": "int",
    "night_index": "int",
    **{f"day_ahead_position_mw_{q}": "float" for q in ("p10", "p50", "p90")},
    **{f"final_position_mw_{q}": "float" for q in ("p10", "p50", "p90")},
    "baseline_mw_p50": "float",
    "metered_mw_p50": "float",
    "settled_mw_p50": "float",
    # Trading contract v1 §10.5f (lane J5): net shape change and 1-h firm figures.
    "unmanaged_mw_p50": "float",
    **{f"net_change_mw_{q}": "float" for q in ("p10", "p50", "p90")},
    "deliverable_turn_down_1h_mw_p10": "float",
    "deliverable_turn_up_1h_mw_p10": "float",
    **{f"day_ahead_price_gbp_per_mwh_{q}": "float" for q in ("p10", "p50", "p90")},
    "price_curve_source": "str",
    "evidence_kind": "str",
}
CONTROL_GROUP_COLUMNS = {
    "night_index": "Int64",
    "metric": "str",
    "unit": "str",
    "control_ev_count": "int",
    "treated_ev_count": "int",
    **_T_STATS,
    "evidence_kind": "str",
}
TRADING_FIELDS = (
    "deviation_world_slot",
    "position_updates",
    "trading_ledger_world",
    "trading_week_world",
    "trading_ledger_summary",
    "trading_kpis",
    "trading_checks",
    "shock_attribution_summary",
    "open_position_profile",
    "flex_cost_curve",
    "shape_premium_summary",
    "household_value_distribution",
    "household_value_summary",
    "revenue_by_segment",
    "supplier_positions",
)
"""Trading frames every action result carries (``control_group_summary`` may be None)."""


def world_stats(values) -> dict[str, float]:
    """World count, mean and linear P10/P50/P90 of per-world values, NaN left out."""

    kept = np.asarray(values, dtype=float)
    kept = kept[~np.isnan(kept)]
    if len(kept) == 0:
        return {"world_count": 0, "mean": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan}
    q = np.quantile(kept, (0.1, 0.5, 0.9), method="linear")
    return {
        "world_count": len(kept),
        "mean": float(kept.mean()),
        "p10": float(q[0]),
        "p50": float(q[1]),
        "p90": float(q[2]),
    }


def _safe_ratio(numerator, denominator) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator != 0.0)
    return out


def _deviation_matrix(deviation: pd.DataFrame, column: str) -> np.ndarray:
    rows = deviation.sort_values(["world_id", "slot_index"], kind="stable")
    return rows[column].to_numpy().reshape(-1, SLOT_COUNT)


def trading_ledger_from(
    deviation: pd.DataFrame,
    slots: pd.DataFrame,
    *,
    terms: dict[str, float],
    unmet_kwh: np.ndarray,
    intraday_cash: np.ndarray,
    intraday_volume: np.ndarray,
    event_payment_gbp: np.ndarray | None = None,
    event_delivered_mwh: np.ndarray | None = None,
    rebalancing_cash: np.ndarray | None = None,
    rebalancing_volume: np.ndarray | None = None,
) -> pd.DataFrame:
    """Reference ``trading_ledger_world`` (§4.7, §5.3) from the per-slot deviation frame.

    ``intraday_cash`` (GBP) and ``intraday_volume`` (kWh traded) are the full
    strategy's intraday trades per (world, slot); ``unmet_kwh`` is (world,
    night).  ``event_payment_gbp`` and ``event_delivered_mwh`` (world,
    night) are the grid requests' payment and delivery, strategy-free
    (zero when not given).
    Day-ahead position ``q`` is ``position_da_only_kwh``; perfect foresight
    sells ``V`` day-ahead.
    ``rebalancing_cash`` and ``rebalancing_volume`` are the frozen-book
    re-run's (dispatch §6.2) per (world, slot); None means the re-run repeats
    the actual trades (dispatch off), so re-optimisation is 0.
    """

    m = {c: _deviation_matrix(deviation, c) for c in DEVIATION_WORLD_SLOT_COLUMNS}
    worlds = m["settled_kwh"].shape[0]
    night = slots["night_index"].to_numpy()
    nights = np.unique(night)
    v, e, r = m["settled_kwh"], m["baseline_effect_kwh"], m["true_reduction_kwh"]
    p_da, sip, kind = m["day_ahead_gbp_per_mwh"], m["imbalance_gbp_per_mwh"], m["shock_kind"]
    settles = v > 0.0
    effect_cash = np.where(settles, e * p_da, 0.0) / 1000.0
    q = m["position_da_only_kwh"]
    zeros = np.zeros_like(v)
    plans = {
        "da_only": (q, q, zeros, zeros),
        "full": (q, m["position_full_kwh"], intraday_cash, intraday_volume),
        "perfect_foresight": (v, v, zeros, zeros),
    }
    frozen = {
        "full": (
            intraday_cash if rebalancing_cash is None else rebalancing_cash,
            intraday_volume if rebalancing_volume is None else rebalancing_volume,
        )
    }

    def per_night(values: np.ndarray) -> np.ndarray:
        return np.stack([values[:, night == n].sum(axis=1) for n in nights], axis=1)

    rows = []
    for strategy in TRADING_STRATEGIES:
        sold, final, cash, volume = plans[strategy]
        slot_cash = {
            "day_ahead_revenue": sold * p_da / 1000.0 - effect_cash,
            "intraday_pnl": cash,
            "imbalance": (v - final) * sip / 1000.0,
        }
        b = {
            "day_ahead_revenue_gbp": per_night(slot_cash["day_ahead_revenue"]),
            "intraday_pnl_gbp": per_night(cash),
            "trading_cost_gbp": -terms["trading.half_spread_gbp_per_mwh"]
            * per_night(volume)
            / 1000.0,
            "imbalance_gbp": per_night(slot_cash["imbalance"]),
            "baseline_effect_gbp": per_night(effect_cash),
            "grid_event_payment_gbp": np.zeros((worlds, len(nights)))
            if event_payment_gbp is None
            else np.asarray(event_payment_gbp, dtype=float),
            "supplier_compensation_gbp": -terms["trading.supplier_compensation_gbp_per_mwh"]
            * per_night(v)
            / 1000.0,
        }
        gross = sum(b.values())
        b["customer_revenue_share_gbp"] = -terms["trading.customer_revenue_share"] * np.maximum(
            gross, 0.0
        )
        b["unmet_charge_penalty_gbp"] = (
            -terms["trading.unmet_charge_penalty_gbp_per_kwh"] * unmet_kwh
        )
        b["net_gbp"] = sum(b[name] for name in TRADING_BUCKETS)
        for name, values in slot_cash.items():
            for k in SHOCK_KINDS:
                b[f"{name}_{k}_gbp"] = per_night(np.where(kind == k, values, 0.0))
        frozen_cash, frozen_volume = frozen.get(strategy, (zeros, zeros))
        b["intraday_pnl_rebalancing_gbp"] = per_night(frozen_cash)
        b["intraday_pnl_reoptimisation_gbp"] = (
            b["intraday_pnl_gbp"] - b["intraday_pnl_rebalancing_gbp"]
        )
        b["trading_cost_rebalancing_gbp"] = (
            -terms["trading.half_spread_gbp_per_mwh"] * per_night(frozen_volume) / 1000.0
        )
        b["trading_cost_reoptimisation_gbp"] = (
            b["trading_cost_gbp"] - b["trading_cost_rebalancing_gbp"]
        )
        for k in SHOCK_KINDS:
            b[f"settled_mwh_{k}"] = per_night(np.where(kind == k, v, 0.0)) / 1000.0
        volumes = {
            "sold_day_ahead_mwh": sold,
            "final_position_mwh": final,
            "settled_mwh": v,
            "flexibility_mwh": np.where(settles, r, 0.0),
            "baseline_effect_mwh": np.where(settles, e, 0.0),
            "imbalance_mwh": v - final,
            "abs_imbalance_mwh": np.abs(v - final),
            "intraday_traded_mwh": volume,
            "delivered_within_final_mwh": np.minimum(v, final),
            "delivered_within_day_ahead_mwh": np.minimum(v, sold),
        }
        for name, values in volumes.items():
            b[name] = per_night(values) / 1000.0
        b["event_delivered_mwh"] = (
            np.zeros((worlds, len(nights)))
            if event_delivered_mwh is None
            else np.asarray(event_delivered_mwh, dtype=float)
        )
        b["unmet_charge_kwh"] = unmet_kwh
        first = [np.flatnonzero(night == n)[0] for n in nights]
        for w in range(worlds):
            for i, n in enumerate(nights):
                rows.append(
                    {
                        "world_id": w,
                        "strategy": strategy,
                        "night_index": int(n),
                        "night_start_local_date": slots["local_date"].iat[first[i]],
                        "day_label": slots["day_label"].iat[first[i]],
                        **{name: float(values[w, i]) for name, values in b.items()},
                        "evidence_kind": "illustrative_synthetic",
                    }
                )
    ledger = pd.DataFrame(rows)
    ledger["_s"] = ledger["strategy"].map(TRADING_STRATEGIES.index)
    ledger = ledger.sort_values(["world_id", "_s", "night_index"], kind="stable")
    return typed_frame(ledger.reset_index(drop=True), TRADING_LEDGER_COLUMNS)


def trading_week_from(ledger: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c, k in TRADING_WEEK_COLUMNS.items() if k == "float"]
    week = ledger.groupby(["world_id", "strategy"], sort=False, as_index=False)[numeric].sum()
    week["evidence_kind"] = "illustrative_synthetic"
    return typed_frame(week, TRADING_WEEK_COLUMNS)


def _ledger_unit(column: str, period: str) -> str:
    if column.endswith("_gbp"):
        return f"GBP per {period}"
    if column.endswith("_kwh"):
        return f"kWh per {period}"
    return f"MWh per {period}"


def trading_ledger_summary_from(ledger: pd.DataFrame, week: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = (*TRADING_MONEY, *TRADING_DISPATCH_SPLIT, *TRADING_VOLUMES)
    for strategy in TRADING_STRATEGIES:
        blocks = [(pd.NA, "week", week.loc[week["strategy"].eq(strategy)])]
        by_night = ledger.loc[ledger["strategy"].eq(strategy)]
        blocks += [(int(n), "night", g) for n, g in by_night.groupby("night_index", sort=True)]
        for night, period, frame in blocks:
            ordered = frame.sort_values("world_id")
            for metric in metrics:
                rows.append(
                    {
                        "strategy": strategy,
                        "night_index": night,
                        "metric": metric,
                        "unit": _ledger_unit(metric, period),
                        **world_stats(ordered[metric]),
                    }
                )
    return typed_frame(pd.DataFrame(rows), TRADING_LEDGER_SUMMARY_COLUMNS)


def _cvar5(values) -> float:
    kept = np.asarray(values, dtype=float)
    kept = kept[~np.isnan(kept)]
    if len(kept) == 0:
        return np.nan
    return float(kept[kept <= np.quantile(kept, 0.05, method="linear")].mean())


def day_ahead_spread_from(forecast_prices: pd.DataFrame, slots: pd.DataFrame) -> np.ndarray:
    """Per world: mean over session nights of max - min day-ahead price (§9.1)."""

    prices = (
        forecast_prices.sort_values(["world_id", "slot_index"])["wholesale_forecast_gbp_per_mwh"]
        .to_numpy(dtype=float)
        .reshape(-1, SLOT_COUNT)
    )
    night = slots["night_index"].to_numpy()
    spreads = [
        prices[:, night == n].max(axis=1) - prices[:, night == n].min(axis=1)
        for n in np.unique(night)
    ]
    return np.mean(spreads, axis=0)


def dispatch_moved_mwh_from(dispatch_world_slot: pd.DataFrame | None, worlds: int) -> np.ndarray:
    """Per world: ``sum_t max(0, -moved_kwh) / 1000`` (intraday-dispatch-v1 §6.3); 0 when off."""

    if dispatch_world_slot is None:
        return np.zeros(worlds)
    ordered = dispatch_world_slot.sort_values(["world_id", "slot_index"])
    moved = ordered["moved_kwh"].to_numpy(dtype=float).reshape(worlds, -1)
    return np.clip(-moved, 0.0, None).sum(axis=1) / 1000.0


def baseline_erosion_from(
    deviation: pd.DataFrame,
    slots: pd.DataFrame,
    *,
    working_nights: int = 5,
    non_working_nights: int = 2,
    window_slots: int | None = None,
) -> dict[str, np.ndarray]:
    """Reference baseline-erosion values per world (decision 0004 item 67).

    Rebuilds the smart-night baseline from ``deviation_world_slot``'s
    metered import: for each night, the mean of the world's other study
    nights of the same day class (nearest earlier first, wrapping round the
    week, at most ``working_nights`` / ``non_working_nights``; the other
    class if none) at the same London half-hour, plus the in-day adjustment
    over the first ``window_slots`` slots when given.  Settles it with the
    run's settlement mask and values both settled volumes at the day-ahead
    price (GBP per week, MWh per week), over the week and over the
    ``TRADING_EROSION_EVENING`` slots.
    """

    metered = _deviation_matrix(deviation, "metered_kwh").astype(float)
    open_ = _deviation_matrix(deviation, "settlement_open").astype(bool)
    price = _deviation_matrix(deviation, "day_ahead_gbp_per_mwh").astype(float)
    settled = _deviation_matrix(deviation, "settled_kwh").astype(float)
    night = slots["night_index"].to_numpy()
    half_hour = slots["local_half_hour"].to_numpy()
    working = slots["day_type"].eq("weekday").to_numpy()
    nights = list(np.unique(night))
    count = len(nights)
    baseline = np.zeros_like(metered)
    for i, n in enumerate(nights):
        is_working = working[night == n][0]
        earlier = [nights[(i - k) % count] for k in range(1, count)]
        same = [m for m in earlier if working[night == m][0] == is_working]
        chosen = (same or earlier)[: working_nights if is_working else non_working_nights]
        if not chosen:
            continue
        pool = np.isin(night, chosen)
        idx = np.flatnonzero(night == n)
        for t in idx:
            cells = pool & (half_hour == half_hour[t])
            baseline[:, t] = metered[:, cells if cells.any() else pool].mean(axis=1)
        if window_slots:
            first = idx[:window_slots]
            baseline[:, idx] += (metered[:, first] - baseline[:, first]).mean(axis=1)[:, None]
    eroded = open_ * np.clip(baseline - metered, 0.0, None)
    labels = slots["local_time_label"].to_numpy(dtype=object)
    evening = (labels >= TRADING_EROSION_EVENING[0]) & (labels < TRADING_EROSION_EVENING[1])
    return {
        "settled_value_gbp_per_week": (settled * price).sum(axis=1) / 1000.0,
        "settled_mwh_eroded_per_week": eroded.sum(axis=1) / 1000.0,
        "settled_value_eroded_gbp_per_week": (eroded * price).sum(axis=1) / 1000.0,
        "evening_settled_value_gbp_per_week": (settled * price)[:, evening].sum(axis=1) / 1000.0,
        "evening_settled_value_eroded_gbp_per_week": (eroded * price)[:, evening].sum(axis=1)
        / 1000.0,
    }


def trading_kpis_from(
    week: pd.DataFrame,
    *,
    vehicle_count: int,
    deferrable_2h_kw: np.ndarray,
    spread: np.ndarray,
    erosion: dict[str, np.ndarray],
    dispatch_moved_mwh: np.ndarray | None = None,
) -> pd.DataFrame:
    """Reference ``trading_kpis`` (§5.5, §9.1, dispatch §6.3) from ``trading_week_world``.

    ``dispatch_moved_mwh`` (world,) is ``dispatch_moved_mwh_from``; None
    means the dispatch is off, so the row is 0.  ``erosion`` is
    ``baseline_erosion_from``.
    """

    def col(strategy: str, name: str) -> np.ndarray:
        rows = week.loc[week["strategy"].eq(strategy)].sort_values("world_id")
        return rows[name].to_numpy(dtype=float)

    def margin(strategy: str) -> np.ndarray:
        return sum(
            col(strategy, n)
            for n in (
                "day_ahead_revenue_gbp",
                "intraday_pnl_gbp",
                "trading_cost_gbp",
                "imbalance_gbp",
            )
        )

    pf = margin("perfect_foresight")
    rows = []
    for s in TRADING_STRATEGIES:
        values = {
            "net_gbp_per_week": col(s, "net_gbp"),
            "net_gbp_per_ev_year": col(s, "net_gbp") * 52.0 / vehicle_count,
            "settled_mwh_per_week": col(s, "settled_mwh"),
            "imbalance_volume_share": _safe_ratio(
                col(s, "abs_imbalance_mwh"), col(s, "settled_mwh")
            ),
            "baseline_effect_share": _safe_ratio(
                col(s, "baseline_effect_mwh"), col(s, "settled_mwh")
            ),
            "event_delivered_mwh_per_week": col(s, "event_delivered_mwh"),
            "flex_margin_gbp_per_week": margin(s),
            "value_of_intraday_gbp_per_week": margin("full") - margin("da_only"),
            "cost_of_uncertainty_gbp_per_week": pf - margin(s),
            "imbalance_gbp_per_mwh_traded": _safe_ratio(
                col(s, "imbalance_gbp"), col(s, "final_position_mwh")
            ),
            "imbalance_share_of_traded": _safe_ratio(
                col(s, "abs_imbalance_mwh"), col(s, "final_position_mwh")
            ),
            "firmness": _safe_ratio(
                col(s, "delivered_within_final_mwh"), col(s, "final_position_mwh")
            ),
            "firmness_day_ahead": _safe_ratio(
                col(s, "delivered_within_day_ahead_mwh"), col(s, "sold_day_ahead_mwh")
            ),
            "flex_margin_gbp_per_mw_year": _safe_ratio(margin(s) * 52.0, deferrable_2h_kw / 1000.0),
            "day_ahead_spread_gbp_per_mwh": spread,
            "rebalancing_gbp_per_week": col(s, "intraday_pnl_rebalancing_gbp"),
            "reoptimisation_gbp_per_week": col(s, "intraday_pnl_reoptimisation_gbp"),
            "dispatch_moved_mwh_per_week": (
                np.zeros(len(spread)) if dispatch_moved_mwh is None else dispatch_moved_mwh
            ),
            **erosion,
        }
        for metric, unit in TRADING_KPI_METRICS:
            if metric in TRADING_FULL_ONLY_METRICS and s != "full":
                continue
            if metric == "capture_rate":
                positive = pf > 0.0
                stats = world_stats(margin(s)[positive] / pf[positive])
                stats["mean"] = float(margin(s).mean() / pf.mean()) if pf.mean() > 0.0 else np.nan
                cvar = np.nan
            elif metric in TRADING_EROSION_SHARES:
                default, eroded = (erosion[key] for key in TRADING_EROSION_SHARES[metric])
                positive = default > 0.0
                stats = world_stats(1.0 - eroded[positive] / default[positive])
                stats["mean"] = (
                    float(1.0 - eroded.mean() / default.mean()) if default.mean() > 0.0 else np.nan
                )
                cvar = np.nan
            else:
                stats = world_stats(values[metric])
                cvar = _cvar5(values[metric]) if metric in TRADING_CVAR_METRICS else np.nan
            rows.append({"strategy": s, "metric": metric, "unit": unit, **stats, "cvar5": cvar})
    return typed_frame(pd.DataFrame(rows), TRADING_KPI_COLUMNS)


def shock_attribution_from(week: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in TRADING_STRATEGIES:
        ordered = week.loc[week["strategy"].eq(s)].sort_values("world_id")
        for kind in SHOCK_KINDS:
            for metric, unit, column in (
                ("day_ahead_revenue_gbp", "GBP per week", f"day_ahead_revenue_{kind}_gbp"),
                ("intraday_pnl_gbp", "GBP per week", f"intraday_pnl_{kind}_gbp"),
                ("imbalance_gbp", "GBP per week", f"imbalance_{kind}_gbp"),
                ("settled_mwh", "MWh per week", f"settled_mwh_{kind}"),
            ):
                rows.append(
                    {
                        "strategy": s,
                        "shock_kind": kind,
                        "metric": metric,
                        "unit": unit,
                        **world_stats(ordered[column]),
                    }
                )
    return typed_frame(pd.DataFrame(rows), SHOCK_ATTRIBUTION_COLUMNS)


def _noon_order(index: int) -> int:
    return (index - 24) % 48


def open_position_from(deviation: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    half_hour = slots["local_half_hour"].to_numpy()
    v = _deviation_matrix(deviation, "settled_kwh").astype(float)
    rows = []
    for s in TRADING_STRATEGIES:
        open_kw = (v - _deviation_matrix(deviation, f"position_{s}_kwh").astype(float)) / 0.5
        for metric, values in (
            ("open_position_kw", open_kw),
            ("abs_open_position_kw", np.abs(open_kw)),
        ):
            for h in sorted(range(48), key=_noon_order):
                cells = half_hour == h
                per_world = (
                    values[:, cells].mean(axis=1) if cells.any() else np.full(len(values), np.nan)
                )
                rows.append(
                    {
                        "strategy": s,
                        "metric": metric,
                        "unit": "kW",
                        "local_half_hour": f"{h // 2:02d}:{30 * (h % 2):02d}",
                        "profile_order": _noon_order(h),
                        **world_stats(per_world),
                    }
                )
    return typed_frame(pd.DataFrame(rows), OPEN_POSITION_COLUMNS)


def shape_premium_from(world: pd.DataFrame, forecast_prices: pd.DataFrame) -> pd.DataFrame:
    price = (
        forecast_prices.sort_values(["world_id", "slot_index"])["wholesale_forecast_gbp_per_mwh"]
        .to_numpy(dtype=float)
        .reshape(-1, SLOT_COUNT)
    )
    per_path = {}
    # Shape premium stays normal/selected/difference only (decision 0007
    # keeps the optional "timed" path out of it).
    for path in ("normal", "selected"):
        load = metric_matrix(world, path, "home_import_kwh")
        volume = load.sum(axis=1)
        weighted = _safe_ratio((load * price).sum(axis=1), volume)
        baseload = price.mean(axis=1)
        premium = weighted - baseload
        per_path[path] = dict(
            zip(
                [m for m, _ in SHAPE_PREMIUM_METRICS],
                (weighted, baseload, premium, premium * volume / 1000.0),
                strict=True,
            )
        )
    per_path["difference"] = {
        m: per_path["selected"][m] - per_path["normal"][m] for m, _ in SHAPE_PREMIUM_METRICS
    }
    rows = [
        {
            "path_id": path,
            "metric": metric,
            "unit": unit,
            **world_stats(per_path[path][metric]),
            "evidence_kind": "illustrative_synthetic",
        }
        for path in ("normal", "selected", "difference")
        for metric, unit in SHAPE_PREMIUM_METRICS
    ]
    return typed_frame(pd.DataFrame(rows), SHAPE_PREMIUM_COLUMNS)


def supplier_positions_from(
    deviation: pd.DataFrame,
    forecast_prices: pd.DataFrame,
    slots: pd.DataFrame,
    source: str,
    availability_bands: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """§9.3e positions with the §10.5f columns: net change from ``deviation_world_slot``,
    the 1-h firm figures from ``availability_bands`` (``realised`` ``p10``), NaN without it."""

    def mw(column: str) -> np.ndarray:
        return _deviation_matrix(deviation, column).astype(float) / 0.5 / 1000.0

    price = (
        forecast_prices.sort_values(["world_id", "slot_index"])["wholesale_forecast_gbp_per_mwh"]
        .to_numpy(dtype=float)
        .reshape(-1, SLOT_COUNT)
    )
    london = pd.DatetimeIndex(slots["interval_start_london"])
    frame = pd.DataFrame(
        {
            "slot_index": slots["slot_index"].to_numpy(),
            "interval_start_utc": slots["interval_start_utc"],
            "interval_start_london": slots["interval_start_london"],
            "settlement_date": [f"{t:%Y-%m-%d}" for t in london],
            "settlement_period": ((london - london.normalize()) // HALF_HOUR).to_numpy() + 1,
            "night_index": slots["night_index"].to_numpy(),
        }
    )
    for name, values in (
        ("day_ahead_position_mw", mw("position_da_only_kwh")),
        ("final_position_mw", mw("position_full_kwh")),
        ("day_ahead_price_gbp_per_mwh", price),
    ):
        q = np.quantile(values, (0.1, 0.5, 0.9), axis=0, method="linear")
        for i, name_q in enumerate(("p10", "p50", "p90")):
            frame[f"{name}_{name_q}"] = q[i]
    for name, column in (
        ("baseline", "baseline_kwh"),
        ("metered", "metered_kwh"),
        ("settled", "settled_kwh"),
    ):
        frame[f"{name}_mw_p50"] = np.quantile(mw(column), 0.5, axis=0, method="linear")
    unmanaged = mw("unmanaged_kwh")
    change = mw("metered_kwh") - unmanaged
    frame["unmanaged_mw_p50"] = np.quantile(unmanaged, 0.5, axis=0, method="linear")
    for level, name_q in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90")):
        frame[f"net_change_mw_{name_q}"] = np.quantile(change, level, axis=0, method="linear")
    for direction in ("turn_down", "turn_up"):
        column = f"deliverable_{direction}_1h_mw_p10"
        if availability_bands is None:
            frame[column] = np.nan
            continue
        rows = availability_bands.loc[
            availability_bands["statistic"].eq("realised")
            & availability_bands["direction"].eq(direction)
            & availability_bands["duration_hours"].eq(1.0)
        ].sort_values("slot_index")
        frame[column] = rows["p10"].to_numpy(dtype=float) / 1000.0
    frame["price_curve_source"] = source
    frame["evidence_kind"] = "illustrative_synthetic"
    return typed_frame(frame, SUPPLIER_POSITION_COLUMNS)


def _validate_dispatch_split(result, ledger: pd.DataFrame, updates: pd.DataFrame, by) -> None:
    """The ledger rules of intraday dispatch contract §8 (lane K2).

    Each split pair sums to its bucket; both pairs are 0 on the day-ahead-only
    and perfect-foresight rows; the frozen-book re-run starts from ``q`` (its
    day-ahead row equals ``position_da_only_kwh`` through ``position_kwh``);
    with dispatch off the re-run repeats the actual trades, so
    ``frozen_trade_kwh == trade_kwh`` in every row and re-optimisation is
    exactly 0.  The sampled-world recomputation of rebalancing from
    ``frozen_trade_kwh x price_gbp_per_mwh`` runs with the other buckets.
    """

    for bucket, parts in (
        ("intraday_pnl_gbp", TRADING_DISPATCH_SPLIT[:2]),
        ("trading_cost_gbp", TRADING_DISPATCH_SPLIT[2:]),
    ):
        total = ledger[bucket]
        residual = ledger[list(parts)].sum(axis=1) - total
        assert (residual.abs() <= 1e-9 * np.maximum(1.0, total.abs())).all(), (
            f"{bucket} split sums to the bucket"
        )
    for strategy in ("da_only", "perfect_foresight"):
        assert (by[strategy][list(TRADING_DISPATCH_SPLIT)] == 0.0).all().all(), (
            f"no dispatch split on {strategy}"
        )
    day_ahead = updates.loc[updates["stage"].eq("day_ahead")]
    assert np.array_equal(day_ahead["frozen_position_kwh"], day_ahead["position_kwh"]), (
        "frozen re-run starts from q"
    )
    assert np.array_equal(day_ahead["frozen_trade_kwh"], day_ahead["trade_kwh"]), "frozen DA trade"
    # Whether the run dispatched is read from its frames (present exactly
    # when it did; ``validate_dispatch_frames`` ties them to the switch
    # record), so a direct ``run_forecast`` result without records is
    # judged by what it ran, not by a missing record.
    if getattr(result, "dispatch_world_slot", None) is None:
        assert np.array_equal(updates["frozen_trade_kwh"], updates["trade_kwh"]), (
            "dispatch off: frozen trades are the actual trades"
        )
        assert np.array_equal(updates["frozen_position_kwh"], updates["position_kwh"]), (
            "dispatch off: frozen positions are the actual positions"
        )
        full = by["full"]
        for name in ("intraday_pnl_reoptimisation_gbp", "trading_cost_reoptimisation_gbp"):
            assert (full[name] == 0.0).all(), f"dispatch off: {name} is exactly 0"


def _term(result, name: str) -> float | None:
    for record in result.assumptions or ():
        if record.name == name:
            return float(record.value)
    return None


def _text_term(result, name: str) -> str | None:
    for record in result.assumptions or ():
        if record.name == name:
            return str(record.value)
    return None


FIRM_MW_FIELDS = (
    "world_nights",
    "availability_world_slot",
    "availability_bands",
    "availability_manufacturer_world",
    "firm_share_by_fleet_size",
    "availability_backtest",
    "availability_reliability",
    "availability_backtest_summary",
    "product_sheet",
    "availability_value_summary",
    "settlement_file",
    "settlement_file_name",
    "charge_completion_summary",
    "firmness_by_manufacturer",
    "manufacturer_summary",
)
"""§10.0 result fields that are ``None`` on a no-action result (``blackout_windows`` is not)."""


def _validate_firm_mw(result, world: pd.DataFrame) -> None:
    """§10.9 hooks: slot flags, blackout table and the J2, J3 and J5 lane validators.

    The lane validators live in their own fixture modules (one per lane,
    §10.9) and import this module, so they are imported here, when used.
    A result without the frames (the contract v2 fixture) is checked for the
    slot flags only.
    """

    slots = result.study_slots
    blackout = getattr(result, "blackout_windows", None)
    if blackout is not None:
        labels = {label for row in blackout["slot_labels"] for label in row}
        expected = slots["local_time_label"].isin(labels).to_numpy()
        assert (slots["blackout"].to_numpy() == expected).all(), "study_slots.blackout"
    if result.model != "action":
        for name in FIRM_MW_FIELDS:
            assert getattr(result, name, None) is None, f"{name} must be None on a no-action result"
        return
    # A real action run (its replay state carries the firm-MW inputs) has
    # every frame; the contract v2 fixture has none and is exempt.
    state = getattr(result, "replay_state", None)
    if getattr(state, "availability_inputs", None) is not None:
        for name in FIRM_MW_FIELDS:
            assert getattr(result, name, None) is not None, f"{name} required on an action result"
    if getattr(result, "availability_world_slot", None) is None:
        return
    from fixtures.availability_backtest_contract import validate_availability_backtest
    from fixtures.availability_contract import validate_availability
    from fixtures.product_contract import (
        validate_completion,
        validate_firmness,
        validate_manufacturer_summary,
        validate_position_additions,
        validate_product_sheet,
        validate_settlement_file,
        validate_value_summary,
    )

    summary = result.manufacturer_summary
    assert summary is not None, "manufacturer_summary required with the availability frames"
    makers = summary.loc[summary["manufacturer_id"].ne("fleet")].reset_index(drop=True)
    table = makers.loc[
        :,
        [
            "manufacturer_id",
            "manufacturer_label",
            "share",
            "ev_count",
            "response_rate",
            "outage_probability_per_night",
        ],
    ]
    counts = result.units["manufacturer_id"].value_counts()
    counts = counts.reindex(table["manufacturer_id"], fill_value=0).to_numpy()
    assert (table["ev_count"].to_numpy() == counts).all(), "maker EV counts match the units"
    decision_time = _text_term(result, "availability.intraday_decision_local_time") or "17:00"
    validate_availability(
        result,
        response_rate=table["response_rate"].to_numpy(dtype=float),
        outage_probability=table["outage_probability_per_night"].to_numpy(dtype=float),
        decision_local_time=decision_time,
        home_import_kw=metric_matrix(world, "selected", "home_import_kw"),
    )
    validate_availability_backtest(result)
    world_slot = result.availability_world_slot
    validate_product_sheet(
        result.product_sheet,
        world_slot,
        day_ahead_decision_local_hour=_term(result, "day_ahead_publication_local_hour") or 13.0,
        decision_local_time=decision_time,
    )
    prices = result.forecast_prices.sort_values(["world_id", "slot_index"])
    validate_value_summary(
        result.availability_value_summary,
        world_slot,
        prices["wholesale_forecast_gbp_per_mwh"].to_numpy(dtype=float).reshape(-1, SLOT_COUNT),
    )
    representative = result.representative_world_id
    deviation = result.deviation_world_slot
    ledger = result.trading_ledger_world
    full = ledger.loc[ledger["world_id"].eq(representative) & ledger["strategy"].eq("full")]
    validate_settlement_file(
        result.settlement_file,
        deviation_world=deviation.loc[deviation["world_id"].eq(representative)],
        customer_share=full.sort_values("night_index")["customer_revenue_share_gbp"].to_numpy(),
        vehicle_count=result.vehicle_count,
    )
    assert isinstance(result.settlement_file_name, str), "settlement_file_name"
    completion = result.charge_completion_summary
    validate_completion(completion, list(dict.fromkeys(completion["group_id"])))
    validate_firmness(
        result.firmness_by_manufacturer, result.availability_manufacturer_world, table
    )
    validate_manufacturer_summary(summary, table)
    validate_position_additions(result.supplier_positions, deviation, result.availability_bands)


def _validate_trading(result, world: pd.DataFrame, slots: pd.DataFrame) -> None:
    """Contract checks of the trading overlay frames (trading contract v1 §5.9, §9.11)."""

    worlds = result.world_count
    for name in TRADING_FIELDS:
        assert getattr(result, name, None) is not None, f"{name} required on an action result"
    assert result.price_curve_source in ("synthetic", "user curve"), "price_curve_source"
    deviation = result.deviation_world_slot
    _check_dtypes(deviation, DEVIATION_WORLD_SLOT_COLUMNS, "deviation_world_slot")
    assert len(deviation) == worlds * SLOT_COUNT, "deviation_world_slot is world x slot"
    _check_unique(deviation, ["world_id", "slot_index"], "deviation_world_slot")
    m = {c: _deviation_matrix(deviation, c) for c in DEVIATION_WORLD_SLOT_COLUMNS}
    # §10.4, §10.9: the commitment rule, from the run's record when it has
    # one; a direct run in newsvendor mode shows itself by its commit levels.
    level = m["commit_level"].astype(float)
    rule = _text_term(result, "trading.commitment_rule") or (
        "newsvendor" if np.isfinite(level).any() else "fixed_share"
    )
    assert (np.isnan(level) | ((level >= 0.0) & (level <= 1.0))).all(), "commit_level in [0, 1]"
    if rule == "fixed_share":
        assert np.isnan(level).all(), "commit_level is NaN in fixed-share mode"
        assert np.isnan(m["forecast_error_quantile_kwh"].astype(float)).all(), (
            "forecast_error_quantile_kwh is NaN in fixed-share mode"
        )
    b, u, mt = (m[c].astype(float) for c in ("baseline_kwh", "unmanaged_kwh", "metered_kwh"))
    tol = 1e-9
    assert np.allclose(m["deviation_kwh"], b - mt, atol=tol), "D = B - M"
    assert np.allclose(m["true_reduction_kwh"], u - mt, atol=tol), "R = U - M"
    assert np.allclose(m["baseline_effect_kwh"], b - u, atol=tol), "E = B - U"
    identity = m["deviation_kwh"] - m["true_reduction_kwh"] - m["baseline_effect_kwh"]
    assert np.abs(identity.astype(float)).max() <= tol, "D = R + E"
    for name in ("baseline", "unmanaged", "metered"):
        assert np.allclose(m[f"{name}_kw"], m[f"{name}_kwh"].astype(float) / 0.5), f"{name}_kw"
    settled = m["settled_kwh"].astype(float)
    expected_settled = m["settlement_open"].astype(bool) * np.maximum(
        m["deviation_kwh"].astype(float), 0.0
    )
    assert np.allclose(settled, expected_settled, atol=tol), "V = mask x max(0, D)"
    assert (settled >= 0.0).all(), "settled volume is never negative"
    for s in TRADING_STRATEGIES:
        assert (m[f"position_{s}_kwh"].astype(float) >= 0.0).all(), f"{s} positions >= 0"
    assert (m["expected_metered_kwh"].astype(float) >= 0.0).all(), "expected import >= 0"
    expected_unmanaged = m["expected_unmanaged_kwh"].astype(float)
    assert (expected_unmanaged >= 0.0).all(), "expected unmanaged import >= 0"
    # Supplier contract §2: F = x^U - x moves the same clipped need of every
    # expected session in time, so it sums to 0 over each (world, night).
    turn_down = expected_unmanaged - m["expected_metered_kwh"].astype(float)
    slot_night = slots["night_index"].to_numpy()
    for n in np.unique(slot_night):
        night_sum = turn_down[:, slot_night == n].sum(axis=1)
        assert np.allclose(night_sum, 0.0, atol=1e-9), "sum of x^U - x over a night is 0"
    events = getattr(result, "events", None)
    requests = (
        pd.DataFrame()
        if events is None
        else events.loc[
            events["enabled"].astype(bool) & events["event_type"].isin(("turn_down", "turn_up"))
        ]
    )
    if requests.empty:
        # No grid request: only blackout half-hours (§10.5b) close settlement.
        closed = slots["blackout"].to_numpy(dtype=bool)
        assert (m["settlement_open"].astype(bool) == ~closed).all(), (
            "settlement open outside blackouts without requests"
        )
    assert np.array_equal(m["position_perfect_foresight_kwh"], m["settled_kwh"]), "PF sells V"
    assert np.allclose(u, metric_matrix(world, "normal", "home_import_kwh")), "U is normal import"
    assert np.allclose(mt, metric_matrix(world, "selected", "home_import_kwh")), "M is selected"
    kind = np.where(
        m["surprise_shock_gw"].astype(float) != 0.0,
        "surprise",
        np.where(m["known_shock_gw"].astype(float) != 0.0, "known", "none"),
    )
    assert (m["shock_kind"] == kind).all(), "shock_kind recomputes from the GW columns"

    ledger = result.trading_ledger_world
    _check_dtypes(ledger, TRADING_LEDGER_COLUMNS, "trading_ledger_world")
    nights = sorted(set(slots["night_index"]))
    assert len(ledger) == worlds * 3 * len(nights), "one ledger row per world, strategy, night"
    _check_unique(ledger, ["world_id", "strategy", "night_index"], "trading_ledger_world")
    net = ledger["net_gbp"].to_numpy()
    bucket_sum = ledger[list(TRADING_BUCKETS)].sum(axis=1).to_numpy()
    assert (np.abs(net - bucket_sum) <= 1e-9 * np.maximum(1.0, np.abs(net))).all(), "net = buckets"
    for bucket in ("day_ahead_revenue", "intraday_pnl", "imbalance"):
        parts = ledger[[f"{bucket}_{k}_gbp" for k in SHOCK_KINDS]].sum(axis=1)
        total = ledger[f"{bucket}_gbp"]
        assert (np.abs(parts - total) <= 1e-9 * np.maximum(1.0, total.abs())).all(), (
            f"{bucket} kinds"
        )
    by = {
        s: ledger.loc[ledger["strategy"].eq(s)].sort_values(["world_id", "night_index"])
        for s in TRADING_STRATEGIES
    }
    pf = by["perfect_foresight"]
    assert (pf["imbalance_gbp"] == 0.0).all(), "perfect foresight has zero imbalance exactly"
    for s in ("perfect_foresight", "da_only"):
        assert (by[s]["intraday_pnl_gbp"] == 0.0).all() and (by[s]["trading_cost_gbp"] == 0.0).all()
    for name in (
        "supplier_compensation_gbp",
        "customer_revenue_share_gbp",
        "unmet_charge_penalty_gbp",
    ):
        assert (ledger[name] <= 0.0).all(), f"{name} <= 0"
    assert (ledger["unmet_charge_kwh"] >= 0.0).all(), "unmet_charge_kwh >= 0"
    for name in ("grid_event_payment_gbp", "unmet_charge_penalty_gbp", "unmet_charge_kwh"):
        for s in ("da_only", "full"):
            assert np.array_equal(by[s][name].to_numpy(), pf[name].to_numpy()), (
                f"{name} strategy-free"
            )
    assert np.allclose(
        ledger["flexibility_mwh"] + ledger["baseline_effect_mwh"], ledger["settled_mwh"], atol=1e-9
    ), "flexibility + baseline effect = settled"

    # Every bucket recomputed from the deviation frame; the full strategy's
    # intraday trades are known for the sampled worlds (position_updates).
    updates = result.position_updates
    _check_dtypes(updates, POSITION_UPDATE_COLUMNS, "position_updates")
    _check_unique(updates, ["world_id", "slot_index", "decision_utc"], "position_updates")
    sampled = sorted(set(updates["world_id"]))
    assert set(sampled) <= set(result.sampled_world_ids), "position_updates holds sampled worlds"
    cash = np.full((worlds, SLOT_COUNT), np.nan)
    volume = np.full((worlds, SLOT_COUNT), np.nan)
    # The frozen-book re-run's trades (dispatch §6.2), priced at the row's
    # traded price (B2): the rebalancing half of the split.
    frozen_cash = np.full((worlds, SLOT_COUNT), np.nan)
    frozen_volume = np.full((worlds, SLOT_COUNT), np.nan)
    intraday = updates.loc[updates["stage"].eq("intraday")]
    for w in sampled:
        cash[w] = 0.0
        volume[w] = 0.0
        frozen_cash[w] = 0.0
        frozen_volume[w] = 0.0
        rows = intraday.loc[intraday["world_id"].eq(w)]
        at = rows["slot_index"].to_numpy()
        np.add.at(cash[w], at, (rows["trade_kwh"] * rows["price_gbp_per_mwh"] / 1000.0).to_numpy())
        np.add.at(volume[w], at, rows["trade_kwh"].abs().to_numpy())
        np.add.at(
            frozen_cash[w],
            at,
            (rows["frozen_trade_kwh"] * rows["price_gbp_per_mwh"] / 1000.0).to_numpy(),
        )
        np.add.at(frozen_volume[w], at, rows["frozen_trade_kwh"].abs().to_numpy())
    _validate_dispatch_split(result, ledger, updates, by)
    terms = {name: _term(result, name) for name in TRADING_TERM_NAMES}
    unmet = pf["unmet_charge_kwh"].to_numpy().reshape(worlds, len(nights))
    # Grid-event payment and delivery depend on the zone baselines, which
    # the result does not store: they are checked for sign, for zero without
    # requests and (above) for being strategy-free, then fed to the reference.
    assert (ledger["grid_event_payment_gbp"] >= 0.0).all(), "grid-event payments >= 0"
    if requests.empty:
        assert (ledger["grid_event_payment_gbp"] == 0.0).all(), "no requests, no payment"
        assert (ledger["event_delivered_mwh"] == 0.0).all(), "no requests, no delivery"
    if all(value is not None for value in terms.values()):
        reference = trading_ledger_from(
            deviation,
            slots,
            terms=terms,
            unmet_kwh=unmet,
            intraday_cash=np.nan_to_num(cash),
            intraday_volume=np.nan_to_num(volume),
            event_payment_gbp=pf["grid_event_payment_gbp"].to_numpy().reshape(worlds, len(nights)),
            event_delivered_mwh=pf["event_delivered_mwh"].to_numpy().reshape(worlds, len(nights)),
            rebalancing_cash=np.nan_to_num(frozen_cash),
            rebalancing_volume=np.nan_to_num(frozen_volume),
        )
        known_full = ledger["strategy"].ne("full") | ledger["world_id"].isin(sampled)
        exact = [c for c in TRADING_LEDGER_COLUMNS if TRADING_LEDGER_COLUMNS[c] == "float"]
        independent_of_intraday = [
            c
            for c in exact
            if c
            not in (
                "intraday_pnl_gbp",
                "trading_cost_gbp",
                "customer_revenue_share_gbp",
                "net_gbp",
                "intraday_traded_mwh",
                *(f"intraday_pnl_{k}_gbp" for k in SHOCK_KINDS),
                *TRADING_DISPATCH_SPLIT,
            )
        ]
        _assert_close(ledger, reference, independent_of_intraday, "trading_ledger_world")
        _assert_close(
            ledger.loc[known_full].reset_index(drop=True),
            reference.loc[known_full].reset_index(drop=True),
            exact,
            "trading_ledger_world (sampled full)",
        )
        gross = ledger[[b for b in TRADING_BUCKETS[:7]]].sum(axis=1)
        share = -terms["trading.customer_revenue_share"] * gross.clip(lower=0.0)
        assert np.allclose(ledger["customer_revenue_share_gbp"], share, atol=1e-9), "customer share"
        assert np.allclose(
            ledger["unmet_charge_penalty_gbp"],
            -terms["trading.unmet_charge_penalty_gbp_per_kwh"] * ledger["unmet_charge_kwh"],
        ), "unmet-charge penalty"

    # position_updates (§5.2): DA rows equal q, last rows equal the final full
    # position, trades add up to it, nothing after gate closure.
    for w in sampled:
        rows = updates.loc[updates["world_id"].eq(w)].sort_values(["slot_index", "decision_utc"])
        first = rows.groupby("slot_index").head(1)
        last = rows.groupby("slot_index").tail(1)
        assert (first["stage"] == "day_ahead").all(), "first update per slot is the DA decision"
        assert np.allclose(first["position_kwh"], m["position_da_only_kwh"][w].astype(float)), (
            "DA row = q"
        )
        assert np.allclose(first["trade_kwh"], first["position_kwh"]), "DA trade = q"
        expected = m["expected_metered_kwh"][w].astype(float)
        assert np.allclose(first["forecast_metered_kwh"], expected), "DA forecast = expected import"
        commitment = _term(result, "trading.day_ahead_commitment_share")
        if rule == "fixed_share" and commitment is not None and m["settlement_open"][w].all():
            # q = c x max(0, B0 - x) with no event pausing trading (§4.4, §9.2).
            target = np.maximum(first["baseline_known_kwh"].to_numpy() - expected, 0.0)
            assert np.allclose(first["position_kwh"], commitment * target), "q = c max(0, B0 - x)"
        if rule == "newsvendor" and m["settlement_open"][w].all():
            # q = max(0, F_hat + e) with F_hat = max(0, B0 - x), or 0 where the
            # day-ahead price visible at the decision was at or below zero
            # (§10.4, §10.9).  That visible price can be the expected shape,
            # which the result does not store; such a slot has commit level 0
            # (alpha = 0 when P_DA <= 0), so a 0 position is accepted there only.
            target = np.maximum(first["baseline_known_kwh"].to_numpy() - expected, 0.0)
            error = m["forecast_error_quantile_kwh"][w].astype(float)
            position = first["position_kwh"].to_numpy()
            unpriced = (position == 0.0) & (level[w] == 0.0)
            assert (np.isclose(position, np.maximum(0.0, target + error)) | unpriced).all(), (
                "newsvendor q = max(0, F_hat + e)"
            )
        assert np.allclose(last["position_kwh"], m["position_full_kwh"][w].astype(float)), (
            "last = f"
        )
        total = rows.groupby("slot_index")["trade_kwh"].sum().to_numpy()
        assert np.allclose(total, m["position_full_kwh"][w].astype(float), atol=1e-9), (
            "trades sum to f"
        )
        gate = rows["interval_start_utc"] - pd.Timedelta(minutes=60)
        assert (rows["decision_utc"] <= gate).all(), "no decision after gate closure"
        assert (rows["hours_to_gate_closure"] >= 0).all(), "hours to gate closure"

    week = result.trading_week_world
    _check_dtypes(week, TRADING_WEEK_COLUMNS, "trading_week_world")
    reference_week = trading_week_from(ledger)
    _assert_close(
        week,
        reference_week,
        [c for c, k in TRADING_WEEK_COLUMNS.items() if k == "float"],
        "trading_week_world",
    )

    summary = result.trading_ledger_summary
    _check_dtypes(summary, TRADING_LEDGER_SUMMARY_COLUMNS, "trading_ledger_summary")
    _assert_close(
        summary,
        trading_ledger_summary_from(ledger, week),
        ["world_count", "mean", "p10", "p50", "p90"],
        "trading_ledger_summary",
    )
    week_rows = summary.loc[summary["night_index"].isna()]
    for s in TRADING_STRATEGIES:
        means = week_rows.loc[week_rows["strategy"].eq(s)].set_index("metric")["mean"]
        assert np.isclose(means[list(TRADING_BUCKETS)].sum(), means["net_gbp"], atol=1e-6), (
            "means add"
        )

    kpis = result.trading_kpis
    _check_dtypes(kpis, TRADING_KPI_COLUMNS, "trading_kpis")
    spread = day_ahead_spread_from(result.forecast_prices, slots)
    per_mw = kpis["metric"].eq("flex_margin_gbp_per_mw_year")
    reference_kpis = trading_kpis_from(
        week,
        vehicle_count=result.vehicle_count,
        deferrable_2h_kw=np.ones(worlds),
        spread=spread,
        erosion=baseline_erosion_from(
            deviation,
            slots,
            working_nights=int(_term(result, "trading.baseline_working_nights") or 5),
            non_working_nights=int(_term(result, "trading.baseline_non_working_nights") or 2),
            window_slots=int(_term(result, "trading.baseline_adjustment_window_slots") or 3)
            if _term(result, "trading.baseline_in_day_adjustment")
            else None,
        ),
        dispatch_moved_mwh=dispatch_moved_mwh_from(
            getattr(result, "dispatch_world_slot", None), worlds
        ),
    )
    assert list(zip(kpis["strategy"], kpis["metric"], strict=True)) == list(
        zip(reference_kpis["strategy"], reference_kpis["metric"], strict=True)
    ), "trading_kpis rows"
    # The per-MW-year margin divides by a per-world capacity the result does
    # not store; its row is checked for sign only.
    _assert_close(
        kpis.loc[~per_mw].reset_index(drop=True),
        reference_kpis.loc[~per_mw].reset_index(drop=True),
        ["world_count", "mean", "p10", "p50", "p90", "cvar5"],
        "trading_kpis",
    )
    with_cvar = kpis.dropna(subset=["cvar5", "p10"])
    assert (with_cvar["cvar5"] <= with_cvar["p10"] + 1e-9).all(), "cvar5 <= p10"
    for metric in TRADING_FULL_ONLY_METRICS:
        assert kpis.loc[kpis["metric"].eq(metric), "strategy"].tolist() == ["full"], metric
    # A fleet property of the one dispatched path (dispatch §6.3).
    moved = kpis.loc[kpis["metric"].eq("dispatch_moved_mwh_per_week"), ["mean", "p50"]]
    assert len(moved.drop_duplicates()) == 1, "dispatch_moved_mwh_per_week same on every strategy"
    # Baseline erosion (item 67) is a fleet property too, outside the ledger.
    for metric in TRADING_EROSION_METRICS:
        rows = kpis.loc[kpis["metric"].eq(metric), ["mean", "p50"]]
        assert len(rows.drop_duplicates()) == 1, f"{metric} same on every strategy"
    settled_mwh = kpis.loc[kpis["metric"].eq("settled_mwh_eroded_per_week"), "mean"]
    assert (settled_mwh >= 0.0).all(), "settled_mwh_eroded_per_week >= 0"
    for metric in ("firmness", "firmness_day_ahead"):
        rows = kpis.loc[kpis["metric"].eq(metric)]
        assert (rows[["p10", "p50", "p90"]].stack() <= 1.0 + 1e-9).all(), f"{metric} <= 1"
    spreads = kpis.loc[kpis["metric"].eq("day_ahead_spread_gbp_per_mwh"), "mean"]
    assert spreads.nunique() == 1 and (spreads >= 0.0).all(), "one day-ahead spread"

    checks = result.trading_checks
    _check_dtypes(checks, TRADING_CHECK_COLUMNS, "trading_checks")
    assert tuple(checks["check_id"]) == TRADING_CHECK_IDS, "trading_checks rows"
    assert checks["passed"].all(), "every trading check passes"

    attribution = result.shock_attribution_summary
    _check_dtypes(attribution, SHOCK_ATTRIBUTION_COLUMNS, "shock_attribution_summary")
    _assert_close(
        attribution,
        shock_attribution_from(week),
        ["world_count", "mean", "p10", "p50", "p90"],
        "shock_attribution_summary",
    )

    profile = result.open_position_profile
    _check_dtypes(profile, OPEN_POSITION_COLUMNS, "open_position_profile")
    _assert_close(
        profile,
        open_position_from(deviation, slots),
        ["world_count", "mean", "p10", "p50", "p90"],
        "open_position_profile",
    )
    pf_profile = profile.loc[
        profile["strategy"].eq("perfect_foresight"), ["mean", "p10", "p50", "p90"]
    ]
    assert (pf_profile.fillna(0.0) == 0.0).all().all(), "perfect foresight has no open position"

    shape = result.shape_premium_summary
    _check_dtypes(shape, SHAPE_PREMIUM_COLUMNS, "shape_premium_summary")
    _assert_close(
        shape,
        shape_premium_from(world, result.forecast_prices),
        ["world_count", "mean", "p10", "p50", "p90"],
        "shape_premium_summary",
    )
    # Identity with the cost effect (§9.3b): the shape-cost difference is the
    # home-import cost effect minus the baseload value of the volume change.
    cost = result.cost_effect.sort_values("world_id")
    price = (
        result.forecast_prices.sort_values(["world_id", "slot_index"])[
            "wholesale_forecast_gbp_per_mwh"
        ]
        .to_numpy(dtype=float)
        .reshape(-1, SLOT_COUNT)
    )
    volume_change = (
        metric_matrix(world, "selected", "home_import_kwh")
        - metric_matrix(world, "normal", "home_import_kwh")
    ).sum(axis=1)
    shape_cost_difference = (
        cost["illustrative_selected_minus_normal_energy_cost_gbp"].to_numpy()
        - price.mean(axis=1) * volume_change / 1000.0
    )
    difference_mean = shape.loc[
        shape["path_id"].eq("difference") & shape["metric"].eq("shape_cost_gbp_per_week"), "mean"
    ].iat[0]
    assert np.isclose(difference_mean, shape_cost_difference.mean(), atol=1e-6), (
        "shape cost identity"
    )

    positions = result.supplier_positions
    _check_dtypes(positions, SUPPLIER_POSITION_COLUMNS, "supplier_positions")
    _assert_close(
        positions,
        supplier_positions_from(
            deviation,
            result.forecast_prices,
            slots,
            result.price_curve_source,
            getattr(result, "availability_bands", None),
        ),
        [c for c, k in SUPPLIER_POSITION_COLUMNS.items() if k in ("float", "int")],
        "supplier_positions",
    )
    for _, periods in positions.groupby("settlement_date")["settlement_period"]:
        values = periods.to_numpy()
        assert (np.diff(values) == 1).all(), "settlement periods count up by one"

    curve = result.flex_cost_curve
    _check_dtypes(curve, FLEX_COST_CURVE_COLUMNS, "flex_cost_curve")
    for (_, _), block in curve.groupby(["day_type", "local_half_hour"], sort=False):
        available = block.loc[block["metric"].eq("available_kw")].sort_values(
            "threshold_gbp_per_mwh"
        )
        assert np.allclose(available["threshold_gbp_per_mwh"], COST_CURVE_THRESHOLDS), (
            "cost curve grid"
        )
        for stat in ("mean", "p10", "p50", "p90"):
            values = available[stat].to_numpy()
            assert (np.diff(values[~np.isnan(values)]) >= -1e-9).all(), (
                "available kW rises with price"
            )
        movable = block.loc[block["metric"].eq("movable_kw"), "mean"].iat[0]
        charging = block.loc[block["metric"].eq("charging_kw"), "mean"].iat[0]
        if not np.isnan(movable):
            assert available["mean"].max() <= movable + 1e-9 <= charging + 2e-9, (
                "available <= movable <= charging"
            )

    distribution = result.household_value_distribution
    _check_dtypes(distribution, HOUSEHOLD_DISTRIBUTION_COLUMNS, "household_value_distribution")
    for _, group in distribution.groupby("group_id", sort=False):
        assert len(group) == len(HOUSEHOLD_BIN_EDGES) + 1, "household bins"
        assert np.isclose(group["mean"].sum(), 1.0, atol=1e-9), "mean bin shares sum to 1"
    household = result.household_value_summary
    _check_dtypes(household, HOUSEHOLD_SUMMARY_COLUMNS, "household_value_summary")
    for _, group in household.groupby("group_id", sort=False):
        assert tuple(group["statistic"]) == HOUSEHOLD_STATISTICS, "household statistics"
        means = group.set_index("statistic")["mean"]
        assert np.isclose(
            means["mean_value"],
            means["mean_customer_saving"] + means["mean_revenue_share"],
            atol=1e-6,
        ), "value = saving + revenue share"

    segments = result.revenue_by_segment
    _check_dtypes(segments, REVENUE_SEGMENT_COLUMNS, "revenue_by_segment")
    full_week = week.loc[week["strategy"].eq("full")]
    weekly = segments.loc[segments["metric"].eq("gbp_per_week")]
    for bucket in TRADING_MONEY:
        rows = weekly.loc[weekly["bucket"].eq(bucket)]
        all_mean = rows.loc[rows["segment_type"].eq("all"), "mean"].iat[0]
        assert np.isclose(all_mean, full_week[bucket].mean(), atol=1e-6), f"{bucket} all segment"
        for segment_type in ("cohort", "zone"):
            part = rows.loc[rows["segment_type"].eq(segment_type), "mean"]
            if len(part):
                assert np.isclose(part.sum(), all_mean, atol=1e-6), f"{bucket} {segment_type} sum"

    control = getattr(result, "control_group_summary", None)
    if control is not None:
        _check_dtypes(control, CONTROL_GROUP_COLUMNS, "control_group_summary")
        for _, group in control.groupby("night_index", dropna=False, sort=False):
            means = group.set_index("metric")["mean"]
            if not means.isna().any():
                assert np.isclose(
                    means["estimation_error_kw_per_ev"],
                    means["estimated_bias_kw_per_ev"] - means["true_bias_kw_per_ev"],
                    atol=1e-9,
                ), "error = estimated - true"
