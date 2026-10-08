# Result contract v2

Status: draft for lead acceptance, 28 September 2026 (completion plan task C1), amended after review (B1–B4, Q1–Q2), then amended for decision 0004 item 38 (smart charging). Authority: decision 0004 items 1–38 (32 and 37: EVs never strand, they top up in public; 38: price-optimised smart charging replaces the 18:00 import cap of items 2, 8, 25 and 30, its per-EV eligibility screen, the planning world and the select-or-not decision), `docs/DASHBOARD_DESIGN.md` (streamlining pass and section 5), chart audit `docs/reviews/2026-09-28-chart-audit.md` (B1–B5, O1–O7). Base: `61c3fc6`; item 38 amendment on `250938f`.

**Item 38 change log.** Removed: `planning_world_count` (result field and settings key), `action_decision`, `action_eligibility`, `selected_schedule`, the `not_selected` action status, the ceiling and eligibility fields of `ActionSummary`, and the replay's `authorised_import_ceiling_kw`. Added: `early_departure_count` and `early_departure_shortfall_kwh` on `fleet_world_intervals` (3.4), a smart-charging `ActionSummary` (4.3), `smart_charging_world` (4.4), `smart_charging_summary` (4.5), `price_band_shift` (4.6) and the replay's `early_departure_shortfall_kwh` (5). Both models now draw the same random worlds for one seed, so a model change alone no longer unpairs Compare (6).

**Item 52 changes (29 September 2026, plan A4, A5, B4).** The study runs from London 12:00 for seven session nights after a seven-day warm-up with prices (editable 3–14 days). `study_slots` gains `night_index`; `local_date`, `day_type` and `day_label` now name the session night (London time − 12 h), so every day-keyed frame (`price_band_shift`, plug-in KPIs and heatmap by day type, `session_distribution_bands` (3.10c) day types, the average-day day types, `daily_audit`, 4.7a, 4.7b) keys on nights; the half-hour plug heatmaps (3.10b) stay on the calendar weekday with per-cell slot counts. The shock table, the intraday path and the price frames on `SimulatedForecast.market_prices` cover warm-up and study; the result's `price_shocks` keeps study shocks with study slot indices. `forecast_available_at_utc` is per slot: 13:00 London on the day before delivery. The item 49 study-end rule is retired.

**Item 54 additions (29 September 2026, plan C2, C4, D-1, D-3, E1).** Added, no existing column changed: `deferrable_power_bands` (3.6d), `flexibility_weekly_summary` (3.6e), `plug_in_half_hour_heatmap` and `plug_out_heatmap` (3.10b), the coincidence-factor columns of `weekly_peak_summary` (4.5a), `price_relative_bands` (4.7a) and `cheapest_half_hour_summary` (4.7b). Plan D-2 and D-5 add `session_distribution_bands` (3.10c) and the twelve per-archetype source reference records (8.2).

This is the final shape the dashboard reads after model tasks M4 (one runner, one result type) and M5 (summaries). UI tasks U1–U4 build against the synthetic fixture in `tests/fixtures/result_fixture.py`; M5 must make a real result pass the same validator, `validate_result_v2(result)`, in a real-run test before U5 switches the views to real results.

Final module and type names (M4 and M5 implement them; the fixture mirrors them with a `Fixture` prefix):

| Name | Module | Owner |
| --- | --- | --- |
| `ForecastResult`, `PlugInSummary`, `ActionSummary` | `axle_studio.model.forecast` | M4 (type), M5 (summary fields) |
| `run_forecast(...)` | `axle_studio.model.forecast` (the one runner; inputs are M4's choice) | M4 |
| `Assumption`, the default records | `axle_studio.model.assumptions` | M3 |
| `replay_one_ev`, `replay_one_ev_bands`, `OneEvReplay` | `axle_studio.model.individual` | M5 |
| `compare_runs`, `RunComparison`, the summary builders | `axle_studio.model.summaries` | M5 |

`run_forecast` in `model/forecast.py` builds the physical result and then calls `build_summaries` in `model/summaries.py` for the section 3 summary frames (`cohort_summary`, `average_day_bands`, `weekly_bands`, `plug_in_events`, `sampled_world_ids`, `plug_in_world_kpis`, `plug_in_summary`) and the section 4 derived frames (`difference_*`, `action_summary`, `cost_effect_summary`). Both are built (M4 and M5 are done), and a real result passes `validate_result_v2`.

Status marks: **EXISTS** means the value is on today's result (current name given); **NEW (Mx)** means task Mx adds it. "Rename" means the value exists under another name or shape and M5 reshapes it.

## 1. Rules that apply to every field

1. **One result type.** `ForecastResult` (module `axle_studio.model.forecast`, created by M4) replaces the two earlier result types, `ExplicitNoActionForecastResult` and `ExplicitActionForecastResult`, whose modules were removed. Field `model` says which model ran. Action-only fields are `None` on a no-action result, never zero-filled or duplicated. Smart charging (decision 0004 item 38) applies to every EV on the selected path, so an action run always has `action_status="selected"`; the earlier "no action selected" state (lead decision Q2) no longer arises.
2. **Paths.** `path_id` is `"normal"` (no Axle action: charge as soon as plugged in) or `"selected"` (smart charging, decision 0004 item 38; once intraday-dispatch-v1 lands, the dispatched fleet: day-ahead plans for locked EVs, hourly re-plans on the latest intraday prices for free EVs). Action results carry both; no-action results carry `"normal"` only. Today's no-action runner duplicates the normal path as `"selected"`; M5 drops the duplicate so no chart can show a fake selected path. An optional third path, `"timed"` (decision 0007), is the unmanaged rule with home charging barred outside an editable daily window (`timed_start_local_hour`, London clock hours; both models), present when the "Timed tariff policy" switch (`timed_tariff_enabled`) is on (the default) at the set start time, absent when off. It is reported in `fleet_world_intervals`, `cohort_world_intervals`, `fleet_interval_bands`, `cohort_interval_bands`, and, from model step 2 (decision 0007), also in `zone_world_intervals`/`zone_import_bands`/`zone_summary`, `charge_completion_summary`, `session_distribution_bands` (a `departure_soc_percent_timed` metric), `smart_charging_world`/`smart_charging_summary`/`price_band_shift`/`weekly_peak_summary`, `cost_effect`/`cost_effect_summary`/`not_recovered_summary`'s siblings (`timed_cost_effect` etc., §4.8a), `household_ev_world`/`household_outcomes_summary` and the one-EV replay (§5). Every other frame stays `"normal"`/`"selected"` only, named explicitly in the code rather than discovered by iterating whatever paths a frame happens to carry.
3. **Worlds first.** Every P10/P50/P90 "across worlds" is `numpy.quantile(..., method="linear")` over per-world values that were already aggregated inside each world. Differences are taken per world before quantiles. The UI never subtracts, averages or sums percentiles; it plots columns the model supplies. Component quantiles in a table do not add up to the total's quantile, and the UI says so.
4. **Time.** Stored instants are timezone-aware UTC (`datetime64[ns, UTC]`). Every per-slot frame also carries `interval_start_london` (`datetime64[ns, Europe/London]`), and `study_slots` carries display labels, so the UI plots London time (item 27) and keeps UTC for hover and tables. The study is 336 fixed UTC half-hour slots starting at London 12:00 on the start date (items 1 and 52); a clock change inside the span is allowed, so a session night may hold 46, 48 or 50 slots, and the horizon then ends at 13:00 (spring) or 11:00 (autumn) London. Reporting days are **session nights**: London 12:00 to 12:00, dated by the evening (London time − 12 h), so a night's evening peak and its overnight charging share one day key.
5. **Units are in the name or the `unit` column.**
   - `*_kw` is average power over the half-hour (kW). `*_kwh` is energy in that half-hour (kWh per half-hour) unless the column says `weekly`/`total`.
   - `*_share` is a fraction in [0, 1]. `*_percent` is 0–100. SoC is always `*_soc_percent` (0–100). Counts are integers named `*_count`.
   - This fixes the `*_fraction`-holds-counts problem: today's `driving_fraction`, `away_on_trip_fraction` and `public_charging_fraction` in fleet frames are sums of per-EV time fractions (EV-equivalents). The contract has `driving_share`, `away_share` and `public_charging_share`, each divided by `unit_count`.
   - Money is `*_gbp`, always illustrative (decisions 0003, 0004 item 4); no field is Axle cash, a bid or a settlement.
6. **Evidence.** Every frame that holds model values has an `evidence_kind` column or the result's `evidence_kind` applies. Values: `"illustrative"` (no-action), `"illustrative_synthetic"` (action), `"synthetic"` (prices).
7. **Dtypes.** Integers `int64`; floats `float64`; flags `bool`; text `object` (Python `str`); dates `object` (`datetime.date`); nullable integer `Int64` only where stated. Frames have a default `RangeIndex`; keys are columns, unique as stated.
8. **Missing is missing.** NaN (floats), NaT (times) and `<NA>` (`Int64`) mean "not available" (for example an archetype with no sampled EVs, a session still plugged at the horizon end, a half-hour that does not exist on a clock-change day). The UI shows "Unavailable", never 0 (design, Number formatting). Zero means a real zero.
9. **Category order.** Rows and legends follow these orders: `path_id` normal, selected, then the optional `timed` (decision 0007, rule 2) last when present; `day_type` weekday, weekend, all; `metric` as listed in 3.5; archetypes in source order `average_uk`, `intelligent_octopus`, `infrequent_charging`, `infrequent_driving`, `scheduled_charging`, `always_plugged_in`; `location` home_plugged, home_unplugged, driving, away, public_charging. Categorical columns are plain `object` strings, not pandas `Categorical`; the order constants live in `model/forecast.py`.
10. **Extra columns.** Frames marked *exact* have exactly the listed columns in the listed order. Frames marked *at least* may carry extra model audit columns (for example `conservation_residual_kwh`), which the UI must not read.

## 2. `ForecastResult` scalar fields

| Field | Type | Meaning | Status |
| --- | --- | --- | --- |
| `model` | `"action"` \| `"no_action"` | Which model ran | NEW (M4) |
| `model_label` | str | `"Explicit wholesale action"` or `"Explicit no-action"` | NEW (M4) |
| `policy_id` | str | `"explicit_synthetic_wholesale_v1"` or `"explicit_no_action_v1"` | EXISTS `policy_id` |
| `evidence_kind` | str | `"illustrative_synthetic"` or `"illustrative"` | EXISTS `evidence_kind` |
| `run_completed_at_utc` | `pd.Timestamp` (UTC) | When the run finished; run history orders by it | NEW (M4) |
| `seed` | int | The one `default_rng(seed)` seed | EXISTS `settings.seed` |
| `vehicle_count` | int | EVs simulated | EXISTS `settings.vehicle_count` |
| `world_count` | int | Evaluation worlds (simulated weeks) | EXISTS `settings.evaluation_world_count` |
| `study_start_local_date` | `datetime.date` | London date of the first study slot | EXISTS `settings.start_local_date` |
| `horizon_start_utc` | `pd.Timestamp` (UTC) | Start of slot 0 = London 12:00 on `study_start_local_date` (item 52) | NEW (M5) |
| `horizon_end_utc` | `pd.Timestamp` (UTC) | End of slot 335 = start + 168 h | NEW (M5) |
| `study_slot_count` | int | Always 336 | NEW (M5) |
| `has_clock_change` | bool | True when the London offset changes inside the horizon | NEW (M5) |
| `settings_snapshot` | `dict[str, object]` | Run controls, exactly these keys: `model` (str), `start_local_date` (`date`), `warmup_days`, `study_days`, `vehicle_count`, `seed`, `evaluation_world_count` (int). Plain values, so Compare diffs two snapshots with `==` | EXISTS as `settings` (typed `RunSettings`); NEW plain dict (M3); `planning_world_count` removed (item 38) |
| `assumptions` | `tuple[Assumption, ...]` | Every assumption value used by this run (section 8) | NEW (M3) |
| `action_status` | `"selected"` \| `"not_applicable"` | `selected` on every action result (smart charging applies to every EV, item 38); `not_applicable` for no-action | Amended (item 38) |
| `public_charging_status` | `"modelled"` | Public top-ups (decision 0004 item 32) are modelled in both models | EXISTS on no-action as `public_charging_status="unavailable_unmodelled"`; NEW values (M5) |
| `not_recovered_world_count` | int \| None | Worlds with `not_recovered_material` true (materially not recovered, decision 0004 item 45), not every world with `energy_not_recovered`; `None` for no-action | NEW (M5), amended (item 45) |
| `timed_not_recovered_world_count` | int \| None | Exactly `not_recovered_world_count`'s rule, counted from `timed_cost_effect` (§4.8a) instead of `cost_effect`; `None` whenever `timed_cost_effect` is (no-action result, or `timed_start_local_hour` unset) | NEW (decision 0007, model step 2) |
| `representative_world_id` | int | World whose weekly normal-path home import is the lower median (rank `(n−1)//2`, ties by `world_id`); One EV default week | NEW (M5) |
| `sampled_world_ids` | `tuple[int, ...]` | Worlds held in `plug_in_events`: at most 10, `representative_world_id` first, then the lowest other ids (Q1) | NEW (M5) |
| `replay_state` | opaque | Whatever the model needs to replay one EV (units, sampled inputs, the smart charger's decision-time inputs). The UI never reads it | NEW (M5); smart-charging inputs replace the ceiling (item 38) |

The typed settings object and sampled inputs (`evaluation_inputs`, temperatures, efficiencies, `trip_behaviour`, CV mappings) move into `replay_state` or are dropped by M4. The UI reads none of them. There is no planning world (item 38).

## 3. Frames on every result

### 3.1 `study_slots` (exact) — NEW (M5)

One row per study slot; the only place the UI gets display labels.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `slot_index` | int64 | 0–335 |
| `interval_start_utc` | datetime64[ns, UTC] | Slot start |
| `interval_end_utc` | datetime64[ns, UTC] | Slot start + 30 min |
| `interval_start_london` | datetime64[ns, Europe/London] | Same instant in London time |
| `local_date` | object (`date`) | Session night (item 52): the London date of the slot start minus 12 h, so Monday 12:00 to Tuesday 11:30 is Monday. The one or two slots a spring change pushes past the last noon count to the last night |
| `night_index` | int64 | Night 0–6: `local_date` minus `study_start_local_date` in days |
| `day_type` | object | `"weekday"` or `"weekend"` of `local_date` (the night's evening), so Friday night's small hours are a weekday |
| `local_half_hour` | int64 | 0–47 from London wall-clock time; the repeated autumn hour maps both copies to the same index |
| `local_time_label` | object | `"18:30"` |
| `day_label` | object | `"Mon 28"`, of `local_date` (the night's evening date) |

### 3.2 `units` (at least) — EXISTS `units`, NEW columns (M5)

One row per EV. Consumer: One EV selector and traits caption.

| Column | Dtype | Meaning | Status |
| --- | --- | --- | --- |
| `unit_id` | object | Unique EV id | EXISTS |
| `cohort_id` | object | One of the six archetype ids | EXISTS |
| `cohort_label` | object | Display name, e.g. `"Average (UK)"` | Rename of `cohort_source_name` |
| `daily_miles_mean` | float64 | Mean daily miles (UI shows ×365 as miles/yr) | EXISTS |
| `physical_capacity_kwh` | float64 | Battery capacity, kWh | EXISTS |
| `home_charger_limit_kw` | float64 | Home charger limit, kW | EXISTS |
| `preferred_target_soc_percent` | float64 | Preferred target SoC, 0–100 (not the physical ceiling) | Rename of `preferred_target_soc_fraction` ×100 |

Model columns kept as extras include the home connection window (decision 0004 item 42): `runtime_arrival_local_hour` and `runtime_departure_local_hour` (weekday London clock hours) and `runtime_weekend_arrival_local_hour` and `runtime_weekend_departure_local_hour` (weekend). A session runs from the arrival on one London date to the departure on the next, each end taking the window of its own date; Average UK and Intelligent Octopus have separate weekend windows (17:00 → 10:00, an illustrative adaptation of CNZ p.11 Fig.4 and p.12 Fig.5, records `<cohort>.weekend_connection_*_local_hour` and `weekend_sd_minutes.<cohort>`); the other cohorts are flat. The smart charger's expected departure uses the departure date's window.

### 3.3 `cohort_summary` (exact) — NEW (M5)

Always six rows, one per archetype, in source order, including archetypes with no sampled EVs (audit B1: 50 EVs can sample five cohorts). Consumer: Drivers ▸ Archetypes table; Overview empty state shares.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `cohort_id` | object | Archetype id |
| `cohort_label` | object | Display name |
| `source_population_share` | float64 | Source share, fraction (0.40, 0.30, 0.10, 0.10, 0.09, 0.01) |
| `ev_count` | int64 | EVs of this archetype in the run; may be 0 |
| `plug_ins_per_ev_week_p50` | float64 | From `plug_in_world_kpis` (group = this archetype, day type `all`): P50 across worlds of `plug_ins_per_ev_per_week`; NaN if `ev_count` is 0 |
| `median_plug_in_soc_percent_p50` | float64 | Same source: P50 across worlds of `median_plug_in_soc_percent`, worlds with no plug-ins left out; NaN if none |
| `peak_plug_in_local_hour` | Int64 | Same source: the most common per-world `modal_plug_in_local_hour` (ties go to the earlier hour); `<NA>` if none |
| `unserved_travel_kwh_p50` | float64 | Per world, weekly normal-path unserved travel (battery-side kWh); P50 across worlds; NaN if `ev_count` is 0 |

Views list rows with `ev_count == 0` as "no EVs sampled" and draw panels only for the rest.

### 3.4 `fleet_world_intervals` (at least) — EXISTS, renamed columns (M5)

One row per `(world_id, path_id, slot_index)`. World-level fleet sums; the input for every across-world quantile. Consumer: model functions (bands, difference bands, compare) and Data expanders; charts read the band frames. Carries the optional `timed` path (decision 0007) when `timed_tariff_enabled` is on; every consumer that reads this frame for a pairwise or per-EV result (cost, savings, trading, zones, the chunked summaries) names its two paths explicitly rather than reading whatever is present, so `timed` never reaches them.

| Column | Dtype | Unit | Meaning | Status |
| --- | --- | --- | --- | --- |
| `world_id` | int64 | | 0 … `world_count`−1 | EXISTS |
| `path_id` | object | | `normal`/`selected`, plus optional `timed` | EXISTS |
| `slot_index` | int64 | | 0–335 | NEW |
| `interval_start_utc`, `interval_end_utc` | datetime64[ns, UTC] | | Slot bounds | EXISTS |
| `interval_start_london` | datetime64[ns, Europe/London] | | Display time | NEW |
| `unit_count` | int64 | EVs | Fleet size | EXISTS |
| `physical_capacity_kwh` | float64 | kWh | Fleet battery capacity | EXISTS |
| *fleet metrics* | see 3.5 | | One column per metric in the table below | see 3.5 |
| `early_departure_count` | int64 | sessions | Selected path: home sessions whose EV left (unplugged or drove off) in this half-hour while its smart plan still owed more than 1e-6 kWh. Always 0 on the normal path, which has no plans | NEW (item 38) |
| `early_departure_shortfall_kwh` | float64 | kWh | Battery-side energy those plans still owed (planned grid kWh not delivered × home charging efficiency); 0 on the normal path | NEW (item 38) |

The two early-departure columns are sums over EVs of the kernel's per-EV values, so they add across EVs like the flows. A session too short for any charging rule to reach the target also counts: its plan was unfinished when it left, even if the plan was "as soon as possible".

### 3.5 Fleet metrics

The same names are columns in `fleet_world_intervals` and values of `metric` in the band frames.

| Metric | Unit (`unit` value) | Meaning | Status |
| --- | --- | --- | --- |
| `connected_count` | `EVs` | EVs plugged in at home for the whole half-hour | EXISTS `connected_count` |
| `connected_share` | `fraction` | `connected_count / unit_count` | EXISTS `connected_share` |
| `home_import_kwh` | `kWh per half-hour` | Home grid import | EXISTS `home_grid_import_kwh` |
| `home_import_kw` | `kW` | `home_import_kwh / 0.5` | NEW |
| `public_import_kwh` | `kWh per half-hour` | Destination public grid import | EXISTS `public_grid_import_kwh` |
| `total_import_kw` | `kW` | Home + public grid import as average power; labelled "home + public grid import" (audit O7) | Rename of `realised_grid_kw` |
| `closing_battery_kwh` | `kWh` | Fleet battery stock at slot end | EXISTS |
| `battery_soc_percent` | `percent` | Capacity-weighted fleet SoC at slot end | Rename of `closing_soc_percent` |
| `driving_share` | `fraction` | Share of EV-time driving | Rename of `driving_fraction ÷ unit_count` |
| `away_share` | `fraction` | Share of EV-time away from home (driving, parked or public charging) | Rename of `away_on_trip_fraction ÷ unit_count` |
| `public_charging_share` | `fraction` | Share of EV-time at a public charger | Rename of `public_charging_fraction ÷ unit_count` |
| `unserved_travel_kwh` | `kWh per half-hour` | Battery-side travel energy not served (outbound + return) | NEW sum of `unserved_*_travel_battery_kwh` |

### 3.6 `fleet_interval_bands` (exact) — EXISTS, reshaped long (M5)

One row per `(metric, path_id, slot_index)`, sorted by those keys. Both paths on action results, plus the optional `timed` path (decision 0007) when the policy is on. Consumers: Drivers ▸ Fleet week (normal path), Smart charging ▸ Response (both paths, and the third `timed` line when present, `action_response._paths_figure`).

| Column | Dtype | Meaning |
| --- | --- | --- |
| `metric` | object | Fleet metric name (3.5) |
| `unit` | object | Unit string from 3.5 |
| `path_id` | object | `normal`/`selected`, plus optional `timed` |
| `slot_index` | int64 | 0–335 |
| `interval_start_utc`, `interval_end_utc` | datetime64[ns, UTC] | Slot bounds |
| `interval_start_london` | datetime64[ns, Europe/London] | Display time |
| `world_count` | int64 | Worlds summarised |
| `mean`, `p10`, `p50`, `p90` | float64 | Across worlds of the `fleet_world_intervals` column; `p10 ≤ p50 ≤ p90` |

Today this frame is wide (`<metric>_<stat>`) with the old names; the long shape lets one metric control drive any chart.

### 3.6a `weekly_bands` (exact) — NEW (M5), every result

One row per `(path_id, metric)`; columns `path_id`, `metric`, `unit` (`kWh per week`), `world_count`, `mean`, `p10`, `p50`, `p90`, `max`. Per world, the weekly sum of the `fleet_world_intervals` column; then statistics across worlds. Metrics: `home_import_kwh`, `public_import_kwh`, `unserved_travel_kwh`. Consumer: no-action Overview "Unserved travel (P50 kWh)" tile; Fleet week caption ("Unserved travel: 0 kWh in all weeks" uses `max`).

### 3.6b `fleet_interval_ev_bands` (exact) — NEW (decision 0004 items 39, 40), every result

Spread **across EVs** for each half-hour, for Fleet week. One row per `(metric, path_id, slot_index)`, metrics in the order below, paths `normal` then `selected` (action results only). Columns `metric`, `unit`, `path_id`, `slot_index`, `interval_start_utc`, `interval_end_utc`, `interval_start_london`, `world_count`, `p10`, `p50`, `p90`.

Computation, which the caption must state: in each simulated week (world) and half-hour, the 10th, 50th and 90th percentile **across EVs** of each EV's own value (`numpy.quantile(..., method="linear")` over the fleet's EVs); then each of those three is replaced by its **median across simulated weeks**. So `p10` is "the typical week's 10th-percentile EV", not a forecast bound. The median keeps order, so `p10 ≤ p50 ≤ p90` holds.

| `metric` | `unit` | Per-EV value |
| --- | --- | --- |
| `battery_soc_percent` | `percent` | Closing SoC of the EV at the slot end, 0–100 |
| `home_import_kw` | `kW` | The EV's home grid import in the slot ÷ 0.5 h |

This frame is added alongside the across-weeks bands, not instead of them: `fleet_interval_bands` (3.6) keeps the P10–P90 of the fleet total across simulated weeks, and every time-series chart still shows it (item 40). Plugged-in share per archetype per slot is already `cohort_interval_bands` metric `connected_share` (3.7; `p50` is the median across weeks); the fleet line is `fleet_interval_bands` `connected_share`. It is computed in `model/summaries.py` from the per-EV chunks the summaries already run (at most 10 worlds at a time), so it adds no random draw.

### 3.6c `flexibility_bands` (exact) — NEW (decision 0004 item 43), every result

How much home charging could be moved, for each half-hour: Drivers ▸ Fleet week (turn-up headroom and movable energy; the Flexibility lens and the Overview flexible-power chart were retired by decision 0004 item 54). Row groups in the order below, each with paths `normal` then `selected` (action results only) and one row per slot. Columns `metric`, `unit`, `spread`, `path_id`, `slot_index`, `interval_start_utc`, `interval_end_utc`, `interval_start_london`, `world_count` (int64), `p10`, `p50`, `p90` (float64). Keys are `(metric, spread, path_id, slot_index)`. `None` only on a no-action result from a direct `run_forecast` call given no `departure_margin_hours` (the expected departure needs it); `run_forecast_from_assumptions`, and so the app, always gives one.

Every metric counts only EVs **plugged in at home for the whole half-hour** (the kernel's condition for home charging) and reads each EV's battery stock at the start of the half-hour. An EV **can charge** when it is plugged in and below its preferred target; the kernel never charges an EV at or above target. The **expected departure** is the smart charger's (item 38): the EV's typical departure clock time for that date's weekday or weekend window (item 42) minus the departure margin, taken as the first one at least one half-hour after the slot start.

| `metric` | `unit` | `spread` | Per-world value in each half-hour |
| --- | --- | --- | --- |
| `flexible_power_kw` | `kW` | `across_weeks` | **Turn-up ceiling**: the most the EVs that can charge could draw together, the sum of their home charging power. An EV close to target counts its full power (an instantaneous ceiling; `movable_energy_kwh` says how long it lasts). |
| `turn_up_headroom_kw` | `kW` | `across_weeks` | `flexible_power_kw` minus the path's own fleet home import in the slot (kW): ceiling still unused. Never negative (clipped at 0 against floating-point rounding). |
| `movable_energy_kwh` | `kWh` | `across_weeks` | Grid-side energy still needed to reach the preferred target, `(target − stock) ÷ home charge efficiency`, summed over the EVs that can charge: the energy that must be delivered before the expected departure. |
| `time_slack_hours` | `hours` | `across_evs` | Per EV that can charge: hours from the slot start to its expected departure minus hours needed to reach target at full power (`grid kWh needed ÷ home charging power`). Zero = must charge now; negative = cannot reach target in time. EVs at target are left out (they will not charge). In each world, P10/P50/P90 across those EVs; each then the median across worlds. |
| `time_slack_hours` | `hours` | `across_weeks` | Each world's median EV slack (as above); P10/P50/P90 across worlds (item 40: the across-weeks band is always shown). |

Power, headroom and energy: the per-world value is a sum over EVs inside the world; `p10`, `p50`, `p90` are `numpy.quantile(..., method="linear")` across worlds and `world_count` is the world count. Time slack rows use only worlds with at least one EV that can charge in the slot (`world_count`, which may be below the run's world count); all three are NaN when no world has one.

**Reading the smart path.** The selected path's flexible power and movable energy are larger than the normal path's because charging waits for cheap forecast half-hours, so EVs stay below target for longer. That is headroom still unused while waiting for cheap slots, not extra flexibility: the same EVs are plugged in on both paths. `turn_up_headroom_kw` shows it directly. Computed in `model/summaries.py` from the per-EV chunks, so it adds no random draw.

### 3.6d `deferrable_power_bands` (exact) — NEW (decision 0004 item 54, plan C2), every result

How much of the unmanaged fleet's charging could wait, and for how long: "Deferrable power by slack" (replaces the time-slack chart). Row groups in the order `under_1h`, `1_to_2h`, `2_to_4h`, `4_to_8h`, `8h_or_more` (buckets), `at_least_1h`, `at_least_2h`, `at_least_4h`, `at_least_8h` (cumulative: kW that can wait at least N hours), `total`, each with one row per slot. Columns `slack_bucket`, `unit` (`"kW"`), `path_id` (always `"normal"`: the unmanaged counterfactual whose charging would be moved), `slot_index`, `interval_start_utc`, `interval_end_utc`, `interval_start_london`, `world_count` (int64), `p10`, `p50`, `p90` (float64). Keys `(slack_bucket, slot_index)`. `None` exactly when `flexibility_bands` is `None`.

Per world and slot, each bucket is the sum of home charging power (kW) of the EVs that **can charge** (3.6c: plugged in for the whole slot, below preferred target) whose **whole-slot slack** falls in it. Whole-slot slack = hours from the slot start to the expected departure (3.6c) − `ceil(grid kWh needed ÷ (P × 0.5 h)) × 0.5 h`, because the kernel dispatches whole half-hours. Buckets are `[lower, upper)`: slack exactly 2 h is in `2_to_4h`; negative slack (cannot make target in time) is in `under_1h` ("must run now"). `at_least_<N>h` is each world's sum of the buckets with slack ≥ N h (so `at_least_8h` equals `8h_or_more`), and `total` each world's sum over all buckets, which equals the normal path's `flexible_power_kw`; then linear P10/P50/P90 across worlds for every row. The cumulative and total rows are summed per world **before** quantiles: bucket quantiles do not add up to them, and the UI must never stack or add bucket quantiles to show "at least N hours" or a total (plot those rows instead).

### 3.6e `flexibility_weekly_summary` (exact) — NEW (decision 0004 item 54, plan E1), every result

One row of per-week flexibility figures on the unmanaged path for Overview ▸ Key stats section B. `None` exactly when `flexibility_bands` is `None`. Each figure is computed per world (simulated week) first, then linear P10/P50/P90 across worlds (`_p10`, `_p50`, `_p90` columns, float64). "At a time" means the London wall-clock half-hour starting then (18:00 = 18:00–18:30), on every study date.

| Column | Dtype | Per-world value |
| --- | --- | --- |
| `path_id` | object | `"normal"` |
| `world_count` | int64 | Worlds used (all) |
| `fleet_charger_capacity_kw` | float64 | Sum of every EV's home charging power (`units.home_charger_limit_kw`), kW; one value |
| `peak_deferrable_kw_*` | float64 | The week's highest half-hour of total deferrable power (3.6d `total`), kW |
| `peak_modal_slot_index` | int64 | Study slot most often the week's peak (earliest wins ties, inside a week and between slots) |
| `peak_modal_interval_start_utc`, `peak_modal_interval_start_london` | UTC, London | That slot's keys |
| `peak_modal_week_count` | int64 | Weeks whose peak falls in that slot |
| `hours_at_least_quarter_capacity_*` | float64 | Half-hours with total deferrable power ≥ 25% of `fleet_charger_capacity_kw`, × 0.5 h (hours per week). 25% is a reporting threshold (lead decision, 29 September 2026), not a model assumption; a share of the fleet's own capacity means the same for any fleet size, where a fixed 1 MW read 0 for 200 EVs |
| `movable_energy_1800_kwh_*` | float64 | Mean over the week's 18:00 half-hours of normal-path `movable_energy_kwh` (3.6c), kWh |
| `deferrable_kw_1800_*`, `deferrable_kw_2200_*`, `deferrable_kw_0300_*` | float64 | Mean over the week's half-hours at that time of total deferrable power, kW |
| `w_per_plugged_in_ev_1800_*`, `_2200_*`, `_0300_*` | float64 | 1,000 × the week's sum of total deferrable kW at that time ÷ its sum of normal-path `connected_count` at that time: W per plugged-in EV (EVs at target count in the denominator, so this is below the charger power). Weeks with no EV plugged in at that time have none and are left out; all three NaN if no week has one |
| `deferrable_at_least_2h_kw_1900_*` | float64 | Mean over the week's 19:00 half-hours of `at_least_2h` deferrable power (3.6d), kW |

### 3.7 `cohort_interval_bands` (exact) — EXISTS as `cohort_interval_distribution`, reshaped (M5)

`cohort_id` first, then the columns of 3.6; one row per `(cohort_id, metric, path_id, slot_index)`. Same metric list, computed from per-world cohort sums (`cohort_world_intervals` today), including the optional `timed` path (decision 0007) when the policy is on. Only cohorts with `ev_count > 0` appear. Consumer: Drivers ▸ Archetypes expander (one archetype across the week).

### 3.8 `average_day_bands` (exact) — NEW (M5)

The brief's sketch 2 and the archetype small multiples. One row per `(group_id, path_id, day_type, metric, local_half_hour)`, 48 half-hours each.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `group_id` | object | `"fleet"` or a `cohort_id` with `ev_count > 0` |
| `path_id` | object | `normal`/`selected` |
| `day_type` | object | `"weekday"`, `"weekend"` or `"all"` |
| `metric` | object | `"connected_share"` or `"battery_soc_percent"` |
| `unit` | object | `"fraction"` or `"percent"` |
| `spread` | object | `"across_worlds"` or `"across_evs"` (the caption names it) |
| `local_half_hour` | int64 | 0–47 |
| `local_time_label` | object | `"00:00"` … `"23:30"` |
| `centre_stat`, `low_stat`, `high_stat` | object | Names of the statistics in `centre`, `low`, `high` |
| `centre`, `low`, `high` | float64 | `low ≤ high`; also `low ≤ centre ≤ high` when `spread` is `across_worlds`. NaN when no slot of that day type has this local half-hour |
| `world_count` | int64 | Worlds summarised |
| `day_count` | int64 | Study dates of this day type |

Computation, which the UI must caption exactly:

- `connected_share`, **spread across worlds** (`p50`, `p10`, `p90`): in each world, the group's connected share at each slot, averaged over the study slots of that day type that fall on that London half-hour (two slots on the repeated autumn hour, none on the skipped spring half-hour); then P10/P50/P90 across worlds.
- `battery_soc_percent`, **spread across EVs** (`mean_across_evs`, `p5_across_evs`, `p95_across_evs`): in each world and slot, the mean, 5th and 95th percentile of per-EV closing SoC across the group's EVs; each averaged over the day type's slots at that half-hour as above; then the median across worlds of each. The centre stays the **mean** (the brief's sketch), so with a skewed fleet it can sit outside the world-median P5–P95 band; only `low ≤ high` is guaranteed (review B1). This is population variation (the brief's band), not forecast uncertainty (decision 0004 item 22, design section 1).

### 3.9 `plug_in_events` (at least) — NEW (M5)

Fleet plug-in sessions, **normal path only** (the action does not change when drivers plug in; selected-path events for one EV come from the replay). **A sample of at most 10 worlds** (lead decision Q1: M5 builds per-EV summaries in chunks of ≤10 worlds): the worlds listed in the result field `sampled_world_ids: tuple[int, ...]`, representative world first, then the lowest other ids; rows follow that world order. Every-world statistics come from `plug_in_world_kpis` and `plug_in_summary`, never from this table. One row per `(world_id, unit_id, event_index)`. A session counts when its plug-in instant falls inside the horizon; a session already plugged at the horizon start is excluded; a session still plugged at the end has `plug_out_utc = NaT`. Consumer: Drivers ▸ Plug-ins Data expander; archetype script (S2).

| Column | Dtype | Meaning |
| --- | --- | --- |
| `world_id` | int64 | |
| `path_id` | object | Always `"normal"` here |
| `unit_id`, `cohort_id` | object | |
| `event_index` | int64 | 0, 1, … per `(world_id, unit_id)` in time order |
| `plug_in_utc` | datetime64[ns, UTC] | Start of the first connected slot |
| `plug_in_london` | datetime64[ns, Europe/London] | Same instant, London |
| `local_date` | object (`date`) | Session night of the plug-in (as `study_slots.local_date`) |
| `day_type` | object | `weekday`/`weekend` |
| `plug_in_local_hour` | int64 | 0–23, London |
| `plug_in_soc_percent` | float64 | SoC at plug-in, 0–100 (a stock-flow output) |
| `plug_out_utc` | datetime64[ns, UTC] | End of the last connected slot; NaT if still plugged at horizon end |
| `plug_out_soc_percent` | float64 | SoC at plug-out; NaN if still plugged |
| `home_import_kwh` | float64 | Grid-side kWh imported during the session inside the horizon |
| `battery_added_kwh` | float64 | Battery-side kWh added (after charging efficiency) |
| `still_plugged_at_horizon_end` | bool | True when `plug_out_utc` is NaT |

### 3.10 `plug_in_world_kpis` and `plug_in_summary` — NEW (M5)

`plug_in_world_kpis` (exact) covers **every** world: one row per `(world_id, group_id, day_type, metric)`; columns `world_id` (int64), `group_id` (`"fleet"` or an archetype with `ev_count > 0`), `day_type` (weekday/weekend/all), `metric`, `unit`, `value` (float64). Metrics are the four KPI metrics below plus `modal_plug_in_local_hour` (`hour (London)`, ties go to the earlier hour). Counts are 0 in a world with no plug-ins; SoC and hour values are NaN there. It feeds `plug_in_summary.kpis`, `cohort_summary` and Compare's median plug-in SoC. For the sampled worlds it must equal a recomputation from `plug_in_events`.

A `PlugInSummary` with three exact frames, from every world's normal-path plug-ins. Consumer: Drivers ▸ Plug-ins histograms and KPIs; Overview "Median SoC at plug-in" tile.

`hour_bands`: one row per `(day_type, local_hour)`, `day_type` in weekday/weekend/all, `local_hour` 0–23.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `day_type`, `local_hour` | object, int64 | Keys |
| `world_count` | int64 | Worlds with at least one plug-in of this day type |
| `count_p50` | float64 | Plug-ins starting in this hour, P50 across worlds |
| `share_mean` | float64 | Mean across worlds of the per-world share; sums to 1 over the 24 hours of a day type (the P10/P50/P90 columns do not and must not be summed) |
| `share_p10`, `share_p50`, `share_p90` | float64 | Per world, this hour's plug-ins ÷ that world's plug-ins of this day type; quantiles across worlds with at least one such plug-in |

`soc_bands`: one row per `(day_type, soc_bin_lower_percent)`, bins of 5 points `[0, 5)` … `[95, 100]`; columns `day_type`, `soc_bin_lower_percent` (int64), `soc_bin_upper_percent` (int64), `world_count`, `share_mean`, `share_p10`, `share_p50`, `share_p90` with the same per-world share rule; `share_mean` sums to 1 over the 20 bins.

`kpis`: one row per `(day_type, metric)`; columns `day_type`, `metric`, `unit`, `p10`, `p50`, `p90`: quantiles across worlds of the fleet rows of `plug_in_world_kpis`, worlds with a missing value left out:

| Metric | Unit | Per-world value |
| --- | --- | --- |
| `plug_ins_per_week` | `plug-ins` | Fleet plug-ins of this day type in the week |
| `plug_ins_per_ev_per_week` | `plug-ins per EV` | Above ÷ `vehicle_count` |
| `median_plug_in_soc_percent` | `percent` | Median `plug_in_soc_percent` |
| `share_below_10_percent_soc` | `fraction` | Share of plug-ins with SoC < 10% |

The weekday and weekend rows of `kpis` already split plug-ins per EV per week by day type (`plug_ins_per_ev_per_week`: weekday plug-ins per EV over the week's five weekdays, weekend over its two weekend days; `all` is their sum), and `plug_in_world_kpis` gives the same per archetype; Drivers ▸ Plug-ins compares them with CNZ (item 43).

### 3.10a `plug_in_heatmap` (exact) — NEW (decision 0004 item 43), every result

When drivers plug in, by weekday and London hour: Drivers ▸ Plug-ins heatmap. One row per `(weekday, local_hour)`, 7 × 24 = 168 rows, Monday first, hours 0–23. From every world's normal-path plug-ins (the same sessions as 3.9–3.10).

| Column | Dtype | Meaning |
| --- | --- | --- |
| `weekday` | int64 | 0 = Monday … 6 = Sunday, of the plug-in's session night (a 01:00 Saturday plug-in is Friday's) |
| `weekday_label` | object | `"Mon"` … `"Sun"` |
| `local_hour` | int64 | London hour the plug-in starts in, 0–23 |
| `day_count` | int64 | Study nights on this weekday (1 in a 7-night study) |
| `world_count` | int64 | Worlds summarised (every world) |
| `mean`, `p10`, `p50`, `p90` | float64 | Per world, plug-ins starting in this cell ÷ EV-days (`vehicle_count × day_count`): the chance an EV plugs in during that hour on that day (strictly plug-ins per EV-day); then mean and quantiles across worlds. NaN when `day_count` is 0 |

Dividing by EV-days, not by the week's plug-ins, keeps busy and quiet days comparable. The per-world values of a week add up to its plug-ins per EV, so Σ over cells of `mean × day_count` equals the mean across worlds of `plug_ins_per_ev_per_week` (fleet, `all`). A session already running at the horizon start is not a plug-in (3.9), so the first study hour slightly undercounts. The UI plots `p50` as the heatmap and shows P10–P90 on hover; it does not add cells' quantiles.

### 3.10b `plug_in_half_hour_heatmap` and `plug_out_heatmap` (exact) — NEW (decision 0004 items 51, 54, plan D-1), every result

When drivers plug in and when they unplug (unplug = departure, item 51), by London calendar weekday and wall-clock half-hour: Drivers ▸ Plug-ins heatmaps with a shared colour scale. Same columns for both; one row per `(weekday, local_half_hour)`, 7 × 48 = 336 rows, Monday first. From every world's normal path. `plug_in_heatmap` (3.10a, hourly) stays for the current view until it moves to these.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `weekday` | int64 | 0 = Monday … 6 = Sunday, of the London calendar date of the event instant |
| `weekday_label` | object | `"Mon"` … `"Sun"` |
| `local_half_hour` | int64 | 0–47, London wall-clock half-hour of the event |
| `local_time_label` | object | `"00:00"` … `"23:30"` |
| `slot_count` | int64 | Study slots in this cell (1 in a plain 7-day study; 2 for the repeated autumn half-hour; 0 for the skipped spring half-hour or a weekday the study does not reach) |
| `world_count` | int64 | Worlds summarised (every world) |
| `mean`, `p10`, `p50`, `p90` | float64 | Per world, events in this cell ÷ (`vehicle_count` × `slot_count`): events per EV-day in that half-hour; then mean and linear quantiles across worlds. NaN when `slot_count` is 0; a real 0 when the cell has slots but no event |

Events: a **plug-in** is the first slot of a session that starts inside the study (the sessions of 3.9, so a session already running at the horizon start has no plug-in). A **plug-out** is the start of the slot after a session's last fully connected slot, for every session that ends inside the study, *including* sessions already running at the horizon start: their departure is observed, and leaving them out would empty the first morning. Sessions still plugged in at the horizon end have no plug-out. Counting the study's own slots per cell keeps the rate honest when the study does not cover whole days (a clock change, or a horizon starting at noon). Σ over cells of `mean × slot_count` equals the mean fleet `plug_ins_per_ev_per_week` (`all`) for plug-ins; for plug-outs it is at most that plus 1 (one running session per EV at the start). The UI plots `p50` and shows P10–P90 on hover; it never adds cells' quantiles.

### 3.10c `session_distribution_bands` (exact) — NEW (decision 0004 item 54, plan D-2), every result

How plug-in sessions are distributed, per archetype: Drivers ▸ Sessions small multiples (bars at `share_p50` with P10–P90 whiskers; dashed source markers from section 8.2). One row per `(group_id, day_type, metric, bin_index)` in that order: `group_id` is `"fleet"` then each archetype with `ev_count > 0` in source order; `day_type` weekday/weekend/all (of the plug-in's session night, `study_slots.day_type` of its first slot, as in 3.9); metrics and bins in the order below. From every world's normal-path sessions, built in the same per-EV chunk pass as 3.10.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `group_id`, `day_type`, `metric`, `unit` | object | Keys and the metric's unit |
| `bin_index` | int64 | 0 … bin count − 1 |
| `bin_lower`, `bin_upper` | float64 | Bin edges in the metric's unit; a value is in the bin when `bin_lower ≤ value < bin_upper` |
| `bin_label` | object | Display label: `"18:30"` for times, `"5–10"` otherwise, `"< -10"` / `"≥ 75"` for open-ended edge bins |
| `world_count` | int64 | Worlds with at least one session of this group and day type with the metric observed |
| `share_mean` | float64 | Mean across those worlds of the per-world share; sums to 1 over a metric's bins |
| `share_p10`, `share_p50`, `share_p90` | float64 | Per world, sessions in this bin ÷ that world's sessions with the metric observed; linear quantiles across those worlds. They do not sum to 1 and must not be summed. All four are NaN when `world_count` is 0 |

| `metric` | `unit` | Bins | Per-session value |
| --- | --- | --- | --- |
| `plug_in_time` | `hour (London)` | 48 × 0.5 h from 0 | London wall-clock time of the first fully connected slot |
| `departure_time` | `hour (London)` | 48 × 0.5 h from 0 | London time the EV unplugs: the start of the slot after the last connected one (unplug = departure, item 51) |
| `plug_in_soc_percent` | `percent` | 20 × 5 from 0 (100% in the last) | `plug_in_soc_percent` of 3.9, a stock-flow output |
| `energy_needed_kwh` | `kWh battery-side` | 16 × 5 kWh from 0; last open above | max(preferred target stock − stock at plug-in, 0) |
| `dwell_hours` | `hours` | 24 × 2 h from 0; last open above | Fully connected half-hours × 0.5 h (the kernel charges only in whole connected slots) |
| `flexible_kwh` | `kWh battery-side` | 16 × 5 kWh from 0; last open above | min(energy needed, dwell × home charging power × home charge efficiency): battery energy the session could take at full power |
| `slack_hours` | `hours` | 30 × 2 h from −12; first open below, last open above | dwell − energy needed ÷ (power × efficiency): hours charging could wait and still reach the target; negative when it cannot even at full power (3.6c's time slack uses the expected departure; this uses the actual unplug) |

Population: the sessions of 3.9 (a session already running at the horizon start has no observed plug-in and is left out). `departure_time`, `dwell_hours`, `flexible_kwh` and `slack_hours` need the unplug, so a session still plugged in at the horizon end counts only in the other three metrics; each metric's share uses its own observed sessions. This censoring under-represents long and late-week sessions (for example Always Plugged-in and the last night), so the dwell, slack, departure and flexible-energy distributions lean short; the noon-to-noon horizon (decision 0004 item 52) reduces but does not remove it, and the Sessions view caption says so. Energy is battery-side so it compares with the workbook's battery kWh per plug-in (8.2). Values are rounded to 1e-9 before binning, so the float residue of charging exactly to target does not move an at-target session across a bin edge (its need is 0 and its slack exactly its dwell). Values outside a metric's range count in its first or last bin.

CNZ reference markers are **not** model output. The view reads them from `assumptions` by name (section 8.1), held in `CNZ_CONTEXT` in `model/assumptions.py` (moved there by M3 from a removed module). They are comparison context, not calibration targets.

Model step 2 (decision 0007): the frame also carries `departure_soc_percent` (every session's stock at unplug) and, action results only, `departure_soc_percent_smart` (the selected path's stock at the same slot; NaN without smart charging), both `percent`, bins as `plug_in_soc_percent`'s. On a run whose `timed_start_local_hour` is set it additionally carries `departure_soc_percent_timed` (the timed path's own stock at the same slot, same bins); absent, not NaN-filled, on a run without it, so that run's rows are identical to before this metric existed.

## 4. Action-only fields (`None` on no-action results)

### 4.1 `difference_bands` (exact) — NEW (M5), decision 0004 item 12

Selected minus normal, **per world, then quantiles** (never band subtraction). One row per `(metric, slot_index)`; columns `metric`, `unit`, `slot_index`, `interval_start_utc`, `interval_end_utc`, `interval_start_london`, `world_count`, `mean`, `p10`, `p50`, `p90`. Metrics are every fleet metric in 3.5; `unit` is the 3.5 unit, except `battery_soc_percent` whose unit is `percentage points`. Consumer: Smart charging ▸ Response paired-difference chart.

### 4.2 `difference_weekly` and `difference_weekly_bands` (exact) — NEW (M5), items 12–13

`difference_weekly`: one row per `(world_id, metric)`; columns `world_id`, `metric`, `unit`, `normal_value`, `selected_value`, `selected_minus_normal`. Metrics: `home_import_kwh`, `public_import_kwh`, `unserved_travel_kwh` (weekly totals, `kWh per week`) and `closing_battery_kwh` (last slot, `kWh`). `difference_weekly_bands`: one row per metric; `metric`, `unit`, `world_count`, `mean`, `p10`, `p50`, `p90` of `selected_minus_normal` across worlds. Consumer: Response lens KPIs and Data expander.

### 4.3 `action_summary` — amended (item 38)

An `ActionSummary` for the summary sentence on every Smart charging lens and the Overview "Action chosen" tile.

| Field | Type | Meaning |
| --- | --- | --- |
| `action_id` | str | `"smart_charging_v1"` |
| `action_label` | str | Plain text: `"Smart home charging in the cheapest forecast half-hours before expected departure"` |
| `departure_margin_hours` | float | Safety margin (h) subtracted from each cohort's typical home departure to get the expected departure; editable illustrative assumption, default 1.0 |
| `forecast_available_at_utc` | `pd.Timestamp` (UTC) | When the first study slot's day-ahead price was published (13:00 London the day before the start date, plan B4): `forecast_prices.forecast_available_at_utc` of slot 0. Each later day's prices are published at 13:00 the day before it |
| `vehicle_count` | int | Fleet size; smart charging applies to every EV |

**The policy (decision 0004 item 38).** The normal path charges as soon as the EV is plugged in for a whole half-hour, at home charging power, up to the preferred target. On the selected path, when an EV is first plugged in for a whole half-hour (or when planning starts, on the last warm-up night, if it is already plugged in; item 52), the charger plans the session from what it knows then: the stock at plug-in, the preferred target, home charging power, the expected departure (the cohort's typical home departure clock time on the first London date where that time minus the margin is at least one half-hour away, converted at that date's own offset) and its own world's day-ahead prices as published by then (decision 0004 item 48, plan B4): a price for London delivery date D is visible from 13:00 London on D − 1; a half-hour not yet published is ranked on the expected shape, the mean published price of the same London half-hour over the 7 most recent published days (`action.visible_day_ahead_prices`, the one rule the trading overlay reuses). It plans `(target − stock) / efficiency` grid kWh in the cheapest day-ahead half-hours from plug-in to the expected departure, each capped at home charging power × 0.5 h, cheapest first (ties go to the earlier half-hour; the last chosen half-hour may be partial). If the window cannot hold the need, every half-hour in it charges at full power, which is "as soon as possible". Windows stop at the horizon end, because energy planned after it would be counted as not recovered. The study runs noon to noon (item 52), so every overnight session ends before that cut; the item 49 rule that made sessions past the end charge at once is retired (measured: the last morning's smart import matches the other weekday mornings, no spike before the cut). An EV still plugged in when its window ends starts a new plan for the next expected departure. The plan then runs against the actual session: half-hours after the EV leaves are missed, so it leaves below target, and a later trip may trigger a public top-up (items 32, 37) or the week may end with energy not recovered; both are priced in `cost_effect` (items 4, 5, 13). Evaluation (realised) prices never enter the plan. Smart charging draws no random number.

**Prices (decision 0004 items 48, 53, 55, 56 and 58; plan §2B B1, B3, B5 and §7 H1).** Each world has its own synthetic market, generated by `sampling.generate_market_price_paths` after the worlds:

- *Net demand* (GW) = a 3-harmonic demand shape by London clock hour (weekday/weekend × winter/summer, the season set by the start month) + heating (`heating_gw_per_c` per °C of that world's sampled daily temperature below `heating_threshold_c`) − wind (`wind_installed_gw` × a daily capacity factor, logit-normal AR(1) across days) − solar (item 58: `solar_installed_gw` × a seasonal clear-sky half-sine between sunrise and sunset × a daily clearness, logit-normal AR(1) across days; zero at night) + known shocks.
- *Day-ahead price* (£/MWh) = the expected intraday close given day-ahead information, minus the deliberate premium `da_id_premium_gbp_per_mwh` (default 0; Mike's no-free-money rule). The no-surprise price is the convex supply curve `f(N) = p_ref + s·(N − N_ref) + A·exp((N − N_scarcity)/w)` (monotone and convex, negative at very low net demand) + a daily level (AR(1) across days) + AR(1) half-hour noise (`day_ahead_sd_gbp_per_mwh`, φ = `ar1_phi`). The expected impact of the stochastic surprises is added exactly (their distribution is enumerated and convolved on a 0.1 GW grid, no random draw), because the convex curve makes zero-mean surprises raise the mean price. Every price (day-ahead, intraday path and close, imbalance) is clipped to the GB day-ahead auction limits, `price_floor_gbp_per_mwh` −£500 and `price_cap_gbp_per_mwh` £4,000 (Nord Pool N2EX notices of 2021 and 2022); negative prices otherwise arise from the model. The demand shapes, `ar1_phi`, the level persistence and the imbalance shares and premiums are calibrated to the Elexon fit of 12 months of GB day-ahead and system prices (`docs/research/elexon-price-calibration.md`; contains BMRS data © Elexon Limited copyright and database right 2026); every other value is illustrative. A default October run gives a mean near £80/MWh, a weekday trough at 03:00 and peak at 17:30, and about 2% negative half-hours; a summer run troughs at midday (solar), as the fit does.
- *Shocks* (item 56): mild (about two a night) and big (about five a week) net-demand shocks, up or down, sizes capped at `shock_max_gw`, starting only between London 15:00 and 09:00, trapezoid in time; each is known day-ahead (in net demand and the day-ahead price) or a surprise, revealed `surprise_reveal_hours` (default 2 h) before it starts (trading contract §2.5, Q10).
- *Intraday close* (£/MWh) = the no-surprise price + hourly martingale updates from day-ahead publication (13:00 London the day before) to gate closure (`gate_closure_minutes`, 60), AR(1)-correlated across adjacent periods, + the supply-curve impact of the surprises revealed by gate closure. Mean(close − day-ahead) equals the premium within sampling error. The hour-by-hour path includes a surprise from the first step after its reveal, and the not-yet-revealed part of the expected impact before it.
- *Imbalance price* (£/MWh) = intraday close + the impact of surprises revealed only after gate closure + a two-state NIV premium (long or short, a persistent Markov chain; a covering surprise moves the chance of short by `niv_surprise_shift` in its direction; otherwise NIV is independent of prices) + Student-t(3) tails.

Roles: every EV in a world plans on that world's day-ahead path, so the cheapest half-hour moves between days and worlds, while EVs in the same world still share it; home import (the customer's tariff leg) is valued at the same day-ahead price (B3), so a price the charger could not see never moves the saving. The intraday close is the realised price (`evaluation_prices`); with the imbalance price and the shock table it is carried for the trading ledger and values nothing in 4.8. Information assumption (plan B4, item 53): day-ahead prices for a London date are published at 13:00 London the day before (`forecast_available_at_utc`, per slot), and a plan ranks only prices already published; later half-hours are ranked on the expected shape (above), built from published day-ahead (expected-close) prices. Prices are generated over the warm-up and the study (item 52), with each priced date's sampled temperature, so the shape has published history from the first plan; the result's price frames and `price_shocks` keep the study slots. Random order: market channels are drawn last, each with a fixed draw count (day-ahead noise, daily level, wind, intraday updates, NIV states, imbalance tails, shock candidates, then solar clearness), so a price edit moves no world and no other price channel; temperatures feed the heating term by value, so a weather edit also moves prices.

### 4.4 `smart_charging_world` (exact) — NEW (item 38)

One row per world, in `world_id` order. Consumer: Smart charging lenses (follow-up UI task).

| Column | Dtype | Meaning |
| --- | --- | --- |
| `world_id` | int64 | |
| `normal_average_price_gbp_per_mwh` | float64 | Fleet home import in the week weighted by that world's day-ahead price (plan B3, item 53), normal path, £/MWh; NaN when the path imports nothing |
| `selected_average_price_gbp_per_mwh` | float64 | Same for the selected path. Home import is bought at day-ahead prices, so this is the price paid, the same price the cost effect (4.8) uses |
| `moved_home_import_kwh` | float64 | Sum over half-hours of `max(0, normal − selected)` fleet home import: energy the selected path took in other half-hours. Fleet sums net out opposite moves by different EVs, so this is a lower bound on EV-level moves |
| `moved_home_import_share` | float64 | `moved_home_import_kwh` ÷ the week's normal home import; NaN when that is 0 |
| `early_departure_count` | int64 | Weekly sum of `early_departure_count` (3.4), selected path |
| `early_departure_shortfall_kwh` | float64 | Weekly sum of `early_departure_shortfall_kwh` (3.4), selected path, battery-side kWh |
| `evidence_kind` | object | `"illustrative_synthetic"` |

Model step 2 (decision 0007): on a run whose `timed_start_local_hour` is set, the frame additionally carries `timed_average_price_gbp_per_mwh` (float64), the same import-weighted day-ahead price for the timed path, inserted after `selected_average_price_gbp_per_mwh`; absent on a run without the timed path. `moved_home_import_*` and `early_departure_*` are not redefined for it.

### 4.5 `smart_charging_summary` (exact) — NEW (item 38)

One row per metric in this order; columns `metric`, `unit`, `world_count`, `mean`, `p10`, `p50`, `p90`, statistics across worlds of the 4.4 column, worlds with NaN left out (`world_count` says how many were used).

| `metric` | `unit` | Per-world value |
| --- | --- | --- |
| `normal_average_price_gbp_per_mwh` | `GBP/MWh` | 4.4 column |
| `selected_average_price_gbp_per_mwh` | `GBP/MWh` | 4.4 column |
| `average_price_change_gbp_per_mwh` | `GBP/MWh` | Selected minus normal average price inside each world first (item 12) |
| `moved_home_import_share` | `fraction` | 4.4 column |
| `early_departure_count` | `sessions per week` | 4.4 column |
| `early_departure_shortfall_kwh` | `kWh per week` | 4.4 column |

On a run whose `timed_start_local_hour` is set, a `timed_average_price_gbp_per_mwh` (`GBP/MWh`) row joins these, the 4.4 column of the same name; it has no "change" row of its own (that reading stays selected minus normal).

Public top-ups against normal are `difference_weekly_bands` metric `public_import_kwh` (4.2); the weekly illustrative saving is `cost_effect_summary` component `total` (4.9, negative is a saving).

### 4.5a `weekly_peak_summary` (exact) — NEW (decision 0004 items 48–49)

Each simulated week's own peak fleet home import. The herding finding is measured this way, not as the peak of the per-slot P50 across weeks: the cheapest half-hour moves between weeks, so the median band smears the peak out (item 49). One row per path the run's `fleet_world_intervals` carries: `path_id` in order `normal`, `selected`, plus `timed` (decision 0007, model step 2) on a run whose `timed_start_local_hour` is set. Built in `model/summaries.py`; `None` on a no-action result.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `path_id` | object | `normal` or `selected` |
| `unit` | object | `"kW"` |
| `world_count` | int64 | Worlds used (all of them) |
| `p10`, `p50`, `p90` | float64 | Per world, the maximum over the 336 study slots of fleet `home_import_kw` on this path; then `numpy.quantile(..., method="linear")` across worlds |
| `modal_peak_slot_index` | int64 | The study slot that is most often a week's peak; the earliest slot wins a tie, both inside a week and between equally common slots |
| `modal_peak_interval_start_utc`, `modal_peak_interval_start_london` | UTC, London | That slot's keys from `study_slots` |
| `modal_peak_week_count` | int64 | Weeks whose peak falls in that slot |
| `ratio_world_count` | int64 | Worlds with a normal-path peak above 0 |
| `ratio_to_normal_p10`, `ratio_to_normal_p50`, `ratio_to_normal_p90` | float64 | Per world, this path's peak ÷ that week's normal-path peak, then linear quantiles across the `ratio_world_count` worlds (1 on the `normal` row; NaN if no world has a normal peak). The ratio is taken per week first: the ratio of two medians is not the median of the ratio (item 12) |
| `coincidence_world_count` | int64 | Worlds with at least one EV plugged in at this path's peak slot (decision 0004 item 54) |
| `coincidence_factor_p10`, `coincidence_factor_p50`, `coincidence_factor_p90` | float64 | **Coincidence factor**, per world: this path's peak ÷ (normal-path `connected_count` at that peak slot × the one home charging power, `units.home_charger_limit_kw`); then linear quantiles across the `coincidence_world_count` worlds. 1 = every plugged-in charger drawing full power at the peak. Connection is the same on both paths, so the normal count serves both. Per week first, like the ratio. In [0, 1]: home import needs a fully connected slot and never exceeds charger power. The model raises if EVs have different home charging powers (the formula assumes one) |

Consumer: Response "each week's own smart peak" tile, the herding caption, a coincidence-factor tile and the Key stats coordination rows.

### 4.6 `price_band_shift` (exact) — NEW (item 38)

Home import moved out of each day-ahead price third, per session night (London 12:00 to 12:00, item 52), so a night's evening peak and the overnight half-hours it moved to sit in one row. Each world's 336 day-ahead prices are split at that world's own 1/3 and 2/3 quantiles: `low` below the first, `high` above the second, `middle` otherwise (those are the prices that world's charger ranked, decision 0004 item 48). One row per `(local_date, price_band)`, dates in order, bands `low`, `middle`, `high`.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `local_date` | object (`date`) | Session night (as `study_slots.local_date`) |
| `day_label` | object | `"Mon 28"` |
| `price_band` | object | `low`, `middle`, `high` |
| `forecast_price_min_gbp_per_mwh`, `forecast_price_max_gbp_per_mwh` | float64 | Range of the band's day-ahead prices on that date across all worlds; NaN if no world has a half-hour of the band that date |
| `mean_slot_count` | float64 | Half-hours of that band on that date, averaged over worlds (each world has its own bands); the three add up to the date's study half-hours |
| `unit` | object | `"kWh per day"` |
| `world_count`, `mean`, `p10`, `p50`, `p90` | int64, float64 | Per world, `normal − selected` fleet home import summed over the band's half-hours that date (positive: moved out of the band; negative: moved into it); then statistics across worlds. This "moved" reading always stays normal versus selected |
| `normal_kwh_p10`, `normal_kwh_p50`, `normal_kwh_p90`, `selected_kwh_p10`, `selected_kwh_p50`, `selected_kwh_p90` | float64 | Each path's own fleet home import in the band that date, per world, then P10/P50/P90 across worlds (worlds first), so Plan can show "energy by forecast-price band, normal vs smart" (item 43). The difference of two quantiles is not the quantile of the difference; the moved-out columns above give that |
| `timed_kwh_p10`, `timed_kwh_p50`, `timed_kwh_p90` | float64 | Model step 2 (decision 0007): the timed path's own fleet home import in the band that date, same reading as the two above, present only on a run whose `timed_start_local_hour` is set |

### 4.7 `forecast_prices` and `evaluation_prices` (exact) — EXISTS `forecast_synthetic_prices`, `evaluation_synthetic_prices`

`forecast_prices`: one row per `(world_id, slot_index)`, world-major (decision 0004 item 48; before it, one row per slot); `world_id`, `slot_index` (NEW), `interval_start_utc`, `interval_end_utc`, `interval_start_london` (NEW), `forecast_available_at_utc` (per slot: 13:00 London on the day before the slot's London delivery date, the decision-time cut-off of plan B4), `wholesale_forecast_gbp_per_mwh` (£/MWh, that world's synthetic day-ahead price), `evidence_kind="synthetic"`. The prices each world's smart charger ranks. Consumer: `forecast_price_bands` (below) and `price_band_shift`.

`forecast_price_bands` (exact, NEW, item 48): one row per slot, in `slot_index` order; `slot_index`, `interval_start_utc`, `interval_end_utc`, `interval_start_london`, `world_count` (int64), `p10`, `p50`, `p90` (float64, £/MWh): each slot's day-ahead price across worlds, `numpy.quantile(..., method="linear")`. A price is one value per world and slot, so no world-first aggregation is needed. Built in `model/summaries.py`; `None` on a no-action result. Consumer: Plan price chart (median line and P10–P90 band).

#### 4.7a `price_relative_bands` (exact) — NEW (decision 0004 item 54, plan D-3), action only

Where each half-hour's day-ahead price sits within its own session night, the stretch a smart charger ranks. Per world and **whole** session night (London 12:00 to 12:00, item 52; one whose half-hours are all in the study: 46, 48 or 50; the slots a spring change pushes past the last noon are left out because a part-night mean is biased), each half-hour's day-ahead price minus that night's mean day-ahead price in that world (£/MWh). All world-nights in a cell are pooled; mean and linear P10/P50/P90. Rows: `weekday` 0–6 (Monday first, the night's evening weekday, so Monday's 03:00 cell is Tuesday 03:00) × 48 half-hours, then 48 rows over all days. Columns `weekday` (`Int64`, `<NA>` on the all-days rows), `weekday_label` (`"Mon"` … `"Sun"`, `"All days"`), `local_half_hour` (int64, 0–47), `local_time_label` (`"HH:MM"`), `unit` (`"GBP/MWh"`), `sample_count` (int64, world-nights pooled; the repeated autumn half-hour gives two a night), `world_count` (int64), `mean`, `p10`, `p50`, `p90` (float64; NaN with no sample). Built only from `forecast_prices` columns, so it holds for any price generator. `None` on a no-action result. Consumer: Plan price heatmap (diverging, centred on 0) and half-hour spread profile; captions say the prices are synthetic.

#### 4.7b `cheapest_half_hour_summary` (exact) — NEW (plan E1), action only

One row. Per world and whole session night (as above), the London clock time of the half-hour with the lowest day-ahead price (earliest on a tie), in decimal hours (5.5 = 05:30); then P10/P50/P90 over all world-nights with `method="nearest"`, so each answer is a real half-hour start. Quantiles are taken on hours since the night's noon and turned back into clock times, so they follow the night's order (13:00 is early, 11:30 late). `world_day_count` counts world-nights. Columns `world_day_count` (int64), `local_hour_p10`, `local_hour_p50`, `local_hour_p90` (float64), `local_time_label_p10`, `local_time_label_p50`, `local_time_label_p90` (object, `"HH:MM"`). Moved from `scripts/headline_numbers.py`, which now reads it. `None` on a no-action result. Consumer: Key stats coordination row and the headline script.

`evaluation_prices`: one row per `(world_id, slot_index)`; `world_id`, `slot_index` (NEW), `interval_start_utc`, `interval_end_utc`, `evaluation_context_price_gbp_per_mwh` (£/MWh, the intraday close at gate closure: the day-ahead price plus its forecast error, which is now the intraday news and surprise shocks, item 53), `imbalance_price_gbp_per_mwh` (£/MWh, NEW, the system imbalance price), `system_long` (bool, NEW, the NIV state that set its premium), `outcome_available_at_utc`, `evidence_kind="synthetic"`. Realised prices never enter a plan. Since item 53 they value nothing in 4.4 or 4.8, which use the day-ahead price. Consumer: Cost Data expander; the trading ledger (trading contract).

`forecast_prices` also carries `system_net_demand_gw` (float64, NEW, item 53): the synthetic net demand behind each day-ahead price, known shocks included.

`price_shocks` (exact, NEW, decision 0004 item 56): the stochastic net-demand shocks that start in the study, one row per shock, ordered by `world_id` then `start_slot_index`; `None` on a no-action result. Columns: `world_id` (int64), `shock_class` (`mild`/`big`), `start_slot_index` (int64, 0–335), `start_utc` (UTC), `duration_slots` (int64 ≥ 1), `size_gw` (float64 > 0, at most `shock_max_gw`; the sign is `direction`), `direction` (`up`/`down`), `known_day_ahead` (bool), `evidence_kind="synthetic"`. Each shock is a trapezoid in net demand: weight `min(1, (k + 1)/(r + 1), (D − k)/(r + 1))` for half-hour `k` of `D`, `r = D // 4`; one running past the horizon end is cut there. A shock that starts in the warm-up and runs into the study moves early study prices but is not in this table (it is in `SimulatedForecast.market_prices.shocks`). The known and surprise profiles (world × slot, GW) and the hour-by-hour intraday path of every world (`intraday_path_gbp_per_mwh`, world × slot × hours before gate closure, 0 = the close; about 20 MB at 100 worlds with the 7-day warm-up, whose slots come first) are on `SimulatedForecast.market_prices` for the trading overlay, not on the result. Consumer: the trading contract's `market_shocks`.

### 4.8 `cost_effect` (exact) — EXISTS `wholesale_world_cost_effect`

One row per world, columns exactly as `calculate_wholesale_world_cost_effect` returns today: `world_id`, `normal_home_import_kwh`, `selected_home_import_kwh`, `illustrative_selected_minus_normal_energy_cost_gbp`, `selected_minus_normal_closing_battery_kwh`, `selected_minus_normal_public_import_kwh`, `selected_minus_normal_unserved_travel_kwh`, `illustrative_selected_minus_normal_public_charge_cost_gbp`, `illustrative_unrecovered_energy_value_gbp`, `illustrative_unserved_travel_value_gbp`, `illustrative_selected_minus_normal_total_gbp`, `energy_not_recovered`, `unrecovered_kwh`, `unrecovered_share`, `not_recovered_material`, `sessions_affected_count`, `evidence_kind`. The kWh columns are weekly totals (closing battery is the last-slot difference). The total is the sum of the four `illustrative_*_gbp` components (items 4, 13). `illustrative_selected_minus_normal_energy_cost_gbp` values the home-import difference at each world's day-ahead price (`forecast_prices.wholesale_forecast_gbp_per_mwh`; plan B3, decision 0004 item 53), not the realised price. Consumer: Cost strip plot and flags.

Not-recovered magnitude (decision 0004 item 45). `energy_not_recovered` stays the yes/no "any shortfall beyond 1e-6 kWh" and at fleet scale is true in almost every week, so the magnitude is reported beside it:

| Column | Dtype | Meaning |
| --- | --- | --- |
| `unrecovered_kwh` | float64 | Closing battery shortfall + extra public import + added unserved travel, each counted only when positive and beyond 1e-6 kWh. Mixes battery-side and grid-side kWh like the valuation; a closing surplus does not offset it |
| `unrecovered_share` | float64 | `unrecovered_kwh` ÷ the week's normal-path home import (a fraction); NaN when that is 0 |
| `not_recovered_material` | bool | `unrecovered_kwh` > 0 and `unrecovered_share` × 100 ≥ assumption `not_recovered_material_share_percent` (illustrative, default 1); also true when the share is NaN but energy is short. `not_recovered_world_count` counts these |
| `sessions_affected_count` | int64 | Plug-in sessions in the week (including any already plugged in at the study start) whose last fully connected half-hour ends with selected-path battery stock below the normal path's by more than 1e-6 kWh. Added by `build_summaries` from the per-EV pass, so `SimulatedForecast.cost_effect` lacks it |

### 4.8a `timed_cost_effect` (exact) — NEW (decision 0007, model step 2)

The optional timed path's own cost effect: exactly `calculate_wholesale_world_cost_effect`'s schema (4.8's column list, unchanged), called a second time with `selected_path_id="timed"` so every `selected_*` column reads as "timed minus normal" instead. Same prices (`forecast_prices.wholesale_forecast_gbp_per_mwh`), same public rate, same shortfall rule, so the ordinary saving and the timed saving are directly comparable. `None` when the run's `timed_start_local_hour` is unset (including every no-action result, which has no cost effect at all). `sessions_affected_count` is added by `build_summaries` exactly as 4.8's own column is, from `sessions_affected_per_world` paired against `timed`'s closing battery stock instead of `selected`'s: a real per-world count of sessions whose last fully connected half-hour ends with timed-path battery stock below the normal path's by more than 1e-6 kWh, not a schema-parity placeholder. Consumer: Smart charging ▸ 3 Value and risk (Timed tariff beside Smart).

### 4.9 `cost_effect_summary` (exact) — NEW (M5)

Five rows in this order: `home_import_cost`, `public_charge_cost`, `unrecovered_energy_value`, `unserved_travel_value`, `total`. Columns: `component`, `source_column` (the `cost_effect` column), `unit="GBP"`, `mean`, `p10`, `p50`, `p90` (across worlds, each component separately), `evidence_kind`. Consumer: Cost components table; the view says component quantiles do not add up.

### 4.9a `not_recovered_summary` (exact) — NEW (decision 0004 item 45)

Three rows in this order; columns `metric`, `unit`, `world_count`, `mean`, `p10`, `p50`, `p90`: statistics across worlds of the 4.8 column, worlds with NaN left out (`world_count` says how many were used). Consumer: the Overview and Cost not-recovered tiles (follow-up UI task).

| `metric` | `unit` |
| --- | --- |
| `unrecovered_kwh` | `kWh per week` |
| `unrecovered_share` | `fraction` |
| `sessions_affected_count` | `sessions per week` |

### 4.9b `timed_cost_effect_summary` and `timed_not_recovered_summary` (exact) — NEW (decision 0007, model step 2)

Exactly 4.9's and 4.9a's schemas, built by the same two functions (`cost_effect_summary`, `not_recovered_summary`) from `timed_cost_effect` (4.8a) instead of `cost_effect`. Both `None` when `timed_cost_effect` is `None`.

### 4.10 Trading overlay and Supplier frames — NEW (trading contract v1 §5, §9; task W3-market)

The exact columns, formulas and rules are in `docs/contracts/trading-events-v1.md` §5 (F1, F2) and §9 (I3–I5, I6); this section lists what an action result carries and where the implementation adds to or narrows that text. Every frame is action-only (`None` on a no-action result) and illustrative; the only net money figure is "illustrative simulated trading P&L" (decision 0005), never Axle cash. Built by `model/market.py` (`run_trading`: `deviation_world_slot`, `position_updates`, `trading_ledger_world`, `trading_week_world`) and `model/summaries.py` (the rest).

| Field | Contract | Notes |
| --- | --- | --- |
| `deviation_world_slot` | trading §5.1 | Adds, after `metered_kw`: `expected_metered_kwh`, the aggregator's expected metered import at each night's day-ahead decision (`x_t` of §4.4, kWh per half-hour), and `expected_unmanaged_kwh`, `x^U_t` of supplier contract v1 §2 (the same day-ahead expected sessions with every session unmanaged, `ρ = 1`; `x^U − x` sums to 0 over each world-night), for the supplier P&L view |
| `position_updates` | §5.2 | Sampled worlds (`sampled_world_ids`), strategy `full`: the day-ahead row then one row per hourly intraday decision |
| `trading_ledger_world`, `trading_week_world` | §5.3, §5.4, §9.1 | Nine buckets and `net_gbp` (their exact sum), shock "of which" columns, volumes including `delivered_within_*` |
| `trading_ledger_summary`, `trading_kpis`, `trading_checks` | §5.5, §9.1 | `trading_checks` rows in order: `buckets_sum_to_net`, `perfect_foresight_zero_imbalance`, `deviation_identity`, `mean_intraday_revision`, `p10_ordering`, `baseline_class_fallback` (always present, 0 when unused), and an added informative `baseline_missing_nights` (below) |
| `shock_attribution_summary` | §5.6a | From the ledger's "of which" columns (`market_shocks` and `shock_summary` come from the events lane) |
| `open_position_profile` | §9.1 | |
| `flex_cost_curve`, `shape_premium_summary`, `household_value_distribution`, `household_value_summary`, `revenue_by_segment`, `supplier_positions` | §9.3a–e | `revenue_by_segment` has `all`, `cohort` and `zone` segments |
| `control_group_summary` | §9.5 | `None` while `trading.control_group` is off |
| `price_curve_source` (str), `user_price_curve` (DataFrame or None) | §9.4 | `"synthetic"` or `"user curve"` |

Implementation decisions to note (each labelled in code):

- **Too-short warm-up.** Night `n`'s baseline reads history nights `<= n - 2` only (§4.1, B3). A warm-up shorter than two nights (hand-built runs; the dialog's minimum is 3) leaves night 0 no history the day-ahead decision could know, so its baseline is 0 and nothing is sold or settled on it; `trading_checks` counts such nights in `baseline_missing_nights` rather than borrowing a later night (which would leak).
- **Household saving.** The home-import and public-charge parts of each EV's saving are its own linear share of `cost_effect`; the unrecovered-energy and unserved-travel values are not linear in EVs (the world values the fleet's net shortfall), so each world's value is split over EVs in proportion to their own positive shortfall. The parts sum to the world's columns exactly.
- **Records.** The trading-overlay records (`assumptions.TRADING_OVERLAY`, §7 and §9.9) are not dialog-editable yet; the UI task adds them to Edit assumptions with their causal tests. The string switch `trading.baseline_history` is `trading.baseline_trap` (0 = `warmup_unmanaged`, 1 = `flexed_study_nights`), and `trading.control_group` is 0/1. The day-ahead decision instant and gate closure reuse the price records `day_ahead_publication_local_hour` and `gate_closure_minutes`.
- **User price curve.** `market.validate_user_price_curve` and `market.apply_user_price_curve` (§9.4). The typical shape the offset replaces is the run's own generated mean day-ahead price per day class and London half-hour; unpublished half-hours are still ranked by the B4 rule on recent published (already shifted) days, not on `curve[h]` directly (a stated simplification, accepted by the lead).
- **Events.** `market.run_trading` takes the validated events table: each decision uses `events.trading_mask` at its own instant and settlement the full mask; each request's delivery and payment (`events.event_delivery`, against the scope's adjusted baseline, zones from their own warm-up history) is booked on its window's first slot, so on the night the window starts; the expected plan adds the planner signals of requests known at the decision for each EV's zone. `event_response_bands` receives the same scope baselines for its `baseline_kw` and `delivered_kw` rows. `event_delivered_mwh` is the signed delivery against the baseline (`events.event_delivery`'s `delivered_kwh`).
- **Non-response.** One uniform per (world, study night, EV) (`sampling.sample_non_response`, drawn after the prices) and one check in the kernel: a session ignores its plan when its uniform is below `max(own probability, control-outage share)`; the own probability is its archetype's, 1 for a control EV. The trader expects `rho_i` and does not know an outage in advance.
- **Units.** Action results' `units` gain `control_group` (bool).

The validator (`_validate_trading` in `tests/fixtures/result_contract.py`) checks column sets and dtypes, the deviation identity and settled-volume rule (settlement open everywhere when the result has no grid request), every ledger bucket recomputed from `deviation_world_slot` (grid-event payment and delivery, which need the unstored zone baselines, are checked for sign, zero without requests and being strategy-free) (and the full strategy's intraday trades from `position_updates` for sampled worlds), the day-ahead position `q = c x max(0, B0 - x)` on sampled worlds, week sums, the ledger summary, KPIs (except `flex_margin_gbp_per_mw_year`, whose per-world capacity the result does not store), shock attribution, the open-position profile, the shape premium and its identity with `cost_effect`, supplier positions, cost-curve monotonicity, household bin shares and value identity, and segment sums.

## 5. One-EV replay — NEW (M5, decision 0004 items 10, 20)

```python
def replay_one_ev(result: ForecastResult, unit_id: str, world_id: int | None = None) -> OneEvReplay
def replay_one_ev_bands(result: ForecastResult, unit_id: str) -> pd.DataFrame
```

In `model/individual.py`. Runs the vectorised kernel on a one-EV slice of the stored inputs in `replay_state`, with that EV's smart-charging inputs and public charging, so action results replay correctly (audit B2, B3) and clock-change weeks work. `world_id=None` means `result.representative_world_id`. Unknown `unit_id` or `world_id` raises `KeyError`. The UI caches by `(run, unit_id, world_id)` (audit O6).

`OneEvReplay` fields: `unit_id`, `world_id`, `is_representative_world` (bool), `traits` (`dict`: the `units` row), `intervals`, `plug_events`, `daily_audit`.

`intervals` (exact): one row per `(path_id, slot_index)`, both paths for action results.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `path_id` | object | `normal`/`selected` |
| `slot_index` | int64 | |
| `interval_start_utc`, `interval_end_utc` | datetime64[ns, UTC] | |
| `interval_start_london` | datetime64[ns, Europe/London] | |
| `location` | object | `home_plugged`, `home_unplugged`, `driving`, `away`, `public_charging` |
| `connected` | bool | Plugged in at home for the whole slot (shading) |
| `battery_soc_percent` | float64 | SoC at slot end, 0–100 |
| `closing_battery_kwh` | float64 | Stock at slot end |
| `home_import_kwh`, `public_import_kwh`, `unserved_travel_kwh` | float64 | kWh in the slot |
| `early_departure_shortfall_kwh` | float64 | Battery-side kWh the smart plan still owed when this EV left early, in the half-hour it left; 0 otherwise and always 0 on the normal path (item 38) |

`plug_events` (at least): the columns of `plug_in_events` (3.9) for this EV and world, **both paths** on action results.

`daily_audit` (exact): the readiness audit (USER_FEEDBACK M12): a home-charging target is not proof the car is ready for its trip. One row per `(path_id, local_date)`, `local_date` the session night; the departure, trip and SoC are the next morning's, the trip that night's charging prepares for (item 52).

| Column | Dtype | Meaning |
| --- | --- | --- |
| `path_id` | object | `normal`/`selected` |
| `local_date` | object (`date`) | Session night |
| `day_label` | object | `"Mon 28"` |
| `drives_today` | bool | The EV makes a trip the morning after this night |
| `departure_utc` | datetime64[ns, UTC] | First departure; NaT when no trip (also the One EV departure marker) |
| `departure_london` | datetime64[ns, Europe/London] | Same, London |
| `target_soc_percent` | float64 | Preferred pre-drive target, 0–100 |
| `departure_soc_percent` | float64 | Actual SoC at departure; NaN when no trip |
| `trip_energy_need_kwh` | float64 | Battery-side energy the day's trips need (outbound + return); 0 when no trip |
| `shortfall_kwh` | float64 | Battery-side travel energy not served that date (outbound + return); 0 when served |
| `departed_below_target` | bool | `departure_soc_percent` below target by more than 1e-6 points; False when no trip |

The per-EV normal-path intervals, summed over all EVs of a world, must equal that world's `fleet_world_intervals` within 1e-9 kWh (the audit's reconciliation test, now including action results).

`replay_one_ev_bands` (exact): long frame `metric` (`battery_soc_percent`, `connected_share`), `unit`, `path_id`, `slot_index`, `interval_start_utc`, `interval_end_utc`, `interval_start_london`, `world_count`, `p10`, `p50`, `p90`: this EV across worlds, for the "Show P10–P90 across weeks" option.

## 6. Run comparison — NEW (M5, decision 0004 item 6)

```python
def compare_runs(a: ForecastResult, b: ForecastResult) -> RunComparison
```

In `model/summaries.py`. Compares the headline path of each run (`selected` for action, `normal` for no-action) world by world, B − A.

Requirements: raises `ValueError` when `horizon_start_utc`, `study_slot_count` or `world_count` differ, because worlds cannot be paired. Otherwise it returns a comparison and sets `matched_futures = False` with a reason for each of: different `seed`; different `vehicle_count`. Both models draw the population and then the evaluation worlds from the one generator (the action model's price noise comes last), so a different `model` alone still pairs the same futures (item 38 removed the planning world that used to be drawn first). The view shows the reasons (design 4.4) and still shows the comparison.

`RunComparison` fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `matched_futures` | bool | Same seed and fleet size, so differences are paired per world |
| `mismatch_reasons` | `tuple[str, ...]` | Plain-text reasons; empty when matched |
| `path_id_a`, `path_id_b` | str | Headline path used for each run |
| `changed` | DataFrame (exact) | `name`, `label`, `unit`, `value_a`, `value_b` (object): every assumption (by `name`) or `settings_snapshot` key that differs, assumptions first in record order, then settings keys in snapshot order. Settings rows use the key as `label` and `""` as `unit` |
| `illustrative_total_available` | bool | False when either run is no-action; the view then says "Illustrative total unavailable: run A/B has no Axle action" |
| `difference_bands` | DataFrame (exact) | Shape of 4.1; per world B − A of the headline path, then quantiles |
| `weekly_differences` | DataFrame (exact) | `world_id`, `metric`, `unit`, `value_a`, `value_b`, `b_minus_a`; metrics `home_import_kwh` (`kWh per week`), `unserved_travel_kwh` (`kWh per week`), `median_plug_in_soc_percent` (normal path in both runs, from `plug_in_world_kpis`, every world; `percentage points`), and `illustrative_total_gbp` (`GBP`) only when `illustrative_total_available` |
| `kpis` | DataFrame (exact) | `metric`, `unit`, `p10`, `p50`, `p90` of `b_minus_a` across worlds |

## 7. London-time presentation rule

Store UTC; present London. Charts use `interval_start_london` (or `study_slots.day_label`/`local_time_label`) on the x axis and put UTC in hover; tables show both. Views never convert time zones themselves except to format a label. `has_clock_change` lets a caption say "clocks change on Sun 25 Oct; that day has 50 half-hours".

## 8. `Assumption` records — NEW (M3, decision 0004 item 21)

```python
@dataclass(frozen=True)
class Assumption:
    name: str                  # unique snake_case key, e.g. "public_charge_gbp_per_kwh"
    label: str                 # dialog/table label, e.g. "Public charge rate"
    value: float | int | str | tuple[float, ...]
    unit: str                  # "GBP/kWh", "kW", "percent", "fraction", "" for unitless
    evidence: str              # "source" | "illustrative" | "synthetic"
    source: str                # citation (report page, sheet row) or "illustrative choice"
    meaning: str               # one plain sentence
    editable: bool             # shown as a widget in Edit assumptions
    bounds: tuple[float, float] | None   # inclusive; required when editable and numeric
    group: str                 # dialog tab: "Fleet", "Driving", "Plugging & home charging",
                               # "Battery", "Public & prices", "Weather", "Simulation"
    affects: str               # what changes when it changes, for the help tooltip
```

`name`, `value`, `unit`, `evidence`, `source`, `meaning`, `editable` and `bounds` come from item 21; `label`, `group` and `affects` come from the design (section 3.3 and How it works ▸ Assumptions). Invariants: names unique; an editable numeric value lies inside its bounds; `source`-class records are not editable. Every result carries the records it ran with, so Compare and the stale chip can diff them.

### 8.1 CNZ context records

Four `evidence="source"`, `editable=False`, `group="Plugging & home charging"` records; `source` cites the CNZ report page (records in `assumptions.CNZ_CONTEXT`); `meaning` says "comparison context, not a calibration target".

| `name` | `value` | `unit` | Meaning |
| --- | --- | --- | --- |
| `cnz_median_plug_in_soc_percent` | 52.0 | `percent` | Reported median SoC at plug-in |
| `cnz_share_plug_ins_below_10_percent_soc` | 0.03 | `fraction` | Reported share of plug-ins below 10% SoC |
| `cnz_share_plug_ins_below_20_percent_soc_upper_bound` | 0.10 | `fraction` | Reported as "fewer than 10%" below 20% SoC, so an upper bound, not a point value |
| `cnz_weekday_plug_in_mode_local_hour` | 18 | `hour (London)` | Reported weekday plug-in mode, about 18:00 |

### 8.2 Per-archetype session source records (decision 0004 item 54, plan D-2, D-5)

Twelve `evidence="source"`, `editable=False`, `group="Plugging & home charging"` records in `assumptions.SESSION_SOURCE_REFERENCES`, two per archetype, named `<cohort_id>.source_plug_in_soc_percent` and `<cohort_id>.source_battery_kwh_per_plug_in`. They are literal `'Source archetypes'` cells checked against `docs/sources/cohort-crosswalk.md`, carried in `assumptions` of every result built from `model/assumptions.py` (the synthetic fixture omits them, and a direct `run_forecast` call has no records) for the Sessions lens's dashed markers and the Archetypes comparison table; a view shows no marker when a record is absent. Audit references only: modelled plug-in SoC stays a stock-flow output and neither value feeds or calibrates the simulation.

| Archetype | Plug-in SoC (`percent`, cell N as fraction × 100) | kWh per plug-in (`kWh battery-side per plug-in`, cell M) |
| --- | --- | --- |
| `average_uk` | 68 (N6) | 7 (M6) |
| `intelligent_octopus` | 52 (N7) | 22 (M7) |
| `infrequent_charging` | 18 (N8) | 37 (M8) |
| `infrequent_driving` | 73 (N9) | 4 (M9) |
| `scheduled_charging` | 68 (N10) | 7 (M10) |
| `always_plugged_in` | 68 (N11) | 7 (M11) |

M is battery-side (the crosswalk's "battery kWh/plug-in"), so it is marked against `energy_needed_kwh` of 3.10c.

## 9. View-to-field map

| View | Reads |
| --- | --- |
| Run bar | `model_label`, `seed`, `vehicle_count`, `world_count`, `study_start_local_date`, `evidence_kind`, `run_completed_at_utc` |
| Overview | `average_day_bands` (`group_id="fleet"`), `plug_in_summary.kpis`, `weekly_bands` (no-action unserved tile), `flexibility_weekly_summary` and `weekly_peak_summary` (At a glance flexibility tiles, item 54), `action_summary`, `cost_effect_summary`, `not_recovered_world_count`, `cohort_summary`; plus `timed_not_recovered_world_count` (decision 0007, model step 2) for the Timed tariff "N of M weeks materially not recovered" tile |
| Drivers ▸ Fleet week (flexibility metrics) | `deferrable_power_bands` (item 54: deferrable power by slack, replacing the time-slack chart) |
| Overview ▸ Key stats | `flexibility_weekly_summary`, `weekly_peak_summary` (coincidence factor), `cheapest_half_hour_summary` (item 54, plan E1) |
| Drivers ▸ Plug-ins | `plug_in_summary`, `plug_in_heatmap` (item 43; `plug_in_half_hour_heatmap` and `plug_out_heatmap` once the view moves to half-hours, item 54), CNZ assumptions, `plug_in_events` (Data expander; caption names the sampled weeks) |
| Drivers ▸ Sessions | `session_distribution_bands` (item 54, plan D-2; `departure_soc_percent_timed` when `timed_start_local_hour` is set, decision 0007), the section 8.2 source records |
| Drivers ▸ One EV | `units`, `replay_one_ev` (intervals, plug events, daily audit and departure markers, plus the timed path when present, decision 0007), `replay_one_ev_bands`, `representative_world_id` |
| Drivers ▸ Archetypes | `cohort_summary`, `average_day_bands` (cohort groups), `cohort_interval_bands` |
| Drivers ▸ Fleet week (flexibility metrics) | `flexibility_bands` (items 43, 46, 54): turn-up headroom (smart path) and movable energy (unmanaged) at the across-weeks spread; time slack rows in the data expander only |
| Drivers ▸ Fleet week | `fleet_interval_bands` (`path_id="normal"`), `fleet_interval_ev_bands` (across EVs, item 39), `cohort_interval_bands` (`connected_share` per archetype), `weekly_bands` (unserved caption), `study_slots` |
| Smart charging ▸ Plan | `action_summary`, `forecast_price_bands` (item 48; each world's own day-ahead path, P10–P90 across worlds), `smart_charging_summary` (price-paid KPI, plus the `timed` row when present, decision 0007), `price_band_shift` (plus `timed_kwh_*` when present), `fleet_interval_bands` (`home_import_kw`, both paths, plus `timed` when present) |
| Smart charging ▸ Response | `fleet_interval_bands` (both paths, plus `timed` when present, decision 0007), `difference_bands` (largest median cut and rise in a half-hour, normal vs selected only), `difference_weekly_bands`, `smart_charging_summary` (moved-share KPI), `weekly_peak_summary` (each week's own peak, herding caption, plus `timed` row when present; items 48–49), `action_summary` |
| Smart charging ▸ Value and risk | `cost_effect`, `cost_effect_summary`, `not_recovered_summary` (item 45), `not_recovered_world_count`, `smart_charging_summary` (early-departure row), `action_summary`, `public_charge_gbp_per_kwh` assumption; plus `timed_cost_effect`, `timed_cost_effect_summary`, `timed_not_recovered_summary`, `timed_not_recovered_world_count` (decision 0007, model step 2) for the Timed tariff saving beside Smart |
| Compare | `compare_runs(a, b)` |
| How it works ▸ Assumptions, Edit assumptions | `assumptions` (and `model/assumptions.py` defaults) |

## 10. Fixture and checker

`tests/fixtures/result_fixture.py` (SYNTHETIC FIXTURE, not model output) provides `make_result(model="action" | "no_action", evs=6, worlds=4, seed=42, clock_change=None | "autumn" | "spring", home_charger_kw=7.0)`, plus `replay_one_ev`, `replay_one_ev_bands` and `compare_runs` with the signatures above, and the validators `validate_result_v2(result)`, `validate_one_ev_replay_v2(replay, result)` and `validate_run_comparison_v2(comparison)`. The fixture's selected path is a toy stand-in for smart charging (charge only 02:00–06:00 London, or at once on the last study date), which gives some early departures. One flagged world gets a small synthetic public top-up on the selected path, so "home + public" import and the public cost component are non-zero.

The column specs, reference computations and validators live in `tests/fixtures/result_contract.py`; `result_fixture.py` re-exports every name, so imports from either module work. The validators are duck-typed (attribute and column checks, not `isinstance`), so M5's real-run test calls them unchanged. They raise `AssertionError` naming the field and rule that failed. Besides columns, dtypes, keys and ranges they check:

- `fleet_interval_ev_bands` has one row per metric, path and slot, ordered quantiles, SoC within 0–100 and per-EV kW within 0 and the largest home charging power;
- `flexibility_bands` has its row groups in order with the stated units and spreads, ordered quantiles, `flexible_power_kw` and `turn_up_headroom_kw` within 0 and (the most EVs plugged in that slot in any world) × the largest home charging power, `movable_energy_kwh` non-negative, all three zero wherever no world has a plugged-in EV, and `time_slack_hours` NaN exactly where `world_count` is 0;
- world-first recomputation of `fleet_interval_bands`, `weekly_bands`, `difference_bands`, `difference_weekly(_bands)` and `cost_effect_summary` from the stored per-world frames;
- unit relations in `fleet_world_intervals`: `home_import_kw = home_import_kwh / 0.5`, `total_import_kw = (home + public kWh) / 0.5`, `connected_share = connected_count / unit_count`, `battery_soc_percent = 100 × closing / capacity` (catches kWh stored as kW and counts stored as shares);
- archetype band means adding up to fleet band means for additive metrics;
- `plug_in_heatmap` has the 168 cells in order, non-negative ordered quantiles, and Σ `mean × day_count` equal to the mean fleet `plug_ins_per_ev_per_week` (`all`) from `plug_in_world_kpis`;
- plug-in histogram shares in [0, 1] with `share_mean` summing to 1; `plug_in_world_kpis` equal to a recomputation from `plug_in_events` for the sampled worlds; `plug_in_summary.kpis` and the `cohort_summary` plug-in columns equal to recomputations from `plug_in_world_kpis`;
- `price_band_shift` per-path energy columns are non-negative with ordered quantiles;
- `weekly_peak_summary` has the two path rows in order, ordered quantiles, a modal slot inside the study, a ratio of exactly 1 on the normal row and a coincidence factor within 0–1, and equals a recomputation from `fleet_world_intervals` and the one home charging power in `units`;
- `deferrable_power_bands` has its row groups in order with one row per slot, non-negative ordered quantiles, a `total` equal to the normal path's `flexible_power_kw` band, every other row at or below it, cumulative rows shrinking as N rises and `at_least_8h` equal to `8h_or_more`; `flexibility_weekly_summary` has non-negative ordered quantiles, the fleet charger capacity from `units`, a peak at least every slot's `total` quantile and at most the capacity, hours within the week, W per plugged-in EV at most the largest charger power and a modal slot inside the study; both are `None` exactly when `flexibility_bands` is;
- `plug_in_half_hour_heatmap` and `plug_out_heatmap` have the 336 cells in order, `slot_count` matching `study_slots`, NaN exactly where `slot_count` is 0, and non-negative ordered quantiles; the plug-in cells add up to plug-ins per EV and, when every world is sampled, equal a recomputation from `plug_in_events`; plug-outs are at least the closed sampled sessions and at most plug-ins + 1 per EV;
- `session_distribution_bands` has its rows in group × day type × metric × bin order with the stated units and edges, ordered shares within 0–1, NaN exactly where `world_count` is 0, `share_mean` summing to 1 over each metric's bins, plug-in time, SoC and need present in exactly the worlds with plug-ins and, when every world is sampled, equals a recomputation from `plug_in_events` and `units` (efficiency from `battery_added_kwh ÷ home_import_kwh`); when any section 8.2 record is present, all twelve are, each source and read-only;
- `price_relative_bands` and `cheapest_half_hour_summary` equal recomputations from `forecast_prices`;
- `smart_charging_world`, `smart_charging_summary` and `price_band_shift` equal recomputations from `fleet_world_intervals` and the price frames; early-departure columns are non-negative and zero on the normal path; `forecast_prices` has one world-major row per (world, slot), each published at 13:00 London the day before its delivery date; the fields removed by item 38 are absent;
- the trading overlay and Supplier frames of 4.10. The fixture's trading frames use a SYNTHETIC toy baseline and positions (not the BL01-lite rules) and derive the ledger and every summary with the validator's own reference arithmetic.

Importing: tests under `tests/` import `from fixtures.result_fixture import make_result, validate_result_v2`. That works when `tests` is on `sys.path`; the lead adds `"tests"` to `[tool.pytest.ini_options] pythonpath` so view tests in `tests/ui/` can import it too.
