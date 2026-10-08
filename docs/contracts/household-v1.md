# Household and customer analysis contract v1

Status: draft for lead acceptance, 29 September 2026 (task contract-household), revised after the lead's review of `4397276` (blocking B1–B10, rulings Q1–Q2 and optionals O1–O9 applied; §13). Design only: no model or view code exists for anything in §3–§10 below. Base `main` at `26025b1` (J2 merged at `cea4c67`; the market follow-up with `household_frames`' `per_ev` return merged at `79bd22b`); branch-only code read: `claude/contract-replay` at `3c923b1`.

Authority: Mike, 29 September 2026: "we need analysis for the individuals like each unit has an availability profile with p5, p10, p50 etc bands and any other interesting metrics if we are not already, think of things households would want to see, comparisons etc, things suppliers would want to see, things they could market etc, like costs savings etc. and also at the cohort level. we may already have these, but if not, plan, accept plan and implement. also keep thinking about ux/ui." Read with decision 0004 items 54 (screens and filtering), 58 (supplier views, value per household), 62 (supplier and partner views), 65 (horizons, partner-facing measures, SoC at departure) and 66 (replay customer view); decisions 0003 (no aggregate without an approved component set; missing terms unavailable, not zero), 0005 (the only trading net) and 0006 (the only supplier net); the lead's household rulings of 30 September 2026 (`docs/plans/2026-09-30-overnight-review-log.md`: yearly claims from each customer's own average week; "If customers paid day-ahead prices"; "firm" only for the across-weeks P10); `docs/contracts/results-v2.md` §1 (every rule applies to every frame here), §3.9–3.10c, §4.8, §5; `docs/contracts/trading-events-v1.md` §9.3c–d and §10.2a, §10.5d–e, §10.5g; `docs/contracts/supplier-v1.md` §0, §3, §4, §12; `docs/contracts/replay-v1.md` §2; `docs/contracts/intraday-dispatch-v1.md` §7.3; `docs/DASHBOARD_DESIGN.md` §3.4, §3.5, 390 px rules; `tests/ui/test_copy_rules.py`.

**Reference convention.** `R§n` is a section of results-v2, `T§n` of trading-events-v1, `S§n` of supplier-v1, `P§n` of replay-v1, `D§n` of intraday-dispatch-v1; a bare `§n` is this document.

Every £, kg and kW figure below is illustrative and synthetic. Nothing here is a bill, a tariff, a settlement, a bid or Axle cash. This contract adds no random draw, no physics and no new money formula: every per-EV figure is a per-EV reading of an aggregate the run already produces, and the two frames it adds reconcile to those aggregates exactly (§8).

## 0. Terms

| Term | Meaning |
| --- | --- |
| EV, unit, customer, household, device | One row of `units`; the same thing seen by the driver, the supplier and a device maker. `unit_id` is the key everywhere |
| Treated `T` | EVs not in the hold-out control group (`units.control_group`, T§9.5; all EVs when the column is absent). Group statistics (§3.2) use treated EVs only, as T§9.3c and S§0 do; a control EV keeps its own row in §3.1 and its own card (§3.3), labelled "control group, not in the product" |
| Group | `fleet` (all treated EVs) or a cohort with at least one treated EV, in source order (R§1 rule 9): the groups `summaries.household_frames` already uses |
| `u_it`, `m_it` | The EV's home grid import in study slot `t` on the unmanaged (`normal`) and smart (`selected`) path, kWh (the per-EV chunk-pass arrays, `physics.UNIT_INTERVAL_QUANTITIES`) |
| `pub^p_it` | Its public grid import on path `p` (public top-ups, decision 0004 items 32 and 37) |
| `P_wt` | That world's day-ahead price after all shocks and the user curve, £/MWh: `forecast_prices.wholesale_forecast_gbp_per_mwh`, the price `cost_effect` values the customer leg at (R§4.8, plan B3) |
| `r_pub` | The `public_charge_gbp_per_kwh` record (decision 0004 item 4) |
| `I_wt` | The illustrative synthetic carbon intensity of S§3.8, gCO₂/kWh per (world, slot), from `supplier.carbon_intensity_gco2_per_kwh` (lane S1, Q2); absent until S1 lands |
| `slot_count`, `N_n` | `len(study_slots)` (336 under R§2, never written as a literal) and the night count `study_slots.night_index.max() + 1` (7); `night_starts` the first slot of each night, so a clock-change night's 46 or 50 slots are handled by `numpy.add.reduceat` / `maximum.reduceat` on `night_starts`, never by a reshape |
| Session | A maximal run of fully connected slots from `summaries.find_sessions` (a run starting at slot 0 was already plugged in at the horizon start and has no observed plug-in, R§3.9); **closed** when it ends inside the study (`end_slot < slot_count`), the population T§10.5e's charge completion uses |
| Session end | A slot that is connected and whose next slot is not (or the horizon ends), **including** a session running at the horizon start: the population of `summaries.sessions_affected_per_world`. Sessions and session ends are two populations and are never mixed in one ratio (B9) |
| Evening window | The study slots whose London wall-clock label lies in `availability.PRODUCT_WINDOWS["evening"]`, 17:00–21:00, on every night (T§10.5a); at 1 h it never runs past the study end |
| Typical week | A statistic taken per simulated week first and then quantiled across weeks. On screen every weekly figure with a P10–P90 across weeks is worded "in a typical week" |
| EV mean | One EV's mean over the simulated weeks of a weekly value (NaN weeks left out): "this customer's average week". Every per-year and per-month claim is built from EV means (B1), never from one week's spread × 52 |
| Two spreads | **Across weeks**: quantiles over worlds of a per-world value (forecast uncertainty). **Across EVs**: per world, quantiles over the group's EVs of each EV's own value, then the median (or the full statistics) across worlds (population spread in a typical week, the R§3.6b pattern; each statistic's median comes from its own week, so captions say "medians across weeks", never "the median week", O1). **Across EV means**: quantiles over the group's EVs of each EV's mean over weeks (population spread of average years; weeks are inside the average, so it has no across-week spread). Every frame and caption names which |
| Horizon | S§0 vocabulary: `week_ahead` (the spread across simulated weeks) or `scenario` (a month or year scaled from the average week, not a forecast). Weekly → yearly is × 52, weekly → monthly × 52 ÷ 12, the S§0 rule |
| Pass-through reading | The customer's saving plus its share of the trading product's customer revenue share, as if the day-ahead price reached the customer (T§9.3c `value_i`, S§3.1). The supplier's flat-tariff reading (S§3.1) values the same shifted energy from the other side; the two are never added. Every marketing claim uses the pass-through reading (Q1) |

`kWh ÷ 1000 × £/MWh` is the only money formula for energy; public energy is `kWh × £/kWh`. World first, quantiles last (R§1 rule 3): every `mean`, `p10`, `p50`, `p90` below is `numpy.quantile(..., method="linear")` over per-world values (or, where the row says so, over EV means), NaN left out and `world_count` saying how many were used (`summaries._world_stats`).

## 1. Gap analysis: what exists, what is specified elsewhere, what is missing

Checked on `main` at `92a8df8` and re-checked at `26025b1`. "Exists" means model code and, where said, a view on `main`.

### 1.1 Per EV

| Question a household or supplier asks of one EV | State | Where |
| --- | --- | --- |
| What did one week look like: plugged in, SoC, charging, top-ups, both paths | Exists (view) | `model/individual.py:replay_one_ev` → `OneEvReplay.intervals`, `plug_events`; `ui/views/one_ev.py:render_one_ev` |
| Was the car ready each morning: target vs departure SoC, trip need, shortfall (one week) | Exists (view) | `individual.py:_daily_audit` (USER_FEEDBACK M12); One EV "Daily readiness" tab |
| This EV's SoC and plugged-in spread across weeks | Exists (P10/P50/P90 only) | `individual.py:replay_one_ev_bands`; One EV band |
| Traits (archetype, miles, battery, charger, target) | Exists | `forecast.py:_units_frame`; `one_ev.py:_traits_caption` |
| Locked to the day-ahead plan or re-planning intraday | Specified | D§7.3 `units.dispatch_locked` (lane K4, not merged) |
| What the charger knew, planned and saved as one week unfolded | Specified | P§2 `replay_one_ev_timeline` (lane R1c, not merged): plan in force, cumulative cost, saving to date, one world |
| **Availability profile with P5/P10/P50/P90/P95 across weeks (turn-down and turn-up kW by half-hour)** | **Missing** | J2 (merged at `cea4c67`) computes per-EV contributions `a`, `ā`, `â` (`availability.ev_contributions`, T§10.2a) but reduces them to fleet, maker and per-EV across-world *sums and sums of squares* (`availability.accumulate`, `_add_ev_moments`) and per-window means for the sub-fleet curve; no per-EV quantile survives the chunk loop |
| **Money across weeks: value and saving £/week bands, an average year, cost per kWh, share of charging in cheap hours** | **Missing** | The per-EV saving exists as an array inside the chunk pass (`summaries.household_chunk`, `household_saving_gbp`: (world, EV) £/week) and the per-EV value is returned by `household_frames` as `per_ev["value_gbp_per_month"]` (S§4.1, merged at `79bd22b`) for the partner frames; no per-EV row is stored, no cost per kWh, no cheap-hours share |
| **Readiness across weeks: share of sessions complete at departure, departure SoC, times affected** | **Missing per EV** | `sessions_affected_per_world` sums over EVs before storing (R§4.8); T§10.5e `charge_completion_summary` (lane J5, not merged) is per group, not per EV |
| **This EV vs its archetype vs the fleet (percentile rank, a "typical household")** | **Missing** | Nothing ranks an EV inside its group |
| Per-EV settled volume and revenue-share payment, one sample week | Specified | T§10.5d `settlement_file` (J5): the representative world only; a CSV, not a view of one EV |
| CO₂ shifted per EV | Missing per EV | S§3.8 `carbon_shift_world` (S1) is fleet ÷ `N_T`, an average, not per EV |

### 1.2 Per cohort and fleet

| Question | State | Where |
| --- | --- | --- |
| Plug-in timing, SoC, sessions per archetype vs CNZ and the workbook | Exists (views) | `summaries.cohort_summary`, `average_day_bands`, `cohort_interval_bands`, `session_distribution_bands`; `ui/views/archetypes.py`, `sessions.py` |
| Spread across EVs of SoC and home import per half-hour (fleet) | Exists | `summaries.fill_ev_bands`, `fleet_interval_ev_bands` (R§3.6b); Fleet week |
| Value per household per month: distribution across EVs and summary (mean, P10/P50/P90 across EVs, share worse off, saving vs revenue share) per group | Exists (model only) | `summaries.household_frames` → `household_value_distribution`, `household_value_summary` (T§9.3c). **No view on `main` reads them**: the Supplier page (plan I5 / S§11) has not landed (`ui/registry.py` has five pages) |
| Trading buckets by archetype and zone | Exists (model only) | `summaries.revenue_by_segment` (T§9.3d) |
| Fleet deliverable MW bands (P05/P10/P50/P90 across weeks), firm share, sub-fleet curve | Exists (model only, J2) | `availability.build_frames` → `availability_bands`, `firm_share_by_fleet_size` |
| Probability and floor cost of a guaranteed £X/month, by group | Specified | S§4.3 `household_value_exceedance` (S1) |
| Gross flexibility cash per device, share earning, £/kW of charger, cycles, dispatch success, by cohort and maker | Specified | S§4.2 `partner_world`, `partner_summary` (S1); Supplier ▸ 5 Partners (S2) |
| Supplier gain per customer per month (before fee) | Specified | S§3.3 `supplier_pnl_world` (S1): fleet ÷ `N_T`, not by cohort |
| Charge completion by departure, per group and path | Specified | T§10.5e (J5) |
| SoC at departure, both paths, per group | Specified | S§4.6 two `session_distribution_bands` metrics (S1) |
| **Distribution across EVs, per group, of: value and saving per week and per average year, cost per kWh, cheap-hours share, readiness, sessions affected, nights plugged in, flexible kW per customer** | **Missing** | Only `value` per month (T§9.3c) has an across-EV distribution, and only in a typical week (medians across weeks); everything else is a fleet sum or a per-world share |
| **Marketable claims with honest bands ("a typical customer's average year is worth £X–£Y", "N% of sessions ready at departure")** | **Missing** | No frame or view states a customer-facing claim with its across-customer and across-week spread |
| **Who gains least (by cohort; the P10 customer)** | **Missing** | `household_value_summary.p10_across_evs` exists for value only, and no view shows it |
| Key stats rows for households | Missing | `ui/views/key_stats.py` has sections A–F; S§6 adds G, T§10.10 adds H; no household section |

### 1.3 What this contract therefore adds

Two stored frames and one on-demand object, built from arrays the run already produces:

1. `household_ev_world` (§3.1): one row per (world, EV) of weekly outcomes: energy, cost, saving, value, cheap-hours energy, sessions, readiness, CO₂ and evening flexibility. The cheap all-EV route.
2. `household_outcomes_summary` (§3.2): per group, three readings of each outcome: the across-EV distribution in a typical week, the session- or energy-weighted group ratio, and the distribution across customers' average weeks (the yearly and monthly claims).
3. `HouseholdCard` (§3.3): for one EV on demand, its outcomes across weeks with cohort and fleet comparison and percentile rank, its availability profile with P5–P95 across weeks and the share of weeks it is available (the on-demand one-EV route, the `replay_one_ev_bands` precedent), its plug-in and departure timing, and reliability figures.

Reused, not rebuilt: `household_saving_gbp`, `allocation_weights` and `household_frames`' `per_ev` (T§9.3c, S§4.1) for the saving and the value; the price-third rule of `price_band_shift` and the per-session values of `session_distribution_rows`, each behind a small helper the lead extracts (§4, B7); `find_sessions` and T§10.5e's completion rule; `sessions_affected_per_world`'s rule; J2's 1-h evening window means, kept per EV by a small `availability.py` hunk (§4, B5) so no contribution is computed twice; `availability.ev_contributions` for the one-EV profile; S§3.8's intensity for CO₂ (Q2); `household_value_summary` for the median-week share worse off; `supplier_pnl_summary` (S1) for the supplier's per-customer figure; `charge_completion_summary` (J5) as a cross-check.

## 2. Design in one paragraph

Per-EV × world × slot arrays cannot be kept for every EV (§5), so the design has two routes. In the existing per-EV chunk loop (`summaries._per_ev_chunks`, ten worlds at a time) a new `model/household.py` reduces each chunk's per-EV arrays to **weekly scalars per (world, EV)**, about twenty-five numbers per EV per world, and stores them as `household_ev_world`; `household_outcomes_summary` is then a reduction of that frame over groups, world first for weekly readings and EV-mean first for yearly ones. Everything a cohort, the fleet, the marketing view and the Key stats need comes from those two frames, and the validator recomputes one from the other. For **one EV** the card runs the kernel on that EV alone for every world (`kernel_slice(state, all worlds, [ev])` and `physics.simulate_unit_intervals`, exactly as `replay_one_ev_bands` runs `simulate_fleet_intervals` today), which is exact because EVs are physically independent (`summaries.KernelSlice`), and computes the EV's own availability contributions with J2's `ev_contributions` using the flexibility and availability inputs the result now keeps on `ReplayState` (B8), its sessions with `find_sessions`, and quantiles across worlds. The card reads its money and readiness rows from `household_ev_world` (no kernel run for those) and its comparison columns from `household_outcomes_summary`. Nothing is computed in a view.

## 3. Frames and the card

### 3.1 `household_ev_world` (exact) — action results; `None` exactly when `household_value_summary` is `None`

One row per `(world_id, unit_id)`, world-major, EVs in `units` order; `world_count × vehicle_count` rows. Weekly totals over the study slots unless the column says otherwise. Every row is one EV in one simulated week.

| Column | Dtype | Meaning (per world `w`, EV `i`) |
| --- | --- | --- |
| `world_id` | int64 | Key |
| `unit_id`, `cohort_id` | object | Keys; `cohort_id` repeated from `units` so a group filter needs no join |
| `treated` | bool | `not units.control_group` (True for every EV when the column is absent) |
| `earning` | bool | `household_frames`' `per_ev["earning"]` (S§4.1): the EV delivered turn-down in at least one settled slot of the week (a physical test, not the sign of its cash) |
| `home_import_kwh_normal`, `home_import_kwh_selected` | float64 | `Σ_t u_it`, `Σ_t m_it` (kWh per week, grid side) |
| `cheap_import_kwh_normal`, `cheap_import_kwh_selected` | float64 | `Σ_{t ∈ low_w} u_it`, `Σ_{t ∈ low_w} m_it`: the home import that fell in the world's **low** day-ahead price third. `low_w` is each world's own price third by the rule of `summaries.price_band_shift` (R§4.6: the study half-hours split at that world's 1/3 and 2/3 linear quantiles; `low` strictly below the first), through the helper `summaries.price_third_bands(price) -> (world, slot)` of `"low"`/`"middle"`/`"high"` that `price_band_shift` itself calls after the lead's hunk (§4, B7), so the two cannot drift |
| `public_import_kwh_normal`, `public_import_kwh_selected` | float64 | `Σ_t pub^p_it` |
| `home_cost_gbp_normal`, `home_cost_gbp_selected` | float64 | `Σ_t u_it P_wt ÷ 1000`, `Σ_t m_it P_wt ÷ 1000`: the customer leg at the day-ahead price (pass-through reading, B3); the difference is the EV's share of `cost_effect.illustrative_selected_minus_normal_energy_cost_gbp` |
| `public_cost_gbp_normal`, `public_cost_gbp_selected` | float64 | `pub × r_pub` |
| `saving_gbp` | float64 | `summaries.household_saving_gbp` for this EV, bit for bit (£ per week, positive = the driver pays less): minus its home and public cost differences and its pro-rata share of the world's unrecovered-energy and unserved-travel values (the T§9.3c rule, R§4.10 "Household saving") |
| `value_gbp_per_month` | float64 | `household_frames`' `per_ev["value_gbp_per_month"]` (S§4.1, merged at `79bd22b`), bit for bit: `(saving + allocated customer revenue share) × 52 ÷ 12`, T§9.3c's `value_i`, the pass-through reading. Stored per EV so `household_value_summary` is now a recomputation from this frame (validator). The share itself is not stored; it is `value × 12 ÷ 52 − saving` (validator, B6) |
| `supplier_energy_saving_gbp` | float64 | `home_cost_gbp_normal − home_cost_gbp_selected`: the energy component of the supplier P&L (S§3.2 component 1) for this customer under the flat-tariff reading; exact because that component is linear in EVs. The hedge-error and grid-event components are fleet cash and are **not** allocated (§12 Q5). Values the same shifted energy as `saving_gbp` from the other side; never added to it |
| `session_count` | int64 | Closed sessions (§0) in the week; connection is the same on both paths (decision 0004 item 32), so one count serves both |
| `session_end_count` | int64 | Session ends (§0) in the week: the `sessions_affected_per_world` population, so `sessions_affected_count` always reads "x of y session ends" (B9). `≥ session_count` |
| `nights_plugged_count` | int64 | Nights with at least one fully connected slot: `numpy.maximum.reduceat(connected, night_starts, axis=1)` summed over nights (0–`N_n`; a clock-change night counts once whatever its slot count, B4) |
| `completed_count_normal`, `completed_count_selected` | int64 | Closed sessions whose battery stock at the last connected slot on that path is at least `target_i − 1e-6 kWh` (`NOT_RECOVERED_TOLERANCE_KWH`; T§10.5e's rule, so J5's group shares are a cross-check). `target_i = physical_capacity_kwh × preferred_target_soc_fraction`, the preferred target, not the physical ceiling |
| `sessions_affected_count` | int64 | Session ends where the selected stock is below the normal stock by more than 1e-6 kWh (`sessions_affected_per_world`'s expression summed over slots only): "how often smart charging left me with less". Sums over EVs to `cost_effect.sessions_affected_count` |
| `departure_soc_mean_normal`, `departure_soc_mean_selected` | float64 | Mean over closed sessions of `100 × stock at the last connected slot ÷ capacity` (0–100, a stock-flow output); NaN when `session_count` is 0 |
| `co2_shifted_kg` | float64 | `Σ_t (u_it − m_it) I_wt ÷ 1000` (positive = less CO₂ under smart), the per-EV term of S§3.8's `co2_shifted_kg` (an exact split, since the formula is linear in EVs). NaN until the run carries S1's intensity (§10, Q2) |
| `evening_turn_down_kw_1h`, `evening_turn_up_kw_1h` | float64 | J2's per-EV 1-h evening window mean of the **realised** contribution `a_i(t, 2)` (T§10.2a, selected path), read from `availability` totals' `evening_ev_kw` (§4, B5; the `window_means` `accumulate` already forms for the sub-fleet curve, kept per EV for the 1-h cells): kW this EV could hold for an hour when called, on average over the evenings of the week. 0 for a control EV, a non-responder's sessions and a maker-outage night (they take no instructions, T§10.2a). NaN until J1b's per-EV plan outputs land (J2 is merged; its hook is inert until then) |
| `evidence_kind` | object | `"illustrative_synthetic"` |

Why weekly scalars and not per-slot rows: a household reads its week in totals (what I paid, how often I was ready), the supplier's marketing reads distributions of those totals across customers, and the totals reconcile exactly to the fleet frames the other views show. Per-slot per-EV values stay in the one-EV route (§3.3, §5).

**Model step 2 (decision 0007).** On a run whose `timed_start_local_hour` is set, the frame additionally carries, in the same naming pattern as the `_normal`/`_selected` columns above: `home_import_kwh_timed`, `cheap_import_kwh_timed`, `public_import_kwh_timed`, `home_cost_gbp_timed`, `public_cost_gbp_timed`, `completed_count_timed`, `departure_soc_mean_timed`. Each is the timed path's own figure, read the same way as the selected path's (the timed path's own per-EV kernel pass, same session boundaries as `session_count`/`session_end_count`, since plug-in and unplug are exogenous). `co2_shifted_kg`, `session_count`, `session_end_count`, `nights_plugged_count` and `sessions_affected_count` are **not** redefined for it: CO2 shift and sessions-affected stay the selected-versus-normal reading. Absent, not NaN-filled, on a run without the timed path, so that run's frame is schema-identical to before this existed.

**`household.metric_values(household_ev_world) -> pd.DataFrame`.** The one place the per-(world, EV) metric values of §3.2 are formed from the columns above (ratios, scalings, differences): `world_id`, `unit_id`, `cohort_id`, `treated`, then one float64 column per metric. The summary, the card and the validator's recomputation all call it, so a ratio is defined once.

### 3.2 `household_outcomes_summary` (exact) — action results; `None` exactly when §3.1 is

Three readings of each outcome per group. One row per `(group_id, metric, statistic)`: groups `fleet` then each cohort with treated EVs in source order; metrics in the order of the table below; statistics in the order of the three sets below, each set only where the table allows it.

**Set A, "in a typical week" (across EVs per world, then across weeks).** `mean_across_evs`, `p10_across_evs`, `p50_across_evs`, `p90_across_evs`: per world `w`, the statistic over the group's treated EVs of the per-(world, EV) value (linear quantiles; NaN EVs left out; NaN when none); then `world_count` (worlds with a value), `mean`, `p10`, `p50`, `p90` across worlds. `p50_across_evs` × `p50` is "the median customer in a typical week"; because each statistic's median comes from its own week there is no one "median week", and captions say "medians across weeks" (O1); `mean_across_evs` × (`p10`, `p90`) the average customer's week-to-week range. **`week_ahead` metrics only.**

**Set R, the group ratio (per world Σ numerator ÷ Σ denominator, then across weeks).** `group_ratio`: for a ratio metric, the group's session-weighted or energy-weighted share in the world (numerator and denominator in the table; NaN when the denominator is 0), then the statistics across worlds. A `_difference` ratio metric's group ratio is the selected ratio minus the normal ratio inside the world. **Ratio metrics only** (B2): this is the reading a claim about "sessions" or "energy" must use, because a mean of per-customer shares weights a one-session customer like a seven-session one.

**Set B, across customers' average weeks (EV means first, then across EVs).** `mean_of_ev_means`, `p10_of_ev_means`, `p50_of_ev_means`, `p90_of_ev_means`, `share_of_ev_means_below_zero`: per treated EV, its mean over worlds of the per-(world, EV) value (NaN worlds left out; NaN if none); then over the group's EVs (NaN EVs left out) the mean, the linear quantiles and the share below 0. These rows have **no across-week spread**: the weeks are inside each EV's average, so `mean = p50 =` the statistic, `p10 = p90 = NaN`, and `world_count` is the smallest number of worlds any of the group's EVs was averaged over. **Every metric**, and the only reading a per-year or per-month claim may use (B1): "8 in 10 customers' average years are worth £P10–£P90". `share_of_ev_means_below_zero` on a money metric is "customers whose average year is worse than nothing", the yearly counterpart of `household_value_summary.share_worse_off` (which is per week).

| `metric` | `unit` | `horizon` | Per-(world, EV) value (`metric_values`) | Ratio for set R (numerator ÷ denominator) | `better_is` |
| --- | --- | --- | --- | --- | --- |
| `value_gbp_per_week` | `GBP per week` | `week_ahead` | `value_gbp_per_month × 12 ÷ 52` (the S§0 conversion inverted) | | `higher` |
| `value_gbp_per_month` | `GBP per household per month` | `scenario` | `value_gbp_per_month` | | `higher` |
| `value_gbp_per_year` | `GBP per year` | `scenario` | `value_gbp_per_month × 12` | | `higher` |
| `saving_gbp_per_week` | `GBP per week` | `week_ahead` | `saving_gbp` | | `higher` |
| `saving_gbp_per_year` | `GBP per year` | `scenario` | `saving_gbp × 52` | | `higher` |
| `supplier_energy_saving_gbp_per_week` | `GBP per week` | `week_ahead` | `supplier_energy_saving_gbp` | | `higher` |
| `supplier_energy_saving_gbp_per_month` | `GBP per customer per month` | `scenario` | `× 52 ÷ 12` | | `higher` |
| `home_cost_gbp_per_kwh_normal`, `home_cost_gbp_per_kwh_selected` | `GBP per kWh` | `week_ahead` | `home_cost_gbp_p ÷ home_import_kwh_p`, NaN when 0 | `Σ home_cost_gbp_p ÷ Σ home_import_kwh_p` (energy-weighted) | `lower` |
| `home_cost_gbp_per_kwh_difference` | `GBP per kWh` | `week_ahead` | selected − normal inside the (world, EV) | ratio difference | `lower` |
| `home_import_kwh_per_week_normal`, `home_import_kwh_per_week_selected` | `kWh per week` | `week_ahead` | `home_import_kwh_p` | | `` |
| `public_import_kwh_per_week_normal`, `public_import_kwh_per_week_selected` | `kWh per week` | `week_ahead` | `public_import_kwh_p` | | `lower` |
| `cheap_share_normal`, `cheap_share_selected` | `fraction` | `week_ahead` | `cheap_import_kwh_p ÷ home_import_kwh_p`, NaN when 0 | `Σ cheap_import_kwh_p ÷ Σ home_import_kwh_p` | `higher` |
| `cheap_share_difference` | `fraction` | `week_ahead` | selected − normal | ratio difference | `higher` |
| `completed_share_normal`, `completed_share_selected` | `fraction` | `week_ahead` | `completed_count_p ÷ session_count`, NaN when 0 sessions | `Σ completed_count_p ÷ Σ session_count` (session-weighted, B2) | `higher` |
| `completed_share_difference` | `fraction` | `week_ahead` | selected − normal | ratio difference | `higher` |
| `sessions_per_week` | `sessions per week` | `week_ahead` | `session_count` | | `` |
| `session_ends_per_week` | `session ends per week` | `week_ahead` | `session_end_count` | | `` |
| `sessions_affected_count` | `session ends per week` | `week_ahead` | `sessions_affected_count` | | `lower` |
| `sessions_affected_share` | `fraction` | `week_ahead` | `sessions_affected_count ÷ session_end_count`, NaN when 0 (B9) | `Σ sessions_affected_count ÷ Σ session_end_count` | `lower` |
| `nights_plugged_share` | `fraction` | `week_ahead` | `nights_plugged_count ÷ N_n` | `Σ nights_plugged_count ÷ (N_n × ev_count)` | `` |
| `departure_soc_mean_normal`, `departure_soc_mean_selected` | `percent` | `week_ahead` | `departure_soc_mean_p` | | `higher` |
| `co2_shifted_kg_per_week` | `kg CO2 per week` | `week_ahead` | `co2_shifted_kg` | | `higher` |
| `co2_shifted_kg_per_month` | `kg CO2 per customer per month` | `scenario` | `× 52 ÷ 12` | | `higher` |
| `evening_turn_down_kw_1h`, `evening_turn_up_kw_1h` | `kW` | `week_ahead` | the §3.1 columns | | `higher` |

Columns: `group_id`, `metric`, `statistic`, `unit`, `horizon`, `better_is` (object: `higher`, `lower` or `""`, a reading aid so a view can word a rank without knowing the metric), `ev_count` (int64, treated EVs in the group), `ev_value_count` (int64: for set A and R rows the smallest per-world count of the group's EVs with a value, for set B rows the count with a defined EV mean; equal to `ev_count` unless the metric is NaN for some EVs, as `completed_share_*` and `departure_soc_mean_*` are for an EV with no closed session, for example one plugged in all week, whose absence would otherwise be silent, O4), `world_count` (int64), `mean`, `p10`, `p50`, `p90` (float64), `evidence_kind`. About 7 groups × (31 metrics × 5 set-B rows + 26 weekly metrics × 4 set-A rows + 11 ratio metrics × 1 set-R row) ≈ 1,900 rows.

Why a `scenario` metric has set B only: one week's across-EV P10–P90 multiplied by 52 is not a yearly range (a customer's bad week and good week average out over a year), so no row exists from which a view could read one (B1). Why weekly metrics have set B too: "the average customer's typical cost per kWh over the year" is a fair claim, and it is the EV-mean reading.

The `_difference` metrics exist so the Key stats "Δ vs unmanaged" column reads a paired difference the model supplied (a per-EV, per-world subtraction before any statistic) and never subtracts two rows.

**Model step 2 (decision 0007).** On a run whose `timed_start_local_hour` is set, these metrics join the table above, appended after the base set (so the base rows never move or renumber): `home_cost_gbp_per_kwh_timed`, `home_import_kwh_per_week_timed`, `public_import_kwh_per_week_timed` (same ratios and horizons as the `_normal`/`_selected` rows of the same figure), `cheap_share_timed`, `completed_share_timed` (ratio metrics, so they also get a set-R `group_ratio` row) and `departure_soc_mean_timed` (`percent`, `week_ahead`). No `_difference` metric for the timed path: that reading stays selected versus normal. All three sets (A, R where applicable, B) apply the same way as the base metrics. Absent on a run without the timed path.

### 3.3 `HouseholdCard`: one EV on demand

```python
def household_card(result, unit_id: str) -> HouseholdCard      # model/household.py
```

Unknown `unit_id` raises `KeyError`. Cached on `result.replay_state.replay_cache` under `("household", unit_id)` through `individual._remember`, a private import accepted with a note in the code ("the simpler choice; promote to a public helper when a third caller appears", O6), so revisiting an EV costs nothing. Requires §3.1 (raises `ValueError("household frames are not on this result")` on a no-action result or one without them, and the view shows the no-action message instead).

```python
@dataclass(frozen=True)
class HouseholdCard:
    unit_id: str
    cohort_id: str
    treated: bool                       # False: control group, not in the product
    dispatch_locked: bool | None        # units.dispatch_locked (D§7.3); None when the column is absent
    world_count: int
    outcomes: pd.DataFrame              # §3.3a, exact
    availability: pd.DataFrame | None   # §3.3b, exact; None until J1b lands or when ReplayState lacks the inputs
    availability_day: pd.DataFrame | None
    session_timing: pd.DataFrame        # §3.3c, exact
    reliability: pd.DataFrame           # §3.3d, exact
```

**3.3a `outcomes`.** One row per metric of §3.2, in that order. This EV across weeks, beside its groups.

| Column | Meaning |
| --- | --- |
| `metric`, `unit`, `horizon`, `better_is` | As §3.2 |
| `world_count`, `mean`, `p10`, `p50`, `p90` | **`week_ahead` metrics:** this EV's per-world values (`metric_values` rows for this EV), statistics across worlds ("in a typical week"). **`scenario` metrics:** `mean` is the EV mean scaled (its average week × 52 or × 52 ÷ 12: "about £52 a year"); `p10`, `p50`, `p90` are NaN, so no view can present one week's spread × 52 as a yearly range (B1) |
| `cohort_p10`, `cohort_p50`, `cohort_p90` | **`week_ahead`:** its cohort's set-A `p10_across_evs`, `p50_across_evs`, `p90_across_evs` rows at their `p50` (the archetype's customers in a typical week, each quantile at its own median across weeks). **`scenario`:** the cohort's set-B `p10_of_ev_means`, `p50_of_ev_means`, `p90_of_ev_means` (the archetype's customers' average years) |
| `fleet_p10`, `fleet_p50`, `fleet_p90` | The same for `fleet` |
| `cohort_rank_p50`, `fleet_rank_p50` | **`week_ahead`:** per world, the share of the group's treated EVs (with a value) whose value is **at or below** this EV's, then the median across worlds (0–1; 1 = at the top). **`scenario`:** the one rank of this EV's mean among the group's EV means (no median to take). NaN for a control EV and where this EV's value is NaN. With `better_is`, the view words it ("ahead of 62% of its archetype" for `higher`; "72% of its archetype pay more" is `1 − rank` for `lower`, a monotone transform of a median and so exact) |

No kernel run: everything above is a read of §3.1 and §3.2 for this EV.

**3.3b `availability` and `availability_day`.** This EV's realised 1-h contribution on the selected path (T§10.2a `a_i(t, 2)`, both directions), from one `simulate_unit_intervals` run on `kernel_slice(state, every world, [ev], selected=True)` and `availability.ev_contributions` with the inputs the result keeps on `ReplayState` (B8, §4): `replay_state.flexibility` (the run's `FlexibilityInputs`: `expected_end_slots(hours_to_departure[:, [ev]])`, `power_kw[ev]`, `target_kwh[ev]`, `efficiency`) and `replay_state.availability_inputs` (`decision_slots(study_slots, decision_local_time)`, `blackout`). The one-EV slice reproduces the EV's part of the fleet run exactly (`KernelSlice`; tested against the chunk arrays, §9). `None` when either input is `None` on the result, and until J1b's per-EV plan outputs exist (`ev_contributions` needs them); the card is still built and the view says "Available after the firm-MW model lands".

`availability` (exact): one row per `(direction, slot_index)`, directions `turn_down` then `turn_up`; columns `direction`, `slot_index`, `night_index`, `interval_start_utc`, `interval_start_london`, `unit` (`"kW"`), `world_count`, `available_share` (float64: the share of worlds in which the EV's kW is above 0 in this slot, O2: "how often this driver is on the driveway and able to move charging at this time"; NaN with `world_count` 0), `mean`, `p05`, `p10`, `p50`, `p90`, `p95`, `firm_share` (`p10 ÷ p50` across weeks, NaN when `p50` is 0; the T§10 convention, so "P90 firm" is the `p10` column; the only place this card uses the word firm, B10), `evidence_kind`. Quantiles across worlds of the EV's per-world kW; NaN with `world_count` 0 where the window runs past the study end (the last night's last half-hour at 1 h). This is the "availability profile with P5, P10, P50 bands" for one unit. P5 and P95 are added here (and nowhere else) because a household or a supplier reading one device wants the tails; fleet bands keep R§3.6's P10–P90 plus J2's P05.

`availability_day` (exact): the same folded to a typical day, one row per `(direction, day_type, local_half_hour)`, `day_type` weekday/weekend/all: per world the `nanmean` over the day type's nights' slots at that London half-hour (the `average_day_bands` rule, R§3.8: two slots on the repeated autumn hour, none on the skipped spring one), then the same statistics across worlds, `available_share` included (the share of worlds whose day-type mean is above 0); columns `direction`, `day_type`, `local_half_hour`, `local_time_label`, `unit`, `world_count`, `available_share`, `mean`, `p05`, `p10`, `p50`, `p90`, `p95`, `firm_share`, `evidence_kind`. NaN when no slot of that day type has the half-hour.

**3.3c `session_timing`** (exact). When this driver plugs in and leaves, pooled over its sessions in every world. One row per `(metric, bin_index)`, metrics `plug_in_time` then `departure_time`, 48 half-hour bins from 00:00 (the R§3.10c bins and labels); columns `metric`, `unit` (`hour (London)`), `bin_index` (0–47 from 00:00), `profile_order` (int64, 0–47 from 12:00: the session-night order the view plots in, so an overnight habit is one hump and not two ends, B3), `bin_lower`, `bin_upper`, `bin_label`, `session_count` (int64: sessions observed for the metric, pooled over worlds), `share` (float64: sessions in the bin ÷ `session_count`; sums to 1 over a metric's bins; NaN when `session_count` is 0), `evidence_kind`. `plug_in_time` counts every session of §0 (the normal path's `find_sessions`, the same sessions on both paths); `departure_time` only closed ones (the unplug is the departure, decision 0004 item 51). **Pooling rule:** one EV has about five to seven sessions a week, too few for a per-week share to mean anything, so sessions are pooled across the simulated weeks, as `price_relative_bands` and `cheapest_half_hour_summary` pool world-nights (R§4.7a–b); the caption says "pooled over N weeks (M sessions)". This is a description of the driver's habit, not a forecast band, and the frame carries no P10/P90.

**3.3d `reliability`** (exact). One row per metric, in order; columns `metric`, `unit`, `sample_kind` (`"weeks"` or `"sessions pooled across weeks"`), `sample_count` (int64), `mean`, `p10`, `p50`, `p90` (float64; over worlds for `weeks`, over pooled sessions otherwise), `p10_label`, `p50_label`, `p90_label` (object: `"HH:MM"` on the two clock-time rows, `""` elsewhere), `ratio_p10_to_p50` (float64: `p10 ÷ p50`, NaN when `p50 ≤ 0`; on the across-weeks kW rows this is the EV's own firm share), `evidence_kind`.

**Clock-time quantiles (B3).** Each session's clock time is taken as hours since the night's noon (`(clock − 12) mod 24`: 18:00 is 6, 02:30 is 14.5, the `cheapest_half_hour_summary` rule), the quantiles are taken on those, and each quantile is converted back (`(q + 12) mod 24`) for the decimal value and the `HH:MM` label. So a driver who plugs in at 23:00, 23:30 and 00:30 reads a P50 of 23:30, not the 23:00 a plain clock quantile would give (tested with plug-ins either side of midnight, §9).

| `metric` | `unit` | `sample_kind` | Value |
| --- | --- | --- | --- |
| `nights_plugged_share` | `fraction` | weeks | `nights_plugged_count ÷ N_n` per world |
| `sessions_per_week` | `sessions per week` | weeks | `session_count` per world |
| `session_ends_per_week` | `session ends per week` | weeks | `session_end_count` per world (the "of y" in "x of y session ends") |
| `plug_in_time` | `hour (London)` | sessions | Decimal London clock of the first fully connected slot, quantiled on hours since noon |
| `departure_time` | `hour (London)` | sessions | Decimal clock of the unplug (closed sessions), the same rule |
| `dwell_hours` | `hours` | sessions | Fully connected half-hours × 0.5 (closed sessions) |
| `energy_needed_kwh` | `kWh battery-side` | sessions | `max(target − stock at plug-in, 0)` |
| `flexible_kwh_per_night` | `kWh battery-side` | weeks | Per world `Σ` over closed sessions of `min(energy needed, dwell × power × efficiency)` (R§3.10c's `flexible_kwh`) `÷ N_n` |
| `evening_turn_down_kw_1h`, `evening_turn_up_kw_1h` | `kW` | weeks | The §3.1 columns for this EV; `ratio_p10_to_p50` is its evening firm share across weeks. NaN until §3.3b exists |

Every per-session value comes from `summaries.session_values(...)`, the helper the lead extracts from `session_distribution_rows` (§4, B7), so the Sessions lens and this card agree by construction.

**Model step 2 (decision 0007): the card itself is unchanged.** `household_card`'s "cost per kWh, cheap-hours share, completion" for Timed tariff read through the existing `outcomes` (3.3a), which already generalises to whatever metrics `household_outcomes_summary` carries (the timed metrics above, when the run has them); no new field or kernel pass was needed there. `one_ev_arrays`/`build_card` (§3.3b/c/d: availability, session timing, reliability) stay the normal/selected-path reading only -- extending them to a third per-slot series for one EV is a separate, unscoped piece of work (the writer's handoff explains why).

**Lead hunks in files this contract does not own** (each a few lines; the lead adds them at HH1's integration, as T§10.7 does for call lines):

1. `summaries.session_values(sessions, *, study_slots, capacity_kwh, target_kwh, power_kw, efficiency) -> dict[str, np.ndarray]`: the seven R§3.10c per-session values that `session_distribution_rows` forms before binning (`plug_in_time`, `departure_time`, `plug_in_soc_percent`, `energy_needed_kwh`, `dwell_hours`, `flexible_kwh`, `slack_hours`, with the rounding and the open-session NaN rule), moved into a helper that `session_distribution_rows` then calls (B7). The card calls it on the one-EV sessions.
2. `summaries.price_third_bands(price: np.ndarray) -> np.ndarray`: the `low_edge, high_edge = np.quantile(price, (1/3, 2/3), axis=1, keepdims=True)` and `np.where(price < low_edge, "low", np.where(price > high_edge, "high", "middle"))` lines of `price_band_shift`, moved into a helper that `price_band_shift` calls (B7). `household.accumulate` uses `== "low"`.
3. `availability.new_totals` allocates `totals["evening_ev_kw"] = np.full((world_count, vehicle_count, 2), np.nan)` and `accumulate`, inside its direction loop at `duration_slots == 2` (the 1-h cell), stores `window_means[:, EVENING, :]` (`EVENING = list(PRODUCT_WINDOWS).index("evening")`) into `totals["evening_ev_kw"][worlds, :, direction_index]` (B5). About six lines; `build_frames` leaves the array in `totals`, and the lead passes `availability_totals["evening_ev_kw"]` to `household.build_frames` before the totals are dropped. No contribution is computed twice.
4. `summaries.ReplayState` gains `flexibility: FlexibilityInputs | None = None` and `availability_inputs: availability.AvailabilityInputs | None = None`, set by `forecast.run_forecast` (B8): `forecast` calls `summaries.flexibility_inputs(...)` once and passes the result to both `ReplayState` and `build_summaries` (which today builds it internally); `availability_inputs` stays `None` until J1a/J1b wire `AvailabilityInputs`. The card reads them; no rebuild.
5. The two `household` call lines in `build_summaries` and `_trading_summaries`, the two `ForecastResult` fields, the `validate_result_v2` hook line and the `make_result` line (§10).

**Chunk loop (`household.new_totals(world_count, vehicle_count, study_slots)` → `accumulate` → `build_frames`).** In `summaries.build_summaries`, inside `for worlds, by_path in _per_ev_chunks(...)`, after `sessions = find_sessions(...)` (so the normal-path sessions are reused, not found twice), the lead adds one call:

```python
household.accumulate(totals, worlds, by_path, sessions,
                     day_ahead=day_ahead_study[worlds], low_band=low_band[worlds],
                     intensity=None if intensity is None else intensity[worlds])
```

`low_band = summaries.price_third_bands(day_ahead_study) == "low"` and `intensity` (world, study slot, from `supplier.carbon_intensity_gco2_per_kwh` once S1 lands, else `None`) are formed once outside the loop. `accumulate` reduces the chunk to (world, EV) scalars and keeps nothing per slot: sums over the slot axis for energy and cost (`low_band[..., np.newaxis]` masking the cheap import; the two absolute home costs are formed here, both being needed for cost per kWh per path, rather than by extending `household_chunk`, which already forms their difference for the saving: the market lane's function stays untouched, the extra broadcast is bounded, and the validator ties the difference to `household_chunk`'s `home_cost_gbp`, O7); `numpy.maximum.reduceat(connected, night_starts, axis=1)` for nights plugged; `session_end_count` and `sessions_affected_count` from `sessions_affected_per_world`'s expressions summed over slots only; the selected-path stock gathered at `(sessions.world, sessions.end_slot − 1, sessions.ev)` for completion and departure SoC (one gather; `sessions.plug_out_kwh` already holds the normal path's) with `closed = sessions.end_slot < slot_count`; the intensity broadcast for CO₂. It takes no flexibility or availability argument (B5). `saving_gbp`, `value_gbp_per_month` and `earning` are not recomputed: `build_frames(totals, *, units, saving_gbp, per_ev, evening_ev_kw, groups)` takes the `(world, EV)` arrays `_trading_summaries` already forms (`household_saving_gbp(...)` and `household_frames`' third return `per_ev`, B6) and stores them, so §3.1 equals T§9.3c bit for bit (validator). The lead adds the `build_frames` call after the `household_frames` call in `_trading_summaries`, passing `availability_totals["evening_ev_kw"]` (or `None` before J1b, when every value is NaN).

**Card (`household.household_card`).** Outcomes and ranks from the two frames through `metric_values`. Availability and timing from one one-EV kernel run over every world (`simulate_unit_intervals` on the slice, both paths: the normal path's `connected` and stock for sessions, the selected path's arrays for `ev_contributions`), then `find_sessions`, `session_values`, the hours-since-noon quantiles and `numpy.quantile` across worlds. The run is cached with the card. The existing `replay_one_ev_bands` runs the same slice through `simulate_fleet_intervals`; the two runs stay separate in v1 (they return different shapes) and the writer notes the duplication for a later tidy (§11).

**No draw, no leakage.** Nothing here samples; two runs with one seed give identical frames. The card reads the run's stored inputs only.

## 5. Compute budget and storage (1,000 EVs × 100 worlds; the writer measures and reports)

| What | Size or cost | Choice |
| --- | --- | --- |
| Per-EV × world × slot arrays for every EV | 1,000 × 100 × 336 × 8 B = 269 MB per metric and direction | **Rejected** for storage: not practical for a local run (decision 0004 item 33 asks memory to stay practical) |
| Per-EV × world × 48-half-hour typical-day profile for every EV | 38 MB per (metric, day type); about 230 MB for two directions × three day types | **Rejected**: a profile is a one-EV view, so it is built on demand |
| `household_ev_world` | 100,000 rows × 25 numeric columns ≈ 20 MB, plus three key columns | **Chosen**: the all-EV route; on the result and not in `slim_run` (Compare does not read it, §6.4) |
| `household_outcomes_summary` | About 1,900 rows | Trivial |
| `totals["evening_ev_kw"]` in J2's accumulators | 100 × 1,000 × 2 × 8 B = 1.6 MB, freed with the totals | The B5 hunk; replaces a second `ev_contributions` call (about 30 MB of temporaries and a quarter of J2's per-chunk time, now avoided) |
| Chunk-loop additions per 10-world chunk | Reductions over the (10, 336, 1000) arrays already in memory (27 MB each); one (10, 336) bool mask; one `reduceat`; one gather; one intensity broadcast | Bounded by the existing chunk; no new per-slot storage across chunks |
| Card: one EV, every world | One `simulate_unit_intervals` run on (100 worlds, `slot_count`, 1 EV), `ev_contributions` on (100, `slot_count`, 1) at one duration, `find_sessions` and `session_values`: the `replay_one_ev_bands` precedent, cached | On demand, one EV at a time; the writer reports the measured seconds |

## 6. UI

Views read frames and the card; they filter, format and plot. They compute no quantile, share, ratio, physics or money. Every chart uses Plotly through `ui/style.py` (decision 0004 item 29); no pie charts (item 43). KPI tiles use `components/kpi.py` (at most four a row, label ≤ 28 characters with no unit); captions ≤ 140 characters with no internal citation or snake_case (`tests/ui/test_copy_rules.py`); "Unmanaged"/"Smart" for the paths (item 50); "Unavailable" for NaN, never 0. Both new lenses are added to `test_copy_rules.CHECKED_VIEWS` when they land.

**Wording rules that every household figure follows** (the reviewer checks them on the rendered app): a weekly figure with a P10–P90 across weeks says "in a typical week"; a band across customers is captioned "medians across weeks", never "the median week", because each statistic's median comes from a different week (O1); a figure whose metric is NaN for some customers says "of N customers with closed sessions" from `ev_value_count` whenever that is below `ev_count` (O4); a group figure for cost per kWh, readiness or any share reads the group ratio, never the mean of per-customer ratios (O9); a per-year or per-month figure is an average-week figure and its band, where shown, is across customers' average years ("8 in 10 customers' average years"); "firm" is used only for an across-weeks P10 and the P10 across customers is "P10 customer" (B10); a claim about sessions or energy reads the group ratio, not a mean of per-customer shares (B2).

### 6.1 Drivers ▸ Household (new lens, after One EV, before Archetypes)

```
Lens("Household", "What does this driver get: ready mornings, savings and flexibility?")
```

Why a lens and not more One EV: One EV is sketch 1, one car in one week, the physics; Household is the same car across every simulated week, the outcomes. Both lenses share one EV selectbox: HH3 extracts `one_ev.ev_picker(st, units) -> unit_id` (label `"EV"`, `options` in `units` order, One EV's `format_func`, default and `key="one-ev-unit"`) and both lenses call it, because two widgets with one key must have identical label and options or Streamlit resets the choice on a lens switch (O8); a small hunk in `one_ev.py`, owned by HH3 (K5 also touches that file, so the lead sequences). Switching lenses keeps the driver; Household has no week picker. Mounted through `pages._action_lens` (a no-action result gets the existing message and the switch-model button).

Layout, top to bottom (1440 px; at 390 px every row stacks, the small multiples keep two columns, legends sit below):

1. **Traits and status.** The One EV traits caption (`one_ev._traits_caption`, imported), then one `badge_line`: "In the product" or "Hold-out control group, not in the product" (`card.treated`), and, when `card.dispatch_locked` is not `None`, "Locked to its day-ahead plan" or "Re-plans hourly on the latest intraday price".
2. **KPI row (4 tiles).**
   - "Ready at departure": `completed_share_selected` `p50` as a percent; context "in a typical week P10–P90 80–100% · unmanaged 91%"; "Unavailable (no closed sessions)" for an EV with none, such as one plugged in all week (O4).
   - "Value per year": `value_gbp_per_year` `mean` as money (the EV's average week × 52); context "archetype's customers £38–£71 (P10–P90 average years) · illustrative" from `cohort_p10`, `cohort_p90` of that row. The weekly spread is in chart A, never here as a yearly range (B1).
   - "Sessions affected": `sessions_affected_count` `p50`; context "of 7 session ends in a typical week (P50)" from `session_ends_per_week` (B9).
   - "Evening turn-down": `evening_turn_down_kw_1h` `p50` in kW; context "firm (P10 across weeks) 3.4 kW · 1 h, smart path", or "Unavailable until the firm-MW model lands".
3. **Chart A "This driver, the archetype and the fleet, in a typical week"** (`make_subplots`, 2 × 2 small multiples, height `2 × small_multiple_panel + gap`): one panel each for Value per week (£), Smart cost per kWh (£/kWh), Ready at departure (%), Charging in the cheapest third (%), all `week_ahead` metrics. In each panel three horizontal rows, top to bottom: **This EV** (P10–P90 across weeks as a bar, P50 as a marker; teal, the smart-path colour, because it is this EV on the smart path), **Archetype** (`cohort_p10`–`cohort_p90` with `cohort_p50`: the archetype's customers in a typical week, medians across weeks; the archetype's `ARCHETYPE_COLOURS` entry) and **Fleet** (`fleet_*`; the unmanaged grey). Hover on the EV row: "ahead of 62% of its archetype" from `cohort_rank_p50` and `better_is`. Caption: "This EV: P10–P90 across 100 weeks. Archetype and fleet: P10–P90 across EVs, medians across weeks. Illustrative." Definition expander: the two spreads in two lines, the pass-through reading, and "a year is the average week × 52; its band is across customers' average years, on the tile". Data: `card.outcomes`.
4. **Chart B "Flexibility this driver can offer"** (height `time_series`): a direction segmented control [Turn-down | Turn-up] in the page body (not the header, which holds the EV picker). One panel over the week, London time axis (`style.london_time_axis`): P50 line (teal) with the P10–P90 band (`band_fill` at `BAND_ALPHA`) and the P5–P95 band (the same colour at half that alpha), from `card.availability`; `available_share` in the hover ("on the driveway and able to move charging in 74% of weeks", O2). Legend: "P50", "P10–P90", "P5–P95" (three entries). Caption: "1 h turn-down this EV could hold on the smart path. Bands: P10–P90 and P5–P95 across 100 simulated weeks. Illustrative." Data expander: `card.availability_day` (the typical day by day type, with `available_share`) and a two-line definition: what counts (plugged in through the hour, follows its plan, can still reach target) and the T§10 convention ("P90 firm is the P10 line"). When `availability` is `None`: `st.info("Available after the firm-MW model lands.")` and no chart.
5. **Chart C "When this driver plugs in and leaves"** (height `histogram`): one panel, the 48 London half-hour bins in `profile_order` (12:00 first, so an overnight habit is one hump), two bar series from `card.session_timing`: plug-in time (the `plugged` grey) and departure time (`MUTED_INK`); no teal (this is behaviour, not the smart path). Caption: "Share of this driver's sessions by London half-hour, pooled over 100 weeks (620 sessions). Unmanaged path." Under it one caption from `card.reliability`: "Plugged in 6 of 7 nights (P50); usually in by 18:30, out by 07:30 (P50); 11 kWh a night could move (P50)." (the labels from `p50_label`).
6. **Data and definition** expander (`chart_block` gives each chart its own; this last one is for the card): `card.outcomes` and `card.reliability` as tables; the definition names the sessions population, the session-end population and the horizon-end censoring (R§3.10c), and says a control EV's smart path equals its unmanaged path.

What a household wants to know and where it reads it: "Will my car be ready?" (tile 1, panel 3 of chart A, the reliability line); "How much do I save?" (tile 2 for the year, panel 1 of chart A for a typical week, with the honest range and "illustrative"); "How often was I affected?" (tile 3, "x of y session ends"); "Am I a good customer for this?" (chart B, its `available_share`, and the rank hover). What an Axle interviewer reads: the same numbers reconcile to the fleet frames (§8), the two spreads are named, and no view arithmetic.

### 6.2 Supplier ▸ 6 Customers (new lens, after "5 Partners" in the S§11 order)

```
Lens("6 Customers", "What can we tell customers, and which customers gain least?")
```

The supplier's marketing view. Partners (S§11) speaks to device makers; this lens speaks about the supplier's own customers. It lands after the Supplier page shell exists (plan I5 / S2). Controls: the shared group selector (`registry.GROUP_KEY`; `fleet` default, then cohorts with treated EVs). Every figure below is the chosen group's.

1. **Claims block.** A subheader **"If customers paid day-ahead prices"** (`st.subheader`, not a caption, Q1) above a KPI row of four claims, each with its honest band; the wording rules are fixed here so the view only formats:
   - "Typical value per year": `value_gbp_per_year` × `p50_of_ev_means` (the median customer's average year, pass-through reading: saving plus share); context "£38–£71 for 8 in 10 customers, each averaged over 100 weeks · illustrative" from `p10_of_ev_means` and `p90_of_ev_means` (B1).
   - "Sessions ready at departure": `completed_share_selected` × `group_ratio` × `p50` as a percent (session-weighted, B2); context "in a typical week P10–P90 88–95% · unmanaged 93%" (`p10`, `p90`; the normal metric's `group_ratio` `p50`).
   - "Smart charging cost": `home_cost_gbp_per_kwh_selected` × `group_ratio` × `p50` as p/kWh (`× 100`, a display constant; energy-weighted: the group's overall £/kWh, not the average of each customer's own, O9); context "in a typical week · unmanaged 9.1 p/kWh · day-ahead wholesale, synthetic".
   - "Charging in cheapest third": `cheap_share_selected` × `group_ratio` × `p50` as a percent (energy-weighted); context "in a typical week · unmanaged 34% · each week's own cheapest third of half-hours".
   Under the row, one caption: "Simulated fleet, synthetic prices, pass-through reading: what customers would pay if day-ahead prices reached them. Not a tariff."
2. **Chart "Value per year across customers, by archetype"** (height `histogram`): horizontal range bars, one per cohort in source order (their `ARCHETYPE_COLOURS`), each `p10_of_ev_means`–`p90_of_ev_means` with the `p50_of_ev_means` marker (set B, average years); the fleet row first in grey. The verdict caption names the least-gaining archetype by its P10 customer: "Least: Infrequent driving, P10 customer £4 a year (average over weeks)." Caption: "P10–P90 of customers' average years (100 weeks each) per archetype; marker: P50 customer. Illustrative." Data: the `value_gbp_per_year` set-B rows. The value-per-month histogram (`household_value_distribution`, per world then medians across weeks) stays on Supplier ▸ 3 (S§11) and is not repeated.
3. **Chart "Flexible kW per customer, by archetype"** (height `histogram`): grouped bars per cohort, two series (turn-down teal, turn-up the `difference` amber, both the smart path's), `p50_across_evs` × `p50` with whiskers from `p10_across_evs` × `p50` and `p90_across_evs` × `p50`, the whisker legend reading "P10–P90 customers, medians across weeks" and the hover "P10 customer 1.2 kW" (never "firm", B10). Caption: "Realised 1 h evening flexibility per customer, smart path; P10–P90 across customers, medians across weeks. Illustrative." `st.info("Available after the firm-MW model lands.")` while NaN.
4. **Value row (4 tiles).** "Supplier energy saving" (`supplier_energy_saving_gbp_per_month` × `mean_of_ev_means`; context "energy component only, flat-tariff reading; hedge and events are fleet cash"); "Saving in a typical week" (`saving_gbp_per_week` × `mean_across_evs` × `p50`; context "before the trading share; value = saving + share"); "Customers worse off" (`value_gbp_per_year` × `share_of_ev_means_below_zero` as a percent; context "average year below £0 · in a typical week x%" from `household_value_summary` `share_worse_off` `p50`); "CO₂ shifted per customer" (`co2_shifted_kg_per_month` × `mean_of_ev_means`; context "illustrative synthetic intensity tied to net demand", "Unavailable" until S1).
5. **Data and definition**: the group's `household_outcomes_summary` rows; the definition states the two readings (S§3.1), the three spreads, and that "customer" means a treated EV of the simulated fleet.

Marketing honesty rules (the reviewer checks them on the rendered app): every claim carries its band and "illustrative"; the yearly claim reads the median customer's average year and its band across customers' average years, never one week × 52; the readiness claim reads the session-weighted share and says "sessions", not "customers" or "mornings"; no £ figure appears without "synthetic prices" or "illustrative" in the same block; the supplier figure and the customer figure are never in one sum or one bar; nothing says "guaranteed" (the guaranteed-floor analysis is S§4.3 on Partners).

### 6.3 Overview ▸ Key stats: section I "Households" (after H, T§10.10)

Rows in this order, `fleet` group, each P50/P10/P90 across weeks of the named row; the Reference cell carries the horizon and, where useful, the set-B reading; "Δ vs unmanaged" reads a `_difference` row of the same statistic or is blank:

| Metric (label) | Unit | Row | Reference | Δ vs unmanaged | Verdict rule |
| --- | --- | --- | --- | --- | --- |
| Value per customer, typical week | £ per week (illustrative) | `value_gbp_per_week`, `mean_across_evs` | "pass-through; average week × 52 ≈ £X a year (P50 customer); P10–P90 customers £a–£b" from `value_gbp_per_year` set B | | spread rule (`_spread_verdict`) |
| Sessions ready at departure, smart | fraction | `completed_share_selected`, `group_ratio` | "week-ahead; session-weighted over N customers with closed sessions (`ev_value_count`); unmanaged P50 z" | `completed_share_difference`, `group_ratio` | "no worse than unmanaged" when the difference `p10 ≥ −0.01`, else "check early departures" |
| Smart charging cost | £ per kWh | `home_cost_gbp_per_kwh_selected`, `group_ratio` | "week-ahead; energy-weighted; day-ahead wholesale" | `home_cost_gbp_per_kwh_difference`, `group_ratio` | "cheaper" when the difference `p90 < 0` |
| Charging in the cheapest third | fraction | `cheap_share_selected`, `group_ratio` | "week-ahead; each week's own cheapest third" | `cheap_share_difference`, `group_ratio` | |
| Session ends affected | fraction | `sessions_affected_share`, `group_ratio` | "of N session ends per customer (P50)" from `session_ends_per_week` `mean_across_evs` | | "risk low" when `p50 < 0.05`, else "" |
| Nights plugged in | fraction | `nights_plugged_share`, `group_ratio` | | | |
| Evening turn-down per customer | kW | `evening_turn_down_kw_1h`, `mean_across_evs` | "week-ahead; 1 h, smart path; P10 customer x kW (medians across weeks)" | | "Unavailable until firm-MW J1b" when NaN |
| CO₂ shifted per customer | kg per week | `co2_shifted_kg_per_week`, `mean_across_evs` | "scenario: × 52 ÷ 12 per month; illustrative synthetic intensity" | | "Unavailable until supplier S1" when NaN |

The 0.05 threshold reuses `LOW_RISK_SHARE` (section E's "early departures under 5% of sessions"), now applied to session ends.

### 6.4 Compare

No change in v1. `slim_run` does not carry the household frames (Compare pairs fleet aggregates; a per-customer distribution across two runs is not a paired difference, the S§7 side-by-side reasoning). A later task may add the Key stats section I rows as `compare_runs` records once the frames are on both runs; it would read `household_outcomes_summary` only.

### 6.5 Interview notes

One paragraph in `docs/explainers/interview-notes.md` (lane S4 may fold it into the supplier demo): Drivers ▸ Household on the default EV ("ready 9 sessions in 10, an average year worth about £50, could hold 3 kW for an hour most evenings"), then Supplier ▸ 6 Customers ("if customers paid day-ahead prices, a typical customer's average year is worth £X–£Y; the least-gaining archetype is Z; never add the supplier and customer figures").

## 7. Labelling (decisions 0003, 0005, 0006; S§12; the lead's household rulings)

- Every money column is `*_gbp`, illustrative, from synthetic prices; every frame carries `evidence_kind = "illustrative_synthetic"`.
- `saving_gbp` and `value_gbp_per_month` are the pass-through reading (T§9.3c); every marketing claim uses it under the header "If customers paid day-ahead prices" (Q1). `supplier_energy_saving_gbp` is the flat-tariff reading's energy component (S§3.1, decision 0006). The two are never added, never drawn in one bar, and each block that shows one names its reading.
- No aggregate is introduced: the only nets remain decision 0005's trading P&L and decision 0006's supplier gain. `supplier_energy_saving_gbp` is one component of the latter, shown as a component and labelled so; it is not a per-customer supplier net.
- Per-year and per-month figures are average-week figures (× 52, × 52 ÷ 12), carry `horizon = scenario`, and their bands are across customers' average years; a weekly band is worded "in a typical week" (B1, the 30 September ruling).
- "Firm" names an across-weeks P10 only; a P10 across customers is "P10 customer" (B10).
- The revenue share inside `value` is an allocation of fleet settlement by settled flexibility (T§9.3c), never a per-household settlement; the Data expander says so.
- A control EV's card says it is outside the product; group statistics exclude it.
- CO₂ figures say "illustrative synthetic carbon intensity tied to the model's own net demand; not observed grid data" (S§3.8).
- Availability per EV is labelled "realised, smart path, 1 h" and the T§10 P-value convention is stated once per lens.
- Nothing here is Axle cash, a bid, a tariff or a guarantee.

## 8. Validator rules (`validate_result_v2` additions; `tests/fixtures/household_contract.py`)

`validate_household_v2(result)`, called by `validate_result_v2` (one hook line the lead adds), and `validate_household_card_v2(card, result)`. Tolerances `1e-9 × max(1, |value|)` unless stated.

- **Presence.** Both frames `None` on a no-action result and `None` exactly when `household_value_summary` is `None`; otherwise exact columns, order and dtypes as §3.1–§3.2; keys unique; `household_ev_world` has `world_count × vehicle_count` rows, world-major in `units` order; `treated == ~units.control_group` (all True without the column).
- **Fleet reconciliation per world** (against `fleet_world_intervals` and `cost_effect`): `Σ_i home_import_kwh_p` equals the path's weekly `home_import_kwh` and `Σ_i public_import_kwh_p` the weekly `public_import_kwh`, within 1e-6 kWh; `home_cost_gbp_selected − home_cost_gbp_normal` equals `household_chunk`'s `home_cost_gbp` per (world, EV) (O7), so `Σ_i` of it is `illustrative_selected_minus_normal_energy_cost_gbp`; `Σ_i saving_gbp = −illustrative_selected_minus_normal_total_gbp` (the existing T§9.3c identity, 1e-6 £); `Σ_i sessions_affected_count = sessions_affected_count`; `Σ_i (value_gbp_per_month × 12 ÷ 52 − saving_gbp) = −Σ_n customer_revenue_share_gbp(w, full, n)` (`trading_ledger_world`) and `value_gbp_per_month × 12 ÷ 52 = saving_gbp` for every control EV (weight 0, B6).
- **T§9.3c and S§4.1 identity.** `value_gbp_per_month` and `earning` equal `household_frames`' `per_ev` arrays bit for bit; per group, the world-first `mean_value`, `p10_across_evs`, `p50_across_evs`, `p90_across_evs`, `share_worse_off`, `mean_customer_saving` and `mean_revenue_share` recomputed from `household_ev_world` (treated rows) equal `household_value_summary`'s rows.
- **Cheap band.** `cheap_import_kwh_p ≤ home_import_kwh_p`; with the lead's helper, `summaries.price_third_bands(day-ahead prices) == "low"` reproduces the mask the fixture's `price_band_shift_from` reference derives per world, and on a sampled world `Σ_i cheap_import_kwh_normal` equals that reference's per-world `normal` low-band energy summed over the week's nights (a weekly total cannot be read per night, O5).
- **Counts, nights and NaN rules.** `0 ≤ completed_count_p ≤ session_count ≤ session_end_count`; `sessions_affected_count ≤ session_end_count`; `0 ≤ nights_plugged_count ≤ N_n` with `N_n = study_slots.night_index.max() + 1`, and on the clock-change fixtures (`make_result(clock_change="autumn")`, `"spring"`) the count equals a per-night `any` recomputed from the fixture's per-EV replays over `night_starts` (B4); `departure_soc_mean_*` in [0, 100], NaN exactly when `session_count` is 0; `co2_shifted_kg` NaN in every row exactly when the result has no `carbon_shift_world`, otherwise `Σ_i co2_shifted_kg = carbon_shift_world.co2_shifted_kg` per world; `evening_*_kw_1h` NaN in every row exactly when `availability_world_slot` is absent or its 1-h `deliverable_kw` is all NaN (J1b not landed), otherwise `≥ 0`, 0 for control EVs, and `Σ_i evening_turn_down_kw_1h` equals the mean over the evening-window slots of `availability_world_slot.deliverable_kw` (`turn_down`, 1 h) per world within 1e-6 kW (the evening window never runs past the study end at 1 h, so the identity is exact; likewise `turn_up`).
- **Summary.** `household_outcomes_summary` equals the recomputation from `metric_values(household_ev_world)` over treated EVs: set A world-first, set R from the numerator and denominator sums, set B from EV means; rows in order; set A and R rows have ordered quantiles; set B rows have `mean = p50` and `p10 = p90 = NaN`; set A rows exist only for `week_ahead` metrics and set R rows only for ratio metrics; shares within [0, 1]; `ev_count` equals the group's treated count and `ev_value_count` its recomputation, at most `ev_count` and equal to it for a metric that is never NaN (O4); `_difference` rows equal recomputations of the per-(world, EV) subtraction (set A) or of the ratio difference (set R); `better_is` in {`higher`, `lower`, `""`}.
- **J5 cross-check** (once `charge_completion_summary` is on the result and `units.control_group` is all False, so the two populations coincide): `completed_share_p` `group_ratio` `p10`/`p50`/`p90` for `fleet` and each cohort equal J5's `completed_share_p10/p50/p90` rows (`departure = all`), and `Σ_i session_count` per world averaged equals `session_count_mean` (B2).
- **Model step 2 (decision 0007).** `*_timed` columns and metrics are present exactly when the result's `fleet_world_intervals` carries the timed path (`timed_start_local_hour` set) and absent otherwise, in both frames together (never one without the other); `Σ_i home_import_kwh_timed` reconciles to the timed path's weekly `home_import_kwh` the same way as `_normal`/`_selected` (fleet reconciliation above); `completed_count_timed ≤ session_count`; `departure_soc_mean_timed` in [0, 100], NaN exactly when `session_count` is 0; no `_difference` row exists for a `_timed` metric. A run without the timed path has both frames byte-identical (columns, order, dtypes, values) to a result built before model step 2 existed.
- **Card.** `outcomes` rows equal this EV's `metric_values` rows quantiled (weekly) or its EV mean scaled with NaN quantiles (scenario) and the summary's group rows copied (set A × `p50` for weekly, set B for scenario); ranks in [0, 1] or NaN (NaN for a control EV); `availability`: one row per (direction, slot), `p05 ≤ p10 ≤ p50 ≤ p90 ≤ p95`, `available_share` in [0, 1] and equal to the share of worlds with kW > 0 on the fixture, `firm_share = p10 ÷ p50` where `p50 > 0`, NaN with `world_count` 0 exactly on windows past the end; `availability_day` equals the fold of the per-world profile the test recomputes on the fixture; `session_timing` shares sum to 1 per metric or are all NaN with `session_count` 0, `profile_order` a permutation of 0–47 starting at the 12:00 bin; `reliability` rows in order, the clock rows equal the hours-since-noon recomputation with matching labels, `ratio_p10_to_p50` as defined.

The synthetic fixture: `household_contract.make_household_frames(result)` builds both frames from the fixture's per-EV replays (`make_result`'s `replay_one_ev` over every EV and world), its price frames and its toy ledger, and `make_household_card(result, unit_id)`; the lead wires one line into `make_result`. The fixture's frames pass the validator, including on `clock_change="autumn"` and `"spring"`; a real 30 EV × 4 week run passes `validate_result_v2` in the existing real-run test.

## 9. Tests (checklist; each lane adds the rows it implements)

Model (`tests/model/test_household.py`, `test_household_card.py`):

- [ ] **Hand fixture, one EV, two slots.** Prices [100, 50] £/MWh, `u = [10, 0]`, `m = [2, 8]`, `r_pub = 0.79`, one public top-up of 1 kWh on the smart path: `home_cost_normal = 1.00`, `home_cost_selected = 0.60`, `public_cost_selected = 0.79`, `supplier_energy_saving = 0.40`, `saving_gbp = −(−0.40 + 0.79) = −0.39`; low band = slot 1 (the world's lower third), `cheap_import_normal = 0`, `cheap_import_selected = 8`, so the cheap shares read 0 and 0.8.
- [ ] **Completion, readiness and the two populations.** A session that ends at target on both paths counts complete on both; one that leaves early on the smart path (stock below target − 1e-6) counts complete on the normal path only and adds one `sessions_affected_count`; a session still plugged in at the horizon end is in neither count; an EV plugged in all week (one run from slot 0) has `session_count` 0, NaN readiness and departure SoC, and lowers `ev_value_count` for those metrics by one while `ev_count` is unchanged (O4); a session running at slot 0 that ends in the study is in `session_end_count` and not in `session_count`, so `session_end_count = session_count + 1` on that fixture and no ratio can read "1 of 0" (B9); `departure_soc_mean` equals the hand mean.
- [ ] **Nights on clock-change weeks (B4).** On `make_result(clock_change="autumn")` (a 50-slot night) and `"spring"` (46), an EV connected once on every night gives `N_n`; connected on nights 0, 2, 3 gives 3; never connected gives 0 and NaN readiness; the code path uses `night_starts` and `slot_count`, never a reshape to 48 or the literal 336 (a grep-level assert in the test).
- [ ] **Control EV.** `treated = False`, `value × 12 ÷ 52 = saving` (share 0), absent from every group statistic, present in `household_ev_world`; its card has NaN ranks and the "not in the product" flag.
- [ ] **Reconciliation on a real run** (30 EVs × 4 weeks): every §8 fleet identity; `value_gbp_per_month` and `earning` equal `per_ev` bit for bit; `household_value_summary` recomputed from the frame bit for bit; the summary's three-set recomputation; set-B rows have NaN `p10`/`p90` and `mean = p50`.
- [ ] **Group ratio vs mean of shares (B2).** A two-customer fixture (one with one session, complete; one with six sessions, three complete) gives `completed_share_selected` `mean_across_evs` 0.75 and `group_ratio` 4 ÷ 7; the J5 cross-check identity once `charge_completion_summary` exists.
- [ ] **EV means (B1).** A fixture where one EV saves £10 in week 0 and −£8 in week 1 and another saves £1 in both: set-B `mean_of_ev_means` of `saving_gbp_per_year` is 52 × 1, `p10_of_ev_means` and `p90_of_ev_means` are both 52 × 1 (both EVs average £1), while the set-A `p10_across_evs` × `p10` of `saving_gbp_per_week` is below 0 (about −6): the yearly rows do not inherit one week's spread; `share_of_ev_means_below_zero` is 0 here and 0.5 when the first EV's weeks are £10 and −£12.
- [ ] **CO₂ and evening kW gating.** NaN exactly when the upstream inputs are absent; with them present, the `Σ_i` identities of §8 on the fixture (a two-maker, one-blackout fixture for the kW, so a blackout slot and an outage night give 0 for the EVs concerned); `evening_ev_kw` from the J2 totals equals `nanmean` over the evening slots of the 1-h `ev_contributions` realised array recomputed in the test (B5).
- [ ] **Slice identity.** For each EV of a 6-EV, 4-world run, the card's per-world evening kW (the card's own `ev_contributions` on the one-EV slice, from `ReplayState.flexibility` and `.availability_inputs`, B8) equals the EV's column of `ev_contributions` on the full chunk arrays and the EV's §3.1 column; its per-world sessions equal `plug_in_events`' rows for that EV where the world is sampled.
- [ ] **Card quantiles.** On a 3-EV fixture with known per-world values, `cohort_rank_p50` and `fleet_rank_p50` equal the hand ranks (weekly: per-world rank then median; scenario: the rank of EV means); `outcomes` P10/P50/P90 equal `numpy.quantile` of the EV's rows for weekly metrics and are NaN for scenario metrics with `mean` = the scaled EV mean; `availability_day` equals the hand fold on a clock-change fixture (two slots on the repeated hour); `available_share` equals the hand share.
- [ ] **Clock-time quantiles either side of midnight (B3).** Plug-ins at 23:00, 23:30 and 00:30 give `plug_in_time` `p50` 23.5 with label `"23:30"` (a plain clock quantile would give 23.0); departures at 06:30, 07:00, 07:30 give 7.0 `"07:00"`; `session_timing` `profile_order` puts the 23:30 and 00:30 bins adjacent.
- [ ] **Pooled timing.** `session_timing` shares sum to 1; a driver with every plug-in at 18:00 gives share 1 in that bin; open sessions count in `plug_in_time` and not in `departure_time`.
- [ ] **Helpers.** `session_distribution_rows` with `session_values` extracted reproduces its seeded bins bit for bit; `price_band_shift` with `price_third_bands` extracted reproduces its seeded frame bit for bit (B7, the lead's two hunks each carry this regression).
- [ ] **Cache and determinism.** Two `household_card` calls for one EV run the kernel once; two runs with one seed give identical frames and cards.
- [ ] **Editing (a test, not a validator; O5).** Editing a `CARBON` record changes only `co2_shifted_kg` and the rows that read it; editing `trading.customer_revenue_share` changes only `value_gbp_per_month` and the rows that read it; editing a `SUPPLIER` record (reward mode, flat reward, platform fee) changes nothing here; every kernel frame and `deviation_world_slot` are identical before and after.
- [ ] **Runtime and memory.** Per-chunk seconds added and the frame's bytes at 1,000 EVs × 100 worlds, and the card's seconds, measured and reported in the handoff.

UI (`tests/ui/test_household_view.py`, `test_customers_view.py`, AppTest on the fixture):

- [ ] Both lenses render with and without every optional part (`availability` `None`, CO₂ NaN, control EV chosen, a cohort with no treated EVs, a no-action result); no view arithmetic (the existing lint: no `numpy.quantile`, no subtraction of columns).
- [ ] The Household EV picker shares One EV's key and is built by the one `ev_picker` helper (identical label and options, O8): choosing an EV on one lens selects it on the other, and the choice survives a lens switch.
- [ ] Claims read the rows §6.2 names (a fixture with distinct set-B quantiles shows those three numbers in the yearly tile; a fixture where the mean of shares differs from the group ratio shows the group ratio in the readiness tile); the subheader "If customers paid day-ahead prices" is present; every £ block contains "illustrative" or "synthetic"; no tile or caption contains "× 52" applied to a P10 or P90 across weeks.
- [ ] Copy rules pass for both lenses (`CHECKED_VIEWS`); KPI labels ≤ 28 characters without units; at most one "Finding:" caption per lens; legends ≤ five entries; "firm" appears only beside an across-weeks P10.
- [ ] Key stats section I rows present on an action result, "Unavailable …" where the upstream frame is absent, absent on a no-action result.
- [ ] Browser pass at 1440 and 390 px (decision 0004 item 26): the 2 × 2 small multiples keep two columns; nothing clipped; legends below.

## 10. Build lanes and dependencies (disjoint ownership; the lead adds the call and hook lines at integration, as T§10.7)

Lane letters `HH` (household) avoid the events lanes' `H` and every letter in use.

| Lane | Outcome | Owns | Depends on | Risk and review |
| --- | --- | --- | --- | --- |
| **HH1** frames (High: a new result schema with per-customer money) | §3.1, §3.2, `metric_values`, §4's `new_totals`/`accumulate`/`build_frames`; §8 validators; the fixture builders | `src/axle_studio/model/household.py` (new), `tests/model/test_household.py`, `tests/fixtures/household_contract.py`. The lead adds the §4 hunks 1–5: `summaries.session_values` and `summaries.price_third_bands` (each with its bit-for-bit regression), `availability` `evening_ev_kw`, the two `ReplayState` fields, the call lines, the two `ForecastResult` fields (`household_ev_world`, `household_outcomes_summary`, `None` by default), the `validate_result_v2` hook line, the `make_result` line | **Starts now** (`household_frames`' `per_ev` return is on `main` at `79bd22b`, B6); integrates after the lead's §4 hunks land. `co2_shifted_kg` is NaN until HH1b; `evening_*_kw_1h` is NaN until J1b lands (J2 is merged; its hook is inert without the per-EV plan outputs) | Lead-accepted interface (this contract) and one strong independent review; a specialist second review only if the reviewer finds a reconciliation risk |
| **HH1b** CO₂ fill-in (Normal) | Per-EV CO₂ once S1 lands. **Q2:** `supplier.carbon_intensity_gco2_per_kwh(system_net_demand_gw, records) -> (world, slot)` is added to lane S1's scope (`supplier.py` is not yet built; S§3.8 defines the formula and S1 owns the function); HH1b reads it through the `intensity` argument of §4 | One hunk in `household.py`, the gated tests of §9 | S1 merged | One focused review |
| **HH2** card (Normal) | §3.3 complete: `household_card`, `HouseholdCard`, the one-EV run, timing and reliability, `validate_household_card_v2` | `household.py` card functions (after HH1 merges, same file), `tests/model/test_household_card.py`, the card hunk in `tests/fixtures/household_contract.py` | HH1; `ReplayState.flexibility` and `.availability_inputs` (B8); `availability`/`availability_day` gated on J1b (the card builds them only when the per-EV plan outputs exist) | One review; the slice-identity test is the load-bearing check |
| **HH3** UI (Normal) | §6.1 Household lens; §6.3 Key stats section I; the copy-rules list | `src/axle_studio/ui/views/household.py` (new), `tests/ui/test_household_view.py`; hunks: the `ev_picker` extraction in `ui/views/one_ev.py` (O8), the lens entry in `ui/registry.py` and `ui/pages.py`, section I in `ui/views/key_stats.py`, `CHECKED_VIEWS` in `tests/ui/test_copy_rules.py` (files S2, J6, K5 and R2 also touch: the lead sequences the hunks) | HH1 and HH2 fixtures | One review plus the browser pass |
| **HH4** Customers lens (Normal) | §6.2 | `src/axle_studio/ui/views/customers.py` (new), `tests/ui/test_customers_view.py`; the lens entry hunks | The Supplier page shell (plan I5 / lane S2) merged; HH1; `household_value_summary` (merged); `supplier_pnl_summary` (S1) only if a later tile reads it | One review plus the browser pass |

Order: HH1 now, integrated after the lead's §4 hunks; HH2 after HH1 merges; HH3 after HH2 (the Household lens can land on Drivers before any Supplier page exists); HH1b when S1 merges; HH4 after the Supplier page shell. Files this contract's lanes do **not** touch: `individual.py` (R1c's; the card imports `_remember` and adds nothing), `availability.py` and `summaries.py` (the lead's §4 hunks only), `supplier.py` (S1's).

## 11. Limitations (state in Limits and the captions where a viewer would look)

- Clock-time quantiles are anchored at noon, so a driver whose departures or daytime plug-ins fall either side of 12:00 has those times split across the anchor, as midnight did before; overnight sessions, the common case, are unaffected.

- Per-EV money is the pass-through reading at the day-ahead price and synthetic prices; no retail tariff is modelled (S§15). The supplier figure per customer is the energy component only.
- The revenue share inside `value` is an allocation of fleet settlement by settled flexibility, not a meter-level settlement (T§9.3c); a customer's "value" here is not a statement.
- Sessions still plugged in at the horizon end have no departure, so readiness, dwell and departure timing lean short (R§3.10c); the noon-to-noon horizon reduces this. Sessions and session ends are two populations (§0) and each ratio names its own.
- Timing distributions per EV pool sessions across simulated weeks: a habit, not a forecast band.
- "In a typical week" bands are week-to-week spread; "8 in 10 customers" bands are population spread, in a typical week (medians across weeks) for weekly figures and across average years for yearly ones. The percentile rank is within the simulated fleet, not a population claim. A yearly figure is an average week × 52; the run has no seasons (§12 Q9).
- Availability per EV is the realised 1-h figure on the smart path: 0 when the session ignores its plan or its maker is out that night, and NaN before J1b lands. Turn-up counts only to the preferred target (T§10.2a). `available_share` counts weeks with any positive kW, whatever its size.
- CO₂ uses S§3.8's illustrative intensity, not observed grid carbon.
- The card runs the one-EV kernel once for availability and timing while `replay_one_ev_bands` runs it again for the SoC band; the two are cached separately (a later tidy may share one run).
- Shared timing variation is not modelled (decision 0004 item 64), so per-EV availability bands may be narrow on timing.

## 12. Open questions, each with the answer (all recommendations accepted by the lead's review of `4397276`; 3 and 7 as qualified)

1. **Where the per-EV view lives.** A new Drivers ▸ Household lens sharing One EV's EV picker (accepted: One EV stays sketch 1, one week; Household is the same car across weeks).
2. **Where the marketing view lives.** Supplier ▸ 6 Customers after Partners (accepted: Partners is device-maker facing; this is the supplier's own customers).
3. **What "8 in 10 customers" means.** For weekly claims, P10–P90 across customers in a typical week, each quantile at its median across weeks (accepted for weekly claims only; O1 fixes the wording); for yearly and monthly claims, P10–P90 across customers' average weeks scaled (B1; §3.2 set B). Never a pooled quantile over every (world, customer) pair, which mixes the two spreads.
4. **"Cheap hours".** Each week's own cheapest third of half-hours by day-ahead price, the `price_band_shift` rule through the shared helper (accepted; a clock window is a tariff's choice and this run has no tariff).
5. **Supplier value per customer by cohort.** The energy component only, exact and labelled (accepted; supplier-v1 defines no allocation of the hedge-error and grid-event cash).
6. **Availability durations on the card.** 1 h only (accepted: the product's headline duration, S§3.9).
7. **Per-EV CO₂.** Reuse S1's intensity through `supplier.carbon_intensity_gco2_per_kwh`, now in S1's scope (accepted subject to Q2, which adds the function to S1).
8. **"Times affected" per EV.** `sessions_affected_count` over session ends and `completed_share` over closed sessions, each with its own denominator (accepted with B9); a per-EV `early_departure_count` output from J1b is deferred.
9. **Year figure.** The average week × 52, labelled a scenario (accepted; a seasonal weighting needs a season model the run does not have; Compare across start dates is the honest route, S§7).
10. **Percentile rank.** Per world, then the median, for weekly metrics; the rank of EV means for yearly ones (accepted with B1).
11. **Storage.** `household_ev_world` as a result frame the validator recomputes from (accepted, about 20 MB).
12. **P5 and P95 on the card only** (accepted).
13. **Key stats section letter.** "I. Households" after J6's "H. Firm MW" (accepted).

## 13. Review of `4397276`: what changed, and what remains open

Blocking, applied: B1 (set B statistics of EV means for every per-year and per-month claim, `share_of_ev_means_below_zero`, "in a typical week" wording, no yearly range from one week's spread on the card or the tile; §0, §3.2, §3.3a, §6, §7), B2 (`group_ratio` set R, session-weighted readiness labelled "Sessions ready at departure", the J5 cross-check as that identity; §3.2, §6.2, §6.3, §8), B3 (hours-since-noon quantiles with labels, `profile_order`, the midnight test; §3.3c–d, §9), B4 (`reduceat` on `night_starts`, `slot_count` and `N_n` throughout, clock-change tests; §0, §3.1, §8, §9), B5 (`evening_ev_kw` kept in J2's totals by a lead hunk; `household.accumulate` takes no flexibility or availability argument; §3.1, §4, §5), B6 (`value_gbp_per_month` and `earning` from `household_frames`' `per_ev`, the market follow-up now merged at `79bd22b`; `revenue_share_gbp` dropped; §3.1, §8, §10), B7 (`summaries.session_values` and `summaries.price_third_bands` as lead hunks with bit-for-bit regressions; §4, §9), B8 (`ReplayState.flexibility` and `.availability_inputs`; §3.3b, §4), B9 (`session_end_count`, `sessions_affected_share`, "x of y session ends"; §0, §3.1, §3.2, §6), B10 ("firm" only for an across-weeks P10; "P10 customer" elsewhere; §3.3b, §6, §7).

Rulings, applied: Q1 (pass-through claims under the subheader "If customers paid day-ahead prices"; §6.2, §7), Q2 (`supplier.carbon_intensity_gco2_per_kwh` in S1's scope; §0, §10). §12 items accepted as the lead ruled.

Optionals, applied: O1 ("in a typical week (medians across weeks)" replaces "the median week" everywhere, because each statistic's median comes from a different week; §0, §3.2, §3.3a, §6, §11, §12), O2 (`available_share` per slot and per typical-day half-hour on the card, in the hover and the expander; §3.3b, §6.1), O3 (J2 and the market follow-up are merged, so only J1b gates the evening kW and the card's profile and only S1 gates CO₂; replay-v1 §2 is lane R1c; the status line's base refreshed; §1, §3.1, §10), O4 (`ev_value_count` on every summary row and the "of N customers with closed sessions" wording, so the EVs a NaN metric leaves out are counted; §3.2, §6, §8, §9), O5 (the editing rule moved from the validator to the tests with the correct record effects, `SUPPLIER` records changing nothing here; the cheap-band check is a weekly identity; §8, §9), O6 (the private `individual._remember` import with a note, the simpler choice; §3.3), O7 (HH1 forms both absolute home costs and the validator ties their difference to `household_chunk`'s `home_cost_gbp`; `household_chunk` is not extended; §4, §8), O8 (one `ev_picker` helper, extracted from `one_ev.py` by HH3, builds the shared widget on both lenses; §6.1, §9, §10), O9 (group tiles and Key stats rows read the group ratio, and the cost tile says it is the group's overall £/kWh; §6, §6.2, §6.3).

Nothing remains open from the review of `4397276`.
