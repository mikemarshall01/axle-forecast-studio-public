# Supplier P&L and Partners contract v1

Status: draft for lead acceptance, 29 September 2026 (plan task W3-contract-supplier-pnl). Revised after the lead's review of `01b1081` (blocking B1–B7, the question on `M`, the optionals) and for decision 0004 item 65 / plan §13 (horizons and partner-facing measures). Base `76c8037` (`main` at `e60a9ed` merged). Design only: no model code exists for anything below except where §1 says a market-lane commit has landed.

Authority: decision 0004 item 62(a) (Supplier P&L view, Partners view, Key stats rows, growth-lever calculator, Compare records, the supplier demo) and item 65 (forecast horizons, hedge-block view, de-rated firm capacity, "your fleet" framing, CO₂ shifted, partner-facing measures, Limits); plan `docs/plans/2026-09-29-analyst-trading-polish-plan.md` §12(1) and §13; decision 0003 (no aggregate without a defined, approved component set that reconciles without double counting; missing terms are unavailable, not zero); decision 0005 (the illustrative simulated trading P&L and its buckets); `docs/contracts/trading-events-v1.md` (the ledger, the double-count rule, the Supplier frames, the firm-MW addendum §10 as merged to `main`); `docs/contracts/results-v2.md` (its §1 rules apply to every frame here; §4.10 lists the trading frames as implemented by the market lane).

**Reference convention.** `T§n` is a section of `docs/contracts/trading-events-v1.md`; `R§n` is a section of `docs/contracts/results-v2.md`; a bare `§n` is a section of this document.

Every price, volume, tonne and £ figure below is synthetic or illustrative. The one new net figure is the "illustrative simulated supplier P&L" of §3; it is never "Axle cash", and it is never added to the trading ledger's net (§3.1).

## 0. Terms, sign conventions and horizons

| Term | Meaning |
| --- | --- |
| Supplier | The retail supplier whose customers' EVs are in the product: the party that buys the fleet's home-charging energy at wholesale, hedges it day-ahead and settles the error half-hourly. In this model it is also the party operating the product, so it receives third-party grid-event cash (§3.2) and the ledger's supplier compensation is an in-house transfer (§3.1) |
| Partner | A charger or device maker (`units.manufacturer_id`, T§10.1d) whose enrolled devices earn flexibility cash |
| Treated EVs `T`, `N_T` | EVs not in the hold-out control group (T§9.5, `units.control_group`); the enrolled devices. Control EVs are not in the product and are left out of every per-customer and per-device figure, as in T§9.3c |
| `U_t` | Fleet home import on the unmanaged (`normal`) path, kWh per slot: `deviation_world_slot.unmanaged_kwh` |
| `M_t` | Fleet home import on the **dispatched** path, kWh per slot: `deviation_world_slot.metered_kwh`. Today the smart path (`selected`); once the price-responsive intraday dispatch of T§11 (plan §12(2), F3) lands, the path that includes that dispatch. `x_t` below stays the forecast at `τ_DA(n)`, so intraday re-optimisation shows up in the hedge error, never in the hedge |
| `P_t` | The day-ahead price after all shocks and the user curve, £/MWh: `deviation_world_slot.day_ahead_gbp_per_mwh` (equal to `forecast_prices.wholesale_forecast_gbp_per_mwh`, the price `cost_effect` values the tariff leg at) |
| `SIP_t` | The imbalance price, £/MWh: `deviation_world_slot.imbalance_gbp_per_mwh` |
| `B⁰_t` | The unadjusted BL01-lite baseline the day-ahead position uses (T§4.1, in whichever history mode `trading.baseline_trap` selects). It is the unmanaged fleet's hedge profile and it cancels out of every component (§3.2) |
| `x_t` | The aggregator's expected metered import at `τ_DA(n)` (T§4.4), kWh per slot: `deviation_world_slot.expected_metered_kwh` |
| `x^U_t` | The expected unmanaged import from the same expected sessions with every session unmanaged (`ρ_i = 1`), kWh per slot: `deviation_world_slot.expected_unmanaged_kwh` (§2) |
| `F_t = x^U_t − x_t` | The forecast reduction: the turn-down the aggregator expected at `τ_DA(n)`, unmasked and before the commitment share |
| `R_t = U_t − M_t` | The realised (true) reduction, `deviation_world_slot.true_reduction_kwh` |
| `H^U_t = B⁰_t`, `H^S_t = B⁰_t − F_t` | The profiled hedges the unmanaged and the smart supplier buy day-ahead (lead choice, like-for-like: the same baseline profile, the smart one lowered by the aggregator's forecast reduction). Both are known at `τ_DA(n)` |
| `n`, `slots(n)` | Session night 0–6 and its slot count (46, 48 or 50) |
| Sign of money | The supplier's view: positive = gain to the supplier, negative = cost. Costs are stored negative so every component adds to the net |
| Month and week | Weekly → monthly is `× 52 ÷ 12`; monthly → weekly is `× 12 ÷ 52`. Every "per month" figure is scaled from the simulated week and labelled "extrapolated from one simulated week" |
| Season | `winter` or `summer` from the London month of the run's first priced slot (the warm-up start, seven days before the study) and the `summer_months` record: the one rule the prices generator uses (`sampling.generate_market_price_paths`, plan D7, decision 0004 item 53). `supplier.build_frames` derives it the same way and stores it as a `season` column on every summary frame here, so the lens can say "winter run" without re-deriving it |
| Horizon | Item 65: each figure says which horizon it answers. Vocabulary: `intraday` (17:00 on the day), `day_ahead` (13:00 D−1), `week_ahead` (the distribution across simulated weeks: what is known before a week starts) and `scenario` (month, season and year figures scaled from weeks: not forecasts). Every summary frame here carries a `horizon` column; §12 maps the T§9 and T§10 frames the lenses also show |

`kWh ÷ 1000 × £/MWh` is the only money formula for energy. World first, quantiles last: every `mean`, `p10`, `p50`, `p90` below is `numpy.quantile(..., method="linear")` over per-world values.

## 1. Modules, lanes and what this contract consumes

Plain functions over NumPy arrays and pandas DataFrames, as T§1. No view computes a price, a cost, an allocation or a quantile.

| Module | Owns | Lane |
| --- | --- | --- |
| `model/supplier.py` (new) | `supplier_pnl_world`, `supplier_pnl_summary` (§3.3–3.4); `hedge_block_world`, `hedge_block_summary` (§3.7); `carbon_shift_world`, `carbon_shift_summary` (§3.8); `partner_world`, `partner_summary` (§4.2); `household_value_exceedance` (§4.3); `revenue_by_market_summary` (§4.5); their builder `build_frames(...)`; `HORIZON_BY_FRAME` (§12) | S1 |
| `model/growth.py` (new) | `annual_gbp`, `funnel`, `tornado`, `months_to_fund_discount`, `cumulative_payout_fan` (§5) | S3 |
| `assumptions.py` | The `SUPPLIER` and `CARBON` records of §8 | S1 |
| `summaries.py` | One hook line after the T§9.3 Supplier frames; the R§3.10c departure-SoC metrics (§4.6); `slim_run` and `compare_runs` records (§7) | S1 (hook, §4.6), S3 (Compare) |
| `tests/fixtures/supplier_contract.py` (new) | The frame validators of §9, called from `validate_result_v2` (one hook line, added by the lead at integration) | S1 |
| UI | Supplier ▸ 3 Supplier P&L (renamed from "3 Shape and value"), Supplier ▸ 5 Partners (new), Key stats section G rows, the "Supplier" dialog tab, the "your fleet" framing, the Limits entries (§11, §15) | S2 |
| UI | Compare rows and the growth expander (§5, §7) | S3 |
| `docs/explainers/interview-notes.md` | The supplier demo (§13) | S4 |

**Inputs consumed.** Everything below is action-only; every frame here is `None` on a no-action result, and `None` on an action result until the market lane's frames exist.

| Input | Owner | Status | Used for |
| --- | --- | --- | --- |
| `deviation_world_slot` (`unmanaged_kwh`, `metered_kwh`, `true_reduction_kwh`, `expected_metered_kwh`, `day_ahead_gbp_per_mwh`, `imbalance_gbp_per_mwh`, `night_index`) | Market lane F1 (T§5.1, R§4.10) | Landed on `claude/w3-market` at `e958228` (`expected_metered_kwh` after `metered_kw`); not on `main` | Energy cost, shape split, hedge error |
| **`deviation_world_slot.expected_unmanaged_kwh`** (new column, §2) | Market lane F1 | Landed on `claude/market-followup` (replaces the earlier `unmanaged_hedge_kwh`) | `F_t` |
| `trading_week_world` (`full`: the nine buckets, `net_gbp`) and the seven-bucket gross per (world, night) of `trading_ledger_world` | Market lane (T§4.7, T§5.3, T§5.4) | Landed at `e958228` | Grid-event cash; partner gross; revenue by market |
| `cost_effect.illustrative_selected_minus_normal_energy_cost_gbp` | Landed (R§4.8) | Landed | The energy component (exact reuse) |
| Per-EV arrays from the T§9.3c pass: `reduction_kwh` (world, EV, night), `allocation_weights` output `a` (world, EV, night), `value` (world, EV) | Market lane (`summaries.household_frames`, `allocation_weights` at `e958228`) | Landed on `claude/market-followup`: `household_frames` returns `per_ev` (§4.1); `reduction_kwh` is `household_worlds["reduction_kwh"]` and `a` is `allocation_weights` at the call site | Partner and exceedance frames |
| `units.manufacturer_id`, `units.home_charger_limit_kw`, `units.physical_capacity_kwh`, `units.control_group` | Firm-MW J1a (T§10.1d); R§3.2; market lane | `manufacturer_id` not landed; the rest landed | Manufacturer groups; £ per kW; cycles |
| Per-EV kernel arrays in the summaries chunk pass: `home_battery_added_kwh`, `public_battery_added_kwh` (`physics.UNIT_INTERVAL_QUANTITIES`), `connected`, and `plan_status` (T§10.1e) | Physics (landed); firm-MW J1b for `plan_status` | Landed except `plan_status` | Cycles; dispatch success rate |
| Day-ahead price frame column `system_net_demand_gw` (world, warm-up + study slot) | Prices lane (T§1.4) | Landed | Carbon intensity (§3.8) |
| `product_sheet` (T§10.5a), `charge_completion_summary`, `firmness_by_manufacturer` (T§10.5e, T§10.5g) | Firm-MW J5 | Not landed | Read by the lenses only; no frame here depends on them |
| `trading.customer_revenue_share` (0.5), `trading.baseline_trap` (0/1) in `assumptions.TRADING_OVERLAY` | Market lane (T§7, R§4.10) | Landed at `e958228` | Customer payments; the trap switch name |

## 2. The column the market lane adds to `deviation_world_slot` (T§5.1 changed)

At `e958228` the frame's column order is `… metered_kwh, baseline_kw, unmanaged_kw, metered_kw, expected_metered_kwh, deviation_kwh, …`. This contract adds one column directly after `expected_metered_kwh`:

| Column | Dtype | Meaning |
| --- | --- | --- |
| `expected_unmanaged_kwh` | float64 | `x^U_t = Σ_i w_i u_it`: the same `market.expected_smart_kwh` call that gives `x_t` for the day-ahead decision of night `n` (same `sessions`, same `first_slot`), with `non_response_probability` set to 1 for every EV, so every session takes its `unmanaged_schedule_kwh`. Unmasked, before the commitment share and before the newsvendor correction. 0 outside the night windows like `x_t` |

Why the same call: `x` and `x^U` then share the same expected sessions (typical clocks, warm-up need, weights, control EVs at `ρ = 1`), so `F = x^U − x` is the aggregator's expected turn-down and nothing else. The method bias of the expected sessions (typical clocks put each cohort's arrivals into one slot, T§4.3) and the control EVs are in both and cancel; a hedge built from `U`'s warm-up profile against `x` would not cancel them. The cancellation covers only what is common to `x^U` and `x`: the sessions and their timing. `F`'s own error, the forecast turn-down landing in different slots or amounts from the realised reduction `R`, remains, and is exactly what component 2 measures.

Validator (market lane, then S1): `expected_unmanaged_kwh` equals a recomputation on sampled worlds with `ρ = 1`; **`Σ_{t ∈ n} F_t = 0` within 1e-9 kWh per (world, night)**, because `unmanaged_schedule_kwh` and `plan_cheapest_slots` deliver the same clipped need for every session (T§4.4); `expected_metered_kwh` equals `baseline_known_kwh − trade_kwh ÷ c` on the day-ahead row of `position_updates` wherever `position_kwh > 0` and the mask is open (fixed-share mode, `c > 0`); the T§5.10 leakage test perturbs the realised sessions after `τ_DA` and both columns are unchanged.

## 3. Supplier P&L

### 3.1 What it is, and why the ledger is shown beside it and never added

The supplier P&L answers Axle's supplier customer: what does smart charging of these customers do to my procurement cost, my hedge error under half-hourly settlement, my third-party income and my customer payments, per customer per month, with its spread across simulated weeks. It is a new aggregate, so decision 0003 applies: the component set, their owners and signs are fixed here (§3.2), the cash entries reconcile without double counting (this section and §3.5), and the inclusion rule is visible in code, tests and the lens. The lead records its acceptance as a decision (proposed 0006, "illustrative simulated supplier P&L", the same form as 0005) before the net is shown; until then the lens shows the components table and no net (§16 Q1).

**The supplier P&L assumes a supplier that hedges the aggregator's full forecast; adding the ledger would value the same shifted energy twice.** The smart supplier's hedge is `B⁰ − F`: it buys the unmanaged profile and leaves out the turn-down the aggregator forecast. The ledger's day-ahead position is its aggregator-side counterpart (scaled by `c`, masked, turn-down only, and including the expected baseline effect), and the ledger's imbalance on the forecast error is the same error this P&L settles at `SIP − P`. So the energy and hedge-error components already hold the day-ahead value of the sold turn-down and its imbalance. The ledger net (`trading_kpis`, `full`, `net_gbp_per_week`) is shown beside the supplier net as a reference and never added; the lens caption says so. Of the nine ledger buckets, only the grid-event payment is third-party cash (component 3). The ledger's `supplier_compensation_gbp` is a transfer between the aggregator and the supplier; with the supplier operating the product it is in-house and nets to zero (default £0 in any case, decision 0005), so it is neither a component nor a deduction here.

**Tariff assumption, stated on the lens.** The supplier P&L assumes a **flat (non-time-of-use) retail tariff**: the customer pays the same p/kWh whenever the EV charges, so the supplier keeps the whole procurement saving and pays the customer only through the reward component (4). The customer-value frames of T§9.3c take the opposite, **pass-through** reading: the customer's illustrative saving as if the day-ahead price reached the customer (`cost_effect`, decision 0004 item 4). The two readings value the same shifted energy from two sides of one tariff and are never added; each lens caption names its reading. The retail tariff itself is not modelled (the supplier's retail margin on any volume change is absent; §15).

### 3.2 Per-world definitions

Per world `w`, over the 336 study slots, £ per week, for hedge variant `v ∈ {profiled, flat}`:

```
energy_cost_unmanaged   = Σ_t U_t P_t ÷ 1000
energy_cost_smart       = Σ_t M_t P_t ÷ 1000
energy_saving           = energy_cost_unmanaged − energy_cost_smart                       (component 1; = −cost_effect.illustrative_selected_minus_normal_energy_cost_gbp)

P̄                       = mean_t P_t                                                     (the T§9.3b baseload price)
shape_saving            = shape_cost_normal − shape_cost_selected                         (T§9.3b per-world shape costs; "of which")
volume_value            = P̄ × (Σ_t U_t − Σ_t M_t) ÷ 1000                                 ("of which")
energy_saving           = shape_saving + volume_value                                     (exact identity, §3.5)

hedge error of a supplier with hedge H and metered m:   Σ_t (m_t − H_t)(SIP_t − P_t) ÷ 1000
unmanaged supplier:  m = U, H = B⁰                      smart supplier:  m = M, H = B⁰ − F
hedge_error_saving (profiled) = Σ_t [(U_t − B⁰_t) − (M_t − B⁰_t + F_t)] (SIP_t − P_t) ÷ 1000
                              = Σ_t (R_t − F_t)(SIP_t − P_t) ÷ 1000                      (component 2; B⁰ cancels)
flat hedge: H̄ = the night mean of the profiled hedge;  since Σ_{t∈n} F_t = 0 (§2) the two flat hedges are equal, so
hedge_error_saving (flat)     = Σ_t R_t (SIP_t − P_t) ÷ 1000                             (component 2 for the flat rows)
worth_of_profiled_hedge       = profiled − flat = −Σ_t F_t (SIP_t − P_t) ÷ 1000            (reported beside, not a component)

grid_event_payment      = trading_week_world.grid_event_payment_gbp (strategy full)       (component 3; identical across strategies, T§5.9)

gross_gain              = energy_saving + hedge_error_saving + grid_event_payment
customer_payment        = −share × max(0, gross_gain)             mode 0, revenue_share (share = trading.customer_revenue_share)
                        = −reward × N_T × 12 ÷ 52                   mode 1, flat (reward = supplier.customer_reward_gbp_per_ev_per_month; NaN when unset)   (component 4, ≤ 0 or NaN)
net_gain_before_fee     = energy_saving + hedge_error_saving + grid_event_payment + customer_payment

platform_fee            = −fee × N_T × 12 ÷ 52   (fee = supplier.platform_fee_gbp_per_ev_per_month; NaN when unset)   (component 5, ≤ 0 or NaN)
net_gain                = net_gain_before_fee + platform_fee                              (NaN when the fee is unset: unavailable, not zero, decision 0003)

per customer per month  = value × 52 ÷ 12 ÷ N_T     for net_gain_before_fee, net_gain, customer_payment and platform_fee
```

Why each choice:

- **Energy at day-ahead, hedge-independent.** A supplier who buys hedge `H` at day-ahead and settles `m − H` at SIP pays `Σ H P + Σ (m − H) SIP = Σ m P + Σ (m − H)(SIP − P)`. So the procurement cost splits exactly into the energy at day-ahead (shape-dependent, hedge-independent) and the hedge error (hedge-dependent, zero when `SIP = P`). The energy component is the `cost_effect` home-import column with its sign flipped, reused and not recomputed, so the Value and risk lens and this lens cannot disagree.
- **Shape and volume "of which".** T§9.3b's shape cost is the load-weighted premium over baseload times the volume, so the energy saving is the shape saving plus the baseload value of any volume change (early departures move energy to public top-ups; unrecovered energy is not bought at home). With equal weekly volumes the shape saving is the whole energy saving. Both are reported so the reader sees that the saving is shape, not less energy. The volume term is valued at baseload wholesale only; the retail side of a volume change is not modelled (§15).
- **Like-for-like hedges, and why `B⁰` cancels.** Both suppliers hedge the same baseline profile; the smart one lowers it by the aggregator's forecast reduction. The paired saving is then `Σ (R − F)(SIP − P)`: what the fleet delivered against what was hedged for, priced at the imbalance premium. Over-delivery (`R > F`) is a surplus sold at SIP, a gain when the system is short; under-delivery is bought back at SIP. Because `B⁰` cancels, the component does not depend on the history mode (`trading.baseline_trap`) and needs no `B⁰` column. The unpaired per-supplier hedge-error costs would need `B⁰`, which the frame does not store; they are not columns here (§16 Q3 if wanted).
- **Hedge error priced at `SIP − P`, not `ID − P`.** The supplier's hedge is bought day-ahead and the error is settled at the imbalance price; supplier intraday re-hedging is not modelled (§15). The aggregator's intraday trading is in the ledger, beside.
- **The flat variant.** A supplier who hedges each night's energy as a flat block carries the whole realised reduction into imbalance. `profiled − flat` is the worth of hedging the profile rather than a flat block under half-hourly settlement (item 62a's "against a profiled and a flat hedge"). `profiled` is the headline; `flat` is shown beside.
- **Customer payment on the supplier's own gross.** In mode 0 the supplier passes the accepted share (0.5, decision 0005) of its own positive weekly gain to customers; the base is the supplier's gain, not the aggregator's trading gross, so every component of this P&L is a supplier cash flow and the payment cannot exceed what funds it. Weekly `max(0, ·)` (the ledger uses nightly `max(0, ·)`, T§8 Q7): a supplier settles a reward monthly, and a weekly floor is the closer reading; the lens says "on the week's positive gain". §16 Q2 offers the ledger bucket as the alternative. Mode 1 is item 62a's "customer rewards" as a fixed £/EV/month; its amount is unset (NaN) until Mike sets it, so in mode 1 with no amount the payment, the nets and the per-customer figures are unavailable, not zero.
- **Platform fee unavailable unless set.** Decision 0003: a missing commercial term is unavailable, not zero. `net_gain` is NaN and the lens shows "Platform fee: not set" and the before-fee net as the headline, labelled so.

### 3.3 `supplier_pnl_world` (exact)

One row per `(world_id, hedge_variant)`, world-major, `profiled` then `flat`. Money in £ per simulated week unless the name says per customer per month.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `world_id` | int64 | Key |
| `hedge_variant` | object | `profiled` or `flat` |
| `energy_cost_unmanaged_gbp`, `energy_cost_smart_gbp` | float64 | Σ U P ÷ 1000, Σ M P ÷ 1000 |
| `energy_saving_gbp` | float64 | Component 1 |
| `shape_saving_gbp`, `volume_value_gbp` | float64 | Of which (they sum to the component) |
| `forecast_reduction_mwh`, `realised_reduction_mwh` | float64 | Σ F ÷ 1000 (0 by the §2 identity, kept as the audit it is), Σ R ÷ 1000, per week |
| `hedge_error_saving_gbp` | float64 | Component 2, per variant |
| `worth_of_profiled_hedge_gbp` | float64 | `profiled − flat`, the same value in both rows; beside, not a component |
| `grid_event_payment_gbp` | float64 | Component 3 (≥ 0) |
| `gross_gain_gbp` | float64 | Components 1–3 |
| `customer_payment_gbp` | float64 | Component 4 (≤ 0 or NaN) |
| `net_gain_before_fee_gbp` | float64 | Components 1–4 |
| `platform_fee_gbp` | float64 | Component 5 (≤ 0), NaN when unset |
| `net_gain_gbp` | float64 | Components 1–5, NaN when the fee or a mode-1 reward is unset |
| `treated_ev_count` | int64 | `N_T`, the same in every row |
| `net_gain_before_fee_per_customer_per_month_gbp`, `net_gain_per_customer_per_month_gbp`, `customer_payment_per_customer_per_month_gbp`, `platform_fee_per_customer_per_month_gbp` | float64 | × 52 ÷ 12 ÷ `N_T` |
| `evidence_kind` | object | `"illustrative_synthetic"` |

`energy_*`, `shape_saving_gbp`, `volume_value_gbp`, the two reduction volumes, `worth_of_profiled_hedge_gbp` and `grid_event_payment_gbp` are identical in a world's two rows; the validator checks it.

### 3.4 `supplier_pnl_summary` (exact)

One row per `(hedge_variant, metric)`, variants in order, metrics in the column order of §3.3 from `energy_cost_unmanaged_gbp` to `platform_fee_per_customer_per_month_gbp` (`treated_ev_count` excluded). Columns: `hedge_variant`, `metric`, `unit` (`GBP per week`, `MWh per week` or `GBP per customer per month`), `horizon` (`week_ahead` for weekly rows, `scenario` for per-month rows), `season`, `world_count` (worlds with a non-NaN value: 0 for the fee rows when unset), `mean`, `p10`, `p50`, `p90`, `evidence_kind`. The means of components 1–4 add up to the mean before-fee net, and the means of all five to the mean net where the fee is set; quantiles do not, and the waterfall uses means and says so (as the ledger waterfall, T§5.5).

### 3.5 Identities (checked by the validator and tests)

Per world and variant, within `1e-9 × max(1, |value|)` unless stated:

1. `energy_saving_gbp = −cost_effect.illustrative_selected_minus_normal_energy_cost_gbp` (exact reuse, bit for bit) and `= Σ_t (U_t − M_t) P_t ÷ 1000` recomputed from `deviation_world_slot`.
2. `energy_saving_gbp = shape_saving_gbp + volume_value_gbp`; and the quantiles of `−shape_saving_gbp` across worlds equal `shape_premium_summary` (`difference`, `shape_cost_gbp_per_week`), so the shape saving equals minus the T§9.3b difference row and, when `Σ U = Σ M`, equals minus the `cost_effect` home-import column.
3. `SIP_t = P_t` in every slot ⇒ `hedge_error_saving_gbp = 0` in both variants and `worth_of_profiled_hedge_gbp = 0`.
4. `hedge_error_saving_gbp` (profiled) `= Σ (R − F)(SIP − P) ÷ 1000` and (flat) `= Σ R (SIP − P) ÷ 1000` recomputed; `forecast_reduction_mwh = 0` within 1e-9 (the §2 identity); `worth_of_profiled_hedge_gbp` = profiled − flat.
5. `gross_gain_gbp` = components 1–3; `net_gain_before_fee_gbp` = components 1–4, NaN exactly when (mode 1) the reward is unset; `net_gain_gbp` = components 1–5, NaN exactly when the fee is unset or `net_gain_before_fee_gbp` is NaN; `customer_payment_gbp ≤ 0` where defined; in mode 0 `customer_payment_gbp = −share × max(0, gross_gain_gbp)`; in mode 1 it is the same in every row.
6. Per-customer columns = the weekly column × 52 ÷ 12 ÷ `treated_ev_count`; `treated_ev_count = N −` control count; `platform_fee_per_customer_per_month_gbp = −fee` exactly where set.
7. `supplier_pnl_summary` equals the world-first recomputation from `supplier_pnl_world`; `world_count` is the non-NaN count; `season` and `horizon` as §0.

### 3.6 Worked example (hand-checkable)

One world, one night of two slots, `N_T = 2`, share 0.5, mode 0, fee unset, no events.

| slot | 0 | 1 |
| --- | --- | --- |
| `P` (£/MWh) | 100 | 50 |
| `SIP` (£/MWh) | 120 | 40 |
| `U` (kWh) | 10 | 0 |
| `M` (kWh) | 2 | 8 |
| `x^U` | 10 | 0 |
| `x` | 4 | 6 |

Energy: unmanaged `10 × 100 ÷ 1000 = £1.00`; smart `(2 × 100 + 8 × 50) ÷ 1000 = £0.60`; saving `£0.40`. `P̄ = 75`, `Σ U = Σ M = 10` so `volume_value = 0`; load-weighted prices 100 (unmanaged, premium +25, shape cost `25 × 10 ÷ 1000 = £0.25`) and 60 (smart, premium −15, shape cost `−£0.15`); shape saving `0.25 − (−0.15) = £0.40 = energy_saving`.

Hedge error: `F = x^U − x = [6, −6]` (sums to 0), `R = U − M = [8, −8]`, `R − F = [2, −2]`, `SIP − P = [+20, −10]`. Profiled saving `(2 × 20 + (−2) × (−10)) ÷ 1000 = +£0.06`: the fleet over-delivered 2 kWh when the system was short (sold at a £20 premium) and under-delivered 2 kWh when it was long (bought back £10 cheaper). Flat saving `Σ R (SIP − P) ÷ 1000 = (160 + 80) ÷ 1000 = +£0.24`; worth of the profiled hedge `0.06 − 0.24 = −£0.18` (in this week the flat hedge happened to do better, because the forecast reduction landed on the slots where the imbalance premium ran the other way). With `x^U = [9, 1]` and `x = [3, 7]` instead (the expected sessions mis-timed by the same amount on both sides) `F` is unchanged and so is every figure: the bias common to `x^U` and `x` cancels, and what component 2 measures is `F`'s own timing error against `R` (here 2 kWh in each slot). With `SIP = P` both savings are 0 (identity 3).

Profiled: gross `0.40 + 0.06 + 0 = £0.46`; customer payment `−0.23`; net before fee `£0.23`; fee NaN so net NaN; before-fee net per customer per month `0.23 × 52 ÷ 12 ÷ 2 = £0.498`. With a fee of £1/EV/month set: `platform_fee = −1 × 2 × 12 ÷ 52 = −£0.4615` per week, net `−£0.2315`, per customer per month `−£0.502` (= `0.498 − 1.000`). Flat: gross `£0.64`, customer `−0.32`, net before fee `£0.32`, per customer `£0.693`.

### 3.7 Hedge-block view (item 65 (1)): `hedge_block_world` and `hedge_block_summary` (exact)

The standard blocks a supplier buys: `baseload` (every study slot), `peak` (slots whose London wall-clock start is 07:00–18:30 on a Monday–Friday **calendar date**, so 07:00–19:00 by the clock; a session night's morning half is on the next date and is classed by that date) and `off_peak` (every other slot). Block hours are **derived from `study_slots`**, never hard-coded: `block_hours = 0.5 × ` the number of study slots whose `interval_start_london` weekday and time put them in the block. Under R§2 (`study_slot_count` always 336) `baseload` is 168 h. `peak` is 60 h in a week without a clock change whatever the start day; in a clock-change week the 336-slot study ends at 11:00 (autumn) or 13:00 (spring) London, so when that end falls on a weekday the last day's peak holds 4 or 6 h instead of 5 and the peak is 59 or 61 h. `off_peak` is the remainder. The frame stores the hours so the check is explicit and the validator checks the derivation, not fixed numbers.

Per world, block and path `p ∈ {normal, selected}`: `volume_mwh = Σ_{t ∈ block} m^p_t ÷ 1000` and `mean_mw = volume_mwh ÷ block_hours`; `difference` rows are selected − normal inside the world. The headline is `peak` `mean_mw` on the `difference` row: **the change in peak-block requirement in MW**, negative when smart charging moves energy out of the peak.

`hedge_block_world`: one row per `(world_id, block, path_id)`, blocks in the order above, paths `normal`, `selected`, `difference`. Columns: `world_id`, `block`, `path_id`, `block_hours` (float64, the same for every world), `volume_mwh`, `mean_mw`, `evidence_kind`. `hedge_block_summary`: one row per `(block, path_id, metric)`, metrics `volume_mwh` (`MWh per week`), `mean_mw` (`MW`), `volume_mwh_per_month` (`MWh per month`, `× 52 ÷ 12`); columns `block`, `path_id`, `metric`, `unit`, `horizon` (`week_ahead`; `scenario` for the monthly rows), `season`, `block_hours`, `world_count`, `mean`, `p10`, `p50`, `p90`, `evidence_kind`. Identities: per world `baseload = peak + off_peak` for volumes; `difference = selected − normal`; `baseload` `normal` `volume_mwh × 1000 = Σ U`; `mean_mw × block_hours = volume_mwh`.

### 3.8 CO₂ shifted (item 65 (4)): `carbon_shift_world` and `carbon_shift_summary` (exact)

An illustrative synthetic carbon intensity tied to net demand, simple by design: when net demand (demand minus wind and solar) is low the margin is clean; when it is high, gas fills it.

```
I_t = clip( I_ref + s × (D_t − D_ref),  I_floor,  I_cap )        gCO₂ per kWh
D_t = system_net_demand_gw of the day-ahead frame (world, slot): demand net of wind and solar, with known shocks; surprise shocks are not in it (stated)
D_ref = supply_reference_net_demand_gw (20 GW, the supply curve's reference)
co2_unmanaged_kg = Σ_t U_t I_t ÷ 1000;   co2_smart_kg = Σ_t M_t I_t ÷ 1000;   co2_shifted_kg = co2_unmanaged_kg − co2_smart_kg   (positive = less CO₂ under smart)
```

The three `CARBON` records (§8) are illustrative placeholders (§16 Q5): `I_ref` 180 gCO₂/kWh at the 20 GW reference, `s` 12 gCO₂/kWh per GW, floor 20 and cap 450 (a gas-set margin). The intensity is a mix intensity, not a marginal one, and it is not observed data; the lens says "illustrative synthetic carbon intensity tied to the model's own net demand". Per world: the three kg figures and `co2_shifted_kg_per_ev_per_month = co2_shifted_kg × 52 ÷ 12 ÷ N_T`. `carbon_shift_world`: one row per `world_id` with those four columns and `evidence_kind`; `carbon_shift_summary`: one row per metric, columns `metric`, `unit` (`kg CO2 per week`, `kg CO2 per EV per month`), `horizon`, `season`, `world_count`, `mean`, `p10`, `p50`, `p90`, `evidence_kind`. Identity: `co2_shifted_kg = Σ_t R_t I_t ÷ 1000`; with `s = 0` it equals `I_ref × Σ R ÷ 1000`.

### 3.9 De-rated firm capacity (item 65 (2)): read from `product_sheet`, no new frame

The "winter-evening firm turn-down, P90–P95" is `product_sheet` (T§10.5a) at `window = evening`, `direction = turn_down`, `duration_hours = 1`: `window_mean_mw_p10` (90 % exceedance) and `window_mean_mw_p05` (95 %), day-ahead horizon, with `firm_share` beside as the de-rating against the median. It is labelled with the run's `season`: "winter-evening" only when the run is a winter run (start month outside `summer_months`), otherwise "summer-evening", never silently. The lens and Key stats read the stored columns; nothing is recomputed. The T§10 two-convention note applies ("P90 here is the table's P10 column").

## 4. Partners

### 4.1 Per-world definitions

Groups `G`: `fleet` (all treated EVs), each cohort with treated EVs (source order), each manufacturer `m1`…`m4` with treated EVs (once `manufacturer_id` exists). Per world `w` and group, from the T§9.3c per-EV arrays: `g_{i,n}` (`reduction_kwh`, world × EV × night: the EV's positive true reduction in the night's settled slots), `a_{i,n}` (`allocation_weights`), `value_i` (£ per household per month) and the ledger's seven-bucket gross per (world, `full`, night), `gross_n` (T§4.7: the buckets above the customer share).

**The named change (owner: the market lane, `summaries.household_frames` on `claude/w3-market`; S1 makes it after that branch merges if the lane has closed).** `household_frames` returns a third value, `per_ev: dict[str, np.ndarray]` with `value_gbp_per_month` (world, EV; the `value` it already computes) and `earning` (world, EV, bool: `reduction_kwh.sum(axis=2) > 0`), alongside the two frames; the call site passes `per_ev`, `weights`, `household_worlds["reduction_kwh"]`, the `full` ledger rows and `units` to `supplier.build_frames`. No frame changes.

```
gross_i                       = Σ_n a_{i,n} gross_n                                (£ per week; the EV's allocated gross flexibility cash)
earning_i                     = Σ_n g_{i,n} > 0                                    (the device turned down in at least one settled slot)
gross_flex_gbp_per_week_G     = Σ_{i ∈ G} gross_i
enrolled_G, earning_G         = |G|, #{i ∈ G : earning_i}
share_earning_G               = earning_G ÷ enrolled_G
gross per enrolled device     = gross_flex_gbp_per_week_G × 52 ÷ 12 ÷ enrolled_G   (£ per device per month)
gross per earning device      = gross_flex_gbp_per_week_G × 52 ÷ 12 ÷ earning_G    (NaN when earning_G = 0)
gbp_per_kw_charger_per_year_G = gross_flex_gbp_per_week_G × 52 ÷ Σ_{i ∈ G} home_charger_limit_kw_i
customer value per device     = mean_{i ∈ G} value_i                              (£ per household per month, T§9.3c, pass-through reading)
dispatch_success_rate_G       = #{sessions s of EVs in G : plan_status_{t,i} = 0 in every connected slot t of s} ÷ #{sessions s of EVs in G}     (selected path; session-weighted; NaN with no session)
cycles_G^p                    = mean_{i ∈ G} Σ_t (home_battery_added_kwh^p_{t,i} + public_battery_added_kwh^p_{t,i}) ÷ physical_capacity_kwh_i   (equivalent full cycles per EV per week, p ∈ {normal, selected})
```

Why: "£ per enrolled device per month" is the number a device maker can multiply by its installed base; it is **gross flexibility cash before the customer share and penalties** (item 62a), the ledger's gross, so it includes the baseline-effect bucket (which is not flexibility; the lens caption names it, §16 Q6). An earning device is defined physically (it delivered turn-down in a settled slot), not by the sign of its allocated cash, so a night with negative gross does not turn earners into non-earners. When a night has no earner, `allocation_weights` spreads that night's gross equally over treated EVs (T§9.3c), so non-earners can carry cash; `gross per enrolled = share_earning × gross per earning` therefore holds exactly whenever `earning_G > 0` and is not defined otherwise. £ per kW of charger per year is the maker's figure per unit of installed charger power (item 65). The dispatch success rate is session-weighted (lead decision on Q11): a session counts as followed when `plan_status = 0` (T§10.1e) in every one of its connected slots, so one ignored replan fails the whole session. The sessions are the selected-path home sessions ending in the study from `find_sessions` in the per-EV chunk pass, the same session table T§10.5e's charge completion reads, so the two rates share a denominator. Equivalent full cycles show that smart charging moves energy in time, not amount: the paired difference should be small, and where it is not (public top-ups after early departures) it says so.

Charge-by-departure and firmness by manufacturer are read from the firm-MW frames (`charge_completion_summary` rows `selected`, groups `fleet`, cohorts, makers; `firmness_by_manufacturer` 1-h turn-down rows), not rebuilt here.

### 4.2 `partner_world` (exact) and `partner_summary` (exact)

`partner_world`: one row per `(world_id, group_type, group_id)`, world-major, groups `fleet`, cohorts, manufacturers in that order. Columns: `world_id`, `group_type` (`fleet`, `cohort`, `manufacturer`), `group_id` (`fleet`, the cohort id, `m1`…`m4`), `enrolled_ev_count` (int64), `earning_ev_count` (int64), `share_earning`, `gross_flex_gbp_per_week`, `gross_flex_gbp_per_enrolled_device_per_month`, `gross_flex_gbp_per_earning_device_per_month`, `gbp_per_kw_charger_per_year`, `customer_value_gbp_per_device_per_month`, `dispatch_success_rate` (NaN until `plan_status` exists), `equivalent_full_cycles_per_ev_per_week_normal`, `equivalent_full_cycles_per_ev_per_week_selected`, `equivalent_full_cycles_per_ev_per_week_difference`, `evidence_kind`.

`partner_summary`: one row per `(group_type, group_id, metric)`, metrics in the column order above from `share_earning` to `equivalent_full_cycles_per_ev_per_week_difference`, units `fraction`, `GBP per week`, `GBP per device per month`, `GBP per kW per year`, `cycles per EV per week`. Columns: `group_type`, `group_id`, `metric`, `unit`, `horizon` (`week_ahead` for per-week and cycle rows, `scenario` for per-month and per-year rows), `season`, `enrolled_ev_count`, `world_count`, `mean`, `p10`, `p50`, `p90`, `evidence_kind`. "£ per device per month by season" (item 65) is this frame's `season` column read across runs in Compare; no separate frame.

### 4.3 `household_value_exceedance` (exact)

The probability and cost of a guaranteed £X per month (item 62a). Thresholds `X = 0, 1, …, 40` £ per month (the task's grid; the lens default is £10). Per world and group:

```
share_at_or_above_G(X)              = #{i ∈ G : value_i ≥ X} ÷ enrolled_G
floor_top_up_G(X)                   = Σ_{i ∈ G} max(0, X − value_i) ÷ enrolled_G         (£ per enrolled device per month: what guaranteeing £X costs)
```

One row per `(group_type, group_id, metric, threshold_gbp_per_month)`, groups as §4.2, metrics `share_at_or_above` (`fraction`) then `floor_top_up_gbp_per_device_per_month` (`GBP per device per month`), thresholds ascending. Columns: `group_type`, `group_id`, `metric`, `threshold_gbp_per_month` (int64), `unit`, `horizon` (`scenario`), `season`, `enrolled_ev_count`, `world_count`, `mean`, `p10`, `p50`, `p90`, `evidence_kind`. About 2 × 41 × 11 ≈ 900 rows.

Per-world invariants: `share_at_or_above` is non-increasing in `X`; `floor_top_up` is non-decreasing in `X` and each £1 step raises it by between `1 − share_at_or_above(X)` and `1 − share_at_or_above(X + 1)`. **Exact:** `1 − share_at_or_above(0)` is the share of EVs with `value_i < 0`, which is `household_value_summary`'s `share_worse_off` (the two events are complements). What the validator checks (lead review B5): the step bounds and the `share_worse_off` identity per world on the fixture and on the `mean` column; the `p10`, `p50`, `p90` columns only monotone in `X` and, for shares, within 0–1 (a quantile of a bounded step need not obey the per-world step bound).

`value_i` is always the T§9.3c customer value under the pass-through reading (saving plus the allocated ledger customer share), **independent of `supplier.customer_reward_mode`** (lead decision on Q4). The supplier's customer payment per EV (§3.3, `customer_payment_per_customer_per_month_gbp`) is shown separately on Supplier P&L and is never mixed into `value_i` or this frame. Caption: "customer value under the pass-through reading (saving plus the trading product's customer share); the supplier's reward is on Supplier P&L and is not included here".

### 4.4 `revenue_by_segment` gains `segment_type = "manufacturer"` (T§9.3d changed)

After the `zone` segments: `segment_type = "manufacturer"`, `segment_id` `m1`…`m4` (makers with treated EVs), the same buckets, metrics and columns; the rows appear once the population carries `manufacturer_id`, as the zone rows appear once it carries `zone_id` (R§4.10). Per world the manufacturer segments sum to the `all` segment like the cohort and zone segments (validator). Owner: the market lane's `revenue_by_segment` (a segment loop entry), or S1 after merge.

### 4.5 `revenue_by_market_summary` (exact): the waterfall over modelled markets (item 65)

Strategy `full`, per world from `trading_week_world`. Markets in this order, each a sum of ledger buckets: `day_ahead` (`day_ahead_revenue_gbp`), `intraday` (`intraday_pnl_gbp + trading_cost_gbp`), `imbalance` (`imbalance_gbp`), `grid_events` (`grid_event_payment_gbp`); then `baseline_effect` (`baseline_effect_gbp`, not a market and shown apart, decision 0005), `deductions` (`supplier_compensation_gbp + customer_revenue_share_gbp + unmet_charge_penalty_gbp`) and `net` (`net_gbp`, the exact sum of the six rows above it). Then the **unmodelled** markets, listed and never shown as zero (decision 0005): `balancing_mechanism`, `frequency_response`, `capacity_market`, `dno_services`, with `modelled = False` and NaN statistics.

One row per `(market, metric)`, metrics `gbp_per_week` and `gbp_per_enrolled_device_per_month` (`÷ N_T × 52 ÷ 12`). Columns: `market`, `modelled` (bool), `metric`, `unit`, `horizon` (`week_ahead`, `scenario` for the per-device rows), `season`, `world_count` (0 on unmodelled rows), `mean`, `p10`, `p50`, `p90`, `evidence_kind`. Identity: the `mean` of the four markets plus `baseline_effect` plus `deductions` equals the `mean` of `net` within `1e-9 × max(1, |net|)`; quantiles do not add and the lens says so. The partner take and the customer share are shown beside it as the lever (§5) and the record (`trading.customer_revenue_share`).

### 4.6 SoC at departure on both paths: two `session_distribution_bands` metrics (R§3.10c changed; owner S1)

R§3.10c gains two metrics after `slack_hours`, same frame, keys and columns: `departure_soc_percent` (`percent`, 20 × 5 bins from 0, 100 % in the last; the battery stock at the session's last connected slot ÷ capacity × 100 on the **normal** path, a stock-flow output) and `departure_soc_percent_smart` (the same sessions on the **selected** path; plug-in and unplug are exogenous, so the session set is identical and only the stock differs). Both need the unplug, so they count only sessions that end in the study, like `dwell_hours`. Owner: lane S1, a bounded hunk in the R§3.10c builder (`summaries.py`), because it is the same per-EV session pass; the Sessions lens adds the two small multiples with the "early departures cut the smart-path SoC" caption.

### 4.7 Identities (validator and tests)

1. Per world, the cohort groups' `gross_flex_gbp_per_week` sum to the fleet's, and so do the manufacturer groups'; the fleet's equals the `full` strategy's seven-bucket gross summed over the week, because the weights sum to 1 per (world, night).
2. `enrolled_ev_count` per group equals the treated count in `units` (cohort and manufacturer counts); control EVs are absent.
3. Where `earning_ev_count > 0`: `gross_flex_gbp_per_enrolled_device_per_month = share_earning × gross_flex_gbp_per_earning_device_per_month` per world within `1e-9 × max(1, |value|)`; where it is 0 the per-earning figure is NaN and the per-enrolled figure is whatever the equal-share fallback allocated.
4. `0 ≤ share_earning ≤ 1`; `earning_ev_count ≤ enrolled_ev_count`; `0 ≤ dispatch_success_rate ≤ 1` or NaN; cycles ≥ 0 and `difference = selected − normal`.
5. `customer_value_gbp_per_device_per_month` for `fleet` equals `household_value_summary` (`fleet`, `mean_value`) per world.
6. `gbp_per_kw_charger_per_year × Σ home_charger_limit_kw ÷ 52 = gross_flex_gbp_per_week` per group.
7. The §4.3 rules as stated there; `household_value_exceedance` equals its recomputation from `value_i` on the fixture.
8. `revenue_by_market_summary` identity of §4.5; unmodelled rows NaN with `modelled = False`.
9. Summaries equal world-first recomputations from the world frames.

## 5. Growth calculator (`model/growth.py`)

Pure functions, no model state, no random draw. They are not physics, policy or settlement, so calling them from the Partners lens with the presenter's lever values is allowed; they live in the model package so they are tested and so the lens computes nothing itself.

```python
def funnel(*, eligible: float, invited_share: float, signed_up_share: float, active_share: float,
           share_earning: float) -> pd.DataFrame:
    """Stages eligible → invited → signed_up → active (the presenter's inputs, each a share of the stage
    before) → earning (active × share_earning, the model's figure). Columns stage, count, share_of_eligible."""

def annual_gbp(*, enrolled_devices: float, gbp_per_enrolled_device_per_month: float, take_rate: float,
               fee_gbp_per_device_per_month: float = 0.0) -> float:
    """enrolled_devices × (gbp_per_enrolled_device_per_month × take_rate + fee) × 12: £ per year to the
    partner, its take of the gross flexibility cash on its active devices plus a per-device fee.
    enrolled_devices is the funnel's active count. Extrapolated from one simulated week; illustrative."""

def tornado(*, enrolled_devices: float, gbp_per_enrolled_device_per_month: float, take_rate: float,
            fee_gbp_per_device_per_month: float = 0.0, swing: float = 0.3) -> pd.DataFrame:
    """Four levers at (1 − swing) and (1 + swing) × value with the others fixed: lever, label, value,
    low_value, high_value, annual_gbp_low, annual_gbp_high, clipped (take_rate clipped to [0, 1]).
    The fee row is present only when the fee is non-zero. Sorted by swing size, descending."""

def months_to_fund_discount(discount_gbp: float, *, gbp_per_enrolled_device_per_month_by_world: np.ndarray,
                            take_rate: float) -> tuple[float, float, float, int]:
    """Months of the partner's flexibility revenue (£ per enrolled device per month × take) needed to
    fund a sticker-price discount of discount_gbp per device. Per world first: months_w = D ÷
    (revenue_w × take_rate), NaN where that revenue is not positive ("not funded"); then (p10, p50, p90)
    with numpy.quantile over the worlds that are not NaN, and their count. The per-world revenue is the
    fleet row of partner_world (gross_flex_gbp_per_enrolled_device_per_month), the frame the result
    carries, so the lens passes a stored column and the division happens world by world."""

def cumulative_payout_fan(*, monthly_gbp_p10: float, monthly_gbp_p50: float, monthly_gbp_p90: float,
                          months: int = 12) -> pd.DataFrame:
    """Rows month 1…months with cumulative_p10, cumulative_p50, cumulative_p90 = month × the monthly
    quantile. It assumes the same simulated week repeats every month (months fully correlated), which is
    why a constant multiple gives the cumulative quantiles; independent months would give a narrower
    band, so this fan is the wider, conservative reading. A scenario, not a forecast."""
```

Band: the lens calls `annual_gbp` three times with `gbp_per_enrolled_device_per_month` set to the `p10`, `p50`, `p90` of `partner_summary` (`fleet`, `gross_flex_gbp_per_enrolled_device_per_month`) and the other levers fixed. Multiplying a per-world quantity by a positive constant and adding a constant preserves quantiles exactly, so those three values are the P10/P50/P90 of the annual figure across simulated weeks at the chosen levers. The point estimate and the tornado use the `p50`. The caption says "band from the simulated weeks at the run's own £ per enrolled device; the levers move the point estimate". The cumulative payout fan (12 months) takes the customer payment per customer per month from `supplier_pnl_summary` (`profiled`, `customer_payment_per_customer_per_month_gbp`, sign flipped to a payout) for the per-customer fan, and the same × the funnel's active count for the fleet fan; its caption says the months are taken as fully correlated (the same simulated week repeating), so the band is wider than independent months would give. `months_to_fund_discount` reads the per-world fleet column of `partner_world` (carried on the result and in the slim record, §7), so the months are computed world by world and quantiled after.

Levers and defaults: `share_earning` and `gbp_per_enrolled_device_per_month` default to the run's fleet `p50`; `eligible`, the three funnel shares, `take_rate`, the fee and the discount have no source in this project and no accepted value, so their defaults are §16 Q7. They are widget state on the lens, not `Assumption` records: they do not enter a run, so editing them must not mark the result stale.

Tests: `annual_gbp` is linear in `enrolled_devices`, `take_rate` and the fee; `funnel` stage counts multiply through and `earning = active × share_earning`; the tornado's multiplicative levers give symmetric swings about the base and the clipped row is marked; the three-call band equals the quantiles of a per-world recomputation on a fixture (tolerance `1e-9 × max(1, |value|)`); `months_to_fund_discount` equals a hand per-world computation on a five-world fixture with one non-positive world left out (`world_count` 4) and the quantiles taken over the rest; `cumulative_payout_fan` is `month × monthly` exactly with 12 rows by default.

## 6. Key stats section G rows (Overview ▸ Key stats)

Section G "Supplier" already holds the T§9.8 rows (cost curve at 18:00, shape premium, household value, share worse off). These rows follow them, in this order. Every supplier P&L row reads the **`profiled`** variant; every row shows its horizon in the Reference cell ("day-ahead", "week-ahead", "scenario: per month"); the view applies its usual rules (P10/P50/P90 as stored; `Δ vs unmanaged` only where the model supplies a paired difference; "Unavailable" for NaN, never 0):

| Metric | Unit | Frame, row | Reference | Verdict rule |
| --- | --- | --- | --- | --- |
| Supplier net gain per customer, before platform fee | £ per customer per month | `supplier_pnl_summary` (`profiled`, `net_gain_before_fee_per_customer_per_month_gbp`) | "scenario: per month; trading net £X/week (section F), shown beside, not added" | "P10 positive: a gain in 9 weeks out of 10" when `p10 > 0`; "P10 negative: a loss in some weeks" when `p10 < 0`; else "" |
| Energy saving at day-ahead | £ per week | `…`, `energy_saving_gbp` | "week-ahead; of which shape £S, volume £V (P50)" | spread rule |
| Hedge-error saving, profiled hedge | £ per week | `…`, `hedge_error_saving_gbp` | "week-ahead; flat hedge £F, profiled hedge worth £W (P50)" | "0 in every week" when `p10 = p90 = 0`, else the spread rule |
| Grid-event payments | £ per week | `…`, `grid_event_payment_gbp` | "week-ahead" | "no events" when `p10 = p90 = 0` |
| Customer payments | £ per week | `…`, `customer_payment_gbp` | mode and rate ("50 % of the week's positive gain" or "flat £R/EV/month") | "Unavailable: flat reward not set" when NaN |
| Platform fee | £ per week | `…`, `platform_fee_gbp` | "scenario: from £/EV/month" | "Unavailable: not set" when NaN |
| Peak-block requirement change | MW | `hedge_block_summary` (`peak`, `difference`, `mean_mw`) | "week-ahead; peak 07:00–19:00 Mon–Fri, H hours this week" | "moved out of the peak" when `p90 < 0`; "moved into the peak" when `p10 > 0`; else "" |
| De-rated firm turn-down, evening 1 h | MW | `product_sheet` (`evening`, `turn_down`, 1 h): `window_mean_mw_p50`, `_p10`, `_p90` in the P50, P10 and P90 cells as stored | "day-ahead; season S; firm = the P10 cell (90 % exceedance); P95 exceedance X MW (`window_mean_mw_p05`); de-rating `firm_share`" | "Unavailable until firm-MW J5" when absent |
| CO₂ shifted | kg CO₂ per EV per month | `carbon_shift_summary` (`co2_shifted_kg_per_ev_per_month`) | "scenario: illustrative synthetic intensity tied to net demand" | spread rule |
| Gross flexibility cash per enrolled device | £ per device per month | `partner_summary` (`fleet`, `gross_flex_gbp_per_enrolled_device_per_month`) | "scenario; includes the baseline-effect bucket" | spread rule |
| Share of devices earning | fraction | `partner_summary` (`fleet`, `share_earning`) | "week-ahead" | |
| Gross flexibility cash per earning device | £ per device per month | `partner_summary` (`fleet`, `gross_flex_gbp_per_earning_device_per_month`) | "scenario" | |
| £ per kW of charger | £ per kW per year | `partner_summary` (`fleet`, `gbp_per_kw_charger_per_year`) | "scenario" | |
| Dispatch success rate | fraction | `partner_summary` (`fleet`, `dispatch_success_rate`) | "week-ahead; share of sessions that followed their plan in every slot" | "Unavailable until firm-MW J1b" when NaN |
| Equivalent full cycles, smart | cycles per EV per week | `partner_summary` (`fleet`, `…_selected`); Δ from `…_difference` | unmanaged P50 | "energy moved in time, not amount" when the difference `p90 − p10` band lies within ±0.05 cycles, else "" |
| Households at or above £10/month | fraction | `household_value_exceedance` (`fleet`, `share_at_or_above`, 10) | "scenario; pass-through reading" | |
| Floor cost of guaranteeing £10/month | £ per device per month | `…`, `floor_top_up_gbp_per_device_per_month`, 10 | "scenario" | |
| Charge completed by departure, smart | fraction | `charge_completion_summary` (`fleet`, `selected`, `all`, `completed_share_*`) | unmanaged row's P50 | "Unavailable until firm-MW J5" when absent |

The after-fee net is not a Key stats row (it is NaN until a fee is set; the lens shows it beside the before-fee net when set). Every £ row's unit cell reads "£ (illustrative)". The view computes nothing: every value is a stored column; the only view arithmetic is the sign and constant-scaling arithmetic the module already allows.

## 7. Compare records (`slim_run`, `compare_runs`)

`RunSummary` gains `supplier_pnl_world`, `hedge_block_world`, `carbon_shift_world` and `household_value_exceedance` (each `pd.DataFrame | None`; `None` on a no-action result or when the frames are absent); `slim_run` copies them. They are small (2 × worlds, 9 × worlds, worlds and about 900 rows).

`compare_runs`:

- `weekly_differences` and `kpis` gain, when both runs carry the frames, the paired B − A metrics `supplier_net_gain_before_fee_per_customer_per_month_gbp` (`GBP per customer per month`, `profiled` rows), `peak_block_mw_change` (`MW`, `hedge_block_world` `peak` `difference` `mean_mw`) and `co2_shifted_kg_per_ev_per_month` (`kg CO2 per EV per month`). The fee-inclusive net is not compared.
- New `RunComparison` field `supplier_side_by_side: pd.DataFrame | None` (exact): one row per `metric` for the fleet at £10, metrics `share_at_or_above` and `floor_top_up_gbp_per_device_per_month`; columns `metric`, `unit`, `threshold_gbp_per_month`, `season_a`, `season_b`, `p10_a`, `p50_a`, `p90_a`, `p10_b`, `p50_b`, `p90_b`. Each run's own quantiles side by side, not a paired difference (a share across EVs has no per-EV pairing across runs); the Compare view says so. `None` when either run lacks the frame. "£ per device per month by season" is read the same way from `partner_summary` (`fleet`), added as a second block of rows with `metric = gross_flex_gbp_per_enrolled_device_per_month` and `threshold_gbp_per_month` NaN.
- `validate_run_comparison_v2` checks the new metric rows and the side-by-side frame (columns, ordered quantiles, present exactly when both inputs carry the frames).

Compare view: three KPI tiles (supplier net per customer per month, peak-block MW change, CO₂ shifted per EV per month, each B − A P50 with P10–P90) and the side-by-side mini table, all under the existing illustrative caption.

## 8. Assumption records (`model/assumptions.py`; T§9.9 records unchanged)

Conventions as `e958228`: switches are whole numbers 0/1 with unit `"switch (1 on, 0 off)"`; an unset numeric record is `float("nan")` and reads "unset" in the dialog, never 0 (decision 0003; `b4f7b2b` makes two NaN records compare equal between runs).

| Name | Group | Default | Unit | Bounds | Evidence | Authority |
| --- | --- | --- | --- | --- | --- | --- |
| `supplier.customer_reward_mode` | `SUPPLIER` | 0 (revenue share) | switch (0 = revenue share of the supplier's positive weekly gain, 1 = flat £/EV/month) | 0–1 | illustrative | item 62a "customer rewards"; §16 Q2 |
| `trading.customer_revenue_share` (existing, `TRADING_OVERLAY`) | | 0.5 | fraction | 0–1 | illustrative | decision 0005; read in mode 0, not duplicated |
| `supplier.customer_reward_gbp_per_ev_per_month` | `SUPPLIER` | NaN (unset; lead answer to the earlier Q2) | GBP per EV per month | 0–50 when set | illustrative | item 62a; used in mode 1 only |
| `supplier.platform_fee_gbp_per_ev_per_month` | `SUPPLIER` | NaN (unset) | GBP per EV per month | 0–50 when set | illustrative | item 62a "optional platform fee"; decision 0003 |
| `carbon.intensity_at_reference_gco2_per_kwh` | `CARBON` | 180 (placeholder, §16 Q5) | gCO₂/kWh | 0–600 | illustrative | item 65 (4) |
| `carbon.intensity_slope_gco2_per_kwh_per_gw` | `CARBON` | 12 (placeholder) | gCO₂/kWh per GW of net demand | 0–50 | illustrative | item 65 (4) |
| `carbon.intensity_floor_gco2_per_kwh`, `carbon.intensity_cap_gco2_per_kwh` | `CARBON` | 20, 450 (placeholders) | gCO₂/kWh | 0–600, floor ≤ cap | illustrative | item 65 (4) |

Dialog: a new **"Supplier"** tab holds the `SUPPLIER` and `CARBON` records together with the T§9.9 supplier records (`supplier.early_departure_risk_charge_gbp_per_mwh`; the price-curve upload stays on Supplier ▸ 2). It is an addition to the R§8 tab list (`ASSUMPTION_GROUPS` gains `"Supplier"`; T§10.8 adds `"Firm MW"` separately), owned by lane S2 (§14); the results-v2 §8 comment and the groups test move with it. Reporting choices, not assumptions: the exceedance thresholds 0–40 in £1 steps and the £10 lens default (task); the hedge variants and blocks; the 52 ÷ 12 month; the 12-month fan; the ±30 % tornado swing and the growth levers (widget state, §5). Editing any record here changes no kernel frame, no price, no position and no ledger row (validator: identical `deviation_world_slot` and `trading_ledger_world` before and after).

**"Your fleet" framing (item 65 (3), owner S2).** The Fleet tab's caption reads: "Set your fleet: size and archetype mix. Every supplier and partner figure then answers for that fleet." Fleet size and the editable cohort mix (decision 0004 item 54) are the pre-sales estimator's inputs; no new record.

## 9. Validator rules (`validate_result_v2` additions, `tests/fixtures/supplier_contract.py`)

- Column sets, order and dtypes exactly as §3.3, §3.4, §3.7, §3.8, §4.2, §4.3, §4.5; keys unique; row order as stated; every frame `None` on a no-action result and `None` exactly when `deviation_world_slot` is `None`; `season` and `horizon` values from the §0 vocabulary.
- `supplier_pnl_world`: the identities of §3.5 (1–6) recomputed from `deviation_world_slot`, `cost_effect`, `trading_week_world`, `units` and the records; hedge-independent columns identical across a world's two rows; `customer_payment_gbp ≤ 0` where defined and NaN exactly when mode 1 has no reward; `platform_fee_gbp ≤ 0` or NaN, NaN exactly when the record is unset; `net_gain_gbp` NaN exactly then.
- `supplier_pnl_summary`: world-first recomputation; `world_count` equals the non-NaN count; the means of components 1–4 add to the before-fee net mean, and of all five to the net mean where the fee is set, within `1e-9 × max(1, |net|)`.
- `hedge_block_world`, `hedge_block_summary`: the §3.7 identities; `block_hours` equal 0.5 × the slot count of each block recomputed from `study_slots` (`baseload` 168 under R§2; `peak` 60, or 59/61 when a clock-change week's end falls in a weekday peak window; `off_peak` the remainder); peak slots are exactly the 07:00–18:30 Monday–Friday calendar slots of `study_slots`.
- `carbon_shift_world`, `carbon_shift_summary`: `I_t` recomputed from the day-ahead frame and the records with the clip; the §3.8 identities.
- `partner_world`, `partner_summary`: §4.7 identities 1–6 and 9; groups in order; `enrolled_ev_count` fixed across worlds; manufacturer rows present exactly when `units.manufacturer_id` exists; `dispatch_success_rate` NaN exactly when `plan_status` is absent.
- `household_value_exceedance`: §4.3 as stated (per-world step bounds and the `share_worse_off` identity on the fixture and on `mean`; quantile columns monotone in `X`, shares within 0–1); thresholds exactly 0…40.
- `revenue_by_segment`: manufacturer segments sum to `all` per world (§4.4). `revenue_by_market_summary`: the §4.5 identity; unmodelled rows NaN with `modelled = False`.
- `session_distribution_bands`: the two §4.6 metrics present with shares within 0–1 summing to 1 over bins; the normal-path metric equals a recomputation from `plug_in_events` when every world is sampled (the existing rule); the smart metric has the same session count per group as the normal one.
- `deviation_world_slot`: `expected_unmanaged_kwh` present after `expected_metered_kwh`, finite, `Σ_{t∈n} F_t = 0` within 1e-9 per (world, night), and consistent with `position_updates` on sampled worlds as §2 states.
- Editing a `SUPPLIER` or `CARBON` record or `trading.customer_revenue_share` changes no kernel frame, price, position or ledger row.

## 10. Sanity-check tests (checklist; tolerances `1e-9 × max(1, |value|)` unless stated, bit-for-bit where "reuse" is said)

- [ ] The worked example of §3.6 as a unit test of `supplier.build_frames` on a two-slot fixture, both variants, share 0.5, fee unset and fee set (£1/EV/month with `N_T = 2` gives `platform_fee = −1 × 2 × 12 ÷ 52 = −0.4615` per week and `−1.000` per customer per month); the method-bias case (`x^U = [9, 1]`, `x = [3, 7]`) gives identical figures; a control EV present in both `x^U` and `x` gives identical figures.
- [ ] `SIP = P` everywhere ⇒ every hedge-error column 0 in both variants (identity 3), with non-trivial forecast errors.
- [ ] A perfect forecast (`F = R`) ⇒ profiled saving 0; the flat variant is then `Σ R (SIP − P) ÷ 1000` and non-zero whenever `SIP ≠ P` and `R ≠ 0`.
- [ ] `energy_saving_gbp` equals minus the `cost_effect` home-import column bit for bit (reuse, not recomputation) and the shape + volume split holds; with equal weekly volumes on a fixture, `shape_saving_gbp` equals it exactly.
- [ ] Mode 0 with a negative gross week pays 0; mode 1 pays the same in every world and NaN when the reward is unset; the mode switch changes nothing but `customer_payment_gbp`, the nets and the per-customer columns.
- [ ] Per-customer columns use the treated count: a control group of 10 % changes `treated_ev_count` and the per-customer figures and nothing else in the frame.
- [ ] Hedge blocks: `block_hours` equal the fixture's own `study_slots` count × 0.5 for every block; a study starting on a Monday and one starting on a Saturday both give 60 peak hours at different slots; the autumn and spring clock-change fixtures (`make_result(clock_change=...)`) starting on a Monday give 59 and 61 peak hours with `baseload` 168; a fixture with all import in one peak slot gives `peak volume = baseload volume` and `off_peak = 0`; `difference` rows are exact per-world subtractions.
- [ ] Carbon: `s = 0` gives `co2_shifted = I_ref × Σ R ÷ 1000`; the clip binds at the floor on a fixture with very low net demand; editing a `CARBON` record changes no other frame.
- [ ] Partners: on a fixture with two makers, per-world manufacturer gross sums to the fleet's; a night with no earner spreads its gross equally (T§9.3c) and the per-enrolled figure is non-zero while the per-earning figure is NaN; where earners exist `share_earning × £ per earning = £ per enrolled`; a maker with no treated EVs is absent; `gbp_per_kw_charger_per_year` on a fixture of 7 kW chargers; `dispatch_success_rate` from a hand `plan_status` array (two of five plan-slots ignored gives 0.6); cycles from a fixture with 10 kWh added into a 50 kWh battery (0.2 cycles) on both paths.
- [ ] Exceedance: hand fixture with values `[−5, 4, 12, 30]` at `X = 10`: share 0.5, floor `(15 + 6) ÷ 4 = 5.25`; monotonicity across 0…40; the per-world step bounds; `1 − share_at_or_above(0) = share_worse_off` exactly (0.25 here).
- [ ] Revenue by market: the six modelled rows' means add to `net`; the four unmodelled rows are NaN and flagged.
- [ ] Departure SoC metrics: a session that ends at target on the normal path and early on the selected path lands in the top bin on `departure_soc_percent` and a lower bin on `departure_soc_percent_smart`; session counts equal.
- [ ] Growth: the §5 tests.
- [ ] Compare: two runs on matched futures give paired differences equal to the per-world subtraction for all three new metrics; the side-by-side frame is `None` when one run is no-action; `validate_run_comparison_v2` passes.
- [ ] Market lane: `expected_unmanaged_kwh` equals the `ρ = 1` recomputation; `Σ_{t∈n} F_t = 0`; perturbing realised sessions after `τ_DA(n)` leaves both expected columns unchanged (§2).
- [ ] Editing any §8 record leaves `deviation_world_slot`, `trading_ledger_world` and every kernel frame identical.

## 11. UI consumption map

Views read these frames and filter, plot or tabulate them; they compute no cost, allocation, quantile or money. Every chart uses Plotly through `ui/style.py` (decision 0004 item 29). Every £ is labelled illustrative; the supplier net is labelled "illustrative simulated supplier P&L"; nothing says "Axle cash". Every figure carries its horizon label (§12).

| View | Reads |
| --- | --- |
| Supplier ▸ 3 Supplier P&L (renamed from "3 Shape and value"; the T§9.8 shape-premium, household-value and revenue-by-segment blocks stay below the new blocks) | Hedge-variant control (`profiled` default, `flat` beside). Waterfall of the five component `mean`s to the net from `supplier_pnl_summary` (caption: means add up, quantiles do not; "Platform fee: not set" bar absent when NaN). Strip or box of `net_gain_before_fee_per_customer_per_month_gbp` per world from `supplier_pnl_world` with P10/P50/P90 tiles. "Of which" tiles for `shape_saving_gbp` and `volume_value_gbp`. Hedge-error tile pair (profiled vs flat) with `worth_of_profiled_hedge_gbp`. Reference tile "Trading net (aggregator), shown beside, never added" from `trading_kpis` (`full`, `net_gbp_per_week`). Hedge-block panel from `hedge_block_summary`: three blocks × unmanaged vs smart (P50 with P10–P90), headline tile "peak-block requirement change, MW" from the `difference` row, block hours in the caption, monthly volumes labelled "scaled from weeks". De-rated firm tile from `product_sheet` (§3.9) with the season label. CO₂ tiles from `carbon_shift_summary`. Captions: the §3.1 double-count rule; the flat-tariff assumption ("the supplier keeps the procurement saving; the customer's saving on the Value and risk lens is the pass-through reading; never add them"); "extrapolated from one simulated week"; the customer-payment mode in words |
| Supplier ▸ 5 Partners (new; after "4 Firm MW" so the firm-MW numbering of T§10.10 is unchanged; §16 Q8) | Group selector over `partner_summary` groups (`fleet` default; cohorts; makers). Tiles: `gross_flex_gbp_per_enrolled_device_per_month`, `share_earning`, `gross_flex_gbp_per_earning_device_per_month`, `gbp_per_kw_charger_per_year`, `customer_value_gbp_per_device_per_month`, `dispatch_success_rate`, cycles (smart, with the paired difference) — P50 with P10–P90; partner take (lever) and customer share (record) beside. `revenue_by_market_summary` as a waterfall of means with the unmodelled markets listed under it as "not modelled". Exceedance chart from `household_value_exceedance` for the chosen group: `share_at_or_above` P50 line with P10–P90 band against `threshold_gbp_per_month`, and `floor_top_up_gbp_per_device_per_month` on a second panel; a threshold slider (0–40, default 10) that only moves a marker and the two tiles beside it. `charge_completion_summary` (`selected`, `all` and `early`, the chosen group; "Unavailable until firm-MW J5" when absent). `firmness_by_manufacturer` 1-h turn-down rows as a small table when present. Sessions-style small multiples of the two §4.6 metrics for the chosen group. Growth expander (§5): the funnel (four inputs, earning from the model), the four-lever tornado (small), the annual figure with its P10–P90 band, "months to fund a £D discount" (input £D, P10–P90 months), the cumulative payout fan (12 months, per customer and for the funnel's active count, captioned as fully correlated months), the §5 caption |
| Overview ▸ Key stats, section G | The §6 rows |
| Drivers ▸ Sessions | The two §4.6 metrics as small multiples (both paths), owner S2 with the Sessions view's existing pattern |
| Compare | The §7 tiles and side-by-side table |
| Edit assumptions | The "Supplier" tab (§8); the "your fleet" caption on the Fleet tab |
| How it works ▸ Limits | The §15 entries |

IA after this: Overview · Drivers · Smart charging · Trading · Supplier (1 Availability and cost curve · 2 Positions · 3 Supplier P&L · 4 Firm MW · 5 Partners) · Compare · How it works.

## 12. Illustrative labelling and horizon labels (decisions 0003 and 0005; item 65)

- Every money column is `*_gbp`, illustrative, synthetic prices; the frames carry `evidence_kind = "illustrative_synthetic"`.
- The supplier net is shown only with the label "illustrative simulated supplier P&L", only once the lead's decision record (§3.1, §16 Q1) exists, and always with its components table and the "trading net shown beside, never added" caption.
- The tariff reading is on every lens that shows a saving: "flat retail tariff: the supplier keeps the procurement saving" (Supplier P&L) or "pass-through reading: the customer's saving as if the day-ahead price reached the customer" (Value and risk, household value, Partners customer value).
- Missing commercial terms (the platform fee; the flat reward until set) render "Unavailable", never 0, and the fee-inclusive net renders "Unavailable" with them.
- "Per month", "per year" and cumulative figures carry "extrapolated from one simulated week"; the monthly hedge-block volumes carry "scaled from weeks".
- Partner figures say "gross flexibility cash before the customer share and penalties; includes the baseline-effect bucket" and "allocated by each EV's share of settled flexibility" (T§9.3c). The waterfall lists unmodelled markets by name, never as zero.
- CO₂ figures say "illustrative synthetic carbon intensity tied to the model's own net demand; not observed grid data".
- The growth calculator says "illustrative; the levers are yours, not the model's; the band is the simulated weeks' spread at the run's own £ per enrolled device"; the fan says "a scenario that repeats the simulated week, not a forecast".
- No figure here is a bid, a settlement, a tariff or Axle cash; the retail tariff is not modelled (§15).

**Horizon labels.** `supplier.HORIZON_BY_FRAME` (a plain dict the lenses import) maps every frame the Supplier lenses show to its label, and every summary frame here also stores the value in its `horizon` column:

| Figure | Horizon | Label shown |
| --- | --- | --- |
| Supplier P&L per-week rows, hedge blocks, CO₂ per week, partner per-week rows, dispatch success, cycles | `week_ahead` | "week-ahead: the spread across simulated weeks" |
| Per-month, per-year, cumulative, exceedance and growth figures | `scenario` | "scenario: scaled from one simulated week, not a forecast" |
| `flex_cost_curve`, `supplier_positions` day-ahead columns, `product_sheet` and `availability_bands` `day_ahead`, the de-rated firm figure | `day_ahead` | "day-ahead: decided at 13:00 the day before" |
| `availability_bands` `intraday`, the 17:00 firm figure | `intraday` | "intraday: decided at 17:00 on the day" |

Season is a scenario label ("winter run" or "summer run" from the start month), never a forecast.

## 13. Demo order (for `docs/explainers/interview-notes.md`, lane S4)

Inserted as step 6a of "A four-minute walk through the app", about 60 seconds, after the Smart charging lenses and before Compare, with one sensitivity row under "Demo sensitivities":

1. **Supplier ▸ 3 Supplier P&L.** Read the waterfall left to right: energy saving at day-ahead (point at the shape/volume split: "it is shape, not less energy"), hedge-error saving under half-hourly settlement (profiled hedge; flip to flat to show the worth of hedging the profile), grid-event payments (0 unless a preset is on), customer payments (50 % of the week's positive gain), platform fee "not set". The per-customer-per-month strip: "£X a month per customer, P10–P90 across 100 weeks; a flat tariff, so the supplier keeps the saving". The hedge-block panel: "the peak-block requirement falls by Y MW". The de-rated firm tile: "Z MW firm in the winter evening at P90, day-ahead". Point at the trading-net tile and say why it is beside and not added (§3.1).
2. **Supplier ▸ 5 Partners.** Fleet tiles: £ per enrolled device per month, share earning, £ per earning device, £ per kW of charger. Switch the group to a maker: "a maker with a 2 % outage night loses its slice; its dispatch success rate shows it". Waterfall by market, then the unmodelled list. Exceedance at £10: "N % of households clear £10 a month; guaranteeing it costs £Y per device". Cycles: "energy moved in time, not amount". Open the growth expander: the funnel, the annual figure with its band, months to fund a £D discount, the payout fan.
3. **Compare.** Edit assumptions ▸ Supplier ▸ reward mode flat, £R/EV/month, Run; Compare: supplier net per customer per month and peak-block MW change B − A, and the £10 exceedance side by side (not paired). Or change the start date across the summer boundary for the season comparison.

Sensitivity row (filled with the default run's numbers when S1–S3 are integrated): "Customer reward share 0.3 (default 0.5)" against supplier net per customer per month, households ≥ £10, floor cost at £10. Every £ in the row is illustrative.

## 14. Lane split, dependencies and review levels

| Lane | Outcome | Owns | Depends on | Risk |
| --- | --- | --- | --- | --- |
| **S1** model summaries | §3, §4 frames (including §4.6); `SUPPLIER` and `CARBON` records; validators of §9; the summaries hook; the §4.4 manufacturer segment and the `household_frames` per-EV return are requested from the market lane, or made by S1 after that branch merges | `src/axle_studio/model/supplier.py`, the `SUPPLIER`/`CARBON` hunk in `assumptions.py`, one hook hunk and the §4.6 hunk in `summaries.py` (at most 40 lines together), `tests/model/test_supplier.py`, `tests/fixtures/supplier_contract.py`, the synthetic-fixture extension in `tests/fixtures/result_fixture.py` | `claude/w3-market` merged with the §2 column and the §4.1 per-EV return; J1a `manufacturer_id` and J1b `plan_status` (S1 builds fleet and cohort groups first and leaves the maker rows and `dispatch_success_rate` to appear when those columns land, saying so) | High (new result schema, finance aggregate): lead-accepted interface (this contract) and one strong independent review; a second specialist review only if the reviewer finds a distinct double-count risk |
| **S2** UI lenses, Key stats, dialog tab, Limits | §11 Supplier P&L and Partners lenses; §6 rows; the "Supplier" dialog tab and the R§8 groups change; the "your fleet" caption; the §15 Limits entries; the Sessions small multiples | `src/axle_studio/ui/views/supplier_pnl.py` (new, absorbing the T§9.8 "Shape and value" blocks), `src/axle_studio/ui/views/partners.py` (new), the lens entries in `ui/registry.py` and `ui/pages.py`, the section-G hunk in `ui/views/key_stats.py`, the tab hunk in `ui/views/parameters.py` and `ASSUMPTION_GROUPS` (with `tests/model/test_assumptions.py`'s groups test), the Limits hunk in `ui/views/methods.py`, the Sessions hunk in `ui/views/sessions.py`, `tests/ui/test_supplier_pnl_view.py`, `tests/ui/test_partners_view.py`, the touched view tests | S1's fixture extension; the Supplier page shell (I5) | Normal (renders with and without every optional frame; no view arithmetic on quantiles) |
| **S3** growth and Compare | §5 module; §7 records and view rows | `src/axle_studio/model/growth.py`, `tests/model/test_growth.py`, the `RunSummary`/`slim_run`/`compare_runs` hunks in `summaries.py`, the `validate_run_comparison_v2` hunk, the Compare view hunk in `ui/views/compare.py`, the `tests/ui/test_compare_view.py` hunk; the growth expander in `partners.py` after S2 merges | S1 frames for Compare; nothing for `growth.py` (can start first) | Normal |
| **S4** interview notes | §13 | `docs/explainers/interview-notes.md` | S1–S3 integrated, numbers from the default run | Routine |

Order: S3's `growth.py` and S1 on a fixture start now; S1 integrates after `claude/w3-market` merges with the §2 column and the §4.1 return; S2 after S1's fixture extension; S3's Compare after S1; S4 last. The lead adds the one `validate_result_v2` hook line and the `forecast.py` field declarations (`supplier_pnl_world`, `supplier_pnl_summary`, `hedge_block_world`, `hedge_block_summary`, `carbon_shift_world`, `carbon_shift_summary`, `partner_world`, `partner_summary`, `household_value_exceedance`, `revenue_by_market_summary`, all `None` by default) at S1's integration.

## 15. Limits entries (owner S2, `ui/views/methods.py`)

Added to `NOT_MODELLED` (item, one-line reason), in this order after "Charging taper":

- Retail tariff: no retail tariff is modelled; the supplier P&L assumes a flat tariff and the customer-value frames a pass-through reading, and neither models the supplier's retail margin on a volume change.
- Supplier intraday re-hedging: the supplier's hedge is bought day-ahead and its error settles at the imbalance price; only the aggregator trades intraday (the ledger).
- Sign-up time, opt-outs, complaints, satisfaction and retention curves: the funnel takes them as the presenter's inputs; nothing in the run moves them (item 65).
- Command latency: a plan is followed or ignored at the half-hour; no dispatch delay is modelled.
- V2G wear, batteries and heat pumps: no export path, no stationary battery or heat pump in the fleet (also T§10).
- Attach-rate uplift: a partner's sales uplift from the product is not modelled; the calculator counts flexibility cash and a fee only.
- Balancing Mechanism: the fleet is not bid into the BM; its revenue is listed on the waterfall as unmodelled, never zero (decision 0005). The existing "Frequency response", "Capacity market" and "Local flexibility" entries are the other three names the waterfall lists; S2 also reconciles the existing "DFS" entry's wording with the grid-event presets the events lane landed (a turn-down request is now modelled as an illustrative event, T§2), which is outside this contract's figures.

Added to `KNOWN_LIMITATIONS`:

- Carbon: the CO₂ figures use an illustrative synthetic intensity tied to the model's own net demand, not observed grid carbon data, and a mix intensity rather than a marginal one.
- Hedge blocks: peak is 07:00–19:00 Monday–Friday by calendar date, 60 h in an ordinary week and 59 or 61 h when a clock-change week ends on a weekday (the 336-slot study then ends at 11:00 or 13:00 London); the frame derives every block's hours from `study_slots` and stores them.
- Horizons: month, season and year figures are scaled from one simulated week (scenarios), not forecasts; only the day-ahead and intraday figures are decided at an instant.
- Shared timing variation (people plugging in earlier or later together) is not modelled (decision 0004 item 64), so partner and supplier bands may be narrow on timing.

## 16. Open questions for the lead (values not invented)

1. **Decision record for the net.** Decision 0003 requires an approved component set before an aggregate is shown. Proposed: decision 0006 "illustrative simulated supplier P&L" with the five components, signs, the like-for-like hedge, the flat-tariff reading and the never-added rule of §3.1. Until it exists, show components only.
2. **Customer payment base.** §3.2 pays `share × max(0, weekly supplier gross gain)`; the alternative is the ledger's own `customer_revenue_share_gbp` bucket (what the simulation pays customers under the trading product, funded by trading cash this P&L excludes). Resolved by the lead: the flat reward defaults to NaN, labelled unset.
3. **Per-supplier hedge-error rows.** The paired component needs no `B⁰`; unpaired "hedge error, unmanaged supplier" and "hedge error, smart supplier" rows would need the market lane to expose `baseline_unadjusted_kwh` (`B⁰`) on `deviation_world_slot`. Not requested unless the lens wants them.
4. **Exceedance reward leg: resolved by the lead.** The exceedance always uses `value_i` (the pass-through reading), independent of the supplier reward mode; the supplier's per-EV customer payment is shown separately on Supplier P&L and never mixed in (§4.3).
5. **Carbon intensity placeholders.** `I_ref` 180 gCO₂/kWh at 20 GW, slope 12 per GW, floor 20, cap 450: a simple linear tie to net demand, illustrative; confirm or replace, and whether the intensity should read the realised (intraday) net demand once the prices lane exposes surprise shocks in GW per slot.
6. **Partner gross and the baseline effect.** The per-device gross is the ledger's seven-bucket gross, so it includes the baseline-effect bucket. Whether to report "of which baseline effect" per device or exclude it from the partner figure.
7. **Growth and funnel defaults.** `eligible`, the three funnel shares, `take_rate`, the fee and the discount have no source here; the lens needs initial widget values. Proposed placeholders for the lead to confirm or replace: eligible 100,000, invited 50 %, signed up 30 %, active 80 %, take rate 20 %, fee unset, discount £100; all labelled illustrative.
8. **Lens order.** "5 Partners" after "4 Firm MW" (keeps T§10.10's numbering; Partners reads J5 frames so it reads naturally after them) against "4 Partners · 5 Firm MW" (story order: supplier → partners → firm MW).
9. **Weekly versus nightly `max(0, ·)`** for the revenue-share payment (§3.2 chooses weekly; the ledger uses nightly, T§8 Q7).
10. **`expected_metered_kwh` under T§10.1e.** Once maker-driven non-response lands, `ρ_i = 1 − r_m (1 − π_m)` for a treated EV; confirm the market lane's exposed `x_t` uses the same `ρ_i` the position uses at that time, and that `x^U` keeps `ρ = 1` for every EV, so `F` stays the aggregator's forecast reduction.
11. **Dispatch success rate weighting: resolved by the lead.** Session-weighted: a session is followed when `plan_status = 0` in every connected slot, over `find_sessions`'s selected-path sessions in the per-EV pass (§4.1).
12. **Payout fan length: resolved by the lead.** 12 months (§5, §8).
