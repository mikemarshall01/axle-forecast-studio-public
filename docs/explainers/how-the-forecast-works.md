# How the forecast works

File references are to `src/axle_studio/model/` unless stated. `docs/CODE_GUIDE.md` gives the reading order through the code.

## 1. What the model answers, and what it does not

I built the model to answer, for 1,000 EVs over one week: when are they plugged in at home, how much energy do they draw, what state of charge (SoC, how full the battery is) do they plug in at, and what happens if Axle times each EV's home charging to the cheapest forecast half-hours before it is due to leave? It draws many possible weeks (Monte Carlo sampling) and reports the spread across them.

It is **illustrative, not calibrated**. The six archetypes come from Axle's workbook; most behaviour values, the weather, public charging and prices are assumptions I state. Nothing is fitted to observed fleet data or back-tested; only the price shape was checked once against Elexon market data (section 10).

It is **not settlement and not Axle cash**. Every £ figure is an illustrative valuation of an energy difference, never a bid, settlement amount, payment or Axle revenue (decisions 0003 and 0004 item 4).

Every value below is a record in `assumptions.py`, with its unit, evidence kind and source.

## 2. The population

The fleet is 1,000 persistent EVs from six workbook cohorts:

| Cohort | Share | Annual miles | Battery | Home plug-in |
| --- | --- | --- | --- | --- |
| Average (UK) | 40% | 9,435 | 60 kWh | every evening, 18:00 to 07:00 |
| Intelligent Octopus average | 30% | 28,105 | 72.5 kWh | every evening, 18:00 to 07:00 |
| Infrequent charging | 10% | 9,435 | 60 kWh | 20% of evenings, 18:00 to 07:00 |
| Infrequent driving | 10% | 5,700 | 60 kWh | every evening, 18:00 to 07:00 |
| Scheduled charging | 9% | 9,435 | 60 kWh | every evening, 22:00 to 09:00 |
| Always plugged-in | 1% | 9,435 | 60 kWh | whenever at home |

All use 3.5 miles per kWh and an 80% preferred target. Home charging has **one power for every EV**, 7 kW by default (decision 0004 item 34 dropped a separate vehicle AC limit).

`sampling.py:build_population` splits the EVs across cohorts by largest remainder, shuffles cohort labels onto EV ids, and draws one **personal mileage multiplier** per EV from a mean-one lognormal, so some drivers always drive more than others. Every simulated week (a **world**, in the code) shares these traits, so weeks differ in what happened, not in who the drivers are.

## 3. The clock and the settled start

Time is a fixed grid of UTC half-hours: 7 study days (336 slots) from London noon on the start date (decision 0004 item 52: noon rather than midnight, so a noon-to-noon block holds a whole overnight session instead of cutting the first night in half), after a **warm-up** (seven days, 336 slots, by default; editable 3–14 days, decision 0004 item 52).

```
 warm-up (336 UTC slots by default)   study (336 UTC slots)
|-----------------------------------|--------------------------------------------|
                                     ^ London noon on the start date
```

`clock.py:utc_half_hour_boundaries` builds the boundaries (673 at the seven-day default). Slot `t` is `[b[t], b[t+1])`, and the stock at `b[t]` is its opening stock.

**Settled start** (decision 0004 item 36). Every EV opens at 80% SoC, its preferred target, and drives and charges normally for the warm-up before anything is reported. The action does not apply through most of the warm-up (both paths share the same charging there), but it starts on the last warm-up night, so the first study night opens with batteries a smart charger has already shaped, not a freshly settled pool (item 52). I settle every EV at the target first because of an artefact review found: with a 55% opening SoC and one warm-up day, cutting home charging power from 7.0 to 3.6 kW *raised* study-week home import by about 190 kWh, because the warm-up refill spilled into the week. A test now checks that lower home charging power never raises study-week home import. Item 52 later widened the warm-up from the three days item 36 first settled on to seven, so the trading baseline (`docs/MODEL_CONTRACT.md`) has a full week of history, five working and two non-working nights, and made it editable.

A clock change inside the week is allowed (item 1). The grid stays 336 UTC slots; "day d" is the London date at its real offset, so a clock-change day has 46 or 50 half-hours. Each EV's expected departure comes from its cohort's typical departure clock time by the same London-to-UTC conversion (`clock.py:london_wall_time_to_utc`, used in `action.py:smart_charging_inputs`).

## 4. What is random, and how

The runner creates **one** generator, `np.random.default_rng(seed)`, and passes it to each sampler in a fixed order (`forecast.py:simulate_forecast`): population, then the 100 evaluation weeks (each draws trips, home connections, then weather), then (action model only) the price paths. There is no separate planning world (decision 0004 item 38 removed it): the smart charger plans against each simulated week's own day-ahead price path (item 48). There is no per-channel seeding; the same seed and settings give the same run.

Independent draws use named `scipy.stats` distributions with `random_state=rng`, and price noise uses statsmodels (decision 0004 item 14):

| Channel | What is drawn | Where |
| --- | --- | --- |
| Drives today | Yes or no per EV-day, weekday or weekend probability per cohort; Bernoulli by inverse transform (`U < p`) | `sampling.py:sample_daily_trip_inputs` |
| Daily mileage | Mean-one lognormal multiplier with the cohort CV; the mean is expected miles divided by drive probability, so the weekly average holds | same |
| Departure clock | Student-t shift around the cohort hour (fat-tailed, 4 degrees of freedom by default, decision 0004 item 51), rounded to 30 minutes, truncated at ±3 hours | same |
| Home connection | Plug-in yes or no per EV-day, then one rounded, truncated Student-t shift for that evening's plug-in and the next morning's unplug | `sampling.py:sample_connection_opportunities` |
| Weather | One temperature per simulated week and day, normal around a synthetic base (7–12 °C, SD 2 °C), shared by all EVs | `sampling.py:sample_daily_temperature` |
| Prices | Each simulated week's own net-demand path (a demand shape, heating, wind, solar and shocks) through an illustrative merit-order supply curve for the day-ahead price, then intraday and imbalance layers on top (item 53) | `sampling.py:generate_market_price_paths` |

Every channel takes a fixed number of draws whatever its parameters, so editing one assumption changes only its own channel and Compare keeps **matched futures** (item 6).

A driving day is one round trip: outbound, a dwell away, and a return, with equal legs at 30 mph. There is no public-charger availability draw (item 32).

The departure and plug-in clocks moved from a normal shift to a fatter-tailed Student-t shift (decision 0004 item 51): most days sit near the cohort's usual time, but the odd very early or late day is more common than a normal distribution would give, truncated rather than clipped so the tail mass spreads inside the bound (`clock_clip_minutes`, `clock_t_df` in Edit assumptions).

The day-ahead price is not a price-only shape: net demand (GW) is a 3-harmonic Fourier demand shape by London clock hour (one set per season × weekday/weekend), plus a heating term (a colder week is also a dearer one, sharing the same sampled temperature that drives driving efficiency), minus wind and solar, plus any shock (decision 0004 items 55, 56). That net demand runs through an illustrative convex merit-order supply curve (`sampling.py:supply_curve_gbp_per_mwh`), so the same shock moves price more when the system is already tight, then adds a persistent daily level and half-hour AR(1) noise (`e_t = φ e_(t−1) + u_t` from `ArmaProcess.generate_sample`). The intraday close adds hourly martingale updates to gate closure plus the price impact of surprise shocks; the imbalance (SIP) price adds a long/short NIV premium and a fat tail (decision 0004 item 53 sets what each price channel is for). Every coefficient is a record in `assumptions.py` (`SYSTEM`, `SHOCKS`, `PRICES`, `TRADING`); none of it is market data.

## 5. The energy balance, half-hour by half-hour

`physics.py:_simulate_path` loops over slots and handles every simulated week and EV at once as NumPy arrays. For each EV in each slot:

```
S_next = S_open + η_home · I_home + η_pub · I_public − e_travel
```

In plain words: the battery's energy at the end of this half-hour equals its energy at the start, plus home grid energy times home charging efficiency, plus public top-up energy times public charging efficiency, minus the energy used driving in this half-hour.

`S` is battery-side kWh; `I_home` and `I_public` are grid-side kWh; `η_home = 0.92`, `η_pub = 0.9`. The run fails if the conservation residual exceeds 1e-12 kWh or stock leaves `[0, capacity]`, so SoC is always 0–100%. Arrival SoC is never sampled; it is the stock when the EV gets home.

```
slot [b[t], b[t+1])
  1. travel           outbound leg, then return leg, spread over each leg's minutes
  2. public top-up    only if a leg would take stock below the threshold
  3. occupancy        driving, parked away, or at home
  4. home connection  plug-in session overlap with home time, cut at the next departure
  5. home charge      only if connected at home for the whole slot
  -> check conservation and 0 <= S <= capacity
```

**Travel.** A leg needing `E` kWh over `D` minutes draws `E × overlap / D` in each slot it overlaps. Energy is miles divided by the day's effective efficiency (`physics.py:effective_driving_efficiency`):

```
efficiency = base × (1 − min(0.5, 0.01 × |T − 20 °C|))
```

In plain words: driving efficiency is the car's base efficiency cut by 1% for every degree the day's temperature sits away from a mild 20 °C reference, in either direction, capped at a 50% cut so a car never reaches zero efficiency; so a 10 °C day costs 10% more energy per mile (illustrative values).

**Threshold top-ups; EVs never strand** (decision 0004 item 32, `physics.py:_top_up_battery_kwh`). If a leg would take the battery below the top-up threshold (10% SoC by default, lowered from 20% by item 37), the EV tops up at a public charger to the top-up target (80% SoC) at the moment it reaches the threshold, then carries on. If one top-up is not enough, it tops up again, as many times as the leg needs:

```
n = ceil(remaining / (target − threshold))
battery added = (target − first level) + (n − 1) × (target − threshold)
I_public = battery added / η_pub
```

In plain words: the number of top-ups needed is the remaining energy required divided by how much one top-up adds (target minus threshold), rounded up. The battery energy added is the first top-up's fill up to the target, plus a full top-up's worth for each further stop. The grid energy drawn at the public charger is that battery energy divided by the public charging efficiency.

Every top-up fills to the target, as a driver stopping to charge would. I record the import in the slot where it happens and do not model the time spent charging. There is no separate destination charging: this is the only way energy enters the battery away from home. Every trip is served, so unserved travel is zero by construction and stays in the results as a check.

**Home charging needs a full slot.** Connected time is the session's overlap with time at home, cut at the next departure; a return at 18:10 charges from 18:30. Then

```
I_home = min( P × 0.5 h,  (target × C − S) / η_home )    if full slot and S < target × C
P = home charging power (item 34)
```

In plain words: home import in this half-hour is whichever is smaller, the charger's power run for half an hour, or the grid energy still needed to reach the target state of charge given the home charging efficiency; nothing is imported once the battery has reached its target, or unless the EV is connected at home for the whole slot.

On the smart path this import is capped again by the smart charger's plan for that half-hour (section 7): the plan can only move charging to a cheaper forecast half-hour, never raise it above this rule.

**Preferred target is not the physical ceiling.** Home charging stops at 80% but never discards energy above it: an EV arriving at 85% does not charge.

## 6. Monte Carlo worlds

A **world** is the code's name for one simulated week for all 1,000 EVs: trips, plug-ins, weather and prices. On screen it is always called a simulated week.

```
             one population (1,000 EVs, fixed traits)
                                    |
                       simulated weeks (100)
                        prices with AR(1) noise
                                    |
                     unmanaged path  |  smart path
                      same inputs, paired per week
```

The smart charger needs no separate planning world (decision 0004 item 38 removed it): it plans each session from its own simulated week's day-ahead price path (decision 0004 item 48) and the cohort's typical departure clock time, both known at plug-in. Day-ahead prices publish progressively, at 13:00 London the day before delivery (item 53), so a plan only ranks half-hours already published by the moment it is made; anything later in its window is ranked on the recent published shape instead. It never sees that world's actual departures, or the forecast error the realised price adds on top.

**Paired channels.** In one simulated week, both paths get the same trips, plug-ins, weather and prices. `physics.py:simulate_fleet_intervals` runs the kernel on them without and with the smart-charging plan, so smart minus unmanaged measures only the action.

**World-first aggregation.** Each simulated week's EV (and cohort) quantities are summed per slot first; fleet SoC is total stock over total capacity. Only then does `physics.py:_band_frame` take the mean, P10, P50 and P90 across worlds. Percentiles are never summed: with cohort world totals `[0, 100]` and `[100, 0]`, every fleet total is 100, yet the cohort P95s add to 190.

**What the bands mean.** On most charts P10–P90 is the spread **across simulated weeks** of a fleet total: forecast uncertainty. The Overview SoC band is P5–P95 **across EVs** within a week: driver variety, not uncertainty (`summaries.py:average_day_bands`, item 22). Every chart says which.

**Difference bands.** Smart minus unmanaged is taken in each simulated week first, then summarised (`summaries.py:paired_difference_bands`, item 12). The UI never subtracts percentiles.

## 7. Smart charging

**The action** is price-optimised smart home charging (`action.py`, decision 0004 item 38, which replaced the 18:00–18:30 import ceiling, per-EV eligibility screen and planning-world select-or-not decision of items 2, 8, 25 and 30). The unmanaged path (`normal` in the code) charges as soon as the EV is plugged in; the smart path (`selected` in the code) plans each home session instead.

**Decision-time information.** At each home plug-in the smart charger knows only: the EV's battery stock at that moment and its preferred target SoC; the plug-in half-hour (the first whole half-hour the EV is connected for); the expected departure, which is the cohort's typical home departure clock time minus a safety margin (`departure_margin_hours`, an editable illustrative assumption, default two hours since decision 0004 item 51), so an evening plug-in expects to leave the next morning; and its own week's day-ahead price path (decision 0004 item 48). Day-ahead prices for a delivery date publish at 13:00 London the day before (item 53): a plan only ranks half-hours already published by then, and ranks anything later in its window on the expected shape (the mean published price for that half-hour over the most recent published days) instead. An ordinary evening plug-in sees every price to the next morning; only a plan made before 13:00 that day (an EV still plugged in after its expected departure, or an always-plugged EV) ranks part of its window on the shape rather than the real day-ahead price. It never sees that week's actual departure time, or the forecast error the realised price adds on top of what it ranked.

**Planning the session** (`action.py:plan_cheapest_slots`). The charger plans `(target − stock) / efficiency` grid kWh, filled into the cheapest forecast half-hours between plug-in and the expected departure, each capped at home charging power × 0.5 h. Filling the cheapest slot first is optimal here: the cost is linear in energy and every slot shares the same power cap, so no other plan can beat it. If the window cannot hold the need, every slot in it charges at full power: "as soon as possible" within the window.

**Executing the plan** (`physics.py:_smart_charging_slot`). The plan is run against the EV's *actual* session in each simulated week. An EV still plugged in when its window ends (it left later than expected, or is always plugged in) starts a new plan for its next expected departure. If the EV leaves before its plan finishes, the missed slots are simply not charged: it leaves below target, and any resulting public top-up or energy not recovered by the end of the week is priced (items 4, 5, 13, 32, 37). A plan window that would run past the study's end is simply clipped there: the window is still long enough that filling its cheapest slots first stays a fair comparison, so no special early-charging rule is needed (the noon-to-noon horizon, item 52, retired the item 49 rule that once forced such sessions to charge as soon as possible). Common random numbers hold throughout: smart charging draws no random number, so both paths see the same random futures.

**Paired evaluation and cost.** `action.py:calculate_wholesale_world_cost_effect` values each simulated week:

| Component | Calculation |
| --- | --- |
| Home import cost | Σ (smart − unmanaged home import) × that week's day-ahead synthetic price |
| Public charge cost | (smart − unmanaged public import) × £0.79/kWh |
| Unrecovered energy value | max(0, unmanaged − smart closing stock) × £0.79/kWh |
| Unserved travel value | max(0, smart − unmanaged unserved travel) × £0.79/kWh |
| Illustrative total | sum of the four |

£0.79/kWh is an editable illustrative rate, not a tariff (item 4). Unrecovered energy is priced as if bought back at a public charger. Unserved travel is priced so unused battery energy cannot look like a saving (item 13); it is zero under item 32 but stays in the total. Battery-side kWh are valued at the grid-side rate, ignoring public losses. Differences within 1e-6 kWh count as zero. `energy_not_recovered` flags a simulated week whose smart path ends with lower stock, more public import or more unserved travel than the unmanaged path, beyond 1e-6 kWh (item 5). It is a fleet-wide yes/no that can fire on one EV missing a single half-hour of charge, so read the size beside it before calling a negative total a saving: `not_recovered_summary` reports unrecovered kWh as a share of weekly home import and the EV-sessions affected, with a material threshold (1% by default, decision 0004 item 45).

**A finding: herding onto the same cheap half-hours.** Within one week every EV plans on the same day-ahead path, so smart charging can move most of the fleet onto the same cheap half-hours: in the default run (1,000 EVs, 100 weeks, seed 42, study starting 12 January 2026, intraday dispatch on), each week's own smart peak is about 4,550 kW (P10–P90 4,330–4,724 kW), about 1.76 times the normal evening peak of about 2,565 kW (P10–P90 2,465–2,708 kW), even though each week's day-ahead prices move the cheapest half-hour across a wide band (P10 13:00 to P90 08:00, P50 01:00, across days and weeks). The model has no site or network limit, and within one week every EV plans on the same day-ahead path, so nothing spreads that load.

## 8. How to read the dashboard

Nine pages (`docs/DASHBOARD_DESIGN.md`, item 28). Time axes are London time; the centre line is P50 with a P10–P90 band; unmanaged is a lighter line and smart is teal (item 27). Nothing runs until you click **Run simulation**.

**Overview.** Bars: share of the fleet plugged in at home by half-hour on an average day. Line: mean SoC with a P5–P95 band **across EVs**.

**Drivers.** *Plug-ins*: histograms of plug-in time and plug-in SoC, whiskers P10–P90 **across weeks**, CNZ values as context, not a target. *Sessions*: per-archetype small multiples of each session's plug-in time, departure time, SoC at plug-in, energy needed and flexibility. *One EV*: the median week replayed through the same kernel (`individual.py:replay_one_ev`), with SoC for both paths and top-ups marked. *Household*: the same one EV, across every simulated week, framed as what that driver gets. *Archetypes*: one panel per cohort. *Fleet week*: 336 half-hours across the simulated weeks.

**Smart charging.** *1 Plan*: that week's own day-ahead price path (item 48) the charger ranked, the smart-charging outcome (price paid, share of home import moved, early departures) and how much import moved out of each forecast-price third. *2 Response*: home import for both paths, then the paired difference band, showing when charging moved to and from (section 7 covers the herding finding). *3 Value and risk*: one dot per week for the illustrative total, flagged weeks marked, and the driver-side risk (early departures, shortfall).

**Trading.** *1 Market*: the day-ahead price path the fleet traded against that week. *2 Position*: what the illustrative trader sold day-ahead, and what the fleet actually moved. *3 P&L and risk*: what the trading overlay made or lost, and how firm it was (section 11).

**Supplier.** *1 Availability and cost curve*: what the fleet could move, and at what price. *2 Positions*: what was sold day-ahead, and what was held at the end. *3 Supplier P&L*: what smart charging does to a supplier's week. *4 Firm MW*: how much turn-down can be promised firmly (section 13). *5 Charger makers*: what a charger maker's enrolled device earns. *6 Customers*: what customers can be told, and which gain least.

**Partners.** One pitch card per outside audience (*Driver*, *Energy supplier*, *Charger maker*, *Carmaker*, *Fleet and leasing*), reusing the charts and numbers already built for Drivers, Smart charging, Trading and Supplier: what that partner buys and its route to market, four headline numbers, a hero chart with supporting charts, a table of what they would ask (a stored reading, or "not modelled"), and links onward to the pages that answer each one in full.

**Replay.** Plays one simulated week back half-hour by half-hour with a moving "now". *Fleet*: what the fleet knew, did and earned as the week unfolded. *Customer*: the same, for one driver's charger.

**Compare.** The last three runs are kept. `summaries.py:compare_runs` pairs two runs week by week, B minus A; runs with the same seed, fleet size and model share matched futures. Runs with a different number of simulated weeks or a different horizon cannot be paired, and the page says why.

## 9. Limitations and next steps

- **Not calibrated.** Next: fit drive, plug-in and SoC distributions to observed data and back-test on a held-out period.
- **Prices are simulated, not historical.** They come from a net-demand and supply-curve model calibrated to Elexon data (section 10), not from a replay of real days.
- **No site or network limit, one shared price signal per week.** Every EV in a simulated week plans against that week's own day-ahead price path, so smart charging can still herd much of the fleet onto that week's cheapest half-hours (section 7); a real optimiser would see and spread the resulting load. Next: a per-site or per-feeder limit, and diversity in what each EV forecasts.
- **Simple trips.** One round trip a day, fixed speed, equal legs.
- **Top-ups are instant and always available.** Top-up time is not modelled. Because EVs top up at 10% (decision 0004 item 37), no plug-in in the model starts below 10% SoC, though plug-ins between 10% and 20% SoC can happen; CNZ observed about 3% of plug-ins below 10%. The threshold is editable; the gap is stated rather than tuned away.
- **No charging taper, no site limit, shared weather.** Home power is flat up to the target; one daily temperature serves the whole fleet.

## 10. Prices and shocks

The wholesale price is built, not looked up. National net demand comes from a smooth daily demand shape plus a heating term from the day's sampled temperature, minus wind and solar, each with their own day-to-day and half-hourly noise; a supply curve turns net demand into a price, so a tighter system moves price more for the same change in demand. On top of that, scripted events (section 12) and a random shock process add known (day-ahead) or surprise (intraday and imbalance only) swings, up or down (decision 0004 items 55 and 56). The shape was calibrated once against free Elexon half-hourly market data (`scripts/fetch_elexon_prices.py`), never fetched live by the app. Every price is still synthetic, and every £ figure built from it stays illustrative. Notebook 03 (`notebooks/03_prices_and_markets.ipynb`) builds each term one at a time and compares the result against the Elexon shape.

## 11. Trading and the P415-style baseline

A trading desk (the code calls it a trading overlay) sits beside the physics, reading what the fleet actually did without ever changing it, so every trading strategy shares the same simulated futures as the model itself (decision 0004 items 57 and 58). It compares the smart-charged fleet's import against a rolling baseline built the way Great Britain's own P415 balancing product does (BL01: the average of recent similar days), then works out a day-ahead position, an hourly intraday book and an imbalance settlement against that baseline, for three example strategies (perfect foresight, day-ahead only, and the full intraday strategy). The result is one net figure, always labelled "illustrative simulated trading P&L" (decision 0005): never Axle cash, never a bid or a real settlement. Notebook 06 (`notebooks/06_trading.ipynb`) walks the three settlement stages and the baseline in full.

## 12. Events and zones

An event is a scenario you choose, not something the model draws at random: a cold still evening, a surprise price spike, a called turn-down or a local turn-up, each the same in every simulated week so it can be compared with and without it under identical futures (decision 0004 item 55). A known event moves the day-ahead price the smart charger plans against; a surprise event only moves the intraday and imbalance prices it never sees at plug-in. Every EV also belongs to one of four illustrative network zones, used to scope a local event and to show that zone's import against an optional local headroom. Notebook 07 (`notebooks/07_events_and_zones.ipynb`) runs both event kinds and both zone-scoped and fleet-wide requests.

## 13. Shared plug-in factors and firm MW

Real EVs do not plug in independently of one another: a cold night, a school holiday or a maker's cloud outage moves many drivers' habits together. On top of each EV's own habit, the model adds a shared day-level and week-level skip factor (decision 0004 items 60–64) and, per illustrative charger manufacturer, a shared response rate and outage probability (item 59), so a "90% sure" turn-down figure means what it says instead of shrinking towards zero as the fleet grows. Notebook 02 (`notebooks/02_driver_behaviour.ipynb`) builds the shared skip share and the manufacturers. Notebook 09 (`notebooks/09_top_down_quantile_model.ipynb`) walks the deliverable-MW quantiles and the leave-one-week-out calibration backtest behind the firm-MW figures (item 59, plan §11); see also `docs/contracts/trading-events-v1.md` §10 and the built Supplier ▸ 4 Firm MW lens (`docs/DASHBOARD_DESIGN.md` section 4.8).

## 14. The supplier view

A supplier buying this fleet's flexibility does not see the trading overlay's P&L; it sees its own procurement cost fall, its own hedge error move under half-hourly settlement, third-party grid-event income, and a set of customer- and partner-facing figures such as value per household per month and £ per enrolled charger (decision 0004 items 58, 62 and 65; decision 0006). Notebook 08 (`notebooks/08_supplier_view.ipynb`) walks this view. See also `docs/contracts/supplier-v1.md` and the built Supplier page (`docs/DASHBOARD_DESIGN.md` section 4.8).

## 15. Households

The same run also answers what one customer gets: how often their EV was ready at departure, what a typical year is worth to them, how many sessions were affected, and how much turn-down that one driver could offer (decision 0004 items 54, 58, 62 and 65). No concept notebook covers this yet. See `docs/contracts/household-v1.md` and the built Drivers ▸ Household lens (`docs/DASHBOARD_DESIGN.md` section 2).

## Where to look in the code

| Concept | File and function |
| --- | --- |
| Run and draw order | `model/forecast.py:run_forecast`, `simulate_forecast` |
| Population | `model/sampling.py:build_population` |
| Energy balance kernel | `model/physics.py:_simulate_path` |
| Public top-ups | `model/physics.py:_top_up_battery_kwh` |
| World-first bands | `model/physics.py:_band_frame` |
| Smart charging plan | `model/action.py:plan_cheapest_slots`, `model/physics.py:_smart_charging_slot` |
| Cost and flag | `model/action.py:calculate_wholesale_world_cost_effect` |
| Summaries and Compare | `model/summaries.py:build_summaries`, `compare_runs` |
| One EV replay | `model/individual.py:replay_one_ev` |
