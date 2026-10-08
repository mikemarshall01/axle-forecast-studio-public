# Code guide

This is the reading order through the code for an interviewer. It follows one run from its input values to the chart on screen. Each section gives what the module owns, its key functions, its rules and the decisions behind them. Decision numbers refer to `docs/decisions/0004-explicit-action-and-horizon-choices.md` unless stated. The result shape is `docs/contracts/results-v2.md`; the plain-English method is `docs/explainers/how-the-forecast-works.md`.

All model code is in `src/axle_studio/model/`. All dashboard code is in `src/axle_studio/ui/`. The entry point is `streamlit_app.py`.

## 1. `model/assumptions.py`: every value the model uses

Owns every active assumption as an `Assumption` record: value, unit, evidence kind (source, illustrative or synthetic), source, meaning and editable bounds (item 21). No other module states a model value. Modules that need a fixed value, such as the weather reference temperature or the action hour, read it from the records here.

Key functions:

- `resolve_values` applies the dashboard's edited values to the defaults and checks them.
- `validation_errors` checks values against each record's bounds, plus one cross-field rule: the top-up target must be above the top-up threshold.
- `run_settings`, `cohort_fixture` and `forecast_inputs` turn the values into the arguments of `forecast.run_forecast`.
- `result_assumptions` returns the records a result carries, with the values that run used.

Where two sources disagreed, the value traced in `docs/sources/cohort-crosswalk.md` wins (item 21). There is one home charging power, 7 kW (item 34). Runs open at 80% SoC with a warm-up, seven days by default and editable 3–14 (item 36, widened by item 52).

## 2. `model/clock.py`: the UTC half-hour grid

Owns the fixed UTC slot boundaries and the conversion from London wall-clock times to UTC.

Key functions: `utc_half_hour_boundaries` and `london_wall_time_to_utc`.

Rule (item 1): the study is always `48 × study_days` UTC half-hours from London noon on the start date (moved from midnight, item 52, so a noon-to-noon block holds a whole overnight session). Warm-up slots count back from that instant. A London clock change inside the span is allowed, so a local date can hold 46 or 50 half-hours. Behaviour is stated in London time; every interval key is UTC.

## 3. `model/settings.py`: run shape types

Owns three small frozen types. `RunSettings` holds one run's shape (dates, fleet size, worlds, seed) and the few scalars the kernel reads directly. `CohortSpec` and `CohortFixture` describe the six source archetypes. Nothing here holds a value; `assumptions` builds these types.

## 4. `model/sampling.py`: every random draw

Owns the persistent EV population, each world's daily trips, home plug-in sessions, daily weather and the synthetic prices.

Key functions:

- `build_population` assigns EVs to cohorts by largest remainder, shuffles them, and draws one persistent mileage multiplier per EV.
- `sample_daily_trip_inputs` draws whether each EV drives, how far and when it leaves.
- `sample_connection_opportunities` draws whether each EV plugs in at home each evening from its own plug-in clock; the EV unplugs at the first later departure on a day it drives (departure clock), so a day without a trip keeps it plugged in (decision 0004 item 68).
- `sample_daily_temperature` draws one temperature per world and day.
- `generate_market_price_paths` builds each simulated week's own day-ahead price path and its realised (evaluation) price path (item 48): every world gets a different day-ahead curve, so the cheapest half-hour moves from week to week instead of sitting at one fixed clock time.

Rules:

- One `np.random.default_rng(seed)` per run, passed to every sampler in a fixed order (item 14). There is no hashing or per-channel seeding.
- Independent draws use named `scipy.stats` distributions with `random_state=rng`. Mileage multipliers are mean-one lognormals, `s = sqrt(log(1 + CV²))`, scale `exp(-s²/2)`. Drive and plug-in choices are Bernoulli by inverse transform (`U < p`). Clock shifts are normal, rounded to 30 minutes and clipped.
- Two independent AR(1) noise channels, `e_t = φ e_(t-1) + u_t` from statsmodels `ArmaProcess.generate_sample`, fed by the run's generator: day-ahead noise gives each world its own day-ahead path on top of the shared daily cosine shape, and a second, independent channel is the forecast error added to the day-ahead path to give the realised price (item 48). Each channel's innovation SD is `sd × sqrt(1 - φ²)`, so its marginal SD stays at the stated value.
- Common random numbers: every sampler takes a fixed number of draws whatever its parameters. Editing one assumption changes only its own channel, so Compare keeps matched futures (item 6). The comments at each draw name the scipy call that was rejected because it would skip draws.

## 5. `model/physics.py`: the energy balance

Owns each EV's battery, half-hour by half-hour, in every world, and the world-first sums.

Key functions:

- `public_top_up` checks and returns the three public top-up values: efficiency, threshold and target.
- `effective_driving_efficiency` applies the weather equation `efficiency = base × (1 − min(0.5, s × |T − 20 °C|))`.
- `simulate_fleet_intervals` runs the kernel for the normal path, and for the selected path under the smart-charging plan when one is given. It returns fleet and cohort totals per world and their across-world bands.
- `simulate_unit_intervals` runs the same kernel but keeps EVs separate. `summaries.py` calls it in chunks to build per-EV detail, which the summaries and the one-EV replay both read.

The kernel is `_simulate_path`. Its section comment gives the event order inside one slot: travel, public top-ups, occupancy, home connection, home charging, then checks. The energy balance per EV and slot is:

```
closing = opening + home battery added + public battery added − travel served
```

It fails the run if the residual exceeds 1e-12 kWh or stock leaves `[0, capacity]`.

Rules:

- Travel energy is `miles / effective efficiency`, spread evenly over each leg's driving time.
- Threshold top-ups (item 32, in `_top_up_battery_kwh`): when a trip would take the battery below the threshold, the EV tops up to the target at a public charger and carries on. It repeats if one top-up is not enough. EVs never strand, so unserved travel is zero by construction. Grid import is battery energy divided by the public efficiency. Time spent charging is not modelled.
- Home charging (`_home_charge_kwh`) needs connection for the whole slot (decision 0001). It is `min(power × 0.5 h, (target − stock) / efficiency)` on the normal path, where power is the home charger power (item 34). On the selected path, `_smart_charging_slot` caps this further to the smart charger's planned kWh for that half-hour (item 38): the plan can only move charging to cheaper forecast half-hours before the EV's expected departure, never raise it above the normal rule.
- The preferred target (80%) is not the physical ceiling. Home charging stops at the target; the battery can hold up to 100%.
- Warm-up slots are simulated but not reported. Warm-up is shared history on both paths: a smart-charging plan only starts once the study itself begins (`action.py`, "Plans start in the study only").
- World-first: EVs are summed within each world before any mean or percentile is taken across worlds (`_band_frame`). Percentiles are never summed.

## 6. `model/action.py`: price-optimised smart home charging and its cost

Owns the one candidate action, price-optimised smart home charging (decision 0004 item 38, which replaced the 18:00 import cap, per-EV eligibility screen and planning-world select-or-not decision of items 2, 8, 25 and 30). At each home plug-in, the smart charger plans just enough grid energy to reach the preferred target by the cohort's typical departure clock time minus a safety margin, using the cheapest forecast half-hours available in that window. It also owns the illustrative cost effect.

Key functions:

- `smart_charging_inputs` builds the `SmartCharging` decision-time inputs for one run: each world's own day-ahead price path (item 48) and, per EV and London date, the expected departure (the cohort's typical home departure clock time minus `departure_margin_hours`, default two hours, item 51).
- `plan_cheapest_slots` plans grid kWh per half-hour for a batch of sessions, giving the cheapest forecast slot in the window as much as home charging power allows, then the next cheapest, and so on; this is optimal because the cost is linear in energy and every slot shares the same power cap.
- `calculate_wholesale_world_cost_effect` values the plan against each evaluation world's actual departures and realised prices.

How smart charging works:

1. Decision-time information only. The plan uses its own world's day-ahead price path and the cohort's typical departure clock time, both known at plug-in; it never sees that world's actual departure or realised prices. The whole simulated week's day-ahead path is treated as published at the horizon start, so a plan window longer than about a day (a weekend) sees a little further ahead than a real day-ahead auction would by then (item 48).
2. `physics.py:_smart_charging_slot` starts a new plan whenever an EV is plugged in for a whole slot with no plan in force: from its stock at that moment, the plug-in half-hour and the next expected departure, it calls `plan_cheapest_slots` over the cheapest day-ahead half-hours before that departure. A session whose next expected departure falls after the study end cannot see its full window, so it charges as soon as it is plugged in, like the normal path (item 49).
3. Each slot's planned energy caps that slot's home charging (`_home_charge_kwh` still applies too, so the plan can only move charging, never raise it).
4. The plan is executed against the EV's *actual* session in the evaluation worlds. If the EV leaves before the plan finishes, the missed slots are simply not charged: the EV leaves below target, and any resulting public top-up or energy not recovered is priced (items 4, 5, 13, 32, 37).
5. Common random numbers hold: smart charging draws no random number of its own, so both paths see the same random futures.
6. Within one simulated week every EV still ranks the same day-ahead path, so smart charging can still bunch much of the fleet into that week's own cheapest half-hours; there is no site or network limit to stop it, and this stays labelled as a finding rather than corrected with a cap (item 48).

Cost per world is split into four labelled parts: home import at the world's realised synthetic price, public import difference at £0.79/kWh, unrecovered energy and added unserved travel, both at the same rate (items 4 and 13). `energy_not_recovered` flags a world beyond a 1e-6 kWh tolerance (item 5). Every GBP figure is illustrative, never settlement or Axle cash (decision 0003).

Some names below the UI keep the mechanism's original "wholesale" wording, from before item 38 replaced the 18:00-cap wholesale action with price-optimised smart charging: `calculate_wholesale_world_cost_effect`, the policy id `explicit_synthetic_wholesale_v1`, the "explicit wholesale action" model label in the contracts. They name the mechanism, not the app; the app says "Smart charging" throughout. Likewise, several assumptions' `source` strings read "formerly configs/illustrative-*.json" (`model/assumptions.py`): those JSON files were removed when item 21 moved every value into this module, and the string is a provenance trail, not a path that still resolves.

## 7. `model/forecast.py`: the one runner

Owns `run_forecast` and the `ForecastResult` type.

Key functions:

- `run_forecast_from_assumptions` is the app's entry point. It takes every input from `assumptions`.
- `run_forecast` calls `simulate_forecast`, packages the frames under the contract's names and calls `summaries.build_summaries`.
- `simulate_forecast` orders the calls. No-action: population, then evaluation worlds (trips, connections, weather), then the kernel. Action: population, then the same evaluation worlds draw as the no-action model (so both models see identical random futures for a given seed), then the price paths (each world's own day-ahead noise and its independent forecast-error noise, item 48) and the smart-charging inputs (deterministic from the population and each world's day-ahead path, no draw), then the kernel for both paths with the smart-charging plan, then the cost effect. There is no separate planning world: item 38 removed it.

Both paths in a world share the same inputs, so selected minus normal measures only the action.

## 8. `model/summaries.py`: everything derived after a run

Owns every frame the dashboard reads that is not a direct kernel sum.

Key functions:

- `build_summaries` is the single entry point `run_forecast` calls.
- `average_day_bands` gives the Overview's average day. Connected share has a P10 to P90 band across worlds; SoC has a P5 to P95 band across EVs.
- `find_sessions` and `plug_in_summary` give plug-in time and plug-in SoC. Plug-in SoC is the closing stock before the session starts, a stock-flow output, never sampled.
- `paired_difference_bands` and `difference_bands` subtract per world first, then take quantiles (item 12).
- `compare_runs` and `slim_run` pair two runs world by world. They work from slim run records (item 35).

Per-EV detail comes from rerunning `simulate_unit_intervals` on chunks of ten worlds, so memory stays small.

## 9. `model/individual.py`: one EV's week

Owns `replay_one_ev` and `replay_one_ev_bands`. They run the same vectorised kernel on a one-EV slice of the stored inputs (`summaries.kernel_slice`), with that EV's own smart-charging plan and the run's top-up rule (items 10 and 20). EVs are independent, so the sum of all replays equals the fleet frame; the tests check this. Replays are cached on the result.

## 10. The dashboard

`streamlit_app.py` builds top navigation from `ui/registry.PAGES`, draws the run bar, runs the page, and only then runs the model if Run was clicked.

**`ui/run_controller.py`** owns the draft, the explicit Run and the run bar.

- `render_run_bar` draws the model control, the status chip, Edit assumptions and Run.
- `request_run` is the Run button's callback. It only sets a flag.
- `execute_requested_run` runs the model after the page is drawn, through `run_full`, then reruns.
- `run_full` calls `run_model`, which calls `run_forecast_from_assumptions`. It keeps the last three runs as `RunRecord`s; only the latest keeps its full result (item 35). A failed run keeps the previous result.
- `run_status` names the six states. "Stale" means the draft differs from the active run's values.

Nothing else starts a full Monte Carlo: not opening, editing, navigating, Reset or ordinary reruns (decision 0002).

**`ui/registry.py`** lists the pages, their lenses and the question each answers (item 28; see `docs/DASHBOARD_DESIGN.md` section 2 for which pages are built, in progress or pending). **`ui/pages.py`** holds `VIEWS`, the one table from (page, lens) to a view. It also wires the One EV replay, `compare_runs` and the Edit assumptions dialog. `unpairable_reason` explains why two runs cannot be compared.

**`ui/views/`** has one file per screen, such as `overview.render_overview` or `action_response.render_action_response`. Each follows `render_<name>(st, result)` and only reads the result. `parameters.render_assumptions_editor` is the dialog body, generated from the assumption records. **`ui/style.py`** holds the shared Plotly template and colours (items 27 and 29).

## 11. Trading, supplier, availability and household modules

These extend the two-model foundation above without changing it (decision 0004 items 43 and 54–66); each reads the kernel's outputs and adds no physics of its own. All are model code under `src/axle_studio/model/`; not every one is mounted on a dashboard page yet (`docs/DASHBOARD_DESIGN.md` section 2 gives build status).

- **`model/events.py`** owns the scripted events table, its four presets and their validation, and turns each enabled event into the price-shock profiles, planner price adjustments, trading mask, delivery-vs-baseline and outage probability the rest of the model reads (items 55, 56, 58). Entry points: `event_table`, `validate_events`, `scripted_shock_profiles`, `planner_adjustments`, `event_delivery`.
- **`model/market.py`** owns the illustrative trading overlay: the BL01-lite settlement baseline, expected availability and expected plan at each decision, the day-ahead and intraday positions of three strategies, imbalance settlement and the per-night ledger (items 57, 58; decision 0005). It reads the kernel's outputs and never changes them, so all three strategies share one physics run's random futures. Entry points: `run_trading`, `bl01_lite_baseline`, `newsvendor_positions`, `settle`.
- **`model/availability.py`** owns firm-MW deliverable kW per world and slot, for turn-down and turn-up held 0.5–4 hours, at the day-ahead and 17:00-intraday horizons, and how firm each figure is (item 59). Entry points: `ev_contributions`, `build_frames`, `intraday_distribution`.
- **`model/availability_backtest.py`** owns the leave-one-week-out calibration backtest of those forecasts: coverage, pinball loss and reliability, scored on the run's own simulated weeks (item 59; trading contract v1 §10.3). Entry points: `availability_backtest`, `availability_reliability`, `availability_backtest_summary`.
- **`model/product.py`** owns the supplier-facing reading of `availability`'s frames: the product sheet (MW, firmness, recovery, notice per window), a per-EV settlement file, charge completion, and firmness by charger manufacturer (item 59; trading contract v1 §10.5). Entry points: `product_sheet`, `settlement_file`, `charge_completion_summary`, `firmness_by_manufacturer`.
- **`model/supplier.py`** owns the illustrative supplier P&L (procurement cost, hedge error under a profiled vs flat day-ahead hedge, grid-event income, customer payments; decision 0006), the hedge-block and CO₂-shifted views, and the Partners frames for device makers (items 58, 65; supplier contract v1 §3–§4). Entry points: `supplier_pnl_world`, `hedge_block_world`, `carbon_shift_world`, `partner_world`, `build_frames`.
- **`model/growth.py`** owns the Partners lens's growth-expander arithmetic (funnel, annual £, lever tornado, months to fund a discount, cumulative payout fan) from a device maker's own sales-funnel guesses; it is not physics, policy or settlement (item 62), so it may take presenter inputs (supplier contract v1 §5). Entry points: `funnel`, `annual_gbp`, `tornado`, `months_to_fund_discount`.
- **`model/household.py`** owns what one customer gets from smart charging: per-(world, EV) weekly totals (`household_ev_world`) and their group summary in three readings — a typical week, the group ratio, and across customers' average years (household contract v1 §3.1–§3.2). It also owns the one-EV `HouseholdCard` (availability, timing, reliability) that the built Drivers ▸ Household lens reads. Entry points: `build_frames`, `outcomes_summary`, `household_card`.
- **`model/replay.py`** owns the fleet replay of one simulated week: what was known, positioned and earned at each hourly decision instant, re-indexed from the run's own arrays with nothing after that instant leaking in (replay contract v1 §1; item 66). Entry points: `build_replay_week`, `forward_curves`, `positions_in_force`, `energy_saving_to_date`.

## 12. Result-frame validators and lint scripts

Each module above has its own independent-route validator, called from `result_contract.validate_result_v2` on every result that carries the frame, plus a fixture module the tests build against:

- `tests/fixtures/availability_contract.py`: column specs and `validate_availability` for the availability frames, with a synthetic per-EV stand-in.
- `tests/fixtures/availability_backtest_contract.py`: specs and validator for the backtest frames, with synthetic world-slot data.
- `tests/fixtures/product_contract.py`: specs, validators and `make_product_case`, a synthetic trading stand-in for the product frames.
- `tests/fixtures/supplier_contract.py`: `validate_supplier_frames`, recomputing the ten supplier/partner frames independently from the run's own arrays.
- `tests/fixtures/household_contract.py`: `make_household_frames`, a synthetic build of the two household frames independent of `model/household.py`'s chunk hook.
- `tests/fixtures/replay_contract.py`: the machine-readable `ReplayWeek`/`OneEvTimeline` specs, a synthetic fixture replay, and `validate_replay_week_v2` / `validate_one_ev_timeline_v2`.

Two lints check the UI stays inside the design tokens and chart conventions, plus one real-browser audit, each exiting 1 on a finding:

- **`scripts/dashboard_lint.py`** renders every view against the synthetic fixture result and checks the resulting Plotly figures against `docs/DASHBOARD_DESIGN.md` section 3.5 and decision 0004 items 26, 27 and 50 (heights, legends, margins, hover, hidden modebar).
- **`scripts/ui_lint.py`** scans the UI source for colours, `rgba(...)`, pixel lengths or chart font sizes spelled inline instead of read from `ui/style.py` and `.streamlit/config.toml` (`docs/DESIGN_TOKENS.md`).
- **`scripts/axe_audit.py`** is a real-browser check, not a lint: it starts the app itself, runs one simulation, then runs axe-core (WCAG 2.x) and saves 1440/390 px screenshots for every page and lens in `registry.PAGES`. It is opt-in, wrapped by `tests/ui/test_axe_audit.py` behind `AXLE_BROWSER_AUDIT=1` and the `slow` marker (the default suite never starts a browser), and needs `uv run playwright install chromium` once.

## 13. Running the checks

From the repository root, with the `dev` dependency group installed (`uv sync --frozen --group dev`):

```bash
uv run --frozen --group dev ruff format --check .
uv run --frozen --group dev ruff check .
uv run --frozen --group dev pytest -q                     # everything, including the slow notebook test
uv run --frozen --group dev pytest -q -m "not slow"        # the fast suite only

uv run --frozen --group dev env PYTHONPATH=src:tests python scripts/dashboard_lint.py
uv run --frozen --group dev python scripts/ui_lint.py

# opt-in browser audit; `uv run playwright install chromium` once first
AXLE_BROWSER_AUDIT=1 uv run --frozen --group dev pytest -q tests/ui/test_axe_audit.py
```

`tests/test_concept_notebooks.py` is the slow notebook test: it executes every concept notebook (`notebooks/0*.ipynb`, excluding `01_*`) end to end, so a code change that breaks a notebook's prose or numbers fails the suite instead of leaving it silently stale (plan section 10). It runs inside the plain `pytest -q` above and is skipped by `-m "not slow"`.

## Where to look

| Question | Where |
| --- | --- |
| Where is the energy balance? | `physics._simulate_path`, step 6, and the section comment above it |
| How do public top-ups work? | `physics._top_up_battery_kwh` (item 32) |
| How is home charging limited? | `physics._home_charge_kwh` |
| How does the smart charger plan a session? | `action.smart_charging_inputs`, `action.plan_cheapest_slots` and `physics._smart_charging_slot` |
| How is the action valued? | `action.calculate_wholesale_world_cost_effect` |
| How are bands computed world-first? | `physics._band_frame`; differences in `summaries.paired_difference_bands` |
| What is random, and in what order? | `sampling.py` and the module docstring of `forecast.py` |
| Where do assumptions come from? | `assumptions.py`; source cells traced in `docs/sources/cohort-crosswalk.md` |
| How does a Run work in the UI? | `run_controller.request_run`, `render_run_bar`, `execute_requested_run`, `run_full` |
| How does Compare pair runs? | `summaries.compare_runs`, `pages.unpairable_reason` |
| How is one EV replayed? | `individual.replay_one_ev` |
| How does the clock handle a clock change? | `clock.utc_half_hour_boundaries`, `clock.london_wall_time_to_utc` |

Tests mirror this layout in `tests/model/` and `tests/ui/`. `tests/fixtures/scalar_kernel_oracle.py` is a slow, independent scalar version of the kernel (without top-ups) used as a parity check.
