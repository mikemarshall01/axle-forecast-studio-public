---
status: accepted-physical-foundation-contract
ArchitectureReviewRequired: yes
scope: minimum-physical-foundation
---

# Python model contract: minimum physical foundation

## Authority and scope

This is the accepted contract for the minimum physical foundation in the separate Python build, following independent review and lead acceptance on 26 September 2026. Its pending choices and release gates remain pending. The preserved JavaScript repository and its model contract are migration evidence, not the active runtime or authority.

This contract covers the foundations that can be implemented and tested with small synthetic fixtures: the UTC clock, persistent population identity, paired exogenous inputs, battery stock-flow, isolated charge, mobility, public-stop and export helpers, physical route exclusivity, and world-first aggregation, plus the explicit wholesale action and no-action models built on that foundation. Sampling follows the simplified run-level NumPy approach in [random-v1](contracts/random-v1.md), extended by decision 0004 item 14 (`scipy.stats` and statsmodels). It does not claim that a source catalogue, default run, route ledger or release exists.

[Decision 0001](decisions/0001-device-slot-physical-exclusivity.md) is accepted authority for whole-device-slot physical exclusivity, as narrowed by [decision 0004 item 18](decisions/0004-explicit-action-and-horizon-choices.md) (see "One dispatchable route per physical path-slot" and the Superseded appendix). [Decision 0004](decisions/0004-explicit-action-and-horizon-choices.md) is accepted authority for the explicit action, pricing and horizon choices in "Explicit wholesale action model" below. Commercial interpretation beyond these decisions remains pending.

## Synthetic foundation versus release runs

Foundation tests use explicit synthetic values. These values are arithmetic or causal oracles only. They are not observed fleet data, calibrated behaviour, commercial terms, product defaults or a reference forecast. A narrow synthetic foundation may proceed with applicable physical and configuration checks; acceptance does not promote fixtures to source-backed production inputs.

A release or default run requires an independently reviewed source catalogue, derived-input provenance, executable configuration and source-to-field crosswalk. It must keep observed data, editable assumptions, synthetic history, toy posterior output and real backtests distinct. Retail prices must not be labelled wholesale or balancing settlement data. Until those prerequisites are accepted, no source-backed default or release run is authorised.

A full Monte Carlo may start only from one explicit, valid **Run** click. Opening the app, editing or importing settings, navigating, changing a view, Reset and ordinary Streamlit reruns start no full run. An invalid Run starts none, and a duplicate click while a run is active starts no additional run. There is no automatic draft preview: decision 0002's bounded-preview design is superseded and retired (see the Superseded appendix), so a draft edit updates only the draft and never renders a partial result.

## Time and boundary rules

The clock contains strictly increasing UTC boundaries `b[0:T+1]`. Interval `t` is the half-open interval `[b[t], b[t+1])`, lasts exactly 1,800 seconds, and owns flows that occur within it. Battery state `E[t, unit]` is the stock at `b[t]`; interval `t` writes `E[t+1, unit]`.

Local civil time is for display and explicit calendar rules only. [Decision 0004 item 1](decisions/0004-explicit-action-and-horizon-choices.md) replaces the blanket offset-change rejection for the two explicit models: the seven-day study is a fixed 336 UTC half-hour slots starting at London local noon of the study start date (moved from midnight, item 52, so a noon-to-noon block holds a whole overnight session instead of cutting the first night in half), warm-up is fixed 30-minute UTC slots counted back from that instant, and a London offset change inside the span is allowed. "Day d" for trips, connections and weather is the London calendar date, with local clock times converted at that date's actual offset; a change date has 46 or 50 local half-hours, and daily rules (such as an EV's expected departure clock time) apply per local date. The original guard — reject the complete warm-up plus study span if the UTC offset changes anywhere in it, for both spring and autumn changes, including a change only in warm-up — is superseded for the explicit models and kept only as the historical rule for the removed legacy modes (see Superseded appendix).

At a boundary, the preceding interval closes before the new interval begins. A departure or unplug effective exactly at `b[t]` blocks home flow in interval `t`. An arrival effective exactly at `b[t]` may enable interval `t` only when there is no simultaneous departure or unplug for that EV. The precedence for a same-EV arrival and departure or unplug at the same boundary is **pending**; such input must fail validation rather than depend on mutation order.

A return inside an interval does not enable home charging until the next boundary. For an explicitly homebound two-leg journey, the remainder after physical return is `home_unplugged`, not `home_connected`; no home flow occurs in that interval. A later interval can charge only when connection covers the full slot. If the destination is genuinely unknown, the remainder remains `post_return_unclassified_fraction`: home must not be inferred. Resolved primary occupancy fractions plus any unclassified fraction must be non-negative and sum to one within tolerance. The `physical_state` label is a summary and does not replace these within-slot fractions. The `return_miles` leg of the current explicit journey kernel is homebound and its end is home arrival; a more general journey schema must carry destination meaning explicitly. Binary home flow requires `home_connected_full_slot=true`, full home occupancy, and zero drive and public-charge fractions.

## Population and paired worlds

The model generates persistent unit identity and physical traits once, derives a `population_id` from the completed population, then reuses the same values in planning and evaluation worlds. The derived `population_id` describes the completed population; it is not an input to sampling that population.

Planning and evaluation use distinct future samples while sharing the same population. Within one evaluation world, normal and selected counterfactual paths receive the same sampled exogenous arrays for price, weather, system need, travel, plug and fault inputs. Tests check that the same inputs are passed to both paths, not that independently drawn arrays happen to match or differ by chance.

The simulation runner creates `np.random.default_rng(seed)` once per explicit full run and passes it to sampling functions. The planner cannot receive future evaluation inputs. No semantic key encoder, channel registry, per-draw hash or runtime-profile gate is required. Sampling functions still validate external distribution parameters and preserve the physical invariants below.

A decision for interval `t` may use only boundary state and information with `available_at_utc <= b[t]`. Realised faults, later returns, public stops, closing stock and other post-cut-off outcomes cannot enter the decision.

Battery state persists across intervals, trips, stops, connections and days. Arrival SoC is `opening_battery_kwh / physical_capacity_kwh` at the applicable arrival boundary. It is never sampled independently.

## Physical units, signs and conservation

Battery capacity and stock use battery-side kWh. Power uses interval-average grid-side kW. Home and public imports and grid export use non-negative grid-side kWh. Home and public battery additions, served and unserved travel, and V2G removal use non-negative battery-side kWh. Signed home grid power is positive for import and negative for export.

For interval duration `dt_hours`:

```text
signed_grid_kw = (home_grid_import_kwh - grid_export_kwh) / dt_hours

closing_battery_kwh
  = opening_battery_kwh
  + home_battery_added_kwh
  + public_battery_added_kwh
  - served_travel_battery_kwh
  - v2g_battery_removed_kwh
```

The conservation residual is the recorded closing stock minus the right-hand side. It must be zero within the fixture tolerance. At every boundary, `0 <= battery_kwh <= physical_capacity_kwh`. Travel may cross an export reserve; reserve constrains export, not mobility. Infeasible requested travel becomes explicit unserved battery kWh and is never hidden by clipping time, teleporting the EV or drawing a new arrival SoC.

Preferred target, authorised target, BMS ceiling and physical capacity are distinct. Preferred target is a customer or policy target and does not discard existing energy. Charging stops at the lower of authorised target, BMS ceiling and 100% physical SoC. `full_reached` means physical 100%, not target attainment or tolerance.

## Minimum mechanically testable interval result

The final shared schema is pending, but every physical interval result must expose at least:

- identity: `population_id`, `world_id`, `path_id`, `unit_id`, `interval_start_utc`, `interval_end_utc`, and nullable `dispatch_route_id`;
- state and limits: opening and closing battery kWh, physical capacity kWh, preferred and authorised target SoC, BMS ceiling SoC, and reserve kWh;
- command and execution: `requested_grid_kw`, `realised_grid_kw`, `requested_import_kw`, `realised_import_kw`, `curtailed_import_kw`, `requested_export_kw`, `realised_export_kw`, `curtailed_export_kw`, and binding reason codes;
- directional flows: home and public grid-import kWh, grid-export kWh, home and public battery-added kWh, served and unserved travel battery kWh, and V2G battery-removed kWh;
- occupancy and connection: the four location fractions when every classification is resolved, otherwise the resolved fractions plus an explicit unclassified fraction, together with `connected_at_boundary` and `home_connected_full_slot`;
- audit values: opening and closing SoC, active charging hours, conservation residual kWh, and the physical route-exclusivity key.

Directional quantities must remain separate even when the signed net is zero. An inapplicable or unavailable value is not silently converted to zero.

## Physical helpers

### Home import and taper

For requested import, the untapered grid-side base is the non-negative minimum of request, the home charging power and any reviewed site allocation. The explicit models use one home charging power per EV (7 kW by default) with no separate vehicle AC acceptance limit and no site allocation (decision 0004 item 34). Physics, not the planner or UI, integrates the accepted power.

For `s = E/C`, taper start `s0` and minimum fraction `m`:

```text
f(s) = 1                                              when s <= s0
f(s) = max(m, 1 - (1-m)(s-s0)/(1-s0))               when s > s0
dE/dt = home_efficiency * untapered_grid_kw * f(E/C)
```

Integration stops at the applicable charge ceiling. Requested, realised and curtailed import remain separately observable. A site allocation must be non-negative and its sum must not exceed the declared site limit; invalid allocations are rejected, not redistributed by physics.

**Limitation (decision 0004 item 31).** The `f(s)` taper above is a tested isolated physical helper (see the "Exact taper" oracle below); the current explicit wholesale-action and no-action kernel does not call it. No charging taper is modelled in the live app: home import is not slowed near full SoC. State this plainly as a limitation in the app and explainers; do not describe or imply that tapering is an implemented behaviour of a run's result.

### Travel and endogenous public stops

Mobility consumes battery energy along the requested travel legs. When the evolving stock reaches a fixture-supplied public trigger and the charger is available, `serve` mode inserts an explicit non-overlapping stop, imports grid energy, adds battery energy, consumes elapsed time and then resumes travel. Repeated stops are permitted where the route and time window allow. The route request does not prescribe public energy; stops are stock-flow outputs.

The synthetic helper accepts explicit public power and efficiency inputs. The W30 public-stop fixture also supplies `vehicle_acceptance_power_kw=10` and `bms_ceiling_soc_percent=100`; both are explicit, non-binding fixture inputs and not defaults. The production composition and precedence of public efficiency, public charger power, vehicle acceptance and BMS ceiling are **pending**. Source-backed public assumptions and defaults are also pending. Dependent production mobility must stop at this boundary; the isolated fixture below may be implemented without treating its values as defaults.

### Export helper

The low-level export helper may be tested when consent, capability and all fixture gates are explicitly true. It requires `dt_hours > 0`, `0 < eta_d <= 1`, non-negative power limits and reserve, and an explicit non-negative `available_battery_kwh`. For grid-side power limits in kW:

```text
requested_export_kwh = max(0, -requested_grid_kw) * dt_hours
evse_export_limit_kwh = evse_export_limit_kw * dt_hours
vehicle_export_limit_kwh = vehicle_export_limit_kw * dt_hours
site_export_limit_kwh = site_export_allocation_kw * dt_hours
reserve_limited_grid_export_kwh
  = max(0, available_battery_kwh - reserve_kwh) * eta_d

grid_export_kwh = min(
  requested_export_kwh,
  evse_export_limit_kwh,
  vehicle_export_limit_kwh,
  site_export_limit_kwh,
  reserve_limited_grid_export_kwh,
)
v2g_battery_removed_kwh = grid_export_kwh / eta_d
post_export_battery_kwh
  = available_battery_kwh - v2g_battery_removed_kwh
realised_grid_kw = -grid_export_kwh / dt_hours
```

`available_battery_kwh` is supplied directly as the battery stock at the export event; the helper never substitutes interval-opening stock or infers ordering. Any accepted caller must first apply every earlier event in its separately approved interval ordering, then pass the resulting stock once. No caller may compose export with mobility, public charging or home import until that ordering is accepted. In the isolated W30 export fixture there are no earlier flows, so `available_battery_kwh = opening_battery_kwh = 40 kWh`. The helper applies export once and returns the post-export stock; later flows, if eventually authorised, start from that returned stock.

Route-integrated negative commands remain disabled. Released-route eligibility, recovery-debt units and ownership, repayment sources, deadline, success test and terminal treatment are pending finance and policy decisions. The helper does not authorise fleet V2G, settlement, recovery or revenue.

## One dispatchable route per physical path-slot

For each `(world_id, path_id, unit_id, interval_start_utc)`, at most one dispatchable physical route may own the EV's available flexibility. [Decision 0004 item 18](decisions/0004-explicit-action-and-horizon-choices.md) supersedes decision 0001's requirement that the dashboard cover all six commercial mechanisms (wholesale, DFS, frequency response, local flexibility, Capacity Market and supplier optimisation): only the explicit wholesale action model and the explicit no-action model remain, so the wholesale action's smart charging plan is currently the only dispatchable route the model implements. DFS, frequency response, local flexibility and Capacity Market are "not modelled" (shown as such in How it works ▸ Limits) and may return later as explicit-path candidates; the historical six-mechanism dispatchable set is recorded in the Superseded appendix. Normal and selected evaluation paths validate ownership, command reconciliation, realised power and closing stock independently. They are mutually exclusive counterfactual trajectories, never simultaneous deliverable capacity. Within a path, ordinary charging and the selected route reconcile into one realised signed grid-power value and one closing battery stock; the record is not split into simultaneous route-owned kW slices.

Two route claims on one EV-slot fail validation even when their requested kW sum is below the device limit. Different EVs may serve different routes in the same interval, and one EV may change route across non-overlapping intervals. Prior obligations reserve their EV-slots before later offers, but the detailed precedence, fallback and release policy is pending.

Capacity Market accrual, supplier benefit or fees and other non-dispatch accounting do not become physical owners. Supplier optimisation cannot issue an independent physical command. If Capacity Market actually dispatches the EV-slot, that stress dispatch is the sole physical route owner. These statements define physical lineage only; commercial route, obligation, settlement and cash schemas remain pending.

## Explicit wholesale action model

[Decision 0004](decisions/0004-explicit-action-and-horizon-choices.md) accepts the following rules for the two explicit models built on the foundation above. The explicit wholesale action model is the app default; the explicit no-action model stays selectable (item 9). "Wholesale action" here names the mechanism this contract governs, distinct from "Smart charging", the UI page and model-control label the dashboard shows for it (goal review section 8 item 6; `docs/DASHBOARD_DESIGN.md`).

- **Smart charging plan (item 38, superseding items 2, 8, 25 and 30).** The selected action is price-optimised smart home charging, not an authorised import ceiling or a fixed action window. At each home connection, knowing only the EV's stock at that moment, its preferred target, the cohort's typical departure clock time minus a safety margin (`departure_margin_hours`, editable, default two hours, item 51) and its own simulated week's day-ahead price path, treated as published at the study start (item 48), the charger fills the cheapest day-ahead half-hours up to home charging power. It limits and times home import and never forces it; it can only move charging to a cheaper half-hour, never raise it above the normal path's rule. If the EV leaves before its plan finishes, the missed energy is simply not delivered and is priced as a shortfall (items 4, 5, 13). A session whose next expected departure falls after the study end cannot see its full plan window, so it charges as soon as it is plugged in, like the normal path, rather than crowding into the last half-hours before the cut (item 49). There is no fleet-wide or per-EV eligibility screen and no separate planning world: every EV plans and executes its own session from information available at plug-in only.
- **Stochastic day-ahead prices (item 48, which retires item 47's fleet-wide charging cap).** Each simulated week's day-ahead price path runs that world's own net demand (a demand shape, heating, wind, solar and any shock) through an illustrative merit-order supply curve, plus a persistent daily level and half-hour AR(1) noise (items 53, 55, 56), so the cheapest half-hour moves from week to week instead of sitting at one shared trough. The intraday close adds hourly updates to gate closure and surprise-shock impact, and is the realised price the cost effect is valued at. Within one simulated week every EV still ranks the same day-ahead path and there is no site or network limit, so smart charging can still bunch into that week's own cheapest half-hours; this stays labelled as a finding, and no fleet-wide charging cap is modelled.
- **Threshold public top-ups (item 32).** When a trip would take an EV's battery below the top-up threshold (10% SoC by default, editable, lowered from 20% by item 37), the EV tops up at an always-available public charger to the top-up target (80% by default) at the moment it reaches the threshold, as many times as the trip needs, then continues. Top-up energy is public grid import in that slot (battery energy divided by public efficiency); the time spent charging is not modelled. There is no separate destination charging. Both models use the same rule.
- **Pricing (item 4).** Public charging is priced at an illustrative editable assumption, £0.79/kWh, not a tariff claim. Energy not recovered by the end of the horizon is valued at the same illustrative public rate, on the basis the driver would buy it back at a public charger. Home-import cost, public-charge cost, unrecovered-energy value and their illustrative total are shown separately; none is a settlement, bid or Axle-cash amount (decision 0003).
- **Not-recovered flag (item 5).** `energy_not_recovered` trips when the selected path ends with lower battery stock, more public import or more unserved travel than the normal path, beyond a 1e-6 kWh tolerance.
- **Unserved travel (items 13, 32).** EVs never strand, so unserved travel is zero by construction and stays in results as a check. It still trips the not-recovered flag and is valued at the same illustrative public rate and the same 1e-6 kWh tolerance, because unpriced unserved travel would otherwise show a false saving. Battery-side energy is valued at the grid-side public rate as a stated simplification. No charging taper is modelled (see "Home import and taper" above); this is a stated limitation, not an implemented behaviour.
- **Settled start (item 36).** Runs open at the preferred target (80% SoC) with a warm-up (seven days by default, editable 3–14, item 52) that is simulated but not reported, and the action never applies in it, so the reported week is not dominated by start-up refill.
- **Paired difference (item 12).** The model outputs selected-minus-normal P10/P50/P90 bands computed from per-world differences (world-first, per the section above); the UI never subtracts percentiles.

## World-first aggregation

Preserve world alignment. Sum each additive EV and cohort quantity inside each complete world before calculating quantiles or other distribution summaries across worlds. Never sum cohort percentiles, route percentiles or half-hour percentiles to construct a portfolio percentile.

Shared exogenous series such as price and weather are copied or referenced by cohorts, not summed. Fleet SoC is total battery stock divided by total physical capacity, not an unweighted average of unit SoC. Coarsen power within each world by time-weighted average, energy by sum and stock by the closing value before taking cross-world summaries. Every physical and commercial aggregation also preserves `path_id`; an approved paired comparison may align normal and selected outcomes within a world and calculate a difference, but must never sum the paths as deliverable capacity.

## Synthetic physical oracles

Unless a row says otherwise, arithmetic assertions use absolute tolerance `1e-12`. These fixtures are test-only.

| Oracle | Inputs | Required result |
| --- | --- | --- |
| UTC counts | Safe Europe/London local midnight; `1`, `3`, `7` days | `T = 48`, `144`, `336`; `T+1` battery boundaries; every UTC interval is 1,800 seconds. Reject spring or autumn offset changes anywhere in warm-up plus study (the removed legacy modes' rule; the two explicit models instead follow decision 0004 item 1, see "Time and boundary rules"). |
| Mid-slot home return | Interval `[10:00,10:30)`; homebound return ends `10:20` after 20 minutes driving | `driving_fraction=2/3`, `home_unplugged_fraction=1/3`, `home_connected_fraction=0`, `home_connected_full_slot=false`, home energy `0`; first potentially home-connected interval starts `10:30`. |
| Boundary departure | Departure exactly `10:00`; stale request `+7 kW` for `[10:00,10:30)` | Realised home power and energy are `0`; binding reason includes departure or disconnected. |
| Exact taper | `C=10 kWh`, `E0=9 kWh`, untapered `P0=2 kW`, efficiency `1`, `s0=.9`, `m=.5`, `dt=.5 h`; target and BMS above result | `E1=9.786938680574734 kWh`; grid import `.786938680574734 kWh`; average power `1.57387736114947 kW`. |
| Public stop | `serve`; `C=10`, `E0=1.5`; trigger `.1`, target `.2`; drive `.5` battery-kWh, add `1` battery-kWh using `1.25` grid-kWh at `10 kW`, efficiency `.8`, then drive `1`; public power and vehicle acceptance both `10 kW`; BMS ceiling `100%`; charger available and deadline slack sufficient | Vehicle acceptance and BMS ceiling are non-binding. Stop duration `.125 h`; closing stock `1 kWh`; unserved travel `0`. |
| Export helper only | All gates true; `C=60`, explicit `available_battery_kwh=E0=40`, request `-6 kW`, `dt=.5 h`, discharge efficiency `.9`, reserve `12 kWh`; other limits non-binding; no earlier or later flows | Export `3` grid-kWh; remove `3.333333333333333` battery-kWh; close `36.666666666666664 kWh`; realised power `-6 kW`. The reserve cap is `25.2` grid-kWh and is non-binding. |
| Target and BMS | Opening `85%`, preferred target `80%` | Ordinary import is `0`. With authorised target `100%`, BMS ceiling `95%`, sufficient power and time, charging stops at `95%`; `full_reached=false`. |
| Site allocation | Two `7 kW` requests; reviewed allocation `[3,2] kW`; site cap `5 kW`; no taper, efficiency `1`, `dt=.5 h` | Realised power `[3,2] kW`; energy `[1.5,1] kWh`; total `2.5 kWh`. An allocation `[7,7]` is rejected. |
| Stock-derived arrival | Prior closing stock `30 kWh`; capacity `60 kWh` | Next applicable arrival SoC is `.5`, with no arrival-SoC draw. |
| Persistent identity | Planning world `0` and evaluation world `0` share one population | `population_id` and serialised trait arrays are identical; planning and evaluation future samples remain separate. |
| Route exclusivity | Two dispatchable route claims for one EV/world/path/slot | Validation fails. Distinct EVs in that slot, the same EV in distinct slots, and alternative owners on mutually exclusive normal and selected paths are valid; paths are never aggregated as deliverable capacity. |
| World-first P95 | Cohort A worlds `[0,100]`; aligned cohort B worlds `[100,0]`; linear quantile | World totals are `[100,100]`; fleet P95 is `100`, not the invalid marginal-percentile sum `190`. |

## Pending contracts and stop gates

The following are not approved by this document:

- same-boundary arrival plus departure or unplug precedence;
- production public efficiency, charger, vehicle-acceptance and BMS composition;
- route-integrated V2G and every recovery rule;
- commercial route eligibility, reservations, obligations, delivery attribution, baselines, rates, penalties and fallback;
- finance, stakeholder cash, accrual, inventory value and settlement;
- planner scoring, allocation, tie-breaks, information payloads, policy serialisation and policy hashes;
- source-backed sampling until behavioural distributions, source-to-field provenance and parameter values are reviewed;
- behavioural distributions, source-backed cohort values, approved defaults, source snapshots and the release configuration;
- final function signatures and full configuration, result, replay, comparison, export and UI schemas.

Implementation may proceed for the accepted synthetic foundation surface with its applicable physical and configuration checks and matching fixtures. The mid-slot rule applies only to a journey explicitly defined to return home; an unknown destination remains unclassified. Stop and return the unresolved dependency to the lead if a slice needs any pending choice, an absent approved source, or a schema beyond the minimum fields above. Passing these oracles proves arithmetic, causality and internal conservation only. It does not prove empirical accuracy, operational reliability, market eligibility, settlement readiness, profitability, hosted capacity or release readiness.

## Superseded

Historical record only; do not implement from this section without a new product decision.

- **Automatic bounded draft preview.** Decision 0002's preview design (a small, cancellable, clearly labelled "Preview, not a full run" refresh after a committed draft edit) is superseded by Mike on 26 September 2026. The active app is Run-only: a draft edit updates only the draft, and a fresh session has no sampled result until an explicit Run click. See [decision 0002](decisions/0002-automatic-draft-preview.md).
- **Six commercial mechanisms and the cohort/compact/synthetic six-route models.** Decision 0001's requirement that the dashboard cover all six commercial mechanisms (wholesale, DFS, frequency response, local flexibility, Capacity Market and supplier optimisation), with an accepted dispatchable set of "wholesale, DFS, frequency response, local flexibility and an actual Capacity Market stress dispatch", is superseded by [decision 0004 item 18](decisions/0004-explicit-action-and-horizon-choices.md): the cohort, compact (including compact DFS) and synthetic six-route one-day models are removed with their views, tests and configs, because they use connection-clock-as-travel physics that an earlier review (M06) rejected. This also supersedes M09's daily synthetic DFS case and U04's "one-day models remain selectable references". Only the explicit wholesale action model (default) and explicit no-action model remain; other mechanisms are shown as "not modelled" and may return later as explicit-path candidates.
- **Blanket offset-change rejection.** The original rule — reject the complete warm-up plus study span if the UTC offset changes anywhere in it — is superseded for the explicit models by decision 0004 item 1 (see "Time and boundary rules" above) and survives only as the historical rule the removed legacy modes applied.
- **JSON configuration files.** `docs/contracts/config-v1.md`, `docs/contracts/illustrative-market-environment-v1.md` and their loaders are superseded by decision 0004 item 21: `model/assumptions.py` holds every assumption value, unit, evidence kind, source and meaning, and the explicit JSON configs are removed.
- **Vehicle AC acceptance limit in home import.** Home import was the minimum of charger limit and vehicle AC acceptance limit. Superseded by decision 0004 item 34: one home charging power per EV.
- **Destination public charging only when the next trip is infeasible (item 3).** An EV used destination public charging only when its return leg would otherwise fail, with no SoC threshold, subject to a sampled charger availability. Superseded by decision 0004 item 32: threshold top-ups at an always-available public charger; the availability draw is removed.
- **Stranding and stranded EVs' later trips (item 31).** A leg the battery could not serve became unserved travel, the EV was stranded, and every later trip in the horizon counted as unserved travel. Superseded by decision 0004 item 32: EVs never strand.
- **Opening SoC 55% with one warm-up day.** Superseded by decision 0004 item 36: 80% opening SoC and three warm-up days.
- **0 kW authorised import ceiling in a fixed action window, with a per-EV eligibility screen and a planning world (items 2, 8, 25, 30).** The selected action capped home import at 0 kW in the 18:00–18:30 London half-hour of each study day (`defer_daily_1800_v1`); a screen run in a separate planning world marked an EV-day **eligible** only if the deferral shifted its import, added no public charging or unserved travel, and ended the London date with no less stock than normal, and the ceiling applied only to eligible EV-days. Superseded by decision 0004 item 38: price-optimised smart home charging, which every EV plans and executes for itself from information available at plug-in, with no fleet-wide or per-EV screen and no separate planning world (see "Explicit wholesale action model" above).
