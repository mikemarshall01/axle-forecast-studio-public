# Decision 0001: one dispatch route per EV-slot

Status: accepted product/modelling choice from Mike, 26 September 2026. Implementation and the detailed route/finance schema remain pending Wave 0 review.

## Rule

For each simulated EV, world and half-open 30-minute UTC interval, at most one **dispatchable physical route** owns that EV's available flexibility. A different route may use the EV in a later interval. The model reconciles the selected route and ordinary charging/travel constraints into exactly one realised grid-power value and one continuing battery-energy state for that EV-slot.

Clarification accepted 26 September 2026: where a simulation has mutually exclusive counterfactual physical paths, the EV-slot key is `(world_id, path_id, ev_id, interval_start_utc)`. Normal and selected paths are alternative physical trajectories, not simultaneous capacity. Each path independently permits zero or one whole-device route owner, and their ownership, power, energy, obligations, delivery, route mix and cash must never be summed, stacked or jointly settled. Different owners or a null owner on one path do not constitute two claims on one physical trajectory. This does not permit two routes to divide the same EV-slot within either path. The timing and propagation of obligations fixed before the paths diverge require a separate approved policy rule.

Dispatchable routes include wholesale shifting, DFS, frequency response and local flexibility when they seek physical power from the EV. Prior committed obligations and availability reservations consume the relevant EV-slots before a new offer can use them. A route cannot sell the same EV-slot in the opposite direction merely because an aggregate up/down MW total appears to fit. Feasibility is checked at the EV and battery level: connection, charger/vehicle/site power, SoC, reserve, travel, customer target, later recovery and obligation timing.

Capacity Market commitments, supplier benefits/fees, customer payments and other non-dispatch accounting may appear in their separate ledgers without becoming a second physical power owner. Supplier-facing optimisation cannot issue an independent physical command. Any associated load change must be owned by ordinary charging or exactly one dispatchable physical route; supplier benefit and fees reference that same physical lineage only as accounting overlays. If a Capacity Market stress obligation actually dispatches an EV-slot, its prior obligation has precedence and excludes a conflicting route claim. The finance contract will specify payment parties, accrual and cash separately.

The dashboard still covers all six commercial mechanisms across different EVs and intervals. Its comparable dispatchable route-mix denominator contains only wholesale, DFS, frequency and local reservations; Capacity Market and supplier value stay in separate ledgers. This conservative whole-device-slot rule is stricter than the earlier UI blueprint's minimum “each reserved MW at most one route” rule, so it satisfies that no-double-sale requirement without modelling simultaneous disjoint kW slices from one EV. It does not remove any mechanism from the fleet-level dashboard.

## Why

Subtracting reserved MW from an aggregate fleet total prevents an obvious numerical duplicate, but it cannot by itself prove that the same underlying EV can satisfy two instructions with one battery, its actual connection/departure times and recovery obligations. Whole-device-slot exclusivity is simpler to implement, explain, test and audit for the interview release.

## Required checks

1. Two dispatch routes claiming the same `(world_id, path_id, ev_id, interval_start_utc)` fail validation, even if their requested kW sum is below the charger's rating. Different owners on mutually exclusive normal and selected paths are permitted, subject to separately approved pre-fork obligation rules.
2. Distinct EVs can serve different routes simultaneously; one EV can change route across non-overlapping slots.
3. A prior obligation reserves its EV-slot before later offers. An uncleared offer has `obligationMw = 0`, creates no delivery obligation, penalty, settlement or route-mix numerator, and releases capacity only through the predeclared fallback rule.
4. Each EV-slot has one realised grid-power record and one battery closing stock; battery balance, reserve and customer-travel/target constraints hold.
5. Supplier/CM analytical value, Axle cash and stakeholder payments do not duplicate physical delivery or each other's accounting.
6. Sum physical energy and cash inside each world before fleet distributions; do not sum route percentiles.

No model code, route schema, cash contract or hosted result is approved by this decision alone. The preserved JavaScript repository remains a migration reference and is not deleted.
