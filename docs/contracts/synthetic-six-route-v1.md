---
status: accepted-fixture-contract
ArchitectureReviewRequired: yes
scope: illustrative-interview-fixture-only
---

# Small six-route synthetic fixture and ledger seam

> **Superseded by decision 0004 item 18/21 (28 Sep 2026).** The synthetic six-route one-day model is removed with its views, tests and configs. Only the explicit wholesale action model (default) and explicit no-action model remain. Kept for history; do not implement from this document.

This proposed contract is for the first local interview journey, not service eligibility, bidding, settlement, source-backed prices or deployed defaults. It becomes executable only after the illustrative runner's final physical-result review passes and this interface receives an independent early review. The accepted physical contract and decisions 0001/0003 continue to govern.

## Physical policy and counterfactual meaning

`run_illustrative_day(settings)` keeps its current no-action behaviour. An optional `synthetic_route_fixture` supplies fixed, decision-time-known event targets for the **selected** path only. The normal path is ordinary charging with the same sampled exogenous inputs and its own continuing stock. Both paths have one physical state per EV/UTC half-hour. A normal/selected difference is a comparison, never simultaneous energy or capacity.

For this fixture, there are **no commitments made before the normal/selected fork**. A synthetic Capacity Market stress request, if enabled, is a selected-policy event whose trigger is known at its slot boundary; normal is explicitly an un-obligated comparison path. This does not decide how a real pre-existing capacity obligation propagates across counterfactual paths. That remains unavailable outside this fixture. Supplier optimisation and non-dispatch Capacity Market accrual never own a physical slot.

`build_synthetic_route_fixture(settings: RunSettings) -> tuple[dict[str, object], ...]` requires at least two EVs, one study day and the runner's 18:00 local return. It creates this fixed, world-agnostic selected-path schedule on the study date; local 18:00 plus the slot offset is converted to UTC. EV IDs are `ev-001` and `ev-002`. These are targets, not promised deliveries:

| Half-hour slot after 18:00 local | Event / owner | Direction | Selected target kW | Per-unit requested response kW |
| --- | --- | --- | ---: | ---: |
| 0 | `wholesale-early`, `ev-001` | `reduce_import` | 0 | 4 |
| 1 | `wholesale-recovery`, `ev-001` | `increase_import` | 4 | 4 |
| 2 | `dfs-event`, `ev-001` | `reduce_import` | 1 | 4 |
| 3 | `frequency-event`, `ev-001` | `reduce_import` | 3 | 1.5 |
| 4 | `local-event`, `ev-001` and `ev-002` | `reduce_import` | 2 and 3 respectively | 2 each, 4 event total |
| 5 | `capacity-stress`, `ev-001` | `reduce_import` | 0 | 4 |

Use one record per selected EV-slot with these exact fields:

| Field | Type and meaning |
| --- | --- |
| `event_id` | non-empty stable string for this synthetic event |
| `route_id` | `wholesale`, `dfs`, `frequency_response`, `local_flexibility` or an actual `capacity_market` stress event |
| `unit_id` | existing EV identity in the runner population |
| `interval_start_utc` | aware UTC half-hour boundary |
| `decision_available_at_utc` | aware UTC time no later than that interval start |
| `direction` | either direction for `wholesale`/`dfs`; only `reduce_import` for `frequency_response`, `local_flexibility` and `capacity_market` |
| `selected_import_target_kw` | finite non-negative grid-side interval-average kW command, before physical/site constraints |
| `requested_response_kw` | finite non-negative grid-side *per-unit allocated* response magnitude; multi-unit rows sum to the event request |
| `is_stress_dispatch` | `true` only for this selected-path Capacity Market stress request; a fixture-local physical-lineage flag, not evidence of a real agreement or obligation |

`decision_available_at_utc` equals the slot start for this fixture. `is_stress_dispatch` is false for every non-CM record and true for `capacity-stress`. Do not extend `RouteRequest` with direction: pass its existing `requested_grid_kw=requested_response_kw` for whole-device owner validation and carry the target separately. Reject duplicate `(unit_id, interval_start_utc)` fixture records before expanding across worlds, then enforce the complete `(world_id, path_id, unit_id, interval_start_utc)` ownership key in every world. Never let a dictionary overwrite hide competing claims. Pair only normal and selected records with the same `(world_id, unit_id, interval_start_utc)` from the same population and run. Validate decision time before issuing a command. Apply selected targets before site allocation, and recompute site allocation and physical stock independently on each path. Physics may curtail or reject a request; the schedule is not proof of delivery.

## Ledgers and money

`derive_synthetic_route_ledgers(intervals: pd.DataFrame, fixture: Sequence[Mapping[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]` returns the delivery ledger and separate monetary-component table, in that order. It runs only after paired physical intervals exist. Each physical event gets one row per evaluation world and fixture EV-slot. Supplier optimisation gets one analytical row per world and no physical owner. Use these exact delivery columns:

| Column | Meaning |
| --- | --- |
| `world_id`, `path_id`, `comparison_path_id` | Evaluation world integer, `"selected"`, `"normal"` |
| `event_id`, `route_id`, `unit_id` | Fixture event/route/EV; supplier uses `event_id="supplier-window"`, `unit_id=None` |
| `interval_start_utc`, `interval_end_utc` | Aware UTC event slot; supplier uses first event start and last event end |
| `role`, `applicable`, `status`, `reason` | `physical_owner` or `analytical_overlay`; `applicable=True` means fixture-scheduled, **not** commercially eligible. Status is `delivered`, `partial`, `not_delivered` or `analytical`. Reason is `None` for delivered/analytical, otherwise the physical limit or zero-response explanation. |
| `direction`, `selected_import_target_kw`, `reserved_kw` | Target and direction; `reserved_kw=requested_response_kw` is a fixture-local slot reservation, not a market commitment. All `None` for supplier. |
| `normal_realised_import_kw`, `selected_realised_import_kw` | Paired same-world grid-side interval-average kW; both `None` for supplier. |
| `requested_response_kw`, `realised_response_kw` | Additive per-unit request and capped physical delivery, grid-side kW; both `None` for supplier. |
| `response_energy_kwh`, `shortfall_energy_kwh` | Grid-side kWh over actual slot duration; both `None` for supplier. |
| `decision_available_at_utc`, `outcome_available_at_utc` | Decision no later than slot start; physical outcome available at `interval_end_utc`, supplier at window end. |
| `evidence_kind` | `synthetic` |

For `reduce_import`, `paired_physical_response_kw = max(normal_realised_import_kw - selected_realised_import_kw, 0)`; for `increase_import`, reverse the subtraction. Set `realised_response_kw = min(requested_response_kw, paired_physical_response_kw)`, `response_energy_kwh = realised_response_kw * interval_hours`, and `shortfall_energy_kwh = max(requested_response_kw - realised_response_kw, 0) * interval_hours`. The paired imports already expose any uncapped physical difference. A positive request with zero delivery is `not_delivered`, not unavailable; partial delivery is `partial`. A missing pair is an error, not invented zero. No unscheduled physical row is fabricated to fill a chart. All power is grid-side interval-average kW, and response/shortfall energy is grid-side kWh. Five physical routes plus the supplier overlay make all six identifiers visible.

The separate monetary table has exact columns `world_id`, `route_id`, `event_id`, `component_kind`, `value_gbp`, `evidence_kind`, `accounting_status`, `payer`, `payee`. This fixture **includes** exactly three synthetic components. Wholesale and supplier use each world's paired realised imports; CM uses the fixed analytical accrual inputs below:

| Fixture input | `component_kind` | `accounting_status` | Sign meaning |
| --- | --- | --- | --- |
| Wholesale event prices `[0.20, 0.10] GBP/kWh` over its two slots | `forecast_wholesale_charging_cost_saving` | `forecast_estimate` | Positive means lower forecast charging cost, not trade or settlement cash |
| CM `derated_kw=4`, `rate=25 GBP/kW/year`, `overlap_seconds=1800`, `delivery_year_seconds=31536000` | `illustrative_capacity_accrual` | `analytical_non_cash` | Straight-line fixture apportionment, unrelated to stress payment, penalty or recognised accounting accrual |
| Supplier prices `[0.20, 0.10, 0.15, 0.12, 0.11, 0.09] GBP/kWh` over the six event slots | `selected_minus_normal_internal_energy_cost_delta` | `internal_cost_comparison` | Positive means selected costs more; not supplier benefit or Axle cash |

The two price vectors are synthetic forecast inputs available at 17:30 local on the study date; neither is an observed settlement price. Component `event_id` values are `wholesale-window`, `capacity-stress` and `supplier-window` respectively. All three have `evidence_kind="synthetic"`, `payer=None` and `payee=None`. Wholesale uses its two paired EV-slot imports; supplier uses same-world fleet home imports across six event slots; CM is analytical and claims no stress cash. Keep components separate, with no Axle/portfolio cash total. All other unsupported commercial terms remain unavailable, not zero.

## Test meaning

Keep hand-checkable arithmetic oracles separate from the continuous run. The two-slot wholesale oracle is `C=10 kWh`, opening stock `6 kWh`, target `8 kWh`, efficiency `1`, charger/vehicle/site allocation `4 kW`: normal import `[4, 0] kW`, selected `[0, 4] kW`, both close at `8 kWh`, and prices `[0.20, 0.10] GBP/kWh` yield `GBP 0.20` forecast saving. These are **independent primitive oracles**, not successive physical slots on the same EV or exact expectations for the continuous run:

| Primitive | Standalone inputs | Expected output |
| --- | --- | --- |
| DFS | synthetic reference `4 kW`, actual `1 kW`, cleared `4 kW`, reduction, `0.5 h` | delivered `1.5 kWh`, shortfall `0.5 kWh` |
| Frequency response | contract `2 kW`, available `1.5 kW`, physical limit `1 kW`, triggered, `0.5 h` | requested `1.5 kW`, realised `1 kW`, energy `0.5 kWh` |
| Local flexibility | request `4 kW`, available `[2, 3] kW`, physical limit `[2, 1] kW`, both region-matched, `0.5 h` | reserved `4 kW`, realised `3 kW`, shortfall `0.5 kWh` |
| CM analytical accrual | `4 kW`, `25 GBP/kW/year`, `1800 s`, `31536000 s` | about `GBP 0.0057077626`, non-cash |
| Supplier overlay | normal `[2, 0, 2, 2, 4, 2] kWh`, selected `[0, 2, 0.5, 1.5, 2.5, 0] kWh`, prices `[.20, .10, .15, .12, .11, .09] GBP/kWh` | selected-minus-normal cost `-GBP 0.83`, internal synthetic comparison |

The integrated synthetic run instead checks continuous stock, conservation, site/device limits, one owner per EV/path/slot, paired exogenous inputs, causal decision-time availability and world-first aggregation. Every scheduled physical row must have `delivered`, `partial` or `not_delivered` status; the supplier overlay uses `analytical`. Zero response to a positive request is `not_delivered`. Exact delivery depends on realised stock and connection; do not force the standalone primitive numbers into one continuous path. All six route identifiers appear. No real eligibility, payment, penalty or aggregate cash follows from passing these tests.

## Parallel write boundary after review

The runner writer owns only `model/runner.py` and its test: `run_illustrative_day(settings, *, synthetic_route_fixture=None)` applies optional selected targets and preserves the no-action default. A route-ledger writer owns a new `model/synthetic_six_route.py` and its test: the two functions and exact tables above, fixed fixture, primitive calls, physical-response/shortfall ledger and separately labelled monetary components. Neither writer edits the other's files. The lead integrates the two behind this contract and adds the small app-level call after both pass independent task review.
