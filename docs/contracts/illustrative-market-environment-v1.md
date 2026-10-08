---
status: proposed-illustrative-fixture-contract
ArchitectureReviewRequired: yes
scope: six-slot-synthetic-market-context-only
---

# Illustrative market environment v1

> **Superseded by decision 0004 item 18/21 (28 Sep 2026).** This fixture supported the synthetic six-route one-day model, which is removed. Only the explicit wholesale action model (default) and explicit no-action model remain. Kept for history; do not implement from this document.

## Authority and limits

This contract freezes the smallest market-context input for the accepted one-day synthetic six-route fixture. It adds six half-hour slots aligned with the fixture's Europe/London 18:00–21:00 window. Every value is synthetic and clearly invented for the local illustrative scenario. The values are not observed, calibrated, traded, cleared or settled market data.

The accepted physical and commercial contracts remain unchanged. This fixture adds no bids, offers, clearing, instructions, eligibility, baselines, settlement, stakeholder cash or real-market claim. Forecast prices support analytical valuation only. Realised balancing-context prices are context, not settlement prices. Temperature and system need are display-only and do not affect travel, route requests, route selection or physical flow. Sampled balancing context also does not generate or select a route.

The source is `configs/illustrative-market-environment.json`. Its `source_note` is the provenance statement for every value, and every emitted row has `evidence_kind="synthetic"`.

## Slot and UTC derivation

The configuration stores six aligned arrays rather than dated rows. For scenario date `D = settings.start_local_date`, construct an aware local anchor at `D 18:00 Europe/London`. For zero-based slot `i` from 0 to 5:

```text
local_start[i] = D 18:00 Europe/London + i * 30 minutes
interval_start_utc[i] = local_start[i] converted to UTC
interval_end_utc[i] = interval_start_utc[i] + 30 minutes
forecast_available_at_utc[i] = D 17:30 Europe/London converted to UTC
outcome_available_at_utc[i] = interval_end_utc[i]
```

No calendar date is baked into this fixture. Use timezone-aware conversion, not a fixed UTC offset. The runner's existing guard against a Europe/London offset change over the warm-up and study span remains authoritative. Each forecast availability time is no later than its decision boundary. A missing or duplicate exact UTC join is an error; implementations must not forward-fill, back-fill or substitute zero.

## Configuration shape and values

The JSON object has these direct fields:

| Field | Required value or shape |
| --- | --- |
| `schema_version` | `illustrative-market-environment-v1` |
| `evidence_kind` | `synthetic` |
| `source_note` | Non-empty synthetic provenance statement |
| `slot_definition` | `timezone="Europe/London"`, `local_start_time="18:00"`, `interval_minutes=30`, `slot_count=6`, `forecast_available_local_time="17:30"` |
| `wholesale_forecast_gbp_per_mwh` | Six finite values: `[200, 100, 150, 120, 110, 90]` |
| `balancing_context_price_gbp_per_mwh_base` | Six finite invented base values |
| `balancing_context_price_gbp_per_mwh_sd` | Six finite non-negative normal-draw standard deviations |
| `temperature_c_base` | Six finite invented base values |
| `temperature_c_sd` | Six finite non-negative normal-draw standard deviations |
| `system_need_mw_base` | Six finite invented base values |
| `system_need_mw_sd` | Six finite non-negative normal-draw standard deviations |
| `system_need_sign_convention` | Positive means the system is short and needs net upward system response; negative means the system is long |

All seven numeric arrays have exactly six positions in common slot order. The balancing-context price, temperature and system-need bases and standard deviations are illustrative parameters, not estimates or calibration. Normal draws are retained as drawn; this contract adds no clipping or market-behaviour rule.

## Sampling and world ownership

The model creates one `np.random.default_rng(seed)` per run and passes that generator directly to sampling. Preserve the runner's existing draws, then sample the planning market worlds and evaluation market worlds as two consecutive, disjoint calls. Within each call, draw arrays in this order:

```python
balancing_context_price = rng.normal(price_base, price_sd, size=(world_count, 6))
temperature = rng.normal(temperature_base, temperature_sd, size=(world_count, 6))
system_need = rng.normal(system_need_base, system_need_sd, size=(world_count, 6))
```

Do not create another generator, reseed, hash semantic keys or add a channel registry. `world_id` is the zero-based row index within its planning or evaluation table. Planning and evaluation draws are distinct. Each evaluation row is pathless and is shared by the normal and selected counterfactual paths in the same world; it must not be copied and summed as if paths were separate capacity.

The deterministic wholesale forecast is common to all worlds. It is available before every decision boundary and may be used only for the accepted forecast analytical valuation. Sampled outcomes become available at their interval end and cannot inform the decision at that interval start.

> **Further superseded by decision 0004 item 48.** The live model gives each simulated week its own stochastic day-ahead price path (the daily cosine plus that world's own AR(1) noise) rather than one deterministic forecast shared by every world; a second, independent AR(1) forecast error then gives the realised price. See `docs/contracts/results-v2.md`.

## Model-facing tables

`market_forecast` has exactly these fields:

| Field | Meaning |
| --- | --- |
| `interval_start_utc`, `interval_end_utc` | Aware UTC half-hour keys derived above |
| `forecast_available_at_utc` | Aware UTC publication time derived from local 17:30 |
| `wholesale_forecast_gbp_per_mwh` | Deterministic decision-time forecast price; divide by 1,000 only when an accepted calculation requires GBP/kWh |
| `evidence_kind` | `synthetic` |

`planning_market_worlds` and `evaluation_market_worlds` each have exactly these fields:

| Field | Meaning |
| --- | --- |
| `world_id` | Zero-based planning- or evaluation-world identifier |
| `interval_start_utc`, `interval_end_utc` | Exact aware UTC half-hour keys |
| `balancing_context_price_gbp_per_mwh` | Sampled synthetic context price; not settlement |
| `temperature_c` | Sampled display-only air-temperature context |
| `system_need_mw` | Sampled display-only context using the configured sign convention |
| `outcome_available_at_utc` | `interval_end_utc`; unavailable to the same-slot decision |
| `evidence_kind` | `synthetic` |

Keys are unique in `market_forecast` by `interval_start_utc`, and in each context table by `(world_id, interval_start_utc)`. Every table contains all six slots. Non-finite values, incorrect array lengths, negative standard deviations, non-UTC emitted timestamps, unexpected evidence labels and missing UTC joins fail validation.

## UI meaning and unavailable claims

In Markets, **Day-ahead** means forecast price context, not a cleared trade. **Balancing** means synthetic context, not an instruction, dispatch, settlement or realised market outcome. **Bids & offers** remains unavailable. **Spreads** remains unavailable because this contract defines no spread calculation; a later contract may expose one only from explicitly aligned forecast and context values with a synthetic label.

No chart or table may relabel these values as actual wholesale prices, balancing settlement data, commercial performance or cash. All unsupported market and commercial fields remain unavailable rather than zero.
