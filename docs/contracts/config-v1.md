---
status: revised-proposal-pending-review
contract: axle.scenario-config.v1
ArchitectureReviewRequired: yes
---

# Scenario configuration contract v1

> **Superseded by decision 0004 item 18/21 (28 Sep 2026).** The explicit JSON configs and their loaders are removed; `model/assumptions.py` now holds every assumption's value, unit, evidence kind, source and meaning. Kept for history; do not implement from this document.

## 1. Authority and scope

This document previously froze a synthetic input boundary. Its RNG controls have been revised for the interview prototype, but the wider configuration shape is under simplicity review and must not be implemented from this revision until accepted. It is not evidence that the model, source catalogue, reference configuration or released result exists.

`ScenarioConfig` is an immutable input value. Validation, import, Reset, `apply_overrides`, navigation and ordinary Streamlit reruns do not run the model. A full Monte Carlo may start only after an explicit **Run** click against a valid captured configuration hash.

This contract deliberately does not reproduce the legacy 99-key JavaScript object. Generated population traits, exogenous paths, arrival SoC, planner choices, route assignments, clearing, delivery and settlement are outputs or internal state, never configuration inputs. Streamlit may edit and validate this boundary but may not add hidden defaults or calculations.

The new repository does not yet contain its approved source catalogue, reference configuration or released fixture. Its narrow physical foundation contract is accepted, but unresolved physical and release choices remain gated. The legacy model contract, report, Python migration plan and Python-first amendment were used as read-only historical evidence. No legacy value becomes an approved Python default through this document.

## 2. Public representation

The settings JSON and Python field names use the exact lower `snake_case` names below. Aliases, case folding and key coercion are forbidden. Every object is closed: an unlisted key at any depth is an `unknown_field` error.

The root object has exactly four fields:

| JSON path | Type and constraint | Unit | Omission/default |
| --- | --- | --- | --- |
| `/schema_id` | Literal string `axle.scenario-config.v1` | none | Required; no default |
| `/editable_assumptions` | `EditableAssumptions` object | none | Required; no default |
| `/random_controls` | `RandomControls` object | none | Required; no default |
| `/reference_bindings` | `ReferenceBindings` object | none | Required; no default |

`ScenarioConfig` contains no display name, timestamp or user identity. A validated configuration is identified by its content hash. A UI may keep labels outside this boundary.

### 2.1 Editable assumptions

| JSON path | Type and constraint | Unit | Omission/default |
| --- | --- | --- | --- |
| `/editable_assumptions/start_local_date` | Valid Gregorian date in ISO `YYYY-MM-DD` form; the civil day starts at local midnight in `Europe/London`; the accepted window must satisfy the constant-offset rule below | local civil date | Required; no default |
| `/editable_assumptions/horizon_days` | Integer literal `1`, `3` or `7`; booleans are not integers | civil days | Required; no default |
| `/editable_assumptions/vehicle_count` | Integer from `1` to `9007199254740991`; the safe execution maximum is pending | vehicles | Required; no default |
| `/editable_assumptions/preferred_target_soc_fraction` | Finite JSON number, `0 < value <= 1` | fraction, not percent | Required; no default |
| `/editable_assumptions/mobility_reserve_soc_fraction` | Finite JSON number, `0 <= value <= preferred_target_soc_fraction` | fraction, not percent | Required; no default |
| `/editable_assumptions/fleet_import_limit_kw` | Finite JSON number `>= 0`, or `null` for no modelled fleet import limit; the safe maximum is pending | interval-average kW | Required; no default |
| `/editable_assumptions/enabled_route_ids` | Array of unique `RouteId` strings; empty is allowed; input order is not significant | none | Required; no default |

The preferred target is not the physical 100% ceiling. The mobility reserve is a protected lower bound, not a target. Neither field permits independently sampled arrival SoC: arrival SoC follows the battery stock-flow path.

The tested local window is `[start_local_date at 00:00, start_local_date + horizon_days civil days at 00:00)` in `Europe/London`. Its UTC offset must be constant and its elapsed duration must contain exactly `48 * horizon_days` half-hour starts. A daylight-saving transition in that window is a `constraint_violation` at `/editable_assumptions/start_local_date`.

The fixed interval is 30 minutes and is not configurable in v1. `half_hour_count` is derived as `48 * horizon_days` and must not appear in input JSON. Physical power is interval-average kW; battery energy is kWh; physical SoC remains within `[0, 1]`.

The v1 route IDs are:

```text
wholesale
dfs
frequency_response
local_flexibility
capacity_market
supplier_optimisation
```

An enabled route is only a planning candidate. Enabling several routes does not permit simultaneous physical dispatch.

### 2.2 Technical random controls

| JSON path | Type and constraint | Unit | Omission/default |
| --- | --- | --- | --- |
| `/random_controls/root_seed` | Non-negative JSON integer no greater than `9007199254740991`; booleans are not integers | dimensionless seed | Required in saved configuration |
| `/random_controls/evaluation_world_count` | Positive JSON integer; the runner applies a practical execution limit before starting | worlds | Required; no default |
| `/random_controls/planning_world_count` | Positive JSON integer; the runner applies a practical execution limit before starting | worlds | Required; no default |

If the Run UI omits a seed, it generates one non-negative integer only after a valid explicit Run click, stores it in the saved configuration, then creates `np.random.default_rng(root_seed)` once for that run. Validation, editing and previewing do not start a full simulation.

World indices are zero-based within their own role: `planning:0`, `planning:1`, and `evaluation:0`, `evaluation:1` are distinct labels. They do not need numeric offsets or RNG namespaces. The planner receives no future evaluation samples.

The sampler creates persistent traits once, draws distinct planning and evaluation inputs, and reuses each evaluation world's exogenous arrays in normal and selected paths. Tests check this data flow directly.

Run dimensions are technical controls, not behavioural evidence. A fixed seed reproduces a run with the same configuration, model code and dependency versions; it does not promise bit-identical legacy JavaScript draws or stability after changing draw order.

### 2.3 Reference bindings

Each binding is a closed `ReferenceBinding` object:

| Field | Type and constraint | Unit | Omission/default |
| --- | --- | --- | --- |
| `id` | Stable reference ID | none | Required; no default |
| `sha256` | String `sha256:` followed by exactly 64 lower-case hexadecimal characters | content digest | Required; no default |

`ReferenceBindings` has exactly these fields:

| JSON path | Required reference kind | Owns |
| --- | --- | --- |
| `/reference_bindings/fleet_profile` | `fleet` | Cohorts, region allocation, battery capacities, efficiencies, hardware limits, initial stock rule, persistent-trait definitions, trait distributions and generation rules, travel and connection distributions |
| `/reference_bindings/market_profile` | `market` | Synthetic demand, weather and price-path definitions, source snapshots and information-available timestamps |
| `/reference_bindings/route_catalogue` | `routes` | The six route contracts, eligibility, units and commercial assumptions |
| `/reference_bindings/portfolio_profile` | `portfolio` | Candidate construction, risk and selection rules; it does not contain a generated selected policy |
| `/reference_bindings/source_catalogue` | `sources` | Evidence class and source-to-assumption links; it is explanatory and cannot override executable values |

A stable reference ID matches:

```text
^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*\.v[1-9][0-9]*$
```

An accepted reference binding identifies one local artefact. Validation resolves its ID locally, checks its kind and digest, and fails before execution if it is missing or mismatched. Snapshot-mode validation performs no network fetch.

The profile split is the extension point for the reviewed full reference. It can carry the eventual field-by-field crosswalk without adding the legacy 99 keys to this public envelope. A new public root field or a changed meaning requires `axle.scenario-config.v2`; a new immutable profile version does not.

## 3. Stable generated identities and planner-state boundary

These generated identities are stable within one configuration hash:

| Entity | ID rule |
| --- | --- |
| EV | `ev:<zero-based integer>` |
| Evaluation world | `evaluation:<zero-based integer>` |
| Planning world | `planning:<zero-based integer>` |
| Interval | ISO-8601 UTC half-hour start ending in `Z` |
| Configuration | `sha256:<64 lower-case hexadecimal characters>` |

Generated planner state is a separate immutable value. Its minimum identity envelope is `planner_state_schema_id`, `planner_state_id`, `config_hash`, `planning_world_ids` and `selected_policy_id`; its detailed candidate and decision fields belong to the reviewed planner contract. It is produced only after configuration validation and reference resolution. It is never accepted under `ScenarioConfig`, persisted as an editable assumption or copied by `apply_overrides`.

For every EV, world and UTC half-hour slot, every planning-world physical candidate assignment and resolved evaluation-world physical assignment has cardinality zero or one. The physical-owner type is narrower than `RouteId`:

```text
PhysicalOwnerId = wholesale | dfs | frequency_response | local_flexibility | capacity_market
dispatch_route_id[world, path, interval, ev] is PhysicalOwnerId | null
```

`world` ranges independently over the configured planning and evaluation world IDs. `path` is `planning` for planning worlds and `normal` or `selected` for evaluation worlds. Paths are alternative physical trajectories, not additive fleet capacity. `null` means no commercial dispatch. `supplier_optimisation` is never a `PhysicalOwnerId`; its benefit, fee or other analytical value is an overlay on ordinary charging or one physical owner and cannot issue an independent command. `capacity_market` is a `PhysicalOwnerId` only when an actual Capacity Market stress obligation dispatches that EV-slot; a commitment, accrual or other non-dispatch record is not a physical owner. Multiple eligibility records, reservations or route ledgers may exist, but no cell may contain more than one dispatchable route. This prevents double allocation of the same EV/path/slot. Actual instruction, clearing, delivery, penalty and cash remain later evaluation outputs and cannot flow back into the selected policy.

## 4. Validation contract

Parsing rejects malformed JSON, duplicate object keys, non-finite numbers and values outside JSON's data model. Validation does not coerce strings to numbers, numbers to strings, booleans to integers, tuples to arrays or percentages to fractions. It collects independent issues where safe, sorts them by JSON Pointer and then error code, and raises one typed `ConfigValidationError`. No partial configuration is returned.

The serialisable error envelope is:

```json
{
  "schema_id": "axle.config-errors.v1",
  "issues": [
    {
      "path": "/editable_assumptions/horizon_days",
      "code": "invalid_value",
      "message": "Expected one of 1, 3 or 7."
    }
  ]
}
```

`path` is an RFC 6901 JSON Pointer; the empty string identifies the whole document. `code` and `path` are contractual. `message` is explanatory and may improve without a schema change. Values are not echoed into messages.

The v1 error codes are:

| Code | Meaning |
| --- | --- |
| `invalid_json` | Bytes are not one valid JSON document |
| `duplicate_key` | An object contains the same key more than once |
| `unsupported_schema_id` | A present string `schema_id` is not recognised by the supported schema registry |
| `missing_field` | A required field is absent |
| `unknown_field` | A closed object contains an unlisted field |
| `invalid_type` | A value has the wrong JSON type, including a boolean supplied for an integer |
| `invalid_value` | A value fails an enum, format or scalar-range rule |
| `constraint_violation` | Two or more otherwise typed values violate a cross-field rule |
| `duplicate_id` | A set-like ID array contains a repeated value |
| `unknown_route_id` | A route is not in the v1 route-ID set or the bound catalogue |
| `unknown_reference_id` | A bound stable ID is absent from the local accepted registry |
| `reference_kind_mismatch` | The resolved artefact has the wrong declared kind |
| `reference_hash_mismatch` | The resolved bytes do not match the bound digest |
| `invalid_override_path` | An override path is malformed, unknown or identifies a container rather than a leaf |
| `non_editable_override` | An override targets anything outside `editable_assumptions` |

Validation order and deterministic issue locations are:

| Rule | Code | Path | Continuation |
| --- | --- | --- | --- |
| Malformed JSON | `invalid_json` | empty pointer | Stop |
| Duplicate object member | `duplicate_key` | pointer to the repeated member where available, otherwise empty pointer | Stop |
| Missing `schema_id` | `missing_field` | `/schema_id` | Stop version-specific validation |
| Non-string `schema_id` | `invalid_type` | `/schema_id` | Stop version-specific validation |
| Present unrecognised `schema_id` | `unsupported_schema_id` | `/schema_id` | Stop version-specific validation |
| Invalid Gregorian date or scalar/format/enum bound | `invalid_value` | pointer to the field | Continue where safe |
| Daylight-saving window | `constraint_violation` | `/editable_assumptions/start_local_date` | Continue where safe |
| Repeated route ID | `duplicate_id` | pointer to the later array member | Continue where safe |
| Unknown route ID | `unknown_route_id` | pointer to the array member | Continue where safe |
| Missing reference registry entry | `unknown_reference_id` | pointer ending `/id` | Continue with other bindings |
| Wrong reference kind | `reference_kind_mismatch` | pointer ending `/id` | Do not hash-check that binding |
| Wrong reference digest | `reference_hash_mismatch` | pointer ending `/sha256` | Continue with other bindings |

After the schema gate, the validator returns all safely discoverable object-shape and scalar issues, then cross-field issues, then reference ID, kind and digest issues. Issues are sorted by pointer and code after collection. A caller must not start planning or simulation while any issue exists.

## 5. Canonical JSON and configuration hash

Validation normalises only values whose ordering is explicitly non-semantic:

- `enabled_route_ids` is sorted by Unicode code point after duplicate detection.
- Every accepted negative-zero number is normalised to positive zero. JSON numbers have IEEE-754 binary64 semantics; lexical spellings that parse to the same binary64 value are the same value.
- Object members are handled by canonical serialisation, not by input order.
- No other array is reordered.

The canonical form is RFC 8785 JSON Canonicalization Scheme over the complete validated and normalised `ScenarioConfig`, encoded as UTF-8 with no byte-order mark or trailing newline. Strings are Unicode and must satisfy the RFC 8785/I-JSON constraints. The `root_seed` is a JSON integer. Integers remain inside the exact interoperable range `[-9007199254740991, 9007199254740991]`; all numbers are finite binary64 values. Parsing, validation and canonicalisation must use one specified RFC 8785-conformant numeric path rather than host-dependent decimal handling.

`config_hash` is:

```text
"sha256:" + lower_hex(SHA-256(canonical_config_utf8))
```

The hash includes every explicit assumption, technical random control, reference ID and bound reference digest. It excludes labels held by the UI, runtime diagnostics, progress, generated population, exogenous paths and planner state. Two inputs that differ only in object-member order, `enabled_route_ids` order, negative-zero sign or an equivalent RFC 8785 number spelling have the same hash. A change to any other normalised semantic value changes the hash.

Canonicalisation is a required fixture-tested dependency. Falling back to ordinary `json.dumps`, locale-sensitive formatting or ad hoc recursive sorting is not conformant.

## 6. `apply_overrides` semantics

The public operation is conceptually:

```text
apply_overrides(base: ScenarioConfig, overrides: Mapping[JsonPointer, JsonValue])
    -> ScenarioConfig
```

`base` must already be validated. Each override key is an RFC 6901 pointer naming exactly one of these seven editable fields:

```text
/editable_assumptions/start_local_date
/editable_assumptions/horizon_days
/editable_assumptions/vehicle_count
/editable_assumptions/preferred_target_soc_fraction
/editable_assumptions/mobility_reserve_soc_fraction
/editable_assumptions/fleet_import_limit_kw
/editable_assumptions/enabled_route_ids
```

All replacements are applied to a copy, then the complete candidate is validated and normalised once. Success returns a new immutable value and therefore a newly calculated configuration hash. The base value and caller-owned override mapping are never mutated.

The operation is atomic. If any path or candidate value is invalid, it raises `ConfigValidationError`, returns no candidate and leaves `base` unchanged.

Exact rules:

1. Omitted paths retain their base values.
2. A named editable field is replaced, never arithmetically adjusted or recursively guessed.
3. `/editable_assumptions/enabled_route_ids` replaces the whole list. Its descendant/index paths are invalid; there is no merge by array position or route ID.
4. `/editable_assumptions`, the document root and any other structural object path are invalid container paths.
5. A path cannot create or delete a field.
6. `null` is accepted only at `/editable_assumptions/fleet_import_limit_kw`.
7. JSON types must already match; there is no coercion.
8. Technical random controls and reference bindings are not editable through this operation. Changing either requires construction and validation of a complete new `ScenarioConfig`.
9. Duplicate pointers after RFC 6901 decoding are rejected as `duplicate_key`.
10. Override application performs no model run, reference refresh or planner invocation.

## 7. Default and bounds status

There are no implicit v1 field defaults. The tiny fixture below supplies synthetic test values explicitly. It is not the released scenario and is not evidence-backed.

The following remain **pending lead approval** and must not be inferred from the legacy checkout:

- every source-backed or elicited fleet, behaviour, physical, market, route and portfolio value;
- the full reference's start date, preferred-target treatment, mobility reserve, grid limit and planning-world count;
- the canonical IDs and digests of the full reference profiles and source catalogue;
- safe execution limits for vehicles, planning worlds and evaluation worlds;
- any UI range narrower than the validated model boundary.

The accepted product horizons remain 1, 3 and 7 civil days. The released-reference acceptance target remains exactly 1,000 vehicles, 100 evaluation worlds and 336 UTC half-hour intervals for the seven-day case. That target is neither a tested maximum nor a performance, memory, concurrency or hosting claim. The exact full-size run, benchmark and two-viewer gates remain future release evidence.

Schema validity and run admission are distinct. Until measured execution limits are approved, callers may validate positive dimensions but must not present large runs as supported or start an unbounded run. A runner limit must reject an excessive run rather than silently shrink it.

## 8. Valid tiny fixture

This fixture refers to the five fixture-only reference artefacts below. Each row gives the complete UTF-8 byte sequence with no byte-order mark or trailing newline. All values and artefacts are synthetic test inputs, not product defaults.

| Registry ID | Exact artefact bytes | SHA-256 binding |
| --- | --- | --- |
| `fleet.synthetic_tiny.v1` | `{"kind":"fleet","schema_id":"axle.fixture-reference.v1"}` | `sha256:514c7f04cb2530dc4f0ccd460391e962ce0a3093163beb688f32ea16aa4874a3` |
| `market.synthetic_flat.v1` | `{"kind":"market","schema_id":"axle.fixture-reference.v1"}` | `sha256:0ea4e2a66a3371c7b1c3d5e6417e6b873b758f6c8cdb5e43912bfafbaf1be1e1` |
| `routes.synthetic_minimal.v1` | `{"kind":"routes","schema_id":"axle.fixture-reference.v1"}` | `sha256:bb10ca013b2b8dd45ddbd114fc80fb5a12010c853e8dc389593c36fcc3017e7e` |
| `portfolio.synthetic_no_action.v1` | `{"kind":"portfolio","schema_id":"axle.fixture-reference.v1"}` | `sha256:17d9c0ae532c0f9a2e0d20b933c1224cf0d65a8f71d03755f6a0d59b77ad0863` |
| `sources.synthetic_fixture.v1` | `{"kind":"sources","schema_id":"axle.fixture-reference.v1"}` | `sha256:fdfad21adee2d8835bbdf7cc37eafacd6383ac681633fb1b84106a883d9acde2` |

```json
{
  "schema_id": "axle.scenario-config.v1",
  "editable_assumptions": {
    "start_local_date": "2026-01-15",
    "horizon_days": 1,
    "vehicle_count": 2,
    "preferred_target_soc_fraction": 0.8,
    "mobility_reserve_soc_fraction": 0.2,
    "fleet_import_limit_kw": 14,
    "enabled_route_ids": ["wholesale"]
  },
  "random_controls": {
    "root_seed": 42,
    "evaluation_world_count": 2,
    "planning_world_count": 2
  },
  "reference_bindings": {
    "fleet_profile": {
      "id": "fleet.synthetic_tiny.v1",
      "sha256": "sha256:514c7f04cb2530dc4f0ccd460391e962ce0a3093163beb688f32ea16aa4874a3"
    },
    "market_profile": {
      "id": "market.synthetic_flat.v1",
      "sha256": "sha256:0ea4e2a66a3371c7b1c3d5e6417e6b873b758f6c8cdb5e43912bfafbaf1be1e1"
    },
    "route_catalogue": {
      "id": "routes.synthetic_minimal.v1",
      "sha256": "sha256:bb10ca013b2b8dd45ddbd114fc80fb5a12010c853e8dc389593c36fcc3017e7e"
    },
    "portfolio_profile": {
      "id": "portfolio.synthetic_no_action.v1",
      "sha256": "sha256:17d9c0ae532c0f9a2e0d20b933c1224cf0d65a8f71d03755f6a0d59b77ad0863"
    },
    "source_catalogue": {
      "id": "sources.synthetic_fixture.v1",
      "sha256": "sha256:fdfad21adee2d8835bbdf7cc37eafacd6383ac681633fb1b84106a883d9acde2"
    }
  }
}
```

The example above is synthetic. When the simplified configuration parser is implemented, its tests should verify a stable configuration hash without freezing a long canonical JSON line in this document.

Required numeric edge vectors are:

- `0`, `-0` and `-0.0` at a field that permits zero all normalise to canonical `0` and the same hash contribution;
- `1`, `1.0` and `1e0` at an integer field all represent the same mathematical integer and canonicalise to `1`;
- `9007199254740991` is structurally representable where the field's own range permits it, while `9007199254740992` is `invalid_value`;
- `NaN`, `Infinity` and `-Infinity` are `invalid_json`, not configuration values.

Required seed and world-count edge vectors are:

- `root_seed` accepts the JSON integers `0`, `42` and `9007199254740991`; it rejects a negative number, a string, a boolean and `9007199254740992`;
- both world counts accept positive JSON integers and reject zero, negatives and booleans; the runner separately rejects counts above its practical execution limit.

## 9. Invalid examples

Each fragment is applied to an otherwise valid fixture.

```json
{"horizon_days": 2}
```

At `/editable_assumptions/horizon_days`: `invalid_value`. Only 1, 3 and 7 are accepted.

```json
{"evaluation_world_count": true}
```

At `/random_controls/evaluation_world_count`: `invalid_type`. JSON booleans are not integers.

```json
{
  "preferred_target_soc_fraction": 0.8,
  "mobility_reserve_soc_fraction": 0.9
}
```

At `/editable_assumptions/mobility_reserve_soc_fraction`: `constraint_violation`. The reserve cannot exceed the preferred target.

```json
{"enabled_route_ids": ["dfs", "dfs"]}
```

At `/editable_assumptions/enabled_route_ids/1`: `duplicate_id`.

```json
{"enabled_route_ids": ["frequency"]}
```

At `/editable_assumptions/enabled_route_ids/0`: `unknown_route_id`.

```json
{"arrival_soc_fraction": 0.4}
```

At `/editable_assumptions/arrival_soc_fraction`: `unknown_field`. Arrival SoC is generated stock-flow state.

```json
{"/random_controls/root_seed": 99}
```

As an override mapping: `non_editable_override` at `/random_controls/root_seed`.

```json
{"surprise": 1}
```

At any otherwise closed object: `unknown_field` at that field's JSON Pointer.

## 10. Review and dependency gate

An earlier model/config review accepted the previous synthetic input contract on 26 September 2026. The revised RNG controls and any further simplification need a fresh focused review before dependent configuration implementation is integrated. Synthetic physical tests with explicit inputs may continue. This does not authorise source-backed defaults, commercial claims or release qualification.

The source catalogue/defaults, full-reference IDs, planning-world count and execution bounds remain pending for the gated slices above. Any change to the public fields, route IDs, canonicalisation or override rules after acceptance requires a versioned contract amendment and notice to all dependent owners.
