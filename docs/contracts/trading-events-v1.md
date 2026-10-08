# Trading, events and zones contract v1

Status: draft for lead acceptance, 29 September 2026 (plan task W1-trading-contract), revised after review (blocking items B1–B7, lead decisions QA–QC and Q1–Q13, §8). Design only: no model code exists for anything below. **Extended 29 September 2026 (task W1-contract-supplier) by §9: decision 0004 item 58 and plan §8 I3–I7 (trader metrics, commitment share, Supplier page, non-response by archetype, baseline trap, hold-out control group, stress presets); §9 is a draft for lead acceptance. Approved text changed by §9 carries a "Changed by §9" note in place. Extended again 29 September 2026 (task W2-contract-availability) by §10: decision 0004 item 59 and plan §11 (shared plug-in factors, manufacturers, deliverable MW at two horizons, calibration backtest, newsvendor commitment, product sheet, blackout windows, settlement file, firmness by manufacturer, notebook 09) plus Mike's same-day addition (week-level factor, correlation diagnostics, fleet-size study); §10 is a draft for lead acceptance and its changes to earlier text carry a "Changed by §10.x" note in place.**

Authority: decision 0004 items 50–56 (56: stochastic net-demand shocks), item 58 (§9) and item 59 (§10, with plan §11), decision 0005, `docs/plans/2026-09-29-analyst-trading-polish-plan.md` §2F (F1–F3) and §7 (H2–H7), `docs/contracts/results-v2.md` (its §1 rules apply to every frame here), `docs/contracts/random-v1.md`, and the lead's review decisions recorded in §8. Base: `7345583`.

This contract is the shared spec that wave-3/4 writers build against in parallel: `model/market.py` (F1, F2), `model/events.py` (H2–H4), zones (H5), summaries, validators and the Trading, Response and Fleet week views. It assumes the wave-1/2 changes land as planned (A1 clocks, A4 noon-to-noon horizon, A5 seven-day warm-up with prices, B1/H1 system-model prices with the item 56 shocks, B4 publication mask, B5 intraday and imbalance channels). Section 1.4 lists exactly what it consumes from them. Where a name differs when those tasks land, the lead renames here; the meaning must not change.

Every price, volume and £ figure below is synthetic or illustrative. The only net money figure is "illustrative simulated trading P&L" (decision 0005), never "Axle cash".

## 0. Terms and sign conventions

| Term | Meaning |
| --- | --- |
| Session night `n` | Study block `n = 0…6`: London 12:00 on date `D_n` to London 12:00 on `D_n + 1` (item 52). Normally 48 slots; 46 or 50 across a clock change. The study is a fixed 336 UTC slots, so across a change it ends at 13:00 (spring) or 11:00 (autumn) London and night 6 holds 50 or 46 slots: the spring week's two slots past the last noon count to night 6. `study_slots.night_index` gives each slot's night and `study_slots.local_date` is `D_n` (`clock.session_night_dates`, London time − 12 h) |
| Slot | One fixed UTC half-hour (results-v2 §1 rule 4). Keys are `slot_index` + `interval_start_utc`. `s(τ)` is the slot starting at instant `τ` |
| `τ_DA(n)` | The day-ahead decision instant of night `n`: 13:00 London on `D_n − 1` |
| Unmanaged `U` | Fleet home import on `path_id="normal"` (item 50 label), kWh per slot |
| Metered `M` | Fleet home import on the dispatched path, `path_id="selected"` (smart charging), kWh per slot |
| Baseline `B` | BL01-lite settlement baseline (§4.1), kWh per slot; `B⁰` before the in-day adjustment |
| Deviation `D` | `B − M`, the observable settlement quantity |
| Position | Turn-down sold against the baseline, kWh per slot, always ≥ 0. It forecasts the settled deviation (lead decision QC); the cheap charging leg stays on the customer's tariff (decision 0005) |
| Sign of money | Aggregator's view: positive = income, negative = cost (decision 0005) |
| Sign of volume | Turn-down (importing less) is positive |
| Shock | A net-demand shock (GW, trapezoid in time). Either stochastic (item 56, drawn by the prices generator, different in every world) or scripted (a price-shock preset of §2, the same in every world). Each is **known** (in the day-ahead price) or a **surprise** (intraday and imbalance only) |
| Shock kind of a slot | `surprise` if any surprise shock has non-zero weight in the slot, else `known` if any known shock does, else `none`. Surprise wins because it is what moves intraday and imbalance |

`kWh ÷ 1000 × £/MWh` is the only money formula for energy; no field mixes kWh and MWh without saying so in its name.

## 1. Module layout and call order

Plain functions over NumPy arrays and pandas DataFrames. No classes except small frozen dataclasses where a function returns several arrays together; no registries; no optimisation solver (lead decision QA). Array shapes are written `(world, slot)` etc.; floats are `float64`, counts `int64`, flags `bool`, instants `int64` UTC nanoseconds inside the model and `datetime64[ns, UTC]` in frames.

### 1.1 `model/events.py` (H2–H5 event side; W3, short-notice replanning W4)

Owns the events table, its presets and validation, and the mapping of each scripted event to shock profiles, planner signals, trading masks and grid-event delivery. The stochastic shocks of item 56 are owned by the prices generator; scripted price shocks are layered on top of them (§2.5). Draws no random number.

```python
EVENT_TYPES = ("price_shock_known", "price_shock_surprise", "turn_down", "turn_up")
EVENT_COLUMNS = (...)                      # §2.1, in order
EVENT_PRESETS: dict[str, pd.DataFrame]     # §2.2, one-row frames keyed by preset id

def empty_events() -> pd.DataFrame:
    """The default: no events (existing results stay unchanged)."""

def validate_events(events: pd.DataFrame, study_slots: pd.DataFrame, zone_ids: tuple[str, ...]) -> pd.DataFrame:
    """Return a typed copy or raise ValueError naming the row and column (§2.3)."""

def event_slots(events: pd.DataFrame, study_slots: pd.DataFrame) -> pd.DataFrame:
    """One row per enabled event: event_id, event_type, scope, size, payment_gbp_per_mwh,
    start_slot, end_slot (exclusive), notice_utc (§2.1), notice_slot (first study slot starting at or
    after notice_utc; 0 if before the study)."""

def scripted_shock_profiles(events: pd.DataFrame, slots: pd.DataFrame,
                            slot_count: int) -> tuple[np.ndarray, np.ndarray]:
    """known_gw (slot,) and surprise_gw (slot,): the enabled price-shock presets as signed net-demand
    profiles over warm-up + study slots, trapezoid like item 56's shocks (§2.5). The prices generator
    adds them to its own stochastic profiles before the supply curve; zeros reproduce it bit for bit."""

def scripted_shock_rows(events: pd.DataFrame, slots: pd.DataFrame, world_count: int) -> pd.DataFrame:
    """The scripted price shocks as shock-table rows (one per world, source="scripted"), §5.6a."""

def planner_adjustments(events: pd.DataFrame, slots: pd.DataFrame, zone_count: int,
                        slot_count: int) -> tuple[np.ndarray, np.ndarray]:
    """notice_slot (event,) int64 and adjustment_gbp_per_mwh (event, zone, slot) float64:
    +payment inside a turn_down window, −payment inside a turn_up window, 0 elsewhere.
    A consumer applies event e only to decisions at or after notice_slot[e] (§2.4)."""

def trading_mask(slots: pd.DataFrame, slot_count: int, *, known_at_utc: int | None) -> np.ndarray:
    """(slot,) bool: False inside the window of every turn_down or turn_up request whose notice is at or
    before known_at_utc (int64 UTC ns), True elsewhere. known_at_utc=None uses every request: the
    settlement mask (§4.6)."""

def event_delivery(events: pd.DataFrame, slots: pd.DataFrame, scope_baseline_kwh: np.ndarray,
                   scope_metered_kwh: np.ndarray) -> pd.DataFrame:
    """Per (world, event): delivered kWh against the baseline and the illustrative payment (§2.4).
    scope arrays are (world, scope, study slot), scope 0 = fleet, 1–4 = zones."""
```

**Changed by §9.6 (item 58, I7):** `EVENT_TYPES` gains `"control_outage"`; `EVENT_PRESETS` is replaced by `preset_rows(preset_id, study_slots)` because the new sunny-weekend preset depends on the study start weekday; a new `outage_probability(...)` gives the kernel the control outage's non-response probability.

**Changed by §10.5b:** `trading_mask(..., blackout=...)` takes the study blackout mask and returns False on blackout slots at every `known_at_utc`; `validate_events` rejects a request whose window lies wholly inside a blackout.

### 1.2 `model/market.py` (F1 in W3, F2 in W4)

Owns the BL01-lite baseline, the expected availability, the expected smart plan, day-ahead and intraday positions, imbalance settlement, the three strategies and the trading ledger. It reads kernel outputs and never changes them (plan §2F), so all three strategies come from one physics run and share its random futures. It draws no random number.

```python
STRATEGIES = ("da_only", "full", "perfect_foresight")

def bl01_lite_baseline(warmup_kwh: np.ndarray, warmup_slots: pd.DataFrame, study_slots: pd.DataFrame, *,
                       working_nights: int, non_working_nights: int) -> np.ndarray:
    """warmup_kwh (world, [scope,] warm-up slot) unmanaged home import -> B⁰ (world, [scope,] study slot) kWh (§4.1)."""

def in_day_adjustment(baseline_kwh: np.ndarray, metered_kwh: np.ndarray, study_slots: pd.DataFrame, *,
                      window_slots: int) -> np.ndarray:
    """(world, [scope,] study slot) additive adjustment, constant within each night (§4.1)."""

def expected_sessions(units: pd.DataFrame, study_slots: pd.DataFrame, night_index: int, *,
                      warmup_need_kwh: np.ndarray, departure_margin_hours: float,
                      decision_slot: int | None = None, include: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Expected home session per EV for one night, from information held at the decision (§4.3).
    Keys: weight (world, ev) plug-in probability or 0; need_kwh (world, ev) grid kWh, capped to what the
    (re-based) window can take; start (ev,) and deadline (ev,) slot offsets in the night; slot_cap_kwh (ev,).
    With decision_slot set (intraday, §4.5), start is re-based to max(start, decision_slot) and sessions
    whose deadline is at or before decision_slot get weight 0."""

def expected_unmanaged_kwh(sessions: dict[str, np.ndarray], night_slots: int) -> np.ndarray:
    """(world, ev, night slot) expected full-power-from-start import per session (the u_i of §4.4)."""

def expected_smart_kwh(sessions: dict[str, np.ndarray], window_price_gbp_per_mwh: np.ndarray, *,
                       non_response_probability: float) -> np.ndarray:
    """(world, night slot) x = Σ_i w_i [ρ u_i + (1 − ρ) plan_i], plan_i from action.plan_cheapest_slots on
    each session's visible window prices (world, ev, night slot; +inf outside its window) (§4.4)."""

def positions_kwh(baseline_kwh: np.ndarray, forecast_metered_kwh: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(world, slot) target position = mask × max(0, B − M̂) (lead decision QC)."""

def intraday_positions_kwh(...) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """F2. Final positions (world, study slot) kWh, per-slot intraday cash and traded volume, and the
    position_updates rows for sampled worlds (§4.5)."""

def settle(deviation: pd.DataFrame, positions_kwh: dict[str, np.ndarray], intraday: dict[str, np.ndarray],
           prices: dict[str, np.ndarray], event_payments: pd.DataFrame, unmet_charge_kwh: np.ndarray,
           assumptions: Mapping[str, float]) -> pd.DataFrame:
    """trading_ledger_world (§5.3): one row per (world, strategy, night); net is the exact bucket sum."""

def run_trading(...) -> dict[str, pd.DataFrame]:
    """The whole overlay: baseline, expectations, positions per strategy, settlement. Returns the frames of §5."""
```

`run_trading` loops nights (and, in F2, decision hours); everything else is vectorised over worlds and EVs. The frame-building summaries (bands, KPIs, checks) live in `model/summaries.py` like every other band.

### 1.3 Other modules touched

| Module | Addition | Task |
| --- | --- | --- |
| `assumptions.py` | `TRADING`, `EVENTS`, `ZONES` groups (§7); `non_response_probability` in the behaviour group | F1, H2, H5, F2 |
| `sampling.build_population` | `zone_id` column; one zone permutation at the end of the population draws (§3) | H5 |
| `sampling.py` | `sample_non_response(rng, world_count, study_night_count, vehicle_count) -> (world, night, ev) float64` uniforms | F2 |
| `physics.py` | zone group sums; warm-up normal-path sums; per-EV warm-up session need; departure shortfall per path; plan books (§1.4); planner adjustment and non-response inside `_smart_charging_slot` (§2.4, §4.5) | H4, H5, F1, F2 |
| `action.py` | `SmartCharging` gains `event_notice_slot`, `event_adjustment_gbp_per_mwh`, `zone_index`, `non_response` (plain arrays; `for_slice` slices them); `plan_cheapest_slots` is reused unchanged by `market.expected_smart_kwh` | H4, F1, F2 |
| `forecast.py` | call order below; `ForecastResult` gains the fields of §5 | F1, H2, H5 |
| `summaries.py` | the band, KPI and check frames of §5 | F1, H4, H5 |

**Changed by §10.0:** the plug-in factors, manufacturer assignment and outages (`sampling.py`), the holiday skip shifts (`clock.py`), the (world, night, EV) non-response array, the per-EV `planned_home_import_kwh` and `plan_status` outputs and the blackout planning rule (`action.py`, `physics.py`), and the new modules `model/availability.py`, `model/availability_backtest.py` and `model/product.py` are listed in §10.0.

### 1.4 What this contract consumes from waves 1–2

| Input | Shape | Owner | Used for |
| --- | --- | --- | --- |
| `study_slots.night_index` (int64 0–6) and `study_slots.local_date` (date, `D_n`; this contract's `night_start_local_date` is that column) | per study slot | A4, **landed** | nights, ledger keys |
| Warm-up slots with the same London keys: `SimulatedForecast.warmup_slots` (`summaries.build_warmup_slots`): `slot_index` −48W…−1, `local_half_hour`, `night_index` −W…−1, `local_date`, `day_type` (weekday/weekend of the night's evening = working/non-working) | per warm-up slot | A4/A5, **landed** | baseline |
| Day-ahead price: day-ahead frame column `wholesale_forecast_gbp_per_mwh` | (world, warm-up + study slot) | B1/H1 (`sampling.generate_market_price_paths`) | expected plan, DA revenue, planner |
| Net demand: day-ahead frame column `system_net_demand_gw` (includes known shocks) | (world, slot) | H1 | shock price increments |
| Supply curve `sampling.supply_curve_gbp_per_mwh(net_demand_gw, **curve values)` | vectorised function | H1 | shock price increments |
| Publication instant of each day-ahead price: `sampling.day_ahead_publication_utc_ns(interval_start_utc)` (re-exported by `action`) (13:00 London D−1, `PRICES["day_ahead_publication_local_hour"]`, the record the intraday updates start from; the generator stamps it on the day-ahead frame as `forecast_available_at_utc`); the single visible-price rule `action.visible_day_ahead_prices(day_ahead (world, slot), interval_start_utc, decision_utc) -> (world, slot)`, applied to the day-ahead (expected-close) price `wholesale_forecast_gbp_per_mwh`: a slot published at or before the decision keeps its price, any other slot gets the expected shape, the mean published price of the same London wall-clock half-hour over the `ACTION["expected_price_shape_days"]` (7) most recent published delivery days (fallback: the mean of those days' prices when a half-hour has none). It reads only published prices, so perturbing an unpublished price never changes its output (tested) | function | B4, **landed** | the expected plan sees exactly what the smart planner sees |
| Intraday close: realised frame column `evaluation_context_price_gbp_per_mwh` | (world, slot) | B5 | reported close |
| **Intraday path** `intraday_path_gbp_per_mwh` (world, slot, `h`), `h` = whole hours before gate closure, `h = 0` the close, 36 steps: the price after the update made `h` hours before gate closure (the generator already draws `_INTRADAY_UPDATE_DRAWS = 36` hourly updates per slot and returns only their sum) | (world, slot, h), in memory only | B5, **to add** (a cumulative sum of the existing draws; no new draw) | F2 intraday trades |
| Imbalance price: realised frame column `imbalance_price_gbp_per_mwh`, and `system_long` | (world, slot) | B5 | settlement |
| Shock table (item 56): `world_id`, `shock_class` (`mild`/`big`), `start_slot_index`, `start_utc`, `duration_slots`, `size_gw`, `direction` (`up`/`down`), `known_day_ahead`, `evidence_kind` | one row per shock | B1/H1 (third return value of `generate_market_price_paths`) | shock kinds, Market marks, attribution |
| Known and surprise shock profiles, GW per (world, slot) (`_shock_profiles`) | (world, slot) × 2 | H1, **to expose** with scripted profiles added (§2.5) | slot shock kind, price increments |
| Gate closure `gate_closure_hours` (1 h before slot start) | constant | B5 | intraday |

Kernel outputs this contract adds (physics lane):

- `warmup_home_import_kwh` (world, warm-up slot), **landed** (A5): `SimulatedForecast.warmup_home_import_kwh`, filled by `simulate_fleet_intervals(..., warmup_home_import_kwh=out)` from the normal (unmanaged) path. `warmup_zone_home_import_kwh` (world, zone, warm-up slot) is still to add (H5). The paths are identical in the warm-up except its last night, where smart planning already runs (item 52), so the baseline must read the normal path;
- `warmup_session_need_kwh` (world, ev): the mean grid kWh a home session needed at plug-in on the normal path in the warm-up, NaN where the EV had no warm-up session, and `warmup_session_count` (world, ev);
- zone world sums (§3);
- `departure_shortfall_kwh` per path, (world, study slot): Σ over home sessions ending in the slot of `max(0, target stock − stock at unplug)`, battery-side kWh (for B3's penalty, §4.7);
- F2 plan books `book_kwh[path]` (world, hourly decision, 48): fleet grid kWh the EVs plugged in at the start of that decision slot will import in each of the next 48 slots as then planned (selected: plans in force, including non-responders' full-power plans; normal: full power until target). The book at a decision uses only states at that instant.

### 1.5 Call order in `forecast.simulate_forecast` (action model)

Noon-to-noon study of 336 slots after a 7-day warm-up (item 52); warm-up slots are simulated but not shown.

1. Validate the external inputs once: settings, events (`events.validate_events`), zone shares and headrooms, trading assumptions.
2. `rng = np.random.default_rng(seed)`; `units = build_population(...)`, now with `zone_id` (zone permutation last in the population).
3. `evaluation = sample(settings)`: trips, connections, temperatures (unchanged order).
4. `known_gw, surprise_gw = events.scripted_shock_profiles(events, slots, n)`: the price-shock presets as net-demand profiles.
5. Market channels (B1/H1/B5) over warm-up and study, drawn in the order the prices generator sets, including the item 56 stochastic shocks; the scripted profiles are added to its known and surprise profiles before the supply curve. Scripted shocks draw nothing, so the draws are the same with and without presets.
6. F2: `non_response = sample_non_response(rng, ...)`, always drawn, after every price channel.
7. `notice_slot, adjustment = events.planner_adjustments(events, slots, 4, n)`; `smart = smart_charging_inputs(..., day-ahead prices (world, warm-up + study slot), notice_slot, adjustment, zone_index, non_response)`; it applies the B4 visible-price rule itself, once per 13:00 publication epoch, from the last warm-up night on (A5).
8. `simulate_fleet_intervals(...)`: both paths, now with zone sums, warm-up sums, warm-up session need, departure shortfall and (F2) plan books.
9. `calculate_wholesale_world_cost_effect(...)`: the customer's tariff leg at the day-ahead price (B3). Not a trading bucket.
10. `market.run_trading(...)`: baseline → expected sessions → expected plan → positions per strategy → settlement.
11. `build_summaries(...)`: adds the §5 frames, including `event_response_bands`, the shock frames of §5.6a, `zone_import_bands` and `zone_summary`.

The no-action model runs steps 1–3, 8 (normal path) and 11. Its trading and event fields are `None`; its zone frames exist (zones are part of the population).

**Changed by §10.0:** the call order with the shared factors, manufacturers, blackout masks and the availability, backtest and product frames is restated in full in §10.0.

## 2. Events (H2–H4, H6)

### 2.1 Events table (exact)

A DataFrame edited in Edit assumptions (`st.data_editor`), stored on the result as `events` after validation. Default: empty. One row per event.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `event_id` | object | Unique short id, e.g. `"dfs_mon"` |
| `event_type` | object | One of `EVENT_TYPES` |
| `enabled` | bool | Disabled rows are kept but ignored |
| `night_index` | int64 | Session night 0–6 |
| `start_local_time` | object | `"HH:MM"` London wall clock on a half-hour. `≥ 12:00` falls on `D_n`, `< 12:00` on `D_n + 1`, so every time names one instant inside night `n` |
| `duration_minutes` | int64 | Multiple of 30, 30–720; the window must end by the study end |
| `size` | float64 | Price shocks: added system net demand, GW (positive = tighter). Requests: cap on paid delivery, MW; NaN = no cap |
| `size_unit` | object | `"GW"` for price shocks, `"MW"` for requests (set by the validator from the type) |
| `notice` | object | `"day_ahead"` or `"short"` |
| `notice_minutes` | Int64 | `short` only: minutes before the window start the event becomes known (0–720); `<NA>` for `day_ahead` |
| `scope` | object | `"national"` or a zone id (`"zone_1"`…`"zone_4"`) |
| `payment_gbp_per_mwh` | float64 | Requests only, ≥ 0, illustrative; NaN for price shocks |

Notice instant (`notice_utc`). A `day_ahead` event of night `n` is known at `τ_DA(n)` (13:00 London on `D_n − 1`), the night's day-ahead decision instant, so the day-ahead position and every plan of that night can see it; this holds for a morning event on `D_n + 1` too (review B2). A `short` event is known at window start − `notice_minutes`.

### 2.2 Presets (illustrative values)

Each preset appends one row. Values are illustrative scenario choices, labelled so on screen; none is a forecast, a DFS rate or a DNO tariff.

| Preset id | Label | `event_type` | `night_index` | start | duration | `size` | `notice` | `scope` | `payment_gbp_per_mwh` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cold_still_evening` | Cold still evening (known day-ahead) | `price_shock_known` | 2 | 16:30 | 180 | +4.0 GW | `day_ahead` | `national` | NaN |
| `surprise_evening_spike` | Surprise evening spike | `price_shock_surprise` | 4 | 17:00 | 120 | +3.0 GW | `short`, 90 min | `national` | NaN |
| `dfs_turn_down` | Demand-flexibility turn-down called day-ahead | `turn_down` | 1 | 17:30 | 60 | NaN (no cap) | `day_ahead` | `national` | 500 |
| `local_turn_up` | Local turn-up in one zone | `turn_up` | 5 | 23:00 | 120 | NaN (no cap) | `day_ahead` | `zone_2` | 60 |

"Cold still evening" is expressed directly as net demand, standing for a colder evening (heating term) plus lower wind; it does not change the sampled temperatures, so EV driving efficiency is unaffected (stated as a limitation).

### 2.3 Validation rules (`validate_events`)

`event_id` unique and non-empty; `event_type` in `EVENT_TYPES`; `night_index` 0–6; `start_local_time` a real London time on that date on a half-hour (a skipped spring hour is rejected; the repeated autumn hour takes its first occurrence); `duration_minutes` a multiple of 30 in 30–720 and the window ends by the study end; price shocks: `scope="national"` (prices are national, item 53), `size` finite in −10…+10 GW, `payment_gbp_per_mwh` NaN; `price_shock_known` needs `notice="day_ahead"`; `price_shock_surprise` needs `notice="short"`; requests: `payment_gbp_per_mwh` finite and ≥ 0, `size` NaN or > 0; `scope` is `national` or a defined zone; two enabled requests may not overlap in time when their scopes overlap (national overlaps every zone), so delivery is paid once. Until H6 (W4) lands, a request with `notice="short"` is rejected with "short-notice requests are not modelled yet".

### 2.4 What each type does

No decision uses an event before its notice instant (review B2): planner signals apply only to plans made at or after `notice_slot`; each trading decision builds its mask with `trading_mask(..., known_at_utc=τ)`; only settlement uses the full mask.

| Type | Net demand and price | Planner (selected path) | Ledger |
| --- | --- | --- | --- |
| `price_shock_known` | Its trapezoid profile (§2.5) is added to the known net-demand profile, so it is inside `system_net_demand_gw` and the day-ahead price, and flows on to intraday and imbalance, which start from day-ahead. The supply curve is convex, so the same GW moves price more when the system is tight (item 55) | None directly: plans rank the shocked day-ahead price once it is published (B4) | Through prices only; its slots are `known` for attribution (§4.7) |
| `price_shock_surprise` | Its profile is added to the surprise profile: increment `f(ND + S) − f(ND)` on intraday from the reveal instant (§2.5) and on imbalance. Day-ahead is unchanged | None (plans use day-ahead prices only) | Intraday P&L and imbalance (H3); its slots are `surprise` for attribution |
| `turn_down` | None | Every plan made at or after `notice_slot` for an EV in scope adds `+payment_gbp_per_mwh` to its ranking price in window slots, so charging there is ranked as dearer by the payment it would forgo | Grid-event payment: `Σ_slots min(max(0, B^s − M^s), cap) ÷ 1000 × payment`, with `B^s`, `M^s` the scope's adjusted baseline and metered import and `cap = size × 1000 × 0.5` kWh per slot. Window slots are removed from trading once the event is known (§4.6) |
| `turn_up` | None | Same, with `−payment_gbp_per_mwh` (a bonus) | Payment on `max(0, M^s − B^s)`, same cap rule. Window slots removed from trading once known |

The adjustment changes plan ranking only; it never forces import, and the realised physics is unchanged. Rebound (charging moved just before or after the window) is reported, not penalised (§5.6). Plans made before the notice keep their original ranking; H6 (W4) adds re-planning of plugged-in EVs at the notice instant.

### 2.5 Scripted presets on top of the stochastic shock process (item 56)

The prices generator draws each world's stochastic shocks: frequent mild shocks and about five big shocks per simulated week, up or down, starting only between London 15:00 and 09:00, trapezoid in time, each known or a surprise (its assumptions: rates, sizes, tail, up-probability, durations, window, known shares). They are background weather of the market, present in every run, and are not rows of the events table.

A scripted price shock is one more shock of the same shape and meaning, the same in every world, layered on top:

- Profile: the trapezoid of item 56 (ramp over the first quarter, hold, ramp down over the last quarter; weight `min(1, (k + 1)/(r + 1), (D − k)/(r + 1))` for half-hour `k` of `D`, `r = D // 4`), signed by `size` (positive = up). Overlaps with stochastic shocks add.
- `price_shock_known` adds to the known profile; `price_shock_surprise` to the surprise profile. The generator applies both profiles exactly as it applies its own, so a scripted shock and a stochastic one of the same size, time and class move prices identically.
- Scripted shocks are not bound by the 15:00–09:00 start window (a scenario may place a shock anywhere); they are bound by §2.3.
- Reveal instant of a surprise: stochastic surprises are revealed a fixed illustrative lead before their start (2 h, owned and made editable by the prices lane, §8 Q10); scripted surprises use their `notice_minutes`. Intraday steps made before the reveal do not include the increment; the close and SIP always do when the reveal is at or before gate closure. A surprise revealed after gate closure moves SIP only.
- The result's shock table (§5.6a) holds both, with `source` = `stochastic` or `scripted`.

## 3. Zones (H5)

Four illustrative network zones `zone_1`…`zone_4`, labels "Zone A"…"Zone D", no geography, default share 0.25 each (§8 Q4). Each has an editable share (fractions summing to 1) and an optional headroom (kW, NaN = off). Headroom is reported against, never enforced by the physics in this contract.

Assignment: one draw at population build, the last draw of `build_population` (after the cohort permutation and personal mileage multipliers; **changed by §9.5 and §10.1d:** the control-group permutation and then the manufacturer permutation now follow it, so the zone draw is no longer last). Exact counts per zone by largest remainder on `vehicle_count × share` (ties to the lower zone index), then `rng.permutation` of the zone index array, the same method as cohorts. The draw count depends only on `vehicle_count`, so editing shares moves no other channel. Zones are independent of cohorts. `units` gains `zone_id` (object).

The kernel adds zone sums like its cohort sums: `zone_world_intervals` (at least) with keys `(world_id, path_id, zone_id, slot_index)` and columns `unit_count`, `connected_count`, `home_import_kwh`, `home_import_kw`, `early_departure_count`, `early_departure_shortfall_kwh`. Zone sums add up to the fleet sums exactly. `path_id` is `normal` and `selected`, plus the optional `timed` path (decision 0007, model step 2) whenever the run tracks it (`physics.simulate_fleet_intervals` given both `zone_sums` and `timed_start_allowed`); the timed pass still never enters the trading overlay.

Summaries report per zone: import bands per slot against headroom (§5.7) and a per-zone weekly summary (§5.8).

## 4. Trading overlay (F1–F3)

### 4.1 BL01-lite baseline

Per world, per scope (fleet, or zone for events), per study slot:

- Day class of a night: `working` when `D_n` is Monday–Friday, `non_working` on Saturday and Sunday (no bank holidays). The class goes by the evening's date because the evening carries most of the turn-down.
- **Changed by §9.5 (item 58, lead review of `dd6f29d`, B3): one history rule for both baseline modes.** The history of night `n` is every night with index `≤ n − 2`: warm-up nights (−7…−1) on the normal path, plus, in trap mode only (`trading.baseline_history = "flexed_study_nights"`), study nights `0 … n − 2` on the selected (metered) path. Night `n − 1` is excluded because it ends at 12:00 on `D_n`, after `τ_DA(n)` (13:00 on `D_n − 1`); the earlier text averaged warm-up night −1 into night 0's baseline, which the night-0 day-ahead position could not have known.
- Unadjusted baseline `B⁰_t` = mean of the history's home import over every slot of the same `local_half_hour` in the last (at most) `working_nights` (5) history nights of the night's day class, or the last (at most) `non_working_nights` (2) for a non-working night. **Fallback:** if the history holds no night of that class (possible with a short warm-up), the same count of the most recent nights of the other class is used, and the result's `trading_checks` gains an informative `baseline_class_fallback` row counting the nights that used it. Both copies of a repeated autumn half-hour enter the mean; a slot missing in spring simply has fewer values.
- In the default mode `B⁰` uses warm-up nights only, so the study week's own nights never enter it and flexed days cannot erode it. Real BL01 rolls forward over flexed days and decays; the trap mode (§9.5) measures that.
- In-day adjustment (on by default, switchable): per world and night, `a_n = mean(M_t − B⁰_t)` over the first `window_slots` (3) slots of the night (12:00–13:30 London); `B_t = B⁰_t + a_n`. It is known from 13:30 on `D_n`; decisions before then use `B⁰`, later intraday decisions use `B` (§4.5). Fleet home import is close to zero at 12:00–13:30 (few EVs are home and charging), so this adjustment is nearly inert: it keeps the BL01 shape without moving the baseline much (§8 Q3).

The baseline reads the normal (unmanaged) path's warm-up import. The two kernel paths are identical in the warm-up except its last night, where smart planning already runs so the first study night starts from smart batteries (item 52).

**Changed by decision 0004 item 67 (baseline erosion, what-if only).** Beside the default `B`, `market.flexed_history_baseline` builds a steady-state baseline learnt from smart nights only: for night `n`, the world's other study nights on the selected (metered) path, same day class, at most `working_nights` / `non_working_nights`, nearest earlier nights first wrapping round the study, the other class as fallback, the same `local_half_hour` matching and the same in-day adjustment. The model has no smart warm-up, so the week's other smart nights stand in for "previous weeks' flexed nights"; only night `n` itself is left out, and the `n − 2` cut-off is not applied because nothing is traded on this baseline. `run_trading` settles it with the same settlement mask, `V_flex = mask × max(0, B_flex − M)`, and returns it as `TradingRun.settled_flexed_baseline` (world, study slot, kWh). It is a what-if beside the ledger and never enters any position, deviation, ledger bucket or check; the default ledger is unchanged. With a 7-night study a working night learns from 4 nights and a non-working night from 1, so the whole-week figure carries baseline noise that the one-sided rule settles; the headline is the evening window (§5.5).

### 4.2 Deviation identity and settled volume

Per world and study slot, in kWh:

```
deviation        D = B − M
true reduction   R = U − M
baseline effect  E = B − U
D = R + E        (exact, checked by the validator)
settled volume   V = mask_t × max(0, D_t)            (lead decision QB, review B7: V ≥ 0)
```

Settlement is gated on the observable deviation `D > 0` inside the settlement mask (lead decision QB). Only turn-down is settled; turn-up (the cheap charging leg) stays on the customer's tariff (decision 0005). Two rejected alternatives: settling every slot, which would charge the aggregator imbalance on the customer's own cheap charging; and gating on `R > 0`, which a settlement body cannot observe because `U` is a counterfactual that is never metered.

`R` and `E` are still reported: they split each settled slot into flexibility and baseline effect (§4.7).

### 4.3 Expected availability at a decision (no leakage)

For night `n` and decision instant `τ`, each EV's expected session uses only its typical clocks and parameters and the world's own warm-up history:

- `weight` = cohort `plug_probability` (the sampler's parameter, not the realised draw); 0 for EVs excluded by `include`.
- `start` = the cohort's typical plug-in clock on `D_n` (weekday or weekend window); `deadline` = typical departure on `D_n + 1` minus the departure margin, the same instant the smart planner targets (`action.expected_departures_utc_ns`). Both are converted to slot offsets inside the night and clipped to the night.
- `need_kwh` = `warmup_session_need_kwh[world, ev]`; if NaN, the mean over EVs of the same cohort with a warm-up session in that world; if none, 0. Capped at `slot_cap_kwh × (deadline − start)`. Night 0's day-ahead decision `τ_DA(0)` (13:00 on `D_0 − 1`) falls in warm-up night −1, so its need uses only the warm-up sessions whose first connected slot starts before `τ_DA(0)`, the same cut-off as the baseline's `n − 2` rule (review B3); a session already connected at the first warm-up slot is never counted, because its plug-in stock is unknown.
- `slot_cap_kwh` = home charging power × 0.5.
- Intraday (`decision_slot = s(τ)`, review B1): `start ← max(start, s(τ))`; a session with `deadline ≤ s(τ)` gets weight 0; the need cap uses the re-based window.

Limitations, stated in Limits:

- **Forecast upward bias.** Deterministic typical clocks put each cohort's expected arrivals into one slot, so the expected unmanaged import, and with it the forecast deviation `B − x`, is too peaky: too high in the typical arrival slots and too low beside them. The positive part then sells more turn-down in those peak slots than the fleet delivers there. Expected need also ignores early departures, which cut real delivery. Both bias the day-ahead position upward in peak slots; imbalance and intraday re-trading show it rather than hide it.
- **Always-plugged tail.** The always-plugged cohort (window 15:00 → 14:00, 1 % of the source fleet) ends its session two hours after the night ends. Its expected session is clipped at the night end, and the 12:00–14:00 tail of the previous session is not forecast in the next night.

### 4.4 Day-ahead position (F1)

Decision instant `τ_DA(n)`. Prices seen: the visible day-ahead price at `τ_DA` under the B4 rule (published prices for `D_n`; the typical shape for `D_n + 1`), plus the planner adjustments of events known at `τ_DA` for the session's zone. The committed volume is paid at the realised day-ahead clearing price (price-taker).

Expected smart import (lead decision QA):

```
plan_i  = action.plan_cheapest_slots(need_i, cap_i, window_price_i)     (+inf outside [start_i, deadline_i))
u_i     = full power from start_i until need_i is met                   (non-responder / unmanaged schedule)
x_t     = Σ_i w_i [ ρ u_it + (1 − ρ) plan_it ]                          (ρ = non-response probability, 0 in F1)
q_t     = mask_DA,t × max(0, B⁰_t − x_t)                                (DA position, turn-down only, QC)
```

**Changed by §9.2 (item 58, I4):** the day-ahead position is `q_t = c × mask_DA,t × max(0, B⁰_t − x_t)`, with `c` the commitment share (default 0.8); the remainder is left to intraday (`full`) or imbalance (`da_only`). `c = 1` reproduces the formula above. **Changed by §9.5 (I6):** `ρ` becomes a per-session probability `ρ_i` (the EV's cohort value, 1 for a control-group EV). **Changed by §10.1e:** `ρ_i = 1 − r_m (1 − π_m)` from the EV's manufacturer response rate and outage probability (1 for a control-group EV); the per-cohort values are retired. **Changed by §10.4:** with `trading.commitment_rule = "newsvendor"` the day-ahead position is `q = mask × max(0, F̂ + e(α))`, the point forecast plus a leave-one-week-out error quantile at level `α = clip(P_DA ÷ k, 0, 1)`, in place of `c × F̂`.

`mask_DA = trading_mask(..., known_at_utc=τ_DA(n))`. Why the per-session cheapest-slot rule and no LP (QA): each session's problem (linear cost, the same cap in every slot, one deadline) is solved exactly by filling its cheapest slots first, so the expectation of the smart fleet's import under typical sessions is exactly the weighted sum of per-session plans. It is the dispatcher's own rule and function, so the trader forecasts what the fleet will actually do; it is vectorised over worlds and EVs; it needs no solver, cannot fail to converge, and breaks price ties the same way as the dispatcher (earlier slot first). A fleet-aggregate LP was rejected: it pools one EV's spare power for another's need, so it plans a fleet the dispatcher never produces, and it adds a solver and tie ambiguity.

Positions forecast the settled deviation `D = B − M̂` (QC). `B⁰` is known at `τ_DA`, so the expected baseline effect is sold at day-ahead along with the flexibility; the ledger still reports it as its own bucket (§4.7).

**Worked example** (hand-checkable, one world, 4 slots, ρ = 0). Two sessions with weight 1, both present from slot 0: session A needs 6 kWh by the end of slot 3, session B needs 2 kWh by the end of slot 1; each can take 5 kWh per slot. Warm-up baseline `B⁰ = [7, 1, 0, 0]`.

| slot | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| visible price π (£/MWh) | 100 | 80 | 30 | 40 |
| plan A (cheapest of slots 0–3) | 0 | 0 | 5 | 1 |
| plan B (cheapest of slots 0–1) | 0 | 2 | 0 | 0 |
| x = plan A + plan B | 0 | 2 | 5 | 1 |
| q = max(0, B⁰ − x) | 7 | 0 | 0 | 0 |

Plan cost `(2×80 + 5×30 + 1×40)/1000 = £0.35`. DA position 7 kWh in slot 0, cash `7/1000 × 100 = £0.70`. Slot 0 then meters `M = 1` with `U = 7`:

- With `B = 7`: `D = 6`, `R = 6`, `E = 0`, `V = 6`. DA-only imbalance cash `(6 − 7)/1000 × SIP_0`; with `SIP_0 = £120/MWh`, −£0.12. Baseline effect `0 × 100/1000 = £0`, so the buckets are day-ahead revenue on flexibility £0.70, imbalance −£0.12, baseline effect £0, slot total £0.58.
- With `B = 8` (the warm-up over-states this slot): `D = 7`, `R = 6`, `E = 1`, `V = 7`, `f = q = 7`, so imbalance cash is 0 and slot cash is £0.70. Baseline effect `1 × 100/1000 = £0.10`; day-ahead revenue on flexibility `0.70 − 0.10 = £0.60`. The £0.10 was sold at day-ahead but earned by the baseline, not by moving charging.

### 4.5 Intraday re-positioning (F2)

Decision instants: every whole London hour `τ` from the night's start (12:00 `D_n`) until each slot's gate closure (slot start − 60 min). Earlier hours bring no new fleet information and are skipped; a price-only re-trade is out of scope. At each `τ` (review B1):

- Sessions not yet started at `τ` (`include`): `expected_sessions(..., decision_slot=s(τ))` re-bases their start to `max(start, s(τ))`, drops those whose deadline is at or before `s(τ)`, re-caps their need on the re-based window, and `expected_smart_kwh` plans them over `[s(τ), night end)` with prices visible at `τ` and events known at `τ`.
- Forecast metered import `M̂_t(τ)` = `book_kwh["selected"][world, τ, t − s(τ)]` (EVs plugged in at `τ`, with their plans in force) + the expected plan above.
- Target `f_t(τ) = mask_τ,t × max(0, B_t(τ) − M̂_t(τ))`, with `mask_τ = trading_mask(..., known_at_utc=τ)` and `B_t(τ)` = `B⁰` before 13:30 `D_n`, `B` from then on. Only open slots (`τ ≤ gate_t`) are re-positioned; closed slots keep their last position.
- Trade `Δ_t(τ) = f_t(τ) − f_t(τ_prev)`, with `f_t` before the first intraday decision equal to `q_t`. Price: the latest intraday update at or before `τ`, `intraday_path_gbp_per_mwh[world, t, h]` with `h = ceil((gate_t − τ) / 1 h)` (review B6: `ceil` picks an update that has already happened; `floor` would use a price from the future).
- The last target at or before gate closure is the final position `f_t`.

The dispatch never follows the trader in F1–F2; the trader only forecasts better. F3 (optional) adds a third kernel path whose planner ranks the latest intraday price.

Non-response (F2, §8 Q2, default 0.05, illustrative behaviour assumption): one uniform per (world, study night, EV) from `sample_non_response`. A home session belongs to the night of its plug-in slot (a session already plugged in at the study start uses night 0). It ignores its smart plan when its night's uniform is below the probability; its plan becomes "full power from plug-in" (the planner's flat-price trick of item 49). **Changed by §9.5 (item 58, I6):** the probability is the EV's cohort value, 1 for a hold-out control-group EV, and raised to a charger control outage's size for sessions it covers (§9.6); the uniform and its indexing are unchanged. The trader sees this once the EV is plugged in (it is in the book) and expects it at rate ρ before then. **Changed by §10.1e:** the probability is `1 − r_m` for a treated EV of manufacturer `m`, 1 on a night that maker's control is out, 1 for a control EV, still raised to a scripted control outage's size; the "full power from plug-in" is dispatched by the normal rule while the smart plan stays on record (`planned_home_import_kwh`).

Runtime: `expected_smart_kwh` is one `plan_cheapest_slots` call per (night, decision): 7 day-ahead calls over all (world, EV) sessions and up to about 160 intraday calls, each over only the sessions still to start at `τ` (weight-0 rows are dropped before the call, so late-evening calls are small). The F2 writer measures the cost per call and the total at 1,000 EVs × 100 worlds and reports both; item 33 allows the time. The full intraday path is held in memory only, about 10 MB for the study slots at 100 worlds (100 × 336 × 36 float64); the result stores `position_updates` for sampled worlds (§8 Q11).

### 4.6 Masks, strategies and imbalance

Requests pause trading in their windows so an event's delivery is paid once, by the event. Each decision uses the mask known at its instant (§2.4); settlement uses the full mask. A short-notice request (W4) announced after the day-ahead decision therefore leaves any day-ahead position in its window to be bought back intraday (`full`) or settled as imbalance (`da_only`). Zone-scope requests pause fleet-wide trading in their window: a v1 simplification, labelled as such (§8 Q8). **Changed by §10.5b:** blackout slots (`availability.blackout_windows`) are False in every mask, decision and settlement alike, because a blackout is known before the run.

| Strategy | DA position | Intraday | Final position `f` |
| --- | --- | --- | --- |
| `da_only` | `q` from §4.4 | none | `q` |
| `full` | `q` from §4.4 | §4.5 | last intraday target |
| `perfect_foresight` | `V_t` (knows the realised settled deviation), sold at day-ahead | none | `V_t` |

All three settle against the same `V`, `R`, `E` and prices.

### 4.7 Ledger buckets (decision 0005)

Per world, strategy and slot, the trading cash is `C_t = q_t P_DA,t + Σ_τ Δ_t(τ) P_ID,t,h + (V_t − f_t) SIP_t` (÷ 1000). The baseline effect is carved out of the day-ahead revenue at the day-ahead price: `B⁰` is known at `τ_DA`, so the expected baseline effect is what the day-ahead position sells (QC). Intraday and imbalance stay as actual cash. A proportional `R : E` split of each slot's cash was rejected: its share `E/D` is unbounded as `D → 0` (D = 0.01 kWh, E = 5 kWh gives offsetting buckets of about ±£150 on a slot worth pennies).

Per (world, strategy, night), sums over the night's slots, money in GBP:

| Column | Formula | Sign |
| --- | --- | --- |
| `day_ahead_revenue_gbp` | `Σ q_t P_DA,t / 1000 − baseline_effect_gbp`, labelled "day-ahead revenue on flexibility" | ± |
| `intraday_pnl_gbp` | `Σ Σ_τ Δ_t(τ) P_ID,t,h / 1000` (selling more turn-down earns, buying back pays) | ± |
| `trading_cost_gbp` | `−c_spread × Σ_τ Σ_t |Δ_t(τ)| / 1000` (intraday volume only; the DA auction has one clearing price; not split) | − |
| `imbalance_gbp` | `Σ (V_t − f_t) SIP_t / 1000` | ± |
| `baseline_effect_gbp` | `Σ_{V_t > 0} E_t P_DA,t / 1000`, shown separately so it is never mistaken for flexibility | ± |
| `grid_event_payment_gbp` | Payments of the requests whose window starts in this night (§2.4) | + |
| `supplier_compensation_gbp` | `−c_supplier × Σ V_t / 1000` (default £0/MWh: P415 mutualises it; ≤ 0 because `V ≥ 0`) | − |
| `customer_revenue_share_gbp` | `−share × max(0, gross)`, `gross` = the seven buckets above (per night row, §8 Q7; more generous to the customer than weekly settlement, and labelled so) | − |
| `unmet_charge_penalty_gbp` | `−c_unmet × unmet_charge_kwh`, `unmet_charge_kwh = max(0, S_selected − S_normal)` per world and night, `S_path` = Σ `departure_shortfall_kwh` of that path over the night's slots (review B3) | − |
| `net_gbp` | Exact sum of the nine buckets | ± |

`day_ahead_revenue_gbp + intraday_pnl_gbp + imbalance_gbp + baseline_effect_gbp = Σ C_t`, the whole trading cash. The unmet-charge penalty counts only the shortfall smart charging added over unmanaged charging, so a session too short to reach target on either path is not charged to the aggregator. The customer energy cost effect (decision 0004 item 4, B3) stays a separate illustrative figure outside the ledger. Views never add it to the trading net: the customer's cheaper charging and the aggregator's turn-down revenue come from the same shifted energy, so adding them would double-count it, and a caption on the Trading page says so.

**Shock attribution (item 56).** Every flexibility cash bucket is a sum over slots, so it splits exactly by the slot's shock kind (§0). The ledger carries "of which" columns for the three buckets shocks move:

```
day_ahead_revenue_gbp = day_ahead_revenue_known_gbp + day_ahead_revenue_surprise_gbp + day_ahead_revenue_none_gbp
intraday_pnl_gbp      = intraday_pnl_known_gbp      + intraday_pnl_surprise_gbp      + intraday_pnl_none_gbp
imbalance_gbp         = imbalance_known_gbp         + imbalance_surprise_gbp         + imbalance_none_gbp
```

The baseline effect is not split by shock kind: it is a property of the baseline, not of market news. A slot's intraday P&L is all trades for that delivery slot, so a re-trade made hours before a surprise is attributed to the surprise if the delivery slot overlaps it. That is the intended reading: "P&L on half-hours a surprise hit", not "P&L caused by the surprise" (§8 Q12). The causal figure is a Compare run with shock rates set to 0 on the same random futures. The "of which" columns are not buckets: `net_gbp` sums the nine buckets only.

### 4.8 Random-draw order after this contract

1. Population: cohort permutation → personal mileage multipliers → **zone permutation** (new, `vehicle_count` draws) → **control-group permutation** (changed by §9.5, item 58: `vehicle_count` draws, always taken, even with the control group off).
2. Evaluation worlds: trips → connections → temperatures (as A1 leaves them).
3. Market channels in the order the prices generator sets, including the item 56 shock candidates (a fixed number per world and night with uniform thinning, so editing a shock rate moves no other channel).
4. **Non-response uniforms** (new, F2): `world × study nights × vehicle_count`, always drawn.

Scripted events, baselines, expected plans, intraday and settlement draw nothing. So a Compare of "with vs without event", of any trading assumption, or of shock rates, runs on identical random futures; adding zones or non-response re-baselines seeded tests once (plan §3: regenerate pins deliberately).

**Changed by §10.0:** the manufacturer permutation closes the population draws; the week and day plug-in factors are drawn between the trips and the plug uniforms; the manufacturer outage uniforms close the run. The full order is in §10.0.

## 5. Result frames

All frames follow results-v2 §1 (UTC keys plus `interval_start_london`, world-first quantiles with `numpy.quantile(..., method="linear")`, default RangeIndex, `object` strings, `evidence_kind="illustrative_synthetic"`). Action-only frames are `None` on a no-action result. New `ForecastResult` fields: `events` (validated table, §2.1, empty when none on an action result; `None` on a no-action result, as §1.5), `zone_ids` (tuple), and the frames below.

### 5.1 `deviation_world_slot` (exact)

One row per `(world_id, slot_index)`, world-major; 336 × worlds rows.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `world_id`, `slot_index` | int64 | Keys |
| `interval_start_utc` | datetime64[ns, UTC] | |
| `interval_start_london` | datetime64[ns, Europe/London] | |
| `night_index` | int64 | 0–6 |
| `baseline_kwh`, `unmanaged_kwh`, `metered_kwh` | float64 | `B`, `U`, `M`, kWh per half-hour |
| `baseline_kw`, `unmanaged_kw`, `metered_kw` | float64 | The same ÷ 0.5 h, for the Position chart |
| `deviation_kwh`, `true_reduction_kwh`, `baseline_effect_kwh` | float64 | `D`, `R`, `E` |
| `settled_kwh` | float64 | `V` (§4.2), ≥ 0 |
| `settlement_open` | bool | Full settlement mask |
| `position_da_only_kwh`, `position_full_kwh`, `position_perfect_foresight_kwh` | float64 | Final position per strategy (`position_full_kwh` is NaN until F2) |
| `day_ahead_gbp_per_mwh`, `intraday_close_gbp_per_mwh`, `imbalance_gbp_per_mwh` | float64 | Prices after all shocks (stochastic and scripted) |
| `known_shock_gw`, `surprise_shock_gw` | float64 | Signed net-demand shock in the slot, stochastic plus scripted (GW) |
| `shock_kind` | object | `surprise`, `known` or `none` (§0) |
| `evidence_kind` | object | |

**Changed by §10.4:** two columns after `position_perfect_foresight_kwh`: `commit_level` (float64, the newsvendor level `α`; NaN in fixed-share mode) and `forecast_error_quantile_kwh` (float64, `e(α)`; NaN in fixed-share mode).

### 5.2 `position_updates` (exact, sampled worlds)

Only `sampled_world_ids` (at most 10), strategy `full`. One row per (world, slot, decision): the DA decision and every intraday decision.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `world_id`, `slot_index`, `night_index` | int64 | |
| `interval_start_utc`, `interval_start_london` | UTC, London | Slot |
| `decision_utc`, `decision_london` | UTC, London | `τ` |
| `stage` | object | `day_ahead` or `intraday` |
| `hours_to_gate_closure` | int64 | `h`; for the DA row, `ceil` of the hours from `τ_DA` to gate closure |
| `baseline_known_kwh`, `forecast_metered_kwh` | float64 | `B_t(τ)` as known at `τ`, `M̂_t(τ)` |
| `position_kwh` | float64 | Target after this decision |
| `trade_kwh` | float64 | `Δ` (for the DA row, equals `q_t`) |
| `price_gbp_per_mwh` | float64 | DA clearing price or intraday price at `h` |
| `evidence_kind` | object | |

Keys `(world_id, slot_index, decision_utc)` are unique. Before F2 the frame holds DA rows only.

### 5.3 `trading_ledger_world` (exact)

One row per `(world_id, strategy, night_index)`; strategies in `STRATEGIES` order. Columns: `world_id`, `strategy`, `night_index`, `night_start_local_date` (date), `day_label`, the ten money columns of §4.7 in order (nine buckets and net), the nine shock-attribution columns of §4.7 (`day_ahead_revenue_{known,surprise,none}_gbp`, `intraday_pnl_{…}_gbp`, `imbalance_{…}_gbp`), `settled_mwh_{known,surprise,none}`, then volumes in MWh per night: `sold_day_ahead_mwh` (Σq), `final_position_mwh` (Σf), `settled_mwh` (ΣV), `flexibility_mwh` (Σ_{V>0} R, the settled volume from real flexibility), `baseline_effect_mwh` (Σ_{V>0} E), `imbalance_mwh` (Σ(V − f)), `abs_imbalance_mwh` (Σ|V − f|), `intraday_traded_mwh` (Σ|Δ|), `event_delivered_mwh`, and `unmet_charge_kwh`, then `evidence_kind`. **Changed by §9.1 (item 58, I3):** two volume columns are added after `intraday_traded_mwh`: `delivered_within_final_mwh` (Σ min(V_t, f_t)) and `delivered_within_day_ahead_mwh` (Σ min(V_t, q_t)).

### 5.4 `trading_week_world` (exact)

The same columns without `night_index`, `night_start_local_date` and `day_label`, one row per `(world_id, strategy)`: each column is the sum of that world's seven night rows. It exists so no view sums rows.

### 5.5 `trading_ledger_summary`, `trading_kpis`, `trading_checks` (exact)

`trading_ledger_summary`: one row per `(strategy, night_index, metric)`; `night_index` is `Int64`, `<NA>` for the week (from `trading_week_world`), 0–6 for nights (from `trading_ledger_world`). `metric` runs over the money and volume columns of §5.3 in order. Columns: `strategy`, `night_index`, `metric`, `unit` (`GBP per night`/`GBP per week`, `MWh …`, `kWh …`), `world_count`, `mean`, `p10`, `p50`, `p90`. Means of the buckets add up to the mean net; quantiles do not, and the waterfall uses means and says so.

`trading_kpis`: one row per `(strategy, metric)`, columns `strategy`, `metric`, `unit`, `world_count`, `mean`, `p10`, `p50`, `p90`, `cvar5`:

| `metric` | `unit` | Per-world value |
| --- | --- | --- |
| `net_gbp_per_week` | `GBP per week` | `trading_week_world.net_gbp` |
| `net_gbp_per_ev_year` | `GBP per EV per year` | `net_gbp × 52 ÷ vehicle_count`, labelled "extrapolated from one simulated week" |
| `settled_mwh_per_week` | `MWh per week` | `settled_mwh` |
| `imbalance_volume_share` | `fraction` | `abs_imbalance_mwh ÷ settled_mwh` (NaN when 0) |
| `baseline_effect_share` | `fraction` | `baseline_effect_mwh ÷ settled_mwh` (NaN when 0; may be negative) |
| `event_delivered_mwh_per_week` | `MWh per week` | `event_delivered_mwh` |

`world_count` is the number of worlds whose value is not NaN; the statistics use only those. `cvar5` is the mean of those per-world values at or below their own 5th percentile (linear), for the two money rows; NaN otherwise. **Changed by §9.1 (item 58, I3):** further rows are added (§9.1), and `cvar5` is also filled for the new `GBP per week` rows (not for rates or fractions).

**Changed by decision 0004 item 67:** seven baseline-erosion rows follow the dispatch rows, identical on every strategy row (a property of the one settled fleet, like `day_ahead_spread_gbp_per_mwh`), `cvar5` NaN. Values are `Σ V × P_DA ÷ 1000` at the realised day-ahead price, against the default baseline (`V`) or the smart-night baseline (`V_flex`, §4.1):

| `metric` | `unit` | Per-world value |
| --- | --- | --- |
| `settled_value_gbp_per_week` | `GBP per week` | `Σ V P_DA ÷ 1000` over the week |
| `settled_mwh_eroded_per_week` | `MWh per week` | `Σ V_flex ÷ 1000` |
| `settled_value_eroded_gbp_per_week` | `GBP per week` | `Σ V_flex P_DA ÷ 1000` |
| `baseline_erosion_share` | `fraction` | `1 − eroded value ÷ default value` |
| `evening_settled_value_gbp_per_week` | `GBP per week` | as `settled_value_…`, evening slots only |
| `evening_settled_value_eroded_gbp_per_week` | `GBP per week` | as `settled_value_eroded_…`, evening slots only |
| `evening_baseline_erosion_share` | `fraction` | `1 − evening eroded value ÷ evening default value` |

Each share row's `mean` is one minus the ratio of mean values (as `capture_rate`); its quantiles and `world_count` cover the worlds with a positive default value. The evening is `summaries.EROSION_EVENING_WINDOW = ("16:00", "20:00")` London, by `local_time_label`: where smart charging moves load out of, and the window of the model-questions Q-4 evidence. It is a reporting window, deliberately not the availability product's `PRODUCT_WINDOWS["evening"]` (17:00–21:00, §10.5a), which sizes a sellable product rather than measuring the lost turn-down. The validator recomputes every row from `deviation_world_slot` and `study_slots` with the run's baseline assumption records.

`trading_checks`: one row per runtime check, columns `check_id`, `description`, `value` (float64), `tolerance` (float64), `passed` (bool). Rows:

- `buckets_sum_to_net`: max absolute residual over all ledger rows; tolerance 1e-9 × max(1, |net|).
- `perfect_foresight_zero_imbalance`: max |imbalance cash| for that strategy; tolerance 0.
- `deviation_identity`: max |D − R − E|; tolerance 1e-9 kWh.
- `mean_intraday_revision` (`full` only; **changed by §9.2**: with commitment share `c > 0` the revision is `f − q ÷ c`, the change in the forecast rather than the deliberately uncommitted remainder; NaN when `c = 0`; **changed by §10.4:** in newsvendor mode the revision is `f − F̂`, the change against the point forecast): mean over worlds of each world's mean `f − q` over open slots, with tolerance 3 standard errors of that mean across worlds (§8 Q6). Informative only, like `p10_ordering`: `passed` is always True and the description says "failure expected while the §4.3 DA forecast bias stands".
- `p10_ordering`: 1.0 when P10 net is PF ≥ full ≥ DA-only, else 0.0. Empirical and informative only: `passed` is always True and the description says "expected, not guaranteed" (review B5).

The P&L lens shows this table as the sanity check.

### 5.6 `event_response_bands` (exact)

One row per `(event_id, series, metric, slot_index)` for the slots from 4 slots (2 h) before the window start to 8 slots (4 h) after its end, clipped to the study. Columns: `event_id`, `event_type`, `scope`, `series` (`normal`, `selected`, `difference`, `market`), `metric`, `unit`, `slot_index`, `interval_start_utc`, `interval_start_london`, `relative_slot` (int64, 0 = first window slot), `in_window` (bool), `world_count`, `mean`, `p10`, `p50`, `p90`.

| `series` | `metric` | Per-world value (in scope) |
| --- | --- | --- |
| `normal`, `selected` | `home_import_kw` | Scope's home import on that path |
| `selected` | `baseline_kw` | Scope's adjusted baseline |
| `selected` | `delivered_kw` | `(B − M)` for turn_down, `(M − B)` for turn_up, ÷ 0.5 (can be negative) |
| `difference` | `home_import_kw` | Selected minus normal inside each world first; negative before or after a window is rebound |
| `market` | `price_increment_gbp_per_mwh` | Scripted price shocks: that shock's own increment, `f(ND) − f(ND − shock)` if known, `f(ND + S) − f(ND)` if a surprise |
| `market` | `day_ahead_gbp_per_mwh`, `imbalance_gbp_per_mwh` | Prices after all shocks |

### 5.6a Shock frames (item 56; action results)

**`market_shocks` (exact).** The generator's shock table plus the scripted shocks, one row per (world, shock), ordered by `world_id`, `start_slot_index`. Columns: `world_id`, `shock_id` (object; `s<world>-<n>` for stochastic, the `event_id` for scripted), `source` (`stochastic`/`scripted`), `shock_class` (`mild`, `big`, `scripted`), `direction` (`up`/`down`), `known_day_ahead` (bool), `start_slot_index` (int64, warm-up slots negative), `start_utc`, `start_london`, `duration_slots` (int64), `size_gw` (float64, > 0; the sign is `direction`), `in_study` (bool: overlaps a study slot), `peak_price_increment_gbp_per_mwh` (float64: the largest `|f(ND) − f(ND − shock)|` over its slots for known shocks, `|f(ND + S) − f(ND)|` for surprises; the increment of that shock alone), `evidence_kind="synthetic"`.

**`shock_summary` (exact).** One row per `(shock_class, known_day_ahead, direction)` present in the defaults' order (`mild`, `big`, `scripted`; known first; up first). Columns: the three keys, `world_count`, `count_per_week_p10`, `count_per_week_p50`, `count_per_week_p90` (per world, study shocks of that group, then quantiles; a world with none counts 0), `size_gw_p50`, `peak_price_increment_gbp_per_mwh_p50` (across all study shocks of the group, NaN if none). Consumer: Market "shocks this week" tiles and caption.

**`shock_attribution_summary` (exact).** One row per `(strategy, shock_kind, metric)`, `shock_kind` in `known`, `surprise`, `none`; `metric` in `day_ahead_revenue_gbp`, `intraday_pnl_gbp`, `imbalance_gbp`, `settled_mwh`. Columns: `strategy`, `shock_kind`, `metric`, `unit`, `world_count`, `mean`, `p10`, `p50`, `p90`, per world the weekly sum of the matching "of which" column, then quantiles. Means add up to the bucket mean; quantiles do not.

**Deferred to a later wave (§8 Q13): `known_shock_response_bands`.** An event study of the fleet's reaction to known shocks, aligned on each shock's own start because stochastic shocks fall at different times in each world. It is not part of v1; the sketch kept for that wave is: rows `(direction, relative_slot, metric)`, relative slots −4…+12, metrics `day_ahead_increment_gbp_per_mwh`, `selected_minus_normal_kw` (paired within the world first), `selected_kw`, `normal_kw`, with the spread across shocks (pooled over worlds) and a caption saying so. In v1 the fleet's reaction to a known shock is shown for scripted known shocks through `event_response_bands`.

### 5.7 `zone_import_bands` (exact, every result)

One row per `(zone_id, path_id, slot_index)`. Columns: `zone_id`, `zone_label`, `path_id`, `slot_index`, `interval_start_utc`, `interval_start_london`, `ev_count` (int64), `headroom_kw` (float64, NaN when off), `world_count`, `mean`, `p10`, `p50`, `p90` (zone `home_import_kw` across worlds), `above_headroom_world_share` (share of worlds whose zone import exceeds the headroom in that slot; NaN when off). `path_id` is `normal`, `selected`, plus `timed` (decision 0007, model step 2) on a run whose `timed_start_local_hour` is set, read from `zone_world_intervals`, whose kernel now tracks the timed path's zone sums too; absent on a run without it.

### 5.8 `zone_summary` (exact, every result)

One row per `(zone_id, path_id)`: `zone_id`, `zone_label`, `share` (assumption), `ev_count`, `headroom_kw`, `peak_kw_p10`, `peak_kw_p50`, `peak_kw_p90` (each world's own weekly peak of zone import, then quantiles, as `weekly_peak_summary`), `hours_above_headroom_p10`, `hours_above_headroom_p50`, `hours_above_headroom_p90` (per world, half-hours above headroom × 0.5; NaN when off). Same optional `timed` row as 5.7.

### 5.9 Validator rules (`validate_result_v2` additions)

- Column sets, order and dtypes exactly as above; keys unique; action-only frames `None` on no-action results; zone frames present on both.
- `deviation_world_slot`: `D − R − E` within 1e-9 kWh; `R = U − M` and `E = B − U` recomputed; kW = kWh ÷ 0.5; `settled_kwh = settlement_open × max(0, D)` and ≥ 0; positions ≥ 0; `position_perfect_foresight_kwh == settled_kwh`; `unmanaged_kwh` and `metered_kwh` equal `fleet_world_intervals` home import on the normal and selected paths.
- `trading_ledger_world`: `net_gbp` equals the nine-bucket sum within 1e-9 × max(1, |net|); the four cash buckets sum to `Σ C_t`; every bucket recomputed from `deviation_world_slot`, `position_updates` (sampled worlds) and the event payments; `perfect_foresight` rows have `imbalance_gbp == 0` exactly and `intraday_pnl_gbp == trading_cost_gbp == 0`; `da_only` rows have zero intraday and trading cost; `supplier_compensation_gbp ≤ 0`, `customer_revenue_share_gbp ≤ 0`, `unmet_charge_penalty_gbp ≤ 0`, `unmet_charge_kwh ≥ 0`; `grid_event_payment_gbp` and `unmet_charge_penalty_gbp` identical across strategies; `flexibility_mwh + baseline_effect_mwh == settled_mwh`.
- `trading_week_world` equals the per-world sum of `trading_ledger_world`; `trading_ledger_summary` and `trading_kpis` equal world-first recomputations; `cvar5 ≤ p10` where defined.
- `position_updates`: only sampled worlds; the DA row's `position_kwh` equals `position_da_only_kwh`; the last row per slot equals `position_full_kwh`; no `decision_utc` after gate closure; `trade_kwh` sums to the final position; `baseline_known_kwh` equals `B⁰` for decisions before 13:30 `D_n`, and (changed by §9.5, B3) `B⁰` of night `n` is recomputed from history nights `≤ n − 2` only, so no baseline reads a slot after `τ_DA(n)`.
- `event_response_bands`: ordered quantiles; one event block per enabled event; `relative_slot` consistent with `event_slots`.
- Shock attribution: in every ledger row each "of which" triple sums to its bucket within 1e-9 × max(1, |bucket|); `shock_kind` equals its recomputation from the two GW columns; `market_shocks` has one row per enabled scripted price shock per world, and its stochastic rows equal the generator's table; `shock_summary` and `shock_attribution_summary` equal recomputations.
- `zone_import_bands`: zone means add up to the fleet `home_import_kw` mean per path and slot (additive); `ev_count` adds up to `vehicle_count`; zone `ev_count` matches the shares by largest remainder.

### 5.10 Sanity-check tests (plan §2F, H2–H5), as a checklist

- [ ] No flex, no earnings: with non-response 1 (the selected path equals the normal path) or zero need, and a controlled fixture whose baseline equals the unmanaged import (`B = U`, `B⁰ = x` at every decision), every bucket is 0 for every strategy (review B4).
- [ ] Buckets sum to net on every ledger row (also a runtime check); the cash buckets sum to the trading cash.
- [ ] Perfect foresight has zero imbalance.
- [ ] Mean(final − DA position) ≈ 0: a hard pass only in a controlled fixture with no forecast error; in a real run it is the informative `mean_intraday_revision` row (3 standard errors across worlds), whose failure is expected while the §4.3 DA forecast bias stands (§8 Q6).
- [ ] In-day adjustment nearly inert (§8 Q3): at the default run shape, fleet import in 12:00–13:30 is close to 0, so `|a_n|` is small against the night's peak baseline (for example below 1 % of it); a controlled fixture with import in that window shows the adjustment shifting `B` by exactly `a_n`.
- [ ] With σ_ID = 0, no imbalance premium or tails, no spread, and surprise shocks off (stochastic surprise share 0 and no scripted surprise), so intraday = SIP = day-ahead, `full`, `da_only` and `perfect_foresight` give equal net per world (review B4).
- [ ] Strategy ordering (review B5): under single imbalance pricing, over- and under-delivery settle at the same price, so the mean ordering PF ≥ full ≥ DA-only is not expected and not tested. The P10 ordering is an empirical check at 200 EVs × 50 worlds, seed 42, reported in `trading_checks`; a failure is a finding, not a test failure.
- [ ] Editing a trading assumption (spread, revenue share, supplier compensation, penalty, baseline switch) changes no kernel frame and no price.
- [ ] The worked example of §4.4 (both baseline cases) as a unit test of `expected_smart_kwh`, `positions_kwh` and `settle`.
- [ ] `expected_smart_kwh` equals the weighted sum of `plan_cheapest_slots` per session, and with weight 1 and typical sessions equal to the realised sessions it reproduces the dispatcher's selected-path import.
- [ ] Leakage: perturbing anything after a decision instant (for example the sampled departures and plug-ins after `τ`) leaves `book_kwh` at `τ`, the expected plan at `τ` and the position target at `τ` unchanged; the same for `τ_DA` and the DA position. An event whose notice is after `τ` changes no mask or planner signal at `τ`.
- [ ] Intraday re-base (review B1): a session whose typical start has passed is planned from `s(τ)`; one whose deadline has passed is dropped; its need is capped on the re-based window.
- [ ] Intraday price index (review B6): the price used at `τ` is an update made at or before `τ`.
- [ ] Unmet-charge penalty (review B3): equal shortfall on both paths gives 0 penalty.
- [ ] Baseline (changed by §9.5, B3): a warm-up with constant import gives `B⁰` equal to it; night `n` uses only history nights `≤ n − 2`, the last at most five working or two non-working nights of its class; with a 3-day warm-up whose history lacks the night's class, the other-class fallback is used and counted in `baseline_class_fallback`; perturbing night `n − 1` leaves night `n`'s `B⁰` unchanged; a clock-change warm-up keeps the right local half-hours.
- [ ] Events: an empty table leaves every existing frame identical; a scripted known shock with no overlapping surprise moves day-ahead, intraday close and SIP by the same increment; a scripted surprise leaves day-ahead unchanged and moves only intraday steps after its reveal and SIP; the increment is larger when net demand is higher (convex curve); a turn-down preset lowers selected-path import in its window for EVs in scope and pays `delivered × payment`; turn-up raises it; an out-of-scope zone is unchanged; plans made before notice ignore the event.
- [ ] Shocks: with every shock rate 0 and no presets, `shock_kind` is `none` everywhere and all "of which" columns except `*_none_*` are 0; a scripted known shock and a stochastic known shock of the same size, time and class give identical prices; a scripted surprise leaves day-ahead and the smart plans unchanged; a scripted known up-shock placed in the overnight trough (for example 02:00–04:00) lowers fleet selected-path import in its window against the same seed without it, and leaves the normal path unchanged.
- [ ] Zones: exact counts by largest remainder; editing shares changes no other draw; zone sums add to fleet sums.
- [ ] Non-response 0 reproduces F1 exactly; non-response 1 makes the selected path equal to the normal path; the uniform is taken per (world, study night, EV) whatever the probability.

## 6. UI consumption map

Views read these fields and plot or tabulate them; they never compute positions, deviations or money.

| View | Reads |
| --- | --- |
| Trading ▸ 1 Market | `forecast_price_bands` (DA median, P10–P90), the B1/B5 market price frames for `representative_world_id` (intraday close, SIP, daily level), the negative-price share from the prices contract, and the supply curve points from the H1 contract (normal vs shocked net demand). **Shock marks** on the representative-week price chart from `market_shocks` (that world, `in_study`): one shaded vertical span per shock over its slots, fill by known (solid, lighter) vs surprise (hatched or outlined), an up or down marker at its start, big and scripted shocks labelled and mild ones unlabelled; hover gives class, direction, GW and peak price increment. Legend keys "Known day-ahead shock" and "Surprise shock". Tiles and caption from `shock_summary` (shocks per week by class, P50 with P10–P90). Scripted events also come from `events`; `event_response_bands` (`series="market"`) gives their price increment |
| Trading ▸ 2 Position | `deviation_world_slot` for `representative_world_id` on one chosen night: `baseline_kw`, `unmanaged_kw`, `metered_kw`, sold turn-down shaded from `position_*_kwh`; `position_updates` (position fan from DA to final, with intraday price); `trading_ledger_summary` rows `settled_mwh` per night (P10–P90 across weeks). **Item 67:** the baseline-erosion section from the `trading_kpis` `full` rows: headline tile `evening_baseline_erosion_share` (mean ratio), then the evening default and smart-night settled values (P50, P10–P90); the whole-week value, smart-night value and share in a "Whole week" expander, with one sentence that outside the evening the smart-night baseline is noisy in a 7-night study and adds spurious settlement |
| Trading ▸ 3 P&L and risk | `trading_week_world` (weekly net per strategy as strip or box); `trading_ledger_summary` week rows (`mean` waterfall per strategy, caption: means add up, quantiles do not); `trading_kpis` tiles (net P50, net P10, CVaR5, £/EV/yr), baseline-effect and imbalance shares; `trading_checks` table; the double-counting caption of §4.7 |
| Trading ▸ 3 P&L and risk, shock split | `shock_attribution_summary`: per strategy, stacked bars of `intraday_pnl_gbp` and `imbalance_gbp` means by shock kind, caption "P&L on half-hours a surprise shock hit, not caused by it"; P50 with P10–P90 in the table |
| Smart charging ▸ Response, event view | `events`, `event_response_bands` (`normal`, `selected`, `difference`; `baseline_kw`, `delivered_kw`; rebound after the window), payment from `trading_ledger_summary` (`grid_event_payment_gbp`). Scripted known price shocks show the fleet's reaction to a known shock; the stochastic-shock event study is deferred (§5.6a) |
| Drivers ▸ Fleet week, zone view | `zone_import_bands` (zone import vs headroom, `path_id` both), `zone_summary` (peaks, hours above headroom, EV counts); the Group selector is not used here |
| Overview ▸ Key stats, section F | `trading_kpis` for `full`: `net_gbp_per_week` (P50, P10, `cvar5`), `net_gbp_per_ev_year` (labelled extrapolated), `settled_mwh_per_week`, `imbalance_volume_share`, `baseline_effect_share` |
| Compare | Existing slim records plus `trading_week_world` and `trading_kpis` (paired by world, same random futures) |
| Edit assumptions | `events` editor with preset buttons; `TRADING`, `EVENTS`, `ZONES` groups |

Every £ label reads "illustrative simulated trading P&L" or "illustrative"; never "Axle cash" (decisions 0003, 0005).

**Changed by §9.8 (item 58):** the Supplier page, the trader-metric rows on Trading and the new Key stats rows are mapped in §9.8.

## 7. Assumption records (`model/assumptions.py`)

| Name | Default | Unit | Evidence | Authority |
| --- | --- | --- | --- | --- |
| `trading.half_spread_gbp_per_mwh` | 1.0 | GBP/MWh | illustrative | plan D6 |
| `trading.customer_revenue_share` | 0.5 | fraction | illustrative | decision 0005, plan D6 |
| `trading.supplier_compensation_gbp_per_mwh` | 0.0 | GBP/MWh | illustrative | decision 0005 (P415 mutualised) |
| `trading.unmet_charge_penalty_gbp_per_kwh` | 0.79 | GBP/kWh | illustrative | decision 0005; §8 Q1 (the illustrative public rate of decision 0004 item 4) |
| `trading.baseline_in_day_adjustment` | on | switch | illustrative | plan F1 |
| `trading.baseline_working_nights`, `trading.baseline_non_working_nights` | 5, 2 | nights | illustrative | plan D5, F1 |
| `trading.baseline_adjustment_window_slots` | 3 | slots | illustrative | §8 Q3 |
| `trading.day_ahead_decision_local_time` | 13:00 on D−1 | London time | illustrative | plan F1 |
| `trading.gate_closure_minutes` | 60 | minutes | illustrative | plan B5 |
| `behaviour.non_response_probability` | 0.05 | fraction | illustrative | plan F2; §8 Q2. **Changed by §9.5:** replaced by one record per cohort (§9.9). **Changed by §10.1e:** the per-cohort records are retired in turn; non-response comes from `manufacturers.response_rate.<id>` (default 0.95 = 1 − 0.05) and `manufacturers.outage_probability_per_night.<id>` (§10.8) |
| `zones.share.zone_1`…`zone_4` | 0.25 each | fraction | illustrative | item 55, plan H5; §8 Q4 |
| `zones.headroom_kw.zone_1`…`zone_4` | NaN (off) | kW | illustrative | plan H5 |

Price-channel parameters (σ_ID, NIV premia, long probability, Student-t tails, shock rates and the surprise reveal lead) belong to the prices contract (B5, item 56).

## 8. Review outcome and lead decisions (29 September 2026)

Blocking review items, all applied: B1 intraday re-base (§4.3, §4.5); B2 no leakage from events (§2.1, §2.4, §4.6); B3 unmet-charge penalty on the added shortfall (§4.7); B4 test conditions (§5.10); B5 strategy ordering restated (§5.5, §5.10); B6 `ceil` intraday price index (§4.5); B7 settled volume ≥ 0 (§4.2).

Lead decisions:

- **QA.** No LP: the expected smart import is `action.plan_cheapest_slots` applied to the expected sessions, `x = Σ w_i [ρ u_i + (1 − ρ) plan_i]` (§4.4, with the reason). This replaces plan §2F's `linprog`.
- **QB.** Settlement is gated on the observable deviation `D > 0` inside the turn-down mask (§4.2).
- **QC.** Positions forecast the settled deviation `D = B − M̂`. `B⁰` is known at `τ_DA`, so the baseline effect is traded at day-ahead and still reported as its own bucket, carved out of day-ahead revenue at the day-ahead price, `Σ_{V>0} E · P_DA` (§4.7; the earlier `R : E` cash split was rejected on re-review as unbounded).
- **Q8.** Zone-scope requests pause fleet-wide trading in their window in v1, labelled as a simplification.

Questions Q1–Q7 and Q9–Q13, accepted with these resolutions:

1. Unmet-charge penalty £0.79/kWh (the illustrative public rate), illustrative.
2. Non-response probability 0.05, an illustrative behaviour assumption.
3. In-day adjustment window: the first 3 slots of each night (12:00–13:30). Fleet import is near zero then, so the adjustment is nearly inert; a test says so.
4. Zone shares 0.25 each.
5. One day-ahead decision per night at `τ_DA(n)`; the morning half is ranked on the typical shape until published (B4 rule), stated as a simplification.
6. `mean(final − DA)`: informative in real runs (3 standard errors across worlds; failure expected while the §4.3 DA forecast bias stands); a hard pass only in a controlled fixture.
7. Revenue share on `max(0, gross)` per night row, labelled "more generous to the customer than weekly settlement".
9. Names from waves 1–2 as in §1.4, confirmed or renamed by those lanes; one shared B4 visible-price rule.
10. Stochastic surprises are revealed a fixed illustrative lead (2 h) before their start, owned and made editable by the prices lane.
11. The full intraday path is held in memory only (about 10 MB for the study slots at 100 worlds); the result stores sampled worlds (`position_updates`).
12. Slot-overlap shock attribution, labelled "on half-hours a shock hit", with Compare at shock rates 0 for the causal figure.
13. `known_shock_response_bands` deferred to a later wave (§5.6a).

No open questions remain for v1. The item 58 additions in §9 were resolved by the lead in review (§9.10).

## 9. Supplier and trader additions (decision 0004 item 58, plan §8 I3–I7)

Status: draft for lead acceptance, 29 September 2026 (task W1-contract-supplier). Base `1140ef1`. Design only.

Authority: decision 0004 item 58 (B trader metrics, C commitment share, D Supplier page, E non-response by archetype, baseline trap and control group, F stress presets), plan §8 rows I3–I7. I1 (solar) and I2 (day-ahead = expected price given day-ahead information, premium default 0) belong to the prices lane; this section only consumes their result. Because of I2, mean intraday minus day-ahead is the premium (0 by default), so intraday P&L in v1 is rebalancing, not a price bet (item 58).

The rules of §1 and results-v2 §1 apply unchanged: plain functions over arrays and DataFrames, one physics run for all strategies, no decision uses information published after its decision instant, per-world values first and quantiles across worlds last (`numpy.quantile(..., method="linear")`), no view computes money, positions or physics. Every £ is illustrative; no figure here is "Axle cash".

Additions draw no new random number except the control-group permutation (§9.5), which is always drawn.

### 9.1 Trader metrics in `trading_kpis` (I3)

New rows are appended to `trading_kpis` (§5.5) after the existing six, one row per `(strategy, metric)` in `STRATEGIES` order, same columns. Each row's per-world value is computed inside the world first, from `trading_week_world` or the named frame, then the statistics across worlds. A NaN per-world value is left out (`world_count` counts the rest).

One helper quantity, not a ledger bucket:

```
flex_margin_gbp = day_ahead_revenue_gbp + intraday_pnl_gbp + trading_cost_gbp + imbalance_gbp     (per world, strategy, week)
```

It is the trading cash on flexibility after the spread cost. It leaves out the baseline effect (identical for the three strategies, and not flexibility), grid-event payments and the unmet-charge penalty (identical across strategies), and supplier compensation and the revenue share (commercial terms, not trading skill). So strategy differences in `flex_margin_gbp` are exactly the differences the trader controls.

| `metric` | `unit` | Per-world value (strategy `s`) | Notes |
| --- | --- | --- | --- |
| `flex_margin_gbp_per_week` | `GBP per week` | `flex_margin_gbp` of `s` | Exposed so the other rows can be read against it |
| `capture_rate` | `fraction` | `flex_margin(s) ÷ flex_margin(perfect_foresight)` in each world with a perfect-foresight margin > 0 | Lead decision (§9.10 Q12): `mean` is the **ratio of means**, `mean(flex_margin(s)) ÷ mean(flex_margin(perfect_foresight))` over all worlds, NaN if that denominator is ≤ 0; `p10`, `p50`, `p90` are quantiles of the per-world ratios over the worlds with a perfect-foresight margin > 0, and `world_count` is that number of worlds (the caption says so); `cvar5` NaN. 1.0 for `perfect_foresight` |
| `value_of_intraday_gbp_per_week` | `GBP per week` | `flex_margin(full) − flex_margin(da_only)`, paired within the world | On the `full` row only (a comparison, not a strategy property) |
| `cost_of_uncertainty_gbp_per_week` | `GBP per week` | `flex_margin(perfect_foresight) − flex_margin(s)`, paired within the world | 0 for `perfect_foresight` |
| `imbalance_gbp_per_mwh_traded` | `GBP per MWh` | `imbalance_gbp ÷ final_position_mwh` (NaN when 0) | Sign as the ledger: negative is a cost. 0 for `perfect_foresight` |
| `imbalance_share_of_traded` | `fraction` | `abs_imbalance_mwh ÷ final_position_mwh` (NaN when 0) | The existing `imbalance_volume_share` stays (denominator settled volume). For `da_only` it includes the uncommitted `(1 − c)` share settled at imbalance by design (§9.2); the caption says so |
| `firmness` | `fraction` | `delivered_within_final_mwh ÷ final_position_mwh` (NaN when 0) | Share of the final committed turn-down actually delivered; 1 for `perfect_foresight`. For `da_only` the final position is the `c` share only, so its firmness is measured against a deliberately smaller commitment; the caption says so |
| `firmness_day_ahead` | `fraction` | `delivered_within_day_ahead_mwh ÷ sold_day_ahead_mwh` (NaN when 0) | Against the day-ahead commitment `q`; equal for `da_only` and `full` (same `q`) |
| `flex_margin_gbp_per_mw_year` | `GBP per MW per year` | `flex_margin_gbp × 52 ÷ (K_w ÷ 1000)`, `K_w` = that world's `deferrable_at_least_2h_kw_1900` (results-v2 3.6e per-world value: mean over the week's 19:00 half-hours of unmanaged power that can wait at least 2 h, kW); NaN when `K_w = 0` | Lead decision (§9.10 Q11). Labelled "trading margin, excludes grid-event payments and the customer share; extrapolated from one simulated week, per MW of 19:00 capacity that can wait at least 2 hours". The denominator is a property of the fleet, the same for every strategy |
| `day_ahead_spread_gbp_per_mwh` | `GBP per MWh` | Mean over the seven session nights of (max − min) of the day-ahead price over the night's slots | A market property: identical in the three strategy rows (the validator checks it). Session night, not calendar day, because the evening peak and overnight trough EVs arbitrage lie in one noon-to-noon night |

The existing `net_gbp_per_week` row already carries P10 (`p10`) and CVaR5 (`cvar5`); no separate P10 or CVaR5 metric is added. `cvar5` is filled only for money-per-period rows: the two existing money rows and the new `GBP per week` rows (`flex_margin_gbp_per_week`, `value_of_intraday_gbp_per_week`, `cost_of_uncertainty_gbp_per_week`); it is NaN for rates (£/MWh, £/MW/yr) and fractions (the §5.5 change).

**`open_position_profile` (exact, action results).** The trader's open position by time of day. One row per `(strategy, metric, local_half_hour)`, strategies in `STRATEGIES` order, metrics `open_position_kw` then `abs_open_position_kw`, half-hours in noon-to-noon order (`12:00` … `11:30`). Columns: `strategy`, `metric`, `unit` (`"kW"`), `local_half_hour` (object `"HH:MM"`, London), `profile_order` (int64 0–47), `world_count`, `mean`, `p10`, `p50`, `p90`. Per world: the mean over the week's study slots with that London label of `(V_t − f_t) ÷ 0.5` (signed: positive = delivered more turn-down than sold, spilled at SIP; negative = short) and of `|V_t − f_t| ÷ 0.5`. Both copies of a repeated autumn half-hour enter the mean; a label missing on a spring night uses the other nights. `perfect_foresight` rows are exactly 0.

### 9.2 Commitment share (I4)

Item 58 (C): the day-ahead position commits a share of the forecast deviation; the rest is traded intraday.

```
q_t = c × mask_DA,t × max(0, B⁰_t − x_t)          (0 ≤ c ≤ 1; §4.4 changed in place)
```

- `da_only` keeps `f = q`, so the uncommitted `(1 − c)` share is settled as imbalance. `full` re-positions from `q` exactly as §4.5. For a slot with **at least one intraday decision**, its intraday targets `f_t(τ)` do not depend on `c`, so its final position is the same for every `c`; only the first intraday trade `Δ = f_t(τ₁) − q_t` and so the price mix change. The first two slots of each night (12:00 and 12:30 London) have **no** intraday decision: decisions start at 12:00 and their gate closes at 11:00 and 11:30. There `full` keeps `f = q`, so `(1 − c)` of their target is settled at imbalance for `full` too; the Position caption says so. Home import is near zero at that time (§4.1), so the effect is small but not zero. `perfect_foresight` still sells `V_t` at day-ahead (it has nothing to be uncertain about).
- Where: `market.run_trading` multiplies the §4.4 target by `c` for the day-ahead row; `positions_kwh` is unchanged and still gives intraday targets.
- Parameter record: `trading.day_ahead_commitment_share`, default 0.8, fraction, valid 0–1, illustrative, authority decision 0004 item 58 (C), plan I4. Owner: the market lane (F1/F2 writer), in the `TRADING` group of `assumptions.py`.
- **Changed by §10.4:** a second rule, `trading.commitment_rule = "newsvendor"`, replaces `c × F̂` by the point forecast plus a leave-one-week-out forecast-error quantile at the level `P_DA ÷ k`; the fixed share stays the default and reproduces this subsection bit for bit.
- Tests: `c = 1` reproduces the v1 ledger bit for bit; `c = 0` gives `q = 0`, zero day-ahead revenue on flexibility and `da_only` imbalance volume equal to `V`; `q` scales linearly in `c`; `full` final positions are identical for `c = 0.5` and `c = 1` in every slot with at least one intraday decision, and equal `c × ` the §4.4 target in the 12:00 and 12:30 slots; `firmness_day_ahead` does not fall as `c` falls on a controlled fixture with no forecast error; the `mean_intraday_revision` check uses `f − q ÷ c`.

### 9.3 Supplier page frames (I5)

All are built in `model/summaries.py` from existing kernel and trading outputs, action results only (`None` on a no-action result). The per-EV frames use the existing per-EV chunk pass (`_per_ev_chunks`), which already reconciles with the fleet frames.

#### 9.3a `flex_cost_curve` (exact)

What turn-down a supplier could buy at a chosen London half-hour, and at what price. Item 58 (D): "MW available above each £/MWh at a chosen half-hour". Label (lead review B2): "turn-down in each simulated week's sessions, costed at day-ahead prices visible at the day-ahead decision (overnight half-hours on the typical shape)". It measures the simulated sessions as they happened, so it is a description of what was movable, not a forecast a supplier held at day-ahead.

Per world `w`, study night `n` and London half-hour `h` (slot `t` of night `n` with that label):

1. **Who can move.** Every EV `i` with normal-path (unmanaged) home import `m^N_it > 0` in `t`. Its power is `k_i = m^N_it ÷ 0.5` kW.
2. **Where it could go.** Candidate slots `t'` of night `n` after the EV's unmanaged charging block ends (the first slot after `t` in which its normal-path import is 0) and ending at or before `min(realised unplug, expected departure)`: the expected departure is results-v2 3.6c (typical departure minus the margin), the realised unplug is the session's own end in that world, and both are clipped to the night end. Capping at the realised unplug stops a slot the EV was no longer plugged in for from counting as available (lead review B2). The unmanaged block is contiguous from plug-in to target, so these are the slots with the EV's full charger free; the partial last slot of the block is not used (conservative). An EV with no candidate cannot move (excluded at every price).
3. **Prices seen.** The visible day-ahead price at `τ_DA(n)` under the B4 rule (published for `D_n`, the typical shape for `D_n + 1`), the same prices the day-ahead position uses (§9.10 Q2).
4. **Cost to move** (£/MWh): `cost_i = min_{t'} P̃_{t'} − P̃_t + r`, where `r` is the editable early-departure risk charge (§9.9, default £24/MWh, derived and rough). A negative cost means moving the charging saves money at day-ahead prices.
5. **Available at price `p`**: `A_{w,n,h}(p) = Σ_i k_i [cost_i ≤ p]`, kW.

Because the block end and the capped deadline are the same for every slot of one session's unmanaged block, one range minimum per session serves every half-hour of it (vectorised, no loop over half-hours).

The per-world value for `(day_type, h, p)` is the mean of `A` over the week's nights of that day type (and both copies of a repeated autumn half-hour); a world with no such night is NaN. Day type goes by `D_n`: `weekday` Monday–Friday, `weekend` Saturday–Sunday, `all`. Then P10/P50/P90 across worlds.

Rows: one per `(day_type, local_half_hour, metric, threshold_gbp_per_mwh)`, day types `all`, `weekday`, `weekend`; all 48 half-hours are built so the view's half-hour selector (default 18:00, §9.9) never recomputes. Metrics: `available_kw` for each threshold of the reporting grid (−£300 to +£300/MWh in £10 steps, 61 thresholds; lead decision, §9.10 Q13); `movable_kw` (threshold NaN: all EVs with a candidate) and `charging_kw` (threshold NaN: all unmanaged charging at `h`, the curve's ceiling). Columns: `day_type`, `local_half_hour`, `profile_order`, `metric`, `threshold_gbp_per_mwh` (float64), `unit` (`"kW"`), `world_count`, `mean`, `p10`, `p50`, `p90`, `risk_charge_gbp_per_mwh` (the assumption used), `evidence_kind`. About 3 × 48 × 63 ≈ 9,100 rows.

Invariants (validator): within a `(day_type, h)` block, each world's `available_kw` is non-decreasing in `p` and at most its `movable_kw`, which is at most its `charging_kw`; so the quantile columns are non-decreasing in `p` too. Raising `r` by a multiple of £10 shifts the curve right by exactly that amount on the grid. A fixture where an EV unplugs before its expected departure excludes the slots after its unplug.

#### 9.3b `shape_premium_summary` (exact)

The cost of the fleet's load shape against a flat block, unmanaged versus smart. Per world, over the 336 study slots, with `P_t` the day-ahead price after all shocks (and the user curve, §9.4) and `M^p_t` fleet home import on path `p`:

```
load_weighted_price_p = Σ_t M^p_t P_t ÷ Σ_t M^p_t            (NaN if no import)
baseload_price        = mean_t P_t
shape_premium_p       = load_weighted_price_p − baseload_price  (£/MWh; positive = the load buys dearer than baseload)
shape_cost_p          = shape_premium_p × Σ_t M^p_t ÷ 1000      (£ per week)
```

Rows `(path_id, metric)`: `path_id` in `normal`, `selected`, `difference` (selected minus normal inside each world first); metrics `load_weighted_price_gbp_per_mwh`, `baseload_price_gbp_per_mwh` (identical on both paths; 0 on `difference`), `shape_premium_gbp_per_mwh`, `shape_cost_gbp_per_week`. Columns `path_id`, `metric`, `unit`, `world_count`, `mean`, `p10`, `p50`, `p90`, `evidence_kind`. The tariff leg is valued at day-ahead (B3), so `shape_cost_gbp_per_week` on `difference` is the home-import part of the customer energy cost effect minus the baseload value of the volume change; the validator checks that identity against `cost_effect` per world.

#### 9.3c `household_value_distribution` and `household_value_summary` (exact)

Value per household per month, and its spread across EVs. Per world `w` and treated EV `i` (control-group EVs, §9.5, are left out: they are not in the product):

- `saving_i` (£ per week) = minus the per-EV analogue of `cost_effect.illustrative_selected_minus_normal_total_gbp`: the EV's home import cost difference at day-ahead, public charge cost difference, unrecovered energy value and unserved travel value, each computed from that EV's own per-EV arrays. Summed over EVs, each component equals the world's `cost_effect` column (validator, 1e-6 £).
- `share_i` (£ per week) = `Σ_n −customer_revenue_share_gbp(w, full, n) × a_{i,n}`, the customer's revenue share from the `full` strategy allocated to EVs night by night with the allocation weight `a_{i,n}` below.
- `value_i` = `(saving_i + share_i) × 52 ÷ 12`, £ per household per month, labelled "extrapolated from one simulated week".

Allocation weight (also used by §9.3d): `a_{i,n} = g_{i,n} ÷ Σ_j g_{j,n}` with `g_{i,n} = Σ_{t ∈ n, V_t > 0} max(0, u_it − m_it)`, the EV's own positive true reduction (normal minus selected import) in the night's settled slots; equal shares over treated EVs when `Σ_j g_{j,n} = 0`. The weights sum to 1 per (world, night). It is an allocation of fleet settlement, not a per-household settlement, and is labelled so (§9.10 Q10).

`household_value_distribution`: one row per `(group_id, bin_index)`, `group_id` `fleet` then each cohort with treated EVs. Bins of £1 per month from −£20 to +£40 plus two open-ended outer bins (lead decision, §9.10 Q13). Columns `group_id`, `bin_index`, `bin_lower_gbp`, `bin_upper_gbp` (±inf for the outer bins), `world_count`, `mean`, `p10`, `p50`, `p90` of the per-world share of the group's EVs in the bin. Per-world shares sum to 1 over bins; quantiles do not, and the view draws P50 bars with P10–P90 whiskers and says so.

`household_value_summary`: one row per `(group_id, statistic)`, statistics `mean_value`, `p10_across_evs`, `p50_across_evs`, `p90_across_evs`, `share_worse_off` (EVs with `value_i < 0`), `mean_customer_saving`, `mean_revenue_share` (the last two monthly, adding to `mean_value` per world). Per world across the group's EVs first, then `mean`, `p10`, `p50`, `p90` across worlds. Columns `group_id`, `statistic`, `unit` (`GBP per household per month` or `fraction`), `ev_count`, `world_count`, `mean`, `p10`, `p50`, `p90`, `evidence_kind`.

Caption on every household-value view: "Customer value only. Do not add it to the aggregator's trading net: the customer's saving and the aggregator's turn-down revenue come from the same shifted energy" (the §4.7 double-counting rule).

#### 9.3d `revenue_by_segment` (exact)

Trading money by market bucket, archetype and zone. For strategy `full`, each ledger bucket of each (world, night) is allocated to treated EVs with `a_{i,n}` (§9.3c) and summed over the week by segment. Rows `(bucket, segment_type, segment_id, metric)`: `bucket` the nine buckets of §4.7 then `net_gbp`; `segment_type` `all` (one segment, `segment_id="all"`), then `cohort` (each cohort id with treated EVs), then `zone` (`zone_1`…`zone_4`); no cohort × zone cross. `metric` `gbp_per_week` and `gbp_per_ev_per_week` (÷ the segment's treated EV count, fixed across worlds; NaN for an empty segment). Columns `bucket`, `segment_type`, `segment_id`, `metric`, `unit`, `ev_count`, `world_count`, `mean`, `p10`, `p50`, `p90`, `evidence_kind`. Per world, the `all` segment equals `trading_week_world` for `full`, and the cohort segments and the zone segments each sum to it (validator; quantiles do not add). A penalty or imbalance caused by one EV is spread by the flexibility share, not traced to that EV; the caption says "allocated by each EV's share of settled flexibility".

#### 9.3e `supplier_positions` (exact) and the positions export

One row per study slot, for the positions table and the CSV download (the view writes this frame with `to_csv(index=False)` and adds nothing). Strategy `full`. Columns, in order:

| Column | Dtype | Meaning |
| --- | --- | --- |
| `slot_index` | int64 | |
| `interval_start_utc`, `interval_start_london` | UTC, London | |
| `settlement_date` | object `YYYY-MM-DD` | London calendar date of the slot |
| `settlement_period` | int64 | 1-based London half-hour of that date (1–46 / 48 / 50 on clock-change dates) |
| `night_index` | int64 | 0–6 |
| `day_ahead_position_mw_p10`, `_p50`, `_p90` | float64 | `q_t ÷ 0.5 ÷ 1000` across worlds |
| `final_position_mw_p10`, `_p50`, `_p90` | float64 | `f_t` of `full`, same conversion |
| `baseline_mw_p50`, `metered_mw_p50`, `settled_mw_p50` | float64 | `B`, `M`, `V` |
| `day_ahead_price_gbp_per_mwh_p10`, `_p50`, `_p90` | float64 | Day-ahead price after all shocks and the user curve |
| `price_curve_source` | object | `"synthetic"` or `"user curve"` (§9.4) |
| `evidence_kind` | object | `"illustrative_synthetic"` |

Download file name `axle_positions_<study start date>_seed<seed>.csv`. Per-slot quantiles do not add across slots; the table caption says so and says every value is illustrative.

**Changed by §10.5f:** after `settled_mw_p50` the frame gains `unmanaged_mw_p50`, `net_change_mw_p10`, `net_change_mw_p50`, `net_change_mw_p90` (metered minus unmanaged, the net shape change with its rebound) and the 1-h firm figures `deliverable_turn_down_1h_mw_p10`, `deliverable_turn_up_1h_mw_p10`.

### 9.4 Uploaded price curve (I5)

Item 58 (D): "upload of the supplier's own 48-value price curve".

**Input.** A CSV (UTF-8, at most 10 kB) with exactly the header `half_hour_start,price_gbp_per_mwh` and 48 data rows: `half_hour_start` is `"HH:MM"` London wall clock, each of `00:00`, `00:30` … `23:30` exactly once (any order); `price_gbp_per_mwh` a finite decimal. Validation (`validate_user_price_curve`, prices module) returns a 48-row frame sorted by half-hour or raises `ValueError` naming the row and rule: wrong or extra columns, missing or repeated half-hours, non-numeric or non-finite values, values outside the prices lane's floor and cap (−£500 to +£4,000/MWh, §9.10 Q9).

**Where it lives.** An assumption `supplier.user_price_curve` (default none). The upload control is on Supplier ▸ 2 Positions; uploading only sets the draft (with inline validation errors) and a "Run to apply" note; it never starts a run. It is exported and imported with the other settings and checked again with the external inputs at Run (§1.5 step 1).

**What it does.** It replaces the typical daily shape of the market, not its uncertainty. For every warm-up and study slot `t` with London label `h` and day class `k` (as the prices generator's typical shape uses them):

```
offset_t = curve[h] − typical_shape_k[h]
DA_t  += offset_t;   intraday_path[t, all h] += offset_t;   SIP_t += offset_t
then clip each of them to [price_floor_gbp_per_mwh, price_cap_gbp_per_mwh]   (−500, 4000: the prices lane's records)
```

and the B4 rule ranks unpublished half-hours on `curve[h]` instead of the typical shape. The daily level walk, half-hour noise, shocks, intraday updates and imbalance premia stay on top, so worlds still differ. Intraday and imbalance move by the same offset so `ID − DA` and `SIP − DA` are unchanged except in slots where the re-clip binds (the I2 premium test still holds away from the cap and floor; shifting day-ahead alone would hand the trader a shape-dependent free spread). The re-clip uses the prices lane's own cap and floor records (`price_cap_gbp_per_mwh` 4000, `price_floor_gbp_per_mwh` −500), so a user curve can never produce a price the generator could not (lead review B4). Shock increments stay as the generator computed them on its own net demand (stated as a simplification). On a clock-change date the 46 or 50 local half-hours read their own label (a repeated autumn label uses its one value twice).

**Deterministic, no draws.** The offset is added after generation, so every random channel is identical with and without a curve and Compare pairs them. The result gains `price_curve_source` (`"synthetic"` or `"user curve"`) and `user_price_curve` (the validated frame or `None`); every price, cost and trading view shows "Day-ahead shape: user curve" when it is set.

**Tests.** A curve equal to the typical shape reproduces the run bit for bit; a constant offset curve shifts DA, every intraday step and SIP by exactly that offset and leaves `ID − DA` and `SIP − DA` unchanged in every slot where no price reaches the cap or floor; a curve that pushes a slot past the cap leaves every price there at the cap; draw counts are identical; the validator rejects 47 rows, a repeated `18:00`, `NaN`, an extra column and an out-of-bounds value; the clock-change mapping is checked on an autumn and a spring study.

### 9.5 Non-response by archetype, baseline trap and control group (I6)

**Non-response by archetype.** **Changed by §10.1e (item 59): the per-cohort records below are retired; non-response comes from each EV's charger manufacturer (response rate and per-night outage), with the same uniform and the same `max` rule, so this paragraph's draw and kernel wiring stand and only the source of `ρ` changes.** `behaviour.non_response_probability` becomes one record per cohort, `behaviour.non_response_probability.<cohort_id>`, each defaulting to the approved 0.05 (§8 Q2) until a source supports different values (§9.10 Q3). The draw is unchanged: one uniform per (world, study night, EV) from `sample_non_response`; a session ignores its plan when its uniform is below its effective probability

```
p_session = 1                                              if the EV is in the control group
          = max(ρ_cohort(i), control-outage size at its plug-in slot and zone)   otherwise (§9.6)
```

`action.SmartCharging.non_response` carries the per-EV probabilities (and the kernel the outage array); the trader's expected plan uses `ρ_i` per session (§4.4 change). Tests: all cohorts at 0.05 reproduce the single-value run exactly; a cohort at 1 makes only that cohort's selected import equal its normal import; the uniforms are identical whatever the probabilities.

**Baseline-trap switch.** `trading.baseline_history`, `"warmup_unmanaged"` (default, §4.1) or `"flexed_study_nights"`. The history rule is §4.1 as changed (B3), one rule for both modes: night `n` uses nights `≤ n − 2`; in trap mode study nights `0 … n − 2` enter on the selected (metered) path after the warm-up nights, with the same at-most-5/2 window and other-class fallback. What changes: only `B⁰` and so `B`, `D`, `E`, `V`, positions and the ledger; kernel frames, prices and plans are unchanged (the dispatch never reads the baseline), so on/off is a Compare on identical futures. Expected finding, to be reported and not tuned: from night 2 on (the first night whose history holds a study night) the baseline's evening falls and its overnight rises toward the flexed shape, so settled turn-down and the baseline effect shrink night by night and trading net falls. That is the BL01 decay the §4.1 finding describes, now measured. No new frame: the effect shows in `trading_ledger_summary` per night (`settled_mwh`, `baseline_effect_mwh`, `net_gbp`) and on the Position chart. Tests: trap mode on nights 0–1 equals the default (no study night yet in the history); a fixture with constant metered import `K` in the study nights moves `B⁰` of later nights toward `K` exactly by the window mean; no night's baseline reads a slot after `τ_DA(n)`.

**Hold-out control group.** A switch `trading.control_group` (default off) and a share `trading.control_group_share` (default 0.1, used only when the switch is on; lead decision §9.10 Q4). Assignment, stratified by cohort: the last population draw (§4.8 change), `rng.permutation(vehicle_count)`, always taken whether the switch is on or off. The control size `K = round(share × vehicle_count)` (half up) is split across cohorts by largest remainder on `K × cohort count ÷ vehicle_count` (ties to the earlier cohort in the defaults' order), giving `k_c`; the control group is the first `k_c` EVs of each cohort in the permutation order, `units.control_group` (bool). Control EVs are never flexed (`p_session = 1`): their selected import equals their normal import. They stay in the fleet baseline, metered import and settlement (so they dilute delivery), and the trader expects them at `ρ_i = 1`. They are left out of the household value frames (§9.3c) and get weight 0 in allocation.

The kernel adds control-group sums like zone sums: `warmup_control_home_import_kwh` (world, warm-up slot) and `control_home_import_kwh` (world, study slot), normal path (equal on both paths; the validator checks it). The baseline is linear in import, so the treated group's baseline is the fleet's minus the control's (`B_T = B − B_c`, `U_T = U − U_c`) and needs no extra scope.

`control_group_summary` (exact, `None` when the switch is off): per world and per night (and week), over the slots where the fleet settles (`V_t > 0`, the slots where a baseline error is paid):

```
estimated_bias_kw_per_ev = mean_t (B_c,t − U_c,t) ÷ (n_c × 0.5)     (U_c = M_c: control metered is its own unmanaged)
true_bias_kw_per_ev      = mean_t (B_T,t − U_T,t) ÷ (n_t × 0.5)     (U_T known only inside the simulation)
estimation_error_kw_per_ev = estimated − true
```

Rows `(night_index, metric)`, `night_index` Int64 (0–6, `<NA>` for the week), metrics as above; columns `night_index`, `metric`, `unit`, `control_ev_count`, `treated_ev_count`, `world_count`, `mean`, `p10`, `p50`, `p90`, `evidence_kind`. A world or night with no settled slot is NaN. Expected finding (to confirm, not tune): with warm-up history the estimate tracks the true bias, with noise shrinking as the control group grows; in trap mode the control group's own history was never flexed, so it misses the trap and the error is about minus the true bias. The Limits note says using the control group as the counterfactual (P415 alternative methodology) is out of scope. Tests: switch off reproduces the run except the always-taken permutation; exact control count per cohort by largest remainder; control selected import equals normal; `B_T + B_c = B`; with a baseline equal to the unmanaged import on a fixture both biases are 0.

### 9.6 Stress presets (I7) and the charger control outage event

Presets are appended by `events.preset_rows(preset_id, study_slots) -> pd.DataFrame` (§1.1 change), which returns the rows for this study. The four §2.2 presets are unchanged (one row each). Added (values are lead decisions, §9.10 Q5–Q7, all illustrative):

| Preset id | Label | `event_type` | Rows | start, duration | `size` | `notice` | `scope` | `payment` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sunny_negative_weekend` | Sunny negative-price weekend | `price_shock_known` | One per Saturday and Sunday whose whole window lies inside the study (1 or 2 rows), each on the night whose `D_n` is the day before, so its start `< 12:00` falls on the Saturday or Sunday | 10:00, 360 min | −6.0 GW (placeholder; the prices lane may retune it once the I1/H1 curve is calibrated) | `day_ahead` | `national` | NaN |
| `charger_control_outage` | Charger control outage: plans not followed | `control_outage` | One | night 3, 15:00, 720 min (15:00–03:00) | 0.5: share of covered sessions that do not follow their plan | `short`, 0 min | `national` | NaN |
| `cold_still_week` | Cold still week | `price_shock_known` | Seven, one per night 0–6, ids `cold_still_week_n0` … `cold_still_week_n6` | 16:30, 180 min each | +4.0 GW (the approved `cold_still_evening` values) | `day_ahead` | `national` | NaN |

The sunny weekend is a known surplus shock (solar forecasts are known day-ahead), so it enters the day-ahead price and plans rank it. Scripted shocks are not bound by the 15:00–09:00 window (§2.5). Few EVs are at home at weekend midday, so the expected finding is how little of the negative-price window the fleet can use; it is reported as found.

**`control_outage` type (§2.1, §2.3 change).** A control failure: chargers do not follow their plans, but plug-ins are still reported, so the book still sees them. `size_unit = "fraction"`; `size` in (0, 1]; `payment_gbp_per_mwh` NaN; `notice = "short"` with `notice_minutes = 0`; `scope` `national` or a zone; it may overlap any event. It is allowed before H6 because it is not a request. Effect: a plan made in the window (at plug-in, or at a replan after an earlier window ends; lead decision after review, simpler than tracking each plan's plug-in slot and equal to it for the presets) whose EV is in scope uses `p_session = max(ρ_cohort, size)` (§9.5) with its existing uniform, so no draw moves and base-rate non-responders stay non-responders. It changes the selected path only; it does not pause trading or pay anything. The trader does not know it in advance: the day-ahead position and expected plans for sessions not yet started use `ρ_i`; once an affected EV is plugged in, the book (§4.5) holds its plug-in and its full-power charging, so intraday re-positioning sees it. `events.outage_probability(events, slots, zone_count, slot_count) -> (zone, slot) float64` gives the maximum outage size by plug-in slot (0 elsewhere). **Changed by §10.1e (R2):** the scripted control outage stacks on the random manufacturer outage of §10.1d through the same `max`, per plan made: a plan made in the window uses `p_session = max(ρ_{w,n,i}, size)` with the session's one uniform, where `ρ_{w,n,i}` is already 1 on a maker-outage night and `1 − r_m` otherwise; maker outage and base non-response are per session, the event's raise is per plan; neither is known to any day-ahead or intraday decision except through the book.

Tests: `sunny_negative_weekend` gives 1 or 2 rows by start weekday and every window inside the study; it lowers day-ahead prices in its window only; `charger_control_outage` with size 1 makes every covered in-scope session charge at full power and leaves sessions outside its window, the normal path and every draw unchanged; the DA position is identical with and without the outage.

### 9.7 Call order and draw order after §9

§1.5 is unchanged except: step 2 builds `units` with `control_group` after `zone_id`; step 5 adds the user-curve offset after generation (§9.4); step 7 passes per-EV non-response probabilities and the control-outage array; step 10 applies the commitment share and the baseline history switch; step 11 adds `open_position_profile`, `flex_cost_curve`, `shape_premium_summary`, `household_value_distribution`, `household_value_summary`, `revenue_by_segment`, `supplier_positions` and `control_group_summary`, and the new `trading_kpis` rows. Draw order: §4.8 with the control-group permutation last in the population. Adding it re-baselines seeded tests once; regenerate pins deliberately (plan §3). **Changed by §10.0:** the manufacturer permutation now closes the population, the plug-in factors sit between the trips and the plug uniforms, and the outage uniforms close the run.

### 9.8 UI consumption map (Supplier page, Trading metrics, Key stats)

Views read these frames and filter, plot or tabulate them; they compute no price, cost, position or allocation. Every £ is labelled illustrative; `price_curve_source = "user curve"` adds "Day-ahead shape: user curve".

| View | Reads |
| --- | --- |
| Supplier ▸ 1 Availability and cost curve | Turn-down availability: `deferrable_power_bands` `total` (normal path) in MW with P10–P90; turn-up: `flexibility_bands` `turn_up_headroom_kw` on the selected path (item 50: smart path only). Cost curve: `flex_cost_curve` for the chosen half-hour (selectbox, default 18:00) and the shared day-type control, `available_kw` P50 line with P10–P90 band against `threshold_gbp_per_mwh`, `charging_kw` as a dashed ceiling; caption (lead review B2): "Turn-down in each simulated week's sessions, costed at day-ahead prices visible at the day-ahead decision (overnight half-hours on the typical shape). Includes a £X/MWh early-departure risk charge (derived, rough)." |
| Supplier ▸ 2 Positions | `supplier_positions` table and CSV download (as stored); price-curve uploader writing only the draft `supplier.user_price_curve`, with validation errors inline and "Run to apply"; `open_position_profile` for `full` as a small chart |
| Supplier ▸ 3 Shape and value | `shape_premium_summary` (premium and weekly shape cost, unmanaged vs smart, paired difference); `household_value_distribution` (P50 bars, P10–P90 whiskers, Group selector over `group_id`) with `household_value_summary` tiles (median £/household/month, share worse off); `revenue_by_segment` as bars by bucket, with a control choosing `segment_type` (all, archetype, zone), caption on the allocation rule; the household caption of §9.3c |
| Trading ▸ 2 Position | Adds `open_position_profile` (open position by half-hour, per strategy) |
| Trading ▸ 3 P&L and risk | Adds a trader-metrics table from the §9.1 `trading_kpis` rows (per strategy: capture rate, value of intraday, cost of uncertainty, imbalance £/MWh and share of traded, firmness, firmness day-ahead, `flex_margin_gbp_per_mw_year` labelled "trading margin, excludes grid-event payments and the customer share; extrapolated", day-ahead spread); caption: for DA-only, imbalance and firmness include the uncommitted `(1 − c)` share by design |
| Trading ▸ 3 P&L and risk, control group | `control_group_summary` (estimated vs true bias per night) when present |
| Overview ▸ Key stats, section F (added rows) | `trading_kpis` `full`: `capture_rate`, `value_of_intraday_gbp_per_week`, `cost_of_uncertainty_gbp_per_week`, `imbalance_gbp_per_mwh_traded`, `firmness`, `flex_margin_gbp_per_mw_year` (trading margin, extrapolated), `day_ahead_spread_gbp_per_mwh` |
| Overview ▸ Key stats, new section G Supplier | `flex_cost_curve` `all`, 18:00: `available_kw` at £0/MWh and `charging_kw`; `shape_premium_summary` `shape_premium_gbp_per_mwh` (normal, selected, difference); `household_value_summary` `fleet` `mean_value` and `share_worse_off` |
| Compare | Existing records plus the new `trading_kpis` rows and `shape_premium_summary` (paired by world) |
| Edit assumptions | `trading.day_ahead_commitment_share`, `trading.baseline_history`, `trading.control_group` and `trading.control_group_share`, per-cohort non-response, `supplier.early_departure_risk_charge_gbp_per_mwh`; the new presets in the events editor |

IA after this: Overview · Drivers · Smart charging · Trading · Supplier · Compare · How it works (plan §8). **Changed by §10.10:** Supplier gains a fourth lens, "4 Firm MW"; the Trading P&L, Key stats and Edit assumptions additions are mapped in §10.10.

### 9.9 Assumption records added (`model/assumptions.py`)

| Name | Default | Unit | Evidence | Authority |
| --- | --- | --- | --- | --- |
| `trading.day_ahead_commitment_share` | 0.8 | fraction, 0–1 | illustrative | item 58 (C), plan I4 |
| `behaviour.non_response_probability.<cohort_id>` (six) | 0.05 each | fraction | illustrative | §8 Q2 value; item 58 (E); §9.10 Q3. **Changed by §10.1e:** retired; see the manufacturer records of §10.8 |
| `trading.baseline_history` | `warmup_unmanaged` | switch | illustrative | item 58 (E), plan I6 |
| `trading.control_group` | off | switch | illustrative | item 58 (E) "optional"; §9.10 Q4 |
| `trading.control_group_share` | 0.1 (used when the switch is on) | fraction, 0 < share < 1, at least one EV in each group | illustrative | §9.10 Q4 |
| `supplier.early_departure_risk_charge_gbp_per_mwh` | 24 | GBP/MWh | illustrative, derived and rough: about 3% early departures (item 51) × £790/MWh public charge rate | item 58 (D); §9.10 Q1 |
| `supplier.user_price_curve` | none | 48 × GBP/MWh | user-supplied | item 58 (D) |
| Events presets `sunny_negative_weekend`, `charger_control_outage`, `cold_still_week` | §9.6 | | illustrative | item 58 (F), plan I7; lead decisions §9.10 Q5–Q7 |

Reporting choices, not model assumptions (like results-v2's 25% threshold): the cost-curve default half-hour 18:00 (task), its threshold grid (−£300…+£300 in £10 steps) and the household value bins (£1 from −£20 to +£40 plus two open outer bins).

### 9.10 Questions resolved by the lead (review of `dd6f29d`)

All resolved; no open questions remain for §9.

1. **Early-departure risk charge:** £24/MWh, labelled "derived and rough" (about 3% early departures × the £790/MWh public charge rate).
2. **Cost-curve prices:** day-ahead prices visible at `τ_DA(n)`, overnight half-hours on the typical shape; candidates end at `min(realised unplug, expected departure)` (B2).
3. **Per-archetype non-response:** 0.05 for every cohort until a source differs.
4. **Control group:** a switch (default off) plus a share (default 0.1 when on), stratified by cohort by largest remainder, first `k_c` EVs of each cohort in the permutation order.
5. **Sunny weekend:** −6 GW placeholder, 10:00–16:00.
6. **Charger control outage:** night 3, 15:00–03:00, size 0.5; renamed from "telemetry outage" (Q-A: a control failure; the book still sees plug-ins).
7. **Cold still week:** seven `cold_still_evening`-valued rows with unique ids.
8. **Baseline cut-off:** one rule for both modes, history nights `≤ n − 2`, at most 5 working / 2 non-working, other-class fallback (§4.1 changed, B3).
9. **Price-curve bounds:** the prices lane's floor and cap (−£500, +£4,000/MWh); prices are re-clipped after the offset (B4).
10. **Revenue-share allocation:** by each EV's share of settled flexibility (§9.3c).
11. **£/MW/yr:** `flex_margin_gbp_per_mw_year`, labelled "trading margin, excludes grid-event payments and the customer share; extrapolated".
12. **Capture rate:** `mean` is the ratio of means (NaN if the denominator is ≤ 0); `p10`/`p50`/`p90` are per-world ratio quantiles over worlds with a positive perfect-foresight margin; `world_count` is that set; `cvar5` NaN.
13. **Reporting grids:** cost-curve thresholds −£300…+£300 in £10 steps; household bins £1 from −£20 to +£40 plus two open outer bins.

### 9.11 Sanity-check tests (checklist)

- [ ] `trading_kpis` new rows equal world-first recomputations from `trading_week_world`, `deviation_world_slot` and the per-world 19:00 deferrable power; `capture_rate` is 1 and `cost_of_uncertainty` 0 for `perfect_foresight`; `capture_rate.mean` equals the ratio of mean margins and its `world_count` the number of worlds with a positive perfect-foresight margin; `value_of_intraday` exists on the `full` row only; `cvar5` is NaN on rate and fraction rows; with σ_ID = 0, no premium, tails, spread or surprises, `value_of_intraday` and `cost_of_uncertainty` are 0 and `capture_rate` is 1 for every strategy (§5.10 equal-net case), for any commitment share.
- [ ] `firmness ≤ 1`, `firmness_day_ahead ≤ 1`, both 1 for `perfect_foresight`; `day_ahead_spread` identical across strategy rows and ≥ 0.
- [ ] `open_position_profile`: perfect-foresight rows 0; Σ over labels of (each world's signed profile × its slot count at that label × 0.5) equals its `imbalance_mwh` × 1000.
- [ ] Commitment share tests of §9.2.
- [ ] `flex_cost_curve` invariants of §9.3a; a hand fixture (one EV, three slots, known prices) gives the exact cost and kW; an EV with no free slot before its deadline is never available.
- [ ] `shape_premium_summary`: a flat price gives premium 0 on both paths; the identity with `cost_effect`.
- [ ] Household value: per-EV components reconcile with `cost_effect`; allocation weights sum to 1 per (world, night); allocated revenue share sums to the `full` bucket; control EVs are absent.
- [ ] `revenue_by_segment`: per world, the `all` segment equals `trading_week_world` (`full`), and cohort and zone segments each sum to it.
- [ ] `supplier_positions` quantiles equal recomputations from `deviation_world_slot` and `position_updates`; settlement periods run 1…N per London date with N = 46/48/50.
- [ ] User curve tests of §9.4; non-response, trap and control tests of §9.5; preset and outage tests of §9.6.
- [ ] Editing any §9 trading or supplier assumption (commitment share, baseline history, risk charge) changes no kernel frame and no price; the user curve and per-cohort non-response change no draw; the control share changes only which EVs are control.

## 10. Firm-MW availability with shared factors (decision 0004 items 59 and 60, plan §11)

Status: draft for lead acceptance, 29 September 2026 (task W2-contract-availability), revised after the lead's review of `d681279` (intraday whole-night redesign, blocking items B2–B7, the skip-share resolution of Q1, Q-a, Q-c, optionals O1, O2 and O10), Mike's decision 0004 item 60 (skip-share defaults) and the lead's review of `f9d0950` (`σ_day` recalibrated to the evening-availability correlation target, optionals O3–O9, the answers to the open questions). Revised again after the re-review of `a50a8c8` (R1: per-slot remaining need and the plan in force at the decision slot; R2: `p_session` and `plan_status` per plan made, as main's §9.6; Q2: tonight's maker outage reaches the late part; per-slot grids), on top of a branch-local merge of `main` at `8597ea6` (events and zones landed, decision items 61 and 62), and with Mike's decision 0004 item 64 (`σ_day = 0.5`) applied. Base `149a0d6`. Design only: no model code exists for anything below. Approved text changed by this section carries a "Changed by §10.x" note in place.

Authority: decision 0004 item 59 (shared factors, manufacturers, deliverable MW at two horizons, calibration backtest, newsvendor commitment, product sheet, blackout windows, price-weighted availability, settlement file, charge completion, net shape change, firmness by manufacturer, notebook 09); Mike's addition of 29 September 2026 (a week-level plug-in factor for parameter uncertainty; the across-week SD, implied correlation and effective fleet size per half-hour; a P90/P50-against-fleet-size study; herding and manufacturer diversification named as the other shared-factor sources); decision 0004 item 60 (Mike, 29 September 2026: the skip-share formulation, median skip about 12 % on an ordinary night, `σ_day` set so the plug-in rate of the always-plug cohorts has a relative day-to-day SD of about 15 %, `σ_week` about a third of it, and about 10–15 % fewer sessions accepted); decision 0004 item 61 (lead, 29 September 2026: `σ_day` calibrated so the implied correlation of evening turn-down availability is about 0.003, correcting the reading of item 60); decision 0004 item 63 (lead: the skip-share arithmetic) and item 64 (Mike, 29 September 2026: `σ_day = 0.5`, `σ_week` about a third; shared timing variation listed as a limitation); plan §11 (lanes J1–J7). It consumes the noon-to-noon horizon as landed on `claude/w2-noon-horizon` (`study_slots.night_index`, `local_date` = the session night, `SimulatedForecast.warmup_slots`, `action.visible_day_ahead_prices`), the trading overlay of §4–§5 and the §9 additions (commitment share, per-session non-response, control group, control-outage event, Supplier frames).

The rules of §1 and results-v2 §1 apply unchanged: plain functions over arrays and DataFrames, one `np.random.default_rng(seed)` per run passed to samplers, a fixed draw count per channel whatever the parameter values, per-world values first and quantiles across worlds last, no decision reads information published after its instant, no view computes physics, positions or money. Every MW, MWh and £ below is illustrative and synthetic; nothing is a bid, a settlement or Axle cash.

**Two P-value conventions.** This contract's `p10` is `numpy.quantile(..., 0.1)`: the value exceeded in 90 % of worlds. Trading language calls that figure the "P90" (90 % exceedance), and a firm product is often sold at 95 % exceedance, this contract's `p05`. Mike's "P90/P50" is therefore `p10 ÷ p50`, named `firm_share` in every frame below, with `firm_share_p05 = p05 ÷ p50` beside it, so no reader has to guess which convention a column uses.

### 10.0 Scope, modules, result fields, call order and draw order

**Why shared factors.** A firm-MW promise is only as honest as the tails of the fleet's deliverable MW. If every EV's plug-in were independent, the fleet's relative spread would fall as `1/√N` and a 10,000-EV fleet could sell nearly its median as firm. Real fleets share risks: a day when fewer people plug in, a plug-in rate that was itself estimated from a small sample, a charger maker whose cloud goes down, and the whole fleet ranking the same prices. Each of these makes EVs positively correlated, so the fleet variance has an `N²` term that does not diversify away:

```
Var(Σ_i X_i) = Σ_i Var(X_i) + Σ_{i≠j} Cov(X_i, X_j)          (exact)
             ≈ N p(1 − p) k² + N² k² Var(p_shared)             (N identical EVs, k kW each, plug-in probability p moved by a shared factor)
```

The four shared-factor sources in this model, and where each lives:

| Source | Mechanism | Where |
| --- | --- | --- |
| Day-level plug-in factor | One logit shock per (world, date) moves every EV's odds of **skipping** its plug-in that night | §10.1b, new |
| Week-level plug-in factor | One logit shock per world moves every EV's skip odds for the whole week: "the source plug-in rate is itself an estimate" | §10.1b, new |
| Manufacturer outage | One draw per (world, night, maker) removes a maker's whole slice from control that night | §10.1d, new |
| Tariff synchronisation (herding) | Every EV in a world ranks the same day-ahead price path, so plans coincide in the cheapest half-hours (decision 0004 item 48; measured by the coincidence factor, results-v2 4.5a) | Already modelled; the bands inherit it |

Temperature (one draw per world and date, feeding driving energy and prices) and the scripted control-outage event (§9.6) are shared too and stay as they are. The bands of §10.2 inherit all of them because they are quantiles of per-world fleet sums; the diagnostics of §10.2c (`sd`, `implied_rho`, `n_eff`, `firm_share`) make the correlation visible, and §10.6b's fleet-size frame shows it levelling off.

**Modules.**

| Module | Owns | Lane |
| --- | --- | --- |
| `sampling.py` | `sample_plug_in_factors`, the skip share inside `sample_connection_opportunities`, `assign_manufacturers` (last population draw), `sample_manufacturer_outages` (last channel of the run) | J1a |
| `assumptions.py` | `BEHAVIOUR` skip-share records, the two holiday switches, `MANUFACTURERS`, `AVAILABILITY` records, `trading.commitment_rule` (§10.8) | J1a |
| `clock.py` | `holiday_flags(settings, bank_holiday_monday, half_term_week) -> dict` with `holiday` and `skip_logit_shift` per sampled date (holidays act on the skip share only, §10.1c) | J1a |
| `forecast.py` | `SimulatedForecast` fields `plug_in_factors`, `manufacturer_outage`, `session_non_response_probability`; the `ForecastResult` fields below; call order | J1a (draw order, fields); J1b adds its kernel wiring after J1a merges; the lead adds one call line per later lane at integration |
| `action.py`, `physics.py` | `SmartCharging.non_response` as (world, night, EV); non-responders keep their smart plan on record and charge by the normal rule; per-EV outputs `planned_home_import_kwh`, `plan_remaining_need_kwh`, `decision_plan_kwh` and `plan_status`; blackout planning rule (§10.5b) | J1b |
| `events.py` | `trading_mask` reads the blackout slots; `validate_events` rejects a request wholly inside a blackout | J1b |
| `market.py` | Newsvendor commitment (§10.4) | J4 |
| `model/availability.py` (new) | Deliverable MW per world (realised, potential, known and late parts), the intraday whole-night distribution, `availability_world_slot`, `availability_bands`, `availability_manufacturer_world`, `firm_share_by_fleet_size`, `world_nights`; product window constants | J2 |
| `model/availability_backtest.py` (new) | `availability_backtest`, `availability_reliability`, `availability_backtest_summary` | J3 |
| `model/product.py` (new) | `product_sheet`, `availability_value_summary`, `settlement_file`, `charge_completion_summary`, `firmness_by_manufacturer`, `manufacturer_summary`; the `supplier_positions` columns of §10.5f | J5 |
| `summaries.py` | Two hook calls in the existing per-EV chunk loop (J2 first, J5 after J2 merges); `slim_run`/`compare_runs` records | J2, J5 |
| UI | Supplier ▸ 4 Firm MW lens, Trading P&L firmness block, Key stats section H, Edit assumptions tab | J6 |
| `notebooks/09_top_down_quantile_model.ipynb`, `pyproject.toml` dev group | Notebook 09 | J7 |

**New `ForecastResult` fields** (action results; `None` on a no-action result unless marked "every result"): `blackout_windows` (validated frame, every result), `world_nights`, `availability_world_slot`, `availability_bands`, `availability_manufacturer_world`, `firm_share_by_fleet_size`, `availability_backtest`, `availability_reliability`, `availability_backtest_summary`, `product_sheet`, `availability_value_summary`, `settlement_file` with the scalar `settlement_file_name` (str), `charge_completion_summary`, `firmness_by_manufacturer`, `manufacturer_summary`. Existing frames changed: `units` gains `manufacturer_id` (every result); `study_slots` and `warmup_slots` gain `holiday` and `blackout` (bool, every result); `deviation_world_slot` gains `commit_level` and `forecast_error_quantile_kwh` (§10.4); `supplier_positions` gains the columns of §10.5f. All new frames carry `evidence_kind="illustrative_synthetic"`.

**Call order** (§1.5 and §9.7 changed in place):

1. Validate external inputs once: settings, events, zones, trading assumptions, **blackout windows (`availability.validate_blackout_windows`), manufacturer shares, response rates and outage probabilities, the skip-share records**; the two holiday switches are booleans and need no validation.
2. `rng = np.random.default_rng(seed)`; `units = build_population(...)`: cohort permutation → mileage multipliers → zone permutation → control-group permutation → **manufacturer permutation**.
3. `flags = clock.holiday_flags(settings, ...)`; evaluation worlds: departures → trips → **`sample_plug_in_factors`** → connections (with the skip share of §10.1b, whose logit includes `flags["skip_logit_shift"]`) → temperatures. No sampler's clocks change on a holiday.
4. Scripted shock profiles; market channels (unchanged).
5. Non-response uniforms (F2, unchanged).
6. **`sample_manufacturer_outages`** (last draw of the run).
7. `session_non_response_probability` (world, night, EV) from the control group, the outages and the response rates (§10.1e); `smart = smart_charging_inputs(..., non_response=that array, blackout=study-and-warm-up blackout mask)`.
8. Kernel, both paths, with the per-EV outputs of §10.1e.
9. Cost effect; `market.run_trading(...)` with the commitment rule (§10.4) and blackout masks.
10. `build_summaries(...)`: the existing frames plus, in the per-EV chunk loop, `availability.accumulate(...)` and `product.accumulate(...)`; then `availability.build_frames(...)`, `availability_backtest.build_frames(...)`, `product.build_frames(...)`.

**Draw order after §10** (§4.8 changed in place):

1. Population: cohort permutation → personal mileage multipliers → zone permutation → control-group permutation → **manufacturer permutation** (`vehicle_count` draws, always taken, largest-remainder counts by share, §10.1d).
2. Evaluation worlds: departures → trips → **week factor (`world_count` standard normals) → day factor (`world_count × sampled_day_count` standard normals)** → plug uniforms → plug-in shift uniforms → temperatures.
3. Market channels (unchanged order and counts).
4. Non-response uniforms `world × 7 × vehicle_count` (unchanged).
5. **Manufacturer outage uniforms `world × 7 × 4`**, always drawn.

The two plug-in factors are drawn immediately before the plug uniforms they modify, because acceptance is decided inside `sample_connection_opportunities` and the alternative (append the factors at the end of the run and return the plug uniforms so acceptance can be re-decided in `forecast`) would split one decision across two modules. The consequence is that the plug, plug-in-shift, temperature, market and non-response draws move once: seeded pins are regenerated deliberately in J1a (plan §3), the same rule the zone and control-group permutations already follow. Nothing in §10.2–§10.6 draws: availability, its backtest, the newsvendor rule, the product frames and the fleet-size sub-sampling are all deterministic functions of the run.

### 10.1 Shared factors

#### 10.1a Notation

`W` worlds, `N` EVs, `M = 4` manufacturers. Sampled London dates `d = 0 … sampled_day_count − 1` (`settings.sampled_day_count`, warm-up dates first, then the study evenings, then the morning after the last night); the evening of session night `n` is date `d = warmup_days + n`. `p_{d,i}` is the cohort plug probability the connections sampler already uses for date `d` and EV `i` (weekday or weekend value where the cohort has one); `q_{w,d}` the shared **skip share** of §10.1b; `u^{plug}_{w,d,i}` the existing plug uniform; `u^{nr}_{w,n,i}` the F2 non-response uniform; `m(i)` the EV's manufacturer index; `r_m` a manufacturer's response rate and `π_m` its per-night outage probability (§10.1d).

#### 10.1b The skip share: day-level and week-level plug-in factors (decision 0004 item 60)

Five of the six source cohorts plug in with probability 1.0, so a shock on the logit of `p` itself could not move them (`logit(1) = +∞`). Item 60 resolves this (option A of the review): the source rates are "would plug in" rates, and a separate **skip share** `q_{w,d}` is the share of those sessions that do not happen on a given night, shared by every EV in the world that night and carrying both factors on its logit:

```
y_w      ~ N(0, 1)                                    one per world       (week factor: parameter uncertainty)
z_{w,d}  ~ N(0, 1)                                    one per (world, date)   (day factor)
logit q_{w,d} = μ_skip + σ_week y_w + σ_day z_{w,d} + h_d      (h_d the holiday shift of §10.1c, 0 on other dates)
p_eff_{w,d,i} = p_{d,i} × (1 − q_{w,d})                          clocked cohorts; the always-plugged cohort is excluded (its one session is the whole horizon)
accepted_{w,d,i} = u^{plug}_{w,d,i} < p_eff_{w,d,i}                 (the existing comparison and uniform)
```

`μ_skip = logit(median skip)`: the logit-normal is symmetric on the logit scale, so the record is the median skip on an ordinary night, not the mean. The shift multiplies the **odds of skipping** by `exp(·)`, so `q` stays in (0, 1) and a cohort with `p < 1` (infrequent charging, 0.2) is scaled the same way as the others: its realised mean plug-in falls from 0.2 to `0.2 × (1 − E[q])`, about 0.174 at the defaults, the same relative shift every clocked cohort takes (from 1.0 to about 0.87). That is the intended meaning and is not renormalised.

Defaults (decision 0004 items 60, 61, 63 and 64), all arithmetic on the logit-normal with `μ_skip = logit(0.12) = −1.992`:

| Record | Default | How it was set |
| --- | --- | --- |
| `behaviour.plug_in_skip_median` | 0.12 | Mike: about 12 % of would-be sessions skipped on an ordinary night (item 60) |
| `behaviour.plug_in_day_factor_sd` (`σ_day`) | 0.5 | Mike's choice (item 64): milder and more plausible for plug-in behaviour alone than the 1.0 that reproduces the source example's correlation exactly (item 63); the arithmetic and the alternatives below |
| `behaviour.plug_in_week_factor_sd` (`σ_week`) | 0.17 | About a third of `σ_day`, for estimation uncertainty in the rate |

The arithmetic, in four steps (items 61 and 63):

1. Mike's source example: a share of cars charging in one evening half-hour of 12 % with a day-to-day SD of 1.8 points (relative SD 15 %) has pairwise correlation `ρ = 0.018² ÷ (0.12 × 0.88) = 0.0031` and `n_eff = 10,000 ÷ (1 + 9,999 × 0.0031) ≈ 320` at 10,000 EVs.
2. In the skip model every share the skip drives has the same relative day-to-day SD, `relSD = SD(q) ÷ (1 − E[q])`, and an indicator with mean `π̄` driven by it has pairwise correlation `ρ = relSD² × π̄ ÷ (1 − π̄)`: one `σ` gives a small `ρ` for a rare indicator and a large one for a common indicator. `n_eff` (`1/ρ` in the limit) is therefore not comparable between indicators with different means; the relative SD is.
3. Setting `ρ = 0.003` at `π̄ = 0.12` gives `relSD = √(0.003 × 0.88 ÷ 0.12) = 0.148`, which the logit-normal reaches at `σ_day = 1.00` (0.105 at 0.75, 0.065 at 0.50): the value item 60's "15 % relative SD of the plug-in rate" also gave, so under a multiplicative skip the two readings of "±15 %" are one statement, and an `n_eff ≈ 8` is that same `σ` read through the whole-fleet plug-in indicator (mean 0.84), not a stronger factor. Mike chose the milder 0.5 (item 64), which puts the evening-availability correlation at about 0.0006 rather than 0.003.
4. What each `σ_day` implies at `N` = 1,000 and 10,000 (normal approximation, `p10 ÷ p50 ≈ 1 − 1.28 × CV`, `CV² = (1 − π̄)(1 + (N − 1)ρ) ÷ (N π̄)`; plug-in-driven variation only, response and outages come on top):

| `σ_day` | relSD of any skip-driven share | Evening availability (`π̄ = 0.12`): `ρ`; `n_eff` at 1k / 10k; P90/P50 at 1k / 10k | Whole-fleet plug-in count (`π̄ = 0.84`): `ρ`; `n_eff` at 1k / 10k; P90/P50 at 1k / 10k |
| --- | --- | --- | --- |
| 0.35 | 4.4 % | 0.0003; 790 / 2,760; 0.88 / 0.93 | 0.013; 69 / 74; 0.94 / 0.94 |
| **0.50 (default, item 64)** | **6.5 %** | **0.0006; 640 / 1,480; 0.86 / 0.91** | **0.028; 34 / 35; 0.92 / 0.92** |
| 0.75 | 10.5 % | 0.0015; 400 / 630; 0.83 / 0.86 | 0.066; 15 / 15; 0.87 / 0.87 |
| 1.00 (item 63: the exact match to the source example) | 14.8 % | 0.0030; 250 / 320; 0.78 / 0.81 | 0.119; 8 / 8; 0.81 / 0.81 |
| 1.25 | 19.4 % | 0.0051; 160 / 190; 0.73 / 0.75 | 0.179; 6 / 6; 0.75 / 0.75 |

So at the default the evening availability has `ρ ≈ 0.0006`, `n_eff ≈ 640` at 1,000 EVs and `≈ 1,500` at 10,000, with P90/P50 of about 0.86 and 0.91 from plug-in variation alone (response and outages come on top); the whole-fleet plug-in count has `ρ ≈ 0.028`, `n_eff ≈ 35` and P90/P50 ≈ 0.92 at either size; every skip-driven share varies about 6.5 % day to day. The table stays so a different `σ_day` is a one-number change. **Shared timing variation** (people plugging in earlier or later together on some days), which would move the evening share without moving the plug-in rate, is not modelled and is a stated limitation (item 64; the Limits note of the lens says so). Item 59's earlier phrase "about ±15 %, logit scale" (`σ = 0.15`) is superseded: it would give a relative SD of about 1.8 %.

What each factor does to the skip share and the plug-in rate of a `p = 1` cohort (median 0.12; a row is that factor at ±1 or ±2 standard deviations with the other factor at 0):

| Factor | Logit SD | Skip at −2 SD | Skip at −1 SD | Skip at +1 SD | Skip at +2 SD | Plug-in rate at ±1 SD |
| --- | --- | --- | --- | --- | --- | --- |
| Day, `σ_day = 0.5` (odds × 1.65 per SD) | 0.50 | 0.048 | 0.076 | 0.184 | 0.270 | 0.816 … 0.924 |
| Week, `σ_week = 0.17` (odds × 1.19 per SD) | 0.17 | 0.088 | 0.103 | 0.139 | 0.161 | 0.861 … 0.897 |
| Both | 0.53 | 0.045 | 0.074 | 0.188 | 0.282 | 0.812 … 0.926 |

A holiday shift of +0.30 (§10.1c) moves the median skip from 0.12 to 0.155 (plug-in rate 0.845 at the median) on top of both factors. Mean sessions: `E[q] = 0.130` with the day factor alone and 0.131 with both, so the clocked cohorts plug in on about 87 % of their would-be nights (a 13 % reduction, within Mike's accepted "about 10–15 % fewer sessions"); a +1 SD night has about 18 % skipping and a +2 SD night about 27 %.

Both factors behave like correlation. For EVs with the same `p`, `Corr(accepted_i, accepted_j) = Var(p_eff) ÷ (p̄(1 − p̄))` with `Var(p_eff) = p² Var(q)`, so the fleet plug-in count has variance `N p̄(1 − p̄) − N Var(p_eff) + N² Var(p_eff)`; the `N²` term is what `firm_share` (§10.2c) levels off on. The week factor is the parameter-uncertainty reading: every night of a simulated week shares one error in the rate. It is not a seasonal or trend term.

Draws: `sample_plug_in_factors(rng, world_count, day_count) -> tuple[np.ndarray, np.ndarray]` returns `y` (world,) and `z` (world, day), `stats.norm.rvs(size=..., random_state=rng)` in that order, one block each, whatever the SDs, the median or the holidays are (a zero SD still draws). `forecast.simulate_forecast` builds `q` and passes it as `sample_connection_opportunities(..., skip_share=q)` (a new (world, day) keyword, `None` = zeros = today's behaviour). `SimulatedForecast.plug_in_factors` keeps `{"week": y, "day": z, "skip_share": q}` for `world_nights` (§10.1f) and the notebook.

Tests (J1a): `skip_share=None` reproduces the previous run bit for bit apart from the moved draws (a pin regenerated once); the draw count is `W + W × days` whatever the values; the always-plugged cohort is never skipped; with one `p = 1` cohort, `σ_day = 1.0` (a fixture value, not the default), `σ_week = 0`, no holidays and 400 worlds, the across-world variance of a night's fleet plug-in count matches `N p̄(1 − p̄) + (N² − N) Var(p_eff)` within three standard errors, `E[q]` and `Var(q)` computed in the test by a 200,001-point quadrature over `z` of `expit(μ + σ z)`; the same at `p = 0.2` with `p_eff = 0.2 (1 − q)`; the week factor alone makes all seven nights of a world move together (the across-night mean of the count carries the `N²` term, the within-world night-to-night differences do not).

#### 10.1c Holiday presets (two switches, O1)

Two boolean records (§10.8), off by default, in place of an editable table: `behaviour.holiday_bank_holiday_monday` and `behaviour.holiday_half_term_week`. Holidays act on the plug-in skip share only (lead decision, 29 September 2026): they change no trip, departure clock, plug-in clock or planner deadline, so no holiday mask reaches any sampler, `action` or `market` (the B7 argument of the previous draft is not needed). `clock.holiday_flags(settings, *, bank_holiday_monday, half_term_week) -> dict[str, np.ndarray]` returns, per sampled date, `holiday` (bool) and `skip_logit_shift` `h_d` (float64):

| Switch | Dates | Skip-logit shift `h_d` |
| --- | --- | --- |
| `bank_holiday_monday` | The first Monday whose date is a study evening (`D_n`); if the study has no Monday evening the switch has no effect and the dialog says so | +0.30 (the half-term shift reused rather than a second value invented; §10.11 item 2) |
| `half_term_week` | Every study evening `D_0 … D_6` | +0.30 (odds of skipping × 1.35: median skip 0.155, plug-in about 84.5 % on an ordinary holiday night; lead review) |

`study_slots.holiday` (bool) marks the dates so a Sessions or Fleet-week chart can show "bank holiday" or "half-term" without re-keying every day-typed frame; `study_slots.day_type` stays weekday/weekend of the calendar date. The switches change no draw count and no draw position: they change the skip share a draw is compared with.

Tests (J1a): both switches off reproduce the run; `bank_holiday_monday` on a study with a Monday evening adds exactly 0.30 to that date's `skip_logit_shift` and nothing else changes (trips, clocks and plans identical); `half_term_week` adds exactly 0.30 to every study evening's shift and nothing to warm-up dates; a study without a Monday evening is unchanged with the switch on.

#### 10.1d Manufacturers

Four illustrative charger manufacturers `m1`…`m4`, labels "Maker A"…"Maker D" (no real brand, like the zones). Records per maker (§10.8): `share` (fractions summing to 1), `response_rate` `r_m` (fraction, the share of a maker's sessions that follow their plan on a normal night), `outage_probability_per_night` `π_m` (the chance the maker's cloud control is out for a whole session night).

Assignment: one draw at population build, the last draw of `build_population` after the control-group permutation (§4.8 changed): exact counts per maker by largest remainder on `vehicle_count × share` (ties to the lower index), then `rng.permutation` of the index array, the zone method (§3). Independent of cohort, zone and control group. `units` gains `manufacturer_id` (object) on every result; `manufacturer_index` (int64 0–3) is the kernel-side column.

Outages: `sample_manufacturer_outages(rng, world_count, night_count, manufacturer_count) -> np.ndarray` returns (world, night, maker) uniforms, `stats.uniform.rvs`, one block whatever the probabilities; `out_{w,n,m} = u < π_m`. Nights are study nights 0–6, the same indexing as the non-response uniforms (§4.5); the last warm-up night, where planning already runs (item 52), has no uniforms of its own: a plan made there belongs to night 0 and uses night 0's uniform and `ρ`, maker outage included, so a warm-up plan still in force at the study start already follows night 0's rule (Changed at the J1b review, W1: this keeps the kernel's existing night mapping rather than adding a warm-up exemption). An outage removes the maker's whole slice for the night: every home session of its EVs that belongs to night `n` (by plug-in slot) ignores its plan (§10.1e). Chargers still report plug-ins during an outage (as the §9.6 control outage): the book of §4.5 sees the sessions, only control is lost.

`SimulatedForecast.manufacturer_outage` keeps the (world, night, maker) bool array; `world_nights` (§10.1f) and `manufacturer_summary` (§10.5h) report it.

Tests (J1a): exact counts by largest remainder for shares (0.4, 0.3, 0.2, 0.1) at 10, 11 and 1,000 EVs; editing shares moves no other draw (the population's cohort, zone and control assignments and every world draw are identical); the outage block is drawn whatever the probabilities; `π = 1` for one maker makes every one of its sessions charge by the normal rule in every study night and leaves the other makers, the normal path and every draw unchanged (J1b); `π = 0` for all makers reproduces the base-rate run.

#### 10.1e Non-response becomes manufacturer-driven (replaces §9.5's per-cohort records)

Note (lead, 29 September 2026, decision 0004 item 68): a multi-day session (an EV that does not drive stays plugged in) takes each night's own non-response uniform and maker outage when its plan is re-made at the end of a window; its `plan_status` can therefore differ from night to night.

Item 59: "non-response comes from manufacturer response and outages". The six `behaviour.non_response_probability.<cohort_id>` records of §9.5 and §9.9 are retired (Changed by §10.1): two overlapping knobs for one observed thing (a session that does not follow its plan) would double count, the cohort values were the one approved 0.05 with no source behind a cohort split, and the maker split is what a supplier can diversify. A cohort-level term can return if a source supports it.

```
ρ_{w,n,i} = 1                                   control-group EV (§9.5)
          = 1                                   out_{w,n,m(i)}   (the maker is out that night)
          = 1 − r_{m(i)}                        otherwise                        (per session: one uniform u^{nr}_{w,n,i}, one night)
p_plan    = max(ρ_{w,n,i}, size)                for a plan made inside a covering control-outage window (main's §9.6: per plan made)
          = ρ_{w,n,i}                           for any other plan
ignores   = u^{nr}_{w,n,i} < p_plan             decided when each plan is made, with the session's one uniform
```

`forecast` builds `session_non_response_probability` (world, night, EV) float64 from the control-group column, the outage array and the response rates, and passes it as `SmartCharging.non_response` (Changed by §10.1: §9.5 said a per-EV vector; `for_slice` slices the world and EV axes). The kernel resolves `p_plan` when each plan is made (at plug-in and at every replan, main's §9.6 rule) and compares it with the session's one uniform, so maker outage and base non-response hold for the whole session and the scripted event's raise applies to the plans made in its window (R2). The one uniform per (world, night, EV) means: raising an outage probability or lowering a response rate turns responders into non-responders without ever turning a non-responder back, so a Compare of two maker settings runs on identical futures.

Kernel changes that §10.2 needs (J1b; `UNIT_INTERVAL_QUANTITIES` gains the three outputs; the fleet frames are unchanged):

- **Non-responders keep their smart plan on record and charge by the normal rule.** §4.5 described a non-responder's plan as "full power from plug-in (the flat-price trick)". Instead the planner plans every session on prices as usual and stores the plan; a session that ignores it is dispatched by `_home_charge_kwh` (the normal rule) for its whole session. Same physics, and the aggregator's plan is what the potential and the intraday forecast of §10.2 need, because the aggregator does not know which EV will ignore it.
- Per-EV outputs on the selected path, shape (world, study slot, EV): `planned_home_import_kwh` (the plan in force's grid kWh for the slot, 0 with no plan; 0 on the normal path); `plan_remaining_need_kwh` (R1: the grid kWh the plan in force still has to deliver at the opening of the slot, the need that plan was made for minus that plan's kWh from its plan time to `s − 1`, floored at 0; 0 with no plan); `decision_plan_kwh` (R1: for slot `s` of night `n`, the kWh for `s` of the plan that was in force at the night's decision slot `s_n`, for an EV plugged in at `s_n`; 0 for other EVs, for slots outside that plan's window and for slots before `s_n`; the kernel snapshots the plans at `s_n` and reads the snapshot at later slots, so a replan after `s_n` never changes it); and `plan_status` (float64 code of the plan in force in the slot: 0 follows, 1 base non-response, 2 maker outage, 3 control-outage event, 4 control group; 0 on the normal path and outside a session). The code is set when each plan is made (R2, main's §9.6), in this order: 4 if the EV is in the control group; else 1 if `u < 1 − r_m` (the session would ignore any plan); else 2 if its maker is out that night (a session that would otherwise have followed, so §10.2d can restore its slice); else 3 if this plan is made inside a covering control-outage window and `u < size`; else 0. Maker outage and base non-response are per session (one uniform, one night); the event's raise is per plan made, so a replan outside the window returns a code-3 session to 0. For a code-0 EV plugged in for the whole slot with a plan in force, `home_grid_import_kwh == min(planned, normal limit)` and the two differ only after a public top-up raised the stock (validator on the fixture: equal within 1e-9 kWh for every such cell).

Trader's expectation (§4.4 changed by §10.1): for a session not yet started, `ρ_i = 1 − r_{m(i)} (1 − π_{m(i)})` (non-response by either route) for a treated EV and 1 for a control EV; once plugged in, the book shows what the EV does.

Interaction with §9.6's scripted `control_outage` event: unchanged in form, `p_session = max(ρ, size)`, so it stacks on top of maker non-response; the maker outage is a random channel of every run, the event a scripted scenario. Neither is known day-ahead; neither is known to the intraday forecast at `τ` except through the book of sessions already plugged in.

Tests (J1a for the array, J1b for the kernel): `r = 1` for every maker and `π = 0` make the selected path equal the F1 run; `r = 0` makes the selected path equal the normal path; `plan_status` codes are 4 for control EVs, 1 when the uniform is below `1 − r` (on outage nights too), 2 for the other sessions of a maker on its outage night, 3 for a plan made inside a covering control-outage window with `1 − r ≤ u < size`, 0 otherwise, and a replan outside the window returns a code-3 session to 0; `planned_home_import_kwh` equals the import of a responding EV; a non-responder's `planned_home_import_kwh` is its smart plan, not its full-power import; `plan_remaining_need_kwh` falls by exactly the plan's kWh slot by slot and restarts at a replan; `decision_plan_kwh` equals `planned_home_import_kwh` for an EV plugged in at `s_n` that never replans; and a short-notice event announced after 17:00 (a control outage now, a request that replans plugged-in EVs once H6 lands) leaves `decision_plan_kwh` and the known part's inputs unchanged (R1): on the event's night and before it, `decision_plan_kwh`, `intraday_known_mean_kw`, `intraday_potential_kw` and `intraday_eligible_count` are exactly equal, and each `intraday_known_p*_kw` is within one grid bin of its exact value in each run (the per-slot grid's `x_max` also spans the late part, which the event moves, §10.2d).

#### 10.1f `world_nights` (exact, action results)

One row per (world_id, night_index), world-major: `world_id`, `night_index`, `night_start_local_date` (date), `day_type`, `holiday` (bool), `temperature_c` (that evening's sampled temperature), `solar_clearness` (the prices generator's daily clearness for the evening date; NaN if the generator has none), `plug_in_week_factor` (`y_w`, repeated), `plug_in_day_factor` (`z_{w,d}` of the evening date), `skip_logit_shift` (`h_d`), `plug_in_skip_share` (`q_{w,d}`), `outage_m1` … `outage_m4` (bool), `plugged_in_count_at_decision` (int64, EVs connected for the whole slot `s_n`, §10.2d), `plugged_in_share_at_decision` (÷ `N`), `evidence_kind`. `temperature_c` and `solar_clearness` are the sampled values: as features they are a **perfect weather forecast**, and the notebook says so (B4). It is what the Firm MW lens hovers ("this week: cold, maker B out on night 3, 38 % on the driveway at 17:00") and the feature table of notebook 09.

### 10.2 Deliverable MW

#### 10.2a Per-EV contributions: realised, potential, known and late

Defined on the **selected (smart) path**, the fleet the aggregator operates, for study slot `t`, EV `i`, world `w`, direction `dir ∈ {turn_down, turn_up}` and duration `D` in slots for `D_hours ∈ {0.5, 1, 2, 4}` (0.5 h is the single half-hour the product sheet's energy figure needs; 1, 2 and 4 h are item 59's). The window is `Wd = {t, …, t + D − 1}`; a window running past the study end is NaN.

| Symbol | Meaning |
| --- | --- |
| `k_i`, `c_i = k_i × 0.5` | Home charging power (kW) and the most one slot can import (kWh) |
| `m_{is}` | Selected-path home grid import in slot `s` (kWh, per-EV chunk pass) |
| `g_{is}`, `n_{is}` | The plan in force's kWh for slot `s` and its remaining need at the opening of `s` (`planned_home_import_kwh`, `plan_remaining_need_kwh`, §10.1e) |
| `need_{it}` | Realised: grid kWh still needed to reach the preferred target at the opening of `t`, `max(0, target_i − opening stock) ÷ η`. Under the plan: `n_{it}` |
| `E_{it}` | The deadline, as an exclusive slot index: `min(next expected departure after t (the planner's), realised session end)`, the §9.3a rule (lead review B2), clipped to the study end. `Ê_i` is the planner's expected departure alone |
| `after = max(0, E − (t + D))` | Slots left after the window before the deadline |
| plugged through | Connected for the whole of every slot in `Wd` (realised); under the plan, `t + D ≤ Ê_i` |
| follows | `plan_status_{w,t,i} = 0` (the session responds and its maker is online) |

```
turn-down:  a^TD_i(t, D) = min_{s ∈ Wd} m_{is} ÷ 0.5                      [kW]  if plugged through, follows and need_{it} ≤ c_i × after;   else 0
turn-up:    a^TU_i(t, D) = max(0, min( min_{s ∈ Wd} (c_i − m_{is}),  (need_{it} − Σ_{s ∈ Wd} m_{is}) ÷ D )) ÷ 0.5   [kW]  if plugged through and follows;   else 0
```

Turn-down is charging the plan does in every slot of the window that can pause for the whole window and still reach target by the deadline: the EV's own minimum over the window, so the fleet figure is a sum of independent per-EV terms (given the shared factors), which §10.2d needs. Turn-up is the extra power a plugged-in, below-target EV can absorb in every slot of the window: its per-slot headroom or its spare need spread over the window, whichever is smaller. Spare need counts only up to the preferred target SoC, never the physical ceiling (results-v2 §1): a turn-up call charges nothing the customer did not ask for, which is conservative (O9). Both are 0 for an EV that ignores its plan (it does not take instructions), for a control EV (never flexed), in any window that overlaps a blackout slot (§10.5b), and for the normal path (no plans).

Four variants of the same formulas, in one table:

| Quantity | Import and need | Connection and deadline | Response |
| --- | --- | --- | --- |
| Realised `a_i` | `m`, realised `need_{it}` | Realised; `E_{it}` | Follows only |
| Potential `ā_i` (B6) | `g`, `n` (the plan in force: a non-responder's realised import is full power, not what the aggregator could have dispatched) | Realised; `E_{it}` | Everyone (as if every maker were online and every session responded) |
| Plan-based `â_i` (the intraday known part, §10.2d) | The plan in force at `s_n`: `decision_plan_kwh` for `g`, and `n_{i s_n}` minus that plan's kWh from `s_n` to `t − 1` for the need (R1) | Under that plan; `Ê_i` | Random `R_i`, `O_m` |
| Late realised `a_i` for `i ∉ P_{w,n}` | As realised | As realised | Follows only |

Fleet sums, kW: `A = Σ_i a_i` (realised deliverable), `Ā = Σ_i ā_i` (potential), and the realised split at the night's decision slot `s_n` (§10.2d) into the **known** part `A^K = Σ_{i ∈ P_{w,n}} a_i` (EVs already plugged in at `s_n`) and the **late** part `A^L = Σ_{i ∉ P_{w,n}} a_i` (EVs that plug in after it), `A = A^K + A^L`. Also accumulated per slot: `eligible_power_kw = Σ_{i: a_i > 0} k_i` (the power that would resume after a turn-down call, §10.5a) and the per-maker realised and potential sums (§10.2e).

Implementation (J2): in the per-EV chunk loop, from the selected-path arrays `home_grid_import_kwh`, `opening_battery_kwh`, `connected`, `planned_home_import_kwh`, `plan_remaining_need_kwh`, `decision_plan_kwh`, `plan_status`, the deadline from `flexibility_inputs` and the realised session end from `connected`; sliding-window minima with `numpy.lib.stride_tricks.sliding_window_view(..., D, axis=1)`; about 30 MB per (10-world chunk, duration) at 1,000 EVs. Nothing is stored per EV except the across-world sums and sums of squares of §10.2c (336 × N × 8 × 2 float64, about 43 MB at 1,000 EVs, freed after the frames are built).

#### 10.2b Day-ahead horizon: the band across weeks

The day-ahead forecast for slot `t` is the distribution of `A_{·,t,D,dir}` across worlds: every simulated week is an exchangeable draw of what that night could be, the same reading as every P10/P50/P90 in this contract, and the backtest (§10.3) leaves the forecast week out. It uses nothing from the week being forecast, so it is a legitimate day-ahead figure even though it is built after the run.

#### 10.2c Correlation diagnostics (Mike's addition)

For the realised (day-ahead) rows, from the per-EV across-world variance `v_{i,t} = Var_w(a_i(w, t, D))` (ddof 1):

```
Var_w(A_t) = Σ_i v_{i,t} + ρ_t [ (Σ_i √v_{i,t})² − Σ_i v_{i,t} ]        (the equicorrelation that reproduces the fleet variance)
implied_rho_t = (Var_w(A_t) − Σ_i v_{i,t}) ÷ ((Σ_i √v_{i,t})² − Σ_i v_{i,t})      NaN when the denominator is 0
N_c,t = #{i : v_{i,t} > 0}                       contributing EVs (they vary across weeks at t)
n_eff_t = N_c,t ÷ (1 + (N_c,t − 1) × max(ρ_t, 0))
firm_share_t = p10_t ÷ p50_t,   firm_share_p05_t = p05_t ÷ p50_t        NaN when p50_t = 0
```

`implied_rho` is reported unclipped (sampling noise can make it slightly negative); `n_eff` uses the clipped value. With independent EVs `ρ → 0` and `firm_share → 1` as `N` grows; with shared factors `ρ` stays positive and `firm_share` levels off below 1. §10.6b's fleet-size frame shows both against `N`.

Validator and tests (J2), restated for the skip share: on a controlled fixture (one `p = 1` cohort, fixed clocks with zero scale so every accepted session plugs in at the same slot and leaves at the same slot, a need that fits after a 4-h pause, flat prices, `r = 1`, `π = 0`, no holidays, `σ_week = 0`, 400 worlds), at the plug-in slot and `D = 0.5 h`, `p_eff = 1 − q`: the across-world SD of `deliverable_kw` equals `k √(N p̄(1 − p̄) + (N² − N) Var(p_eff))` within three standard errors of a sample SD (`sd ÷ √(2(W − 1))`), `p̄ = 1 − E[q]` and `Var(p_eff) = Var(q)` from the quadrature of §10.1b; with `r < 1` the same holds with `p̄ r` in place of `p̄` and `Var(p_eff) r²` in place of `Var(p_eff)` (the "kW × response" of Mike's formula, exact because response is a second independent Bernoulli per EV); `implied_rho` on that fixture equals `Var(q) ÷ (p̄(1 − p̄))` within the same tolerance; with `σ_day = σ_week = 0`, `π = 0` and `r = 1`, `implied_rho` is 0 within tolerance and `1 − firm_share` falls with `N` (at `N` = 50, 200, 800 at one seed: `(1 − firm_share) × √N` stays within a factor 2 of its value at `N = 50`); with `σ_day = 1.0` on (a fixture value), `1 − firm_share` at `N = 800` is at least half of its value at `N = 200` (it levels off). The writer pins the exact tolerances on the fixture and reports the measured values in the handoff.

#### 10.2d Intraday horizon: the whole night from a 17:00 decision

`τ_n` is the intraday decision instant of night `n`: `availability.intraday_decision_local_time` (default "17:00") London on `D_n`; `s_n = s(τ_n)`. The forecast made at `τ_n` covers every slot `t ≥ s_n` of night `n` (slots before `s_n` are past: NaN) and is the distribution of the **whole-fleet** deliverable `A`, the same target as the day-ahead band, as the sum of two independent parts:

**Known part `K`.** The EVs connected for the whole slot `s_n` (the plugged-in set `P_{w,n}`): the aggregator knows each one's **plan in force at `s_n`** (`decision_plan_kwh`, §10.1e: a session replanned after `s_n` still counts with its last plan made at or before `s_n`, because that is all the aggregator held at the decision, R1), its remaining need under that plan at `s_n` (`plan_remaining_need_kwh[s_n]`; its stock at plug-in was reported by the charger: an assumption, stated in Limits) and its expected departure `Ê_i`, so its plan-based contribution `â_i(t, D)` (§10.2a) is known and only whether it will respond is not. `R_i ~ Bernoulli(r_{m(i)})` is whether the session **responds to a future dispatch call** in the window; the model has one response draw per session (a session that ignores its plan also ignores calls), but the forecast does not read the book of §4.5 to see who has already deviated, so `R_i` is distinct from the book's view of plan-following so far (Q-a; reading the book is a later refinement). `O_m ~ Bernoulli(π_m)` is whether maker `m` is out tonight. All independent:

```
K = Σ_m (1 − O_m) Σ_{i ∈ P ∩ m} â_i R_i
S_m = Σ_{i ∈ P ∩ m} â_i,   Q_m = Σ_{i ∈ P ∩ m} â_i²
for each outage state o ∈ {0,1}^M (16 states):  P(o) = Π_m π_m^{o_m} (1 − π_m)^{1 − o_m}
   μ_o = Σ_{m: o_m = 0} r_m S_m,      σ²_o = Σ_{m: o_m = 0} r_m (1 − r_m) Q_m
F_K(x) = Σ_o P(o) Φ((x − μ_o) ÷ σ_o)          (a point mass at μ_o when σ_o = 0);   E[K] = Σ_m (1 − π_m) r_m S_m
```

Exact over outage states; normal within a state for the weighted sum of independent response Bernoullis (accurate from a few dozen plugged-in EVs per maker and rough below, which the Limits note says).

**Expected (late) part `L`.** The sessions not yet plugged in at `s_n`. Their arrival, need and departure come from the assumption distributions: the cohort plug probability and the skip share, the plug-in clock conditional on not having arrived by `τ_n`, the departure clock, and the response draws. The simplest honest way to obtain that distribution without a new draw is to read it off the other simulated weeks: in world `w'`, the EVs not on the driveway at `s_n` are one realisation of exactly that conditional process. Because tonight's maker outage hits the late sessions of the same night as it hits `K` (Q2), the late part is stored **per maker with the outage slice restored**: `L_{w',m}(t, D) = Σ_{i ∉ P_{w',n}, m(i) = m} [ a_i if plan_status = 0;  ā_i if plan_status = 2 ]`, the maker's late EVs counted as they contributed, or at their potential `ā_i` (§10.2a: the plan in force, the remaining need and the realised connection; not `â_i`, which is 0 for EVs not plugged in at `s_n`) had their maker been online (code 2 is given only to a session that would otherwise have followed, §10.1e, so the restoration is exact). For each outage state `o` the late sample of world `w` is `{ Σ_{m: o_m = 0} L_{w',m} : w' ≠ w }`, uniform over the `W − 1` values: the other weeks' late sessions with tonight's outage state applied. One thing this ignores, stated in Limits: what `w`'s early arrivals say about `w`'s own day and week factors (a busy 17:00 driveway hints at a low-skip night), so the forecast is wider than a fully conditional one, which is conservative.

**The sum.** `A_τ = K + L` with the outage state shared between the parts: `F_A(x) = Σ_o P(o) × (1 ÷ (W − 1)) Σ_{w' ≠ w} Φ((x − Σ_{m: o_m = 0} L_{w',m} − μ_o) ÷ σ_o)`. Levels `L_q = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)` are read from this CDF on a fixed grid so the cost stays linear: one grid per (slot, direction, duration) of `G = 512` equal bins over `[0, x_max]`, `x_max` = the largest `Σ_m S_m + Σ_m L_{w',m}` over worlds at that slot (per-slot grids, so a quiet afternoon slot is not binned on the evening's range). Per outage state the known part's bin masses are the differences of `Φ((· − μ_o) ÷ σ_o)` at the bin edges (a point mass when `σ_o = 0`), the late part's are the leave-one-world-out histogram of that state's `W − 1` sums (the full-sample histogram once per slot minus world `w`'s own bin), the state's sum masses are their discrete convolution (a batched FFT, `scipy.signal.fftconvolve` along the last axis over all rows of a world chunk; per-row `numpy.convolve` is too slow at about 3.4 million convolutions per 100 worlds), and the mixture is the `P(o)`-weighted sum of the 16 states' masses; the CDF is its cumulative sum and a quantile is the grid value where it first reaches the level, linearly interpolated within the bin. Precision is one bin, `x_max ÷ 512` at that slot, a reporting precision the validator reproduces with the same grid. Means are exact: `E[K] = Σ_m (1 − π_m) r_m S_m`, `E[L] = Σ_m (1 − π_m) × mean_{w' ≠ w} L_{w',m}`, `E[A_τ] = E[K] + E[L]`. The known part's own quantiles come from the state mixture of its normals on the same grid, the late part's from the state mixture of its histograms, so `intraday_late_*` carries tonight's outage uncertainty too. The per-state normals and histograms are built per world chunk and discarded once the quantiles are written; nothing of them is stored except the quantile columns and the per-maker `L_{w,m}` audit columns of §10.2e. `availability.intraday_distribution(S, Q, r, pi, late_by_maker_loo, levels, grid)` is the public function (vectorised over leading axes), tested against a 20,000-draw Monte Carlo on a 300-EV fixture with a 30-world late sample (each quantile within one bin plus 2 % of the range), including a case with `π_m = 0.5` for one maker so the shared outage is exercised. A further test: a late EV of an out maker (plan_status 2) contributes its `ā_i`, not 0, to `late_restored_kw_m<k>`.

Cost: 16 states × 513 edges per (world, slot ≥ `s_n`, direction, duration), about 2 × 10⁹ `Φ` evaluations at 100 worlds, 16 convolutions of 512-point vectors per cell and one histogram per state and slot: tens of seconds at 1,000 EVs × 100 worlds, within item 33's allowance; the writer measures and reports it. A fixed-draw inner simulation was the alternative; it was rejected because its draws would either sit in a summary (the only sampler outside the run's channels) or need a `W × 7 × K × N` uniform block, and it would add Monte Carlo noise to the backtest.

Control EVs contribute 0 (known). A scripted control-outage event (§9.6) announced after `τ_n` is unknown at `τ_n` and is in neither part: `K` reads the plan in force at `s_n` (R1), so the event's replans leave `K` unchanged, and the backtest shows the miss. Blackout windows give 0.

#### 10.2e Frames

**`availability_world_slot` (at least).** One row per (world_id, slot_index, direction, duration_hours), world-major, directions `turn_down` then `turn_up`, durations 0.5, 1, 2, 4. Columns: `world_id`, `slot_index`, `night_index` (int64), `interval_start_utc`, `interval_start_london`, `direction` (object), `duration_hours` (float64), `deliverable_kw` (`A`), `deliverable_known_kw` (`A^K`), `deliverable_late_kw` (`A^L`), `potential_kw` (`Ā`), `eligible_power_kw`, `intraday_mean_kw` (`E[A_τ]`), `intraday_p05_kw`, `intraday_p10_kw`, `intraday_p25_kw`, `intraday_p50_kw`, `intraday_p75_kw`, `intraday_p90_kw`, `intraday_p95_kw`, `intraday_known_mean_kw`, `intraday_known_p05_kw`, `intraday_known_p10_kw`, `intraday_known_p25_kw`, `intraday_known_p50_kw`, `intraday_known_p75_kw`, `intraday_known_p90_kw`, `intraday_known_p95_kw` (all seven levels, so §10.3 can score the known part at each; lead ruling on J3 Q1), `intraday_late_mean_kw`, `intraday_late_p10_kw`, `intraday_late_p50_kw`, `intraday_late_p90_kw` (the late part with tonight's outage uncertainty, §10.2d), `late_restored_kw_m1` … `late_restored_kw_m4` (this world's `L_{w,m}`, audit columns the validator and the notebook recompute from), `intraday_potential_kw` (`Σ_m S_m`), `intraday_eligible_count` (int64, EVs in `P` with `â_i > 0`), `blackout` (bool), `evidence_kind`. About 269,000 rows at 100 worlds (about 70 MB with the known, late and per-maker columns; O4), kept because the backtest, the Position-style intraday chart and the settlement file read it. Validator (§10.9): `deliverable_kw = deliverable_known_kw + deliverable_late_kw`; `intraday_mean_kw = intraday_known_mean_kw + intraday_late_mean_kw`.

**`availability_manufacturer_world` (exact).** One row per (world_id, night_index, manufacturer_id, direction, duration_hours): `world_id`, `night_index`, `manufacturer_id`, `direction`, `duration_hours`, `ev_count` (int64), `outage` (bool), `realised_mwh` (Σ over the night's slots of the maker's `a_i` sums × 0.5 ÷ 1000), `potential_mwh` (same with `ā_i`), `evidence_kind`. 22,400 rows at 100 worlds. Consumer: §10.5g.

**`availability_bands` (exact).** One row per (horizon, statistic, direction, duration_hours, slot_index): `horizon` `day_ahead` (statistic `realised`) then `intraday` (statistics `conditional_p05`, `conditional_p10`, `conditional_p50`, `conditional_p90`, `conditional_known_p50`); `direction`, `duration_hours`, `slot_index`, `night_index`, `interval_start_utc`, `interval_start_london`, `unit` (`"kW"`), `world_count`, `mean`, `sd`, `p05`, `p10`, `p50`, `p90`, `firm_share`, `firm_share_p05`, `implied_rho`, `n_eff`, `contributing_ev_count` (int64), `blackout`, `evidence_kind`. Per world the value is `deliverable_kw` for `realised`, the matching `intraday_p*_kw` column for the conditional statistics and `intraday_known_p50_kw` for `conditional_known_p50`; then `mean`, `sd` (ddof 1), linear P05/P10/P50/P90 across worlds and the two firm shares. `implied_rho`, `n_eff` and `contributing_ev_count` are filled on `realised` rows only (§10.2c) and NaN on intraday rows (a conditional quantile has no per-EV decomposition). NaN slots (window past the end; intraday before `s_n`) give NaN statistics with `world_count` 0. 16,128 rows.

**`firm_share_by_fleet_size` (exact, Mike's addition).** How the firm figure changes with fleet size, from this run alone and at no extra simulation cost: the deliverable of the first `n` EVs in `units` order is `Σ_{i < n} a_i(w, t, D)`, and because cohort, zone, control and maker assignments are all permutations of the population, the first `n` units are a random sub-fleet with the same shared factors, the same prices and the same weather (EVs are physically independent given those, §1.4 and `KernelSlice`). One row per (fleet_size, window, direction, duration_hours): `fleet_size` (int64: each of 100, 300, 1,000, 3,000 that is at most `N`, plus `N` itself, ascending), `window` (`evening`, `overnight`, `morning`, §10.5a), `direction`, `duration_hours`, `ev_count_share` (`fleet_size ÷ N`), `world_count`, `mean_kw`, `sd_kw`, `p05_kw`, `p10_kw`, `p50_kw`, `p90_kw`, `firm_share`, `firm_share_p05`, `implied_rho`, `n_eff`, `evidence_kind`. Per world the value is the mean over the window's slots on all seven nights of the sub-fleet deliverable; `implied_rho` and `n_eff` from the per-EV window means as in §10.2c, over the first `n` EVs (cumulative sums over EVs in `units` order). Cost: one cumulative sum per (world, slot, direction, duration) over EVs in the chunk pass. Sizes above `N` are not shown (O2: no precomputed reference file; a run at the wanted size is a Run click).

### 10.3 Calibration backtest (leave-one-week-out)

For each held-out world `w`, slot `t`, direction, duration, horizon and level `α ∈ L_q`:

```
day-ahead:       q̂_{w,t}(α) = numpy.quantile({ A_{w',t} : w' ≠ w }, α, method="linear")   target A_{w,t}     (the other W − 1 weeks stand in for history; nothing from week w)
intraday:        q̂_{w,t}(α) = the §10.2d quantile of A_τ in world w                         target A_{w,t}     (the whole-fleet realised deliverable, the same target as day-ahead; its late part already leaves w out)
intraday_known:  q̂_{w,t}(α) = the §10.2d quantile of K in world w                           target A^K_{w,t}   (the driveway part alone, to see whether the response-and-outage mixture is calibrated by itself)
hit_{w,t}(α) = target ≤ q̂_{w,t}(α)
pinball_{w,t}(α) = (target − q̂_{w,t}(α)) × (α − 1[target < q̂_{w,t}(α)])           [kW]
```

`coverage(α)` is the share of hits and should be about `α`; `coverage_strict(α)` is the share with `target < q̂_{w,t}(α)`, reported beside it because with point masses (0 kW slots, near-certain known parts) a calibrated forecast has `coverage_strict ≤ α ≤ coverage` rather than `coverage ≈ α` (lead ruling on J3 Q2: no randomised hits on ties); `pinball` is the proper score of a quantile forecast (lower is better; at α = 0.5 it is half the absolute error). A world-slot is **degenerate** and left out when both the target and every forecast quantile are 0 (nothing to forecast: a daytime slot with no home charging) or when either is NaN. Slots before `s_n` have no intraday forecast and are left out of the two intraday horizons. The expected finding for `intraday`, to report and not tune: coverage near nominal but a wider band than a fully conditional forecast would give (the late part ignores what the early arrivals say about the night, §10.2d).

**`availability_backtest` (exact).** One row per (horizon, direction, duration_hours, local_half_hour, level), horizons in the order above, half-hours in noon-to-noon order: `horizon`, `direction`, `duration_hours`, `local_half_hour` (object `"HH:MM"`), `profile_order` (int64 0–47), `level` (float64), `sample_count` (int64, world-nights kept), `coverage`, `coverage_strict`, `pinball_loss_kw` (means over the kept world-nights), `mean_forecast_kw` (mean `q̂`), `mean_realised_kw` (mean target), `evidence_kind`. Both copies of a repeated autumn half-hour enter their label. 8,064 rows (3 × 2 × 4 × 48 × 7; O3).

**`availability_reliability` (exact).** The same without the half-hour, pooled over all kept world-slots: one row per (horizon, direction, duration_hours, level), columns `horizon`, `direction`, `duration_hours`, `level`, `sample_count`, `coverage`, `coverage_strict`, `pinball_loss_kw`, `mean_forecast_kw`, `mean_realised_kw`, `evidence_kind`. 168 rows; the reliability diagram plots `coverage` against `level`.

**`availability_backtest_summary` (exact).** One row per (horizon, direction, duration_hours): `coverage_p05`, `coverage_strict_p05`, `coverage_p10`, `coverage_strict_p10`, `coverage_p50`, `coverage_strict_p50`, `coverage_p90`, `coverage_strict_p90` (the `coverage` and `coverage_strict` at levels 0.05, 0.1, 0.5, 0.9), `pinball_mean_kw` (mean of `pinball_loss_kw` over the seven levels, a discrete CRPS proxy), `sharpness_p10_p90_kw` (mean over kept world-slots of `q̂(0.9) − q̂(0.1)`), `sample_count`, `evidence_kind`. 24 rows; the lens headline and Key stats read it.

Tests (J3): on a fixture where every world's deliverable at a slot is an independent draw from one known distribution, day-ahead coverage at each level is within three binomial standard errors of the level on the per-half-hour rows, and on the pooled rows within the exact leave-one-out band `[(⌊α(W − 2)⌋ + 1) ÷ W, (⌊α(W − 2)⌋ + 2) ÷ W]` (pooled binomial errors are smaller than the leave-one-out quantile's finite-sample offset of about `(1 − 2α) ÷ W`; accepted by the lead); on a fixture with a 0 kW point mass, `coverage_strict ≤ α ≤ coverage` within the same tolerance; when the known part is built from the realised plugged-in set with `r = 1`, `π = 0` and no early departures, and every late contribution is 0, the intraday quantiles all equal the realised value and coverage is 1 at every level with pinball 0; a deliberately biased forecast (every quantile halved) shows coverage below level and a larger pinball; degenerate slots are excluded (`sample_count` 0, NaN statistics); `availability_reliability` equals the sample-weighted pooling of `availability_backtest`; the summary equals its recomputation; the `intraday_known` rows use `deliverable_known_kw` as the target.

### 10.4 Newsvendor commitment (§4.4 and §9.2 changed in place)

A switch `trading.commitment_rule`: `fixed_share` (default; §9.2, `q = c × F̂`) or `newsvendor`. In newsvendor mode the day-ahead position of night `n` for slot `t` in world `w` is

```
F̂_{w,t}      = mask_DA,t × max(0, B⁰_{w,t} − x_{w,t})                       the §4.4 point forecast (kWh)
e_{w,t}(α)   = numpy.quantile({ V_{w',t} − F̂_{w',t} : w' ≠ w }, α)           leave-one-world-out forecast-error quantile (kWh)
k_{w,t}      = mean { SIP_{w',t} : w' ≠ w, system short in (w', t) }         expected shortfall cost (£/MWh), leave-one-world-out (B3); the mean over w' ≠ w of SIP when no other world is short at t
α_{w,t}      = clip(P_DA,w,t ÷ k_{w,t}, 0, 1)                                 0 when k ≤ 0 or P_DA ≤ 0
q_{w,t}      = mask_DA,t × max(0, F̂_{w,t} + e_{w,t}(α_{w,t}))
```

`α` is the commit level of the settled-deviation distribution: `P(V ≤ q) = α`, so `P(V ≥ q) = 1 − p/k`, the "commit quantile 1 − p/k" of item 59 in exceedance terms. Derivation: with revenue `p` per MWh sold and a shortfall bought back at `k` per MWh, `Π(q) = p q − k E[(q − V)⁺]`, `Π'(q) = p − k P(V < q) = 0` gives `F(q*) = p/k`. A negative day-ahead price commits nothing (turn-down is worth nothing then); `p ≥ k` commits the largest error the other weeks saw. What the rule assumes (B2): delivered-but-uncommitted volume earns nothing (unpaid spill) and a shortfall is bought back at the short-state imbalance price, which is a **firm product with a non-delivery penalty**. Under this contract's single-price ledger (§4.7) spill is paid at SIP and a shortfall is charged at SIP, so the rule is not the expected-cash optimum there: it is a risk choice, and the ledger shows what it costs or earns against the fixed share on the same futures (Compare, or the two rules side by side in `trading_kpis`).

The error quantile is a calibrated correction of the §4.4 forecast, not a replacement: it keeps the world's own baseline and expected plan and adds what the other simulated weeks say about that forecast's error at this slot (in reality, past weeks). `V` is `mask × max(0, B − M)` (§4.2), a kernel-and-baseline quantity that does not depend on any position, so `run_trading` computes it for every world before the day-ahead step; `P_DA,w,t` is published at `τ_DA(n)`; `k_{w,t}` and `e_{w,t}` read only the other worlds, so the rule reads nothing from week `w` after its instant. `full` then re-positions intraday exactly as §4.5 from this `q`; `da_only` keeps `f = q`; `perfect_foresight` is unchanged. Blackout slots are in `mask_DA` (§10.5b).

Reported: `deviation_world_slot` gains `commit_level` (`α_{w,t}`, NaN in fixed-share mode) and `forecast_error_quantile_kwh` (`e_{w,t}(α_{w,t})`, NaN in fixed-share mode) after `position_perfect_foresight_kwh` (§5.1 changed). `trading_checks.mean_intraday_revision` uses `f − F̂` in newsvendor mode (§5.5 changed). The Position chart's hover shows the level. A firm-MW **call-off** commitment is a different product from this position: the position sells the plan's own reduction against the baseline; the product sheet's firm MW (§10.5a) is dispatchable power on top of the plan; the lens says so.

Tests (J4): `fixed_share` reproduces the §9.2 ledger bit for bit; with every world's error identical, `q = F̂ + error` at any level; `α = 0` gives `q = max(0, F̂ + min error)`, `α = 1` the max; `q` is non-decreasing in `P_DA`; perturbing world `w`'s own `V`, `F̂` and `SIP` leaves its `e_{w,t}(α)` and `k_{w,t}` unchanged (leave-one-out); `P_DA ≤ 0` gives `q = 0`; `commit_level ∈ [0, 1]`; with two worlds the error sample has one value and `e` equals it at every level; the `mean_intraday_revision` row uses `f − F̂`.

### 10.5 Product sheet, blackout windows, price-weighted availability, settlement file, charge completion, net shape change, firmness by manufacturer

#### 10.5a `product_sheet` (exact)

Product windows, London wall clock, in noon-to-noon order (constants `availability.PRODUCT_WINDOWS`, reporting choices like the 18:00 default of §9.9, accepted by the lead): `evening` 17:00–21:00, `overnight` 21:00–06:00, `morning` 06:00–12:00. The 12:00–17:00 afternoon is in no window (little home charging and before the intraday decision). A window is the set of study slots whose `local_half_hour` label lies in it, on all seven nights; both copies of a repeated autumn half-hour count.

One row per (window, direction, duration_hours). Per world first, then linear P05/P10/P50/P90 across worlds:

| Column | Per-world value or rule |
| --- | --- |
| `window`, `window_start_local`, `window_end_local` (`"HH:MM"`), `direction`, `duration_hours`, `world_count` | Keys |
| `window_mean_mw_p05`, `_p10`, `_p50`, `_p90` | Mean over the window's slots of `deliverable_kw ÷ 1000` |
| `window_min_mw_p10`, `_p50`, `_p90` | Minimum over the window's slots (the MW that holds through the whole window) |
| `firm_share`, `firm_share_p05` | `window_mean_mw_p10 ÷ window_mean_mw_p50`, `window_mean_mw_p05 ÷ window_mean_mw_p50` |
| `intraday_firm_mw_p50` | Mean over the window's slots at or after `s_n` of `intraday_p10_kw ÷ 1000`, then the median across worlds: the typical week's firm figure at 17:00 |
| `intraday_known_share_p50` | Per world `Σ intraday_known_mean_kw ÷ Σ intraday_mean_kw` over the same slots (NaN when 0), then the median: how much of the 17:00 figure is already on the driveway |
| `movable_energy_mwh_p10`, `_p50`, `_p90` | On the `duration_hours = 0.5` row only (NaN otherwise): Σ over the window's slots of `deliverable_kw × 0.5 ÷ 1000`, the energy the fleet could shift out of (turn-down) or into (turn-up) the window, each half-hour counted once |
| `recovery_energy_mwh_p50` | Turn-down rows: `duration_hours × window_mean_mw` per world, the energy a full-length call defers past the window; NaN for turn-up (it pulls charging forward; the later reduction is the `net_change` of §10.5f) |
| `rebound_capacity_mw_p50` | Turn-down rows: mean over the window's slots of `eligible_power_kw ÷ 1000`, the power that resumes after a call at worst; NaN for turn-up |
| `recovery_hours_p50` | Turn-down rows: per world `recovery_energy_mwh ÷ rebound_capacity_mw` (NaN when 0), then the median |
| `max_event_length_hours` | The largest `duration_hours` in the grid whose `window_mean_mw_p50` is at least 25 % of the 0.5-h row's (`availability.MAX_EVENT_SHARE = 0.25`, the §3.6e capacity-share precedent); NaN if none; repeated on every row of the (window, direction) block |
| `notice_day_ahead_hours` | Hours from `τ_DA(n)` (13:00 `D_n − 1`) to the window start: 28, 32, 41 |
| `notice_intraday_hours` | Hours from `τ_n` to the window start: 0, 4, 13 at the 17:00 default |
| `ramp_hours` | 0.5: the model dispatches in whole half-hours, so a call takes effect at the next slot boundary; a property of the model, not a measurement |
| `evidence_kind` | |

24 rows. Quantile columns do not add across windows or durations; the caption says so. NaN rule (O5): slots whose window runs past the study end (the last night's morning for 2 h and 4 h, §10.2a) are left out of a world's window mean and minimum (`nanmean`, `nanmin`); a (window, duration) with no defined slot in a world is NaN for that world; the statistics use the worlds with a value and `world_count` counts them. The 2-h and 4-h morning rows therefore cover the last night only up to 10:00 and 08:00, and the caption says so.

#### 10.5b Blackout windows

A setting `availability.blackout_windows`: an editable table of windows (lead decision), default empty, edited like `events`, columns `start_local_time` (`"HH:MM"` London wall clock on a half-hour) and `duration_minutes` (a multiple of 30, 30–720); a window may wrap midnight; windows may not overlap once mapped to the noon-to-noon order. `availability.validate_blackout_windows(frame) -> pd.DataFrame` returns the typed table with `end_local_time` and `slot_labels` added, or raises naming the row and rule. The frame is stored as `blackout_windows`; `study_slots.blackout` and `warmup_slots.blackout` (bool) mark the slots (the last warm-up night plans too, item 52). Meaning: half-hours in which the aggregator promises not to move charging, in either direction.

What it does:

1. **Planner (J1b, `_smart_charging_slot`).** At plan time, with `w_price` the window's visible prices (`+inf` outside the window) and `bl` the blackout mask over the window's slots:
   ```
   u      = plan_cheapest_slots(need, cap, where(isfinite(w_price), 0, +inf))    the unmanaged trajectory: full power from plug-in
   fixed  = u × bl                                                                what the EV would have charged in blackout slots anyway
   plan   = fixed + plan_cheapest_slots(need − fixed.sum(), cap, where(bl, +inf, w_price))
   ```
   Blackout slots keep exactly the unmanaged charging (nothing moved out) and take nothing else (nothing moved in); the rest of the need goes to the cheapest slots outside. Because the plan enters a blackout slot with the same energy the unmanaged path would have taken there or less, the kernel's `min(plan, normal limit)` executes `fixed` unchanged. Two calls of the existing planner; no new machinery. A need the slots outside the blackout cannot hold leaves the session short: `plan_cheapest_slots` gives `+inf` slots nothing, so the plan never breaches the blackout, and the shortfall shows as an early-departure or not-recovered cost like any other (O6).
2. **Trading masks (J1b in `events.trading_mask`).** `trading_mask(..., blackout=study mask)` returns False on blackout slots at every `known_at_utc` (a blackout is known before the run): no position, no intraday trade and no settlement there (§4.6 changed). A request (§2) whose window lies wholly inside a blackout is rejected by `validate_events` ("nothing can be delivered in a blackout"); partial overlap is allowed and delivery is measured outside the blackout slots only.
3. **Availability (J2).** Any window overlapping a blackout slot has `deliverable_kw`, its parts, `potential_kw` and every intraday column 0; `blackout` is True on the rows whose own slot is in a blackout.

Tests (J1b, J2): an empty table reproduces the run bit for bit; a blackout covering an EV's whole window makes its selected import equal its normal import; a blackout over the cheapest half-hours moves the plan to the next cheapest outside and the session still reaches target when the outside slots can hold the need; battery energy is conserved and the normal path is unchanged; the mask is False in blackout slots for every `known_at_utc`; the validator rejects a 10-minute start, a 45-minute duration, an overlap and a zero duration; a need the outside slots cannot hold leaves the EV short with no import in the blackout beyond the unmanaged amount; availability is 0 in and across blackout slots.

#### 10.5c `availability_value_summary` (exact): price-weighted availability

One row per (direction, duration_hours, metric); per world first, then `mean`, `p10`, `p50`, `p90` across worlds; columns `direction`, `duration_hours`, `metric`, `unit`, `world_count`, `mean`, `p10`, `p50`, `p90`, `evidence_kind`. Weights use the day-ahead price after all shocks and the user curve, `P_{w,t}`: `ω_t = max(0, P_{w,t})` for turn-down (turn-down is worth the price it avoids) and `ω_t = max(0, −P_{w,t})` for turn-up (worth something only in negative-price half-hours).

| `metric` | `unit` | Per-world value |
| --- | --- | --- |
| `price_weighted_mw` | `MW` | `Σ_t A_t ω_t ÷ Σ_t ω_t ÷ 1000`, NaN when `Σ ω = 0` (no negative price all week for turn-up) |
| `simple_mean_mw` | `MW` | `mean_t A_t ÷ 1000`, for the comparison |
| `value_at_day_ahead_gbp_per_week` | `GBP per week` | `Σ_t A_t × 0.5 ÷ 1000 × ω_t`, labelled "illustrative: every deliverable MWh at the day-ahead price; not a P&L, not a bid, not comparable with the trading net" |

NaN rule (O5): a world whose weight sum is 0 has NaN `price_weighted_mw` (turn-up in a week with no negative price); the statistics across worlds use the worlds with a value, `world_count` counts them, and a metric with no world is NaN; `value_at_day_ahead_gbp_per_week` is 0, not NaN, in such a world.

#### 10.5d `settlement_file` (exact): a per-EV settlement file for the sample week

The representative world only (`representative_world_id`); `settlement_file_name` = `axle_settlement_<study start date>_seed<seed>_world<id>.csv`, written by the view with `to_csv(index=False)` and nothing added. Per slot, the EV rows share out only the fleet's **true reduction** `R_t = U_t − M_t` (B5): each EV's settled volume is its own measured reduction in the settled slots. The **baseline effect** `E_t = B_t − U_t` is not the work of any EV, so it is one unallocated fleet row per slot. One row per (unit, slot) plus one baseline-effect row per slot, unit-major with the fleet row last in each slot: `(N + 1) × 336` rows (about 30 MB at 1,000 EVs; one sample week, lead decision). The EV id stands in for the meter point (MPAN), and the caption says so.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `meter_point_id` | object | `unit_id`, or `"baseline_effect"` on the fleet row |
| `cohort_id`, `manufacturer_id`, `zone_id` | object | `"fleet"` on the fleet row |
| `control_group` | bool | False on the fleet row |
| `slot_index`, `night_index` | int64 | |
| `interval_start_utc`, `interval_start_london` | UTC, London | |
| `settlement_date`, `settlement_period` | object, int64 | As `supplier_positions` (§9.3e) |
| `unmanaged_kwh`, `metered_kwh` | float64 | The EV's normal-path and selected-path home import (`u_{it}`, `m_{it}`); NaN on the fleet row |
| `deviation_kwh` | float64 | EV rows: `u_{it} − m_{it}`, the EV's true reduction, positive = turned down, negative = rebound or turn-up. Fleet row: `E_t` |
| `settled_kwh` | float64 | EV rows: `1[V_t > 0] × (u_{it} − m_{it})`, signed; fleet row: `1[V_t > 0] × E_t`. Per slot the EV rows sum to `1[V_t > 0] R_t` and all rows to `V_t` (§4.2: `V = D = R + E` where settled) |
| `day_ahead_gbp_per_mwh` | float64 | After all shocks and the user curve |
| `payment_gbp` | float64 | EV rows: the customer's revenue share of the `full` strategy for the night, `−customer_revenue_share_gbp(w, full, n)`, allocated to the EV with §9.3c's `a_{i,n}` and spread over its slots in proportion to `max(0, settled_kwh)` (equally over the night's slots when it has none). Σ over the file = the week's customer share; a per-EV allocation of fleet settlement, not a per-meter baseline. 0 on the fleet row |
| `evidence_kind` | object | |

Control EVs get `payment_gbp = 0` (weight 0 in §9.3c) but keep their `settled_kwh` (they are in the fleet's metered import). Validator: per slot, Σ EV `settled_kwh` = `1[V_t > 0] × true_reduction_kwh` and Σ all rows = `settled_kwh` of `deviation_world_slot` for that world, within 1e-9; Σ `payment_gbp` per night equals the ledger's customer share for (world, `full`, night); `unmanaged_kwh` and `metered_kwh` sum to the fleet frames; the fleet row's `deviation_kwh` equals `baseline_effect_kwh`.

#### 10.5e `charge_completion_summary` (exact): charge completion by departure

Per world, path and group, over the home sessions that end in the study (`find_sessions`): a session is **complete** when the battery stock at its last connected slot is at least `target_i − 1e-6 kWh` (`NOT_RECOVERED_TOLERANCE_KWH`). `departure` splits sessions into `on_time` (the realised unplug is at or after the planner's expected departure for that session) and `early` (before it); `all` is both. One row per (group_id, path_id, departure), `group_id` `fleet`, then each cohort with EVs (source order), then each manufacturer `m1`…`m4`; `path_id` normal then selected, plus `timed` (decision 0007, model step 2) on a run whose `timed_start_local_hour` is set; departure `all`, `on_time`, `early`. Columns: `group_id`, `path_id`, `departure`, `ev_count` (int64), `world_count`, `session_count_mean` (float64, sessions per world), `completed_share_mean`, `completed_share_p10`, `completed_share_p50`, `completed_share_p90` (per world the share of the group's sessions that are complete, NaN with none), `shortfall_kwh_mean`, `shortfall_kwh_p50` (per world the mean `target − stock at unplug` over incomplete sessions, battery kWh; NaN with none), `evidence_kind`. On the normal path early departures cut completion too; the paired difference across paths is read from the two rows by eye, never computed by the view. The timed path's own `path_id` block reads the same way, with no third-path difference computed anywhere. Validator: `all` session counts equal `on_time + early`; a session complete on the selected path with no public top-up and no early departure has `departure_shortfall_kwh` 0 in the kernel; the `timed` block, when present, follows the same `all = on_time + early` rule.

#### 10.5f `supplier_positions` additions (§9.3e changed): net shape change with rebound

After `settled_mw_p50`, in order: `unmanaged_mw_p50` (`U ÷ 0.5 ÷ 1000`), `net_change_mw_p10`, `net_change_mw_p50`, `net_change_mw_p90` (per world `(M − U) ÷ 0.5 ÷ 1000`: negative where charging moved out of the slot, positive where it landed, including the rebound after a turn-down window; linear quantiles across worlds), `deliverable_turn_down_1h_mw_p10`, `deliverable_turn_up_1h_mw_p10` (the 1-h day-ahead firm figures, `availability_bands` `realised` `p10 ÷ 1000`). The positions chart draws `net_change_mw_p50` as the net shape change under the positions; the CSV download carries the new columns.

#### 10.5g `firmness_by_manufacturer` (exact)

One row per (manufacturer_id, direction, duration_hours) from `availability_manufacturer_world`: `manufacturer_id`, `manufacturer_label`, `share` (the record), `ev_count`, `response_rate`, `outage_probability_per_night`, `direction`, `duration_hours`, `world_count`, `firmness_mean`, `firmness_p10`, `firmness_p50`, `firmness_p90` (per world `Σ_n realised_mwh ÷ Σ_n potential_mwh`, NaN when the potential is 0), `outage_nights_mean`, `outage_nights_p50` (per world the count of nights out), `evidence_kind`. 32 rows. Firmness here is physical (delivered ÷ deliverable if every session had followed its plan, the plan-based potential of B6); the ledger's `firmness` (§9.1) is against the traded position and is fleet-level, and the two are labelled apart.

#### 10.5h `manufacturer_summary` (exact): the diversification explainer

One row per manufacturer plus a `fleet` row: `manufacturer_id` (`m1`…`m4`, `fleet`), `manufacturer_label`, `share`, `ev_count`, `response_rate` (NaN on the fleet row), `outage_probability_per_night` (the fleet row holds `1 − Π_m (1 − π_m)`, the chance at least one maker is out on a night), `evening_potential_mw_p50` (per world the mean over the evening window of the maker's 1-h turn-down potential from `availability_manufacturer_world`, MW; the fleet row the total), `largest_share` (repeated), `evidence_kind`. The lens shows it beside the band with the reading: a night with one maker out removes its slice (about `share × evening potential`), so the firm figure depends on the largest slice and on `P(any outage)`; with makers of equal share the loss from one outage is `1/M` of the fleet, so more, smaller makers are diversification, exactly as more independent EVs are. Herding (§10.0) is the opposite kind of shared factor: it concentrates deliverable turn-down in the cheapest half-hours and deliverable turn-up in the evening, because every EV in a world ranks one price path; the band shows it as the shape of the day-ahead P50 across the night.

### 10.6 Notebook 09 and the fleet-size study

#### 10.6a Notebook 09: a top-down quantile model against the bottom-up bands

`notebooks/09_top_down_quantile_model.ipynb`, following plan §10's rules (imports package functions, small fast sample, "try it" cells, Plotly with a paired table, sanity asserts, passes `ruff format --check`, runs in the slow-marked notebook test). Concept: a supplier's data scientist has no physics model, only fleet-level history; fit a top-down quantile model on the simulated history and compare its bands with the bottom-up bands of §10.2–§10.3.

1. **Data.** `run_forecast_from_assumptions(...)` at a small size (the "try it" defaults: 100 EVs, 40 worlds, seed 7; about 60 s including the availability frames and the intraday convolution, accepted by the lead) plus, for the enrolled-EVs feature, a second run at 200 EVs (same seed; optional cell). History rows = (world, night, slot) with target `deliverable_kw` (turn-down, 1 h) from `availability_world_slot`. Features from `study_slots` and `world_nights`, each one known at the day-ahead decision `τ_DA(n)` (13:00 on `D_n − 1`, one hour into night `n − 1`): `profile_order` (and its sine and cosine), weekday of the night, `holiday`, `temperature_c` and `solar_clearness` (labelled a **perfect weather forecast**: the sampled values, not a forecast with error, B4), `enrolled_evs` (`vehicle_count`), and recent availability as the realised value **two nights back** at the same half-hour (lag 96 slots) and that night's evening mean (B4: night `n − 1` is still running at `τ_DA(n)`, the same cut-off the §4.1 baseline history uses; `n − 2` is complete). Nights 0 and 1 are dropped. No realised quantity of the night itself or of night `n − 1` is a feature.
2. **Split by world**, not by row (slots of one week share its factors): 60 % train, 20 % calibration, 20 % test.
3. **Model.** `sklearn.ensemble.GradientBoostingRegressor(loss="quantile", alpha=α)` for α in 0.1, 0.5, 0.9 (`n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8`, "try it" values, not assumptions).
4. **Split conformal correction** (conformalised quantile regression, Romano, Patterson and Candès 2019): on the calibration worlds `E = max(q̂_0.1 − y, y − q̂_0.9)`; `Q = ` the `⌈(n + 1)(1 − 0.2)⌉ ÷ n` empirical quantile of `E`; the corrected band is `[q̂_0.1 − Q, q̂_0.9 + Q]`, guaranteed 80 % marginal coverage on exchangeable data.
5. **Comparison on the test worlds**: coverage of the 0.1–0.9 band, pinball at 0.1, 0.5, 0.9 and sharpness for (a) the bottom-up day-ahead band from the training worlds only (the §10.3 rule with the test week left out), (b) the bottom-up intraday whole-night band, (c) top-down raw, (d) top-down + conformal. One Plotly chart of a test week (realised, bottom-up and top-down bands), a reliability plot, a paired table, and a "P90/P50 against fleet size" cell that reads `firm_share_by_fleet_size` from the run (and, as a "try it", a second run at a larger `N`).
6. **Asserts.** Conformal coverage on the test worlds approximately 80 %, asserted with a tolerant band (0.60–0.95) because rows within a world are dependent and there are only about eight test worlds (O7); the conformal band is never narrower than the raw band; `firm_share` is non-decreasing in fleet size within noise (the last point at least the first); shapes and NaN rules.
7. **Limitations** cell: synthetic history fitted on the model's own output, so the comparison tests estimators, not reality; the top-down model cannot see maker outages before they happen; the weather features are a perfect forecast; the intraday bottom-up band uses the 17:00 driveway and is not comparable with a day-ahead band, which the text says.

Dependency: `scikit-learn` in the `dev` dependency group of `pyproject.toml` (the current release uv resolves, pinned by `uv.lock`; `GradientBoostingRegressor(loss="quantile")` has existed since 0.x, so no version subtlety). Not a runtime dependency of the app.

#### 10.6b Fleet-size study: the live sub-fleet curve

`firm_share_by_fleet_size` (§10.2e) is computed on every run, exact for `n ≤ N` from the run's own per-EV contributions with no extra simulation, and is the study (O2: the precomputed 3,000-EV reference file of the first draft is dropped; a larger fleet is a Run click). The lens plots `firm_share` and `firm_share_p05` against `fleet_size` per window and direction, with `n_eff` and `implied_rho` in the hover, and the caption "flat beyond a few hundred EVs: shared factors, not sampling noise, set the firm share". At the defaults the plug-in-driven part of the curve is expected to flatten within a few hundred EVs (`n_eff` about 640 at 1,000 and 1,500 at 10,000 EVs for the evening availability at the item 64 default, §10.1b) and to keep rising only as response noise averages out; that expectation is reported, not tuned.

### 10.7 Build lanes

Disjoint file ownership; `forecast.py` call lines and the `validate_result_v2` hook lines for J2, J3 and J5 are added by the lead at each lane's integration (one call each), so no lane co-owns `forecast.py` with J1a. Each lane adds its frames' validator in its own fixture module, adds a regression or invariant test for every rule it implements, regenerates seeded pins only where this section says a draw moves, and reports the commands it ran.

| Lane | Outcome | Owns | Depends on | Must add (tests) |
| --- | --- | --- | --- | --- |
| **J1a** shared factors and manufacturers (High risk: sampling) | §10.1a–d complete; the `session_non_response_probability` array of §10.1e; `world_nights` inputs; `ForecastResult` fields of §10.0 declared (`None`); `SimulatedForecast` fields; draw order | `sampling.py`, `assumptions.py`, `clock.py`, `settings.py` (if a field is needed), `forecast.py`, `tests/model/test_shared_factors.py`, `tests/model/test_manufacturers.py`, `tests/model/test_world_nights.py`, pins in `tests/model/test_sampling_common_random_numbers.py` | H5 (zone and control-group permutations) merged; F2's `sample_non_response` merged, or J1a adds it and says so | §10.1b, §10.1c, §10.1d tests and the array test of §10.1e; draw-count and draw-order tests; a Compare pairing test (editing any §10.1 record changes no other channel) |
| **J1b** kernel wiring and blackout planner (High risk: physics) | §10.1e kernel changes; §10.5b items 1–2; the `forecast.py` wiring of `SmartCharging.non_response` and the blackout masks (a small hunk after J1a merges) | `action.py`, `physics.py`, `events.py` (the `trading_mask` blackout input and the `validate_events` blackout rule), `tests/model/test_blackout.py`, `tests/model/test_plan_status.py`; the `forecast.py` hunk after J1a | J1a merged; §9.5's `SmartCharging.non_response` plumbing (F2) merged or absorbed | §10.1e kernel tests, including the R1 test (a short-notice event announced after 17:00 leaves `decision_plan_kwh` unchanged) and the R2 per-plan code tests; §10.5b planner and mask tests; battery conservation and one-route-per-slot on the fixture |
| **J2** deliverable MW and diagnostics (High risk: new result schema) | §10.2 complete: `availability.py`, the chunk-loop hook in `summaries.py`, `PRODUCT_WINDOWS`, `intraday_distribution`, blackout zeroing, `firm_share_by_fleet_size` | `src/axle_studio/model/availability.py`, the hook hunk in `summaries.py` (at most 20 lines), `tests/model/test_availability.py`, `tests/fixtures/availability_contract.py` | J1b's per-EV outputs and J1a's `manufacturer_id`; develops against a hand fixture of per-EV arrays and integrates after J1b | The §10.2c variance, `implied_rho` and `firm_share` tests; the hand fixture of §10.2a (one EV, six slots: exact `a`, `ā`, `â` for every duration and direction); monotonicity `a(D)` non-increasing in `D`; the intraday distribution against Monte Carlo (§10.2d), its known/late split and the shared-outage case (Q2: with `π_m = 1` for one maker, the late quantiles exclude that maker's restored slice in every sampled world); the sub-fleet identity (`n = N` row equals the fleet band); frame validators |
| **J3** calibration backtest | §10.3 complete | `src/axle_studio/model/availability_backtest.py`, `tests/model/test_availability_backtest.py`, `tests/fixtures/availability_backtest_contract.py` | J2 (reads `availability_world_slot`) | The §10.3 tests |
| **J4** newsvendor commitment | §10.4 complete | `src/axle_studio/model/market.py` (§10.4 hunks), `tests/model/test_market_newsvendor.py` | F1/F2 `market.py` merged | The §10.4 tests; `fixed_share` bit-for-bit |
| **J5** product frames | §10.5a, c–h complete; `slim_run`/`compare_runs` records for `product_sheet` and `firmness_by_manufacturer` | `src/axle_studio/model/product.py`, the `supplier_positions` columns and the second hook hunk in `summaries.py` (after J2 merges), `tests/model/test_product.py`, `tests/fixtures/product_contract.py` | J2 frames; the trading ledger and §9.3c allocation (settlement file) | Product-sheet recomputation from `availability_world_slot`; `max_event_length` rule; value summary NaN rule; settlement-file sums of §10.5d including the baseline-effect row; completion counts (§10.5e); firmness recomputation; `manufacturer_summary` `P(any outage)`; net-change quantiles |
| **J6** UI | Supplier ▸ 4 Firm MW lens; Trading ▸ 3 firmness-by-manufacturer block and commitment-rule caption; Overview ▸ Key stats section H; Edit assumptions tab and the two holiday switches; Compare rows | `src/axle_studio/ui/views/supplier_firm_mw.py` (new), the lens entries in `ui/registry.py` and `ui/pages.py`, the small hunks in the Trading P&L, Overview and Parameters views (after the W4 UI lanes merge), `tests/ui/test_supplier_firm_mw_view.py` and the touched view tests | The Supplier page (W4 I5), J2–J5 frames | AppTest renders with and without every optional frame; no view arithmetic on quantiles (the existing lint); the download button writes `settlement_file` as stored |
| **J7** notebook | §10.6a complete | `notebooks/09_top_down_quantile_model.ipynb`, `pyproject.toml` and `uv.lock` (dev group), the notebook entry in the slow test | J1–J3 | The notebook's own asserts; `ruff format --check` on the notebook |

Parallel sets: {J1a, J2 on a fixture, J4} first; then J1b (after J1a); then J3 and J5 (after J2 merges; J5 also after the trading ledger); then J6 and J7. Review: J1a, J1b, J2 and J4 are High risk (§REVIEW_POLICY): lead-accepted interface (this section) and one strong independent review each, with a specialist second review for J1a's sampling change and J2's variance formula and intraday distribution. J3, J5, J7 are Normal. J6 is Normal unless the lens computes anything, which it must not.

### 10.8 Assumption records (`model/assumptions.py`, J1a)

| Name | Default | Unit | Bounds | Evidence | Authority |
| --- | --- | --- | --- | --- | --- |
| `behaviour.plug_in_skip_median` | 0.12 | fraction (median skip share on an ordinary night) | 0–0.5 | illustrative | decision 0004 item 60 |
| `behaviour.plug_in_day_factor_sd` | 0.5 | logit SD per (world, date) | 0–3 | illustrative: Mike's choice; the arithmetic and alternatives in §10.1b | item 64 (items 61 and 63 for the arithmetic) |
| `behaviour.plug_in_week_factor_sd` | 0.17 | logit SD per world | 0–3 | illustrative: about a third of `σ_day` | item 64 |
| `behaviour.holiday_bank_holiday_monday` | off | switch (+0.30 on the skip logit for one date) | | illustrative | item 59 "optional presets"; O1; lead answer (skip share only) |
| `behaviour.holiday_half_term_week` | off | switch (+0.30 on the skip logit) | | illustrative | item 59; lead review |
| `manufacturers.share.m1` … `m4` | 0.25 each | fraction, sum 1 | 0–1 | illustrative | item 59; lead decision (equal shares, the zone precedent) |
| `manufacturers.response_rate.m1` … `m4` | 0.95 each | fraction | 0–1 | illustrative (`1 − 0.05`, the approved §8 Q2 value) | item 59 |
| `manufacturers.outage_probability_per_night.m1` … `m4` | 0.02 each | probability per night | 0–1 | illustrative | item 59; lead decision |
| `availability.intraday_decision_local_time` | `"17:00"` | London time on a half-hour, 12:00–23:30 | | illustrative | item 59 ("at 17:00") |
| `availability.blackout_windows` | empty table | rows of (`start_local_time`, `duration_minutes`) | §10.5b | user choice | item 59; lead decision |
| `trading.commitment_rule` | `fixed_share` | switch (`fixed_share`, `newsvendor`) | | illustrative | item 59; §10.4 |
| Retired: `behaviour.non_response_probability.<cohort_id>` (§9.9) | | | | | replaced by the maker response rates (§10.1e) |

Reporting constants, not assumptions (in `availability.py` and `availability_backtest.py`): durations (0.5, 1, 2, 4) h; levels (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95); product windows; `MAX_EVENT_SHARE` 0.25; the fleet-size grid (100, 300, 1,000, 3,000); the intraday grid `G = 512`. Dialog placement: a new Edit assumptions tab `"Firm MW"` for the manufacturer, availability and commitment records (lead decision); the three skip-share records and the two holiday switches sit in "Plugging & home charging" next to the plug probabilities they act on.

### 10.9 Validator rules (`validate_result_v2` additions, one fixture module per lane)

- Column sets, order and dtypes exactly as above; keys unique; every §10 frame `None` on a no-action result except `blackout_windows` and `units.manufacturer_id`; `study_slots.holiday` and `.blackout` present on every result.
- `units.manufacturer_id`: counts per maker equal the largest-remainder counts of the shares; `manufacturer_index` consistent.
- `availability_world_slot`: `0 ≤ deliverable_kw ≤ potential_kw`; `deliverable_kw = deliverable_known_kw + deliverable_late_kw`; `deliverable_kw` non-increasing in `duration_hours` within (world, slot, direction); `intraday_p05 ≤ … ≤ intraday_p95`, all within `[0, intraday_potential_kw + max late]`; `intraday_mean_kw = intraday_known_mean_kw + intraday_late_mean_kw`; `intraday_known_mean_kw = Σ_m (1 − π_m) r_m S_m` and `intraday_late_mean_kw = Σ_m (1 − π_m) × mean_{w' ≠ w} late_restored_kw_m<k>` recomputed; `Σ_m late_restored_kw_m<k> ≥ deliverable_late_kw` (restoration adds, never removes); with every `π_m = 0` the late quantiles equal `numpy.quantile` of the other worlds' `Σ_m late_restored_kw_m<k>`; NaN exactly on windows past the end and intraday slots before `s_n`; 0 on and across blackout slots; the `duration_hours = 0.5` turn-down value never exceeds the selected-path fleet `home_import_kw` of that slot; `eligible_power_kw ≥ deliverable_kw`.
- `availability_bands`: equal to the world-first recomputation from `availability_world_slot`; ordered quantiles `p05 ≤ p10 ≤ p50 ≤ p90`; `firm_share = p10 ÷ p50` and `firm_share_p05 = p05 ÷ p50` where `p50 > 0`; `implied_rho` and `n_eff` NaN on intraday rows and equal to the §10.2c recomputation on realised rows of the fixture; `n_eff ≤ contributing_ev_count` where `implied_rho ≥ 0`.
- `availability_manufacturer_world`: maker `realised_mwh` and `potential_mwh` sum to the fleet's per-night sums of `availability_world_slot` (× 0.5 ÷ 1000) within 1e-9; `outage` equals `world_nights.outage_m<k>`; `realised_mwh = 0` on outage nights.
- `firm_share_by_fleet_size`: the `fleet_size = N` row equals the window means of `availability_bands` realised rows; `p50_kw` non-decreasing in `fleet_size`; sizes ascending and each at most `N`.
- `availability_backtest`, `availability_reliability`, `availability_backtest_summary`: equal to recomputations from `availability_world_slot` under the §10.3 rules (the `intraday_known` horizon against `deliverable_known_kw`); `0 ≤ coverage ≤ 1`; `pinball_loss_kw ≥ 0`; `sample_count` sums match; summary columns equal the matching level rows.
- `deviation_world_slot`: `commit_level` in [0, 1] or NaN; in newsvendor mode `position_da_only_kwh = mask × max(0, F̂ + forecast_error_quantile_kwh)` with `F̂` recomputed, and `e` and `k` equal to the leave-one-out recomputations on the fixture; in fixed-share mode both columns NaN and the §5.9 rules unchanged.
- `product_sheet`, `availability_value_summary`, `firmness_by_manufacturer`, `manufacturer_summary`: equal to recomputations; `window_min ≤ window_mean` per world; `max_event_length_hours` follows the 25 % rule; `P(any outage) = 1 − Π(1 − π_m)`; `intraday_known_share_p50` in [0, 1] or NaN.
- `settlement_file`: the sums of §10.5d (EV rows to `1[V > 0] R_t`, all rows to `V_t`, the fleet row to `E_t`); one EV row per (unit, study slot) and one fleet row per slot; only the representative world.
- `charge_completion_summary`: `all = on_time + early` counts; shares in [0, 1] or NaN.
- `supplier_positions`: the new columns equal recomputations from `deviation_world_slot` (net change) and `availability_bands` (firm figures).
- `world_nights`: one row per (world, night); `outage_m<k>` equals the outage array; `plug_in_skip_share = expit(logit(median) + σ_week y + σ_day z + h_d)` recomputed from the stored factors and the switches; `plugged_in_count_at_decision` equals the count of EVs connected for the whole slot `s_n`.

### 10.10 UI consumption map

Views filter, plot and tabulate; they compute no availability, quantile, ratio, position or money. Every MW is labelled illustrative and synthetic. The lens carries the two-convention note of §10 ("P90 here means this table's P10 column; P95 its P05").

| View | Reads |
| --- | --- |
| Supplier ▸ 4 Firm MW: controls | Segmented direction (Turn-down, Turn-up), duration (0.5, 1, 2, 4 h), horizon (Day-ahead, Intraday at 17:00), the shared day-type control is not used (bands are per slot over the week) |
| Supplier ▸ 4 Firm MW: day-ahead band | `availability_bands` `day_ahead` `realised` (P05–P90 band with the P10 line marked as the firm figure, P50 line, `sd` in hover, `blackout` shading) with `availability_world_slot` `deliverable_kw` of `representative_world_id` as "this week"; tiles from `product_sheet` (evening 1 h: `window_mean_mw_p10`, `_p50`, `firm_share`, `firm_share_p05`) |
| Supplier ▸ 4 Firm MW: intraday at 17:00 | `availability_world_slot` for `representative_world_id`, one chosen night: the whole-night band `intraday_p05_kw`–`intraday_p95_kw` from `s_n`, the known-part line `intraday_known_p50_kw` ("on the driveway at 17:00") and `deliverable_kw` realised with its `deliverable_known_kw` split; `world_nights.plugged_in_share_at_decision` in the caption; `availability_bands` `intraday` `conditional_p10` and `conditional_known_p50` P50 lines across weeks ("typical firm figure at 17:00") |
| Supplier ▸ 4 Firm MW: diversification | `availability_bands` realised rows' `implied_rho`, `n_eff`, `firm_share` as a small profile chart; `firm_share_by_fleet_size` (per window) with `n_eff` and `implied_rho` in the hover and the §10.6b caption; `manufacturer_summary` table with the §10.5h reading; `world_nights` in the hover of the representative week |
| Supplier ▸ 4 Firm MW: product sheet | `product_sheet` table by window (MW P05/P10/P50, min-through-window, movable MWh, known share at 17:00, max event length, notice, ramp, recovery); `availability_value_summary` tiles (`price_weighted_mw`, `value_at_day_ahead_gbp_per_week` with its "not a P&L" caption) |
| Supplier ▸ 4 Firm MW: calibration | `availability_reliability` reliability diagram (coverage against level, three horizons); `availability_backtest` coverage-by-half-hour heatmap at level 0.1 and the pinball profile; `availability_backtest_summary` tiles |
| Supplier ▸ 4 Firm MW: firmness and completion | `firmness_by_manufacturer` table; `charge_completion_summary` tiles (fleet, both paths, `all` and `early`) and a per-maker table; `settlement_file` download button (`settlement_file_name`, as stored) |
| Supplier ▸ 2 Positions | `supplier_positions` new columns: `net_change_mw_p50` under the positions, firm 1-h columns in the table and CSV |
| Trading ▸ 2 Position | `deviation_world_slot.commit_level` in the hover; caption naming the commitment rule and, in newsvendor mode, the §10.4 risk-choice sentence |
| Trading ▸ 3 P&L and risk | `firmness_by_manufacturer` (1 h turn-down rows) next to the §9.1 firmness metrics, captioned "physical firmness by maker; the trading firmness above is against the position"; `trading_checks` as before |
| Overview ▸ Key stats, new section H Firm MW | `product_sheet` evening turn-down 1 h (`window_mean_mw_p10`, `_p50`, `firm_share`, `intraday_firm_mw_p50`); `availability_backtest_summary` day-ahead 1 h turn-down `coverage_p10`; `availability_bands` evening `n_eff` (the 19:00 slot); `manufacturer_summary` fleet `outage_probability_per_night` |
| Drivers ▸ Fleet week and Sessions | `study_slots.holiday` shading with a "holiday: skip shift +0.30" caption when either switch is on |
| Compare | `product_sheet` (evening and overnight turn-down 1 h rows) and `firmness_by_manufacturer` paired by world; `firm_share` per run side by side |
| Edit assumptions | The `"Firm MW"` tab (§10.8), the blackout windows editor (start, duration) with inline validation, the commitment-rule switch; the three skip-share records (with the §10.1b arithmetic as their help text) and the two holiday switches in "Plugging & home charging" |

### 10.11 Resolved, and what remains open

Resolved (recorded so the reviewer can check them off): the skip-share formulation (former Q1; §10.1b) and its defaults as recalibrated to the lead's evening-availability correlation target (§10.1b, with the arithmetic showing it equals item 60's value); holiday presets as two switches acting on the skip share only, half-term +0.30 (former Q3; O1; lead answer; §10.1c), which makes the B7 argument to `market.expected_sessions` unnecessary; the newsvendor `k` as the leave-one-world-out short-state SIP mean with the unpaid-spill reading and no alternative fractile (former Q5; B2, B3; §10.4); the intraday whole-night design with known and late parts and the whole-fleet target (former Q8; §10.2d, §10.3); the outage mixture kept (former Q9); no precomputed fleet-size reference file (O2; §10.6b); J1 split into J1a and J1b (O10; §10.7); notebook features per B4; the settlement file's true-reduction allocation with a baseline-effect row (B5); the plan-based potential (B6); `R` as response to a future call (Q-a); P05 added to the bands and the product sheet (Q-c); O3 (backtest 8,064 rows), O4 (world-slot size), O5 (NaN rules in §10.5a and §10.5c), O6 (a need the outside slots cannot hold leaves the session short, never breaching the blackout), O7 (tolerant coverage assert), O8 (`plan_status` precedence), O9 (turn-up counts only to the preferred target); and the lead's answers: manufacturer shares 0.25 and outage 0.02 per night accepted; product windows and the 25 % rule accepted; the settlement file as one sample week; blackout windows as a (start, duration) table; the "Firm MW" tab; the notebook's 60 s; the 0.5-h duration kept; the lane order as proposed; the turn-up price weights; the 512-bin intraday grid. Fourth round: R1 (per-slot remaining need under the plan in force; `K` reads the plan in force at `s_n`), R2 (`p_session` and `plan_status` per plan made, as main's §9.6), Q2 (per-maker late sums with outage slices restored, convolved per outage state), per-slot `x_max`, the per-chunk build of the state normals, and the item 61 citation; `σ_day = 0.5` and `σ_week = 0.17` by decision 0004 item 64, with the alternatives table kept and shared timing variation listed as a limitation.

Lead rulings on the J2 build (review of `aa65c29`, 29 September 2026):

- "`realised_mwh = 0` on outage nights" (§10.9, `availability_manufacturer_world`) is a fixture-only test, not a general validator rule: maker sums aggregate by slot night and a real session can span two nights, keeping its own night's `plan_status`.
- The `intraday_known_mean_kw` recomputation (§10.9) is exact only when every maker has the same `(1 − π_m) r_m`; otherwise the validator checks that it lies in `[min_m, max_m] (1 − π_m) r_m × intraday_potential_kw`, because the per-maker `S_m` are not stored.
- `contributing_ev_count` is 0 (int64) on intraday rows of `availability_bands`, not NaN (§10.2e).
- With one world there is no other week to read the late part from, so every intraday column of `availability_world_slot` is NaN.
- `availability.validate_blackout_windows` (§10.0 step 1, §10.5b) belongs to lane J5.
- Intraday distribution cost: measured 53 s (J2 total about 77 s, with 18 s in the chunk loop) at 100 worlds × 1,000 EVs on the synthetic per-EV fixture; about 80 s accepted for now.

Lead notes on the J5 build (review of `0a324a9`, 29 September 2026):

- `max_event_length_hours` (§10.5a) is NaN when the 0.5-h row's `window_mean_mw_p50` is at most 0 (or NaN): with no half-hour figure there is no event to size, and a zero base would otherwise qualify every duration.
- The settlement file (§10.5d) is unit-major: every slot of one meter point together, EVs in `units` order, with all the fleet `baseline_effect` rows as one block at the end.
- Compare shows the firm-MW rows (`product_sheet` evening and overnight turn-down 1 h, `firmness_by_manufacturer` 1-h turn-down `firmness_p50`) side by side for runs A and B, with no per-world pairing and no difference column: the frames hold no per-world values, and a difference of two quantiles is not a quantile of the paired difference.
- `supplier_positions` has one schema on every action result (option b): the §10.5f columns are always present, with the 1-h firm figures NaN when the run has no `availability_bands`.

Lead notes on the J1b review (29–30 September 2026):

- W1: a warm-up plan still in force at the study start uses night 0's uniform and `ρ`, maker outage included (current behaviour kept; §10.1d amended).
- W2: the trader's expected plan (`market.expected_smart_kwh`) applies the blackout rule (`action.plan_around_blackout` with the study blackout mask), as the kernel's planner does.
- W3: the §9.5-records reproduction test is struck (the per-archetype records and that code path are retired by §10.1e).
- Q4: the R1 test wording is as in §10.1e: known-part inputs and mean exact, quantiles within one grid bin each.
- The supplier dispatch success rate (supplier contract §4.2) is filled from `plan_status`, session weighted over the selected-path sessions ending in the study.

Open:

1. **Bank-holiday shift.** The bank-holiday switch reuses the half-term shift (+0.30 on the skip logit for its one date) rather than inventing a second value; confirm or set.
