# Intraday dispatch contract v1 (price-responsive dispatch)

Status: draft for lead acceptance, 29 September 2026 (task W3-contract-intraday), revised after the lead's review of `18dee6f` (blocking items B1–B6, the answers to Q1–Q13 and the optionals, all applied and recorded in §14). Design only: no model code exists for anything below. Base `b960ffd` with `main` at `e60a9ed` merged (the final firm-MW §10 with per-plan `plan_status`, `plan_remaining_need_kwh` and `decision_plan_kwh`; decision 0004 items 63–65).

This file is the "§11" addendum that plan §12 asks for, kept as its own file because `docs/contracts/trading-events-v1.md` is being revised by other writers. It extends that contract's §4–§5 (trading overlay and frames), §9.2 (commitment share) and §10 (firm-MW availability, now on `main`). It references those sections rather than restating them; where this file changes an approved rule it says "Changed by §11.x" and the lead carries the note into that file at integration. `docs/contracts/supplier-v1.md` is read as on `claude/w3-contract-supplier-pnl` at `01b1081`.

Authority: decision 0004 item 62(b) (Mike, 29 September 2026: "a committed share of expected flexibility is locked day-ahead; the rest re-optimises intraday as prices move. When an intraday price spikes, plugged-in EVs with slack turn down as far as they can still reach target; when prices are very cheap (or negative), EVs with headroom turn up. Responses can be triggered by a request (DFS, local turn-up) or by the price itself crossing an editable spread threshold, and requests are valued against the price curve at that time"), which promotes plan §2F lane F3 from optional to required; items 48–65 as they shape the planner, prices, availability and horizons; decision 0005 (ledger buckets); plan §12; trading-events-v1 §0–§10; `docs/contracts/results-v2.md` §1 rules for every frame; the lead's review decisions of §14.

Every price, volume and £ figure below is synthetic or illustrative. The only net money figure remains "illustrative simulated trading P&L" (decision 0005), never "Axle cash".

## 0. What changes, in one paragraph

Today the selected path plans each session once, at plug-in, on the day-ahead prices published by then (B4 rule, item 48), and never looks at a price again. The trading overlay (F2, §4.5) re-positions hourly on the intraday price but the fleet does not move. This addendum makes the fleet follow the price: a committed share `c` of EVs is **locked** to that day-ahead behaviour, and the remaining **free** EVs re-plan at every hourly decision instant on the latest intraday price known then, moving charging out of spiking half-hours and into cheap ones as far as the session can still reach its target by expected departure, but only when the new plan beats an editable threshold plus the round-trip bid–ask spread per MWh moved. Grid requests enter the same ranking as a price adjustment, so a spike, a DFS payment and a cheap half-hour compete on one scale. The kernel simulates a third path (the day-ahead plan, for reference) beside unmanaged and the dispatched path; the dispatched path is what `path_id="selected"` reports and the ledger settles; the intraday P&L splits into rebalancing (the trades the trader makes when the fleet stays on its day-ahead plans) and re-optimisation (the extra trades the dispatch's response causes), both at the traded intraday prices. No new random draw is taken.

## 1. Terms

| Term | Meaning |
| --- | --- |
| Day-ahead plan path | The selected path as contracted today: every EV planned at plug-in on B4-visible day-ahead prices plus known request adjustments; re-planned only as §2.4 / H6 / §10.5b say. Kernel key `"day_ahead"`. Reported as reference series `day_ahead_plan`, never as a `path_id` |
| Dispatched path | The selected path with this addendum on: locked EVs behave exactly as on the day-ahead plan path; free EVs re-plan hourly on the latest intraday price (§4). Reported as `path_id="selected"` |
| Locked EV | An EV whose plans never rank an intraday price (§2), `units.dispatch_locked` |
| Free EV | Any other EV; the intraday-dispatchable remainder |
| `c` | `trading.day_ahead_commitment_share` (§9.2 record, default 0.8): the share of the forecast deviation sold day-ahead **and** (Changed by §11.2) the share of EVs locked |
| `τ` | A decision instant: a run-slot start on a whole London hour (= UTC whole hour) from the last warm-up night onward; `s(τ)` its slot |
| `P̃_t(τ)` | The latest price of slot `t` known at `τ` (§3), the one rule for the dispatcher and the trader |
| `θ` | `trading.replan_threshold_gbp_per_mwh` (§9, £10/MWh illustrative): the editable spread threshold |
| `s` | `trading.half_spread_gbp_per_mwh` (§7 of the trading contract, default 1.0): half bid–ask, so a move costs `2s` per MWh moved (sell more turn-down in one slot, buy back in another) |
| Frozen-book re-run | The §4.5 intraday loop run on the day-ahead plan path's book, at the actual traded prices: what the trader would have traded had the fleet stayed on its day-ahead plans (§6.2) |
| Rebalancing | The frozen-book re-run's trades, valued at the traded intraday prices: intraday trading caused by forecast updates alone (arrivals, non-responders in the book, early departures) |
| Re-optimisation | The actual trades minus the frozen-book trades, at the traded prices: what the dispatch's response to the price caused |

Signs, units and slot keys follow trading-events-v1 §0. Energy moved is grid kWh; `kWh ÷ 1000 × £/MWh` is the only money formula.

## 2. The split: locked and free EVs (item 62(b) "committed share")

**Choice: by EV, not by energy fraction.** Each EV is either locked or free for the whole run. Rejected: splitting each session's need into a locked `c` part and a free `1 − c` part. Two plans per EV would share one charger cap per slot (a joint constraint the cheapest-slots rule does not express), "turn down as far as the EV can still reach target" does not decompose across the halves, and the ledger could not say which EVs delivered the day-ahead commitment. By EV, the locked set is the physical meaning of the day-ahead commitment: an aggregator that sold firm turn-down day-ahead keeps those EVs on the plan it sold, and chases the intraday price with the rest.

**Assignment (deterministic, no draw; the control group's counting rule of §9.5 without its permutation).** `K = round-half-up(c × N)` EVs are locked. `K` is split across cohorts by largest remainder on `K × n_c ÷ N` (ties to the earlier cohort in the defaults' order), giving `k_c`; the locked EVs are the first `k_c` EVs of each cohort in `units` row order. Within a cohort the row order is already arbitrary (the population's cohort permutation), so the locked set is exchangeable with any other subset in distribution, and editing `c` moves no random channel. `units.dispatch_locked` (bool) carries the flag on **every** result: True for those EVs on an action result with `trading.intraday_dispatch` on, False for every EV otherwise (a no-action result, or the switch off). Control-group EVs (§9.5) and non-responders keep their own status whichever set they fall in: a locked or free EV that ignores its plan charges by the normal rule as §10.1e says. `c = 1` locks every EV (the dispatched path equals the day-ahead plan path); `c = 0` frees every EV.

**Why `c` serves twice.** Mike's framing is one proportion: "a proportion goes to trading spread and locking in, then you have intraday trading". The day-ahead position `q = c × F̂` (§9.2) and the locked share are two views of the same commitment, and one record keeps a Compare of `c` honest (both move together).

**Newsvendor mode (§10.4; lead decision Q3).** With `trading.commitment_rule = "newsvendor"` the day-ahead position comes from the error quantile and no longer scales with `c`; the locked share is still `c`. The two are then independent settings: `c` says how much of the fleet is held to its day-ahead plan, the newsvendor rule says how much turn-down is sold on it. The Position caption says so in that mode. A locked share derived from the committed volume was rejected because the volume varies by slot and cannot name an EV set.

Stated limitation: the locked set's expected deviation is about `c` of the fleet's, not exactly (cohort mix and personal multipliers), so `q = c × F̂` and "what the locked EVs deliver" differ by ordinary forecast error; the ledger shows it as imbalance, as it does every other error.

## 3. The latest known price: one rule (`action.latest_known_prices`)

For world `w`, slot `t` and decision instant `τ` (int64 UTC ns), with `pub_t` the day-ahead publication instant of `t` (`sampling.day_ahead_publication_utc_ns`), `gate_t = start_t − gate_closure_minutes` and `H = 36` intraday steps (`intraday_path_gbp_per_mwh[w, t, h]`, `h` whole hours before gate closure, `h = 0` the close):

```
h_t(τ)  = clip( ceil((gate_t − τ) / 1 h), 0, H − 1 )
P̃_t(τ) = visible_day_ahead_prices(day_ahead, starts, τ)[w, t]     if pub_t > τ        (B4: unpublished, the expected shape)
        = intraday_path_gbp_per_mwh[w, t, h_t(τ)]                  otherwise
```

- `ceil` picks an update that has already happened at `τ` (trading contract review B6); `floor` would read a future update.
- A slot past gate closure (`τ ≥ gate_t`) ranks at its close, which was fixed at `gate_t ≤ τ` and so is known. Its deviation from position settles at SIP (§4.7); the dispatch may still move energy there (§4.2 says why).
- For `h_t(τ)` above the slot's last update the path equals the expected close, `P_DA + premium` (the generator's construction), so at publication a free EV and a locked EV rank the same prices up to the constant `da_id_premium_gbp_per_mwh` (0 by default).
- It reads only updates made at or before `τ` and prices published by `τ`, so perturbing any later update or unpublished price never changes its output (tested, §8).

Signature: `latest_known_prices(intraday_path (world, run slot, H), visible_day_ahead (world, run slot), publication_utc_ns (run slot,), gate_utc_ns (run slot,), decision_utc_ns) -> (world, run slot)`. The kernel calls it per decision on the EVs' windows (a gather, §5.2); `market.py` calls it for the expected plan of free sessions (§6.1). It is the intraday sibling of `visible_day_ahead_prices` and lives beside it.

## 4. Intraday re-dispatch on the dispatched path

### 4.1 Decision instants, the plan at plug-in and the re-plan

Free EVs plan and re-plan at the latest known price; locked EVs plan on B4-visible day-ahead prices exactly as today. In `_smart_charging_slot` on the dispatched path the steps of one run slot `s` starting at `τ` are, in this order (B1; binding on the F2 book as well):

0. **Re-plans (new; free EVs; decision slots only).** Before anything about slot `s` is read, at a slot whose start is a whole London hour. A candidate is an EV that at the slot start has a plan in force (`start ≤ s < end`), is free, has `plan_status = 0` (B6: the session follows its plan and its maker is online), has remaining planned energy `r = Σ_{t ≥ s} g_t > 0` (B5: the incumbent plan's own kWh from `s` on, not the `remaining_kwh` counter, which actual import decrements) and has at least two slots left in `[s, end)`. Nothing about connection in slot `s` is read: whether the EV stays plugged in through `s` is not known at `τ`, so a candidate that leaves in `s` is re-planned in vain and then voided at step 1, which is harmless. With `g_t` the incumbent's kWh for `t ∈ [s, end)` and the ranking price `π_t = P̃_t(τ) + adj_t(τ)` (the request adjustments announced by `τ` for the EV's zone, §2.4), `+inf` outside the window and on blackout slots (whose fixed energy is carried as §10.5b):

   ```
   n = fixed + plan_cheapest_slots(r − Σ fixed, cap, π)        (fixed = the incumbent's energy on blackout slots, 0 with no blackout)
   saving_gbp = Σ_{t: π_t finite} (g_t − n_t) π_t / 1000       (≥ 0: n is optimal on π; = 0 when the window cannot hold r, then n = g)
   moved_kwh  = Σ_t max(0, g_t − n_t)                          (energy leaving its incumbent slots; equal to the energy arriving elsewhere)
   re-plan iff  saving_gbp − (θ + 2 s) × moved_kwh / 1000  >  1e-9
   ```

   Slots with `π_t = +inf` carry `g_t = n_t = 0` and are skipped in the sum (B5: `0 × inf` is not a number). A re-plan replaces the incumbent's kWh at offsets `≥ s` by `n`, leaves `start`, `end`, `remaining_kwh` and the plan's status untouched (the total is unchanged), and counts one re-plan for the slot. **A re-plan is not a "plan made"** for §9.6 / §10.1e (B6): it resolves no `p_plan`, sets no `plan_status` and reads no uniform; a non-responder (`plan_status ≠ 0`) is never a candidate. Otherwise the incumbent stands.
0b. **Book snapshot (F2, where §1.4 / §10.1e put it).** `book_kwh["selected"][w, τ, :]` holds the plans in force at the slot start after step 0, so the trader re-positions on the re-plans made at `τ` at the same `P̃_t(τ)` the dispatch ranked (same `h`). A plug-in at `τ` is not in the book at `τ`: its plan is made in step 2, after the snapshot, and enters the next decision's book. `decision_plan_kwh` (§10.1e) is the same snapshot at `s_n`.
1. **Early departures**, as now.
2. **New plans at plug-in** (changed for free EVs). A free EV plugged in for the whole slot with no plan in force is planned from its stock, this slot and its next expected departure on `P̃_t(τ)` with `τ` = this slot's start (any slot, half-hour or hour), plus the request adjustments announced by `τ` for its zone, then `plan_cheapest_slots` as now. A locked EV uses the epoch's B4 prices as now. This is a "plan made": `p_plan` and `plan_status` are resolved as §10.1e says, and the blackout rule (§10.5b item 1) applies to both.
3. **This slot's energy**, as now.

**Cadence: hourly, matching §4.5** (lead decision Q6). The trader's decision instants are whole hours, and the trader's book at `τ` must hold the plans the fleet will follow, so the dispatch re-plans at those instants and not between them. Half-hourly re-planning would bring news for half the slots (updates are timed relative to each slot's own gate closure) but put the book out of step with the plans.

### 4.2 What the threshold rule does and does not do

- **Turn-down into a spike, as far as target allows.** A spike in a planned slot raises `π` there; the cheapest-slots re-plan moves that energy to the cheapest remaining slots. If the remaining window cannot hold `r` without the spiking slot, both plans fill every slot and `n = g`: the EV charges through the spike. The target is never sacrificed to a price; a session with no slack forgoes the spike (and any DFS payment) rather than leave below target. Paying a driver to accept a lower target is out of scope.
- **Turn-up when cheap or negative, within the preferred target.** A slot that has become cheap pulls energy in by the same rule, up to the session's remaining need. A negative price is just a very cheap slot. "Notably cheap" is exactly `θ`: the new plan must save at least `θ` per MWh moved after paying the spread.
- **Requests valued against the curve.** A turn-down window adds `+payment` to `π` inside it (charging there forgoes the payment), a turn-up window `−payment` (§2.4). The dispatch responds to whichever is worth more on that one scale: a £500/MWh DFS window beats a £200/MWh spike; a £2,000/MWh spike outside the window beats the DFS. Payment is still settled per §2.4 (delivery against the baseline, capped, once), and trading stays paused in request windows (§4.6); the spike's value flows through intraday P&L and imbalance.
- **The ranking price is a proxy for value, accepted for v1 (lead decision B4).** The ledger pays turn-down only: settlement is one-sided (`V = mask × max(0, B − M)`, §4.2 of the trading contract) and the customer's leg is valued at the day-ahead price (B3). So a free EV turning up into a cheap or negative half-hour, or importing above the baseline in any half-hour, earns nothing in the trading ledger, and the ranking price over-states what the ledger will pay for such a move. Its value is real for a supplier and shows in the supplier P&L (`docs/contracts/supplier-v1.md` §3.2: `M_t` is the selected-path metered import, so it includes the dispatch, and the smart hedge error `Σ_t (M_t − H^S_t)(SIP_t − P_t)` prices an intraday turn-up at `SIP − DA`), which is where a supplier sees it. The captions of §10 say so.
- **Noise is not chased.** The intraday path drifts even without news (the unrevealed expected surprise impact decays as reveal windows pass) and carries hourly updates of a few £/MWh (`intraday_hourly_sd_gbp_per_mwh` 2.5). With `θ = 0` and `s = 0` any strictly cheaper plan wins; the spread alone stops a £1/MWh wobble; `θ` sets the margin above that. `θ` is a policy knob, `s` a market fact, so they are separate records.
- **Herding.** Every free EV in a world ranks the same intraday path, so they bunch into the same intraday-cheap slots as smart EVs bunch into the cheapest day-ahead slots (item 48). The coincidence factor and weekly peak on the selected path report it; expected finding, not a defect.
- **Worked example** (one free EV, four slots left, cap 5 kWh/slot, `r = 6`, incumbent `g = [5, 1, 0, 0]` planned on `[30, 40, 80, 100]` £/MWh, `θ = 10`, `s = 1`, so the bar is `£12/MWh` moved):

  | latest `π` | `n` | saving | moved | bar | re-plan? |
  | --- | --- | --- | --- | --- | --- |
  | `[30, 400, 80, 100]` (spike in slot 1) | `[5, 0, 1, 0]` | `(1×400 − 1×80)/1000 = £0.32` | 1 kWh | `£0.012` | yes: £320/MWh moved |
  | `[400, 40, 80, 100]` (spike in slot 0) | `[0, 5, 1, 0]` | `(5×400 − 4×40 − 1×80)/1000 = £1.76` | 5 kWh | `£0.06` | yes; still 6 kWh by slot 3 |
  | `[30, 40, 38, 100]` (slot 2 a little cheaper) | `[5, 0, 1, 0]` | `(1×40 − 1×38)/1000 = £0.002` | 1 kWh | `£0.012` | no: £2/MWh moved |
  | any, with `r = 20` | `[5, 5, 5, 5] = g` | 0 | 0 | | no: no slack |

### 4.3 What stays out of v1 (accepted by the lead, Q4–Q5)

- **Opportunistic turn-up beyond the preferred target** (charge past the driver's target when prices are negative). Not simple honestly: it needs a cap and a consent assumption with no source (the preferred target is the driver's setting, where ordinary home charging stops; the model's rule that it is not the physical ceiling, AGENTS.md and `physics._simulate_path`, protects it), it changes the next night's need so the day-ahead plan versus dispatched comparison would mix two effects, and it needs a wear or customer-terms record. No record is reserved for it (a record for an unbuilt feature is machinery). Consequence stated in Limits: a turn-up request late at night, when most sessions are at target, delivers little.
- **A lock override for very large spikes.** A locked EV never breaks its plan for a price; that is what "locked" means.
- **Price impact.** Prices are national and the fleet is a price-taker (item 53); the dispatch chases prices without moving them.
- **A strategy-specific dispatch** (`da_only` on the day-ahead plan path, `full` on the dispatched path; lead decision Q8). Rejected for v1: it breaks the one-physics-path rule of §4.6 and the §5.9 invariants (identical grid-event payments and unmet-charge penalty across strategies), and doubles the ledger's inputs. The value of the dispatch itself is a Compare with the switch off on the same futures (§6.4), the same route the plan uses for shock attribution (§8 Q12); the strategy captions of §10 say what each strategy's row means.

### 4.4 Interplay with the rest of the model

| With | Rule |
| --- | --- |
| Non-response (§10.1e; B6, Q7) | A non-responder keeps its plug-in plan on record and charges by the normal rule, on both smart paths. It is **never re-planned**: only `plan_status = 0` sessions are candidates, and a re-plan carries the plan's status rather than resolving a new one. What the book shows for it follows §1.4 / §10.1e |
| Manufacturer outage, control-outage event, control group | All give a non-zero `plan_status` to the sessions they cover: those EVs charge by the normal rule on both smart paths whether locked or free, and are not re-planned |
| Blackout windows (§10.5b) | Fixed energy on blackout slots is carried through every re-plan (`fixed` above); `π = +inf` there |
| Short-notice requests (H6; lead decision Q12) | On the dispatched path a free EV picks a newly announced request up at its next hourly decision through `adj_t(τ)`; no separate notice handling. Locked EVs follow whatever H6 makes the day-ahead plan path do (re-plan at the notice slot on day-ahead prices), so the identity "locked EV = day-ahead plan path" survives H6 |
| Trading masks (§4.6) | Unchanged: request windows pause trading; the dispatch still responds to them |
| Availability (§10.2) | Reads the selected path's plans in force per slot (`planned_home_import_kwh`, `plan_remaining_need_kwh`), which now change hourly for free EVs; the known part at `s_n` reads `decision_plan_kwh`, the snapshot of step 0b at `s_n`, so a re-plan after 17:00 never changes it (R1) |
| Newsvendor commitment (§10.4) | `q` from the rule; the locked share stays `c` (§2) |
| Customer cost effect (B3) | Valued at day-ahead on the selected path as now. A free EV moving into an intraday-cheap, day-ahead-dear slot raises the customer's day-ahead-valued cost while the aggregator earns intraday P&L; both are reported, neither is hidden, and the revenue share is the only reconciliation (§4.7's double-counting caption stands) |
| Supplier P&L (supplier-v1 §3.2) | `M_t` includes the dispatch; the smart hedge `H^S = expected_metered_kwh` is the `τ_DA` forecast and does not change; the hedge error at `SIP − DA` is where a supplier sees the intraday value of the dispatch (B4) |
| User price curve (§9.4) | The offset is applied to every intraday step, so `P̃` carries it |

### 4.5 Physical invariants (kernel checks run on every path)

Battery-energy conservation and the 0–100 % stock check per EV-slot (existing); home import `= min(plan, normal limit)` so charger power and the preferred target bound every slot (existing); charging only when connected for the whole slot (existing); home import is the only route a plan touches, so one dispatchable route per EV-slot holds (decision 0001). Added: a re-plan never changes a plan's remaining total when the window can hold it (`Σ n = Σ g = r`), and is a no-op when it cannot; the dispatched path draws nothing, so two runs with one seed give identical dispatch; slots past gate closure still obey every check above.

## 5. Kernel changes (`action.py`, `physics.py`)

### 5.1 Inputs: `IntradayDispatch` (frozen dataclass in `action.py`)

What the intraday dispatcher knows, beside `SmartCharging` (what the day-ahead planner knows). Built by `action.intraday_dispatch_inputs(settings, units, market_prices.intraday_path_gbp_per_mwh, *, replan_threshold_gbp_per_mwh, half_spread_gbp_per_mwh, gate_closure_minutes)`, reading `locked` from `units.dispatch_locked` (§2; the assignment itself is `action.locked_evs(units, commitment_share) -> (EV,) bool`, called by `forecast` when it builds `units`); the B4-visible day-ahead prices it needs at run time are already on `SmartCharging`:

| Field | Shape / type | Meaning |
| --- | --- | --- |
| `locked` | (EV,) bool | `units.dispatch_locked` |
| `intraday_path_gbp_per_mwh` | (world, run slot, H) float64 | The generator's path, warm-up and study (already in memory, about 11 MB at 100 worlds) |
| `publication_utc_ns`, `gate_utc_ns` | (run slot,) int64 | For §3 |
| `decision_slot` | (run slot,) bool | Start on a whole London hour and `price_epoch_by_slot ≥ 0` |
| `replan_threshold_gbp_per_mwh`, `half_spread_gbp_per_mwh` | float | `θ`, `s` |

`for_slice(world_index, ev_index)` slices `locked` and the path's world axis (chunked summaries, one-EV replay). Validation: `locked` per EV; the path finite and shaped `(evaluation worlds, run slots, H)`; `θ ≥ 0`, `s ≥ 0`. Nothing in it is unknown at the instant it is read: the kernel indexes the path only through `h_t(τ)`.

### 5.2 The third path and its outputs

`simulate_fleet_intervals(..., smart_charging=smart, intraday_dispatch=dispatch, dispatch_sums=holder)`. With `dispatch` given, three passes on the same sampled inputs: `normal`, `day_ahead` (smart only: the reference) and `selected` (smart + dispatch). The returned frames keep two paths (`normal`, `selected`); the reference path's outputs go to a `DispatchSums` holder, the `ZoneSums` pattern:

| `DispatchSums` field | Shape | Meaning |
| --- | --- | --- |
| `day_ahead_home_import_kwh` | (world, study slot) | Fleet home import on the day-ahead plan path |
| `day_ahead_zone_home_import_kwh` | (world, zone, study slot) | The same per zone (event response in a zone scope) |
| `locked_home_import_kwh` | (world, study slot) | Locked EVs' import on the selected path (equal on both smart paths; the validator checks it) |
| `free_home_import_kwh`, `free_day_ahead_home_import_kwh` | (world, study slot) | Free EVs on the selected and the day-ahead plan path |
| `replan_count` | (world, study slot) | Number of free EVs whose plan changed at that slot's decision (0 off decision slots) |
| `book_kwh` | dict path → (world, decision, 48) | F2's plan books for `selected`, `day_ahead` and `normal` (§1.4); the `day_ahead` book feeds the frozen-book re-run (§6.2) |

`simulate_unit_intervals` and the per-EV chunk pass (`_per_ev_chunks`) take the same `dispatch` (sliced) so per-EV frames, the One EV replay and the fleet frames reconcile; the chunk pass simulates `normal` and `selected` as today and not the reference path (no per-EV frame reads it). The One EV lens reads `units.dispatch_locked` (Q9; no replay scalar). `SimulatedForecast` gains `intraday_dispatch` and `dispatch_sums`. With `trading.intraday_dispatch` off, no third pass runs, the holder is `None`, and the selected path is the day-ahead plan path exactly as today.

### 5.3 Call order (§1.5 / §9.7 / §10.0 steps 2, 7 and 8 changed)

Step 2 adds `units.dispatch_locked` after the population draws (no draw; from `c`, the switch and the model). Step 7 builds `smart` as now, then `dispatch = intraday_dispatch_inputs(...)` when the switch is on; step 8 runs the kernel with both. Draw order: unchanged (§4.8, §10.0). Editing `c`, `θ`, `s` or the switch changes no draw and no price.

## 6. Trading overlay changes (`market.py`)

### 6.1 Expected plan per session (§4.4, §4.5 changed by §11.6)

- Day-ahead decision `τ_DA(n)`: unchanged. The trader knows only day-ahead prices; a free EV's expected plan on those prices is its best forecast (the intraday path is a martingale from them).
- Intraday decision `τ` (§4.5): for sessions not yet started, `expected_smart_kwh` receives per-session window prices: the B4-visible day-ahead price for a locked EV, `P̃_t(τ)` for a free EV (one `np.where(locked[:, None], visible, latest)`), plus adjustments known at `τ`. The free session's actual plug-in plan will use the price at its plug-in instant, which the trader cannot know; `P̃(τ)` is its best estimate.
- Forecast metered import `M̂_t(τ)` = the selected book at `τ` (which holds the re-plans made at `τ` and not the plug-ins at `τ`, §4.1) + that expected plan. Targets, trades, prices and the final position as §4.5.

### 6.2 Strategies and the frozen-book re-run (B2)

All three strategies settle on the one dispatched path (`V`, `R`, `E`, SIP; §4.6 unchanged):

| Strategy | Position | Dispatch it settles on | Reading (the caption of each row, Q8) |
| --- | --- | --- | --- |
| `da_only` | `q`, no intraday trades | dispatched | "Sold day-ahead and did not hedge the fleet's intraday moves: the dispatch's deviation from the day-ahead position is settled at the imbalance price" |
| `full` | `q`, then §4.5 on the selected book | dispatched | "The product: day-ahead position, hourly re-positioning on the dispatched fleet" |
| `perfect_foresight` | `V` at day-ahead | dispatched | "Perfect foresight of volume, not of intraday prices: sells the settled deviation of the dispatched fleet at the day-ahead price and never trades intraday" |

**Frozen-book re-run (the rebalancing definition).** Run the §4.5 loop once more with the fleet frozen on its day-ahead plans: the day-ahead plan path's book (`book_kwh["day_ahead"]`), every expected plan ranking the B4-visible day-ahead price (as F2 does), the same `q`, the same masks, the actual baseline `B_t(τ)` (one settlement baseline per run) and the **actual traded prices** `P_ID,t,h(τ)`. Its trades are `Δ⁰_t(τ)`; the actual `full` trades on the dispatched book are `Δ_t(τ)`. Then per (world, night):

```
intraday_pnl_rebalancing_gbp     = Σ_τ Σ_t Δ⁰_t(τ) P_ID,t,h(τ) / 1000
intraday_pnl_reoptimisation_gbp  = Σ_τ Σ_t (Δ_t(τ) − Δ⁰_t(τ)) P_ID,t,h(τ) / 1000     (= intraday_pnl_gbp − rebalancing)
trading_cost_rebalancing_gbp     = − s Σ_τ Σ_t |Δ⁰_t(τ)| / 1000
trading_cost_reoptimisation_gbp  = trading_cost_gbp − trading_cost_rebalancing_gbp     (may be positive: the dispatch's trades can net against the forecast's)
```

Rebalancing is the trading the forecast updates alone would have caused, at the prices it was actually done at; re-optimisation is the extra volume the dispatch's response caused, at the same prices. `Δ⁰` is the trade set F2's `full` strategy makes with the switch off, up to the in-day adjustment: `B_t(τ)` reads the dispatched fleet's 12:00–13:30 import, so the re-run's targets can differ from a switch-off run's by `a_n` (near zero, §4.1; exact with `trading.baseline_in_day_adjustment` off). With the switch off the two books coincide, `Δ = Δ⁰` in every row and re-optimisation is exactly 0: item 58's "intraday P&L in v1 is all rebalancing", literally. Rejected: valuing the same trades at day-ahead as rebalancing (it books a dispatch-caused trade's `Δ × P_DA` as rebalancing), and freezing the re-run's prices at day-ahead (it mixes the trader's re-pricing of forecast trades, mean zero by I2, into "what the dispatch caused"). Not split: imbalance and settled volume (the dispatch changes `V` too; §6.4 says where that shows).

**Capture rate and cost of uncertainty (B3).** `perfect_foresight` has perfect foresight of volume, not of intraday prices: it never trades intraday, while `full` earns re-optimisation P&L on price moves. So with the dispatch on, `capture_rate > 1` and `cost_of_uncertainty_gbp_per_week < 0` can occur and are not errors; the §9.1 rows carry the caption "against perfect foresight of volume, not of intraday prices". `trading_checks.p10_ordering` stays informative (passed always True), as §5.5 says.

### 6.3 Ledger and frames changed (§5.2, §5.3, §5.4, §5.5 changed by §11.6)

- `trading_ledger_world` and `trading_week_world`: four "of which" columns after the shock-attribution columns, in the order above. They are not buckets (`net_gbp` still sums nine); 0 on `da_only` and `perfect_foresight` rows; both pairs kept (Q11).
- `position_updates` (sampled worlds, `full`): two columns after `price_gbp_per_mwh`: `frozen_position_kwh`, `frozen_trade_kwh` (the frozen-book re-run's target and trade at the same (world, slot, decision), "frozen" meaning the fleet frozen on its day-ahead plans; on the day-ahead row both equal `q_t`).
- `trading_ledger_summary`: `metric` runs over the four new columns too (`GBP per night` / `GBP per week`).
- `trading_kpis` (columns as §5.5): `rebalancing_gbp_per_week` and `reoptimisation_gbp_per_week` (`GBP per week`, `cvar5` filled) on the `full` row only; `dispatch_moved_mwh_per_week` (`MWh per week`: `Σ_t max(0, −moved_kwh_t) / 1000` from §7.1, the energy the dispatch moved out of its day-ahead half-hour; screen label "Energy moved by intraday dispatch"; a fleet property, identical on the three strategy rows like `day_ahead_spread`; 0 by definition with the switch off).
- `trading_checks`: `dispatch_split_sums_to_intraday` (max over `full` rows of `|rebalancing + reoptimisation − intraday_pnl|` and the same for trading cost; tolerance `1e-9 × max(1, |bucket|)`).

### 6.4 Where the value of the dispatch shows

`reoptimisation_gbp_per_week` is the intraday P&L the dispatch's response produced, before imbalance, and is not the dispatch's whole value: it leaves out the changed `V` and its SIP settlement, the customer's day-ahead-valued cost, and every move the one-sided ledger does not pay (B4: turn-up into cheap half-hours and import above the baseline earn nothing in the trading ledger). The whole value is a Compare run with `trading.intraday_dispatch` off on the same seed, paired by world in every frame; the supplier's share of it is the hedge-error saving of the supplier P&L (supplier-v1 §3.2), which prices `M − H^S` at `SIP − DA`. The Trading P&L caption says exactly this.

## 7. Result frames (action results; `None` with the switch off or on a no-action result, except `units.dispatch_locked`)

### 7.1 `dispatch_world_slot` (exact)

One row per `(world_id, slot_index)`, world-major.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `world_id`, `slot_index`, `night_index` | int64 | Keys |
| `interval_start_utc`, `interval_start_london` | UTC, London | |
| `day_ahead_plan_kwh` | float64 | Fleet home import on the day-ahead plan path |
| `dispatched_kwh` | float64 | Fleet home import on the selected path (equals `fleet_world_intervals`) |
| `locked_kwh`, `free_kwh`, `free_day_ahead_plan_kwh` | float64 | §5.2 |
| `moved_kwh` | float64 | `dispatched_kwh − day_ahead_plan_kwh` (= `free_kwh − free_day_ahead_plan_kwh`); negative = turned down against the day-ahead plan in that half-hour, positive = turned up. Caption: "over a session night it sums to about zero: the dispatch moves energy in time, not in amount, apart from early departures and public top-ups" |
| `replan_count` | int64 | §5.2 |
| `latest_close_gbp_per_mwh` | float64 | The intraday close (repeated from the market frames so the view needs one frame) |
| `evidence_kind` | object | `illustrative_synthetic` |

### 7.2 `dispatch_bands` (exact)

One row per `(series, metric, slot_index)`, `series` in `day_ahead_plan`, `dispatched`, `difference` (dispatched minus day-ahead plan inside each world first), `metric` `home_import_kw` (kWh ÷ 0.5), plus `series="dispatched"`, `metric="replan_count"`. Columns: `series`, `metric`, `unit`, `slot_index`, `interval_start_utc`, `interval_start_london`, `world_count`, `mean`, `p10`, `p50`, `p90`, `evidence_kind`. World-first quantiles (results-v2 §1 rule 3).

### 7.3 `dispatch_split` (exact) and `units.dispatch_locked`

`dispatch_split`: one row per cohort in the defaults' order: `cohort_id`, `ev_count`, `locked_count`, `free_count`, `commitment_share`, `replan_threshold_gbp_per_mwh`, `evidence_kind`. The counts reproduce §2 exactly and equal the per-cohort sums of `units.dispatch_locked`. `units.dispatch_locked` (bool) is present on every result (§2).

### 7.4 `event_response_bands` (§5.6 changed by §11.7)

Series `day_ahead_plan` with metric `home_import_kw` (the scope's import on the day-ahead plan path) is added after `selected` so the event view shows day-ahead plan, dispatched and unmanaged around a scripted event. Present only with the switch on. Zone scopes read `day_ahead_zone_home_import_kwh`.

### 7.5 `ForecastResult`

New fields `dispatch_world_slot`, `dispatch_bands`, `dispatch_split` (all `None` unless action and switch on); `units` gains `dispatch_locked` on every result; plus the ledger, updates and KPI changes of §6.3. `path_id="selected"` keeps its name; its meaning extends to "the dispatched fleet" (results-v2 §1 rule 2, one line for the lead to carry over).

## 8. Validator rules (`validate_result_v2` additions, one fixture module) and sanity tests

Validator:

- Column sets, order and dtypes as §7; keys unique; frames `None` exactly when the switch is off or the model is no-action; `units.dispatch_locked` bool on every result.
- `units.dispatch_locked`: all False on a no-action result and with the switch off; otherwise `K = round-half-up(c × N)` True values, split by largest remainder across cohorts, the first `k_c` rows of each cohort.
- `dispatch_world_slot`: `dispatched_kwh` equals the selected-path `home_grid_import_kwh` of `fleet_world_intervals`; `locked_kwh + free_kwh = dispatched_kwh` and `locked_kwh + free_day_ahead_plan_kwh = day_ahead_plan_kwh` within 1e-9 kWh; `moved_kwh` recomputed; `replan_count ≥ 0`, 0 on non-decision slots, `≤ free_count` summed over cohorts; `latest_close_gbp_per_mwh` equals the realised frame's close.
- `dispatch_bands` equal to the world-first recomputation; ordered quantiles.
- `dispatch_split`: counts as §2; sum `= vehicle_count`; equal to the sums of `units.dispatch_locked`; `commitment_share` and `replan_threshold_gbp_per_mwh` equal the records.
- Ledger: each split pair sums to its bucket within `1e-9 × max(1, |bucket|)`; both pairs 0 on `da_only` and `perfect_foresight` rows; for sampled worlds `intraday_pnl_rebalancing_gbp` equals `Σ frozen_trade_kwh × price_gbp_per_mwh / 1000` over the intraday rows of `position_updates` (B2: the row's traded price) and `trading_cost_rebalancing_gbp` equals `−s Σ |frozen_trade_kwh| / 1000`; with the switch off `frozen_trade_kwh == trade_kwh` in every row and both re-optimisation columns are exactly 0; `dispatch_moved_mwh_per_week` identical across strategy rows and equal to the recomputation from `dispatch_world_slot`; `frozen_position_kwh` on the day-ahead row equals `position_da_only_kwh`.
- `event_response_bands`: the `day_ahead_plan` block present iff the switch is on; ordered quantiles.

Sanity-check tests (checklist; each lane adds the ones it implements):

- [ ] **Equality with the day-ahead plan.** With `intraday_hourly_sd_gbp_per_mwh = 0`, both shock classes' surprise share 0 (`*_shock_known_share = 1`) or every shock rate 0, no scripted surprise and `da_id_premium_gbp_per_mwh = 0` (a constant premium also passes, away from the cap and floor): the selected path equals the day-ahead plan path in every kernel frame, `moved_kwh = 0`, `replan_count = 0`, both re-optimisation columns 0 and every strategy's net equals its value with the switch off.
- [ ] **Locked identity.** Per-EV import of every locked EV is identical on the two smart paths (per-EV chunk pass); `c = 1` makes the whole selected path equal the day-ahead plan path whatever the prices; `c = 0` frees every EV; `K` and `k_c` as §2 (including a half-up case, for example `c = 0.5` with `N = 5`); editing `c` changes no draw; `units.dispatch_locked` is all False on a no-action result and with the switch off.
- [ ] **Turn-down never breaks target when slack allows.** The §4.2 worked example as a unit test of the re-plan rule (all four rows, `r` taken from the plan array, `+inf` slots skipped), and a kernel fixture: one free EV, a spike placed in a planned slot with two or more spare slots before expected departure moves the energy out and the EV still reaches target at departure; with the window exactly full the plan is unchanged and the EV charges through the spike; battery conservation holds on both paths.
- [ ] **Turn-up.** A negative price appearing in an unplanned slot inside the window pulls energy into it up to the remaining need and never beyond the preferred target; a turn-up request known at `τ` does the same through `adj`; a session already at target delivers nothing.
- [ ] **Requests on one scale.** A DFS window (`+500`) and a `+200` spike outside it: the plan avoids the window; a `+2,000` spike outside the window: the plan charges in the window instead; delivery and payment as §2.4 in both cases.
- [ ] **Threshold and spread.** `θ` very large: no re-plan anywhere (free EVs equal their day-ahead plans); `θ = s = 0`: a re-plan whenever a strictly cheaper plan exists; a `£2/MWh` saving with `s = 1` does not re-plan; the bar rises by exactly `2` per unit of `s` (hand fixture).
- [ ] **No leakage.** Perturbing `intraday_path[w, t, h]` for every `h < h_t(τ)` (updates not yet made at `τ`) and every unpublished day-ahead price leaves the selected book at `τ` and the dispatched import up to `s(τ)` unchanged; the same for a scripted surprise whose reveal is after `τ`. `latest_known_prices` returns the close for a slot past gate closure and the B4 shape for an unpublished slot. A re-plan at `τ` reads nothing about connection in `s(τ)`: perturbing the realised unplug inside `s(τ)` leaves the re-plan made at `τ` unchanged (the plan is then voided at step 1 in the perturbed run).
- [ ] **Decision instants.** Re-plans occur only on whole-hour slots from the last warm-up night; a plug-in on a half-hour slot ranks `P̃` at that half-hour; across a clock change the hourly instants stay on UTC hours.
- [ ] **Frozen-book re-run.** With the switch off, `frozen_trade_kwh == trade_kwh` in every sampled row and both re-optimisation columns are exactly 0; with the switch on and `trading.baseline_in_day_adjustment` off, `Δ⁰` equals the trade set of the same seed run with the switch off, row for row; the split pairs sum to their buckets; on a fixture where one free EV moves energy between two open slots at `τ`, `Δ − Δ⁰` is `+m` in the slot it left and `−m` in the slot it entered, and the trading-cost remainder is `−2sm/1000` (or positive on a fixture where the move nets against a forecast trade).
- [ ] **Book order (B1).** The book at `τ` holds the re-plans made at `τ` (a free EV whose plan changed at `τ` shows the new plan in `book_kwh["selected"][w, τ]`); a plug-in at `τ` is not in the book at `τ` and is in the next decision's book; `decision_plan_kwh` at `s_n` equals the book at `s_n` for EVs plugged in for the whole of `s_n`, plus the plans of sessions plugging in at `s_n` (lead ruling at K1 approval: the 17:00 snapshot stays where the plan-status wiring put it).
- [ ] **Blackout and non-response (B6).** A re-plan keeps blackout-slot energy fixed; a non-responder is never re-planned (its plan on record equals its plug-in plan in every slot, and its import equals its normal-path import); a control EV, a maker-outage session and a control-outage session are never moved; a re-plan leaves `plan_status` unchanged and consumes no uniform.
- [ ] **Pairing.** Editing `θ`, `c`, `s` or the switch changes no draw and no price; two runs with one seed give identical `dispatch_world_slot`.
- [ ] **Frames.** Validators above on the fixture; the no-action result and the switch-off result carry `None` and an all-False `units.dispatch_locked`.

## 9. Assumption records (`model/assumptions.py`, `TRADING` group)

| Name | Default | Unit | Evidence | Authority |
| --- | --- | --- | --- | --- |
| `trading.intraday_dispatch` | on (lead decision Q1; seeded pins regenerated deliberately, plan §3; the milestone verification follows) | switch | illustrative product choice | decision 0004 item 62(b), plan §12 |
| `trading.replan_threshold_gbp_per_mwh` | 10 (lead decision Q2) | GBP/MWh, `≥ 0` | illustrative: a margin above the £2.5/MWh hourly update SD and the £2/MWh round-trip spread | item 62(b) "editable spread threshold" |
| `trading.day_ahead_commitment_share` (existing §9.2 record) | 0.8 | fraction, 0–1 | illustrative | item 58 (C); **Changed by §11.2:** also the locked share of EVs |
| `trading.half_spread_gbp_per_mwh` (existing) | 1.0 | GBP/MWh | illustrative | reused in the re-plan bar (`2s`) |

No opportunistic turn-up record (§4.3). The editor labels on screen: "Intraday dispatch follows the latest price", "Re-plan threshold (£/MWh moved)", and the commitment share's affects-text gains "and the share of EVs locked to their day-ahead plan".

## 10. UI consumption map (views read frames; they compute no plan, position or money)

| View | Reads |
| --- | --- |
| Smart charging ▸ Response, new "Day-ahead plan vs dispatched" block | `dispatch_bands` (`day_ahead_plan` and `dispatched` P50 with P10–P90; `difference` as a second panel; `replan_count` P50 as a thin bar row); the representative world's `dispatch_world_slot` (`day_ahead_plan_kwh`, `dispatched_kwh` as kW and `latest_close_gbp_per_mwh` on a secondary axis) with shock spans from `market_shocks` (that world, `in_study`) so the response around surprises is visible; caption from `dispatch_split` ("`K` of `N` EVs locked to their day-ahead plan (share `c`); free EVs re-plan hourly when the saving exceeds £`θ`/MWh moved plus the spread") and the `moved_kwh` caption of §7.1. Legend labels: "Unmanaged", "Day-ahead plan", "Dispatched (follows the latest intraday price)" |
| Smart charging ▸ Response, event view | `event_response_bands` with the `day_ahead_plan` series beside `selected` and `normal` |
| Trading ▸ 3 P&L and risk | The intraday P&L and trading-cost bars split into rebalancing and re-optimisation from `trading_ledger_summary` week rows (`mean`; "means add up, quantiles do not"; "the re-optimisation trading cost can be positive when the dispatch's trades net against the forecast's"); tiles `reoptimisation_gbp_per_week` (P50, P10, CVaR5) and `dispatch_moved_mwh_per_week` ("Energy moved by intraday dispatch") from `trading_kpis`; the §6.4 caption; the B4 caption ("the trading ledger pays turn-down only and values the customer's leg at day-ahead, so turn-up into cheap half-hours and import above the baseline earn nothing here; the supplier's share of that value is the hedge-error saving on the Supplier page"); the strategy captions of §6.2 on the strategy rows, and "against perfect foresight of volume, not of intraday prices" on the capture-rate and cost-of-uncertainty rows (B3); `trading_checks` row |
| Trading ▸ 2 Position | Unchanged; the position fan already shows the trades. Hover on `position_updates` may show `frozen_trade_kwh`. In newsvendor mode the caption says the locked share is `c` and the position is not (§2) |
| Overview ▸ Key stats, section F | Rows `reoptimisation_gbp_per_week` and `dispatch_moved_mwh_per_week` (`full`), labelled illustrative |
| Supplier ▸ P&L (supplier-v1 §11) | No new frame; its hedge-error caption gains "includes the intraday dispatch's moves, priced at SIP − DA" when the switch is on |
| Drivers ▸ One EV | "Locked to day-ahead plan" / "Re-planned intraday" from `units.dispatch_locked` |
| Compare | The new `trading_kpis` rows flow through the existing records (paired by world) |
| Edit assumptions | The two new records and the commitment share, on the Trading tab |

Every £ label reads "illustrative simulated trading P&L" or "illustrative". Charts use the shared Plotly template (decision 0004 item 29).

## 11. Runtime and memory (estimate; the K1 writer measures and reports)

At 1,000 EVs × 100 worlds: a third kernel pass adds about half the current two-path kernel time. Re-plans: at each of about 192 decision slots, `plan_cheapest_slots` over the free EVs with a plan in force (at most `(1 − c) × N × W` rows, 20,000 at `c = 0.8`, fewer early in the evening) with a window width of about 50: roughly 20–40 ms per decision, 5–10 s per run at `c = 0.8`, up to about 40 s at `c = 0`; the price gather and saving test are one fancy-index and two reductions per decision. The per-EV chunk pass (`summaries._per_ev_chunks`) re-simulates the selected path per chunk and now re-plans too, so it costs about the same again (5–10 s at `c = 0.8`). The frozen-book re-run is one more F2-sized overlay pass (the F2 writer's measured cost). Memory: the intraday path is already held (about 11 MB); the extra book about 7 MB; `DispatchSums` under 2 MB. Item 33 allows the time; the writer reports the measured figures in the handoff.

## 12. Build lanes and dependencies (disjoint ownership; the lead adds one call line per lane at integration, as §10.7)

| Lane | Outcome | Owns | Depends on | Must add (tests) |
| --- | --- | --- | --- | --- |
| **K1** kernel dispatch (High risk: physics, planner cut-offs) | §3, §4, §5 complete: `latest_known_prices`, `locked_evs`, `IntradayDispatch`, `intraday_dispatch_inputs`, the re-plan rule as a pure function (`replan_when_worth(...)`), step 0 and the snapshot order, the third pass, `DispatchSums`, the two `TRADING` records | `action.py`, `physics.py`, the two-record hunk in `assumptions.py`, `tests/model/test_intraday_dispatch.py` | F2's kernel plumbing (`book_kwh`, snapshot) and J1b (`_smart_charging_slot` per-plan status, `plan_remaining_need_kwh`, `decision_plan_kwh`, blackout) merged: all edit `_smart_charging_slot`. K1 after J1b (lead decision Q13); K1 and H6 are independent of each other, and whichever lands second rebases onto the first | Rows 1–8, 10–12 of §8; battery conservation and one-route on the fixture; the timing of §11 |
| **K2** overlay split (High risk: ledger) | §6 complete | `market.py` hunks, `tests/model/test_market_dispatch.py` | F1/F2 and J4 merged; K1's `latest_known_prices` and `book_kwh["day_ahead"]` | Row 9 of §8; the worked example of §4.4 of the trading contract with a free session; ledger validator rules; the B3 caption rows |
| **K3** frames and validator (Normal) | §7 complete; `slim_run` records | `summaries.py` hunks (frame builders, the `event_response_bands` series), `tests/fixtures/dispatch_contract.py`, `tests/model/test_dispatch_summaries.py` | K1 (develops against a hand fixture of `DispatchSums`; integrates after K1) | Frame validators; world-first recomputation; `None` rules; the `units.dispatch_locked` rules |
| **K4** forecast wiring (Normal) | `SimulatedForecast` and `ForecastResult` fields; `units.dispatch_locked`; the call lines of §5.3; the regenerated seeded pins (Q1) | `forecast.py` hunk (after K1 and K3 merge), the pin files | K1, K3 | An action run with the switch on and off; a no-action run |
| **K5** UI (Normal) | §10 complete | Response and Trading P&L view hunks, Key stats, One EV and Supplier caption hunks, Parameters tab hunk, `tests/ui/test_action_response.py` additions and the touched view tests | K2–K4 and the Trading and Supplier page lanes | AppTest renders with and without every optional frame; no view arithmetic on quantiles |

Order: K1 and K2 (K2 on a fixture book) in parallel once their dependencies merge; then K3; then K4; then K5. Review: K1 and K2 are High (lead-accepted interface, one strong independent review each; a specialist second review for K1's planner cut-off and conservation). The plan's sweep script (`scripts/sweep_trading.py`, F3) stays a separate optional lane.

## 13. Limitations (to state in Limits and captions)

Hourly cadence (news for half the slots waits up to 30 minutes); no lock override; no charging past the preferred target; price-taker (no price impact from the fleet's own moves); slots past gate closure rank at the close although their deviation settles at SIP; with a non-zero `da_id_premium` a free EV ranks published slots on `P_DA + premium` and unpublished ones on the B4 shape, a constant bias across the publication boundary (0 at the default premium; a limitation, lead decision Q10); the ranking price is a proxy for value: one-sided baseline settlement and the day-ahead-valued customer leg pay nothing in the trading ledger for turn-up into cheap half-hours or for import above the baseline, and the supplier P&L's hedge error is where that value shows (B4); the frozen-book re-run keeps the actual baseline, so its trades match a switch-off run only up to the in-day adjustment; the locked set's expected deviation is about, not exactly, `c` of the fleet's; a Compare with the switch off is the only whole-value figure; re-optimisation is "the extra trades the dispatch caused, at traded prices", including the dispatch's herding into intraday-cheap slots; a non-responder is never re-planned, so its plan on record is its plug-in plan; a re-plan that moves energy later in the window raises the early-departure shortfall risk (the departure margin of item 51 is the only protection; the selected path's `early_departure_shortfall_kwh` reports it, and a Compare with the switch off shows the change).

## 14. Resolved by the lead's review of `18dee6f`, and what remains open

Blocking items, all applied: B1 (re-plans first, on plans in force at the slot start, reading nothing about connection in the slot; then the F2 snapshot; then early departures, new plans and energy; a plug-in at `τ` is not in the book at `τ`; §4.1, §6.1, §8), B2 (the split by volume at the traded prices, `Δ⁰` from the frozen-book re-run; trading cost split the same way with the remainder possibly positive; exactly 0 with the switch off; validator on `frozen_trade_kwh × price`; the "exactly" claim softened to "up to the in-day adjustment"; §1, §6.2, §6.3, §8), B3 (`capture_rate > 1` and `cost_of_uncertainty < 0` can occur; the "perfect foresight of volume, not of intraday prices" caption; `p10_ordering` informative; §6.2, §10), B4 (the ranking price accepted as a proxy for v1; one-sided settlement and the day-ahead-valued customer leg stated in §4.2, §6.4, §13 and the captions; the supplier P&L named as where the intraday value shows), B5 (`r = Σ_{t ≥ s(τ)} g_t`; `+inf` slots skipped in the saving sum; §4.1), B6 (candidates need `plan_status = 0`; a re-plan carries the plan's status and is not a "plan made"; a non-responder is never re-planned; §4.1, §4.4, §8, §13).

Questions, as the reviewer accepted them: Q1 dispatch on by default, pins regenerated, milestone verification follows (§9, K4); Q2 £10/MWh illustrative (§9); Q3 the newsvendor-mode text (§2); Q4 opportunistic turn-up out of scope, Q5 no lock override, Q6 hourly cadence (§4.3, §4.1); Q7 non-responders never re-planned, with B6; Q8 `da_only` on the dispatched path, with captions (§6.2, §10); Q9 `units.dispatch_locked` on every result, the replay scalar dropped (§2, §5.2, §7.3); Q10 the premium boundary bias a limitation (§13); Q11 both split pairs kept (§6.3); Q12 locked EVs follow H6 (§4.4); Q13 K1 after J1b, whichever of K1 and H6 lands second rebases (§12).

Optionals, applied: the `K` rule (round-half-up of `c × N`, then largest remainder; §2), the `dispatch_moved_mwh_per_week` screen label (§6.3, §10), the `moved_kwh` caption (§7.1, §10), the early-departure effect in Limits (§13), the per-EV chunk pass in the runtime estimate (§11).

No open questions remain for this contract. Two notes for the lead at integration: results-v2 §1 rule 2's one-line extension of `selected` (§7.5), and supplier-v1's hedge-error caption (§10), both in files owned by other lanes.

## Lead notes at integration

- With the dispatch switch off no third kernel pass runs, so the frozen-book re-run uses the selected book (which then equals the day-ahead book).
- "Energy moved by intraday dispatch" is a net fleet figure per half-hour; the tooltip says "(net per half-hour)".
- results-v2 §1 rule 2: `selected` means the dispatched fleet once this contract lands; supplier-v1 §10: the hedge-error caption says it "includes the intraday dispatch's moves, priced at SIP − DA".
