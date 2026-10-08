# Replay contract v1 ("play the week")

Status: draft for lead acceptance, 29 September 2026 (task contract-replay), revised after the lead's review of `3c923b1` (blocking items B1 and B2, the lane split, the answers to the ten questions and optionals O1–O11, all applied and recorded in §8). Design only: no model or view code exists for anything below. Base `f185231`; market and events code read on `main` at `92a8df8`.

Authority: decision 0004 item 66 (Mike, 29 September 2026: a "play the week" screen with a draggable or played "now" cursor; at each instant what was known then against what happened; the fleet's actions and running money; a customer view for one EV; Plotly animation frames; small world and EV samples), read with items 48–65 as they shape prices, plans, positions and horizons; decision 0005 (the only net money figure is "illustrative simulated trading P&L"); `docs/contracts/trading-events-v1.md` §0, §4, §5.1–5.2, §10.1e; `docs/contracts/intraday-dispatch-v1.md` §3, §4.1, §5.2; `docs/contracts/results-v2.md` §1 (every rule applies to every frame here), §4.7, §5; `docs/DASHBOARD_DESIGN.md` (page contract, chart block, 390 px rules, copy rules).

Every price, volume and £ figure below is synthetic or illustrative. The replay adds no new random draw, no new physics and no new settlement: it re-indexes what the run already produced by *what was known at each instant*, and the views play that back.

## 0. Terms

| Term | Meaning |
| --- | --- |
| Sampled worlds `W_r` | `result.sampled_world_ids` (results-v2 §2: at most 10, `representative_world_id` first). The replay is built for these worlds only, the same set `position_updates` holds (trading §5.2). "World w" below means one of them; `wi` is its row in the replay arrays |
| Decision instant `τ_k` | `horizon_start_utc + k` hours, `k = 0 … 168`. Whole UTC hours are whole London hours (the offset is a whole number of hours), so these are exactly the trader's and dispatcher's hourly instants (trading §4.5, dispatch §1) plus one closing row. `s(τ_k) = 2k` is the study slot starting at `τ_k`; the end row `k = 168` has `s = 336` (nothing ahead, everything realised) |
| `K` | 169, the number of decision rows and of animation frames |
| Known at `τ` | Readable at `τ` under the run's own cut-offs: day-ahead prices published by `τ` (B4, `action.visible_day_ahead_prices`), intraday updates made by `τ` (the ceil rule, dispatch §3 `latest_known_prices`), positions from decisions at or before `τ`, events with notice by `τ`, shocks by their reveal rule (§1.5) |
| Realised | What the run's evaluation produced for a slot: the intraday close and imbalance price, both paths' home import, the settled deviation, the EV's SoC and charging. Shown only for slots that have ended by `τ` (`t < s(τ)`) |
| Forward curve at `τ` | For slots `t ≥ s(τ)`: the day-ahead price as visible at `τ` (published price, or the expected shape for unpublished slots) and, for published slots, the latest intraday value known at `τ` |
| Running £ | Cumulative money over slots that have ended by `τ`: the drivers' energy saving (customer leg, day-ahead-valued) and the aggregator's trading cash (strategy `full`), each defined in §1.4 |
| Frame | One Plotly animation frame, one per `τ_k`; the slider and play button step through them client-side |

Signs and units follow trading §0: kWh per half-hour, £/MWh, `kWh ÷ 1000 × £/MWh` the only money formula; money from the aggregator's view for trading and from the driver's view for saving (positive = paid less).

## 1. What the result carries: `ForecastResult.replay_week`

A frozen dataclass `ReplayWeek` in a new module `model/replay.py`; `None` on a no-action result. It holds NumPy arrays rather than long frames because every field is a `(world, decision, slot)` cube that the view slices by `[wi, k]` to fill frames; a long frame would be twice the bytes and would be pivoted back to arrays at view time. Arrays are `float64`, shapes are checked by the validator, and nothing in it is read by the UI except through the fields below.

```python
@dataclass(frozen=True)
class ReplayWeek:
    world_ids: tuple[int, ...]                      # == result.sampled_world_ids, same order
    decisions: pd.DataFrame                         # §1.1, exact, K rows
    published: np.ndarray                           # (K, 336) bool: pub_t <= tau_k
    day_ahead_visible_gbp_per_mwh: np.ndarray       # (W_r, K, 336)
    intraday_known_gbp_per_mwh: np.ndarray          # (W_r, K, 336), NaN where not published
    position_kwh: np.ndarray | None                 # (W_r, K, 336), NaN before the night's DA decision; None without trading
    energy_saving_to_date_gbp: np.ndarray           # (W_r, 337): cumulative at slot boundaries 0..336
    trading_cash_to_date_gbp: np.ndarray | None     # (W_r, 337); None without trading
    shocks: pd.DataFrame                            # §1.5, exact
    events: pd.DataFrame                            # §1.5, exact
```

Built once by `replay.build_replay_week(...)` in `forecast.run_forecast` (action model, after the trading overlay and `build_summaries`), from the run's `SimulatedForecast.market_prices` (day-ahead frame and `intraday_path_gbp_per_mwh`), the trading run (`position_updates`, `slot_cash_gbp`, `event_delivery`, `trading_ledger_world`), `events.event_slots`, `fleet_world_intervals`, `forecast_prices`, `market_shocks`, `events` and the assumption values it names. It draws nothing and changes no other field. The lead adds the one call line and the result field at integration.

### 1.1 `decisions` (exact)

One row per decision instant, `K = 169` rows in order.

| Column | Dtype | Meaning |
| --- | --- | --- |
| `decision_index` | int64 | `k`, 0–168 |
| `decision_utc` | datetime64[ns, UTC] | `τ_k` |
| `decision_london` | datetime64[ns, Europe/London] | Same instant |
| `slot_index` | int64 | `s(τ_k) = 2k`; 336 on the end row |
| `night_index` | int64 | `study_slots.night_index` of slot `2k`; 6 on the end row |
| `label` | object | Slider label, London time: `"Tue 18:00"`; the end row `"End of week"` |

### 1.2 Forward curves known at `τ` (the leakage-critical part)

For world `w`, decision `k` and study slot `t`, with `pub_t = sampling.day_ahead_publication_utc_ns(interval_start_utc)[t]`, `gate_t = start_t − gate_closure_minutes`, `H = 36`, and the run's day-ahead array `day_ahead_run` (world, run slot) and intraday path over the run slots (warm-up plus study, `warmup_count` warm-up slots first, starting at `run_starts`):

```
published[k, t]                       = pub_t <= tau_k
day_ahead_visible[wi, k, t]           = visible_day_ahead_prices(day_ahead_run, run_starts, tau_k)[w, warmup_count + t]
h_t(tau_k)                            = clip(ceil((gate_t - tau_k) / 1 h), 0, H - 1)
intraday_known[wi, k, t]              = intraday_path[w, warmup_count + t, h_t(tau_k)]   if published[k, t]
                                      = NaN                                              otherwise
```

- Both rules are the run's own, called, not restated: `action.visible_day_ahead_prices` (B4) gives a published slot its price and an unpublished slot the expected shape (the mean of the same London half-hour over the 7 most recent published delivery days); `action.latest_known_prices` (dispatch §3) gives the ceil-rule intraday value, which is the close for a slot past gate closure. `latest_known_prices` is not on `main` yet (dispatch lane K1): until it lands, R1b writes the same three-line rule as a private `replay._known_intraday(...)` with a comment naming the swap, and the K1 or integration step replaces it with one call; the leakage test of §5 runs against either. The replay stores the intraday value only where the day-ahead price is published, because the intraday path is a martingale from the published day-ahead price and has no meaning before publication (dispatch §3 falls back to the B4 shape there; the replay keeps that fallback visible as the day-ahead curve, not as a second line).
- For `t < s(τ_k)` the intraday value is the close (`h = 0`), so the same array serves the realised line; the validator checks it equals `evaluation_prices.evaluation_context_price_gbp_per_mwh` there. On the end row every slot is published and at its close.
- Why one array over all 336 slots rather than only `t ≥ s(τ)`: the view masks by `k` on the client with no arithmetic, the validator can check the realised part against the realised frame, and the storage is the same.
- No leakage: `published`, `day_ahead_visible[:, k]` and `intraday_known[:, k]` depend only on prices published by `τ_k` and updates made by `τ_k`. Perturbing any day-ahead price with `pub_t > τ_k` or any `intraday_path[w, t, h]` with `h < h_t(τ_k)` leaves row `k` unchanged (tested, §5).

### 1.3 Positions in force at `τ`

`position_kwh[wi, k, t]` is the `position_kwh` of the last `position_updates` row for `(w, t)` with `decision_utc ≤ τ_k` (strategy `full`; a forward fill over decisions in time order), NaN when no row exists yet, which is every slot of a night whose day-ahead decision `τ_DA(n) = 13:00` London on `D_n − 1` is after `τ_k` (night 0's decision falls in the warm-up, so its positions are in force from `k = 0`). For a slot past gate closure this is its final position (`deviation_world_slot.position_full_kwh`), for an open slot the current target. Built in `replay.py` by a groupby forward fill; the rule "at or before `τ`" is the trader's (a decision at `τ` uses information at `τ` and is in force from `τ`).

The trades made at `τ_k` are not stored again: the view reads the `position_updates` rows with `decision_utc == τ_k` (an exact match, no cut-off logic), showing `trade_kwh`, `price_gbp_per_mwh` and `stage`. `position_kwh` is `None` only when `position_updates` is `None`.

### 1.4 Running money at slot boundaries

Both arrays are indexed by slot boundary `b = 0 … 336`: the value at `b` sums the slots `t < b`, so the frame at `τ_k` reads index `2k`. They are money over slots that have *ended* by `τ`.

**Drivers' energy saving (customer leg), `energy_saving_to_date_gbp[wi, b]`**, positive when the smart path has cost the drivers less so far:

```
saving_t = (U_t - M_t) x P_DA[w, t] / 1000  -  (Pub_sel,t - Pub_norm,t) x public_charge_gbp_per_kwh
```

with `U`, `M` the fleet home import per path and `Pub` the fleet public import per path (`fleet_world_intervals` `home_import_kwh` and `public_import_kwh`, both paths), `P_DA` that world's day-ahead price (`forecast_prices.wholesale_forecast_gbp_per_mwh`) and the rate the run's `public_charge_gbp_per_kwh` record. The customer leg is valued at the day-ahead price, as `cost_effect` values it (results-v2 §4.8, decision 0004 item 53, trading review B3), never at the realised price; the caption says so. The two end-of-week items of the illustrative total (unrecovered energy value, unserved travel value; items 4 and 13) are not flows and are not accumulated: the end frame shows them from `cost_effect` beside the running figure.

`action.py` gains `slot_cost_effect_gbp(normal_home_kwh, selected_home_kwh, normal_public_kwh, selected_public_kwh, day_ahead_gbp_per_mwh, public_charge_gbp_per_kwh) -> (world, slot)` (selected minus normal, the sign `cost_effect` uses), and `calculate_wholesale_world_cost_effect` sums it for its two flow components, so the weekly figure and the running figure are one formula (a refactor with a regression test against the current per-world output). The replay stores `−cumsum` of it for the sampled worlds. At `b = 336` the value equals `−(illustrative_selected_minus_normal_energy_cost_gbp + illustrative_selected_minus_normal_public_charge_cost_gbp)` of `cost_effect` (validator).

**Aggregator's trading cash, `trading_cash_to_date_gbp[wi, b]`**, strategy `full`, booked in the delivery half-hour it settles (lead ruling):

```
c_t = q_t P_DA,t / 1000  +  sum_tau Delta_t(tau) P_ID,t,h / 1000  +  (V_t - f_t) SIP_t / 1000
      - s x sum_tau |Delta_t(tau)| / 1000  -  c_supplier V_t / 1000
b_e = min(end_slot_e, last slot of the night the window starts in + 1)          (a slot boundary)
cash_to_date[b] = sum_{t < b} c_t
                + sum_{events e with b_e <= b} payment_gbp[w, e]
                + sum_{nights n whose last slot < b} (customer_revenue_share_gbp + unmet_charge_penalty_gbp)[w, full, n]
```

`c_t` is the sum of the six slot-resolved buckets of trading §4.7 (day-ahead revenue and the baseline effect recombine to `q P_DA`; intraday, imbalance, trading cost and supplier compensation are per slot). `market.py` gains `slot_cash_components(day_ahead_position, final_position, intraday_cash, intraday_volume, inputs, assumptions) -> dict[str, np.ndarray]` beside `settle`, returning those six `(world, slot)` arrays by bucket name; `settle` night-sums that dict for those buckets instead of forming them inline (its return is unchanged; the grid-event payment, customer share and penalty stay as they are), and `run_trading` stores the `full` strategy's summed components as `TradingRun.slot_cash_gbp` (world, study slot). The replay builder cumsums it and adds the two night buckets from `trading_ledger_world`. No settlement formula is written a second time.

**Grid-event payments are booked when the window has ended, not when it starts (review B1).** `run_trading` books each request's payment on the window's first slot, and the payment is computed from metered import across the whole window (`events.event_delivery`), so booking it there would put future half-hours' outcomes into "so far" while the window is running. The replay therefore books `payment_gbp[w, e]` (from `TradingRun.event_delivery`, one row per world and event) at the boundary after the window's last slot, `end_slot_e` of `events.event_slots` (exclusive, so it is already the boundary), capped at the end of the night the window starts in (`event_slots.night_index`), which is the night the ledger books it in, so the night-end reconciliation below holds. Request windows normally lie inside a session night; a window that crosses 12:00 is booked at the cap, before its last slot has ended, a stated limitation of that case (the alternative, booking it after the crossing, would break the night-end identity with the ledger).

Why by delivery slot and not by transaction time: the ledger is by delivery night, so at every night end the running figure equals the cumulative `net_gbp` (validator), and the week's last value is `trading_week_world.net_gbp`; booking at transaction time would post night 0's day-ahead revenue before the study starts and mix decisions into a line meant for outcomes. Rejected alternative: recomputing `c_t` from `deviation_world_slot` and `position_updates` in `replay.py` (what the validator does): a second implementation of settlement in model code.

### 1.5 Shocks and events known by `τ`

`shocks` (exact): the `market_shocks` rows (trading §5.6a) of the sampled worlds that overlap the study, in that frame's order and with its columns, plus one column `known_from_utc` (datetime64[ns, UTC]): for `known_day_ahead` True, `day_ahead_publication_utc_ns(start_utc)` (it entered the day-ahead price of its first slot); for a stochastic surprise, `start_utc − surprise_reveal_hours` (the run's record: the instant intraday trading learns of it, the reveal rule of the generator); for a scripted surprise (`source == "scripted"`), that event's `notice_utc`, joined on `shock_id == event_id` (lead ruling: the events table is the authority on when a scripted event becomes known). A view shows a shock from the frame with `τ_k ≥ known_from_utc`.

`events` (exact): one row per enabled scripted event whose type is **not** in `events.PRICE_SHOCK_TYPES` (price shocks are already in `shocks`; requests and control outages belong here), from `events.event_slots` (trading §1.1): `event_id`, `event_type`, `scope`, `size`, `payment_gbp_per_mwh`, `start_utc` (`study_slots.interval_start_utc[start_slot]`), `end_utc` (`study_slots.interval_end_utc[end_slot − 1]`, `end_slot` being exclusive), `known_from_utc` (`notice_utc` converted from int64 UTC nanoseconds to datetime64[ns, UTC]), `evidence_kind`. Empty frames when there are none.

### 1.6 Realised series the views read from existing frames

Nothing realised is stored twice. For the chosen world the views read: `deviation_world_slot` (`unmanaged_kw`, `metered_kw`, `settled_kwh`, `day_ahead_gbp_per_mwh`, `intraday_close_gbp_per_mwh`, `imbalance_gbp_per_mwh`, `shock_kind`), `position_updates` (trades at `τ`), `cost_effect` (end-frame totals), `trading_week_world` and `trading_ledger_world` (`full`, end-frame totals and the night buckets), `study_slots` (labels). With no trading frames the fleet lens reads `fleet_world_intervals` (`home_import_kw`, both paths) and `forecast_prices`/`evaluation_prices` instead. The money side of the fleet lens (`deviation_world_slot`, `cost_effect`, the trading frames) stays normal versus selected only; it is not redefined for the timed path (model step 2 leaves it out, item 6). The fleet lens's plain import line is the one exception: with or without trading frames it draws the optional `timed` path (decision 0007) from `fleet_world_intervals` for the replayed world whenever the run has it, beside the unmanaged and smart lines, because that frame's `normal` row is the same quantity as the ledger's `unmanaged_kw`.

### 1.7 Sizes and runtime (estimate; the R1 writer measures and reports)

At `W_r = 10`: each `(10, 169, 336)` float64 cube is 4.5 MB; three cubes 13.6 MB; `published` 57 KB; the two money arrays 54 KB; `decisions`, `shocks`, `events` under 100 KB. About 14 MB on the result, 1.4 MB per sampled world, held only by the latest kept run (`RunRecord` keeps the full result for the latest run only). Build time: 169 calls of the two price rules over 10 worlds × 336 slots (gathers), one groupby forward fill over at most `10 × 336 × 37` update rows, two cumulative sums: well under one second at 1,000 EVs × 100 worlds. No effect on the per-EV chunk pass.

## 2. Customer view data: `replay_one_ev_timeline`

```python
def replay_one_ev_timeline(result, unit_id: str, world_id: int) -> OneEvTimeline
```

In `model/individual.py`, beside `replay_one_ev`. `world_id` must be a sampled world (the customer lens also shows the forward curve, which exists only for sampled worlds; one world control serves both lenses); another world raises `KeyError`. Cached on `replay_state.replay_cache` under `("timeline", unit_id, world)`; it shares the one-EV kernel run with `replay_one_ev` (one run per `(EV, world)` fills both objects, so the cache key of the run is shared and the timeline costs the book and cumulative sums on top).

```python
@dataclass(frozen=True)
class OneEvTimeline:
    unit_id: str
    world_id: int
    dispatch_locked: bool | None          # units.dispatch_locked; None when the column is absent
    decisions: pd.DataFrame               # the same frame as replay_week.decisions
    plan_at_decision_kwh: np.ndarray      # (K, 336): the plan in force at tau_k, grid kWh per slot
    plan_status: np.ndarray               # (336,) int64: code of the plan in force in each slot, selected path
    cumulative: pd.DataFrame              # exact, one row per (path_id, slot_boundary)
    saving_to_date_gbp: np.ndarray        # (337,): unmanaged cost so far minus smart cost so far
```

**Plan in force at `τ_k`.** `plan_at_decision_kwh[k, 2k + j] = book[0, k, j]` for `j < market.BOOK_WIDTH` (50 on `main`; referenced by name, never as a literal) and `2k + j < 336`; 0 elsewhere and on the end row. `book` is the selected path's plan book (trading §1.4, `physics._record_trading`) recorded on the one-EV kernel run: `simulate_fleet_intervals(..., trading_output={"selected": {"book_kwh": zeros(1, 168, market.BOOK_WIDTH), "book_decision": ...}})` with `book_decision` from a new helper `market.book_decision_slots(settings, study_slots) -> (book_slots, book_decision)` (the whole-hour rule now inline in `forecast._trading_kernel_outputs`; the lead's forecast hunk calls the helper, so the fleet book and the one-EV book cannot drift). Summed over the one EV in the slice, the book *is* that EV's plan in force at the slot start, read before the slot's early departures and new plans (B1): a plug-in exactly at `τ_k` shows its plan from frame `k + 1`; a session that leaves during `s(τ_k)` still shows its plan at `k` and none at `k + 1`. The selected book holds a non-responder's plan on record (trading §1.4: "including non-responders' full-power plans"; §10.1e: the smart plan stays on record while the session charges by the normal rule), so the snapshot shows that plan with its `plan_status` (1–4) and the view says "plan on record, not followed"; R1c confirms this against the kernel with a test. A free EV's re-plans (dispatch §4.1) appear at the decision they were made, because `KernelSlice` carries the sliced `IntradayDispatch` (dispatch §5.2). The plan window is at most the always-plugged cohort's 46 slots, so `market.BOOK_WIDTH` columns hold every plan.

**`plan_status`** is the kernel's per-EV output (trading §10.1e) for this EV on the selected path; 0 on a result without it.

**`cumulative` (exact)**: one row per `(path_id, slot_boundary)`, `path_id` in `individual._paths(result)` (`PATH_ORDER`, `("normal", "selected")`, plus the optional `timed` path, decision 0007, model step 2, when this result's fleet frames carry it; just `("normal",)` for a no-action result), `slot_boundary` 0–336: `path_id` (object), `slot_boundary` (int64), `home_import_to_date_kwh`, `public_import_to_date_kwh` (float64, cumulative sums of the replay's `intervals` columns over slots `< b`), `cost_to_date_gbp` (float64: home import at that world's day-ahead price plus public import at `public_charge_gbp_per_kwh`, the customer leg exactly as §1.4 values it, through `action.slot_cost_effect_gbp` on the one-EV arrays). The timed row's own running cost comes from the same one-EV kernel run, which also computes the timed path when the result has it (no extra kernel pass). `saving_to_date_gbp[b] = cost_to_date(normal)[b] − cost_to_date(selected)[b]`, always normal versus selected, whatever else `cumulative` carries. Summed over every EV of a sampled world, `saving_to_date_gbp[336]` equals `energy_saving_to_date_gbp[wi, 336]` (reconciliation test, the same identity the fleet frames already pass).

Everything else the customer lens shows (plugged-in, SoC, home and public import per slot, plug events, departures, early-departure and top-up markers) comes from `replay_one_ev`'s `intervals`, `plug_events` and `daily_audit` unchanged.

## 3. UI: the Replay page

### 3.1 Page, lenses and controls

A new top-bar page in `registry.PAGES` after Smart charging (and after any Trading or Supplier page that has landed) and before Compare:

```
Page("Replay", "replay", "What did the fleet know, do and earn at each moment of one simulated week?",
     (Lens("Fleet", "What did the fleet know, do and earn as the week unfolded?"),
      Lens("Customer", "What did one driver's charger know, plan and save as the week unfolded?")))
```

A separate page rather than a lens on Trading (lead ruling): it has its own header controls, serves both the trading and the driver story, and item 66 names it as a screen. `pages.VIEWS` maps both lenses through `_action_lens` (a no-action result gets the existing message and the switch button). An action result with `replay_week is None` (a result from before this contract, or a run kept without it) shows `st.info("Replay is not available for this run.")` and nothing else. The header controls slot holds `Simulated week` (options `replay_week.world_ids`, labels `"Week w (median)"` for the representative world, else `"Week w"`; one session key shared by both lenses) and, on Customer, the `EV` selectbox with the One EV lens's labels and default. No `@st.fragment`: `header.controls` reads a slot that `render_page` sets only for the plain render, so the view renders in the page like every other one; a control change reruns the page, which is cheap because the per-frame arrays are cached (§3.6). No control ever starts a run.

### 3.2 One figure per lens, everything animated is a trace

Each lens renders one `go.Figure` built with `make_subplots` (shared x, London time axis from `style.london_time_axis`, UTC in hover), styled by `style_figure`, mounted through `chart_block` with an explicit height. The figure has `K = 169` frames named `str(k)`; `frames[k].traces` lists the per-frame trace indices and `frames[k].data` carries only the arrays that change (`y` for full-length traces, `x` and `y` for sparse ones); Plotly merges a frame's trace attributes into the base trace, so `x` is sent once for full-length traces (the R2 writer confirms this merge in the browser on the first build; if it does not hold, `x` is sent per frame as epoch milliseconds and the payload figures below roughly double). Base data equals the initial frame (`sliders[0].active`), which is `k = 0` unless the Jump control (§3.6) chose another.

Every animated element is a trace, so the slider and play button use `frame.redraw = False` (no full relayout per step, which is what makes dragging smooth):

- the "now" line is a two-point scatter per panel (`x = [τ, τ]`, `y` = the panel's fixed range), `showlegend=False`, `hoverinfo="skip"`;
- the running counters are one `mode="text"` scatter in a slim top row with hidden axes (three labelled numbers), not `st.metric` tiles, because tiles cannot change without a rerun; the rerun-free requirement wins, and static week totals sit in ordinary KPI tiles above the chart;
- shocks and events known by `τ` are one marker trace at the top of the price panel (`▲` up, `▼` down, a square for a request), hover: class, direction, size, duration, "in the day-ahead price" or "surprise, seen h hours ahead", or the request's payment;
- realised traces are full-length with `None` for `t ≥ s(τ_k)` (the future is hidden, not dimmed: a dimmed line still reads its values in unified hover); forward traces have `None` for `t < s(τ_k)` and where the array is NaN.

Y-axis ranges are fixed from the static data (so hidden points never rescale a panel and the now line spans it). Layout: `sliders[0]` with 169 steps (`method="animate"`, `args=[[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]`, `label = decisions.label[k]`, `currentvalue.prefix = "Now: "`, tick marks hidden), with its `x` and `len` set to the x-axis domain (`xaxis.domain`, paper coordinates) so the grip sits directly under the now line at every width; one `updatemenus` button pair (Play: `[None, {"frame": {"duration": PLAY_MS, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}]`; Pause: `[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]`) in the top margin. `PLAY_MS = 150` is a named module constant in `ui/views/replay.py` (lead ruling: the week plays in about 25 s). The slider sits in the bottom margin above the legend; the view sets an explicit bottom margin (`style._bottom_margin`'s content rows plus the slider's height, measured in the browser) since a view's own bottom margin wins in `style_figure`.

Trace types used, each to be verified in the browser on the first build as animating correctly with `redraw: False`: line scatter (prices, imports, SoC, now lines), marker scatter (shocks and events, trades, the One EV markers), text-mode scatter (counters) and bar (charging, plan in force, plan on record). Nothing per frame changes a trace's type, pattern or marker style: a state that needs a different look (the plan on record, §3.4) is its own trace whose values are zero when it does not apply.

Legend names, used identically in §3.3 and §3.4: `Realised price`, `Day-ahead (published)`, `Latest intraday`, `Unmanaged`, `Smart`, `Sold turn-down`, `Plan in force`, `Plan on record, not followed`. The expected-shape trace has `showlegend=False` (its hover reads "expected shape, not yet published"); markers, now lines and counters stay out of the legend.

### 3.3 Fleet lens

Rows, top to bottom (heights from `CHART_HEIGHTS`; the R2 writer adds a `replay_counter` entry of about 40 px and uses `time_series` for the two panels):

1. **Counters** (text trace): `Now Tue 18:00 · Drivers' saving so far (illustrative) £1,234 · Trading P&L so far (illustrative simulated) −£56`.
2. **Prices (£/MWh)**: per-frame `Realised price` (the intraday close, solid, ink; hover also shows that slot's day-ahead and imbalance prices, lead ruling) to now; `Day-ahead (published)` (dashed, unmanaged grey) from now; the expected-shape trace (dotted, same grey, `showlegend=False`) from now for unpublished slots; `Latest intraday` (dashed, the `difference` amber: the trader's curve; teal stays reserved for the smart path) from now for published slots; the shock and event marker trace. Static: none.
3. **Fleet actions (kW)**: per-frame `Unmanaged` and `Smart` home import (the run's path styles) to now; `Sold turn-down` as a dashed amber step over every slot with a position (kW = kWh ÷ 0.5; the position as it stood at now: past slots final, open slots the current target); trade markers at `τ` on that line (`▲` sold more, `▼` bought back; hover `Δ` kWh, price, stage). With no trading frames the row shows the two import lines only and the counters drop the trading figure.

The legend holds six entries (`Realised price`, `Day-ahead (published)`, `Latest intraday`, `Unmanaged`, `Smart`, `Sold turn-down`; lead ruling, above the One EV lens's five because this is a two-panel figure) and wraps to at most three rows at 390 px. KPI tiles above the chart (static, `components/kpi.py`): the week's illustrative saving and trading net for this world from `cost_effect` and `trading_week_world`, labelled "this simulated week". Caption (≤ 140 characters, no citations): `Week 3, one simulated week. Dashed: prices known at the cursor; solid: what had happened. Illustrative.` The Data and definition expander shows `decisions` joined with the two running figures per instant and explains: hourly instants; the hidden future; the day-ahead valuation of the saving; cash booked by delivery half-hour; positions are the aggregator's `full` strategy; the ledger pays turn-down only.

### 3.4 Customer lens

Rows:

1. **Counters**: `Now Tue 18:00 · Cost so far (illustrative): unmanaged £4.12, smart £2.60 · Saved so far £1.52` (`style.money(decimals=2)`; the label "Cost so far (illustrative)" is the lead's wording for a day-ahead-valued tariff leg).
2. **Prices** (the same per-frame traces as the fleet lens, without the trade markers): what this EV's charger could see when it planned.
3. **Battery SoC (%)** with `Plugged in at home` shading (per-frame, hidden beyond now, because a future plug-in is not yet known), `SoC (smart)` and `SoC (unmanaged)` to now, and the One EV markers (`▲` plug-in, `▼` departure, `◆` public top-up, `✕` left before plan finished) as per-frame sparse traces holding only events at or before `τ`.
4. **Home charging (kWh per half-hour)**: per-frame `Smart` and `Unmanaged` bars to now (overlay, as One EV draws them); `Plan in force` outlined teal bars from now (`plan_at_decision_kwh[k]` where `plan_status[s(τ)] = 0`, sparse `x`/`y`); and a separate `Plan on record, not followed` bar trace (grey outline) that carries the same values instead when `plan_status[s(τ)] ≠ 0` and zero otherwise, its hover naming the status ("this session charges as unmanaged; maker offline" and so on). Two traces rather than one trace whose pattern changes per frame, so no frame changes a trace's style (§3.2).

Caption: `Week 3, one driver. Saving = unmanaged minus smart cost so far at the day-ahead price, plus public top-ups. Illustrative.` A second caption line from `dispatch_locked`: `Locked to its day-ahead plan` or `Re-plans hourly on the latest intraday price` (omitted when `None`). The traits caption of One EV is reused. The expander tables are the timeline's `cumulative` and the plan snapshot at the instant the Jump control (§3.6) selects: Streamlit tables cannot follow the client-side slider, and the expander says "at the jumped-to instant".

### 3.5 Payload estimate and bound

Per full-length per-frame trace: 336 values at one decimal place plus commas, about 2.2 KB (`None` costs about the same as a number). Fleet lens per frame: four price traces, three action traces (8.8 + 6.6 KB), trades and markers (about 1 KB), two now lines and the counters (under 1 KB): about 17 KB × 169 frames ≈ 2.9 MB plus the base figure and slider steps (about 40 KB). Customer lens: four price traces, shading, two SoC traces, two bar traces (about 20 KB), the two plan bar traces and markers (about 3 KB), three now lines and counters: about 24 KB × 169 ≈ 4.0 MB. Bound (lead ruling): `len(figure.to_json()) ≤ 4.5 MB` per lens with hourly frames on a 336-slot fixture, asserted by a UI test and reported with the measured value. If a real result exceeds it, the writer applies, in this order and recording which: (1) one decimal place everywhere (already assumed), (2) merge the day-ahead and expected-shape traces into one with the publication status in hover, (3) two-hourly frames (85). Plotly holds the frames client-side; stepping between them touches only the per-frame traces, so drag latency is the redraw of about ten scatter and bar traces of 336 points, well under a frame at 60 Hz on a laptop.

### 3.6 390 px, accessibility and caching

At 390 px the header controls stack under the title, the two KPI tiles stack, the figure keeps its full height and width with the legend below the slider, and the slider grip is touch-draggable; the R2 writer passes the browser check at 1440 and 390 px (decision 0004 item 26). Plotly's slider and buttons are not keyboard-operable (a known plotly.js limitation), so the view adds a `Jump to` selectbox over `decisions.label` inside a `Keyboard access` expander; choosing an instant reruns the page with that frame as the base data and `sliders[0].active` (a cheap figure reassembly, never a run), and the expander tables follow it (§3.4). This is the accessible path and is stated as a limitation of the animated control. Directly dragging the now line is not in v1 (§7). The per-frame arrays are built once per `(world, EV)` and kept on `result.replay_state.replay_cache` under `("replay_figure", world_id, unit_id)` (`unit_id` `None` for the fleet lens), the same per-result cache and eviction `individual.py` uses, so the Jump control and lens switches only reassemble the figure. Not a process-wide cache keyed on `id(result)`: an id is reused after garbage collection, so a new run could be served the previous run's frames (review B2).

### 3.7 Consumption map

| View | Reads | Computes |
| --- | --- | --- |
| Replay ▸ Fleet | `replay_week` (all fields), `deviation_world_slot`, `position_updates` (rows at `τ`), `cost_effect`, `trading_week_world`, `study_slots`, `sampled_world_ids`, `representative_world_id`; without trading: `fleet_world_intervals`, `forecast_prices`, `evaluation_prices` | Masks by `k` and unit conversions the contract states (kWh ÷ 0.5); no cut-off logic, no money, no quantiles |
| Replay ▸ Customer | `replay_week` (`decisions`, price cubes, `published`), `replay_one_ev` (intervals, plug events, daily audit), `replay_one_ev_timeline`, `units`, the public top-up assumption values (captions) | The same |

Every £ label reads "illustrative"; trading money is "illustrative simulated trading P&L"; "Unmanaged"/"Smart" for the paths; "Baseline" only for the settlement baseline, which this screen does not draw (§7).

## 4. Validator rules (`validate_result_v2` additions; one fixture module `tests/fixtures/replay_contract.py`)

`validate_replay_week_v2(result)` (called by `validate_result_v2` when `replay_week` is not `None`):

- `replay_week` is `None` exactly on a no-action result; `world_ids == sampled_world_ids`; `decisions` exact columns, dtypes and 169 rows; `decision_utc` equals `horizon_start_utc + k h`; `slot_index == 2k` (336 on the end row); `night_index` equals `study_slots` at that slot; labels London.
- Shapes `(W_r, K, 336)` for the three cubes, `(K, 336)` for `published`, `(W_r, 337)` for the money arrays; finite except where stated.
- `published[k, t] == (forecast_prices.forecast_available_at_utc[t] <= decision_utc[k])`; `intraday_known` NaN exactly where not published; `day_ahead_visible[wi, k, t] == forecast_prices` day-ahead price wherever published; for `t < 2k`, `intraday_known == evaluation_prices.evaluation_context_price_gbp_per_mwh` (which equals `deviation_world_slot.intraday_close_gbp_per_mwh`); on the end row both cubes equal the realised frames; for unpublished `t`, `day_ahead_visible` is finite and constant across every `k` with the same set of published slots (the expected shape changes only at a publication). The exact recomputation of the shape needs the warm-up prices, which the result does not carry, so it is a model test on the run (§5), not a validator rule.
- `position_kwh` equals a pandas forward fill of `position_updates` (`full`) by `decision_utc ≤ decision_utc[k]`, NaN before the first row; on the end row equals `deviation_world_slot.position_full_kwh`; `None` exactly when `position_updates` is `None`.
- `energy_saving_to_date_gbp[:, 0] == 0` (it may fall as well as rise), and at 336 equals `−(energy cost + public cost)` of `cost_effect` for the sampled worlds within `1e-9 × max(1, |value|)`; a recomputation from `fleet_world_intervals` and `forecast_prices` matches every boundary.
- `trading_cash_to_date_gbp[:, 0] == 0`; at each night's end boundary equals the cumulative `net_gbp` of `trading_ledger_world` (`full`) over nights `≤ n`; at 336 equals `trading_week_world.net_gbp`; `None` exactly when the ledger is `None`; the increments between two consecutive boundaries equal `slot_cash_gbp` of that slot plus the payments of the events whose booking boundary `b_e` is that slot's end (recomputed from `event_slots` and `event_delivery`), so no payment is counted before its window has ended (or the night cap).
- `shocks`: the sampled worlds' study shocks with `market_shocks`' columns plus `known_from_utc`; `known_from_utc ≤ start_utc`; for known shocks equal to the publication instant of `start_utc`; `events`: one row per enabled event with `known_from_utc == notice_utc`.

`validate_one_ev_timeline_v2(timeline, result)`: shapes; `decisions` identical to `replay_week.decisions`; `plan_at_decision_kwh ≥ 0`, zero outside `[2k, 2k + market.BOOK_WIDTH)` and on the end row; `plan_status` codes in 0–4; `cumulative` exact columns and `path_id` set equal to `individual._paths(result)` (so a `timed` block appears exactly when the result's fleet frames carry it, decision 0007), sums matching `replay_one_ev` intervals for each path present; `saving_to_date_gbp` equals the difference of the normal and selected paths' `cost_to_date_gbp`, whatever else `cumulative` carries; for a locked EV that never re-plans, `plan_at_decision_kwh[k, t]` equals its `planned_home_import_kwh[t]` for `t ≥ 2k` inside the plan window (checked where the per-EV output is available on the fixture).

The synthetic fixture `make_result` gains a `replay_week` built by the fixture's own generator from its price and trading frames, and a `replay_one_ev_timeline`, both passing the validators; the fixture's intraday cube is a deterministic toy path that converges to its close.

## 5. Tests (checklist; each lane adds the rows it implements)

Model (`tests/model/test_replay.py`, `tests/model/test_individual.py` additions):

- [ ] **No leakage, prices.** On a two-world, eight-slot market fixture: perturbing every `intraday_path[w, t, h]` with `h < h_t(τ_k)` and every day-ahead price with `pub_t > τ_k` leaves `published[k]`, `day_ahead_visible[:, k]` and `intraday_known[:, k]` bit-identical; a later frame does change.
- [ ] **No leakage, positions.** Perturbing `position_updates` rows with `decision_utc > τ_k` leaves `position_kwh[:, k]` unchanged; a row at exactly `τ_k` is included.
- [ ] **Known-at rules.** The validator rules of §4 on the fixture and on a real 30 EV × 4 week run (`validate_result_v2`); the realised part of `intraday_known` equals the close; the end row equals the realised frames; night 0's positions are in force at `k = 0`, night 1's from the frame at 13:00 on `D_0`.
- [ ] **Money reconciliation.** Running trading cash equals cumulative `net_gbp` at every night end and the week net at 336 (fixture with non-zero customer share, penalty and one grid request); `settle` built on `slot_cash_components` reproduces the current ledger bit for bit on the existing seeded trading fixture (regression); `slot_cost_effect_gbp` summed over the week reproduces `calculate_wholesale_world_cost_effect`'s two flow components on the existing seeded fixture (regression); `energy_saving_to_date_gbp[:, 336]` equals minus those components.
- [ ] **No leakage, grid-event payments (review B1).** On a fixture with a turn-down window inside one night: perturbing the metered import in the window's slots after `τ_k` (and the payment that follows from it) leaves `trading_cash_to_date_gbp[:, 2k]` unchanged for every `k` with `2k < end_slot`, and the payment appears exactly at boundary `end_slot`; with a window that crosses 12:00 the payment appears at the night cap and the test states that as the documented limitation.
- [ ] **Shocks and events.** Known shocks known at publication; surprises at `start − reveal`; a scripted event at its notice; an empty events table gives an empty frame.
- [ ] **Sizes and pairing.** Shapes as §1; `world_ids == sampled_world_ids`; two runs with one seed give identical replay arrays; build time and bytes at 1,000 EVs × 100 worlds measured and reported.
- [ ] **Customer plan snapshots.** On a one-EV run the recorded book equals `plan_at_decision_kwh`; `market.book_decision_slots` reproduces the `book_slots` and `book_decision` the fleet run used (the forecast hunk calls it); a locked EV that never re-plans matches its `planned_home_import_kwh` from `2k` on; a re-plan made at `τ_k` appears at `k` and not at `k − 1`; a plug-in exactly at `τ_k` is absent at `k` and present at `k + 1` (B1); perturbing the realised unplug inside `s(τ_k)` leaves the snapshot at `k` unchanged; a non-responder's snapshot is its smart plan on record (not its full-power import) with `plan_status ≠ 0`, confirming §2's reading of the selected book (O11); the end row is zero.
- [ ] **Customer money.** `cumulative` sums equal `cumsum` of the replay intervals; `saving_to_date_gbp[336]` equals this EV's energy plus public cost effect; summed over all EVs of a sampled world it equals the fleet running saving at 336 within `1e-9` kWh-scaled tolerance.
- [ ] **Cache.** `replay_one_ev` and `replay_one_ev_timeline` for one `(EV, world)` run the kernel once.

UI (`tests/ui/test_replay_view.py`, AppTest on the fixture):

- [ ] Renders both lenses on the fixture with trading frames, without `position_updates`, without the ledger, with `replay_week=None` (message only) and on a no-action result (message and switch button).
- [ ] 169 frames, each naming only existing per-frame trace indices, full-length frame traces carrying `y` only; base data equals the active frame; the now lines sit at `decision_utc[k]`.
- [ ] Hidden future: for sampled `k`, realised traces are `None` at indices `≥ 2k` and forward traces `None` below `2k` and where unpublished; the plan bars cover only `t ≥ 2k`.
- [ ] Payload: `len(figure.to_json())` within §3.5's bound on a 336-slot fixture, value printed in the test log.
- [ ] Controls: week options equal `world_ids` with the representative world labelled first; the EV selector matches One EV's labels; the Jump control sets `sliders[0].active` and the base data; nothing calls the runner.
- [ ] Copy: both lenses added to `test_copy_rules.CHECKED_VIEWS`; captions ≤ 140 characters; no citations or snake_case on screen; "illustrative" on every £; no "Axle cash"; "Unmanaged"/"Smart" only.
- [ ] Browser pass at 1440 and 390 px: play, pause, drag, jump; legend and slider do not overlap; nothing clipped.

## 6. Build lanes and dependencies (disjoint ownership; the lead adds one call line and the result field at integration)

| Lane | Outcome | Owns | Depends on | Risk and review |
| --- | --- | --- | --- | --- |
| **R1a** fixture and validators (Normal) | §4 complete on the synthetic fixture | `tests/fixtures/replay_contract.py`, the `make_result` hunk in `tests/fixtures/result_fixture.py`, the `validate_result_v2` hook hunk in `tests/fixtures/result_contract.py` | This contract accepted; the fixture's existing trading frames | One review; lands first so R1b, R1c and R2 build against it |
| **R1b** fleet replay data (High: decision-time cut-offs, money) | §1 complete | `model/replay.py` (new, including `_known_intraday` until K1 lands), `action.py` hunk (`slot_cost_effect_gbp`, the refactor), `market.py` hunk (`slot_cash_components`, `TradingRun.slot_cash_gbp`), `tests/model/test_replay.py` | R1a only: everything it reads is on `main` now (`position_updates`, `TradingRun` with `event_delivery`, `visible_day_ahead_prices`, `day_ahead_publication_utc_ns`, `intraday_path_gbp_per_mwh`, `market_shocks`, `event_slots`) | One strong independent review; a specialist second review only for the leakage tests if the reviewer asks |
| **R1c** customer timeline (High: planner cut-offs) | §2 complete | `individual.py` hunk (`replay_one_ev_timeline`, the book and `plan_status` on the one-EV run), `market.py` hunk (`book_decision_slots`), `test_individual.py` additions | R1a; J1b (`plan_status`, per-EV plan outputs); dispatch K1–K4 for the re-plans to appear (`KernelSlice` carrying `IntradayDispatch`, `units.dispatch_locked`): R1c may land before K on the day-ahead plan path and its tests then cover locked behaviour only | One strong independent review |
| **R2** UI (Normal) | §3 complete | `ui/views/replay.py` (new), `registry.py` and `pages.py` hunks, `style.py` height entry, `tests/ui/test_replay_view.py`, `test_copy_rules.py` list | R1a (builds on the fixture in parallel with R1b and R1c); the Trading and Supplier page lanes only for the page order | One review plus the browser pass |
| Lead | The `replay_week` field on `ForecastResult`, the call line in `run_forecast`, the `forecast._trading_kernel_outputs` hunk that calls `book_decision_slots`, integration, `docs/VERIFY_APP.md` journey step "Replay: play, drag, jump, both lenses" | `forecast.py` hunk, `VERIFY_APP.md` | R1b, R1c, R2 | Integrated validation |

Order: R1a; then R1b, R1c and R2 in parallel (R1c waits for J1b); then the lead integrates and runs the combined checks.

## 7. Left out of v1 (state as limitations where a viewer would look for them)

- **Dragging the now line itself.** The Plotly slider is the drag handle; the line follows it. Plotly can let a shape be dragged (`config.edits.shapePosition`) but nothing binds that drag to the frames without custom JavaScript (a `st.components.v2` component), which is not simple; recorded as a v2 candidate.
- **Keyboard operation of the Plotly slider** (§3.6): the Jump control is the accessible path.
- **Worlds outside the sample** and any across-weeks band: the replay is one simulated week by design (item 66); the other lenses carry the spread.
- **The settlement baseline and the day-ahead reference path** (`dispatch_world_slot`) as lines: Trading ▸ Position and Smart charging ▸ Response show them; the position line here is enough to read the trades. Adding either later is one static or per-frame trace and one legend entry.
- **A running supplier P&L, a per-EV trading revenue share, hover of hidden values, a speed control, export.** The customer's revenue share exists only per night at fleet level (trading §4.7) and is not allocated to EVs.
- **A grid request whose window crosses 12:00** has its payment booked at the end of the night it starts in (§1.4), so the running P&L shows that payment before the window's last slots end. Presets are evening windows, so this is rare; the Fleet lens's "Data and definition" text says so.
- **Trading with the intraday dispatch off** shows the same screen: the locked/free caption is then absent.

## 8. Resolved by the lead's review of `3c923b1`, and what remains open

Blocking items, both applied: B1 (grid-event payments booked at the boundary after the window's last slot, capped at the end of the night the window starts in; the leakage test; §1.4, §4, §5), B2 (the figure cache lives on `result.replay_state.replay_cache`, never a process-wide cache keyed on `id(result)`; §3.6).

Lane split (§6): R1b is the fleet data of §1, after R1a, on what `main` holds today, with a private `_known_intraday` until K1's `latest_known_prices` lands (§1.2); R1c is the customer timeline of §2, after J1b. The stale F2 line of §1.3 is removed.

Answers, carried into the body: (1) the realised line is the intraday close, with the day-ahead and imbalance prices in hover (§3.3); (2) cash by delivery half-hour, with B1 (§1.4); (3) `PLAY_MS = 150`, a named constant (§3.2); (4) a separate page (§3.1); (5) the customer leg at the day-ahead price, labelled "Cost so far (illustrative)" (§3.4); (6) the future is hidden (§3.2); (7) 4.5 MB per lens, hourly frames, the fallback order kept (§3.5); (8) six legend entries, one set of names for §3.2–§3.4, the unpublished trace out of the legend (§3.2, §3.3); (9) a scripted surprise is known from its event's `notice_utc`, joined on `shock_id == event_id`; stochastic shocks use the reveal rule (§1.5); (10) no `slim_run` change: Compare's slim records do not carry `replay_week`.

Optionals, all applied: O1 slider `x`/`len` on the x-axis domain (§3.2); O2 no fragment, because `header.controls` reads a slot only `render_page` sets (§3.1, §3.6); O3 a separate "Plan on record, not followed" trace and the list of trace types to verify in the browser (§3.2, §3.4); O4 expander tables follow the Jump control only (§3.4, §3.6); O5 `replay_week.events` excludes `events.PRICE_SHOCK_TYPES`, `end_utc` from `end_slot` and `study_slots`, `notice_utc` converted from int64 ns (§1.5); O6 `slot_cash_components` beside `settle`, which uses it, rather than a changed `settle` return (§1.4); O7 `book_decision_slots` in `market.py`, called by the lead's forecast hunk (§2, §6); O8 `market.BOOK_WIDTH` (50 on `main`) referenced by name (§2); O9 the counter reads "Trading P&L so far (illustrative simulated)" (§3.3); O10 the lead records in the polish plan that item 66 supersedes "a replay UI" under Leave out; O11 the selected book holds a non-responder's plan on record, stated in §2 and confirmed by an R1c test (§5).

No open questions remain for this contract.
