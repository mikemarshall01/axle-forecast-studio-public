# Dashboard design: Axle Forecast Studio

Status: design spec for build tasks, 28 September 2026, amended the same day by the streamlining pass below, and further amended by decision 0004 items 54–66 (29 September 2026, the analyst/trading/polish plan, `docs/plans/2026-09-29-analyst-trading-polish-plan.md`), which add pages and lenses on top of the five-page base. Authority order: the Axle brief, then decision 0004 (items 1–66), then Mike's feedback register (`docs/USER_FEEDBACK.md`), then `PRODUCT.md`/`DESIGN.md`. Where this spec changes `DESIGN.md` (seven workspaces, series colours, sidebar controls), this spec wins and `DESIGN.md` is updated by the shell task. The streamlining pass changes the page count accepted in 0004 item 22 from six to five; that change needs Mike's approval (section 7, question 5) and nothing else in item 22 changes. Section 2 carries the current page/lens map: Trading, Supplier (including Firm MW and Customers), Partners and Replay (including Customer) and the Household lens have since merged to `main`, so the whole nine-page map is built.

Numbers inside wireframes are placeholders to show layout and wording, not results.

Basis: only two models remain (0004 item 18): **Explicit wholesale action** (default; the app's Smart charging model since the rename below, decision 0004 item 38 replaced the wholesale-ceiling mechanism the old name described) and **Explicit no-action**. Cohort, compact, compact DFS and synthetic six-route modes, and every view that serves only them, are gone. Nothing below depends on `compact_*`, `cohorts`, `overview`, `routes`, `stakeholder`, `portfolio_*`, `regions`, `market_delivery` or `scenario_compare` views.

## 0. Streamlining pass

Goal: the same functions with less on screen. An interviewer with ten minutes should see one question, one hero chart and at most four numbers per screen, and Mike should be able to change an assumption and rerun without leaving the chart he is talking about. Every function in the accepted spec stays reachable; the table says where it went.

| Before | After | Reason |
| --- | --- | --- |
| Six pages: Overview, Drivers, Wholesale action, Scenarios, Parameters, Methods & notes | **Five pages: Overview · Drivers · Smart charging · Compare · How it works** | Parameters is an editing task, not a place to visit; it moves into a dialog. "How it works" and "Compare" are plain words an interviewer reads without a tour. Wholesale action is later renamed Smart charging (goal review section 8 item 6): the page's own content is the smart-charging plan, response and cost, not the wholesale-ceiling action the old name described. |
| Parameters page | **Edit assumptions** dialog (`st.dialog`, large) opened from the run bar on every page; its footer holds **Run simulation** | Change-an-assumption-and-see-the-effect happens on the page being discussed. No page switch mid-demo. |
| Methods ▸ Assumptions read-only table and the Parameters page listing the same records | One read-only table in How it works ▸ Assumptions with an **Edit assumptions** button that opens the same dialog | One listing, one editor, same `assumptions.py` source. |
| Reset draft button in the run bar | **Reset to active run** and **Reset to defaults** inside the dialog | Reset only matters while editing; it leaves the always-visible bar. |
| Model selectbox | Two-option `st.segmented_control`: **Smart charging · No action** | Both choices visible, one click, no dropdown. |
| Six status states as `st.info`/`st.warning`/`st.error` banners, plus "(stale)" appended to every chart caption | One status chip (`st.badge`) with a word, plus a muted identity caption. Only **Run failed** adds a banner, because its reason must be read. No per-chart "(stale)" | A banner wall pushes the hero chart below the fold on every page. The chip is on every page, worded, so stale stays unmistakable. |
| Scenarios page with "Save active result as…", naming, and a manual cap of three | **Compare** page. Every completed run is kept automatically (last three, labelled by what changed, e.g. "Run 3 · home charger 11 kW"); Compare defaults to previous vs latest | Removes the save-and-name step; the compare flow becomes edit → Run → Compare. Still session-only, matched futures by default (0004 item 6). |
| Drivers ▸ Fleet week with a Normal/Selected/Both path selector | Fleet week shows the normal path only; normal vs selected lives only in Smart charging ▸ Response | Removes a duplicate chart and a control. Drivers is behaviour; Smart charging is the change. |
| Fleet week metrics: Plugged in, Home import, Fleet SoC, Driving, Away (five) | Four: **Plugged in · Home import · Battery SoC · Travel** (Travel draws driving and away as two named lines) | Fits one segmented row at desktop; no data lost. |
| Drivers lens order Fleet week, Plug-ins, Archetypes, One EV | **Plug-ins · One EV · Archetypes · Fleet week** | Brief first (0004 item 16): the two literal asks, then sketch 1, then population checks. |
| Archetypes: 3×2 small multiples, "Compact layout" toggle, archetype selector and a third band chart on screen | 2×3 small multiples (reads at both widths, no toggle); the comparison table stays visible; the per-archetype week band moves into an expander | Cuts a viewport workaround control and a third chart. |
| One EV: world selector and "Show P10–P90 across weeks" on screen; plug-event table and traits below | Default median week; "Choose another simulated week" expander holds the world picker and band checkbox; traits as one caption line; plug events in an expander titled with the count | One control above the chart (the EV). |
| Every chart followed by a "Table" expander, a caption and sometimes a definition paragraph | One pattern per chart: title with unit, chart, one caption line (spread + evidence), one **Data and definition** expander (table + two-line definition) | Consistent and short; detail on demand. |
| Overview no-action tiles reading "Not in this model" | No-action Overview shows four real tiles (plug-ins per EV per week, unserved travel) instead | No dead tiles. |
| Wholesale action on the no-action model: an info message | The same message plus one button, **Switch to smart charging**, which sets the model control (Run is still explicit) | Gives the page a way forward. |
| Page title, purpose caption, lens control, sometimes a second heading | One header row: title left, lens control right; one muted line under it naming the question the current lens answers | Saves about 120 px above every hero chart. |
| Sidebar hidden but present | `initial_sidebar_state="collapsed"` and no sidebar content | Nothing can hide behind it at 390 px. |

Added from 0004 item 26: every chart has an explicit height (table in 3.5, repeated in each wireframe), full container width, legends below the plot at every width, and no overlapping or clipped titles, legends or annotations.

Kept as is: the two models, all charts that answer Q1–Q8, the plug-event table, archetype table, cost components with ⚠ shortfall flags, the Not modelled list, both explainers (0004 item 24), evidence badges, London time axis, the tokens and number formats.

This pass fixed five pages (Overview · Drivers · Smart charging · Compare · How it works, item 22 as amended by item 28). It is a point-in-time record of that decision, not the current page count: decision 0004 items 54–66 (29 September 2026) added three further pages (Trading, Supplier, Replay) and new lenses within Drivers and Smart charging, all now built and merged. Section 2 gives the current, built map.

## 1. Audience and the questions it answers

The audience is an Axle interviewer sitting beside Mike for a pair-programming session. They know EV flexibility well, have read their own brief, and have minutes, not an hour. Each answer must be one chart away from the landing page, labelled with its evidence class, and backed by a table.

| # | Question | Brief text it answers | Chart that answers it |
| --- | --- | --- | --- |
| Q1 | When are EVs plugged in? | "when somebody is plugged in" | Overview **Average day** bars; Drivers ▸ Plug-ins **Plug-in time** histogram |
| Q2 | What SoC do they plug in at? | "the state of charge of the battery when they plug in" | Drivers ▸ Plug-ins **SoC at plug-in** histogram, CNZ median 52% marker |
| Q3 | Is it agent-based? What does one driver look like? | "agent-based … individual user behaviour"; sketch 1 (plugged-in shading + SoC line, 7pm–7am) | Drivers ▸ One EV **plugged-in spans + SoC** chart and plug-event table |
| Q4 | How much do drivers vary? | sketch 2 ("% plugged in" bars, SoC mean and 95th percentile); "capture variation in the population" | Overview **Average day**: % plugged-in bars + SoC mean with across-EV P5–P95 band |
| Q5 | Do the archetypes reproduce population observations? | "recapitulate population-level observations … archetypes"; the six-pattern sheet | Drivers ▸ Archetypes: six small multiples + comparison table against CNZ context |
| Q6 | What does the smart charger do, and why? | "how much flexibility is available" | Smart charging ▸ Plan: forecast price with the smart charger's half-hours shaded, energy by forecast-price band |
| Q7 | What changed physically, and what did it cost or save? Did drivers lose out? | flexibility; end use | Smart charging ▸ Response: normal vs smart and the paired difference band; ▸ Value and risk: weekly saving with shortfall magnitude and early departures |
| Q8 | What is assumed, what is not modelled, and what if we change it? | "what assumptions did you make … which parts did you choose to spend time on … end use" | How it works; Edit assumptions dialog; Compare paired difference |

Two brief sketches are the design's anchors. Sketch 1 becomes the One EV chart; sketch 2 becomes the Overview Average day chart. An interviewer should recognise both on sight.

**Variation has two meanings, and the UI must name which one it shows.** The brief's sketch 2 band is spread **across drivers** (population variation). Our existing P10–P90 bands are spread **across simulated futures** (Monte Carlo uncertainty in a fleet total). The Average day chart shows across-EV spread; weekly fleet charts show across-world spread. Captions say "across EVs" or "across 100 simulated weeks" every time.

## 2. Information architecture

Top navigation, fixed order, text labels without icons. Landing page: **How it works**, opening on its **Why this model** lens (Fable landing pass and the 30 Sep "why-note" ruling: a first-time reader sees Mike's note on what the model is for and why it is built this way, then what is calculated, assumed and left out, before any result; Why this model and The forecast both point on to Overview ▸ At a glance's own "Start here" tour). The table below is the current map, not the five-page design of section 0: decision 0004 items 54–66 add Trading, Supplier and Replay, and new lenses within Drivers and Smart charging. **Built** is merged to `main`; **in progress** has an accepted contract and merged model frames but no UI lens yet; **pending** has an accepted contract and a UI build on a branch not yet merged.

| Page (nav label) | Status | The one question | Lenses (segmented control) | Replaces |
| --- | --- | --- | --- | --- |
| Overview | Built | What does a simulated week of these drivers look like? | At a glance · Key stats (Key stats added by item 54) | Command Centre |
| Drivers | Built | How do individual drivers and archetypes behave? | Plug-ins · Sessions (item 54) · One EV · **Household** (built — household-v1 §6.1, lane HH3; items 54, 58, 62, 65) · Archetypes · Fleet week (Flexibility's metrics merged in, item 54; gains a Network metric, **Zone import**, item 55) | Fleet & drivers (9 lenses) |
| Smart charging | Built | What did the smart charger do, what changed, what did it cost, and at what risk to drivers? | 1 Plan · 2 Response (gains an **Events** metric, items 55, 56, 58) · 3 Value and risk | Markets, Portfolio & revenue |
| Trading | Built | What did the illustrative simulated trading overlay buy, sell and settle? | 1 Market · 2 Position · 3 P&L and risk (items 43, 57, 58; `docs/contracts/trading-events-v1.md` §4, §5, §9) | none (new page; distinct from the "Markets as a separate page" cut below) |
| Supplier | Built | What can a supplier buy from this fleet, and what is it worth? | 1 Availability and cost curve · 2 Positions · 3 Supplier P&L · 4 Firm MW · **5 Charger makers** (renamed from "5 Partners" once the Partners page below existed, so there are not two Partners; items 58, 59, 62, 65; `docs/contracts/supplier-v1.md`) · **6 Customers** (built — household-v1 §6.2, lane HH4) | none (new page) |
| Partners | Built | Who is this for, and what would each partner want to see? | Driver · Energy supplier · Charger maker · Carmaker · Fleet and leasing (Mike, 30 Sep overnight review log; a pitch card per outside audience, reusing the Drivers/Smart charging/Trading/Supplier frames and chart builders, no new model logic) | none (new page) |
| Replay | Built | What did the fleet know, do and earn at each moment of one simulated week? | Fleet · Customer (item 66; `docs/contracts/replay-v1.md` §3) | none (new page) |
| Compare | Built | What changed between two runs? | none | Scenarios (5 lenses) |
| How it works | Built | How is it calculated, what is assumed, what is left out? | Why this model · The forecast · Assumptions · Limits · How it was built | Methods & evidence (5 lenses), Parameters (read-only part) |

Not a page: **Edit assumptions** is a dialog opened from the run bar on every page and from How it works ▸ Assumptions. It replaces the Parameters page's editing function.

Cut outright as a page: Portfolio & revenue, Command Centre, and the old six-route Markets page (Trading, above, is a different mechanism — a trading overlay on the two explicit models, item 43/57/58 — not a revival of Markets). The old Fleet lenses Customer outcomes, Distribution (folded into Plug-ins), Relationship, Time pattern, Compare and Stress/world stay cut.

Regions as a page stays cut, but the reason has changed since the streamlining pass: illustrative network zones are now an explicit-model mechanism (decision 0004 item 55; `zone_import_bands`, `zone_summary`), surfaced as the Network: Zone import metric on Drivers ▸ Fleet week and read into the Trading and Supplier frames, not given a page of their own. DFS turn-down and local turn-up requests are likewise now modelled, as scripted events read on Smart charging ▸ 2 Response ▸ Events (item 55; items 56 and 58 add the stochastic shock process and the trader's view of it), scoped to the fleet or to one zone. "Supplier optimisation", cut here in the original pass, is instead being built as the pending Supplier page (items 58 and 62). Frequency response, capacity market, V2G export (optional and last, item 57's H7), full network/DNO geography beyond the four illustrative zones, and settlement cash remain out of scope and appear once, in How it works ▸ Limits under **Not modelled**, with one line each on why. No page, tab or disabled control exists for them.

Navigation: `st.navigation(pages, position="top")` (Streamlit 1.64 is installed), `st.set_page_config(layout="wide", page_title="Axle Forecast Studio", initial_sidebar_state="collapsed")`, and `st.logo` with a small ink-coloured SVG wordmark so the product name sits at the left of the nav. Nav labels are one or two plain words; no icons (they add colour and noise to a nav bar that already reads). The nav never changes by model or state: Smart charging stays visible for the no-action model and explains itself. At 390 px Streamlit collapses the top navigation into its menu; the run bar is in the main area, so Run stays visible without opening anything. Lenses use `st.segmented_control` (any number of short labels: Mike, 30 September 2026, allowed more than four so Drivers and Supplier keep every lens; the control wraps onto extra rows at desktop and 390 px, so labels stay short and numbered where order matters), persisted in `st.session_state` per page. In-app navigation keeps the session; a hard reload or typed URL starts a new session and loses the results, and the empty state says so.

## 3. Global layout

### Layout grid

Every page uses the same vertical rhythm, top to bottom:

1. **Run bar** (one row, section 3.1).
2. **Header row**: `st.columns([1, 2])`, page title (`st.subheader`-sized, not `st.title`) left, lens control right-aligned. Under it one muted line: the question the current lens answers, followed by its evidence badge.
3. **KPI row**: at most four `st.metric` in `st.columns(4)`, no delta arrows, definitions in `help=`. Pages or lenses with no headline number skip the row.
4. **Hero chart**: full container width, explicit height from `CHART_HEIGHTS` in 3.5 (time series 340 px).
5. **Secondary content**: at most one more chart or a two-column pair (`st.columns(2)`, histograms 280 px each) or one short table that is itself the answer (archetype comparison, cost components).
6. **Detail**: collapsed expanders only (data tables, definitions, alternative selections).

No `st.divider` except the one under the run bar. No bordered containers around charts (the flat workbench rule). Links to other pages use `st.page_link`.

### 3.1 Run bar (every page, top of main area)

```
Desktop 1440
+----------------------------------------------------------------------------------------------+
| [wordmark] Axle Forecast Studio   Overview  Drivers  Smart charging  Compare  How it works   |  <- Streamlit top nav
+----------------------------------------------------------------------------------------------+
| Model [Smart charging|No action]   (● Current) Run 2 · 1,000 EVs · 100 weeks ·               |
|                                        seed 42 · from Mon 28 Sep     (Edit assumptions) (Run simulation) |
+----------------------------------------------------------------------------------------------+
```

`st.columns([2.2, 4, 1.4, 1.4], vertical_alignment="center")`: model segmented control; status chip plus identity caption; secondary **Edit assumptions**; primary **Run simulation** (`type="primary"`, teal, the only primary button on any page). At 390 px the four columns stack in that order, so Run is the fourth item below the header and never behind a menu.

| State | Chip (`st.badge`) | Identity caption | Other treatment |
| --- | --- | --- | --- |
| No result | none | "No result yet" | Page body shows the empty state (below) |
| Running | **Running** (grey, spinner icon) | "1,000 EVs × 100 simulated weeks…" | Run disabled; `st.status` placeholder in the bar; page body stays readable |
| Current | **Current** (green) | "Run 2 · 1,000 EVs · 100 weeks · seed 42 · from Mon 28 Sep · illustrative" | none |
| Stale | **Stale · 3 changes** (orange) | "Showing Run 2; draft differs" (the dialog lists the changes) | none; charts stay as they are |
| Failed | **Run failed** (red) | "Still showing Run 2" | one `st.error` under the bar with the reason |
| Invalid draft | **Draft invalid** (red) | "Fix 1 field in Edit assumptions" | Run disabled; the field shows its error in the dialog |

Colour is never the only cue: every chip carries its word. Running mechanics (T2): Run sets a session flag and reruns; the script renders the whole page first, then executes the run at the end of the script into the bar's `st.empty()` status placeholder, then `st.rerun()`. The page therefore stays fully readable while the run works, which the demo relies on.

Default run size stays the accepted 1,000 EVs × 100 simulated weeks. The empty state states the expected time ("about 40 s"); fleet size and worlds are in Edit assumptions ▸ Simulation for a quicker run.

### 3.2 Empty state (every page before the first run)

```
+--------------------------------------------------------------------------------+
| Run bar (no chip, "No result yet")                                             |
| <Page title>                                                                   |
| Nothing to show yet. Run simulation to forecast 1,000 simulated EV drivers     |
| over 100 possible weeks (about 40 s). A browser reload clears results.         |
|                                  (Run simulation)                              |
| What this app shows (Overview only): three sentences + the six archetypes      |
| with source shares 40 / 30 / 10 / 10 / 9 / 1%  [Source]                        |
+--------------------------------------------------------------------------------+
```

The body button is the same primary action as the bar (same callback), placed where the eye lands; no placeholder charts. How it works needs no result and renders fully before any run.

### 3.3 Edit assumptions dialog

```
Dialog, width="large" (full screen at 390 px)
+------------------------------------------------------------------------------+
| Edit assumptions                                                     [×]    |
| 3 changes from Run 2  (Reset to active run)  (Reset to defaults)            |
| [Fleet] [Driving] [Plugging & home charging] [Battery] [Public & prices]    |
| [Weather] [Simulation]                                  <- st.tabs          |
|  Home charger power       [ 11.0 ] kW   [Illustrative]  ⓘ                   |
|                           Draft 11.0 · Run 2 used 7.0                        |
|  Public charge rate       [ 0.79 ] £/kWh [Illustrative] ⓘ                   |
|  Cohort shares            40 / 30 / 10 / 10 / 9 / 1 %   [Source]            |
| ---------------------------------------------------------------------------- |
|                                       (Close)   (Run simulation)             |
+------------------------------------------------------------------------------+
```

One row per field: label, widget, unit, evidence badge; meaning, source value and "affects: …" in the widget `help=` tooltip (ⓘ), not in paragraphs under each field. The "Draft · Run used" caption appears only on changed fields. Fixed source values are read-only text with a Source badge. Synthetic price settings sit under Public & prices. Widget changes rerun only the dialog (dialogs are fragments), so editing never starts a run and never closes the dialog. **Run simulation** in the footer sets the run flag, closes the dialog and reruns (0004 items 11, 21; DESIGN.md "never start a full run from an edit"). Observed defect to fix here: the fleet-size slider displayed 200 while its caption and the run used 1,000; the widget value and the draft must be one state.

### 3.4 Chart block pattern

Every chart is the same block: bold title stating the quantity and unit; the Plotly chart; one caption line naming the spread ("P10–P90 across 100 simulated weeks" or "P5–P95 across EVs") and the evidence class; one collapsed `st.expander("Data and definition")` with the same frame through `components/chart_table.py` and a two-line definition. Nothing else sits between charts.

### 3.5 Tokens and visual polish

Keep the `.streamlit/config.toml` dark theme and `DESIGN.md` neutrals. The app already replaced the Plotly series colours in `ui/style.py` (chart audit O2): the previous normal (`#3E7298` dotted) and selected (`#0F766E`) lines had near-equal lightness, and a later contrast fix (final critique B-21) raised the band fill from 0.3 opacity, which measured only 1.78-1.99:1 on `#0E1117`, short of WCAG 1.4.11's 3:1 floor for a graphical object that carries meaning:

| Token | Value | Line | Meaning |
| --- | --- | --- | --- |
| `normal` | `#9AA9BF` | solid, 1.5 px | Unmanaged charging (no Axle action) |
| `selected` | `#2DD4BF` | solid, 2.5 px | Smart (selected) Axle action path |
| `difference` | `#F2B45A` | solid, 2 px | Smart minus unmanaged (paired, per simulated week) |
| `timed` | `#F2B45A` | dashed, 2 px | Optional third "Timed tariff" path (decision 0007), present when the "Timed tariff policy" switch is on (the default, start 00:00), absent when off; reuses the `difference` value, never teal |
| `observed` | `#E9EDF3` | dashed vertical/horizontal rule + text label | CNZ observed context value |
| `synthetic` | `#A78BFA` | marker / fill | A synthetic-event or perfect-foresight series, distinct from the three path colours |
| `band_alpha` | 0.55 | fill only | P10–P90 (worlds) or P5–P95 (EVs) |
| `flag` | `#F87171` + "⚠ not recovered" text | marker | Energy-not-recovered world or EV |
| `plugged` | `#9AA9BF` at 0.6 | bars / shaded spans | Plugged in at home |
| `grid` | `#2A3140` | 1 px | Gridlines only; no plot border |

These values are current as of this revision (`SERIES_COLOURS`, `PATH_STYLES`, `BAND_ALPHA`, `PLUGGED_ALPHA` and `BORDER` in `src/axle_studio/ui/style.py`). This table has drifted from the module before (a dotted normal line, 0.22 band alpha, 0.35 plugged alpha, a `#232A36` grid, none matching the module they claimed to describe) and can again, so treat `ui/style.py` as the source of truth and this table as a snapshot of it, the same caveat `CHART_HEIGHTS` gets below.

On screen the Smart charging ▸ Response chart's legend and lines read Unmanaged, Timed tariff, Smart in that order (`style.PATH_DISPLAY_ORDER`, decision 0007), not the frames' own normal/selected/timed order, so the optional path sits between the two it is compared against.

Teal stays the single UI accent (Run, current selection) and the selected-path colour; these are the same meaning ("what Axle chose"). Nothing else is teal: KPI values, headings and links to pages use ink or the mint link colour. Evidence badges use `st.badge`: **Source (CNZ/Axle sheet)** grey, **Illustrative assumption** orange, **Synthetic** violet, **Model output** none.

Typography: Streamlit's system stack throughout; three sizes only on a page (page title, chart title in bold body size, body/caption). Chart text 13 px, axis titles muted ink. Numbers in KPI tiles use `st.metric`'s default size; no custom large numerals.

Chart chrome: one shared Plotly template in `ui/style.py`: transparent paper and plot background, gridlines `grid`, no zero line except on difference charts, `hovermode="x unified"`, hover labels "Mon 28 18:30 (17:30 UTC) · 2,450 kW", modebar hidden (`config={"displayModeBar": False}`).

Chart size and fit (0004 item 26). Every chart sets an explicit `height` and is drawn with `st.plotly_chart(fig, use_container_width=True)` (full container width). Streamlit cannot read the viewport width, so one height per chart serves both 1440 and 390 px; the heights below are the floor at both widths. Legends are horizontal and **below** the plot (`legend=dict(orientation="h", y=-0.18, yanchor="top", x=0)`) on every chart at every width, so the plot keeps its full width at 390 px and the layout never changes between widths. Margins are set from content, not zeroed: left 56 px (y title and ticks), right 56 px on dual-axis charts else 16 px, top 8 px (titles are Streamlit text above the chart, not Plotly titles), bottom 72 px for the tick labels plus the legend row, more if a legend wraps. Axis titles use `automargin=True`; annotations (CNZ markers, decision-time marker, ⚠ labels) are anchored inside the plot area (`yref="paper"`, `y` ≤ 0.95, `xanchor` away from the nearer edge) so nothing overlaps a legend or clips at the edge. The T2 template sets these defaults; each view sets only the height.

The literal pixel values live only in `CHART_HEIGHTS` in `src/axle_studio/ui/style.py`, not duplicated here (they have drifted from an earlier copy of this table before): as of this revision `time_series` 340, `time_series_dual_axis` 360, `histogram` 280, `small_multiple_panel` 180, `strip` 240, `diagram` 260, `replay_counter` 72, `replay_panel` 240. The table below names which token each chart uses.

| Chart | `CHART_HEIGHTS` key |
| --- | --- |
| Overview Average day | `time_series_dual_axis` |
| Drivers ▸ Plug-ins: Plug-in time; SoC at plug-in | `histogram` each |
| Drivers ▸ One EV: plugged-in spans + SoC | `time_series_dual_axis` |
| Drivers ▸ One EV expander: P10–P90 across weeks | overlay on the same chart |
| Drivers ▸ Archetypes small multiples (3 rows × 2 cols) | `small_multiple_panel` per panel |
| Drivers ▸ Archetypes expander: one archetype week band | `time_series` |
| Drivers ▸ Fleet week | `time_series` |
| Smart charging ▸ Plan: forecast price | `time_series` |
| Smart charging ▸ Response: normal vs selected | `time_series` |
| Smart charging ▸ Response: paired difference | `time_series` |
| Smart charging ▸ Value and risk: strip plot | `strip` |
| Trading, Supplier: time-series charts | `time_series` |
| Replay: running counters, each animated panel | `replay_counter`, `replay_panel` |
| Compare: B − A paired difference | `time_series` |
| How it works ▸ The forecast diagram | `diagram` |

### Chart conventions

X axis is **Europe/London local time** (the study is defined on London dates, 0004 item 1); the current UTC axis goes. Weekly charts mark weekends with a faint background rectangle and day labels "Mon 28". Every chart has a y-axis title with unit. The UI never subtracts or averages percentiles; it plots columns the model supplies.

### Number formatting (extend `format_quantity`)

Fleet power kW, 0 dp, thousands separator (`2,450 kW`); fleet energy kWh 0 dp; one-EV energy kWh 1 dp; SoC and shares % 0 dp; counts integer; money `£` 2 dp with explicit sign (`−£12.40`, `+£3.05`) and the word *illustrative* in the label, never on its own; prices £/MWh 0 dp; public rate £/kWh 2 dp. Missing values show "Unavailable", never 0.

### 390 px behaviour (all pages)

Top nav collapses into Streamlit's menu. Run bar columns stack (model, chip, Edit assumptions, Run). Header row stacks (title, then lens control, wrapping). KPI rows use `st.columns(4)`, which Streamlit stacks below 640 px; that yields four short full-width rows, acceptable because each tile is one line of label and one value. Charts stack one per row (two-column pairs stack because Streamlit stacks columns), keep their full height from the size table and full width, and carry their legend below the plot; nothing is squashed or shrunk to fit. The small multiples keep two columns, which at 390 px gives panels about 170 px wide by 220 px tall, still readable because each panel has one bar series. The dialog is full screen.

## 4. Pages

### 4.1 Overview (landing)

Question: what does a simulated week of these drivers look like? Answers Q1 and Q4 directly; links to the rest.

```
Desktop 1440
+----------------------------------------------------------------------------------------------+
| Run bar                                                                                      |
| Overview                                                          Day [Weekday|Weekend|All]  |
| A simulated week of 1,000 drivers from Axle's six archetypes. [Model output · illustrative]  |
| Plugged in at 18:00   Median SoC at plug-in   Action chosen            Illustrative cost    |
| 62%                   48%                     Smart charging           −£41 median           |
|                       CNZ observed 52%        Ready 1 h before dep.     0/100 wks material    |
| **Average day: % plugged in at home and battery SoC**                          height 400 px |
|  %  |▇▇▇                                  ▇▇▇▇▇  bars: % plugged in                  SoC %   |
|     |▇▇▇▇▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇▇▇▇▇▇  line: mean SoC, band P5–P95 across EVs      |
|     +--00:00------06:00------12:00------18:00------24:00 (London)                            |
| Spread across drivers, not forecast uncertainty. Model output from illustrative assumptions. |
| > Data and definition                                                                        |
| Next: Plug-in times and SoC → · One driver's week → · Why this action →                     |
+----------------------------------------------------------------------------------------------+
```

The day-type toggle sits in the header row where other pages put their lens (default Weekday). KPI tiles (`st.metric`, no delta arrows): **Plugged in at 18:00** (weekday share, normal path, P50 across worlds; `help` names the denominator); **Median SoC at plug-in** (normal path; "CNZ observed 52%" as the caption under the value); **Action chosen** ("Smart charging", captioned with the departure margin, e.g. "Ready 1 h before departure" — decision 0004 item 38 applies smart charging to every EV, so there is no eligible-EV count to show, or "No action" with the no-action model); **Illustrative cost** (median total across weeks + count of weeks materially not recovered, a magnitude against a material threshold rather than a fleet-wide yes/no flag, decision 0004 item 45). With the no-action model the last two tiles become **Plug-ins per EV per week** (P50) and **Unserved travel** (P50 kWh).

Average day chart: combined bar + line with secondary y-axis. Bars = share of fleet plugged in at home by local half-hour (48 bars), averaged over study days of the chosen type, P50 across worlds, `plugged` colour. Line = mean SoC across EVs, band = P5–P95 across EVs (world-median of each), `normal` colour. "Next" is one row of three `st.page_link` items, each landing on the named lens (the lens key is set in session state before switching).

**Flexible power (decision 0004 items 43 and 46, amending this hero).** A second full-width line chart, below the average-day chart, from `flexibility_bands`, plots each half-hour's flexible power across the whole simulated week for both paths (not folded into a 48-slot average day, because the smart charger's plan runs across the whole week): the turn-up ceiling (home charging power summed over plugged-in, below-target EVs), P10–P90 across weeks. The normal-path line is labelled "Available (normal)"; the smart-path line, "Smart (headroom unused)", reads higher than its own draw and is captioned "headroom still unused while waiting for cheap slots" (`turn_up_headroom_kw` = flexible power minus that path's actual home import), not extra flexibility — the smart charger is holding the same capacity back for a cheaper half-hour, not creating more of it. This answers "how much charging can be moved, when" directly on the landing page.

Empty state: section 3.2 plus "What this app shows" (three sentences) and the six archetype names with source shares from `assumptions.py`, labelled Source.

390 px: title, toggle, subtitle, four stacked tiles, chart at full width and its 400 px height with the legend below, links as a vertical list.

### 4.2 Drivers

Question: how do individual drivers and archetypes behave? Physical behaviour at archetype and single-driver level from the same run (U12). Default lens: Plug-ins.

**Update (decision 0004 item 54 and the analyst/trading/polish plan, `docs/plans/2026-09-29-analyst-trading-polish-plan.md`).** The lens list and wireframes below are the original streamlining-pass design and are not redrawn here. As built, Drivers runs Plug-ins · Sessions · One EV · Archetypes · Fleet week: Sessions is new (per-archetype small multiples of plug-in time, departure time, SoC at plug-in, energy needed, dwell, flexible kWh and slack, each against a source marker), and Flexibility's metrics (deferrable power, turn-up headroom, movable energy, time slack) merged into Fleet week's metric control rather than staying a lens of their own. Fleet week's metric control also gained a Network metric, **Zone import**, reading `zone_import_bands`/`zone_summary` for one illustrative zone at a time against its optional headroom (item 55). The Household lens (household-v1 §6.1, lane HH3) is built and sits after One EV, before Archetypes, in `ui/registry.py`. See section 2 for the current map.

**Plug-ins** (Q1, Q2; decision 0004 item 16 brief-first)

```
Desktop 1440
+----------------------------------------------------------------------------------------------+
| Run bar                                                                                      |
| Drivers                       [Plug-ins | One EV | Archetypes | Fleet week | Flexibility]   |
| When do drivers plug in, and at what battery level? [Model output]      Day [Weekday|Weekend]|
| Plug-ins per week   Median SoC at plug-in    Plug-ins below 10% SoC                          |
| 6,812 (P50)         48%  · CNZ 52%           2.9%  · CNZ 3%                                   |
| **Plug-in time (London), share per hour**    | **SoC at plug-in, share per 5% bin**          |
|  height 320 px                               |  height 320 px                                |
|  bars, whiskers P10–P90 across weeks         |  bars, whiskers P10–P90 across weeks          |
|  ¦ CNZ weekday mode ~18:00                   |  ¦ CNZ observed median 52%                    |
| Normal charging path. CNZ values are context, not a calibration target (report p.13).        |
| > Data and definition                        | > Data and definition                          |
+----------------------------------------------------------------------------------------------+
```

The day-type control sits at the right end of the question line (`st.columns([3, 1])`). Bars are world-median shares; whiskers P10–P90 across worlds. Normal path only, stated in the caption; selected-path plug-in SoC is not overlaid.

**One EV** (Q3; decision 0004 items 10, 20)

```
+----------------------------------------------------------------------------------------------+
| Drivers                       [Plug-ins | One EV | Archetypes | Fleet week | Flexibility]   |
| What does one driver's week look like? [Model output]    EV [ev-0137 · Commuter (weekday) v] |
| Commuter (weekday) · 8,400 miles/yr · 60 kWh battery · 7 kW charger · target SoC 80%        |
| **Plugged in and battery SoC, Mon 28 Sep – Sun 4 Oct (median week)**           height 400 px |
|  ░░░░ shaded = plugged in at home   ── SoC normal   ── SoC selected                          |
|  ▲ plug-in (hover: time, SoC at plug-in)   ▼ departure                                       |
| One simulated week (the week with median fleet home import), not an average.                |
| > Plug-in events (7)        time · SoC at plug-in · kWh added before departure · shortfall   |
| > Choose another simulated week     world [0…99] · [ ] Show P10–P90 across weeks             |
| > Data and definition                                                                        |
+----------------------------------------------------------------------------------------------+
```

This is sketch 1. One world at a time (a real driver's week), default the world whose weekly home import is the median. The EV selectbox (searchable, archetype in the label) is the only control above the chart; the traits are one caption line. Decision 0004 items 10 and 20 landed the single-EV replay's import ceiling and public charging, so action results now show both paths (Normal and Smart) on this one EV, with its own smart-charging plan and any public top-up it needed; there is no longer an "unavailable for action results" placeholder.

**Archetypes** (Q5)

```
+----------------------------------------------------------------------------------------------+
| Drivers                       [Plug-ins | One EV | Archetypes | Fleet week | Flexibility]   |
| Do the six archetypes reproduce what CNZ observed? [Model output · Source shares]            |
| **Average day, % plugged in, by archetype** (2 cols × 3 rows, shared y, 220 px/panel, 732 px) |
|  [Scheduled 40%]  [Commuter 30%]                                                             |
|  [ ...  10%   ]   [ ...  10%  ]                                                              |
|  [ ...   9%   ]   [Always plugged 1%]                                                        |
| Archetype · source share · EVs in run · plug-ins/week (P50) · median plug-in SoC ·           |
|   peak plug-in hour · unserved travel                    <- visible table, 6 rows            |
| > One archetype across the week   archetype [select] · metric [select] → band, 380 px       |
+----------------------------------------------------------------------------------------------+
```

`make_subplots(rows=3, cols=2, shared_yaxes=True)` at both widths, so no layout toggle. The comparison table is visible because it is the answer to Q5. The week band reuses `explicit_cohort_bands` logic, retitled, inside the expander.

**Fleet week** (Q1, Q4 over time)

```
+----------------------------------------------------------------------------------------------+
| Drivers                       [Plug-ins | One EV | Archetypes | Fleet week | Flexibility]   |
| How does the whole fleet move through the week? [Model output]                               |
|                                  Metric [Plugged in | Home import | Battery SoC | Travel]    |
| **Fleet plugged in at home, Mon 28 Sep – Sun 4 Oct**                           height 380 px |
|  ~~~~ line P50, band P10–P90 across 100 simulated weeks ~~~~                                 |
| Normal charging (no Axle action). Unserved travel: 0 kWh in all weeks.                       |
| > Data and definition                                                                        |
+----------------------------------------------------------------------------------------------+
```

Plotly line + band for 336 half-hours. Battery SoC is capacity-weighted. Travel draws driving and away-on-trip shares as two named lines. The normal path only; the action's effect is Smart charging ▸ Response. Unserved travel is a caption fact, not a chart of zeros.

**Flexibility** (Q6, Q7 groundwork; decision 0004 items 43 and 46; the last Drivers lens, since it is the last step of "who → when → how much can move → what smart charging does" before Smart charging itself)

```
+----------------------------------------------------------------------------------------------+
| Drivers                       [Plug-ins | One EV | Archetypes | Fleet week | Flexibility]   |
| How much charging is available to move, and when? [Model output]                            |
|                    Metric [Flexible power | Turn-up headroom | Movable energy | Time slack]  |
| **Flexible power (kW)**                                                        height 380 px |
|  ~~~~ Available (normal), P10–P90 across weeks ~~~~   ~~~~ Smart (headroom unused) ~~~~~~~~~~ |
| Flexible power minus each path's own draw shows headroom, not extra flexibility.             |
| > Data and definition                                                                        |
+----------------------------------------------------------------------------------------------+
```

One metric control, one chart, reusing `flexibility_bands` (contract v2 section 3.6c) exactly as the Overview hero reads it, so the two never disagree. Flexible power and turn-up headroom read in kW, movable energy in kWh, time slack in hours; every metric shows the P10–P90 band across simulated weeks (item 40), always on. Time slack is shown at its across-weeks spread only; the model's separate across-EV spread belongs to a per-EV view, not this fleet-level lens. No-action model: normal path only, both paths on an action result.

390 px (all lenses): lens control wraps to two rows; Plug-ins histograms stack one per row at 320 px each; small multiples keep 2 columns at 220 px per panel; every legend sits below its plot; expanders full width.

### 4.3 Smart charging

Question: what did the smart charger do, what changed, what did it cost, and at what risk to drivers? Q6 and Q7, told as decision 0004 item 43's flexibility story. The lenses are renamed and numbered so the presenter walks them in order (item 43, superseding item 28's Decision/Response/Cost naming): **1 Plan · 2 Response · 3 Value and risk**. A one-sentence summary sits under the header on all three lenses, generated by one shared text helper so the three lenses never phrase it differently:

"Smart charging: each EV charges just enough to reach its target, in the cheapest forecast half-hours before its usual departure time minus 1 hour. An EV that leaves earlier misses the rest of its planned charge."

No-action model: the page shows "This result has no Axle action." and one secondary button **Switch to smart charging** that sets the model control; Run stays an explicit click. No charts.

**Update (decision 0004 items 55, 56 and 58).** The 2 Response lens's metric control gained a third option, **Events**, beside Home import and Battery SoC: for one chosen scripted event (a cold still evening, a surprise spike, a DFS turn-down or a local turn-up, item 55), it reads `events` and `event_response_bands` for delivered kW against the baseline, the rebound and the payment bucket, scoped to the fleet or to the event's zone. See section 2 for the current lens list.

**1 Plan**

```
Desktop 1440
+----------------------------------------------------------------------------------------------+
| Run bar                                                                                      |
| Smart charging                                [1 Plan | 2 Response | 3 Value and risk]       |
| Smart charging: each EV charges just enough to reach its target in the cheapest forecast     |
| half-hours before its usual departure time minus 1 hour.                                     |
| **Forecast price, planning week (£/MWh)**                       [Synthetic]    height 380 px |
|  price line; the half-hours the fleet's smart plans used, shaded teal on a second axis       |
| Average price paid: 116 -> 53 £/MWh (normal -> smart). 83% of home import moved to a         |
| half-hour cheaper than the one it would otherwise have used.                                 |
| **Energy by forecast-price third (kWh/week)**                                  height 320 px |
|  bars: normal vs smart, Cheap · Mid · Expensive                                              |
| Smart charging concentrates import in the cheapest third; normal charging follows plug-in    |
| time instead.                                                                                |
| > Data and definition                                                                        |
+----------------------------------------------------------------------------------------------+
```

Price caption: "Synthetic AR(1) price path; not a market forecast." The shaded half-hours are the union of every EV's planned slots that week, not one EV's plan; Drivers ▸ One EV marks one EV's own plan and its plug-in and departure markers.

**2 Response**

```
+----------------------------------------------------------------------------------------------+
| Smart charging                                [1 Plan | 2 Response | 3 Value and risk]       |
| Smart charging: … (summary sentence)               Metric [Home import | Battery SoC]        |
| **Home import: normal vs smart (kW)**                                          height 380 px |
|  ···· normal P50 + band   ── smart P50 + band   (P10–P90 across 100 weeks)                   |
| **Smart minus normal, paired per week (kW)**                                   height 380 px |
|  amber band; a new peak where the fleet's cheapest forecast half-hour falls, a cut where the |
|  evening plug-in peak used to be                                                             |
| Finding: within one simulated week, smart charging can still herd much of the fleet onto     |
| that week's own cheapest half-hours, because every EV plans against one shared day-ahead     |
| price signal per week and there is no site or network limit to stop it. Each simulated week  |
| has its own day-ahead price path (decision 0004 item 48), so the peak's size and timing vary |
| week to week rather than sitting at one fixed clock time; this stays labelled as a finding,  |
| not corrected with a fleet-wide charging cap.                                                |
| Differences are computed per simulated week, then summarised; not the gap between the bands. |
| > Data and definition                                                                        |
+----------------------------------------------------------------------------------------------+
```

"Plugged in" is dropped from this lens's metric control (item 43): plug-in share is Drivers' question, not what changed under the action. Two stacked charts sharing an x range, 380 px each (item 26: time series never below 380 px); one metric control drives both; one shared data expander holds both frames.

**3 Value and risk**

```
+----------------------------------------------------------------------------------------------+
| Smart charging                                [1 Plan | 2 Response | 3 Value and risk]       |
| Smart charging: … (summary sentence)                                                         |
| Median weekly saving   P10–P90              Early departures         Unrecovered energy      |
| −£41                    −£63 to −£12          47.5 sessions/week      2.1% of weekly import   |
|                                               (of about 1,280 plug-ins)  ⚠ above threshold     |
| **Weekly saving (selected − normal, £)**                                       height 320 px |
|  strip plot, 100 dots; coloured by shortfall magnitude, not a fleet-wide binary ⚠ (item 45)   |
| Negative = cheaper than normal. Illustrative; not a bid, settlement or Axle cash (0003).      |
| Component                     P50       P10       P90                                        |
| Home import cost               …         …         …         <- visible table, 5 rows       |
| Public charge cost              …                                                            |
| Unrecovered energy value          …                                                          |
| Unserved travel value               …                                                        |
| Total                                  …                                                      |
| Public rate £0.79/kWh is an editable illustrative assumption.  (Edit assumptions)             |
| Driver risk: EV-sessions that left before their smart plan finished, and the battery-side    |
| energy those plans still owed, so a viewer sees who bore the cost of the saving, not only     |
| the £.                                                                                        |
| > Data and definition (per-week table with flags)                                            |
+----------------------------------------------------------------------------------------------+
```

Renamed from "Cost" (item 43): the lens leads with the weekly saving and the driver-side risk together, because a saving that leaves drivers short is not free. The not-recovered indicator is a magnitude (unrecovered kWh as a share of weekly home import, and EV-sessions affected) with a material threshold for the ⚠, not the earlier fleet-wide yes/no flag (item 45); the strip plot is coloured by that magnitude instead of an all-or-nothing marker. The components table stays visible because it is the cost table the decisions require.

390 px: the numbered lens control fits one row; KPIs stack; the components table scrolls inside its own frame, never the page.

### 4.4 Compare (Q8, U03, 0004 item 6)

```
Desktop 1440
+----------------------------------------------------------------------------------------------+
| Run bar                                                                                      |
| Compare                                  A [Run 2 · base v]   B [Run 3 · home charger 11 kW v] |
| What changed between two runs? [Model output]   Matched futures: yes (seed 42, 100 weeks)    |
| Changed: home charger power 7.0 → 11.0 kW                                                    |
| Weekly home import    Median plug-in SoC   Unserved travel     Illustrative total            |
| +1,240 kWh (B − A)    +3 pts               0 kWh               +£4.10                        |
| **B − A, paired per week: Home import (kW)**  height 380 px  Metric [Home import | SoC | Plugged in] |
|  amber band P10–P90 across 100 weeks around zero                                             |
| > Data and definition                                                                        |
+----------------------------------------------------------------------------------------------+
```

Every completed run is kept automatically in the session with its settings snapshot, labelled "Run N · <first changed assumption>" (or "base" for defaults); at most three are held and the oldest drops with a toast. A defaults to the previous run, B to the latest. KPIs show B − A P50 with P10–P90 in `help` and a caption. Both runs use the same seed and sampling order by default, so the difference is paired per world. If seeds or fleet size differ: "Not matched: B uses seed 7. Differences mix the parameter effect with sampling noise." and the comparison still shows. Session only; no export. Empty state (fewer than two runs): "Run once, change an assumption in Edit assumptions, run again. Each run is kept here automatically (last three)." with the Edit assumptions button.

390 px: A and B selectors stack under the title; KPIs stack.

### 4.5 How it works (U09, brief "brief notes on decisions", 0004 item 24)

Renders fully before any run. Lenses:

- **Why this model** (default, the landing lens): Mike's opening note, `docs/explainers/why-this-model.md`, shown inline in the prose column (what the model is for, how a forecast is drawn, every parameter named, where detail matters, uncertainty as the margin, one model for many customers, what it is not yet), then the Next button on to Overview ▸ At a glance's tour.
- **The forecast**: five numbered steps with one diagram (population → evaluation weeks → each EV's smart-charging plan at plug-in → the kernel for the normal and smart paths → world-first summaries), each with the equation in `st.latex` where one exists; the full `docs/explainers/how-the-forecast-works.md` in an expander below.
- **Assumptions**: one read-only table from `assumptions.py` (label, value, unit, evidence badge, source, meaning, affects), filterable by evidence class with a `st.segmented_control` (All · Source · Illustrative · Synthetic), showing draft and active-run values when they differ. **Edit assumptions** button at the top opens the dialog.
- **Limits**: **Not modelled** list first (one line each on why), then known limitations (no calibration, synthetic prices, 7 kW home chargers only, DST handling), then the CNZ report and Axle sheet crosswalk in an expander.
- **How it was built**: Mike's build notes (what he spent time on, what he left out, how the design serves Axle's end use), then `docs/explainers/how-it-was-built.md` in an expander.

```
Desktop 1440
+----------------------------------------------------------------------------------------------+
| Run bar                                                                                      |
| How it works    [Why this model | The forecast | Assumptions | Limits | How it was built]    |
| How is the forecast calculated?                                                              |
| 1 Population  →  2 Evaluation weeks  →  3 Smart-charging plan  →  4 Kernel  →  5 Summaries   |
| [diagram, height 260 px]                                                                     |
| 1. Sample 1,000 drivers …  $E_{t+1} = E_t + \eta P_t \Delta t - d_t$                          |
| …                                                                                            |
| > Read the full explainer                                                                    |
+----------------------------------------------------------------------------------------------+
```

Text blocks keep a readable measure: prose sits in `st.columns([3, 1])`'s wider column at desktop.

### 4.6 Presentation path (about 4 minutes)

1. **Overview, empty (0:00).** Click Run. While it runs (about 40 s) the page stays readable: read the three-sentence intro and the six archetypes with their source shares.
2. **Overview, result (0:45).** The Average day chart is the brief's sketch 2: bars plugged in, SoC mean with the across-driver band. Point at the two headline tiles and the CNZ 52% context.
3. **Drivers ▸ Plug-ins (1:30).** The brief's two literal asks side by side: plug-in time and SoC at plug-in, with CNZ markers.
4. **Drivers ▸ One EV (2:00).** Sketch 1: one driver's week, plug-in markers, the plug-event table. Archetypes only if asked.
5. **Smart charging 1 → 2 → 3 (2:30).** How the smart charger plans a session and the price it paid, what changed in home import (the paired difference, and the herding finding), and the weekly saving with the driver-side risk (early departures, shortfall).
6. **Edit assumptions → Run → Compare (3:30).** Change home charger power or the public rate in the dialog, Run from the dialog footer, open Compare: matched futures, B − A.
7. **How it works ▸ Limits (if asked).** The Not modelled list and why.

### 4.7 Trading (built; decision 0004 items 43, 57, 58)

Question: what did the illustrative simulated trading overlay buy, sell and settle? A trading overlay reads the physics kernel's outputs and never changes them; all strategies come from one physics run, so common random numbers hold (plan §2F). The planner is `action.plan_cheapest_slots` applied to expected sessions, not a fleet-aggregate LP (item 57's lead correction). Full spec: `docs/contracts/trading-events-v1.md` §4 (overlay), §5 (frames), §9 (trader metrics, commitment share, Supplier page frames). Lenses, quoting `ui/registry.py`:

1. **1 Market** ("What prices did the fleet trade against this week?") — day-ahead median with its P10–P90 across weeks; intraday close and the system imbalance price for a representative week; the daily price-level path; the negative-price share as a realism check.
2. **2 Position** ("What did the trader sell, and how did the fleet actually move?") — a representative night's fleet kW (baseline vs unmanaged vs metered, sold turn-down shaded); the position fan from the day-ahead position to the final position against the intraday price; the P10–P90 band of nightly deviation MWh across weeks.
3. **3 P&L and risk** ("What did the illustrative trading overlay make or lose, and how firm was it?") — weekly net per strategy as a strip or box across weeks; a bucket waterfall per strategy; tiles for net P50, net P10, CVaR5 and £/EV/yr; the baseline-effect and imbalance shares; a sanity-check table. Every £ is illustrative simulated trading P&L (decision 0005), never Axle cash.

Intraday dispatch (`docs/contracts/intraday-dispatch-v1.md`, item 62(b)) is landing now (lane K4): a committed share of EVs stays locked to the day-ahead plan while the rest re-plan hourly on the latest intraday price when the saving clears the re-plan threshold, splitting intraday P&L into rebalancing and re-optimisation. The dispatch kernel and frames are merged (`trading.intraday_dispatch`, `trading.replan_threshold_gbp_per_mwh` in Edit assumptions); K4 wires them into the 2 Position lens and the trader metrics and turns the switch on by default.

### 4.8 Supplier (built, including lens 6 Customers; decision 0004 items 58, 59, 62, 65)

Question: what can a supplier buy from this fleet, and what is it worth? Full spec: `docs/contracts/supplier-v1.md`; firm MW in `docs/contracts/trading-events-v1.md` §10; lens 6 in `docs/contracts/household-v1.md` §6.2. Lenses, quoting `ui/registry.py`:

1. **1 Availability and cost curve** ("What could the fleet move, and at what price?") — turn-down/turn-up MW with P10–P90, and a flexibility cost curve (MW available above each £/MWh at a chosen half-hour) (item 58(D)).
2. **2 Positions** ("What was sold day-ahead, and what was held at the end?") — a positions table with CSV download, and upload of the supplier's own 48-value price curve (item 58(D)).
3. **3 Supplier P&L** ("What does smart charging do to a supplier's week?") — energy cost unmanaged vs smart, hedge error (profiled vs flat), shape premium, grid-event payments, customer rewards, an optional platform fee, net supplier gain per customer per month with P10–P90 (decision 0006), plus the hedge-block view, de-rated firm capacity and CO₂-shifted figures (items 58(D), 65). The trading ledger's net is shown beside it and never added (supplier-v1 §3.1).
4. **4 Firm MW** ("How much turn-down can be promised firmly?") — deliverable MW per half-hour for turn-down and turn-up, held for 1, 2 and 4 hours, at two horizons, with a leave-one-week-out calibration backtest, a product sheet, blackout windows and firmness by manufacturer (item 59; trading-events-v1 §10).
5. **5 Charger makers** (renamed from "5 Partners" once the Partners page existed, §4.10 below; "What does a charger maker's enrolled device earn?") — £ per enrolled device per month, share of devices earning, charge-by-departure rate, guaranteed-floor coverage, by cohort and manufacturer, plus the growth-lever calculator (items 62, 65).
6. **6 Customers** (household-v1 §6.2, lane HH4; "What can we tell customers, and which customers gain least?") — the supplier's marketing view: typical value per year, sessions ready at departure and smart charging cost, each "if customers paid day-ahead prices" with its honest band, by archetype, naming the least-gaining archetype (items 58, 62, 65).

### 4.9 Replay (built; decision 0004 item 66)

Question: what did the fleet know, do and earn at each moment of one simulated week? Full spec: `docs/contracts/replay-v1.md` §3. A "play the week" screen for one simulated week: a time cursor (a vertical "now" line) that can be dragged or played forward and back, drawn with Plotly animation frames so dragging is smooth without reruns; world and EV samples are kept small (item 66). Lenses, quoting `ui/registry.py`:

1. **Fleet** ("What did the fleet know, do and earn as the week unfolded?") — at each instant, what was known then (day-ahead prices as published, the latest intraday path for later half-hours) against what actually happened (realised prices so far), the fleet's actions (unmanaged vs smart import, positions and trades, grid events and shocks) and running money (saving, trading P&L) to that instant.
2. **Customer** ("What did one driver's charger know, plan and save as the week unfolded?") — the same idea for one EV: its plugged-in periods, SoC, the plan in force at the cursor, charging so far and its own saving to date.

### 4.10 Partners (built; Mike, 30 Sep overnight review log)

Question: who is this for, and what would each partner want to see? A pitch card per outside audience, after Supplier and before Replay. Every lens reuses an existing chart builder and frame with the same default row selection its own page uses (fleet group, that page's own default day type, the profiled hedge, a 1 h turn-down) — no new model output and no new figure. One caption under the lens selector on every persona: EV-only model; batteries, heat pumps and V2G are not modelled (How it works ▸ Limits; no DNO/NESO persona, for the same reason). Lenses, quoting `ui/registry.py`:

1. **Driver** ("What would a driver want to see?") — value per year, charged-by-departure share, price paid (smart vs unmanaged) and early departures as tiles; `customers._value_year_figure` (fleet only) as the hero; `action_cost._strip_figure`, `partners._departure_figure`, `plug_ins._soc_figure` and `overview._average_day_figure` as supporting charts.
2. **Energy supplier** ("What would an energy supplier want to see?") — supplier gain per customer, energy moved, evening value lost to baseline erosion and shape-cost saving as tiles; `supplier_pnl._waterfall_figure` as the hero; `action_response._paths_figure`, `supplier_pnl._block_figure`, `supplier_availability._curve_figure` and `supplier_firm_mw._day_ahead_figure` as supporting charts, plus a "Grid route: firm MW" expander (calibration, fleet-size diversification, the product sheet) captioned "Illustrative; not a registered product."
3. **Charger maker** ("What would a charger maker want to see?") — gross cash per device, devices earning, £ per kW of charger and months to fund a discount as tiles (the growth calculator's own accepted defaults, `partners.FUNNEL_DEFAULTS`); `partners._market_figure` as the hero; `partners._exceedance_figure`, `partners._funnel_figure`, `partners._payout_figure` and the firmness-by-maker table as supporting charts.
4. **Carmaker** ("What would a carmaker want to see?") — median plug-in SoC, nights plugged in, battery cycles and CO₂ shifted as tiles; `sessions._figure` (Dwell) as the hero; `plug_ins._heatmap_figure`, `partners._departure_figure`, `plug_ins._hour_figure` and `customers._kw_figure` as supporting charts.
5. **Fleet and leasing** ("What would a fleet or leasing operator want to see?") — saving per vehicle, ready at departure, public top-up change and battery cycles as tiles; `household._outcomes_figure` (via the shared `ev_picker`, its EV choice shared with Drivers ▸ One EV/Household) as the hero; `sessions._figure` (Flexible kWh) and `action_cost._strip_figure` as the only two supporting charts (thinner than the other personas: the six archetypes are household drivers, and there is no depot, site limit or shift pattern to draw on), with a stated caveat and drill-through limited to One EV and Household.

Every persona also carries a "What they would ask" table (Want, Reading, Where): the reading is pulled from `key_stats.key_stats_table` by its exact Metric text, or reads "Not modelled" when the digest's want has no stored column (never a value invented for the page), and "Where" always names a screen, never a frame.

## 5. Data contract per chart

"Exists" means on the result today; "Needed" is a new model output owned by T1. The UI reads these fields only.

| Chart | Source | Status |
| --- | --- | --- |
| Run bar identity | `settings`, `policy_id`, `evidence_kind` | Exists |
| Overview average day: % plugged in by local half-hour | new `average_day_bands`: per path × day type × local half-hour, P10/P50/P90 across worlds of connected share | **Needed** |
| Overview average day: SoC mean and P5–P95 **across EVs** | same frame: per world, per-EV SoC quantiles at each slot, then world-median | **Needed** (fast aggregate only returns fleet sums) |
| Fleet week bands | `fleet_interval_bands` (`connected_count`, `realised_grid_kw`, `closing_soc_percent`, `driving_fraction`, `away_on_trip_fraction` × mean/p10/p50/p90). Note: `*_fraction` columns are fleet sums in EV units, not 0–1 shares; the view divides by `unit_count` or T1 renames them | Exists (naming fix recommended) |
| Plug-in time and SoC histograms | new `plug_in_event_summary`: per world × path × day type, counts by local hour and by 5% SoC bin, plus P10/P50/P90 of shares across worlds and median SoC | **Needed** (only the one-EV replay has plug events) |
| CNZ reference marks | `cnz_plug_comparison` source constants (median 52%, <10% share 3%, weekday mode ~18:00), moved into `assumptions.py` as Source-class context | Exists, move |
| Archetype table and small multiples | `cohort_interval_distribution` + `average_day_bands` and `plug_in_event_summary` split by `cohort_id` | Partly needed |
| One EV chart and events | single-EV kernel slice (0004 item 20): intervals, `plug_events` (`connection_start_utc`, `plug_in_soc_fraction`, `pre_charge_battery_kwh`), daily audit (shortfall) | **Needed** for action results |
| Smart charging ▸ Plan: price ranking and smart-charging outcome | `action_summary`; `smart_charging_world`/`smart_charging_summary` (`average_price_change_gbp_per_mwh`, `moved_home_import_share`, `early_departure_count`, `early_departure_shortfall_kwh`); `price_band_shift` (per-path `normal_kwh_*`/`selected_kwh_*` by forecast-price band, decision 0004 item 38) | Exists |
| Overview ▸ At a glance flexibility tiles, Drivers ▸ Fleet week flexibility metrics (decision 0004 item 54 retired the flexible-power hero chart and the Flexibility lens) | `flexibility_weekly_summary`, `weekly_peak_summary`, `deferrable_power_bands`, `flexibility_bands` (`turn_up_headroom_kw`, `movable_energy_kwh`; `time_slack_hours` in the data expander only) | Built |
| Drivers ▸ Plug-ins heatmap | `plug_in_heatmap` (chance a plug-in starts, by weekday × London hour, mean and P10/P50/P90 across weeks; item 43) | Exists (model); heatmap chart **Needed** |
| Forecast price | `forecast_synthetic_prices.wholesale_forecast_gbp_per_mwh`, `forecast_available_at_utc` | Exists |
| Normal vs selected bands | `fleet_interval_bands` with `path_id` normal/selected | Exists |
| Paired difference band | new `fleet_difference_bands`: per slot, P10/P50/P90 of per-world (selected − normal) for import kW, SoC %, connected | **Needed** (0004 item 12) |
| Cost strip plot and components | `wholesale_world_cost_effect` (all `illustrative_*_gbp`, `selected_minus_normal_*_kwh`, `energy_not_recovered`) | Exists; unserved-travel value column per 0004 item 13 to confirm |
| Compare differences | new `compare_matched_runs(a, b)` returning per-world weekly totals and B − A quantiles, refusing silently mismatched worlds | **Needed** |
| Edit assumptions dialog, How it works ▸ Assumptions | `model/assumptions.py` records (value, unit, evidence, source, meaning, affects) | **Needed** (0004 item 21) |

## 6. Build plan

Every task follows `AGENTS.md`: isolated `.worktrees/claude-<task>` worktree, owned files only, one fresh independent review of the frozen commit, lead integrates. View functions share one signature: `render_<name>(st, result) -> None` (the `stale` argument is dropped: stale is shown once by the run-bar chip), read only from the result, no model calls except the one-EV replay entry point. Chart blocks use one helper, `chart_block(st, fig, *, title, caption, frame, definition)`, in `components/chart_table.py` (T2), which renders section 3.4. Every view task (T3–T6) sets each chart's height from the section 3.5 size table and passes the browser check at 1440 and 390 px with no clipped or overlapping legend, axis title or annotation (0004 item 26).

| Task | Owns (disjoint) | Model | Depends on | Acceptance |
| --- | --- | --- | --- | --- |
| **T1 Model summaries** | `model/explicit_summaries.py` (new), additive fields in `model/explicit_action_runner.py` and `model/explicit_runner.py` result dataclasses, per-EV quantile and plug-event hooks in `model/explicit_array_aggregation.py`, their tests | Opus | action runner on main; the smart-charging planner (item 38) and flexibility inputs (items 43, 46) for the new frames; assumptions module for CNZ constants | New frames exist on both results with documented columns; plug-in counts reconcile with a scalar replay on a 3-EV fixture; difference bands computed world-first (test that differs from band subtraction); `compare_matched_runs` refuses silently mismatched worlds; runtime at 1,000 × 100 within +15% of current |
| **T2 Shell** | `streamlit_app.py`, `ui/registry.py`, `ui/pages.py`, `ui/run_controller.py` (run bar, run-at-end mechanics, dialog wrapper, run history of the last three results with settings snapshots and labels), `ui/style.py` (tokens, Plotly template), `ui/components/chart_table.py` (`chart_block`), `.streamlit/config.toml`, `assets/wordmark.svg`, `DESIGN.md`; deletion of retired views and their tests | Sonnet | removal of superseded modes (0004 item 18) | Five pages in order with top nav and no sidebar; run bar with model segmented control, chip for all six states, Edit assumptions and one primary Run; empty state with body Run button; `@st.dialog` wrapper calls T6's editor body (a placeholder body until T6 lands); zero full runs on open, navigation, lens change, dialog edits or dialog Reset (AppTest); run history keeps three and labels them by first changed field; retired lenses gone from registry test; tokens and template in `style.py`, including the item-26 legend-below and margin defaults; browser 1440 and 390: every chart at its table height with no clipped or overlapping legend, axis title or annotation, Run reachable without opening any menu, page readable while running, dialog opens, edits, closes and runs |
| **T3 Drivers** | `ui/views/drivers_fleet.py`, `plug_ins.py`, `archetypes.py`, `one_ev.py` (adapt `fleet.py`, `explicit_distribution.py`, `explicit_cohort_bands.py`, `explicit_individual_bands.py`, then delete the originals), their tests | Sonnet | T1 (plug-ins, average day), T2 contract; One EV action support needs item 20 | Lens order Plug-ins, One EV, Archetypes, Fleet week; each renders from a small fixture result via `chart_block`; London x-axis; CNZ markers labelled; Fleet week normal path only with four metrics; Archetypes 3×2 subplots, visible table, band in expander; One EV has one control above the chart and shows the item-10 message for action results until replay lands |
| **T4 Smart charging** | `ui/views/action_decision.py`, `action_response.py`, `action_cost.py`, tests | Sonnet (Opus review) | T1 difference bands and the smart-charging summary fields (`smart_charging_world`, `price_band_shift`); live action runner wiring | Summary sentence on all three lenses; reason codes mapped to plain text; flagged weeks visible without colour; components table visible; no Axle-cash wording (grep test); no-action result shows the message and a Switch to smart charging button that changes only the draft model |
| **T5 Overview and Compare** | `ui/views/overview.py` (new file replacing the legacy one), `ui/views/compare.py` (replaces `scenario_compare.py`, `scenario_reference.py`), tests | Sonnet | T1 (`average_day_bands`, `compare_matched_runs`), T2 run history API | Overview KPIs match T1 frames on a fixture, including the no-action tile set; page links land on the named lens; Compare defaults to previous vs latest from T2's run history; unmatched seeds labelled; empty state with fewer than two runs |
| **T6 Assumptions editor and How it works** | `ui/views/parameters.py` (dialog body `render_assumptions_editor(st, draft)`), `ui/views/methods.py` (How it works; replaces `explicit_methods.py`, `calculations.py`, `variable_guide.py`, `evidence.py`, `assumptions.py` view), tests | Sonnet | `model/assumptions.py` refactor (0004 item 21); explainers (0004 item 24) | Every editable field comes from `assumptions.py`, grouped in tabs, meaning in `help`; each has a causal test or is read-only; draft/active caption only on changed fields; Reset to active and Reset to defaults work; How it works has four lenses, Not modelled list first in Limits, both explainers rendered; slider/draft mismatch fixed with a regression test |

`ui/scenario_store.py` is no longer created; T2's run history replaces it. `downloads.py` is outside this spec and left to the lead.

Order: T1 and T2 start now in parallel. T3–T6 start when T2's page contract (`render_<name>(st, result)`, `chart_block`, run history API, dialog hook) is frozen; they build against a fixture result before T1 lands, then switch to real frames. Final: combined suite, then `docs/VERIFY_APP.md` browser journey at 1440 and 390 px following the presentation path (section 4.6): open, Run, every page and lens, edit an assumption in the dialog (stale chip), Run from the dialog, Compare two runs.

**Further build lanes (items 54–66).** The table above is T1–T6 only. Trading (F1–F3), Supplier (S1–S4) and Replay (R1+) build lanes, and the Household lanes HH1–HH4, are specified in their own contracts rather than duplicated here: `docs/contracts/trading-events-v1.md` §12 (with §9.7 and §10.7), `docs/contracts/supplier-v1.md` §14, `docs/contracts/replay-v1.md` §6, `docs/contracts/household-v1.md` §10. As of this revision: all of HH1–HH4 and the Trading, Supplier and Replay UI builds are merged to `main`.

## 7. Open questions for Mike

1. **Across-EV spread on the Overview (P5–P95) versus across-world spread everywhere else.** Recommended: yes, show both, always named, because the brief's sketch shows driver variation and our fleet bands show forecast uncertainty.
2. **Top navigation instead of the sidebar.** Recommended: top, so Run and navigation both work at phone width. The fallback is a sidebar for pages, with the run bar still in the main area.
3. **One EV defaults to one representative week, not the all-world median line.** Recommended: one week, because a median SoC line across worlds is not a driver anyone could observe; the band stays as an option. This reverses part of M05's default, so it needs Mike's approval.
4. **Selected-path colour stays teal (the UI accent).** Recommended: yes; "what Axle chose" and "current selection" mean the same thing here.
5. **Five pages instead of six (amends 0004 item 22).** Parameters becomes the Edit assumptions dialog and Scenarios becomes Compare with automatic run history. Recommended: yes, because the demo's edit-and-rerun step then happens on the page being discussed and saving scenarios needs no extra step. Fallback if the dialog proves awkward in the browser (for example a rerun closing it mid-edit): keep the same editor body as an editable How it works ▸ Assumptions lens; no other page changes.
