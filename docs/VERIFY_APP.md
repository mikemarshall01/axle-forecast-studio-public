# Independent Python/Streamlit app verification

Use this at a usable milestone and before release, not after every small task. The verifier is independent of the implementation under test, reads the accepted task criteria and candidate revision, and reports evidence rather than trusting a worker's completion claim.

## Local checks

Run in the candidate checkout, with its locked development environment:

```bash
uv run --frozen --group dev ruff check .
uv run --frozen --group dev ruff format --check .
uv run --frozen --group dev pytest -q
```

Execute both notebooks top to bottom on the project kernel (decision 0004 item 17) and the archetype script (item 16):

```bash
uv run --frozen --group dev jupyter nbconvert --to notebook --execute --output /tmp/01_check.ipynb notebooks/01_model_metric_walkthrough.ipynb
uv run --frozen --group dev jupyter nbconvert --to notebook --execute --output /tmp/10_check.ipynb notebooks/10_plug_propensity_toy.ipynb
uv run --frozen --group dev python scripts/run_archetypes.py
```

Report each command's actual result. If the environment cannot be installed or a check cannot run, mark it `NOT RUN` with the reason; never turn that into a pass. Use a dedicated port and isolated temporary outputs for browser or container checks so active worker processes do not collide.

## User journey

Use Streamlit `AppTest` for repeatable state and widget assertions, plus one real-browser pass for layout and interaction behaviour at 1440 px and at 390 px. At the relevant milestone, verify:

1. Opening the app, navigating the nine top-bar pages (How it works, Overview, Drivers, Smart charging, Trading, Supplier, Partners, Replay, Compare), opening and closing the Edit-assumptions dialog, Reset (to active run and to defaults), changing a lens and ordinary Streamlit reruns start **zero** full Monte Carlo runs. The app opens on How it works ▸ Why this model (Mike's opening note), not Overview (Fable landing pass).
2. There is no automatic draft preview: decision 0002's bounded-preview design is superseded and retired (status confirmed 26 September 2026), so editing the draft never renders a partial or "Preview, not a full run" result. Only an explicit, valid **Run simulation** click starts a full run.
3. An invalid draft disables Run and shows its field error inside the Edit-assumptions dialog; a duplicate click while a run is active starts no additional run; a failed run retains the previous active result and shows the one-line failure reason.
4. The run bar shows the correct one of the six worded states — No result, Running, Current, Stale · N changes, Run failed, Draft invalid — the chip's word (not colour alone) is the cue, and the identity caption (seed, EV count, world count, model, run date) matches the active result.
5. Both explicit models are reachable from the run-bar's two-option control: **Smart charging** (default) and **No action** (selectable). A smart-charging run applies price-optimised smart home charging (decision 0004 item 38): at each home plug-in, the charger plans to reach the preferred target by the cohort's typical departure clock time minus the departure margin (editable, default two hours, item 51), using the cheapest half-hours of that simulated week's own day-ahead price path (item 48), capped at home charging power — there is no fixed action window, no per-EV eligibility screen and no separate planning world; every EV plans and executes its own session, a plan window running past the study end is simply clipped there (the noon-to-noon study horizon, item 52, made this the general case and retired item 49's separate "charges as soon as plugged in" rule), and one that leaves earlier than its plan expected simply leaves short. No EV strands: when a trip would take a battery below the top-up threshold (10% SoC by default, item 37) the EV tops up at an always-available public charger to the target (80%) and continues, with the top-up recorded as public import in that slot (item 32), and unserved travel is zero in every world as a check; runs start settled at 80% SoC after a warm-up (seven days by default, editable 3–14, item 52); the `energy_not_recovered` flag and the paired selected-minus-normal P10/P50/P90 band are computed world-first. No charging taper is applied by the kernel — the app and explainers state this as a limitation, not an implemented behaviour. A no-action run shows its own Overview tiles, not a "not in this model" placeholder.
6. Each of the nine pages renders its hero chart and KPI row from the stored result only, with no page recalculating physics, policy or settlement: How it works (Why this model · The forecast · Assumptions · Limits · How it was built, needing no result); Overview (At a glance · Key stats); Drivers (Plug-ins · Sessions · One EV · Household · Archetypes · Fleet week); Smart charging (1 Plan · 2 Response · 3 Value and risk); Trading (1 Market · 2 Position · 3 P&L and risk); Supplier (1 Availability and cost curve · 2 Positions · 3 Supplier P&L · 4 Firm MW · 5 Charger makers · 6 Customers); Partners (Driver · Energy supplier · Charger maker · Carmaker · Fleet and leasing); Replay (Fleet · Customer); Compare (no lens, also needing no result).
7. The Edit-assumptions dialog opens from the run bar on every page and from How it works ▸ Assumptions, edits only the draft, offers Reset to active run and Reset to defaults, and its footer **Run simulation** starts a full run from a valid draft.
8. Compare shows the automatically kept run history (last three runs, each labelled by what changed) and a per-slot paired difference band between the selected pair; the combined total is shown as unavailable, not zero, when either selected run is the no-action model (decision 0003). **PASS/FAIL (decision 0004 item 6):** two runs differing only in an assumption edit share the same seed and sampling order (matched futures), so draws in every unchanged channel are identical and the per-slot B − A band is a true paired comparison, not a difference of independent samples. `FAIL` if Compare offers or labels as "matched" any pair of runs that do not share seed and sampling order.
9. Every chart is fully visible with no clipped, overlapping or squashed title, legend or annotation at both 1440 px and 390 px (decision 0004 item 26): explicit height from `CHART_HEIGHTS` in `ui/style.py` suited to content (time series ≥340 px, small multiples ≥180 px each, histograms ≥280 px), full container width, and legends moved below the plot at 390 px.
10. Chart conventions match decision 0004 item 27: time axes show London local time with UTC kept in hover text and tables; the standard centre line is the median (P50) with a named P10–P90 band for spread across simulated worlds; the normal path is a lighter solid line, the selected path is teal, and bands use about 0.3 opacity. Carve-out: the Overview average-day chart shows spread across EVs (population variation), not across worlds, per `docs/DASHBOARD_DESIGN.md` — its SoC band is the design's stated P5–P95 across-EV band, not P10–P90. Every chart with a band names which spread it shows ("across EVs" or "across N simulated weeks"); `FAIL` if a chart's caption is silent on this or mislabels one spread as the other.
11. Every chart in the app, in `scripts/run_archetypes.py` and in both notebooks is Plotly rendered through the shared template in `ui/style.py` (decision 0004 item 29); a repository grep finds no `matplotlib`, `altair` or `st.*_chart` built-in call, and each notebook chart is paired with a short text table.
12. A small deterministic fixture preserves world-first aggregation and the other accepted physical invariants: additive EV and cohort quantities are summed within each complete world before quantiles are taken across worlds, and no percentile (cohort, route or half-hour) is summed to build a fleet or paired-difference percentile.
13. A run of 1,000 EVs × 100 simulated worlds completes; its wall-clock time and peak memory (RSS) are measured and recorded (no time limit, decision 0004 item 33).
14. Two viewer sessions keep draft, active result, run history and view state separate; shared caches contain no mutable per-viewer result state.
15. `docs/explainers/how-it-was-built.md` and `docs/explainers/how-the-forecast-works.md` are present, shown on How it works, and their file and function references are checked against the current code.
16. Original PDFs, credentials, local databases, generated full arrays and secrets (including `.streamlit/secrets.toml`) are absent from commits, exports and the Docker image, except the precomputed default run the Docker build generates from public code for the hosted demo (AGENTS.md).

Only check a requirement when its implementation and accepted contract exist; an unimplemented required journey is `FAIL` for a full-app milestone, not `NOT APPLICABLE`. Hosted access, resource and two-viewer checks need the separately authorised hosting step. Do not push, deploy or select a fallback provider from this checklist alone.

## Report

Record the candidate revision, environment, exact commands and `PASS`/`FAIL`/`NOT RUN` result for each check, plus browser evidence, limitations and a final verdict. A failure blocks the app-complete claim until repaired and rechecked. A task-level reviewer may approve an earlier slice without claiming that this whole-app checklist has passed.
