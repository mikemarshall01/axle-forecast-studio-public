# Axle Forecast Studio

Axle Forecast Studio is a Monte Carlo model of an illustrative 1,000-EV fleet, written in Python, with a Streamlit dashboard. I built it to answer two plain questions about six driver types, when they plug in at home and how full the battery is when they do, and then the questions on top: how much charging can move to cheaper half-hours, what that is worth, what it costs the driver, and how firmly it could be promised. It simulates each EV's battery half-hour by half-hour over one week, in many simulated weeks, and tests one illustrative action, smart home charging: each home session is planned against that week's own day-ahead price path, filling the cheapest half-hours first to reach the driver's preferred battery target by their expected departure.

It is an interview prototype. The six driver archetypes come from Axle's workbook; the other behaviour values, the weather, public charging and prices are illustrative or synthetic assumptions I state. Nothing is calibrated to observed fleet data, and no figure is a real tariff, settlement amount or Axle cash.

## Try it online

A hosted copy is linked in my submission email. It opens on How it works ▸ Why this model, with a saved full-size run (Run 0: 1,000 EVs over 100 simulated weeks) already loaded, so every page has numbers straight away. You can change any assumption under **Edit assumptions** and click **Run simulation**. To keep the shared machine quick, live runs there are capped at 300 EVs over 50 weeks (under a minute) and only a few can run at once. A local copy has no limits.

## Run it yourself

You need Python 3.12 (`>=3.12,<3.13`) and [uv](https://docs.astral.sh/uv/). uv can install Python 3.12 for you. A local copy has no run limits.

```bash
uv sync --frozen --group dev
uv run --frozen python scripts/build_demo_run.py     # optional, once: saves the default run (a few minutes)
uv run --frozen --group dev streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). With the saved default run built, the app opens on it as **Run 0**; without it, nothing is shown until you click **Run simulation**. The model runs only when you click Run. The default is the smart charging model, 1,000 EVs and 100 simulated weeks, which takes a few minutes and a few GB of memory; a smaller fleet or fewer weeks under **Edit assumptions** runs faster. Change a value, then Run again; **Compare** shows the difference between the last two runs. `docs/explainers/demo-run.md` explains the saved run, including `--vehicles` and `--weeks` for a quicker, smaller one.

Fast tests (the slow ones execute the notebooks):

```bash
uv run --frozen --group dev pytest -m "not slow"
```

### Archetype script

`scripts/run_archetypes.py` runs the no-action model under preset driver archetypes (for example "commutes every weekday") and prints plug-in timing and plug-in SoC next to the CNZ report's figures, as context, not a target.

```bash
uv run --frozen --group dev python scripts/run_archetypes.py
uv run --frozen --group dev python scripts/run_archetypes.py --html archetypes.html
```

### Notebooks

`notebooks/01_model_metric_walkthrough.ipynb` walks through the model and its metrics. `notebooks/02` to `09` are concept notebooks, one per area (driver behaviour, prices and markets, smart charging, flexibility, trading, events and zones, the supplier view, the top-down quantile model). `notebooks/10_plug_propensity_toy.ipynb` is a toy PyMC plug-propensity model on synthetic counts and needs the `research` group. All use the kernel `axle-forecast-studio`:

```bash
uv sync --frozen --group dev --group research
uv run --frozen --group dev python -m ipykernel install --user --name axle-forecast-studio --display-name "Axle Forecast Studio (.venv)"
uv run --frozen --group dev --group research jupyter lab
```

### Docker (optional)

```bash
docker build -t axle-forecast-studio:local .
docker run --rm -p 127.0.0.1:8501:8501 axle-forecast-studio:local
```

The build runs the default model once (a few minutes) so the container opens on Run 0. This binds the dashboard to this machine only. For a shared host, the environment variables `AXLE_MAX_CONCURRENT_RUNS` (full runs at once) and `AXLE_MAX_EV_WEEKS` (largest fleet size × weeks per run) cap live runs; both are unset by default (`src/axle_studio/ui/run_guard.py`). `PORT` sets the port (default 8501).

## Checks

```bash
uv run --frozen --group dev ruff check .
uv run --frozen --group dev ruff format --check .
uv run --frozen --group dev pytest -q
```

## Project layout

```
streamlit_app.py            entry point: navigation, run bar, page
src/axle_studio/model/
  assumptions.py            every value, with unit, evidence kind and source
  clock.py                  fixed UTC half-hour grid, London time conversion
  settings.py               RunSettings and the cohort types
  sampling.py               population, trips, plug-ins, weather, prices
  physics.py                half-hour energy balance and world-first sums
  action.py                 price-optimised smart home charging and its illustrative cost
  events.py                 scripted events, presets and shock profiles
  market.py                 illustrative trading overlay: baseline, positions, settlement
  availability.py           firm-MW deliverable kW per world and slot
  availability_backtest.py  leave-one-week-out calibration backtest
  product.py                supplier-facing product sheet and settlement file
  supplier.py               illustrative supplier P&L and Partners frames
  growth.py                 Partners lens's growth-calculator arithmetic
  household.py              what one customer gets from smart charging
  replay.py                 fleet replay of one simulated week
  forecast.py               run_forecast and ForecastResult
  summaries.py              derived frames and Compare
  individual.py             one-EV replay
src/axle_studio/ui/
  run_controller.py         draft, explicit Run, run bar, run history
  demo_run.py               the saved default run (Run 0), shared across sessions
  run_guard.py              optional run limits for a shared host
  registry.py, pages.py     pages, lenses and the page-to-view table
  views/                    one file per page or lens
  style.py                  shared Plotly template
scripts/build_demo_run.py   builds the saved default run
scripts/run_archetypes.py   archetype runs and CNZ comparison
notebooks/                  walkthrough and toy research notebooks
tests/                      model and UI tests, fixtures
```

## Documentation

- `docs/CODE_GUIDE.md`: reading order through the code.
- `docs/explainers/why-this-model.md`: what the model is for and why it is built this way (the app's opening note).
- `docs/explainers/how-the-forecast-works.md`: the method in plain English.
- `docs/explainers/how-it-was-built.md`: the build process and key decisions.
- `docs/explainers/data-and-sources.md`: which values are source, illustrative or synthetic, and where each came from.
- `docs/decisions/`: accepted decisions; `0004-explicit-action-and-horizon-choices.md` holds most of them.
- `docs/contracts/results-v2.md`: the result shape the dashboard reads.
- `docs/MODEL_CONTRACT.md`: physical rules and invariants.
- `docs/DASHBOARD_DESIGN.md`: page layout and chart conventions.
- `docs/VERIFY_APP.md`: the whole-app verification checklist.

The opening note and the two explainers are also shown in the app under **How it works**.
