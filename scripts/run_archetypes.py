"""Run the illustrative explicit no-action forecast under preset behaviour archetypes.

This script owns nothing about the physical model. It calls the shared
``run_forecast(..., model="no_action")`` API (``model/forecast.py``) once per
named archetype and prints a compact population-level summary, so a reader
can see whether preset fleet behaviours (e.g. "commutes every weekday")
recapitulate the sort of plug-in timing and plug-in state-of-charge (SoC)
patterns a real fleet would show -- including a side-by-side comparison
against the CNZ May 2022 report context recorded in ``model/assumptions.py``
(``CNZ_CONTEXT``). That comparison is context, never a calibration target:
this script never fits the model to reproduce those figures.

Two existing model inputs drive an archetype: ``trip_behaviour`` (weekday/
weekend drive probability, dwell time, drive speed, and the departure and
plug-in clock spreads) governs *driving* and clock noise; the cohort
fixture's own ``arrival_local_hour``/``departure_local_hour`` (weekday and
weekend) and ``plug_probability`` fields set the home clock itself: when the
EV plugs in, and when it leaves.  Plug-out is departure (decision 0004 item
51), so the departure hour is both the unplug and the trip start, and an
archetype must override the fixture's clock or its pattern mostly reflects
the fixture's shared default night-time window.  No new model parameter is
introduced -- only existing fixture/behaviour fields, applied uniformly. All
archetype results here are illustrative/synthetic, not observed CNZ or fitted
behaviour -- see the loaded fixture and config files.

Usage (from the repo root, with the project environment active):

    PYTHONPATH=src uv run --frozen --no-sync python scripts/run_archetypes.py
    PYTHONPATH=src uv run --frozen --no-sync python scripts/run_archetypes.py --html out.html
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from axle_studio.model import assumptions
from axle_studio.model.assumptions import CNZ_CONTEXT, PUBLIC_CHARGING
from axle_studio.model.forecast import ForecastResult, run_forecast
from axle_studio.model.settings import CohortFixture, RunSettings
from axle_studio.model.summaries import PLUG_IN_EVENT_WORLD_LIMIT
from axle_studio.ui import style

# Colour-blind-safe qualitative palette (Okabe & Ito, 2008), not the shared
# ``style.SERIES_COLOURS`` tokens: those name normal/selected/difference
# *paths*, but this script's chart compares several archetypes side by side,
# all on the same "normal" (no-action) path, so archetype identity needs its
# own distinct colours (chart audit B8).
_ARCHETYPE_COLOURS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

_START_LOCAL_DATE = date(2026, 1, 12)

# The six per-cohort fields `run_forecast` already accepts as
# `trip_behaviour["by_cohort"][cohort_id]` (see model/forecast.py): whether
# and how far a cohort drives each day, and how widely its departure and
# plug-in clocks spread (truncated Student-t scales, decision 0004 item 51).
_TRIP_BEHAVIOUR_FIELDS = (
    "weekday_drive_probability",
    "weekend_drive_probability",
    "destination_dwell_minutes",
    "drive_speed_mph",
    "departure_scale_minutes",
    "plug_in_scale_minutes",
)

# Separately, `CohortSpec`'s arrival/departure hours (weekday and weekend)
# and `.plug_probability` set the home clock: when a session starts, when the
# EV leaves (which ends the session and starts any trip) and how often it
# plugs in. Every fixture cohort loads with the same default clock (mostly
# 18:00 plug-in, 07:00 departure), so an archetype that only varies
# `trip_behaviour` still shows everyone on the same schedule.
_CONNECTION_FIELDS = (
    "connection_departure_local_hour",
    "connection_arrival_local_hour",
    "connection_weekend_departure_local_hour",
    "connection_weekend_arrival_local_hour",
    "connection_plug_probability",
)

# The fixture's cohort named "always_plugged_in" is not an archetype lever:
# axle_studio.model.sampling._ALWAYS_PLUGGED_COHORT hard-codes that
# exact cohort_id to one connection session covering the whole run, ignoring
# arrival/departure/plug_probability entirely. Overriding its fields would
# have no effect, so `_archetype_fixture` below leaves this one cohort's row
# untouched for every archetype. It is a fixed 1%-of-fleet background
# presence common to all four archetypes, not part of any archetype's story
# -- which is also why no archetype below is named after it.
_ALWAYS_PLUGGED_FIXTURE_COHORT_ID = "always_plugged_in"

# Named illustrative archetype presets. "meaning" is a one-line description
# for the printed table, stated in terms of the actual hours/probabilities
# below it. The rest are values for the fields listed above: the first six
# for `trip_behaviour`, the last five for the cohort's home clock.
#
# The fixture's "intelligent_octopus" cohort (30% of the fleet) drives ~77
# miles/day on average -- the model derives each day's *conditional* distance
# as that mean divided by the day's drive probability, so a very low
# probability implies an implausibly long single trip for that cohort. Every
# archetype below therefore keeps both drive probabilities at 0.4 or above and
# pairs any large dwell time with a higher drive speed, so even a high-mileage,
# long-dwell day stays well inside 24 hours for every cohort.
#
# `sample_connection_opportunities` also requires each day's departure,
# moved by up to the shared clip (180 minutes), to come before that
# evening's plug-in, moved by up to the clip too unless its plug-in spread is
# 0. So a departure and plug-in more than 6 hours apart always fit, and a
# fixed plug-in (spread 0) only needs to be more than 3 hours after the
# departure; every clock below keeps that gap.
ARCHETYPES: dict[str, dict[str, object]] = {
    "weekday_commuter": {
        "meaning": (
            "Drives to work most weekdays, less at weekends; "
            "reconnects at home every evening (18:00-07:00, until 11:00 at weekends)."
        ),
        "weekday_drive_probability": 0.95,
        "weekend_drive_probability": 0.50,
        "destination_dwell_minutes": 240,
        "drive_speed_mph": 40,
        "departure_scale_minutes": 20,
        "plug_in_scale_minutes": 20,
        "connection_departure_local_hour": 7,
        "connection_arrival_local_hour": 18,
        "connection_weekend_departure_local_hour": 11,
        "connection_weekend_arrival_local_hour": 18,
        "connection_plug_probability": 1.0,
    },
    "weekend_long_trips": {
        "meaning": (
            "Drives less midweek; takes a long weekend trip and reconnects "
            "less reliably on a late return (20:00-08:00, 09:00 at weekends, 85% of nights)."
        ),
        "weekday_drive_probability": 0.50,
        "weekend_drive_probability": 0.95,
        "destination_dwell_minutes": 300,
        "drive_speed_mph": 45,
        "departure_scale_minutes": 30,
        "plug_in_scale_minutes": 60,
        "connection_departure_local_hour": 8,
        "connection_arrival_local_hour": 20,
        "connection_weekend_departure_local_hour": 9,
        "connection_weekend_arrival_local_hour": 20,
        "connection_plug_probability": 0.85,
    },
    "home_most_of_the_day": {
        # Named for the clock schedule below, not for actual occupancy: the
        # EV unplugs at its 09:00 departure (spread 20 minutes) and plugs in
        # at a fixed 13:00 every day, whether or not it drove that day
        # (only 65% of days), so this is not "plugs in whenever home" -- an
        # EV that stayed home all day is still unplugged for that window.
        "meaning": (
            "Drives a short errand on 65% of days; unplugged on a fixed "
            "clock schedule (09:00-13:00) every day regardless of whether "
            "it drove, reconnecting reliably the rest of the day "
            "(13:00-09:00, every time)."
        ),
        "weekday_drive_probability": 0.65,
        "weekend_drive_probability": 0.65,
        "destination_dwell_minutes": 120,
        "drive_speed_mph": 35,
        "departure_scale_minutes": 20,
        "plug_in_scale_minutes": 0,
        "connection_departure_local_hour": 9,
        "connection_arrival_local_hour": 13,
        "connection_weekend_departure_local_hour": 9,
        "connection_weekend_arrival_local_hour": 13,
        "connection_plug_probability": 1.0,
    },
    "shift_worker": {
        "meaning": (
            "Leaves early every day for a fixed-length shift; reconnects "
            "on return in the afternoon (17:00-06:00)."
        ),
        "weekday_drive_probability": 0.90,
        "weekend_drive_probability": 0.85,
        "destination_dwell_minutes": 480,
        "drive_speed_mph": 30,
        "departure_scale_minutes": 15,
        "plug_in_scale_minutes": 15,
        "connection_departure_local_hour": 6,
        "connection_arrival_local_hour": 17,
        "connection_weekend_departure_local_hour": 6,
        "connection_weekend_arrival_local_hour": 17,
        "connection_plug_probability": 1.0,
    },
}


def _archetype_trip_behaviour(name: str, fixture: CohortFixture) -> dict[str, object]:
    """Build a `trip_behaviour` mapping applying one archetype's driving pattern.

    Applied to every one of the fixture's six cohorts uniformly: it stands
    for "the whole illustrative fleet drives like this", not a real observed
    sub-population.
    """
    preset = ARCHETYPES[name]
    per_cohort_fields = {field: preset[field] for field in _TRIP_BEHAVIOUR_FIELDS}
    # The clock bound and tail weight come from model/assumptions.py so this
    # script does not scatter its own copy of those constants.
    clocks = assumptions.CONNECTION_CLOCKS
    return {
        "by_cohort": {cohort.cohort_id: dict(per_cohort_fields) for cohort in fixture.cohorts},
        "clock_clip_minutes": clocks["clock_clip_minutes"].value,
        "clock_t_df": clocks["clock_t_df"].value,
    }


def _archetype_fixture(name: str, base_fixture: CohortFixture) -> CohortFixture:
    """Return a copy of `base_fixture` with one archetype's connection window applied.

    Every cohort except `_ALWAYS_PLUGGED_FIXTURE_COHORT_ID` gets the
    archetype's weekday and weekend arrival/departure hours and
    `plug_probability` (see the module comment above `_CONNECTION_FIELDS` for
    why this is a second, necessary lever alongside `trip_behaviour`). No model file is
    edited: this builds a plain in-memory copy via `dataclasses.replace`,
    using only fields `CohortSpec` already defines.
    """
    preset = ARCHETYPES[name]
    cohorts = tuple(
        cohort
        if cohort.cohort_id == _ALWAYS_PLUGGED_FIXTURE_COHORT_ID
        else replace(
            cohort,
            departure_local_hour=preset["connection_departure_local_hour"],
            arrival_local_hour=preset["connection_arrival_local_hour"],
            weekend_departure_local_hour=preset["connection_weekend_departure_local_hour"],
            weekend_arrival_local_hour=preset["connection_weekend_arrival_local_hour"],
            plug_probability=preset["connection_plug_probability"],
        )
        for cohort in base_fixture.cohorts
    )
    return replace(base_fixture, cohorts=cohorts)


def _base_settings(*, vehicle_count: int, world_count: int, seed: int) -> RunSettings:
    """Settings shared by every archetype run: the app's defaults, with size and seed varied.

    Taken from ``model/assumptions.py`` so the script starts from the same
    settled state as the app (decision 0004 item 36).
    """
    return assumptions.run_settings(
        {"vehicle_count": vehicle_count, "evaluation_world_count": world_count, "seed": seed},
        _START_LOCAL_DATE,
    )


def run_archetype(
    name: str,
    fixture: CohortFixture,
    *,
    vehicle_count: int,
    world_count: int,
    seed: int,
) -> ForecastResult:
    """Run one named archetype preset through `run_forecast(..., model="no_action")`.

    One `np.random.default_rng(seed)` call happens inside `run_forecast`
    itself; every archetype reuses the same caller-supplied `seed` so runs
    stay simple, reproducible and comparable.
    """
    settings = _base_settings(vehicle_count=vehicle_count, world_count=world_count, seed=seed)
    # The connection window is a fixture-level property (CohortSpec), so the
    # archetype's copy of the fixture -- not the shared base fixture -- is
    # what gets run.
    archetype_fixture = _archetype_fixture(name, fixture)
    # Every other input is the app's default from model/assumptions.py;
    # decision 0004 item 32: both models top up at a public charger, so the
    # no-action run takes the same public assumptions as the app.
    inputs = assumptions.forecast_inputs(
        None, warmup_days=settings.warmup_days, study_days=settings.study_days
    )
    inputs["trip_behaviour"] = _archetype_trip_behaviour(name, archetype_fixture)
    return run_forecast(settings, archetype_fixture, **inputs, model="no_action")


def _kpi_p50(kpis: pd.DataFrame, *, day_type: str, metric: str) -> float:
    """Read one `plug_in_summary.kpis` cross-world median (contract 3.10, `PLUG_KPI_UNITS`)."""
    match = kpis.loc[kpis["day_type"].eq(day_type) & kpis["metric"].eq(metric), "p50"]
    return float(match.iat[0]) if len(match) else float("nan")


def _weekday_modal_plug_in_local_hour(result: ForecastResult) -> float:
    """Median across worlds of the fleet's weekday modal plug-in local hour.

    `plug_in_summary.kpis` only cross-world-summarises the four
    `PLUG_KPI_UNITS` metrics (`model/summaries.py`), not
    `modal_plug_in_local_hour`, so this reads the per-world value straight
    from `plug_in_world_kpis` (contract 3.10, `group_id="fleet"`) -- built
    from every world, not the plug-in-event sample -- and takes the median
    across worlds itself, matching how every other cross-world figure here
    is combined.
    """
    kpis = result.plug_in_world_kpis
    mask = (
        kpis["group_id"].eq("fleet")
        & kpis["day_type"].eq("weekday")
        & kpis["metric"].eq("modal_plug_in_local_hour")
    )
    values = kpis.loc[mask, "value"].dropna().to_numpy(dtype=float)
    return float(np.quantile(values, 0.5, method="linear")) if len(values) else float("nan")


def summarise_archetype(result: ForecastResult) -> dict[str, object]:
    """Summarise one result into population-level, illustrative statistics.

    Plug-in timing and energy come from `result.fleet_world_intervals`, the
    model's world-first fleet aggregate (one row per world/half-hour
    interval, warmup days already excluded, contract v2 names). Real
    per-plug-in statistics -- the population's actual SoC at connection, not
    a fleet-level SoC-trough stand-in -- come from `result.plug_in_summary`
    (contract 3.10, cross-world median at day type "all") and
    `result.plug_in_world_kpis` (the weekday modal hour, which
    `plug_in_summary` does not cross-world-summarise). The one exception is
    the share below 20% SoC, which has no built-in KPI (only the 10%
    threshold does): it is computed directly from `result.plug_in_events`,
    the sampled plug-in rows `build_summaries` keeps explicitly "for the Data
    expander and scripts" (`model/summaries.py` module docstring) -- exactly
    this use, not a chart statistic.

    Plug-in timing reads the already world-first-aggregated P10/P50/P90
    (`result.fleet_interval_bands`, contract 3.6) rather than taking its own
    mean across `fleet_world_intervals`: those quantiles are the same ones
    the rest of the app and the notebooks use for "spread across simulated
    weeks", so this script's HTML chart can show a named median line and a
    P10-P90 band at no extra simulation cost (chart audit B8).
    """
    rows = result.fleet_world_intervals
    rows = rows.loc[rows["path_id"].eq("normal")].sort_values(
        ["world_id", "interval_start_utc"], kind="stable"
    )

    bands = result.fleet_interval_bands
    bands = bands.loc[
        bands["metric"].eq("connected_share") & bands["path_id"].eq("normal")
    ].sort_values("interval_start_utc")
    band_index = pd.DatetimeIndex(bands["interval_start_utc"])
    # Median, across the simulated weeks, of the fleet's connected share at
    # each UTC half-hour of the study week. Its min/max give a compact read on
    # how much the archetype's plugged-in share swings over the week.
    weekly_plugged_in_pct = pd.Series(bands["p50"].to_numpy() * 100.0, index=band_index)
    weekly_plugged_in_pct_p10 = pd.Series(bands["p10"].to_numpy() * 100.0, index=band_index)
    weekly_plugged_in_pct_p90 = pd.Series(bands["p90"].to_numpy() * 100.0, index=band_index)

    # Energy: fleet-wide home import per world, per day, per EV, then a plain
    # mean of that per-world figure across worlds.
    study_days = result.settings_snapshot["study_days"]
    vehicle_count = result.vehicle_count
    per_world_home_kwh = rows.groupby("world_id")["home_import_kwh"].sum()
    mean_kwh_per_ev_per_day = float((per_world_home_kwh / study_days / vehicle_count).mean())

    # Public top-ups (decision 0004 items 32 and 37): a public charger is
    # always available and tops an EV up to the target SoC before any trip
    # would take it below the threshold (defaults 10% -> 80%), in both models.
    # The kernel exposes top-up energy, not a count, so this reports public
    # top-up energy per EV per week.
    per_world_public_kwh = rows.groupby("world_id")["public_import_kwh"].sum()
    public_topup_kwh_per_ev_per_week = float((per_world_public_kwh / vehicle_count).mean())

    kpis = result.plug_in_summary.kpis
    median_plug_in_soc_percent = _kpi_p50(kpis, day_type="all", metric="median_plug_in_soc_percent")
    plug_ins_per_ev_per_week = _kpi_p50(kpis, day_type="all", metric="plug_ins_per_ev_per_week")
    share_below_10_percent_soc = _kpi_p50(kpis, day_type="all", metric="share_below_10_percent_soc")

    # No built-in KPI reaches 20%, so this is computed from the plug-in-event
    # sample (see the docstring above) rather than from a full-world KPI --
    # unlike every other figure here, it is over at most
    # `PLUG_IN_EVENT_WORLD_LIMIT` sampled simulated weeks, not every world.
    events = result.plug_in_events
    share_below_20_percent_soc_sampled = (
        float((events["plug_in_soc_percent"] < 20.0).mean()) if len(events) else float("nan")
    )

    weekday_modal_plug_in_local_hour = _weekday_modal_plug_in_local_hour(result)

    return {
        "min_plugged_in_pct": float(weekly_plugged_in_pct.min()),
        "max_plugged_in_pct": float(weekly_plugged_in_pct.max()),
        "median_plug_in_soc_percent": median_plug_in_soc_percent,
        "plug_ins_per_ev_per_week": plug_ins_per_ev_per_week,
        "share_below_10_percent_soc": share_below_10_percent_soc,
        "share_below_20_percent_soc_sampled": share_below_20_percent_soc_sampled,
        "mean_kwh_per_ev_per_day": mean_kwh_per_ev_per_day,
        "public_topup_kwh_per_ev_per_week": public_topup_kwh_per_ev_per_week,
        "weekday_modal_plug_in_local_hour": weekday_modal_plug_in_local_hour,
        "weekly_plugged_in_pct": weekly_plugged_in_pct,
        "weekly_plugged_in_pct_p10": weekly_plugged_in_pct_p10,
        "weekly_plugged_in_pct_p90": weekly_plugged_in_pct_p90,
    }


_CNZ_CONTEXT_ROW_LABEL = "CNZ context (May 2022 report; not calibration)"
# Read the top-up rule from the assumptions so printed notes never go stale.
_THRESHOLD = float(PUBLIC_CHARGING["public_top_up_threshold_soc_percent"].value)
_TARGET = float(PUBLIC_CHARGING["public_top_up_target_soc_percent"].value)


def build_cnz_comparison_table(cnz_rows: list[dict[str, object]]) -> pd.DataFrame:
    """Archetype plug-in medians/shares/mode next to the CNZ context (`model/assumptions.py`).

    `CNZ_CONTEXT` states these figures are "comparison context only, never a
    calibration target": this recapitulates each archetype's own
    population-level plug-in statistics next to that context; it never fits
    the model to reproduce it. The 20%-SoC column is the CNZ report's upper
    bound ("fewer than 10% below 20% SoC"), not a point estimate, so it is
    only ever a ceiling an archetype figure should sit under, never a target
    to match exactly. The archetype side of that column
    (`share_below_20_percent_soc_sampled`) is a sampled statistic (at most
    `PLUG_IN_EVENT_WORLD_LIMIT` simulated weeks), unlike the other three
    columns, which use every simulated week.
    """
    context_row = {
        "archetype": _CNZ_CONTEXT_ROW_LABEL,
        "median_plug_in_soc_percent": CNZ_CONTEXT["cnz_median_plug_in_soc_percent"].value,
        "share_below_10_percent_soc": CNZ_CONTEXT["cnz_share_plug_ins_below_10_percent_soc"].value,
        "share_below_20_percent_soc_sampled": (
            CNZ_CONTEXT["cnz_share_plug_ins_below_20_percent_soc_upper_bound"].value
        ),
        "weekday_modal_plug_in_local_hour": (
            CNZ_CONTEXT["cnz_weekday_plug_in_mode_local_hour"].value
        ),
    }
    return pd.DataFrame([*cnz_rows, context_row])


def _build_weekly_curve_figure(weekly_curves: dict[str, pd.DataFrame]) -> go.Figure:
    """Build the plug-in-share-over-the-week figure: a median plus a P10-P90 band per archetype.

    `weekly_curves[name]` has "p10"/"p50"/"p90" columns (plug-in %, UTC-indexed),
    the world-first quantiles `summarise_archetype` already reads from
    `fleet_interval_bands` -- no extra simulation, so the band is cheap here
    (chart audit B8). The London-time x-axis and light, notebook-readable
    styling reuse the same `ui/style.py` tokens as the app and notebooks
    (`style.london_time_axis`, `style.apply_notebook_style`); colours are the
    script's own per-archetype palette (`_ARCHETYPE_COLOURS`), not the path
    colours, since this chart compares archetypes, not normal/selected paths.
    """
    figure = go.Figure()
    utc_index: pd.DatetimeIndex | None = None
    for (name, curve), colour in zip(weekly_curves.items(), _ARCHETYPE_COLOURS, strict=False):
        utc_index = pd.DatetimeIndex(curve.index)
        local_start = utc_index.tz_convert(style.DISPLAY_TIMEZONE)
        figure.add_trace(
            go.Scatter(
                x=local_start,
                y=curve["p10"].to_numpy(),
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
                legendgroup=name,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=local_start,
                y=curve["p90"].to_numpy(),
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=style.band_fill(colour),
                name=f"{name} P10-P90",
                legendgroup=name,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=local_start,
                y=curve["p50"].to_numpy(),
                mode="lines",
                line={"color": colour, "width": 2},
                name=f"{name} median",
                legendgroup=name,
            )
        )
    figure.update_layout(
        title=(
            "Illustrative plug-in share over the study week, by archetype (synthetic): "
            "median across simulated weeks, P10-P90 band"
        ),
        xaxis_title="London local time",
        yaxis_title="Fleet plugged in (%)",
        yaxis_range=[0, 100],
    )
    if utc_index is not None:
        figure.update_xaxes(**style.london_time_axis(pd.Series(utc_index)))
    return style.apply_notebook_style(figure, height=style.CHART_HEIGHTS["time_series"])


def _write_html_chart(weekly_curves: dict[str, pd.DataFrame], path: Path) -> None:
    """Write the weekly plug-in share chart (median + P10-P90 band per archetype) to HTML."""
    _build_weekly_curve_figure(weekly_curves).write_html(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--vehicles", type=int, default=200, help="EVs per archetype run (default: 200)"
    )
    parser.add_argument(
        "--worlds", type=int, default=20, help="Evaluation worlds per archetype run (default: 20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Shared illustrative RNG seed used for every archetype run (default: 2026)",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Optional path to write a Plotly HTML chart of plug-in %% over the week",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    fixture = assumptions.cohort_fixture()

    print(
        f"Illustrative archetype runs: {args.vehicles} EVs x {args.worlds} worlds, "
        f"seed={args.seed}. Synthetic presets, not observed CNZ behaviour.\n"
    )

    # Goal review item 22: the "meaning" column used to sit inside the first
    # table, wide enough (about 200 characters) to wrap badly in an
    # 80-column terminal. Printed here as its own list instead, so the table
    # below stays to its short numeric/name columns.
    print("Archetype presets:")
    for name, preset in ARCHETYPES.items():
        print(f"  - {name}: {preset['meaning']}")
    print()

    table_rows: list[dict[str, object]] = []
    cnz_rows: list[dict[str, object]] = []
    weekly_curves: dict[str, pd.DataFrame] = {}
    for name, preset in ARCHETYPES.items():
        result = run_archetype(
            name,
            fixture,
            vehicle_count=args.vehicles,
            world_count=args.worlds,
            seed=args.seed,
        )
        summary = summarise_archetype(result)
        weekly_curves[name] = pd.DataFrame(
            {
                "p10": summary.pop("weekly_plugged_in_pct_p10"),
                "p50": summary.pop("weekly_plugged_in_pct"),
                "p90": summary.pop("weekly_plugged_in_pct_p90"),
            }
        )
        table_rows.append(
            {
                "archetype": name,
                "min_plugged_in_pct": summary["min_plugged_in_pct"],
                "max_plugged_in_pct": summary["max_plugged_in_pct"],
                "median_plug_in_soc_percent": summary["median_plug_in_soc_percent"],
                "plug_ins_per_ev_per_week": summary["plug_ins_per_ev_per_week"],
                "share_below_20_percent_soc_sampled": summary["share_below_20_percent_soc_sampled"],
                "mean_home_kwh_per_ev_per_day": summary["mean_kwh_per_ev_per_day"],
                "public_topup_kwh_per_ev_per_week": summary["public_topup_kwh_per_ev_per_week"],
            }
        )
        cnz_rows.append(
            {
                "archetype": name,
                "median_plug_in_soc_percent": summary["median_plug_in_soc_percent"],
                "share_below_10_percent_soc": summary["share_below_10_percent_soc"],
                "share_below_20_percent_soc_sampled": summary["share_below_20_percent_soc_sampled"],
                "weekday_modal_plug_in_local_hour": summary["weekday_modal_plug_in_local_hour"],
            }
        )

    table = pd.DataFrame(table_rows)
    with pd.option_context("display.width", 120, "display.max_colwidth", 60):
        print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(
        "\nNote: public_topup_kwh_per_ev_per_week is real public top-up energy "
        f"(decision 0004 items 32 and 37: no stranding, top-up {_THRESHOLD:.0f}% -> "
        f"{_TARGET:.0f}% SoC from model/assumptions.py), not a count of top-up "
        "events -- the model does not expose an event count, only the energy each "
        "top-up moved."
    )
    print(
        f"Note: share_below_20_percent_soc_sampled is from at most "
        f"{PLUG_IN_EVENT_WORLD_LIMIT} sampled simulated weeks "
        f"(PLUG_IN_EVENT_WORLD_LIMIT); other columns use every week."
    )

    print(
        f"\nCNZ comparison (context, not calibration -- CNZ May 2022 report via "
        f"model/assumptions.py CNZ_CONTEXT; {_CNZ_CONTEXT_ROW_LABEL} in the last row):"
    )
    cnz_table = build_cnz_comparison_table(cnz_rows)
    with pd.option_context("display.width", 120, "display.max_colwidth", 60):
        print(cnz_table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(
        f"Top-ups at the {_THRESHOLD:.0f}% threshold (decision 0004 item 37) mean no "
        f"plug-in falls below {_THRESHOLD:.0f}% in the model; plug-ins between "
        f"{_THRESHOLD:.0f}% and 20% can occur."
    )

    if args.html is not None:
        _write_html_chart(weekly_curves, args.html)
        print(f"\nWrote illustrative plug-in chart to {args.html}")


if __name__ == "__main__":
    main()
