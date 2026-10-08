"""Smoke test for scripts/run_archetypes.py: two presets at a tiny run size.

The script lives outside the `axle_studio` package, so it is loaded directly
from its file path rather than imported by module name.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_archetypes.py"
_VEHICLE_COUNT = 60
_WORLD_COUNT = 8
_SEED = 11


def _load_run_archetypes():
    spec = importlib.util.spec_from_file_location("run_archetypes", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ra = _load_run_archetypes()


@pytest.fixture(scope="module")
def commuter_result():
    """One tiny `weekday_commuter` run, shared by every test that needs it."""
    fixture = ra.assumptions.cohort_fixture()
    return ra.run_archetype(
        "weekday_commuter",
        fixture,
        vehicle_count=_VEHICLE_COUNT,
        world_count=_WORLD_COUNT,
        seed=_SEED,
    )


@pytest.fixture(scope="module")
def home_most_of_the_day_result():
    """One tiny `home_most_of_the_day` run, shared by every test that needs it."""
    fixture = ra.assumptions.cohort_fixture()
    return ra.run_archetype(
        "home_most_of_the_day",
        fixture,
        vehicle_count=_VEHICLE_COUNT,
        world_count=_WORLD_COUNT,
        seed=_SEED,
    )


def _mean_in_window(curve: pd.Series, *, start_hour: int, end_hour: int) -> float:
    """Mean plugged-in % over a UTC hour-of-day window, across the whole study week."""
    frame = curve.rename("pct").reset_index()
    frame.columns = ["interval_start_utc", "pct"]
    hour = frame["interval_start_utc"].dt.hour
    return float(frame.loc[(hour >= start_hour) & (hour < end_hour), "pct"].mean())


def _daytime_mean_by_daytype(curve: pd.Series, *, start_hour: int = 9, end_hour: int = 15):
    """Mean plugged-in % during a shared midday window, split weekday vs weekend.

    weekday_commuter leaves at 07:00 on weekdays and 11:00 at weekends (the
    departure is also the unplug, decision 0004 item 51), so this UTC window
    is where the archetype's day-type contrast should be visible.
    """
    frame = curve.rename("pct").reset_index()
    frame.columns = ["interval_start_utc", "pct"]
    hour = frame["interval_start_utc"].dt.hour
    in_window = (hour >= start_hour) & (hour < end_hour)
    is_weekday = frame["interval_start_utc"].dt.dayofweek < 5
    weekday_mean = frame.loc[in_window & is_weekday, "pct"].mean()
    weekend_mean = frame.loc[in_window & ~is_weekday, "pct"].mean()
    return float(weekday_mean), float(weekend_mean)


def _assert_finite_summary(summary: dict[str, object]) -> None:
    for key in (
        "min_plugged_in_pct",
        "max_plugged_in_pct",
        "median_plug_in_soc_percent",
        "plug_ins_per_ev_per_week",
        "share_below_10_percent_soc",
        "share_below_20_percent_soc_sampled",
        "mean_kwh_per_ev_per_day",
        "public_topup_kwh_per_ev_per_week",
        "weekday_modal_plug_in_local_hour",
    ):
        assert math.isfinite(summary[key]), key
    assert np.isfinite(summary["weekly_plugged_in_pct"].to_numpy()).all()


def test_weekday_commuter_plugged_in_share_differs_weekday_vs_weekend(commuter_result) -> None:
    summary = ra.summarise_archetype(commuter_result)
    weekday_mean, weekend_mean = _daytime_mean_by_daytype(summary["weekly_plugged_in_pct"])

    # weekday_commuter drives to work most weekdays (95%) but far less at
    # weekends (50%) -- both trip_behaviour fields this script sets -- so more
    # of the fleet should be away, and unplugged, during weekday midday hours
    # than during the same UTC window at weekends.
    assert weekend_mean > weekday_mean
    _assert_finite_summary(summary)


def test_home_most_of_the_day_has_far_higher_daytime_connected_share(
    commuter_result, home_most_of_the_day_result
) -> None:
    """A real, connection-window-driven difference, not the special-cohort artefact.

    Both archetypes' populations include the same fixed 1% of EVs on the
    fixture's "always_plugged_in" cohort id, which `sampling`
    hard-codes to a permanent connection regardless of any archetype field
    (see `_ALWAYS_PLUGGED_FIXTURE_COHORT_ID` in the script). That shared 1%
    can only shift both curves by the same small, near-identical amount, so
    it cannot explain a large gap between the two archetypes: any large gap
    must come from the other 99% of the population, i.e. from
    `home_most_of_the_day`'s own, much wider connection window (13:00-09:00,
    vs weekday_commuter's 18:00-07:00) applied to those cohorts. Daytime
    (09:00-17:00 UTC) isolates this: weekday_commuter's cohorts are almost
    all working or driving then, while home_most_of_the_day's cohorts are
    connected for all but a four-hour midday errand.
    """
    commuter_summary = ra.summarise_archetype(commuter_result)
    home_most_of_the_day_summary = ra.summarise_archetype(home_most_of_the_day_result)
    commuter_daytime = _mean_in_window(
        commuter_summary["weekly_plugged_in_pct"], start_hour=9, end_hour=17
    )
    home_most_of_the_day_daytime = _mean_in_window(
        home_most_of_the_day_summary["weekly_plugged_in_pct"], start_hour=9, end_hour=17
    )

    # A one-percentage-point shared-cohort effect could not produce a gap this
    # large (empirically tens of percentage points); a small fixed margin is
    # enough to make the assertion meaningful without being seed-fragile.
    assert home_most_of_the_day_daytime > commuter_daytime + 10.0

    _assert_finite_summary(commuter_summary)
    _assert_finite_summary(home_most_of_the_day_summary)


def test_cnz_comparison_table_has_context_row_and_finite_archetype_values(
    commuter_result, home_most_of_the_day_result
) -> None:
    """The CNZ section (task item 3) recapitulates each archetype next to real CNZ context."""
    cnz_rows = []
    for name, result in (
        ("weekday_commuter", commuter_result),
        ("home_most_of_the_day", home_most_of_the_day_result),
    ):
        summary = ra.summarise_archetype(result)
        cnz_rows.append(
            {
                "archetype": name,
                "median_plug_in_soc_percent": summary["median_plug_in_soc_percent"],
                "share_below_10_percent_soc": summary["share_below_10_percent_soc"],
                "share_below_20_percent_soc_sampled": summary["share_below_20_percent_soc_sampled"],
                "weekday_modal_plug_in_local_hour": summary["weekday_modal_plug_in_local_hour"],
            }
        )

    table = ra.build_cnz_comparison_table(cnz_rows)

    assert list(table["archetype"]) == [
        "weekday_commuter",
        "home_most_of_the_day",
        ra._CNZ_CONTEXT_ROW_LABEL,
    ]
    # The context row's figures are the literal CNZ_CONTEXT values, not derived.
    context = table.loc[table["archetype"].eq(ra._CNZ_CONTEXT_ROW_LABEL)].iloc[0]
    assert context["median_plug_in_soc_percent"] == pytest.approx(52.0)
    assert context["share_below_10_percent_soc"] == pytest.approx(0.03)
    assert context["share_below_20_percent_soc_sampled"] == pytest.approx(0.10)
    assert context["weekday_modal_plug_in_local_hour"] == pytest.approx(18)

    archetype_rows = table.loc[table["archetype"].ne(ra._CNZ_CONTEXT_ROW_LABEL)]
    numeric_columns = [
        "median_plug_in_soc_percent",
        "share_below_10_percent_soc",
        "share_below_20_percent_soc_sampled",
        "weekday_modal_plug_in_local_hour",
    ]
    assert np.isfinite(archetype_rows[numeric_columns].to_numpy(dtype=float)).all()


def test_weekly_plugged_in_band_is_ordered_and_matches_the_named_median(commuter_result) -> None:
    """Chart audit B8: the HTML chart's band must be a real P10-P90 spread, not a duplicate line."""
    summary = ra.summarise_archetype(commuter_result)
    p10 = summary["weekly_plugged_in_pct_p10"]
    p50 = summary["weekly_plugged_in_pct"]
    p90 = summary["weekly_plugged_in_pct_p90"]

    assert p10.index.equals(p50.index)
    assert p90.index.equals(p50.index)
    assert (p10.to_numpy() <= p50.to_numpy() + 1e-9).all()
    assert (p50.to_numpy() <= p90.to_numpy() + 1e-9).all()
    assert np.isfinite(p10.to_numpy()).all()
    assert np.isfinite(p90.to_numpy()).all()


def test_build_weekly_curve_figure_names_the_median_and_uses_the_light_notebook_style(
    commuter_result, home_most_of_the_day_result
) -> None:
    """Chart audit B8: London-time axis, a named median line, a P10-P90 band, light template."""
    weekly_curves = {}
    for name, result in (
        ("weekday_commuter", commuter_result),
        ("home_most_of_the_day", home_most_of_the_day_result),
    ):
        summary = ra.summarise_archetype(result)
        weekly_curves[name] = pd.DataFrame(
            {
                "p10": summary["weekly_plugged_in_pct_p10"],
                "p50": summary["weekly_plugged_in_pct"],
                "p90": summary["weekly_plugged_in_pct_p90"],
            }
        )

    figure = ra._build_weekly_curve_figure(weekly_curves)

    # Three traces per archetype: the invisible P10 line, the filled P90 band, the named P50 line.
    assert len(figure.data) == 3 * len(weekly_curves)
    median_trace_names = [
        trace.name for trace in figure.data if trace.mode == "lines" and trace.line.width
    ]
    assert median_trace_names == ["weekday_commuter median", "home_most_of_the_day median"]
    band_trace_names = [trace.name for trace in figure.data if trace.fill == "tonexty"]
    assert band_trace_names == ["weekday_commuter P10-P90", "home_most_of_the_day P10-P90"]
    assert "median" in figure.layout.title.text
    assert figure.layout.xaxis.title.text == "London local time"
    # London-time ticks (style.london_time_axis), not raw UTC.
    assert any(
        label.split(" ")[0] in {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
        for label in figure.layout.xaxis.ticktext
    )
    # The light notebook template (chart audit B8), not the dark app template.
    assert figure.layout.template.layout.paper_bgcolor == "#FFFFFF"
    assert figure.layout.height == ra.style.CHART_HEIGHTS["time_series"]


def test_main_prints_meaning_as_a_list_not_a_wide_table_column(capsys) -> None:
    # Goal review item 22: the printed first table used to include a
    # "meaning" column (about 200 characters per cell) that wrapped badly in
    # an 80-column terminal; it is now a separate list above the table, and
    # the table itself carries none of that free text.
    ra.main(["--vehicles", "6", "--worlds", "2", "--seed", str(_SEED)])

    out = capsys.readouterr().out
    assert "Archetype presets:" in out
    assert "meaning" not in out
    for name, preset in ra.ARCHETYPES.items():
        assert f"  - {name}: {preset['meaning']}" in out
