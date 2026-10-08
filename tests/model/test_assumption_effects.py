"""Model-side causal tests: each editable assumption moves its output the expected way (U14).

Every run goes through ``run_forecast_from_assumptions``, the path the Run
button uses, at the defaults apart from the one edited value and a small
fleet.  The seed is shared, so each comparison is on matched futures and the
difference is the edit's effect, not sampling noise.  Directions are the
physical ones stated in each record's ``affects`` text.
"""

from __future__ import annotations

from datetime import date
from functools import cache

import numpy as np
import pandas as pd
import pytest

from axle_studio.model import assumptions
from axle_studio.model.forecast import ForecastResult, run_forecast_from_assumptions

_START = date(2026, 1, 12)
_SMALL = {"vehicle_count": 60, "evaluation_world_count": 4}


@cache
def _run(model: str = "no_action", **edits: float) -> ForecastResult:
    return run_forecast_from_assumptions(_START, model=model, values=_SMALL | edits)


def _normal(result: ForecastResult, metric: str) -> float:
    """Study-week total of one fleet metric on the normal path, summed over worlds."""

    frame = result.fleet_world_intervals
    return float(frame.loc[frame["path_id"].eq("normal"), metric].sum())


def _peak_home_kw(result: ForecastResult) -> float:
    frame = result.fleet_world_intervals
    return float(frame.loc[frame["path_id"].eq("normal"), "home_import_kw"].max())


# Decision 0004 item 51: the editable clock scales, one per cohort and clock.
_CLOCK_SCALES = tuple(
    name
    for name, record in assumptions.CONNECTION_CLOCKS.items()
    if record.editable and name != "clock_t_df"
)


# Supplier contract v1 section 8: reporting records tested at the bottom of
# this file; each changes only its own supplier or CO2 frames.
_SUPPLIER_RECORDS = (
    "supplier.early_departure_risk_charge_gbp_per_mwh",
    "supplier.customer_reward_mode",
    "supplier.customer_reward_gbp_per_ev_per_month",
    "supplier.platform_fee_gbp_per_ev_per_month",
    "carbon.intensity_at_reference_gco2_per_kwh",
    "carbon.intensity_slope_gco2_per_kwh_per_gw",
    "carbon.intensity_floor_gco2_per_kwh",
    "carbon.intensity_cap_gco2_per_kwh",
)


def test_every_editable_record_has_a_causal_test_here() -> None:
    covered = {
        "sensitivity_fraction_per_c",
        "home_charging_power_kw",
        "home_charge_efficiency",
        "efficiency_fraction",
        "public_charge_gbp_per_kwh",
        "public_top_up_threshold_soc_percent",
        "public_top_up_target_soc_percent",
        "heating_gw_per_c",
        "wind_installed_gw",
        "solar_installed_gw",
        "surprise_reveal_hours",
        "da_id_premium_gbp_per_mwh",
        "mild_shock_rate_per_night",
        "big_shock_rate_per_week",
        "big_shock_median_gw",
        "shock_up_probability",
        "big_shock_known_share",
        "supply_reference_price_gbp_per_mwh",
        "daily_level_sd_gbp_per_mwh",
        "day_ahead_sd_gbp_per_mwh",
        "intraday_hourly_sd_gbp_per_mwh",
        "vehicle_count",
        "evaluation_world_count",
        "warmup_days",
        "seed",
        "departure_margin_hours",
        "timed_tariff_enabled",
        "timed_start_local_hour",
        "not_recovered_material_share_percent",
        *_CLOCK_SCALES,
        "clock_t_df",
        *assumptions.COHORT_SHARE_NAMES,
        *assumptions.ZONE_SHARE_NAMES,
        *_SUPPLIER_RECORDS,
        # Trading contract v1 §10.8 (lane J6): shared factors, manufacturers
        # and the commitment rule, made editable now that the Firm MW UI
        # reads them.
        "behaviour.plug_in_skip_median",
        "behaviour.plug_in_day_factor_sd",
        "behaviour.plug_in_week_factor_sd",
        "behaviour.holiday_bank_holiday_monday",
        "behaviour.holiday_half_term_week",
        *assumptions.MANUFACTURER_SHARE_NAMES,
        *assumptions.MANUFACTURER_RESPONSE_RATE_NAMES,
        *assumptions.MANUFACTURER_OUTAGE_NAMES,
        "trading.commitment_rule",
        # Intraday dispatch contract v1 §10 (K5): newly editable on the
        # Public & prices tab.
        "trading.intraday_dispatch",
        "trading.replan_threshold_gbp_per_mwh",
        "trading.day_ahead_commitment_share",
    }
    assert {record.name for record in assumptions.editable_records()} == covered


def test_lower_home_charging_power_never_raises_study_week_home_import() -> None:
    # Decision 0004 item 36: with a settled start (80 % opening SoC, a
    # warm-up of three days, seven since item 52) a slower home charger can
    # only move energy out of the study week's home import (to public
    # top-ups), never add to it.  With
    # the old 55 % start and one warm-up day, 7.0 -> 5.0 kW raised it,
    # because warm-up refill spilled into the week.  The check is on the
    # total over matched worlds; one world can move by tens of kWh either way
    # as charging shifts across the week's first and last nights.
    powers = (7.0, 5.0, 3.6, 2.2)
    home = []
    public = []
    for power in powers:
        result = run_forecast_from_assumptions(
            _START,
            model="no_action",
            values={
                "vehicle_count": 200,
                "evaluation_world_count": 10,
                "home_charging_power_kw": power,
            },
        )
        home.append(_normal(result, "home_import_kwh"))
        public.append(_normal(result, "public_import_kwh"))

    assert all(later <= earlier for earlier, later in zip(home, home[1:], strict=False))
    assert home[-1] < home[0]
    # The energy goes to public top-ups instead.
    assert public[-1] > public[0]


def test_home_charging_power_caps_the_evening_peak() -> None:
    slow = _run(home_charging_power_kw=2.0)
    assert _peak_home_kw(slow) < _peak_home_kw(_run())
    # No EV draws more than its home charging power in any half-hour.
    assert _peak_home_kw(slow) <= 2.0 * _SMALL["vehicle_count"] + 1e-9
    assert _normal(slow, "public_import_kwh") > _normal(_run(), "public_import_kwh")


def test_more_weather_sensitivity_means_more_travel_energy_and_import() -> None:
    cold = _run(sensitivity_fraction_per_c=0.03)
    assert _normal(cold, "home_import_kwh") > _normal(_run(), "home_import_kwh")
    assert _normal(cold, "public_import_kwh") > _normal(_run(), "public_import_kwh")


def test_lower_home_charge_efficiency_needs_more_home_import() -> None:
    lossy = _run(home_charge_efficiency=0.7)
    assert _normal(lossy, "home_import_kwh") > _normal(_run(), "home_import_kwh")


def test_lower_public_efficiency_needs_more_public_import_for_the_same_top_ups() -> None:
    lossy = _run(efficiency_fraction=0.6)
    assert _normal(lossy, "public_import_kwh") > _normal(_run(), "public_import_kwh")
    # Top-ups are decided on battery energy, so home charging is unchanged.
    assert _normal(lossy, "home_import_kwh") == pytest.approx(_normal(_run(), "home_import_kwh"))


def test_higher_top_up_threshold_moves_energy_from_home_to_public() -> None:
    eager = _run(public_top_up_threshold_soc_percent=40.0)
    assert _normal(eager, "public_import_kwh") > _normal(_run(), "public_import_kwh")
    assert _normal(eager, "home_import_kwh") < _normal(_run(), "home_import_kwh")


def _mean_top_up_slot_kwh(result: ForecastResult) -> float:
    """Mean fleet public import over the half-hours with any top-up (kWh)."""

    frame = result.fleet_world_intervals
    public = frame.loc[frame["path_id"].eq("normal"), "public_import_kwh"]
    return float(public[public > 0.0].mean())


def test_higher_top_up_target_means_bigger_top_ups() -> None:
    full = _run(public_top_up_target_soc_percent=95.0)
    # The record's effect: each top-up is bigger.
    assert _mean_top_up_slot_kwh(full) > _mean_top_up_slot_kwh(_run())
    # The study-week total also rises, but bigger top-ups are also rarer and
    # the seven-day warm-up (decision 0004 item 52) can hold one that would
    # otherwise land in the week, so at 60 EVs x 4 weeks the total is noise
    # (1,242 -> 1,225 kWh); over 200 EVs x 10 weeks it rises clearly.
    larger = {"vehicle_count": 200, "evaluation_world_count": 10}
    totals = [
        _normal(
            run_forecast_from_assumptions(
                _START,
                model="no_action",
                values=larger | {"public_top_up_target_soc_percent": target},
            ),
            "public_import_kwh",
        )
        for target in (80.0, 95.0)
    ]
    assert totals[1] > totals[0]


def test_warmup_days_change_the_simulated_history_not_the_study_horizon() -> None:
    # Decision 0004 item 52: the warm-up (default 7, editable 3-14) is
    # simulated before the study and never shown.  A shorter warm-up keeps
    # the same noon-to-noon study slots but draws different futures, since
    # every channel has one value per sampled date.
    short = _run(warmup_days=3)
    default = _run()
    assert (short.settings_snapshot["warmup_days"], default.settings_snapshot["warmup_days"]) == (
        3,
        7,
    )
    pd.testing.assert_frame_equal(short.study_slots, default.study_slots)
    assert short.horizon_start_utc == default.horizon_start_utc
    assert short.replay_state.evaluation.daily_temperature_c.shape[1] == 3 + 7 + 1
    assert not np.array_equal(
        short.fleet_world_intervals["home_import_kwh"],
        default.fleet_world_intervals["home_import_kwh"],
    )


def test_public_rate_scales_the_value_of_unrecovered_energy() -> None:
    # A slow home charger leaves some deferred energy unrecovered by the end of
    # the week, which the cost table values at the public rate (decision 0004
    # items 4 and 13).  Doubling the rate doubles that value.
    slow = {"home_charging_power_kw": 1.5, "vehicle_count": 100, "evaluation_world_count": 6}
    column = "illustrative_unrecovered_energy_value_gbp"
    base = run_forecast_from_assumptions(_START, model="action", values=slow)
    dear = run_forecast_from_assumptions(
        _START, model="action", values=slow | {"public_charge_gbp_per_kwh": 1.58}
    )

    assert base.cost_effect[column].sum() > 0.0
    np.testing.assert_allclose(dear.cost_effect[column], 2.0 * base.cost_effect[column])
    # The rate only values the outcome: the physics is unchanged.
    np.testing.assert_array_equal(
        dear.fleet_world_intervals["home_import_kwh"], base.fleet_world_intervals["home_import_kwh"]
    )


def _smart(result: ForecastResult, metric: str) -> float:
    summary = result.smart_charging_summary.set_index("metric")
    return float(summary.loc[metric, "mean"])


def test_larger_departure_margin_means_fewer_early_departures_and_dearer_charging() -> None:
    # Decision 0004 item 38: the margin shortens the smart-charging window, so
    # fewer plans are still running when the driver leaves, but the window
    # holds fewer cheap half-hours.
    cautious = _run("action", departure_margin_hours=3.0)
    base = _run("action")
    assert _smart(cautious, "early_departure_count") < _smart(base, "early_departure_count")
    assert _smart(cautious, "selected_average_price_gbp_per_mwh") > _smart(
        base, "selected_average_price_gbp_per_mwh"
    )
    # The margin only moves the selected path.
    np.testing.assert_array_equal(
        _normal(cautious, "home_import_kwh"), _normal(base, "home_import_kwh")
    )


def test_timed_tariff_switch_adds_a_barred_path_without_moving_normal_or_selected() -> None:
    # Decision 0007 follow-up (8 October 2026): on by default (the separate
    # ``timed_tariff_enabled`` switch, since a bounded number_input has no
    # reachable "unset" value); off, it changes nothing. Either way it adds
    # or removes only the optional "timed" path, barred from London noon
    # until the start hour, moving neither the normal nor the selected path.
    on = _run("action", timed_tariff_enabled=1, timed_start_local_hour=0.0)
    base = _run("action", timed_tariff_enabled=0)
    assert set(on.fleet_world_intervals["path_id"]) == {"normal", "selected", "timed"}
    assert set(base.fleet_world_intervals["path_id"]) == {"normal", "selected"}
    np.testing.assert_array_equal(_normal(on, "home_import_kwh"), _normal(base, "home_import_kwh"))
    on_frame, base_frame = on.fleet_world_intervals, base.fleet_world_intervals
    np.testing.assert_array_equal(
        on_frame.loc[on_frame["path_id"].eq("selected"), "home_import_kwh"].to_numpy(),
        base_frame.loc[base_frame["path_id"].eq("selected"), "home_import_kwh"].to_numpy(),
    )
    # The barred window (London noon to midnight, start hour 0) never imports.
    timed = on_frame.loc[on_frame["path_id"].eq("timed")]
    barred = timed["interval_start_london"].dt.hour >= 12
    assert barred.any() and (~barred).any()
    assert (timed.loc[barred, "home_import_kwh"] == 0.0).all()
    assert (timed.loc[~barred, "home_import_kwh"] > 0.0).any()


def test_material_share_only_changes_which_weeks_count_as_not_recovered() -> None:
    # Decision 0004 item 45: a reporting threshold.  At 0 % every week with
    # any shortfall counts; energy and cost do not move.
    strict = _run("action", not_recovered_material_share_percent=0.0)
    base = _run("action")
    cost = strict.cost_effect
    assert strict.not_recovered_world_count == int(cost["energy_not_recovered"].sum())
    assert strict.not_recovered_world_count >= base.not_recovered_world_count
    columns = ["unrecovered_kwh", "illustrative_selected_minus_normal_total_gbp"]
    np.testing.assert_array_equal(cost[columns], base.cost_effect[columns])


def _forecast_prices(result: ForecastResult) -> np.ndarray:
    return result.forecast_prices["wholesale_forecast_gbp_per_mwh"].to_numpy()


def _net_demand(result: ForecastResult) -> np.ndarray:
    return result.forecast_prices["system_net_demand_gw"].to_numpy()


def test_reference_price_shifts_the_whole_forecast_price_path() -> None:
    # The supply curve's gas-set reference price adds to every half-hour.
    higher = _run("action", supply_reference_price_gbp_per_mwh=108.0)
    # Exact except where a rare surprise's price would reach the auction cap.
    np.testing.assert_allclose(
        _forecast_prices(higher), _forecast_prices(_run("action")) + 30.0, atol=0.01
    )


def test_more_heating_per_degree_raises_net_demand_and_prices() -> None:
    # Decision 0004 item 53: the sampled temperatures (all below the 15.5 °C
    # threshold in the default autumn scenario here) add heating demand.
    base, colder = _run("action"), _run("action", heating_gw_per_c=1.5)
    assert (_net_demand(colder) >= _net_demand(base)).all()
    assert _forecast_prices(colder).mean() > _forecast_prices(base).mean() + 5.0


def test_more_installed_wind_lowers_net_demand_and_prices() -> None:
    base, windier = _run("action"), _run("action", wind_installed_gw=50.0)
    assert (_net_demand(windier) < _net_demand(base)).all()
    assert _forecast_prices(windier).mean() < _forecast_prices(base).mean() - 5.0


def test_daily_level_moves_whole_days_only() -> None:
    # The level adds the same amount to every half-hour of a London day.
    base, flat = _run("action"), _run("action", daily_level_sd_gbp_per_mwh=0.0)
    gap = base.forecast_prices.assign(gap=_forecast_prices(base) - _forecast_prices(flat))
    day = gap["interval_start_london"].dt.date
    per_day = gap.groupby(["world_id", day])["gap"]
    # Up to the tiny change the level makes to where the auction cap binds
    # in the expected surprise impact.
    assert (per_day.max() - per_day.min()).max() < 1e-3
    assert gap["gap"].abs().max() > 1.0


def test_intraday_update_sd_widens_the_gap_between_intraday_and_day_ahead() -> None:
    def spread(result: ForecastResult) -> float:
        gap = result.evaluation_prices[
            "evaluation_context_price_gbp_per_mwh"
        ].to_numpy() - _forecast_prices(result)
        return float(gap.std())

    wide = _run("action", intraday_hourly_sd_gbp_per_mwh=6.0)
    assert spread(wide) > spread(_run("action"))
    np.testing.assert_array_equal(_forecast_prices(wide), _forecast_prices(_run("action")))


def test_shock_rates_sizes_signs_and_notice_reach_the_shock_table() -> None:
    # Decision 0004 item 56.
    base = _run("action")
    none = _run("action", mild_shock_rate_per_night=0.0, big_shock_rate_per_week=0.0)
    assert none.price_shocks.empty and not base.price_shocks.empty
    more = _run("action", big_shock_rate_per_week=14.0)
    big = lambda result: result.price_shocks.query("shock_class == 'big'")  # noqa: E731
    assert len(big(more)) > len(big(base))
    larger = _run("action", big_shock_median_gw=8.0)
    assert big(larger)["size_gw"].median() > big(base)["size_gw"].median()
    assert _run("action", shock_up_probability=1.0).price_shocks["direction"].eq("up").all()
    known = _run("action", big_shock_known_share=1.0)
    assert big(known)["known_day_ahead"].all()
    assert not np.allclose(_net_demand(known), _net_demand(base))


def test_more_solar_lowers_midday_net_demand_only() -> None:
    # Decision 0004 item 58.
    base, sunnier = _run("action"), _run("action", solar_installed_gw=30.0)
    hour = base.forecast_prices["interval_start_london"].dt.hour.to_numpy()
    lower = _net_demand(sunnier) - _net_demand(base)
    assert (lower[(hour >= 10) & (hour < 14)] < 0).all()
    assert np.allclose(lower[(hour < 7) | (hour >= 17)], 0.0)


def test_da_id_premium_lowers_day_ahead_by_exactly_the_premium() -> None:
    # No free money: day-ahead is the expected intraday close less the premium.
    base, premium = _run("action"), _run("action", da_id_premium_gbp_per_mwh=5.0)
    np.testing.assert_allclose(_forecast_prices(premium), _forecast_prices(base) - 5.0, atol=0.01)
    np.testing.assert_array_equal(
        premium.evaluation_prices["evaluation_context_price_gbp_per_mwh"],
        base.evaluation_prices["evaluation_context_price_gbp_per_mwh"],
    )


def test_surprise_notice_moves_surprises_between_intraday_and_imbalance() -> None:
    # With no notice a surprise's first hour lands after gate closure: the
    # intraday close no longer sees it (imbalance still does, either way),
    # and the day-ahead expectation covers less of it.
    base, none = _run("action"), _run("action", surprise_reveal_hours=0.0)
    close = "evaluation_context_price_gbp_per_mwh"
    moved = none.evaluation_prices[close] - base.evaluation_prices[close]
    assert (moved.abs() > 0.01).any()
    np.testing.assert_allclose(
        none.evaluation_prices["imbalance_price_gbp_per_mwh"],
        base.evaluation_prices["imbalance_price_gbp_per_mwh"],
        atol=1e-6,
    )
    assert not np.allclose(_forecast_prices(none), _forecast_prices(base))


def test_day_ahead_noise_moves_the_cheapest_half_hour_and_smart_charging() -> None:
    # Decision 0004 item 48: with no half-hour noise the cheapest half-hour
    # of a day falls at few clock times (only known shocks move it); with the
    # default noise it moves much more, and the smart path follows it while
    # the normal path does not change.
    def cheapest_clock_times(result: ForecastResult) -> set[str]:
        # Per world and session night (noon to noon, decision 0004 item 52),
        # so every group is a whole night.
        prices = result.forecast_prices.merge(
            result.study_slots[["slot_index", "night_index"]], on="slot_index"
        )
        local = prices["interval_start_london"]
        cheapest = prices.groupby(["world_id", "night_index"])[
            "wholesale_forecast_gbp_per_mwh"
        ].idxmin()
        return set(local[cheapest].dt.strftime("%H:%M"))

    flat = _run("action", day_ahead_sd_gbp_per_mwh=0.0)
    noisy = _run("action")
    assert len(cheapest_clock_times(noisy)) > len(cheapest_clock_times(flat)) + 3

    def selected(result: ForecastResult) -> np.ndarray:
        frame = result.fleet_world_intervals
        return frame.loc[frame["path_id"].eq("selected"), "home_import_kwh"].to_numpy()

    assert not np.allclose(selected(flat), selected(noisy))
    assert _normal(flat, "home_import_kwh") == _normal(noisy, "home_import_kwh")


def test_run_size_and_seed_set_the_simulated_fleet_and_futures() -> None:
    bigger = _run(vehicle_count=80, evaluation_world_count=5)
    assert (bigger.vehicle_count, bigger.world_count) == (80, 5)
    assert bigger.fleet_world_intervals["unit_count"].max() == 80
    assert bigger.fleet_world_intervals["world_id"].nunique() == 5

    reseeded = _run(seed=7)
    assert reseeded.seed == 7
    assert _normal(reseeded, "home_import_kwh") != _normal(_run(), "home_import_kwh")


def test_editing_zone_shares_moves_only_zone_membership() -> None:
    # Trading contract v1 §3: the zone draw is the last population draw and
    # its size depends only on the fleet size, so moving 0.15 from zone_1 to
    # zone_4 changes which EVs are in which zone and nothing physical: the
    # fleet's normal path and every other EV trait are identical.
    default = _run()
    edited = _run(**{"zones.share.zone_1": 0.10, "zones.share.zone_4": 0.40})
    counts = edited.units["zone_id"].value_counts().reindex(assumptions.ZONE_IDS).tolist()
    assert counts == [6, 15, 15, 24]
    assert default.units["zone_id"].value_counts().tolist() == [15] * 4
    pd.testing.assert_frame_equal(
        default.fleet_world_intervals, edited.fleet_world_intervals, check_exact=True
    )
    assert (
        edited.zone_summary.loc[edited.zone_summary["zone_id"].eq("zone_4"), "ev_count"]
        .eq(24)
        .all()
    )


def test_editing_cohort_shares_moves_exactly_those_cohorts_counts() -> None:
    # Decision 0004 item 54: moving 5 points from Average UK to Always
    # plugged-in (both editable, sum still 100%) must change only those two
    # cohorts' counts, by the same exact-count allocation build_population
    # always uses (sampling._cohort_counts), and must not perturb the other
    # four cohorts, the run's world count, or the vehicle count itself.
    default_counts = _run().units["cohort_id"].value_counts().to_dict()
    edited_counts = (
        _run(
            **{
                "average_uk.population_share_percent": 35.0,
                "always_plugged_in.population_share_percent": 6.0,
            }
        )
        .units["cohort_id"]
        .value_counts()
        .to_dict()
    )

    assert default_counts == {
        "average_uk": 24,
        "intelligent_octopus": 18,
        "infrequent_charging": 6,
        "infrequent_driving": 6,
        "scheduled_charging": 5,
        "always_plugged_in": 1,
    }
    assert edited_counts == {
        "average_uk": 21,
        "intelligent_octopus": 18,
        "infrequent_charging": 6,
        "infrequent_driving": 6,
        "scheduled_charging": 5,
        "always_plugged_in": 4,
    }
    unchanged = {
        "intelligent_octopus",
        "infrequent_charging",
        "infrequent_driving",
        "scheduled_charging",
    }
    for cohort_id in unchanged:
        assert edited_counts[cohort_id] == default_counts[cohort_id]
    assert sum(edited_counts.values()) == sum(default_counts.values()) == _SMALL["vehicle_count"]


def test_weekend_connection_window_brings_weekend_plug_ins_earlier() -> None:
    # Decision 0004 item 42: Average UK and Intelligent Octopus arrive about
    # 17:00 at weekends and 18:00 on weekdays (CNZ adaptation), so more of
    # their weekend plug-ins start before 17:30.  Their 6 h weekend dwell
    # (follow-up to item 42) brings the 10:00 weekend trip home about 16:50,
    # so driving days plug in near the window's 17:00 opening too.  With
    # multi-day sessions (decision 0004 item 68) an EV that does not drive
    # on a weekend day is still plugged in from the night before, so most
    # weekend plug-ins follow a trip home and the gap is smaller (about
    # 0.04 at 300 EVs x 10 sampled weeks, against 0.12 before item 68).
    events = _run().plug_in_events
    events = events.loc[events["cohort_id"].isin(["average_uk", "intelligent_octopus"])]
    london = events["plug_in_london"]
    before = (london.dt.hour + london.dt.minute / 60.0) < 17.5
    weekday = before[events["day_type"].eq("weekday")].mean()
    weekend = before[events["day_type"].eq("weekend")].mean()
    assert weekend > weekday + 0.03


def _london_minutes(times: pd.Series) -> pd.Series:
    local = pd.to_datetime(times, utc=True).dt.tz_convert("Europe/London")
    return local.dt.hour * 60 + local.dt.minute


def _clock_times(result: ForecastResult, name: str) -> pd.Series:
    """London clock minutes of the events one clock-scale record moves.

    A departure scale moves plug-outs (plug-out is departure), read from the
    normal path's plug-in events.  A plug-in scale moves the sampled plug-in
    clock on its day type, read from the run's accepted sessions: with
    multi-day sessions (decision 0004 item 68) most realised weekend
    plug-ins follow a trip home, so their times mostly show the return,
    not the clock.  The record's cohort only.
    """

    field, cohort_id = name.split(".")
    if field == "departure_scale_minutes":
        events = result.plug_in_events
        events = events.loc[events["path_id"].eq("normal") & events["cohort_id"].eq(cohort_id)]
        return _london_minutes(events["plug_out_utc"].dropna())
    inputs = result.replay_state.evaluation.inputs
    cohort = (result.units["cohort_id"] == cohort_id).to_numpy()
    sessions = inputs["connection_session_accepted"] & cohort[np.newaxis, np.newaxis, :]
    starts = pd.Series(pd.to_datetime(inputs["connection_start_utc"][sessions], utc=True))
    weekend = starts.dt.tz_convert("Europe/London").dt.dayofweek >= 5
    return _london_minutes(starts[weekend == field.startswith("weekend")])


@pytest.mark.parametrize("name", _CLOCK_SCALES)
def test_a_larger_clock_scale_spreads_that_cohorts_clock_wider(name: str) -> None:
    # Decision 0004 item 51: at scale 0 the clock never moves, so its events
    # bunch at the cohort's usual time; the default scale spreads them.
    # The always-plugged cohort is 1 EV of 60 and rarely unplugs, so it needs
    # more worlds for two or more plug-outs.  The zone and control-group
    # draws joined the population (trading contract v1 §4.8) and moved the
    # worlds; 16 worlds give three plug-outs at seed 42 rather than relying
    # on a lucky one or two.
    worlds = {"evaluation_world_count": 16} if name.endswith("always_plugged_in") else {}
    still = _clock_times(_run(**{name: 0.0}, **worlds), name)
    spread = _clock_times(_run(**worlds), name)
    assert len(still) > 0 and len(spread) > 0
    assert spread.std() > still.std()


def test_fewer_t_degrees_of_freedom_give_more_very_early_or_late_departures() -> None:
    # Decision 0004 item 51: df 4 has fatter tails than df 100 (practically a
    # normal), so more plug-outs land 2.5 h or more from their cohort's time.
    def far_share(result: ForecastResult) -> float:
        events = result.plug_in_events
        events = events.loc[events["path_id"].eq("normal") & events["plug_out_utc"].notna()]
        minutes = _london_minutes(events["plug_out_utc"])
        usual = minutes.groupby([events["cohort_id"], events["day_type"]]).transform("median")
        return float(((minutes - usual).abs() >= 150).mean())

    assert far_share(_run()) > far_share(_run(clock_t_df=100.0))


# --- Supplier and carbon reporting records (supplier contract v1 section 8) ---


def _action(**edits: float) -> ForecastResult:
    return _run("action", **edits)


def _pnl(result: ForecastResult, metric: str) -> float:
    frame = result.supplier_pnl_summary
    row = frame.loc[frame["hedge_variant"].eq("profiled") & frame["metric"].eq(metric)]
    return float(row["mean"].iloc[0])


def _kernel_unchanged(base: ForecastResult, edited: ForecastResult) -> None:
    pd.testing.assert_frame_equal(base.fleet_world_intervals, edited.fleet_world_intervals)


def test_risk_charge_reaches_the_cost_curve_and_nothing_else() -> None:
    base = _action()
    edited = _action(**{"supplier.early_departure_risk_charge_gbp_per_mwh": 60.0})
    assert edited.flex_cost_curve["risk_charge_gbp_per_mwh"].eq(60.0).all()
    assert not base.flex_cost_curve["risk_charge_gbp_per_mwh"].eq(60.0).any()
    _kernel_unchanged(base, edited)


def test_flat_reward_mode_pays_the_reward_per_customer() -> None:
    base = _action()
    unset = _action(**{"supplier.customer_reward_mode": 1})
    paid = _action(
        **{
            "supplier.customer_reward_mode": 1,
            "supplier.customer_reward_gbp_per_ev_per_month": 5.0,
        }
    )
    # An unset flat reward is unavailable, never 0 (decision 0003).
    assert np.isnan(_pnl(unset, "customer_payment_per_customer_per_month_gbp"))
    assert _pnl(paid, "customer_payment_per_customer_per_month_gbp") == pytest.approx(-5.0)
    _kernel_unchanged(base, paid)


def test_platform_fee_lowers_the_after_fee_net_by_the_fee() -> None:
    base = _action()
    charged = _action(**{"supplier.platform_fee_gbp_per_ev_per_month": 2.0})
    assert np.isnan(_pnl(base, "net_gain_per_customer_per_month_gbp"))
    before = _pnl(charged, "net_gain_before_fee_per_customer_per_month_gbp")
    after = _pnl(charged, "net_gain_per_customer_per_month_gbp")
    assert after == pytest.approx(before - 2.0)
    _kernel_unchanged(base, charged)


def _co2_unmanaged(result: ForecastResult) -> float:
    return float(result.carbon_shift_world["co2_unmanaged_kg"].sum())


@pytest.mark.parametrize(
    ("name", "value", "direction"),
    [
        ("carbon.intensity_at_reference_gco2_per_kwh", 360.0, 1),
        ("carbon.intensity_floor_gco2_per_kwh", 300.0, 1),
        ("carbon.intensity_cap_gco2_per_kwh", 100.0, -1),
    ],
)
def test_carbon_intensity_records_move_unmanaged_co2_the_stated_way(
    name: str, value: float, direction: int
) -> None:
    base = _action()
    edited = _action(**{name: value})
    change = _co2_unmanaged(edited) - _co2_unmanaged(base)
    assert direction * change > 0
    _kernel_unchanged(base, edited)


def test_zero_carbon_slope_changes_the_co2_frames_only() -> None:
    base = _action()
    flat = _action(**{"carbon.intensity_slope_gco2_per_kwh_per_gw": 0.0})
    assert not np.allclose(
        base.carbon_shift_world["co2_unmanaged_kg"], flat.carbon_shift_world["co2_unmanaged_kg"]
    )
    _kernel_unchanged(base, flat)


# --- Firm MW: shared factors, manufacturers, the commitment rule (trading
# contract v1 §10, §10.8, lane J6) ------------------------------------------
#
# The skip-share, holiday, manufacturer-share, response-rate and outage
# records all change who plugs in or whose plan is followed, so (unlike the
# reporting-only records above) the kernel path itself moves; there is no
# ``_kernel_unchanged`` check for those. Only the commitment rule (§10.4) is
# a trading-layer choice that leaves the kernel untouched.


def _skip_share(result: ForecastResult) -> pd.Series:
    return result.world_nights["plug_in_skip_share"]


def test_higher_skip_median_raises_the_mean_skip_share() -> None:
    base = _action()
    higher = _action(**{"behaviour.plug_in_skip_median": 0.3})
    assert _skip_share(higher).mean() > _skip_share(base).mean()


def test_higher_day_factor_sd_widens_the_night_to_night_skip_spread() -> None:
    base = _action()
    wider = _action(**{"behaviour.plug_in_day_factor_sd": 1.5})
    assert _skip_share(wider).std() > _skip_share(base).std()


def test_higher_week_factor_sd_widens_the_week_to_week_skip_spread() -> None:
    def week_mean_spread(result: ForecastResult) -> float:
        # One mean per world, then the spread across worlds: the week
        # factor moves a whole simulated week's skip share together
        # (§10.1b), which the day factor's night-to-night spread above does
        # not isolate.
        return float(result.world_nights.groupby("world_id")["plug_in_skip_share"].mean().std())

    base = _action()
    wider = _action(**{"behaviour.plug_in_week_factor_sd": 1.0})
    assert week_mean_spread(wider) > week_mean_spread(base)


def test_bank_holiday_monday_shifts_the_skip_logit_on_the_first_monday_evening_only() -> None:
    # _START (2026-01-12) is a Monday, so night 0's evening is the date the
    # switch shifts (§10.1c); every other night is untouched.
    edited = _action(**{"behaviour.holiday_bank_holiday_monday": 1})
    nights = edited.world_nights
    shifted = nights.loc[nights["night_index"].eq(0), "skip_logit_shift"]
    other = nights.loc[nights["night_index"].ne(0), "skip_logit_shift"]
    # A plain sum (0.0 + 0.30), exact in float64; pytest.approx compared
    # against a whole pandas Series does not reduce to one bool reliably.
    assert (shifted.to_numpy() == assumptions.HOLIDAY_SKIP_LOGIT_SHIFT).all()
    assert (other.to_numpy() == 0.0).all()
    assert (_action().world_nights["skip_logit_shift"].to_numpy() == 0.0).all()


def test_half_term_week_shifts_the_skip_logit_every_study_evening() -> None:
    edited = _action(**{"behaviour.holiday_half_term_week": 1})
    shift = edited.world_nights["skip_logit_shift"].to_numpy()
    assert (shift == assumptions.HOLIDAY_SKIP_LOGIT_SHIFT).all()


def test_manufacturer_shares_move_ev_counts_between_makers() -> None:
    base = _action()
    skewed = _action(
        **{
            "manufacturers.share.m1": 0.7,
            "manufacturers.share.m2": 0.1,
            "manufacturers.share.m3": 0.1,
            "manufacturers.share.m4": 0.1,
        }
    )
    base_counts = base.units["manufacturer_id"].value_counts()
    skewed_counts = skewed.units["manufacturer_id"].value_counts()
    assert skewed_counts["m1"] > base_counts["m1"]
    assert skewed_counts["m2"] < base_counts["m2"]


def _firmness_p50(result: ForecastResult, maker_id: str) -> float:
    firmness = result.firmness_by_manufacturer
    row = firmness.loc[
        firmness["manufacturer_id"].eq(maker_id)
        & firmness["direction"].eq("turn_down")
        & firmness["duration_hours"].eq(1.0)
    ]
    return float(row["firmness_p50"].iloc[0])


def test_lower_response_rate_lowers_that_makers_firmness_only() -> None:
    # Firmness is delivered ÷ deliverable-if-every-session-had-followed
    # (§10.5g); a maker's own response draw only touches its own sessions.
    base = _action()
    lower = _action(**{"manufacturers.response_rate.m1": 0.5})
    assert _firmness_p50(lower, "m1") < _firmness_p50(base, "m1")
    assert _firmness_p50(lower, "m2") == pytest.approx(_firmness_p50(base, "m2"))


def _m1_turn_down_1h_realised_mwh(result: ForecastResult) -> float:
    frame = result.availability_manufacturer_world
    rows = frame.loc[
        frame["manufacturer_id"].eq("m1")
        & frame["direction"].eq("turn_down")
        & frame["duration_hours"].eq(1.0)
    ]
    return float(rows["realised_mwh"].sum())


def test_certain_outage_collapses_that_makers_realised_energy() -> None:
    # A maker's whole slice ignores its plan on an outage night (§10.1d); the
    # week total is not asserted at exactly 0 (a session can span two
    # nights and keep its own night's status, overnight review log, 30 Sep),
    # only collapsed relative to the base run.
    base = _action()
    out = _action(**{"manufacturers.outage_probability_per_night.m1": 1.0})
    assert out.world_nights["outage_m1"].all()
    assert not out.world_nights["outage_m2"].all()
    assert _m1_turn_down_1h_realised_mwh(out) < 0.05 * _m1_turn_down_1h_realised_mwh(base)


def test_newsvendor_commitment_fills_commit_level_and_leaves_the_kernel_unchanged() -> None:
    fixed = _action()
    newsvendor = _action(**{"trading.commitment_rule": "newsvendor"})
    assert fixed.deviation_world_slot["commit_level"].isna().all()
    assert newsvendor.deviation_world_slot["commit_level"].notna().any()
    assert newsvendor.deviation_world_slot["forecast_error_quantile_kwh"].notna().any()
    # §10.4: a trading-layer risk choice on the position, not the physics.
    _kernel_unchanged(fixed, newsvendor)


# --- Intraday dispatch (contract v1 §9, §10; K5 made these three editable) ---


def test_intraday_dispatch_switch_turns_the_dispatch_frames_on_and_off() -> None:
    # Default is on (lead decision Q1); off drops the third kernel pass, so
    # every dispatch frame is None and no EV is locked (contract §7, §8).
    on = _action()
    off = _action(**{"trading.intraday_dispatch": 0})
    assert on.dispatch_world_slot is not None
    assert on.units["dispatch_locked"].any()
    assert off.dispatch_world_slot is None
    assert off.dispatch_bands is None
    assert off.dispatch_split is None
    assert not off.units["dispatch_locked"].any()


def test_lower_replan_threshold_raises_the_replan_count() -> None:
    # §4.2 worked example: theta = 0 re-plans on any strictly cheaper plan;
    # a very large theta (near the 500 GBP/MWh bound) leaves free EVs on
    # their day-ahead plan, since no ordinary intraday move clears the bar.
    lenient = _action(**{"trading.replan_threshold_gbp_per_mwh": 0.0})
    strict = _action(**{"trading.replan_threshold_gbp_per_mwh": 500.0})
    lenient_replans = int(lenient.dispatch_world_slot["replan_count"].sum())
    strict_replans = int(strict.dispatch_world_slot["replan_count"].sum())
    assert lenient_replans > strict_replans
    assert strict_replans == 0


def test_commitment_share_sets_the_locked_ev_count_and_c_equals_one_matches_day_ahead() -> None:
    # §2: K = round-half-up(c x N) EVs locked; c=1 locks every EV, so the
    # dispatched path equals the day-ahead plan path exactly (no free EV to
    # re-plan); c=0 frees every EV.
    full_lock = _action(**{"trading.day_ahead_commitment_share": 1.0})
    none_locked = _action(**{"trading.day_ahead_commitment_share": 0.0})
    assert full_lock.units["dispatch_locked"].all()
    assert not none_locked.units["dispatch_locked"].any()
    world_slot = full_lock.dispatch_world_slot
    assert (world_slot["dispatched_kwh"] == world_slot["day_ahead_plan_kwh"]).all()
    assert (world_slot["replan_count"] == 0).all()
