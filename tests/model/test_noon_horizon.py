"""Noon-to-noon study, seven-day warm-up with prices and publication-time prices.

Decision 0004 item 52 (plan A4, A5) and plan B4 (item 53): the study runs
from London 12:00 for seven session nights after a warm-up that has prices
and a smart last night, and a smart plan ranks only day-ahead prices already
published when it plans.  Every run here is SYNTHETIC model output.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from axle_studio.model import assumptions
from axle_studio.model.action import (
    day_ahead_publication_utc_ns,
    smart_charging_inputs,
    visible_day_ahead_prices,
)
from axle_studio.model.clock import utc_half_hour_boundaries
from axle_studio.model.forecast import run_forecast_from_assumptions, simulate_forecast
from axle_studio.model.physics import simulate_fleet_intervals
from axle_studio.model.summaries import cheapest_half_hour_summary, price_relative_bands

LONDON = "Europe/London"


def _simulated(start: date, values: dict, model: str = "action"):
    resolved = assumptions.resolve_values(values)
    settings = assumptions.run_settings(resolved, start)
    return simulate_forecast(
        settings,
        assumptions.cohort_fixture(resolved),
        **assumptions.forecast_inputs(
            resolved, warmup_days=settings.warmup_days, study_days=settings.study_days
        ),
        model=model,
    )


# --------------------------------------------------------------------------
# Horizon and session nights (plan A4)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "last_london"),
    [
        (date(2026, 1, 12), "2026-01-19 11:30"),
        # Spring-forward on Sunday 29 March 2026: 336 UTC slots end at 13:00.
        (date(2026, 3, 23), "2026-03-30 12:30"),
        # Fall-back on Sunday 25 October 2026: they end at 11:00.
        (date(2026, 10, 19), "2026-10-26 10:30"),
    ],
)
def test_study_is_336_slots_from_london_noon_over_seven_nights(start, last_london) -> None:
    result = run_forecast_from_assumptions(
        start, model="no_action", values={"vehicle_count": 12, "evaluation_world_count": 2}
    )
    slots = result.study_slots
    london = slots["interval_start_london"]
    assert result.study_slot_count == len(slots) == 336
    assert london.iat[0] == pd.Timestamp(f"{start} 12:00", tz=LONDON)
    assert london.iat[-1] == pd.Timestamp(last_london, tz=LONDON)
    assert sorted(set(slots["night_index"])) == list(range(7))
    assert result.settings_snapshot["warmup_days"] == 7


def test_every_clocked_session_is_inside_the_study_or_outside_it() -> None:
    # With a Monday start the study's two boundaries are Monday noon, after
    # every weekday departure (latest 09:00 + 3 h) and before every plug-in
    # (earliest 15:00 - 3 h), so a clocked home session crosses a boundary
    # only when the EV does not drive that date and stays plugged in
    # (decision 0004 item 68's multi-day sessions).  Always-plugged EVs,
    # whose one session spans the whole run, are at the edges too.
    simulated = _simulated(date(2026, 1, 12), {"vehicle_count": 200, "evaluation_world_count": 4})
    inputs = simulated.evaluation.inputs
    accepted = inputs["connection_session_accepted"]
    start = inputs["connection_start_utc"].astype("datetime64[ns]")
    end = inputs["connection_end_utc"].astype("datetime64[ns]")
    horizon = simulated.interval_start_utc
    first = horizon[0].tz_convert(None).to_datetime64()
    last = (horizon[-1] + pd.Timedelta(minutes=30)).tz_convert(None).to_datetime64()
    clocked = (simulated.units["cohort_id"] != "always_plugged_in").to_numpy()
    sessions = accepted & clocked[np.newaxis, np.newaxis, :]
    drives = inputs["drives_today"]
    # Sampled date of each boundary's noon: the study start date, and the
    # morning after the last night (the last sampled date).
    boundary_dates = (simulated.settings.warmup_days, drives.shape[1] - 1)
    for boundary, day in zip((first, last), boundary_dates):
        crosses = (sessions & (start < boundary) & (end > boundary)).any(axis=1)
        assert not (crosses & drives[:, day, :]).any()
        assert crosses.any()  # some non-driving EV stays plugged in across it
    inside = sessions & (start >= first) & (end <= last)
    assert inside.sum() > 0


def test_first_study_night_is_a_smart_charging_night() -> None:
    # Behaviour review: with the midnight start the first night imported about
    # 93 kW on the smart path between 00:00 and 06:00 against about 600 kW on
    # other weeknights, because plans started at the first study slot.  With
    # the noon start and smart planning on the last warm-up night, Monday
    # night (00:00-06:00 Tuesday) is within 20% of the other weeknights.
    result = run_forecast_from_assumptions(
        date(2026, 10, 12), values={"vehicle_count": 300, "evaluation_world_count": 10}
    )
    fleet = result.fleet_world_intervals
    selected = fleet.loc[fleet["path_id"].eq("selected")]
    london = selected["interval_start_london"]
    small_hours = selected.loc[london.dt.hour < 6]
    by_night = (
        small_hours.merge(result.study_slots[["slot_index", "night_index"]], on="slot_index")
        .groupby(["night_index", "world_id"])["home_import_kw"]
        .mean()
        .groupby("night_index")
        .mean()
    )
    first = by_night.loc[0]
    other_weeknights = by_night.loc[[1, 2, 3]].mean()
    # With the net-demand prices the cheap half-hours spread wider than
    # 00:00-06:00 and vary by week, so this ratio is loose; the evening-peak
    # share check below is the steadier one.
    assert first == pytest.approx(other_weeknights, rel=0.35)
    assert first > 300.0
    # A smart night moves the evening peak away: the share of the night's
    # smart import in 17:00-21:00 London is as small on the first night as
    # on the others (an unmanaged night puts about 60% there).
    nights = selected.merge(result.study_slots[["slot_index", "night_index"]], on="slot_index")
    hour = nights["interval_start_london"].dt.hour
    evening = nights.loc[(hour >= 17) & (hour < 21)].groupby("night_index")["home_import_kwh"]
    share = evening.sum() / nights.groupby("night_index")["home_import_kwh"].sum()
    assert share.loc[0] <= share.loc[1:].max() + 0.01
    assert share.loc[0] < 0.2  # an unmanaged night puts about 0.58 there


# --------------------------------------------------------------------------
# Warm-up with prices and the unmanaged warm-up baseline (plan A5)
# --------------------------------------------------------------------------


def test_warmup_reports_unmanaged_import_per_slot_with_london_keys() -> None:
    values = {"vehicle_count": 30, "evaluation_world_count": 3, "warmup_days": 4}
    action = _simulated(date(2026, 1, 12), values)
    no_action = _simulated(date(2026, 1, 12), values, model="no_action")

    warmup = action.warmup_home_import_kwh
    assert warmup.shape == (3, 4 * 48)
    slots = action.warmup_slots
    assert slots["slot_index"].tolist() == list(range(-192, 0))
    assert slots["night_index"].tolist() == [n for n in range(-4, 0) for _ in range(48)]
    assert slots["interval_end_utc"].iat[-1] == action.interval_start_utc[0]
    # Unmanaged (normal-path) import: identical in both models, never
    # negative, never above every EV at full home power, and not empty.
    np.testing.assert_array_equal(warmup, no_action.warmup_home_import_kwh)
    power = assumptions.HOME_CHARGING["home_charging_power_kw"].value
    assert (warmup >= 0.0).all() and (warmup <= 30 * power * 0.5 + 1e-9).all()
    assert warmup.sum() > 0.0
    # Prices are generated over the warm-up and the study.
    assert len(action.forecast_prices) == 3 * (4 + 7) * 48
    assert action.forecast_prices["interval_start_utc"].iat[0] == slots["interval_start_utc"].iat[0]


# --------------------------------------------------------------------------
# Publication-time prices (plan B4)
# --------------------------------------------------------------------------


def _starts(start: date, warmup_days: int, study_days: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(utc_half_hour_boundaries(start, warmup_days, study_days)[:-1])


def test_publication_is_13_00_london_the_day_before_delivery() -> None:
    starts = _starts(date(2026, 10, 19), 7, 7)  # includes the autumn change
    published = pd.to_datetime(day_ahead_publication_utc_ns(starts), utc=True).tz_convert(LONDON)
    delivery = starts.tz_convert(LONDON).tz_localize(None).normalize()
    assert (published.strftime("%H:%M") == "13:00").all()
    assert (published.tz_localize(None).normalize() == delivery - pd.Timedelta(days=1)).all()


def test_visible_prices_keep_published_and_shape_the_rest_by_hand() -> None:
    # Two worlds, 3 warm-up and 2 study days.  A decision at 18:00 London on
    # 13 January sees delivery days up to 14 January (published 13:00 on the
    # 13th); 15 January is not published, so it takes the mean of the same
    # half-hour over the published days (at most 7; here 10-14 January).
    starts = _starts(date(2026, 1, 13), 3, 2)
    rng = np.random.default_rng(3)
    prices = rng.normal(80.0, 20.0, size=(2, len(starts)))
    decision = pd.Timestamp("2026-01-13 18:00", tz=LONDON).tz_convert("UTC")
    visible = visible_day_ahead_prices(prices, starts, decision)

    delivery = starts.tz_convert(LONDON).tz_localize(None).normalize()
    published = delivery <= pd.Timestamp("2026-01-14")
    np.testing.assert_array_equal(visible[:, published], prices[:, published])
    half_hour = np.asarray(
        starts.tz_convert(LONDON).hour * 2 + starts.tz_convert(LONDON).minute // 30
    )
    for slot in np.flatnonzero(~published):
        same = published & (half_hour == half_hour[slot])
        np.testing.assert_allclose(visible[:, slot], prices[:, same].mean(axis=1))


def test_visible_prices_ignore_every_unpublished_price() -> None:
    starts = _starts(date(2026, 1, 12), 7, 7)
    rng = np.random.default_rng(5)
    prices = rng.normal(80.0, 20.0, size=(3, len(starts)))
    decision = pd.Timestamp("2026-01-14 09:30", tz=LONDON).tz_convert("UTC")
    unpublished = day_ahead_publication_utc_ns(starts) > decision.value
    perturbed = prices.copy()
    perturbed[:, unpublished] += rng.normal(0.0, 500.0, size=(3, int(unpublished.sum())))

    np.testing.assert_array_equal(
        visible_day_ahead_prices(prices, starts, decision),
        visible_day_ahead_prices(perturbed, starts, decision),
    )


def test_smart_plans_never_use_a_price_before_it_is_published() -> None:
    # Perturb every day-ahead price published after 13:00 London on Wednesday
    # 14 January (delivery from Thursday 15 January): the smart path's home
    # import before that instant is identical, because every plan made
    # before it ranks only prices already published.  After it, plans see
    # the new prices, so the import changes.
    simulated = _simulated(
        date(2026, 1, 12), {"vehicle_count": 60, "evaluation_world_count": 2, "warmup_days": 3}
    )
    settings, units = simulated.settings, simulated.units
    starts = _starts(settings.start_local_date, settings.warmup_days, settings.study_days)
    prices = (
        simulated.forecast_prices["wholesale_forecast_gbp_per_mwh"]
        .to_numpy()
        .reshape(settings.evaluation_world_count, len(starts))
    )
    cutoff = pd.Timestamp("2026-01-14 13:00", tz=LONDON).tz_convert("UTC")
    late = day_ahead_publication_utc_ns(starts) > cutoff.value
    perturbed = prices.copy()
    perturbed[:, late] = np.random.default_rng(9).normal(0.0, 300.0, size=(2, int(late.sum())))

    def selected_import(day_ahead: np.ndarray) -> pd.DataFrame:
        smart = smart_charging_inputs(settings, units, day_ahead, departure_margin_hours=2.0)
        fleet, *_ = simulate_fleet_intervals(
            settings,
            units,
            simulated.evaluation.inputs,
            evaluation_effective_efficiency_miles_per_battery_kwh=(
                simulated.evaluation.effective_efficiency_miles_per_battery_kwh
            ),
            public_top_up=simulated.public_top_up,
            smart_charging=smart,
        )
        return fleet.loc[fleet["path_id"].eq("selected")]

    base, moved = selected_import(prices), selected_import(perturbed)
    before = base["interval_start_utc"] < cutoff
    np.testing.assert_array_equal(
        base.loc[before, "home_grid_import_kwh"].to_numpy(),
        moved.loc[before, "home_grid_import_kwh"].to_numpy(),
    )
    assert not np.array_equal(
        base.loc[~before, "home_grid_import_kwh"].to_numpy(),
        moved.loc[~before, "home_grid_import_kwh"].to_numpy(),
    )


# --------------------------------------------------------------------------
# Day-keyed price summaries use session nights
# --------------------------------------------------------------------------


def _prices_frame(start: date, by_world: list[np.ndarray]) -> pd.DataFrame:
    starts = _starts(start, 0, 7)
    london = pd.Series(starts.tz_convert(LONDON))
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "world_id": w,
                    "slot_index": np.arange(len(starts)),
                    "interval_start_london": london,
                    "wholesale_forecast_gbp_per_mwh": values,
                }
            )
            for w, values in enumerate(by_world)
        ],
        ignore_index=True,
    )


def test_relative_prices_cover_all_seven_nights_of_a_noon_study() -> None:
    # Keyed by calendar date, a noon-to-noon study has no whole Monday; keyed
    # by session night every weekday row (the evening's weekday) is filled.
    rng = np.random.default_rng(1)
    frame = _prices_frame(date(2026, 1, 12), [rng.normal(80, 10, 336) for _ in range(2)])
    bands = price_relative_bands(frame)
    by_day = bands.loc[bands["weekday"].notna()]
    assert by_day["sample_count"].eq(2).all()
    # Relative prices sum to zero over each whole night, so each weekday's
    # 48 cell means do too.
    sums = by_day.groupby("weekday")["mean"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 0.0, atol=1e-9)


def test_cheapest_half_hour_quantiles_follow_the_night_not_the_clock() -> None:
    # Cheapest half-hours at 23:00 (two nights) and 01:00 (five nights) of a
    # noon-to-noon night: in night order 23:00 comes first, so P10 is 23:00
    # and P50 01:00.  Plain clock-hour quantiles would make P10 01:00.
    starts = _starts(date(2026, 1, 12), 0, 7)
    clock = np.asarray(starts.tz_convert(LONDON).strftime("%H:%M"))
    night = np.repeat(np.arange(7), 48)
    prices = np.full(336, 100.0)
    prices[(clock == "23:00") & (night < 2)] = 10.0
    prices[(clock == "01:00") & (night >= 2)] = 10.0
    summary = cheapest_half_hour_summary(_prices_frame(date(2026, 1, 12), [prices])).iloc[0]
    assert summary["world_day_count"] == 7
    assert summary["local_time_label_p10"] == "23:00"
    assert summary["local_time_label_p50"] == "01:00"
