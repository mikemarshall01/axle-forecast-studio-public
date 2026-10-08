"""Synthetic market prices: net demand, day-ahead, intraday, imbalance and shocks.

Decision 0004 items 48, 53, 55 and 56; plan §2B (B1, B5) and §7 (H1).  Every
price here is synthetic and illustrative, not market data.
"""

import numpy as np
import pandas as pd
import pytest

from axle_studio.model import assumptions
from axle_studio.model.assumptions import PRICES
from axle_studio.model.sampling import (
    ILLUSTRATIVE_PRICE_NOISE_AR1_COEFFICIENT,
    MARKET_PRICE_INPUTS,
    MarketPrices,
    day_ahead_publication_utc_ns,
    demand_shape_gw,
    generate_market_price_paths,
    sample_daily_temperature,
    solar_daylight_shape,
    supply_curve_gbp_per_mwh,
)

_CURVE_NAMES = (
    "supply_reference_net_demand_gw",
    "supply_reference_price_gbp_per_mwh",
    "supply_slope_gbp_per_mwh_per_gw",
    "scarcity_threshold_gw",
    "scarcity_width_gw",
    "scarcity_price_gbp_per_mwh",
)
_NO_NOISE = {
    "day_ahead_sd_gbp_per_mwh": 0.0,
    "daily_level_sd_gbp_per_mwh": 0.0,
    "wind_logit_sd": 0.0,
    "intraday_hourly_sd_gbp_per_mwh": 0.0,
    "imbalance_tail_scale_gbp_per_mwh": 0.0,
    "solar_clearness_logit_sd": 0.0,
}
_NO_SHOCKS = {"mild_shock_rate_per_night": 0.0, "big_shock_rate_per_week": 0.0}


def defaults(**overrides: object) -> dict[str, object]:
    values = dict(
        assumptions.forecast_inputs(None, warmup_days=0, study_days=7)["price_assumptions"]
    )
    values.update(overrides)
    return values


def starts(first: str = "2026-10-05", days: int = 7) -> pd.DatetimeIndex:
    """London-midnight-aligned UTC half-hours, as the runner prices them."""

    local = pd.date_range(first, periods=48 * days, freq="30min", tz="Europe/London")
    return local.tz_convert("UTC")


def market(
    worlds: int = 3,
    *,
    seed: int = 42,
    temperature_c: float | np.ndarray = 9.0,
    interval_start_utc: pd.DatetimeIndex | None = None,
    scripted_known_shock_gw: np.ndarray | None = None,
    scripted_surprise_shock_gw: np.ndarray | None = None,
    scripted_surprise_reveal_hours: float = 0.0,
    **overrides: object,
) -> MarketPrices:
    index = starts() if interval_start_utc is None else interval_start_utc
    local = index.tz_convert("Europe/London")
    days = (local[-1].normalize() - local[0].normalize()).days + 1
    temperature = np.broadcast_to(np.asarray(temperature_c, dtype=float), (worlds, days))
    return generate_market_price_paths(
        np.random.default_rng(seed),
        interval_start_utc=index,
        evaluation_world_count=worlds,
        daily_temperature_c=temperature,
        price_assumptions=defaults(**overrides),
        scripted_known_shock_gw=scripted_known_shock_gw,
        scripted_surprise_shock_gw=scripted_surprise_shock_gw,
        scripted_surprise_reveal_hours=scripted_surprise_reveal_hours,
    )


def make_paths(
    worlds: int = 3, **kwargs: object
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The day-ahead frame, the realised frame and the shock table."""

    prices = market(worlds, **kwargs)
    return prices.day_ahead, prices.realised, prices.shocks


def matrix(frame: pd.DataFrame, column: str, worlds: int) -> np.ndarray:
    return frame[column].to_numpy().reshape(worlds, -1)


def test_ar1_coefficient_is_the_assumptions_record() -> None:
    # One source for the price-noise persistence (decision 0004 items 14, 21).
    # 0.92 is the Elexon half-hour residual's lag-1 autocorrelation (item 53).
    assert ILLUSTRATIVE_PRICE_NOISE_AR1_COEFFICIENT == PRICES["ar1_phi"].value == 0.92


def test_the_runner_passes_exactly_the_generators_inputs() -> None:
    assert set(defaults()) == set(MARKET_PRICE_INPUTS)


@pytest.mark.parametrize("days", [1, 3, 7])
def test_shapes_keys_and_availability_cutoffs(days: int) -> None:
    forecast, evaluation, shocks = make_paths(interval_start_utc=starts(days=days))
    rows = 48 * days * 3
    assert len(forecast) == len(evaluation) == rows
    assert list(forecast) == [
        "world_id",
        "interval_start_utc",
        "interval_end_utc",
        "forecast_available_at_utc",
        "wholesale_forecast_gbp_per_mwh",
        "system_net_demand_gw",
        "evidence_kind",
    ]
    assert list(evaluation) == [
        "world_id",
        "interval_start_utc",
        "interval_end_utc",
        "evaluation_context_price_gbp_per_mwh",
        "imbalance_price_gbp_per_mwh",
        "system_long",
        "outcome_available_at_utc",
        "evidence_kind",
    ]
    assert list(shocks) == [
        "world_id",
        "shock_class",
        "start_slot_index",
        "start_utc",
        "duration_slots",
        "size_gw",
        "direction",
        "known_day_ahead",
        "evidence_kind",
    ]
    pd.testing.assert_frame_equal(
        forecast[["world_id", "interval_start_utc", "interval_end_utc"]],
        evaluation[["world_id", "interval_start_utc", "interval_end_utc"]],
    )
    # Plan B4: each day's prices are published at 13:00 London the day before.
    published = pd.to_datetime(
        day_ahead_publication_utc_ns(pd.DatetimeIndex(forecast["interval_start_utc"])), utc=True
    )
    assert (forecast["forecast_available_at_utc"].to_numpy() == published.to_numpy()).all()
    assert (forecast["forecast_available_at_utc"] < forecast["interval_start_utc"]).all()
    assert evaluation["outcome_available_at_utc"].eq(evaluation["interval_end_utc"]).all()
    for frame in (forecast, evaluation, shocks):
        assert frame["evidence_kind"].eq("synthetic").all()
    for column in ("wholesale_forecast_gbp_per_mwh", "system_net_demand_gw"):
        assert np.isfinite(forecast[column]).all()
    for column in ("evaluation_context_price_gbp_per_mwh", "imbalance_price_gbp_per_mwh"):
        assert np.isfinite(evaluation[column]).all()
    assert evaluation["system_long"].dtype == bool


# ---------------------------------------------------------------------------
# The system model: demand shape, heating, wind and the supply curve (H1)
# ---------------------------------------------------------------------------


def test_supply_curve_is_monotone_convex_and_negative_when_net_demand_is_very_low() -> None:
    curve = {name: defaults()[name] for name in _CURVE_NAMES}
    net = np.linspace(-5.0, 45.0, 501)
    price = supply_curve_gbp_per_mwh(net, **curve)
    assert (np.diff(price) > 0).all()
    assert (np.diff(price, n=2) >= -1e-9).all()
    assert price[0] < 0.0
    # Convexity is the point (decision 0004 item 55): the same 4 GW adds
    # more when the system is tight than when it is slack.
    slack = supply_curve_gbp_per_mwh(np.array([16.0, 12.0]), **curve)
    tight = supply_curve_gbp_per_mwh(np.array([34.0, 30.0]), **curve)
    assert tight[0] - tight[1] > 2 * (slack[0] - slack[1])


def test_supply_curve_passes_through_its_reference_point() -> None:
    curve = {name: defaults()[name] for name in _CURVE_NAMES} | {"scarcity_price_gbp_per_mwh": 0.0}
    at_reference = supply_curve_gbp_per_mwh(
        np.array([curve["supply_reference_net_demand_gw"]]), **curve
    )
    assert at_reference[0] == pytest.approx(curve["supply_reference_price_gbp_per_mwh"])


def test_default_winter_weekday_demand_shape_has_a_night_trough_and_an_evening_peak() -> None:
    # Net of a median day's solar (the shape is demand before solar).
    values = defaults()
    hours = np.arange(48) / 2
    solar = (
        values["solar_installed_gw"]
        * solar_daylight_shape(hours, values, "winter")
        * values["solar_clearness_median"]
    )
    shape = demand_shape_gw(values["demand_shape_gw.winter_weekday"], hours) - solar
    assert 2.0 <= hours[shape.argmin()] <= 5.0
    assert 16.0 <= hours[shape.argmax()] <= 19.0
    assert demand_shape_gw((25.0, 0, 0, 0, 0, 0, 0), hours) == pytest.approx(np.full(48, 25.0))


def test_zero_noise_prices_are_the_supply_curve_of_the_net_demand() -> None:
    # With every random SD and shock rate zero, each half-hour's day-ahead
    # price is f(demand shape + heating − median wind − median-day solar),
    # recomputed here.
    values = defaults(**_NO_NOISE, **_NO_SHOCKS)
    forecast, evaluation, shocks = make_paths(2, temperature_c=9.0, **_NO_NOISE, **_NO_SHOCKS)
    local = starts().tz_convert("Europe/London")
    hours = np.asarray(local.hour + local.minute / 60)
    weekend = np.asarray(local.dayofweek >= 5)
    demand = np.where(
        weekend,
        demand_shape_gw(values["demand_shape_gw.winter_weekend"], hours),
        demand_shape_gw(values["demand_shape_gw.winter_weekday"], hours),
    )
    heating = values["heating_gw_per_c"] * (values["heating_threshold_c"] - 9.0)
    wind = values["wind_installed_gw"] * values["wind_median_capacity_factor.winter"]
    solar = (
        values["solar_installed_gw"]
        * solar_daylight_shape(hours, values, "winter")
        * values["solar_clearness_median"]
    )
    net = demand + heating - wind - solar
    expected = supply_curve_gbp_per_mwh(net, **{name: values[name] for name in _CURVE_NAMES})

    assert shocks.empty
    for world in range(2):
        np.testing.assert_allclose(matrix(forecast, "system_net_demand_gw", 2)[world], net)
        np.testing.assert_allclose(
            matrix(forecast, "wholesale_forecast_gbp_per_mwh", 2)[world], expected
        )
    # No intraday news and no surprises: the intraday close is the day-ahead
    # price, and imbalance is it plus the state's premium.
    np.testing.assert_allclose(
        evaluation["evaluation_context_price_gbp_per_mwh"],
        forecast["wholesale_forecast_gbp_per_mwh"],
    )
    premium = np.where(
        evaluation["system_long"],
        values["imbalance_long_premium_gbp_per_mwh"],
        values["imbalance_short_premium_gbp_per_mwh"],
    )
    np.testing.assert_allclose(
        evaluation["imbalance_price_gbp_per_mwh"],
        evaluation["evaluation_context_price_gbp_per_mwh"] + premium,
    )


def test_heating_raises_net_demand_only_below_the_threshold() -> None:
    # H1: each °C below the threshold adds heating_gw_per_c for the whole
    # day.  With no wind, noise or shocks the worlds differ only by their
    # temperatures: 5, 10, 20 and 25 °C against the 15.5 °C threshold.
    temperatures = np.array([[5.0] * 7, [10.0] * 7, [20.0] * 7, [25.0] * 7])
    still, _, _ = make_paths(
        4, temperature_c=temperatures, wind_installed_gw=0.0, **_NO_NOISE, **_NO_SHOCKS
    )
    net = matrix(still, "system_net_demand_gw", 4)
    rate = defaults()["heating_gw_per_c"]
    np.testing.assert_allclose(net[0] - net[1], 5.0 * rate)
    np.testing.assert_allclose(net[1] - net[2], 5.5 * rate)
    np.testing.assert_allclose(net[2], net[3])


def test_a_colder_world_is_a_dearer_world_on_the_same_draws() -> None:
    warm, _, _ = make_paths(20, temperature_c=14.0, **_NO_SHOCKS)
    cold, _, _ = make_paths(20, temperature_c=2.0, **_NO_SHOCKS)
    difference = (
        cold["wholesale_forecast_gbp_per_mwh"] - warm["wholesale_forecast_gbp_per_mwh"]
    ).to_numpy()
    assert (difference > 0).all()


def test_wind_capacity_factor_persists_across_days() -> None:
    # The latent wind AR(1) is per day: net demand minus the fixed demand
    # shape and heating is minus wind, constant within a London day.
    forecast, _, _ = make_paths(200, **_NO_SHOCKS)
    still, _, _ = make_paths(200, wind_installed_gw=0.0, **_NO_SHOCKS)
    wind = (
        matrix(still, "system_net_demand_gw", 200) - matrix(forecast, "system_net_demand_gw", 200)
    ).reshape(200, 7, 48)
    assert np.allclose(wind, wind[:, :, :1])
    daily = wind[:, :, 0]
    lag_one = np.corrcoef(daily[:, :-1].ravel(), daily[:, 1:].ravel())[0, 1]
    assert lag_one > 0.4
    assert ((daily > 0) & (daily < defaults()["wind_installed_gw"])).all()


# ---------------------------------------------------------------------------
# Reproducibility and common random numbers
# ---------------------------------------------------------------------------


def test_seed_replay_ignores_numpy_global_state() -> None:
    np.random.seed(1)
    first = make_paths()
    np.random.seed(2)
    second = make_paths()
    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_draw_count_is_fixed_whatever_the_values() -> None:
    # Plan M6: zero SDs and zero shock rates take the same draws as the
    # defaults, so a price edit never shifts a later channel.
    temperature = np.full((2, 7), 9.0)
    rngs = []
    for overrides in ({}, {**_NO_NOISE, **_NO_SHOCKS}, {"imbalance_tail_df": 30.0}):
        rng = np.random.default_rng(4)
        generate_market_price_paths(
            rng,
            interval_start_utc=starts(),
            evaluation_world_count=2,
            daily_temperature_c=temperature,
            price_assumptions=defaults(**overrides),
        )
        rngs.append(rng.random())
    assert rngs[0] == rngs[1] == rngs[2]


def test_editing_one_channel_leaves_the_others_unchanged() -> None:
    base_forecast, base_evaluation, _ = make_paths(5)
    base_drift = (
        base_evaluation["evaluation_context_price_gbp_per_mwh"]
        - base_forecast["wholesale_forecast_gbp_per_mwh"]
    )
    # A wider intraday SD changes only intraday, not the day-ahead price.
    forecast, evaluation, _ = make_paths(5, intraday_hourly_sd_gbp_per_mwh=5.0)
    pd.testing.assert_frame_equal(forecast, base_forecast)
    drift = (
        evaluation["evaluation_context_price_gbp_per_mwh"]
        - forecast["wholesale_forecast_gbp_per_mwh"]
    )
    assert not np.allclose(drift, base_drift)
    # A wider day-ahead noise leaves the intraday drift (close − day-ahead)
    # and the NIV states as they were, except where the price cap or floor
    # now binds differently.
    forecast, evaluation, _ = make_paths(5, day_ahead_sd_gbp_per_mwh=20.0)
    drift = (
        evaluation["evaluation_context_price_gbp_per_mwh"]
        - forecast["wholesale_forecast_gbp_per_mwh"]
    )
    assert np.isclose(drift, base_drift, atol=1e-9).mean() > 0.99
    pd.testing.assert_series_equal(evaluation["system_long"], base_evaluation["system_long"])


# ---------------------------------------------------------------------------
# Intraday and imbalance (B5)
# ---------------------------------------------------------------------------


def test_intraday_close_is_a_martingale_around_the_day_ahead_price() -> None:
    # Mean(final − day-ahead) ≈ 0; a period traded for longer (an evening,
    # published 13:00 the day before) ends further from day-ahead than one
    # just after midnight; news moves adjacent periods together.
    worlds = 400
    forecast, evaluation, _ = make_paths(worlds, **_NO_SHOCKS)
    drift = matrix(evaluation, "evaluation_context_price_gbp_per_mwh", worlds) - matrix(
        forecast, "wholesale_forecast_gbp_per_mwh", worlds
    )
    sd_hour = defaults()["intraday_hourly_sd_gbp_per_mwh"]
    assert drift.mean() == pytest.approx(0.0, abs=0.3)
    local = starts().tz_convert("Europe/London")
    midnight = np.asarray((local.hour == 0) & (local.minute == 0))
    evening = np.asarray((local.hour == 22) & (local.minute == 0))
    # 00:00 trades 13:00 → 23:00 the day before: 10 updates; 22:00 trades 32.
    assert drift[:, midnight].std() == pytest.approx(sd_hour * np.sqrt(10), rel=0.1)
    assert drift[:, evening].std() == pytest.approx(sd_hour * np.sqrt(32), rel=0.1)
    lag_one = np.corrcoef(drift[:, :-1].ravel(), drift[:, 1:].ravel())[0, 1]
    assert lag_one > 0.6


def test_imbalance_states_and_premiums_follow_their_records() -> None:
    worlds = 300
    values = defaults()
    _, evaluation, _ = make_paths(worlds, **_NO_SHOCKS)
    long = matrix(evaluation, "system_long", worlds)
    assert long.mean() == pytest.approx(values["imbalance_long_share"], abs=0.02)
    lag_one = np.corrcoef(long[:, :-1].ravel(), long[:, 1:].ravel())[0, 1]
    assert lag_one == pytest.approx(values["imbalance_state_persistence"], abs=0.03)
    gap = (
        evaluation["imbalance_price_gbp_per_mwh"]
        - evaluation["evaluation_context_price_gbp_per_mwh"]
    )
    # Student-t(3) tails have median zero, so the median gap is the premium.
    assert gap[evaluation["system_long"]].median() == pytest.approx(
        values["imbalance_long_premium_gbp_per_mwh"], abs=0.5
    )
    assert gap[~evaluation["system_long"]].median() == pytest.approx(
        values["imbalance_short_premium_gbp_per_mwh"], abs=0.5
    )
    tail = gap - np.where(
        evaluation["system_long"],
        values["imbalance_long_premium_gbp_per_mwh"],
        values["imbalance_short_premium_gbp_per_mwh"],
    )
    assert (tail.abs() > 40.0).mean() > 0.001  # fat tails: rare large moves


# ---------------------------------------------------------------------------
# Net-demand shocks (decision 0004 item 56)
# ---------------------------------------------------------------------------


def test_shocks_start_only_inside_the_evening_to_morning_window() -> None:
    _, _, shocks = make_paths(200)
    local = shocks["start_utc"].dt.tz_convert("Europe/London")
    hour = local.dt.hour + local.dt.minute / 60
    assert len(shocks) > 0
    assert ((hour >= 15.0) | (hour < 9.0)).all()
    assert ((local.dt.minute == 0) | (local.dt.minute == 30)).all()
    assert shocks["start_slot_index"].between(0, 7 * 48 - 1).all()
    assert shocks["start_utc"].eq(starts()[shocks["start_slot_index"]]).all()


def test_shock_counts_sizes_and_durations_match_their_records() -> None:
    worlds = 400
    values = defaults()
    _, _, shocks = make_paths(worlds)
    per_world_week = shocks.groupby("shock_class").size() / worlds
    # Seven nights of starts fall in a week (the previous evening's early
    # hours come in, the last evening's late hours go out).
    assert per_world_week["mild"] == pytest.approx(
        7 * values["mild_shock_rate_per_night"], rel=0.05
    )
    assert per_world_week["big"] == pytest.approx(values["big_shock_rate_per_week"], rel=0.08)
    mild = shocks.query("shock_class == 'mild'")
    big = shocks.query("shock_class == 'big'")
    assert mild["size_gw"].mean() == pytest.approx(values["mild_shock_scale_gw"], rel=0.05)
    assert big["size_gw"].median() == pytest.approx(values["big_shock_median_gw"], rel=0.08)
    assert mild["duration_slots"].between(1, 4).all()
    assert big["duration_slots"].between(2, 8).all()
    assert shocks["direction"].eq("up").mean() == pytest.approx(0.5, abs=0.03)
    assert mild["known_day_ahead"].mean() == pytest.approx(0.5, abs=0.03)
    assert big["known_day_ahead"].mean() == pytest.approx(0.4, abs=0.05)


def test_zero_shock_rates_reproduce_the_no_shock_prices_exactly() -> None:
    # Zero rates take the same draws; with no shock the expected surprise
    # impact is zero, so day-ahead, intraday and imbalance equal a run with
    # the shock draws present but never active (tiny rates below any draw).
    no_rate = market(10, **_NO_SHOCKS)
    assert no_rate.shocks.empty
    assert not no_rate.known_shock_gw.any() and not no_rate.surprise_shock_gw.any()
    base = no_rate.day_ahead
    rebuilt = supply_curve_gbp_per_mwh(
        base["system_net_demand_gw"].to_numpy(), **{n: defaults()[n] for n in _CURVE_NAMES}
    )
    assert np.isfinite(rebuilt).all()
    tiny = market(10, mild_shock_rate_per_night=1e-12, big_shock_rate_per_week=1e-12)
    assert tiny.shocks.empty
    pd.testing.assert_frame_equal(tiny.realised, no_rate.realised, atol=1e-6)
    pd.testing.assert_frame_equal(tiny.day_ahead, no_rate.day_ahead, atol=1e-6)


def test_known_shocks_move_day_ahead_and_surprises_move_only_intraday_and_imbalance() -> None:
    worlds = 10
    base_forecast, base_evaluation, _ = make_paths(worlds, **_NO_SHOCKS)
    all_known = {"mild_shock_known_share": 1.0, "big_shock_known_share": 1.0}
    known_forecast, known_evaluation, known = make_paths(worlds, **all_known)
    all_surprise = {
        "mild_shock_known_share": 0.0,
        "big_shock_known_share": 0.0,
        "niv_surprise_shift": 0.0,
    }
    surprise_forecast, surprise_evaluation, surprise = make_paths(worlds, **all_surprise)
    assert known["known_day_ahead"].all() and not surprise["known_day_ahead"].any()

    def drift(forecast: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
        return (
            evaluation["evaluation_context_price_gbp_per_mwh"]
            - forecast["wholesale_forecast_gbp_per_mwh"]
        ).to_numpy()

    # Known: the day-ahead net demand and price move, the intraday drift not.
    assert not np.allclose(
        known_forecast["system_net_demand_gw"], base_forecast["system_net_demand_gw"]
    )
    np.testing.assert_allclose(
        drift(known_forecast, known_evaluation), drift(base_forecast, base_evaluation), atol=1e-9
    )
    # Surprise: day-ahead net demand is untouched (day-ahead prices only
    # carry the expected surprise impact), and the intraday close moves only
    # in shocked half-hours.
    np.testing.assert_allclose(
        surprise_forecast["system_net_demand_gw"], base_forecast["system_net_demand_gw"]
    )
    moved = (
        surprise_evaluation["evaluation_context_price_gbp_per_mwh"]
        - base_evaluation["evaluation_context_price_gbp_per_mwh"]
    ).to_numpy()
    shocked = np.zeros((worlds, 7 * 48), dtype=bool)
    for row in surprise.itertuples():
        shocked[row.world_id, row.start_slot_index : row.start_slot_index + row.duration_slots] = (
            True
        )
    shocked = shocked.reshape(-1)
    assert np.allclose(moved[~shocked], 0.0)
    assert (np.abs(moved[shocked]) > 0).all()
    np.testing.assert_allclose(
        surprise_evaluation["imbalance_price_gbp_per_mwh"]
        - surprise_evaluation["evaluation_context_price_gbp_per_mwh"],
        base_evaluation["imbalance_price_gbp_per_mwh"]
        - base_evaluation["evaluation_context_price_gbp_per_mwh"],
        atol=1e-9,
    )


def test_a_shock_is_a_trapezoid_in_net_demand() -> None:
    # One world, only big known up shocks of a fixed 4 h (8 half-hours) and
    # no size spread: each shows as 1/3, 2/3, 1, 1, 1, 1, 2/3, 1/3 of 4 GW.
    shocked_values = {
        **_NO_NOISE,
        "mild_shock_rate_per_night": 0.0,
        "big_shock_rate_per_week": 1.0,
        "big_shock_log_sd": 0.0,
        "big_shock_min_hours": 4.0,
        "big_shock_max_hours": 4.0,
        "big_shock_known_share": 1.0,
        "shock_up_probability": 1.0,
    }
    for seed in range(20):
        forecast, _, shocks = make_paths(1, seed=seed, **shocked_values)
        if len(shocks) == 1 and shocks["start_slot_index"].iat[0] <= 7 * 48 - 8:
            break
    else:
        pytest.fail("no seed gave exactly one whole shock")
    base, _, _ = make_paths(1, seed=seed, **_NO_NOISE, **_NO_SHOCKS)
    added = (forecast["system_net_demand_gw"] - base["system_net_demand_gw"]).to_numpy()
    start = int(shocks["start_slot_index"].iat[0])
    expected = np.zeros_like(added)
    expected[start : start + 8] = 4.0 * np.array([1, 2, 3, 3, 3, 3, 2, 1]) / 3
    np.testing.assert_allclose(added, expected, atol=1e-9)


def test_shock_profiles_are_returned_and_scripted_profiles_draw_nothing() -> None:
    # Trading contract §1.4 and §2.5: the known and surprise profiles come
    # back separately, and a scripted profile adds to them before the supply
    # curve without taking any draw, so the stochastic futures stay put.
    worlds = 4
    base = market(worlds)
    np.testing.assert_allclose(
        base.day_ahead["system_net_demand_gw"].to_numpy().reshape(worlds, -1)
        - market(worlds, **_NO_SHOCKS)
        .day_ahead["system_net_demand_gw"]
        .to_numpy()
        .reshape(worlds, -1),
        base.known_shock_gw,
        atol=1e-9,
    )
    assert base.surprise_shock_gw.shape == (worlds, 7 * 48)
    assert (base.surprise_shock_gw != 0).any()

    scripted = np.zeros(7 * 48)
    scripted[36:40] = 3.0  # a 2 h, 3 GW known evening event in every world
    with_event = market(worlds, scripted_known_shock_gw=scripted)
    np.testing.assert_allclose(with_event.known_shock_gw, base.known_shock_gw + scripted)
    np.testing.assert_allclose(with_event.surprise_shock_gw, base.surprise_shock_gw)
    pd.testing.assert_frame_equal(with_event.shocks, base.shocks)
    moved = (
        (
            with_event.day_ahead["wholesale_forecast_gbp_per_mwh"]
            - base.day_ahead["wholesale_forecast_gbp_per_mwh"]
        )
        .to_numpy()
        .reshape(worlds, -1)
    )
    assert (moved[:, 36:40] > 0).all()
    assert np.allclose(np.delete(moved, np.s_[36:40], axis=1), 0.0)

    surprise_event = market(worlds, scripted_surprise_shock_gw=scripted)
    pd.testing.assert_frame_equal(surprise_event.day_ahead, base.day_ahead)
    lifted = (
        (
            surprise_event.realised["evaluation_context_price_gbp_per_mwh"]
            - base.realised["evaluation_context_price_gbp_per_mwh"]
        )
        .to_numpy()
        .reshape(worlds, -1)
    )
    assert (lifted[:, 36:40] > 0).all()


def test_intraday_path_steps_from_day_ahead_to_the_close() -> None:
    # Trading contract §1.4: path[world, slot, h] is the intraday price h
    # whole hours before gate closure for every world; h = 0 is the close,
    # and before a period's day-ahead publication it is the day-ahead price.
    prices = market(12, **_NO_SHOCKS)
    path = prices.intraday_path_gbp_per_mwh
    assert path.shape == (12, 7 * 48, 36)
    day_ahead = prices.day_ahead["wholesale_forecast_gbp_per_mwh"].to_numpy().reshape(12, -1)
    close = prices.realised["evaluation_context_price_gbp_per_mwh"].to_numpy().reshape(12, -1)
    np.testing.assert_allclose(path[:, :, 0], close)
    np.testing.assert_allclose(path[:, :, 35], day_ahead)
    # Midnight trades 10 hours (13:00 → 23:00 the day before): from h = 10
    # back it has had no update yet.
    np.testing.assert_allclose(path[:, 0, 10:] - day_ahead[:, :1], 0.0, atol=1e-9)
    assert not np.allclose(path[:, 0, 9], day_ahead[:, 0])


def test_surprises_reach_the_intraday_path_at_their_reveal_lead() -> None:
    # Contract §2.5, Q10: a stochastic surprise is known surprise_reveal_hours
    # before it starts.  A half-hour's own step h = floor(k/2 + reveal − gate)
    # is the first that includes it; earlier steps do not.
    values = {
        **_NO_NOISE,
        "mild_shock_rate_per_night": 0.0,
        "big_shock_rate_per_week": 1.0,
        "big_shock_known_share": 0.0,
        "shock_up_probability": 1.0,
        "surprise_reveal_hours": 3.0,
        "niv_surprise_shift": 0.0,
    }
    prices = market(40, **values)
    quiet = market(40, **_NO_NOISE, **_NO_SHOCKS)
    shock = prices.shocks.iloc[0]
    world, start = int(shock["world_id"]), int(shock["start_slot_index"])
    path = prices.intraday_path_gbp_per_mwh[world, start]
    # First half-hour (k = 0): known from 3 − 1 = 2 hours before gate closure.
    impact = path - quiet.intraday_path_gbp_per_mwh[world, start]
    # The biggest step in the path is when this shock is revealed, between
    # h = 3 and h = 2 (before it only the expected part of other possible
    # surprises moves, a little, as they are revealed or ruled out).
    jumps = impact[:-1] - impact[1:]
    assert int(np.argmax(jumps)) == 2
    assert jumps[2] > 5 * np.abs(np.delete(jumps, 2)).max()
    # A lead shorter than gate closure leaves the first half-hour to
    # imbalance only: the close ignores it, the imbalance price does not.
    short = market(40, **{**values, "surprise_reveal_hours": 0.0})
    first = world * 336 + start
    close_gap = (
        short.realised["evaluation_context_price_gbp_per_mwh"].iat[first]
        - short.day_ahead["wholesale_forecast_gbp_per_mwh"].iat[first]
    )
    imbalance_gap = (
        short.realised["imbalance_price_gbp_per_mwh"].iat[first]
        - short.realised["evaluation_context_price_gbp_per_mwh"].iat[first]
    )
    assert close_gap < 0.0  # day-ahead priced the expected surprise; the close saw none
    long = bool(short.realised["system_long"].iat[first])
    premium = defaults()[
        "imbalance_long_premium_gbp_per_mwh" if long else "imbalance_short_premium_gbp_per_mwh"
    ]
    assert imbalance_gap - premium > 1.0  # the late surprise lifts imbalance


@pytest.mark.parametrize("premium", [0.0, 5.0])
def test_no_free_money_intraday_close_exceeds_day_ahead_by_the_premium_only(
    premium: float,
) -> None:
    # Mike's rule: day-ahead is the expected intraday close given day-ahead
    # information, less the deliberate premium.  With default shocks (convex
    # curve, surprises) mean(close − day-ahead) is the premium within 3 SE.
    worlds = 400
    prices = market(worlds, seed=11, da_id_premium_gbp_per_mwh=premium)
    gap = (
        (
            prices.realised["evaluation_context_price_gbp_per_mwh"]
            - prices.day_ahead["wholesale_forecast_gbp_per_mwh"]
        )
        .to_numpy()
        .reshape(worlds, -1)
    )
    per_world = gap.mean(axis=1)
    standard_error = per_world.std(ddof=1) / np.sqrt(worlds)
    assert abs(per_world.mean() - premium) < 3 * standard_error


def test_an_up_surprise_makes_the_system_short_more_often() -> None:
    worlds = 200
    base = {"big_shock_known_share": 0.0, "mild_shock_known_share": 0.0}
    up = market(worlds, shock_up_probability=1.0, **base)
    covered = up.surprise_shock_gw.reshape(-1) > 0
    long = up.realised["system_long"].to_numpy()
    assert long[covered].mean() < long[~covered].mean() - 0.1
    none = market(worlds, shock_up_probability=1.0, niv_surprise_shift=0.0, **base)
    assert abs(none.realised["system_long"].to_numpy()[covered].mean() - 0.53) < 0.05


def test_every_price_stays_inside_the_auction_limits_at_the_upper_bounds() -> None:
    # Review finding: without limits the exponential scarcity term reached
    # £17.5m/MWh at bounded edits.  Every editable price record at its upper
    # bound, plus the dialog-free tails at their widest: prices stay finite
    # and inside the GB auction floor and cap.
    upper = {
        record.name: record.bounds[1]
        for section in (
            assumptions.SYSTEM,
            assumptions.SHOCKS,
            assumptions.PRICES,
            assumptions.TRADING,
        )
        for record in section.values()
        if record.editable
    }
    prices = market(20, temperature_c=-5.0, **upper)
    floor = defaults()["price_floor_gbp_per_mwh"]
    cap = defaults()["price_cap_gbp_per_mwh"]
    for values in (
        prices.day_ahead["wholesale_forecast_gbp_per_mwh"],
        prices.realised["evaluation_context_price_gbp_per_mwh"],
        prices.realised["imbalance_price_gbp_per_mwh"],
        prices.intraday_path_gbp_per_mwh.reshape(-1),
    ):
        values = np.asarray(values)
        assert np.isfinite(values).all()
        assert values.min() >= floor and values.max() <= cap


# ---------------------------------------------------------------------------
# Solar (decision 0004 item 58)
# ---------------------------------------------------------------------------


def test_solar_is_zero_at_night_and_higher_and_longer_in_summer() -> None:
    values = defaults()
    hours = np.arange(48) / 2
    winter = solar_daylight_shape(hours, values, "winter")
    summer = solar_daylight_shape(hours, values, "summer")
    assert (winter[(hours < 7.5) | (hours >= 16.0)] == 0).all()
    assert (summer[(hours < 4.5) | (hours >= 21.0)] == 0).all()
    assert summer.max() > winter.max() and (summer > 0).sum() > (winter > 0).sum()
    assert 12.0 <= hours[summer.argmax()] <= 13.5


def test_solar_clearness_persists_across_days_and_lowers_midday_net_demand() -> None:
    # Net demand with and without solar differs only by solar output, which
    # is constant in shape within a day and scaled by that day's clearness.
    worlds = 200
    with_solar = market(worlds, **_NO_SHOCKS, interval_start_utc=starts("2027-07-05"))
    without = market(
        worlds, **_NO_SHOCKS, solar_installed_gw=0.0, interval_start_utc=starts("2027-07-05")
    )
    solar = (
        without.day_ahead["system_net_demand_gw"].to_numpy()
        - with_solar.day_ahead["system_net_demand_gw"].to_numpy()
    ).reshape(worlds, 7, 48)
    assert (solar >= -1e-9).all()
    assert np.allclose(solar[:, :, :8], 0.0)  # 00:00-04:00 London
    noon = solar[:, :, 25]  # 12:30 London
    lag_one = np.corrcoef(noon[:, :-1].ravel(), noon[:, 1:].ravel())[0, 1]
    assert lag_one > 0.25


# ---------------------------------------------------------------------------
# Clock changes and invalid inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("first", ["2026-03-23", "2026-10-19"])
def test_accepts_a_london_offset_change_and_follows_the_local_clock(first: str) -> None:
    # Decision 0004 item 1: the runner's 336 UTC slots may cross a clock
    # change; the demand shape follows London clock time on both sides.
    index = starts(first).tz_convert("UTC")
    index = pd.date_range(index[0], periods=336, freq="30min", tz="UTC")
    shape = (0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # peak at local 12:00, trough at 00:00
    forecast, _, _ = make_paths(
        1,
        interval_start_utc=index,
        solar_installed_gw=0.0,
        **_NO_NOISE,
        **_NO_SHOCKS,
        **{
            f"demand_shape_gw.{s}_{d}": shape
            for s in ("winter", "summer")
            for d in ("weekday", "weekend")
        },
    )
    local = forecast["interval_start_utc"].dt.tz_convert("Europe/London")
    by_day = forecast.assign(date=local.dt.date, hour=local.dt.hour + local.dt.minute / 60)
    whole_days = by_day.groupby("date").filter(lambda day: len(day) >= 46)
    peak_hours = whole_days.loc[
        whole_days.groupby("date")["system_net_demand_gw"].idxmax(), "hour"
    ].to_numpy()
    assert (peak_hours == 12.0).all()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("day_ahead_sd_gbp_per_mwh", float("nan")),
        ("day_ahead_sd_gbp_per_mwh", -1.0),
        ("intraday_hourly_sd_gbp_per_mwh", -1.0),
        ("wind_ar1_phi", 1.0),
        ("wind_median_capacity_factor.winter", 0.0),
        ("supply_slope_gbp_per_mwh_per_gw", -1.0),
        ("scarcity_width_gw", 0.0),
        ("imbalance_long_share", 1.5),
        ("imbalance_state_persistence", 1.0),
        ("demand_shape_gw.winter_weekday", (1.0, 2.0)),
        ("mild_shock_rate_per_night", 7.0),
        ("big_shock_rate_per_week", 22.0),
        ("shock_window_end_local_hour", 15.0),
        ("big_shock_min_hours", 0.75),
        ("mild_shock_max_hours", 0.25),
        ("big_shock_median_gw", 0.0),
        ("shock_max_gw", 0.0),
        ("price_floor_gbp_per_mwh", 5000.0),
        ("surprise_reveal_hours", -1.0),
        ("niv_surprise_shift", 1.5),
        ("solar_clearness_median", 1.0),
        ("solar_sunset_local_hour.summer", 4.0),
    ],
)
def test_rejects_invalid_assumptions(name: str, value: object) -> None:
    with pytest.raises(ValueError):
        make_paths(**{name: value})


def test_rejects_a_missing_assumption() -> None:
    values = defaults()
    del values["gate_closure_minutes"]
    with pytest.raises(ValueError, match="gate_closure_minutes"):
        generate_market_price_paths(
            np.random.default_rng(0),
            interval_start_utc=starts(),
            evaluation_world_count=1,
            daily_temperature_c=np.full((1, 7), 9.0),
            price_assumptions=values,
        )


# ---------------------------------------------------------------------------
# Realism checks on a seeded 100-week run (plan §4, B1 acceptance)
# ---------------------------------------------------------------------------


def _hundred_weeks(first: str) -> pd.DataFrame:
    """Default prices for 100 simulated weeks with sampled default weather."""

    rng = np.random.default_rng(42)
    inputs = assumptions.forecast_inputs(None, warmup_days=0, study_days=7)
    temperature = sample_daily_temperature(
        rng,
        world_count=100,
        base_temperature_c_by_day=inputs["base_temperature_c_by_day"],
        sd_c=inputs["weather_sd_c"],
    )
    forecast = generate_market_price_paths(
        rng,
        interval_start_utc=starts(first),
        evaluation_world_count=100,
        daily_temperature_c=temperature,
        price_assumptions=inputs["price_assumptions"],
    ).day_ahead
    local = forecast["interval_start_utc"].dt.tz_convert("Europe/London")
    return forecast.assign(date=local.dt.date, hour=local.dt.hour + local.dt.minute / 60)


def test_default_prices_look_like_gb_day_ahead_prices_on_the_review_checks() -> None:
    prices = _hundred_weeks("2026-10-05")
    price = prices["wholesale_forecast_gbp_per_mwh"]
    by_hour = prices.groupby("hour")["wholesale_forecast_gbp_per_mwh"].mean()
    daily_mean = prices.groupby(["world_id", "date"])["wholesale_forecast_gbp_per_mwh"].mean()

    assert 70.0 <= price.mean() <= 95.0
    assert 2.0 <= by_hour.idxmin() <= 5.0
    assert 16.0 <= by_hour.idxmax() <= 19.0
    assert daily_mean.quantile(0.9) - daily_mean.quantile(0.1) >= 20.0
    assert 0.01 <= (price < 0).mean() <= 0.04


def test_a_summer_run_uses_the_summer_shape_with_its_midday_solar_trough() -> None:
    # Decision 0004 item 53: the season is set by the start month.  The
    # Elexon summer weekday fit troughs at 13:30 (solar) and peaks at 20:00.
    prices = _hundred_weeks("2027-07-05")
    weekday = prices[prices["interval_start_utc"].dt.tz_convert("Europe/London").dt.dayofweek < 5]
    by_hour = weekday.groupby("hour")["wholesale_forecast_gbp_per_mwh"].mean()
    assert 12.0 <= by_hour.idxmin() <= 15.0
    assert 19.0 <= by_hour.idxmax() <= 21.0


def test_sunny_summer_weekends_have_the_most_negative_half_hours() -> None:
    # Item 58: solar pushes midday net demand down, so negative prices
    # concentrate on sunny summer weekend middays, and more solar gives more.
    def midday_weekend_negative_share(**overrides: float) -> float:
        prices = market(200, interval_start_utc=starts("2027-07-05"), **overrides)
        frame = prices.day_ahead
        local = frame["interval_start_utc"].dt.tz_convert("Europe/London")
        midday_weekend = (local.dt.dayofweek >= 5) & local.dt.hour.between(10, 15)
        return float((frame.loc[midday_weekend, "wholesale_forecast_gbp_per_mwh"] < 0).mean())

    share = midday_weekend_negative_share()
    overall = float(
        (
            market(200, interval_start_utc=starts("2027-07-05")).day_ahead[
                "wholesale_forecast_gbp_per_mwh"
            ]
            < 0
        ).mean()
    )
    assert share > 2 * overall
    assert midday_weekend_negative_share(solar_installed_gw=30.0) > share
