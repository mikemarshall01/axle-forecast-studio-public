"""The optional "timed" path (decision 0007): the normal rule, except home charging
is held off in a daily London wall-clock stretch from 12:00 until an editable start
hour, ``timed_start_local_hour``. Displayed as "Timed tariff"; no tariff price is
modelled and every path is costed the same way.

Hand-checkable cases first: the mask alone (no kernel), then kernel invariants on a
small hand-built fixture (the pattern ``test_smart_charging.py`` uses: one EV, a 10 kWh
battery, a 4 kW home charger, 100% charging efficiency, so every kWh can be checked by
eye), then one full-pipeline run through the contract validator.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from fixtures.result_contract import validate_result_v2

from axle_studio.model.action import smart_charging_inputs, timed_start_allowed
from axle_studio.model.clock import utc_half_hour_boundaries
from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.model.physics import PublicTopUp, simulate_fleet_intervals
from axle_studio.model.settings import RunSettings

_TOP_UP = PublicTopUp(efficiency_fraction=0.9, threshold_soc_fraction=0.1, target_soc_fraction=0.8)
_WINTER = date(2025, 1, 7)  # GMT, so London clock time equals UTC
_LONDON = "Europe/London"


# --------------------------------------------------------------------------
# The mask alone (action.timed_start_allowed): no kernel, no random draw
# --------------------------------------------------------------------------


def _settings(
    *, start: date = _WINTER, days: int = 2, warmup: int = 1, opening: float = 0.5
) -> RunSettings:
    return RunSettings(
        start_local_date=start,
        warmup_days=warmup,
        study_days=days,
        vehicle_count=1,
        seed=1,
        evaluation_world_count=1,
        opening_soc_fraction=opening,
        reserve_soc_fraction=0.1,
        home_charge_efficiency=1.0,
        weather_efficiency_sensitivity_fraction_per_c=0.0,
    )


def _london_slots(settings: RunSettings) -> pd.DatetimeIndex:
    """Every run half-hour's London wall clock, warm-up then study (the kernel's own order)."""

    starts = pd.DatetimeIndex(
        utc_half_hour_boundaries(
            settings.start_local_date, settings.warmup_days, settings.study_days
        )[:-1]
    )
    return starts.tz_convert(_LONDON)


def _expected_mask(london: pd.DatetimeIndex, start_hour: float) -> np.ndarray:
    """Hand-derived reference mask: barred from London 12:00 to ``start_hour``."""

    minutes_after_noon = (london.hour * 60 + london.minute) - 12 * 60
    position = np.asarray(minutes_after_noon) % (24 * 60)  # 0 at noon, wraps past midnight
    barred_minutes = (start_hour * 60 - 12 * 60) % (24 * 60)
    return position >= barred_minutes


@pytest.mark.parametrize("start_hour", [0.0, 23.5, 1.0])
def test_mask_matches_a_hand_derived_barred_window(start_hour: float) -> None:
    # T=0: barred 12:00-24:00, allowed 00:00-12:00. T=23:30: barred 12:00-23:30
    # (11.5 h), allowed the rest. T=01:00: barred 12:00-01:00 next day (13 h).
    settings = _settings(days=2)
    london = _london_slots(settings)
    mask = timed_start_allowed(settings, start_hour)
    np.testing.assert_array_equal(mask, _expected_mask(london, start_hour))


def test_an_18_00_plug_in_waits_for_midnight_and_a_01_00_plug_in_may_charge_at_once() -> None:
    settings = _settings(days=1)
    london = _london_slots(settings)
    mask = timed_start_allowed(settings, 0.0)
    by_time = dict(zip(london, mask, strict=True))

    evening = pd.Timestamp("2025-01-07 18:00", tz=_LONDON)
    midnight = pd.Timestamp("2025-01-08 00:00", tz=_LONDON)
    one_am = pd.Timestamp("2025-01-08 01:00", tz=_LONDON)
    assert not by_time[evening]
    assert by_time[midnight]
    assert by_time[one_am]


def test_mask_needs_no_special_handling_across_a_clock_change_week() -> None:
    # UK clocks go back on the last Sunday of October; the repeated 01:00-02:00
    # half-hour sits inside the always-allowed 00:00-12:00 stretch for T=0, so
    # both repeats read True with no special-casing in the mask builder.
    settings = _settings(start=date(2026, 10, 22), days=7, warmup=1)
    london = _london_slots(settings)
    assert len(london) == 48 * 8
    mask = timed_start_allowed(settings, 0.0)
    np.testing.assert_array_equal(mask, _expected_mask(london, 0.0))

    # 25 Oct 2026 holds 50 half-hour slots (one repeated hour) instead of the
    # usual 48, and 01:00 occurs twice (BST, then GMT); both occurrences sit
    # inside the always-allowed 00:00-12:00 stretch.
    fold_date = london.date == date(2026, 10, 25)
    assert fold_date.sum() == 50
    repeated_one_am = fold_date & (london.hour == 1) & (london.minute == 0)
    assert repeated_one_am.sum() == 2
    assert mask[repeated_one_am].all()


def test_mask_needs_no_special_handling_across_a_spring_forward_week() -> None:
    # UK clocks go forward on the last Sunday of March: London 01:00-02:00
    # does not exist that day (the clock jumps straight from 01:00 GMT to
    # 02:00 BST), so it holds 46 half-hour slots instead of the usual 48.
    # The gap sits inside the always-allowed 00:00-12:00 morning stretch for
    # T=0, so the mask needs no special-casing either side of it.
    settings = _settings(start=date(2026, 3, 23), days=7, warmup=1)
    london = _london_slots(settings)
    assert len(london) == 48 * 8
    mask = timed_start_allowed(settings, 0.0)
    np.testing.assert_array_equal(mask, _expected_mask(london, 0.0))

    spring_date = london.date == date(2026, 3, 29)
    assert spring_date.sum() == 46  # one skipped hour: 48 - 2
    before_gap = spring_date & (london.hour == 0) & (london.minute == 30)
    after_gap = spring_date & (london.hour == 2) & (london.minute == 0)
    assert before_gap.sum() == 1 and after_gap.sum() == 1
    assert mask[before_gap].all() and mask[after_gap].all()
    morning = spring_date & (london.hour < 12)
    assert morning.sum() == 22  # 24 half-hours less the skipped hour
    assert mask[morning].all()
    afternoon = spring_date & (london.hour >= 12)
    assert afternoon.sum() == 24
    assert not mask[afternoon].any()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (24.0, "between 0 and 23.5"),
        (-1.0, "between 0 and 23.5"),
        (1.25, "half-hour multiple"),
    ],
)
def test_invalid_start_hours_are_rejected(value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        timed_start_allowed(_settings(), value)


# --------------------------------------------------------------------------
# The kernel: barred home charging on the "timed" path
# --------------------------------------------------------------------------


def _units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population_id": ["population:timed"],
            "unit_id": ["ev-001"],
            "cohort_id": ["average_uk"],
            "physical_capacity_kwh": [10.0],
            "efficiency_miles_per_battery_kwh": [1.0],
            "home_charger_limit_kw": [4.0],
            "preferred_target_soc_fraction": [0.8],
            "runtime_departure_local_hour": [7],
            "runtime_arrival_local_hour": [18],
        }
    )


def _inputs(settings: RunSettings) -> dict[str, np.ndarray]:
    shape = (1, settings.sampled_day_count, 1)
    timestamps = np.full(shape, pd.NaT, dtype=object)
    return {
        "drives_today": np.zeros(shape, dtype=bool),
        "outbound_miles": np.zeros(shape),
        "return_miles": np.zeros(shape),
        "trip_departure_utc": timestamps.copy(),
        "destination_dwell_seconds": np.zeros(shape),
        "drive_speed_mph": np.full(shape, 30.0),
        "connection_session_accepted": np.zeros(shape, dtype=bool),
        "connection_start_utc": timestamps.copy(),
        "connection_end_utc": timestamps.copy(),
        "desired_pre_drive_soc_fraction": np.full(shape, 0.8),
    }


def _session(inputs: dict[str, np.ndarray], day: int, start: str, end: str) -> None:
    inputs["connection_session_accepted"][0, day, 0] = True
    inputs["connection_start_utc"][0, day, 0] = pd.Timestamp(start, tz="UTC")
    inputs["connection_end_utc"][0, day, 0] = pd.Timestamp(end, tz="UTC")


def _run_prices(settings: RunSettings, value: float = 100.0) -> np.ndarray:
    return np.full(48 * (settings.warmup_days + settings.study_days), value)


def _paths(fleet: pd.DataFrame) -> dict[str, pd.DataFrame]:
    grouped = {
        path: frame.sort_values("interval_start_utc").reset_index(drop=True)
        for path, frame in fleet.groupby("path_id")
    }
    for frame in grouped.values():
        assert frame["conservation_residual_kwh"].abs().max() <= 1e-9
    return grouped


def _run(
    settings: RunSettings,
    inputs: dict[str, np.ndarray],
    *,
    start_hour: float | None,
    with_smart: bool = False,
) -> dict[str, pd.DataFrame]:
    mask = None if start_hour is None else timed_start_allowed(settings, start_hour)
    smart = None
    if with_smart:
        prices = _run_prices(settings)
        smart = smart_charging_inputs(
            settings, _units(), prices[np.newaxis, :], departure_margin_hours=1.0
        )
    fleet = simulate_fleet_intervals(
        settings,
        _units(),
        inputs,
        public_top_up=_TOP_UP,
        smart_charging=smart,
        timed_start_allowed=mask,
    )[0]
    return _paths(fleet)


def test_no_home_import_in_any_barred_slot_across_two_nights() -> None:
    # Two ordinary overnight sessions (18:00-07:00) with a short commute
    # between them, so the EV needs charging on both mornings: without the
    # barred rule the timed path would charge from plug-in (18:00, which the
    # barred window already covers), so any import inside a barred
    # half-hour is unambiguously the rule failing to hold, and charging
    # must still happen somewhere (the allowed morning hours) or the fixture
    # would prove nothing.
    settings = _settings(days=2, warmup=1, opening=0.1)
    inputs = _inputs(settings)
    _session(inputs, 1, "2025-01-07 18:00", "2025-01-08 07:00")
    _session(inputs, 2, "2025-01-08 18:00", "2025-01-09 07:00")
    inputs["drives_today"][0, 2, 0] = True
    inputs["trip_departure_utc"][0, 2, 0] = pd.Timestamp("2025-01-08 08:00", tz="UTC")
    inputs["outbound_miles"][0, 2, 0] = 2.0
    inputs["return_miles"][0, 2, 0] = 2.0
    paths = _run(settings, inputs, start_hour=0.0)

    timed = paths["timed"]
    london = pd.DatetimeIndex(timed["interval_start_utc"]).tz_convert(_LONDON)
    barred = ~_expected_mask(london, 0.0)
    assert barred.any() and (~barred).any()
    assert (timed.loc[barred, "home_grid_import_kwh"] == 0.0).all()
    assert (timed.loc[~barred, "home_grid_import_kwh"] > 0.0).any()


def test_warmup_barring_leaves_less_battery_entering_the_study() -> None:
    # Decision 0007: the rule runs over the whole simulated run, warm-up
    # included, so the study opens from timed-start batteries. T=11:30
    # leaves only the single half-hour before each day's noon allowed, so
    # the one warm-up day (Jan6 12:00-Jan7 12:00) holds exactly one allowed
    # slot (Jan7 11:30-12:00); forcing that same warm-up unrestricted
    # instead lets the EV reach target long before the study starts.
    settings = _settings(days=1, warmup=1, opening=0.2)
    inputs = _inputs(settings)
    _session(inputs, 0, "2025-01-06 12:00", "2025-01-07 13:00")

    barred_warmup = timed_start_allowed(settings, 11.5)
    unrestricted_warmup = barred_warmup.copy()
    unrestricted_warmup[: 48 * settings.warmup_days] = True

    fleet_a = simulate_fleet_intervals(
        settings, _units(), inputs, public_top_up=_TOP_UP, timed_start_allowed=barred_warmup
    )[0]
    fleet_b = simulate_fleet_intervals(
        settings,
        _units(),
        inputs,
        public_top_up=_TOP_UP,
        timed_start_allowed=unrestricted_warmup,
    )[0]
    opening_a = _paths(fleet_a)["timed"]["opening_battery_kwh"].iat[0]
    opening_b = _paths(fleet_b)["timed"]["opening_battery_kwh"].iat[0]

    # Opening 2 kWh, target 8 kWh, 4 kW charger (2 kWh/slot): barred warm-up
    # gets one allowed slot (+2 kWh -> 4 kWh); unrestricted warm-up reaches
    # the 8 kWh target in three slots, long before the study starts.
    assert opening_a == pytest.approx(4.0)
    assert opening_b == pytest.approx(8.0)
    assert opening_a < opening_b


def test_a_session_too_short_to_fill_leaves_below_target_no_fallback() -> None:
    # Connected 11:00-13:30: five full half-hour slots, but only the first
    # two (11:00-12:00) are allowed before the daily barred stretch begins
    # at noon. Hard rule: the other three slots add nothing, even though
    # the EV is still connected and still below target.
    settings = _settings(days=1, warmup=1, opening=0.2)
    inputs = _inputs(settings)
    _session(inputs, 1, "2025-01-07 11:00", "2025-01-07 13:30")
    paths = _run(settings, inputs, start_hour=0.0)

    timed = paths["timed"]
    target_kwh = 0.8 * 10.0
    last_slot = timed.loc[timed["interval_start_utc"] == pd.Timestamp("2025-01-07 13:00", tz="UTC")]
    closing_at_unplug = last_slot["closing_battery_kwh"].iat[0]
    # Opening 2 kWh + two allowed slots (2 kWh each) = 6 kWh, short of the 8 kWh target.
    assert closing_at_unplug == pytest.approx(6.0)
    assert closing_at_unplug < target_kwh


def test_normal_and_selected_are_bit_for_bit_unchanged_with_the_setting_on_or_off() -> None:
    settings = _settings(days=2, warmup=1, opening=0.3)
    inputs = _inputs(settings)
    _session(inputs, 1, "2025-01-07 18:00", "2025-01-08 07:00")
    _session(inputs, 2, "2025-01-08 18:00", "2025-01-09 07:00")

    off = _run(settings, inputs, start_hour=None, with_smart=True)
    on = _run(settings, inputs, start_hour=0.0, with_smart=True)

    assert set(off) == {"normal", "selected"}
    assert set(on) == {"normal", "selected", "timed"}
    pdt.assert_frame_equal(off["normal"], on["normal"])
    pdt.assert_frame_equal(off["selected"], on["selected"])


# --------------------------------------------------------------------------
# Full pipeline: the four frames carry the optional path, everything else stays two
# --------------------------------------------------------------------------


def test_full_run_with_the_setting_on_passes_the_contract_validator_and_adds_one_path() -> None:
    values = {"vehicle_count": 12, "evaluation_world_count": 2, "timed_start_local_hour": 0.0}
    result = run_forecast_from_assumptions(date(2026, 1, 12), values=values)

    validate_result_v2(result)
    # The original slice-1 four, plus the model step 2 extension (decision
    # 0007): zones and charge completion also carry a path_id, and
    # weekly_peak_summary's path order is the run's own, so all of these gain
    # "timed" too. See tests/model/test_timed_policy.py for the full
    # frame-by-frame coverage of the step 2 extension; this test only needs
    # to stay accurate about which frames do and do not pick it up.
    for name in (
        "fleet_world_intervals",
        "cohort_world_intervals",
        "fleet_interval_bands",
        "cohort_interval_bands",
        "zone_world_intervals",
        "zone_import_bands",
        "zone_summary",
        "weekly_peak_summary",
        "charge_completion_summary",
    ):
        paths = set(getattr(result, name)["path_id"])
        assert paths == {"normal", "selected", "timed"}, name
    # Strictly pairwise frames (item 6): never pick up "timed", with or
    # without the model step 2 extension.
    for name in (
        "weekly_bands",
        "fleet_interval_ev_bands",
        "flexibility_bands",
        "average_day_bands",
    ):
        paths = set(getattr(result, name)["path_id"])
        assert paths == {"normal", "selected"}, name
    # price_band_shift (contract 4.6) has no path_id column: each path is a
    # column prefix instead.  Model step 2 adds timed_kwh_* columns here too.
    assert {"normal_kwh_p50", "selected_kwh_p50", "timed_kwh_p50"} <= set(
        result.price_band_shift.columns
    )


def test_full_run_with_the_setting_off_is_unchanged() -> None:
    # Decision 0007 follow-up (8 October 2026): the path is present when the
    # "Timed tariff policy" switch is on (the default, start 00:00); off
    # means the switch, not the hour field (a bounded number_input has no
    # reachable "unset" value in Streamlit).
    values = {
        "vehicle_count": 12,
        "evaluation_world_count": 2,
        "timed_tariff_enabled": 0,
    }
    result = run_forecast_from_assumptions(date(2026, 1, 12), values=values)

    validate_result_v2(result)
    for name in ("fleet_world_intervals", "cohort_world_intervals"):
        assert set(getattr(result, name)["path_id"]) == {"normal", "selected"}


def test_full_run_at_the_plain_defaults_carries_the_timed_path() -> None:
    # Decision 0007 follow-up: on by default (start 00:00), so a run with no
    # edits at all already has the third path.
    result = run_forecast_from_assumptions(
        date(2026, 1, 12), values={"vehicle_count": 12, "evaluation_world_count": 2}
    )
    validate_result_v2(result)
    assert set(result.fleet_world_intervals["path_id"]) == {"normal", "selected", "timed"}


def test_no_action_model_with_the_setting_on_carries_normal_and_timed_only() -> None:
    # The timed path is independent of the smart-charging model: a no-action
    # run has no "selected" path (contract rule 2), but still gets "timed"
    # when the start hour is set.
    result = run_forecast_from_assumptions(
        date(2026, 1, 12),
        model="no_action",
        values={"vehicle_count": 12, "evaluation_world_count": 2, "timed_start_local_hour": 0.0},
    )
    validate_result_v2(result)
    for name in (
        "fleet_world_intervals",
        "cohort_world_intervals",
        "fleet_interval_bands",
        "cohort_interval_bands",
    ):
        assert set(getattr(result, name)["path_id"]) == {"normal", "timed"}, name
