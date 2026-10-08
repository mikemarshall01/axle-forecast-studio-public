"""Scripted events: the table, presets, validation and what each event feeds.

Trading contract v1 §2 and §9.6 (decision 0004 items 55, 56 and 58).  The
first half checks ``model/events.py`` on its own with hand-checkable
numbers; the second runs one hand-built EV through the kernel to show a
grid request and a control outage change the smart plan only from the
right instant and for the right EVs.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from axle_studio.model import events as ev
from axle_studio.model.action import smart_charging_inputs
from axle_studio.model.assumptions import EVENT_PRESETS, ZONE_IDS
from axle_studio.model.physics import PublicTopUp, simulate_fleet_intervals
from axle_studio.model.sampling import trapezoid_weight
from axle_studio.model.settings import RunSettings
from axle_studio.model.summaries import build_study_slots, build_warmup_slots

_MONDAY = date(2026, 1, 12)  # GMT, so London clock time equals UTC
_STUDY = build_study_slots(_MONDAY)


def _row(**changes: object) -> dict[str, object]:
    row = {
        "event_id": "e1",
        "event_type": "turn_down",
        "enabled": True,
        "night_index": 1,
        "start_local_time": "17:30",
        "duration_minutes": 60,
        "size": np.nan,
        "notice": "day_ahead",
        "notice_minutes": None,
        "scope": "national",
        "payment_gbp_per_mwh": 500.0,
    }
    return row | changes


def _table(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _valid(*rows: dict[str, object], slots: pd.DataFrame = _STUDY) -> pd.DataFrame:
    return ev.validate_events(_table(*rows), slots, ZONE_IDS)


# --------------------------------------------------------------------------
# Presets and validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("start", [_MONDAY, date(2026, 3, 26), date(2026, 10, 22)])
def test_every_preset_validates_including_across_clock_changes(start: date) -> None:
    slots = build_study_slots(start)
    table = ev.event_table(list(EVENT_PRESETS), slots)
    valid = ev.validate_events(table, slots, ZONE_IDS)
    assert list(valid.columns) == list(ev.EVENT_COLUMNS)
    assert valid["enabled"].all()
    assert str(valid["notice_minutes"].dtype) == "Int64"
    units = dict(zip(valid["event_type"], valid["size_unit"], strict=False))
    assert units == {
        "price_shock_known": "GW",
        "price_shock_surprise": "GW",
        "turn_down": "MW",
        "turn_up": "MW",
        "control_outage": "fraction",
    }
    # Cold still week: one row per night with unique ids (lead decision Q7).
    week = valid.loc[valid["event_id"].str.startswith("cold_still_week")]
    assert list(week["night_index"]) == list(range(7))


@pytest.mark.parametrize(
    ("start", "nights"),
    [
        (_MONDAY, [4, 5]),  # Fri and Sat nights: Saturday and Sunday mornings
        (date(2026, 1, 8), [1, 2]),  # a Thursday start
        (date(2026, 1, 10), [0]),  # Saturday start: the next Saturday runs past the study
        (date(2026, 1, 11), [5]),  # Sunday start: the last Sunday runs past the study
    ],
)
def test_sunny_weekend_has_one_row_per_weekend_day_inside_the_study(
    start: date, nights: list[int]
) -> None:
    slots = build_study_slots(start)
    rows = ev.validate_events(ev.preset_rows("sunny_negative_weekend", slots), slots, ZONE_IDS)
    assert list(rows["night_index"]) == nights
    windows = ev.event_slots(rows, slots)
    assert (windows["end_slot"] <= len(slots)).all()
    london = windows["start_utc"].dt.tz_convert("Europe/London")
    assert set(london.dt.day_name()) <= {"Saturday", "Sunday"}
    assert (london.dt.strftime("%H:%M") == "10:00").all()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"event_type": "blackout"}, "event_type"),
        ({"night_index": 7}, "night_index"),
        ({"start_local_time": "17:15"}, "start_local_time"),
        ({"duration_minutes": 45}, "duration_minutes"),
        ({"duration_minutes": 750}, "duration_minutes"),
        ({"night_index": 6, "start_local_time": "11:00", "duration_minutes": 120}, "study end"),
        ({"payment_gbp_per_mwh": -1.0}, "payment_gbp_per_mwh"),
        ({"size": 0.0}, "size"),
        ({"scope": "zone_9"}, "scope"),
        ({"notice": "short", "notice_minutes": 60}, "not modelled yet"),
        ({"notice": "day_ahead", "notice_minutes": 30}, "notice_minutes"),
        (
            {"event_type": "price_shock_known", "size": 4.0, "payment_gbp_per_mwh": np.nan}
            | {"scope": "zone_1"},
            "price shocks are national",
        ),
        (
            {"event_type": "price_shock_known", "size": 0.0, "payment_gbp_per_mwh": np.nan},
            "non-zero",
        ),
        (
            {"event_type": "price_shock_known", "size": 12.0, "payment_gbp_per_mwh": np.nan},
            "non-zero",
        ),
        (
            {"event_type": "price_shock_known", "size": 4.0, "payment_gbp_per_mwh": np.nan}
            | {"notice": "short", "notice_minutes": 60},
            "known day-ahead",
        ),
        (
            {"event_type": "price_shock_surprise", "size": 4.0, "payment_gbp_per_mwh": np.nan},
            "short notice",
        ),
        (
            {"event_type": "control_outage", "size": 1.5, "payment_gbp_per_mwh": np.nan}
            | {"notice": "short", "notice_minutes": 0},
            "share of sessions",
        ),
        (
            {"event_type": "control_outage", "size": 0.5, "payment_gbp_per_mwh": np.nan},
            "known to nobody",
        ),
    ],
)
def test_validation_names_the_row_and_column(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message) as error:
        _valid(_row(**changes))
    assert "events row 0" in str(error.value)


def test_skipped_spring_hour_is_rejected() -> None:
    # Night 1 of a study from Fri 27 Mar 2026 is Sat 28 Mar; 01:00 names Sun
    # 29 Mar 01:00 London, which the spring change skips.
    slots = build_study_slots(date(2026, 3, 27))
    with pytest.raises(ValueError, match="does not exist"):
        ev.validate_events(_table(_row(start_local_time="01:00")), slots, ZONE_IDS)


def test_overlapping_requests_are_rejected_only_when_their_scopes_overlap() -> None:
    national = _row()
    local = _row(event_id="e2", scope="zone_2", start_local_time="18:00")
    with pytest.raises(ValueError, match="overlaps request"):
        _valid(national, local)
    # Two different zones may overlap; a disabled row is ignored.
    _valid(_row(scope="zone_1"), local)
    _valid(_row(enabled=False), local)
    # Two scripted surprises may not overlap: one reveal per slot.
    surprise = _row(
        event_type="price_shock_surprise",
        size=3.0,
        payment_gbp_per_mwh=np.nan,
        notice="short",
        notice_minutes=90,
    )
    with pytest.raises(ValueError, match="overlaps surprise"):
        _valid(surprise, surprise | {"event_id": "e2", "start_local_time": "18:00"})


def test_duplicate_ids_and_missing_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="used twice"):
        _valid(_row(), _row(night_index=2))
    with pytest.raises(ValueError, match="missing columns"):
        ev.validate_events(_table(_row()).drop(columns="scope"), _STUDY, ZONE_IDS)


def test_empty_table_is_valid_and_typed() -> None:
    valid = ev.validate_events(ev.empty_events(), _STUDY, ZONE_IDS)
    assert valid.empty and list(valid.columns) == list(ev.EVENT_COLUMNS)
    assert ev.event_slots(valid, _STUDY).empty


# --------------------------------------------------------------------------
# Timing and notice
# --------------------------------------------------------------------------


def test_day_ahead_notice_is_the_nights_day_ahead_decision_and_short_notice_counts_back() -> None:
    table = _valid(
        _row(),  # night 1 (Tue 13 Jan) 17:30
        _row(event_id="morning", start_local_time="02:00", night_index=0),  # Tue 02:00
        _row(
            event_id="spike",
            event_type="price_shock_surprise",
            night_index=4,
            start_local_time="17:00",
            size=3.0,
            payment_gbp_per_mwh=np.nan,
            notice="short",
            notice_minutes=90,
        ),
    )
    windows = ev.event_slots(table, _STUDY).set_index("event_id")
    # Night 1's day-ahead decision is 13:00 on Mon 12 Jan: study slot 2.
    assert pd.Timestamp(windows.at["e1", "notice_utc"], tz="UTC") == pd.Timestamp(
        "2026-01-12 13:00", tz="UTC"
    )
    assert windows.at["e1", "notice_slot"] == 2
    assert windows.at["e1", "start_slot"] == 48 + 11 and windows.at["e1", "end_slot"] == 48 + 13
    # A morning event of night 0 is known at night 0's decision, the day
    # before the study (review B2): before the study frame, so slot 0.
    assert windows.at["morning", "start_slot"] == 28
    assert pd.Timestamp(windows.at["morning", "notice_utc"], tz="UTC") == pd.Timestamp(
        "2026-01-11 13:00", tz="UTC"
    )
    assert windows.at["morning", "notice_slot"] == 0
    # A short-notice event is known 90 minutes before 17:00 on Fri 16 Jan.
    assert windows.at["spike", "notice_slot"] == windows.at["spike", "start_slot"] - 3
    # On a warm-up + study frame the same events sit 48 x warm-up slots later.
    run = pd.concat([build_warmup_slots(_MONDAY, 2), _STUDY], ignore_index=True)
    shifted = ev.event_slots(table, run).set_index("event_id")
    assert shifted.at["e1", "start_slot"] == windows.at["e1", "start_slot"] + 96
    assert shifted.at["morning", "notice_slot"] == 96 - 23 * 2  # Sun 13:00 in the warm-up


def test_scripted_profiles_use_the_item_56_trapezoid_and_route_by_kind() -> None:
    table = ev.validate_events(
        ev.event_table(["cold_still_evening", "surprise_evening_spike"], _STUDY), _STUDY, ZONE_IDS
    )
    known, surprise = ev.scripted_shock_profiles(table, _STUDY, len(_STUDY))
    windows = ev.event_slots(table, _STUDY).set_index("event_id")
    start = windows.at["cold_still_evening", "start_slot"]
    # Six half-hours: ramp over the first quarter, hold, ramp down (4 GW).
    np.testing.assert_allclose(known[start : start + 6], 4.0 * np.array([0.5, 1, 1, 1, 1, 0.5]))
    np.testing.assert_allclose(
        known[start : start + 6], 4.0 * trapezoid_weight(np.arange(6), np.full(6, 6))
    )
    assert known.sum() == pytest.approx(4.0 * 5) and known[:start].sum() == 0.0
    spike = windows.at["surprise_evening_spike", "start_slot"]
    assert surprise[spike : spike + 4] == pytest.approx(3.0 * np.array([0.5, 1, 1, 0.5]))
    assert np.count_nonzero(surprise) == 4
    # Revealed 90 minutes before 17:00 with a 60-minute gate: half-hour k is
    # known from floor(k/2 + 0.5) whole hours before its gate closure.
    reveal = ev.scripted_surprise_reveal_hours(
        table, _STUDY, len(_STUDY), gate_closure_minutes=60.0
    )
    np.testing.assert_array_equal(reveal[spike : spike + 4], [0.0, 1.0, 1.0, 2.0])
    empty = ev.validate_events(ev.empty_events(), _STUDY, ZONE_IDS)
    for profile in ev.scripted_shock_profiles(empty, _STUDY, len(_STUDY)):
        assert not profile.any()


def test_planner_adjustments_sign_scope_and_notice() -> None:
    table = _valid(
        _row(),
        _row(
            event_id="up",
            event_type="turn_up",
            night_index=5,
            start_local_time="23:00",
            duration_minutes=120,
            scope="zone_2",
            payment_gbp_per_mwh=60.0,
        ),
        _row(event_id="off", night_index=3, enabled=False),
    )
    notice, adjustment = ev.planner_adjustments(table, _STUDY, len(ZONE_IDS), len(_STUDY))
    assert adjustment.shape == (2, 4, len(_STUDY))
    windows = ev.event_slots(table, _STUDY)
    np.testing.assert_array_equal(notice, windows["notice_slot"])
    down, up = windows.iloc[0], windows.iloc[1]
    # Turn-down: +payment in every zone inside the window, 0 elsewhere.
    assert (adjustment[0][:, down.start_slot : down.end_slot] == 500.0).all()
    assert adjustment[0].sum() == 500.0 * 4 * 2
    # Turn-up: a -60 bonus in zone_2 only.
    assert (adjustment[1][1, up.start_slot : up.end_slot] == -60.0).all()
    assert adjustment[1][[0, 2, 3]].sum() == 0.0


def test_trading_mask_uses_only_requests_known_at_the_decision() -> None:
    table = _valid(_row())
    windows = ev.event_slots(table, _STUDY)
    window = slice(int(windows["start_slot"].iat[0]), int(windows["end_slot"].iat[0]))
    notice = int(windows["notice_utc"].iat[0])
    assert ev.trading_mask(windows, len(_STUDY), known_at_utc=notice - 1).all()
    at_notice = ev.trading_mask(windows, len(_STUDY), known_at_utc=notice)
    assert not at_notice[window].any() and at_notice.sum() == len(_STUDY) - 2
    np.testing.assert_array_equal(
        ev.trading_mask(windows, len(_STUDY), known_at_utc=None), at_notice
    )


def test_event_delivery_pays_capped_turn_down_against_the_baseline() -> None:
    # One world; scope 0 is the fleet.  Turn-down 17:30-18:30 (two slots),
    # capped at 0.01 MW = 5 kWh per slot, paid 500 GBP/MWh.
    table = _valid(_row(size=0.01))
    window = ev.event_slots(table, _STUDY)
    start = int(window["start_slot"].iat[0])
    baseline = np.zeros((1, 5, len(_STUDY)))
    metered = np.zeros((1, 5, len(_STUDY)))
    baseline[0, 0, start : start + 2] = [10.0, 3.0]
    metered[0, 0, start : start + 2] = [2.0, 4.0]
    delivery = ev.event_delivery(table, _STUDY, baseline, metered)
    assert list(delivery.columns) == list(ev.EVENT_DELIVERY_COLUMNS)
    row = delivery.iloc[0]
    # Delivered B - M = 8 and -1 kWh; paid min(max(0, .), 5) = 5 + 0 kWh.
    assert row.delivered_kwh == pytest.approx(7.0)
    assert row.paid_kwh == pytest.approx(5.0)
    assert row.payment_gbp == pytest.approx(5.0 / 1000.0 * 500.0)
    assert row.night_index == 1
    # A turn-up pays on M - B, in its own zone's scope.
    up = _valid(_row(event_type="turn_up", scope="zone_3"))
    metered[0, 3, start] = 6.0
    turn_up = ev.event_delivery(up, _STUDY, baseline, metered).iloc[0]
    assert turn_up.delivered_kwh == pytest.approx(6.0) and turn_up.paid_kwh == pytest.approx(6.0)


def test_outage_probability_is_the_largest_share_by_zone_and_plug_in_slot() -> None:
    outage = dict(
        event_type="control_outage",
        size=0.5,
        payment_gbp_per_mwh=np.nan,
        notice="short",
        notice_minutes=0,
        night_index=3,
        start_local_time="15:00",
        duration_minutes=720,
    )
    table = _valid(
        _row(event_id="a", **outage), _row(event_id="b", **outage | {"size": 0.8}, scope="zone_4")
    )
    probability = ev.outage_probability(table, _STUDY, len(ZONE_IDS), len(_STUDY))
    windows = ev.event_slots(table, _STUDY)
    first, last = int(windows["start_slot"].iat[0]), int(windows["end_slot"].iat[0])
    assert (probability[:3, first:last] == 0.5).all() and (probability[3, first:last] == 0.8).all()
    assert probability[:, :first].sum() == 0.0 and probability[:, last:].sum() == 0.0


# --------------------------------------------------------------------------
# One EV through the kernel: requests and outages move the plan
# --------------------------------------------------------------------------

_TOP_UP = PublicTopUp(efficiency_fraction=0.9, threshold_soc_fraction=0.1, target_soc_fraction=0.8)
_WINTER = date(2025, 1, 7)


def _settings() -> RunSettings:
    return RunSettings(
        start_local_date=_WINTER,
        warmup_days=1,
        study_days=2,
        vehicle_count=1,
        seed=1,
        evaluation_world_count=1,
        opening_soc_fraction=0.6,
        reserve_soc_fraction=0.1,
        home_charge_efficiency=1.0,
        weather_efficiency_sensitivity_fraction_per_c=0.0,
    )


def _units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "population_id": ["population:events"],
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
    # One session, Tue 7 Jan 18:00 to Wed 8 Jan 07:00 (day 1 is the study
    # start date after one warm-up day); no trips.
    shape = (1, settings.sampled_day_count, 1)
    timestamps = np.full(shape, pd.NaT, dtype=object)
    inputs = {
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
    inputs["connection_session_accepted"][0, 1, 0] = True
    inputs["connection_start_utc"][0, 1, 0] = pd.Timestamp("2025-01-07 18:00", tz="UTC")
    inputs["connection_end_utc"][0, 1, 0] = pd.Timestamp("2025-01-08 07:00", tz="UTC")
    return inputs


def _run_slot(instant: str) -> int:
    """Run slot (one warm-up day first) of a UTC instant; the study starts 7 Jan 12:00."""

    return 48 + int(
        (pd.Timestamp(instant, tz="UTC") - pd.Timestamp("2025-01-07 12:00", tz="UTC"))
        // pd.Timedelta(minutes=30)
    )


def _selected_imports(**smart_inputs: object) -> dict[str, float]:
    """Selected-path home import by UTC slot start, for one EV needing 2 kWh."""

    settings = _settings()
    prices = np.full(48 * 3, 100.0)
    prices[_run_slot("2025-01-08 03:00")] = 10.0
    prices[_run_slot("2025-01-08 04:00")] = 20.0
    smart = smart_charging_inputs(
        settings, _units(), prices[np.newaxis, :], departure_margin_hours=1.0, **smart_inputs
    )
    fleet = simulate_fleet_intervals(
        settings, _units(), _inputs(settings), public_top_up=_TOP_UP, smart_charging=smart
    )[0]
    selected = fleet.loc[fleet["path_id"].eq("selected")]
    charged = selected.loc[selected["home_grid_import_kwh"] > 1e-12]
    return dict(
        zip(
            charged["interval_start_utc"].dt.strftime("%d %H:%M"),
            charged["home_grid_import_kwh"],
            strict=True,
        )
    )


def _turn_down_at_three(notice_slot: int, zone: int = 0) -> dict[str, np.ndarray]:
    adjustment = np.zeros((1, 2, 48 * 3))
    adjustment[0, zone, _run_slot("2025-01-08 03:00")] = 500.0
    return {
        "zone_index": np.array([0]),
        "event_notice_slot": np.array([notice_slot]),
        "event_adjustment_gbp_per_mwh": adjustment,
    }


def test_a_known_turn_down_moves_the_plan_out_of_its_window() -> None:
    # Without the request the EV charges its 2 kWh at 03:00 (10 GBP/MWh).
    assert _selected_imports() == {"08 03:00": 2.0}
    # Announced before the 18:00 plug-in: 03:00 now ranks at 510, so 04:00.
    assert _selected_imports(**_turn_down_at_three(_run_slot("2025-01-07 13:00"))) == {
        "08 04:00": 2.0
    }
    # A turn-up bonus pulls the plan into its window instead.
    bonus = _turn_down_at_three(_run_slot("2025-01-07 13:00"))
    bonus["event_adjustment_gbp_per_mwh"] = np.zeros((1, 1, 48 * 3))
    bonus["event_adjustment_gbp_per_mwh"][0, 0, _run_slot("2025-01-08 01:00")] = -500.0
    assert _selected_imports(**bonus) == {"08 01:00": 2.0}


def test_no_plan_sees_a_request_before_its_notice_or_outside_its_zone() -> None:
    # Announced after the plug-in: the plan made at 18:00 keeps its ranking.
    late = _turn_down_at_three(_run_slot("2025-01-07 18:30"))
    assert _selected_imports(**late) == {"08 03:00": 2.0}
    # A request for another zone does not reach this EV (zone 0).
    other_zone = _turn_down_at_three(_run_slot("2025-01-07 13:00"), zone=1)
    assert _selected_imports(**other_zone) == {"08 03:00": 2.0}


@pytest.mark.parametrize(
    ("share", "uniform", "expected"),
    [
        (1.0, 0.99, {"07 18:00": 2.0}),  # everyone covered ignores the plan
        (0.5, 0.3, {"07 18:00": 2.0}),  # below the share: full power at plug-in
        (0.5, 0.7, {"08 03:00": 2.0}),  # above it: the plan is followed
        (0.0, 0.0, {"08 03:00": 2.0}),  # no outage
    ],
)
def test_control_outage_makes_covered_sessions_charge_at_plug_in(
    share: float, uniform: float, expected: dict[str, float]
) -> None:
    outage = np.zeros((1, 48 * 3))
    outage[0, _run_slot("2025-01-07 18:00")] = share
    imports = _selected_imports(
        zone_index=np.array([0]),
        outage_probability=outage,
        non_response_uniform=np.full((1, 2, 1), uniform),
    )
    assert imports == expected


def test_outage_needs_the_uniforms_and_inputs_are_shape_checked() -> None:
    settings = _settings()
    prices = np.full((1, 48 * 3), 100.0)
    outage = np.full((1, 48 * 3), 0.5)
    with pytest.raises(ValueError, match="uniforms"):
        smart_charging_inputs(
            settings, _units(), prices, departure_margin_hours=1.0, outage_probability=outage
        )
    with pytest.raises(ValueError, match="world, study night and EV"):
        smart_charging_inputs(
            settings,
            _units(),
            prices,
            departure_margin_hours=1.0,
            non_response_uniform=np.zeros((1, 3, 1)),
        )
