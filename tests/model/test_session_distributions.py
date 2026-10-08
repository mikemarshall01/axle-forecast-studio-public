"""Session distributions per archetype (decision 0004 item 54, plan D-2, contract 3.10c)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fixtures.result_fixture import make_result, validate_result_v2

from axle_studio.model import assumptions
from axle_studio.model.forecast import run_forecast_from_assumptions
from axle_studio.model.summaries import (
    SESSION_DISTRIBUTION_METRICS,
    build_study_slots,
    find_sessions,
    session_distribution_bands,
    session_distribution_rows,
)


def _hand_rows() -> pd.DataFrame:
    """Two worlds, two EVs of different archetypes, sessions laid out by hand.

    Study from Monday 12 January 2026 (GMT, so London = UTC and slot k starts
    at k x 30 min after Monday 00:00).  60 kWh batteries, 48 kWh target (80%),
    7 kW charger, 90% efficiency, so a battery-side 6.3 kW.
    """

    slots = build_study_slots(date(2026, 1, 12))
    connected = np.zeros((2, len(slots), 2), dtype=bool)
    closing = np.full((2, len(slots), 2), 30.0)
    # The study starts at London noon (decision 0004 item 52), so slot 0 is
    # Monday 12:00.  World 0, EV 0: plugs in Monday 18:00 (slot 12) at 30 kWh
    # (50%), unplugs Tuesday 07:00 (slot 38): dwell 13 h, need 18 kWh, 18 / 6.3 = 2.857 h to
    # charge, slack 10.14 h, flexible 18 kWh.
    connected[0, 12:38, 0] = True
    # World 0, EV 1: plugs in Saturday 17:30 (slot 5 x 48 + 11 = 251) at 45.5
    # kWh and is still plugged at the horizon end: departure, dwell, flexible
    # and slack are unobserved.
    connected[0, 251:, 1] = True
    closing[0, 250, 1] = 45.5
    # World 1, EV 0: a 1-hour session at 0 kWh (slots 16-17, Monday 20:00):
    # need 48 kWh, flexible 1 h x 6.3 = 6.3 kWh, slack 1 - 7.62 = -6.62 h.
    connected[1, 16:18, 0] = True
    closing[1, 15, 0] = 0.0
    # Already running at the horizon start: no observed plug-in, left out.
    connected[1, 0:10, 1] = True
    sessions = find_sessions(connected, closing, np.zeros_like(closing))
    return session_distribution_rows(
        sessions,
        study_slots=slots,
        world_ids=np.array([0, 1]),
        cohort_codes=np.array([0, 1]),
        capacity_kwh=np.array([60.0, 60.0]),
        target_kwh=np.array([48.0, 48.0]),
        power_kw=np.array([7.0, 7.0]),
        efficiency=0.9,
    )


def _bin(metric: str, value: float) -> int:
    _, lower, width, count = SESSION_DISTRIBUTION_METRICS[metric]
    return int(np.clip(np.floor((value - lower) / width), 0, count - 1))


def test_rows_bin_each_session_by_hand() -> None:
    rows = _hand_rows().sort_values(["world_id", "cohort_code"]).reset_index(drop=True)
    assert len(rows) == 3  # the session running at the horizon start is not a plug-in
    first, open_session, short = rows.iloc[0], rows.iloc[1], rows.iloc[2]
    assert first["plug_in_time"] == 36 and first["departure_time"] == 14  # 18:00, 07:00
    assert first["plug_in_soc_percent"] == 10  # 50% in [50, 55)
    assert first["energy_needed_kwh"] == _bin("energy_needed_kwh", 18.0) == 3
    assert first["dwell_hours"] == _bin("dwell_hours", 13.0) == 6
    assert first["flexible_kwh"] == 3
    assert first["slack_hours"] == _bin("slack_hours", 13.0 - 18.0 / 6.3) == 11
    assert not first["weekend"]
    # Saturday plug-in at 17:30 with 45.5 kWh: open, so no departure metrics.
    assert open_session["weekend"] and open_session["plug_in_time"] == 35
    assert open_session["energy_needed_kwh"] == 0  # 2.5 kWh
    for metric in ("departure_time", "dwell_hours", "flexible_kwh", "slack_hours"):
        assert open_session[metric] == -1
    # 0 kWh: SoC bin 0, need 48 kWh, flexible 6.3 kWh, slack -6.62 h.
    assert short["plug_in_soc_percent"] == 0 and short["energy_needed_kwh"] == 9
    assert short["flexible_kwh"] == 1
    assert short["slack_hours"] == _bin("slack_hours", 1.0 - 48.0 / 6.3) == 2


def test_an_at_target_session_needs_nothing_and_its_slack_is_its_dwell() -> None:
    # Charging exactly to target leaves float residue; it must not move a
    # session across a bin edge (need 0 kWh, slack = dwell = 2 h exactly).
    slots = build_study_slots(date(2026, 1, 12))
    connected = np.zeros((1, len(slots), 1), dtype=bool)
    connected[0, 12:16, 0] = True  # Monday 18:00-20:00
    closing = np.full((1, len(slots), 1), 48.0 - 1e-14)
    sessions = find_sessions(connected, closing, np.zeros_like(closing))
    rows = session_distribution_rows(
        sessions,
        study_slots=slots,
        world_ids=np.array([0]),
        cohort_codes=np.array([0]),
        capacity_kwh=np.array([60.0]),
        target_kwh=np.array([48.0]),
        power_kw=np.array([7.0]),
        efficiency=0.9,
    )
    assert rows["energy_needed_kwh"].iat[0] == 0
    assert rows["slack_hours"].iat[0] == _bin("slack_hours", 2.0) == 7
    assert rows["plug_in_soc_percent"].iat[0] == 16  # 80% in [80, 85)


def test_bands_are_per_world_shares_then_quantiles_across_worlds() -> None:
    rows = _hand_rows()
    bands = session_distribution_bands(rows, 2, {"fleet": None, "a": 0, "b": 1})
    key = ["group_id", "day_type", "metric", "bin_index"]
    indexed = bands.set_index(key)
    # Fleet, all days, plug-in time: world 0 has 18:00 and 17:30 (1/2 each),
    # world 1 has 20:00 (1).  18:00 bin: shares 0.5 and 0 -> mean 0.25.
    six = indexed.loc[("fleet", "all", "plug_in_time", 36)]
    assert six["world_count"] == 2 and six["bin_label"] == "18:00"
    assert six[["share_mean", "share_p10", "share_p50", "share_p90"]].tolist() == pytest.approx(
        [0.25, 0.05, 0.25, 0.45]
    )
    # Departure is observed in both worlds (world 0 only for its closed session).
    seven = indexed.loc[("fleet", "all", "departure_time", 14)]
    assert seven["world_count"] == 2 and seven["share_mean"] == pytest.approx(0.5)
    # Archetype b has only the open session: no departure in any world, NaN.
    missing = bands.loc[bands["group_id"].eq("b") & bands["metric"].eq("dwell_hours")]
    assert missing["world_count"].eq(0).all() and missing["share_mean"].isna().all()
    # Weekday excludes the Saturday session.
    weekday = indexed.loc[("fleet", "weekday", "plug_in_time", 35)]
    assert weekday["share_mean"] == 0.0
    # Shares sum to 1 per group, day type and metric where any world has one.
    present = bands.loc[bands["world_count"] > 0]
    assert np.allclose(present.groupby(["group_id", "day_type", "metric"])["share_mean"].sum(), 1)
    labels = indexed["bin_label"]
    assert labels.loc[("fleet", "all", "slack_hours", 0)] == "< -10"
    assert labels.loc[("fleet", "all", "slack_hours", 29)] == "≥ 46"
    assert labels.loc[("fleet", "all", "plug_in_soc_percent", 19)] == "95–100"


def test_source_reference_records_match_the_workbook_cells() -> None:
    expected = {
        "average_uk": (68.0, 7.0, 6),
        "intelligent_octopus": (52.0, 22.0, 7),
        "infrequent_charging": (18.0, 37.0, 8),
        "infrequent_driving": (73.0, 4.0, 9),
        "scheduled_charging": (68.0, 7.0, 10),
        "always_plugged_in": (68.0, 7.0, 11),
    }
    for cohort_id, (soc, kwh, row) in expected.items():
        records = assumptions.SESSION_SOURCE_REFERENCES[cohort_id]
        soc_record = records["source_plug_in_soc_percent"]
        kwh_record = records["source_battery_kwh_per_plug_in"]
        assert soc_record.value == soc and soc_record.source.startswith(
            f"'Source archetypes'!N{row}"
        )
        assert kwh_record.value == kwh and kwh_record.source == f"'Source archetypes'!M{row}"
        for record in (soc_record, kwh_record):
            assert record.evidence == "source" and not record.editable


def test_real_run_distributions_pass_the_contract_and_look_plausible() -> None:
    result = run_forecast_from_assumptions(
        date(2026, 1, 12),
        model="no_action",
        values={"vehicle_count": 60, "evaluation_world_count": 3},
    )
    # Every world is sampled, so the validator recomputes the frame exactly.
    validate_result_v2(result)
    names = {record.name for record in result.assumptions}
    assert "intelligent_octopus.source_plug_in_soc_percent" in names
    bands = result.session_distribution_bands
    fleet = bands.loc[bands["group_id"].eq("fleet") & bands["day_type"].eq("weekday")]
    plug_in = fleet.loc[fleet["metric"].eq("plug_in_time")].set_index("bin_index")
    # Weekday plug-ins cluster in the evening around the 18:00 arrival.
    evening = plug_in.loc[30:42, "share_mean"].sum()
    assert evening > 0.6
    # Nothing charges beyond the preferred target: a session never needs more
    # than the largest target (72.5 kWh x 80% = 58 kWh, bin 11).
    need = fleet.loc[fleet["metric"].eq("energy_needed_kwh")].set_index("bin_index")
    assert need.loc[12:, "share_mean"].eq(0.0).all()

    # The validator catches a missing source record and a wrong share.
    partial = tuple(
        r for r in result.assumptions if r.name != "average_uk.source_plug_in_soc_percent"
    )
    with pytest.raises(AssertionError, match="average_uk.source_plug_in_soc_percent"):
        validate_result_v2(replace(result, assumptions=partial))
    broken = bands.copy()
    broken.loc[broken["metric"].eq("dwell_hours") & broken["bin_index"].eq(3), "share_p90"] += 0.01
    with pytest.raises(AssertionError, match="session_distribution_bands"):
        validate_result_v2(replace(result, session_distribution_bands=broken))


def test_fixture_bin_labels_match_the_model() -> None:
    # Views are built against the fixture, so its labels must be the model's.
    fixture = make_result("no_action").session_distribution_bands
    model = session_distribution_bands(
        _hand_rows(),
        2,
        {"fleet": None, **{c: i for i, c in enumerate(fixture["group_id"].unique()[1:])}},
    )
    keys = ["group_id", "day_type", "metric", "bin_index"]
    joined = fixture.merge(model, on=keys, suffixes=("_fixture", "_model"))
    assert len(joined) > 0
    assert (joined["bin_label_fixture"] == joined["bin_label_model"]).all()
    assert (joined["unit_fixture"] == joined["unit_model"]).all()
