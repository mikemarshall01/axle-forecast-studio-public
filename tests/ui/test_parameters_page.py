"""AppTest checks of the Edit assumptions dialog body (views/parameters.py)."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from axle_studio.model import assumptions
from axle_studio.ui import run_controller as rc
from axle_studio.ui.views import parameters

APP_PATH = Path(__file__).parents[2] / "streamlit_app.py"


def _open_dialog(app: AppTest) -> None:
    app.get_by_key("edit-assumptions").click().run()
    assert not app.exception


def _navigate(app: AppTest, page_name: str) -> None:
    app._page_hash = next(
        page_hash
        for page_hash, page in app._registered_pages.items()
        if page["page_name"] == page_name
    )
    app.run()
    assert not app.exception


def _show(app: AppTest, key: str) -> None:
    """Pick the group pill that holds field ``key``; only that group's fields are drawn."""

    # AppTest reruns the whole script on a pill click, which closes the
    # dialog (a browser reruns only the dialog), so reopen it: the dialog
    # remembers the group last chosen.
    name = key.removeprefix("assumption::")
    group = next(record.group for record in assumptions.editable_records() if record.name == name)
    if app.get_by_key(parameters.GROUP_KEY).value != group:
        app.get_by_key(parameters.GROUP_KEY).set_value(group).run()
        _open_dialog(app)
    assert not app.exception


def _app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


def _widget_keys(app: AppTest) -> set[str]:
    return {widget.key for widget in app if getattr(widget, "key", None)}


def test_dialog_has_one_pill_per_group_and_one_field_per_editable_record() -> None:
    # Polish plan G9: pills, not tabs (seven tabs overflowed at 390 px);
    # only the chosen group's fields are drawn, so walk every group.
    app = _app()
    _open_dialog(app)

    pills = app.get_by_key(parameters.GROUP_KEY)
    # The last pill is the scripted-event presets (decision 0004 items 55-58).
    assert list(pills.options) == [*assumptions.ASSUMPTION_GROUPS, parameters.EVENTS_GROUP]
    assert not app.tabs
    fields: set[str] = set()
    for group in assumptions.ASSUMPTION_GROUPS:
        app.get_by_key(parameters.GROUP_KEY).set_value(group).run()
        _open_dialog(app)
        fields |= {key for key in _widget_keys(app) if key.startswith("assumption::")}
    expected = {f"assumption::{record.name}" for record in assumptions.editable_records()}
    assert fields == expected
    # Fixed values (for example the CNZ context and the other cohort columns) are not widgets.
    for fixed in (
        "average_uk.daily_miles_mean",
        "study_days",
    ):
        assert f"assumption::{fixed}" not in fields
    # Decision 0004 item 54: the cohort mix is the one cohort field that is a
    # widget, one per cohort.
    for name in assumptions.COHORT_SHARE_NAMES:
        assert f"assumption::{name}" in fields
    # Decision 0004 item 52 made the warm-up editable (3-14 days, default 7).
    warmup = app.get_by_key("assumption::warmup_days")
    assert warmup.value == 7 and (warmup.proto.min, warmup.proto.max) == (3, 14)


def test_simulation_fields_come_fleet_then_weeks_then_seed() -> None:
    app = _app()
    _open_dialog(app)

    keys = [widget.key for widget in app.number_input]
    assert keys[:3] == [
        "assumption::vehicle_count",
        "assumption::evaluation_world_count",
        "assumption::seed",
    ]


def test_clock_spread_fields_sit_in_an_advanced_expander() -> None:
    app = _app()
    _open_dialog(app)
    _show(app, "assumption::home_charging_power_kw")

    advanced = next(block for block in app.expander if block.label.startswith("Advanced"))
    inside = {widget.key for widget in advanced.number_input}
    assert "assumption::clock_t_df" in inside
    assert "assumption::plug_in_scale_minutes.average_uk" in inside
    assert "assumption::home_charging_power_kw" not in inside


def test_a_field_keeps_its_draft_value_after_its_group_was_hidden() -> None:
    # Streamlit drops a widget's key while its group is not drawn; the field
    # must come back showing the draft, not its minimum.
    app = _app()
    _open_dialog(app)
    _show(app, "assumption::home_charging_power_kw")
    app.get_by_key("assumption::home_charging_power_kw").set_value(3.6).run()
    _open_dialog(app)
    _show(app, "assumption::home_charging_power_kw")
    app.get_by_key(parameters.GROUP_KEY).set_value("Weather").run()
    _open_dialog(app)
    assert "assumption::home_charging_power_kw" not in _widget_keys(app)
    _show(app, "assumption::home_charging_power_kw")

    assert app.get_by_key("assumption::home_charging_power_kw").value == 3.6


def test_fields_take_label_unit_bounds_and_help_from_the_records() -> None:
    app = _app()
    _open_dialog(app)
    _show(app, "assumption::home_charging_power_kw")

    power = app.get_by_key("assumption::home_charging_power_kw")
    record = assumptions.HOME_CHARGING["home_charging_power_kw"]
    assert power.label == "Home charging power (kW)"
    assert (power.proto.min, power.proto.max) == record.bounds
    assert power.value == 7.0
    assert record.meaning in power.help and record.affects in power.help
    # Mike found "AC" confusing: no field says it (decision 0004 item 34).
    assert not any(" AC" in widget.label for widget in app.number_input)
    _show(app, "assumption::public_top_up_threshold_soc_percent")
    threshold = app.get_by_key("assumption::public_top_up_threshold_soc_percent")
    target = app.get_by_key("assumption::public_top_up_target_soc_percent")
    # Default threshold re-baselined from 20 % to 10 % by decision 0004 item 37.
    assert (threshold.value, target.value) == (10.0, 80.0)
    _show(app, "assumption::seed")
    seed = app.get_by_key("assumption::seed")
    assert isinstance(seed.value, int) and seed.proto.step == 1


def test_intraday_dispatch_fields_are_editable_with_their_records_bounds() -> None:
    # Intraday dispatch contract v1 §10 (K5): three fields newly editable on
    # the Public & prices tab, generic per-record rendering.
    app = _app()
    _open_dialog(app)
    _show(app, "assumption::trading.intraday_dispatch")

    switch_names = {widget.key for widget in app.toggle}
    assert "assumption::trading.intraday_dispatch" in switch_names
    switch = app.get_by_key("assumption::trading.intraday_dispatch")
    assert switch.value is True  # on by default (lead decision Q1)
    assert "assumption::trading.intraday_dispatch" not in {w.key for w in app.number_input}

    threshold = app.get_by_key("assumption::trading.replan_threshold_gbp_per_mwh")
    record = assumptions.TRADING_OVERLAY["trading.replan_threshold_gbp_per_mwh"]
    assert threshold.value == 10.0
    assert (threshold.proto.min, threshold.proto.max) == record.bounds == (0.0, 500.0)

    share = app.get_by_key("assumption::trading.day_ahead_commitment_share")
    share_record = assumptions.TRADING_OVERLAY["trading.day_ahead_commitment_share"]
    assert share.value == 0.8
    assert (share.proto.min, share.proto.max) == share_record.bounds == (0.0, 1.0)
    assert "locked to their day-ahead plan" in share.help


def test_cohort_share_fields_default_to_the_source_shares_with_a_source_default_note() -> None:
    # Decision 0004 item 54: six editable percent fields, one per cohort,
    # defaulting to the source share, bounds 0-100.
    app = _app()
    _open_dialog(app)
    _show(app, "assumption::average_uk.population_share_percent")

    expected_defaults = {
        "average_uk.population_share_percent": 40.0,
        "intelligent_octopus.population_share_percent": 30.0,
        "infrequent_charging.population_share_percent": 10.0,
        "infrequent_driving.population_share_percent": 10.0,
        "scheduled_charging.population_share_percent": 9.0,
        "always_plugged_in.population_share_percent": 1.0,
    }
    captions = "\n".join(item.value for item in app.caption)
    for name, default in expected_defaults.items():
        widget = app.get_by_key(f"assumption::{name}")
        assert widget.value == default
        assert (widget.proto.min, widget.proto.max) == (0.0, 100.0)
        assert "items 21 and 54" in widget.help
    for default in (40, 30, 10, 9, 1):
        assert f"Source default: {default}%" in captions
    assert "Total 100%" in captions


def test_cohort_shares_off_100_disables_run_and_a_balancing_edit_restores_it() -> None:
    app = _app()
    _open_dialog(app)
    _show(app, "assumption::average_uk.population_share_percent")
    app.get_by_key("assumption::average_uk.population_share_percent").set_value(45.0).run()

    assert app.session_state["draft_errors"] == {
        "average_uk.population_share_percent": (
            "the six cohort shares must sum to 100% (currently 105%)"
        )
    }
    # A widget edit closes the dialog fragment in AppTest's simulation (as the
    # other dialog tests in this file also find), so it is reopened before
    # reading anything the dialog itself draws.
    _open_dialog(app)
    assert app.get_by_key("run-from-dialog").disabled
    captions = "\n".join(item.value for item in app.caption)
    assert "Total 105%" in captions
    assert "must sum to 100%" in captions
    assert [element.value for element in app.error] == [
        "Draft invalid: Population share: Average (UK) the six cohort shares must sum "
        "to 100% (currently 105%)."
    ]

    # Take the same 5 points back from another cohort: the six sum to 100% again.
    app.get_by_key("assumption::intelligent_octopus.population_share_percent").set_value(25.0).run()

    assert app.session_state["draft_errors"] == {}
    _open_dialog(app)
    assert not app.get_by_key("run-from-dialog").disabled
    assert not app.error


def test_simulation_field_labels_hide_implementation_detail() -> None:
    # Goal review action 10: "Evaluation worlds (worlds)" and "Random seed
    # (np.random.default_rng seed)" read as internals; a first-time user
    # sees plain field names instead. The override is cosmetic only.
    app = _app()
    _open_dialog(app)

    weeks = app.get_by_key("assumption::evaluation_world_count")
    seed = app.get_by_key("assumption::seed")

    assert weeks.label == "Simulated weeks"
    assert seed.label == "Random seed"
    record = assumptions.SIMULATION["evaluation_world_count"]
    assert weeks.value == record.value
    assert (weeks.proto.min, weeks.proto.max) == record.bounds
    # The tooltip explains the seed in plain words: no function name anywhere
    # a reader sees (voice pass, Group D).
    assert "np.random" not in seed.help
    assert "reproduce the same run" in seed.help


def test_dialog_opens_on_the_simulation_group_by_default() -> None:
    app = _app()
    _open_dialog(app)

    assert app.get_by_key(parameters.GROUP_KEY).value == "Simulation"
    assert "assumption::vehicle_count" in _widget_keys(app)


def test_dialog_edits_survive_closing_and_navigation_without_running() -> None:
    app = _app()
    _open_dialog(app)
    _show(app, "assumption::home_charge_efficiency")
    app.get_by_key("assumption::home_charge_efficiency").set_value(0.87).run()
    _open_dialog(app)
    _show(app, "assumption::sensitivity_fraction_per_c")
    app.get_by_key("assumption::sensitivity_fraction_per_c").set_value(0.023).run()
    _navigate(app, "Drivers")
    _open_dialog(app)

    _show(app, "assumption::home_charge_efficiency")
    assert app.get_by_key("assumption::home_charge_efficiency").value == 0.87
    _show(app, "assumption::sensitivity_fraction_per_c")
    assert app.get_by_key("assumption::sensitivity_fraction_per_c").value == 0.023
    assert app.session_state["draft_values"]["home_charge_efficiency"] == 0.87
    assert app.session_state["run_log"].history == []


def test_fleet_size_field_and_draft_stay_one_value_after_the_dialog_closes() -> None:
    # Regression (design 3.3): the slider once showed 200 while its caption
    # and the run still used 1,000.
    app = _app()
    _open_dialog(app)
    app.get_by_key("assumption::vehicle_count").set_value(200).run()
    for page in ("Drivers", "Compare", "Overview"):
        _navigate(app, page)
    _open_dialog(app)

    assert app.get_by_key("assumption::vehicle_count").value == 200
    assert app.session_state["draft_values"]["vehicle_count"] == 200


def test_changed_fields_show_draft_and_run_values_and_resets_restore_them() -> None:
    app = _app()
    _open_dialog(app)
    app.get_by_key("assumption::vehicle_count").set_value(20).run()
    _open_dialog(app)
    app.get_by_key("assumption::evaluation_world_count").set_value(2).run()
    _open_dialog(app)
    app.get_by_key("run-from-dialog").click().run()
    assert not app.exception
    assert len(app.session_state["run_log"].history) == 1

    _open_dialog(app)
    _show(app, "assumption::home_charging_power_kw")
    app.get_by_key("assumption::home_charging_power_kw").set_value(3.6).run()
    _open_dialog(app)
    _show(app, "assumption::home_charging_power_kw")
    captions = "\n".join(item.value for item in app.caption)
    assert "Draft 3.6 · Run 1 used 7" in captions
    assert "1 change from Run 1" in captions

    app.get_by_key("reset-to-active").click().run()
    assert app.session_state["draft_values"]["home_charging_power_kw"] == 7.0
    assert app.session_state["draft_values"]["vehicle_count"] == 20
    _open_dialog(app)
    app.get_by_key("reset-to-defaults").click().run()
    assert app.session_state["draft_values"] == assumptions.editable_defaults()
    # Only the one Run click ran the model.
    assert len(app.session_state["run_log"].history) == 1


def test_editing_one_group_keeps_an_edit_made_in_another_group() -> None:
    # Review of ecacff9, blocking 1: in the browser a pill click reruns only
    # the dialog, so the hidden Simulation fields' widget keys can keep old
    # values; editing a field in another group once copied those back over
    # the draft (seed 7 reverted to 42). The stale key is set by hand here
    # because AppTest's full rerun would otherwise resync every key from the
    # draft and hide the bug.
    app = _app()
    _open_dialog(app)
    app.get_by_key("assumption::seed").set_value(7).run()
    _open_dialog(app)
    _show(app, "assumption::home_charging_power_kw")
    app.session_state["assumption::seed"] = 42  # stale key of the hidden group
    app.get_by_key("assumption::home_charging_power_kw").set_value(3.6).run()

    draft = app.session_state["draft_values"]
    assert draft["seed"] == 7
    assert draft["home_charging_power_kw"] == 3.6
    assert draft["vehicle_count"] == assumptions.editable_defaults()["vehicle_count"]


def test_each_field_shows_its_evidence_class_as_a_badge() -> None:
    app = _app()
    _open_dialog(app)

    badges = [element.value for element in app.markdown if "-badge[" in element.value]
    fields = [widget.key for widget in app.number_input]
    evidence_words = ("Source", "Illustrative", "Synthetic")
    field_badges = [b for b in badges if any(f"[{word}]" in b for word in evidence_words)]
    assert len(field_badges) == len(fields)


# --- Firm MW tab (trading contract v1 §10.8): manufacturer shares, the
# blackout table and the newsvendor/event conflict ------------------------


def test_manufacturer_shares_off_one_disables_run_and_a_balancing_edit_restores_it() -> None:
    # Mirrors test_cohort_shares_off_100_disables_run_and_a_balancing_edit_restores_it:
    # the four manufacturer shares (0.25 each by default) must sum to 1.
    app = _app()
    _open_dialog(app)
    _show(app, "assumption::manufacturers.share.m1")
    app.get_by_key("assumption::manufacturers.share.m1").set_value(0.5).run()

    assert app.session_state["draft_errors"] == {
        "manufacturers.share.m1": "the four manufacturer shares must sum to 1 (currently 1.25)"
    }
    _open_dialog(app)
    assert app.get_by_key("run-from-dialog").disabled
    captions = "\n".join(item.value for item in app.caption)
    assert "Total 1.25" in captions
    assert "must sum to 1" in captions

    app.get_by_key("assumption::manufacturers.share.m2").set_value(0.0).run()

    assert app.session_state["draft_errors"] == {}
    _open_dialog(app)
    assert not app.get_by_key("run-from-dialog").disabled


def test_invalid_blackout_table_shows_the_validation_banner_and_blocks_run() -> None:
    # st.data_editor has no AppTest interaction API (Streamlit 1.64), so the
    # edit is applied the way its own on_change path would: a direct call to
    # run_controller.set_blackout_windows, then a rerun to see the dialog.
    app = _app()
    _open_dialog(app)
    _show(app, "assumption::trading.commitment_rule")

    bad = pd.DataFrame({"start_local_time": ["23:05"], "duration_minutes": [60]})
    rc.set_blackout_windows(app.session_state, bad)
    _open_dialog(app)

    captions = "\n".join(item.value for item in app.caption)
    assert "start_local_time" in captions
    assert app.get_by_key("run-from-dialog").disabled
    assert any("Blackout windows" in error.value for error in app.error)

    rc.set_blackout_windows(app.session_state, rc.EMPTY_BLACKOUT_WINDOWS)
    _open_dialog(app)

    assert not app.get_by_key("run-from-dialog").disabled


def test_newsvendor_conflict_caption_names_the_reason_and_blocks_run() -> None:
    app = _app()
    _open_dialog(app)
    _show(app, "assumption::evaluation_world_count")
    app.get_by_key("assumption::evaluation_world_count").set_value(1).run()
    _open_dialog(app)
    _show(app, "assumption::trading.commitment_rule")
    app.get_by_key("assumption::trading.commitment_rule").set_value("newsvendor").run()
    _open_dialog(app)

    captions = "\n".join(item.value for item in app.caption)
    assert "Newsvendor needs more than one simulated week" in captions
    assert app.get_by_key("run-from-dialog").disabled
    assert any("Commitment rule" in error.value for error in app.error)

    app.get_by_key("assumption::trading.commitment_rule").set_value("fixed_share").run()

    _open_dialog(app)
    assert not app.get_by_key("run-from-dialog").disabled


def test_a_plain_on_off_switch_has_no_redundant_caption() -> None:
    # "switch (1 on, 0 off)" printed "1 on, 0 off." under the toggle, which
    # says nothing the toggle does not already show.
    app = _app()
    _open_dialog(app)
    _show(app, "assumption::trading.intraday_dispatch")

    captions = [caption.value for caption in app.caption]
    assert not any(text.startswith("1 on, 0 off") for text in captions)


def test_fixed_values_table_names_the_archetype_of_each_per_archetype_row() -> None:
    # Six archetypes share one label ("Battery capacity"); without the
    # archetype the fixed-values rows cannot be told apart.
    records = tuple(
        record
        for record in assumptions.records_in_group("Battery")
        if not record.editable and record.label == "Battery capacity"
    )
    assert len(records) == len(assumptions.COHORT_SOURCE_NAMES)

    table = parameters._fixed_records_table(records)

    assert list(table.columns) == ["Archetype", "Label", "Value", "Unit", "Evidence"]
    assert table["Archetype"].tolist() == list(assumptions.COHORT_SOURCE_NAMES.values())
    assert table["Archetype"].is_unique


def test_fixed_values_table_of_shared_values_has_no_archetype_column() -> None:
    shared = tuple(
        record for record in assumptions.records_in_group("Simulation") if not record.editable
    )
    assert shared

    table = parameters._fixed_records_table(shared)

    assert list(table.columns) == ["Label", "Value", "Unit", "Evidence"]


def test_timed_tariff_switch_is_on_and_the_start_hour_shows_0_and_accepts_23_5() -> None:
    # Decision 0007 follow-up (8 October 2026): a bounded number_input has no
    # reachable "unset" value in Streamlit (an empty commit resolves to
    # min_value, not NaN), so "on by default" is the separate
    # ``timed_tariff_enabled`` switch, and the hour is a plain field that
    # never needs to be cleared.
    app = _app()
    _open_dialog(app)
    _show(app, "assumption::timed_tariff_enabled")

    switch = app.get_by_key("assumption::timed_tariff_enabled")
    assert switch.value is True
    assert "assumption::timed_tariff_enabled" not in {w.key for w in app.number_input}

    field = app.get_by_key("assumption::timed_start_local_hour")
    assert field.value == 0.0

    field.set_value(23.5).run()
    assert not app.exception
    assert app.session_state["draft_values"]["timed_start_local_hour"] == 23.5
    assert "timed_start_local_hour" not in app.session_state["draft_errors"]


def test_toggling_the_timed_tariff_switch_controls_the_number_of_paths_in_a_run() -> None:
    app = _app()
    _open_dialog(app)
    app.get_by_key("assumption::vehicle_count").set_value(12).run()
    _open_dialog(app)
    app.get_by_key("assumption::evaluation_world_count").set_value(2).run()

    _open_dialog(app)
    _show(app, "assumption::timed_tariff_enabled")
    app.get_by_key("assumption::timed_tariff_enabled").set_value(False).run()
    _open_dialog(app)
    app.get_by_key("run-from-dialog").click().run()
    assert not app.exception

    off_result = rc.active_result(app.session_state)
    assert off_result is not None
    assert set(off_result.fleet_world_intervals["path_id"]) == {"normal", "selected"}

    _open_dialog(app)
    _show(app, "assumption::timed_tariff_enabled")
    app.get_by_key("assumption::timed_tariff_enabled").set_value(True).run()
    _open_dialog(app)
    app.get_by_key("run-from-dialog").click().run()
    assert not app.exception

    on_result = rc.active_result(app.session_state)
    assert on_result is not None
    assert set(on_result.fleet_world_intervals["path_id"]) == {"normal", "selected", "timed"}
