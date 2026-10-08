"""Overview ▸ Key stats: one dense table built from existing frames (plan E1, item 54)."""

from __future__ import annotations

import dataclasses
from datetime import date
from functools import cache

import pandas as pd
import pytest
from fixtures.result_fixture import make_result

from axle_studio.model.forecast import ForecastResult, run_forecast_from_assumptions
from axle_studio.ui.style import PATH_LABELS, money
from axle_studio.ui.views import key_stats
from axle_studio.ui.views.key_stats import COLUMNS, key_stats_table, render_key_stats


class RecordingStreamlit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def __getattr__(self, name: str):
        def record(*args: object, **kwargs: object) -> None:
            self.calls.append((name, args, kwargs))

        return record


def row(table, metric: str):
    return table.loc[table["Metric"].eq(metric)].iloc[0]


@pytest.fixture
def action_result():
    return make_result("action", evs=6, worlds=8)


@pytest.fixture
def no_action_result():
    return make_result("no_action", evs=6, worlds=8)


def test_renders_one_dataframe_with_the_plan_columns(action_result) -> None:
    st = RecordingStreamlit()

    render_key_stats(st, action_result)

    frames = [args[0] for name, args, _ in st.calls if name == "dataframe"]
    assert len(frames) == 1
    assert list(frames[0].columns) == list(COLUMNS)
    kwargs = next(kwargs for name, _, kwargs in st.calls if name == "dataframe")
    assert set(kwargs["column_config"]) == set(COLUMNS)
    captions = [args[0] for name, args, _ in st.calls if name == "caption"]
    assert all(len(caption) <= 140 for caption in captions)


def test_metric_and_unit_columns_fit_their_longest_text(action_result) -> None:
    # Regression: "Unit" at 75 px clipped "£ (illustrative) per customer per
    # month"; "Metric" at 200 px cut "Below-target power per plugged-in EV
    # at 18:00" before the time, so three rows read identically. About 5.3 px
    # per character at the table font (measured), so 6 px leaves room for padding.
    table = key_stats_table(action_result)
    for column in ("Metric", "Unit"):
        width = key_stats._COLUMN_CONFIG[column]["width"]
        assert isinstance(width, int)
        assert table[column].str.len().max() * 6 <= width


def test_sections_in_order_and_smart_rows_only_on_action(action_result, no_action_result) -> None:
    action_sections = list(dict.fromkeys(key_stats_table(action_result)["Section"]))
    assert action_sections == [
        key_stats.SECTION_A,
        key_stats.SECTION_B,
        key_stats.SECTION_C,
        key_stats.SECTION_D,
        key_stats.SECTION_E,
        key_stats.SECTION_F,
        key_stats.SECTION_G,
        key_stats.SECTION_H,
        key_stats.SECTION_I,
    ]
    no_action_sections = list(dict.fromkeys(key_stats_table(no_action_result)["Section"]))
    assert no_action_sections == [key_stats.SECTION_A, key_stats.SECTION_B, key_stats.SECTION_F]
    trading = key_stats_table(no_action_result).iloc[-1]  # section I is action-only
    assert trading["Verdict"] == key_stats.TRADING_PLACEHOLDER


def test_values_are_the_model_columns(action_result) -> None:
    table = key_stats_table(action_result)
    flex = action_result.flexibility_weekly_summary.iloc[0]
    peak = row(table, "Peak charger power below target")
    assert (peak["P10"], peak["P50"], peak["P90"]) == tuple(
        f"{flex[f'peak_deferrable_kw_{q}']:,.0f}" for q in ("p10", "p50", "p90")
    )
    assert peak["Verdict"] == f"most often {flex['peak_modal_interval_start_london']:%a %H:%M}"
    # Final critique B-1: the model's "deferrable" total is the charger power of
    # plugged-in EVs below target (it includes must-charge-now EVs and runs
    # above actual import), so no section B row calls it deferrable.
    flexibility = table.loc[table["Section"].eq(key_stats.SECTION_B), "Metric"]
    assert not flexibility.str.contains("Deferrable|deferrable").any()

    kpis = action_result.plug_in_summary.kpis.set_index(["day_type", "metric"])
    soc = row(table, "Median SoC at plug-in")
    assert soc["P50"] == f"{kpis.loc[('all', 'median_plug_in_soc_percent'), 'p50']:.1f}"
    assert soc["Reference"] == "CNZ 52%"

    peaks = action_result.weekly_peak_summary.set_index("path_id")
    ratio = row(table, "Smart ÷ unmanaged peak")
    assert ratio["P50"] == f"{peaks.loc['selected', 'ratio_to_normal_p50']:.2f}"
    assert row(table, "Weekly peak, smart")["Δ vs unmanaged"] == (
        f"×{peaks.loc['selected', 'ratio_to_normal_p50']:.2f}"
    )
    cheapest = action_result.cheapest_half_hour_summary.iloc[0]
    assert row(table, "Cheapest day-ahead half-hour")["P50"] == cheapest["local_time_label_p50"]


def test_saving_flips_the_cost_tails_rather_than_requantiling(action_result) -> None:
    # The cost total is smart minus unmanaged; a saving's P10 is the cost's
    # P90 negated, never a new quantile.
    summary = action_result.cost_effect_summary.copy()
    total = summary["component"].eq("total")
    summary.loc[total, ["p10", "p50", "p90"]] = [-300.0, -200.0, -50.0]
    result = dataclasses.replace(action_result, cost_effect_summary=summary)

    saving = row(key_stats_table(result), "Weekly saving, illustrative")

    assert (saving["P10"], saving["P50"], saving["P90"]) == ("£50", "£200", "£300")
    assert saving["Verdict"] == "saves"
    per_ev = row(key_stats_table(result), "Saving per EV, illustrative")
    assert per_ev["P50"] == f"£{200 / result.vehicle_count:.2f}"


def test_verdict_rules() -> None:
    assert key_stats._spread_verdict(9.6, 10.0, 10.4) == "tight"
    assert key_stats._spread_verdict(2.0, 10.0, 9.0 + 10.0) == "wide"
    assert key_stats._spread_verdict(8.0, 10.0, 12.0) == ""
    assert key_stats._spread_verdict(0.0, 0.0, 0.0) == ""  # no ratio to a zero median
    assert key_stats._versus_cnz(63.4, 52.0) == "above CNZ by 11 pts"
    assert key_stats._versus_cnz(51.0, 52.0) == "near CNZ"
    assert key_stats._versus_cnz(40.0, None) == ""


def test_herding_and_risk_verdicts(action_result) -> None:
    peaks = action_result.weekly_peak_summary.copy()
    peaks.loc[peaks["path_id"].eq("selected"), "ratio_to_normal_p50"] = 1.8
    summary = action_result.smart_charging_summary.copy()
    summary.loc[summary["metric"].eq("early_departure_count"), "p50"] = 0.5
    result = dataclasses.replace(
        action_result, weekly_peak_summary=peaks, smart_charging_summary=summary
    )
    table = key_stats_table(result)

    assert row(table, "Smart ÷ unmanaged peak")["Verdict"] == "herding high"
    early = row(table, "Early departures")
    sessions = action_result.plug_in_summary.kpis.set_index(["day_type", "metric"]).loc[
        ("all", "plug_ins_per_week"), "p50"
    ]
    assert early["Verdict"] == f"risk low, about {100 * 0.5 / sessions:.1f}%"
    assert early["Reference"] == f"of about {sessions:,.0f} sessions"


def test_missing_values_read_unavailable_never_zero(action_result) -> None:
    summary = action_result.flexibility_weekly_summary.copy()
    summary[["w_per_plugged_in_ev_0300_p10", "w_per_plugged_in_ev_0300_p50"]] = float("nan")
    result = dataclasses.replace(action_result, flexibility_weekly_summary=summary)

    per_ev = row(key_stats_table(result), "Below-target power per plugged-in EV at 03:00")

    assert per_ev["P50"] == "Unavailable"


def test_no_snake_case_on_screen(action_result) -> None:
    table = key_stats_table(action_result)
    for column in ("Metric", "Unit", "Reference", "Verdict"):
        assert not table[column].str.contains(r"[a-z]_[a-z]").any(), column


def test_cnz_points_reads_the_record_unit(action_result) -> None:
    assert key_stats._cnz_points(action_result, "cnz_median_plug_in_soc_percent") == 52.0
    assert key_stats._cnz_points(
        action_result, "cnz_share_plug_ins_below_10_percent_soc"
    ) == pytest.approx(3.0)
    assert key_stats._cnz_points(action_result, "no_such_record") is None


def test_p50_leads_the_statistics() -> None:
    assert list(COLUMNS[2:5]) == ["P50", "P10", "P90"]


# --- Section I, Households (household contract v1 §6.3) -------------------------


def _household_rows(table):
    return table.loc[table["Section"].eq(key_stats.SECTION_I)]


def _fleet(result, metric: str, statistic: str):
    summary = result.household_outcomes_summary
    return summary.loc[
        summary["group_id"].eq("fleet")
        & summary["metric"].eq(metric)
        & summary["statistic"].eq(statistic)
    ].iloc[0]


def test_section_i_rows_in_order_on_action_only(action_result, no_action_result) -> None:
    rows = _household_rows(key_stats_table(action_result))

    assert list(rows["Metric"]) == [
        "Value per customer, mean",
        "Sessions ready at departure, smart",
        "Smart charging cost",
        "Charging in the cheapest third",
        "Session ends affected",
        "Nights plugged in",
        "Evening turn-down per customer",
        "CO₂ shifted per customer",
    ]
    assert _household_rows(key_stats_table(no_action_result)).empty


def test_section_i_reads_group_ratios_and_set_b_years(action_result) -> None:
    table = key_stats_table(action_result)
    ready = _fleet(action_result, "completed_share_selected", "group_ratio")
    ready_row = row(table, "Sessions ready at departure, smart")
    assert ready_row["P50"] == key_stats._num(ready["p50"] * 100.0, 0)
    assert (
        f"over {int(ready['ev_value_count'])} customers with closed sessions"
        in (ready_row["Reference"])
    )

    value_row = row(table, "Value per customer, mean")
    value = _fleet(action_result, "value_gbp_per_week", "mean_across_evs")
    year = _fleet(action_result, "value_gbp_per_year", "p50_of_ev_means")
    assert value_row["P50"] == key_stats.money(value["p50"], decimals=2)
    # Q-10: the two statistics are named apart, and the year is not "≈ week × 52".
    assert "mean across customers each week" in value_row["Reference"]
    assert (
        f"Median customer's average year {key_stats.money(year['mean'])}"
        in (value_row["Reference"])
    )

    cost_row = row(table, "Smart charging cost")
    change = _fleet(action_result, "home_cost_gbp_per_kwh_difference", "group_ratio")
    assert cost_row["Verdict"] == ("cheaper" if change["p90"] < 0 else "")


def test_section_i_charging_cost_is_in_pence_per_kwh(action_result) -> None:
    # Final critique B-10: one customer unit, p/kWh at 1 dp (an exact × 100 of
    # the stored £/kWh), with the £/MWh equivalence stated in the reference.
    table = key_stats_table(action_result)
    cost = _fleet(action_result, "home_cost_gbp_per_kwh_selected", "group_ratio")
    change = _fleet(action_result, "home_cost_gbp_per_kwh_difference", "group_ratio")
    cost_row = row(table, "Smart charging cost")
    assert cost_row["Unit"] == "p/kWh"
    assert cost_row["P50"] == key_stats._num(cost["p50"] * 100.0, 1)
    assert cost_row["Δ vs unmanaged"].endswith(" p/kWh")
    assert key_stats._num(abs(change["p50"]) * 100.0, 1) in cost_row["Δ vs unmanaged"]
    assert "1 p/kWh = £10/MWh" in cost_row["Reference"]
    assert not table["Unit"].eq("£ per kWh").any()


def test_section_i_missing_values_read_unavailable(action_result) -> None:
    table = key_stats_table(action_result)
    turn_down = row(table, "Evening turn-down per customer")
    # The fixture has no firm-MW inputs, so evening kW is NaN (contract §3.1).
    assert turn_down["P50"] == "Unavailable"
    assert turn_down["Verdict"] == "Unavailable: this run has no evening turn-down figures"

    absent = dataclasses.replace(action_result, household_outcomes_summary=None)
    rows = _household_rows(key_stats_table(absent))
    assert list(rows["Verdict"]) == [key_stats.HOUSEHOLDS_UNAVAILABLE]


# --- Section G, Supplier (supplier contract v1 §6) -------------------------------


def test_supplier_rows_read_the_profiled_stored_quantiles(action_result) -> None:
    table = key_stats_table(action_result)
    summary = action_result.supplier_pnl_summary
    stored = summary.loc[
        summary["hedge_variant"].eq("profiled")
        & summary["metric"].eq("net_gain_before_fee_per_customer_per_month_gbp")
    ].iloc[0]
    net = row(table, "Supplier gain per customer, before fee")
    assert net["P50"] == key_stats.money(stored["p50"], decimals=2, signed=True)
    assert net["P10"] == key_stats.money(stored["p10"], decimals=2, signed=True)
    assert net["Unit"].startswith("£ (illustrative)")
    # The trading net is quoted beside, never added.
    assert "not added" in net["Reference"]


def test_unset_platform_fee_is_unavailable_never_zero(action_result) -> None:
    fee = row(key_stats_table(action_result), "Platform fee")
    assert (fee["P10"], fee["P50"], fee["P90"]) == ("Unavailable",) * 3
    assert fee["Verdict"] == "Unavailable: not set"


def test_section_g_no_longer_carries_a_firm_mw_row(action_result) -> None:
    # Final critique B-5: the evening firm figure lives once, in section H.
    table = key_stats_table(action_result)
    supplier = table.loc[table["Section"].eq(key_stats.SECTION_G), "Metric"]
    assert not supplier.str.contains("turn-down, evening", case=False).any()


def test_section_h_says_so_when_the_availability_frames_are_absent(action_result) -> None:
    firm_mw = key_stats_table(action_result).loc[
        key_stats_table(action_result)["Section"].eq(key_stats.SECTION_H)
    ]
    assert len(firm_mw) == 1
    assert firm_mw.iloc[0]["Verdict"] == key_stats.FIRM_MW_UNAVAILABLE


def _with_firm_mw_frames(action_result, *, coverage_strict_p10: float | None = 0.004):
    """The action fixture plus hand-built firm-MW frames (the fixture has none)."""

    sheet = pd.DataFrame(
        [
            {
                "window": "evening",
                "direction": "turn_down",
                "duration_hours": 1.0,
                "window_mean_mw_p05": 0.1,
                "window_mean_mw_p10": 0.2,
                "window_mean_mw_p50": 0.3,
                "window_mean_mw_p90": 0.4,
                "firm_share": 0.66,
                "intraday_firm_mw_p50": 0.25,
            }
        ]
    )
    backtest_summary = pd.DataFrame(
        [
            {
                "horizon": "day_ahead",
                "direction": "turn_down",
                "duration_hours": 1.0,
                "coverage_p10": 0.12,
                **(
                    {}
                    if coverage_strict_p10 is None
                    else {"coverage_strict_p10": coverage_strict_p10}
                ),
            }
        ]
    )
    # Model questions Q-2: n_eff is read as the median over half-hours with
    # turn-down (P50 > 0); the 19:00 row with a zero median is left out.
    bands = pd.DataFrame(
        [
            {
                "horizon": "day_ahead",
                "statistic": "realised",
                "direction": "turn_down",
                "duration_hours": 1.0,
                "night_index": 0,
                "interval_start_london": pd.Timestamp(when, tz="Europe/London"),
                "p50": p50,
                "n_eff": n_eff,
            }
            for when, p50, n_eff in (
                ("2026-01-12 19:00", 0.0, 1.0),
                ("2026-01-12 22:00", 5.0, 3.0),
                ("2026-01-12 22:30", 6.0, 4.0),
                ("2026-01-12 23:00", 7.0, float("nan")),
            )
        ]
    )
    manufacturers = pd.DataFrame(
        [{"manufacturer_id": "fleet", "outage_probability_per_night": 0.08}]
    )
    result = dataclasses.replace(action_result)
    object.__setattr__(result, "product_sheet", sheet)
    object.__setattr__(result, "availability_backtest_summary", backtest_summary)
    object.__setattr__(result, "availability_bands", bands)
    object.__setattr__(result, "manufacturer_summary", manufacturers)
    return result


def test_section_h_reads_the_stored_firm_mw_frames(action_result) -> None:
    table = key_stats_table(_with_firm_mw_frames(action_result))
    turn_down = row(table, "Evening turn-down, 1 h")
    assert turn_down["Section"] == key_stats.SECTION_H
    assert (turn_down["P10"], turn_down["P50"], turn_down["P90"]) == ("0.200", "0.300", "0.400")
    assert "firm share (P10÷P50) 66%" in turn_down["Reference"]
    assert "typical 17:00 forecast 0.25 MW" in turn_down["Reference"]
    assert row(table, "Effective EV count, turn-down half-hours")["P50"] == "3.5"
    assert row(table, "Chance any maker is out, one night")["P50"] == "8%"


def test_evening_firm_row_appears_once_with_consistent_exceedance_names(action_result) -> None:
    # Final critique B-5 (the row was rendered twice, once as a section G row)
    # and B-2 (P10 is 90% exceedance, P05 is 95% exceedance, never "P95").
    table = key_stats_table(_with_firm_mw_frames(action_result))
    evening = table.loc[table["Metric"].str.contains("turn-down, evening|Evening turn-down, 1 h")]
    assert list(evening["Section"]) == [key_stats.SECTION_H]
    reference = evening.iloc[0]["Reference"]
    assert "firm = P10 column (90% exceedance)" in reference
    assert "P05 (95% exceedance) 0.100 MW" in reference
    assert "P95" not in reference
    assert "beyond the smart schedule" in reference


def test_calibration_row_shows_both_hit_rates_beside_the_target(action_result) -> None:
    # Final critique B-6 and the 30 Sep ruling: strictly below to at or below,
    # with the 10% target; a calibrated forecast lies between the two.
    table = key_stats_table(_with_firm_mw_frames(action_result, coverage_strict_p10=0.004))
    calibration = row(table, "Day-ahead calibration, P10 level")
    assert calibration["P50"] == "0.4–12.0%"
    assert "target 10%" in calibration["Reference"]
    assert calibration["Verdict"] == "target within the range"

    over = key_stats_table(_with_firm_mw_frames(action_result, coverage_strict_p10=0.11))
    assert row(over, "Day-ahead calibration, P10 level")["Verdict"].startswith("over-covered")

    only_weak = key_stats_table(_with_firm_mw_frames(action_result, coverage_strict_p10=None))
    assert row(only_weak, "Day-ahead calibration, P10 level")["P50"] == "12.0% at or below"


def test_section_f_reads_the_full_strategy_trading_kpis(action_result) -> None:
    # Trading contract v1 section 9.1 (decision 0005): section F reads the
    # ``full`` strategy row of each named metric, never recomputing money.
    table = key_stats_table(action_result)
    full = action_result.trading_kpis.set_index(["strategy", "metric"]).loc["full"]

    net = full.loc["net_gbp_per_week"]
    net_row = row(table, "Illustrative trading net P&L")
    assert net_row["Unit"] == "£ per week"
    assert (net_row["P10"], net_row["P50"], net_row["P90"]) == (
        money(net["p10"], signed=True),
        money(net["p50"], signed=True),
        money(net["p90"], signed=True),
    )
    assert net_row["Verdict"] == "not Axle cash"
    assert money(net["cvar5"], signed=True) in net_row["Reference"]

    settled = full.loc["settled_mwh_per_week"]
    settled_row = row(table, "Settled turn-down volume")
    assert settled_row["P50"] == f"{settled['p50']:.2f}"

    imbalance = full.loc["imbalance_volume_share"]
    imbalance_row = row(table, "Imbalance share of settled volume")
    assert imbalance_row["P50"] == f"{100 * imbalance['p50']:.1f}"


def test_section_f_placeholder_names_a_no_action_result(no_action_result) -> None:
    table = key_stats_table(no_action_result)
    trading = row(table, "Trading P&L")
    assert trading["Verdict"] == key_stats.TRADING_PLACEHOLDER
    assert "smart charging" in key_stats.TRADING_PLACEHOLDER.lower()


def test_section_f_carries_the_9_8_trader_metric_rows(action_result) -> None:
    # Trading contract v1 sections 9.1 and 9.8 (decision 0004 item 58): the
    # trader-metric rows added to trading_kpis, full strategy, human labels.
    table = key_stats_table(action_result)
    full = action_result.trading_kpis.set_index(["strategy", "metric"]).loc["full"]

    capture = full.loc["capture_rate"]
    capture_row = row(table, "Capture rate")
    assert capture_row["P50"] == f"{100 * capture['p50']:.1f}"
    assert str(int(capture["world_count"])) in capture_row["Reference"]

    value_of_intraday = full.loc["value_of_intraday_gbp_per_week"]
    value_row = row(table, "Value of intraday")
    assert value_row["P50"] == money(value_of_intraday["p50"], signed=True)
    assert value_row["Reference"] == "full strategy only"

    cost = full.loc["cost_of_uncertainty_gbp_per_week"]
    assert row(table, "Cost of uncertainty")["P50"] == money(cost["p50"], signed=True)

    imbalance = full.loc["imbalance_gbp_per_mwh_traded"]
    # Final critique B-4: the ledger sign is kept (positive = income), so the
    # label says "cash", not "cost".
    imbalance_row = row(table, "Imbalance cash per MWh traded")
    assert imbalance_row["P50"] == money(imbalance["p50"], decimals=1, signed=True)
    assert "positive is income" in imbalance_row["Reference"]
    assert not table["Metric"].str.contains("Imbalance cost").any()

    firmness = full.loc["firmness"]
    assert row(table, "Firmness")["P50"] == f"{100 * firmness['p50']:.1f}"

    margin = full.loc["flex_margin_gbp_per_mw_year"]
    margin_row = row(table, "Trading margin per MW per year")
    assert margin_row["P50"] == money(margin["p50"], signed=True)
    assert margin_row["Reference"] == "extrapolated from one simulated week"

    spread = full.loc["day_ahead_spread_gbp_per_mwh"]
    spread_row = row(table, "Day-ahead spread")
    assert spread_row["P50"] == money(spread["p50"], signed=True)
    assert spread_row["Reference"] == "same for every strategy"


def test_section_f_carries_the_intraday_dispatch_rows(action_result) -> None:
    # Intraday dispatch contract v1 sections 6.3 and 10 (K5): 0 on the
    # SYNTHETIC fixture (no dispatch ever ran), still reported, labelled
    # illustrative.
    table = key_stats_table(action_result)
    full = action_result.trading_kpis.set_index(["strategy", "metric"]).loc["full"]

    reoptimisation = full.loc["reoptimisation_gbp_per_week"]
    reopt_row = row(table, "Illustrative re-optimisation P&L")
    assert reopt_row["P50"] == money(reoptimisation["p50"], signed=True)
    assert "intraday dispatch off" in reopt_row["Reference"]

    moved = full.loc["dispatch_moved_mwh_per_week"]
    moved_row = row(table, "Energy moved by intraday dispatch")
    assert moved_row["Unit"] == "MWh per week, illustrative"
    assert moved_row["P50"] == f"{moved['p50']:.2f}"
    assert "intraday dispatch off" in moved_row["Reference"]


# --- Timed tariff path (decision 0007), off by default -----------------------
#
# The SYNTHETIC fixture never carries a "timed" path, its cost-effect sibling
# or the ``_timed`` household/completion columns (make_result's toy physics
# has no timed-start rule), so this is tested against a real small run
# instead, the same route test_action_response.py's own timed-tariff tests
# and this module's Overview sibling use.

_TIMED_START = date(2026, 2, 9)
_TIMED_VALUES = {"vehicle_count": 30, "evaluation_world_count": 3}


@cache
def _timed_result(*, on: bool) -> ForecastResult:
    # The policy is on by default; "absent" switches it off explicitly.
    values = {**_TIMED_VALUES, "timed_tariff_enabled": 1 if on else 0}
    if on:
        values["timed_start_local_hour"] = 0.0
    return run_forecast_from_assumptions(_TIMED_START, values=values)


_TIMED_METRICS = (
    "Price paid, timed tariff",
    "Weekly saving, timed tariff, illustrative",
    "Saving per EV, timed tariff, illustrative",
    "Weekly peak, timed tariff",
    "Timed tariff ÷ unmanaged peak",
    "Coincidence factor, timed tariff",
    "Sessions ready at departure, timed tariff",
    "Timed tariff charging cost",
    "Charging in the cheapest third, timed tariff",
    "Charge completed by departure, timed tariff",
)


def test_timed_tariff_rows_absent_with_the_setting_off() -> None:
    # "With it absent the screen must be exactly as today": none of the new
    # rows appear, and every Smart row this run still carries is unchanged.
    result = _timed_result(on=False)
    assert result.timed_cost_effect_summary is None
    table = key_stats_table(result)

    for metric in _TIMED_METRICS:
        assert table.loc[table["Metric"].eq(metric)].empty, metric
    assert row(table, "Weekly saving, illustrative")["Verdict"] in ("saves", "no saving")
    assert row(table, "Weekly peak, smart")["Metric"] == "Weekly peak, smart"


def test_timed_tariff_rows_present_beside_smart_when_set() -> None:
    result = _timed_result(on=True)
    table = key_stats_table(result)

    for metric in _TIMED_METRICS:
        rows = table.loc[table["Metric"].eq(metric)]
        assert len(rows) == 1, metric
        # Every row this section adds is still a Section C/D/I value, never a
        # blank placeholder.
        assert rows.iloc[0]["P50"] != ""

    # Beside Smart, never instead of it: the Smart-named rows are still there.
    for metric in (
        "Price paid, smart",
        "Weekly saving, illustrative",
        "Saving per EV, illustrative",
        "Weekly peak, smart",
        "Smart ÷ unmanaged peak",
        "Coincidence factor, smart",
        "Sessions ready at departure, smart",
        "Smart charging cost",
        "Charging in the cheapest third",
        "Charge completed by departure, smart",
    ):
        assert not table.loc[table["Metric"].eq(metric)].empty, metric


def test_section_c_timed_price_and_saving_equal_the_model_frames() -> None:
    result = _timed_result(on=True)
    table = key_stats_table(result)

    price = key_stats._metric_row(
        result.smart_charging_summary, "metric", "timed_average_price_gbp_per_mwh"
    )
    price_row = row(table, "Price paid, timed tariff")
    assert (price_row["P10"], price_row["P50"], price_row["P90"]) == (
        key_stats._num(price["p10"], 0),
        key_stats._num(price["p50"], 0),
        key_stats._num(price["p90"], 0),
    )
    assert "no tariff rate applied" in price_row["Reference"]

    total = key_stats._metric_row(result.timed_cost_effect_summary, "component", "total")
    saving = row(table, "Weekly saving, timed tariff, illustrative")
    assert (saving["P10"], saving["P50"], saving["P90"]) == (
        money(-total["p90"]),
        money(-total["p50"]),
        money(-total["p10"]),
    )
    assert saving["Verdict"] == ("saves" if -total["p50"] > 0 else "no saving")
    per_ev = row(table, "Saving per EV, timed tariff, illustrative")
    assert per_ev["P50"] == money(-total["p50"] / result.vehicle_count, decimals=2)


def test_section_d_timed_peak_and_coincidence_equal_the_model_frame() -> None:
    result = _timed_result(on=True)
    table = key_stats_table(result)
    peaks = result.weekly_peak_summary.set_index("path_id")

    peak_row = row(table, "Weekly peak, timed tariff")
    timed = peaks.loc["timed"]
    assert peak_row["P50"] == f"{timed['p50']:,.0f}"
    assert peak_row["Δ vs unmanaged"] == f"×{timed['ratio_to_normal_p50']:.2f}"

    ratio_row = row(table, "Timed tariff ÷ unmanaged peak")
    assert ratio_row["P50"] == f"{timed['ratio_to_normal_p50']:.2f}"

    coincidence_row = row(table, "Coincidence factor, timed tariff")
    assert coincidence_row["P50"] == f"{timed['coincidence_factor_p50']:.2f}"

    # Display order (style.PATH_DISPLAY_ORDER): unmanaged, timed tariff, smart.
    coincidence_rows = table.loc[
        table["Metric"].str.startswith("Coincidence factor,")
        & table["Section"].eq(key_stats.SECTION_D)
    ]
    assert list(coincidence_rows["Metric"]) == [
        f"Coincidence factor, {PATH_LABELS[path].lower()}"
        for path in ("normal", "timed", "selected")
    ]


def test_section_i_timed_rows_equal_the_household_frame() -> None:
    result = _timed_result(on=True)
    table = key_stats_table(result)
    summary = result.household_outcomes_summary

    ready = _fleet(result, "completed_share_timed", "group_ratio")
    ready_row = row(table, "Sessions ready at departure, timed tariff")
    assert ready_row["P50"] == key_stats._num(ready["p50"] * 100.0, 0)
    ready_normal = _fleet(result, "completed_share_normal", "group_ratio")
    assert (
        f"unmanaged P50 {key_stats._num(ready_normal['p50'] * 100.0, 0)}%" in ready_row["Reference"]
    )
    # No change row exists for the timed path (household contract v1 §3.2
    # model step 2): this row never carries a delta.
    assert ready_row["Δ vs unmanaged"] == ""

    cost = _fleet(result, "home_cost_gbp_per_kwh_timed", "group_ratio")
    cost_row = row(table, "Timed tariff charging cost")
    assert cost_row["P50"] == key_stats._num(cost["p50"] * 100.0, 1)
    assert cost_row["Unit"] == "p/kWh"
    assert "costed the same way as smart" in cost_row["Reference"]

    cheap = _fleet(result, "cheap_share_timed", "group_ratio")
    cheap_row = row(table, "Charging in the cheapest third, timed tariff")
    assert cheap_row["P50"] == key_stats._num(cheap["p50"] * 100.0, 0)

    assert not summary["metric"].eq("completed_share_difference_timed").any()


def test_completion_row_timed_equals_the_model_frame() -> None:
    result = _timed_result(on=True)
    table = key_stats_table(result)

    timed = result.charge_completion_summary.loc[
        result.charge_completion_summary["group_id"].eq("fleet")
        & result.charge_completion_summary["path_id"].eq("timed")
        & result.charge_completion_summary["departure"].eq("all")
    ].iloc[0]
    completion_row = row(table, "Charge completed by departure, timed tariff")
    assert completion_row["P50"] == key_stats._num(100 * timed["completed_share_p50"], 1)
    assert completion_row["Section"] == key_stats.SECTION_G
