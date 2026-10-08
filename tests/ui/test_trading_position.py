"""Trading lens 2, Position (trading contract v1 sections 4.2, 4.4-4.6, 5.1-5.3)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from fixtures.result_fixture import make_result
from kpi_calls import kpi_calls

from axle_studio.ui.style import CHART_HEIGHTS, PATH_LABELS, SERIES_COLOURS, percent
from axle_studio.ui.views import trading_position
from axle_studio.ui.views.trading_position import (
    STRATEGY_COLOURS,
    STRATEGY_LABELS,
    STRATEGY_ORDER,
    _night_options,
    _night_slots,
    render_trading_position,
)


class RecordingStreamlit:
    """Fake ``st``; ``selectbox`` returns whichever configured choice matches its options."""

    def __init__(
        self, calls: list | None = None, *, night: int | None = None, strategy: str | None = None
    ) -> None:
        self.calls: list[tuple[str, tuple, dict]] = calls if calls is not None else []
        self._night = night
        self._strategy = strategy

    def __getattr__(self, name: str):
        def record(*args: object, **kwargs: object) -> object:
            self.calls.append((name, args, kwargs))
            return self

        return record

    def __enter__(self) -> "RecordingStreamlit":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def columns(self, spec: int | list) -> tuple["RecordingStreamlit", ...]:
        count = spec if isinstance(spec, int) else len(spec)
        self.calls.append(("columns", (spec,), {}))
        return tuple(
            RecordingStreamlit(self.calls, night=self._night, strategy=self._strategy)
            for _ in range(count)
        )

    def selectbox(self, *args: object, **kwargs: object) -> object:
        self.calls.append(("selectbox", args, kwargs))
        options = kwargs.get("options", [])
        for configured in (self._night, self._strategy):
            if configured is not None and configured in options:
                return configured
        index = kwargs.get("index") or 0
        return options[index] if options else None


def calls_for(st: RecordingStreamlit, name: str) -> list[tuple[tuple, dict]]:
    return [(args, kwargs) for call_name, args, kwargs in st.calls if call_name == name]


def figures_from(st: RecordingStreamlit) -> list:
    return [args[0] for args, _ in calls_for(st, "plotly_chart")]


def test_night_options_covers_every_session_night_once() -> None:
    result = make_result("action")
    options = _night_options(result.study_slots)
    assert [index for index, _ in options] == list(range(7))
    assert all(isinstance(label, str) and label for _, label in options)


def test_night_slots_filters_to_one_world_and_night() -> None:
    result = make_result("action")
    night = _night_slots(result.deviation_world_slot, world_id=0, night_index=2)
    assert (night["world_id"] == 0).all()
    assert (night["night_index"] == 2).all()
    assert list(night["slot_index"]) == sorted(night["slot_index"])


def test_render_both_models_without_crashing() -> None:
    action_result = make_result("action")
    no_action_result = make_result("no_action")

    action_st = RecordingStreamlit()
    render_trading_position(action_st, action_result)
    assert len(figures_from(action_st)) == 4, (
        "night profile, position fan, open-position profile, settled band"
    )

    no_action_st = RecordingStreamlit()
    render_trading_position(no_action_st, no_action_result)
    assert figures_from(no_action_st) == []
    assert calls_for(no_action_st, "info")[0][0] == ("This result has no Axle action.",)


def test_night_chart_shows_baseline_unmanaged_and_metered_lines() -> None:
    result = make_result("action")
    st = RecordingStreamlit(night=0)

    render_trading_position(st, result)

    night_figure = figures_from(st)[0]
    names = {t.name for t in night_figure.data if t.showlegend is not False and t.name}
    assert "Baseline (BL01-lite)" in names
    assert PATH_LABELS["normal"] in names
    assert PATH_LABELS["selected"] in names
    representative = result.representative_world_id
    night = _night_slots(result.deviation_world_slot, representative, 0)
    metered_line = next(t for t in night_figure.data if t.name == PATH_LABELS["selected"])
    assert pd.Series(metered_line.y).round(3).tolist() == night["metered_kw"].round(3).tolist()
    # The shaded fill is named for the smart path, not "Sold turn-down" alone
    # (dashboard_lint's teal-reserved rule: a teal trace's name must say so).
    assert any(t.name == "Smart settled turn-down" for t in night_figure.data)


def test_strategy_selector_changes_the_final_position_series() -> None:
    result = make_result("action")
    representative = result.representative_world_id
    night = _night_slots(result.deviation_world_slot, representative, 0)

    full_st = RecordingStreamlit(night=0, strategy="full")
    render_trading_position(full_st, result)
    full_figure = figures_from(full_st)[1]
    full_line = next(t for t in full_figure.data if "final position" in (t.name or "").lower())
    assert (
        pd.Series(full_line.y).round(3).tolist()
        == (night["position_full_kwh"] / 0.5).round(3).tolist()
    )
    assert full_line.line.color == STRATEGY_COLOURS["full"]

    pf_st = RecordingStreamlit(night=0, strategy="perfect_foresight")
    render_trading_position(pf_st, result)
    pf_figure = figures_from(pf_st)[1]
    pf_line = next(t for t in pf_figure.data if "perfect foresight" in (t.name or "").lower())
    assert (
        pd.Series(pf_line.y).round(3).tolist()
        == (night["position_perfect_foresight_kwh"] / 0.5).round(3).tolist()
    )
    assert pf_line.line.color == STRATEGY_COLOURS["perfect_foresight"]


def test_full_strategy_colour_never_reuses_the_reserved_teal() -> None:
    # dashboard_lint's teal-reserved rule: teal is reserved sitewide for the
    # physical smart-charging dispatch path (decision 0004 item 22), so a
    # trading strategy -- even the one named "full" -- must use a different
    # colour, on every chart that colours by STRATEGY_COLOURS.
    assert STRATEGY_COLOURS["full"] != SERIES_COLOURS["selected"]


def test_kpi_tiles_report_sold_settled_and_imbalance_for_the_chosen_strategy() -> None:
    result = make_result("action")
    st = RecordingStreamlit(night=0, strategy="da_only")

    render_trading_position(st, result)

    # The first row; the baseline-erosion row (item 67) follows at the foot.
    tiles = kpi_calls(st)[:3]
    labels = [tile[0] for tile in tiles]
    assert labels == ["Sold turn-down", "Settled turn-down", "Imbalance, this strategy"]
    assert tiles[0][3] == f"{STRATEGY_LABELS['da_only']} strategy, this night"


def test_settled_band_is_identical_across_strategies() -> None:
    # Settled volume is a physical quantity, the same for every strategy
    # (trading contract v1 4.6): the chart needs no strategy picker.
    result = make_result("action")
    da_only_st = RecordingStreamlit(night=0, strategy="da_only")
    render_trading_position(da_only_st, result)

    full_st = RecordingStreamlit(night=0, strategy="full")
    render_trading_position(full_st, result)

    settled_da = figures_from(da_only_st)[3]
    settled_full = figures_from(full_st)[3]
    line_da = next(t for t in settled_da.data if t.name == "Settled turn-down" and t.showlegend)
    line_full = next(t for t in settled_full.data if t.name == "Settled turn-down" and t.showlegend)
    assert list(line_da.y) == list(line_full.y)


def test_strategy_control_is_a_selectbox_not_a_segmented_control() -> None:
    # Coordinator review of 3ccdc20 (BLOCKING a): a segmented control's pills
    # sit side by side and overflowed the header's controls slot at 1280 px
    # ("Perfect foresight" clipped to "Per"); a selectbox elides its own
    # text instead, so it cannot spill past its column at any width.
    result = make_result("action")
    st = RecordingStreamlit(night=0, strategy="perfect_foresight")

    render_trading_position(st, result)

    assert calls_for(st, "segmented_control") == []
    strategy_calls = [
        (args, kwargs)
        for args, kwargs in calls_for(st, "selectbox")
        if kwargs.get("key") == "trading-position-strategy"
    ]
    assert len(strategy_calls) == 1
    _, kwargs = strategy_calls[0]
    assert list(kwargs["options"]) == list(STRATEGY_ORDER) or set(kwargs["options"]) == set(
        STRATEGY_ORDER
    )


def test_position_fan_chart_uses_the_dual_axis_height() -> None:
    result = make_result("action")
    st = RecordingStreamlit(night=0, strategy="full")

    render_trading_position(st, result)

    fan_figure = figures_from(st)[1]
    assert fan_figure.layout.height == CHART_HEIGHTS["time_series_dual_axis"]


def test_open_position_chart_has_one_line_per_strategy_and_pf_is_zero() -> None:
    result = make_result("action")
    st = RecordingStreamlit(night=0, strategy="full")

    render_trading_position(st, result)

    profile_figure = figures_from(st)[2]
    names = {t.name for t in profile_figure.data}
    assert names == {STRATEGY_LABELS[s] for s in STRATEGY_ORDER}
    pf_line = next(t for t in profile_figure.data if t.name == STRATEGY_LABELS["perfect_foresight"])
    assert all(value == 0 for value in pf_line.y)


def test_view_makes_no_model_calls() -> None:
    source = Path(trading_position.__file__).read_text(encoding="utf-8")
    assert "axle_studio.model" not in source
    assert "run_forecast" not in source


def test_baseline_erosion_tiles_sit_beside_the_default_settled_value() -> None:
    # Decision 0004 item 67: default and smart-night settled value, and the
    # share lost, read from trading_kpis without recalculating anything.
    result = make_result("action")
    st = RecordingStreamlit(night=0, strategy="full")

    render_trading_position(st, result)

    tiles = kpi_calls(st)[3:]
    assert [tile[0] for tile in tiles] == [
        "Evening value lost",
        "Evening value, default",
        "Evening value, smart nights",
    ]
    kpis = result.trading_kpis.loc[result.trading_kpis["strategy"].eq("full")].set_index("metric")
    share = kpis.loc["evening_baseline_erosion_share", "mean"]
    assert tiles[0][1] == percent(100 * share)
    captions = " ".join(str(args[0]) for args, _ in calls_for(st, "caption"))
    assert "unmanaged nights only, like BL01" in captions
    assert "spurious settlement" in captions
    # The whole-week figures sit in the expander's table, not in a tile.
    assert any(args and args[0] == "Whole week" for args, _ in calls_for(st, "expander"))
    tables = [args[0] for args, _ in calls_for(st, "dataframe")]
    week = next(t for t in tables if "Whole week" in t.columns)
    assert list(week["Whole week"]) == [
        "Settled value, default baseline",
        "Settled value, smart-night baseline",
        "Value lost to baseline erosion",
    ]


def test_baseline_erosion_section_is_absent_on_an_older_result() -> None:
    kpis = make_result("action").trading_kpis
    older = kpis.loc[~kpis["metric"].str.contains("erosion|eroded")]
    st = RecordingStreamlit(night=0, strategy="full")

    trading_position._baseline_erosion(st, older, "Evidence: synthetic.")

    assert st.calls == []


def _with_commitment(result, rule: str, *, dispatched: bool):
    """The fixture result with its commitment-rule record set to ``rule`` and,
    when ``dispatched``, a two-cohort ``dispatch_split`` at share 0.6.

    The fixture mirror has no ``dispatch_split`` field, so the copy is a
    namespace of its fields plus that one (the view reads it by attribute).
    """

    records = tuple(
        dataclasses.replace(r, value=rule) if r.name == "trading.commitment_rule" else r
        for r in result.assumptions
    )
    if not any(r.name == "trading.commitment_rule" for r in records):
        template = result.assumptions[0]
        records += (dataclasses.replace(template, name="trading.commitment_rule", value=rule),)
    split = (
        pd.DataFrame(
            {
                "cohort_id": ["a", "b"],
                "ev_count": [6, 4],
                "locked_count": [4, 2],
                "free_count": [2, 2],
                "commitment_share": [0.6, 0.6],
                "replan_threshold_gbp_per_mwh": [10.0, 10.0],
                "evidence_kind": ["synthetic", "synthetic"],
            }
        )
        if dispatched
        else None
    )
    fields = {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}
    return SimpleNamespace(**(fields | {"assumptions": records, "dispatch_split": split}))


def test_newsvendor_caption_says_the_locked_share_is_c_and_the_position_is_not() -> None:
    # intraday-dispatch-v1 §2 and §10: in newsvendor mode with dispatch on,
    # Position says the locked share is c and the position is not c x forecast.
    result = _with_commitment(make_result("action"), "newsvendor", dispatched=True)
    st = RecordingStreamlit()
    render_trading_position(st, result)

    captions = [args[0] for args, _ in calls_for(st, "caption")]
    note = [c for c in captions if c.startswith("Newsvendor:")]
    assert len(note) == 1
    assert "6 of 10 EVs locked (share 60%)" in note[0]
    assert "not 60% of the forecast" in note[0]
    # Copy rule: captions at most 140 characters, even at a 100,000 EV fleet.
    assert len(note[0].replace("6 of 10", "100,000 of 100,000")) <= 140


def test_newsvendor_caption_absent_under_fixed_share_or_without_dispatch() -> None:
    base = make_result("action")
    for rule, dispatched in (("fixed_share", True), ("newsvendor", False)):
        result = _with_commitment(base, rule, dispatched=dispatched)
        assert trading_position._newsvendor_caption(result) is None
        st = RecordingStreamlit()
        render_trading_position(st, result)
        captions = [args[0] for args, _ in calls_for(st, "caption")]
        assert not any(c.startswith("Newsvendor:") for c in captions)


def test_sold_turn_down_fill_has_no_gaps_and_zero_height_outside_sold_slots() -> None:
    # A NaN-gapped "tonexty" fill joined its pieces with diagonal wedges
    # across unsold half-hours; the band now collapses onto the smart line.
    result = make_result("action")
    night = _night_slots(result.deviation_world_slot, result.representative_world_id, 0)
    figure = trading_position._night_figure(night)
    low, high = figure.data[0], figure.data[1]
    low_y, high_y = pd.Series(low.y, dtype=float), pd.Series(high.y, dtype=float)

    assert low_y.notna().all() and high_y.notna().all()
    unsold = (night["settled_kwh"] <= 0.0).to_numpy()
    assert (low_y[unsold].to_numpy() == high_y[unsold].to_numpy()).all()
    sold = ~unsold
    assert (low_y[sold].to_numpy() == night["baseline_kw"].to_numpy()[sold]).all()
