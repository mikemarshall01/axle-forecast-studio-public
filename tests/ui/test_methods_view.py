"""Tests for the "How it works" lenses (docs/DASHBOARD_DESIGN.md section 4.5).

Rules under test: each lens renders with and without a result (design
section 3.2: the page renders fully before any run); a missing explainer or
opening-note file degrades to a note instead of a crash; the Not modelled list carries
all eight items with a reason each; the "What this model cannot answer"
list carries all nine lines and is distinct from Not modelled and Known
limitations; the Assumptions lens draws no "Edit assumptions" button of its
own (layout ruling 3: the run bar's is always visible); and this view never
runs the model.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest
from fixtures.result_fixture import make_result

from axle_studio.model.assumptions import assumption_rows
from axle_studio.ui.views import methods


class RecordingStreamlit:
    """A minimal Streamlit stand-in that records every call.

    ``columns`` returns the same recording object for each column (both are
    then usable as ``with ...:`` context managers and share one call log),
    which is enough to exercise layout code without a real Streamlit
    runtime. ``button`` and ``segmented_control`` are configurable so a test
    can simulate a click or a chosen filter.
    """

    def __init__(
        self,
        *,
        button_clicks: frozenset[str] = frozenset(),
        segmented_control_values: dict[str, str] | None = None,
    ) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self._button_clicks = button_clicks
        self._segmented_control_values = segmented_control_values or {}

    def __getattr__(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self

        return record

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def columns(self, spec, **kwargs):
        self.calls.append(("columns", (spec,), kwargs))
        return [self for _ in spec]

    def button(self, label, *, key=None, **kwargs):
        self.calls.append(("button", (label,), {"key": key, **kwargs}))
        return key in self._button_clicks

    def segmented_control(self, label, *, key=None, default=None, **kwargs):
        self.calls.append(
            ("segmented_control", (label,), {"key": key, "default": default, **kwargs})
        )
        return self._segmented_control_values.get(key, default)

    def markdown_text(self) -> str:
        return "\n".join(str(args[0]) for name, args, _kwargs in self.calls if name == "markdown")

    def rendered_table(self) -> pd.DataFrame:
        """The first DataFrame passed to a recorded ``st.dataframe`` call."""

        return self.rendered_tables()[0]

    def rendered_tables(self) -> list[pd.DataFrame]:
        """Every DataFrame passed to a recorded ``st.dataframe`` call, in call order.

        The assumptions lens now makes up to two such calls: the always-
        visible shared table, then the collapsed cohort-rows table inside an
        expander when any cohort rows survive the evidence filter (goal
        review action 14).
        """

        return [args[0] for name, args, _kwargs in self.calls if name == "dataframe"]


ALL_LENSES = (
    methods.render_why_this_model,
    methods.render_forecast_method,
    methods.render_assumptions,
    methods.render_limits,
    methods.render_build_notes,
)


# --------------------------------------------------------------------------
# Renders with and without a result; no model runs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("render", ALL_LENSES)
def test_lens_renders_with_no_result(render) -> None:
    st = RecordingStreamlit()

    render(st, None)  # must not raise, and must not need a result

    assert st.calls  # something was drawn


@pytest.mark.parametrize("model", ["action", "no_action"])
@pytest.mark.parametrize("render", ALL_LENSES)
def test_lens_renders_with_a_result(render, model) -> None:
    st = RecordingStreamlit()
    result = make_result(model, evs=6, worlds=3)

    render(st, result)

    assert st.calls


def test_module_never_calls_a_model_runner() -> None:
    # AGENTS.md: a full Monte Carlo starts only from an explicit Run click.
    # This view has no route to one; check the source names no runner.
    source = inspect.getsource(methods)
    forbidden_names = (
        "run_explicit_action_forecast",
        "run_explicit_no_action_forecast",
        "run_model(",
    )
    for forbidden in forbidden_names:
        assert forbidden not in source


# --------------------------------------------------------------------------
# The forecast: explainer file present/missing
# --------------------------------------------------------------------------


def test_forecast_explainer_missing_shows_a_being_written_note(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(methods, "FORECAST_EXPLAINER_PATH", tmp_path / "missing.md")
    st = RecordingStreamlit()

    methods.render_forecast_method(st, None)

    assert "missing from this build" in st.markdown_text()


def test_forecast_explainer_present_is_shown_in_the_expander(monkeypatch, tmp_path) -> None:
    explainer = tmp_path / "how-the-forecast-works.md"
    explainer.write_text("# How the forecast works\n\nStep by step detail.", encoding="utf-8")
    monkeypatch.setattr(methods, "FORECAST_EXPLAINER_PATH", explainer)
    st = RecordingStreamlit()

    methods.render_forecast_method(st, None)

    assert "Step by step detail." in st.markdown_text()
    assert ("expander", ("Read the full explainer",), {}) in st.calls


def test_forecast_steps_and_equations_are_shown() -> None:
    st = RecordingStreamlit()

    methods.render_forecast_method(st, None)

    text = st.markdown_text()
    for title, _description, _equation, _equation_plain in methods._FORECAST_STEPS:
        assert title in text
    latex_calls = [args[0] for name, args, _kwargs in st.calls if name == "latex"]
    assert len(latex_calls) == 2  # only steps 4 and 5 carry an equation


def test_every_equation_has_a_plain_word_reading_right_under_it() -> None:
    # Fable clarity pass: a reader who does not read LaTeX still gets the
    # meaning, in the same order as the equation (title -> prose -> latex ->
    # plain-word markdown), naming what each symbol stands for.
    for title, _description, equation, equation_plain in methods._FORECAST_STEPS:
        if equation:
            assert equation_plain, f"{title} has an equation but no plain-word reading"
        else:
            assert equation_plain is None, f"{title} has no equation but carries a plain reading"

    st = RecordingStreamlit()
    methods.render_forecast_method(st, None)
    ordered = [
        (name, args[0])
        for name, args, _kwargs in st.calls
        if name in ("markdown", "latex") and args
    ]
    # The physics equation is followed by its plain-word reading before the
    # next step's markdown; same for the summaries equation.
    latex_positions = [index for index, (name, _value) in enumerate(ordered) if name == "latex"]
    assert len(latex_positions) == 2
    for position in latex_positions:
        following_name, following_text = ordered[position + 1]
        assert following_name == "markdown"
        assert following_text.startswith("In plain words:")

    text = st.markdown_text()
    assert "home charging efficiency" in text and "public" in text  # step 4's reading
    assert "percentile across simulated weeks" in text  # step 5's reading


@pytest.mark.parametrize("render", [methods.render_why_this_model, methods.render_forecast_method])
def test_opening_note_and_forecast_lens_point_on_to_overviews_start_here_tour(render) -> None:
    # How it works is the landing page (Fable landing pass) and Why this
    # model its landing lens; a first-time reader needs a way on to
    # Overview's own five-stop tour from both the note and The forecast.
    st = RecordingStreamlit(button_clicks=frozenset({"how-it-works-to-overview-tour"}))

    render(st, None)

    button_calls = {args[0]: kwargs for name, args, kwargs in st.calls if name == "button"}
    label = next(
        label for label in button_calls if label.startswith("Next: Overview ▸ At a glance")
    )
    assert button_calls[label]["on_click"] is methods._switch_to
    assert button_calls[label]["args"] == ("Overview", "At a glance")


# --------------------------------------------------------------------------
# Why this model: the opening note present/missing
# --------------------------------------------------------------------------


def test_why_this_model_note_present_is_shown_verbatim(monkeypatch, tmp_path) -> None:
    note = tmp_path / "why-this-model.md"
    note.write_text("### Why this model\n\nAxle asked two plain questions.", encoding="utf-8")
    monkeypatch.setattr(methods, "WHY_THIS_MODEL_PATH", note)
    st = RecordingStreamlit()

    methods.render_why_this_model(st, None)

    assert "Axle asked two plain questions." in st.markdown_text()


def test_why_this_model_note_missing_shows_a_note_not_a_crash(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(methods, "WHY_THIS_MODEL_PATH", tmp_path / "missing.md")
    st = RecordingStreamlit()

    methods.render_why_this_model(st, None)

    assert "missing from this build" in st.markdown_text()


def test_why_this_model_note_is_published_and_carries_mikes_two_added_lines() -> None:
    # The real file, as shipped: Mike's opening note with the two lines he
    # asked to be covered (one model for strategy and products; real data as
    # the test of the model).
    text = methods.WHY_THIS_MODEL_PATH.read_text(encoding="utf-8")
    assert text.startswith("### Why this model")
    assert "raise retention, cut churn and win new customers" in text
    assert "generate what we see in real data" in text


def test_pipeline_diagram_is_rendered_as_inline_svg_not_an_image_file() -> None:
    # Goal review action 21: "a small pipeline diagram -- plain SVG or
    # Plotly, no image files".
    st = RecordingStreamlit()

    methods.render_forecast_method(st, None)

    svg_calls = [
        (args, kwargs)
        for name, args, kwargs in st.calls
        if name == "markdown" and kwargs.get("unsafe_allow_html")
    ]
    assert svg_calls, "expected one markdown(..., unsafe_allow_html=True) call for the diagram"
    svg_text = svg_calls[0][0][0]
    assert svg_text.count("<svg") == 2  # desktop row layout, mobile grid layout
    assert "@media (max-width: 520px)" in svg_text
    for title, _line1, _line2 in methods._PIPELINE_STAGES:
        assert title in svg_text
    assert methods._PIPELINE_STAGES[3][0] == "Smart charging"  # highlighted, decision 0004 item 27


def test_step_3_describes_smart_charging_not_the_removed_18_00_cap() -> None:
    # decision 0004 item 38 supersedes the single 18:00 candidate and the
    # per-EV eligibility screen this step used to describe.
    title, description, _equation, _equation_plain = methods._FORECAST_STEPS[2]
    assert title == "3 Smart charging plan"
    assert "18:00" not in description
    assert "screened" not in description
    for phrase in ("expected departure", "cheapest forecast half-hours", "item 38"):
        assert phrase in description


def test_step_2_describes_evaluation_weeks_not_a_removed_planning_world() -> None:
    # Goal review N1: decision 0004 item 38 removed the separate planning
    # world; each evaluation week now samples and plans against its own
    # day-ahead path (item 48), matching how-the-forecast-works.md section 4.
    # The step may still name "planning world" while explaining it is gone
    # (that is the point); it must never describe one as still sampling.
    title, description, _equation, _equation_plain = methods._FORECAST_STEPS[1]
    assert title == "2 Evaluation weeks"
    assert "one planning world samples" not in description.lower()
    phrases = ("no separate planning world", "evaluation world", "day-ahead price path", "item 38")
    for phrase in phrases:
        assert phrase in description


# --------------------------------------------------------------------------
# Assumptions: result vs injectable provider, evidence filter
# --------------------------------------------------------------------------


def test_assumptions_table_comes_from_the_result_when_one_exists() -> None:
    st = RecordingStreamlit()
    result = make_result("action", evs=6, worlds=3)

    methods.render_assumptions(st, result)

    # The last table is the "Full source references" expander (review of
    # ecacff9): the full source text, off the visible table.
    *tables, references = st.rendered_tables()
    assert list(references.columns) == list(methods._REFERENCE_COLUMNS)
    assert len(references) == len(result.assumptions)
    assert sum(len(table) for table in tables) == len(result.assumptions)
    assert "Home charger power" in tables[0]["Label"].tolist()
    # The shared table drops "Archetype" (every row would say "All").
    assert list(tables[0].columns) == list(methods._SHARED_ASSUMPTION_TABLE_COLUMNS)
    # This fixture's assumptions tuple carries no "."-delimited cohort names
    # (unlike the real model's 146-record shape), so there is nothing to
    # collapse: one table, no archetype expander -- the expander must not
    # appear when there is nothing to put in it.
    assert len(tables) == 1
    expander_titles = [args[0] for name, args, _kwargs in st.calls if name == "expander"]
    assert not any(title.startswith("Archetype-specific values") for title in expander_titles)


def test_assumptions_table_splits_shared_and_cohort_rows_for_real_records() -> None:
    # The real model's 146-record shape (unlike the fixture above) does
    # carry per-cohort names, so this exercises the actual split.
    st = RecordingStreamlit()

    methods.render_assumptions(st, None)

    shared, cohort, _references = st.rendered_tables()
    assert len(shared) + len(cohort) == len(assumption_rows())
    assert list(cohort.columns) == list(methods._ASSUMPTION_TABLE_COLUMNS)
    assert set(cohort["Archetype"]) == {
        "Average (UK)",
        "Intelligent Octopus average",
        "Infrequent charging",
        "Infrequent driving",
        "Scheduled charging",
        "Always plugged-in",
    }


def test_assumptions_table_uses_the_injected_provider_before_a_run() -> None:
    provided = pd.DataFrame(
        [
            {
                "name": "example",
                "label": "Example assumption",
                "value": 1.5,
                "unit": "kW",
                "evidence": "illustrative",
                "source": "illustrative choice",
                "meaning": "An example.",
                "editable": True,
                "bounds": (0.0, 5.0),
                "group": "Fleet",
                "affects": "Nothing real.",
            }
        ],
        columns=list(methods.CONTRACT_ASSUMPTION_COLUMNS),
    )
    st = RecordingStreamlit()

    methods.render_assumptions(st, None, assumption_rows_provider=lambda: provided)

    table = st.rendered_table()
    assert table["Label"].tolist() == ["Example assumption"]
    assert table["Evidence"].tolist() == ["Illustrative"]


def test_default_assumption_provider_shows_every_default_record_before_a_run() -> None:
    st = RecordingStreamlit()

    methods.render_assumptions(st, None)

    shared, cohort, _references = st.rendered_tables()
    assert len(shared) + len(cohort) == len(assumption_rows())
    assert list(shared.columns) == list(methods._SHARED_ASSUMPTION_TABLE_COLUMNS)
    assert list(cohort.columns) == list(methods._ASSUMPTION_TABLE_COLUMNS)
    captions = [args[0] for name, args, _kwargs in st.calls if name == "caption"]
    assert any("default values" in caption for caption in captions)


def test_assumptions_table_adds_an_archetype_column_from_the_record_name() -> None:
    # Goal review action 14: a cohort record's name carries the cohort id as
    # one "."-delimited segment, in a different position for different
    # record shapes ("average_uk.daily_miles_mean" vs "sd_minutes.average_uk").
    assert methods._archetype_label("average_uk.daily_miles_mean") == "Average (UK)"
    assert methods._archetype_label("sd_minutes.average_uk") == "Average (UK)"
    assert methods._archetype_label("seed") == "All"
    assert methods._archetype_label("study_base_temperature_c.day_3") == "All"


def test_cohort_rows_collapse_into_an_expander_by_default() -> None:
    st = RecordingStreamlit()

    methods.render_assumptions(st, None)

    expander_titles = [args[0] for name, args, _kwargs in st.calls if name == "expander"]
    assert any(title.startswith("Archetype-specific values (") for title in expander_titles)
    assert any("across six archetypes" in title for title in expander_titles)


def test_assumptions_table_is_full_width_not_the_narrow_prose_column() -> None:
    # Goal review action 14: the table used to sit in _prose_column's 3-of-4
    # width, clipping the Meaning column; st.columns([3, 1]) must not wrap
    # either st.dataframe call any more.
    st = RecordingStreamlit()

    methods.render_assumptions(st, None)

    columns_calls = [args for name, args, _kwargs in st.calls if name == "columns"]
    assert ([3, 1],) not in columns_calls


def test_assumptions_evidence_filter_narrows_the_table() -> None:
    st = RecordingStreamlit(segmented_control_values={"how-it-works-evidence-filter": "Source"})
    result = make_result("action", evs=6, worlds=3)

    methods.render_assumptions(st, result)

    table = st.rendered_table()
    assert not table.empty
    assert set(table["Evidence"]) == {"Source"}


def test_assumptions_no_longer_splits_into_filter_and_edit_columns() -> None:
    # Layout ruling 3 (30 Sep 2026): the filter and the (now-removed) Edit
    # assumptions button used to share the header's controls slot via
    # st.columns([3, 2]); the filter now draws directly, its own row.
    st = RecordingStreamlit()

    methods.render_assumptions(st, None)

    columns_calls = [args for name, args, _kwargs in st.calls if name == "columns"]
    assert ([3, 2],) not in columns_calls


def test_assumptions_draws_no_edit_button_the_run_bars_one_is_always_visible() -> None:
    # Layout ruling 3: a second "Edit assumptions" button here duplicated
    # the run bar's own button (every page, always visible), which opens
    # the identical dialog -- so this lens draws none.
    st = RecordingStreamlit()

    methods.render_assumptions(st, None)

    button_labels = [args[0] for name, args, _kwargs in st.calls if name == "button"]
    assert "Edit assumptions" not in button_labels


# --------------------------------------------------------------------------
# Limits: Not modelled list and known limitations
# --------------------------------------------------------------------------


def test_not_modelled_list_has_every_item_with_a_reason() -> None:
    st = RecordingStreamlit()

    methods.render_limits(st, None)

    text = st.markdown_text()
    expected_items = [
        "DFS",
        "Frequency response",
        "Capacity market",
        "Local flexibility",
        "V2G",
        "Geography",
        "Settlement cash",
        "Charging taper",
        # Supplier contract v1 §15, after "Charging taper" in its order.
        "Retail tariff",
        "Supplier intraday re-hedging",
        "Sign-up, opt-outs and retention",
        "Command latency",
        "V2G wear, batteries and heat pumps",
        "Attach-rate uplift",
        "Balancing Mechanism",
    ]
    assert [item for item, _reason in methods.NOT_MODELLED] == expected_items
    for item, reason in methods.NOT_MODELLED:
        assert item in text
        # Citations move to the tooltip on screen (polish plan G7).
        assert methods.split_citations(reason)[0] in text


def test_cannot_answer_list_has_nine_plain_english_lines() -> None:
    st = RecordingStreamlit()

    methods.render_limits(st, None)

    text = st.markdown_text()
    assert "**What this model cannot answer**" in text
    assert len(methods.CANNOT_ANSWER) == 9
    for line in methods.CANNOT_ANSWER:
        assert line in text


def test_cannot_answer_list_is_distinct_from_not_modelled_and_known_limitations() -> None:
    not_modelled_items = {item for item, _reason in methods.NOT_MODELLED}
    assert not not_modelled_items & set(methods.CANNOT_ANSWER)
    assert not set(methods.KNOWN_LIMITATIONS) & set(methods.CANNOT_ANSWER)


def test_known_limitations_are_shown() -> None:
    st = RecordingStreamlit()

    methods.render_limits(st, None)

    text = st.markdown_text()
    for line in methods.KNOWN_LIMITATIONS:
        assert methods.split_citations(line)[0] in text
    assert "decision 0004" not in text


def test_split_citations_moves_decision_numbers_to_the_tooltip() -> None:
    # Polish plan G7: no "decision 0004 item N" on screen; plain words in the
    # same parenthesis stay, and a citation-only parenthesis goes entirely.
    clean, tooltip = methods.split_citations(
        "One power (7 kW by default, editable; decision 0004 item 34). Noise is AR(1) "
        "(decision 0004 item 48)."
    )
    assert clean == "One power (7 kW by default, editable). Noise is AR(1)."
    assert tooltip == "Source: decision 0004 item 34; decision 0004 item 48."
    assert methods.split_citations("No citation here.") == ("No citation here.", None)


def table_has_citation(table: pd.DataFrame) -> bool:
    return table["Source"].str.contains(r"decision \d{4}|item \d", case=False).any()


def test_short_source_gives_a_plain_name_and_keeps_the_detail_column() -> None:
    # Polish plan G9: a clean Source column; the full text stays beside it.
    assert methods.short_source("'Source archetypes'!C6") == "Axle cohort sheet"
    assert methods.short_source("Nord Pool operational message, 8 June 2022: cap") == (
        "Nord Pool market notice"
    )
    assert methods.short_source("decision 0004 item 32") == "Project decision"
    assert not table_has_citation(methods._assumption_rows(methods.assumption_rows()))
    assert methods.short_source("calibrated to the Elexon fit (BMRS data © Elexon)") == (
        "Elexon price fit, © Elexon"
    )
    assert methods.short_source("illustrative choice (decision 0004 item 56)") == (
        "Illustrative choice"
    )
    assert methods.short_source("CNZ May 2022 report p.13: reported median 52%") == (
        "CNZ report (May 2022)"
    )
    table = methods._assumption_rows(methods.assumption_rows())
    assert table["Source"].str.len().max() <= 30
    assert (table["Source detail"] != "").all()


def test_known_limitations_name_the_herding_finding_without_a_fixed_figure() -> None:
    # Decision 0004 item 48: each simulated week now has its own day-ahead
    # price path, so the herding peak's clock time and size vary week to
    # week; the limitation names the mechanism, not one run's stale figures
    # or a cap that was never built (item 48 retired that plan outright).
    text = " ".join(methods.KNOWN_LIMITATIONS)
    assert "No site or network limit" in text
    assert "982 kW" not in text and "537 kW" not in text and "05:30" not in text
    assert "one shared price signal per week" in text
    assert "not corrected with a cap" in text
    assert "decision 0004 item 48" in text
    # Item 52 retired item 49's study-end rule and item 48's whole-week
    # look-ahead: plans see only published prices.
    assert "decision 0004 item 49" not in text
    assert "13:00 London the day before" in text and "noon to noon" in text


def test_limits_lens_renders_the_same_with_a_result() -> None:
    st = RecordingStreamlit()
    result = make_result("no_action", evs=6, worlds=3)

    methods.render_limits(st, result)  # must not special-case a present result

    assert methods.NOT_MODELLED[0][0] in st.markdown_text()


def test_limits_lens_references_the_cohort_crosswalk_in_an_expander() -> None:
    st = RecordingStreamlit()

    methods.render_limits(st, None)

    expander_titles = [args[0] for name, args, _kwargs in st.calls if name == "expander"]
    assert "CNZ report and Axle sheet crosswalk" in expander_titles


# --------------------------------------------------------------------------
# How it was built: build notes present/missing, explainer present/missing
# --------------------------------------------------------------------------


def test_build_notes_missing_shows_no_draft_narrative(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(methods, "BUILD_NOTES_PATH", tmp_path / "missing.md")
    monkeypatch.setattr(methods, "BUILD_EXPLAINER_PATH", tmp_path / "missing-explainer.md")
    st = RecordingStreamlit()

    methods.render_build_notes(st, None)

    captions = [args[0] for name, args, _kwargs in st.calls if name == "caption"]
    assert "Build notes are not published yet." in captions
    assert "missing from this build" in st.markdown_text()


def test_build_notes_present_is_shown_verbatim(monkeypatch, tmp_path) -> None:
    notes = tmp_path / "build-notes.md"
    notes.write_text("What I spent time on, and what I left out.", encoding="utf-8")
    monkeypatch.setattr(methods, "BUILD_NOTES_PATH", notes)
    st = RecordingStreamlit()

    methods.render_build_notes(st, None)

    assert "What I spent time on, and what I left out." in st.markdown_text()


def test_build_explainer_present_is_shown_in_the_expander(monkeypatch, tmp_path) -> None:
    explainer = tmp_path / "how-it-was-built.md"
    explainer.write_text("Process, architecture and key decisions.", encoding="utf-8")
    monkeypatch.setattr(methods, "BUILD_EXPLAINER_PATH", explainer)
    st = RecordingStreamlit()

    methods.render_build_notes(st, None)

    assert "Process, architecture and key decisions." in st.markdown_text()


def test_explainer_path_constants_point_under_docs_explainers() -> None:
    paths = (
        methods.WHY_THIS_MODEL_PATH,
        methods.FORECAST_EXPLAINER_PATH,
        methods.BUILD_EXPLAINER_PATH,
        methods.BUILD_NOTES_PATH,
    )
    expected_dir = Path(methods.__file__).resolve().parents[4] / "docs" / "explainers"
    for path in paths:
        assert path.parent == expected_dir
