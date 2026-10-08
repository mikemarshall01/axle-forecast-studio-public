"""Tests for shared presentation settings, not model arithmetic."""

import colorsys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go

from axle_studio.ui.style import (
    APP_CSS,
    ARCHETYPE_COLOURS,
    AXLE_LIGHT_TEMPLATE,
    AXLE_TEMPLATE,
    BAND_ALPHA,
    BORDER,
    CHART_HEIGHTS,
    LIGHT_BACKGROUND,
    LIGHT_GRID,
    LIGHT_INK,
    LIGHT_MUTED_INK,
    MUTED_INK,
    PATH_DISPLAY_ORDER,
    PATH_LABELS,
    PATH_STYLES,
    PLUGGED_ALPHA,
    RISK,
    SEQUENTIAL,
    SERIES_COLOURS,
    UNAVAILABLE,
    UNMANAGED_GLOSSARY,
    apply_notebook_style,
    assumption_value,
    band_and_line,
    band_fill,
    format_quantity,
    hover_time_labels,
    kw,
    london_time_axis,
    money,
    percent,
    present_paths,
    style_figure,
)


def _contrast_ratio(foreground: str, background: str) -> float:
    """WCAG relative contrast between two ``#RRGGBB`` colours."""

    def luminance(colour: str) -> float:
        channels = [int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return sum(weight * channel for weight, channel in zip((0.2126, 0.7152, 0.0722), linear))

    lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_paths_differ_by_colour_and_width_and_normal_is_solid() -> None:
    assert PATH_STYLES["normal"]["color"] != PATH_STYLES["selected"]["color"]
    assert PATH_STYLES["normal"]["width"] != PATH_STYLES["selected"]["width"]
    assert PATH_STYLES["normal"]["dash"] == "solid"  # decision 0004 item 27
    assert PATH_STYLES["selected"]["color"] == "#2DD4BF"


def test_timed_path_is_not_teal_and_has_its_own_dash() -> None:
    # Decision 0007: the optional third "Timed tariff" path never uses teal
    # (reserved for Smart) and stays distinguishable in greyscale too.
    teal = PATH_STYLES["selected"]["color"]
    assert PATH_STYLES["timed"]["color"] != teal
    assert PATH_STYLES["timed"]["dash"] != PATH_STYLES["normal"]["dash"]
    assert PATH_STYLES["timed"]["color"] in SERIES_COLOURS.values()


def test_series_lines_clear_4_5_to_1_on_the_dark_surfaces() -> None:
    # The app is dark-only (config.toml), so only the dark canvas and surface count.
    for name in ("normal", "selected", "difference", "timed", "observed", "flag"):
        for background in ("#0E1117", "#171B23"):
            assert _contrast_ratio(SERIES_COLOURS[name], background) >= 4.5, name


def test_band_fill_uses_the_band_opacity() -> None:
    assert BAND_ALPHA == 0.55
    assert band_fill("#2DD4BF") == "rgba(45,212,191,0.55)"


def _composited(foreground: str, background: str, alpha: float) -> str:
    """A translucent ``foreground`` over an opaque ``background``, as the eye sees it."""

    fg = [int(foreground[index : index + 2], 16) for index in (1, 3, 5)]
    bg = [int(background[index : index + 2], 16) for index in (1, 3, 5)]
    blended = [alpha * f + (1 - alpha) * b for f, b in zip(fg, bg, strict=True)]
    return "#{:02X}{:02X}{:02X}".format(*(round(channel) for channel in blended))


def test_band_and_plugged_fills_clear_3_to_1_non_text_contrast_on_the_dark_surfaces() -> None:
    # Final critique B-21 (computed): the old BAND_ALPHA 0.3 and PLUGGED_ALPHA
    # 0.35 composited to 1.78-1.99:1 against the dark canvas, below WCAG
    # 1.4.11's 3:1 floor for a graphical object needed to understand the
    # content -- a P10-P90 band carries the forecast's uncertainty, not
    # decoration. This checks the *composited* fill, not the raw token, the
    # same distinction that missed the regression the first time.
    for name in ("normal", "selected", "difference"):
        for background in ("#0E1117", "#171B23"):
            blended = _composited(SERIES_COLOURS[name], background, BAND_ALPHA)
            assert _contrast_ratio(blended, background) >= 3.0, name
    for background in ("#0E1117", "#171B23"):
        blended = _composited(SERIES_COLOURS["plugged"], background, PLUGGED_ALPHA)
        assert _contrast_ratio(blended, background) >= 3.0, "plugged"


def test_styled_figure_has_explicit_height_legend_below_and_keeps_view_choices() -> None:
    figure = go.Figure(layout={"yaxis": {"title": {"text": "kW"}}, "margin": {"r": 56}})

    layout = style_figure(figure, height=CHART_HEIGHTS["time_series"]).layout

    assert layout.height == 340
    # Legend anchored to the figure's own bottom, not a fraction of the plot
    # height (polish plan G2).
    assert layout.template.layout.legend.yref == "container"
    assert layout.template.layout.legend.y == 0
    assert layout.template.layout.legend.yanchor == "bottom"
    assert layout.template.layout.hovermode == "x unified"
    assert layout.yaxis.title.text == "kW"
    assert layout.margin.r == 56


def test_styled_figure_sets_transparent_backgrounds_explicitly() -> None:
    # Streamlit writes its own chart background unless the figure sets one;
    # charts are flat on the page canvas (polish plan G1, D10).
    layout = style_figure(go.Figure(), height=CHART_HEIGHTS["histogram"]).layout

    assert layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert AXLE_TEMPLATE.layout.xaxis.gridcolor == BORDER == "#2A3140"


def test_chart_text_is_one_12px_size_with_muted_axis_titles() -> None:
    layout = AXLE_TEMPLATE.layout
    assert layout.font.size == 12
    assert layout.legend.font.size == 12
    assert layout.hoverlabel.font.size == 12
    for axis in (layout.xaxis, layout.yaxis):
        assert axis.tickfont.size == 12
        assert axis.title.font.size == 12
        assert axis.title.font.color == MUTED_INK


def _two_paths() -> go.Figure:
    figure = go.Figure()
    for path in ("normal", "selected"):
        band_and_line(
            figure,
            [1, 2],
            [0, 1],
            [1, 2],
            [2, 3],
            name=PATH_LABELS[path],
            colour=PATH_STYLES[path]["color"],
        )
    return figure


def test_bottom_margin_fits_ticks_only_when_there_is_no_legend_or_title() -> None:
    # Audit: a fixed 104 px margin left plots at 60-65% of their height even
    # with no legend. One trace, no x title: only the tick row is reserved.
    layout = style_figure(go.Figure(go.Scatter(x=[1], y=[1])), height=340).layout

    assert layout.margin.b == 24


def test_bottom_margin_adds_title_and_legend_rows() -> None:
    no_title = style_figure(_two_paths(), height=340).layout.margin.b
    titled = _two_paths().update_xaxes(title="London time")
    with_title = style_figure(titled, height=340).layout.margin.b

    assert no_title == 24 + 4 + 28  # ticks + gap + one legend row
    assert with_title == no_title + 20  # the x-axis title gets its own row


def test_long_legend_reserves_the_rows_it_wraps_to_at_390px() -> None:
    figure = go.Figure(
        [go.Scatter(x=[1], y=[1], name=f"A fairly long legend entry {index}") for index in range(4)]
    )

    assert style_figure(figure, height=340).layout.margin.b >= 24 + 4 + 3 * 28


def test_a_views_own_bottom_margin_wins() -> None:
    figure = go.Figure(layout={"margin": {"b": 90}})

    assert style_figure(figure, height=340).layout.margin.b == 90


def test_band_and_line_is_one_legend_entry_per_path() -> None:
    figure = _two_paths()

    listed = [trace.name for trace in figure.data if trace.showlegend is not False]
    assert listed == ["Unmanaged", "Smart"]
    assert len(figure.data) == 6
    band_low, band_high, line = figure.data[:3]
    assert band_low.legendgroup == band_high.legendgroup == line.legendgroup == "Unmanaged"
    assert band_high.fill == "tonexty" and band_high.fillcolor == band_fill("#9AA9BF")
    assert band_low.hoverinfo == "skip" and band_high.hoverinfo == "skip"
    assert line.line.width == 2.0


def test_band_and_line_places_traces_on_a_secondary_axis() -> None:
    figure = band_and_line(
        go.Figure(), [1], [0], [1], [2], name="Smart", colour="#2DD4BF", yaxis="y2"
    )

    assert {trace.yaxis for trace in figure.data} == {"y2"}


def test_london_axis_defaults_to_one_tick_per_day_for_390px_safety() -> None:
    # One week, decision 0004 item 1: 336 UTC half-hour slots from London
    # local midnight. Review finding: the old every-6-hours default gave 28
    # ticks that overlapped at 390 px on One EV, Response and Fleet week;
    # seven daily ticks read at both 1440 and 390 px (Plotly cannot read the
    # viewport, so one density must serve both).
    local_start = pd.Timestamp("2026-09-28 00:00", tz="Europe/London")
    starts = pd.Series(pd.date_range(local_start.tz_convert("UTC"), periods=336, freq="30min"))

    axis = london_time_axis(starts)

    assert axis["ticktext"] == ["Mon 28", "Tue 29", "Wed 30", "Thu 1", "Fri 2", "Sat 3", "Sun 4"]
    assert len(axis["tickvals"]) == 7
    assert axis["tickangle"] == 0
    assert axis["automargin"] is True


def test_london_axis_compact_true_drops_the_day_of_month() -> None:
    # Goal review V3: a dual-axis chart's narrower plot area left as little
    # as 2 px between adjacent day labels at 390 px ("Tue 29" and "Wed 30"
    # ran together); a single week never repeats a weekday, so
    # ``compact=True`` drops the day-of-month number rather than shortening
    # to fewer ticks (which would lose a day's data point on some charts).
    local_start = pd.Timestamp("2026-09-28 00:00", tz="Europe/London")
    starts = pd.Series(pd.date_range(local_start.tz_convert("UTC"), periods=336, freq="30min"))

    axis = london_time_axis(starts, compact=True)

    assert axis["ticktext"] == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert len(axis["tickvals"]) == 7
    assert axis["tickangle"] == 0
    assert axis["automargin"] is True


def test_london_axis_compact_variant_keeps_hourly_ticks_across_clock_change() -> None:
    # 25 October 2026: London moves from BST (UTC+1) to GMT at 02:00 local.
    # A caller that wants intraday ticks opts in with every_hours; the
    # default (tested above) never produces this density.
    starts = pd.Series(pd.date_range("2026-10-24T23:00Z", "2026-10-25T13:00Z", freq="30min"))

    axis = london_time_axis(starts, every_hours=6)

    assert axis["ticktext"] == ["Sun 25", "06:00", "12:00"]
    assert [str(tick) for tick in axis["tickvals"]] == [
        "2026-10-24 23:00:00+00:00",
        "2026-10-25 06:00:00+00:00",
        "2026-10-25 12:00:00+00:00",
    ]


def test_hover_labels_show_london_time_then_utc() -> None:
    labels = hover_time_labels(pd.Series(pd.to_datetime(["2026-09-28T17:30Z"], utc=True)))

    assert labels == ["Mon 28 18:30 (17:30 UTC)"]


def test_quantity_formatting_preserves_missing_values() -> None:
    assert format_quantity(None, "kWh") == "Unavailable"
    assert format_quantity(float("nan"), "kWh") == "Unavailable"
    assert format_quantity(0, "kWh") == "0.0 kWh"
    assert format_quantity(1234.567, "GBP", decimals=2) == "1,234.57 GBP"


def test_quantity_uses_true_minus_and_no_space_before_percent() -> None:
    # Audit: "82 %" beside "36%", and hyphens beside U+2212 (polish plan G6).
    assert format_quantity(-3.25, "kW") == "\u22123.2 kW"
    assert format_quantity(82, "%", decimals=0) == "82%"
    # A value that rounds to zero never shows a sign.
    assert format_quantity(-0.04, "kWh") == "0.0 kWh"


def test_money_is_whole_pounds_at_fleet_scale_with_true_minus() -> None:
    assert money(-1250.51) == "\u2212£1,251"
    assert money(1391.2) == "£1,391"
    assert money(1391.2, signed=True) == "+£1,391"
    assert money(0.2, signed=True) == "£0"
    assert money(float("nan")) == UNAVAILABLE
    assert money(None) == UNAVAILABLE


def test_money_per_ev_keeps_pence() -> None:
    assert money(-1.5, decimals=2) == "\u2212£1.50"
    assert money(-0.004, decimals=2) == "£0.00"


def test_percent_and_kw_formats() -> None:
    assert percent(82.4) == "82%"
    assert percent(-3.14, decimals=1) == "\u22123.1%"
    assert percent(5, signed=True) == "+5%"
    assert percent(float("inf")) == UNAVAILABLE
    assert kw(1234.4) == "1,234 kW"
    assert kw(-12.6, signed=True) == "\u221213 kW"
    assert kw(None) == UNAVAILABLE


def test_path_labels_map_internal_ids_to_user_facing_names() -> None:
    # The model's own path_id stays "normal"/"selected" (contract v2); every
    # view maps it through this one dict so a rename never needs a second,
    # drifting copy.
    assert PATH_LABELS == {"normal": "Unmanaged", "selected": "Smart", "timed": "Timed tariff"}
    # Decision 0004 item 50: "Unmanaged", never "Normal" or "Baseline".
    assert UNMANAGED_GLOSSARY == (
        "Unmanaged = charge at full power from plug-in to target; every moved and saving "
        "figure is measured against it."
    )
    # Headroom is measured against each path's own draw, not the unmanaged one.
    assert "headroom" not in UNMANAGED_GLOSSARY


def test_path_display_order_puts_timed_tariff_between_unmanaged_and_smart() -> None:
    # Lead decision, 8 October 2026: on screen the legend reads Unmanaged,
    # Timed tariff, Smart -- not the frames' own normal/selected/timed order.
    assert PATH_DISPLAY_ORDER == ("normal", "timed", "selected")
    assert [PATH_LABELS[path] for path in PATH_DISPLAY_ORDER] == [
        "Unmanaged",
        "Timed tariff",
        "Smart",
    ]


def test_present_paths_filters_display_order_to_what_a_frame_has() -> None:
    # Model step 2: the shared helper every view's path loop becomes.
    assert present_paths(["selected", "normal"]) == ["normal", "selected"]
    assert present_paths(["normal", "timed", "selected"]) == ["normal", "timed", "selected"]
    assert present_paths(pd.Series(["normal", "selected", "normal"])) == ["normal", "selected"]
    assert present_paths(["normal"]) == ["normal"]
    assert present_paths([]) == []


def test_archetype_colours_are_six_distinct_and_avoid_the_smart_teal() -> None:
    assert len(set(ARCHETYPE_COLOURS)) == 6
    assert SERIES_COLOURS["selected"] not in ARCHETYPE_COLOURS
    for colour in ARCHETYPE_COLOURS:
        red, green, blue = (int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5))
        hue_degrees = 360 * colorsys.rgb_to_hls(red, green, blue)[0]
        # The smart teal #2DD4BF sits at about 173 degrees; keep clear of it.
        assert not 150 <= hue_degrees <= 195, colour
        assert _contrast_ratio(colour, "#0E1117") >= 4.5, colour


def test_sequential_and_risk_scales_avoid_teal_and_end_risk_in_flag_red() -> None:
    assert SERIES_COLOURS["selected"] not in [colour for _, colour in SEQUENTIAL]
    assert [stop for stop, _ in SEQUENTIAL] == sorted(stop for stop, _ in SEQUENTIAL)
    assert RISK[-1][1] == SERIES_COLOURS["flag"]
    assert RISK[0][1] == SERIES_COLOURS["normal"]


def test_assumption_value_looks_up_by_name_and_returns_none_when_absent() -> None:
    result = SimpleNamespace(
        assumptions=(SimpleNamespace(name="a", value=1), SimpleNamespace(name="b", value=2))
    )

    assert assumption_value(result, "b") == 2
    assert assumption_value(result, "missing") is None
    # A result run without assumption records (None, not an empty tuple)
    # must not raise.
    assert assumption_value(SimpleNamespace(assumptions=None), "a") is None


def test_light_template_is_a_white_canvas_with_dark_readable_ink() -> None:
    # Chart audit B8: notebooks are read on GitHub or a light-mode canvas, not
    # the dark Streamlit shell, so the light variant needs an explicit
    # (non-transparent) light background and dark ink, unlike ``AXLE_TEMPLATE``.
    layout = AXLE_LIGHT_TEMPLATE.layout
    assert layout.paper_bgcolor == LIGHT_BACKGROUND
    assert layout.plot_bgcolor == LIGHT_BACKGROUND
    assert layout.font.color == LIGHT_INK
    assert _contrast_ratio(LIGHT_INK, LIGHT_BACKGROUND) >= 4.5
    assert _contrast_ratio(LIGHT_MUTED_INK, LIGHT_BACKGROUND) >= 4.5
    assert _contrast_ratio(LIGHT_GRID, LIGHT_BACKGROUND) < 4.5  # a subtle gridline, not text


def test_light_template_keeps_the_same_series_colours_and_legend_layout() -> None:
    # B8: "same fonts, colours for normal/selected/bands" -- only the canvas
    # (background/ink/gridlines) differs between the app and notebook templates.
    assert AXLE_LIGHT_TEMPLATE.layout.colorway == AXLE_TEMPLATE.layout.colorway
    assert AXLE_LIGHT_TEMPLATE.layout.font.size == AXLE_TEMPLATE.layout.font.size
    assert AXLE_LIGHT_TEMPLATE.layout.legend == AXLE_TEMPLATE.layout.legend
    assert AXLE_LIGHT_TEMPLATE.layout.legend.yanchor == "bottom"  # below the plot


def test_apply_notebook_style_sets_light_template_height_and_keeps_view_choices() -> None:
    figure = go.Figure(layout={"yaxis": {"title": {"text": "SoC (%)"}}, "margin": {"r": 20}})

    layout = apply_notebook_style(figure, height=CHART_HEIGHTS["time_series"]).layout

    assert layout.height == 340
    assert layout.template.layout.paper_bgcolor == LIGHT_BACKGROUND
    assert layout.paper_bgcolor == LIGHT_BACKGROUND
    assert layout.yaxis.title.text == "SoC (%)"
    assert layout.margin.r == 20


def test_fresh_browser_gets_dark_theme_independent_of_system_scheme() -> None:
    config_path = Path(__file__).parents[2] / ".streamlit" / "config.toml"
    theme = tomllib.loads(config_path.read_text(encoding="utf-8"))["theme"]

    assert theme["base"] == "dark"
    assert theme["backgroundColor"] == "#0E1117"
    assert "light" not in theme
    assert "dark" not in theme


def test_theme_tokens_follow_the_polish_type_scale_and_muted_borders() -> None:
    # Polish plan G1: 14 px body, compact headings, 4 px radius, muted borders
    # (the old #64748B border was the brightest thing on screen).
    config_path = Path(__file__).parents[2] / ".streamlit" / "config.toml"
    theme = tomllib.loads(config_path.read_text(encoding="utf-8"))["theme"]

    assert theme["baseFontSize"] == 14
    assert theme["headingFontSizes"] == ["24px", "20px", "17px", "15px", "13px", "12px"]
    assert set(theme["headingFontWeights"]) == {600}
    assert theme["baseRadius"] == "4px"
    assert theme["borderColor"] == BORDER
    assert theme["showWidgetBorder"] is False


def test_app_css_is_one_style_block_with_main_padding_and_kpi_type() -> None:
    assert APP_CSS.count("<style>") == 1
    assert '[data-testid="stMainBlockContainer"]' in APP_CSS
    assert "font-size: 26px" in APP_CSS and "tabular-nums" in APP_CSS
