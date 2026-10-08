# Design tokens: where they live and how to use them

Status: reference, 29 September 2026. The tokens themselves are code; this page only says where each one lives and the rules for using it. It does not restate values that would go stale. Root `DESIGN.md` is the design-system narrative (north star, layout, voice); where its colour frontmatter disagrees with the files below, the files win.

## Sources of truth

| Token group | Lives in | Read it with |
| --- | --- | --- |
| App theme: canvas, surface, sidebar, text, link, primary (teal) accent, borders, base font size, heading ladder, radius | `.streamlit/config.toml` `[theme]` | Streamlit applies it; charts read nothing from it (Plotly cannot), so chart chrome repeats the canvas idea with transparent backgrounds |
| Chart series colours (`SERIES_COLOURS`), archetype colours, `SEQUENTIAL` and `RISK` scales, ink and muted ink, border/grid | `src/axle_studio/ui/style.py` | Import the constant; never spell the hex in a view |
| Band opacity (`BAND_ALPHA`), plugged-in opacity, line width and dash per path (`PATH_STYLES`), user-facing path names (`PATH_LABELS`) | `style.py` | `band_and_line(...)` and `PATH_STYLES[path]` |
| Chart heights (`CHART_HEIGHTS`), one chart text size (`CHART_FONT_PX`), margins, legend placement, London time axis | `style.py` (`AXLE_TEMPLATE`, `style_figure`, `london_time_axis`) | `chart_block(...)` applies them; a view sets only its height key |
| KPI tile type (label 12 px, value 26 px, unit 14 px, context 12 px) and page padding | `style.py` `APP_CSS` | Rendered by `components/kpi.py`; no other inline CSS |
| Number formatting: true minus, `£`, `%`, `kW`, `Unavailable` for missing | `style.py` `format_quantity`, `money`, `percent`, `kw` | Always format through these |
| Light variant for notebooks | `style.py` `AXLE_LIGHT_TEMPLATE`, `LIGHT_*` | `apply_notebook_style` |

There is no `tokens.json`. The lints import `style.py` directly, so a second file would only be a copy that could drift.

## Colour roles

- Teal (`SERIES_COLOURS["selected"]`, theme `primaryColor`) means one thing everywhere: the smart (selected) Axle action path and the active control. No other series, scale or badge uses teal.
- `normal` (light grey-blue) is the unmanaged counterfactual; `difference` (amber) is smart minus unmanaged; `observed` (near-white, dashed) is a CNZ context value; `flag` (red) marks energy not recovered and always travels with `FLAG_LABEL` text.
- `timed` (decision 0007) is the optional third "Timed tariff" path on the Smart charging ▸ Response chart, present when the "Timed tariff policy" switch is on (the default, start 00:00), absent when off: charging held off until an editable clock hour, named for the common real-world pattern of a timed tariff. No tariff price is modelled and every path is costed the same way. It reuses `difference`'s value rather than a new hex and is drawn dashed, never teal (reserved for `selected`); `style.PATH_DISPLAY_ORDER` puts it second on screen (Unmanaged, Timed tariff, Smart), ahead of frame order's last-of-three `"timed"`.
- Archetype colours are six equal-lightness hues that skip teal. `SEQUENTIAL` is one blue-violet hue rising in lightness; `RISK` runs grey to amber to the flag red.
- Bands are the line's colour at `BAND_ALPHA`; the P10–P90 band and its P50 line share one legend entry.
- Evidence badges: Source grey, Illustrative orange, Synthetic violet, Model output none.

## Type scale and spacing

The numbers below describe style.py and .streamlit/config.toml as of 29 September 2026; if they disagree, the code wins.

- Body 14 px, headings 24/20/17/15/13/12 px at weight 600, one system sans stack (`config.toml`). Three sizes on a page: page title, chart title in bold body size, body or caption.
- Chart text is one size (`CHART_FONT_PX`) for ticks, axis titles, legend and hover; subplot titles use the same value.
- KPI tiles use tabular figures and never wrap the value. Captions are at most 140 characters, KPI labels at most 28 with no unit (`tests/ui/test_copy_rules.py`).
- Chart margins come from content: left 56 px, right 16 px (56 on a dual-axis chart), top 8 px, bottom sized to tick row, axis title and legend rows. Legends sit below the plot at every width.

## Rules

1. A view never writes a colour, `rgba(...)`, `px` length or chart font size inline. `scripts/ui_lint.py` and `tests/ui/test_ui_lint.py` fail on new ones; the current exceptions are listed there with reasons and are candidates for the next polish pass.
2. A figure never reaches `st.plotly_chart` except through `chart_block`, which applies the template, height and margins. `scripts/dashboard_lint.py` and `tests/ui/test_dashboard_lint.py` check the result: no pie, no unaccepted secondary axis, titled y axes with units, token colours only, a band on every time-series line, named traces, a legend for two or more series, hover on every chart.
3. A new colour or size is added to `style.py` with a docstring saying what it means, then used; not the other way round.
4. Streamlit theme values change in `config.toml` only; the `tests/ui/test_style.py` theme tests read that file.
