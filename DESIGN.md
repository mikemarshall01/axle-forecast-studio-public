---
name: Axle Forecast Studio
description: A dark modelling workbench for an inspectable EV fleet forecast.
colors:
  primary: "#0F766E"
  dark-canvas: "#0E1117"
  dark-surface: "#171B23"
  dark-sidebar: "#131720"
  dark-ink: "#E9EDF3"
  dark-link: "#6EE7D0"
  dark-border: "#64748B"
  series-normal: "#9AA9BF"
  series-selected: "#2DD4BF"
  series-difference: "#F2B45A"
  series-observed: "#E9EDF3"
  series-flag: "#F87171"
  series-grid: "#232A36"
typography:
  body:
    fontFamily: "sans-serif"
---

# Design System: Axle Forecast Studio

## Overview

**Creative North Star: "The model workbench"**

A participant should be able to sit beside Mike, understand what result is on screen, change an assumption and find the effect without a tour. The design takes its density, quiet hierarchy and navigation discipline from Linear, not its assets or branding. The interview screen uses one dark theme even when the browser's system preference is light.

Use the eight pages in top navigation (Overview, Drivers, Smart charging, Trading, Supplier, Replay, Compare, How it works; `docs/DASHBOARD_DESIGN.md` sections 0 and 2, decision 0004 item 28), a run bar at the top of every page, and one header row per page: title left, lens control right, the lens's question underneath. Assumptions are edited in an **Edit assumptions** dialog opened from the run bar, not on a page. The sidebar is not used, so nothing essential hides behind it at phone width. Put the action or decision near the data it affects. The app should open to useful content, not a decorative hero. A chart and its table are two views of the same stored result.

**Key Characteristics:** restrained colour; compact but legible controls; clear active state; broad data area; source labels beside numbers; no ornamental animation.

## Colors

The native Streamlit theme in `.streamlit/config.toml` owns the app palette. `src/axle_studio/ui/style.py` holds the Plotly series tokens, the shared `axle` Plotly template, chart heights, London axis labels and displayed-quantity conventions. `src/axle_studio/ui/components/chart_table.py` holds `chart_block`, the one chart pattern. This document describes how to use them; do not fork a second theme in arbitrary page CSS.

### Primary

- **Deep teal** (`primary`): primary Run action, selection and focus. It is deliberately dark enough for Streamlit's white primary-button text to meet AA contrast.
- **Mint link** (`dark-link`): text links on dark surfaces. Do not use the darker primary as body-sized link text on the dark canvas.

### Neutral

- **Near-black canvas** (`dark-canvas`): the main working area.
- **Raised charcoal** (`dark-surface`) and **sidebar charcoal** (`dark-sidebar`): controls and navigation, separated tonally rather than with shadows.
- **Clear ink** (`dark-ink`): body text and key labels. Muted text must still meet AA where it carries information.
- **Clear divider** (`dark-border`): control and navigation boundaries that remain visible against both dark surfaces. Use sparingly, not as decorative card outlines.

### Chart series

- **Unmanaged** (`series-normal`, solid 1.5 px): charging at full power from plug-in to target, no Axle action. A lighter solid line, not dotted (decision 0004 item 27, which overrides the design table's dotted line).
- **Smart** (`series-selected`, solid 2.5 px): the smart-charging Axle action path (decision 0004 item 38). It is teal because it means the same as the teal UI accent: what Axle chose.
- **Difference** (`series-difference`, solid 2 px): smart minus unmanaged, paired per simulated week.
- **Observed** (`series-observed`, dashed rule with a text label): a CNZ observed context value.
- **Flag** (`series-flag` with the text "⚠ not recovered"): a week or EV whose energy is not recovered.
- **Grid** (`series-grid`, 1 px): gridlines only; no plot border.
- The centre line is the median (P50) with a P10–P90 band, named in the legend. Bands use 0.3 fill opacity (item 27); plugged-in bars and spans use `series-normal` at 0.35.

**The One Accent Rule.** Use teal for actions, current selection and the selected action path, not for every metric or chart series. The Run button keeps the darker `primary` teal so its white label meets AA contrast. Plotly series need named, distinguishable labels and a table equivalent; colour alone never carries meaning.

## Typography

**Display Font:** the Streamlit sans-serif system stack.
**Body Font:** the same system stack.
**Label/Mono Font:** use Streamlit's native code style only for identifiers, timestamps and values that benefit from fixed-width alignment.

**Character:** technical without becoming a terminal. Prefer weight, spacing and direct wording over oversized headings.

### Hierarchy

- **Workspace title:** one concise page title, visibly above the selected lens and result state.
- **Section title:** short and specific to the question answered by the content beneath it.
- **Body:** plain UK English, with explanatory prose limited to a readable line length.
- **Label:** units, scope, evidence class and state remain legible at ordinary text size; do not hide essential meaning in a tiny caption.

**The Numbers Need Names Rule.** Every published metric needs a definition, unit, scope and evidence label close enough to understand it without visiting Methods & evidence first.

## Elevation

Use flat tonal layers and fine dividers. There is no decorative shadow vocabulary. Let the run bar, a control surface and a data view be distinguishable by background and spacing. Avoid nested cards or wide glows.

**The Flat Workbench Rule.** A result panel earns its boundary from its function, not from a shadow or oversized radius.

## Components

### Buttons

- **Primary:** reserve the native Streamlit primary treatment for **Run simulation** (run bar, empty state and dialog footer; all the same action). It must remain an explicit click, never an automatic side effect of editing or navigation.
- **Secondary:** **Edit assumptions**, **Reset to active run**, **Reset to defaults** and view actions use ordinary native controls. Destructive result clearing, if added later, needs a distinct label and confirmation.
- **States:** the run bar shows one worded chip (`st.badge`) plus a muted identity line: no chip with "No result yet", Running (controls disabled, fleet size and week count in words, the page stays readable), Current, Stale · N changes, Run failed (previous result kept, plus the only banner, with the reason) and Draft invalid (Run disabled). Colour is a second cue only. Do not render a live Cancel action unless cancellation actually works.

### Inputs / Fields

- Group assumptions by modelling purpose; show the current value, unit, valid range and source/illustrative badge.
- A valid change updates the draft only and marks the active result stale; there is no preview run. An invalid change displays an explanation, disables Run and never replaces the active full result.
- A widget always shows the draft value the next Run will use.
- Use native widgets and keyboard focus states. Do not use a form that requires an Apply click.

### Navigation

- Keep the eight pages in stable order in the top navigation, with text labels and the ink wordmark (`assets/wordmark.svg`) at the left; Overview is the landing page. The active page and lens must be obvious without relying on colour alone.
- Lenses use a segmented control, right-aligned in the header row, that wraps at narrow widths and remembers its choice when you leave the page and come back.
- One page title (subheader size), then the lens's question as one muted line. No second heading repeats the lens name.
- Show the run state and active result identity in the run bar on every page; one divider under the bar and nowhere else.

### Charts, tables and detail

- Each Plotly chart has a corresponding table based on the same stored data, with units in axis/table labels and a short definition beside the view. The series colours in `ui/style.py` clear non-text contrast on the configured dark surfaces.
- Every chart is a `chart_block`: bold title with unit, the chart, one caption naming the spread and evidence class, and a collapsed **Data and definition** expander with the table.
- Time axes are labelled in Europe/London local time (`london_time_axis`) while x values stay in UTC, so clock-change weeks plot in order; hover shows London time then UTC (`hover_time_labels`).
- Every chart has an explicit height (`CHART_HEIGHTS`; the values live only in `src/axle_studio/ui/style.py`, not duplicated here), fills its container width, keeps its legend below the plot at every width and hides the modebar.
- Every caption names its spread: across EVs, or across simulated weeks.
- Normal and selected paths use consistent line/marker distinctions and share evaluation-world inputs. Do not add their physical values as if both happened.
- Use one clear drill path from fleet to cohort to world to individual where the stored result supports it. Unavailable detail says why it is unavailable; it is not silently shown as zero.
- Financial components are separate, payer/payee/sign-labelled where known, and visibly illustrative or unavailable. No aggregate Axle cash total is implied.

## Do's and Don'ts

### Do:

- **Do** make draft and active full-run states unmistakable, including stale and failed states.
- **Do** put source observations, editable assumptions, synthetic history, toy posterior and real backtests in different labelled categories.
- **Do** use AA-contrast text and keyboard-accessible native controls; respect reduced-motion preferences by avoiding unnecessary motion.
- **Do** keep the interface useful at desktop and narrow widths and inspect both in a browser before calling it ready.

### Don't:

- **Don't** build a generic light SaaS landing page, decorative metric-card grid or chart-heavy screen with no clear decision path.
- **Don't** use neon accents, gradients, glassmorphism, ornamental motion or a literal imitation of Linear's proprietary branding.
- **Don't** display false precision: unlabelled synthetic values, implied real settlement cash, hidden assumptions or charts without equivalent data tables.
- **Don't** start a full Monte Carlo run from an edit, import, navigation, Reset or ordinary rerun.
