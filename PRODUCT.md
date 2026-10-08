# Product

## Register

product

## Users

Mike and an interview collaborator use the app together for a day or two to explore how assumptions about an EV fleet affect physical flexibility and illustrative market outcomes. They need to edit parameters, run a simulation deliberately, compare outcomes and trace displayed numbers back to calculations and evidence.

## Product Purpose

Axle Forecast Studio is an inspectable Python Monte Carlo model with a Streamlit dashboard. It makes the path from source observations and assumptions through fleet state and one smart-charging action understandable. Two explicit models are offered: **smart charging** (the default), where each plugged-in EV plans its own cheapest charging schedule to be ready by its expected departure minus a safety margin (decision 0004 item 38), and **no action**, kept selectable as the counterfactual (charge at full power from plug-in to target). A successful session lets someone edit an assumption in the Edit-assumptions dialog, click Run for a full forecast, and inspect the resulting charts, tables and underlying definitions without confusing illustrative results with operational or commercial facts. There is no automatic draft preview: nothing runs before an explicit Run click.

## Brand Personality

Calm, precise, credible. Copy is direct and explanatory, not promotional. The interface should feel like a serious modelling workbench that remains approachable to a first-time interview participant.

## Anti-references

- A generic light SaaS landing page, decorative metric-card grid or chart-heavy screen with no clear decision path.
- Neon accents, gradients, glassmorphism, ornamental motion or a literal imitation of Linear's proprietary branding.
- False precision: unlabelled synthetic values, implied real settlement cash, hidden assumptions or charts without equivalent data tables.

## Design Principles

1. Put the next modelling action and its state in view: draft and active full result are visually and verbally distinct, and a run bar chip always names the current state.
2. Show a useful overview first, then let users move to the smart-charging action, individual EV and archetype detail without changing the underlying result.
3. Keep evidence next to the number it qualifies; make unit, scope and source status easy to find.
4. Keep the smart-charging and no-action results comparable while preserving their different physical and illustrative-commercial meanings; mechanisms with no explicit-path model are named as "not modelled", not shown as stubs.
5. Use familiar Streamlit controls and Plotly interactions, with no custom interface layer unless it solves a current problem.

## Accessibility & Inclusion

Target WCAG AA contrast for text and controls, keyboard-accessible native widgets and table equivalents for charts. Respect reduced-motion preferences; no animation is required for the interview task. Colour must not be the only way to distinguish states or series.
