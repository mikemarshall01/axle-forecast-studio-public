# Usability test script: five tasks, five people

Purpose: a 20-minute think-aloud session with someone from the interview demo audience (an Axle engineer, a trader, a supplier analyst or a curious generalist). Steve Krug's rule of thumb: five people find most of what matters; run it with three if that is who you can get. No automation replaces this: it tests whether the screens answer the questions people actually ask.

## Before you start (2 minutes)

Open the app on the tester's own laptop or phone if you can; otherwise share yours. Say: "This tests the app, not you. Please think aloud, say what you expect before you click, and tell me when something is confusing. I will not help until you are stuck for a minute." Start with the empty state, no run yet. Have a timer and this sheet.

## The five tasks

For each task record: completed alone (yes / with a hint / no), time, where they hesitated, what they said, and the one thing you would change.

1. **Run the model and say what you are looking at.** "Get a forecast on screen. Then tell me, in one sentence, what the first page shows." Success: they find Run simulation without prompting, wait through the run, and describe the average day chart as a simulated week of drivers rather than real data. Watch for: confusion while the run is in progress (is the status clear?), reading "Model output · illustrative" or missing it.

2. **Find the fleet's evening peak and its spread.** "What is the fleet's home charging power at 6 pm on a weekday, and how sure is the model?" Success: they reach Drivers ▸ Fleet week (or Overview ▸ Key stats), read the P50 with its P10–P90 range, and say the band is across simulated weeks. Watch for: reading the band edge as a hard limit, mixing up kW and kWh, hunting for the time on the axis.

3. **Change one assumption and find its effect.** "Assume every home charger is 3.6 kW instead of 7 kW. Show me what changes." Success: they open Edit assumptions, find the charger power field, run again, and use Compare (or the KPI tiles) to state the difference in one figure. Watch for: not noticing the stale chip after editing, expecting the change to apply without a Run, not knowing Compare exists.

4. **Explain what smart charging did and what it cost.** "In one minute, tell a trader what Axle's smart charging changed this week and what the illustrative cost was." Success: they walk Smart charging ▸ Plan, Response, Value and risk in order, name the unmanaged path as the comparison, and say "illustrative" or "not a settlement figure" without being told. Watch for: reading the money figure as real revenue, missing the weeks-not-recovered count, asking what "Unmanaged" means.

5. **Do it on a phone.** "Repeat task 2 on your phone (or at 390 px)." Success: navigation, the lens control and the chart are usable without pinch-zoom; the legend is readable below the plot. Watch for: hidden navigation, controls wrapping oddly, KPI rows too tall, charts clipped.

## After (5 minutes)

Ask three questions and write the answers verbatim: "What would you use this for?", "What did you not trust?", "What was the most annoying moment?" Then rate each task 0–2 (0 failed, 1 hint needed, 2 alone) and add a one-line severity note per problem (blocks a task / slows a task / cosmetic). Put the sheet in `docs/reviews/usability-<date>-<initials>.md`; three or more sheets with the same blocking problem is a fix, not a debate.
