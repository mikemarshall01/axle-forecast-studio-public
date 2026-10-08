### Build notes

#### What assumptions did I make?

The six archetypes and their shares, annual mileage and battery/charger specs come straight from the sheet Axle gave me. Everything else about how a driver behaves day to day is an assumption, and I've tried to label it that way everywhere it shows up.

The biggest one: a source plug-in or plug-out time is evidence the car was connected, not proof of how long the driver was away. So I don't treat those clocks as journey length. Instead I simulate a trip each day (or not), debit the energy it uses, and let plug-in state and battery level fall out of that. Arrival state of charge is an output of the simulation, never an input.

I also had no real market data, so any price or cost I show is illustrative: a synthetic price path I generate myself, and a public-charging rate of £0.79/kWh that's editable, not a quoted tariff. I don't add these into a single "cash" total, because I haven't checked who actually pays whom, in which direction. The study runs for a fixed seven days in UTC half-hours, anchored to London noon, noon to noon, so a clock change partway through doesn't quietly shorten or stretch the week.

#### What I spent time on, and what I left out

I spent most of my time on getting the physics right: one battery, one charger, one route of action per car per half-hour, energy that actually balances, and a state of charge that never goes below zero or above full. That felt like the part most likely to be wrong in a way that wouldn't show up unless I checked carefully, so I built tests that check conservation and plausible bounds rather than trusting the output because it looked reasonable on a chart.

One thing I got wrong the first time and had to fix: if a car ends up with less charge and a missed trip compared with normal charging, and I don't put a value on that missed trip, the model can make an action look like a saving when it actually just left the driver short. I now price that missed travel the same way as any other energy the driver would have to buy back, so the numbers don't lie by omission. Later I dropped stranding altogether: a car that would run low now tops up at a public charger first, so no trip is missed. The pricing stays in as a check.

What I left out, deliberately: frequency response, capacity market payments, vehicle-to-grid and real geography or eligibility rules. I started with a blunter idea, a single import cap at one time of day, but that only asks whether Axle can defer one half-hour; it doesn't say how much charging can be moved, when, or with what confidence, which is the question I actually need to answer. So each EV now plans its own charging session at plug-in, filling the cheapest forecast half-hours before it expects to leave, and the trading, supplier and household views all build on that one plan. I also didn't calibrate against real settlement data, because I don't have any, and left out real geography or eligibility rules, which need information I don't have.

I used an AI-assisted, heavily reviewed workflow to write a lot of the code faster than I could alone, with independent review on every piece before it went in. I checked the physics and the decisions myself; I didn't hand-write every line.

#### How I designed it for its end use

I built this to answer questions, not just show numbers. Someone from Axle should be able to ask "what if" and see the answer in a minute or two. So any assumption can be changed and run again, and Compare puts the two runs side by side on the same random weeks, so the difference is only the change you made. Each page answers one question with one main chart, and the working behind every number is one click away. The full-size run is saved in advance, so the app opens straight away. The model and the screens are kept apart: Python works out every number and the dashboard only draws it, so what you see is exactly what the simulation produced. The outputs are also shaped for the people who would use them: drivers, suppliers, charger makers, carmakers and fleets.

I came up with the model myself: its structure, the inputs, parameters and variables, and the actors in it. The agents built it to my design, and I made every decision along the way. I worked directly with the agents on every decision: I set what each piece had to do, chose between options they laid out, and had the reasoning written down. Some of the main choices:

- Six driver types with their own battery size, mileage and plug-in habits, straight from Axle's sheet.
- A trip and a plug-in for every car, every day. Battery level is an output, never an input.
- The battery balances half-hour by half-hour, and never goes below 0% or above 100%.
- A car that would run flat tops up at a public charger first, so no trip is ever missed.
- Colder or hotter days make driving a bit less efficient, the way real cars behave.
- Each simulated week gets its own made-up wholesale price, loosely shaped on real GB price patterns.
- The smart charger fills the cheapest hours before a car is due to leave, not a fixed time of day.
- A trading desk, a supplier and a household view all read the same plan, so their numbers agree.
- Firm MW allows for chargers responding late or not at all, and is checked against weeks held out of its own forecast.

I kept the model and the display strictly separate: Python does every calculation, and the dashboard only reads and draws what Python worked out. That's partly good practice and partly self-interest: it means I can trust that what's on screen is what the simulation actually produced, and so can anyone reviewing it with me.

The explainer below has the rest of what went into it.
