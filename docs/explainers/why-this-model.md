### Why this model

#### What it is for

Axle asked two plain questions about six driver types: when do they plug in at home, and how full is the battery (state of charge, SoC) when they do. I built the model to answer those, then the questions on top: how much charging can move to cheaper half-hours, what it is worth, what it costs the driver, and how firmly we could promise it. It forecasts from data where I have it, with the uncertainty shown.

#### How a forecast is drawn

Each driver is a small set of habits: how often they drive, how far, when they leave, when they plug in, how often they skip a night. I draw one plausible week for each driver from those habits, add 1,000 of them up, and a fleet shape appears. Draw the week again, 100 times by default, and the spread across those weeks is the uncertainty.

Weather and prices are drawn the same way. Each day gets one temperature, shared by every driver; a cold day costs more energy per mile (so does a very hot one) and lifts heating demand, so the price rises too. The smart charger sees that week's day-ahead prices and fills the cheapest half-hours before each driver is due to leave. Each week runs twice on the same draws, unmanaged and smart, so the difference is the smart charging alone.

```
habits per driver -> one drawn week each -> add up -> fleet shape
weather and prices: one draw per week, shared by every driver
repeat 100 weeks -> the spread across weeks is the P10–P90 band
```

P50 is the middle week; eight weeks in ten fall between P10 and P90. Each chart says whether its band is across weeks (uncertainty) or across drivers (variety).

The model covers driving distance; the timing of behaviour (plug-in and departure clocks, weekday against weekend, skipped nights, staying plugged in over a day off); and the physics (a half-hour battery energy balance, charger power and efficiency, weather's effect on energy per mile, public top-ups when the battery runs low).

#### Every parameter named

Every input is a named parameter with a value, a unit and a source or stated assumption: 264 of them, 50 from a source, the rest illustrative or synthetic, 80 editable in Edit assumptions. A habit is not one number but a distribution. Keeping it simple, each has three parts: its mean (the typical value), its standard deviation (how much it varies) and its shape. You could go further and control for skew, kurtosis and other features of the distribution. Plug-in and departure clocks use a Student-t with heavier tails than a normal, so the odd very early or late plug-in or departure happens. Daily mileage is a lognormal around the mean, so no one drives negative miles. Skips are yes/no chances. I chose each one and recorded why, so any can be checked and changed.

Units are grouped, each group with its own parameters. The six archetypes are exactly that (battery, mileage, plug-in chance, plug-in and departure clocks), and four illustrative zones group EVs for reporting and local events. Groups do not have to be fixed types. With a track record, units could be grouped by how they behave: those that show a steady chance, above a target, of being plugged in between set hours, with charging schedules that have been met and are predictable, can back a firm product. Those that often leave before their charging plan finishes can be flagged, priced for the risk or kept out of what we promise. It also shows which units would benefit from a different plan: a longer safety margin for the early leavers, a later ready time for those who stay plugged in, or a different tariff for those who gain least. The model already tracks each driver's plan completion and early departures, so the same idea works on simulated units today. One run feeds every page; change a value and Compare shows the effect on the same random futures.

What is calibrated: archetype shares, mileage and batteries from Axle's sheet, and the price shape, checked against Elexon half-hourly data. What is not: driver behaviour. The CNZ report is context, not a target. In the default run the median plug-in SoC comes out at about 58%, against the 52% CNZ observed, because near-daily plug-ins keep batteries high. That gap is reported, not tuned away. The test of the model is that it should generate what we see in real data. Each new data source is a chance to check it and adjust it.

#### Where detail matters: trading

The trading view settles against a baseline, the average of recent unmanaged nights, the way the P415 baseline does. Turn-down is sold the day before, re-traded hourly intraday, and what the fleet actually did settles at the imbalance price. Only turn-down is settled, never the rebound. Smart charging already moves charging out of 17:00–21:00, so a smart-charged evening leaves almost nothing to turn down; the firm product lives overnight.

#### Uncertainty is the margin

Firm means the worst week in ten: the P10 across weeks. That is what we can sell. The P50 is what we hope for. The gap between them is uncertainty, and better data narrows it. That is why data makes us more competitive.

#### Different customers, one model

A driver asks whether the car is ready, what a year saves, and whether the battery is cycled harder. A supplier asks about hedge error, firm capacity and churn. A charger maker asks what an enrolled device earns a month; a fleet, what a vehicle is worth a year. The Partners page holds one card per audience and says "Not modelled" where it should, and Supplier ▸ Customers shows which customers gain least, so we know who to target. Sign-up and retention are inputs you set, not outputs. A ready car and a cheaper bill keep a customer and win the next one. One model holds all our data and knowledge, so it guides our own strategy and the products we sell, and those products raise retention, cut churn and win new customers.

#### What it is not, yet

Every price is synthetic and every £ is illustrative, never Axle cash. Today it is EV-only with one national price: no home batteries, heat pumps, solar or vehicle-to-grid, and a zone is a reporting split, not a place. Each would be another group with its own habits and physics on the same drawn weeks, prices and trading, with the same customer outputs; so would vans, fleets or regions with their own grid conditions. Limits lists everything else it cannot answer.
