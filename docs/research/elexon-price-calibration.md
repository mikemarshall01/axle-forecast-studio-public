# Elexon price calibration (decision 0004 item 53, plan section 2B item B2)

Calibration numbers for the realistic day-ahead price generator (plan item
B1: `level_d = level_{d-1} + noise` plus a 3-harmonic weekday/weekend x
winter/summer shape plus half-hour AR(1) noise), fitted from real GB
balancing data. **These numbers are for Lane 2's B1 writer to read and paste
into `model/assumptions.py`; this script and this document do not change the
model.** Nothing here runs inside the Streamlit app -- the app has no live
Elexon dependency (plan "not building" list item 3).

## Method

1. `scripts/fetch_elexon_prices.py` downloads two free, unauthenticated
   Elexon Insights Solution datasets over stdlib `urllib`:
   - **Market index** (`balancing/pricing/market-index`, provider
     `APXMIDP`): half-hourly day-ahead auction price (GBP/MWh) and traded
     volume (MWh). Chunked into 7-day requests (the endpoint's own maximum
     `from`/`to` span) and concatenated.
   - **System (imbalance) prices** (`balancing/settlement/system-prices/{date}`):
     one calendar day per request (the endpoint takes a single date, not a
     range) -- system sell/buy price (equal, post GB's single-imbalance-price
     reform), Net Imbalance Volume (NIV) and the price-derivation code.
2. `scripts/fit_price_shape.py` reads the CSVs and computes, entirely from
   the API's own `settlement_date`/`settlement_period` fields (already
   defined on the London settlement calendar -- no separate timezone
   conversion is needed or applied):
   - the mean price by half-hour-of-day (0-47) for each of four groups
     (weekday/weekend x winter Oct-Mar/summer Apr-Sep), fitted with a
     3-harmonic Fourier series by ordinary least squares
     (`np.linalg.lstsq`, matching the model's own `level + shape + noise`
     structure);
   - the daily-mean price series (one global AR(1) random-walk level, not
     split by weekday/season -- the weekday/season pattern is the shape,
     above): its SD, lag-1 AR(1) coefficient, negative-price half-hour
     share, and P10/P50/P90 spread;
   - the half-hour residual left after removing a day's own mean and its
     group's zero-mean shape deviation: its SD and lag-1 AR(1) coefficient
     (the noise term plan item B1 calls the "existing half-hour AR(1)
     noise");
   - system price vs market index price, split by NIV sign (plan item B5's
     two-state NIV premium).

## Date range and data fetched

- Requested range: **2025-09-28 to 2026-09-28 inclusive** (the last 12
  months up to the day before the script was run, since the current day's
  settlement is not yet published).
- Market index: 17,566 half-hourly rows over 366 calendar days (APXMIDP
  only). One day (2025-09-28, the first requested day) is short two periods:
  the fetch script's request windows are UTC calendar-day boundaries, and
  during British Summer Time London midnight is 23:00 UTC the day before, so
  the very first day's first hour falls just outside the window. This is at
  most 2 of 17,568 half-hours (0.01%) and is not corrected -- see
  Limitations.
- System prices: 17,568 half-hourly rows over 366 calendar days (this
  endpoint is queried by settlement date directly, so it has no UTC/London
  boundary issue).
- Two genuine GB clock-change days are in range: 2025-10-26 (50 periods,
  clocks back) and 2026-03-29 (46 periods, clocks forward). Handled by the
  fitting script's half-hour bucket, `(settlement_period - 1) % 48`, which
  folds the two extra autumn periods onto buckets 0/1 -- an immaterial
  simplification given 366 days of data (see Limitations).
- Files: `data/elexon/market_index_2025-09-28_2026-09-28.csv`,
  `data/elexon/system_prices_2025-09-28_2026-09-28.csv` (both gitignored,
  raw data is not committed) plus `data/elexon/ATTRIBUTION.txt`.

## Results

### Intraday shape: 3-harmonic Fourier fit, GBP/MWh

`shape(t) = a0 + a1*cos(2*pi*t/48) + b1*sin(2*pi*t/48) + a2*cos(4*pi*t/48) + b2*sin(4*pi*t/48) + a3*cos(6*pi*t/48) + b3*sin(6*pi*t/48)`,
`t` = half-hour-of-day bucket (0 = 00:00-00:30 London, ..., 47 = 23:30-00:00).

| group | a0 | a1 | b1 | a2 | b2 | a3 | b3 | R² | trough | trough £/MWh | peak | peak £/MWh | swing £/MWh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weekday_winter | 84.810 | -9.348 | -10.094 | -9.510 | -6.600 | 4.218 | 3.666 | 0.968 | 03:00 | 64.07 | 17:30 | 108.52 | 44.44 |
| weekday_summer | 114.739 | 13.734 | -4.288 | -15.106 | -14.769 | -1.052 | 1.544 | 0.960 | 13:30 | 81.54 | 20:00 | 146.72 | 65.17 |
| weekend_winter | 77.324 | 0.067 | -11.647 | -7.830 | -0.897 | 2.338 | 5.121 | 0.972 | 12:30 | 66.86 | 18:00 | 101.92 | 35.06 |
| weekend_summer | 89.659 | 31.541 | -14.796 | -13.542 | -12.218 | -4.669 | 2.822 | 0.990 | 13:00 | 46.49 | 20:00 | 140.27 | 93.77 |

Reading: winter days trough overnight (03:00 weekday, midday-ish 12:30
weekend, both still well above the summer trough) and peak in the early
evening (17:30-18:00), the classic GB winter demand pattern. Summer days
trough around midday (solar depresses the day-ahead price) and peak later,
19:30-20:00. All four fits explain 96-99% of the variance in their group's
mean half-hourly profile.

### Daily-mean level (whole range, all 366 days together)

| stat | value |
|---|---|
| SD of daily means | £29.95/MWh |
| AR(1) coefficient (lag-1, on daily means) | 0.644 |
| Negative-price half-hour share | 2.12% |
| Daily-mean P10 | £57.53/MWh |
| Daily-mean P50 | £93.11/MWh |
| Daily-mean P90 | £134.73/MWh |
| Daily-mean P90 - P10 | £77.19/MWh |

### Half-hour residual noise (after removing daily mean and group shape)

| stat | value |
|---|---|
| AR(1) coefficient (lag-1) | 0.920 |
| SD | £21.61/MWh |

### System (imbalance) price vs market index price

17,566 settlement periods joined (market index and system prices share
almost all periods; a handful drop at the edges of the two fetches).

| stat | value |
|---|---|
| Share of periods with NIV > 0 | 47.2% |
| Share of periods with NIV < 0 | 52.8% |
| Mean SIP - MIP premium when NIV > 0 | +£20.18/MWh |
| Mean SIP - MIP premium when NIV < 0 | -£16.80/MWh |

Sign convention: NIV > 0 is treated as "short" (National Grid ESO net
accepted more balancing offers than bids) and NIV < 0 as "long", the
convention used in GB power-market commentary (e.g. Modo Energy, LCP Delta).
Elexon's own published NIV glossary entry (checked via WebFetch while
writing this script) does not spell out the sign explicitly, so treat the
long/short *labels* as an assumption -- the NIV-sign split and the
SIP-minus-MIP premium numbers themselves are read directly from the data
regardless of which label is attached to which sign. The premium's sign
(positive when NIV > 0, negative when NIV < 0) matches the direction plan
item B5 expected (short pays a premium, long a discount), which is
corroborating but not confirming evidence for the label.

## Limitations

- **First-day shortfall**: the requested range's first calendar day
  (2025-09-28) is missing its first two half-hour periods, a UTC-vs-London
  BST boundary artefact of the fetch script's chunk windows (see Date
  range). 2 of 17,568 half-hours; not corrected.
- **Clock-change days folded, not modelled**: `(settlement_period - 1) % 48`
  folds the autumn clock-change day's two extra periods onto buckets 0 and
  1, and the spring clock-change day simply has two fewer observations.
  Affects 2 of 366 days; not a separate model concern.
- **NIV sign convention is an assumption**, not confirmed from Elexon's own
  published definition (see System price summary above).
- **One year of data, one regime**: the fit reflects UK wholesale prices
  from late September 2025 to late September 2026 only. It is a plausible
  calibration target for a synthetic price generator, not a long-run GB
  price distribution -- a different 12 months (different gas prices, wind
  years, interconnector flows) would fit somewhat different numbers.
- **APXMIDP only**: the plan specifies this provider; N2EXMIDP (the other
  live day-ahead index) is not used and would give a related but distinct
  fit.
- **Harmonic fit is to the *mean* profile, not every half-hour**: the
  3-harmonic curve is least-squares fitted to each group's 48-point mean
  profile (as the plan specifies), not to the raw half-hourly series
  directly; the residual noise stats above are computed against that group
  mean shape, which is the intended decomposition (`level + shape + noise`).
- **This is a calibration exercise, not settlement data reproduced in the
  app.** No Axle-cash or settlement figure is derived from this data; it
  only informs illustrative synthetic-price generator coefficients.

## Attribution

Contains BMRS data © Elexon Limited copyright and database right 2026.
Source: Elexon Insights Solution API, `data.elexon.co.uk/bmrs`
(`balancing/pricing/market-index`, `dataProviders=APXMIDP`;
`balancing/settlement/system-prices/{date}`). No API key required. Fetched
2026-09-28 for the range above; see `data/elexon/ATTRIBUTION.txt` (written
by `scripts/fetch_elexon_prices.py`, gitignored alongside the raw CSVs).

## Reproducing

```bash
PYTHONPATH=src uv run --frozen --no-sync python scripts/fetch_elexon_prices.py --out-dir data/elexon
PYTHONPATH=src uv run --frozen --no-sync python scripts/fit_price_shape.py \
    --market-index-csv data/elexon/market_index_2025-09-28_2026-09-28.csv \
    --system-prices-csv data/elexon/system_prices_2025-09-28_2026-09-28.csv \
    --chart /tmp/elexon_price_shape_check.html
```

The check chart (`/tmp/elexon_price_shape_check.html`, not committed) plots
each group's observed mean profile against its fitted 3-harmonic curve.
