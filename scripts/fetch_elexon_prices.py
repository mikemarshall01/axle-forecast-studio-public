"""Fetch free, public Elexon Insights balancing-price data for calibrating the
synthetic day-ahead price generator (decision 0004 item 53; plan
`docs/plans/2026-09-29-analyst-trading-polish-plan.md` section 2B item B2).

This script owns nothing about the model: it only downloads and reshapes two
public GB balancing datasets, unauthenticated (no API key), and writes them
to CSV for `scripts/fit_price_shape.py` to read offline. Nothing here runs in
the Streamlit app -- the app never depends on live Elexon data.

Datasets fetched (Elexon Insights Solution API, `data.elexon.co.uk/bmrs`):

- Market Index (`balancing/pricing/market-index`): half-hourly day-ahead
  auction prices and traded volume. Filtered to the APXMIDP provider (the
  long-running APX/Nord Pool day-ahead auction index), the provider named in
  the accepted plan, so day-ahead calibration is not mixed with the newer
  N2EXMIDP index. The endpoint accepts only a 7-day (inclusive) `from`/`to`
  span per request, so a wide range is fetched in 7-day chunks and the
  results concatenated.
- System (imbalance) prices (`balancing/settlement/system-prices/{date}`):
  one calendar day per request -- this endpoint takes a single settlement
  date, not a range, so a wide range costs one request per day. Used to
  compare the settlement (system) price against the market index price for
  the intraday/imbalance channel calibration (plan item B5).

Licence: Contains BMRS data (c) Elexon Limited copyright and database right
2026. The full attribution text is also written to
`data/elexon/ATTRIBUTION.txt` alongside the CSVs. `data/elexon/` is
gitignored -- raw fetched data is never committed.

Usage (from the repo root):

    PYTHONPATH=src uv run --frozen --no-sync python scripts/fetch_elexon_prices.py
    PYTHONPATH=src uv run --frozen --no-sync python scripts/fetch_elexon_prices.py \\
        --from-date 2025-09-01 --to-date 2026-08-31

With no dates given, fetches the last 12 months up to yesterday (settlement
data for "today" is not yet published when the script runs).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

MARKET_INDEX_URL = "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
SYSTEM_PRICES_URL = (
    "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{date}"
)
DATA_PROVIDER = "APXMIDP"

# The market-index endpoint rejects any request where `to` minus `from`
# exceeds 7 days (checked live: an 8-day span returns HTTP 400 "The date
# range between From and To inclusive must not exceed 7 days"). Chunk
# requests to exactly this span and de-duplicate on the overlap boundary
# rather than trying to compute non-overlapping half-hour edges by hand.
MARKET_INDEX_CHUNK_DAYS = 7

REQUEST_TIMEOUT_S = 30
POLITE_DELAY_S = 0.3
"""Pause between requests. The API is free and unauthenticated with no
published rate limit; this keeps a ~365-request system-price fetch polite
rather than hammering a public endpoint."""

ATTRIBUTION = "Contains BMRS data © Elexon Limited copyright and database right 2026"
SOURCE_NOTE = (
    "Source: Elexon Insights Solution API, data.elexon.co.uk/bmrs "
    "(balancing/pricing/market-index, dataProviders=APXMIDP; "
    "balancing/settlement/system-prices/{date}). No API key required."
)


def _get_json(url: str, *, max_attempts: int = 3) -> dict:
    """GET one JSON document, retrying transient network failures.

    A public unauthenticated API occasionally times out or resets under
    load; retrying a couple of times with a short pause is simpler and more
    robust than failing a multi-hour fetch on one blip.
    """
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "axle-forecast-studio-price-calibration/1.0",
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < max_attempts:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"failed to fetch {url} after {max_attempts} attempts") from last_error


def _date_chunks(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    """Split [start, end] (inclusive, calendar dates) into chunks of at most chunk_days days."""
    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


_MARKET_INDEX_COLUMNS = [
    "settlement_date",
    "settlement_period",
    "price_gbp_per_mwh",
    "volume_mwh",
    "start_time_utc",
]


def _tidy_market_index_chunk(data: list[dict]) -> pd.DataFrame:
    """Rename one chunk's raw rows to tidy columns, for concatenation with a consistent dtype."""
    raw = pd.DataFrame(
        data, columns=["settlementDate", "settlementPeriod", "price", "volume", "startTime"]
    )
    return raw.rename(
        columns={
            "settlementDate": "settlement_date",
            "settlementPeriod": "settlement_period",
            "price": "price_gbp_per_mwh",
            "volume": "volume_mwh",
            "startTime": "start_time_utc",
        }
    )[_MARKET_INDEX_COLUMNS]


def fetch_market_index(start: date, end: date) -> pd.DataFrame:
    """Fetch half-hourly APXMIDP day-ahead auction prices for [start, end] inclusive (UTC days).

    Returns one row per (settlement_date, settlement_period) with columns
    `settlement_date` (London settlement date, as published by Elexon),
    `settlement_period` (1-48, occasionally 1-46/1-50 on clock-change days),
    `price_gbp_per_mwh`, `volume_mwh` and `start_time_utc`.

    Request windows are UTC calendar-day boundaries, not London local
    midnight, so during British Summer Time (UTC+1) the first requested day
    can be missing its first one or two periods and a day just past `end`
    can leak in. The leak is trimmed below; the first-day shortfall is not
    (it is at most two of a year's 17,000+ half-hours) -- see the
    limitations note in docs/research/elexon-price-calibration.md.
    """
    frames = []
    for chunk_start, chunk_end in _date_chunks(start, end, MARKET_INDEX_CHUNK_DAYS):
        from_param = f"{chunk_start.isoformat()}T00:00:00Z"
        # +1 day so the chunk's own last day is fully covered; the API's `to`
        # is an inclusive instant, so this deliberately overlaps the next
        # chunk's `from` by one period -- de-duplicated below.
        to_param = f"{(chunk_end + timedelta(days=1)).isoformat()}T00:00:00Z"
        url = f"{MARKET_INDEX_URL}?from={from_param}&to={to_param}&dataProviders={DATA_PROVIDER}"
        payload = _get_json(url)
        frames.append(_tidy_market_index_chunk(payload.get("data", [])))
        time.sleep(POLITE_DELAY_S)
    if not frames:
        return pd.DataFrame(columns=_MARKET_INDEX_COLUMNS)
    tidy = pd.concat(frames, ignore_index=True)
    tidy = tidy.drop_duplicates(subset=["settlement_date", "settlement_period"])
    # ISO date strings compare lexicographically, so this is a plain range
    # filter: drops the day-after-`end` periods the UTC/London BST offset
    # can pull in past the chunk boundary above.
    start_iso, end_iso = start.isoformat(), end.isoformat()
    in_range = (tidy["settlement_date"] >= start_iso) & (tidy["settlement_date"] <= end_iso)
    tidy = tidy.loc[in_range]
    return tidy.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)


_SYSTEM_PRICES_COLUMNS = [
    "settlement_date",
    "settlement_period",
    "start_time_utc",
    "system_sell_price_gbp_per_mwh",
    "system_buy_price_gbp_per_mwh",
    "net_imbalance_volume_mwh",
    "price_derivation_code",
]


def _tidy_system_prices_chunk(data: list[dict]) -> pd.DataFrame:
    """Rename one day's raw rows to tidy columns, for concatenation with a consistent dtype."""
    raw = pd.DataFrame(
        data,
        columns=[
            "settlementDate",
            "settlementPeriod",
            "startTime",
            "systemSellPrice",
            "systemBuyPrice",
            "netImbalanceVolume",
            "priceDerivationCode",
        ],
    )
    return raw.rename(
        columns={
            "settlementDate": "settlement_date",
            "settlementPeriod": "settlement_period",
            "startTime": "start_time_utc",
            "systemSellPrice": "system_sell_price_gbp_per_mwh",
            "systemBuyPrice": "system_buy_price_gbp_per_mwh",
            "netImbalanceVolume": "net_imbalance_volume_mwh",
            "priceDerivationCode": "price_derivation_code",
        }
    )[_SYSTEM_PRICES_COLUMNS]


def fetch_system_prices(start: date, end: date) -> pd.DataFrame:
    """Fetch one day of GB imbalance settlement prices per day for [start, end] inclusive.

    One HTTP request per calendar day -- the endpoint takes a single
    settlement date, not a range. Returns columns `settlement_date`,
    `settlement_period`, `start_time_utc`, `system_sell_price_gbp_per_mwh`,
    `system_buy_price_gbp_per_mwh` (equal under GB's single imbalance-price
    reform, kept as two columns to match the published fields),
    `net_imbalance_volume_mwh` and `price_derivation_code`.
    """
    frames = []
    day = start
    while day <= end:
        payload = _get_json(SYSTEM_PRICES_URL.format(date=day.isoformat()))
        frames.append(_tidy_system_prices_chunk(payload.get("data", [])))
        time.sleep(POLITE_DELAY_S)
        day += timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=_SYSTEM_PRICES_COLUMNS)
    tidy = pd.concat(frames, ignore_index=True)
    return tidy.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)


def _write_attribution(out_dir: Path, start: date, end: date) -> None:
    """Write the BMRS licence attribution as a sidecar file next to the fetched CSVs."""
    text = (
        f"{ATTRIBUTION}\n"
        f"{SOURCE_NOTE}\n"
        f"Fetched: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"Requested range: {start.isoformat()} to {end.isoformat()} (inclusive)\n"
    )
    (out_dir / "ATTRIBUTION.txt").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_to = date.today() - timedelta(days=1)
    default_from = default_to - timedelta(days=365)
    parser.add_argument("--from-date", type=date.fromisoformat, default=default_from)
    parser.add_argument("--to-date", type=date.fromisoformat, default=default_to)
    parser.add_argument("--out-dir", type=Path, default=Path("data/elexon"))
    parser.add_argument("--skip-market-index", action="store_true")
    parser.add_argument("--skip-system-prices", action="store_true")
    args = parser.parse_args()

    if args.from_date > args.to_date:
        parser.error("--from-date must not be after --to-date")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    span = f"{args.from_date.isoformat()}_{args.to_date.isoformat()}"

    if not args.skip_market_index:
        print(f"Fetching market index (APXMIDP) {args.from_date} to {args.to_date} ...")
        market_index = fetch_market_index(args.from_date, args.to_date)
        path = args.out_dir / f"market_index_{span}.csv"
        market_index.to_csv(path, index=False)
        print(f"  wrote {len(market_index)} rows to {path}")

    if not args.skip_system_prices:
        print(f"Fetching system (imbalance) prices {args.from_date} to {args.to_date} ...")
        system_prices = fetch_system_prices(args.from_date, args.to_date)
        path = args.out_dir / f"system_prices_{span}.csv"
        system_prices.to_csv(path, index=False)
        print(f"  wrote {len(system_prices)} rows to {path}")

    _write_attribution(args.out_dir, args.from_date, args.to_date)
    print(f"Attribution written to {args.out_dir / 'ATTRIBUTION.txt'}")


if __name__ == "__main__":
    main()
