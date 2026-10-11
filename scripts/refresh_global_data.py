"""Refresh the global FMP snapshot in ``data/global`` with today's data.

Rebuilds the same set of files as the 2025-06-14 snapshot, prefixed with
today's date so the older snapshot stays:

* ``<date>_all_stocks_universe.csv`` — FMP's full stock list
* ``<date>_<exchange>_company_profiles.csv`` — profiles of every ``stock``
  listed on each exchange in ``PROFILE_EXCHANGES``
* ``<date>_<name>_dividends.pkl`` — ``{symbol: dividend history}`` for every
  stock on the exchanges in ``DIVIDEND_SETS`` (all profile exchanges) that has
  paid a dividend

A file that already exists is skipped (unless ``--force``), so an interrupted
run picks up where it stopped.

Usage:
    python -m scripts.refresh_global_data
    python -m scripts.refresh_global_data --only profiles --exchanges "Tokyo Stock Exchange"
"""
import argparse
import datetime
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import os
import pandas as pd
import requests

BASE_URL = "https://financialmodelingprep.com/api/v3"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "global"
PROFILE_BATCH_SIZE = 100
# Above this share of failed requests (say, a network outage) a file isn't written, so a rerun retries it.
MAX_FAILED_SHARE = 0.02

PROFILE_EXCHANGES = (
    "Australian Securities Exchange",
    "B3 S.A.",
    "Bombay Stock Exchange",
    "Canadian Securities Exchange",
    "Deutsche Börse",
    "Euronext Paris",
    "Hong Kong Stock Exchange",
    "Jakarta Stock Exchange",
    "KOSDAQ",
    "Korea Exchange",
    "London Stock Exchange",
    "NASDAQ Capital Market",
    "NASDAQ Copenhagen",
    "NASDAQ Global Market",
    "NASDAQ Global Select",
    "NASDAQ",
    "National Stock Exchange of India",
    "New York Stock Exchange",
    "Shanghai Stock Exchange",
    "Shenzhen Stock Exchange",
    "Stock Exchange of Thailand",
    "Stockholm Stock Exchange",
    "Swiss Exchange",
    "Taipei Exchange",
    "Taiwan Stock Exchange",
    "Tel Aviv Stock Exchange",
    "Tokyo Stock Exchange",
    "Toronto Stock Exchange Ventures",
    "Toronto Stock Exchange",
    "Warsaw Stock Exchange",
    # Added after the 2025 snapshot: the rest of Southeast Asia FMP lists.
    "Malaysian Stock Exchange",
    "Stock Exchange of Singapore",
    "Ho Chi Minh Stock Exchange",
)

# FMP names that changed since the 2025 snapshot; files keep the 2025 name so snapshots line up.
EXCHANGE_ALIASES = {
    "Jakarta Stock Exchange": ("Indonesia Stock Exchange",),
}

DIVIDEND_SETS = {
    "mse": ("Malaysian Stock Exchange",),
    "nasdaq": ("NASDAQ Global Select",),
    "nyse": ("New York Stock Exchange",),
    "sse": ("Stock Exchange of Singapore",),
    "stockholm": ("Stockholm Stock Exchange",),
}
# Every other profile exchange too, its file named after the exchange like the profiles.
_COVERED = {exchange for exchanges in DIVIDEND_SETS.values() for exchange in exchanges}
# macOS file names ignore case, so "NASDAQ" would collide with "nasdaq" (Global Select).
_TAKEN = {name.lower() for name in DIVIDEND_SETS}
DIVIDEND_SETS.update({
    (f"{exchange} other" if exchange.lower() in _TAKEN else exchange): (exchange,)
    for exchange in PROFILE_EXCHANGES if exchange not in _COVERED
})


class RateLimiter:
    """Spaces calls evenly so the workers stay under the plan's calls per minute."""

    def __init__(self, calls_per_minute: int):
        self.interval = 60.0 / calls_per_minute
        self.lock = threading.Lock()
        self.next_call = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next_call - now
            self.next_call = max(now, self.next_call) + self.interval
        if delay > 0:
            time.sleep(delay)


class Fmp:
    def __init__(self, api_key: str, calls_per_minute: int, retries: int = 4):
        self.api_key = api_key
        self.limiter = RateLimiter(calls_per_minute)
        self.retries = retries
        self.session = requests.Session()

    def get(self, path: str):
        url = f"{BASE_URL}/{path}"
        separator = "&" if "?" in path else "?"
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            try:
                response = self.session.get(f"{url}{separator}apikey={self.api_key}", timeout=(10, 120))
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {response.status_code}")
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and "Error Message" in data:
                    raise RuntimeError(data["Error Message"])
                return data
            except Exception as exc:
                if attempt == self.retries:
                    # Never echo the URL: it carries the API key.
                    raise RuntimeError(f"{path.split('?')[0]}: {exc}") from None
                time.sleep(2 ** attempt)


def stocks_on(universe: pd.DataFrame, exchanges) -> list[str]:
    names = {alias for exchange in exchanges for alias in (exchange, *EXCHANGE_ALIASES.get(exchange, ()))}
    selected = universe[(universe["type"] == "stock") & universe["exchange"].isin(names)]
    return selected["symbol"].dropna().astype(str).drop_duplicates().tolist()


def refresh_universe(fmp: Fmp, path: Path, force: bool) -> pd.DataFrame:
    if path.exists() and not force:
        print(f"Universe: using existing {path.name}")
        return pd.read_csv(path)
    print("Universe: downloading the stock list…", flush=True)
    universe = pd.DataFrame(fmp.get("stock/list"))
    universe.to_csv(path, index=False)
    print(f"Universe: {len(universe)} symbols → {path.name}")
    return universe


def refresh_profiles(fmp: Fmp, universe: pd.DataFrame, exchanges, date: str, workers: int, force: bool) -> None:
    for exchange in exchanges:
        path = OUT_DIR / f"{date}_{exchange}_company_profiles.csv"
        if path.exists() and not force:
            print(f"Profiles: skipping {exchange} (exists)")
            continue
        symbols = stocks_on(universe, [exchange])
        if not symbols:
            print(f"Profiles: WARNING no stocks listed on {exchange!r} — renamed in FMP?")
            continue
        batches = [symbols[i:i + PROFILE_BATCH_SIZE] for i in range(0, len(symbols), PROFILE_BATCH_SIZE)]

        def fetch(batch):
            try:
                return fmp.get(f"profile/{','.join(batch)}"), None
            except Exception as exc:
                return [], f"{batch[0]}…: {exc}"

        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(fetch, batches))
        rows = [row for data, _ in results for row in data]
        errors = [error for _, error in results if error]
        if len(errors) > MAX_FAILED_SHARE * len(batches):
            print(f"Profiles: {exchange}: {len(errors)}/{len(batches)} batches failed — not written, rerun to retry")
            continue
        profiles = pd.DataFrame(rows)
        profiles.to_csv(path, index=False)
        note = f", {len(errors)} failed batches" if errors else ""
        print(f"Profiles: {exchange}: {len(profiles)}/{len(symbols)}{note}", flush=True)
        for error in errors:
            print(f"    {error}")


def refresh_dividends(fmp: Fmp, universe: pd.DataFrame, sets: dict, date: str, workers: int, force: bool) -> None:
    for name, exchanges in sets.items():
        path = OUT_DIR / f"{date}_{name}_dividends.pkl"
        if path.exists() and not force:
            print(f"Dividends: skipping {name} (exists)")
            continue
        symbols = stocks_on(universe, exchanges)
        if not symbols:
            print(f"Dividends: WARNING no stocks listed on {exchanges} — renamed in FMP?")
            continue
        print(f"Dividends: {name}: {len(symbols)} symbols…", flush=True)

        def fetch(symbol):
            try:
                data = fmp.get(f"historical-price-full/stock_dividend/{symbol}")
                return symbol, pd.DataFrame(data.get("historical", []) if isinstance(data, dict) else []), None
            except Exception as exc:
                return symbol, None, str(exc)

        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(fetch, symbols))
        dividends = {symbol: frame for symbol, frame, _ in results if frame is not None and not frame.empty}
        errors = [f"{symbol}: {error}" for symbol, _, error in results if error]
        if len(errors) > MAX_FAILED_SHARE * len(symbols):
            print(f"Dividends: {name}: {len(errors)}/{len(symbols)} failed — not written, rerun to retry")
            for error in errors[:3]:
                print(f"    {error}")
            continue
        with open(path, "wb") as handle:
            pickle.dump(dividends, handle)
        note = f", {len(errors)} failed" if errors else ""
        print(f"Dividends: {name}: {len(dividends)} payers of {len(symbols)}{note} → {path.name}", flush=True)
        for error in errors[:10]:
            print(f"    {error}")


def compare_with_previous(date: str, previous: str) -> None:
    rows = []
    for exchange in PROFILE_EXCHANGES:
        new, old = (OUT_DIR / f"{d}_{exchange}_company_profiles.csv" for d in (date, previous))
        if new.exists():
            rows.append({
                "exchange": exchange,
                previous: len(pd.read_csv(old)) if old.exists() else None,
                date: len(pd.read_csv(new)),
            })
    if rows:
        print(f"\nProfiles per exchange, {previous} vs {date}:")
        print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["universe", "profiles", "dividends"], action="append",
                        help="run only these parts (repeatable; default: all)")
    parser.add_argument("--exchanges", nargs="*", help="limit profiles to these exchanges")
    parser.add_argument("--date", default=datetime.date.today().isoformat(), help="file-name prefix")
    parser.add_argument("--previous", default="2025-06-14", help="snapshot to compare counts against")
    parser.add_argument("--calls-per-minute", type=int, default=int(os.environ.get("FMP_CALLS_PER_MINUTE", "240")))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="overwrite files that already exist")
    args = parser.parse_args()

    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        sys.exit("FMP_API_KEY is not set")
    parts = set(args.only or ["universe", "profiles", "dividends"])
    fmp = Fmp(api_key, args.calls_per_minute)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    universe_path = OUT_DIR / f"{args.date}_all_stocks_universe.csv"
    universe = refresh_universe(fmp, universe_path, args.force and "universe" in parts)
    if "profiles" in parts:
        refresh_profiles(fmp, universe, args.exchanges or PROFILE_EXCHANGES, args.date, args.workers, args.force)
    if "dividends" in parts:
        refresh_dividends(fmp, universe, DIVIDEND_SETS, args.date, args.workers, args.force)
    compare_with_previous(args.date, args.previous)


if __name__ == "__main__":
    main()
