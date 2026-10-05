"""One-off: build the Parquet price files from the local pickle cache and upload them to R2.

    python -m pipeline.migrate_prices_to_parquet --exchange jkse                     # local file only
    python -m pipeline.migrate_prices_to_parquet --exchange jkse --fill-gap          # plus refetch since the cache ended
    python -m pipeline.migrate_prices_to_parquet --exchange jkse --fill-gap --upload # then upload to R2

``--fill-gap`` refetches every symbol from FMP starting a week before the last
cached date, since the pickle cache may be months behind production. Uploading
needs the R2 bucket and the ``R2_*`` environment variables.
"""
import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import harvest.prices as prices
from pipeline import historical_prices_core as core


logger = logging.getLogger(__name__)

GAP_OVERLAP_DAYS = 7


def pickle_path(exchange: str) -> Path:
    return prices.REPOSITORY_ROOT / "data" / exchange / "historical_prices.pkl"


def load_pickle_cache(exchange: str) -> pd.DataFrame:
    frame = pd.read_pickle(pickle_path(exchange))
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError(f"No prices in {pickle_path(exchange)}")
    return prices.normalize_prices(frame[["symbol", "date", "close"]])


def fetch_gap(frame: pd.DataFrame, max_concurrency: int) -> pd.DataFrame:
    """Refetch every cached symbol from a week before the cache's last date."""
    start = (frame["date"].max() - pd.Timedelta(days=GAP_OVERLAP_DAYS)).strftime("%Y-%m-%d")
    symbols = frame["symbol"].unique().tolist()
    logger.info("Refetching %s symbols from %s", len(symbols), start)

    fetched, empty, failed = [], [], []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(core._fetch_with_retries, symbol, start): symbol for symbol in symbols}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                failed.append(futures[future])
                continue
            if result is None or result.empty:
                empty.append(futures[future])
            else:
                fetched.append(result)
    # Delisted symbols return no data and keep their cached history.
    logger.info(
        "Gap fill: %s symbols fetched, %s without new data, %s failed",
        len(fetched), len(empty), len(failed),
    )
    if failed:
        # Usually rate limiting; rerun with a lower --max-concurrency to fill them.
        logger.warning("Gap fill downloads failed for: %s", ", ".join(sorted(failed)))
    return pd.concat(fetched, ignore_index=True) if fetched else pd.DataFrame()


def summarize(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "symbols": int(frame["symbol"].nunique()),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
    }


def migrate(exchange: str, fill_gap: bool, upload: bool, max_concurrency: int) -> dict:
    frame = load_pickle_cache(exchange)
    logger.info("Pickle cache: %s", summarize(frame))
    if fill_gap:
        frame = prices.merge_prices(frame, fetch_gap(frame, max_concurrency))

    local = prices.write_prices(exchange, frame, path=prices.local_path(exchange))
    summary = {"exchange": exchange, **summarize(frame), "local_file": local}
    if upload:
        summary["r2_object"] = prices.write_prices(exchange, frame)
        # Read back through the same path the app uses, to catch credential or format problems now.
        check = prices.read_prices(exchange, symbols=[frame["symbol"].iloc[0]])
        if check.empty:
            raise RuntimeError("Uploaded file could not be read back from R2")
    logger.info("Migration summary: %s", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exchange", choices=["jkse", "sp500"], required=True)
    parser.add_argument("--fill-gap", action="store_true", help="Refetch prices since the cache ended")
    parser.add_argument("--upload", action="store_true", help="Upload the result to R2")
    parser.add_argument("--max-concurrency", type=int, default=core.DEFAULT_MAX_CONCURRENCY)
    args = parser.parse_args(argv)
    try:
        migrate(args.exchange, args.fill_gap, args.upload, args.max_concurrency)
        return 0
    except Exception:
        logger.exception("Price migration failed")
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sys.exit(main())
