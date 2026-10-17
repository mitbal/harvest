import argparse
import datetime as dt
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

import harvest.data as hd
import harvest.prices as prices


logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 10
DEFAULT_MAX_CONCURRENCY = 10
BACKFILL_START_DATE = "2010-01-01"


def get_local_path(exchange: str) -> Path:
    """Return the repository-local historical-price cache path."""
    return prices.local_path(exchange)


def fetch_stock_price(symbol: str, start_date: str) -> Optional[pd.DataFrame]:
    """Fetch and normalize daily OHLCV prices for one symbol."""
    frame = hd.get_daily_stock_price(symbol, start_from=start_date)
    if frame is None or frame.empty:
        return None

    frame = frame.copy()
    frame["symbol"] = symbol
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    columns = [column for column in prices.OHLCV_COLUMNS if column in frame.columns]
    return frame[["symbol", "date", *columns]]


def _fetch_with_retries(symbol: str, start_date: str) -> Optional[pd.DataFrame]:
    for attempt in range(DEFAULT_RETRIES + 1):
        try:
            return fetch_stock_price(symbol, start_date)
        except Exception as exc:
            if attempt == DEFAULT_RETRIES:
                raise
            logger.warning(
                "Price download failed for %s: %s; retrying in %ss (%s/%s)",
                symbol,
                exc,
                DEFAULT_RETRY_DELAY,
                attempt + 1,
                DEFAULT_RETRIES,
            )
            time.sleep(DEFAULT_RETRY_DELAY)
    return None


def _load_local_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = prices.read_prices(path.parent.name, columns=None, path=path)
        logger.info("Loaded %s historical-price records from %s", len(frame), path)
        return frame
    except Exception as exc:
        logger.warning("Ignoring unreadable historical-price cache %s: %s", path, exc)
        return pd.DataFrame()


def write_to_r2(exchange: str, new_frame: pd.DataFrame, mode: str) -> int:
    """Merge new rows into the exchange's R2 price file; returns the stored row count."""
    if mode == "backfill":
        existing = pd.DataFrame()
    else:
        # A failed read must stop the run: writing new rows alone would wipe the history.
        existing = prices.read_prices(exchange, columns=None)
        if existing.empty:
            raise RuntimeError(f"R2 price history for {exchange} is empty; run a backfill first")
    merged = prices.merge_prices(existing, new_frame)
    prices.write_prices(exchange, merged)
    return len(merged)


def run_historical_pipeline(
    exchange: str = "jkse",
    mode: str = "incremental",
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    use_local_cache: bool = False,
    cache_path: Optional[Path] = None,
    symbols: Optional[list[str]] = None,
    start_date: Optional[str] = None,
) -> dict:
    """Fetch historical prices and merge them into the R2 price history.

    Incremental mode refetches the last 7 days (or from ``start_date``) and
    merges them into the stored history; backfill rebuilds it from 2010.
    """
    if exchange not in {"jkse", "sp500"}:
        raise ValueError("exchange must be either 'jkse' or 'sp500'")
    if mode not in {"incremental", "backfill"}:
        raise ValueError("mode must be either 'incremental' or 'backfill'")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")

    logger.info("Starting historical-price pipeline for %s in %s mode", exchange, mode)
    if symbols is None:
        stocks = hd.get_all_idx_stocks() if exchange == "jkse" else hd.get_all_sp500_stocks()
        if stocks.empty or "symbol" not in stocks.columns:
            raise RuntimeError(f"No stock symbols found for {exchange}")
        selected_symbols = stocks["symbol"].dropna().astype(str).drop_duplicates().tolist()
    else:
        selected_symbols = list(dict.fromkeys(str(symbol) for symbol in symbols if symbol))
    if not selected_symbols:
        raise RuntimeError(f"No stock symbols found for {exchange}")

    if start_date is None:
        start_date = (
            BACKFILL_START_DATE
            if mode == "backfill"
            else (dt.datetime.now() - dt.timedelta(days=7)).strftime("%Y-%m-%d")
        )

    all_data = []
    failed_symbols = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(_fetch_with_retries, symbol, start_date): symbol
            for symbol in selected_symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                if result is not None and not result.empty:
                    all_data.append(result)
                else:
                    failed_symbols.append(symbol)
            except Exception as exc:
                failed_symbols.append(symbol)
                logger.error("Price download exhausted retries for %s: %s", symbol, exc)

    if not all_data:
        raise RuntimeError(f"No historical prices were fetched for {exchange}")

    new_frame = pd.concat(all_data, ignore_index=True).drop_duplicates(
        subset=["symbol", "date"], keep="last"
    )
    stored_rows = write_to_r2(exchange, new_frame, mode)

    if use_local_cache:
        local_path = cache_path or get_local_path(exchange)
        existing_frame = _load_local_cache(local_path) if mode == "incremental" else pd.DataFrame()
        if symbols is not None and not existing_frame.empty:
            existing_frame = existing_frame[
                existing_frame["symbol"].astype(str).isin(selected_symbols)
            ]
        final_frame = prices.merge_prices(existing_frame, new_frame)
        prices.write_prices(exchange, final_frame, path=local_path)
        logger.info("Wrote %s historical-price records to %s", len(final_frame), local_path)

    summary = {
        "exchange": exchange,
        "mode": mode,
        "start_date": start_date,
        "symbols_total": len(selected_symbols),
        "symbols_succeeded": len(selected_symbols) - len(failed_symbols),
        "symbols_failed": len(failed_symbols),
        "failed_symbols": sorted(failed_symbols),
        "rows_upserted": len(new_frame),
        "rows_stored": stored_rows,
        "local_cache_enabled": use_local_cache,
    }
    logger.info("Historical-price summary: %s", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch daily prices into the price store")
    parser.add_argument("--exchange", choices=["jkse", "sp500"], required=True)
    parser.add_argument("--mode", choices=["incremental", "backfill"], default="incremental")
    parser.add_argument("--start-date", help="Refetch from this date (YYYY-MM-DD) in incremental mode")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_historical_pipeline(
            exchange=args.exchange,
            mode=args.mode,
            max_concurrency=args.max_concurrency,
            start_date=args.start_date,
        )
        return 0
    except Exception:
        logger.exception("Historical-price pipeline failed")
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sys.exit(main())
