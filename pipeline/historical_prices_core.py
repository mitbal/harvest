import datetime as dt
import logging
import os
import pickle
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
from supabase import Client, create_client

import harvest.data as hd


logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 10
DEFAULT_MAX_CONCURRENCY = 10
BACKFILL_START_DATE = "2010-01-01"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def get_supabase_client() -> Client:
    """Create a Supabase client from the pipeline environment."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")
    return create_client(url, key)


def get_local_path(exchange: str) -> Path:
    """Return the repository-local historical-price cache path."""
    return REPOSITORY_ROOT / "data" / exchange / "historical_prices.pkl"


def fetch_stock_price(symbol: str, start_date: str) -> Optional[pd.DataFrame]:
    """Fetch and normalize daily prices for one symbol."""
    frame = hd.get_daily_stock_price(symbol, start_from=start_date)
    if frame is None or frame.empty:
        return None

    frame = frame.copy()
    frame["symbol"] = symbol
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    return frame[["symbol", "date", "close"]]


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
        with path.open("rb") as cache_file:
            frame = pickle.load(cache_file)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Historical-price cache does not contain a DataFrame")
        logger.info("Loaded %s historical-price records from %s", len(frame), path)
        return frame
    except Exception as exc:
        logger.warning("Ignoring unreadable historical-price cache %s: %s", path, exc)
        return pd.DataFrame()


def _write_local_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
        pickle.dump(frame, temporary_file)
    temporary_path.replace(path)


def upsert_to_db(frame: pd.DataFrame, client: Optional[Client] = None) -> None:
    """Upsert one historical-price chunk into Supabase."""
    if frame.empty:
        return
    supabase = client or get_supabase_client()
    supabase.table("historical_prices").upsert(frame.to_dict(orient="records")).execute()


def run_historical_pipeline(
    exchange: str = "jkse",
    mode: str = "incremental",
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    use_local_cache: bool = False,
    cache_path: Optional[Path] = None,
) -> dict:
    """Fetch historical prices and persist new rows to Supabase."""
    if exchange not in {"jkse", "sp500"}:
        raise ValueError("exchange must be either 'jkse' or 'sp500'")
    if mode not in {"incremental", "backfill"}:
        raise ValueError("mode must be either 'incremental' or 'backfill'")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")

    logger.info("Starting historical-price pipeline for %s in %s mode", exchange, mode)
    stocks = hd.get_all_idx_stocks() if exchange == "jkse" else hd.get_all_sp500_stocks()
    if stocks.empty or "symbol" not in stocks.columns:
        raise RuntimeError(f"No stock symbols found for {exchange}")

    symbols = stocks["symbol"].dropna().astype(str).tolist()
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
            for symbol in symbols
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
    client = get_supabase_client()
    chunk_size = 1000
    for offset in range(0, len(new_frame), chunk_size):
        upsert_to_db(new_frame.iloc[offset : offset + chunk_size], client=client)

    if use_local_cache:
        local_path = cache_path or get_local_path(exchange)
        existing_frame = _load_local_cache(local_path) if mode == "incremental" else pd.DataFrame()
        final_frame = (
            pd.concat([existing_frame, new_frame], ignore_index=True)
            if not existing_frame.empty
            else new_frame
        )
        final_frame = final_frame.drop_duplicates(
            subset=["symbol", "date"], keep="last"
        ).reset_index(drop=True)
        _write_local_cache(local_path, final_frame)
        logger.info("Wrote %s historical-price records to %s", len(final_frame), local_path)

    summary = {
        "exchange": exchange,
        "mode": mode,
        "start_date": start_date,
        "symbols_total": len(symbols),
        "symbols_succeeded": len(symbols) - len(failed_symbols),
        "symbols_failed": len(failed_symbols),
        "failed_symbols": sorted(failed_symbols),
        "rows_upserted": len(new_frame),
        "local_cache_enabled": use_local_cache,
    }
    logger.info("Historical-price summary: %s", summary)
    return summary
