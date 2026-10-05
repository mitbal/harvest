"""One-off: rebuild an exchange's price file with open, high, low, close and volume.

    python -m pipeline.backfill_ohlcv --exchange jkse              # local file only
    python -m pipeline.backfill_ohlcv --exchange jkse --upload     # back up R2, then replace it

Every symbol in the current file plus everything listed today is refetched from
FMP since 2010. A symbol that comes back replaces its whole history, so old and
new closes are never mixed within one stock; a symbol that doesn't (usually
delisted) keeps its existing close-only rows. Uploading first copies the current
R2 object to ``prices/<exchange>.backup-<date>.parquet``.
"""
import argparse
import datetime
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import harvest.data as hd
import harvest.prices as prices
from pipeline import historical_prices_core as core


logger = logging.getLogger(__name__)


def fetch_all(symbols: list[str], max_concurrency: int) -> tuple[list[pd.DataFrame], list[str], list[str]]:
    """OHLCV since 2010 per symbol; returns frames, symbols without data, and failed symbols."""
    frames, empty, failed = [], [], []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(core._fetch_with_retries, symbol, core.BACKFILL_START_DATE): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                failed.append(futures[future])
                continue
            if result is None or result.empty:
                empty.append(futures[future])
            else:
                frames.append(result)
    return frames, empty, failed


def listed_symbols(exchange: str) -> list[str]:
    try:
        stocks = hd.get_all_idx_stocks() if exchange == "jkse" else hd.get_all_sp500_stocks()
        return stocks["symbol"].dropna().astype(str).tolist()
    except Exception as exc:
        logger.warning("Listed symbols unavailable, using the stored file only: %s", exc)
        return []


def backup_r2(exchange: str) -> str:
    """Server-side copy of the current R2 object; returns the backup key."""
    config = prices.r2_config()
    if config is None:
        raise RuntimeError("R2 is not configured; set the R2_* settings")
    filesystem = prices._filesystem(config)
    source = f"{config['bucket']}/{prices.r2_key(exchange)}"
    target = f"{config['bucket']}/prices/{exchange}.backup-{datetime.date.today().isoformat()}.parquet"
    filesystem.copy_file(source, target)
    return target


def backfill(exchange: str, upload: bool, max_concurrency: int) -> dict:
    existing = prices.read_prices(exchange, columns=None) if upload else prices.read_prices(
        exchange, columns=None, path=prices.local_path(exchange))
    if existing.empty:
        raise RuntimeError(f"No stored {exchange} history to backfill")
    symbols = sorted(set(existing["symbol"]) | set(listed_symbols(exchange)))
    logger.info("Fetching OHLCV for %s symbols", len(symbols))

    frames, empty, failed = fetch_all(symbols, max_concurrency)
    if failed:
        # Usually rate limiting; one slower pass picks most of them up.
        logger.info("Retrying %s failed symbols", len(failed))
        retry_frames, retry_empty, failed = fetch_all(failed, max(1, max_concurrency // 2))
        frames += retry_frames
        empty += retry_empty
    if not frames:
        raise RuntimeError("No OHLCV data was fetched")

    fresh = pd.concat(frames, ignore_index=True)
    refreshed = set(fresh["symbol"])
    kept = existing[~existing["symbol"].isin(refreshed)]
    merged = prices.normalize_prices(pd.concat([kept, fresh], ignore_index=True))
    if merged["symbol"].nunique() < existing["symbol"].nunique():
        raise RuntimeError("Backfilled file has fewer symbols than the stored one; not writing")

    summary = {
        "exchange": exchange,
        "rows_before": len(existing),
        "rows_after": len(merged),
        "symbols_before": int(existing["symbol"].nunique()),
        "symbols_after": int(merged["symbol"].nunique()),
        "symbols_refreshed": len(refreshed),
        "symbols_kept_close_only": int(kept["symbol"].nunique()),
        "symbols_without_data": len(empty),
        "symbols_failed": sorted(failed),
        "rows_with_ohlc": int(merged["open"].notna().sum()),
        "local_file": prices.write_prices(exchange, merged, path=prices.local_path(exchange)),
    }
    if upload:
        summary["backup"] = backup_r2(exchange)
        summary["r2_object"] = prices.write_prices(exchange, merged)
        check = prices.read_prices(exchange, symbols=sorted(refreshed)[:1], columns=prices.OHLCV_COLUMNS)
        if check.empty or check["open"].isna().all():
            raise RuntimeError("Uploaded file could not be read back with OHLCV columns")
    logger.info("Backfill summary: %s", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exchange", choices=["jkse", "sp500"], required=True)
    parser.add_argument("--upload", action="store_true", help="Back up and replace the R2 file")
    parser.add_argument("--max-concurrency", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        backfill(args.exchange, args.upload, args.max_concurrency)
        return 0
    except Exception:
        logger.exception("OHLCV backfill failed")
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sys.exit(main())
