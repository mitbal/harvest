"""Inputs for the trading jobs: universe, dividend history, prices and the Supabase client.

Every loader reads what the daily pipeline already publishes (Redis, Supabase
storage, R2), with the repository's local files as a development fallback
where noted. Prices are read strictly from R2 so a job never publishes
signals from a stale local copy.
"""
import logging
import os
import pickle
from typing import Iterable, Optional

import pandas as pd
from supabase import Client, create_client

import harvest.data as hd
import harvest.dividend_play as dp
import harvest.prices as prices
from harvest.universe import load_company_profiles, load_universe, read_redis_frame


__all__ = [
    "validate_environment", "get_supabase_client", "read_redis_frame", "load_company_profiles",
    "load_universe", "load_dividend_events", "load_price_histories", "load_jkse_index", "load_ohlcv_histories",
]
logger = logging.getLogger(__name__)

REPOSITORY_ROOT = prices.REPOSITORY_ROOT
REQUIRED_ENVIRONMENT = (
    "FMP_API_KEY", "REDIS_URL", "SUPABASE_URL", "SUPABASE_KEY",
    "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET",
)


def validate_environment() -> None:
    missing = [name for name in REQUIRED_ENVIRONMENT if not os.environ.get(name)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


def get_supabase_client() -> Client:
    """Server-side client; the key must be the service-role key to write past row level security."""
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def load_dividend_events(client: Optional[Client], exchange: str = "jkse") -> dict[str, pd.DataFrame]:
    """Ex-dates and amounts per symbol from the pickle the daily pipeline stores."""
    path = f"data/{exchange}/dividends.pkl"
    raw = None
    if client is not None:
        try:
            raw = pickle.loads(client.storage.from_("harvest_dividend").download(path))
        except Exception as exc:
            logger.warning("Dividend history not available from storage: %s", exc)
    if raw is None:
        local = REPOSITORY_ROOT / path
        logger.warning("Using local dividend history %s", local)
        raw = pd.read_pickle(local)

    events = {}
    for symbol, records in (raw or {}).items():
        frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        frame = dp.prepare_dividends(frame)
        if not frame.empty:
            events[symbol] = frame
    return events


def load_price_histories(symbols: Iterable[str], start, exchange: str = "jkse") -> dict[str, pd.DataFrame]:
    """Clean ``date, close`` frames per symbol from R2; failures propagate."""
    frame = prices.read_prices(exchange, symbols=list(symbols), start=start)
    return {
        symbol: dp.prepare_prices(group)
        for symbol, group in frame.groupby("symbol")
    }


def load_jkse_index() -> pd.DataFrame:
    """Daily JKSE closes (``close`` by date) for the market regime, from FMP."""
    frame = hd.get_daily_stock_price("^JKSE", start_from="2000-01-01")
    if frame is None or frame.empty:
        raise RuntimeError("JKSE index prices are unavailable")
    frame = frame.assign(date=pd.to_datetime(frame["date"])).set_index("date").sort_index()
    return frame[["close"]]


def load_ohlcv_histories(symbols: Iterable[str], start, exchange: str = "jkse") -> dict[str, pd.DataFrame]:
    """OHLCV frames by date per symbol from R2; failures propagate."""
    frame = prices.read_prices(exchange, symbols=list(symbols), start=start, columns=prices.OHLCV_COLUMNS)
    return {symbol: group.drop(columns="symbol").set_index("date") for symbol, group in frame.groupby("symbol")}
