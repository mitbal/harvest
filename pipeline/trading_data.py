"""Inputs for the trading jobs: universe, dividend history, prices and the Supabase client.

Every loader reads what the daily pipeline already publishes (Redis, Supabase
storage, R2), with the repository's local files as a development fallback
where noted. Prices are read strictly from R2 so a job never publishes
signals from a stale local copy.
"""
import logging
import os
import pickle
import threading
import time
from typing import Callable, Iterable, Optional

import pandas as pd
import requests
from supabase import Client, create_client

import harvest.data as hd
import harvest.dividend_play as dp
import harvest.position as pos
import harvest.prices as prices
from harvest.universe import load_company_profiles, load_universe, read_redis_frame


__all__ = [
    "validate_environment", "get_supabase_client", "read_redis_frame", "load_company_profiles",
    "load_universe", "load_dividend_events", "load_price_histories", "load_jkse_index", "load_ohlcv_histories",
    "with_retries", "load_long_term_universe", "load_long_term_fundamentals",
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


RETRY_STATUS = {429, 500, 502, 503, 504}
# FMP answers 429 when the scan's workers call it faster than the plan allows.
FMP_CALLS_PER_MINUTE = int(os.environ.get("FMP_CALLS_PER_MINUTE", "240"))


class RateLimiter:
    """Spaces calls at least ``60 / per_minute`` seconds apart across threads."""

    def __init__(self, per_minute: int):
        self.interval = 60.0 / per_minute
        self.lock = threading.Lock()
        self.next_call = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_call)
            self.next_call = slot + self.interval
        time.sleep(max(0.0, slot - now))


fmp_limiter = RateLimiter(FMP_CALLS_PER_MINUTE)


def with_retries(fetch: Callable, *args, attempts: int = 4, base_delay: float = 5.0, **kwargs):
    """``fetch(*args, **kwargs)``, retried with backoff on rate limits, server errors and timeouts."""
    for attempt in range(attempts):
        try:
            return fetch(*args, **kwargs)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in RETRY_STATUS or attempt == attempts - 1:
                raise
        except (requests.ConnectionError, requests.Timeout):
            if attempt == attempts - 1:
                raise
        time.sleep(base_delay * 2 ** attempt)


def load_long_term_universe() -> pd.DataFrame:
    """Every actively traded IDX company with FMP profile fields, largest first."""
    universe = with_retries(pos.load_idx_universe)
    if universe.empty:
        raise RuntimeError("The IDX universe is unavailable from FMP")
    return universe


def load_long_term_fundamentals(symbol: str) -> tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """Daguerreo quarterly income statements and FMP statements for one stock; FMP errors propagate."""
    financials = with_retries(hd.get_financial_data, symbol, period="quarter", source="dag")

    def statement(name: str) -> pd.DataFrame:
        fmp_limiter.wait()
        return hd.get_fmp_statement(symbol, name)

    frames = [with_retries(statement, name) for name in pos.STATEMENT_FIELDS]
    return financials, pos.prepare_statements(*frames)
