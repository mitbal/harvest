"""Inputs for the trading jobs: universe, dividend history, prices and the Supabase client.

Every loader reads what the daily pipeline already publishes (Redis, Supabase
storage, R2), with the repository's local files as a development fallback
where noted. Prices are read strictly from R2 so a job never publishes
signals from a stale local copy.
"""
import io
import logging
import os
import pickle
from typing import Iterable, Optional

import pandas as pd
import redis
from supabase import Client, create_client

import harvest.data as hd
import harvest.dividend_play as dp
import harvest.prices as prices


logger = logging.getLogger(__name__)

REPOSITORY_ROOT = prices.REPOSITORY_ROOT
REQUIRED_ENVIRONMENT = (
    "FMP_API_KEY", "REDIS_URL", "SUPABASE_URL", "SUPABASE_KEY",
    "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET",
)
PROFILE_BATCH_SIZE = 100


def validate_environment() -> None:
    missing = [name for name in REQUIRED_ENVIRONMENT if not os.environ.get(name)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


def get_supabase_client() -> Client:
    """Server-side client; the key must be the service-role key to write past row level security."""
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def read_redis_frame(key: str) -> Optional[pd.DataFrame]:
    """A Parquet DataFrame written by ``daily_data.store_frames_to_redis``, or None."""
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    client = redis.from_url(url, socket_connect_timeout=10, socket_timeout=30)
    try:
        value = client.get(key)
    finally:
        client.close()
    if value is None:
        return None
    return pd.read_parquet(io.BytesIO(value))


def load_company_profiles(symbols: list[str]) -> pd.DataFrame:
    """``companyName`` and ``volAvg`` per stock from FMP, in batches."""
    frames = []
    for offset in range(0, len(symbols), PROFILE_BATCH_SIZE):
        batch = symbols[offset:offset + PROFILE_BATCH_SIZE]
        try:
            frames.append(hd.get_company_profile(batch).reset_index())
        except Exception as exc:
            logger.warning("Company profiles unavailable for %s symbols: %s", len(batch), exc)
    if not frames:
        return pd.DataFrame(columns=["stock", "companyName", "volAvg"])
    profiles = pd.concat(frames, ignore_index=True).rename(columns={"symbol": "stock"})
    return profiles[[c for c in ("stock", "companyName", "volAvg") if c in profiles.columns]]


def load_universe(exchange: str = "jkse") -> pd.DataFrame:
    """Dividend statistics per stock (``div_score_<exch>``) with company name and liquidity."""
    frame = read_redis_frame(f"div_score_{exchange}")
    if frame is None or frame.empty:
        path = REPOSITORY_ROOT / "data" / exchange / "div_score.csv"
        logger.warning("div_score_%s not in Redis; using %s", exchange, path)
        frame = pd.read_csv(path)
    frame = frame.rename(columns={"symbol": "stock"})
    profiles = load_company_profiles(frame["stock"].dropna().astype(str).tolist())
    if not profiles.empty:
        frame = frame.drop(columns=[c for c in ("companyName", "volAvg") if c in profiles.columns and c in frame.columns])
        frame = frame.merge(profiles, on="stock", how="left")
    for column in ("price", "yield", "mktCap", "numDividendYear", "volAvg", "DScore"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates("stock").reset_index(drop=True)


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
