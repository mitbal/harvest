"""Stock universe shared by the trading pages and jobs: dividend scores plus company profiles."""
import io
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

import harvest.data as hd


logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROFILE_BATCH_SIZE = 100


def read_redis_frame(key: str) -> Optional[pd.DataFrame]:
    """A Parquet DataFrame written by ``daily_data.store_frames_to_redis``, or None."""
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    import redis

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
