from pathlib import Path
from typing import Optional

import pandas as pd
from prefect import flow, task
from supabase import Client

from pipeline import historical_prices_core as core


DEFAULT_RETRIES = core.DEFAULT_RETRIES
DEFAULT_RETRY_DELAY = core.DEFAULT_RETRY_DELAY
DEFAULT_MAX_CONCURRENCY = core.DEFAULT_MAX_CONCURRENCY
BACKFILL_START_DATE = core.BACKFILL_START_DATE
get_supabase_client = core.get_supabase_client
get_local_path = core.get_local_path


@task(log_prints=True, retries=DEFAULT_RETRIES, retry_delay_seconds=DEFAULT_RETRY_DELAY)
def fetch_stock_price(symbol: str, start_date: str) -> Optional[pd.DataFrame]:
    """Prefect compatibility task for one price download."""
    return core.fetch_stock_price(symbol, start_date)


@task(log_prints=True)
def upsert_to_db(frame: pd.DataFrame, client: Optional[Client] = None) -> None:
    """Prefect compatibility task for one database upsert."""
    core.upsert_to_db(frame, client=client)


@flow(name="Historical Price Pipeline", log_prints=True)
def run_historical_pipeline(
    exch: str = "jkse",
    mode: str = "incremental",
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    use_local_cache: bool = True,
    cache_path: Optional[Path] = None,
):
    """Prefect compatibility wrapper around the plain historical pipeline."""
    return core.run_historical_pipeline(
        exchange=exch,
        mode=mode,
        max_concurrency=max_concurrency,
        use_local_cache=use_local_cache,
        cache_path=cache_path,
    )


if __name__ == "__main__":
    run_historical_pipeline(exch="jkse", mode="incremental")
