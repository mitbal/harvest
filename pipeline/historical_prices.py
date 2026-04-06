import os
import datetime
import pickle
import pandas as pd
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from prefect import flow, task
from supabase import create_client, Client
import harvest.data as hd

# Configuration
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 10
DEFAULT_MAX_CONCURRENCY = 10
BACKFILL_START_DATE = '2010-01-01'

def get_supabase_client() -> Client:
    """Initialize and return the Supabase client."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")
    return create_client(url, key)

def get_local_path(exch: str) -> str:
    """Get the local filesystem path for the historical data cache."""
    return f"data/{exch}/historical_prices.pkl"

@task(log_prints=True, retries=DEFAULT_RETRIES, retry_delay_seconds=DEFAULT_RETRY_DELAY)
def fetch_stock_price(symbol: str, start_date: str) -> Optional[pd.DataFrame]:
    """Fetch daily stock price for a single symbol from FMP."""
    try:
        df = hd.get_daily_stock_price(symbol, start_from=start_date)
        if df is not None and not df.empty:
            # We only need symbol, date, and close
            df['symbol'] = symbol
            # Ensure date is in YYYY-MM-DD format
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            return df[['symbol', 'date', 'close']]
        return None
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

@task(log_prints=True)
def upsert_to_db(df: pd.DataFrame):
    """Upsert the dataframe to the historical_prices table in Supabase."""
    if df.empty:
        return
    
    supabase = get_supabase_client()
    records = df.to_dict(orient='records')
    
    # Supabase allows upsert with on_conflict
    try:
        # Note: 'upsert' in supabase-py uses the primary key by default for conflicts
        response = supabase.table("historical_prices").upsert(records).execute()
        return response
    except Exception as e:
        print(f"Error upserting to database: {e}")
        # If batch is too large, we might need to chunk it (handled in flow)
        raise e

@flow(name="Historical Price Pipeline")
def run_historical_pipeline(exch: str = 'jkse', mode: str = 'incremental'):
    """
    Orchestrates the historical price download and storage process.
    
    Args:
        exch: 'jkse' or 'sp500'
        mode: 'backfill' (since 2010) or 'incremental' (last 7 days)
    """
    print(f"Starting historical price pipeline for {exch} (mode: {mode})")
    
    # 1. Get stock list
    if exch == 'jkse':
        stocks = hd.get_all_idx_stocks()
    elif exch == 'sp500':
        stocks = hd.get_all_sp500_stocks()
    else:
        raise ValueError("exch must be either 'jkse' or 'sp500'")
    
    symbol_list = stocks['symbol'].tolist()
    
    # 2. Determine start date
    if mode == 'backfill':
        start_date = BACKFILL_START_DATE
    else:
        # Default to last 7 days for incremental updates
        start_date = (datetime.datetime.today() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    
    # 3. Load existing local data (optional optimization)
    local_file = get_local_path(exch)
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    
    existing_df = pd.DataFrame()
    if os.path.exists(local_file) and mode == 'incremental':
        try:
            with open(local_file, 'rb') as f:
                existing_df = pickle.load(f)
            print(f"Loaded {len(existing_df)} records from {local_file}")
            # we could potentially optimize start_date further here based on existing_df
        except Exception as e:
            print(f"Error loading local cache: {e}")

    # 4. Fetch data in parallel
    print(f"Fetching data for {len(symbol_list)} stocks in parallel...")
    all_data = []
    with ThreadPoolExecutor(max_workers=DEFAULT_MAX_CONCURRENCY) as executor:
        futures = {executor.submit(fetch_stock_price, s, start_date): s for s in symbol_list}
        for future in futures:
            res = future.result()
            if res is not None:
                all_data.append(res)
    
    if not all_data:
        print("No new data fetched.")
        return

    new_df = pd.concat(all_data).drop_duplicates(subset=['symbol', 'date'])
    print(f"Fetched total of {len(new_df)} new records.")

    # 5. Save to local file
    if not existing_df.empty:
        # Combine and deduplicate
        final_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['symbol', 'date'])
    else:
        final_df = new_df
        
    print(f"Updating local cache at {local_file} with {len(final_df)} records.")
    with open(local_file, 'wb') as f:
        pickle.dump(final_df, f)

    # 6. Upsert to DB in chunks (to prevent payload size limits)
    print("Upserting data to Postgres...")
    chunk_size = 1000
    for i in range(0, len(new_df), chunk_size):
        chunk = new_df.iloc[i : i + chunk_size]
        upsert_to_db(chunk)
    
    print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    # You can change exch and mode here or pass them via CLI if integrated with Prefect
    # default to incremental jkse
    # run_historical_pipeline(exch='jkse', mode='incremental')
    # run_historical_pipeline(exch='jkse', mode='backfill')
    # run_historical_pipeline(exch='sp500', mode='incremental')
    run_historical_pipeline(exch='sp500', mode='backfill')
