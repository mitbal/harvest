import os
import json
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import redis
import numpy as np
import pandas as pd
from prefect import flow, task
from supabase import create_client, Client
from prefect.artifacts import create_markdown_artifact, create_table_artifact

import harvest.data as hd


DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 30
DEFAULT_MAX_CONCURRENCY = 5


def store_df_to_redis(key: str, df: pd.DataFrame) -> None:
    """Stores a Pandas DataFrame to Redis as JSON."""

    redis_url = os.environ['REDIS_URL']
    r = redis.from_url(redis_url)
    try:
        df_json = df.to_json(orient='records')
        data = {'date': datetime.now().strftime('%Y-%m-%d'), 'content': df_json}
        r.set(key, json.dumps(data))
    except redis.exceptions.ConnectionError as e:
        print(f"Error connecting to Redis: {e}")
    except Exception as e:
        print(f"Error storing data to Redis: {e}")


def store_to_supabase_storage(filename: str, content: dict) -> None:
    """Stores the given content as a JSON file in Supabase storage.

    Args:
        filename: The name of the file to be stored (e.g., "data.json").
        content: A dictionary containing the data to be stored.
    """

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL and Key must be set as environment variables.")

    supabase: Client = create_client(supabase_url, supabase_key)

    bucket_name = "harvest_dividend"

    try:
        json_data = pickle.dumps(content)

        response = supabase.storage.from_(bucket_name).upload(
            path=filename,
            file=json_data,
            file_options={"contentType": "application/octet-stream",
                          'upsert': 'true'}
        )
        print(response)
        print(f"Successfully uploaded {filename} to Supabase storage.")

    except Exception as e:
        print(f"Error storing data to Supabase: {e}")


@flow
def run_daily(exch: str = 'jkse', mcap_filter: int = 100_000_000_000):
    """
    Runs the daily data pipeline for a given exchange.

    Args:
        exch:  Exchange ('jkse' or 'sp500').
        mcap_filter: Market capitalization filter.
    """
    if exch not in ('jkse', 'sp500'):
        raise ValueError('exch must be either jkse or sp500')
    
    if exch == 'jkse':
        idxs = hd.get_all_idx_stocks()
        syariah = pd.read_csv(f'data/{exch}/syariah.csv', sep=';')
        syariah['symbol'] = syariah['Kode'].apply(lambda x: x+'.JK')
    else:
        idxs = hd.get_all_sp500_stocks()
        syariah = None

    stock_list = idxs['symbol'].to_list()
    cp_df = hd.get_company_profile(stock_list)
    stock_dividend_list = cp_df[(cp_df['lastDiv'] != 0)].index.to_list()

    dividends = download_dividends(stock_dividend_list)
    financials = download_financials(stock_dividend_list)
    # prices = download_prices(stock_dividend_list)

    if exch == 'jkse':
        cp_df = cp_df.merge(syariah, on='symbol', how='left')
        cp_df['is_syariah'] = ~cp_df['Kode'].isnull()
        cp_df.set_index('symbol', inplace=True)

    div_stats = compute_div_score(cp_df, financials, dividends, sl=exch)
    store_df_to_redis(f'div_score_{exch}', div_stats.reset_index())

    div_cal = prep_div_cal(cp_df, dividends, filter=mcap_filter)
    store_df_to_redis(f'div_cal_{exch}', div_cal)

    # store_to_supabase_storage(f'data/{exch}/prices.pkl', prices)
    store_to_supabase_storage(f'data/{exch}/dividends.pkl', dividends)
    store_to_supabase_storage(f'data/{exch}/financials.pkl', financials)


@flow
def download_financials(stock_list, max_concurrency: int = DEFAULT_MAX_CONCURRENCY):
    """Download financial data in parallel using ThreadPoolExecutor."""
    return _download_data(stock_list, download_single_fin, "financial", max_concurrency)


@flow
def download_dividends(stock_list, max_concurrency: int = DEFAULT_MAX_CONCURRENCY):
    """Download dividend data in parallel using ThreadPoolExecutor."""
    return _download_data(stock_list, download_single_dividend, "dividend", max_concurrency)


@flow
def download_prices(stock_list, max_concurrency: int = DEFAULT_MAX_CONCURRENCY):
    """Download price data in parallel using ThreadPoolExecutor."""
    return _download_data(stock_list, download_single_price, "price", max_concurrency)


def _download_data(stock_list, download_func, data_type, max_concurrency):
    """
    Helper function to download data in parallel.
    """
    data = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(download_func, stock): stock for stock in stock_list}
        for future in futures:
            stock = futures[future]
            try:
                result = future.result()
                if result is not None:
                    data[stock] = result
            except Exception as e:
                print(f"Error downloading {data_type} data for {stock}: {e}")

    # Create artifact to track data completeness
    total_stocks = len(stock_list)
    successful_downloads = len(data)
    failed_downloads = total_stocks - successful_downloads

    markdown_content = f"""
    # {data_type.capitalize()} Data Download Summary
    - Total stocks processed: {total_stocks}
    - Successful downloads: {successful_downloads} 
    - Failed downloads: {failed_downloads}
    - Success rate: {(successful_downloads/total_stocks)*100:.1f}%
    """

    create_markdown_artifact(
        key=f"{data_type}-data-summary",
        markdown=markdown_content,
        description=f"Summary of {data_type} data download completeness"
    )

    return data

@task(log_prints=True, retries=DEFAULT_RETRIES, retry_delay_seconds=DEFAULT_RETRY_DELAY)
def download_single_fin(stock: str):
    """Downloads a single financial report."""
    print(f'download financial report {stock}')
    try:
        fin = hd.get_financial_data(stock, period='quarter')
        return fin
    except Exception as e:
        print(f'Error downloading financial report {stock}: {e}')
        raise e


@task(log_prints=True, retries=DEFAULT_RETRIES, retry_delay_seconds=DEFAULT_RETRY_DELAY)
def download_single_dividend(stock: str):
    """Downloads dividend history for a single stock."""
    print(f'download dividend history for {stock}')
    try:
        div = hd.get_dividend_history_single_stock(stock, source='dag')
        return div
    except Exception as e:
        print(f'Error downloading dividend history for {stock}: {e}')
        raise e


@task(log_prints=True, retries=DEFAULT_RETRIES, retry_delay_seconds=DEFAULT_RETRY_DELAY)
def download_single_price(stock: str):
    """Downloads price history for a single stock."""
    print(f'download price history for {stock}')
    try:
        price = hd.get_daily_stock_price(stock, start_from='2010-01-01')
        return price
    except Exception as e:
        print(f'Error downloading price history for {stock}: {e}')
        raise e


@task
def prep_div_cal(cp, div_dict, filter, year=2024):

    cp = cp[cp['mktCap'] >= filter].copy()
    cp.reset_index(drop=False, inplace=True)

    div_df = hd.prep_div_cal(div_dict, cp, year)
    return div_df


@task
def compute_div_score(cp_df: pd.DataFrame, fin_dict: dict, div_dict: dict, sl: str = 'jkse') -> pd.DataFrame:
    """Computes the dividend score for each stock."""

    df = cp_df[(cp_df['lastDiv'] != 0)].copy()
    df['yield'] = 0
    df['revenueGrowth'] = np.nan
    df['netIncomeGrowth'] = np.nan
    df['avgFlatAnnualDivIncrease'] = np.nan
    df['avgPctAnnualDivIncrease'] = np.nan
    df['numDividendYear'] = np.nan
    df['positiveYear'] = np.nan
    df['numOfYear'] = np.nan

    stock_list = df.index.tolist()
    for symbol in stock_list:

        try:
            fin_df = fin_dict[symbol]
            fin_stats = hd.calc_fin_stats(fin_df)
            df.loc[symbol, 'revenueGrowth'] = fin_stats['trim_mean_10y_revenue_growth']
            df.loc[symbol, 'netIncomeGrowth'] = fin_stats['trim_mean_10y_netIncome_growth']
            df.loc[symbol, 'medianProfitMargin'] = fin_stats['median_profit_margin']
            
            div_df = div_dict[symbol]
            agg_year = div_df[div_df['dividend_type'] != 'special'].groupby('fiscal_year')['adjDividend'].sum().to_frame()
            final_year = div_df[div_df['dividend_type'] == 'final']['fiscal_year'].to_list()[0]
            last_div = agg_year.loc[final_year, 'adjDividend']

            div_df = hd.preprocess_div(div_df)
            div_stats = hd.calc_div_stats(div_df)
            
            div_incs = np.array([div_stats['historical_mean_flat'],
                                div_stats['div_inc_5y_mean_flat']])
            div_incs = np.nan_to_num(div_incs, nan=0.0)
            
            df.loc[symbol, 'lastDiv'] = last_div
            df.loc[symbol, 'yield'] = last_div / df.loc[symbol, 'price'] * 100
            df.loc[symbol, 'avgFlatAnnualDivIncrease'] = np.min(div_incs)
            df.loc[symbol, 'avgPctAnnualDivIncrease'] = div_stats['historical_mean_pct']
            df.loc[symbol, 'numDividendYear'] = div_stats['num_dividend_year']
            df.loc[symbol, 'positiveYear'] = div_stats['num_positive_year']
            df.loc[symbol, 'numOfYear'] = datetime.today().year - datetime.strptime(df.loc[symbol, 'ipoDate'], '%Y-%m-%d').year

        except Exception as e:
            print(f'error {e}', symbol)
            continue
    
    # patented dividend score
    df['DScore'] = hd.calc_div_score(df)

    features = ['price', 'lastDiv', 'yield', 'sector', 'industry', 'mktCap', 'ipoDate',
               'revenueGrowth', 'netIncomeGrowth', 'medianProfitMargin',
               'avgFlatAnnualDivIncrease', 'numDividendYear', 'DScore']
    
    if sl == 'jkse':
        features = ['is_syariah'] + features
    final_df = df[features]


    markdown_content = f"""
    # Dividend Score Summary
    Number of non-null score: {final_df[final_df['DScore'].notnull()].shape[0]}
    Percentage of non-null score: {final_df[final_df['DScore'].notnull()].shape[0] / final_df.shape[0] * 100:.2f}%
    """

    create_markdown_artifact(
        key="div-score-summary",
        markdown=markdown_content,
        description= "The summary for dividend score calculation"
    )
    
    create_table_artifact(
        key="div-score-table",
        table=final_df.reset_index().to_dict(orient='records'),
        description= "The final table of dividend score"
    )

    return final_df


if __name__ == '__main__':
    
    run_daily(exch='jkse', mcap_filter=100_000_000_000)
    # run_daily(exch='sp500', mcap_filter=10_000_000_000)
