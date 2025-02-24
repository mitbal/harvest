import os
import json
from datetime import timedelta, datetime

import redis
import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.cache_policies import INPUTS
from concurrent.futures import ThreadPoolExecutor
from prefect.artifacts import create_markdown_artifact

import harvest.data as hd


def store_df_to_redis(key, df):

    url = os.environ['REDIS_URL']
    r = redis.from_url(url)
    df_json = json.dumps(df.to_dict(orient='records'))

    today = datetime.now().strftime('%Y-%m-%d')

    data = {}
    data['date'] = today
    data['content'] = df_json

    r.set(key, json.dumps(data))


@flow
def run_daily(exch='jkse', mcap_filter=100_000_000_000):

    if exch == 'jkse':
        idxs = hd.get_all_idx_stocks()
    elif exch == 'sp500':
        idxs = hd.get_all_sp500_stocks()
    else:
        raise ValueError('exch must be either jkse or sp500')
    
    stock_list = idxs['symbol'].to_list()
    
    cp_df = hd.get_company_profile(stock_list)
    financials = download_financials(stock_list)
    dividends = download_dividends(stock_list)

    if exch == 'jkse':
        syariah = pd.read_csv(f'data/{exch}/syariah.csv', sep=';')
        syariah['symbol'] = syariah['Kode'].apply(lambda x: x+'.JK')
        cp_df = cp_df.merge(syariah, on='symbol', how='left')
        cp_df['is_syariah'] = ~cp_df['Kode'].isnull()
    cp_df.set_index('symbol', inplace=True)

    div_stats = compute_div_score(cp_df, financials, dividends, sl=exch)
    store_df_to_redis(f'div_score_{exch}', div_stats.reset_index())

    div_cal = prep_div_cal(cp_df, financials, dividends, filter=mcap_filter)
    store_df_to_redis(f'div_cal_{exch}', div_cal)


@flow
def download_financials(stock_list, max_concurrency=5):  # Added max_concurrency as flow parameter
    """Download price data in parallel using ThreadPoolExecutor."""

    fins = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(download_single_fin, stock): stock for stock in stock_list}
        for future in futures:
            stock = futures[future]
            try:
                result = future.result()
                if result is not None:
                    fins[stock] = result
            except Exception as e:
                print(f"Error downloading financial data for {stock}: {e}")

    # Create artifact to track data completeness
    total_stocks = len(stock_list)
    successful_downloads = len(fins)
    failed_downloads = total_stocks - successful_downloads
    
    markdown_content = f"""
    # Financial Data Download Summary
    - Total stocks processed: {total_stocks}
    - Successful downloads: {successful_downloads} 
    - Failed downloads: {failed_downloads}
    - Success rate: {(successful_downloads/total_stocks)*100:.1f}%
    """
    
    create_markdown_artifact(
        key="financial-data-summary",
        markdown=markdown_content,
        description="Summary of financial data download completeness"
    )

    return fins


@task(log_prints=True, retries=3, retry_delay_seconds=10)
def download_single_fin(stock):
    print(f'download financial report {stock}')
    try:
        fin = hd.get_financial_data(stock, period='quarter')
        return fin
    except Exception as e:
        print(f'Error downloading financial report {stock}: {e}')
        return None

@flow
def download_dividends(stock_list, max_concurrency=5):
    """Download dividend data in parallel using ThreadPoolExecutor."""

    dividends = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(download_single_dividend, stock): stock for stock in stock_list}
        for future in futures:
            stock = futures[future]
            try:
                result = future.result()
                if result is not None:
                    dividends[stock] = result
            except Exception as e:
                print(f"Error downloading dividend data for {stock}: {e}")
    
    total_stocks = len(stock_list)
    successful_downloads = len(dividends)
    failed_downloads = total_stocks - successful_downloads

    markdown_content = f"""
    # Dividend Data Download Summary
    - Total stocks processed: {total_stocks}
    - Successful downloads: {successful_downloads}
    - Failed downloads: {failed_downloads} 
    - Success rate: {(successful_downloads/total_stocks)*100:.1f}%
    """

    create_markdown_artifact(
        key="dividend-data-summary", 
        markdown=markdown_content,
        description="Summary of dividend data download completeness"
    )
    
    return dividends

@task(log_prints=True, retries=3, retry_delay_seconds=10)
def download_single_dividend(stock):
    print(f'download dividend history for {stock}')
    try:
        div = hd.get_dividend_history_single_stock(stock)
        return div
    except Exception as e:
        print(f'Error downloading dividend history for {stock}: {e}')
        return None

@task
def prep_div_cal(cp, div_dict, filter):

    cp = cp[cp['mktCap'] >= filter].copy()
    cp.reset_index(drop=False, inplace=True)

    year = 2024
    div_df = pd.DataFrame()
    for key, val in div_dict.items():
        if key in ['GGRP.JK', 'IKBI.JK']:
            continue
        temp = pd.DataFrame(val)
        temp['ticker'] = key

        if key in ['ISAT.JK', 'KDSI.JK']:
            temp['adjDividend'] = temp['adjDividend'] / 4

        div_df = pd.concat([div_df, temp])
    div_df.reset_index(drop=True, inplace=True)
    div_df['date'] = pd.to_datetime(div_df['date'])

    div_year = div_df[div_df['date'].dt.year == year]
    merged = div_year.merge(cp[['symbol', 'price']], left_on='ticker', right_on='symbol')
    merged['yield'] = merged['adjDividend'] / merged['price'] * 100

    div_2024 = merged[['date', 'symbol', 'adjDividend', 'price']].copy()
    div_2024['date'] = div_2024['date'].dt.strftime('%Y-%m-%d')

    return div_2024

@task
def compute_div_score(cp_df, fin_dict, div_dict, sl='jkse'):
    
    df = cp_df[(cp_df['isActivelyTrading']) & (cp_df['lastDiv'] != 0)].copy()
    df['yield'] = df['lastDiv'] / df['price'] * 100
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
                    
            div_df = div_dict[symbol]
            div_df = hd.preprocess_div(div_df)
            div_stats = hd.calc_div_stats(div_df)
            
            div_incs = np.array([div_stats['historical_mean_flat'],
                                div_stats['div_inc_5y_mean_flat']])
            div_incs = np.nan_to_num(div_incs, nan=0.0)
            
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

    if sl == 'jkse':
        features = ['price', 'lastDiv', 'yield', 'sector', 'industry', 'mktCap', 'ipoDate', 'is_syariah',
               'revenueGrowth', 'netIncomeGrowth', 
               'avgFlatAnnualDivIncrease', 'numDividendYear', 'DScore']
    else:
        features = ['price', 'lastDiv', 'yield', 'sector', 'industry', 'mktCap', 'ipoDate',
               'revenueGrowth', 'netIncomeGrowth', 
               'avgFlatAnnualDivIncrease', 'numDividendYear', 'DScore']
    
    return df[features]


if __name__ == '__main__':
    
    run_daily(exch='jkse', mcap_filter=100_000_000_000)
    run_daily(exch='sp500', mcap_filter=10_000_000_000)
