import pickle
from datetime import timedelta

import pandas as pd
from tqdm import tqdm
from prefect import flow, task
from prefect.cache_policies import INPUTS
from concurrent.futures import ThreadPoolExecutor

import harvest.data as hd

@flow
def download_all(start_from, exch='jkse'):

    if exch == 'jkse':
        idxs = hd.get_all_idx_stocks()
    elif exch == 'sp500':
        idxs = hd.get_all_sp500_stocks()
    else:
        raise ValueError('exch must be either jkse or sp500')
    
    stock_list = idxs['symbol'].to_list()
    
    cp_df = hd.get_company_profile(stock_list)
    cp_df.to_csv(f'data/{exch}/company_profiles.csv')

    prices = download_prices(stock_list, start_from=start_from)
    with open(f'data/{exch}/prices.pkl', 'wb') as f:
        pickle.dump(prices, f)

    financials = download_financials(stock_list)
    with open(f'data/{exch}/financials.pkl', 'wb') as f:
        pickle.dump(financials, f)

    dividends = download_dividends(stock_list)
    with open(f'data/{exch}/dividends.pkl', 'wb') as f:
        pickle.dump(dividends, f)

    shares = download_shares_outstanding(stock_list)
    shares.to_csv(f'data/{exch}/shares.csv', index=False)

    # download alternative data
    gold = download_single_price('GCUSD', start_from)
    gold.to_csv('data/gold.csv', index=False)

    oil = download_single_price('CLUSD', start_from)
    oil.to_csv('data/oil.csv', index=False)

    dollar = download_single_price('USDIDR', start_from)
    dollar.to_csv('data/dollar.csv', index=False)

    btc = download_single_price('BTCUSD', start_from)
    btc.to_csv('data/btc.csv', index=False)

    spy = download_single_price('SPY', start_from)
    spy.to_csv('data/spy.csv', index=False)

    jkse = download_single_price('^JKSE', start_from)
    jkse.to_csv('data/jkse.csv', index=False)


@task
def download_shares_outstanding(stock_list):

    shares = []
    for stock in tqdm(stock_list):
        print(f'download shares {stock}')
        share = hd.get_shares_outstanding(stock)
        shares += [pd.DataFrame(share)]
    share_df = pd.concat(shares)

    return share_df


@flow
def download_prices(stock_list, start_from=None, max_concurrency=10):  # Added max_concurrency as flow parameter
    """Download price data in parallel using ThreadPoolExecutor."""

    prices = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(download_single_price, stock, start_from): stock for stock in stock_list}
        for future in tqdm(futures, desc="Downloading prices"):
            stock = futures[future]
            try:
                result = future.result()
                if result is not None:
                    prices[stock] = result
            except Exception as e:
                print(f"Error downloading price for {stock}: {e}")

    return prices

@task(cache_policy=INPUTS, cache_expiration=timedelta(days=1), log_prints=True)
def download_single_price(stock, start_from):
    print(f'download price {stock}')
    try:
        price = hd.get_daily_stock_price(stock, start_from=start_from)
        return price
    except Exception as e:
        print(f'Error downloading price {stock}: {e}')
        return None

@flow
def download_financials(stock_list, start_from=None, max_concurrency=10):  # Added max_concurrency as flow parameter
    """Download price data in parallel using ThreadPoolExecutor."""

    fins = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(download_single_fin, stock): stock for stock in stock_list}
        for future in tqdm(futures, desc="Downloading financials data"):
            stock = futures[future]
            try:
                result = future.result()
                if result is not None:
                    fins[stock] = result
            except Exception as e:
                print(f"Error downloading financial data for {stock}: {e}")

    return fins


@task(cache_policy=INPUTS, cache_expiration=timedelta(days=1), log_prints=True)
def download_single_fin(stock):
    print(f'download financial report {stock}')
    try:
        fin = hd.get_financial_data(stock, period='quarter')
        return fin
    except Exception as e:
        print(f'Error downloading financial report {stock}: {e}')
        return None

@task
def download_dividends(stock_list):

    dividends = hd.get_dividend_history(stock_list)

    return dividends


if __name__ == '__main__':
    
    download_all(start_from='2000-01-01', exch='jkse')
    download_all(start_from='2000-01-01', exch='sp500')
