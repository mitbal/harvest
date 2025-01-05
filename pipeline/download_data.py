import pickle
from datetime import timedelta

from tqdm import tqdm
from prefect import flow, task
from prefect.cache_policies import INPUTS

import harvest.data as hd

@flow
def download_all(start_from):

    idxs = hd.get_all_idx_stocks()
    stock_list = idxs['symbol'].to_list()
    
    cp_df = hd.get_company_profile(stock_list)
    cp_df.to_csv('data/company_profiles.csv')

    prices = download_prices(stock_list, start_from=start_from)
    with open('data/prices.pkl', 'wb') as f:
        pickle.dump(prices, f)

    financials = download_financials(stock_list)
    with open('data/financials.pkl', 'wb') as f:
        pickle.dump(financials, f)

    dividends = download_dividends(stock_list)
    with open('data/dividends.pkl', 'wb') as f:
        pickle.dump(dividends, f)

    # download alternative data
    gold = download_single_price('GCUSD', start_from)
    gold.to_csv('data/gold.csv', index=False)

    oil = download_single_price('CLUSD', start_from)
    oil.to_csv('data/oil.csv', index=False)

    dollar = download_single_price('USDIDR', start_from)
    dollar.to_csv('data/dollar.csv', index=False)

    btc = download_single_price('BTCUSD', start_from)
    btc.to_csv('data/btc.csv', index=False)

@flow
def download_prices(stock_list, start_from=None):

    prices = {}
    for stock in tqdm(stock_list):
        future = download_single_price.submit(stock, start_from)
        result = future.result()
        if result is not None:
            prices[stock] = result

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
def download_financials(stock_list):
    
    financials = {}
    for stock in tqdm(stock_list):
        financials[stock] = download_single_fin(stock)

    return financials

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
    
    download_all(start_from='2000-01-01')
