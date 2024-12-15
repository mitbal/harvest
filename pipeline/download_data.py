import pickle
import logging

from tqdm import tqdm
from prefect import flow, task

import harvest.data as hd

@flow
def download_all(start_from):

    idxs = hd.get_all_idx_stocks()
    stock_list = idxs['symbol'].to_list()
    
    cp_df = hd.get_company_profile(stock_list)
    cp_df.to_csv('data/company_profiles.csv', index=False)

    prices = download_prices(stock_list, start_from=start_from)
    with open('data/prices.pkl', 'wb') as f:
        pickle.dump(prices, f)

    financials = download_financials(stock_list)

    dividends = download_dividends(stock_list)

@flow
def download_prices(stock_list, start_from=None):

    prices = {}
    for stock in tqdm(stock_list):
        prices[stock] = download_single_price(stock, start_from)

    return prices

@task
def download_single_price(stock, start_from):
    print(f'download {stock}')
    return hd.get_daily_stock_price(stock, start_from=start_from)

@task
def download_financials(stock_list):
    
    financials = {}
    for stock in tqdm(stock_list):
        financials[stock] = hd.get_financial_data(stock, period='quarter')

    with open('data/financials.pkl', 'wb') as f:
        pickle.dump(financials, f)

    return financials

@task
def download_dividends(stock_list):

    dividends = hd.get_dividend_history(stock_list)

    with open('data/dividends.pkl', 'wb') as f:
        pickle.dump(dividends, f)

    return dividends


if __name__ == '__main__':
    
    download_all(start_from='2000-01-01')
