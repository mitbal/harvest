import pickle
import logging

from tqdm import tqdm

import harvest.data as hd


def download_prices(stock_list):

    prices = {}
    for stock in tqdm(stock_list):
        prices[stock] = hd.get_daily_stock_price(stock, n_days=3650)

    with open('data/prices.pkl', 'wb') as f:
        pickle.dump(prices, f)

    return prices


def download_financials(stock_list):
    
    financials = {}
    for stock in tqdm(stock_list):
        financials[stock] = hd.get_financial_data(stock, period='quarter')

    with open('data/financials.pkl', 'wb') as f:
        pickle.dump(financials, f)

    return financials


def download_dividends(stock_list):

    dividends = hd.get_dividend_history(stock_list)

    with open('data/dividends.pkl', 'wb') as f:
        pickle.dump(dividends, f)

    return dividends


if __name__ == '__main__':
    
    idxs = hd.get_all_idx_stocks()
    stock_list = idxs['symbol'].to_list()
    
    cp_df = hd.get_company_profile(stock_list)

    prices = download_prices(stock_list)

    financials = download_financials(stock_list)

    dividends = download_dividends(stock_list)
