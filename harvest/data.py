import os
import datetime
import requests
import pandas as pd


def get_daily_stock_price(stock, api_key=None, n_days=365):
    if api_key is None:
        api_key = os.environ['FMP_API_KEY']

    start_date = (datetime.datetime.today() - datetime.timedelta(days=n_days)).strftime('%Y-%m-%d')
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{stock}?apikey={api_key}&from={start_date}'
    r = requests.get(url)
    intraday  = r.json()
    
    return pd.DataFrame(intraday['historical'])


def get_sector_industry_pe(date=None, api_key=None):

    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    sector_url = f'https://financialmodelingprep.com/api/v4/sector_price_earning_ratio?date={date}&exchange=JKT&apikey={api_key}'
    industry_url = f'https://financialmodelingprep.com/api/v4/industry_price_earning_ratio?date={date}&exchange=JKT&apikey={api_key}'

    sector = requests.get(sector_url).json()
    industry = requests.get(industry_url).json()

    sector_df = pd.DataFrame(sector)
    industry_df = pd.DataFrame(industry)

    return sector_df, industry_df


def get_company_ratio(stock, api_key):
    
    url = f'https://financialmodelingprep.com/api/v3/ratios/{stock}?period=quarter&apikey={api_key}'
    ratio = requests.get(url).json()
    return pd.DataFrame(ratio)
