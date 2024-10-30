import os
import datetime
import requests
import pandas as pd


def get_all_idx_stocks(api_key=None):
    
    if api_key is None:
        api_key = os.environ['FMP_API_KEY']

    url = f'https://financialmodelingprep.com/api/v3/stock/list?apikey={api_key}'
    r = requests.get(url)
    sl = r.json()

    sdf = pd.DataFrame(sl)
    idx = sdf[sdf['exchangeShortName'] == 'JKT'].reset_index(drop=True)

    return idx


def get_company_profile(stocks, api_key=None):
    
    if api_key is None:
        api_key = os.environ['FMP_API_KEY']
    
    stock_param = ','.join(stocks)
    url = f'https://financialmodelingprep.com/api/v3/profile/{stock_param}?apikey={api_key}'
    r  = requests.get(url)
    cp = r.json()

    cp_df = pd.DataFrame(cp).set_index('symbol')
    return cp_df


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


def get_dividend_history(stocks, api_key=None):
    
    if api_key is None:
        api_key = os.environ['FMP_API_KEY']

    divs = []
    for i in range(int(len(stocks)/5)+1):
        stock_list = [s+'.JK' for s in stocks[i*5:(i+1)*5]]
        stocks_param = ','.join(stock_list)
        dividend_history_url = f'https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{stocks_param}?apikey={api_key}'
        dr = requests.get(dividend_history_url)
        drj = dr.json()
        
        if len(stock_list) > 1:
            div = drj['historicalStockList']
        else:
            if drj:
               div = [drj]
        divs.append(div)
    
    div_df = pd.concat([pd.DataFrame(div) for div in divs])
    div_df.set_index('symbol', inplace=True)
    divs = {x[:-3]: y for x, y in zip(div_df.index, div_df['historical'])}

    return divs


def get_financial_data(stock, period='quarter', api_key=None):

    if api_key is None:
        api_key = os.environ['FMP_API_KEY']

    url = f'https://financialmodelingprep.com/api/v3/income-statement/{stock}?period={period}&apikey={api_key}'
    r = requests.get(url)
    fs = r.json()
    return pd.DataFrame(fs)
