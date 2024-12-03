import os
import datetime
import requests

import numpy as np
import pandas as pd
import vectorbt as vbt

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


def preprocess_div(div_df):
    """
    Aggregate dividend payment in yearly basis (in case there is/are one or more interim)
    and fill in the year where it does not pay dividend with 0
    """
    div_year_df = div_df.copy()
    div_year_df['year'] = div_df.apply(lambda x: int(x['date'].split('-')[0]), axis=1)
    div_year_df = div_year_df.groupby('year')['adjDividend'].sum().to_frame().reset_index()
    
    start_year = div_year_df.loc[0, 'year']
    end_year = div_year_df.loc[len(div_year_df)-1, 'year']

    years = list(range(start_year, end_year + 1))
    df_temp = pd.DataFrame({'year': years, 'value': [0]*len(years)})
    full_div_df = pd.merge(df_temp, div_year_df, on='year', how='left')
    full_div_df = full_div_df.fillna(0)

    return full_div_df


def calc_div_stats(div_df):

    stats = {}
    
    div_df['inc_flat'] = div_df['adjDividend'] - div_df['adjDividend'].shift(1)
    div_df['inc_pct'] = div_df['inc_flat'] / div_df['adjDividend'].shift(1) * 100

    stats['historical_mean_flat'] = div_df['inc_flat'].mean()
    stats['div_inc_2y_mean_flat'] = div_df['inc_flat'][-2:].mean()
    stats['div_inc_5y_mean_flat'] = div_df['inc_flat'][-5:].mean()
    stats['exponential_weighted_mean_flat'] = div_df['inc_flat'].ewm(com=0.5).mean().mean()

    stats['historical_mean_pct'] = div_df['inc_pct'].mean()
    stats['div_inc_2y_mean_pct'] = div_df['inc_pct'][-2:].mean()
    stats['div_inc_5y_mean_pct'] = div_df['inc_pct'][-5:].mean()
    stats['exponential_weighted_mean_pct'] = div_df['inc_pct'].ewm(com=0.5).mean().mean()

    stats['num_positive_year'] = np.sum(div_df['inc_flat'] > 0) + 1 #the first year is considered positive
    stats['num_dividend_year'] = len(div_df)
    stats['pct_positive_year'] = stats['num_positive_year'] / stats['num_dividend_year'] * 100

    return stats


def calc_pe_history(price_df, fin_df):

    pdf = price_df[['date', 'close']].copy()
    pdf['date'] = pd.to_datetime(pdf['date'])
    pdf = pdf.sort_values('date')

    edf = fin_df[['date', 'eps']].sort_values(ascending=True, by='date').rolling(window=4, on='date').sum()
    edf['date'] = pd.to_datetime(edf['date'])
    edf = edf.sort_values('date')

    pe_df = pd.merge_asof(pdf, edf, on='date', direction='backward')
    pe_df['pe'] = pe_df['close'] / pe_df['eps']

    return pe_df


def make_labels(price_df, threshold):

    labels = vbt.LEXLB.run(price_df, threshold, threshold).labels
    return labels

def calc_labels_stats(price_df, labels):
    
    buy = price_df[np.where(labels == -1, True, False)]
    sell = price_df[np.where(labels == 1, True, False)]

    buy['date'] = pd.to_datetime(buy['date'])
    sell['date'] = pd.to_datetime(sell['date'])
    sell['sell_date'] = sell['date']

    trades = pd.merge_asof(buy[['date', 'close']], sell[['date', 'close', 'sell_date']], on='date', direction='forward', suffixes=['_buy', '_sell'])
    trades['gain'] = (trades['close_sell'] / trades['close_buy'])*100 - 100
    trades['duration'] = trades['sell_date'] - trades['date']

    return trades
