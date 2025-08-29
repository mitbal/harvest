import os
import io
import datetime
import requests

import scipy
import numpy as np
import pandas as pd
import PIL.Image as Image


def get_all_idx_stocks(api_key=None):
    
    if api_key is None:
        api_key = os.environ['FMP_API_KEY']

    url = f'https://financialmodelingprep.com/api/v3/stock/list?apikey={api_key}'
    r = requests.get(url)
    sl = r.json()

    sdf = pd.DataFrame(sl)
    idx = sdf[sdf['exchangeShortName'] == 'JKT'].reset_index(drop=True)

    return idx


def get_all_sp500_stocks(api_key=None):
    if api_key is None:
        api_key = os.environ['FMP_API_KEY']

    url = f'https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={api_key}'
    r = requests.get(url)
    
    sp_df = pd.DataFrame(r.json())
    return sp_df


def get_company_profile(stocks, api_key=None):
    
    if api_key is None:
        api_key = os.environ['FMP_API_KEY']
    
    stock_param = ','.join(stocks)
    url = f'https://financialmodelingprep.com/api/v3/profile/{stock_param}?apikey={api_key}'
    r  = requests.get(url)
    cp = r.json()

    cp_df = pd.DataFrame(cp).set_index('symbol')
    return cp_df


def get_daily_stock_price(stock, api_key=None, n_days=365, start_from=None):
    if api_key is None:
        api_key = os.environ['FMP_API_KEY']

    if start_from is None:
        start_date = (datetime.datetime.today() - datetime.timedelta(days=n_days)).strftime('%Y-%m-%d')
    else:
        start_date = start_from
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


def get_dividend_history_single_stock(stock, api_key=None, source='fmp'):

    if source == 'fmp':
        return get_dividend_history_single_stock_fmp(stock, api_key)
    elif source == 'dag':
        return get_dividend_history_single_stock_dag(stock)
    else:
        raise ValueError("Invalid source. Currently only 'fmp' is supported.")


def get_dividend_history_single_stock_fmp(stock, api_key=None):
    """
    Downloads dividend history for a single stock from Financial Modeling Prep and returns it as a Pandas DataFrame.

    Args:
        stock (str): The stock symbol (e.g., "AAPL").
        api_key (str, optional): Your Financial Modeling Prep API key.
                                 If not provided, it will be read from the
                                 'FMP_API_KEY' environment variable. Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing the dividend history, indexed by date,
                          with columns 'dividend' and 'adjDividend'. Returns None
                          if no dividends are found or if an error occurs.
    """
    if api_key is None:
        api_key = os.environ['FMP_API_KEY']

    dividend_history_url = f'https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{stock}?apikey={api_key}'
    try:
        dr = requests.get(dividend_history_url)
        dr.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        drj = dr.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching dividend history for {stock}: {e}")
        return None

    if drj and 'historical' in drj:
        try:
            div_list = drj['historical']
            df = pd.DataFrame(div_list)
            if not df.empty:
                return df
            else:
                print(f"No historical dividend data found for {stock}.")
                return None
        except KeyError:
            print(f"No historical dividend data found for {stock}.")
            return None

    else:
        print(f"No data received for {stock}.")
        return None


def get_dividend_history_single_stock_dag(stock):

    url = f'https://raw.githubusercontent.com/mitbal/daguerreo-data/refs/heads/main/jkse/dividends/{stock[:4]}.csv'
    r = requests.get(url)
    if r.text != '404: Not Found':
        df = pd.read_csv(io.StringIO(r.text))
        df.rename(columns={'ex_date': 'date', 'dividend': 'adjDividend'}, inplace=True)
    else:
        df = None

    return df


def get_dividend_history(stocks, api_key=None):
    
    if api_key is None:
        api_key = os.environ['FMP_API_KEY']

    divs = []
    for i in range(int(len(stocks)/5)+1):
        stock_list = [s for s in stocks[i*5:(i+1)*5]]
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
    divs = {x: y for x, y in zip(div_df.index, div_df['historical'])}

    return divs


def get_financial_data(stock, period='quarter', api_key=None):

    if api_key is None:
        api_key = os.environ['FMP_API_KEY']

    url = f'https://financialmodelingprep.com/api/v3/income-statement/{stock}?period={period}&apikey={api_key}'
    r = requests.get(url)
    fs = r.json()
    return pd.DataFrame(fs)


def get_shares_outstanding(stock, api_key=None):

    if api_key is None:
        api_key = os.environ['FMP_API_KEY']

    url = f'https://financialmodelingprep.com/api/v4/shares_float?symbol={stock}&apikey={api_key}'
    r = requests.get(url)
    fs = r.json()
    return pd.DataFrame(fs)


def get_company_logo(stock, api_key=None):
    if api_key is None:
        api_key = os.environ['FMP_API_KEY']
    url = f'https://financialmodelingprep.com/image-stock/{stock}.png?apikey={api_key}'
    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))

    return img


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


def calc_div_score(df):

    score = (df['lastDiv'] + df['avgFlatAnnualDivIncrease']*5)/df['price'] \
            * ((df['numDividendYear']*2) / (25+df['numOfYear'])) \
            * ((df['positiveYear']*2 / (25+df['numOfYear']))) \
            * (1+(df['revenueGrowth']*0.5)*5)*df['lastDiv'] / 100
            
    return score
            

def calc_pe_history(price_df, fin_df, n_shares=None, currency='IDR', exchange_rate=16_276):

    pdf = price_df[['date', 'close']].copy()
    pdf['date'] = pd.to_datetime(pdf['date'])
    pdf = pdf.sort_values('date')

    if n_shares is not None:
        if currency == 'IDR':
            fin_df['converted_eps'] = fin_df.apply(lambda x: x['netIncome']/n_shares*exchange_rate if x['reportedCurrency'] == 'USD' else x['eps'], axis=1)
        else:
            fin_df['converted_eps'] = fin_df['eps']
    else:
        if currency == 'IDR':
            fin_df['converted_eps'] = fin_df.apply(lambda x: x['eps']*exchange_rate if x['reportedCurrency'] == 'USD' else x['eps'], axis=1)
        else:
            fin_df['converted_eps'] = fin_df['eps']

    edf = fin_df[['date', 'converted_eps']].sort_values(ascending=True, by='date').rolling(window=4, on='date').sum()
    edf['date'] = pd.to_datetime(edf['date'])
    edf = edf.sort_values('date')

    pe_df = pd.merge_asof(pdf, edf, on='date', direction='backward')
    pe_df['pe'] = pe_df['close'] / pe_df['converted_eps']

    return pe_df


def calc_pe_stats(pe_df):
    
    stats = {}

    last_date = pe_df['date'].max()
    periods = [2, 3, 10]

    for p in periods:
        start_date = last_date - datetime.timedelta(days=p*365)
        pe_period = pe_df[pe_df['date'] > start_date].reset_index(drop=True).copy()

        mean = pe_period['pe'].mean()
        ci = pe_period['pe'].quantile([.05, 0.95]).values

        stats[f'last_{p}y_mean'] = mean
        stats[f'last_{p}y_min_ci'] = ci[0]
        stats[f'last_{p}y_max_ci'] = ci[1]

    return stats


def calc_growth_stats(fin_df, metric='revenue'):

    rolling_metric = fin_df[['date', metric]].sort_values(ascending=True, by='date').rolling(window=4, on='date').sum()

    stats = {}
    periods = [2, 5, 10]
    for p in periods:
        growth = (rolling_metric[metric] / rolling_metric.shift(4)[metric])[::-1] -1
        growth = growth[:p*4].dropna()
        
        stats[f'median_{p}y_{metric}_growth'] = np.median(growth) *100
        stats[f'trim_mean_{p}y_{metric}_growth'] = (scipy.stats.trim_mean(growth, 0.1)) *100

    return stats


def calc_fin_stats(fin_df):

    stats = {}
    stats = stats | calc_growth_stats(fin_df, metric='revenue')
    stats = stats | calc_growth_stats(fin_df, metric='netIncome')

    rolling_revenue = fin_df[['date', 'revenue']].sort_values(ascending=True, by='date').rolling(window=4, on='date').sum()
    rolling_income = fin_df[['date', 'netIncome']].sort_values(ascending=True, by='date').rolling(window=4, on='date').sum()
    profit_margin = rolling_income['netIncome'] / rolling_revenue['revenue'] * 100
    stats['median_profit_margin'] = np.nanmedian(profit_margin)

    return stats


def calc_labels_stats(price_df, labels):
    
    buy = price_df[np.where(labels == 'buy', True, False)].copy()
    sell = price_df[np.where(labels == 'sell', True, False)].copy()

    buy['date'] = pd.to_datetime(buy['date'])
    sell['date'] = pd.to_datetime(sell['date'])
    sell['sell_date'] = sell['date']

    trades = pd.merge_asof(buy[['date', 'close']], sell[['date', 'close', 'sell_date']], on='date', direction='forward', suffixes=['_buy', '_sell'])
    trades['gain'] = (trades['close_sell'] / trades['close_buy'])*100 - 100
    trades['duration'] = trades['sell_date'] - trades['date']

    return trades


def calc_valuation(stock_list, price_dict, fin_dict):
    
    pe_dict = {}
    pe_stats_dict = {}
    rev_growth_dict = {}
    for stock in stock_list:
        try:
            price_df = price_dict[stock]
            fin_df = fin_dict[stock]

            pe_dict[stock] = calc_pe_history(price_df, fin_df)
            pe_stats_dict[stock] = calc_pe_stats(pe_dict[stock])

            rev_growth = calc_growth_stats(fin_dict[stock], 'revenue')
            rev_growth_dict[stock] = rev_growth

        except Exception as e:
            print(f'Error {e} data for {stock}')
            continue

        # if len(price_df) > 0 and len(fin_df) > 0:
    
    pe_stats_df = pd.DataFrame(pe_stats_dict).transpose()
    rev_growth_df = pd.DataFrame(rev_growth_dict).transpose()

    watchlist = pe_stats_df.copy()
    proc_stock_list = list(pe_dict.keys())
    watchlist['current_pe'] = [pe_dict[s]['pe'].values[-1] for s in proc_stock_list]
    watchlist['target'] = (watchlist['last_2y_mean'] / watchlist['current_pe']) - 1
    watchlist['risk'] = 1 - watchlist['last_2y_min_ci'] / watchlist['current_pe']
    watchlist['upside'] = watchlist['last_2y_max_ci'] / watchlist['current_pe'] - 1

    val_df = rev_growth_df.join(watchlist)
    return val_df


def prep_div_cal(div_dict, cp, year=2025):

    div_df = pd.DataFrame()
    for key, val in div_dict.items():
        if key in ['GGRP.JK', 'IKBI.JK', 'RCCC.JK']:
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

    div_df = merged[['date', 'symbol', 'adjDividend', 'price']].copy()
    div_df['date'] = div_df['date'].dt.strftime('%Y-%m-%d')

    return div_df


def prep_treemap(df, size_var='mktCap', color_var=None):

    if color_var is not None:
        yields = df[color_var]
        threshold = np.percentile(yields, [25, 50, 75])
        df['color_grad'] = pd.cut(df[color_var],
                                        bins=[-float('inf'), threshold[0], threshold[1], threshold[2], float('inf')],
                                        labels=[0, 33, 66, 100]).astype(float)

        sector_df = df.groupby('sector')[[size_var, 'color_grad']].sum()
        industry_df = df.groupby('industry')[[size_var, 'color_grad']].sum()
    
    else:
        sector_df = df.groupby('sector')[size_var].sum().to_frame()
        industry_df = df.groupby('industry')[size_var].sum().to_frame()

    map_sec_ind = df.groupby('sector')['industry'].apply(list).to_frame()
    map_ind_stock = df.reset_index().groupby('industry')['stock'].apply(list).to_frame()

    tree_data = []
    sectors = sector_df.index.to_list()
    for sector in sectors:
        children = []
        industries = set(map_sec_ind.loc[sector, 'industry'])
        for industry in industries:
            gc = []
            stocks = set(map_ind_stock.loc[industry, 'stock'])
            for stock in stocks:

                value = [df.loc[stock, size_var]]
                if color_var is not None:
                    value += [int(df.loc[stock, 'color_grad'])]

                path = sector+'/'+industry+'/'+stock
                gc += [{
                    'value': value,
                    'name': stock,
                    'path': path
                }]

            value = [industry_df.loc[industry, size_var]]
            if color_var is not None:
                value += [int(industry_df.loc[industry, 'color_grad'])]

            path = sector+'/'+industry
            children += [{
                'value': value,
                'name': industry,
                'path': path,
                'children': gc
            }]
        
        value = [sector_df.loc[sector, size_var]]
        if color_var is not None:
            value += [int(sector_df.loc[sector, 'color_grad'])]
        tree_data += [{
            'value': value,
            'name': sector,
            'path': sector,
            'children': children
        }]

    return tree_data


def simulate_simple_compounding(initial_value, num_year, avg_yield):
    
    investments = [initial_value]
    returns = []
    for i in range(num_year):
        returns += [int(investments[i] * avg_yield)]
        investments += [investments[i] + returns[i]]
    returns += [investments[-1] * avg_yield]

    return_df = pd.DataFrame({'investment': investments, 'returns': returns})[:num_year]
    return_df['year'] = [f'Year {i+1:02d}' for i in range(len(return_df))]

    return return_df


def simulate_dividend_compounding(
        stock_name,
        price_df,
        div_df,
        start_year=2014,
        end_year=2024,
        initial_investment=100,
        monthly_topup=0,
):
    
    cash = initial_investment
    num_stock = 0
    activities = []
    porto = []
    
    def buy_stock(cash, price_df, date):
        
        buy_date = price_df[price_df['date'] >= date].iloc[-1]
        price = buy_date['close']
        buy = cash / price / 100
        cash -= int(buy) * price * 100
        
        activities.append(f'{buy_date["date"]}: buy {int(buy)} lots of {stock_name} @ {price}')
        porto.append({'date': buy_date['date'], 'num_stock': int(buy), 'price': price})

        return int(buy), cash

    for y in range(start_year, end_year+1):

        if y == start_year:
            buy_lot, cash = buy_stock(cash, price_df, f'{y}-01-01')
            num_stock += buy_lot

        for m in range(1, 13):

            if y == start_year and m == 1:
                continue

            cash += monthly_topup

            buy_date = price_df[price_df['date'] >= f'{y}-{m:02d}-01'].iloc[-1]
            
            if div_df is None:
                div_date = None
            else:
                div_date = div_df[(div_df['date'] >= f'{y}-{m:02d}-01') & (div_df['date'] <= f'{y}-{m:02d}-31')]
            
            if div_df is None or len(div_date) == 0:
                # no dividend this month
                buy_lot, cash = buy_stock(cash, price_df, f'{y}-{m:02d}-01')
                num_stock += buy_lot

            elif buy_date['date'][0] < div_date['date'].iloc[0]:
                # buy first and then get dividend with the new number of stock
                buy_lot, cash = buy_stock(cash, price_df, f'{y}-{m:02d}-01')
                num_stock += buy_lot

                div_payment = div_date['adjDividend'].sum()
                div = int(div_payment * num_stock * 100)
                cash += div
                activities.append(f'{div_date["date"].iloc[0]} receive dividend {div_payment} for {num_stock}. Total {div}')
            
            else:
                # get dividend first and then buy with the new cash
                div_payment = div_date['adjDividend'].sum()
                div = int(div_payment * num_stock * 100)
                cash += div

                close_price = buy_date['close']
                buy_lot = cash / close_price / 100
                num_stock += int(buy_lot)
                cash -= int(buy_lot) * close_price * 100
                activities.append(f'buy {int(buy_lot)} lots of {stock_name} @ {close_price} at {buy_date["date"]}')

    return porto, activities
