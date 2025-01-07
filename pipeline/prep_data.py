import os
import json
import pickle
from datetime import datetime

import redis
import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas_ta as ta

import harvest.data as hd


def compute_all():

    cp = pd.read_csv('data/company_profiles.csv')
    stock_list = cp[cp['isActivelyTrading']]['symbol'].tolist()
    cp.set_index('symbol', inplace=True)

    with open('data/prices.pkl', 'rb') as f:
        price_dict = pickle.load(f)
    
    with open('data/financials.pkl', 'rb') as f:
        fin_dict = pickle.load(f)

    feat_dict = compute_features(stock_list, price_dict, fin_dict)
    labels_dict = compute_labels(stock_list, price_dict)
    
    combined_dict = {}
    for stock in stock_list:
        try:
            combined_df = feat_dict[stock].join(labels_dict[stock])
            combined_dict[stock] = combined_df 
        except:
            print(f'error {stock}')   
    with open('data/features.pkl', 'wb') as f:
        pickle.dump(combined_dict, f)

    val_table = compute_valuation(stock_list, price_dict, fin_dict)
    val_table.to_csv('data/valuation.csv', index_label='stock')

    feat_stats = compute_feature_stats(combined_dict)
    feat_stats.to_csv('data/feat_stats.csv', index_label='stock')

    with open('data/dividends.pkl', 'rb') as f:
        div_dict = pickle.load(f)

    div_stats = compute_div_score(cp, fin_dict, div_dict)
    div_stats.to_csv('data/div_score.csv', index_label='stock')
    store_df_to_redis('div_score', div_stats)


def store_df_to_redis(key, df):

    url = os.environ['REDIS_URL']
    r = redis.from_url(url)
    df_json = json.dumps(df.reset_index().to_dict(orient='records'))
    r.set(key, df_json)


def compute_features(stock_list, price_dict, fin_dict):

    feat_dict = {}
    for stock in tqdm(stock_list):
        print(f'processing {stock}')
        prices = price_dict[stock][['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True).set_index('date')

        # calculate technical feature
        try:
            rsi = prices.ta.rsi().to_frame()
            super_trend = prices.ta.supertrend()
            bbands = prices.ta.bbands()
            stochrsi = prices.ta.stochrsi()

            pe_history = hd.calc_pe_history(prices.reset_index(), fin_dict[stock])
            pe_history = pe_history.set_index('date')['pe'].to_frame()
            pe_history.index = pe_history.index.astype('str')

            feat_dict[stock] = rsi.join(super_trend).join(bbands).join(stochrsi).join(prices).join(pe_history)
        
        except Exception as e:
            print(f'error {stock}: {e}')
    
    return feat_dict


def compute_feature_stats(feat_dict):

    stats_dict = {}
    stock_list = list(feat_dict.keys())
    for stock in stock_list:
        feat_df = feat_dict[stock]

        stats = {}
        stats['count_entry_labels'] = (feat_df['trade_signal'] == 'buy').sum()
        stats['avg_value'] = (feat_df['volume']*feat_df['close']).mean()
        stats['avg_pe'] = feat_df['pe'].mean()
        stats['avg_daily_return'] = feat_df['return_1d'].mean()

        stats_dict[stock] = stats
    
    return pd.DataFrame(stats_dict).transpose()
    

def compute_labels(stock_list, price_dict):
    
    labels_dict = {}
    for stock in tqdm(stock_list):
        print(f'processing {stock}')
        prices = price_dict[stock][['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True).set_index('date')

        # buy sell flag based on local minima and maxima
        labels = hd.make_labels(prices['close'], threshold=0.15)
        label_df = prices[['close']].copy()
        label_df['trade_signal'] = labels

        # day trading signal, buy today at close price and sell tomorrow
        label_df['return_1d'] = label_df['close'].pct_change(-1).shift(-1) * 100
        label_df['return_5d'] = label_df['close'].pct_change(-5).shift(-5) * 100
        label_df['return_20d'] = label_df['close'].pct_change(-20).shift(-20) * 100

        labels_dict[stock] = label_df[['trade_signal', 'return_1d', 'return_5d', 'return_20d']]

    return labels_dict


def compute_valuation(stock_list, price_dict, fin_dict):

    val_table = hd.calc_valuation(stock_list, price_dict, fin_dict)
    return val_table


def compute_div_score(cp_df, fin_dict, div_dict):
    
    df = cp_df[cp_df['isActivelyTrading']].copy()
    df['yield'] = df['lastDiv'] / df['price'] * 100

    stock_list = df.index.tolist()
    for symbol in stock_list:

        if df.loc[symbol, 'lastDiv'] == 0:
            continue

        try:
            fin_df = fin_dict[symbol]
            fin_stats = hd.calc_fin_stats(fin_df)
            df.loc[symbol, 'revenueGrowth'] = fin_stats['mean_10y_revenue_growth']
            df.loc[symbol, 'netIncomeGrowth'] = fin_stats['mean_10y_netIncome_growth']
                    
            div_df = pd.DataFrame(div_dict[symbol])
            div_df = hd.preprocess_div(div_df)
            div_stats = hd.calc_div_stats(div_df)
            
            df.loc[symbol, 'avgFlatAnnualDivIncrease'] = div_stats['historical_mean_flat']
            df.loc[symbol, 'avgPctAnnualDivIncrease'] = div_stats['historical_mean_pct']
            df.loc[symbol, 'numDividendYear'] = div_stats['num_dividend_year']
            df.loc[symbol, 'positiveYear'] = div_stats['num_positive_year']
            df.loc[symbol, 'numOfYear'] = datetime.today().year - datetime.strptime(df.loc[symbol, 'ipoDate'], '%Y-%m-%d').year

        except Exception as e:
            print('error', symbol)
            continue
    
    # patented dividend score
    df['DScore'] = hd.calc_div_score(df)

    return df[['price', 'lastDiv', 'yield', 'sector', 'industry', 'mktCap', 'ipoDate',
               'revenueGrowth', 'netIncomeGrowth', 
               'avgFlatAnnualDivIncrease', 'avgPctAnnualDivIncrease', 'numDividendYear', 'DScore']]


if __name__ == '__main__':

    compute_all()
