import pickle

import pandas as pd
from tqdm import tqdm
import pandas_ta as ta

import harvest.data as hd


def compute_all():

    cp = pd.read_csv('data/company_profiles.csv')
    stock_list = cp[cp['isActivelyTrading']]['symbol'].tolist()

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
        stats['avg_daily_return'] = feat_df['daily_return'].mean()

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

if __name__ == '__main__':

    compute_all()
