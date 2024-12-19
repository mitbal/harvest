import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import vectorbt as vbt

from sklearn.ensemble import GradientBoostingClassifier

def eval_all():

    with open('data/features.pkl', 'rb') as f:
        feat_dict = pickle.load(f)
    
    # with open('data/lgbm.pkl', 'rb') as f:
    with open('data/gbc.pkl', 'rb') as f:
        gbc = pickle.load(f)
    
    test_start_date = '2024-01-01'
    test_end_date = '2024-12-31'

    feat_stats = pd.read_csv('data/feat_stats.csv')
    stock_list = feat_stats[(feat_stats['avg_pe'] > 0) & (feat_stats['avg_value'] > 10_000_000_000)]['stock'].values.tolist()

    ret_dict = {}
    pred_dict = {}
    for stock in tqdm(stock_list):
        feat_df = feat_dict[stock]
        test_df = feat_df[test_start_date:test_end_date]
        X = test_df.loc[:, test_df.columns != 'flag'].replace([np.inf, -np.inf], np.nan, inplace=False).fillna(0)
        pred = gbc.predict(X)
        pred_dict[stock] = pred

        try:
            entry = (pred == -1)
            exits = (pred == 1)
            pf = vbt.Portfolio.from_signals(test_df['close'], entry, exits, freq='d')
            ret = pf.stats()
            ret_dict[stock] = ret
            print(stock, ret['Total Return [%]'])
        except Exception as e:
            print(f'error {stock}: {e}')
    
    with open('data/return_gbm.pkl', 'wb') as f:
        pickle.dump(ret_dict, f)

if __name__ == '__main__':

    eval_all()
