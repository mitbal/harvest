import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import vectorbt as vbt

from sklearn.ensemble import GradientBoostingClassifier

import harvest.data as hd

def eval_all():

    with open('data/features.pkl', 'rb') as f:
        feat_dict = pickle.load(f)
    
    model_dict = {
        'lgbm': 'data/lgbm.pkl',
        'xgb': 'data/xgb.pkl'
    }

    test_start_date = '2024-01-01'
    test_end_date = '2024-12-31'
    feat_stats_df = pd.read_csv('data/feat_stats.csv')
    val_df = pd.read_csv('data/valuation.csv')

    att_df = feat_stats_df.merge(val_df, on='stock')
    stock_list = att_df[(att_df['avg_value'] > 10_000_000_000) & (att_df['current_pe'] > 0) & (att_df['target'] > 0.03) & (att_df['current_pe'] < 30)]['stock'].values.tolist()

    for model_name, model_path in model_dict.items():

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        ret_dict = eval_single_model(model, feat_dict, stock_list, test_start_date, test_end_date)
        ret_df = pd.DataFrame(ret_dict).transpose()
        print(f'{model_name} total_return: {ret_df["Total Return [%]"].sum()}')
        print(f'{model_name} real alpha: {ret_df["real_alpha"].sum()}')
        print(f'{model_name} ideal alpha diff: {ret_df["ideal_alpha_diff"].sum()}')
        
        with open(f'data/return_{model_name}.pkl', 'wb') as f:
            pickle.dump(ret_dict, f)


def eval_single_model(model, feat_dict, stock_list, start_date, end_date):
    print(f'evaluating {model}...')

    ret_dict = {}
    for stock in tqdm(stock_list):
        feat_df = feat_dict[stock]
        test_df = feat_df[start_date:end_date]
        
        stats = eval_single_stock(model, test_df)
        ret_dict[stock] = stats
    return ret_dict


def eval_single_stock(model, test_df):

    X = test_df.loc[:, test_df.columns != 'flag'].replace([np.inf, -np.inf], np.nan, inplace=False).fillna(0)
    pred = model.predict(X) - 1

    entry = (pred == -1)
    exits = (pred == 1)
    model_stats = hd.calc_return(test_df['close'], entry, exits)
    model_stats['real_alpha'] = model_stats['Total Return [%]'] - model_stats['Benchmark Return [%]']

    entry = (test_df['flag'] == -1)
    exits = (test_df['flag'] == 1)
    ideal_stats = hd.calc_return(test_df['close'], entry, exits)

    model_stats['ideal_alpha_diff'] = ideal_stats['Total Return [%]'] - model_stats['Total Return [%]']
    return model_stats


if __name__ == '__main__':

    eval_all()
