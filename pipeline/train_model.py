import time

import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def train_all():

    train_start_date = '2015-01-01'
    train_end_date = '2022-12-31'

    val_start_date = '2023-01-01'
    val_end_date = '2023-12-31'

    with open('data/features.pkl', 'rb') as f:
        feat_dict = pickle.load(f)

    feat_stats = pd.read_csv('data/feat_stats.csv')
    stock_list = feat_stats[(feat_stats['avg_pe'] > 0) & (feat_stats['avg_value'] > 10_000_000_000)]['stock'].values.tolist()

    X, y, weights = prep_train(feat_dict, stock_list, train_start_date, train_end_date)

    models_dict = {
        'xgb': XGBClassifier(
            learning_rate=0.1,
            n_estimators=1000,
            max_depth=5,
            min_child_weight=2),
        'lgbm': LGBMClassifier(
            num_iterations=1000
        )
    }
    for model_name, model in models_dict.items():
        model = train_single(model, X, y, weights)
        with open(f'data/{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)


def prep_train(feat_dict, stock_list, train_start_date, train_end_date):

    full_train = pd.DataFrame()
    for stock in stock_list:
        feat_df = feat_dict[stock]

        train_df = feat_df[train_start_date:train_end_date]
        full_train = pd.concat([full_train, train_df])

    X = full_train.loc[:, ~full_train.columns.isin(['trade_signal', 'daily_return'])].replace([np.inf, -np.inf], np.nan, inplace=False).fillna(0)
    y = full_train['trade_signal']
    y_encoded = y.map({'buy': 0, 'sell': 2, 'hold': 1})

    # assign weight to each sample
    def get_weight(x):
        return 25 if x in ['buy', 'sell'] else 1
    weights = [get_weight(x) for x in y]

    return X, y_encoded, weights


def train_single(model, X, y, sample_weights):
    print(f'Start training {model}')
    start_time = time.time()
    model.fit(X, y, sample_weight=sample_weights)
    end_time = time.time()

    print(f'Total training time: {end_time - start_time}')

    return model

if __name__ == '__main__':

    train_all()
