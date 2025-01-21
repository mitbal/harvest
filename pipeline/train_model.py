import time
import tomllib

import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


def train_all(exch='jkse', model_codename='oslo'):

    with open(f'pipeline/{model_codename}_config.toml', 'rb') as f:
        config = tomllib.load(f)

    with open(f'data/{exch}/features.pkl', 'rb') as f:
        feat_dict = pickle.load(f)

    feat_stats = pd.read_csv(f'data/{exch}/feat_stats.csv')
    stock_list = feat_stats[(feat_stats['avg_pe'] > 0) & (feat_stats['avg_value'] > 10_000_000_000)]['stock'].values.tolist()

    X, y, weights = prep_train(feat_dict, stock_list, config)

    # models_dict = config['models']
    for m in config['models']:
        if m['type'] == 'XGBClassifier':
            model = XGBClassifier(**m['hyperparameters'])
        elif m['type'] == 'XGBRegressor':
            model = XGBRegressor(**m['hyperparameters'])
        elif m['type'] == 'LGBMClassifier':
            model = LGBMClassifier(**m['hyperparameters'])
        elif m['type'] == 'LGBMRegressor':
            model = LGBMRegressor(**m['hyperparameters'])
        model = train_single(model, X, y, weights)
        with open(f'data/{exch}/{model_codename}_{m['type']}.pkl', 'wb') as f:
            pickle.dump(model, f)


def prep_train(feat_dict, stock_list, config):

    full_train = pd.DataFrame()
    for stock in stock_list:
        feat_df = feat_dict[stock]

        train_df = feat_df[config['training']['start_date']:config['training']['end_date']]
        full_train = pd.concat([full_train, train_df])

    X = full_train.loc[:, config['features']].replace([np.inf, -np.inf], np.nan, inplace=False).fillna(0)
    y = full_train[config['label']]
    if config['type'] != 'regression':
        y_encoded = y.map({'buy': 0, 'hold': 1, 'sell': 2})
    else:
        y_encoded = y

    # assign weight to each sample
    def get_weight(x):
        return config['pos_weight'] if x in ['buy', 'sell'] else 1
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

    train_all(exch='jkse', model_codename='oslo')
    train_all(exch='jkse', model_codename='stockholm')
    train_all(exch='jkse', model_codename='copenhagen')
    # train_all(exch='sp500')
