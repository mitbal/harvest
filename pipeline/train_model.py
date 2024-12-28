import time
import tomllib

import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def train_all():

    with open('pipeline/training_config.toml', 'rb') as f:
        config = tomllib.load(f)

    with open('data/features.pkl', 'rb') as f:
        feat_dict = pickle.load(f)

    feat_stats = pd.read_csv('data/feat_stats.csv')
    stock_list = feat_stats[(feat_stats['avg_pe'] > 0) & (feat_stats['avg_value'] > 10_000_000_000)]['stock'].values.tolist()

    X, y, weights = prep_train(feat_dict, stock_list, config)

    # models_dict = config['models']
    for m in config['models']:
        if m['type'] == 'XGBClassifier':
            model = XGBClassifier(**m['hyperparameters'])
        elif m['type'] == 'LGBMClassifier':
            model = LGBMClassifier(**m['hyperparameters'])
        model = train_single(model, X, y, weights)
        with open(f'data/{m['type']}.pkl', 'wb') as f:
            pickle.dump(model, f)


def prep_train(feat_dict, stock_list, config):

    full_train = pd.DataFrame()
    for stock in stock_list:
        feat_df = feat_dict[stock]

        train_df = feat_df[config['training']['start_date']:config['training']['end_date']]
        full_train = pd.concat([full_train, train_df])

    X = full_train.loc[:, config['features']].replace([np.inf, -np.inf], np.nan, inplace=False).fillna(0)
    y = full_train[config['label']]
    y_encoded = y.map({'buy': 0, 'hold': 1, 'sell': 2})

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

    train_all()
