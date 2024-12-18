import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

def train_all():

    train_start_date = '2015-01-01'
    train_end_date = '2022-12-31'

    val_start_date = '2023-01-01'
    val_end_date = '2023-12-31'

    full_train = pd.DataFrame()

    with open('data/features.pkl', 'rb') as f:
        feat_dict = pickle.load(f)

    for stock in feat_dict.keys():
        feat_df = feat_dict[stock]
        train_df = feat_df[train_start_date:train_end_date]
        full_train = pd.concat([full_train, train_df])

    gbc = GradientBoostingClassifier()

    X = full_train.loc[:, full_train.columns != 'flag'].replace([np.inf, -np.inf], np.nan, inplace=False).fillna(0)
    y = full_train['flag']

    def get_weight(x):
        return 20 if x in [-1, 1] else 1
    weights = [get_weight(x) for x in y]

    gbc.fit(X, y, sample_weight=weights)

    with open('data/gbc.pkl', 'wb') as f:
        pickle.dump(gbc, f)

if __name__ == '__main__':

    train_all()
