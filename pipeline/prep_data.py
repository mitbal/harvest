import pickle
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm

import harvest.data as hd

def calc_all():

    cp = pd.read_csv('data/company_profiles.csv')
    stock_list = cp[cp['isActivelyTrading']]['symbol'].tolist()

    with open('data/prices.pkl', 'rb') as f:
        price_dict = pickle.load(f)
    
    with open('data/financials.pkl', 'rb') as f:
        fin_dict = pickle.load(f)

    feat_dict = {}
    for stock in tqdm(stock_list):
        print(f'processing {stock}')
        prices = price_dict[stock][['date', 'open', 'high', 'low', 'close']].sort_values('date').reset_index(drop=True).set_index('date')

        # calculate technical feature
        try:
            rsi = prices.ta.rsi().to_frame()
            super_trend = prices.ta.supertrend()
            bbands = prices.ta.bbands()
            stochrsi = prices.ta.stochrsi()

            pe_history = hd.calc_pe_history(prices.reset_index(), fin_dict[stock])
            pe_history = pe_history.set_index('date')['pe'].to_frame()
            pe_history.index = pe_history.index.astype('str')

            labels = hd.make_labels(prices['close'], threshold=0.1)
            abc = labels.to_frame()
            abc.columns = ['flag']

            feat_dict[stock] = rsi.join(super_trend).join(bbands).join(stochrsi).join(prices).join(pe_history).join(abc)
        except Exception as e:
            print(f'error {stock}: {e}')

    with open('data/features.pkl', 'wb') as f:
        pickle.dump(feat_dict, f)

if __name__ == '__main__':

    calc_all()
