from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import pandas_ta as ta

import harvest.data as hd

@st.cache_data
def get_stock_list():

    idf = hd.get_all_idx_stocks()
    df = hd.get_company_profile(idf['symbol'].values.tolist())
    return df


@st.cache_data
def get_valuation(df):

    stock_list = df.index.values.tolist()
    val_dict = {}
    
    for stock in stock_list:
        print(stock)
        start_date = datetime.now() - timedelta(days=365*1)
        price_df = hd.get_daily_stock_price(stock, start_from=start_date)
        fin = hd.get_financial_data(stock)
        n_share = hd.get_shares_outstanding(stock)['outstandingShares'].tolist()[0]
        # currency = fin['currency'].values[-1]
        currency = 'IDR'

        pe_df = hd.calc_pe_history(price_df, fin, n_shares=n_share, currency=currency)
        pe_ttm = pe_df['pe'].values[-1]
        current_price = price_df['close'].values[0]
        median_pe = pe_df['pe'].median()

        val_dict[stock] = {
            'target_val': median_pe,
            'target_price': (median_pe/pe_ttm)*current_price,
            'current_price': current_price,
            'current_val': pe_ttm,
            'pct_diff': (median_pe / pe_ttm) -1
        }
        
        # df.loc[stock, 'valuation'] = hd.get_valuation(stock)
    return val_dict


# def get_technical_indicator(df):

#     stock_list = df.index.values.tolist()
#     val_dict = {}
    
#     for stock in stock_list:
#         print(stock)
#         start_date = datetime.now() - timedelta(days=365*1)
#         price_df = hd.get_daily_stock_price(stock, start_from=start_date)

#         rsi = price_df.ta.rsi().to_frame()



df = get_stock_list()
df['div_yield'] = df['lastDiv'] / df['price']

short_list = df[(df['div_yield'] > 0.05)].sort_values('mktCap', ascending=False)

st.dataframe(short_list)
# st.write(len(short_list))
val_df = pd.DataFrame(get_valuation(short_list[:100])).T
st.write(val_df)

stock_name = st.text_input('Enter Stock', key='stock', value='BBCA.JK')

start_date = datetime.now() - timedelta(days=365*1)
price_df = hd.get_daily_stock_price(stock_name, start_from=start_date)

rsi = price_df.ta.rsi().to_frame()

st.dataframe(rsi)
