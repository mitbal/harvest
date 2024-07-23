import yfinance as yf
import streamlit as st
from st_aggrid import AgGrid

st.set_page_config(layout='wide')

import pandas as pd

import locale
# locale.setlocale()
# locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

df = pd.read_csv('porto.csv', delimiter=';')

st.write('Current Portfolio')
# st.dataframe(df)

@st.cache_data
def enrich_data(porto):

    divs = {}
    drs = {}
    prices = {}
    for s in porto['Symbol']:
        t = yf.Ticker(s+'.JK')

        try:
            drs[s] = t.info['dividendRate']
            divs[s] = t.get_dividends()
            prices[s] = t.info['previousClose']
        except Exception as e:
            print(s, e) 
    
    df = porto.merge(pd.DataFrame({'Symbol': drs.keys(), 'div_rate': drs.values()})).\
        merge(pd.DataFrame({'Symbol': prices.keys(), 'last_price': prices.values()}))

    return df, divs


# get realtime stock data from yahoo finance


df_display = pd.DataFrame()

df, divs = enrich_data(df)

df['current_lot'] = df['Available Lot'].apply(lambda x: x.replace(',', '')).astype(float)
df['avg_price'] = df['Average Price'].apply(lambda x: x.replace(',', '')).astype(float)
df['total_invested'] = df['current_lot'] * df['avg_price']
df['yield_on_cost'] = df['div_rate'] / df['avg_price']

AgGrid(df)


# from st_aggrid import AgGrid
# import pandas as pd

# df = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv')
# AgGrid(df)
