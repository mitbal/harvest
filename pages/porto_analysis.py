import yfinance as yf
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode

st.set_page_config(layout='wide')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('porto.csv', delimiter=';')

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


df_display = pd.DataFrame()

# get realtime stock data from yahoo finance
df, divs = enrich_data(df)

df['current_lot'] = df['Available Lot'].apply(lambda x: x.replace(',', '')).astype(float)
df['avg_price'] = df['Average Price'].apply(lambda x: x.replace(',', '')).astype(float)
df['total_invested'] = df['current_lot'] * df['avg_price'] * 100
df['yield_on_cost'] = df['div_rate'] / df['avg_price']

annual_dividend = (df['current_lot']*df['div_rate']).sum() * 100
total_yield_on_cost = annual_dividend / df['total_invested'].sum() * 100

col1, col2 = st.columns(2)
with col1:
    st.metric('Total Dividend Yield on Cost', value=f'{total_yield_on_cost:.2f} %')
with col2:
    st.metric('Dividend Annual Income', value=f'IDR {annual_dividend:,.2f}')

st.write('Current Portfolio')

builder = GridOptionsBuilder.from_dataframe(df)
builder.configure_pagination(enabled=True)
builder.configure_selection(selection_mode='single', use_checkbox=False)
builder.configure_column('Symbol', editable=False)
grid_options = builder.build()

selection = AgGrid(df, gridOptions=grid_options, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)
