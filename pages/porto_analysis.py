import yfinance as yf
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode

st.set_page_config(layout='wide')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file, delimiter=';', dtype='str')

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


# Perform dividend modelling and prediction for selected stock
try:
    if selection:
        symbol = f'{selection['selected_rows']['Symbol'].iloc[0]}'
        st.write('You select: ' +symbol)
except Exception:
    st.stop()

fig, ax = plt.subplots(figsize=(8, 3.5))

def fit_linear(divs, ax):
    
    X = np.arange(len(divs)).reshape(-1, 1)

    temp = divs.to_frame().reset_index()
    temp['year'] = temp['Date'].apply(lambda x: x.year)
    temp = temp.groupby('year')['Dividends'].sum().to_frame().reset_index()

    y = temp['Dividends'].to_numpy().reshape(-1, 1)
    X = temp['year'].to_numpy().reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(X, y)

    y_hat = lr.predict(X)
    temp['Prediction'] = y_hat
    sns.lineplot(data=temp, x='year', y='Dividends', ax=ax)
    sns.lineplot(data=temp, x='year', y='Prediction', ax=ax)

    return lr.score(X, y), lr.predict([[temp['year'].iloc[-1]+1]])[0][0]

score, pred = fit_linear(divs[symbol], ax)

col1, col2, col3 = st.columns([0.25, 0.4, 0.35])

with col1:
    AgGrid(divs[symbol][::-1].to_frame().reset_index(), height=200)

with col2:
    st.pyplot(fig)

with col3:
    st.write(f'R squared: {score:.2f}')
    st.write(f'Prediction for Next Year Dividend: {pred:.2f}')
    
    last = divs[symbol][-1]
    if pred > divs[symbol][-1]:
        color = 'green'
    else:
        color = 'red'

    st.write(f'Difference compared to the previous year: **:{color}[{pred-last:.2f}]**')
    st.write(f'Percentage difference compared to the previous year: **:{color}[{(pred-last)/last*100:.2f}%]**')
