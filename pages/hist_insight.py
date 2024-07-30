from datetime import date, datetime

import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from st_aggrid import AgGrid

st.set_page_config(layout='wide')

st.title('Historical Insight')

uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is None:
    st.stop()
df = pd.read_csv(uploaded_file, delimiter=';')


## First section, overall 
con = st.container(border=True)

with con:
    col1, col2 = st.columns(2)

    with col1:
        total_dividend = f"IDR {df['Total Dividend'].sum():,}"
        st.metric(label='Total Dividend', value=total_dividend)

        cola, colb = st.columns(2)
        with cola:
            num_transaction = len(df)
            st.metric(label='Number of Transaction', value=num_transaction)

        with colb:
            num_stocks = len(df['Stock'].unique())
            st.metric(label='Number of Stocks', value=num_stocks)
        
        cola, colb = st.columns(2)
        with cola:
            first_date = df['Date'].values[-1]
            st.metric(label='First Transaction Date', value=first_date)

        with colb:
            last_date = df['Date'][0]
            st.metric(label='Last Transaction Date', value=last_date)

        duration = datetime.strptime(last_date, '%Y-%m-%d') - datetime.strptime(first_date, '%Y-%m-%d')
        st.metric(label='Total Duration', value=str(duration))

    with col2:
        st.write('List of transactions')
        AgGrid(df, height=300)


## Second section, summarization

col1, col2 = st.columns(2)

top10 = df.groupby('Stock')['Total Dividend'].sum().sort_values(ascending=False)[:10]
number = df.groupby('Stock')['Date'].count()

with col1:

    fig, ax = plt.subplots(figsize=(2, 2))
    top10.plot(kind='pie', ax=ax)

    st.write('Top 10 dividend by Stock')
    st.pyplot(fig)

with col2:
    st.dataframe(pd.DataFrame(top10).join(number, on='Stock'))

import july

st.write('Dividend Calendar')

counts = df.groupby('Date')['Stock'].count().to_frame().reset_index()
ax = july.heatmap(counts['Date'], counts['Stock'])

st.pyplot(ax.get_figure())

st.write('Sector Analysis')

col1, col2 = st.columns(2)

import yfinance as yf

sectors = {}
for s in df['Stock'].unique():
    ticker = yf.Ticker(s+'.JK')
    sectors[s] = ticker.info['sector']

df_sectors = pd.DataFrame.from_dict({'stock': sectors.keys(), 'sector': sectors.values()})
df = df.merge(df_sectors, left_on='Stock', right_on='stock')
agg_sector = df.groupby('sector')['Total Dividend'].sum().sort_values(ascending=False)

with col1:
    st.dataframe(agg_sector)

with col2:
    fig2, ax2 = plt.subplots()
    agg_sector.plot(kind='pie', ax=ax2)
    st.pyplot(fig2)

# Single stock analysis
stock = st.text_input('Select Stock', value='ADRO')

history = yf.Ticker(stock+'.JK').history(period='3mo').reset_index()

import plotly.graph_objects as go
candlestick = go.Candlestick(
                            x=history['Date'],
                            open=history['Open'],
                            high=history['High'],
                            low=history['Low'],
                            close=history['Close']
                            )

fig = go.Figure(data=[candlestick])

st.plotly_chart(fig)
