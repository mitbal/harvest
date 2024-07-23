import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, datetime

st.set_page_config(layout='wide')

st.title('Dividend Harvesting')

with st.sidebar:
    start_date = st.date_input('Select start date', date(2024, 1, 1))

    end_date = st.date_input('Select end date', date(2024, 12, 31))

    target = st.text_input('Target', 100_000_000)

df = pd.read_csv('stockbit.csv', delimiter=';')

col1, col2 = st.columns(2)

total_dividend = str(df['Total Dividend'].sum())
percentage = float(total_dividend) / float(target) * 100

with col1:
    st.metric(label='Total Dividend', value=total_dividend)

with col2:
    st.metric(label='Percent from Target', value=percentage)


st.write('Last 10 dividend')
st.dataframe(df[:10])

col1, col2 = st.columns(2)

top10 = df.groupby('Stock')['Total Dividend'].sum().sort_values(ascending=False)[:10]
number = df.groupby('Stock')['Date'].count()

with col1:

    fig, ax = plt.subplots()
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


st.image('test2.gif')
