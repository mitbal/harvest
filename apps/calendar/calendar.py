import os
import json
import pickle
import calendar

import redis
import pandas as pd
import streamlit as st

import harvest.plot as hp
import harvest.data as hd

st.title('Dividend Calendar 2025')

url = os.environ['REDIS_URL']
r = redis.from_url(url)
j = r.get('div_hist_2024')
df = pd.DataFrame(json.loads(j))
df['date'] = pd.to_datetime(df['date'])
df['yield'] = df['adjDividend'] / df['price'] * 100

cal = hp.plot_dividend_calendar(df, show_next_year=True)
st.altair_chart(cal)

idx = 1
for i in range(3):

    row = st.container()
    row_cols = row.columns(4)

    for j in range(4):

        month_df = df[df.date.dt.month == idx].copy().reset_index(drop=True)
        month_df.sort_values(by='yield', ascending=False, inplace=True)
        month_df.reset_index(drop=True, inplace=True)
        month_df.set_index('symbol', inplace=True)
        month_df['ranking'] = list(range(len(month_df)))
        month_df['div_yield'] = month_df['yield'].apply(lambda x: f'{x:2.2f}%')
        month_df['ex_date'] = month_df['date'].dt.strftime('%b %d')

        row_cols[j].write(f'{calendar.month_name[idx]}')
        row_cols[j].dataframe(month_df[['ranking', 'ex_date', 'adjDividend', 'price', 'div_yield']], height=210)
        idx += 1
    