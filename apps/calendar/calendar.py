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

sl = st.segmented_control(label='Stock List', 
                         options=['JKSE', 'S&P500'],
                         selection_mode='single',
                         default='JKSE')
    
if sl is None:
    print('Please select one of the options above')
    st.stop()

if sl == 'JKSE':
    exch = 'jkse'
else:
    exch = 'sp500'

div_cal_key = f'{exch}_div_cal'
div_score_key = f'{exch}_div_score'

url = os.environ['REDIS_URL']
r = redis.from_url(url)

@st.cache_data
def get_data_from_redis(key):
    j = r.get(key)
    return pd.DataFrame(json.loads(j))


df = get_data_from_redis(div_cal_key)
df['date'] = pd.to_datetime(df['date'])
df['yield'] = df['adjDividend'] / df['price'] * 100

div_score_df = get_data_from_redis(div_score_key)

if sl == 'JKSE':
    show_next_year = True
    div_score_df = div_score_df[['symbol', 'is_syariah']]
    is_syariah = st.toggle('Syariah Only?')
    if is_syariah:
        df = df.merge(div_score_df, on='symbol', how='left')
        df = df[df.is_syariah == True]
else:
    show_next_year = False

# st.dataframe(df)
cal = hp.plot_dividend_calendar(df, show_next_year=False, sl=sl)
st.altair_chart(cal)

idx = 1
for i in range(3):

    row = st.container()
    row_cols = row.columns(4)

    for j in range(4):

        month_df = df[df.date.dt.month == idx].copy().reset_index(drop=True)
        month_df.sort_values(by='yield', ascending=False, inplace=True)

        month_df['rank'] = list(range(1, len(month_df)+1))
        month_df['div_yield'] = month_df['yield'].apply(lambda x: f'{x:2.2f}%')
        month_df['ex_date'] = month_df['date'].dt.strftime('%d %b')
        if sl == 'JKSE':
            month_df['symbol'] = month_df['symbol'].apply(lambda x: x[:-3])
        month_df.rename(columns={'symbol': 'stock',
                                 'adjDividend': 'dividend'}, inplace=True)

        row_cols[j].write(f'{calendar.month_name[idx]}')
        row_cols[j].dataframe(hide_index=True, 
                              data=month_df[['rank', 'stock', 'ex_date', 'div_yield', 'dividend', 'price']], height=210)
        idx += 1
    