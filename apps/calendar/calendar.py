import os
import json
import time
import logging
import calendar

import redis
import lesley
import pandas as pd
import streamlit as st

import harvest.plot as hp
from harvest.utils import setup_logging


try:
    st.set_page_config(
        layout='wide',
        page_icon='📅',
        page_title='Dividend Calendar',
    )
except Exception as e:
    print('Set Page config has been called before')

st.title('Dividend Calendar 2025')

sl = st.sidebar.segmented_control(
    label='Stock List', 
    options=['JKSE', 'S&P500'],
    selection_mode='single',
    default='JKSE'
)

if sl is None:
    print('Please select one of the options above')
    st.stop()

if sl == 'JKSE':
    exch = 'jkse'
else:
    exch = 'sp500'

div_cal_key = f'div_cal_{exch}'
div_score_key = f'div_score_{exch}'

url = os.environ['REDIS_URL']

#### Function definition


@st.cache_resource
def get_logger(name, level=logging.INFO):

    logger = setup_logging(name, level)
    logger.info('a new user is opening the calendar page')
    return logger

logger = get_logger('calendar')


@st.cache_resource
def connect_redis(redis_url):
    r = redis.from_url(redis_url)
    return r

r = redis.from_url(url)


@st.cache_data(ttl=60*60, show_spinner='Downloading dividend data')
def get_data_from_redis(key):

    start = time.time()
    j = r.get(key)
    end = time.time()

    rjson = json.loads(j)
    last_updated = rjson['date']
    
    logger.info(f'get redis key: {key}, total time: {end-start:.4f} seconds, last updated: {last_updated}')

    content = rjson['content']
    return pd.DataFrame(json.loads(content))


def prep_div_month(df, month_idx=1):
    
    month_df = df[df.date.dt.month == month_idx].copy().reset_index(drop=True)
    month_df.sort_values(by='yield', ascending=False, inplace=True)

    month_df['rank'] = list(range(1, len(month_df)+1))
    month_df['div_yield'] = month_df['yield'].apply(lambda x: f'{x:2.2f}%')
    month_df['ex_date'] = month_df['date'].dt.strftime('%d %b')
    month_df['url_link'] = month_df['symbol'].apply(lambda x: f'https://panendividen.com/stock_picker?stock={x}')
    
    month_df.rename(columns={'symbol': 'stock',
                            'adjDividend': 'dividend'}, inplace=True)
    
    return month_df[['rank', 'url_link', 'ex_date', 'div_yield', 'dividend', 'price']]


#### End of Function definition

column_config = {
    'url_link': st.column_config.LinkColumn(
        'Stock',
        help='The code of the stock',
        max_chars=10,

        display_text=r"https://panendividen\.com/stock_picker\?stock=([A-Z]+)(?=\.JK|$)"
    ),
    'rank': st.column_config.NumberColumn(
        'Rank',
        help='Rank of the stock based on yield for this month',
        format='%d',
    ),
    'ex_date': st.column_config.TextColumn(
        'Ex Date',
        help='Ex-dividend date',
    ),
    'div_yield': st.column_config.TextColumn(
        'Yield',
        help='Dividend yield',
    ),
    'dividend': st.column_config.NumberColumn(
        'Dividend',
        help='Dividend amount',
        format='%.01f',
    ),
    'price': st.column_config.NumberColumn(
        'Price',
        help='Stock price',
        format='localized'
    ),
}


df = get_data_from_redis(div_cal_key)
df['date'] = pd.to_datetime(df['date'])
df['yield'] = df['adjDividend'] / df['price'] * 100

div_score_df = get_data_from_redis(div_score_key)

if sl == 'JKSE':
    show_next_year = True
    div_score_df = div_score_df[['symbol', 'is_syariah']]
    is_syariah = st.sidebar.toggle('Syariah Only?')
    if is_syariah:
        df = df.merge(div_score_df, on='symbol', how='left')
        df = df[df['is_syariah'] == True]
else:
    show_next_year = False

view_control = st.sidebar.segmented_control('Select View', ['Full Year', 'Single Month'], default='Full Year')
if view_control == 'Single Month':
    months = list(calendar.month_name)
    select_month = st.sidebar.selectbox('Select Month', months[1:])

if view_control == 'Full Year':
    cal = hp.plot_dividend_calendar(df, show_next_year=False, sl=sl)
    st.altair_chart(cal)

    idx = 1
    for i in range(3):

        row = st.container()
        row_cols = row.columns(4)

        for j in range(4):
            row_cols[j].write(f'{calendar.month_name[idx]}')

            month_df = prep_div_month(df, month_idx=idx)
            row_cols[j].dataframe(
                hide_index=True,
                column_config=column_config,
                data=month_df, height=210)
            idx += 1

else:
    st.write(select_month)
    cols = st.columns(2)
    month_index = months.index(select_month)

    labels = df.apply(lambda x: f"{x['date'].strftime('%d %b')}: {x['symbol']} ({x['yield']:2.2f}%)", axis=1)
    month_cal = lesley.month_plot(df['date'], df['yield'], labels=labels, month=month_index, show_date=True, width=500)
    cols[0].altair_chart(month_cal)

    month_df = prep_div_month(df, month_idx=month_index)
    cols[1].dataframe(
        hide_index=True,
        column_config=column_config,
        data=month_df
    )
