import os
import json
import calendar

import redis
import pandas as pd
import streamlit as st

import harvest.plot as hp


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
r = redis.from_url(url)

@st.cache_data(ttl=60*60)
def get_data_from_redis(key):
    j = r.get(key)

    rjson = json.loads(j)
    last_updated = rjson['date']
    print(f'Last Updated: {last_updated}')
    content = rjson['content']
    return pd.DataFrame(json.loads(content))


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
        month_df['url_link'] = month_df['symbol'].apply(lambda x: f'https://panendividen.com/stock_picker?stock={x}')
        if sl == 'JKSE':
            month_df['symbol'] = month_df['symbol'].apply(lambda x: x[:-3])
        month_df.rename(columns={'symbol': 'stock',
                                 'adjDividend': 'dividend'}, inplace=True)

        row_cols[j].write(f'{calendar.month_name[idx]}')
        row_cols[j].dataframe(hide_index=True,
                              column_config={
                                "url_link": st.column_config.LinkColumn(
                                    "Stock",
                                    help="Stock Name",
                                    max_chars=10,
                                    display_text=r"https://panendividen\.com/stock_picker\?stock=([A-Z]+(?:\.JK)?)"
                                ),},
                              data=month_df[['rank', 'url_link', 'ex_date', 'div_yield', 'dividend', 'price']], height=210)
        idx += 1
    