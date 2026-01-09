import os
import json
import time
import logging
import calendar
from datetime import datetime

import redis
import lesley
import pandas as pd
import streamlit as st
import altair as alt

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

current_year = datetime.today().year

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

div_years_key = f'div_cal_years_{exch}'
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
    if j is None:
        logger.error(f'Missing redis key: {key}')
        st.error('Dividend calendar data is not available. Please refresh after the pipeline finishes.')
        st.stop()
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


years_df = get_data_from_redis(div_years_key)
available_years = sorted(years_df['year'].astype(int).unique().tolist())
if len(available_years) == 0:
    available_years = [current_year]
default_year = current_year if current_year in available_years else max(available_years)
selected_year = st.sidebar.selectbox('Year', available_years, index=available_years.index(default_year))

div_cal_key = f'div_cal_{exch}_{selected_year}'
df = get_data_from_redis(div_cal_key)
df['date'] = pd.to_datetime(df['date'])
df['yield'] = df['adjDividend'] / df['price'] * 100

st.title(f'Dividend Calendar {selected_year+1}')
st.write(f'*based on {selected_year} data*')

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
month_index = None
if view_control == 'Single Month':
    months = list(calendar.month_name)
    select_month = st.sidebar.selectbox('Select Month', months[1:])
    month_index = months.index(select_month)

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

    labels = df.apply(lambda x: f"{x['date'].strftime('%d %b')}: {x['symbol']} ({x['yield']:2.2f}%)", axis=1)
    month_cal = lesley.month_plot(df['date'], df['yield'], labels=labels, month=month_index, show_date=True, width=500)
    cols[0].altair_chart(month_cal)

    month_df = prep_div_month(df, month_idx=month_index)
    cols[1].dataframe(
        hide_index=True,
        column_config=column_config,
        data=month_df
    )

stats_df = df if view_control == 'Full Year' else df[df['date'].dt.month == month_index]
if stats_df.empty:
    st.info('No dividend statistics available for this selection.')
else:
    st.divider()
    st.subheader('Dividend Statistics')

    c1, c2, c3 = st.columns(3)
    c1.metric('Events', len(stats_df))
    c2.metric('Avg Yield', f"{stats_df['yield'].mean():.2f}%")
    c3.metric('Median Yield', f"{stats_df['yield'].median():.2f}%")

    top_row = stats_df.loc[stats_df['yield'].idxmax()]
    st.caption(f"Highest yield: {top_row['symbol']} on {top_row['date'].strftime('%d %b')} at {top_row['yield']:.2f}% (dividend {top_row['adjDividend']:.2f}).")

    charts = st.columns(2)

    if view_control == 'Full Year':
        monthly_summary = (
            stats_df
            .assign(month=stats_df['date'].dt.month)
            .groupby('month')['yield']
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'avg_yield', 'count': 'events'})
        )
        month_order = list(calendar.month_abbr[1:])
        monthly_summary['month_name'] = monthly_summary['month'].apply(lambda m: calendar.month_abbr[m])
        monthly_summary['month_name'] = pd.Categorical(monthly_summary['month_name'], categories=month_order, ordered=True)
        monthly_summary = monthly_summary.sort_values('month_name')

        combo_base = alt.Chart(monthly_summary)
        events_bar = combo_base.mark_bar(color='#1f77b4').encode(
            x=alt.X('month_name:N', sort=month_order, title='Month'),
            y=alt.Y('events:Q', title='Dividend count'),
            tooltip=[alt.Tooltip('month_name:N', title='Month'), alt.Tooltip('events:Q', title='Count')]
        )
        yield_line = combo_base.mark_line(color='#ff7f0e', point=True).encode(
            x=alt.X('month_name:N', sort=month_order),
            y=alt.Y('avg_yield:Q', title='Avg yield (%)', axis=alt.Axis(titleColor='#ff7f0e'), scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('month_name:N', title='Month'), alt.Tooltip('avg_yield:Q', format='.2f', title='Avg yield %')]
        )
        charts[0].altair_chart((events_bar + yield_line).resolve_scale(y='independent').properties(height=260), use_container_width=True)
    else:
        day_summary = (
            stats_df
            .assign(day=stats_df['date'].dt.day)
            .groupby('day')['yield']
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'avg_yield', 'count': 'events'})
        )
        combo_base = alt.Chart(day_summary)
        events_bar = combo_base.mark_bar(color='#1f77b4').encode(
            x=alt.X('day:O', title='Ex-Date day'),
            y=alt.Y('events:Q', title='Dividend count'),
            tooltip=[alt.Tooltip('day:O', title='Day'), alt.Tooltip('events:Q', title='Count')]
        )
        yield_line = combo_base.mark_line(color='#ff7f0e', point=True).encode(
            x='day:O',
            y=alt.Y('avg_yield:Q', title='Avg yield (%)', axis=alt.Axis(titleColor='#ff7f0e'), scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('day:O', title='Day'), alt.Tooltip('avg_yield:Q', format='.2f', title='Avg yield %')]
        )
        charts[0].altair_chart((events_bar + yield_line).resolve_scale(y='independent').properties(height=260), use_container_width=True)

    kde = alt.Chart(stats_df[['yield']]).transform_density(
        'yield',
        as_=['Yield', 'Density']
    ).mark_area(
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='#008631', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        ),
        line={'color': '#008631'}
    ).encode(
        x=alt.X('Yield:Q', title='Dividend Yield (%)'),
        y=alt.Y('Density:Q', title='', axis=alt.Axis(tickSize=0, domain=False, labelAngle=0, labelFontSize=0)),
        tooltip=[alt.Tooltip('Yield:Q', format='.2f', title='Yield %')]
    )
    mean_rule = alt.Chart(pd.DataFrame({'mean_yield': [stats_df['yield'].mean()]})).mark_rule(color='red').encode(
        x='mean_yield:Q',
        tooltip=[alt.Tooltip('mean_yield:Q', format='.2f', title='Average yield')]
    )
    charts[1].altair_chart((kde + mean_rule).properties(height=260), use_container_width=True)

    freq_df = (
        df.groupby('symbol')['date']
        .nunique()
        .reset_index(name='payments')
    )
    freq_df['payments'] = freq_df['payments'].clip(upper=4)
    category_order = ['Once per year', 'Twice per year', 'Thrice per year', '4 times per year']
    freq_labels = {1: category_order[0], 2: category_order[1], 3: category_order[2], 4: category_order[3]}
    freq_df['category'] = freq_df['payments'].map(freq_labels)

    freq_summary = (
        freq_df.groupby('category')
        .agg(
            count=('symbol', 'size'),
            stocks=('symbol', lambda s: ', '.join(sorted(s)))
        )
        .reindex(category_order)
        .reset_index()
        .rename(columns={'category': 'frequency'})
    )
    freq_summary['count'] = freq_summary['count'].fillna(0).astype(int)
    freq_summary['stocks'] = freq_summary['stocks'].fillna('')

    st.subheader('Payout Frequency (per year)')
    st.dataframe(
        freq_summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            'frequency': st.column_config.TextColumn('Frequency'),
            'count': st.column_config.NumberColumn('Companies', format='%d'),
            'stocks': st.column_config.TextColumn('Stocks')
        }
    )
