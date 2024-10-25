import calendar
from datetime import datetime

import numpy as np
import pandas as pd
import altair as alt

import lesley
import yfinance as yf
import plotly.graph_objects as go

import streamlit as st

st.set_page_config(layout='wide')
st.title('Historical Insight')

uploaded_file = st.file_uploader('Choose a file')
if uploaded_file:
    st.session_state['history_file'] = uploaded_file

if st.session_state['history_file'] != 'EMPTY':
    st.session_state['history_file'].seek(0)
    df = pd.read_csv(st.session_state['history_file'], delimiter=';')
else:
    st.stop()


## First section, overall 
con = st.container(border=True)
with con:

    col1, col2 = st.columns([0.45, 0.55])
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
        years = int(duration.days / 365)
        months = int((duration.days % 365) / 30)
        days = ((duration.days % 365) % 30)
        st.metric(label='Total Duration', value=f'{years} years, {months} months, {days} days')

    with col2:
        st.write('List of transactions')

        df_display = df[['Date', 'Stock', 'Lot']].copy(deep=True)
        df_display['Dividend'] = df['Price'].astype('float')
        df_display['Total'] = df['Total Dividend'].astype('float')

        st.dataframe(df_display, hide_index=True)

## Second section, summarization
con2 = st.container(border=True)
with con2:

    df['year'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').year)
    df['month'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').month)
    df_summary = df.groupby('Stock').agg({'Total Dividend': 'sum', 'Date': 'count'}).reset_index()

    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        st.dataframe(df_summary, hide_index=True)

    with col2:
        bar_plot = alt.Chart(df_summary).mark_bar().encode(
            x='Stock',
            y='Total Dividend'
        ).interactive()
        st.altair_chart(bar_plot, use_container_width=True)


## Now year on year analysis
con3 = st.container(border=True)
with con3:

    col1, col2 = st.columns(2)
    with col1:

        df_year = df.groupby('year').agg({'Total Dividend': 'sum'}).reset_index()
        df_year['year'] = df_year['year'].astype('str')

        plot_year = alt.Chart(df_year).mark_bar().encode(
            x='year',
            y='Total Dividend'
        ).interactive()

        st.altair_chart(plot_year, use_container_width=True)

    with col2:

        df['month_name'] = df['month'].apply(lambda x: calendar.month_name[x])

        df_month = df.groupby(['year', 'month_name']).agg({'Total Dividend': 'sum'}).reset_index()
        plot_month = alt.Chart(df_month).mark_bar().encode(
            x=alt.X('month_name', sort=list(calendar.month_name)),
            color='year:N',
            y='Total Dividend:Q'
        ).interactive()

        st.altair_chart(plot_month, use_container_width=True)


## fourth section, dividend calendar
con4 = st.container(border=True)
with con4:

    div_cols = st.columns(2)
    with div_cols[0]:
        year_list = df['year'].unique()
        year_select = st.selectbox('Select the Year', year_list)
    with div_cols[1]:
        method = st.selectbox('Select Aggregation', ['Count', 'Sum', 'Percentage'])

    filtered_df = df[df['year'] == year_select]
    if method == 'Count':
        agg = filtered_df.groupby('Date')['Stock'].count().to_frame().reset_index()
        heatmap_chart = lesley.cal_heatmap(pd.to_datetime(agg['Date']), agg['Stock'], height=300)
    elif method == 'Sum':
        agg = filtered_df.groupby('Date')['Total Dividend'].sum().to_frame().reset_index()
        heatmap_chart = lesley.cal_heatmap(pd.to_datetime(agg['Date']), agg['Total Dividend'], height=300)
    else:
        agg = filtered_df.groupby('Date')['Total Dividend'].sum().to_frame().reset_index()
        agg['pct'] = agg['Total Dividend'] / agg['Total Dividend'].sum()
        heatmap_chart = lesley.cal_heatmap(pd.to_datetime(agg['Date']), agg['pct'], height=300)

    st.write('Dividend Calendar')
    st.altair_chart(heatmap_chart)


# Section 5, Sector and Industries Analysis
con5 = st.container(border=True)
with con5:

    sectors = {}
    industries = {}
    for s in df['Stock'].unique():
        ticker = yf.Ticker(s+'.JK')
        sectors[s] = ticker.info['sector']
        industries[s] = ticker.info['industry']

    df_sectors = pd.DataFrame.from_dict({'stock': sectors.keys(), 'sector': sectors.values()})
    df_industries = pd.DataFrame.from_dict({'stock': industries.keys(), 'industry': industries.values()})
    df = df.merge(df_sectors, left_on='Stock', right_on='stock')
    df = df.merge(df_industries, left_on='Stock', right_on='stock')
    
    agg_sector = df.groupby('sector')['Total Dividend'].sum().sort_values(ascending=False).reset_index()
    agg_industry = df.groupby('industry')['Total Dividend'].sum().sort_values(ascending=False).reset_index()    

    st.write('Sector Analysis')
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(agg_sector, hide_index=True)
        st.dataframe(agg_industry, hide_index=True)

    with col2:
        plot_sectors = alt.Chart(agg_sector).mark_arc().encode(
            theta='Total Dividend:Q',
            color='sector',
        ).interactive()
        st.altair_chart(plot_sectors, use_container_width=True)

        plot_industries = alt.Chart(agg_industry).mark_arc().encode(
            theta='Total Dividend:Q',
            color='industry',
        ).interactive()
        st.altair_chart(plot_industries, use_container_width=True)


# Last section, single stock analysis
con6 = st.container(border=True)
with con6:

    col1, col2 = st.columns([0.35, 0.65])
    with col1:
        stock_list = np.sort(df['Stock'].unique())
        stock = st.selectbox('Select Stock', stock_list) 
        filt = df_display[df_display['Stock'] == stock][['Date', 'Lot', 'Dividend', 'Total']].reset_index().drop(columns='index')
        
        st.dataframe(filt, hide_index=True)

    with col2:
        history = yf.Ticker(stock+'.JK').history(period='1y').reset_index()
        candlestick = go.Candlestick(
                                    x=history['Date'],
                                    open=history['Open'],
                                    high=history['High'],
                                    low=history['Low'],
                                    close=history['Close']
                                    )

        fig = go.Figure(data=[candlestick])
        st.plotly_chart(fig)
