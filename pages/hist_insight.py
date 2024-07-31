from datetime import date, datetime

import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

import july
import yfinance as yf
import plotly.graph_objects as go

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, JsCode

st.set_page_config(layout='wide')

st.title('Historical Insight')

uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is None:
    st.stop()
df = pd.read_csv(uploaded_file, delimiter=';')


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

        k_sep_formatter = JsCode("""
            function(params) {
                return (params.value == null) ? params.value : params.value.toLocaleString(); 
            }
        """)

        builder = GridOptionsBuilder.from_dataframe(df_display)
        builder.configure_column('Dividend', header_name='Dividend (IDR)', type=['numericColumn', 'numberColumnFilter', 'customNumericFormat'], precision=1)
        builder.configure_column('Total', header_name='Total (IDR)', type=['numericColumn', 'numberColumnFilter', 'customNumericFormat'], precision=0, valueFormatter=k_sep_formatter)
        
        grid_options = builder.build()
        AgGrid(df_display, 
               height=300,
               gridOptions=grid_options,
               allow_unsafe_jscode=True,
               columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)


## Second section, summarization
con2 = st.container(border=True)
with con2:

    df['year'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').year)
    df['month'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').month)
    df_summary = df.groupby('Stock').agg({'Total Dividend': 'sum', 'Date': 'count'}).reset_index()

    col1, col2 = st.columns([0.3, 0.7])
    with col1:

        builder = GridOptionsBuilder.from_dataframe(df_summary)
        builder.configure_column('Total Dividend', header_name='Total Dividend', type=['numericColumn', 'numberColumnFilter', 'customNumericFormat'], precision=0, valueFormatter=k_sep_formatter)
        builder.configure_column('Date', header_name='Count', type=['numericColumn', 'numberColumnFilter', 'customNumericFormat'], precision=0)
        
        grid_options = builder.build()
        AgGrid(df_summary, 
               height=310,
               gridOptions=grid_options,
               allow_unsafe_jscode=True,
               columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)
    
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

        df_month = df.groupby(['year', 'month']).agg({'Total Dividend': 'sum'}).reset_index()
        plot_month = alt.Chart(df_month).mark_bar().encode(
            x='month:N',
            xOffset='year:N',
            color='year:N',
            y='Total Dividend:Q'
        ).interactive()

        st.altair_chart(plot_month, use_container_width=True)


## fourth section, dividend calendar
con4 = st.container(border=True)
with con4:

    year_list = df['year'].unique()
    year_select = st.selectbox('Select the Year', year_list)

    filtered_df = df[df['year'] == year_select]
    counts = filtered_df.groupby('Date')['Stock'].count().to_frame().reset_index()
    counts = pd.concat([pd.DataFrame({'Date': date(year=year_select, month=1, day=1), 'Stock': 0}, index=[0]), 
                        counts,
                        pd.DataFrame({'Date': date(year=year_select, month=12, day=31), 'Stock': 0}, index=[0])]).reset_index()
    ax = july.heatmap(counts['Date'], counts['Stock'], month_grid=True, cmap='github')

    st.write('Dividend Calendar')
    st.pyplot(ax.get_figure())


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
        st.dataframe(agg_sector)
        st.dataframe(agg_industry)

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
        # st.d÷at÷aframe(filt)
        AgGrid(filt)

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
