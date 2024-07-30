from datetime import date, datetime

import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

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

    col1, col2 = st.columns(2)

    with col1:
        AgGrid(df_summary)
    
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
        # df_year['year'] = df_year['year'].apply(lambda x: datetime(year=x, month=1, day=1))
        df_year['year'] = df_year['year'].astype('str')
        # st.dataframe(df_year)

        plot_year = alt.Chart(df_year).mark_bar().encode(
            x='year',
            y='Total Dividend'
        ).interactive()

        st.altair_chart(plot_year, use_container_width=True)

    with col2:
        df_month = df.groupby(['year', 'month']).agg({'Total Dividend': 'sum'}).reset_index()
        # st.dataframe(df_month)

        plot_month = alt.Chart(df_month).mark_bar().encode(
            x='month:N',
            xOffset='year:N',
            color='year:N',
            y='Total Dividend:Q'
        ).interactive()

        st.altair_chart(plot_month, use_container_width=True)

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
