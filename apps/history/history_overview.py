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

with st.expander('Data Input', expanded=True):
    method = st.radio('Method', ['Upload CSV', 'Paste Raw', 'Form'], horizontal=True)

    with st.form('abc'):

        if method == 'Upload CSV':
            uploaded_file = st.file_uploader('Choose a file', type='csv')

            if uploaded_file:
                st.session_state['history_file'] = uploaded_file
            
        elif method == 'Paste Raw':
            raw = st.text_area('Paste the Raw Data Here')
        
        elif method == 'Form':
            if st.session_state['history_df'] is None:
                example_df = pd.DataFrame(
                    [
                        {'Stock': 'ASII', 'Lot': '200', 'Dividend': '5000', 'Total Dividend': '1,068,400', 'Date': '15 Jan 2025'},
                    ]
                )
            else:
                example_df = st.session_state['history_df'].copy(deep=True)
            edited_df = st.data_editor(example_df, num_rows='dynamic', hide_index=True)

        submit = st.form_submit_button('Submit data')
        if submit:

            if method == 'Upload CSV':
                if st.session_state['history_file'] != 'EMPTY':
                    st.session_state['history_file'].seek(0)
                    st.session_state['history_df'] = pd.read_csv(st.session_state['history_file'], sep=',', dtype='str')

            elif method == 'Paste Raw':
                raw_lines = np.array(raw.split('\n'))

                stocks = []
                lots = []
                divs = []
                amounts = []
                dates = []
                for i, s in enumerate(raw_lines):

                    if 'DIVIDEND' not in s:
                        continue

                    stock = s.split(' ')[1]
                    lot = raw_lines[i+2]
                    div = raw_lines[i+4]
                    amount = raw_lines[i+6]
                    dt = raw_lines[i+14]

                    stocks += [stock]
                    lots += [lot]
                    divs += [div]
                    amounts += [amount]
                    dates   += [dt]

                    df = pd.DataFrame({'Stock': stocks, 'Lot': lots, 'Dividend': divs, 'Total Dividend': amounts, 'Date': dates})
                    st.session_state['history_df'] = df
                
            elif method == 'Form':
                df = edited_df.copy(deep=True)
                st.session_state['history_df'] = df


if 'history_df' not in st.session_state:
    st.session_state['history_df'] = None

if st.session_state['history_df'] is None:
    st.stop()

df = st.session_state['history_df'].copy()
df['Date'] = pd.to_datetime(df['Date']).dt.date
df['Total Dividend'] = df['Total Dividend'].str.replace(',', '').astype(float)


with st.expander('Overview', expanded=True):

    col1, col2 = st.columns([0.45, 0.55])
    with col1:
        total_dividend = f"IDR {df['Total Dividend'].sum():,}"
        st.metric(label='Total Historical Dividend', value=total_dividend)

        cola, colb = st.columns(2)
        with cola:
            num_transaction = len(df)
            st.metric(label='Number of Payment', value=num_transaction)

        with colb:
            num_stocks = len(df['Stock'].unique())
            st.metric(label='Number of Unique Stocks', value=num_stocks)
        
        cola, colb = st.columns(2)
        with cola:
            first_date = df['Date'].values[-1]
            st.metric(label='First Payment Date', value=f'{first_date}')

        with colb:
            last_date = df['Date'][0]
            st.metric(label='Last Payment Date', value=f'{last_date}')

        duration = last_date - first_date
        years = int(duration.days / 365)
        months = int((duration.days % 365) / 30)
        days = ((duration.days % 365) % 30)
        st.metric(label='Total Duration', value=f'{years} years, {months} months, {days} days')

    with col2:
        st.write('Last Payment Transactions')

        df_display = df[['Date', 'Stock', 'Lot', 'Dividend']].copy(deep=True)
        df_display['Total'] = df['Total Dividend'].astype('float')

        st.dataframe(df_display, hide_index=True, height=320)


with st.expander('Stock Aggregation', expanded=True):

    df['year'] = df['Date'].apply(lambda x: x.year)
    df['month'] = df['Date'].apply(lambda x: x.month)
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


with st.expander('Year on Year Comparison', expanded=True):
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

    param_cols = st.columns(3)
    first_year_param = df['year'].unique().tolist()
    select_first_year = param_cols[0].selectbox('Select First Year', first_year_param)

    second_year_param = first_year_param.copy()
    second_year_param.remove(select_first_year)
    select_second_year = param_cols[1].selectbox('Select Second Year', second_year_param)
    select_month = param_cols[2].selectbox('Select Month', list(calendar.month_name)[1:], 0)

    last_year_df = df[(df['year'] == select_first_year) & (df['month_name'] == select_month)][['Date', 'Stock', 'Lot', 'Dividend', 'Total Dividend']]
    curr_year_df = df[(df['year'] == select_second_year) & (df['month_name'] == select_month)][['Date', 'Stock', 'Lot', 'Dividend', 'Total Dividend']]

    month_cols = st.columns([1, 2, 2])
    month_cols[1].dataframe(last_year_df, hide_index=True)
    month_cols[2].dataframe(curr_year_df, hide_index=True)

    mdf = pd.DataFrame({
        'Year': [select_second_year, select_first_year],
        'Total Dividend': [curr_year_df['Total Dividend'].sum(), last_year_df['Total Dividend'].sum()]
    })

    mc = alt.Chart(mdf).mark_bar().encode(
        x = 'Year:O',
        y = 'sum(Total Dividend)',
        color = 'Year:N'
    )
    month_cols[0].altair_chart(mc)


with st.expander('Dividend Calendar', expanded=True):
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


with st.expander('Sector and Industry Analysis', expanded=True):
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


with st.expander('Single Stock View', expanded=True):
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
