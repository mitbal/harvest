import io
import os

import lesley
import calendar
import requests
import pendulum
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LinearRegression


st.set_page_config(layout='wide')
st.title('Portfolio Analysis')


if 'porto_df' not in st.session_state:
    st.session_state['porto_df'] = None

with st.expander('Data Input', expanded=True):
    method = st.radio('Method', ['Upload CSV', 'Paste Raw', 'Form'], horizontal=True)

    with st.form('abc'):

        if method == 'Upload CSV':
            uploaded_file = st.file_uploader("Choose a file")

            if uploaded_file:
                st.session_state['porto_file'] = uploaded_file
            
        elif method == 'Paste Raw':
            raw = st.text_area('Paste the Raw Data Here')
        
        elif method == 'Form':
            if st.session_state['porto_df'] is None:
                example_df = pd.DataFrame(
                    [
                        {'Symbol': 'ASII', 'Available Lot': '100', 'Average Price': '5000'},
                        {'Symbol': 'BBCA', 'Available Lot': '10', 'Average Price': '10000'},
                        {'Symbol': 'EXCL', 'Available Lot': '20', 'Average Price': '2000'},
                        {'Symbol': 'INDF', 'Available Lot': '30', 'Average Price': '6000'},
                        {'Symbol': 'PTBA', 'Available Lot': '50', 'Average Price': '3000'}
                    ]
                )
            else:
                example_df = st.session_state['porto_df'].copy(deep=True)
            edited_df = st.data_editor(example_df, num_rows='dynamic')

        target = st.number_input(
            label='Input Target Annual Income (in million IDR)', 
            value=120, step=1, 
            format='%d'
        )

        baseline = st.number_input(
            label='Benchmark Performance (in percent)',
            value=6.35, step=.01
        )

        submit = st.form_submit_button('Submit data')
        if submit:

            if method == 'Upload CSV':
                if st.session_state['porto_file'] != 'EMPTY':
                    st.session_state['porto_file'].seek(0)
                    st.session_state['porto_df'] = pd.read_csv(st.session_state['porto_file'], sep=',', dtype='str')

            elif method == 'Paste Raw':
                rows = np.array(raw.split())

                stock = rows[range(0, len(rows), 9)]
                lot = rows[range(1, len(rows), 9)]
                price = rows[range(3, len(rows), 9)]

                df = pd.DataFrame({
                    'Symbol': stock,
                    'Available Lot': lot,
                    'Average Price': price
                })
                st.session_state['porto_df'] = df

            elif method == 'Paste CSV':
                input_str = io.StringIO(raw)
                df = pd.read_csv(input_str, sep=';', dtype='str')
                st.session_state['porto_df'] = df
                
            elif method == 'Form':
                df = edited_df.copy(deep=True)
                st.session_state['porto_df'] = df

api_key = os.environ['FMP_API_KEY']

@st.cache_data
def get_company_profile_data(porto):

    stocks = ','.join([s+'.JK' for s in porto['Symbol']])
    company_profile_url = f'https://financialmodelingprep.com/api/v3/profile/{stocks}?apikey={api_key}'
    cpr = requests.get(company_profile_url)
    
    cp_df = pd.DataFrame(cpr.json())
    cp_df['Symbol'] = cp_df['symbol'].apply(lambda x: x[:-3])
    df = porto.merge(cp_df[['Symbol', 'price', 'sector', 'lastDiv']])
    df.rename(columns={'lastDiv': 'div_rate', 'price': 'last_price'}, inplace=True)

    return df

@st.cache_data
def get_dividend_data(porto):
    divs = []
    for i in range(int(len(porto)/5)+1):
        stock_list = [s+'.JK' for s in porto['Symbol'][i*5:(i+1)*5]]
        stocks = ','.join(stock_list)
        dividend_history_url = f'https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{stocks}?apikey={api_key}'
        dr = requests.get(dividend_history_url)
        drj = dr.json()
        
        if len(stock_list) > 1:
            div = drj['historicalStockList']
        else:
            if drj:
               div = [drj]
        divs.append(div)

    div_df = pd.concat([pd.DataFrame(div) for div in divs])
    div_df.set_index('symbol', inplace=True)
    divs = {x[:-3]: y for x, y in zip(div_df.index, div_df['historical'])}

    return divs


if st.session_state['porto_df'] is None:
    st.stop()

df = get_company_profile_data(st.session_state['porto_df'])
divs = get_dividend_data(st.session_state['porto_df'])

df['current_lot'] = df['Available Lot'].apply(lambda x: x.replace(',', '')).astype(float)
df['avg_price'] = df['Average Price'].apply(lambda x: x.replace(',', '')).astype(float)
df['total_invested'] = df['current_lot'] * df['avg_price'] * 100
df['yield_on_cost'] = df['div_rate'] / df['avg_price'] * 100
df['yield_on_price'] = df['div_rate'] / df['last_price'] * 100
df['total_dividend'] = df['div_rate'] * df['current_lot'] * 100

incs = []
years = []
for symbol in df['Symbol']:

    div = pd.DataFrame(divs[symbol])
    if len(div) == 0:
        incs += [0]
        years += [0]
        continue
    div['year'] = [x.year for x in pd.to_datetime(div['date'])]
    agg_year = div.groupby('year')['adjDividend'].sum().to_frame().reset_index()
    inc = agg_year['adjDividend'].shift(-1) - agg_year['adjDividend']
    avg_annual_increase = np.mean(inc)
    incs += [avg_annual_increase]
    years += [len(agg_year)]
df['numDividendYear'] = years
df['avgAnnualDivIncrease'] = incs

annual_dividend = df['total_dividend'].sum()
total_investment = df['total_invested'].sum()
achieve_percentage = annual_dividend / target * 100 / 1_000_000
total_yield_on_cost = annual_dividend / total_investment * 100


con_overall = st.container(border=True)
with con_overall:
    col1, col2, col3, col4 = st.columns(4, gap='small')
    with col1:
        delta = total_yield_on_cost - baseline
        if delta > 0:
            text_delta = f'{delta:.2f}% above benchmark'
        else:
            text_delta = f'{delta:.2f}% below benchmark'
        st.metric('Total Dividend Yield on Cost', 
                  value=f'{total_yield_on_cost:.2f} %',
                  delta=text_delta)
    with col2:
        st.metric('Dividend Annual Income', value=f'IDR {annual_dividend:,.0f}')
    with col3:
        st.metric('Total Investment', value=f'IDR {total_investment:,.0f}')
    with col4:
        st.metric('Percent on Target', value=f'{achieve_percentage:.2f} %')


df_display = df[['Symbol', 'Available Lot', 'avg_price', 'total_invested', 'div_rate', 'last_price', 
                 'yield_on_cost', 'yield_on_price', 'total_dividend', 'numDividendYear', 'avgAnnualDivIncrease']].copy(deep=True)

con_table = st.container(border=True)
with con_table:

    tabs = st.tabs(['Table View', 'Bar Chart View', 'Sectoral View', 'Calendar View'])
    
    with tabs[0]:
        st.write('Current Portfolio')
        main_event = st.dataframe(df_display, on_select='rerun', selection_mode='single-row', hide_index=True)

    with tabs[1]:
        div_bar = alt.Chart(df_display).mark_bar().encode(
            x=alt.X('Symbol'),
            y=alt.Y('total_dividend')
        )
        yield_bar = alt.Chart(df_display).mark_line(color='orange').encode(
            x=alt.X('Symbol'),
            y=alt.Y('yield_on_cost', scale=alt.Scale(domain=[0, 100])),
        )
        combined_chart = (div_bar + yield_bar).resolve_scale(y='independent')
        st.altair_chart(combined_chart)

    with tabs[2]:
        sector_cols = st.columns([.7,.5,1])
        
        with sector_cols[0]:
            sector_df = df.groupby('sector')['total_dividend'].sum().to_frame().sort_values('total_dividend', ascending=False).reset_index()
            event = st.dataframe(sector_df, 
                                 selection_mode=['single-row'], 
                                 on_select='rerun',
                                 hide_index=True,
                                 key='data')

        with sector_cols[1]:
            if len(event.selection['rows']) > 0:
                row_idx = event.selection['rows'][0]
                sector_name = sector_df.loc[row_idx, 'sector']
                st.dataframe(df[df['sector'] == sector_name][['Symbol', 'total_dividend']].sort_values('total_dividend', ascending=False), hide_index=True, )
            else:
                st.info('Select one of the sector on the table on the left')

        sector_pie = alt.Chart(df).mark_arc().encode(
            theta='sum(total_dividend)',
            color='sector'
        ).interactive()

        with sector_cols[2]:
            st.altair_chart(sector_pie)

    with tabs[3]:

        view_type = st.radio('Select View', ['Calendar', 'Bar Chart', 'Table'], horizontal=True)

        # prepare calendar data
        div_lists = []
        for index, row in df.iterrows():

            r = row.to_dict()
            stock = r['Symbol']
            if len(divs[stock]) == 0:
                continue
            div_df = pd.DataFrame(divs[stock])
            div_df['year'] = div_df['date'].apply(lambda x: x.split('-')[0])
            div_df['date'] = pd.to_datetime(div_df['date']).dt.tz_localize(None)

            end_date = pd.Timestamp('today').to_datetime64()
            start_date = (end_date - pd.Timedelta(days=365)).to_datetime64()

            current_year = datetime.today().year
            last_year_div = div_df[(pd.to_datetime(div_df['date']) >= start_date) & (pd.to_datetime(div_df['date']) < end_date)]
            last_year_div.loc[:, 'Symbol'] = stock
            last_year_div.loc[:, 'Lot'] = r['current_lot']

            div_lists += [last_year_div]

        all_divs = pd.concat(div_lists).reset_index(drop=True)       
        all_divs['total_dividend'] = (all_divs['Lot'] * all_divs['adjDividend'] * 100).astype('int')
        all_divs['Date'] = pd.to_datetime(all_divs['date']).dt.tz_localize(None)
        all_divs['new_date'] = all_divs['date'].apply(lambda x: x + pd.Timedelta(days=14)) # payment date on average 2 weeks after ex-date
        all_divs['month'] = all_divs['new_date'].apply(lambda x: x.month)
        
        month_div = all_divs.groupby('month')['total_dividend'].sum().to_frame().reset_index()
        month_div['month_name'] = month_div['month'].apply(lambda x: calendar.month_name[x])
        
        if view_type == 'Calendar':
            all_divs['new_date'] = all_divs['new_date'].apply(lambda x: datetime(year=current_year+1, month=x.month, day=x.day))
            cal = lesley.calendar_plot(all_divs['new_date'], all_divs['total_dividend'], nrows=3)
            st.altair_chart(cal)
        
        elif view_type == 'Bar Chart':
            bar_cols = st.columns(2)
            bar_cols[0].dataframe(month_div[['month_name', 'total_dividend']], hide_index=True)

            month_bar = alt.Chart(month_div).mark_bar().encode(
                x=alt.X('month:N'),
                y=alt.Y('total_dividend')
            )
            bar_cols[1].altair_chart(month_bar)
        
        else:
            row_1 = st.container()
            with row_1:
                row_1_cols = st.columns(6)
                for c, i in zip(row_1_cols, range(1, 7)):
                    m = all_divs[all_divs['month'] == i]
                    c.write(calendar.month_name[i])
                    c.dataframe(m[['Symbol', 'total_dividend']].sort_values('total_dividend', ascending=False), hide_index=True)

            row_2 = st.container()
            with row_2:
                row_2_cols = st.columns(6)
                for c, i in zip(row_2_cols, range(7, 13)):
                    m = all_divs[all_divs['month'] == i]
                    c.write(calendar.month_name[i])
                    c.dataframe(m[['Symbol', 'total_dividend']].sort_values('total_dividend', ascending=False), hide_index=True)


def fit_linear(divs):

    df_train = pd.DataFrame(divs)
    df_train['year'] = df_train['date'].apply(lambda x: int(x.split('-')[0]))
    df_train = df_train.groupby('year')['adjDividend'].sum().to_frame().reset_index()
    
    # insert missing year into df_train
    start_year = df_train.loc[0, 'year']
    end_year = df_train.loc[len(df_train)-1, 'year']

    years = list(range(start_year, end_year + 1))
    df_temp = pd.DataFrame({'year': years, 'value': [0]*len(years)})
    df_train = pd.merge(df_temp, df_train, on='year', how='left')
    df_train = df_train.fillna(0)

    year = pendulum.today().year
    y = df_train[df_train['year'] < year]['adjDividend'].to_numpy().reshape(-1, 1)
    X = np.arange(y.shape[0]).reshape(-1, 1)

    weight = np.append([2*y.shape[0]], np.ones(y.shape[0]-1)) # make the first dividend as kind of intercept
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y, sample_weight=weight)

    X_predict = np.arange(y.shape[0]+2).reshape(-1, 1)
    y_hat = lr.predict(X_predict)
    
    df_predict = pd.DataFrame()
    df_predict['year'] = np.append(df_train['year'].values, [year+1])
    df_predict['Prediction'] = y_hat

    return lr.score(X, y), df_predict['Prediction'].values[-1], df_train, df_predict



detail_section = st.container(border=True)
with detail_section:
    
    if main_event.selection['rows']:
        symbol = df_display.iloc[main_event.selection['rows'][0]]['Symbol']

        st.write('You select: ', symbol)

        score, pred, df_train, df_predict = fit_linear(divs[symbol])

        col1, col2, col3 = st.columns([0.25, 0.4, 0.35])
        with col1:
            st.dataframe(pd.DataFrame(divs[symbol])[['date', 'adjDividend']], hide_index=True)

        with col2:
            df_train['inc'] = df_train['adjDividend'].shift(1) - df_train['adjDividend']
            div_bar = alt.Chart(df_train).mark_bar().encode(
                alt.X('year:N'),
                alt.Y('adjDividend'),
                color=alt.condition(alt.datum['inc'] > 0, alt.value('#ff796c'), alt.value('#008631'))
            ).properties(
                height=300,
                width=450
            )

            pred_line = alt.Chart(df_predict).mark_line().encode(
                alt.X('year:N'),
                alt.Y('Prediction'),
                color=alt.value('#87CEFA')
            )

            st.altair_chart(div_bar + pred_line)

        with col3:
            st.write(f'R squared: {score:.2f}')
            st.write(f'Prediction for Next Year Dividend: {pred:.2f}')
            
            last = pd.DataFrame(divs[symbol])['adjDividend'].values[0]
            if pred > last:
                color = 'green'
            else:
                color = 'red'

            st.write(f'Difference compared to the previous year: **:{color}[{pred-last:.2f}]**')
            st.write(f'Percentage difference compared to the previous year: **:{color}[{(pred-last)/last*100:.2f}%]**')

            current_year = datetime.now().year
            number_of_year = len(df_train)
            consistency = number_of_year / (current_year - df_train['year'][0] +1)
            st.write(f'Number of year {number_of_year}, consistency {consistency*100:.2f}%')

            df_train['inc'] = df_train['adjDividend'].shift(-1) - df_train['adjDividend']
            avg_annual_increase = np.mean(df_train['inc'])
            num_positive_year = np.sum(df_train['inc'] > 0)
            st.write(f'Average annual increase {avg_annual_increase:.2f}, with number of positive year {num_positive_year}, increase percentage {num_positive_year / number_of_year*100:.2f}%')


future_section = st.container(border=True)
with future_section:
    st.write('Future Projection')
    # first method, assume flat percentage increase each year based on current yield

    future_cols = st.columns(2)
    number_of_year = future_cols[0].number_input('Number of Year', value=25, min_value=1, max_value=50)
    inc = future_cols[1].number_input('Input annual percentage increase', value=total_yield_on_cost, min_value=1.0, max_value=15.0, step=0.1)
    futures = [0]*number_of_year
    for i in range(number_of_year):
        futures[i] = annual_dividend * (1+inc/100)**i

    df_future = pd.DataFrame({'years': [f'Year {i+1:2d}' for i in range(number_of_year)], 'returns': futures})
    df_future['achieved'] = df_future['returns'] > (target*1_000_000)
    future_chart = alt.Chart(df_future).mark_bar().encode(
        x=alt.X('years'),
        y=alt.Y('returns'),
        color=alt.condition(alt.datum['achieved'], alt.value('#008631'), alt.value('#87CEFA')),
    ).properties(
        width=1000
    )
    st.altair_chart(future_chart)
