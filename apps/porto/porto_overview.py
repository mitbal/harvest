import io
import os

import lesley
import calendar
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from datetime import datetime

import harvest.plot as hp
import harvest.data as hd

st.set_page_config(layout='wide')
st.title('Portfolio Analysis')


if 'porto_df' not in st.session_state:
    st.session_state['porto_df'] = None

with st.expander('Data Input', expanded=True):
    method = st.radio('Method', ['Upload CSV', 'Paste Raw', 'Form'], horizontal=True)

    with st.form('abc'):

        if method == 'Upload CSV':
            uploaded_file = st.file_uploader('Choose a file', type='csv')

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

        form_cols = st.columns(2)

        target = form_cols[0].number_input(
            label='Input Target Annual Income (in million IDR)', 
            value=120, step=1, 
            format='%d'
        )

        baseline = form_cols[1].number_input(
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

    stocks = [x+'.JK' for x in porto['Symbol'].to_list()]
    cp_df = hd.get_company_profile(stocks)
    
    cp_df['Symbol'] = [x[:-3] for x in cp_df.index.to_list()]
    df = porto.merge(cp_df[['Symbol', 'price', 'sector', 'lastDiv']])
    df.rename(columns={'lastDiv': 'div_rate', 'price': 'last_price'}, inplace=True)

    return df


@st.cache_data
def get_dividend_data(porto):

    return hd.get_dividend_history(porto['Symbol'].to_list())


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
current_investment_value = (df['current_lot'] * df['last_price'] * 100).sum()
achieve_percentage = annual_dividend / target * 100 / 1_000_000
total_yield_on_cost = annual_dividend / total_investment * 100

df_display = df[['Symbol', 'Available Lot', 'avg_price', 'total_invested', 'div_rate', 'last_price', 
                 'yield_on_cost', 'yield_on_price', 'total_dividend', 'numDividendYear', 'avgAnnualDivIncrease']].copy(deep=True)


# Overall summary
with st.container(border=True):
    overall_cols = st.columns([8,10,10,10,6], gap='small')
    with overall_cols[0]:
        delta = total_yield_on_cost - baseline
        if delta > 0:
            text_delta = f'{delta:.2f}% above benchmark'
        else:
            text_delta = f'{delta:.2f}% below benchmark'
        st.metric('Total Dividend Yield on Cost', 
                  value=f'{total_yield_on_cost:.2f} %',
                  delta=text_delta)
    
    overall_cols[1].metric('Dividend Annual Income', value=f'IDR {annual_dividend:,.0f}')
    overall_cols[2].metric('Total Investment', value=f'IDR {total_investment:,.0f}')
    overall_cols[3].metric('Current Market Value', value=f'IDR {current_investment_value:,.0f}', delta=f'{current_investment_value-total_investment:,.0f} IDR')
    overall_cols[4].metric('Percent on Target', value=f'{achieve_percentage:.2f} %')

# Table List
with st.container(border=True):

    tabs = st.tabs(['Table View', 'Bar Chart View', 'Sectoral View', 'Calendar View'])
    
    with tabs[0]:
        st.write('Current Portfolio')

        cfig = {
            'yield_on_cost': st.column_config.NumberColumn(
                'Yield on Cost (in pct)',
                format='%.2f',
            ),
            'yield_on_price': st.column_config.NumberColumn(
                'Yield on Price (in pct)',
                format='%.2f',
            ),
            'div_rate': st.column_config.NumberColumn(
                'Last Dividend Paid',
                format='%.0f'
            )
        }

        main_event = st.dataframe(df_display, 
                                  on_select='rerun', 
                                  selection_mode='single-row', 
                                  hide_index=True,
                                  column_config=cfig)

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
            last_year_div = div_df[(pd.to_datetime(div_df['date']) >= start_date) & (pd.to_datetime(div_df['date']) < end_date)].copy(deep=True)
            last_year_div['Symbol'] = stock
            last_year_div['Lot'] = r['current_lot']

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

# Detailed single stock
with st.expander('Dividend History', expanded=True):
    
    if main_event.selection['rows']:
        symbol = df_display.iloc[main_event.selection['rows'][0]]['Symbol']

        st.write('You select: ', symbol)

        div_hist_cols = st.columns([3, 10, 5])
        with div_hist_cols[0]:
            st.dataframe(pd.DataFrame(divs[symbol])[['date', 'adjDividend']], hide_index=True)

        with div_hist_cols[1]:
            div_df = pd.DataFrame(divs[symbol])
            div_bar = hp.plot_dividend_history(div_df)

            regression = div_bar.transform_loess('year', 'adjDividend', bandwidth=0.7).mark_line().encode(color=alt.value('#000000'))
            st.altair_chart(div_bar + regression)

        with div_hist_cols[2]:

            symbol_last_div = df[df['Symbol'] == symbol].iloc[0]['div_rate']
            symbol_flat_inc = df[df['Symbol'] == symbol].iloc[0]['avgAnnualDivIncrease']
            next_year_dividend = symbol_last_div + symbol_flat_inc
            inc_rate = symbol_flat_inc / symbol_last_div * 100

            st.markdown(f'Next Year Dividend Prediction: **:{"green"}[{next_year_dividend:.2f}]** IDR')
            st.write(f'Percentage Increase from Last Year: {inc_rate:.2f} %')
            
            df_train = div_df.copy()
            df_train['year'] = df_train['date'].apply(lambda x: int(x.split('-')[0]))
            df_train = df_train.groupby('year')['adjDividend'].sum().to_frame().reset_index()
            df_train['inc'] = df_train['adjDividend'].shift(-1) - df_train['adjDividend']
            
            num_positive_year = np.sum(df_train['inc'] > 0) + 1 # the first year is considered positive
            num_dividend_year = df[df['Symbol'] == symbol].iloc[0]['numDividendYear']
            pct_positive_year = num_positive_year/num_dividend_year * 100

            st.write(f'Number of Positive Years (Dividend increase from the year before): {num_positive_year}')
            st.write(f'Percentage of positive years: {pct_positive_year:.02f} %')

# Project future earnings
with st.expander('Future Projection', expanded=True):
    # Assume growth based on current yield with reinvestment

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
