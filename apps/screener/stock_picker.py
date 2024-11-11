import os
import json
import requests
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from datetime import date, datetime, timedelta

import harvest.plot as hp
import harvest.data as hd

st.title('Jajan Saham')

# get list of all stocks
api_key = os.environ['FMP_API_KEY']

@st.cache_data
def compute_div_feature(cp_df, div_df):

    df = cp_df.copy()
    df['yield'] = df['lastDiv'] / df['price'] * 100

    for rows in df.iterrows():

        symbol = rows[0]
        try:
            div = pd.DataFrame(json.loads(div_df.loc[symbol, 'historical'].replace("'", '"')))
        except Exception as e:
            print('error', symbol)
            continue

        if len(div) == 0:
            continue
        
        div['year'] = [x.year for x in pd.to_datetime(div['date'])]
        agg_year = div.groupby('year')['adjDividend'].sum().to_frame().reset_index()
        inc_flat = agg_year['adjDividend'].shift(-1) - agg_year['adjDividend']
        inc_pct = inc_flat / agg_year['adjDividend'] * 100
        avg_flat_annual_increase = np.mean(inc_flat[-4:])
        # avg_pct_annual_increase = np.nanmedian(inc_pct)
        avg_pct_annual_increase = np.clip(avg_flat_annual_increase / df.loc[symbol, 'lastDiv'] * 100, 0, 100)
        df.loc[symbol, 'avgFlatAnnualDivIncrease'] = avg_flat_annual_increase
        df.loc[symbol, 'avgPctAnnualDivIncrease'] = avg_pct_annual_increase
        df.loc[symbol, 'numDividendYear'] = len(agg_year)
        df.loc[symbol, 'positiveYear'] = np.sum(inc_flat > 0)
        df.loc[symbol, 'numOfYear'] = datetime.today().year - datetime.strptime(df.loc[symbol, 'ipoDate'], '%Y-%m-%d').year
    
    # patented dividend score
    df['DScore'] = (df['lastDiv'] + df['avgFlatAnnualDivIncrease']*4)/df['price'] * (df['numDividendYear'] / (df['numOfYear']+25)/2) * (df['positiveYear'] / (df['numOfYear']+25)/2) * 100

    return df[['price', 'lastDiv', 'yield', 'sector', 'industry', 'mktCap', 'ipoDate', 
               'avgFlatAnnualDivIncrease', 'avgPctAnnualDivIncrease', 'numDividendYear', 'DScore']]


@st.cache_data
def get_company_profile(use_cache=False):
    
    bei = hd.get_all_idx_stocks(api_key=api_key)
    cp_df = hd.get_company_profile(bei['symbol'], api_key=api_key)

    return cp_df


def get_historical_dividend(use_cache=True):
    
    if use_cache:
        div_df = pd.read_csv('data/dividend_historical.csv').set_index('symbol')
    else:
        div_df  = pd.DataFrame()
    
    return div_df


def calc_statistics(full, market_cap_filter, dividend_filter):

    num_of_all_stocks = len(full)
    num_filtered_market_cap = len(full[full['mktCap'] > market_cap_filter*1000_000_1000])
    num_filtered_dividend_year = len(full[full['numDividendYear'] > dividend_filter])

    text = f"""
    Statistics as of {datetime.today().strftime('%Y-%m-%d')}  
    Number of stocks in Jakarta Stock Exchange is {num_of_all_stocks}  
    Number of stocks that have market capitalization above {market_cap_filter}B Rupiah is {num_filtered_market_cap}  
    Number of stocks that have paid dividend at lear {dividend_filter} year is {num_filtered_dividend_year}
    """

    return text


def get_daily_price(stock):
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{stock}?apikey={api_key}&from={start_date}'
    r = requests.get(url)
    intraday  = r.json()
    
    return pd.DataFrame(intraday['historical'])


### End of Function definition

cp_df = get_company_profile(use_cache=False)
div_df = get_historical_dividend(use_cache=True)
sector_df, industry_df = hd.get_sector_industry_pe((date.today()-timedelta(days=1)).isoformat(), api_key)

final_df = compute_div_feature(cp_df, div_df)

full_table_section = st.container(border=True)
with full_table_section:

    st.write('Filter')
    filter_cols = st.columns(2)
    minimum_market_cap = filter_cols[0].number_input('Minimum Market Capitalization (in Billion Rupiah)', value=1000, min_value=100, max_value=1000_1000)
    minimum_year = filter_cols[1].number_input('Minimum Number of Year Dividend Paid', value=1, min_value=0, max_value=25)

    st.markdown(calc_statistics(final_df, minimum_market_cap, minimum_year))
    final_df['Emiten'] = [x[:-3] for x in final_df.index]
    filtered_df = final_df[(final_df['mktCap'] >= minimum_market_cap*1000_000_000)
                            & (final_df['numDividendYear'] > minimum_year)
                            & (final_df['lastDiv'] > 0)].reset_index().set_index('symbol').sort_values('DScore', ascending=False)

    tabs = st.tabs(['Table View', 'Scatter View'])
    with tabs[0]:

        cfig={
            "yield": st.column_config.NumberColumn(
                "Yield (in pct)",
                help="The dividend yield on the current price",
                min_value=0,
                max_value=100,
                step=0.01,
                format="%.02f",
            )
        }

        event = st.dataframe(filtered_df, selection_mode=['single-row'], on_select='rerun', column_config=cfig)
    
    with tabs[1]:

        attributes = ['yield', 'avgPctAnnualDivIncrease', 'mktCap', 'DScore']

        scatter_cols = st.columns(2)
        x_axis = scatter_cols[0].selectbox('Select X Axis', attributes)
        minus_one_attribute = attributes[:]
        minus_one_attribute.remove(x_axis)
        y_axis = scatter_cols[1].selectbox('Select Y Axis', minus_one_attribute)
        point_selection = alt.selection_point(name='point')

        sp = alt.Chart(filtered_df).mark_point().encode(
            x=alt.X(x_axis, scale=alt.Scale(type='log')),
            y=alt.Y(y_axis, scale=alt.Scale()),
            tooltip='Emiten',
            color='sector',
            opacity=alt.condition(point_selection, alt.value(1), alt.value(0.2))
        ).add_selection(
            point_selection
        ).properties(
            height=400,
            width=1000
        ).interactive()
        sp_event = st.altair_chart(sp, on_select='rerun')

if len(event.selection['rows']) > 0:
    row_idx = event.selection['rows'][0]
    stock = filtered_df.iloc[row_idx]
    stock_name = stock.name
elif sp_event.selection['point']:
    row_idx = sp_event.selection['point'][0]
    stock = filtered_df.loc[row_idx['Emiten']+'.JK']
    stock_name = row_idx['Emiten']+'.JK'
else:
    st.stop()

# Company profile
with st.expander('Company Profile', expanded=False):
    st.write(cp_df.loc[stock_name, 'description'])

with st.expander('Dividend History', expanded=False):
    dividend_history_cols = st.columns([2, 5, 2])
    sdf = pd.DataFrame(json.loads(div_df.loc[stock.name, 'historical'].replace("'", '"')))
    dividend_history_cols[0].dataframe(sdf[['date', 'adjDividend']], hide_index=True)

    yearly_dividend_chart = hp.plot_dividend_history(sdf)
    dividend_history_cols[1].altair_chart(yearly_dividend_chart)

    fin = hd.get_financial_data(stock_name)
    with dividend_history_cols[2]:
        sector_name = filtered_df.loc[stock_name, 'sector']
        industry_name = filtered_df.loc[stock_name, 'industry']

        eps_ttm = float(fin['eps'][:4].sum())
        pe_ttm = filtered_df.loc[stock_name, 'price']/eps_ttm
        try:
            sector_pe = float(sector_df[sector_df['sector'] == sector_name]['pe'].to_list()[0])
            industry_pe = float(industry_df[industry_df['industry'] == industry_name]['pe'].to_list()[0])
        except Exception:
            sector_pe = industry_pe = -1
            print('sector or industry not found')
        
        st.write(f'EPS TTM {eps_ttm}')
        st.write(f'PE TTM {pe_ttm:.02f}')
        st.write(f'{sector_name} PE {sector_pe:.02f}')
        st.write(f'{industry_name} PE {industry_pe:.02f}')

    fin_chart = hp.plot_quarter_income(fin[fin['calendarYear'] > '2016'])
    st.altair_chart(fin_chart)

    daily_df = get_daily_price(stock_name)
    candlestick_chart = hp.plot_candlestick(daily_df)
    st.altair_chart(candlestick_chart)
