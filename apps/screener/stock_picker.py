import os
import json
import requests
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from datetime import datetime

st.title('Jajan Saham')

# get list of all stocks
api_key = os.environ['FMP_API_KEY']

@st.cache_data
def compute_div_feature(cp_df, div_df):

    df = cp_df.copy()
    df = df[df['mktCap'] > 1_000_000_000_000]
    df['yield'] = df['lastDiv'] / df['price'] * 100

    for rows in df.iterrows():

        symbol = rows[0]
        div = pd.DataFrame(json.loads(div_df.loc[symbol, 'historical'].replace("'", '"')))
        if len(div) == 0:
            continue
        
        div['year'] = [x.year for x in pd.to_datetime(div['date'])]
        agg_year = div.groupby('year')['adjDividend'].sum().to_frame().reset_index()
        inc_flat = agg_year['adjDividend'].shift(-1) - agg_year['adjDividend']
        inc_pct = inc_flat / agg_year['adjDividend'] * 100
        avg_flat_annual_increase = np.mean(inc_flat)
        # avg_pct_annual_increase = np.nanmedian(inc_pct)
        avg_pct_annual_increase = np.clip(avg_flat_annual_increase / df.loc[symbol, 'lastDiv'] * 100, 0, 100)
        df.loc[symbol, 'avgFlatAnnualDivIncrease'] = avg_flat_annual_increase
        df.loc[symbol, 'avgPctAnnualDivIncrease'] = avg_pct_annual_increase
        df.loc[symbol, 'numDividendYear'] = len(agg_year)

    # patented dividend score
    df['DScore'] = df['yield'] * df['avgPctAnnualDivIncrease'] * (df['numDividendYear'] / 25)

    return df[['price', 'lastDiv', 'yield', 'sector', 'industry', 'mktCap', 'ipoDate', 
               'avgFlatAnnualDivIncrease', 'avgPctAnnualDivIncrease', 'numDividendYear', 'DScore']]


@st.cache_data
def get_company_profile(use_cache=False):
    
    url = f'https://financialmodelingprep.com/api/v3/stock/list?apikey={api_key}'
    r = requests.get(url)
    sl = r.json()

    sdf = pd.DataFrame(sl)
    bei = sdf[sdf['exchangeShortName'] == 'JKT'].reset_index(drop=True)

    stocks = ','.join(bei['symbol'])
    company_profile_url = f'https://financialmodelingprep.com/api/v3/profile/{stocks}?apikey={api_key}'
    r = requests.get(company_profile_url)
    cp = r.json()

    cp_df = pd.DataFrame(cp).set_index('symbol')
    return cp_df


def get_historical_dividend(use_cache=True):
    
    if use_cache:
        div_df = pd.read_csv('data/dividend_historical.csv').set_index('symbol')
    else:
        div_df  = pd.DataFrame()
    
    return div_df


def get_financial_data():
    pass


cp_df = get_company_profile(use_cache=False)
div_df = get_historical_dividend(use_cache=True)
fin_df = get_financial_data()

final_df = compute_div_feature(cp_df, div_df)

full_table_section = st.container(border=True)
with full_table_section:
    st.write('Statistics')
    st.write(f'Number of stocks in Jakarta Stock Exchange as of today {datetime.today().strftime('%Y-%m-%d')} is {len(cp_df)}')
    st.write(f'Number of stocks that have market capitalization above 1T Rupiah is {len(final_df)}')
    st.write(f'Number of stocks that have paid dividend at least once is {len(final_df[final_df['numDividendYear'] > 0])}')
    filtered_df = final_df[final_df['numDividendYear'] > 0].reset_index().set_index('symbol').sort_values('DScore', ascending=False)
    event = st.dataframe(filtered_df, selection_mode=['single-row'], on_select='rerun')


detail_section = st.container(border=True)
with detail_section:
    if len(event.selection['rows']) > 0:
        row_idx = event.selection['rows'][0]
        stock = filtered_df.iloc[row_idx]
        
        st.write(cp_df.loc[stock.name, 'description'])
        st.dataframe(pd.DataFrame(json.loads(div_df.loc[stock.name, 'historical'].replace("'", '"'))))



# calculate patented dividend score

# display the table and the scatter plot
scatter_section = st.container(border=True)
with scatter_section:
    x_axis = st.selectbox('Select X Axis', ['yield'])
    y_axis = st.selectbox('Select Y Axis', ['avgPctAnnualDivIncrease'])

    selection = alt.selection_point(fields=['sector'], bind='legend')

    filtered_df = final_df[(final_df['avgPctAnnualDivIncrease'] < 100)
                           & (final_df['numDividendYear'] > 5)
                           & (final_df['lastDiv'] > 0)
                           & (final_df['avgPctAnnualDivIncrease'] > 0)].reset_index()
    sp = alt.Chart(filtered_df).mark_point().encode(
        x=alt.X(x_axis, scale=alt.Scale(type='log')),
        y=alt.Y(y_axis, scale=alt.Scale()),
        tooltip='symbol',
        color='sector',
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    ).add_params(
        selection
    ).properties(
        height=400,
        width=1000
    ).interactive()
    st.altair_chart(sp)
