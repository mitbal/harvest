import os
import json
import requests
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

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
        inc = agg_year['adjDividend'].shift(-1) - agg_year['adjDividend']
        avg_annual_increase = np.mean(inc)
        df.loc[symbol, 'avgAnnualDivIncrease'] = avg_annual_increase

    return df[['price', 'lastDiv', 'yield', 'sector', 'industry', 'mktCap', 'ipoDate', 'avgAnnualDivIncrease']]


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

event = st.dataframe(final_df, selection_mode=['single-row'], on_select='rerun')

detail_con = st.container(border=True)

# display the detailed section of a single stock
if len(event.selection['rows']) > 0:
    row_idx = event.selection['rows'][0]
    stock = final_df.iloc[row_idx]

    st.dataframe(pd.DataFrame(json.loads(div_df.loc[stock.name, 'historical'].replace("'", '"'))))

# get the attributes


# calculate patented dividend score


# display the table and the scatter plot

