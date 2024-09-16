import os
import json
import requests
import pandas as pd
import streamlit as st

st.title('Jajan Saham')

# get list of all stocks
api_key = os.environ['FMP_API_KEY']

def compute_div_feature(cp_df):

    df = cp_df.copy()
    df = df[df['mktCap'] > 1_000_000_000_000]
    df['yield'] = df['lastDiv'] / df['price']

    return df[['price', 'lastDiv', 'sector', 'industry', 'mktCap', 'yield', 'ipoDate']]


@st.cache_data
def get_company_profile():
    
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


# cp_df = pd.read_csv('data/company_profiles.csv')
cp_df = get_company_profile()
# div_df = pd.read_csv('data/dividend_historical.csv').set_index('symbol')

final_df = compute_div_feature(cp_df)
# final_df = compute_div_feature(cp_df, div_df)
# st.dataframe(cp_df, hide_index=True)

st.dataframe(final_df)

# get the attributes

# calculate patented dividend score


# display the table and the scatter plot
