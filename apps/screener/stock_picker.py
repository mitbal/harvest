import os
import json
import requests
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from datetime import datetime, timedelta

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
        df.loc[symbol, 'positiveYear'] = np.sum(inc_flat > 0)
        df.loc[symbol, 'numOfYear'] = datetime.today().year - datetime.strptime(df.loc[symbol, 'ipoDate'], '%Y-%m-%d').year
    
    # patented dividend score
    df['DScore'] = df['yield'] * df['avgPctAnnualDivIncrease'] * (df['numDividendYear'] / (df['numOfYear']+25)/2) * (df['positiveYear'] / (df['numOfYear']+25)/2)

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


def get_financial_data(stock):
    period = 'quarter' # or 'annual'
    url = f'https://financialmodelingprep.com/api/v3/income-statement/{stock}?period={period}&apikey={api_key}'
    r = requests.get(url)
    fs = r.json()
    return pd.DataFrame(fs)


def calc_statistics():
    text = f"""
    Statistics as of {datetime.today().strftime('%Y-%m-%d')}  
    Number of stocks in Jakarta Stock Exchange is {len(cp_df)}  
    Number of stocks that have market capitalization above 1T Rupiah is {len(final_df)}  
    Number of stocks that have paid dividend at least once is {len(final_df[final_df['numDividendYear'] > 0])}
    """

    return text


def get_daily_price(stock):
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{stock}?apikey={api_key}&from={start_date}'
    r = requests.get(url)
    intraday  = r.json()
    
    return pd.DataFrame(intraday['historical'])


def plot_candlestick(daily_df):
    open_close_color = alt.condition(
            "datum.open <= datum.close",
            alt.value("#06982d"),
            alt.value("#ae1325")
    )

    base = alt.Chart(daily_df).encode(
        alt.X('date:T',
          axis=alt.Axis(
              format='%m/%Y',
              labelAngle=-45
          )
        ),
        color=open_close_color
    )

    rule = base.mark_rule().encode(
        alt.Y(
            'low:Q',
            title='Price',
            scale=alt.Scale(zero=False),
        ),
        alt.Y2('high:Q')
    )

    bar = base.mark_bar().encode(
        alt.Y('open:Q'),
        alt.Y2('close:Q')
    )

    candlestick = (rule + bar).properties(
        width=1150,
        height=400
    )
    # st.altair_chart(candlestick)
    return candlestick

### End of Function definition

cp_df = get_company_profile(use_cache=False)
div_df = get_historical_dividend(use_cache=True)

final_df = compute_div_feature(cp_df, div_df)

full_table_section = st.container(border=True)
with full_table_section:
    st.markdown(calc_statistics())
    final_df['Emiten'] = [x[:-3] for x in final_df.index]
    filtered_df = final_df[(final_df['avgPctAnnualDivIncrease'] < 100)
                            & (final_df['numDividendYear'] > 0)
                            & (final_df['lastDiv'] > 0)
                            & (final_df['avgPctAnnualDivIncrease'] > 0)].reset_index().set_index('symbol').sort_values('DScore', ascending=False)

    tabs = st.tabs(['Table View', 'Scatter View'])
    with tabs[0]:
        event = st.dataframe(filtered_df, selection_mode=['single-row'], on_select='rerun')
    
    with tabs[1]:
        x_axis = st.selectbox('Select X Axis', ['yield'])
        y_axis = st.selectbox('Select Y Axis', ['avgPctAnnualDivIncrease'])
        point_selection = alt.selection(type='point', name='point')

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

    st.write(sp_event)
