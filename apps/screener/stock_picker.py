import os
import json
import time

import redis
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from datetime import date, datetime, timedelta

import harvest.plot as hp
import harvest.data as hd

st.title('Jajan Saham')

api_key = os.environ['FMP_API_KEY']
redis_url = os.environ['REDIS_URL']

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

        if df.loc[symbol, 'lastDiv'] == 0:
            continue
        
        try:
            div['year'] = [x.year for x in pd.to_datetime(div['date'])]
        except Exception as e:
            print('error', symbol)
            continue

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


@st.cache_resource
def connect_redis(redis_url):
    r = redis.from_url(redis_url)
    return r


@st.cache_data(ttl=60*60*24)
def get_div_score_table(key='jkse_div_score'):

    # try from cache from redis first
    r = connect_redis(redis_url)
    rjson = r.get(key)
    if rjson is not None:
        final_df = pd.DataFrame(json.loads(rjson))
    else:    
        # if not found in cache, compute from scratch
        print('redis error, computing dividend score from scratch')
        cp_df = get_company_profile(use_cache=False)
        div_df = get_historical_dividend(use_cache=True)
        final_df = compute_div_feature(cp_df, div_df)

    final_df.rename(columns={'symbol': 'stock'}, inplace=True)
    return final_df.set_index('stock')


@st.cache_data
def get_specific_stock_detail(stock_name):

    n_share = hd.get_shares_outstanding(stock_name)['outstandingShares'].tolist()[0]

    fin = hd.get_financial_data(stock_name)
    cp_df = hd.get_company_profile([stock_name])
    start_date = (datetime.today() - timedelta(days=365*2)).isoformat()
    price_df = hd.get_daily_stock_price(stock_name, start_from=start_date)
    sdf = pd.DataFrame(hd.get_dividend_history([stock.name])[stock.name])
    sector_df, industry_df = hd.get_sector_industry_pe((date.today()-timedelta(days=2)).isoformat(), api_key)

    return fin, cp_df, price_df, sdf, sector_df, industry_df, n_share


### End of Function definition

start = time.time()

final_df = get_div_score_table('jkse_div_score')
# final_df = get_div_score_table('sp500_div_score')


end = time.time()
print(f'Elapsed time {end-start}')

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
                            & (final_df['lastDiv'] > 0)].sort_values('DScore', ascending=False)

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

        attributes = ['yield', 'avgFlatAnnualDivIncrease', 'mktCap', 'DScore']

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

fin, cp_df, price_df, sdf, sector_df, industry_df, n_share = get_specific_stock_detail(stock_name)

with st.expander('Company Profile', expanded=False):    
    st.write(cp_df.loc[stock_name, 'description'])

with st.expander('Dividend History', expanded=False):
    dividend_history_cols = st.columns([3, 10, 4])
    dividend_history_cols[0].dataframe(sdf[['date', 'adjDividend']], hide_index=True)

    yearly_dividend_chart = hp.plot_dividend_history(sdf, 
                                                     extrapolote=True, 
                                                     n_future_years=5, 
                                                     last_val=final_df.loc[stock_name, 'lastDiv'], 
                                                     inc_val=final_df.loc[stock_name, 'avgFlatAnnualDivIncrease'])
    dividend_history_cols[1].altair_chart(yearly_dividend_chart, use_container_width=True)

    with dividend_history_cols[2]:
        last_div = filtered_df.loc[stock.name, 'lastDiv']
        inc_val = filtered_df.loc[stock_name, 'avgFlatAnnualDivIncrease']
        curr_price = filtered_df.loc[stock.name, 'price']
        next_div = last_div + inc_val
        next_yield = next_div / curr_price * 100
        st.write(f'Next year dividend payment: {next_div:0.2f} IDR')
        st.write(f'Yield on current price: {next_yield:0.2f}%')

with st.expander('Financial Information', expanded=False):
    fin_cols = st.columns(4)
    period = fin_cols[0].radio('Select Period', ['quarter', 'annual'], horizontal=True)
    metric = fin_cols[1].radio('Select Metrics', ['netIncome', 'eps', 'revenue'], horizontal=True)
    
    fin_chart = hp.plot_financial(fin, period=period, metric=metric)
    st.altair_chart(fin_chart)

with st.expander('Price Movement', expanded=False):
    candlestick_chart = hp.plot_candlestick(price_df, width=1000, height=300)
    st.altair_chart(candlestick_chart.interactive())

with st.expander('Valuation Analysis', expanded=False):
    val_cols = st.columns(3, gap='large')

    sector_name = filtered_df.loc[stock_name, 'sector']
    industry_name = filtered_df.loc[stock_name, 'industry']

    try:
        sector_pe = float(sector_df[sector_df['sector'] == sector_name]['pe'].to_list()[0])
        industry_pe = float(industry_df[industry_df['industry'] == industry_name]['pe'].to_list()[0])
    except Exception:
        sector_pe = industry_pe = -1
        print('sector or industry not found')

    pe_df = hd.calc_pe_history(price_df, fin, n_shares=n_share)
    pe_ttm = pe_df['pe'].values[-1]
    pe_dist_chart = hp.plot_pe_distribution(pe_df, pe_ttm)
    val_cols[0].altair_chart(pe_dist_chart)

    pe_ts_chart = hp.plot_pe_timeseries(pe_df)
    val_cols[1].altair_chart(pe_ts_chart)

    with val_cols[2]:
        ci = pe_df['pe'].quantile([.05, .95]).values
        st.markdown(f'''
            Current PE: {pe_ttm:.2f}\n
            Mean PE (last 3 year): {pe_df['pe'].mean():.2f}\n
            95% Confidence Interval range: {ci[0]:.2f} - {ci[1]:.2f}
        ''')
        if industry_pe != -1:
            st.write(f'Industry {industry_name} PE: {industry_pe:.2f}')
        if sector_pe != -1:
            st.write(f'Sector {sector_name} PE: {sector_pe:.2f}')
