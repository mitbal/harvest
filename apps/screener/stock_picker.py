import os
import json
import time

import redis
import pandas as pd
import streamlit as st
from datetime import date, datetime, timedelta

import harvest.plot as hp
import harvest.data as hd

try:
    st.set_page_config(layout='wide')
except Exception as e:
    print('Set Page config has been called before')

st.title('Jajan Saham')

api_key = os.environ['FMP_API_KEY']
redis_url = os.environ['REDIS_URL']

@st.cache_resource
def connect_redis(redis_url):
    r = redis.from_url(redis_url)
    return r


@st.cache_data(ttl=60*60*24)
def get_div_score_table(key='jkse_div_score', show_spinner='Downloading dividend table...'):

    # try from cache from redis first
    r = connect_redis(redis_url)
    rjson = r.get(key)
    if rjson is not None:
        div_score_json = json.loads(rjson)
        last_updated = div_score_json['date']
        print('data last updated', last_updated)
        final_df = pd.DataFrame(json.loads(div_score_json['content']))
    else:
        final_df = pd.read_csv('dividend_historical.csv')

    final_df.rename(columns={'symbol': 'stock'}, inplace=True)
    return final_df.set_index('stock')


@st.cache_data(ttl=60*60, show_spinner=False)
def get_specific_stock_detail(stock_name):

    start_time = time.time()

    progress_bar = st.progress(0, text='Downloading stock data... Please wait')

    n_share = hd.get_shares_outstanding(stock_name)['outstandingShares'].tolist()[0]

    fin = hd.get_financial_data(stock_name)
    progress_bar.progress(20, text='Downloading stock data... Progresss 20%')

    cp_df = hd.get_company_profile([stock_name])
    progress_bar.progress(40, text='Downloading stock data... Progresss 40%')

    start_date = (datetime.today() - timedelta(days=365*2)).isoformat()
    price_df = hd.get_daily_stock_price(stock_name, start_from=start_date)
    progress_bar.progress(60, text='Downloading stock data... Progresss 60%')
    
    sdf = pd.DataFrame(hd.get_dividend_history([stock_name])[stock_name])
    progress_bar.progress(80, text='Downloading stock data... Progresss 80%')

    sector_df, industry_df = hd.get_sector_industry_pe((date.today()-timedelta(days=2)).isoformat(), api_key)
    progress_bar.progress(100, text='Progress 100% complete')

    end_time = time.time()
    print(f'Total download time for {stock_name}: {end_time-start_time}')

    time.sleep(0.5)
    progress_bar.empty()

    return fin, cp_df, price_df, sdf, sector_df, industry_df, n_share


### End of Function definition

sl = st.sidebar.segmented_control(label='Stock List', 
                         options=['JKSE', 'S&P500'],
                         selection_mode='single',
                         default='JKSE')

if sl is None:
    print('Please select one of the options above')
    st.stop()

if sl == 'JKSE':
    key = 'div_score_jkse'
    mcap_value = 1000
    currency = 'IDR'
else:
    key = 'div_score_sp500'
    mcap_value = 100
    currency = 'USD'

start = time.time()
final_df = get_div_score_table(key)

end = time.time()
print(f'Elapsed time {end-start}')

minimum_market_cap = st.sidebar.number_input(f'Minimum Market Capitalization (in Billion {currency})', value=mcap_value, min_value=100, max_value=1000_1000)
minimum_year = st.sidebar.number_input('Minimum Number of Year Dividend Paid', value=1, min_value=0, max_value=25)

if sl == 'JKSE':
        is_syariah = st.sidebar.toggle('Syariah Only?')
        if is_syariah:
            final_df = final_df[final_df['is_syariah'] == True]

full_table_section = st.container(border=True)
with full_table_section:

    filtered_df = final_df[(final_df['mktCap'] >= minimum_market_cap*1000_000_000)
                            & (final_df['numDividendYear'] > minimum_year)
                            & (final_df['lastDiv'] > 0)].sort_values('DScore', ascending=False)

    tabs = st.tabs(['Table View'])
    with tabs[0]:

        cfig={
            'stock': st.column_config.TextColumn(
                'Stock',
                help='Stock Code',
            ),
            'is_syariah': st.column_config.CheckboxColumn(
                'Syariah',
                help='Is the stock Syariah?',
                default=False,
            ),
            'price': st.column_config.NumberColumn(
                'Price',
                help='Current Stock Price',
            ),
            'yield': st.column_config.NumberColumn(
                'Dividend Yield',
                help='The dividend yield on the current price (in pct)',
                min_value=0,
                max_value=100,
                step=0.01,
                format='%.02f',
            ),
            'sector': st.column_config.TextColumn(
                'Sector',
                help='The sector of the stock',
            ),
            'industry': st.column_config.TextColumn(
                'Industry',
                help='The industry of the stock',
            ),
            'mktCap': st.column_config.NumberColumn(
                'Market Cap',
                help='Market Capitalization on the current price'
            ),
            'lastDiv': st.column_config.NumberColumn(
                'Last Dividend',
                help='Last Dividend Paid in Last/Current Fiscal Year',
                format='%.02f',
            ),
        }

        event = st.dataframe(filtered_df, selection_mode=['single-row'], on_select='rerun', column_config=cfig)

if len(event.selection['rows']) > 0:
    row_idx = event.selection['rows'][0]
    stock = filtered_df.iloc[row_idx]
    stock_name = stock.name
elif len(st.query_params) > 0:
    stock_name = st.query_params['stock']
else:
    st.stop()

fin, cp_df, price_df, sdf, sector_df, industry_df, n_share = get_specific_stock_detail(stock_name)

with st.expander('Company Profile', expanded=False):    
    st.write(cp_df.loc[stock_name, 'description'])

with st.expander('Dividend History', expanded=True):
    dividend_history_cols = st.columns([3, 10, 4])
    dividend_history_cols[0].dataframe(sdf[['date', 'adjDividend']], hide_index=True)

    yearly_dividend_chart = hp.plot_dividend_history(sdf, 
                                                     extrapolote=True, 
                                                     n_future_years=5, 
                                                     last_val=final_df.loc[stock_name, 'lastDiv'], 
                                                     inc_val=final_df.loc[stock_name, 'avgFlatAnnualDivIncrease'])
    dividend_history_cols[1].altair_chart(yearly_dividend_chart, use_container_width=True)

    with dividend_history_cols[2]:
        last_div = filtered_df.loc[stock_name, 'lastDiv']
        inc_val = filtered_df.loc[stock_name, 'avgFlatAnnualDivIncrease']
        curr_price = filtered_df.loc[stock_name, 'price']
        next_div = last_div + inc_val
        next_yield = next_div / curr_price * 100
        st.write(f'Next year dividend payment: {next_div:0.2f} IDR')
        st.write(f'Yield on current price: {next_yield:0.2f}%')

with st.expander('Financial Information', expanded=True):
    fin_cols = st.columns([0.3, 0.4, 0.3])
    period = fin_cols[0].radio('Select Period', ['quarter', 'annual'], horizontal=True)
    metric = fin_cols[1].radio('Select Metrics', ['netIncome', 'eps', 'revenue'], horizontal=True)
    
    fin_chart = hp.plot_financial(fin, period=period, metric=metric, currency=currency)
    with st.container(height=500):
        st.altair_chart(fin_chart, use_container_width=False)

with st.expander('Price Movement', expanded=True):
    candlestick_chart = hp.plot_candlestick(price_df, width=1000, height=300)
    st.altair_chart(candlestick_chart, use_container_width=True)

with st.expander('Valuation Analysis', expanded=True):
    val_cols = st.columns(3, gap='large')

    sector_name = filtered_df.loc[stock_name, 'sector']
    industry_name = filtered_df.loc[stock_name, 'industry']

    try:
        sector_pe = float(sector_df[sector_df['sector'] == sector_name]['pe'].to_list()[0])
        industry_pe = float(industry_df[industry_df['industry'] == industry_name]['pe'].to_list()[0])
    except Exception:
        sector_pe = industry_pe = -1
        print('sector or industry not found')

    pe_df = hd.calc_pe_history(price_df, fin, n_shares=n_share, currency=currency)
    pe_ttm = pe_df['pe'].values[-1]
    pe_dist_chart = hp.plot_pe_distribution(pe_df, pe_ttm)
    val_cols[0].altair_chart(pe_dist_chart, use_container_width=True)

    pe_ts_chart = hp.plot_pe_timeseries(pe_df)
    val_cols[1].altair_chart(pe_ts_chart, use_container_width=True)

    with val_cols[2]:
        ci = pe_df['pe'].quantile([.05, .95]).values
        st.markdown(f'''
            Current PE: {pe_ttm:.2f}\n
            Mean PE (last 5 year): {pe_df['pe'].mean():.2f}\n
            95% Confidence Interval range: {ci[0]:.2f} - {ci[1]:.2f}
        ''')
        if industry_pe != -1:
            st.write(f'Industry {industry_name} PE: {industry_pe:.2f}')
        if sector_pe != -1:
            st.write(f'Sector {sector_name} PE: {sector_pe:.2f}')
