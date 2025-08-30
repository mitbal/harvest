import os
import json
import time

import logging
from pythonjsonlogger.jsonlogger import JsonFormatter

import redis
import numpy as np
import altair as alt
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_echarts5 import st_echarts
from datetime import date, datetime, timedelta

import harvest.plot as hp
import harvest.data as hd
from harvest.utils import setup_logging


try:
    st.set_page_config(layout='wide')
except Exception as e:
    print('Set Page config has been called before')

st.title('Jajan Saham')

api_key = os.environ['FMP_API_KEY']
redis_url = os.environ['REDIS_URL']


### Start of Function definition


@st.cache_resource
def get_logger(name, level=logging.INFO):

    logger = setup_logging(name, level)
    logger.info('a new user is opening the stock picker page')
    return logger


@st.cache_resource
def connect_redis(redis_url):
    r = redis.from_url(redis_url)
    return r


@st.cache_data(ttl=60*60*24, show_spinner='Downloading dividend data')
def get_div_score_table(key='jkse_div_score', show_spinner='Downloading dividend table...'):

    # try from cache from redis first
    start = time.time()

    r = connect_redis(redis_url)
    rjson = r.get(key)

    end = time.time()
    logger.info(f'redis get {key} took {end-start:.04f} seconds')

    if rjson is not None:
        div_score_json = json.loads(rjson)
        last_updated = div_score_json['date']
        logger.info(f'dividend table last updated: {last_updated}')
        final_df = pd.DataFrame(json.loads(div_score_json['content']))
    else:
        final_df = pd.read_csv('dividend_historical.csv')

    final_df.rename(columns={'symbol': 'stock'}, inplace=True)
    return final_df.set_index('stock')


@st.cache_data(ttl=60*60, show_spinner=False)
def get_specific_stock_detail(stock_name):

    try:

        start_time = time.time()
        progress_bar = st.progress(0, text='Downloading stock data... Please wait')

        n_share = hd.get_shares_outstanding(stock_name)['outstandingShares'].tolist()[0]

        fin = hd.get_financial_data(stock_name)
        progress_bar.progress(20, text='Downloading historical financial data... Progresss 20%')

        cp_df = hd.get_company_profile([stock_name])
        progress_bar.progress(40, text='Downloading stock data... Progresss 40%')

        start_date = (datetime.today() - timedelta(days=365*2)).isoformat()
        price_df = hd.get_daily_stock_price(stock_name, start_from=start_date)
        progress_bar.progress(60, text='Downloading historical price data... Progresss 60%')
        
        if sl == 'JKSE':
            sdf = hd.get_dividend_history_single_stock(stock_name, source='dag')
        else:
            sdf = hd.get_dividend_history_single_stock(stock_name, source='fmp')
        progress_bar.progress(80, text='Downloading historical dividend data... Progresss 80%')

        sector_df, industry_df = hd.get_sector_industry_pe((date.today()-timedelta(days=2)).isoformat(), api_key)
        progress_bar.progress(100, text='Progress 100% complete')

        end_time = time.time()
        logger.info(f'Total download time for {stock_name}: {end_time-start_time:.04f}')

        time.sleep(0.2)
        progress_bar.empty()

        return fin, cp_df, price_df, sdf, sector_df, industry_df, n_share

    except:
        logger.error(f'Error in downloading data for {stock_name}')
        st.error(f'Cannot find the stock {stock_name}. Please check the stock name again and dont forget to add .JK for Indonesian stocks', icon="🚨")
        progress_bar.empty()
        st.stop()


### End of Function definition

sl = st.sidebar.segmented_control(label='Stock List', 
                         options=['JKSE', 'S&P500'],
                         selection_mode='single',
                         default='JKSE')

if sl is None:
    st.stop()

if sl == 'JKSE':
    key = 'div_score_jkse'
    mcap_value = 300
    currency = 'IDR'
else:
    key = 'div_score_sp500'
    mcap_value = 100
    currency = 'USD'

minimum_market_cap = st.sidebar.number_input(f'Minimum Market Capitalization (in Billion {currency})', value=mcap_value, min_value=100, max_value=1000_1000)
minimum_year = st.sidebar.number_input('Minimum Number of Year Dividend Paid', value=1, min_value=0, max_value=25)

logger = get_logger('screener')

final_df = get_div_score_table(key)

if sl == 'JKSE':
    is_syariah = st.sidebar.toggle('Syariah Only?')
    if is_syariah:
        final_df = final_df[final_df['is_syariah'] == True]

full_table_section = st.container(border=True)
with full_table_section:

    filtered_df = final_df[(final_df['mktCap'] >= minimum_market_cap*1000_000_000)
                            & (final_df['numDividendYear'] > minimum_year)
                            & (final_df['lastDiv'] > 0)].sort_values('DScore', ascending=False)

    view = st.segmented_control(label='View Option', 
                         options=['Table', 'Treemap', 'Scatter Plot', 'Distribution'],
                         selection_mode='single',
                         default='Table')

    if view == 'Table':

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
                format='localized',
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
                help='Market Capitalization on the current price',
                format='localized'
            ),
            'ipoDate': st.column_config.DateColumn(
                'IPO Date',
                help='IPO Date',
            ),
            'revenueGrowth': st.column_config.NumberColumn(
                'Revenue Growth',
                help='Average revenue growth in the last 5 years',
                format='%.02f',
            ),
            'netIncomeGrowth': st.column_config.NumberColumn(
                'Income Growth',
                help='Average net income growth in the last 5 years',
                format='%.02f',
            ),
            'medianProfitMargin': st.column_config.NumberColumn(
                'Profit Margin',
                help='Median profit margin in the last 5 years',
                format='%.02f',
            ),
            'avgFlatAnnualDivIncrease': st.column_config.NumberColumn(
                'Dividend Growth',
                help='Average annual dividend increase in the last 5 years',
                format='%.02f',
            ),
            'numOfYear': st.column_config.NumberColumn(
                'Num of Year',
                help='Number of years the stock has been listed',
            ),
            'numDividendYear': st.column_config.NumberColumn(
                'Num of Dividend Year',
                help='Number of years the stock has paid dividend',
            ),
            'positiveYear': st.column_config.NumberColumn(
                'Positive Year',
                help='Number of years the stock has increased dividend',
            ),
            'score': st.column_config.NumberColumn(
                'Score',
                help='Score of the stock',
                format='%.02f',
            ),
            'DScore': st.column_config.NumberColumn(
                'Dividend Score',
                help='Dividend Score of the stock',
                format='%.02f',
            ),
            'lastDiv': st.column_config.NumberColumn(
                'Last Dividend',
                help='Last Dividend Paid in Last/Current Fiscal Year',
                format='%.02f',
            ),
        }

        event = st.dataframe(filtered_df, selection_mode=['single-row'], on_select='rerun', column_config=cfig)

    elif view == 'Treemap':
        
        treemap_cols = st.columns([1,1,3])
        size_var = treemap_cols[0].selectbox(options=['Market Cap', 'Dividend Yield'], label='Select Size Variable')
        color_var = treemap_cols[1].selectbox(options=['None', 'Dividend Yield', 'Profit Margin', 'Revenue Growth'], label='Select Color Variable', index=1)

        df_tree = filtered_df[['sector', 'industry']]
        df_tree['Market Cap'] = filtered_df['mktCap'] / 1_000_000_000
        df_tree['Dividend Yield'] = filtered_df['yield']
        df_tree['Profit Margin'] = filtered_df['medianProfitMargin']
        df_tree['Revenue Growth'] = filtered_df['revenueGrowth']
        df_tree = df_tree.dropna()

        if color_var == 'None':
            color_var = None
            show_gradient = False
        else:
            show_gradient = True

        tree_data = hd.prep_treemap(df_tree, size_var=size_var, color_var=color_var)
        option = hp.plot_treemap(tree_data, title=f'Biggest stock in each sector based on {size_var}', show_gradient=show_gradient)
        st_echarts(option, height='600px', width='1200px')
    
    elif view == 'Scatter Plot':

        sp = alt.Chart(filtered_df.reset_index()).mark_circle().encode(
            y='yield',
            x='revenueGrowth',
            color='sector',
            tooltip=[
                'stock', 'yield', 'revenueGrowth'
            ]
        ).interactive()
        st.altair_chart(sp)

    elif view == 'Distribution':

        dist = alt.Chart(filtered_df).mark_bar(
            opacity=0.3,
            binSpacing=0
        ).encode(
            x=alt.X('yield').bin(maxbins=100),
            y=alt.Y('count()').stack(None),
        )
        st.altair_chart(dist)


if view == 'Table' and len(event.selection['rows']) > 0:
    row_idx = event.selection['rows'][0]
    stock = filtered_df.iloc[row_idx]
    stock_name = stock.name
elif 'stock' in st.query_params:
    stock_name = st.query_params['stock']
else:
    stock_name = None

select_stock = st.text_input(
    label='Click the checkbox on the leftside of the table above or type the name of the stock to get detailed information',
    value=stock_name
)

if select_stock:
    stock_name = select_stock.upper()
    if sl == 'JKSE' and '.JK' not in stock_name:
        stock_name += '.JK'
else:
    st.stop()

fin, cp_df, price_df, sdf, sector_df, industry_df, n_share = get_specific_stock_detail(stock_name)


with st.expander('Company Profile', expanded=False):    
    st.write(cp_df.loc[stock_name, 'description'])
    currency = cp_df.loc[stock_name, 'currency']


with st.expander(f'Dividend History: {stock_name}', expanded=True):
    
    if sdf is not None:

        dividend_history_cols = st.columns([3, 10, 4])
        dividend_history_cols[0].dataframe(
            sdf[['date', 'adjDividend']],
            column_config={
                'date': st.column_config.DateColumn(
                    'Ex-Date',
                ),
                'adjDividend': st.column_config.NumberColumn(
                    'Dividend',
                    help='Dividend paid per share',
                    format='%.01f',
                ),
            },
            hide_index=True)

        try:
            last_val = final_df.loc[stock_name, 'lastDiv']
            inc_val = final_df.loc[stock_name, 'avgFlatAnnualDivIncrease']
            extrapolate = True
        except:
            logger.error(f'Stock {stock_name} not found on table')
            last_val = 0
            inc_val = 0
            extrapolate = False

        yearly_dividend_chart = hp.plot_dividend_history(sdf, 
                                                        extrapolote=True, 
                                                        n_future_years=5, 
                                                        last_val=last_val, 
                                                        inc_val=inc_val)
        dividend_history_cols[1].altair_chart(yearly_dividend_chart, use_container_width=True)

        with dividend_history_cols[2]:
            if stock_name in filtered_df.index:
                last_div = filtered_df.loc[stock_name, 'lastDiv']
                inc_val = filtered_df.loc[stock_name, 'avgFlatAnnualDivIncrease']
                curr_price = filtered_df.loc[stock_name, 'price']
                next_div = last_div + inc_val
                next_yield = next_div / curr_price * 100

                stats = hd.calc_div_stats(hd.preprocess_div(sdf))

                dividend_markdown = f'''
                Estimated next year dividend payment: **:green[{next_div:0.2f} IDR]**\n
                Yield on current price: **:green[{next_yield:0.2f}%]**

                Number of years paying dividend: **{stats['num_dividend_year']:,}**

                Number of years increasing dividend: **{stats['num_positive_year']:,}**

                Positive consistency rate: **:green[{stats['num_positive_year']/stats['num_dividend_year']*100:.2f}%]**
                '''
                st.markdown(dividend_markdown)
    else:
        st.write('No dividend history available')


with st.expander(f'Financial Information: {stock_name}', expanded=True):
    fin_cols = st.columns([0.3, 0.4, 0.3])
    period = fin_cols[0].radio('Select Period', ['quarter', 'annual'], horizontal=True)
    
    if period == 'quarter':
        metric = fin_cols[1].radio('Select Metrics', ['revenue', 'netIncome'], horizontal=True)
        
        fin_chart = hp.plot_financial(fin, period=period, metric=metric, currency=currency)
        with st.container(height=500):
            st.altair_chart(fin_chart, use_container_width=False)

    else:
        annual_cols = st.columns([40,40,20])
        
        annual_cols[0].write('Annual Revenue Chart')
        revenue_chart = hp.plot_financial(fin, period=period, metric='revenue', currency=currency)
        annual_cols[0].altair_chart(revenue_chart, use_container_width=True)
        
        annual_cols[1].write('Annual Net Income Chart')
        income_chart = hp.plot_financial(fin, period=period, metric='netIncome', currency=currency)
        annual_cols[1].altair_chart(income_chart, use_container_width=True)

        with annual_cols[2]:
            if stock_name in filtered_df.index:
                st.write('**Financial Metrics Summary**')
                st.write(f'Average Revenue Growth: {filtered_df.loc[stock_name, "revenueGrowth"]:.2f}%')
                st.write(f'Average Net Income Growth: {filtered_df.loc[stock_name, "netIncomeGrowth"]:.2f}%')
                st.write(f'Median Net Profit Margin: {filtered_df.loc[stock_name, "medianProfitMargin"]:.2f}%')


with st.expander('Price Movement', expanded=True):
    candlestick_chart = hp.plot_candlestick(price_df, width=1000, height=300)
    st.altair_chart(candlestick_chart, use_container_width=True)


with st.expander(f'Valuation Analysis: {stock_name}', expanded=True):
    cols = st.columns(3, gap='large')
    year = cols[0].slider('Select Number of Year', min_value=1, max_value=5)

    val_cols = st.columns(3, gap='large')

    if stock_name in filtered_df.index:
        sector_name = filtered_df.loc[stock_name, 'sector']
        industry_name = filtered_df.loc[stock_name, 'industry']

        try:
            sector_pe = float(sector_df[sector_df['sector'] == sector_name]['pe'].to_list()[0])
            industry_pe = float(industry_df[industry_df['industry'] == industry_name]['pe'].to_list()[0])
        except Exception:
            sector_pe = industry_pe = -1
            logger.error(f'sector or industry not found for {stock_name} in sector {sector_name} and industry {industry_name}')
    else:
        sector_pe = industry_pe = -1

    start_date = datetime.now() - timedelta(days=365*year)
    last_year_df = price_df[price_df['date']>= str(start_date)]

    pe_df = hd.calc_pe_history(last_year_df, fin, n_shares=n_share, currency=currency)
    pe_ttm = pe_df['pe'].values[-1]
    current_price = price_df['close'].values[0]
    median_pe = pe_df['pe'].median()
    pe_dist_chart = hp.plot_pe_distribution(pe_df, pe_ttm)
    val_cols[0].altair_chart(pe_dist_chart, use_container_width=True)

    pe_ts_chart = hp.plot_pe_timeseries(pe_df)
    val_cols[1].altair_chart(pe_ts_chart, use_container_width=True)

    with val_cols[2]:
        ci = pe_df['pe'].quantile([.05, .95]).values
        markdown_table = f'''
        | Metric | Value |
        | ------ | ----- |
        | Current PE | {pe_ttm:.2f} |
        | Current Price | {int(current_price):,} |
        | Median last {year} year PE | {median_pe:.2f} |
        | Fair Price | {int((median_pe/pe_ttm)*current_price):,} |
        | 95% Confidence Interval range PE | {ci[0]:.2f} - {ci[1]:.2f} |
        | 95% Confidence Interval range Price | {int((ci[0]/pe_ttm)*current_price):,} - {int((ci[1]/pe_ttm)*current_price):,} |
        '''
        if industry_pe != -1 and sector_pe != -1:
            markdown_table += f"| Industry: {industry_name} PE | {industry_pe:.2f} | \n \
        | Sector: {sector_name} PE | {sector_pe:.2f} |"
        
        st.markdown(markdown_table)

    diff = median_pe / pe_ttm
    fair_threshold = pe_df['pe'].quantile([.45, .55]).values
    if pe_ttm >= fair_threshold[0] and pe_ttm <= fair_threshold[1]:
        assessment = '**Fair Valued**'
    elif pe_ttm < fair_threshold[0]:
        assessment = f'**:green[Undervalued]**. Potential Upside: **:green[{(diff-1)*100:.2f}% - {abs((ci[1]/pe_ttm-1)*100):.2f}%]**'
    else:
        assessment = f'**:red[Overvalued]**. Potential Downside: **:red[{(1-diff)*100:.2f}% - {abs((ci[0]/pe_ttm-1)*100):.2f}%]**'
    st.write(f'Assessment: {assessment}')


with st.expander(f'Compounding Simulation: {stock_name}'):

    this_year = datetime.now().year
    
    cols = st.columns(2)
    start_year = cols[0].number_input(label='Start Year', value=2021, min_value=2010, max_value=this_year-2)
    end_year = cols[1].number_input(label='End Year', value=this_year-1, min_value=start_year+1, max_value=this_year-1)

    cols = st.columns(2)
    initial_value = cols[0].number_input(label='Initial investment (in million)', value=10, min_value=1, max_value=1000)
    monthly_topup = cols[1].number_input(label='Monthly Topup (in million)', value=1, min_value=0, max_value=100)

    porto, activities = hd.simulate_dividend_compounding(stock_name, 
                                                price_df,
                                                sdf,
                                                start_year,
                                                end_year,
                                                initial_value * 1_000_000,
                                                monthly_topup * 1_000_000)
    
    st.write('Activities:')
    st.write(activities)
    st.write('Portfolio:')

    porto_df = pd.DataFrame(porto)
    porto_df['total'] = porto_df['num_stock'] * porto_df['price'] * 100
    porto_df['cum_stock'] = porto_df['num_stock'].cumsum()
    porto_df['cum_total'] = porto_df['total'].cumsum()
    porto_df['avg_price'] = porto_df['cum_total'] / porto_df['cum_stock'] / 100

    porto_df['current_value'] = porto_df['price'] * porto_df['cum_stock'] * 100

    cols = st.columns(2)
    cols[0].write(porto_df[['date', 'cum_total', 'current_value']])

    return_chart1 = alt.Chart(porto_df).mark_line().encode(
        x=alt.X('date', axis=alt.Axis(labels=False)),
        y=alt.Y('cum_total').scale(zero=False)
    )
    return_chart2 = alt.Chart(porto_df).mark_line().encode(
        x=alt.X('date', axis=alt.Axis(labels=False)),
        y=alt.Y('current_value').scale(zero=False),
        color=alt.value('green')
    )
    cols[1].altair_chart(return_chart1 + return_chart2)
